// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// The Z/2^k ladder (slot-22 BIN-2): native kernels, rung selection, the wide dispatch and the
// adaptive rung.  See Ring2k.h and ENGINE_API.md Sec. 8.2.  This file is compiled with
// -ftree-vectorize (cmake/slot22/BIN2.cmake): the kernel bodies below are plain loops that the
// compiler vectorises once per ISA (portable / AVX2 / AVX-512 under target attributes).

#include "Ring2k.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <vector>

#include "BitMatrix.h"
#include "../ModArithmetic/Kernels.h"   // (S24 CL-5) kernels::isa_auto, the ENGINE_KERNEL ceiling

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#  define R2K_HAVE_X86 1
#  define R2K_AVX2_FN   __attribute__((target("avx2")))
#  define R2K_AVX512_FN __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl")))
#else
#  define R2K_HAVE_X86 0
#endif

#if defined(__GNUC__) || defined(__clang__)
#  define R2K_INLINE inline __attribute__((always_inline))
#else
#  define R2K_INLINE inline
#endif

namespace engine {
namespace ring2k {

namespace {

// ---------------------------------------------------------------------------
// Kernel bodies (wraparound; narrow types computed in `unsigned`).
// ---------------------------------------------------------------------------
template<class T> using Wd = typename std::conditional<(sizeof(T) < sizeof(unsigned)), unsigned, T>::type;

template<class T>
R2K_INLINE void axpy_body(T* __restrict r, const T* __restrict p, T f, size_t n) {
    const Wd<T> ff = f;
    for (size_t j = 0; j < n; ++j) r[j] = T(Wd<T>(r[j]) - ff * Wd<T>(p[j]));
}

// The register-blocked micro-kernel: a 4 x (2 vectors) block of C accumulates sum_t L[i][t] U[t][j]
// in eight vector registers over the whole inner dimension, then is subtracted once.  GCC / Clang
// vector extensions (unsigned lanes: every operation wraps mod 2^W, exactly as the scalar code); the
// enclosing function's target attribute picks the instructions.  Unaligned loads / stores by memcpy.
template<class T, unsigned VB>
R2K_INLINE void micro4(T* C, size_t ldc, const T* L, size_t ldl, const T* U, size_t ldu, uint32_t np) {
    typedef T V __attribute__((vector_size(VB)));
    constexpr unsigned NV = VB / sizeof(T);
    V a00 = {}, a01 = {}, a10 = {}, a11 = {}, a20 = {}, a21 = {}, a30 = {}, a31 = {};
    for (uint32_t t = 0; t < np; ++t) {
        V u0, u1;
        std::memcpy(&u0, U + t * ldu, VB);
        std::memcpy(&u1, U + t * ldu + NV, VB);
        const V f0 = V{} + L[t], f1 = V{} + L[ldl + t], f2 = V{} + L[2 * ldl + t], f3 = V{} + L[3 * ldl + t];
        a00 += f0 * u0; a01 += f0 * u1;
        a10 += f1 * u0; a11 += f1 * u1;
        a20 += f2 * u0; a21 += f2 * u1;
        a30 += f3 * u0; a31 += f3 * u1;
    }
    const V* acc[4][2] = {{&a00, &a01}, {&a10, &a11}, {&a20, &a21}, {&a30, &a31}};
    for (unsigned r = 0; r < 4; ++r)
        for (unsigned h = 0; h < 2; ++h) {
            V c;
            T* p = C + r * ldc + h * NV;
            std::memcpy(&c, p, VB);
            c -= *acc[r][h];
            std::memcpy(p, &c, VB);
        }
}

// C (mrows x ncols) -= L (mrows x np) * U (np x ncols).  Column tiles keep the U tile in L2; inside a
// tile, 4-row blocks run the micro-kernel on JB-column blocks; the row / column tails take AXPYs.
template<class T, unsigned VB>
R2K_INLINE void gemm_body(T* C, size_t ldc, uint32_t mrows, const T* L, size_t ldl, const T* U, size_t ldu,
                          uint32_t np, uint32_t ncols) {
    if (mrows == 0 || np == 0 || ncols == 0) return;
    constexpr unsigned JB = 2 * VB / sizeof(T);
    size_t jt = (size_t(1) << 18) / (sizeof(T) * np);
    jt = jt < JB ? JB : jt - jt % JB;
    for (size_t j0 = 0; j0 < ncols; j0 += jt) {
        const size_t jn = j0 + jt < ncols ? jt : ncols - j0;
        const size_t jfull = jn - jn % JB;
        uint32_t i = 0;
        for (; i + 4 <= mrows; i += 4) {
            const T* l = L + i * ldl;
            T* c = C + i * ldc + j0;
            for (size_t j = 0; j < jfull; j += JB) micro4<T, VB>(c + j, ldc, l, ldl, U + j0 + j, ldu, np);
            if (jfull < jn)
                for (uint32_t r = 0; r < 4; ++r)
                    for (uint32_t t = 0; t < np; ++t)
                        if (l[r * ldl + t]) axpy_body<T>(c + r * ldc + jfull, U + t * ldu + j0 + jfull, l[r * ldl + t], jn - jfull);
        }
        for (; i < mrows; ++i) {
            T* c = C + i * ldc + j0;
            const T* l = L + i * ldl;
            for (uint32_t t = 0; t < np; ++t)
                if (l[t]) axpy_body<T>(c, U + t * ldu + j0, l[t], jn);
        }
    }
}

// (slot-23 BIN3O1) The Taylor-shift sweep: row j -= f row (j + 1), j = n - 1 down to i, for i = 0 .. n - 1 (pass i).
// Each pass acts on every column separately, so the sweep runs per chunk of NV columns (one vector), and KP passes at a
// time in one bottom-up walk over the rows (a register wavefront): b[t] holds the chunk of row (j + 1) right after pass
// i0 + t, so at row j pass i0 + t is x -= f b[t], b[t] = x.  Each pass still reads row j + 1 after its own update and
// row j after the earlier passes: the same operations on the same operands as the pass-by-pass loop, hence the same
// wraparound values.  Pass i0 + t touches the rows j >= i0 + t only; the bottom row n is never written.
template<class T, unsigned VB, int MODE>   // MODE 0: x += v (f = -1); 1: x -= v (f = +1); 2: x -= f v
R2K_INLINE void shift_sweep(T* X, size_t ld, uint32_t n, size_t w, T f) {
    typedef T V __attribute__((vector_size(VB)));
    constexpr unsigned NV = VB / sizeof(T);
    constexpr uint32_t KP = 16;
    const V fv = V{} + f;
    size_t c = 0;
    for (; c + NV <= w; c += NV) {
        for (uint32_t i0 = 0; i0 < n; i0 += KP) {
            const uint32_t kk = n - i0 < KP ? n - i0 : KP;
            V bv[KP];
            V bot;
            std::memcpy(&bot, X + static_cast<size_t>(n) * ld + c, VB);
            for (uint32_t t = 0; t < KP; ++t) bv[t] = bot;
            uint32_t j = n;
            if (kk == KP) {
                for (; j-- > i0 + KP - 1;) {   // these rows take all KP passes
                    T* xr = X + static_cast<size_t>(j) * ld + c;
                    V x;
                    std::memcpy(&x, xr, VB);
#pragma GCC unroll 16
                    for (uint32_t t = 0; t < KP; ++t) {
                        if (MODE == 0) x += bv[t];
                        else if (MODE == 1) x -= bv[t];
                        else x -= fv * bv[t];
                        bv[t] = x;
                    }
                    std::memcpy(xr, &x, VB);
                }
                ++j;
            }
            for (; j-- > i0;) {   // the bottom rows of the block: passes i0 .. j only
                const uint32_t tm = j - i0 + 1 < kk ? j - i0 + 1 : kk;
                T* xr = X + static_cast<size_t>(j) * ld + c;
                V x;
                std::memcpy(&x, xr, VB);
                for (uint32_t t = 0; t < tm; ++t) {
                    if (MODE == 0) x += bv[t];
                    else if (MODE == 1) x -= bv[t];
                    else x -= fv * bv[t];
                    bv[t] = x;
                }
                std::memcpy(xr, &x, VB);
            }
        }
    }
    const Wd<T> ff = f;
    for (; c < w; ++c)   // the column tail, pass by pass
        for (uint32_t i = 0; i < n; ++i)
            for (uint32_t j = n; j-- > i;) {
                T& x = X[static_cast<size_t>(j) * ld + c];
                const T y = X[static_cast<size_t>(j + 1) * ld + c];
                if (MODE == 0) x = T(Wd<T>(x) + Wd<T>(y));
                else if (MODE == 1) x = T(Wd<T>(x) - Wd<T>(y));
                else x = T(Wd<T>(x) - ff * Wd<T>(y));
            }
}
template<class T, unsigned VB>
R2K_INLINE void shift_body(T* X, size_t ld, uint32_t n, size_t w, T f) {
    if (n == 0 || w == 0) return;
    if (f == T(~T(0))) shift_sweep<T, VB, 0>(X, ld, n, w, f);
    else if (f == T(1)) shift_sweep<T, VB, 1>(X, ld, n, w, f);
    else shift_sweep<T, VB, 2>(X, ld, n, w, f);
}

template<class T> struct KernelSet {
    void (*axpy)(T*, const T*, T, size_t);
    void (*gemm)(T*, size_t, uint32_t, const T*, size_t, const T*, size_t, uint32_t, uint32_t);
    void (*shift)(T*, size_t, uint32_t, size_t, T);
};

#define R2K_DEFINE_BODIES(SUFFIX, ATTR, VB)                                                                     \
    template<class T> ATTR void axpy_##SUFFIX(T* r, const T* p, T f, size_t n) { axpy_body<T>(r, p, f, n); }   \
    template<class T> ATTR void gemm_##SUFFIX(T* C, size_t ldc, uint32_t mrows, const T* L, size_t ldl,          \
                                              const T* U, size_t ldu, uint32_t np, uint32_t ncols) {            \
        gemm_body<T, VB>(C, ldc, mrows, L, ldl, U, ldu, np, ncols);                                              \
    }                                                                                                            \
    template<class T> ATTR void shift_##SUFFIX(T* X, size_t ld, uint32_t n, size_t w, T f) {                    \
        shift_body<T, VB>(X, ld, n, w, f);                                                                       \
    }
R2K_DEFINE_BODIES(portable, , 16)
#if R2K_HAVE_X86
R2K_DEFINE_BODIES(avx2, R2K_AVX2_FN, 32)
R2K_DEFINE_BODIES(avx512, R2K_AVX512_FN, 64)
#endif
#undef R2K_DEFINE_BODIES

// (S24 CL-5) the CPU facts from CpuFeatures.h (false on a build without the bodies)
#if R2K_HAVE_X86
bool has_avx2() { return cpu::avx2(); }
bool has_avx512() { return cpu::avx512_f_bw_dq_vl(); }
#else
bool has_avx2() { return false; }
bool has_avx512() { return false; }
#endif

std::atomic<uint8_t> g_forced{static_cast<uint8_t>(KernelPath::Auto)};

template<class T>
KernelSet<T> kernels_for(KernelPath p) {
#if R2K_HAVE_X86
    if (p == KernelPath::Avx512) return {&axpy_avx512<T>, &gemm_avx512<T>, &shift_avx512<T>};
    if (p == KernelPath::Avx2) return {&axpy_avx2<T>, &gemm_avx2<T>, &shift_avx2<T>};
#endif
    (void)p;
    return {&axpy_portable<T>, &gemm_portable<T>, &shift_portable<T>};
}

std::atomic<uint32_t> g_panel{64}, g_reduce_block{64};
std::atomic<uint64_t> g_lu_calls{0}, g_lu_pivots{0}, g_closure_rows{0}, g_smith_levels{0};

} // namespace

bool kernel_path_available(KernelPath p) {
    switch (p) {
    case KernelPath::Auto:
    case KernelPath::Portable: return true;
    case KernelPath::Avx2: return has_avx2();
    case KernelPath::Avx512: return has_avx512();
    }
    return false;
}
void set_kernel_path(KernelPath p) {
    ENGINE_CHECK_MSG(kernel_path_available(p), "ring2k::set_kernel_path: path not available on this CPU / build");
    g_forced.store(static_cast<uint8_t>(p), std::memory_order_relaxed);
}
KernelPath active_kernel_path() {
    const KernelPath f = static_cast<KernelPath>(g_forced.load(std::memory_order_relaxed));
    if (f != KernelPath::Auto) return f;
    return kernels::isa_auto(has_avx2(), has_avx512());   // (S24 CL-5) under the ENGINE_KERNEL ceiling
}
const char* kernel_path_name(KernelPath p) { return cpu::isa_path_name(p); }

#define R2K_DEFINE_NATIVE(T)                                                                                   \
    void kernel_axpy(T* r, const T* p, T f, size_t n) { kernels_for<T>(active_kernel_path()).axpy(r, p, f, n); } \
    void kernel_gemm_sub(T* C, size_t ldc, uint32_t mrows, const T* L, size_t ldl, const T* U, size_t ldu,        \
                         uint32_t np, uint32_t ncols) {                                                         \
        kernels_for<T>(active_kernel_path()).gemm(C, ldc, mrows, L, ldl, U, ldu, np, ncols);                    \
    }                                                                                                          \
    void kernel_shift_rows(T* X, size_t ld, uint32_t n, size_t w, T f) {                                       \
        kernels_for<T>(active_kernel_path()).shift(X, ld, n, w, f);                                            \
    }
R2K_DEFINE_NATIVE(uint8_t)
R2K_DEFINE_NATIVE(uint16_t)
R2K_DEFINE_NATIVE(uint32_t)
R2K_DEFINE_NATIVE(uint64_t)
#undef R2K_DEFINE_NATIVE

Params params() {
    Params p;
    p.panel = g_panel.load(std::memory_order_relaxed);
    p.reduce_block = g_reduce_block.load(std::memory_order_relaxed);
    return p;
}
void set_params(const Params& p) {
    ENGINE_CHECK_MSG(p.panel >= 1 && p.reduce_block >= 1, "ring2k::set_params: panel and reduce_block must be >= 1");
    g_panel.store(p.panel, std::memory_order_relaxed);
    g_reduce_block.store(p.reduce_block, std::memory_order_relaxed);
}
Counters counters() {
    Counters c;
    c.lu_calls = g_lu_calls.load(std::memory_order_relaxed);
    c.lu_pivots = g_lu_pivots.load(std::memory_order_relaxed);
    c.closure_rows = g_closure_rows.load(std::memory_order_relaxed);
    c.smith_levels = g_smith_levels.load(std::memory_order_relaxed);
    return c;
}
void reset_counters() {
    g_lu_calls = 0; g_lu_pivots = 0; g_closure_rows = 0; g_smith_levels = 0;
}
void add_counters(const Counters& d) {
    if (d.lu_calls) g_lu_calls.fetch_add(d.lu_calls, std::memory_order_relaxed);
    if (d.lu_pivots) g_lu_pivots.fetch_add(d.lu_pivots, std::memory_order_relaxed);
    if (d.closure_rows) g_closure_rows.fetch_add(d.closure_rows, std::memory_order_relaxed);
    if (d.smith_levels) g_smith_levels.fetch_add(d.smith_levels, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// Rungs
// ---------------------------------------------------------------------------
namespace {

template<uint32_t... Ns> struct NList {};
using R5List = NList<3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64>;

template<uint32_t... Ns>
std::vector<uint32_t> list_values(NList<Ns...>) { return {Ns...}; }

// Calls f(E{}) with E = WordN<N> for the instantiated N.
template<class F, uint32_t N0, uint32_t... Ns>
auto dispatch_words(uint32_t N, F&& f, NList<N0, Ns...>) -> decltype(f(WordN<N0>{})) {
    if (N == N0) return f(WordN<N0>{});
    if constexpr (sizeof...(Ns) > 0) {
        return dispatch_words(N, std::forward<F>(f), NList<Ns...>{});
    } else {
        ENGINE_CHECK_MSG(false, "ring2k: R5 word count not instantiated");
        return f(WordN<N0>{});
    }
}

// Calls f(E{}) with the element type of the rung.
template<class F>
auto dispatch_rung(const RungChoice& c, F&& f) -> decltype(f(uint8_t{})) {
    switch (c.rung) {
    case Rung::R1: return f(uint8_t{});
    case Rung::R2_16: return f(uint16_t{});
    case Rung::R2_32: return f(uint32_t{});
    case Rung::R3: return f(uint64_t{});
    case Rung::R4: return f(u128{});
    case Rung::R5: return dispatch_words(c.words, std::forward<F>(f), R5List{});
    case Rung::R0: break;
    }
    ENGINE_CHECK_MSG(false, "ring2k: rung R0 has no element type (k = 1 is binary::BitMatrix)");
    return f(uint8_t{});
}

binary::BitMatrix to_bits(const WideMatrix& A) {
    binary::BitMatrix B(A.rows, A.cols);
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint32_t j = 0; j < A.cols; ++j)
            if (A.entry(i, j)[0] & 1u) B.set(i, j, true);
    return B;
}
WideHowell bits_to_howell(const binary::BitMatrix& R, const std::vector<uint32_t>& pivots) {
    WideHowell h;
    h.H = WideMatrix(static_cast<uint32_t>(pivots.size()), R.cols(), 1);
    for (uint32_t i = 0; i < pivots.size(); ++i)
        for (uint32_t j = 0; j < R.cols(); ++j) h.H.entry(i, j)[0] = R.get(i, j) ? 1u : 0u;
    h.pivots = pivots;
    h.pivot_v.assign(pivots.size(), 0);
    return h;
}

} // namespace

std::string rung_label(const RungChoice& c) {
    switch (c.rung) {
    case Rung::R0: return "R0(bits)";
    case Rung::R1: return "R1(u8)";
    case Rung::R2_16: return "R2(u16)";
    case Rung::R2_32: return "R2(u32)";
    case Rung::R3: return "R3(u64)";
    case Rung::R4: return "R4(u128)";
    case Rung::R5: return "R5(N=" + std::to_string(c.words) + ")";
    }
    return "?";
}
const std::vector<uint32_t>& r5_word_counts() {
    static const std::vector<uint32_t> v = list_values(R5List{});
    return v;
}
RungChoice select_rung(uint32_t k) {
    ENGINE_CHECK_MSG(k >= 1 && k <= 4096, "ring2k::select_rung: need 1 <= k <= 4096 (R5 tops out at N = 64)");
    if (k == 1) return {Rung::R0, 1, 1};
    if (k <= 8) return {Rung::R1, 1, 8};
    if (k <= 16) return {Rung::R2_16, 1, 16};
    if (k <= 32) return {Rung::R2_32, 1, 32};
    if (k <= 64) return {Rung::R3, 1, 64};
    if (k <= 128) return {Rung::R4, 2, 128};
    for (uint32_t N : r5_word_counts())
        if (64 * N >= k) return {Rung::R5, N, 64 * N};
    ENGINE_CHECK_MSG(false, "ring2k::select_rung: unreachable");
    return {};
}
RungChoice next_rung(const RungChoice& c) {
    if (c.capacity >= 4096) return {c.rung, c.words, 0};
    return select_rung(c.capacity + 1);
}
std::vector<RungChoice> rungs_for(uint32_t k) {
    std::vector<RungChoice> v;
    if (k == 1) v.push_back(select_rung(1));
    RungChoice c = select_rung(k < 2 ? 2 : k);
    while (c.capacity != 0) {
        v.push_back(c);
        c = next_rung(c);
    }
    return v;
}

WideMatrix::WideMatrix(uint32_t r, uint32_t c, uint32_t k_) : rows(r), cols(c), k(k_), nw((k_ + 63) / 64) {
    ENGINE_CHECK_MSG(k_ >= 1, "ring2k::WideMatrix: k >= 1");
    w.assign(static_cast<size_t>(r) * c * nw, 0);
}
void WideMatrix::canonicalise() {
    const uint32_t s = k % 64;
    if (s == 0) return;
    for (size_t e = 0; e < static_cast<size_t>(rows) * cols; ++e) w[e * nw + nw - 1] &= (uint64_t(1) << s) - 1;
}

SmithResult smith_on(const RungChoice& rung, const WideMatrix& A) {
    if (rung.rung == Rung::R0) {
        ENGINE_CHECK_MSG(A.k == 1, "ring2k::smith_on: R0 carries k = 1 only");
        SmithResult s;
        s.k = 1;
        const uint32_t r = binary::rank(to_bits(A));
        s.sigma.assign(r, 0);
        if (r) s.level_ranks.assign(1, r);
        return s;
    }
    ENGINE_CHECK_MSG(A.k <= rung.capacity, "ring2k::smith_on: k exceeds the rung's capacity");
    return dispatch_rung(rung, [&](auto tag) {
        using E = decltype(tag);
        return smith<Traits<E>>(from_wide<E>(A), A.k);
    });
}
WideHowell howell_on(const RungChoice& rung, const WideMatrix& A) {
    if (rung.rung == Rung::R0) {
        ENGINE_CHECK_MSG(A.k == 1, "ring2k::howell_on: R0 carries k = 1 only");
        binary::BitMatrix B = to_bits(A);
        const binary::BitRref r = binary::rref(B);
        return bits_to_howell(B, r.pivots);
    }
    ENGINE_CHECK_MSG(A.k <= rung.capacity, "ring2k::howell_on: k exceeds the rung's capacity");
    return dispatch_rung(rung, [&](auto tag) {
        using E = decltype(tag);
        return to_wide(howell<Traits<E>>(from_wide<E>(A), A.k));
    });
}
WideHowell kernel_module_on(const RungChoice& rung, const WideMatrix& A) {
    if (rung.rung == Rung::R0) {
        ENGINE_CHECK_MSG(A.k == 1, "ring2k::kernel_module_on: R0 carries k = 1 only");
        binary::BitMatrix B = to_bits(A);
        const binary::BitRref r = binary::rref(B);
        binary::BitMatrix K = binary::kernel_basis(B, r.pivots);
        const binary::BitRref rk = binary::rref(K);
        return bits_to_howell(K, rk.pivots);
    }
    ENGINE_CHECK_MSG(A.k <= rung.capacity, "ring2k::kernel_module_on: k exceeds the rung's capacity");
    return dispatch_rung(rung, [&](auto tag) {
        using E = decltype(tag);
        return to_wide(kernel_module<Traits<E>>(from_wide<E>(A), A.k));
    });
}
SmithResult smith_naive_on(const RungChoice& rung, const WideMatrix& A) {
    ENGINE_CHECK_MSG(rung.rung != Rung::R0 && A.k <= rung.capacity, "ring2k::smith_naive_on: not R0; k <= capacity");
    return dispatch_rung(rung, [&](auto tag) {
        using E = decltype(tag);
        SmithResult s = smith_naive<Traits<E>>(from_wide<E>(A), A.k);
        std::sort(s.sigma.begin(), s.sigma.end());
        return s;
    });
}
WideHowell howell_naive_on(const RungChoice& rung, const WideMatrix& A) {
    ENGINE_CHECK_MSG(rung.rung != Rung::R0 && A.k <= rung.capacity, "ring2k::howell_naive_on: not R0; k <= capacity");
    return dispatch_rung(rung, [&](auto tag) {
        using E = decltype(tag);
        return to_wide(howell_naive<Traits<E>>(from_wide<E>(A), A.k));
    });
}
SmithResult smith_wide(const WideMatrix& A) { return smith_on(select_rung(A.k), A); }
WideHowell  howell_wide(const WideMatrix& A) { return howell_on(select_rung(A.k), A); }
WideHowell  kernel_module_wide(const WideMatrix& A) { return kernel_module_on(select_rung(A.k), A); }

AdaptiveSmith smith_adaptive(const std::function<WideMatrix(uint32_t K)>& produce, uint32_t k_min, uint32_t guard,
                             uint32_t k_cap, int64_t rank_hint) {
    ENGINE_CHECK_MSG(k_min >= 2 && k_min <= k_cap && k_cap <= 4096, "ring2k::smith_adaptive: need 2 <= k_min <= k_cap <= 4096");
    AdaptiveSmith out;
    RungChoice rung = select_rung(k_min);
    for (;;) {
        const uint32_t K = rung.capacity < k_cap ? rung.capacity : k_cap;
        const WideMatrix A = produce(K);
        ENGINE_CHECK_MSG(A.k == K, "ring2k::smith_adaptive: produce(K) must return a matrix over Z/2^K");
        out.smith = smith_on(rung, A);
        out.rung = rung;
        out.K = K;
        const uint32_t lim = K > guard ? K - guard : 0;
        uint32_t below = 0, near = 0, smax = 0;
        for (uint32_t s : out.smith.sigma) {
            if (s < lim) ++below; else ++near;
            if (s > smax) smax = s;
        }
        const bool rank_ok = rank_hint < 0 || static_cast<int64_t>(below) == rank_hint;
        out.resolved = near == 0 && rank_ok;
        out.verified = out.resolved && rank_hint >= 0;
        const uint32_t mn = A.rows < A.cols ? A.rows : A.cols;
        out.zeros = mn - static_cast<uint32_t>(out.smith.sigma.size());
        const char* verdict = !out.resolved ? (K >= k_cap ? "UNRESOLVED at k_cap" : "rerun one rung up")
                            : (rank_hint < 0 && out.zeros > 0) ? "resolved (unverified: no rank hint)" : "resolved";
        char buf[360];
        std::snprintf(buf, sizeof(buf),
                      "RING2K_ADAPTIVE run=%u rung=%s K=%u guard=%u invariants=%zu zeros=%u sigma_max=%u near_cap=%u "
                      "rank_hint=%lld below_cap=%u -> %s",
                      out.reruns, rung_label(rung).c_str(), K, guard, out.smith.sigma.size(), out.zeros, smax, near,
                      static_cast<long long>(rank_hint), below, verdict);
        out.log.emplace_back(buf);
        if (out.resolved || K >= k_cap) break;
        const RungChoice nx = next_rung(rung);
        if (nx.capacity == 0) break;
        rung = nx;
        ++out.reruns;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Explicit instantiations of the production rungs.
// ---------------------------------------------------------------------------
#define R2K_INSTANTIATE(E)                                                                                  \
    template EchelonInfo echelon_blocked<Traits<E>>(Mat<E>&, uint32_t, PivotRule, uint32_t, uint32_t, uint32_t); \
    template SmithResult smith<Traits<E>>(Mat<E>, uint32_t);                                                \
    template HowellForm<E> howell_pass<Traits<E>>(Mat<E>, uint32_t, uint32_t);                       \
    template HowellForm<E> howell<Traits<E>>(Mat<E>, uint32_t);                                      \
    template HowellForm<E> howell_naive<Traits<E>>(const Mat<E>&, uint32_t);                                \
    template HowellForm<E> kernel_module<Traits<E>>(const Mat<E>&, uint32_t);                               \
    template SmithResult smith_naive<Traits<E>>(Mat<E>, uint32_t);
R2K_INSTANTIATE(uint8_t)
R2K_INSTANTIATE(uint16_t)
R2K_INSTANTIATE(uint32_t)
R2K_INSTANTIATE(uint64_t)
R2K_INSTANTIATE(u128)
#define R2K_W(N) R2K_INSTANTIATE(WordN<N>)
R2K_W(3) R2K_W(4) R2K_W(5) R2K_W(6) R2K_W(8) R2K_W(10) R2K_W(12) R2K_W(16) R2K_W(20) R2K_W(24) R2K_W(32) R2K_W(40)
R2K_W(48) R2K_W(64)
#undef R2K_W
#undef R2K_INSTANTIATE

} // namespace ring2k
} // namespace engine
