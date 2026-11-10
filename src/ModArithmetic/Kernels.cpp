// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#include "pch.h"
#include "Kernels.h"
#include "../Engine/Threads.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ===========================================================================
// Kernels.cpp -- see Kernels.h for the contracts.
//
// GEMM structure (uint32_t elements; the measured regime of plan D2):
//
//   C op= A*B is computed as an OUTER-PRODUCT register-blocked microkernel:
//   for each k, broadcast A(i,k) for MR rows and stream B(k, j..j+16) as two
//   zero-extended 8 x u64 vectors, accumulating MR x 16 exact u64 partial
//   sums (16 zmm accumulators at MR = 8). This is the "k-j" order of
//   03 Sec. 7.2 (12-23x over the legacy loop) with the 4x16 / 8x16 register
//   block of 04 App. A. There is no horizontal reduction: every accumulator
//   lane IS a C cell, so the tile's reductions are one Barrett per cell,
//   amortised over the whole k-block.
//
//   Layout requirement of the microkernel: B's rows must be contiguous
//   (B(k, j+1) adjacent to B(k, j)), i.e. B row-major or a transposed view of
//   a column-major buffer. A is read by scalar broadcast, so any A layout
//   works with no penalty (a column-major A even reads the MR broadcast
//   sources from one cache line). A B operand whose rows are not contiguous
//   is packed once into a thread-local row-major scratch (O(kn)).
//
//   Cache blocking: k in blocks of KC = min(k_max, 256) (the overflow bound
//   AND an 8 x 256 x 4 B = 8 KB A tile that stays in L1 across the j sweep),
//   n in panels of NC = 512 (a 256 x 512 x 4 B = 512 KB B panel in L2).
//
//   Two microkernel bodies: AVX-512 (compiled via a target attribute so the
//   default -O2 build can also run it on an AVX-512 CPU; selected at runtime)
//   and portable C++. Both produce identical bits: the accumulated integers
//   are exact and the reduction is the same ModCtx::reduce_u64.
//
// uint64_t elements use unsigned __int128 accumulators with the same
// k-blocking; correct and simple (speed is a GPU matter, plan D2).
// ===========================================================================

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#  include <immintrin.h>
#  define ENGINE_HAVE_AVX512_TARGET 1
#  define ENGINE_AVX512_FN __attribute__((target("avx512f,avx512vl,avx512dq,avx512bw")))
#  define ENGINE_VNNI_FN __attribute__((target("avx512f,avx512vl,avx512dq,avx512bw,avx512vnni")))
#  define ENGINE_AVX2_FN __attribute__((target("avx2,fma")))
#else
#  define ENGINE_HAVE_AVX512_TARGET 0
#  define ENGINE_AVX512_FN
#  define ENGINE_VNNI_FN
#  define ENGINE_AVX2_FN
#endif

namespace engine {
namespace kernels {

// ---------------------------------------------------------------------------
// Kernel path selection
// ---------------------------------------------------------------------------
namespace {

KernelPath g_kernel_path = KernelPath::Auto;

// (S24 CL-5) the CPU facts from CpuFeatures.h; "available" = this build compiled the body && the CPU has it
bool cpu_has_avx512() { return ENGINE_HAVE_AVX512_TARGET && cpu::avx512_f_bw_dq_vl(); }
// (slot-24 AVX2K) the AVX2 microkernel's ISA: avx2 + fma (every AVX2 CPU since Haswell / Zen 1 has both).
bool cpu_has_avx2() { return ENGINE_HAVE_AVX512_TARGET && cpu::avx2_fma(); }

// (slot-24 AVX2K) ENGINE_KERNEL, resolved exactly once per process (a function-local static): the path Auto
// follows. Read-only: every path is bit-identical, so the setting can change only the speed.
KernelPath kernel_path_from_env() {
    static const KernelPath v = [] {
        const char* e = std::getenv("ENGINE_KERNEL");
        if (e == nullptr || e[0] == '\0') return KernelPath::Auto;
        KernelPath r = KernelPath::Auto;
        bool known = true;
        if (std::strcmp(e, "portable") == 0 || std::strcmp(e, "scalar") == 0) r = KernelPath::Portable;
        else if (std::strcmp(e, "avx2") == 0) r = KernelPath::Avx2;
        else if (std::strcmp(e, "avx512") == 0) r = KernelPath::Avx512;
        else if (std::strcmp(e, "auto") != 0) known = false;
        if (!known) {
            std::fprintf(stderr, "WARNING engine: ENGINE_KERNEL='%s' is none of portable, avx2, avx512, auto; "
                                 "the kernel path stays auto\n", e);
        } else if ((r == KernelPath::Avx2 && !cpu_has_avx2()) || (r == KernelPath::Avx512 && !cpu_has_avx512())) {
            std::fprintf(stderr, "WARNING engine: ENGINE_KERNEL='%s' is unavailable on this CPU / build; "
                                 "the kernel path stays auto\n", e);
            r = KernelPath::Auto;
        }
        std::fprintf(stderr, "EFFECTIVE ENGINE_KERNEL path=%s value=%s cpu=avx2:%d,avx512:%d "
                             "governs=gemm,ring2k,ring3k(portable),bitmatrix,tritmatrix,rho_fill,descent_smith\n",
                     kernel_path_name(r), e, cpu_has_avx2() ? 1 : 0, cpu_has_avx512() ? 1 : 0);
        return r;
    }();
    return v;
}

} // namespace

const char* kernel_path_name(KernelPath path) { return cpu::isa_path_name(path); }
cpu::IsaPath isa_ceiling() { return kernel_path_from_env(); }

void set_kernel_path(KernelPath path) {
    if (path == KernelPath::Avx512) {
        ENGINE_CHECK_MSG(cpu_has_avx512(), "set_kernel_path: AVX-512 kernel unavailable on this CPU/build");
    }
    if (path == KernelPath::Avx2) {
        ENGINE_CHECK_MSG(cpu_has_avx2(), "set_kernel_path: AVX2 kernel unavailable on this CPU/build");
    }
    g_kernel_path = path;
}
KernelPath kernel_path() { return g_kernel_path; }
bool avx512_kernel_available() { return cpu_has_avx512(); }
bool avx2_kernel_available() { return cpu_has_avx2(); }
KernelPath active_kernel_path() {
    KernelPath k = g_kernel_path;
    if (k == KernelPath::Auto) k = kernel_path_from_env();
    if (k == KernelPath::Auto)
        k = cpu_has_avx512() ? KernelPath::Avx512 : cpu_has_avx2() ? KernelPath::Avx2 : KernelPath::Portable;
    return k;
}
bool avx512_kernel_active() { return active_kernel_path() == KernelPath::Avx512; }

// ---------------------------------------------------------------------------
// (slot-23 PERF-P5) VNNI route selection
// ---------------------------------------------------------------------------
namespace {

bool cpu_has_vnni() { return ENGINE_HAVE_AVX512_TARGET && cpu::avx512_vnni(); }

// The ENGINE_VNNI setting, resolved exactly once per process (a function-local static: thread-safe, so the
// EFFECTIVE / WARNING lines print once even when threads race on their first gemm; G4 audit F3).
// "0" and "off" select Off; anything else (unset included) is Auto.
VnniPath vnni_path_from_env() {
    static const VnniPath v = [] {
        const char* e = std::getenv("ENGINE_VNNI");
        const VnniPath r = (e != nullptr && (std::strcmp(e, "0") == 0 || std::strcmp(e, "off") == 0)) ? VnniPath::Off
                                                                                                       : VnniPath::Auto;
        if (e != nullptr) {
            // (slot-24 G4 audit F2) an unrecognised value keeps the route ON; say so (the L23_RING2K style).
            if (e[0] != '\0' && r == VnniPath::Auto && std::strcmp(e, "1") != 0)
                std::fprintf(stderr, "WARNING engine: ENGINE_VNNI='%s' is neither 0, off nor 1; the VNNI route stays ON "
                                     "(only exactly 0 or off turns it off)\n", e);
            // (slot-24 G4) the EFFECTIVE convention: a set ENGINE_VNNI names what it selected (stderr, so no stdout
            // transcript moves; unset prints nothing). The rollback arm covers both limb counts.
            std::fprintf(stderr, "EFFECTIVE ENGINE_VNNI arm=%s value=%s vnni_available=%d one_limb_max=%u two_limb_max=%u\n",
                         r == VnniPath::Off ? "off" : "auto", e, cpu_has_vnni() ? 1 : 0, VNNI_MAX_MODULUS, VNNI2_MAX_MODULUS);
        }
        return r;
    }();
    return v;
}

// -1 = not yet read (follow ENGINE_VNNI), else the VnniPath value.
std::atomic<int> g_vnni_path{-1};
std::atomic<uint64_t> g_vnni_routed{0};
std::atomic<uint64_t> g_vnni2_routed{0};

[[maybe_unused]] bool vnni_shadow_env() {
    static const bool on = [] {
        const char* e = std::getenv("ENGINE_VNNI_SHADOW");
        return e != nullptr && e[0] == '1' && e[1] == '\0';
    }();
    return on;
}

// ENGINE_VNNI_SHADOW=1: the comparisons made, reported once at process exit (stderr) so a shadow run shows it was
// not vacuous: `VNNI_SHADOW calls=<routed calls compared> cells=<C cells compared> differ=<calls with a difference>`.
// A difference throws (the harness catches it, so the process can go on); differ counts those calls (audit R2).
std::atomic<uint64_t> g_shadow_calls{0}, g_shadow_cells{0}, g_shadow_differ{0};
struct ShadowReport {
    ~ShadowReport() {
        if (vnni_shadow_env())
            std::fprintf(stderr, "VNNI_SHADOW calls=%llu cells=%llu differ=%llu\n",
                         static_cast<unsigned long long>(g_shadow_calls.load()),
                         static_cast<unsigned long long>(g_shadow_cells.load()),
                         static_cast<unsigned long long>(g_shadow_differ.load()));
    }
} g_shadow_report;

// Below these sizes the packing does not pay; a pure speed choice (values are
// identical on both routes).
constexpr uint32_t VNNI_MIN_M = 8, VNNI_MIN_N = 32, VNNI_MIN_K = 16;
// (slot-24 G4 audit F1) From p >= 2^13 the u32 kernel's epilogue switches to its FP reduction (FP_REDUCE_MIN_P below)
// and becomes ~20 % faster, so on small tiles the two-limb route loses there (measured, G4_report.md "Audit fixes":
// 64^3 0.79x, 96^3 0.94x, 128 x 32 x 128 1.00x, 256 x 16 x 256 1.07x at p = 12289 / 16381). For p >= 2^13 Auto
// therefore also needs m, n >= 128. Below 2^13 nothing changes. A pure speed choice (values are identical).
constexpr uint32_t VNNI2_LARGE_P_MIN = 1u << 13, VNNI2_LARGE_P_MIN_MN = 128;

// (slot-25 PERF-VF (a)) The two-limb route's work-per-thread floor; see vnni2_work_floor() in Kernels.h.
// ENGINE_VNNI2_FLOOR=<digits> (read once; 0 = no floor, the slot-24 Auto rule) overrides it and prints one EFFECTIVE line.
uint64_t vnni2_work_floor_from_env() {
    static const uint64_t v = [] {
        uint64_t r = VNNI2_WORK_FLOOR_DEFAULT;
        const char* e = std::getenv("ENGINE_VNNI2_FLOOR");
        if (e != nullptr) {
            // (PERF-VF audit S1) digits only: strtoull would accept leading whitespace and a sign (" -1" -> 2^64-1),
            // and it saturates silently on overflow, so both are rejected explicitly.
            char* end = nullptr;
            errno = 0;
            const unsigned long long x = std::isdigit(static_cast<unsigned char>(e[0])) ? std::strtoull(e, &end, 10) : 0ull;
            const bool ok = std::isdigit(static_cast<unsigned char>(e[0])) && errno != ERANGE && end != nullptr && *end == '\0';
            if (ok) r = static_cast<uint64_t>(x);
            else std::fprintf(stderr, "WARNING engine: ENGINE_VNNI2_FLOOR='%s' is not a non-negative integer; the default %llu stays\n",
                              e, static_cast<unsigned long long>(VNNI2_WORK_FLOOR_DEFAULT));
            std::fprintf(stderr, "EFFECTIVE ENGINE_VNNI2_FLOOR floor=%llu value=%s threads=%u\n",
                         static_cast<unsigned long long>(r), e, vnni_route_threads());
        }
        return r;
    }();
    return v;
}

} // namespace

uint32_t vnni_route_threads() {
    // The global pool's size rule (Engine/Threads.h kThreadPool), resolved once; resolve() prints nothing.
    static const uint32_t t = [] {
        const engine::threads::Result r = engine::threads::resolve(std::getenv("ENGINE_THREADS"), engine::threads::kThreadPool,
                                                                   engine::threads::hardware(), 0);
        return r.threads == 0 ? 1u : r.threads;
    }();
    return t;
}
uint64_t vnni2_work_floor() { return vnni2_work_floor_from_env(); }
bool vnni2_work_passes(uint32_t m, uint32_t n, uint32_t k, uint32_t threads) {
    const uint64_t work = static_cast<uint64_t>(m) * n * k;
    const uint64_t floor = vnni2_work_floor();
    const uint64_t t = threads == 0 ? 1u : threads;
    // (PERF-VF audit S1) saturating floor * threads: a huge floor must not wrap to a small (or zero) bound.
    const uint64_t bound = floor > UINT64_MAX / t ? UINT64_MAX : floor * t;
    return work >= bound;
}

void set_vnni_path(VnniPath path) {
    (void)vnni_path_from_env();   // (G4 audit F3) a set ENGINE_VNNI still prints its EFFECTIVE line when overridden
    if (path == VnniPath::Force) {
        ENGINE_CHECK_MSG(cpu_has_vnni(), "set_vnni_path: AVX-512 VNNI unavailable on this CPU/build");
    }
    g_vnni_path.store(static_cast<int>(path));
}
VnniPath vnni_path() {
    int v = g_vnni_path.load();
    if (v < 0) {
        int expected = -1;
        g_vnni_path.compare_exchange_strong(expected, static_cast<int>(vnni_path_from_env()));
        v = g_vnni_path.load();
    }
    return static_cast<VnniPath>(v);
}
bool vnni_kernel_available() { return cpu_has_vnni(); }
uint64_t vnni_routed_calls() { return g_vnni_routed.load(std::memory_order_relaxed); }
uint64_t vnni2_routed_calls() { return g_vnni2_routed.load(std::memory_order_relaxed); }
bool vnni_route_taken(uint32_t p, uint32_t m, uint32_t n, uint32_t k) {
    if (vnni_limbs_for(p) == 0 || m == 0 || n == 0 || k == 0) return false;
    switch (vnni_path()) {
        case VnniPath::Off:   return false;
        case VnniPath::Force: return cpu_has_vnni();
        default:
            if (p >= VNNI2_LARGE_P_MIN && (m < VNNI2_LARGE_P_MIN_MN || n < VNNI2_LARGE_P_MIN_MN)) return false;
            // (slot-25 PERF-VF (a)) the two-limb route's work-per-thread floor (a speed choice only)
            if (vnni_limbs_for(p) == 2 && !vnni2_work_passes(m, n, k, vnni_route_threads())) return false;
            return cpu_has_vnni() && active_kernel_path() == KernelPath::Avx512 &&
                   m >= VNNI_MIN_M && n >= VNNI_MIN_N && k >= VNNI_MIN_K;
    }
}

// ---------------------------------------------------------------------------
// (slot-24 AVX2K) AVX2 pair-route selection
// ---------------------------------------------------------------------------
namespace {
// ENGINE_AVX2_PAIR, resolved once: "0" / "off" = Off (the rollback arm: the AVX2 path then runs its u64 microkernel
// for every p); anything else, unset included, = Auto. A set value prints one EFFECTIVE line on stderr.
Avx2PairPath avx2_pair_path_from_env() {
    static const Avx2PairPath v = [] {
        const char* e = std::getenv("ENGINE_AVX2_PAIR");
        const Avx2PairPath r = (e != nullptr && (std::strcmp(e, "0") == 0 || std::strcmp(e, "off") == 0)) ? Avx2PairPath::Off
                                                                                                           : Avx2PairPath::Auto;
        if (e != nullptr)
            std::fprintf(stderr, "EFFECTIVE ENGINE_AVX2_PAIR arm=%s value=%s avx2_available=%d max_modulus=%u\n",
                         r == Avx2PairPath::Off ? "off" : "auto", e, cpu_has_avx2() ? 1 : 0, AVX2_PAIR_MAX_MODULUS);
        return r;
    }();
    return v;
}
std::atomic<int> g_avx2_pair_path{-1};
std::atomic<uint64_t> g_avx2_pair_routed{0};
// Below these sizes the packing does not pay (a pure speed choice; values are identical on both routes).
constexpr uint32_t AVX2P_MIN_M = 4, AVX2P_MIN_N = 16, AVX2P_MIN_K = 16;
} // namespace

void set_avx2_pair_path(Avx2PairPath path) {
    (void)avx2_pair_path_from_env();
    if (path == Avx2PairPath::Force) ENGINE_CHECK_MSG(cpu_has_avx2(), "set_avx2_pair_path: AVX2 unavailable on this CPU/build");
    g_avx2_pair_path.store(static_cast<int>(path));
}
Avx2PairPath avx2_pair_path() {
    int v = g_avx2_pair_path.load();
    if (v < 0) {
        int expected = -1;
        g_avx2_pair_path.compare_exchange_strong(expected, static_cast<int>(avx2_pair_path_from_env()));
        v = g_avx2_pair_path.load();
    }
    return static_cast<Avx2PairPath>(v);
}
uint64_t avx2_pair_routed_calls() { return g_avx2_pair_routed.load(std::memory_order_relaxed); }
bool avx2_pair_route_taken(uint32_t p, uint32_t m, uint32_t n, uint32_t k) {
    if (p < 2 || p > AVX2_PAIR_MAX_MODULUS || m == 0 || n == 0 || k == 0 || !cpu_has_avx2()) return false;
    switch (avx2_pair_path()) {
        case Avx2PairPath::Off:   return false;
        case Avx2PairPath::Force: return true;
        default:
            return active_kernel_path() == KernelPath::Avx2 && m >= AVX2P_MIN_M && n >= AVX2P_MIN_N && k >= AVX2P_MIN_K;
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------
namespace {

constexpr uint32_t KC_MAX = 256;   // k-block (also bounded by ctx.k_max)
constexpr uint32_t NC     = 512;   // B panel width
constexpr uint32_t NR     = 16;    // microkernel columns
constexpr uint32_t TILE   = 16;    // transpose / copy tile

template<class Elt>
inline void combine_cell(const ModCtx<Elt>& ctx, Elt& c, Elt r, GemmOp op) {
    c = (op == GemmOp::Sub) ? ctx.sub(c, r) : ctx.add(c, r);
}

template<class E1, class E2>
bool windows_overlap(const DenseView<E1>& a, const DenseView<E2>& b) {
    if (static_cast<const void*>(a.buffer()) != static_cast<const void*>(b.buffer())) return false;
    const bool rows_meet = a.row0() < b.row0() + b.buf_rows() && b.row0() < a.row0() + a.buf_rows();
    const bool cols_meet = a.col0() < b.col0() + b.buf_cols() && b.col0() < a.col0() + a.buf_cols();
    return rows_meet && cols_meet && !a.empty() && !b.empty();
}

template<class Elt>
void debug_check_canonical(const ModCtx<Elt>& ctx, DenseView<const Elt> v, const char* what) {
    (void)ctx; (void)v; (void)what;
#ifndef NDEBUG
    for (uint32_t i = 0; i < v.rows(); ++i)
        for (uint32_t j = 0; j < v.cols(); ++j)
            ENGINE_DEBUG_CHECK_MSG(v.at(i, j) < ctx.p, what);
#endif
}

template<class Elt>
void zero_view(DenseView<Elt> v) {
    const auto s = v.strides();
    for (uint32_t i = 0; i < v.rows(); ++i) {
        Elt* row = s.base + i * s.rs;
        for (uint32_t j = 0; j < v.cols(); ++j) row[j * s.cs] = 0;
    }
}

// Thread-local packing scratch (arena-ready: a non-owning DenseBuf over it).
template<class Elt>
std::vector<Elt>& pack_scratch() {
    static thread_local std::vector<Elt> s;
    return s;
}

// Pack a k x n view into row-major contiguous storage (ld = n).
template<class Elt>
const Elt* pack_rowmajor(DenseView<const Elt> B, std::vector<Elt>& scratch) {
    const uint32_t k = B.rows(), n = B.cols();
    const size_t need = static_cast<size_t>(k) * n;
    if (scratch.size() < need) scratch.resize(need);
    const auto s = B.strides();
    Elt* out = scratch.data();
    // tile the copy so a column-major source is read in cache-friendly order
    for (uint32_t k0 = 0; k0 < k; k0 += TILE) {
        const uint32_t k1 = std::min(k, k0 + TILE);
        for (uint32_t j0 = 0; j0 < n; j0 += TILE) {
            const uint32_t j1 = std::min(n, j0 + TILE);
            for (uint32_t kk = k0; kk < k1; ++kk) {
                const Elt* src = s.base + kk * s.rs;
                Elt* dst = out + static_cast<size_t>(kk) * n;
                for (uint32_t j = j0; j < j1; ++j) dst[j] = src[j * s.cs];
            }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// uint32_t microkernels.  Signature (both bodies):
//   A   : element (i, kk) at A[i*ars + kk*acs], i < MR, kk < k
//   B   : element (kk, j) at B[kk*brs + j],      j < ncols <= 16 (rows contiguous)
//   C   : element (i, j)  at C[i*crs + j*ccs]
//   op  : Add or Sub (Set has been turned into zero + Add by the driver)
// Accumulates exactly in u64 (k <= k_max guaranteed by the driver), reduces
// with ctx.reduce_u64 and combines into C.
// ---------------------------------------------------------------------------

// Portable body. MR <= 4 so that the MR x 16 u64 accumulators (<= 16 ymm /
// 8 zmm) can stay in registers when the compiler vectorises the j loop
// (-O3); at -O2 without -ftree-vectorize this is scalar code and ~2 G.
template<int MR>
void micro_u32_portable(const ModCtx<uint32_t>& ctx,
                        const uint32_t* __restrict A, size_t ars, size_t acs,
                        const uint32_t* __restrict B, size_t brs, uint32_t k,
                        uint32_t* __restrict C, size_t crs, size_t ccs, uint32_t ncols, GemmOp op) {
    static_assert(MR <= 4, "portable tile: keep the accumulators register-resident");
    uint64_t acc[MR][NR];
#pragma GCC unroll 4
    for (int i = 0; i < MR; ++i)
        for (uint32_t j = 0; j < NR; ++j) acc[i][j] = 0;

    if (ncols == NR) {
        for (uint32_t kk = 0; kk < k; ++kk) {
            const uint32_t* b = B + static_cast<size_t>(kk) * brs;
#pragma GCC unroll 4
            for (int i = 0; i < MR; ++i) {
                const uint64_t a = A[i * ars + kk * acs];
                for (uint32_t j = 0; j < NR; ++j) acc[i][j] += a * b[j];
            }
        }
    } else {
        for (uint32_t kk = 0; kk < k; ++kk) {
            const uint32_t* b = B + static_cast<size_t>(kk) * brs;
#pragma GCC unroll 4
            for (int i = 0; i < MR; ++i) {
                const uint64_t a = A[i * ars + kk * acs];
                for (uint32_t j = 0; j < ncols; ++j) acc[i][j] += a * b[j];
            }
        }
    }
#pragma GCC unroll 4
    for (int i = 0; i < MR; ++i) {
        uint32_t* c = C + i * crs;
        for (uint32_t j = 0; j < ncols; ++j) combine_cell(ctx, c[j * ccs], ctx.reduce_u64(acc[i][j]), op);
    }
}

// 8-row portable tile = two 4-row tiles (same bits: each cell is its own exact sum).
void micro_u32_portable8(const ModCtx<uint32_t>& ctx,
                         const uint32_t* A, size_t ars, size_t acs,
                         const uint32_t* B, size_t brs, uint32_t k,
                         uint32_t* C, size_t crs, size_t ccs, uint32_t ncols, GemmOp op) {
    micro_u32_portable<4>(ctx, A, ars, acs, B, brs, k, C, crs, ccs, ncols, op);
    micro_u32_portable<4>(ctx, A + 4 * ars, ars, acs, B, brs, k, C + 4 * crs, crs, ccs, ncols, op);
}

#if ENGINE_HAVE_AVX512_TARGET
// Smallest p for which the FP64-quotient reduction below is exact.
constexpr uint32_t FP_REDUCE_MIN_P = 1u << 13;
static_assert(FP_REDUCE_MIN_P == VNNI2_LARGE_P_MIN, "the two-limb large-p threshold follows the u32 FP-reduction switch");

// x mod p for 8 exact u64 lanes, p in [2^13, 2^32).
//   q = trunc(fl(fl(x) * fl(1/p))).  Each of the three roundings has relative
//   error <= u = 2^-53, so |d - x/p| < (x/p) * 3.01u < (2^64/p) * 3.01 * 2^-53
//   = 3.01 * 2^11 / p < 0.76 for p >= 2^13; hence q in {floor(x/p) - 1,
//   floor(x/p), floor(x/p) + 1} and r = x - q*p lies in [-p, 2p).  r is formed
//   modulo 2^64 (vpmullq is a low multiply) and |r| < 2^33, so its int64
//   reading is exact: one masked "+p if negative", one masked "-p if >= p".
//   Canonical result == ModCtx::reduce_u64(x) bit for bit (the residue is
//   unique); ENGINE_DEBUG_CHECKed against it in non-NDEBUG builds.
ENGINE_AVX512_FN inline __m256i fp_reduce_u64(__m512i x, __m512d inv_p, __m512i p64) {
    const __m512d xd = _mm512_cvtepu64_pd(x);
    const __m512i q  = _mm512_cvttpd_epu64(_mm512_mul_pd(xd, inv_p));
    __m512i r = _mm512_sub_epi64(x, _mm512_mullo_epi64(q, p64));
    const __mmask8 neg = _mm512_cmplt_epi64_mask(r, _mm512_setzero_si512());
    r = _mm512_mask_add_epi64(r, neg, r, p64);
    const __mmask8 big = _mm512_cmpge_epu64_mask(r, p64);
    r = _mm512_mask_sub_epi64(r, big, r, p64);
    return _mm512_cvtepi64_epi32(r);
}

// c op= r on 8 canonical 32-bit lanes: the legacy mod_add / mod_sub formulas
// (wrap test `s < c`, so p > 2^31 is exact too).
ENGINE_AVX512_FN inline __m256i combine_lanes(__m256i c, __m256i r, __m256i p32, GemmOp op) {
    if (op == GemmOp::Sub) {
        const __m256i s = _mm256_sub_epi32(c, r);
        const __mmask8 wrap = _mm256_cmplt_epu32_mask(c, r);          // c < r  =>  s wrapped: add p
        return _mm256_mask_add_epi32(s, wrap, s, p32);
    }
    const __m256i s = _mm256_add_epi32(c, r);
    const __mmask8 fix = _mm256_cmplt_epu32_mask(s, c) | _mm256_cmpge_epu32_mask(s, p32);
    return _mm256_mask_sub_epi32(s, fix, s, p32);
}

template<int MR>
ENGINE_AVX512_FN
void micro_u32_avx512(const ModCtx<uint32_t>& ctx,
                      const uint32_t* A, size_t ars, size_t acs,
                      const uint32_t* B, size_t brs, uint32_t k,
                      uint32_t* C, size_t crs, size_t ccs, uint32_t ncols, GemmOp op) {
    const unsigned m16 = (ncols >= 16) ? 0xFFFFu : ((1u << ncols) - 1u);
    const __mmask8 mlo = static_cast<__mmask8>(m16 & 0xFFu);
    const __mmask8 mhi = static_cast<__mmask8>(m16 >> 8);

    // The MR x 2 accumulators MUST live in zmm registers for the whole k loop.
    // At -O2 GCC does not unroll the i loop on its own and then keeps the
    // arrays in memory (one load + one store per MAC, measured 2x slower), so
    // the unroll is forced; after full unrolling SRA scalarises the arrays.
    __m512i acc0[MR], acc1[MR];
#pragma GCC unroll 8
    for (int i = 0; i < MR; ++i) { acc0[i] = _mm512_setzero_si512(); acc1[i] = _mm512_setzero_si512(); }

    const uint32_t* b = B;
    for (uint32_t kk = 0; kk < k; ++kk, b += brs) {
        // B(kk, j) zero-extended to u64 lanes; masked lanes are 0 and never stored
        const __m512i b0 = _mm512_cvtepu32_epi64(_mm256_maskz_loadu_epi32(mlo, b));
        const __m512i b1 = _mm512_cvtepu32_epi64(_mm256_maskz_loadu_epi32(mhi, b + 8));
        const uint32_t* a = A + kk * acs;
#pragma GCC unroll 8
        for (int i = 0; i < MR; ++i) {
            // low dword of every qword lane = A(i, kk); mul_epu32 reads only the low dwords
            const __m512i av = _mm512_set1_epi32(static_cast<int>(a[i * ars]));
            acc0[i] = _mm512_add_epi64(acc0[i], _mm512_mul_epu32(av, b0));
            acc1[i] = _mm512_add_epi64(acc1[i], _mm512_mul_epu32(av, b1));
        }
    }

    // ---- epilogue: reduce the MR x 16 exact sums and combine into C ----
    if (ctx.p >= FP_REDUCE_MIN_P && ccs == 1) {
        // Vectorised Barrett with an FP64 quotient (see fp_reduce_u64 for the
        // exactness argument), canonical add/sub in 32-bit lanes with the
        // legacy wrap test, masked 256-bit stores. Row-major C only.
        const __m512d inv_p = _mm512_set1_pd(1.0 / static_cast<double>(ctx.p));
        const __m512i p64   = _mm512_set1_epi64(static_cast<long long>(ctx.p));
        const __m256i p32   = _mm256_set1_epi32(static_cast<int>(ctx.p));
#pragma GCC unroll 8
        for (int i = 0; i < MR; ++i) {
            uint32_t* c = C + i * crs;
            const __m256i r0 = fp_reduce_u64(acc0[i], inv_p, p64);
            const __m256i r1 = fp_reduce_u64(acc1[i], inv_p, p64);
#ifndef NDEBUG
            {   // cross-check the FP-quotient reduction against the scalar Barrett
                alignas(64) uint64_t x[NR]; alignas(32) uint32_t r[NR];
                _mm512_store_si512(reinterpret_cast<__m512i*>(x), acc0[i]);
                _mm512_store_si512(reinterpret_cast<__m512i*>(x + 8), acc1[i]);
                _mm256_store_si256(reinterpret_cast<__m256i*>(r), r0);
                _mm256_store_si256(reinterpret_cast<__m256i*>(r + 8), r1);
                for (uint32_t j = 0; j < NR; ++j)
                    ENGINE_DEBUG_CHECK_MSG(r[j] == ctx.reduce_u64(x[j]), "fp_reduce_u64 != reduce_u64");
            }
#endif
            const __m256i c0 = _mm256_maskz_loadu_epi32(mlo, c);
            const __m256i c1 = _mm256_maskz_loadu_epi32(mhi, c + 8);
            _mm256_mask_storeu_epi32(c,     mlo, combine_lanes(c0, r0, p32, op));
            _mm256_mask_storeu_epi32(c + 8, mhi, combine_lanes(c1, r1, p32, op));
        }
        return;
    }
    alignas(64) uint64_t tmp[NR];
#pragma GCC unroll 8
    for (int i = 0; i < MR; ++i) {
        _mm512_store_si512(reinterpret_cast<__m512i*>(tmp), acc0[i]);
        _mm512_store_si512(reinterpret_cast<__m512i*>(tmp + 8), acc1[i]);
        uint32_t* c = C + i * crs;
        for (uint32_t j = 0; j < ncols; ++j) combine_cell(ctx, c[j * ccs], ctx.reduce_u64(tmp[j]), op);
    }
}
#endif

#if ENGINE_HAVE_AVX512_TARGET
// ---------------------------------------------------------------------------
// (slot-24 AVX2K) the AVX2 body, for CPUs with AVX2 but no AVX-512 (an AMD
// EPYC 7763 / Zen 3 node, an Intel Core i9-9900KF). Same outer-product scheme as the
// AVX-512 body on 256-bit registers: a sub-tile of RG <= 4 rows x 8 columns
// keeps RG x 2 ymm accumulators of 4 exact u64 sums each (<= 8 accumulators
// + 2 B vectors + 1 broadcast = 11 of the 16 ymm registers); B(kk, j..j+8) is
// zero-extended to two 4 x u64 vectors, A(i, kk) is broadcast and multiplied
// with vpmuludq (low dwords only), i.e. every lane accumulates the exact
// integer sum_kk A(i,kk) B(kk,j) < k_max (p-1)^2 < 2^64. An MR x 16 tile is
// swept as ceil(MR/4) x 2 sub-tiles (the B rows stay in L1 between them).
// Epilogue: the exact sums are reduced with ctx.reduce_u64 and combined with
// combine_cell -- the portable body's own scalar formulas, so the bits are the
// portable body's by construction (and the AVX-512 body's, whose residues are
// the unique canonical ones too). Column tails (< 8 columns) use vpmaskmovd,
// which never touches a masked-off lane's memory.
// ---------------------------------------------------------------------------
template<int RG, bool FULL>
ENGINE_AVX2_FN
inline void avx2_subtile(const ModCtx<uint32_t>& ctx,
                         const uint32_t* A, size_t ars, size_t acs,
                         const uint32_t* B, size_t brs, uint32_t k,
                         uint32_t* C, size_t crs, size_t ccs, uint32_t n8, GemmOp op) {
    __m256i acc0[RG], acc1[RG];
#pragma GCC unroll 4
    for (int i = 0; i < RG; ++i) { acc0[i] = _mm256_setzero_si256(); acc1[i] = _mm256_setzero_si256(); }
    __m128i mlo = _mm_setzero_si128(), mhi = _mm_setzero_si128();
    if (!FULL) {
        const __m128i lane = _mm_setr_epi32(0, 1, 2, 3);
        mlo = _mm_cmpgt_epi32(_mm_set1_epi32(static_cast<int>(n8)), lane);
        mhi = _mm_cmpgt_epi32(_mm_set1_epi32(static_cast<int>(n8) - 4), lane);
    }
    const uint32_t* b = B;
    for (uint32_t kk = 0; kk < k; ++kk, b += brs) {
        const __m128i bl = FULL ? _mm_loadu_si128(reinterpret_cast<const __m128i*>(b))
                                : _mm_maskload_epi32(reinterpret_cast<const int*>(b), mlo);
        const __m128i bh = FULL ? _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + 4))
                                : _mm_maskload_epi32(reinterpret_cast<const int*>(b + 4), mhi);
        const __m256i b0 = _mm256_cvtepu32_epi64(bl);
        const __m256i b1 = _mm256_cvtepu32_epi64(bh);
        const uint32_t* a = A + kk * acs;
#pragma GCC unroll 4
        for (int i = 0; i < RG; ++i) {
            const __m256i av = _mm256_set1_epi32(static_cast<int>(a[i * ars]));
            acc0[i] = _mm256_add_epi64(acc0[i], _mm256_mul_epu32(av, b0));
            acc1[i] = _mm256_add_epi64(acc1[i], _mm256_mul_epu32(av, b1));
        }
    }
    alignas(32) uint64_t tmp[8];
#pragma GCC unroll 4
    for (int i = 0; i < RG; ++i) {
        _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), acc0[i]);
        _mm256_store_si256(reinterpret_cast<__m256i*>(tmp + 4), acc1[i]);
        uint32_t* c = C + i * crs;
        for (uint32_t j = 0; j < n8; ++j) combine_cell(ctx, c[j * ccs], ctx.reduce_u64(tmp[j]), op);
    }
}

template<int MR>
ENGINE_AVX2_FN
void micro_u32_avx2(const ModCtx<uint32_t>& ctx,
                    const uint32_t* A, size_t ars, size_t acs,
                    const uint32_t* B, size_t brs, uint32_t k,
                    uint32_t* C, size_t crs, size_t ccs, uint32_t ncols, GemmOp op) {
    constexpr int RG = MR < 4 ? MR : 4;
    static_assert(MR % RG == 0, "MR is 1, 2, 4 or 8");
    for (int g = 0; g < MR; g += RG) {
        const uint32_t* a = A + static_cast<size_t>(g) * ars;
        uint32_t* c = C + static_cast<size_t>(g) * crs;
        for (uint32_t c0 = 0; c0 < ncols; c0 += 8) {
            const uint32_t n8 = std::min<uint32_t>(8, ncols - c0);
            if (n8 == 8) avx2_subtile<RG, true>(ctx, a, ars, acs, B + c0, brs, k, c + c0 * ccs, crs, ccs, 8, op);
            else         avx2_subtile<RG, false>(ctx, a, ars, acs, B + c0, brs, k, c + c0 * ccs, crs, ccs, n8, op);
        }
    }
}
#endif

using MicroU32 = void (*)(const ModCtx<uint32_t>&, const uint32_t*, size_t, size_t,
                          const uint32_t*, size_t, uint32_t, uint32_t*, size_t, size_t,
                          uint32_t, GemmOp);

struct MicroSetU32 { MicroU32 m8, m4, m2, m1; };

MicroSetU32 select_micro_u32() {
#if ENGINE_HAVE_AVX512_TARGET
    const KernelPath kp = active_kernel_path();
    if (kp == KernelPath::Avx512) {
        return MicroSetU32{&micro_u32_avx512<8>, &micro_u32_avx512<4>,
                           &micro_u32_avx512<2>, &micro_u32_avx512<1>};
    }
    if (kp == KernelPath::Avx2) {   // (slot-24 AVX2K)
        return MicroSetU32{&micro_u32_avx2<8>, &micro_u32_avx2<4>,
                           &micro_u32_avx2<2>, &micro_u32_avx2<1>};
    }
#endif
    return MicroSetU32{&micro_u32_portable8, &micro_u32_portable<4>,
                       &micro_u32_portable<2>, &micro_u32_portable<1>};
}

// Driver for uint32_t: B rows contiguous (brs = row stride), op in {Add, Sub}.
void gemm_u32_driver(const ModCtx<uint32_t>& ctx,
                     const uint32_t* A, size_t ars, size_t acs,
                     const uint32_t* B, size_t brs,
                     uint32_t* C, size_t crs, size_t ccs,
                     uint32_t m, uint32_t n, uint32_t k, GemmOp op) {
    const MicroSetU32 mk = select_micro_u32();
    const uint32_t kc = static_cast<uint32_t>(std::min<uint64_t>(KC_MAX, ctx.k_max));
    for (uint32_t k0 = 0; k0 < k; k0 += kc) {
        const uint32_t kb = std::min(kc, k - k0);
        ENGINE_CHECK_MSG(kb <= ctx.k_max, "gemm: k-block exceeds the delayed-accumulation bound");
        for (uint32_t jc = 0; jc < n; jc += NC) {
            const uint32_t nc = std::min(NC, n - jc);
            uint32_t i0 = 0;
            while (i0 < m) {
                const uint32_t mr_left = m - i0;
                const uint32_t mr = mr_left >= 8 ? 8 : mr_left >= 4 ? 4 : mr_left >= 2 ? 2 : 1;
                const MicroU32 micro = mr == 8 ? mk.m8 : mr == 4 ? mk.m4 : mr == 2 ? mk.m2 : mk.m1;
                const uint32_t* a = A + i0 * ars + k0 * acs;
                uint32_t* c = C + i0 * crs;
                for (uint32_t j0 = 0; j0 < nc; j0 += NR) {
                    const uint32_t ncols = std::min(NR, nc - j0);
                    micro(ctx, a, ars, acs,
                          B + static_cast<size_t>(k0) * brs + jc + j0, brs, kb,
                          c + (jc + j0) * ccs, crs, ccs, ncols, op);
                }
                i0 += mr;
            }
        }
    }
}

// uint64_t: exact __int128 accumulation, k-blocked; strides for everything.
void gemm_u64_driver(const ModCtx<uint64_t>& ctx,
                     const uint64_t* A, size_t ars, size_t acs,
                     const uint64_t* B, size_t brs, size_t bcs,
                     uint64_t* C, size_t crs, size_t ccs,
                     uint32_t m, uint32_t n, uint32_t k, GemmOp op) {
    const uint32_t kc = static_cast<uint32_t>(std::min<uint64_t>(KC_MAX, ctx.k_max));
    for (uint32_t k0 = 0; k0 < k; k0 += kc) {
        const uint32_t kb = std::min(kc, k - k0);
        ENGINE_CHECK_MSG(kb <= ctx.k_max, "gemm: k-block exceeds the delayed-accumulation bound");
        for (uint32_t i = 0; i < m; ++i) {
            const uint64_t* a = A + i * ars + static_cast<size_t>(k0) * acs;
            for (uint32_t j = 0; j < n; ++j) {
                const uint64_t* b = B + static_cast<size_t>(k0) * brs + j * bcs;
                u128 acc = 0;
                for (uint32_t kk = 0; kk < kb; ++kk) acc += static_cast<u128>(a[kk * acs]) * b[kk * brs];
                combine_cell(ctx, C[i * crs + j * ccs], ctx.reduce(acc), op);
            }
        }
    }
}

// (slot-11 WP-18a) u128: exact u256 accumulation, k-blocked at min(k_max,
// 256); strides for everything.  The Sec. 5 bound: kb products of canonical
// residues sum to <= kb (p-1)^2 <= p*2^128 - 1, REDC's precondition.
void gemm_u128_driver(const ModCtx<u128>& ctx,
                      const u128* A, size_t ars, size_t acs,
                      const u128* B, size_t brs, size_t bcs,
                      u128* C, size_t crs, size_t ccs,
                      uint32_t m, uint32_t n, uint32_t k, GemmOp op) {
    const uint32_t kc = static_cast<uint32_t>(std::min<uint64_t>(KC_MAX, ctx.k_max));
    for (uint32_t k0 = 0; k0 < k; k0 += kc) {
        const uint32_t kb = std::min(kc, k - k0);
        ENGINE_CHECK_MSG(kb <= ctx.k_max, "gemm: k-block exceeds the delayed-accumulation bound");
        for (uint32_t i = 0; i < m; ++i) {
            const u128* a = A + i * ars + static_cast<size_t>(k0) * acs;
            for (uint32_t j = 0; j < n; ++j) {
                const u128* b = B + static_cast<size_t>(k0) * brs + j * bcs;
                u256 acc;
                for (uint32_t kk = 0; kk < kb; ++kk) acc += detail::mul_wide128(a[kk * acs], b[kk * brs]);
                combine_cell(ctx, C[i * crs + j * ccs], ctx.reduce(acc), op);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// (slot-23 PERF-P5) The VNNI route of gemm<uint32_t> for p <= 128
// (ENGINE_API.md Sec. 6.1).
//
//   Canonical residues are <= 127, so each fits a byte that reads the same as
//   u8 and as s8; the two operand roles of vpdpbusd are interchangeable.
//   Packed A block (MC x KC): row panels of 8 rows; panel ip, k-quad q holds
//     8 dwords, dword r = the bytes A(8ip+r, 4q .. 4q+3) (zero past m / k).
//   Packed B block (KC x NC): column panels of 32 columns; panel jp, k-quad q
//     holds 128 bytes, bytes 4c .. 4c+3 = B(4q .. 4q+3, 32jp+c).
//   Microkernel 8 x 32: vpdpbusd(acc, B quad (u8), broadcast A dword (s8))
//   adds the exact sum of 4 products to each i32 lane. Over one k-block of
//   kb <= VNNI_KC terms a lane is <= 1024 * 127^2 < 2^24: exact. The block's
//   sums are reduced to canonical residues (an FP64 quotient, corrected) and
//   combined into C with the legacy mod_add / mod_sub, exactly as the u32
//   driver combines its k-blocks. Every C cell is therefore the canonical
//   residue of the same exact integer on both routes: bit-identical by
//   construction (padding contributes 0 * 0).
// ---------------------------------------------------------------------------
#if ENGINE_HAVE_AVX512_TARGET
constexpr uint32_t VNNI_MR = 8, VNNI_NR = 32;
constexpr uint32_t VNNI_MC = 128;    // rows per packed A block (128 x 1024 B = 128 KB, L2)
constexpr uint32_t VNNI_NC = 2048;   // columns per packed B block (one 32-column panel = 32 KB)
static_assert(VNNI_KC % 4 == 0, "VNNI k-block must be a multiple of 4");
static_assert(static_cast<uint64_t>(VNNI_KC) * 127u * 127u < (uint64_t(1) << 31), "VNNI k-block overflows i32");

struct VnniScratch { std::vector<int32_t> a; std::vector<int8_t> b; };
VnniScratch& vnni_scratch() { static thread_local VnniScratch s; return s; }

// Pack A(i0 .. i0+mc, k0 .. k0+kb) into 8-row panels; returns the OR of the entries.
uint32_t vnni_pack_a(const uint32_t* A, size_t ars, size_t acs, uint32_t i0, uint32_t mc,
                     uint32_t k0, uint32_t kb, std::vector<int32_t>& buf) {
    const uint32_t kq = (kb + 3) / 4, np = (mc + VNNI_MR - 1) / VNNI_MR;
    const size_t words = static_cast<size_t>(np) * kq * VNNI_MR;
    if (buf.size() < words) buf.resize(words);
    std::memset(buf.data(), 0, words * sizeof(int32_t));
    uint8_t* o = reinterpret_cast<uint8_t*>(buf.data());
    const uint32_t* base = A + i0 * ars + static_cast<size_t>(k0) * acs;
    uint32_t any = 0;
    if (acs <= ars) {   // rows contiguous (or nearly): walk each row
        for (uint32_t r = 0; r < mc; ++r) {
            const uint32_t* a = base + r * ars;
            uint8_t* d = o + (static_cast<size_t>(r / VNNI_MR) * kq * VNNI_MR + (r % VNNI_MR)) * 4;
            for (uint32_t kk = 0; kk < kb; ++kk) {
                const uint32_t v = a[kk * acs];
                any |= v;
                d[(kk >> 2) * (VNNI_MR * 4) + (kk & 3)] = static_cast<uint8_t>(v);
            }
        }
    } else {            // columns contiguous: walk each column
        for (uint32_t kk = 0; kk < kb; ++kk) {
            const uint32_t* a = base + static_cast<size_t>(kk) * acs;
            uint8_t* d = o + (kk >> 2) * (VNNI_MR * 4) + (kk & 3);
            for (uint32_t r = 0; r < mc; ++r) {
                const uint32_t v = a[r * ars];
                any |= v;
                d[(static_cast<size_t>(r / VNNI_MR) * kq * VNNI_MR + (r % VNNI_MR)) * 4] = static_cast<uint8_t>(v);
            }
        }
    }
    return any;
}

// Pack B(k0 .. k0+kb, j0 .. j0+nc) into 32-column panels; returns the OR of the entries.
uint32_t vnni_pack_b(const uint32_t* B, size_t brs, size_t bcs, uint32_t k0, uint32_t kb,
                     uint32_t j0, uint32_t nc, std::vector<int8_t>& buf) {
    const uint32_t kq = (kb + 3) / 4, np = (nc + VNNI_NR - 1) / VNNI_NR;
    const size_t panel = static_cast<size_t>(kq) * VNNI_NR * 4;
    const size_t bytes = static_cast<size_t>(np) * panel;
    if (buf.size() < bytes) buf.resize(bytes);
    std::memset(buf.data(), 0, bytes);
    int8_t* o = buf.data();
    const uint32_t* base = B + static_cast<size_t>(k0) * brs + j0 * bcs;
    uint32_t any = 0;
    if (bcs <= brs) {   // rows contiguous: walk each row of the block
        for (uint32_t kk = 0; kk < kb; ++kk) {
            const uint32_t* b = base + static_cast<size_t>(kk) * brs;
            int8_t* d = o + (kk >> 2) * (VNNI_NR * 4) + (kk & 3);
            for (uint32_t j = 0; j < nc; ++j) {
                const uint32_t v = b[j * bcs];
                any |= v;
                d[(j / VNNI_NR) * panel + (j % VNNI_NR) * 4] = static_cast<int8_t>(v);
            }
        }
    } else {            // columns contiguous: walk each column of the block
        for (uint32_t j = 0; j < nc; ++j) {
            const uint32_t* b = base + j * bcs;
            int8_t* d = o + (j / VNNI_NR) * panel + (j % VNNI_NR) * 4;
            for (uint32_t kk = 0; kk < kb; ++kk) {
                const uint32_t v = b[kk * brs];
                any |= v;
                d[(kk >> 2) * (VNNI_NR * 4) + (kk & 3)] = static_cast<int8_t>(v);
            }
        }
    }
    return any;
}

// x mod p for 8 i32 lanes with 0 <= x < 2^31 and 2 <= p <= 128: q = trunc(fl(x * fl(1/p)));
// x and q*p are exact in FP64 / i32 and the quotient is off by at most one, so one masked
// "+p if negative" and one masked "-p if >= p" give the canonical residue.
// (slot-24 G4) Also used by the two-limb route, 128 < p <= 2^14, with 0 <= x <= 2130576384: the computed
// quotient is within x * 2^-52 * (1 + 2^-52) < 2^-20 of x/p, so trunc is off by at most one; q * p <= x + p
// <= 2130592768 < 2^31, so mullo / sub stay exact in i32.
ENGINE_VNNI_FN inline __m256i vnni_reduce8(__m256i x, __m512d inv_p, __m256i p32) {
    const __m256i q = _mm512_cvttpd_epi32(_mm512_mul_pd(_mm512_cvtepi32_pd(x), inv_p));
    __m256i r = _mm256_sub_epi32(x, _mm256_mullo_epi32(q, p32));
    r = _mm256_mask_add_epi32(r, _mm256_cmplt_epi32_mask(r, _mm256_setzero_si256()), r, p32);
    r = _mm256_mask_sub_epi32(r, _mm256_cmpge_epi32_mask(r, p32), r, p32);
    return r;
}

// (slot-24 G4: factored out of vnni_micro, unchanged) reduce the 8 x 32 tile of exact non-negative i32 lane sums
// (< 2^31) to canonical residues and combine them into C (rows x ncols of it) with mod_add / mod_sub.
ENGINE_VNNI_FN __attribute__((always_inline)) inline
void vnni_store_tile(const ModCtx<uint32_t>& ctx, const __m512i* acc0, const __m512i* acc1,
                     uint32_t* C, size_t crs, size_t ccs, uint32_t rows, uint32_t ncols, GemmOp op) {
    if (ccs == 1) {
        const __m512d inv_p = _mm512_set1_pd(1.0 / static_cast<double>(ctx.p));
        const __m256i p32   = _mm256_set1_epi32(static_cast<int>(ctx.p));
        const uint32_t mbits = ncols >= 32 ? 0xFFFFFFFFu : ((1u << ncols) - 1u);
        const __mmask8 mq[4] = {static_cast<__mmask8>(mbits), static_cast<__mmask8>(mbits >> 8),
                                static_cast<__mmask8>(mbits >> 16), static_cast<__mmask8>(mbits >> 24)};
#pragma GCC unroll 8
        for (uint32_t r = 0; r < VNNI_MR; ++r) {
            if (r >= rows) break;
            uint32_t* c = C + r * crs;
            const __m256i x[4] = {_mm512_castsi512_si256(acc0[r]), _mm512_extracti64x4_epi64(acc0[r], 1),
                                  _mm512_castsi512_si256(acc1[r]), _mm512_extracti64x4_epi64(acc1[r], 1)};
            for (int h = 0; h < 4; ++h) {
                if (mq[h] == 0) break;
                const __m256i red = vnni_reduce8(x[h], inv_p, p32);
#ifndef NDEBUG
                {   // cross-check the FP-quotient reduction against the scalar remainder
                    alignas(32) int32_t xs[8]; alignas(32) uint32_t rs[8];
                    _mm256_store_si256(reinterpret_cast<__m256i*>(xs), x[h]);
                    _mm256_store_si256(reinterpret_cast<__m256i*>(rs), red);
                    for (int t = 0; t < 8; ++t)
                        ENGINE_DEBUG_CHECK_MSG(rs[t] == static_cast<uint32_t>(xs[t]) % ctx.p, "vnni_reduce8 != remainder");
                }
#endif
                const __m256i cv = _mm256_maskz_loadu_epi32(mq[h], c + 8 * h);
                _mm256_mask_storeu_epi32(c + 8 * h, mq[h], combine_lanes(cv, red, p32, op));
            }
        }
        return;
    }
    alignas(64) int32_t tmp[VNNI_NR];
    for (uint32_t r = 0; r < rows; ++r) {
        _mm512_store_si512(reinterpret_cast<__m512i*>(tmp), acc0[r]);
        _mm512_store_si512(reinterpret_cast<__m512i*>(tmp + 16), acc1[r]);
        uint32_t* c = C + r * crs;
        for (uint32_t j = 0; j < ncols; ++j)
            combine_cell(ctx, c[j * ccs], static_cast<uint32_t>(tmp[j]) % ctx.p, op);
    }
}

ENGINE_VNNI_FN
void vnni_micro(const ModCtx<uint32_t>& ctx, const int32_t* Ap, const int8_t* Bp, uint32_t kq,
                uint32_t* C, size_t crs, size_t ccs, uint32_t rows, uint32_t ncols, GemmOp op) {
    __m512i acc0[VNNI_MR], acc1[VNNI_MR];
#pragma GCC unroll 8
    for (uint32_t r = 0; r < VNNI_MR; ++r) { acc0[r] = _mm512_setzero_si512(); acc1[r] = _mm512_setzero_si512(); }
    for (uint32_t q = 0; q < kq; ++q) {
        const __m512i b0 = _mm512_loadu_si512(Bp + static_cast<size_t>(q) * 128);
        const __m512i b1 = _mm512_loadu_si512(Bp + static_cast<size_t>(q) * 128 + 64);
        const int32_t* a = Ap + static_cast<size_t>(q) * VNNI_MR;
#pragma GCC unroll 8
        for (uint32_t r = 0; r < VNNI_MR; ++r) {
            const __m512i av = _mm512_set1_epi32(a[r]);
            acc0[r] = _mm512_dpbusd_epi32(acc0[r], b0, av);
            acc1[r] = _mm512_dpbusd_epi32(acc1[r], b1, av);
        }
    }
    vnni_store_tile(ctx, acc0, acc1, C, crs, ccs, rows, ncols, op);
}

// C(m x n) op= A(m x k) * B(k x n), op in {Add, Sub}; p <= VNNI_MAX_MODULUS; any strides.
void gemm_u32_vnni(const ModCtx<uint32_t>& ctx,
                   const uint32_t* A, size_t ars, size_t acs,
                   const uint32_t* B, size_t brs, size_t bcs,
                   uint32_t* C, size_t crs, size_t ccs,
                   uint32_t m, uint32_t n, uint32_t k, GemmOp op) {
    ENGINE_CHECK_MSG(ctx.p >= 2 && ctx.p <= VNNI_MAX_MODULUS, "gemm (VNNI route): modulus above VNNI_MAX_MODULUS");
    VnniScratch& s = vnni_scratch();
    for (uint32_t j0 = 0; j0 < n; j0 += VNNI_NC) {
        const uint32_t nc = std::min(VNNI_NC, n - j0);
        const uint32_t npj = (nc + VNNI_NR - 1) / VNNI_NR;
        for (uint32_t k0 = 0; k0 < k; k0 += VNNI_KC) {
            const uint32_t kb = std::min(VNNI_KC, k - k0);
            const uint32_t kq = (kb + 3) / 4;
            // The lane bound that matters is the byte bound 127, which the entry gate below enforces whatever p is
            // (audit R3): kb * 127^2 < 2^31, not kb * (p-1)^2.
            ENGINE_CHECK_MSG(static_cast<uint64_t>(kb) * 127u * 127u < (uint64_t(1) << 31),
                             "gemm (VNNI route): k-block exceeds the i32 accumulation bound");
            const uint32_t orb = vnni_pack_b(B, brs, bcs, k0, kb, j0, nc, s.b);
            ENGINE_CHECK_MSG(orb < 128u, "gemm (VNNI route): B entry >= 128 in a Canonical operand");
            for (uint32_t i0 = 0; i0 < m; i0 += VNNI_MC) {
                const uint32_t mc = std::min(VNNI_MC, m - i0);
                const uint32_t npi = (mc + VNNI_MR - 1) / VNNI_MR;
                const uint32_t ora = vnni_pack_a(A, ars, acs, i0, mc, k0, kb, s.a);
                ENGINE_CHECK_MSG(ora < 128u, "gemm (VNNI route): A entry >= 128 in a Canonical operand");
                for (uint32_t jp = 0; jp < npj; ++jp) {
                    const uint32_t ncols = std::min(VNNI_NR, nc - jp * VNNI_NR);
                    const int8_t* bp = s.b.data() + static_cast<size_t>(jp) * kq * VNNI_NR * 4;
                    for (uint32_t ip = 0; ip < npi; ++ip) {
                        const uint32_t rows = std::min(VNNI_MR, mc - ip * VNNI_MR);
                        vnni_micro(ctx, s.a.data() + static_cast<size_t>(ip) * kq * VNNI_MR, bp, kq,
                                   C + (i0 + ip * VNNI_MR) * crs + (j0 + jp * VNNI_NR) * ccs, crs, ccs,
                                   rows, ncols, op);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// (slot-24 G4) The two-limb VNNI route of gemm<uint32_t> for
// VNNI_MAX_MODULUS < p <= VNNI2_MAX_MODULUS = 2^14 (ENGINE_API.md Sec. 6.2).
//
//   Entries: 0 <= a, b <= 16383 (canonical residues are <= p - 1 <= 16383;
//   the entry gate refuses an entry >= 2^14 behind a Canonical tag).
//   Limb split on A: a = a0 + 128 a1 with a0 = a & 127, a1 = a >> 7, both in
//   [0, 127]. Partner on B: b~ = (128 b) mod p in [0, p - 1]. Then
//       a b = a0 b + a1 (128 b) == a0 b + a1 b~   (mod p).
//   Packed A block (MC2 x KC2): row panels of 8 rows; panel ip, term kk holds
//     8 dwords, dword r = (a0 | a1 << 16) of A(8ip + r, kk) (zero past m / k).
//   Packed B block (KC2 x NC2): column panels of 32 columns; panel jp, term kk
//     holds 32 dwords, dword c = (b | b~ << 16) of B(kk, 32jp + c).
//   Microkernel 8 x 32: vpdpwssd(acc, B dwords, broadcast A dword) adds
//   b a0 + b~ a1 (two s16 x s16 products; all words are in [0, 16383], so
//   their s16 and u16 readings coincide) to each i32 lane, exactly.
//
//   No-overflow bound (per lane, per k-block of kb <= VNNI2_KC = 512 terms):
//     sum <= kb * (127 * 16383 + 127 * 16383) = kb * 4161282
//         <= 512 * 4161282 = 2130576384 < 2^31 - 1 = 2147483647,
//   so the non-saturating vpdpwssd never wraps and the lane holds the exact
//   non-negative integer S = sum_kk (a0 b + a1 b~) == sum_kk a b (mod p).
//   Reduction cadence: once per k-block (every 512 terms), by vnni_reduce8
//   (exact for 0 <= x <= 2130576384, p <= 2^14: see its comment), then
//   combined into C with mod_add / mod_sub exactly as the u32 driver combines
//   its k-blocks. Each block's residue equals the residue of the block's exact
//   sum of products, so every C cell equals the u32 route's: bit-identical by
//   construction (padding contributes 0 * 0).
// ---------------------------------------------------------------------------
constexpr uint32_t VNNI2_MC = 128;    // rows per packed A block (128 x 512 x 4 B = 256 KB, L2)
constexpr uint32_t VNNI2_NC = 1024;   // columns per packed B block (one 32-column panel = 64 KB; block 2 MB)
constexpr uint32_t VNNI2_ENTRY_LIMIT = 16384;   // entries must be < 2^14
static_assert(VNNI2_MAX_MODULUS == VNNI2_ENTRY_LIMIT, "two-limb route: entries <= p - 1 must be < 2^14");
static_assert(static_cast<uint64_t>(VNNI2_KC) * 2u * 127u * (VNNI2_ENTRY_LIMIT - 1u) < (uint64_t(1) << 31),
              "two-limb VNNI k-block overflows i32");
static_assert(static_cast<uint64_t>(VNNI2_KC) * 2u * 127u * (VNNI2_ENTRY_LIMIT - 1u) + VNNI2_MAX_MODULUS < (uint64_t(1) << 31),
              "two-limb VNNI reduction: q * p may overflow i32");

struct Vnni2Scratch { std::vector<int32_t> a, b; };
Vnni2Scratch& vnni2_scratch() { static thread_local Vnni2Scratch s; return s; }

// Pack A(i0 .. i0+mc, k0 .. k0+kb) into 8-row panels of (a0 | a1 << 16) dwords; returns the OR of the entries.
uint32_t vnni2_pack_a(const uint32_t* A, size_t ars, size_t acs, uint32_t i0, uint32_t mc,
                      uint32_t k0, uint32_t kb, std::vector<int32_t>& buf) {
    const uint32_t np = (mc + VNNI_MR - 1) / VNNI_MR;
    const size_t words = static_cast<size_t>(np) * kb * VNNI_MR;
    if (buf.size() < words) buf.resize(words);
    std::memset(buf.data(), 0, words * sizeof(int32_t));
    uint32_t* o = reinterpret_cast<uint32_t*>(buf.data());
    const uint32_t* base = A + i0 * ars + static_cast<size_t>(k0) * acs;
    uint32_t any = 0;
    auto limbs = [](uint32_t v) { return (v & 127u) | (((v >> 7) & 127u) << 16); };
    if (acs <= ars) {   // rows contiguous (or nearly): walk each row
        for (uint32_t r = 0; r < mc; ++r) {
            const uint32_t* a = base + r * ars;
            uint32_t* d = o + static_cast<size_t>(r / VNNI_MR) * kb * VNNI_MR + (r % VNNI_MR);
            for (uint32_t kk = 0; kk < kb; ++kk) {
                const uint32_t v = a[kk * acs];
                any |= v;
                d[static_cast<size_t>(kk) * VNNI_MR] = limbs(v);
            }
        }
    } else {            // columns contiguous: walk each column
        for (uint32_t kk = 0; kk < kb; ++kk) {
            const uint32_t* a = base + static_cast<size_t>(kk) * acs;
            uint32_t* d = o + static_cast<size_t>(kk) * VNNI_MR;
            for (uint32_t r = 0; r < mc; ++r) {
                const uint32_t v = a[r * ars];
                any |= v;
                d[static_cast<size_t>(r / VNNI_MR) * kb * VNNI_MR + (r % VNNI_MR)] = limbs(v);
            }
        }
    }
    return any;
}

// Pack B(k0 .. k0+kb, j0 .. j0+nc) into 32-column panels of (b | b~ << 16) dwords, b~ = 128 b mod p; returns the
// OR of the entries. b~ is computed exactly: x = 128 (b & 0x3FFF) < 2^21, q = floor(x * floor(2^32 / p) / 2^32) is
// floor(x / p) or one less (the error x * frac(2^32 / p) / 2^32 < 1), so r = x - q p is in [0, 2p): one correction.
uint32_t vnni2_pack_b(const uint32_t* B, size_t brs, size_t bcs, uint32_t k0, uint32_t kb,
                      uint32_t j0, uint32_t nc, uint32_t p, std::vector<int32_t>& buf) {
    const uint32_t np = (nc + VNNI_NR - 1) / VNNI_NR;
    const size_t panel = static_cast<size_t>(kb) * VNNI_NR;
    const size_t words = static_cast<size_t>(np) * panel;
    if (buf.size() < words) buf.resize(words);
    std::memset(buf.data(), 0, words * sizeof(int32_t));
    uint32_t* o = reinterpret_cast<uint32_t*>(buf.data());
    const uint32_t* base = B + static_cast<size_t>(k0) * brs + j0 * bcs;
    const uint64_t minv = (uint64_t(1) << 32) / p;
    uint32_t any = 0;
    auto pair = [p, minv](uint32_t v) {
        const uint32_t b = v & 0x3FFFu;
        const uint64_t x = static_cast<uint64_t>(b) << 7;
        uint32_t r = static_cast<uint32_t>(x - ((x * minv) >> 32) * p);
        if (r >= p) r -= p;
        return b | (r << 16);
    };
    if (bcs <= brs) {   // rows contiguous: walk each row of the block
        for (uint32_t kk = 0; kk < kb; ++kk) {
            const uint32_t* b = base + static_cast<size_t>(kk) * brs;
            uint32_t* d = o + static_cast<size_t>(kk) * VNNI_NR;
            for (uint32_t j = 0; j < nc; ++j) {
                const uint32_t v = b[j * bcs];
                any |= v;
                d[(j / VNNI_NR) * panel + (j % VNNI_NR)] = pair(v);
            }
        }
    } else {            // columns contiguous: walk each column of the block
        for (uint32_t j = 0; j < nc; ++j) {
            const uint32_t* b = base + j * bcs;
            uint32_t* d = o + (j / VNNI_NR) * panel + (j % VNNI_NR);
            for (uint32_t kk = 0; kk < kb; ++kk) {
                const uint32_t v = b[kk * brs];
                any |= v;
                d[static_cast<size_t>(kk) * VNNI_NR] = pair(v);
            }
        }
    }
    return any;
}

ENGINE_VNNI_FN
void vnni2_micro(const ModCtx<uint32_t>& ctx, const int32_t* Ap, const int32_t* Bp, uint32_t kb,
                 uint32_t* C, size_t crs, size_t ccs, uint32_t rows, uint32_t ncols, GemmOp op) {
    __m512i acc0[VNNI_MR], acc1[VNNI_MR];
#pragma GCC unroll 8
    for (uint32_t r = 0; r < VNNI_MR; ++r) { acc0[r] = _mm512_setzero_si512(); acc1[r] = _mm512_setzero_si512(); }
    for (uint32_t kk = 0; kk < kb; ++kk) {
        const __m512i b0 = _mm512_loadu_si512(Bp + static_cast<size_t>(kk) * VNNI_NR);
        const __m512i b1 = _mm512_loadu_si512(Bp + static_cast<size_t>(kk) * VNNI_NR + 16);
        const int32_t* a = Ap + static_cast<size_t>(kk) * VNNI_MR;
#pragma GCC unroll 8
        for (uint32_t r = 0; r < VNNI_MR; ++r) {
            const __m512i av = _mm512_set1_epi32(a[r]);
            acc0[r] = _mm512_dpwssd_epi32(acc0[r], b0, av);
            acc1[r] = _mm512_dpwssd_epi32(acc1[r], b1, av);
        }
    }
    vnni_store_tile(ctx, acc0, acc1, C, crs, ccs, rows, ncols, op);
}

// C(m x n) op= A(m x k) * B(k x n), op in {Add, Sub}; VNNI_MAX_MODULUS < p <= VNNI2_MAX_MODULUS; any strides.
void gemm_u32_vnni2(const ModCtx<uint32_t>& ctx,
                    const uint32_t* A, size_t ars, size_t acs,
                    const uint32_t* B, size_t brs, size_t bcs,
                    uint32_t* C, size_t crs, size_t ccs,
                    uint32_t m, uint32_t n, uint32_t k, GemmOp op) {
    ENGINE_CHECK_MSG(ctx.p > VNNI_MAX_MODULUS && ctx.p <= VNNI2_MAX_MODULUS,
                     "gemm (two-limb VNNI route): modulus outside (VNNI_MAX_MODULUS, VNNI2_MAX_MODULUS]");
    Vnni2Scratch& s = vnni2_scratch();
    for (uint32_t j0 = 0; j0 < n; j0 += VNNI2_NC) {
        const uint32_t nc = std::min(VNNI2_NC, n - j0);
        const uint32_t npj = (nc + VNNI_NR - 1) / VNNI_NR;
        for (uint32_t k0 = 0; k0 < k; k0 += VNNI2_KC) {
            const uint32_t kb = std::min(VNNI2_KC, k - k0);
            // The lane bound: limbs <= 127 (by construction) and B words <= 16383 (the entry gate below, whatever p
            // is, and b~ <= p - 1 <= 16383): kb * 2 * 127 * 16383 < 2^31.
            ENGINE_CHECK_MSG(static_cast<uint64_t>(kb) * 2u * 127u * (VNNI2_ENTRY_LIMIT - 1u) < (uint64_t(1) << 31),
                             "gemm (two-limb VNNI route): k-block exceeds the i32 accumulation bound");
            const uint32_t orb = vnni2_pack_b(B, brs, bcs, k0, kb, j0, nc, ctx.p, s.b);
            ENGINE_CHECK_MSG(orb < VNNI2_ENTRY_LIMIT, "gemm (two-limb VNNI route): B entry >= 2^14 in a Canonical operand");
            for (uint32_t i0 = 0; i0 < m; i0 += VNNI2_MC) {
                const uint32_t mc = std::min(VNNI2_MC, m - i0);
                const uint32_t npi = (mc + VNNI_MR - 1) / VNNI_MR;
                const uint32_t ora = vnni2_pack_a(A, ars, acs, i0, mc, k0, kb, s.a);
                ENGINE_CHECK_MSG(ora < VNNI2_ENTRY_LIMIT, "gemm (two-limb VNNI route): A entry >= 2^14 in a Canonical operand");
                for (uint32_t jp = 0; jp < npj; ++jp) {
                    const uint32_t ncols = std::min(VNNI_NR, nc - jp * VNNI_NR);
                    const int32_t* bp = s.b.data() + static_cast<size_t>(jp) * kb * VNNI_NR;
                    for (uint32_t ip = 0; ip < npi; ++ip) {
                        const uint32_t rows = std::min(VNNI_MR, mc - ip * VNNI_MR);
                        vnni2_micro(ctx, s.a.data() + static_cast<size_t>(ip) * kb * VNNI_MR, bp, kb,
                                    C + (i0 + ip * VNNI_MR) * crs + (j0 + jp * VNNI_NR) * ccs, crs, ccs,
                                    rows, ncols, op);
                    }
                }
            }
        }
    }
}
#endif

#if ENGINE_HAVE_AVX512_TARGET
// ---------------------------------------------------------------------------
// (slot-24 AVX2K) the AVX2 pair route of gemm<uint32_t> (ENGINE_API.md Sec. 6.3): for p <= AVX2_PAIR_MAX_MODULUS
// every canonical residue is <= 4095 and fits a non-negative i16, so vpmaddwd multiplies 16 i16 pairs at once and
// adds adjacent products into 8 exact i32 lanes: one instruction = 16 multiply-adds (the u64 microkernel's vpmuludq
// does 4). A is packed into 6-row panels of k-pair dwords (A(i,k) | A(i,k+1) << 16), B into 16-column panels of
// k-pair dwords (B(k,j) | B(k+1,j) << 16); an odd tail k is paired with a zero. Every lane is the exact integer
// sum_kk A(i,kk) B(kk,j) over a k-block of kb terms with (kb / 2) * 2 (p-1)^2 <= 2^31 - 1 (the bound
// avx2_pair_kc(p) is chosen by, ENGINE_CHECKed per block), so no i32 lane can overflow; each block is reduced to
// the canonical residue (FP64 quotient, one correction each way: exact for 0 <= x < 2^31) and combined with
// mod_add / mod_sub. A residue of an exact integer is unique, so the route is bit-identical to the u32
// microkernels by construction. Operand entries >= p behind a Canonical tag are refused (ENGINE_CHECK): the lane
// bound needs them < p.
// ---------------------------------------------------------------------------
constexpr uint32_t AVX2P_MR = 6, AVX2P_NR = 16;
constexpr uint32_t AVX2P_KC_MAX = 512;   // k terms per block at most (B panel 256 pairs x 64 B = 16 KB, L1)
constexpr uint32_t AVX2P_MC = 96;        // rows per packed A block (16 panels x 256 pairs x 24 B = 96 KB, L2)
constexpr uint32_t AVX2P_NC = 512;       // columns per packed B block (32 panels, 512 KB)
static_assert(AVX2P_MC % AVX2P_MR == 0, "A block = whole panels");
static_assert(2ull * (AVX2_PAIR_MAX_MODULUS - 1) * (AVX2_PAIR_MAX_MODULUS - 1) * 64u <= 0x7FFFFFFFull,
              "AVX2 pair route: the largest modulus must admit a k-block of >= 128 terms");
static_assert(AVX2_PAIR_MAX_MODULUS - 1 <= 0x7FFF, "AVX2 pair route: residues must be non-negative i16");

// The k-block (even, >= 128, <= AVX2P_KC_MAX) for p <= AVX2_PAIR_MAX_MODULUS: the largest kb with
// (kb / 2) * 2 (p-1)^2 <= 2^31 - 1.
uint32_t avx2_pair_kc(uint32_t p) {
    const uint64_t pp = 2ull * (p - 1) * (p - 1);
    const uint64_t pairs = pp == 0 ? AVX2P_KC_MAX / 2 : std::min<uint64_t>(AVX2P_KC_MAX / 2, 0x7FFFFFFFull / pp);
    return static_cast<uint32_t>(2 * pairs);
}

struct Avx2PairScratch { std::vector<uint32_t> a, b; };
Avx2PairScratch& avx2_pair_scratch() { static thread_local Avx2PairScratch s; return s; }

// Pack A(i0 .. i0+mc, k0 .. k0+kb) into 6-row panels of k-pair dwords; returns the largest entry. Every dword of the
// packed block is written (rows past mc and the odd k tail are zero), so no clearing pass is needed.
uint32_t avx2_pair_pack_a(const uint32_t* A, size_t ars, size_t acs, uint32_t i0, uint32_t mc,
                          uint32_t k0, uint32_t kb, std::vector<uint32_t>& buf) {
    const uint32_t np = (mc + AVX2P_MR - 1) / AVX2P_MR, kp = (kb + 1) / 2, kfull = kb / 2;
    const size_t panel = static_cast<size_t>(kp) * AVX2P_MR, words = np * panel;
    if (buf.size() < words) buf.resize(words);
    const uint32_t* base = A + i0 * ars + static_cast<size_t>(k0) * acs;
    uint32_t mx = 0;
    for (uint32_t r = 0; r < np * AVX2P_MR; ++r) {
        uint32_t* d = buf.data() + (r / AVX2P_MR) * panel + (r % AVX2P_MR);
        if (r >= mc) {
            for (uint32_t q = 0; q < kp; ++q) d[static_cast<size_t>(q) * AVX2P_MR] = 0;
            continue;
        }
        const uint32_t* a = base + r * ars;
        for (uint32_t q = 0; q < kfull; ++q) {
            const uint32_t v0 = a[(2 * q) * acs], v1 = a[(2 * q + 1) * acs];
            mx = std::max(mx, std::max(v0, v1));
            d[static_cast<size_t>(q) * AVX2P_MR] = (v0 & 0xFFFFu) | (v1 << 16);
        }
        if (kb & 1u) {
            const uint32_t v0 = a[(kb - 1) * acs];
            mx = std::max(mx, v0);
            d[static_cast<size_t>(kfull) * AVX2P_MR] = v0 & 0xFFFFu;
        }
    }
    return mx;
}

// Pack B(k0 .. k0+kb, j0 .. j0+nc) into 16-column panels of k-pair dwords; returns the largest entry. (An entry
// >= 2^16 is caught by the caller's max check before any product is formed, so the truncating shifts are harmless.)
uint32_t avx2_pair_pack_b(const uint32_t* B, size_t brs, size_t bcs, uint32_t k0, uint32_t kb,
                          uint32_t j0, uint32_t nc, std::vector<uint32_t>& buf) {
    const uint32_t np = (nc + AVX2P_NR - 1) / AVX2P_NR, kp = (kb + 1) / 2;
    const size_t panel = static_cast<size_t>(kp) * AVX2P_NR, words = np * panel;
    if (buf.size() < words) buf.resize(words);
    const uint32_t* base = B + static_cast<size_t>(k0) * brs + j0 * bcs;
    uint32_t mx = 0;
    if (bcs <= brs) {   // rows contiguous: two rows of the block at a time, 16 columns per panel row (every dword written)
        for (uint32_t q = 0; q < kp; ++q) {
            const uint32_t* b0 = base + static_cast<size_t>(2 * q) * brs;
            const uint32_t* b1 = (2 * q + 1 < kb) ? b0 + brs : nullptr;
            for (uint32_t jp = 0; jp < np; ++jp) {
                uint32_t* d = buf.data() + jp * panel + static_cast<size_t>(q) * AVX2P_NR;
                const uint32_t jb = jp * AVX2P_NR, cnt = std::min(AVX2P_NR, nc - jb);
                if (b1 != nullptr && bcs == 1 && cnt == AVX2P_NR) {
                    uint32_t m = 0;
                    for (uint32_t t = 0; t < AVX2P_NR; ++t) {
                        const uint32_t v0 = b0[jb + t], v1 = b1[jb + t];
                        m = std::max(m, std::max(v0, v1));
                        d[t] = (v0 & 0xFFFFu) | (v1 << 16);
                    }
                    mx = std::max(mx, m);
                    continue;
                }
                for (uint32_t t = 0; t < AVX2P_NR; ++t) {
                    uint32_t w = 0;
                    if (t < cnt) {
                        const uint32_t v0 = b0[(jb + t) * bcs], v1 = b1 != nullptr ? b1[(jb + t) * bcs] : 0u;
                        mx = std::max(mx, std::max(v0, v1));
                        w = (v0 & 0xFFFFu) | (v1 << 16);
                    }
                    d[t] = w;
                }
            }
        }
    } else {            // columns contiguous: walk each column of the block
        std::memset(buf.data(), 0, words * sizeof(uint32_t));
        for (uint32_t j = 0; j < nc; ++j) {
            const uint32_t* b = base + j * bcs;
            uint32_t* d = buf.data() + (j / AVX2P_NR) * panel + (j % AVX2P_NR);
            for (uint32_t kk = 0; kk < kb; ++kk) {
                const uint32_t v = b[kk * brs];
                mx = std::max(mx, v);
                d[static_cast<size_t>(kk >> 1) * AVX2P_NR] |= (v & 0xFFFFu) << ((kk & 1u) * 16u);
            }
        }
    }
    return mx;
}

// x mod p for 8 i32 lanes with 0 <= x < 2^31, p <= 2^12: q = trunc(fl(x * fl(1/p))) is within x * 2^-52 (1 + 2^-52)
// < 2^-20 of x/p, so off by at most one; q p <= x + p < 2^31 + 2^12 is formed by vpmulld modulo 2^32 and |x - q p|
// <= p, so the i32 difference is exact: one "+p if negative", one "-p if >= p" give the canonical residue.
ENGINE_AVX2_FN inline __m256i avx2_reduce8(__m256i x, __m256d inv_p, __m256i p32, __m256i pm1) {
    const __m128i q0 = _mm256_cvttpd_epi32(_mm256_mul_pd(_mm256_cvtepi32_pd(_mm256_castsi256_si128(x)), inv_p));
    const __m128i q1 = _mm256_cvttpd_epi32(_mm256_mul_pd(_mm256_cvtepi32_pd(_mm256_extracti128_si256(x, 1)), inv_p));
    const __m256i q = _mm256_inserti128_si256(_mm256_castsi128_si256(q0), q1, 1);
    __m256i r = _mm256_sub_epi32(x, _mm256_mullo_epi32(q, p32));
    r = _mm256_add_epi32(r, _mm256_and_si256(_mm256_cmpgt_epi32(_mm256_setzero_si256(), r), p32));
    r = _mm256_sub_epi32(r, _mm256_and_si256(_mm256_cmpgt_epi32(r, pm1), p32));
    return r;
}

// c op= r on 8 canonical lanes (p <= 2^12: no lane can wrap, the signed compares are exact).
ENGINE_AVX2_FN inline __m256i avx2_combine8(__m256i c, __m256i r, __m256i p32, __m256i pm1, GemmOp op) {
    if (op == GemmOp::Sub) {
        const __m256i s = _mm256_sub_epi32(c, r);
        return _mm256_add_epi32(s, _mm256_and_si256(_mm256_cmpgt_epi32(_mm256_setzero_si256(), s), p32));
    }
    const __m256i s = _mm256_add_epi32(c, r);
    return _mm256_sub_epi32(s, _mm256_and_si256(_mm256_cmpgt_epi32(s, pm1), p32));
}

ENGINE_AVX2_FN
void avx2_pair_micro(const ModCtx<uint32_t>& ctx, const uint32_t* Ap, const uint32_t* Bp, uint32_t kp,
                     uint32_t* C, size_t crs, size_t ccs, uint32_t rows, uint32_t ncols, GemmOp op) {
    __m256i acc0[AVX2P_MR], acc1[AVX2P_MR];
#pragma GCC unroll 6
    for (uint32_t r = 0; r < AVX2P_MR; ++r) { acc0[r] = _mm256_setzero_si256(); acc1[r] = _mm256_setzero_si256(); }
    for (uint32_t q = 0; q < kp; ++q) {
        const __m256i b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(Bp + static_cast<size_t>(q) * AVX2P_NR));
        const __m256i b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(Bp + static_cast<size_t>(q) * AVX2P_NR + 8));
        const uint32_t* a = Ap + static_cast<size_t>(q) * AVX2P_MR;
#pragma GCC unroll 6
        for (uint32_t r = 0; r < AVX2P_MR; ++r) {
            const __m256i av = _mm256_set1_epi32(static_cast<int>(a[r]));
            acc0[r] = _mm256_add_epi32(acc0[r], _mm256_madd_epi16(av, b0));
            acc1[r] = _mm256_add_epi32(acc1[r], _mm256_madd_epi16(av, b1));
        }
    }
    if (ccs == 1) {
        const __m256d inv_p = _mm256_set1_pd(1.0 / static_cast<double>(ctx.p));
        const __m256i p32 = _mm256_set1_epi32(static_cast<int>(ctx.p));
        const __m256i pm1 = _mm256_set1_epi32(static_cast<int>(ctx.p - 1));
        const __m256i lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        const __m256i m0 = _mm256_cmpgt_epi32(_mm256_set1_epi32(static_cast<int>(ncols)), lane);
        const __m256i m1 = _mm256_cmpgt_epi32(_mm256_set1_epi32(static_cast<int>(ncols) - 8), lane);
#pragma GCC unroll 6
        for (uint32_t r = 0; r < AVX2P_MR; ++r) {
            if (r >= rows) break;
            int* c = reinterpret_cast<int*>(C + r * crs);
            const __m256i x[2] = {acc0[r], acc1[r]};
            const __m256i mk[2] = {m0, m1};
            for (int h = 0; h < 2; ++h) {
                if (h == 1 && ncols <= 8) break;
                const __m256i red = avx2_reduce8(x[h], inv_p, p32, pm1);
#ifndef NDEBUG
                {   // cross-check the FP-quotient reduction against the scalar remainder
                    alignas(32) int32_t xs[8]; alignas(32) uint32_t rs[8];
                    _mm256_store_si256(reinterpret_cast<__m256i*>(xs), x[h]);
                    _mm256_store_si256(reinterpret_cast<__m256i*>(rs), red);
                    for (int t = 0; t < 8; ++t)
                        ENGINE_DEBUG_CHECK_MSG(rs[t] == static_cast<uint32_t>(xs[t]) % ctx.p, "avx2_reduce8 != remainder");
                }
#endif
                if (ncols - 8u * h >= 8) {
                    const __m256i cv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c + 8 * h));
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + 8 * h), avx2_combine8(cv, red, p32, pm1, op));
                } else {
                    const __m256i cv = _mm256_maskload_epi32(c + 8 * h, mk[h]);
                    _mm256_maskstore_epi32(c + 8 * h, mk[h], avx2_combine8(cv, red, p32, pm1, op));
                }
            }
        }
        return;
    }
    alignas(32) int32_t tmp[AVX2P_NR];
    for (uint32_t r = 0; r < rows; ++r) {
        _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), acc0[r]);
        _mm256_store_si256(reinterpret_cast<__m256i*>(tmp + 8), acc1[r]);
        uint32_t* c = C + r * crs;
        for (uint32_t j = 0; j < ncols; ++j)
            combine_cell(ctx, c[j * ccs], static_cast<uint32_t>(tmp[j]) % ctx.p, op);
    }
}

// C(m x n) op= A(m x k) * B(k x n), op in {Add, Sub}; 2 <= p <= AVX2_PAIR_MAX_MODULUS; any strides.
void gemm_u32_avx2_pair(const ModCtx<uint32_t>& ctx,
                        const uint32_t* A, size_t ars, size_t acs,
                        const uint32_t* B, size_t brs, size_t bcs,
                        uint32_t* C, size_t crs, size_t ccs,
                        uint32_t m, uint32_t n, uint32_t k, GemmOp op) {
    ENGINE_CHECK_MSG(ctx.p >= 2 && ctx.p <= AVX2_PAIR_MAX_MODULUS, "gemm (AVX2 pair route): modulus above AVX2_PAIR_MAX_MODULUS");
    const uint32_t kc = avx2_pair_kc(ctx.p);
    Avx2PairScratch& s = avx2_pair_scratch();
    for (uint32_t j0 = 0; j0 < n; j0 += AVX2P_NC) {
        const uint32_t nc = std::min(AVX2P_NC, n - j0);
        const uint32_t npj = (nc + AVX2P_NR - 1) / AVX2P_NR;
        for (uint32_t k0 = 0; k0 < k; k0 += kc) {
            const uint32_t kb = std::min(kc, k - k0), kp = (kb + 1) / 2;
            ENGINE_CHECK_MSG(static_cast<uint64_t>(kp) * 2u * (ctx.p - 1) * (ctx.p - 1) <= 0x7FFFFFFFull,
                             "gemm (AVX2 pair route): k-block exceeds the i32 accumulation bound");
            const uint32_t mb = avx2_pair_pack_b(B, brs, bcs, k0, kb, j0, nc, s.b);
            ENGINE_CHECK_MSG(mb < ctx.p, "gemm (AVX2 pair route): B entry >= p in a Canonical operand");
            for (uint32_t i0 = 0; i0 < m; i0 += AVX2P_MC) {
                const uint32_t mc = std::min(AVX2P_MC, m - i0);
                const uint32_t npi = (mc + AVX2P_MR - 1) / AVX2P_MR;
                const uint32_t ma = avx2_pair_pack_a(A, ars, acs, i0, mc, k0, kb, s.a);
                ENGINE_CHECK_MSG(ma < ctx.p, "gemm (AVX2 pair route): A entry >= p in a Canonical operand");
                for (uint32_t jp = 0; jp < npj; ++jp) {
                    const uint32_t ncols = std::min(AVX2P_NR, nc - jp * AVX2P_NR);
                    const uint32_t* bp = s.b.data() + static_cast<size_t>(jp) * kp * AVX2P_NR;
                    for (uint32_t ip = 0; ip < npi; ++ip) {
                        const uint32_t rows = std::min(AVX2P_MR, mc - ip * AVX2P_MR);
                        avx2_pair_micro(ctx, s.a.data() + static_cast<size_t>(ip) * kp * AVX2P_MR, bp, kp,
                                        C + (i0 + ip * AVX2P_MR) * crs + (j0 + jp * AVX2P_NR) * ccs, crs, ccs,
                                        rows, ncols, op);
                    }
                }
            }
        }
    }
}
#endif

template<class Elt>
void check_gemm_shapes(const ModCtx<Elt>& ctx, DenseView<const Elt> A, DenseView<const Elt> B,
                       DenseView<Elt> C, GemmOp op) {
    ENGINE_CHECK_MSG(A.cols() == B.rows() && C.rows() == A.rows() && C.cols() == B.cols(),
                     "gemm: shape mismatch");
    ENGINE_CHECK_MSG(A.invariant() == Invariant::Canonical && B.invariant() == Invariant::Canonical,
                     "gemm: operands must be Canonical");
    ENGINE_CHECK_MSG(C.invariant() == Invariant::Canonical, "gemm: C must be a Canonical buffer");
    ENGINE_CHECK_MSG(!windows_overlap(C, A) && !windows_overlap(C, B), "gemm: C aliases an operand");
    debug_check_canonical(ctx, A, "gemm: A entry >= p");
    debug_check_canonical(ctx, B, "gemm: B entry >= p");
    if (op != GemmOp::Set) debug_check_canonical(ctx, DenseView<const Elt>(C), "gemm: C entry >= p");
}

} // namespace

// ---------------------------------------------------------------------------
// gemm / gemm_nt / gemv / dot_delayed
// ---------------------------------------------------------------------------

template<>
void gemm<uint32_t>(const ModCtx<uint32_t>& ctx, DenseView<const uint32_t> A,
                    DenseView<const uint32_t> B, DenseView<uint32_t> C, GemmOp op) {
    check_gemm_shapes(ctx, A, B, C, op);
    if (op == GemmOp::Set) { zero_view(C); op = GemmOp::Add; }
    if (C.empty() || A.cols() == 0) return;

    const auto sa = A.strides();
    const auto sc = C.strides();
    // The pre-P5 u32 route, into the C window (cbase, crs, ccs).
    auto u32_route = [&](uint32_t* cbase, size_t crs, size_t ccs) {
        const uint32_t* bp;
        size_t brs;
        if (B.col_stride() == 1) {
            const auto sb = B.strides();
            bp = sb.base; brs = sb.rs;
        } else {
            bp = pack_rowmajor<uint32_t>(B, pack_scratch<uint32_t>());
            brs = B.cols();
        }
        gemm_u32_driver(ctx, sa.base, sa.rs, sa.cs, bp, brs, cbase, crs, ccs,
                        A.rows(), B.cols(), A.cols(), op);
    };
#if ENGINE_HAVE_AVX512_TARGET
    if (vnni_route_taken(ctx.p, A.rows(), B.cols(), A.cols())) {
        const uint32_t m = A.rows(), n = B.cols();
        // ENGINE_VNNI_SHADOW=1: run the u32 route on a copy of C and compare (read-only; throws on a difference).
        std::vector<uint32_t> shadow;
        if (vnni_shadow_env()) {
            shadow.resize(static_cast<size_t>(m) * n);
            for (uint32_t i = 0; i < m; ++i)
                for (uint32_t j = 0; j < n; ++j) shadow[static_cast<size_t>(i) * n + j] = sc.base[i * sc.rs + j * sc.cs];
            u32_route(shadow.data(), n, 1);
        }
        const auto sb = B.strides();
        if (vnni_limbs_for(ctx.p) == 1) {
            gemm_u32_vnni(ctx, sa.base, sa.rs, sa.cs, sb.base, sb.rs, sb.cs, sc.base, sc.rs, sc.cs,
                          m, n, A.cols(), op);
        } else {   // (slot-24 G4) 128 < p <= 2^14
            gemm_u32_vnni2(ctx, sa.base, sa.rs, sa.cs, sb.base, sb.rs, sb.cs, sc.base, sc.rs, sc.cs,
                           m, n, A.cols(), op);
            g_vnni2_routed.fetch_add(1, std::memory_order_relaxed);
        }
        g_vnni_routed.fetch_add(1, std::memory_order_relaxed);
        if (!shadow.empty()) {
            bool same = true;
            for (uint32_t i = 0; i < m && same; ++i)
                for (uint32_t j = 0; j < n && same; ++j)
                    same = shadow[static_cast<size_t>(i) * n + j] == sc.base[i * sc.rs + j * sc.cs];
            g_shadow_calls.fetch_add(1, std::memory_order_relaxed);
            g_shadow_cells.fetch_add(shadow.size(), std::memory_order_relaxed);
            if (!same) g_shadow_differ.fetch_add(1, std::memory_order_relaxed);
            ENGINE_CHECK_MSG(same, "ENGINE_VNNI_SHADOW: the VNNI route differs from the u32 route");
        }
        return;
    }
    // (slot-24 AVX2K) the AVX2 pair route (ENGINE_API.md Sec. 6.3), p <= AVX2_PAIR_MAX_MODULUS on the AVX2 path.
    if (avx2_pair_route_taken(ctx.p, A.rows(), B.cols(), A.cols())) {
        const auto sb = B.strides();
        gemm_u32_avx2_pair(ctx, sa.base, sa.rs, sa.cs, sb.base, sb.rs, sb.cs, sc.base, sc.rs, sc.cs,
                           A.rows(), B.cols(), A.cols(), op);
        g_avx2_pair_routed.fetch_add(1, std::memory_order_relaxed);
        return;
    }
#endif
    u32_route(sc.base, sc.rs, sc.cs);
}

template<>
void gemm<uint64_t>(const ModCtx<uint64_t>& ctx, DenseView<const uint64_t> A,
                    DenseView<const uint64_t> B, DenseView<uint64_t> C, GemmOp op) {
    check_gemm_shapes(ctx, A, B, C, op);
    if (op == GemmOp::Set) { zero_view(C); op = GemmOp::Add; }
    if (C.empty() || A.cols() == 0) return;
    const auto sa = A.strides();
    const auto sb = B.strides();
    const auto sc = C.strides();
    gemm_u64_driver(ctx, sa.base, sa.rs, sa.cs, sb.base, sb.rs, sb.cs, sc.base, sc.rs, sc.cs,
                    A.rows(), B.cols(), A.cols(), op);
}

template<>
void gemm<u128>(const ModCtx<u128>& ctx, DenseView<const u128> A,
                DenseView<const u128> B, DenseView<u128> C, GemmOp op) {
    check_gemm_shapes(ctx, A, B, C, op);
    if (op == GemmOp::Set) { zero_view(C); op = GemmOp::Add; }
    if (C.empty() || A.cols() == 0) return;
    const auto sa = A.strides();
    const auto sb = B.strides();
    const auto sc = C.strides();
    gemm_u128_driver(ctx, sa.base, sa.rs, sa.cs, sb.base, sb.rs, sb.cs, sc.base, sc.rs, sc.cs,
                     A.rows(), B.cols(), A.cols(), op);
}

template<>
void gemv<u128>(const ModCtx<u128>& ctx, DenseView<const u128> A, DenseView<const u128> x,
                DenseView<u128> y, GemmOp op) {
    ENGINE_CHECK_MSG(x.cols() == 1 && y.cols() == 1, "gemv: x and y must be column views");
    gemm<u128>(ctx, A, x, y, op);
}

template<>
u128 dot_delayed<u128>(const ModCtx<u128>& ctx, const u128* a, const u128* b, size_t k) {
    ENGINE_CHECK_MSG(k <= ctx.k_max, "dot_delayed: k exceeds the delayed-accumulation bound k_max");
    u256 s;
    for (size_t i = 0; i < k; ++i) s += detail::mul_wide128(a[i], b[i]);
    return ctx.reduce(s);
}

template<class Elt>
void gemm_nt(const ModCtx<Elt>& ctx, DenseView<const Elt> A, DenseView<const Elt> B,
             DenseView<Elt> C, GemmOp op) {
    gemm<Elt>(ctx, A, B.transposed_view(), C, op);
}

template<class Elt>
Elt dot_delayed(const ModCtx<Elt>& ctx, const Elt* a, const Elt* b, size_t k) {
    ENGINE_CHECK_MSG(k <= ctx.k_max, "dot_delayed: k exceeds the delayed-accumulation bound k_max");
    using Accum = typename ModCtx<Elt>::Accum;
    Accum s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    size_t i = 0;
    for (; i + 4 <= k; i += 4) {
        s0 += static_cast<Accum>(a[i]) * b[i];
        s1 += static_cast<Accum>(a[i + 1]) * b[i + 1];
        s2 += static_cast<Accum>(a[i + 2]) * b[i + 2];
        s3 += static_cast<Accum>(a[i + 3]) * b[i + 3];
    }
    for (; i < k; ++i) s0 += static_cast<Accum>(a[i]) * b[i];
    return ctx.reduce((s0 + s1) + (s2 + s3));
}

template<class Elt>
void gemv(const ModCtx<Elt>& ctx, DenseView<const Elt> A, DenseView<const Elt> x,
          DenseView<Elt> y, GemmOp op) {
    ENGINE_CHECK_MSG(x.cols() == 1 && y.cols() == 1, "gemv: x and y must be column views");
    check_gemm_shapes(ctx, A, x, y, op);
    if (op == GemmOp::Set) { zero_view(y); op = GemmOp::Add; }
    const uint32_t m = A.rows(), k = A.cols();
    if (m == 0 || k == 0) return;
    using Accum = typename ModCtx<Elt>::Accum;
    const auto sa = A.strides();
    const auto sx = x.strides();
    const auto sy = y.strides();
    const uint32_t kc = static_cast<uint32_t>(std::min<uint64_t>(KC_MAX, ctx.k_max));
    for (uint32_t k0 = 0; k0 < k; k0 += kc) {
        const uint32_t kb = std::min(kc, k - k0);
        ENGINE_CHECK_MSG(kb <= ctx.k_max, "gemv: k-block exceeds the delayed-accumulation bound");
        const Elt* xp = sx.base + static_cast<size_t>(k0) * sx.rs;
        for (uint32_t i = 0; i < m; ++i) {
            const Elt* a = sa.base + i * sa.rs + static_cast<size_t>(k0) * sa.cs;
            Accum s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            uint32_t t = 0;
            for (; t + 4 <= kb; t += 4) {
                s0 += static_cast<Accum>(a[(t + 0) * sa.cs]) * xp[(t + 0) * sx.rs];
                s1 += static_cast<Accum>(a[(t + 1) * sa.cs]) * xp[(t + 1) * sx.rs];
                s2 += static_cast<Accum>(a[(t + 2) * sa.cs]) * xp[(t + 2) * sx.rs];
                s3 += static_cast<Accum>(a[(t + 3) * sa.cs]) * xp[(t + 3) * sx.rs];
            }
            for (; t < kb; ++t) s0 += static_cast<Accum>(a[t * sa.cs]) * xp[t * sx.rs];
            combine_cell(ctx, sy.base[i * sy.rs], ctx.reduce((s0 + s1) + (s2 + s3)), op);
        }
    }
}

// ---------------------------------------------------------------------------
// axpy / scal
// ---------------------------------------------------------------------------

template<class Elt>
void axpy(const ModCtx<Elt>& ctx, Elt alpha, DenseView<const Elt> x, DenseView<Elt> y, bool subtract) {
    ENGINE_CHECK_MSG(x.rows() == y.rows() && x.cols() == y.cols(), "axpy: shape mismatch");
    ENGINE_CHECK_MSG(x.invariant() == Invariant::Canonical && y.invariant() == Invariant::Canonical,
                     "axpy: operands must be Canonical");
    ENGINE_CHECK_MSG(alpha < ctx.p, "axpy: alpha must be canonical");
    const auto sx = x.strides();
    const auto sy = y.strides();
    const uint32_t m = x.rows(), n = x.cols();
    if constexpr (std::is_same<Elt, uint32_t>::value) {
        if (ctx.shoup_available()) {
            const uint32_t ash = ctx.shoup_precompute(alpha);
            for (uint32_t i = 0; i < m; ++i) {
                const Elt* xr = sx.base + i * sx.rs;
                Elt* yr = sy.base + i * sy.rs;
                if (subtract) for (uint32_t j = 0; j < n; ++j) yr[j * sy.cs] = ctx.sub(yr[j * sy.cs], ctx.shoup_mul(xr[j * sx.cs], alpha, ash));
                else          for (uint32_t j = 0; j < n; ++j) yr[j * sy.cs] = ctx.add(yr[j * sy.cs], ctx.shoup_mul(xr[j * sx.cs], alpha, ash));
            }
            return;
        }
    }
    if constexpr (std::is_same<Elt, u128>::value) {   // (slot-11 WP-18a) the Montgomery-128 pair
        const u128 ap = ctx.mul_prep(alpha);
        for (uint32_t i = 0; i < m; ++i) {
            const Elt* xr = sx.base + i * sx.rs;
            Elt* yr = sy.base + i * sy.rs;
            if (subtract) for (uint32_t j = 0; j < n; ++j) yr[j * sy.cs] = ctx.sub(yr[j * sy.cs], ctx.mul_prepped(xr[j * sx.cs], ap));
            else          for (uint32_t j = 0; j < n; ++j) yr[j * sy.cs] = ctx.add(yr[j * sy.cs], ctx.mul_prepped(xr[j * sx.cs], ap));
        }
        return;
    }
    for (uint32_t i = 0; i < m; ++i) {
        const Elt* xr = sx.base + i * sx.rs;
        Elt* yr = sy.base + i * sy.rs;
        if (subtract) for (uint32_t j = 0; j < n; ++j) yr[j * sy.cs] = ctx.sub(yr[j * sy.cs], ctx.mul(alpha, xr[j * sx.cs]));
        else          for (uint32_t j = 0; j < n; ++j) yr[j * sy.cs] = ctx.add(yr[j * sy.cs], ctx.mul(alpha, xr[j * sx.cs]));
    }
}

template<class Elt>
void scal(const ModCtx<Elt>& ctx, Elt alpha, DenseView<Elt> x) {
    ENGINE_CHECK_MSG(x.invariant() == Invariant::Canonical, "scal: operand must be Canonical");
    ENGINE_CHECK_MSG(alpha < ctx.p, "scal: alpha must be canonical");
    const auto sx = x.strides();
    const uint32_t m = x.rows(), n = x.cols();
    if constexpr (std::is_same<Elt, uint32_t>::value) {
        if (ctx.shoup_available()) {
            const uint32_t ash = ctx.shoup_precompute(alpha);
            for (uint32_t i = 0; i < m; ++i) {
                Elt* xr = sx.base + i * sx.rs;
                for (uint32_t j = 0; j < n; ++j) xr[j * sx.cs] = ctx.shoup_mul(xr[j * sx.cs], alpha, ash);
            }
            return;
        }
    }
    if constexpr (std::is_same<Elt, u128>::value) {   // (slot-11 WP-18a) the Montgomery-128 pair
        const u128 ap = ctx.mul_prep(alpha);
        for (uint32_t i = 0; i < m; ++i) {
            Elt* xr = sx.base + i * sx.rs;
            for (uint32_t j = 0; j < n; ++j) xr[j * sx.cs] = ctx.mul_prepped(xr[j * sx.cs], ap);
        }
        return;
    }
    for (uint32_t i = 0; i < m; ++i) {
        Elt* xr = sx.base + i * sx.rs;
        for (uint32_t j = 0; j < n; ++j) xr[j * sx.cs] = ctx.mul(alpha, xr[j * sx.cs]);
    }
}

// ---------------------------------------------------------------------------
// trmm -- C = T*B, T unit-triangular by structure tag
// ---------------------------------------------------------------------------

template<class Elt>
void trmm(const ModCtx<Elt>& ctx, DenseView<const Elt> T, DenseView<const Elt> B, DenseView<Elt> C) {
    const uint32_t m = T.rows(), n = B.cols();
    ENGINE_CHECK_MSG(T.cols() == m && B.rows() == m && C.rows() == m && C.cols() == n, "trmm: shape mismatch");
    const Structure st = T.structure();
    ENGINE_CHECK_MSG(st == Structure::UpperUnitri || st == Structure::LowerUnitri,
                     "trmm: T must be tagged UpperUnitri or LowerUnitri");
    ENGINE_CHECK_MSG(!windows_overlap(C, B) && !windows_overlap(C, T), "trmm: C aliases an operand");
    ENGINE_CHECK_MSG(B.invariant() == Invariant::Canonical && T.invariant() == Invariant::Canonical &&
                     C.invariant() == Invariant::Canonical, "trmm: operands must be Canonical");
    if (m == 0 || n == 0) return;

    copy<Elt>(B, C);                                   // unit diagonal
    const bool upper = (st == Structure::UpperUnitri);
    constexpr uint32_t RB = 64;                        // row block
    for (uint32_t i0 = 0; i0 < m; i0 += RB) {
        const uint32_t i1 = std::min(m, i0 + RB);
        // rows i in [i0, i1): the part of T's row strictly inside the block
        for (uint32_t i = i0; i < i1; ++i) {
            const uint32_t k0 = upper ? i + 1 : i0;
            const uint32_t k1 = upper ? i1 : i;
            if (k1 > k0) {
                gemm<Elt>(ctx, T.sub(i, 1, k0, k1 - k0), B.rows_range(k0, k1 - k0),
                          C.row(i), GemmOp::Add);
            }
        }
        // the part of the block's rows entirely off the block: one GEMM
        const uint32_t k0 = upper ? i1 : 0;
        const uint32_t k1 = upper ? m : i0;
        if (k1 > k0) {
            gemm<Elt>(ctx, T.sub(i0, i1 - i0, k0, k1 - k0), B.rows_range(k0, k1 - k0),
                      C.rows_range(i0, i1 - i0), GemmOp::Add);
        }
    }
}

// ---------------------------------------------------------------------------
// structured applies
// ---------------------------------------------------------------------------

template<class Elt>
void apply_signed_antidiag(const ModCtx<Elt>& ctx, DenseView<const Elt> M, DenseView<const Elt> B,
                           DenseView<Elt> C, Side side) {
    const uint32_t d = M.rows();
    ENGINE_CHECK_MSG(M.cols() == d, "apply_signed_antidiag: M must be square");
    ENGINE_CHECK_MSG(M.structure() == Structure::SignedAntiDiag, "apply_signed_antidiag: M must be tagged SignedAntiDiag");
    ENGINE_CHECK_MSG(B.rows() == C.rows() && B.cols() == C.cols(), "apply_signed_antidiag: shape mismatch");
    ENGINE_CHECK_MSG(!windows_overlap(C, B), "apply_signed_antidiag: C aliases B");
    const Elt minus_one = ctx.p - 1;
    if (side == Side::Left) {
        ENGINE_CHECK_MSG(B.rows() == d, "apply_signed_antidiag: shape mismatch");
        for (uint32_t i = 0; i < d; ++i) {
            const Elt s = M.at(i, d - 1 - i);
            ENGINE_CHECK_MSG(s == 1 || s == minus_one, "apply_signed_antidiag: anti-diagonal entry not +-1");
            const auto src = B.row(d - 1 - i);
            const auto dst = C.row(i);
            if (s == 1) copy<Elt>(src, dst);
            else for (uint32_t j = 0; j < B.cols(); ++j) dst.at(0, j) = ctx.neg(src.at(0, j));
        }
    } else {
        ENGINE_CHECK_MSG(B.cols() == d, "apply_signed_antidiag: shape mismatch");
        for (uint32_t j = 0; j < d; ++j) {
            const Elt s = M.at(d - 1 - j, j);
            ENGINE_CHECK_MSG(s == 1 || s == minus_one, "apply_signed_antidiag: anti-diagonal entry not +-1");
            const auto src = B.col(d - 1 - j);
            const auto dst = C.col(j);
            if (s == 1) copy<Elt>(src, dst);
            else for (uint32_t i = 0; i < B.rows(); ++i) dst.at(i, 0) = ctx.neg(src.at(i, 0));
        }
    }
}

template<class Elt>
void apply_diagonal(const ModCtx<Elt>& ctx, DenseView<const Elt> D, DenseView<const Elt> B,
                    DenseView<Elt> C, Side side) {
    const uint32_t d = D.rows();
    ENGINE_CHECK_MSG(D.cols() == d, "apply_diagonal: D must be square");
    ENGINE_CHECK_MSG(D.structure() == Structure::Diagonal, "apply_diagonal: D must be tagged Diagonal");
    ENGINE_CHECK_MSG(B.rows() == C.rows() && B.cols() == C.cols(), "apply_diagonal: shape mismatch");
    ENGINE_CHECK_MSG(!windows_overlap(C, B), "apply_diagonal: C aliases B");
    if (side == Side::Left) {
        ENGINE_CHECK_MSG(B.rows() == d, "apply_diagonal: shape mismatch");
        for (uint32_t i = 0; i < d; ++i) {
            const Elt s = D.at(i, i);
            for (uint32_t j = 0; j < B.cols(); ++j) C.at(i, j) = ctx.mul(s, B.at(i, j));
        }
    } else {
        ENGINE_CHECK_MSG(B.cols() == d, "apply_diagonal: shape mismatch");
        for (uint32_t j = 0; j < d; ++j) {
            const Elt s = D.at(j, j);
            for (uint32_t i = 0; i < B.rows(); ++i) C.at(i, j) = ctx.mul(B.at(i, j), s);
        }
    }
}

// ---------------------------------------------------------------------------
// copy / transpose / canonicalise / swap_rows
// ---------------------------------------------------------------------------

template<class Elt>
void copy(DenseView<const Elt> src, DenseView<Elt> dst) {
    ENGINE_CHECK_MSG(src.rows() == dst.rows() && src.cols() == dst.cols(), "copy: shape mismatch");
    const auto s = src.strides();
    const auto t = dst.strides();
    if (windows_overlap(dst, src)) {
        // the only overlap allowed is the identical window: a no-op
        ENGINE_CHECK_MSG(static_cast<const void*>(s.base) == static_cast<const void*>(t.base) &&
                         s.rs == t.rs && s.cs == t.cs, "copy: overlapping distinct windows");
        return;
    }
    const uint32_t m = src.rows(), n = src.cols();
    for (uint32_t i0 = 0; i0 < m; i0 += TILE) {
        const uint32_t i1 = std::min(m, i0 + TILE);
        for (uint32_t j0 = 0; j0 < n; j0 += TILE) {
            const uint32_t j1 = std::min(n, j0 + TILE);
            for (uint32_t i = i0; i < i1; ++i) {
                const Elt* sr = s.base + i * s.rs;
                Elt* dr = t.base + i * t.rs;
                for (uint32_t j = j0; j < j1; ++j) dr[j * t.cs] = sr[j * s.cs];
            }
        }
    }
    if (dst.covers_whole_buffer()) dst.buffer()->invariant = src.invariant();
}

template<class Elt>
void transpose(DenseView<const Elt> src, DenseView<Elt> dst) {
    copy<Elt>(src.transposed_view(), dst);
}

template<class Elt>
void canonicalise(const ModCtx<Elt>& ctx, DenseView<Elt> v) {
    ENGINE_CHECK_MSG(v.invariant() == Invariant::Lazy2p || v.invariant() == Invariant::Canonical,
                     "canonicalise: view must be Lazy2p (or already Canonical)");
    const auto s = v.strides();
    for (uint32_t i = 0; i < v.rows(); ++i) {
        Elt* r = s.base + i * s.rs;
        for (uint32_t j = 0; j < v.cols(); ++j) r[j * s.cs] = ctx.canonicalise(r[j * s.cs]);
    }
    if (v.covers_whole_buffer()) v.buffer()->invariant = Invariant::Canonical;
}

void canonicalise(const ModCtx<uint32_t>& ctx, DenseView<const uint64_t> acc, DenseView<uint32_t> dst) {
    ENGINE_CHECK_MSG(acc.invariant() == Invariant::Accum64, "canonicalise: source must be Accum64");
    ENGINE_CHECK_MSG(acc.rows() == dst.rows() && acc.cols() == dst.cols(), "canonicalise: shape mismatch");
    const auto s = acc.strides();
    const auto t = dst.strides();
    for (uint32_t i = 0; i < acc.rows(); ++i) {
        const uint64_t* sr = s.base + i * s.rs;
        uint32_t* dr = t.base + i * t.rs;
        for (uint32_t j = 0; j < acc.cols(); ++j) dr[j * t.cs] = ctx.reduce_u64(sr[j * s.cs]);
    }
    if (dst.covers_whole_buffer()) dst.buffer()->invariant = Invariant::Canonical;
}

template<class Elt>
void swap_rows(DenseView<Elt> v, uint32_t r1, uint32_t r2) {
    ENGINE_CHECK(r1 < v.rows() && r2 < v.rows());
    if (r1 == r2) return;
    const auto s = v.strides();
    Elt* a = s.base + r1 * s.rs;
    Elt* b = s.base + r2 * s.rs;
    for (uint32_t j = 0; j < v.cols(); ++j) std::swap(a[j * s.cs], b[j * s.cs]);
}

// ---------------------------------------------------------------------------
// explicit instantiations
// ---------------------------------------------------------------------------
#define ENGINE_KERNELS_INSTANTIATE(Elt)                                                                     \
    template void gemm_nt<Elt>(const ModCtx<Elt>&, DenseView<const Elt>, DenseView<const Elt>, DenseView<Elt>, GemmOp); \
    template void gemv<Elt>(const ModCtx<Elt>&, DenseView<const Elt>, DenseView<const Elt>, DenseView<Elt>, GemmOp);    \
    template Elt  dot_delayed<Elt>(const ModCtx<Elt>&, const Elt*, const Elt*, size_t);                     \
    template void axpy<Elt>(const ModCtx<Elt>&, Elt, DenseView<const Elt>, DenseView<Elt>, bool);           \
    template void scal<Elt>(const ModCtx<Elt>&, Elt, DenseView<Elt>);                                       \
    template void trmm<Elt>(const ModCtx<Elt>&, DenseView<const Elt>, DenseView<const Elt>, DenseView<Elt>); \
    template void apply_signed_antidiag<Elt>(const ModCtx<Elt>&, DenseView<const Elt>, DenseView<const Elt>, DenseView<Elt>, Side); \
    template void apply_diagonal<Elt>(const ModCtx<Elt>&, DenseView<const Elt>, DenseView<const Elt>, DenseView<Elt>, Side); \
    template void copy<Elt>(DenseView<const Elt>, DenseView<Elt>);                                          \
    template void transpose<Elt>(DenseView<const Elt>, DenseView<Elt>);                                     \
    template void canonicalise<Elt>(const ModCtx<Elt>&, DenseView<Elt>);                                    \
    template void swap_rows<Elt>(DenseView<Elt>, uint32_t, uint32_t);

ENGINE_KERNELS_INSTANTIATE(uint32_t)
ENGINE_KERNELS_INSTANTIATE(uint64_t)
#undef ENGINE_KERNELS_INSTANTIATE

// (slot-11 WP-18a) u128: gemm / gemv / dot_delayed are the explicit
// specialisations above; everything else is the generic template.
template void gemm_nt<u128>(const ModCtx<u128>&, DenseView<const u128>, DenseView<const u128>, DenseView<u128>, GemmOp);
template void axpy<u128>(const ModCtx<u128>&, u128, DenseView<const u128>, DenseView<u128>, bool);
template void scal<u128>(const ModCtx<u128>&, u128, DenseView<u128>);
template void trmm<u128>(const ModCtx<u128>&, DenseView<const u128>, DenseView<const u128>, DenseView<u128>);
template void apply_signed_antidiag<u128>(const ModCtx<u128>&, DenseView<const u128>, DenseView<const u128>, DenseView<u128>, Side);
template void apply_diagonal<u128>(const ModCtx<u128>&, DenseView<const u128>, DenseView<const u128>, DenseView<u128>, Side);
template void copy<u128>(DenseView<const u128>, DenseView<u128>);
template void transpose<u128>(DenseView<const u128>, DenseView<u128>);
template void canonicalise<u128>(const ModCtx<u128>&, DenseView<u128>);
template void swap_rows<u128>(DenseView<u128>, uint32_t, uint32_t);

} // namespace kernels
} // namespace engine
