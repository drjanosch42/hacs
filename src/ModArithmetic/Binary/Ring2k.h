// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once

// The Z/2^k ladder (slot-22 BIN-2; docs/implementation/slot22/L23_master_plan.md Sec. 2.2, ENGINE_API.md Sec. 8.2).
//
// One template body, one element type per rung:
//   R0  k = 1          binary::BitMatrix (BIN-1), reached through the wide API below
//   R1  k <= 8         uint8_t
//   R2  k <= 16 / 32   uint16_t / uint32_t
//   R3  k <= 64        uint64_t
//   R4  k <= 128       unsigned __int128
//   R5  k <= 64 N      WordN<N> (N little-endian 64-bit words, truncated product, N <= 64)
//   R6  any k          nf::BigInt reference (Ring2kReference.h; oracle only, never production)
//
// NOTHING EVER REDUCES.  Every rung computes in Z/2^W (W = the element width) by native wraparound
// (or truncated N-word arithmetic); Z/2^k, k <= W, is its quotient, so a result mod 2^k is read by
// masking.  Only three things look at the value mod 2^k: the valuation (ctz of the masked value,
// capped at k), the zero test, and the output (masked to k bits).  Exact division by 2^v is a right
// shift; the inverse of an odd unit is Newton's x <- x (2 - u x) to the full width W (so u x == 1
// mod 2^W, and an elimination step x - f * pivot is exactly zero mod 2^W).
//
// What is computed (every output is a unique function of the input, ENGINE_API.md Sec. 13.5 / 13.6):
//   * smith(A, k)        the Smith valuations sigma_1 <= ... (the non-zero diagonal 2^sigma_i of the
//                        Smith form over Z/2^k, sigma_i < k); colspan / rowspan(A) = sum Z/2^(k - sigma_i).
//                        Algorithm: valuation LEVELS.  At level s, the F_2 column rank profile of the
//                        matrix mod 2 (BitMatrix echelon, rank r_s) is moved to the front, and a blocked
//                        right-looking LU whose pivot is the first ODD entry of each of those r_s columns
//                        (one always exists) gives r_s invariants equal to s; the Schur complement on the
//                        remaining rows and columns is even, is halved exactly and becomes level s + 1
//                        over Z/2^(k-s-1).  Panels of `panel` columns, trailing update = unit-lower TRSM
//                        on the panel's pivot rows + one GEMM.  (No column is ever skipped: a skipped
//                        column would keep even entries in the later pivot rows that no update reaches.)
//   * howell(A, k)       the Howell form (ENGINE_API.md Sec. 13.5; equal entrywise to engine::elim::howell):
//                        a blocked MINIMAL-VALUATION LU echelon first (rowspan unchanged), then the
//                        Sec. 13.5 pass with its closure rows on the <= n echelon rows, then a blocked
//                        reduce-above (floor quotients per block, one GEMM per block).
//   * kernel_module(A,k) the Howell form of the right kernel {x : A x = 0} (the Sec. 13.6 rule, via
//                        howell([A^T | I_n])), equal entrywise to engine::elim::kernel_module.
//   * length             sum (k - v_i) over the Howell pivots = sum (k - sigma_i).
// References (unblocked, validation use): smith_naive (global minimal-valuation pivoting, the
// prototype's "smith" mode), howell_naive (the Sec. 13.5 pass on A itself).
//
// Single-threaded, except under smithpar::ScopedEnable (slot-24 PAR-2K, SmithPar.h; the descent's Smith calls only):
// echelon_blocked then splits the trailing GEMM (row / column tiles of C), the panel TRSM (column tiles) and the panel's
// row updates (row tiles, the next pivot search fused per tile) over threads.  Every entry gets the serial arithmetic
// (wraparound has one answer) and the pivot rule reads the same values, so every output is bit-identical for every
// thread count; HECKE_DESCENT_SMITH_MT=0 or ENGINE_THREADS=1 is the serial code verbatim.  The native rungs' AXPY / GEMM kernels have portable / AVX2 / AVX-512 bodies
// picked at runtime (set_kernel_path forces one; every body is bit-identical by construction:
// wraparound integer arithmetic has one answer).  R5 (WordN<N>) has SoA-limb SIMD GEMM / AXPY and
// mask-carry row kernels on the same paths (slot-23 R5V, r5k:: below); R4 stays scalar.

#include <cstddef>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "../ModArithmetic/EngineCheck.h"
#include "../ModArithmetic/CpuFeatures.h"
#include "BitMatrix.h"
#include "SmithPar.h"

namespace engine {
namespace ring2k {

using u128 = unsigned __int128;

// ---------------------------------------------------------------------------
// R5's element: N little-endian 64-bit words, arithmetic mod 2^(64 N).
// ---------------------------------------------------------------------------
template<uint32_t N>
struct WordN {
    static_assert(N >= 1 && N <= 64, "WordN: 1 <= N <= 64");
    uint64_t w[N];
};

// ---------------------------------------------------------------------------
// Kernel paths of the native rungs (u8 / u16 / u32 / u64).
// ---------------------------------------------------------------------------
using KernelPath = cpu::IsaPath;   // (S24 CL-5, C-06) the one enum (CpuFeatures.h); Auto follows ENGINE_KERNEL (Kernels.h isa_ceiling)
bool        kernel_path_available(KernelPath p);
void        set_kernel_path(KernelPath p);   // Auto = the best available; ENGINE_CHECKs availability
KernelPath  active_kernel_path();            // never Auto
const char* kernel_path_name(KernelPath p);

// The native kernels (Ring2k.cpp): r -= f * p over n entries; C (mrows x ncols, ldc) -= L (mrows x np,
// ldl) * U (np x ncols, ldu).  Wraparound mod 2^(8 sizeof(T)).
// (slot-23 BIN3O1) kernel_shift_rows: on the (n + 1) x w block X (row stride ld), for i = 0 .. n - 1 and j = n - 1 down
// to i, row j -= f * row (j + 1) -- exactly that sequence of kernel_axpy calls, in one call (the degree-1 amalgam reader's
// Taylor shift p(t) -> p(t - f), AmalgamReader.cpp; f = -1 / +1 run as plain add / subtract loops: the same wraparound).
#define RING2K_DECLARE_NATIVE(T)                                                                          \
    void kernel_axpy(T* r, const T* p, T f, size_t n);                                                    \
    void kernel_gemm_sub(T* C, size_t ldc, uint32_t mrows, const T* L, size_t ldl, const T* U, size_t ldu, \
                         uint32_t np, uint32_t ncols);                                                   \
    void kernel_shift_rows(T* X, size_t ld, uint32_t n, size_t w, T f);
RING2K_DECLARE_NATIVE(uint8_t)
RING2K_DECLARE_NATIVE(uint16_t)
RING2K_DECLARE_NATIVE(uint32_t)
RING2K_DECLARE_NATIVE(uint64_t)
#undef RING2K_DECLARE_NATIVE

// ---------------------------------------------------------------------------
// R5's SIMD kernels (slot-23 R5V; Ring2kR5.cpp, ENGINE_API.md Sec. 8.2).
//
// The storage stays array-of-structures (Mat<WordN<N>>: N words per element, row-major); each kernel packs
// its operands into a STRUCTURE-OF-ARRAYS tile of B-bit limbs (B = 27 or 28, chosen per N so that a 64-bit
// lane can absorb >= 16 rank-one updates of limb products without overflow), computes VW columns at once
// (one 64-bit lane per column: 2 portable / 4 AVX2 / 8 AVX-512) with 32 x 32 -> 64-bit lane products and
// delayed carries, and unpacks.  The row kernels (dst +=, -=, = - src) propagate the word carries of 64
// words at once through generate / propagate bit masks.  Every result is mod 2^(64 N) and hence unique:
// the SIMD kernels are bit-identical to the scalar Traits<WordN<N>> bodies, which stay as the oracle
// (Ring2kR5SimdValidation).  The path is the native rungs' one (set_kernel_path / active_kernel_path);
// set_r5_kernel(Scalar) routes Traits<WordN<N>> back to the scalar bodies.  Only the N of the reader's
// ladder (3, 4, 5, 6, 8, 10, 12, 16) have SIMD bodies; every other N is scalar.
// ---------------------------------------------------------------------------
enum class R5Kernel : uint8_t { Simd, Scalar };
void        set_r5_kernel(R5Kernel k);
R5Kernel    r5_kernel();
const char* r5_kernel_name(R5Kernel k);
constexpr bool r5_simd_instantiated(uint32_t N) {
    return N == 3 || N == 4 || N == 5 || N == 6 || N == 8 || N == 10 || N == 12 || N == 16;
}
namespace r5k {
// Element strides (ldc, ldl, ldu) count ELEMENTS (N words each).  ENGINE_CHECK: r5_simd_instantiated(N).
// Precondition: the output is disjoint from the inputs (gemm_sub: C, L, U pairwise disjoint; axpy: r, p disjoint
// and f not inside r); the SIMD and scalar bodies read and write in different orders, so overlap breaks equality.
void gemm_sub(uint32_t N, uint64_t* C, size_t ldc, uint32_t mrows, const uint64_t* L, size_t ldl, const uint64_t* U,
              size_t ldu, uint32_t np, uint32_t ncols);                         // C -= L U
void axpy(uint32_t N, uint64_t* r, const uint64_t* p, const uint64_t* f, size_t n);   // r -= f p
void add_row(uint32_t N, uint64_t* d, const uint64_t* s, size_t n);                    // d += s
void sub_row(uint32_t N, uint64_t* d, const uint64_t* s, size_t n);                    // d -= s
void neg_row(uint32_t N, uint64_t* d, size_t n);                                       // d = -d
// The limb layout of a path (diagnostic): B bits per limb, NL limbs, TC rank-one updates between carries.
struct Layout { uint32_t B = 0, NL = 0, TC = 0, lanes = 0; };
Layout layout(uint32_t N, KernelPath p);
// VALIDATION ONLY (Ring2kR5SimdValidation's mutation controls; never set in production, 0 by default): 1 drops the
// GEMM / AXPY borrow chain, 2 disables the carries between rank-one updates, 3 drops the row kernels' element
// top-word mask.  Each makes some result wrong, which the validation must catch.
void     set_fault(uint32_t f);
uint32_t fault();
} // namespace r5k

// ---------------------------------------------------------------------------
// Element arithmetic of a rung: Traits<E>.  Every operation is mod 2^kBits.
// ---------------------------------------------------------------------------
template<class E> struct Traits;

namespace detail {

inline uint32_t ctz64(uint64_t x) { return static_cast<uint32_t>(__builtin_ctzll(x)); }

// Native unsigned integers (uint8_t ... uint64_t, u128).  Narrow types compute in `unsigned` so no
// integer promotion to (signed) int can overflow.
template<class T>
struct NativeTraits {
    using E = T;
    using Wd = typename std::conditional<(sizeof(T) < sizeof(unsigned)), unsigned, T>::type;
    static constexpr uint32_t kBits = 8 * sizeof(T);
    static constexpr uint32_t kWords = (kBits + 63) / 64;
    static const char* name() {
        return sizeof(T) == 1 ? "u8" : sizeof(T) == 2 ? "u16" : sizeof(T) == 4 ? "u32" : sizeof(T) == 8 ? "u64" : "u128";
    }
    static E zero() { return E(0); }
    static E one() { return E(1); }
    static E from_u64(uint64_t v) { return E(v); }
    static bool eq(E a, E b) { return a == b; }
    static bool is_zero(E a) { return a == 0; }
    static E add(E a, E b) { return E(Wd(a) + Wd(b)); }
    static E sub(E a, E b) { return E(Wd(a) - Wd(b)); }
    static E neg(E a) { return E(Wd(0) - Wd(a)); }
    static E mul(E a, E b) { return E(Wd(a) * Wd(b)); }
    static bool odd(E a) { return (a & 1u) != 0; }
    static uint32_t ctz(E a) {
        if (a == 0) return kBits;
        if constexpr (sizeof(T) == 16) {
            const uint64_t lo = static_cast<uint64_t>(a);
            return lo ? ctz64(lo) : 64u + ctz64(static_cast<uint64_t>(a >> 64));
        } else {
            return ctz64(static_cast<uint64_t>(a));
        }
    }
    static E shr(E a, uint32_t v) { return v >= kBits ? E(0) : E(a >> v); }
    static E shl(E a, uint32_t v) { return v >= kBits ? E(0) : E(Wd(a) << v); }
    static E low_mask(uint32_t k) { return k >= kBits ? E(~E(0)) : E((E(1) << k) - 1); }
    static E mask(E a, uint32_t k) { return E(a & low_mask(k)); }
    static bool zero_mod(E a, uint32_t k) { return mask(a, k) == 0; }
    static uint32_t val(E a, uint32_t k) { const uint32_t c = ctz(mask(a, k)); return c < k ? c : k; }
    // Newton: u * u == 1 mod 8 for odd u, so x0 = u is right to 3 bits; each step doubles.
    static E inv_odd(E u) {
        E x = u;
        for (uint32_t b = 3; b < kBits; b *= 2) x = mul(x, sub(E(2), mul(u, x)));
        return x;
    }
    static E from_words(const uint64_t* w, uint32_t nw) {
        if constexpr (sizeof(T) == 16) {
            return nw >= 2 ? (static_cast<u128>(w[1]) << 64) | w[0] : (nw == 1 ? static_cast<u128>(w[0]) : u128(0));
        } else {
            return nw >= 1 ? E(w[0]) : E(0);
        }
    }
    static void to_words(E a, uint64_t* w, uint32_t nw) {
        for (uint32_t i = 0; i < nw; ++i) w[i] = 0;
        if (nw == 0) return;
        if constexpr (sizeof(T) == 16) {
            w[0] = static_cast<uint64_t>(a);
            if (nw >= 2) w[1] = static_cast<uint64_t>(a >> 64);
        } else {
            w[0] = static_cast<uint64_t>(a);
        }
    }
    static void axpy(E* r, const E* p, E f, size_t n) {
        if constexpr (sizeof(T) == 16) {
            for (size_t j = 0; j < n; ++j) r[j] -= f * p[j];
        } else {
            kernel_axpy(r, p, f, n);
        }
    }
    static void gemm_sub(E* C, size_t ldc, uint32_t mrows, const E* L, size_t ldl, const E* U, size_t ldu,
                         uint32_t np, uint32_t ncols) {
        if constexpr (sizeof(T) == 16) {
            for (uint32_t i = 0; i < mrows; ++i)
                for (uint32_t t = 0; t < np; ++t) {
                    const E f = L[i * ldl + t];
                    if (f) axpy(C + i * ldc, U + t * ldu, f, ncols);
                }
        } else {
            kernel_gemm_sub(C, ldc, mrows, L, ldl, U, ldu, np, ncols);
        }
    }
};

} // namespace detail

template<> struct Traits<uint8_t>  : detail::NativeTraits<uint8_t>  {};
template<> struct Traits<uint16_t> : detail::NativeTraits<uint16_t> {};
template<> struct Traits<uint32_t> : detail::NativeTraits<uint32_t> {};
template<> struct Traits<uint64_t> : detail::NativeTraits<uint64_t> {};
template<> struct Traits<u128>     : detail::NativeTraits<u128>     {};

template<uint32_t N>
struct Traits<WordN<N>> {
    using E = WordN<N>;
    static constexpr uint32_t kBits = 64 * N;
    static constexpr uint32_t kWords = N;
    static const char* name() { return "words"; }
    static E zero() { E r; for (uint32_t i = 0; i < N; ++i) r.w[i] = 0; return r; }
    static E one() { E r = zero(); r.w[0] = 1; return r; }
    static E from_u64(uint64_t v) { E r = zero(); r.w[0] = v; return r; }
    static bool eq(const E& a, const E& b) { return std::memcmp(a.w, b.w, sizeof(a.w)) == 0; }
    static bool is_zero(const E& a) { for (uint32_t i = 0; i < N; ++i) if (a.w[i]) return false; return true; }
    static E add(const E& a, const E& b) {
        E r;
        unsigned long long c = 0;
        for (uint32_t i = 0; i < N; ++i) {
            const u128 t = static_cast<u128>(a.w[i]) + b.w[i] + c;
            r.w[i] = static_cast<uint64_t>(t);
            c = static_cast<unsigned long long>(t >> 64);
        }
        return r;
    }
    static E sub(const E& a, const E& b) {
        E r;
        uint64_t borrow = 0;
        for (uint32_t i = 0; i < N; ++i) {
            const uint64_t x = a.w[i], y = b.w[i];
            const uint64_t d = x - y - borrow;
            borrow = (x < y) || (x == y && borrow) ? 1u : 0u;
            r.w[i] = d;
        }
        return r;
    }
    static E neg(const E& a) { return sub(zero(), a); }
    // Truncated schoolbook product: N (N + 1) / 2 word multiplies, the carries out of word N - 1 dropped.
    static E mul(const E& a, const E& b) {
        E r = zero();
        for (uint32_t i = 0; i < N; ++i) {
            const uint64_t ai = a.w[i];
            if (!ai) continue;
            uint64_t carry = 0;
            for (uint32_t j = 0; i + j < N; ++j) {
                const u128 t = static_cast<u128>(ai) * b.w[j] + r.w[i + j] + carry;
                r.w[i + j] = static_cast<uint64_t>(t);
                carry = static_cast<uint64_t>(t >> 64);
            }
        }
        return r;
    }
    static bool odd(const E& a) { return (a.w[0] & 1u) != 0; }
    static uint32_t ctz(const E& a) {
        for (uint32_t i = 0; i < N; ++i)
            if (a.w[i]) return 64 * i + detail::ctz64(a.w[i]);
        return kBits;
    }
    static E shr(const E& a, uint32_t v) {
        if (v >= kBits) return zero();
        const uint32_t q = v / 64, s = v % 64;
        E r = zero();
        for (uint32_t i = 0; i + q < N; ++i) {
            uint64_t x = a.w[i + q] >> s;
            if (s && i + q + 1 < N) x |= a.w[i + q + 1] << (64 - s);
            r.w[i] = x;
        }
        return r;
    }
    static E shl(const E& a, uint32_t v) {
        if (v >= kBits) return zero();
        const uint32_t q = v / 64, s = v % 64;
        E r = zero();
        for (uint32_t i = N; i-- > q;) {
            uint64_t x = a.w[i - q] << s;
            if (s && i - q >= 1) x |= a.w[i - q - 1] >> (64 - s);
            r.w[i] = x;
        }
        return r;
    }
    static E mask(const E& a, uint32_t k) {
        if (k >= kBits) return a;
        E r = a;
        const uint32_t q = k / 64, s = k % 64;
        if (s) { r.w[q] &= (uint64_t(1) << s) - 1; }
        else   { r.w[q] = 0; }
        for (uint32_t i = q + 1; i < N; ++i) r.w[i] = 0;
        return r;
    }
    static bool zero_mod(const E& a, uint32_t k) {
        const uint32_t full = k / 64 < N ? k / 64 : N;
        for (uint32_t i = 0; i < full; ++i) if (a.w[i]) return false;
        if (k < kBits && (k % 64)) return (a.w[k / 64] & ((uint64_t(1) << (k % 64)) - 1)) == 0;
        return true;
    }
    static uint32_t val(const E& a, uint32_t k) {
        const uint32_t c = ctz(a);
        return c < k ? c : k;
    }
    static E inv_odd(const E& u) {
        E x = u;
        const E two = from_u64(2);
        for (uint32_t b = 3; b < kBits; b *= 2) x = mul(x, sub(two, mul(u, x)));
        return x;
    }
    static E from_words(const uint64_t* w, uint32_t nw) {
        E r = zero();
        for (uint32_t i = 0; i < N && i < nw; ++i) r.w[i] = w[i];
        return r;
    }
    static void to_words(const E& a, uint64_t* w, uint32_t nw) {
        for (uint32_t i = 0; i < nw; ++i) w[i] = i < N ? a.w[i] : 0;
    }
    // The scalar bodies (the oracle of the SIMD kernels, slot-23 R5V).
    static void axpy_scalar(E* r, const E* p, const E& f, size_t n) {
        for (size_t j = 0; j < n; ++j) r[j] = sub(r[j], mul(f, p[j]));
    }
    static void gemm_sub_scalar(E* C, size_t ldc, uint32_t mrows, const E* L, size_t ldl, const E* U, size_t ldu,
                                uint32_t np, uint32_t ncols) {
        for (uint32_t i = 0; i < mrows; ++i)
            for (uint32_t t = 0; t < np; ++t) {
                const E& f = L[i * ldl + t];
                if (!is_zero(f)) axpy_scalar(C + i * ldc, U + t * ldu, f, ncols);
            }
    }
    static void add_row_scalar(E* d, const E* s, size_t n) { for (size_t j = 0; j < n; ++j) d[j] = add(d[j], s[j]); }
    static void sub_row_scalar(E* d, const E* s, size_t n) { for (size_t j = 0; j < n; ++j) d[j] = sub(d[j], s[j]); }
    static void neg_row_scalar(E* d, size_t n) { for (size_t j = 0; j < n; ++j) d[j] = neg(d[j]); }

    // The production entry points: the SIMD kernels (r5k::, the active kernel path) unless set_r5_kernel(Scalar)
    // or N has no SIMD body.  Bit-identical either way.
    static constexpr bool kR5Rows = true;   // the row kernels below exist (the BIN-3 reader's Taylor shifts use them)
    static bool simd() { return r5_simd_instantiated(N) && r5_kernel() == R5Kernel::Simd; }
    static uint64_t* words(E* x) { return reinterpret_cast<uint64_t*>(x); }
    static const uint64_t* words(const E* x) { return reinterpret_cast<const uint64_t*>(x); }
    static void axpy(E* r, const E* p, const E& f, size_t n) {
        if (simd()) r5k::axpy(N, words(r), words(p), f.w, n);
        else axpy_scalar(r, p, f, n);
    }
    static void gemm_sub(E* C, size_t ldc, uint32_t mrows, const E* L, size_t ldl, const E* U, size_t ldu,
                         uint32_t np, uint32_t ncols) {
        if (simd()) r5k::gemm_sub(N, words(C), ldc, mrows, words(L), ldl, words(U), ldu, np, ncols);
        else gemm_sub_scalar(C, ldc, mrows, L, ldl, U, ldu, np, ncols);
    }
    static void add_row(E* d, const E* s, size_t n) { if (simd()) r5k::add_row(N, words(d), words(s), n); else add_row_scalar(d, s, n); }
    static void sub_row(E* d, const E* s, size_t n) { if (simd()) r5k::sub_row(N, words(d), words(s), n); else sub_row_scalar(d, s, n); }
    static void neg_row(E* d, size_t n) { if (simd()) r5k::neg_row(N, words(d), n); else neg_row_scalar(d, n); }
};
static_assert(sizeof(WordN<3>) == 3 * sizeof(uint64_t) && sizeof(WordN<16>) == 16 * sizeof(uint64_t),
              "ring2k: WordN<N> must be exactly N packed words (the R5 kernels address a row as words)");

// ---------------------------------------------------------------------------
// Dense row-major storage and results.
// ---------------------------------------------------------------------------
template<class E>
struct Mat {
    uint32_t rows = 0, cols = 0;
    std::vector<E> a;
    Mat() = default;
    Mat(uint32_t r, uint32_t c) : rows(r), cols(c), a(static_cast<size_t>(r) * c, Traits<E>::zero()) {}
    E*       row(uint32_t i)       { return a.data() + static_cast<size_t>(i) * cols; }
    const E* row(uint32_t i) const { return a.data() + static_cast<size_t>(i) * cols; }
    E&       at(uint32_t i, uint32_t j)       { return a[static_cast<size_t>(i) * cols + j]; }
    const E& at(uint32_t i, uint32_t j) const { return a[static_cast<size_t>(i) * cols + j]; }
};

struct SmithResult {
    uint32_t k = 0;
    std::vector<uint32_t> sigma;          // ascending; each < k (the non-zero Smith diagonal entries 2^sigma)
    std::vector<uint32_t> level_ranks;    // level_ranks[s] = #{sigma == s} (smith() only)
    uint32_t length() const { uint32_t s = 0; for (uint32_t x : sigma) s += k - x; return s; }
};

template<class E>
struct HowellForm {
    uint32_t k = 0;
    Mat<E> H;                             // h x n, entries canonical (masked to k bits)
    std::vector<uint32_t> pivots;         // strictly increasing leading columns
    std::vector<uint32_t> pivot_v;        // row i leads with exactly 2^pivot_v[i], 0 <= v < k
    uint32_t length() const { uint32_t s = 0; for (uint32_t v : pivot_v) s += k - v; return s; }
};

// Tunables (validation / tuning; values never affect a result).
struct Params {
    uint32_t panel = 64;          // LU panel width (columns)
    uint32_t reduce_block = 64;   // reduce-above block (pivots)
};
Params params();
void   set_params(const Params& p);

// Instrumentation (diagnostic only).
struct Counters { uint64_t lu_calls = 0, lu_pivots = 0, closure_rows = 0, smith_levels = 0; };
Counters counters();
void     reset_counters();
void     add_counters(const Counters& d);

// ---------------------------------------------------------------------------
// The algorithms (templates on the rung's traits; the production rungs are instantiated in Ring2k.cpp).
// ---------------------------------------------------------------------------
enum class PivotRule : uint8_t { FirstOdd, MinValuation };

struct EchelonInfo {
    uint32_t rank = 0;
    std::vector<uint32_t> pivots;    // ascending
    std::vector<uint32_t> pivot_v;
};

namespace detail {

template<class E>
void swap_rows(Mat<E>& A, uint32_t i, uint32_t j) {
    if (i == j) return;
    E* a = A.row(i);
    E* b = A.row(j);
    for (uint32_t c = 0; c < A.cols; ++c) std::swap(a[c], b[c]);
}

// (slot-24 PAR-2K) The threaded pieces of echelon_blocked.  T = smithpar::threads() > 1.  Each splits its output into
// disjoint tiles and runs the serial kernel on each tile, so every entry receives exactly the serial arithmetic.

// C (mrows x ncols) -= L (mrows x np) U (np x ncols): row tiles (multiples of 4 rows, the micro-kernel's height), and
// column tiles (multiples of 64) when there are too few rows for the threads.
template<class Tr>
void gemm_sub_mt(uint32_t T, typename Tr::E* C, size_t ldc, uint32_t mrows, const typename Tr::E* L, size_t ldl,
                 const typename Tr::E* U, size_t ldu, uint32_t np, uint32_t ncols) {
    const uint64_t work = static_cast<uint64_t>(mrows) * ncols * np;
    const uint64_t mw = smithpar::min_work();
    if (T <= 1 || mrows == 0 || ncols == 0 || np == 0 || work < 2 * mw) {
        Tr::gemm_sub(C, ldc, mrows, L, ldl, U, ldu, np, ncols);
        return;
    }
    uint64_t want = 4ull * T;
    if (mw > 0 && work / mw < want) want = work / mw;
    if (want < 2) want = 2;
    const uint32_t row_cap = (mrows + 15) / 16;
    const uint32_t rt = static_cast<uint32_t>(want < row_cap ? want : row_cap);
    uint32_t ct = static_cast<uint32_t>((want + rt - 1) / rt);
    const uint32_t col_cap = (ncols + 63) / 64;
    if (ct > col_cap) ct = col_cap;
    if (ct == 0) ct = 1;
    smithpar::parallel_for(rt * ct, [&](uint32_t t) {
        const uint32_t ri = t / ct, ci = t % ct;
        const uint32_t r0 = smithpar::split_at(mrows, rt, ri, 4), r1 = smithpar::split_at(mrows, rt, ri + 1, 4);
        const uint32_t c0 = smithpar::split_at(ncols, ct, ci, 64), c1 = smithpar::split_at(ncols, ct, ci + 1, 64);
        if (r1 > r0 && c1 > c0)
            Tr::gemm_sub(C + static_cast<size_t>(r0) * ldc + c0, ldc, r1 - r0, L + static_cast<size_t>(r0) * ldl, ldl, U + c0, ldu,
                         np, c1 - c0);
    });
}

// The unit-lower TRSM on the panel's pivot rows r .. r + np - 1, columns [c1, n): column tiles (each column its own solve).
template<class Tr>
void panel_trsm(uint32_t T, Mat<typename Tr::E>& A, const std::vector<typename Tr::E>& L, uint32_t panel, uint32_t r, uint32_t np,
                uint32_t c1) {
    const uint32_t n = A.cols;
    auto body = [&](uint32_t a, uint32_t b) {
        for (uint32_t t = 1; t < np; ++t) {
            const typename Tr::E* lt = L.data() + static_cast<size_t>(r + t) * panel;
            for (uint32_t t2 = 0; t2 < t; ++t2)
                if (!Tr::is_zero(lt[t2])) Tr::axpy(A.row(r + t) + a, A.row(r + t2) + a, lt[t2], b - a);
        }
    };
    const uint64_t work = static_cast<uint64_t>(np) * np / 2 * (n - c1);
    const uint64_t mw = smithpar::min_work();
    if (T <= 1 || np < 2 || work < 2 * mw) { body(c1, n); return; }
    uint64_t want = 2ull * T;
    if (mw > 0 && work / mw < want) want = work / mw;
    const uint32_t w = n - c1, cap = (w + 63) / 64;
    const uint32_t tiles = static_cast<uint32_t>(want < cap ? want : cap);
    if (tiles < 2) { body(c1, n); return; }
    smithpar::parallel_for(tiles, [&](uint32_t t) {
        const uint32_t a = smithpar::split_at(w, tiles, t, 64), b = smithpar::split_at(w, tiles, t + 1, 64);
        if (b > a) body(c1 + a, c1 + b);
    });
}

} // namespace detail

// Blocked right-looking LU over Z/2^k (entries understood mod 2^k).  On return rows [0, rank) are the
// echelon rows (row t leads at pivots[t] with valuation pivot_v[t]; exactly zero in every earlier
// pivot column; under MinValuation == 0 mod 2^k left of its pivot), rows [rank, m) are zero mod 2^k in
// every pivot column and (FirstOdd) even / (MinValuation) zero mod 2^k elsewhere.  Row operations
// only, all unimodular, so rowspan is unchanged.
//
// Pivots are sought in the columns [0, col_limit) only; the row operations reach every column.
// CAUTION (FirstOdd): a column without an odd entry is skipped, but the later pivot rows keep EVEN,
// non-zero entries there which no row operation updates (the panel AXPY starts at the pivot column,
// the trailing update at the panel end).  Under MinValuation those entries are 0 mod 2^k and nothing
// is lost; under FirstOdd the caller must therefore put the F_2 pivot columns first and pass
// col_limit = the F_2 rank, so that no column is ever skipped (smith() does exactly this).
namespace device {
// (slot-24 PAR-GPU) the device-resident LU of the u32 / u64 rungs (Ring2kDevice.h, Ring2kDeviceLU.h): false = not taken.
template<class Tr>
bool echelon_device(Mat<typename Tr::E>& A, uint32_t k, PivotRule rule, uint32_t panel, uint32_t climit, EchelonInfo& info);
} // namespace device

template<class Tr>
EchelonInfo echelon_blocked(Mat<typename Tr::E>& A, uint32_t k, PivotRule rule, uint32_t panel,
                            uint32_t col_limit = 0xffffffffu, uint32_t device_panel = 0) {
    using E = typename Tr::E;
    ENGINE_CHECK_MSG(k >= 1 && k <= Tr::kBits, "ring2k::echelon_blocked: need 1 <= k <= the rung's width");
    if (panel == 0) panel = 1;
    EchelonInfo info;
    const uint32_t m = A.rows, n = A.cols;
    if (m == 0 || n == 0) return info;
    const uint32_t climit = col_limit < n ? col_limit : n;
    // (PAR-GPU) the opt-in device route (ENGINE_DEVICE=gpu; Ring2kDevice.h): same values, else the host code below.
    // (PANEL-IMPL) device_panel = its own width (smith / howell: smithpar::route_device_panel); 0 = `panel` (direct callers).
    if (device::echelon_device<Tr>(A, k, rule, device_panel ? device_panel : panel, climit, info)) return info;
    std::vector<E> L(static_cast<size_t>(m) * panel, Tr::zero());
    uint32_t r = 0, c = 0;
    uint64_t piv_count = 0;
    // (PAR-2K) T > 1: the threaded route -- the left-looking ("lazy") panel below, detail::panel_trsm, detail::gemm_sub_mt.
    const uint32_t T = smithpar::threads();
    // Lazy-panel state: lvl[i - r] = the panel pivots already applied to row i; the pivots' columns, valuations, unit inverses.
    std::vector<uint32_t> lvl, pj, pbv;
    std::vector<E> pui;
    while (c < climit && r < m) {
        const uint32_t c1 = c + panel < climit ? c + panel : climit;
        uint32_t np = 0;
        if (T <= 1) {
            for (uint32_t j = c; j < c1 && r + np < m; ++j) {
                const uint32_t prow = r + np;
                uint32_t best = m, bv = k;
                for (uint32_t i = prow; i < m; ++i) {
                    const E& x = A.at(i, j);
                    if (rule == PivotRule::FirstOdd) {
                        if (Tr::odd(x)) { best = i; bv = 0; break; }
                    } else {
                        const uint32_t v = Tr::val(x, k);
                        if (v < bv) { bv = v; best = i; if (v == 0) break; }
                    }
                }
                if (best == m) {
                    // BIN-2 audit F4 / R3: under FirstOdd a skipped column always leaves a wrong Schur complement.
                    ENGINE_CHECK_MSG(rule != PivotRule::FirstOdd,
                                     "ring2k::echelon_blocked: FirstOdd met a column without an odd entry (put the F_2 rank profile "
                                     "first and pass col_limit = the F_2 rank; see the CAUTION)");
                    continue;
                }
                if (best != prow) {
                    detail::swap_rows(A, best, prow);
                    E* la = L.data() + static_cast<size_t>(best) * panel;
                    E* lb = L.data() + static_cast<size_t>(prow) * panel;
                    for (uint32_t t = 0; t < panel; ++t) std::swap(la[t], lb[t]);
                }
                const E* P = A.row(prow);
                const E ui = Tr::inv_odd(Tr::shr(P[j], bv));
                for (uint32_t i = prow + 1; i < m; ++i) {
                    E* R = A.row(i);
                    E& slot = L[static_cast<size_t>(i) * panel + np];
                    if (Tr::zero_mod(R[j], k)) { slot = Tr::zero(); continue; }
                    const E f = Tr::mul(Tr::shr(R[j], bv), ui);
                    slot = f;
                    Tr::axpy(R + j, P + j, f, c1 - j);
                }
                info.pivots.push_back(j);
                info.pivot_v.push_back(bv);
                ++np;
            }
        } else {
            // THE LAZY PANEL (left-looking inside the panel).  Row i receives the panel pivots t = 0, 1, ... in order, each
            // exactly the serial step (slot t = 0 when the pivot-column entry is 0 mod 2^k, else f = (R[j_t] >> v_t) u_t^-1,
            // R[j_t, c1) -= f P_t[j_t, c1)), but only when it is needed: the pivot search of column j brings the rows it
            // reads up to date one at a time and stops where the serial search stops, so it reads the same values; every
            // other row is brought up to date at the end of the panel, each row independently -- one parallel region per
            // panel instead of one per pivot.  Each row's operation sequence, every L slot the TRSM / GEMM read and every
            // pivot choice are the serial ones, so every entry is bit-identical.
            lvl.assign(m - r, 0);
            pj.clear();
            pbv.clear();
            pui.clear();
            auto bring = [&](uint32_t i, uint32_t to) {
                uint32_t& li = lvl[i - r];
                if (li >= to) return;
                E* R = A.row(i);
                E* Li = L.data() + static_cast<size_t>(i) * panel;
                for (uint32_t t = li; t < to; ++t) {
                    const uint32_t j = pj[t];
                    if (Tr::zero_mod(R[j], k)) { Li[t] = Tr::zero(); continue; }
                    const E f = Tr::mul(Tr::shr(R[j], pbv[t]), pui[t]);
                    Li[t] = f;
                    Tr::axpy(R + j, A.row(r + t) + j, f, c1 - j);
                }
                li = to;
            };
            for (uint32_t j = c; j < c1 && r + np < m; ++j) {
                const uint32_t prow = r + np;
                uint32_t best = m, bv = k;
                for (uint32_t i = prow; i < m; ++i) {
                    bring(i, np);
                    const E& x = A.at(i, j);
                    if (rule == PivotRule::FirstOdd) {
                        if (Tr::odd(x)) { best = i; bv = 0; break; }
                    } else {
                        const uint32_t v = Tr::val(x, k);
                        if (v < bv) { bv = v; best = i; if (v == 0) break; }
                    }
                }
                if (best == m) {
                    ENGINE_CHECK_MSG(rule != PivotRule::FirstOdd,
                                     "ring2k::echelon_blocked: FirstOdd met a column without an odd entry (put the F_2 rank profile "
                                     "first and pass col_limit = the F_2 rank; see the CAUTION)");
                    continue;
                }
                if (best != prow) {
                    detail::swap_rows(A, best, prow);
                    E* la = L.data() + static_cast<size_t>(best) * panel;
                    E* lb = L.data() + static_cast<size_t>(prow) * panel;
                    for (uint32_t t = 0; t < panel; ++t) std::swap(la[t], lb[t]);
                    std::swap(lvl[best - r], lvl[prow - r]);
                }
                pj.push_back(j);
                pbv.push_back(bv);
                pui.push_back(Tr::inv_odd(Tr::shr(A.at(prow, j), bv)));
                info.pivots.push_back(j);
                info.pivot_v.push_back(bv);
                ++np;
            }
            // every row below the panel's pivot rows to level np: independent rows, one region
            if (r + np < m) {
                smithpar::count_panel_split();
                const uint32_t rows = m - (r + np), r1 = r + np, npp = np;
                smithpar::for_rows(rows, static_cast<uint64_t>(rows) * npp * (c1 - c) / 2,
                                   [&](uint32_t i0, uint32_t i1) { for (uint32_t i = i0; i < i1; ++i) bring(r1 + i, npp); });
            }
        }
        if (np > 0 && c1 < n) {
            // Unit-lower TRSM on the panel's pivot rows, then one GEMM on the rows below.
            if (T <= 1) {
                for (uint32_t t = 1; t < np; ++t) {
                    const E* lt = L.data() + static_cast<size_t>(r + t) * panel;
                    for (uint32_t t2 = 0; t2 < t; ++t2)
                        if (!Tr::is_zero(lt[t2])) Tr::axpy(A.row(r + t) + c1, A.row(r + t2) + c1, lt[t2], n - c1);
                }
                if (r + np < m)
                    Tr::gemm_sub(A.row(r + np) + c1, n, m - (r + np), L.data() + static_cast<size_t>(r + np) * panel, panel,
                                 A.row(r) + c1, n, np, n - c1);
            } else {
                detail::panel_trsm<Tr>(T, A, L, panel, r, np, c1);
                if (r + np < m)
                    detail::gemm_sub_mt<Tr>(T, A.row(r + np) + c1, n, m - (r + np),
                                            L.data() + static_cast<size_t>(r + np) * panel, panel, A.row(r) + c1, n, np, n - c1);
            }
        }
        piv_count += np;
        r += np;
        c = c1;
    }
    info.rank = r;
    Counters d; d.lu_calls = 1; d.lu_pivots = piv_count;
    add_counters(d);
    return info;
}

// Smith valuations by levels (see the header comment).  A is taken by value: pass an rvalue
// (std::move, or a temporary such as from_wide(...)) and the caller's copy is the working copy.
// Memory (BIN-2 audit R2): the column permutation and the halved Schur complement are done IN PLACE
// in A's storage, so the peak is one m x n matrix of E plus the m x n bit matrix of each level.
template<class Tr>
SmithResult smith(Mat<typename Tr::E> A, uint32_t k) {
    using E = typename Tr::E;
    ENGINE_CHECK_MSG(k >= 1 && k <= Tr::kBits, "ring2k::smith: need 1 <= k <= the rung's width");
    SmithResult res;
    res.k = k;
    const uint32_t dpanel = smithpar::route_device_panel(params().panel);   // (PANEL-IMPL) the device LU's own width
    uint64_t levels = 0;
    std::vector<E> tmp;
    for (uint32_t s = 0; s < k && A.rows > 0 && A.cols > 0; ++s) {
        const uint32_t kk = k - s;
        const bool f2r = smithpar::f2_route();   // (PAR-F2) the lazy F_2 echelon and the vectorised / threaded level passes
        bool any = false;
        {
            smithpar::PhaseTimer tm(smithpar::Phase::LevelPass);
            if (!f2r) {
                for (const E& x : A.a) if (!Tr::zero_mod(x, kk)) { any = true; break; }
            } else {
                // (PAR-F2) the zero test over row tiles (a boolean: whichever tile decides it)
                std::atomic<bool> hit{false};
                smithpar::for_rows(A.rows, static_cast<uint64_t>(A.rows) * A.cols / 4u, [&](uint32_t i0, uint32_t i1) {
                    for (uint32_t i = i0; i < i1 && !hit.load(std::memory_order_relaxed); ++i) {
                        const E* ai = A.row(i);
                        for (uint32_t j = 0; j < A.cols; ++j)
                            if (!Tr::zero_mod(ai[j], kk)) { hit.store(true, std::memory_order_relaxed); return; }
                    }
                });
                any = hit.load();
            }
        }
        if (!any) break;
        ++levels;
        // The F_2 column rank profile of A mod 2 (BIN-1's M4RI echelon), its columns moved to the front:
        // the odd-pivot LU then finds a pivot in each of the first r0 columns and never skips one.
        uint32_t r0 = 0;
        std::vector<uint32_t> order;
        {
            binary::BitMatrix B(A.rows, A.cols);
            {
                smithpar::PhaseTimer tm(smithpar::Phase::F2Pack);
                // (PAR-2K) row tiles under threads (each row of B is its own words)
                smithpar::for_rows(A.rows, static_cast<uint64_t>(A.rows) * A.cols, [&](uint32_t i0, uint32_t i1) {
                    for (uint32_t i = i0; i < i1; ++i) {
                        const E* ai = A.row(i);
                        if constexpr (std::is_same<E, uint32_t>::value || std::is_same<E, uint64_t>::value) {
                            // (PAR-F2) the vector pack of the low bits (the same bits as B.set on every odd entry)
                            if (f2r) { binary::pack_low_bits(ai, A.cols, B.row(i)); continue; }
                        }
                        for (uint32_t j = 0; j < A.cols; ++j) if (Tr::odd(ai[j])) B.set(i, j, true);
                    }
                });
            }
            smithpar::PhaseTimer tme(smithpar::Phase::F2Echelon);
            // (PAR-F2) the lazy blocked M4RI over row tiles: binary::echelon's matrix, rank and pivots, bit for bit
            const binary::BitRref br = f2r ? binary::echelon_lazy(B) : binary::echelon(B);
            r0 = br.rank;
            order.assign(br.pivots.begin(), br.pivots.end());
            order.reserve(A.cols);
            for (uint32_t j = 0, t = 0; j < A.cols; ++j) {
                if (t < br.pivots.size() && br.pivots[t] == j) { ++t; continue; }
                order.push_back(j);
            }
        }
        res.level_ranks.resize(s + 1, 0);
        res.level_ranks[s] = r0;
        for (uint32_t t = 0; t < r0; ++t) res.sigma.push_back(s);
        if (r0 > 0) {
            // Permute the columns in place, one row at a time (a row buffer, not a matrix copy).
            {
            smithpar::PhaseTimer tmp_(smithpar::Phase::Perm);
            if (smithpar::threads() <= 1) {
                tmp.resize(A.cols);
                for (uint32_t i = 0; i < A.rows; ++i) {
                    E* ai = A.row(i);
                    for (uint32_t j = 0; j < A.cols; ++j) tmp[j] = ai[order[j]];
                    for (uint32_t j = 0; j < A.cols; ++j) ai[j] = tmp[j];
                }
            } else {
                // (PAR-2K) row tiles, a row buffer per tile
                smithpar::for_rows(A.rows, static_cast<uint64_t>(A.rows) * A.cols, [&](uint32_t i0, uint32_t i1) {
                    std::vector<E> rb(A.cols);
                    for (uint32_t i = i0; i < i1; ++i) {
                        E* ai = A.row(i);
                        for (uint32_t j = 0; j < A.cols; ++j) rb[j] = ai[order[j]];
                        for (uint32_t j = 0; j < A.cols; ++j) ai[j] = rb[j];
                    }
                });
            }
            }
            EchelonInfo e;
            {
                // (PANEL-IMPL) the host panel by this level's matrix: 512 at >= 12 000 rows on the threaded route (SmithPar.h)
                const uint32_t panel = smithpar::route_panel_2k(params().panel, A.rows);
                smithpar::PhaseTimer tml(smithpar::Phase::LU);
                e = echelon_blocked<Tr>(A, kk, PivotRule::FirstOdd, panel, r0, dpanel);
            }
            ENGINE_CHECK_MSG(e.rank == r0 && (r0 == 0 || e.pivots.back() == r0 - 1),
                             "ring2k::smith: the odd-pivot LU disagrees with the F_2 rank profile");
        }
        smithpar::PhaseTimer tms(smithpar::Phase::Schur);
        // The even Schur complement: rows [r0, m), columns [r0, n) (the non-pivot ones), halved and
        // compacted to the front of A's storage.  The forward copy is safe: destination <= source index.
        const uint32_t sr = A.rows - r0, sc = A.cols - r0;
        if (smithpar::threads() <= 1) {
            for (uint32_t i = 0; i < sr; ++i) {
                const size_t src0 = static_cast<size_t>(r0 + i) * A.cols + r0;   // (r0 == 0: A unpermuted, all even)
                const size_t dst0 = static_cast<size_t>(i) * sc;
                for (uint32_t jj = 0; jj < sc; ++jj) {
                    const E x = A.a[src0 + jj];
                    ENGINE_DEBUG_CHECK_MSG(!Tr::odd(x), "ring2k::smith: Schur complement not even");
                    A.a[dst0 + jj] = Tr::shr(x, 1);
                }
            }
        } else {
            // (PAR-2K) two passes: (1) row tiles halve each source row's columns [r0, n) into the front of THAT row
            // (in-row forward copy); (2) the rows move to their compact place in ascending order (dst(i) ends at
            // (i + 1) sc <= (r0 + i + 1) n = the start of source row i + 1, so no unread row is overwritten).
            const uint32_t n0 = A.cols;
            smithpar::for_rows(sr, static_cast<uint64_t>(sr) * sc, [&](uint32_t i0, uint32_t i1) {
                for (uint32_t i = i0; i < i1; ++i) {
                    E* row = A.a.data() + static_cast<size_t>(r0 + i) * n0;
                    for (uint32_t jj = 0; jj < sc; ++jj) {
                        const E x = row[r0 + jj];
                        ENGINE_DEBUG_CHECK_MSG(!Tr::odd(x), "ring2k::smith: Schur complement not even");
                        row[jj] = Tr::shr(x, 1);
                    }
                }
            });
            if (f2r) {
                smithpar::compact_rows(A.a.data(), r0, n0, sr, sc);   // (PAR-F2) the same move, over row tiles
            } else {
                for (uint32_t i = 0; i < sr; ++i) {
                    const size_t src0 = static_cast<size_t>(r0 + i) * n0, dst0 = static_cast<size_t>(i) * sc;
                    if (src0 != dst0) std::memmove(static_cast<void*>(A.a.data() + dst0), A.a.data() + src0, sizeof(E) * sc);
                }
            }
        }
        A.rows = sr;
        A.cols = sc;
        A.a.resize(static_cast<size_t>(sr) * sc);   // shrinks the size, keeps the capacity: no reallocation
    }
    Counters d; d.smith_levels = levels;
    add_counters(d);
    return res;
}

// The Sec. 13.5 pass (active list, minimal valuation / topmost, scale to 2^v, exact elimination, closure
// rows at the bottom), then reduce above in blocks.  Used on A itself (howell_naive) and on the echelon
// rows (howell).  A is taken by value (pass an rvalue): its storage holds the input rows; the closure
// rows (at most one per non-unit pivot) are allocated ON DEMAND, one row each (BIN-2 audit R2: the old
// eager (m + n) x n store doubled the peak).
template<class Tr>
HowellForm<typename Tr::E> howell_pass(Mat<typename Tr::E> A, uint32_t k, uint32_t reduce_block) {
    using E = typename Tr::E;
    const uint32_t m = A.rows, n = A.cols;
    HowellForm<E> out;
    out.k = k;
    std::vector<E>& store = A.a;                    // the input rows, ids [0, m)
    std::vector<std::vector<E>> closure;            // closure rows, ids [m, m + closure.size())
    auto row = [&](uint32_t id) -> E* {
        return id < m ? store.data() + static_cast<size_t>(id) * n : closure[id - m].data();
    };
    std::vector<uint32_t> active(m);
    for (uint32_t i = 0; i < m; ++i) active[i] = i;
    std::vector<uint32_t> prow_id;
    uint64_t closures = 0;
    for (uint32_t c = 0; c < n && !active.empty(); ++c) {
        size_t best = active.size();
        uint32_t bv = k;
        for (size_t t = 0; t < active.size(); ++t) {
            const uint32_t v = Tr::val(row(active[t])[c], k);
            if (v < bv) { bv = v; best = t; if (v == 0) break; }
        }
        if (best == active.size()) continue;
        const uint32_t pid = active[best];
        active.erase(active.begin() + static_cast<std::ptrdiff_t>(best));
        E* P = row(pid);
        // The unit is inverted from P[c] >> v with the bits above k still in it, so ui == u^-1 only mod
        // 2^(k-v): the scaled row is a unit multiple of Sec. 13.5's; equal output by uniqueness (audit F3).
        const E ui = Tr::inv_odd(Tr::shr(P[c], bv));
        for (uint32_t j = c; j < n; ++j) P[j] = Tr::mask(Tr::mul(P[j], ui), k);   // P[c] == 2^bv exactly
        for (uint32_t id : active) {
            E* R = row(id);
            if (Tr::zero_mod(R[c], k)) continue;
            Tr::axpy(R + c, P + c, Tr::shr(Tr::mask(R[c], k), bv), n - c);
        }
        if (bv > 0) {
            // At most one closure row per pivot, so at most n (audit F10: the check sits on the growth path).
            ENGINE_CHECK_MSG(closure.size() < n, "ring2k::howell_pass: more closure rows than columns");
            closure.emplace_back(n, Tr::zero());
            P = row(pid);   // emplace_back may move the closure rows (a closure pivot row lives there)
            E* Z = closure.back().data();
            for (uint32_t j = c + 1; j < n; ++j) Z[j] = Tr::shl(P[j], k - bv);
            active.push_back(m + static_cast<uint32_t>(closure.size() - 1));
            ++closures;
        }
        prow_id.push_back(pid);
        out.pivots.push_back(c);
        out.pivot_v.push_back(bv);
    }
    const uint32_t h = static_cast<uint32_t>(prow_id.size());
    out.H = Mat<E>(h, n);
    for (uint32_t i = 0; i < h; ++i) {
        const E* src = row(prow_id[i]);
        E* dst = out.H.row(i);
        for (uint32_t j = 0; j < n; ++j) dst[j] = Tr::mask(src[j], k);
    }
    std::vector<E>().swap(store);
    std::vector<std::vector<E>>().swap(closure);
    // Reduce above, pivots left to right, in blocks [t0, t1).
    if (reduce_block == 0) reduce_block = 1;
    Mat<E>& H = out.H;
    std::vector<E> Q, vals;
    for (uint32_t t0 = 0; t0 < h; t0 += reduce_block) {
        const uint32_t t1 = t0 + reduce_block < h ? t0 + reduce_block : h;
        const uint32_t b = t1 - t0;
        const uint32_t c0 = out.pivots[t0];
        if (t0 > 0) {
            // Rows [0, t0): the floor quotients against the block, sequentially on the block's pivot
            // columns only, then one GEMM over the columns >= c0.
            Q.assign(static_cast<size_t>(t0) * b, Tr::zero());
            vals.assign(b, Tr::zero());
            bool nonzero = false;
            for (uint32_t i = 0; i < t0; ++i) {
                const E* hi = H.row(i);
                for (uint32_t t = 0; t < b; ++t) vals[t] = hi[out.pivots[t0 + t]];
                E* q = Q.data() + static_cast<size_t>(i) * b;
                for (uint32_t t = 0; t < b; ++t) {
                    const E qt = Tr::shr(Tr::mask(vals[t], k), out.pivot_v[t0 + t]);
                    q[t] = qt;
                    if (Tr::is_zero(qt)) continue;
                    nonzero = true;
                    const E* ht = H.row(t0 + t);
                    for (uint32_t t3 = t + 1; t3 < b; ++t3)
                        vals[t3] = Tr::sub(vals[t3], Tr::mul(qt, ht[out.pivots[t0 + t3]]));
                }
            }
            if (nonzero) Tr::gemm_sub(H.row(0) + c0, n, t0, Q.data(), b, H.row(t0) + c0, n, b, n - c0);
        }
        // The triangle inside the block.
        for (uint32_t t = t0 + 1; t < t1; ++t) {
            const uint32_t ct = out.pivots[t];
            for (uint32_t i = t0; i < t; ++i) {
                E* hi = H.row(i);
                const E q = Tr::shr(Tr::mask(hi[ct], k), out.pivot_v[t]);
                if (!Tr::is_zero(q)) Tr::axpy(hi + ct, H.row(t) + ct, q, n - ct);
            }
        }
    }
    for (E& x : H.a) x = Tr::mask(x, k);
    Counters d; d.closure_rows = closures;
    add_counters(d);
    return out;
}

// A is taken by value (pass an rvalue): the minimal-valuation LU runs in A's storage, the echelon rows
// stay in place (masked; the zero rows dropped by a size shrink) and are handed to howell_pass, so no
// second copy of the matrix is made (BIN-2 audit R2).
template<class Tr>
HowellForm<typename Tr::E> howell(Mat<typename Tr::E> A, uint32_t k) {
    using E = typename Tr::E;
    ENGINE_CHECK_MSG(k >= 1 && k <= Tr::kBits, "ring2k::howell: need 1 <= k <= the rung's width");
    const EchelonInfo e = echelon_blocked<Tr>(A, k, PivotRule::MinValuation, smithpar::route_panel_2k(params().panel, A.rows),
                                              0xffffffffu, smithpar::route_device_panel(params().panel));
    A.a.resize(static_cast<size_t>(e.rank) * A.cols);
    for (E& x : A.a) x = Tr::mask(x, k);
    A.rows = e.rank;
    return howell_pass<Tr>(std::move(A), k, params().reduce_block);
}

template<class Tr>
HowellForm<typename Tr::E> howell_naive(const Mat<typename Tr::E>& A, uint32_t k) {
    using E = typename Tr::E;
    ENGINE_CHECK_MSG(k >= 1 && k <= Tr::kBits, "ring2k::howell_naive: need 1 <= k <= the rung's width");
    Mat<E> M = A;
    for (E& x : M.a) x = Tr::mask(x, k);
    return howell_pass<Tr>(std::move(M), k, 1);
}

// The right kernel {x in (Z/2^k)^n : A x = 0} as a Howell form (ENGINE_API.md Sec. 13.6 rule).
template<class Tr>
HowellForm<typename Tr::E> kernel_module(const Mat<typename Tr::E>& A, uint32_t k) {
    using E = typename Tr::E;
    const uint32_t m = A.rows, n = A.cols;
    Mat<E> B(n, m + n);
    for (uint32_t i = 0; i < m; ++i)
        for (uint32_t j = 0; j < n; ++j) B.at(j, i) = A.at(i, j);
    for (uint32_t j = 0; j < n; ++j) B.at(j, m + j) = Tr::one();
    const HowellForm<E> h = howell<Tr>(std::move(B), k);
    HowellForm<E> out;
    out.k = k;
    uint32_t first = 0;
    while (first < h.pivots.size() && h.pivots[first] < m) ++first;
    const uint32_t r = static_cast<uint32_t>(h.pivots.size()) - first;
    out.H = Mat<E>(r, n);
    for (uint32_t t = 0; t < r; ++t) {
        out.pivots.push_back(h.pivots[first + t] - m);
        out.pivot_v.push_back(h.pivot_v[first + t]);
        for (uint32_t j = 0; j < n; ++j) out.H.at(t, j) = h.H.at(first + t, m + j);
    }
    return out;
}

// Global minimal-valuation pivoting (the prototype's "smith" mode): reference for smith().
template<class Tr>
SmithResult smith_naive(Mat<typename Tr::E> A, uint32_t k) {
    using E = typename Tr::E;
    SmithResult res;
    res.k = k;
    const uint32_t m = A.rows, n = A.cols;
    std::vector<uint32_t> colmap(n);
    for (uint32_t j = 0; j < n; ++j) colmap[j] = j;
    for (uint32_t r = 0; r < m && r < n; ++r) {
        uint32_t bi = m, bj = n, bv = k;
        for (uint32_t i = r; i < m && bv > 0; ++i)
            for (uint32_t j = r; j < n; ++j) {
                const uint32_t v = Tr::val(A.at(i, j), k);
                if (v < bv) { bv = v; bi = i; bj = j; if (v == 0) break; }
            }
        if (bi == m) break;
        detail::swap_rows(A, bi, r);
        if (bj != r) for (uint32_t i = 0; i < m; ++i) std::swap(A.at(i, bj), A.at(i, r));
        const E* P = A.row(r);
        const E ui = Tr::inv_odd(Tr::shr(P[r], bv));
        for (uint32_t i = r + 1; i < m; ++i) {
            E* R = A.row(i);
            if (Tr::zero_mod(R[r], k)) continue;
            Tr::axpy(R + r, P + r, Tr::mul(Tr::shr(Tr::mask(R[r], k), bv), ui), n - r);
        }
        // The pivot has minimal valuation in the whole block, so clearing its row by column operations
        // touches only the pivot row (its column is zero below now): the remaining block is unchanged.
        res.sigma.push_back(bv);
    }
    return res;
}

// ---------------------------------------------------------------------------
// The wide (rung-independent) interface and the adaptive rung.
// ---------------------------------------------------------------------------
enum class Rung : uint8_t { R0, R1, R2_16, R2_32, R3, R4, R5 };

struct RungChoice {
    Rung rung = Rung::R0;
    uint32_t words = 0;      // R5: N; otherwise the element's 64-bit word count (R0..R3: 1, R4: 2)
    uint32_t capacity = 0;   // the largest k this rung carries
};
std::string rung_label(const RungChoice& c);   // e.g. "R3(u64)", "R5(N=4)"
// The instantiated R5 word counts, ascending (the adaptive ladder's steps above R4).
const std::vector<uint32_t>& r5_word_counts();
// The smallest rung with capacity >= k (k = 1 -> R0).  ENGINE_CHECK: 1 <= k <= 4096.
RungChoice select_rung(uint32_t k);
// The next rung up (R0 -> R1 -> ... -> R4 -> R5 with the next instantiated N); capacity 0 past the top.
RungChoice next_rung(const RungChoice& c);
// Every rung able to carry k, ascending (R0 only for k = 1).
std::vector<RungChoice> rungs_for(uint32_t k);

// An m x n matrix over Z/2^k as nw = ceil(k / 64) little-endian words per entry (entries masked to k bits).
struct WideMatrix {
    uint32_t rows = 0, cols = 0, k = 0, nw = 0;
    std::vector<uint64_t> w;
    WideMatrix() = default;
    WideMatrix(uint32_t r, uint32_t c, uint32_t k_);
    uint64_t*       entry(uint32_t i, uint32_t j)       { return w.data() + (static_cast<size_t>(i) * cols + j) * nw; }
    const uint64_t* entry(uint32_t i, uint32_t j) const { return w.data() + (static_cast<size_t>(i) * cols + j) * nw; }
    void canonicalise();   // mask every entry to k bits
    bool operator==(const WideMatrix& o) const {
        return rows == o.rows && cols == o.cols && k == o.k && nw == o.nw && w == o.w;
    }
};

struct WideHowell {
    WideMatrix H;
    std::vector<uint32_t> pivots, pivot_v;
    uint32_t length() const { uint32_t s = 0; for (uint32_t v : pivot_v) s += H.k - v; return s; }
    bool operator==(const WideHowell& o) const { return H == o.H && pivots == o.pivots && pivot_v == o.pivot_v; }
};

template<class E>
Mat<E> from_wide(const WideMatrix& A) {
    Mat<E> M(A.rows, A.cols);
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint32_t j = 0; j < A.cols; ++j) M.at(i, j) = Traits<E>::mask(Traits<E>::from_words(A.entry(i, j), A.nw), A.k);
    return M;
}
template<class E>
WideMatrix to_wide(const Mat<E>& M, uint32_t k) {
    WideMatrix A(M.rows, M.cols, k);
    for (uint32_t i = 0; i < M.rows; ++i)
        for (uint32_t j = 0; j < M.cols; ++j) Traits<E>::to_words(Traits<E>::mask(M.at(i, j), k), A.entry(i, j), A.nw);
    return A;
}
template<class E>
WideHowell to_wide(const HowellForm<E>& h) {
    WideHowell w;
    w.H = to_wide(h.H, h.k);
    w.pivots = h.pivots;
    w.pivot_v = h.pivot_v;
    return w;
}

// Dispatch on a chosen rung (capacity >= A.k; R0 only for k = 1) or on select_rung(A.k).
SmithResult smith_on(const RungChoice& rung, const WideMatrix& A);
WideHowell  howell_on(const RungChoice& rung, const WideMatrix& A);
WideHowell  kernel_module_on(const RungChoice& rung, const WideMatrix& A);
// The unblocked references on a rung (validation use; not R0).
SmithResult smith_naive_on(const RungChoice& rung, const WideMatrix& A);
WideHowell  howell_naive_on(const RungChoice& rung, const WideMatrix& A);
SmithResult smith_wide(const WideMatrix& A);
WideHowell  howell_wide(const WideMatrix& A);
WideHowell  kernel_module_wide(const WideMatrix& A);

// The adaptive rung (plan Sec. 2.2's precision rule, generalised).  `produce(K)` returns the matrix mod
// 2^K (an exact integer matrix reduced, say).  The first run uses the full capacity K of
// select_rung(k_min) (a wider rung costs nothing extra in precision).  A run is RESOLVED when every
// Smith valuation is < K - guard and, if rank_hint >= 0, exactly rank_hint valuations are < K - guard
// (rank_hint = the rank over Q, e.g. from a run mod a large prime).  Otherwise the cell reruns one rung
// up (next_rung, K = its capacity), up to k_cap; every run is logged.
//
// CAUTION (BIN-2 audit F1 / R1): an invariant 2^sigma with sigma >= K is a zero diagonal entry over Z/2^K
// and is ABSENT from `sigma`.  Without rank_hint, `resolved` means only that no VISIBLE sigma lies within
// `guard` of K; an invariant >= K is indistinguishable from a zero entry.  `zeros` = min(rows, cols) -
// |sigma| counts those entries and is logged; the log line then reads "resolved (unverified: no rank hint)"
// when zeros > 0.  A caller that needs every invariant passes the rank over Q (BIN-3's degree-1 reader does).
struct AdaptiveSmith {
    SmithResult smith;          // of the last run
    RungChoice rung;            // of the last run
    uint32_t K = 0;
    uint32_t reruns = 0;
    uint32_t zeros = 0;         // min(rows, cols) - |sigma| of the last run: the zero diagonal entries mod 2^K
    bool resolved = false;
    bool verified = false;      // resolved AND checked against a rank hint (rank_hint >= 0)
    std::vector<std::string> log;   // one "RING2K_ADAPTIVE ..." line per run
};
AdaptiveSmith smith_adaptive(const std::function<WideMatrix(uint32_t K)>& produce, uint32_t k_min, uint32_t guard,
                             uint32_t k_cap, int64_t rank_hint = -1);

} // namespace ring2k
} // namespace engine

// (slot-24 PAR-GPU) the device route's host driver (its one call site is the dispatch in echelon_blocked).
#include "Ring2kDeviceLU.h"
