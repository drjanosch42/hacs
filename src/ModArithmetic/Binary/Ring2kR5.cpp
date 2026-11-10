// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// R5's SIMD kernels (slot-23 R5V): the structure-of-arrays limb GEMM / AXPY and the mask-carry row kernels
// of Ring2k.h's r5k::, one body (Ring2kR5Body.inc) compiled per kernel path (portable / AVX2 / AVX-512) and
// picked at run time by the native rungs' dispatch (active_kernel_path()).  Bit-identical to the scalar
// Traits<WordN<N>> bodies (the oracle; Ring2kR5SimdValidation).  Values never depend on the path.

#include "Ring2k.h"

#include <atomic>
#include <utility>
#include <vector>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#  define R5_HAVE_X86 1
#  include <immintrin.h>
#else
#  define R5_HAVE_X86 0
#endif

#if defined(__GNUC__) || defined(__clang__)
#  define R5_INL inline __attribute__((always_inline))
#else
#  define R5_INL inline
#endif

namespace engine {
namespace ring2k {

namespace {

// The limb width for N words: the widest B in 28 .. 24 with r5_chunk(N) >= 16 rank-one updates between carries.
constexpr uint32_t r5_nl(uint32_t N, uint32_t B) { return (64 * N + B - 1) / B; }
// The number of rank-one updates a 64-bit lane absorbs: TC NL (2^B - 1)^2 + 2^B < 2^64 with margin
// (TC NL <= 2^(64 - 2B) - NL; the + 2^B of a carried start value is far below the NL 2^(2B) slack).
constexpr uint32_t r5_tc(uint32_t N, uint32_t B) {
    const uint64_t cap = (uint64_t(1) << (64 - 2 * B)) / r5_nl(N, B);
    return cap > 1 ? static_cast<uint32_t>(cap - 1) : 0;
}
constexpr uint32_t r5_limb_bits(uint32_t N) {
    for (uint32_t B = 28; B >= 24; --B)
        if (r5_tc(N, B) >= 16) return B;
    return 24;
}
constexpr uint32_t r5_chunk(uint32_t N) {
    const uint32_t t = r5_tc(N, r5_limb_bits(N));
    return t > 256 ? 256 : t;
}
static_assert(r5_limb_bits(3) == 28 && r5_limb_bits(6) == 28 && r5_limb_bits(8) == 27 && r5_limb_bits(16) == 27,
              "R5V: the limb widths of DEEP's N");
static_assert(r5_chunk(16) >= 16 && r5_chunk(3) >= 16, "R5V: at least 16 updates between carries");

} // namespace

// ---------------------------------------------------------------------------
// The three bodies.
// ---------------------------------------------------------------------------
namespace {

#define R5_NS r5_portable
#define R5_VB 16
#if R5_HAVE_X86 && defined(__SSE2__)
#  define R5_MUL(a, b) ((V)_mm_mul_epu32((__m128i)(a), (__m128i)(b)))   // x86-64 baseline
#else
#  define R5_MUL(a, b) ((a) * (b))
#endif
#define R5_ROWMASKS 0
#include "Ring2kR5Body.inc"
#undef R5_NS
#undef R5_VB
#undef R5_MUL
#undef R5_ROWMASKS

#if R5_HAVE_X86
#pragma GCC push_options
#pragma GCC target("avx2")
#define R5_NS r5_avx2
#define R5_VB 32
#define R5_MUL(a, b) ((V)_mm256_mul_epu32((__m256i)(a), (__m256i)(b)))
#define R5_ROWMASKS 1
#define R5_LTMASK(a, b)                                                                                      \
    _mm256_movemask_pd((__m256d)_mm256_cmpgt_epi64((__m256i)((b) ^ (uint64_t(1) << 63)),                   \
                                                   (__m256i)((a) ^ (uint64_t(1) << 63))))
#define R5_EQMASK(a, b) _mm256_movemask_pd((__m256d)_mm256_cmpeq_epi64((__m256i)(a), (__m256i)(b)))
#define R5_LANEBITS(bits) ((((V){} + (bits)) >> (V){0, 1, 2, 3}) & 1)
#define R5_ADDBITS(v, bits) ((v) + R5_LANEBITS(bits))
#define R5_SUBBITS(v, bits) ((v) - R5_LANEBITS(bits))
#include "Ring2kR5Body.inc"
#undef R5_NS
#undef R5_VB
#undef R5_MUL
#undef R5_ROWMASKS
#undef R5_LTMASK
#undef R5_EQMASK
#undef R5_ADDBITS
#undef R5_SUBBITS
#undef R5_LANEBITS
#pragma GCC pop_options

#pragma GCC push_options
#pragma GCC target("avx512f,avx512bw,avx512dq,avx512vl")
#define R5_NS r5_avx512
#define R5_VB 64
#define R5_MUL(a, b) ((V)_mm512_mul_epu32((__m512i)(a), (__m512i)(b)))
#define R5_ROWMASKS 1
#define R5_LTMASK(a, b) static_cast<unsigned>(_mm512_cmplt_epu64_mask((__m512i)(a), (__m512i)(b)))
#define R5_EQMASK(a, b) static_cast<unsigned>(_mm512_cmpeq_epu64_mask((__m512i)(a), (__m512i)(b)))
#define R5_ADDBITS(v, bits) ((V)_mm512_mask_add_epi64((__m512i)(v), (__mmask8)(bits), (__m512i)(v), _mm512_set1_epi64(1)))
#define R5_SUBBITS(v, bits) ((V)_mm512_mask_sub_epi64((__m512i)(v), (__mmask8)(bits), (__m512i)(v), _mm512_set1_epi64(1)))
#include "Ring2kR5Body.inc"
#undef R5_NS
#undef R5_VB
#undef R5_MUL
#undef R5_ROWMASKS
#undef R5_LTMASK
#undef R5_EQMASK
#undef R5_ADDBITS
#undef R5_SUBBITS
#undef R5_LANEBITS
#pragma GCC pop_options
#endif

std::atomic<uint8_t> g_r5{static_cast<uint8_t>(R5Kernel::Simd)};
std::atomic<uint32_t> g_fault{0};   // validation-only fault injection (r5k::set_fault); 0 in production

// Calls f(std::integral_constant<uint32_t, N>{}) for the instantiated N.
template<class F>
void with_n(uint32_t N, F&& f) {
    switch (N) {
    case 3: f(std::integral_constant<uint32_t, 3>{}); return;
    case 4: f(std::integral_constant<uint32_t, 4>{}); return;
    case 5: f(std::integral_constant<uint32_t, 5>{}); return;
    case 6: f(std::integral_constant<uint32_t, 6>{}); return;
    case 8: f(std::integral_constant<uint32_t, 8>{}); return;
    case 10: f(std::integral_constant<uint32_t, 10>{}); return;
    case 12: f(std::integral_constant<uint32_t, 12>{}); return;
    case 16: f(std::integral_constant<uint32_t, 16>{}); return;
    default: break;
    }
    ENGINE_CHECK_MSG(false, "ring2k::r5k: no SIMD body for this N (r5_simd_instantiated)");
}

// The path's entry points (out of line, so each carries its ISA).
#define R5_ENTRY(NS, ATTR)                                                                                       \
    template<uint32_t N> ATTR void NS##_gemm(uint64_t* C, size_t ldc, uint32_t mr, const uint64_t* L, size_t ldl,  \
                                             const uint64_t* U, size_t ldu, uint32_t np, uint32_t nc, uint32_t fl) { \
        NS::gemm_sub<N>(C, ldc, mr, L, ldl, U, ldu, np, nc, fl);                                                 \
    }                                                                                                            \
    template<uint32_t N> ATTR void NS##_axpy(uint64_t* r, const uint64_t* p, const uint64_t* f, size_t n,           \
                                             uint32_t fl) {                                                      \
        NS::axpy<N>(r, p, f, n, fl);                                                                             \
    }                                                                                                            \
    template<uint32_t N, int OP> ATTR void NS##_row(uint64_t* d, const uint64_t* s, size_t n, uint32_t fl) {     \
        NS::row_op<N, OP>(d, s, n, fl);                                                                          \
    }
R5_ENTRY(r5_portable, )
#if R5_HAVE_X86
R5_ENTRY(r5_avx2, __attribute__((target("avx2"))))
R5_ENTRY(r5_avx512, __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl"))))
#endif
#undef R5_ENTRY

template<int OP>
void row_dispatch(uint32_t N, uint64_t* d, const uint64_t* s, size_t n) {
    if (n == 0) return;
    const KernelPath p = active_kernel_path();
    const uint32_t fl = g_fault.load(std::memory_order_relaxed);
    with_n(N, [&](auto nt) {
        constexpr uint32_t NN = decltype(nt)::value;
#if R5_HAVE_X86
        if (p == KernelPath::Avx512) { r5_avx512_row<NN, OP>(d, s, n, fl); return; }
        if (p == KernelPath::Avx2) { r5_avx2_row<NN, OP>(d, s, n, fl); return; }
#endif
        (void)p;
        r5_portable_row<NN, OP>(d, s, n, fl);
    });
}

} // namespace

void set_r5_kernel(R5Kernel k) { g_r5.store(static_cast<uint8_t>(k), std::memory_order_relaxed); }
R5Kernel r5_kernel() { return static_cast<R5Kernel>(g_r5.load(std::memory_order_relaxed)); }
const char* r5_kernel_name(R5Kernel k) { return k == R5Kernel::Simd ? "simd" : "scalar"; }

namespace r5k {

void gemm_sub(uint32_t N, uint64_t* C, size_t ldc, uint32_t mrows, const uint64_t* L, size_t ldl, const uint64_t* U,
              size_t ldu, uint32_t np, uint32_t ncols) {
    if (mrows == 0 || np == 0 || ncols == 0) return;
    const KernelPath p = active_kernel_path();
    const uint32_t fl = g_fault.load(std::memory_order_relaxed);
    with_n(N, [&](auto nt) {
        constexpr uint32_t NN = decltype(nt)::value;
#if R5_HAVE_X86
        if (p == KernelPath::Avx512) { r5_avx512_gemm<NN>(C, ldc, mrows, L, ldl, U, ldu, np, ncols, fl); return; }
        if (p == KernelPath::Avx2) { r5_avx2_gemm<NN>(C, ldc, mrows, L, ldl, U, ldu, np, ncols, fl); return; }
#endif
        (void)p;
        r5_portable_gemm<NN>(C, ldc, mrows, L, ldl, U, ldu, np, ncols, fl);
    });
}
void axpy(uint32_t N, uint64_t* r, const uint64_t* p, const uint64_t* f, size_t n) {
    if (n == 0) return;
    ENGINE_CHECK_MSG(n <= 0xffffffffu, "ring2k::r5k::axpy: n < 2^32");
    const KernelPath kp = active_kernel_path();
    const uint32_t fl = g_fault.load(std::memory_order_relaxed);
    with_n(N, [&](auto nt) {
        constexpr uint32_t NN = decltype(nt)::value;
#if R5_HAVE_X86
        if (kp == KernelPath::Avx512) { r5_avx512_axpy<NN>(r, p, f, n, fl); return; }
        if (kp == KernelPath::Avx2) { r5_avx2_axpy<NN>(r, p, f, n, fl); return; }
#endif
        (void)kp;
        r5_portable_axpy<NN>(r, p, f, n, fl);
    });
}
void add_row(uint32_t N, uint64_t* d, const uint64_t* s, size_t n) { row_dispatch<0>(N, d, s, n); }
void sub_row(uint32_t N, uint64_t* d, const uint64_t* s, size_t n) { row_dispatch<1>(N, d, s, n); }
void neg_row(uint32_t N, uint64_t* d, size_t n) { row_dispatch<2>(N, d, nullptr, n); }

void set_fault(uint32_t f) {
    ENGINE_CHECK_MSG(f <= 3, "ring2k::r5k::set_fault: 0 (none), 1, 2 or 3");
    g_fault.store(f, std::memory_order_relaxed);
}
uint32_t fault() { return g_fault.load(std::memory_order_relaxed); }

Layout layout(uint32_t N, KernelPath p) {
    Layout l;
    if (!r5_simd_instantiated(N)) return l;
    l.B = r5_limb_bits(N);
    l.NL = r5_nl(N, l.B);
    l.TC = r5_chunk(N);
    const KernelPath q = p == KernelPath::Auto ? active_kernel_path() : p;
    l.lanes = q == KernelPath::Avx512 ? 8 : q == KernelPath::Avx2 ? 4 : 2;
    return l;
}

} // namespace r5k
} // namespace ring2k
} // namespace engine
