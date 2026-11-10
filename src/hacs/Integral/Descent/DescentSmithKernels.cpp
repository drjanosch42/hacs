// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// PAR-D2: the kernels of the blocked Smith form (see DescentSmithKernels.h).
#include "DescentSmithKernels.h"

#include <algorithm>
#if defined(__x86_64__)
#include <immintrin.h>
#endif

#include "../../../ModArithmetic/EngineCheck.h"
#include "../../../ModArithmetic/Kernels.h"   // (S24 CL-5) CpuFeatures.h, the ENGINE_KERNEL ceiling

namespace integral_descent {
namespace maps {
namespace smithk {

KZ MakeKZ(const ZMod& Z) {
    ENGINE_CHECK_MSG(Z.m < (uint64_t(1) << 31), "smithk: ell^k < 2^31");
    KZ z;
    z.two = Z.two;
    z.k = Z.k;
    z.ell = Z.ell;
    z.m = static_cast<uint32_t>(Z.m);
    z.mask = Z.mask;
    if (!z.two) {
        uint32_t x = z.ell;   // Newton: x = ell^{-1} mod 2^32
        for (int i = 0; i < 5; ++i) x *= 2u - z.ell * x;
        ENGINE_CHECK(x * z.ell == 1u);
        z.inv32 = x;
        z.lim = 0xFFFFFFFFu / z.ell;
        uint32_t y = z.m;     // m^{-1} mod 2^32
        for (int i = 0; i < 5; ++i) y *= 2u - z.m * y;
        ENGINE_CHECK(y * z.m == 1u);
        z.mprime = 0u - y;
        z.r1 = static_cast<uint32_t>((uint64_t(1) << 32) % z.m);
    }
    return z;
}

// ---------------------------------------------------------------------------------------------------------------------------
// scalar (the reference)
// ---------------------------------------------------------------------------------------------------------------------------
namespace {

void upd_scalar(const KZ& z, uint32_t* row, const uint32_t* const* U, const uint32_t* f, uint32_t nf, uint32_t n) {
    if (z.two) {
        uint32_t x = 0;
        for (; x + 4 <= nf; x += 4) {
            const uint32_t f0 = f[x], f1 = f[x + 1], f2 = f[x + 2], f3 = f[x + 3];
            const uint32_t *u0 = U[x], *u1 = U[x + 1], *u2 = U[x + 2], *u3 = U[x + 3];
            for (uint32_t j = 0; j < n; ++j) row[j] -= f0 * u0[j] + f1 * u1[j] + f2 * u2[j] + f3 * u3[j];
        }
        for (; x < nf; ++x) {
            const uint32_t f0 = f[x];
            const uint32_t* u0 = U[x];
            for (uint32_t j = 0; j < n; ++j) row[j] -= f0 * u0[j];
        }
        const uint32_t mask = z.mask;
        for (uint32_t j = 0; j < n; ++j) row[j] &= mask;
        return;
    }
    const uint64_t m = z.m;
    uint32_t x = 0;
    for (; x + 3 <= nf; x += 3) {
        const uint64_t f0 = f[x], f1 = f[x + 1], f2 = f[x + 2];
        const uint32_t *u0 = U[x], *u1 = U[x + 1], *u2 = U[x + 2];
        for (uint32_t j = 0; j < n; ++j) {
            const uint32_t t = static_cast<uint32_t>((f0 * u0[j] + f1 * u1[j] + f2 * u2[j]) % m);
            row[j] = row[j] >= t ? row[j] - t : static_cast<uint32_t>(row[j] + m - t);
        }
    }
    for (; x < nf; ++x) {
        const uint64_t f0 = f[x];
        const uint32_t* u0 = U[x];
        for (uint32_t j = 0; j < n; ++j) {
            if (!u0[j]) continue;
            const uint32_t t = static_cast<uint32_t>(f0 * u0[j] % m);
            row[j] = row[j] >= t ? row[j] - t : static_cast<uint32_t>(row[j] + m - t);
        }
    }
}

uint32_t first_unit_scalar(const KZ& z, const uint32_t* a, uint32_t n) {
    if (z.two) {
        for (uint32_t j = 0; j < n; ++j)
            if (a[j] & 1u) return j;
        return n;
    }
    for (uint32_t j = 0; j < n; ++j)
        if (a[j] * z.inv32 > z.lim) return j;
    return n;
}

// the tail [n0, n) of upd: the same exact map, scalar
void upd_tail(const KZ& z, uint32_t* row, const uint32_t* const* U, const uint32_t* f, uint32_t nf, uint32_t n0, uint32_t n) {
    if (n0 >= n) return;
    const uint32_t* Ut[64];
    uint32_t x0 = 0;
    while (x0 < nf) {
        const uint32_t cnt = std::min<uint32_t>(64, nf - x0);
        for (uint32_t q = 0; q < cnt; ++q) Ut[q] = U[x0 + q] + n0;
        upd_scalar(z, row + n0, Ut, f + x0, cnt, n - n0);
        x0 += cnt;
    }
}

// ---------------------------------------------------------------------------------------------------------------------------
// SIMD bodies (x86-64 only; runtime dispatch).  ell = 2: 32-bit mullo / sub, masked at the end (exact mod 2^32).  Odd ell:
// Montgomery: f' = f 2^32 mod m, T = f'_0 u_0 + f'_1 u_1 < 2 m^2 < m 2^32, REDC(T) = (T + ((T mod 2^32) m') m) / 2^32 in [0, 2m),
// = f_0 u_0 + f_1 u_1 mod m after one conditional subtraction (min_u32(v, v - m)); row - x by d = row - x, min_u32(d, d + m).
// ---------------------------------------------------------------------------------------------------------------------------
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define SMITHK_HAVE_X86 1
#define SMITHK_AVX2 __attribute__((target("avx2")))
#define SMITHK_AVX512 __attribute__((target("avx512f")))

inline uint32_t to_mont(const KZ& z, uint32_t f) { return static_cast<uint32_t>(uint64_t(f) * z.r1 % z.m); }

SMITHK_AVX512 inline __m512i redc512(__m512i T, __m512i mp, __m512i mv) {   // T < m 2^32 (64-bit lanes) -> [0, 2m) in the low dwords
    const __m512i q = _mm512_mul_epu32(T, mp);
    return _mm512_srli_epi64(_mm512_add_epi64(T, _mm512_mul_epu32(q, mv)), 32);
}
SMITHK_AVX2 inline __m256i redc256(__m256i T, __m256i mp, __m256i mv) {
    const __m256i q = _mm256_mul_epu32(T, mp);
    return _mm256_srli_epi64(_mm256_add_epi64(T, _mm256_mul_epu32(q, mv)), 32);
}

SMITHK_AVX512 void upd_avx512(const KZ& z, uint32_t* row, const uint32_t* const* U, const uint32_t* f, uint32_t nf, uint32_t n) {
    const uint32_t n16 = n & ~15u;
    if (n16 == 0) return upd_scalar(z, row, U, f, nf, n);
    if (z.two) {
        const __m512i mask = _mm512_set1_epi32(static_cast<int>(z.mask));
        uint32_t x = 0;
        for (; x + 4 <= nf; x += 4) {
            const __m512i f0 = _mm512_set1_epi32(static_cast<int>(f[x])), f1 = _mm512_set1_epi32(static_cast<int>(f[x + 1]));
            const __m512i f2 = _mm512_set1_epi32(static_cast<int>(f[x + 2])), f3 = _mm512_set1_epi32(static_cast<int>(f[x + 3]));
            const uint32_t *u0 = U[x], *u1 = U[x + 1], *u2 = U[x + 2], *u3 = U[x + 3];
            for (uint32_t j = 0; j < n16; j += 16) {
                const __m512i a = _mm512_add_epi32(_mm512_mullo_epi32(f0, _mm512_loadu_si512(u0 + j)), _mm512_mullo_epi32(f1, _mm512_loadu_si512(u1 + j)));
                const __m512i b = _mm512_add_epi32(_mm512_mullo_epi32(f2, _mm512_loadu_si512(u2 + j)), _mm512_mullo_epi32(f3, _mm512_loadu_si512(u3 + j)));
                _mm512_storeu_si512(row + j, _mm512_sub_epi32(_mm512_loadu_si512(row + j), _mm512_add_epi32(a, b)));
            }
        }
        for (; x < nf; ++x) {
            const __m512i f0 = _mm512_set1_epi32(static_cast<int>(f[x]));
            const uint32_t* u0 = U[x];
            for (uint32_t j = 0; j < n16; j += 16)
                _mm512_storeu_si512(row + j, _mm512_sub_epi32(_mm512_loadu_si512(row + j), _mm512_mullo_epi32(f0, _mm512_loadu_si512(u0 + j))));
        }
        for (uint32_t j = 0; j < n16; j += 16) _mm512_storeu_si512(row + j, _mm512_and_si512(_mm512_loadu_si512(row + j), mask));
    } else {
        const __m512i mv = _mm512_set1_epi64(static_cast<long long>(z.m));
        const __m512i m32 = _mm512_set1_epi32(static_cast<int>(z.m));
        const __m512i mp = _mm512_set1_epi64(static_cast<long long>(z.mprime));
        for (uint32_t x = 0; x < nf; x += 2) {
            const bool two_terms = x + 1 < nf;
            const __m512i f0 = _mm512_set1_epi64(static_cast<long long>(to_mont(z, f[x])));
            const __m512i f1 = _mm512_set1_epi64(static_cast<long long>(two_terms ? to_mont(z, f[x + 1]) : 0u));
            const uint32_t* u0 = U[x];
            const uint32_t* u1 = two_terms ? U[x + 1] : U[x];
            for (uint32_t j = 0; j < n16; j += 16) {
                const __m512i a = _mm512_loadu_si512(u0 + j), b = _mm512_loadu_si512(u1 + j);
                const __m512i Te = _mm512_add_epi64(_mm512_mul_epu32(f0, a), _mm512_mul_epu32(f1, b));
                const __m512i To = _mm512_add_epi64(_mm512_mul_epu32(f0, _mm512_srli_epi64(a, 32)), _mm512_mul_epu32(f1, _mm512_srli_epi64(b, 32)));
                __m512i v = _mm512_mask_blend_epi32(0xAAAA, redc512(Te, mp, mv), _mm512_slli_epi64(redc512(To, mp, mv), 32));   // [0, 2m)
                v = _mm512_min_epu32(v, _mm512_sub_epi32(v, m32));                                                            // [0, m)
                const __m512i d = _mm512_sub_epi32(_mm512_loadu_si512(row + j), v);
                _mm512_storeu_si512(row + j, _mm512_min_epu32(d, _mm512_add_epi32(d, m32)));
            }
        }
    }
    upd_tail(z, row, U, f, nf, n16, n);
}

SMITHK_AVX2 inline __m256i ld8(const uint32_t* p) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)); }
SMITHK_AVX2 inline void st8(uint32_t* p, __m256i v) { _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v); }

SMITHK_AVX2 void upd_avx2(const KZ& z, uint32_t* row, const uint32_t* const* U, const uint32_t* f, uint32_t nf, uint32_t n) {
    const uint32_t n8 = n & ~7u;
    if (n8 == 0) return upd_scalar(z, row, U, f, nf, n);
    if (z.two) {
        const __m256i mask = _mm256_set1_epi32(static_cast<int>(z.mask));
        uint32_t x = 0;
        for (; x + 4 <= nf; x += 4) {
            const __m256i f0 = _mm256_set1_epi32(static_cast<int>(f[x])), f1 = _mm256_set1_epi32(static_cast<int>(f[x + 1]));
            const __m256i f2 = _mm256_set1_epi32(static_cast<int>(f[x + 2])), f3 = _mm256_set1_epi32(static_cast<int>(f[x + 3]));
            const uint32_t *u0 = U[x], *u1 = U[x + 1], *u2 = U[x + 2], *u3 = U[x + 3];
            for (uint32_t j = 0; j < n8; j += 8) {
                const __m256i a = _mm256_add_epi32(_mm256_mullo_epi32(f0, ld8(u0 + j)), _mm256_mullo_epi32(f1, ld8(u1 + j)));
                const __m256i b = _mm256_add_epi32(_mm256_mullo_epi32(f2, ld8(u2 + j)), _mm256_mullo_epi32(f3, ld8(u3 + j)));
                st8(row + j, _mm256_sub_epi32(ld8(row + j), _mm256_add_epi32(a, b)));
            }
        }
        for (; x < nf; ++x) {
            const __m256i f0 = _mm256_set1_epi32(static_cast<int>(f[x]));
            const uint32_t* u0 = U[x];
            for (uint32_t j = 0; j < n8; j += 8) st8(row + j, _mm256_sub_epi32(ld8(row + j), _mm256_mullo_epi32(f0, ld8(u0 + j))));
        }
        for (uint32_t j = 0; j < n8; j += 8) st8(row + j, _mm256_and_si256(ld8(row + j), mask));
    } else {
        const __m256i mv = _mm256_set1_epi64x(static_cast<long long>(z.m));
        const __m256i m32 = _mm256_set1_epi32(static_cast<int>(z.m));
        const __m256i mp = _mm256_set1_epi64x(static_cast<long long>(z.mprime));
        for (uint32_t x = 0; x < nf; x += 2) {
            const bool two_terms = x + 1 < nf;
            const __m256i f0 = _mm256_set1_epi64x(static_cast<long long>(to_mont(z, f[x])));
            const __m256i f1 = _mm256_set1_epi64x(static_cast<long long>(two_terms ? to_mont(z, f[x + 1]) : 0u));
            const uint32_t* u0 = U[x];
            const uint32_t* u1 = two_terms ? U[x + 1] : U[x];
            for (uint32_t j = 0; j < n8; j += 8) {
                const __m256i a = ld8(u0 + j), b = ld8(u1 + j);
                const __m256i Te = _mm256_add_epi64(_mm256_mul_epu32(f0, a), _mm256_mul_epu32(f1, b));
                const __m256i To = _mm256_add_epi64(_mm256_mul_epu32(f0, _mm256_srli_epi64(a, 32)), _mm256_mul_epu32(f1, _mm256_srli_epi64(b, 32)));
                __m256i v = _mm256_blend_epi32(redc256(Te, mp, mv), _mm256_slli_epi64(redc256(To, mp, mv), 32), 0xAA);
                v = _mm256_min_epu32(v, _mm256_sub_epi32(v, m32));
                const __m256i d = _mm256_sub_epi32(ld8(row + j), v);
                st8(row + j, _mm256_min_epu32(d, _mm256_add_epi32(d, m32)));
            }
        }
    }
    upd_tail(z, row, U, f, nf, n8, n);
}

SMITHK_AVX512 uint32_t first_unit_avx512(const KZ& z, const uint32_t* a, uint32_t n) {
    uint32_t j = 0;
    if (z.two) {
        const __m512i one = _mm512_set1_epi32(1);
        for (; j + 16 <= n; j += 16) {
            const __mmask16 k = _mm512_test_epi32_mask(_mm512_loadu_si512(a + j), one);
            if (k) return j + static_cast<uint32_t>(__builtin_ctz(k));
        }
    } else {
        const __m512i inv = _mm512_set1_epi32(static_cast<int>(z.inv32)), lim = _mm512_set1_epi32(static_cast<int>(z.lim));
        for (; j + 16 <= n; j += 16) {
            const __mmask16 k = _mm512_cmpgt_epu32_mask(_mm512_mullo_epi32(_mm512_loadu_si512(a + j), inv), lim);
            if (k) return j + static_cast<uint32_t>(__builtin_ctz(k));
        }
    }
    return j + first_unit_scalar(z, a + j, n - j);
}
SMITHK_AVX2 uint32_t first_unit_avx2(const KZ& z, const uint32_t* a, uint32_t n) {
    uint32_t j = 0;
    if (z.two) {
        const __m256i one = _mm256_set1_epi32(1);
        for (; j + 8 <= n; j += 8) {
            const __m256i v = _mm256_and_si256(ld8(a + j), one);
            const int k = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(v, one)));
            if (k) return j + static_cast<uint32_t>(__builtin_ctz(static_cast<unsigned>(k)));
        }
    } else {
        const __m256i inv = _mm256_set1_epi32(static_cast<int>(z.inv32)), lim1 = _mm256_set1_epi32(static_cast<int>(z.lim + 1));
        for (; j + 8 <= n; j += 8) {
            const __m256i y = _mm256_mullo_epi32(ld8(a + j), inv);
            const int k = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(_mm256_max_epu32(y, lim1), y)));   // y >= lim + 1
            if (k) return j + static_cast<uint32_t>(__builtin_ctz(static_cast<unsigned>(k)));
        }
    }
    return j + first_unit_scalar(z, a + j, n - j);
}

bool cpu_avx2() { return engine::cpu::avx2(); }   // (S24 CL-5) CpuFeatures.h
bool cpu_avx512() { return engine::cpu::avx512f(); }
#endif

}  // namespace

int Clamp(int simd) {
#ifdef SMITHK_HAVE_X86
    const int best = cpu_avx512() ? 2 : (cpu_avx2() ? 1 : 0);
    if (simd < 0) {   // (S24 CL-5) auto: the best body at or below the ENGINE_KERNEL ceiling (an explicit request: the CPU only)
        const engine::cpu::IsaPath p = engine::kernels::isa_auto(cpu_avx2(), cpu_avx512());
        return p == engine::cpu::IsaPath::Avx512 ? 2 : p == engine::cpu::IsaPath::Avx2 ? 1 : 0;
    }
#else
    const int best = 0;
    if (simd < 0) return best;
#endif
    return std::min(simd, best);
}
const char* PathName(int path) { return path == 2 ? "avx512" : (path == 1 ? "avx2" : "scalar"); }
Kernels Select(int simd) {
    const int p = Clamp(simd);
    Kernels K;
    K.upd = &upd_scalar;
    K.first_unit = &first_unit_scalar;
    K.path = 0;
#ifdef SMITHK_HAVE_X86
    if (p == 2) {
        K.upd = &upd_avx512;
        K.first_unit = &first_unit_avx512;
        K.path = 2;
    } else if (p == 1) {
        K.upd = &upd_avx2;
        K.first_unit = &first_unit_avx2;
        K.path = 1;
    }
#endif
    return K;
}

}  // namespace smithk
}  // namespace maps
}  // namespace integral_descent
