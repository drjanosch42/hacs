// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// RnsKernels.cpp -- garner_step: portable, AVX2 and AVX-512 bodies (RnsKernels.h; slot-25 EXZ2-P2).

#include "RnsKernels.h"

#include "EngineCheck.h"
#include "Kernels.h"   // kernels::isa_auto (the ENGINE_KERNEL ceiling)
#include "ModCtx.h"

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#  include <immintrin.h>
#  define RNS_HAVE_X86 1
#  define RNS_AVX512_FN __attribute__((target("avx512f")))
#  define RNS_AVX2_FN __attribute__((target("avx2")))
#else
#  define RNS_HAVE_X86 0
#  define RNS_AVX512_FN
#  define RNS_AVX2_FN
#endif

namespace engine {
namespace rns {

namespace {
uint32_t PowMod(uint64_t a, uint64_t e, uint64_t m) {
    uint64_t r = 1 % m;
    a %= m;
    while (e) {
        if (e & 1) r = r * a % m;
        a = a * a % m;
        e >>= 1;
    }
    return static_cast<uint32_t>(r);
}
// (Fix round 1, audit F11) the Shoup products are ModCtx's own (shoup_precompute / shoup_mul_lazy: a * w mod q in [0, 2q) for any
// a < 2^32, q <= 2^31), not a copy of its formulas; the vector bodies below are the same formula lane-wise.
inline uint32_t Canon(uint32_t x, uint32_t q) { return x >= q ? x - q : x; }

// one entry, portable (the reference arithmetic of every body)
inline uint32_t StepOne(const GarnerPlan& P, const ModCtx<uint32_t>& ctx, uint32_t k, const uint32_t* const* dg, const uint32_t* res, size_t e) {
    const uint32_t q = P.primes[k];
    const size_t L = P.primes.size();
    uint32_t acc = 0;
    for (uint32_t i = 0; i < k; ++i) {
        const uint32_t t = Canon(ctx.shoup_mul_lazy(dg[i][e], P.coef[k * L + i], P.coef_shoup[k * L + i]), q);
        acc = Canon(acc + t, q);
    }
    const uint32_t r = res[e];
    const uint32_t diff = r >= acc ? r - acc : r + (q - acc);
    return Canon(ctx.shoup_mul_lazy(diff, P.inv[k], P.inv_shoup[k]), q);
}

void StepPortable(const GarnerPlan& P, uint32_t k, const uint32_t* const* dg, const uint32_t* res, uint32_t* out, size_t e0, size_t E) {
    if (e0 >= E) return;
    const ModCtx<uint32_t> ctx(P.primes[k]);
    for (size_t e = e0; e < E; ++e) out[e] = StepOne(P, ctx, k, dg, res, e);
}

#if RNS_HAVE_X86
RNS_AVX2_FN inline __m256i ShoupLazy8(__m256i a, __m256i w, __m256i wsh, __m256i q) {
    const __m256i p0 = _mm256_mul_epu32(a, wsh);
    const __m256i p1 = _mm256_mul_epu32(_mm256_srli_epi64(a, 32), wsh);
    const __m256i qh = _mm256_blend_epi32(_mm256_srli_epi64(p0, 32), p1, 0xAA);
    return _mm256_sub_epi32(_mm256_mullo_epi32(a, w), _mm256_mullo_epi32(qh, q));
}
RNS_AVX2_FN inline __m256i Canon8(__m256i x, __m256i q) { return _mm256_min_epu32(x, _mm256_sub_epi32(x, q)); }
RNS_AVX2_FN void StepAvx2(const GarnerPlan& P, uint32_t k, const uint32_t* const* dg, const uint32_t* res, uint32_t* out, size_t E) {
    const uint32_t qs = P.primes[k];
    const size_t L = P.primes.size();
    const __m256i q = _mm256_set1_epi32(static_cast<int>(qs));
    const __m256i iv = _mm256_set1_epi32(static_cast<int>(P.inv[k])), ivs = _mm256_set1_epi32(static_cast<int>(P.inv_shoup[k]));
    size_t e = 0;
    for (; e + 8 <= E; e += 8) {
        __m256i acc = _mm256_setzero_si256();
        for (uint32_t i = 0; i < k; ++i) {
            const __m256i d = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dg[i] + e));
            const __m256i w = _mm256_set1_epi32(static_cast<int>(P.coef[k * L + i]));
            const __m256i ws = _mm256_set1_epi32(static_cast<int>(P.coef_shoup[k * L + i]));
            acc = Canon8(_mm256_add_epi32(acc, Canon8(ShoupLazy8(d, w, ws, q), q)), q);
        }
        const __m256i r = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(res + e));
        // diff = r - acc mod q = canon(r + q - acc), r, acc < q
        const __m256i diff = Canon8(_mm256_sub_epi32(_mm256_add_epi32(r, q), acc), q);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + e), Canon8(ShoupLazy8(diff, iv, ivs, q), q));
    }
    StepPortable(P, k, dg, res, out, e, E);
}

RNS_AVX512_FN inline __m512i ShoupLazy16(__m512i a, __m512i w, __m512i wsh, __m512i q) {
    const __m512i p0 = _mm512_mul_epu32(a, wsh);
    const __m512i p1 = _mm512_mul_epu32(_mm512_srli_epi64(a, 32), wsh);
    const __m512i qh = _mm512_mask_blend_epi32(static_cast<__mmask16>(0xAAAA), _mm512_srli_epi64(p0, 32), p1);
    return _mm512_sub_epi32(_mm512_mullo_epi32(a, w), _mm512_mullo_epi32(qh, q));
}
RNS_AVX512_FN inline __m512i Canon16(__m512i x, __m512i q) { return _mm512_min_epu32(x, _mm512_sub_epi32(x, q)); }
RNS_AVX512_FN void StepAvx512(const GarnerPlan& P, uint32_t k, const uint32_t* const* dg, const uint32_t* res, uint32_t* out, size_t E) {
    const uint32_t qs = P.primes[k];
    const size_t L = P.primes.size();
    const __m512i q = _mm512_set1_epi32(static_cast<int>(qs));
    const __m512i iv = _mm512_set1_epi32(static_cast<int>(P.inv[k])), ivs = _mm512_set1_epi32(static_cast<int>(P.inv_shoup[k]));
    size_t e = 0;
    for (; e + 16 <= E; e += 16) {
        __m512i acc = _mm512_setzero_si512();
        for (uint32_t i = 0; i < k; ++i) {
            const __m512i d = _mm512_loadu_si512(reinterpret_cast<const void*>(dg[i] + e));
            const __m512i w = _mm512_set1_epi32(static_cast<int>(P.coef[k * L + i]));
            const __m512i ws = _mm512_set1_epi32(static_cast<int>(P.coef_shoup[k * L + i]));
            acc = Canon16(_mm512_add_epi32(acc, Canon16(ShoupLazy16(d, w, ws, q), q)), q);
        }
        const __m512i r = _mm512_loadu_si512(reinterpret_cast<const void*>(res + e));
        const __m512i diff = Canon16(_mm512_sub_epi32(_mm512_add_epi32(r, q), acc), q);
        _mm512_storeu_si512(reinterpret_cast<void*>(out + e), Canon16(ShoupLazy16(diff, iv, ivs, q), q));
    }
    StepPortable(P, k, dg, res, out, e, E);
}
#endif
}  // namespace

GarnerPlan::GarnerPlan(const std::vector<uint32_t>& p) : primes(p) {
    const size_t L = p.size();
    coef.assign(L * L, 0);
    coef_shoup.assign(L * L, 0);
    inv.assign(L, 0);
    inv_shoup.assign(L, 0);
    for (size_t k = 0; k < L; ++k) {
        const uint32_t q = p[k];
        ENGINE_CHECK_MSG(q > 2 && q <= (uint32_t(1) << 31) && (q & 1u), "GarnerPlan: every modulus odd, 2 < q <= 2^31");
        const ModCtx<uint32_t> ctx(q);
        for (size_t i = 0; i < k; ++i) ENGINE_CHECK_MSG(p[i] != q, "GarnerPlan: the moduli must be distinct");
        uint64_t pw = 1;
        for (size_t i = 0; i < k; ++i) {
            coef[k * L + i] = static_cast<uint32_t>(pw);
            coef_shoup[k * L + i] = ctx.shoup_precompute(static_cast<uint32_t>(pw));
            pw = pw * (p[i] % q) % q;
        }
        ENGINE_CHECK_MSG(pw != 0, "GarnerPlan: the moduli are not coprime");
        inv[k] = PowMod(pw, q - 2, q);   // q prime (CrtPrimes); the plan is for primes only
        ENGINE_CHECK_MSG(uint64_t(inv[k]) * pw % q == 1, "GarnerPlan: a modulus is not prime");
        inv_shoup[k] = ctx.shoup_precompute(inv[k]);
    }
}

bool garner_avx2_available() { return RNS_HAVE_X86 && cpu::avx2(); }
bool garner_avx512_available() { return RNS_HAVE_X86 && cpu::avx512f(); }

IsaPath garner_resolved_path(IsaPath path) {
    if (path == IsaPath::Auto) return kernels::isa_auto(garner_avx2_available(), garner_avx512_available());
    if (path == IsaPath::Avx512 && !garner_avx512_available()) return garner_avx2_available() ? IsaPath::Avx2 : IsaPath::Portable;
    if (path == IsaPath::Avx2 && !garner_avx2_available()) return IsaPath::Portable;
    return path;
}

void garner_step(const GarnerPlan& plan, uint32_t k, const uint32_t* const* digits, const uint32_t* res, uint32_t* out, size_t E, IsaPath path) {
    ENGINE_CHECK_MSG(k < plan.primes.size(), "garner_step: k beyond the plan");
    const IsaPath p = garner_resolved_path(path);
#if RNS_HAVE_X86
    if (p == IsaPath::Avx512) return StepAvx512(plan, k, digits, res, out, E);
    if (p == IsaPath::Avx2) return StepAvx2(plan, k, digits, res, out, E);
#else
    (void)p;
#endif
    StepPortable(plan, k, digits, res, out, 0, E);
}

void garner_step_oracle(const GarnerPlan& plan, uint32_t k, const uint32_t* const* digits, const uint32_t* res, uint32_t* out, size_t E) {
    const uint64_t q = plan.primes[k];
    for (size_t e = 0; e < E; ++e) {
        // acc = sum_{i<k} d_i prod_{j<i} q_j  mod q, by Horner from the top digit (independent of the plan's coefficient table)
        uint64_t acc = 0;
        for (uint32_t i = k; i-- > 0;) acc = (acc * (plan.primes[i] % q) + digits[i][e] % q) % q;
        const uint64_t diff = (res[e] % q + q - acc) % q;
        uint64_t pw = 1;
        for (uint32_t j = 0; j < k; ++j) pw = pw * (plan.primes[j] % q) % q;
        out[e] = static_cast<uint32_t>(diff * PowMod(pw, q - 2, q) % q);
    }
}

void garner_digits(const GarnerPlan& plan, const std::vector<const uint32_t*>& res, size_t E, std::vector<std::vector<uint32_t>>& digits,
                   IsaPath path) {
    const size_t L = plan.primes.size();
    ENGINE_CHECK_MSG(res.size() == L, "garner_digits: one residue plane per prime");
    digits.assign(L, std::vector<uint32_t>(E, 0));
    std::vector<const uint32_t*> dp(L, nullptr);
    for (size_t k = 0; k < L; ++k) {
        for (size_t e = 0; e < E; ++e) ENGINE_DEBUG_CHECK(res[k][e] < plan.primes[k]);
        dp[k] = digits[k].data();
        garner_step(plan, static_cast<uint32_t>(k), dp.data(), res[k], digits[k].data(), E, path);
    }
}

bool garner_negative(const GarnerPlan& plan, const std::vector<std::vector<uint32_t>>& digits, size_t e) {
    for (size_t k = plan.primes.size(); k-- > 0;) {
        const uint32_t h = (plan.primes[k] - 1) / 2;
        if (digits[k][e] > h) return true;
        if (digits[k][e] < h) return false;
    }
    return false;   // exactly (P - 1)/2: non-negative
}

}  // namespace rns
}  // namespace engine
