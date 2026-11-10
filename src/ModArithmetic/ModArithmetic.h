// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#ifndef MOD_ARITHMETIC_H
#define MOD_ARITHMETIC_H

#include <cstdint>

#ifdef _MSC_VER
#include <intrin.h>   // _umul128, _udiv128 (MSVC .vcxproj builds; see mod_mul64)
#endif

namespace ModArithmetic {

    // =======================================================================
    // matrix-engine Phase 2 -- the four hot primitives are DEFINED HERE.
    //
    // They used to be declared here and defined (without `inline`, with no
    // LTO) in ModArithmetic.cpp, so every inner-loop element of every matrix
    // operation cost an out-of-line call:
    // docs/matrix-engine/03-profiling-baseline.md Sec. 10, finding F2, counts
    // 30 unresolved call relocations to mod_mul from FlatMatrix.cpp.o alone
    // and attributes the whole 5x gap between the shipped GEMM (0.17
    // Gmodmul/s) and the same loop written inside one TU (0.90 Gmodmul/s) to
    // that boundary. Nothing can be vectorised across a call, either, so this
    // is also a prerequisite for the kernel retrofit.
    //
    // The BODIES are byte-for-byte the ones that stood in ModArithmetic.cpp;
    // only the storage class changed. The overflow arguments they rest on
    // (and the Phase-0 fixes to mod_add / mod_inv for p >= 2^31) are quoted
    // verbatim above each function and are exercised by
    // mod_arithmetic_validation at ell in {2147483647, 2147483659, 4294967291}.
    //
    // mod_inv / mod_pow / mod_nCr stay out-of-line: they are O(log p) or
    // table-driven, never appear in an inner loop, and mod_nCr owns a
    // thread_local cache that must have exactly one home.
    // =======================================================================

    inline uint32_t mod_mul(uint32_t a, uint32_t b, uint32_t p) {
        uint64_t product = (uint64_t)a * b;
        return (uint32_t)(product % p);
    }

    // Fast modular addition.
    // a, b in [0, p), p <= 2^32-1.  `result < a` detects the uint32_t wrap that
    // occurs as soon as a + b >= 2^32 (possible for p > 2^31); in both the wrap
    // and the result >= p case a single subtraction of p yields the canonical
    // residue, because the wrapped value and the true sum a + b are congruent
    // mod 2^32 and (a + b) - p lies in [0, p) < 2^32.  This is the same test
    // (and the same argument) as in the 64-bit twin mod_add64 below.
    inline uint32_t mod_add(uint32_t a, uint32_t b, uint32_t p) {
        uint32_t result = a + b;
        if (result < a || result >= p) result -= p;
        return result;
    }

    // Fast modular subtraction.
    // a, b in [0, p): the expression a + p - b is evaluated in uint32_t, i.e.
    // modulo 2^32; its true value p - (b - a) lies in (0, p) < 2^32, so the
    // intermediate wrap of a + p (possible for p > 2^31) cancels and the result
    // is exact for every p <= 2^32-1.
    inline uint32_t mod_sub(uint32_t a, uint32_t b, uint32_t p) {
        return (a >= b) ? (a - b) : (a + p - b);
    }

    // Fast modular negation
    inline uint32_t mod_minus(uint32_t a, uint32_t p) {
        return (a == 0) ? 0 : p - a;
    }

    uint32_t mod_inv(uint32_t a, uint32_t p);

    /**
     * @brief Computes nCr (binomial coefficient "n choose r") modulo p.
     * Uses Lucas's Theorem for correctness when p <= n or p <= r.
     * @param n The total number of items.
     * @param r The number of items to choose.
     * @param p The prime modulus.
     * @return (n choose r) mod p.
     */
    uint32_t mod_nCr(uint32_t n, uint32_t r, uint32_t p);

    /**
     * @brief Computes (base^exp) % mod.
     * @param base The base.
     * @param exp The exponent.
     * @param mod The modulus.
     * @return (base^exp) % p.
     */
    uint32_t mod_pow(uint32_t base, uint32_t exp, uint32_t p);

    // =======================================================================
    // Stage S13 -- 64-bit arithmetic backend for coefficient primes ell < 2^64
    // ([C, thm:backend] "Mod-ell backend" in docs/latex/hecke_action.tex).
    //
    // These mirror the uint32_t primitives above EXACTLY in structure; the only
    // difference is that the exact product a*b (which for ell < 2^64 no longer
    // fits in uint64_t) is formed in `unsigned __int128` -- NOT arbitrary
    // precision (invariant I5). On MSVC (the .vcxproj builds), where __int128
    // is unavailable, the wide product/quotient go through the _umul128 /
    // _udiv128 intrinsics (see ModArithmetic.cpp).
    //
    // The uint32_t primitives stay the primary regime and are exact on the FULL
    // documented range 0 < ell < 2^32 (the intermediate product is uint64_t).
    //
    // Phase 0 note (matrix-engine, defects #2 and #33): that bound used to be a
    // claim, not a fact -- mod_inv accumulated in int32_t and mod_add ignored
    // the uint32_t carry, so both were silently wrong (no throw) for every
    // ell >= 2^31, and mod_add's error propagated through mod_nCr_small into
    // mod_nCr and hence into every rho_n.  Both are fixed; mod_sub, mod_minus,
    // mod_mul, mod_pow and mod_nCr were audited and are exact as written.
    // mod_arithmetic_validation now checks the whole set against a slow
    // __int128 reference at ell in {2147483647, 2147483659, 4294967291}.
    // =======================================================================
    // The four hot 64-bit twins are inline here for the same reason as their
    // uint32_t counterparts above (Phase 2 / doc 03 Sec. 10 F2); bodies
    // unchanged from ModArithmetic.cpp.
    inline uint64_t mod_mul64(uint64_t a, uint64_t b, uint64_t p) {
#ifdef _MSC_VER
        // MSVC has no __int128: use the 128-bit multiply/divide intrinsics.
        unsigned long long hi;
        unsigned long long lo = _umul128(a, b, &hi);
        unsigned long long rem;
        // hi must be < p for _udiv128 not to overflow; a,b < p <= 2^64-1 gives
        // hi = floor(a*b / 2^64) <= p-2 < p, so the precondition holds.
        _udiv128(hi, lo, p, &rem);
        return rem;
#else
        unsigned __int128 product = (unsigned __int128)a * (unsigned __int128)b;
        return (uint64_t)(product % (unsigned __int128)p);
#endif
    }

    inline uint64_t mod_add64(uint64_t a, uint64_t b, uint64_t p) {
        // a, b in [0, p), p <= 2^64-1. `r < a` detects the uint64_t wrap; in
        // both the wrap and the r >= p case a single subtraction of p yields
        // the canonical residue (see the S13 note: the wrapped value r and the
        // true sum a+b are congruent mod 2^64, so r - p == (a+b) - p mod 2^64).
        uint64_t r = a + b;
        if (r < a || r >= p) r -= p;
        return r;
    }

    inline uint64_t mod_sub64(uint64_t a, uint64_t b, uint64_t p) {
        // a, b in [0, p): a + (p - b) stays < p (== p - (b - a)), no overflow.
        return (a >= b) ? (a - b) : (a + (p - b));
    }

    inline uint64_t mod_minus64(uint64_t a, uint64_t p) {
        return (a == 0) ? 0 : p - a;
    }

    uint64_t mod_inv64(uint64_t a, uint64_t p);
    uint64_t mod_pow64(uint64_t base, uint64_t exp, uint64_t p);
    uint64_t mod_nCr64(uint64_t n, uint64_t r, uint64_t p);

    // -----------------------------------------------------------------------
    // Width-dispatched wrappers (Stage S13). Element-templated linear-algebra
    // code (FlatMatrixT<Elt>, MatrixT<Elt>, ...) calls these so the SAME source
    // picks the exact uint32_t primitive for Elt = uint32_t (byte-identical to
    // the pre-S13 code -- these are `inline` and forward verbatim) and the
    // __int128-backed 64-bit primitive for Elt = uint64_t. Overload resolution
    // is exact on the element width (no implicit narrowing/widening), so the
    // uint32_t regime is a pure no-op wrapper.
    // -----------------------------------------------------------------------
    inline uint32_t mod_mul_w(uint32_t a, uint32_t b, uint32_t p) { return mod_mul(a, b, p); }
    inline uint64_t mod_mul_w(uint64_t a, uint64_t b, uint64_t p) { return mod_mul64(a, b, p); }
    inline uint32_t mod_add_w(uint32_t a, uint32_t b, uint32_t p) { return mod_add(a, b, p); }
    inline uint64_t mod_add_w(uint64_t a, uint64_t b, uint64_t p) { return mod_add64(a, b, p); }
    inline uint32_t mod_sub_w(uint32_t a, uint32_t b, uint32_t p) { return mod_sub(a, b, p); }
    inline uint64_t mod_sub_w(uint64_t a, uint64_t b, uint64_t p) { return mod_sub64(a, b, p); }
    inline uint32_t mod_minus_w(uint32_t a, uint32_t p) { return mod_minus(a, p); }
    inline uint64_t mod_minus_w(uint64_t a, uint64_t p) { return mod_minus64(a, p); }
    inline uint32_t mod_inv_w(uint32_t a, uint32_t p) { return mod_inv(a, p); }
    inline uint64_t mod_inv_w(uint64_t a, uint64_t p) { return mod_inv64(a, p); }
    inline uint32_t mod_pow_w(uint32_t base, uint32_t exp, uint32_t p) { return mod_pow(base, exp, p); }
    inline uint64_t mod_pow_w(uint64_t base, uint64_t exp, uint64_t p) { return mod_pow64(base, exp, p); }
    inline uint32_t mod_nCr_w(uint32_t n, uint32_t r, uint32_t p) { return mod_nCr(n, r, p); }
    inline uint64_t mod_nCr_w(uint64_t n, uint64_t r, uint64_t p) { return mod_nCr64(n, r, p); }

}

#endif // MOD_ARITHMETIC_H
