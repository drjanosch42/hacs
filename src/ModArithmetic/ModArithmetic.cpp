// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

/*
 * ModArithmetic.cpp
 *
 * This file provides the implementation for a suite of modular arithmetic functions,
 * which are fundamental building blocks for linear algebra and other computations
 * over finite fields (F_p). The functions are designed for efficiency, using
 * platform-specific optimizations where appropriate.
 *
 * Key Implementations:
 * - Basic Operations: `mod_add`, `mod_sub`, and `mod_mul` provide fast modular
 *   addition, subtraction, and multiplication. `mod_mul` uses x86 inline
 *   assembly for 32-bit builds and standard 64-bit integer arithmetic for
 *   64-bit builds to prevent overflow and ensure speed.
 * - Modular Inverse: `mod_inv` computes the multiplicative inverse of a number
 *   modulo p using the Extended Euclidean Algorithm. It is essential for division.
 * - Modular Exponentiation: `mod_pow` implements the highly efficient
 *   exponentiation by squaring algorithm to compute (base^exp) % p.
 * - Binomial Coefficients: `mod_nCr` computes "n choose r" modulo p. It uses
 *   Lucas's Theorem to handle large n and r by reducing the problem to smaller
 *   computations involving base-p representations, making it correct and
 *   efficient even when n or r are larger than the prime p.
 */

#include "pch.h"
#include "ModArithmetic.h"
#include <stdexcept>
#include <vector>
#include <algorithm>

#ifdef _MSC_VER
#include <intrin.h>   // _umul128, _udiv128 (MSVC .vcxproj builds; see mod_mul64)
#endif

namespace ModArithmetic {

    // matrix-engine Phase 2 (doc 03 Sec. 10, finding F2): mod_mul, mod_add,
    // mod_sub and mod_minus -- and their *64 twins further down -- moved to
    // ModArithmetic.h as `inline` definitions. Their bodies are unchanged; the
    // out-of-line call they used to force cost ~26 cycles per matrix element
    // and blocked every vectorisation. mod_inv / mod_pow / mod_nCr stay here.

    // Extended Euclidean Algorithm for modular inverse.
    //
    // The accumulators are int64_t (they were int32_t, which silently overflowed
    // -- and returned garbage without throwing -- for every p >= 2^31, e.g.
    // p = 2^31 + 11 = 2147483659, since the very first assignment r = p is then
    // not representable).  int64_t is sufficient for all p <= 2^32-1: the Bezout
    // cofactors satisfy |t_i| < p, hence |quotient * newt| <= |t_{i-1}| + |t_{i+1}|
    // < 2p < 2^33, and the remainders stay in [0, p).  This mirrors mod_inv64,
    // which uses __int128 for exactly the same reason one width up.
    uint32_t mod_inv(uint32_t a, uint32_t p) {
        int64_t t = 0, newt = 1;
        int64_t r = static_cast<int64_t>(p), newr = static_cast<int64_t>(a);

        while (newr != 0) {
            int64_t quotient = r / newr;
            int64_t temp_t = t;
            t = newt;
            newt = temp_t - quotient * newt;

            int64_t temp_r = r;
            r = newr;
            newr = temp_r - quotient * newr;
        }

        if (r > 1) {
            throw std::runtime_error("Modular inverse does not exist.");
        }

        if (t < 0) {
            t += static_cast<int64_t>(p);
        }

        return static_cast<uint32_t>(t);
    }

    // We assume p prime, i.e. we reduce exp modulo p-1.
    // Exact for every p <= 2^32-1: the only arithmetic on residues is mod_mul,
    // whose intermediate product is formed in uint64_t.
    uint32_t mod_pow(uint32_t base, uint32_t exp, uint32_t p) {
        uint32_t res = 1;
        base %= p;
        if (!base && exp)
            return 0;
        exp  %= p-1;
        while (exp > 0) {
            if (exp % 2 == 1) res = mod_mul(res, base, p);
            base = mod_mul(base, base, p);
            exp /= 2;
        }
        return res;
    }

// ===========================================================================
// Factorial / inverse-factorial tables for the binomial coefficients
// (matrix-engine Phase 1, item 7.0; docs/matrix-engine/03-profiling-baseline.md
// Sec. 5.2 A1).
//
// mod_nCr_small(n, r, p) is only ever reached with n, r < p (its callers are
// mod_nCr / mod_nCr64, which feed it the base-p Lucas digits). For n < p none
// of 1..n is divisible by p, so
//         C(n,r) = n! * (r!)^{-1} * ((n-r)!)^{-1}   in F_p
// is exact -- the same value the Pascal DP produced, but in O(1) after an O(n)
// table build, instead of the O(n*r) additions AND the r+1 word heap
// allocation the DP performed on EVERY call. (rho_n at n_sym = 160 made
// ~4*10^4 such calls per evaluation.)
//
// The tables are cached per (n, ell): a change of modulus resets them, a larger
// n extends them geometrically. They are `thread_local`, so the primitives keep
// the property they had before -- no shared mutable state, hence no new data
// race (full thread-safety is not a Phase 1 requirement, but this must not get
// worse).
//
// The Pascal DP survives, verbatim, as the n >= p fallback: mod_nCr_small has
// external linkage, and for n >= p the factorial route would divide by zero.
// ===========================================================================
namespace {

    template<class Elt>
    struct FactorialTables {
        Elt p = 0;                  // modulus these tables belong to (0 = unset)
        std::vector<Elt> fact;      // fact[i]     = i!        mod p, i < p
        std::vector<Elt> inv_fact;  // inv_fact[i] = (i!)^{-1} mod p, i < p
    };

    inline uint32_t ft_mul(uint32_t a, uint32_t b, uint32_t p) { return mod_mul(a, b, p); }
    inline uint64_t ft_mul(uint64_t a, uint64_t b, uint64_t p) { return mod_mul64(a, b, p); }
    inline uint32_t ft_inv(uint32_t a, uint32_t p) { return mod_inv(a, p); }
    inline uint64_t ft_inv(uint64_t a, uint64_t p) { return mod_inv64(a, p); }

    // Returns tables valid for indices 0 .. n (precondition: n < p).
    template<class Elt>
    const FactorialTables<Elt>& factorial_tables(Elt n, Elt p) {
        static thread_local FactorialTables<Elt> tab;
        if (tab.p != p) {
            tab.p = p;
            tab.fact.clear();
            tab.inv_fact.clear();
        }
        const size_t need = static_cast<size_t>(n) + 1;
        if (tab.fact.size() >= need) return tab;

        size_t target = need > tab.fact.size() * 2 ? need : tab.fact.size() * 2;
        // Never tabulate index >= p: i! == 0 mod p there and inv_fact would not
        // exist. n < p guarantees need <= p, so the cap never truncates below need.
        if (static_cast<Elt>(target) > p || target > static_cast<size_t>(p)) {
            target = static_cast<size_t>(p);
        }

        const size_t old = tab.fact.size();
        tab.fact.resize(target);
        if (old == 0) {
            tab.fact[0] = Elt(1) % p;
        }
        for (size_t i = (old == 0 ? size_t(1) : old); i < target; ++i) {
            tab.fact[i] = ft_mul(tab.fact[i - 1], static_cast<Elt>(i), p);
        }
        tab.inv_fact.assign(target, Elt(0));
        tab.inv_fact[target - 1] = ft_inv(tab.fact[target - 1], p);
        for (size_t i = target - 1; i > 0; --i) {
            tab.inv_fact[i - 1] = ft_mul(tab.inv_fact[i], static_cast<Elt>(i), p);
        }
        return tab;
    }

    template<class Elt>
    Elt nCr_via_tables(Elt n, Elt r, Elt p) {
        const FactorialTables<Elt>& tab = factorial_tables<Elt>(n, p);
        return ft_mul(tab.fact[static_cast<size_t>(n)],
                      ft_mul(tab.inv_fact[static_cast<size_t>(r)],
                             tab.inv_fact[static_cast<size_t>(n - r)], p), p);
    }

} // namespace

    // Helper for mod_nCr: computes nCr mod p for n, r < p (the Lucas digits).
    // Exact for every p <= 2^32-1: the only arithmetic is mod_add, fixed above.
    // (`r < 0` was tested here on an unsigned argument, i.e. never; it is
    // dropped, mirroring mod_nCr_small64.)
    uint32_t mod_nCr_small(uint32_t n, uint32_t r, uint32_t p) {
        if (r > n) return 0;
        if (r == 0 || r == n) return 1;

        // Primary route (n < p, i.e. every call coming out of mod_nCr's Lucas
        // loop): tabulated factorials, O(1) per call, no allocation.
        if (n < p) return nCr_via_tables<uint32_t>(n, r, p);

        if (r > n / 2) r = n - r; // Use symmetry C(n,r) = C(n, n-r)

        std::vector<uint32_t> C(r + 1, 0);
        C[0] = 1;
        for (uint32_t i = 1; i <= n; ++i) {
            for (uint32_t j = std::min(i, r); j > 0; --j) {
                C[j] = mod_add(C[j], C[j - 1], p);
            }
        }
        return C[r];
        /* Alternative by direct formula - less efficient if many calls for same p
        if (r < 0 || r > n) return 0;
        if (r == 0 || r == n) return 1;
        if (r > n / 2) r = n - r;

        uint32_t numerator = 1;
        for (uint32_t i = 0; i < r; ++i) {
            numerator = mod_mul(numerator, n - i, p);
        }
        uint32_t denominator = 1;
        for (uint32_t i = 1; i <= r; ++i) {
            denominator = mod_mul(denominator, i, p);
        }
        return mod_mul(numerator, ModArithmetic::mod_inv(denominator, p), p);
        */
    }


    uint32_t mod_nCr(uint32_t n, uint32_t r, uint32_t p) {
        if (r > n) return 0;
        //if (p == 0) throw std::runtime_error("mod_nCr: prime p cannot be 0.");
        //if (p == 1) return 0; // All non-empty C(n,r) are 0 mod 1. C(n,0)=1.

        // Lucas's Theorem
        if (r == 0) return 1;
        uint32_t res = 1;
        while (r > 0) {
            uint32_t ni = n % p;
            uint32_t ri = r % p;
            res = mod_mul(res, mod_nCr_small(ni, ri, p), p);
            n /= p;
            r /= p;
        }
        return res;
    }

    // =======================================================================
    // Stage S13 -- 64-bit primitives (ell < 2^64), [C, thm:backend].
    // Structurally identical to the uint32_t versions; the only wide operation
    // is the exact product, formed in `unsigned __int128` (I5: NOT bignum).
    // =======================================================================

    // mod_mul64 / mod_add64 / mod_sub64 / mod_minus64: see the Phase-2 note
    // above -- inline in ModArithmetic.h, bodies unchanged.

    // Extended Euclidean Algorithm for modular inverse (signed __int128 so the
    // moduli/residues up to 2^64-1 and the Bezout cofactors never overflow).
    uint64_t mod_inv64(uint64_t a, uint64_t p) {
        __int128 t = 0, newt = 1;
        __int128 r = (__int128)p, newr = (__int128)a;

        while (newr != 0) {
            __int128 quotient = r / newr;
            __int128 temp_t = t;
            t = newt;
            newt = temp_t - quotient * newt;

            __int128 temp_r = r;
            r = newr;
            newr = temp_r - quotient * newr;
        }

        if (r > 1) {
            throw std::runtime_error("Modular inverse does not exist.");
        }
        if (t < 0) {
            t += (__int128)p;
        }
        return (uint64_t)t;
    }

    // We assume p prime, i.e. we reduce exp modulo p-1 (mirrors mod_pow).
    uint64_t mod_pow64(uint64_t base, uint64_t exp, uint64_t p) {
        uint64_t res = 1;
        base %= p;
        if (!base && exp)
            return 0;
        exp %= p - 1;
        while (exp > 0) {
            if (exp & 1ull) res = mod_mul64(res, base, p);
            base = mod_mul64(base, base, p);
            exp >>= 1;
        }
        return res;
    }

    // Helper for mod_nCr64, computes nCr mod p when n, r < p (Pascal recurrence,
    // mirrors mod_nCr_small).
    static uint64_t mod_nCr_small64(uint64_t n, uint64_t r, uint64_t p) {
        if (r > n) return 0;
        if (r == 0 || r == n) return 1;

        // Primary route (n < p): tabulated factorials. See the note above.
        if (n < p) return nCr_via_tables<uint64_t>(n, r, p);

        if (r > n / 2) r = n - r; // symmetry C(n,r) = C(n, n-r)

        std::vector<uint64_t> C(r + 1, 0);
        C[0] = 1;
        for (uint64_t i = 1; i <= n; ++i) {
            for (uint64_t j = std::min<uint64_t>(i, r); j > 0; --j) {
                C[j] = mod_add64(C[j], C[j - 1], p);
            }
        }
        return C[r];
    }

    uint64_t mod_nCr64(uint64_t n, uint64_t r, uint64_t p) {
        if (r > n) return 0;
        if (r == 0) return 1;
        // Lucas's Theorem
        uint64_t res = 1;
        while (r > 0) {
            uint64_t ni = n % p;
            uint64_t ri = r % p;
            res = mod_mul64(res, mod_nCr_small64(ni, ri, p), p);
            n /= p;
            r /= p;
        }
        return res;
    }

}
