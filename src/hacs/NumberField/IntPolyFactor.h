// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// IntPolyFactor.h -- the factorisation of a MONIC polynomial over Z into irreducible factors with multiplicities
// (slot-24 NB-TOURFIX; the notebook's factored char_Z(T_p) of the integral descent record).  In-house, STL + BigInt only
// (owner rule: no external CAS).
//
// THE METHOD (exact throughout; Cohen, A Course in Computational Algebraic Number Theory, Sec. 3.5).
//   1. Yun's squarefree decomposition f = prod_i g_i^i over Z (g_i squarefree, pairwise coprime; every gcd is taken by
//      the primitive pseudo-remainder sequence over Z and is monic by Gauss's lemma, so every division is by a monic
//      polynomial and stays in Z[X]).
//   2. Zassenhaus on each g_i: a prime q < 2^16 with g_i mod q squarefree and the fewest factors among the first
//      kPrimeTrials such primes (fq_factor_squarefree of PureField.h); a linear multifactor Hensel lift of the factors
//      to q^k > 2 B, B = 2^d (d + 1) max|a_j| (Mignotte's bound, with room); recombination of subsets by increasing size
//      (a constant-term test, then exact trial division).  A factor found by exhaustive recombination is irreducible;
//      the last cofactor is irreducible when the enumeration finished.
//   3. A cap on the recombination (kMaxSubsets subsets): when it is hit, the remaining cofactor is reported as ONE factor
//      with irreducible = false (a product of irreducibles not separated), never guessed.
//   4. Verification: the product of factor^multiplicity is recomputed and compared with f coefficient by coefficient
//      (NF_CHECK); `verified` records it.
//
// Polynomials are coefficient vectors, constant term first (as PolyAlg.h / Polred.h).
#include <cstdint>
#include <string>
#include <vector>

#include "BigInt.h"

namespace nf {

struct ZFactor {
    std::vector<BigInt> poly;   // monic, constant term first
    unsigned multiplicity = 1;
    bool irreducible = true;    // false: the recombination cap was hit; a product of irreducibles, not separated
};
struct ZFactorisation {
    std::vector<ZFactor> factors;  // sorted by (degree, then -c_{d-1}, ..., -c_0 lexicographically): linear factors by root
    bool complete = true;          // every factor irreducible
    bool verified = false;         // prod factor^multiplicity == f, exactly
    // "(X + 24)^2 (X - 2049)"; one factor of multiplicity 1 is written without parentheses.
    std::string text(const std::string& var = "X") const;
};

// f monic (f.back() == 1) of degree >= 1; NF_CHECK otherwise.
ZFactorisation factor_monic_over_z(const std::vector<BigInt>& f);

// Helpers (exposed for the checks): the product, and the polynomial's text ("X^3 + X^2 - 8X - 12").
std::vector<BigInt> zpoly_mul(const std::vector<BigInt>& a, const std::vector<BigInt>& b);
std::string zpoly_text(const std::vector<BigInt>& f, const std::string& var = "X");

}  // namespace nf
