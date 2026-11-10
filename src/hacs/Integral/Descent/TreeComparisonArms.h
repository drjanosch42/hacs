// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_DESCENT_TREE_COMPARISON_ARMS_H
#define INTEGRAL_DESCENT_TREE_COMPARISON_ARMS_H

// DESCENT D5: the chain-level and cohomology arms (M), (C), (N) of HeckeIsotropyModelTreeValidation
// (Integral/Model/IsotropyModelTree.h; [C, prop:isomodeltree](1)(a), (3), (4)).  q_top: the degree the comparisons are
// built to (6 standard, 8 slow); slow: more cells and primes.  check(ok, what) counts one check.
#include <cstdint>
#include <string>
#include <vector>

namespace integral_descent {

using TreeCheckFn = bool (*)(bool, const std::string&);
void RunTreeComparisonArms(uint32_t q_top, bool slow, TreeCheckFn check);

// (E) (D5 audit m3: replaces "explicit d^q == the (2)(d) copy", which ran the same closed-form branch twice).  The production
// cochains TreeLocalForm (Integral/Model/TreeCochains.h) over F_p (p prime to q_ell) on Sym^n (N = 1) or Ind_{Gamma_0(N)}^Gamma
// Sym^n against the independent chain-level model TreeModelChain (IsotropyModelTree.h; its own cells, coboundary and idempotent
// projectors on the dense coefficient module), degree by degree, q = 0 .. q_max (ell >= 5: q <= 1, the Bredon tree):
//   * d^{q+1} d^q = 0 as matrices (the audit's N2: the groups route Smith-reduces each d^q alone and never sees a non-complex);
//   * c_q = rank of TreeModelChain's projector on C^q ([C, prop:isomodeltree](2)(e) against (1)(c));
//   * dim H^q = c_q - rank d^q - rank d^{q-1} == dim H^q of TreeModelChain's coboundary on im P (ranks of THIS form).
struct TreeFormCheck {
    uint32_t checks = 0;
    bool dd_zero = true, ranks = true, dims = true;
    std::string first;   // the first failing statement
    bool ok() const { return dd_zero && ranks && dims; }
};
TreeFormCheck CheckTreeFormAgainstChain(uint32_t ell, uint32_t N, uint32_t n, uint32_t p, uint32_t q_max);

// (L1) (D5 audit m3: replaces the tautology "rank P_0 - rank delta^0 == b_0").  The production tree form over F_p at a good prime
// p (the descent's rank prime) on Sym^n: dim H^q = c_q - rank d^q - rank d^{q-1}, the ranks computed HERE on the reduced
// matrices, == h[q] for q = 0 .. q_max (h: the shipped C_7 Betti numbers at p, 0 above degree 1).  why: the first mismatch.
bool TreeFieldDimsEqual(uint32_t ell, uint32_t n, uint32_t p, const std::vector<uint32_t>& h, uint32_t q_max, std::string& why);

}  // namespace integral_descent

#endif
