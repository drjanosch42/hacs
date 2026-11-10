// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef ORBIFOLD_MP_REFINEMENT_H
#define ORBIFOLD_MP_REFINEMENT_H

// ===========================================================================
// (hecke-extensions slot 22, MP-2) The refinement map of MP sets
//     u : S(X~; T~) --> S(X; T)        for a Gamma-equivariant subdivision X~ of X
// (MP_math_plan.md v0.2 [MP] Sec. 7.1 "refinement", Thm 7.3 (3); MP_impl_plan.md
// v0.2 Sec. 2.4 "Refinement R_Z -> M7").
//
// LABELS.  For each top simplex s~ of X~ the coarse top simplex s containing it
// is read off the subdivision record of integral_model::ModelData (`refines`:
// root simplex -> the fine simplices whose sum it is; an unsplit simplex is
// itself), and a label h_{s~} with iota~(s~) <= h iota(s).  Such h form a coset
// h_0 Gamma_s (pointwise stabiliser: {+-I} on every triangle of SL_2(Z)); the
// label is its least GKey (MP_impl_plan.md Sec. 2.4).  ContainedIn() decides the
// containment for ANY h: every vertex of iota~(s~) is, as a point of X~, one of
// h . (the vertices of s and the subdivision vertices inside s) -- the vertex
// set of the subdivided closed simplex h iota(s), which is convex.
//
// THE MAP on F' ([MP] Sec. 7.1): ((g_i, s~_i)) |-> ((g_i h_{s~_i}, s_i)), a
// Gamma-map commuting with the augmentations; on the g_0 = 1 normal form
//     u(1; s~_0, ..., s~_k; gamma_1, ..., gamma_k)
//        = h_{s~_0} . (1; s_0, ..., s_k; h_{s~_0}^{-1} gamma_1 h_{s~_1}, ..., h_{s~_{k-1}}^{-1} gamma_k h_{s~_k}),
// so the cochain map is (u^* f)(s~) = rho(h_{s~_0}) f(u(s~)) -- the factor
// rho(h_{s~_0}) is the Gamma-linearity (the plan's one-line formula omits it;
// with h = -I it is (-1)^n).  A degenerate image carries the zero cochain
// (normalised complex).  Different labels give homotopic maps of free
// resolutions (both lift id_Z), hence the same map on cohomology (R6).
// ===========================================================================

#include <cstdint>
#include <string>
#include <vector>

#include "Descriptor.h"
#include "Nerve.h"
#include "SparseLength.h"
#include "../../Integral/Model/IsotropyModel.h"

namespace orbifold_mp {

struct RefinementLabels {
    std::vector<uint32_t> coarse;   // fine top index -> coarse top index
    std::vector<Mat22> h;           // fine top index -> label
};

// From the subdivision record of `fine` (built from `coarse` by SubdivideEdgeOrbit): the containing coarse
// simplex of every fine top simplex (ENGINE_CHECK: exactly one) and the least-GKey label of h_0 Gamma_s.
RefinementLabels LabelsFromRecord(const integral_model::ModelData& fine, const Descriptor& Dfine,
                                  const integral_model::ModelData& coarse, const Descriptor& Dcoarse);

// iota~(fine_t) <= h iota(coarse_t) (decidable form above).
bool ContainedIn(const integral_model::ModelData& fine, const Descriptor& Dfine, const Descriptor& Dcoarse,
                 uint32_t fine_t, uint32_t coarse_t, const Mat22& h);

struct StringMap {
    // image[k][j]: for the j-th string of the FULL level k of the fine set, the index of u(s~) in the full
    // level k of the coarse set (InternedMPSet::kNoIndex if u(s~) is not a string of S(X)).
    std::vector<std::vector<uint64_t>> image;
    uint64_t outside = 0;            // strings whose image is not in S(X) (the NC counts these)
    uint64_t degenerate_images = 0;  // non-degenerate fine strings with a degenerate image
};
StringMap MapStrings(const InternedMPSet& Sf, const InternedMPSet& Sc, const RefinementLabels& L, uint32_t k_top);

// u^*_q : C^q(S(X)) -> C^q(S(X~)) over Z/(ell^e), Sym^n: rows S~_q^nd * (n + 1), columns S_q^nd * (n + 1).
// ENGINE_CHECK: no image outside S(X) at level q.  twist = false drops rho(h_{s~_0}) (the negative control: the
// plan's one-line formula; not a cochain map once the labels differ by -I between simplices, n odd).
SparseRows RefinementPullback(const InternedMPSet& Sf, const InternedMPSet& Sc, const RefinementLabels& L,
                              const StringMap& U, uint32_t q, uint32_t n, uint32_t ell, uint32_t e, bool twist = true);

}  // namespace orbifold_mp

#endif
