// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef ORBIFOLD_MP_COMPARISON_H
#define ORBIFOLD_MP_COMPARISON_H

// ===========================================================================
// (hecke-extensions slot 22, MP-2) The comparison map
//     Phi : Z[F'_.]^N  -->  Iso_.          (MP_impl_plan.md v0.2 Sec. 2.4;
//                                            MP_math_plan.md v0.2 [MP] Sec. 2.2 Step 3, Sec. 7.5)
// from the (normalised) free resolution Z[F'_.] of Z whose Gamma-cochains are
// the MP cochains C^.(S; A_M) of [MP] Def. 1.5 ON THE NOSE ([MP] Thm 2.1, Steps
// 1 and 3), to the isotropy-resolution model Iso_. of [C, def:isomodel]
// (Integral/Model/IsotropyModel) -- a lift of id_Z between two free
// Z[Gamma]-resolutions of Z, hence unique up to homotopy (comparison theorem),
// and Phi^* : C^._I -> C^.(S; A_M), f |-> f o Phi, a quasi-isomorphism in every
// degree over every ring.  The code CERTIFIES the chain-level statement and
// MEASURES the cohomological one (CochainMaps.h) in the degrees it builds.
//
// THE FREE RESOLUTION.  Z[F'_k] is free on the strings with g_0 = 1 (Step 1);
// its normalised quotient has the basis e_s, s in S_k^nd, and
//     d e_s = gamma_1 . e_{d_0 s} + sum_{i >= 1} (-1)^i e_{d_i s}
// (degenerate faces dropped): the face d_0 of ((1, s_0), (g_1, s_1), ...) is
// g_1 . ((1, s_1), (g_1^{-1} g_2, s_2), ...) -- the twist rho(gamma_1) of the
// MP differential is the Gamma-linearity of Hom_Gamma(Z[F'], M).
//
// THE LIFT (generator by generator, extended Gamma-linearly from g_0 = 1):
//   Phi_0(e_s)  = h . e_{0, v, 0}, v the representative of the orbit of the FIRST
//                 vertex of iota(s) that lies in an orbit of the target model
//                 (point = h . v); aug o Phi_0 = aug;
//   Phi_k(e_s)  = fill(Phi_{k-1}(d e_s)), fill = PeriodicityClass::fill, the
//                 Z-linear contracting homotopy of Iso_. (the staircase of
//                 [C, prop:isomodel](2)), which ENGINE_CHECKs its input is a cycle
//                 (augmentation 0 in degree 0) and d y = z.
// The target is the SHIPPED 7-vertex Iso_. for every source triangulation (M7
// and R_Z alike): Phi only needs a resolution with a contracting homotopy.
// certify() re-derives d Phi_k(e_s) = Phi_{k-1}(d e_s) exactly over Z[Gamma]
// on every generator, from the stored values alone, and aug o Phi_0 = aug.
//
// Phi^*_q : C^q_I -> C^q(S; A_{Sym^n}) over Z/m (Pullback): row (s, a), column
// (cell, b), entry sum_{c g in Phi_q(e_s)|cell} c rho_n(g)[a][b] -- the layouts of
// IsotropyModel::coboundary and orbifold_mp::Coboundary.
//
// Exact int64 coefficients (hecke overflow checks inside GroupRingElt); no
// bignum; no golden object; no setting.
// ===========================================================================

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "Descriptor.h"
#include "Nerve.h"
#include "SparseLength.h"
#include "../../Integral/Model/IsotropyModel.h"
#include "../../Integral/Model/PeriodicityClass.h"

namespace orbifold_mp {

struct PhiCertificate {
    bool aug = false;              // aug o Phi_0 = aug on every S_0 generator
    bool chain_map = false;        // d Phi_k(e_s) = Phi_{k-1}(d e_s), every s in S_k^nd, k = 1 .. k_top
    uint64_t generators_checked = 0;
    std::string first_failure;
    bool all() const { return aug && chain_map; }
};

class PhiLift {
public:
    // Builds Phi_0 .. Phi_{k_top} on the non-degenerate strings of S (k_top <= S.k_max(), and
    // k_top <= P.model().q_top() so that the fillings land in the built degrees).  S and P must outlive *this.
    PhiLift(const InternedMPSet& S, const Descriptor& D, const integral_model::PeriodicityClass& P, uint32_t k_top);

    uint32_t k_top() const { return static_cast<uint32_t>(start_.size()) - 1; }
    // Phi_k(e_s) for the j-th non-degenerate string of S_k, as a chain of Iso_k.
    integral_model::Chain value(uint32_t k, uint64_t j) const;

    PhiCertificate certify() const;
    // The negative control (MP_impl_plan_review.md R5(b)): add one unit term to Phi_k(e_s) (s the j-th
    // non-degenerate string) at cell 0; certify() must then fail.
    void perturb_for_negative_control(uint32_t k, uint64_t j);

    // Phi^*_q over Z/(ell^e): rows S_q^nd * (n + 1), columns w_q * (n + 1).
    SparseRows Pullback(uint32_t q, uint32_t n, uint32_t ell, uint32_t e) const;

    // Statistics (the report, rule 19).
    uint64_t total_terms() const;
    uint64_t distinct_elements() const { return elems_.size(); }
    int64_t max_abs_coefficient() const;
    double build_seconds() const { return build_s_; }

private:
    struct Term {
        uint32_t cell;
        uint32_t g;      // index into elems_
        int64_t c;
    };
    const InternedMPSet* S_;
    const integral_model::PeriodicityClass* P_;
    std::vector<integral_model::GKey> elems_;
    std::map<integral_model::GKey, uint32_t> index_;
    std::vector<std::vector<uint64_t>> start_;   // [k][j] .. [k][j+1]: the terms of Phi_k(e_{nd j})
    std::vector<std::vector<Term>> terms_;
    double build_s_ = 0.0;

    uint32_t intern(const integral_model::GKey& g);
    void store(uint32_t k, const integral_model::Chain& c);
    // Phi_{k-1}(d e_s) for the j-th non-degenerate string of S_k, from the stored values.
    integral_model::Chain boundary_image(uint32_t k, uint64_t j) const;
};

}  // namespace orbifold_mp

#endif
