// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef ORBIFOLD_MP_SUBGROUP_MP_H
#define ORBIFOLD_MP_SUBGROUP_MP_H

// ===========================================================================
// (hecke-extensions slot 22, MP-3) Finite-index subgroups B <= A <= Gamma on the
// MP model: Shapiro on one F', restriction and corestriction, and the
// forget-coset morphism ([C] 1.22.0-pre.6 prop:mprescor (1)-(3),
// def:mpmorphism, thm:mpnatural; MP_math_plan.md v0.2.1 Sec. 7.1 "subgroup /
// level change", Sec. 7.4; MP_impl_plan.md v0.2.1 Sec. 2.5).  Levels are
// hecke::level::CosetLevel (LES-1's CongruenceLevel, Gamma0LevelN, GammaHLevelN):
// right cosets A w_j, w_1 = I, class_of(g) = the j with A g = A w_j.
//
// (1) SHAPIRO ON ONE F' (prop:mprescor (1)).  F'_Gamma (the labelled Cech nerve of
// the SL_2(Z)-descriptor) is a free A-set for every A, so
//     Hom_A(N Z[F'], M) = Hom_Gamma(N Z[F'], Coind_A^Gamma M) = C^.(S_Gamma; Coind_A M),
// with Coind_A M = M^{A\Gamma} in the coset-representative (MONOMIAL) form:
// coordinate j = phi(w_j), and gamma acting by the block (j, j.gamma) = rho(h_j),
// h_j = w_j gamma w_{j.gamma}^{-1} in A (CoindRep) -- LES-1's M_{A,1}(gamma) of
// eq:MHchi.  res^A_B : Coind_A M -> Coind_B M has the block (i, a(i)) =
// rho(v_i w_{a(i)}^{-1}) and cor^A_B = the trace, block (a(i), i) =
// rho(v_i w_{a(i)}^{-1})^{-1} (LES-1's alpha_{B/A}, omega_{B/A}); both are Gamma-maps
// and cor o res = [A : B] exactly (ResCorMaps).  The coset representatives are
// the labels here: dropping rho(.) (or using a wrong representative) breaks the
// Gamma-equivariance at n >= 1.
//
// (2) THE MP SET OF B (prop:mprescor (2)).  SubgroupDescriptor(D, B) is the
// descriptor of the SAME X as a B-complex: vertex B-orbits = the double cosets
// B\Gamma/Gamma_v (representative point c.v, c the least-index coset representative
// of the double coset, stabiliser B cap c Gamma_v c^{-1}, cyclic), top simplices
// the B-orbits (sigma, r), r in B\Gamma/Gamma_sigma (Gamma_sigma the pointwise
// stabiliser of the triangle, {+-I}), iota_B(sigma, r) = r iota(sigma); every
// point carries an element of B.  InternedMPSet on it is S_B = B\F'_B; every
// arrow lies in B.
//
// (3) THE FORGET-COSET MORPHISM F'_B -> F'_A (prop:mprescor (2)): label table
// (sigma, r^B) -> ((sigma, r^A), h), h = r^B u (r^A)^{-1} in A for the u in
// Gamma_sigma with A r^B u = A r^A (u = I preferred), so iota_B = h iota_A on the
// nose.  Its cochain map is LabelledPullbackRep with L(h) = rho(h): NOT central,
// the label factor is load-bearing ([C, def:mpmorphism] (3), F-MP2 audit Sec. 1).
// For A = Gamma the labels are the representatives r themselves.
//
// (4) THE SHAPIRO COMPARISON C^.(S_Gamma; Coind_B M) -> C^.(S_B; M), induced by the
// B-map F'_B -> F'_Gamma, ((b, (sigma, r))) -> ((b r, sigma)): on normal forms
// u(x_s) = r_{s_0} . x_{u(s)}, so (u^* f)(s) = ev_{r_{s_0}} f(u(s)), ev_h(phi) =
// phi(h) = rho(h w_t^{-1}) phi_t, t = class_B(h) (EvaluationMatrix).  A cochain map
// between two free B-resolutions' cochains over id_Z -- a quasi-isomorphism.
// ===========================================================================

#include <cstdint>
#include <string>
#include <vector>

#include "Descriptor.h"
#include "ModuleCochains.h"
#include "Refinement.h"
#include "../../Hecke/HeckeLevelEll.h"

namespace orbifold_mp {

using hecke::level::CosetLevel;

struct SubgroupModel {
    std::string name;
    Descriptor D;                      // the B-descriptor
    std::vector<uint32_t> sigma;       // B-top index -> Gamma-top index (of the source descriptor)
    std::vector<Mat22> r;              // B-top index -> r with iota_B(t) = r iota(sigma)
    std::vector<uint32_t> r_class;     // class_B(r) (1-based)
    std::vector<std::vector<Mat22>> stab_sigma;   // Gamma-top index -> its pointwise stabiliser (elements)
    uint32_t vertex_orbits = 0;
    bool minus_I_in = false;
};

// ENGINE_CHECKs every point element in B and every B-stabiliser cyclic of the predicted order.
SubgroupModel SubgroupDescriptor(const Descriptor& D, const CosetLevel& B, const std::string& name);

// iota_B(t) = h iota_A(t') as point sets of X (Gamma-point keys of D).
bool SameSimplex(const Descriptor& D, const SubgroupModel& Bm, uint32_t t, const SubgroupModel& Am, uint32_t tA, const Mat22& h);

// The forget-coset labels B -> A (A >= B; ENGINE_CHECK h in A and SameSimplex).  `exact_out` (optional) counts the
// labels with u = I.
RefinementLabels ForgetCosetLabels(const Descriptor& D, const SubgroupModel& Bm, const CosetLevel& B, const SubgroupModel& Am,
                                   const CosetLevel& A, uint32_t* exact_out = nullptr);
// The labels B -> Gamma of the Shapiro comparison and of the forget-coset map to Gamma: (sigma, r).
RefinementLabels LabelsToGamma(const SubgroupModel& Bm);

// Coind_B^Gamma Res Sym^n in the monomial form (dim = [Gamma : B] (n + 1)).
ModuleRep CoindRep(const CosetLevel& B, uint32_t n, uint32_t ell, uint32_t e);

struct ResCor {
    uint32_t index = 0;                // [A : B]
    std::vector<uint32_t> fibre;       // 0-based: a(i) for i in B\Gamma
    Vec res, cor;                      // (m_B d) x (m_A d) and (m_A d) x (m_B d)
};
// `wrong_rep` (VALIDATION ONLY): the negative control "wrong coset representative" -- the block of B-class i0 uses
// v_{i0} T in place of v_{i0}.  `drop_rho` (VALIDATION ONLY): every block the identity.
ResCor ResCorMaps(const CosetLevel& A, const CosetLevel& B, uint32_t n, uint32_t ell, uint32_t e, int64_t wrong_rep = -1,
                  bool drop_rho = false);

// ev_h : Coind_B M -> M, d x (m_B d).
Vec EvaluationMatrix(const CosetLevel& B, const Mat22& h, uint32_t n, uint32_t ell, uint32_t e);

// Dense (dim x dim) products / identities over Z/m for the small coefficient matrices.
Vec MatMulSmall(const Vec& A, const Vec& B, uint32_t r, uint32_t k, uint32_t c, uint32_t m);
Vec IdentitySmall(uint32_t d, uint32_t scalar, uint32_t m);

}  // namespace orbifold_mp

#endif
