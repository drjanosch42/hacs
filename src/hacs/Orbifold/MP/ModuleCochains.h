// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef ORBIFOLD_MP_MODULE_COCHAINS_H
#define ORBIFOLD_MP_MODULE_COCHAINS_H

// ===========================================================================
// (hecke-extensions slot 22, MP-3) The MP cochains with an ARBITRARY finite
// coefficient module, and the maps MP-3 needs between them ([C] 1.22.0-pre.6
// sec:mpmodel: def:mpset, def:mpmorphism, thm:mpnatural, prop:mpboundary,
// prop:mprescor; MP_impl_plan.md v0.2.1 Sec. 2.5).
//
// MODULES.  A ModuleRep is a free Z/m-module of rank dim (m = ell^e <= 255) with a
// left Gamma-action, given as dense matrices rho(g) (dim x dim, row-major,
// entries < m), memoised by GKey.  Sym^n = integral_model::rho_n (the shipped
// [C, def:rhon]); Coind (SubgroupMP.h) = the monomial induced module.
//
// THE MAPS, all as CSR (SparseRows, rows = target coordinates):
//  * CoboundaryRep: d^k of C^.(S; N), (df)(s) = rho(gamma_1) f(d_0 s) + sum_{i >= 1} (-1)^i f(d_i s)
//    ([C, def:mpset]); for N = Sym^n it is Nerve.h's Coboundary entry for entry (checked by the validations).
//  * LabelledPullbackRep: the cochain map of a labelled morphism ([C, def:mpmorphism] (3)):
//        (u^* f)(s) = L(h_{s_0}) f(u(s)),   u(s) = (s'_i ; h_{s_{i-1}}^{-1} gamma_i h_{s_i}),
//    L(h) a d_out x d_in matrix -- rho(h) for a morphism of Sym^n-complexes (THE LABEL FACTOR), the
//    evaluation ev_h : Coind_B M -> M for the Shapiro comparison.  `drop_factor` replaces L(h) by the
//    identity (the negative control of thm:mpnatural (4)).  A degenerate image gives the zero row.
//  * CoefficientMap: a Gamma-module map phi : N -> N' applied string by string (res, cor of prop:mprescor (1)).
//  * PhiPullbackRep: Phi^*_q : C^q_I(N) -> C^q(S; N) for MP-2's certified Phi (Comparison.h), any module.
//  * CICoboundaryRep: delta^q of C_I(N) = Hom_Gamma(Iso_., N) (IsotropyModel), dense, optionally on the
//    boundary subcomplex R_d (= Hom(Isob_., N), [C, prop:isomodel] (IM2)).
//  * ConeDifferential: the cone of a cochain map phi : C -> D in the RELATIVE indexing
//        Cone^q = C^q (+) D^{q-1},  d(c, x) = (d c, phi(c) - d x),
//    so that ... -> H^q(Cone) -> H^q(C) -> H^q(D) -> H^{q+1}(Cone) -> ... (H^q(Cone) = H^q(X, dX) for
//    phi = j^*, = H^q(A, B; M) for phi = res).
// Exact; no randomisation; no golden object; no setting.  Pure host C++.
// ===========================================================================

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "CochainMaps.h"
#include "Comparison.h"
#include "MP2Common.h"
#include "Nerve.h"
#include "Refinement.h"
#include "SparseLength.h"
#include "../../Integral/Model/IsotropyModel.h"
#include "../../../ModArithmetic/DenseBuf.h"
#include "../../../ModArithmetic/ModCtx.h"

namespace orbifold_mp {

class ModuleRep {
public:
    using Fn = std::function<Vec(const Mat22&)>;
    ModuleRep(std::string name, uint32_t dim, uint32_t ell, uint32_t e, Fn f);
    const std::string& name() const { return name_; }
    uint32_t dim() const { return dim_; }
    uint32_t ell() const { return ell_; }
    uint32_t e() const { return e_; }
    uint32_t m() const { return m_; }
    // rho(g), dim x dim row-major, entries < m (memoised; ENGINE_CHECK the shape).
    const Vec& rho(const Mat22& g) const;
    engine::ModCtx<uint32_t> ctx() const { return RingCtx(ell_, e_); }

private:
    std::string name_;
    uint32_t dim_, ell_, e_, m_;
    Fn f_;
    std::shared_ptr<std::map<GKey, Vec>> memo_;
};

// Sym^n (Z/ell^e)^2 with integral_model::rho_n.
ModuleRep SymRep(uint32_t n, uint32_t ell, uint32_t e);

// d^k : C^k(S; N) -> C^{k+1}(S; N).  ENGINE_CHECK: k + 1 <= S.k_max(), m <= 255.
SparseRows CoboundaryRep(const InternedMPSet& S, uint32_t k, const ModuleRep& N);

// (u^* f)(s) = L(h_{s_0}) f(u(s)): rows Sf_q^nd x d_out, columns Sc_q^nd x d_in.  ENGINE_CHECK: no image outside S_c.
using LabelMatrix = std::function<Vec(const Mat22&)>;   // d_out x d_in, row-major, entries < m
SparseRows LabelledPullbackRep(const InternedMPSet& Sf, const InternedMPSet& Sc, const RefinementLabels& L, const StringMap& U,
                               uint32_t q, uint32_t d_out, uint32_t d_in, uint32_t m, const LabelMatrix& Lmat,
                               bool drop_factor = false);

// phi (d_out x d_in, row-major, entries < m) on every non-degenerate string of S_q.
SparseRows CoefficientMap(const InternedMPSet& S, uint32_t q, const Vec& phi, uint32_t d_out, uint32_t d_in, uint32_t m);

// Phi^*_q with coefficients N: rows S_q^nd x dim, columns w_q x dim.
SparseRows PhiPullbackRep(const PhiLift& Phi, const InternedMPSet& S, uint32_t q, uint32_t w_q, const ModuleRep& N);

// delta^q of C_I(N) (rows cells(q + 1) x dim, columns cells(q) x dim), dense; boundary_only: on R_d.
engine::DenseBuf<uint32_t> CICoboundaryRep(const integral_model::IsotropyModel& CI, uint32_t q, const ModuleRep& N,
                                           bool boundary_only);
// The restriction C^q_I(N) -> C^q_{R_d}(N) (rows boundary cells(q) x dim, columns cells(q) x dim), dense.
engine::DenseBuf<uint32_t> CIRestrictToBoundary(const integral_model::IsotropyModel& CI, uint32_t q, uint32_t dim);
// phi on every cell of C^q_I (dense; rows cells(q) x d_out, columns cells(q) x d_in).
engine::DenseBuf<uint32_t> CICoefficientMap(const integral_model::IsotropyModel& CI, uint32_t q, const Vec& phi,
                                            uint32_t d_out, uint32_t d_in, uint32_t m);

// d^q_Cone (relative indexing, header).  dC = d^q_C (C^q -> C^{q+1}), phi = phi^q (C^q -> D^q), dDm1 = d^{q-1}_D
// (D^{q-1} -> D^q; nullptr at q = 0), dimDm1 = dim D^{q-1}.
SparseRows ConeDifferential(const SparseRows& dC, const SparseRows& phi, const SparseRows* dDm1, uint32_t dimDm1, uint32_t m);
// The same, dense (for C_I).
engine::DenseBuf<uint32_t> DenseCone(const engine::DenseBuf<uint32_t>& dC, const engine::DenseBuf<uint32_t>& phi,
                                     const engine::DenseBuf<uint32_t>* dDm1, uint32_t dimDm1, uint32_t m);

// The MP side of a cohomology computation with module N: d^0 .. d^{q_top}, lengths.
struct RepSide {
    uint32_t ell = 2, e = 1, m = 2, dim = 1;
    std::vector<SparseRows> d;
    std::vector<uint64_t> len_im, dimC;
    bool have_lengths = true;
    // lengths = false: the differentials only (for maps into this complex and cones); lenH() then refuses.
    RepSide(const InternedMPSet& S, const ModuleRep& N, uint32_t q_top, bool lengths = true);
    int64_t lenH(uint32_t q) const;
    ImageInH image(uint32_t q, const std::vector<Vec>& mapped, bool cocycle_check = true) const;
};

// Cohomology lengths of a complex given by its differentials d^0 .. d^{q_top} (CSR) and dim C^0 .. C^{q_top}.
std::vector<int64_t> LengthsOf(const std::vector<SparseRows>& d, const std::vector<uint64_t>& dimC, uint32_t ell, uint32_t e);

// The C_I side with module N (dense): delta^0 .. delta^{q_top}, the cocycles and lengths of H^0 .. H^{q_top}.
struct CIRepSide {
    uint32_t ell = 2, e = 1, m = 2;
    std::vector<engine::DenseBuf<uint32_t>> d;
    std::vector<SourceH> src;
    std::vector<uint32_t> dimC;
    CIRepSide(const integral_model::IsotropyModel& CI, const ModuleRep& N, uint32_t q_top, bool boundary_only);
    int64_t lenH(uint32_t q) const { return static_cast<int64_t>(src.at(q).lenH()); }
    ImageInH image(uint32_t q, const std::vector<Vec>& mapped) const;
};
// Lengths of H^q of a dense complex (d^0 .. d^{q_top}, dim C^0 .. C^{q_top}).
std::vector<int64_t> DenseLengthsOf(const engine::ModCtx<uint32_t>& ctx, const std::vector<engine::DenseBuf<uint32_t>>& d,
                                    const std::vector<uint32_t>& dimC);

// Is B A = A' C (both products CSR over Z/m)?  (The chain-map identity phi^{q+1} d^q_A = d^q_B phi^q.)
bool CommutesExactly(const SparseRows& phiNext, const SparseRows& dA, const SparseRows& dB, const SparseRows& phi, uint32_t m);

}  // namespace orbifold_mp

#endif
