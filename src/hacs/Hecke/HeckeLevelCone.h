// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_LEVEL_CONE_H
#define HECKE_LEVEL_CONE_H
// ===========================================================================
// (hecke-extensions slot 23, LES-2) ROUTE A: THE LEVEL CONE Cone(alpha_{B/A}) ON THE FIXED 7-VERTEX MODEL.
// Binding text: [C] v1.22.0-pre.3 sec:nebles -- def:levelcone, lem:levelconeexact (eq:levelsixterm), prop:levelsplit,
// prop:modeltorsion, prop:levelconehecke, obl:levelconeexact (ii), (iii), (v), (vi).  Spec: LES0_spec.md v0.1 section 3
// (Route A); plan docs/implementation/slot23/00-slot23-plan.md section 2 LES-2; predictions
// docs/implementation/slot23/LES2_predictions.md (committed before the first run).
//
// WHAT IS BUILT (a thin assembly; the cone itself is the generic EquivariantMappingCone):
//  * LevelModel / BuildLevelModel -- the character model L^chi_A of lem:inducedchar on (K, R) over F_ell (any prime,
//    ell = 2, 3 included: the model's Bredon cohomology there, prop:modeltorsion), through the shipped
//    InducedRelationsChi (eq:MHchi), its cohomology computed (intermediates retained: the cone reads C_eq).
//  * RestrictionBlockTable -- alpha_{B/A} of eq:levelres (LES-1's RestrictionFibreMap) as the constant per-simplex fibre
//    table of an EquivariantCochainComplexMorphismOnK  C^*_eq(L_A) -> C^*_eq(L_B)  (lem:levelresmorph (3): alpha_* is
//    post-composition with the module map, one d_B x d_A block at every simplex).
//  * LevelPairCone -- the morphism and the cone  Cone(alpha)^q = C^{q+1}(L_A) (+) C^q(L_B),  D(c, c') = (-dc, dc' + alpha c)
//    (def:levelcone = eq:conediff with r := alpha_*), i.e. EquivariantMappingCone with bulk = L_A, boundary = L_B (the
//    unequal-fibre form, LES-2 step 0).  Indexing as there: the stored degree q is H^q(A, B) := H^{q-1}(Cone(alpha)),
//    so the cone's (pi, r, iota) ARE the six-term sequence eq:levelsixterm: r^q = res^q, iota^q : H^q(B) -> H^{q+1}(A,B),
//    pi^q : H^q(A,B) -> H^q(A); its ConeCertificate (C1-C13) is obl:levelconeexact (ii), (iii) on the stored bases.
//  * ApplyFibre / ApplyCoboundary -- alpha and the simplicial coboundary on flat cochains (validation instruments for the
//    d alpha = alpha d and zig-zag certificates).
//
// Ring: F_ell only (the Z/ell^K form is Route B, LES-3; the cone's module route keeps the equal-fibre refusal).
// Nothing here is reached from a shipped record, report, campaign job id, setting or golden text.
// ===========================================================================

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "HeckeLevelPair.h"
#include "HeckeNebentypus.h"
#include "../EquivariantCochainComplexMorphismOnK.h"
#include "../EquivariantCohomologyOnK.h"
#include "../EquivariantMappingCone.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"
#include "../../ModArithmetic/ModCtx.h"

namespace hecke {
namespace level {
namespace les {

// h |-> chi(a_h mod N), the eq:MHchi / eq:nebcorr block scalar of a character datum (N = 1: the constant 1).
neb::BlockScalar ScalarOf(const CharacterDatum& chi);

struct LevelModel {
    const CosetLevel* lvl = nullptr;     // not owned
    CharacterDatum chi;
    uint32_t n_sym = 0;
    uint32_t p = 0;                      // F_p
    std::unique_ptr<LocalSystemSimplicialComplex> L;
    std::unique_ptr<EquivariantCohomologyOnK> H;
    uint32_t d() const { return lvl->m() * (n_sym + 1); }
};
// The character model of (A, chi) on K over ctx (a field), cohomology computed.  ENGINE_CHECK: ctx is a field.
std::unique_ptr<LevelModel> BuildLevelModel(const SimplicialComplex& K, const CosetLevel& A, const CharacterDatum& chi,
                                            uint32_t n_sym, const engine::ModCtx<uint32_t>& ctx);

// alpha (m_B x m_A blocks of d) as the constant fibre table of every simplex of K in degrees 0 .. top (d_B x d_A blocks).
FiberBlockTable RestrictionBlockTable(const SimplicialComplex& K, const FibreMatrix& alpha);

// Y = F applied simplex-wise to flat q-cochains: X row-major (N_q * F.m_src() * d) x ncols -> (N_q * F.m_dst() * d) x ncols.
std::vector<uint32_t> ApplyFibre(const FibreMatrix& F, uint32_t N_q, const std::vector<uint32_t>& X, uint32_t ncols,
                                 const engine::ModCtx<uint32_t>& ctx);
// Y = (delta_q (x) I_D) X: the simplicial coboundary C^q(K; F^D) -> C^{q+1}(K; F^D) on row-major flat cochains.
std::vector<uint32_t> ApplyCoboundary(const SimplicialComplex& K, uint32_t q, uint32_t D, const std::vector<uint32_t>& X,
                                      uint32_t ncols, const engine::ModCtx<uint32_t>& ctx);

class LevelPairCone {
public:
    // A, B: computed models over the same F_ell with B <= A (alpha from RestrictionFibreMap(A, chi_A, B, chi_B)).
    // All three must outlive this object.
    LevelPairCone(const LevelModel& A, const LevelModel& B, const FibreMatrix& alpha, std::string name);
    // Builds the morphism alpha_* (its induced maps) and the cone; the cone's own certificate runs inside and a failing
    // asserted item throws std::runtime_error (that is how a wrong alpha -- not a Gamma-map -- is caught).
    void compute();
    bool computed() const { return cone_ != nullptr && cone_->computed(); }

    const LevelModel& A() const { return *A_; }
    const LevelModel& B() const { return *B_; }
    const FibreMatrix& alpha() const { return *alpha_; }
    const EquivariantCochainComplexMorphismOnK& restriction() const;
    const EquivariantMappingCone& cone() const;

    // eq:levelsixterm, read off the cone (H^q(A,B) = dim_Hc(q)).
    struct SixTerm {
        uint32_t h0A = 0, h1A = 0, h2A = 0, h0B = 0, h1B = 0, h2B = 0;
        uint32_t rank_res0 = 0, rank_res1 = 0, rank_res2 = 0;
        uint32_t H0c = 0, H1c = 0, H2c = 0, H3c = 0;
        bool spots_exact = false;          // every spot of the cone's certificate asserted and exact
        uint32_t spots = 0, spots_asserted = 0;
        uint32_t ker_res1() const { return h1A - rank_res1; }
    };
    SixTerm six_term() const;

private:
    const LevelModel* A_;
    const LevelModel* B_;
    const FibreMatrix* alpha_;
    std::string name_;
    std::unique_ptr<EquivariantCochainComplexMorphismOnK> res_;
    std::unique_ptr<EquivariantMappingCone> cone_;
};

} // namespace les
} // namespace level
} // namespace hecke

#endif // HECKE_LEVEL_CONE_H
