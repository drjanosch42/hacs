// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef SL2Z_BOUNDARY_PAIR_H
#define SL2Z_BOUNDARY_PAIR_H

// hecke-extensions slot 3, WP-5 (phase E5-2) -- the PIPELINE-SIDE pair
// (K, K_bd) of the 7-vertex model with its restriction r = j^*
// ([C] = docs/latex/gl2_contract.tex, cited by label: def:pair;
// prop:circlemodel for the circle model of the boundary).
//
// WHY THIS UNIT EXISTS (slot-3 plan Sec. 2.3 item 4, audit F3).  Until this
// package the only library construction of the restriction morphism lived
// in HeckeExt/HeckeBoundary.cpp (build_boundary_side), which rule 1 of
// docs/matrix-engine/tools/check_heckeext_includes.sh forbids the pipeline
// (Hecke/, the generic cone) to include.  The construction is lifted here
// VERBATIM -- the same calls in the same order -- and HeckeExt/HeckeBoundary
// is re-routed through it, value-preserving (HeckeBoundaryParabolicValidation's
// printed summary lines are the proof).  The cone (EquivariantMappingCone)
// and slot-3 WP-6's cone action consume this object; the ext layer keeps
// consuming it one-way.
//
// What it owns, on one COMPUTED bulk cohomology coh_K of the 7-vertex model K:
//   * K_bd      = CreateSL2ZBoundaryPathTriangulation(): the horocycle path
//                 v0-v1-v2-v3 with K's own vertex labels, simplices in
//                 SL2ZTessellationTables::boundary_simplices(q) order;
//   * lssc_bd   = the circle model C^*_eq(K_bd, Rel_bd; L), Rel_bd = the cusp
//                 relation f(v3) = rho_n(T) f(v0) PLUS the -I self-relations
//                 ([C, prop:circlemodel](4), v1.4.0), GenerateSL2ZBoundaryRelationsElt;
//   * coh_bd    = its cohomology, computed (H^0 = V^T, H^1 = V_T at every ell);
//   * scm       = SimplicialComplexMorphism(&K_bd, &K), the inclusion j (checked valid);
//   * res       = EquivariantCochainComplexMorphismOnK(coh_bd, coh_K, scm, I_{d_V}),
//                 computed: DOMAIN = the boundary (the target of the induced
//                 map on cohomology, the class's contravariant naming),
//                 CODOMAIN = the bulk; r^q = get_induced_H_eq_maps()[q].
// Record labels for the cone ([C, prop:circlemodel](3); owner confirmation 11
// of the slot-1 plan): boundary_model() = "circle",
// boundary_is_group_cohomology() = (2 in R^x) = (ell != 2).
//
// Placement: Hecke/, uint32_t only (the morphism class is), no Matrix.h, no
// raw mod_* call, no unordered_ container; compiled into cohomology_app.

#include "../EquivariantCochainComplexMorphismOnK.h"
#include "../EquivariantCohomologyOnK.h"
#include "../EquivariantMappingCone.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"

#include <cstdint>
#include <memory>

namespace hecke {

class SL2ZBoundaryPair {
public:
    // coh_K: the COMPUTED bulk cohomology on K (the 7-vertex model; checked
    // by vertex count), with prime ell and representation dimension n + 1.
    // K and coh_K must outlive this object.  Everything is built in the
    // constructor (the same five steps as HeckeExt/HeckeBoundary's former
    // build_boundary_side, in the same order); throws std::runtime_error on
    // an inconsistency.
    SL2ZBoundaryPair(const EquivariantCohomologyOnK* coh_K, const SimplicialComplex& K);
    ~SL2ZBoundaryPair();
    SL2ZBoundaryPair(const SL2ZBoundaryPair&) = delete;
    SL2ZBoundaryPair& operator=(const SL2ZBoundaryPair&) = delete;

    uint32_t ell() const { return ell_; }
    uint32_t n_sym() const { return n_; }
    uint32_t d_V() const { return n_ + 1; }

    const SimplicialComplex& K() const { return *K_; }
    const SimplicialComplex& K_bd() const { return K_bd_; }
    const EquivariantCohomologyOnK& bulk_cohomology() const { return *coh_K_; }
    const EquivariantCohomologyOnK& boundary_cohomology() const { return *coh_bd_; }
    const SimplicialComplexMorphism& inclusion() const { return *scm_; }
    const EquivariantCochainComplexMorphismOnK& restriction() const { return *res_; }
    uint32_t dim_H_bulk(uint32_t q) const;
    uint32_t dim_H_bd(uint32_t q) const;

    // (slot-9 WP-4; E8 item 4b) The pair's ring: its bulk's.  Over Z/ell^k the
    // boundary complex and the restriction are over the same ring, and the
    // field accessors dim_H_bulk / dim_H_bd throw (no dim; lengths instead).
    uint32_t coefficient_ring_k() const { return ring_k_; }
    bool is_composite() const { return ring_k_ >= 2; }

    // The record labels of the circle model.
    const char* boundary_model() const { return "circle"; }
    bool boundary_is_group_cohomology() const { return ell_ != 2; }

    // The cone options that carry the labels above (derived degree fields).
    EquivariantMappingConeOptions cone_options() const;
    // Convenience: the cone of this pair, computed.
    std::unique_ptr<EquivariantMappingCone> make_cone() const;

private:
    const EquivariantCohomologyOnK* coh_K_;
    const SimplicialComplex* K_;
    uint32_t ell_ = 0, n_ = 0;
    SimplicialComplex K_bd_;
    std::unique_ptr<LocalSystemSimplicialComplex> lssc_bd_;
    std::unique_ptr<EquivariantCohomologyOnK> coh_bd_;
    std::unique_ptr<SimplicialComplexMorphism> scm_;
    std::unique_ptr<EquivariantCochainComplexMorphismOnK> res_;
    uint32_t ring_k_ = 1;
    void build_composite();   // (slot-9 WP-4) the Z/ell^k constructor tail
};

} // namespace hecke

#endif // SL2Z_BOUNDARY_PAIR_H
