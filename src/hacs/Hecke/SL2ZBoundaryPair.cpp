// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 3, WP-5 (phase E5-2) -- the pipeline-side pair of the
// 7-vertex model.  See the header.

#include "SL2ZBoundaryPair.h"

#include "../LocalSystemSL2Z.h"
#include "../../ModArithmetic/FlatMatrix.h"
#include "CoefficientRing.h"
#include "HeckeSettings.h"

#include <stdexcept>
#include <string>

namespace hecke {

SL2ZBoundaryPair::SL2ZBoundaryPair(const EquivariantCohomologyOnK* coh_K, const SimplicialComplex& K)
    : coh_K_(coh_K), K_(&K) {
    // (slot-9 WP-4; E8 item 4b; [E] v1.10.0-pre.1 sec:ring (f)) A bulk over the
    // Z/ell^k IN FORCE is honoured (ell >= 5): the circle model over the same
    // ring, the identity-fibre restriction, both computed.  Every other
    // combination keeps slot 8's refusal (the ring in force against the input's).
    const uint32_t k_in = coh_K ? coh_K->coefficient_ring_k() : 1u;
    if (k_in >= 2 && CoefficientRingK() == k_in) {
        build_composite();
        return;
    }
    // (slot-8 WP-3) [E] sec:ring (b): the pair (and its cone) is item 4b's (slot 9).
    RequireFieldCoefficientRing("SL2ZBoundaryPair", k_in);
    if (coh_K_ == nullptr) throw std::invalid_argument("SL2ZBoundaryPair: null bulk cohomology");
    if (!coh_K_->is_cohomology_computed()) throw std::runtime_error("SL2ZBoundaryPair: the bulk cohomology is not computed");
    ell_ = coh_K_->get_prime();
    const uint32_t d_V = coh_K_->get_representation_dim();
    if (d_V == 0) throw std::runtime_error("SL2ZBoundaryPair: representation dimension 0");
    n_ = d_V - 1;
    if (K_->get_simplices().size() != 3 || K_->get_num_q_simplices(0) != 7)
        throw std::runtime_error("SL2ZBoundaryPair: K is not the 7-vertex model");

    // The five steps of HeckeExt/HeckeBoundary's former build_boundary_side,
    // verbatim and in order (value-preserving lift).
    K_bd_ = CreateSL2ZBoundaryPathTriangulation();
    lssc_bd_ = std::make_unique<LocalSystemSimplicialComplex>(
        K_bd_, GenerateSL2ZBoundaryRelationsElt<uint32_t>(K_bd_, n_, ell_), n_ + 1, ell_);
    coh_bd_ = std::make_unique<EquivariantCohomologyOnK>(lssc_bd_.get());
    coh_bd_->compute_cohomology();
    scm_ = std::make_unique<SimplicialComplexMorphism>(&K_bd_, K_);
    if (!scm_->is_valid()) throw std::runtime_error("SL2ZBoundaryPair: K_bd is not a subcomplex of K");
    res_ = std::make_unique<EquivariantCochainComplexMorphismOnK>(coh_bd_.get(), coh_K_, scm_.get(), FlatMatrix(n_ + 1, ell_));
    res_->compute_induced_maps_on_cohomology();
}

SL2ZBoundaryPair::~SL2ZBoundaryPair() = default;

// (slot-9 WP-4) The Z/ell^k pair: the same five steps as the field
// constructor, with the ring's relation list (GenerateSL2ZBoundaryRelationsRing)
// and the identity fibre over ell^k.  The default SmallEllPolicy refuses
// ell in {2, 3} by name (the cone is honoured at ell >= 5 only).
void SL2ZBoundaryPair::build_composite() {
    if (!coh_K_->is_cohomology_computed()) throw std::runtime_error("SL2ZBoundaryPair: the bulk cohomology is not computed");
    const engine::ModCtx<uint32_t>& bulk_ctx = coh_K_->ring_context();
    const engine::ModCtx<uint32_t> ctx =
        CoefficientRingContext<uint32_t>("SL2ZBoundaryPair", bulk_ctx.ell, bulk_ctx.k);   // refuses ell in {2, 3}
    ring_k_ = ctx.k;
    ell_ = ctx.ell;
    const uint32_t d_V = coh_K_->get_representation_dim();
    if (d_V == 0) throw std::runtime_error("SL2ZBoundaryPair: representation dimension 0");
    n_ = d_V - 1;
    if (K_->get_simplices().size() != 3 || K_->get_num_q_simplices(0) != 7)
        throw std::runtime_error("SL2ZBoundaryPair: K is not the 7-vertex model");
    K_bd_ = CreateSL2ZBoundaryPathTriangulation();
    lssc_bd_ = std::make_unique<LocalSystemSimplicialComplex>(
        K_bd_, GenerateSL2ZBoundaryRelationsRing<uint32_t>(K_bd_, n_, ctx), n_ + 1, ctx.p);
    lssc_bd_->declare_coefficient_ring(ctx.ell, ctx.k);   // (slot-11 WP-12, F-2) the ring as (ell, k)
    coh_bd_ = std::make_unique<EquivariantCohomologyOnK>(lssc_bd_.get());
    coh_bd_->compute_cohomology();
    scm_ = std::make_unique<SimplicialComplexMorphism>(&K_bd_, K_);
    if (!scm_->is_valid()) throw std::runtime_error("SL2ZBoundaryPair: K_bd is not a subcomplex of K");
    res_ = std::make_unique<EquivariantCochainComplexMorphismOnK>(coh_bd_.get(), coh_K_, scm_.get(), FlatMatrix(n_ + 1, ctx.p));
    res_->compute_induced_maps_on_cohomology();
}

uint32_t SL2ZBoundaryPair::dim_H_bulk(uint32_t q) const {
    const auto& d = coh_K_->get_H_eq_dimensions();
    return q < d.size() ? d[q] : 0u;
}

uint32_t SL2ZBoundaryPair::dim_H_bd(uint32_t q) const {
    const auto& d = coh_bd_->get_H_eq_dimensions();
    return q < d.size() ? d[q] : 0u;
}

EquivariantMappingConeOptions SL2ZBoundaryPair::cone_options() const {
    EquivariantMappingConeOptions o;
    o.boundary_model = boundary_model();
    o.boundary_is_group_cohomology_known = true;
    o.boundary_is_group_cohomology = boundary_is_group_cohomology();
    o.name = "Cone(r), SL(2,Z) 7-vertex model rel. the circle model, n = " + std::to_string(n_) + ", ell = " + std::to_string(ell_);
    return o;
}

std::unique_ptr<EquivariantMappingCone> SL2ZBoundaryPair::make_cone() const {
    auto cone = std::make_unique<EquivariantMappingCone>(coh_K_, coh_bd_.get(), res_.get(), cone_options());
    cone->compute();
    return cone;
}

} // namespace hecke
