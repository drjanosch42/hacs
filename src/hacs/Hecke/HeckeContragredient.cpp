// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 6, WP-1 (phase E7a) -- see HeckeContragredient.h.
// [C] = docs/latex/gl2_contract.tex, cited by label.

#include "HeckeContragredient.h"

#include "FpPoly.h"
#include "../LocalSystemSL2Z.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/Fnv.h"
#include "HeckeSettings.h"

#include <utility>
#include "../../ModArithmetic/ScalarMod.h"

namespace hecke {

namespace {

using engine::scalar::mulmod;

} // namespace

FlatMatrix PhiNMatrix(uint32_t n_sym, uint32_t ell) {
    ENGINE_CHECK_MSG(ell >= 2, "PhiNMatrix: ell must be >= 2");
    const uint32_t d = n_sym + 1;
    // fact[k] = k! mod ell, k = 0..n.
    std::vector<uint32_t> fact(d, 1u % ell);
    for (uint32_t k = 1; k < d; ++k) fact[k] = mulmod(fact[k - 1], k % ell, ell);
    FlatMatrix Phi = FlatMatrix::CreateZero(d, d, ell);
    for (uint32_t t = 0; t < d; ++t) {
        // [C, eq:BnGram]: B_n(u_t, u_{n-t}) = (-1)^t t! (n-t)!, and
        // Phi_n(u_t) = B_n(u_t, .) = sum_j B_n(u_t, u_j) u_j^vee: column t,
        // row n - t.
        uint32_t v = mulmod(fact[t], fact[n_sym - t], ell);
        if ((t & 1u) && v != 0) v = ell - v;
        Phi.set_entry(n_sym - t, t, v);
    }
    return Phi;
}

bool PhiNIsIsomorphism(uint32_t n_sym, uint32_t ell) {
    // [C, lem:symdual]: iff n! in F_ell^x iff n <= ell - 1.
    return n_sym < ell;
}

FiberBlockTable PhiNFiberBlockTable(const SimplicialComplex& K, uint32_t n_sym, uint32_t ell) {
    const FlatMatrix Phi = PhiNMatrix(n_sym, ell);
    FiberBlockTable table(K.get_simplices().size());
    for (uint32_t q = 0; q < table.size(); ++q) {
        const uint32_t N_q = K.get_num_q_simplices(q);
        table[q].resize(N_q);
        for (uint32_t s = 0; s < N_q; ++s) table[q][s].push_back(PerSimplexBlock{ s, Phi });
    }
    return table;
}

FlatMatrix PairingGram(uint32_t n_sym, uint32_t ell) {
    // [C, prop:pairing](2): <u_t, u_j^vee> = delta_{tj}.
    return FlatMatrix::CreateScalar(n_sym + 1, 1u % ell, ell);
}

uint32_t EvaluationPairing(const FlatMatrix& v, const FlatMatrix& phi) {
    ENGINE_CHECK_MSG(v.get_cols() == 1 && phi.get_cols() == 1 && v.get_rows() == phi.get_rows() &&
                         v.get_p() == phi.get_p(),
                     "EvaluationPairing: v and phi must be columns of one length over one F_ell");
    const uint32_t ell = v.get_p();
    uint64_t acc = 0;
    for (uint32_t j = 0; j < v.get_rows(); ++j) acc = (acc + static_cast<uint64_t>(v.get_entry(j, 0)) * phi.get_entry(j, 0)) % ell;
    return static_cast<uint32_t>(acc);
}

bool HeckeIntertwines(const MatrixT<uint32_t>& A_D, const FlatMatrix& P, const MatrixT<uint32_t>& A_M,
                      uint32_t factor, uint32_t ell, bool* nonvacuous_out) {
    ENGINE_CHECK_MSG(A_D.rows == A_D.cols && A_M.rows == A_M.cols && P.get_rows() == A_D.rows &&
                         P.get_cols() == A_M.rows,
                     "HeckeIntertwines: A_D (d_D x d_D), P (d_D x d_M), A_M (d_M x d_M) shapes disagree");
    const uint32_t dD = A_D.rows, dM = A_M.rows;
    // lhs = A_D P, rhs = factor * P A_M, both dD x dM; plain exact loops (d <= ~200).
    bool equal = true, nonvacuous = false;
    for (uint32_t r = 0; r < dD; ++r) {
        for (uint32_t c = 0; c < dM; ++c) {
            uint64_t lhs = 0, pa = 0;
            for (uint32_t k = 0; k < dD; ++k) lhs = (lhs + static_cast<uint64_t>(A_D.data[r][k]) * P.get_entry(k, c)) % ell;
            for (uint32_t k = 0; k < dM; ++k) pa = (pa + static_cast<uint64_t>(P.get_entry(r, k)) * A_M.data[k][c]) % ell;
            if (pa != 0) nonvacuous = true;
            const uint64_t rhs = (pa * (factor % ell)) % ell;
            if (lhs != rhs) equal = false;
        }
    }
    if (nonvacuous_out) *nonvacuous_out = nonvacuous;
    return equal;
}

ContragredientCell::ContragredientCell(uint32_t n_sym, uint32_t ell)
    : n_(n_sym), ell_(ell), Phi_(PhiNMatrix(n_sym, ell)) {
    // (slot-8 WP-3) [E] sec:ring (b): the E7 cell builds its own F_ell pipeline.
    RequireFieldCoefficientRing("ContragredientCell");
    K_ = std::make_unique<SimplicialComplex>(CreateSL2ZFundamentalDomainTriangulation());
    model_ = std::make_unique<hecke::SL2ZTessellationTables>(*K_);
    for (const auto& datum : GenerateSL2ZRelationsWithGroupData(*K_, n_, ell_)) {
        rels_M_.push_back(datum.rel);
        g_r_.push_back(datum.g_r);
    }
    const std::vector<SL2ZRelationDatum> dual = GenerateSL2ZContragredientRelationsWithGroupData(*K_, n_, ell_);
    ENGINE_CHECK_MSG(dual.size() == g_r_.size(), "ContragredientCell: the two relation lists differ in length");
    for (size_t r = 0; r < dual.size(); ++r) {
        ENGINE_CHECK_MSG(dual[r].g_r == g_r_[r], "ContragredientCell: the two relation lists differ in g_r");
        rels_D_.push_back(dual[r].rel);
    }
    lssc_M_ = std::make_unique<LocalSystemSimplicialComplex>(*K_, rels_M_, n_ + 1, ell_);
    lssc_D_ = std::make_unique<LocalSystemSimplicialComplex>(*K_, rels_D_, n_ + 1, ell_);
    coh_M_ = std::make_unique<EquivariantCohomologyOnK>(lssc_M_.get());
    coh_D_ = std::make_unique<EquivariantCohomologyOnK>(lssc_D_.get());
    coh_M_->compute_cohomology();
    coh_D_->compute_cohomology();
    morph_ = std::make_unique<EquivariantCochainComplexMorphismOnK>(
        coh_D_.get(), coh_M_.get(), PhiNFiberBlockTable(*K_, n_, ell_), "Phi_n (lem:symdual)");
    morph_->compute_induced_maps_on_cohomology();
}

const std::vector<FlatMatrix>& ContragredientCell::induced_Phi() const {
    return morph_->get_induced_H_eq_maps();
}

ContragredientCell::HeckePair ContragredientCell::hecke_pair(uint32_t p) const {
    HeckePair hp;
    hp.p = p;
    hp.cosets = std::make_unique<HeckeCosetTables>(p, 1, ell_, n_);
    hp.tau = std::make_unique<HeckeTauTable>(*K_, *model_, *hp.cosets, ell_, n_);
    hp.op_M = std::make_unique<HeckeOperatorOnK>(coh_M_.get(), hp.cosets.get(), hp.tau.get());
    hp.op_D = std::make_unique<HeckeOperatorOnK>(coh_D_.get(), hp.cosets.get(), hp.tau.get(),
                                                 FibreRepresentation::ContragredientN);
    hp.op_M->compute();
    hp.op_D->compute();
    return hp;
}

std::vector<DualityGoldenObject> ContragredientGoldenObjects(const EquivariantCohomologyOnK& coh_D,
                                                            const std::vector<DualityHeckeInput>& heckes) {
    // Hashes exactly as Golden/GoldenHarness.cpp's Collector::scalar / poly.
    std::vector<DualityGoldenObject> out;
    const auto& dims = coh_D.get_H_eq_dimensions();
    for (uint32_t q = 0; q < dims.size(); ++q)
        out.push_back(DualityGoldenObject{ "dim_H_Dn^" + std::to_string(q), 'A', dims[q], 0,
                                           engine::FnvU64(engine::kFnvOffset64, dims[q]) });
    for (const DualityHeckeInput& h : heckes) {
        ENGINE_CHECK_MSG(h.A_D != nullptr, "ContragredientGoldenObjects: null A(D_n)(T_p) list");
        for (uint32_t q = 0; q < h.A_D->size(); ++q) {
            const MatrixT<uint32_t>& A = (*h.A_D)[q];
            const FpPolyT<uint32_t> cp = (A.rows == 0) ? FpPolyT<uint32_t>(A.p, std::vector<uint32_t>{ 1u })
                                                       : CharPolyBerkowitz(A);
            uint64_t hh = engine::FnvU64(engine::kFnvOffset64, cp.coeffs.size());
            for (uint32_t c : cp.coeffs) hh = engine::FnvU64(hh, static_cast<uint64_t>(c));
            out.push_back(DualityGoldenObject{ "charpoly_A_Dn(T_" + std::to_string(h.p) + ")^" + std::to_string(q),
                                               'A', cp.coeffs.size(), 1, hh });
        }
    }
    return out;
}

} // namespace hecke
