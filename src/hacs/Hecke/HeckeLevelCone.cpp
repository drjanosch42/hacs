// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 23, LES-2) Route A: the level cone -- see the header and [C] sec:nebles.
#include "HeckeLevelCone.h"

#include <utility>

#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/FlatMatrix.h"

namespace hecke {
namespace level {
namespace les {

neb::BlockScalar ScalarOf(const CharacterDatum& chi) {
    const uint64_t N = chi.N == 0 ? 1 : chi.N;
    if (N == 1) return [](const Mat22&) -> uint32_t { return 1u; };
    return neb::TableScalar(N, chi.table);
}

std::unique_ptr<LevelModel> BuildLevelModel(const SimplicialComplex& K, const CosetLevel& A, const CharacterDatum& chi,
                                            uint32_t n_sym, const engine::ModCtx<uint32_t>& ctx) {
    ENGINE_CHECK_MSG(ctx.is_field(), "BuildLevelModel: Route A is over F_ell (the Z/ell^K form is Route B, LES-3)");
    auto M = std::make_unique<LevelModel>();
    M->lvl = &A;
    M->chi = chi;
    M->n_sym = n_sym;
    M->p = ctx.p;
    const uint32_t D = A.m() * (n_sym + 1);
    M->L.reset(new LocalSystemSimplicialComplex(K, neb::InducedRelationsChi(K, A, n_sym, ctx, ScalarOf(chi)), D, ctx.p));
    M->H.reset(new EquivariantCohomologyOnK(M->L.get()));
    M->H->compute_cohomology();
    return M;
}

FiberBlockTable RestrictionBlockTable(const SimplicialComplex& K, const FibreMatrix& alpha) {
    const uint32_t dB = alpha.m_dst() * alpha.d(), dA = alpha.m_src() * alpha.d();
    const FlatMatrix F(dB, dA, alpha.p(), alpha.dense());
    FiberBlockTable tab(K.get_dimension() + 1);
    for (uint32_t q = 0; q <= K.get_dimension(); ++q) {
        const uint32_t N = K.get_num_q_simplices(q);
        tab[q].resize(N);
        for (uint32_t s = 0; s < N; ++s) tab[q][s].push_back(PerSimplexBlock{s, F});
    }
    return tab;
}

std::vector<uint32_t> ApplyFibre(const FibreMatrix& F, uint32_t N_q, const std::vector<uint32_t>& X, uint32_t ncols,
                                 const engine::ModCtx<uint32_t>& ctx) {
    const uint32_t d = F.d(), Ds = F.m_src() * d, Dd = F.m_dst() * d;
    ENGINE_CHECK_MSG(X.size() == static_cast<size_t>(N_q) * Ds * ncols, "ApplyFibre: operand shape");
    std::vector<uint32_t> Y(static_cast<size_t>(N_q) * Dd * ncols, 0);
    const uint64_t p = ctx.p;
    for (uint32_t s = 0; s < N_q; ++s)
        for (uint32_t i = 1; i <= F.m_dst(); ++i)
            for (const auto& jb : F.row(i)) {
                const uint32_t j = jb.first;
                const std::vector<uint32_t>& blk = jb.second;
                for (uint32_t r = 0; r < d; ++r) {
                    uint32_t* y = &Y[(static_cast<size_t>(s) * Dd + (i - 1) * d + r) * ncols];
                    for (uint32_t c = 0; c < d; ++c) {
                        const uint64_t a = blk[static_cast<size_t>(r) * d + c];
                        if (a == 0) continue;
                        const uint32_t* x = &X[(static_cast<size_t>(s) * Ds + (j - 1) * d + c) * ncols];
                        for (uint32_t k = 0; k < ncols; ++k) y[k] = static_cast<uint32_t>((y[k] + a * x[k]) % p);
                    }
                }
            }
    return Y;
}

std::vector<uint32_t> ApplyCoboundary(const SimplicialComplex& K, uint32_t q, uint32_t D, const std::vector<uint32_t>& X,
                                      uint32_t ncols, const engine::ModCtx<uint32_t>& ctx) {
    const uint32_t Nq = K.get_num_q_simplices(q), Nq1 = K.get_num_q_simplices(q + 1);
    ENGINE_CHECK_MSG(X.size() == static_cast<size_t>(Nq) * D * ncols, "ApplyCoboundary: operand shape");
    std::vector<uint32_t> Y(static_cast<size_t>(Nq1) * D * ncols, 0);
    if (Nq == 0 || Nq1 == 0) return Y;
    const FlatMatrix d = K.coboundary_operator_elt<uint32_t>(q, ctx.p);
    ENGINE_CHECK_MSG(d.get_rows() == Nq1 && d.get_cols() == Nq, "ApplyCoboundary: coboundary shape");
    const uint64_t p = ctx.p;
    for (uint32_t i = 0; i < Nq1; ++i)
        for (uint32_t j = 0; j < Nq; ++j) {
            const uint64_t a = d.get_entry(i, j);
            if (a == 0) continue;
            for (uint32_t r = 0; r < D; ++r) {
                uint32_t* y = &Y[(static_cast<size_t>(i) * D + r) * ncols];
                const uint32_t* x = &X[(static_cast<size_t>(j) * D + r) * ncols];
                for (uint32_t k = 0; k < ncols; ++k) y[k] = static_cast<uint32_t>((y[k] + a * x[k]) % p);
            }
        }
    return Y;
}

LevelPairCone::LevelPairCone(const LevelModel& A, const LevelModel& B, const FibreMatrix& alpha, std::string name)
    : A_(&A), B_(&B), alpha_(&alpha), name_(std::move(name)) {
    ENGINE_CHECK_MSG(A.H && B.H && A.H->is_cohomology_computed() && B.H->is_cohomology_computed(),
                     "LevelPairCone: both models must be computed");
    ENGINE_CHECK_MSG(A.p == B.p && alpha.p() == A.p, "LevelPairCone: one field for both models and alpha");
    ENGINE_CHECK_MSG(A.n_sym == B.n_sym && alpha.d() == A.n_sym + 1, "LevelPairCone: one n for both models and alpha");
    ENGINE_CHECK_MSG(alpha.m_src() == A.lvl->m() && alpha.m_dst() == B.lvl->m(), "LevelPairCone: alpha is m_B x m_A blocks");
    // (each local system holds its own copy of K: compare the simplices, not the pointers)
    ENGINE_CHECK_MSG(A.L->get_base_K_complex()->get_simplices() == B.L->get_base_K_complex()->get_simplices(),
                     "LevelPairCone: both models on the same K");
}

void LevelPairCone::compute() {
    if (computed()) return;
    const SimplicialComplex& K = *A_->L->get_base_K_complex();
    // domain = L_B (the target of the induced map on H^*), codomain = L_A: the class's contravariant naming.
    res_.reset(new EquivariantCochainComplexMorphismOnK(B_->H.get(), A_->H.get(), RestrictionBlockTable(K, *alpha_),
                                                        "alpha_{B/A} (" + name_ + ")"));
    res_->compute_induced_maps_on_cohomology();
    EquivariantMappingConeOptions o;
    o.name = "Cone(alpha_{B/A}) " + name_;
    cone_.reset(new EquivariantMappingCone(A_->H.get(), B_->H.get(), res_.get(), o));
    cone_->compute();
}

const EquivariantCochainComplexMorphismOnK& LevelPairCone::restriction() const {
    ENGINE_CHECK_MSG(res_ != nullptr, "LevelPairCone::restriction: compute() has not run");
    return *res_;
}

const EquivariantMappingCone& LevelPairCone::cone() const {
    ENGINE_CHECK_MSG(computed(), "LevelPairCone::cone: compute() has not run");
    return *cone_;
}

LevelPairCone::SixTerm LevelPairCone::six_term() const {
    const EquivariantMappingCone& C = cone();
    SixTerm s;
    const std::vector<uint32_t>& hA = A_->H->get_H_eq_dimensions();
    const std::vector<uint32_t>& hB = B_->H->get_H_eq_dimensions();
    auto at = [](const std::vector<uint32_t>& v, size_t i) { return i < v.size() ? v[i] : 0u; };
    s.h0A = at(hA, 0); s.h1A = at(hA, 1); s.h2A = at(hA, 2);
    s.h0B = at(hB, 0); s.h1B = at(hB, 1); s.h2B = at(hB, 2);
    s.rank_res0 = C.r_built(0) ? C.rank_r(0) : 0;
    s.rank_res1 = C.r_built(1) ? C.rank_r(1) : 0;
    s.rank_res2 = C.r_built(2) ? C.rank_r(2) : 0;
    const std::vector<uint32_t>& c = C.dim_Hc_all();
    s.H0c = at(c, 0); s.H1c = at(c, 1); s.H2c = at(c, 2); s.H3c = at(c, 3);
    bool all = true;
    for (const ConeLesSpot& sp : C.les_spots()) {
        ++s.spots;
        if (sp.asserted) {
            ++s.spots_asserted;
            all = all && sp.exact && (!(sp.in_built && sp.out_built) || sp.composite_zero);
        }
    }
    s.spots_exact = all && s.spots_asserted > 0;
    return s;
}

} // namespace les
} // namespace level
} // namespace hecke
