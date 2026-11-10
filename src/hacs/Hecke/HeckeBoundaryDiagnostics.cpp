// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 2, WP-7 (phase E4b) -- the two recorded diagnostics
// eta_zero / cochain_commutes. See HeckeBoundaryDiagnostics.h.

#include "HeckeBoundaryDiagnostics.h"

#include "SL2ZTessellationTables.h"
#include "../EquivariantCohomologyOnK.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/FlatMatrix.h"
#include "HeckeSettings.h"

#include <array>
#include <map>
#include <sstream>

namespace hecke {

namespace {

// The signed canonical accumulation map of HeckeBoundaryDatum's certification:
// {q, sigma_hat, a, b, c, d} -> coefficient over Z, sign folded in.
using CanonMap = std::map<std::array<int64_t, 6>, int64_t>;

void accumulate(CanonMap& acc, const SL2ZTessellationTables& model, uint32_t q, const ChainTerm& t) {
    if (t.coeff == 0) return;
    const KuSimplexKey key = model.canonical_key(t.gamma, q, t.sigma_idx);
    const std::array<int64_t, 6> k = {
        static_cast<int64_t>(q), static_cast<int64_t>(key.sigma_hat),
        key.gamma_canon.a, key.gamma_canon.b, key.gamma_canon.c, key.gamma_canon.d
    };
    acc[k] = checked_add<int64_t>(acc[k], checked_mul<int64_t>(static_cast<int64_t>(t.coeff),
                                                                 static_cast<int64_t>(key.sign)));
}

bool canon_equal(const CanonMap& x, const CanonMap& y) {
    // Zero entries are equivalent to absent ones.
    for (const auto& kv : x) {
        if (kv.second == 0) continue;
        const auto it = y.find(kv.first);
        if (it == y.end() || it->second != kv.second) return false;
    }
    for (const auto& kv : y) {
        if (kv.second == 0) continue;
        const auto it = x.find(kv.first);
        if (it == x.end() || it->second != kv.second) return false;
    }
    return true;
}

FlatMatrix flatten_rows(const FlatMatrix& cochain) {
    const uint32_t R = cochain.get_rows(), C = cochain.get_cols();
    std::vector<uint32_t> d(static_cast<size_t>(R) * C);
    for (uint32_t r = 0; r < R; ++r)
        for (uint32_t c = 0; c < C; ++c) d[static_cast<size_t>(r) * C + c] = cochain.get_entry(r, c);
    return FlatMatrix(R * C, 1, cochain.get_p(), d);
}

bool flat_equal(const FlatMatrix& X, const FlatMatrix& Y) {
    if (X.get_rows() != Y.get_rows() || X.get_cols() != Y.get_cols()) return false;
    for (uint32_t r = 0; r < X.get_rows(); ++r)
        for (uint32_t c = 0; c < X.get_cols(); ++c)
            if (X.get_entry(r, c) != Y.get_entry(r, c)) return false;
    return true;
}

} // namespace

std::string CuspSupportWalk::to_string() const {
    std::ostringstream oss;
    oss << "cusp-support walk: m=" << num_cosets << " slots=" << slots << " off_boundary=" << slots_off_boundary
        << " differ=" << slots_differ << " equal=" << slots_equal << " -> cusp_supported="
        << (cusp_supported() ? "true" : "false") << " eta_zero=" << (eta_zero() ? "true" : "false");
    for (uint32_t q = 0; q < off_boundary_per_degree.size(); ++q)
        oss << " [q=" << q << ": off_boundary=" << off_boundary_per_degree[q] << " differ=" << differ_per_degree[q] << "]";
    return oss.str();
}

CuspSupportWalk cusp_support_walk(const HeckeTauTable& tau, const HeckeBoundaryTauTable& taub) {
    CuspSupportWalk w;
    const SL2ZTessellationTables& model = taub.model();
    w.num_cosets = taub.num_cosets();
    const uint32_t Q = taub.num_degrees(); // boundary_dim + 1
    w.off_boundary_per_degree.assign(Q, 0);
    w.differ_per_degree.assign(Q, 0);
    for (uint32_t q = 0; q < Q; ++q) {
        const std::vector<uint32_t>& bd = taub.boundary_simplices(q);
        for (uint32_t sigma : bd) {
            for (uint32_t j = 1; j <= w.num_cosets; ++j) {
                ++w.slots;
                const std::vector<ChainTerm>& bulk = tau.tau(j, q, sigma);
                const std::vector<ChainTerm>& bnd = taub.tau(j, q, sigma);
                bool off = false;
                CanonMap cb, cd;
                for (const ChainTerm& t : bulk) {
                    if (!model.is_boundary_simplex(q, t.sigma_idx)) off = true;
                    accumulate(cb, model, q, t);
                }
                for (const ChainTerm& t : bnd) accumulate(cd, model, q, t);
                if (off) { ++w.slots_off_boundary; ++w.off_boundary_per_degree[q]; }
                if (!canon_equal(cb, cd)) {
                    ++w.slots_differ;
                    ++w.differ_per_degree[q];
                    if (w.witnesses.size() < 8) {
                        std::ostringstream oss;
                        oss << "(j=" << j << ",q=" << q << ",sigma=" << sigma << (off ? ",interior-support" : ",boundary-support") << ")";
                        w.witnesses.push_back(oss.str());
                    }
                } else {
                    ++w.slots_equal;
                }
            }
        }
    }
    ENGINE_CHECK(w.slots_equal + w.slots_differ == w.slots);
    return w;
}

std::string CochainCommutes::to_string() const {
    std::ostringstream oss;
    oss << "cochain_commutes(p=" << p << ",nu=" << nu << "): vectors=" << vectors << " failing=" << failing << " -> "
        << (holds() ? "true" : "false");
    for (uint32_t q = 0; q < vectors_per_degree.size(); ++q)
        oss << " [q=" << q << ": " << failing_per_degree[q] << "/" << vectors_per_degree[q] << "]";
    return oss.str();
}

CochainCommutes cochain_commutes(const EquivariantCohomologyOnK& bulk_coh, const HeckeOperatorOnK& bulk_op,
                                 const HeckeBoundaryOperator& bd_op,
                                 const EquivariantCochainComplexMorphismOnK& res) {
    RequireFieldCoefficientRing("cochain_commutes (HeckeBoundaryDiagnostics)", bulk_coh.coefficient_ring_k());   // (slot-8 WP-3)
    ENGINE_CHECK_MSG(bulk_coh.is_cohomology_computed(), "cochain_commutes: bulk cohomology not computed");
    ENGINE_CHECK_MSG(res.get_domain_lssc_cohomology() == &bd_op.cohomology(),
                     "cochain_commutes: the restriction's domain is not the boundary operator's cohomology");
    ENGINE_CHECK_MSG(res.get_codomain_lssc_cohomology() == &bulk_coh,
                     "cochain_commutes: the restriction's codomain is not the bulk cohomology");
    ENGINE_CHECK_MSG(bd_op.ell() == bulk_coh.get_prime() && bd_op.d_V() == bulk_coh.get_representation_dim(),
                     "cochain_commutes: ell / d_V mismatch between bulk and boundary");
    CochainCommutes cc;
    cc.p = bd_op.p_hecke();
    cc.nu = bd_op.nu();
    const uint32_t Q_bulk = static_cast<uint32_t>(bulk_coh.get_H_eq_dimensions().size());
    const uint32_t Q_bd = bd_op.num_degrees();
    cc.vectors_per_degree.assign(Q_bulk, 0);
    cc.failing_per_degree.assign(Q_bulk, 0);
    const auto& C = bulk_coh.get_C_eq_bases_fm();
    for (uint32_t q = 0; q < Q_bulk && q < Q_bd; ++q) {
        if (q >= C.size()) continue;
        for (const FlatMatrix& c : C[q]) {
            const FlatMatrix c_flat = flatten_rows(c);
            const FlatMatrix lhs = res.apply_to_flat_cochain(q, bulk_op.apply_Theta(q, c_flat));
            const FlatMatrix rhs = bd_op.apply_Theta(q, res.apply_to_flat_cochain(q, c_flat));
            ++cc.vectors_per_degree[q];
            ++cc.vectors;
            if (!flat_equal(lhs, rhs)) { ++cc.failing_per_degree[q]; ++cc.failing; }
        }
    }
    // Degrees above the boundary dimension: C^q_bd = 0, r^q = 0, the identity
    // is vacuous (0 vectors compared).
    return cc;
}

} // namespace hecke
