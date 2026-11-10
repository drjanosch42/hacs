// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// See the header.
#include "HeckeThetaBlockValidation.h"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "../../../ModArithmetic/BlockKernels.h"
#include "../../../ModArithmetic/BlockMat.h"
#include "../../../ModArithmetic/DenseBuf.h"
#include "../../../ModArithmetic/DenseView.h"
#include "../../EquivariantCohomologyOnK.h"
#include "../../LocalSystemSL2Z.h"
#include "../../SimplicialCohomology.h"
#include "../HeckeCosetTables.h"
#include "../HeckeOperatorOnK.h"
#include "../HeckeTauTable.h"
#include "../SL2ZTessellationTables.h"

namespace {

using engine::BlockMat;
using engine::BlockRef;
using engine::DenseBuf;
using engine::DenseView;
using engine::Layout;
using engine::cview;
using engine::view;
using Clock = std::chrono::steady_clock;

uint32_t g_fail = 0;
uint64_t g_checks = 0;
void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        if (g_fail <= 60) std::cout << "  [FAIL] " << what << std::endl;
    }
}
template<class T> std::string str(T v) { return std::to_string(v); }

template<class Elt>
bool same_fm(const FlatMatrixT<Elt>& a, const FlatMatrixT<Elt>& b) {
    if (a.get_rows() != b.get_rows() || a.get_cols() != b.get_cols() || a.get_p() != b.get_p()) return false;
    for (uint32_t i = 0; i < a.get_rows(); ++i)
        for (uint32_t j = 0; j < a.get_cols(); ++j)
            if (a.get_entry(i, j) != b.get_entry(i, j)) return false;
    return true;
}

template<class Elt>
bool same_mat(const MatrixT<Elt>& a, const MatrixT<Elt>& b) {
    if (a.rows != b.rows || a.cols != b.cols) return false;
    for (uint32_t i = 0; i < a.rows; ++i)
        for (uint32_t j = 0; j < a.cols; ++j)
            if (a.data[i][j] != b.data[i][j]) return false;
    return true;
}

// Non-owning row-major view of a FlatMatrix (materialised if scalar-flagged).
template<class Elt>
DenseBuf<Elt> borrow_flat(const FlatMatrixT<Elt>& m) {
    m.populate_flat_data();
    const std::vector<Elt>& fd = m.get_flat_data_unsafe();
    return DenseBuf<Elt>(const_cast<Elt*>(fd.data()), fd.empty() ? 1 : fd.size(),
                         m.get_rows(), m.get_cols(), Layout::RowMajor);
}

// The (sigma, sigma_k) pairs with a non-zero coefficient mod ell that the
// tau-table names in degree q -- the admissible pattern of Theta^{(q)}.
template<class Elt>
std::set<std::pair<uint32_t, uint32_t>> tau_pattern(const hecke::HeckeTauTableT<Elt>& tau, uint32_t m,
                                                    uint32_t q, uint32_t N_q, Elt ell) {
    std::set<std::pair<uint32_t, uint32_t>> pat;
    for (uint32_t sigma = 0; sigma < N_q; ++sigma)
        for (uint32_t i = 1; i <= m; ++i)
            for (const hecke::ChainTerm& ct : tau.tau(i, q, sigma)) {
                int64_t r = static_cast<int64_t>(ct.coeff) % static_cast<int64_t>(ell);
                if (r < 0) r += static_cast<int64_t>(ell);
                if (r != 0) pat.insert({sigma, ct.sigma_idx});
            }
    return pat;
}

// (b) + (d) on one computed operator.
template<class Elt>
void check_block_objects(const hecke::HeckeOperatorOnKT<Elt>& op, const hecke::HeckeTauTableT<Elt>& tau,
                         const SimplicialComplex& K, uint32_t m, uint32_t d_V, Elt ell,
                         const std::string& tag, uint64_t& refs_total, uint64_t& windows_distinct) {
    const auto& flat = op.get_Theta_flat_matrices();
    check(flat.size() == 3, tag + ": 3 degrees of Theta_flat");
    using Win = std::tuple<uint32_t, uint32_t, uint32_t, uint8_t>;
    std::map<Win, DenseBuf<Elt>> seen;   // window -> materialised block (consistency)
    for (uint32_t q = 0; q < 3 && q < flat.size(); ++q) {
        const uint32_t N_q = K.get_num_q_simplices(q);
        const BlockMat<Elt>& T = op.Theta_block(q);
        check(T.nbr() == N_q && T.nbc() == N_q, tag + " q=" + str(q) + ": Theta_block grid is N_q x N_q");
        check(T.rows() == N_q * d_V && T.cols() == N_q * d_V, tag + " q=" + str(q) + ": Theta_block scalar shape");
        bool uniform = true;
        for (uint32_t s = 0; s < N_q; ++s) uniform = uniform && T.row_dim(s) == d_V && T.col_dim(s) == d_V;
        check(uniform, tag + " q=" + str(q) + ": uniform (N_q, d_V) partition on both axes");
        check(T.p() == ell, tag + " q=" + str(q) + ": Theta_block modulus");
        // Equality with the flat shim as a mathematical matrix.
        const DenseBuf<Elt> F = borrow_flat<Elt>(flat[q]);
        check(engine::blk::equal_dense<Elt>(T, cview(F)), tag + " q=" + str(q) + ": Theta_block == Theta_flat");
        // Pattern is inside the tau-table's pattern; no Zero refs stored.
        const std::set<std::pair<uint32_t, uint32_t>> pat = tau_pattern<Elt>(tau, m, q, N_q, ell);
        bool inside = true, nonzero_refs = true, consistent = true;
        for (uint32_t br = 0; br < T.nbr(); ++br)
            for (uint32_t k = T.row_begin(br); k < T.row_end(br); ++k) {
                const BlockRef<Elt>& r = T.ref_at(k);
                if (r.is_zero()) nonzero_refs = false;
                if (pat.count({br, T.col_at(k)}) == 0) inside = false;
                ++refs_total;
                if (r.has_data()) {
                    const Win w{r.pool_id, r.r0, r.c0, r.flags};
                    DenseBuf<Elt> M(d_V, d_V);
                    engine::kernels::copy<Elt>(T.data_view(r, d_V, d_V), view(M));
                    auto it = seen.find(w);
                    if (it == seen.end()) { seen.emplace(w, std::move(M)); ++windows_distinct; }
                    else if (!it->second.same_entries(M)) consistent = false;
                }
            }
        check(inside, tag + " q=" + str(q) + ": structural non-zeros lie in the tau-table pattern");
        check(nonzero_refs, tag + " q=" + str(q) + ": no Zero ref stored");
        check(consistent, tag + " q=" + str(q) + ": refs naming one pool window materialise identically");
        // Every tau pair with a non-zero coefficient whose accumulated block is
        // non-zero must be present: equal_dense above already proves the
        // values; here we only assert the flat matrix is zero outside `pat`.
        bool zero_outside = true;
        for (uint32_t s = 0; s < N_q && zero_outside; ++s)
            for (uint32_t t = 0; t < N_q && zero_outside; ++t) {
                if (pat.count({s, t})) continue;
                for (uint32_t a = 0; a < d_V && zero_outside; ++a)
                    for (uint32_t b = 0; b < d_V; ++b)
                        if (flat[q].get_entry(s * d_V + a, t * d_V + b) != 0) { zero_outside = false; break; }
            }
        check(zero_outside, tag + " q=" + str(q) + ": Theta_flat vanishes outside the tau-table pattern");
    }
}

// (c): the public apply API on the full standard basis.
template<class Elt>
void check_apply(const hecke::HeckeOperatorOnKT<Elt>& op, const SimplicialComplex& K, uint32_t d_V, Elt ell,
                 const std::string& tag) {
    const auto& flat = op.get_Theta_flat_matrices();
    for (uint32_t q = 0; q < 3; ++q) {
        const uint32_t dim = K.get_num_q_simplices(q) * d_V;
        bool ok_flat = true, ok_r2 = true;
        for (uint32_t b = 0; b < dim; ++b) {
            FlatMatrixT<Elt> e_b(dim, 1, ell);
            e_b.set_entry(b, 0, Elt(1));
            const FlatMatrixT<Elt> y1 = op.apply_Theta(q, e_b);
            const FlatMatrixT<Elt> y2 = op.apply_route2(q, e_b);
            if (y1.get_rows() != dim || y2.get_rows() != dim) { ok_flat = ok_r2 = false; break; }
            for (uint32_t r = 0; r < dim; ++r) {
                if (y1.get_entry(r, 0) != flat[q].get_entry(r, b)) ok_flat = false;
                if (y1.get_entry(r, 0) != y2.get_entry(r, 0)) ok_r2 = false;
            }
        }
        check(ok_flat, tag + " q=" + str(q) + ": apply_Theta(e_b) == column b of Theta_flat, all b");
        check(ok_r2, tag + " q=" + str(q) + ": apply_Theta(e_b) == apply_route2(e_b), all b");
    }
}

template<class Elt>
void run_case(const SimplicialComplex& K, const hecke::SL2ZTessellationTables& model,
              Elt ell, uint32_t n_sym, uint64_t p_hecke, uint64_t& refs_total, uint64_t& windows_distinct,
              double& t_engine, double& t_legacy) {
    const std::string tag = "(ell=" + str(ell) + ", n=" + str(n_sym) + ", p=" + str(p_hecke) +
                            (sizeof(Elt) == 8 ? ", u64)" : ")");
    const uint32_t d_V = n_sym + 1;
    try {
        const auto rels = GenerateSL2ZRelationsElt<Elt>(K, n_sym, ell);
        const LocalSystemSimplicialComplexT<Elt> lssc(K, rels, d_V, ell);
        EquivariantCohomologyOnKT<Elt> coh(&lssc);
        coh.compute_cohomology();
        const hecke::HeckeCosetTablesT<Elt> cosets(p_hecke, 1, ell, n_sym);
        const hecke::HeckeTauTableT<Elt> tau(K, model, cosets, ell, n_sym);
        const uint32_t m = cosets.num_cosets();

        // Engine arm.
        hecke::HeckeOperatorOnKT<Elt> op_e(&coh, &cosets, &tau);
        const auto t0 = Clock::now();
        op_e.compute();
        t_engine += std::chrono::duration<double>(Clock::now() - t0).count();

        // Legacy arm, same inputs.
        hecke::HeckeOperatorOnKT<Elt> op_l(&coh, &cosets, &tau);
        {
            hecke::HeckeOperatorOptions::ScopedLegacyHeckeOperatorPath force(true);
            const auto t1 = Clock::now();
            op_l.compute();
            t_legacy += std::chrono::duration<double>(Clock::now() - t1).count();
        }

        // (a) entrywise agreement of everything published.
        const auto& Te = op_e.get_Theta_flat_matrices();
        const auto& Tl = op_l.get_Theta_flat_matrices();
        const auto& Ee = op_e.get_Theta_eq_matrices();
        const auto& El = op_l.get_Theta_eq_matrices();
        const auto& Ae = op_e.get_H_matrices();
        const auto& Al = op_l.get_H_matrices();
        check(Te.size() == 3 && Tl.size() == 3 && Ee.size() == 3 && El.size() == 3 && Ae.size() == 3 && Al.size() == 3,
              tag + ": three degrees on both arms");
        for (uint32_t q = 0; q < 3; ++q) {
            check(same_fm<Elt>(Te[q], Tl[q]), tag + " q=" + str(q) + ": Theta_flat engine == legacy");
            check(same_fm<Elt>(Ee[q], El[q]), tag + " q=" + str(q) + ": Theta_eq engine == legacy");
            check(same_mat<Elt>(Ae[q], Al[q]), tag + " q=" + str(q) + ": A^{(q)} engine == legacy");
        }
        check(op_e.to_json() == op_l.to_json(), tag + ": to_json() (incl. certificate counters) engine == legacy");

        // (b) + (d) on both arms (the legacy arm rebuilds Theta_block from its
        // dense output; the accessor contract is the same).
        uint64_t dummy_refs = 0, dummy_windows = 0;
        check_block_objects<Elt>(op_e, tau, K, m, d_V, ell, tag + " [engine]", refs_total, windows_distinct);
        check_block_objects<Elt>(op_l, tau, K, m, d_V, ell, tag + " [legacy]", dummy_refs, dummy_windows);

        // (c) on the engine arm (the binding one).
        check_apply<Elt>(op_e, K, d_V, ell, tag + " [engine]");
    } catch (const std::exception& e) {
        check(false, tag + ": exception: " + e.what());
    }
}

} // namespace

uint32_t HeckeThetaBlockValidation() {
    std::cout << "=== HeckeThetaBlockValidation (matrix-engine Phase 3 Stage B, HeckeOperatorOnK on the block model) ==="
              << std::endl;
    g_fail = 0;
    g_checks = 0;
    const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    const hecke::SL2ZTessellationTables model(K);

    uint64_t refs_total = 0, windows_distinct = 0;
    double t_engine = 0, t_legacy = 0;
    const std::vector<uint32_t> ns = {0, 1, 2, 3, 4, 6, 8, 12};
    const std::vector<uint32_t> ells = {3, 5, 7, 134217757u};
    const std::vector<uint64_t> ps = {2, 3};
    uint32_t cases = 0;
    for (uint32_t ell : ells)
        for (uint32_t n : ns)
            for (uint64_t p : ps) {
                run_case<uint32_t>(K, model, ell, n, p, refs_total, windows_distinct, t_engine, t_legacy);
                ++cases;
            }
    // The uint64_t instantiation, extended regime.
    run_case<uint64_t>(K, model, (uint64_t(1) << 61) - 1, 4, 2, refs_total, windows_distinct, t_engine, t_legacy);
    ++cases;

    std::cout << "  cases " << cases << ", checks " << g_checks << ", failing " << g_fail
              << "; engine-arm compute() total " << t_engine << " s, legacy-arm " << t_legacy << " s" << std::endl;
    std::cout << "  fused-route Theta refs (all cases, all q): " << refs_total << ", distinct pool windows behind them: "
              << windows_distinct << " (shared blocks, not copies)" << std::endl;
    if (g_fail == 0) std::cout << "HeckeThetaBlockValidation: PASSED" << std::endl;
    return g_fail;
}
