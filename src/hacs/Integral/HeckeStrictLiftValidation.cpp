// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 10, WP-10 (Line 2, L2-1b (1)): HeckeStrictLiftValidation.
// [C] = docs/latex/gl2_contract.tex v1.9.1, cited by label; [IB] =
// docs/implementation/integral_line_design_brief.md.  See the header for the
// arms.  Every cell is summarised on one line.
#include "HeckeStrictLiftValidation.h"

#include "IntegralCharpolyCrt.h"
#include "Model/IsotropyModel.h"
#include "Model/StrictLift.h"

#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"
#include "../Hecke/CoefficientRing.h"
#include "../Hecke/FpPoly.h"
#include "../Hecke/HeckeBoundaryDatum.h"
#include "../Hecke/HeckeCosetTables.h"
#include "../Hecke/HeckeOperatorOnK.h"
#include "../Hecke/HeckeSettings.h"
#include "../Hecke/HeckeTauTable.h"
#include "../Hecke/SL2ZTessellationTables.h"
#include "../../ModArithmetic/DenseElim.h"
#include "../../ModArithmetic/DenseView.h"
#include "../../ModArithmetic/FlatMatrix.h"
#include "../../ModArithmetic/Kernels.h"
#include "../../ModArithmetic/ModCtx.h"

#include <chrono>
#include <cstdint>
#include <exception>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace {

using engine::DenseBuf;
using engine::ModCtx;
using integral_model::IsotropyModel;
using integral_model::StrictLift;
using integral_model::StrictLiftBundle;
using integral_model::StrictLiftCertificate;

struct Fails {
    uint32_t n = 0;
    uint64_t checks = 0;
    uint32_t printed = 0;
    void check(bool ok, const std::string& what) {
        ++checks;
        if (!ok) {
            ++n;
            if (printed++ < 60) std::cout << "  [FAIL] " << what << std::endl;
        }
    }
};

template<class Elt>
DenseBuf<Elt> mul(const ModCtx<Elt>& ctx, const DenseBuf<Elt>& A, const DenseBuf<Elt>& B) {
    ENGINE_CHECK(A.cols == B.rows);
    DenseBuf<Elt> C(A.rows, B.cols);
    if (A.rows == 0 || B.cols == 0 || A.cols == 0) return C;
    engine::kernels::gemm(ctx, engine::cview(A), engine::cview(B), engine::view(C));
    return C;
}

template<class Elt>
bool equal(const DenseBuf<Elt>& A, const DenseBuf<Elt>& B) {
    if (A.rows != B.rows || A.cols != B.cols) return false;
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint32_t j = 0; j < A.cols; ++j)
            if (A.at(i, j) != B.at(i, j)) return false;
    return true;
}

template<class Elt>
DenseBuf<Elt> from_flat(const FlatMatrixT<Elt>& F) {
    DenseBuf<Elt> M(F.get_rows(), F.get_cols());
    for (uint32_t i = 0; i < F.get_rows(); ++i)
        for (uint32_t j = 0; j < F.get_cols(); ++j) M.at(i, j) = F.get_entry(i, j);
    return M;
}

// A list of cochains (each N_q x d_V, flat index s * d_V + k) as the columns of a matrix.
template<class Elt>
DenseBuf<Elt> columns_of(const std::vector<FlatMatrixT<Elt>>& B, uint32_t dim) {
    DenseBuf<Elt> M(dim, static_cast<uint32_t>(B.size()));
    for (uint32_t t = 0; t < B.size(); ++t) {
        const uint32_t r = B[t].get_rows(), c = B[t].get_cols();
        for (uint32_t s = 0; s < r; ++s)
            for (uint32_t v = 0; v < c; ++v) M.at(s * c + v, t) = B[t].get_entry(s, v);
    }
    return M;
}

// One (n, ell[, k]) cell: the model complex, its cohomology, and Theta^s_g of
// the shipped HeckeOperatorOnK on the BoundaryStrict table (and, optionally,
// the canonical operator for the H^1 comparison).
struct Cell {
    SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    std::unique_ptr<hecke::SL2ZTessellationTables> tess;
    std::unique_ptr<LocalSystemSimplicialComplexT<uint32_t>> lssc;
    std::unique_ptr<EquivariantCohomologyOnKT<uint32_t>> coh;
    std::unique_ptr<hecke::HeckeCosetTablesT<uint32_t>> cosets;
    std::unique_ptr<hecke::HeckeBoundaryTauTableT<uint32_t>> taub;
    std::unique_ptr<hecke::HeckeTauTableT<uint32_t>> tau_s;
    std::unique_ptr<hecke::HeckeOperatorOnKT<uint32_t>> op_s;
    std::unique_ptr<hecke::HeckeTauTableT<uint32_t>> tau_c;
    std::unique_ptr<hecke::HeckeOperatorOnKT<uint32_t>> op_c;
};

std::unique_ptr<Cell> build_cell(uint32_t n, uint32_t ell, uint32_t k, uint64_t p, bool canonical) {
    std::unique_ptr<hecke::ScopedCoefficientRing> ring;
    if (k >= 2) ring = std::make_unique<hecke::ScopedCoefficientRing>(k);
    auto c = std::make_unique<Cell>();
    c->tess = std::make_unique<hecke::SL2ZTessellationTables>(c->K);
    if (k >= 2) {
        const ModCtx<uint32_t> ctx =
            hecke::CoefficientRingContext<uint32_t>("HeckeStrictLiftValidation", ell, k);
        auto rels = hecke::GenerateSL2ZRelationsRing<uint32_t>(c->K, n, ctx);
        c->lssc = std::make_unique<LocalSystemSimplicialComplexT<uint32_t>>(c->K, std::move(rels), n + 1, ctx.p);
    } else {
        auto rels = GenerateSL2ZRelationsElt<uint32_t>(c->K, n, ell);
        c->lssc = std::make_unique<LocalSystemSimplicialComplexT<uint32_t>>(c->K, std::move(rels), n + 1, ell);
    }
    c->coh = std::make_unique<EquivariantCohomologyOnKT<uint32_t>>(c->lssc.get());
    c->coh->compute_cohomology();
    c->cosets = std::make_unique<hecke::HeckeCosetTablesT<uint32_t>>(p, 1, ell, n);
    c->taub = std::make_unique<hecke::HeckeBoundaryTauTableT<uint32_t>>(c->K, *c->tess, *c->cosets);
    c->tau_s = std::make_unique<hecke::HeckeTauTableT<uint32_t>>(c->K, *c->tess, *c->cosets, ell, n, *c->taub);
    c->op_s = std::make_unique<hecke::HeckeOperatorOnKT<uint32_t>>(c->coh.get(), c->cosets.get(), c->tau_s.get());
    c->op_s->compute();
    if (canonical) {
        c->tau_c = std::make_unique<hecke::HeckeTauTableT<uint32_t>>(c->K, *c->tess, *c->cosets, ell, n);
        c->op_c = std::make_unique<hecke::HeckeOperatorOnKT<uint32_t>>(c->coh.get(), c->cosets.get(), c->tau_c.get());
        c->op_c->compute();
    }
    return c;
}

bool same_cosets(const hecke::HeckeCosetTablesT<uint32_t>& a, const hecke::HeckeCosetTablesT<uint32_t>& b) {
    if (a.num_cosets() != b.num_cosets() || !(a.g() == b.g())) return false;
    for (uint32_t j = 1; j <= a.num_cosets(); ++j)
        if (!(a.s(j) == b.s(j)) || !(a.delta(j) == b.delta(j))) return false;
    return true;
}

// (C) eps^* Theta^s = Theta^ eps^* on C^q_eq, q = 0, 1, 2.  Returns the number
// of degrees where it holds; `detail` receives the first failing degree.
uint32_t strict_compat(const Cell& c, const StrictLift& lift, const ModCtx<uint32_t>& ctx, uint32_t n,
                       std::string* detail) {
    const IsotropyModel& R = lift.model();
    uint32_t ok = 0;
    for (uint32_t q = 0; q <= 2; ++q) {
        const uint32_t Nq = static_cast<uint32_t>(IsotropyModel::K_simplices()[q].size());
        DenseBuf<uint32_t> Ts, Ceq;
        if (ctx.is_field()) {
            Ts = from_flat(c.op_s->get_Theta_flat_matrices()[q]);
            Ceq = columns_of(c.coh->get_C_eq_bases_fm()[q], Nq * (n + 1));
        } else {
            Ts = c.op_s->Theta_module(q);
            Ceq = c.coh->module_degree(q).C;
        }
        const DenseBuf<uint32_t> E = R.comparison(ctx, n, q);
        const DenseBuf<uint32_t> Th = lift.theta_hat(ctx, n, q);
        const DenseBuf<uint32_t> lhs = mul(ctx, E, mul(ctx, Ts, Ceq));
        const DenseBuf<uint32_t> rhs = mul(ctx, Th, mul(ctx, E, Ceq));
        if (equal(lhs, rhs)) {
            ++ok;
        } else if (detail && detail->empty()) {
            *detail = "degree " + std::to_string(q) + " (dim C^q_eq = " + std::to_string(Ceq.cols) + ")";
        }
    }
    return ok;
}

// The matrix of Theta^ (degree 1) on the classes of the columns of W (a basis
// of H^1 modulo B^1 = im delta^0), column convention T[w_i] = sum_j A_ji [w_j].
// Throws std::runtime_error if Theta^ w_i leaves span(W) + B^1.
// (S13 WP-12, slot-10 WP-14 m-4) The W-coordinates from solve_multi_rhs on [W | im delta^0] are
// unique although im delta^0's columns are dependent: two solutions differ by a kernel vector
// (a, b) with W a in B^1, so a = 0 because eps^* is injective on H^1 classes at ell >= 5.
MatrixT<uint32_t> h1_matrix(const StrictLift& lift, const ModCtx<uint32_t>& ctx, uint32_t n,
                            const DenseBuf<uint32_t>& W) {
    const IsotropyModel& R = lift.model();
    const DenseBuf<uint32_t> d0 = R.coboundary(ctx, n, 0);
    const DenseBuf<uint32_t> Y = mul(ctx, lift.theta_hat(ctx, n, 1), W);
    DenseBuf<uint32_t> Bm(W.rows, W.cols + d0.cols);
    for (uint32_t i = 0; i < W.rows; ++i) {
        for (uint32_t j = 0; j < W.cols; ++j) Bm.at(i, j) = W.at(i, j);
        for (uint32_t j = 0; j < d0.cols; ++j) Bm.at(i, W.cols + j) = d0.at(i, j);
    }
    const auto coords = engine::elim::solve_multi_rhs(ctx, engine::cview(Bm), engine::cview(Y));
    std::vector<std::vector<uint32_t>> A(W.cols, std::vector<uint32_t>(W.cols, 0));
    for (uint32_t i = 0; i < W.cols; ++i)
        for (uint32_t j = 0; j < W.cols; ++j) A[j][i] = coords[i][j];
    return MatrixT<uint32_t>(W.cols, W.cols, ctx.p, A);
}

// An intrinsic basis of H^1(C_I (x) F_ell): a complement of B^1 in Z^1.
DenseBuf<uint32_t> intrinsic_h1(const IsotropyModel& R, const ModCtx<uint32_t>& ctx, uint32_t n) {
    DenseBuf<uint32_t> d1 = R.coboundary(ctx, n, 1);
    const DenseBuf<uint32_t> d0 = R.coboundary(ctx, n, 0);
    const auto rr = engine::elim::rref(ctx, engine::view(d1));
    const DenseBuf<uint32_t> Zrows = engine::elim::kernel_basis(ctx, engine::cview(d1), rr.pivots);
    DenseBuf<uint32_t> Z(Zrows.cols, Zrows.rows);
    for (uint32_t i = 0; i < Zrows.rows; ++i)
        for (uint32_t j = 0; j < Zrows.cols; ++j) Z.at(j, i) = Zrows.at(i, j);
    const auto idx = engine::elim::complement_basis(ctx, engine::cview(d0), engine::cview(Z), static_cast<DenseBuf<uint32_t>*>(nullptr));
    DenseBuf<uint32_t> H(Z.rows, static_cast<uint32_t>(idx.size()));
    for (uint32_t t = 0; t < idx.size(); ++t)
        for (uint32_t i = 0; i < Z.rows; ++i) H.at(i, t) = Z.at(i, idx[t]);
    return H;
}

std::string poly_str(const hecke::FpPoly& f) { return f.to_string(); }

}  // namespace

uint32_t HeckeStrictLiftValidation() {
    const auto t0 = std::chrono::steady_clock::now();
    Fails F;
    std::cout << "=== HeckeStrictLiftValidation (slot-10 WP-10, L2-1b (1): tau_Iso, Theta^_g, strict "
                 "compatibility; [C, prop:strictlift]) ===\n";

    // ---- (A) construction and certificates -------------------------------------
    std::cout << "--- (A) tau_Iso for T_p, degrees 0..6, certified exactly over Z[Gamma''] ---\n";
    std::map<uint64_t, std::unique_ptr<StrictLiftBundle>> bundles;
    for (uint64_t p : {2ull, 3ull, 5ull}) {
        try {
            bundles[p] = std::make_unique<StrictLiftBundle>(p, 5);
        } catch (const std::exception& e) {
            F.check(false, "(A) p = " + std::to_string(p) + ": the strict lift could not be built: " + e.what());
            continue;
        }
        const StrictLift& L = bundles[p]->lift();
        const StrictLiftCertificate C = L.certify();
        F.check(C.t1_twisted_equivariance, "(A) p=" + std::to_string(p) + " (T1) " + C.first_failure);
        F.check(C.t2_chain_map, "(A) p=" + std::to_string(p) + " (T2) " + C.first_failure);
        F.check(C.t3_augmentation, "(A) p=" + std::to_string(p) + " (T3) " + C.first_failure);
        F.check(C.strict_eps, "(A) p=" + std::to_string(p) + " (S) eps tau_Iso = tau^s eps " + C.first_failure);
        F.check(C.boundary_preserved, "(A) p=" + std::to_string(p) + " (B) tau_Iso(R_d) in R_d " + C.first_failure);
        F.check(C.boundary_eps, "(A) p=" + std::to_string(p) + " (B) eps tau^d_Iso = tau_bd eps " + C.first_failure);
        F.check(L.q_top() == 6, "(A) p=" + std::to_string(p) + ": built to q_max + 1 = 6");
        std::cout << "  p=" << p << " m=" << L.num_cosets() << " q_top=" << L.q_top() << ": T1 " << C.t1_checks
                  << " (" << C.schreier_generators << " Schreier gens), T2 " << C.t2_checks << ", T3 " << C.t3_checks
                  << ", S " << C.strict_checks << ", B " << C.boundary_checks << " -> "
                  << (C.all() ? "ALL HOLD" : "FAILED") << "\n    sizes: max|coeff|=" << C.max_abs_coefficient
                  << " max|entry|=" << C.max_abs_entry << " distinct group elements=" << C.distinct_group_elements
                  << "\n    terms/degree:";
        for (size_t q = 0; q < C.terms.size(); ++q)
            std::cout << " q" << q << "=" << C.terms[q] << "(max " << C.max_terms[q] << ", l1 " << C.max_l1[q] << ")";
        std::cout << "\n";
    }
    if (bundles.size() != 3) {
        std::cout << "HeckeStrictLiftValidation: FAILED (" << F.n << " failing check(s)); later arms skipped\n";
        return F.n;
    }

    // ---- (B) Theta^ is a cochain map ------------------------------------------
    std::cout << "--- (B) Theta^_g a cochain map on C_I, q = 0..5, over fields and chain rings ---\n";
    {
        struct RingSpec { uint32_t ell, k; };
        const std::vector<RingSpec> rings = {{2, 1}, {3, 1}, {5, 1}, {7, 1}, {691, 1}, {5, 2}, {7, 3},
                                             {2, 2}, {2, 3}, {3, 2}, {3, 3}};
        uint32_t cells = 0, ok = 0;
        for (const auto& rs : rings)
            for (uint32_t n : {0u, 1u, 2u, 4u, 10u}) {
                const ModCtx<uint32_t> ctx = (rs.k == 1) ? ModCtx<uint32_t>(rs.ell)
                                                          : ModCtx<uint32_t>::prime_power(rs.ell, rs.k);
                for (const auto& [p, B] : bundles) {
                    const StrictLift& L = B->lift();
                    bool all = true;
                    for (uint32_t q = 0; q + 1 <= L.q_top() && q <= L.model().q_max(); ++q) {
                        const DenseBuf<uint32_t> d = L.model().coboundary(ctx, n, q);
                        const bool eq = equal(mul(ctx, d, L.theta_hat(ctx, n, q)), mul(ctx, L.theta_hat(ctx, n, q + 1), d));
                        all = all && eq;
                        F.check(eq, "(B) delta Theta^ != Theta^ delta: p=" + std::to_string(p) + " n=" +
                                        std::to_string(n) + " ring Z/" + std::to_string(rs.ell) + "^" +
                                        std::to_string(rs.k) + " q=" + std::to_string(q));
                    }
                    ++cells;
                    ok += all ? 1 : 0;
                }
            }
        std::cout << "  " << ok << " / " << cells << " (p, n, ring) cells: delta Theta^ = Theta^ delta in every degree\n";

        // The 64-bit element type (the M0 / M1(a) regime Z/ell^K with ell^K >= 2^32):
        // Theta^ over Z/2^40 and Z/3^30 is a cochain map and reduces entrywise to
        // the 32-bit Theta^ over Z/8 and Z/27 (the entries are reductions of one
        // integral matrix).
        uint32_t ok64 = 0, cells64 = 0;
        for (const auto& [ell, K, k32] : {std::tuple<uint64_t, uint32_t, uint32_t>{2, 40, 3}, {3, 30, 3}})
            for (uint32_t n : {0u, 2u, 4u, 10u})
                for (const auto& [p, B] : bundles) {
                    const StrictLift& L = B->lift();
                    const ModCtx<uint64_t> c64 = ModCtx<uint64_t>::prime_power(ell, K);
                    const ModCtx<uint32_t> c32 = ModCtx<uint32_t>::prime_power(static_cast<uint32_t>(ell), k32);
                    bool all = true;
                    for (uint32_t q = 0; q <= L.model().q_max(); ++q) {
                        const DenseBuf<uint64_t> d = L.model().coboundary(c64, n, q);
                        const DenseBuf<uint64_t> T0 = L.theta_hat(c64, n, q);
                        all = all && equal(mul(c64, d, T0), mul(c64, L.theta_hat(c64, n, q + 1), d));
                        const DenseBuf<uint32_t> T32 = L.theta_hat(c32, n, q);
                        bool red = (T32.rows == T0.rows && T32.cols == T0.cols);
                        for (uint32_t i = 0; red && i < T0.rows; ++i)
                            for (uint32_t j = 0; red && j < T0.cols; ++j) red = (T0.at(i, j) % c32.p == T32.at(i, j));
                        all = all && red;
                    }
                    F.check(all, "(B) 64-bit Theta^ over Z/" + std::to_string(ell) + "^" + std::to_string(K) +
                                     ": not a cochain map or not reducing to Z/" + std::to_string(ell) + "^" +
                                     std::to_string(k32) + " (p=" + std::to_string(p) + ", n=" + std::to_string(n) + ")");
                    ++cells64;
                    ok64 += all ? 1 : 0;
                }
        std::cout << "  " << ok64 << " / " << cells64
                  << " 64-bit cells (Z/2^40, Z/3^30): cochain map, and reducing entrywise to the 32-bit Theta^ over "
                     "Z/8, Z/27\n";
    }

    // ---- (C) strict compatibility --------------------------------------------
    std::cout << "--- (C) eps^* Theta^s_g = Theta^_g eps^* on C^q_eq (q = 0,1,2), rings with a shipped Theta^s_g "
                 "(B-1) ---\n";
    {
        struct RingSpec { uint32_t ell, k; };
        const std::vector<RingSpec> rings = {{2, 1}, {3, 1}, {5, 1}, {7, 1}, {5, 2}, {5, 3}, {7, 2}, {7, 3}};
        for (const auto& rs : rings) {
            uint32_t cells = 0, ok = 0, datum_ok = 0;
            std::string first;
            for (uint32_t n = 0; n <= 20; ++n)
                for (uint64_t p : {2ull, 3ull, 5ull}) {
                    if (p == 5 && n > 12) continue;
                    if (p == rs.ell) continue;   // T_p with p != ell (U_ell is not in scope)
                    const StrictLiftBundle& B = *bundles[p];
                    std::unique_ptr<Cell> c;
                    try {
                        c = build_cell(n, rs.ell, rs.k, p, false);
                    } catch (const std::exception& e) {
                        F.check(false, "(C) cell build failed: n=" + std::to_string(n) + " ell=" + std::to_string(rs.ell) +
                                           "^" + std::to_string(rs.k) + " p=" + std::to_string(p) + ": " + e.what());
                        continue;
                    }
                    const bool same = integral_model::SameIntegralDatum(*c->tau_s, B.tau_s()) &&
                                      same_cosets(*c->cosets, B.cosets());
                    datum_ok += same ? 1 : 0;
                    F.check(same, "(C) the cell's tau^s / cosets are not the bundle's integral datum: n=" +
                                      std::to_string(n) + " ell=" + std::to_string(rs.ell) + " p=" + std::to_string(p));
                    const ModCtx<uint32_t> ctx = (rs.k == 1) ? ModCtx<uint32_t>(rs.ell)
                                                              : ModCtx<uint32_t>::prime_power(rs.ell, rs.k);
                    std::string detail;
                    const uint32_t good = strict_compat(*c, B.lift(), ctx, n, &detail);
                    F.check(good == 3, "(C) eps^* Theta^s != Theta^ eps^*: n=" + std::to_string(n) + " ring Z/" +
                                           std::to_string(rs.ell) + "^" + std::to_string(rs.k) + " p=" +
                                           std::to_string(p) + " at " + detail);
                    ++cells;
                    ok += (good == 3) ? 1 : 0;
                }
            std::cout << "  ring " << (rs.k == 1 ? "F_" : "Z/") << rs.ell << (rs.k == 1 ? "" : "^" + std::to_string(rs.k))
                      << ": " << ok << " / " << cells << " (n <= 20, p) cells entrywise equal in q = 0,1,2; datum "
                      << datum_ok << " / " << cells << " identical to the lift's\n";
        }
        std::cout << "  Z/ell^k at ell in {2, 3}: NOT separately checked -- no shipped Theta^s_g there (decision 4); "
                     "the identity follows from (A)'s exact Z-certificate (eq:cosettransport is functorial in the ring).\n"
                  << "  (This arm is a consistency check of the transport code, not of the lift.)\n";
    }

    // ---- (D) the operator on H^1 at ell >= 5 ---------------------------------------
    std::cout << "--- (D) Theta^_g on H^1(C_I (x) F_ell) through eps^* = shipped A^{(1)}(T_p) (canonical tau), "
                 "ell >= 5 ---\n";
    {
        struct DC { uint32_t n, ell; };
        const std::vector<DC> cells = {{10, 691}, {12, 5}, {16, 7}, {20, 691}, {2, 7}, {4, 11},
                                       {14, 13}, {18, 11}, {22, 23}, {24, 691}, {12, 10007}};
        for (const auto& dc : cells)
            for (uint64_t p : {2ull, 3ull, 5ull}) {
                if (p == 5 && dc.n > 12) continue;
                if (p == dc.ell) continue;
                const ModCtx<uint32_t> ctx(dc.ell);
                auto c = build_cell(dc.n, dc.ell, 1, p, true);
                const auto& Hb = c->coh->get_H_eq_bases_fm()[1];
                const uint32_t N1 = static_cast<uint32_t>(IsotropyModel::K_simplices()[1].size());
                const DenseBuf<uint32_t> Z = columns_of(Hb, N1 * (dc.n + 1));
                const StrictLift& L = bundles[p]->lift();
                const DenseBuf<uint32_t> W = mul(ctx, L.model().comparison(ctx, dc.n, 1), Z);
                const MatrixT<uint32_t>& A = c->op_c->get_H_matrices()[1];
                bool same = false;
                std::string why;
                try {
                    const MatrixT<uint32_t> Ah = h1_matrix(L, ctx, dc.n, W);
                    same = (Ah.rows == A.rows && Ah.cols == A.cols && Ah.data == A.data);
                    if (!same) why = " (matrices differ)";
                } catch (const std::exception& e) {
                    why = std::string(" (") + e.what() + ")";
                }
                F.check(same, "(D) (n, ell) = (" + std::to_string(dc.n) + ", " + std::to_string(dc.ell) + ") p=" +
                                  std::to_string(p) + ": Theta^ on H^1 via eps^* != shipped A^(1)(T_p)" + why);
                std::cout << "  (" << dc.n << ", " << dc.ell << ") p=" << p << ": dim H^1 = " << Hb.size() << ", "
                          << (same ? "EQUAL" : "DIFFERENT") << "\n";
            }
    }

    // ---- (E) O5, intrinsically ------------------------------------------------------
    std::cout << "--- (E) char(Theta^_g | H^1(C_I (x) F_ell)) = O5 mod ell, H^1 intrinsic to the free model ---\n";
    for (const integral::O5Row& row : integral::O5Table()) {
        if (row.p != 2 && !(row.p == 3 && row.n == 10)) continue;
        for (uint32_t ell : {691u, 10007u, 1000003u}) {
            const ModCtx<uint32_t> ctx(ell);
            const StrictLift& L = bundles[row.p]->lift();
            const DenseBuf<uint32_t> H = intrinsic_h1(L.model(), ctx, row.n);
            F.check(H.cols == integral::RankLambda1(row.n),
                    "(E) n=" + std::to_string(row.n) + " ell=" + std::to_string(ell) + ": dim H^1 = " +
                        std::to_string(H.cols) + " != rank Lambda^1");
            hecke::FpPoly chi(ell), P(ell);
            bool eq = false;
            try {
                chi = hecke::CharPolyBerkowitz(h1_matrix(L, ctx, row.n, H));
                P = integral::OracleCharpolyMod(row, ell);
                eq = (chi == P);
            } catch (const std::exception& e) {
                F.check(false, std::string("(E) threw: ") + e.what());
            }
            F.check(eq, "(E) n=" + std::to_string(row.n) + " p=" + std::to_string(row.p) + " ell=" + std::to_string(ell) +
                            ": char = " + poly_str(chi) + " != O5 " + row.verbatim);
            std::cout << "  n=" << row.n << " p=" << row.p << " ell=" << ell << ": dim H^1=" << H.cols << ", char "
                      << (eq ? "== " : "!= ") << row.verbatim << " mod ell\n";
        }
    }

    // ---- (F) negative control ---------------------------------------------------------
    std::cout << "--- (F) negative control: one perturbed tau_Iso entry ---\n";
    {
        const StrictLift& L = bundles[2]->lift();
        // A degree-1, t = 0 cell (an edge generator e_{1, rep, 0}) at coset j = 2.
        const uint32_t cell = L.model().cell_index(1, 0, 0);
        const StrictLift bad = L.perturbed(1, 2, cell);
        const StrictLiftCertificate C = bad.certify();
        F.check(!C.all(), "(F) certify() accepted a perturbed tau_Iso");
        F.check(!C.strict_eps, "(F) the strict identity did not catch the perturbation");
        std::cout << "  certify() on the perturbed lift: " << (C.all() ? "ACCEPTED (bad)" : "REJECTED") << " -- "
                  << C.first_failure << "\n";
        auto c = build_cell(10, 7, 1, 2, false);
        std::string detail;
        const uint32_t good = strict_compat(*c, bad, ModCtx<uint32_t>(7), 10, &detail);
        F.check(good < 3, "(F) the strict compatibility (C) did not catch the perturbation at (10, F_7)");
        std::cout << "  strict compatibility at (n, ring) = (10, F_7): " << good << " / 3 degrees hold -> "
                  << (good < 3 ? "CAUGHT at " + detail : std::string("NOT caught (bad)")) << "\n";
    }

    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "HeckeStrictLiftValidation: " << F.checks << " checks, " << F.n << " failing, " << secs << " s\n";
    std::cout << (F.n == 0 ? "HeckeStrictLiftValidation: PASSED\n" : "HeckeStrictLiftValidation: FAILED\n");
    return F.n;
}
