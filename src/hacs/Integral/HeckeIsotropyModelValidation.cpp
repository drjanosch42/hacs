// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 9, WP-10 (Line 2, L2-1a): HeckeIsotropyModelValidation.
// [C] = docs/latex/gl2_contract.tex v1.8.1, cited by label; [IB] =
// docs/implementation/integral_line_design_brief.md.  See the header for the
// arms.  Every cell is printed.
#include "HeckeIsotropyModelValidation.h"

#include "Model/IsotropyModel.h"

#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../EquivariantCohomologyOnK.h"
#include "../SimplicialCohomology.h"
#include "../../ModArithmetic/DenseElim.h"
#include "../../ModArithmetic/FlatMatrix.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace {

using integral_model::GKey;
using integral_model::IsotropyModel;
using integral_model::Simplex;
using hecke::Mat22;

// ---------------------------------------------------------------------------
// [IB] Appendix A: H^q(SL_2(Z); Sym^n Z^2) over Z, FROZEN.  Per n: rank H^1,
// and the cyclic summands of H^1, H^2 (= H^4 = H^6), H^3 (= H^5) as prime
// powers ([C, lem:bulkperiodic]: H^2 = H^4, H^3 = H^5, period 2 above degree 1).
// H^0 = Z at n = 0 and 0 otherwise; H^q has rank 0 for q >= 2.
// Rows n = 0..8 and the even n = 10..20 are Appendix A verbatim; the odd rows
// n = 9..19, which Appendix A does not print, are the output of the same probe
// (docs/implementation/integral_line_probe.py, the "group" row, run at base
// ea90668 with Q = 7) -- O1 extended, labelled as such, never adjusted.
struct AppendixRow {
    uint32_t n;
    uint32_t r1;
    std::vector<uint32_t> t1, t2, t3;
    bool probe_only;
};
const std::vector<AppendixRow>& AppendixA() {
    static const std::vector<AppendixRow> rows = {
        {0, 0, {}, {4, 3}, {}, false},
        {1, 0, {}, {2}, {}, false},
        {2, 1, {2}, {}, {2, 2}, false},
        {3, 0, {2}, {2}, {2}, false},
        {4, 1, {4, 3}, {4}, {2, 2, 3}, false},
        {5, 0, {2}, {2, 2}, {2}, false},
        {6, 1, {2, 4, 3, 5}, {2, 3}, {2, 2, 2}, false},
        {7, 0, {2}, {2, 2, 2}, {2}, false},
        {8, 1, {2, 8, 3, 7}, {2, 4}, {2, 2, 2}, false},
        {9, 0, {2, 2}, {2, 2, 2}, {2, 2}, true},
        {10, 3, {2, 4, 3}, {2}, {2, 2, 2, 2, 2, 3}, false},
        {11, 0, {2, 2}, {2, 2, 2, 2}, {2, 2}, true},
        {12, 1, {2, 4, 8, 3, 9, 5, 11}, {2, 2, 4, 3}, {2, 2, 2, 2}, false},
        {13, 0, {2, 2}, {2, 2, 2, 2, 2}, {2, 2}, true},
        {14, 3, {2, 2, 4, 3, 13}, {2, 2}, {2, 2, 2, 2, 2, 2}, false},
        {15, 0, {2, 2, 2}, {2, 2, 2, 2, 2}, {2, 2, 2}, true},
        {16, 3, {2, 4, 16, 3, 3, 7}, {2, 2, 4}, {2, 2, 2, 2, 2, 2, 3}, false},
        {17, 0, {2, 2, 2}, {2, 2, 2, 2, 2, 2}, {2, 2, 2}, true},
        {18, 3, {2, 2, 4, 4, 3, 9, 5, 17}, {2, 2, 2, 3}, {2, 2, 2, 2, 2, 2, 2}, false},
        {19, 0, {2, 2, 2}, {2, 2, 2, 2, 2, 2, 2}, {2, 2, 2}, true},
        {20, 3, {2, 2, 4, 8, 3, 3, 5, 19}, {2, 2, 2, 4}, {2, 2, 2, 2, 2, 2, 2}, false},
    };
    return rows;
}

// ell-adic exponent a if v = ell^a (a >= 1), else 0.
uint32_t ell_exponent(uint32_t v, uint32_t ell) {
    uint32_t a = 0;
    while (v > 1 && v % ell == 0) { v /= ell; ++a; }
    return (v == 1) ? a : 0;
}

// Universal coefficients: the length of H^q(C (x) Z/ell^k) for C the free
// integral complex with the Appendix A cohomology, q = 0..5:
//   k * rank H^q + sum_{Z/ell^a in H^q} min(a, k) + sum_{Z/ell^a in H^{q+1}} min(a, k).
std::vector<uint32_t> uct_lengths(const AppendixRow& row, uint32_t ell, uint32_t k) {
    auto tors = [&](uint32_t q) -> const std::vector<uint32_t>& {
        static const std::vector<uint32_t> none;
        if (q == 0) return none;
        if (q == 1) return row.t1;
        return (q % 2 == 0) ? row.t2 : row.t3;   // H^2 = H^4 = H^6, H^3 = H^5
    };
    auto rank = [&](uint32_t q) -> uint32_t { return q == 0 ? (row.n == 0 ? 1 : 0) : (q == 1 ? row.r1 : 0); };
    auto tsum = [&](uint32_t q) {
        uint32_t s = 0;
        for (uint32_t v : tors(q)) {
            const uint32_t a = ell_exponent(v, ell);
            if (a > 0) s += std::min(a, k);
        }
        return s;
    };
    std::vector<uint32_t> out(6);
    for (uint32_t q = 0; q <= 5; ++q) out[q] = k * rank(q) + tsum(q) + tsum(q + 1);
    return out;
}

std::string vec_str(const std::vector<uint32_t>& v) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < v.size(); ++i) os << (i ? "," : "") << v[i];
    os << "]";
    return os.str();
}

// y = M x over ctx (dense).
template<class Elt>
std::vector<Elt> matvec(const engine::ModCtx<Elt>& ctx, const engine::DenseBuf<Elt>& M, const std::vector<Elt>& x) {
    std::vector<Elt> y(M.rows, 0);
    for (uint32_t i = 0; i < M.rows; ++i) {
        Elt acc = 0;
        for (uint32_t j = 0; j < M.cols; ++j)
            if (x[j] != 0 && M.at(i, j) != 0) acc = ctx.add(acc, ctx.mul(M.at(i, j), x[j]));
        y[i] = acc;
    }
    return y;
}

uint32_t rank_of_columns(const engine::ModCtx<uint32_t>& ctx, uint32_t rows,
                         const std::vector<std::vector<uint32_t>>& cols) {
    if (cols.empty() || rows == 0) return 0;
    engine::DenseBuf<uint32_t> M(static_cast<uint32_t>(cols.size()), rows);   // columns as rows
    for (uint32_t c = 0; c < cols.size(); ++c)
        for (uint32_t r = 0; r < rows; ++r) M.at(c, r) = cols[c][r];
    return engine::elim::rref(ctx, engine::view(M)).rank;
}

// Flatten a pipeline K-cochain (N_q x d_V FlatMatrix; row s = value on simplex s).
std::vector<uint32_t> flatten(const FlatMatrix& f) {
    std::vector<uint32_t> v(static_cast<size_t>(f.get_rows()) * f.get_cols());
    for (uint32_t s = 0; s < f.get_rows(); ++s)
        for (uint32_t c = 0; c < f.get_cols(); ++c) v[static_cast<size_t>(s) * f.get_cols() + c] = f.get_entry(s, c);
    return v;
}

// The simplicial coboundary of K on flat V-valued cochains:
// (delta f)(tau) = sum_i (-1)^i f(tau minus its i-th vertex).
std::vector<uint32_t> K_coboundary(const engine::ModCtx<uint32_t>& ctx, const std::vector<uint32_t>& f,
                                   uint32_t q, uint32_t D) {
    const auto& K = IsotropyModel::K_simplices();
    std::vector<uint32_t> out(K[q + 1].size() * D, 0);
    for (uint32_t tau = 0; tau < K[q + 1].size(); ++tau) {
        const Simplex& sim = K[q + 1][tau];
        for (uint32_t i = 0; i < sim.size(); ++i) {
            Simplex face;
            for (uint32_t k = 0; k < sim.size(); ++k) if (k != i) face.push_back(sim[k]);
            const uint32_t fi = static_cast<uint32_t>(std::find(K[q].begin(), K[q].end(), face) - K[q].begin());
            for (uint32_t c = 0; c < D; ++c) {
                const uint32_t v = f[fi * D + c];
                out[tau * D + c] = (i % 2 == 0) ? ctx.add(out[tau * D + c], v) : ctx.sub(out[tau * D + c], v);
            }
        }
    }
    return out;
}

struct PipelineRun {
    std::vector<uint32_t> dims;                          // q = 0, 1, 2
    std::vector<std::vector<std::vector<uint32_t>>> C;   // flat C_eq basis per q
    std::vector<std::vector<std::vector<uint32_t>>> H;   // flat H_eq basis per q
};

PipelineRun run_pipeline(const SimplicialComplex& K, uint32_t n, uint32_t ell, bool want_bases) {
    std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels;
    for (const auto& datum : GenerateSL2ZRelationsWithGroupData(K, n, ell)) rels.push_back(datum.rel);
    LocalSystemSimplicialComplex lssc(K, rels, n + 1, ell);
    EquivariantCohomologyOnK coh(&lssc);
    coh.compute_cohomology();
    PipelineRun out;
    out.dims = coh.get_H_eq_dimensions();
    out.dims.resize(3, 0);
    if (want_bases) {
        const auto& Cb = coh.get_C_eq_bases_fm();
        const auto& Hb = coh.get_H_eq_bases_fm();
        out.C.resize(3);
        out.H.resize(3);
        for (uint32_t q = 0; q < 3; ++q) {
            if (q < Cb.size()) for (const auto& f : Cb[q]) out.C[q].push_back(flatten(f));
            if (q < Hb.size()) for (const auto& f : Hb[q]) out.H[q].push_back(flatten(f));
        }
    }
    return out;
}

// eps^* against the pipeline's own cochains: for every basis vector f of
// C^q_eq, delta_I(eps^* f) == eps^*(delta_K f) (q = 0, 1; at q = 2,
// delta_I(eps^* f) == 0), and eps^* is injective on C^q_eq.  Returns failures.
uint32_t check_eps_cochain_map(const IsotropyModel& M, const engine::ModCtx<uint32_t>& ctx, uint32_t n,
                               const PipelineRun& run, std::string& note) {
    uint32_t fails = 0;
    const uint32_t D = n + 1;
    uint64_t vecs = 0;
    for (uint32_t q = 0; q <= 2; ++q) {
        const auto E = M.comparison(ctx, n, q);
        const auto dI = M.coboundary(ctx, n, q);
        std::vector<std::vector<uint32_t>> images;
        for (const auto& f : run.C[q]) {
            const auto ef = matvec(ctx, E, f);
            images.push_back(ef);
            const auto lhs = matvec(ctx, dI, ef);
            std::vector<uint32_t> rhs(lhs.size(), 0);
            if (q < 2) rhs = matvec(ctx, M.comparison(ctx, n, q + 1), K_coboundary(ctx, f, q, D));
            if (lhs != rhs) ++fails;
            ++vecs;
        }
        const uint32_t rk = rank_of_columns(ctx, M.rank(q) * D, images);
        if (rk != run.C[q].size()) ++fails;
    }
    note = "eps^* chain map on " + std::to_string(vecs) + " C_eq vectors, injective";
    return fails;
}

}  // namespace

uint32_t HeckeIsotropyModelValidation() {
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    uint32_t failures = 0;
    uint32_t checks = 0;
    auto check = [&](bool ok, const std::string& what) {
        ++checks;
        if (!ok) {
            ++failures;
            std::cerr << "HeckeIsotropyModelValidation: FAIL " << what << "\n";
        }
        return ok;
    };

    const IsotropyModel M(5);   // q_max = 5, built to degree 6
    const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();

    // ---- (A) the descriptor against the shipped K and relation list ----------
    {
        check(K.get_simplices() == IsotropyModel::K_simplices(), "(A) K_simplices == CreateSL2ZFundamentalDomainTriangulation");
        const auto data = GenerateSL2ZRelationsWithGroupData(K, 2, 5);
        check(data.size() == 31, "(A) the shipped relation list has 31 rows ([C, def:model7rel])");
        // The transport of every simplex of K to its representative.
        auto transport = [&](uint32_t q, uint32_t idx, uint32_t& rep, Mat22& g) {
            const Simplex& sim = IsotropyModel::K_simplices()[q][idx];
            for (uint32_t r = 0; r < M.reps(q).size(); ++r)
                if (M.reps(q)[r] == sim) { rep = r; g = Mat22::I(); return; }
            for (const auto& nr : M.non_representatives())
                if (nr.face == sim) { rep = nr.rep; g = nr.g; return; }
            rep = 999;
        };
        // Every relation f(lhs) = rho(gamma) f(rhs) (lhs = gamma rhs) is
        // consistent with the model's orbits: same representative rho, and
        // t_l^{-1} gamma t_r in <g_rho> = Stab(rho) ([C, lem:model7coincidences] (1)).
        uint32_t consistent = 0;
        for (const auto& d : data) {
            uint32_t rl = 0, rr = 0;
            Mat22 tl = Mat22::I(), tr = Mat22::I();
            transport(d.rel.q_dim, d.rel.s_lhs_idx_in_Kq, rl, tl);
            transport(d.rel.q_dim, d.rel.s_rhs_idx_in_Kq, rr, tr);
            bool ok = (rl == rr) && rl != 999;
            if (ok) {
                const Mat22 h = tl.inv_unimodular() * d.g_r * tr;
                bool in_stab = false;
                Mat22 p = Mat22::I();
                for (uint32_t j = 0; j < M.stabiliser_order(d.rel.q_dim, rl); ++j) {
                    if (p == h) in_stab = true;
                    p = p * M.stabiliser_generator(d.rel.q_dim, rl);
                }
                ok = in_stab;
            }
            consistent += ok ? 1 : 0;
            check(ok, "(A) relation " + d.rel.tag + " consistent with the model's orbits and stabilisers");
        }
        // Every non-representative transport and every stabiliser generator
        // occurs in the shipped list (directly or inverted).
        uint32_t matched = 0;
        for (const auto& nr : M.non_representatives()) {
            const uint32_t q = static_cast<uint32_t>(nr.face.size()) - 1;
            const auto& Kq = IsotropyModel::K_simplices()[q];
            const uint32_t fi = static_cast<uint32_t>(std::find(Kq.begin(), Kq.end(), nr.face) - Kq.begin());
            const uint32_t ri = static_cast<uint32_t>(std::find(Kq.begin(), Kq.end(), M.reps(q)[nr.rep]) - Kq.begin());
            bool found = false;
            for (const auto& d : data) {
                if (d.rel.q_dim != q) continue;
                if (d.rel.s_lhs_idx_in_Kq == fi && d.rel.s_rhs_idx_in_Kq == ri && d.g_r == nr.g) found = true;
                if (d.rel.s_lhs_idx_in_Kq == ri && d.rel.s_rhs_idx_in_Kq == fi && d.g_r.inv_unimodular() == nr.g) found = true;
            }
            matched += found ? 1 : 0;
            check(found, "(A) transport of a non-representative is a shipped relation");
        }
        uint32_t stabs = 0;
        for (uint32_t s = 0; s < 3; ++s)
            for (uint32_t r = 0; r < M.reps(s).size(); ++r) {
                const auto& Ks = IsotropyModel::K_simplices()[s];
                const uint32_t ri = static_cast<uint32_t>(std::find(Ks.begin(), Ks.end(), M.reps(s)[r]) - Ks.begin());
                bool found = false;
                for (const auto& d : data)
                    if (d.rel.q_dim == s && d.rel.s_lhs_idx_in_Kq == ri && d.rel.s_rhs_idx_in_Kq == ri &&
                        d.g_r == M.stabiliser_generator(s, r))
                        found = true;
                stabs += found ? 1 : 0;
                check(found, "(A) stabiliser generator of a representative is a shipped self-relation");
            }
        std::cout << "  (A) descriptor: K == shipped K; " << consistent << "/31 relations consistent; "
                  << matched << "/4 transports and " << stabs << "/19 stabiliser generators matched\n";
    }

    // ---- (B) ranks ([C, prop:isomodel] (2)) -------------------------------------
    {
        const std::vector<uint32_t> w = {5, 14, 19, 19, 19, 19, 19};
        const std::vector<uint32_t> wb = {3, 6, 6, 6, 6, 6, 6};
        bool ok = true;
        for (uint32_t q = 0; q <= M.q_top(); ++q) {
            ok = check(M.rank(q) == w[q], "(B) w_" + std::to_string(q)) && ok;
            ok = check(M.boundary_rank(q) == wb[q], "(B) w^d_" + std::to_string(q)) && ok;
        }
        const engine::ModCtx<uint32_t> ctx(5);
        uint32_t shapes = 0;
        for (uint32_t n = 0; n <= 40; ++n)
            for (uint32_t q = 0; q <= M.q_max(); ++q) {
                const auto dq = M.coboundary(ctx, n, q);
                const auto bq = M.boundary_coboundary(ctx, n, q);
                const bool sh = dq.cols == w[q] * (n + 1) && dq.rows == w[q + 1] * (n + 1) &&
                                bq.cols == wb[q] * (n + 1) && bq.rows == wb[q + 1] * (n + 1);
                ok = check(sh, "(B) shape of delta^" + std::to_string(q) + " at n=" + std::to_string(n)) && ok;
                ++shapes;
            }
        std::cout << "  (B) ranks w_q = 5,14,19,19,19,19,19 and w^d_q = 3,6,6,...: rank C^q_I = w_q (n+1) on "
                  << shapes << " (n <= 40, q <= 5) coboundaries: " << (ok ? "ok" : "MISMATCH") << "\n";
    }

    // ---- (C) exact certificates ------------------------------------------------
    {
        const auto cert = M.certify();
        check(cert.d_squared_zero, "(C) d o d = 0 over Z[Gamma] " + cert.first_failure);
        check(cert.boundary_subcomplex, "(C) R_d is a subcomplex (IM2)");
        check(cert.eps_chain_map, "(C) eps d = d eps over Z (IM1)");
        check(cert.eps_boundary, "(C) eps(R_d) in C(Xbdry)");
        std::cout << "  (C) certificates over Z[Gamma]: d^2=0 " << cert.d_squared_zero << " ("
                  << cert.d_squared_terms_checked << " products), R_d closed " << cert.boundary_subcomplex
                  << ", eps chain map " << cert.eps_chain_map << " (" << cert.eps_cells_checked
                  << " cells), eps(R_d) in Xbdry " << cert.eps_boundary << "; "
                  << cert.distinct_group_elements << " distinct group elements, max |coefficient| "
                  << cert.max_abs_coefficient << ", max |matrix entry| " << cert.max_abs_entry << "\n";
        // The shape of d: entries per component, degree by degree (d^(2) must
        // be present -- d^(1) d^(1) != 0 at v4, v6 -- or the model degenerates).
        uint32_t n0 = 0, n1 = 0, n2 = 0;
        std::ostringstream per;
        for (uint32_t q = 1; q <= M.q_top(); ++q) {
            uint32_t e0 = 0, e1 = 0, e2 = 0;
            for (const auto& e : M.differential(q)) {
                const auto& src = M.cells(q)[e.col];
                const auto& dst = M.cells(q - 1)[e.row];
                if (src.s == dst.s) ++e0; else if (src.s == dst.s + 1) ++e1; else ++e2;
            }
            per << (q > 1 ? " " : "") << "d_" << q << ":" << e0 << "/" << e1 << "/" << e2;
            n0 += e0; n1 += e1; n2 += e2;
        }
        check(n2 > 0, "(C) d^(2) is non-zero");
        std::cout << "  (C) entries d^(0)/d^(1)/d^(2) per degree: " << per.str() << " (total " << n0 << "/" << n1
                  << "/" << n2 << ")\n";
    }

    // ---- (D) rho_n: the model's = the shipped ComputeSymmetricPowerMatrixFM ------
    {
        const auto G = M.group_elements();
        uint32_t cmp = 0;
        bool ok = true;
        for (const auto& [n, ell] : std::vector<std::pair<uint32_t, uint32_t>>{{24, 5}, {10, 13}, {7, 2}, {12, 3}, {40, 134217757}}) {
            const engine::ModCtx<uint32_t> ctx(ell);
            for (const GKey& g : G) {
                const auto mine = integral_model::rho_n(ctx, integral_model::mat_of(g), n);
                FlatMatrix A(2, 2, ell);
                auto red = [ell](int64_t v) { int64_t r = v % static_cast<int64_t>(ell); return static_cast<uint32_t>(r < 0 ? r + ell : r); };
                A.set_entry(0, 0, red(g[0])); A.set_entry(0, 1, red(g[1]));
                A.set_entry(1, 0, red(g[2])); A.set_entry(1, 1, red(g[3]));
                const FlatMatrix ref = ComputeSymmetricPowerMatrixFM(A, n);
                bool eq = true;
                for (uint32_t i = 0; i <= n; ++i)
                    for (uint32_t j = 0; j <= n; ++j) eq = eq && mine.at(i, j) == ref.get_entry(i, j);
                ok = check(eq, "(D) rho_n agrees with ComputeSymmetricPowerMatrixFM") && ok;
                ++cmp;
            }
        }
        std::cout << "  (D) rho_n(g) == ComputeSymmetricPowerMatrixFM on " << cmp
                  << " (g, n, ell) triples: " << (ok ? "ok" : "MISMATCH") << "\n";
    }

    // ---- (E) ell >= 5 of the golden grid, n <= 24 ---------------------------------
    {
        const std::vector<uint32_t> ells = {5, 13, 691, 10007, 134217757};
        uint32_t cells = 0;
        for (uint32_t ell : ells) {
            const engine::ModCtx<uint32_t> ctx(ell);
            for (uint32_t n = 0; n <= 24; ++n) {
                const auto H = M.cohomology_lengths(ctx, n);
                const PipelineRun run = run_pipeline(K, n, ell, true);
                std::vector<uint32_t> pipe(6, 0);
                for (uint32_t q = 0; q < 3; ++q) pipe[q] = run.dims[q];
                bool ok = check(H == pipe, "(E) dim H^q(C_I (x) F_" + std::to_string(ell) + ") == pipeline at n=" +
                                               std::to_string(n) + ": model " + vec_str(H) + " pipeline " + vec_str(pipe));
                std::string note;
                const uint32_t ef = check_eps_cochain_map(M, ctx, n, run, note);
                ok = check(ef == 0, "(E) eps^* cochain map / injective at ell=" + std::to_string(ell) + " n=" + std::to_string(n)) && ok;
                // eps^* an isomorphism on H^0, H^1: the images of the pipeline's
                // H basis are independent modulo B^q(C_I).
                const uint32_t D = n + 1;
                for (uint32_t q = 0; q <= 1; ++q) {
                    const auto E = M.comparison(ctx, n, q);
                    std::vector<std::vector<uint32_t>> cols;
                    uint32_t rb = 0;
                    if (q > 0) {
                        const auto dprev = M.coboundary(ctx, n, q - 1);
                        for (uint32_t j = 0; j < dprev.cols; ++j) {
                            std::vector<uint32_t> c(dprev.rows);
                            for (uint32_t i = 0; i < dprev.rows; ++i) c[i] = dprev.at(i, j);
                            cols.push_back(c);
                        }
                        rb = rank_of_columns(ctx, M.rank(q) * D, cols);
                    }
                    for (const auto& h : run.H[q]) cols.push_back(matvec(ctx, E, h));
                    const uint32_t rk = rank_of_columns(ctx, M.rank(q) * D, cols);
                    ok = check(rk == rb + run.H[q].size(),
                               "(E) eps^* injective on H^" + std::to_string(q) + " at ell=" + std::to_string(ell) +
                                   " n=" + std::to_string(n)) && ok;
                }
                std::cout << "  (E) ell=" << ell << " n=" << n << ": dim H^0..5 model " << vec_str(H)
                          << " == pipeline " << vec_str(pipe) << "; " << note << "; eps^* iso on H^0, H^1 -- "
                          << (ok ? "ok" : "FAIL") << "\n";
                ++cells;
            }
        }
        std::cout << "  (E) " << cells << " cells (ell in {5, 13, 691, 10007, 134217757}, n <= 24)\n";
    }

    // ---- (F), (G) ell in {2, 3}, n <= 20: Appendix A through UCT ----------------
    {
        uint32_t cells = 0;
        const std::vector<uint32_t> golden_n = {0, 1, 2, 4, 10, 16};
        for (uint32_t ell : {2u, 3u}) {
            const engine::ModCtx<uint32_t> ctx(ell);
            for (const auto& row : AppendixA()) {
                const auto H = M.cohomology_lengths(ctx, row.n);
                const auto want = uct_lengths(row, ell, 1);
                bool ok = check(H == want, "(F) dim H^q(C_I (x) F_" + std::to_string(ell) + ") at n=" + std::to_string(row.n) +
                                               ": model " + vec_str(H) + " Appendix A/UCT " + vec_str(want));
                std::string note = "";
                if (std::find(golden_n.begin(), golden_n.end(), row.n) != golden_n.end()) {
                    const PipelineRun run = run_pipeline(K, row.n, ell, true);
                    const uint32_t ef = check_eps_cochain_map(M, ctx, row.n, run, note);
                    ok = check(ef == 0, "(F) eps^* cochain map / injective on the Bredon complex at ell=" +
                                            std::to_string(ell) + " n=" + std::to_string(row.n)) && ok;
                    note = "; " + note + " (Bredon C_eq(F_" + std::to_string(ell) + "), pipeline dims " + vec_str(run.dims) + ")";
                }
                std::cout << "  (F) ell=" << ell << " n=" << row.n << ": dim H^0..5 model " << vec_str(H)
                          << " == Appendix A/UCT " << vec_str(want) << (row.probe_only ? " [probe row]" : "")
                          << note << " -- " << (ok ? "ok" : "FAIL") << "\n";
                ++cells;
            }
        }
        const auto H0 = M.cohomology_lengths(engine::ModCtx<uint32_t>(2), 0);
        check(H0 == std::vector<uint32_t>(6, 1), "(G) n=0, ell=2: dim H^q = 1 for q = 0..5 (O2 mod 2)");
        std::cout << "  (F) " << cells << " cells; (G) n=0 ell=2: " << vec_str(H0) << " (O2 mod 2: all 1)\n";
    }

    // ---- (H) composite contexts Z/ell^k ------------------------------------------
    {
        uint32_t cells = 0;
        for (const auto& [ell, k, nmax] : std::vector<std::tuple<uint32_t, uint32_t, uint32_t>>{
                 {2, 2, 12}, {2, 3, 8}, {3, 2, 12}, {5, 2, 12}}) {
            const auto ctx = engine::ModCtx<uint32_t>::prime_power(ell, k);
            for (const auto& row : AppendixA()) {
                if (row.n > nmax) continue;
                const auto L = M.cohomology_lengths(ctx, row.n);
                const auto want = uct_lengths(row, ell, k);
                const bool ok = check(L == want, "(H) lengths over Z/" + std::to_string(ell) + "^" + std::to_string(k) +
                                                     " at n=" + std::to_string(row.n) + ": model " + vec_str(L) +
                                                     " Appendix A/UCT " + vec_str(want));
                std::cout << "  (H) Z/" << ell << "^" << k << " n=" << row.n << ": length H^0..5 model " << vec_str(L)
                          << " == Appendix A/UCT " << vec_str(want) << " -- " << (ok ? "ok" : "FAIL") << "\n";
                ++cells;
            }
        }
        // C_I(Z/ell^k) reduces entrywise to C_I(F_ell) ([C, prop:isomodel] (3)).
        const auto c4 = engine::ModCtx<uint32_t>::prime_power(2, 2);
        const engine::ModCtx<uint32_t> c2(2);
        bool red_ok = true;
        for (uint32_t q = 0; q <= M.q_max(); ++q) {
            const auto A = M.coboundary(c4, 6, q);
            const auto B = M.coboundary(c2, 6, q);
            for (uint32_t i = 0; i < A.rows; ++i)
                for (uint32_t j = 0; j < A.cols; ++j) red_ok = red_ok && (A.at(i, j) % 2 == B.at(i, j));
        }
        check(red_ok, "(H) C_I(Z/4) reduces to C_I(F_2) entrywise at n = 6");
        std::cout << "  (H) " << cells << " composite cells; C_I(Z/4) mod 2 == C_I(F_2) at n=6: " << (red_ok ? "ok" : "FAIL") << "\n";
    }

    // ---- (I) the boundary subcomplex R_d ----------------------------------------
    {
        uint32_t cells = 0;
        for (uint32_t ell : {5u, 13u}) {
            const engine::ModCtx<uint32_t> ctx(ell);
            for (uint32_t n = 0; n <= 24; ++n) {
                // H^q(Gamma_d; V), Gamma_d = <-I> x <T>, 2 invertible: H^q(<T>; V^{-I}):
                // (dim V^T, dim V^T, 0, ...) for n even, 0 for n odd.
                std::vector<uint32_t> want(6, 0);
                if (n % 2 == 0) {
                    FlatMatrix T(2, 2, ell);
                    T.set_entry(0, 0, 1); T.set_entry(0, 1, 1); T.set_entry(1, 1, 1);
                    const FlatMatrix R = ComputeSymmetricPowerMatrixFM(T, n) - FlatMatrix(n + 1, ell);
                    want[0] = want[1] = (n + 1) - R.rank();
                }
                const auto H = M.cohomology_lengths(ctx, n, true);
                const bool ok = check(H == want, "(I) R_d at ell=" + std::to_string(ell) + " n=" + std::to_string(n) +
                                                     ": model " + vec_str(H) + " expected " + vec_str(want));
                if (n % 6 == 0 || !ok)
                    std::cout << "  (I) R_d ell=" << ell << " n=" << n << ": " << vec_str(H) << " == H^q(<T>; V^{-I}) "
                              << vec_str(want) << " -- " << (ok ? "ok" : "FAIL") << "\n";
                ++cells;
            }
        }
        const auto H2 = M.cohomology_lengths(engine::ModCtx<uint32_t>(2), 0, true);
        const auto H3 = M.cohomology_lengths(engine::ModCtx<uint32_t>(3), 0, true);
        check(H2 == std::vector<uint32_t>({1, 2, 2, 2, 2, 2}), "(I) R_d at ell=2, n=0: H^q(Z/2 x Z; F_2)");
        check(H3 == std::vector<uint32_t>({1, 1, 0, 0, 0, 0}), "(I) R_d at ell=3, n=0: H^q(Z/2 x Z; F_3)");
        std::cout << "  (I) " << cells << " cells at ell in {5, 13}; n=0: ell=2 " << vec_str(H2)
                  << " (Z/2 x Z: 1,2,2,...), ell=3 " << vec_str(H3) << " (1,1,0,...)\n";
    }

    const double secs = std::chrono::duration<double>(clk::now() - t0).count();
    std::cout << "HeckeIsotropyModelValidation: " << checks << " checks, " << failures << " failing ("
              << secs << " s, indicative)\n";
    return failures;
}
