// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 9, WP-11 (Line 2, M0): HeckeFreeModelValidation.
// [C] = docs/latex/gl2_contract.tex v1.8.1, cited by label; [IB] =
// docs/implementation/integral_line_design_brief.md.  See the header for the
// arms.  Every cell is printed.
#include "HeckeFreeModelValidation.h"

#include "Descent/DescentOracle.h"
#include "Model/IntegralGroups.h"
#include "Model/IsotropyModel.h"

#include "../EquivariantCohomologyOnK.h"
#include "../Hecke/CoefficientRing.h"
#include "../Hecke/HeckeSettings.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"
#include "../../ModArithmetic/DenseElim.h"
#include "../../ModArithmetic/DenseView.h"
#include "../../ModArithmetic/EngineCheck.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

using integral_model::AbelianGroupType;
using integral_model::IsotropyModel;
using integral_model::ModelIntegralCohomology;
using U32Ctx = engine::ModCtx<uint32_t>;

constexpr uint32_t kGoodPrime = 10007;   // >= 5 and > n + 1 for every n <= 20
constexpr uint32_t kNMax = 20;
constexpr uint32_t kQMax = 5;

uint32_t g_checks = 0;
uint32_t g_fail = 0;
bool check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        std::cerr << "HeckeFreeModelValidation: FAIL " << what << "\n";
    }
    return ok;
}

std::string vec_str(const std::vector<uint32_t>& v) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < v.size(); ++i) os << (i ? "," : "") << v[i];
    os << "]";
    return os.str();
}

// ---------------------------------------------------------------------------
// O1, FROZEN: H^q(SL_2(Z); Sym^n Z^2), q = 0..6, n = 0..20 -- the "group" rows of
// docs/implementation/integral_line_probe.py, copied VERBATIM (provenance unchanged).  Since slot-24 DESCENT D0
// the table and its parser live in Integral/Descent/DescentOracle.cpp (moved verbatim, one copy for both
// validations); never adjusted.
const std::vector<std::vector<AbelianGroupType>>& O1() { return integral_descent::AppendixAGroups(); }

// ---- the shipped 7-vertex pipeline -------------------------------------------
struct FieldPipe {
    std::vector<uint32_t> dims;          // dim H^q, q = 0..2
    std::vector<uint32_t> C_dims;        // dim C^q_eq, q = 0..2
    uint32_t num_degrees = 0;            // the OBJECT's own degree count (WP-8 m-6): the
                                          // triangulation is 2-dimensional, so this complex
                                          // never has a q = 3, 4, 5 -- checked before dims
                                          // is padded to size 3 below.
};
FieldPipe field_pipeline(uint32_t n, uint32_t ell) {
    const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels;
    for (const auto& datum : GenerateSL2ZRelationsWithGroupData(K, n, ell)) rels.push_back(datum.rel);
    LocalSystemSimplicialComplex lssc(K, rels, n + 1, ell);
    EquivariantCohomologyOnK coh(&lssc);
    coh.compute_cohomology();
    FieldPipe out;
    out.dims = coh.get_H_eq_dimensions();
    out.num_degrees = static_cast<uint32_t>(out.dims.size());
    out.dims.resize(3, 0);
    for (uint32_t q = 0; q < 3; ++q) out.C_dims.push_back(coh.C_eq_dimension(q));
    return out;
}

// y = A B over ctx (dense, naive).
engine::DenseBuf<uint32_t> matmul(const U32Ctx& ctx, const engine::DenseBuf<uint32_t>& A,
                                  const engine::DenseBuf<uint32_t>& B) {
    ENGINE_CHECK(A.cols == B.rows);
    engine::DenseBuf<uint32_t> C(A.rows, B.cols);
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint32_t k = 0; k < A.cols; ++k) {
            const uint32_t a = A.at(i, k);
            if (a == 0) continue;
            for (uint32_t j = 0; j < B.cols; ++j)
                if (B.at(k, j) != 0) C.at(i, j) = ctx.add(C.at(i, j), ctx.mul(a, B.at(k, j)));
        }
    return C;
}
bool equal(const engine::DenseBuf<uint32_t>& A, const engine::DenseBuf<uint32_t>& B) {
    if (A.rows != B.rows || A.cols != B.cols) return false;
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint32_t j = 0; j < A.cols; ++j)
            if (A.at(i, j) != B.at(i, j)) return false;
    return true;
}
bool is_zero(const engine::DenseBuf<uint32_t>& A) {
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint32_t j = 0; j < A.cols; ++j)
            if (A.at(i, j) != 0) return false;
    return true;
}
uint32_t len_of(const U32Ctx& ctx, const engine::DenseBuf<uint32_t>& A) {
    if (A.rows == 0 || A.cols == 0) return 0;
    return engine::elim::length<uint32_t>(ctx, engine::cview(A));
}

// ---- a Dickson cell over Z/ell^K: the K-mode and the cone of eps^* ------------
struct DicksonCell {
    uint32_t n = 0, ell = 0, K = 0;
    bool certified = false;
    std::vector<AbelianGroupType> ladic;       // q = 0..2: the Z_(ell)-type of H^q(C_eq(Z))_(ell)
    std::vector<uint32_t> cone_lengths;        // length H^q(Cone eps^*), q = -1 .. 5 (index q + 1)
    bool eps_chain_map = false;
    bool cone_d2_zero = false;
};

// C_eq(Z/ell^K) through the shipped pipeline (ell >= 5), K = 2, 3, ... until
// the K-mode certifies ([C, prop:certifiede](5) on C_eq(Z) at ell >= 5); then
// the cone of eps^* : C_eq(Z/ell^K) -> C_I(Z/ell^K) in the C_eq basis the
// pipeline stores (module_degree(q).C, d^q = module_degree(q).X).
DicksonCell dickson_cell(const IsotropyModel& M, uint32_t n, uint32_t ell) {
    DicksonCell out;
    out.n = n;
    out.ell = ell;
    const uint32_t budget = std::min<uint32_t>(8, integral_model::DefaultKBudget(ell));
    for (uint32_t K = 2; K <= budget; ++K) {
        hecke::ScopedCoefficientRing ring(K);
        const U32Ctx ctx = U32Ctx::prime_power(ell, K);
        SimplicialComplex K7 = CreateSL2ZFundamentalDomainTriangulation();
        auto rels = hecke::GenerateSL2ZRelationsRing<uint32_t>(K7, n, ctx);
        LocalSystemSimplicialComplex lssc(K7, std::move(rels), n + 1, ctx.p);
        EquivariantCohomologyOnK coh(&lssc);
        coh.compute_cohomology();
        if (!coh.ladic_type_certified()) continue;
        out.K = K;
        out.certified = true;
        for (uint32_t q = 0; q < 3; ++q) {
            const auto t = coh.ladic_type(q);
            AbelianGroupType g;
            g.free_rank = t.free_rank;
            for (uint32_t a : t.torsion) g.add_torsion(ell, a);
            out.ladic.push_back(g);
        }
        // ---- the cone ([C, prop:isomodel](4); X3) ----
        const uint32_t D = n + 1;
        std::vector<uint32_t> c(3), w;
        std::vector<engine::DenseBuf<uint32_t>> E(3), X(3);
        for (uint32_t q = 0; q < 3; ++q) {
            const auto& md = coh.module_degree(q);
            c[q] = md.C.cols;
            X[q] = md.X;   // c_{q+1} x c_q (0 rows at q = 2)
            E[q] = matmul(ctx, M.comparison(ctx, n, q), md.C);   // (w_q D) x c_q
        }
        for (uint32_t q = 0; q <= M.q_top(); ++q) w.push_back(M.rank(q) * D);
        std::vector<engine::DenseBuf<uint32_t>> dI;
        for (uint32_t q = 0; q <= M.q_max(); ++q) dI.push_back(M.coboundary(ctx, n, q));
        // eps^* a chain map in coordinates: dI^q E_q = E_{q+1} X_q (q = 0, 1), dI^2 E_2 = 0.
        bool cm = true;
        for (uint32_t q = 0; q < 3; ++q) {
            const auto lhs = matmul(ctx, dI[q], E[q]);
            if (q < 2) cm = cm && equal(lhs, matmul(ctx, E[q + 1], X[q]));
            else cm = cm && is_zero(lhs);
        }
        out.eps_chain_map = cm;
        auto cdim = [&](int j) -> uint32_t { return (j >= 0 && j <= 2) ? c[static_cast<uint32_t>(j)] : 0u; };
        auto wdim = [&](int q) -> uint32_t { return (q >= 0 && q <= static_cast<int>(M.q_top())) ? w[static_cast<uint32_t>(q)] : 0u; };
        // D^q : Cone^q = C_7^{q+1} (+) C_I^q  ->  Cone^{q+1},  D(a, b) = (-d a, eps^* a + d b).
        auto cone_d = [&](int q) {
            const uint32_t ca = cdim(q + 1), cb = wdim(q), ra = cdim(q + 2), rb = wdim(q + 1);
            engine::DenseBuf<uint32_t> Dq(ra + rb, ca + cb);
            if (q + 1 >= 0 && q + 1 <= 2) {
                const uint32_t j = static_cast<uint32_t>(q + 1);
                for (uint32_t r = 0; r < X[j].rows && r < ra; ++r)
                    for (uint32_t s = 0; s < ca; ++s) Dq.at(r, s) = ctx.neg(X[j].at(r, s));
                for (uint32_t r = 0; r < rb; ++r)
                    for (uint32_t s = 0; s < ca; ++s) Dq.at(ra + r, s) = E[j].at(r, s);
            }
            if (q >= 0 && q <= static_cast<int>(M.q_max())) {
                const auto& d = dI[static_cast<uint32_t>(q)];
                for (uint32_t r = 0; r < rb; ++r)
                    for (uint32_t s = 0; s < cb; ++s) Dq.at(ra + r, ca + s) = d.at(r, s);
            }
            return Dq;
        };
        std::vector<engine::DenseBuf<uint32_t>> Dc;   // Dc[q + 1] = D^q, q = -1 .. q_max
        for (int q = -1; q <= static_cast<int>(M.q_max()); ++q) Dc.push_back(cone_d(q));
        bool d2 = true;
        for (size_t i = 0; i + 1 < Dc.size(); ++i) d2 = d2 && is_zero(matmul(ctx, Dc[i + 1], Dc[i]));
        out.cone_d2_zero = d2;
        for (int q = -1; q <= static_cast<int>(M.q_max()); ++q) {
            const uint32_t total = (cdim(q + 1) + wdim(q)) * K;
            const uint32_t lin = len_of(ctx, Dc[static_cast<size_t>(q + 1)]);
            const uint32_t lprev = (q >= 0) ? len_of(ctx, Dc[static_cast<size_t>(q)]) : 0u;
            out.cone_lengths.push_back(total - lin - lprev);
        }
        break;
    }
    return out;
}

// ---- the tests-(1)/(2) harness and its negative control ([X] test (6)) ---------
enum class ComplexKind { FreeModelTruncated, SevenVertexModel };
struct HarnessRun {
    bool refused = false;
    std::string reason;
    std::vector<uint32_t> dims;   // dim H^q, q = 0..q_max (empty when refused)
};
// [C, rmk:truncdevice]: the periodicity tests read group cohomology, which
// only the free model computes (truncated = true, q_max >= 4 so that both
// H^2 ~ H^4 and H^3 ~ H^5 are reported).  The shipped 7-vertex complex is
// the Bredon object at ell in {2, 3} ([C, prop:bredon](3)); the harness refuses
// it and asserts nothing there.  The q_max floor is the validation's, never
// the pipeline's.
//
// (hecke-extensions slot 10, WP-8, m-6): the SevenVertexModel branch used to
// refuse purely on the CALLER's ComplexKind label -- a self-referential
// control that catches a bug in this file's own dispatch, never a bug in
// shipped code.  It now builds the real shipped complex and refuses on an
// OBJECT property instead: field_pipeline()'s num_degrees is the triangulated
// complex's OWN degree count (3, q = 0..2, [C, def:model7]) -- it is not a
// free resolution at q_max >= 4 because it has no such degree, not because
// this function was told so.  A future change that fed a genuine free
// resolution through this branch would therefore no longer be refused merely
// because it asked to be.
HarnessRun harness(ComplexKind kind, uint32_t q_max, uint32_t n, uint32_t ell) {
    HarnessRun r;
    if (kind == ComplexKind::SevenVertexModel) {
        const FieldPipe seven = field_pipeline(n, ell);
        if (q_max >= seven.num_degrees) {
            r.refused = true;
            r.reason = "the shipped 7-vertex complex has " + std::to_string(seven.num_degrees) +
                       " cohomological degrees (q = 0.." + std::to_string(seven.num_degrees - 1) +
                       "), below q_max = " + std::to_string(q_max) +
                       ": not a free resolution (Bredon object at ell in {2, 3})";
            return r;
        }
        // q_max within the object's own range: return its real dims (no case
        // reaches this today -- (T6) always asks for kQMax = 5 -- but the
        // refusal is then genuinely the object's, not a missing branch's).
        r.dims.assign(seven.dims.begin(), seven.dims.begin() + (q_max + 1));
        return r;
    }
    if (q_max < 4) {
        r.refused = true;
        r.reason = "q_max = " + std::to_string(q_max) + " below the validation's floor 4 ([X] E9: q_max >= 4; >= 5 for both H^2 ~ H^4 and H^3 ~ H^5)";
        return r;
    }
    const IsotropyModel M(q_max);
    r.dims = M.cohomology_lengths(U32Ctx(ell), n);
    return r;
}

}  // namespace

uint32_t HeckeFreeModelValidation() {
    g_checks = 0;
    g_fail = 0;
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    std::cout << "HeckeFreeModelValidation (slot-9 WP-11, line 2 M0; [C] sec:integral, prop:certifiede, "
                 "prop:badprimes, prop:isomodel; oracle O1 = [IB] Appendix A)" << std::endl;
    std::cout << "  (P) O1: integral_line_probe.py group rows, n <= 20, q <= 6, frozen in this file; good prime "
              << kGoodPrime << "; q_max = " << kQMax << " (C_I built to degree " << kQMax + 1 << ")\n";

    const IsotropyModel M(kQMax);
    const auto& table = O1();
    check(table.size() == kNMax + 1, "(P) O1 table has 21 rows");

    // ---- (M), (O9), (O8) the assembly, n = 0 .. 20 -------------------------------
    std::vector<std::vector<AbelianGroupType>> assembled(kNMax + 1);
    std::vector<DicksonCell> dickson;
    std::vector<ModelIntegralCohomology> model_only(kNMax + 1);
    uint32_t refused_levels = 0;   // (n, ell, q, K) with K below the certified level
    for (uint32_t n = 0; n <= kNMax; ++n) {
        // C_I at 2 and 3, K per differential; free ranks at the good prime.
        const ModelIntegralCohomology G = integral_model::AssembleFromModel(M, n, {2, 3}, kGoodPrime);
        std::ostringstream cert;
        for (const auto& pc : G.primes) {
            cert << " ell=" << pc.ell << " K[d^0..d^5]=[";
            for (uint32_t q = 0; q <= kQMax; ++q) cert << (q ? "," : "") << pc.d[q].K;
            cert << "]";
        }
        check(G.all_certified(), "(M) n=" + std::to_string(n) + ": every d^q certified at 2 and 3 within budget");
        // Non-vacuity of the rule: every level below the certified K was REFUSED
        // (r_K < rank_Q there), so a summand Z/ell^a with a >= K was really seen
        // as missing ([C, prop:certifiede](3), last sentence).
        for (const auto& pc : G.primes)
            for (const auto& d : pc.d) {
                bool below_refused = d.certified && d.r_K_tried.size() == d.K - 1;
                for (size_t i = 0; below_refused && i + 1 < d.r_K_tried.size(); ++i)
                    below_refused = d.r_K_tried[i] < d.rank_Q;
                check(below_refused, "(M) n=" + std::to_string(n) + " ell=" + std::to_string(pc.ell) + " d^" +
                                         std::to_string(d.q) + ": every K below the certified one is refused");
                if (d.K > 2) refused_levels += d.K - 2;
            }
        std::vector<AbelianGroupType> H = G.H;

        // ell >= 5: Dickson primes through C_7's K-mode; C_I cross-check; the others torsion-free.
        const auto dp = integral_model::DicksonPrimes(n);
        std::ostringstream dtext;
        for (const auto& [ell, count] : dp) {
            DicksonCell dc = dickson_cell(M, n, ell);
            const bool ok_c = check(dc.certified, "(M) n=" + std::to_string(n) + " ell=" + std::to_string(ell) +
                                                      ": C_7 K-mode certified within budget");
            if (ok_c) {
                // tors H^1 at ell from C_7; H^0 and H^2 of C_eq(Z)_(ell) torsion-free, ranks = rank_Q.
                const AbelianGroupType t1 = dc.ladic[1].primary_part(ell);
                if (t1.torsion.count(ell))
                    for (uint32_t a : t1.torsion.at(ell)) H[1].add_torsion(ell, a);
                check(dc.ladic[0].torsion.empty() && dc.ladic[2].torsion.empty(),
                      "(M) n=" + std::to_string(n) + " ell=" + std::to_string(ell) + ": C_eq(Z)_(ell) torsion only in H^1");
                check(dc.ladic[1].free_rank == G.H[1].free_rank,
                      "(M) n=" + std::to_string(n) + " ell=" + std::to_string(ell) + ": rank H^1(C_7) == rank H^1(C_I)");
                const AbelianGroupType pp = dc.ladic[1].primary_part(ell);
                const uint32_t summands = pp.torsion.count(ell) ? static_cast<uint32_t>(pp.torsion.at(ell).size()) : 0u;
                check(summands == count, "(M) n=" + std::to_string(n) + " ell=" + std::to_string(ell) +
                                             ": number of ell-primary summands of H^1 == [C, prop:badprimes](1) count " +
                                             std::to_string(count));
                // The same prime on C_I ([C, prop:certifiede](5) at ell >= 5).
                const auto pcI = integral_model::CertifyAtPrime(M, n, ell, G.rank_Q);
                AbelianGroupType fromI;
                bool okI = pcI.all_certified();
                for (uint32_t q = 0; q < kQMax; ++q)
                    for (uint32_t a : pcI.d[q].nonunit) {
                        if (q == 0) fromI.add_torsion(ell, a);   // tors H^1 = nonunit(d^0)
                        else okI = false;                        // no ell >= 5 torsion above H^1
                    }
                okI = okI && pcI.d[kQMax].nonunit.empty();
                check(okI && fromI == dc.ladic[1].primary_part(ell),
                      "(M) n=" + std::to_string(n) + " ell=" + std::to_string(ell) +
                          ": C_I's ell-part == C_7's (tors H^1 only)");
                dtext << " " << ell << ":K=" << dc.K << ":" << dc.ladic[1].primary_part(ell).to_string()
                      << (okI ? "(=C_I)" : "(C_I DIFFERS)");
            }
            dickson.push_back(std::move(dc));
        }
        for (uint32_t ell : {5u, 7u, 11u, 13u, 17u, 19u, 23u}) {
            bool is_d = false;
            for (const auto& pr : dp) is_d = is_d || pr.first == ell;
            if (is_d) continue;
            const U32Ctx f(ell);
            bool free_here = true;
            for (uint32_t q = 0; q <= kQMax; ++q) {
                auto Dq = M.coboundary(f, n, q);
                const uint32_t r = (Dq.rows && Dq.cols) ? engine::elim::rref(f, engine::view(Dq)).rank : 0u;
                free_here = free_here && r == G.rank_Q[q];
            }
            check(free_here, "(M) n=" + std::to_string(n) + " ell=" + std::to_string(ell) +
                                 ": not a Dickson prime, and rank_F_ell d^q == rank_Q d^q for q <= 5 (no ell-torsion)");
        }

        // Cross-check the free ranks against the shipped pipeline at the good prime (6 invertible).
        const FieldPipe fp = field_pipeline(n, kGoodPrime);
        check(fp.dims[0] == H[0].free_rank && fp.dims[1] == H[1].free_rank && fp.dims[2] == H[2].free_rank,
              "(M) n=" + std::to_string(n) + ": free ranks == dim H^q(C_7 (x) F_10007) " + vec_str(fp.dims));

        // (M) against O1 exactly.
        bool all = true;
        std::ostringstream groups;
        for (uint32_t q = 0; q <= kQMax; ++q) {
            const bool eq = H[q] == table[n][q];
            all = all && eq;
            check(eq, "(M) n=" + std::to_string(n) + " q=" + std::to_string(q) + ": assembled " + H[q].to_string() +
                          " == O1 " + table[n][q].to_string());
            groups << " | H^" << q << "=" << H[q].to_string();
        }
        // tors H^6 (= non-unit divisors of d^5 at 2, 3) against O1's H^6 torsion (bonus O9).
        AbelianGroupType t6 = table[n][6];
        t6.free_rank = 0;
        check(G.tors_H_top == t6, "(O9) n=" + std::to_string(n) + ": tors H^6 = " + G.tors_H_top.to_string() +
                                      " == O1 " + t6.to_string());
        // (O9) periodicity on the integral groups.
        check(H[2] == H[4] && H[3] == H[5], "(O9) n=" + std::to_string(n) + ": H^2 == H^4 and H^3 == H^5");

        // Held-out level: [C, prop:certifiede](3) predicts H^q(C_I (x) Z/ell^f) for every f.
        std::ostringstream held;
        for (const auto& pc : G.primes) {
            const uint32_t f = pc.K_max() + 1;
            const auto L = M.cohomology_lengths(U32Ctx::prime_power(pc.ell, f), n);
            const auto P = integral_model::PredictedLengths(G, pc.ell, f);
            check(L == P, "(M) n=" + std::to_string(n) + ": UCT prediction at Z/" + std::to_string(pc.ell) + "^" +
                              std::to_string(f) + " " + vec_str(P) + " == computed " + vec_str(L));
            held << " Z/" << pc.ell << "^" << f << ":" << vec_str(L);
        }
        std::cout << "  (M) n=" << n << ((n % 2 == 1 && n >= 9) ? " [probe row]" : "") << ":" << groups.str() << " -- "
                  << (all ? "== O1" : "MISMATCH") << "\n"
                  << "      certified" << cert.str() << "; ell>=5 (C_7 K-mode):" << (dp.empty() ? " none" : dtext.str())
                  << "; held-out" << held.str() << "; tors H^6 " << G.tors_H_top.to_string() << "\n";
        assembled[n] = H;
        model_only[n] = G;
    }

    check(refused_levels > 0, "(M) the certified-K rule refused some level (non-vacuous)");
    std::cout << "  (M) certified-K rule: " << refused_levels
              << " (n, ell, d^q, K) levels refused below their certified K (the rule is non-vacuous)\n";

    // ---- (O2) n = 0 -------------------------------------------------------------
    {
        std::vector<std::string> got;
        for (uint32_t q = 0; q <= kQMax; ++q) got.push_back(assembled[0][q].invariant_factor_string());
        const std::vector<std::string> want = {"Z", "0", "Z/12", "0", "Z/12", "0"};
        check(got == want, "(O2) n = 0: Z, 0, Z/12, 0, Z/12, 0");
        std::cout << "  (O2) n=0: ";
        for (uint32_t q = 0; q <= kQMax; ++q) std::cout << (q ? ", " : "") << got[q];
        std::cout << " (classical H^*(SL_2(Z); Z)) -- " << (got == want ? "ok" : "FAIL") << "\n";
    }

    // ---- (X3) eps^* (x) Z/ell^K at every Dickson cell ----------------------------------
    {
        uint32_t cells = 0;
        for (const auto& dc : dickson) {
            if (!dc.certified) continue;
            const bool zero = std::all_of(dc.cone_lengths.begin(), dc.cone_lengths.end(), [](uint32_t v) { return v == 0; });
            check(dc.eps_chain_map, "(X3) n=" + std::to_string(dc.n) + " ell=" + std::to_string(dc.ell) +
                                        ": eps^* a cochain map on the pipeline's C_eq(Z/ell^K) basis");
            check(dc.cone_d2_zero, "(X3) n=" + std::to_string(dc.n) + " ell=" + std::to_string(dc.ell) + ": D_cone^2 = 0");
            check(zero, "(X3) n=" + std::to_string(dc.n) + " ell=" + std::to_string(dc.ell) +
                            ": Cone(eps^*) (x) Z/ell^K acyclic through degree 5, lengths " + vec_str(dc.cone_lengths));
            std::cout << "  (X3) n=" << dc.n << " ell=" << dc.ell << " K=" << dc.K << ": length H^q(Cone eps^*), q=-1..5, "
                      << vec_str(dc.cone_lengths) << " -> eps^* iso on H^0..H^5 over Z/" << dc.ell << "^" << dc.K << " -- "
                      << ((zero && dc.eps_chain_map && dc.cone_d2_zero) ? "ok" : "FAIL") << "\n";
            ++cells;
        }
        check(cells == 10, "(X3) ten Dickson cells for n <= 20 (Appendix A: Z/5 at 6, 12, 18, 20; Z/7 at 8, 16; 11, 13, 17, 19 once)");
        std::cout << "  (X3) " << cells << " Dickson cells\n";
    }

    // ---- (O8) odd n: three different objects, labelled -----------------------------
    {
        for (uint32_t n = 1; n <= kNMax; n += 2) {
            const FieldPipe good = field_pipeline(n, kGoodPrime);
            const FieldPipe two = field_pipeline(n, 2);
            const bool c7z_zero = good.C_dims == std::vector<uint32_t>({0, 0, 0});
            check(c7z_zero, "(O8) n=" + std::to_string(n) + ": C_{7,Z} = 0 (rank C^q_eq over F_10007 = 0, "
                                                             "[C, prop:ringinforce](1),(3))");
            bool two_only = true, nonzero = false;
            for (uint32_t q = 0; q <= kQMax; ++q) {
                const auto& g = assembled[n][q];
                two_only = two_only && g.free_rank == 0 && (g.torsion.empty() || (g.torsion.size() == 1 && g.torsion.count(2)));
                for (const auto& kv : g.torsion)
                    for (uint32_t a : kv.second) two_only = two_only && a == 1;   // 2 H^q = 0 ([C, prop:badprimes](2))
                nonzero = nonzero || !g.is_zero();
            }
            check(two_only && nonzero, "(O8) n=" + std::to_string(n) +
                                           ": the group object (from C_I) is non-zero and killed by 2");
            const bool bredon_nonzero = two.dims != std::vector<uint32_t>({0, 0, 0});
            check(bredon_nonzero, "(O8) n=" + std::to_string(n) + ": C_7 (x) F_2 (the Bredon object) has non-zero H^*");
            std::cout << "  (O8) n=" << n << ": C_{7,Z} = 0 [rank C^q_eq(F_10007) " << vec_str(good.C_dims)
                      << "]; group object from C_I: H^1..3 = " << assembled[n][1].to_string() << " | "
                      << assembled[n][2].to_string() << " | " << assembled[n][3].to_string()
                      << "; Bredon object C_7 (x) F_2: dim H^0..2 " << vec_str(two.dims) << " -- "
                      << ((c7z_zero && two_only && nonzero && bredon_nonzero) ? "ok" : "FAIL") << "\n";
        }
    }

    // ---- (T1), (T2), (T6) the [X] tests through the harness ----------------------
    {
        for (uint32_t ell : {2u, 3u}) {
            const HarnessRun r = harness(ComplexKind::FreeModelTruncated, kQMax, 0, ell);
            check(!r.refused && r.dims == std::vector<uint32_t>(kQMax + 1, 1),
                  "(T1) n=0: dim H^q(Gamma; F_" + std::to_string(ell) + ") = 1, q = 0..5");
            std::cout << "  (T1) n=0 ell=" << ell << ": dim H^0..5 " << vec_str(r.dims) << " -- "
                      << (r.dims == std::vector<uint32_t>(kQMax + 1, 1) ? "ok" : "FAIL") << "\n";
        }
        uint32_t t2 = 0;
        for (uint32_t ell : {2u, 3u})
            for (uint32_t n = 0; n <= kNMax; ++n) {
                const HarnessRun r = harness(ComplexKind::FreeModelTruncated, kQMax, n, ell);
                bool ok = !r.refused;
                for (uint32_t q = 2; ok && q + 2 <= kQMax; ++q) ok = r.dims[q] == r.dims[q + 2];
                check(ok, "(T2) ell=" + std::to_string(ell) + " n=" + std::to_string(n) + ": dim H^q = dim H^{q+2}, q = 2, 3 " +
                              vec_str(r.dims));
                ++t2;
            }
        std::cout << "  (T2) dim H^q(C_I (x) F_ell) = dim H^{q+2} for q = 2, 3 at ell in {2, 3}, n <= 20: " << t2
                  << " cells\n";
        // (T6) the negative control.
        const HarnessRun seven = harness(ComplexKind::SevenVertexModel, kQMax, 0, 2);
        check(seven.refused && seven.dims.empty(), "(T6) the harness refuses the shipped 7-vertex model");
        const HarnessRun low = harness(ComplexKind::FreeModelTruncated, 3, 0, 2);
        check(low.refused && low.dims.empty(), "(T6) the harness refuses q_max = 3");
        const FieldPipe b = field_pipeline(0, 2);
        std::cout << "  (T6) 7-vertex model: refused (" << seven.reason << "); its Bredon dim H^0..2 at n=0 ell=2 is "
                  << vec_str(b.dims) << ", which (T1) would have mis-read; q_max = 3: refused (" << low.reason << ")\n";
    }

    // ---- (R) M0 items 3 and 7 -------------------------------------------------------
    // (hecke-extensions slot 10, WP-8, m-7): this used to re-run
    // HeckeBredonTreeRingValidation() (O7) and HeckeIntegralCharpolyValidation()
    // (O5) IN PROCESS, even though both are already independently registered
    // names that --all runs on their own -- so they ran twice, and the second
    // honours HECKE_INTEGRAL_CAMPAIGN=1 (~30 s extra), silently growing this
    // standard-tier name from ~4 s to ~35 s under that switch with no arm name
    // saying so.  M0 items 3 and 7 are exactly those two names' own --list
    // lines (the catalogue), cited here instead of re-run.
    std::cout << "  (R) M0 item 3 (O7) and item 7 (O5): see the independently registered "
                 "HeckeBredonTreeRingValidation and HeckeIntegralCharpolyValidation --list lines\n";

    const double secs = std::chrono::duration<double>(clk::now() - t0).count();
    std::cout << "HeckeFreeModelValidation: " << g_checks << " checks, " << g_fail << " failing (" << secs
              << " s, indicative)\n";
    return g_fail;
}
