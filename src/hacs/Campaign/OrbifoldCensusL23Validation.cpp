// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 22, L23-3) OrbifoldCensusL23Validation / OrbifoldCensusL23SlowValidation -- see
// OrbifoldCensusL23Validation.h.  Report docs/implementation/slot22/L23_3_report.md.
//
// Standard parts:
//   (A) the sum-rule oracle (l23::sum_rule_oracle, C++) against the 200 frozen rows of Bin3SumRules.inc (fbin3/sumrules.py,
//       exact rationals): v det(T_ell), v det(T_ell^2 - T_ell) = v det T_ell + v det(T_ell - 1), v det(T_p - (1+p)),
//       v det(T_p - (1+p^{n+1})), v(p^n - 1), p in {2, 3, 5, 7} \ {ell}.
//   (B) l23::h1_layers at ell = 2, n = 1 .. 24 and ell = 3, even n = 2 .. 36 against the frozen layers_proto.py rows
//       (h0, h1, dim H^1_Br, dim H^1_!, dim(H^1_! cap H^1_Br), dim L) and the MV rows (h0, h1, bredon_h1 = dim H^1_Br) of
//       L233Oracle.inc; the closed forms dim T^1/ell = h0 ([C] lem:bocksteinallell), codim H^1_Br = h0 resp.
//       [n = 0, 4 mod 6] ([R] Thm B(5), C(1)), dim L = [n = 0, 4 mod 6], L cap H^1_Br = 0, L in T^1/3 iff n = 4 mod 6
//       ([R] Thm C(3), (4)); T_ell preserves H^1_Br, H^1_! ([R] Thm E, Q10).
//   (C) orbifold_cell/v1 cells (ComputeOrbifoldCell, no I/O) at ell = 2, 3, n = 0 .. 24: status ok; the (S) sum-rule gates on
//       every cell with a lattice, the (B/!/L) gates where specified; the MV rows (h0..h3, rank Lambda^1, t1, t2, t3, bredon h1);
//       the schema (sumrules, dim_h1_br, dim_h1_cusp, t_ell_triple_B, L at ell = 3; the oracle's K outside the hashed body;
//       boundary_split replaced by the value) and the defaults (fingerprint 0, unchanged ids).
//   (D) negative controls: the oracle without the p^{k-1} a_{m/p} term; a reader with r_inf dropped (mutate 1) against the
//       sum rules; h1_layers with r_inf dropped (T_ell on H^1 differs) and with a wrong c(T) (mutate 3: dim H^1_! moves);
//       a tampered frozen row; a non-convex Jordan rank sequence (3, 2, 0) (L23_3_audit.md M1).
// Slow parts: (E) reader + oracle on every even n <= 200 at both primes; (F) the C_I (lat) reading at sampled n;
//   (G) layers vs MV at n = 50 .. 200; (H) the oracle at the four L23-4 calibration cells (n = 240 ... 700).
#include "OrbifoldCensusL23Validation.h"

#include "CampaignOrbifold.h"
#include "CampaignSpec.h"
#include "../Integral/Model/IntegralHecke.h"
#include "../Integral/Model/IsotropyModel.h"
#include "../Integral/Model/StrictLift.h"
#include "../Orbifold/L23/AmalgamReader.h"
#include "../Orbifold/L23/SumRuleOracle.h"
#include "../../ModArithmetic/DenseElim.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct SumRuleFrozen { uint32_t ell, n; int vTl, vTl2, hodge; int vP[4], vE[4], vN[4]; };
const SumRuleFrozen kSumRules[] = {
#include "../Orbifold/L23/Bin3SumRules.inc"
};
struct LayerFrozen { uint32_t ell, n, h0, h1, br, cusp, cusp_br; int L; };
const LayerFrozen kLayers[] = {
#define L233_LAYER_ROWS
#include "../Orbifold/L23/L233Oracle.inc"
#undef L233_LAYER_ROWS
};
struct MvFrozen { uint32_t ell, n, h0, h1, h2, h3, bredon, defect, rank, t1, t2, t3; };
const MvFrozen kMv[] = {
#define L233_MV_ROWS
#include "../Orbifold/L23/L233Oracle.inc"
#undef L233_MV_ROWS
};
struct CalTp { uint32_t p, tp, eis; };
struct CalV { uint32_t tp, eis; };
struct CalFrozen { uint32_t ell, n, sum_tell; CalTp s[3]; uint32_t vtell; CalV v[3]; };
const CalFrozen kCal[] = {
#define L233_CAL_ROWS
#include "../Orbifold/L23/L233Oracle.inc"
#undef L233_CAL_ROWS
};

uint32_t g_checks = 0, g_fail = 0;
bool check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        std::cout << "  [FAIL] " << what << std::endl;
    }
    return ok;
}
std::string str(uint64_t x) { return std::to_string(x); }
const std::vector<uint32_t> kPrimes = {2, 3, 5, 7};
int pidx(uint32_t p) { return p == 2 ? 0 : p == 3 ? 1 : p == 5 ? 2 : 3; }
double since(std::chrono::steady_clock::time_point t) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count(); }

const MvFrozen* MvRow(uint32_t ell, uint32_t n) {
    for (const MvFrozen& r : kMv) if (r.ell == ell && r.n == n) return &r;
    return nullptr;
}

// The frozen row against the oracle; returns the number of mismatching fields.
uint32_t CompareSumRow(const SumRuleFrozen& r, const l23::SumRuleRow& S) {
    uint32_t bad = 0;
    if (!S.resolved) return 1;
    if (static_cast<int>(S.vdet_tell) != r.vTl) ++bad;
    if (r.vTl2 >= 0 && static_cast<int>(S.vdet_tell + S.vdet_tell_minus_1) != r.vTl2) ++bad;
    for (uint32_t p : kPrimes) {
        if (p == r.ell) continue;
        const int i = pidx(p);
        if (static_cast<int>(S.vdet_tp.at(p)) != r.vP[i] || static_cast<int>(S.vdet_eis.at(p)) != r.vE[i] ||
            static_cast<int>(S.v_pn1.at(p)) != r.vN[i])
            ++bad;
    }
    return bad;
}

uint64_t FinSum(const l23::Divisors& D) {
    uint64_t s = 0;
    for (uint32_t x : D.finite) s += x;
    return s;
}
// The sum rules on a reader cell (the gates of the record, recomputed here): returns the number violated, -1 if not checkable.
int SumRulesOnReader(const l23::LatticeCell& L, const l23::SumRuleRow& S) {
    if (!L.ok || !S.resolved) return -1;
    int bad = 0;
    bad += (L.ed_tell.at_least == 0 && FinSum(L.ed_tell) == 2ull * S.vdet_tell) ? 0 : 1;
    for (const auto& [p, D] : L.ed_tp) bad += (D.at_least == 0 && FinSum(D) == 2ull * S.vdet_tp.at(p) + S.v_pn1.at(p)) ? 0 : 1;
    for (const auto& [p, D] : L.ed_eis) bad += (D.at_least == 1 && FinSum(D) == 2ull * S.vdet_eis.at(p)) ? 0 : 1;
    if (L.ell == 3 && L.pm_done) {
        bad += (FinSum(L.ed_tell_plus) == S.vdet_tell && FinSum(L.ed_tell_minus) == S.vdet_tell) ? 0 : 1;
        for (const auto& [p, D] : L.ed_tp_plus) bad += (FinSum(D) == S.vdet_tp.at(p) + S.v_pn1.at(p)) ? 0 : 1;
        for (const auto& [p, D] : L.ed_tp_minus) bad += (FinSum(D) == S.vdet_tp.at(p)) ? 0 : 1;
        for (const auto& [p, D] : L.ed_eis_plus) bad += (D.at_least == 1 && FinSum(D) == S.vdet_eis.at(p)) ? 0 : 1;
        for (const auto& [p, D] : L.ed_eis_minus) bad += (FinSum(D) == S.vdet_eis.at(p)) ? 0 : 1;
    }
    return bad;
}

l23::ReaderOptions ReaderOpts(uint32_t ell, uint32_t n, uint32_t mutate = 0) {
    l23::ReaderOptions o;
    o.ell = ell;
    o.m_target = campaign::OrbifoldAutoMTarget(ell, n);
    o.cap = o.m_target;
    o.primes = kPrimes;
    o.pm = (ell == 3);
    o.mutate = mutate;
    return o;
}

campaign::CampaignSpec Spec(const std::string& text) {
    std::istringstream is(text);
    campaign::CampaignSpec s = campaign::ParseCampaignSpec(is, "<l233-spec>");
    s.output_dir = "unused";
    return s;
}

// ---- the C_I (lat) reading (IntegralHecke.h): ell^{e_1} IM(red_{K' -> K}) ~= Lambda^1 (x) Z/ell^{K - e_1} ----------------
struct LatRead {
    bool ok = false;
    std::string why;
    uint32_t m = 0, rank = 0;
    std::map<uint32_t, std::vector<uint32_t>> ed_tp, ed_eis;   // Smith valuations, >= m clipped to m, ascending
    std::vector<uint32_t> ed_tell;
};
struct Bundles {
    std::map<uint32_t, std::unique_ptr<integral_model::StrictLiftBundle>> b;
    const integral_model::StrictLiftBundle& get(uint32_t p) {
        auto it = b.find(p);
        if (it == b.end()) it = b.emplace(p, std::make_unique<integral_model::StrictLiftBundle>(p, 3)).first;
        return *it->second;
    }
};
LatRead LatReading(uint32_t ell, uint32_t n, uint32_t e1, Bundles& BB) {
    using U32 = uint32_t;
    LatRead out;
    const uint32_t Kp = (ell == 2) ? 31 : 19, e2 = (ell == 2) ? 3 : 1, K = Kp - e2;   // e2 >= exp T^2 ([C] cl:amalgam(4))
    if (K <= e1 + 1) { out.why = "K - e1 too small"; return out; }
    const integral_model::IsotropyModel& M = BB.get(2).model();
    const engine::ModCtx<U32> big = integral_model::RingContext(ell, Kp), small = integral_model::RingContext(ell, K);
    const integral_model::ModelCohomology<U32> MC(M, big, n);
    engine::DenseBuf<U32> G = integral_model::ReducedCocycles(MC, 1, small);
    U32 s = 1;
    for (uint32_t i = 0; i < e1; ++i) s *= ell;
    for (uint32_t i = 0; i < G.rows; ++i)
        for (uint32_t j = 0; j < G.cols; ++j) G.at(i, j) = static_cast<U32>((static_cast<uint64_t>(G.at(i, j)) * s) % small.p);
    const engine::DenseBuf<U32> B1 = M.coboundary(small, n, 0);
    const integral_model::InducedOperator<U32> N = integral_model::Subquotient(small, G, B1);
    out.m = K - e1;
    out.rank = N.rank();
    for (uint32_t e : N.exponents)
        if (e != out.m) { out.why = "a summand of exponent " + str(e) + " != m = " + str(out.m) + " (not free)"; return out; }
    const engine::ModCtx<U32> Rm = integral_model::RingContext(ell, out.m);
    auto sv = [&](const engine::DenseBuf<U32>& X) {
        std::vector<uint32_t> sig;
        if (X.rows) for (uint32_t f : engine::elim::invariant_factors(Rm, engine::cview(X))) sig.push_back(out.m - f);
        while (sig.size() < out.rank) sig.push_back(out.m);
        std::sort(sig.begin(), sig.end());
        return sig;
    };
    uint64_t Mm = 1;
    for (uint32_t i = 0; i < out.m; ++i) Mm *= ell;
    for (uint32_t p : kPrimes) {
        const engine::DenseBuf<U32> Th = BB.get(p).lift().theta_hat(small, n, 1);
        engine::DenseBuf<U32> A = integral_model::OperatorMatrix(small, N, B1, Th);
        for (uint32_t i = 0; i < A.rows; ++i)
            for (uint32_t j = 0; j < A.cols; ++j) A.at(i, j) = static_cast<U32>(A.at(i, j) % Mm);
        if (p == ell) { out.ed_tell = sv(A); continue; }
        uint64_t pn1 = 1;
        for (uint32_t i = 0; i <= n; ++i) pn1 = pn1 * p % Mm;
        out.ed_tp[p] = sv(integral_model::AddScalar(Rm, A, static_cast<U32>((Mm - (1 + p) % Mm) % Mm)));
        out.ed_eis[p] = sv(integral_model::AddScalar(Rm, A, static_cast<U32>((Mm - (1 + pn1) % Mm) % Mm)));
    }
    out.ok = true;
    return out;
}
// The reader's divisors clipped at m (finite x -> min(x, m), at_least -> m), ascending.
std::vector<uint32_t> Clip(const l23::Divisors& D, uint32_t m) {
    std::vector<uint32_t> v;
    for (uint32_t x : D.finite) v.push_back(std::min(x, m));
    for (uint32_t i = 0; i < D.at_least; ++i) v.push_back(m);
    std::sort(v.begin(), v.end());
    return v;
}

} // namespace

// ===========================================================================================================================
uint32_t OrbifoldCensusL23Validation() {
    g_checks = g_fail = 0;
    const auto t0 = std::chrono::steady_clock::now();
    std::cout << "OrbifoldCensusL23Validation: sum-rule oracle, amalgam layers, census cells n <= 24 (slot-22 L23-3)" << std::endl;

    // ---------------- (A) the oracle against the 200 frozen rows ----------------------------------------------------------
    uint32_t a_rows = 0, a_ok = 0, a_maxK = 0;
    double a_s = 0, a_max_s = 0;
    {
        const auto ta = std::chrono::steady_clock::now();
        for (const SumRuleFrozen& r : kSumRules) {
            const l23::SumRuleRow S = l23::sum_rule_oracle(r.ell, r.n, kPrimes);
            ++a_rows;
            a_max_s = std::max(a_max_s, S.seconds);
            a_maxK = std::max(a_maxK, S.K);
            const uint32_t bad = CompareSumRow(r, S);
            if (check(bad == 0, "(A) oracle vs Bin3SumRules.inc at (ell=" + str(r.ell) + ", n=" + str(r.n) + "): " + str(bad) + " field(s) differ" +
                                    (S.log.empty() ? std::string() : " -- " + S.log.back())))
                ++a_ok;
        }
        a_s = since(ta);
        std::cout << "  (A) SUMRULE_FROZEN rows=" << a_rows << " agree=" << a_ok << " max_K=" << a_maxK << " s=" << a_s << " max_cell_s=" << a_max_s
                  << std::endl;
    }

    // ---------------- (B) the layers against the prototype and the MV oracle --------------------------------------------
    uint32_t b_rows = 0, b_ok = 0;
    {
        const auto tb = std::chrono::steady_clock::now();
        for (const LayerFrozen& r : kLayers) {
            if (r.n == 0) continue;   // the campaign starts the amalgam layers at n = 1
            const l23::LayerCell lc = l23::h1_layers(r.ell, r.n, 8, r.ell == 2 ? 4 : 2);
            ++b_rows;
            const std::string at = " (ell=" + str(r.ell) + " n=" + str(r.n) + ")";
            bool ok = check(lc.h0 == r.h0 && lc.h1 == r.h1 && lc.dim_br == r.br && lc.dim_cusp == r.cusp && lc.dim_cusp_br == r.cusp_br,
                            "(B) h0 / h1 / dim H^1_Br / dim H^1_! / cap vs layers_proto.py" + at + ": " + str(lc.h0) + "/" + str(lc.h1) + "/" +
                                str(lc.dim_br) + "/" + str(lc.dim_cusp) + "/" + str(lc.dim_cusp_br) + " vs " + str(r.h0) + "/" + str(r.h1) + "/" +
                                str(r.br) + "/" + str(r.cusp) + "/" + str(r.cusp_br));
            const MvFrozen* mv = MvRow(r.ell, r.n);
            ok = check(mv && mv->h0 == lc.h0 && mv->h1 == lc.h1 && mv->bredon == lc.dim_br, "(B) h0 / h1 / bredon_h1 vs mv_numpy.py" + at) && ok;
            ok = check(lc.t1_cocycles && lc.dim_t1 == lc.h0, "(B) dim T^1/ell = h0 ([C] lem:bocksteinallell)" + at + ": " + str(lc.dim_t1)) && ok;
            const uint32_t w = (r.n % 6 == 0 || r.n % 6 == 4) ? 1 : 0;
            ok = check(lc.h1 - lc.dim_br == (r.ell == 2 ? lc.h0 : w), "(B) codim H^1_Br ([R] Thm B(5) / C(1))" + at) && ok;
            ok = check(lc.br_stable, "(B) T_ell preserves H^1_Br ([R] Thm E)" + at) && ok;
            // (L23_3_audit.md m1) vacuous while dim H^1_! = 0 (every row here): kept as a check of the code path only
            ok = check(lc.cusp_stable, "(B) T_ell preserves H^1_! ([R] Q10; vacuous, dim H^1_! = " + str(lc.dim_cusp) + ")" + at) && ok;
            ok = check(lc.j_whole.valid && lc.j_whole.dim == lc.h1 && lc.j_whole.eig1 == std::vector<uint32_t>{1} && lc.j_whole.other == 0 &&
                           std::all_of(lc.j_whole.eig0.begin(), lc.j_whole.eig0.end(), [](uint32_t b) { return b == 1; }),
                       "(B) T_ell a rank-one idempotent on H^1 (CONJ, OBSERVED n <= 24; here a check of the layer code)" + at) && ok;
            ok = check(lc.j_br.dim + lc.j_quot.dim == lc.h1 && lc.j_br.dim == lc.dim_br, "(B) the (B) triple's dimensions add up" + at) && ok;
            if (r.ell == 3) {
                ok = check(lc.l_done && static_cast<int>(lc.dim_L) == r.L && lc.dim_L == w && lc.dim_L_br == 0 && lc.dim_L + lc.dim_br == lc.h1,
                           "(B) dim L = [n = 0, 4 mod 6], L cap H^1_Br = 0, H^1 = H^1_Br + L ([R] Thm C(3))" + at) && ok;
                // [R] Thm C(4) restated (L23_3_report.md): T^1/3 leaves H^1_Br iff n = 4 mod 6; L itself is never inside T^1/3 here
                ok = check(lc.dim_t1_br + ((r.n % 6 == 4 && lc.dim_t1) ? 1u : 0u) == lc.dim_t1 && !lc.L_in_T1,
                           "(B) dim(T^1/3 cap H^1_Br) = dim T^1/3 - [n = 4 mod 6] (C(4) restated, OBSERVED); L not in T^1/3 (OBSERVED)" + at + ": " + str(lc.dim_t1_br) + " / " + str(lc.dim_t1)) && ok;
            }
            if (ok) ++b_ok;
        }
        std::cout << "  (B) LAYERS rows=" << b_rows << " agree=" << b_ok << " s=" << since(tb) << std::endl;
    }

    // ---------------- (C) census cells n = 0 .. 24 ------------------------------------------------------------------------
    uint32_t c_cells = 0, c_ok = 0, c_gates = 0, c_sum_gates = 0, c_layer_gates = 0, c_mv = 0, c_findings = 0;
    {
        const auto tc = std::chrono::steady_clock::now();
        const campaign::CampaignSpec spec = Spec("orbifold = c_i\nn_sym = 0..24\nell = 2,3\n");
        check(campaign::OrbifoldFingerprint(spec) == 0 && campaign::OrbifoldCellId(spec, 2, 10) == "orb_ell2_n10" &&
                  campaign::OrbifoldCellId(spec, 3, 24) == "orb_ell3_n24",
              "(C) defaults: fingerprint 0, ids orb_ell2_n10 / orb_ell3_n24 unchanged (L23-3 adds no key)");
        for (uint32_t ell : {2u, 3u})
            for (uint32_t n = 0; n <= 24; ++n) {
                const campaign::OrbifoldCell c = campaign::ComputeOrbifoldCell(spec, ell, n);
                ++c_cells;
                const std::string at = " (ell=" + str(ell) + " n=" + str(n) + ")";
                bool ok = check(c.status == "ok" && c.gate_failures() == 0,
                                "(C) status ok" + at + ": " + c.status + (c.failures.empty() ? std::string() : " -- " + c.failures.front()));
                c_findings += c.conj_findings();
                uint32_t sg = 0, lg = 0;
                for (const campaign::OrbifoldGate& g : c.gates) {
                    ++c_gates;
                    if (g.name.rfind("sumrule_", 0) == 0) ++sg;
                    if (g.name.rfind("layers_", 0) == 0 || g.name.rfind("br_", 0) == 0 || g.name.rfind("cusp_", 0) == 0 || g.name.rfind("L_", 0) == 0 ||
                        g.name == "t1_mod_3_leaves_br_iff_n4mod6")
                        ++lg;
                }
                c_sum_gates += sg;
                c_layer_gates += lg;
                const bool lattice = c.reader_ok && c.rank_lambda > 0;
                // T_ell + 3 x (tp, eis) + resolved = 8; ell = 3 adds the halves: 2 + 3 x 4 = 14
                const uint32_t want_sg = lattice ? (ell == 2 ? 8u : 22u) : 0u;
                ok = check(sg == want_sg && c.sumrules_ran == lattice, "(C) the (S) sum-rule gates: " + str(sg) + ", expected " + str(want_sg) + at) && ok;
                const bool layered = n > 0 && (ell == 2 || n % 2 == 0);
                // (L23_3_audit.md n4) exact: layers_{h1, dim_h1V_mod_ell, t_ell_whole, jordan_valid, bockstein_cocycles}, br_hecke_stable,
                // cusp_hecke_stable, + br_codim_eq_h0 (ell = 2) | br_codim, br_eq_ker_e_cup, L_dim, L_direct_sum, t1_mod_3_... (ell = 3),
                // + layers_dim_t1 when the reader certified
                const uint32_t want_lg = layered ? (ell == 2 ? 8u : 12u) + (c.reader_ok ? 1u : 0u) : 0u;
                ok = check(c.layers_ran == layered && lg == want_lg, "(C) the (B/!/L) gates where specified: " + str(lg) + ", expected " + str(want_lg) + at) && ok;
                if (const MvFrozen* mv = (ell == 3 && n % 2) ? nullptr : MvRow(ell, n)) {
                    ++c_mv;
                    ok = check(c.h[0] == mv->h0 && c.h[1] == mv->h1 && c.h[2] == mv->h2 && c.h[3] == mv->h3 && c.rank_lambda == mv->rank &&
                                   (n == 0 || c.t1 == mv->t1) && c.t2 == mv->t2 && c.t3 == mv->t3 && (!layered || c.am_dim_br == mv->bredon),
                               "(C) h0..h3, rank Lambda^1, t1, t2, t3, dim H^1_Br vs mv_numpy.py" + at) && ok;
                }
                // schema
                const std::string body = campaign::OrbifoldCellBody(spec, c), file = campaign::OrbifoldCellJson(spec, c);
                bool sch = body.find("\"sumrules\":{\"ran\":") != std::string::npos && body.find("\"dim_h1_br\":{") != std::string::npos &&
                           body.find("\"dim_h1_cusp\":{") != std::string::npos && body.find("sumrule_path") == std::string::npos &&
                           file.find("\"sumrule_path\":{\"K\":") != std::string::npos && file.find("\"layers\":") != std::string::npos;
                if (layered) {
                    sch = sch && body.find("\"t_ell_triple_B\":{\"q\":1") != std::string::npos && body.find("\"kind\":\"computed\",\"value\":" + str(c.dim_h1_cusp)) != std::string::npos &&
                          body.find("\"boundary_split\":{\"kind\":\"replaced\",\"value\":\"dim_h1_cusp = " + str(c.dim_h1_cusp) + "\"}") != std::string::npos;
                    if (ell == 3) sch = sch && body.find("\"L\":{\"dim\":") != std::string::npos;
                }
                ok = check(sch, "(C) schema: sumrules / dim_h1_br / dim_h1_cusp / t_ell_triple_B / L in the body, the oracle's K outside it" + at) && ok;
                if (ok) ++c_ok;
            }
        std::cout << "  (C) CELLS cells=" << c_cells << " ok=" << c_ok << " gates=" << c_gates << " sumrule_gates=" << c_sum_gates
                  << " layer_gates=" << c_layer_gates << " mv_rows=" << c_mv << " conj_findings=" << c_findings << " s=" << since(tc) << std::endl;
    }

    // ---------------- (D) negative controls -------------------------------------------------------------------------------
    uint32_t caught = 0;
    {
        // the oracle without the p^{k-1} a_{m/p} term
        const SumRuleFrozen* r22 = nullptr;
        for (const SumRuleFrozen& r : kSumRules) if (r.ell == 2 && r.n == 22) r22 = &r;
        const auto td = std::chrono::steady_clock::now();
        auto prog = [&](const char* what) { std::cout << "  (D) " << what << " caught=" << caught << " t=" << since(td) << " s" << std::endl; };
        // the p^{k-1} a_{m/p} term is 2-adically invisible at d = 2; at (2, 64) and (3, 64) (d = 5, p = 3 / 2 enter) it is not
        {
            bool all = true;
            for (uint32_t ell : {2u, 3u}) {
                const SumRuleFrozen* r64 = nullptr;
                for (const SumRuleFrozen& r : kSumRules) if (r.ell == ell && r.n == 64) r64 = &r;
                all = all && r64 && CompareSumRow(*r64, l23::sum_rule_oracle(ell, 64, kPrimes, 0, 4096, 1)) > 0;
            }
            if (all) ++caught;
        }
        prog("oracle without p^{k-1} a_{m/p}");
        // a tampered frozen row
        SumRuleFrozen t = *r22;
        t.vE[1] += 1;
        if (CompareSumRow(t, l23::sum_rule_oracle(2, 22, kPrimes)) > 0) ++caught;
        prog("tampered frozen row");
        // a reader with r_inf dropped, at both primes, against the sum rules
        for (uint32_t ell : {2u, 3u}) {
            const l23::LatticeCell L = l23::read_lattice(22, ReaderOpts(ell, 22, 1));
            const int bad = SumRulesOnReader(L, l23::sum_rule_oracle(ell, 22, kPrimes));
            if (bad != 0) ++caught;   // (-1: the mutated reader failed its own checks -- also caught)
            prog(ell == 2 ? "reader r_inf dropped, ell = 2" : "reader r_inf dropped, ell = 3");
        }
        // h1_layers with r_inf dropped: T_ell on H^1 is no longer the rank-one idempotent of (B) (n = 10, 12, 16 at ell = 2)
        {
            bool moved = false;
            for (uint32_t n : {10u, 12u, 16u}) {
                const l23::LayerCell a = l23::h1_layers(2, n, 8, 4), b = l23::h1_layers(2, n, 8, 4, 1);
                moved = moved || !b.j_whole.valid || a.j_whole.rank != b.j_whole.rank || a.j_whole.eig1 != b.j_whole.eig1 ||
                        a.j_whole.eig0 != b.j_whole.eig0 || !b.br_stable;
            }
            if (moved) ++caught;
            prog("h1_layers r_inf dropped");
        }
        // a wrong c(T): dim H^1_! or its stability moves somewhere in n <= 12 at both primes
        {
            bool moved = false;
            for (uint32_t ell : {2u, 3u})
                for (uint32_t n = 2; n <= 12 && !moved; n += 2) {
                    const l23::LayerCell a = l23::h1_layers(ell, n, 8, ell == 2 ? 4 : 2), b = l23::h1_layers(ell, n, 8, ell == 2 ? 4 : 2, 3);
                    moved = a.dim_cusp != b.dim_cusp || a.dim_cusp_br != b.dim_cusp_br || !b.cusp_stable;
                }
            if (moved) ++caught;
        }
        // (L23_3_audit.md M1) a strictly decreasing but non-convex rank sequence (T_ell not preserving the subquotient, e.g.
        // U = span{a2, b1, c2} in blocks a1->a2->a3, b1->b2, c1->c2 gives r = (3, 2, 0)) must read invalid, not 2^32 blocks;
        // a genuine sequence (3, 2, 1, 0) still reads one block of size 3
        {
            std::vector<uint32_t> blocks;
            const bool bad_rejected = !l23::jordan_blocks_from_ranks({3, 2, 0}, blocks) && blocks.empty();
            const bool good_read = l23::jordan_blocks_from_ranks({3, 2, 1, 0}, blocks) && blocks == std::vector<uint32_t>{3};
            const bool grow_rejected = !l23::jordan_blocks_from_ranks({3, 4}, blocks);
            if (bad_rejected && good_read && grow_rejected) ++caught;
            prog("jordan ranks (3, 2, 0) non-convex");
        }
        check(caught == 7, "(D) negative controls caught " + str(caught) + " / 7");
        std::cout << "  (D) NEGATIVE_CONTROL caught=" << caught << "/7" << std::endl;
    }

    const double s = since(t0);
    std::cout << "ORBIFOLD_CENSUS_L23 sumrule_rows=" << a_ok << "/" << a_rows << " layer_rows=" << b_ok << "/" << b_rows << " cells=" << c_ok << "/"
              << c_cells << " sumrule_gates=" << c_sum_gates << " layer_gates=" << c_layer_gates << " negative=" << caught << "/7 checks=" << g_checks
              << " failing=" << g_fail << " s=" << s << " verdict=" << (g_fail == 0 ? "OK" : "FAIL") << std::endl;
    return g_fail;
}

// ===========================================================================================================================
uint32_t OrbifoldCensusL23SlowValidation() {
    g_checks = g_fail = 0;
    const auto t0 = std::chrono::steady_clock::now();
    std::cout << "OrbifoldCensusL23SlowValidation: reader vs sum rules n <= 200, C_I (lat), layers vs MV, n = 700 (slot-22 L23-3)" << std::endl;

    // ---------------- (E) reader + oracle on every even n <= 200 -----------------------------------------------------------
    uint32_t e_cells = 0, e_ok = 0;
    std::map<std::pair<uint32_t, uint32_t>, l23::LatticeCell> reader;   // kept for (F)
    {
        const auto te = std::chrono::steady_clock::now();
        for (uint32_t ell : {2u, 3u})
            for (uint32_t n = 2; n <= 200; n += 2) {
                const l23::LatticeCell L = l23::read_lattice(n, ReaderOpts(ell, n));
                const l23::SumRuleRow S = l23::sum_rule_oracle(ell, n, kPrimes);
                ++e_cells;
                const int bad = SumRulesOnReader(L, S);
                if (check(bad == 0, "(E) sum rules on the reader at (ell=" + str(ell) + ", n=" + str(n) + "): " + str(bad) + " violated")) ++e_ok;
                if (n <= 64) reader.emplace(std::make_pair(ell, n), L);
            }
        std::cout << "  (E) READER_SUMRULES cells=" << e_cells << " ok=" << e_ok << " s=" << since(te) << std::endl;
    }

    // ---------------- (F) the C_I (lat) reading at sampled n ----------------------------------------------------------------
    uint32_t f_cells = 0, f_ok = 0;
    {
        const auto tf = std::chrono::steady_clock::now();
        Bundles BB;
        for (uint32_t ell : {2u, 3u})
            for (uint32_t n : {10u, 16u, 22u, 34u, 46u, 64u}) {
                const l23::LatticeCell& L = reader.at({ell, n});
                uint32_t e1 = 0;
                for (uint32_t a : L.torsion) e1 = std::max(e1, a);
                const LatRead R = LatReading(ell, n, e1, BB);
                ++f_cells;
                const std::string at = " (ell=" + str(ell) + " n=" + str(n) + ")";
                bool ok = check(R.ok && R.rank == L.rank_lambda, "(F) the (lat) reading is free of rank rank Lambda^1" + at + ": " +
                                                                       (R.ok ? str(R.rank) + " vs " + str(L.rank_lambda) : R.why));
                if (R.ok) {
                    ok = check(R.ed_tell == Clip(L.ed_tell, R.m), "(F) ed_tell: C_I (lat) = reader, clipped at m = " + str(R.m) + at) && ok;
                    for (uint32_t p : kPrimes) {
                        if (p == ell) continue;
                        ok = check(R.ed_tp.at(p) == Clip(L.ed_tp.at(p), R.m) && R.ed_eis.at(p) == Clip(L.ed_eis.at(p), R.m),
                                   "(F) ed_tp / ed_eis at p = " + str(p) + ": C_I (lat) = reader, clipped at m = " + str(R.m) + at) && ok;
                    }
                }
                if (ok) ++f_ok;
                std::cout << "  (F) LAT ell=" << ell << " n=" << n << " m=" << R.m << " rank=" << R.rank << (ok ? " agree" : " DIFFER") << std::endl;
            }
        std::cout << "  (F) CI_LAT cells=" << f_cells << " agree=" << f_ok << " s=" << since(tf) << std::endl;
    }

    // ---------------- (G) layers vs MV at larger n ----------------------------------------------------------------------------
    uint32_t g_cells = 0, g_ok = 0;
    {
        const auto tg = std::chrono::steady_clock::now();
        for (uint32_t ell : {2u, 3u})
            for (uint32_t n : {50u, 100u, 150u, 200u}) {
                const l23::LayerCell lc = l23::h1_layers(ell, n, 10, ell == 2 ? 4 : 2);
                const MvFrozen* mv = MvRow(ell, n);
                ++g_cells;
                const uint32_t w = (n % 6 == 0 || n % 6 == 4) ? 1 : 0;
                bool ok = check(mv && mv->h0 == lc.h0 && mv->h1 == lc.h1 && mv->bredon == lc.dim_br && lc.dim_t1 == lc.h0 && lc.br_stable && lc.cusp_stable,
                                "(G) h0 / h1 / bredon_h1 vs mv_numpy.py, dim T^1/ell = h0, stability (ell=" + str(ell) + " n=" + str(n) + ")");
                if (ell == 3)
                    ok = check(lc.dim_L == w && lc.dim_L_br == 0 && lc.dim_t1_br + ((n % 6 == 4) ? 1u : 0u) == lc.dim_t1,
                               "(G) L ([R] Thm C(3)) and T^1/3 vs H^1_Br (C(4) restated) n=" + str(n)) && ok;
                if (ok) ++g_ok;
                std::cout << "  (G) LAYERS ell=" << ell << " n=" << n << " h1=" << lc.h1 << " br=" << lc.dim_br << " cusp=" << lc.dim_cusp << " t1=" << lc.dim_t1
                          << " (" << lc.seconds << " s)" << std::endl;
            }
        std::cout << "  (G) LAYERS_MV cells=" << g_cells << " agree=" << g_ok << " s=" << since(tg) << std::endl;
    }

    // ---------------- (H) the oracle at the L23-4 calibration cells (n = 240 ... 700) -------------------------------------------
    uint32_t h_cells = 0, h_ok = 0;
    {
        const auto th = std::chrono::steady_clock::now();
        for (const CalFrozen& r : kCal) {
            const l23::SumRuleRow S = l23::sum_rule_oracle(r.ell, r.n, kPrimes);
            ++h_cells;
            bool ok = S.resolved && S.vdet_tell == r.vtell && r.sum_tell == 2 * S.vdet_tell;
            for (uint32_t i = 0; i < 3; ++i) {
                const uint32_t p = r.s[i].p;
                ok = ok && S.vdet_tp.at(p) == r.v[i].tp && S.vdet_eis.at(p) == r.v[i].eis && r.s[i].tp == 2 * S.vdet_tp.at(p) + S.v_pn1.at(p) &&
                     r.s[i].eis == 2 * S.vdet_eis.at(p);
            }
            if (check(ok, "(H) oracle vs the L23-4 record sums at (ell=" + str(r.ell) + ", n=" + str(r.n) + ")")) ++h_ok;
            std::cout << "  (H) " << (S.log.empty() ? std::string("-") : S.log.back()) << " vdet_tell=" << S.vdet_tell << " total " << S.seconds << " s" << std::endl;
        }
        std::cout << "  (H) CAL_SUMRULES cells=" << h_cells << " agree=" << h_ok << " s=" << since(th) << std::endl;
    }

    const double s = since(t0);
    std::cout << "ORBIFOLD_CENSUS_L23_SLOW reader_sumrules=" << e_ok << "/" << e_cells << " ci_lat=" << f_ok << "/" << f_cells << " layers_mv=" << g_ok << "/"
              << g_cells << " calibration=" << h_ok << "/" << h_cells << " checks=" << g_checks << " failing=" << g_fail << " s=" << s
              << " verdict=" << (g_fail == 0 ? "OK" : "FAIL") << std::endl;
    return g_fail;
}
