// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 23, F3L) OrbifoldF3lValidation -- see OrbifoldF3lValidation.h.
#include "OrbifoldF3lValidation.h"

#include "CampaignOrbifold.h"
#include "CampaignSpec.h"
#include "../Integral/Model/IsotropyModel.h"
#include "../Integral/Model/StrictLift.h"
#include "../../ModArithmetic/DenseElim.h"
#include "../../ModArithmetic/Fnv.h"
#include "../../ModArithmetic/ModCtx.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace {

using engine::DenseBuf;
using engine::ModCtx;
using hecke::Mat22;
using Clock = std::chrono::steady_clock;

uint32_t g_checks = 0, g_fail = 0;
bool check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        if (g_fail <= 40) std::cout << "  [FAIL] " << what << std::endl;
    }
    return ok;
}
std::string str(uint64_t x) { return std::to_string(x); }
std::string sstr(int64_t x) { return std::to_string(x); }
double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

// (S24 L23-H1) formerly GL-1's H-1 mask; orbifold_cell/v2 has no wall time in the body: the identity, counting the oracle-gate sites.
std::string MaskBody(const std::string& body, uint32_t* count) {
    static const std::regex re(R"re((\{"name":"sumrule_oracle_resolved",.*?"detail":"[^"]*?-> resolved)("))re");
    if (count) {
        *count = 0;
        for (auto it = std::sregex_iterator(body.begin(), body.end(), re); it != std::sregex_iterator(); ++it) ++*count;
    }
    return std::regex_replace(body, re, "$1$2");   // identity: orbifold_cell/v2 has no wall time in the body (S24 L23-H1)
}
// The log's PROFILE lines carry wall times: mask every decimal fraction.
std::string MaskLog(const std::vector<std::string>& log) {
    static const std::regex re(R"([0-9]+\.[0-9]+)");
    std::string s;
    for (const std::string& l : log) s += std::regex_replace(l, re, "<f>") + "\n";
    return s;
}
template<class Elt>
bool SameBuf(const DenseBuf<Elt>& a, const DenseBuf<Elt>& b) {
    if (a.rows != b.rows || a.cols != b.cols) return false;
    for (uint32_t i = 0; i < a.rows; ++i)
        for (uint32_t j = 0; j < a.cols; ++j)
            if (a.at(i, j) != b.at(i, j)) return false;
    return true;
}
campaign::CampaignSpec Spec(const std::string& text) {
    std::istringstream is(text);
    campaign::CampaignSpec s = campaign::ParseCampaignSpec(is, "<f3l-spec>");
    s.output_dir = "unused";
    return s;
}
std::string GStr(const Mat22& g) { return "[" + sstr(g.a) + "," + sstr(g.b) + ";" + sstr(g.c) + "," + sstr(g.d) + "]"; }

struct PStats { uint64_t cmp = 0, up = 0, down = 0, refused = 0, entries = 0; };

// One (ctx, g, n): the recurrence (when it runs) equals the cubic sum entrywise; it refuses exactly when neither a nor
// c is a unit mod p.
template<class Elt>
void RhoCase(const ModCtx<Elt>& R, const std::string& rname, const Mat22& g, uint32_t n, PStats& st) {
    const DenseBuf<Elt> ref = integral_model::rho_n_cubic<Elt>(R, g, n);
    DenseBuf<Elt> rec;
    const bool ran = integral_model::rho_n_recurrence<Elt>(R, g, n, rec);
    auto unit = [&](int64_t v) {
        const uint64_t p = static_cast<uint64_t>(R.p);
        int64_t r = v % static_cast<int64_t>(p);
        if (r < 0) r += static_cast<int64_t>(p);
        uint64_t a = static_cast<uint64_t>(r), b = p;
        while (b) { const uint64_t t = a % b; a = b; b = t; }
        return a == 1;
    };
    const bool expect = unit(g.a) || unit(g.c);
    ++st.cmp;
    check(ran == expect, "(P) " + rname + " g=" + GStr(g) + " n=" + str(n) + ": the recurrence runs iff a or c is a unit");
    if (!ran) { ++st.refused; return; }
    (unit(g.a) ? st.up : st.down) += 1;
    st.entries += uint64_t(n + 1) * (n + 1);
    check(SameBuf(ref, rec), "(P) " + rname + " g=" + GStr(g) + " n=" + str(n) + ": rho_n recurrence == cubic entrywise");
}

template<class Elt>
void RhoArm(const ModCtx<Elt>& R, const std::string& rname, const std::vector<uint32_t>& ns, std::mt19937_64& rng, PStats& st,
            uint32_t randoms) {
    std::vector<Mat22> gs = {
        Mat22{1, 0, 0, 1},  Mat22{1, 1, 0, 1},  Mat22{0, -1, 1, 0}, Mat22{1, 0, 2, 1},  Mat22{1, 0, 0, 2},  Mat22{2, 0, 0, 1},
        Mat22{1, 0, 0, 3},  Mat22{3, 0, 0, 1},  Mat22{0, 1, 3, 5},  Mat22{3, 1, 1, 0},  Mat22{3, 2, 5, 7},  Mat22{9, 4, 2, 1},
        Mat22{0, 7, 0, 5},  Mat22{6, 1, 9, 2},  Mat22{-4, 3, -7, 5}, Mat22{1, 5, 0, 7}, Mat22{0, 5, 1, 7}, Mat22{4, -1, -9, 3},
        Mat22{2, 1, 4, 3},  Mat22{5, 2, 10, 4}, Mat22{25, 1, 5, 7}, Mat22{16, 3, 8, 1},
    };
    std::uniform_int_distribution<int64_t> e(-60, 60);
    for (uint32_t i = 0; i < randoms; ++i) gs.push_back(Mat22{e(rng), e(rng), e(rng), e(rng)});
    // planted: a = 0 mod ell (descending path) and c = 0 mod ell, from the context's own prime
    const int64_t l = static_cast<int64_t>(R.ell ? R.ell : R.p);
    // a multiple of ell drawn from e, with the multiplier folded into the range where 1 + l*m fits in int64
    // (l = 2^61 - 1 on the u64 arm: a raw l * e(rng) overflows the signed product)
    const int64_t mmax = (std::numeric_limits<int64_t>::max() - 1) / l;
    auto lmul = [&]() { int64_t m = e(rng); if (m > mmax || m < -mmax) m %= (mmax + 1); return l * m; };
    for (uint32_t i = 0; i < 4; ++i) {
        gs.push_back(Mat22{lmul(), e(rng), 1 + lmul(), e(rng)});
        gs.push_back(Mat22{1 + lmul(), e(rng), lmul(), e(rng)});
        gs.push_back(Mat22{lmul(), e(rng), lmul(), e(rng)});   // neither a unit: refused (cubic fallback)
    }
    for (uint32_t n : ns)
        for (const Mat22& g : gs) RhoCase<Elt>(R, rname, g, n, st);
}

}  // namespace

uint32_t OrbifoldF3lValidation() {
    std::cout << "=== OrbifoldF3lValidation (slot-23 F3L: rho_n recurrence + bit-sliced F_3 elimination in ell = 3 cells, route on vs off) ===" << std::endl;
    g_checks = g_fail = 0;
    const auto t0 = Clock::now();
    std::mt19937_64 rng(0xF3F3F3ull);

    // ---------------- (P) rho_n: recurrence vs the cubic sum ----------------
    PStats ps;
    {
        std::vector<uint32_t> small;
        for (uint32_t n = 0; n < 40; ++n) small.push_back(n);
        const std::vector<uint32_t> big = {63, 64, 79, 80, 81, 82, 127, 128, 241, 242, 243, 244, 255, 256, 400, 401};
        for (uint32_t p : {2u, 3u, 5u, 7u, 13u, 691u, 134217757u}) {
            const ModCtx<uint32_t> F(p);
            RhoArm<uint32_t>(F, "F_" + str(p), small, rng, ps, 6);
            RhoArm<uint32_t>(F, "F_" + str(p), (p == 3 || p == 2) ? big : std::vector<uint32_t>{81, 242}, rng, ps, 1);
        }
        const std::vector<std::pair<uint32_t, uint32_t>> rings = {{3, 2}, {3, 19}, {2, 4}, {2, 31}, {5, 3}};
        for (const auto& [l, k] : rings) {
            const ModCtx<uint32_t> R = ModCtx<uint32_t>::prime_power(l, k);
            RhoArm<uint32_t>(R, "Z/" + str(l) + "^" + str(k), small, rng, ps, 4);
            RhoArm<uint32_t>(R, "Z/" + str(l) + "^" + str(k), {80, 81, 242}, rng, ps, 1);
        }
        const ModCtx<uint64_t> F64(2305843009213693951ull);   // 2^61 - 1
        RhoArm<uint64_t>(F64, "F_(2^61-1)", {0, 1, 2, 7, 26, 27, 81}, rng, ps, 4);
        const ModCtx<uint64_t> Z64 = ModCtx<uint64_t>::prime_power(3ull, 30);
        RhoArm<uint64_t>(Z64, "Z/3^30 (u64)", {0, 1, 5, 26, 81}, rng, ps, 4);
    }
    check(ps.up > 0 && ps.down > 0 && ps.refused > 0, "(P) non-vacuous: ascending, descending and refused cases all occur");
    {   // the public rho_n: on = off, and the counters move on the selected path only
        const ModCtx<uint32_t> F3(3u);
        const std::vector<Mat22> gs = {Mat22{1, 0, 0, 3}, Mat22{3, 1, 2, 5}, Mat22{3, 1, 6, 5}, Mat22{1, 2, 3, 7}};
        for (const Mat22& g : gs)
            for (uint32_t n : {0u, 1u, 26u, 27u, 100u}) {
                const integral_model::RhoNTotals a = integral_model::rho_n_totals();
                DenseBuf<uint32_t> on, off;
                { integral_model::ScopedF3lRoute s(true); on = integral_model::rho_n(F3, g, n); }
                const integral_model::RhoNTotals b = integral_model::rho_n_totals();
                { integral_model::ScopedF3lRoute s(false); off = integral_model::rho_n(F3, g, n); }
                const integral_model::RhoNTotals c = integral_model::rho_n_totals();
                check(SameBuf(on, off), "(P) public rho_n on == off at g=" + GStr(g) + " n=" + str(n));
                const bool fb = (g.a % 3 == 0) && (g.c % 3 == 0);
                check(b.recurrence - a.recurrence == (fb ? 0u : 1u) && b.fallback - a.fallback == (fb ? 1u : 0u) && b.cubic == a.cubic &&
                          c.cubic - b.cubic == 1 && c.recurrence == b.recurrence && c.fallback == b.fallback,
                      "(P) counters at g=" + GStr(g) + " n=" + str(n) + ": on = recurrence (or fallback), off = cubic");
            }
    }
    {   // (diagnostic, not a gate) one rho_n at d_V = 401 over F_3, each path
        const ModCtx<uint32_t> F3(3u);
        const Mat22 g{4, 7, 9, 13};
        const auto a = Clock::now();
        const DenseBuf<uint32_t> c = integral_model::rho_n_cubic<uint32_t>(F3, g, 400);
        const auto b = Clock::now();
        DenseBuf<uint32_t> r;
        const bool ran = integral_model::rho_n_recurrence<uint32_t>(F3, g, 400, r);
        const auto e = Clock::now();
        check(ran && SameBuf(c, r), "(P) the timing pair at d_V = 401 agrees");
        std::cout << "  F3L_RHO_TIMING d=401 p=3 cubic_ms=" << 1e3 * secs(a, b) << " recurrence_ms=" << 1e3 * secs(b, e)
                  << " ratio=" << secs(a, b) / std::max(1e-9, secs(b, e)) << std::endl;
    }
    std::cout << "  F3L_RHO cases=" << ps.cmp << " ascending=" << ps.up << " descending=" << ps.down << " refused=" << ps.refused
              << " entries=" << ps.entries << std::endl;

    // ---------------- (T) theta_hat, the census's T_p, on = off ----------------
    uint32_t th = 0;
    {
        const std::vector<ModCtx<uint32_t>> ctxs = {ModCtx<uint32_t>(3u), ModCtx<uint32_t>(2u), ModCtx<uint32_t>::prime_power(3u, 2)};
        for (uint32_t p : {2u, 3u, 5u, 7u}) {
            const integral_model::StrictLift& L = campaign::OrbifoldStrictLift(p);
            for (const auto& R : ctxs)
                for (uint32_t n : {0u, 1u, 4u, 9u, 26u, 27u})
                    for (uint32_t q = 0; q <= 3; ++q) {
                        DenseBuf<uint32_t> on, off;
                        { integral_model::ScopedF3lRoute s(true); on = L.theta_hat(R, n, q); }
                        { integral_model::ScopedF3lRoute s(false); off = L.theta_hat(R, n, q); }
                        ++th;
                        check(SameBuf(on, off), "(T) theta_hat p=" + str(p) + " mod " + str(R.p) + " n=" + str(n) + " q=" + str(q) + " on == off");
                    }
        }
    }
    std::cout << "  F3L_THETA matrices=" << th << std::endl;

    // ---------------- (R) whole records, route on vs off ----------------
    const campaign::CampaignSpec spec = Spec("orbifold = c_i\nn_sym = 0..64\nell = 2,3\n");
    const std::vector<std::pair<uint32_t, uint32_t>> cells = {
        {3, 0},  {3, 1},  {3, 2},  {3, 3},  {3, 4},  {3, 5},  {3, 6},  {3, 7},  {3, 8},  {3, 9},  {3, 10}, {3, 11}, {3, 12},
        {3, 16}, {3, 17}, {3, 25}, {3, 26}, {3, 27}, {3, 28}, {3, 29}, {3, 40}, {3, 53}, {3, 54}, {3, 80}, {3, 81}, {3, 82},
        {2, 0},  {2, 1},  {2, 2},  {2, 9},  {2, 10}, {2, 26}, {2, 28},
    };
    const bool tern_env = engine::elim::ternary_f3_enabled();   // ENGINE_TERNARY_F3=1 in the environment: off still routes
    uint32_t rec_cells = 0, masked_sites = 0, ok_cells = 0, l3_cells = 0;
    double r_on = 0, r_off = 0;
    for (const auto& [ell, n] : cells) {
        const engine::elim::TernaryF3Totals x0 = engine::elim::ternary_f3_totals();
        const campaign::OrbifoldF3lCounters k0 = campaign::OrbifoldF3lCalls();
        const auto a = Clock::now();
        campaign::OrbifoldCell on, off;
        { integral_model::ScopedF3lRoute s(true); on = campaign::ComputeOrbifoldCell(spec, ell, n); }
        const auto b = Clock::now();
        const engine::elim::TernaryF3Totals x1 = engine::elim::ternary_f3_totals();
        const campaign::OrbifoldF3lCounters k1 = campaign::OrbifoldF3lCalls();
        { integral_model::ScopedF3lRoute s(false); off = campaign::ComputeOrbifoldCell(spec, ell, n); }
        const auto c = Clock::now();
        const engine::elim::TernaryF3Totals x2 = engine::elim::ternary_f3_totals();
        const campaign::OrbifoldF3lCounters k2 = campaign::OrbifoldF3lCalls();
        r_on += secs(a, b);
        r_off += secs(b, c);
        ++rec_cells;
        const std::string at = "(" + str(ell) + ", " + str(n) + ")";
        if (ell == 3) {
            ++l3_cells;
            check(k1.ternary_cells - k0.ternary_cells == 1 && x1.rrefs + x1.rank_profiles > x0.rrefs + x0.rank_profiles,
                  "(R) route on at " + at + ": the cell ran under the ternary scope and routed F_3 eliminations");
            check(k2.ternary_cells == k1.ternary_cells && (tern_env || (x2.rrefs == x1.rrefs && x2.rank_profiles == x1.rank_profiles)),
                  "(R) route off at " + at + ": no ternary scope, no ternary elimination (unless ENGINE_TERNARY_F3=1)");
        } else {
            check(k1.ternary_cells == k0.ternary_cells && k2.ternary_cells == k1.ternary_cells, "(R) ell = 2 at " + at + ": never the ternary scope");
        }
        check(k1.rho_recurrence > k0.rho_recurrence && k1.rho_cubic == k0.rho_cubic, "(R) route on at " + at + ": rho_n by the recurrence only");
        check(k2.rho_cubic > k1.rho_cubic && k2.rho_recurrence == k1.rho_recurrence && k2.rho_fallback == k1.rho_fallback,
              "(R) route off at " + at + ": rho_n by the cubic sum only");
        uint32_t m_on = 0, m_off = 0;
        const std::string b_on = campaign::OrbifoldCellBody(spec, on), b_off = campaign::OrbifoldCellBody(spec, off);
        const std::string mb_on = MaskBody(b_on, &m_on), mb_off = MaskBody(b_off, &m_off);
        // (L23-H1 audit F3) orbifold_cell/v2: no wall time may come back into the body (the identity "mask" alone would not notice)
        check(b_on.find(" s)\"") == std::string::npos && b_off.find(" s)\"") == std::string::npos && b_on.find("\"schema\":\"orbifold_cell/v2\"") == 0,
              "(R) v2 record body carries no wall time" + std::string(" (") + std::to_string(m_on) + " oracle gate site)");
        masked_sites += m_on;
        check(m_on == m_off && m_on <= 1, "(R) at " + at + ": the masked sites agree (" + str(m_on) + " / " + str(m_off) + ")");
        check(mb_on == mb_off, "(R) record body at " + at + " byte-identical route on vs off (orbifold_cell/v2: no mask)");
        check(engine::FnvBytes(engine::kFnvOffset64, mb_on) == engine::FnvBytes(engine::kFnvOffset64, mb_off), "(R) masked record_fnv1a64 at " + at);
        if (m_on == 0) check(b_on == b_off, "(R) unmasked body at " + at + " byte-identical (no sum-rule site)");
        check(MaskLog(on.log) == MaskLog(off.log), "(R) log at " + at + " identical after masking the decimal fractions");
        check(on.status == off.status && on.h[0] == off.h[0] && on.h[1] == off.h[1] && on.h[2] == off.h[2] && on.h[3] == off.h[3] &&
                  on.ci_q1_exponents == off.ci_q1_exponents && on.ladic_type[2] == off.ladic_type[2] && on.ladic_type[3] == off.ladic_type[3] &&
                  on.sub_dim[1] == off.sub_dim[1] && on.sub_dim[2] == off.sub_dim[2] && on.e_cup_h0_h2 == off.e_cup_h0_h2 &&
                  on.e_cup_h1_h3 == off.e_cup_h1_h3 && on.e_cup_equivariant == off.e_cup_equivariant && std::equal(on.t_ell_rank, on.t_ell_rank + 4, off.t_ell_rank) &&
                  on.tp_scalar == off.tp_scalar,
              "(R) the (F) / (Z) fields at " + at);
        check(on.rung == off.rung && on.K == off.K && on.m == off.m && on.e_A == off.e_A && on.e_1 == off.e_1 && on.reruns == off.reruns &&
                  on.sumrule_K == off.sumrule_K && on.sumrule_max_invariant == off.sumrule_max_invariant,
              "(R) reader_path / sumrule_path at " + at + " (values outside the hash)");
        ok_cells += on.status == "ok";
    }
    check(ok_cells == rec_cells, "(R) every compared cell has status ok (" + str(ok_cells) + " / " + str(rec_cells) + ")");
    check(masked_sites > 0, "(R) non-vacuous: the H-1 mask was exercised");
    std::cout << "  F3L_RECORDS cells=" << rec_cells << " ell3=" << l3_cells << " ok=" << ok_cells << " masked_sites=" << masked_sites
              << " on_s=" << r_on << " off_s=" << r_off << std::endl;

    // ---------------- (C) the switch and its EFFECTIVE line ----------------
    {
        integral_model::ScopedF3lRoute s(false);
        const std::string e = campaign::OrbifoldF3lEffectiveLine();
        check(!integral_model::F3lRouteEnabled() && e.rfind("EFFECTIVE l23_f3l=off source=scoped rho_n=cubic", 0) == 0, "(C) scoped off: " + e);
    }
    {
        integral_model::ScopedF3lRoute s(true);
        const std::string e = campaign::OrbifoldF3lEffectiveLine();
        check(integral_model::F3lRouteEnabled() && e.rfind("EFFECTIVE l23_f3l=on source=scoped rho_n=recurrence ell3_f3_elim=ternary", 0) == 0,
              "(C) scoped on: " + e);
    }
    {   // the scope is restored after a cell (no leak of the ternary override into the caller)
        const bool before = engine::elim::ternary_f3_enabled();
        { integral_model::ScopedF3lRoute s(true); (void)campaign::ComputeOrbifoldCell(spec, 3, 2); }
        check(engine::elim::ternary_f3_enabled() == before, "(C) the cell's ScopedTernaryF3 is released on return");
    }
    std::cout << "  " << campaign::OrbifoldF3lEffectiveLine() << std::endl;

    std::cout << "ORBIFOLD_F3L rho_cases=" << ps.cmp << " theta=" << th << " record_cells=" << rec_cells << " masked_sites=" << masked_sites
              << " checks=" << g_checks << " failing=" << g_fail << " s=" << secs(t0, Clock::now()) << " verdict=" << (g_fail ? "FAIL" : "OK")
              << std::endl;
    return g_fail;
}
