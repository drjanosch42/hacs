// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 23, Z2K) OrbifoldRing2kValidation -- see OrbifoldRing2kValidation.h.
#include "OrbifoldRing2kValidation.h"

#include "CampaignOrbifold.h"
#include "CampaignSpec.h"
#include "../Integral/Model/IntegralHecke.h"
#include "../Integral/Model/IsotropyModel.h"
#include "../../ModArithmetic/Fnv.h"

#include <chrono>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using engine::DenseBuf;
using U32 = uint32_t;
using Clock = std::chrono::steady_clock;

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
double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

// (S24 L23-H1) orbifold_cell/v1 carried one wall-clock string inside a record body, the sum-rule oracle's "-> resolved (0.274 s)"
// at the end of the detail of the gate sumrule_oracle_resolved (GPU_scout.md H-1), masked here to "(<t> s)".  orbifold_cell/v2 keeps
// the time outside the body, so this is the IDENTITY now; it only counts the oracle-gate sites (`count`, the non-vacuity figure).
std::string MaskBody(const std::string& body, uint32_t* count) {
    static const std::regex re(R"re((\{"name":"sumrule_oracle_resolved",.*?"detail":"[^"]*?-> resolved)("))re");
    if (count) {
        *count = 0;
        for (auto it = std::sregex_iterator(body.begin(), body.end(), re); it != std::sregex_iterator(); ++it) ++*count;
    }
    return std::regex_replace(body, re, "$1$2");   // identity: orbifold_cell/v2 has no wall time in the body (S24 L23-H1)
}
// A log line: every decimal fraction (the PROFILE laps, the oracle's wall time) masked; integers stay.
std::string MaskLog(const std::vector<std::string>& log) {
    static const std::regex re(R"([0-9]+\.[0-9]+)");
    std::string s;
    for (const std::string& l : log) s += std::regex_replace(l, re, "<f>") + "\n";
    return s;
}
bool SameHowell(const campaign::OrbifoldHowell& a, const campaign::OrbifoldHowell& b) {
    if (a.H.rows != b.H.rows || a.H.cols != b.H.cols || a.pivots != b.pivots || a.pivot_v != b.pivot_v) return false;
    for (uint32_t i = 0; i < a.H.rows; ++i)
        for (uint32_t j = 0; j < a.H.cols; ++j)
            if (a.H.at(i, j) != b.H.at(i, j)) return false;
    return true;
}
std::string jl(const std::vector<uint32_t>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + std::to_string(v[i]);
    return s + "]";
}
campaign::CampaignSpec Spec(const std::string& text) {
    std::istringstream is(text);
    campaign::CampaignSpec s = campaign::ParseCampaignSpec(is, "<z2k-spec>");
    s.output_dir = "unused";
    return s;
}

}  // namespace

uint32_t OrbifoldRing2kValidation() {
    g_checks = g_fail = 0;
    const auto t0 = Clock::now();
    const integral_model::IsotropyModel& M = campaign::OrbifoldReferenceModel();

    // ---------------- (M) the readings on the real coboundaries, route on = off ----------------
    const std::vector<uint32_t> mat_n = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 24, 31, 32, 33, 47, 48, 63, 64, 65, 96, 100, 128};
    uint32_t smith_cmp = 0, kernel_cmp = 0, nonzero_torsion = 0, kernel_rows = 0, nonunit_pivots = 0;
    double t_on = 0, t_off = 0;
    for (uint32_t n : mat_n) {
        for (uint32_t q = 1; q <= 3; ++q) {
            const uint32_t K = (q == 1) ? 31u : 4u;
            const DenseBuf<U32> D = M.coboundary(integral_model::RingContext(2, K), n, q - 1);
            auto a = Clock::now();
            const std::vector<uint32_t> on = campaign::OrbifoldSmithTorsion(2, K, D, true);
            auto b = Clock::now();
            const std::vector<uint32_t> off = campaign::OrbifoldSmithTorsion(2, K, D, false);
            auto c = Clock::now();
            t_on += secs(a, b);
            t_off += secs(b, c);
            ++smith_cmp;
            if (!on.empty()) ++nonzero_torsion;
            check(on == off, "(M) Smith torsion of delta^" + str(q - 1) + " over Z/2^" + str(K) + " at n = " + str(n) + ": ring2k " + jl(on) +
                                 " vs engine " + jl(off));
        }
        for (uint32_t q = 1; q <= 2; ++q) {
            const DenseBuf<U32> D = M.coboundary(integral_model::RingContext(2, 4), n, q);
            auto a = Clock::now();
            const campaign::OrbifoldHowell on = campaign::OrbifoldKernelModule(2, 4, D, true);
            auto b = Clock::now();
            const campaign::OrbifoldHowell off = campaign::OrbifoldKernelModule(2, 4, D, false);
            auto c = Clock::now();
            t_on += secs(a, b);
            t_off += secs(b, c);
            ++kernel_cmp;
            kernel_rows += on.H.rows;
            for (uint32_t v : on.pivot_v) nonunit_pivots += v > 0;
            check(SameHowell(on, off), "(M) Howell kernel module of delta^" + str(q) + " over Z/16 at n = " + str(n) + " (" + str(on.H.rows) + " x " +
                                           str(on.H.cols) + " vs " + str(off.H.rows) + " x " + str(off.H.cols) + ")");
        }
    }
    check(nonzero_torsion > 0 && nonunit_pivots > 0, "(M) non-vacuous: some torsion and some non-unit Howell pivot were compared");
    std::cout << "  RING2K_ROUTE_MATRIX cells=" << mat_n.size() << " smith=" << smith_cmp << " kernels=" << kernel_cmp << " nonzero_torsion=" << nonzero_torsion
              << " kernel_rows=" << kernel_rows << " nonunit_pivots=" << nonunit_pivots << " ring2k_s=" << t_on << " engine_s=" << t_off << std::endl;

    // ---------------- (R) whole records, route on vs off ----------------
    const campaign::CampaignSpec spec = Spec("orbifold = c_i\nn_sym = 0..64\nell = 2,3\n");
    const std::vector<uint32_t> rec_n = {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 22, 24, 31, 32, 33, 40, 48, 64, 80, 100, 128};
    uint32_t rec_cells = 0, masked_sites = 0, ok_cells = 0, tors_cells = 0;
    double r_on = 0, r_off = 0;
    for (uint32_t n : rec_n) {
        const campaign::OrbifoldRing2kCounters k0 = campaign::OrbifoldRing2kCalls();
        auto a = Clock::now();
        campaign::OrbifoldCell on, off;
        {
            campaign::ScopedOrbifoldRing2k s(true);
            on = campaign::ComputeOrbifoldCell(spec, 2, n);
        }
        auto b = Clock::now();
        const campaign::OrbifoldRing2kCounters k1 = campaign::OrbifoldRing2kCalls();
        {
            campaign::ScopedOrbifoldRing2k s(false);
            off = campaign::ComputeOrbifoldCell(spec, 2, n);
        }
        auto c = Clock::now();
        const campaign::OrbifoldRing2kCounters k2 = campaign::OrbifoldRing2kCalls();
        r_on += secs(a, b);
        r_off += secs(b, c);
        ++rec_cells;
        check(k1.ring2k_smith - k0.ring2k_smith == 3 && k1.ring2k_kernel - k0.ring2k_kernel == 2 && k1.engine_smith == k0.engine_smith &&
                  k1.engine_kernel == k0.engine_kernel,
              "(R) route on at (2, " + str(n) + "): 3 ladder Smith + 2 ladder kernel calls, no engine call");
        check(k2.engine_smith - k1.engine_smith == 3 && k2.engine_kernel - k1.engine_kernel == 2 && k2.ring2k_smith == k1.ring2k_smith &&
                  k2.ring2k_kernel == k1.ring2k_kernel,
              "(R) route off at (2, " + str(n) + "): 3 engine Smith + 2 engine kernel calls, no ladder call");
        uint32_t m_on = 0, m_off = 0;
        const std::string b_on = campaign::OrbifoldCellBody(spec, on), b_off = campaign::OrbifoldCellBody(spec, off);
        const std::string mb_on = MaskBody(b_on, &m_on), mb_off = MaskBody(b_off, &m_off);
        // (L23-H1 audit F3) orbifold_cell/v2: no wall time may come back into the body (the identity "mask" alone would not notice)
        check(b_on.find(" s)\"") == std::string::npos && b_off.find(" s)\"") == std::string::npos && b_on.find("\"schema\":\"orbifold_cell/v2\"") == 0,
              "(R) v2 record body carries no wall time" + std::string(" (") + std::to_string(m_on) + " oracle gate site)");
        masked_sites += m_on;
        check(m_on == m_off && m_on <= 1, "(R) at (2, " + str(n) + "): the masked sites agree (" + str(m_on) + " / " + str(m_off) + ", at most the one H-1 site)");
        check(mb_on == mb_off, "(R) record body at (2, " + str(n) + ") byte-identical route on vs off (orbifold_cell/v2: no mask)");
        check(engine::FnvBytes(engine::kFnvOffset64, mb_on) == engine::FnvBytes(engine::kFnvOffset64, mb_off),
              "(R) masked record_fnv1a64 at (2, " + str(n) + ")");
        if (m_on == 0) check(b_on == b_off, "(R) unmasked body at (2, " + str(n) + ") byte-identical (no sum-rule site)");
        check(MaskLog(on.log) == MaskLog(off.log), "(R) log at (2, " + str(n) + ") identical after masking the decimal fractions");
        check(on.status == off.status && on.ci_q1_exponents == off.ci_q1_exponents && on.ladic_type[2] == off.ladic_type[2] &&
                  on.ladic_type[3] == off.ladic_type[3] && on.sub_dim[1] == off.sub_dim[1] && on.sub_dim[2] == off.sub_dim[2],
              "(R) the (Z) fields at (2, " + str(n) + ")");
        check(on.rung == off.rung && on.K == off.K && on.m == off.m && on.e_A == off.e_A && on.e_1 == off.e_1 && on.reruns == off.reruns &&
                  on.sumrule_K == off.sumrule_K && on.sumrule_max_invariant == off.sumrule_max_invariant,
              "(R) reader_path / sumrule_path at (2, " + str(n) + ") (values outside the hash; Z2K audit M1)");
        ok_cells += on.status == "ok";
        tors_cells += !on.ci_q1_exponents.empty() || !on.ladic_type[2].empty() || !on.ladic_type[3].empty();
    }
    check(ok_cells == rec_cells, "(R) every compared cell has status ok (" + str(ok_cells) + " / " + str(rec_cells) + ")");
    check(masked_sites > 0 && tors_cells > 0, "(R) non-vacuous: the H-1 mask was exercised and some cell carries (Z) torsion");
    {   // ell = 3 never reaches the ladder, whatever the switch says
        const campaign::OrbifoldRing2kCounters k0 = campaign::OrbifoldRing2kCalls();
        campaign::ScopedOrbifoldRing2k s(true);
        campaign::ScopedOrbifoldMod3k s3(false);   // (slot-23 Z3K) ell = 3's own ladder off: the engine count below stays exact
        const campaign::OrbifoldCell c3 = campaign::ComputeOrbifoldCell(spec, 3, 10);
        const campaign::OrbifoldRing2kCounters k1 = campaign::OrbifoldRing2kCalls();
        check(c3.status == "ok" && k1.ring2k_smith == k0.ring2k_smith && k1.ring2k_kernel == k0.ring2k_kernel && k1.engine_smith - k0.engine_smith == 3,
              "(R) ell = 3 with the route on: engine calls only");
    }
    std::cout << "  RING2K_ROUTE_RECORDS cells=" << rec_cells << " ok=" << ok_cells << " with_torsion=" << tors_cells << " masked_sites=" << masked_sites
              << " on_s=" << r_on << " off_s=" << r_off << std::endl;

    // ---------------- (C) negative controls and the switch ----------------
    {
        const std::string body = "{\"x\":[1,2],\"gates\":[{\"name\":\"sumrule_oracle_resolved\",\"kind\":\"G\",\"ok\":true,\"detail\":"
                                 "\"SUMRULE_ORACLE ell=2 n=10 dimS=0 K=64 max_invariant=0 -> resolved\"}]}";
        uint32_t m = 0;
        check(MaskBody(body, &m) == body && m == 1, "(C) v2: one oracle gate site, the body carries no wall time (nothing to mask)");
        std::string u = body;
        u[u.find("1,2")] = '3';
        check(MaskBody(body, nullptr) != MaskBody(u, nullptr), "(C) the masked comparison flags a one-character change of a value");
        std::string v = body;
        v.replace(v.find("resolved"), 8, "resolvee");
        check(MaskBody(body, nullptr) != MaskBody(v, nullptr), "(C) the mask is anchored on the oracle's phrase (no free-floating mask)");
        std::string w = body;
        w.replace(w.find("sumrule_oracle_resolved"), 23, "sumrule_oracle_elsewhere");
        uint32_t mw = 1;
        check(MaskBody(w, &mw) == w && mw == 0, "(C) the mask is anchored on the gate sumrule_oracle_resolved (no other gate is masked)");
    }
    {   // (Z2K audit L2) K = 1 has no engine rollback (composite-only), so the ladder refuses it too
        const DenseBuf<U32> D = M.coboundary(integral_model::RingContext(2, 4), 6, 0);
        bool threw_s = false, threw_k = false;
        try { (void)campaign::OrbifoldSmithTorsion(2, 1, D, true); } catch (const std::logic_error&) { threw_s = true; }
        try { (void)campaign::OrbifoldKernelModule(2, 1, D, true); } catch (const std::logic_error&) { threw_k = true; }
        check(threw_s && threw_k, "(C) the ring2k route refuses K = 1 / E = 1 (ENGINE_CHECK 2 <= K <= 32)");
    }
    {
        campaign::ScopedOrbifoldRing2k s(false);
        const std::string e = campaign::OrbifoldRing2kEffectiveLine();
        check(!campaign::OrbifoldRing2kEnabled() && e.rfind("EFFECTIVE l23_ring2k=off source=scoped", 0) == 0, "(C) scoped off: " + e);
    }
    {
        campaign::ScopedOrbifoldRing2k s(true);
        const std::string e = campaign::OrbifoldRing2kEffectiveLine();
        check(campaign::OrbifoldRing2kEnabled() && e.rfind("EFFECTIVE l23_ring2k=on source=scoped", 0) == 0, "(C) scoped on: " + e);
    }
    std::cout << "  " << campaign::OrbifoldRing2kEffectiveLine() << std::endl;

    std::cout << "ORBIFOLD_RING2K matrix_cells=" << mat_n.size() << " smith=" << smith_cmp << " kernels=" << kernel_cmp << " record_cells=" << rec_cells
              << " masked_sites=" << masked_sites << " checks=" << g_checks << " failing=" << g_fail << " s=" << secs(t0, Clock::now())
              << " verdict=" << (g_fail ? "FAIL" : "OK") << std::endl;
    return g_fail;
}
