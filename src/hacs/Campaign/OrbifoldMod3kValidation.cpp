// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 23, Z3K) OrbifoldMod3kValidation -- see OrbifoldMod3kValidation.h.
#include "OrbifoldMod3kValidation.h"

#include "CampaignOrbifold.h"
#include "CampaignSpec.h"
#include "../Integral/Model/IntegralHecke.h"
#include "../Integral/Model/IsotropyModel.h"
#include "../../ModArithmetic/Binary/Ring3k.h"
#include "../../ModArithmetic/DenseElim.h"
#include "../../ModArithmetic/Fnv.h"
#include "../../ModArithmetic/ModCtx.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using engine::DenseBuf;
using engine::ModCtx;
using U32 = uint32_t;
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
double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

// (S24 L23-H1) formerly the H-1 mask of the oracle's wall time (GPU_scout.md H-1); orbifold_cell/v2 has no wall time in the
// body, so this is the identity and only counts the oracle-gate sites (as OrbifoldRing2kValidation's MaskBody).
std::string MaskBody(const std::string& body, uint32_t* count) {
    static const std::regex re(R"re((\{"name":"sumrule_oracle_resolved",.*?"detail":"[^"]*?-> resolved)("))re");
    if (count) {
        *count = 0;
        for (auto it = std::sregex_iterator(body.begin(), body.end(), re); it != std::sregex_iterator(); ++it) ++*count;
    }
    return std::regex_replace(body, re, "$1$2");   // identity: orbifold_cell/v2 has no wall time in the body (S24 L23-H1)
}
std::string MaskLog(const std::vector<std::string>& log) {
    static const std::regex re(R"([0-9]+\.[0-9]+)");
    std::string s;
    for (const std::string& l : log) s += std::regex_replace(l, re, "<f>") + "\n";
    return s;
}
bool SameBuf(const DenseBuf<U32>& a, const DenseBuf<U32>& b) {
    if (a.rows != b.rows || a.cols != b.cols) return false;
    for (uint32_t i = 0; i < a.rows; ++i)
        for (uint32_t j = 0; j < a.cols; ++j)
            if (a.at(i, j) != b.at(i, j)) return false;
    return true;
}
bool SameHowell(const campaign::OrbifoldHowell& a, const campaign::OrbifoldHowell& b) {
    return a.pivots == b.pivots && a.pivot_v == b.pivot_v && SameBuf(a.H, b.H);
}
bool SameHowell3(const engine::ring3k::HowellForm& a, const engine::ring3k::HowellForm& b) {
    return a.k == b.k && a.pivots == b.pivots && a.pivot_v == b.pivot_v && SameBuf(a.H, b.H);
}
bool SameHowellE(const engine::ring3k::HowellForm& a, const engine::elim::HowellResult<U32>& b) {
    return a.pivots == b.pivots && a.pivot_v == b.pivot_v && SameBuf(a.H, b.H);
}
std::string jl(const std::vector<uint32_t>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + std::to_string(v[i]);
    return s + "]";
}
campaign::CampaignSpec Spec(const std::string& text) {
    std::istringstream is(text);
    campaign::CampaignSpec s = campaign::ParseCampaignSpec(is, "<z3k-spec>");
    s.output_dir = "unused";
    return s;
}
uint32_t Pow3(uint32_t e) { uint32_t p = 1; for (uint32_t i = 0; i < e; ++i) p *= 3u; return p; }
ModCtx<U32> Ctx(uint32_t k) { return k == 1 ? ModCtx<U32>(3u) : ModCtx<U32>::prime_power(3u, k); }

// C = A B over ctx (naive; small sizes).
DenseBuf<U32> Mul(const ModCtx<U32>& R, const DenseBuf<U32>& A, const DenseBuf<U32>& B) {
    DenseBuf<U32> C(A.rows, B.cols);
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint32_t t = 0; t < A.cols; ++t) {
            const U32 a = A.at(i, t);
            if (a == 0) continue;
            for (uint32_t j = 0; j < B.cols; ++j) C.at(i, j) = R.add(C.at(i, j), R.mul(a, B.at(t, j)));
        }
    return C;
}
// A random unimodular n x n matrix over Z/3^k: (unit lower) (unit upper) with unit diagonal scalings, rows permuted.
DenseBuf<U32> Unimodular(const ModCtx<U32>& R, uint32_t n, std::mt19937_64& rng) {
    const U32 M = R.p;
    DenseBuf<U32> L(n, n), U(n, n);
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = 0; j < i; ++j) L.at(i, j) = static_cast<U32>(rng() % M);
        U32 u;
        do { u = static_cast<U32>(rng() % M); } while (u % 3u == 0);
        L.at(i, i) = u;
        U.at(i, i) = 1;
        for (uint32_t j = i + 1; j < n; ++j) U.at(i, j) = static_cast<U32>(rng() % M);
    }
    DenseBuf<U32> P = Mul(R, L, U);
    std::vector<uint32_t> perm(n);
    for (uint32_t i = 0; i < n; ++i) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);
    DenseBuf<U32> Q(n, n);
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t j = 0; j < n; ++j) Q.at(i, j) = P.at(perm[i], j);
    return Q;
}

enum class Mode { Uniform, Zero, HighVal, Planted, PlantedDup };
const char* ModeName(Mode m) {
    switch (m) {
        case Mode::Uniform: return "uniform";
        case Mode::Zero: return "zero";
        case Mode::HighVal: return "highval";
        case Mode::Planted: return "planted";
        default: return "planted_dup";
    }
}

}  // namespace

uint32_t OrbifoldMod3kValidation() {
    g_checks = g_fail = 0;
    const auto t0 = Clock::now();

    // ---------------- (L) the rung against the references and the engine ----------------
    uint32_t l_mats = 0, l_planted = 0, l_tors = 0, l_nonunit = 0, l_variants = 0;
    {
        std::mt19937_64 rng(0x5a3c0de3ull);
        struct Shape { uint32_t m, n; };
        const std::vector<Shape> shapes = {{1, 1}, {1, 9}, {9, 1}, {3, 70}, {70, 3}, {17, 17}, {63, 63}, {64, 64}, {65, 65},
                                           {40, 130}, {130, 40}, {97, 101}};
        const std::vector<uint32_t> ks = {1, 2, 3, 4, 5, 6, 7, 19};   // 5 / 6: the u16 / dense variant switch
        const std::vector<Mode> modes = {Mode::Uniform, Mode::Zero, Mode::HighVal, Mode::Planted, Mode::PlantedDup};
        const engine::ring3k::Params def = engine::ring3k::params();
        engine::ring3k::Params tiny;
        tiny.panel = 5;
        tiny.reduce_block = 3;
        for (int pi = 0; pi < 2; ++pi) {
            engine::ring3k::set_params(pi == 0 ? def : tiny);
            const std::string pn = pi == 0 ? "default" : "tiny";
            for (const Shape& sh : shapes)
                for (uint32_t k : ks)
                    for (Mode mode : modes) {
                        // keep the run short: the large shapes only at k in {2, 19} and in two modes on the tiny panel
                        const bool large = sh.m * sh.n > 4000;
                        if (large && k != 2 && k != 19 && k != 3) continue;
                        if (large && pi == 1 && mode != Mode::Planted && mode != Mode::HighVal) continue;
                        const ModCtx<U32> R = Ctx(k);
                        const U32 M = Pow3(k);
                        DenseBuf<U32> A(sh.m, sh.n);
                        std::vector<uint32_t> planted;
                        if (mode == Mode::Uniform) {
                            for (uint32_t i = 0; i < sh.m; ++i)
                                for (uint32_t j = 0; j < sh.n; ++j) A.at(i, j) = static_cast<U32>(rng() % M);
                        } else if (mode == Mode::HighVal) {
                            for (uint32_t i = 0; i < sh.m; ++i)
                                for (uint32_t j = 0; j < sh.n; ++j) {
                                    const uint32_t v = 1 + static_cast<uint32_t>(rng() % (k + 1));   // 1 .. k + 1 (>= k: zero)
                                    A.at(i, j) = v >= k ? 0u : R.mul(Pow3(v), static_cast<U32>(rng() % M));
                                }
                        } else if (mode == Mode::Planted || mode == Mode::PlantedDup) {
                            const uint32_t r = std::min(sh.m, sh.n);
                            DenseBuf<U32> D(sh.m, sh.n);
                            for (uint32_t i = 0; i < r; ++i) {
                                // sigma in 0 .. k (k = an invisible zero); force 0, k - 1 and k to occur when r allows
                                uint32_t s = static_cast<uint32_t>(rng() % (k + 1));
                                if (i == 0) s = 0;
                                if (i == 1) s = k - 1;
                                if (i == 2) s = k;
                                if (s < k) { D.at(i, i) = Pow3(s); planted.push_back(s); }
                            }
                            std::sort(planted.begin(), planted.end());
                            A = Mul(R, Mul(R, Unimodular(R, sh.m, rng), D), Unimodular(R, sh.n, rng));
                            if (mode == Mode::PlantedDup && sh.m >= 2)
                                for (uint32_t j = 0; j < sh.n; ++j) A.at(sh.m - 1, j) = A.at(0, j);
                        }
                        ++l_mats;
                        const std::string tag = std::string("(L) ") + ModeName(mode) + " " + str(sh.m) + "x" + str(sh.n) + " k=" + str(k) + " panel=" + pn;
                        const engine::ring3k::SmithResult S = engine::ring3k::smith(A, k);
                        const engine::ring3k::SmithResult Sn = engine::ring3k::smith_naive(A, k);
                        check(S.sigma == Sn.sigma, tag + ": smith " + jl(S.sigma) + " vs smith_naive " + jl(Sn.sigma));
                        uint32_t lr = 0;
                        for (uint32_t x : S.level_ranks) lr += x;
                        check(lr == S.sigma.size() && std::is_sorted(S.sigma.begin(), S.sigma.end()), tag + ": level ranks sum to |sigma|, sigma ascending");
                        if (mode == Mode::Planted) {
                            ++l_planted;
                            check(S.sigma == planted, tag + ": smith " + jl(S.sigma) + " vs the planted " + jl(planted));
                        }
                        for (uint32_t s : S.sigma) l_tors += s > 0;
                        const engine::ring3k::HowellForm H = engine::ring3k::howell(A, k);
                        const engine::ring3k::HowellForm Hn = engine::ring3k::howell_naive(A, k);
                        check(SameHowell3(H, Hn), tag + ": howell = howell_naive");
                        check(H.length() == S.length(), tag + ": length(howell) = length(smith)");
                        for (uint32_t v : H.pivot_v) l_nonunit += v > 0;
                        // both variants of each entry point at every k (the production one is picked by k: Ring3k.h)
                        check(engine::ring3k::smith_levels(A, k).sigma == S.sigma, tag + ": smith_levels = smith");
                        check(SameHowell3(engine::ring3k::howell_blocked(A, k), H), tag + ": howell_blocked = howell");
                        if (k <= 5) {
                            ++l_variants;
                            check(engine::ring3k::smith_lengths(A, k).sigma == S.sigma, tag + ": smith_lengths = smith");
                        }
                        const engine::ring3k::HowellForm Kr = engine::ring3k::kernel_module(A, k);
                        check(Kr.length() + S.length() == sh.n * k, tag + ": length(ker) + length(rowspan) = n k");
                        if (k >= 2) {
                            const std::vector<uint32_t> e = engine::elim::invariant_factors(R, engine::cview(A));
                            std::vector<uint32_t> from_s;
                            for (uint32_t s : S.sigma) if (s > 0) from_s.push_back(k - s);
                            std::vector<uint32_t> from_e;
                            for (uint32_t x : e) if (x < k) from_e.push_back(x);
                            std::sort(from_s.rbegin(), from_s.rend());
                            check(from_s == from_e, tag + ": {k - sigma : sigma >= 1} = engine invariant_factors below k");
                            check(SameHowellE(H, engine::elim::howell(R, engine::cview(A))), tag + ": howell = elim::howell entrywise");
                            check(SameHowellE(Kr, engine::elim::kernel_module(R, engine::cview(A))), tag + ": kernel_module = elim::kernel_module entrywise");
                        }
                    }
        }
        engine::ring3k::set_params(def);
        check(l_planted > 0 && l_tors > 0 && l_nonunit > 0 && l_variants > 0, "(L) non-vacuous: planted Smith forms, torsion, non-unit Howell pivots and both variants were compared");
        std::cout << "  RING3K_RUNG matrices=" << l_mats << " small_k_variants=" << l_variants << " planted=" << l_planted << " torsion=" << l_tors << " nonunit_pivots=" << l_nonunit << std::endl;
    }

    // ---------------- (M) the readings on the real coboundaries, route on = off ----------------
    const integral_model::IsotropyModel& Mdl = campaign::OrbifoldReferenceModel();
    const std::vector<uint32_t> mat_n = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 26, 27, 28, 36, 54, 80, 81, 82, 100, 128};
    uint32_t smith_cmp = 0, kernel_cmp = 0, nonzero_torsion = 0, kernel_rows = 0, nonunit_pivots = 0;
    double t_on = 0, t_off = 0;
    for (uint32_t n : mat_n) {
        for (uint32_t q = 1; q <= 3; ++q) {
            const uint32_t K = (q == 1) ? 19u : 2u;
            const DenseBuf<U32> D = Mdl.coboundary(integral_model::RingContext(3, K), n, q - 1);
            auto a = Clock::now();
            const std::vector<uint32_t> on = campaign::OrbifoldSmithTorsion(3, K, D, true);
            auto b = Clock::now();
            const std::vector<uint32_t> off = campaign::OrbifoldSmithTorsion(3, K, D, false);
            auto c = Clock::now();
            t_on += secs(a, b);
            t_off += secs(b, c);
            ++smith_cmp;
            if (!on.empty()) ++nonzero_torsion;
            check(on == off, "(M) Smith torsion of delta^" + str(q - 1) + " over Z/3^" + str(K) + " at n = " + str(n) + ": ring3k " + jl(on) +
                                 " vs engine " + jl(off));
        }
        for (uint32_t q = 1; q <= 2; ++q) {
            const DenseBuf<U32> D = Mdl.coboundary(integral_model::RingContext(3, 2), n, q);
            auto a = Clock::now();
            const campaign::OrbifoldHowell on = campaign::OrbifoldKernelModule(3, 2, D, true);
            auto b = Clock::now();
            const campaign::OrbifoldHowell off = campaign::OrbifoldKernelModule(3, 2, D, false);
            auto c = Clock::now();
            t_on += secs(a, b);
            t_off += secs(b, c);
            ++kernel_cmp;
            kernel_rows += on.H.rows;
            for (uint32_t v : on.pivot_v) nonunit_pivots += v > 0;
            check(SameHowell(on, off), "(M) Howell kernel module of delta^" + str(q) + " over Z/9 at n = " + str(n) + " (" + str(on.H.rows) + " x " +
                                           str(on.H.cols) + " vs " + str(off.H.rows) + " x " + str(off.H.cols) + ")");
        }
    }
    check(nonzero_torsion > 0 && nonunit_pivots > 0, "(M) non-vacuous: some torsion and some non-unit Howell pivot were compared");
    std::cout << "  RING3K_ROUTE_MATRIX cells=" << mat_n.size() << " smith=" << smith_cmp << " kernels=" << kernel_cmp << " nonzero_torsion=" << nonzero_torsion
              << " kernel_rows=" << kernel_rows << " nonunit_pivots=" << nonunit_pivots << " ring3k_s=" << t_on << " engine_s=" << t_off << std::endl;

    // ---------------- (R) whole records, route on vs off ----------------
    const campaign::CampaignSpec spec = Spec("orbifold = c_i\nn_sym = 0..64\nell = 2,3\n");
    const std::vector<uint32_t> rec_n = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 18, 24, 26, 27, 28, 36, 54, 80, 100, 128};
    uint32_t rec_cells = 0, masked_sites = 0, ok_cells = 0, tors_cells = 0;
    double r_on = 0, r_off = 0;
    for (uint32_t n : rec_n) {
        const campaign::OrbifoldMod3kCounters k0 = campaign::OrbifoldMod3kCalls();
        auto a = Clock::now();
        campaign::OrbifoldCell on, off;
        {
            campaign::ScopedOrbifoldMod3k s(true);
            on = campaign::ComputeOrbifoldCell(spec, 3, n);
        }
        auto b = Clock::now();
        const campaign::OrbifoldMod3kCounters k1 = campaign::OrbifoldMod3kCalls();
        {
            campaign::ScopedOrbifoldMod3k s(false);
            off = campaign::ComputeOrbifoldCell(spec, 3, n);
        }
        auto c = Clock::now();
        const campaign::OrbifoldMod3kCounters k2 = campaign::OrbifoldMod3kCalls();
        r_on += secs(a, b);
        r_off += secs(b, c);
        ++rec_cells;
        check(k1.ring3k_smith - k0.ring3k_smith == 3 && k1.ring3k_kernel - k0.ring3k_kernel == 2 && k1.engine_smith == k0.engine_smith &&
                  k1.engine_kernel == k0.engine_kernel,
              "(R) route on at (3, " + str(n) + "): 3 ring3k Smith + 2 ring3k kernel calls, no engine call");
        check(k2.engine_smith - k1.engine_smith == 3 && k2.engine_kernel - k1.engine_kernel == 2 && k2.ring3k_smith == k1.ring3k_smith &&
                  k2.ring3k_kernel == k1.ring3k_kernel,
              "(R) route off at (3, " + str(n) + "): 3 engine Smith + 2 engine kernel calls, no ring3k call");
        uint32_t m_on = 0, m_off = 0;
        const std::string b_on = campaign::OrbifoldCellBody(spec, on), b_off = campaign::OrbifoldCellBody(spec, off);
        const std::string mb_on = MaskBody(b_on, &m_on), mb_off = MaskBody(b_off, &m_off);
        // (L23-H1 audit F3) orbifold_cell/v2: no wall time may come back into the body (the identity "mask" alone would not notice)
        check(b_on.find(" s)\"") == std::string::npos && b_off.find(" s)\"") == std::string::npos && b_on.find("\"schema\":\"orbifold_cell/v2\"") == 0,
              "(R) v2 record body carries no wall time" + std::string(" (") + std::to_string(m_on) + " oracle gate site)");
        masked_sites += m_on;
        check(m_on == m_off && m_on <= 1, "(R) at (3, " + str(n) + "): the masked sites agree (" + str(m_on) + " / " + str(m_off) + ", at most the one H-1 site)");
        check(mb_on == mb_off, "(R) record body at (3, " + str(n) + ") byte-identical route on vs off (orbifold_cell/v2: no mask)");
        check(engine::FnvBytes(engine::kFnvOffset64, mb_on) == engine::FnvBytes(engine::kFnvOffset64, mb_off),
              "(R) masked record_fnv1a64 at (3, " + str(n) + ")");
        if (m_on == 0) check(b_on == b_off, "(R) unmasked body at (3, " + str(n) + ") byte-identical (no sum-rule site)");
        check(MaskLog(on.log) == MaskLog(off.log), "(R) log at (3, " + str(n) + ") identical after masking the decimal fractions");
        check(on.status == off.status && on.ci_q1_exponents == off.ci_q1_exponents && on.ladic_type[2] == off.ladic_type[2] &&
                  on.ladic_type[3] == off.ladic_type[3] && on.sub_dim[1] == off.sub_dim[1] && on.sub_dim[2] == off.sub_dim[2],
              "(R) the (Z) fields at (3, " + str(n) + ")");
        check(on.rung == off.rung && on.K == off.K && on.m == off.m && on.e_A == off.e_A && on.e_1 == off.e_1 && on.reruns == off.reruns &&
                  on.sumrule_K == off.sumrule_K && on.sumrule_max_invariant == off.sumrule_max_invariant,
              "(R) reader_path / sumrule_path at (3, " + str(n) + ") (values outside the hash; Z2K audit M1)");
        ok_cells += on.status == "ok";
        tors_cells += !on.ci_q1_exponents.empty() || !on.ladic_type[2].empty() || !on.ladic_type[3].empty();
    }
    check(ok_cells == rec_cells, "(R) every compared cell has status ok (" + str(ok_cells) + " / " + str(rec_cells) + ")");
    check(masked_sites > 0 && tors_cells > 0, "(R) non-vacuous: the H-1 mask was exercised and some cell carries (Z) torsion");
    {   // ell = 2 never reaches ring3k, whatever the switch says
        const campaign::OrbifoldMod3kCounters k0 = campaign::OrbifoldMod3kCalls();
        campaign::ScopedOrbifoldMod3k s(true);
        campaign::ScopedOrbifoldRing2k s2(false);
        const campaign::OrbifoldCell c2 = campaign::ComputeOrbifoldCell(spec, 2, 10);
        const campaign::OrbifoldMod3kCounters k1 = campaign::OrbifoldMod3kCalls();
        check(c2.status == "ok" && k1.ring3k_smith == k0.ring3k_smith && k1.ring3k_kernel == k0.ring3k_kernel && k1.engine_smith - k0.engine_smith == 3,
              "(R) ell = 2 with the ring3k route on (ring2k off): engine calls only");
    }
    std::cout << "  RING3K_ROUTE_RECORDS cells=" << rec_cells << " ok=" << ok_cells << " with_torsion=" << tors_cells << " masked_sites=" << masked_sites
              << " on_s=" << r_on << " off_s=" << r_off << std::endl;

    // ---------------- (C) negative controls and the switch ----------------
    {
        const std::string body = "{\"x\":[1,2],\"gates\":[{\"name\":\"sumrule_oracle_resolved\",\"kind\":\"G\",\"ok\":true,\"detail\":"
                                 "\"SUMRULE_ORACLE ell=3 n=10 dimS=0 K=40 max_invariant=0 -> resolved\"}]}";
        uint32_t m = 0;
        check(MaskBody(body, &m) == body && m == 1, "(C) v2: one oracle gate site, the body carries no wall time (nothing to mask)");
        std::string u = body;
        u[u.find("1,2")] = '3';
        check(MaskBody(body, nullptr) != MaskBody(u, nullptr), "(C) the masked comparison flags a one-character change of a value");
    }
    {   // K = 1 has no engine rollback (composite-only), so the ring3k route refuses it too; a non-canonical entry is refused
        const DenseBuf<U32> D = Mdl.coboundary(integral_model::RingContext(3, 2), 6, 0);
        bool threw_s = false, threw_k = false, threw_nc = false, threw_big = false;
        try { (void)campaign::OrbifoldSmithTorsion(3, 1, D, true); } catch (const std::logic_error&) { threw_s = true; }
        try { (void)campaign::OrbifoldKernelModule(3, 1, D, true); } catch (const std::logic_error&) { threw_k = true; }
        DenseBuf<U32> B(2, 2);
        B.at(0, 0) = 9;   // not canonical mod 9
        try { (void)engine::ring3k::smith(B, 2); } catch (const std::logic_error&) { threw_nc = true; }
        try { (void)engine::ring3k::smith(DenseBuf<U32>(1, 1), 20); } catch (const std::logic_error&) { threw_big = true; }
        check(threw_s && threw_k, "(C) the ring3k route refuses K = 1 / E = 1 (ENGINE_CHECK 2 <= K <= 19)");
        check(threw_nc && threw_big, "(C) ring3k refuses a non-canonical entry and k = 20 (3^20 > 2^31)");
    }
    {
        campaign::ScopedOrbifoldMod3k s(false);
        const std::string e = campaign::OrbifoldMod3kEffectiveLine();
        check(!campaign::OrbifoldMod3kEnabled() && e.rfind("EFFECTIVE l23_mod3k=off source=scoped", 0) == 0, "(C) scoped off: " + e);
    }
    {
        campaign::ScopedOrbifoldMod3k s(true);
        const std::string e = campaign::OrbifoldMod3kEffectiveLine();
        check(campaign::OrbifoldMod3kEnabled() && e.rfind("EFFECTIVE l23_mod3k=on source=scoped", 0) == 0, "(C) scoped on: " + e);
    }
    std::cout << "  " << campaign::OrbifoldMod3kEffectiveLine() << std::endl;

    std::cout << "ORBIFOLD_MOD3K rung_matrices=" << l_mats << " matrix_cells=" << mat_n.size() << " smith=" << smith_cmp << " kernels=" << kernel_cmp
              << " record_cells=" << rec_cells << " masked_sites=" << masked_sites << " checks=" << g_checks << " failing=" << g_fail
              << " s=" << secs(t0, Clock::now()) << " verdict=" << (g_fail ? "FAIL" : "OK") << std::endl;
    return g_fail;
}
