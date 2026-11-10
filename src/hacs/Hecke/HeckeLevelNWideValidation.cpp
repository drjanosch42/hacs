// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 17, WP-5) The width arms (W5b) / (W5n) of HeckeLevelNSparseValidation -- see the header.
// [C] sec:levelN (def:cosetrepsN, lem:cosetrepsN, prop:levelorbit); slot-16 WP-13 finding F-1.
#include "HeckeLevelNWideValidation.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "ExactInt.h"
#include "HeckeLevelEll.h"
#include "HeckeLevelN.h"
#include "HeckeLevelNSparse.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/ModCtx.h"
#include "../Util/Timing.h"

namespace {

using oc_util::Seconds;

using hecke::HeckeOverflow;
using hecke::Mat22;
using namespace hecke::level;

constexpr uint32_t kP = 1000003;

struct Checker {
    uint32_t checks = 0, fails = 0;
    void check(bool ok, const std::string& what) {
        ++checks;
        if (!ok) {
            ++fails;
            if (fails <= 40) std::cout << "  FAIL: " << what << "\n";
        }
    }
};

std::string Cell(uint64_t N, uint32_t n, uint32_t p) {
    return "(N=" + std::to_string(N) + ", n=" + std::to_string(n) + ", F_" + std::to_string(p) + ")";
}

bool SameVecs(const std::vector<SparseVec>& a, const std::vector<SparseVec>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i].col != b[i].col || a[i].val != b[i].val) return false;
    return true;
}

// A deterministic pseudo-random vector over F_p (splitmix64).
std::vector<uint32_t> Probe(size_t len, uint32_t p, uint64_t seed) {
    std::vector<uint32_t> v(len);
    uint64_t x = seed;
    for (uint32_t& e : v) {
        x += 0x9e3779b97f4a7c15ull;
        uint64_t z = x;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        e = static_cast<uint32_t>((z ^ (z >> 31)) % p);
    }
    return v;
}

// The S and ST generators of prop:levelorbit (1) at one width.
std::vector<SparseVec> Generators(const Gamma0LevelN& lvl, uint32_t n, const engine::ModCtx<uint32_t>& ctx) {
    std::vector<SparseVec> g = OrbitWiseFixedGenerators(lvl, Mat22::S(), 2, n, ctx);
    std::vector<SparseVec> u = OrbitWiseFixedGenerators(lvl, Mat22::ST(), 3, n, ctx);
    g.insert(g.end(), u.begin(), u.end());
    return g;
}

// (W5b) the sparse generators: forced-wide == shipped, and the shipped path never escalates.
void ArmGenerators(Checker& g, uint64_t N, uint32_t n, uint32_t p, std::vector<std::string>& lines) {
    const engine::ModCtx<uint32_t> ctx(p);
    const Gamma0LevelN lvl(N);
    const auto t0 = std::chrono::steady_clock::now();
    ResetLevelNWideProducts();
    const std::vector<SparseVec> auto_gens = Generators(lvl, n, ctx);
    const uint64_t esc = LevelNWideProducts();
    std::vector<SparseVec> wide_gens;
    uint64_t wide = 0;
    {
        ScopedLevelNWidth w(LevelNWidth::kForceWide);
        ResetLevelNWideProducts();
        wide_gens = Generators(lvl, n, ctx);
        wide = LevelNWideProducts();
    }
    uint64_t nnz = 0;
    for (const SparseVec& v : auto_gens) nnz += v.nnz();
    const std::string tag = "(W5b) " + Cell(N, n, p);
    g.check(esc == 0, tag + ": the shipped path takes the int64 tier on every product (no escalation below the old reach)");
    g.check(wide > 0, tag + ": the forced-wide path forms its products in __int128 (non-vacuous)");
    g.check(SameVecs(auto_gens, wide_gens), tag + ": the S / ST generators are entry for entry the same at both widths");
    std::ostringstream os;
    os << "LEVEL_N_WIDE (W5b) N=" << N << " n=" << n << " p=" << p << " m=" << lvl.m() << " generators=" << auto_gens.size()
       << " generator_nnz=" << nnz << " auto_escalations=" << esc << " forced_wide_products=" << wide
       << " identical=" << (SameVecs(auto_gens, wide_gens) ? 1 : 0) << " wall_s=" << Seconds(t0);
    lines.push_back(os.str());
}

// (W5b) the dense routes at small N: MonomialOf, OrbitWiseFixedSpace.
void ArmDense(Checker& g, uint64_t N, uint32_t n, uint32_t p) {
    const engine::ModCtx<uint32_t> ctx(p);
    const Gamma0LevelN lvl(N);
    const std::string tag = "(W5b) dense " + Cell(N, n, p);
    for (const auto& go : {std::make_pair(Mat22::S(), 2u), std::make_pair(Mat22::ST(), 3u)}) {
        ResetLevelNWideProducts();
        const MonomialAction a = MonomialOf(lvl, go.first, n, ctx);
        const FixedSpace fa = OrbitWiseFixedSpace(lvl, go.first, go.second, n, ctx);
        const uint64_t esc = LevelNWideProducts();
        ScopedLevelNWidth w(LevelNWidth::kForceWide);
        const MonomialAction b = MonomialOf(lvl, go.first, n, ctx);
        const FixedSpace fb = OrbitWiseFixedSpace(lvl, go.first, go.second, n, ctx);
        const std::string o = go.second == 2 ? " S" : " ST";
        g.check(esc == 0, tag + o + ": no escalation on the shipped path");
        g.check(a.perm == b.perm && a.dense() == b.dense(), tag + o + ": MonomialOf equal at both widths (entrywise)");
        g.check(fa.span == fb.span && fa.orbits == fb.orbits && fa.free_orbits == fb.free_orbits,
                tag + o + ": OrbitWiseFixedSpace equal at both widths");
    }
}

// (W5b) LevelChangeMap, restriction and trace.
void ArmLevelChange(Checker& g, uint64_t M, uint64_t N, uint32_t n, uint32_t p, bool full_dense) {
    const engine::ModCtx<uint32_t> ctx(p);
    const Gamma0LevelN big(M), small(N);
    for (LevelChangeKind kind : {LevelChangeKind::kRestriction, LevelChangeKind::kTrace}) {
        const std::string tag = "(W5b) level change " + std::to_string(M) + " -> " + std::to_string(N) + " n=" + std::to_string(n) +
                                (kind == LevelChangeKind::kRestriction ? " res" : " trace");
        ResetLevelNWideProducts();
        const LevelChangeMap a(big, small, kind, n, ctx);
        const uint64_t esc = LevelNWideProducts();
        ScopedLevelNWidth w(LevelNWidth::kForceWide);
        ResetLevelNWideProducts();
        const LevelChangeMap b(big, small, kind, n, ctx);
        const uint64_t wide = LevelNWideProducts();
        g.check(esc == 0 && wide > 0, tag + ": the shipped path never escalates, the forced one always does");
        bool same = a.fibre() == b.fibre();
        for (uint64_t s = 1; same && s <= 3; ++s) {
            const std::vector<uint32_t> x = Probe(a.src_dim(), p, 0x5eed0000 + s);
            same = a.apply(x) == b.apply(x);
        }
        if (full_dense) same = same && a.dense() == b.dense();
        g.check(same, tag + ": equal at both widths (" + std::string(full_dense ? "dense matrix" : "fibre map and three probes") + ")");
    }
}

// (W5n) the synthetic product beyond int64.
void ArmSynthetic(Checker& g) {
    const FullLevel full;
    // T^a L T^{-a} = [[1 + a, -a^2], [1, 1 - a]] at a = 2^33: entries ~2^66 (beyond int64, its determinant's products
    // ~2^67 well inside __int128).
    const int64_t a = int64_t{1} << 33;
    const Mat22 X{1, a, 0, 1}, Y{1, 0, 1, 1}, Z{1, -a, 0, 1};
    for (bool invert : {false, true}) {
        const std::string tag = std::string("(W5n) synthetic T^(2^33) L T^(-2^33)") + (invert ? " inverted" : "");
        bool narrow_refused = false;
        try {
            ScopedLevelNWidth w(LevelNWidth::kForceNarrow);
            (void)SubgroupProductForRho(full, X, Y, Z, kP, invert, "synthetic: not in SL_2(Z)");
        } catch (const HeckeOverflow&) {
            narrow_refused = true;
        }
        g.check(narrow_refused, tag + ": the old int64 width refuses loudly (HeckeOverflow)");
        ResetLevelNWideProducts();
        const Mat22 r = SubgroupProductForRho(full, X, Y, Z, kP, invert, "synthetic: not in SL_2(Z)");
        // independent: X Y Z mod P from the residues of the factors.
        auto mm = [](const std::array<uint64_t, 4>& u, const std::array<uint64_t, 4>& v) {
            return std::array<uint64_t, 4>{(u[0] * v[0] + u[1] * v[2]) % kP, (u[0] * v[1] + u[1] * v[3]) % kP,
                                           (u[2] * v[0] + u[3] * v[2]) % kP, (u[2] * v[1] + u[3] * v[3]) % kP};
        };
        auto res = [](const Mat22& mat) {
            const std::array<uint32_t, 4> t = mat.reduced_mod(kP);
            return std::array<uint64_t, 4>{t[0], t[1], t[2], t[3]};
        };
        std::array<uint64_t, 4> h = mm(mm(res(X), res(Y)), res(Z));
        if (invert) h = {h[3], (kP - h[1]) % kP, (kP - h[2]) % kP, h[0]};
        const std::array<uint32_t, 4> got = r.reduced_mod(kP);
        g.check(LevelNWideProducts() == 1 && got[0] == h[0] && got[1] == h[1] && got[2] == h[2] && got[3] == h[3],
                tag + ": the shipped path escalates once and its residues equal an independent computation mod P");
    }
}

// (W5n) the real cell beyond the old reach: Gamma_0(60060), n = 0.
void ArmBeyond(Checker& g, std::vector<std::string>& lines) {
    const uint64_t N = 60060;
    const engine::ModCtx<uint32_t> ctx(kP);
    const auto t0 = std::chrono::steady_clock::now();
    const Gamma0LevelN lvl(N);
    const std::string tag = "(W5n) " + Cell(N, 0, kP);
    bool narrow_refused = false;
    try {
        ScopedLevelNWidth w(LevelNWidth::kForceNarrow);
        (void)Generators(lvl, 0, ctx);
    } catch (const HeckeOverflow&) {
        narrow_refused = true;
    }
    g.check(narrow_refused, tag + ": the old int64 width throws HeckeOverflow before the presentation exists (F-1 reproduced)");
    ResetLevelNWideProducts();
    const std::vector<SparseVec> gS = OrbitWiseFixedGenerators(lvl, Mat22::S(), 2, 0, ctx);
    const std::vector<SparseVec> gU = OrbitWiseFixedGenerators(lvl, Mat22::ST(), 3, 0, ctx);
    const uint64_t esc = LevelNWideProducts();
    g.check(esc > 0, tag + ": the shipped path completes on the __int128 tier (" + std::to_string(esc) + " escalations)");
    // prop:levelorbit (3) at n = 0 (one generator per orbit, the fixed points included since 2, 3 are units):
    // |gens(S)| = (m - nu_2)/2 + nu_2, |gens(ST)| = (m - nu_3)/3 + nu_3; each family partitions the m columns.
    const uint64_t m = lvl.m(), nu2 = Nu2(N), nu3 = Nu3(N);
    g.check(gS.size() == (m - nu2) / 2 + nu2 && gU.size() == (m - nu3) / 3 + nu3,
            tag + ": the generator counts are the orbit counts of prop:levelorbit (3)");
    for (const auto* fam : {&gS, &gU}) {
        std::vector<uint8_t> hit(m, 0);
        bool part = true;
        for (const SparseVec& v : *fam)
            for (size_t k = 0; k < v.col.size(); ++k) {
                if (v.col[k] >= m || hit[v.col[k]] || v.val[k] == 0) part = false;
                else hit[v.col[k]] = 1;
            }
        for (uint8_t h : hit) part = part && h;
        g.check(part, tag + std::string(fam == &gS ? " S" : " ST") + ": the orbit supports partition the m columns");
    }
    std::ostringstream os;
    os << "LEVEL_N_WIDE (W5n) N=" << N << " n=0 p=" << kP << " m=" << m << " narrow_refused=" << (narrow_refused ? 1 : 0)
       << " escalations=" << esc << " gens_S=" << gS.size() << " gens_ST=" << gU.size() << " wall_s=" << Seconds(t0);
    lines.push_back(os.str());
}

} // namespace

uint32_t HeckeLevelNWideArms(uint32_t& checks) {
    const auto t0 = std::chrono::steady_clock::now();
    std::cout << "  -- WP-5 width arms (slot-17, F-1: (W5b) bit-identity forced __int128 = shipped on N <= 30030, "
                 "(W5n) the old width refuses beyond 5.5e4)" << std::endl;
    Checker g;
    std::vector<std::string> lines;
    // (W5b) the slot-16 ladder's levels (N <= 30030) and small cells.
    const std::vector<std::array<uint64_t, 3>> gen_cells{{30030, 0, kP}, {15015, 2, kP}, {4620, 4, kP}, {2310, 10, kP},
                                                         {210, 4, kP},  {30, 2, 7},      {60, 0, 5},    {1000, 2, 11}};
    for (const auto& c : gen_cells) ArmGenerators(g, c[0], static_cast<uint32_t>(c[1]), static_cast<uint32_t>(c[2]), lines);
    for (const auto& c : std::vector<std::array<uint64_t, 3>>{{30, 2, kP}, {210, 0, kP}, {60, 4, 7}, {2310, 0, kP}})
        ArmDense(g, c[0], static_cast<uint32_t>(c[1]), static_cast<uint32_t>(c[2]));
    ArmLevelChange(g, 6, 30, 2, kP, true);
    ArmLevelChange(g, 2310, 4620, 2, kP, false);
    ArmLevelChange(g, 15015, 30030, 0, kP, false);
    // (W5n) the negative control.
    ArmSynthetic(g);
    ArmBeyond(g, lines);
    for (const std::string& l : lines) std::cout << "  " << l << "\n";
    std::cout << "  [WP-5 width arms: " << g.checks << " checks, " << g.fails << " failing, " << Seconds(t0) << " s]" << std::endl;
    checks += g.checks;
    return g.fails;
}
