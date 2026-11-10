// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 22, P6) HeckeLValueShapiroValidation -- see HeckeLValueShapiroValidation.h.  The frozen tables below are
// docs/implementation/slot22/p6/P6_oracle.out verbatim (the predictions, committed before this file was first run).
#include "HeckeLValueShapiroValidation.h"

#include "BirchModularSymbol.h"
#include "Gamma0NShapiro.h"
#include "LValueRoute.h"
#include "ModularSymbolMeasure.h"

#include "../HeckeLambdaTwoVar.h"   // P9: Gamma0Weight2Symbol, AtCusp, MultiplicativeMeasure (slot 21, own code)

#include "../../Hecke/HeckeSettings.h"   // hecke::NotImplementedInThisBuild
#include "../../../ModArithmetic/Fnv.h"
#include "../../Util/CellList.h"

#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace hecke_ext::lvalue;
using namespace hecke_ext::lvalue::shapiro;
namespace TV = hecke_ext::lambda::twovar;

namespace {

double Now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
double PeakRssGb() {
    struct rusage ru {};
    getrusage(RUSAGE_SELF, &ru);
    return static_cast<double>(ru.ru_maxrss) / (1024.0 * 1024.0);
}

struct Log {
    uint32_t checks = 0, failing = 0;
    void check(const std::string& arm, bool ok, const std::string& what) {
        ++checks;
        if (!ok) {
            ++failing;
            std::cout << "  FAIL (" << arm << ") " << what << std::endl;
        }
    }
};

uint64_t Pow(uint64_t b, uint32_t e) {
    uint64_t r = 1;
    for (uint32_t i = 0; i < e; ++i) r *= b;
    return r;
}
std::string Hex(uint64_t h) {
    std::ostringstream o;
    o << "0x" << std::hex << std::setw(16) << std::setfill('0') << h;
    return o.str();
}
uint64_t FnvVec(const Vec& v) {
    uint64_t h = engine::kFnvOffset64;
    for (uint64_t x : v) h = engine::FnvU64(h, x);
    return h;
}
template<class F>
bool ThrowsNI(F f) {
    try {
        f();
    } catch (const hecke::NotImplementedInThisBuild&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}
template<class F>
bool ThrowsLogicNotNI(F f) {
    try {
        f();
    } catch (const hecke::NotImplementedInThisBuild&) {
        return false;
    } catch (const std::logic_error&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}
std::vector<uint32_t> CharList(uint64_t ell) {
    std::set<uint32_t> s{0u, 1u, 2u, static_cast<uint32_t>((ell - 1) / 2), static_cast<uint32_t>(ell - 2)};
    return std::vector<uint32_t>(s.begin(), s.end());
}
// the normalised generator of a rank-1 free module (else empty): shapiro::RankOneGenerator (Gamma0NShapiro.h; moved there
// in slot 23, G-EVT23, so the golden harness's SHAPIRO points share it)

// ---- the oracle's tables (P6_oracle.out) -------------------------------------------------------------------------------------
struct DRow { uint64_t N; uint32_t n, k, s, c; int w13m, w13p, w13, w7m, w7p, w7; };
const DRow kD[] = {
    {2, 0, 2, 0, 2, 0, 1, 1, 0, 1, 1},     {2, 2, 4, 0, 2, 0, 2, 2, 0, 2, 2},       {2, 4, 6, 0, 2, 0, 2, 2, 0, 2, 2},
    {2, 6, 8, 1, 2, 1, 3, 4, 1, 3, 4},     {2, 8, 10, 1, 2, 1, 3, 4, 1, 3, 4},      {2, 10, 12, 2, 2, 2, 4, 6, 2, 4, 6},
    {2, 12, 14, 2, 2, 2, 4, 6, 3, 4, 7},   {3, 0, 2, 0, 2, 0, 1, 1, 0, 1, 1},       {3, 2, 4, 0, 2, 0, 2, 2, 0, 2, 2},
    {3, 4, 6, 1, 2, 1, 3, 4, 1, 3, 4},     {3, 6, 8, 1, 2, 1, 3, 4, 1, 3, 4},       {3, 8, 10, 2, 2, 2, 4, 6, 2, 4, 6},
    {3, 10, 12, 3, 2, 3, 5, 8, 3, 5, 8},   {3, 12, 14, 3, 2, 3, 5, 8, 4, 5, 9},     {5, 0, 2, 0, 2, 0, 1, 1, 0, 1, 1},
    {5, 2, 4, 1, 2, 1, 3, 4, 1, 3, 4},     {5, 4, 6, 1, 2, 1, 3, 4, 1, 3, 4},       {5, 6, 8, 3, 2, 3, 5, 8, 3, 5, 8},
    {5, 8, 10, 3, 2, 3, 5, 8, 3, 5, 8},    {5, 10, 12, 5, 2, 5, 7, 12, 5, 7, 12},   {5, 12, 14, 5, 2, 5, 7, 12, 6, 7, 13},
    {7, 0, 2, 0, 2, 0, 1, 1, -1, -1, -1},  {7, 2, 4, 1, 2, 1, 3, 4, -1, -1, -1},    {7, 4, 6, 3, 2, 3, 5, 8, -1, -1, -1},
    {7, 6, 8, 3, 2, 3, 5, 8, -1, -1, -1},  {7, 8, 10, 5, 2, 5, 7, 12, -1, -1, -1},  {7, 10, 12, 7, 2, 7, 9, 16, -1, -1, -1},
    {7, 12, 14, 7, 2, 7, 9, 16, -1, -1, -1}, {11, 0, 2, 1, 2, 1, 2, 3, 1, 2, 3},    {11, 2, 4, 2, 2, 2, 4, 6, 2, 4, 6},
    {11, 4, 6, 4, 2, 4, 6, 10, 4, 6, 10},  {11, 6, 8, 6, 2, 6, 8, 14, 6, 8, 14},    {11, 8, 10, 8, 2, 8, 10, 18, 8, 10, 18},
    {11, 10, 12, 10, 2, 10, 12, 22, 10, 12, 22}, {11, 12, 14, 12, 2, 12, 14, 26, 13, 14, 27},
    {4, 0, 2, 0, 3, 0, 2, 2, 0, 2, 2},     {4, 2, 4, 0, 3, 0, 3, 3, 0, 3, 3},       {4, 4, 6, 1, 3, 1, 4, 5, 1, 4, 5},
    {6, 0, 2, 0, 4, 0, 3, 3, 0, 3, 3},     {6, 2, 4, 1, 4, 1, 5, 6, 1, 5, 6},       {6, 4, 6, 3, 4, 3, 7, 10, 3, 7, 10},
};
struct HRow { uint64_t N; uint32_t n; uint64_t ell; uint32_t R; std::vector<std::pair<uint64_t, uint64_t>> ap; uint64_t fnv; };
const std::vector<HRow> kH = {
    {2, 6, 13, 3, {{3, 12}, {5, 1987}, {7, 1016}, {11, 1092}, {13, 1382}}, 0x74ea35cc0a9c5f5full},
    {3, 4, 13, 3, {{2, 2191}, {5, 6}, {7, 2157}, {11, 1633}, {13, 638}}, 0x3011a94e64e092d4ull},
    {5, 2, 13, 3, {{2, 2193}, {3, 2}, {7, 6}, {11, 32}, {13, 2159}}, 0x84d77af93ebb3e65ull},
    {7, 2, 13, 3, {{2, 2196}, {3, 2195}, {5, 16}, {11, 2189}, {13, 28}}, 0xbb3b2094152cb961ull},
    {11, 0, 13, 3, {{2, 2195}, {3, 2196}, {5, 1}, {7, 2195}, {13, 4}}, 0x15b53d1989c67ed5ull},
    {11, 0, 11, 3, {{2, 1329}, {3, 1330}, {5, 1}, {7, 1329}, {13, 4}}, 0xe13ed86a21fd74d1ull},
    {11, 0, 5, 3, {{2, 123}, {3, 124}, {5, 1}, {7, 123}, {13, 4}}, 0xf427814a941fc3c5ull},
    {11, 0, 7, 3, {{2, 341}, {3, 342}, {5, 1}, {7, 341}, {13, 4}}, 0x4790e4bbcc34da69ull},
};
struct P9Row { uint64_t ell; int sign; std::vector<uint64_t> psi; };
const std::vector<P9Row> kP9 = {
    {11, 1, {1, 0, 5, 668, 663, 1326, 1326, 663, 668, 5, 0, 1330}},  {11, -1, {0, 0, 0, 1, 1, 0, 0, 1330, 1330, 0, 0, 0}},
    {5, 1, {1, 0, 5, 65, 60, 120, 120, 60, 65, 5, 0, 124}},          {5, -1, {0, 0, 0, 1, 1, 0, 0, 124, 124, 0, 0, 0}},
    {7, 1, {1, 0, 5, 174, 169, 338, 338, 169, 174, 5, 0, 342}},      {7, -1, {0, 0, 0, 1, 1, 0, 0, 342, 342, 0, 0, 0}},
    {13, 1, {1, 0, 5, 1101, 1096, 2192, 2192, 1096, 1101, 5, 0, 2196}}, {13, -1, {0, 0, 0, 1, 1, 0, 0, 2196, 2196, 0, 0, 0}},
};
struct MRow { uint64_t N; uint32_t n; uint64_t ell; uint32_t R, M; uint64_t a_ell, alpha; uint32_t moments; uint64_t fnv; };
const MRow kM[] = {
    {11, 0, 5, 3, 3, 1, 96, 4, 0x5e5977f49428b378ull},      {11, 0, 7, 3, 3, 341, 222, 5, 0x4b4bb28e517b5669ull},
    {5, 2, 13, 3, 3, 2159, 2159, 15, 0xcf8fc1da2951b65dull}, {7, 2, 13, 3, 2, 28, 28, 15, 0x7f920eb72d235418ull},
    {3, 4, 13, 3, 2, 638, 638, 25, 0x613b173a5c94db80ull},   {2, 6, 13, 3, 2, 1382, 1382, 35, 0x2a2362bfcf9d46f9ull},
    {11, 0, 13, 3, 2, 4, 719, 5, 0x9ebc4157e7f3f4c1ull},
};
struct TWRow { uint64_t T; std::vector<uint32_t> orders; uint64_t fnv; };
const uint64_t kTwFnvT1 = 0xadfddff97c8088c8ull;
const std::vector<TWRow> kTW = {
    {2, {1}, 0xd68c3587e00c02cfull},    {3, {2}, 0x83b71be53e746dabull},    {5, {2, 4}, 0x0c069a536f809bfbull},
    {7, {2, 6}, 0xb3207730b6992719ull}, {11, {2}, 0x9d99f82db44d3e8full},
};
struct TWNRow { uint64_t N; uint32_t n; uint64_t T; uint32_t d; uint32_t moments, swap_failures; uint64_t fnv; };
const TWNRow kTWN[] = {
    {11, 0, 3, 2, 5, 0, 0xa1d20abb841900c1ull},
    {5, 2, 3, 2, 15, 0, 0x9c5680ede7bf0c0full},
    {2, 6, 5, 4, 35, 2, 0xab83db05a64fb7f9ull},
};

// ---- (S0) the route ------------------------------------------------------------------------------------------------------
void ArmRoute(Log& log) {
    Domain d;
    d.model = ModelId::Gamma0NShapiro;
    d.cohomology = "H^1_c";
    d.n_sym = 2;
    d.level = 5;
    d.ell = 13;
    d.m = 3;
    d.subquotient = "W(Ind V)";
    bool built = false;
    try {
        BirchModularSymbolRoute r(d, 1, 0);
        built = r.shapiro_cell().basis.size() == 4;
    } catch (const std::exception& e) {
        std::cout << "  route at Gamma0NShapiro threw: " << e.what() << std::endl;
    }
    log.check("S0", built, "the route accepts ModelId::Gamma0NShapiro at N = 5, n = 2 (the refusal lifted); W(Ind V) of rank 4");
    Domain mp = d, cm = d, s7 = d, odd = d, e3 = d;
    mp.model = ModelId::MoerdijkPronk;
    cm.model = ModelId::CharacterModel;
    s7.model = ModelId::SL2ZSevenVertex;
    odd.n_sym = 3;
    e3.ell = 3;
    log.check("S0", ThrowsNI([&] { BirchModularSymbolRoute x(mp, 1, 0); }) && ThrowsNI([&] { BirchModularSymbolRoute x(cm, 1, 0); }),
              "Moerdijk-Pronk and the character model are still refused by name (NotImplementedInThisBuild)");
    log.check("S0", ThrowsNI([&] { BirchModularSymbolRoute x(s7, 1, 0); }), "the 7-vertex model at level 5 is refused (level N is Gamma0NShapiro)");
    log.check("S0", ThrowsLogicNotNI([&] { BirchModularSymbolRoute x(d, 5, 0); }), "gcd(M, N) > 1 is refused (M = 5 at N = 5)");
    log.check("S0", ThrowsLogicNotNI([&] { BirchModularSymbolRoute x(odd, 1, 0); }) && ThrowsLogicNotNI([&] { BirchModularSymbolRoute x(e3, 1, 0); }),
              "odd n and ell < 5 are refused");
    const BirchModularSymbolRoute r(d, 3, 1);
    log.check("S0", ThrowsNI([&] { (void)r.index_statement(); }) && ThrowsNI([&] { (void)r.cell(); }),
              "index_statement() (the level-one (L1)) and cell() (the cone) refuse at level N");
    log.check("S0", r.normalisation().l_function.find("Gamma_0(5)") != std::string::npos, "the normalisation record names Gamma_0(5)");
    Domain d11 = d;
    d11.level = 11;
    d11.n_sym = 0;
    d11.ell = 11;
    const BirchModularSymbolRoute r11(d11, 1, 0);
    CohomologyClass x;
    x.coordinates.assign(r11.shapiro_cell().basis.size(), 0);
    x.coordinates[0] = 1;
    MeasureRequest req;
    log.check("S0", ThrowsNI([&] { (void)r11.measure(x, req); }), "measure() at ell | N (U_ell) refuses (NotImplementedInThisBuild)");
    std::cout << "SHAPIRO_ROUTE lifted=" << (built ? 1 : 0) << std::endl;
}

// ---- (S1) ranks ------------------------------------------------------------------------------------------------------------
void ArmRanks(Log& log) {
    uint32_t cells = 0, bad = 0, formula_bad = 0, jump = 0;
    for (const DRow& r : kD) {
        ++cells;
        const uint64_t s = DimCuspFormsGamma0(r.N, r.k), c = CuspsGamma0(r.N);
        const bool fok = s == r.s && c == r.c;
        if (!fok) ++formula_bad;
        log.check("S1", fok, "N=" + std::to_string(r.N) + " k=" + std::to_string(r.k) + ": Eichler-Shimura s_k / cusps = " + std::to_string(s) + " / " +
                                 std::to_string(c) + " (oracle " + std::to_string(r.s) + " / " + std::to_string(r.c) + ")");
        for (int which = 0; which < 2; ++which) {
            const uint64_t ell = which == 0 ? 13 : 7;
            const uint32_t m = which == 0 ? 3 : 1;
            const int em = which == 0 ? r.w13m : r.w7m, ep = which == 0 ? r.w13p : r.w7p, ea = which == 0 ? r.w13 : r.w7;
            if (em < 0) continue;
            const Gamma0NCell Cm = RunGamma0NCell(r.N, r.n, ell, m, -1), Cp = RunGamma0NCell(r.N, r.n, ell, m, +1), Ca = RunGamma0NCell(r.N, r.n, ell, m, 0);
            const bool ok = static_cast<int>(Cm.basis.size()) == em && static_cast<int>(Cp.basis.size()) == ep && static_cast<int>(Ca.basis.size()) == ea &&
                            Cm.free_module && Cp.free_module && Ca.free_module;
            if (!ok) ++bad;
            log.check("S1", ok, "N=" + std::to_string(r.N) + " n=" + std::to_string(r.n) + " ell=" + std::to_string(ell) + "^" + std::to_string(m) +
                                    ": W-/W+/W = " + std::to_string(Cm.basis.size()) + "/" + std::to_string(Cp.basis.size()) + "/" +
                                    std::to_string(Ca.basis.size()) + " (oracle " + std::to_string(em) + "/" + std::to_string(ep) + "/" + std::to_string(ea) +
                                    "), free " + std::to_string(Cm.free_module && Cp.free_module && Ca.free_module));
            if (which == 0) {
                const uint64_t tot = 2 * s + c - (r.k == 2 ? 1 : 0);
                const bool prime = r.N != 4 && r.N != 6;
                const bool es = static_cast<uint64_t>(Ca.basis.size()) == tot && static_cast<uint64_t>(Cm.basis.size()) == s &&
                                (!prime || static_cast<uint64_t>(Cp.basis.size()) == s + c - (r.k == 2 ? 1 : 0));
                log.check("S1", es, "N=" + std::to_string(r.N) + " n=" + std::to_string(r.n) + " ell=13: ranks == Eichler-Shimura (total 2s + c" +
                                        std::string(r.k == 2 ? " - 1" : "") + ")");
            } else if (em != static_cast<int>(s)) {
                ++jump;
            }
        }
    }
    log.check("S1", jump == 4, "the ell = 7, k = 14 jump (W^- = s_k + 1) at exactly the four predicted cells N = 2, 3, 5, 11: " + std::to_string(jump));
    // the reduced system's size at the largest cell (the block structure)
    const Gamma0NCell big = RunGamma0NCell(11, 12, 13, 3, 0);
    std::cout << "SHAPIRO_RANKS cells=" << cells << " mismatching=" << bad << " formula_mismatch=" << formula_bad << " ell7_k14_jumps=" << jump
              << " largest=(11,12) L=" << big.L << " s_orbits=" << big.s_orbits << " s_fixed=" << big.s_fixed << " u_orbits=" << big.u_orbits
              << " unknowns=" << big.unknowns << " (full " << big.L * 13 << ") rows=" << big.relation_rows << " secs=" << big.seconds << std::endl;
}

// ---- (S2) Shapiro consistency ----------------------------------------------------------------------------------------------
void ArmShapiro(Log& log) {
    const uint32_t n = 10;
    const uint64_t ell = 13;
    const uint32_t m = 3;
    const Gamma0NCell W1 = RunGamma0NCell(1, n, ell, m, 0);
    const Ring& R = W1.R;
    const SymRho rho(n, R);
    const P1N one(1);
    log.check("S2", W1.basis.size() == 3 && W1.free_module, "W_1 (Shapiro, N = 1) at n = 10 is free of rank 3 = 2 s_12 + 1");
    // HeckeTpN at N = 1 = the level-one HeckeTp
    uint32_t hb = 0;
    for (const Vec& b : W1.basis)
        for (uint64_t p : {2ull, 3ull, 5ull, 7ull, 13ull})
            if (HeckeTpN(rho, one, b, p) != birch::HeckeTp(rho, b, p)) ++hb;
    log.check("S2", hb == 0, "HeckeTpN at N = 1 == BirchModularSymbol's HeckeTp (p in {2,3,5,7,13}) at " + std::to_string(hb) + " mismatches");
    // W_1 = the span of the cone symbols Psi(H^1_c) (slot-19 G1), m = 1 and 3
    for (uint32_t mm : {1u, 3u}) {
        const Gamma0NCell S = RunGamma0NCell(1, n, ell, mm, 0);
        const birch::BirchConeCell cone = birch::RunBirchConeCell(n, ell, mm, birch::BirchCellOptions());
        std::vector<Vec> both = S.basis;
        both.insert(both.end(), cone.P.begin(), cone.P.end());
        const uint32_t ls = birch::SpanLength(S.R, S.basis, n + 1), lc = birch::SpanLength(S.R, cone.P, n + 1), lb = birch::SpanLength(S.R, both, n + 1);
        log.check("S2", ls == lc && lc == lb, "m=" + std::to_string(mm) + ": span W_1 (Shapiro) == span Psi(H^1_c) (cone): lengths " + std::to_string(ls) +
                                                   " / " + std::to_string(lc) + " / joint " + std::to_string(lb));
    }
    for (uint64_t N : {2ull, 3ull, 5ull, 7ull, 11ull}) {
        const P1N p1(N);
        const uint32_t L = p1.size();
        const Gamma0NCell WN = RunGamma0NCell(N, n, ell, m, 0);
        uint32_t inw = 0, cr = 0, tres = 0, tcores = 0;
        for (const Vec& b : W1.basis) {
            const Vec r = ResFromLevelOne(b, L);
            if (!InWN(rho, p1, r)) ++inw;
            Vec scaled = b;
            for (auto& v : scaled) v = R.mul(v, (N + 1) % R.q);
            if (CoresToLevelOne(R, r, L, n) != scaled) ++cr;
            for (uint64_t p : {2ull, 3ull, 5ull, 7ull}) {
                if (N % p == 0) continue;
                if (HeckeTpN(rho, p1, r, p) != ResFromLevelOne(birch::HeckeTp(rho, b, p), L)) ++tres;
            }
        }
        for (const Vec& y : WN.basis)
            for (uint64_t p : {2ull, 3ull, 5ull, 7ull}) {
                if (N % p == 0) continue;
                if (CoresToLevelOne(R, HeckeTpN(rho, p1, y, p), L, n) != birch::HeckeTp(rho, CoresToLevelOne(R, y, L, n), p)) ++tcores;
            }
        uint32_t wn_bad = 0;
        for (const Vec& y : WN.basis)
            if (!InWN(rho, p1, y)) ++wn_bad;
        log.check("S2", inw == 0 && wn_bad == 0, "N=" + std::to_string(N) + ": Res(W_1) and the generators of W_N satisfy Manin's relations (" +
                                                     std::to_string(inw) + ", " + std::to_string(wn_bad) + " failures)");
        log.check("S2", cr == 0, "N=" + std::to_string(N) + ": Cores Res = N + 1 at " + std::to_string(cr) + " failures");
        log.check("S2", tres == 0 && tcores == 0, "N=" + std::to_string(N) + ": T_p Res = Res T_p (" + std::to_string(tres) + "), Cores T_p = T_p Cores (" +
                                                      std::to_string(tcores) + ") failures");
        std::cout << "SHAPIRO_RESCORES N=" << N << " L=" << L << " rankW=" << WN.basis.size() << " res_in_W=" << (inw == 0) << " cores_res=" << (cr == 0)
                  << " hecke_res=" << tres << " hecke_cores=" << tcores << std::endl;
    }
}

// ---- (S3) newform lines ----------------------------------------------------------------------------------------------------
void ArmNewforms(Log& log) {
    for (const HRow& h : kH) {
        const Gamma0NCell C = RunGamma0NCell(h.N, h.n, h.ell, h.R, -1);
        const Vec P = RankOneGenerator(C);
        const std::string tag = "(" + std::to_string(h.N) + "," + std::to_string(h.n + 2) + ") ell=" + std::to_string(h.ell);
        if (P.empty()) {
            log.check("S3", false, tag + ": W^- is not free of rank 1 (rank " + std::to_string(C.basis.size()) + ")");
            continue;
        }
        const SymRho rho(h.n, C.R);
        const P1N p1(h.N);
        log.check("S3", FnvVec(P) == h.fnv, tag + ": fnv of the normalised symbol " + Hex(FnvVec(P)) + " (oracle " + Hex(h.fnv) + ")");
        std::ostringstream line;
        uint32_t bad = 0, eta_bad = 0;
        for (const auto& pa : h.ap) {
            uint64_t lam = 0;
            const bool eig = EigenvalueN(rho, p1, P, pa.first, lam);
            if (!eig || lam != pa.second) ++bad;
            if (h.N != 7) {
                const int64_t e = EtaProductCoefficient(h.N, pa.first);
                if (C.R.from_int(e) != lam) ++eta_bad;
            }
            line << " a_" << pa.first << "=" << lam;
        }
        log.check("S3", bad == 0, tag + ": a_p != the oracle at " + std::to_string(bad) + " of " + std::to_string(h.ap.size()));
        if (h.N != 7) log.check("S3", eta_bad == 0, tag + ": a_p != the eta product eta(z)^r eta(Nz)^r at " + std::to_string(eta_bad));
        std::cout << "SHAPIRO_NEWFORM N=" << h.N << " k=" << h.n + 2 << " ell=" << h.ell << " R=" << h.R << line.str() << " fnv=" << Hex(FnvVec(P)) << std::endl;
    }
}

// ---- (S4) 11a vs P9 ------------------------------------------------------------------------------------------------------
void ArmP9(Log& log) {
    const P1N p1(11);
    for (const P9Row& row : kP9) {
        const uint32_t R3 = 3;
        const Gamma0NCell C = RunGamma0NCell(11, 0, row.ell, R3, row.sign);
        const Ring& R = C.R;
        const std::string tag = "11a ell=" + std::to_string(row.ell) + " sign=" + (row.sign > 0 ? "+" : "-");
        Vec v;
        if (C.basis.size() == 1) {
            v = C.basis[0];
        } else {
            std::vector<Vec> g;
            std::vector<uint32_t> e;
            EigenSubmodule(C, 2, R.q - 2, g, e);
            if (g.size() == 1 && e[0] == R3) v = g[0];
        }
        if (v.empty() || !NormaliseFirstUnit(R, v)) {
            log.check("S4", false, tag + ": the T_2 = -2 line is not free of rank 1");
            continue;
        }
        Vec ours(12);
        for (int64_t c = 0; c < 11; ++c) ours[static_cast<size_t>(c)] = v[p1.idx(c, 1)];
        ours[11] = v[p1.idx(1, 0)];
        log.check("S4", ours == row.psi, tag + ": the normalised symbol in P9's order == the oracle");
        const TV::Gamma0Symbol sy = TV::Gamma0Weight2Symbol(11, row.sign, -2, row.ell, R3);
        bool prop = sy.free_rank1;
        uint64_t u = 0;
        if (prop) {
            size_t i0 = 12;
            for (size_t i = 0; i < 12; ++i)
                if (R.unit(sy.psi[i])) {
                    i0 = i;
                    break;
                }
            prop = i0 < 12 && R.unit(ours[i0]);
            if (prop) {
                u = R.mul(ours[i0], R.inv(sy.psi[i0]));
                for (size_t i = 0; i < 12; ++i)
                    if (ours[i] != R.mul(u, sy.psi[i])) prop = false;
            }
        }
        log.check("S4", prop, tag + ": == u * P9's Gamma0Weight2Symbol (free rank 1: " + std::to_string(sy.free_rank1) + ", u = " + std::to_string(u) + ")");
        if (!prop) continue;
        const SymRho rho(0, R);
        uint32_t cusp_bad = 0, cusp_tot = 0;
        for (int64_t M = 1; M <= 40; ++M)
            for (int64_t a = 0; a < M; ++a) {
                ++cusp_tot;
                if (PhiCuspN(rho, p1, v, a, M)[0] != R.neg(R.mul(u, sy.AtCusp(a, M)))) ++cusp_bad;
            }
        log.check("S4", cusp_bad == 0, tag + ": Phi({a/M, oo}) != -u AtCusp(a, M) at " + std::to_string(cusp_bad) + " / " + std::to_string(cusp_tot));
        if (row.ell == 11) {
            const auto mu = TV::MultiplicativeMeasure(sy, 11, 3, 1);
            uint32_t mb = 0, mt = 0;
            for (uint32_t mm = 0; mm <= 3; ++mm) {
                const int64_t Mm = static_cast<int64_t>(Pow(11, mm));
                for (int64_t a = 0; a < Mm; ++a) {
                    ++mt;
                    if (PhiCuspN(rho, p1, v, a, Mm)[0] != R.mul(u, mu[mm][static_cast<size_t>(a)])) ++mb;
                }
            }
            log.check("S4", mb == 0, tag + ": Phi({a/11^m, oo}) != u * P9's MultiplicativeMeasure (a_11 = 1) at " + std::to_string(mb) + " / " + std::to_string(mt));
        }
        std::cout << "SHAPIRO_P9 ell=" << row.ell << " sign=" << (row.sign > 0 ? "+" : "-") << " rankW=" << C.basis.size() << " p9_free=" << sy.free_rank1
                  << " unit=" << u << " cusps=" << cusp_tot - cusp_bad << "/" << cusp_tot << std::endl;
    }
}

// ---- (S5) level-N measures -------------------------------------------------------------------------------------------------
void ArmLevelNMeasure(Log& log) {
    for (const MRow& r : kM) {
        const Gamma0NCell C = RunGamma0NCell(r.N, r.n, r.ell, r.R, -1);
        const Vec P = RankOneGenerator(C);
        const std::string tag = "(" + std::to_string(r.N) + "," + std::to_string(r.n + 2) + ") ell=" + std::to_string(r.ell);
        if (P.empty()) {
            log.check("S5", false, tag + ": W^- not free of rank 1");
            continue;
        }
        const Ring& R = C.R;
        const SymRho rho(r.n, R);
        const P1N p1(r.N);
        uint64_t a = 0;
        const bool eig = EigenvalueN(rho, p1, P, r.ell, a);
        const uint64_t alpha = eig && R.unit(a) ? birch::UnitRoot(R, a, r.n + 2) : 0;
        log.check("S5", eig && a == r.a_ell && alpha == r.alpha, tag + ": a_ell = " + std::to_string(a) + ", alpha = " + std::to_string(alpha) + " (oracle " +
                                                                    std::to_string(r.a_ell) + ", " + std::to_string(r.alpha) + ")");
        if (!eig || alpha == 0) continue;
        const TameMeasure tm = BuildTameMeasure(rho, p1, P, 1, a, alpha, r.M, true);
        const uint32_t tt = tm.three_term_failures(), df = tm.distribution_failures();
        const MomentIdentityReading mi = TameMomentIdentity(rho, p1, P, tm, {1 % R.q}, CharList(r.ell));
        log.check("S5", tt == 0 && df == 0, tag + ": three-term " + std::to_string(tt) + ", distribution " + std::to_string(df) + " failures");
        log.check("S5", mi.failures == 0 && mi.total == r.moments, tag + ": moment identity " + std::to_string(mi.failures) + " / " + std::to_string(mi.total) +
                                                                       " (oracle 0 / " + std::to_string(r.moments) + ")");
        log.check("S5", tm.fnv_units(2) == r.fnv, tag + ": fnv mu(a + ell^2) " + Hex(tm.fnv_units(2)) + " (oracle " + Hex(r.fnv) + ")");
        // the route's type: BuildMeasureN == the tame builder at T = 1 on m >= 1, and its m = 0 ball distributes
        const ModularSymbolMeasure me = BuildMeasureN(rho, p1, P, a, alpha, r.M, true);
        bool same = me.fnv_units(2) == r.fnv;
        for (uint32_t mm = 1; mm <= r.M; ++mm) same = same && me.mu[mm] == tm.mu[mm];
        uint32_t d0 = 0;
        for (uint32_t mm = 0; mm < r.M; ++mm) d0 += birch::DistributionFailures(me, mm);
        log.check("S5", same && d0 == 0, tag + ": BuildMeasureN == the tame builder at T = 1; distribution from m = 0: " + std::to_string(d0) + " failures");
        std::cout << "SHAPIRO_MEAS N=" << r.N << " k=" << r.n + 2 << " ell=" << r.ell << " M=" << r.M << " a_ell=" << a << " alpha=" << alpha << " three_term=" << tt
                  << " distribution=" << df << " moments=" << mi.failures << "/" << mi.total << " fnv=" << Hex(tm.fnv_units(2)) << std::endl;
    }
    // the route at level N: evaluate_all_j and measure() through BirchModularSymbolRoute (11a at ell = 7)
    Domain d;
    d.model = ModelId::Gamma0NShapiro;
    d.cohomology = "H^1_c";
    d.n_sym = 0;
    d.level = 11;
    d.ell = 7;
    d.m = 3;
    const BirchModularSymbolRoute route(d, 1, 0), route3(d, 3, 0);
    const Gamma0NCell& nc = route.shapiro_cell();
    const Gamma0NCell Cm = RunGamma0NCell(11, 0, 7, 3, -1);
    Vec P = RankOneGenerator(Cm);
    const Ring& R = nc.R;
    const SymRho rho(0, R);
    const P1N p1(11);
    Vec coords;
    const bool solved = !P.empty() && birch::Solve(R, birch::Smith(R, birch::ColumnsToMat(nc.basis, static_cast<uint32_t>(P.size())), true), P, coords);
    log.check("S5", solved, "11a's minus symbol lies in the route's W(Ind V) (coordinates found)");
    if (!solved) return;
    CohomologyClass x;
    x.coordinates = coords;
    const bool same_sym = route.symbol(x) == P;
    const auto ev1 = route.evaluate_all_j({x}), ev3 = route3.evaluate_all_j({x});
    const bool ev_ok = ev1.size() == 1 && ev1[0] == EvAllJN(rho, p1, P, birch::TrivialCharacter(R)) && ev3.size() == 1 &&
                       ev3[0] == EvAllJN(rho, p1, P, birch::KroneckerCharacter(-3, R));
    MeasureRequest req;
    req.level = 3;
    const ModularSymbolMeasure rm = route.measure(x, req);
    uint64_t a7 = 0;
    EigenvalueN(rho, p1, P, 7, a7);
    const ModularSymbolMeasure dm = BuildMeasureN(rho, p1, P, a7, birch::UnitRoot(R, a7, 2), 3, true);
    log.check("S5", same_sym && ev_ok && rm.mu == dm.mu && rm.fnv_units(2) == 0x4b4bb28e517b5669ull,
              "the route at level 11: symbol(x) == P, evaluate_all_j == EvAllJN (chi = 1 and chi_{-3}), measure() == BuildMeasureN (fnv as the oracle)");
}

// ---- (S6) tame twists ------------------------------------------------------------------------------------------------------
void ArmTame(Log& log) {
    const uint32_t n = 10, R3 = 3, M = 2;
    const uint64_t ell = 13;
    const Gamma0NCell C = RunGamma0NCell(1, n, ell, R3, -1);
    const Vec P = RankOneGenerator(C);
    if (P.empty()) {
        log.check("S6", false, "Delta's minus symbol (N = 1) is not free of rank 1");
        return;
    }
    const Ring& R = C.R;
    const SymRho rho(n, R);
    const P1N one(1);
    uint64_t a = 0;
    const bool eig = EigenvalueN(rho, one, P, ell, a);
    log.check("S6", eig && a == R.from_int(-577738) && a == 73, "Delta: a_13 = tau(13) = -577738 == 73 mod 13^3 (found " + std::to_string(a) + ")");
    if (!eig) return;
    const uint64_t alpha = birch::UnitRoot(R, a, n + 2);
    const TameMeasure t1 = BuildTameMeasure(rho, one, P, 1, a, alpha, M, true);
    const ModularSymbolMeasure lv1 = birch::BuildMeasure(rho, P, a, alpha, M, true);
    bool same = t1.fnv_units(2) == kTwFnvT1;
    for (uint32_t mm = 1; mm <= M; ++mm) same = same && t1.mu[mm] == lv1.mu[mm];
    log.check("S6", same, "T = 1: the tame builder == the level-one BuildMeasure table by table; fnv " + Hex(t1.fnv_units(2)) + " (oracle " + Hex(kTwFnvT1) + ")");
    const std::vector<uint32_t> chars = CharList(ell);
    for (const TWRow& row : kTW) {
        const TameMeasure tm = BuildTameMeasure(rho, one, P, row.T, a, alpha, M, true);
        const uint32_t tt = tm.three_term_failures(), df = tm.distribution_failures();
        const std::string tag = "T=" + std::to_string(row.T);
        log.check("S6", tt == 0 && df == 0, tag + ": three-term " + std::to_string(tt) + ", distribution " + std::to_string(df) + " failures");
        log.check("S6", tm.fnv_units(2) == row.fnv, tag + ": fnv " + Hex(tm.fnv_units(2)) + " (oracle " + Hex(row.fnv) + ")");
        for (uint32_t d : row.orders) {
            const std::vector<uint64_t> chi = TameCharacter(row.T, d, 1, R);
            const MomentIdentityReading mi = TameMomentIdentity(rho, one, P, tm, chi, chars);
            log.check("S6", mi.failures == 0 && mi.total == 55, tag + " d=" + std::to_string(d) + ": moment identity " + std::to_string(mi.failures) + " / " +
                                                                    std::to_string(mi.total) + " (predicted 0 / 55)");
            std::cout << "SHAPIRO_TAME T=" << row.T << " d=" << d << " three_term=" << tt << " distribution=" << df << " moments=" << mi.failures << "/" << mi.total
                      << " fnv=" << Hex(tm.fnv_units(2)) << std::endl;
        }
    }
}

// (TWN, referee R1) the tame twist at level N > 1: the package's goal, gated on the combination (N > 1, T > 1)
void ArmTameLevelN(Log& log) {
    const uint64_t ell = 13;
    for (const TWNRow& r : kTWN) {
        const Gamma0NCell C = RunGamma0NCell(r.N, r.n, ell, 3, -1);
        const Vec P = RankOneGenerator(C);
        const std::string tag = "TWN (" + std::to_string(r.N) + "," + std::to_string(r.n + 2) + ") T=" + std::to_string(r.T) + " d=" + std::to_string(r.d);
        if (P.empty()) {
            log.check("S6", false, tag + ": W^- not free of rank 1");
            continue;
        }
        const Ring& R = C.R;
        const SymRho rho(r.n, R);
        const P1N p1(r.N);
        uint64_t a = 0;
        if (!EigenvalueN(rho, p1, P, ell, a) || !R.unit(a)) {
            log.check("S6", false, tag + ": no unit eigenvalue a_ell");
            continue;
        }
        const uint64_t alpha = birch::UnitRoot(R, a, r.n + 2);
        const TameMeasure tm = BuildTameMeasure(rho, p1, P, r.T, a, alpha, 2, true);
        const uint32_t tt = tm.three_term_failures(), df = tm.distribution_failures();
        const std::vector<uint64_t> chi = TameCharacter(r.T, r.d, 1, R);
        const MomentIdentityReading mi = TameMomentIdentity(rho, p1, P, tm, chi, CharList(ell));
        const MomentIdentityReading sw = TameMomentIdentity(rho, p1, P, tm, chi, CharList(ell), true);
        log.check("S6", tt == 0 && df == 0, tag + ": three-term " + std::to_string(tt) + ", distribution " + std::to_string(df) + " failures");
        log.check("S6", mi.failures == 0 && mi.total == r.moments, tag + ": moment identity " + std::to_string(mi.failures) + " / " + std::to_string(mi.total) +
                                                                      " (predicted 0 / " + std::to_string(r.moments) + ")");
        log.check("S6", tm.fnv_units(2) == r.fnv, tag + ": fnv " + Hex(tm.fnv_units(2)) + " (oracle " + Hex(r.fnv) + ")");
        log.check("S6", sw.failures == r.swap_failures, tag + ": chi <-> chibar control: " + std::to_string(sw.failures) + " failures (predicted " +
                                                            std::to_string(r.swap_failures) + (r.swap_failures ? ", caught" : ", invisible: the parity / chi(ell) rule") + ")");
        std::cout << "SHAPIRO_TAMEN N=" << r.N << " k=" << r.n + 2 << " T=" << r.T << " d=" << r.d << " three_term=" << tt << " distribution=" << df
                  << " moments=" << mi.failures << "/" << mi.total << " swap=" << sw.failures << " fnv=" << Hex(tm.fnv_units(2)) << std::endl;
    }
}

// ---- (S7) negative controls ------------------------------------------------------------------------------------------------
void ArmNegative(Log& log) {
    const Gamma0NCell wrong = RunGamma0NCellWrongCosets(5, 2, 13, 3, 0);
    log.check("S7", wrong.basis.size() != 4, "NEG the coset map without the permutation: rank " + std::to_string(wrong.basis.size()) + " != 4 (caught)");
    const Gamma0NCell C = RunGamma0NCell(2, 6, 13, 3, -1);
    const Vec P = RankOneGenerator(C);
    if (P.empty()) {
        log.check("S7", false, "the (2,8) line is missing");
        return;
    }
    const Ring& R = C.R;
    const SymRho rho(6, R);
    const P1N p2(2);
    Vec scaled = P;
    for (auto& v : scaled) v = R.mul(v, 12);
    log.check("S7", HeckeTpNWrongAdjoint(rho, p2, P, 3) != scaled, "NEG rho(alpha) for rho(adj alpha) in T_3 does not give a_3 = 12 (caught)");
    // the tame measure of Delta: no ell^{k-2}, chi <-> chibar in E, alpha (1 + ell)
    const Gamma0NCell D = RunGamma0NCell(1, 10, 13, 3, -1);
    const Vec PD = RankOneGenerator(D);
    const SymRho rD(10, D.R);
    const P1N one(1);
    uint64_t a = 0;
    if (PD.empty() || !EigenvalueN(rD, one, PD, 13, a)) {
        log.check("S7", false, "Delta's symbol is missing");
        return;
    }
    const uint64_t alpha = birch::UnitRoot(D.R, a, 12);
    const TameMeasure noln = BuildTameMeasure(rD, one, PD, 3, a, alpha, 2, false);
    log.check("S7", noln.distribution_failures() > 0, "NEG no ell^{k-2}: distribution failures " + std::to_string(noln.distribution_failures()) + " (caught)");
    const TameMeasure wr = BuildTameMeasure(rD, one, PD, 3, a, D.R.mul(alpha, 14), 2, true);
    log.check("S7", wr.distribution_failures() > 0, "NEG alpha (1 + ell): distribution failures " + std::to_string(wr.distribution_failures()) + " (caught)");
    for (const std::pair<uint64_t, uint32_t> td : {std::make_pair(5ull, 4u), std::make_pair(7ull, 6u), std::make_pair(19ull, 6u), std::make_pair(19ull, 3u)}) {
        const uint64_t T = td.first;
        const uint32_t d = td.second;
        const TameMeasure tm = BuildTameMeasure(rD, one, PD, T, a, alpha, 2, true);
        const std::vector<uint64_t> chi = TameCharacter(T, d, 1, D.R);
        const MomentIdentityReading mi = TameMomentIdentity(rD, one, PD, tm, chi, CharList(13), true);
        // chi(13) is real exactly when 13 == +-1 mod T in the kernel of chi^2: at T = 7, 13 == -1 and chi(-1) = +-1, so the exchange is
        // invisible there (a vacuous control, recorded in P6_report.md as a deviation from the spec's NEG row); T = 5: chi(3) = +-i.
        const bool real = D.R.mul(chi[13 % T], chi[13 % T]) == 1 % D.R.q;
        if (real)
            log.check("S7", mi.failures == 0, "NEG chi <-> chibar in E at T=" + std::to_string(T) + " d=" + std::to_string(d) +
                                                  ": vacuous (chi(13) = chi(-1) is real), the exchanged identity holds: " + std::to_string(mi.failures) + " / " +
                                                  std::to_string(mi.total));
        else
            log.check("S7", mi.failures > 0, "NEG chi <-> chibar in E at T=" + std::to_string(T) + " d=" + std::to_string(d) + ": " + std::to_string(mi.failures) +
                                                 " / " + std::to_string(mi.total) + " failures (caught)");
        std::cout << "SHAPIRO_NEG chi_swap T=" << T << " d=" << d << " chi_ell_real=" << real << " failures=" << mi.failures << "/" << mi.total << std::endl;
    }
}

// ---- (S8) read-only scale arm: HECKE_SHAPIRO_CELLS="N:n:ell:m,..." (ranks vs Eichler-Shimura, cost of the reduced system) ----
void ArmCells(Log& log) {
    const char* env = std::getenv("HECKE_SHAPIRO_CELLS");
    if (!env || !*env) return;
    std::vector<std::vector<uint64_t>> cells;
    for (const oc_util::cells::Token& t : oc_util::cells::Tokenize(env)) {
        const size_t pos = t.pos;
        const std::string& item = t.text;
        const std::vector<uint64_t>& f = t.fields;
        const bool ok = t.Fits(4, 4) && f[0] >= 1 && f[0] <= P1N::kMaxLevel && f[1] % 2 == 0 && f[1] <= 400 && f[2] >= 5 && f[3] >= 1 && f[3] <= 8;
        if (!ok) {
            log.check("S8", false, "HECKE_SHAPIRO_CELLS: cell " + std::to_string(pos) + " ('" + item + "') is malformed (N:n:ell:m, n even) -- REFUSED");
            continue;
        }
        cells.push_back(f);
    }
    std::cout << oc_util::cells::EffectiveLine("HeckeLValueShapiroValidation", "CELLS", cells.size()) << std::endl;
    for (const auto& f : cells) {
        const uint64_t N = f[0], ell = f[2];
        const uint32_t n = static_cast<uint32_t>(f[1]), m = static_cast<uint32_t>(f[3]);
        const double t = Now();
        const Gamma0NCell Cm = RunGamma0NCell(N, n, ell, m, -1), Cp = RunGamma0NCell(N, n, ell, m, +1);
        const double secs = Now() - t;
        const uint64_t s = DimCuspFormsGamma0(N, n + 2), c = CuspsGamma0(N);
        bool prime = N > 1;
        for (uint64_t d = 2; d * d <= N; ++d)
            if (N % d == 0) prime = false;
        const uint64_t ep = s + c - (n == 0 ? 1 : 0);
        const bool ok = Cm.free_module && Cp.free_module && Cm.basis.size() == s && (!prime || Cp.basis.size() == ep) &&
                        (Cm.basis.size() + Cp.basis.size() == 2 * s + c - (n == 0 ? 1 : 0));
        log.check("S8", ok || n + 1 >= ell, "CELL (" + std::to_string(N) + "," + std::to_string(n) + "," + std::to_string(ell) + "^" + std::to_string(m) +
                                                "): W-/W+ = " + std::to_string(Cm.basis.size()) + "/" + std::to_string(Cp.basis.size()) +
                                                " vs Eichler-Shimura " + std::to_string(s) + "/" + std::to_string(ep));
        std::cout << "SHAPIRO_CELL N=" << N << " n=" << n << " ell=" << ell << " m=" << m << " L=" << Cm.L << " W-=" << Cm.basis.size() << " W+=" << Cp.basis.size()
                  << " s_k=" << s << " cusps=" << c << " free=" << (Cm.free_module && Cp.free_module) << " unknowns=" << Cm.unknowns << " rows=" << Cm.relation_rows
                  << " secs=" << secs << " rss_gb=" << PeakRssGb() << (n + 1 >= ell ? " (n + 1 >= ell: the mod-ell jump zone, not gated)" : "") << std::endl;
    }
}

} // namespace

uint32_t HeckeLValueShapiroValidation() {
    const double t0 = Now();
    Log log;
    std::cout << "HeckeLValueShapiroValidation: the Birch route on Gamma_0(N) by Shapiro + the tame twists (slot-22 P6; P6_spec.md)" << std::endl;
    const std::vector<std::pair<const char*, std::function<void(Log&)>>> arms = {
        {"S0", ArmRoute}, {"S1", ArmRanks}, {"S2", ArmShapiro}, {"S3", ArmNewforms},
        {"S4", ArmP9}, {"S5", ArmLevelNMeasure}, {"S6", ArmTame}, {"S6N", ArmTameLevelN}, {"S7", ArmNegative}, {"S8", ArmCells},
    };
    for (const auto& arm : arms) {
        const double ta = Now();
        const uint32_t before = log.failing;
        arm.second(log);
        std::cout << "  arm " << arm.first << ": failing " << (log.failing - before) << ", secs " << std::fixed << std::setprecision(2) << (Now() - ta)
                  << std::defaultfloat << std::endl;
    }
    std::cout << "P6_ARMS checks=" << log.checks << " failing=" << log.failing << " secs=" << std::fixed << std::setprecision(1) << (Now() - t0)
              << " peak_rss_gb=" << std::setprecision(3) << PeakRssGb() << std::defaultfloat << std::endl;
    return log.failing;
}
