// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 18, WP-5) The arms (W18o) / (W18g) of HeckeLevelNSparseValidation -- see the header.
// [C] sec:levelN, rmk:levelNscale (4) (the generator order, a value-free choice), the code map's items (b), (d).
#include "HeckeLevelNOrderValidation.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "ExactInt.h"
#include "HeckeLevelEll.h"
#include "HeckeLevelN.h"
#include "HeckeLevelNSparse.h"
#include "HeckeNebentypus.h"
#include "../Registry/LegacyValidation.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/ModCtx.h"
#include "../Util/Timing.h"

namespace {

using oc_util::Seconds;

using hecke::HeckeOverflow;
using hecke::Mat22;
using hecke::Mat22W;
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

// (W18o-pin) one cell: the sorted order's presentation, its digest against the value pinned before the as-built order
// was retired (S24 CL-G2).
void ArmOrderPinned(Checker& g, const LevelNOrderCell& c, std::vector<std::string>& lines) {
    const engine::ModCtx<uint32_t> ctx(c.p);
    const Gamma0LevelN lvl(c.N);
    const auto t0 = std::chrono::steady_clock::now();
    const std::vector<SparseVec> gS = OrbitWiseFixedGenerators(lvl, Mat22::S(), 2, c.n, ctx);
    const std::vector<SparseVec> gU = OrbitWiseFixedGenerators(lvl, Mat22::ST(), 3, c.n, ctx);
    const uint32_t D = lvl.m() * (c.n + 1);
    const SparseCokernel B(D, gS, gU, c.p);
    const uint64_t fnv = PresentationDigest(B);
    g.check(fnv == c.fnv, "(W18o-pin) " + Cell(c.N, c.n, c.p) + ": PresentationDigest equals the pinned slot-24 value");
    std::ostringstream os;
    os << "LEVEL_N_ORDER (W18o-pin) N=" << c.N << " n=" << c.n << " p=" << c.p << " D=" << D << " h1=" << B.h1()
       << " final_nnz=" << B.stats().final_nnz << " presentation_fnv=0x" << std::hex << fnv << std::dec
       << " pinned=" << (fnv == c.fnv ? 1 : 0) << " merge_ops_sorted=" << B.stats().merge_ops << " occ_sorted="
       << B.stats().occ_entries << " peak_nnz_sorted=" << B.stats().peak_nnz << " wall_s=" << Seconds(t0);
    lines.push_back(os.str());
}

// (W18g) Gamma1LevelN::contains_wide.
void ArmGammaOneWide(Checker& g, std::vector<std::string>& lines) {
    // exactness on int64 inputs: members and non-members
    for (uint64_t N : {5ull, 7ull, 12ull, 13ull}) {
        const hecke::neb::Gamma1LevelN G(N);
        const int64_t n = static_cast<int64_t>(N);
        const std::vector<Mat22> probes{Mat22::I(), Mat22{1, 1, 0, 1}, Mat22{1, 0, n, 1}, Mat22{1 - n, 1, -n, 1},
                                        Mat22{-1, 0, 0, -1}, Mat22{0, -1, 1, 0}, Mat22{2 * n + 1, 1, 2 * n, 1},
                                        Mat22{1, 0, 2 * n, 1} * Mat22{1, 3, 0, 1}, Mat22{1, 0, 1, 1}};
        for (const Mat22& h : probes)
            g.check(G.contains_wide(Mat22W::from(h)) == G.contains(h),
                    "(W18g) Gamma_1(" + std::to_string(N) + "): contains_wide agrees with contains on an int64 input");
    }
    // a product beyond int64 in Gamma_1(N): T^{2^40} [[1, 0], [N, 1]] T^{-2^40} (c = N, d = 1 - N 2^40 = 1 mod N)
    const uint64_t N = 13;
    const hecke::neb::Gamma1LevelN G(N);
    const int64_t t = int64_t(1) << 40;
    const Mat22 X{1, t, 0, 1}, Y{1, 0, static_cast<int64_t>(N), 1}, Z{1, -t, 0, 1};
    bool narrow_refused = false, default_refused = false, shipped_ok = false, residues_ok = false;
    {
        ScopedLevelNWidth w(LevelNWidth::kForceNarrow);
        try {
            (void)SubgroupProductForRho(G, X, Y, Z, kP, false, "(W18g) not in Gamma_1(N)");
        } catch (const HeckeOverflow&) {
            narrow_refused = true;
        }
    }
    {
        const Mat22W h = Mat22W::from(X) * Mat22W::from(Y) * Mat22W::from(Z);
        try {
            (void)G.CosetLevel::contains_wide(h);   // the pre-slot-18 behaviour on Gamma_1(N)
        } catch (const HeckeOverflow&) {
            default_refused = true;
        }
        ResetLevelNWideProducts();
        const Mat22 r = SubgroupProductForRho(G, X, Y, Z, kP, false, "(W18g) not in Gamma_1(N)");
        shipped_ok = LevelNWideProducts() == 1 && G.contains_wide(h);
        // independent: h = [[1 + N t, -N t^2], [N, 1 - N t]] reduced mod kP in u128
        using u128 = unsigned __int128;
        const u128 P = kP, tt = static_cast<u128>(t) % P, NN = N % P;
        const uint64_t a = static_cast<uint64_t>((1 + NN * tt) % P), b = static_cast<uint64_t>((P - NN * tt % P * tt % P) % P),
                       c = static_cast<uint64_t>(NN), d = static_cast<uint64_t>((1 + P - NN * tt % P) % P);
        residues_ok = r.a == static_cast<int64_t>(a) && r.b == static_cast<int64_t>(b) && r.c == static_cast<int64_t>(c) &&
                      r.d == static_cast<int64_t>(d);
    }
    g.check(narrow_refused, "(W18g) the old width refuses the product beyond int64 on Gamma_1(13) (HeckeOverflow)");
    g.check(default_refused, "(W18g) negative control: the CosetLevel default contains_wide refused it on Gamma_1(13) (pre-slot-18)");
    g.check(shipped_ok, "(W18g) the shipped path escalates once and Gamma1LevelN::contains_wide accepts the member");
    g.check(residues_ok, "(W18g) the residues mod 1000003 equal an independent u128 computation");
    // forced wide == shipped on the Gamma_1(N) model's monomial matrices M_A(S), M_A(ST) (eq:MH; MonomialOf is the
    // SubgroupProductForRho caller the Gamma_1 model reaches -- the orbit-wise fixed spaces need -I in the group)
    uint32_t cells = 0;
    for (const auto& c : std::vector<std::array<uint64_t, 2>>{{5, 0}, {7, 2}, {12, 2}, {13, 0}, {21, 0}}) {
        const hecke::neb::Gamma1LevelN G1(c[0]);
        const uint32_t n = static_cast<uint32_t>(c[1]);
        const engine::ModCtx<uint32_t> ctx(kP);
        for (const Mat22& gm : {Mat22::S(), Mat22::ST()}) {
            ResetLevelNWideProducts();
            const MonomialAction a = MonomialOf(G1, gm, n, ctx);
            const uint64_t esc = LevelNWideProducts();
            ScopedLevelNWidth w(LevelNWidth::kForceWide);
            ResetLevelNWideProducts();
            const MonomialAction b = MonomialOf(G1, gm, n, ctx);
            const uint64_t wide = LevelNWideProducts();
            const std::string tag = "(W18g) Gamma_1(" + std::to_string(c[0]) + "), n = " + std::to_string(n) +
                                    (gm == Mat22::S() ? " S" : " ST");
            g.check(esc == 0 && wide > 0, tag + ": the shipped path never escalates, the forced one always does");
            g.check(a.perm == b.perm && a.dense() == b.dense(), tag + ": MonomialOf equal at both widths (entrywise)");
        }
        ++cells;
    }
    lines.push_back("LEVEL_N_ORDER (W18g) gamma1_cells=" + std::to_string(cells) + " narrow_refused=" + std::to_string(narrow_refused) +
                    " default_refused=" + std::to_string(default_refused) + " shipped_escalates=" + std::to_string(shipped_ok) +
                    " residues_ok=" + std::to_string(residues_ok));
}

} // namespace

uint32_t HeckeLevelNOrderArms(uint32_t& checks) {
    const auto t0 = std::chrono::steady_clock::now();
    Checker g;
    std::vector<std::string> lines;
    std::cout << "  -- slot-18 WP-5 arms ([C] rmk:levelNscale (4): (W18o) the generator order is value-free; (W18g) "
                 "Gamma1LevelN::contains_wide)"
              << std::endl;
    // (W18o): the default ladder's cells, small fields, d > 1, and the W5-F1 shape (composite N, P(N) = 43); the digests
    // are PresentationDigest's values printed by the slot-24 binary before the retirement (identical under both orders).
    const std::vector<LevelNOrderCell> ocells{
        {1571, 0, 5, false, 0xc55374b9caa66bafull},   {2003, 0, kP, false, 0x07377f577a4bf26full},
        {1000, 0, kP, false, 0x23a9491b9f034533ull},  {997, 2, kP, false, 0x79f1eddb224a33e7ull},
        {210, 4, kP, false, 0xe2f8179cf51f1074ull},   {30, 2, 7, false, 0x3b72484f97c1272cull},
        {1001, 4, 13, false, 0x18b8081ecb64ecb0ull},  {1155, 2, kP, false, 0xe2329c8319031359ull},
        {2310, 0, 17, false, 0x221667b67f09a310ull},  {1290, 0, kP, true, 0x1a7d758137598bd2ull},
        {2838, 0, kP, true, 0xe3d6bc4406d00824ull},   {9030, 0, 13, true, 0x20a2806b2d202aa0ull}};
    for (const LevelNOrderCell& c : ocells) ArmOrderPinned(g, c, lines);
    // (W18o-as_built) the retired order's bit-identity twin: OC_ENABLE_LEGACY=ON only
    if (oc::legacy::RunLegacyArm("HeckeLevelNOrderValidation", "as_built")) {
#if OC_ENABLE_LEGACY
        HeckeLevelNOrderAsBuiltArms(ocells, [&g](bool ok, const std::string& what) { g.check(ok, what); }, lines);
#endif
    }
    std::cout << "  [(W18o) " << ocells.size() << " cells done at " << Seconds(t0) << " s]" << std::endl;
    ArmGammaOneWide(g, lines);
    std::cout << "  [(W18g) done at " << Seconds(t0) << " s]" << std::endl;
    for (const std::string& l : lines) std::cout << "  " << l << "\n";
    checks += g.checks;
    return g.fails;
}
