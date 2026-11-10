// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// hecke-extensions slot 14, WP-6 -- HeckeRingLoewyValidation (standard).
//
// The Z/ell^K Hecke-algebra route (HeckeExt/HeckeRingHeckeAlgebra, route R1 of
// docs/implementation/slot14/WP-M1_loewy_note.md = [M1]) against its oracles:
//
//   (P) planted modules: (Z/49)^2 with T^2 = 7 (A = Z/49[T]/(T^2 - 7), free of
//       rank 2, A/7 = F_7[T]/T^2 and the same on N/7); (Z/49)^2 with T = 7 * swap
//       (A_1 = F_7 but A_2/7 = (1,1): a "hidden" algebra, eps = 1 -- [M1]
//       Prop. 1's saturation in miniature); Z/7^3 (+) Z/7 with T = diag(0, 1)
//       read with shift t = 1 (the free-part reading (iv): length 2, dim 1).
//   (a) [M1] Sec. 5.1: (82, 7) and (142, 7) at K <= 4 block by block (A_1,
//       A_K/7 per K, eps) and the (10, 691) control.
//   (b) A_1 = the census algebra A (Hilbert function, census edition 2 via
//       docs/implementation/slot12/WP-12_forms_census/forms_census.tsv column
//       A_hilbert) on a fixed sample of 20 non-reduced rows;
//   (c) A_K/ell at the stable K = the forms-side T^f_m/ell (slot-12 WP-12,
//       column M_T_hilbert) on the same 20 rows.
//   (d) one torsion (Dickson-type) block, (112, 7) system (5,1,3,0): free over
//       Z/49, not over Z/7^3 ([M1] Sec. 5.4) -- its torsion, free rank and
//       free-part reading.
//   (W) the module runner: the native route (uint32_t while ell^K < 2^32) and
//       hida::RunModuleCell (uint64_t) give identical LOEWY_RING lines at (58, 7).
//   (e) C-F9 ([E] pr:latticesplit, CONJ): m_A^2 = 0 on every non-reduced block
//       met in (a)-(d).  PINNED to what the WP-6 runs observed (held on every
//       such block); a change is a FINDING about a conjecture, to be explained,
//       not a test to be repaired.
//   (C1) (slot-15 WP-5) the certified verdict: every stable-family verdict met in
//       (a)-(d) is stable_cert=certified (HeckeRingHeckeAlgebra's CertifyStable,
//       [C, cor:stablecert] in the B1 form of the WP-4 referee), every other is
//       none; certify = false gives `reading` and changes only the appended field;
//       the certificate's content at (142, 7) Delta.
//   (C2) negative controls: a perturbed claim (length, dim, eps, f, a stable
//       verdict one level early, a wrong n, K_cls <= e_max) and a perturbed A_K
//       (a planted module through the algebraic core CheckStableReading; a
//       perturbed T_2 on (142, 7)) are refused by a throw.
//   (C3) the digit route: identical LOEWY_RING lines on the u64 route and the
//       digit route (ModuleRoute::Digit) at (58, 7), (30, 5), (34, 5); beyond
//       2^64 (ell = 5, K = 28, the NP5R family) the digit module's readings meet
//       [C, prop:saturation](3)-(5) / [C, prop:satdefect](2) predicted from the u64
//       route at K = 26, 27, and CertifyStable certifies them.
//   (ENV) HECKE_RING_LOEWY_CELLS="n:ell[:KMAX],..." (read-only, not an emitter):
//       the route over the listed cells, one LOEWY_RING line per block, one
//       LOEWY_RING_SUMMARY line; KMAX defaults to 7 at ell = 7, 6 otherwise.
//       Gated on the route's own consistency only; C-F9 and the verdicts are
//       REPORTED (a LOEWY_RING_CF9_FINDING line per failing block).
// ===========================================================================

#include "HeckeRingLoewyValidation.h"

#include "HeckeRingHeckeAlgebra.h"

#include "../Hecke/HeckeSettings.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../Util/ValidationCheck.h"
#include "../Util/CellList.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

using namespace hecke_ext::ringalg;
using hecke::ordinary::RingMat;

oc_util::Checker Check("  FAIL: ", false);

using H = std::vector<uint32_t>;

std::string Txt(const H& h) { return HilbertText(h); }

RingMat Mat(uint64_t ell, const std::vector<uint32_t>& exps, const std::vector<uint64_t>& a) {
    RingMat M;
    M.ell = ell;
    M.row_exp = exps;
    M.a = a;
    return M;
}

OpModule Planted(uint64_t ell, const std::vector<uint32_t>& exps, const std::vector<uint64_t>& T) {
    OpModule N;
    N.ell = ell;
    N.exps = exps;
    N.primes = {2};
    N.op = {Mat(ell, exps, T)};
    return N;
}

void ArmP() {
    std::cout << "(P) planted modules\n";
    {   // T^2 = 7 on (Z/49)^2
        const OpModule N = Planted(7, {2, 2}, {0, 7, 1, 0});
        const AlgebraReading A = ReadAlgebra(N, {0});
        Check(A.length == 4 && A.dim_mod_ell == 2 && A.hilbert == H({1, 1}), "(P) T^2 = 7: A = Z/49[T]/(T^2-7), A/7 = (1,1), got len " +
              std::to_string(A.length) + " " + Txt(A.hilbert));
        const AlgebraReading A1 = ReadAlgebra(ReduceModEll(N), {0});
        Check(A1.dim_mod_ell == 2 && A1.hilbert == H({1, 1}), "(P) T^2 = 7: A_1 = F_7[T]/T^2, got " + Txt(A1.hilbert));
        const std::vector<JointSystem> js = RationalSystems(N);
        Check(js.size() == 1 && js[0].a == std::vector<uint64_t>({0}) && js[0].dim == 2, "(P) T^2 = 7: one system a = 0 of dim 2");
        const Block B = Localize(N, {0});
        Check(B.local.exps == std::vector<uint32_t>({2, 2}), "(P) T^2 = 7: e_m N = N");
    }
    {   // T = 7 * swap on (Z/49)^2: A_1 = F_7, A_2/7 = (1,1), eps = 2*2 - 3 = 1
        const OpModule N = Planted(7, {2, 2}, {0, 7, 7, 0});
        const AlgebraReading A = ReadAlgebra(N, {0});
        const AlgebraReading A1 = ReadAlgebra(ReduceModEll(N), {0});
        Check(A.length == 3 && A.hilbert == H({1, 1}) && A1.hilbert == H({1}),
              "(P) T = 7 swap: len 3, A_2/7 = (1,1), A_1 = (1); got len " + std::to_string(A.length) + " " + Txt(A.hilbert) + " " +
                  Txt(A1.hilbert));
        Check(2 * static_cast<int64_t>(A.dim_mod_ell) - A.length == 1, "(P) T = 7 swap: eps_2 = 1");
    }
    {   // Z/7^3 (+) Z/7, T = diag(7, 0): A = {(x, y) : x = y mod 7}, length 3; the free-part reading with t = 1
        const OpModule N = Planted(7, {3, 1}, {7, 0, 0, 0});
        const AlgebraReading full = ReadAlgebra(N, {0});
        const AlgebraReading fp = ReadAlgebra(N, {0}, 1);
        Check(full.length == 3 && full.hilbert == H({1}), "(P) Z/7^3 + Z/7: length(A) = 3, A/7 = (1), got len " +
              std::to_string(full.length) + " " + Txt(full.hilbert));
        Check(fp.length == 2 && fp.dim_mod_ell == 1 && fp.hilbert == H({1}),
              "(P) Z/7^3 + Z/7: the free part at level 2 is Z/49, got len " + std::to_string(fp.length) + " " + Txt(fp.hilbert));
    }
    {   // Z/7^3 (+) Z/7, T = diag(0, 1): two systems, the blocks Z/7^3 and Z/7 (a torsion block)
        const OpModule N = Planted(7, {3, 1}, {0, 0, 0, 1});
        const std::vector<JointSystem> js = RationalSystems(N);
        Check(js.size() == 2, "(P) diag(0, 1): two systems (0 and 1)");
        const Block B0 = Localize(N, {0}), B1 = Localize(N, {1});
        Check(B0.local.exps == std::vector<uint32_t>({3}) && B1.local.exps == std::vector<uint32_t>({1}),
              "(P) diag(0, 1): the blocks are Z/7^3 and Z/7");
        bool threw = false;
        try { (void)ReadAlgebra(N, {0}); } catch (const std::logic_error&) { threw = true; }
        Check(threw, "(P) diag(0, 1): reading a non-local module at one m is refused");
    }
    {   // slot-14 WP-16 (audit S-2): the stopping rule's loud check fires on an engineered bad input --
        // the planted T^2 = 7 algebra (dim A/7 = 2) against a claimed rank_T = 1 -- and is silent at rank_T = 2.
        const AlgebraReading A = ReadAlgebra(Planted(7, {2, 2}, {0, 7, 1, 0}), {0});
        bool threw = false;
        std::string what;
        try { CheckWithinRankT(A, 1); } catch (const std::logic_error& e) { threw = true; what = e.what(); }
        Check(threw && what.find("exceeds rank T") != std::string::npos,
              "(P) S-2: dim A_K/ell = 2 > rank_T = 1 throws (the rank formula check), got '" + what + "'");
        bool quiet = true;
        try { CheckWithinRankT(A, 2); } catch (const std::logic_error&) { quiet = false; }
        Check(quiet, "(P) S-2: dim A_K/ell = 2 <= rank_T = 2 does not throw");
        std::cout << "  (P) S-2 negative arm: " << (threw ? "ENGINE_CHECK fired: " + what : std::string("DID NOT FIRE")) << "\n";
    }
    {   // slot-14 WP-16 (audit M-9): an odd n is refused by name, not answered with zero blocks
        bool threw = false;
        LoewyOptions o;
        o.kmax = 2;
        try { (void)ComputeRingLoewyCell(11, 7, o); } catch (const std::logic_error& e) {
            threw = std::string(e.what()).find("n must be even") != std::string::npos;
        }
        Check(threw, "(P) M-9: ComputeRingLoewyCell(11, 7) is refused (odd n)");
    }
}

// ---- the cell runs of arms (a)-(d), shared --------------------------------
using CellKey = std::pair<std::pair<uint32_t, uint64_t>, std::vector<uint64_t>>;
std::map<CellKey, LoewyCell> g_cells;

// (S15-O1 perf) validation-local cache of default-route module runs, keyed (n, ell, K, primes);
// the arms only read the result, so every check is unchanged.  Cleared with g_cells.
std::map<std::tuple<uint32_t, uint64_t, uint32_t, std::vector<uint64_t>>, OpModule> g_modules;
const OpModule& RunModuleCached(uint32_t n, uint64_t ell, uint32_t K, const std::vector<uint64_t>& primes) {
    const auto key = std::make_tuple(n, ell, K, primes);
    auto it = g_modules.find(key);
    if (it == g_modules.end()) it = g_modules.emplace(key, RunModule(n, ell, K, primes)).first;
    return it->second;
}

const LoewyCell& Cell(uint32_t n, uint64_t ell, uint32_t kmax, const std::vector<uint64_t>& primes = {}) {
    const CellKey key{{n, ell}, primes};
    auto it = g_cells.find(key);
    if (it != g_cells.end() && it->second.kmax >= kmax) return it->second;
    LoewyOptions o;
    o.kmax = kmax;
    o.primes = primes;
    LoewyCell c = ComputeRingLoewyCell(n, ell, o);
    std::cout << "  cell (" << n << ", " << ell << ") KMAX " << kmax << ": " << c.blocks.size() << " blocks, e_max " << c.e_max
              << ", " << c.seconds << " s (module runs " << c.module_seconds << " s)\n";
    for (const LoewyBlock& b : c.blocks) std::cout << "    " << LoewyRingLine(c, b) << "\n";
    return g_cells[key] = std::move(c);
}

const LoewyBlock* Find(const LoewyCell& c, const std::vector<uint64_t>& sys) {
    for (const LoewyBlock& b : c.blocks)
        if (b.system == sys) return &b;
    return nullptr;
}

const LoewyStep* StepAt(const LoewyBlock& b, uint32_t K) {
    for (const LoewyStep& s : b.history)
        if (s.K == K) return &s;
    return nullptr;
}

void ArmA() {
    std::cout << "(a) [M1] Sec. 5.1\n";
    {
        const LoewyCell& c = Cell(82, 7, 4);
        const LoewyBlock* b = Find(c, {4, 0, 0, 0});
        Check(b != nullptr, "(a) (82, 7): the block (4,0,0,0) exists");
        if (b) {
            Check(b->D == 8 && b->A1.hilbert == H({1, 2}), "(a) (82, 7): D = 8, A_1 = (1,2)");
            const LoewyStep* s2 = StepAt(*b, 2);
            Check(s2 && s2->full.hilbert == H({1, 3}), "(a) (82, 7): A_2/7 = (1,3)");
            Check(b->verdict == "stable" && b->K_stab == 2 && b->eps == 1, "(a) (82, 7): stable at K = 2, eps = 1");
        }
    }
    {
        const LoewyCell& c = Cell(142, 7, 3);
        const LoewyBlock* b = Find(c, {4, 0, 0, 0});
        Check(b != nullptr, "(a) (142, 7): the block (4,0,0,0) exists");
        if (b) {
            Check(b->D == 12 && b->A1.hilbert == H({1, 3}), "(a) (142, 7) Delta: D = 12, A_1 = (1,3)");
            const LoewyStep* s2 = StepAt(*b, 2);
            const LoewyStep* s3 = StepAt(*b, 3);
            Check(s2 && s2->full.hilbert == H({1, 3}), "(a) (142, 7) Delta: A_2/7 = (1,3)");
            Check(s3 && s3->full.hilbert == H({1, 3, 2}), "(a) (142, 7) Delta: A_3/7 = (1,3,2), Loewy 3 (the RCOMP datum)");
            Check(b->verdict == "stable" && b->K_stab == 3 && b->eps == 4, "(a) (142, 7) Delta: stable at K = 3, eps = 4");
        }
        const LoewyBlock* d = Find(c, {5, 6, 4, 0});
        Check(d != nullptr, "(a) (142, 7): the block (5,6,4,0) exists");
        if (d) {
            Check(d->D == 6 && d->A1.hilbert == H({1}), "(a) (142, 7) (5,6,4,0): D = 6, A_1 = (1) (reduced mod 7)");
            const LoewyStep* s2 = StepAt(*d, 2);
            const LoewyStep* s3 = StepAt(*d, 3);
            Check(s2 && s2->full.hilbert == H({1, 1}), "(a) (142, 7) (5,6,4,0): A_2/7 = (1,1)");
            Check(s3 && s3->full.hilbert == H({1, 1, 1}), "(a) (142, 7) (5,6,4,0): A_3/7 = (1,1,1)");
            Check(d->verdict == "stable" && d->eps == 3 && d->hidden(), "(a) (142, 7) (5,6,4,0): stable, eps = 3, a hidden congruence");
        }
    }
    {
        const LoewyCell& c = Cell(10, 691, 3, {2, 3});
        bool seen = false;
        for (const LoewyBlock& b : c.blocks) {
            if (b.A1.dim_mod_ell < 2) continue;
            seen = true;
            Check(b.A1.hilbert == H({1, 1}) && b.last_T() && b.last_T()->hilbert == H({1, 1}) && b.eps == 0,
                  "(a) (10, 691) control: A_1 = A_K/691 = (1,1), eps = 0");
        }
        Check(seen, "(a) (10, 691): the non-reduced block (the 691 | B_12 congruence) is met");
    }
}

struct CensusRow {
    uint32_t n;
    uint64_t ell;
    std::vector<uint64_t> primes, system;
    H census_A, forms_T;
};

// 20 non-reduced H^1 rows of census edition 2 (forms_census.tsv: system_primes,
// system, A_hilbert, M_T_hilbert), n <= 142, the three census primes and 691.
const std::vector<CensusRow>& Rows() {
    static const std::vector<CensusRow> rows = {
        {10, 691, {2, 3, 5, 7, 691}, {667, 252, 684, 531, 1}, {1, 1}, {1, 1}},
        {34, 7, {2, 3, 5, 7}, {4, 0, 0, 0}, {1, 1}, {1, 1}},
        {38, 13, {2, 3, 5, 7, 13}, {8, 6, 6, 0, 0}, {1, 1}, {1, 1}},
        {54, 11, {2, 3, 5, 7, 11}, {7, 7, 3, 10, 0}, {1, 1}, {1, 1}},
        {58, 7, {2, 3, 5, 7}, {4, 0, 0, 0}, {1, 1}, {1, 2}},
        {64, 13, {2, 3, 5, 7, 13}, {5, 6, 6, 3, 0}, {1, 1}, {1, 1}},
        {66, 11, {2, 3, 5, 7, 11}, {3, 10, 4, 4, 0}, {1, 1}, {1, 1}},
        {66, 7, {2, 3, 5, 7}, {1, 0, 0, 0}, {1, 1}, {1, 2}},
        {74, 7, {2, 3, 5, 7}, {2, 0, 0, 0}, {1, 1}, {1, 2}},
        {76, 11, {2, 3, 5, 7, 11}, {0, 6, 10, 0, 0}, {1, 1}, {1, 1}},
        {78, 13, {2, 3, 5, 7, 13}, {10, 5, 4, 8, 0}, {1, 1}, {1, 1}},
        {82, 7, {2, 3, 5, 7}, {4, 0, 0, 0}, {1, 2}, {1, 3}},
        {88, 11, {2, 3, 5, 7, 11}, {0, 7, 6, 0, 0}, {1, 1}, {1, 1}},
        {90, 7, {2, 3, 5, 7}, {1, 0, 0, 0}, {1, 1}, {1, 2, 1}},
        {90, 13, {2, 3, 5, 7, 13}, {1, 4, 4, 6, 0}, {1, 1}, {1, 1}},
        {98, 7, {2, 3, 5, 7}, {2, 0, 0, 0}, {1, 1}, {1, 2, 1}},
        {106, 7, {2, 3, 5, 7}, {4, 0, 0, 0}, {1, 1}, {1, 3, 1}},
        {114, 7, {2, 3, 5, 7}, {1, 0, 0, 0}, {1, 1}, {1, 2, 2}},
        {118, 7, {2, 3, 5, 7}, {4, 0, 0, 0}, {1, 2}, {1, 3, 1}},
        {142, 7, {2, 3, 5, 7}, {4, 0, 0, 0}, {1, 3}, {1, 3, 2}},
    };
    return rows;
}

void ArmBC() {
    std::cout << "(b)/(c) 20 census rows: A_1 = census A, A_K/ell = forms-side T^f_m/ell\n";
    for (const CensusRow& r : Rows()) {
        const uint32_t kmax = (r.ell == 691 || r.n == 142) ? 3 : 4;
        const LoewyCell& c = Cell(r.n, r.ell, kmax, r.ell == 691 ? r.primes : std::vector<uint64_t>{});
        std::vector<uint64_t> sys;   // the row's system on the cell's primes
        bool ok = c.primes == r.primes;
        if (ok) sys = r.system;
        for (auto& x : sys) x %= r.ell;
        const LoewyBlock* b = ok ? Find(c, sys) : nullptr;
        const std::string tag = "(" + std::to_string(r.n) + ", " + std::to_string(r.ell) + ")";
        Check(b != nullptr, "(b) " + tag + ": the census system is a block of the route");
        if (!b) continue;
        Check(b->A1.hilbert == r.census_A, "(b) " + tag + ": A_1 = census A " + Txt(r.census_A) + ", got " + Txt(b->A1.hilbert));
        const AlgebraReading* T = b->last_T();
        const H got = T ? T->hilbert : H({1});
        Check(b->verdict == "stable" || b->verdict == "reduced_rank1", "(c) " + tag + ": stable within KMAX, got " + b->verdict);
        Check(got == r.forms_T, "(c) " + tag + ": A_K/ell = forms T^f_m/ell " + Txt(r.forms_T) + ", got " + Txt(got));
    }
}

void ArmD() {
    std::cout << "(d) the torsion block (112, 7) (5,1,3,0)\n";
    const LoewyCell& c = Cell(112, 7, 4);
    const LoewyBlock* b = Find(c, {5, 1, 3, 0});
    Check(b != nullptr, "(d) (112, 7): the block (5,1,3,0) exists");
    if (!b) return;
    Check(c.e_max >= 2 && c.K_cls == c.e_max + 1, "(d) (112, 7): e_max >= 2 (a torsion summand of order 7^2)");
    Check(!b->torsion.empty() && b->torsion.front() == 2, "(d) (112, 7) (5,1,3,0): its torsion has exponent 2 (free over Z/49, not over Z/7^3)");
    Check(b->verdict.rfind("torsion_", 0) == 0 && b->verdict != "torsion_not_stable_at_KMAX",
          "(d) (112, 7) (5,1,3,0): the torsion block is read to a certified stop, got " + b->verdict);
    // pinned to the WP-6 run (the Python route skipped this block, [M1] Sec. 5.4)
    Check(b->D == 5 && b->free_rank == 4 && b->torsion == std::vector<uint32_t>({2}) && b->rank_T == 2,
          "(d) (112, 7) (5,1,3,0): D = 5 = free rank 4 + one torsion summand Z/49");
    Check(b->verdict == "torsion_free_stable" && b->K_stab == 4, "(d) (112, 7) (5,1,3,0): the free part is stable at K = 4 (level 2)");
    const LoewyStep& st = b->history.back();
    Check(st.has_free && st.free_level == 2 && st.free_part.hilbert == H({1, 1}) && st.full.hilbert == H({1, 1}),
          "(d) (112, 7) (5,1,3,0): the free part T_free/7 = (1,1), the full A_4/7 = (1,1)");
    Check(b->A1.hilbert == H({1}) && b->hidden(), "(d) (112, 7) (5,1,3,0): reduced mod 7 (A_1 = F_7), a hidden congruence");
    Check(std::string(b->cert()) == "free" && b->certified_T() == &st.free_part,
          "(d) (112, 7) (5,1,3,0): hidden is read on the certified free part (cert=free, audit S-6)");
    Check(b->D == b->free_rank + b->torsion.size(), "(d) (112, 7): D = free rank + torsion generators");
}

// (W) the module runner: the native route (uint32_t below 2^32) against
// hida::RunModuleCell (uint64_t) -- the same objects, so the same LOEWY_RING lines.
void ArmW() {
    std::cout << "(W) the native module runner against hida::RunModuleCell (uint64_t)\n";
    LoewyOptions o;
    o.kmax = 3;
    const LoewyCell a = ComputeRingLoewyCell(58, 7, o);
    o.route = ModuleRoute::Hida;
    const LoewyCell b = ComputeRingLoewyCell(58, 7, o);
    Check(a.blocks.size() == b.blocks.size() && !a.blocks.empty(), "(W) (58, 7): the same blocks on both runners");
    for (size_t i = 0; i < std::min(a.blocks.size(), b.blocks.size()); ++i)
        Check(LoewyRingLine(a, a.blocks[i]) == LoewyRingLine(b, b.blocks[i]), "(W) (58, 7): block " + std::to_string(i) +
              " identical on both runners");
    std::cout << "  native " << a.seconds << " s, hida " << b.seconds << " s\n";
}

void ArmE() {
    std::cout << "(e) C-F9 ([E] pr:latticesplit, CONJ) on every non-reduced block met -- pinned: held\n";
    uint32_t nonred = 0, held = 0;
    for (const auto& kv : g_cells)
        for (const LoewyBlock& b : kv.second.blocks) {
            if (!b.nonreduced_A1()) continue;
            ++nonred;
            if (b.cf9_holds()) ++held;
            else std::cout << "  LOEWY_RING_CF9_FINDING " << LoewyRingLine(kv.second, b) << "\n";
        }
    std::cout << "  non-reduced blocks met: " << nonred << ", m_A^2 = 0 on " << held << "\n";
    Check(nonred > 0, "(e) some non-reduced block is met");
    Check(held == nonred, "(e) FINDING (pinned): C-F9 held on every non-reduced block the WP-6 runs met");
}

// ---- (slot-15 WP-5) the certified verdict and the digit route ----------------

bool Refused(const std::function<void()>& f, std::string& what) {
    try {
        f();
    } catch (const std::logic_error& e) {
        what = e.what();
        return true;
    }
    return false;
}

// (C1) [C, cor:stablecert]: every stable-family verdict met in (a)-(d) is certified
// (CertifyStable ran inside ComputeRingLoewyCell and did not throw), every other
// verdict carries stable_cert=none; with certify = false the verdict is a reading
// and the LOEWY_RING line differs ONLY in the appended field (the old text is kept).
void ArmC1() {
    std::cout << "(C1) the certified verdict ([C, cor:stablecert], B1 form) on every stable verdict met\n";
    uint32_t fam = 0, cert = 0, other = 0, none = 0;
    for (const auto& kv : g_cells)
        for (const LoewyBlock& b : kv.second.blocks) {
            if (b.stable_family()) {
                ++fam;
                if (b.stable_cert == "certified") ++cert;
                else std::cout << "  NOT CERTIFIED: " << LoewyRingLine(kv.second, b) << "\n";
            } else {
                ++other;
                if (b.stable_cert == "none") ++none;
            }
        }
    std::cout << "  LOEWY_RING_CERT label=cor:stablecert stable_family=" << fam << " certified=" << cert << " other=" << other
              << " none=" << none << "\n";
    Check(fam > 0 && cert == fam, "(C1) every stable-family verdict is certified (" + std::to_string(cert) + " / " +
                                      std::to_string(fam) + ")");
    Check(none == other, "(C1) every other verdict carries stable_cert=none");
    {   // certify = false: the reading, and the old text byte-identical
        LoewyOptions o;
        o.kmax = 4;
        o.certify = false;
        const LoewyCell r = ComputeRingLoewyCell(82, 7, o);
        const LoewyCell& c = Cell(82, 7, 4);
        bool same = r.blocks.size() == c.blocks.size() && !r.blocks.empty(), reading = true;
        for (size_t i = 0; same && i < r.blocks.size(); ++i) {
            std::string a = LoewyRingLine(r, r.blocks[i]), b = LoewyRingLine(c, c.blocks[i]);
            const std::string ta = " stable_cert=" + r.blocks[i].stable_cert, tb = " stable_cert=" + c.blocks[i].stable_cert;
            same = same && a.size() > ta.size() && a.compare(a.size() - ta.size(), ta.size(), ta) == 0 &&
                   b.compare(b.size() - tb.size(), tb.size(), tb) == 0 &&
                   a.substr(0, a.size() - ta.size()) == b.substr(0, b.size() - tb.size());
            if (r.blocks[i].stable_family() && r.blocks[i].stable_cert != "reading") reading = false;
        }
        Check(same, "(C1) certify = false: the LOEWY_RING lines differ only in the appended stable_cert field");
        Check(reading, "(C1) certify = false: a stable-family verdict is stable_cert=reading");
        std::cout << "  (82, 7) KMAX 4: certify = false " << r.seconds << " s, certify = true " << c.seconds << " s\n";
    }
    {   // the certificate's content at (142, 7) Delta: level 3, dim = rank_T = 6, eps = 4
        const OpModule& N3 = RunModuleCached(142, 7, 3, {2, 3, 5, 7});
        const LoewyCell& c = Cell(142, 7, 3);
        const LoewyBlock* b = Find(c, {4, 0, 0, 0});
        Check(b != nullptr, "(C1) (142, 7) Delta present");
        if (b) {
            const OpModule& Nc = c.K_cls == 3 ? N3 : RunModuleCached(142, 7, c.K_cls, {2, 3, 5, 7});
            const StableCertificate s = CertifyStable(142, Nc, N3, *b, c.K_cls);
            std::cout << "  (142, 7) Delta: label=" << s.label << " level=" << s.level << " f=" << s.f << " rank_T=" << s.rank_T
                      << " dim=" << s.dim << " length=" << s.length << " eps=" << s.eps
                      << " monomial_generators=" << s.monomial_generators << "\n";
            Check(s.label == "cor:stablecert" && s.level == 3 && s.f == 12 && s.rank_T == 6 && s.dim == 6 && s.eps == 4 &&
                      s.length == 14,
                  "(C1) (142, 7) Delta: certified at level 3, f = 12, dim = rank_T = 6, length 14, eps = 4");
        }
    }
}

// (C2) negative controls: the certificate refuses a perturbed claim and a perturbed A_K.
void ArmC2() {
    std::cout << "(C2) negative controls: a perturbed A_K / claim is refused\n";
    const std::vector<uint64_t> S = {2, 3, 5, 7};
    const LoewyCell& c = Cell(142, 7, 3);
    const LoewyBlock* d = Find(c, {4, 0, 0, 0});
    Check(d != nullptr, "(C2) (142, 7) Delta present");
    if (!d) return;
    const OpModule& N3 = RunModuleCached(142, 7, 3, S);
    const OpModule& N2 = RunModuleCached(142, 7, 2, S);
    const OpModule& Nc = c.K_cls == 3 ? N3 : (c.K_cls == 2 ? N2 : RunModuleCached(142, 7, c.K_cls, S));
    std::string w;
    Check(!Refused([&] { (void)CertifyStable(142, Nc, N3, *d, c.K_cls); }, w), "(C2) the unperturbed claim is certified: " + w);
    auto expect = [&](const std::string& tag, const LoewyBlock& claim, const OpModule& NK, const std::string& needle) {
        std::string what;
        const bool r = Refused([&] { (void)CertifyStable(142, Nc, NK, claim, c.K_cls); }, what);
        Check(r && what.find(needle) != std::string::npos, "(C2) " + tag + " is refused ('" + needle + "'), got '" + what + "'");
        std::cout << "  (C2) " << tag << ": " << (r ? "refused -- " + what.substr(what.find("refuses")) : std::string("NOT REFUSED"))
                  << "\n";
    };
    {   LoewyBlock p = *d; p.history.back().full.length += 1; expect("length(A_K) + 1", p, N3, "length(A'_K)"); }
    {   LoewyBlock p = *d; p.history.back().full.dim_mod_ell -= 1; expect("dim A_K/ell - 1", p, N3, "dim A'_K/ell"); }
    {   LoewyBlock p = *d; p.eps += 1; expect("eps + 1", p, N3, "eps_K"); }
    {   LoewyBlock p = *d; p.free_rank += 2; p.rank_T += 1; expect("f + 2", p, N3, "f = rank(e L')"); }
    {   // stable claimed one level early: A_2/7 = (1,3), dim 4 < rank_T = 6
        LoewyBlock p = *d;
        p.history.pop_back();
        p.K_stab = 2;
        p.eps = 2 * static_cast<int64_t>(p.history.back().full.dim_mod_ell) - p.history.back().full.length;
        expect("stable at K = 2 (one level early)", p, N2, "criterion");
    }
    {   // the whole-module Setting check (2'): a wrong n (the dimension formula of another weight) is refused
        std::string what;
        const bool r = Refused([&] { (void)CertifyStable(140, Nc, N3, *d, c.K_cls); }, what);
        Check(r && what.find("free rank of H^1") != std::string::npos, "(C2) (142, 7) Delta with n = 140 is refused, got '" + what + "'");
        std::cout << "  (C2) wrong n: " << (r ? "refused -- " + what.substr(what.find("refuses")) : std::string("NOT REFUSED")) << "\n";
    }
    {   // (2') K_cls <= e_max: the torsion block (112, 7) (5,1,3,0) (e_max = 2) claimed against the module at K = 2 as "K_cls"
        const LoewyCell& t = Cell(112, 7, 4);
        const LoewyBlock* tb = Find(t, {5, 1, 3, 0});
        Check(tb != nullptr, "(C2) (112, 7) (5,1,3,0) present");
        if (tb) {
            const OpModule& M2 = RunModuleCached(112, 7, 2, S), &Mc = RunModuleCached(112, 7, t.K_cls, S), &MK = RunModuleCached(112, 7, tb->K_stab, S);
            std::string what;
            Check(!Refused([&] { (void)CertifyStable(112, Mc, MK, *tb, t.K_cls); }, what), "(C2) (112, 7) torsion_free_stable certified " + what);
            const bool r = Refused([&] { (void)CertifyStable(112, M2, MK, *tb, 2); }, what);
            Check(r && what.find("K_cls <= e_max") != std::string::npos, "(C2) (112, 7) with K_cls = 2 <= e_max is refused, got '" + what + "'");
            std::cout << "  (C2) K_cls <= e_max: " << (r ? "refused -- " + what.substr(what.find("refuses")) : std::string("NOT REFUSED")) << "\n";
        }
    }
    // (audit M6) further refusals: verdict shapes, satdefect(2) monotonicity, reduced_rank1 / torsion_stable,
    // the nilpotency / idempotent construction on a planted module, (2')'s torsion multiset.
    {   LoewyBlock p = *d; p.history[0].full.length = 2; expect("eps^sat decreasing along the history", p, N3, "decreasing"); }
    {
        const LoewyBlock* rr = nullptr;
        for (const LoewyBlock& b : c.blocks)
            if (b.verdict == "reduced_rank1") { rr = &b; break; }
        Check(rr != nullptr, "(C2) (142, 7) has a reduced_rank1 block");
        if (rr) {
            LoewyBlock p = *rr;
            p.rank_T = 2;
            std::string what;
            const bool r = Refused([&] { (void)CertifyStable(142, Nc, Nc, p, c.K_cls); }, what);
            Check(r && what.find("rank_T = ceil(f/2)") != std::string::npos, "(C2) reduced_rank1 with rank_T = 2 is refused, got '" + what + "'");
            std::cout << "  (C2) reduced_rank1 rank_T = 2: " << (r ? "refused -- " + what.substr(what.find("refuses")) : std::string("NOT REFUSED")) << "\n";
        }
    }
    {
        const LoewyCell& t = Cell(112, 7, 4);
        const LoewyBlock* tb = Find(t, {5, 1, 3, 0});
        if (tb) {
            const OpModule& Mc = RunModuleCached(112, 7, t.K_cls, S), &MK = RunModuleCached(112, 7, tb->K_stab, S);
            auto texpect = [&](const std::string& tag, const LoewyBlock& claim, const std::string& needle) {
                std::string what;
                const bool r = Refused([&] { (void)CertifyStable(112, Mc, MK, claim, t.K_cls); }, what);
                Check(r && what.find(needle) != std::string::npos, "(C2) (112, 7) " + tag + " is refused ('" + needle + "'), got '" + what + "'");
                std::cout << "  (C2) (112, 7) " << tag << ": " << (r ? "refused -- " + what.substr(what.find("refuses")) : std::string("NOT REFUSED")) << "\n";
            };
            { LoewyBlock p = *tb; p.verdict = "stable"; texpect("'stable' on a torsion block", p, "'stable' on a block with torsion"); }
            { LoewyBlock p = *tb; p.history.back().has_free = false; texpect("torsion_free_stable without a free reading", p, "free part was not read"); }
            { LoewyBlock p = *tb; p.verdict = "torsion_stable"; texpect("torsion_stable with f > 0", p, "pure-torsion block"); }
            {   // (2')'s torsion multiset: the (112, 7) module read as n = 110 or 114 (same free rank when d_k agrees)
                bool tested = false;
                for (uint32_t nn : {110u, 114u, 116u, 108u}) {
                    if (hecke::ordinary::FreeRankH1(nn) != hecke::ordinary::FreeRankH1(112)) continue;
                    const hecke::ordinary::TorsionPrepass pp = hecke::ordinary::TorsionExponentPrepass(nn, 7);
                    const hecke::ordinary::TorsionPrepass p0 = hecke::ordinary::TorsionExponentPrepass(112, 7);
                    if (pp.torsion == p0.torsion || pp.e_max >= t.K_cls) continue;
                    std::string what;
                    const bool r = Refused([&] { (void)CertifyStable(nn, Mc, MK, *tb, t.K_cls); }, what);
                    Check(r && what.find("pre-pass torsion") != std::string::npos,
                          "(C2) (112, 7) read as n = " + std::to_string(nn) + ": the torsion multiset is refused, got '" + what + "'");
                    std::cout << "  (C2) torsion multiset (n = " << nn << "): " << (r ? "refused -- " + what.substr(what.find("refuses")) : std::string("NOT REFUSED")) << "\n";
                    tested = true;
                    break;
                }
                Check(tested, "(C2) a weight with the same free rank and another torsion multiset was found for (2')");
            }
        }
    }
    {   // torsion_stable: (18, 5) (4,3,0,4), eps != 0 refused
        const LoewyCell& t = Cell(18, 5, 4);
        const LoewyBlock* tb = Find(t, {4, 3, 0, 4});
        Check(tb != nullptr && tb->verdict == "torsion_stable" && tb->stable_cert == "certified", "(C2) (18, 5) (4,3,0,4) is a certified torsion_stable block");
        if (tb) {
            const std::vector<uint64_t> S5 = {2, 3, 5, 7};
            const OpModule& Mc = RunModuleCached(18, 5, t.K_cls, S5), &MK = RunModuleCached(18, 5, tb->K_stab, S5);
            LoewyBlock p = *tb;
            p.eps = 1;
            std::string what;
            const bool r = Refused([&] { (void)CertifyStable(18, Mc, MK, p, t.K_cls); }, what);
            Check(r && what.find("pure-torsion block carries eps") != std::string::npos, "(C2) torsion_stable with eps = 1 is refused, got '" + what + "'");
            std::cout << "  (C2) torsion_stable eps = 1: " << (r ? "refused -- " + what.substr(what.find("refuses")) : std::string("NOT REFUSED")) << "\n";
        }
    }
    {   // planted (Z/49)^2, T = diag(0, 7), system {1}: 1 is not an eigenvalue mod 7 -- refused by the projector
        // construction, one step before the nilpotency check (which is thereby a consistency check, audit M1)
        OpModule N;
        N.ell = 7;
        N.exps = {2, 2};
        N.primes = {2};
        N.op = {Mat(7, {2, 2}, {0, 0, 0, 7})};
        LoewyBlock claim;
        claim.system = {1};
        claim.D = 2;
        claim.free_rank = 2;
        claim.rank_T = 1;
        claim.verdict = "stable";
        claim.K_stab = 2;
        LoewyStep st;
        st.K = 2;
        st.full = ReadAlgebra(N, {0});
        claim.history = {st};
        std::string what;
        const bool r = Refused([&] { (void)CheckStableReading(N, N, claim, 2); }, what);
        Check(r, "(C2) planted diag(0, 7) at the system {1} is refused, got '" + what + "'");
        std::cout << "  (C2) planted non-eigenvalue system: " << (r ? "refused -- " + what : std::string("NOT REFUSED")) << "\n";
        StableCertificate core;
        OpModule G = N;
        G.op = {Mat(7, {2, 2}, {7, 0, 0, 7})};   // T = 7 I: A = Z/49, stable at K = 2, eps 0
        LoewyBlock ok = claim;
        ok.system = {0};
        ok.history[0].full = ReadAlgebra(G, {0});
        ok.eps = 2 * static_cast<int64_t>(ok.history[0].full.dim_mod_ell) - ok.history[0].full.length;
        what.clear();
        Check(!Refused([&] { core = CheckStableReading(G, G, ok, 2); }, what) &&
                  core.label.find("core only") != std::string::npos,
              "(C2) the core's label says 'core only' (audit M5): " + core.label + " " + what);
    }
    {   // a perturbed A_K on a planted module: (Z/49)^2 with T = 7 I (A = Z/49, stable at K = 2, eps 0) is certified;
        // the same claim against T = 7 swap (A of length 3, dim 2) is refused.
        auto planted = [](const std::vector<uint64_t>& T) {
            OpModule N;
            N.ell = 7;
            N.exps = {2, 2};
            N.primes = {2};
            N.op = {Mat(7, {2, 2}, T)};
            return N;
        };
        const OpModule good = planted({7, 0, 0, 7}), bad = planted({0, 7, 7, 0});
        LoewyBlock claim;
        claim.system = {0};
        claim.D = 2;
        claim.free_rank = 2;
        claim.rank_T = 1;
        claim.verdict = "stable";
        claim.K_stab = 2;
        LoewyStep st;
        st.K = 2;
        st.full = ReadAlgebra(good, {0});
        claim.history = {st};
        claim.eps = 2 * static_cast<int64_t>(st.full.dim_mod_ell) - st.full.length;
        std::string what;
        Check(!Refused([&] { (void)CheckStableReading(good, good, claim, 2); }, what), "(C2) planted T = 7 I: certified " + what);
        const bool r = Refused([&] { (void)CheckStableReading(good, bad, claim, 2); }, what);
        Check(r, "(C2) planted: the claim of T = 7 I against the perturbed A_K (T = 7 swap) is refused, got '" + what + "'");
        std::cout << "  (C2) planted perturbed A_K: " << (r ? "refused -- " + what.substr(what.find("refuses")) : std::string("NOT REFUSED"))
                  << "\n";
    }
    {   // a perturbed A_K on the cell: T_2 at K = 3 plus ell^{b_i - 1} in every off-diagonal entry (i, j)
        // (still a homomorphism of (+) Z/7^{b}): the operators no longer commute on the block
        OpModule P = N3;
        const uint32_t r = P.rank();
        RingMat& T2 = P.op[0];
        for (uint32_t i = 0; i < r; ++i) {
            uint64_t m = 1, u = 1;
            for (uint32_t k = 0; k < T2.row_exp[i]; ++k) m *= 7;
            u = m / 7;
            for (uint32_t j = 0; j < r; ++j)
                if (i != j) T2.a[static_cast<size_t>(i) * r + j] = (T2.a[static_cast<size_t>(i) * r + j] + u) % m;
        }
        std::string what;
        const bool refused = Refused([&] { (void)CertifyStable(142, Nc, P, *d, c.K_cls); }, what);
        Check(refused, "(C2) (142, 7): the Delta claim against a perturbed T_2 is refused, got '" + what + "'");
        std::cout << "  (C2) (142, 7) perturbed T_2: " << (refused ? "refused -- " + what : std::string("NOT REFUSED")) << "\n";
    }
}

// (C3) the digit route: equality with the u64 route at ell^K < 2^64 (the oracle route
// Digit forces the digit route below 2^64), and one cell beyond 2^64 (ell = 5, K = 28:
// 5^28 > 2^64, the NP5R family) against [C, prop:saturation](3)-(4) / [C, prop:satdefect](2).
void ArmC3() {
    std::cout << "(C3) the digit route: u64 route == digit route below 2^64; (34, 5), (30, 5) at K = 28 beyond 2^64\n";
    struct Pair { uint32_t n; uint64_t ell; uint32_t kmax; };
    for (const Pair& pr : {Pair{58, 7, 3}, Pair{30, 5, 4}, Pair{34, 5, 3}}) {
        LoewyOptions o;
        o.kmax = pr.kmax;
        const LoewyCell a = ComputeRingLoewyCell(pr.n, pr.ell, o);
        o.route = ModuleRoute::Digit;
        const LoewyCell b = ComputeRingLoewyCell(pr.n, pr.ell, o);
        const std::string tag = "(" + std::to_string(pr.n) + ", " + std::to_string(pr.ell) + ")";
        bool same = a.blocks.size() == b.blocks.size() && !a.blocks.empty();
        for (size_t i = 0; same && i < a.blocks.size(); ++i) same = LoewyRingLine(a, a.blocks[i]) == LoewyRingLine(b, b.blocks[i]);
        Check(same, "(C3) " + tag + " KMAX " + std::to_string(pr.kmax) + ": identical LOEWY_RING lines on the u64 and the digit route");
        std::cout << "  " << tag << ": u64 " << a.seconds << " s, digit " << b.seconds << " s, " << a.blocks.size() << " blocks, "
                  << (same ? "identical" : "DIFFER") << "\n";
        const OpModule Nn = RunModule(pr.n, pr.ell, 2, a.primes), Nd = RunModule(pr.n, pr.ell, 2, a.primes, ModuleRoute::Digit);
        std::vector<uint32_t> en = Nn.exps, ed = Nd.exps;
        std::sort(en.begin(), en.end());
        std::sort(ed.begin(), ed.end());
        Check(Nd.is_digit() && !Nn.is_digit() && en == ed, "(C3) " + tag + ": H^1(Z/ell^2) has the same exponents on both routes");
    }
    // beyond 2^64
    for (uint32_t n : {34u, 30u}) {
        const uint64_t ell = 5;
        const std::vector<uint64_t> S = {2, 3, 5, 7};
        const std::string tag = "(" + std::to_string(n) + ", 5)";
        LoewyOptions o;
        o.kmax = 4;
        const LoewyCell c = ComputeRingLoewyCell(n, ell, o);
        const auto t0 = std::chrono::steady_clock::now();
        const OpModule N26 = RunModule(n, ell, 26, S), N27 = RunModule(n, ell, 27, S), N28 = RunModule(n, ell, 28, S);
        const double s28 = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        Check(!N26.is_digit() && !N27.is_digit() && N28.is_digit() && N28.digit_K == 28,
              "(C3) " + tag + ": K = 26, 27 on the u64 route (5^27 < 2^64), K = 28 on the digit route (5^28 > 2^64)");
        auto split = [](const OpModule& N, uint32_t K) {
            uint32_t f = 0;
            std::vector<uint32_t> tor;
            for (uint32_t x : N.exps) {
                if (x == K) ++f;
                else tor.push_back(x);
            }
            std::sort(tor.begin(), tor.end());
            return std::make_pair(f, tor);
        };
        Check(split(N27, 27) == split(N28, 28), "(C3) " + tag + ": H^1(Z/5^28) = the free rank at 28 plus the same torsion as at 27");
        const OpModule Ncls = RunModule(n, ell, c.K_cls, S);
        uint32_t certified = 0, read = 0;
        for (const LoewyBlock& b : c.blocks) {
            if (b.verdict != "stable" && b.verdict != "torsion_free_stable") continue;
            ++read;
            const uint32_t t = b.torsion.empty() ? 0 : b.torsion.front();
            AlgebraReading R[3];
            const OpModule* Ns[3] = {&N26, &N27, &N28};
            for (int i = 0; i < 3; ++i) R[i] = ReadAlgebra(Localize(*Ns[i], b.system).local, b.system, t);
            int64_t eps[3];
            for (int i = 0; i < 3; ++i) eps[i] = static_cast<int64_t>(26 + i - t) * b.rank_T - R[i].length;
            const std::string bt = tag + " (" + [&] {
                std::string s;
                for (size_t i = 0; i < b.system.size(); ++i) s += (i ? "," : "") + std::to_string(b.system[i]);
                return s;
            }() + ")";
            std::cout << "  " << bt << " " << b.verdict << " rank_T=" << b.rank_T << " t=" << t << ": eps_26,27,28 = " << eps[0] << ","
                      << eps[1] << "," << eps[2] << " len_28=" << R[2].length << " hilbert_28=" << HilbertText(R[2].hilbert)
                      << " (K = " << b.K_stab << ": eps " << b.eps << ")\n";
            Check(R[0].dim_mod_ell == b.rank_T && R[1].dim_mod_ell == b.rank_T && R[2].dim_mod_ell == b.rank_T,
                  "(C3) " + bt + ": dim A_K/5 = rank_T at K = 26, 27, 28 ([C, prop:saturation](3): stable at every larger K)");
            const AlgebraReading& Rs = b.history.back().has_free ? b.history.back().free_part : b.history.back().full;
            Check(R[0].hilbert == Rs.hilbert && R[1].hilbert == Rs.hilbert && R[2].hilbert == Rs.hilbert,
                  "(C3) " + bt + ": the Hilbert function at K = 26, 27, 28 is the certified one ([C, prop:saturation](5))");
            Check(eps[0] == eps[1] && eps[1] >= b.eps, "(C3) " + bt + ": eps^sat has reached its plateau by K = 26 (= delta, [C, prop:satdefect](2))");
            Check(R[2].length == static_cast<uint32_t>(static_cast<int64_t>(28 - t) * b.rank_T - eps[1]),
                  "(C3) " + bt + ": length(A_28) = (28 - t) rank_T - delta, predicted from the u64 route ([C, prop:saturation](4))");
            // the certificate on the digit module at K = 28
            LoewyBlock claim = b;
            LoewyStep st;
            st.K = 28;
            if (t > 0) {
                st.has_free = true;
                st.free_level = 28 - t;
                st.free_part = R[2];
                st.full = ReadAlgebra(Localize(N28, b.system).local, b.system);
            } else {
                st.full = R[2];
            }
            claim.history = {st};
            claim.K_stab = 28;
            claim.eps = eps[2];
            std::string what;
            StableCertificate sc;
            const bool refused = Refused([&] { sc = CertifyStable(n, Ncls, N28, claim, c.K_cls); }, what);
            Check(!refused && sc.level == 28 - t && sc.dim == b.rank_T && sc.eps == eps[1],
                  "(C3) " + bt + ": CertifyStable certifies the digit module at K = 28 (" + what + ")");
            if (!refused) {
                ++certified;
                std::cout << "  " << bt << " at K = 28 (digit): stable_cert=certified label=" << sc.label << " level=" << sc.level
                          << " length=" << sc.length << " eps=" << sc.eps << "\n";
            }
        }
        Check(read > 0 && certified == read, "(C3) " + tag + ": a stable block exists and is certified at K = 28");
        std::cout << "  " << tag << " module runs K = 26, 27, 28: " << s28 << " s\n";
    }
}

// ---- the env hook ----------------------------------------------------------
struct HookCell { uint32_t n; uint64_t ell; uint32_t kmax; };

std::vector<HookCell> ParseCells(const std::string& s) {
    std::vector<HookCell> out;
    std::string v = s;
    v.erase(std::remove_if(v.begin(), v.end(), ::isspace), v.end());
    for (const oc_util::cells::Token& t : oc_util::cells::Tokenize(v)) {
        if (t.text.empty()) continue;
        if (!t.Fits(2, 3)) throw std::invalid_argument("HECKE_RING_LOEWY_CELLS: bad item '" + t.text + "' (n:ell[:KMAX])");
        const std::vector<uint64_t>& f = t.fields;
        HookCell c;
        c.n = static_cast<uint32_t>(f[0]);
        c.ell = f[1];
        c.kmax = f.size() == 3 ? static_cast<uint32_t>(f[2]) : (c.ell == 7 ? 7u : 6u);
        out.push_back(c);
    }
    return out;
}

void ArmEnv(const char* env) {
    const std::vector<HookCell> cells = ParseCells(env);
    std::cout << "HECKE_RING_LOEWY_EFFECTIVE cells=" << cells.size() << " source=HECKE_RING_LOEWY_CELLS\n";
    std::map<std::string, uint32_t> verdicts, loewyT, certs;
    uint32_t blocks = 0, nonred = 0, cf9_fail = 0, hidden = 0, tors = 0, tors_nonred = 0, max_loewy = 0, refused = 0;
    for (const HookCell& hc : cells) {
        LoewyOptions o;
        o.kmax = hc.kmax;
        LoewyCell c;
        try {
            c = ComputeRingLoewyCell(hc.n, hc.ell, o);
        } catch (const hecke::NotImplementedInThisBuild& e) {
            ++refused;
            std::cout << "LOEWY_RING_REFUSED n=" << hc.n << " ell=" << hc.ell << " " << e.what() << "\n";
            continue;
        }
        std::cout << "LOEWY_RING_CELL n=" << c.n << " ell=" << c.ell << " kmax=" << c.kmax << " e_max=" << c.e_max
                  << " blocks=" << c.blocks.size() << " seconds=" << c.seconds << " module_seconds=" << c.module_seconds << "\n";
        for (const LoewyBlock& b : c.blocks) {
            std::cout << LoewyRingLine(c, b) << "\n";
            ++blocks;
            ++verdicts[b.verdict];
            ++certs[b.stable_cert];
            Check(b.stable_family() == (b.stable_cert == "certified"), "(ENV) a stable-family verdict is certified, no other");
            if (!b.torsion.empty()) {
                ++tors;
                if (b.nonreduced_A1()) ++tors_nonred;
            }
            if (b.nonreduced_A1()) {
                ++nonred;
                if (!b.cf9_holds()) {
                    ++cf9_fail;
                    std::cout << "LOEWY_RING_CF9_FINDING " << LoewyRingLine(c, b) << "\n";
                }
            }
            if (b.hidden()) ++hidden;
            const AlgebraReading* T = b.last_T();
            if (T && T->dim_mod_ell > 1) {
                ++loewyT[std::to_string(T->loewy())];
                max_loewy = std::max(max_loewy, T->loewy());
            }
            Check(b.A1.dim_mod_ell <= (T ? T->dim_mod_ell : 1), "(ENV) A_1 is a quotient of A_K/ell");
        }
    }
    auto dict = [](const std::map<std::string, uint32_t>& m) {
        std::string s = "{";
        bool first = true;
        for (const auto& kv : m) {
            s += (first ? "" : ",") + kv.first + ":" + std::to_string(kv.second);
            first = false;
        }
        return s + "}";
    };
    std::cout << "LOEWY_RING_SUMMARY blocks=" << blocks << " cells=" << cells.size() - refused << " refused=" << refused
              << " verdicts=" << dict(verdicts) << " nonreduced_A1=" << nonred << " cf9_fail=" << cf9_fail << " hidden=" << hidden
              << " torsion_blocks=" << tors << " torsion_nonreduced_A1=" << tors_nonred << " loewy_T=" << dict(loewyT)
              << " max_loewy_T=" << max_loewy << " stable_cert=" << dict(certs) << "\n";   // slot-15 WP-5: one appended field
}

} // namespace

uint32_t HeckeRingLoewyValidation() {
    Check.reset();
    g_cells.clear();
    g_modules.clear();
    const auto t0 = std::chrono::steady_clock::now();
    std::cout << "=== HeckeRingLoewyValidation (slot-14 WP-6: the Z/ell^K Hecke-algebra route, [M1] route R1) ===\n";
    try {
        const char* env = std::getenv("HECKE_RING_LOEWY_CELLS");
        if (env && *env) {
            ArmEnv(env);
        } else {
            ArmP();
            ArmA();
            ArmBC();
            ArmD();
            ArmE();
            ArmW();
            ArmC1();   // slot-15 WP-5
            ArmC2();
            // (C3) moved to HeckeRingLoewyDigitSlowValidation (slow tier, S15-O1)
        }
    } catch (const std::exception& e) {
        Check(false, std::string("exception: ") + e.what());
    }
    g_cells.clear();
    g_modules.clear();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "HeckeRingLoewyValidation: " << Check.checks << " checks, " << Check.fails << " failing, " << s << " s\n";
    return Check.fails == 0 ? 0 : 1;
}

// (S15-O1, owner 2026-10-01) arm (C3), the digit route (u64 == digit, ell = 5, K = 28 beyond 2^64),
// split off as its own `slow` name; the checks and values are unchanged.
uint32_t HeckeRingLoewyDigitSlowValidation() {
    Check.reset();
    g_cells.clear();
    g_modules.clear();
    const auto t0 = std::chrono::steady_clock::now();
    std::cout << "=== HeckeRingLoewyDigitSlowValidation (S15-O1: arm (C3) of HeckeRingLoewyValidation) ===\n";
    try {
        ArmC3();
    } catch (const std::exception& e) {
        Check(false, std::string("exception: ") + e.what());
    }
    g_cells.clear();
    g_modules.clear();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "HeckeRingLoewyDigitSlowValidation: " << Check.checks << " checks, " << Check.fails << " failing, " << s << " s\n";
    return Check.fails == 0 ? 0 : 1;
}
