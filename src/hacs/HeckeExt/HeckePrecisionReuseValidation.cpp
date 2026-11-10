// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 23, G1) HeckePrecisionReuseValidation (standard) and HeckePrecisionReuseSlowValidation
// (slow): the gate of [C, obl:precisionreuse](v) (v1.23.0-pre.3, docs/implementation/slot23/G1_clause_note.md).
//
// The route under test: ringalg::LevelCache, used by ComputeRingLoewyCell (HeckeRingHeckeAlgebra) and
// ComputeSaturationDefectCell (HeckeSaturationDefect) -- one module run per cell, every level read below the
// latest run derived as M_{K*}|K ([C, lem:precisionreduction](2)).  The oracle: the per-K route (one module
// run per level read, the route through v1.23.0-pre.2), selected by ScopedPrecisionReuse(false) -- the same
// switch as the rollback arm HECKE_PRECISION_REUSE=0.  Both scopes are set here, so the validation does not
// depend on the environment.
//
// Per grid cell: the whole printed output of the two routes (the LOEWY_RING / SAT_DEFECT lines and the cell
// header) is compared byte for byte; K_run (the levels read, [C, obl:precisionreuse](vi)) must be equal;
// timings and Lv_run (the levels run) are excepted, Lv_run is checked for what the policy says it is.
//
// Grid (standard), each item of [C, obl:precisionreuse](v):
//   (G1) (108, 7) KMAX 6, Loewy + sat: K_stab = 3 with a plateau (sat eps = [2,3,3]) -- the standard tier's cell
//        in place of (300, 7) -- and a Dickson torsion block; one run at K* = 6 (ell^6 < 2^32), 2 and 3 derived;
//   (G2) (158, 7) KMAX 3, Loewy: a not_stable_at_KMAX block (its K_stab is 4 at KMAX 6);
//   (G3) (108, 7) KMAX 6, Loewy, policy Lazy: a restart (first run at K_cls = 2, the request for 3 reruns at 6);
//   (G4) (10, 691) KMAX 4, Loewy, policies Auto and Max: run levels with ell^K* < 2^32 and >= 2^32 (691^4);
//   (G5) (34, 5) KMAX 3, Loewy on ModuleRoute::Digit (native_range, as HeckeRingLoewyValidation (C3)), derived to
//        K = 2 < 2^64, against the per-K route's NATIVE reading;
//   (G6) (10, 37), (24, 41), Loewy + sat: only reduced_rank1 blocks at ell >= 37 (one run, nothing derived);
//   (G7) (40, 5) KMAX 3, Loewy + sat: a Dickson-locus torsion block (torsion_free_stable; refused:torsion_block);
//   (G8) (108, 7), Loewy, an injected failure of the run above the request (ScopedInjectRunFailure, as a cap
//        would): the cell drops to the per-K route for the levels still unread ([C, obl:precisionreuse](i));
//   (N)  the checks of [C, obl:precisionreuse](iii) on (108, 7)'s module at K = 3: the positive case, K = 1 against
//        ReduceModEll entry for entry, and the two negative controls -- a derived module keeping a_i in place of
//        min(a_i, K) (must fire the "exponent" check) and one entry of T_p on a block of dimension >= 2 changed by
//        ell^{K-1} where ell^{K-1}[E_ij, T_q] != 0 (must fire the "commute" check); a control that passes fails.
//        (G1 audit F3) two further negative controls on M_3|2: an entry changed by +1 mod ell^{a_i} (must fire the
//        "ReduceModEll" check, derived against run) and an entry pushed out of [0, ell^{a_i}) (must fire "range");
//   (N2) (G1 audit F2) the digit -> native branch of ReduceToLevel at K >= 2: a ModuleRoute::Digit run of (34, 5)
//        at K = 3 reduced to K = 2 in both forms -- the native entries are the digit sums entry for entry, both pass
//        (iii), and the block and ReadAlgebra agree on the two forms for every joint system;
//   (G9) (G1 audit F1) (60, 5) KMAX 6, Loewy + sat: K_cls = 3 (e_max = 2), so the derived N_cls keeps torsion
//        summands of exponent 2 while the free summands are cut from K* = 6 to K_cls; a certified block.
// Slow: (300, 7) KMAX 6 on the sat route (the D = 24 block, eps = [9,16,21,25,27,27]) and (300, 11) (two levels).

#include "HeckePrecisionReuseValidation.h"

#include "HeckeRingHeckeAlgebra.h"
#include "HeckeSaturationDefect.h"
#include "../Util/ValidationCheck.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace ra = hecke_ext::ringalg;
namespace sd = hecke_ext::satdefect;

oc_util::Checker Check("  FAIL: ", false);

std::string Txt(const std::vector<uint32_t>& v) {
    std::ostringstream s;
    s << "[";
    for (size_t i = 0; i < v.size(); ++i) s << (i ? "," : "") << v[i];
    return s.str() + "]";
}
std::string Txt64(const std::vector<uint64_t>& v) {
    std::ostringstream s;
    for (size_t i = 0; i < v.size(); ++i) s << (i ? "," : "") << v[i];
    return s.str();
}

struct Out {
    std::string text;                    // the printed output, timings excepted
    std::vector<uint32_t> K_run, Lv_run;
    std::string mode;
    double seconds = 0.0, module_seconds = 0.0;
    uint32_t K_cls = 0, kneed = 0;
    std::vector<std::string> verdicts;   // per block (Loewy verdict / sat status)
    std::vector<uint32_t> K_stab;
    std::vector<std::vector<uint32_t>> eps;   // sat only
    std::vector<bool> sat_final;              // sat only
    std::vector<std::pair<uint32_t, uint32_t>> cert_from;   // per certified block
    std::vector<uint32_t> cert_level;                       // K_cls / the claim's level, per certified block
};

struct Scope {
    std::unique_ptr<ra::ScopedPrecisionReuse> reuse;
    std::unique_ptr<ra::ScopedRunLevelPolicy> policy;
    std::unique_ptr<ra::ScopedInjectRunFailure> inject;
    Scope(bool derived, ra::RunLevelPolicy p, bool inj) {
        reuse.reset(new ra::ScopedPrecisionReuse(derived));
        policy.reset(new ra::ScopedRunLevelPolicy(p));
        if (inj) inject.reset(new ra::ScopedInjectRunFailure());
    }
};

Out RunLoewy(uint32_t n, uint64_t ell, uint32_t kmax, ra::ModuleRoute route, bool derived,
             ra::RunLevelPolicy pol = ra::RunLevelPolicy::Auto, bool inject = false) {
    const Scope s(derived, pol, inject);
    ra::LoewyOptions o;
    o.kmax = kmax;
    o.route = route;
    const ra::LoewyCell c = ra::ComputeRingLoewyCell(n, ell, o);
    Out r;
    std::ostringstream os;
    os << "LOEWY_CELL n=" << c.n << " ell=" << c.ell << " kmax=" << c.kmax << " primes={" << Txt64(c.primes) << "} e_max=" << c.e_max
       << " K_cls=" << c.K_cls << " blocks=" << c.blocks.size() << "\n";
    for (const ra::LoewyBlock& b : c.blocks) {
        os << ra::LoewyRingLine(c, b) << "\n";
        r.verdicts.push_back(b.verdict);
        r.K_stab.push_back(b.K_stab);
        if (b.stable_cert == "certified") {
            r.cert_from.push_back({b.cert_from_cls, b.cert_from_K});
            r.cert_level.push_back(b.verdict == "reduced_rank1" ? c.K_cls : b.K_stab);
        }
    }
    r.text = os.str();
    r.K_run = c.K_run;
    r.Lv_run = c.Lv_run;
    r.mode = c.level_mode;
    r.seconds = c.seconds;
    r.module_seconds = c.module_seconds;
    r.K_cls = c.K_cls;
    r.kneed = std::max(c.kmax, c.K_cls);
    return r;
}

Out RunSat(uint32_t n, uint64_t ell, uint32_t kmax, bool derived, ra::RunLevelPolicy pol = ra::RunLevelPolicy::Auto) {
    const Scope s(derived, pol, false);
    sd::SatOptions o;
    o.kmax = kmax;
    const sd::SatCell c = sd::ComputeSaturationDefectCell(n, ell, o);
    Out r;
    std::ostringstream os;
    os << "SAT_CELL n=" << c.n << " ell=" << c.ell << " kmax=" << c.kmax << " primes={" << Txt64(c.primes) << "} e_max=" << c.e_max
       << " K_cls=" << c.K_cls << " blocks=" << c.blocks.size() << "\n";
    for (const sd::SatBlock& b : c.blocks) {
        os << sd::SatDefectLine(c, b) << "\n";
        r.verdicts.push_back(b.status);
        r.K_stab.push_back(b.K_stab);
        r.eps.push_back(b.eps);
        r.sat_final.push_back(b.sat_final);
        if (b.stable_cert == "certified") {
            r.cert_from.push_back({b.cert_from_cls, b.cert_from_K});
            r.cert_level.push_back(b.verdict == "reduced_rank1" ? c.K_cls : b.K_stab);
        }
    }
    r.text = os.str();
    r.K_run = c.K_run;
    r.Lv_run = c.Lv_run;
    r.mode = c.level_mode;
    r.seconds = c.seconds;
    r.module_seconds = c.module_seconds;
    r.K_cls = c.K_cls;
    r.kneed = std::max(c.kmax, c.K_cls);
    return r;
}

std::string FirstDiff(const std::string& a, const std::string& b) {
    std::istringstream x(a), y(b);
    std::string l1, l2;
    for (int i = 1;; ++i) {
        const bool g1 = static_cast<bool>(std::getline(x, l1)), g2 = static_cast<bool>(std::getline(y, l2));
        if (!g1 && !g2) return "(none)";
        if (!g1 || !g2 || l1 != l2)
            return "line " + std::to_string(i) + ":\n    per_K:   " + (g1 ? l1 : "<end>") + "\n    derived: " + (g2 ? l2 : "<end>");
    }
}

bool StartsWith(const std::string& s, const std::string& p) { return s.compare(0, p.size(), p) == 0; }

// The comparison of [C, obl:precisionreuse](v) and the record checks of (vi), (vii).
void Compare(const std::string& tag, const Out& oracle, const Out& d, bool expect_reuse) {
    std::cout << "PRECISION_REUSE_CELL " << tag << " K_run=" << Txt(oracle.K_run) << " Lv_run(per_K)=" << Txt(oracle.Lv_run)
              << " Lv_run(derived)=" << Txt(d.Lv_run) << " mode=" << d.mode << " seconds per_K=" << oracle.seconds << " derived=" << d.seconds
              << " module_seconds per_K=" << oracle.module_seconds << " derived=" << d.module_seconds << "\n";
    const bool same = oracle.text == d.text;
    Check(same, "(v) " + tag + ": the printed output of the derived route is byte-identical to the per-K route's");
    if (!same) std::cout << "  first difference at " << FirstDiff(oracle.text, d.text) << "\n";
    Check(oracle.K_run == d.K_run, "(vi) " + tag + ": K_run (the levels read) is byte-stable: " + Txt(oracle.K_run) + " vs " + Txt(d.K_run));
    Check(StartsWith(oracle.mode, "per_K:rollback") && oracle.Lv_run == oracle.K_run,
          "(v) " + tag + ": the oracle is the per-K route (one run per level read), mode " + oracle.mode);
    for (const auto& cf : oracle.cert_from) Check(cf.first != 0 && cf.second != 0, "(vii) " + tag + ": the oracle records its certificate's run levels");
    if (expect_reuse) {
        Check(StartsWith(d.mode, "reuse:"), "(i) " + tag + ": the derived route is in scope (mode " + d.mode + ")");
        for (uint32_t k : d.Lv_run)
            Check(k >= d.K_cls && k <= d.kneed, "(ii) " + tag + ": every run level lies in [K_cls, K_need]");
        Check(!d.Lv_run.empty() && d.Lv_run.back() >= (d.K_run.empty() ? 0u : d.K_run.back()),
              "(ii) " + tag + ": the levels read lie below the largest run (downward only)");
        Check(d.module_seconds <= d.seconds + 1e-9, "(vi) " + tag + ": module_seconds counts the runs only (<= the cell's seconds)");
        for (size_t i = 0; i < d.cert_from.size(); ++i)
            Check(d.cert_from[i].first >= oracle.K_cls && d.cert_from[i].second >= d.cert_level[i] &&
                      std::find(d.Lv_run.begin(), d.Lv_run.end(), d.cert_from[i].second) != d.Lv_run.end(),
                  "(vii) " + tag + ": a certificate's modules are recorded with the run level they were derived from");
    }
}

bool Has(const std::vector<std::string>& v, const std::string& x) { return std::find(v.begin(), v.end(), x) != v.end(); }

void ArmGrid() {
    using P = ra::RunLevelPolicy;
    // (G1) K_stab = 3 with a plateau, a Dickson torsion block; one run at 6, 2 and 3 derived
    std::cout << "(G1) (108, 7) KMAX 6: Loewy + sat, K_stab = 3 with a plateau, one run at K* = 6\n";
    const Out L108 = RunLoewy(108, 7, 6, ra::ModuleRoute::Native, false);
    {
        const Out d = RunLoewy(108, 7, 6, ra::ModuleRoute::Native, true);
        Compare("(G1) Loewy (108, 7)", L108, d, true);
        Check(*std::max_element(L108.K_stab.begin(), L108.K_stab.end()) >= 3, "(G1) (108, 7): a block with K_stab >= 3 (non-vacuity)");
        Check(d.Lv_run == std::vector<uint32_t>({6}) && L108.K_run.size() >= 2, "(G1) (108, 7) Loewy: Lv_run = [6], two or more levels derived");
        Check(Has(L108.verdicts, "torsion_free_stable") || Has(L108.verdicts, "torsion_stable"), "(G1) (108, 7): a Dickson torsion block");
    }
    {
        const Out o = RunSat(108, 7, 6, false), d = RunSat(108, 7, 6, true);
        Compare("(G1) sat (108, 7)", o, d, true);
        bool plateau3 = false;
        for (size_t i = 0; i < o.eps.size(); ++i)
            if (o.verdicts[i] == "ok" && o.K_stab[i] >= 3 && o.sat_final[i]) plateau3 = true;
        Check(plateau3, "(G1) (108, 7) sat: a block stable at K >= 3 with a final plateau (non-vacuity)");
        Check(Has(o.verdicts, "refused:torsion_block"), "(G1) (108, 7) sat: refused:torsion_block present");
        Check(d.Lv_run == std::vector<uint32_t>({6}), "(G1) (108, 7) sat: Lv_run = [6]");
    }
    // (G2) a not_stable_at_KMAX block
    std::cout << "(G2) (158, 7) KMAX 3: Loewy, a not_stable_at_KMAX block\n";
    {
        const Out o = RunLoewy(158, 7, 3, ra::ModuleRoute::Native, false), d = RunLoewy(158, 7, 3, ra::ModuleRoute::Native, true);
        Compare("(G2) Loewy (158, 7) KMAX 3", o, d, true);
        Check(Has(o.verdicts, "not_stable_at_KMAX"), "(G2) (158, 7) KMAX 3: a not_stable_at_KMAX block (non-vacuity)");
        Check(d.Lv_run == std::vector<uint32_t>({3}) && o.K_run == std::vector<uint32_t>({2, 3}), "(G2) (158, 7): Lv_run = [3], K_run = [2,3]");
    }
    // (G3) a restart
    std::cout << "(G3) (108, 7) KMAX 6: Loewy, policy lazy -- a restart\n";
    {
        const Out d = RunLoewy(108, 7, 6, ra::ModuleRoute::Native, true, P::Lazy);
        Compare("(G3) Loewy (108, 7) lazy", L108, d, true);
        Check(d.Lv_run.size() >= 2 && d.Lv_run.front() == L108.K_cls && d.Lv_run.back() == 6,
              "(G3) (108, 7) lazy: a first run at K_cls and a restart at 6, Lv_run = " + Txt(d.Lv_run));
        const Out a = RunLoewy(108, 7, 6, ra::ModuleRoute::Native, true, P::AtRequest);
        Compare("(G3) Loewy (108, 7) at_request", L108, a, true);
        Check(a.Lv_run == L108.K_run, "(G3) (108, 7) at_request: one restart per level read above K*");
    }
    // (G4) the u32 and the u64 word
    std::cout << "(G4) (10, 691) KMAX 4: Loewy, run levels below and above 2^32\n";
    {
        const Out o = RunLoewy(10, 691, 4, ra::ModuleRoute::Native, false);
        const Out a = RunLoewy(10, 691, 4, ra::ModuleRoute::Native, true, P::Auto);
        const Out m = RunLoewy(10, 691, 4, ra::ModuleRoute::Native, true, P::Max);
        Compare("(G4) Loewy (10, 691) auto", o, a, true);
        Compare("(G4) Loewy (10, 691) max", o, m, true);
        Check(!a.Lv_run.empty() && a.Lv_run.back() <= 3, "(G4) (10, 691) auto: every run in the u32 word (691^3 < 2^32)");
        Check(m.Lv_run == std::vector<uint32_t>({4}), "(G4) (10, 691) max: one run at K* = 4, 691^4 >= 2^32 (the u64 word)");
        Check(o.K_run.size() >= 1 && o.K_run.front() == 2, "(G4) (10, 691): K = 2 is read, derived from the u64 run");
    }
    // (G5) the digit route, derived to a level below 2^64, against the per-K route's native reading
    std::cout << "(G5) (34, 5) KMAX 3: Loewy on ModuleRoute::Digit (native_range) derived, against the per-K native route\n";
    {
        const Out o = RunLoewy(34, 5, 3, ra::ModuleRoute::Native, false);
        const Out d = RunLoewy(34, 5, 3, ra::ModuleRoute::Digit, true);
        Compare("(G5) Loewy (34, 5) digit", o, d, true);
        Check(d.Lv_run == std::vector<uint32_t>({3}) && std::find(o.K_run.begin(), o.K_run.end(), 2u) != o.K_run.end(),
              "(G5) (34, 5): one digit run at K* = 3, K = 2 derived from it");
    }
    // (G6) only reduced_rank1 blocks at ell >= 37
    for (const auto& c : std::vector<std::pair<uint32_t, uint64_t>>{{10, 37}, {24, 41}}) {
        const std::string t = "(" + std::to_string(c.first) + ", " + std::to_string(c.second) + ")";
        std::cout << "(G6) " << t << " KMAX 6: Loewy + sat, only reduced_rank1 blocks\n";
        const Out o = RunLoewy(c.first, c.second, 6, ra::ModuleRoute::Native, false), d = RunLoewy(c.first, c.second, 6, ra::ModuleRoute::Native, true);
        Compare("(G6) Loewy " + t, o, d, true);
        bool all = !o.verdicts.empty();
        for (const std::string& v : o.verdicts) all = all && v == "reduced_rank1";
        Check(all, "(G6) " + t + ": every block is reduced_rank1 (non-vacuity)");
        Check(d.Lv_run == o.K_run && o.K_run == std::vector<uint32_t>({o.K_cls}), "(G6) " + t + ": one run at K_cls, nothing above it");
        const Out so = RunSat(c.first, c.second, 6, false), sdv = RunSat(c.first, c.second, 6, true);
        Compare("(G6) sat " + t, so, sdv, true);
    }
    // (G7) a Dickson-locus torsion block
    std::cout << "(G7) (40, 5) KMAX 3: Loewy + sat, a Dickson-locus torsion block\n";
    {
        const Out o = RunLoewy(40, 5, 3, ra::ModuleRoute::Native, false), d = RunLoewy(40, 5, 3, ra::ModuleRoute::Native, true);
        Compare("(G7) Loewy (40, 5)", o, d, true);
        Check(Has(o.verdicts, "torsion_free_stable"), "(G7) (40, 5): a torsion_free_stable block");
        const Out so = RunSat(40, 5, 3, false), sdv = RunSat(40, 5, 3, true);
        Compare("(G7) sat (40, 5)", so, sdv, true);
        Check(Has(so.verdicts, "refused:torsion_block"), "(G7) (40, 5) sat: refused:torsion_block");
    }
    // (G9) K_cls = 3: torsion summands of exponent 2 kept, free summands cut from K* to K_cls (G1 audit F1)
    std::cout << "(G9) (60, 5) KMAX 6: Loewy + sat, K_cls = 3 (e_max = 2)\n";
    {
        const Out o = RunLoewy(60, 5, 6, ra::ModuleRoute::Native, false), d = RunLoewy(60, 5, 6, ra::ModuleRoute::Native, true);
        Compare("(G9) Loewy (60, 5)", o, d, true);
        Check(o.K_cls == 3, "(G9) (60, 5): K_cls = 3 (non-vacuity), K_cls = " + std::to_string(o.K_cls));
        Check(!o.cert_from.empty() && !d.cert_from.empty(), "(G9) (60, 5) Loewy: a certified block on both routes");
        Check(d.Lv_run == std::vector<uint32_t>({6}) && o.K_run.size() >= 2 && o.K_run.front() == 2,
              "(G9) (60, 5) Loewy: one run at K* = 6, the levels read " + Txt(o.K_run) + " derived");
        const Out so = RunSat(60, 5, 6, false), sdv = RunSat(60, 5, 6, true);
        Compare("(G9) sat (60, 5)", so, sdv, true);
        Check(so.K_cls == 3 && sdv.Lv_run == std::vector<uint32_t>({6}), "(G9) (60, 5) sat: K_cls = 3, Lv_run = [6]");
    }
    // (G8) a failure of a run above the request: the per-K route for the levels still unread
    std::cout << "(G8) (108, 7) KMAX 6: an injected failure of the run above the request\n";
    {
        const Out a = RunLoewy(108, 7, 6, ra::ModuleRoute::Native, true, P::Auto, /*inject=*/true);
        Compare("(G8) Loewy (108, 7) auto+failure", L108, a, false);
        Check(StartsWith(a.mode, "per_K:failure_at_K=6") && a.Lv_run == L108.K_run,
              "(G8) (108, 7) auto: the first run (at 6 > 2) fails, every level is then run per K; mode " + a.mode);
        const Out z = RunLoewy(108, 7, 6, ra::ModuleRoute::Native, true, P::Lazy, /*inject=*/true);
        Compare("(G8) Loewy (108, 7) lazy+failure", L108, z, false);
        Check(StartsWith(z.mode, "per_K:failure_at_K=6") && z.Lv_run == L108.K_run,
              "(G8) (108, 7) lazy: the run at K_cls = 2 is kept, the restart at 6 fails, 3 is run per K; mode " + z.mode);
    }
}

// (N) the checks of [C, obl:precisionreuse](iii) and their negative controls
void ArmChecks() {
    std::cout << "(N) [C, obl:precisionreuse](iii) on (108, 7) at K = 3: the checks and the four negative controls\n";
    const ra::ScopedPrecisionReuse on(true);
    const ra::ScopedRunLevelPolicy pol(ra::RunLevelPolicy::Max);
    const std::vector<uint64_t> primes = {2, 3, 5, 7};
    ra::LevelCache L(108, 7, primes, ra::ModuleRoute::Native, 2, 3);
    const ra::OpModule& d2 = L.get(2);   // the run at K* = 3, then 2 derived
    const ra::OpModule& run = L.get(3);  // M_3 | 3 = the run module itself
    Check(L.levels_run() == std::vector<uint32_t>({3}) && L.source(2) == 3, "(N) one run at 3, K = 2 derived from it");
    uint32_t top = 0;
    for (uint32_t a : run.exps) top = std::max(top, a);
    Check(top == 3, "(N) the run has a summand of order 7^3 (so K = 2 < max a_i)");
    bool exps_ok = d2.rank() == run.rank();
    for (uint32_t i = 0; exps_ok && i < run.rank(); ++i) exps_ok = d2.exps[i] == std::min(run.exps[i], 2u);
    Check(exps_ok, "(N) M_3|2 has the exponents min(a_i, 2), D summands");
    bool threw = false;
    try { ra::CheckDerivedModule(run, ra::ReduceToLevel(run, 2), 2); } catch (const std::logic_error&) { threw = true; }
    Check(!threw, "(N) the positive case: M_3|2 passes (iii)");
    {
        const ra::OpModule one = ra::ReduceToLevel(run, 1, true), ref = ra::ReduceModEll(run);
        bool same = one.exps == ref.exps && one.op.size() == ref.op.size();
        for (size_t i = 0; same && i < one.op.size(); ++i) same = one.op[i] == ref.op[i];
        Check(same, "(N) K = 1: ReduceToLevel(M_3, 1) equals ReduceModEll entry for entry");
    }
    threw = false;
    {
        ra::OpModule same4;
        try { same4 = ra::ReduceToLevel(run, 4); } catch (const std::logic_error&) { threw = true; }
        bool same = !threw && same4.exps == run.exps && same4.op.size() == run.op.size();
        for (size_t i = 0; same && i < run.op.size(); ++i) same = same4.op[i] == run.op[i];
        Check(same, "(N) ReduceToLevel at K above every exponent of a native module is the module itself (nothing derived upwards)");
    }
    // negative control 1: keep a_i in place of min(a_i, K) at K = 2 < max a_i -- must fire the exponent check
    {
        ra::OpModule bad = ra::ReduceToLevel(run, 2);
        bad.exps = run.exps;
        bad.op = run.op;
        std::string msg;
        try { ra::CheckDerivedModule(run, bad, 2); } catch (const std::logic_error& e) { msg = e.what(); }
        std::cout << "  (NC1) " << (msg.empty() ? std::string("NOT CAUGHT") : msg) << "\n";
        Check(msg.find("(iii) exponent") != std::string::npos, "(NC1) a derived module keeping a_i is caught by (iii)'s exponent check");
    }
    // negative control 2: T_p's entry (i, j) + ell^{K-1} on a block of dimension >= 2, rows of exponent K, where
    // ell^{K-1}[E_ij, T_q] != 0 for some q != p -- must fire the commutation check
    {
        bool done = false;
        for (const ra::JointSystem& js : ra::RationalSystems(d2)) {
            if (done || js.dim < 2) continue;
            ra::Block B = ra::Localize(d2, js.a);
            const uint32_t D = B.local.rank(), K = 2;
            bool free = D >= 2;
            for (uint32_t x : B.local.exps) free = free && x == K;
            if (!free) continue;
            threw = false;
            try { ra::CheckLevelModule(B.local, D, K); } catch (const std::logic_error&) { threw = true; }
            Check(!threw, "(NC2) the unperturbed block passes (iii)");
            const uint64_t ell = 7, q2 = 49;
            for (size_t p = 0; p < B.local.op.size() && !done; ++p)
                for (size_t q = 0; q < B.local.op.size() && !done; ++q) {
                    if (q == p) continue;
                    const ra::RingMat& Tq = B.local.op[q];
                    for (uint32_t i = 0; i < D && !done; ++i)
                        for (uint32_t j = 0; j < D && !done; ++j) {
                            // [E_ij, T_q]_{rc} = delta_{ri} Tq[j][c] - delta_{cj} Tq[r][i]; non-zero mod ell somewhere?
                            bool nz = false;
                            for (uint32_t r = 0; r < D && !nz; ++r)
                                for (uint32_t c = 0; c < D && !nz; ++c) {
                                    const int64_t v = static_cast<int64_t>(r == i ? Tq.at(j, c) % ell : 0) -
                                                      static_cast<int64_t>(c == j ? Tq.at(r, i) % ell : 0);
                                    nz = ((v % 7) + 7) % 7 != 0;
                                }
                            if (!nz) continue;
                            ra::Block P = B;
                            uint64_t& x = P.local.op[p].a[static_cast<size_t>(i) * D + j];
                            x = (x + ell) % q2;   // + ell^{K-1}
                            std::string msg;
                            try { ra::CheckLevelModule(P.local, D, K); } catch (const std::logic_error& e) { msg = e.what(); }
                            std::cout << "  (NC2) system (" << Txt64(js.a) << ") D=" << D << " T_" << B.local.primes[p] << "[" << i << "," << j
                                      << "] += 7, against T_" << B.local.primes[q] << ": " << (msg.empty() ? std::string("NOT CAUGHT") : msg) << "\n";
                            Check(msg.find("(iii) commute") != std::string::npos, "(NC2) the perturbed entry is caught by (iii)'s commutation check");
                            done = true;
                        }
                }
        }
        Check(done, "(NC2) a block of dimension >= 2 with a perturbable entry was found (non-vacuity)");
    }
    // (G1 audit F3) negative controls 3 and 4: the checks that tie a derived module to its run
    {
        uint32_t row = run.rank();
        for (uint32_t i = 0; i < d2.rank() && row == run.rank(); ++i)
            if (d2.exps[i] == 2) row = i;
        Check(row < run.rank(), "(NC3/NC4) M_3|2 has a row of exponent 2 (non-vacuity)");
        if (row < run.rank()) {
            const uint64_t q2 = 49;
            const size_t at = static_cast<size_t>(row) * d2.rank();
            ra::OpModule c3 = d2;
            c3.op[0].a[at] = (c3.op[0].a[at] + 1) % q2;   // + 1: a different residue mod ell
            std::string msg;
            try { ra::CheckDerivedModule(run, c3, 2); } catch (const std::logic_error& e) { msg = e.what(); }
            std::cout << "  (NC3) " << (msg.empty() ? std::string("NOT CAUGHT") : msg) << "\n";
            Check(msg.find("(iii) ReduceModEll") != std::string::npos, "(NC3) an entry changed mod ell is caught by (iii)'s ReduceModEll check");
            ra::OpModule c4 = d2;
            c4.op[0].a[at] += q2;   // the same residue mod ell, outside [0, ell^{a_i})
            msg.clear();
            try { ra::CheckDerivedModule(run, c4, 2); } catch (const std::logic_error& e) { msg = e.what(); }
            std::cout << "  (NC4) " << (msg.empty() ? std::string("NOT CAUGHT") : msg) << "\n";
            Check(msg.find("(iii) range") != std::string::npos, "(NC4) an entry outside [0, ell^{a_i}) is caught by (iii)'s range check");
        }
    }
    // (N2) (G1 audit F2) the digit -> native branch of ReduceToLevel at K = 2 >= 2
    std::cout << "(N2) (34, 5): a ModuleRoute::Digit run at K = 3, reduced to K = 2 in the digit and the native form\n";
    {
        const std::vector<uint64_t> pr5 = {2, 3, 5, 7};
        const ra::OpModule N = ra::RunModule(34, 5, 3, pr5, ra::ModuleRoute::Digit);
        const ra::OpModule Nd = ra::ReduceToLevel(N, 2, /*native_out=*/false), Nn = ra::ReduceToLevel(N, 2, /*native_out=*/true);
        Check(N.is_digit() && Nd.is_digit() && Nd.digit_K == 2 && !Nn.is_digit(), "(N2) the run is a digit module; the two forms at K = 2");
        const size_t r = N.rank();
        bool multi = false;
        for (uint32_t a : Nn.exps) multi = multi || a == 2;
        Check(multi, "(N2) a row of exponent 2 (two digits summed, non-vacuity)");
        bool same = Nd.exps == Nn.exps && Nd.dop.size() == Nn.op.size() && Nn.op.size() == pr5.size();
        for (size_t p = 0; same && p < Nn.op.size(); ++p)
            for (size_t k = 0; same && k < r * r; ++k) {
                uint64_t v = 0, w = 1;
                for (uint32_t dgt = 0; dgt < 2; ++dgt, w *= 5) v += static_cast<uint64_t>(Nd.dop[p][k * 2 + dgt]) * w;
                same = Nn.op[p].a[k] == v && Nn.op[p].row_exp == Nn.exps;
            }
        Check(same, "(N2) the native form is the digit sum of the digit form, entry for entry");
        bool pass = true;
        try { ra::CheckDerivedModule(N, Nn, 2); ra::CheckDerivedModule(N, Nd, 2); } catch (const std::logic_error& e) {
            pass = false;
            std::cout << "  (N2) " << e.what() << "\n";
        }
        Check(pass, "(N2) both forms pass (iii) against the digit run");
        const std::vector<ra::JointSystem> sys = ra::RationalSystems(Nd);
        Check(!sys.empty(), "(N2) at least one joint system");
        bool agree = true;
        for (const ra::JointSystem& js : sys) {
            const ra::Block bn = ra::Localize(Nn, js.a), bd = ra::Localize(Nd, js.a);   // ReadAlgebra reads a block
            const ra::AlgebraReading x = ra::ReadAlgebra(bn.local, js.a), y = ra::ReadAlgebra(bd.local, js.a);
            agree = agree && bn.local.exps == bd.local.exps && x.length == y.length && x.dim_mod_ell == y.dim_mod_ell &&
                    x.hilbert == y.hilbert;
        }
        Check(agree, "(N2) ReadAlgebra agrees on the digit and the native form for every joint system (" + std::to_string(sys.size()) + ")");
    }
}

}  // namespace

uint32_t HeckePrecisionReuseValidation() {
    Check.reset();
    const auto t0 = std::chrono::steady_clock::now();
    std::cout << "=== HeckePrecisionReuseValidation (slot-23 G1: [C, obl:precisionreuse](v), the derived levels against the per-K route) ===\n";
    std::cout << "  " << ra::PrecisionReuseEffectiveLine() << "\n";
    {
        const ra::ScopedPrecisionReuse off(false);
        const std::string e = ra::PrecisionReuseEffectiveLine();
        Check(e.find("levels=per_K source=scoped") != std::string::npos, "(v) the rollback arm's EFFECTIVE line: " + e);
    }
    {
        const ra::ScopedPrecisionReuse on(true);
        const std::string e = ra::PrecisionReuseEffectiveLine();
        Check(e.find("levels=derived source=scoped") != std::string::npos, "(v) the derived route's EFFECTIVE line: " + e);
    }
    {
        const ra::ScopedPrecisionReuse on(true);
        ra::LevelCache h(10, 37, {2, 3, 5, 7, 37}, ra::ModuleRoute::Hida, 2, 3);
        Check(h.mode() == "per_K:route_hida", "(i) ModuleRoute::Hida takes the per-K route, by name");
    }
    try {
        ArmChecks();
        ArmGrid();
    } catch (const std::exception& e) {
        Check(false, std::string("an arm threw: ") + e.what());
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "HeckePrecisionReuseValidation: " << Check.checks << " checks, " << Check.fails << " failing, " << s << " s\n";
    return Check.fails ? 1u : 0u;
}

uint32_t HeckePrecisionReuseSlowValidation() {
    Check.reset();
    const auto t0 = std::chrono::steady_clock::now();
    std::cout << "=== HeckePrecisionReuseSlowValidation (slot-23 G1: [C, obl:precisionreuse](v) at (300, 7) and (300, 11), sat route) ===\n";
    std::cout << "  " << ra::PrecisionReuseEffectiveLine() << "\n";
    try {
        {
            std::cout << "(S1) (300, 7) KMAX 6: K_stab >= 3 and a plateau, eps = [9,16,21,25,27,27]\n";
            const Out o = RunSat(300, 7, 6, false), d = RunSat(300, 7, 6, true);
            Compare("(S1) sat (300, 7)", o, d, true);
            bool found = false;
            for (const auto& e : o.eps) found = found || e == std::vector<uint32_t>({9, 16, 21, 25, 27, 27});
            Check(found, "(S1) (300, 7): the D = 24 block reads eps = [9,16,21,25,27,27]");
            Check(o.K_run == std::vector<uint32_t>({2, 3, 4, 5, 6}) && d.Lv_run == std::vector<uint32_t>({6}),
                  "(S1) (300, 7): five levels read, one run (K* = 6)");
        }
        {
            std::cout << "(S2) (300, 11) KMAX 6: two levels\n";
            const Out o = RunSat(300, 11, 6, false), d = RunSat(300, 11, 6, true);
            Compare("(S2) sat (300, 11)", o, d, true);
            Check(o.K_run == std::vector<uint32_t>({2, 3}) && d.Lv_run.size() == 1, "(S2) (300, 11): two levels read, one run");
        }
    } catch (const std::exception& e) {
        Check(false, std::string("an arm threw: ") + e.what());
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "HeckePrecisionReuseSlowValidation: " << Check.checks << " checks, " << Check.fails << " failing, " << s << " s\n";
    return Check.fails ? 1u : 0u;
}
