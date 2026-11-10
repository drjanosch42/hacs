// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HeckeHidaEisLambdaValidation (standard) -- slot-25 package H1: the Eisenstein component over Lambda at irregular pairs, the finite-
// weight readings of [U] part 6 (sec:u-eislambda, docs/latex/hida_hecke_algebra.tex 0.7.0-pre.2), and the record's G8 under the
// non-default settings value heckez.G8 = hida ([C, sec:recordschema] item 8 (i), (j), (k)).  Card: 06-work-packages.md "H1";
// report docs/implementation/slot25/H1_report.md.  The 547 truncation itself is HeckeLambdaValidation's read-only arm
// HECKE_LAMBDA_CELLS="547:486:3:3" (too long for the standard tier).
//
//   (A) [U, prop:u-H11] on the C2 chains (04-record-census.md Sec. 3, C2; 05 Sec. 2.7): eleven irregular pairs (ell, a), every
//       chain weight k <= 258 (691: k = 12, 702 and 200), the forms side mod ell^3 (HeckeHidaEisLambda ReadBlock): the ordinary
//       Eisenstein block of e S_k has rank g = 1; (U1) on the forms side, L_forms = min_q v(a_q - e_q) = v(B_k/k) ([U, pr:u-U1]: a
//       smaller value is a defect); the generator q of I^0 found at the first weight (one integer per pair) passes at every chain
//       weight ("then for every such k"); K9's integers and kernel checks at the first weight (HIDA_K9 lines, consumed by L-K9);
//   (B) [U, pr:u-U1] on cohomology: L_coh = v(B_k/k) and g = 1 at eight chain weights (HeckeHidaComponents, K = 3); the read-only arm
//       HECKE_HIDA_EIS_ALL=1 adds every chain weight, (700, 691), (198, 691) and (D2) at (484, 547) (D = 5, g = 2, L_coh = 1);
//   (C) G8 under heckez.G8 = hida: the 15 golden default records of HZREC_report.md Sec. 10.2 recomputed bit-identical (the default
//       settings line is frozen; no default record moves); hida records at (1,10), (1,22), (1,30): a new fingerprint and id, the settings
//       text differing in the G8 line only, every G0 - G7 field equal to the default record's, the G8 values equal to (A)'s independent
//       readings, the scan clean, the cell file round-trips; tamper (a G8 value outside its list, an unknown G8 path, a G8 field in a
//       default record, all re-hashed) rejected; a value other than off / hida refused.
// Prints "EFFECTIVE HeckeHidaEisLambdaValidation arm=<DEFAULT|ALL> cells=<k>" and the gate line
// "HIDA_EIS_LAMBDA pairs=11 h11_held=.. u1_forms_held=.. coh_held=.. k9_ok=.. golden_default_unchanged=15/15 g8_records=3 g8_agree=..
//  tampered_detected=3 verdict=OK".  Writes nothing, no golden object, no digest of record.
#include "HeckeHidaEisLambdaValidation.h"

#include <sys/resource.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "HeckeHidaComponents.h"
#include "HeckeHidaEisLambda.h"
#include "HeckeZRecordSource.h"
#include "../Hecke/HeckeSettings.h"
#include "../Integral/HeckeZRecord.h"
#include "../Util/ValidationCheck.h"

namespace {

namespace E = hecke_ext::eislambda;
namespace HC = hecke_ext::hidacomp;
using namespace heckez::record;

oc_util::Checker Check("  [FAIL] ");

double Now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
long PeakRssMiB() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss / 1024;
}

struct Pair {
    uint64_t ell;
    uint32_t a;
    std::vector<uint32_t> ks;
};
// the C2 chains (04 Sec. 3: weights k_0 + j(ell - 1) <= 258 of the irregular pairs ell <= 157) with their base weight k = a, and
// 691 at a = 12 (k = 12, 702: T1 and (1,700)) and a = 200 (C9's (198,691))
const std::vector<Pair>& Chains() {
    static const std::vector<Pair> v = {
        {37, 32, {32, 68, 104, 140, 176, 212, 248}}, {59, 44, {44, 102, 160, 218}}, {67, 58, {58, 124, 190, 256}},
        {101, 68, {68, 168}},  {103, 24, {24, 126, 228}}, {131, 22, {22, 152}}, {149, 130, {130}}, {157, 62, {62, 218}},
        {157, 110, {110}},     {691, 12, {12, 702}},      {691, 200, {200}},
    };
    return v;
}

// the frozen golden default records (HZREC_report.md Sec. 10.2; settings fingerprint 0x8ffc29730c4f9672)
const std::vector<std::pair<uint32_t, std::string>>& GoldenDefault() {
    static const std::vector<std::pair<uint32_t, std::string>> v = {
        {2, "0x45c3bcb67cc1ee2f"},  {4, "0x460a286b68f29837"},  {6, "0x2da2a8ba88b0a02e"},  {8, "0x53aee77a43a3b7f3"},
        {10, "0x5e84b4c2540ff22a"}, {12, "0x72ab7ce1b1233c79"}, {14, "0x5874ce9b1cdbe5d6"}, {16, "0xddccf07d2ed83548"},
        {18, "0xa44222cead23af84"}, {20, "0x3c8d14fcae2dc259"}, {22, "0xa68f6f1daf281d13"}, {24, "0xea54e1459921c7fc"},
        {26, "0xfd6e49500d0243ed"}, {28, "0xee63f3bc98b50f7b"}, {30, "0x03e757d0491d3616"},
    };
    return v;
}
constexpr const char* kDefaultFingerprint = "0x8ffc29730c4f9672";

const Field* FindF(const HeckeStructureRecord& R, const std::string& p) { return R.find(p); }

}  // namespace

uint32_t HeckeHidaEisLambdaValidation() {
    Check.reset();
    const double t0 = Now();
    const char* env_all = std::getenv("HECKE_HIDA_EIS_ALL");
    const bool all = env_all && std::string(env_all) == "1";
    // (B)'s cohomology cells (n, ell): default eight; ALL every chain weight + (700, 691), (198, 691) and (484, 547)
    std::vector<std::pair<uint32_t, uint64_t>> coh = {{30, 37}, {66, 37}, {42, 59}, {56, 67}, {66, 101}, {22, 103}, {20, 131}, {60, 157}};
    if (all) {
        coh.clear();
        for (const auto& p : Chains())
            for (uint32_t k : p.ks) coh.push_back({k - 2, p.ell});
        coh.push_back({484, 547});
    }
    std::cout << "EFFECTIVE HeckeHidaEisLambdaValidation arm=" << (all ? "ALL" : "DEFAULT") << " cells=" << Chains().size() + coh.size() + 18
              << " (pairs " << Chains().size() << ", cohomology " << coh.size() << ", records 18)" << std::endl;

    // ---- (A) prop:u-H11 on the C2 chains, the forms side
    uint32_t h11 = 0, u1f = 0, n_w = 0, k9 = 0;
    std::map<std::pair<uint64_t, uint32_t>, E::BlockReading> readings;   // (ell, k) -> reading, for (C)
    for (const auto& p : Chains()) {
        uint64_t q0 = 0;
        bool held = true;
        std::string vb, lf;
        for (uint32_t k : p.ks) {
            ++n_w;
            E::BlockReading b;
            try {
                b = E::ReadBlock(p.ell, k, 3);
            } catch (const std::exception& e) {
                Check(false, "ReadBlock(" + std::to_string(p.ell) + ", " + std::to_string(k) + ") threw: " + e.what());
                held = false;
                continue;
            }
            Check(b.g == 1, "g = 1 at (" + std::to_string(p.ell) + ", " + std::to_string(k) + ") (cuspidal rank one, [U] part 6)");
            const bool u1 = b.v_Bk >= 1 && b.L_forms == b.v_Bk;
            Check(u1, "(U1) forms side L_forms = v(B_k/k) at (" + std::to_string(p.ell) + ", " + std::to_string(k) + "): " +
                          std::to_string(b.L_forms) + " vs " + std::to_string(b.v_Bk));
            u1f += u1;
            vb += (vb.empty() ? "" : ",") + std::to_string(b.v_Bk);
            lf += (lf.empty() ? "" : ",") + std::to_string(b.L_forms);
            if (k == p.ks.front()) {
                q0 = b.I0_q;
                bool ok = false;
                const std::string line = E::K9Line(b, &ok);
                std::cout << line << std::endl;
                Check(ok, "K9 kernel checks at (" + std::to_string(p.ell) + ", " + std::to_string(k) + ")");
                k9 += ok;
            }
            // the generator q of the first weight at this weight: v(a_q - e_q) = v(B_k/k) (prop:u-H11: one k decides every k)
            bool qok = false;
            for (size_t i = 0; i < b.primes.size(); ++i)
                if (b.primes[i] == q0) qok = b.v_det[i] == b.v_Bk;
            held = held && q0 != 0 && qok;
            readings.emplace(std::make_pair(p.ell, k), std::move(b));
        }
        Check(held, "prop:u-H11 generator T_" + std::to_string(q0) + " at every chain weight of (" + std::to_string(p.ell) + ", " +
                        std::to_string(p.a) + ")");
        h11 += held;
        std::cout << "HIDA_H11 ell=" << p.ell << " a=" << p.a << " weights=" << p.ks.size() << " g=1 v_Bk=[" << vb << "] L_forms=[" << lf
                  << "] I0_generator=" << (q0 ? "T_" + std::to_string(q0) : std::string("none_found")) << " every_weight=" << (held ? "held" : "FAILED")
                  << std::endl;
    }

    // ---- (B) (U1) on cohomology
    uint32_t coh_ok = 0;
    for (const auto& c : coh) {
        try {
            const HC::ComponentCell cc = HC::ComputeComponentCell(c.first, c.second, 3);
            std::cout << "  " << HC::ComponentLine(cc) << std::endl;
            const uint32_t g_exp = c.second == 547 ? 2 : 1;
            const bool ok = cc.eis.present && cc.eis.g == g_exp && cc.eis.D == 1 + 2 * g_exp && cc.eis.L == cc.v_bk.v && !cc.v_bk.lower_bound &&
                            cc.eis.L >= 1;
            Check(ok, "(U1) on cohomology at (" + std::to_string(c.first) + ", " + std::to_string(c.second) + "): L_coh = v(B_k/k), g = " +
                          std::to_string(g_exp));
            coh_ok += ok;
        } catch (const std::exception& e) {
            Check(false, "ComputeComponentCell(" + std::to_string(c.first) + ", " + std::to_string(c.second) + ") threw: " + e.what());
        }
    }

    // ---- (C) G8 under heckez.G8 = hida
    const hecke::ScopedCoefficientRing ring(0);   // coefficient_ring = Z
    const RecordSources src = DefaultRecordSources();
    const RecordSettings Sdef;
    RecordSettings Sh;
    Sh.g8 = "hida";
    uint32_t gold_ok = 0;
    std::map<uint32_t, HeckeStructureRecord> def;
    for (const auto& [n, h] : GoldenDefault()) {
        HeckeStructureRecord R = BuildHeckeStructureRecord(1, n, "full", Sdef, src);
        const bool same = R.record_fnv1a64 == h && R.settings_fingerprint == kDefaultFingerprint && R.status == "final";
        if (!same) std::cout << "  [GOLDEN] (1," << n << ") record_fnv1a64=" << R.record_fnv1a64 << " expected " << h << " fp=" << R.settings_fingerprint << "\n";
        Check(same, "golden default (1," + std::to_string(n) + ") record_fnv1a64 = " + h + " (frozen, HZREC_report.md Sec. 10.2)");
        gold_ok += same;
        bool g8 = false;
        for (const Field& f : R.fields) g8 = g8 || f.group == "G8";
        Check(!g8, "the default record (1," + std::to_string(n) + ") has no G8 field");
        Check(R.golden_eligible(), "the default record (1," + std::to_string(n) + ") is golden-eligible (final, default settings)");
        def.emplace(n, std::move(R));
    }
    Check(Sdef.caps_text().find("heckez.G8 = off (default off: no G8 field; another value is H1's Hida block per maximal ideal)\n") !=
              std::string::npos,
          "the default heckez.G8 line is the frozen text, byte for byte");
    uint32_t g8_agree = 0, n_g8 = 0;
    // (n, ell, k) whose G8 at the Eisenstein ideal is compared with (A)'s reading
    const std::vector<std::pair<uint32_t, uint64_t>> hcells = {{10, 691}, {22, 103}, {30, 37}};
    // the hida records' hashes (not golden: a regression pin of H1's G8; H1_report.md Sec. 3), settings fingerprint 0x08cec5d00e4d4294
    const std::map<uint32_t, std::string> hida_pin = {{10, "0x771eece58525a74b"}, {22, "0xf4d751373d547dca"}, {30, "0x3147c178aeb99cf5"}};
    HeckeStructureRecord H10;
    for (const auto& [n, ell] : hcells) {
        HeckeStructureRecord H = BuildHeckeStructureRecord(1, n, "full", Sh, src);
        const HeckeStructureRecord& D = def.at(n);
        ++n_g8;
        Check(H.settings_fingerprint != D.settings_fingerprint && H.id != D.id && H.record_fnv1a64 != D.record_fnv1a64,
              "hida (1," + std::to_string(n) + "): a new fingerprint, id and record hash");
        // the settings texts differ in the heckez.G8 line only
        auto strip = [](const std::string& t) {
            std::string o;
            std::istringstream is(t);
            std::string line;
            while (std::getline(is, line))
                if (line.compare(0, 12, "heckez.G8 = ") != 0) o += line + "\n";
            return o;
        };
        Check(strip(H.settings_text) == strip(D.settings_text) && H.settings_text != D.settings_text,
              "hida (1," + std::to_string(n) + "): the settings text differs in the heckez.G8 line only");
        // every hashed G0 - G7 field equal (G0.status included: G8 is final)
        std::vector<std::string> a, b;
        for (const Field& f : D.fields)
            if (f.group >= "G0" && f.group <= "G7" && f.group.size() == 2) a.push_back(f.path + "=" + f.value);
        for (const Field& f : H.fields)
            if (f.group >= "G0" && f.group <= "G7" && f.group.size() == 2) b.push_back(f.path + "=" + f.value);
        Check(a == b, "hida (1," + std::to_string(n) + "): every G0 - G7 field equals the default record's");
        const ScanResult sc = ScanRecord(H);
        for (const auto& p : sc.problems) std::cout << "  [SCAN] hida (1," << n << ") " << p << "\n";
        Check(sc.values_outside_lists == 0 && sc.exact_under_bound == 0 && sc.overall_rule_ok, "hida (1," + std::to_string(n) + "): scan clean");
        Check(H.status == D.status, "hida (1," + std::to_string(n) + "): the overall status is the default record's");
        // (H1 audit F1) item 8 (k): only the default settings are golden -- a final hida record is not golden-eligible
        Check(H.status == "final" && !H.golden_eligible() && CellFileText(H, "h1_validation").find("\"golden_eligible\":false") != std::string::npos,
              "hida (1," + std::to_string(n) + "): final but golden_eligible = false (default settings only)");
        // the G8 fields
        std::cout << "  G8 hida (1," << n << ") id=" << H.id << " record_fnv1a64=" << H.record_fnv1a64 << ":";
        for (const Field& f : H.fields)
            if (f.group == "G8") std::cout << " " << f.path << "=" << f.value;
        std::cout << std::endl;
        // the Eisenstein ideal at ell: find the m[i] with a chain_key
        const std::string pre = "G8.ell=" + std::to_string(ell);
        std::string mi;
        for (const Field& f : H.fields)
            if (f.group == "G8" && f.path.compare(0, pre.size() + 3, pre + ".m[") == 0 && f.path.find(".chain_key") != std::string::npos)
                mi = f.path.substr(0, f.path.find(".chain_key"));
        const uint32_t k = n + 2;
        const auto itb = readings.find({ell, k});
        bool agree = !mi.empty() && itb != readings.end() && FindF(H, "G8.scope") && FindF(H, "G8.scope")->value == "level1" &&
                     FindF(H, pre + ".blocks") && FindF(H, pre + ".blocks")->value == "read";
        if (agree) {
            const E::BlockReading& rb = itb->second;
            auto val = [&](const std::string& f) { const Field* x = FindF(H, mi + "." + f); return x ? x->value : std::string("<none>"); };
            agree = val("ordinary") == "true" && val("g") == std::to_string(rb.g) && val("v_Bk") == std::to_string(rb.v_Bk) &&
                    val("L_k") == std::to_string(rb.L_forms) && val("structure") == "fibre_LxL" &&
                    val("I0_generator") == "T_" + std::to_string(rb.I0_q) &&
                    val("chain_key") == "(1," + std::to_string(ell) + "," + std::to_string(k % (ell - 1)) + ")";
        }
        Check(H.record_fnv1a64 == hida_pin.at(n) && H.settings_fingerprint == "0x08cec5d00e4d4294",
              "hida (1," + std::to_string(n) + ") record_fnv1a64 = the pinned " + hida_pin.at(n) + " (got " + H.record_fnv1a64 + ")");
        if (n == 30) {   // the non-Eisenstein ideals: ordinary iff T_ell is a unit on the block (checked independently: H1_report.md Sec. 3)
            auto v = [&](const std::string& p) { const Field* x = FindF(H, p); return x ? x->value : std::string("<none>"); };
            Check(v("G8.ell=29.m[1].ordinary") == "false" && v("G8.ell=29.m[0].ordinary") == "true" && v("G8.ell=67.m[0].ordinary") == "true" &&
                      v("G8.ell=67.m[1].ordinary") == "true" && v("G8.ell=683.m[1].I0_generator") == "T_2" &&
                      v("G8.ell=273067.blocks") == "not_run(forms_cap)" && v("G8.ell=5.blocks") == "not_run(class_zero)",
                  "hida (1,30): T_29 nilpotent on S_32 mod 29 (ordinary = false), units at 67; 683's Eisenstein ideal; the caps");
        }
        Check(agree, "hida (1," + std::to_string(n) + "): G8 at the Eisenstein ideal at " + std::to_string(ell) +
                         " = (A)'s reading (g, v_Bk, L_k, fibre_LxL, I0_generator, chain_key)");
        g8_agree += agree;
        // the cell file round-trips
        {
            HeckeStructureRecord back;
            std::string why;
            const bool rt = VerifyCellText(CellFileText(H, "h1_validation"), H.id, &back, &why) && back.record_fnv1a64 == H.record_fnv1a64;
            Check(rt, "hida (1," + std::to_string(n) + "): the cell file verifies (" + why + ")");
        }
        if (n == 10) H10 = H;
    }
    // tamper, every time re-hashed
    uint32_t tampered = 0;
    {
        HeckeStructureRecord X = H10;
        for (Field& f : X.fields)
            if (f.path.find(".structure") != std::string::npos && f.group == "G8") f.value = "gorenstein";
        X.rehash();
        tampered += ScanRecord(X).values_outside_lists > 0;
    }
    {
        HeckeStructureRecord X = H10;
        Field f{"G8", "G8.ell=691.m[0].lambda_adic_rank", "3", "int", ""};
        X.fields.insert(X.fields.end(), f);
        X.rehash();
        tampered += ScanRecord(X).values_outside_lists > 0;
    }
    {
        HeckeStructureRecord X = def.at(10);
        Field f{"G8", "G8.scope", "level1", "g8_scope", ""};
        X.fields.push_back(f);
        X.rehash();
        tampered += ScanRecord(X).values_outside_lists > 0;
    }
    Check(tampered == 3, "tamper: a G8 value outside its list, an unknown G8 path, a G8 field in a default record -- all rejected");
    {
        RecordSettings Sx;
        Sx.g8 = "lambda";
        bool refused = false;
        try {
            BuildHeckeStructureRecord(1, 10, "full", Sx, src);
        } catch (const std::invalid_argument&) {
            refused = true;
        }
        Check(refused, "heckez.G8 = lambda (not a value) is refused (invalid_argument)");
    }
    const bool ok = Check.fails == 0;
    std::cout << "HIDA_EIS_LAMBDA pairs=" << Chains().size() << " h11_held=" << h11 << "/" << Chains().size() << " u1_forms_held=" << u1f << "/" << n_w
              << " coh_held=" << coh_ok << "/" << coh.size() << " k9_ok=" << k9 << "/" << Chains().size() << " golden_default_unchanged=" << gold_ok
              << "/" << GoldenDefault().size() << " g8_records=" << n_g8 << " g8_agree=" << g8_agree << "/" << n_g8 << " tampered_detected=" << tampered
              << " verdict=" << (ok ? "OK" : "FAIL") << std::endl;
    std::cout << "HeckeHidaEisLambdaValidation: checks=" << Check.checks << " failing=" << Check.fails << " wall_s=" << Now() - t0
              << " peak_rss_mib=" << PeakRssMiB() << std::endl;
    return ok ? 0 : 1;
}
