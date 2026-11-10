// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 17, WP-8) see HeckeSmallCarriesValidation.h.
#include "HeckeSmallCarriesValidation.h"
#include "HeckeSatCertDumpValidation.h"

#include "../Hecke/HeckeLevelNDegenValidation.h"
#include "../Hecke/HeckeLevelNSparse.h"
#include "../Util/ValidationCheck.h"

#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {
oc_util::Checker Check("  [FAIL] ");
template <class F> bool Throws(F&& f) {
    try {
        f();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}
} // namespace

uint32_t HeckeSmallCarriesValidation() {
    Check.reset();
    // ---- (1) the strict grammar behind HECKE_LEVEL_N_SPARSE_CELLS
    {
        const auto ok = HookParseCells("1571:0:5,210:4:7", "HECKE_LEVEL_N_SPARSE_CELLS", 3);
        Check(ok.size() == 2 && ok[0][0] == 1571 && ok[1][2] == 7, "(1) a good list parses");
        for (const char* bad : {"", "1571:0:5,", ",1571:0:5", "1571:0:5,,210:4:7", "1571:0", "1571:0:5:", "1571:0:x", "1571:-2:5"})
            Check(Throws([&] { HookParseCells(bad, "HECKE_LEVEL_N_SPARSE_CELLS", 3); }), std::string("(1) refused: '") + bad + "'");
    }
    // ---- (2) the rule-19 refusal, hook_rss/v1 (slot-17 WP-17, WP-8 audit MAJOR-1 / MINOR-2 / MINOR-3): the per-cell factor,
    // the cells WP-5 ran (admitted) and the ones it could not (refused), the cap at the boundary, and the override.
    {
        const HookRssCap def = HookRssCapFrom(nullptr);
        const auto rss = [](HookKind k, uint64_t N, uint32_t n) { return kHookRssBaselineMiB * 1048576.0 + HookPredictedRssBytes(k, N, n); };
        const auto admitted = [&](HookKind k, uint64_t N, uint32_t n, const HookRssCap& cap) {
            return !Throws([&] { HookRefuseOverCap("T", 1, "N=" + std::to_string(N), rss(k, N, n), cap); });
        };
        Check(HookRssFactor(HookKind::kSparse, 99330, 0) == 64.0, "(2) 99330 = 2*3*5*7*11*43 is in the W5-F1 regime (factor 64)");
        Check(HookRssFactor(HookKind::kSparse, 90090, 0) == 16.0 && HookRssFactor(HookKind::kSparse, 100003, 0) == 16.0,
              "(2) factor 16 at d = 1 (90090, the prime 100003)");
        Check(HookRssFactor(HookKind::kSparse, 60060, 2) == 4.0 && HookRssFactor(HookKind::kDegen, 60060, 2) == 6.0 &&
                  HookRssFactor(HookKind::kR3k, 60060, 2) == 6.0,
              "(2) factor 4 (SPARSE) / 6 (DEGEN, R3K) at d >= 3");
        // every measured rung is predicted ABOVE its measured peak (WP-5 section 3, MiB) and admitted at the default cap
        const struct { uint64_t N; uint32_t n; double measured_mib; } rungs[] = {
            {30030, 0, 390}, {60060, 0, 970}, {90090, 0, 1639}, {100003, 0, 400}, {60060, 2, 2567}, {1571, 0, 103}};
        for (const auto& r : rungs) {
            Check(rss(HookKind::kSparse, r.N, r.n) / 1048576.0 > r.measured_mib,
                  "(2) SPARSE " + std::to_string(r.N) + ":" + std::to_string(r.n) + " predicted above its measured peak");
            Check(admitted(HookKind::kSparse, r.N, r.n, def), "(2) SPARSE " + std::to_string(r.N) + ":" + std::to_string(r.n) + " admitted");
        }
        // the DEGEN cells (sum of the M and N presentations), measured 1442 / 2111 MiB (WP-5) and 3.14 GB (slot 16, 15015 -> 30030, n = 2)
        const auto degen = [](uint64_t M, uint64_t N, uint32_t n) {
            return kHookRssBaselineMiB * 1048576.0 + HookPredictedRssBytes(HookKind::kDegen, M, n) + HookPredictedRssBytes(HookKind::kDegen, N, n);
        };
        Check(degen(30030, 60060, 0) / 1048576.0 > 1442 && degen(30030, 90090, 0) / 1048576.0 > 2111 &&
                  degen(15015, 30030, 2) / 1048576.0 > 3140 && degen(15015, 30030, 2) < 8.0 * 1073741824.0,
              "(2) DEGEN: the three measured scale cells are predicted above their peak and under the cap");
        Check(!admitted(HookKind::kSparse, 99330, 0, def), "(2) 99330:0 (W5-F1, past the 8 GB watchdog) is refused");
        Check(!admitted(HookKind::kSparse, 90090, 2, def), "(2) 90090:2 (k = 4, unrun) is refused at the default cap");
        Check(!admitted(HookKind::kSparse, 1048575, 200, def), "(2) (1048575, 200) is refused");
        // the override: whole GiB in [1, 1024], strict; it admits 90090:2 deliberately and can also lower the cap
        const HookRssCap big = HookRssCapFrom("12");
        Check(big.gib == 12 && big.overridden && HookRssCapText(big).find("rss_cap_source=HECKE_LEVEL_N_HOOK_RSS_CAP_GIB") != std::string::npos,
              "(2) HECKE_LEVEL_N_HOOK_RSS_CAP_GIB=12 parses and is named on the EFFECTIVE text");
        Check(HookRssCapText(def) == " rss_cap_gib=8 rss_cap_source=default rss_model=hook_rss/v1", "(2) the default EFFECTIVE text");
        Check(admitted(HookKind::kSparse, 90090, 2, big), "(2) 90090:2 is admitted under the override (12 GiB)");
        Check(!admitted(HookKind::kSparse, 30030, 0, HookRssCapFrom("1")), "(2) a lowered cap (1 GiB) refuses 30030:0");
        for (const char* bad : {"", "0", "1025", "8.5", "-8", "8G", " 8", "99999"})
            Check(Throws([&] { HookRssCapFrom(bad); }), std::string("(2) override refused: '") + bad + "'");
        // the boundary: just below / above the cap
        const double cap = 8.0 * 1073741824.0;
        Check(!Throws([&] { HookRefuseOverCap("T", 1, "b", 0.99 * cap, def); }), "(2) just below the cap: admitted");
        Check(Throws([&] { HookRefuseOverCap("T", 1, "b", 1.01 * cap, def); }), "(2) just above the cap: refused");
    }
    // ---- (3) HECKE_DUMP_SAT_CERT_CELLS
    {
        std::string why;
        Check(HeckeSatCertDumpParseCells("168:13:2,220:17:2,258:37:2", &why), "(3) the default list parses");
        for (const char* bad : {"168:13:2,", ",168:13:2", "168:13:2,,220:17:2", "168:13:2:", "", "168:13", "169:13:2", "168:12:2", "168:13:1"}) {
            why.clear();
            const bool ok = HeckeSatCertDumpParseCells(bad, &why);
            Check(!ok && !why.empty(), std::string("(3) refused: '") + bad + "' (" + why + ")");
        }
        why.clear();
        HeckeSatCertDumpParseCells("168:13:2,", &why);
        Check(why.rfind("cell 2 ", 0) == 0, "(3) a trailing comma is refused BY POSITION (cell 2): " + why);
    }
    std::cout << "HeckeSmallCarriesValidation: " << Check.checks << " checks, " << Check.fails << " failing" << std::endl;
    return Check.fails ? 1 : 0;
}
