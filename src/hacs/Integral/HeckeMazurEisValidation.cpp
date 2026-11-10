// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HeckeMazurEisValidation -- slot-25 H3 (see the header).

#include "HeckeMazurEisValidation.h"

#include "HeckeMazurEis.h"

#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Cell { uint32_t N, ell; };

bool ParseCells(const char* spec, std::vector<Cell>& out, uint32_t* refused) {
    std::stringstream ss(spec);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) continue;
        const size_t c = tok.find(':');
        try {
            if (c == std::string::npos) throw std::invalid_argument("no ':'");
            const unsigned long N = std::stoul(tok.substr(0, c)), l = std::stoul(tok.substr(c + 1));
            if (N > 100000 || l > 100000) throw std::invalid_argument("range");
            out.push_back({static_cast<uint32_t>(N), static_cast<uint32_t>(l)});
        } catch (const std::exception&) {
            std::cout << "  [REFUSED] HECKE_MAZUR_EIS_CELLS token '" << tok << "' (want N:ell)\n";
            ++*refused;
        }
    }
    return true;
}

}  // namespace

uint32_t HeckeMazurEisValidation() {
    std::cout << "============ HeckeMazurEisValidation (slot-25 H3) ============\n";
    uint32_t checks = 0, failing = 0;
    auto check = [&](bool ok, const std::string& what) {
        ++checks;
        if (!ok) {
            ++failing;
            std::cout << "  [MISMATCH] " << what << "\n";
        }
    };

    std::vector<Cell> cells;
    std::string arm = "default";
    uint32_t refused = 0;
    const char* all = std::getenv("HECKE_MAZUR_EIS_ALL");
    const char* spec = std::getenv("HECKE_MAZUR_EIS_CELLS");
    if (all && std::string(all) == "1") {
        arm = "ALL";
        for (const auto& c : heckez::MazurEisCellList()) cells.push_back({c.N, c.ell});
    } else if (spec && *spec) {
        arm = "CELLS";
        ParseCells(spec, cells, &refused);
    } else {
        cells = {{11, 5}, {31, 5}, {41, 5}, {61, 5}, {71, 7}, {101, 5}, {113, 7}, {131, 13}, {151, 5}, {181, 5}};
    }
    if (arm != "default")
        std::cout << "  EFFECTIVE HeckeMazurEisValidation arm=" << arm << " cells=" << cells.size() << " parse_refused=" << refused << std::endl;

    uint32_t n_cells = 0, a_ok = 0, b_app = 0, b_ok = 0, c_ok = 0, good_ok = 0, first_rej = 0, struct_fail = 0, d_rep = 0, errors = 0;
    for (const Cell& cl : cells) {
        const auto t0 = std::chrono::steady_clock::now();
        heckez::MazurEisResult R;
        try {
            R = heckez::MazurEisStep(cl.N, cl.ell);
        } catch (const std::invalid_argument& e) {
            std::cout << "  [REFUSED] (" << cl.N << "," << cl.ell << "): " << e.what() << "\n";
            ++refused;
            continue;
        } catch (const std::exception& e) {
            std::cout << "  [ERROR] (" << cl.N << "," << cl.ell << "): " << e.what() << "\n";
            ++errors;
            check(false, "cell (" + std::to_string(cl.N) + "," + std::to_string(cl.ell) + ") threw");
            continue;
        }
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        ++n_cells;
        std::cout << "  MAZUR_EIS_CELL " << R.text() << " primes=" << R.primes_used << " digest=0x" << std::hex << R.digest()
                  << std::dec << " secs=" << secs << std::endl;
        const std::string tag = "(" + std::to_string(R.N) + "," + std::to_string(R.ell) + ")";
        a_ok += R.a_ok;
        c_ok += R.c_ok;
        good_ok += R.good_q_ok;
        first_rej += R.first_good_rejected;
        struct_fail += R.structure_failures;
        if (R.b_applicable) { ++b_app; b_ok += R.b_ok; }
        if (R.v_nu >= 2) ++d_rep;
        check(R.structure_failures == 0, tag + ": chi(T_q) = (x - 1 - q) c_q^2 at every word prime");
        check(R.c_ok, tag + ": (c) a good q with v_ell F_q(1 + q) = v_ell(nu) among the candidates");
        check(R.good_q_ok, tag + ": the recorded q re-checked good (two modular powers)");
        check(R.a_ok, tag + ": (a) deg F = e_W = (dim W - 1)/2 and (e >= 2) <=> Merel");
        if (R.b_applicable) check(R.b_ok, tag + ": (b) F(y + 1 + q) Eisenstein (ell || nu)");
        if (arm == "default") {
            if (R.N == 11) check(R.e == 1 && R.q != 0, tag + ": e = 1 (11a, T^0_m = Z_5)");
            if (R.N == 181) check(R.e == 3 && R.merel, tag + ": e = 3 (the e = 3 cell of [C, LN6])");
            if (R.N == 101) check(R.v_nu == 2 && !R.b_applicable, tag + ": ell^2 | nu (nu = 25), (b) not applicable, (d) reported");
        }
    }

    // the A7-M1 negative control: (61, 5) at the good q = 37 (61a: a_37 = 8 = 1 + 37 mod 5) -- rejected by the valuation check
    std::string neg = "skipped";
    if (arm == "default") {
        heckez::MazurEisSettings s;
        s.forced_q = {37};
        const auto R = heckez::MazurEisStep(61, 5, s);
        std::cout << "  MAZUR_EIS_NEGATIVE " << R.text() << std::endl;
        const bool rej = R.candidates.size() == 1 && R.candidates[0].good && !R.candidates[0].accepted &&
                         R.candidates[0].e == 2 && R.candidates[0].v_f == 2;
        check(rej, "(61,5) forced q = 37: good, e = 2, v = 2, rejected by the valuation check (A7-M1)");
        neg = rej ? "rejected" : "ACCEPTED";
    }

    // Gate against vacuous passes: zero cells fails; the pinned arms must run exactly their cell count, with no refusal.
    const bool count_ok = n_cells > 0 && (arm != "default" || (n_cells == 10 && refused == 0)) && (arm != "ALL" || n_cells == 180);
    if (!count_ok) std::cout << "  [COUNT] unexpected cell count: cells=" << n_cells << " refused=" << refused << " (arm " << arm << ")" << std::endl;
    const bool ok = failing == 0 && errors == 0 && count_ok && (arm != "CELLS" || refused == 0);
    std::cout << "HECKE_MAZUR_EIS arm=" << arm << " cells=" << n_cells << " a_ok=" << a_ok << " b_ok=" << b_ok << "/" << b_app
              << " c_ok=" << c_ok << " good_q_ok=" << good_ok << " d_reported=" << d_rep << " first_good_rejected=" << first_rej
              << " structure_failures=" << struct_fail << " refused=" << refused << " negative_control=" << neg
              << " checks=" << checks << " failing=" << failing << " verdict=" << (ok ? "OK" : "FAIL") << std::endl;
    return failing + errors + (arm == "CELLS" ? refused : 0) + (count_ok ? 0 : 1);
}
