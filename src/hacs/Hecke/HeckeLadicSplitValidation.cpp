// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HeckeLadicSplitValidation.cpp -- see the header.
//
// hecke-extensions slot 18, WP-19b: the p != ell digit split ([X] slot-14
// re-cut row "The p != ell digit-cell split", re-carried by slot-13 WP-12):
// RunCell split at the Hecke step (hecke::ladic_cell::DigitCellSession), so a
// digit handle's hecke(H, p) for a different p shares the cell.
//
// The gate is bit-identity.  RunCell(req) is DigitCellSession(req).run(
// req.primes) -- one code path -- and the arms below check that a session
// reused across primes, in any order and after any number of earlier runs,
// returns for each prime list exactly the record a FRESH RunCell returns:
// canonical_text (every value and, with keep_objects, every serialised object)
// and the fields canonical_text leaves out (the word levels tried, the digit
// layout, the Hida flags).  Only hecke_seconds (timing) may differ.  The value
// agreement of RunCell with the shipped native route is
// HeckeLadicCellValidation's oracle (i), unchanged.
//
// Arms:
//  (S1) (10, 691) k = 7 (691^7 >= 2^64: a true digit cell), the default
//       K-mode source, keep_objects: one session, primes run one at a time in
//       the order 5, 691, 2, 3, then {2, 3, 5, 691} with the Hida idempotent,
//       then 2 again -- each against a fresh RunCell; base() against
//       RunCell with no prime.
//  (S2) widths 32 / 64 / 128 at a TEST-ONLY native-range cell (12, 5) fixed
//       K = 6, primes {3, 2, 5} one at a time and together.
//  (S3) the opt-in Prepass source at (10, 691) k = 7.
//  (S4) the module consumer: keep_hecke at module_K = 7, k = 2.
//  (S5) refusals: a cell RunCell refuses is refused by the session with the
//       same exception type and text.
//  (T)  timing (printed, never gated): one cell + four Hecke steps against
//       four fresh cells.
#include "HeckeLadicSplitValidation.h"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "LadicCell.h"
#include "HeckeSettings.h"
#include "../Util/Timing.h"

namespace {

using oc_util::Seconds;

using namespace hecke::ladic_cell;

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

// Every field of a record except the timing: canonical_text plus what it omits.
std::string FullText(const CellRecord& r) {
    std::ostringstream o;
    o << r.canonical_text();
    const Precision& p = r.precision;
    o << "levels=";
    for (uint32_t v : p.word_levels_tried) o << v << ",";
    o << " width=" << p.digit_width << " L=" << p.digits_L << " m=" << p.digit_m << " prepass=" << p.prepass << "\n";
    const HidaOut& h = r.hida;
    o << "hida present=" << h.present << " idem=" << h.idempotent << " comm=" << h.commutes
      << " nil=" << h.complement_nilpotent << " match=" << h.rank_match << "\n";
    o << "objects C=" << r.objects.C.size() << " Theta=" << r.objects.Theta.size() << " A=" << r.objects.A.size()
      << " decE=" << r.objects.dec_exponents.size() << "\n";
    return o.str();
}

std::string PrimesText(const std::vector<uint64_t>& P) {
    std::string s = "{";
    for (size_t i = 0; i < P.size(); ++i) s += (i ? "," : "") + std::to_string(P[i]);
    return s + "}";
}

// One session over `runs` (each a prime list, in order), every run against a
// fresh RunCell with the same Request and primes.
void SessionAgainstFresh(Checker& g, const Request& base_req, const std::vector<std::vector<uint64_t>>& runs,
                         const std::string& tag, bool check_base) {
    try {
        DigitCellSession s(base_req);
        if (check_base) {
            // base() carries no object (they are filled per run): compare with a prime-free,
            // object-free RunCell.
            Request rq = base_req;
            rq.primes.clear();
            rq.keep_objects = false;
            rq.keep_hecke = false;
            const CellRecord fresh = RunCell(rq);
            g.check(FullText(s.base()) == FullText(fresh), tag + ": base() == RunCell(no prime)");
            g.check(s.base().hecke.empty() && !s.base().hida.present, tag + ": base() carries no Hecke datum");
        }
        for (const std::vector<uint64_t>& P : runs) {
            const CellRecord shared = s.run(P);
            Request rq = base_req;
            rq.primes = P;
            const CellRecord fresh = RunCell(rq);
            const bool same = FullText(shared) == FullText(fresh);
            g.check(same, tag + " " + PrimesText(P) + ": shared-cell record == fresh RunCell record");
            g.check(shared.hecke.size() == P.size(), tag + " " + PrimesText(P) + ": one HeckeOut per prime");
            if (same)
                std::cout << "  " << tag << " " << PrimesText(P) << ": identical (" << FullText(shared).size()
                          << " bytes)\n";
        }
    } catch (const std::exception& e) {
        g.check(false, tag + ": threw " + e.what());
    }
}

Request DigitRequest(uint32_t n, uint64_t ell, uint32_t k, Width w) {
    Request rq;
    rq.n_sym = n;
    rq.ell = ell;
    rq.k = k;
    rq.width = w;
    return rq;
}

void ArmS1(Checker& g) {
    Request rq = DigitRequest(10, 691, 7, EngineLadicWidth());
    rq.keep_objects = true;
    rq.hida = true;
    SessionAgainstFresh(g, rq, {{5}, {691}, {2}, {3}, {2, 3, 5, 691}, {2}}, "(S1) (10,691) k=7", true);
}

void ArmS2(Checker& g) {
    for (Width w : {Width::W32, Width::W64, Width::W128}) {
        Request rq = DigitRequest(12, 5, 6, w);
        rq.allow_native_range = true;
        rq.fixed_K = 6;
        rq.keep_objects = true;
        rq.hida = true;
        SessionAgainstFresh(g, rq, {{3}, {2}, {5}, {3, 2, 5}},
                            "(S2) (12,5) K=6 w" + std::to_string(static_cast<uint32_t>(w)), w == Width::W64);
    }
}

void ArmS3(Checker& g) {
    Request rq = DigitRequest(10, 691, 7, EngineLadicWidth());
    rq.precision_source = PrecisionSource::Prepass;
    rq.hida = true;
    SessionAgainstFresh(g, rq, {{3}, {691}, {2, 691}}, "(S3) prepass (10,691) k=7", false);
}

void ArmS4(Checker& g) {
    Request rq = DigitRequest(10, 691, 2, EngineLadicWidth());
    rq.module_K = 7;
    rq.keep_hecke = true;
    rq.hida = false;
    SessionAgainstFresh(g, rq, {{2}, {3}, {2, 3}}, "(S4) module_K=7 keep_hecke (10,691)", false);
}

// The outcome of a call that may throw: "ok", or the exception's class and text.
template<class F>
std::string Outcome(F f) {
    try {
        f();
        return "ok";
    } catch (const hecke::NotImplementedInThisBuild& e) {
        return std::string("NotImplementedInThisBuild: ") + e.what();
    } catch (const std::logic_error& e) {
        return std::string("logic_error: ") + e.what();
    } catch (const std::exception& e) {
        return std::string("exception: ") + e.what();
    }
}

void ArmS5(Checker& g) {
    struct Case { uint32_t n; uint64_t ell; uint32_t k; const char* why; };
    const Case cases[] = {
        {10, 691, 2, "691^2 < 2^64: a native cell (logic_error)"},
        {10, 3, 41, "ell = 3: the small-ell refusal"},
        {11, 691, 7, "n odd: the K-mode range refusal"},
    };
    for (const Case& c : cases) {
        const Request rq = DigitRequest(c.n, c.ell, c.k, EngineLadicWidth());
        const std::string a = Outcome([&] { (void)RunCell(rq); });
        const std::string b = Outcome([&] { DigitCellSession s(rq); });
        const std::string tag = "(S5) (" + std::to_string(c.n) + "," + std::to_string(c.ell) + ") k=" + std::to_string(c.k);
        g.check(a != "ok" && a == b, tag + ": RunCell and the session refuse alike (" + c.why + ")");
        std::cout << "  " << tag << ": both refuse -- " << a.substr(0, 100) << "...\n";
    }
}

void ArmT(Checker& g) {
    const std::vector<uint64_t> P = {2, 3, 5, 7};
    Request rq = DigitRequest(24, 691, 7, EngineLadicWidth());
    rq.hida = false;
    auto t0 = std::chrono::steady_clock::now();
    for (uint64_t p : P) { Request r = rq; r.primes = {p}; (void)RunCell(r); }
    const double fresh = Seconds(t0);
    t0 = std::chrono::steady_clock::now();
    DigitCellSession s(rq);
    const double cell = s.cell_seconds();
    for (uint64_t p : P) (void)s.run({p});
    const double shared = Seconds(t0);
    std::cout << "  LADIC_SPLIT_TIMING cell=(24,691) k=7 primes=" << PrimesText(P) << " fresh_s=" << fresh
              << " shared_s=" << shared << " cell_s=" << cell << " ratio=" << (shared > 0 ? fresh / shared : 0.0)
              << "  (timing only, not gated)\n";
    g.check(true, "(T) timing printed");
}

} // namespace

uint32_t HeckeLadicSplitValidation() {
    std::cout << "\n=== HeckeLadicSplitValidation (slot-18 WP-19b: the p != ell digit split) ===\n";
    Checker g;
    ArmS1(g);
    ArmS2(g);
    ArmS3(g);
    ArmS4(g);
    ArmS5(g);
    ArmT(g);
    std::cout << "HeckeLadicSplitValidation: " << g.checks << " checks, " << g.fails << " failing\n";
    return g.fails == 0 ? 0u : 1u;
}
