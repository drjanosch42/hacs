// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 3, WP-1 (phase E1b-0) -- see
// HeckeUEllReachValidation.h for the arm-by-arm statement and
// docs/implementation/slot3/WP-1_report.md for the diagnosis this pins.
//
// THE DEFECT (docs/KNOWN_ISSUES.md Sec. 3, closed by this package).
// [C, alg:tau] step 1 builds z = (g^{-1} s_i) . pos(v) through the projective
// adj(g) = diag(N, 1) trick and reduces it to the fundamental domain.  The
// POINT's cleared coordinates reach ||z|| ~ ||pos||^2 N^2 (measured 144.8 N^2
// on the shipped 7-vertex model, ||pos|| <= 12), and QuadPoint::moebius forms
// t^2 and r*t from them, i.e. products ~ ||z||^2 ~ 2.1e4 N^4.  The int64 tier
// therefore dies at 145 N^2 >= 2^31.5, i.e. N >= ~4580 -- exactly the measured
// bracket 3617 < N < 5003.  The overflow was NEVER in the coset construction
// (its own products are ~N^2, and the coset tables build fine at ell = 43867),
// and it is NOT a failure of [C, lem:bounds]: the certified bound
// B = max(B_tab, 4 N^2) on the STORED transporters is 1.0e8 at N = 5003, with
// 2 B^2 = 2.0e16 << 2^63, and HeckeTauTable::certify() clause (e) passes.
// What overflowed is an INTERMEDIATE of step 1 that no clause of
// [C, lem:bounds] covers.
//
// THE REPAIR is [C, thm:backend]'s own third tier: QuadPointT<__int128>
// (QuadPoint.h), reached by hecke::escalate_on_overflow, which attempts the
// int64 tier first and widens ONLY after a HeckeOverflow.  Wherever int64
// completes, nothing changes -- bit-identity is a property of the control
// flow.  The forced-tier arms below assert that the two tiers also AGREE
// where both complete, which is the part control flow does not give for free.

#include "HeckeUEllReachValidation.h"

#include "ExactInt.h"
#include "QuadPoint.h"
#include "HeckeCosetTables.h"
#include "HeckeTauTable.h"
#include "HeckeOperatorFamily.h"
#include "HeckeSettings.h"
#include "SL2ZTessellationTables.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../EquivariantCohomologyOnK.h"
#include "../../Engine/EngineRuntime.h"

#include <chrono>
#include <cstdint>
#include <functional>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

using hecke::ExactTier;
using hecke::HeckeCosetTables;
using hecke::HeckeOverflow;
using hecke::Mat22;
using hecke::QuadPoint;
using hecke::QuadPointW;
using hecke::ScopedExactTier;
using hecke::SL2ZTessellationTables;

uint32_t g_checks = 0;
uint32_t g_fails = 0;

void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fails;
        std::cout << "  *** FAIL [U_ell reach] " << what << std::endl;
    }
}

double seconds_since(const std::chrono::steady_clock::time_point& t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// --- (1) QuadPointW against QuadPoint -------------------------------------

// Fixed-seed xorshift64 (deterministic across runs and build settings) --
// the construction HeckeCosetTables.cpp's self_check() uses.
struct XorShift64 {
    uint64_t state;
    explicit XorShift64(uint64_t seed) : state(seed) {}
    uint64_t next() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    }
    int64_t in_range(int64_t lo, int64_t hi) { // [lo, hi]
        const uint64_t span = static_cast<uint64_t>(hi - lo) + 1u;
        return lo + static_cast<int64_t>(next() % span);
    }
};

void ArmQuadPointTiers() {
    const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    const SL2ZTessellationTables model(K);

    // (1a) The shipped vertex positions widen exactly and compare equal.
    for (uint32_t v = 0; v < 7; ++v) {
        const QuadPoint P = model.vertex_position(v);
        const QuadPointW W = QuadPointW::widen(P);
        check(W.equals_narrow(P), "widen(pos(v)) != pos(v)");
        check(W.cmp_abs_re_half() == P.cmp_abs_re_half(), "cmp_abs_re_half tier mismatch on pos(v)");
        check(W.cmp_norm_one() == P.cmp_norm_one(), "cmp_norm_one tier mismatch on pos(v)");
    }

    // (1b) Seeded unimodular transforms of every vertex position: wherever the
    // narrow tier completes, the wide tier must give the SAME point and the
    // SAME predicate signs.  The entry range is chosen so that most cases fit
    // int64 and a few do not (the mixed regime is the point).
    XorShift64 rng(0x5eed'0001ull);
    uint32_t narrow_overflows = 0;
    for (uint32_t it = 0; it < 4000; ++it) {
        const int64_t scale = (it % 4 == 3) ? 3000000000ll : 100000ll;
        const int64_t a = rng.in_range(-scale, scale);
        const int64_t b = rng.in_range(-scale, scale);
        const int64_t c = rng.in_range(-scale, scale);
        // Keep det > 0 without solving anything: use [[a,b],[c,d]] with a
        // freely chosen d and reject det <= 0 (moebius requires det > 0).
        const int64_t d = rng.in_range(-scale, scale);
        Mat22 g{ a, b, c, d };
        int64_t det = 0;
        try {
            det = g.det();
        } catch (const HeckeOverflow&) {
            continue;
        }
        if (det <= 0) continue;
        const QuadPoint P = model.vertex_position(it % 7);

        // The narrow tier can fail EITHER in moebius (an int64 product) OR in
        // its FP64-filtered predicates, whose certified bound B has its own
        // exact-conversion limit; both are hecke::HeckeOverflow, and both are
        // what escalate_on_overflow widens on.  The whole narrow step is
        // therefore one all-or-nothing attempt, exactly as in construct_vertex.
        bool narrow_ok = true;
        QuadPoint zn(0, 1, 1, 1);
        int n_re = 0, n_norm = 0;
        try {
            zn = P.moebius(g);
            n_re = zn.cmp_abs_re_half();
            n_norm = zn.cmp_norm_one();
        } catch (const HeckeOverflow&) {
            narrow_ok = false;
            ++narrow_overflows;
        }

        bool wide_ok = true;
        QuadPointW zw(0, 1, 1, 1);
        int w_re = 0, w_norm = 0;
        try {
            zw = QuadPointW::widen(P).moebius(g);
            w_re = zw.cmp_abs_re_half();
            w_norm = zw.cmp_norm_one();
        } catch (const HeckeOverflow&) {
            wide_ok = false;
        }

        if (narrow_ok) {
            // The wide tier must never fail where the narrow one succeeds, and
            // must produce the identical gcd-normalised point and the identical
            // predicate signs.
            check(wide_ok, "wide tier failed where the narrow tier succeeded");
            if (wide_ok) {
                check(zw.equals_narrow(zn), "moebius tier mismatch (point)");
                check(w_re == n_re, "tier mismatch (cmp_abs_re_half)");
                check(w_norm == n_norm, "tier mismatch (cmp_norm_one)");
            }
        }
    }
    // The sweep must actually have exercised the regime the repair is for.
    check(narrow_overflows > 0,
          "the QuadPoint sweep never left the int64 regime (arm would be vacuous)");
    std::cout << "  [1] QuadPointW vs QuadPoint: 4000 seeded transforms, "
              << narrow_overflows << " past the int64 tier." << std::endl;
}

// --- (2) coset tables (the audit's F13 vacuity check) ----------------------

void ArmCosetTables(const std::vector<std::pair<uint64_t, uint32_t>>& pr, uint32_t ell,
                    uint32_t n_sym) {
    for (const auto& c : pr) {
        const auto t0 = std::chrono::steady_clock::now();
        uint64_t N_expected = 1;
        for (uint32_t k = 0; k < c.second; ++k) N_expected *= c.first;
        try {
            const HeckeCosetTables tables(c.first, c.second, ell, n_sym);
            check(tables.self_check() == 0, "coset self_check() failed");
            check(tables.self_check_upper() == 0, "coset self_check_upper() failed");
            check(tables.N() == N_expected, "coset N() != p^r");
            check(tables.num_cosets() == N_expected + N_expected / c.first,
                  "coset count != p^r + p^{r-1}");
            std::cout << "  [2] cosets p=" << c.first << " r=" << c.second
                      << " (N=" << N_expected << ", m=" << tables.num_cosets()
                      << ") self-checked in " << seconds_since(t0) << " s" << std::endl;
        } catch (const std::exception& e) {
            check(false, std::string("coset tables threw at N=") +
                             std::to_string(N_expected) + ": " + e.what());
        }
    }
}

// --- (3) the forced-wide-tier arm ------------------------------------------

// Every stored chain of a tau table, flattened for an entrywise comparison.
struct TauDump {
    std::vector<int64_t> flat;
    uint64_t bound_B = 0;
    uint64_t bound_B_tab = 0;
    uint32_t m = 0;
};

TauDump DumpTau(const hecke::HeckeTauTable& tab, const SimplicialComplex& K) {
    TauDump d;
    d.bound_B = tab.certified_bound_B();
    d.bound_B_tab = tab.certified_bound_B_tab();
    d.m = tab.num_cosets();
    for (uint32_t i = 1; i <= tab.num_cosets(); ++i) {
        for (uint32_t q = 0; q <= 2; ++q) {
            const uint32_t ns = static_cast<uint32_t>(K.get_simplices()[q].size());
            for (uint32_t s = 0; s < ns; ++s) {
                const std::vector<hecke::ChainTerm>& terms = tab.tau(i, q, s);
                d.flat.push_back(static_cast<int64_t>(terms.size()));
                for (const hecke::ChainTerm& t : terms) {
                    d.flat.push_back(t.coeff);
                    d.flat.push_back(t.gamma.a);
                    d.flat.push_back(t.gamma.b);
                    d.flat.push_back(t.gamma.c);
                    d.flat.push_back(t.gamma.d);
                    d.flat.push_back(static_cast<int64_t>(t.sigma_idx));
                }
            }
        }
    }
    return d;
}

// Builds the tau table for g = diag(1, p^r) at (ell, n_sym) under `tier` and
// returns its dump. Throws whatever the construction throws.
TauDump BuildTauUnderTier(const SimplicialComplex& K, const SL2ZTessellationTables& model,
                          uint64_t p, uint32_t r, uint32_t ell, uint32_t n_sym,
                          ExactTier tier) {
    const ScopedExactTier guard(tier);
    const HeckeCosetTables cosets(p, r, ell, n_sym);
    const hecke::HeckeTauTable tab(K, model, cosets, ell, n_sym);
    check(tab.certify() == 0, "tau certify() != 0");
    return DumpTau(tab, K);
}

// A^{(q)}(U_ell) = T_p(ell) at p = ell, computed end to end under `tier`.
std::vector<std::vector<uint32_t>> BuildUEllUnderTier(uint32_t ell, uint32_t n_sym,
                                                      ExactTier tier,
                                                      std::vector<uint32_t>* dims_out) {
    const ScopedExactTier guard(tier);
    const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    const SL2ZTessellationTables model(K);
    LocalSystemSimplicialComplex lssc(K, GenerateSL2ZRelationsElt(K, n_sym, ell), n_sym + 1, ell);
    EquivariantCohomologyOnK coh(&lssc);
    coh.compute_cohomology();
    hecke::HeckeOperatorFamily fam(&coh, K, model, ell, n_sym);
    const std::vector<Matrix>& U = fam.T_p(ell);
    const std::vector<uint32_t>& dims = coh.get_H_eq_dimensions();
    if (dims_out) *dims_out = dims;
    std::vector<std::vector<uint32_t>> out;
    for (size_t q = 0; q < U.size(); ++q) {
        std::vector<uint32_t> flat;
        flat.push_back(U[q].rows);
        flat.push_back(U[q].cols);
        for (uint32_t a = 0; a < U[q].rows; ++a)
            for (uint32_t b = 0; b < U[q].cols; ++b) flat.push_back(U[q].data[a][b]);
        out.push_back(std::move(flat));
    }
    return out;
}

void ArmForcedWide(uint64_t p, uint32_t r, uint32_t ell, uint32_t n_sym, bool also_operator) {
    const auto t0 = std::chrono::steady_clock::now();
    const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    const SL2ZTessellationTables model(K);
    try {
        const TauDump narrow = BuildTauUnderTier(K, model, p, r, ell, n_sym, ExactTier::Auto);
        const TauDump wide = BuildTauUnderTier(K, model, p, r, ell, n_sym, ExactTier::ForceWide);
        check(narrow.m == wide.m, "forced-wide: coset count differs");
        check(narrow.bound_B == wide.bound_B, "forced-wide: certified bound B differs");
        check(narrow.bound_B_tab == wide.bound_B_tab, "forced-wide: B_tab differs");
        check(narrow.flat == wide.flat, "forced-wide: tau chains differ entrywise");
        check(!narrow.flat.empty(), "forced-wide: empty tau dump (vacuous arm)");
        std::cout << "  [3] forced-wide p=" << p << " r=" << r << " ell=" << ell
                  << " n=" << n_sym << ": " << narrow.flat.size()
                  << " tau scalars identical, B=" << narrow.bound_B << " ("
                  << seconds_since(t0) << " s)" << std::endl;
    } catch (const std::exception& e) {
        check(false, std::string("forced-wide tau arm threw: ") + e.what());
    }

    if (!also_operator) return;
    try {
        std::vector<uint32_t> dims_a, dims_b;
        const auto A_auto = BuildUEllUnderTier(ell, n_sym, ExactTier::Auto, &dims_a);
        const auto A_wide = BuildUEllUnderTier(ell, n_sym, ExactTier::ForceWide, &dims_b);
        check(dims_a == dims_b, "forced-wide: dim H^q differs");
        check(A_auto == A_wide, "forced-wide: A^{(q)}(U_ell) differs entrywise");
        size_t entries = 0;
        for (const auto& v : A_auto) entries += v.size();
        std::cout << "  [3] forced-wide A^{(q)}(U_" << ell << ") at n=" << n_sym << ": "
                  << entries << " scalars identical" << std::endl;
    } catch (const std::exception& e) {
        check(false, std::string("forced-wide operator arm threw: ") + e.what());
    }
}

// --- (4) escalation accounting ---------------------------------------------

void ArmEscalationAccounting() {
    const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    const SL2ZTessellationTables model(K);
    // Every golden cell has p = ell only at small primes, and every good-prime
    // cell has N = p^r <= a few thousand: below the measured threshold the
    // Auto path must never widen, so the pre-WP-1 code is what runs.
    const std::pair<uint64_t, uint32_t> cells[] = {
        { 2, 1 }, { 3, 1 }, { 5, 1 }, { 7, 1 }, { 2, 2 }, { 3, 2 }, { 691, 1 }, { 3617, 1 }
    };
    for (const auto& c : cells) {
        hecke::reset_exact_tier_escalations();
        const HeckeCosetTables cosets(c.first, c.second, 691u, 2u);
        const hecke::HeckeTauTable tab(K, model, cosets, 691u, 2u);
        (void)tab;
        check(hecke::exact_tier_escalations() == 0,
              "Auto escalated below the int64 threshold (p=" + std::to_string(c.first) +
                  ", r=" + std::to_string(c.second) + ")");
    }
    std::cout << "  [4] escalation accounting: 0 escalations on 8 sub-threshold cells."
              << std::endl;
}

// --- (5) the negative control ----------------------------------------------

void ArmNegativeControl() {
    const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    const SL2ZTessellationTables model(K);
    // ForceNarrow is the pre-WP-1 build. At ell = 5003 it must still throw the
    // named exception -- otherwise the reach would have come from somewhere
    // other than the wide tier and this package would be measuring nothing.
    bool threw_named = false;
    try {
        const ScopedExactTier guard(ExactTier::ForceNarrow);
        const HeckeCosetTables cosets(5003u, 1u, 5003u, 0u);
        const hecke::HeckeTauTable tab(K, model, cosets, 5003u, 0u);
        (void)tab;
    } catch (const HeckeOverflow& e) {
        threw_named = (std::string(e.what()).find("overflow") != std::string::npos);
    } catch (const std::exception&) {
        threw_named = false;
    }
    check(threw_named, "ForceNarrow at ell = 5003 did NOT reproduce the int64 overflow");

    // And the escalation counter must actually fire under Auto there: one
    // escalation per step-1 vertex slot that the narrow tier cannot do.
    hecke::reset_exact_tier_escalations();
    try {
        const HeckeCosetTables cosets(5003u, 1u, 5003u, 0u);
        const hecke::HeckeTauTable tab(K, model, cosets, 5003u, 0u);
        check(tab.certify() == 0, "ell = 5003 tau table does not certify");
        // [C, lem:bounds] is NOT what failed: the certified bound is still
        // deep inside the int64 tier at N = 5003 (2 B^2 << 2^63, asserted by
        // certify() clause (e)); the overflow was an uncovered step-1
        // intermediate. Pin that, it is the WP-8 contract finding.
        const uint64_t B = tab.certified_bound_B();
        check(B >= 4ull * 5003ull * 5003ull, "certified B below 4 N^2");
        check(B < (1ull << 31), "certified B outside the int64 tier's own criterion B < 2^31");
    } catch (const std::exception& e) {
        check(false, std::string("Auto at ell = 5003 threw: ") + e.what());
    }
    const uint64_t esc = hecke::exact_tier_escalations();
    check(esc > 0, "Auto at ell = 5003 escalated 0 times (the wide tier was never needed?)");
    std::cout << "  [5] negative control: ForceNarrow at ell=5003 still overflows; Auto "
                 "escalated " << esc << " times and certifies." << std::endl;
}

// --- (6) the `u_ell` setting's thread propagation --------------------------
//
// HeckeUEllBudgetValidation owns every other `u_ell` arm, but it lives under
// HeckeExt/, where check_heckeext_includes.sh rule 2 forbids
// src/Engine/. The propagation arm therefore lives here -- the
// HeckeLocalityValidation arm6(e) pattern for `locality`, applied to the key
// slot 3 adds.
void ArmUEllPropagation() {
    using hecke::MathSettings;
    using hecke::ScopedUEll;
    using hecke::UEllMode;
    using hecke::UEllSetting;

    UEllSetting pinned;
    pinned.mode = UEllMode::BudgetMb;
    pinned.budget_mb = 512;

    std::vector<uint32_t> seen(2, 0);
    {
        const ScopedUEll guard(pinned);
        std::vector<std::function<void()>> tasks;
        for (int i = 0; i < 2; ++i) {
            tasks.push_back([i, &seen] {
                seen[static_cast<size_t>(i)] = hecke::math_settings().u_ell.budget_mb;
            });
        }
        engine::tape::run_concurrently(tasks);
    }
    check(seen[0] == 512 && seen[1] == 512,
          "a task thread does not see the launching thread's u_ell budget");
    check(hecke::math_settings().u_ell == UEllSetting(),
          "the launching thread did not return to the default u_ell");

    // Two concurrent pipelines under different `u_ell` must not leak.
    uint32_t a_in = 0, a_out = 0, b_in = 0, b_out = 0;
    std::thread ta([&a_in, &a_out] {
        UEllSetting off;
        off.mode = UEllMode::Off;
        const ScopedUEll guard(off);
        std::vector<std::function<void()>> t = { [&a_in] {
            a_in = static_cast<uint32_t>(hecke::math_settings().u_ell.mode);
        } };
        engine::tape::run_concurrently(t);
        a_out = static_cast<uint32_t>(hecke::math_settings().u_ell.mode);
    });
    std::thread tb([&b_in, &b_out] {
        UEllSetting pin;
        pin.mode = UEllMode::BudgetMb;
        pin.budget_mb = 99;
        const ScopedUEll guard(pin);
        std::vector<std::function<void()>> t = { [&b_in] {
            b_in = hecke::math_settings().u_ell.budget_mb;
        } };
        engine::tape::run_concurrently(t);
        b_out = hecke::math_settings().u_ell.budget_mb;
    });
    ta.join();
    tb.join();
    check(a_in == static_cast<uint32_t>(UEllMode::Off) &&
              a_out == static_cast<uint32_t>(UEllMode::Off) && b_in == 99 && b_out == 99,
          "two concurrent pipelines under different u_ell settings leaked into each other");
    check(hecke::math_settings().is_default(),
          "a concurrent u_ell override reached the main thread");
    std::cout << "  [6] u_ell propagates onto run_concurrently() tasks and does not leak "
                 "between threads." << std::endl;
}

} // namespace

uint32_t HeckeUEllReachValidation() {
    g_checks = 0;
    g_fails = 0;
    const auto t0 = std::chrono::steady_clock::now();
    std::cout << "================= HeckeUEllReachValidation (slot 3 WP-1, E1b-0) "
                 "=================\n";
    try {
        ArmQuadPointTiers();
        // The coset construction alone was never the site: N = 5041 > 4580 and
        // it builds. (ell = 43867 is HeckeUEllReachSlow's.)
        ArmCosetTables({ { 3617, 1 }, { 71, 2 } }, 691u, 2u);
        ArmForcedWide(5, 1, 5, 2, true);
        ArmForcedWide(691, 1, 691, 2, true);
        ArmForcedWide(71, 2, 691, 2, false);
        ArmEscalationAccounting();
        ArmNegativeControl();
        ArmUEllPropagation();
    } catch (const std::exception& e) {
        std::cout << "HeckeUEllReachValidation: TOP-LEVEL EXCEPTION: " << e.what() << std::endl;
        ++g_fails;
    }
    std::cout << "U_ELL_REACH checks=" << g_checks << " failures=" << g_fails
              << " wall_s=" << seconds_since(t0) << std::endl;
    return g_fails;
}

uint32_t HeckeUEllReachSlow() {
    g_checks = 0;
    g_fails = 0;
    const auto t0 = std::chrono::steady_clock::now();
    std::cout << "================= HeckeUEllReachSlow (slot 3 WP-1, E1b-0) "
                 "=================\n";
    try {
        // (8) the forced-wide arm at the old limit.
        ArmForcedWide(3617, 1, 3617, 2, false);

        // (6) the reach proof: the tau table past the old int64 threshold.
        const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
        const SL2ZTessellationTables model(K);
        for (uint32_t ell : { 5003u, 10007u }) {
            const auto ts = std::chrono::steady_clock::now();
            hecke::reset_exact_tier_escalations();
            const HeckeCosetTables cosets(ell, 1u, ell, 0u);
            const hecke::HeckeTauTable tab(K, model, cosets, ell, 0u);
            check(tab.certify() == 0, "tau certify() != 0 at ell = " + std::to_string(ell));
            check(hecke::exact_tier_escalations() > 0,
                  "no escalation at ell = " + std::to_string(ell));
            std::cout << "  [6] tau at ell=" << ell << " certified, "
                      << hecke::exact_tier_escalations() << " escalations ("
                      << seconds_since(ts) << " s)" << std::endl;
        }

        // (6) U_5003 end to end at n = 2, with HeckeBadPrimeValidation's
        // oracle rows. U_{ell^2} is NOT attempted: N = ell^2 = 25 030 009
        // would be 25 M cosets. rank(U^r) uses the T_{p^2} recursion, which is
        // gated on ell >= 5 and needs no second coset table.
        {
            const auto ts = std::chrono::steady_clock::now();
            const uint32_t ell = 5003, n_sym = 2;
            LocalSystemSimplicialComplex lssc(
                K, GenerateSL2ZRelationsElt(K, n_sym, ell), n_sym + 1, ell);
            EquivariantCohomologyOnK coh(&lssc);
            coh.compute_cohomology();
            hecke::HeckeOperatorFamily fam(&coh, K, model, ell, n_sym);
            const std::vector<Matrix>& U = fam.T_p(ell);
            const std::vector<uint32_t>& dims = coh.get_H_eq_dimensions();
            check(U.size() == dims.size(), "U_5003: wrong number of degrees");
            uint32_t max_d = 0;
            for (uint32_t d : dims) max_d = std::max(max_d, d);
            for (size_t q = 0; q < U.size(); ++q) {
                check(U[q].rows == dims[q] && U[q].cols == dims[q] && U[q].p == ell,
                      "U_5003: ill-typed matrix at q = " + std::to_string(q));
            }
            const uint32_t R = max_d + 3;
            for (size_t q = 0; q < U.size(); ++q) {
                std::vector<uint32_t> ranks;
                for (uint32_t r = 1; r <= R; ++r) {
                    Matrix Ur = fam.T_p_pow(ell, r)[q];
                    ranks.push_back(Ur.rank());
                }
                bool mono = true;
                for (size_t i = 1; i < ranks.size(); ++i)
                    if (ranks[i] > ranks[i - 1]) mono = false;
                check(mono, "U_5003: rank(U^r) not non-increasing at q = " + std::to_string(q));
                if (dims[q] >= 1) {
                    const uint32_t stable = ranks[dims[q] - 1];
                    bool stationary = true;
                    for (size_t i = dims[q] - 1; i < ranks.size(); ++i)
                        if (ranks[i] != stable) stationary = false;
                    check(stationary,
                          "U_5003: rank(U^r) not stationary by r = d_q at q = " +
                              std::to_string(q));
                }
            }
            std::cout << "  [6] U_5003 end to end at n=2: dims=" << dims[0] << "," << dims[1]
                      << "," << dims[2] << ", oracle rows OK (" << seconds_since(ts) << " s)"
                      << std::endl;
        }

        // (7) the ell = 43867 coset tables. U_43867 end to end is NOT
        // asserted: u_ell = auto refuses it (HeckeUEllBudgetValidation pins
        // the refusal), and the slot-3 plan scopes it out explicitly.
        ArmCosetTables({ { 43867, 1 } }, 691u, 2u);

        // (9) THE REACH LIMIT of clause (e), and it is a DIFFERENT one from step 1's.
        // The step-1 __int128 tier would carry ell to ~1.8e8, but it is not
        // what binds: certify() clause (e) -- [C] lem:bounds(5) /
        // obl:backend -- requires the certified bound on the STORED
        // transporters, B = max(B_tab, 4 N^2), to satisfy 2 B^2 < 2^63, i.e.
        // B <= 2^31 - 1 (N <= 23170), for the NARROW tier.  Until slot 24 the
        // table at ell = 43867 BUILT (step 1's wide tier did its job) and was
        // then REFUSED with code 51.  (slot-24 ELL-W) Above 23170 the table is
        // now certified in the WIDE tier (2 N B <= 2^63 - 1, the stored-class
        // products in __int128 after an int64 overflow; HeckeWideTauValidation).
        // Under the rollback HECKE_TAU_WIDE=0 this arm asserts the pre-ELL-W
        // refusal with code 51 exactly as before -- so both arms stay pinned.
        const bool wide_on = hecke::tau_wide_tier_enabled();
        std::cout << "  " << hecke::tau_wide_tier_effective_line() << std::endl;
        {
            const auto ts = std::chrono::steady_clock::now();
            check(4ull * 23170ull * 23170ull <= 2147483647ull,
                  "4 N^2 at N = 23170 is outside the int64 tier");
            check(4ull * 23171ull * 23171ull > 2147483647ull,
                  "4 N^2 at N = 23171 is inside the int64 tier (the limit is not 23170)");
            std::string msg;
            bool refused = false;
            bool wide_ok = false;
            hecke::reset_exact_tier_escalations();
            try {
                const HeckeCosetTables cosets(43867u, 1u, 691u, 0u);
                const hecke::HeckeTauTable tab(K, model, cosets, 691u, 0u);
                wide_ok = tab.certify() == 0 && tab.wide_tier() &&
                          tab.certified_bound_B() == 4ull * 43867ull * 43867ull;
            } catch (const std::exception& e) {
                msg = e.what();
                refused = (msg.find("certification FAILED (code 51)") != std::string::npos);
            }
            if (wide_on) {
                check(wide_ok, "ell = 43867 did not certify in the wide tier (got: " + msg + ")");
            } else {
                check(refused,
                      "HECKE_TAU_WIDE=0: ell = 43867 was not refused by certify() clause (e) / code 51 (got: " + msg +
                          ")");
            }
            check(msg.find("exact multiply overflow") == std::string::npos,
                  "ell = 43867 still throws the OLD int64 overflow (the wide tier did not fire)");
            check(hecke::exact_tier_escalations() > 0,
                  "ell = 43867 built its step-1 reduction without escalating");
            std::cout << "  [9] ell=43867: built (escalations="
                      << hecke::exact_tier_escalations() << "), then "
                      << (wide_on ? "CERTIFIED in the wide tier (2 N B <= 2^63 - 1; B = 4N^2 > 2^31-1)"
                                  : "REFUSED by certify() code 51 (HECKE_TAU_WIDE=0) -- 4 N^2 > 2^31-1")
                      << " (" << seconds_since(ts) << " s)" << std::endl;
        }

        // (10) The narrow limit is EXACT, not approximate: ell = 23167 (the
        // largest prime with 4 N^2 <= 2^31 - 1) certifies in the narrow tier,
        // and the next prime, ell = 23173, is the first wide table (refused
        // with code 51 under HECKE_TAU_WIDE=0).  Both tables are built at
        // n_sym = 0, so this costs seconds rather than the minutes a whole
        // U_ell at that size would.
        {
            const auto ts = std::chrono::steady_clock::now();
            hecke::reset_exact_tier_escalations();
            uint64_t esc_ok = 0;
            try {
                const HeckeCosetTables cosets(23167u, 1u, 691u, 0u);
                const hecke::HeckeTauTable tab(K, model, cosets, 691u, 0u);
                check(tab.certify() == 0, "ell = 23167 does not certify (the limit is not 23170)");
                check(!tab.wide_tier(), "ell = 23167 is not in the narrow tier");
                check(tab.certified_bound_B() == 4ull * 23167ull * 23167ull,
                      "ell = 23167: B is not 4 N^2");
                esc_ok = hecke::exact_tier_escalations();
                check(esc_ok > 0, "ell = 23167 built without escalating");
            } catch (const std::exception& e) {
                check(false, std::string("ell = 23167 threw: ") + e.what());
            }
            std::string msg;
            bool refused = false;
            bool wide_ok = false;
            try {
                const HeckeCosetTables cosets(23173u, 1u, 691u, 0u);
                const hecke::HeckeTauTable tab(K, model, cosets, 691u, 0u);
                wide_ok = tab.certify() == 0 && tab.wide_tier();
            } catch (const std::exception& e) {
                msg = e.what();
                refused = (msg.find("certification FAILED (code 51)") != std::string::npos);
            }
            if (wide_on) check(wide_ok, "ell = 23173 is not the first wide table (got: " + msg + ")");
            else check(refused, "HECKE_TAU_WIDE=0: ell = 23173 was not refused with code 51 (got: " + msg + ")");
            std::cout << "  [10] the narrow limit is exact: ell=23167 certifies narrow (B=4N^2="
                      << 4ull * 23167ull * 23167ull << ", escalations=" << esc_ok
                      << "), ell=23173 is " << (wide_on ? "the first WIDE table" : "refused with code 51 (HECKE_TAU_WIDE=0)")
                      << " (" << seconds_since(ts) << " s)" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "HeckeUEllReachSlow: TOP-LEVEL EXCEPTION: " << e.what() << std::endl;
        ++g_fails;
    }
    std::cout << "U_ELL_REACH_SLOW checks=" << g_checks << " failures=" << g_fails
              << " wall_s=" << seconds_since(t0) << std::endl;
    return g_fails;
}
