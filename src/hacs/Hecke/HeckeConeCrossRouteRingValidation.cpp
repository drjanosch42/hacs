// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#include "HeckeConeCrossRouteRingValidation.h"

// hecke-extensions slot 10, WP-3: obl:coneexact(vii) over Z/ell^k -- see the
// header for the arm list.  The file name ends in `Validation` (the test
// harness: the raw-mod census and the assert policy treat it as such).

#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "CoefficientRing.h"
#include "HeckeBoundaryDatum.h"
#include "HeckeBoundaryOperator.h"
#include "HeckeConeAction.h"
#include "HeckeCosetTables.h"
#include "HeckeOperatorOnK.h"
#include "HeckePairHomotopy.h"
#include "HeckeSettings.h"
#include "HeckeTauTable.h"
#include "SL2ZBoundaryPair.h"
#include "SL2ZTessellationTables.h"
#include "../EquivariantCohomologyOnK.h"
#include "../EquivariantMappingCone.h"
#include "../LocalSystemSL2Z.h"
#include "../../ModArithmetic/ModCtx.h"

namespace {

using engine::DenseBuf;
using engine::ModCtx;

uint64_t g_checks = 0;
uint32_t g_fail = 0;
uint32_t g_printed = 0;

void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        if (g_printed++ < 60) std::cout << "  [FAIL] " << what << std::endl;
    }
}

template<class F>
bool throws_niitb(F&& f, const std::string& must_contain, const std::string& also = "") {
    try { f(); } catch (const hecke::NotImplementedInThisBuild& e) {
        const std::string m = e.what();
        return m.find(hecke::kNotImplementedInThisBuildTag) == 0 && m.find(must_contain) != std::string::npos &&
               (also.empty() || m.find(also) != std::string::npos);
    } catch (...) { return false; }
    return false;
}

template<class Elt>
bool dense_equal(const DenseBuf<Elt>& a, const DenseBuf<Elt>& b) {
    if (a.rows != b.rows || a.cols != b.cols) return false;
    for (uint32_t i = 0; i < a.rows; ++i)
        for (uint32_t j = 0; j < a.cols; ++j)
            if (a.at(i, j) != b.at(i, j)) return false;
    return true;
}

std::string vec_str(const std::vector<uint32_t>& v) {
    std::ostringstream o;
    o << "(";
    for (size_t i = 0; i < v.size(); ++i) o << (i ? "," : "") << v[i];
    o << ")";
    return o.str();
}

// The integral tables of one T_p, built at the default ring (they are ring-free).
struct Tables {
    std::unique_ptr<hecke::HeckeCosetTables> cosets;
    std::unique_ptr<hecke::HeckeTauTable> tau;
    std::unique_ptr<hecke::HeckeBoundaryTauTable> taub;
    std::unique_ptr<hecke::HeckePairHomotopy> eta;
};

// One level Z/ell^k of one cell: the composite bulk, pair, cone, and per
// prime the operators and route 1.  Built under the ring in force.
struct Level {
    uint32_t k = 0;
    std::unique_ptr<LocalSystemSimplicialComplex> lssc;
    std::unique_ptr<EquivariantCohomologyOnK> coh;
    std::unique_ptr<hecke::SL2ZBoundaryPair> pair;
    std::unique_ptr<EquivariantMappingCone> cone;
    std::vector<std::unique_ptr<hecke::HeckeOperatorOnK>> bulk;
    std::vector<std::unique_ptr<hecke::HeckeBoundaryOperator>> bd;
    std::vector<std::unique_ptr<hecke::HeckeConeAction>> act;
};

struct PointSpec {
    uint32_t n;
    uint32_t ell;
    std::vector<uint32_t> levels;
    std::vector<uint64_t> primes;
};

// ---- (R) one cell: every level, every prime, three routes -------------------
void cell_arm(const PointSpec& ps, bool negative_control) {
    const auto t0 = std::chrono::steady_clock::now();
    const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    const hecke::SL2ZTessellationTables model(K);
    const uint32_t n = ps.n, ell = ps.ell;
    const std::string cell = "(" + std::to_string(n) + ", " + std::to_string(ell) + ")";

    std::vector<Tables> tb(ps.primes.size());
    for (size_t i = 0; i < ps.primes.size(); ++i) {
        Tables& t = tb[i];
        t.cosets = std::make_unique<hecke::HeckeCosetTables>(ps.primes[i], 1, ell, n);
        t.tau = std::make_unique<hecke::HeckeTauTable>(K, model, *t.cosets, ell, n);
        t.taub = std::make_unique<hecke::HeckeBoundaryTauTable>(K, model, *t.cosets);
        t.eta = std::make_unique<hecke::HeckePairHomotopy>(*t.tau, *t.taub);
        check(!t.tau->used_mod_ell_solve(), "(R) " + cell + " p = " + std::to_string(ps.primes[i]) + ": tau is integral");
    }

    for (uint32_t k : ps.levels) {
        hecke::ScopedCoefficientRing ring(k);
        const std::string at = cell + " Z/ell^" + std::to_string(k);
        const ModCtx<uint32_t> ctx = hecke::CoefficientRingContext<uint32_t>("HeckeConeCrossRouteRingValidation", ell, k);
        check(static_cast<uint64_t>(ctx.p) < (uint64_t(1) << 32), "(R) " + at + ": ell^k < 2^32 (the uint32_t regime)");
        Level L;
        L.k = k;
        L.lssc = std::make_unique<LocalSystemSimplicialComplex>(K, hecke::GenerateSL2ZRelationsRing<uint32_t>(K, n, ctx), n + 1, ctx.p);
        L.coh = std::make_unique<EquivariantCohomologyOnK>(L.lssc.get());
        L.coh->compute_cohomology();
        L.pair = std::make_unique<hecke::SL2ZBoundaryPair>(L.coh.get(), K);
        L.cone = L.pair->make_cone();
        check(L.cone->is_composite() && L.cone->module_cone().coneexact_iv_certified(), "(R) " + at + ": the composite cone");
        const ModuleConeT<uint32_t>& MC = L.cone->module_cone();
        const uint32_t Q = MC.q_max();
        std::ostringstream line;
        line << "  CROSS_ROUTE_RING " << at << " Hc=";
        for (uint32_t q = 0; q <= Q; ++q) line << (q ? "|" : "") << vec_str(MC.H_invariant_factors(q));
        for (size_t i = 0; i < ps.primes.size(); ++i) {
            const uint64_t p = ps.primes[i];
            const std::string where = at + " T_" + std::to_string(p);
            L.bd.push_back(std::make_unique<hecke::HeckeBoundaryOperator>(&L.pair->boundary_cohomology(), tb[i].cosets.get(),
                                                                          tb[i].taub.get()));
            L.bd.back()->compute();
            L.bulk.push_back(std::make_unique<hecke::HeckeOperatorOnK>(L.coh.get(), tb[i].cosets.get(), tb[i].tau.get()));
            L.bulk.back()->compute();
            L.act.push_back(std::make_unique<hecke::HeckeConeAction>(L.cone.get(), L.bulk.back().get(), L.bd.back().get(),
                                                                     tb[i].eta.get()));
            hecke::HeckeConeAction& act = *L.act.back();
            act.compute();
            check(act.is_composite(), "(R) " + where + ": route 1 is the Z/ell^k route");
            const hecke::ModuleConeActionT<uint32_t>& r1 = act.module_action();

            // Route 2: the strict pair.
            bool r2_ok = false, r3_ok = false;
            std::string r2_err, r3_err;
            std::unique_ptr<hecke::HeckeConeStrictPairRoute> r2;
            std::unique_ptr<hecke::HeckeConeRelativeRoute> r3;
            try {
                r2 = std::make_unique<hecke::HeckeConeStrictPairRoute>(&act);
                r2->compute();
                r2_ok = true;
            } catch (const std::exception& e) { r2_err = e.what(); }
            check(r2_ok, "(R) " + where + ": the strict-pair route ran over the ring: " + r2_err);
            // Route 3: the relative sub-complex.
            try {
                r3 = std::make_unique<hecke::HeckeConeRelativeRoute>(&act);
                r3->compute();
                r3_ok = true;
            } catch (const std::exception& e) { r3_err = e.what(); }
            check(r3_ok, "(R) " + where + ": route 3 ran over the ring: " + r3_err);
            if (!r2_ok || !r3_ok) continue;

            const auto& rec2 = r2->record();
            const auto& rec3 = r3->record();
            check(rec2.ring_k == k && rec3.ring_k == k, "(R) " + where + ": both routes report Z/ell^k");
            check(rec2.built && rec2.agrees() && rec2.degrees_compared == Q + 1, "(R) " + where + ": strict-pair record agrees");
            check(rec2.eta_zero && rec2.solve_regime == "Z", "(R) " + where + ": eta^s == 0, tau^s over Z");
            check(rec2.tau_differs, "(R) " + where + ": tau^s differs from the shipped tau (non-vacuity)");
            check(rec2.cochain_commutes, "(R) " + where + ": r Theta^s == Theta^bd r on C_eq (strictness)");
            check(r2->action_strict() != nullptr && r2->action_strict()->is_composite(), "(R) " + where + ": Phi^s over the ring");
            check(rec3.agrees(), "(R) " + where + ": route-3 record agrees");
            check(r3->relative_cohomology().is_composite() && r3->relative_cohomology().is_relative(),
                  "(R) " + where + ": ker r is the relative sub-complex over Z/ell^k");
            check(rec3.vanishing_simplices > 0, "(R) " + where + ": ker r forces boundary simplices to zero");
            uint32_t nonzero_deg = 0;
            for (uint32_t q = 0; q <= Q; ++q) {
                const DenseBuf<uint32_t>& A1 = r1.A_c(q);
                const DenseBuf<uint32_t>& A2 = r2->action_strict()->module_action().A_c(q);
                const DenseBuf<uint32_t>& A3 = r3->A_c_transported_module(q);
                check(dense_equal(A1, A2), "(R) " + where + " q = " + std::to_string(q) + ": A_c route 1 == strict-pair route");
                check(dense_equal(A1, A3), "(R) " + where + " q = " + std::to_string(q) + ": A_c route 1 == route 3");
                check(rec3.H_rel_invariant_factors.size() > q && rec3.H_rel_invariant_factors[q] == MC.H_invariant_factors(q),
                      "(R) " + where + " q = " + std::to_string(q) + ": H^q(ker r) ~ H^q_c (invariant factors)");
                bool nz = false;
                for (uint32_t a = 0; a < A1.rows && !nz; ++a)
                    for (uint32_t b = 0; b < A1.cols; ++b)
                        if (A1.at(a, b) != 0) { nz = true; break; }
                nonzero_deg += nz ? 1u : 0u;
            }
            check(nonzero_deg > 0, "(R) " + where + ": A_c is non-zero in some degree (non-vacuity)");
            // (hecke-extensions slot 10, WP-8, m-12): these three were PRINTED
            // only (below), so a route-3 build that took the trivial (E == 0,
            // Theta' == Theta) shortcut on every cell would have passed
            // silently.  Asserting them records that route 3's own correction
            // (E, Theta') is non-trivially exercised at every cell, not just
            // this one line's text.
            check(rec3.E_nonzero, "(R) " + where + ": route 3's E is non-zero in some degree (non-vacuity, m-12)");
            check(rec3.theta_prime_differs, "(R) " + where + ": route 3's Theta' differs from Theta on C_eq (non-vacuity, m-12)");
            check(!rec3.r_commutes_normative, "(R) " + where + ": r Theta != Theta^bd r on the shipped (non-strict) tau (m-12)");
            line << " T_" << p << "[r2 " << rec2.degrees_compared << "/" << rec2.degrees_compared << ", r3 "
                 << rec3.degrees_compared << "/" << rec3.degrees_compared << ", E" << (rec3.E_nonzero ? "!=0" : "==0")
                 << ", Th'" << (rec3.theta_prime_differs ? "!=Th" : "==Th") << ", c5 " << rec3.c5_checks << "]";

            // (N) the negative control, once (the first prime of the first level).
            if (negative_control && i == 0 && k == ps.levels.front()) {
                hecke::HeckeTauTable ts(K, model, *tb[i].cosets, ell, n, *tb[i].taub);
                bool seam_ok = false;
                try {
                    hecke::HeckeConeStrictPairRoute rs(&act, &ts);
                    rs.compute();
                    seam_ok = rs.record().agrees() && rs.record().external_table && rs.tau_strict() == &ts;
                } catch (...) { seam_ok = false; }
                check(seam_ok, "(N) " + where + ": the seam's unperturbed strict table agrees");
                // Perturb one tau^s coefficient (+1) of a degree-1 slot after construction.
                bool perturbed = false;
                for (uint32_t s = 0; s < K.get_num_q_simplices(1) && !perturbed; ++s) {
                    auto& terms = const_cast<std::vector<hecke::ChainTerm>&>(ts.tau(1, 1, s));
                    if (terms.empty()) continue;
                    terms[0].coeff += 1;
                    perturbed = true;
                }
                check(perturbed, "(N) " + where + ": a tau^s term was perturbed");
                check(ts.certify() != 0, "(N) " + where + ": the perturbed tau^s fails its own certificate");
                bool refused = false;
                std::string msg;
                try {
                    hecke::HeckeConeStrictPairRoute rs(&act, &ts);
                    rs.compute();
                } catch (const std::runtime_error& e) {
                    msg = e.what();
                    refused = msg.find("HeckeConeStrictPairRoute") != std::string::npos && msg.find("ABORTED") != std::string::npos;
                }
                check(refused, "(N) " + where + ": the perturbed tau^s is detected (the route aborts)");
                std::cout << "  (N) " << where << " perturbed tau^s: " << (refused ? "DETECTED" : "NOT DETECTED") << " -- "
                          << msg.substr(0, 220) << std::endl;
                // Beyond the certificate: the same perturbed table pushed through the route's own
                // chain (eta^s, Theta^s over the ring, route 1 on (tau^s, tau_bd, eta^s)) by hand --
                // it must fail an assertion or give a different A_c, never agree silently.
                // A FRESH strict table perturbed in a TRIANGLE slot (q = 2): eta^s lives on the
                // boundary (degrees 0, 1), so its fill is unaffected and the perturbation reaches
                // Theta^s and the cone action.
                hecke::HeckeTauTable ts2(K, model, *tb[i].cosets, ell, n, *tb[i].taub);
                bool perturbed2 = false;
                for (uint32_t s = 0; s < K.get_num_q_simplices(2) && !perturbed2; ++s) {
                    auto& terms = const_cast<std::vector<hecke::ChainTerm>&>(ts2.tau(1, 2, s));
                    if (terms.empty()) continue;
                    terms[0].coeff += 1;
                    perturbed2 = true;
                }
                check(perturbed2, "(N) " + where + ": a triangle slot of a fresh tau^s was perturbed");
                std::string where_caught = "agreed (NOT DETECTED)";
                bool caught = false;
                try {
                    hecke::HeckePairHomotopy eta_p(ts2, *tb[i].taub, act.eta_policy());
                    hecke::HeckeOperatorOnK bulk_p(L.coh.get(), tb[i].cosets.get(), &ts2);
                    bulk_p.compute();
                    hecke::HeckeConeAction act_p(L.cone.get(), &bulk_p, L.bd.back().get(), &eta_p);
                    act_p.compute();
                    for (uint32_t q = 0; q <= Q && !caught; ++q)
                        if (!dense_equal(act_p.module_action().A_c(q), r1.A_c(q))) {
                            caught = true;
                            where_caught = "A_c differs from route 1 in degree " + std::to_string(q);
                        }
                } catch (const std::exception& e) {
                    caught = true;
                    where_caught = std::string("aborted: ") + std::string(e.what()).substr(0, 160);
                }
                check(caught, "(N) " + where + ": the perturbed tau^s, bypassing its certificate, is still detected downstream");
                std::cout << "  (N) " << where << " perturbed tau^s past its certificate: " << where_caught << std::endl;
            }
        }
        // The comparison discriminates: two different T_p give different A_c in some degree.
        if (L.act.size() >= 2) {
            bool differ = false;
            for (uint32_t q = 0; q <= Q && !differ; ++q)
                differ = !dense_equal(L.act[0]->module_action().A_c(q), L.act[1]->module_action().A_c(q));
            check(differ, "(R) " + at + ": A_c(T_" + std::to_string(ps.primes[0]) + ") != A_c(T_" + std::to_string(ps.primes[1]) +
                              ") in some degree (the entrywise comparison discriminates)");
        }
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::cout << line.str() << "  " << secs << " s" << std::endl;
    }
}

// ---- (W) above 2^32: route 1 runs (uint64_t), the cross-routes refuse by the regime ----
void regime_arm() {
    using Elt = uint64_t;
    const uint32_t n = 10;
    const uint64_t ell = 134217757ull;
    const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    const hecke::SL2ZTessellationTables model(K);
    const SimplicialComplex K_bd = CreateSL2ZBoundaryPathTriangulation();
    hecke::HeckeCosetTablesT<Elt> cosets(2, 1, ell, n);
    hecke::HeckeTauTableT<Elt> tau(K, model, cosets, ell, n);
    hecke::HeckeBoundaryTauTableT<Elt> taub(K, model, cosets);
    hecke::HeckePairHomotopyT<Elt> eta(tau, taub);
    bool route1 = false;
    std::string err;
    {
        hecke::ScopedCoefficientRing ring(2);
        try {
            const ModCtx<Elt> ctx = hecke::CoefficientRingContext<Elt>("HeckeConeCrossRouteRingValidation", ell, 2);
            LocalSystemSimplicialComplexT<Elt> lssc(K, hecke::GenerateSL2ZRelationsRing<Elt>(K, n, ctx), n + 1, ctx.p);
            EquivariantCohomologyOnKT<Elt> coh(&lssc);
            coh.compute_cohomology();
            LocalSystemSimplicialComplexT<Elt> lssc_b(K_bd, hecke::GenerateSL2ZBoundaryRelationsRing<Elt>(K_bd, n, ctx), n + 1, ctx.p);
            EquivariantCohomologyOnKT<Elt> coh_b(&lssc_b);
            coh_b.compute_cohomology();
            std::vector<DenseBuf<Elt>> rf;
            for (uint32_t q = 0; q <= 1; ++q) rf.push_back(RestrictionFlatRing<Elt>(K, K_bd, q, n + 1, ctx));
            ModuleConeT<Elt> cone(&coh, &coh_b, std::move(rf));
            cone.compute();
            hecke::HeckeOperatorOnKT<Elt> bulk(&coh, &cosets, &tau);
            bulk.compute();
            hecke::HeckeBoundaryOperatorT<Elt> bd(&coh_b, &cosets, &taub);
            bd.compute();
            hecke::ModuleConeActionT<Elt> act(&cone, &bulk, &bd, &eta);
            act.compute();
            route1 = act.computed() && act.record().routes_bit_identical && ctx.p > (uint64_t(1) << 32);
        } catch (const std::exception& e) { err = e.what(); }
    }
    check(route1, "(W) (10, 134217757) Z/ell^2: route 1 (ModuleConeActionT<uint64_t>) runs above 2^32: " + err);
    check(throws_niitb([&] { hecke::HeckeConeStrictPairRoute::RequireRingRegime(ell, 2); }, "HeckeConeStrictPairRoute: ", "uint32_t"),
          "(W) the strict-pair route refuses Z/134217757^2 by the uint32_t regime's named refusal");
    check(throws_niitb([&] { hecke::HeckeConeRelativeRoute::RequireRingRegime(ell, 2); }, "HeckeConeRelativeRoute: ", "uint32_t"),
          "(W) route 3 refuses Z/134217757^2 by the uint32_t regime's named refusal");
    // In range the regime rule is silent (the seven points).
    bool silent = true;
    try {
        hecke::HeckeConeStrictPairRoute::RequireRingRegime(691, 2);
        hecke::HeckeConeRelativeRoute::RequireRingRegime(5, 3);
    } catch (...) { silent = false; }
    check(silent, "(W) the regime rule is silent at 691^2 and 5^3");
    std::cout << "  (W) (10, 134217757) Z/ell^2: route 1 " << (route1 ? "ran" : "FAILED") << "; cross-routes refuse by the uint32_t regime"
              << std::endl;
}

// ---- (X) the three flipped members, both ways -------------------------------
void closure_arm() {
    const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    const hecke::SL2ZTessellationTables model(K);
    const uint32_t n = 10, ell = 691;
    hecke::HeckeCosetTables cosets(2, 1, ell, n);
    hecke::HeckeTauTable tau(K, model, cosets, ell, n);
    hecke::HeckeBoundaryTauTable taub(K, model, cosets);
    hecke::HeckePairHomotopy eta(tau, taub);
    // The F_ell normative action.
    LocalSystemSimplicialComplex lssc_f(K, GenerateSL2ZRelationsElt<uint32_t>(K, n, ell), n + 1, ell);
    EquivariantCohomologyOnK coh_f(&lssc_f);
    coh_f.compute_cohomology();
    hecke::SL2ZBoundaryPair pair_f(&coh_f, K);
    auto cone_f = pair_f.make_cone();
    hecke::HeckeOperatorOnK bulk_f(&coh_f, &cosets, &tau);
    bulk_f.compute();
    hecke::HeckeBoundaryOperator bd_f(&pair_f.boundary_cohomology(), &cosets, &taub);
    bd_f.compute();
    hecke::HeckeConeAction act_f(cone_f.get(), &bulk_f, &bd_f, &eta);
    act_f.compute();
    // The composite normative action at Z/ell^2.
    std::unique_ptr<LocalSystemSimplicialComplex> lssc;
    std::unique_ptr<EquivariantCohomologyOnK> coh;
    std::unique_ptr<hecke::SL2ZBoundaryPair> pair;
    std::unique_ptr<EquivariantMappingCone> cone;
    std::unique_ptr<hecke::HeckeOperatorOnK> bulk;
    std::unique_ptr<hecke::HeckeBoundaryOperator> bd;
    std::unique_ptr<hecke::HeckeConeAction> act;
    const std::string R = "coefficient_ring=Z/ell^2 is not honoured in this build";
    {
        hecke::ScopedCoefficientRing ring(2);
        const ModCtx<uint32_t> ctx = hecke::CoefficientRingContext<uint32_t>("probe", ell, 2);
        lssc = std::make_unique<LocalSystemSimplicialComplex>(K, hecke::GenerateSL2ZRelationsRing<uint32_t>(K, n, ctx), n + 1, ctx.p);
        coh = std::make_unique<EquivariantCohomologyOnK>(lssc.get());
        coh->compute_cohomology();
        pair = std::make_unique<hecke::SL2ZBoundaryPair>(coh.get(), K);
        cone = pair->make_cone();
        bulk = std::make_unique<hecke::HeckeOperatorOnK>(coh.get(), &cosets, &tau);
        bulk->compute();
        bd = std::make_unique<hecke::HeckeBoundaryOperator>(&pair->boundary_cohomology(), &cosets, &taub);
        bd->compute();
        act = std::make_unique<hecke::HeckeConeAction>(cone.get(), bulk.get(), bd.get(), &eta);
        act->compute();

        // Honour (the ring in force == the normative action's).
        bool h2 = false, h3 = false, hrel = false;
        try { hecke::HeckeConeStrictPairRoute x(act.get()); h2 = true; } catch (...) {}
        try { hecke::HeckeConeRelativeRoute x(act.get()); h3 = true; } catch (...) {}
        try {
            EquivariantCohomologyOnK x(lssc.get(), std::vector<std::vector<uint32_t>>{{0}});
            x.compute_cohomology();
            hrel = x.is_composite() && x.is_relative() && !x.is_flat_route();
        } catch (...) {}
        check(h2, "(X) HeckeConeStrictPairRoute HONOURS a composite normative action under its ring (changed row)");
        check(h3, "(X) HeckeConeRelativeRoute HONOURS a composite normative action under its ring (changed row)");
        check(hrel, "(X) EquivariantCohomologyOnK (relative sub-complex) HONOURS Z/ell^2 at ell = 691 (changed row)");
        // Refuse: an F_ell normative action under the ring; null.
        check(throws_niitb([&] { hecke::HeckeConeStrictPairRoute x(&act_f); }, "HeckeConeStrictPairRoute: " + R),
              "(X) HeckeConeStrictPairRoute refuses an F_ell normative action under Z/ell^2");
        check(throws_niitb([&] { hecke::HeckeConeRelativeRoute x(&act_f); }, "HeckeConeRelativeRoute: " + R),
              "(X) HeckeConeRelativeRoute refuses an F_ell normative action under Z/ell^2");
        check(throws_niitb([&] { hecke::HeckeConeStrictPairRoute x(nullptr); }, "HeckeConeStrictPairRoute: " + R),
              "(X) HeckeConeStrictPairRoute refuses null under Z/ell^2");
        check(throws_niitb([&] { hecke::HeckeConeRelativeRoute x(nullptr); }, "HeckeConeRelativeRoute: " + R),
              "(X) HeckeConeRelativeRoute refuses null under Z/ell^2");
        // The relative sub-complex constructor at ell = 3 (the Bredon object) keeps refusing.
        const ModCtx<uint32_t> c9 = ModCtx<uint32_t>::prime_power(3u, 2);
        LocalSystemSimplicialComplex l9(K, hecke::GenerateSL2ZRelationsRing<uint32_t>(K, n, c9), n + 1, 9u);
        check(throws_niitb([&] { EquivariantCohomologyOnK x(&l9, std::vector<std::vector<uint32_t>>{{0}}); },
                           "EquivariantCohomologyOnK (relative sub-complex): ", "at ell = 3"),
              "(X) EquivariantCohomologyOnK (relative sub-complex) keeps refusing Z/ell^2 at ell = 3");
        // An F_ell local system under the ring: the base constructor's refusal, unchanged.
        check(throws_niitb([&] { EquivariantCohomologyOnK x(&lssc_f, std::vector<std::vector<uint32_t>>{{0}}); },
                           "EquivariantCohomologyOnK: " + R),
              "(X) EquivariantCohomologyOnK (relative sub-complex) refuses an F_ell local system under Z/ell^2");
    }
    {   // another k in force
        hecke::ScopedCoefficientRing ring(3);
        check(throws_niitb([&] { hecke::HeckeConeStrictPairRoute x(act.get()); }, "HeckeConeStrictPairRoute: "),
              "(X) HeckeConeStrictPairRoute refuses a Z/ell^2 normative action under Z/ell^3");
        check(throws_niitb([&] { hecke::HeckeConeRelativeRoute x(act.get()); }, "HeckeConeRelativeRoute: "),
              "(X) HeckeConeRelativeRoute refuses a Z/ell^2 normative action under Z/ell^3");
    }
    // F_ell in force, composite input.
    check(throws_niitb([&] { hecke::HeckeConeStrictPairRoute x(act.get()); }, "HeckeConeStrictPairRoute: " + R),
          "(X) HeckeConeStrictPairRoute refuses a Z/ell^2 normative action under F_ell");
    check(throws_niitb([&] { hecke::HeckeConeRelativeRoute x(act.get()); }, "HeckeConeRelativeRoute: " + R),
          "(X) HeckeConeRelativeRoute refuses a Z/ell^2 normative action under F_ell");
    // F_ell everywhere: the field routes, unchanged (a smoke row; HeckeConeActionValidation owns them).
    bool field_ok = false;
    try {
        hecke::HeckeConeStrictPairRoute x(&act_f);
        x.compute();
        hecke::HeckeConeRelativeRoute y(&act_f);
        y.compute();
        field_ok = x.record().agrees() && y.record().agrees() && x.record().ring_k == 1 && y.record().ring_k == 1 && !y.is_composite();
    } catch (...) {}
    check(field_ok, "(X) at the default ring both field routes still agree (ring_k = 1)");
}

} // namespace

uint32_t HeckeConeCrossRouteRingValidation() {
    g_checks = 0;
    g_fail = 0;
    g_printed = 0;
    const auto t0 = std::chrono::steady_clock::now();
    std::cout << "HeckeConeCrossRouteRingValidation: [C] obl:coneexact(vii) over Z/ell^k -- routes 1, 2, 3 at the seven E8 points"
              << std::endl;
    const std::vector<PointSpec> points = {
        {10, 691, {2}, {2, 3, 5}},
        {6, 5, {2}, {2, 3}},
        {16, 5, {2}, {2, 3}},
        {30, 5, {2, 3}, {2, 3}},
        {60, 5, {2, 3}, {2, 3}},
    };
    for (size_t i = 0; i < points.size(); ++i) {
        try {
            cell_arm(points[i], i == 0);
        } catch (const std::exception& e) {
            check(false, "(R) cell (" + std::to_string(points[i].n) + ", " + std::to_string(points[i].ell) + ") threw: " + e.what());
        }
    }
    try { regime_arm(); } catch (const std::exception& e) { check(false, std::string("(W) threw: ") + e.what()); }
    try { closure_arm(); } catch (const std::exception& e) { check(false, std::string("(X) threw: ") + e.what()); }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "HeckeConeCrossRouteRingValidation: " << g_checks << " checks, " << g_fail << " failing, " << secs << " s" << std::endl;
    return g_fail;
}
