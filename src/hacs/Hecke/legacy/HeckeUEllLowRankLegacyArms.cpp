// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Hecke/legacy/HeckeUEllLowRankLegacyArms.cpp -- the LEVER arms (A)-(G) of HeckeUEllLowRankValidation: the RETIRED
// U_ell low-rank lever HECKE_U_ELL_LOW_RANK=0 / ScopedUEllLowRank (slot-13 WP-4, 3015a534; retired by S24 CL-R1, owner
// ruling 2 of 2026-10-30; CODEAUDIT-B B6).  Compiled only with OC_ENABLE_LEGACY=ON (cmake/legacy.cmake); called from
// HeckeUEllLowRankValidation.cpp under oc::legacy::RunLegacyArm("HeckeUEllLowRankValidation", "lever").  Extracted
// verbatim from HeckeUEllLowRankValidation.cpp (FieldCell is a copy of the one that stays there); only the Checker
// became the `check` callback.
//
// (hecke-extensions slot 13, WP-4) HeckeUEllLowRankValidation -- the U_ell
// memory lever (HeckeOperatorOptions::ScopedUEllLowRank, HeckeOperatorOnK.h).
//
// The lever cannot change a value: every block of Theta_g is the same matrix
// mod ell^k, only bracketed differently (rho_n(g gamma_k) = U V, a sum of
// <= k outer products, since g gamma_k is singular mod ell whenever ell | N).
// This validation holds it to that, entrywise, on the shipped pipeline:
//   (A) F_ell, U_ell at ell in {5, 7, 11, 13, 691} and small n (plus T_{ell^2}
//       at 5 and 7): A^{(q)}, Theta_flat^{(q)}, Theta_eq^{(q)} for every q and
//       the certificate bundle equal to the default route's.  The rank claim
//       is asserted per block INSIDE the lever (ENGINE_CHECK in the
//       factoriser: <= min(k, n + 1) outer products), so a run that returns
//       has passed it on every pool block and every route-2 block.
//   (B) the refusal: the lever on T_2 (ell does not divide N) throws, and the
//       default route is unaffected afterwards.
//   (C) Z/ell^2 at ell in {5, 7, 11, 13, 691} and Z/5^3 (the module route):
//       H_matrix_module(q) and Theta_module(q) equal to the default route's.
//   (D) printed, not gated: the peak RSS of `U_691` at n = 128 with and
//       without the lever (the step-0 anchor of the WP-4 report).
//   (E) (S13 WP-14) the three lever guards.
//   (F) (S13 WP-16) the LEGACY arm (ScopedLegacyHeckeOperatorPath +
//       engine::elim::ScopedLegacy, i.e. what ENGINE_BACKEND=legacy selects,
//       reachable in the default build): the lever's A^(q), Theta_flat^(q),
//       Theta_eq^(q) on the legacy arm equal to the default (engine, no lever)
//       route's and to the legacy arm's without the lever; the legacy arm
//       reads the factored tau pool as U V.  Plus the lever's two refusals on
//       the legacy arm (T_2; a factored tau pool computed without the lever).
//   (G) (slot-14 fix of the WP-17 section 8.2 cross-config finding) the
//       CONTRAGREDIENT fibre over a factored tau pool: E7's ContragredientCell
//       builds ONE tau table for U_5 and computes op_M (SymN, lever in effect,
//       pool factored) and op_D (ContragredientN, lever never in effect) on it.
//       Under the default Auto lever the legacy arm's fused route read the
//       factored SymN pool for op_D too and failed O12(a) at every e7 golden
//       point at ell = 5.  Both operators on the legacy arm under Auto must
//       equal the default route's with the lever off.
#include "../HeckeUEllLowRankValidation.h"

#if !OC_ENABLE_LEGACY
#error "HeckeUEllLowRankLegacyArms.cpp is an OC_ENABLE_LEGACY=ON source (cmake/legacy.cmake)"
#endif

#include <cstdint>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "../CoefficientRing.h"
#include "../HeckeContragredient.h"   // (G) ContragredientCell
#include "../HeckeCosetTables.h"
#include "../HeckeOperatorFamily.h"   // tau_construction_mutex()
#include "../HeckeOperatorOnK.h"
#include "../HeckeSettings.h"
#include "../HeckeTauTable.h"
#include "../SL2ZTessellationTables.h"
#include "../../EquivariantCohomologyOnK.h"
#include "../../LocalSystemSL2Z.h"
#include "../../LocalSystemSimplicialComplex.h"
#include "../../Util/Rss.h"
#include "../../../ModArithmetic/DenseElim.h"   // engine::elim::ScopedLegacy (S13 WP-16)

namespace {

using Check = std::function<void(bool, const std::string&)>;

struct FieldCell {
    std::unique_ptr<SimplicialComplex> K;
    std::unique_ptr<hecke::SL2ZTessellationTables> model;
    std::unique_ptr<LocalSystemSimplicialComplex> lssc;
    std::unique_ptr<EquivariantCohomologyOnK> coh;
    FieldCell(uint32_t n, uint32_t ell) {
        {
            std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
            K.reset(new SimplicialComplex(CreateSL2ZFundamentalDomainTriangulation()));
            model.reset(new hecke::SL2ZTessellationTables(*K));
        }
        std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels;
        for (const auto& datum : GenerateSL2ZRelationsWithGroupData(*K, n, ell)) rels.push_back(datum.rel);
        lssc.reset(new LocalSystemSimplicialComplex(*K, std::move(rels), n + 1, ell));
        coh.reset(new EquivariantCohomologyOnK(lssc.get()));
        coh->compute_cohomology();
    }
};

// One operator T_{p^r} on a computed F_ell cell, with or without the lever.
struct FieldOp {
    std::unique_ptr<hecke::HeckeCosetTables> cosets;
    std::unique_ptr<hecke::HeckeTauTable> tau;
    std::unique_ptr<hecke::HeckeOperatorOnK> op;
    FieldOp(const FieldCell& c, uint64_t p, uint32_t r, uint32_t n, uint32_t ell, bool lever, bool legacy = false) {
        // The lever is read by the tau table at construction (the factored
        // pool) and by compute(): one scope around both.  `legacy` (S13 WP-16)
        // selects the legacy arm for compute() in-process.
        hecke::HeckeOperatorOptions::ScopedUEllLowRank guard(lever);
        std::unique_ptr<engine::elim::ScopedLegacy> legacy_elim;
        if (legacy) legacy_elim.reset(new engine::elim::ScopedLegacy());
        hecke::HeckeOperatorOptions::ScopedLegacyHeckeOperatorPath legacy_path(
            legacy || hecke::HeckeOperatorOptions::legacy_hecke_operator_path_forced());
        {
            std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
            cosets.reset(new hecke::HeckeCosetTables(p, r, ell, n));
            tau.reset(new hecke::HeckeTauTable(*c.K, *c.model, *cosets, ell, n));
        }
        op.reset(new hecke::HeckeOperatorOnK(c.coh.get(), cosets.get(), tau.get()));
        op->compute();
    }
};

void ArmField(const Check& g) {
    struct C { uint32_t n; uint32_t ell; uint32_t r; };
    const std::vector<C> cells = {
        {2, 5, 1},   {10, 5, 1},  {24, 5, 1},  {0, 7, 1},   {10, 7, 1},  {22, 7, 1},  {12, 11, 1},
        {20, 11, 1}, {12, 13, 1}, {24, 13, 1}, {10, 691, 1}, {22, 691, 1}, {36, 691, 1},
        {10, 5, 2},  {12, 7, 2},
    };
    for (const C& c : cells) {
        const std::string tag = "(A) n=" + std::to_string(c.n) + " ell=" + std::to_string(c.ell) +
                                " T_{ell^" + std::to_string(c.r) + "}";
        try {
            const FieldCell cell(c.n, c.ell);
            const FieldOp def(cell, c.ell, c.r, c.n, c.ell, false);
            const FieldOp lev(cell, c.ell, c.r, c.n, c.ell, true);
            const auto& A0 = def.op->get_H_matrices();
            const auto& A1 = lev.op->get_H_matrices();
            const auto& T0 = def.op->get_Theta_flat_matrices();
            const auto& T1 = lev.op->get_Theta_flat_matrices();
            const auto& E0 = def.op->get_Theta_eq_matrices();
            const auto& E1 = lev.op->get_Theta_eq_matrices();
            g(A0.size() == A1.size() && T0.size() == T1.size() && E0.size() == E1.size(), tag + " degree count");
            for (size_t q = 0; q < A0.size() && q < A1.size(); ++q) {
                g(A0[q].data == A1[q].data, tag + " A^(" + std::to_string(q) + ") entrywise");
                g(T0[q] == T1[q], tag + " Theta_flat^(" + std::to_string(q) + ") entrywise");
                g(E0[q] == E1[q], tag + " Theta_eq^(" + std::to_string(q) + ") entrywise");
            }
            const auto c0 = def.op->certificates(), c1 = lev.op->certificates();
            g(c0.routes_bit_identical && c1.routes_bit_identical, tag + " O12(a) on both routes");
            g(c0.relation_checks == c1.relation_checks && c0.cocycle_checks == c1.cocycle_checks &&
                  c0.theta_eq_solves == c1.theta_eq_solves && c1.theta_eq_residual_max == 0,
              tag + " certificate counters equal");
            std::cout << "  " << tag << ": dims";
            for (size_t q = 0; q < A0.size(); ++q) std::cout << " " << A0[q].data.size();
            std::cout << " -- equal\n";
        } catch (const std::exception& e) {
            g(false, tag + " threw: " + e.what());
        }
    }
}

void ArmRefusal(const Check& g) {
    const uint32_t n = 10, ell = 691;
    const FieldCell cell(n, ell);
    bool refused = false;
    try {
        const FieldOp lev(cell, 2, 1, n, ell, true);
    } catch (const std::logic_error& e) {
        refused = std::string(e.what()).find("ell | N") != std::string::npos;
    } catch (const std::exception&) {
    }
    g(refused, "(B) the lever refuses T_2 at ell = 691 (ell does not divide N)");
    try {
        const FieldOp def(cell, 2, 1, n, ell, false);
        g(def.op->certificates().routes_bit_identical, "(B) the default T_2 afterwards");
    } catch (const std::exception& e) {
        g(false, std::string("(B) default T_2 threw: ") + e.what());
    }
    std::cout << "  (B) T_2 at (10, 691) refused under the lever: " << (refused ? "yes" : "NO") << "\n";
}

void ArmModule(const Check& g) {
    struct C { uint32_t n; uint64_t ell; uint32_t k; };
    const std::vector<C> cells = {
        {10, 5, 2}, {24, 5, 2}, {10, 7, 2}, {22, 7, 2}, {12, 11, 2}, {12, 13, 2}, {10, 691, 2}, {24, 5, 3},
    };
    for (const C& c : cells) {
        const std::string tag = "(C) n=" + std::to_string(c.n) + " Z/" + std::to_string(c.ell) + "^" + std::to_string(c.k);
        try {
            const hecke::ScopedCoefficientRing ring(c.k);
            const engine::ModCtx<uint64_t> ctx =
                hecke::CoefficientRingContext<uint64_t>("HeckeUEllLowRankValidation", c.ell, c.k);
            std::unique_ptr<SimplicialComplex> K;
            std::unique_ptr<hecke::SL2ZTessellationTables> model;
            {
                std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
                K.reset(new SimplicialComplex(CreateSL2ZFundamentalDomainTriangulation()));
                model.reset(new hecke::SL2ZTessellationTables(*K));
            }
            LocalSystemSimplicialComplexT<uint64_t> lssc(*K, hecke::GenerateSL2ZRelationsRing<uint64_t>(*K, c.n, ctx),
                                                         c.n + 1, ctx.p);
            EquivariantCohomologyOnKT<uint64_t> coh(&lssc);
            coh.compute_cohomology();
            std::unique_ptr<hecke::HeckeCosetTablesT<uint64_t>> cosets;
            std::unique_ptr<hecke::HeckeTauTableT<uint64_t>> tau;
            {
                std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
                cosets.reset(new hecke::HeckeCosetTablesT<uint64_t>(c.ell, 1, c.ell, c.n));
                tau.reset(new hecke::HeckeTauTableT<uint64_t>(*K, *model, *cosets, c.ell, c.n));
            }
            hecke::HeckeOperatorOnKT<uint64_t> def(&coh, cosets.get(), tau.get());
            {
                hecke::HeckeOperatorOptions::ScopedUEllLowRank off(false);   // the opt-out oracle (owner decision 2026-09-28)
                def.compute();
            }
            hecke::HeckeOperatorOnKT<uint64_t> lev(&coh, cosets.get(), tau.get());
            {
                hecke::HeckeOperatorOptions::ScopedUEllLowRank guard(true);
                lev.compute();
            }
            for (uint32_t q = 0; q < 3; ++q) {
                const auto& A0 = def.H_matrix_module(q);
                const auto& A1 = lev.H_matrix_module(q);
                bool eq = A0.rows == A1.rows && A0.cols == A1.cols;
                for (uint32_t i = 0; eq && i < A0.rows; ++i)
                    for (uint32_t j = 0; j < A0.cols; ++j)
                        if (A0.at(i, j) != A1.at(i, j)) { eq = false; break; }
                g(eq, tag + " A^(" + std::to_string(q) + ") on the cyclic summands entrywise");
                const auto& T0 = def.Theta_module(q);
                const auto& T1 = lev.Theta_module(q);
                bool teq = T0.rows == T1.rows && T0.cols == T1.cols;
                for (uint32_t i = 0; teq && i < T0.rows; ++i)
                    for (uint32_t j = 0; j < T0.cols; ++j)
                        if (T0.at(i, j) != T1.at(i, j)) { teq = false; break; }
                g(teq, tag + " Theta^(" + std::to_string(q) + ") over the ring entrywise");
            }
            std::cout << "  " << tag << ": H^1 summands " << def.H_matrix_module(1).rows << " -- equal\n";
        } catch (const std::exception& e) {
            g(false, tag + " threw: " + e.what());
        }
    }
}

void ArmRss() {
    const uint32_t n = 128, ell = 691;
    const FieldCell cell(n, ell);
    double peak[2] = {0.0, 0.0};
    // The lever FIRST: a freed-but-retained heap from the default run would
    // otherwise hide the lever's own peak.
    for (int lever = 1; lever >= 0; --lever) {
        const bool reset = oc_util::ResetPeakRss();
        const double base = oc_util::CurrentRssMiB();
        {
            const FieldOp op(cell, ell, 1, n, ell, lever == 1);
            (void)op;
        }
        peak[lever] = reset ? oc_util::PeakRssMiB() - base : -1.0;
    }
    std::cout << std::fixed << std::setprecision(1) << "  (D) U_691 at n = 128 (cosets + tau + operator), peak RSS above "
              << "the cohomology: default " << peak[0] << " MiB, lever " << peak[1]
              << " MiB (printed, not gated; -1 = VmHWM reset unavailable)\n";
}

// (S13 WP-14, A1) the three lever guards / branches the audit found untested.
void ArmGuards(const Check& g) {
    const uint32_t n = 10, ell = 691;
    const FieldCell cell(n, ell);
    // (i) tau built OUTSIDE the scope, compute inside: the !tau_factored branch.
    try {
        const FieldOp def(cell, ell, 1, n, ell, false);
        std::unique_ptr<hecke::HeckeCosetTables> cosets;
        std::unique_ptr<hecke::HeckeTauTable> tau;
        {
            std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
            hecke::HeckeOperatorOptions::ScopedUEllLowRank off(false);   // (owner decision 2026-09-28: Auto is the default)
            cosets.reset(new hecke::HeckeCosetTables(ell, 1, ell, n));
            tau.reset(new hecke::HeckeTauTable(*cell.K, *cell.model, *cosets, ell, n));
        }
        hecke::HeckeOperatorOnK op(cell.coh.get(), cosets.get(), tau.get());
        {
            hecke::HeckeOperatorOptions::ScopedUEllLowRank guard(true);
            op.compute();
        }
        const auto& A0 = def.op->get_H_matrices();
        const auto& A1 = op.get_H_matrices();
        bool eq = A0.size() == A1.size();
        for (size_t q = 0; eq && q < A0.size(); ++q) eq = A0[q].data == A1[q].data;
        g(eq, "(E-i) tau outside the scope, compute inside: A^(q) equal to the default");
    } catch (const std::exception& e) {
        g(false, std::string("(E-i) threw: ") + e.what());
    }
    // (ii) tau built inside the scope, compute outside: the loud refusal.
    {
        bool refused = false;
        std::unique_ptr<hecke::HeckeCosetTables> cosets;
        std::unique_ptr<hecke::HeckeTauTable> tau;
        try {
            {
                hecke::HeckeOperatorOptions::ScopedUEllLowRank guard(true);
                std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
                cosets.reset(new hecke::HeckeCosetTables(ell, 1, ell, n));
                tau.reset(new hecke::HeckeTauTable(*cell.K, *cell.model, *cosets, ell, n));
            }
            hecke::HeckeOperatorOnK op(cell.coh.get(), cosets.get(), tau.get());
            hecke::HeckeOperatorOptions::ScopedUEllLowRank off(false);   // (owner decision 2026-09-28: Auto is the default)
            op.compute();
        } catch (const std::logic_error& e) {
            refused = std::string(e.what()).find("ScopedUEllLowRank") != std::string::npos;
        } catch (const std::exception&) {
        }
        g(refused, "(E-ii) tau under the lever, compute not: refused naming ScopedUEllLowRank");
    }
    // (iii) RhoRefT::get() on a factored pool: the dense-read trap.
    {
        bool trapped = false;
        try {
            hecke::HeckeOperatorOptions::ScopedUEllLowRank guard(true);
            std::unique_ptr<hecke::HeckeCosetTables> cosets;
            std::unique_ptr<hecke::HeckeTauTable> tau;
            {
                std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
                cosets.reset(new hecke::HeckeCosetTables(ell, 1, ell, n));
                tau.reset(new hecke::HeckeTauTable(*cell.K, *cell.model, *cosets, ell, n));
            }
            bool any = false;
            const auto& xts = tau->xi_terms(1, 0, 0);
            if (!xts.empty()) {
                any = true;
                (void)xts.front().rho_g_gamma.get();
            }
            g(any, "(E-iii) the table has a term");
        } catch (const std::logic_error&) {
            trapped = true;
        } catch (const std::exception&) {
        }
        g(trapped, "(E-iii) RhoRefT::get() on a factored pool throws");
    }
}

// (S13 WP-16) The lever on the LEGACY arm, in-process (the default build).
// Before WP-16 the legacy arm read the factored pool densely and hit the
// RhoRefT trap (ENGINE_BACKEND=legacy: SIGABRT through the unguarded harness).
void ArmLegacy(const Check& g) {
    struct C { uint32_t n; uint32_t ell; uint32_t r; };
    const std::vector<C> cells = { {10, 5, 1}, {24, 13, 1}, {10, 691, 1}, {22, 691, 1}, {12, 7, 2} };
    for (const C& c : cells) {
        const std::string tag = "(F) legacy arm n=" + std::to_string(c.n) + " ell=" + std::to_string(c.ell) +
                                " T_{ell^" + std::to_string(c.r) + "}";
        try {
            const FieldCell cell(c.n, c.ell);
            const FieldOp def(cell, c.ell, c.r, c.n, c.ell, false, false);
            const FieldOp leg(cell, c.ell, c.r, c.n, c.ell, false, true);
            const FieldOp lev(cell, c.ell, c.r, c.n, c.ell, true, true);
            g(lev.tau->rho_pool_is_low_rank(), tag + ": the tau pool is factored under the lever");
            const auto& A0 = def.op->get_H_matrices();
            const auto& AL = leg.op->get_H_matrices();
            const auto& A1 = lev.op->get_H_matrices();
            const auto& T0 = def.op->get_Theta_flat_matrices();
            const auto& T1 = lev.op->get_Theta_flat_matrices();
            const auto& E0 = def.op->get_Theta_eq_matrices();
            const auto& E1 = lev.op->get_Theta_eq_matrices();
            g(A0.size() == A1.size() && AL.size() == A1.size() && T0.size() == T1.size() && E0.size() == E1.size(),
              tag + " degree count");
            for (size_t q = 0; q < A0.size() && q < A1.size() && q < AL.size(); ++q) {
                g(A0[q].data == A1[q].data, tag + " A^(" + std::to_string(q) + ") = the default route's");
                g(AL[q].data == A1[q].data, tag + " A^(" + std::to_string(q) + ") = the legacy arm's without the lever");
                g(T0[q] == T1[q], tag + " Theta_flat^(" + std::to_string(q) + ") = the default route's");
                g(E0[q] == E1[q], tag + " Theta_eq^(" + std::to_string(q) + ") = the default route's");
            }
            g(lev.op->certificates().routes_bit_identical, tag + " O12(a) on the legacy arm under the lever");
            std::cout << "  " << tag << ": equal\n";
        } catch (const std::exception& e) {
            g(false, tag + " threw: " + e.what());
        }
    }
    const uint32_t n = 10, ell = 691;
    const FieldCell cell(n, ell);
    {   // T_2 under the lever, legacy arm: the same refusal as compute_block's.
        bool refused = false;
        try {
            const FieldOp lev(cell, 2, 1, n, ell, true, true);
        } catch (const std::logic_error& e) {
            refused = std::string(e.what()).find("ell | N") != std::string::npos;
        } catch (const std::exception&) {
        }
        g(refused, "(F) legacy arm: the lever refuses T_2 at ell = 691 (ell | N)");
    }
    {   // tau factored, legacy compute without the lever: refused by name.
        bool refused = false;
        try {
            std::unique_ptr<hecke::HeckeCosetTables> cosets;
            std::unique_ptr<hecke::HeckeTauTable> tau;
            {
                hecke::HeckeOperatorOptions::ScopedUEllLowRank guard(true);
                std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
                cosets.reset(new hecke::HeckeCosetTables(ell, 1, ell, n));
                tau.reset(new hecke::HeckeTauTable(*cell.K, *cell.model, *cosets, ell, n));
            }
            engine::elim::ScopedLegacy legacy_elim;
            hecke::HeckeOperatorOptions::ScopedLegacyHeckeOperatorPath legacy_path(true);
            hecke::HeckeOperatorOnK op(cell.coh.get(), cosets.get(), tau.get());
            hecke::HeckeOperatorOptions::ScopedUEllLowRank off(false);   // (owner decision 2026-09-28: Auto is the default)
            op.compute();
        } catch (const std::logic_error& e) {
            refused = std::string(e.what()).find("ScopedUEllLowRank") != std::string::npos;
        } catch (const std::exception&) {
        }
        g(refused, "(F) legacy arm: a factored tau pool computed without the lever is refused naming ScopedUEllLowRank");
    }
}

// (G) The contragredient fibre over a factored tau pool, on the legacy arm.
void ArmContragredientLegacy(const Check& g) {
    namespace HeckeOperatorOptions = hecke::HeckeOperatorOptions;
    for (uint32_t n : {10u, 24u}) {
        const uint32_t ell = 5;
        const std::string tag = "(G) E7 legacy arm n=" + std::to_string(n) + " ell=5 U_5";
        try {
            const hecke::ContragredientCell cell(n, ell);
            hecke::ContragredientCell::HeckePair ref, leg;
            {   // the oracle: default (engine) route, lever off
                HeckeOperatorOptions::ScopedUEllLowRank off(false);
                ref = cell.hecke_pair(ell);
            }
            {   // the legacy arm under the default Auto lever (what ENGINE_BACKEND=legacy runs)
                HeckeOperatorOptions::ScopedUEllLowRank autom(HeckeOperatorOptions::UEllLowRankMode::Auto);
                engine::elim::ScopedLegacy legacy_elim;
                HeckeOperatorOptions::ScopedLegacyHeckeOperatorPath legacy_path(true);
                leg = cell.hecke_pair(ell);
            }
            g(leg.tau->rho_pool_is_low_rank(), tag + ": the shared tau pool is factored under Auto");
            const hecke::HeckeOperatorOnK* R[2] = { ref.op_M.get(), ref.op_D.get() };
            const hecke::HeckeOperatorOnK* L[2] = { leg.op_M.get(), leg.op_D.get() };
            const char* fib[2] = { " SymN", " ContragredientN" };
            for (int f = 0; f < 2; ++f) {
                const auto& A0 = R[f]->get_H_matrices();
                const auto& A1 = L[f]->get_H_matrices();
                const auto& T0 = R[f]->get_Theta_flat_matrices();
                const auto& T1 = L[f]->get_Theta_flat_matrices();
                g(A0.size() == A1.size() && T0.size() == T1.size(), tag + fib[f] + " degree count");
                for (size_t q = 0; q < A0.size() && q < A1.size(); ++q) {
                    g(A0[q].data == A1[q].data, tag + fib[f] + " A^(" + std::to_string(q) + ") = the default route's");
                    g(T0[q] == T1[q], tag + fib[f] + " Theta_flat^(" + std::to_string(q) + ") = the default route's");
                }
                g(L[f]->certificates().routes_bit_identical, tag + fib[f] + " O12(a) on the legacy arm");
            }
            std::cout << "  " << tag << ": op_M and op_D equal\n";
        } catch (const std::exception& e) {
            g(false, tag + " threw: " + e.what());
        }
    }
}

} // namespace

void HeckeUEllLowRankLeverArms(const std::function<void(bool, const std::string&)>& check) {
    ArmField(check);
    ArmRefusal(check);
    ArmModule(check);
    ArmGuards(check);
    ArmLegacy(check);
    ArmContragredientLegacy(check);
    ArmRss();
}
