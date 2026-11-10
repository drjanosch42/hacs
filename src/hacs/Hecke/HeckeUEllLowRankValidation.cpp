// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 13, WP-4) HeckeUEllLowRankValidation -- the U_ell
// memory lever (HeckeOperatorOptions::ScopedUEllLowRank, HeckeOperatorOnK.h).
//
// The lever cannot change a value: every block of Theta_g is the same matrix
// mod ell^k, only bracketed differently (rho_n(g gamma_k) = U V, a sum of
// <= k outer products, since g gamma_k is singular mod ell whenever ell | N).
// This validation holds it to that, entrywise, on the shipped pipeline.
//
// (S24 CL-R1) The lever is RETIRED (owner ruling 2 of 2026-10-30): the arms
// (A)-(G) below compare the lever against its opt-out, so they live in
// Hecke/legacy/HeckeUEllLowRankLegacyArms.cpp and run only in an
// OC_ENABLE_LEGACY=ON build (OFF: `SKIP-ARM HeckeUEllLowRankValidation
// arm=lever: OC_ENABLE_LEGACY=OFF`).  Every build runs
//   (M) the default Auto mode: in effect exactly for U_ell (ell | N, SymN) and
//       not for T_2 (ell does not divide N); the tau table's rho pool is
//       factored exactly then; both operators pass O12(a); the dense-read trap
//       (RhoRefT::get() on the factored pool) throws -- (E-iii) on the
//       default pool.
// The lever arms (ON builds):
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
#include "HeckeUEllLowRankValidation.h"

#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "HeckeCosetTables.h"
#include "HeckeOperatorFamily.h"   // tau_construction_mutex()
#include "HeckeOperatorOnK.h"
#include "HeckeTauTable.h"
#include "SL2ZTessellationTables.h"
#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../Registry/LegacyValidation.h"   // oc::legacy::RunLegacyArm (S24 CL-R1)

namespace {


struct Checker {
    uint64_t checks = 0, failures = 0;
    void operator()(bool ok, const std::string& what) {
        ++checks;
        if (!ok) {
            ++failures;
            std::cout << "  [FAIL] " << what << "\n";
        }
    }
};

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

// (S24 CL-R1) (M) the default Auto mode, in every build.
void ArmMode(Checker& g) {
    namespace HeckeOperatorOptions = hecke::HeckeOperatorOptions;
    // An ON build started with an env opt-out (HECKE_U_ELL_LOW_RANK=0 / HECKE_COSET_STREAM=0) runs the old route by
    // default; (M) is about the default route, so it then says so and checks nothing (an OFF build refuses the env).
    if (HeckeOperatorOptions::u_ell_low_rank_mode() != HeckeOperatorOptions::UEllLowRankMode::Auto ||
        !HeckeOperatorOptions::coset_streaming()) {
        std::cout << "  (M) skipped: the process default is an env opt-out (OC_ENABLE_LEGACY=ON build)\n";
        return;
    }
    struct C { uint32_t n; uint32_t ell; };
    for (const C& c : std::vector<C>{ {10, 5}, {24, 13}, {10, 691}, {22, 691} }) {
        const std::string tag = "(M) n=" + std::to_string(c.n) + " ell=" + std::to_string(c.ell);
        try {
            const FieldCell cell(c.n, c.ell);
            for (uint64_t p : {static_cast<uint64_t>(c.ell), uint64_t{2}}) {
                const bool want = (p == c.ell);
                const std::string ptag = tag + (want ? " U_ell" : " T_2");
                g(hecke::HeckeOperatorOptions::u_ell_low_rank_in_effect(p, c.ell, FibreRepresentation::SymN) == want,
                  ptag + ": the lever in effect iff ell | N");
                g(!hecke::HeckeOperatorOptions::u_ell_low_rank_in_effect(p, c.ell, FibreRepresentation::ContragredientN),
                  ptag + ": never in effect for the contragredient fibre");
                std::unique_ptr<hecke::HeckeCosetTables> cosets;
                std::unique_ptr<hecke::HeckeTauTable> tau;
                {
                    std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
                    cosets.reset(new hecke::HeckeCosetTables(p, 1, c.ell, c.n));
                    tau.reset(new hecke::HeckeTauTable(*cell.K, *cell.model, *cosets, c.ell, c.n));
                }
                g(cosets->streamed(), ptag + ": the coset table streams (the retired HECKE_COSET_STREAM's default)");
                g(tau->rho_pool_is_low_rank() == want, ptag + ": the tau pool is factored iff the lever is in effect");
                if (want) {   // (E-iii) on the default pool: the dense-read trap
                    bool trapped = false, any = false;
                    try {
                        const auto& xts = tau->xi_terms(1, 0, 0);
                        if (!xts.empty()) {
                            any = true;
                            (void)xts.front().rho_g_gamma.get();
                        }
                    } catch (const std::logic_error&) {
                        trapped = true;
                    } catch (const std::exception&) {
                    }
                    g(any && trapped, ptag + ": RhoRefT::get() on the factored default pool throws");
                }
                hecke::HeckeOperatorOnK op(cell.coh.get(), cosets.get(), tau.get());
                op.compute();
                g(op.certificates().routes_bit_identical, ptag + ": O12(a)");
            }
            std::cout << "  " << tag << ": Auto in effect for U_ell only, pools as expected\n";
        } catch (const std::exception& e) {
            g(false, tag + " threw: " + e.what());
        }
    }
}

} // namespace

uint32_t HeckeUEllLowRankValidation() {
    std::cout << "=== HeckeUEllLowRankValidation (slot-13 WP-4: the U_ell memory lever) ===\n";
    Checker g;
    ArmMode(g);
#if OC_ENABLE_LEGACY
    if (oc::legacy::RunLegacyArm("HeckeUEllLowRankValidation", "lever"))
        HeckeUEllLowRankLeverArms([&g](bool ok, const std::string& what) { g(ok, what); });
#else
    (void)oc::legacy::RunLegacyArm("HeckeUEllLowRankValidation", "lever");
#endif
    std::cout << "HeckeUEllLowRankValidation: " << g.checks << " checks, " << g.failures << " failures\n";
    return g.failures == 0 ? 0u : 1u;
}
