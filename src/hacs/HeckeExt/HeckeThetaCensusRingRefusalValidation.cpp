// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 12, WP-1 -- HeckeThetaCensusRingRefusalValidation.
//
// Carry F-1 ([X] v1.15 Sec. 8 "WP-15 F-1"; [E] sec:ring (b1)): every closure
// member of the theta / census line must refuse a non-default
// coefficient_ring, THROUGH A NAMED GUARD NAMING ITSELF, so a caller that
// reaches a member directly (bypassing the top-level driver) still gets a
// named refusal rather than a silently-wrong F_ell answer. Two members
// already refused before this package (via a guard belonging to a DIFFERENT
// object further up the call chain): HeckeCensusSplit (hecke_ext::
// ComputeCensusSplit) THROUGH HeckeExtDriver's own constructor-entry guard
// (HeckeExt/HeckeExtDriver.cpp), and HeckeOperatorOnK's QuotientL fibre
// representation THROUGH the SAME "contragredient" guard the ContragredientN
// fibre already used (Hecke/HeckeOperatorOnK.cpp, init_composite(): "if
// (rep_ != FibreRepresentation::SymN) RefuseCoefficientRing(\"HeckeOperatorOnK
// (contragredient)\", ...)" covers rep_ == QuotientL too, unconditionally).
// Two members had NO guard of their own and GAIN ONE here: the Q_n relation
// list (GenerateSL2ZQuotientLRelationsWithGroupData, LocalSystemSL2Z.cpp) and
// the Q_n fibre builder (ComputeQuotientLMatrixFM, LocalSystemUtils.cpp) --
// both purely F_ell routines with no notion of a ring, so a caller reaching
// either one directly under a composite ring setting used to get an F_ell
// answer with no diagnostic at all. Neither addition changes a production
// value: hecke_ext::DicksonThetaCell's constructor already refuses FIRST
// (HeckeDicksonTheta.cpp, "hecke::RequireFieldCoefficientRing(\"DicksonThetaCell\")"),
// before either function is ever reached on the shipped path -- the two new
// guards are purely defensive, reached only by a caller that skips the
// driver.
//
// Five closure members, one arm each: (kind) the exception raised under
// coefficient_ring = Z/ell^2 is exactly hecke::NotImplementedInThisBuild (not
// some other exception, not a silent pass); (name) its text names the member
// itself; (default) the SAME call still succeeds, unchanged, at the default
// ring F_ell -- proving the new guards do not misfire on the shipped path.

#include "HeckeCensusSplit.h"
#include "HeckeDicksonTheta.h"

#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../LocalSystemUtils.h"
#include "../SimplicialCohomology.h"
#include "../Hecke/CoefficientRing.h"
#include "../Hecke/HeckeCosetTables.h"
#include "../Hecke/HeckeOperatorOnK.h"
#include "../Hecke/HeckeSettings.h"
#include "../Hecke/HeckeTauTable.h"
#include "../Hecke/SL2ZTessellationTables.h"
#include "../../ModArithmetic/FlatMatrix.h"
#include "../../ModArithmetic/ModCtx.h"

#include <chrono>
#include <cstdio>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

struct Counter {
    uint64_t checks = 0, failures = 0;
    void check(bool ok, const std::string& what) {
        ++checks;
        if (!ok) {
            ++failures;
            std::cout << "  [FAIL] " << what << std::endl;
        }
    }
};

// 0 = no throw at all; 1 = threw, but not hecke::NotImplementedInThisBuild
// (the wrong kind); 2 = threw hecke::NotImplementedInThisBuild ("what_out"
// receives its text).
int ProbeThrow(const std::function<void()>& f, std::string* what_out) {
    try {
        f();
    } catch (const hecke::NotImplementedInThisBuild& e) {
        if (what_out) *what_out = e.what();
        return 2;
    } catch (...) {
        return 1;
    }
    return 0;
}

// One closure member: `composite_call` must refuse (kind + name); the SAME
// underlying computation, `default_call`, must still succeed at F_ell.
void ProbeMember(Counter& ct, const std::string& label, const std::string& who,
                 const std::function<void()>& composite_call, const std::function<void()>& default_call) {
    std::string what;
    const int kind = ProbeThrow(composite_call, &what);
    ct.check(kind == 2, "(kind) " + label + " refuses via hecke::NotImplementedInThisBuild under "
                             "coefficient_ring=Z/ell^2 (got " +
                             (kind == 0 ? std::string("no throw") : std::string("the wrong exception kind")) + ")");
    ct.check(kind == 2 && what.find(who) != std::string::npos,
             "(name) " + label + " names itself (\"" + who + "\") in the refusal text; got: " + what);
    bool default_ok = false;
    std::string default_err;
    try {
        default_call();
        default_ok = true;
    } catch (const std::exception& e) {
        default_err = e.what();
    } catch (...) {
        default_err = "non-std::exception";
    }
    ct.check(default_ok, "(default) " + label + " still computes at the default ring F_ell (no regression)" +
                              (default_ok ? "" : "; threw: " + default_err));
}

uint32_t Finish(const char* name, const Counter& ct, std::chrono::steady_clock::time_point t0) {
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("%s: %llu checks, %llu failing (%.1f s)\n", name, static_cast<unsigned long long>(ct.checks),
                static_cast<unsigned long long>(ct.failures), s);
    return ct.failures == 0 ? 0u : 1u;
}

} // namespace

uint32_t HeckeThetaCensusRingRefusalValidation() {
    const auto t0 = std::chrono::steady_clock::now();
    Counter ct;

    const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    constexpr uint32_t kN = 4, kEll = 5;  // small, cheap Q_n cell (n + ell + 1 = 10)

    // ---- member 1: the Q_n relation list (GenerateSL2ZQuotientLRelationsWithGroupData) --------
    ProbeMember(
        ct, "the Q_n relation list (GenerateSL2ZQuotientLRelationsWithGroupData)",
        "GenerateSL2ZQuotientLRelationsWithGroupData",
        [&] {
            hecke::ScopedCoefficientRing ring(2);
            (void)GenerateSL2ZQuotientLRelationsWithGroupData(K, kN, kEll);
        },
        [&] { (void)GenerateSL2ZQuotientLRelationsWithGroupData(K, kN, kEll); });

    // ---- member 2: the Q_n fibre builder (ComputeQuotientLMatrixFM) ----------------------------
    const FlatMatrix identity2x2(2, 2, kEll, std::vector<uint32_t>{ 1, 0, 0, 1 });
    ProbeMember(
        ct, "the Q_n fibre builder (ComputeQuotientLMatrixFM)", "ComputeQuotientLMatrixFM",
        [&] {
            hecke::ScopedCoefficientRing ring(2);
            (void)ComputeQuotientLMatrixFM(identity2x2, kN);
        },
        [&] { (void)ComputeQuotientLMatrixFM(identity2x2, kN); });

    // ---- member 3: the theta driver itself (DicksonThetaCell), pre-existing (F-2, slot-11) -----
    ProbeMember(
        ct, "the theta driver (hecke_ext::DicksonThetaCell)", "DicksonThetaCell",
        [&] {
            hecke::ScopedCoefficientRing ring(2);
            hecke_ext::DicksonThetaCell c(kN, kEll);
        },
        [&] { hecke_ext::DicksonThetaCell c(kN, kEll); });

    // ---- member 4: the census split, THROUGH HeckeExtDriver's own guard (pre-existing) --------
    // ell = 691 is not a Glover ell ({5,7,11,13}), so (d6) is skipped ("not_evaluated:ell")
    // regardless of `weights` -- the cell is as cheap as any other (10, 691) run elsewhere in
    // this suite.
    ProbeMember(
        ct, "the census split (hecke_ext::ComputeCensusSplit, through HeckeExtDriver)", "HeckeExtDriver",
        [&] {
            hecke::ScopedCoefficientRing ring(2);
            (void)hecke_ext::ComputeCensusSplit(10, 691, hecke_ext::CensusSplitOptions{}, nullptr, nullptr);
        },
        [&] { (void)hecke_ext::ComputeCensusSplit(10, 691, hecke_ext::CensusSplitOptions{}, nullptr, nullptr); });

    // ---- member 5: HeckeOperatorOnK(QuotientL), THROUGH the "contragredient" guard (pre-existing) --
    // Positive arm: a genuine Q_n cohomology object at F_ell -- the SAME kind of object
    // HeckeDicksonThetaValidation's (Hk) arm builds.
    const std::vector<SL2ZRelationDatum> dq = GenerateSL2ZQuotientLRelationsWithGroupData(K, kN, kEll);
    std::vector<LocalSystemSimplicialComplex::SimplexRelation> relsQ;
    relsQ.reserve(dq.size());
    for (const SL2ZRelationDatum& d : dq) relsQ.push_back(d.rel);
    LocalSystemSimplicialComplex lsscQ(K, relsQ, kEll + 1, kEll);
    EquivariantCohomologyOnK cohQ(&lsscQ);
    cohQ.compute_cohomology();
    hecke::SL2ZTessellationTables model(K);
    hecke::HeckeCosetTables cosets(2, 1, kEll, kN);
    hecke::HeckeTauTable tau(K, model, cosets, kEll, kN);

    // Negative arm: any composite-ring cohomology object (SymN shape; init_composite()'s
    // contragredient guard fires before the representation's dimension is even read, so the
    // object need not itself be Q_n-shaped).
    std::unique_ptr<LocalSystemSimplicialComplex> lsscC;
    std::unique_ptr<EquivariantCohomologyOnK> cohC;
    {
        hecke::ScopedCoefficientRing ring(2);
        const engine::ModCtx<uint32_t> ctx =
            hecke::CoefficientRingContext<uint32_t>("HeckeThetaCensusRingRefusalValidation", kEll, 2);
        std::vector<LocalSystemSimplicialComplex::SimplexRelation> relsC =
            hecke::GenerateSL2ZRelationsRing<uint32_t>(K, kN, ctx);
        lsscC = std::make_unique<LocalSystemSimplicialComplex>(K, std::move(relsC), kN + 1, ctx.p);
        cohC = std::make_unique<EquivariantCohomologyOnK>(lsscC.get());
        cohC->compute_cohomology();
    }
    ProbeMember(
        ct, "HeckeOperatorOnK(FibreRepresentation::QuotientL)", "HeckeOperatorOnK (contragredient)",
        [&] {
            hecke::ScopedCoefficientRing ring(2);
            hecke::HeckeOperatorOnK x(cohC.get(), &cosets, &tau, FibreRepresentation::QuotientL);
        },
        [&] { hecke::HeckeOperatorOnK x(&cohQ, &cosets, &tau, FibreRepresentation::QuotientL); });

    return Finish("HeckeThetaCensusRingRefusalValidation", ct, t0);
}
