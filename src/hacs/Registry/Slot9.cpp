// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Slot-9 registration (closed slot): the former per-package fragments Registry/Slot9/<pkg>.cpp folded verbatim
// (S24 CL-10), each package function kept; RegisterSlot9() calls them in the old RegisterAllValidations() order.

// ===== Slot 9 WP3 ===========================================
// Slot-9 WP-3 registration fragment (hecke-extensions slot 9, WP-1).
// Owned by package WP-3 ONLY: register this package's validations here and
// nowhere else in the shared structure.  Called explicitly from
// RegisterAllValidations() (never a static initialiser).
#include "../ValidationRegistry.h"
#include "../Integral/HeckeBredonTreeRingValidation.h"   // slot-9 WP-3 (E8 at ell in {2, 3}, O7)

void RegisterSlot9WP3() {
    // (hecke-extensions slot 9, WP-3; decision 4 (A); plan B-1, M-6) E8 at
    // ell in {2, 3}: EquivariantCohomologyOnK over Z/ell^k by the
    // flat-coordinate module route (the Bredon object, [C] def:ringsemantics(2),
    // prop:bredon(2)-(3)), lengths and invariant factors against the frozen O7
    // table (the probe's model_R) on 96 cells (ell in {2, 3}, k in {2, 3, 4},
    // even n <= 20, odd n <= 9), with O7's own controls at k = 1 (the F_ell
    // pipeline) and at ell = 5 (slot 8's route), and an in-process tree model
    // over R as a consistency check.  A NEW `standard` name (M-6):
    // HeckeBredonTreeValidation is untouched.  Writes nothing.
    ValidationRegistry::register_validation("HeckeBredonTreeRingValidation",
                                            &HeckeBredonTreeRingValidation,
                                            ValidationTier::Standard);
}

// ===== Slot 9 WP4 ===========================================
// Slot-9 WP-4 registration fragment (hecke-extensions slot 9, WP-1).
// Owned by package WP-4 ONLY: register this package's validations here and
// nowhere else in the shared structure.  Called explicitly from
// RegisterAllValidations() (never a static initialiser).  Empty until WP-4 lands.
#include "../ValidationRegistry.h"

void RegisterSlot9WP4() {}

// ===== Slot 9 WP5 ===========================================
// Slot-9 WP-5 registration fragment (hecke-extensions slot 9, WP-1).
// Owned by package WP-5 ONLY: register this package's validations here and
// nowhere else in the shared structure.  Called explicitly from
// RegisterAllValidations() (never a static initialiser).  Empty until WP-5 lands.
#include "../ValidationRegistry.h"

void RegisterSlot9WP5() {}

// ===== Slot 9 WP6 ===========================================
// Slot-9 WP-6 registration fragment (hecke-extensions slot 9, WP-1).
// Owned by package WP-6 ONLY: register this package's validations here and
// nowhere else in the shared structure.  Called explicitly from
// RegisterAllValidations() (never a static initialiser).
#include "../ValidationRegistry.h"
#include "../EngineElimPassOptionsValidation.h"   // slot-9 WP-6 (engine findings 2-3)

void RegisterSlot9WP6() {
    // (hecke-extensions slot 9, WP-6; decision 7 (A)) the tape elimination
    // driver compiles under consumer_pass_options() -- the ScopedPassOptions
    // override included -- and pass_options_fingerprint() keys the EFFECTIVE
    // device policy: (A) nothing moves at the defaults, (B) the fingerprint
    // equals the pre-WP-6 one on every pair-and-policy options value and sets
    // the device bit on a `policy`-only one, (C) a batch = false override
    // reaches every per-panel program with an entrywise-equal RREF, (D) a
    // `policy = gpu` override puts GPU nodes in them ([SKIP] without a
    // device).  Sub-second.  Writes nothing.
    ValidationRegistry::register_validation("EngineElimPassOptionsValidation",
                                            &EngineElimPassOptionsValidation,
                                            ValidationTier::Fast);
}

// ===== Slot 9 WP7 ===========================================
// Slot-9 WP-7 registration fragment (hecke-extensions slot 9, WP-1).
// Owned by package WP-7 ONLY: register this package's validations here and
// nowhere else in the shared structure.  Called explicitly from
// RegisterAllValidations() (never a static initialiser).
#include "../ValidationRegistry.h"
#include "../EngineArenaFirstFitValidation.h"   // slot-9 WP-7 (lever L3)
#include "../Hecke/HeckeFamilyConcurrentBuildValidation.h"   // slot-9 WP-7 (lever L2)

void RegisterSlot9WP7() {
    // (hecke-extensions slot 9, WP-7, lever L3; decision 8 (A)) the tape
    // scheduler's O(log L) arena first-fit against an independent linear-scan
    // re-derivation of the placement rule on randomized programs: every
    // temp's arena_offset and arena_elts equal, both branches (interior gap,
    // append) exercised, arena run == owned-storage run == poisoned-arena run
    // entrywise.  ~1 s.  Writes nothing.
    ValidationRegistry::register_validation("EngineArenaFirstFitValidation",
                                            &EngineArenaFirstFitValidation,
                                            ValidationTier::Fast);
    // (lever L2) two HeckeOperatorFamily constructions of U_ell + T_p at
    // once -- build_pipelines() no longer holds tau_construction_mutex()
    // around them -- equal the serial builds entrywise, with each thread's
    // PredicateStats deltas unchanged; two independent cells and two
    // families on one shared K.  The CheckThread run is the race claim.
    // ~2 s.  Writes nothing.
    ValidationRegistry::register_validation("HeckeFamilyConcurrentBuildValidation",
                                            &HeckeFamilyConcurrentBuildValidation,
                                            ValidationTier::Fast);
}

// ===== Slot 9 WP8 ===========================================
// Slot-9 WP-8 registration fragment (hecke-extensions slot 9, WP-1).
// Owned by package WP-8 ONLY: register this package's validations here and
// nowhere else in the shared structure.  Called explicitly from
// RegisterAllValidations() (never a static initialiser).
#include "../ValidationRegistry.h"
#include "../Campaign/CampaignRingValidation.h"   // slot-9 WP-8 (C5c, hecke_ring_report/v1)

void RegisterSlot9WP8() {
    // (hecke-extensions slot 9, WP-8; C5c; decision 6 (A)) The campaign's job
    // loop under coefficient_ring = Z/ell^<k>: hecke_ring_report/v1 ([E]
    // sec:ring (g)).  The E8 points' objects == golden (a set equality), the
    // weight-12 free charpoly at 691 and at the uint64_t cell, ell in {2, 3}
    // against O7 with the three refusals, r = 2 refused at the job loop, the
    // record's member order, summary.tsv's 21 + 15 columns, index.json's key,
    // resume, jobs = 1 == jobs = 2, and the spec-validation refusals (ext pass,
    // ell^k >= 2^64).  Writes only under the system temp directory.
    ValidationRegistry::register_validation("CampaignRingValidation", &CampaignRingValidation,
                                            ValidationTier::Standard);
}

// ===== Slot 9 WP9 ===========================================
// Slot-9 WP-9 registration fragment (hecke-extensions slot 9, WP-1).
// Owned by package WP-9 ONLY: register this package's validations here and
// nowhere else in the shared structure.  Called explicitly from
// RegisterAllValidations() (never a static initialiser).
//
// WP-9's carries are PerformanceSmokeValidation.cpp, Integral/IntegralCharpolyCrt.cpp
// and EquivariantCohomologyOnK.h's get_prime() (edited in place, no new
// registration needed for those).  The three checks below are the lead's
// additions from the WP-14a review of WP-3/WP-10/WP-11
// (docs/implementation/slot9/wp14a_review.md, findings m-1, m-2, m-4):
// each is a NEW, self-contained validation so that no file owned by another
// package (HeckeReductionValidation.cpp / HeckeBredonTreeRingValidation.cpp /
// HeckeFreeModelValidation.cpp) is touched.  m-6 and m-7 are recorded for
// WP-17 in the package report instead (they need an edit inside WP-11's
// owned HeckeFreeModelValidation.cpp, which WP-9 may not touch).

#include "../ValidationRegistry.h"

#include "../EquivariantCohomologyOnK.h"
#include "../Hecke/CoefficientRing.h"
#include "../Hecke/HeckeSettings.h"
#include "../Integral/Model/IsotropyModel.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"
#include "../../ModArithmetic/DenseBuf.h"
#include "../../ModArithmetic/DenseView.h"
#include "../../ModArithmetic/Kernels.h"
#include "../../ModArithmetic/ModCtx.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>

namespace {

// ===========================================================================
// m-1 (WP-14a review of WP-3): SmallEllPolicyScopeValidation.
//
// `hecke::SmallEllPolicy::Bredon` is a public enum value any caller of
// `hecke::CoefficientRingContext` may pass; the plan's rule ("a lift that
// leaks past EquivariantCohomologyOnK") was, before this check, held only by
// a comment at Hecke/CoefficientRing.h:53-60 and by the golden-gated arms of
// the callers that exist TODAY.  This grep-based guard makes the rule
// checkable: the literal token `SmallEllPolicy::Bredon` may occur only in
// - EquivariantCohomologyOnK.cpp        (the one caller: init_coefficient_ring())
// - Hecke/CoefficientRing.h / .cpp      (the enum's declaration, its doc
//                                        comment, and the guard's own
//                                        `!= SmallEllPolicy::Bredon` comparison
//                                        inside CoefficientRingContext itself)
// - any `*Validation.cpp` / `*Validation.h` file (the plan's own exemption:
//   "no NON-VALIDATION TU other than EquivariantCohomologyOnK.cpp")
// A future caller that passes the policy anywhere else trips this the moment
// it is added, rather than waiting for a slot-9-style review to notice.
// ===========================================================================

// (slot-14 WP-2, [C3] finding 5) The original candidate list was cwd-relative
// only, so a run from a cluster case directory on /scratch never found the
// tree: the guard silently printed "SKIPPED" and still returned 0, so the
// harness printed "[PASS] SmallEllPolicyScopeValidation" for a rule that was
// never checked. Fall back to OC_SOURCE_DIR and a /proc/self/exe walk-up,
// exactly as ContractCensusValidation / MatrixRetirementValidation /
// ConsumerMigrationValidation do.
bool LooksLikeRepoRootForM1(const std::string& dir) {
    std::ifstream f((dir + "/src/hacs/Hecke/CoefficientRing.h").c_str());
    return f.good();
}

std::string FindRepoRoot() {
#ifdef OC_SOURCE_DIR
    {
        const std::string cmake_root(OC_SOURCE_DIR);
        if (!cmake_root.empty() && LooksLikeRepoRootForM1(cmake_root)) return cmake_root;
    }
#endif
    {
        char buf[4096];
        const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
        if (n > 0) {
            buf[n] = '\0';
            std::string dir(buf);
            for (int up = 0; up < 8; ++up) {
                const size_t slash = dir.find_last_of('/');
                if (slash == std::string::npos || slash == 0) break;
                dir = dir.substr(0, slash);
                if (LooksLikeRepoRootForM1(dir)) return dir;
            }
        }
    }
    const std::vector<std::string> candidates = {".", "..", "../..", "../.."};
    for (const std::string& root : candidates) {
        if (LooksLikeRepoRootForM1(root)) return root;
    }
    return std::string();
}

bool IsExemptFile(const std::string& rel) {
    // Exact allow-list: the declaration/implementation of the guard itself.
    if (rel == "src/hacs/EquivariantCohomologyOnK.cpp") return true;
    if (rel == "src/hacs/Hecke/CoefficientRing.h") return true;
    if (rel == "src/hacs/Hecke/CoefficientRing.cpp") return true;
    // This file: it IS the scope check (this function's own prose and its
    // sibling m-2 validation, which deliberately exercises the honouring
    // policy) -- a test/oracle TU in the same sense a *Validation.cpp is,
    // just not named that way because it also carries WP-9's registrations.
    if (rel == "src/hacs/Registry/Slot9.cpp") return true;
    // Every validation / test-harness TU is exempt by the plan's own wording.
    auto ends_with = [](const std::string& s, const std::string& suf) {
        return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
    };
    if (ends_with(rel, "Validation.cpp") || ends_with(rel, "Validation.h")) return true;
    return false;
}

uint32_t SmallEllPolicyScopeValidation() {
    std::cout << "SmallEllPolicyScopeValidation (slot-9 WP-9, m-1): "
                 "SmallEllPolicy::Bredon occurs in no non-validation TU other than "
                 "EquivariantCohomologyOnK.cpp" << std::endl;
    const std::string root = FindRepoRoot();
    if (root.empty()) {
        std::cout << "  [SKIP] repository root not found from the working directory, from "
                     "OC_SOURCE_DIR or from /proc/self/exe." << std::endl;
        return 0;
    }
    const std::string needle = "SmallEllPolicy::Bredon";
    const std::filesystem::path base = std::filesystem::path(root) / "src/hacs";
    uint32_t violations = 0;
    bool saw_expected_site = false;
    std::error_code ec;
    for (auto it = std::filesystem::recursive_directory_iterator(
             base, std::filesystem::directory_options::skip_permission_denied, ec);
         it != std::filesystem::recursive_directory_iterator(); it.increment(ec)) {
        if (ec) break;
        if (!it->is_regular_file(ec)) continue;
        const std::filesystem::path& p = it->path();
        const std::string ext = p.extension().string();
        if (ext != ".cpp" && ext != ".h") continue;
        std::ifstream f(p);
        if (!f.good()) continue;
        // Path relative to `root`, with forward slashes, for a stable message
        // and for IsExemptFile's exact-match allow-list.
        std::string rel = std::filesystem::relative(p, root, ec).generic_string();
        if (ec) continue;
        std::string line;
        uint32_t lineno = 0;
        bool file_has_hit = false;
        while (std::getline(f, line)) {
            ++lineno;
            if (line.find(needle) == std::string::npos) continue;
            file_has_hit = true;
            if (rel == "src/hacs/EquivariantCohomologyOnK.cpp") saw_expected_site = true;
            if (!IsExemptFile(rel)) {
                std::cout << "  [FAIL-DETAIL] " << rel << ":" << lineno
                          << ": SmallEllPolicy::Bredon outside the allow-list" << std::endl;
                ++violations;
            }
        }
        (void)file_has_hit;
    }
    if (!saw_expected_site) {
        std::cout << "  [FAIL-DETAIL] EquivariantCohomologyOnK.cpp no longer mentions "
                     "SmallEllPolicy::Bredon -- the expected site moved or was removed"
                  << std::endl;
        ++violations;
    }
    std::cout << "SmallEllPolicyScopeValidation: " << violations << " violation(s)." << std::endl;
    return violations;
}

// ===========================================================================
// m-2 (WP-14a review of WP-3): one HONOURING u64 cell of the flat-coordinate
// (Bredon) route at ell = 2, exercising `compute_cohomology_flat_module<uint64_t>`
// (instantiated since slot-9 WP-3 but, before this check, reached only by a
// default-policy REFUSAL row at (3, Z/2^33) -- never an honouring one).
//
// The oracle here is not a fresh computation: it is the K-SCALING LAW of the
// flat-coordinate route, read off the frozen O7 table of
// HeckeBredonTreeRingValidation.cpp (docs/implementation/integral_line_probe.py
// model_R, blob f50f3cd3..., frozen 2026-09-24), which already gives
// ell=2, k=4, n=4: lengths (2, 6, 0), invariant factors H^0=[2], H^1=[4,2],
// H^2=[].  This route computes H^q with COEFFICIENTS IN THE RING Z/ell^K
// directly (WP-14a review Sec. 1.1: "H^q(Hom_Z[Gamma](C_.(Xbulk), V_R)) for
// the relation list over R"), not a reduction of an integral H^q(Z) -- so an
// invariant factor equal to K itself is a FREE-RANK contribution (it reads
// back as Z/ell^K at every K, growing with K), while one strictly below K is
// GENUINE ell-power torsion and is K-INDEPENDENT once K exceeds it.  The K=4
// row's H^1 = [4, 2] is exactly one of each: "4" == K_ref is the free-rank
// artifact, "2" < K_ref is the stable torsion factor.  A much larger
// certified K must therefore replace every entry equal to the old K with the
// new K (unchanged count) and leave every strictly-smaller entry alone; this
// is checked here at k = 33 (u64, since 2^33 does not fit uint32_t) --
// predicted H^0 = [2] (no free part), H^1 = [33, 2], H^2 = [] (unchanged),
// lengths (2, 35, 0). u64 is required because 2^33 does not fit uint32_t, and
// this is the honouring cell m-2 asked for: `compute_cohomology_flat_module<uint64_t>`
// is instantiated (slot-9 WP-3) but was, before this check, reached only by a
// default-policy REFUSAL row at a composite ring -- never an honouring one.
// ===========================================================================

uint32_t HeckeBredonU64FlatRouteValidation() {
    std::cout << "HeckeBredonU64FlatRouteValidation (slot-9 WP-9, m-2): one honouring u64 "
                 "Bredon cell (ell=2, n=4, k=33), cross-checked against the frozen O7 "
                 "(ell=2, k=4, n=4) row by the route's K-scaling law" << std::endl;
    uint32_t failing = 0;
    auto check = [&](bool ok, const std::string& what) {
        if (!ok) {
            std::cout << "  [FAIL-DETAIL] " << what << std::endl;
            ++failing;
        }
    };

    const uint32_t ell = 2, n = 4, k = 33;
    const engine::ModCtx<uint64_t> ctx = engine::ModCtx<uint64_t>::prime_power(ell, k);
    hecke::ScopedCoefficientRing ring(k, hecke::SettingSource::Spec);

    SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    auto rels = hecke::GenerateSL2ZRelationsRing<uint64_t>(K, n, ctx);
    LocalSystemSimplicialComplexT<uint64_t> lssc(K, std::move(rels), n + 1, ctx.p);
    EquivariantCohomologyOnKT<uint64_t> coh(&lssc);
    coh.compute_cohomology();

    check(coh.is_composite(), "the cell did not honour coefficient_ring (is_composite() false)");
    check(coh.is_flat_route(), "the cell did not take the flat-coordinate (Bredon) route");

    const std::vector<uint32_t> lengths = coh.H_lengths();
    const std::vector<uint32_t> expect_lengths = {2, 35, 0};
    check(lengths == expect_lengths,
          "H_lengths() != {2, 35, 0} (the K=4 O7 row's free-rank part rescaled to K=33)");

    const std::vector<std::vector<uint32_t>> expect_inv = {{2}, {33, 2}, {}};
    for (uint32_t q = 0; q < 3 && q < lengths.size(); ++q) {
        const std::vector<uint32_t>& got = coh.H_invariant_factors(q);
        check(got == expect_inv[q],
              "H_invariant_factors(" + std::to_string(q) +
                  ") != the K=4 O7 row's free-rank entries rescaled to K=33");
    }

    std::cout << "HeckeBredonU64FlatRouteValidation: " << failing << " failing." << std::endl;
    return failing;
}

// ===========================================================================
// m-4 (WP-14a review of WP-10): a DIRECT delta^2 = 0 check on C_I(R), over
// Z/4 and F_5, at a small n.  `cohomology_lengths()` never checks
// `coboundary(q+1) . coboundary(q) == 0` itself -- it is implied by d^2 = 0 in
// Z[Gamma] plus rho_n being a ring homomorphism, and is caught only
// INDIRECTLY today, by every oracle comparison.  This certifies the cochain
// complex as such directly, entrywise, by dense matrix multiplication over
// the same ModCtx the shipped accessors use.
// ===========================================================================

template <class Elt>
bool DeltaSquaredZero(const integral_model::IsotropyModel& M, const engine::ModCtx<Elt>& ctx,
                     uint32_t n, uint32_t q) {
    const engine::DenseBuf<Elt> dq = M.coboundary<Elt>(ctx, n, q);
    const engine::DenseBuf<Elt> dq1 = M.coboundary<Elt>(ctx, n, q + 1);
    if (dq1.cols != dq.rows) return false; // shape mismatch is itself a defect
    engine::DenseBuf<Elt> prod(dq1.rows, dq.cols);
    engine::kernels::gemm<Elt>(ctx, engine::cview(dq1), engine::cview(dq), engine::view(prod),
                               engine::kernels::GemmOp::Set);
    for (uint32_t i = 0; i < prod.rows; ++i)
        for (uint32_t j = 0; j < prod.cols; ++j)
            if (prod.at(i, j) != 0) return false;
    return true;
}

uint32_t IsotropyModelDeltaSquaredValidation() {
    std::cout << "IsotropyModelDeltaSquaredValidation (slot-9 WP-9, m-4): "
                 "coboundary(q+1) . coboundary(q) == 0 on C_I(R), over Z/4 and F_5"
              << std::endl;
    uint32_t failing = 0, checks = 0;
    integral_model::IsotropyModel M(/*q_max=*/5);
    const engine::ModCtx<uint32_t> z4 = engine::ModCtx<uint32_t>::prime_power(2u, 2u);
    const engine::ModCtx<uint32_t> f5(5u);
    const std::vector<uint32_t> ns = {2, 4};
    for (uint32_t n : ns) {
        // coboundary(q) is only defined for q <= q_max() (IsotropyModel.cpp's
        // own ENGINE_CHECK); q_top() = q_max() + 1 is the top CELL degree, not
        // a valid coboundary source, so the pair (q, q+1) needs q+1 <= q_max().
        for (uint32_t q = 0; q + 1 <= M.q_max(); ++q) {
            ++checks;
            if (!DeltaSquaredZero<uint32_t>(M, z4, n, q)) {
                std::cout << "  [FAIL-DETAIL] Z/4, n=" << n << ", q=" << q
                          << ": coboundary(q+1).coboundary(q) != 0" << std::endl;
                ++failing;
            }
            ++checks;
            if (!DeltaSquaredZero<uint32_t>(M, f5, n, q)) {
                std::cout << "  [FAIL-DETAIL] F_5, n=" << n << ", q=" << q
                          << ": coboundary(q+1).coboundary(q) != 0" << std::endl;
                ++failing;
            }
        }
    }
    std::cout << "IsotropyModelDeltaSquaredValidation: " << checks << " checks, " << failing
              << " failing." << std::endl;
    return failing;
}

} // namespace

void RegisterSlot9WP9() {
    ValidationRegistry::register_validation("SmallEllPolicyScopeValidation",
                                            &SmallEllPolicyScopeValidation,
                                            ValidationTier::Fast);
    ValidationRegistry::register_validation("HeckeBredonU64FlatRouteValidation",
                                            &HeckeBredonU64FlatRouteValidation,
                                            ValidationTier::Fast);
    ValidationRegistry::register_validation("IsotropyModelDeltaSquaredValidation",
                                            &IsotropyModelDeltaSquaredValidation,
                                            ValidationTier::Fast);
}

// ===== Slot 9 WP10 ===========================================
// Slot-9 WP-10 registration fragment (hecke-extensions slot 9, WP-1).
// Owned by package WP-10 ONLY: register this package's validations here and
// nowhere else in the shared structure.  Called explicitly from
// RegisterAllValidations() (never a static initialiser).
#include "../ValidationRegistry.h"
#include "../Integral/HeckeIsotropyModelValidation.h"

void RegisterSlot9WP10() {
    // (hecke-extensions slot 9, WP-10 / Line 2, L2-1a; [C, def:isomodel],
    // [C, prop:isomodel]) the isotropy-resolution model R, R_d, eps on Xbulk,
    // degree-generic (built to q_max + 1 = 6), C_I over any ModCtx: ranks,
    // exact certificates over Z[Gamma], the ell >= 5 comparison with the
    // shipped pipeline, Appendix A of the integral-line brief through
    // universal coefficients at ell in {2, 3} (fields and Z/ell^k).
    // `standard`. Writes nothing.
    ValidationRegistry::register_validation("HeckeIsotropyModelValidation",
                                            &HeckeIsotropyModelValidation,
                                            ValidationTier::Standard);
}

// ===== Slot 9 WP11 ===========================================
// Slot-9 WP-11 registration fragment (hecke-extensions slot 9, WP-1).
// Owned by package WP-11 ONLY: register this package's validations here and
// nowhere else in the shared structure.  Called explicitly from
// RegisterAllValidations() (never a static initialiser).
#include "../ValidationRegistry.h"
#include "../Integral/HeckeFreeModelValidation.h"

void RegisterSlot9WP11() {
    // (hecke-extensions slot 9, WP-11 / Line 2, M0; [C, prop:certifiede],
    // [C, prop:badprimes], [C, prop:isomodel]) the integral groups
    // H^q(Gamma; Sym^n Z^2), n <= 20, q <= 5, assembled from a good-prime
    // rank, C_I (x) Z/ell^K at 2 and 3 (K certified per differential) and
    // C_7 (x) Z/ell^K at the Dickson primes, equal to Appendix A of the
    // integral-line brief exactly; O2, X3 (the cone of eps^*), O9, O8; [X]'s
    // E9 tests (1), (2) and the negative control (6), re-hosted ([IB] OD-8
    // (a)); M0 items 3 and 7 re-run.  Tier by R-M0.  Writes nothing.
    ValidationRegistry::register_validation("HeckeFreeModelValidation",
                                            &HeckeFreeModelValidation,
                                            ValidationTier::Standard);
}

// ===== Slot 9 WP12 ===========================================
// Slot-9 WP-12 registration fragment (hecke-extensions slot 9, WP-1).
// Owned by package WP-12 ONLY: register this package's validations here and
// nowhere else in the shared structure.  Called explicitly from
// RegisterAllValidations() (never a static initialiser).  Empty until WP-12 lands.
#include "../ValidationRegistry.h"

void RegisterSlot9WP12() {}

// ===== Slot 9 WP13 ===========================================
// Slot-9 WP-13 registration fragment (hecke-extensions slot 9, WP-1).
// Owned by package WP-13 ONLY: register this package's validations here and
// nowhere else in the shared structure.  Called explicitly from
// RegisterAllValidations() (never a static initialiser).
#include "../ValidationRegistry.h"

// (slot-9 WP-13, console C5b) the certificate-free local-algebra route and the
// binding-support builders behind the extension-layer console bindings.
#include "../HeckeExt/HeckeExtAlgebraInvariantsValidation.h"

void RegisterSlot9WP13() {
    ValidationRegistry::register_validation("HeckeExtAlgebraInvariantsValidation",
                                            &HeckeExtAlgebraInvariantsValidation,
                                            ValidationTier::Standard);
}

void RegisterSlot9() {
    RegisterSlot9WP3();
    RegisterSlot9WP4();
    RegisterSlot9WP5();
    RegisterSlot9WP6();
    RegisterSlot9WP7();
    RegisterSlot9WP8();
    RegisterSlot9WP9();
    RegisterSlot9WP10();
    RegisterSlot9WP11();
    RegisterSlot9WP12();
    RegisterSlot9WP13();
}
