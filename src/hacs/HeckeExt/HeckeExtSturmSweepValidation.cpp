// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions phase E1b (slot 3, WP-2) -- HeckeExtSturmSweepValidation.
//
// See HeckeExtSturmSweepValidation.h.  assert() is not used: every check is
// counted and reported, and the return value is the number of failures, so a
// Release build reports rather than aborts (the HeckeExtAlgebraValidation
// idiom).

#include "HeckeExtSturmSweepValidation.h"

#include "HeckeGenerationCertificates.h"
#include "HeckeLocalAlgebra.h"
#include "HeckeSturmBound.h"
#include "HeckeSturmSweep.h"
#include "HeckeUEllBudget.h"

#include "../Hecke/DicksonCount.h"
#include "../Hecke/HeckeModuleDecomposition.h"
#include "../Hecke/HeckeOperatorFamily.h"
#include "../Hecke/HeckeSettings.h"
#include "../Hecke/SL2ZTessellationTables.h"
#include "../Hecke/SubspaceOps.h"
#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"

#include "../../ModArithmetic/Matrix.h"
#include "../Util/ValidationCheck.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using hecke_ext::CommutantOutcome;
using hecke_ext::CommutantRoute;
using hecke_ext::ExtMatrix;
using hecke_ext::GenerationKind;
using hecke_ext::LocalAlgebra;
using hecke_ext::SturmBound;
using hecke_ext::SturmCertificate;
using hecke_ext::SturmFinal;
using hecke_ext::SturmSweep;
using hecke_ext::SturmSweepOptions;
using hecke_ext::SturmSweepOutcome;
using hecke_ext::SturmUnavailableMode;

oc_util::Checker Check("  [FAIL] ");

// The enums below carry their own ToString / to_string; everything else goes
// through operator<<.  A non-template overload wins over the template.
template <class T>
std::string Show(const T& x) {
    std::ostringstream o;
    o << x;
    return o.str();
}
std::string Show(hecke_ext::UEllReason r) { return hecke_ext::to_string(r); }
std::string Show(CommutantOutcome o) { return hecke_ext::ToString(o); }
std::string Show(SturmSweepOutcome o) { return hecke_ext::ToString(o); }
std::string Show(SturmFinal v) { return hecke_ext::ToString(v); }
std::string Show(GenerationKind k) { return hecke_ext::ToString(k); }
std::string Show(hecke_ext::SturmOutcome o) { return hecke_ext::ToString(o); }

template <class T>
void CheckEq(const T& got, const T& want, const std::string& what) {
    ++Check.checks;
    if (!(got == want)) {
        ++Check.fails;
        std::cout << "  [FAIL] " << what << ": got " << Show(got) << ", want " << Show(want)
                  << std::endl;
    }
}

std::string PrimeList(const std::vector<uint64_t>& v) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) oss << ",";
        oss << v[i];
    }
    oss << "]";
    return oss.str();
}

// ---------------------------------------------------------------------------
// The pipeline stack -- the HeckeIdentityValidation pattern.
// ---------------------------------------------------------------------------

struct Stack {
    SimplicialComplex K;
    hecke::SL2ZTessellationTables model;
    LocalSystemSimplicialComplex lssc;
    EquivariantCohomologyOnK coh;

    Stack(uint32_t n_sym, uint32_t ell)
        : K(CreateSL2ZFundamentalDomainTriangulation()),
          model(K),
          lssc(K, BuildRelations(K, n_sym, ell), n_sym + 1, ell),
          coh(&lssc) {
        coh.compute_cohomology();
    }

    static std::vector<LocalSystemSimplicialComplex::SimplexRelation>
    BuildRelations(const SimplicialComplex& K, uint32_t n_sym, uint32_t ell) {
        std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels;
        for (const auto& datum : GenerateSL2ZRelationsWithGroupData(K, n_sym, ell)) {
            rels.push_back(datum.rel);
        }
        return rels;
    }
};

// One (n, ell) point with its family, ready for Stage 2 onwards.
struct Point {
    uint32_t n = 0, ell = 0, degree = 1;
    Stack stack;
    hecke::HeckeOperatorFamily family;
    std::vector<uint64_t> primes;     // the operator set the decomposition uses
    bool u_ell_included = false;

    Point(uint32_t n_, uint32_t ell_, const std::vector<uint64_t>& S, bool u_ell_in,
          uint32_t deg = 1)
        : n(n_), ell(ell_), degree(deg), stack(n_, ell_),
          family(&stack.coh, stack.K, stack.model, ell_, n_),
          primes(S), u_ell_included(u_ell_in) {}

    uint32_t dim() const {
        const auto& d = stack.coh.get_H_eq_dimensions();
        return degree < d.size() ? d[degree] : 0u;
    }

    std::vector<std::pair<uint64_t, ExtMatrix>> ops() {
        std::vector<std::pair<uint64_t, ExtMatrix>> out;
        for (uint64_t q : primes) {
            const std::vector<Matrix>& A = family.T_p(q);
            out.push_back({q, degree < A.size() ? A[degree] : ExtMatrix()});
        }
        return out;
    }
};

// The membership test re-derived independently of the sweep: the algebra
// generated by `gens` (via hecke::algebra_closure) and one RREF reduction.
bool InAlgebra(const std::vector<ExtMatrix>& gens, const ExtMatrix& x, uint32_t D,
               uint32_t ell) {
    uint32_t dim_A = 0;
    const Matrix coords = hecke::algebra_closure(gens, D, ell, dim_A);
    hecke::RrefRowSpan span(D * D, ell);
    for (uint32_t i = 0; i < dim_A; ++i) span.insert(coords.data[i]);
    std::vector<uint32_t> v;
    v.reserve(static_cast<size_t>(D) * D);
    for (uint32_t i = 0; i < D; ++i) {
        for (uint32_t j = 0; j < D; ++j) v.push_back(x.data[i][j]);
    }
    return span.contains(v);
}

// ---------------------------------------------------------------------------
// Arm 1 -- Stage 1: the operator set and the `u_ell` budget check.
// ---------------------------------------------------------------------------

void RunStageOneArm() {
    std::cout << "-- arm 1: Stage 1, the base operator set and the u_ell verdict\n";
    using hecke_ext::ComputeStageOneOperators;
    using hecke_ext::StageOneOperators;
    using hecke::UEllMode;
    using hecke::UEllSetting;

    // (a) `auto` at (10, 691): U_691 is within the frozen budget, so 691 joins
    //     S_0 = {2, 3, 5, 7}.
    {
        const StageOneOperators s = ComputeStageOneOperators(10, 691, UEllSetting());
        Check(s.includes_u_ell(), "auto: U_691 not included at n = 10");
        CheckEq(s.u_ell.reason, hecke_ext::UEllReason::WithinBudget,
                "auto at (10, 691): wrong reason");
        const std::vector<uint64_t> want = {2, 3, 5, 7, 691};
        Check(s.primes == want, "auto at (10, 691): S_0 = " + PrimeList(s.primes));
        Check(s.u_ell.predicted_mb <= 2048.0, "auto at (10, 691): prediction over budget");
        Check(std::string(s.u_ell.model_version) ==
                  std::string(hecke_ext::kUEllRssModelVersion),
              "the frozen model version moved");
    }

    // (b) THE PINNED OVER-BUDGET POINT (slot-3 exit item 3, WP-1's pin):
    //     (16, 43867) is refused, and 43867 is therefore not in S_0.
    {
        const StageOneOperators s = ComputeStageOneOperators(16, 43867, UEllSetting());
        Check(!s.includes_u_ell(), "auto at (16, 43867): U_ell was NOT refused");
        CheckEq(s.u_ell.reason, hecke_ext::UEllReason::OverBudget,
                "auto at (16, 43867): wrong reason");
        const std::vector<uint64_t> want = {2, 3, 5, 7};
        Check(s.primes == want, "auto at (16, 43867): S_0 = " + PrimeList(s.primes));
        Check(s.u_ell.predicted_mb > 2048.0,
              "auto at (16, 43867): the prediction is not over budget");
    }

    // (c) `off` removes ell even when ell is one of the four good primes: at
    //     ell <= 7 the good prime ell IS U_ell.
    {
        UEllSetting off;
        off.mode = UEllMode::Off;
        const StageOneOperators a = ComputeStageOneOperators(22, 5, UEllSetting());
        const StageOneOperators b = ComputeStageOneOperators(22, 5, off);
        const std::vector<uint64_t> want_auto = {2, 3, 5, 7};
        const std::vector<uint64_t> want_off = {2, 3, 7};
        Check(a.primes == want_auto, "auto at (22, 5): S_0 = " + PrimeList(a.primes));
        Check(b.primes == want_off, "off at (22, 5): S_0 = " + PrimeList(b.primes));
        CheckEq(b.u_ell.reason, hecke_ext::UEllReason::Off, "off at (22, 5): wrong reason");
        Check(b.u_ell.predicted_mb == 0.0,
              "off formed a prediction (the record must say the user said no)");
    }

    // (d) The fingerprint: 0 at the defaults, non-zero under `off`.
    {
        hecke::MathSettings d;
        Check(d.fingerprint() == 0, "the all-defaults fingerprint is not 0");
        hecke::MathSettings s;
        Check(s.set("u_ell", "off", hecke::SettingSource::Spec), "set(u_ell, off) failed");
        Check(s.fingerprint() != 0, "u_ell = off fingerprints to 0");
    }

    // (e) The two new `sturm.*` keys: registered, parsed, round-tripped, and
    //     0-fingerprinted at their defaults.
    {
        Check(hecke::MathSettings::known_key("sturm"), "`sturm` is not a registered key");
        Check(hecke::MathSettings::known_key("sturm_unavailable"),
              "`sturm_unavailable` is not a registered key");
        const std::vector<std::string>& keys = hecke::MathSettings::keys();
        // Keys are APPENDED, never inserted (slot-4 WP-1 appended
        // `commutant_budget` and `final_mode` after `sturm_unavailable`), so
        // the PREFIX is what the fingerprint depends on and what this arm
        // pins -- the HeckeUEllBudgetValidation idiom.  A `== 4` pin would
        // fail on every later key without anything having moved.
        Check(keys.size() >= 4 && keys[0] == "locality" && keys[1] == "u_ell" &&
                  keys[2] == "sturm" && keys[3] == "sturm_unavailable",
              "the registry order does not begin [locality, u_ell, sturm, sturm_unavailable]");

        hecke::MathSettings s;
        Check(s.set("sturm", "cap=3", hecke::SettingSource::Console), "set(sturm, cap=3) failed");
        Check(s.sturm.mode == hecke::SturmCapMode::Cap && s.sturm.cap == 3,
              "set(sturm, cap=3) did not record the value");
        Check(s.canonical_text() == "sturm = cap=3\n",
              "wrong canonical text for sturm: \"" + s.canonical_text() + "\"");
        Check(s.fingerprint() != 0, "a non-default sturm fingerprints to 0");

        hecke::MathSettings t;
        Check(t.set("sturm_unavailable", "heuristic", hecke::SettingSource::Spec),
              "set(sturm_unavailable, heuristic) failed");
        Check(t.canonical_text() == "sturm_unavailable = heuristic\n",
              "wrong canonical text for sturm_unavailable: \"" + t.canonical_text() + "\"");
        Check(t.fingerprint() != s.fingerprint(), "two distinct settings share a fingerprint");

        hecke::MathSettings u = s;
        Check(u.set("sturm_unavailable", "heuristic", hecke::SettingSource::Spec), "set failed");
        Check(u.canonical_text() == "sturm = cap=3\nsturm_unavailable = heuristic\n",
              "the two-key canonical text is not in registry order: \"" + u.canonical_text() +
                  "\"");
        Check(hecke::MathSettings::defaults().is_default(), "the defaults are not default");
        Check(hecke::MathSettings::defaults().fingerprint() == 0,
              "the defaults fingerprint is not 0 after registering two keys");

        // Rejections.
        const char* bad_sturm[] = {"", "cap", "cap=", "cap=0", "cap=1", "cap=x", "AUTO", "on"};
        for (const char* b : bad_sturm) {
            bool threw = false;
            try {
                hecke::ParseSturm(b);
            } catch (const std::invalid_argument&) {
                threw = true;
            }
            Check(threw, std::string("ParseSturm accepted \"") + b + "\"");
        }
        const char* bad_mode[] = {"", "final", "nonFinal", "heuristics"};
        for (const char* b : bad_mode) {
            bool threw = false;
            try {
                hecke::ParseSturmUnavailable(b);
            } catch (const std::invalid_argument&) {
                threw = true;
            }
            Check(threw, std::string("ParseSturmUnavailable accepted \"") + b + "\"");
        }
        // `sturm = off` maps onto the option, `auto` onto 12*B(k).
        hecke::MathSettings o;
        Check(o.set("sturm", "off", hecke::SettingSource::Console), "set(sturm, off) failed");
        const SturmSweepOptions oo = SturmSweepOptions::from_settings(o, 24);
        Check(oo.sturm_off, "sturm = off did not reach the options");
        const SturmSweepOptions ao = SturmSweepOptions::from_settings(hecke::MathSettings(), 24);
        Check(!ao.sturm_off && ao.cap == SturmBound::default_cap(24),
              "sturm = auto did not resolve to 12*B(k)");
        Check(ao.mode == SturmUnavailableMode::NonFinal,
              "the default sturm.unavailable mode is not nonfinal");
    }
}

// ---------------------------------------------------------------------------
// The per-block driver shared by the real-block arms.
// ---------------------------------------------------------------------------

struct BlockRun {
    uint32_t D = 0;
    bool local = false;
    ExtMatrix basis;                 // the block's basis columns (d x D)
    LocalAlgebra L;
    SturmCertificate cert;
    SturmSweepOutcome outcome = SturmSweepOutcome::InProgress;
};

// Decomposes `p`'s A^{(degree)} over its operator set, runs Stage 3 + (c) on
// every block, and returns one BlockRun per block.
std::vector<BlockRun> RunBlocks(Point& p, const SturmSweepOptions& opts, bool proof_range) {
    std::vector<std::pair<uint64_t, ExtMatrix>> ops = p.ops();
    hecke::HeckeModuleDecomposition decomp(ops, p.ell);
    decomp.compute();

    std::vector<BlockRun> out;
    for (const hecke::PrimaryBlock& b : decomp.blocks()) {
        BlockRun r;
        r.D = b.dim;
        r.basis = b.basis_columns;
        r.L = hecke_ext::ComputeLocalAlgebra(b, ops, p.ell);
        r.local = r.L.ok;
        if (!r.L.ok) {
            out.push_back(std::move(r));
            continue;
        }
        SturmSweepOptions o = opts;
        o.proof_range = proof_range;
        hecke_ext::BlockRestrictionSource src(
            b.basis_columns,
            hecke_ext::FamilyOperatorSupplier(p.family, p.degree, p.ell, p.u_ell_included));
        SturmSweep sweep(r.L, p.n, src, o);
        r.outcome = sweep.run();
        r.cert = sweep.certificate();
        out.push_back(std::move(r));
    }
    return out;
}

// ---------------------------------------------------------------------------
// Arm 2 -- (10, 691): the Ramanujan cell, [X] Sec. 6.2 E1 acceptance (b).
// ---------------------------------------------------------------------------

void RunRamanujanArm() {
    std::cout << "-- arm 2: (n, ell) = (10, 691) -- F_691[eps]/eps^2, (a) fails, (b) excluded, "
                 "(c) sturm(B' = 2)\n";
    const hecke_ext::StageOneOperators s1 =
        hecke_ext::ComputeStageOneOperators(10, 691, hecke::UEllSetting());
    Point p(10, 691, s1.primes, s1.includes_u_ell());
    CheckEq(p.dim(), 3u, "(10, 691): dim H^1");
    // No Dickson classes here, so no maximal ideal is Dickson-type: the whole
    // of X_4 is [E] cl:sturmmodl's proof range.
    CheckEq<uint32_t>(hecke::DicksonInvariantCount(10, 691), 0u,
                      "(10, 691): dim H^0 is not 0");

    const std::vector<BlockRun> blocks =
        RunBlocks(p, SturmSweepOptions::from_settings(hecke::MathSettings(), 12), true);
    CheckEq(blocks.size(), size_t(1), "(10, 691): number of blocks over S_0");
    if (blocks.size() != 1) return;
    const BlockRun& r = blocks[0];
    Check(r.local, "(10, 691): the block did not certify local");
    if (!r.local) return;

    // The algebra: A_S = F_691[eps]/eps^2 on a 3-dimensional V.
    CheckEq(r.D, 3u, "(10, 691): D");
    CheckEq(r.L.f, 1u, "(10, 691): residue degree f");
    CheckEq(r.L.dim_A, 2u, "(10, 691): dim A_S (F_691[eps]/eps^2 has dimension 2)");
    CheckEq(r.L.nu, 2u, "(10, 691): nilpotency index nu");
    CheckEq(r.L.dim_socle, 1u, "(10, 691): dim soc A_S");
    Check(r.L.gorenstein, "(10, 691): A_S is not Gorenstein");
    CheckEq(r.L.embdim, 1u, "(10, 691): embedding dimension");

    // (a) fails: V is not cyclic (dim V/m_S V = 2 > f = 1).
    Check(!r.L.cyclicity.cyclic, "(10, 691): (a) certified, but it must fail");
    CheckEq(r.L.min_generators, 2u, "(10, 691): t = dim_F(V / m_S V)");
    CheckEq(r.L.cyclicity.verify(), 0u, "(10, 691): certificate (a) does not verify");

    // (b) is excluded by the socle bound, with the naive dim C = 5 as the
    // INDEPENDENT oracle (D = 3 <= 24).
    CommutantRoute route(r.L);
    const CommutantOutcome b = route.run();
    CheckEq(b, CommutantOutcome::ExcludedBySocleBound, "(10, 691): (b) outcome");
    CheckEq(route.socle_bound(), 5u, "(10, 691): the socle bound");
    {
        const uint32_t naive = hecke_ext::NaiveCommutantDim(r.L.t, r.D, r.L.ell);
        CheckEq(naive, 5u, "(10, 691): the naive dim C");
        Check(naive >= route.socle_bound(), "(10, 691): the socle bound exceeds the naive dim C");
        Check(naive > r.L.dim_A, "(10, 691): dim C == dim A_S, so (b) would have passed");
    }

    // (c): B' = 2, generators_certified = [2], final = true.
    CheckEq(r.outcome, SturmSweepOutcome::Passed, "(10, 691): the sweep did not pass");
    Check(r.cert.has_B_prime, "(10, 691): no B'");
    CheckEq(r.cert.B_prime, 2u, "(10, 691): B'(12, 691)");
    CheckEq(r.cert.dim_M_k, 2u, "(10, 691): dim M_12");
    const std::vector<uint64_t> want = {2};
    Check(r.cert.sturm_primes == want, "(10, 691): swept " + PrimeList(r.cert.sturm_primes));
    Check(r.cert.generators_certified == want,
          "(10, 691): generators_certified = " + PrimeList(r.cert.generators_certified));
    CheckEq(r.cert.verify(), 0u, "(10, 691): SturmCertificate::verify()");
    const hecke_ext::SturmFinality f = hecke_ext::sweep_finality(r.cert);
    CheckEq(f.value, SturmFinal::True, "(10, 691): the sweep's final value");
    Check(f.reason.empty(), "(10, 691): a reason on a final = true verdict");

    const hecke_ext::GenerationVerdict v =
        hecke_ext::generation_verdict_with_sweep(r.L.cyclicity.cyclic, b, r.cert);
    CheckEq(v.kind, GenerationKind::Sturm, "(10, 691): the generation verdict's kind");
    CheckEq(v.final_value, SturmFinal::True, "(10, 691): the generation verdict's final");

    // The slot-2 deferred arm 1: membership of EVERY A^{(1)}(T_q), q <= 11, in
    // A_S -- re-derived here with an own closure, not read from the sweep.
    {
        const uint64_t qs[] = {2, 3, 5, 7, 11};
        for (uint64_t q : qs) {
            const std::vector<Matrix>& A = p.family.T_p(q);
            const ExtMatrix t = hecke::SolveColumnsExact(r.basis, A[p.degree] * r.basis,
                                                         "sturm-sweep-validation(restrict)");
            Check(InAlgebra(r.L.t, t, r.D, r.L.ell),
                  "(10, 691): T_" + std::to_string(q) + "|_V is not in A_S");
        }
    }

    // The slot-2 deferred arm 2: the JOINT X_1 (+) X_4 decomposition.  At
    // (10, 691) dim H^0 = 0, so X_1 is EMPTY and the joint decomposition is
    // X_4's own -- asserted as such, which is the honest content of the arm
    // here; RunJointGroupArm() below runs the substantive version at a
    // Dickson-exceptional cell where dim H^0 > 0.
    {
        const auto& dims = p.stack.coh.get_H_eq_dimensions();
        CheckEq(dims.size() > 0 ? dims[0] : 0u, 0u, "(10, 691): dim H^0 is not 0");
    }
}

// ---------------------------------------------------------------------------
// Arm 3 -- the joint X_1 (+) X_4 decomposition, at a cell where X_1 is not
// empty.  Deferred from slot-2 WP-5; see the note in arm 2.
// ---------------------------------------------------------------------------

void RunJointGroupArm() {
    std::cout << "-- arm 3: the joint X_1 (+) X_4 decomposition against the per-group ones\n";
    // (12, 5) is Dickson-exceptional, so dim H^0 > 0 and X_1 is non-empty.
    const uint32_t n = 12, ell = 5;
    Check(hecke::DicksonInvariantCount(n, ell) > 0,
          "(12, 5) is not Dickson-exceptional, so X_1 would be empty");

    const hecke_ext::StageOneOperators s1 =
        hecke_ext::ComputeStageOneOperators(n, ell, hecke::UEllSetting());
    Stack stack(n, ell);
    hecke::HeckeOperatorFamily family(&stack.coh, stack.K, stack.model, ell, n);
    const auto& dims = stack.coh.get_H_eq_dimensions();
    const uint32_t d0 = dims.size() > 0 ? dims[0] : 0u;
    const uint32_t d1 = dims.size() > 1 ? dims[1] : 0u;
    Check(d0 > 0 && d1 > 0, "(12, 5): one of the two groups is empty");
    if (d0 == 0 || d1 == 0) return;

    // Per group.
    std::vector<uint32_t> per_group_dims;
    std::vector<LocalAlgebra> per_group_algebras;
    for (uint32_t q = 0; q <= 1; ++q) {
        std::vector<std::pair<uint64_t, ExtMatrix>> ops;
        for (uint64_t pr : s1.primes) ops.push_back({pr, family.T_p(pr)[q]});
        hecke::HeckeModuleDecomposition d(ops, ell);
        d.compute();
        CheckEq(d.certify(), 0u, "(12, 5): the per-group decomposition does not certify");
        for (const hecke::PrimaryBlock& b : d.blocks()) {
            per_group_dims.push_back(b.dim);
            LocalAlgebra L = hecke_ext::ComputeLocalAlgebra(b, ops, ell);
            Check(L.ok, "(12, 5): a per-group block did not certify local");
            per_group_algebras.push_back(std::move(L));
        }
    }

    // Jointly, on X_1 (+) X_4 = H^0 (+) H^1 as one module.
    std::vector<std::pair<uint64_t, ExtMatrix>> jops;
    for (uint64_t pr : s1.primes) {
        const std::vector<Matrix>& A = family.T_p(pr);
        ExtMatrix J(d0 + d1, d0 + d1, ell);
        for (uint32_t i = 0; i < d0; ++i) {
            for (uint32_t j = 0; j < d0; ++j) J.data[i][j] = A[0].data[i][j];
        }
        for (uint32_t i = 0; i < d1; ++i) {
            for (uint32_t j = 0; j < d1; ++j) J.data[d0 + i][d0 + j] = A[1].data[i][j];
        }
        jops.push_back({pr, J});
    }
    hecke::HeckeModuleDecomposition joint(jops, ell);
    joint.compute();
    CheckEq(joint.certify(), 0u, "(12, 5): the joint decomposition does not certify");

    std::vector<uint32_t> joint_dims;
    for (const hecke::PrimaryBlock& b : joint.blocks()) joint_dims.push_back(b.dim);

    // The joint decomposition refines the two per-group ones and has the same
    // total dimension; where two groups share a maximal ideal the joint block
    // is their sum, so the MULTISET of joint dims is the per-group multiset
    // with the shared ideals merged.  What is asserted here is the invariant
    // that holds either way: total dimension, and that every per-group block
    // sits inside exactly one joint block (matched by SameMaximalIdeal).
    uint32_t total_joint = 0, total_group = 0;
    for (uint32_t d : joint_dims) total_joint += d;
    for (uint32_t d : per_group_dims) total_group += d;
    CheckEq(total_joint, d0 + d1, "(12, 5): the joint block dimensions do not sum to dim H^0+H^1");
    CheckEq(total_group, d0 + d1, "(12, 5): the per-group dimensions do not sum");
    Check(joint_dims.size() <= per_group_dims.size(),
          "(12, 5): the joint decomposition has MORE blocks than the per-group ones");

    // Every per-group maximal ideal is EXACTLY ONE joint block's, by Stage 2's
    // own cross-block evaluation test ([XA M5], hecke_ext::SameMaximalIdeal) --
    // which is what "the joint decomposition equals the character-matched
    // per-group one" means.
    for (size_t g = 0; g < per_group_algebras.size(); ++g) {
        const LocalAlgebra& B = per_group_algebras[g];
        if (!B.ok) continue;
        uint32_t matches = 0;
        for (const hecke::PrimaryBlock& jb : joint.blocks()) {
            std::vector<ExtMatrix> t_other;
            for (uint64_t pr : B.primes) {
                ExtMatrix J;
                for (const auto& kv : jops) {
                    if (kv.first == pr) { J = kv.second; break; }
                }
                t_other.push_back(hecke::SolveColumnsExact(
                    jb.basis_columns, J * jb.basis_columns, "joint-arm(restrict)"));
            }
            if (hecke_ext::SameMaximalIdeal(B, t_other).same_ideal) ++matches;
        }
        CheckEq(matches, 1u,
                "(12, 5): per-group block " + std::to_string(g) +
                    " matches " + std::to_string(matches) + " joint blocks, not exactly one");
    }

    std::cout << "  JOINT_DECOMP (12,5) per_group=" << per_group_dims.size()
              << " joint=" << joint_dims.size() << " dims=[";
    for (size_t i = 0; i < joint_dims.size(); ++i) {
        if (i) std::cout << ",";
        std::cout << joint_dims[i];
    }
    std::cout << "]" << std::endl;
}

// ---------------------------------------------------------------------------
// Arm 4 -- the B(k) counterexamples: at (n, ell) = (22, 5) and (26, 7) the
// naive {2, 3} sweep is REJECTED and the B' sweep is accepted
// ([E] cl:sturm(b) / cl:bprime / cl:sturmmodl, [X] Sec. 6.2 E1 acceptance (c),
// slot-3 exit item 4).
// ---------------------------------------------------------------------------

void RunNaiveBoundArm() {
    std::cout << "-- arm 4: (22, 5) and (26, 7) -- the naive B(k) = 3 sweep rejected, "
                 "the B' sweep accepted\n";
    struct Case { uint32_t n, ell, k, B_naive, B_prime; };
    const std::vector<Case> cases = {
        {22, 5, 24, 3, 5},
        {26, 7, 28, 3, 7},
    };

    for (const Case& c : cases) {
        const std::string tag =
            "(" + std::to_string(c.n) + ", " + std::to_string(c.ell) + ")";

        // The two bounds, re-derived by the pipeline-free unit.
        CheckEq(SturmBound::weight_bound(c.k), c.B_naive, tag + ": B(k)");
        {
            SturmBound b(c.k, c.ell);
            b.run();
            Check(b.has_B_prime(), tag + ": no B' below the default cap");
            CheckEq(b.B_prime(), c.B_prime, tag + ": B'(k, ell)");
        }
        // No Dickson classes here, so every maximal ideal of X_4 is on the
        // proof range of [E] cl:sturmmodl -- which is what makes the naive
        // sweep's acceptance a FALSE certificate rather than a heuristic.
        CheckEq<uint32_t>(hecke::DicksonInvariantCount(c.n, c.ell), 0u,
                          tag + ": dim H^0 is not 0");

        // (i) The NAIVE arm: the decomposition and A_S built from the primes
        //     <= B(k) = 3 only, then the B' sweep run on it.  At least one
        //     block must FAIL, with the failing prime above B(k).
        const hecke_ext::StageOneOperators s1 =
            hecke_ext::ComputeStageOneOperators(c.n, c.ell, hecke::UEllSetting());
        {
            const std::vector<uint64_t> naive = hecke_ext::PrimesUpTo(c.B_naive);
            Point p(c.n, c.ell, naive, s1.includes_u_ell());
            const std::vector<BlockRun> blocks = RunBlocks(
                p, SturmSweepOptions::from_settings(hecke::MathSettings(), c.k), true);
            uint32_t failed = 0, passed = 0;
            uint64_t witness = 0;
            for (const BlockRun& r : blocks) {
                Check(r.local, tag + " naive: a block did not certify local");
                if (!r.local) continue;
                CheckEq(r.cert.verify(), 0u, tag + " naive: SturmCertificate::verify()");
                Check(r.cert.has_B_prime && r.cert.B_prime == c.B_prime,
                      tag + " naive: the sweep used the wrong B'");
                if (r.outcome == SturmSweepOutcome::Failed) {
                    ++failed;
                    if (witness == 0) witness = r.cert.failing_prime;
                    // The failure is re-derived independently of the sweep.
                    const size_t idx = r.cert.sturm_primes.size() - 1;
                    Check(!InAlgebra(r.L.t, r.cert.t_q[idx], r.D, r.L.ell),
                          tag + " naive: the failing T_q IS in A_S after all");
                    const hecke_ext::SturmFinality f = hecke_ext::sweep_finality(r.cert);
                    CheckEq(f.value, SturmFinal::False, tag + " naive: final on a failed sweep");
                    Check(f.reason == "certificate_failed:sturm",
                          tag + " naive: reason = \"" + f.reason + "\"");
                } else if (r.outcome == SturmSweepOutcome::Passed) {
                    ++passed;
                }
            }
            Check(failed >= 1,
                  tag + " naive: NO block rejected the B' sweep -- the B(k) counterexample "
                        "does not reproduce on the pipeline's own H^1");
            Check(witness > c.B_naive,
                  tag + " naive: the failing prime " + std::to_string(witness) +
                      " is not above B(k) = " + std::to_string(c.B_naive));
            std::cout << "  NAIVE " << tag << " blocks=" << blocks.size()
                      << " failed=" << failed << " passed=" << passed
                      << " first_failing_prime=" << witness << std::endl;
        }

        // (ii) The B' arm: the decomposition and A_S built from the primes
        //      <= B', the same sweep -- every block must PASS.
        {
            const std::vector<uint64_t> full = hecke_ext::PrimesUpTo(c.B_prime);
            Point p(c.n, c.ell, full, s1.includes_u_ell());
            const std::vector<BlockRun> blocks = RunBlocks(
                p, SturmSweepOptions::from_settings(hecke::MathSettings(), c.k), true);
            for (const BlockRun& r : blocks) {
                Check(r.local, tag + " B': a block did not certify local");
                if (!r.local) continue;
                CheckEq(r.outcome, SturmSweepOutcome::Passed, tag + " B': outcome");
                CheckEq(r.cert.verify(), 0u, tag + " B': SturmCertificate::verify()");
                Check(r.cert.sturm_primes == full,
                      tag + " B': swept " + PrimeList(r.cert.sturm_primes));
                Check(r.cert.generators_certified == full,
                      tag + " B': generators_certified = " +
                          PrimeList(r.cert.generators_certified));
                const hecke_ext::SturmFinality f = hecke_ext::sweep_finality(r.cert);
                CheckEq(f.value, SturmFinal::True, tag + " B': final");
            }
            std::cout << "  BPRIME " << tag << " blocks=" << blocks.size()
                      << " swept=" << PrimeList(full) << std::endl;
        }
    }
}

// ---------------------------------------------------------------------------
// Arm 5 -- both `sturm.unavailable` modes on the forced-cap (k, ell) = (24, 5)
// case, i.e. n = 22, with the real A^{(1)}(T_q); and the R3 contract of
// [X] Sec. 1.3.1 (resume-invariance, a refused lower cap, a no-op resume,
// budget_history).
// ---------------------------------------------------------------------------

// Everything one forced-cap run of the sweep produces on one block.
struct CapRun {
    SturmSweepOutcome outcome = SturmSweepOutcome::InProgress;
    SturmCertificate cert;
    hecke_ext::SturmSweepState state;
};

void RunCapModesArm() {
    std::cout << "-- arm 5: the forced cap 3 at (n, ell) = (22, 5) -- nonfinal vs heuristic, "
                 "and the R3 resume\n";
    const uint32_t n = 22, ell = 5, k = 24;
    const uint32_t default_cap = SturmBound::default_cap(k);   // 12*B(24) = 36
    CheckEq(default_cap, 36u, "default_cap(24)");

    // The cap-3 bound itself: `none_at_cap`, recorded and not thrown.
    {
        SturmBound b(k, ell, 3);
        CheckEq(b.run(), hecke_ext::SturmOutcome::NoneAtCap,
                "(24, 5) at cap 3: the bound is not none_at_cap");
    }

    const hecke_ext::StageOneOperators s1 =
        hecke_ext::ComputeStageOneOperators(n, ell, hecke::UEllSetting());
    Point p(n, ell, hecke_ext::PrimesUpTo(5), s1.includes_u_ell());
    std::vector<std::pair<uint64_t, ExtMatrix>> ops = p.ops();
    hecke::HeckeModuleDecomposition decomp(ops, ell);
    decomp.compute();
    Check(!decomp.blocks().empty(), "(22, 5): no blocks");
    if (decomp.blocks().empty()) return;

    // One representative block -- the first, deterministically.
    const hecke::PrimaryBlock& block = decomp.blocks()[0];
    const LocalAlgebra L = hecke_ext::ComputeLocalAlgebra(block, ops, ell);
    Check(L.ok, "(22, 5): the first block did not certify local");
    if (!L.ok) return;

    auto make_source = [&]() {
        return hecke_ext::BlockRestrictionSource(
            block.basis_columns,
            hecke_ext::FamilyOperatorSupplier(p.family, p.degree, ell, s1.includes_u_ell()));
    };

    // (a) mode `nonfinal` (the DEFAULT) at cap 3: nothing is swept and the
    //     ideal is final = false with reason `sturm_unavailable`.
    CapRun nonfinal;
    {
        SturmSweepOptions o;
        o.cap = 3;
        o.mode = SturmUnavailableMode::NonFinal;
        hecke_ext::BlockRestrictionSource src = make_source();
        SturmSweep sw(L, n, src, o);
        nonfinal.outcome = sw.run();
        nonfinal.cert = sw.certificate();
        nonfinal.state = sw.state();
        CheckEq(nonfinal.outcome, SturmSweepOutcome::Unavailable,
                "cap 3 nonfinal: outcome");
        Check(!nonfinal.cert.has_B_prime, "cap 3 nonfinal: a B' appeared");
        Check(nonfinal.cert.sturm_primes.empty(),
              "cap 3 nonfinal: something was swept -- " +
                  PrimeList(nonfinal.cert.sturm_primes));
        CheckEq(nonfinal.cert.verify(), 0u, "cap 3 nonfinal: verify()");
        const hecke_ext::SturmFinality f = hecke_ext::sweep_finality(nonfinal.cert);
        CheckEq(f.value, SturmFinal::False, "cap 3 nonfinal: final value");
        Check(f.reason == "sturm_unavailable", "cap 3 nonfinal: reason = \"" + f.reason + "\"");
        Check(f.value != SturmFinal::Heuristic,
              "cap 3 nonfinal: \"heuristic\" was produced under the defaults");
        const hecke_ext::GenerationVerdict v = hecke_ext::generation_verdict_with_sweep(
            L.cyclicity.cyclic, CommutantOutcome::ExcludedBySocleBound, nonfinal.cert);
        Check(v.final_value != SturmFinal::Heuristic,
              "cap 3 nonfinal: the verdict is \"heuristic\" under the defaults");
    }

    // (b) mode `heuristic` at cap 3: the primes <= cap are swept, all lie in
    //     A_S, and the ideal is final = "heuristic" with sturm_primes.
    {
        SturmSweepOptions o;
        o.cap = 3;
        o.mode = SturmUnavailableMode::Heuristic;
        hecke_ext::BlockRestrictionSource src = make_source();
        SturmSweep sw(L, n, src, o);
        const SturmSweepOutcome out = sw.run();
        const SturmCertificate cert = sw.certificate();
        CheckEq(out, SturmSweepOutcome::Heuristic, "cap 3 heuristic: outcome");
        const std::vector<uint64_t> want = {2, 3};
        Check(cert.sturm_primes == want,
              "cap 3 heuristic: swept " + PrimeList(cert.sturm_primes));
        Check(cert.generators_certified == want,
              "cap 3 heuristic: generators_certified = " +
                  PrimeList(cert.generators_certified));
        CheckEq(cert.verify(), 0u, "cap 3 heuristic: verify()");
        const hecke_ext::SturmFinality f = hecke_ext::sweep_finality(cert);
        CheckEq(f.value, SturmFinal::Heuristic, "cap 3 heuristic: final value");
        Check(f.reason == "sturm_heuristic", "cap 3 heuristic: reason = \"" + f.reason + "\"");
        const hecke_ext::GenerationVerdict v = hecke_ext::generation_verdict_with_sweep(
            L.cyclicity.cyclic, CommutantOutcome::ExcludedBySocleBound, cert);
        CheckEq(v.final_value, SturmFinal::Heuristic, "cap 3 heuristic: the verdict's final");
        CheckEq(v.kind, GenerationKind::Sturm, "cap 3 heuristic: the verdict's kind");
        // Every member is re-derived independently.
        for (size_t i = 0; i < cert.t_q.size(); ++i) {
            Check(InAlgebra(L.t, cert.t_q[i], L.D, ell),
                  "cap 3 heuristic: T_" + std::to_string(cert.sturm_primes[i]) +
                      " is not in A_S");
        }
    }

    // (c) R3: resume(36) from the cap-3 state == a fresh run(36), entrywise,
    //     modulo budget_history; resume(2) refused; a resume on a complete
    //     state is a strict no-op; budget_history lists both caps.
    {
        SturmSweepOptions o3;
        o3.cap = 3;
        hecke_ext::BlockRestrictionSource src_a = make_source();
        SturmSweep resumed(L, n, src_a, o3);
        resumed.run();

        bool refused = false;
        try {
            resumed.resume(2);
        } catch (const std::invalid_argument& e) {
            refused = std::string(e.what()).find("cap below reached state") != std::string::npos;
        }
        Check(refused, "R3: resume(2) on a cap-3 state was not refused with the named error");

        const SturmSweepOutcome out = resumed.resume(default_cap);
        CheckEq(out, SturmSweepOutcome::Passed, "R3: the resumed sweep did not pass");
        CheckEq(resumed.B_prime(), 5u, "R3: the resumed B'");

        SturmSweepOptions o36;
        o36.cap = default_cap;
        hecke_ext::BlockRestrictionSource src_b = make_source();
        SturmSweep fresh(L, n, src_b, o36);
        CheckEq(fresh.run(), SturmSweepOutcome::Passed, "R3: the fresh sweep did not pass");

        Check(resumed.state().equals(fresh.state(), /*ignore_budget_history=*/true),
              "R3: the resumed state differs from a fresh run's");
        Check(!resumed.state().equals(fresh.state(), /*ignore_budget_history=*/false),
              "R3: the budget histories are equal, so the comparison proves nothing");
        const std::vector<uint32_t> want_hist = {3, default_cap};
        Check(resumed.state().budget_history == want_hist, "R3: budget_history");
        const std::vector<uint32_t> want_fresh = {default_cap};
        Check(fresh.state().budget_history == want_fresh, "R3: the fresh budget_history");

        // Entrywise on the certificate's own payload, not only on the state.
        const SturmCertificate cr = resumed.certificate();
        const SturmCertificate cf = fresh.certificate();
        CheckEq(cr.verify(), 0u, "R3: the resumed certificate does not verify");
        CheckEq(cf.verify(), 0u, "R3: the fresh certificate does not verify");
        Check(cr.sturm_primes == cf.sturm_primes, "R3: the swept lists differ");
        Check(cr.generators_certified == cf.generators_certified,
              "R3: generators_certified differs");
        Check(cr.B_prime == cf.B_prime && cr.dim_M_k == cf.dim_M_k &&
                  cr.closure_dim == cf.closure_dim,
              "R3: the bound scalars differ");
        bool same = cr.t_q.size() == cf.t_q.size();
        for (size_t i = 0; same && i < cr.t_q.size(); ++i) {
            for (uint32_t a = 0; a < cr.t_q[i].rows && same; ++a) {
                for (uint32_t b = 0; b < cr.t_q[i].cols && same; ++b) {
                    same = cr.t_q[i].data[a][b] == cf.t_q[i].data[a][b];
                }
            }
        }
        Check(same, "R3: a stored restriction differs entrywise");

        // A resume on the now-complete state is a STRICT no-op.
        const hecke_ext::SturmSweepState before = resumed.state();
        CheckEq(resumed.resume(default_cap + 24), SturmSweepOutcome::Passed,
                "R3: the no-op resume changed the outcome");
        Check(resumed.state().equals(before, /*ignore_budget_history=*/false),
              "R3: the resume on a complete state was not a strict no-op");
    }

    // (d) `sturm = off`: (c) is not attempted at all.
    {
        SturmSweepOptions o;
        o.cap = default_cap;
        o.sturm_off = true;
        hecke_ext::BlockRestrictionSource src = make_source();
        SturmSweep sw(L, n, src, o);
        CheckEq(sw.run(), SturmSweepOutcome::Unavailable, "sturm = off: outcome");
        const SturmCertificate cert = sw.certificate();
        Check(cert.sturm_primes.empty(), "sturm = off: something was swept");
        CheckEq(cert.verify(), 0u, "sturm = off: verify()");
        Check(hecke_ext::sweep_finality(cert).reason == "sturm_unavailable",
              "sturm = off: reason");
    }
}

// ---------------------------------------------------------------------------
// Arm 6 -- the `u_ell` budget check INSIDE the sweep: at (26, 7) the proof
// list contains ell = 7 (B' = 7 >= ell), so `u_ell = off` makes (c)
// unavailable with `u_ell_blocked` rather than silently dropping T_7.
// ---------------------------------------------------------------------------

void RunUEllInSweepArm() {
    std::cout << "-- arm 6: a refused U_ell inside the sweep at (26, 7), where ell <= B'\n";
    const uint32_t n = 26, ell = 7, k = 28;
    {
        SturmBound b(k, ell);
        b.run();
        Check(b.has_B_prime() && b.B_prime() >= ell,
              "(26, 7): ell is not on the proof list, so this arm proves nothing");
    }

    hecke::UEllSetting off;
    off.mode = hecke::UEllMode::Off;
    const hecke_ext::StageOneOperators s_off =
        hecke_ext::ComputeStageOneOperators(n, ell, off);
    Check(!s_off.includes_u_ell(), "(26, 7) off: U_7 was included");

    // The decomposition is built from the primes <= B' MINUS ell, exactly what
    // Stage 1 hands over under `u_ell = off`.
    std::vector<uint64_t> S;
    for (uint64_t q : hecke_ext::PrimesUpTo(7)) {
        if (q != ell) S.push_back(q);
    }
    Point p(n, ell, S, /*u_ell_included=*/false);
    const std::vector<BlockRun> blocks =
        RunBlocks(p, SturmSweepOptions::from_settings(hecke::MathSettings(), k), true);
    uint32_t blocked = 0;
    for (const BlockRun& r : blocks) {
        Check(r.local, "(26, 7) off: a block did not certify local");
        if (!r.local) continue;
        CheckEq(r.outcome, SturmSweepOutcome::Unavailable, "(26, 7) off: outcome");
        Check(r.cert.u_ell_blocked, "(26, 7) off: u_ell_blocked is not set");
        Check(r.cert.sturm_primes.empty(),
              "(26, 7) off: a partial sweep was published");
        CheckEq(r.cert.verify(), 0u, "(26, 7) off: verify()");
        Check(hecke_ext::sweep_finality(r.cert).reason == "sturm_unavailable",
              "(26, 7) off: reason");
        ++blocked;
    }
    Check(blocked >= 1, "(26, 7) off: no block was blocked by the u_ell verdict");
    std::cout << "  UELL_BLOCKED (26,7) blocks=" << blocks.size() << " blocked=" << blocked
              << std::endl;
}

// ---------------------------------------------------------------------------
// Arm 7 -- the tamper arm: every corruption fails EXACTLY its own clause of
// SturmCertificate::verify() ([X] Sec. 1.3 R17 -- the certificate re-derives,
// it does not read).
// ---------------------------------------------------------------------------

void RunTamperArm() {
    std::cout << "-- arm 7: a tampered SturmCertificate fails exactly its own clause\n";
    const uint32_t n = 22, ell = 5, k = 24;
    const hecke_ext::StageOneOperators s1 =
        hecke_ext::ComputeStageOneOperators(n, ell, hecke::UEllSetting());
    Point p(n, ell, hecke_ext::PrimesUpTo(5), s1.includes_u_ell());
    const std::vector<BlockRun> blocks =
        RunBlocks(p, SturmSweepOptions::from_settings(hecke::MathSettings(), k), true);
    Check(!blocks.empty(), "tamper: no blocks");
    if (blocks.empty()) return;

    const SturmCertificate base = blocks[0].cert;
    CheckEq(base.verify(), 0u, "tamper: the untampered certificate does not verify");
    Check(base.sturm_primes.size() >= 3,
          "tamper: the base certificate swept " + PrimeList(base.sturm_primes) +
              ", too few for the drop cases");
    if (base.sturm_primes.size() < 3) return;

    // (1) shape.
    { SturmCertificate c = base; c.k += 1; CheckEq(c.verify(), 1u, "tamper 1: k"); }
    { SturmCertificate c = base; c.D += 1; CheckEq(c.verify(), 1u, "tamper 1b: D"); }
    // (2) the embedded bound's own data.
    { SturmCertificate c = base; c.bound.closure_dim -= 1;
      CheckEq(c.verify(), 2u, "tamper 2: bound.closure_dim"); }
    { SturmCertificate c = base; c.bound.primes.pop_back(); c.bound.T_q.pop_back();
      CheckEq(c.verify(), 2u, "tamper 2b: a dropped T_q inside the bound"); }
    // (3) the bound describes another problem.
    { SturmCertificate c = base; c.B_prime += 2;
      CheckEq(c.verify(), 3u, "tamper 3: B'"); }
    { SturmCertificate c = base; c.closure_dim += 1;
      CheckEq(c.verify(), 3u, "tamper 3b: closure_dim"); }
    // (4) certificate (a).
    { SturmCertificate c = base; c.cyclicity.a_basis.pop_back();
      CheckEq(c.verify(), 4u, "tamper 4: the A_S basis"); }
    // (5) the prime list -- a DROPPED T_q, consistently removed everywhere.
    { SturmCertificate c = base;
      c.sturm_primes.pop_back(); c.t_q.pop_back(); c.member.pop_back();
      if (!c.generators_certified.empty()) c.generators_certified.pop_back();
      CheckEq(c.verify(), 5u, "tamper 5: a dropped swept prime"); }
    // (6) misalignment.
    { SturmCertificate c = base; c.t_q.pop_back();
      CheckEq(c.verify(), 6u, "tamper 6: t_q misaligned"); }
    { SturmCertificate c = base; c.t_q[0].data[0][0] = (c.t_q[0].data[0][0] + 1) % ell;
      c.t_q[0].rows = 0;
      CheckEq(c.verify(), 6u, "tamper 6b: a restriction of the wrong shape"); }
    // (7) membership.
    { SturmCertificate c = base; c.member[0] = 0;
      CheckEq(c.verify(), 7u, "tamper 7: a flipped membership flag"); }
    { SturmCertificate c = base;
      c.t_q[0].data[0][0] = (c.t_q[0].data[0][0] + 1) % ell;
      const uint32_t code = c.verify();
      Check(code == 7u || code == 8u,
            "tamper 7b: a perturbed restriction gave code " + std::to_string(code));
    }
    // (8) the derived lists.
    { SturmCertificate c = base; c.generators_certified.pop_back();
      CheckEq(c.verify(), 8u, "tamper 8: generators_certified"); }
    { SturmCertificate c = base; c.failing_prime = 3;
      CheckEq(c.verify(), 8u, "tamper 8b: failing_prime"); }
    // (9) the outcome.
    { SturmCertificate c = base; c.outcome = SturmSweepOutcome::Failed;
      CheckEq(c.verify(), 9u, "tamper 9: outcome"); }
    { SturmCertificate c = base; c.proof_range = false;
      CheckEq(c.verify(), 9u, "tamper 9b: proof_range"); }

    // The JSON carries the swept list, the verdict and the R3 object.
    const std::string j = base.to_json();
    Check(j.find("\"schema\":\"sturm_certificate/v1\"") != std::string::npos,
          "to_json: no schema key");
    Check(j.find("\"sturm_primes\":[2,3,5]") != std::string::npos,
          "to_json: wrong sturm_primes");
    Check(j.find("\"outcome\":\"passed\"") != std::string::npos, "to_json: wrong outcome");
    Check(j.find("\"budget_setting\":\"sturm.cap\"") != std::string::npos,
          "to_json: the resumable object does not name its budget setting");
    Check(base.resumable_json().find("\"kind\":\"sturm_sweep\"") != std::string::npos,
          "resumable_json: no kind");
}

// ---------------------------------------------------------------------------
// Arm 8 -- `not_applicable` at n = 0 ([E] v1.4.0 Def. 2.8: dim M_2 = 0).
// ---------------------------------------------------------------------------

void RunNotApplicableArm() {
    std::cout << "-- arm 8: n = 0 is not_applicable (dim M_2 = 0)\n";
    CheckEq(SturmBound::dimension_M_k(2), 0u, "dim M_2");
    const uint32_t n = 0, ell = 5;
    const hecke_ext::StageOneOperators s1 =
        hecke_ext::ComputeStageOneOperators(n, ell, hecke::UEllSetting());
    Point p(n, ell, s1.primes, s1.includes_u_ell(), /*degree=*/0);
    Check(p.dim() > 0, "(0, 5): H^0 is empty");
    if (p.dim() == 0) return;
    const std::vector<BlockRun> blocks =
        RunBlocks(p, SturmSweepOptions::from_settings(hecke::MathSettings(), 2), true);
    Check(!blocks.empty(), "(0, 5): no blocks in H^0");
    for (const BlockRun& r : blocks) {
        if (!r.local) continue;
        CheckEq(r.outcome, SturmSweepOutcome::NotApplicable, "(0, 5): outcome");
        Check(r.cert.sturm_primes.empty(), "(0, 5): something was swept");
        CheckEq(r.cert.verify(), 0u, "(0, 5): verify()");
        Check(hecke_ext::sweep_finality(r.cert).reason == "sturm_unavailable",
              "(0, 5): reason");
    }
}

// ---------------------------------------------------------------------------
// Arm 10 -- acceptance (c) of [X] Sec. 6.2 E1: the ext layer's wall time on the
// {2, 3, 5} family at n = 256, RECORDED against the pipeline's own, never
// asserted (a slower box must not fail the suite; the >= 20 % gate becomes
// real once E2's `--bench ext` exists).
// ---------------------------------------------------------------------------

void RunScaleRecordArm() {
    std::cout << "-- arm 10: the ext layer's wall time at n = 256 (recorded, not asserted)\n";
    const uint32_t n = 256, ell = 134217757;
    const std::vector<uint64_t> S = {2, 3, 5};

    const auto t0 = std::chrono::steady_clock::now();
    Stack stack(n, ell);
    hecke::HeckeOperatorFamily family(&stack.coh, stack.K, stack.model, ell, n);
    std::vector<std::pair<uint64_t, ExtMatrix>> ops;
    for (uint64_t q : S) ops.push_back({q, family.T_p(q)[1]});
    const auto t1 = std::chrono::steady_clock::now();

    hecke::HeckeModuleDecomposition decomp(ops, ell);
    decomp.compute();
    const auto t1b = std::chrono::steady_clock::now();
    uint32_t blocks = 0, swept = 0, passed = 0;
    for (const hecke::PrimaryBlock& b : decomp.blocks()) {
        ++blocks;
        LocalAlgebra L = hecke_ext::ComputeLocalAlgebra(b, ops, ell);
        if (!L.ok) continue;
        SturmSweepOptions o = SturmSweepOptions::from_settings(hecke::MathSettings(), n + 2);
        o.proof_range = (hecke::DicksonInvariantCount(n, ell) == 0);
        hecke_ext::BlockRestrictionSource src(
            b.basis_columns, hecke_ext::FamilyOperatorSupplier(family, 1, ell, false));
        SturmSweep sw(L, n, src, o);
        const SturmSweepOutcome out = sw.run();
        const SturmCertificate cert = sw.certificate();
        CheckEq(cert.verify(), 0u, "n = 256: SturmCertificate::verify()");
        swept += static_cast<uint32_t>(cert.sturm_primes.size());
        if (out == SturmSweepOutcome::Passed) ++passed;
    }
    const auto t2 = std::chrono::steady_clock::now();

    const double pipe = std::chrono::duration<double>(t1 - t0).count();
    const double decomp_s = std::chrono::duration<double>(t1b - t1).count();
    const double ext = std::chrono::duration<double>(t2 - t1b).count();
    std::cout << "  EXT_SCALE n=256 ell=" << ell << " family={2,3,5} pipeline_s=" << pipe
              << " decomp_s=" << decomp_s << " ext_s=" << ext
              << " ratio=" << (pipe > 0 ? ext / pipe : 0.0)
              << " dim_H1=" << (stack.coh.get_H_eq_dimensions().size() > 1
                                    ? stack.coh.get_H_eq_dimensions()[1]
                                    : 0u)
              << " blocks=" << blocks << " passed=" << passed << " swept_ops=" << swept
              << std::endl;
    Check(blocks > 0, "n = 256: no blocks");
}

// ---------------------------------------------------------------------------
// Arm 9 -- OFF the proof range: at a Dickson-exceptional cell [E] cl:sturmmodl
// proves nothing, so a COMPLETE sweep is `Heuristic`, never `Passed`, and in
// the default configuration the ideal is `final = false` with reason
// `sturm_heuristic`.  (The conservative rule used here and in the arms above:
// `proof_range = (DicksonInvariantCount(n, ell) == 0)` -- no Dickson classes at
// all means no Dickson-type maximal ideal.  Deciding it per ideal needs E6's
// residue atlas, which is why `proof_range` is an INPUT the caller asserts.)
// ---------------------------------------------------------------------------

void RunOffProofRangeArm() {
    std::cout << "-- arm 9: off the proof range (a Dickson-exceptional cell) a complete sweep "
                 "is heuristic\n";
    const uint32_t n = 12, ell = 5, k = 14;
    Check(hecke::DicksonInvariantCount(n, ell) > 0,
          "(12, 5) is not Dickson-exceptional, so this arm proves nothing");

    const hecke_ext::StageOneOperators s1 =
        hecke_ext::ComputeStageOneOperators(n, ell, hecke::UEllSetting());
    Point p(n, ell, s1.primes, s1.includes_u_ell());
    if (p.dim() == 0) {
        Check(false, "(12, 5): H^1 is empty");
        return;
    }
    // The SAME cell run both ways: the only difference is the claim.
    const std::vector<BlockRun> on =
        RunBlocks(p, SturmSweepOptions::from_settings(hecke::MathSettings(), k), true);
    const std::vector<BlockRun> off =
        RunBlocks(p, SturmSweepOptions::from_settings(hecke::MathSettings(), k), false);
    CheckEq(on.size(), off.size(), "(12, 5): the two arms disagree on the block count");
    uint32_t heuristic = 0;
    for (size_t i = 0; i < on.size() && i < off.size(); ++i) {
        if (!on[i].local || !off[i].local) continue;
        CheckEq(on[i].cert.verify(), 0u, "(12, 5) proof_range: verify()");
        CheckEq(off[i].cert.verify(), 0u, "(12, 5) off-range: verify()");
        // The COMPUTATION is identical -- only the verdict differs.
        Check(on[i].cert.sturm_primes == off[i].cert.sturm_primes,
              "(12, 5): the two arms swept different primes");
        Check(on[i].cert.generators_certified == off[i].cert.generators_certified,
              "(12, 5): the two arms certified different generators");
        if (on[i].outcome == SturmSweepOutcome::Passed) {
            CheckEq(off[i].outcome, SturmSweepOutcome::Heuristic,
                    "(12, 5) off-range: a complete sweep is not heuristic");
            const hecke_ext::SturmFinality f = hecke_ext::sweep_finality(off[i].cert);
            CheckEq(f.value, SturmFinal::False, "(12, 5) off-range: final under the defaults");
            Check(f.reason == "sturm_heuristic",
                  "(12, 5) off-range: reason = \"" + f.reason + "\"");
            Check(hecke_ext::sturm_route_status(off[i].cert) ==
                      hecke_ext::SturmRouteStatus::Unavailable,
                  "(12, 5) off-range: route status is not unavailable");
            ++heuristic;
        }
    }
    Check(heuristic >= 1, "(12, 5): no block completed its sweep, so the arm proves nothing");
    std::cout << "  OFF_RANGE (12,5) blocks=" << on.size() << " heuristic=" << heuristic
              << std::endl;
}

} // namespace

uint32_t HeckeExtSturmSweepValidation() {
    Check.reset();
    const auto t0 = std::chrono::steady_clock::now();

    std::cout << "=== HeckeExtSturmSweepValidation (hecke-extensions slot 3, WP-2 / E1b) ==="
              << std::endl;
    try {
        RunStageOneArm();
        RunRamanujanArm();
        RunJointGroupArm();
        RunNaiveBoundArm();
        RunCapModesArm();
        RunUEllInSweepArm();
        RunTamperArm();
        RunNotApplicableArm();
        RunOffProofRangeArm();
        RunScaleRecordArm();
    } catch (const std::exception& e) {
        ++Check.fails;
        std::cout << "  [FAIL] uncaught exception: " << e.what() << std::endl;
    }

    const double secs =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "STURM_SWEEP checks=" << Check.checks << " failures=" << Check.fails << " seconds="
              << secs << std::endl;
    return Check.fails;
}
