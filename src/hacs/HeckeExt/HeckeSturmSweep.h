// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_STURM_SWEEP_H
#define HECKE_EXT_STURM_SWEEP_H

// ===========================================================================
// hecke-extensions phase E1b (slot 3, WP-2) -- GENERATION CERTIFICATE (c),
// THE STURM SWEEP ON REAL BLOCKS, and Stage 1's `u_ell` budget check.
//
// WHAT THIS COMPUTES.  [E] = docs/latex/hecke_extensions.tex v1.4.0,
// Certificate `cert:sturm` ("(c) Sturm sweep -- fallback") on the range of
// Claim `cl:sturmmodl`; [X] = hecke_extensions_implementation_plan.md v1.6,
// Sec. 2.2 Stage 1 and Stage 4(c), Sec. 1.3.1 R3.2 / R3.3; [C] =
// docs/latex/gl2_contract.tex v1.4.2 (`obl:algebra` / `thm:relations`, which
// is what makes T_S commutative and therefore makes A_S an algebra at all).
//
// Fix a block V = H_m of a computed decomposition that `hecke::CertifyLocality`
// has certified LOCAL, with its local algebra A_S = F_ell[t_p : p in S]
// (HeckeExt/HeckeLocalAlgebra).  Certificate (c) is:
//
//   1. compute B'(k, ell), k = n + 2 (HeckeExt/HeckeSturmBound -- pure
//      modular-form arithmetic on the Miller basis, no pipeline);
//   2. for EVERY prime q <= B'(k, ell), form A^{(q_deg)}(T_q), restrict it to
//      V and test T_q|_V in A_S by ONE solve against the RREF coordinate
//      basis of A_S (`hecke::RrefRowSpan::contains`);
//   3. if every one lies in A_S then, on the proof range of [E] cl:sturmmodl
//      (a non-Dickson maximal ideal m of X_4 = H^1), A_S = T_m -- no further
//      T_q can enlarge the algebra at m.  Off that range the same computation
//      is a HEURISTIC and is recorded as one.
//
// THE PRIME LIST IS `q <= B'`, AND NOTHING ELSE ([X] Sec. 2.2 Stage 4(c)
// item 2 / [XA B3]).  In particular ell is swept iff ell <= B', because
// [E] cl:sturmmodl writes every T_q -- q = ell included -- as
// F_q(T_{q' <= B'}) + ell*t' with t' acting as 0 on the proof range, so T_ell
// needs no membership test of its own.  The heuristic-range additions
// S_sep(ell) of [E] cl:sep are E6's (the atlas): they are NOT swept here, and
// the certificate says which list it used rather than implying a larger one.
//
// WHY B' AND NEVER B(k) = floor(k/12) + 1.  At (n, ell) = (22, 5) and (26, 7)
// the primes <= B(k) = 3 generate a PROPER subalgebra of T_Z(M_k) ⊗ F_ell
// ([E] cl:sturm(b), cl:bprime), so the naive sweep would certify A_S = T_m
// falsely on the pipeline's own H^1.  HeckeExtSturmSweepValidation pins both
// halves: the {2, 3} sweep is REJECTED (T_5 resp. T_7 is not in A_{2,3}) and
// the B' sweep is accepted.
//
// THE TWO `sturm.*` MODES ([X] Sec. 1.5, OQ15 / slot-1 confirmation 6).
//   * `sturm.cap` -- the cap B_max the bound search stops at; `auto` is
//     12*B(k), `cap=<B>` pins it, `off` skips (c) entirely.
//   * `sturm.unavailable` -- what a `none_at_cap` bound does.  The DEFAULT
//     `nonfinal` leaves the ideal `final = false` with reason
//     `sturm_unavailable` and sweeps NOTHING.  The selectable `heuristic`
//     sweeps the primes <= cap anyway and emits `final = "heuristic"` with
//     `sturm_primes` -- a value that is NEVER produced under the defaults.
// Both are registered mathematical settings (Hecke/HeckeSettings.h, keys
// `sturm` and `sturm_unavailable`, appended after `u_ell` in slot-3 WP-2), so
// a non-default choice changes the fingerprint, the record and the job id.
//
// RESUMABLE ([X] Sec. 1.3.1 R3, the HeckeSturmBound / CommutantRoute pattern).
// The sweep's budget IS `sturm.cap`: `state()` carries the embedded
// `SturmState` plus the swept list and the restrictions, `resume(new_cap)`
// continues the BOUND from its stored state and re-sweeps under the new B',
// a cap below the one already reached is refused with a named error, a resume
// on a complete state is a strict no-op, and `budget_history` lists every cap.
// Resume-invariance holds because the bound is resume-invariant (its own
// header states why) and the sweep is a pure function of (B', mode, V, A_S).
//
// STAGE 1 ([X] Sec. 2.2 Stage 1, [XA B3]).  `ComputeStageOneOperators` is the
// other half of this package: the base operator set
// S_0 = {2, 3, 5, 7} ∪ {ell iff the frozen RSS model admits U_ell}, with the
// `u_ell` setting consulted through HeckeExt/HeckeUEllBudget.h and the verdict
// returned rather than inferred.  `u_ell = off` removes ell from S_0 even when
// ell is one of the four good primes (at ell <= 7, T_ell IS U_ell -- the same
// code path, invariant I7).  The R3 resume of the decision (re-running Stage 1
// with U_ell attached to the same family) is E2's, per slot-3 plan Sec. 2.3
// item 2.
//
// PLACEMENT.  HeckeExt/, MatrixT<uint32_t> throughout.  Unlike
// HeckeSturmBound.{h,cpp} -- which guard rule 3 keeps pipeline-free -- this TU
// MAY include Hecke/ pipeline headers, and does (HeckeOperatorFamily, for the
// family-backed operator source below).  It includes no Engine/, Cuda/ or
// Console/ header and no BlockFlatMatrix.h (guard rule 2), and no
// `unordered_` container (rule 4).
// ===========================================================================

#include "HeckeGenerationCertificates.h"
#include "HeckeLocalAlgebra.h"
#include "HeckeSturmBound.h"
#include "HeckeUEllBudget.h"

#include "../Hecke/HeckeSettings.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// Forward declarations only: the streamed supplier below takes references to
// the pipeline's objects but this header names no pipeline header, so a
// consumer of the sweep still pays for none of them.
template<class Elt> class EquivariantCohomologyOnKT;
using EquivariantCohomologyOnK = EquivariantCohomologyOnKT<uint32_t>;
class SimplicialComplex;

namespace hecke {
class HeckeOperatorFamily;
class SL2ZTessellationTables;
}

namespace hecke_ext {

// ---------------------------------------------------------------------------
// Stage 1 -- the base operator set and the `u_ell` verdict.
// ---------------------------------------------------------------------------

// The four good primes of [E] Stage 1, in the fixed ascending order every
// later stage assumes.
const std::vector<uint64_t>& DefaultGoodPrimes();

struct StageOneOperators {
    uint32_t n = 0;
    uint32_t ell = 0;
    // S_0, ASCENDING and duplicate-free.  `ell` appears iff `u_ell.included`.
    std::vector<uint64_t> primes;
    UEllVerdict u_ell;

    bool includes_u_ell() const;
    // Fixed key order, no hash container.
    std::string to_json() const;
};

// Stage 1 under an explicit `u_ell` setting.  `good_primes` defaults to
// DefaultGoodPrimes(); ell is REMOVED from it when the verdict refuses U_ell,
// because at ell <= 7 the good prime ell IS U_ell.
StageOneOperators ComputeStageOneOperators(uint32_t n, uint32_t ell,
                                           const hecke::UEllSetting& setting);
StageOneOperators ComputeStageOneOperators(uint32_t n, uint32_t ell,
                                           const hecke::UEllSetting& setting,
                                           const std::vector<uint64_t>& good_primes);
// The same under the `u_ell` setting in force (hecke::math_settings()).
StageOneOperators ComputeStageOneOperators(uint32_t n, uint32_t ell);

// ---------------------------------------------------------------------------
// The operator source.
// ---------------------------------------------------------------------------
//
// The sweep never builds a Hecke operator itself: it ASKS for T_q restricted
// to the block.  That keeps the unit testable without a pipeline, lets the
// caller decide between the family cache and a streamed local pipeline
// ([X] Sec. 2.2 Stage 4(c) item 2, "reuse before streaming"), and is where a
// refused U_ell turns into `false` rather than into a 1.3e8-coset run.
class SturmOperatorSource {
public:
    virtual ~SturmOperatorSource() = default;

    // Fills `out` with T_q|_V (D x D over F_ell) and returns true.  Returns
    // false when the operator is not available at all -- today exactly the
    // case q == ell with U_ell refused by the `u_ell` budget.  Must be
    // deterministic and repeatable: the sweep may ask twice (a resume).
    virtual bool restricted_operator(uint64_t q, ExtMatrix& out) = 0;
};

// A source over a supplier of A^{(deg)}(T_q) on the WHOLE cohomology, which it
// restricts to the block by the exact solve `basis * X = A * basis`
// (hecke::SolveColumnsExact -- solvability IS the invariance certificate).
// Results are memoised in an ascending vector, never a hash container.
class BlockRestrictionSource : public SturmOperatorSource {
public:
    // Returns false iff T_q cannot be produced.
    using Supplier = std::function<bool(uint64_t q, ExtMatrix& out)>;

    BlockRestrictionSource(ExtMatrix basis_columns, Supplier supplier);

    bool restricted_operator(uint64_t q, ExtMatrix& out) override;

    // How many restrictions were actually formed (the rest were cache hits).
    uint32_t solves() const { return solves_; }

private:
    ExtMatrix basis_;
    Supplier  supplier_;
    std::vector<uint64_t>  cached_q_;
    std::vector<ExtMatrix> cached_;
    std::vector<uint8_t>   cached_ok_;
    uint32_t solves_ = 0;
};

// A supplier over a computed `hecke::HeckeOperatorFamily`, honouring a Stage-1
// verdict: q == ell is refused (returns false) unless `u_ell_included`.  The
// family's own cache is keyed by (p, r), so a prime already in S_0 costs
// nothing; a prime outside it is built by the family on demand.
BlockRestrictionSource::Supplier FamilyOperatorSupplier(hecke::HeckeOperatorFamily& family,
                                                        uint32_t degree,
                                                        uint64_t ell,
                                                        bool u_ell_included);

// ---------------------------------------------------------------------------
// (hecke-extensions slot 4, WP-2 / phase E2b) THE STREAMED SUPPLIER -- the
// other half of [X] Sec. 2.2 Stage 4(c) item 2's "reuse before streaming".
// ---------------------------------------------------------------------------
//
// A prime q outside S_0 must NOT go through HeckeOperatorFamily: its
// `pipeline_cache_` never evicts (Hecke/HeckeOperatorFamily.h), so a sweep
// that asked the family for three extra primes at n = 1000 would hold three
// extra ~1.4 GB pipelines to the end of the cell.  The streamed route builds
// the pipeline in a LOCAL scope --
//
//     HeckeCosetTables(q, 1, ell, n) -> HeckeTauTable -> HeckeOperatorOnK
//
// under `hecke::tau_construction_mutex()` for the two table CONSTRUCTIONS (the
// process-wide lock of Hecke/HeckeOperatorFamily.h) and under
// `ScopedThetaEqCertification(false)` + `ScopedMemoryLean(true)` for the
// compute(), keeps only A^{(degree)}(T_q) -- which BlockRestrictionSource
// immediately restricts to the D x D block -- and then DESTROYS the pipeline,
// so at most one streamed pipeline is alive at a time.
//
// The two scoped options are `thread_local` (Hecke/HeckeOperatorOnK.h Sec.
// "Scoped*"), so a streamed build on one thread cannot change what another
// thread computes; `CheckThread` on the streamed arm is what pins that.
//
// The value is IDENTICAL to the family's: `ScopedThetaEqCertification(false)`
// skips a CERTIFICATION and `ScopedMemoryLean(true)` releases intermediates
// neither of which A^{(q)} reads -- the same guarantee `--bench --lean`
// carries ("A^(q) unchanged").  The Ramanujan validation's streamed arm
// asserts it entrywise against the family-cached restrictions.

// What a streamed run cost, for the record and for `--bench ext`'s
// `streamed_s` / RSS-per-prime column ([X] plan Sec. 3 WP-2 risk 1).
struct StreamedOperatorStats {
    uint32_t built = 0;            // pipelines actually constructed
    double   wall_s = 0.0;         // summed over them
    double   peak_rss_mb = 0.0;    // the process VmHWM after the last one
    uint64_t last_prime = 0;
};

// `coh`, `K` and `model` must outlive the returned supplier and must be the
// SAME objects the block's operators were computed on -- A^{(degree)}(T_q) is
// only comparable to them in that cohomology's stored basis.  `stats` may be
// null; when it is not, it must outlive the supplier too.
BlockRestrictionSource::Supplier StreamedOperatorSupplier(
        const EquivariantCohomologyOnK& coh, const SimplicialComplex& K,
        const hecke::SL2ZTessellationTables& model, uint32_t ell, uint32_t n_sym,
        uint32_t degree, bool u_ell_included, StreamedOperatorStats* stats);

// (slot-7 WP-3) The streamed build itself, keeping EVERY degree's
// A^{(q)}(T_q) -- what StreamedOperatorSupplier calls and then cuts to one
// degree.  The cell-local `proof_range` decision needs degrees 0 and 1 of the
// same T_q, and one pipeline build serves both.  Same scope rules, same value.
std::vector<ExtMatrix> StreamedOperatorAllDegrees(
        const EquivariantCohomologyOnK& coh, const SimplicialComplex& K,
        const hecke::SL2ZTessellationTables& model, uint32_t ell, uint32_t n_sym, uint64_t q,
        StreamedOperatorStats* stats);

// "Reuse before streaming": `reuse` answers every q in `s0` (ascending,
// duplicate-free), `stream` answers every other q.  Neither is consulted for a
// prime the other owns, so a family pipeline is never built for a streamed
// prime and vice versa.
BlockRestrictionSource::Supplier ReuseBeforeStreamingSupplier(
        std::vector<uint64_t> s0,
        BlockRestrictionSource::Supplier reuse,
        BlockRestrictionSource::Supplier stream);

// ---------------------------------------------------------------------------
// Outcomes.
// ---------------------------------------------------------------------------

enum class SturmSweepOutcome : uint32_t {
    InProgress    = 0,  // constructed but not yet run
    NotApplicable = 1,  // dim M_k == 0 (n = 0, odd n, k < 4): [E] Def. 2.8
    Passed        = 2,  // B' exists, proof range, every T_q|_V (q <= B') in A_S
    Failed        = 3,  // some swept T_q|_V is NOT in A_S -- a finding
    Unavailable   = 4,  // no B' below the cap (mode nonfinal), `sturm = off`,
                        // or a swept ell whose U_ell the budget refused
    Heuristic     = 5   // every swept T_q|_V in A_S, but NOT a proof: either
                        // no B' under mode `heuristic`, or off the proof range
};
const char* ToString(SturmSweepOutcome outcome);

// ---------------------------------------------------------------------------
// Options -- the two `sturm.*` settings, plus the caller's proof-range claim.
// ---------------------------------------------------------------------------

struct SturmSweepOptions {
    // 0 selects SturmBound::default_cap(k) = 12*B(k).
    uint32_t cap = 0;
    // `sturm = off`: (c) is not attempted at all.
    bool sturm_off = false;
    // `sturm.unavailable`.
    SturmUnavailableMode mode = SturmUnavailableMode::NonFinal;
    // [E] cl:sturmmodl's range: a non-Dickson maximal ideal of X_4.  The
    // CALLER asserts it (identifying Dickson-type m needs the residue atlas,
    // E6); the certificate records the claim, and off the range a complete
    // sweep is `Heuristic`, never `Passed`.
    bool proof_range = true;

    // The two `sturm.*` mathematical settings in force, mapped onto `cap` /
    // `sturm_off` / `mode`.  `proof_range` is untouched (it is not a setting).
    static SturmSweepOptions from_settings(const hecke::MathSettings& s, uint32_t k);
    static SturmSweepOptions from_settings(uint32_t k);  // the settings in force
};

// ---------------------------------------------------------------------------
// The resumable state ([X] Sec. 1.3.1 R3.1, the SturmState member pattern).
// ---------------------------------------------------------------------------

struct SturmSweepState {
    uint32_t n = 0, k = 0, ell = 0, D = 0;

    // The budget in force and its history (R3.1 item 4: one entry per
    // run()/resume(); the LAST one is the setting in force).
    uint32_t              cap = 0;
    std::vector<uint32_t> budget_history;

    // The embedded bound's own R3 state -- a resume continues THIS, never a
    // fresh search (R3.1 item 2).
    SturmState bound;

    // Progress: the primes swept, ascending, index-aligned with `member` and
    // `t_q`.  Empty under `Unavailable` / `NotApplicable`.
    std::vector<uint64_t>  swept;
    std::vector<uint8_t>   member;
    std::vector<ExtMatrix> t_q;

    uint64_t failing_prime = 0;   // the first non-member, 0 if none
    bool     u_ell_blocked = false;

    SturmUnavailableMode mode = SturmUnavailableMode::NonFinal;
    bool                 proof_range = true;
    bool                 sturm_off = false;

    SturmSweepOutcome outcome = SturmSweepOutcome::InProgress;

    // Entrywise equality.  `ignore_budget_history` is what resume-invariance
    // needs (R3.1 item 3 wants the same state entrywise, item 4 wants every
    // run/resume to append its budget); it is passed down to `bound`.
    bool equals(const SturmSweepState& other, bool ignore_budget_history) const;

    // Fixed key order, no pointers, no hash container (R3.1 item 1).
    std::string to_json() const;
};

// ---------------------------------------------------------------------------
// The certificate ([X] Sec. 1.3 `SturmCertificate`).
// ---------------------------------------------------------------------------
//
// verify() RE-DERIVES, and reads nothing it concludes from ([X] Sec. 1.3 R17):
//   * the embedded SturmBoundCertificate re-derives B' from the Miller basis;
//   * the embedded CyclicityCertificate re-derives A_S, m_S and the Nakayama
//     quotient from the stored t_p;
//   * the sweep's prime list is REBUILT from (B', cap, mode) by a sieve and
//     compared element by element -- so a dropped T_q fails;
//   * every stored T_q|_V is re-tested for membership in the RE-DERIVED A_S,
//     and the stored flags are compared -- so a flipped flag or a perturbed
//     restriction fails;
//   * `generators_certified`, `failing_prime` and `outcome` are all recomputed.
// What it does NOT do is re-run the cohomology pipeline: the claim it certifies
// is "the STORED T_q|_V lie in the algebra generated by the STORED t_p, and the
// prime list is the one B' prescribes", which is exactly [E] cert:sturm's
// content.  The provenance of T_q|_V is the pipeline's own exact-solve
// invariance certificate (hecke::SolveColumnsExact), not this verify().
struct SturmCertificate {
    uint32_t n = 0, k = 0, ell = 0, D = 0;

    bool     has_B_prime = false;
    uint32_t B_prime = 0;
    uint32_t cap = 0;
    uint32_t dim_M_k = 0;
    uint32_t closure_dim = 0;

    SturmUnavailableMode mode = SturmUnavailableMode::NonFinal;
    bool proof_range = true;
    bool sturm_off = false;

    // The primes actually swept, index-aligned with `t_q` and `member`.
    std::vector<uint64_t>  sturm_primes;
    std::vector<ExtMatrix> t_q;
    std::vector<uint8_t>   member;
    // The swept primes that ARE in A_S, in the same order -- [X] Sec. 1.2's
    // `generators_certified` / [E] cert:sturm's S_m.
    std::vector<uint64_t>  generators_certified;

    uint64_t failing_prime = 0;
    bool     u_ell_blocked = false;

    SturmSweepOutcome outcome = SturmSweepOutcome::InProgress;

    SturmBoundCertificate bound;
    CyclicityCertificate  cyclicity;   // carries ell, D, the primes and t_p

    SturmSweepState state;
    // Where the state goes when its JSON exceeds 64 KiB (R3.1 item 6).  Only
    // the record layer (E2) knows the record stem, so only it can name it.
    std::string state_sidecar_stem;

    // 0 on success, else the index of the first failing clause.
    uint32_t verify() const;
    static const char* verify_message(uint32_t code);

    std::string to_json() const;
    std::string resumable_json() const;
};

// (c)'s contribution to the route-combining verdict of
// HeckeGenerationCertificates' `generation_verdict`.
SturmRouteStatus sturm_route_status(const SturmCertificate& c);

// (outcome, mode) -> the Sturm SWEEP's contribution to `final` ([X] R3.2).
// `Heuristic` is produced ONLY under `sturm.unavailable = heuristic`; in the
// default configuration the same outcome is `final = false` with reason
// `sturm_heuristic`.
SturmFinality sweep_finality(const SturmCertificate& c);

// The full route-combining verdict with (c) supplied by a sweep certificate.
// Identical to `generation_verdict(cyclic, b, sturm_route_status(c))` except
// that the `heuristic` tri-state value of R3.2 is reachable.
GenerationVerdict generation_verdict_with_sweep(bool cyclic_passed,
                                                CommutantOutcome b,
                                                const SturmCertificate& c);

// ---------------------------------------------------------------------------
// The computation.
// ---------------------------------------------------------------------------

class SturmSweep {
public:
    // `L` must be a LocalAlgebra with ok == true; everything it needs is
    // copied, so `L` need not outlive the sweep.  `src` MUST outlive it (a
    // resume asks it again).  `n` is the local system's n_sym; k = n + 2.
    SturmSweep(const LocalAlgebra& L, uint32_t n, SturmOperatorSource& src,
               const SturmSweepOptions& opts = SturmSweepOptions());

    // Runs from scratch under the constructed cap.  Exactly once; use
    // resume() afterwards.
    SturmSweepOutcome run();

    // Continues from state(), never from scratch (R3.1 item 2):
    //   * a cap below the one already reached is REFUSED with the named error
    //     "resume: cap below reached state";
    //   * on anything but a cap-limited `Unavailable` / `Heuristic` state it is
    //     a STRICT no-op -- neither the cap nor the history moves;
    //   * otherwise the embedded bound continues and the sweep is redone under
    //     the new B'.
    SturmSweepOutcome resume(uint32_t new_cap);

    // (slot 4, WP-2 / E2b) The wall the EMBEDDED BOUND took, summed over
    // run() and every resume().  A diagnostic only -- deliberately NOT part of
    // SturmSweepState, whose entrywise `equals` is what resume-invariance is
    // stated against.  `--bench ext`'s `sturm_bound_s` column and the plan's
    // "<= 0.1 s at n <= 256" acceptance read it.
    double bound_wall_s() const { return bound_wall_s_; }

    const SturmSweepState& state() const { return st_; }
    SturmSweepOutcome outcome() const { return st_.outcome; }
    bool     has_B_prime() const { return st_.bound.outcome == SturmOutcome::Found; }
    uint32_t B_prime() const;
    uint32_t cap() const { return st_.cap; }

    SturmCertificate certificate() const;

private:
    void sweep_();                       // the membership loop
    std::vector<uint64_t> prime_list_() const;

    uint32_t ell_ = 0, D_ = 0;
    std::vector<ExtMatrix> a_basis_;     // the RREF basis of A_S
    CyclicityCertificate   cyc_;
    SturmOperatorSource*   src_ = nullptr;
    // The embedded bound, kept ALIVE so that resume() continues it rather than
    // restarting it (R3.1 item 2).  Null until run().
    std::unique_ptr<SturmBound> bound_;
    double bound_wall_s_ = 0.0;
    bool ran_ = false;

    SturmSweepState st_;
};

// The primes <= B, ascending.  Exposed because verify() and the validation
// both re-derive the sweep list from it independently of the sweep.
std::vector<uint64_t> PrimesUpTo(uint32_t B);

} // namespace hecke_ext

#endif // HECKE_EXT_STURM_SWEEP_H
