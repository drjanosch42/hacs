// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_GENERATION_CERTIFICATES_H
#define HECKE_EXT_GENERATION_CERTIFICATES_H

// ===========================================================================
// hecke-extensions phase E1a (slot 2, WP-5) -- GENERATION CERTIFICATE (b),
// the COMMUTANT, with the socle bound and the resumable `CommutantState`.
//
// WHAT THIS COMPUTES.  [E] = docs/latex/hecke_extensions.tex v1.3.3,
// certificate `cert:comm` ("(b) Commutant bound"); [X] =
// docs/implementation/hecke_extensions_implementation_plan.md v1.5.1,
// Sec. 2.2 Stage 4(b) (the v1.2 rework, [XA B2]), Sec. 1.3
// `CommutantCertificate` and Sec. 1.3.1 R3.3 item 3 (`CommutantState`);
// [C] = docs/latex/gl2_contract.tex v1.3.3.
//
// [C] is where the object being certified is defined: `obl:algebra` /
// `thm:relations` make the assignment of double cosets to operators a RING
// homomorphism, so the T_N commute and \Talg_S := F_ell[T_p : p in S] is a
// COMMUTATIVE subalgebra of End(H^q); [C] then delegates the local pieces
// \Talg_\midl to [E] (`cl:pd`(3) is the one contract clause that names them).
// Route (b) is the second of [E]'s three routes to "A_S = \Talg_\midl":
//
//   C := End_{A_S}(V) = { X in End(V) : X t_p = t_p X for every p in S }.
//   Always A_S subset \Talg_\midl subset C, and \Talg_\midl is commutative.
//   If dim C == dim A_S then A_S = \Talg_\midl -- NO further T_q can enlarge
//   the algebra at m, which is exactly what certificate (a) gets from
//   Nakayama when V happens to be cyclic.
//
// THE SOCLE BOUND ([X] Sec. 2.2 Stage 4(b) step 1; derived for the v1.2
// rework, re-derived independently by HeckeExtAlgebraValidation).  With
//
//   t := dim_F(V / m_S V)   (= LocalAlgebra::min_generators), and
//   s := dim_F(soc V),      soc V = { v : m_S v = 0 } = INTERSECT_p ker n_p,
//
// both in UNITS OF f = [K : F_ell] (F = K = A_S/m_S), every F_ell-linear map
// V -> V/m_S V -> soc V -> V whose middle map is K-linear is A_S-linear (both
// ends are A_S/m_S-modules), so C contains Hom_K(top V, soc V), of
// F_ell-dimension f*t*s; and A_S INTERSECT Hom_K(top, soc) is contained in
// { a : a m_S V = 0 } = soc A_S (V is faithful over A_S).  Hence
//
//   dim C  >=  dim A_S + f*t*s - dim_{F_ell} soc A_S,          (the bound)
//   dim C  == dim A_S  requires  f*t*s <= dim_{F_ell} soc A_S. (the gate)
//
// Consequences, all pinned by the validation:
//   * cyclic V (t = 1, V isomorphic to A_S): the bound is an EQUALITY and (a)
//     already certifies -- (b) is `not_needed`;
//   * non-cyclic V (t >= 2): the gate forces dim_{F_ell} soc A_S >= 2f, i.e.
//     A_S NOT Gorenstein.  So (b) is never attempted on a field, on any
//     F[eps]/eps^2, on any Gorenstein A_S -- in particular never at
//     (n, ell) = (10, 691), where t = s = 2, f = 1, dim soc A_S = 1 give the
//     bound dim C >= 2 + 4 - 1 = 5, the exact value of [XA B1];
//   * (b) can succeed WITHOUT (a) only on co-cyclic-like modules over
//     non-Gorenstein A_S, e.g. V = A_S^dual for A_S = F[x,y]/(x,y)^2
//     (t = 2, s = 1, dim soc A_S = 2, End_{A_S}(A_S^dual) = A_S).
// The outcome `excluded_by_socle_bound` is therefore a PROOF that (b) fails,
// recorded with its four numbers, and not an "unknown".
//
// THE STRUCTURED SYSTEM ([X] Sec. 2.2 Stage 4(b) step 2).  The naive route --
// the kernel of X |-> (X t_p - t_p X)_p on D^2 unknowns, O(k D^6) -- is the
// VALIDATION ORACLE ONLY (`NaiveCommutantDim` below, for D <= 24).  The
// shipped route fixes one CYCLIC-MOST element a of A_S, taken from the fixed
// candidate list "the RREF basis elements b_1 .. b_{dim A_S}, then the prefix
// sums b_1+b_2, b_1+b_2+b_3, ..." as the FIRST candidate whose minimal
// polynomial on V has maximal degree (the choice affects the cost, never the
// verdict), puts a in rational canonical form
//
//   R^{-1} a R = (+)_i companion(g_i),   g_i = pi_a^{s_i},  s_1 >= s_2 >= ...
//
// (`hecke::PrimaryRCFBasis`; a is pi_a-primary because A_S is local, so a's
// semisimple part is a polynomial in w_hat and its minimal polynomial is
// irreducible), and solves X t_p = t_p X INSIDE the explicit basis of
// End_{F_ell[a]}(V), which has
//
//   u = SUM_{i,i'} deg gcd(g_i, g_{i'}) = deg(pi_a) * SUM_{i,i'} min(s_i,s_{i'})
//
// elements instead of D^2.  C is contained in End_{F_ell[a]}(V) because
// a lies in A_S, so dim C is the nullity of that u-column system.
//
// THE BUDGET AND `CommutantState` ([X] Sec. 1.3.1 R3.3 item 3).  The cost
// model, in modmul units, is
//
//   cost_formation = 2 * k * u * D^3     (the k*u commutators E_c t_p - t_p E_c)
//   cost_row       = u * u               (reducing one row of width u)
//   total          = cost_formation + (k * D^2) * cost_row
//
// whose second term is [X]'s `k * D^2 * u^2`; the first is added because the
// implementation PAYS it and R3.1 item 4 wants `modmul_spent` to be spent
// work (see docs/implementation/slot2/WP-5_report.md, deviation 1).  The
// system's k*D^2 rows are formed in the FIXED order "operators ascending p,
// then the commutator entry (i, j) row-major" and streamed into an
// incremental canonical RREF, so
//
//   rows_done(budget) = 0                                  if cost_formation > budget
//                     = min(total_rows, (budget - cost_formation) / cost_row)
//
// is a PURE FUNCTION OF THE BUDGET, the reduction of a prefix is the (unique)
// RREF of that prefix, and resume-invariance is immediate: run(b1) then
// resume(b2) gives the same state, certificate and outcome, entrywise, as a
// fresh run(b2) -- modulo `budget_history`, which R3.1 item 4 requires to
// differ.  Over budget the outcome is `over_budget` with the partial system
// kept; it is NEVER a silent fall-back to the naive route.
//
// PLACEMENT.  HeckeExt/, MatrixT<uint32_t> throughout ([X] Sec. 1.1,
// "D <= ~200"), single-threaded, no hash container.  [X] Sec. 1.1 names this
// file `HeckeGenerationCertificates` and gives it "cyclicity, commutant
// (structured + naive oracle), Sturm sweep to B'"; cyclicity shipped with
// WP-4 inside HeckeLocalAlgebra.h (where its data lives) and the Sturm SWEEP
// is E1b -- what is here is (b) plus the route-combining verdict below.
// ===========================================================================

#include "HeckeLocalAlgebra.h"
#include "HeckeSturmBound.h"   // SturmFinal: the R3.2 tri-state, not duplicated
// (slot-4 WP-1 / E2a) `commutant_budget` as a mathematical setting: its ONE
// default, hecke::kDefaultCommutantBudget, lives in Hecke/HeckeSettings.h --
// `Hecke/` may not include `HeckeExt/` (guard rule 1), so the constant cannot
// live here and be read there.
#include "../Hecke/HeckeSettings.h"

#include <cstdint>
#include <string>
#include <vector>

namespace hecke_ext {

// ---------------------------------------------------------------------------
// Outcomes ([X] Sec. 1.2 `GenerationCertificate`, the per-route column for b).
// ---------------------------------------------------------------------------

enum class CommutantOutcome : uint32_t {
    InProgress           = 0,  // constructed but not yet run
    NotNeeded            = 1,  // (a) certified: V is cyclic, C = A_S by Nakayama
    ExcludedBySocleBound = 2,  // f*t*s > dim soc A_S: a PROOF that (b) fails
    OverBudget           = 3,  // the budget stopped the system; state is resumable
    Passed               = 4,  // dim C == dim A_S
    Failed               = 5   // dim C  > dim A_S, computed
};
const char* ToString(CommutantOutcome outcome);

// ---------------------------------------------------------------------------
// The resumable state ([X] Sec. 1.3.1 R3.3 item 3, member by member).
// ---------------------------------------------------------------------------

struct CommutantState {
    uint32_t ell = 0;
    uint32_t D   = 0;
    uint32_t f   = 0;
    uint32_t k   = 0;          // |S|

    // The budget in force and its history (R3.1 item 4: one entry per
    // run()/resume(), the LAST is the setting in force).  Units: modmul.
    uint64_t              budget = 0;
    std::vector<uint64_t> budget_history;

    // --- the socle-bound numbers ([X] Sec. 1.3, "the socle-bound numbers
    // (t, s, f, dim_F soc A_S)"), t and s in UNITS OF f. ------------------
    uint32_t dim_A       = 0;   // dim_{F_ell} A_S
    uint32_t t_top       = 0;   // dim_F(V / m_S V)
    uint32_t s_soc       = 0;   // dim_F(soc V)
    uint32_t dim_soc_A   = 0;   // dim_{F_ell} soc A_S
    uint32_t socle_bound = 0;   // dim_A + f*t_top*s_soc - dim_soc_A
    bool     admits      = false;  // f*t_top*s_soc <= dim_soc_A

    // --- the chosen cyclic-most element and its RCF (empty unless the route
    // was prepared, i.e. unless it got past the two gates). ---------------
    bool                  prepared = false;
    uint32_t              candidate_index = 0;  // into the fixed candidate list
    ExtMatrix             a;                    // the candidate itself
    hecke::FpPoly         pi_a;                 // monic irreducible; g_i = pi_a^{s_i}
    std::vector<uint32_t> inv_sizes;            // s_1 >= s_2 >= ..., aligned with R's blocks
    ExtMatrix             R;                    // RCF change of basis, D x D, invertible
    uint32_t              u = 0;                // dim_{F_ell} End_{F_ell[a]}(V)

    // --- progress ---------------------------------------------------------
    uint64_t total_rows  = 0;   // k * D * D
    uint64_t rows_done   = 0;
    uint64_t modmul_spent = 0;  // a pure function of rows_done (see the header note)

    // The RREF of the first `rows_done` rows, width u, in the canonical
    // `hecke::RrefRowSpan` order -- so it is a function of `rows_done` alone.
    std::vector<std::vector<uint32_t>> system_rref;
    std::vector<uint32_t>              system_pivots;
    uint32_t                           rank = 0;   // == system_rref.size()

    CommutantOutcome outcome = CommutantOutcome::InProgress;
    uint32_t         dim_C   = 0;   // meaningful iff outcome is Passed / Failed

    // Entrywise equality.  `ignore_budget_history` is what resume-invariance
    // needs: R3.1 item 3 demands the same state entrywise, R3.1 item 4 demands
    // that every run/resume append its budget, and the two can only be
    // reconciled by excluding the history from the comparison (the
    // HeckeSturmBound precedent, WP-3).
    bool equals(const CommutantState& other, bool ignore_budget_history) const;

    // Fixed key order, no pointers, no hash container (R3.1 item 1).
    std::string to_json() const;
};

// ---------------------------------------------------------------------------
// The certificate ([X] Sec. 1.3 `CommutantCertificate`).
// ---------------------------------------------------------------------------
//
// verify() RE-DERIVES dim C -- deterministically and WITHOUT the search:
//   * the embedded certificate (a) re-derives A_S, the n_p, m_S and the
//     Nakayama quotient from the stored t_p (clause 4);
//   * soc V, soc A_S and the socle bound are recomputed from V, m_S and A_S
//     ([XA B2]) (clauses 5-8);
//   * the candidate list is rebuilt and `a` and its index re-chosen (10);
//   * R^{-1} a R is compared ENTRYWISE with the companion blocks of (g_i) (12);
//   * the u basis matrices of End_{F_ell[a]}(V) are REBUILT from (g_i) and R
//     by the fixed formula and checked to commute with a and to be
//     independent (13);
//   * the system X t_p = t_p X over that basis is re-formed and row-reduced,
//     and its nullity IS dim C (15-17).
// A truncated (g_i) list, a decremented rank or a perturbed R therefore fails:
// nothing the certificate concludes from is read.
struct CommutantCertificate {
    // Route (a)'s data, which is also (b)'s input: it carries ell, D, f, the
    // primes, t_p, the characters, pi_w, w_hat, the A_S basis and the m_S
    // basis, and its own verify() re-derives every one of them.
    CyclicityCertificate cyclicity;

    uint32_t ell = 0, D = 0, f = 0;

    // The socle-bound numbers, t and s in units of f.
    uint32_t dim_A = 0, t_top = 0, s_soc = 0, dim_soc_A = 0, socle_bound = 0;
    bool     admits = false;

    // True when the caller forced the structured system past the two gates
    // (the validation's "structured == naive wherever both run" arm).  The
    // production route never sets it, and verify() enforces the gates when it
    // is false.
    bool forced = false;

    CommutantOutcome outcome = CommutantOutcome::InProgress;

    // The structured system (meaningful iff state.prepared).
    uint32_t              candidate_index = 0;
    ExtMatrix             a;
    hecke::FpPoly         pi_a;
    std::vector<uint32_t> inv_sizes;
    ExtMatrix             R;
    uint32_t              u = 0, rank = 0, dim_C = 0;

    // The R3 state; `resumable_json()` is the R3.1 record object.
    CommutantState state;
    // Where the state goes when its JSON exceeds 64 KiB (R3.1 item 6).  Empty
    // (the default) means "inline it anyway and flag it": only the record
    // layer (E2) knows the record stem, so only it can name the sidecar.
    std::string state_sidecar_stem;

    // 0 on success, else the index of the first failing clause.
    uint32_t verify() const;
    static const char* verify_message(uint32_t code);

    std::string to_json() const;
    std::string resumable_json() const;
};

// ---------------------------------------------------------------------------
// The computation.
// ---------------------------------------------------------------------------

class CommutantRoute {
public:
    // [X] Sec. 2.2 Stage 4(b) step 2: "attempted only if k*D^2*u^2 <=
    // kCommutantBudget = 2*10^10 modmul (a named constant; for k = 5:
    // u <= 371 at D = 170, u <= 988 at D = 64, the naive u = D^2 at D <= 24)".
    //
    // SINCE SLOT-4 WP-1 (E2a) THIS IS AN ALIAS, NOT A SECOND LITERAL: the one
    // definition is hecke::kDefaultCommutantBudget (Hecke/HeckeSettings.h),
    // which is also the DEFAULT of the `commutant_budget` mathematical setting
    // ([OD] slot-3 confirmation 3).  `kCommutantBudget` is the default's name,
    // the setting is what a run may change, and `budget = 0` below means "the
    // setting in force".
    static constexpr uint64_t kCommutantBudget = hecke::kDefaultCommutantBudget;

    // `L` must be a LocalAlgebra with ok == true; everything it needs is
    // copied, so `L` need not outlive the route (R3 resumes from state()).
    // `force` runs the structured system even where a gate would stop it --
    // a VALIDATION knob ([X] Sec. 5's "structured == naive wherever both
    // run"), recorded in the certificate as `forced` and never set by a
    // production caller.
    // `budget == 0` means THE `commutant_budget` SETTING IN FORCE
    // (hecke::math_settings().commutant_budget.budget), which at its default
    // is kCommutantBudget -- so a default-constructed route is unchanged.  A
    // positive `budget` is taken verbatim, which is what every R3 resume arm
    // and every validation that pins a budget needs.
    explicit CommutantRoute(const LocalAlgebra& L,
                            uint64_t budget = 0,
                            bool force = false);

    // Runs from scratch under the constructed budget.  Exactly once; use
    // resume() afterwards.
    CommutantOutcome run();

    // Continues from state(), never from scratch (R3.1 item 2):
    //   * a budget below the one already spent is REFUSED with the named
    //     error "resume: budget below reached state";
    //   * on a complete state (anything but OverBudget) it is a STRICT no-op --
    //     neither the budget nor the history moves, so the certificate is
    //     literally unchanged;
    //   * otherwise the remaining rows are formed and reduced from the row
    //     reached.
    CommutantOutcome resume(uint64_t new_budget);

    const CommutantState& state() const { return st_; }
    CommutantOutcome outcome() const { return st_.outcome; }
    uint32_t dim_C() const { return st_.dim_C; }
    uint32_t socle_bound() const { return st_.socle_bound; }
    bool admits() const { return st_.admits; }
    uint64_t budget() const { return st_.budget; }

    CommutantCertificate certificate() const;

private:
    void gates_();       // the socle-bound numbers and the two gates
    void prepare_();     // the candidate, the RCF and the End_{F[a]}(V) basis
    void advance_();     // the budgeted row loop
    uint64_t rows_for_budget_(uint64_t budget) const;
    uint64_t charge_(uint64_t rows) const;

    // Copied inputs (D <= ~200, so this is a few hundred KiB at worst).
    uint32_t ell_ = 0, D_ = 0, f_ = 0;
    std::vector<ExtMatrix> t_, n_, a_basis_;
    CyclicityCertificate   cyc_;
    bool force_ = false;

    // Derived at prepare_() time; never persisted (deterministic).
    std::vector<ExtMatrix> end_basis_;              // the u elements
    std::vector<std::vector<ExtMatrix>> comm_;      // comm_[p][c]
    bool comm_formed_ = false;
    uint64_t cost_formation_ = 0, cost_row_ = 0;

    CommutantState st_;
};

// The NAIVE commutant dimension: the nullity of X |-> (X t_p - t_p X)_p on
// D^2 unknowns, O(k D^6).  [X] Sec. 2.2 Stage 4(b): "the validation oracle
// only, for D <= 24" -- never a fall-back for the structured route.
uint32_t NaiveCommutantDim(const std::vector<ExtMatrix>& t, uint32_t D, uint32_t ell);

// ---------------------------------------------------------------------------
// The route-combining verdict ([X] Sec. 1.2 `GenerationCertificate`,
// Sec. 1.3.1 R3.2).
// ---------------------------------------------------------------------------

enum class GenerationKind : uint32_t {
    Unavailable = 0, Cyclic = 1, Commutant = 2, Sturm = 3
};
const char* ToString(GenerationKind kind);

// Route (c)'s verdict, supplied by the caller: the Sturm SWEEP is E1b's, and
// this unit must not pretend to run it.
enum class SturmRouteStatus : uint32_t {
    NotNeeded   = 0,   // (a) or (b) already certified
    Passed      = 1,   // the sweep certified A_S = T_m on its proof range
    Failed      = 2,   // a swept T_q was not in A_S
    Unavailable = 3    // no B'(k, ell) below the cap, or m is off the proof range
};

struct GenerationVerdict {
    GenerationKind kind = GenerationKind::Unavailable;
    SturmFinal     final_value = SturmFinal::False;  // the R3.2 tri-state
    std::string    reason;   // "" iff final_value == True; else from R3.2's list
};

// The map (a, b, c) -> (kind, final, reason) of [X] Sec. 1.3.1 R3.2, in the
// DEFAULT configuration (`final_mode = closure`, `sturm.unavailable =
// nonfinal`, slot-1 confirmation 8).  OQ8 / [OD]: an ideal that cannot reach
// closure is EMITTED with `final = false`, its reason and -- where a budget
// was involved -- its `resumable` object; it is never withheld.  The reasons
// used here are R3.2's fixed list: `closure_unreachable` (the socle-bound
// proof, or a computed dim C > dim A_S, together with (a) failing),
// `budget_exhausted:commutant`, `sturm_unavailable`,
// `certificate_failed:sturm`.  The `heuristic` value is NOT produced here: it
// belongs to the user-selected modes `final_mode = sturm_heuristic` /
// `sturm.unavailable = heuristic`, which the record layer (E2) owns.
GenerationVerdict generation_verdict(bool cyclic_passed,
                                     CommutantOutcome b,
                                     SturmRouteStatus c);

} // namespace hecke_ext

#endif // HECKE_EXT_GENERATION_CERTIFICATES_H
