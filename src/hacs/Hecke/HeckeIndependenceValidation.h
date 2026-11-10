// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_INDEPENDENCE_VALIDATION_H
#define HECKE_INDEPENDENCE_VALIDATION_H

#include <cstdint>

// Stage S8V -- INDEPENDENT agreement & well-definedness/independence
// validation for HeckeOperatorOnK (Stage S7's fused Theta_g pipeline).
// Written by the S8V validation agent directly from
// docs/latex/hecke_action.tex ([C]) -- Theorem~\ref{thm:welldef}
// (independence of the coset representatives s_i, of tau, and of the
// double-coset representative g) and Obligations obl:welldef / obl:indep --
// and from docs/implementation/hecke_action_implementation_plan.md's S8V
// stage description (oracle O12). Only the PUBLIC headers
// HeckeOperatorOnK.h / HeckeCosetTables.h / HeckeTauTable.h / ExactInt.h /
// EquivariantCohomologyOnK.h / LocalSystemSL2Z.h were consulted for API
// signatures; Stage S8's own BarCocycleH1.cpp (an independent exact
// left-coset permutation/kappa machine) was read for context only, per the
// task, and is NOT called by this file -- every check below is re-derived
// and re-implemented from scratch against the public API.
//
// ===========================================================================
// ORACLE O12 -- what is asserted, EXACTLY (no tolerances anywhere, I9), and
// how each item is realized through the public API:
//
// (a) Route bit-identity.
//     HeckeOperatorOnK::compute() already asserts, INTERNALLY, that the
//     fused route (omega-row bracketing) and "route 2" (direct evaluation of
//     [C, eq:heckechain]) are bit-identical on every matrix entry, of every
//     degree, throwing std::runtime_error otherwise ([C, obl:agreement]'s
//     first two routes) -- so simply constructing/computing the operator
//     (done for every setting below) already exercises half of (a). This
//     file ADDS the other half explicitly: apply_Theta(q, e_b), evaluated on
//     every standard basis column e_b of C^q(K;V), must reproduce column b
//     of get_Theta_flat_matrices()[q] exactly (apply_Theta is documented as
//     literally Theta_flat[q] * c_flat; this is a live re-exercise of that
//     contract through the public evaluation API, not a tautology against
//     private state), for every q. Also spot-checks the JSON certificate's
//     "routes_bit_identical":true flag.
//
// (b) Custom coset representatives ([C, thm:welldef](a)).
//     Builds a SECOND HeckeCosetTables via the custom-reps constructor with
//     s_i' = h_i * s_i for pseudorandom (fixed-seed xorshift64, deterministic
//     across runs and across HECKE_PURE_INTEGER settings) h_i in Gamma_0(p):
//     h_i is a short word (2-4 letters) in {L = [[1,0],[p,1]], T = [[1,1],[0,1]]}
//     -- both generate elements of Gamma_0(p) (L's lower-left is p == 0 mod p
//     by construction; T's lower-left is 0 trivially), and Gamma_0(p) is a
//     group, so any product of these two matrices lies in Gamma_0(p); using
//     only positive powers keeps the construction simple with no inverse
//     bookkeeping and still gives a rich, nontrivial set of h_i's. The
//     custom-reps constructor's own consistency check (class_index_of(s_i')
//     == class_index_of(s_i)) is satisfied automatically since h_i in
//     Gamma_0(N) never changes the Gamma_0(N)-coset. Because the tau-table
//     constructor takes the HeckeCosetTables it must be built against, the
//     WHOLE chain (cosets -> tau -> operator) is rebuilt consistently against
//     the custom cosets, exactly as the task instructs.
//
//     A SUBTLETY, discovered empirically and confirmed mathematically (full
//     derivation in HeckeIndependenceValidation.cpp's CheckB_CustomCosetReps):
//     [C, thm:welldef](a)'s proof holds THE SAME tau fixed and only relabels
//     s_i; HeckeTauTable exposes no "relabel an existing tau" hook, so
//     rebuilding the whole chain from a genuinely different HeckeCosetTables
//     necessarily also rebuilds tau via its own independent (certified, but
//     only homotopy-unique per [C, thm:welldef](b)/lem:tauexists) geometric
//     construction -- exercising (a) and (b) TOGETHER. This file therefore
//     HARD-asserts exactly what the contract backs for that combination:
//     (i) a "custom" rebuild with h_i = I for every i (same representatives,
//     exercising only the rebuild machinery) reproduces Theta_flat
//     bit-for-bit -- a real bug here would be a genuine defect; (ii) with
//     genuinely nontrivial pseudorandom h_i, the COHOMOLOGY-level A^{(q)}
//     matrices (get_H_matrices()) are bit-identical to the canonical build's,
//     every q -- verified empirically at all 4 grid settings, matching the
//     signature predicted by (a)+(b) combining to a cohomology-only
//     guarantee. Whether the finer cochain-level Theta_flat identity ALSO
//     happens to hold for the nontrivial h_i is RECORDED (printed) but not
//     hard-asserted (observed to hold at 1 of the 4 fixed-seed draws here and
//     to legitimately differ, via a different-but-valid tau, at the other 3
//     -- not a code defect: see the cpp file for the full argument, including
//     why the model's elliptic vertices make this expected).
//
// (c) tau-vertex policy ([C, thm:welldef](b), ell >= 5).
//     Builds a second and a third HeckeTauTable against the SAME (canonical)
//     cosets with TauVertexPolicy::FixedV6 and with
//     TauVertexPolicy::GeometricNearestLegacy (the pre-2026-09-16 canonical
//     tau: ascending-index tie-break + BFS path templates) instead of the
//     binding default GeometricNearest, and a HeckeOperatorOnK from each. Every grid
//     setting below has ell in {13, 691, 101}, all >= 5, so the check is
//     asserted unconditionally (no gating needed). Assertion: get_H_matrices()
//     (A^{(q)}, every q) are EQUAL between the binding policy and each
//     alternate one. Per the
//     contract (only the induced COHOMOLOGY map is asserted tau-independent;
//     the cochain-level Theta_g need not agree), this file does NOT compare
//     get_Theta_flat_matrices() between the two policies.
//
// (d) Double-coset representative g ([C, thm:welldef](c)).
//     READ THIS CAREFULLY -- (d) is NOT fully reachable through the public
//     API, and this file says so explicitly rather than faking it:
//     HeckeCosetTables hardcodes g = diag(1, N) and Gamma'' = Gamma_0(N)
//     (there is no constructor parameter for an alternate g or an alternate
//     finite-index subgroup); its custom-representatives hook only accepts
//     s_i' in the SAME Gamma_0(N)-class as the canonical s_i (enforced via
//     class_index_of, throws otherwise), so it can realize replacing s_i by
//     h*s_i for h in Gamma_0(N) (that is exactly check (b) above) but NOT a
//     genuinely different g or a conjugate subgroup Gamma''_1 = gamma1
//     Gamma_0(N) gamma1^{-1} for gamma1 not in Gamma_0(N) (which is what
//     [C, thm:welldef](c)'s proof actually constructs: new representatives
//     {s_i gamma1^{-1}} of the DIFFERENT coset space Gamma''_1 \ Gamma). Since
//     s_1 = I always, requiring class_index_of(gamma * s_1) ==
//     class_index_of(s_1) already forces gamma in Gamma_0(N) for ANY uniform
//     left-translate to even be accepted by the constructor -- so the
//     task's own suggested fallback ("left-multiplying all s_i by a common
//     gamma in Gamma leaves A^{(q)} unchanged") is, through this API,
//     necessarily restricted to gamma in Gamma_0(N) too. What this file
//     therefore runs, and asserts, as the explicitly-permitted WEAKER
//     reachable form: for TWO fixed, nontrivial, distinct elements gamma_A =
//     L = [[1,0],[N,1]] and gamma_B = L^2 = [[1,0],[2N,1]] of Gamma_0(N),
//     replace EVERY s_i by gamma * s_i (the SAME gamma across all i, unlike
//     (b)'s independent per-i random h_i), rebuild the whole chain, and
//     assert get_H_matrices() (A^{(q)}, every q) EQUAL to the canonical
//     build's, for BOTH gamma_A and gamma_B. Mathematically this is a
//     (degenerate, constant-h_i) special case of thm:welldef(a), not a new
//     instance of thm:welldef(c) -- it is documented as such, not oversold as
//     a genuine g-conjugation test. A repository-level TODO for anyone
//     wanting the STRONG form of (d): HeckeCosetTables/HeckeTauTable/
//     HeckeOperatorOnK would need a constructor parameterized by an
//     arbitrary Mat22 g (det > 1) and an arbitrary finite-index Gamma'' <=
//     Gamma, not just g = diag(1, p^r) / Gamma'' = Gamma_0(p^r) -- genuinely
//     new plumbing, out of scope for a validation-only stage.
//
// (e) HECKE_PURE_INTEGER=ON vs OFF.
//     This validation contains no floating point and calls no HECKE_PURE_INTEGER
//     - gated code directly; it is deterministic C++/exact-integer/mod-ell
//     arithmetic throughout (like every other Hecke validation). (e) is
//     therefore verified EXTERNALLY, at the Definition-of-Done level: build
//     and run `./cohomology_app HeckeIndependenceValidation` once under
//     `cmake -DHECKE_PURE_INTEGER=OFF .` and once under `=ON`, and diff the
//     two stdout transcripts byte-for-byte (they must be identical -- this
//     file prints only deterministic content: settings, PASS/FAIL, and, on
//     failure, the differing matrices; no addresses, timings, or other
//     nondeterminism).
//
// Settings grid (binding, from the plan's S8V stage): (p, ell, n) in
// {(2,13,10), (3,13,10), (2,691,10), (5,101,2)}. For each setting, checks
// (a)-(d) above are run (in that order); a failure in any one check is
// recorded (with full diagnostic: setting, which choice varied, and the
// differing matrices) and the battery continues to the remaining checks
// and settings (one bad case never aborts the rest, matching the codebase's
// established validation style, e.g. HeckeOracleValidation.cpp).
//
// Returns the total number of failing checks across the whole grid (0 =
// success, I9's "functions returning 0" convention). ANY nonzero return is a
// genuine well-definedness bug in the Hecke construction (the operator
// depending on a choice the contract says it must not) -- NOT a bug in S7/S8
// to be silently patched by this file.
// ===========================================================================
uint32_t HeckeIndependenceValidation();

#endif // HECKE_INDEPENDENCE_VALIDATION_H
