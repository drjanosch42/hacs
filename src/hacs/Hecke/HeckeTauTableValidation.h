// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_TAU_TABLE_VALIDATION_H
#define HECKE_TAU_TABLE_VALIDATION_H

#include <cstdint>

// Stage S4V -- INDEPENDENT validation of Stage S4 (HeckeTauTable), written
// from the mathematical spec of
// docs/implementation/hecke_action_implementation_plan.md ("### S4V --
// tau-table certification validation") and the contract
// docs/latex/hecke_action.tex ([C, def:tau] (T1)-(T3), [C, obl:tau],
// [C, obl:backend], [C, eq:xiexplicit]) WITHOUT calling HeckeTauTable's own
// certify() and WITHOUT reusing its private eval_tau_on_translate() /
// boundary_of_terms() helpers (those are inaccessible from outside the class
// in any case; this file re-derives the boundary operator and the (T1)
// translation rule from the contract's definitions, from scratch, using
// only HeckeTauTable's PUBLIC accessors tau()/xi_terms()/certified_bound_B()/
// used_mod_ell_solve(), the S2 SL2ZTessellationTables::canonical_key public
// API, the S3 HeckeCosetTables public accessors (g/s/delta/class_index_of/
// perm/cocycle) and the S1 hecke::Mat22 exact-integer layer).
//
// What this file checks (S4V oracle, [C, obl:tau] / [C, obl:backend]), for
// every grid point and BOTH hecke::TauVertexPolicy variants:
//   (i)   Chain-map identity d(tau(x)) == tau(d(x)) for every stored
//         representative (i, q, sigma), q in {1,2}: the LHS is computed by
//         THIS file's own boundary operator (Sigma_j (-1)^j face_j, folded
//         through SL2ZTessellationTables::canonical_key -- the sign
//         convention (d^q f)(tau') = Sigma_j (-1)^j f(d_j tau') of
//         [C, sec:conventions]); the RHS is computed by THIS file's own
//         evaluation of tau(d x) = Sigma_j (-1)^j tau(s_i . i(d_j sigma)),
//         fetching each face term via the public tau() accessor at the SAME
//         coset index i (every K-simplex, not just an orbit representative,
//         is a directly addressable tau() slot in this implementation --
//         verified structurally below, not assumed) with a genuine
//         independent (T1) fallback translator available (and exercised by
//         the additional twisted-equivariance cross-check, see below) should
//         a face ever NOT be a directly stored slot.
//   (ii)  Augmentation epsilon(tau(v-tilde)) == 1 for every vertex slot.
//   (iii) Every stored gamma_k has det == 1; every g*gamma_k has det == N.
//   (iv)  max||gamma_k|| (recomputed from the returned chains) <=
//         certified_bound_B(), and 2*B^2 < 2^63 (int64 tier).
//   (v)   xi_terms(i,q,sigma) matches an independent recomputation of
//         rho_n((g*gamma_k) mod ell) via ComputeSymmetricPowerMatrixFM from
//         the exact stored gamma_k (coeff and sigma_idx compared too).
//   (vi)  both TauVertexPolicy::GeometricNearest and TauVertexPolicy::FixedV6
//         build (construction already enforces S4's own certify()) AND
//         independently pass (i)-(v) above.
//
// ADDITIONAL cross-check beyond the six items above (exercises (T1) itself,
// genuinely, via TWO DIFFERENT stored slots -- (i) above never needs a (T1)
// translation in this implementation because every K-simplex is a directly
// stored slot, so it alone would never catch a twisted-equivariance/
// conjugation bug): for every one of the 8 rows of the model's
// relation/generator table (transcribed here directly from
// docs/latex/hecke_action.tex, \S "The arithmetic group datum and the
// 7-vertex model", NOT read from LocalSystemSL2Z.cpp) and every coset index
// i, this file independently derives, from HeckeCosetTables::perm/cocycle
// and the exact conjugation formula g^{-1} h g = [[h.a, N*h.b],[h.c/N,h.d]]
// of (T1) ([C, def:tau]), the predicted chain tau(s_i . i(sigma_lhs)) from
// the stored chain at the OTHER slot tau(i.g_r, q, sigma_rhs), and checks it
// against the directly stored tau(i, q, sigma_lhs) exactly.
//
// Returns the number of discrepancies found (0 == success), matching the
// ValidationFn convention (invariant I9).
uint32_t HeckeTauTableValidation();

#endif // HECKE_TAU_TABLE_VALIDATION_H
