// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_IOTA_TAU_SYMMETRY_VALIDATION_H
#define HECKE_IOTA_TAU_SYMMETRY_VALIDATION_H

#include <cstdint>

// Obligation obl:iota(i) and obl:iota(iv) of docs/latex/gl2_contract.tex
// chapter 9 (the involution iota; v0.12.0) -- the ONLY two clauses of that
// obligation that can be discharged without an implementation of iota^* as a
// cochain map, because both are predicates on finite exact integer data.
//
// The element is iota = diag(-1,1) in GL_2(Z) (def:iota(i)), acting on H by
// z -> -conj(z), with conjugation c_iota(gamma) = iota*gamma*iota.
//
// ---------------------------------------------------------------------------
// WHAT THIS FILE DECIDES
// ---------------------------------------------------------------------------
// (A) [GATING, aborts on failure -- obl:iota(i)] That the seven-vertex model
//     of def:model7 is iota-stable in the sense of (I1), that the shipped
//     exact vertex coordinates of hecke::SL2ZTessellationTables realise
//     z -> -conj(z) as the vertex permutation 0<->3, 1<->2, 4<->5, 6 fixed of
//     lem:iotamodel -- including the hypothesis Re v_1 = -Re v_2 that lemma
//     carries -- and that the orientation sign of every one of the 23 stored
//     simplices, recomputed here as the parity of the permutation sorting the
//     image vertex list back into increasing order, agrees ENTRY FOR ENTRY
//     with lem:iotamodel's printed sign table (six minus signs). All over Z.
//
// (B) [obl:iota(iv); GATING on the canonical tau since 2026-09-16, RECORDED
//     on the two alternate taus] Whether a g-twisted chain datum tau, as
//     shipped by hecke::HeckeTauTable under a given TauVertexPolicy, satisfies
//
//         tau^iota := iota_X o tau o iota_X  ==  tau                    (*)
//
//     on EVERY stored table entry, compared term for term over Z. (*) is the
//     hypothesis of prop:iotacommutes(3b), and by rmk:iotatausym it is the
//     ONLY available route to [iota^*, T_g] = 0 at ell in {2,3}, where
//     thm:welldef(b) -- and with it prop:iotacommutes(3c) -- is unavailable.
//     Per obl:iota(iv) the verdict is RECORDED and a true value licenses
//     nothing beyond (3b). Since the owner's re-baseline of 2026-09-16 the
//     canonical tau (GeometricNearest) is CONSTRUCTED to satisfy (*) --
//     iota_K-equivariant tie-break in alg:tau step (1b), tree-routed path
//     templates in step 2 -- so on that policy a single differing slot on any
//     table of the grid is a FAILURE. The legacy policy
//     (GeometricNearestLegacy: the pre-re-baseline tau) and FixedV6 are run
//     alongside and their verdicts are recorded only.
//
// ---------------------------------------------------------------------------
// HOW iota_X IS REALISED ON THE STORED SIMPLICES (def:iota(iii))
// ---------------------------------------------------------------------------
// A simplex of X = K^u is stored as a pair (gamma, sigma) denoting
// gamma . i(sigma), sigma an index into K's q-simplex list in increasing
// vertex order. From iota(gamma z) = c_iota(gamma)(iota z) (def:iota(i)) and
// iota_X(i(sigma)) = iota_K(sigma) = eps_sigma * i(|iota_K sigma|),
//
//     iota_X (gamma . i(sigma)) = eps_sigma * ( c_iota(gamma) . i(|iota_K sigma|) ),
//
// with eps_sigma the lem:iotamodel sign recomputed in (A). Nothing here is
// chosen: the vertex permutation is forced by the shipped coordinates, and
// the signs are forced by the increasing-vertex-order storage rule
// (rmk:iotasigns). This is checked, in (A), to be consistent -- the two
// representations (gamma, sigma) and (gamma*t, sigma_hat) of one oriented
// K^u-simplex are folded through the PUBLIC
// SL2ZTessellationTables::canonical_key, so the comparison in (B) is a
// comparison of chains, not of representations.
//
// tau on the translate c_iota(s_i) . i(sigma') is evaluated through the (T1)
// rule of [hecke_action.tex, def:tau] re-derived here from the contract
// (NEVER through HeckeTauTable's private eval_tau_on_translate): write
// c_iota(s_i) = h * s_j with j = class_index_of(c_iota(s_i)) and
// h = c_iota(s_i) * s_j^{-1} in Gamma_0(N) (asserted), then
// tau(h . x) = (g^{-1} h g) . tau(x) with the exact integral conjugate
// g^{-1} h g = [[h.a, N*h.b],[h.c/N, h.d]].
//
// Because tau^iota again satisfies (T1) for the same Gamma'' = Gamma_0(N)
// (c_iota(g) = g and c_iota(Gamma'') = Gamma''; prop:iotacommutes(3a)) and
// the stored slots (i, sigma) exhaust the Gamma''-orbits of X-simplices,
// equality on every stored entry is equivalent to (*) as maps on C_*(X).
//
// ---------------------------------------------------------------------------
// WHAT THE RUNS FOUND
// ---------------------------------------------------------------------------
// 2026-09-16, before the re-baseline (now the GeometricNearestLegacy arm,
// reproduced by every run): (A) PASSES; (B) is FALSE for that tau, 3600 of
// 10764 binding slots (q=0 456/3276, q=1 2340/5148, q=2 804/2340), already
// in degree 0, where lem:iotamodel's orientation signs are all +1 and no sign
// convention can be at fault. Minimal witness, at p = 2, r = 1, ell = 2,
// n = 0, slot (i = 1, q = 0, sigma = {v0}):
//
//     tau(i(v0))                  = (-T) . i(v1)
//     (iota_X o tau o iota_X)(v0) = (-T) . i(v2)
//
// Mechanism: g^{-1} s_1 . pos(v0) reduces to 4i, ON THE MIRROR Re z = 0,
// equidistant from v1 and v2, which iota_K exchanges; the ascending-index
// tie-break picks v1. No choice WITHIN a swapped pair can commute with
// iota_K, so the repair is forced off the tied set: the canonical policy
// sends an iota_K-stable tied set to v6, the unique iota_K-fixed vertex.
// That alone left q=1 1848/5148 and q=2 1068/2340 differing: alg:tau step 2
// (BFS templates in ascending neighbour order; sign -1 edges built in the
// reverse direction) was a SECOND obstruction, removed by routing every edge
// chain through the Bass-Serre tree (translates of {v4,v6}, {v5,v6}), where
// a 1-chain with prescribed boundary is unique.
//
// 2026-09-16, after the re-baseline: (B) holds on the canonical tau on ALL
// 288/3 = 96 tables of the grid, 0 differing slots in every degree, every
// comparison exact over Z (no table used the mod-ell fill). The legacy arm
// reproduces the 3600. Every table of every arm still passes the mandatory
// certification (T1)-(T3) at construction and (C) below.
//
// Returns the number of GATING (A)/(C) discrepancies plus, on the canonical
// tau only, (B) discrepancies; 0 on success, per the ValidationFn convention.
// The (B) verdicts of the alternate taus are printed, never returned.
uint32_t HeckeIotaTauSymmetryValidation();

#endif // HECKE_IOTA_TAU_SYMMETRY_VALIDATION_H
