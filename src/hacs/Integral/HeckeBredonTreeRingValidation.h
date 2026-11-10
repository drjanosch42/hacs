// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_BREDON_TREE_RING_VALIDATION_H
#define HECKE_BREDON_TREE_RING_VALIDATION_H

#include <cstdint>

// hecke-extensions slot 9, WP-3 (decision 4 (A); plan B-1, M-6): E8 at
// ell in {2, 3} -- the cohomology of the 7-vertex model over Z/ell^k, against
// the oracle O7.
//
// THE OBJECT. Under coefficient_ring = Z/ell^k at ell in {2, 3},
// EquivariantCohomologyOnK computes the BREDON object over Z/ell^k ([C]
// def:ringsemantics(2)), by the flat-coordinate module route (C^q_eq is not
// free there), reporting lengths and invariant factors per degree only.
// [C] prop:bredon(2)-(3): it is the tree complex
//     V_R^S (+) V_R^U -> V_R^{-I},  (a, b) |-> b - a,
// over the ring in force, so H^0 = V^S cap V^U, H^1 = V^{-I} / (V^S + V^U),
// H^2 = 0.
//
// THE ORACLE, O7 (frozen). docs/implementation/integral_line_probe.py's
// model_R(n, ell, k) (slot-9 WP-3, additive): the tree model over Z/ell^k
// computed over Z with lattices containing ell^k Z^m and an integer Smith
// form -- no Howell form, no modular elimination, nothing shared with the C++
// code under test.  Its table is frozen below with its provenance (the probe's
// git blob id and the generating command), and printed by the validation.
//
// ARMS.
//  (P) provenance printed;
//  (R) rho_n over Z/ell^k at ell in {2, 3} ([IB] K8): RhoOfExactRing ==
//      ComputeSymmetricPowerMatrixRingOracle (the division-free oracle)
//      entrywise, on every g the tested cells touch (the seeds' T, S, ST, TS,
//      -I, and the tree's S, U), and every relation matrix of
//      GenerateSL2ZRelationsRing equals the oracle at its g;
//  (A) THE ORACLE ARM (gating): at ell in {2, 3}, k in {2, 3, 4}, every even
//      n <= 20 and every odd n <= 9 (96 cells; the -I / odd-n cells named),
//      the pipeline's H_lengths() and H_invariant_factors(q), q = 0, 1, 2,
//      equal the frozen O7 table, and the object took the flat-coordinate
//      route;
//  (T) a second, IN-PROCESS tree model over R on the same cells -- a
//      CONSISTENCY CHECK, NOT AN ORACLE ([IB] K6): it shares the Howell /
//      Smith code (DenseElim, ENGINE_API.md Sec. 13.6) with the pipeline; its
//      rho_n is the ring route's own (RhoOfExactRing, i.e.
//      ComputeSymmetricPowerMatrixRing, checked in (R));
//  (K1) O7's own control at k = 1: the probe's model_R(n, ell, 1) lengths
//      equal the shipped F_ell pipeline's dim H^q (32 cells);
//  (N) O7's control at ell = 5: the probe's table equals slot 8's ell >= 5
//      module route (five cells, k in {2, 3}) -- the probe is not tuned to
//      the flat-coordinate route;
//  (G) the dispatch guards: flat_module_degree() refuses at ell >= 5, the
//      field accessors refuse at ell in {2, 3}.
// The refusal arms at ell in {2, 3} (K-mode accessors, HeckeOperatorOnK,
// HeckeReduction, the console) are in HeckeReductionValidation's arm (X).
//
// Registered `standard` (a NEW name, M-6: HeckeBredonTreeValidation is
// untouched).  Returns the number of failing checks (0 = pass).
uint32_t HeckeBredonTreeRingValidation();

#endif // HECKE_BREDON_TREE_RING_VALIDATION_H
