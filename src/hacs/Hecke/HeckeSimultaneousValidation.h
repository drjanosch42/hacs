// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_SIMULTANEOUS_VALIDATION_H
#define HECKE_SIMULTANEOUS_VALIDATION_H

#include <cstdint>

// Stage S11V -- INDEPENDENT validation of the simultaneous (MaxSpec(T))
// decomposition, Hecke/HeckeModuleDecomposition.h's multi-operator branch
// (ComputeMultiOp()), and its machine-readable Hecke/HeckeReport.h JSON
// export, docs/implementation/hecke_action_implementation_plan.md Sec. 3
// ("Level (c) -- block-diagonalized level") and the "S11V -- Simultaneous
// validation" stage.
//
// INDEPENDENCE STATEMENT (read before the .cpp): this file builds the family
// {T_2, T_3} and the JSON report through the PUBLIC pipeline API only
// (GenerateSL2ZRelationsWithGroupData -> EquivariantCohomologyOnK ->
// hecke::HeckeOperatorFamily -> hecke::HeckeModuleDecomposition ->
// hecke::BuildHeckeReportJson), exactly as the task's Definition of Done
// prescribes -- but every ASSERTION below is made against a string emitted
// by BuildHeckeReportJson(), re-parsed by a from-scratch JSON mini-parser
// written LOCALLY in this file (independent of both
// HeckeModuleDecomposition::to_json()'s own embedded example and the
// separate mini-parser embedded in HeckeSimultaneousSmokeValidation.cpp --
// no code or expected value is read from that file). The expected block
// structures, eigen-systems, and char/min-polys are derived from classical
// modular-forms theory (Ramanujan's Delta, level-1 Eichler-Shimura, the
// classical Ramanujan congruence, and the weight-24 LMFDB data), independent
// of anything in this repository's Hecke/ implementation:
//   * n=10 (weight 12): S_12 = C*Delta, a_2(Delta) = -24, a_3(Delta) = 252
//     (LMFDB 1.12.a.a); Eisenstein eigenvalue in weight k = n+2 is
//     1 + p^{n+1};
//   * the classical Ramanujan congruence tau(p) === 1+p^11 (mod 691), for
//     EVERY prime p (re-verified here for p=2 AND p=3 by direct integer
//     arithmetic, not merely quoted for p=2);
//   * n=22 (weight 24): dim S_24 = 2, a single Galois orbit (LMFDB 1.24.a.a,
//     Hecke field degree 2), tr(T_2|S_24) = 1080, det(T_2|S_24) = -20468736,
//     so the minimal polynomial of T_2 on S_24 is X^2-1080X-20468736
//     (irreducible over Q); Eisenstein eigenvalue 1+p^23; the discriminant
//     1080^2+4*20468736 = 24^2*144169 is re-verified by direct arithmetic,
//     so the mod-ell splitting of this quadratic is governed by the Legendre
//     symbol (144169|ell), computed HERE at runtime via
//     ModArithmetic::mod_pow (Euler's criterion), never hard-coded.
// hecke::HeckeModuleDecomposition::certify() is NEVER used as the oracle for
// any assertion below (it is not called by this file at all -- every claim
// is re-derived from the parsed JSON plus this file's own from-scratch
// F_ell[X] toolkit: a division-free cofactor-expansion characteristic
// polynomial of a JSON-reconstructed level-(b) matrix, and a from-scratch
// gcd/Frobenius irreducibility test, both independent of hecke::FpPoly /
// hecke::CharPolyBerkowitz / hecke::CertifyIrreducible).
//
// Oracles asserted (exact; no tolerance):
//   (ell,n)=(13,10):    cohomology_dim(deg 1) = 3; blocks = {dim 2 cuspidal
//     (Delta system: T_2 char (X-2)^2, T_3 char (X-5)^2 mod 13, per-op
//     partitions recorded, not chased for a further split -- isotypic
//     multiplicity 2), dim 1 Eisenstein (eigen_system {2:8, 3:10} mod 13,
//     independently re-derived from 1+2^11=2049===8 and 1+3^11=177148===10
//     mod 13)}; residue_degree=1 CERTIFIED for both blocks.
//   (ell,n)=(691,10):   a single dim 3 block; T_2 char=(X+24)^3,
//     min=(X+24)^2 (Jordan partition (2,1)), residue_degree=1 CERTIFIED;
//     the Ramanujan congruence re-verified for p=3 too
//     (1+3^11 === 252 === a_3(Delta) mod 691), confirming T_3 also collapses
//     into the single block.
//   (ell,n)=(10007,22): weight 24, Legendre(144169|10007) computed at
//     runtime and confirmed = +1; blocks = {dim 2, dim 2, dim 1}, all
//     residue_degree=1 CERTIFIED -- the Galois-conjugate weight-24 pair
//     splits into two distinct rational eigenvalue systems (multiplicity 2
//     each, holo/antiholo), plus the Eisenstein block with eigen_system
//     {2: 1+2^23 mod ell, 3: 1+3^23 mod ell}.
//   (ell,n)=(101,22):   weight 24, Legendre(144169|101) computed at runtime
//     and confirmed = -1; blocks = {dim 4 (residue_degree=2, CERTIFIED, with
//     a stored, independently re-evaluated residue_witness; T_2's factor the
//     irreducible X^2-1080X-20468736 mod 101, re-verified irreducible by
//     this file's OWN gcd/Frobenius test), dim 1 Eisenstein}.
// For every degree-1 block in all four cases, the JSON "eigen_system" value
// theta_p is checked against the independently computed constant mod ell.
// Also: Sum(block dims) == cohomology_dim is re-verified from the parsed
// JSON in every case, and each level-(b) matrix (reconstructed from its JSON
// array) has its characteristic polynomial recomputed by this file's own
// division-free code and checked to equal the PRODUCT of that prime's
// reported per-block char_polys (also read from the JSON) -- a genuine
// cross-check between level (b) and level (c) of the same report.
//
// Returns the number of failing checks (0 = success, I9). On any failure
// prints the offending (ell, n), the parsed block structure, and the
// expected structure/values.
uint32_t HeckeSimultaneousValidation();

#endif // HECKE_SIMULTANEOUS_VALIDATION_H
