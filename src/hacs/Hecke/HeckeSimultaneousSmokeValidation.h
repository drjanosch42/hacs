// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_SIMULTANEOUS_SMOKE_VALIDATION_H
#define HECKE_SIMULTANEOUS_SMOKE_VALIDATION_H

#include <cstdint>

// Stage S11 -- Definition-of-Done smoke validation for the simultaneous
// MaxSpec(T) refinement (hecke::HeckeModuleDecomposition's multi-operator
// path, ComputeMultiOp()) and the Hecke/HeckeReport.{h,cpp} JSON writer.
//
// Runs the REAL pipeline (HeckeOperatorFamily over the 7-vertex SL(2,Z)
// model) to build A^{(1)}(T_2), A^{(1)}(T_3) at:
//   * (ell, n_sym) = (13, 10): asserts dim H^1 = 3, T_2's eigenvalues are
//     {-24 === 2, 2, 2049 === 8} mod 13 and T_3's are
//     {252 === 5, 5, 177148 === 10} mod 13 (charpoly identities, cross-
//     checking the already-validated S7V/S10V oracle O5 reduced mod 13);
//     the SIMULTANEOUS decomposition under the family {T_2, T_3} is
//     EXACTLY one dim-2 block (the cuspidal system of Delta mod 13 -- the
//     holomorphic/antiholomorphic pair shares ALL Hecke eigenvalues, so no
//     operator in the family separates it; per-operator Jordan partitions
//     are recorded, not asserted) plus one dim-1 Eisenstein block (system
//     T_p |-> 1 + p^11 mod 13), with the Eisenstein block's eigen-system
//     values asserted: T_2 |-> 8, T_3 |-> 10;
//   * (ell, n_sym) = (691, 10): asserts a single dim-3 block (the O6
//     Ramanujan-congruence result of Stage S10V, now re-derived under the
//     TWO-operator family {T_2, T_3} rather than just T_2 alone).
// Also builds a Hecke/HeckeReport JSON for the (13, 10) case and round-trips
// it through an embedded, from-scratch mini JSON parser (LOCAL to this file;
// no third-party library, plan Sec. 3 binding requirement), checking a
// representative sample of fields (ell, degree, level_c.cohomology_dim,
// block dims/residue degrees, and the Eisenstein block's eigen_system).
//
// Returns the number of failing checks (0 = success, I9). Does NOT regress
// the Stage S10 single-operator path (HeckeDecompositionSmokeValidation,
// HeckeBlockFormValidation) -- see HeckeModuleDecomposition.cpp: the k = 1
// dispatch (ComputeSingleOp()) is untouched S10 code, called verbatim.
uint32_t HeckeSimultaneousSmokeValidation();

#endif // HECKE_SIMULTANEOUS_SMOKE_VALIDATION_H
