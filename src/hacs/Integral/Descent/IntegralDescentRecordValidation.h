// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_DESCENT_INTEGRAL_DESCENT_RECORD_VALIDATION_H
#define INTEGRAL_DESCENT_INTEGRAL_DESCENT_RECORD_VALIDATION_H

#include <cstdint>

// DESCENT milestone D6 (slot 24): `coefficient_ring = Z` un-reserved, the record integral_descent_record/v1, its campaign
// cells and console verb, and the staged golden points.  Arms (one try each; an exception fails its arm only):
//   (S) the setting: `Z` parses (k = 0), round-trips, a non-zero fingerprint = FNV of "coefficient_ring = Z\n";
//       Z/<m> stays reserved; CoefficientRingK refuses under Z.
//   (F) the refusal closure under Z: EquivariantCohomologyOnK, HeckeOperatorOnK, RequireFieldCoefficientRing refuse by
//       name (NotImplementedInThisBuild); console: a field verb under Z exit 5, `descent` under Z exit 0, `descent`
//       under F_ell exit 3, a bad argument exit 3.
//   (R) the record against three oracles outside the build (descentN_oracle.tsv / descent_oracle_n128.tsv: the groups;
//       d3_hecke_oracle.tsv: char_Z and the torsion Hecke modules; d4_pair_oracle.tsv: the pair) on 13 cells.
//   (J) the record's schema and hash input: deterministic digest, independent of timings; a second non-default setting
//       changes the digest and not the invariants; the JSON's keys; the per-complex fields; refusals by name.
//   (K) the campaign under Z: spec rules, cell ids `desc_N.._n.._s<fp>`, a two-cell run into a temporary directory
//       (files, trailer, summary.tsv, index.json, resume), the field job loop refusing a Z spec.
//   (G) the staged golden points: GoldenDescentRecords() equals the committed prediction
//       docs/implementation/slot24/d6golden/expect.tsv key for key (all tier A), none is in the reference
//       (add-only), and two computations agree hash for hash.
//   (Z) (slot-24 NB-TOURFIX) the factored char_Z of the console member `charpoly_factored`: the factoriser
//       NumberField/IntPolyFactor.h on constructed products (X^4 + 1, X^4 - 10X^2 + 1, X^8 + 1 split modulo every prime)
//       and on every char_Z of d3_hecke_oracle.tsv (product verified); the console member at (1, 10) and (11, 0).
uint32_t IntegralDescentRecordValidation();

#endif
