// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 16, WP-10) HeckeNebentypusValidation -- nebentypus on the
// fixed model by Shapiro: [C, sec:nebentypus] (v1.16.0-pre.2), rmk:nebcheckable
// (NB1)-(NB3), (NB5), (NB6); the convention [GT, conv:nebentypus] and the
// dictionary [GT, bridge:nebentypus-dictionary] as the L16-2 hand-off states them.
//
// Arms: (F) F_q against [E] def:canon's field (HeckeExt/HeckeCanonicalField),
// the Conrey characters, the reduction rule against WP-9b's psi values, the
// Teichmuller lift (lem:teichmuller); (B) (NB2) chi = 1 is the shipped level-N
// model bit for bit (relations, cochain maps, cokernel, Heilbronn, H^1 matrices);
// (C) the character model across THREE routes at d = 1 -- the cokernel with
// H_{N,p,chi} (rmk:nebheilbronn), the fixed model with eq:MHchi / eq:nebcorr, and
// an INDEPENDENT group-cocycle route with a character (own P^1(Z/N), own
// transversals, no tau, no model) -- dims, T_p / U_p charpolys, commutativity,
// prop:neborbit (2) at the parity-zero cells; (D) (NB1) prop:neborbit (4) in
// characteristic 0 against the cokernel on every character mod N <= 30, and the
// classical half against WP-9's dims; (E) (NB3) the Gamma_1(N) model with its
// diamonds, decomposed over F_q (d = 1 and d = 2) into the character models;
// (N5) (NB5) WP-9 / WP-9b's charpolys (docs/implementation/slot16/
// WP-9_nb_predictions.tsv, cusp^2 . Eis), the chi / chibar dictionary read and
// reported (NEB_DICT lines; a mismatch is a finding for the owner, never a re-pin);
// (N6) (NB6) the CM cells; (W) W_K(F_ell) = Z/ell^K with the Teichmuller-lifted
// character on the fixed model; (R) the refusals by name.
// Env hook HECKE_NEBENTYPUS_CELLS="N:chi:k:ell[:d],..." (chi a Conrey index m of
// N.m): REPLACES the default arms by the cokernel route over the listed cells
// (NB1 gated, T_p charpolys reported); prints
// `EFFECTIVE HeckeNebentypusValidation arm=CELLS cells=<k>` and one NEB_CELL line each.
// Standard tier; writes nothing.
#pragma once
#ifndef HECKE_NEBENTYPUS_VALIDATION_H
#define HECKE_NEBENTYPUS_VALIDATION_H

#include <cstdint>

uint32_t HeckeNebentypusValidation();

#endif
