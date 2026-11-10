// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_HECKE_LAMBDA_TOWER_VALIDATION_H
#define HECKE_EXT_HECKE_LAMBDA_TOWER_VALIDATION_H
// (hecke-extensions slot 22, P10) HeckeLambdaTowerValidation (standard): the gates G0 - G14 and the negative controls of
// docs/implementation/slot22/P10_spec.md (v0.3, section 15 = the implementation predictions) on the real Hida tower
// (HeckeExt/HeckeLambdaTower).  G0 (the orientation, prop:u-towerorient) runs first; if it fails the validation stops
// before any Lambda-adic reading.
//
// Default cells: ell in {5, 7}, every even class, r <= 3; (11, 2), (11, 4), (13, 0), (13, 4) at r <= 2; the Gamma_1 oracle at
// r <= 2; WP-HL's forms side at k = 12, 22, 112 (11) and 12, 24, 132 (13; 158 until spec v0.3, erratum E1); P7's truncations (K, m) = (3, 1..3) at x_12.
// Read-only hooks (each prints its EFFECTIVE line):
//   HECKE_LAMBDA_TOWER_R3=1          adds (11, 2, 3) and (13, 0, 3) (RSS predicted from the r = 2 fill first;
//                                    HECKE_LAMBDA_TOWER_RSS_CAP_GIB, default 4, refuses tower_rss_cap above it);
//   HECKE_LAMBDA_TOWER_CELLS="ell:a:r[:K],..."   replaces the tower cell list (malformed token refused by position; a list
//                                    selecting no cell fails; ell > 13 refused by name tower_edge_not_built in this build).
#include <cstdint>

uint32_t HeckeLambdaTowerValidation();

#endif // HECKE_EXT_HECKE_LAMBDA_TOWER_VALIDATION_H
