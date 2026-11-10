// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_TAU_DUMP_VALIDATION_H
#define HECKE_TAU_DUMP_VALIDATION_H

#include <cstdint>

// (hecke-extensions slot 16, L16-4; owner ruling Lean Q14 of 2026-10-01)
// HeckeTauDumpValidation (`fast`): the validation-only, env-interlocked
// TABLE-DUMP hook for the Lean checkers of lean/GL2/Checkers/.
//
//   * HECKE_DUMP_TAU unset or empty: prints one [SKIP]-style line and returns 0
//     -- a no-op, so `--all` / `--all-full` stay read-only.
//   * HECKE_DUMP_TAU=<dir>: <dir> must already exist, must be a directory and
//     must NOT lie inside the source tree (OC_SOURCE_DIR); otherwise the run
//     is refused (returns 1).  It then writes exactly four files into <dir>
//     and nowhere else:
//       tau_p2.txt          the p = 2, nu = 1 tau table of the golden ext
//                           cells' cone action (HeckeCosetTables(2, 1, 5, 2),
//                           default TauVertexPolicy), with the 7-vertex model's
//                           simplex lists, orbit representatives, every
//                           transporter set S(s1, s2) and its vertex
//                           correspondences, the coset representatives s_i,
//                           g, the solve regime, and the FNV-1a-64 of the table
//                           computed exactly as HeckePairHomotopy::
//                           tau_table_fnv64() does (the golden object
//                           tau_table_fnv64_p2_r1 is HashScalar64 of it);
//       rho_n<n>_ell<ell>.txt  for (n, ell) in {(2, 5), (10, 691), (16, 13)}:
//                           the consumer rho_n pool of that cell's p = 2 tau
//                           table ([C, obl:tau](vi), keys (g gamma) mod ell)
//                           and rho_n of T, S, ST, TS, s_i, delta_i by
//                           ComputeSymmetricPowerMatrixFM.
//     It prints `EFFECTIVE HeckeTauDumpValidation arm=DUMP cells=3` and one
//     TAU_DUMP line per file.
//   * HECKE_DUMP_TAU_RHO_CELLS="n:ell,..." (slot 17, L17-4 repair; read-only,
//     meaningful only with HECKE_DUMP_TAU): replaces the three default rho cells
//     by the given list (e.g. the 31 golden ext cone_action points).  Strict:
//     decimal n <= 1000, 5 <= ell < 2^32, no duplicate; the first bad cell is
//     REFUSED by its position (returns 1).  Prints
//     `EFFECTIVE HeckeTauDumpValidation arm=RHO_CELLS cells=<k>`.  Set but
//     empty, or a trailing comma: REFUSED.  Unset: the behaviour above, byte
//     for byte.
//   * HECKE_DUMP_TAU_PAIR="p:nu,..." (slot 18, L18-3; read-only, meaningful
//     only with HECKE_DUMP_TAU): additionally, for each (p, nu), builds the
//     golden recipe's certified tables (HeckeCosetTables(p, nu, ell, 2) with
//     ell = 5, or 13 at p = 5; HeckeTauTable, HeckeBoundaryTauTable,
//     HeckePairHomotopy; u_ell off, as GoldenHarness RunExtTauCell) and writes
//       tau_p<p>_r<nu>.txt    the tau table (tau_p2.txt's format);
//       taubd_p<p>_r<nu>.txt  the tau_bd table ([C, obl:taubdry]): boundary
//                             simplices, slots, regime, table_fnv64();
//       eta_p<p>_r<nu>.txt    the eta table ([C, obl:eta]): slots, regime,
//                             table_fnv64() and the two base-table hashes.
//     Strict: p prime <= 97, 1 <= nu <= 2, no duplicate; the first bad entry
//     is REFUSED by its position (returns 1); set-but-empty is refused.  Prints
//     `EFFECTIVE HeckeTauDumpValidation arm=PAIR tables=<k>`.  Unset: the
//     behaviour above, byte for byte.  The Lean side is
//     lean/tools/taubd_eta_to_lean.py -> GL2/Checkers/{TauBd,Eta}*.lean.
// No record, golden, digest or setting is touched; nothing is computed that
// the golden harness does not already compute.
uint32_t HeckeTauDumpValidation();

#endif // HECKE_TAU_DUMP_VALIDATION_H
