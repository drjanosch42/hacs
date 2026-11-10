// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_DESCENT_HECKE_Z_EXPORT_Z2_VALIDATION_H
#define INTEGRAL_DESCENT_HECKE_Z_EXPORT_Z2_VALIDATION_H

// EXPORT-Z step 2, package EXZ2-P3 (hecke-extensions slot 25; docs/implementation/slot25/EXPORTZ2_design.md Sec. 6, 7.5;
// EXZ2P3_report.md): T_p = [[tau, M], [0, F]] on H^1(Gamma_0(N); Sym^n Z^2) = T (+) Z^r with F EXACT over Z in the picked basis
// iso_dixon/v3 (P1b) by the proven CRT (P2), M in the same basis.
//
// HeckeZExportZ2Validation (standard, <= 10 s): eight cells (the step-1 cells but (8, 10), plus (1, 60)).  Arms:
//   (R) P2's lanes on P1b's REAL basis: proven, the Phi_0 and the Phi_cert bound give the same F (FNV), 1 vs 4 threads and tile 1 vs 128
//       identical, pinned FNV(B), FNV(Phi_cert), FNV(F);
//   (A) char(F_p) = the record's char_Z (independent route: the shipped level-N operator, prop:crtcharpolyN);
//   (C) F_p F_q = F_q F_p over Z, the full T_p on T (+) Z^r commute (torsion rows mod d_i); (11, 0) T_2: (x - 3)(x + 2)^2, (1, 10) T_2:
//       (x - 2049)(x + 24)^2;
//   (G) HZ-TOR: SplitFromModuleZ on free_view local (step 1) and global (step 2) gives the same read at every torsion unit;
//   (H) the GLOBAL ORACLE: free_mod(o, ell, K) == T_p in B by the export route's own operator (ExportInBasis) at the torsion primes and
//       at ell = 5, 7, 13, 1000003, 2147483629 (K_max of the route; the field at ell >= 46341); C_ell F == F_local C_ell, M_global =
//       M_local C_ell recomputed; free_mod prefix-consistent;
//   (L) module_Z("local") is step 1: the step-1 fields of the step-2 object (tau, M_block, F_local, d, retraction FNVs) equal a plain
//       ExportModuleZ byte for byte;
//   (T) teeth: InverseUnimodular refuses diag(2, 1) and accepts a dense unimodular matrix; a perturbed F fails the oracle; C = I fails
//       C F == F_local C where C != I; a basis column times ell is not a basis at ell (basis_ok false).
// HeckeZExportZ2SlowValidation (slow): the same pipeline on (8, 10), (1, 200), (23, 20), (311, 2) with every stage timed (basis, lanes,
// bound, lift, exports, assembly) and both bounds (Phi_0, Phi_cert); arms (R), (A) where the record is cheap, (C), (H).
// No golden object, no record key, no settings key, no env variable.

#include <cstdint>

uint32_t HeckeZExportZ2Validation();
uint32_t HeckeZExportZ2SlowValidation();

#endif
