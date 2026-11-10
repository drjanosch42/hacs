// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_DESCENT_HECKEZ_EXPORT_VALIDATION_H
#define INTEGRAL_DESCENT_HECKEZ_EXPORT_VALIDATION_H

#include <cstdint>

// HeckeZ package EXPORT (slot 25; 00-slot25-plan.md v0.2 Sec. 2.1, slot24/heckez/06-work-packages.md card "EXPORT"):
// the export H^1(Gamma_0(N); Sym^n Z^2) (x) Z/ell^K with its T_p (Integral/Descent/DescentModuleExport.h) against
// independent routes, by BASIS-FREE invariants only (R-E2).  Seven arms; an exception fails its arm only.
//   (a) level 1, ell >= 5: (1, 10) at 691, K = 2 and (1, 26) at 5, K = 2 against the console's Z/ell^K module route
//       (hecke_ext::ringalg::RunModule, the shipped EquivariantCohomologyOnK + HeckeOperatorOnK): the invariant factors, the
//       F_ell-rational joint systems, per system the algebra A_K and its free-part image (length, dim mod ell, Hilbert
//       function; ReadAlgebra on Localize), the characteristic polynomial of T_p on the free part mod ell^{K - t}.
//   (b) the universal-coefficient types (03 B2(1)) of H^1 (x) Z/ell^K, K = 1 .. 4, equal the descent record's groups
//       (BuildDescentRecord, the groups' independent Smith forms) at T1 (1, 10) (ell = 2, 3), T4 (2, 6) (ell = 2, 3, 5),
//       T5 (8, 10) (ell = 2, 3); refusals by name: p | N at ell = 2 and 3 (`blocked(export)`, R-E1), ell = 4.
//   (c) level N: char(T_p | free) = char_Z(T_p) (CharpolyOverZ) mod ell^K at (2, 6) (ell = 2, the tree), (8, 10) (ell = 3,
//       the tree), (11, 0) (ell = 5, K = 2); at K = 1 equal to `level_n`'s F_5 matrices (CokernelHeckeMatrices) at (11, 0);
//       a word prime with ell^2 >= 2^31 (audit F1): (11, 0) and (1, 10) at ell = 65521, K_run = K_max = 1 (the field F_ell),
//       against char_Z mod ell and (11, 0) against level_n's F_65521 matrices.
//   (d) the Bockstein is Hecke-equivariant (03 B1(2), Sec. 6.1 (c)): at (1, 30), ell = 5, K = 2, the eigenvalue of T_p
//       (p = 2, 3) on H^0(V / 5^2) (the shipped module route in degree 0) equals the export's T_p on tors_5 H^1 = Z/25.
//   (e) T3, (11, 0): the Smith form of T_2 + 2 on L / 25 (the export's free block) has the exponents [0] (a unit entry: 02
//       Sec. 6's T3 certificate), so v_5(exp eta_L) = v_5(c(3)) - 0 = 1: eta_L = Z/5.  (The 06 card says "[1]"; its source 02
//       Sec. 1 / Sec. 6 reads [0] -- EXPORT_report.md.)
//   (f) a level-1 torsion prime: (1, 30), ell = 5: H^1/5 and L/5 differ by tors_5/5 (one summand); dim A'_1 read on L/5
//       (= 5 A'_2, the reading of L/25 at shift 1) equals the console route's free-part image of A_3 at shift 2 (= level 1), per system.
//   (g) (audit F4) the mixed block at ell in {2, 3}: at (8, 10) (2, 3), (2, 6) (2), (1, 10) (2, 3), (4, 12) (2), (9, 8) (3), K = 1, 2,
//       K_max, every T_p is an endomorphism of (+)_i Z/ell^{e_i} (A_ik ell^{e_k} = 0 mod ell^{e_i}) and T_p T_q = T_q T_p on the FULL
//       matrix (torsion, free and mixed blocks); the mixed block must be non-zero somewhere.  Forced refusals by ExportOptions:
//       [ModulePrecision] at (1, 30) ell = 5 K_run = 3 and (8, 10) ell = 2 K_run = 5; [RssCeiling] (rss_cap_bytes = 1) at ell = 2, 691.
// Every arm also compares the export's torsion block with the descent's own torsion read at q = 1 (TorsionOnTree /
// TorsionOnCokernel, the record's route) by the choice-free invariants (hecke3::Invariants canon): `descent_records_identical`.
// Gate line: HECKEZ_EXPORT arms=7 failing=<k> descent_records_identical=<yes|no> verdict=<OK|FAIL> budget=<gated(standalone) |
// informational(in-process)> ...  The RSS budget (each arm <= 1 GiB growth over its entry RSS) decides the verdict only in a
// standalone run (entry RSS < 256 MiB); inside --all it is printed, not gated (audit F2).  Seconds are never gated.
uint32_t HeckeZExportValidation();

#endif
