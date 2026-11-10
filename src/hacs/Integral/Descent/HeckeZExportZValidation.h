// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_DESCENT_HECKEZ_EXPORT_Z_VALIDATION_H
#define INTEGRAL_DESCENT_HECKEZ_EXPORT_Z_VALIDATION_H

// HeckeZExportZValidation (slot-25 EXPORT-Z step 1, tier standard; design docs/implementation/slot25/EXPORTZ_design.md Sec. 4, the
// arms that apply to tau and M).  Cells (N, n, P): (1, 10) [2, 3], (2, 6) [3, 5], (8, 10) [3, 5], (9, 8) [2, 5], (11, 0) [2, 3],
// (11, 2) [2, 3], and (audit F4) (1, 30) [2, 3] (Z/25, Z/29) and (5, 12) [2, 3] ((Z/5)^2 at ell = 5 | N, Z/11) (+ the toys of (G)).
// Every arm has a check floor (the deterministic count of record); an arm below its floor FAILS, a skipped block prints a SKIP
// line with its reason, and the ARM line carries floor= and skipped= (audit F3).  Oracles are independent of the read (L0 / L1 coordinates) that produced tau and M:
//  (A) algebra: tau_p an endomorphism of (+) Z/d_i (tau_ij d_j = 0 mod d_i), block-diagonal over ell; tau_p tau_q = tau_q tau_p;
//      the full ell-local block [[tau, M], [0, F_local]] commutes; the retraction pi_ell kills B^1 (pi d^0 = 0), is the identity on
//      the torsion generators and kills the local free generators (d^0 rebuilt from the model, not taken from the export).
//  (B) the record's basis-free invariants: the d_i per ell = the torsion of the record's H^1; r = its free rank; char(F_local) =
//      the record's char_Z(T_p | H^1/tors) mod ell^K.
//  (C) presentation at the cochain level: T_p g_j - sum_i A_ij g_i lies in B^1 + ell^K C^1 for every generator, decided by a
//      fresh Smith form of [d^0 | residuals] over Z/ell^K (its levels equal those of d^0) -- the operators re-built from the
//      shipped constructors, applied to the export's cocycles; a perturbed entry must FAIL it (the oracle has teeth).
//  (D) tau's ell-block read by hecke3::Invariants = the record's torsion canon (q = 1, [C] prop:torsionhecke (1)).
//  (E) splitting invariance: a random h in Hom(L, T_ell) moves the free generators to g_j + sum_i h_ij t_i; the predicted
//      M' = M + tau h - h F_local passes (C) in the moved generators (two seeds per cell); M unmoved does not when tau h != h F.
//  (G) hand checks: (11, 0): T = 0 (n = 0: H^1 = Hom(Gamma^ab, Z) is torsion-free), r = 3; (1, 10): T_2 = Z/2 + Z/4, T_3 = Z/3,
//      r = 3, char_Z(T_2) = (x - 2049)(x + 24)^2; (8, 10): (Z/2)^3 + Z/4 + (Z/8)^2 + Z/3, r = 22 (EXPORT_report.md arm (b));
//      every cell: free_block = "pending(step 2)", splitting = "export_retraction/v1", split = "unknown"; a forced export refusal
//      (rss_cap_bytes = 1) makes the object blocked(export) [RssCeiling] with no tau served; a provider accepted at ell = 2 and
//      refused at ell = 3 leaves free_rank = 0 and no torsion data (audit F1); (1, 30) and (5, 12) by hand.
//  (S) the step-2 entry point (ExportOptions::basis): T_p in a CALLER-SUPPLIED basis B.  One random integral B at q = 691 and
//      q = 65521 (the field), (1, 10): char(F_B) = the record's char_Z mod q^{K_max}; F_B(T_2), F_B(T_3) commute; B = the export's
//      free generators gives its free block (also on the tree at (8, 10), ell = 2); B G gives G^{-1} F_B G; a singular / misshaped B
//      is refused by name.  (audit F5) B = Own G + Tors h + d^0 w (G a random unit) where the export HAS torsion at q: (9, 8) q = 7
//      (cokernel) and (8, 10) ell = 2 (tree): G F_B = F_own G and char(F_B) = char_Z mod ell^{K_max}, K_max < K_run.  The batch:
//      1 = 2 threads in ok, K, why and F; q | N refused by name; an injected lane defect is rethrown after the join (audit F2).
//      Run once under CheckThread (TSan) at the audit fixes (audit F6; EXPORTZ1_report.md "Audit fixes").
// Gate line: HECKEZ_EXPORTZ arms=7 failing=0 cells=.. largest_torsion_prime=.. verdict=OK.

#include <cstdint>

uint32_t HeckeZExportZValidation();

#endif
