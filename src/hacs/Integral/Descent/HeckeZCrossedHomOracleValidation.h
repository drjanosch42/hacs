// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_DESCENT_HECKE_Z_CROSSED_HOM_ORACLE_VALIDATION_H
#define INTEGRAL_DESCENT_HECKE_Z_CROSSED_HOM_ORACLE_VALIDATION_H

// Slot-26 card Z3b (docs/implementation/slot26/Z3b_design.md v0.2, Z3b_report.md): HeckeZCrossedHomOracleValidation (standard).
// An M oracle at ell = 2, 3 (and ell >= 5) that does not use T~_Iso: per cell, op p !| N, torsion unit ell and free basis column j,
//     y_j := kappa' T^x kappa (b_j)        (HeckeZCrossedHomOracle.h: [C, eq:barcorrected] for Gamma_0(N), moved to SL_2(Z) by Shapiro)
//     (M)  pi(R y_j) == M_{.j} + pi_B F_{.j}   (row a mod ell^{sigma_a}; R = RouteBasis, pi = ModuleExport::retraction)
//     (F)  Phi_cert^T y_j == G F_{.j}           (mod q1, q2 and ell^{K_run}; G = Phi_cert^T B)
// Eight cells (1, 10), (2, 6), (9, 8), (11, 2), (1, 30), (5, 12), (1, 60), (4, 12); 28 torsion units, 15 at ell in {2, 3}.  Arms:
//   (K) conventions: the oracle's rho_{M_N}(S), rho_{M_N}(U) = the carrier's d^0 blocks exactly, own rho_n = RhoNZ, w_1 = I, kappa / kappa'
//       identities, d^1 kappa' kappa b = 0 exactly, f(-I) = 0, words multiply back, Phi_cert^T d^0 == 0, H^0 = 0 -- before any Hecke step;
//   (F) the free part at every modulus, and d^1 y == 0;   (M) the M oracle at every unit and op;
//   (Pi) pi(R d^0 m) == 0, pi(R b_i) = the exported pi_B (audit N1), pi(cocycles) = e_a (sanity);
//   (I) R on the ell-torsion (audit M1): g_i = d^0(m_i)/ell for a basis m_i of ker(d^0 mod ell); nu = t_ell, and the socle matrix of
//       pi(R g_i) has rank t_ell;
//   (W) another representative system changes y as a cochain, not (F) or (M);
//   (T) teeth: M + 1, the pre-fix block M_local C, rho_n(delta^iota), and y_j + g_i (a carrier ell-torsion class) must fail;
//   (Tau) (slot-27 Z3b-F, design Q4 follow-up) tau on the ell-socle without T~_Iso: pi(R T^x g_i) == tau|_ell pi(R g_i) for the exact
//       integral socle cocycles g_i of (I), every unit and op; teeth tau + 1 at every diagonal entry of the ell-block, and T^x g_i + g_i.
// Gate line: HECKEZ_XHOM_ORACLE arms=8 failing=0 cells=8 checks=<c> tor_units=28 units_23=15 nonzero_23=<k> nonzero_M_23=<k>
//            tau_nonzero=<k> teeth=<k>/<k> hecke=barcorrected/shapiro verdict=OK seconds=<s>  (slot-27 Z3b-F: nonzero_M_23 = units at
//            ell in {2, 3} with M != 0 for some op, audit F6; ARM lines carry no skipped= field, audit F7).
// Read-only env hook HECKE_XHOM_CELLS="N:n:p1/p2,..." replaces the cell list (prints EFFECTIVE; floors and the default counts are then
// not asserted).  No golden object, no record key, no settings key; nothing of production changes.

#include <cstdint>

uint32_t HeckeZCrossedHomOracleValidation();

#endif
