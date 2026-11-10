// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_DESCENT_PAR_LN_VALIDATION_H
#define INTEGRAL_DESCENT_PAR_LN_VALIDATION_H

// PAR-LN (slot 24; DescentLevelN.h, DescentParAsm.h (d), docs/implementation/slot24/PARLN_report.md): AssembleDescentN's sparse
// ell >= 5 cokernels split into the elimination and the dense residual, the eliminations run concurrently under a joint RSS rule.
//   (S) sparse == dense: every bad prime's sparse cokernel (EliminateSparseCokernel -> FinishSparseCokernelSmith, and the threaded
//       assembly's run at 1, 2, 7, 16 workers) against DenseCokernelSmith (the same generators stacked densely): rank and every
//       valuation; the split equals the one-call RunSparseCokernelSmith field by field;
//   (T) threaded == serial: AssembleDescentN with the lever on at 1, 2, 7, 16 workers against HECKE_DESCENT_PAR_LN off (scoped) on
//       five cells (2 to 4 bad primes): H, refusal, the good prime, every SparseSmithRun (counts, units, residual shape, r_K, certificate, torsion), the
//       ell = 3 shortcut; the concurrency min(workers, #primes);
//   (P) the joint RSS rule: ceilings of 2.5 and 1 predictions give 2 and 1 workers; the measured growth of the elimination stage within
//       the joint prediction; a ceiling below the residual refuses the same primes with the same text in both modes;
//   (E) a failing elimination (K = 1) throws the same message in both modes;
//   (R) BuildDescentRecord's body and digest at two level-N cells, lever off and on at 1, 7, 16 workers.
// Read-only hook HECKE_DESCENT_PARLN_CELLS=N:n[:full|:no2],... (measurement; replaces nothing): per cell the ell >= 5 stage serial and
// threaded (with :no2 also the ell = 3 piece, with :full the whole AssembleDescentN, in both modes), one PARLN_COST line each, and one EFFECTIVE line.
// Prints one EFFECTIVE HeckeDescentParAsm line.

#include <cstdint>

uint32_t IntegralDescentParLnValidation();

#endif
