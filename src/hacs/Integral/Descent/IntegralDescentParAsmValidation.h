// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_DESCENT_PAR_ASM_VALIDATION_H
#define INTEGRAL_DESCENT_PAR_ASM_VALIDATION_H

// PAR-ASM (slot 24; DescentParAsm.h, docs/implementation/slot24/PARASM_report.md): the threaded / concurrent assembly stages of the
// integral descent against their serial code of record (the three levers off, scoped), at 1, 2, 7 and 16 workers:
//   (X) the fork-join primitives: every index once, the LOWEST throwing index re-thrown, nested regions serial; the joint-RSS rule;
//   (M) pair::MatMulZ (threaded MulMod) on random reduced operands at 2^30, 3^19, 5^13, 691^3, entrywise;
//   (B) BuildTreePairComplex at ell = 2, 3 on five cells: every c, cb, d, d_B, a^* entrywise; a fault arm throws the same message;
//   (E) maps::InducedMapEmbed against InducedMap with the dense f: every presentation and map of RunPairLes (F, Lf, ZL, types, ok);
//   (P) AssembleDescentPair on three cells (tree pair + concurrent ell >= 5 pieces): every group, map type, count, piece and refusal;
//   (L) AssembleDescent (level 1, the concurrent Bredon-tree primes) at n = 100, 202: H, every bad-prime run; the joint RSS rule
//       (a small ceiling keeps one prime at a time) and the measured growth within the joint prediction;
//   (R) BuildDescentRecord's digest at (1, 30) and (11, 4).
// Read-only; no env hook is set.  Prints one EFFECTIVE HeckeDescentParAsm line.

#include <cstdint>

uint32_t IntegralDescentParAsmValidation();

#endif
