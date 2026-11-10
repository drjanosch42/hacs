// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_DESCENT_SMITH_BLOCKED_VALIDATION_H
#define INTEGRAL_DESCENT_SMITH_BLOCKED_VALIDATION_H

// PAR-D2 (hecke-extensions slot 24; docs/implementation/slot24/PARD2_report.md): the blocked, threaded, SIMD Smith form with
// factored transforms (Integral/Descent/DescentSmith) against D2's scalar oracle maps::SmithFormOracle, bit for bit.
//
// IntegralDescentSmithBlockedValidation (standard).  Arms:
//   (R) random matrices over Z/ell^k (ell = 2, 3, 5, 7, 691, 46337; k from 1 to the 2^31 bound): dense, low rank with prescribed
//       torsion (X diag(ell^a u) Y), sparse, zero / repeated / all-divisible rows, empty and thin shapes; sigma, L, L^{-1}, R,
//       R^{-1} entrywise equal to the oracle's, at thread counts 1, 2, 7, 16, panel widths 1, 5, 32 and every SIMD path the CPU
//       has (scalar, avx2, avx512); the row / column accessors and L B against the dense transforms;
//   (D) real D2 inputs: every Smith form (r c >= 400) of D2's level pair (tree and dense routes, ell = 2, 3 and >= 5 pieces) and
//       of D4 / D5c's cusp pair, captured, replayed on both engines (full transforms) at thread counts 1 and 16, and at 7 on every
//       SIMD path;
//   (C) the callers: BuildPres, InducedMap, LatticeQuotient and the whole pipeline (AssemblePairMaps, AssembleDescentPair) with
//       HECKE_DESCENT_SMITH_BLOCKED on and off (scoped): every presentation (precision, exponents, generators, coordinates),
//       every map (F, Lf, kernel / image / cokernel), every group, count and refusal text equal;
//   (P) one timing line SMITH_BLOCKED_COST (a real D2 input, oracle vs blocked; printed, not gated).
// Read-only; no env hook is set.

#include <cstdint>

uint32_t IntegralDescentSmithBlockedValidation();

#endif
