// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_CROSS_CHECK_VALIDATION_H
#define HECKE_CROSS_CHECK_VALIDATION_H

// Stage S8 -- Definition-of-Done battery for the two independent cross-check
// routes of [C, obl:agreement]:
//
//   Route A (three-map factorization, HeckeCrossChecks): for every
//   (p, ell, n) in {2,3} x {5,13,691} x {0,2,10}, the A-F3 localizing
//   assertions ((i) beta maps L'-equivariant cochains to L''-equivariant
//   cochains; (ii) beta.alpha == Xi_g bit-identically) followed by the
//   acceptance identity omega.beta.alpha == Theta_g BIT-IDENTICALLY on ALL
//   of C^q(K;V), q = 0,1,2, as matrices, against Stage S7's
//   get_Theta_flat_matrices().
//
//   Route B (bar-cocycle H^1, BarCocycleH1): the independent group-cocycle
//   H^1(Gamma; Sym^n F_ell) with the corrected Hecke formula
//   [C, prop:barformula]; asserts dim H^1 agreement with the pipeline
//   (ell >= 5 on the whole grid) and that charpoly AND minpoly of the bar
//   Hecke matrix equal those of A^{(1)}(T_p) (basis-independent, binding
//   comparison; the division-free helpers of BarCocycleH1.h serve both
//   sides).
//
// These are internal-consistency identities valid for ALL ell >= 5 -- no
// eigenvalue gating, no tolerances (invariant I9). Any failure prints the
// offending (p, ell, n), degree and data, and returns a nonzero localizing
// code.

#include <cstdint>

uint32_t HeckeCrossCheckValidation();

#endif // HECKE_CROSS_CHECK_VALIDATION_H
