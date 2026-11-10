// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_BLOCK_FORM_VALIDATION_H
#define HECKE_BLOCK_FORM_VALIDATION_H

#include <cstdint>

// Stage S10V -- INDEPENDENT decomposition validation for
// hecke::HeckeModuleDecomposition (Stage S10's level-(c) block-diagonalized
// output, docs/implementation/hecke_action_implementation_plan.md Sec. 3),
// on the two REAL non-semisimple / non-split cases singled out by the
// "S10V -- Decomposition validation" stage: oracle O6 (the ell=691
// Ramanujan-congruence non-semisimplicity, amendment A-F2) and oracle O8
// (the n=22/weight-24 Galois-conjugate-pair split-vs-nonsplit branch).
//
// INDEPENDENCE STATEMENT (read before the .cpp): this file consults ONLY
// the public headers HeckeModuleDecomposition.h / FpPoly.h /
// HeckeOperatorOnK.h / HeckeCosetTables.h / HeckeTauTable.h /
// EquivariantCohomologyOnK.h / LocalSystemSL2Z.h / Matrix.h for API
// signatures, and docs/implementation/hecke_action_implementation_plan.md
// (Sec. 0.1 amendment A-F2, Sec. 3 "Level (c)") plus
// docs/latex/hecke_action.tex (rmk:dickson, sec:example oracle 5) plus the
// task's own O6/O8 restatement for the expected block structures -- it does
// NOT read HeckeDecompositionSmokeValidation.cpp's expected values. Every
// certificate the task asks for is RE-DERIVED by this file's OWN code:
//   * a from-scratch F_ell[X] toolkit (add/sub/mul/pow/divrem/gcd/eval,
//     square-and-multiply powmod, a from-scratch Frobenius ladder and the
//     Rabin gcd/Frobenius irreducibility test) -- deliberately NOT
//     hecke::FpPoly's toolkit and NEVER hecke::CertifyIrreducible;
//   * a from-scratch division-free cofactor-expansion characteristic
//     polynomial (valid at every prime ell, including ell <= d) and a
//     from-scratch Horner polynomial-at-matrix evaluator;
//   * a from-scratch block-diagonal-shape checker and sub-block extractor;
//   * a FRESH Matrix::inverse() of change_of_basis(), independent of
//     HeckeModuleDecomposition's internally cached inverse, used to
//     recompute U * conjugated(p) * U^{-1} == A and to cross-check
//     conjugated(p) against U^{-1} A U computed here;
//   * the mod-ell Legendre symbol of 144169 via Euler's criterion
//     (ModArithmetic::mod_pow), computed at runtime, never hard-coded --
//     the O8 branch below is SELECTED by this runtime computation.
// hecke::HeckeModuleDecomposition::certify() is NEVER called by this file
// (compute(), blocks(), change_of_basis(), conjugated() are the only
// HeckeModuleDecomposition members used -- exactly the API needed to
// extract the reported decomposition, whose claims are then independently
// re-derived and re-checked as above).
//
// Oracles (exact; no tolerance; per the task's O6/O8 restatement, itself
// re-derived from Deng (JNT 2023) / classical Eichler-Shimura, see the
// long derivation comment at the top of the .cpp):
//   O6 -- n=10, ell=691, p=2 (Ramanujan-congruence non-semisimple case):
//     dim H^1 = 3; EXACTLY ONE primary block, dim 3, residue_degree 1,
//     char_poly = (X+24)^3; HARD (amendment A-F2): min_poly = (X+24)^2,
//     equivalently rank(A+24I) = 1 (Jordan partition (2,1), NOT (3)).
//     A computed rank != 1 is treated as a genuine mathematical surprise
//     and reported loudly, never silently accepted.
//   O8 -- n=22 (weight 24), ell in {101, 10007}: dim H^1 = 5. The Legendre
//     symbol (144169|ell) is computed at runtime (Euler's criterion) and
//     used to select the branch:
//       (144169|ell) = -1: one dim-4 block, residue_degree 2 (char factor
//         X^2-1080X-20468736, the fused Galois-conjugate weight-24 cusp
//         pair), PLUS the Eisenstein dim-1 block (X-8388609). Block dims
//         {4,1} (order-independent), residue degrees {2,1} asserted, plus
//         the char/min-poly factorizations; the per-block Jordan partitions
//         are RECORDED only (semisimplicity of the mod-ell reduction is not
//         contract-backed).
//       (144169|ell) = +1: two distinct dim-2 blocks, residue_degree 1 each
//         (the two Galois-conjugate cusp forms split over F_ell), PLUS the
//         Eisenstein dim-1 block. Block dims {2,2,1}, residue degrees all 1
//         asserted, plus the factorizations (the two dim-2 blocks' char
//         polys multiply to the theoretical quadratic squared, and each
//         extracted root is independently checked to satisfy the
//         theoretical quadratic -- see the .cpp for how this sidesteps
//         needing a modular square root).
//   Both oracles: reconstruction (det U != 0 via a fresh inverse,
//   U * conjugated(p) * U^{-1} == A exactly, conjugated(p) exactly
//   block-diagonal at the reported sizes, sum of block dims == dim H^q) and
//   independent re-verification of every reported irreducible factor via
//   this file's own gcd(pi, X^{ell^i}-X) / Frobenius test.
//
// Returns the number of failing checks (0 = success, I9). On any failure
// prints the offending (n, ell), the matrix A, its independently computed
// char/min-poly, the reported block dims/residue degrees, and (for O8) the
// computed Legendre symbol.
uint32_t HeckeBlockFormValidation();

#endif // HECKE_BLOCK_FORM_VALIDATION_H
