// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_WIDE_TAU_VALIDATION_H
#define HECKE_WIDE_TAU_VALIDATION_H

#include <cstdint>

// (hecke-extensions slot 24, package ELL-W; docs/implementation/slot24/ELLW_report.md)
// The wide tier of [C, lem:bounds](5) / [C, thm:backend](4) / [C, obl:backend](ii): a tau table whose certified bound
// B = max(B_tab, 4 N^2) exceeds 2^31 - 1 (N > 23170) is certified by clause (e) as 2 N B <= 2^63 - 1 instead of being
// refused with code 51, the stored-class products and determinants being formed in __int128 after an int64 overflow
// (hecke::mul_tiered / det_tiered / inv_unimodular_tiered).
//
// HeckeWideTauValidation (STANDARD; arms [1], [2] at ell <= 5003, [3], [4](b),(c), [5]) and
// HeckeWideTauSlowValidation (SLOW; arm [2] at ell = 23167, arm [3w] and arm [4](a)):
//   [1] the helpers: int64-fitting operands give the int64 value (Auto and ForceWide alike, 20000 random pairs);
//       a product whose int64 intermediates overflow but whose result fits is exact (fault injection: operator*
//       throws, mul_tiered returns the __int128 value); a result outside int64 throws HeckeOverflow; ForceNarrow
//       rethrows (the negative control);
//   [2] forced-wide = narrow on ell <= 23167: the tau tables of U_ell at ell in {5, 691, 5003} (n = 0; 23167 in the slow name) and of
//       T_2, T_3 at ell = 691 (n = 10), chains, B_tab and B entrywise equal under ForceWide and under Auto; every one
//       narrow (wide_tier() == false); Auto performs NO wide-product escalation on any of them;
//   [3] the boundary: ell = 23173 (the first prime past 23170) certifies in the wide tier (code 0, wide_tier());
//       under ExactTier::ForceNarrow it is refused (HeckeOverflow or code 51), never a value;
//   [4] wide only, against independent oracles, at n = 10 (weight 12, S_12 = C Delta, dim H^1 = 3):
//       (a) T_30011 mod 13: charpoly A^(1) = (X - tau(30011))^2 (X - (1 + 30011^11)) mod 13;
//       (b) U_ell at the narrow ell = 1009: charpoly = (X - tau(1009))^2 (X - 1) mod 1009 -- Eichler-Shimura for ell > n + 1
//           (H^1(Sym^n Z_ell) torsion-free, T_ell charpoly (X - tau(ell))^2 (X - (1 + ell^11)) reduced mod ell);
//       (c) U_30011 mod 30011, the same relation -- tau(p) from the q-expansion of Delta = q prod (1 - q^n)^24
//       (Jacobi's eta^3 series), computed here, so the oracle shares no code with the pipeline;
//   [5] the clause hecke::tau_bound_clause at its boundaries (0 / 51 / 52 at N = 23170, 23171, 1048575, 1048576, the
//       B_tab-dominated edge, B = UINT64_MAX), and the code-52 pre-flight at N = 2^20 (refused before the build);
//   [3w] (slow name) ForceWide = Auto entrywise on the wide table at ell = 23173 (the __int128 forms inside the pipeline);
// HECKE_WIDE_TAU_CELLS=<ell>[,...] (read-only, slow name only) REPLACES the slow arms by the U_ell tau tables (p = ell,
// n = 0) at the listed primes: one WIDE_TAU_CELL line each (certify, tier, B_tab, B, escalations) and an EFFECTIVE line.
// With HECKE_TAU_WIDE=0 (the rollback) arms [3] / [4](a,c) assert the pre-ELL-W refusal (code 51) instead.
uint32_t HeckeWideTauValidation();
uint32_t HeckeWideTauSlowValidation();

#endif // HECKE_WIDE_TAU_VALIDATION_H
