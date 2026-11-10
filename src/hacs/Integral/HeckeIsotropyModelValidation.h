// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_ISOTROPY_MODEL_VALIDATION_H
#define HECKE_ISOTROPY_MODEL_VALIDATION_H

#include <cstdint>

// hecke-extensions slot 9, WP-10 (Line 2, L2-1a).  The isotropy-resolution
// model R of [C, def:isomodel] (Integral/Model/IsotropyModel.{h,cpp}) against
// [C, prop:isomodel] and the oracles of the slot-9 plan's "WP-10":
//   (A) the model descriptor against the shipped K and its 31 relations;
//   (B) the ranks w_q = 5, 14, 19, ... and w^d_q = 3, 6, 6, ... for n <= 40;
//   (C) d o d = 0 over Z[Gamma], R_d closed, eps a chain map (exact over Z);
//   (D) the model's rho_n equals the shipped ComputeSymmetricPowerMatrixFM;
//   (E) ell >= 5 of the golden grid, n <= 24: dim H^q(C_I (x) F_ell) = the
//       shipped pipeline's, q <= 5; eps^* a cochain map on the pipeline's own
//       C_eq, injective, and an isomorphism on H^0, H^1;
//   (F) ell in {2, 3}, n <= 20, q <= 5: dim H^q(C_I (x) F_ell) = Appendix A of
//       the integral-line brief through universal coefficients (O1 mod ell);
//       eps^* a cochain map, injective, on the Bredon complex;
//   (G) n = 0, ell = 2: dim H^q = 1, q = 0..5 (O2 mod 2);
//   (H) over Z/ell^k (composite ModCtx): lengths = Appendix A through UCT;
//       C_I(Z/ell^k) reduces entrywise to C_I(F_ell);
//   (I) R_d: H^q(C_{R_d}) = H^q(Gamma_d; V) on the cells where it is known.
// Returns the number of failed checks.
uint32_t HeckeIsotropyModelValidation();

#endif
