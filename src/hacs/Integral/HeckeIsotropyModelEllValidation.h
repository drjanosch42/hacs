// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_ISOTROPY_MODEL_ELL_VALIDATION_H
#define HECKE_ISOTROPY_MODEL_ELL_VALIDATION_H

#include <cstdint>

// hecke-extensions slot 16, WP-16 (stretch).  The ell-local isotropy model
// Iso^(3) of [C, def:isomodelell] (Integral/Model/IsotropyModelEll) against
// [C, rmk:isomodellcheckable] (IL1)-(IL4), the free model Iso of
// [C, def:isomodel] and the shipped Bredon route at ell = 3:
//   (IL0) the builder at ell = 0 (no localisation) IS IsotropyModel, entry by
//         entry over Z[Gamma]; the three StrictLiftBundle models agree;
//   (IL1) Iso^(3), every lift averaged by e_sigma: d o d = 0, IM1, IM2 and
//         well-definedness exactly over Z_(3)[Gamma]; the cell ranks of
//         [C, prop:ellthreeclosed](1); lengths and invariant factors of
//         H^q(Gamma; Sym^n (Z/3^e)^2) equal to Iso's (e <= 3, n <= 20,
//         q <= 5) and to [IB] Appendix A through UCT; the boundary sub-model;
//   (IL2) the closed form H^q = H^q(Z/3; M^{-I}), q >= 2, 2-periodic, over
//         F_3 (n <= 200) and Z/9, Z/27; parity: odd n gives 0 everywhere;
//   (IL3) over F_3, n <= 40: H^0 equal and dim H^1 = dim H^1_Bredon (the
//         shipped pipeline) + dim H^1(Z/3; M^{-I}); the Bredon row as a
//         subcomplex: iso on H^0, injective on H^1 with that cokernel (also
//         over Z/9);
//   (IL4) Phi_3, Psi_3 strict chain maps, quasi-inverse; the transported
//         T'_2, T'_3 = U_3, T'_5 have the characteristic polynomials of T_p on
//         Iso (StrictLift) over F_3, q <= 5;
//   (NC)  the negative control: Psi_3 built WITHOUT the e_sigma average is not
//         well defined (e_x Psi(x) != Psi(x) on every source generator with
//         Q_x != 1, the non-central Q_{v6} = <S> included -- the target Iso is
//         free, so not even the central -I is absorbed there) and its pullback
//         does not land in C^._{Iso^(3)}; it is still a chain map on generators
//         and strict, so only the C-1 check catches it.  The average is a no-op
//         on d itself at ell = 3 (every d-lift starts at s >= 1, Q = {+-I},
//         central and inside every target Q): reported and checked;
//   (MP)  ell = 2 through the same code: certificates and lengths = Iso's.
// Slot-17 WP-7 added the open halves (HeckeIsotropyModelEllOpenHalves.h): (IL3N)
// the level-N clause of (IL3), (IL4e) (IL4) over Z/3^e, (PC) the pair / cone half
// of prop:isomodelell(5), (REF) the synthetic refined descriptor of
// rmk:isomodellmp(3); env hook HECKE_ISOMODEL_ELL_FULL=1 (the heavier grid).
// Returns the number of failed checks.
uint32_t HeckeIsotropyModelEllValidation();

#endif
