// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef ORBIFOLD_MP_BOUNDARY_VALIDATION_H
#define ORBIFOLD_MP_BOUNDARY_VALIDATION_H

#include <cstdint>

// (hecke-extensions slot 22, MP-3; MP_impl_plan.md v0.2.1 Sec. 2.5 / 2.6) OrbifoldMPBoundaryValidation (standard):
// the boundary MP set S(dX) on M7 and R_Z (|S_k|, (SC), d o d = 0), its cohomology = H^.(Gamma_d; Sym^n) over F_ell and
// Z/ell^k (Shapiro, [C, prop:mpboundary] (1)) against the frozen mp3_oracle.out and the shipped boundary model R_d of
// C_I; the labelled inclusion j : F'(dX) -> F'(X) with its label factor ([C, def:mpmorphism]); j^* a cochain map,
// rank of j^* on H^{<=3} = the restriction H^q(Gamma) -> H^q(Gamma_d) (C_I's, and the derived cells); the cone of j^*
// (the pair) exact against the LES and C_I's cone; lem:ell2unbounded at ell = 2.  NCs: a planted trivial horocycle
// stabiliser (wrong boundary stabiliser), a label off by T, the label factor dropped for mixed labels.
// Gate values: docs/implementation/slot22/MP3_predictions.md P5 (committed before the first run).  Read-only.
uint32_t OrbifoldMPBoundaryValidation();

#endif
