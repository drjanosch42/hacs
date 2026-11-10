// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef ORBIFOLD_MP_BOUNDARY_H
#define ORBIFOLD_MP_BOUNDARY_H

// ===========================================================================
// (hecke-extensions slot 22, MP-3) The boundary MP set S(dX) and the boundary
// inclusion j : F'(dX) -> F'(X) ([C] 1.22.0-pre.6 prop:mpboundary;
// MP_math_plan.md v0.2.1 Def. 1.7, Sec. 4.4, Sec. 7.4; MP_impl_plan.md v0.2.1
// Sec. 2.5 "Boundary").
//
// THE DESCRIPTOR.  BoundaryDescriptor(d, D) is D (the descriptor of the
// Gamma-triangulation X, MP-1a) with Sigma_top replaced by the boundary edge
// representatives of the ModelData (bdry[1]: v0v1, v1v2, v2v3 on M7 and R_Z) --
// the top simplices of the 1-dimensional Gamma-complex dX.  The vertex orbits,
// stabilisers and points are D's (the horocycle vertices have stabiliser {+-I}).
// InternedMPSet on it is S(dX) of [C, def:mpset] for dX: its cohomology is
// H^.(Gamma_d; M) by Shapiro (prop:mpboundary (1): (AC) fails, N Z[F'(dX)]
// resolves Z[Gamma/Gamma_d]).  `planted_trivial_stabiliser` (VALIDATION ONLY, the
// negative control "wrong boundary stabiliser") replaces the stabiliser of every
// horocycle vertex orbit by {I}: the -I arrows disappear and H^{>=2} at ell = 2
// is lost.
//
// THE LABELS.  For each boundary edge e: the top simplices sigma of X and the
// labels h with iota(e) <= h iota(sigma), decided on point keys (Descriptor::key)
// over the finite candidate set {g_p u g_w^{-1}} (p a vertex of e, w a vertex of
// sigma in the same orbit, u in the stabiliser).  On X_bulk the geometric
// simplex is unique and the labels form one coset h_0 Gamma_sigma = {h_0, -h_0};
// BoundaryLabels takes the least GKey (as Refinement.h does), and the
// validation checks uniqueness.  j^* is LabelledPullbackRep with L(h) = rho(h)
// -- the label factor of [C, def:mpmorphism] (3).
// ===========================================================================

#include <cstdint>
#include <utility>
#include <vector>

#include "Descriptor.h"
#include "Refinement.h"
#include "../../Integral/Model/IsotropyModel.h"

namespace orbifold_mp {

Descriptor BoundaryDescriptor(const integral_model::ModelData& d, const Descriptor& D, bool planted_trivial_stabiliser = false);

// Every (sigma, h) with iota_sub(t) <= h iota_big(sigma) (sub and big share the vertex orbits of one Gamma).
std::vector<std::pair<uint32_t, Mat22>> ContainingLabels(const Descriptor& Dsub, uint32_t t, const Descriptor& Dbig);
// Is iota_sub(t) <= h iota_big(sigma)?
bool ContainedInSimplex(const Descriptor& Dsub, uint32_t t, const Descriptor& Dbig, uint32_t sigma, const Mat22& h);

// The labels of j : dX -> X (least GKey; ENGINE_CHECK: one geometric simplex per boundary edge).
RefinementLabels BoundaryLabels(const Descriptor& Dbdry, const Descriptor& D);

}  // namespace orbifold_mp

#endif
