// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_HECKE_INVOLUTION_VALIDATION_H
#define HECKE_EXT_HECKE_INVOLUTION_VALIDATION_H

#include <cstdint>

// hecke-extensions slot 4, WP-3 (phase E3) -- the production involution iota^*
// (Hecke/HeckeInvolution) and the sign split (HeckeExt/HeckeSignSplit).
//
// THE ORACLE.  The production iota^* is compared ENTRYWISE against the
// involution HeckeIotaSignValidation builds from [C, def:iota] alone, now
// exported as hecke::iota_oracle (slot-4 WP-3; HeckeIotaSignValidation's own
// printed lines are byte-identical before and after that export).  The two
// constructions are independent: the production object re-derives iota_K from
// the shipped exact vertex coordinates, recomputes every orientation sign, and
// forms rho_n(iota) through the live ComputeSymmetricPowerMatrixFM.
//
// ARMS
//  A1  the 77-cell grid of HeckeIotaSignValidation (n even in 2..30,
//      ell in {5,7,11,13,101,134217757,2^31-1}): production iota_C^{(q)} ==
//      oracle, geometric == oracle geometric, iota^* == -iota^*_geo, and
//      P_q iota_C H_eq == the production iota^* on H^q -- all entrywise, in
//      every degree; iota^2 = I and iota d = d iota on the MATERIALISED
//      matrices; [iota, A^{(1)}(T_p)] = 0 for p in {2,3,5}; the +- split and
//      dim H^1_+- against (d_{n+2}, d_{n+2}-1) off the Dickson locus; and the
//      Eisenstein sign by route (a) (the joint T_p-eigenline of eigenvalue
//      1 + p^{n+1}) and route (b) (the boundary-visible class through the
//      production restriction), reproducing
//      "eisenstein=plus (geometric=minus)" FROM THE PRODUCTION OBJECT.
//  A2  (10, 691): dim H^1_+- = (2, 1), certificate = cyclic on both signs,
//      the Eisenstein line soc(X^+_{4,m}) = m_S X^+_{4,m} DEFINED and in the
//      bottom Loewy layer, and on the unsplit H^1 the socle 2-dimensional so
//      the line is "undefined" ([E] D6).
//  A3  the unsplit decomposition's blocks are the direct sum of the per-sign
//      blocks, block by block (E3 acceptance (c)).
//  A4  compatibility: r^q iota^* = iota^*_bdry r^q (asserted);
//      iota_c := iota^* (+) iota^*_bdry on Cone(r) with D_c iota_c = iota_c D_c
//      (asserted), iota_c on H^q_c, and dim H^q_{c,+-} RECORDED against the
//      mirror split of [C, rmk:iotapairing] (audit F13); [iota_c, A_c^{(q)}]
//      RECORDED, never asserted -- it needs iota h = h iota on the stored
//      C_eq, which no [C] statement provides.
//  A5  a tamper arm: a flipped orientation sign on one simplex fails
//      [C, obl:iota](i) exactly, and +iota^*_geo in place of iota^* fails the
//      degree-0 anchor exactly.
//  A6  the small-ell cells the 77-cell grid does not have: ell = 3 (the
//      commutation RECORDED, [C, prop:iotacommutes](3c) unavailable) and
//      ell = 2 (sign_split = not_applicable, [C, prop:iotasplit](2)'s
//      replacement statement recorded).
//
// Prints IOTA_PRODUCTION ... and IOTA_SIGN_PRODUCTION ... .
// Returns 0 on success; nonzero = number of failing checks.  Standard tier.
uint32_t HeckeInvolutionValidation(void);

#endif // HECKE_EXT_HECKE_INVOLUTION_VALIDATION_H
