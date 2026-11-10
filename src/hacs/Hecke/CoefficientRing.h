// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_COEFFICIENT_RING_H
#define HECKE_COEFFICIENT_RING_H

// ===========================================================================
// (hecke-extensions slot 8, WP-3; E8 item 3) THE COEFFICIENT RING IN FORCE, AND
// THE REFUSAL RULE.  Binding text: [E] v1.9.0-pre.1 sec:ring (a)-(b)
// (docs/latex/hecke_extensions.tex), ENGINE_API.md Sec. 13.8 (1).
//
// `coefficient_ring` (Hecke/HeckeSettings.h) is a mathematical setting: at
// its default `F_ell` nothing in this file changes anything.  Under
// `Z/ell^k` (k >= 2):
//
//  * the two objects that HONOUR the ring -- EquivariantCohomologyOnK and
//    HeckeOperatorOnK -- read it ONCE, at construction, from the settings in
//    force on the constructing thread, and keep the context
//    (CoefficientRingContext below) for their lifetime; a local system over
//    Z/ell^k is built by GenerateSL2ZRelationsRing (the composite rho_n,
//    FlatMatrix.h's ComputeSymmetricPowerMatrixRing);
//  * every other construction site and every field-only consumer REFUSES, by
//    one constructor-entry RequireFieldCoefficientRing(...) raising
//    hecke::NotImplementedInThisBuild whose text names the class and the
//    canonical value ([E] sec:ring (b2)): "<Class>: coefficient_ring=
//    Z/ell^<k> is not honoured in this build".  A refusing member also
//    refuses an INPUT object that was built over a composite ring (an
//    object's ring is its own, whatever is in force when it is consumed).
//
// An honouring object refuses by the same named error where it cannot honour
// the value ([E] sec:ring (b3)): wherever ell^k >= 2^64 (ENGINE_API.md Sec.
// 13's scope) or ell^k does not fit the element type; and, except for
// EquivariantCohomologyOnK (slot-9 WP-3, decision 4 (A): the Bredon object by
// the flat-coordinate route, lengths and invariant factors only), at ell in
// {2, 3} (6 is not a unit, so the model complex is not C (x) Z/ell^k).
// ===========================================================================

#include <cstdint>
#include <string>
#include <vector>

#include "ExactInt.h"
#include "HeckeSettings.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"
#include "../../ModArithmetic/FlatMatrix.h"
#include "../../ModArithmetic/ModCtx.h"

namespace hecke {

// CoefficientRingK / CoefficientRingText / RefuseCoefficientRing /
// RequireFieldCoefficientRing -- the refusal rule's guard -- live in
// HeckeSettings.h (STL-only), so a refusing member includes nothing more.

// (slot-9 WP-3; decision 4 (A)) Whether a caller may hold a context at
// ell in {2, 3}.  `Refuse` (the default, every caller but one) keeps slot 8's
// refusal.  `Bredon` is passed by EquivariantCohomologyOnK's constructor ONLY:
// there the object is the Bredon one over Z/ell^k ([C] def:ringsemantics(2),
// prop:bredon(2)-(3)), computed by the flat-coordinate module route, and every
// consumer that is not honoured there (the K-mode accessors, HeckeOperatorOnK,
// HeckeReduction through the stored-basis accessors) refuses by name itself.
enum class SmallEllPolicy : uint32_t { Refuse = 0, Bredon = 1 };

// ell in {2, 3}: the primes at which 6 is not a unit of Z/ell^k.
inline bool IsBredonPrime(uint64_t ell) { return ell == 2 || ell == 3; }

// The context an HONOURING object keeps: ModCtx::prime_power(ell, k) for
// k >= 2.  Refuses (NotImplementedInThisBuild, naming `who`) at ell in {2, 3}
// unless `small_ell == SmallEllPolicy::Bredon`, at ell^k >= 2^64, and where
// ell^k does not fit Elt (the uint64_t instantiation honours it there).
// k == 1 is a caller error (ENGINE_CHECK).
template<class Elt>
engine::ModCtx<Elt> CoefficientRingContext(const std::string& who, uint64_t ell, uint32_t k,
                                           SmallEllPolicy small_ell = SmallEllPolicy::Refuse);

// The exact integer k-th root of m when m = ell^k with ell a PRIME, else 0.
// (slot-11 WP-12, F-2) Primality by IsPrimeU64 (was: trial division up to
// sqrt(ell)); every value unchanged.  A consumer that knows the ring should
// carry it as (ell, k) instead -- LocalSystemSimplicialComplexT's declared
// ring -- and not derive ell from the modulus at all.
uint64_t ExactKthRoot(uint64_t m, uint32_t k);

// (slot-11 WP-12, F-2) Deterministic Miller-Rabin primality for every
// uint64_t (the twelve-prime base set), O(log n) modular multiplications.
bool IsPrimeU64(uint64_t n);

// rho_n(g) mod ell^k of an EXACT integer matrix (reduced mod ell^k first,
// then FlatMatrix.h's composite route) -- the composite rho_of_exact.
template<class Elt>
FlatMatrixT<Elt> RhoOfExactRing(const engine::ModCtx<Elt>& ctx, const Mat22& g, uint32_t n_sym);

// The 7-vertex model's relation list over Z/ell^k: the SAME rows as
// GenerateSL2ZRelationsElt(K, n_sym, ell) -- the 8 seeds, then the FIX-negI
// self-relations, with the same q_dim / s_lhs / s_rhs / tag -- and only the
// matrix replaced by rho_n(g_r) mod ell^k (the seeds' exact g_r; -I for the
// FIX-negI rows).  The FlatMatrix modulus of every relation is ell^k, so
// LocalSystemSimplicialComplexT<Elt>(K, rels, n_sym + 1, ctx.p) is the local
// system EquivariantCohomologyOnK honours the ring on.
template<class Elt>
std::vector<typename LocalSystemSimplicialComplexT<Elt>::SimplexRelation>
GenerateSL2ZRelationsRing(const SimplicialComplex& K, uint32_t n_sym, const engine::ModCtx<Elt>& ctx);

// (slot-13 WP-5; [C] def:hidalattice variant (H), rmk:stabconvention(3)) Hida's
// lattice L_n: rho^H_n(g) = rho_n(g^T)^T mod ell^k (the transposed exact matrix
// through RhoOfExactRing, then transposed), and the 7-vertex model's relation
// list over Z/ell^k with rho^H_n in place of rho_n -- the SAME rows, order,
// tags as GenerateSL2ZRelationsRing (rho^H_n(-I) = rho_n(-I) = (-1)^n).  Used by
// HeckeExt/HeckeHidaTower's arm (b); HeckeOperatorOnK's module route honours
// FibreRepresentation::HidaLatticeH on a local system built from it.
template<class Elt>
FlatMatrixT<Elt> RhoHidaOfExactRing(const engine::ModCtx<Elt>& ctx, const Mat22& g, uint32_t n_sym);
template<class Elt>
std::vector<typename LocalSystemSimplicialComplexT<Elt>::SimplexRelation>
GenerateSL2ZRelationsRingHida(const SimplicialComplex& K, uint32_t n_sym, const engine::ModCtx<Elt>& ctx);

// (slot-9 WP-4; E8 item 4b; [E] v1.10.0-pre.1 sec:ring (f)) The circle model's
// relation list over Z/ell^k: the SAME rows as GenerateSL2ZBoundaryRelationsElt
// (K_bd, n_sym, ell) -- the cusp relation f(v3) = rho_n(T) f(v0), then one
// FIX-negI self-relation per simplex of K_bd, same q_dim / s_lhs / s_rhs / tag
// -- with every matrix replaced by rho_n(g) mod ell^k (g = T, resp. -I).  The
// boundary complex of SL2ZBoundaryPair over Z/ell^k is built on it.
template<class Elt>
std::vector<typename LocalSystemSimplicialComplexT<Elt>::SimplexRelation>
GenerateSL2ZBoundaryRelationsRing(const SimplicialComplex& K_bd, uint32_t n_sym, const engine::ModCtx<Elt>& ctx);

} // namespace hecke

#endif // HECKE_COEFFICIENT_RING_H
