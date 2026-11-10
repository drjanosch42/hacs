// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_REDUCTION_H
#define HECKE_REDUCTION_H

// ===========================================================================
// (hecke-extensions slot 8, WP-4; E8 item 4a) THE REDUCTIONS AND THE BOCKSTEIN
// ON THE BULK.  Binding text: [E] v1.9.0-pre.1 sec:ring (c5) and (d)
// (docs/latex/hecke_extensions.tex), [X] Sec. 6.2 E8 item 4 (bulk half) with
// finding F3, [C] cl:bockstein; slot-8 plan WP-4 scope 1-3.
//
// Everything here is a map between the stored bases of objects the pipeline
// already computed -- the Z/ell^k objects of the module route (WP-3:
// EquivariantCohomologyOnKT::H_module_basis / H_class_coordinates /
// module_degree) and the F_ell objects of the field route (their D5b bases and
// the projector P_q).  Nothing recomputes a cohomology.
//
//  * red_H^q : H^q(Z/ell^k) -> H^q(F_ell), induced by Z/ell^k -> F_ell, as the
//    (c5) matrix from the cyclic-summand basis b_i of the ring object to the
//    D5b basis of the F_ell object at the same (n, ell): column j = the D5b
//    coordinates of the class of (b_j mod ell), i.e. P_q (b_j mod ell);
//    entries in [0, ell).  [E] sec:ring (d)'s tier-B object red_H^q.
//  * the level reduction H^q(Z/ell^{k+1}) -> H^q(Z/ell^k): column j = the (c5)
//    coordinates, on the lower object's b_i, of (b_j mod ell^k); row i reduced
//    into [0, ell^{e_i}) of the TARGET.
//  * the Bockstein beta^q : H^q(F_ell) -> H^{q+1}(F_ell) of
//    0 -> Z/ell -> Z/ell^2 -> Z/ell -> 0 ([C] cl:bockstein): a D5b class x
//    with cocycle z is lifted to an EQUIVARIANT cochain z~ over Z/ell^2 (z =
//    (C mod ell) c over F_ell, z~ = C c with c lifted to [0, ell), C the
//    free Howell basis of C^q_eq(Z/ell^2)); d z~ = ell w, and beta(x) is the
//    D5b class of w.  In C_eq coordinates d z~ is X_q c (module_degree), so no
//    flat coboundary is re-derived.
//
// None of these maps is an isomorphism in general: on cohomology the reduction
// is injective only after (x) F_ell and has a Tor cokernel (universal
// coefficients; [X] F3), so the certificates are MAP EQUALITIES,
//      A(T_p) o red = red o A(T_p),    beta o A(T_p) = A(T_p) o beta,
// which the *Equivariant functions below evaluate exactly.  A matrix
// reducing entrywise to another is never asserted (it is false).
//
// Every function throws std::logic_error (ENGINE_CHECK) on a shape or ring
// mismatch, and std::runtime_error where a step that must succeed does not
// (a lift whose coboundary is not divisible by ell, a class outside the span).
// ===========================================================================

#include <cstdint>
#include <vector>

#include "../EquivariantCohomologyOnK.h"
#include "../EquivariantMappingCone.h"   // (slot-9 WP-4) ModuleConeT, the F_ell cone
#include "../../ModArithmetic/DenseBuf.h"
#include "../../ModArithmetic/FlatMatrix.h"   // MatrixT (transitively; Matrix.h is retired from direct includes)
#include "../../ModArithmetic/ModCtx.h"

namespace hecke {

// red_H^q: dim H^q(F_ell) x t_q over F_ell (t_q = #cyclic summands of the ring
// object's H^q).  `ring` is over Z/ell^k (k >= 2), `field` over F_ell, both
// computed, at the same (K, n, ell).
template<class Elt>
engine::DenseBuf<Elt> ReduceHToField(const EquivariantCohomologyOnKT<Elt>& ring,
                                     const EquivariantCohomologyOnKT<Elt>& field, uint32_t q);

// H^q(Z/ell^{k_hi}) -> H^q(Z/ell^{k_lo}), k_hi > k_lo >= 2: t_lo x t_hi, row i
// in [0, ell^{e_i(lo)}).
template<class Elt>
engine::DenseBuf<Elt> ReduceHBetweenLevels(const EquivariantCohomologyOnKT<Elt>& hi,
                                           const EquivariantCohomologyOnKT<Elt>& lo, uint32_t q);

// beta^q: dim H^{q+1}(F_ell) x dim H^q(F_ell) over F_ell; `ring2` is over
// Z/ell^2 exactly.  q + 1 must be a degree of the complex.
template<class Elt>
engine::DenseBuf<Elt> BocksteinH(const EquivariantCohomologyOnKT<Elt>& ring2,
                                 const EquivariantCohomologyOnKT<Elt>& field, uint32_t q);

// The rank over F_ell (entries reduced mod ell first).
template<class Elt>
uint32_t RankOverField(Elt ell, const engine::DenseBuf<Elt>& M);

// [E] sec:ring (d): bockstein_rank^0 and higher_torsion_count := dim H^0(F_ell)
// - bockstein_rank^0 (so the bookkeeping identity holds by definition; its
// MEANING -- higher_torsion_count == #{e_i >= 2} -- is asserted by the
// validation wherever the K-mode certifies).
struct BocksteinBookkeeping {
    uint32_t dim_H0 = 0;
    uint32_t bockstein_rank0 = 0;
    uint32_t higher_torsion_count = 0;
};
template<class Elt>
BocksteinBookkeeping BocksteinCounts(const EquivariantCohomologyOnKT<Elt>& ring2,
                                     const EquivariantCohomologyOnKT<Elt>& field);

// ---- the certificates (map equalities, exact) -------------------------------
// A_f (dim_f x dim_f, the F_ell Hecke matrix in the D5b basis) o R ==
// R o (A_ring mod ell)  over F_ell, R = ReduceHToField.
template<class Elt>
bool ReductionToFieldEquivariant(Elt ell, const MatrixT<Elt>& A_field, const engine::DenseBuf<Elt>& A_ring,
                                 const engine::DenseBuf<Elt>& R);
// A_lo o R == R o A_hi, row i compared in [0, ell^{e_i(lo)}); the products are
// formed modulo ell^{k_hi} (every ell^{e(lo)} divides it).
template<class Elt>
bool ReductionBetweenLevelsEquivariant(const engine::ModCtx<Elt>& ctx_hi, const std::vector<uint32_t>& exps_lo,
                                       const engine::DenseBuf<Elt>& A_lo, const engine::DenseBuf<Elt>& A_hi,
                                       const engine::DenseBuf<Elt>& R);
// A_f^{(q+1)} o B == B o A_f^{(q)} over F_ell.
template<class Elt>
bool BocksteinEquivariant(Elt ell, const MatrixT<Elt>& A_next, const MatrixT<Elt>& A_this,
                          const engine::DenseBuf<Elt>& B);
// X o Y == 0 over F_ell (e.g. beta^q o red_H^q = 0, exactness at H^q(F_ell)).
template<class Elt>
bool ComposesToZero(Elt ell, const engine::DenseBuf<Elt>& X, const engine::DenseBuf<Elt>& Y);

// ===========================================================================
// (hecke-extensions slot 9, WP-4; E8 item 4b; [E] v1.10.0-pre.1 sec:ring (f))
// THE BOUNDARY AND THE CONE.  The boundary complex is an EquivariantCohomologyOnKT
// on K_bd, so ReduceHToField / ReduceHBetweenLevels / BocksteinH /
// BocksteinCounts above apply to it verbatim (bdry_red_H^q, the boundary
// Bockstein beta^0_bd).  The cone's Z/ell^k object is the dense module route
// ModuleConeT (EquivariantMappingCone.h); its F_ell object is the tape-route
// EquivariantMappingCone (uint32_t only), whose stored basis Hc^q and
// projector Pc^q the maps below read:
//  * cone_red_Hc^q : H^q_c(Z/ell^k) -> H^q_c(F_ell): column j = Pc^q (b_j mod ell);
//  * the level reduction H^q_c(Z/ell^{k+1}) -> H^q_c(Z/ell^k), as for the bulk;
//  * the cone Bockstein beta^q_c : H^q_c(F_ell) -> H^{q+1}_c(F_ell) of
//    0 -> Z/ell -> Z/ell^2 -> Z/ell -> 0, from the stored bases exactly as
//    BocksteinH (validated; NOT an object in slot 9, [E] (f)).
// Every entry point refuses at ell in {2, 3} by name (ModuleConeT is not
// constructible there; the guard here is explicit, slot-9 WP-3's merge note).
// ReductionIntertwines is the general map equality X_f o R_src = R_tgt o X
// over F_ell (the LES maps commuting with red; A o red = red o A is its
// square special case).
// ===========================================================================
// cone_red_Hc^q: dim H^q_c(F_ell) x t_q.  `field` is over F_ell at the same (n, ell).
engine::DenseBuf<uint32_t> ReduceConeHcToField(const ModuleConeT<uint32_t>& ring, const EquivariantMappingCone& field,
                                               uint32_t q);
// H^q_c(Z/ell^{k_hi}) -> H^q_c(Z/ell^{k_lo}), k_hi > k_lo >= 2.
template<class Elt>
engine::DenseBuf<Elt> ReduceConeHcBetweenLevels(const ModuleConeT<Elt>& hi, const ModuleConeT<Elt>& lo, uint32_t q);
// beta^q_c: dim H^{q+1}_c(F_ell) x dim H^q_c(F_ell); `ring2` over Z/ell^2 exactly.
engine::DenseBuf<uint32_t> BocksteinConeHc(const ModuleConeT<uint32_t>& ring2, const EquivariantMappingCone& field,
                                           uint32_t q);
// X_f (tf x sf, F_ell) o R_src (sf x s) == R_tgt (tf x t) o (X mod ell) (t x s), over F_ell.
template<class Elt>
bool ReductionIntertwines(Elt ell, const engine::DenseBuf<Elt>& X_field, const engine::DenseBuf<Elt>& R_src,
                          const engine::DenseBuf<Elt>& X_ring, const engine::DenseBuf<Elt>& R_tgt);

} // namespace hecke

#endif // HECKE_REDUCTION_H
