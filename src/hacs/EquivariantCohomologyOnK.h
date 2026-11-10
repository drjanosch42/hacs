// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#ifndef EQUIVARIANT_COHOMOLOGY_ON_K_H
#define EQUIVARIANT_COHOMOLOGY_ON_K_H

#include "LocalSystemSimplicialComplex.h" // Includes SimplicialComplex, FlatMatrix
#include "../ModArithmetic/helper.h"
#include "../ModArithmetic/BlockMat.h"
#include "../ModArithmetic/DenseElim.h"   // (slot-8 WP-3) the Z/ell^k module results
#include "../ModArithmetic/EngineCheck.h"

#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept>
#include <sstream>
#include <memory>

// Stage S13: templated on the coefficient element type Elt. The default alias
// EquivariantCohomologyOnK = EquivariantCohomologyOnKT<uint32_t> is byte-
// identical to the pre-S13 class; EquivariantCohomologyOnK64 supports the
// extended coefficient regime ell < 2^64.
//
// ---------------------------------------------------------------------------
// (matrix-engine Phase 3, Stage B) THE BLOCK MODEL IS THE STORAGE.
//
// compute_cohomology() runs on engine::BlockMat / engine::blk (ENGINE_API.md
// Sec. 9).  Per degree q, with the uniform row partition (N_q blocks of height
// d_V) -- which IS the binding L1 layout s*d_V + c (audit 02 Sec. 6.4, plan
// item 7.20; BlockMat::to_flat / from_flat are its single source of truth):
//
//   A_Cq   N_rel_q x N_q block matrix, kinds I / -M / I-M   (BlockBuilder, one
//          ref per block, every rho_n(g) interned once by content)
//   C_eq   ker A_Cq                 = blk::kernel_basis     (dim_q x dim C_eq)
//   A_Zq   N_rel_q x k_q pencils beta_lhs*I - beta_rhs*M
//   Z_eq   (beta (x) I) * ker A_Zq  = one blk::gemm_dense   (dim_q x dim Z_eq)
//   W^{q-1} = C^{q-1}/Z^{q-1}       = blk::factor_basis
//   B_eq   (D_1 (x) I) * W^{q-1}    = one blk::gemm_dense   (dim_q x dim B_eq)
//   H_eq   Z^q/B^q, P_q             = blk::factor_basis (basis + projector)
//
// Every basis is a MULTI-VECTOR (a BlockMat with nbc == 1, column t = one
// cochain in L1 order); P_q is a 1 x N_q block matrix (dim H^q x dim_q).  The
// VALUES are those of the legacy path bit for bit (D5b; golden tier B pins
// C/Z/B/H/P entrywise): the block elimination reproduces the legacy basis
// choices, and the two GEMMs are exact sums of residues.
//
// The pre-Stage-B path (BlockFlatMatrix::flat_kernel + factor_basis_block +
// the flat/unflat/expand loops) is kept as the LEGACY arm (since S24 CL-L2 in
// legacy/EquivariantCohomologyOnKLegacy.inc, compiled only with OC_ENABLE_LEGACY=ON):
//   * ENGINE_BACKEND=legacy (engine::backend_is_legacy()) runs it instead and
//     rebuilds the block objects from its output, so every accessor works
//     under both backends;
//   * -DENGINE_SHADOW_LEGACY=ON (needs -DOC_ENABLE_LEGACY=ON, S24 CL-L1;
//     legacy/EquivariantCohomologyOnKShadow.inc) runs BOTH and compares every
//     stored object (dimensions, all five basis families, projectors)
//     entrywise, throwing std::logic_error on the first difference.
//
// ACCESSORS.  The block objects are the storage and are handed out by
// reference (`*_block(q)`, `block_pool()`).  The FlatMatrix-typed accessors
// (`get_*_bases_fm()`, `get_H_eq_projectors_from_C_eq_flat_fm()`) still
// return `const&` to a per-degree list of N_q x d_V cochains / row-major
// projectors with exactly the legacy shapes and values; that list is
// MATERIALISED LAZILY (once, on first call, via to_flat) and cached, so a
// consumer that never asks for it pays nothing.  The deprecated MatrixT
// accessors copy from that cache on every call.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// (hecke-extensions slot 8, WP-3; E8 item 3; [E] v1.9.0-pre.1 sec:ring) THE
// COEFFICIENT RING.  The object reads `coefficient_ring` (Hecke/HeckeSettings.h)
// ONCE, in its constructor, from the settings in force on the constructing
// thread, and never re-reads it:
//
//  * `F_ell` (the default): nothing below changes -- the block / tape route
//    and the legacy arm run exactly as before, and every accessor above is
//    what it was;
//  * `Z/ell^k` (k >= 2): the local system must be the model's relation list
//    over Z/ell^k (hecke::GenerateSL2ZRelationsRing; its modulus is ell^k,
//    whose exact k-th root is the cell's ell) -- otherwise, and at ell^k
//    beyond the element type, the constructor raises
//    hecke::NotImplementedInThisBuild naming this class.  (Since slot-9 WP-3,
//    at ell in {2, 3} the object is the Bredon one and compute_cohomology()
//    takes the FLAT-COORDINATE route instead -- see the accessors at the end
//    of the class.)  At ell >= 5 compute_cohomology()
//    then runs the MODULE route (compute_cohomology_module()) on the dense
//    Sec. 13.6 entry points with the composite context ModCtx::prime_power:
//      C^q_eq = kernel_module(A_Cq)      A_Cq assembled on a COMPOSITE BlockMat
//                                        (BlockBuilder's ModCtx overload, Sec.
//                                        13.7); asserted free (every Howell
//                                        pivot a unit), its Howell rows the basis
//      X_q    = coordinates of (D (x) I) C^q_eq in the C^{q+1}_eq basis
//               (solve_multi_rhs_module) -- d^q in equivariant coordinates
//      Z^q    = kernel_module(X_q)   (C^q_eq coordinates)
//      B^q    = the columns of X_{q-1}
//      H^q    = Z^q / B^q = cyclic_decomposition(ctx, Z^q, B^q) -- the
//               invariant factors with [E] sec:ring (c)'s basis rule, read in
//               C^q_eq COORDINATES (the (G, Rl) of (c1)); H_module_basis(q) is
//               that basis mapped to flat cochains.
//    and the K-mode of [E] sec:ring (d): r_K(d^q) = the number of non-zero
//    Smith diagonal entries of X_q over Z/ell^K (K = k) = the number of
//    invariant factors of colspan X_q; certified iff r_K(d^0) = rank C^0_eq
//    and r_K(d^1) = rank C^2_eq.  Over Z/ell^k every FIELD accessor (the
//    *_block, *_fm and MatrixT accessors, get_H_eq_dimensions(),
//    C_eq_dimension()) throws a named std::logic_error: there is no `dim`,
//    `length` replaces it ([E] sec:ring (b3)).
//
// The protected (prime, rep_dim) constructor (the F_ell cone's base) REFUSES a
// non-default ring.  The relative sub-complex constructor HONOURS Z/ell^k at
// ell >= 5 through the module route (slot-10 WP-3: the cone's route 3 over the
// ring) and refuses by name at ell in {2, 3}.
// ---------------------------------------------------------------------------
template<class Elt>
class EquivariantCohomologyOnKT {
public:
    using BlockMat  = engine::BlockMat<Elt>;
    using BlockPool = engine::BlockPool<Elt>;

    // (slot-8 WP-3) One degree of the Z/ell^k result (the module route).
    struct ModuleDegree {
        engine::DenseBuf<Elt> C;        // dim_q x c_q: the C^q_eq basis (flat cochains; Howell rows of ker A_Cq)
        std::vector<uint32_t> piv;      // (S23 G2-A) the unit Howell pivots of C: C[piv, :] = I (empty: non-unit branch)
        engine::DenseBuf<Elt> X;        // c_{q+1} x c_q: d^q in C_eq coordinates (0 rows at the top degree)
        engine::DenseBuf<Elt> Zc;       // c_q x z: Z^q generators in C_eq coordinates (Howell rows of ker X_q)
        // (S23 G2-E; audit n1) X and Zc are INVALID (0 x 0, not "no rows") after release_module_intermediates():
        // outside compute_cohomology_module() read them only through module_X(q) / module_Zc(q), which throw then.
        engine::DenseBuf<Elt> H;        // dim_q x t: the cyclic-summand basis b_i as flat cochains
        engine::elim::CyclicDecomposition<Elt> dec;   // H^q = Z^q / B^q in C_eq coordinates ([E] sec:ring (c))
        std::vector<uint32_t> dK;       // invariant_factors(colspan X_q) over Z/ell^K
        uint32_t length_H = 0;          // sum of dec.exponents
    };
    // (slot-9 WP-3; decision 4 (A)) One degree of the FLAT-COORDINATE module
    // route, taken at ell in {2, 3} only, where C^q_eq is not free (the header
    // note).  Everything is in flat cochain coordinates (dim_q = N_q * d_V
    // rows); no basis of H^q is chosen and no K-mode object exists.
    struct FlatModuleDegree {
        engine::DenseBuf<Elt> C;        // dim_q x c: Howell generators of C^q_eq = ker A_Cq (NOT a basis)
        engine::DenseBuf<Elt> Z;        // dim_q x z: Howell generators of Z^q = ker [A_Cq ; d^q]
        engine::DenseBuf<Elt> B;        // dim_q x b: d^{q-1} applied to the columns of C^{q-1}_eq's C
        std::vector<uint32_t> exponents;   // invariant_factors(ctx, Z, B): H^q ~= (+)_i Z/ell^{e_i}, descending
        uint32_t length_C = 0, length_Z = 0, length_B = 0, length_H = 0;   // composition lengths
    };
    // (slot-8 WP-3) [E] sec:ring (d)'s ladic_type^q: the Z_(ell)-type
    // (free rank; torsion exponents, descending) of H^q(C)_(ell).
    struct LadicType {
        uint32_t free_rank = 0;
        std::vector<uint32_t> torsion;
    };
    // (slot-8 WP-3) The R3-shaped state of an UNCERTIFIED level ([E] sec:ring
    // (d)): kind `ladic_level`, budget {K}, reached {r_K(d^0), r_K(d^1)},
    // complete = false.  Its continuation is a FRESH run at a larger k, never
    // `resume` (coefficient_ring is a non-budget setting).
    struct LadicLevelState {
        std::string kind = "ladic_level";
        uint32_t budget_K = 0;
        uint32_t reached_r0 = 0, reached_r1 = 0;
        uint32_t need_r0 = 0, need_r1 = 0;
        bool complete = false;
    };

    // The legacy-shaped (FlatMatrix) representation of one complete result:
    // per degree, the N_q x d_V cochains of C/Z/B/H and the row-major
    // projector P_q.  Produced directly by the legacy arm; materialised on
    // demand from the block objects by the engine arm.
    struct FlatResult {
        std::vector<std::vector<FlatMatrixT<Elt>>> C, Z, B, H;
        std::vector<FlatMatrixT<Elt>> P;
        std::vector<uint32_t> dims;
        bool built = false;
    };

private:
    const LocalSystemSimplicialComplexT<Elt>* lssc; // Pointer to the local system definition on K
    Elt prime;
    uint32_t representation_dim; // d_V

    // ---- Block-model storage (the result) -------------------------------
    std::shared_ptr<BlockPool> pool_;              // one pool per cohomology run
    std::vector<BlockMat> A_C_block_;              // N_rel_q x N_q constraint matrices
    std::vector<BlockMat> A_Z_block_;              // N_rel_q x k_q pencil matrices
    std::vector<BlockMat> C_eq_block_;             // multi-vectors dim_q x dim C^q_eq
    std::vector<BlockMat> Z_eq_block_;             // multi-vectors dim_q x dim Z^q_eq
    std::vector<BlockMat> B_eq_block_;             // multi-vectors dim_q x dim B^q_eq
    std::vector<BlockMat> H_eq_block_;             // multi-vectors dim_q x dim H^q_eq
    std::vector<BlockMat> P_block_;                // 1 x N_q block matrices, dim H^q x dim_q
    std::vector<uint32_t> H_eq_dimensions;         // Dimension of H^q_eq(K,R;L)
    std::vector<uint32_t> C_eq_dims_;              // dim C^q_eq, kept even when C_eq_block_[q] is dropped (lean)

    // ---- Lazy FlatMatrix materialisation for the compatibility accessors --
    mutable FlatResult fm_;
    void ensure_fm_cache() const;

    bool cohomology_computed;
    bool intermediates_retained_ = true;   // see EquivariantCohomologyOptions::retain_intermediates

    // (hecke-extensions slot 6, WP-5) The RELATIVE sub-complex (see the
    // constructor below).  Empty / false on every other object, in which case
    // effective_relations() IS lssc->get_user_defined_simplex_relations() --
    // the same reference, so the default path reads exactly what it read
    // before this package.
    using SimplexRelationT = typename LocalSystemSimplicialComplexT<Elt>::SimplexRelation;
    bool relative_ = false;
    std::vector<std::vector<uint32_t>> vanishing_;          // [q] K-indices of the forced-zero q-simplices
    std::vector<SimplexRelationT> relative_relations_;      // lssc's relations ++ one zero relation per forced simplex
    const std::vector<SimplexRelationT>& effective_relations() const;

    // (slot-8 WP-3) The coefficient ring, fixed at construction.
    uint32_t ring_k_ = 1;                          // 1 = F_ell
    engine::ModCtx<Elt> ring_ctx_;                 // prime_power(ell, k) when ring_k_ >= 2
    std::vector<ModuleDegree> module_;             // the Z/ell^k result, per degree
    // (slot-9 WP-3) ell in {2, 3} under Z/ell^k: the flat-coordinate route.
    bool flat_route_ = false;
    bool module_released_ = false;   // (S23 G2-E) release_module_intermediates() took effect
    bool module_early_release_ = false;   // (S24 G2-H) set_module_early_release(true): free X / Zc inside the H^q loop
    std::vector<FlatModuleDegree> flat_;           // its result, per degree (module_ stays empty)
    void init_coefficient_ring();                  // reads the setting; composite: refuses or sets ring_ctx_
    void compute_cohomology_module();
    void compute_cohomology_flat_module();         // (slot-9 WP-3) the ell in {2, 3} branch
    void require_free_route(const char* what) const;   // refuses a free-route / K-mode accessor at ell in {2, 3}
    void require_field(const char* what) const;    // a field accessor on a composite object: logic_error
    void require_module(const char* what, uint32_t q) const;

    // The two arms of compute_cohomology().
    void compute_cohomology_block();
#if OC_ENABLE_LEGACY
    // (S24 CL-L2) legacy/EquivariantCohomologyOnKLegacy.inc -- ON builds only.
    void compute_cohomology_legacy(FlatResult& out) const;
    void adopt_legacy_result(FlatResult&& legacy);
#endif
#ifdef ENGINE_SHADOW_LEGACY
    void shadow_compare_with_legacy();   // (S24 CL-L1) legacy/EquivariantCohomologyOnKShadow.inc
#endif
    void check_degree(uint32_t q, const char* what) const;

public:
    // Constructor
    EquivariantCohomologyOnKT(const LocalSystemSimplicialComplexT<Elt>* local_system_complex);

    // (hecke-extensions slot 6, WP-5; [C, prop:coneroutes](2), "route 3") The
    // RELATIVE SUB-COMPLEX constructor -- additive: no existing constructor,
    // path or value changes.  vanishing_simplices[q] lists K-indices of
    // q-simplices (sorted, distinct, in range; degrees beyond the list's size
    // force nothing) on which every cochain is FORCED TO ZERO, so that
    //
    //     C^q_rel = { f in C^q_eq(K, Rel; L) : f(sigma) = 0 for sigma in vanishing_simplices[q] }.
    //
    // For the boundary simplices of a pair this is ker r = Hom_{Z[Gamma]}(C(X, X_bd), V)
    // ([C, lem:conerelative](3)), a sub-COMPLEX because r d = d r; the
    // constructor does not check that the given set is a subcomplex -- the
    // caller's route certificate does (a non-subcomplex set would make B^q
    // leave Z^q and factor_basis refuse).  Mechanism: each forced simplex adds
    // ONE relation f(sigma) = 0 * f(sigma) to the relation list this object
    // (and only this object) reads -- its A_C row is the block I, its A_Z
    // pencil beta(sigma) I -- so the SAME two arms (block and legacy) compute
    // C/Z/B/H/P of the sub-complex with their unchanged code.  The zero
    // relation is never vacuous (M = 0 != I).  get_lssc() still returns the
    // UNCHANGED local system: a consumer that re-reads relations from it
    // (HeckeOperatorOnK, HeckeInvolution, ...) must not be built on a
    // relative object, and none is.
    EquivariantCohomologyOnKT(const LocalSystemSimplicialComplexT<Elt>* local_system_complex,
                              const std::vector<std::vector<uint32_t>>& vanishing_simplices);

    // Protected constructor for inheritance purposes (e.g., by the Cone class)
    EquivariantCohomologyOnKT(Elt prime, uint32_t rep_dim);

    // Destructor
    ~EquivariantCohomologyOnKT() = default;

    // --- Computation ---
    // Main method to perform the entire cohomology computation
    virtual void compute_cohomology();

    // --- Getters for Results ---
    const LocalSystemSimplicialComplexT<Elt>* get_lssc() const { return lssc; }
    // (hecke-extensions slot 9, WP-9; slot-8 WP-12 review, m-10) `prime` HOLDS
    // ell^k on a composite (coefficient_ring = Z/ell^k) object -- it is the
    // FlatMatrix modulus this object was built over, not a prime -- so a
    // field-only caller that reaches here on one is one `get_prime()` call
    // away from silently treating a composite modulus as a field
    // characteristic.  Loud failure instead: every in-tree call site is
    // field-only-guarded upstream (RequireFieldCoefficientRing / an
    // init_composite() short-circuit) or reads the ring-aware ring_context().p
    // instead ([X] open item 9).
    Elt get_prime() const {
        ENGINE_CHECK_MSG(!is_composite(),
                         "EquivariantCohomologyOnK::get_prime(): called on a composite "
                         "(coefficient_ring != F_ell) object -- use ring_context().p or "
                         "coefficient_ring_k() instead");
        return prime;
    }
    uint32_t get_representation_dim() const { return representation_dim; }
    bool is_cohomology_computed() const { return cohomology_computed; }
    // (slot-6 WP-5) Whether this is a relative sub-complex, and its forced-zero sets.
    bool is_relative() const { return relative_; }
    const std::vector<std::vector<uint32_t>>& vanishing_simplices() const { return vanishing_; }

    // --- Block-model accessors (matrix-engine Phase 3 Stage B) --------------
    // All throw std::runtime_error before compute_cohomology() and on q out of
    // range.  Row partition of every multi-vector and of P_q's columns is the
    // uniform (N_q, d_V) one: `block_row_dims(q)`; the flat row index of fibre
    // component c on simplex s is s*d_V + c (L1).
    //
    //   A_C_block(q)  N_rel_q x N_q     kinds I / -M / I-M (vacuous rows dropped
    //                                   when the item-7.4 short-circuit is on)
    //   A_Z_block(q)  N_rel_q x k_q     pencils beta_lhs*I - beta_rhs*M
    //   C_eq_block(q) dim_q x dim C^q   ker A_Cq          (nbc == 1)
    //   Z_eq_block(q) dim_q x dim Z^q   (beta (x) I) ker A_Zq
    //   B_eq_block(q) dim_q x dim B^q   (D_1 (x) I) W^{q-1}   (0 columns at q = 0)
    //   H_eq_block(q) dim_q x dim H^q   the Z^q columns chosen by factor_basis
    //   P_block(q)    dim H^q x dim_q   projector C^q_flat -> H^q coordinates,
    //                                   grid {dim H^q} x (N_q, d_V)
    // The dense views: blk::multivector_view(C_eq_block(q), scratch) or
    // `.to_flat()`; P_q acts by blk::gemm_dense(P_block(q), y, out).
    const std::shared_ptr<BlockPool>& block_pool() const;
    std::vector<uint32_t> block_row_dims(uint32_t q) const;
    const BlockMat& A_C_block(uint32_t q) const;
    const BlockMat& A_Z_block(uint32_t q) const;
    const BlockMat& C_eq_block(uint32_t q) const;
    // dim C^q_eq -- available in every mode, including the memory-lean one in
    // which C_eq_block(q) itself has been released.
    uint32_t C_eq_dimension(uint32_t q) const;
    const BlockMat& Z_eq_block(uint32_t q) const;
    const BlockMat& B_eq_block(uint32_t q) const;
    const BlockMat& H_eq_block(uint32_t q) const;
    const BlockMat& P_block(uint32_t q) const;

    // --- FlatMatrix accessors: legacy shapes and values, materialised lazily --
    // Per degree q a list of N_q x d_V cochains (one per basis vector, row s =
    // value on simplex s), resp. the dim H^q x dim_q row-major projector.  The
    // projector list keeps the legacy factor_basis_block shape conventions
    // (0 x 0 when Z^q_eq is empty, dim H^q x dim_q otherwise).  Built once on
    // first call from the block objects; no cost if never called.
    const std::vector<std::vector<FlatMatrixT<Elt>>>& get_Z_eq_bases_fm() const;
    const std::vector<std::vector<FlatMatrixT<Elt>>>& get_B_eq_bases_fm() const;
    const std::vector<std::vector<FlatMatrixT<Elt>>>& get_H_eq_bases_fm() const;
    const std::vector<std::vector<FlatMatrixT<Elt>>>& get_C_eq_bases_fm() const;
    const std::vector<FlatMatrixT<Elt>>& get_H_eq_projectors_from_C_eq_flat_fm() const;

    const std::vector<uint32_t>& get_H_eq_dimensions() const;

    // --- (slot-8 WP-3) The coefficient ring and the Z/ell^k (module) results ----
    // coefficient_ring_k(): 1 for F_ell; k for Z/ell^k.  is_composite() == (k >= 2).
    uint32_t coefficient_ring_k() const { return ring_k_; }
    bool is_composite() const { return ring_k_ >= 2; }
    // The composite context (throws std::logic_error on an F_ell object).
    const engine::ModCtx<Elt>& ring_context() const;
    // The module accessors below throw std::logic_error on an F_ell object and
    // std::runtime_error before compute_cohomology().  Degrees q = 0, 1, 2.
    uint32_t num_degrees() const;
    const ModuleDegree& module_degree(uint32_t q) const;
    // (S23 G2-E; slot23/G2_plan.md Sec. 2.5) THE LEAN RETENTION of the module route: frees every degree's X and Zc
    // (read by nothing after compute_cohomology(): H_class_coordinates, the K-mode and the Hecke operators use C,
    // piv, dec, H, dK).  Called only by hecke_ext::RunModuleT (the sat / Loewy runner) on the stream route.
    // Afterwards module_degree(q).X / .Zc are 0 x 0 and the guarded accessors module_X / module_Zc throw a named
    // std::logic_error; every other accessor and value is unchanged.
    void release_module_intermediates();
    // (S24 G2-H; DenseElim.h HECKE_MODULE_LEAN_PEAK (d)) set BEFORE compute_cohomology() by the runner that will call
    // release_module_intermediates() right after it (hecke_ext::RunModuleT on the stream route): with the lever on, the
    // Z^q / H^q loop runs in descending q and frees X_q / Zc_q as soon as their last reader in the loop has run, and the
    // object ends released (as after release_module_intermediates(), which is then a no-op).  Off, or with the lever
    // or HECKE_MODULE_STREAM off: no effect.  Every object kept is the same, byte for byte.
    void set_module_early_release(bool on) { module_early_release_ = on; }
    bool module_intermediates_released() const { return module_released_; }
    const engine::DenseBuf<Elt>& module_X(uint32_t q) const;    // module_degree(q).X, guarded
    const engine::DenseBuf<Elt>& module_Zc(uint32_t q) const;   // module_degree(q).Zc, guarded
    uint32_t C_eq_rank(uint32_t q) const;                           // C^q_eq is FREE over Z/ell^k
    std::vector<uint32_t> H_lengths() const;                        // lengths_H^q
    const std::vector<uint32_t>& H_invariant_factors(uint32_t q) const;   // invariant_factors_H^q
    const engine::DenseBuf<Elt>& H_module_basis(uint32_t q) const;  // dim_q x t, the b_i
    // (c5): the coordinates of each flat cocycle column of Y (dim_q x ny) on
    // the b_i, the i-th in [0, ell^{e_i}).  Throws std::runtime_error if a
    // column is not in C^q_eq or not a cocycle.
    std::vector<std::vector<Elt>> H_class_coordinates(uint32_t q, engine::DenseView<const Elt> Y) const;
    // The K-mode ([E] sec:ring (d); K = k).  r_K(q) for q in {0, 1} is
    // r_K(d^q).  ladic_applicable(): n > 0 even (ell >= 5 always holds here);
    // outside it the K-mode objects are absent and ladic_type_certified() /
    // ladic_type() throw std::logic_error.  ladic_type(q) also throws at an
    // uncertified level (its accessor never returns a value there).
    uint32_t r_K(uint32_t q) const;
    bool ladic_applicable() const;
    bool ladic_type_certified() const;
    LadicType ladic_type(uint32_t q) const;
    uint32_t ladic_e_max() const;                                  // max torsion exponent of ladic_type^1 (0 if none)
    LadicLevelState ladic_level_state() const;

    // --- (slot-9 WP-3; decision 4 (A)) ell in {2, 3} under Z/ell^k ----------
    // The object is the BREDON one over Z/ell^k ([C] def:ringsemantics(2),
    // prop:bredon(2)-(3)), computed by the flat-coordinate route (the note at
    // compute_cohomology_flat_module()).  There, num_degrees(), H_lengths()
    // and H_invariant_factors(q) HONOUR; module_degree, C_eq_rank,
    // H_module_basis, H_class_coordinates (no stored basis: C^q_eq is not
    // free) and the K-mode (r_K, ladic_*: [C] prop:certifiede(5)) REFUSE with
    // hecke::NotImplementedInThisBuild naming the accessor.  At ell >= 5 the
    // accessors below throw std::logic_error and nothing above changes.
    bool is_flat_route() const { return flat_route_; }
    const FlatModuleDegree& flat_module_degree(uint32_t q) const;

    // --- String Conversion ---
    std::string to_string() const;
    std::string to_string_offset(const uint32_t offset) const;
};

using EquivariantCohomologyOnK = EquivariantCohomologyOnKT<uint32_t>;
using EquivariantCohomologyOnK64 = EquivariantCohomologyOnKT<uint64_t>;

// ---------------------------------------------------------------------------
// (matrix-engine Phase 1, item 7.4) Vacuous-relation short-circuit.
//
// `LocalSystemSL2Z` appends one relation per simplex in every degree with
// M = rho_n(-I) (the FIX-negI relations).  For EVEN n, rho_n(-I) = I, so the
// constraint f(s) = M f(s) reads f(s) = f(s): the corresponding block row of
// A_Cq (and of A_Zq) is identically zero.  85 % of the q=1 rows and 100 % of
// the q=2 rows are vacuous in that regime, and the block elimination was being
// handed all of them (02 §5.4).  A row of zeros contributes no pivot, so
// dropping it leaves the RREF pivots, the free columns and hence the kernel
// BASIS bit-for-bit unchanged -- which is why this is safe against the frozen
// golden tier-B hashes.
//
// For ODD n the same relations are NOT vacuous: rho_n(-I) = -I forces
// 2 f(s) = 0 and hence H^* = 0 (Hecke oracle O2).  The predicate below tests
// M == I on the actual matrix, so those rows are kept and the odd-n semantics
// are preserved exactly.  (At ell = 2, -I == I and the relation genuinely is
// vacuous; the predicate agrees.)
//
// The switch exists so that VacuousRelationValidation can run the pre-7.4
// assembly and compare against it.  It is global, defaults to enabled, and is
// only ever touched by that validation.
// ---------------------------------------------------------------------------
namespace EquivariantCohomologyOptions {
    bool vacuous_relation_shortcircuit_enabled();
    void set_vacuous_relation_shortcircuit(bool enabled);

    // RAII guard: restores the previous setting on scope exit.
    class ScopedVacuousShortcircuit {
        bool saved_;
    public:
        explicit ScopedVacuousShortcircuit(bool enabled)
            : saved_(vacuous_relation_shortcircuit_enabled()) {
            set_vacuous_relation_shortcircuit(enabled);
        }
        ~ScopedVacuousShortcircuit() { set_vacuous_relation_shortcircuit(saved_); }
        ScopedVacuousShortcircuit(const ScopedVacuousShortcircuit&) = delete;
        ScopedVacuousShortcircuit& operator=(const ScopedVacuousShortcircuit&) = delete;
    };

#if OC_ENABLE_LEGACY
    // (matrix-engine Phase 3 Stage B) Force the LEGACY arm of
    // compute_cohomology() (BlockFlatMatrix::flat_kernel + factor_basis_block)
    // regardless of ENGINE_BACKEND.  Validation-only: it lets a validation
    // compare the two arms inside one process.  Global, defaults to off.
    // (S24 CL-L2) ON builds only, with ScopedLegacyCohomologyPath below.
    bool legacy_cohomology_path_forced();
    void set_legacy_cohomology_path_forced(bool forced);
#endif

    // (matrix-engine Phase 4, memory profile 05 Sec. 7 item 3) Whether
    // compute_cohomology() RETAINS the objects no production consumer reads:
    // A_C_block(q), A_Z_block(q) (and the rho_n(g) blocks they pin),
    // Z_eq_block(q) and B_eq_block(q).  C_eq, H_eq and P_q are always kept
    // (HeckeOperatorOnK reads them).  Default ON, so every accessor and the
    // golden objects behave exactly as before; the benchmark driver and a
    // memory-lean pipeline run turn it OFF, after which those accessors (and
    // get_Z_eq_bases_fm / get_B_eq_bases_fm) throw std::runtime_error.  The
    // retained set never influences a value: the objects are dropped only
    // after their last reader inside compute_cohomology().
    bool retain_intermediates();
    void set_retain_intermediates(bool retain);
    class ScopedRetainIntermediates {
        bool saved_;
    public:
        explicit ScopedRetainIntermediates(bool retain) : saved_(retain_intermediates()) { set_retain_intermediates(retain); }
        ~ScopedRetainIntermediates() { set_retain_intermediates(saved_); }
        ScopedRetainIntermediates(const ScopedRetainIntermediates&) = delete;
        ScopedRetainIntermediates& operator=(const ScopedRetainIntermediates&) = delete;
    };
#if OC_ENABLE_LEGACY
    class ScopedLegacyCohomologyPath {
        bool saved_;
    public:
        explicit ScopedLegacyCohomologyPath(bool forced)
            : saved_(legacy_cohomology_path_forced()) {
            set_legacy_cohomology_path_forced(forced);
        }
        ~ScopedLegacyCohomologyPath() { set_legacy_cohomology_path_forced(saved_); }
        ScopedLegacyCohomologyPath(const ScopedLegacyCohomologyPath&) = delete;
        ScopedLegacyCohomologyPath& operator=(const ScopedLegacyCohomologyPath&) = delete;
    };
#endif
}

#endif // EQUIVARIANT_COHOMOLOGY_ON_K_H
