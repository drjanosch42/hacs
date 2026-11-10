// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_OPERATOR_ON_K_H
#define HECKE_OPERATOR_ON_K_H

// Stage S7 -- HeckeOperatorOnK: the normative fused cochain pipeline
// Theta_g = omega_bullet . Xi_g of [C, def:xi] / [C, thm:xi], and the matrix
// output levels (a) and (b) of the plan's Section 3, assembled per
// [C, alg:assembly].
//
// Contract: docs/latex/hecke_action.tex ([C]):
//   * [C, def:xi]: (Xi_g c)(sigma)_i = rho_n(g) phi_c(tau(s_i . i(sigma))),
//     explicitly ([C, eq:xiexplicit])
//       (Xi_g c)(sigma)_i = sum_k n_k^{(i,sigma)}
//                             rho_n(g gamma_k^{(i,sigma)}) c(sigma_k^{(i,sigma)}),
//     and the omega row: (Theta_g c)(sigma) = sum_i rho_n(s_i^{-1}) (Xi_g c)(sigma)_i.
//   * [C, thm:xi]: Theta_g maps C^q_eq to C^q_eq, commutes with d, and realizes
//     T_{Gamma g Gamma} = Psi Ttilde Psi^{-1} on H^q for every q.
//   * [C, eq:heckechain] (the SECOND evaluation route, oracle O12(a)):
//     (Ttilde phi)(x) = sum_i rho_n(delta_i) phi(tau(s_i . x)), delta_i = s_i^{-1} g.
//   * [C, alg:assembly]: column i of A^{(q)} is P_q * flatten(Theta_g z_i),
//     with the runtime assertions (y_i satisfies all L-relations; d y_i = 0),
//     column convention T_p[z_i] = sum_j A^{(q)}_{ji} [z_j].
//   * [C, prop:symconv](2) (monoid multiplicativity, the load-bearing identity
//     behind the two-route bit-identity): rho_n(A) rho_n(B) = rho_n(AB) for
//     A, B in the mod-ell matrix monoid.
//   * [C, obl:welldef], [C, obl:agreement] (first two routes).
//
// ===========================================================================
// THE TWO EVALUATION ROUTES AND THEIR MANDATORY BIT-IDENTITY (oracle O12(a)).
//
// FUSED route (binding; what Theta_block(q) / get_Theta_flat_matrices() store
// and apply_Theta() evaluates): for each target simplex sigma (block row),
// accumulate over i = 1..m and each stored XiTerm (n_k, sigma_k,
// rho_n(g gamma_k)) the d_V x d_V block
//     n_k * rho_n(s_i^{-1}) * rho_n(g gamma_k)   (mod ell)
// into block column sigma_k. The two rho_n factors are the PRECOMPUTED
// reduced tables of S3 (HeckeCosetTables::rho_s_inv) and S4
// (HeckeTauTable::xi_terms), multiplied as stored mod-ell matrices.
//
// ROUTE 2 ([C, eq:heckechain]; apply_route2() and the compute()-time check
// matrix): the block  n_k * rho_n((delta_i gamma_k) mod ell), where the
// integer product delta_i * gamma_k is formed EXACTLY (overflow-checked
// hecke::Mat22 arithmetic, invariant I6: keep exact until the single final
// reduction) and only then reduced mod ell and fed to
// ComputeSymmetricPowerMatrixFM (invariant I2). The determinant-N matrices
// delta_i, g gamma_k, delta_i gamma_k are NEVER inverted (invariant I7 /
// [C, obl:integral]).
//
// These agree entry-for-entry because delta_i gamma_k = s_i^{-1} (g gamma_k)
// exactly over Z and rho_n is multiplicative on the mod-ell monoid
// ([C, prop:symconv](2)); compute() ASSERTS the bit-identity over ALL entries
// of ALL degrees and throws on any mismatch. This is exactly the implementer's
// cross-check demanded by [C, sec:equivalence] ("the same table data evaluated
// in two bracketings ... must produce bit-identical cochains").
// ===========================================================================
//
// (matrix-engine Phase 3, Stage B) THE BLOCK MODEL IS THE STORAGE.
//
// Theta^{(q)} is an N_q x N_q engine::BlockMat over the uniform (N_q, d_V)
// partition -- the binding L1 layout, see below -- whose PATTERN is the set of
// (sigma, sigma_k) pairs the tau-table names (audit 02 Sec. 7.8) and whose
// blocks are accumulated by BlockBuilder::accumulate from the d_V x d_V term
// blocks, one BlockRef per term, never per cell.  The two routes are two
// PROGRAMS over that pattern (program_fused / program_route2; Phase 4 turns
// them into tapes) that differ only in where a term's block comes from:
//   fused    rho_n(s_i^{-1}) * rho_n(g gamma_k)   one engine GEMM per distinct
//            (i, rho-pool slot) pair -- the tau-table's rho pool
//            (HeckeTauTable::rho_pool(), xt.rho_g_gamma.index()) is the memo
//            key, so identical rho_n(g gamma) blocks are SHARED, not copied;
//   route 2  rho_n((delta_i gamma_k) mod ell)      one rho_n per distinct
//            (i, exact gamma_k), in its OWN BlockPool.
// The two results are compared with blk::equal, i.e. AS MATHEMATICAL MATRICES
// (any grids, kinds, pool ids, layouts -- 02 Sec. 6.1 engine note): a block
// that is structurally zero on one side is compared against the materialised
// block of the other, and route 2 lives in a separate pool so that no
// content-addressed dedup can turn the comparison into a tautology.  Any
// mismatch throws exactly as before.
//
// The remaining steps of compute() are block / dense engine calls:
//   Theta5   Y = Theta * H_eq            one blk::gemm_dense over all d_q
//                                        columns of H_eq_block(q);
//   Theta6   relation check (i)          (relation block matrix R_q) * Y == 0,
//                                        one blk::gemm_dense, all i_basis at once;
//   Theta7   cocycle check (ii)          (D_1 (x) I) * Y == 0, the coboundary
//                                        as Scalar +-1 blocks, one gemm_dense;
//   Theta8   Theta_eq                    W = Theta * C_eq (gemm_dense), then
//                                        blk::solve_multi_rhs(C_eq, W) -- ONE
//                                        elimination for every RHS, bit-identical
//                                        to the per-vector solve by uniqueness;
//   Theta9   A-N3 residual (C5)          W - C_eq * Theta_eq as one
//                                        kernels::gemm (GemmOp::Sub), then a
//                                        zero scan -- recomputed independently
//                                        of the solver, must vanish identically;
//   Theta10  A^{(q)} = P_q * Y           one blk::gemm_dense with P_block(q).
//
// FLAT COCHAIN LAYOUT (binding, [C, sec:conventions] "Cochain conventions";
// identical to the coordinate system of H_eq_projectors_from_C_eq_flat and to
// the S6 EquivariantCochainComplexMorphismOnK layout): a q-cochain
// c in C^q(K;V) is the length N_q*d_V column with
//     flat index  s*d_V + k  =  value at simplex s, fiber coordinate k,
// simplices in the stored SimplicialComplex order, fiber basis
// u_0 = x^n, ..., u_n = y^n -- SIMPLEX-MAJOR, FIBER-MINOR.  Since Phase 3 this
// is the row index row_off[s] + k of a BlockMat multi-vector over the uniform
// partition; the ONLY implementation of the map is BlockMat::to_flat /
// from_flat (plan item 7.20) -- this file no longer carries an `s*d_V + k`
// loop of its own, it reads H_eq_block(q) / C_eq_block(q) directly.
//
// MATRIX OUTPUT LEVELS (plan Section 3, binding):
//   Level (a): Theta_block(q), the N_q x N_q block matrix of Theta_g on ALL of
//     C^q(K;V) in flat coordinates (it maps the subspace C^q_eq to itself,
//     [C, thm:xi]); get_Theta_flat_matrices()[q] is the same matrix as an
//     (N_q*d_V)^2 FlatMatrix, MATERIALISED LAZILY (once, on first call, via
//     to_dense) for the consumers that still speak FlatMatrix
//     (HeckeCrossChecks, HeckeOperatorFamily::theta_flat*, the JSON writer, the
//     golden harness).  Plus Theta_eq[q], the restriction to C^q_eq in the
//     stored C_eq basis of EquivariantCohomologyOnK, obtained by EXACT linear
//     solve (change of basis); the solvability of that solve IS runtime
//     assertion (iii), and its exact residual is recomputed independently and
//     asserted to be identically zero (certificate datum, amendment A-N3).
//   Level (b): A[q] = the d_q x d_q matrix of T_p on H^q(K,R;L) w.r.t. the
//     stored H_eq_bases basis, column convention T_p[z_i] = sum_j A_{ji}[z_j],
//     column i = P_q * flatten(Theta_g z_i). NOT symmetrized, NOT normalized
//     (the integral normalization is pinned by O1 and [C, thm:relations]).
//   Empty degrees: d_q = 0 gives a 0 x 0 A[q] (and dim C^q_eq = 0 gives a
//     0 x 0 Theta_eq[q]); nothing crashes (O2, and H^2 at ell >= 5).
//
// RUNTIME ASSERTIONS (all inside compute(); ANY failure throws
// std::runtime_error -- the object then stays in the "not computed" state):
//   (i)   for every H-basis cocycle z_i, y_i := Theta_g z_i satisfies EVERY
//         L-relation f(s_lhs) = rho_n(g_r) f(s_rhs) of the base
//         SimplexRelation list;
//   (ii)  d y_i = 0, with d = d_1 (x) id_V from the base K's coboundary
//         (SimplicialComplex::coboundary_operator);
//   (iii) Theta_flat * (each C_eq basis vector) lies in span(C_eq) -- the
//         exact solve that DEFINES Theta_eq[q] -- with exact zero residual;
//   plus the two-route bit-identity over all matrix entries (O12(a) above).
//
// TWO ROLLBACKS (both value-preserving; the pre-Phase-3 dense assembly is kept
// as the LEGACY arm, since S24 CL-L2 in Hecke/legacy/HeckeOperatorOnKLegacy.inc,
// and both exist only in an OC_ENABLE_LEGACY=ON build):
//   * ENGINE_BACKEND=legacy (engine::backend_is_legacy()) runs the legacy arm
//     instead and rebuilds Theta_block(q) from its dense output, so every
//     accessor works under both backends;
//   * -DENGINE_SHADOW_LEGACY=ON (needs -DOC_ENABLE_LEGACY=ON, S24 CL-L1;
//     Hecke/legacy/HeckeOperatorOnKShadow.inc) runs BOTH arms and compares
//     Theta_flat, Theta_eq, A^{(q)} and the certificate counters entrywise,
//     throwing std::logic_error on the first difference.
//
// JSON EXPORT (to_json(); hand-rolled writer, no third-party library):
// levels (a) + (b) per the plan Section 3 schema conventions. Documented
// ordering (also in Hecke/README.md, amendment A-N4):
//   * matrices are emitted as arrays of ROWS (row-major), entries as decimal
//     integers in [0, ell);
//   * the flat layout of Theta_flat rows/columns is the simplex-major,
//     fiber-minor layout above;
//   * A[q] follows the column convention T_p[z_i] = sum_j A[j][i] [z_j]
//     (so the JSON row index is j, the column index is i);
//   * any polynomial coefficient list (none at this level; levels (c) live in
//     S10/S11) is low -> high degree per A-N4;
//   * the "certificate" object carries the O12(a) route bit-identity flag,
//     the counts of executed runtime checks (i)/(ii)/(iii), and the exact
//     Theta_eq solve residual (identically 0; A-N3).
//
// Placement (documented per the task's instructions): like S2-S5, this class
// consumes ModArithmetic (FlatMatrix / BlockMat / ComputeSymmetricPowerMatrixFM)
// and the model/cohomology layer, so it is compiled into the cohomology_app
// executable sources (NOT into the pure-integer HeckeCore static library),
// while linking HeckeCore for hecke::Mat22 (route 2's exact products).

#include <array>
#include <cstdint>
#include <deque>
#include <map>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "ExactInt.h"
#include "HeckeCosetTables.h"
#include "HeckeTauTable.h"
#include "RhoMemo.h"
#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemUtils.h" // FibreRepresentation (slot-6 WP-1)
#include "../../ModArithmetic/BlockMat.h"
#include "../../ModArithmetic/FlatMatrix.h"
#include "../../ModArithmetic/LegacyConfig.h"   // OC_ENABLE_LEGACY (S24 CL-R1)
// (matrix-engine Phase 3) Matrix.h is no longer included directly. The only
// MatrixT left in this header is the level-(b) A^{(q)} (d_q x d_q, d_q <= ~20),
// which is the input type of HeckeModuleDecomposition -- the one consumer the
// plan deliberately leaves on MatrixT (plan Phase 3, "Matrix retirement";
// audit 03 Sec. 4.3). MatrixT itself comes in through FlatMatrix.h, which needs
// it for its conversion constructor.

namespace hecke {

// ---------------------------------------------------------------------------
// (hecke-extensions slot 8, WP-3; E8 item 3; [E] v1.9.0-pre.1 sec:ring) THE
// COEFFICIENT RING.  The operator's ring is its cohomology object's
// (EquivariantCohomologyOnK::coefficient_ring_k()), and the constructor
// requires the ring in force to be the same -- a field operator on a
// Z/ell^k object, or the reverse, raises hecke::NotImplementedInThisBuild
// naming this class.  At the default nothing below changes.  Over Z/ell^k
// (k >= 2) compute() runs the MODULE route (compute_module()):
//   * Theta^{(q)} as a dense (N_q d_V)^2 matrix mod ell^k, built BOTH ways --
//     fused n_k rho_n(s_i^{-1}) rho_n(g gamma_k) and route 2
//     n_k rho_n(delta_i gamma_k) -- from the EXACT coset / tau data through
//     the composite rho_n (hecke::RhoOfExactRing, E8-P5: at p = ell the
//     delta_i gamma_k are U_ell's, reached by the diag(ell, 1) split), and
//     the two asserted entrywise equal (O12(a) over Z/ell^k); the coset and
//     tau tables' own mod-ell rho tables are NOT read; a tau datum that used
//     the mod-ell triangle fallback is refused (it is not integral);
//   * runtime assertions (i) every relation, (ii) d y = 0 on every basis
//     class, (iii) Theta C^q_eq in C^q_eq by solve_multi_rhs_module with the
//     exact residual recomputed -- the same counters as the field route;
//   * A^{(q)}(T_p) on the cyclic summands ([E] sec:ring (c5)): column j holds
//     the coordinates of T_p b_j, entry (i, j) in [0, ell^{e_i});
//   * free_charpoly_H1(): E8-P4's charpoly of A^{(1)}(T_p) on the free
//     quotient, only at a certified K-mode, over Z/ell^{K - e_max}.
// Over Z/ell^k every FIELD accessor (Theta_block, get_Theta_*, get_H_matrices,
// apply_Theta, apply_route2, to_json) throws a named std::logic_error; the
// contragredient representation is refused.
// ---------------------------------------------------------------------------
// (S23 G2; audit m3) VALIDATION ONLY (HeckeModuleStreamValidation's (N) arm): while in force, the module
// route's W = T C ((iii)) gets +1 at the first non-pivot row of C, column 0 -- a right-hand side off im C
// (C[piv, :] = I) -- on the stream route and the legacy route alike, so (iii) must fail with the legacy
// text on both.  Process-wide, not nestable across threads.
struct ScopedThetaEqPerturbation {
    int saved_;
    explicit ScopedThetaEqPerturbation(bool on);
    ~ScopedThetaEqPerturbation();
    ScopedThetaEqPerturbation(const ScopedThetaEqPerturbation&) = delete;
    ScopedThetaEqPerturbation& operator=(const ScopedThetaEqPerturbation&) = delete;
};

// (S23 G2-E; slot23/G2_plan.md Sec. 2.5) What the module route keeps after compute().  Full (the default):
// every object.  Lean (opt-in; set only by hecke_ext::RunModuleT, the sat / Loewy runner, and only on the
// stream route): Theta^{(q)} and Theta^{(q)}_eq are released after degree q's certificates -- Theta_module() /
// Theta_eq_module() then throw a named std::logic_error; A^{(q)}, the certificate bundle and every printed value
// are unchanged.  With HECKE_MODULE_STREAM=0 the legacy route runs and keeps everything.
enum class ModuleRetention { Full, Lean };

// (S23 G2-D) VALIDATION ONLY (HeckeModuleStreamValidation's (N) arm): while in force, route 2 of the module
// route gets +1 at one entry, so [C, obl:agreement](i) must fail with the same text on both routes.  Sites:
//   Degree0Last       the last entry of degree 0 (N_0 = 7: one chunk);
//   Degree1FirstChunk entry (0, 0) of degree 1, inside its FIRST sigma-row chunk (rows 0..5 of N_1 = 11);
//   Degree1LastChunk  the last entry of degree 1, inside its SECOND chunk (rows 6..10): the chunk-offset
//                     comparison T(s0 d + a, b) is under the control (audit m1);
//   Degree1Middle     (G2-H audit F2) entry (3 d + d/2, dim/2) of degree 1: an INTERIOR block row (sigma = 3) of
//                     either chunking, so a comparison that skipped the middle block rows of a chunk is caught.
// On the legacy route the same entry of T2 is moved.  Process-wide, not nestable.
enum class Route2Site { Off = 0, Degree0Last = 1, Degree1FirstChunk = 2, Degree1LastChunk = 3, Degree1Middle = 4 };
struct ScopedRoute2Perturbation {
    int saved_;
    explicit ScopedRoute2Perturbation(Route2Site site);
    ~ScopedRoute2Perturbation();
    ScopedRoute2Perturbation(const ScopedRoute2Perturbation&) = delete;
    ScopedRoute2Perturbation& operator=(const ScopedRoute2Perturbation&) = delete;
};

template<class Elt>
class HeckeOperatorOnKT {
public:
    using BlockMat  = engine::BlockMat<Elt>;
    using BlockPool = engine::BlockPool<Elt>;
    using BlockRef  = engine::BlockRef<Elt>;

    // cohomology_of_L: the COMPUTED base system L = Sym^n(F_ell^2) on the
    // 7-vertex model (compute_cohomology() must have been called; checked).
    // cosets: S3 tables for g = diag(1, N); tau: the S4-certified g-twisted
    // chain datum built against the SAME cosets object. All three must
    // outlive this object. The constructor cross-checks ell / d_V consistency
    // between the three inputs and the model shape (7/11/5) and throws
    // std::runtime_error / std::invalid_argument on any mismatch.
    //
    // (hecke-extensions slot 6, WP-1 / phase E7a) `rep` is the FIBRE
    // REPRESENTATION of the coefficient system ([C, def:contragredient]):
    //   * FibreRepresentation::SymN (the default) -- rho_n, EXACTLY the
    //     pre-slot-6 operator: every block is read from the same tables
    //     (cosets.rho_s_inv, the tau table's rho pool) and every route-2 /
    //     rho_of_exact block is the same ComputeSymmetricPowerMatrixFM call,
    //     so the default path is byte-identical by construction;
    //   * FibreRepresentation::ContragredientN -- rho_n^vee(h) = rho_n(adj h)^T
    //     on D_n.  Theta_g carries ONE fibre matrix per term, so the operator
    //     on D_n is the same (sigma, i, k) walk with every block rho_n(h)
    //     replaced by rho_n^vee(h), for the SAME integral h: the omega blocks
    //     rho_n^vee(s_i^{-1}) and the xi blocks rho_n^vee((g gamma_k) mod ell)
    //     are rebuilt here from the exact coset / tau data (which are
    //     coefficient-free and reused unchanged), one per rho-pool slot.  The
    //     O12(a) two-route identity holds again because rho_n^vee is
    //     multiplicative on the mod-ell monoid ([C, prop:pairing](1)).
    //     cohomology_of_L must then be the cohomology of the contragredient
    //     local system (GenerateSL2ZContragredientRelationsWithGroupData):
    //     the constructor checks its 8 model seed relations against
    //     rho_n^vee(g_r) and throws std::runtime_error otherwise.
    //   * (slot-11 WP-5) FibreRepresentation::QuotientL -- rho_{Q_n}(h) =
    //     pi rho_{n+ell+1}(h) E on Q_n = Sym^{n+ell+1} / L . Sym^n
    //     (LocalSystemUtils.h), fibre dimension ell + 1.  The same walk as the
    //     contragredient: every block rebuilt from the exact data through
    //     rho_of_exact.  cosets / tau are the tables of Sym^n (their n_sym is
    //     the n of Q_n; their rho tables are shape-checked at n + 1 and not
    //     read); cohomology_of_L must be built on
    //     GenerateSL2ZQuotientLRelationsWithGroupData (seed relations checked).
    //     Its program-cache keys carry their own tag.  Refused over Z/ell^k.
    //   * (slot-13 WP-5) FibreRepresentation::HidaLatticeH -- rho^H_n(h) =
    //     rho_n(h^T)^T on Hida's lattice L_n ([C] def:hidalattice variant (H)).
    //     The MODULE route only (Z/ell^k, k >= 2; refused on the field route):
    //     compute_module()'s rho is the transposed call, the seed relations are
    //     checked against RhoHidaOfExactRing (cohomology_of_L built on
    //     GenerateSL2ZRelationsRingHida).  The low-rank lever refuses it.
    // Selection is by this object -- there is no global switch.
    HeckeOperatorOnKT(const EquivariantCohomologyOnKT<Elt>* cohomology_of_L,
                     const HeckeCosetTablesT<Elt>* cosets, const HeckeTauTableT<Elt>* tau,
                     FibreRepresentation rep = FibreRepresentation::SymN);

    // The fibre representation this operator was built in (slot-6 WP-1).
    FibreRepresentation representation() const { return rep_; }

    // Builds all matrices below; runs the runtime assertions (header note);
    // THROWS std::runtime_error on any failure. Idempotent.
    void compute();

    // Level (a) on the block model: Theta_g on C^q(K;V) as an N_q x N_q
    // BlockMat over the uniform (N_q, d_V) partition (grid == L1 layout).
    // Apply to a dense multi-vector with blk::gemm_dense.  Throws if compute()
    // has not succeeded or q is out of range.
    const BlockMat& Theta_block(uint32_t q) const;
    // The pool the Theta blocks (and the level-(a) working set) live in.
    const std::shared_ptr<BlockPool>& block_pool() const;

    // Level (a), FlatMatrix shapes: Theta_g on flat cochains C^q(K;V), and
    // restricted to C^q_eq in the stored C_eq basis. Size 3 (q = 0, 1, 2);
    // (N_q*d_V)^2 resp. (dim C^q_eq)^2. Throw if compute() has not succeeded.
    // The flat Theta list is materialised from Theta_block(q) on the first call
    // and cached (a consumer that never asks for it pays nothing).
    const std::vector<FlatMatrixT<Elt>>& get_Theta_flat_matrices() const;
    const std::vector<FlatMatrixT<Elt>>& get_Theta_eq_matrices() const;

    // Level (b): A^{(q)} per [C, alg:assembly], d_q x d_q, q = 0, 1, 2;
    // column convention T_p[z_i] = sum_j A_{ji}[z_j]. Empty degree => 0 x 0.
    const std::vector<MatrixT<Elt>>& get_H_matrices() const;

    // (matrix-engine Phase 6) The in-process certificate bundle compute() has
    // ALREADY run -- unconditionally, on every call, in every mode: the
    // O12(a) two-route bit-identity flag and the executed counts of runtime
    // assertions (i)/(ii)/(iii) plus the A-N3 exact residual. These are the
    // same numbers to_json()'s "certificate" object carries; the accessor
    // exists so the Phase 6 campaign driver can RECORD them per job without
    // parsing its own JSON (the campaign's `certify = sampled` mode samples
    // only the EXTERNAL cross-checks -- HeckeCrossChecks, T_p_pow_direct --
    // and never these). Throws if compute() has not succeeded.
    struct CertificateBundle {
        bool routes_bit_identical = false;  // O12(a)
        uint64_t relation_checks = 0;       // assertion (i)
        uint64_t cocycle_checks = 0;        // assertion (ii)
        uint64_t theta_eq_solves = 0;       // assertion (iii)
        Elt theta_eq_residual_max = 0;      // A-N3; always 0
        bool theta_eq_certified = true;     // (Phase 4) Theta8/Theta9 ran (false in the memory-lean level)
    };
    CertificateBundle certificates() const;

    // (slot-24 G6) Releases level (a) after compute(): Theta_block / Theta_eq /
    // Theta_flat, the fused blocks and their pool, the operator's own Omega
    // table.  Afterwards only get_H_matrices() (the same object, so references
    // handed out stay valid) and certificates() are served; every level-(a)
    // accessor, apply_Theta / apply_route2, cosets() / tau(), rho_memo_identity(),
    // to_string() and to_json() throw std::logic_error, and none of them reads
    // the coset or tau table again -- so a caller that only needs A^{(q)} (the
    // HeckeOperatorFamily of a HeckeExt cell, HeckeOperatorFamily::release_level_a)
    // may destroy those tables afterwards.  Field route only.  Memory only: no
    // value moves.
    void release_level_a();
    bool level_a_released() const { return level_a_released_; }

    // --- (slot-8 WP-3) Z/ell^k (see the note above the class) ------------------
    uint32_t coefficient_ring_k() const { return ring_k_; }
    bool is_composite() const { return ring_k_ >= 2; }
    // A^{(q)}(T_p) on the cyclic summands of H^q (t_q x t_q, row-major DenseBuf;
    // column convention T_p b_j = sum_i A_ij b_i, A_ij in [0, ell^{e_i})).
    // Throws std::logic_error on an F_ell operator.
    const engine::DenseBuf<Elt>& H_matrix_module(uint32_t q) const;
    // Theta^{(q)} over Z/ell^k, dense (N_q d_V) x (N_q d_V), flat layout.
    const engine::DenseBuf<Elt>& Theta_module(uint32_t q) const;
    // (S23 G2) Theta^{(q)}_eq over Z/ell^k (c_q x c_q, the (iii) solve's result; read-only, for
    // HeckeModuleStreamValidation's entrywise stream-vs-legacy comparison).
    const engine::DenseBuf<Elt>& Theta_eq_module(uint32_t q) const;
    // (S23 G2-E) the retention of the module route's objects (before compute(); see ModuleRetention).
    void set_module_retention(ModuleRetention r) { retention_ = r; }
    ModuleRetention module_retention() const { return retention_; }
    bool module_objects_released() const { return released_; }   // true iff Lean took effect (stream route)
    // E8-P4 ([E] sec:ring (d); slot-8 plan Sec. 2.3 item 13): the monic
    // characteristic polynomial of A^{(1)}(T_p) on the free quotient
    // H^1(Z/ell^K) / H^1(Z/ell^K)[ell^{e_max}] ~= (Z/ell^{K - e_max})^r, with
    // its precision as part of the object; coefficients low -> high in
    // [0, ell^{K - e_max}).  Defined ONLY where the cohomology object's K-mode
    // is certified; elsewhere it throws a named std::logic_error and never
    // returns a value.
    struct FreeCharpoly {
        uint32_t precision = 0;          // K - e_max: the polynomial is over Z/ell^precision
        uint64_t modulus = 0;            // ell^precision
        uint32_t rank = 0;               // r, the degree
        std::vector<Elt> coeffs;         // r + 1 coefficients, low -> high, leading 1
    };
    FreeCharpoly free_charpoly_H1() const;

    // Route 2 for O12(a): direct evaluation of [C, eq:heckechain] on a flat
    // q-cochain (an (N_q*d_V) x 1 FlatMatrix in the layout above), via the
    // exact tau-terms and rho_n((delta_i gamma_k) mod ell) formed from EXACT
    // integer products delta_i * gamma_k (invariant I6). Matrix-free: does
    // NOT read the fused Theta tables, so it stays an independent route;
    // usable before compute().
    FlatMatrixT<Elt> apply_route2(uint32_t q, const FlatMatrixT<Elt>& c_flat) const;

    // (slot-5 WP-6) The memo-taking form.  `memo` caches the PURE function
    // (i, gamma) |-> rho_n((delta_i gamma) mod ell) of this operator -- see
    // Hecke/RhoMemo.h for the dependency argument -- so a caller that applies
    // route 2 to many operands (HeckeConeAction's standard-basis sweep) pays
    // for each distinct block once instead of once per call.  The arithmetic
    // below is term for term the same whether the table arrives empty or warm,
    // and the one-argument overload above keeps creating a fresh local table,
    // so every external caller sees exactly the pre-package behaviour.
    using RhoMemo = ::hecke::RhoMemoT<engine::DenseBuf<Elt>>;
    FlatMatrixT<Elt> apply_route2(uint32_t q, const FlatMatrixT<Elt>& c_flat, RhoMemo& memo) const;
    // The identity a RhoMemo of THIS operator is stamped with (Hecke/RhoMemo.h).
    typename RhoMemo::Identity rho_memo_identity() const;

    // Fused route: Theta_block(q) * c_flat (one blk::gemm_dense). Requires compute().
    FlatMatrixT<Elt> apply_Theta(uint32_t q, const FlatMatrixT<Elt>& c_flat) const;

    // (hecke-extensions slot 6, WP-4; additive, read-only) The three inputs
    // this operator was built on, so a cross-route (the strict-pair tau^s of
    // HeckeConeStrictPairRoute) can rebuild the same operator on another
    // admissible tau without re-threading the pipeline objects.
    const EquivariantCohomologyOnKT<Elt>& cohomology_of_L() const { return *coh_; }
    const HeckeCosetTablesT<Elt>& cosets() const { require_level_a("cosets"); return *cosets_; }
    const HeckeTauTableT<Elt>& tau() const { require_level_a("tau"); return *tau_; }

    std::string to_string() const { return to_string_offset(0); }
    std::string to_string_offset(uint32_t offset) const;

    // Levels (a)+(b) + certificate bundle, schema in the header note.
    std::string to_json() const;

    // (matrix-engine Phase 4) The tau-pattern part of the fused program's
    // shape signature for degree q: an FNV-1a-64 over the (sigma, i, k) walk
    // -- (sigma, i, sigma_k, n_k mod ell, rho-pool slot) per term with a
    // non-zero coefficient -- and the omega kinds.  The program cache key of
    // compute() carries it (plan D4's cache-poisoning caveat), so a tau table
    // with a different support (e.g. the mod-ell fallback of the triangle
    // fill) misses and records afresh.  Exposed for EngineTapeConsumerValidation.
    static uint64_t tau_pattern_hash(const HeckeTauTableT<Elt>& tau, const HeckeCosetTablesT<Elt>& cosets,
                                     uint32_t q, uint32_t N_q, uint32_t m, Elt ell);

private:
    const EquivariantCohomologyOnKT<Elt>* coh_;
    const HeckeCosetTablesT<Elt>* cosets_;
    const HeckeTauTableT<Elt>* tau_;

    Elt ell_ = 0;    // coefficient prime (the legacy `prime`, invariant I1)
    uint32_t d_V_ = 0;    // n_sym + 1
    uint32_t n_sym_ = 0;
    uint32_t m_ = 0;      // number of cosets
    uint32_t num_degrees_ = 0; // 3 for the 7-vertex model (asserted)

    // (slot-6 WP-1) The fibre representation and, for ContragredientN only,
    // the fused route's rebuilt blocks: omega_dual_[i] = rho_n^vee(s_i^{-1})
    // (index 0 unused) and rho_pool_dual_[slot] = rho_n^vee((g gamma) mod ell)
    // for the tau table's pool slot `slot`.  Both stay EMPTY for SymN, whose
    // accessors below return the shared tables themselves.
    FibreRepresentation rep_ = FibreRepresentation::SymN;
    // (slot-8 WP-3) The coefficient ring (the cohomology object's) and the
    // module route's results.
    uint32_t ring_k_ = 1;
    engine::ModCtx<Elt> ring_ctx_;
    std::vector<engine::DenseBuf<Elt>> A_module_;
    std::vector<engine::DenseBuf<Elt>> Theta_module_;
    std::vector<engine::DenseBuf<Elt>> Theta_eq_module_;
    void init_composite();
    void compute_module();
    void compute_module_legacy();   // HECKE_MODULE_STREAM=0 (and the G2-A / G2-B read-off inside it)
    void compute_module_stream();   // (S23 G2-C / G2-D / G2-E) the default
    ModuleRetention retention_ = ModuleRetention::Full;
    bool released_ = false;
    void require_field_op(const char* what) const;
    void require_module_op(const char* what, uint32_t q) const;
    std::vector<FlatMatrixT<Elt>> omega_dual_;
    std::vector<FlatMatrixT<Elt>> rho_pool_dual_;
    // (slot-14 WP-10) SymN over a STREAMED coset table: the operator's own
    // Omega_i = rho_n(s_i^{-1}) (index 0 unused), materialised by
    // ensure_omega_stream() for the field paths that borrow Omega for the
    // whole computation; EMPTY otherwise (and under the U_ell lever's host
    // assembly, which evaluates Omega_i per coset and drops it).
    std::vector<FlatMatrixT<Elt>> omega_stream_;
    void ensure_omega_stream();
    const FlatMatrixT<Elt>& omega_of(uint32_t i) const;                 // rho(s_i^{-1}) in rep_
    const FlatMatrixT<Elt>& rho_of_term(const XiTermT<Elt>& xt) const;  // rho((g gamma_k) mod ell) in rep_
    void build_contragredient_tables();

    bool computed_ = false;
    bool level_a_released_ = false;   // (slot-24 G6) release_level_a() ran
    void require_level_a(const char* what) const;

    // ---- Block-model storage (the result) -------------------------------
    std::vector<std::deque<engine::DenseBuf<Elt>>> theta_storage_;   // the fused output blocks (owned; see below)
    std::shared_ptr<BlockPool> pool_;          // Theta_block(q) borrows theta_storage_
    std::vector<BlockMat> Theta_block_;        // level (a), per q, N_q x N_q
    std::vector<FlatMatrixT<Elt>> Theta_eq_;   // level (a) restricted, per q
    std::vector<MatrixT<Elt>> A_;              // level (b), per q

    // ---- Lazy FlatMatrix materialisation of Theta (the to_flat shim) ------
    mutable std::vector<FlatMatrixT<Elt>> Theta_flat_;
    mutable bool theta_flat_built_ = false;
    void ensure_theta_flat() const;

    // Certificate bundle (emitted by to_json(); see header note).
    bool routes_bit_identical_ = false;  // O12(a), asserted in compute()
    bool theta_eq_certified_ = true;     // Theta8/Theta9 ran (HeckeOperatorOptions::theta_eq_certification_enabled)
    uint64_t relation_checks_ = 0;       // runtime assertion (i): #(z_i, relation) pairs
    uint64_t cocycle_checks_ = 0;        // runtime assertion (ii): #z_i with d y_i = 0 verified
    uint64_t theta_eq_solves_ = 0;       // runtime assertion (iii): #C_eq basis vectors solved
    Elt theta_eq_residual_max_ = 0; // A-N3: max |residual entry| of the exact solve; always 0

    const SimplicialComplex* K() const; // the base 7-vertex model of coh_

    // rho_n of an exact integer matrix, reduced mod ell (invariant I2) -- in
    // the operator's fibre representation (slot-6 WP-1; SymN = rho_n).
    FlatMatrixT<Elt> rho_of_exact(const Mat22& gamma_exact) const;

    void check_degree(uint32_t q) const;

    // ---- The tape programs (matrix-engine Phase 4, Stage B) ---------------
    // compute_block() records, per degree q, ONE program for the fused
    // assembly and the dense steps Theta5/6/7/8a/10, one program per block
    // row for the STREAMED route-2 comparison (COMPARE slots against the
    // fused blocks; nothing of route 2 is stored), and one for the A-N3
    // residual; all served from the program cache by shape signature
    // (tau-pattern hash included -- plan D4's caveat).  See the .cpp header
    // note.  The fused output blocks are owned here and BORROWED by pool_
    // as Theta_block(q): theta_storage_ must be declared before Theta_block_
    // (members are destroyed in reverse order) and is a deque so that the
    // borrowed addresses never move.
    // The engine arm of compute().
    void compute_block();

#if OC_ENABLE_LEGACY
    // ---- The legacy (pre-Phase-3) dense arm ------------------------------
    // Kept verbatim (modulo the scalar arithmetic going through engine::ModCtx,
    // bit-identical to the mod_* primitives -- ModContextValidation) as the
    // ENGINE_BACKEND=legacy path and the -DENGINE_SHADOW_LEGACY oracle.
    // (S24 CL-L2) Hecke/legacy/HeckeOperatorOnKLegacy.inc -- ON builds only.
    struct LegacyResult {
        std::vector<FlatMatrixT<Elt>> theta_flat, theta_eq;
        std::vector<MatrixT<Elt>> A;
        uint64_t relation_checks = 0, cocycle_checks = 0, theta_eq_solves = 0;
        Elt residual_max = 0;
    };
    using LegacyProductKey = std::array<int64_t, 5>;
    using LegacyBlockCache = std::map<LegacyProductKey, FlatMatrixT<Elt>>;
    FlatMatrixT<Elt> assemble_theta_matrix_legacy(uint32_t q, bool fused,
                                                  LegacyBlockCache& cache) const;
    FlatMatrixT<Elt> flatten_cochain_legacy(const FlatMatrixT<Elt>& cochain) const;
    void compute_legacy(LegacyResult& out) const;
    void adopt_legacy_result(LegacyResult&& legacy);
#endif
#ifdef ENGINE_SHADOW_LEGACY
    void shadow_compare_with_legacy();   // (S24 CL-L1) Hecke/legacy/HeckeOperatorOnKShadow.inc
#endif
};

using HeckeOperatorOnK = HeckeOperatorOnKT<uint32_t>;
using HeckeOperatorOnK64 = HeckeOperatorOnKT<uint64_t>;

// (matrix-engine Phase 3 Stage B) Force the LEGACY arm of compute() (the dense
// per-cell assembly + change_of_basis_fm_multi) regardless of ENGINE_BACKEND.
// Validation-only: it lets HeckeThetaBlockValidation compare the two arms
// inside one process (ENGINE_BACKEND is read once and cached).  Global,
// defaults to off; the same shape as
// EquivariantCohomologyOptions::ScopedLegacyCohomologyPath.
namespace HeckeOperatorOptions {
#if OC_ENABLE_LEGACY   // (S24 CL-L2) ON builds only, with ScopedLegacyHeckeOperatorPath below
    bool legacy_hecke_operator_path_forced();
    void set_legacy_hecke_operator_path_forced(bool forced);
#endif

    // (matrix-engine Phase 4, memory profile 05 Sec. 5.2 / 7 item 2) Whether
    // compute() runs Theta8 / Theta9 -- the Theta_eq solve that IS runtime
    // assertion (iii) and the A-N3 residual.  Default ON (the shipped
    // certification level; every validation, the golden objects and the JSON
    // are unchanged).  OFF is the memory-lean level for large n: Theta_eq is
    // not computed (get_Theta_eq_matrices() returns 0 x 0 per degree),
    // theta_eq_solves stays 0, and to_json() records
    // "theta_eq_certified": false in the certificate block so the omission
    // is explicit, never silent.  A^{(q)} does not depend on it.
    bool theta_eq_certification_enabled();
    void set_theta_eq_certification(bool enabled);
    class ScopedThetaEqCertification {
        bool saved_;
    public:
        explicit ScopedThetaEqCertification(bool enabled) : saved_(theta_eq_certification_enabled()) { set_theta_eq_certification(enabled); }
        ~ScopedThetaEqCertification() { set_theta_eq_certification(saved_); }
        ScopedThetaEqCertification(const ScopedThetaEqCertification&) = delete;
        ScopedThetaEqCertification& operator=(const ScopedThetaEqCertification&) = delete;
    };
#if OC_ENABLE_LEGACY
    class ScopedLegacyHeckeOperatorPath {
        bool saved_;
    public:
        explicit ScopedLegacyHeckeOperatorPath(bool forced)
            : saved_(legacy_hecke_operator_path_forced()) {
            set_legacy_hecke_operator_path_forced(forced);
        }
        ~ScopedLegacyHeckeOperatorPath() { set_legacy_hecke_operator_path_forced(saved_); }
        ScopedLegacyHeckeOperatorPath(const ScopedLegacyHeckeOperatorPath&) = delete;
        ScopedLegacyHeckeOperatorPath& operator=(const ScopedLegacyHeckeOperatorPath&) = delete;
    };
#endif

    // (matrix-engine Phase 6) MEMORY-LEAN mode for the O12(a) comparison.
    // Off by default => the pre-Phase-6 behaviour verbatim (both Theta routes
    // fully assembled, then one blk::equal). On => route 2 is assembled and
    // compared ONE BLOCK ROW AT A TIME against the finished fused Theta and
    // freed as it goes, so the campaign's large-n_sym jobs never hold two
    // Theta's worth of blocks. The CERTIFICATE IS UNCHANGED: every entry of
    // every degree is still compared, and a mismatch throws the same message.
    // Value-preserving in both directions (GoldenRegressionValidation and
    // HeckeThetaBlockValidation gate it).
    bool memory_lean();
    void set_memory_lean(bool lean);
    class ScopedMemoryLean {
        bool saved_;
    public:
        explicit ScopedMemoryLean(bool lean) : saved_(memory_lean()) { set_memory_lean(lean); }
        ~ScopedMemoryLean() { set_memory_lean(saved_); }
        ScopedMemoryLean(const ScopedMemoryLean&) = delete;
        ScopedMemoryLean& operator=(const ScopedMemoryLean&) = delete;
    };

    // ===== The three memory / time levers of the U_ell pipeline ===============
    // (hecke-extensions slot 13 WP-4; slot 14 WP-10.)  OWNER DECISION
    // 2026-09-28 (Q8): all three are ON BY DEFAULT wherever they apply, each
    // with an explicit opt-out -- a scoped switch below and an environment
    // variable read as the thread's initial state -- so the old path stays
    // reachable as the rollback arm and the bit-identity oracle
    // (HeckeUEllStreamValidation compares the default against the opt-out).
    // None is a mathematical setting: none can change a value (every block is
    // the same matrix, bracketed or scheduled differently), so none enters a
    // fingerprint, a record or a job id.  Thread-local, propagated into the
    // engine's host fan-out like the options above.
    //
    // RETIRED (S24 CL-R1; owner ruling 2 of 2026-10-30; register:
    // docs/validation-catalogue.md, "Rollback-lever register").  The four
    // levers (1)-(4) are engine::lever::RetiredLever: with OC_ENABLE_LEGACY=OFF
    // (the default build) each getter returns the default route, the opt-out
    // `<NAME>=0` is REFUSED (stderr, exit 1) at the first read, and the setters
    // and Scoped* guards below do not exist (legacy/HeckeOperatorLeversLegacy.h,
    // ON builds only, where every lever behaves as described here).  The dense
    // U_ell route (ell not dividing N, the non-SymN fibres) and the dense
    // factoriser (non-field rings) stay: they are production routes.
    //
    // (1) THE U_ell LOW-RANK LEVER (slot-13 WP-4).  When the operator's
    // g = diag(1, N) has ell | N (U_ell, T_{ell^r}), every g gamma_k and every
    // delta_i gamma_k is singular mod ell, so rho_n(g gamma_k) mod ell^m is a
    // sum of <= m outer products (rank 1 over F_ell; [S13] WP-4, [AUD11] M-3).
    // With the lever in effect:
    //   * F_ell (compute_block): the fused blocks are assembled on the host as
    //     sum_t cmod_t (Omega_i U_slot) V_slot, with rho_pool slot = U V
    //     factored once (exact min-valuation elimination; the term count <= m
    //     is ENGINE_CHECKed per block -- the rank claim), instead of one
    //     d_V x d_V product temp per distinct (i, slot) pair on the tape;
    //     route 2 factors every rho_n((delta_i gamma_k) mod ell) the same way
    //     and discards the dense block, then compares block by block on the
    //     host.  Program 1 reads the finished blocks as inputs (its own
    //     program-cache name, "hecke.theta.lr").
    //   * Z/ell^k (compute_module): the fused memo and route 2 keep the
    //     factored (<= k outer products) forms instead of dense d_V x d_V
    //     blocks.
    //   * HeckeTauTable: a table CONSTRUCTED with the lever not Off and
    //     ell | N keeps its rho pool factored (HeckeTauTable::rho_pool_low_rank;
    //     dense reads through HeckeTauTable::rho_pool_block(), RhoRefT::get()
    //     throws).  A SymN operator over such a table must have the lever in
    //     effect (ENGINE_CHECKed: an Off computation needs an Off table).
    // Modes: Auto (the DEFAULT since the owner decision) -- in effect exactly
    // where it applies, ell | N and the SymN fibre, silently off elsewhere;
    // On (ScopedUEllLowRank(true), `--bench --u-ell-low-rank`) -- forced,
    // REFUSED (ENGINE_CHECK) where it does not apply, the slot-13 behaviour;
    // Off (ScopedUEllLowRank(false), `--bench --no-u-ell-low-rank`,
    // HECKE_U_ELL_LOW_RANK=0) -- the pre-WP-4 path.  Measured (WP-4 report):
    // `--bench hecke 128 691 691` 2745 -> 574 MB peak, 3.36 -> 1.96 s.
    enum class UEllLowRankMode { Auto, On, Off };
    // (S24 CL-R1) RETIRED lever: in an OC_ENABLE_LEGACY=OFF build the mode is
    // always Auto and HECKE_U_ELL_LOW_RANK=0 is REFUSED at the first read; the
    // setter and ScopedUEllLowRank are in legacy/HeckeOperatorLeversLegacy.h.
    UEllLowRankMode u_ell_low_rank_mode();
    // Whether the lever is in effect for an operator (N, ell, fibre) under the
    // current mode (Auto: iff ell | N and SymN; On: always -- the operator
    // then refuses where it does not apply; Off: never).
    bool u_ell_low_rank_in_effect(uint64_t N, uint64_t ell, FibreRepresentation rep);

    // (2) THE COSET STREAMING LEVER (slot-14 WP-10; default ON, opt-out
    // ScopedCosetStreaming(false), `--bench --no-coset-stream`,
    // HECKE_COSET_STREAM=0).  A HeckeCosetTables CONSTRUCTED with it on stores
    // its exact data only (the unimodular Mat22 s_i, s_i^{-1}, t_i,
    // h_i(gamma), 16 B each) and none of its ten rho_n tables (rho_n(s_i^{-1}),
    // rho_n(t_i), 4 x rho_n(h_i(gamma)), 4 x the Gamma^0(N) cocycles: [S13]
    // WP-4's 34 B per matrix entry per coset, the U_ell memory floor under the
    // lever).  Consumers read a block through the *_block() accessors, which
    // evaluate rho_n on the fly by the SAME function (rho_of) the table would
    // have stored -- the same bytes, the same scalar flag -- so every value,
    // program, pattern hash and cache key is the opt-out one.  The reference
    // accessors (rho_s_inv(i), ...) refuse a streamed table by ENGINE_CHECK.
    // HeckeOperatorOnK: with the U_ell lever in effect (compute_block) Omega_i
    // is evaluated once per coset into a task-local block and dropped; on the
    // other field paths (the tape's borrowed operands, the legacy arm) the
    // operator materialises its own Omega table once (4 of the 34 B); the
    // module route computes Omega from the exact s_i^{-1} as before.
    bool coset_streaming();
    // (S24 CL-R1) RETIRED: always on in an OFF build; set_coset_streaming / ScopedCosetStreaming: legacy/.

    // (3) THE SERIAL PHASE ON THE POOL (slot-14 WP-10; default ON, opt-out
    // ScopedParallelSetup(false), `--bench --no-parallel-setup`,
    // HECKE_PARALLEL_SETUP=0).  The per-block rho_n work of the setup phase
    // runs as host tasks on the consumer pool (engine::tape::run_concurrently,
    // the slot-10 L4 / I-1 pattern) whenever the phase is large enough to pay
    // for the fan-out (items x d_V^2 >= kParallelSetupMinWork; below it the
    // serial loop runs -- the choice cannot move a value either way):
    //   * HeckeCosetTables: its rho_n tables per (gen, coset) (read at
    //     construction; nothing to do on a streamed table);
    //   * HeckeTauTable: the rho pool per slot (dense, or factored under the
    //     U_ell lever) -- the slots, keys and visit order are assigned by the
    //     serial walk first, each task writes its pre-assigned slot from the
    //     serial walk's representative gamma;
    //   * HeckeTauTable::certify() clause (f): one task per distinct reduced
    //     key (per (key, slot) pair under the U_ell lever), each comparing
    //     exactly the blocks the serial loop compares; every (f) failure is
    //     code 60, so the verdict and the code are the serial ones;
    //   * HeckeOperatorOnK: the streamed Omega table (above).
    // Each task computes exactly the serial object; the first failing task's
    // exception, in task order, is what surfaces.  With one consumer-pool
    // worker every fan-out runs serially, in order.
    bool parallel_setup();
    // (S24 CL-R1) RETIRED: always on in an OFF build; set_parallel_setup / ScopedParallelSetup: legacy/.

    // (4) THE RANK-ONE CLOSED FORM (slot-14 G0; default ON, opt-out
    // ScopedRankOneClosedForm(false), HECKE_RANK1_CLOSED_FORM=0).  Where the
    // U_ell lever factors rho_n(g) of a key g singular mod ell over F_ell --
    // HeckeTauTable's rho pool and route 2 of compute_block -- the factors
    // (U, V) are computed in O(d_V) by lowrank::rank_one_rho_factors instead
    // of building the dense d_V x d_V block and factoring it (O(d_V^2)).  The
    // SAME (U, V), byte for byte (factor_low_rank's normalisation), so every
    // block, program, hash and value is the opt-out one; the rank claim stays
    // ENGINE_CHECKed per key (det g == 0 mod ell), and certify() clause (f)
    // keeps its dense witness of every pool slot.  The opt-out is the
    // bit-identity oracle (HeckeUEllStreamValidation arm (H)).  Not applied
    // over Z/ell^k (the module route: <= k terms, no closed form here).
    bool rank_one_closed_form();
    // (S24 CL-R1) RETIRED: always on in an OFF build; set_rank_one_closed_form / ScopedRankOneClosedForm: legacy/.

    // (5) ROUTE 2 STREAMED (S23 P4R; default ON, rollback
    // ScopedRho2Streaming(false), HECKE_RHO2_STREAM=0; the env arm is RETIRED in
    // OC_ENABLE_LEGACY=OFF builds by S25 LEVERS-25, the Scoped override stays).  compute_block()'s
    // dense route 2 (the O12(a) cross-check, [C, eq:heckechain]) no longer
    // materialises the per-degree memo rho2 (one d_V x d_V FlatMatrix per
    // distinct reduced key (delta_i gamma_k) mod ell, ~1.4 GB at (n, ell, p) =
    // (700, 13, 103)): it keeps one d_V x d_V accumulator per route-2 block
    // (as many as the fused blocks) and takes the distinct keys in ascending
    // slot order in batches of 64 (kRho2StreamBatch): the batch's
    // rho_n are evaluated in the host fan-out, each key once, added as
    // cmod . rho_n into every accumulator that uses them (a fan-out over the
    // blocks), and released; then every block is compared entrywise with its
    // fused block (rows in sigma order, as the materialised route).  Route 2 only
    // yields a verdict, so no value can move; the rollback is the materialised
    // memo and its "hecke.route2" programs, unchanged.  The U_ell lever's
    // factored route 2 (rho2_lr, O(d) per key) is not affected.  When the
    // environment variable is set (any value) the first compute_block() of the
    // process prints one line `EFFECTIVE HECKE_RHO2_STREAM route2=<streamed|
    // materialised> (HECKE_RHO2_STREAM=<v>)`.  Gated by HeckeRho2StreamValidation.
    bool rho2_streaming();
    void set_rho2_streaming(bool on);
    class ScopedRho2Streaming {
        bool saved_;
    public:
        explicit ScopedRho2Streaming(bool on) : saved_(rho2_streaming()) { set_rho2_streaming(on); }
        ~ScopedRho2Streaming() { set_rho2_streaming(saved_); }
        ScopedRho2Streaming(const ScopedRho2Streaming&) = delete;
        ScopedRho2Streaming& operator=(const ScopedRho2Streaming&) = delete;
    };
    // (6) THE FIELD-ROUTE RSS DIET (slot-24 G6; default ON, rollback
    // ScopedFieldDiet(false), HECKE_FIELD_DIET=0; exactly "0" selects the
    // rollback).  Two memory-only levers of the census field route:
    //   (a) compute_block()'s dense program 1 IN PARTS: when its CSE temps (the
    //       distinct non-scalar products Omega_i . Rho used by two or more
    //       terms, all live at once in the arena: 2.96 GB at (n, ell, p, q) =
    //       (1000, 13, 103, 1)) would exceed theta_part_budget() (default
    //       kThetaPartBudgetBytes), the products are cut, in first-use order,
    //       into parts of at most budget / (d_V^2 sizeof(Elt)) shared products,
    //       each recorded and run as its own program accumulating into the
    //       fused blocks; program 1 then reads the finished blocks as inputs,
    //       exactly as under the U_ell lever.  Mod ell the block sums are
    //       order-free, so every block, Y, W, Theta_eq and A^{(q)} is the same.
    //       Below the budget the single program runs as before.
    //   (b) HeckeExtDriver releases its HeckeOperatorFamily's level (a) and
    //       tables after Stage 1 (HeckeOperatorFamily::release_level_a): the
    //       cell only reads A^{(q)} afterwards.
    // The same switch (Util/FieldDiet.h, one thread-local flag) also selects
    // the cone certificate's unit-row span route and the cone's shared Z basis
    // (EquivariantMappingCone.h, ConeOptions) and the boundary morphism's lean
    // basis read (EquivariantCochainComplexMorphismOnK.cpp).
    // When the environment variable is set (any value) the first reader of
    // the process prints one line `EFFECTIVE HECKE_FIELD_DIET field_diet=<on|off>
    // (HECKE_FIELD_DIET=<v>)`.  Gated by HeckeThetaRssDietValidation's G6 arm.
    constexpr uint64_t kThetaPartBudgetBytes = uint64_t(256) << 20;
    bool field_diet();
    void set_field_diet(bool on);
    class ScopedFieldDiet {
        bool saved_;
    public:
        explicit ScopedFieldDiet(bool on) : saved_(field_diet()) { set_field_diet(on); }
        ~ScopedFieldDiet() { set_field_diet(saved_); }
        ScopedFieldDiet(const ScopedFieldDiet&) = delete;
        ScopedFieldDiet& operator=(const ScopedFieldDiet&) = delete;
    };
    // Validation hook: the part budget in bytes on the calling thread (a small
    // budget parts the program at small n_sym).
    uint64_t theta_part_budget();
    class ScopedThetaPartBudget {
        uint64_t saved_;
    public:
        explicit ScopedThetaPartBudget(uint64_t bytes);
        ~ScopedThetaPartBudget();
        ScopedThetaPartBudget(const ScopedThetaPartBudget&) = delete;
        ScopedThetaPartBudget& operator=(const ScopedThetaPartBudget&) = delete;
    };
    // (PERF-VF audit S4) Validation hook, calling thread only: after a bound-decided skip of the part
    // pre-walk (slot-25 PERF-VF (b1)) run the pre-walk anyway and ENGINE_CHECK that it would not have
    // parted (and that shared <= floor(#terms / 2)).  Off by default; it never changes a value.
    bool prewalk_audit();
    class ScopedPrewalkAudit {
        bool saved_;
    public:
        explicit ScopedPrewalkAudit(bool on);
        ~ScopedPrewalkAudit();
        ScopedPrewalkAudit(const ScopedPrewalkAudit&) = delete;
        ScopedPrewalkAudit& operator=(const ScopedPrewalkAudit&) = delete;
    };
    // Process-wide monotone counters (diagnostic, never a value): dense
    // degrees whose program 1 ran in parts / whole under the diet, parts run.
    //   (c) after each streamed Sturm / pool prime (and the family release) the
    //       allocator's free pages are returned to the system
    //       (release_free_heap(): glibc malloc_trim(0); a no-op elsewhere), so
    //       the freed tables of one prime do not ratchet the cell's floor.
    void release_free_heap();
    // Diagnostic only (the ENGINE_TRACE_RSS lines): the allocator's bytes in
    // use (glibc mallinfo2: uordblks + hblkhd), MiB; -1 where unavailable.
    double heap_in_use_mib();
    // prewalk_skipped (slot-25 PERF-VF (b1)): of the whole degrees, those whose part pre-walk was
    // skipped because the bound floor(#terms / 2) * d_V^2 sizeof(Elt) already fits the budget.
    // prewalk_audited (PERF-VF audit S4): of those, the skips re-checked by the pre-walk under ScopedPrewalkAudit.
    struct FieldDietStats { uint64_t parted_degrees = 0, parts = 0, whole_degrees = 0, prewalk_skipped = 0, prewalk_audited = 0; };
    FieldDietStats field_diet_stats();
    void field_diet_count(bool parted, uint64_t parts);
    void field_diet_count_prewalk_skipped();
    void field_diet_count_prewalk_audited();

    // Process-wide monotone counters of the dense route 2 (diagnostic, never a
    // value): degrees run streamed / materialised, distinct keys, rho_n
    // evaluations of the streamed route (== keys: each key once), route-2
    // blocks compared by the streamed route.
    // mid_reductions counts the delayed accumulation's in-loop reductions
    // (per block and chunk; 0 in production unless k_max <= 64).
    struct Rho2StreamStats {
        uint64_t streamed_degrees = 0, materialised_degrees = 0, keys = 0, evals = 0, blocks = 0, mid_reductions = 0;
    };
    Rho2StreamStats rho2_stream_stats();
    // VALIDATION-ONLY hooks (HeckeRho2StreamValidation), thread-local, read once
    // per compute() on the calling thread; both null / off in production.
    //   ScopedRoute2Capture: every dense route-2 block (q, sigma, sigma_k) is
    //     copied (row-major, d_V x d_V) into the capture -- streamed: the row's
    //     accumulator; materialised: sum cmod . rho2[slot] over the memo.
    //   ScopedRoute2TargetMutation: the fused block (q, sigma, sigma_k) that
    //     route 2 is COMPARED against is a copy with entry (r, c) + 1, so
    //     compute() must throw O12(a) naming exactly that entry.
    struct Route2Capture {
        std::mutex mu;
        std::map<std::tuple<uint32_t, uint32_t, uint32_t>, std::vector<uint64_t>> blocks;
    };
    Route2Capture* route2_capture();
    class ScopedRoute2Capture {
        Route2Capture* saved_;
    public:
        explicit ScopedRoute2Capture(Route2Capture* c);
        ~ScopedRoute2Capture();
        ScopedRoute2Capture(const ScopedRoute2Capture&) = delete;
        ScopedRoute2Capture& operator=(const ScopedRoute2Capture&) = delete;
    };
    struct Route2TargetMutation {
        bool on = false;
        uint32_t q = 0, sigma = 0, sigma_k = 0, r = 0, c = 0;
    };
    const Route2TargetMutation& route2_target_mutation();
    class ScopedRoute2TargetMutation {
        Route2TargetMutation saved_;
    public:
        explicit ScopedRoute2TargetMutation(const Route2TargetMutation& m);
        ~ScopedRoute2TargetMutation();
        ScopedRoute2TargetMutation(const ScopedRoute2TargetMutation&) = delete;
        ScopedRoute2TargetMutation& operator=(const ScopedRoute2TargetMutation&) = delete;
    };
    //   ScopedRho2StreamReduceEvery (S23 P4R audit F3): the streamed delayed
    //     accumulation reduces after every r products instead of k_max - 1
    //     (r is clamped to [1, k_max - 1], so no overflow and the same
    //     residues); 0 = off.  Exercises the in-loop reduction branch, which
    //     production reaches only when k_max <= 64.
    uint64_t rho2_stream_reduce_every();
    class ScopedRho2StreamReduceEvery {
        uint64_t saved_;
    public:
        explicit ScopedRho2StreamReduceEvery(uint64_t r);
        ~ScopedRho2StreamReduceEvery();
        ScopedRho2StreamReduceEvery(const ScopedRho2StreamReduceEvery&) = delete;
        ScopedRho2StreamReduceEvery& operator=(const ScopedRho2StreamReduceEvery&) = delete;
    };

    // The fan-out the levers use: body(j) for j in [0, n_items), every result
    // slot pre-assigned by the caller, so the schedule cannot reorder a thing.
    // Serial when `parallel` is false or n_items * work_per_item <
    // kParallelSetupMinWork (a fan-out costs a few thread wake-ups: the
    // standing T_2 benchmarks must not pay for it).
    //
    // (slot-14 G0, audit M-2) THE ONE HOST FAN-OUT BUDGET.  Every data-parallel
    // fan-out of the U_ell pipeline (this function; compute_block's host
    // assembly and route-2 evaluation go through it too) runs on ONE
    // process-wide set of helper threads (created lazily, up to the sum over
    // the consumer pools with an open fan-out of their workers minus one)
    // plus the calling thread.  THE BUDGET, PER CONSUMER POOL (G0 repairs,
    // audit S1): at most (the fan-out's consumer-pool workers - 1) helpers are
    // busy at once over all fan-outs OF THAT POOL together, and at most
    // min(workers, n_items) threads work on one fan-out; so the per-job pools
    // of the campaign job loop each keep their own threads - 1 helpers, and
    // one pool (a single cell) is exactly the G0 budget.  The indices are claimed dynamically (an atomic
    // counter per fan-out).  So fan-outs nested inside the structural fan-outs (the
    // degrees of compute(), the (p, r) pipelines of HeckeOperatorFamily)
    // SHARE the budget instead of each spawning its own threads (was: up to
    // P x 32 + 3 x 32 threads on a 32-worker pool), and an idle helper moves
    // to whichever fan-out still has work, so an unbalanced degree is not
    // starved.  The caller always works on its own fan-out, so a nested one
    // cannot deadlock.  Each helper installs the caller's captured thread
    // context (engine::tape::capture_thread_context) for the fan-out it
    // joins.  Failure rule: the exception of the LOWEST failing index is
    // rethrown (indices above a known failure are skipped), which is exactly
    // the serial loop's first failure.  Serial with one consumer-pool worker
    // or under ENGINE_HOST_THREADS=0.
    constexpr uint64_t kParallelSetupMinWork = uint64_t(1) << 22;
    void for_each_index(size_t n_items, bool parallel, const std::function<void(size_t)>& body,
                        uint64_t work_per_item = kParallelSetupMinWork);
}

} // namespace hecke

#if OC_ENABLE_LEGACY
#include "legacy/HeckeOperatorLeversLegacy.h"   // (S24 CL-R1) the retired slot-13/14 levers' setters and Scoped*
#endif

#endif // HECKE_OPERATOR_ON_K_H
