// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_TAU_TABLE_H
#define HECKE_TAU_TABLE_H

// Stage S4 -- HeckeTauTable: the certified g-twisted chain datum tau of
// [C, def:tau], built by [C, alg:tau], with the MANDATORY exact
// certification of [C, obl:tau] and the certified integer bound B_tab of
// [C, lem:bounds](5).
//
// Contract: docs/latex/hecke_action.tex ([C]):
//   * [C, def:tau] (T1)-(T3): tau : C_*(K^u) -> C_*(K^u) additive with
//       (T1) tau(h.x) = (g^{-1} h g) . tau(x) for h in Gamma'' = Gamma_0(N),
//       (T2) boundary tau = tau boundary,
//       (T3) augmentation epsilon(tau(x)) = epsilon(x) in degree 0.
//   * [C, alg:tau]: the 4-step construction (vertices by certified geometric
//     reduction with the iota_K-equivariant tie-break of step (1b); edges by
//     (P2) word reduction + the certified canonical tree-routed path
//     templates of step 2 (SL2ZTessellationTables::PathTemplateFamily);
//     triangles by finite-neighborhood integer (or mod-ell) linear solve;
//     MANDATORY exact certification). Since the re-baseline of 2026-09-16
//     the canonical tau is iota-SYMMETRIC, iota_X tau iota_X = tau
//     ([C, prop:iotacommutes](3b), [C, obl:iota](iv), gated by
//     HeckeIotaTauSymmetryValidation); the two ingredients are the tie-break
//     and the tree routing, and neither alone suffices. A consequence of the
//     tree routing: tau(boundary(triangle)) is a closed loop of tree paths,
//     hence 0, so the canonical tau VANISHES in degree 2 (Theta^{(2)}_flat =
//     0). Admissible ((T1)-(T3) hold), and A^{(q)} is unchanged on the whole
//     golden grid (dim H^2_eq = 0 there).
//   * [C, lem:bounds](5) / [C, thm:backend] / [C, obl:backend]: the
//     build-time-certified magnitude bound B_tab = max ||gamma_k|| and the
//     whole-pipeline bound B = max(B_tab, 4 N^2), asserted to fit the
//     selected backend tier (int64: 2 B^2 < 2^63).
//   * [C, eq:xiexplicit]: the eagerly-reduced consumer table xi_terms
//     (rho_n(g gamma_k) mod ell, [C, def:semigroup] semigroup extension --
//     det(g gamma_k) = N and it is NEVER inverted, invariant I7 /
//     [C, obl:integral]).
//
// ===========================================================================
// INVARIANT I8 (binding; the auditor's overriding concern): CERTIFICATION
// OVER GEOMETRY. The geometric choices below (nearest vertex, path
// templates, neighborhood growth) only PROPOSE the table; the exact,
// unconditional certification (certify(), called by the constructor)
// DISPOSES. A table failing certification is REJECTED: the constructor
// throws and the object can never exist in an uncertified state. The
// certification is never downgraded to a warning and never skipped; the
// correctness of the resulting Hecke operator rests only on it
// ([C, obl:tau] and the remark after [C, alg:tau]).
// ===========================================================================
//
// What is stored. tau is determined by (T1) on the Gamma''-orbits of
// K^u-simplices; the table stores one chain per pair (i, sigma),
// i in [1, m] (S3's 1-based coset indices), sigma in o_q(K), q = 0,1,2 --
// i.e. per orbit representative x = s_i . i(sigma). For p = 2 (m = 3) that
// is exactly 3 * (7 + 11 + 5) = 69 stored chains ([C, sec:example]: small
// enough to inspect by hand; see docs/implementation/tau_table_p2.txt).
//
// NOTE (model fact, deliberate deviation from a parenthetical in the plan's
// S4 risk list): the pairs (i, sigma) are NOT pairwise Gamma''-inequivalent
// -- o_q(K) itself is not a section of the Gamma-groupoid orbits (e.g.
// i(v3) = T . i(v0), so s_i . i(v3) = h_i(T) . s_{i.T} . i(v0) with
// h_i(T) in Gamma_0(N)). tau must therefore assign CONSISTENT values to
// Gamma''-related slots. This implementation constructs exactly one slot
// per Gamma''-orbit geometrically and derives every other slot through the
// single shared (T1) translation helper (eval_tau_on_translate, used by
// BOTH construction and certification); the certification then verifies the
// full twisted-equivariance bookkeeping over ALL stored transporters
// (obl:tau lists it explicitly), NOT the plan's (unsatisfiable) pairwise-
// inequivalence assertion.
//
// Elliptic slots. When a slot s_i . i(v) has a nontrivial stabilizer in
// Gamma'' (mod +-I), (T1) FORCES the assigned vertex: the point
// z = g^{-1} s_i . pos(v) is then an exact elliptic fixed point of Gamma,
// its reduction lands exactly on pos(v4)/pos(v5)/pos(v6), and any valid tau
// must choose that vertex. GeometricNearest does so automatically (distance
// 0 is the unique minimum); FixedV6 applies an exact-position-hit override
// for the same reason (documented at the enum). Either way the certification
// -- not this geometric argument -- is what accepts the table (I8).
//
// adj(g) projective trick (NO rationals anywhere): step 1 needs the point
// z = (g^{-1} s_i) . pos(v). Instead of the rational matrix g^{-1} =
// diag(1, 1/N) we act by adj(g) = diag(N, 1) = N * g^{-1}: the Moebius
// action is projective (PGL_2^+), so the integral matrix adj(g) s_i (det
// N > 0) acts on the upper half-plane exactly as g^{-1} s_i does. All
// arithmetic stays exact and overflow-checked over Z (invariant I5).
//
// Solve regime (recorded, per [C, alg:tau](3)): triangle fills are solved
// over Z first (hand-rolled integer column-echelon elimination with
// divisibility checks; no HNF library, no arbitrary precision); if the
// integer solve is infeasible (or overflows its checked int64 arithmetic)
// the SAME system is solved mod ell and the affected chains are flagged;
// certification of every identity involving a flagged chain is then exact
// mod ell ([C, alg:tau](4)), otherwise exact over Z. If neither solve
// succeeds the neighborhood is grown by one ring of adjacent translates
// (generator moves T, T^{-1}, S) and the solve is retried; termination is
// guaranteed by the acyclicity of K^u. used_mod_ell_solve() reports whether
// ANY slot was solved mod ell (the recorded datum of [C, alg:tau](3)).
//
// ell in {2,3}: the canonical tau exists (the construction never divides by
// 2 or 3 -- the integer solve is preferred and division only happens
// through exact divisibility checks); homotopy uniqueness is NOT asserted
// (contract box after [C, lem:tauexists]) and NO cross-tau assertions are
// made anywhere for ell in {2,3}.
//
// Placement (documented per the task's instructions): like S2/S3, this
// class depends on FlatMatrix / ComputeSymmetricPowerMatrixFM
// (ModArithmetic) and on the SimplicialComplex model, so it is compiled
// into the cohomology_app executable sources (NOT into the pure-integer
// HeckeCore static library), while linking HeckeCore for Mat22/QuadPoint.

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ExactInt.h"
#include "QuadPoint.h"
#include "SL2ZTessellationTables.h"
#include "HeckeCosetTables.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/FlatMatrix.h"
#include "HeckeLowRank.h"   // (slot-13 WP-4) the factored rho pool

class SimplicialComplex;

namespace hecke {

// One term  n_k * (gamma_k . i(sigma_k))  of tau(s_i . i(sigma))
// ([C, alg:tau] output triples). Terms are stored in CANONICAL form:
// sigma_idx is a groupoid-orbit representative of K's q-simplex list and
// gamma is the lexicographically minimal transporter representative
// (KuSimplexKey with sign folded into coeff), so chain equality is
// term-by-term equality. For chains produced by a mod-ell triangle solve
// the coefficient is the balanced residue in (-ell/2, ell/2].
struct ChainTerm {
    int32_t coeff;      // n_k in Z (small; nonzero)
    Mat22 gamma;        // gamma_k in Gamma, exact (kept: certification & O12)
    uint32_t sigma_idx; // sigma_k in o_q(K), same q as the source simplex
};

// Handle on one entry of a HeckeTauTable's deduplicated rho_n pool.
//
// rho_n((g gamma) mod ell) is a pure function of the REDUCED 2x2 matrix
// (g gamma) mod ell, and a whole tau table realizes only a handful of distinct
// reduced keys (8 at p = 2) across its ~125 terms. Storing the (n+1)x(n+1)
// matrix by value in every term therefore cost #tau * d_V^2 * 4 B -- 501 MB at
// n_sym = 1000 (audit 02 Section 5.2 / item 7.3). A RhoRefT holds instead a
// shared pointer to the single immutable pool plus a 32-bit slot index.
//
// It behaves as the matrix it names: it converts implicitly to
// `const FlatMatrixT<Elt>&` and forwards the const observers, so an expression
// like `omega_i * xt.rho_g_gamma` or `xt.rho_g_gamma.get_entry(r, c)` reads and
// compiles exactly as it did when the member was a matrix by value.
//
// The pool is built once, never mutated afterwards, and kept alive by the
// shared_ptr each handle carries, so get() stays valid for as long as the term
// (or any copy of it, including copies outliving the table) does.
template<class Elt>
class RhoRefT {
public:
    using Pool = std::vector<FlatMatrixT<Elt>>;

    RhoRefT() = default;
    RhoRefT(std::shared_ptr<const Pool> pool, uint32_t idx)
        : pool_(std::move(pool)), idx_(idx) {}

    const FlatMatrixT<Elt>& get() const {
        ENGINE_CHECK_MSG(pool_ && idx_ < pool_->size(),
                         "RhoRefT: handle does not name a pooled rho_n matrix.");
        // (slot-13 WP-4) A table built under the U_ell lever keeps its pool
        // FACTORED (HeckeTauTableT::rho_pool_low_rank) and leaves a 0 x 0
        // placeholder here; a dense read of it is a caller error, never a
        // silent empty matrix.  (No real rho_n block is 0 x 0: d_V >= 1.)
        ENGINE_CHECK_MSG((*pool_)[idx_].get_rows() != 0,
                         "RhoRefT: this tau table was built under the U_ell low-rank lever -- its rho pool is "
                         "factored; read HeckeTauTable::rho_pool_low_rank(), or build without the lever.");
        return (*pool_)[idx_];
    }
    operator const FlatMatrixT<Elt>&() const { return get(); } // NOLINT: by design

    // Pool slot. Dedup is by the exact reduced key, so equal indices mean the
    // identical matrix and distinct indices mean distinct matrices; consumers
    // may memoise products on (index(), ...) instead of on matrix values.
    uint32_t index() const { return idx_; }

    // Forwarded const observers of FlatMatrixT (source compatibility).
    Elt get_p() const { return get().get_p(); }
    uint32_t get_rows() const { return get().get_rows(); }
    uint32_t get_cols() const { return get().get_cols(); }
    Elt get_entry(uint32_t r, uint32_t c) const { return get().get_entry(r, c); }
    bool operator==(const FlatMatrixT<Elt>& other) const { return get() == other; }

private:
    std::shared_ptr<const Pool> pool_;
    uint32_t idx_ = 0;
};

using RhoRef = RhoRefT<uint32_t>;
using RhoRef64 = RhoRefT<uint64_t>;

// Eagerly-reduced consumer form of [C, eq:xiexplicit]: one summand
// n_k * rho_n(g gamma_k) c(sigma_k) of (Xi_g c)(sigma)_i.
template<class Elt>
struct XiTermT {
    int32_t coeff;
    uint32_t sigma_idx;
    RhoRefT<Elt> rho_g_gamma; // -> rho_n((g gamma_k) mod ell) in the table's
                              // pool; det(g gamma_k) = N, NEVER inverted
                              // (invariant I7 / [C, obl:integral])
};

using XiTerm = XiTermT<uint32_t>;
using XiTerm64 = XiTermT<uint64_t>;

// Vertex-assignment policy of [C, alg:tau](1). Correctness NEVER depends on
// the policy (I8: any assignment satisfies (T3); certification disposes);
// it only steers support sizes, and provides the O12(c) alternate-tau hook.
enum class TauVertexPolicy {
    // Binding default ([C, alg:tau](1b), re-baselined 2026-09-16): nearest of
    // the 7 model vertices to the reduced point, by exact squared euclidean
    // distance on cleared denominators (sign_lin_sqrt3 for mixed D in {1,3});
    // ties broken by the iota_K-EQUIVARIANT rule of choose_vertex(): a tied
    // set containing a full iota_K-orbit goes to v6 (the unique iota_K-fixed
    // vertex), otherwise to its member in the orbit of smallest index. This
    // is what makes the canonical tau iota-symmetric, iota_X tau iota_X = tau
    // ([C, prop:iotacommutes](3b), [C, obl:iota](iv)), together with the
    // canonical (tree + spoke) path templates of step 2.
    GeometricNearest,
    // The pre-2026-09-16 canonical tau, kept for the record and for the
    // differential validations: the same nearest-vertex rule with ties broken
    // by ASCENDING VERTEX INDEX (not iota_K-equivariant, [C, rmk:iotatausym])
    // and the legacy BFS path templates of step 2. Equal A^{(q)} wherever
    // 6 is invertible ([C, thm:welldef](b)); different Theta_flat.
    GeometricNearestLegacy,
    // O12(c) hook: always assign v6 = i, EXCEPT when the reduced point hits
    // a vertex position exactly (then that vertex is chosen). The exact-hit
    // override is mandatory for slots with nontrivial Gamma''-stabilizer,
    // where (T1) forces the elliptic vertex (see the header note); it is
    // harmless elsewhere.
    FixedV6,
    // (hecke-extensions slot 6, WP-4) The STRICT-PAIR datum tau^s of
    // [C, prop:strictpair]: a g-twisted chain datum over Z on X_bulk with
    // tau^s o j = j o tau_bd, so that eta == 0 is a pair homotopy for
    // (tau^s, tau_bd). Built ONLY by the constructor that takes the boundary
    // table (the plain constructor refuses this value), by the proposition's
    // terminating procedure:
    //   (B) every boundary slot s_i . i(sigma), sigma in K_bd, is
    //       PRE-ASSIGNED j tau_bd(s_i . i(sigma)) verbatim;
    //   (V) every interior vertex slot as GeometricNearest -- the exact
    //       elliptic hit first (for a slot with twisted stabiliser of order
    //       4 or 6 that is the fixed vertex the proposition requires), the
    //       nearest vertex otherwise ("any vertex" for {+-I});
    //   (E) every interior edge slot by the canonical path templates (any
    //       1-chain with the right boundary);
    //   (F) every triangle slot by the unique fill, solved over Z ONLY:
    //       the int64 tier first, the widened __int128 tier after a
    //       HeckeOverflow ([OD] OP2); the mod-ell fallback of the shipped
    //       policies is NEVER used, so the table is a datum over Z for every
    //       ell and solve_regime() is "Z" by construction.
    // The table carries its own mandatory certification (every clause of
    // certify() plus the strictness clause (g)), its own fill search and its
    // own records; the three shipped policies and the default are untouched.
    // It is a CROSS-ROUTE of [C, prop:coneroutes](1) / [C, obl:coneexact](vii)
    // and is never the pipeline's tau.
    BoundaryStrict,
};

template<class Elt> class HeckeBoundaryTauTableT; // Hecke/HeckeBoundaryDatum.h (it includes this header)

// (S23 G2-F; slot23/G2CDE_report.md Sec. 4) What a table STORES beside its chain records.
//   Full        the shipped table: the chain records, the eagerly-reduced xi-term table ([C, eq:xiexplicit]) and the
//               deduplicated rho_n pool behind it (dense, or factored under the U_ell lever), certified by every
//               clause of certify(), (f) included.
//   ChainsOnly  the chain records only -- the same records, built by the same steps 1-3 in the same order, so the same
//               bytes, the same B_tab / B and the same regime -- and NO xi-term table and NO rho pool.  For the MODULE
//               route (HeckeOperatorOnKT over Z/ell^k), which reads tau() and the exact coset data only and evaluates
//               rho_n mod ell^k itself; the field route's accessors (xi_terms, rho_pool*, the pool corrupter) throw a
//               named std::logic_error.  certify() runs (a)-(e) and (g) exactly as on a Full table; (f) has nothing to
//               certify.  Selected by hecke_ext::RunModuleT on the stream route (engine::elim::module_stream_enabled(),
//               rollback HECKE_MODULE_STREAM=0 = Full); no value depends on it.
enum class TauTableContents { Full, ChainsOnly };

// The certified tau-table for g = diag(1, N), N = p_hecke^r.
template<class Elt>
class HeckeTauTableT {
public:
    // K must be the 7-vertex model (CreateSL2ZFundamentalDomainTriangulation);
    // model/cosets must outlive this object. Builds the table by
    // [C, alg:tau] steps 1-3, builds xi_terms once, and runs the MANDATORY
    // exact certification (step 4, [C, obl:tau]): the constructor calls
    // certify() and THROWS (std::runtime_error / HeckeOverflow) on any
    // failure, so a HeckeTauTable can never exist in an uncertified state.
    HeckeTauTableT(const SimplicialComplex& K, const SL2ZTessellationTables& model,
                  const HeckeCosetTablesT<Elt>& cosets, Elt ell, uint32_t n_sym,
                  TauVertexPolicy policy = TauVertexPolicy::GeometricNearest,
                  TauTableContents contents = TauTableContents::Full);

    // (slot-6 WP-4) The strict-pair tau^s of [C, prop:strictpair]
    // (TauVertexPolicy::BoundaryStrict): taub must be built on the SAME coset
    // tables and model (identity, certified by clause (g)) and must outlive
    // this object. Same mandatory certification as above, plus (g); throws on
    // any failure, including a fill that neither integer tier can solve
    // (HeckeOverflow / std::runtime_error) -- it never falls back mod ell.
    HeckeTauTableT(const SimplicialComplex& K, const SL2ZTessellationTables& model,
                  const HeckeCosetTablesT<Elt>& cosets, Elt ell, uint32_t n_sym,
                  const HeckeBoundaryTauTableT<Elt>& taub);

    // tau(s_i . i(sigma)) for i in [1, num_cosets()], q in {0,1,2},
    // sigma_idx indexing K's q-simplex list. Throws std::out_of_range on a
    // bad index. The chain may be legitimately empty (e.g. an edge whose
    // endpoints were assigned the same K^u vertex).
    const std::vector<ChainTerm>& tau(uint32_t i, uint32_t q, uint32_t sigma_idx) const;

    // Parallel eagerly-reduced Xi-term table ([C, eq:xiexplicit]); same
    // indexing, term k corresponds to tau(...)[k].
    const std::vector<XiTermT<Elt>>& xi_terms(uint32_t i, uint32_t q, uint32_t sigma_idx) const;

    // The deduplicated rho_n pool behind xi_terms(): for every stored term,
    // rho_pool()[xt.rho_g_gamma.index()] IS xt.rho_g_gamma, and the entries are
    // pairwise distinct matrices. Exposed for reporting and for consumers that
    // want to memoise per pool slot rather than per matrix value.
    const std::vector<FlatMatrixT<Elt>>& rho_pool() const { require_pool("rho_pool"); return *rho_pool_; }

    // (hecke-extensions slot 13, WP-4) The U_ell memory lever.  A table
    // CONSTRUCTED with the lever not Off (HeckeOperatorOptions: Auto, the
    // default since the owner decision of 2026-09-28, or On) and ell | N keeps
    // every pool slot as U V (<= min(k, n + 1) outer products; rank 1 over
    // F_ell -- ENGINE_CHECKed per slot) and a 0 x 0 placeholder in rho_pool(),
    // so RhoRefT::get() on it throws.  Certification (f) compares every term
    // against U V of its slot, recomputed from its own exact gamma_k, as
    // before.  With the lever Off the table is byte-for-byte the shipped one.
    bool rho_pool_is_low_rank() const { require_pool("rho_pool_is_low_rank"); return !rho_pool_lr_.empty(); }
    // (slot-14 WP-10) A pool slot as a dense block, on either kind of pool:
    // BORROWED from the dense pool, or, on a factored pool, OWNED and
    // evaluated by rho_g_gamma_of(<the slot's first-visited gamma>) -- the call
    // that fills the dense pool, so the same bytes and scalar flag.  The
    // dense readers outside the operator (the cross-checks, the static
    // pattern hash) read the pool through this.
    CosetRhoT<Elt> rho_pool_block(uint32_t slot) const {
        require_pool("rho_pool_block");
        ENGINE_CHECK_MSG(slot < rho_pool_->size(), "HeckeTauTable: no rho pool slot of that index");
        if (rho_pool_lr_.empty()) return CosetRhoT<Elt>(&(*rho_pool_)[slot]);   // = !rho_pool_is_low_rank(), pool already required
        ENGINE_CHECK_MSG(slot < slot_gamma_.size(), "HeckeTauTable: factored pool without its slot representatives");
        return CosetRhoT<Elt>(rho_g_gamma_of(slot_gamma_[slot]));
    }
    // (slot-14 G0 repairs, audit m2) A pool slot's reduced key (g gamma) mod ell,
    // [a, b, c, d] row-major, on either kind of pool: rho_pool()[slot] (or its
    // factors) IS rho_n of this key.  Lets a consumer factor a DENSE pool slot
    // by the rank-one closed form (lowrank::rank_one_rho_factors) instead of
    // the O(d^2) factoriser.
    const std::array<Elt, 4>& rho_pool_key(uint32_t slot) const {
        require_pool("rho_pool_key");
        ENGINE_CHECK_MSG(slot < slot_key_.size(), "HeckeTauTable: no rho pool slot of that index");
        return slot_key_[slot];
    }
    const lowrank::LowRankT<Elt>& rho_pool_low_rank(uint32_t slot) const {
        require_pool("rho_pool_low_rank");
        ENGINE_CHECK_MSG(slot < rho_pool_lr_.size(), "HeckeTauTable: no factored rho pool slot of that index");
        return rho_pool_lr_[slot];
    }

    // (slot-14 WP-16, audit S-4) VALIDATION-ONLY mutator: corrupts ONE pool slot of a
    // completed table so that re-asserting certify() must return 60 (clause (f)) on the
    // serial and on the parallel path.  Dense pool: swaps the contents of slots 0 and 1
    // (the dense pool's entries are pairwise distinct, so both slots then mismatch the
    // matrices recomputed from their terms' exact gammas); a single-slot dense pool is
    // refused.  Factored pool: U[0] of slot 0 += 1 mod ell (row 0 of U V moves by row 0
    // of V, which is non-zero since V has full row rank t >= 1).  Never called by
    // production code; the table is unusable afterwards.
    void corrupt_rho_pool_for_validation() {
        require_pool("corrupt_rho_pool_for_validation");
        if (rho_pool_is_low_rank()) {
            lowrank::LowRankT<Elt>& f = rho_pool_lr_[0];
            ENGINE_CHECK_MSG(!f.U.empty(), "HeckeTauTable: empty factor in slot 0");
            f.U[0] = static_cast<Elt>((static_cast<uint64_t>(f.U[0]) + 1) % static_cast<uint64_t>(ell_));
            return;
        }
        ENGINE_CHECK_MSG(rho_pool_->size() >= 2, "HeckeTauTable: corrupt_rho_pool_for_validation needs two dense slots");
        std::swap((*rho_pool_)[0], (*rho_pool_)[1]);
    }

    // (S23 G2-F fix F2) VALIDATION-ONLY mutator, on either kind of table: negates the coefficient of the first term of
    // the vertex slot (q = 0, sigma = 0, i = 1), so that re-asserting certify() must return 20 (clause (b), the
    // augmentation epsilon == 1; for ell > 2 also on a mod-ell-solved slot) on a Full and on a ChainsOnly table alike.
    // Never called by production code; the table is unusable afterwards.
    void corrupt_record_for_validation() {
        ENGINE_CHECK_MSG(!records_.empty() && !records_[0].empty() && !records_[0][0].empty() &&
                             !records_[0][0][0].terms.empty(),
                         "HeckeTauTable: corrupt_record_for_validation needs a non-empty vertex slot");
        ChainTerm& term = records_[0][0][0].terms[0];
        term.coeff = -term.coeff;
    }

    // The certified whole-pipeline bound B = max(B_tab, 4 N^2) of
    // [C, thm:backend], with B_tab = max ||gamma_k|| over all stored
    // matrices ([C, lem:bounds](5)); in particular
    // max ||gamma_k|| <= certified_bound_B(). Certified at build time:
    // certification asserts 2 B^2 < 2^63 (the int64 backend tier), or, in the
    // wide tier (slot-24 ELL-W, wide_tier()), 2 N B <= 2^63 - 1.
    uint64_t certified_bound_B() const { return bound_B_; }

    // B_tab alone ([C, lem:bounds](5)), for reporting.
    uint64_t certified_bound_B_tab() const { return bound_B_tab_; }

    // (slot-24 ELL-W) The arithmetic tier the table is certified in ([C, lem:bounds](5), [C, thm:backend]):
    // false = narrow (2 B^2 < 2^63, B <= 2^31 - 1: every table the pre-ELL-W code certified, built by the same
    // instructions); true = the wide tier (B > 2^31 - 1, certified by clause (e) as 2 N B <= 2^63 - 1, the
    // stored-class products and determinants formed in __int128 after an int64 overflow -- hecke::mul_tiered).
    bool wide_tier() const { return wide_tier_; }
    // Whether the wide tier could run at build (HECKE_TAU_WIDE != "0" and not ExactTier::ForceNarrow).
    bool wide_tier_allowed() const { return wide_allowed_; }

    // (hecke-extensions slot 2, WP-8; additive) The coset datum and the S2
    // model this table was built against. The pair homotopy eta of
    // Hecke/HeckePairHomotopy certifies ([C, obl:eta](i)) that its tau and
    // tau_bd are built on the SAME g, Gamma'' = Gamma_0(N) and s_j, and does
    // so by identity of these two objects (HeckeBoundaryTauTable exposes the
    // same pair). Read-only; nothing about the table changes.
    const HeckeCosetTablesT<Elt>& cosets() const { return *cosets_; }
    const SL2ZTessellationTables& model() const { return *model_; }
    uint32_t num_cosets() const { return m_; }
    // (slot-6 WP-4; additive) The model complex K the table was built on and
    // its (ell, n_sym), so a second admissible table can be built beside it.
    const SimplicialComplex& complex() const { return *K_; }
    Elt ell() const { return ell_; }
    uint32_t n_sym() const { return n_sym_; }

    // True iff any triangle fill used the mod-ell fallback solve (the
    // recorded datum of [C, alg:tau](3)); certification of identities
    // involving such chains is exact mod ell, otherwise exact over Z.
    bool used_mod_ell_solve() const { return any_mod_ell_; }

    // (slot-6 WP-4) Record fields of the fill search ([C, obl:eta](iv) /
    // [OD] OP2 vocabulary): the solve regime ("Z" iff no fill used the
    // mod-ell fallback, "mod_ell" otherwise -- always "Z" for BoundaryStrict)
    // and, per integer tier, the number of triangle fills it solved (the
    // widened tier is reachable only under BoundaryStrict). A fill whose
    // right-hand side is 0 needs no solve and is counted in the int64 tier.
    TauVertexPolicy policy() const { return policy_; }
    bool is_strict_pair() const { return policy_ == TauVertexPolicy::BoundaryStrict; }
    // (S23 G2-F) TauTableContents: chains_only() iff the table holds its chain records only (no xi table, no rho pool).
    TauTableContents contents() const { return contents_; }
    bool chains_only() const { return contents_ == TauTableContents::ChainsOnly; }
    // (S23 G2-F) The heap payload the table holds in its chain records, its xi-term table and its rho pool (dense
    // blocks or factors, the slot vectors): vector capacities times element sizes, allocator overhead excluded.
    // Reporting / validation only.
    uint64_t stored_bytes() const;
    const char* solve_regime() const { return any_mod_ell_ ? "mod_ell" : "Z"; }
    uint32_t num_fills_int64() const { return num_z_solves_ - num_z_wide_solves_; }
    uint32_t num_fills_widened() const { return num_z_wide_solves_; }
    uint32_t num_fills_mod_ell() const { return num_mod_ell_solves_; }
    // The boundary table a BoundaryStrict table was built against (nullptr
    // otherwise), and the number of slots step (B) pre-assigned from it.
    const HeckeBoundaryTauTableT<Elt>* strict_boundary_table() const { return taub_; }
    uint32_t num_strict_boundary_slots() const { return num_strict_bd_slots_; }

    // The MANDATORY certification ([C, alg:tau](4) / [C, obl:tau]); 0 iff
    // ALL of the following hold EXACTLY (over Z, or mod ell for identities
    // involving mod-ell-solved chains):
    //   (a) structural: every stored gamma_k has det == 1, every term is in
    //       canonical form (idempotent KuSimplexKey, sign +1), coeff != 0;
    //   (b) augmentation (T3): epsilon(tau(vertex slot)) == 1;
    //   (c) chain map (T2): boundary(tau(x)) == tau(boundary(x)) for every
    //       stored representative x of degree 1 and 2, the right side
    //       evaluated ONLY through the shared (T1)/(P1)/(P4) helper and
    //       canonical keys;
    //   (d) twisted equivariance bookkeeping (T1): for EVERY stored
    //       transporter t in S(sigma, sigma') and every i, the slot
    //       (i, sigma) equals the sign-tracked (T1)-translate of the slot
    //       for s_i * t (this includes the elliptic stabilizer
    //       self-constraints);
    //   (e) bounds ([C, obl:backend]): B_tab matches a full recount,
    //       B = max(B_tab, 4 N^2) satisfies 2 B^2 < 2^63 (int64 tier), else
    //       (slot-24 ELL-W) the wide tier's 2 N B <= 2^63 - 1 (code 52 if it
    //       fails; code 51 if the wide tier may not run: HECKE_TAU_WIDE=0 or
    //       a ForceNarrow build);
    //   (f) xi_terms match a recomputation of rho_n((g gamma_k) mod ell)
    //       from the exact stored gamma_k (invariant I2);
    //   (g) (BoundaryStrict only, slot-6 WP-4) STRICTNESS, [C, prop:strictpair]:
    //       71 the boundary table is built on the same coset tables and model
    //       (identity); 72 no fill used the mod-ell fallback (the table is
    //       over Z); 70 for EVERY boundary slot (i, sigma), sigma in K_bd of
    //       every degree the boundary reports, tau^s(s_i . i(sigma)) equals
    //       j tau_bd(s_i . i(sigma)) EXACTLY over Z as canonical chains --
    //       i.e. tau^s o j = j o tau_bd, which is what makes eta == 0 a pair
    //       homotopy. Clauses (a)-(f) certify tau^s as a datum in its own
    //       right, independently of (g).
    // Called by the constructor, which throws on a nonzero result; exposed
    // so validations can re-assert it on the completed object.
    uint32_t certify() const;

    std::string to_string() const { return to_string_offset(0); }
    std::string to_string_offset(uint32_t offset) const;

private:
    struct ChainRecord {
        std::vector<ChainTerm> terms;
        bool mod_ell = false; // built (directly or by derivation) from a
                              // mod-ell triangle solve
        bool stored = false;  // construction-order bookkeeping
        bool derived = false; // filled through the (T1) helper (reporting)
    };

    // Signed canonical accumulation map: {q, sigma_hat, a, b, c, d} -> coeff.
    using CanonMap = std::map<std::array<int64_t, 6>, int64_t>;

    const SimplicialComplex* K_;
    const SL2ZTessellationTables* model_;
    const HeckeCosetTablesT<Elt>* cosets_;
    Elt ell_;
    uint32_t n_sym_;
    TauVertexPolicy policy_;
    TauTableContents contents_ = TauTableContents::Full;   // (S23 G2-F)
    // (S23 G2-F) the field-route accessors refuse a ChainsOnly table (std::logic_error, named text)
    void require_pool(const char* what) const {
        if (contents_ == TauTableContents::Full) return;
        const std::string msg = std::string("HeckeTauTable::") + what +
                                ": this table was built TauTableContents::ChainsOnly (the module route's table: chain "
                                "records only, no xi-term table, no rho pool); build it Full for the field route";
        ENGINE_CHECK_MSG(false, msg.c_str());
    }
    // (slot-6 WP-4) The boundary table of a BoundaryStrict table (else null).
    const HeckeBoundaryTauTableT<Elt>* taub_ = nullptr;
    // iota_K on the vertices (from the shipped coordinates) and its unique
    // fixed vertex; consumed by the tie-break of choose_vertex() only.
    std::array<uint32_t, 7> iota_perm_{};
    uint32_t iota_fixed_vertex_ = 7;
    uint64_t N_;
    uint32_t m_;

    // Simplex tables of the model (copied; validated 7/11/5 shape).
    std::vector<std::vector<std::vector<uint32_t>>> simplices_;
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> edge_index_;

    // records_[q][sigma][i-1]; xi_[q][sigma][i-1].
    std::vector<std::vector<std::vector<ChainRecord>>> records_;
    std::vector<std::vector<std::vector<std::vector<XiTermT<Elt>>>>> xi_;

    // Deduplicated rho_n((g gamma) mod ell) pool referenced by every XiTermT
    // (7.3). Filled once by build_xi_terms() in term-visit order and immutable
    // afterwards; shared with the handles so it outlives the table if needed.
    std::shared_ptr<std::vector<FlatMatrixT<Elt>>> rho_pool_ =
        std::make_shared<std::vector<FlatMatrixT<Elt>>>();
    std::vector<lowrank::LowRankT<Elt>> rho_pool_lr_;   // (slot-13 WP-4) non-empty iff built under the lever
    // (slot-14 WP-10) per slot, the first-visited gamma of its reduced key
    // (the one build_xi_terms evaluates); kept for a factored pool only.
    std::vector<Mat22> slot_gamma_;
    // (slot-14 G0 repairs, audit m2) per slot, its reduced key; every pool.
    std::vector<std::array<Elt, 4>> slot_key_;

    bool any_mod_ell_ = false;
    uint64_t bound_B_tab_ = 0;
    uint64_t bound_B_ = 0;
    bool wide_allowed_ = true;   // (slot-24 ELL-W) recorded at build
    bool wide_tier_ = false;     // (slot-24 ELL-W) B > 2^31 - 1
    uint32_t num_z_solves_ = 0;
    uint32_t num_mod_ell_solves_ = 0;
    uint32_t num_z_wide_solves_ = 0;   // of num_z_solves_: fills the widened tier solved (BoundaryStrict)
    uint32_t num_strict_bd_slots_ = 0; // slots step (B) pre-assigned (BoundaryStrict)

    // --- shared (T1) helper (THE single implementation; construction AND
    // certification route through it) ---------------------------------------
    // Value of tau(gamma . i(sigma)) for ARBITRARY gamma in Gamma, from the
    // stored slot of its Gamma''-class: writes gamma = h s_i via (P1)
    // (i = class_index_of(gamma), h = gamma s_i^{-1} in Gamma_0(N), verified),
    // conjugates h to g^{-1} h g = [[h.a, N h.b],[h.c / N, h.d]] (exact
    // integral; divisibility verified) and returns the canonicalized
    // translate of the stored chain. mod_ell_out reports the source flag.
    std::vector<ChainTerm> eval_tau_on_translate(const Mat22& gamma, uint32_t q,
                                                 uint32_t sigma, bool& mod_ell_out) const;

    // --- construction -------------------------------------------------------
    // The body shared by both constructors (the pre-WP-4 constructor body,
    // verbatim, with step (B) inserted before step 1 for BoundaryStrict).
    void construct_and_certify();
    // [C, prop:strictpair] step (B): pre-assign every boundary slot.
    void build_strict_boundary();
    void build_vertices();
    void build_edges();
    void build_triangles();
    void build_xi_terms();

    // Fills slot (i, q, sigma) through the (T1) helper from an
    // already-stored Gamma''-related slot, if one exists; returns success.
    bool try_derive(uint32_t i, uint32_t q, uint32_t sigma);

    void construct_vertex(uint32_t i, uint32_t v);   // [C, alg:tau](1)
    void construct_edge(uint32_t i, uint32_t e);     // [C, alg:tau](2)
    void construct_triangle(uint32_t i, uint32_t f); // [C, alg:tau](3)

    // --- [C, alg:tau](1), both exact tiers ---------------------------------
    // The three step-1 helpers are templated on the QuadPoint coordinate type
    // Pt (QuadPoint = int64, QuadPointW = __int128; QuadPoint.h). The int64
    // instantiation is the pre-WP-1 code verbatim -- same operations, same
    // FP64-filtered predicates -- and the __int128 one is the escalation tier
    // of [C, thm:backend], reached only after a HeckeOverflow (or under
    // hecke::ScopedExactTier). Both produce the SAME gamma_red and the SAME
    // vertex index; only the intermediate coordinates widen.

    // Fundamental-domain reduction of [C, alg:tau](1): returns gamma_red
    // with gamma_red . z in the closed fundamental domain (predicates
    // cmp_abs_re_half / cmp_norm_one; block T-shifts; overflow-checked).
    template <typename Pt>
    Mat22 reduce_to_fundamental_domain(Pt& z) const;

    // Vertex choice per policy_ (see TauVertexPolicy).
    template <typename Pt>
    uint32_t choose_vertex(const Pt& z_red) const;
    // Path-template family per policy_ (Canonical unless legacy).
    SL2ZTessellationTables::PathTemplateFamily template_family() const;

    // sign(dist^2(z, vertex a) - dist^2(z, vertex b)) exactly, on cleared
    // denominators; mixed D in {1,3} routed through sign_lin_sqrt3.
    template <typename Pt>
    int compare_vertex_distances(const Pt& z, uint32_t a, uint32_t b) const;

    // Step 1 on one tier: z = (g^{-1} s_i) . pos(v), reduced; writes only its
    // two out-parameters, so the Auto-mode retry of hecke::escalate_on_overflow
    // cannot observe a partial write.
    template <typename Pt>
    void step1_vertex_on_tier(const Mat22& M, uint32_t v,
                              Mat22& gred_out, uint32_t& w_out) const;

    // --- canonical chain bookkeeping ---------------------------------------
    void accumulate_canonical(CanonMap& acc, const Mat22& gamma, uint32_t q,
                              uint32_t sigma, int64_t coeff) const;
    std::vector<ChainTerm> canon_map_to_terms(const CanonMap& acc, uint32_t q,
                                              bool mod_ell) const;
    CanonMap terms_to_canon_map(const std::vector<ChainTerm>& terms, uint32_t q) const;
    // boundary of a stored q-chain (q >= 1) as a canonical (q-1)-map.
    CanonMap boundary_of_terms(const std::vector<ChainTerm>& terms, uint32_t q) const;
    bool canon_maps_equal(const CanonMap& x, const CanonMap& y, bool mod_ell) const;

    // face_j of K-simplex sigma (q = 1: vertex id; q = 2: edge index).
    uint32_t face_simplex_index(uint32_t q, uint32_t sigma, uint32_t j) const;

    const ChainRecord& record(uint32_t i, uint32_t q, uint32_t sigma) const;
    ChainRecord& record_mut(uint32_t i, uint32_t q, uint32_t sigma);
    void check_indices(uint32_t i, uint32_t q, uint32_t sigma) const;

    FlatMatrixT<Elt> rho_g_gamma_of(const Mat22& gamma) const; // rho_n((g gamma) mod ell)

    // triangle-system solvers ([C, alg:tau](3)); see HeckeTauTable.cpp.
    // used_wide_out (BoundaryStrict only) reports that the widened integer
    // tier solved the fill; the shipped policies never set it.
    bool solve_triangle_system(const CanonMap& c_map, std::vector<ChainTerm>& x_out,
                               bool& used_mod_ell_out, bool& used_wide_out) const;
    // Clause (g) of certify() (BoundaryStrict); 0 or its code.
    uint32_t certify_strict() const;
};

using HeckeTauTable = HeckeTauTableT<uint32_t>;
using HeckeTauTable64 = HeckeTauTableT<uint64_t>;

} // namespace hecke

#endif // HECKE_TAU_TABLE_H
