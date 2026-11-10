// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_LEVEL_N_H
#define HECKE_LEVEL_N_H
// ===========================================================================
// (hecke-extensions slot 14, WP-12) LEVEL N ON THE FIXED MODEL, FIRST BUILD.
// Binding text: [C] v1.14.0 sec:levelN -- def:cosetrepsN / lem:cosetrepsN (O3:
// the canonical table of Gamma_0(N)\Gamma for EVERY N, by the Chinese remainder
// theorem over the prime-power factors; Definition cosetreps verbatim at prime
// powers; the representatives of Gamma_0(N')\Gamma_0(N) read off the level-N'
// table), def:levelNmodel (the level-N model L_N for every N and every ring:
// ell | N, ell !| N, and the R3 setting F_p, p !| N), and sec:levelshapiro's
// prop:levelcorrespondence (eq:levelcorr) and prop:levelorbit (the orbit-wise
// fixed spaces and H^1 as a cokernel).
//
// WHAT IS BUILT.  HeckeLevelEll.h's three general-N plug points, extended:
//
//  * Gamma0LevelN (plug point 1) -- a CosetLevel for Gamma_0(N), N composite or
//    not, implementing def:cosetrepsN (1)-(5) exactly: the component index j_i
//    at each Q_i = q_i^{nu_i}, the mixed radix (primes ascending), the canonical
//    pair by CRT, the representative by the first applicable case (i) / (ii) /
//    (iii), the class map and the right action (digitwise).  A TABLE, not a
//    search: case (iii)'s least t < c_j is a bounded loop at construction.  No
//    new tau-table: every operator reads the level-1 table of [Gamma g Gamma].
//  * Gamma0SubTransversalN (plug point 2) -- def:cosetrepsN (6): the
//    representatives of Gamma_0(N')\Gamma_0(N) (N | N') are the level-N' table's
//    w_j with N | c_j, in increasing j.  For N' = N p they are the r_t of
//    prop:levelcorrespondence (4)(d): T_p^{(N)} (p !| N, p + 1 of them) and
//    U_p^{(N)} (p | N, p of them).
//  * InducedRelationsLevel -- the relation list of L_A for ANY CosetLevel A over
//    ANY ModCtx (field or Z/ell^k): the rows of GenerateSL2ZRelationsWithGroupData
//    with each matrix replaced by the monomial M_A(g_r) of eq:MH.  At a prime-power
//    level it reproduces MakeInducedLocalSystemGamma0 (field) and
//    InducedRelationsRing (ring) entrywise (the plug-point regression).
//  * MonomialAction / OrbitWiseFixedSpace -- prop:levelorbit (1): M_A(gamma) kept
//    as a permutation of P^1(Z/N) with m d x d blocks (never a D x D matrix), and
//    the fixed space of a finite cyclic <gamma> (gamma = S or ST) computed ORBIT
//    BY ORBIT: a free orbit contributes V with no solve (f_{j.gamma^i} =
//    rho(h_j(gamma^i))^{-1} f_j), a fixed point the image of the averaging
//    idempotent of rho(w_j gamma w_j^{-1}) on V.  DenseFixedSpace is the
//    comparison route ((LN2)): the image of the averaging idempotent of the dense
//    M_A(gamma).  Needs the order of gamma in PSL_2 (2, 3) invertible in the ring.
//  * CokernelDims -- prop:levelorbit (2) over a field: dim Rel_N = dim(M^S + M^ST),
//    h^1 = D - dim Rel_N, h^0 = dim M^S + dim M^ST - dim Rel_N.
//
// Reading operators on H^1 is HeckeLevelEll.h's HFrame / InducedOnH (the dense
// route (O1) of rmk:levelscaling -- cubic in m_N d; adequate at the dense-orbit
// route's reach, m_N d <= a few hundred).  NOT built here (slot 15): the
// Heilbronn--Merel action H_{N,p} of prop:heilbronn (O2), the sparse cokernel
// elimination at large N, degeneracy maps at composite N.
//
// Nothing here is reached from a shipped record, report, campaign job id or
// golden text: pipeline-free and value-free with respect to every shipped object.
// ===========================================================================

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "HeckeLevelEll.h"

namespace hecke {
namespace level {

// One prime-power factor Q = q^nu of N; m = Q + Q/q = |P^1(Z/Q)|.
struct PrimePowerFactor {
    uint64_t q = 0;
    uint32_t nu = 0;
    uint64_t Q = 0;
    uint32_t m = 0;
};

// Prime factorisation, primes ascending (trial division; N < 2^31).
std::vector<PrimePowerFactor> FactorLevel(uint64_t N);

// def:cosetrepsN: the canonical table of Gamma_0(N)\Gamma for every N >= 1.
class Gamma0LevelN final : public CosetLevel {
public:
    // N in [1, 2^31); the table (m_N canonical pairs and representatives) is built
    // eagerly and self-checked (every w_j in Gamma with bottom row of class j, w_1 = I).
    // The 2^31 cap bounds the table's own arithmetic (CRT, case (iii) in u128). The exact
    // reach downstream is smaller: Mat22 is int64 with overflow-checked products, and
    // cocycle / MonomialOf form triple products with entries O(N) (case (iii): O(tN)),
    // fine to N ~ 2^20; beyond that a hecke::HeckeOverflow is the (loud) failure.
    // (slot-16 WP-13 finding F-1) That holds for prime powers only.  At composite N with case-(iii) classes the
    // representatives themselves have entries up to t N ~ N^2, so int64 products of two of them overflow
    // from N ~ 5.5 * 10^4 (N^4 ~ 2^63): N = 30030 runs, N = 60060 threw HeckeOverflow before the sparse
    // presentation was built (its S / ST cocycles; the stored table itself is O(N^2) and fits int64).
    // (slot-17 WP-5, F-1 repaired) Every coset product of this file and of HeckeLevelNSparse now goes through
    // SubgroupProductForRho below: int64 first, __int128 on a HeckeOverflow (attempt-then-escalate), so the
    // reach is the table's own (three entries <= N^2: < 2^127 for every N < 2^31) and every input on which int64
    // completes takes the old path byte for byte.  HeilbronnOperator (an __int128 fallback, slot 16) and
    // DegeneracyOperator (__int128 throughout) were already wide.  cocycle() itself stays int64 (its callers
    // catch the HeckeOverflow); cocycle_wide() is its __int128 twin.
    explicit Gamma0LevelN(uint64_t N);

    uint64_t N() const override { return N_; }
    uint32_t m() const override { return m_; }
    Mat22 rep(uint32_t j) const override;                    // w_j, 1-based, w_1 = I
    uint32_t class_of(const Mat22& gamma) const override;    // cls_N(gamma), gamma in Gamma
    bool contains(const Mat22& gamma) const override;        // gamma in Gamma_0(N)
    bool contains_wide(const Mat22W& gamma) const override;  // the same, exactly, in __int128 (slot-17 WP-5)
    std::string name() const override { return "Gamma_0(" + std::to_string(N_) + ")"; }

    const std::vector<PrimePowerFactor>& factors() const { return fac_; }
    // (1)-(2): the index j(c, d) of a pair primitive modulo N (ENGINE_CHECK otherwise).
    uint32_t index_of_pair(int64_t c, int64_t d) const;
    bool primitive(int64_t c, int64_t d) const;              // gcd(c, d, N) = 1, i.e. modulo every q_i
    // The digits j_i of j (1-based each), primes ascending.
    std::vector<uint32_t> digits(uint32_t j) const;
    // (3): the canonical pair (c_j, d_j) in {0..N-1}^2; (0, 1) at N = 1.
    std::pair<uint64_t, uint64_t> canonical_pair(uint32_t j) const;
    // (4): which case gave w_j: 1 = (i), 2 = (ii), 3 = (iii) (a mixed class).
    uint32_t rep_case(uint32_t j) const;
    // (5): j . delta for integral delta with (c_j, d_j) delta primitive mod N.
    bool act_defined(uint32_t j, const Mat22& delta) const;
    uint32_t act(uint32_t j, const Mat22& delta) const;
    // (5): h_j(delta) = w_j delta w_{j.delta}^{-1} (integral; in Gamma_0(N) for delta in Gamma).
    Mat22 cocycle(uint32_t j, const Mat22& delta) const;
    // (slot-17 WP-5) h_j(delta) formed in __int128 (exact; HeckeOverflow only beyond 2^127).
    Mat22W cocycle_wide(uint32_t j, const Mat22& delta) const;
    // The level-Q_i action on one digit (for the digitwise check of lem:cosetrepsN (4)).
    uint32_t digit_act(size_t i, uint32_t ji, const Mat22& delta) const;

private:
    uint64_t N_ = 1;
    uint32_t m_ = 1;
    std::vector<PrimePowerFactor> fac_;
    std::vector<std::pair<uint64_t, uint64_t>> pair_;        // 1-based
    std::vector<Mat22> w_, w_inv_;                           // 1-based
    std::vector<uint8_t> case_;                              // 1-based
    uint32_t component_index(size_t i, int64_t c, int64_t d) const;
};

// def:cosetrepsN (6): representatives of Gamma_0(N')\Gamma_0(N), N | N': the level-N'
// table's w_j whose canonical pair has N | c_j, in increasing j.  Starts with I.
std::vector<Mat22> Gamma0SubTransversalN(const Gamma0LevelN& level_N, const Gamma0LevelN& level_Nprime);

// ---- (slot-17 WP-5, F-1) the width of the coset products ----------------------------------
// The products h = X Y Z (h required in the subgroup A) behind the blocks rho_n(h) of the level-N model:
// MonomialOf, OrbitWiseFixedSpace, OrbitWiseFixedGenerators, LevelChangeMap.  ATTEMPT-THEN-ESCALATE
// (the ExactTier::Auto policy of ExactInt.h): the int64 Mat22 product X * Y * Z exactly as before; only on a
// HeckeOverflow the same product in Mat22W, A.contains_wide(h) checked exactly, and the matrix handed on is
// the residue matrix of h mod `mod` -- RhoCache::block_of keys on exactly g.reduced_mod(ctx.p), so the block
// is the one an unbounded int64 would have produced.  `invert`: return h^{-1} (adj h, det 1) instead of h.
// The ENGINE_CHECK on membership (message `what`) is the one each call site made before.
enum class LevelNWidth : uint8_t {
    kAuto = 0,          // int64, __int128 on a HeckeOverflow (the shipped default)
    kForceWide = 1,     // VALIDATION ONLY: every product in __int128 (the bit-identity arm compares it with kAuto)
    kForceNarrow = 2,   // VALIDATION ONLY: never escalate -- the negative control reproduces the pre-WP-5 refusal
};
// Thread-local; no shipped caller sets it.  It cannot change a value (kForceNarrow can only add a throw).
class ScopedLevelNWidth {
public:
    explicit ScopedLevelNWidth(LevelNWidth w);
    ~ScopedLevelNWidth();
    ScopedLevelNWidth(const ScopedLevelNWidth&) = delete;
    ScopedLevelNWidth& operator=(const ScopedLevelNWidth&) = delete;
    static LevelNWidth current();
private:
    LevelNWidth prev_;
};
// Products formed in __int128 on this thread (escalations under kAuto, every product under kForceWide).
uint64_t LevelNWideProducts();
void ResetLevelNWideProducts();

Mat22 SubgroupProductForRho(const CosetLevel& A, const Mat22& X, const Mat22& Y, const Mat22& Z, uint32_t mod, bool invert,
                            const char* what);

// ---- the level-A model over any ring --------------------------------------------
// The relation list of L_A (def:levelNmodel, eq:MH) for any CosetLevel A over any
// ModCtx: the (q_dim, s_lhs, s_rhs, tag) rows of GenerateSL2ZRelationsWithGroupData,
// each matrix M_A(g_r) = sum_j E_{j, j.g_r} (x) rho_n(h_j(g_r)) (dense only because
// SimplexRelation::M is a FlatMatrix -- the structural pin of InducedLocalSystem.h).
std::vector<LocalSystemSimplicialComplex::SimplexRelation> InducedRelationsLevel(
    const SimplicialComplex& K, const CosetLevel& A, uint32_t n_sym, const engine::ModCtx<uint32_t>& ctx);

// ---- the monomial action and the orbit-wise fixed spaces (prop:levelorbit) ---------
// (gamma . f)_j = rho(h_j(gamma)) f_{j.gamma}: one block per block row.
struct MonomialAction {
    uint32_t m = 0, d = 0;
    std::vector<uint32_t> perm;          // 1-based: perm[j] = j . gamma
    std::vector<uint32_t> blk;           // 1-based: index into blocks
    std::vector<Block> blocks;           // rho_n(h_j(gamma)) over the ring (RhoCache)
    // y = M x, x and y of length m d.
    std::vector<uint32_t> apply(const engine::ModCtx<uint32_t>& ctx, const std::vector<uint32_t>& x) const;
    // The dense D x D matrix (row-major) -- only for the comparison route.
    std::vector<uint32_t> dense() const;
};
MonomialAction MonomialOf(const CosetLevel& A, const Mat22& gamma, uint32_t n_sym, const engine::ModCtx<uint32_t>& ctx);

struct FixedSpace {
    uint32_t D = 0;
    std::vector<std::vector<uint32_t>> span;   // spanning vectors of length D (a basis over a field)
    uint32_t orbits = 0, free_orbits = 0, fixed_points = 0;
    uint32_t solves = 0;                       // d x d idempotent images taken (fixed points only)
};
// prop:levelorbit (1): M_A^{<gamma>} orbit by orbit; gamma of order `order` in PSL_2(Z)
// (S: 2, ST: 3), n even (so -I acts trivially), order a unit in the ring.
FixedSpace OrbitWiseFixedSpace(const CosetLevel& A, const Mat22& gamma, uint32_t order, uint32_t n_sym,
                               const engine::ModCtx<uint32_t>& ctx);
// The comparison route: the columns of the dense averaging idempotent (1/o) sum_i M(gamma)^i.
FixedSpace DenseFixedSpace(const CosetLevel& A, const Mat22& gamma, uint32_t order, uint32_t n_sym,
                           const engine::ModCtx<uint32_t>& ctx);

// Rank of a set of vectors over a prime field F_p (p = ctx.p, ctx a field).
uint32_t FieldRank(const std::vector<std::vector<uint32_t>>& vecs, uint32_t p);

// prop:levelorbit (2), over a field with 6 a unit, n even: from the two fixed spaces.
struct CokernelDims {
    uint32_t D = 0, dim_S = 0, dim_ST = 0, dim_rel = 0, h0 = 0, h1 = 0;
};
CokernelDims CokernelFromFixedSpaces(const FixedSpace& fS, const FixedSpace& fST, uint32_t p);

// ---- counts ----------------------------------------------------------------------------
// m_N = N prod_{q | N} (1 + 1/q); nu_2(N), nu_3(N) (rmk:levelcost (3)); c(N) the cusps.
uint64_t IndexGamma0(uint64_t N);
uint32_t Nu2(uint64_t N);
uint32_t Nu3(uint64_t N);
uint64_t CuspCount(uint64_t N);
// dim S_k(Gamma_0(N)) for even k >= 2 (the classical dimension formula, Shimura Ch. 2).
int64_t DimCuspForms(uint64_t N, uint32_t k);

} // namespace level
} // namespace hecke

#endif // HECKE_LEVEL_N_H
