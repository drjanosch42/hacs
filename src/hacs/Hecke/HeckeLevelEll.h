// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_LEVEL_ELL_H
#define HECKE_LEVEL_ELL_H
// ===========================================================================
// (hecke-extensions slot 13, WP-6) LEVEL STRUCTURE BY SHAPIRO ON THE FIXED MODEL.
// Binding text: [C] v1.13.0-pre.2 sec:levelshapiro -- def:levelellmodel (the
// level-N model is the induced local system L_{Gamma_0(N)} of def:inducedls on
// the fixed (K, R)), prop:levelcorrespondence (every level-N correspondence
// [A g B] is ONE block formula, eq:levelcorr, on the induced fibre that reads
// only the LEVEL-1 tau-table of [Gamma g Gamma]), prop:levelellU (eq:Ulevell,
// eq:Ulevellmodel: U_ell^{(ell)}, dg_ell, dg_1 and the trace omega at N = ell),
// prop:oldspace, prop:levelorbit, rmk:levelcost, rmk:levelscaling; and
// sec:stabilisation (lem:levelelltorsion, prop:stabilisation, (Hi10)).
//
// WHAT IS BUILT.  N = ell only (the owner's direction, 2026-09-26: general N is
// future work), but through an interface written for general N:
//
//  * CosetLevel -- a finite-index A <= Gamma as its right cosets A\Gamma: the
//    index m, a FIXED representative table w_1 = I, w_2, ..., w_m (a binding
//    datum, rmk:indeplevel) and the class map gamma |-> a with A gamma = A w_a,
//    plus membership.  Two implementations: FullLevel (A = Gamma, N = 1) and
//    Gamma0Level (A = Gamma_0(N) read from a shipped HeckeCosetTables at
//    N = p^nu -- def:cosetreps pins prime powers only).
//    >>> GENERAL-N PLUG POINT 1: a composite-N level needs a pinned table of
//    Gamma_0(N)\Gamma.  [C] v1.14.0 pins it for every N by def:cosetrepsN
//    (O3 of rmk:levelscaling, closed at v1.14.0); it is built by slot-14
//    WP-12 as Gamma0LevelN in HeckeLevelN.h (at N = p^nu it IS this class's
//    shipped table, lem:cosetrepsN (3)).
//  * CorrespondenceSpec -- (A, B, g, the level-1 datum of [Gamma g Gamma], and
//    representatives r_t of B'' \ B, B'' = B cap g A g^{-1}).  The level-1
//    datum is (HeckeCosetTables of g, HeckeTauTable) -- the SHIPPED tables, no
//    new coset table and no new tau-table (prop:levelcorrespondence (1)) -- or
//    the identity datum (g = I, tau = id) for the fibre maps of (4)(b).
//    >>> GENERAL-N PLUG POINT 2: the transversal r_t of B'' \ B.  Built here
//    by Gamma0SubTransversal(N, p) for B'' = Gamma_0(N p) in B = Gamma_0(N),
//    p prime (U_p^{(N)} for p | N, T_p^{(N)} for p !| N), and verified by
//    VerifyTransversal (membership + pairwise distinctness + the count).
//    >>> GENERAL-N PLUG POINT 3: a composite degeneracy d = d_1 d_2 composes
//    two prime-power correspondences through Gamma_0(N d_1)
//    (prop:levelcorrespondence (4)(c)) -- ComposeCochainMaps below; no table
//    for composite d.
//  * BuildLevelCorrespondence -- eq:levelcorr for ANY such spec, as a block-
//    sparse cochain map on flat cochains of the model (block (sigma, i) <-
//    (sigma', a . gamma), coefficient c rho_n(v_i delta_j gamma w_{a.gamma}^{-1}),
//    every product EXACT (Mat22) and integral, never an inverted det-p matrix).
//  * BuildUellLevelEllFused -- eq:Ulevellmodel (the varpi block row), the
//    SECOND model route of U_ell^{(ell)}, compared entrywise with the first.
//  * HFrame / InducedOnH -- reading a cochain map on cohomology in the stored
//    bases (field: the P_q projector; Z/ell^k: H_class_coordinates), with the
//    runtime assertions (every output satisfies the target's relations and is
//    a cocycle) -- the alg:assembly pattern for a map between two local systems.
//
// Coefficients: any engine::ModCtx<uint32_t> -- F_p (p may equal ell or not) or
// Z/ell^k, ell >= 5 (the cochain-level statements hold over every ring, the
// cohomology readings need 6 invertible: prop:lhcompliant(4)).
//
// Cost (rmk:levelcost (5)): per simplex and per (i, t) one level-1 tau-chain and
// one d x d block per term; no D_N x D_N matrix is formed.  Over F_ell the
// U_ell^{(ell)} blocks rho_n(g gamma s_j^{-1}) are singular mod ell (det ell);
// a block of rank <= 1 is stored factored (u v^T) and applied in 2 d per column.
// Reading the operator on H^1 is the dense route (1) (the stored basis and the
// projector), cubic in m_N (k - 1) -- (O1) of rmk:levelscaling.
//
// Nothing here is reached from a shipped record, report, campaign job id or
// golden text: pipeline-free and value-free with respect to every shipped object.
// ===========================================================================

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "ExactInt.h"
#include "HeckeCosetTables.h"
#include "HeckeTauTable.h"
#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"
#include "../../ModArithmetic/ModCtx.h"

namespace hecke {
namespace level {

// ---- level structures -------------------------------------------------------
class CosetLevel {
public:
    virtual ~CosetLevel() = default;
    virtual uint64_t N() const = 0;                          // the level (1 for Gamma)
    virtual uint32_t m() const = 0;                          // [Gamma : A]
    virtual Mat22 rep(uint32_t a) const = 0;                 // w_a, 1-based, w_1 = I
    virtual uint32_t class_of(const Mat22& gamma) const = 0; // A gamma = A w_a
    virtual bool contains(const Mat22& gamma) const = 0;     // gamma in A (gamma in Gamma)
    // (slot-17 WP-5, F-1) the same membership test for gamma given in __int128 (a coset product past the int64
    // reach).  Default: narrow to Mat22 when every entry fits int64 and ask contains(); otherwise a loud
    // HeckeOverflow (no wide test for this level).  Gamma0LevelN and FullLevel override it exactly.
    virtual bool contains_wide(const Mat22W& gamma) const;
    virtual std::string name() const = 0;
};

class FullLevel final : public CosetLevel {
public:
    uint64_t N() const override { return 1; }
    uint32_t m() const override { return 1; }
    Mat22 rep(uint32_t a) const override;
    uint32_t class_of(const Mat22& gamma) const override;
    bool contains(const Mat22& gamma) const override;
    bool contains_wide(const Mat22W& gamma) const override;
    std::string name() const override { return "Gamma"; }
};

// Gamma_0(N), N = cosets.N() = p^nu, representatives and class map = the shipped
// table's (def:cosetreps; the SAME table the induced local system is built on).
class Gamma0Level final : public CosetLevel {
public:
    explicit Gamma0Level(const HeckeCosetTables& cosets) : c_(&cosets) {}
    uint64_t N() const override { return c_->N(); }
    uint32_t m() const override { return c_->num_cosets(); }
    Mat22 rep(uint32_t a) const override { return c_->s(a); }
    uint32_t class_of(const Mat22& gamma) const override { return c_->class_index_of(gamma); }
    bool contains(const Mat22& gamma) const override;
    std::string name() const override { return "Gamma_0(" + std::to_string(c_->N()) + ")"; }
    const HeckeCosetTables& cosets() const { return *c_; }
private:
    const HeckeCosetTables* c_;
};

// g^{-1} x g for g in Delta, x in Gamma, when integral (adj(g) x g / det g, exact);
// returns false if not integral.
bool ConjugateByInverse(const Mat22& g, const Mat22& x, Mat22& out);

// r_t, t = 0 .. index-1: representatives of Gamma_0(N p) \ Gamma_0(N), p prime:
// [[1, 0], [N t, 1]] (t < p), and for p !| N also [[x, y], [N, p]] (x p - y N = 1).
std::vector<Mat22> Gamma0SubTransversal(uint64_t N, uint64_t p);

// ---- correspondences ----------------------------------------------------------
struct CorrespondenceSpec {
    const CosetLevel* A = nullptr;          // source level
    const CosetLevel* B = nullptr;          // target level
    // The level-1 datum of [Gamma g Gamma]; both null: the identity datum (g = I).
    const HeckeCosetTables* g_cosets = nullptr;
    const HeckeTauTable* tau = nullptr;
    std::vector<Mat22> r;                   // B'' \ B, r[0] need not be I
    std::string label;
    Mat22 g() const;                        // diag(1, g_cosets->N()) or I
};

// B'' = B cap g A g^{-1} membership (x in Gamma).
bool InDoublePrime(const CorrespondenceSpec& s, const Mat22& x);
// Every r_t in B, pairwise distinct B''-cosets, and |r| == expected_index; throws
// (ENGINE_CHECK) otherwise.
void VerifyTransversal(const CorrespondenceSpec& s, uint32_t expected_index);

// One d x d block, dense row-major, with an optional rank <= 1 factorisation.
struct Block {
    std::vector<uint32_t> dense;            // d * d
    bool rank1 = false;                     // dense == u v^T (u, v of length d)
    std::vector<uint32_t> u, v;
};

struct BlockTerm {
    uint32_t dst_sigma, dst_blk, src_sigma, src_blk;
    uint32_t coeff;                         // in [0, p)
    uint32_t block;                         // index into BlockCochainMap::blocks
};

// A cochain map C^q(K; V^{m_src}) -> C^q(K; V^{m_dst}) on flat cochains, as a
// sum of block terms.  Flat layout: row sigma * (m d) + (blk - 1) * d + r.
class BlockCochainMap {
public:
    engine::ModCtx<uint32_t> ctx;
    uint32_t q = 0, N_q = 0, d = 0, m_src = 1, m_dst = 1;
    std::vector<Block> blocks;
    std::vector<BlockTerm> terms;
    uint32_t src_dim() const { return N_q * m_src * d; }
    uint32_t dst_dim() const { return N_q * m_dst * d; }
    // X: src_dim x ncols row-major -> dst_dim x ncols row-major.
    std::vector<uint32_t> apply(const std::vector<uint32_t>& X, uint32_t ncols) const;
    // The aggregated block map: (dst_sigma, dst_blk, src_sigma, src_blk) -> the
    // d x d sum.  Equal aggregates <=> equal cochain maps (entrywise).
    std::map<std::array<uint32_t, 4>, std::vector<uint32_t>> aggregate() const;
    size_t num_terms() const { return terms.size(); }
};

// rho_n(g) over ctx (F_p: ComputeSymmetricPowerMatrixFM of g mod p; Z/ell^k:
// RhoOfExactRing), memoised on the reduced key.
class RhoCache {
public:
    RhoCache(const engine::ModCtx<uint32_t>& ctx, uint32_t n_sym) : ctx_(ctx), n_(n_sym) {}
    uint32_t block_of(const Mat22& g, std::vector<Block>& store);   // index into store
private:
    engine::ModCtx<uint32_t> ctx_;
    uint32_t n_;
    std::map<std::array<uint32_t, 4>, uint32_t> memo_;
};

// eq:levelcorr for the spec, degree q, on the model K (the tau table's).
BlockCochainMap BuildLevelCorrespondence(const engine::ModCtx<uint32_t>& ctx, const CorrespondenceSpec& spec,
                                         const SimplicialComplex& K, uint32_t n_sym, uint32_t q);

// eq:Ulevellmodel: U_ell^{(ell)} by the varpi block row, on the level-ell table
// `lvl` (N = ell) and the level-1 U_ell tau-table (built on the SAME coset table).
BlockCochainMap BuildUellLevelEllFused(const engine::ModCtx<uint32_t>& ctx, const Gamma0Level& lvl,
                                       const HeckeTauTable& tau, const SimplicialComplex& K, uint32_t n_sym,
                                       uint32_t q);


// ---- the level-N model over a ring ---------------------------------------------
// L_{Gamma_0(N)} over Z/ell^k (composite ctx): the rows of
// GenerateSL2ZRelationsWithGroupData, each matrix replaced by M_H(g_r) mod ell^k
// (eq:MH, blocks rho_n(h_i(g_r)) of the shipped table's exact cocycles; -I:
// (-1)^n Id).  The field version is the shipped MakeInducedLocalSystemGamma0.
std::vector<LocalSystemSimplicialComplex::SimplexRelation> InducedRelationsRing(
    const SimplicialComplex& K, const HeckeCosetTables& cosets, uint32_t n_sym, const engine::ModCtx<uint32_t>& ctx);

// ---- reading on cohomology ----------------------------------------------------------
// Row-major matrix over ctx.p with per-row exponents (field: all 1, modulus p;
// Z/ell^k: row i meaningful mod ell^{exp_i}).
struct HMat {
    uint32_t rows = 0, cols = 0;
    uint64_t modulus = 0;                   // ctx.p
    uint64_t ell = 0;                       // the prime (field: p)
    bool field = true;
    std::vector<uint32_t> row_exp;          // size rows
    std::vector<uint64_t> a;                // rows * cols
    uint64_t at(uint32_t i, uint32_t j) const { return a[static_cast<size_t>(i) * cols + j]; }
};

class HFrame {
public:
    HFrame(const EquivariantCohomologyOnK* coh, uint32_t q);
    uint32_t q() const { return q_; }
    uint32_t t() const { return t_; }                  // number of basis classes
    uint32_t flat_dim() const { return dim_; }
    const std::vector<uint32_t>& exps() const { return exps_; }
    const std::vector<uint32_t>& basis() const { return basis_; }   // flat_dim x t row-major
    const engine::ModCtx<uint32_t>& ctx() const { return ctx_; }
    // Throws std::runtime_error if a column violates a relation or is not a cocycle.
    void require_equivariant_cocycles(const std::vector<uint32_t>& Y, uint32_t ncols, const std::string& what) const;
    // Coordinates (t x ncols) of equivariant cocycle columns on the stored basis.
    HMat coords(const std::vector<uint32_t>& Y, uint32_t ncols) const;
    // The C^q_eq basis as flat cochains (flat_dim x c row-major), c returned.
    std::vector<uint32_t> ceq_basis(uint32_t& c) const;
private:
    const EquivariantCohomologyOnK* coh_;
    uint32_t q_ = 0, t_ = 0, dim_ = 0, N_q_ = 0, D_ = 0;
    engine::ModCtx<uint32_t> ctx_;
    std::vector<uint32_t> exps_, basis_;
};

// The matrix of the cochain map F on H^q: column j = coords(F(b_j)); the runtime
// assertions run on every column.
HMat InducedOnH(const HFrame& src, const HFrame& dst, const BlockCochainMap& F, const std::string& what);

// Matrix arithmetic on HMat (the composition rule of the header note; results
// canonicalised row by row mod ell^{row_exp}).
HMat HMul(const HMat& A, const HMat& B);
HMat HSub(const HMat& A, const HMat& B);
HMat HAdd(const HMat& A, const HMat& B);
HMat HScale(const HMat& A, uint64_t s);
HMat HIdentity(const HMat& like_square);
HMat HCanonical(HMat A);
bool HEqual(const HMat& A, const HMat& B);
bool HIsZero(const HMat& A);
uint32_t HRankModEll(const HMat& A);
// Characteristic polynomial (field only), det(X - A), low -> high.
std::vector<uint64_t> HCharpoly(const HMat& A);

} // namespace level
} // namespace hecke

#endif // HECKE_LEVEL_ELL_H
