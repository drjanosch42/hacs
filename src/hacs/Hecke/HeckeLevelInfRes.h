// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_LEVEL_INF_RES_H
#define HECKE_LEVEL_INF_RES_H
// ===========================================================================
// (hecke-extensions slot 23, LES-3) ROUTE B: THE TREE COMPLEX OF A LEVEL PAIR, INFLATION--RESTRICTION, THE HIDA p-TOWER.
// Binding text: [C] v1.23.0-pre.2 sec:nebles -- prop:levelinfres, prop:cycliccone, lem:levelfreeeuler, lem:unipotentinv,
// prop:hidanormalsteps, rmk:normalN1, prop:levelconehecke (1)-(3), obl:levelconeexact (iv), (vii); with prop:bredon (2),
// prop:heilbronn, prop:nebgammaHheilbronn, lem:nebchainrref.  Spec docs/implementation/slot22/LES0_spec.md section 3.1 (Route B);
// plan docs/implementation/slot23/00-slot23-plan.md section 2 LES-3; predictions docs/implementation/slot23/LES3_predictions.md
// (committed before the first run).
//
// WHAT IS BUILT.
//  * TreeCokernel -- prop:bredon (2): for M = Ind_A^Gamma V(chi) (LES-1's TwistedMonomialMatrix, eq:MHchi) over R = Z/ell^K
//    (K >= 1, ell >= 5 so that 6 is a unit) the model's cohomology is that of M^<S> (+) M^<U> -> M^<-I>.  M^<gamma> is the image of
//    the norm sum_{k < o} M(gamma)^k (o = 4, 6: units), spanned orbit-wise by the norms of the d = n + 1 unit vectors at one class per
//    cycle of gamma on A \ Gamma; M^- := (1 - M(-I)) M is a complement of M^<-I> containing neither M^S nor M^U.  So
//        H^1 = M^<-I> / (M^S + M^U) = M / Rel',   Rel' := M^S + M^U + M^-,
//    one sparse elimination (P10's ChainCokernel: over F_ell the RREF, over Z/ell^K the unit-pivot RREF of lem:nebchainrref, refused
//    by name chainring_pivot when the quotient is not free); h^1 = D - rank Rel', h^0 = dim M^S + dim M^U + dim M^- - rank Rel' (the
//    three families have orbit-wise ranks; M^- meets M^<-I> in 0).
//  * OnH1 -- the matrix on H^1 of a module map F : Ind_src -> Ind_dst given as a FibreMatrix (alpha, omega, R_sigma of LES-1), from the
//    images of the free-column basis vectors; Escapes counts the generators of Rel'_src whose image leaves Rel'_dst (a module map
//    preserves the tree complex: the count must be 0).  The same for Merel's operators (SymbolOperator: GammaHHeilbronn, any n, chi = 1,
//    F_ell; tower::ChiOperator: ChiHeilbronn / ChiDiamond, n = 0, a character, Z/ell^K).
//  * Dense helpers over Z/ell^K (row-major): products, rank mod ell, the Howell length of a column span (tower::HowellZK), the Jordan
//    type of a unipotent matrix over F_ell.
//  * RouteBSixTerm -- eq:levelsixterm's dimensions read on the two tree complexes (field only): the (LES5) comparison with Route A.
//  * FiniteGroupH1SL2 -- oracle O3: dim H^1(SL_2(F_ell); Sym^n F_ell) by crossed homomorphisms on the Cayley graph of SL_2(F_ell)
//    (generators S, T mod ell), with its own rho_n by substitution (les_model.rho's convention); never the model.
//
// Nothing here is reached from a shipped record, report, campaign job id, setting or golden text.
// ===========================================================================

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "HeckeGammaH.h"
#include "HeckeGammaHTower.h"
#include "HeckeLevelNSparse.h"
#include "HeckeLevelPair.h"
#include "../../ModArithmetic/ModCtx.h"

namespace hecke {
namespace level {
namespace les {

// The ModCtx of Z/ell^K: ModCtx(ell) at K = 1, ModCtx::prime_power(ell, K) otherwise.
engine::ModCtx<uint32_t> RingCtx(uint32_t ell, uint32_t K);

struct TreeCokernelStats {
    uint32_t dim_S = 0, dim_U = 0, dim_minus = 0, rank_rel = 0;
    uint64_t generators = 0;
    double seconds = 0;
};

class TreeCokernel {
public:
    // A, chi over R = Z/ell^K (chi's table with values in [0, ell^K)); ell >= 5 prime (ENGINE_CHECK).  keep_generators: retain the
    // generators of Rel' for Escapes.  byte_cap: ChainCokernel's (0 = none).
    TreeCokernel(const CosetLevel& A, const CharacterDatum& chi, uint32_t n_sym, uint32_t ell, uint32_t K, bool keep_generators = true,
                 uint64_t byte_cap = 0);
    const CosetLevel& level() const { return *A_; }
    uint32_t D() const { return D_; }
    uint32_t d() const { return d_; }
    uint32_t ell() const { return ell_; }
    uint32_t K() const { return K_; }
    uint32_t h1() const { return Q_->h(); }
    uint32_t h0() const { return st_.dim_S + st_.dim_U + st_.dim_minus - st_.rank_rel; }
    const TreeCokernelStats& stats() const { return st_; }
    const tower::ChainCokernel& Q() const { return *Q_; }
    const tower::ZK& ring() const { return R_; }
    const std::vector<SparseVec>& generators() const { return gens_; }

private:
    const CosetLevel* A_;
    uint32_t D_ = 0, d_ = 1, ell_ = 0, K_ = 1;
    tower::ZK R_;
    std::unique_ptr<tower::ChainCokernel> Q_;
    std::vector<SparseVec> gens_;
    TreeCokernelStats st_;
};

// A dense matrix over Z/ell^K, row-major.
struct DMat {
    uint32_t r = 0, c = 0;
    std::vector<uint32_t> a;
    DMat() = default;
    DMat(uint32_t rr, uint32_t cc) : r(rr), c(cc), a(static_cast<size_t>(rr) * cc, 0) {}
    uint32_t& at(uint32_t i, uint32_t j) { return a[static_cast<size_t>(i) * c + j]; }
    uint32_t at(uint32_t i, uint32_t j) const { return a[static_cast<size_t>(i) * c + j]; }
};
DMat Identity(uint32_t h);
DMat MatMul(const tower::ZK& R, const DMat& A, const DMat& B);
DMat MatAdd(const tower::ZK& R, const DMat& A, const DMat& B);
DMat MatSub(const tower::ZK& R, const DMat& A, const DMat& B);
DMat ReduceMod(const DMat& A, uint32_t m);              // entrywise mod m
uint64_t Differ(const DMat& A, const DMat& B);          // UINT64_MAX on a shape mismatch
bool IsZero(const DMat& A);
uint32_t RankModEll(const DMat& A, uint32_t ell);       // the rank of A mod ell
uint32_t SpanLength(const tower::ZK& R, const DMat& A); // the Howell length of the column span of A over Z/ell^K
// The Jordan type of a unipotent u over F_ell (u - 1 nilpotent): out[k] = the number of blocks of size k (k >= 1); ranks[k] = rank (u-1)^k.
std::vector<uint32_t> JordanTypeUnipotent(const DMat& u, uint32_t ell, std::vector<uint32_t>* ranks = nullptr);
std::string JordanText(const std::vector<uint32_t>& jt);   // "J_5^4+J_1^1"

// The matrix of F (m_dst x m_src blocks of d, a module map Ind_src -> Ind_dst) on H^1: h1(dst) x h1(src).
DMat OnH1(const FibreMatrix& F, const TreeCokernel& src, const TreeCokernel& dst);
// The generators of Rel'(src) whose F-image is not in Rel'(dst) (needs src's kept generators).
uint64_t Escapes(const FibreMatrix& F, const TreeCokernel& src, const TreeCokernel& dst);
// Merel's operators on one level.
DMat OnH1(const SymbolOperator& X, const TreeCokernel& Q);
uint64_t Escapes(const SymbolOperator& X, const TreeCokernel& Q);
DMat OnH1(const tower::ChiOperator& X, const TreeCokernel& Q);      // n = 0
uint64_t Escapes(const tower::ChiOperator& X, const TreeCokernel& Q);

// eq:levelsixterm on the two tree complexes over F_ell (K = 1).  alpha: RestrictionFibreMap's.
struct RouteBSix {
    uint32_t h0A = 0, h1A = 0, h0B = 0, h1B = 0, rank_res1 = 0, H1c = 0, H2c = 0;
    uint64_t alpha_escapes = 0;
    uint32_t ker_res1() const { return h1A - rank_res1; }
};
RouteBSix RouteBSixTerm(const CosetLevel& A, const CharacterDatum& chiA, const CosetLevel& B, const CharacterDatum& chiB, uint32_t n_sym,
                        uint32_t ell);

// Oracle O3: dim H^1(SL_2(F_ell); Sym^n F_ell) (ell prime), by crossed homomorphisms on the Cayley graph (generators S, T mod ell);
// *VG = dim V^G, *order = |SL_2(F_ell)|.
uint32_t FiniteGroupH1SL2(uint32_t ell, uint32_t n_sym, uint32_t* VG = nullptr, uint64_t* order = nullptr);

} // namespace les
} // namespace level
} // namespace hecke

#endif // HECKE_LEVEL_INF_RES_H
