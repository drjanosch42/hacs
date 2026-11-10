// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_LEVEL_N_HEILBRONN_H
#define HECKE_LEVEL_N_HEILBRONN_H
// ===========================================================================
// (hecke-extensions slot 15, WP-7) THE HEILBRONN--MEREL ACTION H_{N,p} ON THE
// COKERNEL (MANIN-SYMBOL) PRESENTATION -- obstacle O2 (levelO2) of [C]
// rmk:levelscaling, closed on paper by [C] v1.14.x sec:levelN:
//   lem:merelpath    Merel's set Mer_n = { [[a,b],[c,d]] : ad - bc = n,
//                    a > b >= 0, d > c >= 0 } and its path condition;
//   prop:heilbronn   eq:heilbronn --
//                      H_{N,p} = sum_{eps in Mer_p} sum_{j admissible}
//                                E_{j.eps, j} (x) rho_n(adj h_j(eps)),
//                      adj h_j(eps) = w_{j.eps} (adj eps) w_j^{-1},
//                    for T_p^{(N)} (p !| N: every j admissible) and U_p^{(N)}
//                    (p | N: exactly the j with (c_j, d_j) eps != 0 mod p) by
//                    ONE formula; (1) H(Rel_N) in Rel_N, (2) [H] = [A g A] on
//                    M_N / Rel_N = H^1(Gamma_0(N); V), (3) N = 1, (4) no
//                    tau-table, no coset search, nothing outside Gamma inverted;
//   rmk:heilbronnconv the transcription and its trap (the two mis-transcriptions
//                    (a), (b), kept here as NEGATIVE CONTROLS only);
//   def:kappaN / prop:levelorbit (2)  the presentation M_N / Rel_N,
//                    Rel_N = M^{<S>} + M^{<U>}, U = ST -- exactly the spans
//                    HeckeLevelN.h's OrbitWiseFixedSpace builds and
//                    CokernelFromFixedSpaces counts.
//
// WHAT IS BUILT.
//  * MerelSet(n)              -- Mer_n, enumerated in a fixed order (a, d, b, c).
//  * HeilbronnOperator        -- H_{N,p} on M_N over ANY word-sized ModCtx (field or
//                                Z/ell^k), kept as block terms (dst block, src block,
//                                d x d block): O(|Mer_p| m_N) terms, never a D x D
//                                object except dense() for the comparison routes.
//  * CokernelPresentation     -- M_N / Rel_N over a prime field: the RREF of Rel_N
//                                (dense; the sparse elimination of rmk:levelcost (4)
//                                is WP-8's), the free columns as the basis of the
//                                quotient, reduction to coordinates, membership.
//  * OnCokernel(...)          -- the matrix [H_{N,p}] on M_N / Rel_N (an HMat, field).
//
// The comparison with the tau-route (prop:levelcorrespondence (4)(d) on the model)
// is (LN3), in HeckeLevelNHeilbronnValidation; nothing here reads a tau-table.
//
// Nothing here is reached from a shipped record, report, campaign job id or golden
// text: pipeline-free and value-free with respect to every shipped object.
// MP note: nothing here hard-wires the 7-vertex model (no model is read at all), a
// trivial centre (-I acts by (-1)^n, n even required where the presentation needs
// it), or Bredon-only semantics at ell in {2, 3} (the cokernel refuses 6 not a unit).
// ===========================================================================

#include <cstdint>
#include <string>
#include <vector>

#include "HeckeLevelEll.h"
#include "HeckeLevelN.h"

namespace hecke {
namespace level {

// lem:merelpath: Mer_n for n >= 1, in the fixed order (a asc, d asc, b asc, c asc).
std::vector<Mat22> MerelSet(uint64_t n);

// Which formula HeilbronnOperator implements.  Only kProposition is prop:heilbronn;
// the two traps of rmk:heilbronnconv (2) exist ONLY as negative controls.
enum class HeilbronnVariant : uint8_t {
    kProposition = 0,   // eq:heilbronn
    kTrapA = 1,         // (a): rho_n(h_j(eps)) in place of rho_n(adj h_j(eps))
    kTrapB = 2,         // (b): Mer_p itself as the right-coset set (= eq:heilbronn over adj(Mer_p))
};

struct HeilbronnTerm {
    uint32_t dst = 0, src = 0;   // 1-based block indices: (H x)_dst += B x_src
    uint32_t block = 0;          // index into HeilbronnOperator::blocks()
};

// (slot-16 WP-17, audit WP-13 F1) VALIDATION-ONLY: while an instance is alive on this thread, every
// HeilbronnOperator built on it takes the __int128 (Mat22W) path for EVERY term, not only after an int64
// HeckeOverflow.  Default off; no shipped caller sets it.  It cannot change a value: the wide path computes the
// same integer matrices and hands RhoCache the same residues mod ctx.p (HeckeLevelNSparseValidation (F1)
// compares the two term lists block for block).
class ScopedForceWideHeilbronn {
public:
    ScopedForceWideHeilbronn();
    ~ScopedForceWideHeilbronn();
    ScopedForceWideHeilbronn(const ScopedForceWideHeilbronn&) = delete;
    ScopedForceWideHeilbronn& operator=(const ScopedForceWideHeilbronn&) = delete;
    static bool active();
private:
    bool prev_;
};

class HeilbronnOperator {
public:
    // lvl: the def:cosetrepsN table; p prime (any, p | N or not); n_sym >= 0; ctx any
    // word-sized ring.  Builds the term list eagerly and self-checks every
    // adj h_j(eps): integral, det p, lower left = 0 mod N (lem:cosetrepsN (4)).
    HeilbronnOperator(const Gamma0LevelN& lvl, uint64_t p, uint32_t n_sym, const engine::ModCtx<uint32_t>& ctx,
                      HeilbronnVariant variant = HeilbronnVariant::kProposition);
    // (slot-25 HZ-RAT-PERF (c)) the PARTIAL operator of eq:heilbronn over a sub-list `part` of Mer_p (kProposition only): the same
    // terms, in the same order, as the full operator restricted to those matrices, with the same checks.  For a partition of Mer_p
    // the partial operators sum to H_{N,p} exactly (over ctx); the streamed T_ell export applies them one slice at a time, so the
    // term table of a large p is never materialised whole.
    HeilbronnOperator(const Gamma0LevelN& lvl, uint64_t p, uint32_t n_sym, const engine::ModCtx<uint32_t>& ctx,
                      std::vector<Mat22> part);

    uint64_t N() const { return N_; }
    uint64_t p() const { return p_; }
    uint32_t m() const { return m_; }
    uint32_t d() const { return d_; }
    uint32_t D() const { return m_ * d_; }
    bool bad() const { return N_ % p_ == 0; }                  // U_p^{(N)} (p | N) vs T_p^{(N)}
    std::string label() const { return std::string(bad() ? "U_" : "T_") + std::to_string(p_) + "^(N)"; }
    HeilbronnVariant variant() const { return variant_; }
    const engine::ModCtx<uint32_t>& ctx() const { return ctx_; }

    const std::vector<Mat22>& merel() const { return mer_; }   // the matrix set actually used
    const std::vector<HeilbronnTerm>& terms() const { return terms_; }
    const std::vector<Block>& blocks() const { return blocks_; }
    uint32_t pairs_total() const { return pairs_total_; }       // |set| * m_N
    uint32_t pairs_dropped() const { return pairs_dropped_; }   // inadmissible (p | N only)
    uint32_t mixed_terms() const { return mixed_terms_; }       // terms touching a case-(iii) representative
    uint32_t wide_terms() const { return wide_terms_; }         // terms computed in __int128 (slot-16 WP-13)

    // y = H x (x, y of length D, over ctx).
    std::vector<uint32_t> apply(const std::vector<uint32_t>& x) const;
    // The dense D x D matrix (row-major) -- comparison routes only.
    std::vector<uint32_t> dense() const;

private:
    void build(const Gamma0LevelN& lvl, uint32_t n_sym);   // the term list over mer_
    uint64_t N_ = 1, p_ = 2;
    uint32_t m_ = 1, d_ = 1;
    HeilbronnVariant variant_ = HeilbronnVariant::kProposition;
    engine::ModCtx<uint32_t> ctx_;
    std::vector<Mat22> mer_;
    std::vector<HeilbronnTerm> terms_;
    std::vector<Block> blocks_;
    uint32_t pairs_total_ = 0, pairs_dropped_ = 0, mixed_terms_ = 0, wide_terms_ = 0;
};

// prop:levelorbit (2) / def:kappaN: the presentation M_N / Rel_N over a prime field
// F_p (p = ctx.p, 6 a unit), Rel_N spanned by the orbit-wise fixed spaces of S
// (order 2) and U = ST (order 3).  Dense RREF (the oracle for WP-8's sparse route).
class CokernelPresentation {
public:
    CokernelPresentation(const FixedSpace& fS, const FixedSpace& fU, uint32_t p);
    // Convenience: the fixed spaces built here, orbit by orbit (n even).
    static CokernelPresentation Build(const CosetLevel& A, uint32_t n_sym, const engine::ModCtx<uint32_t>& ctx);
    // (slot-22 LN23, object=curve) n = 0 only (refused by name otherwise), every prime p: the fixed spaces as orbit
    // sums, M / (M^S + M^{ST}) = H_1(X_0(N), cusps; Z) (x) F_p = H^1(Y_0(N); F_p), the cohomology of the open coarse
    // curve ([C] prop:levelNcurve; Manin's presentation).
    static CokernelPresentation BuildCurve(const CosetLevel& A, uint32_t n_sym, const engine::ModCtx<uint32_t>& ctx);

    uint32_t D() const { return D_; }
    uint32_t dim_rel() const { return static_cast<uint32_t>(pivots_.size()); }
    uint32_t h1() const { return static_cast<uint32_t>(free_.size()); }
    uint32_t p() const { return p_; }
    const std::vector<uint32_t>& free_columns() const { return free_; }   // basis e_c of M / Rel
    const std::vector<std::vector<uint32_t>>& rel_basis() const { return rref_; }

    // Coordinates of x mod Rel_N on the free columns (length h1).
    std::vector<uint32_t> coords(const std::vector<uint32_t>& x) const;
    bool in_rel(const std::vector<uint32_t>& x) const;

private:
    uint32_t D_ = 0, p_ = 0;
    std::vector<std::vector<uint32_t>> rref_;   // pivot 1, zero in every other pivot column
    std::vector<uint32_t> pivots_;
    std::vector<uint32_t> free_;
    struct CurveTag {};
    CokernelPresentation(const FixedSpace& fS, const FixedSpace& fU, uint32_t p, CurveTag);
    void init(const FixedSpace& fS, const FixedSpace& fU);
};

// The matrix of [H] on M_N / Rel_N in the free-column basis (field HMat, h1 x h1).
// Throws (ENGINE_CHECK) unless H(Rel_N) is contained in Rel_N (prop:heilbronn (1)),
// checked on every RREF row of Rel_N first.
HMat OnCokernel(const CokernelPresentation& pres, const HeilbronnOperator& H);
// prop:heilbronn (1), reported rather than thrown: the number of Rel_N basis rows
// whose image leaves Rel_N (0 iff H(Rel_N) in Rel_N).
uint32_t RelEscapes(const CokernelPresentation& pres, const HeilbronnOperator& H);

} // namespace level
} // namespace hecke

#endif // HECKE_LEVEL_N_HEILBRONN_H
