// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_GAMMA_H_H
#define HECKE_GAMMA_H_H
// ===========================================================================
// (hecke-extensions slot 22, WAKE-1) THE LEVEL Gamma_H(N) ON THE FIXED MODEL.
// Spec: docs/implementation/slot22/WAKE1_spec.md (v0.2; predictions committed before any run).  Text: the P10-C
// proposal docs/implementation/slot22/P10C_contract_text.tex.txt -- def:nebgammaH, prop:nebgammaH (1), (2), (4) and
// rmk:nebcosetlevel ("one level interface": WAKE-1 writes the class, P10 reuses it, a second Gamma_H class is not to
// be written).  NOT yet in [C] (P10-C is a pre-release under referee); nothing here is reached from a shipped record,
// report, campaign job id, setting or golden text.
//
//   Gamma_H(N) = { gamma in Gamma_0(N) : d_gamma mod N in H },  H <= G_N = (Z/N)^x given by generators.
//
// WHAT IS BUILT.
//  * GammaHLevelN : CosetLevel -- the table of prop:nebgammaH (2): w_{(j,u)} = sigma_u w_j, j the def:cosetrepsN index
//    (Gamma0LevelN), u in U = the least representatives in [1, N) of G_N / H ascending, j-major, u-minor; sigma_u of
//    lem:gammaonetable ([[u^{-1} mod N, beta], [N, u]], sigma_1 = I).  H = G_N is Gamma0LevelN's table entry for entry,
//    H = {1} Gamma1LevelN's.  Besides the CosetLevel members: the bottom-row description of the classes (the class of
//    gamma is the H-orbit of its bottom row mod N) -- class_of_row, act / act_defined for any integral delta -- which
//    is what the Manin-symbol (cokernel) route needs.  PrimeToPart(N, p): H = the elements of order prime to p, i.e.
//    the kernel of G_N -> G_N (x) Z_p: WAKE's H at N = l_0 l_1 (the tame family over Delta = the p-Sylow) and P10's
//    mu_{l-1} at N = l^r (p = l).
//  * GammaHSubTransversal -- Gamma1SubTransversal's matrices (they lie in Gamma_1(N) <= Gamma_H(N)), membership checked.
//  * GammaHDiamondOnModel -- <d> = R_{sigma_d} as a fibre map on the fixed model (DiamondOnModel generalised; the
//    oracle side of the model route).
//  * SymbolOperator / GammaHHeilbronn / GammaHDiamond -- operators on M = F_p[Gamma_H \ Gamma] (x) V_n given by the
//    image of each basis vector: T_n / U_n by Merel's set Mer_n (any n >= 1; pairs (c_J, d_J) eps not primitive mod N
//    dropped, block rho_n(adj h_J(eps)), h_J(eps) = w_J eps w_{J.eps}^{-1} -- lower left 0, lower right in H mod N,
//    checked), <d> by the fibre map.  Applied on the fly (no m |Mer_n| term table): the sparse route of V3 needs that.
//  * QuotientCoords -- the coordinates of every basis vector of M in M / Rel (a SparseCokernel), so that an operator's
//    matrix on the quotient, or on a subspace of it, is read without touching Rel again.
//  * CuspsOf / boundary -- n = 0, -1 in H: the cusps are the T-orbits on the classes (Gamma_H w Gamma_inf), the
//    boundary of the symbol e_J = g{0, inf} is [cusp(J)] - [cusp(J S)]; nu_2, nu_3 = the fixed classes of S, ST; the
//    genus by Riemann--Hurwitz.  Refused by name (gammaH_minus_one) if -1 is not in H.
// ===========================================================================

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "ExactInt.h"
#include "HeckeLevelEll.h"
#include "HeckeLevelN.h"
#include "HeckeLevelNSparse.h"
#include "../SimplicialCohomology.h"
#include "../../ModArithmetic/ModCtx.h"

namespace hecke {
namespace level {

class GammaHLevelN final : public CosetLevel {
public:
    // N in [1, 2^20); every generator a unit mod N; H = the subgroup they generate (BFS closure).  The table is built
    // and self-checked eagerly (w_1 = I; class_of(w_J) = J and class_of_row(bottom_row(J)) = J for every J).
    GammaHLevelN(uint64_t N, const std::vector<uint64_t>& generators);
    // H = { u in G_N : the order of u is prime to p } (generators chosen greedily, ascending).
    static GammaHLevelN PrimeToPart(uint64_t N, uint64_t p);

    uint64_t N() const override { return N_; }
    uint32_t m() const override { return m_; }
    Mat22 rep(uint32_t J) const override;                    // w_J = sigma_{u_r} w_j, 1-based, w_1 = I
    uint32_t class_of(const Mat22& gamma) const override;
    bool contains(const Mat22& gamma) const override;        // gamma in Gamma_H(N)
    bool contains_wide(const Mat22W& gamma) const override;
    std::string name() const override;

    // ---- H and G_N / H -------------------------------------------------------------------------------
    bool in_H(uint64_t u) const;                             // u mod N in H (false at a non-unit)
    uint32_t index_H() const { return static_cast<uint32_t>(U_.size()); }   // [G_N : H]
    const std::vector<uint64_t>& coset_reps() const { return U_; }          // U ascending, U[0] = 1
    uint32_t coset_rank(uint64_t u) const;                   // r with u in U[r] H (ENGINE_CHECK u a unit)
    const std::vector<uint64_t>& H_elements() const { return H_; }          // ascending
    const std::vector<uint64_t>& generators() const { return gens_; }
    bool contains_minus_one() const { return in_H(N_ - 1 + (N_ == 1 ? 1 : 0)); }
    std::string H_text() const;                              // "<g1,g2,...>" (|H| = ...)

    // ---- the table ------------------------------------------------------------------------------------
    const Gamma0LevelN& level0() const { return lvl0_; }
    Mat22 sigma(uint64_t u) const;                           // lem:gammaonetable's sigma_u
    uint32_t index(uint32_t j, uint32_t r) const;            // (j, U[r]) -> J, 1-based j, 0-based r
    uint32_t j_of(uint32_t J) const { return (J - 1) / index_H() + 1; }
    uint32_t r_of(uint32_t J) const { return (J - 1) % index_H(); }
    std::pair<uint64_t, uint64_t> bottom_row(uint32_t J) const;              // of w_J, mod N
    uint32_t class_of_row(int64_t c, int64_t d) const;      // the class whose bottom rows are the H-orbit of (c, d)
    bool act_defined(uint32_t J, const Mat22& delta) const;  // bottom_row(J) delta primitive mod N
    uint32_t act(uint32_t J, const Mat22& delta) const;      // class_of_row(bottom_row(J) delta)

private:
    uint64_t N_ = 1;
    uint32_t m_ = 1;
    Gamma0LevelN lvl0_;
    std::vector<uint64_t> gens_, H_, U_;
    std::vector<uint8_t> inH_;                               // size N
    std::vector<int32_t> rank_;                              // size N; -1 at a non-unit
    std::vector<Mat22> sigma_;                               // sigma_{U[r]}
    std::vector<std::pair<uint64_t, uint64_t>> row0_;        // bottom row of w_j mod N, 1-based j
    // class_of_row: per prime-power factor i of N and per j, which coordinate of row0_[j] is a unit there (0 = c, 1 = d)
    // and its inverse mod Q_i; the CRT idempotents e_i.
    std::vector<uint8_t> unit_coord_;                        // (j - 1) * F + i
    std::vector<uint64_t> unit_inv_;
    std::vector<uint64_t> idem_;
    void build(uint64_t N, const std::vector<uint64_t>& generators);
};

// r_t representing (Gamma_H(N) cap Gamma_0(N p)) \ Gamma_H(N), p prime: Gamma1SubTransversal(N, p) (every r_t in Gamma_1(N)),
// each checked to lie in Gamma_H(N).
std::vector<Mat22> GammaHSubTransversal(const GammaHLevelN& G, uint64_t p);

// <d> = R_{sigma_d} on L_{Gamma_H(N)} (the P10-C text's def:nebgammaH): (R f)_{(j,u)} = rho_n(sigma_u sigma_{u'}^{-1}) f_{(j,u')},
// u' = [d^{-1} u]; checked to commute with the monomial action of S and T.  inverse = true: the inverse normalisation
// (d^{-1} in place of d) -- a NEGATIVE CONTROL only.
BlockCochainMap GammaHDiamondOnModel(const GammaHLevelN& G, uint64_t d, const SimplicialComplex& K, uint32_t n_sym,
                                     const engine::ModCtx<uint32_t>& ctx, uint32_t q, bool inverse = false);

// ---- operators on M = F_p[Gamma_H \ Gamma] (x) V_n, by the images of the basis vectors -------------------------------
// Flat layout of M: column (J - 1) * (n + 1) + a (the layout of SparseCokernel / OrbitWiseFixedGenerators).
class SymbolOperator {
public:
    virtual ~SymbolOperator() = default;
    virtual uint32_t D() const = 0;
    virtual uint32_t p() const = 0;
    virtual std::string label() const = 0;
    // Appends (column, coefficient in [0, p)) pairs whose sum is X e_col (repeated columns allowed).
    virtual void image(uint32_t col, std::vector<std::pair<uint32_t, uint32_t>>& out) const = 0;
};

class GammaHHeilbronn final : public SymbolOperator {
public:
    // [Gamma_H diag(1, n) Gamma_H] by Mer_n (n >= 1, n < 2^20): T_n for (n, N) = 1, U_n for n | N (the dropped pairs), the
    // general double-coset sum of determinant n otherwise.  ctx a prime field.
    GammaHHeilbronn(const GammaHLevelN& G, uint64_t n, uint32_t n_sym, const engine::ModCtx<uint32_t>& ctx);
    uint32_t D() const override { return G_->m() * d_; }
    uint32_t p() const override { return ctx_.p; }
    std::string label() const override;
    void image(uint32_t col, std::vector<std::pair<uint32_t, uint32_t>>& out) const override;
    uint64_t n() const { return n_; }
    const std::vector<Mat22>& merel() const { return mer_; }
private:
    const GammaHLevelN* G_;
    uint64_t n_ = 1;
    uint32_t d_ = 1;
    engine::ModCtx<uint32_t> ctx_;
    std::vector<Mat22> mer_;
    std::vector<uint32_t> rho_block(const Mat22& adj_h) const;   // rho_n(adj h) mod p, d x d row-major
};

class GammaHDiamond final : public SymbolOperator {
public:
    // <d> (d a unit mod N) as the fibre map of GammaHDiamondOnModel on M.  inverse: NEGATIVE CONTROL only.
    GammaHDiamond(const GammaHLevelN& G, uint64_t d, uint32_t n_sym, const engine::ModCtx<uint32_t>& ctx, bool inverse = false);
    uint32_t D() const override { return G_->m() * d_; }
    uint32_t p() const override { return ctx_.p; }
    std::string label() const override;
    void image(uint32_t col, std::vector<std::pair<uint32_t, uint32_t>>& out) const override;
private:
    const GammaHLevelN* G_;
    uint64_t dd_ = 1;
    bool inverse_ = false;
    uint32_t d_ = 1;
    engine::ModCtx<uint32_t> ctx_;
    std::vector<uint32_t> dst_;                              // per class J (1-based): the target class
    std::vector<std::vector<uint32_t>> blk_;                 // per class J: rho_n(sigma_{u} sigma_{u'}^{-1}), d x d
};

// X x for a sparse x (columns ascending, values nonzero, result in the same form).
SparseVec ApplySym(const SymbolOperator& X, const SparseVec& x);
// prop:heilbronn (1) for X: the number of kept generators of Rel whose image leaves Rel (0 iff X(Rel) in Rel).
uint32_t RelEscapesSym(const SparseCokernel& Q, const SymbolOperator& X);

// The coordinates of every basis vector of M in M / Rel on the free-column basis: free column f -> e_{free_index(f)},
// pivot c -> -(tail of c).
class QuotientCoords {
public:
    explicit QuotientCoords(const SparseCokernel& Q);
    uint32_t D() const { return static_cast<uint32_t>(start_.size() - 1); }
    uint32_t h() const { return h_; }
    uint32_t p() const { return p_; }
    // (free index, value) pairs of column col.
    const std::pair<uint32_t, uint32_t>* begin(uint32_t col) const { return &ent_[start_[col]]; }
    const std::pair<uint32_t, uint32_t>* end(uint32_t col) const { return &ent_[start_[col + 1]]; }
    const std::vector<uint32_t>& free_columns() const { return free_; }
    uint64_t nnz() const { return ent_.size(); }
private:
    uint32_t h_ = 0, p_ = 0;
    std::vector<uint32_t> free_;
    std::vector<uint64_t> start_;
    std::vector<std::pair<uint32_t, uint32_t>> ent_;
};

// ---- cusps (n = 0, -1 in H) ------------------------------------------------------------------------------------
struct CuspData {
    uint32_t c = 0;                                          // number of cusps
    std::vector<uint32_t> cusp;                              // per class J (1-based): the cusp id in [0, c)
    std::vector<uint32_t> width;                             // per cusp: the orbit length (the width)
    uint32_t nu2 = 0, nu3 = 0;                               // classes fixed by S, by ST
    int64_t twelve_g = 0;                                    // 12 g = 12 + m - 3 nu_2 - 4 nu_3 - 6 c (Riemann--Hurwitz)
    int64_t genus() const { return twelve_g / 12; }
    bool genus_integral() const { return twelve_g % 12 == 0 && twelve_g >= 0; }
    std::vector<uint32_t> cusp_of_S;                         // per class J: cusp(J S)
};
// Refused by name gammaH_minus_one if -1 is not in H (the orbifold count needs -I in Gamma_H).
CuspData CuspsOf(const GammaHLevelN& G);

} // namespace level
} // namespace hecke

#endif // HECKE_GAMMA_H_H
