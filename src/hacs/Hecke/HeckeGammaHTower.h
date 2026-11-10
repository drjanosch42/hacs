// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_GAMMA_H_TOWER_H
#define HECKE_GAMMA_H_TOWER_H
// ===========================================================================
// (hecke-extensions slot 22, P10) THE CHARACTER MODEL ON Gamma_H(N) OVER Z/ell^K.
// Spec: docs/implementation/slot22/P10_spec.md (v0.3; predictions committed before any run).  Text: [C] v1.22.0-pre.1
// sec:nebwildsec -- def:nebgammaH, prop:nebgammaH (3)-(5), prop:nebgammaHheilbronn, lem:nebchainrref.
//
// A sibling of WAKE-1's Hecke/HeckeGammaH (rmk:nebcosetlevel: the class GammaHLevelN is REUSED, not forked; this file
// adds only what is level-independent): the tame scalar chi(a_h) on the table of GammaHLevelN, the elimination over the
// chain ring Z/ell^K (lem:nebchainrref), the Heilbronn operator with the scalar chi(d_{h_J(eps)})
// (prop:nebgammaHheilbronn) and the diamonds R_{sigma_d} with the scalar chi(a_R) (def:nebgammaH; chi extended to G_N).
// Weight 2 only (n = 0): the module is M = (Z/ell^K)^m, one coordinate per class J (1-based class J <-> column J - 1).
//
// Every refusal is an ENGINE_CHECK (std::logic_error) whose message starts with its name: chainring_pivot (a residual
// that the final unit-pivot rows do not absorb: the quotient is not free), tower_rss_cap (the elimination's stored
// entries exceed the byte cap).
// ===========================================================================

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "ExactInt.h"
#include "HeckeGammaH.h"
#include "HeckeLevelNSparse.h"

namespace hecke {
namespace level {
namespace tower {

// Z/ell^K with ell^K < 2^32 (every product fits a uint64).
struct ZK {
    uint32_t ell = 0, K = 0;
    uint64_t Q = 0;
    ZK() = default;
    ZK(uint32_t ell, uint32_t K);
    uint32_t add(uint32_t a, uint32_t b) const { const uint64_t s = static_cast<uint64_t>(a) + b; return static_cast<uint32_t>(s >= Q ? s - Q : s); }
    uint32_t sub(uint32_t a, uint32_t b) const { return a >= b ? a - b : static_cast<uint32_t>(a + Q - b); }
    uint32_t neg(uint32_t a) const { return a ? static_cast<uint32_t>(Q - a) : 0; }
    uint32_t mul(uint32_t a, uint32_t b) const { return static_cast<uint32_t>(static_cast<uint64_t>(a) * b % Q); }
    uint32_t pow(uint32_t a, uint64_t e) const;
    uint32_t inv(uint32_t a) const;                          // a a unit (ENGINE_CHECK)
    bool unit(uint32_t a) const { return a % ell != 0; }
    uint32_t val(uint32_t a) const;                          // v_ell(a), K at 0
    uint32_t from_i64(int64_t x) const;
    uint32_t teich(uint64_t x) const;                        // the Teichmueller representative of x mod ell (x a unit mod ell)
};

// chi = omega^e on (Z/N)^x (N a power of ell; omega the Teichmueller character), as a table of size N with values in
// Z/ell^K, 0 at the non-units.  e is read mod ell - 1.  trivial: the trivial character (every unit -> 1).
std::vector<uint32_t> TameCharacterTable(uint64_t N, const ZK& R, int64_t e, bool trivial = false);

// (M x)_J = scal[J] x_{tgt[J]}, 1-based (index 0 unused): eq:MHchi at n = 0, scal[J] = chi(a_{h_J(gamma)}).
struct MonomialChi {
    std::vector<uint32_t> tgt, scal, src;                    // src = tgt^{-1}
};
// gamma in SL_2(Z); chi a table of size N (TameCharacterTable).  Checks h_J(gamma) in Gamma_H(N) (mod N).
MonomialChi MonomialChiOf(const GammaHLevelN& G, const Mat22& gamma, const std::vector<uint32_t>& chi);

// The generators of Rel = M^{<S>} + M^{<ST>} (prop:neborbit (3), def:kappaN): the images of the unit vectors under the norms
// sum_{k < ord} M(gamma)^k, zero images dropped; ord_S = 2, ord_ST = 3 when -1 is in H (chi(-1) = 1: M(S)^2 = 1), ord_S = 4,
// ord_ST = 6 otherwise (prop:nebgammaH (3), the general form; the Gamma_1 oracle).  Values in [0, Q).
void RelGeneratorsChi(const MonomialChi& S, const MonomialChi& U, const ZK& R, uint32_t ord_S, uint32_t ord_U,
                      std::vector<SparseVec>& gen_S, std::vector<SparseVec>& gen_U);

struct ChainCokernelStats {
    uint64_t generators = 0, dependent = 0, residual_nonunit = 0;   // residuals with no unit entry (re-checked at the end)
    uint64_t final_nnz = 0, peak_nnz = 0, merge_ops = 0;
    uint64_t peak_bytes = 0;                                 // the elimination's own stored entries (rows + occurrence lists)
    double seconds = 0;
};

// lem:nebchainrref: the reduced row echelon basis of Rel with unit pivots over Z/ell^K, by sparse Gauss--Jordan whose
// pivot is the smallest column carrying a unit (mod ell this is the F_ell RREF, so the pivots are the F_ell pivots);
// a residual with no unit entry is kept and re-reduced against the final rows at the end: it must vanish (else the
// quotient is not free: refused chainring_pivot).  byte_cap > 0: refused tower_rss_cap when the stored entries pass it.
class ChainCokernel {
public:
    // allow_torsion (the wild models of HeckeNebentypusWild, where the quotient has torsion at i = 0, lem:nebwildtors): the
    // residuals that survive the final rows are kept (in free-index space, torsion_relations()) instead of refused; the
    // quotient is then (Z/ell^K)^h / <torsion_relations>.
    ChainCokernel(uint32_t D, std::vector<SparseVec> gens, const ZK& R, uint64_t byte_cap = 0, bool allow_torsion = false);
    uint32_t D() const { return D_; }
    uint32_t h() const { return static_cast<uint32_t>(free_.size()); }
    const ZK& ring() const { return R_; }
    const std::vector<uint32_t>& pivots() const { return pivots_; }      // ascending
    const std::vector<uint32_t>& free_columns() const { return free_; }  // ascending
    bool is_pivot(uint32_t c) const { return free_index_[c] < 0; }
    int32_t free_index(uint32_t c) const { return free_index_[c]; }
    // The tail of pivot c in free-index space: e_c + sum_f tail[f] e_{free f} lies in Rel, so e_c == -tail mod Rel.
    const SparseVec& tail(uint32_t c) const { return tails_[static_cast<size_t>(tail_of_[c])]; }
    // Coordinates (length h, in [0, Q)) of x mod Rel; x dense of length D, entries in [0, Q).
    std::vector<uint32_t> coords(const std::vector<uint32_t>& x) const;
    // Accumulating form: out += coef * (e_col mod Rel).
    void add_coords(uint32_t col, uint32_t coef, std::vector<uint32_t>& out) const;
    bool in_rel(const SparseVec& x) const;                   // x in Rel (coords zero)
    const ChainCokernelStats& stats() const { return st_; }
    const std::vector<SparseVec>& torsion_relations() const { return torsion_; }
private:
    uint32_t D_ = 0;
    ZK R_;
    std::vector<uint32_t> pivots_, free_;
    std::vector<int32_t> free_index_, tail_of_;
    std::vector<SparseVec> tails_, torsion_;
    ChainCokernelStats st_;
};

// An operator on M given by the images of the unit vectors: X e_J = sum (I, s) s e_I (1-based classes).
class ChiOperator {
public:
    virtual ~ChiOperator() = default;
    virtual uint32_t m() const = 0;
    virtual std::string label() const = 0;
    virtual void image(uint32_t J, std::vector<std::pair<uint32_t, uint32_t>>& out) const = 0;
};

// prop:nebgammaHheilbronn at n = 0: H x = sum_{eps in Mer_n} sum_{J admissible} chi(d_{h_J(eps)}) x_J e_{J.eps}, i.e. the image of
// e_J is sum_eps chi(d_{h_J(eps)}) e_{J.eps}.  T_n for (n, N) = 1, U_n for n | N.  Checks h_J(eps) upper triangular mod N with
// d_h in H (F-U5 m7: only chi|_H enters).
class ChiHeilbronn final : public ChiOperator {
public:
    ChiHeilbronn(const GammaHLevelN& G, uint64_t n, const std::vector<uint32_t>& chi, const ZK& R);
    uint32_t m() const override { return G_->m(); }
    std::string label() const override;
    void image(uint32_t J, std::vector<std::pair<uint32_t, uint32_t>>& out) const override;
    uint64_t n() const { return n_; }
    size_t merel_size() const { return mer_.size(); }
private:
    const GammaHLevelN* G_;
    uint64_t n_ = 1;
    const std::vector<uint32_t>* chi_;
    ZK R_;
    std::vector<Mat22> mer_;
    std::vector<std::array<uint64_t, 4>> repN_, adjN_;      // w_J and adj(w_J) mod N, 1-based
};

// The diamond <d> = R_{sigma_d} of def:nebgammaH on the chi-model: (R f)_J = chi(a_{R_J}) f_{src J}, R_J = sigma_u sigma_{u'}^{-1},
// u' = [d^{-1} u]; chi a table on all of G_N (the extension fixes the diamonds).  Checks the fibre-map identity
// h_J(gamma) R_{J.gamma} = R_J h_{src J}(gamma) mod N for gamma = S, ST (the scalars then commute with M(S), M(ST)).
class ChiDiamond final : public ChiOperator {
public:
    ChiDiamond(const GammaHLevelN& G, uint64_t d, const std::vector<uint32_t>& chi, const ZK& R);
    uint32_t m() const override { return static_cast<uint32_t>(dst_.size() - 1); }
    std::string label() const override { return "<" + std::to_string(d_) + ">"; }
    void image(uint32_t J, std::vector<std::pair<uint32_t, uint32_t>>& out) const override;
private:
    uint64_t d_ = 1;
    std::vector<uint32_t> dst_, scal_;                       // e_J -> scal_[J] e_{dst_[J]}
};

// The operator on the quotient M / Rel in the free-column basis, as sparse columns (column f = coords of X e_{free f}).
struct QuotientOperator {
    uint32_t h = 0;
    std::string label;
    std::vector<uint64_t> start;                             // size h + 1
    std::vector<std::pair<uint32_t, uint32_t>> ent;          // (row, value)
    uint64_t nnz() const { return ent.size(); }
    // y = X x (x, y of length h).
    void apply(const ZK& R, const std::vector<uint32_t>& x, std::vector<uint32_t>& y) const;
    // Y = X B for a block of s vectors stored column-major (B[k * h + i]).
    void apply_block(const ZK& R, const std::vector<uint32_t>& B, uint32_t s, std::vector<uint32_t>& Y) const;
};
// The worker count of this file's loops: ENGINE_THREADS if set (>= 1), else min(8, hardware); no value depends on it.
uint32_t TowerThreads();
QuotientOperator OnQuotient(const ChainCokernel& Q, const ChiOperator& X);
// The number of generators g with X g not in Rel (0 iff X(Rel) in Rel; prop:heilbronn (1)).
uint32_t RelEscapesChi(const ChainCokernel& Q, const std::vector<SparseVec>& gens, const ChiOperator& X);

// The Howell form over Z/ell^K (lem:nebwild (2), the chain ring Z/ell^K) of a set of dense rows of length m: pivot rows with
// pivot ell^v, closed under ell^{K-v} (the annihilator rows are added during the elimination).  length() is the length of
// the generated submodule (sum of K - v); contains() tests membership.
class HowellZK {
public:
    HowellZK(const ZK& R, std::vector<std::vector<uint32_t>> rows, uint32_t m);
    uint32_t length() const { return length_; }
    bool contains(std::vector<uint32_t> x) const;
    const std::vector<uint32_t>& pivot_valuations() const { return pval_; }
private:
    ZK R_;
    std::vector<std::vector<uint32_t>> rows_;
    std::vector<uint32_t> pcol_, pval_;
    uint32_t length_ = 0;
};

} // namespace tower
} // namespace level
} // namespace hecke

#endif // HECKE_GAMMA_H_TOWER_H
