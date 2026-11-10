// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_NEBENTYPUS_GALOIS_H
#define HECKE_NEBENTYPUS_GALOIS_H
// ===========================================================================
// (hecke-extensions slot 18, WP-6) THE CHARACTER MODEL OVER THE GALOIS RING W_K(F_q).
// Binding text: [C] v1.18.0-pre.1 sec:nebentypus -- lem:teichmuller (W_K(F_q) = (Z/ell^K)[Y]/(C~), a chain
// ring with ideals (ell^j), residue field F_q; the Teichmuller lift t), prop:neborbit (1), (3) over a
// commutative ring with 6 a unit, rmk:nebcost (4) (the module route with Howell forms over the chain ring),
// and the code map's item (iv): cohomology over W_K(F_q) at d(chi) >= 2, K <= 3, dense then sparse; the
// result read as lengths and invariant factors, never as a dimension.
//
// WHAT IS BUILT (validation / env-hook only: no record key, no setting, no verb, no golden object).
//  * WkArith -- W_K(F_q) element arithmetic on fixed-width coefficient arrays (d <= 4, ell^K < 2^32):
//    the ell-adic valuation (min over the coefficients: W_K is unramified, free over Z/ell^K on 1, Y, ..),
//    exact and floor division by ell^v coefficientwise, the inverse of a unit (F_q inverse, Newton lift).
//  * NebWkCokernel -- M = Ind_{Gamma_0(N)}^Gamma V(t o chi) over W_K (eq:MHchi with the Teichmuller-lifted
//    character, rho_n(h) mod ell^K), the orbit generators of M^<S>, M^<ST> (prop:neborbit (1): the averaging
//    idempotents (1/o) sum gamma^i, o in {2, 3}, units since ell >= 5), and the canonical HOWELL FORM of
//    Rel = M^S + M^ST over the chain ring on one of two routes:
//      dense  -- column by column, the least-valuation pivot of the remaining rows, the annihilator multiple
//                ell^{K-v} (pivot row) re-entering the pool (the Howell saturation);
//      sparse -- incremental insertion of the sparse generators into a pivot-keyed basis (a smaller valuation
//                displaces the stored pivot row, which is reduced and re-inserted), saturation as above.
//    Both end in the same canonical reduction (pivot = ell^v exactly; every entry above a pivot ell^v reduced
//    coefficientwise into [0, ell^v)), so the two routes agree entrywise iff both are correct: the Howell form
//    of a submodule of W_K^D is unique.  H^1 = W_K^D / Rel is read by a Smith reduction of the non-unit
//    pivot rows (the unit-pivot rows split off), as exponents e in [1, K] (W/ell^e; e = K: free); lengths
//    len Rel = sum (K - v), len H^1 = K D - len Rel, len H^0 = K (rk M^S + rk M^ST) - len Rel (the exact
//    sequence 0 -> H^0 -> M^S (+) M^ST -> M -> H^1 -> 0 of prop:neborbit (3); M^S, M^ST free summands).
//  * K = 1 is F_q itself: the Howell form is the RREF, comparable entrywise with NebCokernel /
//    NebSparseCokernel (codes: the F_q code of sum c_i Y^i is sum c_i ell^i).
//
// Refusals (NebentypusRefusal, by name): nebentypus_small_ell (ell < 5), wk_ring_too_large (ell^K >= 2^32 or
// d > 4), wk_dense_cap (dense route above kWkDenseCap).  chi(-1) != (-1)^n gives parity_zero (prop:neborbit (2)).
//
// MP constraint ([C] box after rmk:nebcost (v)): the route reads no simplicial model; -I enters only through
// its scalar chi(-1)(-1)^n (certified on every orbit).
// ===========================================================================

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "HeckeLevelN.h"
#include "HeckeNebentypus.h"

namespace hecke {
namespace neb {

constexpr uint32_t kWkMaxDegree = 4;
// D above which the dense W_K route refuses (wk_dense_cap); the sparse route has no cap of its own.
constexpr uint32_t kWkDenseCap = 2048;

using Wel = std::array<uint64_t, kWkMaxDegree>;

class WkArith {
public:
    WkArith(const Fq& F, uint32_t K);
    const Fq& field() const { return *F_; }
    uint32_t K() const { return K_; }
    uint32_t d() const { return d_; }
    uint64_t ell() const { return ell_; }
    uint64_t modulus() const { return pw_[K_]; }
    uint64_t ell_pow(uint32_t j) const { return pw_[j]; }

    Wel zero() const { return Wel{}; }
    Wel one() const { Wel w{}; w[0] = 1 % pw_[K_]; return w; }
    Wel scalar(int64_t s) const;                         // s mod ell^K, embedded
    Wel lift(Fq::E x) const;                             // coefficientwise lift in [0, ell)
    Fq::E reduce(const Wel& a) const;                    // mod ell, an F_q code
    bool is_zero(const Wel& a) const;
    Wel add(const Wel& a, const Wel& b) const;
    Wel sub(const Wel& a, const Wel& b) const;
    Wel mul(const Wel& a, const Wel& b) const;
    Wel mul_scalar(const Wel& a, uint64_t s) const;      // s in [0, ell^K)
    uint32_t val(const Wel& a) const;                    // the ell-adic valuation; K at 0
    Wel div_exact(const Wel& a, uint32_t v) const;       // a / ell^v coefficientwise (val(a) >= v, checked)
    Wel div_floor(const Wel& a, uint32_t v) const;       // coefficientwise floor(c / ell^v)
    Wel inv(const Wel& u) const;                         // u a unit (val 0), checked
    Wel teichmuller(Fq::E x) const;                      // lift(x)^{q^{K-1}}, lem:teichmuller (2)
    std::string text(const Wel& a) const;
private:
    const Fq* F_;
    uint32_t K_ = 1, d_ = 1;
    uint64_t ell_ = 0;
    std::vector<uint64_t> pw_;                           // ell^0 .. ell^K
    std::vector<uint64_t> C_;                            // the lift of C (degree d, monic)
};

// A sparse row over W_K: ascending columns, non-zero values.
struct WkRow {
    std::vector<uint32_t> col;
    std::vector<Wel> val;
    size_t nnz() const { return col.size(); }
};

enum class WkRoute : uint8_t { kDense = 0, kSparse = 1 };
std::string WkRouteName(WkRoute r);

struct WkStats {
    uint64_t generators = 0, saturation_rows = 0, displaced = 0, final_nnz = 0, merge_ops = 0;
    double generators_s = 0, eliminate_s = 0, smith_s = 0;
};

class NebWkCokernel {
public:
    // The character model of (N, t o chi, n) over W_K(F) (prop:neborbit (3)).  with_ranks also computes the
    // free ranks of M^S, M^ST (two further Howell forms on the same route) for the H^0 length.
    NebWkCokernel(const level::Gamma0LevelN& lvl, uint32_t n_sym, const Fq& F, uint32_t K, const CharacterValues& chi,
                  WkRoute route, bool with_ranks = true);
    const WkArith& ring() const { return W_; }
    WkRoute route() const { return route_; }
    bool parity_zero() const { return parity_zero_; }
    uint32_t D() const { return D_; }
    uint32_t K() const { return W_.K(); }
    uint32_t orbits_S() const { return orbits_S_; }
    uint32_t orbits_U() const { return orbits_U_; }
    bool has_ranks() const { return with_ranks_; }
    uint32_t rank_S() const { return rank_S_; }
    uint32_t rank_U() const { return rank_U_; }
    // The Howell form of Rel: rows in ascending pivot order.
    const std::vector<WkRow>& rows() const { return rows_; }
    const std::vector<uint32_t>& pivots() const { return piv_; }
    const std::vector<uint32_t>& pivot_vals() const { return pval_; }
    uint64_t rel_length() const;
    uint64_t h1_length() const { return parity_zero_ ? 0 : static_cast<uint64_t>(W_.K()) * D_ - rel_length(); }
    int64_t h0_length() const;                           // needs has_ranks()
    // H^1 = (+) W/ell^e, e ascending in [1, K] (e = K: free summands).  Its size is the minimal number of
    // generators, = h1 over F_q.
    const std::vector<uint32_t>& h1_exponents() const { return h1_exp_; }
    uint32_t h1_free_rank() const;
    // Self-check of the Howell conditions: echelon, pivot = ell^v exactly, canonical residues above each pivot,
    // and every saturation ell^{K-v} row reduces to 0 by the rows below it.  Empty string = all held.
    std::string HowellDefects() const;
    const WkStats& stats() const { return stats_; }

private:
    WkArith W_;
    WkRoute route_;
    bool with_ranks_ = true;
    uint32_t D_ = 0, orbits_S_ = 0, orbits_U_ = 0, rank_S_ = 0, rank_U_ = 0;
    bool parity_zero_ = false;
    std::vector<WkRow> rows_;
    std::vector<uint32_t> piv_, pval_, h1_exp_;
    WkStats stats_;
};

// A Howell form of the span of `gens` in W^D on a route (the engine of NebWkCokernel, exposed for the
// validation's synthetic cases): rows ascending by pivot, canonical.
void WkHowellForm(const WkArith& W, uint32_t D, std::vector<WkRow> gens, WkRoute route, std::vector<WkRow>& rows,
                  std::vector<uint32_t>& piv, std::vector<uint32_t>& pval, WkStats* stats = nullptr);
// The exponents of W^D / span(rows) for a Howell form (ascending, in [1, K]).
std::vector<uint32_t> WkCokernelExponents(const WkArith& W, uint32_t D, const std::vector<WkRow>& rows,
                                          const std::vector<uint32_t>& piv, const std::vector<uint32_t>& pval);

} // namespace neb
} // namespace hecke

#endif // HECKE_NEBENTYPUS_GALOIS_H
