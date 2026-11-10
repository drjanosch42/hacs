// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_MODEL_PERIODICITY_CLASS_H
#define INTEGRAL_MODEL_PERIODICITY_CLASS_H

// ===========================================================================
// Line 2, L2-1b (2) (hecke-extensions slot 10, WP-13): the class
// e in H^2(Gamma; Z) of [C, lem:bulkperiodic](1) and the periodicity map
// E_q = e u - of [C, lem:bulkperiodic](2) on the isotropy-resolution model
// R = Iso_. of [C, def:isomodel] (IsotropyModel, slot-9 WP-10).  [C] =
// docs/latex/gl2_contract.tex, cited by label.
//
// THE CLASS e (a 2-cocycle in Hom_{Z[Gamma]}(R_2, Z), i.e. one integer per
// free generator of R_2 -- NOT a functional on the coinvariants
// R (x)_{Z[Gamma]} Z, which compute homology):
//
//   * chi : Gamma -> Z/12 is the homomorphism with chi(S) = 3, chi(ST) = 2
//     (in twelfths of Q/Z); on every finite-order h it equals the zeta_+
//     character of [C, lem:bulkperiodic](1) (Chi12 vs ZetaPlusTwelfths below,
//     checked on the stabilisers, their powers and conjugates).
//   * Phi in Hom_{Z[Gamma]}(R_1, Z) is an integer lift of 12 * (the Q/Z-valued
//     1-cocycle of chi), obtained by composing with the chain map R -> (bar
//     resolution) e_{s,sigma,0} |-> [], e_{0,sigma,1} |-> [g_sigma],
//     e_{1,E,0} |-> sum sign [g_face]; so Phi(e_{0,sigma,1}) = chi(g_sigma) and
//     Phi(e_{1,E,0}) = sum sign chi(g_face).
//   * e := (Phi o d_2) / 12, the Bockstein of chi for 0 -> Z -> Q -> Q/Z; the
//     division is exact (ENGINE_CHECKed).  By naturality of the Bockstein
//     kappa_C(res_C e) = chi|_C = chi_C for every finite cyclic C, which is
//     [C, lem:bulkperiodic](1)'s characterisation; on the column of sigma in
//     Sigma_0 it reads e(e_{0,sigma,2}) = m_sigma * chi(g_sigma) / 12 mod
//     m_sigma.  12 e = delta^1(Phi) exactly, so [e] has order dividing 12.
//   Sign convention: the cochain differential is f |-> f o d (the one
//   IsotropyModel::coboundary uses), and the Bockstein is phi~ o d with it.
//   A convention with the opposite sign gives -e; every periodicity statement
//   below is insensitive to that sign.
//
// THE MAP E (a chain map E : R_{.+2} -> R_. over Z[Gamma] with
// aug o E_0 = e, i.e. the Yoneda representative of e; f |-> f o E_q
// represents e u - on H^q(Gamma; V) for any V):
//
//   * E_0(e_c) = e(e_c) * e_{0,v0,0};
//   * E_q(e_c) for q >= 1 is a filling y of the cycle z = E_{q-1}(d e_c) in
//     R_{q-1}, d y = z, computed by fill() below: the "staircase" of the
//     filtration by s (the proof of [C, prop:isomodel](2)), with the
//     Z-linear contracting homotopies of the periodic columns (as in
//     IsotropyModel) and, at the two places where the columns do not reach,
//     (i) a path in the 1-skeleton (degree 0: paths p_sigma inside K and the
//     chains c_gamma, d c_gamma = (gamma - 1) e_{0,v0,0}, from a word of gamma
//     in S, T, -I), (ii) the UNIQUE 2-chain of C_2(Xbulk) bounding a 1-cycle
//     (Xbulk is contractible and 2-dimensional, so d_2 is injective): found
//     over F_p on a growing candidate set of triangles and certified exactly
//     over Z;
//   * built for q = 0 .. q_top - 2 (= 4 at q_max = 5), so that E_q^* is a
//     cochain map C^q -> C^{q+2} for q <= 3 and H^1 -> H^3, H^2 -> H^4,
//     H^3 -> H^5 are defined; degree 6 cohomology is never read.
//   certify() re-derives: e a cocycle; 12 e = Phi o d_2; the restriction
//   values against zeta_+; aug o E_0 = e; d_q E_q = E_{q-1} d_{q+2} on every
//   generator of R_{q+2}, q = 1 .. 4, exactly over Z[Gamma].
//
// THE CONNECTING MAP of the Mayer-Vietoris sequence of
// Gamma = <S> *_{<-I>} <U>, U = ST (for the kernel clause of
// [C, lem:bulkperiodic](2) at q = 1): the Bass-Serre tree T (vertices
// Gamma/<S> u Gamma/<U>, edges Gamma/<-I>, d[gamma] = gamma<U> - gamma<S>),
// a Gamma-chain map phi : C_.(Xbulk) -> C_.(T) (v6 -> <S>, every other vertex
// representative -> <U>, an edge -> the tree geodesic), and for
// v in V^{-I} the 1-cocycle f_v = (edge gamma<-I> |-> rho(gamma) v) o phi_1 o
// eps_1 on R_1: its class is the connecting image partial(v) (up to one global
// sign).  connecting_cochains(ctx, n) returns f_v for v = N_{-I} x_i, i.e. a
// generating set of partial(N_{-I} V).
//
// DEPENDENCIES: IsotropyModel.h, Hecke/ExactInt.h and ModArithmetic/ only (the
// Integral/Model/ rule); IsotropyModel is used through its public API only.
// Exact integers (int64, overflow-checked); no bignum, no __int128.
// ===========================================================================

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "IsotropyModel.h"

namespace integral_model {

// A chain of R_q: one Z[Gamma]-coefficient per free generator (cell) of
// degree q, in IsotropyModel::cells(q) order.
using Chain = std::vector<GroupRingElt>;

// chi(g) in Z/12 (twelfths of Q/Z): the homomorphism Gamma -> Q/Z with
// chi(S) = 3/12, chi(ST) = 2/12 (so chi(T) = -1/12, chi(-I) = 6/12), evaluated
// on a word of g in S, T, -I.
uint32_t Chi12(const hecke::Mat22& g);
// zeta_+(h) as twelfths, for h of finite order, from the closed form of
// [C, lem:bulkperiodic](1) Step 1: zeta_+(h) = t/2 + i sgn(c) sqrt(4 - t^2)/2,
// t = tr h; zeta_+(+-I) = +-1.  ENGINE_CHECKs that h has finite order.
uint32_t ZetaPlusTwelfths(const hecke::Mat22& h);

struct PeriodicityCertificate {
    bool e_cocycle = false;              // e o d_3 = 0
    bool e_bockstein = false;            // 12 e = Phi o d_2 exactly
    bool e_restrictions = false;         // e(e_{0,sigma,2}) = m chi(g_sigma)/12 mod m, every sigma
    bool chi_is_zeta = false;            // Chi12 == ZetaPlusTwelfths on every tested finite-order element
    bool aug_E0 = false;                 // aug o E_0 = e
    bool chain_map = false;              // d_q E_q = E_{q-1} d_{q+2}, q = 1 .. top
    bool tree_chain_map = false;         // d phi_1 = phi_0 d on edges, phi_1 d = 0 on triangles
    uint64_t chain_terms_checked = 0;
    uint64_t finite_order_elements_checked = 0;
    int64_t max_abs_coefficient = 0;     // over every E_q entry
    int64_t max_abs_entry = 0;           // max |matrix entry| of a group element in E
    uint64_t total_terms = 0;            // sum of |support| over every E_q entry
    uint64_t distinct_group_elements = 0;
    std::string first_failure;
    bool all() const {
        return e_cocycle && e_bockstein && e_restrictions && chi_is_zeta && aug_E0 && chain_map &&
               tree_chain_map;
    }
};

class PeriodicityClass {
public:
    // Builds e, Phi and E_0 .. E_{q_top-2}; every filling is checked exactly
    // (ENGINE_CHECK) as it is made.  Keeps a pointer to M (M must outlive it).
    explicit PeriodicityClass(const IsotropyModel& M);

    const IsotropyModel& model() const { return *M_; }
    uint32_t top() const { return static_cast<uint32_t>(E_.size()) - 1; }   // E_q built for q <= top()

    // e(e_c), c in cells(2); Phi(e_c), c in cells(1).
    const std::vector<int64_t>& e() const { return e_; }
    // E_q(e_c) in R_q, c in cells(q + 2).
    const std::vector<Chain>& E(uint32_t q) const { return E_.at(q); }

    PeriodicityCertificate certify() const;

    // E_q^* : C^q_I -> C^{q+2}_I over ctx, f |-> f o E_q: the dense
    // (w_{q+2} d_V) x (w_q d_V) matrix, block (c, b) = sum n rho_n(gamma) over
    // the terms n * gamma of E_q(e_c)'s coefficient at e_b (IsotropyModel's
    // layout: flat index = cell * d_V + component).
    template<class Elt>
    engine::DenseBuf<Elt> matrix(const engine::ModCtx<Elt>& ctx, uint32_t n, uint32_t q) const;

    // The cocycles f_v, v = N_{-I} x_i = x_i + rho_n(-I) x_i (i = 0 .. n), as
    // the columns of a (w_1 d_V) x d_V matrix in C^1_I.
    template<class Elt>
    engine::DenseBuf<Elt> connecting_cochains(const engine::ModCtx<Elt>& ctx, uint32_t n) const;

    // ---- chain-level tools (public for the validation) ---------------------
    Chain zero_chain(uint32_t q) const;
    Chain apply_d(uint32_t q, const Chain& x) const;          // d_q : R_q -> R_{q-1}, q >= 1
    int64_t augmentation(const Chain& x0) const;               // R_0 -> Z
    // y in R_{q+1} with d y = z, for z a cycle of R_q (q = 0: aug z = 0).
    // ENGINE_CHECKs the input condition and d y = z.
    Chain fill(uint32_t q, const Chain& z) const;

    // Negative-control hooks (the validation's (N) arms): add one unit term to
    // E_q(e_c)'s coefficient at e_b, or 1 to e at cell c.
    void perturb_E_for_negative_control(uint32_t q, uint32_t c, uint32_t b);
    void perturb_e_for_negative_control(uint32_t c);

    // Statistics of the construction (for the report).
    uint64_t x_fillings() const { return x_fillings_; }
    uint32_t x_max_rounds() const { return x_max_rounds_; }
    uint32_t x_max_triangles() const { return x_max_triangles_; }

private:
    const IsotropyModel* M_;
    std::vector<int64_t> e_, phi12_;
    std::vector<std::vector<Chain>> E_;
    std::vector<std::vector<hecke::Mat22>> pow0_;             // powers of g_sigma, sigma in Sigma_0
    std::vector<Chain> path_;                                 // p_sigma in R_1: d p = e_sigma - e_v0
    std::map<GKey, Chain> cgen_;                              // c_gamma for gamma in {S, T, T^-1, -I}
    // The tree map on the nine edge representatives: phi_1(E_j) as
    // (canonical edge key -> multiplicity).
    std::vector<std::map<GKey, int64_t>> tree_phi1_;
    bool tree_ok_ = false;
    std::string tree_failure_;
    mutable uint64_t x_fillings_ = 0;
    mutable uint32_t x_max_rounds_ = 0;
    mutable uint32_t x_max_triangles_ = 0;

    void build_paths();
    void build_class();
    void build_E();
    void build_tree();
    Chain c_gamma(const hecke::Mat22& g) const;
    Chain fill0(const Chain& z) const;
    // The unique w in C_2(Xbulk) with d w = x, for x a 1-cycle given as
    // (edge rep, canonical coset key) -> multiplicity.
    std::map<std::pair<uint32_t, GKey>, int64_t> fill_x1(const std::map<std::pair<uint32_t, GKey>, int64_t>& x) const;
};

}  // namespace integral_model

#endif
