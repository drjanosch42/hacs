// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_LVALUE_MBLOCK_READER_H
#define HECKE_EXT_LVALUE_MBLOCK_READER_H

// ===========================================================================
// (hecke-extensions slot 20, WP-LV3) THE GENERIC m-BLOCK READER of the L-value code ([E] sec:lvalue,
// rmk:lvshapes (H) / (B); slot-20 plan, WP-LV3 (a)).  It REPLACES WP-BIL's quartic-specific reader (hidapair::EigenFour on
// R19-1's frozen factors P_1, P_2 of (258, 37)) by one that needs no frozen data:
//
//   given a Hecke block t = (T_{p*} - lambda_0)|_{V_m} over Z/ell^M (m = (ell, T_{p*} - lambda_0), V_m the generalised
//   eigenspace, read as ker t^{4M}), its characteristic polynomial P (monic, P == x^r mod ell), the reader
//     (1) splits the roots of P over Z_ell by the Newton polygon (nf::newton_polygon, the NumberField core) and the
//         residual polynomials of its segments,
//     (2) fixes, per (segment, residual root y), the local order O = Z_ell[pi]/(pi^e - ell u) of the tame totally
//         ramified extension the roots of that cluster live in (e = the segment's denominator, u = y^{h'} with
//         h h' == 1 mod e; e = 1: O = Z_ell, the split case),
//     (3) refines every root digit by digit inside O (Newton polygons in v_pi of the Taylor shifts P(rho_0 + x)), so a
//         product of local orders is read root by root, to the working precision,
//     (4) returns per root rho the primitive eigen-line g_rho = Q_rho(t) e_j / pi^{content} (Q_rho = P / (x - rho)) of the
//         block, v_pi P'(rho), the content, and the eigen-check v_pi(t g - rho g).
//
// SCOPE (refused by name, hecke::NotImplementedInThisBuild, never a silent wrong value): a residual polynomial with an
// irreducible factor of degree > 1 over F_ell (an unramified residue extension f > 1); a segment denominator e with
// e not dividing ell - 1 (the e conjugates are not all in O); a refinement step whose slope is not integral in v_pi (the
// root needs a further ramified extension); a multiple root to the working precision.  None of the gate cells needs
// more: (258, 37) is e = 2 twice (R19-1's two Eisenstein quadratics, recovered here, not read), the m_Eis blocks at
// depth 1 and 2 are split (e = 1).
//
// The cell reading ReadMBlockCell() is WP-BIL's ReadCell258() made generic in (n, ell, lambda_0): the m-blocks of W
// = H^1_c(V) and of H^1(D) in both geometric signs, their Grams, the per-root mixed pairing values, P9 through
// H^1_!(D)_m (opt-in), and the congruence module read from A_c alone; plus R20-3's change of lattice (below).
//
// MP CONSTRAINT: the reader takes a block matrix and a polynomial, nothing of the 7-vertex model; the cell reading
// takes a hidapair::HidaPairingCell (the PSL_2(Z) amalgam presentation) and is the only place that knows it.
// Host C++, single-threaded, no file, no env, no golden object.
// ===========================================================================

#include "HidaPairing.h"

#include <cstdint>
#include <string>
#include <vector>

namespace hecke_ext {
namespace lvalue {
namespace mblock {

using hidapair::Mat;
using hidapair::Vec;
using hidapair::Zq;

// ---- O = (Z/ell^M)[pi] / (pi^e - ell u), u a unit: elements sum_i a_i pi^i, i < e -------------------------------------
class LocalOrder {
public:
    using Elt = std::vector<uint64_t>;
    LocalOrder() = default;
    LocalOrder(const Zq& R, uint32_t e, uint64_t u);
    const Zq& R() const { return R_; }
    uint32_t e() const { return e_; }
    uint64_t u() const { return u_; }
    uint32_t cap() const { return e_ * R_.m; }   // v_pi of 0 (the working precision, in pi-digits)
    Elt zero() const { return Elt(e_, 0); }
    Elt scalar(uint64_t a) const;
    Elt pi_pow(uint32_t s) const;
    Elt add(const Elt& x, const Elt& y) const;
    Elt sub(const Elt& x, const Elt& y) const;
    Elt mul(const Elt& x, const Elt& y) const;
    uint32_t vpi(const Elt& x) const;            // min_i (e v_ell(a_i) + i), capped at cap()
    uint64_t residue(const Elt& x) const;        // x mod pi, in F_ell
    Elt div_pi_pow(const Elt& x, uint32_t s) const;   // x / pi^s for v_pi(x) >= s (s digits of precision are lost)
    std::string to_string() const;               // "Z_37" | "Z_37[pi]/(pi^2 - 37*5)"

private:
    Zq R_;
    uint32_t e_ = 1;
    uint64_t u_ = 1;
    uint64_t ellu_ = 0;                           // ell * u mod q
};

struct Root {
    uint32_t order = 0;                 // index into RootReading::orders
    LocalOrder::Elt rho;                // the root of P (an eigenvalue of t = T - lambda_0) in O
    uint32_t vpi_dP = 0;                // v_pi P'(rho)
    std::string name;                   // "rho[s/e,y,zeta]"
};
struct RootReading {
    bool complete = false;              // every root found (else `refusal`)
    std::string refusal;
    std::vector<LocalOrder> orders;
    std::vector<Root> roots;
    uint32_t newton_segments = 0;       // of the top-level polygon (nf::newton_polygon)
};
// P: monic, low -> high, over R, P == x^r mod ell.  Never throws on a refusal: complete = false with the reason.
RootReading RootsOfNilpotentCharpoly(const Zq& R, const Vec& P);

struct EigenLineO {
    std::vector<LocalOrder::Elt> g;     // primitive generator of the rho-line, block coordinates
    uint32_t content = 0;               // v_pi content of the best cofactor column (Q_rho(t) e_j)
    uint32_t eigen_check = 0;           // v_pi(t g - rho g)
};
// t: the block matrix (column convention, entries over R), P its charpoly, rho a root in O.
EigenLineO EigenLineAt(const LocalOrder& O, const Mat& t, const Vec& P, const LocalOrder::Elt& rho);
LocalOrder::Elt EvalPoly(const LocalOrder& O, const Vec& P, const LocalOrder::Elt& x);

// The m-block of A (generalised eigenvalue lambda_0): a saturated basis of ker (A - lambda_0)^{4M} and t = (A - lambda_0)|.
struct Block {
    std::vector<Vec> basis;             // ambient coordinates
    Mat t;                              // r x r, column convention
    Vec charpoly;                       // of t, low -> high
};
Block BlockOf(const Zq& R, const Mat& A, uint64_t lambda0);
// The joint m-block: the lambda_0-block of A, then successively the lambda_p-blocks of the other (commuting) operators inside it;
// t and the charpoly are those of A - lambda_0 on the joint block.
Block JointBlockOf(const Zq& R, const Mat& A, uint64_t lambda0, const std::vector<std::pair<const Mat*, uint64_t>>& others);
Mat RestrictTo(const Zq& R, const Mat& A, const std::vector<Vec>& basis);   // A|_basis (A-stable to precision; ENGINE_CHECK)

// ---- the cell reading -----------------------------------------------------------------------------------------------------
struct MBlockForm {
    std::string name;
    std::string order;                  // O, as text
    uint32_t e = 1;
    uint64_t rho_residue_mod_ell = 0;   // the root's pi-adic leading digit (diagnostic)
    uint32_t vpi_dP = 0;
    uint32_t content_c = 0, content_D = 0;
    uint32_t cong = 0;                  // v_pi P'(rho) - content_c: the congruence module of W_m at rho (R19-9 Sec. 3 lemma)
    uint32_t mixed = 0;                 // v_pi <g_rho, h_rho> on W_m x H^1(D)_m
    bool have_p9 = false;
    uint32_t p9 = 0;                    // v_pi <g_rho, h_rho^!> on W_m x H^1_!(D)_m
    uint32_t eigen_check = 0;
    bool eisenstein = false;            // rho == e_2 - lambda_0 to the precision (T_2 g = (1 + 2^{k-1}) g): the boundary line
    // [E] def:lvlattice (1.20.0-pre.2): c_rho^eps = length_O( H^1(D)/tors[rho] / theta_*(H^1(V)/tors[rho]) ), theta = Phi_n of [C] lem:symdual
    // (Gram (-1)^t t! (n-t)!; HidaPairing::theta is Phi_n / n!), in v_pi units (>= 0); valid if have_lattice.  Computed as
    // e v_ell(n!) - content_pi(Theta(h_rho)), Theta = (Phi_n / n!)^{-1} : D -> V integral (u_t^v -> (-1)^t C(n,t) u_{n-t}).
    int32_t lattice_change = 0;
    uint32_t d_lattice = 0;             // d_f (R20-3): the f-line index of H^1(V') -> H^1(V), V' = (+) C(n,t) Z_ell u_t = Theta(D); c_f = e v(n!) - d_f
    bool have_lattice = false;
    LocalOrder::Elt rho;
    uint32_t order_index = 0;
    LocalOrder O;                       // the order rho lives in (working precision ell^{M-2})
};
struct MBlockSign {
    std::string name;                   // geo-even / geo-odd (the W sign)
    bool complete = false;
    std::string refusal;
    uint32_t rank_c = 0, rank_D = 0;
    Vec charpoly_c, charpoly_D;
    uint32_t tr1 = 0, tr2 = 0, tr3 = 0, dr1 = 0, dr2 = 0, dr3 = 0;
    std::vector<uint32_t> gram_pivots;
    bool gram_unimodular = false;
    uint32_t newton_segments = 0;
    std::vector<MBlockForm> forms;
    // the congruence kind read on W_m: length of W_m / sum_p (A_c(T_p) - e_p) W_m, e_p = 1 + p^{k-1}, p in primes \ {ell}
    uint32_t cong_length = 0;
    // P9
    bool have_p9 = false;
    uint32_t p9_rank_WD = 0;
    bool p9_intertwines = false;
    Vec p9_charpoly;
    std::vector<uint32_t> p9_index_pivots;
};
struct MBlockOptions {
    bool p9 = false;                    // H^1_!(D)_m (WP-BIL's P9)
    bool hd_torsion = true;             // the torsion of H^1(D) mod ell^4
    bool lattice_change = false;        // R20-3's constant per root (theta(H^1(V)) vs H^1(D))
    std::vector<uint32_t> primes{2, 3, 5};
    // further generators of m beyond T_2 - lambda_0: (p, lambda_p) (each p among the cell's primes).  Empty: m = (ell, T_2 - lambda_0),
    // WP-BIL's (258, 37) convention; m_Eis: EisensteinJoint() = {(3, 1 + 3^{k-1}), (5, 1 + 5^{k-1})}.
    std::vector<std::pair<uint32_t, uint64_t>> joint;
};
std::vector<std::pair<uint32_t, uint64_t>> EisensteinJoint(const Zq& R, uint32_t n, const std::vector<uint32_t>& primes = {3, 5});
struct MBlockReading {
    uint32_t n = 0;
    uint64_t ell = 0;
    uint32_t M = 0;
    uint64_t lambda0 = 0;
    uint32_t ne = 0, no = 0;
    bool adj2 = false, adj3 = false, adj5 = false, iota = false;
    uint32_t HD_free_rank = 0;
    std::vector<uint32_t> HD_torsion;
    // (opt.lattice_change) Smith valuations of Theta = (Phi_n / n!)^{-1} : H^1(D)/tors -> H^1(V)/tors on the whole lattice (both signs),
    // and the whole-lattice length of [E] def:lvlattice: v_ell[H^1(D)/tors : Phi_n,*(H^1(V)/tors)] = rank * v_ell(n!) - sum(pivots).
    std::vector<uint32_t> theta_index_pivots;
    uint32_t v_nfact = 0;               // v_ell(n!)
    int32_t phi_index = 0;
    MBlockSign sign[2];                 // [0] geo-even (W^{even} contains p_0), [1] geo-odd
    double seconds = 0;
};
MBlockReading ReadMBlockCell(const hidapair::HidaPairingCell& C, uint64_t lambda0, const MBlockOptions& opt);
MBlockReading ReadMBlockCell(uint32_t n, uint64_t ell, uint32_t M, uint64_t lambda0, const MBlockOptions& opt);

// e_2 = 1 + 2^{k-1} mod ell^M (the m_Eis residue of T_2), k = n + 2.
uint64_t EisensteinT2(const Zq& R, uint32_t n);
// The lambda_0 a domain's subquotient names: "T_2-<int>" / "T_2 - <int>" -> that integer, else m_Eis (e_2).
uint64_t Lambda0FromSubquotient(const std::string& subquotient, const Zq& R, uint32_t n);

} // namespace mblock
} // namespace lvalue
} // namespace hecke_ext

#endif // HECKE_EXT_LVALUE_MBLOCK_READER_H
