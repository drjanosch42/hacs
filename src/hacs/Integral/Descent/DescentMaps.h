// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_DESCENT_DESCENT_MAPS_H
#define INTEGRAL_DESCENT_DESCENT_MAPS_H

// ===========================================================================
// DESCENT milestone D2 (hecke-extensions slot 24; docs/implementation/slot24/DESCENT_plan.md v0.3 Sec. 4 "D2").  [C] =
// docs/latex/gl2_contract.tex (v1.23.0-pre.23; pre.19 for the clause C-D3, pre.22 / pre.23 for its item (2)(g) and (2)(b)'s signs,
// refereed C-D3g_referee.md), cited by label: prop:descentmaps (clause C-D3), prop:descentN,
// prop:descentNthree, prop:badprimesN, prop:certifiede, prop:certifiedeell, prop:levelorbit.
//
// WHAT IS COMPUTED.  For the level pair A = Gamma_0(N) >= B = Gamma_0(MN), trivial character, V = Sym^n Z^2, over Z:
//   * the end groups H^q(A), H^q(B) and the relative groups H^q(A, B) := H^{q-1}(Cone(alpha)) ([C, prop:descentmaps](2)(b)),
//     q = 0 .. 5;
//   * the maps res^q = H^q(Gamma; alpha), cor^q = H^q(Gamma; omega) ([C, prop:descentmaps](1)(d)) and the two other maps of the
//     long sequence (2)(b), j^q : H^q(B) -> H^{q+1}(A, B) (c' |-> (0, c')) and dl^q : H^q(A, B) -> H^q(A) ((c, c') |-> c; through
//     pi this is MINUS the connecting map delta of 0 -> M_A -> M_B -> Q -> 0, [C, prop:descentmaps](2)(b) "Signs", D2 audit F8), each
//     reported as a map of finitely generated abelian groups: the types of its KERNEL, IMAGE and COKERNEL;
//   * the exactness of the glued long sequence at every spot H^0(A), H^0(B), H^1(A, B), H^1(A), ... up to H^{q_top}(A), each
//     spot checked AS SUBGROUPS (below), and the fibre identities of (1)(d) on cohomology: cor o res = [A : B]; for a normal pair
//     c_sigma o res = res, res o cor = sum_sigma c_sigma, and the five-term statement of (4)(b) in degree 1 (n > 0: res is an
//     isomorphism onto H^1(B)^G; n = 0: the cokernel of H^1(A) -> H^1(B)^G is killed by |G| and has order dividing |G^ab|).
// Glued by [C, prop:descentmaps](1)(c), (2)(c): a map of finitely generated abelian groups is determined by its rational part and
// its ell-local parts, so kernel / image / cokernel are the free rank plus the ell-primary parts read one prime at a time.
//
// THE LOCAL PIECES (one per prime of the finite set S below; each a separate computation, nothing shared between primes):
//   (2)  ell = 2: the DENSE Iso^(2) reduced complex of [C, prop:certifiedeell](5) for M_A and M_B (the D0 / D1 code,
//        DescentAssembly's SelectBlockJR / ClauseMatrixReduced), turned into J-coordinates on both sides (the block
//        G_x = P_x[R_x, J_x] inverted mod 2^K), the fibre maps alpha, omega, R_sigma applied BLOCK BY BLOCK (prop:descentmaps(1)(b):
//        f P_x = P'_x f, then pi_R' and G'^{-1}), the cone Cone^j = C_A^{j+1} + C_B^j, D(c, c') = (-dc, dc' + alpha c)
//        ((2)(a)), all mod 2^30, cochains to degree q_top (4; 6 in the periodicity check), H^5 := H^3 ((2)(b), [C, prop:descentN](5)).
//        (D5) By default the reduced complex is the TREE model's, Iso^(2)_T ([C, prop:isomodeltree](2)(a), (5); BuildRedTree /
//        ReduceFibreTree in DescentMaps.cpp: the same J-coordinates and block-by-block fibre maps); HECKE_DESCENT_TREE=0 is the
//        dense Iso^(2) rollback (value-preserving), which inherits D1's memory wall.  PredictPairRss (dense) or
//        PredictPairRssTree (tree) is checked BEFORE anything is built and the piece REFUSES above 0.9 x the ceiling.
//   (3)  ell = 3: if nu_3(N) = 0 (so nu_3(MN) = 0) the shortcut of [C, prop:descentNthree] and (2)(f): the cokernel route below
//        at ell = 3, H^q(A, B) (x) Z_(3) = 0 for q >= 3; else dense Iso^(3), as at 2, cochains to degree 6.
//   (5+) ell >= 5 in S: the COKERNEL route ((2)(f), [C, prop:descentN](7)(b)): H^0 = ker [rho(S) - 1; rho(U) - 1] on M,
//        H^1 = coker G (G: D1's generators of M^S + M^U, [C, prop:levelorbit](1)), H^1(A, B) = H^0(Gamma; Q) = 0,
//        H^2(A, B) = H^1(Gamma; Q) = coker [G_B | alpha], everything else 0; the maps alpha, omega, R_sigma on M and the identity
//        of M_B for j^1.  Dense matrices of size D, mod ell^K (ell^K < 2^31).
//   S = {2, 3} cup B(n, N) cup B(n, MN) cup {ell >= 5 : ell | [A : B]} ([C, prop:badprimesN](2) and (2)(e): at ell >= 5 outside S
//   every group is ell-torsion-free and s_ell(Q) = s_ell(M_B) - s_ell(M_A) = 0, so no kernel or cokernel has ell-torsion).
//   n odd: only (2) (2 kills everything, [C, prop:badprimesN](2)).
//
// THE LINEAR ALGEBRA.  Everything is over Z/ell^k with explicit transforms: a minimal-valuation Smith form L A R = Sigma that
// keeps L, L^{-1}, R, R^{-1} on request (SmithForm).  A cohomology group is a PRESENTATION: torsion generators u_t (columns of
// L^{-1} for the Smith form of d^{q-1}) with exponents sigma_t, free generators (a kernel basis of d^q on the complement), and
// a coordinate matrix (rows of L, of R^{-1}); an induced map is the matrix F of the cochain map in these coordinates; kernel,
// image and cokernel come from two more Smith forms of small matrices ([F | Delta_Y], a kernel lattice L_f = {x : F x in
// im Delta_Y}, and L_f / im Delta_X).  EVERY Smith form is CERTIFIED as in [C, prop:certifiede](1)-(3) / prop:certifiedeell(6):
// its number of pivots of valuation < k must equal a rank_Q known from theory -- the Betti numbers b_0, b_1 of
// [C, prop:levelorbit](3) (b_q = 0, q >= 2), the Betti recursion on the cone ((2)(d)(ii), beta_j = b_j(B) - b_j(A)), and the
// rational ranks of the maps (res, cor: b_q(A), j: beta_q, dl: 0, c_sigma: b_q(B)), which follow from cor o res = [A : B];
// otherwise the result is REFUSED.
// THE PRECISION RULE, [C, prop:descentmaps](2)(g) (v1.23.0-pre.22 / pre.23, from the D2 code audit's F1; C-D3g_referee.md): if
// L A R = Sigma over Z/ell^k is CERTIFIED (s pivots u_i ell^sigma_i = rank_Q A, all of valuation < k), sigma_max the largest pivot
// valuation, then an exact Smith form over Z_(ell) exists whose transforms and their inverses agree with L, R, L^{-1}, R^{-1} modulo
// ell^{k - sigma_max}, and ("moreover", (2)(g)(i)) the first s rows of L, the last r - s columns of L^{-1} and the last c - s rows of
// R^{-1} even modulo ell^k; so kernel bases, generators and free coordinates are exact mod ell^{k - sigma_max}, torsion coordinates
// for every sigma_i < k ((2)(g)(ii)), and LatticeQuotient's coordinates (L y)_i / ell^sigma_i mod ell^{k - sigma_i} and its
// full-precision containment test (L y)_{>s} = 0 are exact ((2)(g)(ii), "lattice coordinates").  A matrix formed from them is again
// the reduction of an exact one, so the rule composes with additively decreasing precision ((2)(g)(iii)).  Without the
// certification it fails.  The code tracks the precision through every step (BuildPres refuses at k <= 2 sigma_max -- a
// conservative policy, sufficient and not necessary, (2)(g)(ii); InducedMap, LatticeQuotient at k <= sigma_max + 1) and refuses
// when a needed exponent reaches it.
// THE EXACTNESS CHECK at a spot X (g : W -> X, h : X -> Y): im g = ker h as subgroups of X, tested as the lattice quotient
// L_h / (F_g Z^{g_W} + Delta_X) = 0 (with the inclusion checked entrywise, which is h o g = 0) -- not only equal types.
//
// Not built (DESCENT2_report.md Sec. 6): the Hecke squares (3), (3)(b'), (3)(c) of pre.19 (C-D4's level-N operators): this code
// computes and asserts NO Hecke compatibility of res, cor or the connecting map; the two-row read-out (4)(c) for
// Gamma_1(N) < Gamma_0(N); nebentypus.  (The tree models (5): D5, the default route.)
// ===========================================================================

#include <cstdint>
#include <string>
#include <vector>

#include "DescentAssembly.h"
#include "../../../ModArithmetic/ModCtx.h"

namespace integral_descent {
namespace maps {

// ---- Z/ell^k with ell^k < 2^31 ----------------------------------------------------------------------------------
// (S24 CL-G3, CODEAUDIT C-07) A thin wrapper over the engine's ring ModCtx<uint32_t>::chain(ell, k): the Barrett
// reduction, the valuation, the unit inverse and the table of ell^v are ctx's.  It keeps its public fields (m, two,
// mask: read by the descent) and the ell = 2 mask path (exact mod 2^32, then masked).  Every value is a canonical
// residue, so nothing moved (golden R1; IntegralDescent*, HeckeIsotropyModel*).
struct ZMod {
    uint32_t ell = 2, k = 1;
    uint64_t m = 2;
    bool two = true;
    uint32_t mask = 1;
    engine::ModCtx<uint32_t> ctx{2u};
    ZMod() = default;
    ZMod(uint32_t ell, uint32_t k);
    uint32_t red(uint64_t x) const { return two ? static_cast<uint32_t>(x & mask) : ctx.reduce(x); }
    uint32_t mul(uint32_t a, uint32_t b) const { return two ? ((a * b) & mask) : ctx.reduce(uint64_t(a) * b); }
    uint32_t add(uint32_t a, uint32_t b) const { return two ? ((a + b) & mask) : ctx.reduce(uint64_t(a) + b); }
    uint32_t sub(uint32_t a, uint32_t b) const {
        return two ? ((a - b) & mask) : (a >= b ? a - b : static_cast<uint32_t>(uint64_t(a) + m - b));
    }
    uint32_t val(uint32_t x) const;          // k for 0
    uint32_t inv(uint32_t u) const;          // u a unit
    uint32_t pw(uint32_t v) const;           // ell^v (v <= k; 0 = ell^k)
};

struct Mat {
    uint32_t r = 0, c = 0;
    std::vector<uint32_t> a;
    Mat() = default;
    Mat(uint32_t r_, uint32_t c_) : r(r_), c(c_), a(size_t(r_) * c_, 0) {}
    uint32_t& at(uint32_t i, uint32_t j) { return a[size_t(i) * c + j]; }
    uint32_t at(uint32_t i, uint32_t j) const { return a[size_t(i) * c + j]; }
    uint32_t* row(uint32_t i) { return a.data() + size_t(i) * c; }
    const uint32_t* row(uint32_t i) const { return a.data() + size_t(i) * c; }
    uint64_t bytes() const { return uint64_t(a.size()) * 4; }
};

// ---- Smith form with transforms ---------------------------------------------------------------------------------
struct Smith {
    std::vector<uint32_t> sigma;   // pivot valuations; pivot t at (t, t) of L A R
    Mat L, LinvT;                  // r x r: L A R = Sigma; LinvT's row t = column t of L^{-1}   (if left)
    Mat RT, Rinv;                  // c x c: RT's row t = column t of R; Rinv = R^{-1}             (if right)
    uint32_t rank() const { return static_cast<uint32_t>(sigma.size()); }
    uint32_t smax() const;
};
Smith SmithForm(const ZMod& Z, Mat A, bool left, bool right);

// ---- presentations of cohomology groups and induced maps ------------------------------------------------------------
struct Pres {
    ZMod Z;                        // precision of gens / coord
    uint32_t c = 0;                // cochain rank
    std::vector<uint32_t> tors;    // exponents (>= 1) of the torsion generators 0 .. t-1
    uint32_t free = 0;             // free generators t .. t+free-1
    Mat gensT;                     // g x c: row i = a cocycle representing generator i
    Mat coord;                     // g x c: the class of a cocycle z is coord z (torsion rows mod ell^a)
    bool ok = true;
    std::string why;
    uint32_t g() const { return static_cast<uint32_t>(tors.size()) + free; }
    AbelianGroupType type() const;
};
// H at a degree with cochain rank c, previous differential P (c x a, rational rank rankP) and next differential Nx (c' x c),
// b = the free rank (the Betti number); Nx is used only when b > 0 (rank_Q Nx = c - rankP - b).
Pres BuildPres(const ZMod& Z, const Mat& P, uint32_t rankP, const Mat& Nx, uint32_t c, uint32_t b);
// (PAR-D2) the same, consuming P (the factorisation runs in its storage; one c x a matrix less at the peak).
Pres BuildPres(const ZMod& Z, Mat&& P, uint32_t rankP, const Mat& Nx, uint32_t c, uint32_t b);
Pres ZeroPres(const ZMod& Z, uint32_t c);

struct MapRes {
    ZMod Z;                        // precision of F
    Mat F;                         // g_Y x g_X
    uint32_t rankF = 0;            // rational rank (input)
    AbelianGroupType ker, im, coker;
    Mat Lf;                        // g_X x kL: generators of {x : F x in im Delta_Y}
    uint32_t rankL = 0;
    ZMod ZL;                       // precision of Lf
    bool ok = true;
    std::string why;
    bool precision = false;        // (ICENSUS-P1) !ok by a precision shortfall (a rank not certified, a precision exhausted) -- a
                                   // wider ring may lift it; false with !ok: a failed safeguard (relations, I in L, the rank input)
};
// f: the cochain map (Y.c x X.c) inducing the map on cohomology; rankF its rational rank on cohomology.
MapRes InducedMap(const Pres& X, const Pres& Y, const Mat& f, uint32_t rankF);
// (PAR-ASM) the same for the identity-like cochain map f = sum_{t < w} e_{out_off + t} e_{in_off + t}^T (Y.c x X.c), without forming
// f: entrywise InducedMap(X, Y, f, rankF).  Lever HECKE_DESCENT_EMBED_MAP=0: the dense f is formed and InducedMap is called.
// (PAR-ASM audit F5) yc x xc is the cochain map's EXPECTED shape (the cone's / the piece's ranks, as the replaced code sized f):
// both routes ENGINE_CHECK Y.c == yc and X.c == xc, i.e. that the presentations' cochain ranks are the cone's.
MapRes InducedMapEmbed(const Pres& X, const Pres& Y, uint32_t out_off, uint32_t in_off, uint32_t w, uint32_t rankF, uint32_t yc,
                       uint32_t xc);
// im g = ker h at X (as subgroups); false + why otherwise.
bool ExactAt(const Pres& X, const MapRes& g, const MapRes& h, std::string& why);
// F_left F_right - scalar * I, or - F_minus: each column lies in im Delta_Y (the relations of Y).
bool IdentityOnCohomology(const Pres& Y, const Mat& F, std::string& why);

// ---- the pair ----------------------------------------------------------------------------------------------------
struct MapTypes {
    AbelianGroupType ker, im, coker;
};
struct PieceReport {
    uint32_t ell = 0;
    std::string route;             // "Iso2", "Iso3", "cokernel"
    uint32_t K = 0, q_top = 0;
    uint32_t exact_spots = 0, identity_checks = 0, smith_forms = 0;
    uint64_t rss_pred = 0, rss_meas = 0;   // rss_meas: the sampled peak process RSS during the piece (absolute)
    uint64_t rss_base = 0;                 // the process RSS when the piece started (retained heap of earlier work)
    uint32_t min_prec = 0;         // the smallest precision left at any step
    double seconds = 0;
    bool ok = true;
    std::string why;
};
struct PairOptions {
    uint32_t q_top_two = 4;        // Iso^(2) cochains to this degree (6: the periodicity check, q = 5 computed)
    uint64_t rss_cap_bytes = 40ull << 30;
    bool verify_ends_with_d1 = true;   // the end groups against AssembleDescentN (D1)
    // VALIDATION ONLY (negative controls): perturb alpha (one entry) in every piece; claim a wrong rank of res; raise the
    // cap check with a tiny ceiling.  Any of them must REFUSE.
    bool test_break_alpha = false;
    int32_t test_rank_offset = 0;
    // (D2 audit F5) an offset on the claimed rational rank of the invariants map x |-> (c_sigma x - x)_sigma on H^1(B) (normal
    // pairs; true rank b_1(B) - b_1(A), admissible claims up to b_1(B)): +1 is an admissible but FALSE claim, so it reaches the
    // Smith certification of [F | Delta_Y] (r_K < the claimed rank), not the free-rank guard.  (The +1 on res^1 cannot: its true
    // rank b_1(A) is already min(b_1(A), b_1(B)); dl^q and c_sigma are at the guard's bound too.)
    int32_t test_inv_rank_offset = 0;
};
struct PairResult {
    uint32_t N = 1, MN = 1, n = 0, index = 0;
    bool normal = false;
    uint32_t q_max = 5;                    // groups and res / cor for q = 0 .. 5
    uint32_t q_les = 4;                    // the sequence is checked through H^{q_les}(A)
    std::vector<AbelianGroupType> HA, HB, HAB;   // q = 0 .. 5
    std::vector<MapTypes> res, cor;        // q = 0 .. 5 (q = 5: periodicity at 2 when q_top_two = 4)
    std::vector<MapTypes> j, dl;           // j: q = 0 .. q_les - 1; dl: q = 0 .. q_les
    std::vector<PieceReport> pieces;
    std::vector<uint32_t> primes;          // S
    uint32_t exact_spots = 0, identity_checks = 0;
    bool five_term_checked = false;
    std::string five_term;                 // text of the (4)(b) degree-1 statement checked
    double t_total = 0;
    uint64_t rss_pred = 0;
    bool refused = false;
    std::string refusal;
};
// Predicted peak RSS (bytes) of the dense pair piece at ell (2 or 3) with D_A = psi(N)(n+1), D_B = psi(MN)(n+1): the calibrated
// factor times the structural count below, plus 256 MiB.
uint64_t PredictPairRss(uint32_t ell, uint32_t DA, uint32_t DB, uint32_t q_top);
// The structural count alone (bytes): d_A, d_B, the cone's D^j, the reduced fibre maps, three c x c of the largest Smith form, the
// dense fibre maps and the rho memo.
uint64_t PairRssStructural(uint32_t ell, uint32_t DA, uint32_t DB, uint32_t q_top);
// (D5) the same on the tree route ([C, prop:isomodeltree](5); the default unless HECKE_DESCENT_TREE=0): the tree's reduced ranks,
// no dense rho memo; PredictPairRssTree = the calibrated factor times it plus 256 MiB (DESCENT5_report.md).
uint64_t PairRssStructuralTree(uint32_t ell, uint32_t DA, uint32_t DB, uint32_t q_top);
uint64_t PredictPairRssTree(uint32_t ell, uint32_t DA, uint32_t DB, uint32_t q_top);
PairResult AssemblePairMaps(uint32_t N, uint32_t M, uint32_t n, const PairOptions& opt = {});

}  // namespace maps
}  // namespace integral_descent

#endif
