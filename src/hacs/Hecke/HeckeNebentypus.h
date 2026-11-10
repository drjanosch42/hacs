// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_NEBENTYPUS_H
#define HECKE_NEBENTYPUS_H
// ===========================================================================
// (hecke-extensions slot 16, WP-10) NEBENTYPUS ON THE FIXED MODEL BY SHAPIRO.
// Binding text: [C, sec:nebentypus] (v1.16.0-pre.2): def:nebdata, lem:nebdata,
// lem:teichmuller, rmk:nebfield, lem:inducedchar (eq:MHchi), lem:gammaonetable,
// def:nebmodel, prop:nebdecomp, prop:nebcorr (eq:nebcorr), thm:nebrelations,
// prop:neborbit, rmk:nebheilbronn, rmk:nebcost, the refusal box, rmk:nebcheckable
// (NB1)-(NB6).  The convention is [GT, conv:nebentypus] and the eigenvalue
// dictionary [GT, bridge:nebentypus-dictionary] (lean/blueprint/src/
// gl2_groundtruth.tex; L16-2, audited F1: GO), implemented EXACTLY as the
// hand-off docs/implementation/slot16/L16-2_nebentypus_handoff.md states it:
//   * chi(a_delta) on EVERY block (a_delta the upper-left entry): eq:MHchi's
//     chi(a_{h_j(gamma)}), eq:nebcorr's chi(a_{delta_{i,t}}); on Gamma_0(N) this
//     is chi^{-1}(d_gamma);
//   * the Heilbronn route of rmk:nebheilbronn: chi(d_{h_j(eps)}) rho_n(adj h_j(eps))
//     (a_{adj h} = d_h);
//   * <d> = chi(d) on the character model; T(p, p) = chi(p) p^n;
//   * a character given by a Conrey label is reduced to F_q by the reduction rule of
//     rmk:nebcheckable (NB5): e(1/o) |-> s, s the LEAST element of exact order
//     o = ord chi in F_q (CanonicalLess order of [E] def:canon; on F_ell: the least
//     integer in [1, ell)), so that psi(g) = s at the named generator g of a Conrey
//     label with chi(g) = e(1/o) (g = 3, 2, 2 at N = 7, 9, 13).  Choosing s^{-1}
//     instead IS the chi <-> chibar swap.
//
// WHAT IS BUILT (validation / env-hook only: no record key, no setting, no verb).
//  * Fq -- F_q = F_ell[Y]/(C) for a caller-supplied monic C of degree d, certified
//    irreducible at construction (hecke::CertifyIrreducible); the validation feeds
//    [E] def:canon's C_{ell,d} (HeckeExt/HeckeCanonicalField -- this file may not
//    include HeckeExt/, the layer's one-way rule).  Elements are uint32 codes
//    sum c_i ell^i.  d = 1: ell < 2^31, direct arithmetic; d >= 2: q <= 2^22, log /
//    exp tables.
//  * GaloisRing -- W_K(F_q) = (Z/ell^K)[Y]/(C~) (lem:teichmuller), C~ the
//    coefficientwise lift, ell^K < 2^62; the Teichmuller lift t(x) = x~^{q^{K-1}}.
//  * DirichletCharacter -- characters chi: (Z/N)^x -> mu_L(C) as exponent tables
//    (chi(u) = e(k_u / L)); Conrey labels N.m (odd p^e: the least g primitive mod
//    p^max(e,2); 2^e: the -1, 5 generators); ReduceCharacter -- the reduction rule
//    above, with the refusals character_field and character_wild of the box after
//    rmk:nebcost.
//  * TwistedMonomialOf / NebCokernel / HeilbronnChi -- THE CHARACTER MODEL AS A
//    COKERNEL over F_q: M = Ind_{Gamma_0(N)}^Gamma V(chi) with eq:MHchi (one block
//    per block row, never a D x D matrix), M^<S>, M^<U> orbit by orbit
//    (prop:neborbit (1)), H^1 = M / (M^S + M^U), H^0 = M^S cap M^U
//    (prop:neborbit (3)), and T_p^{(N,chi)} (p !| N) / U_p^{(N,chi)} (p | N) on it
//    by H_{N,p,chi} (rmk:nebheilbronn).  Dense elimination of the relation span:
//    D = m_N (n + 1) <= kNebDenseCap (the sparse elimination of rmk:levelcost (4)
//    with a character is NOT built -- refused by name `neb_dense_cap`).
//  * InducedRelationsChi / BuildLevelCorrespondenceChi -- THE CHARACTER MODEL ON THE
//    FIXED MODEL (K, R): the plug points of sec:levelN with a scalar per block
//    (InducedRelationsLevel with eq:MHchi; eq:levelcorr with eq:nebcorr's scalar),
//    over any word-sized ModCtx in which the scalar lives (F_ell with ord chi |
//    ell - 1, or Z/ell^K with the Teichmuller-lifted table) -- no new coset or
//    tau-table.  With the scalar identically 1 they are the shipped objects
//    entrywise ((NB2), prop:nebdecomp (4)).
//  * Gamma1LevelN / Gamma1SubTransversal / DiamondOnModel -- the Gamma_1(N) model
//    of lem:gammaonetable on the fixed model and the diamonds <d> = R_{sigma_d} as
//    fibre maps (prop:nebdecomp (1)), the oracle of (NB3).
//  * EulerCharacteristicFormula -- prop:neborbit (4) evaluated exactly in
//    characteristic 0 (exponent arithmetic in Q/Z; no field).
//
// Refusals (by name, NebentypusRefusal::code): nebentypus_small_ell (ell in {2, 3}:
// no cohomology, no cokernel, no dimension formula; cochain level only),
// character_wild (ell | ord chi), character_field (ord chi !| q - 1),
// neb_dense_cap, field_too_large.  NOT a refusal: chi(-1) != (-1)^n gives 0
// (prop:neborbit (2)), reported as `parity_zero`.
//
// MP constraint ([C] box after rmk:nebcost (v)): nothing here hard-wires the
// 7-vertex model (the model routes take any (K, R) through the shipped relation
// generator; the cokernel route reads no model), a trivial centre (-I enters only
// through its scalar chi(-1)(-1)^n, checked on the orbits), or Bredon semantics at
// ell in {2, 3} (refused by name, never deferred).
//
// Nothing here is reached from a shipped record, report, campaign job id, setting
// or golden text: pipeline-free and value-free with respect to every shipped object.
// ===========================================================================

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ExactInt.h"
#include "FpPoly.h"
#include "HeckeLevelEll.h"
#include "HeckeLevelN.h"

namespace hecke {
namespace neb {

// A refusal of sec:nebentypus's box, by name.
class NebentypusRefusal : public std::runtime_error {
public:
    NebentypusRefusal(std::string code, const std::string& what)
        : std::runtime_error("nebentypus refusal " + code + ": " + what), code_(std::move(code)) {}
    const std::string& code() const { return code_; }
private:
    std::string code_;
};

// D = m_N (n + 1) above which the dense cokernel elimination refuses (neb_dense_cap).
constexpr uint32_t kNebDenseCap = 4096;

// ---- F_q ------------------------------------------------------------------------
class Fq {
public:
    using E = uint32_t;
    // F_ell[Y]/(C): ell prime < 2^31, C monic of degree d >= 1, certified irreducible
    // here (throws otherwise); d >= 2 needs q = ell^d <= 2^22 (field_too_large).
    Fq(uint32_t ell, const FpPoly& C);
    static Fq Prime(uint32_t ell);                       // C = Y

    uint32_t ell() const { return ell_; }
    uint32_t degree() const { return d_; }
    uint64_t q() const { return q_; }
    const FpPoly& defining_poly() const { return C_; }
    std::string name() const;                            // "F_ell" or "F_{ell^d}"

    E zero() const { return 0; }
    E one() const { return 1; }
    E from_int(int64_t a) const;                         // the image of Z
    E add(E a, E b) const;
    E sub(E a, E b) const;
    E neg(E a) const;
    E mul(E a, E b) const;
    E inv(E a) const;                                    // throws on 0
    E pow(E a, uint64_t e) const;                        // 0^0 = 1
    E frob(E a) const { return pow(a, ell_); }
    std::vector<uint32_t> digits(E a) const;             // c_0 .. c_{d-1}
    E from_digits(const std::vector<uint32_t>& c) const;
    bool in_prime_field(E a) const { return a < ell_; }
    // CanonicalLess on elements ([E] def:canon, HeckeCanonicalField::less): by the
    // degree of the representative, then lexicographically on c_0, c_1, ...
    bool less(E a, E b) const;
    uint64_t order_of(E a) const;                        // multiplicative order, a != 0
    // The CanonicalLess-least element of exact multiplicative order o (o | q - 1).
    E least_root_of_unity(uint64_t o) const;
    std::string text(E a) const;                         // "c" (d = 1) or "[c0,c1,..]"

private:
    uint32_t ell_ = 0, d_ = 1;
    uint64_t q_ = 0;
    FpPoly C_;
    std::vector<uint32_t> exp_, log_;                    // d >= 2: exp_[i] = g^i, log_[x]
    std::vector<uint64_t> qm1_primes_;                   // prime divisors of q - 1
    E poly_mul(E a, E b) const;                          // schoolbook mod C (table build)
};

// Dense matrices over F_q (row-major codes).
struct FqMat {
    uint32_t rows = 0, cols = 0;
    std::vector<uint32_t> a;
    FqMat() = default;
    FqMat(uint32_t r, uint32_t c) : rows(r), cols(c), a(static_cast<size_t>(r) * c, 0) {}
    uint32_t at(uint32_t i, uint32_t j) const { return a[static_cast<size_t>(i) * cols + j]; }
    uint32_t& at(uint32_t i, uint32_t j) { return a[static_cast<size_t>(i) * cols + j]; }
};
FqMat FqMul(const Fq& F, const FqMat& A, const FqMat& B);
FqMat FqAdd(const Fq& F, const FqMat& A, const FqMat& B);
FqMat FqScale(const Fq& F, const FqMat& A, uint32_t s);
FqMat FqIdentity(uint32_t n);
bool FqEqual(const FqMat& A, const FqMat& B);
// RREF in place; returns the pivot columns.
std::vector<uint32_t> FqRref(const Fq& F, FqMat& A);
uint32_t FqRank(const Fq& F, FqMat A);
// A basis of the column space of A, as the columns of a (rows x r) matrix.
FqMat FqColumnBasis(const Fq& F, const FqMat& A);
// X with A X = B for A of full column rank; throws unless solvable.
FqMat FqSolve(const Fq& F, const FqMat& A, const FqMat& B);
// det(X - A), low -> high, monic (Hessenberg reduction).
std::vector<uint32_t> FqCharpoly(const Fq& F, const FqMat& A);
std::string FqPolyText(const Fq& F, const std::vector<uint32_t>& f);

// ---- W_K(F_q) and the Teichmuller lift (lem:teichmuller) --------------------------
class GaloisRing {
public:
    using W = std::vector<uint64_t>;                     // d coefficients in [0, ell^K)
    GaloisRing(const Fq& F, uint32_t K);                 // ell^K < 2^62
    uint32_t K() const { return K_; }
    uint64_t modulus() const { return mod_; }
    const Fq& field() const { return *F_; }
    W one() const;
    W lift(Fq::E x) const;                               // the coefficientwise lift in [0, ell)
    W add(const W& a, const W& b) const;
    W mul(const W& a, const W& b) const;
    W pow(const W& a, uint64_t e) const;
    W scalar(uint64_t s) const;                          // s mod ell^K, embedded
    Fq::E reduce(const W& a) const;                      // mod ell
    W teichmuller(Fq::E x) const;                        // x != 0: lift(x)^{q^{K-1}}
private:
    const Fq* F_;
    uint32_t K_ = 1, d_ = 1;
    uint64_t mod_ = 0;
    std::vector<uint64_t> C_;                            // the lift of C, monic, degree d
};

// ---- Dirichlet characters --------------------------------------------------------------
struct DirichletCharacter {
    uint64_t N = 1;
    uint64_t conrey = 1;                                 // the Conrey index m (0: not from a label)
    uint64_t L = 1;                                      // chi(u) = e(expo[u] / L)
    std::vector<int64_t> expo;                           // size max(N, 1); -1 at non-units
    bool is_unit(int64_t a) const;
    int64_t exponent_at(int64_t a) const;                // in [0, L); ENGINE_CHECK a unit
    uint64_t order() const;
    int parity() const;                                  // chi(-1) in {+1, -1}
    DirichletCharacter inverse() const;
    DirichletCharacter mul(const DirichletCharacter& o) const;
    bool is_trivial() const { return order() == 1; }
    std::string label() const;                           // "N.m"
    static DirichletCharacter Trivial(uint64_t N);
    static DirichletCharacter Conrey(uint64_t N, uint64_t m);
};
// The phi(N) characters of (Z/N)^x, as the Conrey characters N.m, m in ascending order.
std::vector<DirichletCharacter> AllCharacters(uint64_t N);
// (Z/N)^x as residues in ascending order (N = 1: {0}).
std::vector<uint64_t> UnitsMod(uint64_t N);

// chi reduced into F_q by the reduction rule (header note).
struct CharacterValues {
    uint64_t N = 1;
    uint64_t order = 1;
    Fq::E s = 1;                                         // the image of e(1/order)
    std::vector<uint32_t> val;                           // size max(N, 1); 0 at non-units
    Fq::E at(int64_t a) const;                           // chi(a mod N), ENGINE_CHECK a unit
};
// Refuses character_wild (ell | ord chi) and character_field (ord chi !| q - 1).
CharacterValues ReduceCharacter(const DirichletCharacter& chi, const Fq& F);
// The Teichmuller lift t o psi into W_K(F_q) (lem:teichmuller (3)), as a table.
std::vector<GaloisRing::W> LiftCharacter(const CharacterValues& psi, const GaloisRing& W);

// ---- prop:neborbit (4), exactly in characteristic 0 -------------------------------------
struct EulerPrediction {
    bool parity_ok = true;                               // chi(-1) == (-1)^n
    uint64_t m = 0;
    uint32_t nu2 = 0, nu3 = 0;
    int64_t dim_S = 0, dim_U = 0;                        // dim M^<S>, dim M^<U> (prop:neborbit (1))
    int64_t h1_minus_h0 = 0;
};
EulerPrediction EulerCharacteristicFormula(uint64_t N, uint32_t n_sym, const DirichletCharacter& chi);

// ---- the character model as a cokernel over F_q -----------------------------------------
// Block scalar of a block h (det h > 0, a_h a unit mod N), a code of F (cokernel route)
// or an element of a word-sized ModCtx (model route).
using BlockScalar = std::function<uint32_t(const Mat22&)>;
// h |-> chi(a_h), F-codes.
BlockScalar CharacterScalar(const CharacterValues& chi);
// h |-> table[a_h mod N] (the model route over a ModCtx; the table in the ring).
BlockScalar TableScalar(uint64_t N, std::vector<uint32_t> table);

// (gamma . f)_j = s(h_j(gamma)) rho_n(h_j(gamma)) f_{j.gamma}: eq:MHchi kept monomial.
struct TwistedMonomial {
    uint32_t m = 0, d = 0;
    std::vector<uint32_t> perm;                          // 1-based: perm[j] = j . gamma
    std::vector<uint32_t> scal;                          // 1-based: the block scalar (F code)
    std::vector<uint32_t> blk;                           // 1-based: index into blocks
    std::vector<std::vector<uint32_t>> blocks;           // rho_n(h) mod ell, d x d (F_ell codes)
    std::vector<uint32_t> apply(const Fq& F, const std::vector<uint32_t>& x) const;
};
TwistedMonomial TwistedMonomialOf(const level::CosetLevel& A, const Mat22& gamma, uint32_t n_sym, const Fq& F,
                                  const BlockScalar& s);

class HeilbronnChi;

class NebCokernel {
public:
    // The character model of (N, chi, n) over F (prop:neborbit (3)).  Refuses
    // nebentypus_small_ell (ell < 5) and neb_dense_cap; chi(-1) != (-1)^n builds
    // nothing and reports parity_zero() (prop:neborbit (2)).
    NebCokernel(const level::Gamma0LevelN& lvl, uint32_t n_sym, const Fq& F, const CharacterValues& chi);
    bool parity_zero() const { return parity_zero_; }
    uint32_t D() const { return D_; }
    uint32_t dim_S() const { return dim_S_; }
    uint32_t dim_U() const { return dim_U_; }
    uint32_t dim_rel() const { return static_cast<uint32_t>(pivots_.size()); }
    uint32_t h1() const { return parity_zero_ ? 0 : D_ - dim_rel(); }
    uint32_t h0() const { return parity_zero_ ? 0 : dim_S_ + dim_U_ - dim_rel(); }
    const std::vector<uint32_t>& free_columns() const { return free_; }
    // (slot-17 WP-6) the stored RREF (fully reduced rows, insertion order) and their pivots, for the entrywise
    // comparison with the sparse route (HeckeNebentypusSparse.h).
    const std::vector<std::vector<uint32_t>>& rref_rows() const { return rref_; }
    const std::vector<uint32_t>& pivots() const { return pivots_; }
    uint32_t orbits_S() const { return orbits_S_; }
    uint32_t orbits_U() const { return orbits_U_; }
    std::vector<uint32_t> coords(const std::vector<uint32_t>& x) const;
    bool in_rel(const std::vector<uint32_t>& x) const;
    // [H] on M / Rel in the free-column basis; throws unless H(Rel) in Rel.
    FqMat OnH1(const HeilbronnChi& H) const;
    uint32_t RelEscapes(const HeilbronnChi& H) const;
    const Fq& field() const { return *F_; }

private:
    const Fq* F_;
    uint32_t D_ = 0, dim_S_ = 0, dim_U_ = 0, orbits_S_ = 0, orbits_U_ = 0;
    bool parity_zero_ = false;
    std::vector<std::vector<uint32_t>> rref_;
    std::vector<uint32_t> pivots_, free_;
};

// rmk:nebheilbronn: H_{N,p,chi} = sum_{eps in Mer_p} sum_{j admissible}
// E_{j.eps, j} (x) chi(d_{h_j(eps)}) rho_n(adj h_j(eps)) on M_{N,chi}.
class HeilbronnChi {
public:
    struct Term { uint32_t dst = 0, src = 0, block = 0; uint32_t scal = 0; };
    HeilbronnChi(const level::Gamma0LevelN& lvl, uint64_t p, uint32_t n_sym, const Fq& F, const CharacterValues& chi);
    // T(p, p), p !| N (S16 WP-10/11 repairs, audit finding 8): the double coset Gamma_0(N) pI Gamma_0(N) = pI Gamma_0(N),
    // one Heilbronn matrix pI, the term chi(d_{pI}) rho_n(adj pI) = chi(p) p^n on every block -- T(p, p) = chi(p) p^n
    // (the convention of [GT, conv:nebentypus]: chi(d) on Delta_0(N)).  Built through the same term loop as T_p.
    static HeilbronnChi Central(const level::Gamma0LevelN& lvl, uint64_t p, uint32_t n_sym, const Fq& F, const CharacterValues& chi);
    // (slot-17 WP-6; [C] thm:nebrelations (2), (3)) THE DIRECT DOUBLE COSET of determinant p^nu, nu >= 1: Merel's set
    // Mer_{p^nu} (MerelSet(p^nu), every element, the non-primitive ones such as p I included) through the same term loop.
    // For p !| N every pair is admissible and the operator is the FULL T_{p^nu} (the union of the double cosets of
    // determinant p^nu in Delta_0(N)); for p | N the inadmissible pairs are dropped as for U_p, giving U_{p^nu}.  nu = 1 is
    // the ordinary constructor's operator term for term.  p^nu < 2^20.
    static HeilbronnChi Power(const level::Gamma0LevelN& lvl, uint64_t p, uint32_t nu, uint32_t n_sym, const Fq& F,
                              const CharacterValues& chi);
    uint64_t p() const { return p_; }
    uint32_t nu() const { return nu_; }
    bool bad() const { return N_ % p_ == 0; }
    bool central() const { return central_; }
    std::string label() const {
        if (central_) return "T(" + std::to_string(p_) + "," + std::to_string(p_) + ")";
        const std::string base = std::string(bad() ? "U_" : "T_") + std::to_string(p_);
        return nu_ == 1 ? base : base + "^" + std::to_string(nu_);
    }
    uint32_t D() const { return m_ * d_; }
    std::vector<uint32_t> apply(const std::vector<uint32_t>& x) const;
    const std::vector<Term>& terms() const { return terms_; }
    const std::vector<std::vector<uint32_t>>& blocks() const { return blocks_; }
    uint32_t pairs_dropped() const { return dropped_; }
    uint64_t wide_terms() const { return wide_terms_; }   // terms built on the __int128 route (slot-17 WP-6)
private:
    HeilbronnChi(const level::Gamma0LevelN& lvl, uint64_t p, uint32_t n_sym, const Fq& F, const CharacterValues& chi, bool central,
                 uint32_t nu = 1);
    const Fq* F_;
    uint64_t N_ = 1, p_ = 2;
    uint32_t nu_ = 1;
    bool central_ = false;
    uint32_t m_ = 1, d_ = 1, dropped_ = 0;
    uint64_t wide_terms_ = 0;
    std::vector<Term> terms_;
    std::vector<std::vector<uint32_t>> blocks_;
};

// ---- the character model on the fixed model (K, R) ------------------------------------
// InducedRelationsLevel with eq:MHchi: every block rho_n(h_j(g_r)) times s(h_j(g_r)).
// ctx any word-sized ring holding the scalars.
std::vector<LocalSystemSimplicialComplex::SimplexRelation> InducedRelationsChi(
    const SimplicialComplex& K, const level::CosetLevel& A, uint32_t n_sym, const engine::ModCtx<uint32_t>& ctx,
    const BlockScalar& s);
// eq:nebcorr: eq:levelcorr with every block term scaled by s(delta), delta the block
// matrix v_i s_j^{-1} g gamma w_{a.gamma}^{-1} (= delta_{i,t} (gamma-transported)).
level::BlockCochainMap BuildLevelCorrespondenceChi(const engine::ModCtx<uint32_t>& ctx, const level::CorrespondenceSpec& spec,
                                                   const SimplicialComplex& K, uint32_t n_sym, uint32_t q,
                                                   const BlockScalar& s);

// ---- the Gamma_1(N) model (lem:gammaonetable) ---------------------------------------------
class Gamma1LevelN final : public level::CosetLevel {
public:
    explicit Gamma1LevelN(uint64_t N);                   // N >= 3, N < 2^20
    uint64_t N() const override { return N_; }
    uint32_t m() const override { return static_cast<uint32_t>(lvl0_.m() * units_.size()); }
    Mat22 rep(uint32_t J) const override;                // w_{(j,u)} = sigma_u w_j
    uint32_t class_of(const Mat22& gamma) const override;
    bool contains(const Mat22& gamma) const override;    // gamma in Gamma_1(N)
    // (slot-18 WP-5 (d)) the same test, exactly, in __int128 (det 1, c = 0 and d = 1 mod N), so the coset products of
    // level::SubgroupProductForRho escalate on Gamma_1(N) as on Gamma_0(N) instead of the CosetLevel default's refusal.
    bool contains_wide(const Mat22W& gamma) const override;
    std::string name() const override { return "Gamma_1(" + std::to_string(N_) + ")"; }
    const level::Gamma0LevelN& level0() const { return lvl0_; }
    const std::vector<uint64_t>& units() const { return units_; }
    uint32_t index(uint32_t j, uint64_t u) const;        // J(j, u), 1-based
    Mat22 sigma(uint64_t u) const;                       // sigma_u of lem:gammaonetable
private:
    uint64_t N_;
    level::Gamma0LevelN lvl0_;
    std::vector<uint64_t> units_;
    std::vector<uint32_t> rank_;                         // rank_[u] (0-based) or UINT32_MAX
};
// r_t in Gamma_1(N) representing (Gamma_1(N) cap Gamma_0(N p)) \ Gamma_1(N), p prime:
// [[1, 0], [N t, 1]] (t < p), and for p !| N one [[x, y], [N, d]] with p | d, d = 1 mod N.
std::vector<Mat22> Gamma1SubTransversal(uint64_t N, uint64_t p);
// <d> = R_{sigma_d} on L_{Gamma_1(N)} (prop:nebdecomp (1)): the fibre map
// (R f)_{(j,u)} = rho_n(sigma_u sigma_{u'}^{-1}) f_{(j,u')}, u' = d^{-1} u, at every q-simplex.
level::BlockCochainMap DiamondOnModel(const Gamma1LevelN& G, uint64_t d, const SimplicialComplex& K, uint32_t n_sym,
                                      const engine::ModCtx<uint32_t>& ctx, uint32_t q);

} // namespace neb
} // namespace hecke

#endif // HECKE_NEBENTYPUS_H
