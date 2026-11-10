// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_LEVEL_PAIR_H
#define HECKE_LEVEL_PAIR_H
// ===========================================================================
// (hecke-extensions slot 22, LES-1) LEVEL PAIRS B <= A <= Gamma: THE FIBRE MAPS OF THE LEVEL-CHANGE CONE.
// Binding text: [C] v1.22.0-pre.3 sec:nebles (LES-C, refereed F-LES, repairs R1-R6) -- def:levelpair, lem:nebcompat,
// def:levelres (eq:levelres), lem:levelresmorph, def:normalpair, prop:levelinfres (1), prop:modeltorsion,
// prop:gamma0normal, prop:hidanormalsteps, rmk:normalN1, obl:levelconeexact (i), (ix).  Spec:
// docs/implementation/slot22/LES0_spec.md v0.1 (package LES-1); predictions LES_predictions.md section 1 and
// LES1_predictions.md (committed before the gate ran).
//
// WHAT IS BUILT (the fibre layer only; the cone and its six-term sequence are LES-2, the tree-complex route LES-3).
//  * CongruenceLevel(M, K) : CosetLevel -- the preimage in Gamma of a subgroup K <= SL_2(Z/M), classes K \ SL_2(Z/M)
//    (Gamma -> SL_2(Z/M) is onto).  SL_2(Z/M) is enumerated by BFS on S, T (|SL_2(Z/M)| <= 2 * 10^4), K is the closure
//    of its generators.  Principal(M) = Gamma(M); Commutator(M) = the preimage of [SL_2(Z/M), SL_2(Z/M)]: Gamma^2 at
//    M = 2 (A_3 in S_3), Gamma^3 at M = 3 (Q_8 in SL_2(F_3)), rmk:normalN1.  Used ONLY for levels not of Gamma_H type
//    (rmk:nebcosetlevel: no second Gamma_H class).  N() is M (the congruence modulus); the character datum of
//    def:levelpair is carried separately (CharacterDatum), N = 1 for these levels.
//  * CosetPermutation / TorsionContent / ModelTagOf -- the right action of S, T on A \ Gamma; -I in A, and the fixed
//    classes of S (order 4), U^2 (order 3), U (order 6), U = ST: an element of finite order lies in A iff a conjugate
//    of S^{+-1}, U^{+-1}, U^{+-2} does, i.e. iff the permutation has a fixed class.  ModelTagOf is prop:modeltorsion:
//    `group` iff every finite subgroup of A has order prime to ell (ell = 2: -I not in A; ell = 3: no order-3 element).
//  * QuotientOf(B, A) -- B <= A (the fibre i -> a(i) is S, T-equivariant with a(1) = 1), the B-cosets inside A, and
//    normality by Gamma-set automorphisms: B v_i is fixed by B (v_i normalises B) iff B x -> B v_i x is a well-defined
//    automorphism of the Gamma-set B \ Gamma, i.e. iff phi(1) = i, phi(x s) = phi(x) s (s = S, T) is consistent.  B is
//    normal in A iff this holds at every i with a(i) = 1; the automorphisms then give the multiplication table of
//    G = A / B (mult[i][j] = the class of v_i v_j).  Independent of the Python oracle (generator conjugation in SL_2(Z/L)).
//  * CharacterDatum / TwistedMonomialMatrix -- chi on G_N as a table over the coefficient ring; M_{A,chi}(gamma) of
//    eq:MHchi on ANY CosetLevel (block j: chi(a_{h_j}) rho_n(h_j), h_j = w_j gamma w_{j.gamma}^{-1}).
//  * RestrictionFibreMap -- alpha_{B/A} and the trace omega_{B/A} of eq:levelres for any compatible pair of CosetLevels
//    (alpha: block (i, a(i)) = chi(a_{h_i}) rho_n(h_i), h_i = v_i w_{a(i)}^{-1}; omega: block (a(i), i) =
//    chi(a_{h_i})^{-1} rho_n(h_i)^{-1}).  Refuses by name: les_not_contained (B not in A), les_character_level (A not in
//    Gamma_0(N_A) or B not in Gamma_0(N_B)), les_incompatible_character (chi(a_g) != chi'(a_g) on a Schreier generator g of
//    B, lem:nebcompat).  At chi = 1 on Gamma_0 pairs it is the shipped LevelChangeMap entry for entry (lem:levelresmorph (4)).
//  * ConjugationFibreMap -- R_sigma of def:normalpair: (R f)_J = chi(a_g) rho_n(g) f_{J'}, B sigma^{-1} v_J = B v_{J'},
//    g = v_J v_{J'}^{-1} (= sigma h, h in B).  Refuses by name les_not_normal (sigma not in A, or sigma not normalising B).
//  * FibreMatrix -- a block-sparse matrix of d x d blocks (one non-zero block per block row for the monomial maps), with
//    composition, sums and entrywise comparison: the data of obl:levelconeexact (i).
//
// Coefficients: any engine::ModCtx<uint32_t> (F_ell, ell = 2, 3 included; the fibre statements hold over every ring).
// Every product of coset representatives is formed in __int128 (Mat22W) and reduced modulo ell * N_chi, so the block
// rho_n(h) mod ell and the character argument a_h mod N_chi are exact; membership is tested exactly (contains_wide).
// Nothing here is reached from a shipped record, report, campaign job id, setting or golden text.
// ===========================================================================

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ExactInt.h"
#include "HeckeLevelEll.h"
#include "../../ModArithmetic/ModCtx.h"

namespace hecke {
namespace level {
namespace les {

// A refusal of sec:nebles's box, by name (les_not_contained, les_character_level, les_incompatible_character,
// les_not_normal, les_congruence_too_large).
class LevelPairRefusal : public std::runtime_error {
public:
    LevelPairRefusal(std::string code, const std::string& what)
        : std::runtime_error("level-pair refusal " + code + ": " + what), code_(std::move(code)) {}
    const std::string& code() const { return code_; }
private:
    std::string code_;
};

// ---- CongruenceLevel(M, K) ----------------------------------------------------------------------------------
class CongruenceLevel final : public CosetLevel {
public:
    // M >= 1, |SL_2(Z/M)| <= kMaxGroup; K = the subgroup of SL_2(Z/M) generated by the residues of `K_generators`
    // (each of determinant 1 mod M).  Refuses les_congruence_too_large past the cap.
    CongruenceLevel(uint64_t M, const std::vector<Mat22>& K_generators, std::string name);
    static CongruenceLevel Principal(uint64_t M);                 // Gamma(M)
    static CongruenceLevel Commutator(uint64_t M);                // preimage of [SL_2(Z/M), SL_2(Z/M)] (|SL_2(Z/M)| <= 2000)
    static constexpr uint64_t kMaxGroup = 20000;

    uint64_t N() const override { return M_; }
    uint32_t m() const override { return static_cast<uint32_t>(reps_.size()); }
    Mat22 rep(uint32_t a) const override;                         // 1-based, w_1 = I
    uint32_t class_of(const Mat22& gamma) const override;         // gamma in Gamma
    bool contains(const Mat22& gamma) const override;
    bool contains_wide(const Mat22W& gamma) const override;
    std::string name() const override { return name_; }

    uint64_t group_order() const { return elems_.size(); }        // |SL_2(Z/M)|
    uint64_t K_order() const { return K_order_; }

private:
    uint64_t key(uint64_t a, uint64_t b, uint64_t c, uint64_t d) const { return ((a * M_ + b) * M_ + c) * M_ + d; }
    uint32_t class_of_key(uint64_t k) const;
    uint64_t M_ = 1, K_order_ = 1;
    std::string name_;
    std::vector<uint64_t> elems_;                                  // packed residues, BFS order (elems_[0] = I)
    std::vector<Mat22> lifts_;                                     // a lift in SL_2(Z) of each
    std::unordered_map<uint64_t, uint32_t> index_;                 // packed residue -> position in elems_
    std::vector<uint32_t> cls_;                                    // position -> class (1-based)
    std::vector<Mat22> reps_;                                      // class - 1 -> representative
};

// ---- the Gamma-set A \ Gamma --------------------------------------------------------------------------------
struct CosetPermutation {
    uint32_t m = 0;
    std::vector<uint32_t> S, T;                                   // 1-based: S[i] = class_of(w_i S)
    uint32_t U(uint32_t i) const { return T[S[i]]; }              // i . (ST)
    bool minus_I_in = false;                                      // -I in A (class_of(-I) == 1)
};
CosetPermutation PermutationOf(const CosetLevel& A);

struct TorsionContent {
    bool minus_I = false;      // -I in A
    bool order3 = false;       // an element of order 3 (a fixed class of U^2)
    bool order4 = false;       // an element of order 4 (a fixed class of S)
    bool order6 = false;       // an element of order 6 (a fixed class of U)
    uint32_t fixed_S = 0, fixed_U2 = 0;
    bool torsion_free() const { return !minus_I && !order3 && !order4 && !order6; }
};
TorsionContent TorsionOf(const CosetPermutation& P);
enum class ModelTag : uint8_t { kGroup = 0, kBredon = 1 };
// prop:modeltorsion: kGroup iff every finite subgroup of A has order prime to ell.
ModelTag ModelTagOf(const TorsionContent& t, uint64_t ell);
const char* ModelTagName(ModelTag t);

// phi(1) = c, phi(x s) = phi(x) s for s in {S, T}: true iff consistent (then phi, 1-based, is the automorphism
// B x -> B v_c x of the Gamma-set).  `phi` is a reusable buffer.
bool GammaSetAutomorphism(const CosetPermutation& P, uint32_t c, std::vector<uint32_t>& phi);

struct QuotientGroup {
    bool contained = false;                // B <= A
    bool normal = false;                   // B normal in A (valid only if contained)
    uint32_t index = 0;                    // [A : B]
    std::vector<uint32_t> fibre;           // 1-based over B \ Gamma: a(i)
    std::vector<uint32_t> classes;         // the i with a(i) = 1, ascending (classes[0] = 1): the cosets B v_i in A
    uint32_t witness = 0;                  // not normal: a class i with a(i) = 1 whose v_i does not normalise B
    std::vector<uint32_t> mult;            // normal: index x index, positions in `classes` (class of v_i v_j)
    uint32_t pos_of(uint32_t cls) const;   // position of a class in `classes` (ENGINE_CHECK)
    uint32_t order_of(uint32_t pos) const; // normal only
    bool abelian() const;
    bool cyclic() const;
};
// (pB, pA: PermutationOf(B), PermutationOf(A)).  When B is not normal in A the multiplication table is empty.
QuotientGroup QuotientOf(const CosetLevel& B, const CosetPermutation& pB, const CosetLevel& A, const CosetPermutation& pA);

// [Gamma : A cap B] = the size of the Gamma-orbit of (1, 1) in A \ Gamma x B \ Gamma.
uint64_t IntersectionIndex(const CosetPermutation& pA, const CosetPermutation& pB);

// ---- characters and the twisted monomial action ------------------------------------------------------------
struct CharacterDatum {
    uint64_t N = 1;                        // the level N_A (A must lie in Gamma_0(N))
    std::vector<uint32_t> table;           // size N, values in [0, p), 0 at the non-units
    std::string label = "1";
    static CharacterDatum Trivial(uint32_t p);
    uint32_t value(uint64_t a_mod_N) const;     // chi(a), ENGINE_CHECK a unit with a non-zero value
};

// rho_n blocks of residue matrices over one ring, memoised (RhoCache); shareable across the builders below so that a
// family of maps on one cell computes each block once.  Not thread-safe (one per thread).
class RhoBlocks {
public:
    RhoBlocks(const engine::ModCtx<uint32_t>& ctx, uint32_t n_sym) : p_(ctx.p), n_(n_sym), cache_(ctx, n_sym) {}
    // The block rho_n(g mod p) (a copy: the store may grow).
    std::vector<uint32_t> rho(const Mat22& g);
    uint32_t p() const { return p_; }
    uint32_t n() const { return n_; }
private:
    uint32_t p_, n_;
    RhoCache cache_;
    std::vector<Block> store_;
};

class FibreMatrix {
public:
    FibreMatrix() = default;
    FibreMatrix(uint32_t m_dst, uint32_t m_src, uint32_t d, uint32_t p);
    static FibreMatrix Identity(uint32_t m, uint32_t d, uint32_t p, uint32_t scalar = 1);
    uint32_t m_dst() const { return m_dst_; }
    uint32_t m_src() const { return m_src_; }
    uint32_t d() const { return d_; }
    uint32_t p() const { return p_; }
    // Adds s * blk at block (i, j) (1-based), blk d x d row-major with entries in [0, p).
    void add(uint32_t i, uint32_t j, const std::vector<uint32_t>& blk, uint32_t s = 1);
    // Row i (1-based): (j, block) sorted by j, zero blocks dropped after canonical().
    const std::vector<std::pair<uint32_t, std::vector<uint32_t>>>& row(uint32_t i) const { return rows_[i - 1]; }
    void canonical();                                   // sort, merge equal j, drop zero blocks
    size_t num_blocks() const;
    bool monomial() const;                              // at most one block per row
    FibreMatrix operator*(const FibreMatrix& rhs) const;      // (this o rhs)
    FibreMatrix operator+(const FibreMatrix& rhs) const;
    FibreMatrix scaled(uint32_t s) const;
    // Entries in which the two (same-shape) matrices differ; UINT64_MAX on a shape mismatch.
    uint64_t differing_entries(const FibreMatrix& rhs) const;
    std::vector<uint32_t> dense() const;                // (m_dst d) x (m_src d) row-major

private:
    uint32_t m_dst_ = 0, m_src_ = 0, d_ = 1, p_ = 2;
    bool canon_ = true;                                 // rows sorted, merged, no zero block
    std::vector<std::vector<std::pair<uint32_t, std::vector<uint32_t>>>> rows_;
};

// M_{A,chi}(gamma) of eq:MHchi: block (j, j.gamma) = chi(a_{h_j}) rho_n(h_j), h_j = w_j gamma w_{j.gamma}^{-1}, gamma in
// SL_2(Z).  ENGINE_CHECKs h_j in A and A in Gamma_0(chi.N) at h_j.
FibreMatrix TwistedMonomialMatrix(const CosetLevel& A, const CharacterDatum& chi, const Mat22& gamma, uint32_t n_sym,
                                  const engine::ModCtx<uint32_t>& ctx, RhoBlocks* blocks = nullptr);

// ---- the restriction and the trace (def:levelres, eq:levelres) -----------------------------------------------
// VALIDATION ONLY (the negative controls of obl:levelconeexact (ix)); no shipped caller sets either.
struct RestrictionOptions {
    bool neg_inverse_scalar = false;       // chi(a_{h_i}) replaced by chi(a_{h_i})^{-1} in alpha
    bool neg_identity_block = false;       // rho_n(h_i) replaced by I in alpha
};

class RestrictionFibreMap {
public:
    // (A, chi) >= (B, chi') a level pair (def:levelpair).  Refuses by name (LevelPairRefusal) as in the header note.
    RestrictionFibreMap(const CosetLevel& A, const CharacterDatum& chiA, const CosetLevel& B, const CharacterDatum& chiB,
                        uint32_t n_sym, const engine::ModCtx<uint32_t>& ctx, RestrictionOptions opts = {},
                        RhoBlocks* blocks = nullptr);
    const FibreMatrix& alpha() const { return alpha_; }           // m_B x m_A blocks
    const FibreMatrix& trace() const { return omega_; }           // m_A x m_B blocks
    uint32_t index() const { return index_; }                      // [A : B]
    const std::vector<uint32_t>& fibre() const { return fibre_; } // 1-based: a(i)

private:
    uint32_t index_ = 0, schreier_ = 0;
    std::vector<uint32_t> fibre_;
    FibreMatrix alpha_, omega_;
};

class ConjugationFibreMap {
public:
    // sigma in A normalising B; R_sigma on Ind_B (B's character chi' enters through h in B, which equals chi there).
    // pB = PermutationOf(B).  Refuses les_not_normal.  `skip_normaliser_check`: VALIDATION ONLY (the negative control
    // builds the formula for a non-normalising sigma to show that it is not Gamma-equivariant).
    ConjugationFibreMap(const CosetLevel& A, const CharacterDatum& chiA, const CosetLevel& B, const CosetPermutation& pB,
                        const Mat22& sigma, uint32_t n_sym, const engine::ModCtx<uint32_t>& ctx,
                        bool skip_normaliser_check = false, RhoBlocks* blocks = nullptr);
    const FibreMatrix& matrix() const { return R_; }

private:
    FibreMatrix R_;
};

// ---- (slot 23, G-EVT23) the LES-1 level readings --------------------------------------------------------------------------
// Moved verbatim from HeckeLevelNormalityValidation.cpp (arms N1, N2, N5) so that the validation and the golden harness's
// `LESLEVEL` point read the identical loops (value-neutral; the validation's checks are now made on these rows).
struct LevelHandle {
    std::unique_ptr<CosetLevel> lvl;
    CosetPermutation perm;
    const CosetLevel& operator*() const { return *lvl; }
};
LevelHandle MakeLevel(std::unique_ptr<CosetLevel> l);
LevelHandle Gamma0Handle(uint64_t N);
// Gamma_H(N), H = { u in G_N : pred(u) } (ENGINE_CHECK: the predicate set is a subgroup).
LevelHandle GammaHHandle(uint64_t N, const std::function<bool(uint64_t)>& pred);
LevelHandle Gamma1Handle(uint64_t N);
LevelHandle Gamma1Cap0Handle(uint64_t Np, uint64_t L);    // Gamma_1(N') cap Gamma_0(L), N' | L
// value(i) for i in Q.classes, from the representative v_i of B v_i in A: bijective onto Z/M and additive for the
// multiplication table (so the coset map is an isomorphism G = A / B -> Z/M).
bool QuotientMapIso(const QuotientGroup& Q, const CosetLevel& B, uint64_t M, const std::function<uint64_t(const Mat22&)>& value);
// (N1) [C] prop:gamma0normal: Gamma_0(MN) in Gamma_0(N) for N = 1 .. 60, M = 2 .. 24 (N-major, M ascending; 1380 rows).
// iso = the quotient is cyclic and a c / N mod M is an isomorphism onto Z/M (computed only when normal).
struct Gamma0NormalRow {
    uint64_t N = 0, M = 0;
    bool contained = false, normal = false, iso = false;
};
std::vector<Gamma0NormalRow> Gamma0NormalityTable();
// (N2) prop:hidanormalsteps: N' = 2 .. 24, M | N', M >= 2 (60 rows); a, b, c = the three statements of the arm.
struct Gamma1StepRow {
    uint64_t Np = 0, M = 0;
    bool a = false, b = false, c = false;
};
std::vector<Gamma1StepRow> Gamma1StepsTable();
// (N5) prop:modeltorsion: the eight levels of the arm's table, in its order (Gamma, Gamma_0(2), Gamma_0(4), Gamma_0(3),
// Gamma_0(9), Gamma_1(4), Gamma^2, Gamma^3), their torsion content and model tags at ell = 2, 3, 5.
struct ModelTagRow {
    std::string name;
    TorsionContent t;
    ModelTag t2 = ModelTag::kGroup, t3 = ModelTag::kGroup, t5 = ModelTag::kGroup;
};
std::vector<ModelTagRow> ModelTagTable();

} // namespace les
} // namespace level
} // namespace hecke

#endif // HECKE_LEVEL_PAIR_H
