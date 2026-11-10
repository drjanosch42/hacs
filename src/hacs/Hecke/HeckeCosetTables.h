// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_COSET_TABLES_H
#define HECKE_COSET_TABLES_H

// Stage S3 -- Hecke coset tables: the exact coset datum for g = diag(1, N),
// N = p_hecke^r.
//
// Contract: docs/latex/hecke_action.tex ([C]), [C, prop:cosets] (coset
// representatives s_i of Gamma_0(N)\Gamma, delta_i = s_i^{-1} g), the
// contract boxes in [C, sec:heckedatum] (r = 1 representatives) and
// [C, sec:induced] (the bottom-row P^1 class map, cocycles h_i(gamma) of
// [C, prop:blockmatrices]), [C, prop:alpha]/[C, prop:omega] (the rho_n(t_i) /
// rho_n(s_i^{-1}) blocks) and [C, sec:example] (the fully worked p = 2
// tables, reproduced by HeckeCosetTablesSelfCheckValidation's sibling S3V
// oracle).
//
// ===========================================================================
// INDEXING CONVENTION (binding, read before using this class): coset indices
// i range over the CLOSED interval [1, num_cosets()] -- i.e. 1-BASED, exactly
// matching the subscripts s_1, ..., s_m of [C, sec:heckedatum]/[C,
// sec:example]. Index 0 is never a valid coset index and every accessor
// throws std::out_of_range on i == 0 or i > num_cosets(). This choice is
// deliberate: it lets every formula and every printed table in this header,
// in HeckeCosetTables.cpp and in the S3V validation be transcribed verbatim
// from [C] without an off-by-one translation layer.
// ===========================================================================
//
// Right-cosets/right-action convention (binding, [C, prop:cosets]; DO NOT
// silently transpose or swap left/right -- this is the classic silent-bug
// surface for this stage):
//   * Gamma_0(N)\Gamma are RIGHT cosets Gamma_0(N)*gamma, with RIGHT action
//     i . gamma; delta_i := s_i^{-1} * g (s_i on the left, g on the right).
//   * The class map (P1) acts on the BOTTOM ROW of gamma, as a ROW VECTOR
//     multiplied by gamma from the LEFT... concretely: for
//     gamma = [[a,b],[c',d']] and (c:d) the class of Gamma_0(N)*s_i, the
//     class of Gamma_0(N)*s_i*gamma is (c*a + d*c' : c*b + d*d') mod N (this
//     IS the row-vector-times-matrix action (c,d)*gamma; never transpose to
//     a column-vector convention).
//   * The cocycle h_i(gamma) := s_i * gamma * s_{i.gamma}^{-1} lies in
//     Gamma_0(N) (lower-left entry == 0 mod N) by construction
//     ([C, prop:blockmatrices]).
//   * Never invert a det == N matrix (invariant I7 / [C, obl:integral]):
//     only Mat22::inv_unimodular() is ever called, and only on s_i, s_{i.gamma}
//     (both in Gamma = SL(2,Z), det == 1).
//
// Representatives, r = 1 (BINDING, [C, sec:heckedatum] contract box):
//   s_1 := I,  s_{1+k} := [[1,0],[k,1]] (1 <= k <= p-1),  s_{p+1} := S.
//   Class of s_1 is (0:1), of s_{1+k} is (k:1), of s_{p+1} is (1:0).
//   Gamma^0(p) reps (top-row classes, used for rho_t / the alpha blocks of
//   [C, prop:alpha]): t_1 := I, t_{1+k} := [[1,k],[0,1]], t_{p+1} := S.
//
// Representatives, r >= 2 (this stage fixes a single deterministic rule
// that is contract-consistent and specializes to the r = 1 table above
// bit-for-bit -- see HeckeCosetTables.cpp for the worked verification
// against [C, sec:example]):
//   P^1(Z/N), N = p^r, is enumerated as
//     branch A (i = 1, ..., N):        class (k:1), k = i-1, 0 <= k < N;
//     branch B (i = N+1, ..., N+p^{r-1}): class (1:p*c), c = i-N-1,
//                                          0 <= c < p^{r-1}.
//   Both branches have gcd(c,d) == 1 by construction (one coordinate is
//   exactly 1). [C, sec:heckedatum]'s general lift recipe ("lift (c,d) with
//   gcd(c,d,N) == 1 to gamma = [[x,y],[c',d']] in SL_2(Z) via extended gcd,
//   x*d'-y*c' = 1") has NON-UNIQUE Bezout solutions whenever gcd(c',d') == 1
//   is witnessed trivially (e.g. c' = d' = 1): a generic extended-Euclid
//   implementation can legally return a different, equally valid solution
//   from the one the contract prints (verified during development: the
//   textbook iterative extended gcd on (d0=1, c0=1) returns (x,y) = (0,-1),
//   giving s_2 = [[0,-1],[1,1]] for p = 2 -- det 1, correct class, but NOT
//   the contract's s_2 = [[1,0],[1,1]]). To get the exact, bit-identical
//   contract representatives (required for O16 / S3V) this stage therefore
//   uses the closed-form solutions directly, rather than a generic Bezout
//   solver:
//     s_i = [[1,0],[k,1]]      (branch A, k = i-1),
//     s_i = [[0,-1],[1,p*c]]   (branch B, c = i-N-1),
//   each independently verified det == 1 and bottom row == its assigned
//   class (asserted at construction), and specializing EXACTLY to the r = 1
//   table above (branch B, c = 0, gives s_{p+1} = [[0,-1],[1,0]] = S).
//
//   Gamma^0(N) reps t_i (used only for rho_t / the alpha cross-check block;
//   [C] pins only the r = 1 case above -- this is THIS STAGE's own
//   deterministic generalization to r >= 2, chosen to reduce to the r = 1
//   table exactly): t_i = [[1,k],[0,1]] (branch A, k = i-1), t_i =
//   [[-p*c,-1],[1,0]] (branch B, c = i-N-1). At r = 1, c = 0, this gives
//   t_{p+1} = [[0,-1],[1,0]] = S, matching [C] exactly.
//
// All entries are exact signed int64 (Mat22, overflow-checked); mod ell and
// mod N reductions are the only place values are canonicalized to [0, N)
// (invariant I6, early reduction) -- the class map (P1) itself is computed
// entirely mod N via ModArithmetic::mod_mul/mod_add/mod_inv (works for any
// modulus via the extended Euclidean algorithm, not just prime N; correct
// for N = p^r, r >= 2).
//
// Placement (documented per the task's instructions): this class depends on
// FlatMatrix / ComputeSymmetricPowerMatrixFM (ModArithmetic), so -- exactly
// like Stage S2's SL2ZTessellationTables -- it is compiled into the
// cohomology_app executable sources (NOT into the pure-integer HeckeCore
// static library), while still linking HeckeCore for hecke::Mat22.

#include <cstdint>
#include <vector>
#include <array>

#include "ExactInt.h"
#include "../../ModArithmetic/ModCtx.h"
#include "../../ModArithmetic/FlatMatrix.h"
#include "../../ModArithmetic/LegacyConfig.h"   // OC_ENABLE_LEGACY (S24 CL-R1)

namespace hecke {

// (hecke-extensions slot 14, WP-10) A rho_n block of the coset tables, read
// through the *_block() accessors: BORROWED from the stored table (no copy) on
// a default table, OWNED (freshly evaluated by the table's rho_of, the very
// function that would have filled the table) on a table constructed under
// HeckeOperatorOptions::ScopedCosetStreaming.  Hold the handle while the
// reference from get() is in use.
template<class Elt>
class CosetRhoT {
public:
    explicit CosetRhoT(const FlatMatrixT<Elt>* borrowed) : p_(borrowed) {}
    explicit CosetRhoT(FlatMatrixT<Elt>&& owned) : own_(std::move(owned)), p_(nullptr) {}
    const FlatMatrixT<Elt>& get() const { return p_ != nullptr ? *p_ : own_; }
    // (slot-14 WP-16, audit M-4) the block by value: MOVES the owned matrix out of a
    // streamed handle (no copy), copies a borrowed one.  The handle is spent afterwards.
    FlatMatrixT<Elt> take() && { return p_ != nullptr ? *p_ : std::move(own_); }
private:
    FlatMatrixT<Elt> own_;
    const FlatMatrixT<Elt>* p_ = nullptr;
};

template<class Elt>
class HeckeCosetTablesT {
public:
    // N = p_hecke^r, r >= 1. ell = coefficient prime; n_sym = n. Builds and
    // fully certifies (via the constructor-time internal consistency
    // asserts documented in the .cpp; call self_check() for the full DoD
    // battery) the canonical representative set of [C, sec:heckedatum].
    HeckeCosetTablesT(uint64_t p_hecke, uint32_t r, Elt ell, uint32_t n_sym);

    // Hook for O12(b) (alternate coset representatives s_i -> h_i s_i,
    // h_i in Gamma_0(N); [C, thm:welldef](a) guarantees the operator is
    // unchanged). Builds the canonical representatives first (needed for the
    // class map and the Gamma^0(N)/t_i data, which never change), then
    // requires custom_s.size() == num_cosets() with custom_s[i-1] a det == 1
    // representative of the SAME Gamma_0(N)-class as the canonical s_i
    // (checked via class_index_of; throws std::invalid_argument otherwise),
    // and rebuilds s_inv/delta/cocycle/rho_cocycle/rho_s_inv from custom_s.
    HeckeCosetTablesT(uint64_t p_hecke, uint32_t r, Elt ell, uint32_t n_sym,
                      const std::vector<Mat22>& custom_s);

    uint64_t N() const { return N_; }                 // p^r
    uint32_t num_cosets() const { return m_; }         // m = p^r + p^{r-1}
    uint64_t p_hecke() const { return p_; }            // the Hecke prime p (WP-4a, additive)
    uint32_t r() const { return r_; }                  // the exponent nu with N = p^nu

    const Mat22& g() const { return g_; }              // diag(1, N)
    const Mat22& s(uint32_t i) const;                  // s_i in Gamma; s_1 = I
    const Mat22& s_inv(uint32_t i) const;
    const Mat22& delta(uint32_t i) const;               // delta_i = s_i^{-1} g

    // (P1): class of Gamma_0(N)*gamma = bottom row (c:d) in P^1(Z/N).
    uint32_t class_index(uint32_t c_mod_N, uint32_t d_mod_N) const;
    uint32_t class_index_of(const Mat22& gamma) const;

    // The (T1) split-and-conjugate step shared by EVERY twisted chain datum
    // (hecke-extensions WP-4a: the bulk tau of HeckeTauTable and the boundary
    // tau_bd of HeckeBoundaryDatum route through this ONE function, so that
    // construction and certification of either table cannot disagree about
    // how a Gamma''-derived slot is obtained -- [C, alg:tau] step 0,
    // [C, obl:taubdry](ii),(iii)). Writes gamma = h * s_j with
    // j = class_index_of(gamma) and h = gamma * s_j^{-1} in Gamma_0(N) ((P1),
    // verified: det h == 1 and N | h.c), and returns the exact integral
    // conjugate g^{-1} h g = [[h.a, N h.b],[h.c / N, h.d]] for g = diag(1, N)
    // -- integral BECAUSE N | h.c; unimodular, verified. Throws
    // std::runtime_error on any violated invariant (an internal (P1) error).
    Mat22 twisted_conjugate(const Mat22& gamma, uint32_t& class_out) const;

    // Generator tables, gen in {0:T, 1:S, 2:ST, 3:TS} (the S2 dictionary
    // order; generator matrices are Mat22::T()/S()/ST()/TS(), never
    // re-typed literals).
    uint32_t perm(uint32_t gen, uint32_t i) const;               // i . gamma
    const Mat22& cocycle(uint32_t gen, uint32_t i) const;        // h_i(gamma) in Gamma_0(N)
#if OC_ENABLE_LEGACY
    // The reference accessors of the rho_n tables REFUSE a streamed table
    // (ENGINE_CHECK): use the *_block() accessors below, which serve both.
    // (S24 CL-R1) ON builds only: in an OFF build every table is streamed
    // (legacy/HeckeCosetTablesMaterialised.inc).
    const FlatMatrixT<Elt>& rho_cocycle(uint32_t gen, uint32_t i) const; // rho_n(h_i(gamma)) mod ell
    const FlatMatrixT<Elt>& rho_s_inv(uint32_t i) const;                // omega blocks rho_n(s_i^{-1})
    const FlatMatrixT<Elt>& rho_t(uint32_t i) const;                    // alpha blocks rho_n(t_i)
#endif

    // (hecke-extensions slot 14, WP-10) The coset streaming lever
    // (HeckeOperatorOptions::ScopedCosetStreaming, read at construction):
    // true iff this table stores no rho_n table (S24 CL-R1: always, in an
    // OC_ENABLE_LEGACY=OFF build; the lever is retired).  Each *_block() accessor
    // borrows the stored block on a default table and evaluates rho_of(...)
    // of the stored exact matrix on a streamed one -- the same bytes.
    bool streamed() const { return streamed_; }
    CosetRhoT<Elt> rho_cocycle_block(uint32_t gen, uint32_t i) const;
    CosetRhoT<Elt> rho_s_inv_block(uint32_t i) const;
    CosetRhoT<Elt> rho_t_block(uint32_t i) const;
    CosetRhoT<Elt> rho_cocycle_upper_block(uint32_t gen, uint32_t i) const;

    // --- Gamma^0(N) upper-triangular coset data (Stage S5 addition; not
    // present before S5). Mirrors the Gamma_0(N) tables above exactly, but
    // for the RIGHT cosets Gamma^0(N)\Gamma with representatives t_i and the
    // class map on the TOP row (a:b) in P^1(Z/N) instead of the bottom row
    // (c:d) -- the contract box after [C, prop:blockmatrices] ("The optional
    // system L' := L_{Gamma^0(p)} ... uses representatives t_i of
    // Gamma^0(p)\Gamma (class map = top row (a:b))"). Needed by Stage S5's
    // MakeInducedLocalSystemGamma0Upper and by S8's alpha/omega
    // cross-checks; NOT used by the binding Hecke pipeline (S7).
    //
    // h_i(gamma) := t_i * gamma * t_{i.gamma}^{-1} in Gamma^0(N) (upper-right
    // entry == 0 mod N, by the same argument as [C, prop:blockmatrices]
    // applied to the top row). Never touched by apply_custom_reps(), which
    // only rebuilds the Gamma_0(N) s_-derived tables (t_ is fixed data, like
    // rho_t_ above).
    uint32_t perm_upper(uint32_t gen, uint32_t i) const;                // i . gamma, Gamma^0(N) coset
#if OC_ENABLE_LEGACY
    const FlatMatrixT<Elt>& rho_cocycle_upper(uint32_t gen, uint32_t i) const; // rho_n(h_i(gamma)) mod ell (ON builds)
#endif

    // DoD battery (see HeckeCosetTables.cpp for the full list); 0 on success.
    // Checks, for THIS instance's (p_hecke, r, ell, n_sym):
    //  (i)   num_cosets() == p^r + p^{r-1};
    //  (ii)  every h_i(gamma), gamma in {T,S,ST,TS}, i = 1..m: det == 1 and
    //        lower-left entry == 0 mod N;
    //  (iii) chain rule h_i(g1 g2) == h_i(g1) h_{i.g1}(g2) and
    //        (i.g1).g2 == i.(g1 g2), for all 16 ordered pairs from
    //        {T,S,ST,TS} and 100 fixed-seed random length-<=6 words (exact);
    //  (iv)  rho_n multiplicativity: rho_n(h_i(g1 g2)) ==
    //        rho_n(h_i(g1)) * rho_n(h_{i.g1}(g2)) on the same 16 pairs;
    //  (v)   reconstruction s_i * gamma == h_i(gamma) * s_{i.gamma} exactly,
    //        for gamma in {T,S,ST,TS}, i = 1..m.
    uint32_t self_check() const;

    // Stage S5 addition: the same DoD battery (i)-(v) as self_check(), for
    // the NEW Gamma^0(N) tables above (t_i, perm_upper, cocycle_upper,
    // rho_cocycle_upper), independently re-derived via the general (non-
    // cached) perm_upper_of/cocycle_upper_of helpers exactly as self_check()
    // does for the Gamma_0(N) side. 0 on success.
    uint32_t self_check_upper() const;

private:
    uint64_t p_ = 0;
    uint32_t p32_ = 0;
    uint32_t r_ = 0;
    Elt ell_ = 0;
    uint32_t n_sym_ = 0;
    uint64_t N_ = 0;
    uint32_t N32_ = 0;   // N_, cached as uint32_t (class map domain; checked to fit at construction)
    // (slot-14 WP-10) Read once at construction from the thread-local options.
    bool streamed_ = false;         // no rho_n table stored (ScopedCosetStreaming)
    bool parallel_setup_ = false;   // the rho_n tables filled on the consumer pool (ScopedParallelSetup)
    // (matrix-engine Phase 3 Stage B) The modulus context of the LEVEL N.
    // Every scalar operation of the (P1) class map and of the row actions in
    // perm_of / perm_upper_of is arithmetic in Z/N -- NOT in the coefficient
    // field F_ell -- so it needs its own context, built once here instead of
    // re-deriving the reduction constants in every mod_mul call.
    //
    // N = p^r is COMPOSITE for r >= 2 and even for p = 2. engine::ModCtx is
    // defined for every modulus in [2, 2^32): add/sub/neg are the same
    // comparisons the legacy primitives make, mul is a Barrett reduction whose
    // correction step is proved for arbitrary p (ENGINE_API.md Sec. 1,
    // reduce_u64), and inv() forwards to the extended-Euclid mod_inv, which is
    // exactly the general-modulus inverse this class documents its reliance on
    // in the class_index comment. Nothing here uses batch_inv, Shoup or
    // Montgomery, i.e. nothing that would want a prime.
    engine::ModCtx<uint32_t> ctxN_{};
    uint32_t m_ = 0;     // num_cosets()

    Mat22 g_{};

    // All of the following are sized m_+1; index 0 is unused (see the
    // 1-based indexing convention documented above the class).
    std::vector<std::array<uint32_t, 2>> class_repr_; // canonical (c,d) mod N of class i (representative-independent)
    std::vector<Mat22> s_;
    std::vector<Mat22> s_inv_;
    std::vector<Mat22> delta_;
    std::vector<Mat22> t_;                 // Gamma^0(N) reps (top-row classes)
#if OC_ENABLE_LEGACY
    // (S24 CL-R1) the materialised rho_n tables (empty on a streamed table).
    std::vector<FlatMatrixT<Elt>> rho_s_inv_;
    std::vector<FlatMatrixT<Elt>> rho_t_;
#endif

    // Outer index: gen in {0,1,2,3}; inner: coset index i (size m_+1).
    std::vector<std::vector<uint32_t>> perm_;
    std::vector<std::vector<Mat22>> cocycle_;
#if OC_ENABLE_LEGACY
    std::vector<std::vector<FlatMatrixT<Elt>>> rho_cocycle_;
#endif

    // --- Gamma^0(N) upper-triangular data (Stage S5 addition) --------------
    std::vector<Mat22> t_inv_; // t_i^{-1}, cached like s_inv_ (t_i in SL_2(Z): safe, I7)
    // Native top-row (a,b) mod N of t_i -- NOT, in general, the canonical
    // class_index() representative for index i itself: t_i's own index i is
    // chosen to mirror s_i's subscripts bit-for-bit ([C]'s r=1 table), which
    // is a DIFFERENT enumeration of P^1(Z/N) than the (c:d) branch-A/B
    // numbering class_index() computes for (a,b). See t_index_of_class_.
    std::vector<std::array<uint32_t, 2>> class_repr_upper_;
    // Reverse lookup: t_index_of_class_[class_index(top row of t_i)] = i.
    // Built once at construction (bijectivity onto {1,...,m} asserted);
    // used by perm_upper_of to translate a raw P^1(Z/N) class back into a
    // t-index.
    std::vector<uint32_t> t_index_of_class_;
    // Outer index: gen in {0,1,2,3}; inner: coset index i (size m_+1).
    std::vector<std::vector<uint32_t>> perm_upper_;
    std::vector<std::vector<Mat22>> cocycle_upper_;
#if OC_ENABLE_LEGACY
    std::vector<std::vector<FlatMatrixT<Elt>>> rho_cocycle_upper_;
#endif

    void init_common(uint64_t p_hecke, uint32_t r, Elt ell, uint32_t n_sym);
    void apply_custom_reps(const std::vector<Mat22>& custom_s);
    void rebuild_from_s(); // (re)computes s_inv_, delta_, rho_s_inv_, cocycle_, rho_cocycle_ from s_
#if OC_ENABLE_LEGACY
    // (S24 CL-R1) the materialised fills (legacy/HeckeCosetTablesMaterialised.inc); no-ops on a streamed table.
    void materialise_rho_t();
    void materialise_rho_cocycle_upper();
    void materialise_rho_s_inv_cocycle();
#endif

    void check_index(uint32_t i) const;
    void check_gen(uint32_t gen) const;

    static Mat22 generator_matrix(uint32_t gen); // 0:T 1:S 2:ST 3:TS, from Mat22::{T,S,ST,TS}()

    // General (non-generator-restricted) versions of perm/cocycle, used both
    // to populate the generator-indexed caches and, in self_check(), to
    // independently re-derive values for arbitrary composite words (so the
    // chain-rule check is a genuine cross-check, not a tautology against the
    // cache it is validating).
    uint32_t perm_of(uint32_t i, const Mat22& gamma) const;
    Mat22 cocycle_of(uint32_t i, const Mat22& gamma) const;

    // Gamma^0(N) analogues of perm_of/cocycle_of above (Stage S5 addition),
    // used both to populate perm_upper_/cocycle_upper_/rho_cocycle_upper_
    // and, in self_check_upper(), as the independent cross-check.
    uint32_t perm_upper_of(uint32_t i, const Mat22& gamma) const;
    Mat22 cocycle_upper_of(uint32_t i, const Mat22& gamma) const;

    // rho_n(m mod ell) via ComputeSymmetricPowerMatrixFM (invariant I2).
    FlatMatrixT<Elt> rho_of(const Mat22& m) const;
};

using HeckeCosetTables = HeckeCosetTablesT<uint32_t>;
using HeckeCosetTables64 = HeckeCosetTablesT<uint64_t>;

} // namespace hecke

#endif // HECKE_COSET_TABLES_H
