// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_MODEL_ISOTROPY_MODEL_ELL_H
#define INTEGRAL_MODEL_ISOTROPY_MODEL_ELL_H

// ===========================================================================
// hecke-extensions slot 16, WP-16 (stretch): the ell-local isotropy model
// Iso^(ell) of [C, def:isomodelell] in code, on IsotropyModel's plug point
// (a parameter ell; [C] = docs/latex/gl2_contract.tex v1.16.0-pre.3, cited by
// label; the chapter is [C, sec:isomodelell]).
//
// WHAT IS BUILT.
//   * LocalModel(desc, ell, averaging): for each representative sigma of
//     Sigma_s (read from an IsotropyModel's DESCRIPTOR only -- representatives,
//     stabiliser generators and orders, face transports, boundary flags; no
//     vertex count, no edge list, no special role of any vertex beyond "the
//     stabiliser order is divisible by ell", [C, rmk:isomodellmp](3)) the
//     column F'^sigma: P_sigma the ell-Sylow subgroup of Gamma_sigma =
//     <g_sigma>, Q_sigma = <g_sigma^{|P|}> its ell'-part, e_sigma =
//     |Q|^{-1} sum_{u in Q} u.  If P_sigma != 1 the column is periodic,
//     F'_t = R[Gamma_sigma] e_sigma, d = (g - 1) e from odd t and the norm
//     (sum_{i<|P|} g^i) e from even t >= 2; if P_sigma = 1 it is R in degree 0.
//     ell = 0 means NO localisation (Q_sigma = 1, P_sigma = Gamma_sigma): the
//     free model Iso of [C, def:isomodel], rebuilt by the same code -- the
//     validation checks it equals IsotropyModel's differential term by term.
//   * Coefficients live in Q[Gamma] restricted to denominators prime to ell
//     (exact int64 rationals, overflow-checked; Z_(ell) contains them), every
//     group element an exact hecke::Mat22.  A generator e'_{s,sigma,t} =
//     1 (x) e_sigma; an element c e' is stored by its canonical coefficient
//     c e_sigma (right-normalised), so equality is equality in Q[Gamma].
//   * d = d'^(0) + d'^(1) + d'^(2) by the procedure of [C, prop:isomodelell](1):
//     lifts through the R-linear contracting homotopies of the augmented
//     columns, boundary summands first, and EVERY lift replaced by its
//     Q_sigma-average y |-> e_sigma y (the C-1 repair, WP-14 referee).
//     averaging == LiftAveraging::Off is the NEGATIVE CONTROL (the lift as the
//     free procedure would produce it); stats() counts the lifts the average
//     changed.
//   * certify(): d o d = 0 exactly in Q[Gamma] ([C, IM1]), every coefficient of
//     d well defined (e_col a e_row = a, i.e. a lies in Hom(R[Gamma]e_col,
//     R[Gamma]e_row)), the boundary sub-sum a subcomplex ([C, IM2]), and the
//     augmentation eps' a chain map onto C_.(Xbulk) (exact, on canonical cosets).
//   * StrictComparison(src, dst, averaging): the strict lift of the identity of
//     C_.(Xbulk) (x) R between two such models -- Phi_ell : Iso (x) R -> Iso^(ell)
//     (src = the ell = 0 model) and Psi_ell : Iso^(ell) -> Iso (x) R
//     ([C, prop:isomodelell](4)); y_0 := e_sigma y_0', k := e_sigma k', the
//     correction k by descent through the target's columns.  certify_comparison()
//     re-derives: chain map, strictness over Xbulk, and the well-definedness
//     e_x Phi(x) = Phi(x) on every source generator (the un-averaged Psi_ell
//     fails it on every source generator with Q != 1, the non-central
//     Q_{v6} = <S> included -- the negative control).
//   * Cochains over a ModCtx (F_ell or Z/ell^k; denominators inverted mod the
//     modulus): coboundary(ctx, n, q) on the FULL layout (+)_cells M (cell *
//     d_V + component, the L1 layout of IsotropyModel), projector(ctx, n, q) =
//     diag rho_n(e_cell) whose image is C^q = (+) M^{Q_cell}
//     ([C, prop:isomodelell](3)); the coboundary kills (1 - P) and lands in
//     P, so H^q has length  len(P_q) - len(im delta^q) - len(im delta^{q-1}).
//     pullback(ctx, n, q) of a comparison: (f |-> f o Phi) as a matrix.
//
// AUDIT NOTES (slot-16 WP-17, the WP-16 cross-model audit).
//   * (F-1) The d-side average is UNTESTABLE on SL_2(Z): every lift of d'^(1), d'^(2)
//     starts at a simplex of dimension >= 1, whose stabiliser is {+-I} ([C, B3]: a
//     non-central element fixes only isolated points of H), central and contained in
//     every target Q_tau at odd ell -- so e_sigma y = y and LiftAveraging::Off gives the
//     same d (the validation's (NC, d-side) pins this).  The average is load-bearing in
//     Psi_ell (StrictComparison), whose (NC) is the real negative control.  It stays in
//     build_differentials() as the general statement of [C, prop:isomodelell](1).
//   * (F-5) ell is checked prime (trial division) in the constructor.  Nothing ties ell
//     to the ModCtx later passed to coboundary() / projector() / cohomology_lengths();
//     the one unsafe case (a denominator |Q| not a unit of the ring) is refused by
//     inverse_mod's unit check (an ENGINE_CHECK), never computed.
//   * (F-4) A LocalModel is not copyable (by_col_ holds pointers into diff_); it is
//     movable (a move keeps diff_'s element storage).  The descriptor must outlive the
//     model, and StrictComparison's same-descriptor check compares by address.
//   * (F-8) Nothing here hard-wires 7 vertices, a trivial centre or Bredon semantics.
//     Since slot-17 WP-7 the "refined descriptor plugs in unchanged" half of
//     [C, rmk:isomodellmp] is TESTED: IsotropyModel(ModelData, q_max) takes any
//     descriptor as data, SubdivideEdgeOrbit builds the Moerdijk--Pronk-conformant
//     subdivision at zeta^2, and HeckeIsotropyModelEllValidation's (REF) arm compares
//     LocalModel on it with the unrefined model on cohomology.  Since WP-7 the cochains
//     take any CoefficientModule (the level-N module Ind_{Gamma_0(N)}^Gamma V_R of
//     [C, IL3]'s level-N clause; -I |-> -1 admissible).
//
// DEPENDENCIES: IsotropyModel.h, PeriodicColumn.h (decompose), Hecke/ExactInt.h,
// ModArithmetic/ only -- the Integral/Model/ rule.  No HeckeExt/, no bignum.
// ===========================================================================

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "IsotropyModel.h"
#include "../../Hecke/ExactInt.h"
#include "../../../ModArithmetic/DenseBuf.h"
#include "../../../ModArithmetic/ModCtx.h"

namespace integral_model {

// An exact rational num/den, den > 0, gcd(num, den) = 1; int64, overflow-checked
// (hecke::HeckeOverflow).
struct Rat {
    int64_t num = 0;
    int64_t den = 1;
    static Rat of(int64_t n, int64_t d = 1);
    bool is_zero() const { return num == 0; }
    bool operator==(const Rat& o) const { return num == o.num && den == o.den; }
    bool operator!=(const Rat& o) const { return !(*this == o); }
};
Rat operator+(const Rat& a, const Rat& b);
Rat operator*(const Rat& a, const Rat& b);
Rat operator-(const Rat& a);

// An element of Q[Gamma] (finitely many (g, r), r != 0); ordered map.
struct QGroupRingElt {
    std::map<GKey, Rat> terms;
    bool is_zero() const { return terms.empty(); }
    static QGroupRingElt unit(const hecke::Mat22& g, Rat r = Rat::of(1));
    static QGroupRingElt from(const GroupRingElt& z);
    void add(const GKey& g, const Rat& r);
    void add(const QGroupRingElt& o, const Rat& scale = Rat::of(1));
    QGroupRingElt operator*(const QGroupRingElt& o) const;
    bool operator==(const QGroupRingElt& o) const { return terms == o.terms; }
    bool operator!=(const QGroupRingElt& o) const { return !(*this == o); }
    int64_t max_den() const;
    std::string to_string() const;
};

enum class LiftAveraging { On, Off };

// (hecke-extensions slot 17, WP-7) A coefficient module M of finite rank over a
// ModCtx, given by its LEFT action g |-> rho_M(g) (dim x dim; rho_M(gh) =
// rho_M(g) rho_M(h)), so the cochains C^q = (+) M^{Q_sigma} of
// [C, prop:isomodelell](3) are built for ANY Z_(ell)[Gamma]-module the caller
// can write down: Sym^n (SymModule, the default of every n-taking overload,
// value for value the slot-16 code) or Ind_{Gamma_0(N)}^Gamma V_R (the level-N
// clause of [C, IL3]; built by the caller from the shipped Gamma0LevelN table).
// Nothing here assumes -I acts trivially: a module with -I |-> -1 is admissible.
template<class Elt>
struct CoefficientModule {
    std::string name;
    uint32_t dim = 0;
    std::function<engine::DenseBuf<Elt>(const hecke::Mat22&)> rho;
};
template<class Elt>
CoefficientModule<Elt> SymModule(const engine::ModCtx<Elt>& ctx, uint32_t n);

// The column of one representative ([C, def:isomodelell]).
struct ColumnSpec {
    std::vector<hecke::Mat22> pow;   // g^0 .. g^{m-1}
    uint32_t m = 1;                  // |Gamma_sigma|
    uint32_t p_order = 1;            // |P_sigma|
    uint32_t q_order = 1;            // |Q_sigma|
    QGroupRingElt e;                 // e_sigma
    bool periodic() const { return p_order > 1; }
};

struct LCell {
    uint32_t s = 0, rep = 0, t = 0;
    bool boundary = false;
};

struct LEntry {
    uint32_t row = 0;   // cell index in degree q - 1
    uint32_t col = 0;   // cell index in degree q
    QGroupRingElt coeff;
};

// A chain in one degree: cell index -> canonical (right-normalised) coefficient.
using LChain = std::map<uint32_t, QGroupRingElt>;

struct LocalCertificates {
    bool d_squared_zero = false;
    bool well_defined = false;        // e_col a e_row == a on every entry
    bool boundary_subcomplex = false; // IM2
    bool eps_chain_map = false;       // IM1
    uint64_t d_squared_terms_checked = 0;
    uint64_t entries_checked = 0;
    uint64_t eps_cells_checked = 0;
    std::string first_failure;
    bool all() const { return d_squared_zero && well_defined && boundary_subcomplex && eps_chain_map; }
};

struct LiftStats {
    uint64_t lifts = 0;           // lifts produced by a contracting homotopy or the t = 0 formula
    uint64_t changed_by_average = 0;
};

class LocalModel {
public:
    // ell = 0: no localisation (the free model).  The complex is built in
    // degrees 0 .. desc.q_top().
    LocalModel(const IsotropyModel& desc, uint32_t ell, LiftAveraging avg = LiftAveraging::On);
    // (audit WP-16 F-4) by_col_ points into diff_: a copy would dangle; a move keeps the storage.
    LocalModel(const LocalModel&) = delete;
    LocalModel& operator=(const LocalModel&) = delete;
    LocalModel(LocalModel&&) = default;
    LocalModel& operator=(LocalModel&&) = default;

    uint32_t ell() const { return ell_; }
    uint32_t q_max() const { return q_max_; }
    uint32_t q_top() const { return q_max_ + 1; }
    const IsotropyModel& descriptor() const { return *desc_; }
    const ColumnSpec& column(uint32_t s, uint32_t rep) const { return cols_[s][rep]; }
    uint32_t num_reps(uint32_t s) const { return static_cast<uint32_t>(cols_[s].size()); }

    uint32_t rank(uint32_t q) const { return static_cast<uint32_t>(cells_.at(q).size()); }
    const std::vector<LCell>& cells(uint32_t q) const { return cells_.at(q); }
    // -1 if (s, rep, t) is not a cell (P_sigma = 1 and t >= 1, or out of range).
    int64_t cell_index(uint32_t s, uint32_t rep, uint32_t t) const;
    const std::vector<LEntry>& differential(uint32_t q) const { return diff_.at(q); }
    const LiftStats& stats() const { return stats_; }

    LocalCertificates certify() const;

    // ---- chain-level helpers (used by StrictComparison) ---------------------
    // d of a chain of degree q (q >= 1): a chain of degree q - 1.
    LChain apply_d(uint32_t q, const LChain& x) const;
    // The canonical form c e_cell of a coefficient at a cell of degree q.
    QGroupRingElt normalise(uint32_t q, uint32_t cell, const QGroupRingElt& c) const;
    // k of degree q + 1 with d k = z for a cycle z of degree q in ker eps
    // (descent through the columns); ENGINE_CHECKs every input condition.
    LChain contract(uint32_t q, const LChain& z) const;
    // The augmentation of a degree-0 / any chain, as (rep, canonical coset) -> multiplicity
    // over the t = 0 cells (Xbulk's s-chains at s = q).
    std::map<std::pair<uint32_t, GKey>, Rat> augment(uint32_t q, const LChain& x) const;

    // ---- cochains over a ModCtx ----------------------------------------------
    template<class Elt>
    engine::DenseBuf<Elt> coboundary(const engine::ModCtx<Elt>& ctx, uint32_t n, uint32_t q) const;
    template<class Elt>
    engine::DenseBuf<Elt> projector(const engine::ModCtx<Elt>& ctx, uint32_t n, uint32_t q) const;
    // Boundary sub-model: the same restricted to boundary cells.
    template<class Elt>
    engine::DenseBuf<Elt> boundary_coboundary(const engine::ModCtx<Elt>& ctx, uint32_t n, uint32_t q) const;
    template<class Elt>
    engine::DenseBuf<Elt> boundary_projector(const engine::ModCtx<Elt>& ctx, uint32_t n, uint32_t q) const;
    // lengths of H^q, q = 0 .. q_max.
    template<class Elt>
    std::vector<uint32_t> cohomology_lengths(const engine::ModCtx<Elt>& ctx, uint32_t n, bool boundary = false) const;
    // (slot-17 WP-7) The same five over any coefficient module (layout cell * dim + component).
    template<class Elt>
    engine::DenseBuf<Elt> coboundary(const engine::ModCtx<Elt>& ctx, const CoefficientModule<Elt>& M, uint32_t q) const;
    template<class Elt>
    engine::DenseBuf<Elt> projector(const engine::ModCtx<Elt>& ctx, const CoefficientModule<Elt>& M, uint32_t q) const;
    template<class Elt>
    engine::DenseBuf<Elt> boundary_coboundary(const engine::ModCtx<Elt>& ctx, const CoefficientModule<Elt>& M,
                                              uint32_t q) const;
    template<class Elt>
    engine::DenseBuf<Elt> boundary_projector(const engine::ModCtx<Elt>& ctx, const CoefficientModule<Elt>& M,
                                             uint32_t q) const;
    template<class Elt>
    std::vector<uint32_t> cohomology_lengths(const engine::ModCtx<Elt>& ctx, const CoefficientModule<Elt>& M,
                                             bool boundary = false) const;

private:
    const IsotropyModel* desc_;
    uint32_t ell_;
    uint32_t q_max_;
    LiftAveraging avg_;
    std::array<std::vector<ColumnSpec>, 3> cols_;
    std::vector<std::vector<LCell>> cells_;
    std::vector<std::vector<LEntry>> diff_;
    std::vector<std::vector<std::vector<const LEntry*>>> by_col_;   // [q][col] -> entries
    LiftStats stats_;

    void build_columns();
    void build_differentials();
    template<class Elt>
    engine::DenseBuf<Elt> coboundary_impl(const engine::ModCtx<Elt>& ctx, const CoefficientModule<Elt>& M, uint32_t q,
                                          bool bdry) const;
    template<class Elt>
    engine::DenseBuf<Elt> projector_impl(const engine::ModCtx<Elt>& ctx, const CoefficientModule<Elt>& M, uint32_t q,
                                         bool bdry) const;
};

// The column coefficient x'_t (t >= 1): (g - 1) e from odd t, (sum_{i<|P|} g^i) e from even t.
QGroupRingElt ColumnCoefficient(const ColumnSpec& c, uint32_t t);
// The R-linear contracting homotopy h_t : F'_t -> F'_{t+1}, coset by coset, on one
// coefficient (input any representative of c e; output canonical).
QGroupRingElt ColumnHomotopyEll(const ColumnSpec& c, const QGroupRingElt& a, uint32_t t);

// A comparison map src -> dst: value[q][src cell] = a chain of dst in degree q.
struct ComparisonMap {
    std::vector<std::vector<LChain>> value;
    LiftStats stats;
};
struct ComparisonCertificate {
    bool chain_map = false;       // d_dst f(x) = f(d_src x)
    bool strict = false;          // eps_dst f = eps_src on Xbulk's chains
    bool well_defined = false;    // e_x f(x) = f(x) on every source generator
    uint64_t cells_checked = 0;
    uint64_t not_well_defined = 0;
    std::string first_failure;
    bool all() const { return chain_map && strict && well_defined; }
};

// Requires src and dst built on the same descriptor with the same q_max.
ComparisonMap StrictComparison(const LocalModel& src, const LocalModel& dst,
                               LiftAveraging avg = LiftAveraging::On);
ComparisonCertificate CertifyComparison(const LocalModel& src, const LocalModel& dst, const ComparisonMap& f);
// f^* : C^q_dst -> C^q_src over ctx (rows src cells * d_V, cols dst cells * d_V).
template<class Elt>
engine::DenseBuf<Elt> Pullback(const engine::ModCtx<Elt>& ctx, uint32_t n, uint32_t q, const LocalModel& src,
                               const LocalModel& dst, const ComparisonMap& f);

// rho_n of a rational group-ring element over ctx (denominators inverted mod
// ctx.p; ENGINE_CHECKs they are units).
template<class Elt>
engine::DenseBuf<Elt> RhoQ(const engine::ModCtx<Elt>& ctx, const QGroupRingElt& a, uint32_t n,
                           std::map<GKey, engine::DenseBuf<Elt>>& memo);
// (slot-17 WP-7) the same for any coefficient module; the memo holds rho_M(g).
template<class Elt>
engine::DenseBuf<Elt> RhoQ(const engine::ModCtx<Elt>& ctx, const QGroupRingElt& a, const CoefficientModule<Elt>& M,
                           std::map<GKey, engine::DenseBuf<Elt>>& memo);
template<class Elt>
engine::DenseBuf<Elt> Pullback(const engine::ModCtx<Elt>& ctx, const CoefficientModule<Elt>& M, uint32_t q,
                               const LocalModel& src, const LocalModel& dst, const ComparisonMap& f);

}  // namespace integral_model

#endif
