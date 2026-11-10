// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_MODEL_ISOTROPY_MODEL_TREE_H
#define INTEGRAL_MODEL_ISOTROPY_MODEL_TREE_H

// ===========================================================================
// DESCENT milestone D5 (hecke-extensions slot 24; docs/implementation/slot24/DESCENT_plan.md v0.4 Sec. 4 "D5").
// [C] = docs/latex/gl2_contract.tex, cited by label: def:isomodeltree, prop:isomodeltree (C-D6; [C] v1.23.0-pre.23; refereed at pre.21,
// ACCEPT, docs/implementation/slot24/C-D6_referee.md incl. the re-referee), def:isomodel, prop:isomodel, lem:strictlifting.
//
// THE CHAIN-LEVEL HALF OF THE TREE MODELS (the production cochains in closed form are TreeCochains.h):
//
//   * TreeModelChain: the free tree model Iso_T (rmk:treeconeres; def:isomodeltree(4)) and the ell-local tree models
//     Iso^(ell)_T at ell = 2, 3, >= 5 (def:isomodeltree(2), (3): cones of the canonical lifts), as chain data over Q[Gamma]:
//     cells (S, t), (U, t) in degree t and (E, t) in degree t + 1; d(a, z) = (d a + phi'_t z, - d^E z); the column
//     differentials r_{(1-g) e} (odd t), r_{N'} (even t >= 2); the idempotents e_x.  certify(): d o d = 0, well defined
//     (e_row a e_col = a), the lift a chain map over the augmentations ((1)(a)).  For the free model, contract() is the
//     solver h_p of (4)(b) along eps_T (edge column first, global sign -1; the tree columns' contraction negated from even
//     degrees).
//   * PhiTreeEll / PsiTreeEll: the closed-form comparisons Phi^T_ell, Psi^T_ell of (3) (free <-> ell-local), with the
//     corrections k_t, k'_t of (3)(a)-(c).
//   * IntegralTreeComparison: (4) made effective on the code's Iso -- LocalModel(desc, 0), whose differential is the code's
//     IsotropyModel's (checked by the D5 validation) and whose contract() is (4)(b)'s solver for ker cmpI (descending s,
//     the column contraction times (-1)^s):
//       - the tables of (4)(a): r_T, G_T on the 19 representatives, i_T on the tree's three cells;
//       - Phi^Z_T := (L) for f = r_T o cmpI along eps_T, Psi^Z_T := (L) for f = i_T o eps_T along cmpI;
//       - H_T (L') with d H_T + H_T d = Phi^Z Psi^Z - id; H_X with d H_X + H_X d = id - Psi^Z Phi^Z (over G_T);
//       - per ell, H_ell with d H_ell + H_ell d = Psi^T_ell Phi^T_ell - id on the free tree; K := Phi^T_ell H_T Psi^T_ell,
//         K' := H_X - Psi^Z H_ell Phi^Z ((4)(d)).
//     Built once over Z (Phi^Z, Psi^Z, H_T, H_X are integral; certify() checks every coefficient), independently of ell.
//   * Pullback matrices f^* on cochains over any ModCtx and any CoefficientModule ((f^* c)(y) = sum_x rho(b_yx) c(x)), in
//     the full layout (cells x dim); the ell-local cochains are the image of the projector diag rho(e_x).
//
// Phi_T = Phi^T_ell o Phi^Z_T and Psi_T = Psi^Z_T o Psi^T_ell are the maps of (4)(d); their pullbacks are the cochain
// homotopy equivalences C^._{T,ell}(M) <-> C_I(M) through which ANY map between the glued groups enters an ell-local piece
// ((5)'s last sentence) -- in particular the Hecke operators by transport, (6): T'_T := Psi_T^* T Phi_T^* (TransportToTree
// below is that hook; the D5 validation does not exercise it, a cochain-level Hecke operator on Iso at level N being the
// plan's C-D4 / D3).
//
// Bulk only (no Xbdry, Isob, pair or cusp).  No bignum (int64 Rat, overflow-checked); no settings key, no golden object.
// ===========================================================================

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "IsotropyModel.h"
#include "IsotropyModelEll.h"
#include "../../../ModArithmetic/DenseBuf.h"
#include "../../../ModArithmetic/ModCtx.h"

namespace integral_model {

// A tree cell: x in {'S', 'U', 'E'}; total degree t for S, U and t + 1 for E.
struct TreeCell {
    char x = 'S';
    uint32_t t = 0;
};

struct TreeCertificates {
    bool d_squared_zero = false, well_defined = false, lift_over_boundary = false;
    uint64_t checked = 0;
    std::string first_failure;
    bool all() const { return d_squared_zero && well_defined && lift_over_boundary; }
};

class TreeModelChain {
public:
    // ell = 0: the free tree model Iso_T; ell = 2, 3, or a prime >= 5: the canonical ell-local model.  Built in degrees
    // 0 .. q_top.
    TreeModelChain(uint32_t ell, uint32_t q_top);
    uint32_t ell() const { return ell_; }
    uint32_t q_top() const { return q_top_; }
    const std::vector<TreeCell>& cells(uint32_t q) const { return cells_.at(q); }
    uint32_t rank(uint32_t q) const { return static_cast<uint32_t>(cells_.at(q).size()); }
    int64_t index(uint32_t q, char x, uint32_t t) const;   // -1 if absent
    const QGroupRingElt& idem(uint32_t q, uint32_t cell) const;   // e_x of the cell's column
    // d(e_col) = sum coeff e_row; entries of d_q, q = 1 .. q_top.
    const std::vector<LEntry>& differential(uint32_t q) const { return diff_.at(q); }
    LChain apply_d(uint32_t q, const LChain& x) const;
    // Free model only: k of degree q + 1 with d k = z for a cycle z of ker eps_T of degree q ((4)(b)); ENGINE_CHECKed.
    LChain contract(uint32_t q, const LChain& z) const;
    // eps_T of a chain of degree q in {0, 1} as canonical (column, coset representative) -> multiplicity (C_q(T)).
    std::map<std::pair<char, GKey>, Rat> augment(uint32_t q, const LChain& x) const;
    TreeCertificates certify() const;
    // Column data: generator g_x, |Gamma_x|, |P_x| (the ell-Sylow; the free model: |Gamma_x|), e_x.
    const std::vector<hecke::Mat22>& column_pow(char x) const;
    const QGroupRingElt& column_e(char x) const;

    template<class Elt>
    engine::DenseBuf<Elt> coboundary(const engine::ModCtx<Elt>& ctx, const CoefficientModule<Elt>& M, uint32_t q) const;
    template<class Elt>
    engine::DenseBuf<Elt> projector(const engine::ModCtx<Elt>& ctx, const CoefficientModule<Elt>& M, uint32_t q) const;

private:
    uint32_t ell_, q_top_;
    std::vector<std::vector<TreeCell>> cells_;
    std::vector<std::vector<LEntry>> diff_;
    std::vector<std::vector<std::vector<const LEntry*>>> by_col_;
    std::map<char, std::vector<hecke::Mat22>> pow_;
    std::map<char, uint32_t> porder_;
    std::map<char, QGroupRingElt> e_;
    bool col_exists(char x, uint32_t t) const;
    QGroupRingElt col_diff(char x, uint32_t t) const;
};

// A chain map (or homotopy) src -> dst: value[q][src cell] = a chain of dst in degree q (+1 for a homotopy).
struct TreeMap {
    std::vector<std::vector<LChain>> value;
    uint32_t degree_shift = 0;   // 0: a chain map; 1: a homotopy
};
LChain ApplyMap(const TreeMap& f, uint32_t q, const LChain& x);
// (second o first), degrees 0 .. min(q range); first: A -> B, second: B -> C (second's degree shift added).
TreeMap ComposeMaps(const TreeMap& first, const TreeMap& second);

// (3): Phi^T_ell : Iso_T -> Iso^(ell)_T and Psi^T_ell : Iso^(ell)_T -> Iso_T (ell = 2, 3, or >= 5), degrees 0 .. q_top.
TreeMap PhiTreeEll(uint32_t ell, uint32_t q_top);
TreeMap PsiTreeEll(uint32_t ell, uint32_t q_top);

// The tables of (4)(a), on the representatives of the code's descriptor (keyed by their stored vertex lists).
struct RetractionTables {
    // r_T(sigma) for sigma in Sigma_0: ('S' | 'U', coefficient); for sigma in Sigma_1: the coefficient of the edge cell E
    std::map<Simplex, std::pair<char, QGroupRingElt>> r0;
    std::map<Simplex, QGroupRingElt> r1;
    // G_T(sigma): a chain of C_{s+1}(Xbulk) on the representatives (Simplex -> coefficient)
    std::map<Simplex, std::map<Simplex, QGroupRingElt>> G;
    // i_T: S -> v_6, U -> v_4, E -> {v_4, v_6}
    std::map<char, Simplex> iota;
};
RetractionTables DefaultRetractionTables();

struct IsoTreeCertificates {
    bool tables = false;              // (4)(a): r_T chain map, aug-preserving, r_T i_T = id, id - i_T r_T = dG + Gd (19 reps)
    bool iso_is_code_model = false;   // LocalModel(desc, 0)'s differential == IsotropyModel's
    bool phi_chain = false, psi_chain = false, phi_over_r = false, psi_over_iota = false, integral = false;
    bool H_T = false, H_X = false;
    uint64_t checked = 0;
    std::string first_failure;
    bool all() const {
        return tables && iso_is_code_model && phi_chain && psi_chain && phi_over_r && psi_over_iota && integral && H_T && H_X;
    }
};

class IntegralTreeComparison {
public:
    // Builds Phi^Z_T, Psi^Z_T (degrees 0 .. q_top) and H_T, H_X (degrees 0 .. q_top - 1) on LocalModel(desc, 0), desc built
    // with q_max >= q_top + 1.  tables: (4)(a) (the default; a mutated copy is the validation's negative control).
    IntegralTreeComparison(const IsotropyModel& desc, uint32_t q_top, const RetractionTables& tables = DefaultRetractionTables());
    uint32_t q_top() const { return q_top_; }
    const LocalModel& iso() const { return iso_; }
    const TreeModelChain& tree() const { return tree_; }
    const TreeMap& phi() const { return phi_; }     // Iso -> Iso_T
    const TreeMap& psi() const { return psi_; }     // Iso_T -> Iso
    const TreeMap& H_T() const { return HT_; }      // on Iso_T
    const TreeMap& H_X() const { return HX_; }      // on Iso
    IsoTreeCertificates certify() const;

    // Per ell: H_ell (on the free tree) and the composite homotopies of (4)(d), checked; returns false (and why) on a failure.
    struct EllCheck {
        bool phi_T_chain = false, psi_T_chain = false, H_ell = false, K = false, K_prime = false, phi_psi_strict = false;
        uint64_t checked = 0;
        std::string first_failure;
        bool all() const { return phi_T_chain && psi_T_chain && H_ell && K && K_prime && phi_psi_strict; }
    };
    EllCheck check_ell(uint32_t ell) const;
    // Phi_T = Phi^T_ell o Phi^Z_T : Iso -> Iso^(ell)_T and Psi_T = Psi^Z_T o Psi^T_ell : Iso^(ell)_T -> Iso ((4)(d)).
    TreeMap phi_T(uint32_t ell) const;
    TreeMap psi_T(uint32_t ell) const;

private:
    uint32_t q_top_;
    const IsotropyModel* desc_;
    LocalModel iso_;
    TreeModelChain tree_;
    RetractionTables tab_;
    TreeMap phi_, psi_, HT_, HX_;
    bool tables_ok_ = false;
    std::string tables_failure_;
    uint64_t tables_checked_ = 0;
    void check_tables();
};

// f^* : C^q(dst) -> C^q(src) over ctx in the full layouts (rows src cells x dim, cols dst cells x dim).  The models are
// given by their cell counts; src / dst cells index the TreeMap.
template<class Elt>
engine::DenseBuf<Elt> PullbackMap(const engine::ModCtx<Elt>& ctx, const CoefficientModule<Elt>& M, const TreeMap& f, uint32_t q,
                                  uint32_t src_cells, uint32_t dst_cells);

// (6) the hook: T'_T := Psi_T^* o T o Phi_T^* on C^q_{T,ell}(M) for a cochain endomorphism T of C^q_I(M) given as a matrix
// in the full layout of Iso (w_q dim square); returns the matrix on the full layout of Iso^(ell)_T.  Not exercised in D5
// (no cochain Hecke operator on Iso at level N yet: C-D4 / D3); provided so that D3 lands on a fixed interface.
template<class Elt>
engine::DenseBuf<Elt> TransportToTree(const engine::ModCtx<Elt>& ctx, const CoefficientModule<Elt>& M, const IntegralTreeComparison& cmp,
                                      uint32_t ell, uint32_t q, const engine::DenseBuf<Elt>& T_on_Iso);

}  // namespace integral_model

#endif
