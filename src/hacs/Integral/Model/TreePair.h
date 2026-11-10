// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_MODEL_TREE_PAIR_H
#define INTEGRAL_MODEL_TREE_PAIR_H

// ===========================================================================
// DESCENT milestone D5c (hecke-extensions slot 24; docs/implementation/slot24/DESCENT_plan.md v0.4 Sec. 4 "D5c", clause C-D8).
// [C] = docs/latex/gl2_contract.tex (v1.23.0-pre.34, sec:treepair, refereed C-D8_referee.md), cited by label: def:treepair,
// lem:treepaircyl, prop:treepair (1)-(6) and the code map after rmk:treepairscope (obligations (a)-(g)); def:isomodeltree,
// prop:isomodeltree (3), (4); def:descentpair; prop:descentpairhecke.
//
// THE CHAIN-LEVEL HALF OF THE TREE CUSP PAIR (the production cochains are Integral/Descent/DescentTreePair):
//
//   * CuspModelChain: the cusp model B^(ell)_T of def:treepair(1) over Q[Gamma] -- the mapping cone of r_{(1-T) e_e} on the tree's
//     edge column: cells e_t ('A', degree t) and z_t ('Z', degree t + 1), d e_t = eps_t e_{t-1}, d z_t = (1 - T) e_t - eps_t z_{t-1}
//     (eps_t = (1 - c) e_e for odd t, (1 + c) e_e for even t >= 2; c = -I).  ell = 0: the free model B_T (e_e = 1); ell = 2: free,
//     ranks 1, 2, 2, ...; ell = 3 (and >= 5): e_e = (1 + c) / 2, only t = 0 (ranks 1, 1, 0, ...).  For the free model contract() is
//     the solver h_p of prop:isomodeltree(4)(b) along eps_B (the z column on top with sign -1, then the e column; [C] prop:treepair(3)(b)
//     and the C-D8 referee's m2: the solver "for any p that is the column augmentation of a complex filtered by free periodic columns").
//   * AxisMapChain: the axis map a^(ell) : B^(ell)_T -> Iso^(ell)_T of def:treepair(2) (closed form; every term dropped whose target
//     column vanishes).
//   * The geometric tables of def:treepair(4): i_d, r_d, G_d, f_a, g_sq, g' on the representatives of the code's descriptor (v_0 = {0},
//     v_1 = {1}, v_2 = {2}, v_0v_1 = {0,1}, v_1v_2 = {1,2}, v_2v_3 = {2,3}; v_3 = T v_0), checked once ((3)(a)).
//   * TreePairComparison (per ell in {2, 3}): prop:treepair(3) made effective on the code's Iso (LocalModel(Descriptor5(), 0), the Iso
//     of IntegralTreeComparison):
//       Phi^Z_d := (L) for f = r_d o cmpI along eps_B   (Isob -> B_T),
//       Psi^Z_d := (L) for f = i_d o eps_B along cmpI|Isob   (B_T -> Isob; the descent is checked to stay in Isob),
//       Phi^d_ell, Psi^d_ell (closed form: r_{|Q_e|^{floor(t/2)} e_e} on both columns; the identity at ell = 2),
//       Phi_d = Phi^d_ell Phi^Z_d, Psi_d = Psi^Z_d Psi^d_ell,
//       y_ell := (L') along eps_T (x) R on Isob (x) R, A = Phi^Z_T j, B = Psi^T_ell a^(ell) Phi_d, p' = cmpI|Isob, g = g_sq,
//       y'_ell := (L') along cmpI (x) R on B_T (x) R, A = Psi_T a^(ell) Phi^d_ell, B = j Psi^Z_d, p' = eps_B, g = g',
//       Y_Phi = Phi^T_ell y_ell, Y_Psi = y'_ell Psi^d_ell, with
//           Phi_T j - a Phi_d = d Y_Phi + Y_Phi d,     Psi_T a - j Psi_d = d Y_Psi + Y_Psi d       ((3)(c)).
//     certify() checks every identity of (1)(b), (3)(a)-(c) exactly in Q[Gamma] (ell-local idempotents applied): a failure is a
//     failure of the code (prop:treepair(6)), never a refusal -- the production caller throws hecke::HeckeInternalError on it.
//
// Index conventions: a chain is an LChain (cell -> coefficient a, meaning a . cell; maps are Z[Gamma]-linear on the left, d(a . x) =
// a . d(x), the code's convention of IsotropyModelTree).  Chains of Isob are chains of Iso supported on the boundary cells (Iso cell
// indices); maps out of Isob are TreeMaps whose value[q] is indexed by the Iso cell (non-boundary entries empty).
//
// THE PLANTED FAULTS (validation only; the C-D8 code map (f) and the referee's m3): one process-wide selector, None by default, set
// ONLY through ScopedTreePairFault by IntegralDescentTreePairValidation's (N) arm -- never from an env variable, a setting or a CLI
// token.  AxisSForSinv: S for S^{-1} in a (meaningful at ell in {0, 2} only: at ell >= 3 S e_e = S^{-1} e_e, the referee's m3);
// AxisDropBeta: the S^{-1} e_U v^U term of a(z_t) dropped (the ell = 3 control); CuspOneMinusT2: 1 - T^2 for 1 - T in B; JPhiPlusY:
// + Y_Phi^* in j_Phi; EtaDropped: the transported operator without its off-diagonal eta_T.  The first three act on the closed form
// (DescentTreePair) AND on the chain-level a / B here; the last two on the cone maps (DescentTreePair).
//
// No bignum (int64 Rat, overflow-checked), no settings key, no record key, no golden object; nothing here reads an env variable.
// ===========================================================================

#include <cstdint>
#include <string>
#include <vector>

#include "IsotropyModelTree.h"

namespace integral_model {

enum class TreePairFault : uint32_t {
    None = 0,
    AxisSForSinv = 1,     // S for S^{-1} in a^(ell) (ell in {0, 2}: detected; ell >= 3: the same map)
    AxisDropBeta = 2,     // the S^{-1} e_U v^U_{t+1} term of a(z_t) dropped
    CuspOneMinusT2 = 3,   // 1 - T^2 for 1 - T in the cusp model's d z_t
    JPhiPlusY = 4,        // + Y_Phi^* in j_Phi
    EtaDropped = 5,       // T^c without eta_T
};
constexpr uint32_t kTreePairFaultCount = 5;
TreePairFault ActiveTreePairFault();
struct ScopedTreePairFault {
    explicit ScopedTreePairFault(TreePairFault f);
    ~ScopedTreePairFault();
    ScopedTreePairFault(const ScopedTreePairFault&) = delete;
    ScopedTreePairFault& operator=(const ScopedTreePairFault&) = delete;
private:
    uint32_t saved_;
};

// A cusp-model cell: k = 'A' (e_t, degree t) or 'Z' (z_t, degree t + 1).
struct CuspCell {
    char k = 'A';
    uint32_t t = 0;
};

class CuspModelChain {
public:
    // ell = 0: the free cusp model B_T; ell = 2, 3 (or a prime >= 5): B^(ell)_T.  Degrees 0 .. q_top.  The planted fault
    // CuspOneMinusT2 (read at construction) replaces 1 - T by 1 - T^2.
    CuspModelChain(uint32_t ell, uint32_t q_top);
    uint32_t ell() const { return ell_; }
    uint32_t q_top() const { return q_top_; }
    const std::vector<CuspCell>& cells(uint32_t q) const { return cells_.at(q); }
    uint32_t rank(uint32_t q) const { return q <= q_top_ ? static_cast<uint32_t>(cells_[q].size()) : 0u; }
    int64_t index(uint32_t q, char k, uint32_t t) const;   // -1 if absent
    const QGroupRingElt& idem() const { return e_; }       // e_e (every column)
    const std::vector<LEntry>& differential(uint32_t q) const { return diff_.at(q); }
    LChain apply_d(uint32_t q, const LChain& x) const;
    // Free model only: k of degree q + 1 with d k = z for a cycle z of ker eps_B of degree q (ENGINE_CHECKed).
    LChain contract(uint32_t q, const LChain& z) const;
    // d o d = 0 exactly (ell-local: after the idempotent); the count of chains checked.
    bool d_squared_zero(uint64_t& checked, std::string& why) const;

private:
    uint32_t ell_, q_top_;
    bool per_;   // the edge column periodic (ell in {0, 2})
    QGroupRingElt e_;
    std::vector<hecke::Mat22> pw_;   // powers of c
    std::vector<std::vector<CuspCell>> cells_;
    std::vector<std::vector<LEntry>> diff_;
    std::vector<std::vector<std::vector<const LEntry*>>> by_col_;
    QGroupRingElt eps(uint32_t t) const;
};

// a^(ell) : B^(ell)_T -> Iso^(ell)_T (TreeModelChain(ell, ...)), degrees 0 .. q_top (def:treepair(2)); ell = 0 the free axis map.  The
// planted faults AxisSForSinv / AxisDropBeta are read when it is built.
TreeMap AxisMapChain(uint32_t ell, const CuspModelChain& B, const TreeModelChain& L);

// Phi^d_ell : B_T (x) R -> B^(ell)_T and Psi^d_ell back (prop:treepair(3)(b)), degrees 0 .. q_top.
TreeMap PhiCuspEll(const CuspModelChain& free, const CuspModelChain& local);
TreeMap PsiCuspEll(const CuspModelChain& local, const CuspModelChain& free);

struct TreePairCertificates {
    bool tables = false;           // (3)(a): i_d, r_d, f_a, eps_B chain maps; r_d i_d = id; id - i_d r_d = dG_d + G_d d; the g_sq and g' identities
    bool cusp_complex = false;     // d o d = 0 on B_T and B^(ell)_T
    bool axis_chain = false;       // (1)(b): a^(ell) a chain map, well defined (e_x a e_e = a)
    bool axis_over_fa = false;     // (1)(b): eps_T a = f_a eps_B in degrees 0, 1 (free model and ell-local)
    bool phiZd_chain = false, psiZd_chain = false;      // (3)(b): chain maps
    bool phiZd_over_r = false, psiZd_over_i = false;    // over r_d o cmpI, resp. i_d o eps_B (degrees 0, 1)
    bool psiZd_in_isob = false;                         // Psi^Z_d lands in Isob (the descent stays there)
    bool phid_psid_strict = false;                      // Phi^d_ell Psi^d_ell = id strictly
    bool cusp_maps_chain = false;                       // Phi^d_ell, Psi^d_ell chain maps
    bool square_phi = false, square_psi = false;        // (3)(c)
    bool integral = false;                              // Phi^Z_d, Psi^Z_d integral (Z[Gamma] coefficients)
    uint64_t checked = 0;
    std::string first_failure;
    bool all() const {
        return tables && cusp_complex && axis_chain && axis_over_fa && phiZd_chain && psiZd_chain && phiZd_over_r && psiZd_over_i &&
               psiZd_in_isob && phid_psid_strict && cusp_maps_chain && square_phi && square_psi && integral;
    }
};

class TreePairComparison {
public:
    // cmp: the comparison of prop:isomodeltree(4) on the code's Iso, built to degree >= q_top (its tree to q_top + 1).  ell in {2, 3}
    // (or a prime >= 5: the Bredon tree pair, a check only, prop:treepair(6)).  Maps in degrees 0 .. q_top, homotopies 0 .. q_top.
    TreePairComparison(const IntegralTreeComparison& cmp, uint32_t ell, uint32_t q_top);
    uint32_t ell() const { return ell_; }
    uint32_t q_top() const { return q_top_; }
    const IntegralTreeComparison& cmp() const { return *cmp_; }
    const LocalModel& iso() const { return cmp_->iso(); }
    const CuspModelChain& cusp_local() const { return Bl_; }
    bool is_boundary(uint32_t q, uint32_t iso_cell) const;

    const TreeMap& axis() const { return axis_; }              // a^(ell): B^(ell) -> Iso^(ell)_T
    const TreeMap& phi_d() const { return phid_; }             // Isob -> B^(ell)
    const TreeMap& psi_d() const { return psid_; }             // B^(ell) -> Iso (Isob)
    const TreeMap& phi_T() const { return phiT_; }             // Phi_T: Iso -> Iso^(ell)_T
    const TreeMap& psi_T() const { return psiT_; }             // Psi_T: Iso^(ell)_T -> Iso
    const TreeMap& Y_phi() const { return Yphi_; }             // Isob -> Iso^(ell)_T, degree + 1 (value[q] indexed by the Iso cell)
    const TreeMap& Y_psi() const { return Ypsi_; }             // B^(ell) -> Iso, degree + 1

    TreePairCertificates certify() const;
    double build_seconds = 0;

private:
    const IntegralTreeComparison* cmp_;
    uint32_t ell_, q_top_;
    TreeModelChain L_;
    CuspModelChain B0_, Bl_;
    std::vector<std::vector<uint32_t>> isob_;
    std::vector<std::vector<char>> isb_;
    TreeMap axis_, axis0_, phiZd_, psiZd_, phid_, psid_, phiDl_, psiDl_, phiT_, psiT_, Yphi_, Ypsi_, yl_, ypl_;
    // representatives of the boundary simplices and of g' (Iso cell index of (s, rep, t = 0))
    uint32_t cell_of(const Simplex& s) const;
};

}  // namespace integral_model

#endif
