// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_LEVEL_N_DEGEN_H
#define HECKE_LEVEL_N_DEGEN_H
// ===========================================================================
// (hecke-extensions slot 16, WP-13) GENERAL N IN CODE: kappa_N READ ON THE MODEL, THE
// DEGENERACY MAPS alpha_d (d > 1) ON THE COKERNEL PRESENTATION.
// Binding text: [C] v1.16.0-pre.3 sec:levelNdegsec --
//   prop:kappamodel    kappa^mod_N : C^1(K, R; L) -> M, f |-> -f(e_46), and
//                      kappa^{mod,0}_N : g |-> g(v_4); e_46 = {v_4, v_6} the stored edge,
//                      d e_46 = v_6 - v_4.  (1) it computes the cokernel M / Rel,
//                      (3) fibre maps commute with it EXACTLY, (4) the entrywise operator
//                      comparison (model operator at e_46 minus presentation operator in Rel);
//   prop:degenpres     eq:degenpres --
//                        D_{M,N,d} = sum_{eps in Mer_d} sum_{i : (i, eps) admissible}
//                                    E_{i, j(i,eps)} (x) rho_n(w_i adj(eps) w'^{-1}_{j(i,eps)}),
//                      (i, eps) admissible iff row_2(w_i) adj(eps) = 0 mod d,
//                      j(i, eps) = cls_M(row_2(w_i) adj(eps) / d);
//                      (1) D(Rel_M) in Rel_N, (2) [D] = alpha_d, (3) d = 1 is alpha^0 entrywise,
//                      (4) composite d, (5) the trace, (7) the cost with the admissible classes
//                      ENUMERATED PER eps (WP-14 referee C-5), not scanned;
//   rmk:degenpresconv  the traps (a) eps for adj(eps), (b) the adjugate of the block -- kept
//                      here as NEGATIVE CONTROLS only;
//   rmk:levelNscale    the cost / memory model of the ladder;
//   rmk:levelNcheckabletwo (LN7)-(LN9).
//
// WHAT IS BUILT.
//  * KappaModelPoints / KappaModel / KappaModel0 -- the pinned model -> presentation map on
//    flat cochains of the fixed model (the edge is FOUND by its vertex set, and its stored
//    orientation is checked here against the TRIVIAL-COEFFICIENT incidence matrix of K only --
//    a property of the simplicial complex, not of the local system.  The check that bites the
//    real equivariant complex, (d g)(e_46) = g(v_6) - g(v_4) EXACTLY for d^0 on C^0_eq together
//    with "D_0 (x) I spans the stored B^1_eq", is (K) of HeckeLevelNSparseValidation; slot-16
//    WP-17, audit WP-13 F5).
//  * DegeneracyOperator -- D_{M,N,d} over any word-sized prime field or Z/ell^k, as block
//    terms (dst class i, src class j, d_V x d_V block), every block self-checked (integral,
//    det d, = diag(1, d) a with a in Gamma_0(M)).  Two enumerations of the admissible pairs,
//    which must give the same term multiset: kPerEpsilon (the classes of P^1(Z/N) bucketed
//    by their class in P^1(Z/d) once; per eps only the m_d classes mod d are tested, C-5) and
//    kScan (every (i, eps) tested; the comparison route).
//  * OnCokernelDegeneracy / DegeneracyRelEscapes -- [D] between two sparse presentations
//    (HeckeLevelNSparse.h), prop:degenpres (1) on the source's generators first.
//
// Nothing here reads a tau-table, searches a coset or inverts a matrix outside Gamma
// (prop:degenpres (7), obl:integral).  Nothing here is reached from a shipped record,
// report, campaign job id or golden text: pipeline-free and value-free.
// MP note: the model is read only through KappaModelPoints (the edge is located by its vertex
// set; on a refinement it becomes a path, prop:kappamodel); -I enters only through n even;
// 6 must be a unit for the presentation (nothing is asserted at ell in {2, 3}).
// ===========================================================================

#include <cstdint>
#include <string>
#include <vector>

#include "HeckeLevelEll.h"
#include "HeckeLevelN.h"
#include "HeckeLevelNHeilbronn.h"
#include "HeckeLevelNSparse.h"

namespace hecke {
namespace level {

// ---- prop:kappamodel ----------------------------------------------------------------------
struct KappaPoints {
    uint32_t e46 = 0;   // index of the edge {v_4, v_6} among the 1-simplices
    uint32_t v4 = 0;    // index of v_4 among the 0-simplices
    uint32_t v6 = 0;
};
// Locates e_46 = {v_4, v_6} in K by its vertex set and checks the stored orientation of the
// simplicial complex: the trivial-coefficient coboundary row of e_46 is +1 at v_6 and -1 at v_4
// (and zero elsewhere).  This says nothing about the local system; see the header note.
KappaPoints KappaModelPoints(const SimplicialComplex& K);
// The value of column `col` of flat cochains Y (rows sigma * D + r, `ncols` columns) at sigma.
std::vector<uint32_t> CochainAt(const std::vector<uint32_t>& Y, uint32_t ncols, uint32_t col, uint32_t sigma, uint32_t D);
// kappa^mod_N(f) = -f(e_46) over ctx (degree 1); kappa^{mod,0}_N(g) = g(v_4) (degree 0).
std::vector<uint32_t> KappaModel(const engine::ModCtx<uint32_t>& ctx, const std::vector<uint32_t>& Y, uint32_t ncols,
                                 uint32_t col, const KappaPoints& pts, uint32_t D);
std::vector<uint32_t> KappaModel0(const std::vector<uint32_t>& Y, uint32_t ncols, uint32_t col, const KappaPoints& pts,
                                  uint32_t D);

// ---- prop:degenpres ------------------------------------------------------------------------
enum class DegeneracyVariant : uint8_t {
    kProposition = 0,   // eq:degenpres
    kTrapA = 1,         // rmk:degenpresconv (2)(a): eps in place of adj(eps)       (negative control only)
    kTrapB = 2,         // rmk:degenpresconv (2)(b): rho_n(adj B) in place of rho_n(B) (negative control only)
};
enum class DegeneracyEnumeration : uint8_t {
    kPerEpsilon = 0,    // prop:degenpres (7): bucket P^1(Z/N) by P^1(Z/d) once, test m_d classes per eps
    kScan = 1,          // every (i, eps) tested (the comparison route)
};

struct DegeneracyTerm {
    uint32_t dst = 0, src = 0;   // 1-based: (D x)_dst += B x_src, dst in P^1(Z/N), src in P^1(Z/M)
    uint32_t block = 0;          // index into blocks()
};

class DegeneracyOperator {
public:
    // lvl_M, lvl_N the def:cosetrepsN tables, d >= 1 with d M | N, n_sym even or not (the
    // formula does not care; the presentation needs n even), ctx any word-sized ring.
    DegeneracyOperator(const Gamma0LevelN& lvl_M, const Gamma0LevelN& lvl_N, uint64_t d, uint32_t n_sym,
                       const engine::ModCtx<uint32_t>& ctx,
                       DegeneracyEnumeration enumeration = DegeneracyEnumeration::kPerEpsilon,
                       DegeneracyVariant variant = DegeneracyVariant::kProposition);

    uint64_t M() const { return M_; }
    uint64_t N() const { return N_; }
    uint64_t d() const { return dd_; }
    uint32_t fibre() const { return dv_; }                     // d_V = n + 1
    uint32_t src_m() const { return mM_; }
    uint32_t dst_m() const { return mN_; }
    uint32_t src_dim() const { return mM_ * dv_; }
    uint32_t dst_dim() const { return mN_ * dv_; }
    DegeneracyVariant variant() const { return variant_; }
    DegeneracyEnumeration enumeration() const { return enum_; }
    const engine::ModCtx<uint32_t>& ctx() const { return ctx_; }
    std::string label() const;

    uint64_t merel_size() const { return mer_size_; }          // |Mer_d| (1 at d = 1)
    uint64_t pair_tests() const { return pair_tests_; }        // admissibility tests performed
    uint32_t max_admissible_per_eps() const { return max_adm_; }
    const std::vector<DegeneracyTerm>& terms() const { return terms_; }
    const std::vector<Block>& blocks() const { return blocks_; }

    std::vector<uint32_t> apply(const std::vector<uint32_t>& x) const;   // dense, over ctx
    SparseVec apply_sparse(const SparseVec& x) const;                    // field only
    std::vector<uint32_t> dense() const;                                 // dst_dim x src_dim, row-major
    // The terms as a canonical multiset key list (dst, src, the block's entries), sorted --
    // equal for the two enumerations.
    std::vector<std::vector<uint32_t>> canonical_terms() const;

private:
    uint64_t M_ = 1, N_ = 1, dd_ = 1;
    uint32_t dv_ = 1, mM_ = 1, mN_ = 1;
    DegeneracyEnumeration enum_;
    DegeneracyVariant variant_;
    engine::ModCtx<uint32_t> ctx_;
    uint64_t mer_size_ = 0, pair_tests_ = 0;
    uint32_t max_adm_ = 0;
    std::vector<DegeneracyTerm> terms_;
    std::vector<Block> blocks_;
    std::vector<std::vector<uint32_t>> by_src_;                // term indices per source block
};

// prop:degenpres (1), reported: the number of generators of Rel_M (kept by src) whose image
// leaves Rel_N.
uint32_t DegeneracyRelEscapes(const SparseCokernel& src, const SparseCokernel& dst, const DegeneracyOperator& D);
// The matrix (dst.h1 x src.h1) of [D_{M,N,d}] on the two presentations (free-column bases);
// ENGINE_CHECKs prop:degenpres (1) first.
HMat OnCokernelDegeneracy(const SparseCokernel& src, const SparseCokernel& dst, const DegeneracyOperator& D);

} // namespace level
} // namespace hecke

#endif // HECKE_LEVEL_N_DEGEN_H
