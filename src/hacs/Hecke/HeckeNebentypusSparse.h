// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_NEBENTYPUS_SPARSE_H
#define HECKE_NEBENTYPUS_SPARSE_H
// ===========================================================================
// (hecke-extensions slot 17, WP-6) NEBENTYPUS AT SCALE: THE SPARSE COKERNEL WITH A CHARACTER.
// Binding text: [C] sec:nebentypus -- prop:neborbit (1), (3) (the orbit-wise fixed points and the
// cokernel H^1 = M / (M^<S> + M^<ST>) of the character model eq:MHchi), rmk:nebheilbronn (H_{N,p,chi}
// on it), rmk:nebcost (1), (4)'s "every count of rmk:levelcost carries over with F_ell-operations
// replaced by F_q-operations", and the slot-17 code-map scope (i) of sec:nebentypus.
//
// WHAT IS BUILT.
//  * NebSparseCokernel -- the RREF of Rel_{N,chi} = M^<S> + M^<ST> over F_q, computed SPARSELY by the
//    elimination of HeckeLevelNSparse.h (rmk:levelcost (4)) with F_q arithmetic: the generators are
//    the SAME orbit-wise averaging vectors as NebCokernel's (each supported on one orbit: <= 2 blocks
//    for S, <= 3 for ST; S's first, then ST's, in NebCokernel's order), and the pivot of every reduced
//    generator is its SMALLEST column.  Every stored row is then e_c + (entries at free columns > c),
//    so the result IS the reduced row echelon form of Rel in the natural column order -- unique -- and
//    equals NebCokernel's dense RowBasis entrywise wherever both run: the same pivots, the same free
//    columns (the quotient basis), the same rows, hence the same coordinates and the same matrices of
//    every H_{N,p,chi} (HeckeNebentypusValidation (SP) checks all of it entrywise).  It has no dense
//    cap: D = m_N (n + 1) < 2^32 (the column domain), and the memory is the fill (CokernelCostModel).
//  * OnH1 / RelEscapes / HeilbronnChiSparse -- H_{N,p,chi} on the free-column basis column by column
//    from the operator's block terms (no D-length dense apply), and rmk:nebheilbronn's "H(Rel) in Rel"
//    on the kept generators (equivalent to the RREF rows).
//  * NebRoute / NebPresentation -- the route choice of HECKE_NEBENTYPUS_CELLS' route= field:
//    dense (NebCokernel, refusing neb_dense_cap above D = 4096 as shipped), sparse (this file), auto
//    (dense iff D <= kNebDenseCap -- the shipped route byte for byte -- sparse above it).
//
// Refusals: those of NebCokernel except neb_dense_cap (nebentypus_small_ell at ell < 5; chi(-1) !=
// (-1)^n is parity_zero, not a refusal).  The character reduction's refusals (character_wild,
// character_field) happen before, in ReduceCharacter.
//
// MP constraint ([C] box after rmk:nebcost (v)): the cokernel route reads no model (no 7-vertex
// hard-wiring); -I enters only through its scalar chi(-1)(-1)^n (certified on the orbits, as
// NebCokernel); ell in {2, 3} refused by name, never deferred to a Bredon object.
//
// Nothing here is reached from a shipped record, report, campaign job id, setting or golden text.
// ===========================================================================

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "HeckeLevelN.h"
#include "HeckeLevelNSparse.h"   // level::SparseVec
#include "HeckeNebentypus.h"

namespace hecke {
namespace neb {

using level::SparseVec;

struct NebSparseStats {
    uint64_t generators = 0, generator_nnz = 0, dependent = 0;
    uint64_t final_nnz = 0, peak_nnz = 0, merge_ops = 0;
    double generators_s = 0, eliminate_s = 0;
};

class NebSparseCokernel {
public:
    // The character model of (N, chi, n) over F (prop:neborbit (3)), sparse.  keep_generators retains the
    // generators for RelEscapes.
    NebSparseCokernel(const level::Gamma0LevelN& lvl, uint32_t n_sym, const Fq& F, const CharacterValues& chi,
                      bool keep_generators = true);
    bool parity_zero() const { return parity_zero_; }
    uint32_t D() const { return D_; }
    uint32_t dim_S() const { return dim_S_; }
    uint32_t dim_U() const { return dim_U_; }
    uint32_t dim_rel() const { return static_cast<uint32_t>(R_.pivots.size()); }
    uint32_t h1() const { return parity_zero_ ? 0 : static_cast<uint32_t>(R_.free.size()); }
    uint32_t h0() const { return parity_zero_ ? 0 : dim_S_ + dim_U_ - dim_rel(); }
    uint32_t orbits_S() const { return orbits_S_; }
    uint32_t orbits_U() const { return orbits_U_; }
    const std::vector<uint32_t>& pivots() const { return R_.pivots; }        // ascending
    const std::vector<uint32_t>& free_columns() const { return R_.free; }    // ascending: the quotient basis
    bool is_pivot(uint32_t c) const { return R_.row_of_col[c] >= 0; }
    // The RREF row of the pivot c: e_c + tail (tail at free columns > c).
    const SparseVec& tail_of(uint32_t c) const { return R_.rows[static_cast<size_t>(R_.row_of_col[c])]; }
    int32_t free_index(uint32_t c) const { return R_.free_index[c]; }
    const NebSparseStats& stats() const { return stats_; }
    const Fq& field() const { return *F_; }

    std::vector<uint32_t> coords(const SparseVec& x) const;                 // on the free columns
    std::vector<uint32_t> coords_dense(const std::vector<uint32_t>& x) const;
    bool in_rel(const SparseVec& x) const;
    // rmk:nebheilbronn: the number of kept generators whose image leaves Rel (needs keep_generators).
    uint32_t RelEscapes(const HeilbronnChi& H) const;
    // [H] on M / Rel in the free-column basis (h1 x h1); ENGINE_CHECKs RelEscapes(H) == 0 first when the generators
    // were kept.
    FqMat OnH1(const HeilbronnChi& H) const;

private:
    const Fq* F_;
    uint32_t D_ = 0, dim_S_ = 0, dim_U_ = 0, orbits_S_ = 0, orbits_U_ = 0;
    bool parity_zero_ = false;
    sparse::Rref R_;                       // the stored RREF (SparseRref.h, shared with level::SparseCokernel)
    std::vector<SparseVec> gens_;
    NebSparseStats stats_;
    void eliminate(const std::vector<SparseVec>& gens);
};

// H_{N,p,chi} applied to a sparse vector, the terms grouped by source block once.
class HeilbronnChiSparse {
public:
    explicit HeilbronnChiSparse(const HeilbronnChi& H, uint32_t D);
    SparseVec apply(const Fq& F, const SparseVec& x) const;
private:
    const HeilbronnChi* H_;
    uint32_t D_ = 0, d_ = 1;
    std::vector<std::vector<uint32_t>> by_src_;   // block (0-based) -> term indices
};

// The sparse rank of vectors supported in disjoint orbits (the per-orbit ranks summed), over F_q.
uint32_t FqSparseRank(const Fq& F, uint32_t D, const std::vector<SparseVec>& vecs);

// ---- the route ---------------------------------------------------------------------------------
enum class NebRoute : uint8_t { kDense = 0, kSparse = 1, kAuto = 2 };
std::string NebRouteName(NebRoute r);
// "dense" | "sparse" | "auto"; throws std::invalid_argument naming `what` otherwise.
NebRoute ParseNebRoute(const std::string& s, const std::string& what);
// kAuto: dense iff D = m_N (n + 1) <= kNebDenseCap (the shipped route), sparse above.
NebRoute ResolveNebRoute(NebRoute r, uint64_t m_N, uint32_t n_sym);

// One presentation by the chosen route (the dense route refuses neb_dense_cap as shipped).
class NebPresentation {
public:
    NebPresentation(const level::Gamma0LevelN& lvl, uint32_t n_sym, const Fq& F, const CharacterValues& chi,
                    NebRoute route = NebRoute::kAuto);
    NebRoute route() const { return route_; }        // the resolved route (never kAuto)
    bool parity_zero() const;
    uint32_t D() const;
    uint32_t h0() const;
    uint32_t h1() const;
    uint32_t dim_S() const;
    uint32_t dim_U() const;
    FqMat OnH1(const HeilbronnChi& H) const;
    const NebCokernel* dense() const { return dense_.get(); }
    const NebSparseCokernel* sparse() const { return sparse_.get(); }
private:
    NebRoute route_;
    std::unique_ptr<NebCokernel> dense_;
    std::unique_ptr<NebSparseCokernel> sparse_;
};

} // namespace neb
} // namespace hecke

#endif // HECKE_NEBENTYPUS_SPARSE_H
