// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Stage S5 -- see InducedLocalSystem.h for the contract references.

#include "InducedLocalSystem.h"

#include "../LocalSystemSL2Z.h" // GenerateSL2ZRelationsWithGroupData, SL2ZRelationDatum
#include "../../ModArithmetic/FlatMatrix.h"
#include "../../ModArithmetic/BlockMat.h"

#include <array>
#include <memory>
#include <stdexcept>

namespace hecke {

namespace {

using engine::BlockBuilder;
using engine::BlockMat;
using engine::BlockPool;
using engine::BlockRef;
using engine::DenseBuf;
using engine::Layout;

// The dense FlatMatrix a SimplexRelation demands, from a block matrix whose
// grid is m x m over d_V-wide blocks. One materialisation pass over the
// STRUCTURALLY NONZERO blocks only; the structural zeros are never written
// beyond the single zero-fill of the buffer.
FlatMatrix block_to_flat(const BlockMat<uint32_t>& B, uint32_t ell) {
    const uint32_t dim = B.rows();
    if (dim == 0) return FlatMatrix::CreateZero(dim, dim, ell);
    const DenseBuf<uint32_t> d = B.to_dense(Layout::RowMajor);
    std::vector<uint32_t> buf(static_cast<size_t>(dim) * dim, 0);
    for (uint32_t r = 0; r < dim; ++r)
        for (uint32_t c = 0; c < dim; ++c) buf[static_cast<size_t>(r) * dim + c] = d.at(r, c);
    return FlatMatrix(dim, dim, ell, buf);
}

// rho as a row-major DenseBuf (FlatMatrix may store a scalar instead of data).
DenseBuf<uint32_t> to_buf(const FlatMatrix& m) {
    DenseBuf<uint32_t> b(m.get_rows(), m.get_cols());
    for (uint32_t r = 0; r < b.rows; ++r)
        for (uint32_t c = 0; c < b.cols; ++c) b.at(r, c) = m.get_entry(r, c);
    return b;
}

// Maps a relation's exact group element g_r to the S3 generator-dictionary
// index {0:T, 1:S, 2:ST, 3:TS} -- the SAME dictionary HeckeCosetTables::perm/
// rho_cocycle (and perm_upper/rho_cocycle_upper) use. Every seed relation of
// GenerateSL2ZRelationsWithGroupData carries g_r as exactly one of these
// four matrices (invariant I3 / the S2 generator seed table
// SL2ZRelationSeeds_7VertexModel), so this must always succeed for the
// fixed 7-vertex model. (The FIX-negI relations with g_r = -I are handled
// separately in MakeInducedLocalSystemImpl and never reach this function.)
uint32_t gen_index_of(const Mat22& g_r) {
    if (g_r == Mat22::T())  return 0;
    if (g_r == Mat22::S())  return 1;
    if (g_r == Mat22::ST()) return 2;
    if (g_r == Mat22::TS()) return 3;
    throw std::runtime_error(
        "hecke::MakeInducedLocalSystem*: relation g_r is not one of the S2 "
        "generator dictionary {T, S, ST, TS}.");
}

// Assembles M_H(g_r) = sum_i E_{i, i.g_r} (x) rho_n(h_i(g_r)) ([C, eq:MH])
// as an m x m grid of (n_sym+1)-wide blocks, m = cosets.num_cosets().
// `upper` selects the Gamma^0(N) (perm_upper / rho_cocycle_upper) vs.
// Gamma_0(N) (perm / rho_cocycle) coset data.
//
// (matrix-engine Phase 3 Stage B; the Open-Question-Q8 item of audit 02
// Sec. 1.4 T4 / Sec. 7.12.) The matrix has EXACTLY ONE nonzero block per block
// row -- density 1/m -- and the same rho_n(h_i(g_r)) recurs across block rows.
// It is therefore ASSEMBLED as an engine::BlockMat: m block writes, each a
// reference into a content-addressed pool that stores every distinct cocycle
// block once, and no traversal at all of the m^2 - m zero blocks.
//
// What is still NOT block-sparse is the STORAGE handed to the caller.
// LocalSystemSimplicialComplex::SimplexRelation::M is a dense FlatMatrix
// (00-implementation-plan.md, Phase 3, "Matrix retirement": the structural
// pin), so the block matrix is materialised once here. Making the relation
// itself carry a BlockMat is a change to LocalSystemSimplicialComplex and to
// every consumer of SimplexRelation::M, not to this file -- until then a
// consumer that wants the block form recovers it in O(dim^2) with
// BlockMat::from_dense(..., detect = true), which turns each all-zero
// d_V x d_V window back into a structural zero (HeckeCrossChecks.cpp does
// exactly that for the L'' relations).
BlockMat<uint32_t> BuildMHBlock(std::shared_ptr<BlockPool<uint32_t>> pool,
                                const HeckeCosetTables& cosets, uint32_t gen,
                                uint32_t n_sym, uint32_t ell, bool upper) {
    const uint32_t m = cosets.num_cosets();
    const uint32_t block = n_sym + 1;
    const std::vector<uint32_t> dims(m, block);
    BlockBuilder<uint32_t> b(std::move(pool), dims, dims, ell);
    for (uint32_t i = 1; i <= m; ++i) {
        const uint32_t j = upper ? cosets.perm_upper(gen, i) : cosets.perm(gen, i);
        // (slot-14 WP-10) block accessors: a streamed coset table evaluates the block.
        const hecke::CosetRhoT<uint32_t> rho_block = upper ? cosets.rho_cocycle_upper_block(gen, i)
                                                           : cosets.rho_cocycle_block(gen, i);
        const FlatMatrix& rho = rho_block.get();
        if (rho.get_rows() != block || rho.get_cols() != block) {
            throw std::runtime_error(
                "hecke::BuildMH: rho_n(h_i(g_r)) block dimension mismatch "
                "(expected (n_sym+1) x (n_sym+1) -- cosets was likely built "
                "with a different n_sym than the one passed here).");
        }
        if (rho.get_p() != ell) {
            throw std::runtime_error(
                "hecke::BuildMH: rho_n(h_i(g_r)) prime mismatch -- cosets "
                "was likely built with a different ell than the one passed "
                "here.");
        }
        // block row i (1-based -> 0-based), block column i.g_r; interned once
        // per DISTINCT cocycle matrix, and detected as alpha*I when it is one.
        b.set(i - 1, j - 1, b.intern_dense(cview(to_buf(rho)), Layout::RowMajor, true));
    }
    return b.finalize();
}

FlatMatrix BuildMH(const HeckeCosetTables& cosets, uint32_t gen, uint32_t n_sym,
                uint32_t ell, bool upper) {
    auto pool = std::make_shared<BlockPool<uint32_t>>();
    return block_to_flat(BuildMHBlock(std::move(pool), cosets, gen, n_sym, ell, upper), ell);
}

// Shared assembly for both Gamma_0(N) and Gamma^0(N): reuse EXACTLY the
// simplex pairs/tags/order of GenerateSL2ZRelationsWithGroupData, replacing
// only M by M_H(g_r) ([C, def:inducedls]).
LocalSystemSimplicialComplex MakeInducedLocalSystemImpl(
    const SimplicialComplex& K, const HeckeCosetTables& cosets,
    uint32_t ell, uint32_t n_sym, bool upper) {
    const uint32_t m = cosets.num_cosets();
    if (m == 0) {
        throw std::runtime_error("hecke::MakeInducedLocalSystem*: cosets.num_cosets() == 0.");
    }
    const uint32_t block = n_sym + 1;
    const uint32_t rep_dim = m * block;

    const std::vector<SL2ZRelationDatum> base =
        GenerateSL2ZRelationsWithGroupData(K, n_sym, ell);

    // Cache M_H(g_r) per distinct generator (each of T, S, ST, TS occurs
    // several times among the 8 relations of the 7-vertex model) -- mirrors
    // the rho_n(.) cache in GenerateSL2ZRelationsWithGroupData.
    std::array<FlatMatrix, 4> M_H_by_gen = {
        FlatMatrix::CreateZero(rep_dim, rep_dim, ell), FlatMatrix::CreateZero(rep_dim, rep_dim, ell),
        FlatMatrix::CreateZero(rep_dim, rep_dim, ell), FlatMatrix::CreateZero(rep_dim, rep_dim, ell)
    };
    std::array<bool, 4> built = { false, false, false, false };

    // FIX-negI: since the central -I fix, the base relation list carries one
    // self-relation with g_r = -I per simplex of K (see
    // GenerateSL2ZRelationsWithGroupData / [C, thm:explicit_complex_equiv]).
    // -I is central in GL(2,Q) and lies in both Gamma_0(N) and Gamma^0(N),
    // so its induced matrix is computed directly from [C, eq:MH]: the coset
    // permutation is trivial (H*gamma_i*(-I) = H*(-I)*gamma_i = H*gamma_i)
    // and the cocycle is h_i(-I) = gamma_i*(-I)*gamma_i^{-1} = -I; hence
    //   M_H(-I) = I_m (x) rho_n(-I) = (-1)^n * Id_{m*(n_sym+1)},
    // built lazily below with rho_n(-I) derived per invariant I3.
    const Mat22 neg_I = Mat22::I().neg();
    FlatMatrix M_H_neg_I = FlatMatrix::CreateZero(rep_dim, rep_dim, ell);
    bool built_neg_I = false;

    std::vector<LocalSystemSimplicialComplex::SimplexRelation> relations;
    relations.reserve(base.size());
    for (const auto& d : base) {
        if (d.g_r == neg_I) {
            if (!built_neg_I) {
                const std::array<uint32_t, 4> nI_mod = neg_I.reduced_mod(ell);
                const FlatMatrix nI_fm(2, 2, ell,
                    std::vector<uint32_t>{ nI_mod[0], nI_mod[1], nI_mod[2], nI_mod[3] });
                const FlatMatrix rho_neg_I = ComputeSymmetricPowerMatrixFM(nI_fm, n_sym);
                // I_m (x) rho_n(-I): block-DIAGONAL, one interned block shared
                // by all m block rows (and rho_n(-I) = (-1)^n I is detected as
                // a Scalar ref, so it carries no data at all).
                auto pool = std::make_shared<BlockPool<uint32_t>>();
                const std::vector<uint32_t> dims(m, block);
                BlockBuilder<uint32_t> nb(pool, dims, dims, ell);
                const BlockRef<uint32_t> R =
                    nb.intern_dense(cview(to_buf(rho_neg_I)), Layout::RowMajor, true);
                for (uint32_t i = 0; i < m; ++i) nb.set(i, i, R);
                M_H_neg_I = block_to_flat(nb.finalize(), ell);
                built_neg_I = true;
            }
            relations.emplace_back(
                d.rel.q_dim, d.rel.s_lhs_idx_in_Kq, d.rel.s_rhs_idx_in_Kq,
                M_H_neg_I, d.rel.tag);
            continue;
        }
        const uint32_t gen = gen_index_of(d.g_r);
        if (!built[gen]) {
            M_H_by_gen[gen] = BuildMH(cosets, gen, n_sym, ell, upper);
            built[gen] = true;
        }
        // Same (q_dim, s_lhs_idx, s_rhs_idx, tag) as d.rel; only M differs
        // (M_H(g_r) in place of rho_n(g_r)) -- [C, def:inducedls].
        relations.emplace_back(
            d.rel.q_dim, d.rel.s_lhs_idx_in_Kq, d.rel.s_rhs_idx_in_Kq,
            M_H_by_gen[gen], d.rel.tag);
    }

    return LocalSystemSimplicialComplex(K, relations, rep_dim, ell);
}

} // namespace

LocalSystemSimplicialComplex MakeInducedLocalSystemGamma0(
    const SimplicialComplex& K, const HeckeCosetTables& cosets,
    uint32_t ell, uint32_t n_sym) {
    return MakeInducedLocalSystemImpl(K, cosets, ell, n_sym, /*upper=*/false);
}

LocalSystemSimplicialComplex MakeInducedLocalSystemGamma0Upper(
    const SimplicialComplex& K, const HeckeCosetTables& cosets,
    uint32_t ell, uint32_t n_sym) {
    return MakeInducedLocalSystemImpl(K, cosets, ell, n_sym, /*upper=*/true);
}

} // namespace hecke
