// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#include "DenseElimPipelineValidation.h"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../ModArithmetic/LegacyConfig.h"   // (S24 CL-L2) RunLegacyArm (SKIP-ARM)
#include "BlockModelPipelineValidation.h"   // (S24 CL-L2) BuildACqBlockMat: A_Cq without BlockFlatMatrix
#include "../ModArithmetic/DenseBuf.h"
#include "../ModArithmetic/DenseElim.h"
#include "../ModArithmetic/DenseView.h"
#include "../ModArithmetic/FlatMatrix.h"
#include "../ModArithmetic/ModCtx.h"
#include "../ModArithmetic/RefLinAlg.h"
#include "EquivariantCohomologyOnK.h"
#include "LocalSystemSL2Z.h"
#include "LocalSystemSimplicialComplex.h"
#include "SimplicialCohomology.h"

// See DenseElimPipelineValidation.h.

namespace {

using engine::DenseBuf;
using engine::cview;
using engine::view;

uint32_t g_fail = 0;
uint64_t g_checks = 0;
void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        if (g_fail <= 40) std::cout << "  [FAIL] " << what << std::endl;
    }
}
template<class T> std::string str(T v) { return std::to_string(v); }

std::vector<LocalSystemSimplicialComplex::SimplexRelation> BuildRelations(const SimplicialComplex& K,
                                                                          uint32_t n_sym, uint32_t ell) {
    std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels;
    for (const auto& datum : GenerateSL2ZRelationsWithGroupData(K, n_sym, ell)) rels.push_back(datum.rel);
    return rels;
}


DenseBuf<uint32_t> to_buf(const FlatMatrix& f) {
    DenseBuf<uint32_t> b(f.get_rows(), f.get_cols());
    for (uint32_t i = 0; i < b.rows; ++i)
        for (uint32_t j = 0; j < b.cols; ++j) b.at(i, j) = f.get_entry(i, j);
    return b;
}

bool same_buf_fm(const DenseBuf<uint32_t>& a, const FlatMatrix& f) {
    if (a.rows != f.get_rows() || a.cols != f.get_cols()) return false;
    for (uint32_t i = 0; i < a.rows; ++i)
        for (uint32_t j = 0; j < a.cols; ++j)
            if (a.at(i, j) != f.get_entry(i, j)) return false;
    return true;
}

bool same_rows_vs_columns(const DenseBuf<uint32_t>& a, const std::vector<FlatMatrix>& cols) {
    if (a.rows != cols.size()) return false;
    for (uint32_t t = 0; t < a.rows; ++t) {
        if (cols[t].get_cols() != 1 || cols[t].get_rows() != a.cols) return false;
        for (uint32_t j = 0; j < a.cols; ++j) if (a.at(t, j) != cols[t].get_entry(j, 0)) return false;
    }
    return true;
}

bool same_rows_vs_vectors(const DenseBuf<uint32_t>& a, const std::vector<std::vector<uint32_t>>& v) {
    if (a.rows != v.size()) return false;
    for (uint32_t t = 0; t < a.rows; ++t) {
        if (v[t].size() != a.cols) return false;
        for (uint32_t j = 0; j < a.cols; ++j) if (a.at(t, j) != v[t][j]) return false;
    }
    return true;
}

} // namespace   (S24 CL-L2: the legacy .inc includes its headers at file scope)
#if OC_ENABLE_LEGACY
#include "legacy/DenseElimPipelineLegacyArms.inc"   // (S24 CL-L2) BuildACq (BlockFlatMatrix), the legacy halves, whole_pipeline
#else
// (S24 CL-L2) A_Cq without BlockFlatMatrix: the block-model assembly of BlockModelPipelineValidation (the same live
// relations, I - M / I, -M blocks), densified.  The ON build keeps the BlockFlatMatrix assembly (BuildACq).
FlatMatrix AcqFlat(const LocalSystemSimplicialComplex& lssc, uint32_t q) {
    const engine::BlockMat<uint32_t> Ab = BuildACqBlockMat(std::make_shared<engine::BlockPool<uint32_t>>(), lssc, q);
    const DenseBuf<uint32_t> D = Ab.to_dense();
    std::vector<uint32_t> data(static_cast<size_t>(D.rows) * D.cols);
    for (uint32_t i = 0; i < D.rows; ++i)
        for (uint32_t j = 0; j < D.cols; ++j) data[static_cast<size_t>(i) * D.cols + j] = D.at(i, j);
    if (data.empty()) return FlatMatrix::CreateZero(D.rows, D.cols, lssc.get_prime());
    return FlatMatrix(D.rows, D.cols, lssc.get_prime(), data);
}
#endif
namespace {

// Part 1: the real A_Cq matrices, three ways.
void real_matrices(uint32_t n_sym, uint32_t ell) {
    const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    const auto rels = BuildRelations(K, n_sym, ell);
    const LocalSystemSimplicialComplex lssc(K, rels, n_sym + 1, ell);
    const engine::ModCtx<uint32_t> ctx(ell);
    const bool sc = EquivariantCohomologyOptions::vacuous_relation_shortcircuit_enabled();

    for (uint32_t q = 0; q <= 2; ++q) {
        const std::string tag = "A_C" + str(q) + " (n_sym=" + str(n_sym) + (sc ? ", shortcircuit on)" : ", shortcircuit off)");
#if OC_ENABLE_LEGACY
        const BlockFlatMatrix Ablk = BuildACq(lssc, q);
        const FlatMatrix Afm = Ablk.to_flat_matrix();
#else
        const FlatMatrix Afm = AcqFlat(lssc, q);
#endif
        const DenseBuf<uint32_t> A = to_buf(Afm);
        std::cout << "  " << tag << ": " << A.rows << " x " << A.cols << std::flush;

        // engine
        DenseBuf<uint32_t> R = A;
        const engine::elim::RrefResult res = engine::elim::rref<uint32_t>(ctx, view(R));
        const DenseBuf<uint32_t> Kb = engine::elim::kernel_basis<uint32_t>(ctx, cview(R), res.pivots);
        const DenseBuf<uint32_t> Ib = engine::elim::image_basis<uint32_t>(cview(A), res.pivots);
        std::cout << ", rank " << res.rank << ", nullity " << Kb.rows << std::endl;
        if (q == 1 && n_sym == 128) {
            const bool sc = EquivariantCohomologyOptions::vacuous_relation_shortcircuit_enabled();
            check(A.rows == (sc ? 258u : 1677u) && A.cols == 1419 && res.rank == 258,
                  tag + (sc ? " is the shipped 258 x 1419 rank-258 matrix"
                            : " is the 1677 x 1419 rank-258 matrix of doc 03 Sec. 10.2"));
        }

        // oracle
        {
            DenseBuf<uint32_t> Rr = A;
            const std::vector<uint32_t> pv = engine::ref::rref<uint32_t>(ell, view(Rr));
            check(pv == res.pivots && R.same_entries(Rr), tag + " RREF/pivots vs RefLinAlg");
            const auto kr = engine::ref::kernel_basis<uint32_t>(ell, cview(A));
            check(same_rows_vs_vectors(Kb, kr), tag + " kernel vs RefLinAlg");
        }

        // legacy FlatMatrix paths and BlockFlatMatrix::flat_kernel, both modes
        {
#if OC_ENABLE_LEGACY
            legacy_flat_arm(Afm, res, R, Kb, Ib, tag);   // legacy/DenseElimPipelineLegacyArms.inc
#endif
            {
                const FlatMatrix M = Afm;
                check(M.get_pivot_columns() == res.pivots, tag + " pivots vs retrofit FlatMatrix");
                check(same_buf_fm(R, M.get_row_echelon_form_as_flat_matrix()), tag + " RREF vs retrofit FlatMatrix");
                check(same_rows_vs_columns(Kb, M.kernel()), tag + " kernel vs retrofit FlatMatrix::kernel");
                check(same_rows_vs_columns(Ib, M.image()), tag + " image vs retrofit FlatMatrix::image");
            }
#if OC_ENABLE_LEGACY
            legacy_block_kernel_arm(Ablk, Kb, tag);   // legacy/DenseElimPipelineLegacyArms.inc
#endif
        }
    }
}

} // namespace

uint32_t DenseElimPipelineValidation() {
    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();
    g_fail = 0; g_checks = 0;
    std::cout << "=== DenseElimPipelineValidation (real A_Cq matrices + whole compute_cohomology, legacy vs engine) ===" << std::endl;
    (void)oc::legacy::RunLegacyArm("DenseElimPipelineValidation", "legacy-real-matrices");   // (S24 CL-L2) the legacy halves of Part 1
    try {
        real_matrices(128, 134217757u);                       // shipped shapes (short-circuit on)
        {
            EquivariantCohomologyOptions::ScopedVacuousShortcircuit off(false);
            real_matrices(128, 134217757u);                   // pre-7.4 shapes: A_C1 = 1677 x 1419
        }
        if (oc::legacy::RunLegacyArm("DenseElimPipelineValidation", "whole_pipeline")) {   // (S24 CL-L2) ON only
#if OC_ENABLE_LEGACY
            whole_pipeline(32, 134217757u);
            {
                EquivariantCohomologyOptions::ScopedVacuousShortcircuit off(false);
                whole_pipeline(32, 134217757u);
            }
            whole_pipeline(16, 5u);
#endif
        }
    } catch (const std::exception& e) {
        check(false, std::string("exception: ") + e.what());
    }
    const double secs = std::chrono::duration<double>(Clock::now() - t0).count();
    std::cout << "  " << g_checks << " checks, " << g_fail << " failures, " << secs << " s" << std::endl;
    std::cout << (g_fail == 0 ? "DenseElimPipelineValidation: PASS" : "DenseElimPipelineValidation: FAIL") << std::endl;
    return g_fail;
}
