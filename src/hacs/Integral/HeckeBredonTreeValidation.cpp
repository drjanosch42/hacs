// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#include "HeckeBredonTreeValidation.h"

#include "../LocalSystemSL2Z.h"              // CreateSL2ZFundamentalDomainTriangulation,
                                              // GenerateSL2ZRelationsWithGroupData
#include "../LocalSystemSimplicialComplex.h"
#include "../EquivariantCohomologyOnK.h"
#include "../SimplicialCohomology.h"          // SimplicialComplex
#include "../../ModArithmetic/FlatMatrix.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// TreeRhoN -- an INDEPENDENT construction of rho_n(A) for A = (a b; c d),
// straight from [C, def:rhon]'s closed form:
//
//     rho_n(A)_{t,j} = sum_s C(n-j,s) a^{n-j-s} c^s * C(j,t-s) b^{j-t+s} d^{t-s}
//
// with the sum over s in [max(0, t-j), min(n-j, t)] so that both binomial
// arguments stay in range.  Binomials mod ell come from a Pascal's-triangle
// table built here (not ModArithmetic::mod_nCr / mod_nCr_w), and powers are
// accumulated by a plain uint64_t loop reduced by `% ell` (not
// ModArithmetic::mod_pow / mod_pow_w).  Nothing here calls
// ComputeSymmetricPowerMatrixFM or any ModArithmetic::mod_* primitive, so
// this routine shares no code with the production rho_n the shipped pipeline
// uses (K6, K12 of [IB] Sec. 6.4).
//
// ell is always 2 or 3 here (V-B's scope), n <= 40, so the O(n^3) cost is a
// few tens of thousands of machine words -- there is no need for anything
// beyond the most direct translation of the formula.
FlatMatrix TreeRhoN(int64_t a_in, int64_t b_in, int64_t c_in, int64_t d_in, uint32_t n, uint32_t ell) {
    auto reduce = [ell](int64_t v) -> uint32_t {
        int64_t r = v % static_cast<int64_t>(ell);
        if (r < 0) r += static_cast<int64_t>(ell);
        return static_cast<uint32_t>(r);
    };
    const uint32_t a = reduce(a_in);
    const uint32_t b = reduce(b_in);
    const uint32_t c = reduce(c_in);
    const uint32_t d = reduce(d_in);
    const uint32_t d_V = n + 1;

    // Pascal's triangle mod ell, C[i][k] = C(i, k) mod ell, 0 <= i, k <= n.
    std::vector<std::vector<uint32_t>> binom(n + 1, std::vector<uint32_t>(n + 1, 0));
    for (uint32_t i = 0; i <= n; ++i) {
        binom[i][0] = 1 % ell;
        for (uint32_t k = 1; k <= i; ++k) {
            uint32_t left = binom[i - 1][k - 1];
            uint32_t right = (k <= i - 1) ? binom[i - 1][k] : 0;
            binom[i][k] = (left + right) % ell;
        }
    }

    auto powmod = [ell](uint32_t base, uint32_t e) -> uint32_t {
        uint64_t r = 1 % ell;
        uint64_t x = base % ell;
        for (uint32_t i = 0; i < e; ++i) r = (r * x) % ell;
        return static_cast<uint32_t>(r);
    };

    FlatMatrix M(d_V, d_V, ell);
    for (uint32_t j = 0; j <= n; ++j) {
        const uint32_t nj = n - j;
        for (uint32_t t = 0; t <= n; ++t) {
            uint64_t acc = 0;
            const uint32_t s_lo = (t > j) ? (t - j) : 0;
            const uint32_t s_hi = std::min(nj, t);
            for (uint32_t s = s_lo; s <= s_hi; ++s) {
                const uint32_t tms = t - s; // index into C(j, t-s), 0 <= tms <= j
                const uint64_t term1 = (static_cast<uint64_t>(binom[nj][s]) *
                                         powmod(a, nj - s) % ell) *
                                        powmod(c, s) % ell;
                const uint64_t term2 = (static_cast<uint64_t>(binom[j][tms]) *
                                         powmod(b, j - tms) % ell) *
                                        powmod(d, tms) % ell;
                acc = (acc + term1 * term2) % ell;
            }
            M.set_entry(t, j, static_cast<uint32_t>(acc));
        }
    }
    return M;
}

struct TreeDims {
    uint32_t h0 = 0;
    uint32_t h1 = 0;
    uint32_t h2 = 0;
};

// The tree model of [IB] Sec. 2.5(a): S = [[0,-1],[1,0]] (order 4, S^2 = -I),
// U = [[0,-1],[1,1]] (order 6, U^3 = -I), generating the Bass-Serre tree's
// vertex stabilisers, and -I generating the edge stabiliser.
//
//   dim H^0 = dim(V^S \cap V^U)
//   dim H^1 = dim V^{-I} - dim(V^S + V^U)
//   dim H^2 = 0
//
// V^g := ker(rho_n(g) - I).  Since S^2 = U^3 = -I, V^S and V^U both lie in
// V^{-I}, so dim H^1 is never negative (checked by construction, not merely
// hoped for).
TreeDims TreeModelDims(uint32_t n, uint32_t ell) {
    const uint32_t d_V = n + 1;
    const FlatMatrix rho_S = TreeRhoN(0, -1, 1, 0, n, ell);
    const FlatMatrix rho_U = TreeRhoN(0, -1, 1, 1, n, ell);
    const FlatMatrix rho_negI = TreeRhoN(-1, 0, 0, -1, n, ell);

    const FlatMatrix I(d_V, ell); // identity
    const FlatMatrix M_S = rho_S - I;
    const FlatMatrix M_U = rho_U - I;
    const FlatMatrix M_negI = rho_negI - I;

    const uint32_t rank_S = M_S.rank();
    const uint32_t rank_U = M_U.rank();
    const uint32_t rank_negI = M_negI.rank();
    const uint32_t dim_S = d_V - rank_S;
    const uint32_t dim_U = d_V - rank_U;
    const uint32_t dim_negI = d_V - rank_negI;

    // dim(V^S \cap V^U) = d_V - rank of the stacked constraint matrix
    // [M_S; M_U] (a common kernel is exactly the common solution set).
    FlatMatrix stacked(2 * d_V, d_V, ell);
    for (uint32_t r = 0; r < d_V; ++r)
        for (uint32_t c = 0; c < d_V; ++c) stacked.set_entry(r, c, M_S.get_entry(r, c));
    for (uint32_t r = 0; r < d_V; ++r)
        for (uint32_t c = 0; c < d_V; ++c) stacked.set_entry(d_V + r, c, M_U.get_entry(r, c));
    const uint32_t rank_stacked = stacked.rank();
    const uint32_t dim_intersect = d_V - rank_stacked;

    const uint32_t dim_sum = dim_S + dim_U - dim_intersect; // dim(V^S + V^U)

    TreeDims out;
    out.h0 = dim_intersect;
    out.h1 = dim_negI - dim_sum; // dim_sum <= dim_negI by S^2 = U^3 = -I, so no underflow
    out.h2 = 0;
    return out;
}

struct OracleCell {
    uint32_t n;
    uint32_t ell;
    uint32_t h0;
    uint32_t h1;
};

// [IB] (docs/implementation/integral_line_design_brief.md) Sec. 2.5(a), audit
// finding B-1's table: the 16 golden cells' dim_H^{0,1} at ell in {2, 3}
// (n in the golden grid {0, 1, 2, 4, 10, 16, 22, 24}, GoldenHarness.cpp's
// kGridNSym). dim H^2 = 0 at every one of these cells. FROZEN -- transcribed
// verbatim from the design brief, never adjusted to fit a computed value.
constexpr OracleCell kOracle[] = {
    {0, 2, 1, 0}, {1, 2, 0, 1}, {2, 2, 1, 1}, {4, 2, 1, 2},
    {10, 2, 2, 4}, {16, 2, 3, 6}, {22, 2, 4, 8}, {24, 2, 5, 8},
    {0, 3, 1, 0}, {1, 3, 0, 0}, {2, 3, 0, 1}, {4, 3, 1, 1},
    {10, 3, 1, 3}, {16, 3, 2, 4}, {22, 3, 2, 6}, {24, 3, 3, 6},
};

} // namespace

uint32_t HeckeBredonTreeValidation() {
    uint32_t failures = 0;

    // Step 1: the tree model against the FROZEN oracle. If this disagrees
    // anywhere, the package stops -- per [X] plan WP-8, "It does not adjust
    // the model" -- without going on to compare against the shipped
    // pipeline (a bug shared by both would otherwise hide behind a match).
    for (const auto& oc : kOracle) {
        const TreeDims t = TreeModelDims(oc.n, oc.ell);
        if (t.h0 != oc.h0 || t.h1 != oc.h1 || t.h2 != 0) {
            std::cerr << "HeckeBredonTreeValidation: TREE MODEL DISAGREES WITH THE FROZEN "
                      << "ORACLE at (n=" << oc.n << ", ell=" << oc.ell << "): tree=("
                      << t.h0 << "," << t.h1 << "," << t.h2 << ") oracle=("
                      << oc.h0 << "," << oc.h1 << ",0)\n";
            ++failures;
        }
    }
    if (failures > 0) {
        std::cerr << "HeckeBredonTreeValidation: stopping -- the tree model disagrees with "
                  << "the frozen golden oracle on " << failures
                  << " of " << (sizeof(kOracle) / sizeof(kOracle[0]))
                  << " cells; not comparing against the shipped pipeline.\n";
        return failures;
    }

    // Step 2: for ell in {2, 3} and every n in {0, ..., 40} (even and odd),
    // the tree model against the shipped pipeline's live dim H^q
    // (EquivariantCohomologyOnK on the real 7-vertex model).
    const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    uint32_t cells_checked = 0;
    for (uint32_t ell : {2u, 3u}) {
        for (uint32_t n = 0; n <= 40; ++n) {
            const TreeDims tree = TreeModelDims(n, ell);

            std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels;
            for (const auto& datum : GenerateSL2ZRelationsWithGroupData(K, n, ell)) {
                rels.push_back(datum.rel);
            }
            LocalSystemSimplicialComplex lssc(K, rels, n + 1, ell);
            EquivariantCohomologyOnK coh(&lssc);
            coh.compute_cohomology();
            const std::vector<uint32_t>& dims = coh.get_H_eq_dimensions();

            const uint32_t pipe_h0 = dims.size() > 0 ? dims[0] : 0;
            const uint32_t pipe_h1 = dims.size() > 1 ? dims[1] : 0;
            const uint32_t pipe_h2 = dims.size() > 2 ? dims[2] : 0;

            bool ok = (tree.h0 == pipe_h0) && (tree.h1 == pipe_h1) && (tree.h2 == pipe_h2);
            // Every degree beyond q=2 the pipeline might report must vanish
            // too (the tree model's dim H^2 = 0 claim covers the whole
            // higher range: the model is 1-dimensional as a complex).
            for (uint32_t q = 3; ok && q < dims.size(); ++q) ok = (dims[q] == 0);

            if (!ok) {
                std::cerr << "HeckeBredonTreeValidation: MISMATCH at (n=" << n << ", ell=" << ell
                          << "): tree=(" << tree.h0 << "," << tree.h1 << "," << tree.h2
                          << ") pipeline dims=[";
                for (uint32_t q = 0; q < dims.size(); ++q) {
                    if (q) std::cerr << ",";
                    std::cerr << dims[q];
                }
                std::cerr << "]\n";
                ++failures;
            }
            ++cells_checked;
        }
    }

    std::cout << "HeckeBredonTreeValidation: " << cells_checked
              << " cells checked (ell in {2,3}, n in [0,40]), " << failures
              << " mismatches; " << (sizeof(kOracle) / sizeof(kOracle[0]))
              << " frozen golden oracle cells verified.\n";
    return failures;
}
