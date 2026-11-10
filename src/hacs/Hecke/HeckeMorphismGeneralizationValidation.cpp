// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

/*
 * HeckeMorphismGeneralizationValidation.cpp  (Stage S6 self-check)
 *
 * Asserts that the per-simplex sparse-block-row generalization of
 * EquivariantCochainComplexMorphismOnK recovers the legacy constant-fiber-matrix
 * behaviour EXACTLY: for each test local-system pair on S^1, the morphism is built
 * two ways (legacy ctor vs. an explicitly-constructed one-entry FiberBlockTable)
 * and the induced maps on equivariant cohomology are compared elementwise in all
 * degrees. See the header for the full statement.
 */

#include "HeckeMorphismGeneralizationValidation.h"

#include "../EquivariantCochainComplexMorphismOnK.h"
#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemPresets.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"
#include "../../ModArithmetic/Matrix.h"
#include "../../ModArithmetic/FlatMatrix.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <vector>

namespace {

// Independent replica of the vertex-list match used by the class internally to
// map a simplex of K1 to its index in K2 (deliberately NOT calling the class's
// private find_simplex_K1_idx_in_K2). Returns -1 if not found.
int32_t find_source_idx(const SimplicialComplex& K1, const SimplicialComplex& K2,
                        uint32_t q, uint32_t s1_idx) {
    if (q >= K1.get_simplices().size() || s1_idx >= K1.get_simplices()[q].size()) return -1;
    const std::vector<uint32_t>& verts = K1.get_simplices()[q][s1_idx];
    if (q >= K2.get_simplices().size()) return -1;
    const auto& K2q = K2.get_simplices()[q];
    auto it = std::find(K2q.begin(), K2q.end(), verts);
    if (it == K2q.end()) return -1;
    return static_cast<int32_t>(std::distance(K2q.begin(), it));
}

// Compares two induced-H*-map vectors elementwise; returns true iff identical.
bool induced_maps_identical(const std::vector<FlatMatrix>& A, const std::vector<FlatMatrix>& B,
                            std::ostream& log) {
    if (A.size() != B.size()) {
        log << "    MISMATCH: number of degrees " << A.size() << " vs " << B.size() << "\n";
        return false;
    }
    for (size_t q = 0; q < A.size(); ++q) {
        if (A[q].get_rows() != B[q].get_rows() || A[q].get_cols() != B[q].get_cols()) {
            log << "    MISMATCH at q=" << q << ": dims " << A[q].get_rows() << "x" << A[q].get_cols()
                << " vs " << B[q].get_rows() << "x" << B[q].get_cols() << "\n";
            return false;
        }
        for (uint32_t i = 0; i < A[q].get_rows(); ++i) {
            for (uint32_t j = 0; j < A[q].get_cols(); ++j) {
                if (A[q].get_entry(i, j) != B[q].get_entry(i, j)) {
                    log << "    MISMATCH at q=" << q << " (" << i << "," << j << "): "
                        << A[q].get_entry(i, j) << " vs " << B[q].get_entry(i, j) << "\n";
                    return false;
                }
            }
        }
    }
    return true;
}

// Runs the two-way build for one (p, M1, M2, F) case; returns true iff identical.
bool run_case(uint32_t p, const std::vector<std::vector<uint32_t>>& M1_data,
              const std::vector<std::vector<uint32_t>>& M2_data,
              const std::vector<std::vector<uint32_t>>& F_data,
              const std::string& tag) {
    const uint32_t d_V = 2;
    Matrix M1(d_V, d_V, p, M1_data);
    Matrix M2(d_V, d_V, p, M2_data);
    Matrix F(d_V, d_V, p, F_data); // F: V2 -> V1 (d_V1 x d_V2 = 2x2)
    // (matrix-engine Phase 3) the class takes a FlatMatrix fiber map now.

    // Two S^1 local systems (same K_Interval structure), with these monodromies.
    LocalSystemSimplicialComplex lssc1 =
        LocalSystemPresets::CreateS1Generic(M1, p, d_V, "_S6gen_dom_" + tag);
    LocalSystemSimplicialComplex lssc2 =
        LocalSystemPresets::CreateS1Generic(M2, p, d_V, "_S6gen_cod_" + tag);

    EquivariantCohomologyOnK coh1(&lssc1); coh1.compute_cohomology(); // L1 (target of H* map)
    EquivariantCohomologyOnK coh2(&lssc2); coh2.compute_cohomology(); // L2 (source of H* map)

    const SimplicialComplex* K1 = coh1.get_lssc()->get_base_K_complex();
    const SimplicialComplex* K2 = coh2.get_lssc()->get_base_K_complex();

    // --- Build 1: legacy constant-fiber-matrix constructor. ---
    SimplicialComplexMorphism scm(K1, K2);
    EquivariantCochainComplexMorphismOnK morph_legacy(&coh1, &coh2, &scm, F);
    morph_legacy.compute_induced_maps_on_cohomology();

    // --- Build 2: explicit one-entry FiberBlockTable (independent construction). ---
    FiberBlockTable table;
    uint32_t num_deg = static_cast<uint32_t>(K1->get_simplices().size());
    table.assign(num_deg, std::vector<std::vector<PerSimplexBlock>>());
    FlatMatrix F_flat(F);
    for (uint32_t q = 0; q < num_deg; ++q) {
        uint32_t Nq = K1->get_num_q_simplices(q);
        table[q].assign(Nq, std::vector<PerSimplexBlock>());
        for (uint32_t s1 = 0; s1 < Nq; ++s1) {
            int32_t s2 = find_source_idx(*K1, *K2, q, s1);
            if (s2 == -1) continue; // empty block row
            table[q][s1].push_back(PerSimplexBlock{ static_cast<uint32_t>(s2), F_flat });
        }
    }
    EquivariantCochainComplexMorphismOnK morph_block(&coh1, &coh2, std::move(table), "explicit_one_entry_" + tag);
    morph_block.compute_induced_maps_on_cohomology();

    // --- Compare. ---
    std::ostringstream log;
    bool ok = induced_maps_identical(
        morph_legacy.get_induced_H_eq_maps(), morph_block.get_induced_H_eq_maps(), log);
    std::cout << "  [p=" << p << ", " << tag << "] "
              << (ok ? "IDENTICAL" : "MISMATCH") << "\n";
    if (!ok) std::cout << log.str();
    return ok;
}

// Fixed-seed LCG for reproducible pseudo-random monodromies / fiber maps.
struct Lcg {
    uint64_t s;
    explicit Lcg(uint64_t seed) : s(seed) {}
    uint32_t next() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<uint32_t>(s >> 33);
    }
};

} // namespace

uint32_t HeckeMorphismGeneralizationValidation() {
    std::cout << "\n=== HeckeMorphismGeneralizationValidation (Stage S6 self-check) ===\n";
    std::cout << "Legacy ctor vs. explicit one-entry FiberBlockTable; induced H*-maps must be elementwise identical.\n";

    uint32_t failures = 0;
    const std::vector<uint32_t> primes = { 5, 101 };

    for (uint32_t p : primes) {
        // A few structured pairs with non-trivial cohomology (identity, unipotent),
        // exercising nonzero F * h2(s2) multiplies.
        if (!run_case(p,
                {{1,0},{0,1}}, {{1,0},{0,1}}, {{2 % p, 3 % p},{1 % p, 4 % p}}, "id_id")) ++failures;
        if (!run_case(p,
                {{1,1},{0,1}}, {{1,1},{0,1}}, {{0,1},{1,0}}, "unip_unip")) ++failures;
        if (!run_case(p,
                {{1,1},{0,1}}, {{1,0},{0,1}}, {{1,0},{0,1}}, "unip_id")) ++failures;

        // Fixed-seed random pairs.
        Lcg rng(0x9E3779B97F4A7C15ull ^ static_cast<uint64_t>(p));
        for (int trial = 0; trial < 4; ++trial) {
            auto r = [&]() { return rng.next() % p; };
            std::vector<std::vector<uint32_t>> M1d = {{r(), r()},{r(), r()}};
            std::vector<std::vector<uint32_t>> M2d = {{r(), r()},{r(), r()}};
            std::vector<std::vector<uint32_t>> Fd  = {{r(), r()},{r(), r()}};
            std::ostringstream tg; tg << "rand" << trial;
            if (!run_case(p, M1d, M2d, Fd, tg.str())) ++failures;
        }
    }

    if (failures == 0) {
        std::cout << "HeckeMorphismGeneralizationValidation: PASS (all cases elementwise identical).\n";
    } else {
        std::cout << "HeckeMorphismGeneralizationValidation: FAIL (" << failures << " mismatching case(s)).\n";
    }
    return failures;
}
