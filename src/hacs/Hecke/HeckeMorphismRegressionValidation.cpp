// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

/*
 * HeckeMorphismRegressionValidation.cpp  (Stage S6V -- independent validation
 * agent)
 *
 * See the header for the full oracle statement, the capture method, and the
 * two commit hashes. Summary:
 *
 *   Part 1: get_induced_H_eq_maps() of the three S1 preset morphisms and the
 *   four TestSL2Z_To_S1Cusp_RestrictionMorphism(n,ell) restriction morphisms
 *   must be BIT-IDENTICAL to reference values captured, via a throwaway
 *   capture driver built only from the class's PUBLIC API, on the pre-S6
 *   parent commit 41df5e278460f4c1a9f7ee3270878052a0802983.
 *
 *   Part 2: on the S6 tree only, the legacy constant-fiber-matrix ctor and an
 *   explicitly-built one-entry FiberBlockTable must produce elementwise-
 *   identical induced maps for 100 fixed-seed random 2-dimensional local
 *   system pairs on S^1.
 */

#include "HeckeMorphismRegressionValidation.h"

#include "../EquivariantCochainComplexMorphismOnK.h"
#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemPresets.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemUtils.h"
#include "../SimplicialCohomology.h"
#include "../../ModArithmetic/FlatMatrix.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

// =============================================================================
// Part 1: cross-commit bit-identity against reference values captured on the
// parent commit 41df5e278460f4c1a9f7ee3270878052a0802983 (see the header for
// the exact capture commands). Transcribed VERBATIM from the capture driver's
// stdout ("CASE ... / NDEG n / DEG q ROWS r COLS c / <r rows> / ENDCASE").
// =============================================================================

// One degree's worth of an expected induced map: rows x cols, row-major.
struct RefDeg {
    uint32_t rows;
    uint32_t cols;
    std::vector<std::vector<uint32_t>> data; // data.size() == rows, data[i].size() == cols
};

struct RefCase {
    std::string name;
    uint32_t p;
    std::vector<RefDeg> degs; // one per cohomological degree q = 0, 1, 2, ...
};

// --- The three LocalSystemPresets S1 morphisms (p = 5), exactly as built by
//     LocalSystemCohomologyValidation()'s "EquivariantCochainComplexMorphismOnK
//     - S1 Trivial <-> Unipotent" test group (plus the identity preset). ---

const RefCase kRefPresetUnipToTriv{
    "preset_unip_to_triv", 5,
    { RefDeg{1, 1, {{0}}},
      RefDeg{1, 1, {{1}}} }
};

const RefCase kRefPresetTrivToUnip{
    "preset_triv_to_unip", 5,
    { RefDeg{1, 1, {{1}}},
      RefDeg{1, 1, {{0}}} }
};

const RefCase kRefPresetIdentityUnip{
    "preset_identity_unip", 5,
    { RefDeg{1, 1, {{1}}},
      RefDeg{1, 1, {{1}}} }
};

// --- TestSL2Z_To_S1Cusp_RestrictionMorphism(n, ell)'s internal restriction
//     morphism, for the four required (n, ell) pairs. ---

const RefCase kRefSl2zN8Ell5{
    "sl2z_restrict_n8_ell5", 5,
    { RefDeg{2, 0, {{}, {}}},
      RefDeg{2, 1, {{0}, {1}}},
      RefDeg{0, 0, {}} }
};

const RefCase kRefSl2zN10Ell5{
    "sl2z_restrict_n10_ell5", 5,
    { RefDeg{3, 0, {{}, {}, {}}},
      RefDeg{3, 3, {{0, 0, 1}, {0, 4, 0}, {1, 0, 0}}},
      RefDeg{0, 0, {}} }
};

const RefCase kRefSl2zN10Ell691{
    "sl2z_restrict_n10_ell691", 691,
    { RefDeg{1, 0, {{}}},
      RefDeg{1, 3, {{1, 0, 0}}},
      RefDeg{0, 0, {}} }
};

const RefCase kRefSl2zN100Ell5{
    "sl2z_restrict_n100_ell5", 5,
    { RefDeg{21, 2, {
          {0, 1}, {0, 0}, {1, 0}, {0, 0}, {0, 1}, {0, 0}, {4, 0}, {0, 0}, {0, 1}, {0, 0},
          {0, 0}, {0, 0}, {0, 1}, {0, 0}, {4, 0}, {0, 0}, {0, 1}, {0, 0}, {1, 0}, {0, 0},
          {0, 1}
      }},
      RefDeg{21, 19, {
          {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0},
          {0,0,0,1,0,0,0,2,0,4,0,0,0,3,0,3,0,2,4},
          {0,0,1,0,0,0,0,0,0,0,3,0,0,0,0,0,0,0,0},
          {0,0,0,0,0,1,0,0,0,4,0,0,0,0,0,0,0,0,4},
          {0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
          {0,0,0,2,0,0,0,2,0,0,0,0,0,0,0,2,0,0,2},
          {0,0,2,0,0,0,0,0,0,0,4,0,0,0,1,0,0,0,0},
          {0,0,0,0,0,2,0,0,0,1,0,0,0,2,0,0,0,0,0},
          {0,0,0,0,2,0,0,0,3,0,0,0,0,0,0,0,0,0,0},
          {0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
          {0,0,3,0,0,0,0,0,0,0,3,0,0,0,3,0,0,0,0},
          {0,0,0,0,0,3,0,0,0,1,0,0,0,4,0,0,0,2,0},
          {0,0,0,0,3,0,0,0,4,0,0,0,3,0,0,0,0,0,0},
          {0,0,0,3,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0},
          {0,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
          {0,0,0,0,0,4,0,0,0,4,0,0,0,4,0,0,0,4,0},
          {0,0,0,0,4,0,0,0,3,0,0,0,2,0,0,0,1,0,0},
          {0,0,0,4,0,0,0,2,0,0,0,4,0,0,0,0,0,0,0},
          {0,0,4,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0},
          {0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
          {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}
      }},
      RefDeg{0, 0, {}} }
};

// Compares a computed induced-map vector against a RefCase; logs the FIRST
// mismatch (dims or entries) with degree, row, col, expected/got. Returns
// true iff every degree matches exactly.
bool matches_reference(const std::vector<FlatMatrix>& got, const RefCase& ref, std::ostream& log) {
    if (got.size() != ref.degs.size()) {
        log << "  REGRESSION in case '" << ref.name << "': number of degrees "
            << got.size() << " (computed) vs " << ref.degs.size() << " (reference).\n";
        return false;
    }
    bool ok = true;
    for (uint32_t q = 0; q < got.size(); ++q) {
        const FlatMatrix& M = got[q];
        const RefDeg& R = ref.degs[q];
        if (M.get_rows() != R.rows || M.get_cols() != R.cols) {
            log << "  REGRESSION in case '" << ref.name << "', degree q=" << q
                << ": dims " << M.get_rows() << "x" << M.get_cols() << " (computed) vs "
                << R.rows << "x" << R.cols << " (reference).\n";
            ok = false;
            continue;
        }
        for (uint32_t i = 0; i < M.get_rows(); ++i) {
            for (uint32_t j = 0; j < M.get_cols(); ++j) {
                if (M.get_entry(i, j) != R.data[i][j]) {
                    log << "  REGRESSION in case '" << ref.name << "', degree q=" << q
                        << ", entry (" << i << "," << j << "): computed "
                        << M.get_entry(i, j) << " vs reference " << R.data[i][j] << ".\n";
                    ok = false;
                }
            }
        }
    }
    return ok;
}

// --- Reconstruction on the S6 tree, using ONLY the class's public API and
//     the SAME sequence of calls the capture driver used on the parent
//     commit (see the header). These are independent replicas -- they do NOT
//     call TestSL2Z_To_S1Cusp_RestrictionMorphism (void, cannot hand back its
//     internal morphism object) or LocalSystemCohomologyValidation. ---

std::vector<FlatMatrix> build_preset_unip_to_triv() {
    const uint32_t p = 5;
    LocalSystemSimplicialComplex lssc_s1_triv = LocalSystemPresets::CreateS1Scalar(1, p, "_morph_test_triv");
    EquivariantCohomologyOnK eq_s1_triv(&lssc_s1_triv);
    eq_s1_triv.compute_cohomology();

    LocalSystemSimplicialComplex lssc_s1_unip = LocalSystemPresets::CreateS1Unipotent2D(p, "_morph_test_unip");
    EquivariantCohomologyOnK eq_s1_unip(&lssc_s1_unip);
    eq_s1_unip.compute_cohomology();

    EquivariantCochainComplexMorphismOnK morph =
        LocalSystemPresets::CreateS1_UnipotentToTrivialMorphism(&eq_s1_triv, &eq_s1_unip);
    morph.compute_induced_maps_on_cohomology();
    return morph.get_induced_H_eq_maps();
}

std::vector<FlatMatrix> build_preset_triv_to_unip() {
    const uint32_t p = 5;
    LocalSystemSimplicialComplex lssc_s1_triv = LocalSystemPresets::CreateS1Scalar(1, p, "_morph_test_triv");
    EquivariantCohomologyOnK eq_s1_triv(&lssc_s1_triv);
    eq_s1_triv.compute_cohomology();

    LocalSystemSimplicialComplex lssc_s1_unip = LocalSystemPresets::CreateS1Unipotent2D(p, "_morph_test_unip");
    EquivariantCohomologyOnK eq_s1_unip(&lssc_s1_unip);
    eq_s1_unip.compute_cohomology();

    EquivariantCochainComplexMorphismOnK morph =
        LocalSystemPresets::CreateS1_TrivialToUnipotentMorphism(&eq_s1_unip, &eq_s1_triv);
    morph.compute_induced_maps_on_cohomology();
    return morph.get_induced_H_eq_maps();
}

std::vector<FlatMatrix> build_preset_identity_unip() {
    const uint32_t p = 5;
    LocalSystemSimplicialComplex lssc_s1_unip = LocalSystemPresets::CreateS1Unipotent2D(p, "_morph_test_unip");
    EquivariantCohomologyOnK eq_s1_unip(&lssc_s1_unip);
    eq_s1_unip.compute_cohomology();

    EquivariantCochainComplexMorphismOnK morph =
        LocalSystemPresets::CreateS1_IdentityMorphism(&eq_s1_unip);
    morph.compute_induced_maps_on_cohomology();
    return morph.get_induced_H_eq_maps();
}

// Replica of TestSL2Z_To_S1Cusp_RestrictionMorphism(n_sym, p_mod)'s internal
// construction (src/hacs/LocalSystemSL2Z.cpp), built from
// ONLY the public API (CreateSL2ZFundamentalDomainTriangulation,
// ComputeSymmetricPowerMatrix, GenerateSL2ZRelations_7VertexModel,
// LocalSystemPresets::CreateS1Generic, find_simplex_idx, make_edge,
// SimplicialComplexMorphism) since that function itself is void/stdout-only.
std::vector<FlatMatrix> build_sl2z_restriction(uint32_t n_sym, uint32_t p_mod) {
    SimplicialComplex K_sl2z = CreateSL2ZFundamentalDomainTriangulation();
    uint32_t d_V = n_sym + 1;

    FlatMatrix T0(2, 2, p_mod, std::vector<uint32_t>{ 1, 1, 0, 1 });
    FlatMatrix S0(2, 2, p_mod, std::vector<uint32_t>{ 0, p_mod - 1, 1, 0 });
    FlatMatrix M_T = ComputeSymmetricPowerMatrixFM(T0, n_sym);
    FlatMatrix M_S = ComputeSymmetricPowerMatrixFM(S0, n_sym);
    FlatMatrix ST0 = S0 * T0;
    FlatMatrix M_ST = ComputeSymmetricPowerMatrixFM(ST0, n_sym);
    FlatMatrix TS0 = T0 * S0;
    FlatMatrix M_TS = ComputeSymmetricPowerMatrixFM(TS0, n_sym);

    std::vector<LocalSystemSimplicialComplex::SimplexRelation> relations_sl2z =
        GenerateSL2ZRelations_7VertexModel(K_sl2z, n_sym, p_mod, M_T, M_S, M_ST, M_TS);

    LocalSystemSimplicialComplex lssc_sl2z(K_sl2z, relations_sl2z, d_V, p_mod);
    EquivariantCohomologyOnK eq_cohom_sl2z(&lssc_sl2z);
    eq_cohom_sl2z.compute_cohomology();

    // (eq_cohom_s1_cusp on the generic K_Interval is built and computed by
    // TestSL2Z_To_S1Cusp_RestrictionMorphism too, but its result is never fed
    // into the restriction morphism -- only printed. It is intentionally
    // omitted here since it cannot affect get_induced_H_eq_maps().)

    std::vector<std::vector<std::vector<uint32_t>>> s1_path_simplices_data = {
        {{0},{1},{2},{3}},
        {make_edge(0,1), make_edge(1,2), make_edge(2,3)}
    };
    SimplicialComplex K_s1_path_representation(s1_path_simplices_data, "K_S1_Path_Repr_For_SL2Z");

    LocalSystemSimplicialComplex lssc_s1_cusp_on_path_for_morph(
        K_s1_path_representation,
        { LocalSystemSimplicialComplex::SimplexRelation(0,
            find_simplex_idx(K_s1_path_representation, 0, {3}),
            find_simplex_idx(K_s1_path_representation, 0, {0}),
            M_T,
            "S1_cusp_loop_on_path_for_morph") },
        d_V, p_mod
    );
    EquivariantCohomologyOnK eq_cohom_s1_cusp_on_path_for_morph(&lssc_s1_cusp_on_path_for_morph);
    eq_cohom_s1_cusp_on_path_for_morph.compute_cohomology();

    SimplicialComplexMorphism scm_s1_into_sl2z(&K_s1_path_representation, &K_sl2z);
    if (!scm_s1_into_sl2z.is_valid()) {
        throw std::runtime_error("build_sl2z_restriction: SimplicialComplexMorphism invalid for n=" +
            std::to_string(n_sym) + " ell=" + std::to_string(p_mod));
    }

    FlatMatrix F_identity(d_V, p_mod);

    EquivariantCochainComplexMorphismOnK restriction_morph(
        &eq_cohom_s1_cusp_on_path_for_morph,
        &eq_cohom_sl2z,
        &scm_s1_into_sl2z,
        F_identity
    );
    restriction_morph.compute_induced_maps_on_cohomology();
    return restriction_morph.get_induced_H_eq_maps();
}

// Runs one Part-1 case: compares an already-computed induced map against its
// reference, prints PASS/REGRESSION, and returns 1 on mismatch (0 on match).
uint32_t check_case(const std::string& label, const std::vector<FlatMatrix>& computed, const RefCase& ref) {
    std::ostringstream log;
    bool ok = matches_reference(computed, ref, log);
    std::cout << "  [Part 1] " << label << " (p=" << ref.p << "): "
              << (ok ? "BIT-IDENTICAL to parent commit" : "REGRESSION DETECTED") << "\n";
    if (!ok) std::cout << log.str();
    return ok ? 0 : 1;
}

// =============================================================================
// Part 2: legacy ctor vs. explicit one-entry FiberBlockTable, 100 fixed seeds
// (S6-tree-only equivalence; no parent commit involved -- FiberBlockTable is
// an S6 addition).
// =============================================================================

// Independent replica of the vertex-list match the class uses internally to
// map a simplex of K1 to its index in K2 (K1 == K2 == K_Interval here, so
// this is effectively the identity on indices, but implemented from scratch
// rather than calling the class's private find_simplex_K1_idx_in_K2).
int32_t find_source_idx(const SimplicialComplex& K1, const SimplicialComplex& K2, uint32_t q, uint32_t s1_idx) {
    if (q >= K1.get_simplices().size() || s1_idx >= K1.get_simplices()[q].size()) return -1;
    const std::vector<uint32_t>& verts = K1.get_simplices()[q][s1_idx];
    if (q >= K2.get_simplices().size()) return -1;
    const auto& K2q = K2.get_simplices()[q];
    auto it = std::find(K2q.begin(), K2q.end(), verts);
    if (it == K2q.end()) return -1;
    return static_cast<int32_t>(std::distance(K2q.begin(), it));
}

bool induced_maps_identical(const std::vector<FlatMatrix>& A, const std::vector<FlatMatrix>& B, std::ostream& log) {
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

// Fixed-seed xorshift64* PRNG, deliberately a different generator from Stage
// S6's own self-check (which uses an LCG) so this is a genuinely separate
// pseudo-random stream, not a re-derivation of the same sequence.
struct Xorshift64Star {
    uint64_t s;
    explicit Xorshift64Star(uint64_t seed) : s(seed ? seed : 0xA5A5A5A5A5A5A5A5ull) {}
    uint32_t next() {
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        return static_cast<uint32_t>((s * 0x2545F4914F6CDD1Dull) >> 33);
    }
};

// Builds a 2x2 matrix over F_p with entries reduced mod p from an RNG.
FlatMatrix random_2x2(uint32_t p, Xorshift64Star& rng) {
    auto r = [&]() { return static_cast<uint32_t>(rng.next() % p); };
    // Evaluation order of the four r() calls is fixed by list-initialisation
    // of the vector, left to right -- the same order the old
    // Matrix(2, 2, p, {{r(), r()}, {r(), r()}}) nested-list form had.
    return FlatMatrix(2, 2, p, std::vector<uint32_t>{ r(), r(), r(), r() });
}

// One seed's worth of the equivalence check: legacy ctor vs. explicit
// one-entry FiberBlockTable on two independent S1-generic local systems.
// Returns true iff get_induced_H_eq_maps() are elementwise identical.
bool run_seed_case(uint32_t seed_idx, uint32_t p, std::ostream& log) {
    Xorshift64Star rng(0xD1B54A32D192ED03ull ^ (static_cast<uint64_t>(seed_idx) * 0x9E3779B97F4A7C15ull) ^ p);

    FlatMatrix M1 = random_2x2(p, rng); // domain-side monodromy (target of H* map)
    FlatMatrix M2 = random_2x2(p, rng); // codomain-side monodromy (source of H* map)
    FlatMatrix F  = random_2x2(p, rng); // fiber map V2 -> V1

    std::ostringstream tag; tag << "seed" << seed_idx;
    LocalSystemSimplicialComplex lssc1 = LocalSystemPresets::CreateS1Generic(M1, p, 2, "_S6V_dom_" + tag.str());
    LocalSystemSimplicialComplex lssc2 = LocalSystemPresets::CreateS1Generic(M2, p, 2, "_S6V_cod_" + tag.str());

    EquivariantCohomologyOnK coh1(&lssc1); coh1.compute_cohomology();
    EquivariantCohomologyOnK coh2(&lssc2); coh2.compute_cohomology();

    const SimplicialComplex* K1 = coh1.get_lssc()->get_base_K_complex();
    const SimplicialComplex* K2 = coh2.get_lssc()->get_base_K_complex();

    // Build 1: legacy constant-fiber-matrix constructor.
    SimplicialComplexMorphism scm(K1, K2);
    EquivariantCochainComplexMorphismOnK morph_legacy(&coh1, &coh2, &scm, F);
    morph_legacy.compute_induced_maps_on_cohomology();

    // Build 2: explicit one-entry FiberBlockTable, s2 found by an
    // INDEPENDENT vertex-list match (not the class's private helper).
    FiberBlockTable table;
    uint32_t num_deg = static_cast<uint32_t>(K1->get_simplices().size());
    table.assign(num_deg, std::vector<std::vector<PerSimplexBlock>>());
    const FlatMatrix& F_flat = F;
    for (uint32_t q = 0; q < num_deg; ++q) {
        uint32_t Nq = K1->get_num_q_simplices(q);
        table[q].assign(Nq, std::vector<PerSimplexBlock>());
        for (uint32_t s1 = 0; s1 < Nq; ++s1) {
            int32_t s2 = find_source_idx(*K1, *K2, q, s1);
            if (s2 == -1) continue;
            table[q][s1].push_back(PerSimplexBlock{ static_cast<uint32_t>(s2), F_flat });
        }
    }
    EquivariantCochainComplexMorphismOnK morph_block(&coh1, &coh2, std::move(table), "S6V_seed_" + tag.str());
    morph_block.compute_induced_maps_on_cohomology();

    bool ok = induced_maps_identical(morph_legacy.get_induced_H_eq_maps(), morph_block.get_induced_H_eq_maps(), log);
    if (!ok) {
        log << "  seed=" << seed_idx << " p=" << p << " M1=" << M1.to_string()
            << " M2=" << M2.to_string() << " F=" << F.to_string();
    }
    return ok;
}

} // namespace

uint32_t HeckeMorphismRegressionValidation() {
    std::cout << "\n=== HeckeMorphismRegressionValidation (Stage S6V, independent validation agent) ===\n";
    std::cout << "Parent commit:  41df5e278460f4c1a9f7ee3270878052a0802983 (S5 + S5V, branch hecke-action)\n";
    std::cout << "S6 tree:        same commit + Stage S6's uncommitted edits to\n";
    std::cout << "                EquivariantCochainComplexMorphismOnK.{h,cpp} (see this file's header\n";
    std::cout << "                comment for the exact capture commands and worktree procedure).\n";

    uint32_t failures = 0;

    // --- Part 1: cross-commit bit-identity. ---
    std::cout << "\n-- Part 1: bit-identity of get_induced_H_eq_maps() vs. the parent commit --\n";
    try {
        failures += check_case("preset_unip_to_triv", build_preset_unip_to_triv(), kRefPresetUnipToTriv);
        failures += check_case("preset_triv_to_unip", build_preset_triv_to_unip(), kRefPresetTrivToUnip);
        failures += check_case("preset_identity_unip", build_preset_identity_unip(), kRefPresetIdentityUnip);

        failures += check_case("sl2z_restrict(n=8,ell=5)", build_sl2z_restriction(8, 5), kRefSl2zN8Ell5);
        failures += check_case("sl2z_restrict(n=10,ell=5)", build_sl2z_restriction(10, 5), kRefSl2zN10Ell5);
        failures += check_case("sl2z_restrict(n=10,ell=691)", build_sl2z_restriction(10, 691), kRefSl2zN10Ell691);
        failures += check_case("sl2z_restrict(n=100,ell=5)", build_sl2z_restriction(100, 5), kRefSl2zN100Ell5);
    } catch (const std::exception& e) {
        std::cout << "  Part 1 THREW: " << e.what() << "\n";
        ++failures;
    }

    // --- Part 2: legacy ctor vs. explicit FiberBlockTable, 100 fixed seeds. ---
    std::cout << "\n-- Part 2: legacy ctor vs. explicit one-entry FiberBlockTable, 100 fixed seeds --\n";
    uint32_t part2_failures = 0;
    const std::vector<uint32_t> primes = { 5, 691 };
    try {
        for (uint32_t p : primes) {
            for (uint32_t seed_idx = 0; seed_idx < 50; ++seed_idx) {
                std::ostringstream log;
                if (!run_seed_case(seed_idx, p, log)) {
                    ++part2_failures;
                    std::cout << "  MISMATCH (p=" << p << ", seed=" << seed_idx << "):\n" << log.str();
                }
            }
        }
    } catch (const std::exception& e) {
        std::cout << "  Part 2 THREW: " << e.what() << "\n";
        ++part2_failures;
    }
    std::cout << "  Ran " << (primes.size() * 50) << " fixed-seed cases ("
              << (part2_failures == 0 ? "all identical" : "mismatches found") << ").\n";
    failures += part2_failures;

    if (failures == 0) {
        std::cout << "\nHeckeMorphismRegressionValidation: PASS "
                     "(no regression vs. parent commit; legacy/FiberBlockTable equivalence holds on 100 seeds).\n";
    } else {
        std::cout << "\nHeckeMorphismRegressionValidation: FAIL (" << failures << " failing case(s)).\n";
    }
    return failures;
}
