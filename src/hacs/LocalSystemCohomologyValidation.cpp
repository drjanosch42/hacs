// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#include "LocalSystemCohomologyValidation.h"

#include "SimplicialCohomology.h"
#include "LocalSystemSimplicialComplex.h"
#include "EquivariantCohomologyOnK.h"
#include "EquivariantCochainComplexMorphismOnK.h"
#include "../ModArithmetic/FlatMatrix.h"

#include "LocalSystemPresets.h"

#include <vector>
#include <map>
#include <utility>
#include <iostream>
#include <stdexcept>
#include <string>
#include <iomanip>
#include <functional>

// Helper function to make test output more consistent
void print_section_header(const std::string& title) {
    std::cout << "\n" << std::string(60, '-') << "\n";
    std::cout << "--- " << title << " ---" << std::endl;
    std::cout << std::string(60, '-') << "\n";
}

uint32_t LocalSystemCohomologyValidation(void) {
    uint32_t tests_failed = 0;
    auto run_test_group = [&](const std::string& group_name, std::function<void()> test_fn) {
        print_section_header(group_name);
        try {
            test_fn();
            std::cout << "--- " << group_name << ": PASSED ---" << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "\n*** ERROR in " << group_name << ": " << e.what() << std::endl;
            tests_failed++;
        }
    };

    // ... (previous test groups remain the same) ...
    run_test_group("EquivariantCohomologyOnK - Trivial System on K_Square", [&]() {
        uint32_t p = 3;
        uint32_t d_V = 1;
        std::cout << "  Parameters: p = " << p << ", d_V = " << d_V << std::endl;
        std::vector<std::vector<std::vector<uint32_t>>> square_simplices_data = {
            {{0}, {1}, {2}, {3}},
            {{0, 1}, {1, 2}, {2, 3}, {0, 3}, {0, 2}},
            {{0, 1, 2}, {0, 2, 3}}
        };
        SimplicialComplex K_square(square_simplices_data, "Square_K_Test");
        std::vector<LocalSystemSimplicialComplex::SimplexRelation> trivial_relations_list = {};
        LocalSystemSimplicialComplex lssc_trivial_on_K(K_square, trivial_relations_list, d_V, p);
        EquivariantCohomologyOnK eq_cohom_trivial_on_K(&lssc_trivial_on_K);
        eq_cohom_trivial_on_K.compute_cohomology();
        std::cout << eq_cohom_trivial_on_K.to_string_offset(2) << std::endl;
        if (eq_cohom_trivial_on_K.get_H_eq_dimensions().empty() || eq_cohom_trivial_on_K.get_H_eq_dimensions()[0] != d_V) {
            throw std::runtime_error("H0(K_Square; V_trivial) dimension mismatch. Expected " + std::to_string(d_V) +
                ", Got " + (eq_cohom_trivial_on_K.get_H_eq_dimensions().empty() ? "empty" : std::to_string(eq_cohom_trivial_on_K.get_H_eq_dimensions()[0])));
        }
        });

    run_test_group("EquivariantCohomologyOnK - Moebius Strip Presets", [&]() {
        uint32_t p_moeb = 3;
        uint32_t scalar_minus_one_moeb = p_moeb - 1;
        std::cout << "  Testing Moebius Strip with Sign Representation (scalar = -1 mod " << p_moeb << ")..." << std::endl;
        LocalSystemSimplicialComplex lssc_moebius_sign_p3 = LocalSystemPresets::CreateMoebiusScalar(scalar_minus_one_moeb, p_moeb);
        EquivariantCohomologyOnK eq_cohom_moebius_sign_p3(&lssc_moebius_sign_p3);
        eq_cohom_moebius_sign_p3.compute_cohomology();
        std::cout << eq_cohom_moebius_sign_p3.to_string_offset(4) << std::endl;
        const auto& dims_moeb_sign3 = eq_cohom_moebius_sign_p3.get_H_eq_dimensions();
        for (uint32_t q = 0; q < dims_moeb_sign3.size(); ++q) {
            if (dims_moeb_sign3[q] != 0) {
                throw std::runtime_error("H^" + std::to_string(q) + "(Moebius, Sign, p=3) expected 0, got " + std::to_string(dims_moeb_sign3[q]));
            }
        }

        p_moeb = 2;
        std::cout << "  Testing Moebius Strip with Sign Representation (scalar = -1 = 1 mod " << p_moeb << ")..." << std::endl;
        LocalSystemSimplicialComplex lssc_moebius_sign_p2 = LocalSystemPresets::CreateMoebiusScalar(1, p_moeb);
        EquivariantCohomologyOnK eq_cohom_moebius_sign_p2(&lssc_moebius_sign_p2);
        eq_cohom_moebius_sign_p2.compute_cohomology();
        std::cout << eq_cohom_moebius_sign_p2.to_string_offset(4) << std::endl;
        const auto& dims_moeb_sign2 = eq_cohom_moebius_sign_p2.get_H_eq_dimensions();
        if (dims_moeb_sign2.size() < 2 || dims_moeb_sign2[0] != 1 || dims_moeb_sign2[1] != 1) {
            throw std::runtime_error("H*(Moebius, Trivial, p=2) dimension mismatch.");
        }

        p_moeb = 5;
        std::cout << "  Testing Moebius Strip with Unipotent 2D Representation (p=" << p_moeb << ")..." << std::endl;
        LocalSystemSimplicialComplex lssc_moebius_unip = LocalSystemPresets::CreateMoebiusUnipotent2D(p_moeb);
        EquivariantCohomologyOnK eq_cohom_moebius_unip(&lssc_moebius_unip);
        eq_cohom_moebius_unip.compute_cohomology();
        std::cout << eq_cohom_moebius_unip.to_string_offset(4) << std::endl;
        const auto& dims_moeb_unip = eq_cohom_moebius_unip.get_H_eq_dimensions();
        if (dims_moeb_unip.size() < 2 || dims_moeb_unip[0] != 1 || dims_moeb_unip[1] != 1) {
            throw std::runtime_error("H*(Moebius, Unipotent2D, p=5) dimension mismatch.");
        }
        });

    run_test_group("EquivariantCohomologyOnK - S^1 Stabilizer Test", [&]() {
        uint32_t p_stab = 3;
        uint32_t d_V_stab = 2;
        std::cout << "  Testing S^1 with a Stabilizer at vertex 0 (p=" << p_stab << ", d_V=" << d_V_stab << ")..." << std::endl;
        std::vector<std::vector<std::vector<uint32_t>>> s1_simplices_data_stab = { {{0}, {1}}, {{0,1}} };
        SimplicialComplex K_s1_loop_stab(s1_simplices_data_stab, "S1_Loop_K_Stab");
        std::vector<LocalSystemSimplicialComplex::SimplexRelation> s1_stab_relations;
        FlatMatrix M_loop_id_stab(d_V_stab, p_stab);
        s1_stab_relations.push_back(LocalSystemSimplicialComplex::SimplexRelation(0, 1, 0, M_loop_id_stab, "S1_loop_id"));
        FlatMatrix M_stabilizer_stab(d_V_stab, d_V_stab, p_stab, std::vector<uint32_t>{ 1, 1, 0, 1 });
        s1_stab_relations.push_back(LocalSystemSimplicialComplex::SimplexRelation(0, 0, 0, M_stabilizer_stab, "Stabilizer_v0"));
        LocalSystemSimplicialComplex lssc_s1_stabilizer(K_s1_loop_stab, s1_stab_relations, d_V_stab, p_stab);
        EquivariantCohomologyOnK eq_cohom_s1_stabilizer(&lssc_s1_stabilizer);
        eq_cohom_s1_stabilizer.compute_cohomology();
        std::cout << eq_cohom_s1_stabilizer.to_string_offset(4) << std::endl;
        const auto& dims_s1_stab = eq_cohom_s1_stabilizer.get_H_eq_dimensions();
        if (dims_s1_stab.empty() || dims_s1_stab[0] != 1) {
            throw std::runtime_error("H0(S1_Stabilizer) dimension mismatch. Expected 1, got " +
                (dims_s1_stab.empty() ? "empty" : std::to_string(dims_s1_stab[0])));
        }
        });

    // --- Test EquivariantCochainComplexMorphismOnK ---
    run_test_group("EquivariantCochainComplexMorphismOnK - S1 Trivial <-> Unipotent", [&]() {
        uint32_t p = 5;
        // LSSC & Cohomology for S1 Trivial (1D)
        LocalSystemSimplicialComplex lssc_s1_triv = LocalSystemPresets::CreateS1Scalar(1, p, "_morph_test_triv");
        EquivariantCohomologyOnK eq_s1_triv(&lssc_s1_triv);
        eq_s1_triv.compute_cohomology();
        std::cout << "  Computed H*(S1, L_Trivial(1D), p=" << p << "):\n" << eq_s1_triv.to_string_offset(4);
        if (eq_s1_triv.get_H_eq_dimensions().size() < 2 || eq_s1_triv.get_H_eq_dimensions()[0] != 1 || eq_s1_triv.get_H_eq_dimensions()[1] != 1)
            throw std::runtime_error("S1 Trivial cohomology dimensions incorrect for morphism test.");

        // LSSC & Cohomology for S1 Unipotent (2D)
        LocalSystemSimplicialComplex lssc_s1_unip = LocalSystemPresets::CreateS1Unipotent2D(p, "_morph_test_unip");
        EquivariantCohomologyOnK eq_s1_unip(&lssc_s1_unip);
        eq_s1_unip.compute_cohomology();
        std::cout << "  Computed H*(S1, L_Unipotent2D, p=" << p << "):\n" << eq_s1_unip.to_string_offset(4);
        if (eq_s1_unip.get_H_eq_dimensions().size() < 2 || eq_s1_unip.get_H_eq_dimensions()[0] != 1 || eq_s1_unip.get_H_eq_dimensions()[1] != 1)
            throw std::runtime_error("S1 Unipotent cohomology dimensions incorrect for morphism test.");

        // Morphism 1: Unipotent -> Trivial (Projection F=[0,1])
        // H* map: H*(Unip) -> H*(Triv)
        std::cout << "\n  Testing Morphism S1_Unip -> S1_Triv (H* map: H*(Unip) -> H*(Triv))" << std::endl;
        EquivariantCochainComplexMorphismOnK morph_unip_to_triv =
            LocalSystemPresets::CreateS1_UnipotentToTrivialMorphism(&eq_s1_triv, &eq_s1_unip);
        morph_unip_to_triv.compute_induced_maps_on_cohomology();
        std::cout << morph_unip_to_triv.to_string_offset(4) << std::endl;

        const auto& maps_u2t = morph_unip_to_triv.get_induced_H_eq_maps();
        if (maps_u2t.size() < 2) throw std::runtime_error("Unip->Triv: Not enough H* maps computed.");
        // EXPECTED H0 map: [0]
        if (maps_u2t[0].get_rows() != 1 || maps_u2t[0].get_cols() != 1 || maps_u2t[0].get_entry(0, 0) != 0) { // <<< CHANGED EXPECTATION
            throw std::runtime_error("Unip->Triv: H0 map incorrect. Expected [[0]].");
        }
        // EXPECTED H1 map: [1]
        if (maps_u2t[1].get_rows() != 1 || maps_u2t[1].get_cols() != 1 || maps_u2t[1].get_entry(0, 0) != 1) {
            throw std::runtime_error("Unip->Triv: H1 map incorrect. Expected [[1]].");
        }

        // Morphism 2: Trivial -> Unipotent (Inclusion F=[[1],[0]])
        // H* map: H*(Triv) -> H*(Unip)
        std::cout << "\n  Testing Morphism S1_Triv -> S1_Unip (H* map: H*(Triv) -> H*(Unip))" << std::endl;
        EquivariantCochainComplexMorphismOnK morph_triv_to_unip =
            LocalSystemPresets::CreateS1_TrivialToUnipotentMorphism(&eq_s1_unip, &eq_s1_triv);
        morph_triv_to_unip.compute_induced_maps_on_cohomology();
        std::cout << morph_triv_to_unip.to_string_offset(4) << std::endl;

        const auto& maps_t2u = morph_triv_to_unip.get_induced_H_eq_maps();
        if (maps_t2u.size() < 2) throw std::runtime_error("Triv->Unip: Not enough H* maps computed.");
        // EXPECTED H0 map: [1]
        if (maps_t2u[0].get_rows() != 1 || maps_t2u[0].get_cols() != 1 || maps_t2u[0].get_entry(0, 0) != 1) {
            throw std::runtime_error("Triv->Unip: H0 map incorrect. Expected [[1]].");
        }
        // EXPECTED H1 map: [0]  // <<< ADDED/REFINED CHECK
        if (maps_t2u[1].get_rows() != 1 || maps_t2u[1].get_cols() != 1 || maps_t2u[1].get_entry(0, 0) != 0) {
            throw std::runtime_error("Triv->Unip: H1 map incorrect. Expected [[0]].");
        }
        });

    // --- Final Summary ---
    print_section_header("Validation Summary");
    if (tests_failed == 0) {
        std::cout << "All Local System Cohomology and Morphism tests PASSED." << std::endl;
        return 0; // Success
    }
    else {
        std::cout << tests_failed << " test group(s) FAILED." << std::endl;
        return 1; // Failure
    }
}
