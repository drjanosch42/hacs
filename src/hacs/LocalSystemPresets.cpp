// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#include "LocalSystemPresets.h"
#include "LocalSystemUtils.h"
#include "SimplicialCohomology.h"
#include "../ModArithmetic/FlatMatrix.h"
#include "../ModArithmetic/ModCtx.h"

#include <vector>
#include <map>
#include <stdexcept>
#include <sstream>
#include <memory> // For std::unique_ptr if needed for SCM

// --- Helper SimplicialComplex K definitions (from previous response) ---
namespace {
    SimplicialComplex GetKInterval() {
        // Note: This creates a *new copy* of K_Interval each time it's called in a preset
        // if the static object is not managed carefully (e.g. if presets modify K_int_instance).
        // For read-only use, this is fine.
        // Consider making K_int_instance const if it's truly shared and never modified.
        // Or ensure LSSC makes its own deep copy. (LSSC does make a copy).
        static const SimplicialComplex K_int( // Make it const to emphasize it's a template
            { {{0}, {1}, {2}, {3}}, {{0, 1}, {1, 2}, {2, 3}} },
            "K_Interval_v0123_Static" // Give it a distinct name
        );
        return K_int;
    }

    SimplicialComplex GetKRectangle() {
        static const SimplicialComplex K_rect(
            {
                {{0},{1},{2},{3},{4},{5}},
                {{0,1},{0,2},{1,2},{1,3},{2,3},{2,4},{3,4},{3,5},{4,5}},
                {{0,1,2},{1,2,3},{2,3,4},{3,4,5}}
            },
            "K_Rectangle_MoebiusTex_Static"
        );
        return K_rect;
    }
} // anonymous namespace


// --- LSSC Creators (from previous response, slightly adapted if needed) ---
namespace LocalSystemPresets {

    LocalSystemSimplicialComplex CreateS1Generic(const FlatMatrix& M_loop, uint32_t p, uint32_t d_V, const std::string& S1_variant_name) {
        if (M_loop.get_rows() != d_V || M_loop.get_cols() != d_V || M_loop.get_p() != p) {
            throw std::runtime_error("CreateS1Generic: M_loop dimensions or prime mismatch.");
        }
        SimplicialComplex K_int_instance = GetKInterval(); // Gets a copy
        std::vector<LocalSystemSimplicialComplex::SimplexRelation> relations;
        relations.emplace_back(
            0, find_simplex_idx(K_int_instance, 0, { 3 }), find_simplex_idx(K_int_instance, 0, { 0 }),
            M_loop, S1_variant_name + "_loop_closure"
        );
        return LocalSystemSimplicialComplex(K_int_instance, relations, d_V, p);
    }

    LocalSystemSimplicialComplex CreateS1Scalar(uint32_t scalar, uint32_t p, const std::string& name_suffix) {
        uint32_t d_V = 1;
        // (matrix-engine Phase 3 Stage B) The old spelling was
        // `ModArithmetic::mod_add(scalar, 0, p)`, whose only effect is the ONE
        // conditional subtraction that maps [0, 2p) to [0, p) -- literally
        // engine::ModCtx<uint32_t>::canonicalise (ENGINE_API.md Sec. 1), which
        // is the same expression `v >= p ? v - p : v`. Bit-identical for every
        // uint32_t input, the constant folded away for the usual scalar < p.
        const uint32_t s = engine::ModCtx<uint32_t>(p).canonicalise(scalar);
        FlatMatrix M_scalar(d_V, d_V, p, std::vector<uint32_t>{ s });
        std::ostringstream name;
        name << "S1_Scalar(" << s << ")" << name_suffix;
        return CreateS1Generic(M_scalar, p, d_V, name.str());
    }

    LocalSystemSimplicialComplex CreateS1Unipotent2D(uint32_t p, const std::string& name_suffix) {
        uint32_t d_V = 2;
        FlatMatrix M_unip(d_V, d_V, p, std::vector<uint32_t>{ 1, 1, 0, 1 });
        std::ostringstream name;
        name << "S1_Unipotent2D" << name_suffix;
        return CreateS1Generic(M_unip, p, d_V, name.str());
    }

    LocalSystemSimplicialComplex CreateCylinderGeneric(const FlatMatrix& M_loop, uint32_t p, uint32_t d_V, const std::string& cyl_variant_name) {
        if (M_loop.get_rows() != d_V || M_loop.get_cols() != d_V || M_loop.get_p() != p) {
            throw std::runtime_error("CreateCylinderGeneric: M_loop dimensions or prime mismatch.");
        }
        SimplicialComplex K_rect_instance = GetKRectangle(); // Gets a copy
        std::vector<LocalSystemSimplicialComplex::SimplexRelation> relations;
        relations.emplace_back(0, find_simplex_idx(K_rect_instance, 0, { 4 }), find_simplex_idx(K_rect_instance, 0, { 0 }), M_loop, cyl_variant_name + "_v4_v0");
        relations.emplace_back(0, find_simplex_idx(K_rect_instance, 0, { 5 }), find_simplex_idx(K_rect_instance, 0, { 1 }), M_loop, cyl_variant_name + "_v5_v1");
        relations.emplace_back(1, find_simplex_idx(K_rect_instance, 1, { 4,5 }), find_simplex_idx(K_rect_instance, 1, { 0,1 }), M_loop, cyl_variant_name + "_e45_e01");
        return LocalSystemSimplicialComplex(K_rect_instance, relations, d_V, p);
    }

    LocalSystemSimplicialComplex CreateCylinderScalar(uint32_t scalar, uint32_t p, const std::string& name_suffix) {
        uint32_t d_V = 1;
        // Same canonicalisation as CreateS1Scalar (see the note there).
        const uint32_t s = engine::ModCtx<uint32_t>(p).canonicalise(scalar);
        FlatMatrix M_scalar(d_V, d_V, p, std::vector<uint32_t>{ s });
        std::ostringstream name;
        name << "Cylinder_Scalar(" << s << ")" << name_suffix;
        return CreateCylinderGeneric(M_scalar, p, d_V, name.str());
    }

    LocalSystemSimplicialComplex CreateCylinderUnipotent2D(uint32_t p, const std::string& name_suffix) {
        uint32_t d_V = 2;
        FlatMatrix M_unip(d_V, d_V, p, std::vector<uint32_t>{ 1, 1, 0, 1 });
        std::ostringstream name;
        name << "Cylinder_Unipotent2D" << name_suffix;
        return CreateCylinderGeneric(M_unip, p, d_V, name.str());
    }

    LocalSystemSimplicialComplex CreateMoebiusGeneric(const FlatMatrix& M_twist, uint32_t p, uint32_t d_V, const std::string& moebius_variant_name) {
        if (M_twist.get_rows() != d_V || M_twist.get_cols() != d_V || M_twist.get_p() != p) {
            throw std::runtime_error("CreateMoebiusGeneric: M_twist dimensions or prime mismatch.");
        }
        SimplicialComplex K_rect_instance = GetKRectangle(); // Gets a copy
        std::vector<LocalSystemSimplicialComplex::SimplexRelation> relations;
        relations.emplace_back(0, find_simplex_idx(K_rect_instance, 0, { 5 }), find_simplex_idx(K_rect_instance, 0, { 0 }), M_twist, moebius_variant_name + "_v5_v0");
        relations.emplace_back(0, find_simplex_idx(K_rect_instance, 0, { 4 }), find_simplex_idx(K_rect_instance, 0, { 1 }), M_twist, moebius_variant_name + "_v4_v1");
        FlatMatrix M_edge_twist = M_twist * (p - 1); // -M_twist
        relations.emplace_back(1, find_simplex_idx(K_rect_instance, 1, { 4,5 }), find_simplex_idx(K_rect_instance, 1, { 0,1 }), M_edge_twist, moebius_variant_name + "_e45_e01_twist");
        return LocalSystemSimplicialComplex(K_rect_instance, relations, d_V, p);
    }

    LocalSystemSimplicialComplex CreateMoebiusScalar(uint32_t scalar, uint32_t p, const std::string& name_suffix) {
        uint32_t d_V = 1;
        // Same canonicalisation as CreateS1Scalar (see the note there).
        const uint32_t s = engine::ModCtx<uint32_t>(p).canonicalise(scalar);
        FlatMatrix M_scalar_twist(d_V, d_V, p, std::vector<uint32_t>{ s });
        std::ostringstream name;
        name << "Moebius_Scalar(" << s << ")" << name_suffix;
        return CreateMoebiusGeneric(M_scalar_twist, p, d_V, name.str());
    }

    LocalSystemSimplicialComplex CreateMoebiusUnipotent2D(uint32_t p, const std::string& name_suffix) {
        uint32_t d_V = 2;
        FlatMatrix M_unip_twist(d_V, d_V, p, std::vector<uint32_t>{ 1, 1, 0, 1 });
        std::ostringstream name;
        name << "Moebius_Unipotent2D" << name_suffix;
        return CreateMoebiusGeneric(M_unip_twist, p, d_V, name.str());
    }


    // --- Equivariant Cochain Complex Morphism Creators ---

    EquivariantCochainComplexMorphismOnK CreateS1_IdentityMorphism(
        const EquivariantCohomologyOnK* s1_cohomology) {

        if (!s1_cohomology) throw std::invalid_argument("s1_cohomology cannot be null.");
        // The base of s1_cohomology is K_Interval; for an identity f_K, K_domain = K_codomain,
        // so the SimplicialComplexMorphism is built once on a static K_Interval.
        static SimplicialComplex K_int_for_morph = GetKInterval(); // Ensure one instance for SCM
        static const SimplicialComplexMorphism scm_s1_id(&K_int_for_morph, &K_int_for_morph); // Domain K1, Codomain K2

        uint32_t p = s1_cohomology->get_prime();
        uint32_t d_V = s1_cohomology->get_representation_dim();
        FlatMatrix F_id(d_V, p); // Identity matrix of size d_V

        // Compatibility check: F M = M F. If F=Id, this is M=M, always true.
        return EquivariantCochainComplexMorphismOnK(
            s1_cohomology, // Domain for H* map
            s1_cohomology, // Codomain for H* map
            &scm_s1_id,
            F_id);
    }

    EquivariantCochainComplexMorphismOnK CreateS1_UnipotentToTrivialMorphism(
        const EquivariantCohomologyOnK* s1_trivial_cohomology,
        const EquivariantCohomologyOnK* s1_unip_cohomology) {

        if (!s1_trivial_cohomology || !s1_unip_cohomology) {
            throw std::invalid_argument("Cohomology wrapper pointers cannot be null for S1 Unip->Triv.");
        }
        if (s1_trivial_cohomology->get_representation_dim() != 1 ||
            s1_unip_cohomology->get_representation_dim() != 2) {
            throw std::invalid_argument("Dimension mismatch for S1 Unip->Triv (expected dV1=1, dV2=2).");
        }
        if (s1_trivial_cohomology->get_prime() != s1_unip_cohomology->get_prime()) {
            throw std::invalid_argument("Prime mismatch for S1 Unip->Triv.");
        }
        uint32_t p = s1_trivial_cohomology->get_prime();

        // K should be the same K_Interval for both.
        static SimplicialComplex K_int_for_morph = GetKInterval();
        static const SimplicialComplexMorphism scm_s1_id(&K_int_for_morph, &K_int_for_morph);

        FlatMatrix F_proj(1, 2, p, std::vector<uint32_t>{ 0, 1 }); // V_unip (2D) -> V_trivial (1D)

        // Check compatibility: F_proj * M_unip = M_trivial * F_proj
        // M_unip = {{1,1},{0,1}}, M_trivial = {{1}}
        // {{0,1}} * {{1,1},{0,1}} = {{0,1}}
        // {{1}} * {{0,1}} = {{0,1}}
        // This holds.

        return EquivariantCochainComplexMorphismOnK(
            s1_trivial_cohomology, // Domain for H* map (target of F, LSSC1)
            s1_unip_cohomology,    // Codomain for H* map (source of F, LSSC2)
            &scm_s1_id,
            F_proj);
    }

    EquivariantCochainComplexMorphismOnK CreateS1_TrivialToUnipotentMorphism(
        const EquivariantCohomologyOnK* s1_unip_cohomology,
        const EquivariantCohomologyOnK* s1_trivial_cohomology) {

        if (!s1_trivial_cohomology || !s1_unip_cohomology) {
            throw std::invalid_argument("Cohomology wrapper pointers cannot be null for S1 Triv->Unip.");
        }
        if (s1_trivial_cohomology->get_representation_dim() != 1 ||
            s1_unip_cohomology->get_representation_dim() != 2) {
            throw std::invalid_argument("Dimension mismatch for S1 Triv->Unip (expected dV1=2, dV2=1).");
        }
        if (s1_trivial_cohomology->get_prime() != s1_unip_cohomology->get_prime()) {
            throw std::invalid_argument("Prime mismatch for S1 Triv->Unip.");
        }
        uint32_t p = s1_unip_cohomology->get_prime();

        static SimplicialComplex K_int_for_morph = GetKInterval();
        static const SimplicialComplexMorphism scm_s1_id(&K_int_for_morph, &K_int_for_morph);

        FlatMatrix F_incl(2, 1, p, std::vector<uint32_t>{ 1, 0 }); // V_trivial (1D) -> V_unip (2D)

        // Check compatibility: F_incl * M_trivial = M_unip * F_incl
        // M_trivial = {{1}}, M_unip = {{1,1},{0,1}}
        // {{1},{0}} * {{1}} = {{1},{0}}
        // {{1,1},{0,1}} * {{1},{0}} = {{1},{0}}
        // This holds.

        return EquivariantCochainComplexMorphismOnK(
            s1_unip_cohomology,      // Domain for H* map (target of F, LSSC1)
            s1_trivial_cohomology,   // Codomain for H* map (source of F, LSSC2)
            &scm_s1_id,
            F_incl);
    }


    // Placeholder for boundary inclusion to Moebius - more complex
    /*
    EquivariantCochainComplexMorphismOnK CreateMoebius_BoundaryS1_Inclusion_Sign(
        const EquivariantCohomologyOnK* moebius_sign_cohomology, uint32_t p_val) {

        if (!moebius_sign_cohomology) throw std::invalid_argument("Moebius cohomology cannot be null.");
        if (moebius_sign_cohomology->get_prime() != p_val) {
            throw std::invalid_argument("Prime mismatch for Moebius boundary inclusion.");
        }
        if (moebius_sign_cohomology->get_representation_dim() != 1) {
             throw std::invalid_argument("Moebius sign system should be 1D for this preset.");
        }

        // LSSC2 (codomain for H* map) is given by moebius_sign_cohomology.
        // K2 = K_Rectangle

        // LSSC1 (domain for H* map) is for S^1_boundary.
        // Monodromy M_S1_bdy = (M_twist_Moebius)^2.
        // If M_twist_Moebius = [-1], then M_S1_bdy = [1].
        FlatMatrix M_S1_bdy_loop(1, 1, p_val, std::vector<uint32_t>{1}); // (-1)^2 = 1
        LocalSystemSimplicialComplex lssc_s1_bdy = CreateS1Generic(M_S1_bdy_loop, p_val, 1, "S1_MoebiusBoundary");

        // Need EquivariantCohomologyOnK for lssc_s1_bdy. This implies presets for morphisms might need to manage this.
        // This is getting complicated for a simple preset. For now, user should construct the EqCohK objects.
        // For validation, one would:
        // 1. Create LSSC_Moeb_Sign
        // 2. Create LSSC_S1_bdy_from_Moeb_Sign (monodromy M_twist^2)
        // 3. Compute cohomology for both.
        // 4. Define f_K: K_Interval -> K_Rectangle (this is the hard part for a simplicial map)
        // 5. Define F (likely identity [1])
        // 6. Construct EquivariantCochainComplexMorphismOnK

        throw std::runtime_error("CreateMoebius_BoundaryS1_Inclusion_Sign preset not fully implemented due to K mapping complexity.");
    }
    */

} // namespace LocalSystemPresets
