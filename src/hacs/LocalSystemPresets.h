// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#ifndef LOCAL_SYSTEM_PRESETS_H
#define LOCAL_SYSTEM_PRESETS_H

#include "LocalSystemSimplicialComplex.h"
#include "EquivariantCohomologyOnK.h"          // For EquivariantCohomologyOnK
#include "EquivariantCochainComplexMorphismOnK.h" // For EquivariantCochainComplexMorphismOnK
#include "SimplicialCohomology.h"              // For SimplicialComplexMorphism
#include <cstdint>
#include <stdexcept>
#include <string> // For std::string

namespace LocalSystemPresets {

    // --- LSSC Creators (from previous response, kept for completeness) ---
    LocalSystemSimplicialComplex CreateS1Generic(const FlatMatrix& M_loop, uint32_t p, uint32_t d_V, const std::string& S1_variant_name);
    LocalSystemSimplicialComplex CreateS1Scalar(uint32_t scalar, uint32_t p, const std::string& name_suffix = "");
    LocalSystemSimplicialComplex CreateS1Unipotent2D(uint32_t p, const std::string& name_suffix = "");
    LocalSystemSimplicialComplex CreateCylinderGeneric(const FlatMatrix& M_loop, uint32_t p, uint32_t d_V, const std::string& cyl_variant_name);
    LocalSystemSimplicialComplex CreateCylinderScalar(uint32_t scalar, uint32_t p, const std::string& name_suffix = "");
    LocalSystemSimplicialComplex CreateCylinderUnipotent2D(uint32_t p, const std::string& name_suffix = "");
    LocalSystemSimplicialComplex CreateMoebiusGeneric(const FlatMatrix& M_twist, uint32_t p, uint32_t d_V, const std::string& moebius_variant_name);
    LocalSystemSimplicialComplex CreateMoebiusScalar(uint32_t scalar, uint32_t p, const std::string& name_suffix = "");
    LocalSystemSimplicialComplex CreateMoebiusUnipotent2D(uint32_t p, const std::string& name_suffix = "");

    // --- Equivariant Cochain Complex Morphism Creators ---

    /**
     * @brief Creates an identity morphism for an LSSC on S^1.
     *        f_K is identity on K_Interval, F is identity on V.
     * @param s1_cohomology Pointer to the computed EquivariantCohomologyOnK for an S^1 LSSC.
     *                      The underlying K must be compatible with K_Interval.
     * @return An EquivariantCochainComplexMorphismOnK object.
     */
    EquivariantCochainComplexMorphismOnK CreateS1_IdentityMorphism(
        const EquivariantCohomologyOnK* s1_cohomology);

    /**
     * @brief Morphism for S^1: L_Unipotent2D -> L_Trivial(1D).
     *        f_K is identity on K_Interval. F is projection [0,1].
     * @param s1_trivial_cohomology Pointer to H*(S^1, L_Trivial(1D)). K must be K_Interval.
     * @param s1_unip_cohomology Pointer to H*(S^1, L_Unipotent2D). K must be K_Interval.
     * @return An EquivariantCochainComplexMorphismOnK object.
     */
    EquivariantCochainComplexMorphismOnK CreateS1_UnipotentToTrivialMorphism(
        const EquivariantCohomologyOnK* s1_trivial_cohomology, // Domain for H* map (target of F)
        const EquivariantCohomologyOnK* s1_unip_cohomology);    // Codomain for H* map (source of F)

    /**
     * @brief Morphism for S^1: L_Trivial(1D) -> L_Unipotent2D.
     *        f_K is identity on K_Interval. F is inclusion [[1],[0]].
     * @param s1_unip_cohomology Pointer to H*(S^1, L_Unipotent2D). K must be K_Interval.
     * @param s1_trivial_cohomology Pointer to H*(S^1, L_Trivial(1D)). K must be K_Interval.
     * @return An EquivariantCochainComplexMorphismOnK object.
     */
    EquivariantCochainComplexMorphismOnK CreateS1_TrivialToUnipotentMorphism(
        const EquivariantCohomologyOnK* s1_unip_cohomology,    // Domain for H* map (target of F)
        const EquivariantCohomologyOnK* s1_trivial_cohomology); // Codomain for H* map (source of F)


    /**
     * @brief Morphism for S^1_boundary into Moebius strip with sign system.
     *        L_S1 has monodromy M_twist^2. F is identity V->V.
     *        K_S1 is K_Interval. K_Moebius is K_Rectangle.
     *        This preset constructs the LSSC for S1_boundary internally.
     *        The SimplicialComplexMorphism is conceptual (not a direct subcomplex K).
     *        For validation, this might be complex due to f_K not being simple inclusion.
     *        THIS IS A PLACEHOLDER - Requires careful f_K definition.
     */
     // EquivariantCochainComplexMorphismOnK CreateMoebius_BoundaryS1_Inclusion_Sign(
     //    const EquivariantCohomologyOnK* moebius_sign_cohomology,
     //    uint32_t p_val); // p_val to create the S1 LSSC

} // namespace LocalSystemPresets

#endif // LOCAL_SYSTEM_PRESETS_H
