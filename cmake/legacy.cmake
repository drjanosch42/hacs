# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

# cmake/legacy.cmake -- the retired routes' sources (S24 CL-L0; docs/implementation/slot24/CLEANUP_plan_1030.md
# Sec. 2.2).  Included UNCONDITIONALLY from CMakeLists.txt, after every target exists.  Every `legacy/` .cpp is added
# to its target here, inside if(OC_ENABLE_LEGACY), and nowhere else; a `.inc` is not listed (its .cpp includes it
# under `#if OC_ENABLE_LEGACY`).  Each `legacy/` directory carries a README.md (what is there, when retired, by
# which package, how to reactivate).
#
# The packages that fill it: CL-G2 (Hecke/legacy/HeckeLevelNGenOrderLegacy.inc), CL-R1 (the slot-13/14 levers),
# CL-L1 (the shadow), CL-L2 (the backend and BlockFlatMatrix).  Empty at CL-L0; CL-G2 moved the first route, CL-R1 the slot-13/14 levers, CL-L2 the legacy backend.

# (S24 CL-L1) The ENGINE_SHADOW_LEGACY comparisons are .inc files, not sources: src/ModArithmetic/legacy/
# ShadowLegacy.inc (FlatMatrix.cpp, BlockFlatMatrix.cpp), src/hacs/legacy/EquivariantCohomologyOnKShadow.inc
# and src/hacs/Hecke/legacy/HeckeOperatorOnKShadow.inc, included under #ifdef ENGINE_SHADOW_LEGACY;
# CMakeLists.txt admits ENGINE_SHADOW_LEGACY=ON only with OC_ENABLE_LEGACY=ON.
if(OC_ENABLE_LEGACY)
    # S24 CL-L2: the legacy backend.  BlockFlatMatrixT (the deprecated block container, legacy arm only) and its
    # validation; the library halves of the backend are .inc files (legacy/DenseElimLegacy.inc, FlatMatrixLegacy.inc,
    # MicroBenchLegacy.inc; src/hacs/legacy/EquivariantCohomologyOnKLegacy.inc,
    # SimplicialCohomologyLegacy.inc; Hecke/legacy/HeckeOperatorOnKLegacy.inc, HeckeInvolutionLegacy.inc).
    target_sources(ModArithmetic PRIVATE
        src/ModArithmetic/legacy/BlockFlatMatrix.cpp
        src/ModArithmetic/legacy/BlockFlatMatrixValidation.cpp
    )
    # src/ModArithmetic/legacy/      -> target_sources(ModArithmetic PRIVATE ...)
    # src/hacs/legacy/       -> target_sources(cohomology_app PRIVATE ...)
    # src/hacs/Hecke/legacy/ -> target_sources(cohomology_app PRIVATE ...)
    target_sources(cohomology_app PRIVATE
        # S24 CL-G2: the retired generator order HECKE_LEVEL_N_GEN_ORDER=as_built (its library half is
        # Hecke/legacy/HeckeLevelNGenOrderLegacy.inc, included by HeckeLevelNSparse.cpp) -- the (W18o-as_built) arm.
        src/hacs/Hecke/legacy/HeckeLevelNOrderLegacyArms.cpp
        # S24 CL-R1: the retired slot-13/14 levers HECKE_U_ELL_LOW_RANK, HECKE_COSET_STREAM, HECKE_PARALLEL_SETUP,
        # HECKE_RANK1_CLOSED_FORM (library halves: Hecke/legacy/HeckeOperatorLeversLegacy.h, included by
        # HeckeOperatorOnK.h, and Hecke/legacy/HeckeCosetTablesMaterialised.inc, included by HeckeCosetTables.cpp) --
        # the whole HeckeUEllStreamValidation (moved from cmake/slot14/WP10.cmake) and HeckeUEllLowRankValidation's
        # lever arms (A)-(G).
        src/hacs/Hecke/legacy/HeckeUEllStreamValidation.cpp
        src/hacs/Hecke/legacy/HeckeUEllLowRankLegacyArms.cpp
        # S24 CL-L2: the whole names CohomologyBlockModelValidation, HeckeThetaBlockValidation and
        # BarCocycleSolverValidation (moved from CMakeLists.txt's cohomology_app list).
        src/hacs/legacy/CohomologyBlockModelValidation.cpp
        src/hacs/Hecke/legacy/HeckeThetaBlockValidation.cpp
        src/hacs/Hecke/legacy/BarCocycleSolverValidation.cpp
    )
endif()
