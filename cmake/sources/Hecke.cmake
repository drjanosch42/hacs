# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

# cohomology_app sources under src/hacs/Hecke/ added by slots 9..24 (S24 CL-10, CL-10b; slot order kept).
target_sources(cohomology_app PRIVATE
    src/hacs/Hecke/HeckeReductionConeValidation.cpp
    src/hacs/Hecke/HeckeFamilyConcurrentBuildValidation.cpp
    src/hacs/Hecke/HeckeConeCrossRouteRingValidation.cpp
    src/hacs/Hecke/LadicCell.cpp
    src/hacs/Hecke/HeckeLadicCellValidation.cpp
    src/hacs/Hecke/HeckeOrdinaryPart.cpp
    src/hacs/Hecke/HeckeOrdinaryPrecisionValidation.cpp
    src/hacs/Hecke/HeckeUEllLowRankValidation.cpp
    src/hacs/Hecke/HeckeLevelEll.cpp
    src/hacs/Hecke/HeckeHidaLevelEllValidation.cpp
    src/hacs/Hecke/HeckeLevelN.cpp
    src/hacs/Hecke/HeckeLevelNValidation.cpp
    src/hacs/Hecke/HeckeLevelNHeilbronn.cpp
    src/hacs/Hecke/HeckeLevelNHeilbronnValidation.cpp
    src/hacs/Hecke/HeckeLevelNSparse.cpp
    src/hacs/Hecke/HeckeLevelNSparseValidation.cpp
    src/hacs/Hecke/HeckeLevelNR3.cpp
    src/hacs/Hecke/HeckeNebentypus.cpp
    src/hacs/Hecke/HeckeNebentypusValidation.cpp
    src/hacs/Hecke/HeckeLevelNDegen.cpp
    src/hacs/Hecke/HeckeLevelNDegenValidation.cpp
    src/hacs/Hecke/HeckeTauDumpValidation.cpp
    src/hacs/Hecke/HeckeLevelNWideValidation.cpp
    src/hacs/Hecke/HeckeNebentypusSparse.cpp
    src/hacs/Hecke/HeckeLevelNOrderValidation.cpp
    src/hacs/Hecke/HeckeNebentypusGalois.cpp
    src/hacs/Hecke/HeckeLadicSplitValidation.cpp
    src/hacs/Hecke/HeckeLevelNCurveValidation.cpp
    src/hacs/Hecke/HeckeGammaH.cpp
    src/hacs/Hecke/HeckeTameFamily.cpp
    src/hacs/Hecke/HeckeTameFamilyValidation.cpp
    src/hacs/Hecke/HeckeGammaHTower.cpp
    src/hacs/Hecke/HeckeNebentypusWild.cpp
    src/hacs/Hecke/HeckeLevelPair.cpp
    src/hacs/Hecke/HeckeLevelNormalityValidation.cpp
    src/hacs/Hecke/HeckeLevelCone.cpp
    src/hacs/Hecke/HeckeLevelConeValidation.cpp
    src/hacs/Hecke/HeckeThetaRssDietValidation.cpp
    src/hacs/Hecke/HeckeLevelInfRes.cpp
    src/hacs/Hecke/HeckeLevelInfResValidation.cpp
    src/hacs/Hecke/HeckeRho2StreamValidation.cpp
    src/hacs/Hecke/HeckeIotaStreamValidation.cpp
)

# Slot 24 (folded at the slot-24 close, S24 CL-10b; the former cmake/slot24/<pkg>.cmake, slot order kept).
target_sources(cohomology_app PRIVATE
    # --- slot 24 ELLW
    # HeckeWideTauValidation (standard): the wide tier against the narrow one and against the q-expansion of Delta.
    src/hacs/Hecke/HeckeWideTauValidation.cpp
)
