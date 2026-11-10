# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

# cohomology_app sources under src/hacs/Orbifold/ (L23 census) added by slots 9..23 (S24 CL-10; slot order kept).
target_sources(cohomology_app PRIVATE
    src/hacs/Orbifold/MP/Descriptor.cpp
    src/hacs/Orbifold/MP/MPSet.cpp
    src/hacs/Orbifold/MP/OrbifoldMPRulesValidation.cpp
    src/hacs/Orbifold/MP/Nerve.cpp
    src/hacs/Orbifold/MP/SparseLength.cpp
    src/hacs/Orbifold/MP/AmalgamOracle.cpp
    src/hacs/Orbifold/MP/OrbifoldMPNerveValidation.cpp
    src/hacs/Orbifold/MP/OrbifoldMPTeardropValidation.cpp
    src/hacs/Orbifold/MP/CochainMaps.cpp
    src/hacs/Orbifold/MP/Comparison.cpp
    src/hacs/Orbifold/MP/Refinement.cpp
    src/hacs/Orbifold/MP/Models.cpp
    src/hacs/Orbifold/MP/MP2Common.cpp
    src/hacs/Orbifold/MP/OrbifoldMPComparisonValidation.cpp
    src/hacs/Orbifold/MP/OrbifoldMPRefinementValidation.cpp
    src/hacs/Orbifold/MP/OrbifoldMPAllDegreesValidation.cpp
    src/hacs/Orbifold/MP/OrbifoldMPNerveH3Slow.cpp
    src/hacs/Orbifold/MP/ModuleCochains.cpp
    src/hacs/Orbifold/MP/Boundary.cpp
    src/hacs/Orbifold/MP/SubgroupMP.cpp
    src/hacs/Orbifold/MP/MP3Oracle.cpp
    src/hacs/Orbifold/MP/OrbifoldMPBoundaryValidation.cpp
    src/hacs/Orbifold/MP/OrbifoldMPLevelConeValidation.cpp
    src/hacs/Orbifold/MP/MP3bOracle.cpp
    src/hacs/Orbifold/MP/MP3bLevelArms.cpp
    src/hacs/Orbifold/MP/MPFill.cpp
    src/hacs/Orbifold/MP/MPHecke.cpp
    src/hacs/Orbifold/MP/OrbifoldMPHeckeValidation.cpp
    src/hacs/Orbifold/L23/AmalgamReader.cpp
    src/hacs/Orbifold/L23/L23AmalgamReaderValidation.cpp
    src/hacs/Orbifold/L23/SumRuleOracle.cpp
    src/hacs/Orbifold/L23/L23Bin3o1Validation.cpp
)
