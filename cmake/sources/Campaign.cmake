# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

# cohomology_app sources under src/hacs/Campaign/ added by slots 9..25 (S24 CL-10, CL-10b, S25 FOLD; slot order kept).
target_sources(cohomology_app PRIVATE
    src/hacs/Campaign/CampaignRing.cpp
    src/hacs/Campaign/CampaignRingValidation.cpp
    src/hacs/Campaign/CampaignOrbifold.cpp
    src/hacs/Campaign/CampaignOrbifoldValidation.cpp
    src/hacs/Campaign/OrbifoldCensusL23Validation.cpp
    src/hacs/Campaign/OrbifoldRing2kValidation.cpp
    src/hacs/Campaign/OrbifoldMod3kValidation.cpp
    src/hacs/Campaign/OrbifoldF3lValidation.cpp
)

# Slot 24 (folded at the slot-24 close, S24 CL-10b; the former cmake/slot24/<pkg>.cmake, slot order kept).
target_sources(cohomology_app PRIVATE
    # --- slot 24 DESCENT6
    src/hacs/Campaign/CampaignDescent.cpp
)

# Slot 25 (folded at the slot-25 close, S25 FOLD; the former cmake/slot25/<pkg>.cmake, slot order kept).
target_sources(cohomology_app PRIVATE
    # --- slot 25 HeckeZRecord
    src/hacs/Campaign/CampaignHeckeZ.cpp
)
