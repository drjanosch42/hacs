# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

# cohomology_app sources under src/hacs/Util/ added by slots 9..24 (S24 CL-10, CL-10b; slot order kept).
target_sources(cohomology_app PRIVATE
    src/hacs/Util/DeviceRelease.cpp
    src/hacs/Util/AtlasIndex.cpp
)

# Slot 24 (folded at the slot-24 close, S24 CL-10b; the former cmake/slot24/<pkg>.cmake, slot order kept).
target_sources(cohomology_app PRIVATE
    # --- slot 24 CL1
    src/hacs/Util/SharedHelpersValidation.cpp
)
