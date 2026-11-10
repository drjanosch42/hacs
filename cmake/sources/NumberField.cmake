# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

# cohomology_app sources under src/hacs/NumberField/ added by slots 9..23 (S24 CL-10; slot order kept).
target_sources(cohomology_app PRIVATE
    src/hacs/NumberField/NumberFieldValidation.cpp
    src/hacs/NumberField/CwLayerValidation.cpp
    src/hacs/NumberField/LllValidation.cpp
    src/hacs/NumberField/RibetFieldValidation.cpp
    src/hacs/NumberField/PRankValidation.cpp
    src/hacs/NumberField/CyclicCubicValidation.cpp
)
