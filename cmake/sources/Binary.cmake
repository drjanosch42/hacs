# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

# cohomology_app sources under src/hacs/Binary/ added by slots 9..23 (S24 CL-10; slot order kept).
target_sources(cohomology_app PRIVATE
    src/hacs/Binary/Ring2kReference.cpp
    src/hacs/Binary/Ring2kLadderValidation.cpp
    src/hacs/Binary/TernaryF3Validation.cpp
    src/hacs/Binary/Ring2kR5SimdValidation.cpp
)
