# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

# Slot-27 TS-FIX / TS: the theta search (docs/implementation/slot27/TS_design.md).
target_sources(cohomology_app PRIVATE
    src/hacs/Integral/HeckeZThetaSearchValidation.cpp
    src/hacs/Registry/Slot27/HeckeZThetaSearch.cpp
)
