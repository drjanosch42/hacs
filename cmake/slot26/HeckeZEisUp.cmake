# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

# Slot-26 HZ-EISUP: the Eisenstein U_p systems of the full algebra at N > 1 (docs/implementation/slot26/HZEISUP_report.md).
target_sources(cohomology_app PRIVATE
    src/hacs/Integral/HeckeZEisUp.cpp
    src/hacs/Integral/HeckeZEisUpValidation.cpp
    src/hacs/Registry/Slot26/HeckeZEisUp.cpp
)
