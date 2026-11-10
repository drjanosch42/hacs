# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

# Slot-26 Z1 (UP-Z (a)): U_p, p | N, over Z_(ell), ell >= 5 (docs/implementation/slot26/Z1_report.md).
target_sources(cohomology_app PRIVATE
    src/hacs/Integral/HeckeZUp.cpp
    src/hacs/Integral/HeckeZUpValidation.cpp
    src/hacs/Registry/Slot26/HeckeZUp.cpp
)
