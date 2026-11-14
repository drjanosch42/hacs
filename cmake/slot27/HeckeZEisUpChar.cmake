# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

# Slot-27 EISUPb: HZ-EISUP (b), the Eisenstein U_p systems with characters (docs/implementation/slot26/HZEISUPb_design.md v0.4).
target_sources(cohomology_app PRIVATE
    src/hacs/Integral/HeckeZEisUpChar.cpp
    src/hacs/Integral/HeckeZEisUpCharValidation.cpp
    src/hacs/Registry/Slot27/HeckeZEisUpChar.cpp
)
