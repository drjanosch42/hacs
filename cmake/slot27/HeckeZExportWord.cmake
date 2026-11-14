# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

# Slot-27 EXW: the at-root read and the (C3) read on the free block over Z (docs/implementation/slot27/EXW_design.md v0.2).
target_sources(cohomology_app PRIVATE
    src/hacs/Integral/HeckeZExportWordValidation.cpp
    src/hacs/Registry/Slot27/HeckeZExportWord.cpp
)
