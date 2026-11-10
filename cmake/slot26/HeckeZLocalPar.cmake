# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

# Slot 26 Z9 HZ-LOC-PAR (Registry/Slot26/README.md): HeckeZLocalParValidation (standard) and its registration fragment.
target_sources(cohomology_app PRIVATE
    src/hacs/Integral/HeckeZLocalParValidation.cpp
    src/hacs/Registry/Slot26/HeckeZLocalPar.cpp
)
