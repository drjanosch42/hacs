# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

# cohomology_app sources under src/hacs/Registry/ added by slots 9..25 (S24 CL-10, CL-10b, S25 FOLD; slot order kept).
target_sources(cohomology_app PRIVATE
    src/hacs/Registry/Slot9.cpp
    src/hacs/Registry/Slot10.cpp
    src/hacs/Registry/Slot11.cpp
    src/hacs/Registry/Slot12.cpp
    src/hacs/Registry/Slot13.cpp
    src/hacs/Registry/Slot14.cpp
    src/hacs/Registry/Slot15.cpp
    src/hacs/Registry/Slot16.cpp
    src/hacs/Registry/Slot17.cpp
    src/hacs/Registry/Slot18.cpp
    src/hacs/Registry/Slot19.cpp
    src/hacs/Registry/Slot20.cpp
    src/hacs/Registry/Slot21.cpp
    src/hacs/Registry/Slot22.cpp
    src/hacs/Registry/Slot23.cpp
)

# Slot 24 (folded at the slot-24 close, S24 CL-10b): the 20 package fragments Registry/Slot24/<pkg>.cpp are the one file Registry/Slot24.cpp.
target_sources(cohomology_app PRIVATE
    src/hacs/Registry/Slot24.cpp
)

# Slot 25 (folded at the slot-25 close, S25 FOLD): the 13 package fragments Registry/Slot25/<pkg>.cpp are the one file Registry/Slot25.cpp.
target_sources(cohomology_app PRIVATE
    src/hacs/Registry/Slot25.cpp
)
