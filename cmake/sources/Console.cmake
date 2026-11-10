# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

# cohomology_app sources under src/hacs/Console/ added by slots 9..25 (S24 CL-10, CL-10b, S25 FOLD; slot order kept).
target_sources(cohomology_app PRIVATE
    src/hacs/Console/Bindings/Duality.cpp
    src/hacs/Console/CliManualCoverageValidation.cpp
    src/hacs/Console/Bindings/ReadOnlyTags.cpp
    src/hacs/Console/ConsoleReadOnlyTagValidation.cpp
    src/hacs/Console/ConsoleServeValidation.cpp
    src/hacs/Console/ConsoleServeAtlasValidation.cpp
)

# Slot 24 (folded at the slot-24 close, S24 CL-10b; the former cmake/slot24/<pkg>.cmake, slot order kept).
target_sources(cohomology_app PRIVATE
    # --- slot 24 DESCENT6
    src/hacs/Console/Bindings/Descent.cpp
    # --- slot 24 NBR34
    src/hacs/Console/Bindings/Display.cpp
)

# Slot 25 (folded at the slot-25 close, S25 FOLD; the former cmake/slot25/<pkg>.cmake, slot order kept).
target_sources(cohomology_app PRIVATE
    # --- slot 25 NbConfirm (its ConsoleCore sources and the embedded cost models: Special.cmake)
    src/hacs/Console/Bindings/HostProbe.cpp
    src/hacs/Console/Bindings/Calibrate.cpp
    # ConsoleCostModelValidation (fast)
    src/hacs/Console/ConsoleCostModelValidation.cpp
    # ConsoleCancelValidation (standard; NBCONFIRM S5)
    src/hacs/Console/ConsoleCancelValidation.cpp
)
