# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

# Slot-26 Z6 (= H2): the discriminant series of the ramified ordinary families at tame level 1 ([U] part 7, sec:u-H2;
# docs/implementation/slot26/Z6_report.md).
target_sources(cohomology_app PRIVATE
    src/hacs/HeckeExt/HeckeHidaDiscLambda.cpp
    src/hacs/HeckeExt/HeckeHidaDiscLambdaValidation.cpp
    src/hacs/Registry/Slot26/HeckeHidaDiscLambda.cpp
)
