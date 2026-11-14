# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

# Slot-27 BIGNTT M1: the NTT product and Barrett modular product of nf::BigInt (docs/implementation/slot27/BIGNTT_design.md v0.2).
# BigNtt.cpp joins the STL-only NumberField target; the validation is compiled into hacs.
target_sources(NumberField PRIVATE
    src/hacs/NumberField/BigNtt.cpp
)
target_sources(cohomology_app PRIVATE
    src/hacs/Integral/BigIntNttValidation.cpp
    src/hacs/Registry/Slot27/BigIntNtt.cpp
)
