// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#include <cstdint>

// (hecke-extensions slot 9, WP-6) The tape elimination driver compiles under
// the consumers' pass options (the process-wide ScopedPassOptions override
// included), and pass_options_fingerprint() keys the EFFECTIVE device policy.
// See EngineElimPassOptionsValidation.cpp.
uint32_t EngineElimPassOptionsValidation();
