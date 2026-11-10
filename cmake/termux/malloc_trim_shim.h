// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Force-included (cmake/termux/MallocTrim.cmake) where the C library has no malloc_trim (bionic / Termux).
// malloc_trim only returns freed heap to the OS; a no-op changes no value.
#pragma once
#include <cstddef>
#include <malloc.h>

static inline int malloc_trim(std::size_t) { return 0; }
