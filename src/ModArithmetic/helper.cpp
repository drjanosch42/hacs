// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

/*
 * helper.cpp
 *
 * This file implements general-purpose helper and utility functions that are
 * used across the project but do not belong to a specific mathematical or
 * data structure class.
 *
 * Currently Implemented:
 * - `offset_string`: A simple formatting utility that generates a string of
 *   spaces for indenting console output, improving readability of debug
 *   and results printouts.
 */

#include "pch.h"
#include "helper.h"
#include <sstream>

std::string offset_string(const size_t offset)
{
    std::ostringstream oss;
    for (size_t i = 0; i < offset; ++i) {
        oss << " ";
    }
    return oss.str();
}
