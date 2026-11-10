// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#ifndef FLAT_MATRIX_VALIDATION_H
#define FLAT_MATRIX_VALIDATION_H

#include <cstdint> // For uint32_t

/**
 * @brief Runs a suite of validation tests for the FlatMatrix class.
 * @param verbosity Level of detail for output.
 *                  0: Minimal output (only summary and errors).
 *                  1: Print PASS/FAIL for each test category.
 *                  2: Print PASS/FAIL for each individual test.
 *                  3: Print matrix details for relevant tests.
 * @return 0 if all tests pass, 1 otherwise.
 */
uint32_t FlatMatrixValidation(int verbosity);

#endif // FLAT_MATRIX_VALIDATION_H#pragma once
