// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Stage S3 -- see HeckeCosetTablesSelfCheckValidation.h.

#include "HeckeCosetTablesSelfCheckValidation.h"

#include "HeckeCosetTables.h"

#include <iostream>
#include <vector>

uint32_t HeckeCosetTablesSelfCheckValidation() {
    const std::vector<uint64_t> p_heckes = { 2, 3, 5, 7, 11 };
    const std::vector<uint32_t> rs = { 1, 2 };
    const std::vector<uint32_t> ells = { 5, 691 };
    const std::vector<uint32_t> ns = { 0, 2, 10 };

    uint32_t failures = 0;
    uint32_t combinations = 0;

    for (uint64_t p_hecke : p_heckes) {
        for (uint32_t r : rs) {
            for (uint32_t ell : ells) {
                for (uint32_t n_sym : ns) {
                    ++combinations;
                    try {
                        const hecke::HeckeCosetTables tables(p_hecke, r, ell, n_sym);
                        const uint32_t code = tables.self_check();
                        if (code != 0) {
                            std::cout << "HeckeCosetTablesSelfCheckValidation: FAILED "
                                         "self_check() for p_hecke=" << p_hecke
                                      << " r=" << r << " ell=" << ell << " n_sym=" << n_sym
                                      << " (code " << code << ")." << std::endl;
                            ++failures;
                        }
                        // Also exercise num_cosets()/N() directly against the
                        // closed-form oracle p^r + p^{r-1} (DoD item (i)),
                        // independently of self_check()'s own internal check.
                        uint64_t N_expected = 1;
                        for (uint32_t k = 0; k < r; ++k) N_expected *= p_hecke;
                        const uint64_t m_expected = N_expected + N_expected / p_hecke;
                        if (tables.N() != N_expected || tables.num_cosets() != m_expected) {
                            std::cout << "HeckeCosetTablesSelfCheckValidation: FAILED "
                                         "coset-count oracle for p_hecke=" << p_hecke
                                      << " r=" << r << "." << std::endl;
                            ++failures;
                        }
                    } catch (const std::exception& e) {
                        std::cout << "HeckeCosetTablesSelfCheckValidation: EXCEPTION for "
                                     "p_hecke=" << p_hecke << " r=" << r << " ell=" << ell
                                  << " n_sym=" << n_sym << ": " << e.what() << std::endl;
                        ++failures;
                    }
                }
            }
        }
    }

    std::cout << "HeckeCosetTablesSelfCheckValidation: " << (combinations - failures)
              << "/" << combinations << " (p_hecke, r, ell, n_sym) combinations passed."
              << std::endl;
    return failures;
}
