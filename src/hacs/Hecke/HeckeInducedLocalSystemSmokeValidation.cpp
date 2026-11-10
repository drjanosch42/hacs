// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Stage S5 smoke validation. See HeckeInducedLocalSystemSmokeValidation.h.

#include "HeckeInducedLocalSystemSmokeValidation.h"

#include "HeckeCosetTables.h"
#include "InducedLocalSystem.h"
#include "../LocalSystemSL2Z.h" // CreateSL2ZFundamentalDomainTriangulation
#include "../EquivariantCohomologyOnK.h"
#include "../../ModArithmetic/Matrix.h"
#include "../../ModArithmetic/FlatMatrix.h"

#include <iostream>
#include <vector>

namespace {

// Runs EquivariantCohomologyOnK on the given induced local system and checks
// (a) it completes without throwing, (b) get_H_eq_dimensions() has length
// dim(K)+1 = 3 (q = 0,1,2), matching the S5 DoD ("dimension vectors of the
// right length"). Also spot-checks every relation matrix M_H(g_r) for
// invertibility ([C, prop:lhcompliant]; the S5 plan-stage risk note).
// Returns 0 on success; prints diagnostics and returns nonzero otherwise.
uint32_t RunAndCheck(const char* label, LocalSystemSimplicialComplex& L,
                      const SimplicialComplex& K) {
    // Invertibility of every relation matrix M_H(g_r) ([C, prop:lhcompliant]).
    for (const auto& rel : L.get_user_defined_simplex_relations()) {
        FlatMatrix fm(rel.M);
        if (!fm.is_invertible()) {
            std::cout << "HeckeInducedLocalSystemSmokeValidation: " << label
                      << ": relation '" << rel.tag << "' matrix M_H(g_r) is NOT "
                         "invertible (violates [C, prop:lhcompliant])."
                      << std::endl;
            return 1;
        }
    }

    EquivariantCohomologyOnK coh(&L);
    coh.compute_cohomology();

    if (!coh.is_cohomology_computed()) {
        std::cout << "HeckeInducedLocalSystemSmokeValidation: " << label
                  << ": compute_cohomology() did not set is_cohomology_computed()."
                  << std::endl;
        return 2;
    }

    const std::vector<uint32_t>& dims = coh.get_H_eq_dimensions();
    const uint32_t expected_len = K.get_dimension() + 1;
    if (dims.size() != expected_len) {
        std::cout << "HeckeInducedLocalSystemSmokeValidation: " << label
                  << ": H_eq_dimensions has length " << dims.size()
                  << ", expected " << expected_len << "." << std::endl;
        return 3;
    }

    std::cout << "HeckeInducedLocalSystemSmokeValidation: " << label << ": dims =";
    for (uint32_t d : dims) std::cout << " " << d;
    std::cout << std::endl;

    return 0;
}

} // namespace

uint32_t HeckeInducedLocalSystemSmokeValidation() {
    try {
        const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();

        // p_hecke=2, r=1 -- the Gamma_0(2)/Gamma^0(2) worked example of
        // [C, sec:example], reused throughout the Hecke stages.
        const uint64_t p_hecke = 2;
        const uint32_t r = 1;

        // --- 1. Gamma_0(2), (n_sym, ell) = (0, 5) and (2, 5) --------------------
        {
            const uint32_t ell = 5, n_sym = 0;
            const hecke::HeckeCosetTables cosets(p_hecke, r, ell, n_sym);
            const uint32_t sc = cosets.self_check();
            if (sc != 0) {
                std::cout << "HeckeInducedLocalSystemSmokeValidation: "
                             "HeckeCosetTables::self_check() failed (code " << sc
                          << ") for p_hecke=2 r=1 ell=5 n_sym=0." << std::endl;
                return 10;
            }
            const uint32_t scu = cosets.self_check_upper();
            if (scu != 0) {
                std::cout << "HeckeInducedLocalSystemSmokeValidation: "
                             "HeckeCosetTables::self_check_upper() failed (code " << scu
                          << ") for p_hecke=2 r=1 ell=5 n_sym=0." << std::endl;
                return 11;
            }

            LocalSystemSimplicialComplex L =
                hecke::MakeInducedLocalSystemGamma0(K, cosets, ell, n_sym);
            const uint32_t code = RunAndCheck("Gamma_0(2), n=0, ell=5", L, K);
            if (code != 0) return 100 + code;
        }
        {
            const uint32_t ell = 5, n_sym = 2;
            const hecke::HeckeCosetTables cosets(p_hecke, r, ell, n_sym);
            LocalSystemSimplicialComplex L =
                hecke::MakeInducedLocalSystemGamma0(K, cosets, ell, n_sym);
            const uint32_t code = RunAndCheck("Gamma_0(2), n=2, ell=5", L, K);
            if (code != 0) return 200 + code;
        }

        // --- 2. Gamma^0(2), (n_sym, ell) = (0, 5) -- extra robustness ----------
        // (not part of the official S5 DoD, which only requires the
        // Gamma_0(N) variant; exercised here to de-risk S8, which consumes
        // MakeInducedLocalSystemGamma0Upper).
        {
            const uint32_t ell = 5, n_sym = 0;
            const hecke::HeckeCosetTables cosets(p_hecke, r, ell, n_sym);
            LocalSystemSimplicialComplex L =
                hecke::MakeInducedLocalSystemGamma0Upper(K, cosets, ell, n_sym);
            const uint32_t code = RunAndCheck("Gamma^0(2), n=0, ell=5", L, K);
            if (code != 0) return 300 + code;
        }

        return 0;
    }
    catch (const std::exception& e) {
        std::cout << "HeckeInducedLocalSystemSmokeValidation: exception: " << e.what()
                  << std::endl;
        return 400;
    }
}
