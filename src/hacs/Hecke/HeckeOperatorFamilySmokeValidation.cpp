// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Stage S9 -- see HeckeOperatorFamilySmokeValidation.h.

#include "HeckeOperatorFamilySmokeValidation.h"

#include "HeckeOperatorFamily.h"
#include "SL2ZTessellationTables.h"
#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"
#include "../../ModArithmetic/ModArithmetic.h"

#include <iostream>
#include <vector>

namespace {

// Builds the (ell, n_sym) stack (base local system L = Sym^n(F_ell^2) on
// the 7-vertex model, its computed cohomology, and a HeckeOperatorFamily
// over it), then runs, for p in {2,3}:
//   (1) T_p_pow(p,1) == T_p(p)                          (r=1 recursion base);
//   (2) T_pp(p) == p^n_sym * I per degree               ([C, prop:diamond]);
//   (3) T_p_pow(p,2) == T_p_pow_direct(p,2)             (FULL recursion ==
//       FULL double-coset route -- the corrected O9 identity, holds for all
//       ell >= 5 now that both sides are the STANDARD full operator);
//   (4) T_p_pow_direct(p,2) == T_p_pow_primitive_direct(p,2) + p^n_sym * I
//       (explicit witness of the primitive-vs-full double-coset
//        decomposition T_{p^2}^full = T'_{p^2} + p^n I).
// Returns 0 on success; a distinguishing nonzero code otherwise.
uint32_t run_case(uint32_t ell, uint32_t n_sym) {
    const uint32_t d_V = n_sym + 1;

    const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    const hecke::SL2ZTessellationTables model(K);

    std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels;
    for (const auto& datum : GenerateSL2ZRelationsWithGroupData(K, n_sym, ell)) {
        rels.push_back(datum.rel);
    }
    LocalSystemSimplicialComplex lssc(K, rels, d_V, ell);
    EquivariantCohomologyOnK coh(&lssc);
    coh.compute_cohomology();

    hecke::HeckeOperatorFamily family(&coh, K, model, ell, n_sym);
    const auto& dims = coh.get_H_eq_dimensions();

    for (const uint64_t p : {2ull, 3ull}) {
        const uint32_t p_mod = static_cast<uint32_t>(p % ell);
        const uint32_t p_n = ModArithmetic::mod_pow(p_mod, n_sym, ell); // p^n (diamond scalar)

        // (1) T_p_pow(p,1) == T_p(p).
        const std::vector<Matrix>& Tp_ref = family.T_p(p);
        const std::vector<Matrix> Tp_pow1 = family.T_p_pow(p, 1);
        if (Tp_pow1.size() != Tp_ref.size()) return 10;
        for (size_t q = 0; q < Tp_ref.size(); ++q) {
            if (Tp_pow1[q] != Tp_ref[q]) {
                std::cout << "HeckeOperatorFamilySmokeValidation: T_p_pow(" << p
                          << ",1) != T_p(" << p << ") at degree " << q
                          << " (ell=" << ell << ", n=" << n_sym << ")." << std::endl;
                return 11;
            }
        }

        // (2) T_pp(p) == p^n_sym * I per degree.
        const std::vector<Matrix> Tpp = family.T_pp(p);
        if (Tpp.size() != dims.size()) return 20;
        for (size_t q = 0; q < dims.size(); ++q) {
            Matrix expected(dims[q], ell);
            expected = expected * p_n;
            if (Tpp[q] != expected) {
                std::cout << "HeckeOperatorFamilySmokeValidation: T_pp(" << p
                          << ") != " << p_n << "*I at degree " << q
                          << " (ell=" << ell << ", n=" << n_sym << ")." << std::endl;
                return 21;
            }
        }

        // (3) T_p_pow(p,2) == T_p_pow_direct(p,2): FULL recursion == FULL
        // double-coset route (the corrected O9 identity; both are the
        // standard determinant-p^2 operator, so equal for all ell >= 5).
        const std::vector<Matrix> rec = family.T_p_pow(p, 2);
        const std::vector<Matrix> full_direct = family.T_p_pow_direct(p, 2);
        if (rec.size() != full_direct.size()) return 30;
        for (size_t q = 0; q < rec.size(); ++q) {
            if (rec[q] != full_direct[q]) {
                std::cout << "HeckeOperatorFamilySmokeValidation: T_p_pow(" << p
                          << ",2) != T_p_pow_direct(" << p << ",2) at degree " << q
                          << " (ell=" << ell << ", n=" << n_sym << ")." << std::endl;
                std::cout << "  recursion (full):\n" << rec[q].to_string() << std::endl;
                std::cout << "  direct (full):\n" << full_direct[q].to_string() << std::endl;
                return 31;
            }
        }

        // (4) Decomposition witness:
        //     T_p_pow_direct(p,2) == T_p_pow_primitive_direct(p,2) + p^n * I.
        const std::vector<Matrix> prim = family.T_p_pow_primitive_direct(p, 2);
        if (prim.size() != full_direct.size()) return 40;
        for (size_t q = 0; q < full_direct.size(); ++q) {
            Matrix I(dims[q], ell);
            Matrix expected = prim[q] + (I * p_n);
            if (full_direct[q] != expected) {
                std::cout << "HeckeOperatorFamilySmokeValidation: decomposition "
                             "T_p_pow_direct("
                          << p << ",2) != T'_{p^2} + " << p_n << "*I at degree " << q
                          << " (ell=" << ell << ", n=" << n_sym << ")." << std::endl;
                std::cout << "  full:\n" << full_direct[q].to_string() << std::endl;
                std::cout << "  primitive:\n" << prim[q].to_string() << std::endl;
                return 41;
            }
        }
    }

    return 0;
}

} // namespace

uint32_t HeckeOperatorFamilySmokeValidation() {
    try {
        // Grid ell in {5,13,101}, n_sym in {0,2,10}.
        for (const uint32_t ell : {5u, 13u, 101u}) {
            for (const uint32_t n_sym : {0u, 2u, 10u}) {
                const uint32_t code = run_case(ell, n_sym);
                if (code != 0) {
                    std::cout << "HeckeOperatorFamilySmokeValidation: FAILED at ell="
                              << ell << ", n_sym=" << n_sym << " (code " << code << ")."
                              << std::endl;
                    return 1000 + code;
                }
            }
        }

        std::cout << "HeckeOperatorFamilySmokeValidation: PASSED "
                     "(for p in {2,3}, ell in {5,13,101}, n in {0,2,10}: "
                     "T_p_pow(p,1)==T_p(p), T_pp(p)==p^n*I, "
                     "T_p_pow(p,2)==T_p_pow_direct(p,2) [FULL recursion == FULL "
                     "double-coset], and T_p_pow_direct(p,2)==T'_{p^2}+p^n*I "
                     "[primitive-vs-full decomposition witness])."
                  << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cout << "HeckeOperatorFamilySmokeValidation: EXCEPTION: " << e.what() << std::endl;
        return 1;
    }
}
