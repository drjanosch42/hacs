// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Stage S8 -- see HeckeCrossCheckValidation.h.

#include "HeckeCrossCheckValidation.h"

#include "BarCocycleH1.h"
#include "HeckeCosetTables.h"
#include "HeckeCrossChecks.h"
#include "HeckeOperatorOnK.h"
#include "HeckeTauTable.h"
#include "SL2ZTessellationTables.h"
#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string poly_to_string(const std::vector<uint32_t>& poly) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < poly.size(); ++i) {
        if (i) oss << ", ";
        oss << poly[i];
    }
    oss << "] (low -> high)";
    return oss.str();
}

// One DoD grid point. Returns 0 on success; nonzero codes localize the
// failing step (printed alongside the (p, ell, n) triple).
uint32_t run_grid_point(const SimplicialComplex& K,
                        const hecke::SL2ZTessellationTables& model, uint64_t p_hecke,
                        uint32_t ell, uint32_t n_sym) {
    const uint32_t d_V = n_sym + 1;

    // Base system L = Sym^n(F_ell^2) (relations derived from the S2 group
    // data, invariant I3).
    std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels;
    for (const auto& datum : GenerateSL2ZRelationsWithGroupData(K, n_sym, ell)) {
        rels.push_back(datum.rel);
    }
    LocalSystemSimplicialComplex lssc(K, rels, d_V, ell);
    EquivariantCohomologyOnK coh(&lssc);
    coh.compute_cohomology();

    // S3 + S4 + S7 stack for T_p (g = diag(1, p), r = 1).
    const hecke::HeckeCosetTables cosets(p_hecke, /*r=*/1, ell, n_sym);
    const hecke::HeckeTauTable tau(K, model, cosets, ell, n_sym);
    hecke::HeckeOperatorOnK op(&coh, &cosets, &tau);
    op.compute();

    // ---- Route A: three-map factorization (A-F3 (i), (ii), then the
    // ---- acceptance bit-identity against Theta_g). ------------------------
    const hecke::HeckeCrossChecks xc(&coh, &cosets, &tau);
    const uint32_t code_a = xc.run_all(op);
    if (code_a != 0) {
        std::cout << "HeckeCrossCheckValidation: three-map route FAILED (code "
                  << code_a << ") at (p, ell, n) = (" << p_hecke << ", " << ell
                  << ", " << n_sym << ")." << std::endl;
        return code_a; // 101/102, 201/202, 301..304 per HeckeCrossChecks::run_all
    }

    // ---- Route B: bar-cocycle H^1 (ell >= 5 on the whole grid). -----------
    const hecke::BarCocycleH1 bar(ell, n_sym);

    // Internal consistency: both routes compute H^1(Gamma; Sym^n F_ell)
    // (ell >= 5, [C, prop:lhcompliant]-side equivalence) -- dimensions must
    // agree.
    const uint32_t dim_pipeline = coh.get_H_eq_dimensions()[1];
    if (bar.dim_H1() != dim_pipeline) {
        std::cout << "HeckeCrossCheckValidation: bar-route dim H^1 = " << bar.dim_H1()
                  << " != pipeline dim H^1 = " << dim_pipeline << " at (p, ell, n) = ("
                  << p_hecke << ", " << ell << ", " << n_sym << ")." << std::endl;
        return 400;
    }

    const FlatMatrix T_bar = bar.hecke_matrix(cosets);
    const Matrix& A1 = op.get_H_matrices()[1];
    if (T_bar.get_rows() != A1.rows || T_bar.get_cols() != A1.cols) {
        std::cout << "HeckeCrossCheckValidation: bar Hecke matrix shape " << T_bar.get_rows()
                  << "x" << T_bar.get_cols() << " != pipeline " << A1.rows << "x" << A1.cols
                  << " at (p, ell, n) = (" << p_hecke << ", " << ell << ", " << n_sym
                  << ")." << std::endl;
        return 401;
    }

    const std::vector<uint32_t> cp_bar = hecke::CharPolyOfMatrix(T_bar);
    const std::vector<uint32_t> cp_pipe = hecke::CharPolyOfMatrix(A1);
    if (cp_bar != cp_pipe) {
        std::cout << "HeckeCrossCheckValidation: CHARPOLY MISMATCH at (p, ell, n) = ("
                  << p_hecke << ", " << ell << ", " << n_sym << "):\n"
                  << "  bar route:  " << poly_to_string(cp_bar) << "\n"
                  << "  pipeline:   " << poly_to_string(cp_pipe) << std::endl;
        return 402;
    }
    const std::vector<uint32_t> mp_bar = hecke::MinPolyOfMatrix(T_bar);
    const std::vector<uint32_t> mp_pipe = hecke::MinPolyOfMatrix(A1);
    if (mp_bar != mp_pipe) {
        std::cout << "HeckeCrossCheckValidation: MINPOLY MISMATCH at (p, ell, n) = ("
                  << p_hecke << ", " << ell << ", " << n_sym << "):\n"
                  << "  bar route:  " << poly_to_string(mp_bar) << "\n"
                  << "  pipeline:   " << poly_to_string(mp_pipe) << std::endl;
        return 403;
    }

    std::cout << "HeckeCrossCheckValidation: (p, ell, n) = (" << p_hecke << ", " << ell
              << ", " << n_sym << ") OK -- omega.beta.alpha == Theta_g (q = 0,1,2), "
              << "dim H^1 = " << dim_pipeline
              << ", charpoly = " << poly_to_string(cp_bar)
              << ", minpoly = " << poly_to_string(mp_bar) << std::endl;
    return 0;
}

} // namespace

uint32_t HeckeCrossCheckValidation() {
    try {
        const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
        const hecke::SL2ZTessellationTables model(K);

        // DoD grid (plan stage S8): (p, ell, n) in {2,3} x {5,13,691} x {0,2,10}.
        const uint64_t hecke_primes[] = { 2, 3 };
        const uint32_t ells[] = { 5, 13, 691 };
        const uint32_t ns[] = { 0, 2, 10 };

        for (uint64_t p : hecke_primes) {
            for (uint32_t ell : ells) {
                for (uint32_t n : ns) {
                    const uint32_t code = run_grid_point(K, model, p, ell, n);
                    if (code != 0) return code;
                }
            }
        }

        std::cout << "HeckeCrossCheckValidation: PASSED -- all 18 grid points: "
                     "A-F3 (i)+(ii) hold, omega.beta.alpha == Theta_g "
                     "bit-identically on C^q(K;V) (q = 0,1,2), and the bar-cocycle "
                     "route matches charpoly AND minpoly of A^(1)(T_p)."
                  << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cout << "HeckeCrossCheckValidation: EXCEPTION: " << e.what() << std::endl;
        return 1;
    }
}
