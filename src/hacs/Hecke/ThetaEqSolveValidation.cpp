// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// matrix-engine Phase 1, item 7.1 -- see ThetaEqSolveValidation.h.

#include "ThetaEqSolveValidation.h"

#include "HeckeCosetTables.h"
#include "HeckeOperatorOnK.h"
#include "HeckeTauTable.h"
#include "SL2ZTessellationTables.h"
#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"
#include "../../ModArithmetic/FlatMatrix.h"
#include "../../ModArithmetic/ModArithmetic.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

// The binding flat layout ([C, sec:conventions]): an N_q x d_V cochain
// (row per simplex) flattens row-major to index s*d_V + k. Same as
// HeckeOperatorOnK::flatten_cochain (private there) and
// EquivariantCohomologyOnK's matrix_to_flat_column.
FlatMatrix flatten_cochain(const FlatMatrix& cochain, uint32_t ell) {
    const uint32_t total = cochain.get_rows() * cochain.get_cols();
    std::vector<uint32_t> flat(total, 0);
    for (uint32_t r = 0; r < cochain.get_rows(); ++r) {
        for (uint32_t c = 0; c < cochain.get_cols(); ++c) {
            flat[r * cochain.get_cols() + c] = cochain.get_entry(r, c);
        }
    }
    return total == 0 ? FlatMatrix::CreateZero(0, 1, ell) : FlatMatrix(total, 1, ell, flat);
}

// ---------------------------------------------------------------------------
// (A) One (n_sym, ell, p) grid point of the pipeline comparison.
// Returns 0 on success; nonzero codes localize the failure.
// ---------------------------------------------------------------------------

uint32_t run_point(const SimplicialComplex& K, const hecke::SL2ZTessellationTables& model,
                   uint32_t n_sym, uint32_t ell, uint64_t p_hecke,
                   uint64_t& columns_compared) {
    const uint32_t d_V = n_sym + 1;

    std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels;
    for (const auto& datum : GenerateSL2ZRelationsWithGroupData(K, n_sym, ell)) {
        rels.push_back(datum.rel);
    }

    LocalSystemSimplicialComplex lssc(K, rels, d_V, ell);
    EquivariantCohomologyOnK coh(&lssc);
    coh.compute_cohomology();

    const hecke::HeckeCosetTables cosets(p_hecke, /*r=*/1, ell, n_sym);
    const hecke::HeckeTauTable tau(K, model, cosets, ell, n_sym);
    hecke::HeckeOperatorOnK op(&coh, &cosets, &tau);
    op.compute(); // runs the NEW multi-RHS solve + its residual certificate

    const std::vector<FlatMatrix>& theta_flat = op.get_Theta_flat_matrices();
    const std::vector<FlatMatrix>& theta_eq = op.get_Theta_eq_matrices();
    const auto& C_eq_bases = coh.get_C_eq_bases_fm();

    for (uint32_t q = 0; q < theta_flat.size(); ++q) {
        const uint32_t N_q = K.get_num_q_simplices(q);
        const uint32_t dim = N_q * d_V;
        const uint32_t dim_C_eq = static_cast<uint32_t>(C_eq_bases[q].size());
        if (dim_C_eq == 0) continue;

        std::vector<FlatMatrix> c_eq_flat;
        c_eq_flat.reserve(dim_C_eq);
        for (const FlatMatrix& c_j : C_eq_bases[q]) c_eq_flat.push_back(flatten_cochain(c_j, ell));

        std::vector<FlatMatrix> w_all;
        w_all.reserve(dim_C_eq);
        for (uint32_t j = 0; j < dim_C_eq; ++j) w_all.push_back(theta_flat[q] * c_eq_flat[j]);

        // NEW: one augmented RREF for all right-hand sides.
        const std::vector<std::vector<uint32_t>> coords_new =
            change_of_basis_fm_multi(c_eq_flat, w_all);
        if (coords_new.size() != dim_C_eq) return 10;

        for (uint32_t j = 0; j < dim_C_eq; ++j) {
            // OLD (reference): one independent augmented RREF per vector.
            const std::vector<uint32_t> coords_old = change_of_basis_fm(c_eq_flat, w_all[j]);
            if (coords_old.size() != dim_C_eq || coords_new[j].size() != dim_C_eq) return 11;

            for (uint32_t rr = 0; rr < dim_C_eq; ++rr) {
                // (A1) bit-identity new vs old.
                if (coords_new[j][rr] != coords_old[rr]) {
                    std::cout << "ThetaEqSolveValidation: multi-RHS solve differs from the "
                                 "per-vector solve at (n=" << n_sym << ", ell=" << ell
                              << ", p=" << p_hecke << ") q=" << q << ", column " << j
                              << ", coordinate " << rr << ": " << coords_new[j][rr]
                              << " vs " << coords_old[rr] << std::endl;
                    return 12;
                }
                // (A2) both agree with what the pipeline published.
                if (theta_eq[q].get_entry(rr, j) != coords_old[rr]) {
                    std::cout << "ThetaEqSolveValidation: published Theta_eq differs from the "
                                 "per-vector solve at (n=" << n_sym << ", ell=" << ell
                              << ", p=" << p_hecke << ") q=" << q << ", entry (" << rr
                              << "," << j << "): " << theta_eq[q].get_entry(rr, j) << " vs "
                              << coords_old[rr] << std::endl;
                    return 13;
                }
            }

            // (A3) the A-N3 residual certificate, recomputed here against the
            // UNFACTORED data: w_j - sum_r coords_r * c_r must vanish exactly.
            for (uint32_t row = 0; row < dim; ++row) {
                uint32_t acc = 0;
                for (uint32_t rr = 0; rr < dim_C_eq; ++rr) {
                    if (coords_new[j][rr] == 0) continue;
                    acc = ModArithmetic::mod_add(
                        acc,
                        ModArithmetic::mod_mul(coords_new[j][rr],
                                               c_eq_flat[rr].get_entry(row, 0), ell),
                        ell);
                }
                if (ModArithmetic::mod_sub(w_all[j].get_entry(row, 0), acc, ell) != 0) {
                    std::cout << "ThetaEqSolveValidation: nonzero residual at (n=" << n_sym
                              << ", ell=" << ell << ", p=" << p_hecke << ") q=" << q
                              << ", column " << j << ", flat row " << row << std::endl;
                    return 14;
                }
            }
            ++columns_compared;
        }
    }
    return 0;
}

// ---------------------------------------------------------------------------
// (B) Synthetic systems, independent of the Hecke stack.
// ---------------------------------------------------------------------------

uint32_t next_rand(uint64_t& state) { // xorshift64*, fixed seed: deterministic
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return static_cast<uint32_t>((state * 2685821657736338717ull) >> 33);
}

uint32_t run_synthetic() {
    const uint32_t primes[] = { 2u, 3u, 7u, 101u, 134217757u };
    uint64_t state = 0x9E3779B97F4A7C15ull;

    for (uint32_t ell : primes) {
        for (uint32_t rows : { 1u, 4u, 9u, 17u }) {
            for (uint32_t k = 1; k <= rows && k <= 6u; ++k) {
                // A basis of k independent columns: take the first k columns of
                // a unit-lower-triangular matrix padded with random entries
                // (always independent, so the solve must succeed).
                std::vector<FlatMatrix> basis;
                basis.reserve(k);
                for (uint32_t j = 0; j < k; ++j) {
                    std::vector<uint32_t> col(rows, 0);
                    col[j] = 1;
                    for (uint32_t i = j + 1; i < rows; ++i) col[i] = next_rand(state) % ell;
                    basis.emplace_back(rows, 1, ell, col);
                }
                // Right-hand sides: random combinations of the basis.
                const uint32_t num_rhs = 5;
                std::vector<FlatMatrix> rhs;
                rhs.reserve(num_rhs);
                for (uint32_t t = 0; t < num_rhs; ++t) {
                    std::vector<uint32_t> v(rows, 0);
                    for (uint32_t j = 0; j < k; ++j) {
                        const uint32_t lam = next_rand(state) % ell;
                        if (lam == 0) continue;
                        for (uint32_t i = 0; i < rows; ++i) {
                            v[i] = ModArithmetic::mod_add(
                                v[i], ModArithmetic::mod_mul(lam, basis[j].get_entry(i, 0), ell),
                                ell);
                        }
                    }
                    rhs.emplace_back(rows, 1, ell, v);
                }

                const std::vector<std::vector<uint32_t>> multi =
                    change_of_basis_fm_multi(basis, rhs);
                if (multi.size() != num_rhs) return 20;
                for (uint32_t t = 0; t < num_rhs; ++t) {
                    const std::vector<uint32_t> single = change_of_basis_fm(basis, rhs[t]);
                    if (single.size() != multi[t].size()) return 21;
                    for (uint32_t j = 0; j < k; ++j) {
                        if (single[j] != multi[t][j]) {
                            std::cout << "ThetaEqSolveValidation: synthetic mismatch at ell="
                                      << ell << ", rows=" << rows << ", k=" << k << ", rhs="
                                      << t << ", coord " << j << ": " << multi[t][j] << " vs "
                                      << single[j] << std::endl;
                            return 22;
                        }
                    }
                }

                // Edge case: no right-hand side at all.
                if (!change_of_basis_fm_multi(basis, std::vector<FlatMatrix>{}).empty()) return 23;

                // Rejection: a vector outside span(basis) must still throw.
                // (Only possible when the basis does not already span F_ell^rows.)
                // Rows 0..k-1 of the basis are exactly I_k, so a combination
                // that vanishes there has all coefficients 0 and cannot equal
                // e_k -- e_k is outside the span.
                if (k < rows) {
                    std::vector<uint32_t> out(rows, 0);
                    out[k] = 1;
                    std::vector<FlatMatrix> bad = rhs;
                    bad.emplace_back(rows, 1, ell, out);
                    bool threw = false;
                    try {
                        (void)change_of_basis_fm_multi(basis, bad);
                    } catch (const std::exception&) {
                        threw = true;
                    }
                    if (!threw) {
                        std::cout << "ThetaEqSolveValidation: multi-RHS solve accepted a vector "
                                     "outside span(basis) at ell=" << ell << ", rows=" << rows
                                  << ", k=" << k << std::endl;
                        return 24;
                    }
                }
            }
        }
    }
    return 0;
}

} // namespace

uint32_t ThetaEqSolveValidation() {
    try {
        const uint32_t code_syn = run_synthetic();
        if (code_syn != 0) return code_syn;

        const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
        const hecke::SL2ZTessellationTables model(K);

        // Small/medium points of the golden grid (both parities of n_sym, the
        // small primes where dim C^q_eq is largest relative to the cost, one
        // 27-bit prime, and both p_hecke = 2 and 3).
        struct Point { uint32_t n_sym; uint32_t ell; uint64_t p_hecke; };
        const Point points[] = {
            { 0u,  5u,         2u },
            { 1u,  3u,         2u },
            { 2u,  691u,       2u },
            { 2u,  5u,         3u },
            { 4u,  13u,        2u },
            { 4u,  134217757u, 3u },
            { 10u, 134217757u, 2u },
            { 16u, 5u,         2u },
            { 22u, 3u,         2u },
            { 24u, 134217757u, 2u },
            { 32u, 13u,        3u },
            { 48u, 134217757u, 2u },
        };

        uint64_t columns_compared = 0;
        uint32_t points_run = 0;
        uint32_t points_excluded = 0;
        for (const Point& pt : points) {
            uint32_t code = 0;
            try {
                code = run_point(K, model, pt.n_sym, pt.ell, pt.p_hecke, columns_compared);
            } catch (const std::invalid_argument& e) {
                // A point the shipped predicates reject (same handling as the
                // golden harness): record and move on.
                std::cout << "ThetaEqSolveValidation: point (n=" << pt.n_sym << ", ell="
                          << pt.ell << ", p=" << pt.p_hecke << ") EXCLUDED by a shipped "
                             "predicate: " << e.what() << std::endl;
                ++points_excluded;
                continue;
            }
            if (code != 0) {
                std::cout << "ThetaEqSolveValidation: FAILED at (n=" << pt.n_sym << ", ell="
                          << pt.ell << ", p=" << pt.p_hecke << ") with code " << code
                          << std::endl;
                return code;
            }
            ++points_run;
        }

        if (columns_compared == 0) {
            std::cout << "ThetaEqSolveValidation: no C_eq column was compared -- the grid "
                         "degenerated, the comparison proves nothing." << std::endl;
            return 30;
        }

        std::cout << "ThetaEqSolveValidation: PASSED (" << points_run << " grid points, "
                  << points_excluded << " excluded, " << columns_compared
                  << " Theta_eq columns bit-identical to the per-vector solve and to the "
                     "published Theta_eq, all residuals exactly 0; synthetic multi-RHS "
                     "systems match the single-RHS solver and reject out-of-span vectors)."
                  << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cout << "ThetaEqSolveValidation: unexpected exception: " << e.what() << std::endl;
        return 99;
    }
}
