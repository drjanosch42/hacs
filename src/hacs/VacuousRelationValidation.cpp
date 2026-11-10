// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

/*
 * VacuousRelationValidation.cpp
 *
 * matrix-engine Phase 1, item 7.4. See the header for what is checked and why.
 *
 * Independence note: this file computes nothing itself. It drives the shipped
 * public pipeline twice through the SAME entry points (LocalSystemSL2Z's
 * relation generator -> LocalSystemSimplicialComplex -> EquivariantCohomologyOnK)
 * and compares the two runs against each other; the only difference between
 * them is the EquivariantCohomologyOptions switch. The relation classification
 * of part (b) is re-derived here from the relation list, not read out of the
 * cohomology code, so a bug in `relation_is_vacuous` would show up as a
 * disagreement rather than being echoed back.
 */

#include "VacuousRelationValidation.h"

#include "LocalSystemSL2Z.h"
#include "LocalSystemSimplicialComplex.h"
#include "EquivariantCohomologyOnK.h"
#include "SimplicialCohomology.h"

#include <cassert>
#include <iostream>
#include <vector>

namespace {

using Rel = LocalSystemSimplicialComplex::SimplexRelation;

std::vector<Rel> BuildRelations(const SimplicialComplex& K, uint32_t n_sym, uint32_t ell) {
    std::vector<Rel> rels;
    for (const auto& datum : GenerateSL2ZRelationsWithGroupData(K, n_sym, ell)) {
        rels.push_back(datum.rel);
    }
    return rels;
}

// The predicate under test, re-derived independently of EquivariantCohomologyOnK.cpp.
bool IsVacuous(const Rel& rel, uint32_t d_V) {
    if (rel.s_lhs_idx_in_Kq != rel.s_rhs_idx_in_Kq) return false;
    if (rel.M.get_rows() != d_V || rel.M.get_cols() != d_V) return false;
    for (uint32_t r = 0; r < d_V; ++r) {
        for (uint32_t c = 0; c < d_V; ++c) {
            if (rel.M.get_entry(r, c) != (r == c ? 1u : 0u)) return false;
        }
    }
    return true;
}

struct Run {
    std::vector<uint32_t> dims;
    std::vector<std::vector<FlatMatrix>> C_eq, Z_eq, B_eq, H_eq;
    std::vector<FlatMatrix> projectors;
};

Run Compute(const SimplicialComplex& K, const std::vector<Rel>& rels,
            uint32_t d_V, uint32_t ell, bool shortcircuit) {
    EquivariantCohomologyOptions::ScopedVacuousShortcircuit guard(shortcircuit);
    LocalSystemSimplicialComplex lssc(K, rels, d_V, ell);
    EquivariantCohomologyOnK eq(&lssc);
    eq.compute_cohomology();
    Run out;
    out.dims = eq.get_H_eq_dimensions();
    out.C_eq = eq.get_C_eq_bases_fm();
    out.Z_eq = eq.get_Z_eq_bases_fm();
    out.B_eq = eq.get_B_eq_bases_fm();
    out.H_eq = eq.get_H_eq_bases_fm();
    out.projectors = eq.get_H_eq_projectors_from_C_eq_flat_fm();
    return out;
}

bool SameMatrix(const FlatMatrix& a, const FlatMatrix& b) {
    if (a.get_rows() != b.get_rows() || a.get_cols() != b.get_cols() || a.get_p() != b.get_p()) return false;
    for (uint32_t r = 0; r < a.get_rows(); ++r) {
        for (uint32_t c = 0; c < a.get_cols(); ++c) {
            if (a.get_entry(r, c) != b.get_entry(r, c)) return false;
        }
    }
    return true;
}

bool SameBasis(const std::vector<std::vector<FlatMatrix>>& a,
               const std::vector<std::vector<FlatMatrix>>& b) {
    if (a.size() != b.size()) return false;
    for (size_t q = 0; q < a.size(); ++q) {
        if (a[q].size() != b[q].size()) return false;
        for (size_t i = 0; i < a[q].size(); ++i) {
            if (!SameMatrix(a[q][i], b[q][i])) return false;
        }
    }
    return true;
}

} // namespace

uint32_t VacuousRelationValidation(void) {
    const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    const uint32_t max_q = K.get_dimension();

    // 02 Sec. 5.4: seeds + one FIX-negI relation per simplex, per degree.
    const uint32_t expected_total[3] = {13, 13, 5};
    const uint32_t expected_live_even[3] = {6, 2, 0};

    const std::vector<uint32_t> primes = {2u, 3u, 5u, 691u, 134217757u};
    uint32_t failures = 0;

    for (uint32_t ell : primes) {
        for (uint32_t n_sym = 0; n_sym <= 12; ++n_sym) {
            const uint32_t d_V = n_sym + 1;
            const std::vector<Rel> rels = BuildRelations(K, n_sym, ell);

            // ---- (b) relation classification -------------------------------
            uint32_t total[3] = {0, 0, 0};
            uint32_t live[3] = {0, 0, 0};
            uint32_t nonself[3] = {0, 0, 0}; // relations with s_lhs != s_rhs
            for (const auto& rel : rels) {
                if (rel.q_dim > max_q) continue;
                total[rel.q_dim]++;
                if (rel.s_lhs_idx_in_Kq != rel.s_rhs_idx_in_Kq) nonself[rel.q_dim]++;
                if (!IsVacuous(rel, d_V)) live[rel.q_dim]++;
            }
            for (uint32_t q = 0; q <= max_q && q < 3; ++q) {
                if (total[q] != expected_total[q]) {
                    std::cout << "  [FAIL] n=" << n_sym << " ell=" << ell << " q=" << q
                              << ": relation count " << total[q] << " != " << expected_total[q]
                              << std::endl;
                    ++failures;
                }
            }

            // rho_n(-I) = (-1)^n I, so at even n (and at ANY n when ell == 2,
            // where -I == I) the FIX-negI relations are vacuous; at odd n with
            // ell != 2 none of them is.
            const bool negI_vacuous = (n_sym % 2 == 0) || (ell == 2);
            for (uint32_t q = 0; q <= max_q && q < 3; ++q) {
                // n_sym = 0 is the trivial representation: rho_0(g) = I for EVERY
                // g, so every self-relation is vacuous, seeds included -- the
                // model then keeps exactly its non-self relations (3, 2, 0).
                const uint32_t want = (n_sym == 0) ? nonself[q]
                                    : (negI_vacuous ? expected_live_even[q] : expected_total[q]);
                if (live[q] != want) {
                    std::cout << "  [FAIL] n=" << n_sym << " ell=" << ell << " q=" << q
                              << ": live relations " << live[q] << " != " << want << std::endl;
                    ++failures;
                }
            }
            // The A_C1 row-count shrink of 02 Sec. 5.4, stated in scalar rows.
            if (n_sym >= 2 && negI_vacuous && max_q >= 1) {
                const uint32_t rows_before = expected_total[1] * d_V;
                const uint32_t rows_after = live[1] * d_V;
                if (rows_after * expected_total[1] != rows_before * live[1] ||
                    live[1] != 2u) {
                    std::cout << "  [FAIL] n=" << n_sym << " ell=" << ell
                              << ": A_C1 rows " << rows_before << " -> " << rows_after
                              << " (expected a 13 -> 2 block-row shrink)" << std::endl;
                    ++failures;
                }
            }

            // ---- (a) the two code paths must agree bit-for-bit -------------
            const Run on = Compute(K, rels, d_V, ell, true);
            const Run off = Compute(K, rels, d_V, ell, false);

            if (on.dims != off.dims) {
                std::cout << "  [FAIL] n=" << n_sym << " ell=" << ell
                          << ": dim H^q differs between short-circuit on/off" << std::endl;
                ++failures;
            }
            if (!SameBasis(on.C_eq, off.C_eq)) {
                std::cout << "  [FAIL] n=" << n_sym << " ell=" << ell << ": C_eq basis differs" << std::endl;
                ++failures;
            }
            if (!SameBasis(on.Z_eq, off.Z_eq)) {
                std::cout << "  [FAIL] n=" << n_sym << " ell=" << ell << ": Z_eq basis differs" << std::endl;
                ++failures;
            }
            if (!SameBasis(on.B_eq, off.B_eq)) {
                std::cout << "  [FAIL] n=" << n_sym << " ell=" << ell << ": B_eq basis differs" << std::endl;
                ++failures;
            }
            if (!SameBasis(on.H_eq, off.H_eq)) {
                std::cout << "  [FAIL] n=" << n_sym << " ell=" << ell << ": H_eq basis differs" << std::endl;
                ++failures;
            }
            if (on.projectors.size() != off.projectors.size()) {
                std::cout << "  [FAIL] n=" << n_sym << " ell=" << ell << ": projector count differs" << std::endl;
                ++failures;
            } else {
                for (size_t q = 0; q < on.projectors.size(); ++q) {
                    if (!SameMatrix(on.projectors[q], off.projectors[q])) {
                        std::cout << "  [FAIL] n=" << n_sym << " ell=" << ell
                                  << " q=" << q << ": H_eq projector differs" << std::endl;
                        ++failures;
                    }
                }
            }

            // ---- (c) odd-n semantics (Hecke oracle O2) ---------------------
            // rho_n(-I) = -I for odd n; the relation 2 f(s) = 0 kills everything
            // as long as 2 is invertible, i.e. ell != 2.
            if (n_sym % 2 == 1 && ell != 2) {
                for (size_t q = 0; q < on.dims.size(); ++q) {
                    if (on.dims[q] != 0) {
                        std::cout << "  [FAIL] n=" << n_sym << " ell=" << ell
                                  << ": dim H^" << q << " = " << on.dims[q]
                                  << ", expected 0 (odd n, ell != 2)" << std::endl;
                        ++failures;
                    }
                }
            }
        }
    }

    if (failures == 0) {
        std::cout << "VacuousRelationValidation: 65 (n_sym, ell) cells, short-circuit on/off "
                     "agree on dims, C_eq/Z_eq/B_eq/H_eq bases and projectors." << std::endl;
    } else {
        std::cout << "VacuousRelationValidation: " << failures << " failure(s)." << std::endl;
    }
    return failures;
}
