// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Stage S2 smoke validation. See HeckeTessellationSmokeValidation.h.

#include "HeckeTessellationSmokeValidation.h"

#include "SL2ZTessellationTables.h"
#include "ExactInt.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemUtils.h"
#include "../SimplicialCohomology.h"
#include "../../ModArithmetic/Matrix.h"
#include "../../ModArithmetic/FlatMatrix.h"

#include <array>
#include <iostream>
#include <vector>

namespace {

// Fixed-seed xorshift64 (deterministic across runs and build settings).
struct XorShift64 {
    uint64_t state;
    explicit XorShift64(uint64_t seed) : state(seed) {}
    uint64_t next() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    }
    // Uniform-ish integer in [0, n) (n small; modulo bias irrelevant here).
    uint64_t below(uint64_t n) { return next() % n; }
};

// Pseudorandom element of SL(2,Z) as a bounded word in T, S (det == 1 by
// construction); entry growth capped so all products stay far inside the
// overflow-checked int64 regime.
hecke::Mat22 random_gamma(XorShift64& rng, int max_letters) {
    hecke::Mat22 g = hecke::Mat22::I();
    const int letters = 1 + static_cast<int>(rng.below(static_cast<uint64_t>(max_letters)));
    for (int i = 0; i < letters; ++i) {
        hecke::Mat22 factor = hecke::Mat22::I();
        if (rng.below(2) == 0) {
            int64_t e = static_cast<int64_t>(rng.below(17)) - 8; // [-8, 8]
            if (e == 0) e = 1;
            factor = hecke::Mat22{ 1, e, 0, 1 }; // T^e
        } else {
            factor = hecke::Mat22::S();
        }
        const hecke::Mat22 candidate = g * factor;
        if (candidate.max_abs() > (int64_t(1) << 24)) break; // cap growth
        g = candidate;
    }
    if (rng.below(2) == 0) g = g.neg(); // exercise the -I quotient
    return g;
}

// Rebuilds the pre-S2 legacy relation list VERBATIM (the exact code path of
// GenerateSL2ZRelations_7VertexModel before the Stage S2 refactor), for the
// bit-identity comparison of the S2 Definition of Done.
//
// FIX-negI note: since the central -I fix, the current generators return
// these 8 seed relations FIRST (bit-identical, checked below) and then
// append one -I self-relation per simplex of K (also checked below); the
// pre-fix 8-relation list was mathematically incomplete (missing the -I
// equivariance constraint of [C, thm:explicit_complex_equiv] on the free
// simplices), so this replica is compared only against the seed prefix.
std::vector<LocalSystemSimplicialComplex::SimplexRelation>
legacy_relations_pre_s2(const SimplicialComplex& K_sl2z,
                        const Matrix& M_T, const Matrix& M_S,
                        const Matrix& M_ST, const Matrix& M_TS) {
    std::vector<LocalSystemSimplicialComplex::SimplexRelation> sl2z_relations;
    sl2z_relations.emplace_back(0, find_simplex_idx(K_sl2z, 0, { 3 }), find_simplex_idx(K_sl2z, 0, { 0 }), M_T, "T_v3_v0");
    sl2z_relations.emplace_back(0, find_simplex_idx(K_sl2z, 0, { 5 }), find_simplex_idx(K_sl2z, 0, { 4 }), M_T, "T_v5_v4");
    sl2z_relations.emplace_back(1, find_simplex_idx(K_sl2z, 1, { 3,5 }), find_simplex_idx(K_sl2z, 1, { 0,4 }), M_T, "T_e(0,4)_e(3,5)");
    sl2z_relations.emplace_back(0, find_simplex_idx(K_sl2z, 0, { 4 }), find_simplex_idx(K_sl2z, 0, { 5 }), M_S, "S_v4_v5");
    sl2z_relations.emplace_back(0, find_simplex_idx(K_sl2z, 0, { 6 }), find_simplex_idx(K_sl2z, 0, { 6 }), M_S, "S_v6_stabilizer");
    sl2z_relations.emplace_back(1, find_simplex_idx(K_sl2z, 1, { 4,6 }), find_simplex_idx(K_sl2z, 1, { 5,6 }), M_S, "S_e(5,6)_e(4,6)");
    sl2z_relations.emplace_back(0, find_simplex_idx(K_sl2z, 0, { 4 }), find_simplex_idx(K_sl2z, 0, { 4 }), M_ST, "ST_v4_stabilizer");
    sl2z_relations.emplace_back(0, find_simplex_idx(K_sl2z, 0, { 5 }), find_simplex_idx(K_sl2z, 0, { 5 }), M_TS, "TS_v5_stabilizer");
    return sl2z_relations;
}

uint32_t check_legacy_bit_identity(const SimplicialComplex& K, uint32_t n_sym, uint32_t ell) {
    // Pre-S2 path: symmetric powers through the legacy Matrix pipeline.
    Matrix T0(2, 2, ell, { {1, 1}, {0, 1} });
    Matrix S0(2, 2, ell, { {0, ell - 1}, {1, 0} });
    Matrix ST0 = S0 * T0;
    Matrix TS0 = T0 * S0;
    const Matrix M_T = ComputeSymmetricPowerMatrixFM(FlatMatrix(T0), n_sym).to_matrix();
    const Matrix M_S = ComputeSymmetricPowerMatrixFM(FlatMatrix(S0), n_sym).to_matrix();
    const Matrix M_ST = ComputeSymmetricPowerMatrixFM(FlatMatrix(ST0), n_sym).to_matrix();
    const Matrix M_TS = ComputeSymmetricPowerMatrixFM(FlatMatrix(TS0), n_sym).to_matrix();

    const auto old_path = legacy_relations_pre_s2(K, M_T, M_S, M_ST, M_TS);
    const auto new_path = GenerateSL2ZRelations_7VertexModel(K, n_sym, ell, M_T, M_S, M_ST, M_TS);
    const auto with_group_data = GenerateSL2ZRelationsWithGroupData(K, n_sym, ell);

    // FIX-negI: the generators return the 8 seed relations first, then one
    // -I self-relation per simplex of K (8 + 23 = 31 for the 7-vertex model).
    size_t num_simplices = 0;
    for (uint32_t q = 0; q <= K.get_dimension(); ++q) {
        num_simplices += K.get_num_q_simplices(q);
    }
    const size_t expected_size = 8 + num_simplices;
    if (old_path.size() != 8 || new_path.size() != expected_size ||
        with_group_data.size() != expected_size) return 40;

    const hecke::Mat22 expected_g[8] = {
        hecke::Mat22::T(), hecke::Mat22::T(), hecke::Mat22::T(),
        hecke::Mat22::S(), hecke::Mat22::S(), hecke::Mat22::S(),
        hecke::Mat22::ST(), hecke::Mat22::TS(),
    };

    for (size_t r = 0; r < 8; ++r) {
        // Bit-identity of the refactored legacy function vs. the pre-S2 code.
        if (new_path[r].q_dim != old_path[r].q_dim) return 41;
        if (new_path[r].s_lhs_idx_in_Kq != old_path[r].s_lhs_idx_in_Kq) return 42;
        if (new_path[r].s_rhs_idx_in_Kq != old_path[r].s_rhs_idx_in_Kq) return 43;
        if (new_path[r].tag != old_path[r].tag) return 44;
        if (!(new_path[r].M == old_path[r].M)) return 45;
        // The group-data variant carries the same relation ...
        if (with_group_data[r].rel.q_dim != old_path[r].q_dim ||
            with_group_data[r].rel.s_lhs_idx_in_Kq != old_path[r].s_lhs_idx_in_Kq ||
            with_group_data[r].rel.s_rhs_idx_in_Kq != old_path[r].s_rhs_idx_in_Kq ||
            with_group_data[r].rel.tag != old_path[r].tag ||
            !(with_group_data[r].rel.M == old_path[r].M)) return 46;
        // ... the generator dictionary of [C, sec:model] ...
        if (!(with_group_data[r].g_r == expected_g[r])) return 47;
        // ... and the I3 derivation: M == rho_n(g_r) re-derived here.
        const std::array<uint32_t, 4> gm = with_group_data[r].g_r.reduced_mod(ell);
        const FlatMatrix g_fm(2, 2, ell, std::vector<uint32_t>{ gm[0], gm[1], gm[2], gm[3] });
        const Matrix rho = ComputeSymmetricPowerMatrixFM(g_fm, n_sym).to_matrix();
        if (!(with_group_data[r].rel.M == rho)) return 48;
    }

    // FIX-negI: the trailing rows must be -I self-relations covering every
    // simplex of every dimension exactly once, with M == rho_n(-I)
    // re-derived here per invariant I3, identical between both generators.
    const hecke::Mat22 neg_I = hecke::Mat22::I().neg();
    const std::array<uint32_t, 4> nm = neg_I.reduced_mod(ell);
    const FlatMatrix nI_fm(2, 2, ell, std::vector<uint32_t>{ nm[0], nm[1], nm[2], nm[3] });
    const Matrix rho_neg_I = ComputeSymmetricPowerMatrixFM(nI_fm, n_sym).to_matrix();
    std::vector<std::vector<bool>> covered(K.get_dimension() + 1);
    for (uint32_t q = 0; q <= K.get_dimension(); ++q) {
        covered[q].assign(K.get_num_q_simplices(q), false);
    }
    for (size_t r = 8; r < expected_size; ++r) {
        const auto& G = with_group_data[r].rel;
        const auto& N = new_path[r];
        if (!(with_group_data[r].g_r == neg_I)) return 49;
        if (G.s_lhs_idx_in_Kq != G.s_rhs_idx_in_Kq) return 50;
        if (!(G.M == rho_neg_I)) return 51;
        if (N.q_dim != G.q_dim || N.s_lhs_idx_in_Kq != G.s_lhs_idx_in_Kq ||
            N.s_rhs_idx_in_Kq != G.s_rhs_idx_in_Kq || N.tag != G.tag ||
            !(N.M == G.M)) return 52;
        if (G.q_dim > K.get_dimension() ||
            G.s_lhs_idx_in_Kq >= K.get_num_q_simplices(G.q_dim)) return 53;
        if (covered[G.q_dim][G.s_lhs_idx_in_Kq]) return 54;
        covered[G.q_dim][G.s_lhs_idx_in_Kq] = true;
    }
    for (uint32_t q = 0; q <= K.get_dimension(); ++q) {
        for (uint32_t s = 0; s < K.get_num_q_simplices(q); ++s) {
            if (!covered[q][s]) return 55;
        }
    }
    return 0;
}

} // namespace

uint32_t HeckeTessellationSmokeValidation() {
    try {
        const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
        const hecke::SL2ZTessellationTables tables(K);

        // --- 1. Closure fixed point + table self-checks -----------------------
        {
            const uint32_t code = tables.validate_closure();
            if (code != 0) {
                std::cout << "HeckeTessellationSmokeValidation: validate_closure failed (code "
                          << code << ")." << std::endl;
                return 1;
            }
        }

        // --- 2. canonical_key: idempotence + constancy under rewriting --------
        {
            XorShift64 rng(0x53325A544B657931ull); // fixed seed
            const uint32_t num_simplices[3] = {
                K.get_num_q_simplices(0), K.get_num_q_simplices(1), K.get_num_q_simplices(2) };
            for (int iter = 0; iter < 10000; ++iter) {
                const uint32_t q = static_cast<uint32_t>(rng.below(3));
                uint32_t sigma = static_cast<uint32_t>(rng.below(num_simplices[q]));
                hecke::Mat22 gamma = random_gamma(rng, 10);

                const hecke::KuSimplexKey key0 = tables.canonical_key(gamma, q, sigma);

                // Idempotence: the key of the canonical representative is the
                // key itself (all vertex correspondences of this model are
                // order-compatible, so all signs are +1 -- checked in
                // validate_closure; hence full equality).
                const hecke::KuSimplexKey key_of_key =
                    tables.canonical_key(key0.gamma_canon, key0.q, key0.sigma_hat);
                if (key_of_key != key0) {
                    std::cout << "HeckeTessellationSmokeValidation: canonical_key not idempotent "
                              << "(iter " << iter << ")." << std::endl;
                    return 2;
                }

                // Randomized rewritings (gamma*g, sigma') of the same simplex:
                // i(sigma) = g*i(sigma') for g in S(sigma, sigma'), so the pair
                // (gamma*g, sigma') denotes the same K^u-simplex; its key must
                // be constant. Includes g = -I (the -I quotient).
                const int steps = 1 + static_cast<int>(rng.below(5));
                for (int s = 0; s < steps; ++s) {
                    std::vector<uint32_t> orbit;
                    for (uint32_t t = 0; t < num_simplices[q]; ++t) {
                        if (tables.same_groupoid_orbit(q, sigma, t)) orbit.push_back(t);
                    }
                    const uint32_t sigma_next = orbit[rng.below(orbit.size())];
                    const auto& set = tables.transporter(q, sigma, sigma_next);
                    const hecke::Mat22 g = set[rng.below(set.size())];
                    gamma = gamma * g;
                    sigma = sigma_next;
                    const hecke::KuSimplexKey key = tables.canonical_key(gamma, q, sigma);
                    if (key != key0) {
                        std::cout << "HeckeTessellationSmokeValidation: canonical_key not "
                                  << "rewriting-invariant (iter " << iter << ")." << std::endl;
                        return 3;
                    }
                }
            }
        }

        // --- 3. word_in_T_S: 10^4 exact round trips + (P2) bound --------------
        {
            XorShift64 rng(0x503257726474532Aull); // fixed seed
            for (int iter = 0; iter < 10000; ++iter) {
                const hecke::Mat22 gamma = random_gamma(rng, 14);
                int64_t max_intermediate = 0;
                const auto word = tables.word_in_T_S_with_bound(gamma, max_intermediate);
                const hecke::Mat22 rebuilt = hecke::SL2ZTessellationTables::evaluate_word(word);
                if (!(rebuilt == gamma)) {
                    std::cout << "HeckeTessellationSmokeValidation: word_in_T_S round trip "
                              << "failed (iter " << iter << ")." << std::endl;
                    return 4;
                }
                const int64_t input_norm = gamma.max_abs() > 0 ? gamma.max_abs() : 1;
                if (max_intermediate > 2 * input_norm) {
                    std::cout << "HeckeTessellationSmokeValidation: (P2) intermediate-entry "
                              << "bound violated (iter " << iter << ")." << std::endl;
                    return 5;
                }
            }
        }

        // --- 4. Legacy bit-identity + I3 derivation ---------------------------
        {
            const uint32_t cases[3][2] = { { 0, 5 }, { 2, 5 }, { 10, 691 } };
            for (const auto& c : cases) {
                const uint32_t code = check_legacy_bit_identity(K, c[0], c[1]);
                if (code != 0) {
                    std::cout << "HeckeTessellationSmokeValidation: legacy bit-identity check "
                              << "failed for (n=" << c[0] << ", ell=" << c[1] << ") (code "
                              << code << ")." << std::endl;
                    return 6;
                }
            }
        }

        return 0;
    }
    catch (const std::exception& e) {
        std::cout << "HeckeTessellationSmokeValidation: exception: " << e.what() << std::endl;
        return 100;
    }
}
