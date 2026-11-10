// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Stage S3V -- INDEPENDENT validation of Stage S3 (HeckeCosetTables).
// See HeckeCosetTablesValidation.h for the independence statement.
//
// -----------------------------------------------------------------------
// Oracle O16, transcribed VERBATIM from docs/latex/hecke_action.tex,
// \section{Worked micro-example ($p=2$) and validation plan}
// \label{sec:example} (the copy read while writing this file lives at
// docs/latex/hecke_action.tex around line 1385-1483):
//
//   g = [[1,0],[0,2]], m = 3. Right coset representatives (of Gamma''\Gamma,
//   Gamma'' = Gamma_0(2)) and left-coset representatives delta_i = s_i^{-1} g:
//     s_1 = I,             s_2 = [[1,0],[1,1]],        s_3 = S = [[0,-1],[1,0]];
//     delta_1 = [[1,0],[0,2]], delta_2 = [[1,0],[-1,2]], delta_3 = [[0,2],[-1,0]];
//   classes: cls(s_1) = (0:1), cls(s_2) = (1:1), cls(s_3) = (1:0) in P^1(F_2).
//
//   Permutations and cocycle blocks h_i(gamma), s_i*gamma = h_i(gamma)*s_{i.gamma}:
//
//     gamma | 1.gamma 2.gamma 3.gamma | h_1(gamma)        h_2(gamma)         h_3(gamma)
//     ------+-------------------------+------------------------------------------------
//     T     | 1  3  2                 | [[1,1],[0,1]]     [[-1,1],[-2,1]]    [[1,-1],[0,1]]
//     S     | 3  2  1                 | [[1,0],[0,1]]     [[1,-1],[2,-1]]    [[-1,0],[0,-1]]
//     ST    | 2  3  1                 | [[1,-1],[0,1]]    [[1,0],[0,1]]      [[-1,-1],[0,-1]]
//     TS    | 3  1  2                 | [[1,1],[0,1]]     [[1,-1],[2,-1]]    [[-1,0],[2,-1]]
//
//   Consistency checks (contract, must be asserted): multiplicativity
//   h_i(TS) = h_i(T) h_{i.T}(S), (i.T).S = i.(TS) (both stated to hold);
//   M_{Gamma''}(gamma) from eq:MH then satisfies
//   M_{Gamma''}(S)^4 = M_{Gamma''}(ST)^6 = I and
//   M_{Gamma''}(S)^2 = M_{Gamma''}(ST)^3 = M_{Gamma''}(-I).
//
// This file hard-codes exactly the above (Section 1 below), builds the
// M_{Gamma''}(gamma) = sum_i E_{i,i.gamma} tensor rho_n(h_i(gamma)) block
// matrices itself from the public HeckeCosetTables::perm/rho_cocycle API
// (per [C, prop:blockmatrices] / eq:MH) and checks the stated order
// relations exactly over F_ell (Section 2), then runs independent
// group-theoretic self-checks at p=3,5 (r=1) and p=2,r=2 re-derived from
// the contract's general representative recipe rather than read off
// HeckeCosetTables.cpp (Section 3).
// -----------------------------------------------------------------------

#include "HeckeCosetTablesValidation.h"

#include "HeckeCosetTables.h"
#include "ExactInt.h"
#include "../../ModArithmetic/FlatMatrix.h"

#include <array>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using hecke::HeckeCosetTables;
using hecke::Mat22;

// Generator dictionary index, matching the S2/S3 convention documented in
// HeckeCosetTables.h: gen in {0:T, 1:S, 2:ST, 3:TS}.
constexpr uint32_t GEN_T = 0;
constexpr uint32_t GEN_S = 1;
constexpr uint32_t GEN_ST = 2;
constexpr uint32_t GEN_TS = 3;

std::string mat22_str(const Mat22& m) {
    std::ostringstream oss;
    oss << "[[" << m.a << "," << m.b << "],[" << m.c << "," << m.d << "]]";
    return oss.str();
}

void report_mat22_mismatch(uint32_t& failures, const std::string& what,
                            const Mat22& expected, const Mat22& got) {
    std::cout << "HeckeCosetTablesValidation: " << what << " mismatch: expected "
              << mat22_str(expected) << ", got " << mat22_str(got) << "." << std::endl;
    ++failures;
}

// =========================================================================
// Section 1: the p = 2 worked example, entry-by-entry against [C, sec:example].
// =========================================================================

uint32_t check_worked_example_p2() {
    uint32_t failures = 0;

    // rho_n(.) values (rho_cocycle) depend on (ell, n_sym), but s_i, delta_i,
    // classes, perm and the cocycle Mat22 data do not -- they are pure
    // Gamma_0(2)\Gamma bookkeeping. Fix one representative (ell, n_sym) for
    // this section; Section 2 exercises the rho_n-dependent block matrices
    // over the full (n, ell) grid required by the task.
    const uint32_t ell = 5;
    const uint32_t n_sym = 0;
    const HeckeCosetTables tables(2, 1, ell, n_sym);

    if (tables.num_cosets() != 3) {
        std::cout << "HeckeCosetTablesValidation: num_cosets() expected 3, got "
                  << tables.num_cosets() << "." << std::endl;
        ++failures;
    }

    // --- s_i, s_i^{-1}, delta_i, classes -----------------------------------
    const Mat22 expected_s[3] = {
        Mat22{1, 0, 0, 1},   // s_1 = I
        Mat22{1, 0, 1, 1},   // s_2 = [[1,0],[1,1]]
        Mat22{0, -1, 1, 0},  // s_3 = S
    };
    const Mat22 expected_delta[3] = {
        Mat22{1, 0, 0, 2},    // delta_1
        Mat22{1, 0, -1, 2},   // delta_2
        Mat22{0, 2, -1, 0},   // delta_3
    };
    // classes (0:1), (1:1), (1:0) -- i.e. class_index_of(s_i) must recover i.
    for (uint32_t i = 1; i <= 3; ++i) {
        const Mat22& es = expected_s[i - 1];
        const Mat22& ed = expected_delta[i - 1];

        if (!(tables.s(i) == es)) {
            report_mat22_mismatch(failures, "s(" + std::to_string(i) + ")", es, tables.s(i));
        }
        const Mat22 expected_sinv = es.inv_unimodular();
        if (!(tables.s_inv(i) == expected_sinv)) {
            report_mat22_mismatch(failures, "s_inv(" + std::to_string(i) + ")", expected_sinv,
                                   tables.s_inv(i));
        }
        if (!(tables.delta(i) == ed)) {
            report_mat22_mismatch(failures, "delta(" + std::to_string(i) + ")", ed,
                                   tables.delta(i));
        }
        if (tables.class_index_of(es) != i) {
            std::cout << "HeckeCosetTablesValidation: class_index_of(s_" << i
                      << ") expected " << i << ", got " << tables.class_index_of(es) << "."
                      << std::endl;
            ++failures;
        }
    }
    // Direct class map checks: (0:1) -> 1, (1:1) -> 2, (1:0) -> 3.
    {
        const std::array<std::array<uint32_t, 3>, 3> class_cases = {{
            {0u, 1u, 1u}, {1u, 1u, 2u}, {1u, 0u, 3u}
        }};
        for (const auto& cc : class_cases) {
            const uint32_t got = tables.class_index(cc[0], cc[1]);
            if (got != cc[2]) {
                std::cout << "HeckeCosetTablesValidation: class_index(" << cc[0] << "," << cc[1]
                          << ") expected " << cc[2] << ", got " << got << "." << std::endl;
                ++failures;
            }
        }
    }

    // --- permutation table: T:(1,3,2) S:(3,2,1) ST:(2,3,1) TS:(3,1,2) ------
    const std::array<uint32_t, 4> gens = { GEN_T, GEN_S, GEN_ST, GEN_TS };
    const char* gen_names[4] = { "T", "S", "ST", "TS" };
    const uint32_t expected_perm[4][3] = {
        { 1, 3, 2 },  // T
        { 3, 2, 1 },  // S
        { 2, 3, 1 },  // ST
        { 3, 1, 2 },  // TS
    };
    // twelve cocycles h_i(gamma), exactly as printed in [C, sec:example].
    const Mat22 expected_cocycle[4][3] = {
        // T
        { Mat22{1, 1, 0, 1}, Mat22{-1, 1, -2, 1}, Mat22{1, -1, 0, 1} },
        // S
        { Mat22{1, 0, 0, 1}, Mat22{1, -1, 2, -1}, Mat22{-1, 0, 0, -1} },
        // ST
        { Mat22{1, -1, 0, 1}, Mat22{1, 0, 0, 1}, Mat22{-1, -1, 0, -1} },
        // TS
        { Mat22{1, 1, 0, 1}, Mat22{1, -1, 2, -1}, Mat22{-1, 0, 2, -1} },
    };

    for (uint32_t gi = 0; gi < 4; ++gi) {
        const uint32_t gen = gens[gi];
        for (uint32_t i = 1; i <= 3; ++i) {
            const uint32_t got_perm = tables.perm(gen, i);
            if (got_perm != expected_perm[gi][i - 1]) {
                std::cout << "HeckeCosetTablesValidation: perm(" << gen_names[gi] << ", " << i
                          << ") expected " << expected_perm[gi][i - 1] << ", got " << got_perm
                          << "." << std::endl;
                ++failures;
            }
            const Mat22& ec = expected_cocycle[gi][i - 1];
            const Mat22& gc = tables.cocycle(gen, i);
            if (!(gc == ec)) {
                report_mat22_mismatch(failures,
                                       std::string("cocycle(") + gen_names[gi] + ", " +
                                           std::to_string(i) + ")",
                                       ec, gc);
            }
        }
    }

    // Contract-stated multiplicativity sanity, restated directly from the
    // printed tables (redundant with HeckeCosetTables::self_check(), but
    // this is an independent re-check against the transcribed oracle
    // values above, not against the S3 implementation):
    //   h_i(TS) == h_i(T) * h_{i.T}(S),  (i.T).S == i.(TS)
    for (uint32_t i = 1; i <= 3; ++i) {
        const uint32_t iT = expected_perm[0][i - 1];         // i . T
        const uint32_t iT_then_S = expected_perm[1][iT - 1]; // (i.T) . S
        const uint32_t iTS = expected_perm[3][i - 1];        // i . (TS)
        if (iT_then_S != iTS) {
            std::cout << "HeckeCosetTablesValidation: chain rule on the transcribed "
                         "permutation table failed: (i.T).S=" << iT_then_S
                      << " != i.(TS)=" << iTS << " for i=" << i << "." << std::endl;
            ++failures;
        }
        const Mat22 hi_T = expected_cocycle[0][i - 1];
        const Mat22 hiT_S = expected_cocycle[1][iT - 1];
        const Mat22 hi_TS = expected_cocycle[3][i - 1];
        const Mat22 product = hi_T * hiT_S;
        if (!(product == hi_TS)) {
            report_mat22_mismatch(failures,
                                   "transcribed-table chain rule h_" + std::to_string(i) + "(TS)",
                                   hi_TS, product);
        }
    }

    return failures;
}

// =========================================================================
// Section 2: assembled block-matrix order relations.
// M_{Gamma''}(gamma) = sum_i E_{i,i.gamma} tensor rho_n(h_i(gamma))
// ([C, prop:blockmatrices], eq:MH), built here from the public
// perm()/rho_cocycle() accessors -- never from a value hard-coded from S3.
// =========================================================================

FlatMatrix build_block_matrix(const HeckeCosetTables& tables, uint32_t gen, uint32_t m,
                               uint32_t d_V, uint32_t ell) {
    FlatMatrix M(m * d_V, m * d_V, ell); // zero matrix
    for (uint32_t i = 1; i <= m; ++i) {
        const uint32_t j = tables.perm(gen, i); // i . gamma
        const hecke::CosetRhoT<uint32_t> block_h = tables.rho_cocycle_block(gen, i); // (slot-14 WP-10) streamed by default
        const FlatMatrix& block = block_h.get(); // rho_n(h_i(gamma))
        const uint32_t row0 = (i - 1) * d_V;
        const uint32_t col0 = (j - 1) * d_V;
        for (uint32_t r = 0; r < d_V; ++r) {
            for (uint32_t c = 0; c < d_V; ++c) {
                M.set_entry(row0 + r, col0 + c, block.get_entry(r, c));
            }
        }
    }
    return M;
}

FlatMatrix block_diag_repeat(const FlatMatrix& block, uint32_t m, uint32_t d_V, uint32_t ell) {
    FlatMatrix M(m * d_V, m * d_V, ell); // zero matrix
    for (uint32_t i = 0; i < m; ++i) {
        const uint32_t off = i * d_V;
        for (uint32_t r = 0; r < d_V; ++r) {
            for (uint32_t c = 0; c < d_V; ++c) {
                M.set_entry(off + r, off + c, block.get_entry(r, c));
            }
        }
    }
    return M;
}

// M^k via k plain multiplications (k <= 6 throughout this file; no need for
// fast exponentiation). dim = number of rows/cols of M.
FlatMatrix mat_pow(const FlatMatrix& M, uint32_t k, uint32_t dim, uint32_t ell) {
    FlatMatrix result(dim, ell); // identity matrix
    for (uint32_t e = 0; e < k; ++e) {
        result = result * M;
    }
    return result;
}

uint32_t check_order_relations(uint32_t n_sym, uint32_t ell) {
    uint32_t failures = 0;
    const HeckeCosetTables tables(2, 1, ell, n_sym);
    const uint32_t m = tables.num_cosets();
    if (m != 3) {
        std::cout << "HeckeCosetTablesValidation: (n=" << n_sym << ", ell=" << ell
                  << ") num_cosets() expected 3, got " << m << "." << std::endl;
        return failures + 1;
    }
    const uint32_t d_V = n_sym + 1;
    const uint32_t dim = m * d_V;

    const FlatMatrix M_S = build_block_matrix(tables, GEN_S, m, d_V, ell);
    const FlatMatrix M_ST = build_block_matrix(tables, GEN_ST, m, d_V, ell);
    const FlatMatrix Id(dim, ell); // identity

    // rho_n(-I), computed independently via ComputeSymmetricPowerMatrixFM
    // (existing ModArithmetic library code, not part of Stage S3), then
    // block-repeated m times: since -I is central, i.(-I) = i and
    // h_i(-I) = -I for every i, so M_{Gamma''}(-I) = I_m tensor rho_n(-I).
    const Mat22 neg_I{ -1, 0, 0, -1 };
    const auto neg_I_mod = neg_I.reduced_mod(ell);
    const FlatMatrix neg_I_fm(2, 2, ell,
                               std::vector<uint32_t>{ neg_I_mod[0], neg_I_mod[1], neg_I_mod[2],
                                                       neg_I_mod[3] });
    const FlatMatrix rho_neg_I = ComputeSymmetricPowerMatrixFM(neg_I_fm, n_sym);
    const FlatMatrix M_neg_I = block_diag_repeat(rho_neg_I, m, d_V, ell);

    const FlatMatrix S4 = mat_pow(M_S, 4, dim, ell);
    const FlatMatrix ST6 = mat_pow(M_ST, 6, dim, ell);
    const FlatMatrix S2 = mat_pow(M_S, 2, dim, ell);
    const FlatMatrix ST3 = mat_pow(M_ST, 3, dim, ell);

    if (!(S4 == Id)) {
        std::cout << "HeckeCosetTablesValidation: (n=" << n_sym << ", ell=" << ell
                  << ") M(S)^4 != I." << std::endl;
        ++failures;
    }
    if (!(ST6 == Id)) {
        std::cout << "HeckeCosetTablesValidation: (n=" << n_sym << ", ell=" << ell
                  << ") M(ST)^6 != I." << std::endl;
        ++failures;
    }
    if (!(S2 == M_neg_I)) {
        std::cout << "HeckeCosetTablesValidation: (n=" << n_sym << ", ell=" << ell
                  << ") M(S)^2 != M(-I)." << std::endl;
        ++failures;
    }
    if (!(ST3 == M_neg_I)) {
        std::cout << "HeckeCosetTablesValidation: (n=" << n_sym << ", ell=" << ell
                  << ") M(ST)^3 != M(-I)." << std::endl;
        ++failures;
    }

    return failures;
}

// =========================================================================
// Section 3: additional self-checks via the S3 API, from group theory
// derived independently in THIS file (not read from HeckeCosetTables.cpp).
// =========================================================================

// Brute-force modular inverse mod a small prime p (p in {2,3,5} only in this
// file); independent of ModArithmetic::mod_inv, used only for the
// independent re-derivation of the class map below.
int64_t small_mod_inverse(int64_t a, int64_t p) {
    const int64_t am = ((a % p) + p) % p;
    for (int64_t x = 0; x < p; ++x) {
        if ((am * x) % p == 1) return x;
    }
    return -1; // unreachable for a coprime to prime p, a != 0 mod p
}

// Independent re-derivation of the contract's representative recipe
// ([C, prop:cosets] contract box, r = 1): s_1 = I, s_{1+k} = [[1,0],[k,1]]
// (1 <= k <= p-1), s_{p+1} = S.
Mat22 independent_s(int64_t p, uint32_t i) {
    if (i == 1) return Mat22::I();
    if (i == static_cast<uint32_t>(p) + 1) return Mat22::S();
    const int64_t k = static_cast<int64_t>(i) - 1; // 1 <= k <= p-1
    return Mat22{ 1, 0, k, 1 };
}

// Independent re-derivation of the (P1) class map: class of (c:d) in
// P^1(F_p) is index p+1 if d == 0 mod p (i.e. class (1:0)), else index
// 1+k where k = c * d^{-1} mod p (class (k:1)).
uint32_t independent_class(int64_t p, int64_t c, int64_t d) {
    const int64_t cm = ((c % p) + p) % p;
    const int64_t dm = ((d % p) + p) % p;
    if (dm == 0) return static_cast<uint32_t>(p) + 1;
    const int64_t dinv = small_mod_inverse(dm, p);
    const int64_t k = (cm * dinv) % p;
    return static_cast<uint32_t>(1 + k);
}

uint32_t independent_perm(int64_t p, uint32_t i, const Mat22& gamma) {
    const Mat22 si = independent_s(p, i);
    const Mat22 prod = si * gamma;
    return independent_class(p, prod.c, prod.d);
}

Mat22 independent_cocycle(int64_t p, uint32_t i, const Mat22& gamma) {
    const uint32_t j = independent_perm(p, i, gamma);
    const Mat22 si = independent_s(p, i);
    const Mat22 sj = independent_s(p, j);
    return si * gamma * sj.inv_unimodular();
}

// Cross-checks tables' s_i, perm and cocycle against the independent
// re-derivation above, for a given Hecke prime p (r = 1). Runs ALL i and
// ALL four generators (a strict superset of "a couple", as requested).
uint32_t check_group_theory_self_p(uint64_t p_hecke, uint32_t expected_num_cosets) {
    uint32_t failures = 0;
    const uint32_t ell = 5;
    const uint32_t n_sym = 0;
    const HeckeCosetTables tables(p_hecke, 1, ell, n_sym);

    if (tables.num_cosets() != expected_num_cosets) {
        std::cout << "HeckeCosetTablesValidation: p=" << p_hecke << " num_cosets() expected "
                  << expected_num_cosets << " (= p+1), got " << tables.num_cosets() << "."
                  << std::endl;
        ++failures;
    }

    const int64_t p = static_cast<int64_t>(p_hecke);
    const std::array<uint32_t, 4> gens = { GEN_T, GEN_S, GEN_ST, GEN_TS };
    const Mat22 gen_matrices[4] = { Mat22::T(), Mat22::S(), Mat22::ST(), Mat22::TS() };
    const char* gen_names[4] = { "T", "S", "ST", "TS" };

    for (uint32_t i = 1; i <= tables.num_cosets(); ++i) {
        const Mat22 exp_s = independent_s(p, i);
        if (!(tables.s(i) == exp_s)) {
            report_mat22_mismatch(failures, "p=" + std::to_string(p_hecke) + " s(" +
                                                 std::to_string(i) + ")",
                                   exp_s, tables.s(i));
        }
        for (uint32_t gi = 0; gi < 4; ++gi) {
            const uint32_t gen = gens[gi];
            const uint32_t exp_perm = independent_perm(p, i, gen_matrices[gi]);
            const uint32_t got_perm = tables.perm(gen, i);
            if (got_perm != exp_perm) {
                std::cout << "HeckeCosetTablesValidation: p=" << p_hecke << " perm("
                          << gen_names[gi] << ", " << i << ") expected " << exp_perm << ", got "
                          << got_perm << "." << std::endl;
                ++failures;
            }
            const Mat22 exp_cocycle = independent_cocycle(p, i, gen_matrices[gi]);
            const Mat22& got_cocycle = tables.cocycle(gen, i);
            if (!(got_cocycle == exp_cocycle)) {
                report_mat22_mismatch(failures,
                                       "p=" + std::to_string(p_hecke) + " cocycle(" +
                                           gen_names[gi] + ", " + std::to_string(i) + ")",
                                       exp_cocycle, got_cocycle);
            }
            // det == 1 and lower-left == 0 mod p (h_i(gamma) in Gamma_0(p)),
            // asserted independently of self_check().
            if (got_cocycle.det() != 1) {
                std::cout << "HeckeCosetTablesValidation: p=" << p_hecke << " cocycle("
                          << gen_names[gi] << ", " << i << ") has det=" << got_cocycle.det()
                          << " != 1." << std::endl;
                ++failures;
            }
            const int64_t cll = ((got_cocycle.c % p) + p) % p;
            if (cll != 0) {
                std::cout << "HeckeCosetTablesValidation: p=" << p_hecke << " cocycle("
                          << gen_names[gi] << ", " << i << ") lower-left " << got_cocycle.c
                          << " not == 0 mod " << p_hecke << "." << std::endl;
                ++failures;
            }
        }
    }
    return failures;
}

// p = 2, r = 2 (N = 4): num_cosets() == p^2 + p == 6; every h_i(gamma) has
// det == 1 and lower-left == 0 mod 4.
uint32_t check_p2_r2() {
    uint32_t failures = 0;
    const uint32_t ell = 5;
    const uint32_t n_sym = 0;
    const HeckeCosetTables tables(2, 2, ell, n_sym);

    if (tables.N() != 4) {
        std::cout << "HeckeCosetTablesValidation: p=2,r=2 N() expected 4, got " << tables.N()
                  << "." << std::endl;
        ++failures;
    }
    if (tables.num_cosets() != 6) {
        std::cout << "HeckeCosetTablesValidation: p=2,r=2 num_cosets() expected 6 (p^2+p), got "
                  << tables.num_cosets() << "." << std::endl;
        ++failures;
    }

    const int64_t N = 4;
    const std::array<uint32_t, 4> gens = { GEN_T, GEN_S, GEN_ST, GEN_TS };
    const char* gen_names[4] = { "T", "S", "ST", "TS" };

    for (uint32_t gi = 0; gi < 4; ++gi) {
        const uint32_t gen = gens[gi];
        for (uint32_t i = 1; i <= tables.num_cosets(); ++i) {
            const Mat22& h = tables.cocycle(gen, i);
            if (h.det() != 1) {
                std::cout << "HeckeCosetTablesValidation: p=2,r=2 cocycle(" << gen_names[gi]
                          << ", " << i << ") has det=" << h.det() << " != 1." << std::endl;
                ++failures;
            }
            const int64_t cll = ((h.c % N) + N) % N;
            if (cll != 0) {
                std::cout << "HeckeCosetTablesValidation: p=2,r=2 cocycle(" << gen_names[gi]
                          << ", " << i << ") lower-left " << h.c << " not == 0 mod 4."
                          << std::endl;
                ++failures;
            }
        }
    }

    return failures;
}

} // namespace

uint32_t HeckeCosetTablesValidation() {
    try {
        uint32_t failures = 0;

        // Section 1: p = 2 worked example, entry-by-entry.
        failures += check_worked_example_p2();

        // Section 2: assembled block-matrix order relations,
        // n in {0,2,10}, ell in {5,691}.
        const std::array<uint32_t, 3> ns = { 0, 2, 10 };
        const std::array<uint32_t, 2> ells = { 5, 691 };
        for (uint32_t n_sym : ns) {
            for (uint32_t ell : ells) {
                failures += check_order_relations(n_sym, ell);
            }
        }

        // Section 3: additional self-checks via the S3 API, from group
        // theory re-derived independently in this file.
        failures += check_group_theory_self_p(3, 4); // p+1 = 4
        failures += check_group_theory_self_p(5, 6); // p+1 = 6
        failures += check_p2_r2();

        if (failures != 0) {
            std::cout << "HeckeCosetTablesValidation: " << failures
                      << " discrepancy(ies) found (see above)." << std::endl;
            return failures;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cout << "HeckeCosetTablesValidation: exception: " << e.what() << std::endl;
        return 100;
    }
}
