// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#include "pch.h"
#include "SymmetricPowerValidation.h"
#include "FlatMatrix.h"
#include "ModArithmetic.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

// ===========================================================================
// matrix-engine Phase 1, items 7.0 / 7.5. See SymmetricPowerValidation.h.
// ===========================================================================

namespace {

// --- tiny deterministic PRNG (fixed seed; no <random> engine dependence) ---
struct Splitmix {
    uint64_t s;
    explicit Splitmix(uint64_t seed) : s(seed) {}
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    uint32_t below(uint32_t m) { return static_cast<uint32_t>(next() % m); }
};

struct Mat2 {
    uint32_t a, b, c, d;
    std::string label;
};

FlatMatrix make2(const Mat2& m, uint32_t ell) {
    return FlatMatrix(2, 2, ell, { m.a % ell, m.b % ell, m.c % ell, m.d % ell });
}

bool entries_equal(const FlatMatrix& X, const FlatMatrix& Y) {
    if (X.get_rows() != Y.get_rows() || X.get_cols() != Y.get_cols()) return false;
    for (uint32_t i = 0; i < X.get_rows(); ++i)
        for (uint32_t j = 0; j < X.get_cols(); ++j)
            if (X.get_entry(i, j) != Y.get_entry(i, j)) return false;
    return true;
}

// The 2x2 generators the pipeline actually uses, plus degenerate/singular ones
// that only occur at the bad prime (det == 0 mod ell) -- exactly the inputs the
// branch table of ComputeSymmetricPowerMatrixFM must cover.
std::vector<Mat2> structured_generators(uint32_t ell) {
    const uint32_t m1 = ell - 1; // -1 mod ell
    std::vector<Mat2> g = {
        { 1, 1, 0, 1, "T = [[1,1],[0,1]]" },                 // C5 (Pascal)
        { 0, m1, 1, 0, "S = [[0,-1],[1,0]]" },               // C2
        { m1, 0, 0, m1, "-I" },                              // C1
        { 1, 0, 0, 1, "I" },                                 // C1
        { 0, m1, 1, 1, "ST = [[0,-1],[1,1]]" },              // R2 (a == 0)
        { 1, m1, 1, 0, "TS = [[1,-1],[1,0]]" },              // R1
        { 1, 0, 0, 2, "[[1,0],[0,2]] (coset rep, det 2)" },  // C1, det != 1
        { 1, 0, 0, 3, "[[1,0],[0,3]] (coset rep, det 3)" },  // C1
        { 1, 1, 0, 2, "[[1,1],[0,2]] (coset rep, det 2)" },  // R1, det != 1
        { 2, 0, 0, 1, "[[2,0],[0,1]]" },                     // C1
        { 0, 1, 1, 0, "[[0,1],[1,0]] (det -1)" },            // C2
        { 0, 0, 1, 1, "[[0,0],[1,1]] (singular, row 1 = 0)" }, // C3
        { 1, 1, 0, 0, "[[1,1],[0,0]] (singular, row 2 = 0)" }, // C4
        { 0, 0, 0, 0, "zero matrix" },                       // C1
        { 0, 1, 0, 1, "[[0,1],[0,1]] (singular, a == 0)" },  // R2
        { 2, 4, 3, 6, "[[2,4],[3,6]] (singular, det 0)" },   // R1, det == 0
        { 0, 2, 3, 0, "[[0,2],[3,0]] (anti-diagonal, det -6)" }, // C2
    };
    // The [[1,0],[k,1]] family (C6, lower unitriangular).
    for (uint32_t k : { 1u, 2u, 5u, 17u }) {
        g.push_back({ 1, 0, k % ell, 1, "[[1,0],[" + std::to_string(k) + ",1]]" });
    }
    // 20 fixed-seed random 2x2s. The seed is fixed, so the grid is stable
    // across runs; a == 0 mod ell (the [R2] branch) shows up naturally at the
    // small primes and is forced for two of them.
    Splitmix rng(0xC0FFEE123456789ULL ^ ell);
    for (int i = 0; i < 20; ++i) {
        uint32_t a = rng.below(ell), b = rng.below(ell);
        uint32_t c = rng.below(ell), d = rng.below(ell);
        if (i == 3 || i == 11) a = 0;           // force the [R2] branch
        if (i == 7) { b = 0; c = 0; }           // force [C1] with generic diagonal
        g.push_back({ a, b, c, d, "random#" + std::to_string(i) });
    }
    return g;
}

// ---------------------------------------------------------------------------
// (a)/(b): bit-comparison against the pre-Phase-1 reference implementation.
// ---------------------------------------------------------------------------
uint32_t section_reference(const std::vector<uint32_t>& primes,
                           const std::vector<uint32_t>& n_grid,
                           const char* title,
                           bool structured_subset_only) {
    uint32_t failures = 0;
    uint64_t cases = 0;
    for (uint32_t ell : primes) {
        std::vector<Mat2> gens = structured_generators(ell);
        if (structured_subset_only) {
            // (b) spot grid: T, S and one generic matrix only.
            gens = { gens[0], gens[1], { 3 % ell, 5 % ell, 7 % ell, 11 % ell, "generic [[3,5],[7,11]]" } };
        }
        for (uint32_t n : n_grid) {
            for (const Mat2& g : gens) {
                const FlatMatrix A = make2(g, ell);
                const FlatMatrix got = ComputeSymmetricPowerMatrixFM(A, n);
                const FlatMatrix want = ComputeSymmetricPowerMatrixFM_reference(A, n);
                ++cases;
                if (!entries_equal(got, want)) {
                    ++failures;
                    if (failures <= 12) {
                        std::cout << "  [FAIL] " << title << ": ell=" << ell
                                  << " n_sym=" << n << " A=" << g.label << std::endl;
                    }
                }
            }
        }
    }
    std::cout << "  " << (failures ? "[FAIL] " : "[PASS] ") << title << ": "
              << cases << " matrices compared against the reference, "
              << failures << " mismatching" << std::endl;
    return failures;
}

// ---------------------------------------------------------------------------
// (c): structural identities. No O(n^5) reference is involved.
// ---------------------------------------------------------------------------

// Pascal's triangle rows 0..n mod ell, built by addition only -- independent of
// mod_nCr (which the closed forms call), so this is a genuine oracle.
std::vector<std::vector<uint32_t>> pascal_rows(uint32_t n, uint32_t ell) {
    std::vector<std::vector<uint32_t>> P(n + 1);
    P[0] = { 1u % ell };
    for (uint32_t i = 1; i <= n; ++i) {
        P[i].assign(i + 1, 0);
        P[i][0] = 1u % ell;
        P[i][i] = 1u % ell;
        for (uint32_t j = 1; j < i; ++j)
            P[i][j] = ModArithmetic::mod_add(P[i - 1][j - 1], P[i - 1][j], ell);
    }
    return P;
}

// 7.5 closed forms, checked entry by entry against independent oracles.
uint32_t check_closed_forms(uint32_t n, uint32_t ell, uint32_t k_low) {
    uint32_t bad = 0;
    const uint32_t d_V = n + 1;
    const uint32_t m1 = ell - 1;
    const std::vector<std::vector<uint32_t>> P = pascal_rows(n, ell);

    // rho_n(T): upper unitriangular Pascal, entry(t,j) = C(j,t).
    {
        const FlatMatrix R = ComputeSymmetricPowerMatrixFM(FlatMatrix(2, 2, ell, { 1u % ell, 1u % ell, 0u, 1u % ell }), n);
        for (uint32_t j = 0; j < d_V && !bad; ++j)
            for (uint32_t t = 0; t < d_V; ++t) {
                const uint32_t want = (t <= j) ? P[j][t] : 0u;
                if (R.get_entry(t, j) != want) { ++bad; std::cout << "  [FAIL] rho_n(T) closed form n=" << n << " ell=" << ell << " (" << t << "," << j << ")" << std::endl; break; }
            }
    }
    // rho_n(S): signed anti-diagonal, entry(n-j,j) = (-1)^j.
    {
        const FlatMatrix R = ComputeSymmetricPowerMatrixFM(FlatMatrix(2, 2, ell, { 0u, m1, 1u % ell, 0u }), n);
        for (uint32_t j = 0; j < d_V && !bad; ++j)
            for (uint32_t i = 0; i < d_V; ++i) {
                const uint32_t want = (i == n - j) ? ((j % 2 == 0) ? 1u % ell : m1) : 0u;
                if (R.get_entry(i, j) != want) { ++bad; std::cout << "  [FAIL] rho_n(S) closed form n=" << n << " ell=" << ell << std::endl; break; }
            }
    }
    // rho_n(-I) = (-1)^n * I.
    {
        const FlatMatrix R = ComputeSymmetricPowerMatrixFM(FlatMatrix(2, 2, ell, { m1, 0u, 0u, m1 }), n);
        const uint32_t s = (n % 2 == 0) ? 1u % ell : m1;
        for (uint32_t j = 0; j < d_V && !bad; ++j)
            for (uint32_t i = 0; i < d_V; ++i)
                if (R.get_entry(i, j) != (i == j ? s : 0u)) { ++bad; std::cout << "  [FAIL] rho_n(-I) closed form n=" << n << " ell=" << ell << std::endl; break; }
    }
    // rho_n([[1,0],[k,1]]): lower unitriangular, entry(i,j) = C(n-j,i-j) k^(i-j).
    {
        const uint32_t k = k_low % ell;
        const FlatMatrix R = ComputeSymmetricPowerMatrixFM(FlatMatrix(2, 2, ell, { 1u % ell, 0u, k, 1u % ell }), n);
        std::vector<uint32_t> pk(d_V);
        pk[0] = 1u % ell;
        for (uint32_t i = 1; i < d_V; ++i) pk[i] = ModArithmetic::mod_mul(pk[i - 1], k, ell);
        for (uint32_t j = 0; j < d_V && !bad; ++j)
            for (uint32_t i = 0; i < d_V; ++i) {
                const uint32_t want = (i >= j) ? ModArithmetic::mod_mul(P[n - j][i - j], pk[i - j], ell) : 0u;
                if (R.get_entry(i, j) != want) { ++bad; std::cout << "  [FAIL] rho_n([[1,0],[k,1]]) closed form n=" << n << " ell=" << ell << std::endl; break; }
            }
    }
    // General diagonal [[a,0],[0,d]]: entry(j,j) = a^(n-j) d^j (mod_pow oracle).
    {
        const uint32_t a = 5u % ell, d = 3u % ell;
        const FlatMatrix R = ComputeSymmetricPowerMatrixFM(FlatMatrix(2, 2, ell, { a, 0u, 0u, d }), n);
        for (uint32_t j = 0; j < d_V && !bad; ++j)
            for (uint32_t i = 0; i < d_V; ++i) {
                const uint32_t want = (i == j)
                    ? ModArithmetic::mod_mul(ModArithmetic::mod_pow(a, n - j, ell),
                                             ModArithmetic::mod_pow(d, j, ell), ell)
                    : 0u;
                if (R.get_entry(i, j) != want) { ++bad; std::cout << "  [FAIL] rho_n(diag) closed form n=" << n << " ell=" << ell << std::endl; break; }
            }
    }
    return bad;
}

uint32_t mat2_mul_mod(const Mat2& A, const Mat2& B, uint32_t ell, Mat2& out) {
    auto mul = [ell](uint32_t x, uint32_t y) { return ModArithmetic::mod_mul(x % ell, y % ell, ell); };
    auto add = [ell](uint32_t x, uint32_t y) { return ModArithmetic::mod_add(x, y, ell); };
    out.a = add(mul(A.a, B.a), mul(A.b, B.c));
    out.b = add(mul(A.a, B.b), mul(A.b, B.d));
    out.c = add(mul(A.c, B.a), mul(A.d, B.c));
    out.d = add(mul(A.c, B.b), mul(A.d, B.d));
    out.label = A.label + "*" + B.label;
    return 0;
}

// rho_n(A) rho_n(B) == rho_n(AB) on fixed-seed pairs. This is the identity that
// ties the closed-form branches to the recurrence branches: a product of two
// structured matrices is generally unstructured.
uint32_t check_multiplicativity(uint32_t n, uint32_t ell, uint32_t num_pairs, uint64_t seed) {
    uint32_t bad = 0;
    Splitmix rng(seed ^ (uint64_t(ell) << 8) ^ n);
    std::vector<Mat2> pool = structured_generators(ell);
    for (uint32_t t = 0; t < num_pairs; ++t) {
        const Mat2& A = pool[rng.below(static_cast<uint32_t>(pool.size()))];
        const Mat2& B = pool[rng.below(static_cast<uint32_t>(pool.size()))];
        Mat2 AB{};
        mat2_mul_mod(A, B, ell, AB);
        const FlatMatrix lhs = ComputeSymmetricPowerMatrixFM(make2(A, ell), n)
                             * ComputeSymmetricPowerMatrixFM(make2(B, ell), n);
        const FlatMatrix rhs = ComputeSymmetricPowerMatrixFM(make2(AB, ell), n);
        if (!entries_equal(lhs, rhs)) {
            ++bad;
            if (bad <= 8)
                std::cout << "  [FAIL] rho_n(A)rho_n(B) != rho_n(AB): n=" << n
                          << " ell=" << ell << " A=" << A.label << " B=" << B.label << std::endl;
        }
    }
    return bad;
}

// rho_n(A)^{-1} == rho_n(A^{-1}) for det A invertible mod ell.
uint32_t check_inverse(uint32_t n, uint32_t ell, uint32_t num_pairs, uint64_t seed,
                       bool use_explicit_inverse) {
    uint32_t bad = 0;
    Splitmix rng(seed ^ (uint64_t(ell) << 16) ^ n);
    uint32_t done = 0, tries = 0;
    while (done < num_pairs && tries < num_pairs * 40) {
        ++tries;
        Mat2 A{ rng.below(ell), rng.below(ell), rng.below(ell), rng.below(ell), "rand" };
        const uint32_t det = ModArithmetic::mod_sub(ModArithmetic::mod_mul(A.a, A.d, ell),
                                                   ModArithmetic::mod_mul(A.b, A.c, ell), ell);
        if (det == 0) continue;
        ++done;
        const uint32_t di = ModArithmetic::mod_inv(det, ell);
        Mat2 Ainv{ ModArithmetic::mod_mul(A.d, di, ell),
                   ModArithmetic::mod_mul(ModArithmetic::mod_minus(A.b, ell), di, ell),
                   ModArithmetic::mod_mul(ModArithmetic::mod_minus(A.c, ell), di, ell),
                   ModArithmetic::mod_mul(A.a, di, ell), "rand^-1" };
        const FlatMatrix R = ComputeSymmetricPowerMatrixFM(make2(A, ell), n);
        const FlatMatrix Rinv = ComputeSymmetricPowerMatrixFM(make2(Ainv, ell), n);
        if (use_explicit_inverse) {
            // Genuine rho_n(A)^{-1} via Gaussian elimination (small n only).
            if (!entries_equal(R.inverse(), Rinv)) {
                ++bad;
                if (bad <= 8) std::cout << "  [FAIL] rho_n(A)^-1 != rho_n(A^-1): n=" << n << " ell=" << ell << std::endl;
            }
        } else {
            // Equivalent, and O(n^3) instead of an RREF with a d_V x 2 d_V tableau.
            const FlatMatrix prod = R * Rinv;
            const FlatMatrix I = FlatMatrix::CreateScalar(n + 1, 1u % ell, ell);
            if (!entries_equal(prod, I)) {
                ++bad;
                if (bad <= 8) std::cout << "  [FAIL] rho_n(A) rho_n(A^-1) != I: n=" << n << " ell=" << ell << std::endl;
            }
        }
    }
    return bad;
}

uint32_t section_structural(const std::vector<uint32_t>& closed_form_n,
                            const std::vector<uint32_t>& primes,
                            const std::vector<uint32_t>& identity_n,
                            uint32_t pairs_per_n) {
    uint32_t failures = 0;
    uint64_t cf_cases = 0;
    for (uint32_t ell : primes)
        for (uint32_t n : closed_form_n) {
            failures += check_closed_forms(n, ell, 7u);
            cf_cases += 5;
        }
    std::cout << "  " << (failures ? "[FAIL] " : "[PASS] ")
              << "(c) closed forms vs independent Pascal/power oracles: "
              << cf_cases << " matrices, " << failures << " failing" << std::endl;

    uint32_t id_fail = 0;
    uint64_t id_cases = 0;
    for (uint32_t ell : primes)
        for (uint32_t n : identity_n) {
            id_fail += check_multiplicativity(n, ell, pairs_per_n, 0x5EEDULL);
            id_fail += check_inverse(n, ell, pairs_per_n / 4 + 1, 0xBEEFULL, n <= 24);
            id_cases += pairs_per_n + pairs_per_n / 4 + 1;
        }
    std::cout << "  " << (id_fail ? "[FAIL] " : "[PASS] ")
              << "(c) rho_n(A)rho_n(B)=rho_n(AB) and rho_n(A)^-1=rho_n(A^-1): "
              << id_cases << " pairs, " << id_fail << " failing" << std::endl;
    return failures + id_fail;
}

} // namespace

uint32_t SymmetricPowerValidation() {
    std::cout << "\n=== SymmetricPowerValidation (matrix-engine Phase 1, items 7.0/7.5) ===" << std::endl;
    uint32_t failures = 0;

    // (a) exhaustive n_sym <= 48 against the pre-Phase-1 reference.
    {
        std::vector<uint32_t> n_grid;
        for (uint32_t n = 0; n <= 48; ++n) n_grid.push_back(n);
        failures += section_reference({ 2u, 3u, 5u, 691u, 134217757u }, n_grid,
                                      "(a) exhaustive n_sym <= 48", false);
    }
    // (b) spot n_sym in {64, 96, 128, 160}.
    failures += section_reference({ 5u, 134217757u }, { 64u, 96u, 128u, 160u },
                                  "(b) spot n_sym in {64,96,128,160}", true);

    // (c) structural, large n_sym, no reference involved.
    {
        std::vector<uint32_t> cf_n;
        for (uint32_t n = 0; n <= 64; ++n) cf_n.push_back(n);
        for (uint32_t n : { 100u, 128u, 160u, 200u, 256u, 384u, 512u, 768u, 1000u }) cf_n.push_back(n);
        failures += section_structural(cf_n, { 2u, 3u, 5u, 691u, 134217757u },
                                       { 1u, 2u, 3u, 8u, 16u, 32u, 64u, 128u }, 100u);
    }

    std::cout << (failures ? "=== SymmetricPowerValidation: FAILED, "
                           : "=== SymmetricPowerValidation: PASSED, ")
              << failures << " failing cases ===" << std::endl;
    return failures;
}

uint32_t SymmetricPowerValidationSlow() {
    std::cout << "\n=== SymmetricPowerValidationSlow (matrix-engine Phase 1, items 7.0/7.5) ===" << std::endl;
    uint32_t failures = 0;

    // Reference comparison pushed up to n_sym = 224 on the full generator set.
    failures += section_reference({ 3u, 134217757u }, { 176u, 192u, 224u },
                                  "(b+) spot n_sym in {176,192,224}, full generator set", false);

    // Structural identities at the sizes the standard tier cannot afford.
    failures += section_structural({ 1000u }, { 5u, 134217757u },
                                   { 256u, 512u }, 12u);

    std::cout << (failures ? "=== SymmetricPowerValidationSlow: FAILED, "
                           : "=== SymmetricPowerValidationSlow: PASSED, ")
              << failures << " failing cases ===" << std::endl;
    return failures;
}
