// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// BarCocycleSolverValidation.cpp -- see the header for what this gates.

#include "BarCocycleSolverValidation.h"

#include "../../../ModArithmetic/DenseElim.h"
#include "../../../ModArithmetic/ModArithmetic.h"
#include "../../../ModArithmetic/ModCtx.h"

#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using engine::DenseBuf;
using engine::ModCtx;

// ---------------------------------------------------------------------------
// FROZEN ORACLE. This is the body of BarCocycleH1.cpp's solve_in_columns as it
// stood BEFORE matrix-engine Phase 3 Stage B (parent commit 9f2fc84, lines
// 119-163), transcribed verbatim except that the columns arrive as a plain
// row-major buffer instead of a vector<FlatMatrix>. Do not tidy it: its value
// is that it is the thing being replaced.
//
//   Gauss-Jordan on [cols | b]; `col` scans 0..Kc-1 and never the right-hand
//   column; pivot row = topmost row at or below the current pivot row with a
//   nonzero entry; leading 1; eliminate above and below; rows beyond the rank
//   with a nonzero right-hand entry mean "inconsistent"; free coordinates 0.
// ---------------------------------------------------------------------------
bool solve_in_columns_legacy(const std::vector<std::vector<uint32_t>>& cols, // cols[j][r]
                             const std::vector<uint32_t>& b, uint32_t R, uint32_t ell,
                             std::vector<uint32_t>& x_out) {
    const uint32_t Kc = static_cast<uint32_t>(cols.size());
    std::vector<std::vector<uint32_t>> A(R, std::vector<uint32_t>(Kc + 1, 0));
    for (uint32_t j = 0; j < Kc; ++j)
        for (uint32_t r = 0; r < R; ++r) A[r][j] = cols[j][r];
    for (uint32_t r = 0; r < R; ++r) A[r][Kc] = b[r];

    std::vector<int64_t> pivot_row_of_col(Kc, -1);
    uint32_t row = 0;
    for (uint32_t col = 0; col < Kc && row < R; ++col) {
        uint32_t pr = row;
        while (pr < R && A[pr][col] == 0) ++pr;
        if (pr == R) continue;
        std::swap(A[pr], A[row]);
        const uint32_t inv = ModArithmetic::mod_inv(A[row][col], ell);
        for (uint32_t c = col; c <= Kc; ++c)
            A[row][c] = ModArithmetic::mod_mul(A[row][c], inv, ell);
        for (uint32_t rr = 0; rr < R; ++rr) {
            if (rr == row || A[rr][col] == 0) continue;
            const uint32_t f = A[rr][col];
            for (uint32_t c = col; c <= Kc; ++c) {
                A[rr][c] = ModArithmetic::mod_sub(
                    A[rr][c], ModArithmetic::mod_mul(f, A[row][c], ell), ell);
            }
        }
        pivot_row_of_col[col] = row;
        ++row;
    }
    for (uint32_t rr = row; rr < R; ++rr) {
        if (A[rr][Kc] != 0) return false; // inconsistent
    }
    x_out.assign(Kc, 0);
    for (uint32_t col = 0; col < Kc; ++col) {
        if (pivot_row_of_col[col] >= 0) {
            x_out[col] = A[static_cast<size_t>(pivot_row_of_col[col])][Kc];
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// THE ROUTE UNDER TEST: exactly what BarCocycleH1.cpp::solve_in_columns now
// does (engine::elim::solve_multi_rhs, throw turned back into `false`).
// ---------------------------------------------------------------------------
bool solve_in_columns_engine(const std::vector<std::vector<uint32_t>>& cols,
                             const std::vector<uint32_t>& b, uint32_t R, uint32_t ell,
                             std::vector<uint32_t>& x_out) {
    const uint32_t Kc = static_cast<uint32_t>(cols.size());
    DenseBuf<uint32_t> B(R, Kc), Y(R, 1);
    for (uint32_t j = 0; j < Kc; ++j)
        for (uint32_t r = 0; r < R; ++r) B.at(r, j) = cols[j][r];
    for (uint32_t r = 0; r < R; ++r) Y.at(r, 0) = b[r];
    const ModCtx<uint32_t> ctx(ell);
    try {
        const std::vector<std::vector<uint32_t>> coords =
            engine::elim::solve_multi_rhs<uint32_t>(ctx, cview(B), cview(Y));
        x_out = coords.empty() ? std::vector<uint32_t>(Kc, 0) : coords[0];
        x_out.resize(Kc, 0);
        return true;
    } catch (const std::runtime_error&) {
        return false;
    }
}

// Deterministic 64-bit LCG; no std::random, no time, no global state.
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() { s = s * 6364136223846793005ull + 1442695040888963407ull; return s >> 17; }
    uint32_t below(uint32_t n) { return n == 0 ? 0 : static_cast<uint32_t>(next() % n); }
};

uint32_t g_fail = 0;
uint64_t g_cases = 0;

void compare(const std::vector<std::vector<uint32_t>>& cols, const std::vector<uint32_t>& b,
             uint32_t R, uint32_t ell, const std::string& tag) {
    ++g_cases;
    std::vector<uint32_t> x_legacy, x_engine;
    const bool ok_l = solve_in_columns_legacy(cols, b, R, ell, x_legacy);
    const bool ok_e = solve_in_columns_engine(cols, b, R, ell, x_engine);
    if (ok_l != ok_e) {
        std::cout << "  FAIL [" << tag << "] verdict differs: legacy "
                  << (ok_l ? "consistent" : "inconsistent") << ", engine "
                  << (ok_e ? "consistent" : "inconsistent") << std::endl;
        ++g_fail;
        return;
    }
    if (!ok_l) return;
    if (x_legacy.size() != x_engine.size()) {
        std::cout << "  FAIL [" << tag << "] coefficient count " << x_legacy.size()
                  << " != " << x_engine.size() << std::endl;
        ++g_fail;
        return;
    }
    for (size_t i = 0; i < x_legacy.size(); ++i) {
        if (x_legacy[i] != x_engine[i]) {
            std::cout << "  FAIL [" << tag << "] coefficient " << i << ": legacy "
                      << x_legacy[i] << " != engine " << x_engine[i] << std::endl;
            ++g_fail;
            return;
        }
    }
}

// Builds one case: `rank_cols` independent random columns, then `dup_cols`
// copies/zero columns appended (to force free columns), and a right-hand side
// that is either a random combination of the columns (consistent) or a random
// vector (usually not in the span).
void one_case(Rng& rng, uint32_t R, uint32_t rank_cols, uint32_t dup_cols, uint32_t ell,
              bool rhs_in_span, const std::string& tag) {
    std::vector<std::vector<uint32_t>> cols;
    for (uint32_t j = 0; j < rank_cols; ++j) {
        std::vector<uint32_t> c(R, 0);
        for (uint32_t r = 0; r < R; ++r) c[r] = rng.below(ell);
        cols.push_back(c);
    }
    for (uint32_t j = 0; j < dup_cols; ++j) {
        if (cols.empty() || (rng.next() & 1u) == 0u) {
            cols.push_back(std::vector<uint32_t>(R, 0));           // zero column
        } else {
            const std::vector<uint32_t>& src = cols[rng.below(static_cast<uint32_t>(cols.size()))];
            const uint32_t s = rng.below(ell);
            std::vector<uint32_t> c(R, 0);
            for (uint32_t r = 0; r < R; ++r) c[r] = ModArithmetic::mod_mul(src[r], s, ell);
            cols.push_back(c);                                      // dependent column
        }
    }
    std::vector<uint32_t> b(R, 0);
    if (rhs_in_span) {
        for (size_t j = 0; j < cols.size(); ++j) {
            const uint32_t s = rng.below(ell);
            for (uint32_t r = 0; r < R; ++r)
                b[r] = ModArithmetic::mod_add(b[r], ModArithmetic::mod_mul(cols[j][r], s, ell), ell);
        }
    } else {
        for (uint32_t r = 0; r < R; ++r) b[r] = rng.below(ell);
    }
    compare(cols, b, R, ell, tag);
}

} // namespace

uint32_t BarCocycleSolverValidation() {
    std::cout << "=== BarCocycleSolverValidation (BarCocycleH1's RREF-with-RHS solver: "
                 "frozen pre-Stage-B oracle vs engine::elim::solve_multi_rhs) ===" << std::endl;
    g_fail = 0;
    g_cases = 0;

    const uint32_t primes[] = { 5u, 13u, 691u, 134217757u, 2147483647u, 4294967291u };
    for (uint32_t ell : primes) {
        Rng rng(0x5EED0000ull ^ static_cast<uint64_t>(ell));
        std::ostringstream pre;
        pre << "ell=" << ell;

        // Degenerate shapes first: they are where a pivot rule and a
        // free-variable convention differ if they differ at all.
        {
            std::vector<std::vector<uint32_t>> none;
            compare(none, std::vector<uint32_t>{}, 0, ell, pre.str() + " R=0 Kc=0");
            compare(none, std::vector<uint32_t>(4, 0), 4, ell, pre.str() + " Kc=0 b=0");
            std::vector<uint32_t> bnz(4, 0);
            bnz[2] = 1;
            compare(none, bnz, 4, ell, pre.str() + " Kc=0 b!=0");
            std::vector<std::vector<uint32_t>> zero_cols(3, std::vector<uint32_t>(4, 0));
            compare(zero_cols, std::vector<uint32_t>(4, 0), 4, ell, pre.str() + " all-zero cols, b=0");
            compare(zero_cols, bnz, 4, ell, pre.str() + " all-zero cols, b!=0");
        }

        for (uint32_t R : { 1u, 2u, 3u, 5u, 8u, 13u }) {
            for (uint32_t rank_cols = 0; rank_cols <= R + 2; ++rank_cols) {
                for (uint32_t dup = 0; dup <= 2; ++dup) {
                    for (int in_span = 0; in_span < 2; ++in_span) {
                        std::ostringstream tag;
                        tag << pre.str() << " R=" << R << " rank=" << rank_cols
                            << " dup=" << dup << (in_span ? " in-span" : " random-rhs");
                        one_case(rng, R, rank_cols, dup, ell, in_span != 0, tag.str());
                    }
                }
            }
        }
    }

    std::cout << "BarCocycleSolverValidation: " << g_cases << " system(s) compared, "
              << g_fail << " disagreement(s)." << std::endl;
    if (g_fail != 0) {
        std::cout << "BarCocycleSolverValidation: FAILED -- the absorption of "
                     "solve_in_columns into engine::elim::solve_multi_rhs is NOT "
                     "value-preserving; revert it (00 plan, Phase 3: the absorption "
                     "is conditional)." << std::endl;
        return 1;
    }
    std::cout << "BarCocycleSolverValidation: PASSED -- verdict and coefficients agree "
                 "entrywise on every case, including rank-deficient column sets and "
                 "out-of-span right-hand sides." << std::endl;
    return 0;
}
