// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 2, WP-8 -- the shared integer solver of every chain
// fill. Moved verbatim (arithmetic unchanged) from HeckeTauTable.cpp; see
// HeckeIntegerSolve.h for the contract references and the refactor gate.

#include "HeckeIntegerSolve.h"

#include "ExactInt.h"

#include <cstddef>
#include <utility>

namespace hecke {

// ---------------------------------------------------------------------------
// Integer linear solver for  A x = c  over Z ([C, alg:tau](3): hand-rolled
// integer elimination with divisibility checks; no HNF library, no
// arbitrary precision -- invariant I5). Column-echelon reduction by
// unimodular column operations with a tracked transformation U (A U = H):
// per pivot row, Euclid across the remaining columns until at most one
// nonzero entry survives; then forced forward substitution (H y = c has a
// solution over Z iff the greedy pivot division is exact and every
// pivot-free row has zero residual -- complete, since the free components
// of y multiply zero columns of H). All arithmetic overflow-checked; the
// caller decides what a HeckeOverflow means (see HeckeIntegerSolve.h).
// ---------------------------------------------------------------------------
bool solve_integer_system(const std::vector<std::vector<int64_t>>& A_in,
                          const std::vector<int64_t>& c,
                          std::vector<int64_t>& x_out) {
    const size_t R = A_in.size();
    const size_t n = R ? A_in[0].size() : 0;
    if (n == 0) {
        for (int64_t v : c) if (v != 0) return false;
        x_out.assign(0, 0);
        return true;
    }

    std::vector<std::vector<int64_t>> A = A_in; // working copy (A_in kept for the final verification)
    std::vector<std::vector<int64_t>> U(n, std::vector<int64_t>(n, 0));
    for (size_t j = 0; j < n; ++j) U[j][j] = 1;

    const auto col_axpy = [&](size_t dst, size_t src, int64_t q) {
        // column_dst -= q * column_src, in A and U.
        for (size_t r = 0; r < R; ++r) {
            A[r][dst] = checked_sub<int64_t>(A[r][dst], checked_mul<int64_t>(q, A[r][src]));
        }
        for (size_t r = 0; r < n; ++r) {
            U[r][dst] = checked_sub<int64_t>(U[r][dst], checked_mul<int64_t>(q, U[r][src]));
        }
    };
    const auto col_swap = [&](size_t j1, size_t j2) {
        if (j1 == j2) return;
        for (size_t r = 0; r < R; ++r) std::swap(A[r][j1], A[r][j2]);
        for (size_t r = 0; r < n; ++r) std::swap(U[r][j1], U[r][j2]);
    };

    std::vector<std::pair<size_t, size_t>> pivots; // (row, column)
    size_t pc = 0;
    for (size_t row = 0; row < R && pc < n; ++row) {
        uint32_t guard = 0;
        while (true) {
            if (++guard > 100000u) {
                throw HeckeInternalError("hecke: integer elimination failed to terminate.");
            }
            size_t jmin = SIZE_MAX;
            for (size_t j = pc; j < n; ++j) {
                if (A[row][j] != 0 &&
                    (jmin == SIZE_MAX || uabs64(A[row][j]) < uabs64(A[row][jmin]))) {
                    jmin = j;
                }
            }
            if (jmin == SIZE_MAX) break; // no pivot in this row
            bool residue_left = false;
            for (size_t j = pc; j < n; ++j) {
                if (j == jmin || A[row][j] == 0) continue;
                const int64_t q = A[row][j] / A[row][jmin]; // truncated (Euclid step)
                if (q != 0) col_axpy(j, jmin, q);
                if (A[row][j] != 0) residue_left = true;
            }
            if (!residue_left) {
                col_swap(jmin, pc);
                pivots.emplace_back(row, pc);
                ++pc;
                break;
            }
        }
    }

    // Forced forward substitution with divisibility checks.
    std::vector<int64_t> y(n, 0);
    size_t pidx = 0;
    for (size_t row = 0; row < R; ++row) {
        __int128 s = c[row];
        for (size_t j = 0; j < n; ++j) {
            if (y[j] != 0 && A[row][j] != 0) {
                s = checked_sub<__int128>(
                    s, checked_mul<__int128>(static_cast<__int128>(A[row][j]),
                                             static_cast<__int128>(y[j])));
            }
        }
        if (pidx < pivots.size() && pivots[pidx].first == row) {
            const size_t pcol = pivots[pidx].second;
            const __int128 piv = A[row][pcol];
            if (s % piv != 0) return false; // Z-infeasible (divisibility check)
            const __int128 val = s / piv;
            if (val > INT64_MAX || val < INT64_MIN) {
                throw HeckeOverflow("hecke: integer solve value exceeds int64.");
            }
            y[pcol] = static_cast<int64_t>(val);
            ++pidx;
        } else {
            if (s != 0) return false; // inconsistent pivot-free row
        }
    }

    // x = U y, then verify A_in x == c exactly (belt and braces; the
    // certification re-checks the boundary identity independently anyway).
    x_out.assign(n, 0);
    for (size_t i = 0; i < n; ++i) {
        __int128 s = 0;
        for (size_t j = 0; j < n; ++j) {
            if (y[j] != 0 && U[i][j] != 0) {
                s = checked_add<__int128>(
                    s, checked_mul<__int128>(static_cast<__int128>(U[i][j]),
                                             static_cast<__int128>(y[j])));
            }
        }
        if (s > INT64_MAX || s < INT64_MIN) {
            throw HeckeOverflow("hecke: integer solution exceeds int64.");
        }
        x_out[i] = static_cast<int64_t>(s);
    }
    for (size_t r = 0; r < R; ++r) {
        __int128 s = 0;
        for (size_t j = 0; j < n; ++j) {
            if (x_out[j] != 0 && A_in[r][j] != 0) {
                s = checked_add<__int128>(
                    s, checked_mul<__int128>(static_cast<__int128>(A_in[r][j]),
                                             static_cast<__int128>(x_out[j])));
            }
        }
        if (s != c[r]) {
            throw HeckeInternalError("hecke: integer solver self-verification failed.");
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// (slot-6 WP-4) The widened tier: solve_integer_system() above, line for line,
// on __int128 working entries. Kept as a separate function (not a template
// shared with the int64 tier) so that the shipped int64 solver stays the
// verbatim code the golden grid was frozen on.
// ---------------------------------------------------------------------------
namespace {
using W = __int128;
unsigned __int128 uabs_w(W x) {
    return (x < 0) ? (~static_cast<unsigned __int128>(x) + 1u) : static_cast<unsigned __int128>(x);
}
int64_t narrow_w(W v, const char* what) {
    if (v > static_cast<W>(INT64_MAX) || v < static_cast<W>(INT64_MIN)) throw HeckeOverflow(what);
    return static_cast<int64_t>(v);
}
} // namespace

bool solve_integer_system_wide(const std::vector<std::vector<int64_t>>& A_in,
                               const std::vector<int64_t>& c,
                               std::vector<int64_t>& x_out) {
    const size_t R = A_in.size();
    const size_t n = R ? A_in[0].size() : 0;
    if (n == 0) {
        for (int64_t v : c) if (v != 0) return false;
        x_out.assign(0, 0);
        return true;
    }

    std::vector<std::vector<W>> A(R, std::vector<W>(n, 0));
    for (size_t r = 0; r < R; ++r)
        for (size_t j = 0; j < n; ++j) A[r][j] = static_cast<W>(A_in[r][j]);
    std::vector<std::vector<W>> U(n, std::vector<W>(n, 0));
    for (size_t j = 0; j < n; ++j) U[j][j] = 1;

    const auto col_axpy = [&](size_t dst, size_t src, W q) {
        for (size_t r = 0; r < R; ++r) A[r][dst] = checked_sub<W>(A[r][dst], checked_mul<W>(q, A[r][src]));
        for (size_t r = 0; r < n; ++r) U[r][dst] = checked_sub<W>(U[r][dst], checked_mul<W>(q, U[r][src]));
    };
    const auto col_swap = [&](size_t j1, size_t j2) {
        if (j1 == j2) return;
        for (size_t r = 0; r < R; ++r) std::swap(A[r][j1], A[r][j2]);
        for (size_t r = 0; r < n; ++r) std::swap(U[r][j1], U[r][j2]);
    };

    std::vector<std::pair<size_t, size_t>> pivots;
    size_t pc = 0;
    for (size_t row = 0; row < R && pc < n; ++row) {
        uint32_t guard = 0;
        while (true) {
            if (++guard > 100000u) throw HeckeInternalError("hecke: widened integer elimination failed to terminate.");
            size_t jmin = SIZE_MAX;
            for (size_t j = pc; j < n; ++j) {
                if (A[row][j] != 0 && (jmin == SIZE_MAX || uabs_w(A[row][j]) < uabs_w(A[row][jmin]))) jmin = j;
            }
            if (jmin == SIZE_MAX) break;
            bool residue_left = false;
            for (size_t j = pc; j < n; ++j) {
                if (j == jmin || A[row][j] == 0) continue;
                const W q = A[row][j] / A[row][jmin];
                if (q != 0) col_axpy(j, jmin, q);
                if (A[row][j] != 0) residue_left = true;
            }
            if (!residue_left) {
                col_swap(jmin, pc);
                pivots.emplace_back(row, pc);
                ++pc;
                break;
            }
        }
    }

    std::vector<W> y(n, 0);
    size_t pidx = 0;
    for (size_t row = 0; row < R; ++row) {
        W s = c[row];
        for (size_t j = 0; j < n; ++j) {
            if (y[j] != 0 && A[row][j] != 0) s = checked_sub<W>(s, checked_mul<W>(A[row][j], y[j]));
        }
        if (pidx < pivots.size() && pivots[pidx].first == row) {
            const W piv = A[row][pivots[pidx].second];
            if (s % piv != 0) return false;
            y[pivots[pidx].second] = s / piv;
            ++pidx;
        } else {
            if (s != 0) return false;
        }
    }

    x_out.assign(n, 0);
    for (size_t i = 0; i < n; ++i) {
        W s = 0;
        for (size_t j = 0; j < n; ++j) {
            if (y[j] != 0 && U[i][j] != 0) s = checked_add<W>(s, checked_mul<W>(U[i][j], y[j]));
        }
        x_out[i] = narrow_w(s, "hecke: widened integer solution exceeds int64.");
    }
    for (size_t r = 0; r < R; ++r) {
        W s = 0;
        for (size_t j = 0; j < n; ++j) {
            if (x_out[j] != 0 && A_in[r][j] != 0)
                s = checked_add<W>(s, checked_mul<W>(static_cast<W>(A_in[r][j]), static_cast<W>(x_out[j])));
        }
        if (s != c[r]) throw HeckeInternalError("hecke: widened integer solver self-verification failed.");
    }
    return true;
}

} // namespace hecke
