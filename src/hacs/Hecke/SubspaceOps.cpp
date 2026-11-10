// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions phase E0a -- the exact F_ell subspace toolkit.
// See SubspaceOps.h for the placement decision and the value-preservation
// statement (every body below is the verbatim pre-move text).

#include "SubspaceOps.h"

#include "../../ModArithmetic/EngineCheck.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace hecke {

uint32_t RankOf(const Matrix& M) {
    Matrix copy = M; // rank() lazily caches the RREF; work on a copy
    return copy.rank();
}

Matrix SolveColumnsExact(const Matrix& B, const Matrix& C, const char* who) {
    if (B.rows != C.rows || B.p != C.p) {
        throw std::invalid_argument(std::string(who) + ": SolveColumnsExact shape/modulus mismatch");
    }
    const uint32_t d = B.rows, k = B.cols, m = C.cols;
    Matrix aug(d, k + m, B.p);
    for (uint32_t i = 0; i < d; ++i) {
        for (uint32_t j = 0; j < k; ++j) aug.data[i][j] = B.data[i][j];
        for (uint32_t j = 0; j < m; ++j) aug.data[i][k + j] = C.data[i][j];
    }
    const std::vector<uint32_t>& piv = aug.get_pivot_columns();
    if (piv.size() != k) {
        throw std::runtime_error(std::string(who) +
                                 ": exact solve failed (rank defect or inconsistent system)");
    }
    for (uint32_t i = 0; i < k; ++i) {
        if (piv[i] != i) {
            throw std::runtime_error(std::string(who) + ": basis matrix not of full column rank");
        }
    }
    const std::vector<std::vector<uint32_t>>& rref = aug.get_row_echelon_form();
    Matrix X(k, m, B.p);
    for (uint32_t i = 0; i < k; ++i) {
        for (uint32_t j = 0; j < m; ++j) X.data[i][j] = rref[i][k + j];
    }
    return X;
}

Matrix HStack(const std::vector<Matrix>& parts, uint32_t rows, uint32_t ell) {
    uint32_t total = 0;
    for (const Matrix& part : parts) total += part.cols;
    Matrix out(rows, total, ell);
    uint32_t off = 0;
    for (const Matrix& part : parts) {
        if (part.rows != rows) throw std::invalid_argument("HStack: row mismatch");
        for (uint32_t i = 0; i < rows; ++i) {
            for (uint32_t j = 0; j < part.cols; ++j) out.data[i][off + j] = part.data[i][j];
        }
        off += part.cols;
    }
    return out;
}

Matrix KernelAsMatrix(Matrix M) {
    std::vector<Matrix> ker = M.kernel();
    Matrix out(M.cols, static_cast<uint32_t>(ker.size()), M.p);
    for (size_t j = 0; j < ker.size(); ++j) {
        for (uint32_t i = 0; i < M.cols; ++i) out.data[i][j] = ker[j].data[i][0];
    }
    return out;
}

uint32_t NilpotencyExponent(const Matrix& N, uint32_t cap, std::vector<Matrix>* powers_out) {
    Matrix P(N.rows, N.p); // identity
    if (powers_out) powers_out->push_back(P);
    for (uint32_t r = 0; r <= cap; ++r) {
        if (P.is_zero()) return r;
        P = P * N;
        if (powers_out) powers_out->push_back(P);
    }
    if (P.is_zero()) return cap + 1;
    throw std::runtime_error("NilpotencyExponent: matrix not nilpotent within the expected bound");
}

Matrix PrimaryRCFBasis(const Matrix& Aj_in, const FpPoly& pi,
                       std::vector<uint32_t>& sizes_out) {
    const uint32_t ell = Aj_in.p;
    const uint32_t n = Aj_in.rows;
    const uint32_t f = static_cast<uint32_t>(pi.degree());
    sizes_out.clear();

    SplitMix64 rng(kRcfSeed);
    std::vector<Matrix> chain_columns_in_block_coords;

    Matrix Bcur = Aj_in;        // action on the current invariant complement
    Matrix E(n, ell);           // embedding: current coords -> block coords (starts = I)
    uint32_t cur_dim = n;

    while (cur_dim > 0) {
        const Matrix Npi = EvalPolyAtMatrix(pi, Bcur);
        std::vector<Matrix> npows;
        const uint32_t e_cur = NilpotencyExponent(Npi, cur_dim, &npows);
        if (e_cur == 0) throw std::runtime_error("PrimaryRCFBasis: zero-dimensional height");
        // Maximal-height vector: some standard basis vector v with
        // pi(Bcur)^{e-1} v != 0, i.e. a nonzero column of npows[e-1].
        const Matrix& Ne1 = npows[e_cur - 1];
        uint32_t vcol = cur_dim;
        for (uint32_t j = 0; j < cur_dim && vcol == cur_dim; ++j) {
            for (uint32_t i = 0; i < cur_dim; ++i) {
                if (Ne1.data[i][j] != 0) { vcol = j; break; }
            }
        }
        if (vcol == cur_dim) throw std::runtime_error("PrimaryRCFBasis: no maximal-height vector");
        const uint32_t k = e_cur * f; // dim of the cyclic subspace Z(v)

        // Z = [v, Bcur v, ..., Bcur^{k-1} v] in current coords.
        Matrix Z(cur_dim, k, ell);
        Matrix w(cur_dim, 1, ell);
        w.data[vcol][0] = 1;
        for (uint32_t j = 0; j < k; ++j) {
            for (uint32_t i = 0; i < cur_dim; ++i) Z.data[i][j] = w.data[i][0];
            if (j + 1 < k) w = Bcur * w;
        }
        if (RankOf(Z) != k) {
            throw std::runtime_error("PrimaryRCFBasis: cyclic columns unexpectedly dependent");
        }

        chain_columns_in_block_coords.push_back(E * Z);
        sizes_out.push_back(e_cur);

        if (k == cur_dim) break; // the current space is itself cyclic

        // Invariant complement: find phi with rank(Phi * Z) = k.
        Matrix W(0, 0, ell);
        bool found = false;
        const uint32_t kMaxTries = 4096;
        for (uint32_t attempt = 0; attempt < kMaxTries && !found; ++attempt) {
            Matrix phi(1, cur_dim, ell);
            if (attempt < cur_dim) {
                phi.data[0][attempt] = 1; // dual standard basis first (deterministic)
            } else {
                for (uint32_t j = 0; j < cur_dim; ++j) phi.data[0][j] = rng.next_mod(ell);
            }
            // Phi rows: phi * Bcur^i, i = 0..k-1.
            Matrix Phi(k, cur_dim, ell);
            Matrix row = phi;
            for (uint32_t i = 0; i < k; ++i) {
                for (uint32_t j = 0; j < cur_dim; ++j) Phi.data[i][j] = row.data[0][j];
                if (i + 1 < k) row = row * Bcur;
            }
            if (RankOf(Phi * Z) != k) continue;
            W = KernelAsMatrix(Phi);
            if (W.cols != cur_dim - k) {
                throw std::runtime_error("PrimaryRCFBasis: complement of unexpected dimension");
            }
            found = true;
        }
        if (!found) {
            throw std::runtime_error("PrimaryRCFBasis: no splitting functional found "
                                     "(search budget exhausted)");
        }

        // Restrict to the complement; exact solvability certifies invariance.
        Matrix Bnew = SolveColumnsExact(W, Bcur * W, "PrimaryRCFBasis(restriction)");
        E = E * W;
        Bcur = Bnew;
        cur_dim -= k;
    }

    Matrix R = HStack(chain_columns_in_block_coords, n, ell);
    if (R.rows != n || R.cols != n) throw std::runtime_error("PrimaryRCFBasis: basis not square");
    for (size_t t = 1; t < sizes_out.size(); ++t) {
        if (sizes_out[t] > sizes_out[t - 1]) {
            throw std::runtime_error("PrimaryRCFBasis: chain heights not non-increasing");
        }
    }
    return R;
}

// ---------------------------------------------------------------------------
// E0a additions.
// ---------------------------------------------------------------------------

Matrix VStack(const Matrix& A, const Matrix& B) {
    if (A.cols != B.cols || A.p != B.p) {
        throw std::invalid_argument("VStack: column/modulus mismatch");
    }
    Matrix out(A.rows + B.rows, A.cols, A.p);
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint32_t j = 0; j < A.cols; ++j) out.data[i][j] = A.data[i][j];
    for (uint32_t i = 0; i < B.rows; ++i)
        for (uint32_t j = 0; j < B.cols; ++j) out.data[A.rows + i][j] = B.data[i][j];
    return out;
}

bool IsNilpotent(const Matrix& N, uint32_t cap) {
    if (N.rows != N.cols) throw std::invalid_argument("IsNilpotent: non-square matrix");
    if (N.rows == 0) return true;
    Matrix P = N;
    uint32_t e = 1;
    const uint32_t target = (cap == 0) ? 1u : cap;
    while (e < target) {
        if (P.is_zero()) return true;
        P = P * P;
        e = (e > (0xFFFFFFFFu / 2u)) ? 0xFFFFFFFFu : (e * 2u);
    }
    return P.is_zero();
}

Matrix FrobeniusPower(const Matrix& A, uint32_t ell, uint32_t N) {
    if (A.rows != A.cols) throw std::invalid_argument("FrobeniusPower: non-square matrix");
    Matrix cur = A;
    for (uint32_t step = 0; step < N; ++step) {
        // cur <- cur^ell by square-and-multiply on the bits of ell.
        Matrix acc(cur.rows, cur.p); // identity
        Matrix base = cur;
        uint32_t e = ell;
        while (e > 0) {
            if (e & 1u) acc = acc * base;
            e >>= 1;
            if (e > 0) base = base * base;
        }
        cur = acc;
    }
    return cur;
}

// ===========================================================================
// E1s addition (WP-3): the algebra closure.  NOT a move -- new code, so it
// uses ENGINE_CHECK (EngineCheck.h) rather than the pre-move exception idiom of
// the verbatim bodies above.
//
// E1a (WP-4) promoted this section's file-local `RrefSpan` and `VecOf` to the
// header as `RrefRowSpan` / `VecOfMatrix` so HeckeLocalAlgebra shares the one
// canonical span object.  The bodies below are that verbatim text; only the
// names and the `contains()` / `as_matrix()` accessors are new, so
// algebra_closure's output is unchanged by construction.
// ===========================================================================

bool RrefRowSpan::insert(std::vector<uint32_t> v) {
    for (size_t i = 0; i < rows_.size(); ++i) {
        const uint32_t c = v[pivots_[i]];
        if (c == 0) continue;
        for (uint32_t j = pivots_[i]; j < width_; ++j) {
            if (rows_[i][j] == 0) continue;
            v[j] = ModArithmetic::mod_sub(v[j], ModArithmetic::mod_mul(c, rows_[i][j], ell_), ell_);
        }
    }
    uint32_t piv = width_;
    for (uint32_t j = 0; j < width_; ++j) {
        if (v[j] != 0) { piv = j; break; }
    }
    if (piv == width_) return false;
    const uint32_t inv = ModArithmetic::mod_inv(v[piv], ell_);
    if (inv != 1) {
        for (uint32_t j = piv; j < width_; ++j) {
            if (v[j] != 0) v[j] = ModArithmetic::mod_mul(v[j], inv, ell_);
        }
    }
    for (size_t i = 0; i < rows_.size(); ++i) {
        const uint32_t c = rows_[i][piv];
        if (c == 0) continue;
        for (uint32_t j = piv; j < width_; ++j) {
            if (v[j] == 0) continue;
            rows_[i][j] = ModArithmetic::mod_sub(rows_[i][j],
                                                 ModArithmetic::mod_mul(c, v[j], ell_), ell_);
        }
    }
    size_t at = rows_.size();
    for (size_t i = 0; i < pivots_.size(); ++i) {
        if (pivots_[i] > piv) { at = i; break; }
    }
    rows_.insert(rows_.begin() + static_cast<std::ptrdiff_t>(at), std::move(v));
    pivots_.insert(pivots_.begin() + static_cast<std::ptrdiff_t>(at), piv);
    return true;
}

bool RrefRowSpan::contains(const std::vector<uint32_t>& v_in) const {
    ENGINE_CHECK_MSG(v_in.size() == width_, "RrefRowSpan::contains: width mismatch");
    std::vector<uint32_t> v = v_in;
    for (size_t i = 0; i < rows_.size(); ++i) {
        const uint32_t c = v[pivots_[i]];
        if (c == 0) continue;
        for (uint32_t j = pivots_[i]; j < width_; ++j) {
            if (rows_[i][j] == 0) continue;
            v[j] = ModArithmetic::mod_sub(v[j], ModArithmetic::mod_mul(c, rows_[i][j], ell_), ell_);
        }
    }
    for (uint32_t j = 0; j < width_; ++j) {
        if (v[j] != 0) return false;
    }
    return true;
}

Matrix RrefRowSpan::as_matrix() const {
    Matrix out(dim(), width_, ell_);
    for (uint32_t i = 0; i < dim(); ++i) out.data[i] = rows_[i];
    return out;
}

std::vector<uint32_t> VecOfMatrix(const Matrix& M) {
    std::vector<uint32_t> v(static_cast<size_t>(M.rows) * M.cols, 0u);
    for (uint32_t i = 0; i < M.rows; ++i) {
        for (uint32_t j = 0; j < M.cols; ++j) v[static_cast<size_t>(i) * M.cols + j] = M.data[i][j];
    }
    return v;
}

Matrix MatrixOfVec(const std::vector<uint32_t>& v, uint32_t rows, uint32_t cols, uint32_t ell) {
    ENGINE_CHECK_MSG(v.size() == static_cast<size_t>(rows) * cols,
                     "MatrixOfVec: vector length does not match the requested shape");
    Matrix out(rows, cols, ell);
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) out.data[i][j] = v[static_cast<size_t>(i) * cols + j];
    }
    return out;
}

Matrix algebra_closure(const std::vector<Matrix>& generators, uint32_t D, uint32_t ell,
                       uint32_t& dim_out) {
    ENGINE_CHECK_MSG(D > 0, "algebra_closure: D must be positive");
    ENGINE_CHECK_MSG(ell >= 2, "algebra_closure: ell must be at least 2");
    for (const Matrix& g : generators) {
        ENGINE_CHECK_MSG(g.rows == D && g.cols == D, "algebra_closure: generator is not D x D");
        ENGINE_CHECK_MSG(g.p == ell, "algebra_closure: generator over a different modulus");
    }

    const uint32_t width = D * D;
    RrefRowSpan span(width, ell);

    // The unit of the algebra, then the closure of the span under right
    // multiplication by each generator.  `frontier` holds the span elements
    // whose products have not been formed yet; each insert() grows the rank,
    // so the loop runs at most `width` times.
    std::vector<Matrix> frontier;
    {
        Matrix I(D, ell); // identity
        span.insert(VecOfMatrix(I));
        frontier.push_back(std::move(I));
    }
    for (size_t head = 0; head < frontier.size(); ++head) {
        const Matrix cur = frontier[head];
        for (const Matrix& g : generators) {
            Matrix prod = cur * g;
            if (span.insert(VecOfMatrix(prod))) frontier.push_back(std::move(prod));
        }
    }

    dim_out = span.dim();
    return span.as_matrix();
}

// ===========================================================================
// E0b-S (WP-3, hecke-extensions slot 3) -- strict locality mode's linear
// algebra.  See SubspaceOps.h for the statement; [X] Sec. 3.1 steps 1-2 and
// [E] Cert. `sec:localcert` "Strict mode" (1)-(2) for the mathematics.
// ===========================================================================

std::vector<uint32_t> RrefPivots(const Matrix& closure) {
    std::vector<uint32_t> pivots;
    pivots.reserve(closure.rows);
    uint32_t last = 0;
    bool have_last = false;
    for (uint32_t i = 0; i < closure.rows; ++i) {
        uint32_t piv = closure.cols;
        for (uint32_t j = 0; j < closure.cols; ++j) {
            if (closure.data[i][j] != 0) { piv = j; break; }
        }
        ENGINE_CHECK_MSG(piv < closure.cols, "RrefPivots: a zero row -- not an RREF basis");
        ENGINE_CHECK_MSG(!have_last || piv > last,
                         "RrefPivots: pivots are not strictly increasing -- not an RREF basis");
        last = piv;
        have_last = true;
        pivots.push_back(piv);
    }
    return pivots;
}

std::vector<uint32_t> AlgebraCoords(const Matrix& closure, const std::vector<uint32_t>& pivots,
                                    const Matrix& M, uint32_t ell, const char* who) {
    const std::vector<uint32_t> v = VecOfMatrix(M);
    ENGINE_CHECK_MSG(v.size() == static_cast<size_t>(closure.cols),
                     "AlgebraCoords: the matrix does not match the coordinate width");
    ENGINE_CHECK_MSG(pivots.size() == static_cast<size_t>(closure.rows),
                     "AlgebraCoords: pivot list does not match the basis");
    const uint32_t dim = closure.rows;
    std::vector<uint32_t> c(dim, 0u);
    for (uint32_t i = 0; i < dim; ++i) c[i] = v[pivots[i]];
    // Re-expand and compare: the coordinates are only READ OFF the pivots, so
    // this is what certifies that M really lies in the span.
    for (uint32_t j = 0; j < closure.cols; ++j) {
        uint32_t acc = 0;
        for (uint32_t i = 0; i < dim; ++i) {
            if (c[i] == 0 || closure.data[i][j] == 0) continue;
            acc = ModArithmetic::mod_add(acc, ModArithmetic::mod_mul(c[i], closure.data[i][j], ell),
                                         ell);
        }
        if (acc != v[j]) {
            throw std::runtime_error(std::string(who) +
                                     ": element is not in the span of the algebra basis");
        }
    }
    return c;
}

Matrix frobenius_matrix(const Matrix& closure, uint32_t D, uint32_t ell) {
    ENGINE_CHECK_MSG(D > 0, "frobenius_matrix: D must be positive");
    ENGINE_CHECK_MSG(ell >= 2, "frobenius_matrix: ell must be at least 2");
    ENGINE_CHECK_MSG(closure.cols == D * D, "frobenius_matrix: coordinate width != D*D");
    const std::vector<uint32_t> pivots = RrefPivots(closure);
    const uint32_t dim = closure.rows;
    Matrix F(dim, dim, ell);
    for (uint32_t i = 0; i < dim; ++i) {
        const Matrix b = MatrixOfVec(closure.data[i], D, D, ell);
        // b^ell by square-and-multiply on the bits of ell (low -> high).
        Matrix acc(D, ell);      // identity
        Matrix base = b;
        uint32_t e = ell;
        while (e != 0) {
            if (e & 1u) acc = acc * base;
            e >>= 1;
            if (e != 0) base = base * base;
        }
        const std::vector<uint32_t> c = AlgebraCoords(closure, pivots, acc, ell,
                                                      "frobenius_matrix");
        for (uint32_t r = 0; r < dim; ++r) F.data[r][i] = c[r];
    }
    return F;
}

Matrix frobenius_kernel(const Matrix& F, uint32_t N) {
    ENGINE_CHECK_MSG(F.rows == F.cols, "frobenius_kernel: the Frobenius matrix is not square");
    Matrix FN(F.rows, F.p);   // identity
    for (uint32_t i = 0; i < N; ++i) FN = FN * F;
    return KernelAsMatrix(FN);
}

} // namespace hecke
