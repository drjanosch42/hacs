// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 22, MP-2) Exact tools for comparing cochain complexes on cohomology -- see CochainMaps.h.
#include "CochainMaps.h"

#include <algorithm>

#include "../../../ModArithmetic/DenseElim.h"
#include "../../../ModArithmetic/DenseView.h"
#include "../../../ModArithmetic/EngineCheck.h"

namespace orbifold_mp {

engine::ModCtx<uint32_t> RingCtx(uint32_t ell, uint32_t k) {
    ENGINE_CHECK_MSG(k >= 1, "orbifold_mp::RingCtx: k >= 1");
    return (k == 1) ? engine::ModCtx<uint32_t>(ell) : engine::ModCtx<uint32_t>::prime_power(ell, k);
}

uint32_t PowU(uint32_t ell, uint32_t k) {
    uint32_t m = 1;
    for (uint32_t i = 0; i < k; ++i) m *= ell;
    return m;
}

SparseRows FromDense(const engine::DenseBuf<uint32_t>& A, uint32_t m) {
    ENGINE_CHECK_MSG(m >= 2 && m <= 255, "orbifold_mp::FromDense: 2 <= m <= 255");
    SparseRows S;
    S.ncols = A.cols;
    for (uint32_t i = 0; i < A.rows; ++i) {
        for (uint32_t j = 0; j < A.cols; ++j) {
            const uint32_t v = A.at(i, j) % m;
            if (v) {
                S.col.push_back(j);
                S.val.push_back(static_cast<uint8_t>(v));
            }
        }
        S.start.push_back(S.col.size());
    }
    return S;
}

SparseRows FromRows(const std::vector<Vec>& rows, uint32_t ncols, uint32_t m) {
    ENGINE_CHECK_MSG(m >= 2 && m <= 255, "orbifold_mp::FromRows: 2 <= m <= 255");
    SparseRows S;
    S.ncols = ncols;
    for (const Vec& r : rows) {
        ENGINE_CHECK_MSG(r.size() == ncols, "orbifold_mp::FromRows: row length");
        for (uint32_t j = 0; j < ncols; ++j) {
            const uint32_t v = r[j] % m;
            if (v) {
                S.col.push_back(j);
                S.val.push_back(static_cast<uint8_t>(v));
            }
        }
        S.start.push_back(S.col.size());
    }
    return S;
}

SparseRows Stack(const SparseRows& A, const SparseRows& B) {
    ENGINE_CHECK_MSG(A.ncols == B.ncols, "orbifold_mp::Stack: column counts differ");
    SparseRows S = A;
    const uint64_t off = S.col.size();
    S.col.insert(S.col.end(), B.col.begin(), B.col.end());
    S.val.insert(S.val.end(), B.val.begin(), B.val.end());
    for (uint64_t r = 1; r < B.start.size(); ++r) S.start.push_back(off + B.start[r]);
    return S;
}

SparseRows SpMul(const SparseRows& A, const SparseRows& B, uint32_t m) {
    ENGINE_CHECK_MSG(A.ncols == B.rows(), "orbifold_mp::SpMul: inner dimensions differ");
    SparseRows C;
    C.ncols = B.ncols;
    std::vector<uint32_t> acc(B.ncols, 0);
    std::vector<uint8_t> hit(B.ncols, 0);
    std::vector<uint32_t> touched;
    for (uint64_t r = 0; r < A.rows(); ++r) {
        touched.clear();
        for (uint64_t t = A.start[r]; t < A.start[r + 1]; ++t) {
            const uint32_t c = A.col[t], v = A.val[t];
            for (uint64_t u = B.start[c]; u < B.start[c + 1]; ++u) {
                const uint32_t cc = B.col[u];
                if (!hit[cc]) {
                    hit[cc] = 1;
                    touched.push_back(cc);
                }
                acc[cc] = (acc[cc] + v * B.val[u]) % m;
            }
        }
        std::sort(touched.begin(), touched.end());
        for (uint32_t cc : touched) {
            if (acc[cc]) {
                C.col.push_back(cc);
                C.val.push_back(static_cast<uint8_t>(acc[cc]));
            }
            acc[cc] = 0;
            hit[cc] = 0;
        }
        C.start.push_back(C.col.size());
    }
    return C;
}

bool SpEqual(const SparseRows& A, const SparseRows& B) {
    return A.ncols == B.ncols && A.start == B.start && A.col == B.col && A.val == B.val;
}

Vec SpMatVec(const SparseRows& A, const Vec& x, uint32_t m) {
    ENGINE_CHECK_MSG(x.size() == A.ncols, "orbifold_mp::SpMatVec: shapes");
    Vec y(A.rows(), 0);
    for (uint64_t r = 0; r < A.rows(); ++r) {
        uint64_t acc = 0;
        for (uint64_t t = A.start[r]; t < A.start[r + 1]; ++t) acc += static_cast<uint64_t>(A.val[t]) * x[A.col[t]];
        y[r] = static_cast<uint32_t>(acc % m);
    }
    return y;
}

Vec DenseMatVec(const engine::DenseBuf<uint32_t>& A, const Vec& x, uint32_t m) {
    ENGINE_CHECK_MSG(x.size() == A.cols, "orbifold_mp::DenseMatVec: shapes");
    Vec y(A.rows, 0);
    for (uint32_t i = 0; i < A.rows; ++i) {
        uint64_t acc = 0;
        for (uint32_t j = 0; j < A.cols; ++j)
            if (x[j]) acc = (acc + static_cast<uint64_t>(A.at(i, j) % m) * x[j]) % m;
        y[i] = static_cast<uint32_t>(acc);
    }
    return y;
}

bool IsZero(const Vec& v) {
    for (uint32_t x : v)
        if (x) return false;
    return true;
}

std::vector<Vec> KernelGenerators(const engine::ModCtx<uint32_t>& ctx, const engine::DenseBuf<uint32_t>& A) {
    std::vector<Vec> out;
    const uint32_t n = A.cols;
    if (n == 0) return out;
    if (A.rows == 0) {
        for (uint32_t j = 0; j < n; ++j) {
            Vec e(n, 0);
            e[j] = 1;
            out.push_back(e);
        }
        return out;
    }
    if (ctx.is_field()) {
        engine::DenseBuf<uint32_t> R = A;
        const engine::elim::RrefResult rr = engine::elim::rref(ctx, engine::view(R));
        const engine::DenseBuf<uint32_t> K = engine::elim::kernel_basis(ctx, engine::cview(R), rr.pivots);
        for (uint32_t i = 0; i < K.rows; ++i) {
            Vec v(n);
            for (uint32_t j = 0; j < n; ++j) v[j] = K.at(i, j);
            out.push_back(v);
        }
    } else {
        const auto h = engine::elim::kernel_module(ctx, engine::cview(A));
        for (uint32_t i = 0; i < h.H.rows; ++i) {
            Vec v(n);
            for (uint32_t j = 0; j < n; ++j) v[j] = h.H.at(i, j);
            out.push_back(v);
        }
    }
    return out;
}

uint64_t DenseLength(const engine::ModCtx<uint32_t>& ctx, const engine::DenseBuf<uint32_t>& A) {
    if (A.rows == 0 || A.cols == 0) return 0;
    if (ctx.is_field()) {
        engine::DenseBuf<uint32_t> R = A;
        return engine::elim::rref(ctx, engine::view(R)).rank;
    }
    return engine::elim::length(ctx, engine::cview(A));
}

uint64_t SpanLength(const std::vector<Vec>& rows, uint32_t ncols, uint32_t ell, uint32_t k) {
    if (rows.empty() || ncols == 0) return 0;
    return RowSpanLength(FromRows(rows, ncols, PowU(ell, k)), ell, k).length;
}

ImageInH ImageOfCocycles(const std::vector<Vec>& mapped, uint32_t dimCq, const SparseRows* dprev, const SparseRows* dq,
                         uint32_t ell, uint32_t k) {
    const uint32_t m = PowU(ell, k);
    ImageInH R;
    SparseRows B;
    B.ncols = dimCq;
    if (dprev != nullptr) {
        ENGINE_CHECK_MSG(dprev->rows() == dimCq, "orbifold_mp::ImageOfCocycles: d^{q-1} has the wrong row count");
        B = Transpose(*dprev);
        B.ncols = dimCq;
        R.len_B = B.rows() ? RowSpanLength(B, ell, k).length : 0;
    }
    if (dq != nullptr)
        for (const Vec& v : mapped) R.cocycles = R.cocycles && IsZero(SpMatVec(*dq, v, m));
    const SparseRows all = Stack(B, FromRows(mapped, dimCq, m));
    R.len_B_plus = all.rows() ? RowSpanLength(all, ell, k).length : 0;
    return R;
}

}  // namespace orbifold_mp
