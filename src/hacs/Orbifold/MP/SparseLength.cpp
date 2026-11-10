// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 22, MP-1b) Row-span length over Z/ell^e -- see SparseLength.h.
#include "SparseLength.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <map>
#include <queue>

#include "../../../ModArithmetic/DenseBuf.h"
#include "../../../ModArithmetic/DenseElim.h"
#include "../../../ModArithmetic/DenseView.h"
#include "../../../ModArithmetic/EngineCheck.h"
#include "../../../ModArithmetic/ModCtx.h"

namespace orbifold_mp {

namespace {

class Reducer {
public:
    Reducer(uint32_t ncols, uint32_t m, uint32_t ell) : m_(m), ell_(ell), acc_(ncols, 0), mark_(ncols, 0), piv_(ncols, -1) {
        inv_.assign(m, 0);
        for (uint32_t a = 1; a < m; ++a)
            for (uint32_t b = 1; b < m; ++b)
                if ((a * b) % m == 1) inv_[a] = b;
    }

    bool unit(uint32_t v) const { return v % ell_ != 0; }

    // Loads a row into the accumulator.
    void load(const uint32_t* c, const uint8_t* v, uint64_t len) {
        for (uint64_t t = 0; t < len; ++t) set(c[t], v[t]);
    }

    // Reduces the accumulator against the pivots.  stop_at_free: return at the first non-pivot column with a
    // unit value (its column), the remaining heap untouched.  Otherwise reduce every pivot column away.
    // Returns the lead column found (UINT32_MAX if none) and whether it carries a unit.
    uint32_t reduce(bool stop_at_free, bool* lead_unit) {
        uint32_t lead = UINT32_MAX;
        *lead_unit = false;
        while (!heap_.empty()) {
            const uint32_t c = heap_.top();
            const uint32_t a = acc_[c];
            if (a == 0) {
                heap_.pop();
                mark_[c] = 0;
                continue;
            }
            const int64_t p = piv_[c];
            if (p >= 0) {
                heap_.pop();
                mark_[c] = 0;
                const uint64_t s = pstart_[p], e = pstart_[p + 1];
                const uint32_t f = m_ - a;   // acc += (m - a) * pivot; pivot[c] == 1
                ++applications_;
                applied_ += e - s;
                for (uint64_t t = s; t < e; ++t) {
                    const uint32_t cc = pcol_[t];
                    set(cc, (acc_[cc] + f * pval_[t]) % m_);
                }
                continue;
            }
            // c is a non-pivot column with a non-zero value.
            if (lead == UINT32_MAX) {
                lead = c;
                *lead_unit = unit(a);
                if (stop_at_free && *lead_unit) return lead;
            }
            // A non-unit lead (or any later non-pivot column): set it aside and keep reducing past it.
            heap_.pop();
            mark_[c] = 0;
            aside_.push_back(c);
        }
        return lead;
    }

    // Collects the accumulator's non-zeros (heap + aside), sorted, and clears the accumulator.
    void collect(std::vector<uint32_t>* cols, std::vector<uint8_t>* vals) {
        cols->clear();
        vals->clear();
        std::vector<uint32_t> all;
        all.swap(aside_);
        while (!heap_.empty()) {
            all.push_back(heap_.top());
            mark_[heap_.top()] = 0;
            heap_.pop();
        }
        std::sort(all.begin(), all.end());
        all.erase(std::unique(all.begin(), all.end()), all.end());
        for (uint32_t c : all)
            if (acc_[c] != 0) {
                cols->push_back(c);
                vals->push_back(static_cast<uint8_t>(acc_[c]));
                acc_[c] = 0;
            }
    }

    void add_pivot(const std::vector<uint32_t>& cols, const std::vector<uint8_t>& vals) {
        const uint32_t lead = cols.front();
        const uint32_t s = inv_[vals.front()];
        ENGINE_CHECK_MSG(s != 0, "RowSpanLength: pivot lead is not a unit");
        piv_[lead] = static_cast<int64_t>(pstart_.size()) - 1;
        for (size_t t = 0; t < cols.size(); ++t) {
            pcol_.push_back(cols[t]);
            pval_.push_back(static_cast<uint8_t>((vals[t] * s) % m_));
        }
        pstart_.push_back(pcol_.size());
        max_row_ = std::max<uint64_t>(max_row_, cols.size());
    }

    bool is_pivot_col(uint32_t c) const { return piv_[c] >= 0; }
    uint64_t pivots() const { return pstart_.size() - 1; }
    uint64_t pivot_nnz() const { return pcol_.size(); }
    uint64_t max_row() const { return max_row_; }
    uint64_t applications() const { return applications_; }
    uint64_t applied() const { return applied_; }

private:
    void set(uint32_t c, uint32_t v) {
        acc_[c] = v;
        if (v != 0 && !mark_[c]) {
            mark_[c] = 1;
            heap_.push(c);
        }
    }

    uint32_t m_, ell_;
    std::vector<uint32_t> acc_;
    std::vector<uint8_t> mark_;
    std::vector<int64_t> piv_;
    std::vector<uint32_t> inv_;
    std::priority_queue<uint32_t, std::vector<uint32_t>, std::greater<uint32_t>> heap_;
    std::vector<uint32_t> aside_;
    std::vector<uint64_t> pstart_{0};
    std::vector<uint32_t> pcol_;
    std::vector<uint8_t> pval_;
    uint64_t max_row_ = 0;
    uint64_t applications_ = 0, applied_ = 0;
};

}  // namespace

LengthStats RowSpanLength(const SparseRows& A, uint32_t ell, uint32_t e, uint64_t dense_budget, RowOrder row_order) {
    const auto t0 = std::chrono::steady_clock::now();
    ENGINE_CHECK_MSG(ell >= 2 && e >= 1, "RowSpanLength: ell >= 2, e >= 1");
    uint32_t m = 1;
    for (uint32_t i = 0; i < e; ++i) m *= ell;
    ENGINE_CHECK_MSG(m <= 255, "RowSpanLength: ell^e must be <= 255 (uint8 entries)");
    LengthStats st;
    st.rows = A.rows();
    st.cols = A.ncols;
    st.input_nnz = A.nnz();
    Reducer R(A.ncols, m, ell);
    std::vector<uint32_t> cols;
    std::vector<uint8_t> vals;
    std::vector<std::vector<uint32_t>> rcols;
    std::vector<std::vector<uint8_t>> rvals;
    std::vector<uint64_t> order(A.rows());
    for (uint64_t r = 0; r < A.rows(); ++r) order[r] = r;
    if (row_order == RowOrder::Reversed) std::reverse(order.begin(), order.end());
    for (uint64_t r : order) {
        const uint64_t s = A.start[r], len = A.start[r + 1] - s;
        if (len == 0) continue;
        R.load(&A.col[s], &A.val[s], len);
        bool lead_unit = false;
        const uint32_t lead = R.reduce(true, &lead_unit);
        R.collect(&cols, &vals);
        if (lead == UINT32_MAX) continue;   // reduced to zero
        if (lead_unit) {
            R.add_pivot(cols, vals);
        } else {
            rcols.push_back(cols);
            rvals.push_back(vals);
        }
    }
    // Re-reduce the residual rows against every pivot; keep the non-zero ones.
    std::vector<uint32_t> used;
    std::vector<std::vector<uint32_t>> fc;
    std::vector<std::vector<uint8_t>> fv;
    for (size_t i = 0; i < rcols.size(); ++i) {
        R.load(rcols[i].data(), rvals[i].data(), rcols[i].size());
        bool lu = false;
        (void)R.reduce(false, &lu);
        R.collect(&cols, &vals);
        for (uint32_t c : cols) ENGINE_CHECK_MSG(!R.is_pivot_col(c), "RowSpanLength: residual row meets a pivot column");
        if (cols.empty()) continue;
        for (uint32_t c : cols) used.push_back(c);
        fc.push_back(cols);
        fv.push_back(vals);
    }
    std::sort(used.begin(), used.end());
    used.erase(std::unique(used.begin(), used.end()), used.end());
    st.pivots = R.pivots();
    st.pivot_nnz = R.pivot_nnz();
    st.max_pivot_row = R.max_row();
    st.pivot_applications = R.applications();
    st.applied_entries = R.applied();
    st.residual_rows = fc.size();
    st.residual_cols = used.size();
    if (!fc.empty()) {
        ENGINE_CHECK_MSG(e >= 2, "RowSpanLength: a residual over a field");
        ENGINE_CHECK_MSG(static_cast<uint64_t>(fc.size()) * used.size() <= dense_budget,
                         "RowSpanLength: the dense residual exceeds its budget");
        const engine::ModCtx<uint32_t> ctx = engine::ModCtx<uint32_t>::prime_power(ell, e);
        engine::DenseBuf<uint32_t> D(static_cast<uint32_t>(fc.size()), static_cast<uint32_t>(used.size()));
        for (size_t i = 0; i < fc.size(); ++i)
            for (size_t t = 0; t < fc[i].size(); ++t) {
                const uint32_t j = static_cast<uint32_t>(std::lower_bound(used.begin(), used.end(), fc[i][t]) - used.begin());
                D.at(static_cast<uint32_t>(i), j) = fv[i][t];
            }
        st.residual_length = engine::elim::length(ctx, engine::cview(D));
    }
    st.length = static_cast<uint64_t>(e) * st.pivots + st.residual_length;
    st.wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return st;
}

SparseRows Transpose(const SparseRows& A) {
    // (MP-2) Replaces MP-1b's dead ReverseColumns (F-MP1 F6): the column span of A is the row span of A^T.
    SparseRows B;
    B.ncols = static_cast<uint32_t>(A.rows());
    ENGINE_CHECK_MSG(A.rows() < (uint64_t(1) << 32), "orbifold_mp::Transpose: more than 2^32 rows");
    std::vector<uint64_t> cnt(static_cast<size_t>(A.ncols) + 1, 0);
    for (uint32_t c : A.col) ++cnt[c + 1];
    for (size_t c = 0; c < A.ncols; ++c) cnt[c + 1] += cnt[c];
    B.start.assign(cnt.begin(), cnt.end());
    B.col.resize(A.nnz());
    B.val.resize(A.nnz());
    std::vector<uint64_t> pos(cnt.begin(), cnt.end() - 1);
    for (uint64_t r = 0; r < A.rows(); ++r)
        for (uint64_t t = A.start[r]; t < A.start[r + 1]; ++t) {
            const uint64_t at = pos[A.col[t]]++;
            B.col[at] = static_cast<uint32_t>(r);
            B.val[at] = A.val[t];
        }
    return B;
}

}  // namespace orbifold_mp
