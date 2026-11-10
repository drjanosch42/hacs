// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_SPARSE_RREF_H
#define HECKE_SPARSE_RREF_H
// ===========================================================================
// (S24 CL-G2, CODEAUDIT-B B1) THE ONE SPARSE GAUSS--JORDAN KERNEL behind level::SparseCokernel (F_p,
// HeckeLevelNSparse) and neb::NebSparseCokernel (F_q, HeckeNebentypusSparse).  Until slot 24 the two
// classes carried line-for-line copies of it that differed only in the field operations; here it is
// a template over a field policy `Ops`:
//
//     uint32_t add(a, b), mul(a, b), neg(a), inv(a)   -- on reduced codes;
//     uint32_t muladd(a, b, c) = a + b c              -- one fused reduction where the field allows it;
//     uint32_t reduce(a)                              -- a caller's code to a reduced one (F_p: a mod p);
//     uint32_t one()
//
// The algorithm is unchanged: the SMALLEST column of each reduced row is its pivot, so every stored
// row is e_c + (entries at columns > c, none at another pivot) and the result IS the unique reduced row
// echelon form of the span in the natural column order ([C] rmk:levelcost (4), prop:neborbit (3)).
// The consumption order (`seq`) changes no value, only the work counters.
//
// Internal to Hecke/: the public types and statistics stay those of the two classes.
// ===========================================================================

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "../../ModArithmetic/EngineCheck.h"

namespace hecke {
namespace level {

// A sparse vector over a field: strictly increasing columns, nonzero values.
struct SparseVec {
    std::vector<uint32_t> col;
    std::vector<uint32_t> val;
    size_t nnz() const { return col.size(); }
};

} // namespace level

namespace sparse {

using level::SparseVec;

// A dense accumulator over [0, D) with a touched list (reduced codes).
template <class Ops>
class Acc {
public:
    Acc(const Ops& ops, uint32_t D) : ops_(ops), val_(D, 0), mark_(D, 0) {}
    void add(uint32_t c, uint32_t a) {
        if (!mark_[c]) { mark_[c] = 1; touched_.push_back(c); val_[c] = 0; }
        val_[c] = ops_.add(val_[c], a);
    }
    // a * v added.
    void axpy(uint32_t a, const SparseVec& v) {
        for (size_t k = 0; k < v.col.size(); ++k) add(v.col[k], ops_.mul(a, v.val[k]));
    }
    SparseVec take() {
        std::sort(touched_.begin(), touched_.end());
        SparseVec out;
        for (uint32_t c : touched_) {
            if (val_[c] != 0) { out.col.push_back(c); out.val.push_back(val_[c]); }
            mark_[c] = 0;
            val_[c] = 0;
        }
        touched_.clear();
        return out;
    }
private:
    Ops ops_;
    std::vector<uint32_t> val_;
    std::vector<uint8_t> mark_;
    std::vector<uint32_t> touched_;
};

// row - f (e_piv + t), f = row's coefficient at piv; the merged row without piv, and the columns new in it.
template <class Ops>
void MergeOut(const Ops& ops, SparseVec& row, uint32_t piv, const SparseVec& t, const char* who, std::vector<uint32_t>& fresh,
              uint64_t& merge_ops) {
    SparseVec out;
    out.col.reserve(row.col.size() + t.col.size());
    out.val.reserve(row.col.size() + t.col.size());
    uint32_t f = 0;
    for (size_t k = 0; k < row.col.size(); ++k)
        if (row.col[k] == piv) { f = row.val[k]; break; }
    ENGINE_CHECK_MSG(f != 0, (std::string(who) + ": a back substitution into a row without the pivot").c_str());
    const uint32_t mf = ops.neg(f);
    size_t i = 0, j = 0;
    while (i < row.col.size() || j < t.col.size()) {
        if (i < row.col.size() && row.col[i] == piv) { ++i; continue; }
        if (j >= t.col.size() || (i < row.col.size() && row.col[i] < t.col[j])) {
            out.col.push_back(row.col[i]);
            out.val.push_back(row.val[i]);
            ++i;
        } else if (i >= row.col.size() || t.col[j] < row.col[i]) {
            out.col.push_back(t.col[j]);
            out.val.push_back(ops.mul(mf, t.val[j]));
            fresh.push_back(t.col[j]);
            ++j;
        } else {
            const uint32_t v = ops.muladd(row.val[i], mf, t.val[j]);
            if (v) { out.col.push_back(row.col[i]); out.val.push_back(v); }
            ++i;
            ++j;
        }
        ++merge_ops;
    }
    row = std::move(out);
}

// The stored reduced row echelon form.
struct Rref {
    std::vector<SparseVec> rows;           // tails, by insertion
    std::vector<int32_t> row_of_col;       // column -> its row, -1 off the pivots
    std::vector<uint32_t> pivots, free;    // ascending
    std::vector<int32_t> free_index;       // column -> its index among the free columns, -1 at a pivot
};

struct Counters {
    uint64_t generators = 0, generator_nnz = 0, dependent = 0;
    uint64_t final_nnz = 0, peak_nnz = 0, merge_ops = 0;
    uint64_t occ_entries = 0;   // row ids pushed to the occurrence lists
};

// The generators gens[seq[0]], gens[seq[1]], ... eliminated into R (D columns), sparse Gauss--Jordan.
template <class Ops>
void Eliminate(const Ops& ops, uint32_t D, const std::vector<SparseVec>& gens, const std::vector<uint32_t>& seq, const char* who,
               Rref& R, Counters& st) {
    std::vector<std::vector<uint32_t>> occ(D);      // column -> rows whose tail may contain it (lazy)
    Acc<Ops> acc(ops, D);
    uint64_t live_nnz = 0;
    R.row_of_col.assign(D, -1);
    for (const uint32_t gi : seq) {
        const SparseVec& g = gens[gi];
        ENGINE_CHECK_MSG(g.col.empty() || g.col.back() < D, (std::string(who) + ": a generator column out of range").c_str());
        ++st.generators;
        st.generator_nnz += g.nnz();
        // reduce against the fully reduced rows
        for (size_t k = 0; k < g.col.size(); ++k) {
            const uint32_t c = g.col[k];
            if (R.row_of_col[c] >= 0) {
                acc.add(c, 0);   // keep c touched so it cancels to 0 (the pivot entry itself is not added)
                const SparseVec& t = R.rows[static_cast<size_t>(R.row_of_col[c])];
                acc.axpy(ops.neg(ops.reduce(g.val[k])), t);
                st.merge_ops += t.nnz();
            } else {
                acc.add(c, ops.reduce(g.val[k]));
            }
        }
        SparseVec v = acc.take();
        // pivot entries cancelled above (added as 0); nothing else can hold a pivot column
        if (v.col.empty()) { ++st.dependent; continue; }
        const uint32_t piv = v.col[0];          // the smallest column: the natural RREF
        const uint32_t iv = ops.inv(v.val[0]);
        SparseVec tail;
        tail.col.assign(v.col.begin() + 1, v.col.end());
        tail.val.reserve(tail.col.size());
        for (size_t k = 1; k < v.col.size(); ++k) tail.val.push_back(ops.mul(v.val[k], iv));
        // back substitution into every row holding piv
        std::vector<uint32_t> rows_with;
        rows_with.swap(occ[piv]);
        std::sort(rows_with.begin(), rows_with.end());
        rows_with.erase(std::unique(rows_with.begin(), rows_with.end()), rows_with.end());
        for (uint32_t r : rows_with) {
            SparseVec& row = R.rows[r];
            if (!std::binary_search(row.col.begin(), row.col.end(), piv)) continue;   // stale
            std::vector<uint32_t> fresh;
            const size_t before = row.nnz();
            MergeOut(ops, row, piv, tail, who, fresh, st.merge_ops);
            live_nnz = live_nnz + row.nnz() - before;
            for (uint32_t c : fresh) occ[c].push_back(r);
            st.occ_entries += fresh.size();
        }
        const uint32_t id = static_cast<uint32_t>(R.rows.size());
        for (uint32_t c : tail.col) occ[c].push_back(id);
        st.occ_entries += tail.nnz();
        live_nnz += tail.nnz();
        st.peak_nnz = std::max(st.peak_nnz, live_nnz);
        R.row_of_col[piv] = static_cast<int32_t>(id);
        R.rows.push_back(std::move(tail));
    }
    for (uint32_t c = 0; c < D; ++c) {
        if (R.row_of_col[c] >= 0) R.pivots.push_back(c);
        else R.free.push_back(c);
    }
    R.free_index.assign(D, -1);
    for (size_t i = 0; i < R.free.size(); ++i) R.free_index[R.free[i]] = static_cast<int32_t>(i);
    for (const SparseVec& r : R.rows) {
        st.final_nnz += r.nnz();
        for (uint32_t c : r.col)
            ENGINE_CHECK_MSG(R.row_of_col[c] < 0, (std::string(who) + ": a stored row holds another pivot (not reduced)").c_str());
    }
    for (uint32_t c : R.pivots) {
        const SparseVec& t = R.rows[static_cast<size_t>(R.row_of_col[c])];
        ENGINE_CHECK_MSG(t.col.empty() || t.col.front() > c, (std::string(who) + ": a row is not in natural echelon form").c_str());
    }
}

// Coordinates on the free columns of x mod the row space of R.
template <class Ops>
std::vector<uint32_t> Coords(const Ops& ops, const Rref& R, const SparseVec& x, const char* who) {
    std::vector<uint32_t> out(R.free.size(), 0);
    const size_t D = R.row_of_col.size();
    for (size_t k = 0; k < x.col.size(); ++k) {
        const uint32_t c = x.col[k];
        ENGINE_CHECK_MSG(c < D, (std::string(who) + "::coords: column out of range").c_str());
        const uint32_t a = ops.reduce(x.val[k]);
        if (a == 0) continue;
        if (R.row_of_col[c] >= 0) {
            const SparseVec& t = R.rows[static_cast<size_t>(R.row_of_col[c])];
            const uint32_t ma = ops.neg(a);
            for (size_t i = 0; i < t.col.size(); ++i) {
                uint32_t& o = out[static_cast<size_t>(R.free_index[t.col[i]])];
                o = ops.muladd(o, ma, t.val[i]);
            }
        } else {
            uint32_t& o = out[static_cast<size_t>(R.free_index[c])];
            o = ops.add(o, a);
        }
    }
    return out;
}

// x (dense, length D) as a sparse vector of reduced codes.
template <class Ops>
SparseVec FromDense(const Ops& ops, const std::vector<uint32_t>& x) {
    SparseVec s;
    for (uint32_t c = 0; c < x.size(); ++c) {
        const uint32_t a = ops.reduce(x[c]);
        if (a) { s.col.push_back(c); s.val.push_back(a); }
    }
    return s;
}

// The rank of vectors supported in disjoint orbits: echelon (not reduced) with the smallest column as pivot; the rows
// stay orbit-local, so this costs O(sum nnz * orbit size).
template <class Ops>
uint32_t EchelonRank(const Ops& ops, uint32_t D, const std::vector<SparseVec>& vecs) {
    std::vector<int32_t> row_of_col(D, -1);
    std::vector<SparseVec> rows;
    Acc<Ops> acc(ops, D);
    for (const SparseVec& g : vecs) {
        for (size_t k = 0; k < g.col.size(); ++k) acc.add(g.col[k], ops.reduce(g.val[k]));
        SparseVec v = acc.take();
        while (!v.col.empty() && row_of_col[v.col[0]] >= 0) {
            const SparseVec& r = rows[static_cast<size_t>(row_of_col[v.col[0]])];   // normalised: leading 1
            acc.axpy(ops.one(), v);
            acc.axpy(ops.neg(v.val[0]), r);
            v = acc.take();
        }
        if (v.col.empty()) continue;
        const uint32_t iv = ops.inv(v.val[0]);
        for (uint32_t& x : v.val) x = ops.mul(x, iv);
        row_of_col[v.col[0]] = static_cast<int32_t>(rows.size());
        rows.push_back(std::move(v));
    }
    return static_cast<uint32_t>(rows.size());
}

} // namespace sparse
} // namespace hecke

#endif // HECKE_SPARSE_RREF_H
