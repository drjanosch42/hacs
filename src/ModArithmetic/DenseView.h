// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// =====================================================================
// DenseView.h -- a window onto a DenseBuf (Phase 2, plan D3.1)
// =====================================================================
//
// DenseView<Elt> = { buf*, row0, rows, col0, cols, transposed }.
//
//   * A sub-range is an offset: sub(r0, nr, c0, nc) is free.
//   * A LOGICAL transpose is a flag flip: transposed() is free and changes
//     only the (i, j) -> memory map. It does NOT change the memory access
//     pattern -- a row-major r x c buffer read through a transposed view is
//     column-major, and the GEMM fast path decides on the RESOLVED strides,
//     not on the flag. A PHYSICAL relayout is kernels::transpose / copy.
//   * at(i, j) resolves layout + transpose. Kernels never index a buffer
//     directly; they call strides() once and walk pointers.
//
// DenseView<const Elt> is the read-only view; DenseView<Elt> converts to it
// implicitly. The view is a handle (like a span): copying a view does not
// copy data, and const-ness of the view object itself is not const-ness of
// the data.

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "DenseBuf.h"
#include "EngineCheck.h"

namespace engine {

template<class Elt>
class DenseView {
public:
    using Value = std::remove_const_t<Elt>;
    using Buf   = std::conditional_t<std::is_const<Elt>::value,
                                     const DenseBuf<Value>, DenseBuf<Value>>;

    // Resolved addressing: element (i, j) of the LOGICAL view lives at
    // base[i * rs + j * cs].
    struct Strides {
        Elt*   base;
        size_t rs;
        size_t cs;
    };

    DenseView() = default;
    DenseView(Buf& b)   // whole buffer, untransposed
        : buf_(&b), row0_(0), rows_(b.rows), col0_(0), cols_(b.cols), transposed_(false) {}
    DenseView(Buf& b, uint32_t row0, uint32_t rows, uint32_t col0, uint32_t cols,
              bool transposed = false)
        : buf_(&b), row0_(row0), rows_(rows), col0_(col0), cols_(cols), transposed_(transposed) {
        ENGINE_CHECK_MSG(static_cast<uint64_t>(row0) + rows <= b.rows &&
                         static_cast<uint64_t>(col0) + cols <= b.cols,
                         "DenseView: window exceeds buffer");
    }
    // mutable -> const conversion
    template<class E2, class = std::enable_if_t<std::is_const<Elt>::value &&
                                                std::is_same<Value, std::remove_const_t<E2>>::value &&
                                                !std::is_const<E2>::value>>
    DenseView(const DenseView<E2>& o)
        : buf_(o.buffer()), row0_(o.row0()), rows_(o.buf_rows()), col0_(o.col0()),
          cols_(o.buf_cols()), transposed_(o.transposed()) {}

    // Logical shape (after transposition).
    uint32_t rows() const { return transposed_ ? cols_ : rows_; }
    uint32_t cols() const { return transposed_ ? rows_ : cols_; }
    bool     empty() const { return rows() == 0 || cols() == 0; }

    // Window in buffer coordinates (before transposition).
    Buf*     buffer() const   { return buf_; }
    uint32_t row0() const     { return row0_; }
    uint32_t col0() const     { return col0_; }
    uint32_t buf_rows() const { return rows_; }
    uint32_t buf_cols() const { return cols_; }
    bool     transposed() const { return transposed_; }
    bool     covers_whole_buffer() const {
        return row0_ == 0 && col0_ == 0 && rows_ == buf_->rows && cols_ == buf_->cols;
    }
    Invariant invariant() const { return buf_->invariant; }
    Structure structure() const { return buf_->structure; }

    Strides strides() const {
        Elt* base = buf_->data() + buf_->index(row0_, col0_);
        size_t rs = buf_->row_stride(), cs = buf_->col_stride();
        if (transposed_) { size_t t = rs; rs = cs; cs = t; }
        return Strides{base, rs, cs};
    }
    // Pointer to logical (0, 0) and the two resolved strides.
    Elt*   data() const       { return strides().base; }
    size_t row_stride() const { return strides().rs; }
    size_t col_stride() const { return strides().cs; }

    Elt& at(uint32_t i, uint32_t j) const {
        const Strides s = strides();
        return s.base[i * s.rs + j * s.cs];
    }
    Elt& checked_at(uint32_t i, uint32_t j) const {
        ENGINE_CHECK(i < rows() && j < cols());
        return at(i, j);
    }

    // Free derived views (logical coordinates).
    DenseView sub(uint32_t r0, uint32_t nr, uint32_t c0, uint32_t nc) const {
        ENGINE_CHECK_MSG(static_cast<uint64_t>(r0) + nr <= rows() &&
                         static_cast<uint64_t>(c0) + nc <= cols(),
                         "DenseView::sub: window exceeds view");
        if (transposed_) return DenseView(*buf_, row0_ + c0, nc, col0_ + r0, nr, true);
        return DenseView(*buf_, row0_ + r0, nr, col0_ + c0, nc, false);
    }
    DenseView row(uint32_t i) const { return sub(i, 1, 0, cols()); }
    DenseView col(uint32_t j) const { return sub(0, rows(), j, 1); }
    DenseView rows_range(uint32_t r0, uint32_t nr) const { return sub(r0, nr, 0, cols()); }
    DenseView cols_range(uint32_t c0, uint32_t nc) const { return sub(0, rows(), c0, nc); }
    DenseView transposed_view() const {
        return DenseView(*buf_, row0_, rows_, col0_, cols_, !transposed_);
    }

    // True when (logical) element (i, j+1) is adjacent in memory to (i, j):
    // rows are contiguous -- "k-contiguous" for a GEMM left operand.
    bool row_contiguous() const { return col_stride() == 1; }
    // True when (i+1, j) is adjacent to (i, j): columns are contiguous.
    bool col_contiguous() const { return row_stride() == 1; }

private:
    template<class> friend class DenseView;
    Buf*     buf_        = nullptr;
    uint32_t row0_       = 0, rows_ = 0, col0_ = 0, cols_ = 0;
    bool     transposed_ = false;
};

template<class Elt> DenseView<Elt>       view(DenseBuf<Elt>& b)       { return DenseView<Elt>(b); }
template<class Elt> DenseView<const Elt> view(const DenseBuf<Elt>& b) { return DenseView<const Elt>(b); }
template<class Elt> DenseView<const Elt> cview(const DenseBuf<Elt>& b) { return DenseView<const Elt>(b); }

// Entrywise equality of two views' logical contents (any layouts).
template<class E1, class E2>
bool same_entries(const DenseView<E1>& a, const DenseView<E2>& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) return false;
    for (uint32_t i = 0; i < a.rows(); ++i)
        for (uint32_t j = 0; j < a.cols(); ++j)
            if (a.at(i, j) != b.at(i, j)) return false;
    return true;
}

} // namespace engine
