// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// =====================================================================
// DenseBuf.h -- the dense buffer of the matrix engine (Phase 2, plan D3.1)
// =====================================================================
//
// DenseBuf<Elt> is an owned (or, arena-ready, externally backed) contiguous
// buffer with an explicit physical layout, leading dimension, arithmetic
// invariant and structure tag. It replaces FlatMatrix's hard-coded
// `r*cols_+c` (audit 01 Sec. 9.3). It is a plain data holder: every
// arithmetic operation lives in Kernels.h and takes DenseView<Elt>s.
//
//   layout     RowMajor : element (i,j) at data()[i*ld + j],  ld >= cols
//              ColMajor : element (i,j) at data()[j*ld + i],  ld >= rows
//   invariant  Canonical : every entry in [0, p)
//              Lazy2p    : every entry in [0, 2p)   (u32 only; Shoup output)
//              Accum64   : entries are exact uint64_t partial sums of at most
//                          k_max products of canonical residues (u32 modulus;
//                          the buffer is then a DenseBuf<uint64_t>)
//   structure  Generic / UpperUnitri / LowerUnitri / SignedAntiDiag / Diagonal
//              -- a PROMISE about which entries a kernel may read (see
//              ENGINE_API.md): a UpperUnitri buffer's diagonal and strict
//              lower triangle are never read (they may hold anything, e.g.
//              the other factor of a packed LU).
//
// Padding between rows (ld > cols) is never read, never written, never
// compared by any kernel.
//
// ARENA READINESS. The externally-backed constructor takes (pointer,
// capacity); an arena of Phase 4 hands out exactly that. Owned buffers use
// std::vector<Elt>. Nothing else in the engine may depend on which of the
// two backs a buffer.

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "EngineCheck.h"

namespace engine {

enum class Layout    : uint8_t { RowMajor, ColMajor };
enum class Invariant : uint8_t { Canonical, Lazy2p, Accum64 };
enum class Structure : uint8_t { Generic, UpperUnitri, LowerUnitri, SignedAntiDiag, Diagonal };


template<class Elt>
class DenseBuf {
public:
    uint32_t  rows      = 0;
    uint32_t  cols      = 0;
    uint32_t  ld        = 0;
    Layout    layout    = Layout::RowMajor;
    Invariant invariant = Invariant::Canonical;
    Structure structure = Structure::Generic;

    DenseBuf() = default;

    // Owned, zero-initialised. ld == 0 selects the minimal leading dimension.
    DenseBuf(uint32_t rows_, uint32_t cols_, Layout layout_ = Layout::RowMajor,
             uint32_t ld_ = 0, Invariant inv = Invariant::Canonical,
             Structure st = Structure::Generic)
        : rows(rows_), cols(cols_), ld(ld_), layout(layout_), invariant(inv), structure(st) {
        if (ld == 0) ld = minor_dim();
        ENGINE_CHECK_MSG(ld >= minor_dim(), "DenseBuf: leading dimension too small");
        owned_.assign(footprint(), Elt(0));
        data_ = owned_.data();
        capacity_ = owned_.size();
    }

    // Externally backed (arena / borrowed memory): non-owning, not zeroed.
    DenseBuf(Elt* external, size_t capacity, uint32_t rows_, uint32_t cols_,
             Layout layout_, uint32_t ld_ = 0, Invariant inv = Invariant::Canonical,
             Structure st = Structure::Generic)
        : rows(rows_), cols(cols_), ld(ld_), layout(layout_), invariant(inv), structure(st) {
        if (ld == 0) ld = minor_dim();
        ENGINE_CHECK_MSG(ld >= minor_dim(), "DenseBuf: leading dimension too small");
        ENGINE_CHECK_MSG(capacity >= footprint(), "DenseBuf: external capacity too small");
        data_ = external;
        capacity_ = capacity;
    }

    // (S24 G2-H) Owned, ADOPTING a row-major store of exactly rows_ * cols_ entries (moved in, not copied, not
    // zeroed): element (i, j) is v[i * cols_ + j].  howell_rows uses it to make its store the result's H.
    static DenseBuf adopt(std::vector<Elt>&& v, uint32_t rows_, uint32_t cols_) {
        ENGINE_CHECK_MSG(v.size() == static_cast<size_t>(rows_) * cols_, "DenseBuf::adopt: store size != rows * cols");
        DenseBuf b;
        b.rows = rows_; b.cols = cols_; b.ld = cols_;
        b.owned_ = std::move(v);
        b.data_ = b.owned_.empty() ? nullptr : b.owned_.data();
        b.capacity_ = b.owned_.size();
        return b;
    }

    // Copies keep the layout/tags; an externally backed source becomes owned.
    DenseBuf(const DenseBuf& o)
        : rows(o.rows), cols(o.cols), ld(o.ld), layout(o.layout),
          invariant(o.invariant), structure(o.structure),
          owned_(o.data_, o.data_ + o.footprint()) {
        data_ = owned_.data();
        capacity_ = owned_.size();
    }
    DenseBuf& operator=(const DenseBuf& o) {
        if (this != &o) { DenseBuf tmp(o); swap(tmp); }
        return *this;
    }
    DenseBuf(DenseBuf&& o) noexcept { swap(o); }
    DenseBuf& operator=(DenseBuf&& o) noexcept {
        if (this != &o) { DenseBuf tmp; tmp.swap(*this); swap(o); }
        return *this;
    }
    void swap(DenseBuf& o) noexcept {
        std::swap(rows, o.rows); std::swap(cols, o.cols); std::swap(ld, o.ld);
        std::swap(layout, o.layout); std::swap(invariant, o.invariant);
        std::swap(structure, o.structure);
        owned_.swap(o.owned_);
        std::swap(data_, o.data_); std::swap(capacity_, o.capacity_);
        // an owned buffer's data_ points into its own vector: re-anchor both
        if (!owned_.empty())   data_   = owned_.data();
        if (!o.owned_.empty()) o.data_ = o.owned_.data();
    }

    bool   owns() const           { return data_ == nullptr || !owned_.empty(); }  // externally backed <=> false
    Elt*   data()                 { return data_; }
    const Elt* data() const       { return data_; }
    size_t footprint() const      { return static_cast<size_t>(major_dim()) * ld; }   // elements
    uint32_t major_dim() const    { return layout == Layout::RowMajor ? rows : cols; }
    uint32_t minor_dim() const    { return layout == Layout::RowMajor ? cols : rows; }
    // strides of the (i, j) -> offset map
    size_t row_stride() const     { return layout == Layout::RowMajor ? ld : 1; }
    size_t col_stride() const     { return layout == Layout::RowMajor ? 1 : ld; }
    size_t index(uint32_t i, uint32_t j) const { return i * row_stride() + j * col_stride(); }
    Elt&       at(uint32_t i, uint32_t j)       { return data_[index(i, j)]; }
    const Elt& at(uint32_t i, uint32_t j) const { return data_[index(i, j)]; }
    Elt&       checked_at(uint32_t i, uint32_t j) {
        ENGINE_CHECK(i < rows && j < cols); return at(i, j);
    }

    // Whole-footprint fills (padding included; padding is never read anyway).
    void fill(Elt v) { for (size_t i = 0; i < footprint(); ++i) data_[i] = v; }
    void zero()      { fill(Elt(0)); }

    // Entrywise equality of the logical rows x cols contents (padding ignored,
    // layouts may differ).
    bool same_entries(const DenseBuf& o) const {
        if (rows != o.rows || cols != o.cols) return false;
        for (uint32_t i = 0; i < rows; ++i)
            for (uint32_t j = 0; j < cols; ++j)
                if (at(i, j) != o.at(i, j)) return false;
        return true;
    }

private:
    std::vector<Elt> owned_;
    Elt*   data_     = nullptr;
    size_t capacity_ = 0;
};

using DenseBuf32 = DenseBuf<uint32_t>;
using DenseBuf64 = DenseBuf<uint64_t>;

} // namespace engine
