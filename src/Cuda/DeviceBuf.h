// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// =====================================================================
// DeviceBuf.h -- device mirror of a DenseBuf + residency tracking (Phase 5)
// =====================================================================
//
// DeviceBuf<Elt> carries EXACTLY DenseBuf's metadata (rows, cols, ld, layout,
// invariant, structure -- same meaning, same padding rule: padding is never
// read, written or compared) plus a device allocation of `footprint()`
// elements from the CudaCtx arena. It optionally mirrors ONE host DenseBuf
// (non-owning: the host buffer must outlive the mirror) and tracks which side
// is current with two version counters:
//
//   host_version  > device_version  ->  Residency::Host    (upload needed)
//   device_version > host_version   ->  Residency::Device  (download needed)
//   equal (and > 0)                 ->  Residency::Both
//   both 0                          ->  Residency::None    (nothing written yet)
//
// The tape bumps the versions: mark_host_written() after a CPU node wrote
// the host buffer, mark_device_written() after a GPU node wrote the device
// buffer; upload()/download() copy only when the other side is stale (or
// `force`). Transfers are BYTE copies of the whole footprint (padding
// included), staged through the pinned pool; a download waits for the data.
//
// DeviceView<Elt> is the kernels' operand: base pointer + RESOLVED strides
// (element (i, j) at base[i * rs + j * cs], the same contract as
// DenseView::strides()) + logical shape + the tags. It is a POD handed to the
// .cu by value; sub()/transposed_view()/row()/col() are the same free
// operations as on DenseView.

#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <type_traits>
#include <utility>

#include "CudaCtx.h"
#include "DenseBuf.h"
#include "DenseView.h"
#include "EngineCheck.h"

namespace engine {
namespace cuda {

enum class Residency : uint8_t { None, Host, Device, Both };

template<class Elt>
struct DeviceView {
    using Value = std::remove_const_t<Elt>;
    Elt*      base = nullptr;
    size_t    rs = 0, cs = 0;
    uint32_t  rows = 0, cols = 0;
    Invariant invariant = Invariant::Canonical;
    Structure structure = Structure::Generic;

    DeviceView() = default;
    DeviceView(Elt* b, size_t rs_, size_t cs_, uint32_t r, uint32_t c,
               Invariant inv = Invariant::Canonical, Structure st = Structure::Generic)
        : base(b), rs(rs_), cs(cs_), rows(r), cols(c), invariant(inv), structure(st) {}
    // mutable -> const
    template<class E2, class = std::enable_if_t<std::is_const<Elt>::value &&
                                                std::is_same<Value, std::remove_const_t<E2>>::value &&
                                                !std::is_const<E2>::value>>
    DeviceView(const DeviceView<E2>& o)
        : base(o.base), rs(o.rs), cs(o.cs), rows(o.rows), cols(o.cols),
          invariant(o.invariant), structure(o.structure) {}

    bool empty() const { return rows == 0 || cols == 0; }
    bool row_contiguous() const { return cs == 1; }
    bool col_contiguous() const { return rs == 1; }
    Elt* ptr(uint32_t i, uint32_t j) const { return base + i * rs + j * cs; }

    DeviceView sub(uint32_t r0, uint32_t nr, uint32_t c0, uint32_t nc) const {
        ENGINE_CHECK_MSG(static_cast<uint64_t>(r0) + nr <= rows &&
                         static_cast<uint64_t>(c0) + nc <= cols, "DeviceView::sub: window exceeds view");
        DeviceView v(*this);
        v.base = base + r0 * rs + c0 * cs; v.rows = nr; v.cols = nc;
        return v;
    }
    DeviceView row(uint32_t i) const { return sub(i, 1, 0, cols); }
    DeviceView col(uint32_t j) const { return sub(0, rows, j, 1); }
    DeviceView rows_range(uint32_t r0, uint32_t nr) const { return sub(r0, nr, 0, cols); }
    DeviceView cols_range(uint32_t c0, uint32_t nc) const { return sub(0, rows, c0, nc); }
    DeviceView transposed_view() const {
        DeviceView v(*this);
        v.rs = cs; v.cs = rs; v.rows = cols; v.cols = rows;
        return v;
    }
};

// Do two windows share an element? Exact when both windows live on the same
// stride grid with a unit minor stride (the DenseBuf case: rs == ld, cs == 1
// or the transpose), i.e. exactly the situation of DenseView windows of one
// buffer -- disjoint windows of the same buffer are then allowed, as on the
// CPU (Schur update C -= L * U in place). Other stride combinations fall back
// to a conservative address-range test.
template<class E1, class E2>
bool windows_overlap(const DeviceView<E1>& a, const DeviceView<E2>& b) {
    if (a.empty() || b.empty()) return false;
    const char* ab = reinterpret_cast<const char*>(a.base);
    const char* bb = reinterpret_cast<const char*>(b.base);
    if (a.rs == b.rs && a.cs == b.cs && (a.cs == 1 || a.rs == 1)) {
        // grid coordinates: (major, minor) with unit minor stride
        const bool row_major = (a.cs == 1);
        const ptrdiff_t s = static_cast<ptrdiff_t>(row_major ? a.rs : a.cs);
        const ptrdiff_t ra = row_major ? a.rows : a.cols, ca = row_major ? a.cols : a.rows;
        const ptrdiff_t rb = row_major ? b.rows : b.cols, cb = row_major ? b.cols : b.rows;
        const ptrdiff_t delta = (bb - ab) / static_cast<ptrdiff_t>(sizeof(typename DeviceView<E1>::Value));
        if (s <= 0) return true;
        ptrdiff_t di = delta / s, dj = delta - di * s;
        if (dj < 0) { dj += s; --di; }
        auto hits = [&](ptrdiff_t Di, ptrdiff_t Dj) {
            return (Di < ra && Di + rb > 0) && (Dj < ca && Dj + cb > 0);
        };
        // the true minor offset is dj or dj - s (b left of a); both candidates
        // are tested -- the wrong one cannot hit because windows fit in a row
        return hits(di, dj) || hits(di + 1, dj - s);
    }
    const char* a1 = reinterpret_cast<const char*>(a.ptr(a.rows - 1, a.cols - 1) + 1);
    const char* b1 = reinterpret_cast<const char*>(b.ptr(b.rows - 1, b.cols - 1) + 1);
    return !(a1 <= bb || b1 <= ab);
}

template<class Elt>
class DeviceBuf {
public:
    uint32_t  rows      = 0;
    uint32_t  cols      = 0;
    uint32_t  ld        = 0;
    Layout    layout    = Layout::RowMajor;
    Invariant invariant = Invariant::Canonical;
    Structure structure = Structure::Generic;

    DeviceBuf() = default;

    // Device-only buffer (no host twin), arena-allocated, NOT zeroed.
    DeviceBuf(uint32_t rows_, uint32_t cols_, Layout layout_ = Layout::RowMajor,
              uint32_t ld_ = 0, Invariant inv = Invariant::Canonical,
              Structure st = Structure::Generic)
        : rows(rows_), cols(cols_), ld(ld_), layout(layout_), invariant(inv), structure(st) {
        if (ld == 0) ld = minor_dim();
        ENGINE_CHECK_MSG(ld >= minor_dim(), "DeviceBuf: leading dimension too small");
        allocate();
    }

    // Mirror of a host buffer: same metadata; the host buffer is referenced,
    // not copied, and starts as the current side (Residency::Host).
    explicit DeviceBuf(DenseBuf<Elt>& host)
        : rows(host.rows), cols(host.cols), ld(host.ld), layout(host.layout),
          invariant(host.invariant), structure(host.structure), host_(&host) {
        allocate();
        host_version_ = 1;
    }

    ~DeviceBuf() { release(); }
    DeviceBuf(const DeviceBuf&) = delete;
    DeviceBuf& operator=(const DeviceBuf&) = delete;
    DeviceBuf(DeviceBuf&& o) noexcept { swap(o); }
    DeviceBuf& operator=(DeviceBuf&& o) noexcept {
        if (this != &o) { DeviceBuf tmp; tmp.swap(*this); swap(o); }
        return *this;
    }
    void swap(DeviceBuf& o) noexcept {
        std::swap(rows, o.rows); std::swap(cols, o.cols); std::swap(ld, o.ld);
        std::swap(layout, o.layout); std::swap(invariant, o.invariant);
        std::swap(structure, o.structure);
        std::swap(dev_, o.dev_); std::swap(capacity_, o.capacity_);
        std::swap(host_, o.host_);
        std::swap(host_version_, o.host_version_); std::swap(device_version_, o.device_version_);
    }

    Elt*       dptr()             { return dev_; }
    const Elt* dptr() const       { return dev_; }
    size_t footprint() const      { return static_cast<size_t>(major_dim()) * ld; }
    size_t bytes() const          { return footprint() * sizeof(Elt); }
    uint32_t major_dim() const    { return layout == Layout::RowMajor ? rows : cols; }
    uint32_t minor_dim() const    { return layout == Layout::RowMajor ? cols : rows; }
    size_t row_stride() const     { return layout == Layout::RowMajor ? ld : 1; }
    size_t col_stride() const     { return layout == Layout::RowMajor ? 1 : ld; }
    DenseBuf<Elt>* host() const   { return host_; }

    // ---- residency ----------------------------------------------------------
    uint64_t host_version() const   { return host_version_; }
    uint64_t device_version() const { return device_version_; }
    Residency residency() const {
        if (host_version_ == 0 && device_version_ == 0) return Residency::None;
        if (host_version_ > device_version_) return Residency::Host;
        if (device_version_ > host_version_) return Residency::Device;
        return Residency::Both;
    }
    void mark_host_written()   { host_version_ = std::max(host_version_, device_version_) + 1; }
    void mark_device_written() { device_version_ = std::max(host_version_, device_version_) + 1; }
    // Both sides hold the same bytes (the caller copied or verified them
    // itself, e.g. the ENGINE_VERIFY_DEVICE re-execution): Residency::Both.
    void mark_synced()         { host_version_ = device_version_ = std::max(host_version_, device_version_); }
    // Re-point the mirror at another host buffer of the SAME footprint (a
    // tape replay binds new views; the device allocation is kept so captured
    // graphs stay valid). The host side becomes the current one
    // (`host_newer`, the fresh-mirror state) or, for a program temp, None.
    void rebind_host(DenseBuf<Elt>& host, bool host_newer = true) {
        ENGINE_CHECK_MSG(host.rows == rows && host.cols == cols && host.ld == ld && host.layout == layout,
                         "DeviceBuf::rebind_host: host metadata differs from the mirror");
        host_ = &host;
        host_version_ = host_newer ? 1 : 0;
        device_version_ = 0;
    }

    // Host -> device when the host side is newer (or force). Byte copy of
    // the footprint, staged through the pinned pool. Returns true if copied.
    bool upload(Stream s, bool force = false) {
        ENGINE_CHECK_MSG(host_ != nullptr, "DeviceBuf::upload: no host twin");
        if (!force && host_version_ <= device_version_) return false;
        ENGINE_CHECK_MSG(host_->footprint() == footprint(), "DeviceBuf::upload: host footprint changed");
        CudaCtx::get().upload(host_->data(), dev_, bytes(), s);
        device_version_ = host_version_;
        return true;
    }
    // Device -> host when the device side is newer (or force). Waits for the
    // bytes to land. Returns true if copied.
    bool download(Stream s, bool force = false) {
        ENGINE_CHECK_MSG(host_ != nullptr, "DeviceBuf::download: no host twin");
        if (!force && device_version_ <= host_version_) return false;
        ENGINE_CHECK_MSG(host_->footprint() == footprint(), "DeviceBuf::download: host footprint changed");
        CudaCtx::get().download(host_->data(), dev_, bytes(), s);
        host_version_ = device_version_;
        return true;
    }
    // Raw copies for buffers without a host twin.
    void upload_from(const Elt* host_data, size_t count, Stream s) {
        ENGINE_CHECK_MSG(count == footprint(), "DeviceBuf::upload_from: count != footprint");
        CudaCtx::get().upload(host_data, dev_, bytes(), s);
        mark_device_written();
    }
    void download_to(Elt* host_data, size_t count, Stream s) const {
        ENGINE_CHECK_MSG(count == footprint(), "DeviceBuf::download_to: count != footprint");
        CudaCtx::get().download(host_data, dev_, bytes(), s);
    }

    // ---- views ---------------------------------------------------------------
    DeviceView<Elt> view() {
        return DeviceView<Elt>(dev_, row_stride(), col_stride(), rows, cols, invariant, structure);
    }
    DeviceView<const Elt> view() const {
        return DeviceView<const Elt>(dev_, row_stride(), col_stride(), rows, cols, invariant, structure);
    }
    DeviceView<const Elt> cview() const { return view(); }

private:
    void allocate() {
        capacity_ = footprint();
        dev_ = capacity_ ? static_cast<Elt*>(CudaCtx::get().dalloc(capacity_ * sizeof(Elt))) : nullptr;
    }
    void release() {
        if (dev_) CudaCtx::get().dfree(dev_, capacity_ * sizeof(Elt));
        dev_ = nullptr; capacity_ = 0;
    }
    Elt*   dev_ = nullptr;
    size_t capacity_ = 0;
    DenseBuf<Elt>* host_ = nullptr;
    uint64_t host_version_ = 0, device_version_ = 0;
};

} // namespace cuda
} // namespace engine
