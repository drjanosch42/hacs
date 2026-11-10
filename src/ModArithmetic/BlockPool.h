// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// =====================================================================
// BlockPool.h -- content-addressed immutable pool of dense blocks
// (matrix-engine Phase 3, plan D3.2)
// =====================================================================
//
// A BlockPool<Elt> owns the dense data every BlockMat<Elt> refers to.  Blocks
// are IMMUTABLE once interned and CONTENT-ADDRESSED: interning a buffer whose
// logical entries (rows, cols, entrywise values -- layout is NOT part of the
// identity) equal an existing entry's returns that entry's id.  This is the
// "unique_blocks_" dedup of BlockFlatMatrix (audit 01 Sec. 9.5, "the
// substrate's real sparsity mechanism") done right:
//
//   * hash -> id with a FULL entrywise compare on every hash hit (defect #14:
//     the legacy pool compared hashes only, and hashed with std::hash<uint32_t>
//     for both element widths; here the mixing goes through std::hash<Elt>);
//   * REFCOUNTED (defect #15: a block overwritten in a BlockFlatMatrix leaked
//     for the lifetime of the matrix).  Containers (BlockMat, BlockBuilder)
//     acquire()/release(); an entry whose count drops to zero is freed and its
//     id recycled.  A freshly interned entry has count 0 and stays alive until
//     sweep() is called or a container releases it back to 0 -- so
//     `Id id = pool.intern(buf); builder.set(br, bc, BlockRef::dense(id));`
//     is the intended idiom and never dangles;
//   * hand-rolled open hash table (chained buckets in a vector), NO
//     std::unordered_* anywhere (02 Sec. 6.5(5)); bucket walk order is
//     deterministic (insertion order within a chain) and does not influence
//     any result -- an intern either finds an equal block or appends;
//   * the structure tag (Generic / UpperUnitri / ...) travels with the buffer;
//   * BORROWED entries (borrow()) wrap caller-owned memory without copying or
//     hashing: they are never deduplicated and the caller keeps the memory
//     alive for as long as any container references the id.  Used by the
//     elimination front-ends so that a dense a x k basis passed by view is
//     not copied into the pool;
//   * ADDRESS STABILITY: entries live in a std::deque, so a DenseView obtained
//     from the pool stays valid across later intern() / borrow() calls (until
//     the entry itself is freed);
//   * MEMOISED PHYSICAL RELAYOUT (plan D3 layout policy item 3): view(id,
//     Layout) hands out the buffer in the requested physical layout, building
//     the other-layout twin once (16 x 16-tiled kernels::transpose-style copy)
//     and keeping it with the entry, so n mismatched consumers cost one
//     relayout, not n.  The twin holds the same values by construction
//     (an exact permutation), so no layout decision can move a bit.
//
// Thread safety (matrix-engine Phase 4, Stage B): every public operation
// takes the pool's one mutex, so a pool may be SHARED by the concurrently
// running degrees of compute_cohomology() and by concurrently running Hecke
// pipelines (the consumer-level parallel axes (c) and (d)).  The lock covers
// the table and the entry bookkeeping only; the entry BUFFERS are immutable
// once interned and address-stable (deque), so a DenseView handed out by
// buf() / view() stays valid without the lock for as long as a container
// holds the id -- which is exactly the refcount contract above.  The lock is
// never held while a kernel runs, so it cannot serialise arithmetic; ids and
// dedup results are the same whichever thread interns first (an intern either
// finds the equal block or appends), and no result depends on ids.

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "DenseBuf.h"
#include "DenseView.h"
#include "EngineCheck.h"
#include "Kernels.h"

namespace engine {

template<class Elt>
class BlockPool {
public:
    using Id = uint32_t;
    static constexpr Id kNone = 0xFFFFFFFFu;

    struct Stats {
        uint64_t interns      = 0;   // intern() calls
        uint64_t hits         = 0;   // returned an existing id
        uint64_t hash_hits    = 0;   // hash equal, compared entrywise (incl. true hits)
        uint64_t collisions   = 0;   // hash equal, content different
        uint64_t relayouts    = 0;   // twins built
        uint64_t frees        = 0;   // entries released to zero
    };

    explicit BlockPool(uint32_t initial_buckets = 1024)
        : buckets_(initial_buckets == 0 ? 1 : initial_buckets, kNone) {}

    BlockPool(const BlockPool&) = delete;
    BlockPool& operator=(const BlockPool&) = delete;

    // ---- interning -------------------------------------------------------
    // Content-addressed.  The buffer is moved in (or copied for the const
    // overload).  Returns the id of the equal entry if one exists (the
    // argument is then discarded), else a new id.  Refcount unchanged (0 for a
    // new entry): containers acquire.
    Id intern(DenseBuf<Elt>&& b) {
        std::lock_guard<std::mutex> lk(m_);
        ++stats_.interns;
        const uint64_t h = content_hash(cview(b));
        for (Id i = buckets_[h % buckets_.size()]; i != kNone; i = entries_[i].next) {
            Entry& e = entries_[i];
            if (e.hash != h) continue;
            ++stats_.hash_hits;
            if (e.buf.rows == b.rows && e.buf.cols == b.cols && e.buf.same_entries(b)) {
                ++stats_.hits;
                return i;
            }
            ++stats_.collisions;
        }
        if (live_ + 1 > buckets_.size() * 2) rehash(buckets_.size() * 4);
        const Id id = alloc_entry();
        Entry& e = entries_[id];
        e.buf = std::move(b);
        e.hash = h;
        e.borrowed = false;
        link(id);
        return id;
    }
    Id intern(const DenseBuf<Elt>& b) { return intern(DenseBuf<Elt>(b)); }

    // Copies the logical window `v` into a fresh buffer of the given physical
    // layout and interns it.  (structure tag: Generic.)
    Id intern_view(DenseView<const Elt> v, Layout layout = Layout::RowMajor) {
        DenseBuf<Elt> b(v.rows(), v.cols(), layout);
        if (!v.empty()) kernels::copy<Elt>(v, engine::view(b));
        return intern(std::move(b));
    }

    // Borrow caller-owned memory: no copy, no hash, never deduplicated.  The
    // view must be a plain (row- or column-contiguous) window; a transposed
    // view of a column-major buffer is fine (it is row-contiguous).  The
    // caller guarantees the memory outlives every container holding the id.
    // The data is never written through the pool.
    Id borrow(DenseView<const Elt> v) {
        const auto s = v.strides();
        Layout layout;
        uint32_t ld;
        if (s.cs == 1)       { layout = Layout::RowMajor; ld = static_cast<uint32_t>(s.rs == 0 ? v.cols() : s.rs); }
        else if (s.rs == 1)  { layout = Layout::ColMajor; ld = static_cast<uint32_t>(s.cs == 0 ? v.rows() : s.cs); }
        else { ENGINE_CHECK_MSG(false, "BlockPool::borrow: window is neither row- nor column-contiguous"); return kNone; }
        if (layout == Layout::RowMajor && ld < v.cols()) ld = v.cols();
        if (layout == Layout::ColMajor && ld < v.rows()) ld = v.rows();
        const size_t footprint = static_cast<size_t>(layout == Layout::RowMajor ? v.rows() : v.cols()) * ld;
        std::lock_guard<std::mutex> lk(m_);
        const Id id = alloc_entry();
        Entry& e = entries_[id];
        e.buf = DenseBuf<Elt>(const_cast<Elt*>(s.base), footprint == 0 ? 1 : footprint,
                              v.rows(), v.cols(), layout, ld);
        e.hash = 0;
        e.borrowed = true;
        // not linked into the hash table: never a dedup candidate
        return id;
    }

    // ---- refcounting -----------------------------------------------------
    void acquire(Id id) { std::lock_guard<std::mutex> lk(m_); ENGINE_CHECK(live_unlocked(id)); ++entries_[id].refcount; }
    void release(Id id) {
        std::lock_guard<std::mutex> lk(m_);
        ENGINE_CHECK(live_unlocked(id) && entries_[id].refcount > 0);
        if (--entries_[id].refcount == 0) free_entry(id);
    }
    // Frees `id` if it is live and unreferenced (e.g. an interned buffer none
    // of whose windows survived structural-zero detection).
    void sweep_if_unreferenced(Id id) {
        std::lock_guard<std::mutex> lk(m_);
        if (live_unlocked(id) && entries_[id].refcount == 0) free_entry(id);
    }
    // Frees every live entry with refcount 0 (interned but never adopted).
    // Validation-only: under concurrent use another thread's freshly interned,
    // not-yet-adopted block would be swept.
    void sweep() {
        std::lock_guard<std::mutex> lk(m_);
        for (Id i = 0; i < entries_.size(); ++i)
            if (entries_[i].live && entries_[i].refcount == 0) free_entry(i);
    }

    // ---- access ----------------------------------------------------------
    bool is_live(Id id) const { std::lock_guard<std::mutex> lk(m_); return live_unlocked(id); }
    const DenseBuf<Elt>& buf(Id id) const {
        std::lock_guard<std::mutex> lk(m_);
        ENGINE_CHECK(live_unlocked(id));
        return entries_[id].buf;
    }
    DenseView<const Elt> view(Id id) const { return cview(buf(id)); }
    // Memoised physical relayout: the same values in the requested layout.
    DenseView<const Elt> view(Id id, Layout want) {
        std::lock_guard<std::mutex> lk(m_);
        ENGINE_CHECK(live_unlocked(id));
        Entry& e = entries_[id];
        if (e.buf.layout == want) return cview(e.buf);
        if (!e.twin) {
            e.twin.reset(new DenseBuf<Elt>(e.buf.rows, e.buf.cols, want, 0, e.buf.invariant, e.buf.structure));
            if (e.buf.rows != 0 && e.buf.cols != 0) kernels::copy<Elt>(cview(e.buf), engine::view(*e.twin));
            ++stats_.relayouts;
        }
        return cview(*e.twin);
    }
    uint32_t refcount(Id id) const { std::lock_guard<std::mutex> lk(m_); ENGINE_CHECK(live_unlocked(id)); return entries_[id].refcount; }
    bool is_borrowed(Id id) const { std::lock_guard<std::mutex> lk(m_); ENGINE_CHECK(live_unlocked(id)); return entries_[id].borrowed; }

    size_t num_live() const { std::lock_guard<std::mutex> lk(m_); return live_; }
    Stats stats() const { std::lock_guard<std::mutex> lk(m_); return stats_; }

    // FNV-1a-64 over rows, cols and the logical entries in row-major order,
    // each element mixed through std::hash<Elt>.  Layout-independent.
    static uint64_t content_hash(DenseView<const Elt> v) {
        uint64_t h = 1469598103934665603ull;
        auto mix = [&h](uint64_t x) { h ^= x; h *= 1099511628211ull; };
        mix(v.rows());
        mix(v.cols());
        const auto s = v.strides();
        if constexpr (sizeof(Elt) == 16) {
            // (slot-11 WP-18a, ENGINE_API.md Sec. 13.12.3) u128 entries: both
            // 64-bit halves, low limb first (std::hash would drop the high one).
            std::hash<uint64_t> h64;
            for (uint32_t i = 0; i < v.rows(); ++i) {
                const Elt* row = s.base + static_cast<size_t>(i) * s.rs;
                for (uint32_t j = 0; j < v.cols(); ++j) {
                    const Elt x = row[j * s.cs];
                    mix(static_cast<uint64_t>(h64(static_cast<uint64_t>(x))));
                    mix(static_cast<uint64_t>(h64(static_cast<uint64_t>(x >> 64))));
                }
            }
            return h;
        }
        std::hash<Elt> hasher;
        for (uint32_t i = 0; i < v.rows(); ++i) {
            const Elt* row = s.base + static_cast<size_t>(i) * s.rs;
            for (uint32_t j = 0; j < v.cols(); ++j) mix(static_cast<uint64_t>(hasher(row[j * s.cs])));
        }
        return h;
    }

private:
    struct Entry {
        DenseBuf<Elt> buf;
        std::unique_ptr<DenseBuf<Elt>> twin;
        uint64_t hash = 0;
        uint32_t refcount = 0;
        Id       next = kNone;    // hash chain
        bool     borrowed = false;
        bool     live = false;
    };

    bool live_unlocked(Id id) const { return id < entries_.size() && entries_[id].live; }

    Id alloc_entry() {
        Id id;
        if (!free_.empty()) { id = free_.back(); free_.pop_back(); }
        else { id = static_cast<Id>(entries_.size()); entries_.emplace_back(); }
        Entry& e = entries_[id];
        e.live = true;
        e.refcount = 0;
        e.next = kNone;
        e.twin.reset();
        ++live_;
        return id;
    }
    void link(Id id) {
        const size_t b = entries_[id].hash % buckets_.size();
        entries_[id].next = buckets_[b];
        buckets_[b] = id;
    }
    void unlink(Id id) {
        const size_t b = entries_[id].hash % buckets_.size();
        Id* p = &buckets_[b];
        while (*p != kNone && *p != id) p = &entries_[*p].next;
        if (*p == id) *p = entries_[id].next;
        entries_[id].next = kNone;
    }
    void rehash(size_t nb) {
        buckets_.assign(nb, kNone);
        // Re-link in ascending id order so chain order is a pure function of
        // the live set.
        for (Id i = 0; i < entries_.size(); ++i) {
            Entry& e = entries_[i];
            if (!e.live || e.borrowed) continue;
            const size_t b = e.hash % nb;
            e.next = buckets_[b];
            buckets_[b] = i;
        }
    }
    void free_entry(Id id) {
        Entry& e = entries_[id];
        if (!e.borrowed) unlink(id);
        e.buf = DenseBuf<Elt>();
        e.twin.reset();
        e.live = false;
        e.refcount = 0;
        e.borrowed = false;
        --live_;
        ++stats_.frees;
        free_.push_back(id);
    }

    mutable std::mutex m_;          // one lock for the table + bookkeeping (never held across a kernel)
    std::deque<Entry>  entries_;    // deque: entry addresses are stable, so views into pool buffers survive growth
    std::vector<Id>    buckets_;
    std::vector<Id>    free_;
    size_t             live_ = 0;
    Stats              stats_;
};

} // namespace engine
