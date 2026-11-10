// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// =====================================================================
// CudaCtx.cpp -- runtime context, pinned pool, arena, transfers, graphs
// =====================================================================
// The only C++ TU of the layer that talks to the CUDA runtime API; the
// kernels live in cuda_kernels.cu. See CudaCtx.h.

#include "CudaCtx.h"
#include "RetiredDeviceLevers.h"   // (S25 LEVERS-25) ENGINE_GPU_LEAN

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <sstream>
#include <thread>
#include <memory>
#include <vector>

#include "EngineCheck.h"

namespace engine {
namespace cuda {

namespace {

std::string err_str(cudaError_t e, const char* what, const char* file, int line) {
    std::ostringstream os;
    os << "CUDA error in " << what << " (" << file << ":" << line << "): "
       << cudaGetErrorName(e) << " -- " << cudaGetErrorString(e);
    return os.str();
}

#define CUDA_CALL(call)                                                                  \
    do {                                                                                 \
        const cudaError_t e_ = (call);                                                   \
        if (e_ != cudaSuccess) throw ::engine::cuda::CudaError(err_str(e_, #call, __FILE__, __LINE__)); \
    } while (0)

cudaStream_t S(Stream s) { return static_cast<cudaStream_t>(s.h); }
cudaEvent_t  E(Event e)  { return static_cast<cudaEvent_t>(e.h); }

size_t env_mb(const char* name, size_t dflt_mb) {
    const char* e = std::getenv(name);
    if (!e || !*e) return dflt_mb;
    char* end = nullptr;
    const unsigned long long v = std::strtoull(e, &end, 10);
    if (end == e || *end != '\0') return dflt_mb;
    return static_cast<size_t>(v);
}

int env_device_index() {
    const char* e = std::getenv("ENGINE_CUDA_DEVICE");
    if (!e || !*e) return 0;
    return std::atoi(e);
}

struct Availability {
    bool ok = false;
    std::string reason;
    int index = 0;
};

const Availability& probe() {
    static const Availability a = [] {
        Availability r;
        const char* dis = std::getenv("ENGINE_CUDA_DISABLE");
        if (dis && *dis && std::strcmp(dis, "0") != 0) {
            r.reason = "ENGINE_CUDA_DISABLE is set";
            return r;
        }
        int n = 0;
        const cudaError_t e = cudaGetDeviceCount(&n);
        if (e != cudaSuccess) {
            r.reason = std::string("cudaGetDeviceCount: ") + cudaGetErrorString(e);
            (void)cudaGetLastError();   // clear the sticky error
            return r;
        }
        if (n <= 0) { r.reason = "no CUDA device"; return r; }
        r.index = env_device_index();
        if (r.index < 0 || r.index >= n) {
            r.reason = "ENGINE_CUDA_DEVICE out of range";
            return r;
        }
        const cudaError_t e2 = cudaSetDevice(r.index);
        if (e2 != cudaSuccess) {
            r.reason = std::string("cudaSetDevice: ") + cudaGetErrorString(e2);
            (void)cudaGetLastError();
            return r;
        }
        // Touch the context once so that a broken driver fails HERE, not
        // inside a kernel launch.
        const cudaError_t e3 = cudaFree(nullptr);
        if (e3 != cudaSuccess) {
            r.reason = std::string("context creation: ") + cudaGetErrorString(e3);
            (void)cudaGetLastError();
            return r;
        }
        r.ok = true;
        return r;
    }();
    return a;
}

thread_local int t_capture_depth = 0;

} // namespace

bool available() { return probe().ok; }
const char* unavailable_reason() { return probe().ok ? "" : probe().reason.c_str(); }

Device requested_device() {
    static const Device d = [] {
        const char* e = std::getenv("ENGINE_DEVICE");
        if (e && std::strcmp(e, "gpu") == 0) return Device::Gpu;
        if (e && std::strcmp(e, "hybrid") == 0) return Device::Hybrid;
        return Device::Cpu;
    }();
    return d;
}
bool device_enabled() { return available() && requested_device() != Device::Cpu; }

DeviceInfo device_info() {
    if (!available()) throw CudaError(std::string("device_info: ") + unavailable_reason());
    DeviceInfo d;
    d.index = probe().index;
    cudaDeviceProp prop{};
    CUDA_CALL(cudaGetDeviceProperties(&prop, d.index));
    d.name = prop.name;
    d.cc_major = prop.major; d.cc_minor = prop.minor;
    d.sms = prop.multiProcessorCount;
    d.total_mem = prop.totalGlobalMem;
    size_t fr = 0, tot = 0;
    CUDA_CALL(cudaMemGetInfo(&fr, &tot));
    d.free_mem = fr;
    CUDA_CALL(cudaDriverGetVersion(&d.driver_version));
    CUDA_CALL(cudaRuntimeGetVersion(&d.runtime_version));
    return d;
}

bool capturing_now() { return t_capture_depth > 0; }
// (slot-14 WP-3) Captures in progress on ANY thread: the arena's soft trim
// (cudaFree synchronises the device) is skipped while one is open.
namespace { std::atomic<int> g_active_captures{0}; }

// ---------------------------------------------------------------------------
// CopyPool -- a few persistent threads for the staging memcpys.
//
// WSL2 measurement (this box, 64 MB): a single thread reads cudaHostAlloc'ed
// memory at 5.4 GB/s and writes it at 13 GB/s, while 4 threads reach 18 /
// 25 GB/s (8 threads 24 / 28). The DMA itself runs at 55 GB/s, so without
// this the staged path is bounded by the host memcpy, and D2H in particular
// fell to 5.1 GB/s -- below a plain pageable cudaMemcpy. One job = one
// chunk, sliced across the workers; the caller waits for the slices.
// ENGINE_CUDA_COPY_THREADS (default 4; 0 or 1 = inline memcpy).
// ---------------------------------------------------------------------------
class CopyPool {
public:
    explicit CopyPool(int n) {
        for (int i = 0; i < n; ++i) workers_.emplace_back([this] { loop(); });
    }
    ~CopyPool() {
        { std::lock_guard<std::mutex> lk(mu_); stop_ = true; }
        cv_.notify_all();
        for (auto& w : workers_) w.join();
    }
    int size() const { return static_cast<int>(workers_.size()); }
    void copy(void* dst, const void* src, size_t bytes) {
        const size_t T = workers_.size();
        if (T == 0 || bytes < (1u << 20)) { std::memcpy(dst, src, bytes); return; }
        const size_t per = (bytes + T - 1) / T;
        std::vector<Slice> slices;
        for (size_t t = 0; t < T; ++t) {
            const size_t o = t * per;
            if (o >= bytes) break;
            slices.push_back(Slice{static_cast<char*>(dst) + o, static_cast<const char*>(src) + o, std::min(per, bytes - o)});
        }
        {
            std::lock_guard<std::mutex> lk(mu_);
            pending_ = slices.size();
            for (auto& sl : slices) queue_.push_back(sl);
        }
        cv_.notify_all();
        std::unique_lock<std::mutex> lk(mu_);
        done_cv_.wait(lk, [this] { return pending_ == 0; });
    }
private:
    struct Slice { char* dst; const char* src; size_t n; };
    void loop() {
        for (;;) {
            Slice sl;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [this] { return stop_ || !queue_.empty(); });
                if (stop_ && queue_.empty()) return;
                sl = queue_.front(); queue_.pop_front();
            }
            std::memcpy(sl.dst, sl.src, sl.n);
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (--pending_ == 0) done_cv_.notify_all();
            }
        }
    }
    std::vector<std::thread> workers_;
    std::deque<Slice> queue_;
    std::mutex mu_;
    std::condition_variable cv_, done_cv_;
    size_t pending_ = 0;
    bool stop_ = false;
};

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------
struct CudaCtx::Impl {
    static constexpr int kStreams = 3;
    cudaStream_t streams[kStreams] = {};

    // pinned pool
    struct Chunk { void* p = nullptr; cudaEvent_t ev = nullptr; bool busy = false; };
    size_t chunk_bytes = 8u << 20;
    size_t cap_bytes = 128u << 20;
    std::vector<Chunk> chunks;
    size_t next_victim = 0;
    PinnedPoolStats pstats;
    bool pool_exhausted_alloc = false;   // cudaHostAlloc failed once: stop growing

    // arena.  Slabs are keyed by base address so a block finds its slab by
    // upper_bound; each carries its live bytes (slot-14 WP-3), so a slab with
    // none can be returned to the driver.  `bump` is the key of the slab new
    // blocks are cut from (0: none).
    struct Slab { void* base; size_t size; size_t used; size_t live; };
    std::map<uintptr_t, Slab> slabs;
    uintptr_t bump = 0;
    std::map<size_t, std::vector<void*>> free_lists;   // exact size -> blocks
    // soft-trim trigger (slot-14 WP-3): ENGINE_CUDA_ARENA_SLACK_MB, default a
    // quarter of the device; a tuning knob, never changes a value.
    size_t idle_slack = size_t(1) << 30;
    size_t idle_after_trim = 0;
    ArenaStats astats;
    // (slot-23 P7) ENGINE_TRACE_VRAM: the live blocks by address (trace only), and at every new
    // live peak (in steps of 64 MB) the size histogram of the live blocks. Diagnostic only.
    bool trace_vram = false;
    std::map<void*, std::pair<size_t, std::string>> live_blocks;   // tag copied: a tag pointer may die with its caller
    size_t trace_peak_snap = 0;
    std::string trace_peak_hist;
    void trace_alloc(void* p, size_t bytes) {
        if (!trace_vram) return;
        live_blocks[p] = {bytes, trace_tag()};
        if (astats.live_bytes >= trace_peak_snap + (size_t(64) << 20)) {
            trace_peak_snap = astats.live_bytes;
            std::map<std::pair<std::string, size_t>, std::pair<size_t, size_t>> h;   // (tag, size) -> (count, bytes)
            for (const auto& kv : live_blocks) { auto& e = h[{kv.second.second, kv.second.first}]; ++e.first; e.second += kv.second.first; }
            std::vector<std::pair<size_t, std::pair<std::string, size_t>>> v;          // (bytes, key)
            for (const auto& kv : h) v.emplace_back(kv.second.second, kv.first);
            std::sort(v.rbegin(), v.rend());
            std::ostringstream os;
            os << "live_mb=" << (astats.live_bytes >> 20) << " reserved_mb=" << (astats.reserved_bytes >> 20) << " top:";
            for (size_t i = 0; i < v.size() && i < 12; ++i)
                os << ' ' << v[i].second.first << ':' << h[v[i].second].first << 'x' << (v[i].second.second >> 10) << "KB";
            trace_peak_hist = os.str();
        }
    }
    void trace_free(void* p) { if (trace_vram) live_blocks.erase(p); }
    static constexpr size_t kSlabBytes = 64u << 20;
    static constexpr size_t kAlign = 256;

    TransferStats tstats;
    std::mutex mu;
    std::unique_ptr<CopyPool> copiers;

    // ---- pinned pool ----
    // Returns an index of a chunk that is free to be written by the host,
    // kNone when no pinned memory can be obtained at all, kMustDrain when
    // every recyclable chunk is one the caller itself still has in flight
    // (`pending[i] != 0`): the caller must drain before asking again.
    static constexpr size_t kNone = static_cast<size_t>(-1);
    static constexpr size_t kMustDrain = static_cast<size_t>(-2);
    size_t acquire_chunk(const std::vector<char>* pending = nullptr) {
        ++pstats.acquisitions;
        for (size_t i = 0; i < chunks.size(); ++i) if (!chunks[i].busy) return i;
        if (!pool_exhausted_alloc && (chunks.size() + 1) * chunk_bytes <= cap_bytes) {
            Chunk c;
            const cudaError_t e = cudaHostAlloc(&c.p, chunk_bytes, cudaHostAllocDefault);
            if (e == cudaSuccess) {
                CUDA_CALL(cudaEventCreateWithFlags(&c.ev, cudaEventDisableTiming));
                chunks.push_back(c);
                pstats.allocated_bytes += chunk_bytes;
                pstats.peak_bytes = std::max(pstats.peak_bytes, pstats.allocated_bytes);
                return chunks.size() - 1;
            }
            (void)cudaGetLastError();
            pool_exhausted_alloc = true;     // WSL2 pinned cap hit: live with what we have
        }
        if (chunks.empty()) return kNone;   // no pinned memory at all
        // recycle: round-robin over the in-flight chunks, skipping the caller's own
        for (size_t t = 0; t < chunks.size(); ++t) {
            const size_t v = (next_victim + t) % chunks.size();
            if (pending && (*pending)[v]) continue;
            next_victim = v + 1;
            ++pstats.waits;
            CUDA_CALL(cudaEventSynchronize(chunks[v].ev));
            chunks[v].busy = false;
            return v;
        }
        return kMustDrain;
    }
    Slab& slab_of(const void* p) {
        const uintptr_t a = reinterpret_cast<uintptr_t>(p);
        auto it = slabs.upper_bound(a);
        ENGINE_CHECK_MSG(it != slabs.begin(), "CudaCtx arena: pointer below every slab");
        --it;
        ENGINE_CHECK_MSG(a < it->first + it->second.size, "CudaCtx arena: pointer outside every slab");
        return it->second;
    }
    // Frees every slab with no live block (the caller holds `mu`).
    void trim_arena(ReleaseStats& rs) {
        std::vector<std::pair<uintptr_t, size_t>> gone;   // [base, base + size), ascending
        for (const auto& kv : slabs) {
            if (kv.second.live != 0) { ++rs.slabs_kept; continue; }
            gone.emplace_back(kv.first, kv.second.size);
        }
        if (gone.empty()) return;
        auto in_gone = [&](void* p) {
            const uintptr_t a = reinterpret_cast<uintptr_t>(p);
            auto g = std::upper_bound(gone.begin(), gone.end(), std::make_pair(a, ~size_t(0)));
            if (g == gone.begin()) return false;
            --g;
            return a < g->first + g->second;
        };
        for (auto fl = free_lists.begin(); fl != free_lists.end();) {
            auto& v = fl->second;
            v.erase(std::remove_if(v.begin(), v.end(), in_gone), v.end());
            if (v.empty()) fl = free_lists.erase(fl); else ++fl;
        }
        for (const auto& g : gone) {
            auto it = slabs.find(g.first);
            CUDA_CALL(cudaFree(it->second.base));
            astats.reserved_bytes -= std::min(g.second, astats.reserved_bytes);
            ++astats.cuda_frees;
            rs.arena_bytes_freed += g.second;
            ++rs.slabs_freed;
            if (bump == g.first) bump = 0;
            slabs.erase(it);
        }
        // (slot-14 WP-16, audit M-7) the bump slab went: continue in the surviving slab with
        // the largest free tail rather than cutting a new slab while tail space is idle.
        if (bump == 0) {
            size_t best = 0;
            for (const auto& kv : slabs) {
                const size_t tail = kv.second.size - kv.second.used;
                if (tail > best) { best = tail; bump = kv.first; }
            }
        }
    }
    // Frees every pinned chunk (the caller holds `mu`), waiting for any copy
    // still in flight through it first.
    void release_pinned(ReleaseStats& rs) {
        for (Chunk& c : chunks) {
            if (c.ev) { CUDA_CALL(cudaEventSynchronize(c.ev)); CUDA_CALL(cudaEventDestroy(c.ev)); }
            if (c.p) { CUDA_CALL(cudaFreeHost(c.p)); rs.pinned_bytes_freed += chunk_bytes; }
        }
        chunks.clear();
        next_victim = 0;
        pstats.allocated_bytes = 0;
        pool_exhausted_alloc = false;
    }
    void mark_busy(size_t i, cudaStream_t s) {
        chunks[i].busy = true;
        CUDA_CALL(cudaEventRecord(chunks[i].ev, s));
    }
};

namespace {
std::atomic<bool> g_ctx_created{false};
}

namespace { thread_local const char* g_trace_tag = "-"; }
void set_trace_tag(const char* tag) { g_trace_tag = (tag != nullptr && *tag != '\0') ? tag : "-"; }
const char* trace_tag() { return g_trace_tag; }

// (slot-23 P7) ENGINE_TRACE_VRAM=<ms> (any non-empty value; a number >= 50 is the period, default
// 1000): a detached sampler prints `VRAM t=<s> live_mb= reserved_mb= dev_used_mb=` to stderr every
// period, and at exit one `EFFECTIVE DEVICE_MEM ...` line with the arena peaks, the device-used peak and
// the live-block histogram at the live peak. Diagnostic only: it reads counters, never a value.
namespace {
CudaCtx::Impl* g_trace_impl = nullptr;
std::atomic<size_t> g_trace_dev_used_peak{0};
std::atomic<bool> g_trace_stop{false};   // (audit NIT) set at exit: the sampler makes no CUDA call during teardown
std::chrono::steady_clock::time_point g_trace_t0;
}
static void print_vram_trace_summary();
static void start_vram_trace(CudaCtx::Impl* I) {
    const char* e = std::getenv("ENGINE_TRACE_VRAM");
    if (e == nullptr || *e == '\0') return;
    long ms = std::atol(e);
    if (ms < 50) ms = 1000;
    I->trace_vram = true;
    g_trace_impl = I;
    g_trace_t0 = std::chrono::steady_clock::now();
    std::atexit(&print_vram_trace_summary);
    std::thread([I, ms] {
        for (;;) {
            std::this_thread::sleep_for(std::chrono::milliseconds(ms));
            if (g_trace_stop.load()) return;
            size_t fr = 0, tot = 0;
            if (cudaMemGetInfo(&fr, &tot) != cudaSuccess) { (void)cudaGetLastError(); continue; }
            const size_t used = tot - fr;
            size_t prev = g_trace_dev_used_peak.load();
            while (used > prev && !g_trace_dev_used_peak.compare_exchange_weak(prev, used)) {}
            size_t live, res;
            { std::lock_guard<std::mutex> lk(I->mu); live = I->astats.live_bytes; res = I->astats.reserved_bytes; }
            const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - g_trace_t0).count();
            std::fprintf(stderr, "VRAM t=%.1f live_mb=%zu reserved_mb=%zu dev_used_mb=%zu\n", t, live >> 20, res >> 20, used >> 20);
        }
    }).detach();
}
static void print_vram_trace_summary() {
    g_trace_stop.store(true);
    CudaCtx::Impl* I = g_trace_impl;
    if (I == nullptr) return;
    std::lock_guard<std::mutex> lk(I->mu);
    std::fprintf(stderr, "EFFECTIVE DEVICE_MEM peak_live_mb=%zu peak_reserved_mb=%zu dev_used_peak_mb=%zu cuda_mallocs=%llu soft_trims=%llu oom_retries=%llu at_peak: %s\n",
                 I->astats.peak_live_bytes >> 20, I->astats.peak_reserved_bytes >> 20, g_trace_dev_used_peak.load() >> 20,
                 static_cast<unsigned long long>(I->astats.cuda_mallocs), static_cast<unsigned long long>(I->astats.soft_trims),
                 static_cast<unsigned long long>(I->astats.oom_retries), I->trace_peak_hist.c_str());
}

CudaCtx& CudaCtx::get() {
    if (!available()) throw CudaError(std::string("CudaCtx: ") + unavailable_reason());
    static CudaCtx* ctx = new CudaCtx();   // intentionally leaked: outlives every static DeviceBuf
    g_ctx_created.store(true, std::memory_order_relaxed);
    return *ctx;
}

bool context_created() { return g_ctx_created.load(std::memory_order_relaxed); }

ReleaseStats release_device_caches() {
    if (!context_created() || !available()) return ReleaseStats{};
    return CudaCtx::get().release_cached(true);
}

CudaCtx::CudaCtx() : impl_(new Impl()) {
    CUDA_CALL(cudaSetDevice(probe().index));
    for (int i = 0; i < Impl::kStreams; ++i)
        CUDA_CALL(cudaStreamCreateWithFlags(&impl_->streams[i], cudaStreamNonBlocking));
    {
        size_t fr = 0, tot = 0;
        // (slot-23 P7) under the VRAM-lean default (ENGINE_GPU_LEAN not '0...') a sixteenth of the
        // device (1 GB on the 16 GB card), else a quarter as before: the idle free-list bytes are what
        // nvidia-smi shows beside the live peak (slot-22 (13, 700): 5.0 GB reserved for 2.4 GB live).
        const bool lean_on = engine::lever::gpu_lean_lever().enabled();   // (S25 LEVERS-25) RetiredDeviceLevers.h
        if (cudaMemGetInfo(&fr, &tot) == cudaSuccess && tot != 0) impl_->idle_slack = lean_on ? tot / 16 : tot / 4;
        else (void)cudaGetLastError();
        const size_t mb = env_mb("ENGINE_CUDA_ARENA_SLACK_MB", 0);
        if (mb != 0) impl_->idle_slack = mb << 20;
    }
    impl_->cap_bytes = env_mb("ENGINE_CUDA_PINNED_MB", 128) << 20;
    impl_->chunk_bytes = env_mb("ENGINE_CUDA_PINNED_CHUNK_MB", 8) << 20;
    if (impl_->chunk_bytes == 0) impl_->chunk_bytes = 1u << 20;
    if (impl_->cap_bytes < impl_->chunk_bytes) impl_->cap_bytes = impl_->chunk_bytes;
    impl_->pstats.chunk_bytes = impl_->chunk_bytes;
    impl_->pstats.cap_bytes = impl_->cap_bytes;
    const size_t nthreads = env_mb("ENGINE_CUDA_COPY_THREADS", 4);
    impl_->copiers.reset(new CopyPool(nthreads <= 1 ? 0 : static_cast<int>(std::min<size_t>(nthreads, 16))));
    start_vram_trace(impl_);
}

CudaCtx::~CudaCtx() { delete impl_; }

int    CudaCtx::num_streams() const { return Impl::kStreams; }
Stream CudaCtx::stream(int i) const {
    ENGINE_CHECK_MSG(i >= 0 && i < Impl::kStreams, "CudaCtx::stream: index out of range");
    return Stream{impl_->streams[i]};
}
Stream CudaCtx::create_stream() {
    cudaStream_t st = nullptr;
    CUDA_CALL(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    return Stream{st};
}
void CudaCtx::destroy_stream(Stream s) {
    if (s.h) CUDA_CALL(cudaStreamDestroy(S(s)));
}
void CudaCtx::sync(Stream s) const { CUDA_CALL(cudaStreamSynchronize(S(s))); }
void CudaCtx::sync_all() const { CUDA_CALL(cudaDeviceSynchronize()); }

Event CudaCtx::create_event() const {
    cudaEvent_t e = nullptr;
    CUDA_CALL(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    return Event{e};
}
Event CudaCtx::create_timing_event() const {
    cudaEvent_t e = nullptr;
    CUDA_CALL(cudaEventCreate(&e));
    return Event{e};
}
void  CudaCtx::record(Event e, Stream s) const { CUDA_CALL(cudaEventRecord(E(e), S(s))); }
void  CudaCtx::wait(Stream s, Event e) const { CUDA_CALL(cudaStreamWaitEvent(S(s), E(e), 0)); }
void  CudaCtx::sync(Event e) const { CUDA_CALL(cudaEventSynchronize(E(e))); }
void  CudaCtx::destroy(Event e) const { if (e.h) CUDA_CALL(cudaEventDestroy(E(e))); }
float CudaCtx::elapsed_ms(Event a, Event b) const {
    float ms = 0.f;
    CUDA_CALL(cudaEventElapsedTime(&ms, E(a), E(b)));
    return ms;
}

// ---- arena -------------------------------------------------------------------
void* CudaCtx::dalloc(size_t bytes) {
    std::lock_guard<std::mutex> lk(impl_->mu);
    Impl& I = *impl_;
    if (bytes == 0) bytes = Impl::kAlign;
    bytes = (bytes + Impl::kAlign - 1) / Impl::kAlign * Impl::kAlign;
    auto it = I.free_lists.find(bytes);
    if (it != I.free_lists.end() && !it->second.empty()) {
        void* p = it->second.back();
        it->second.pop_back();
        ++I.astats.reuses;
        I.slab_of(p).live += bytes;
        I.astats.live_bytes += bytes;
        I.astats.peak_live_bytes = std::max(I.astats.peak_live_bytes, I.astats.live_bytes);
        I.trace_alloc(p, bytes);
        return p;
    }
    ENGINE_CHECK_MSG(!capturing_now(), "CudaCtx::dalloc: allocation inside a graph capture (allocate before begin())");
    // bump from the current slab, else a new slab
    auto cur = I.bump != 0 ? I.slabs.find(I.bump) : I.slabs.end();
    if (cur == I.slabs.end() || cur->second.used + bytes > cur->second.size) {
        // (slot-14 WP-3) soft trim: once the bytes the arena holds idle (free
        // lists plus slab tails) have grown by more than `idle_slack` since
        // the last trim, return every fully free slab before asking the
        // driver for more.  The exact-size free lists otherwise grow without
        // bound over a long run whose buffer shapes vary (every cell of a
        // validation grid has its own), which is how one validation alone
        // filled a 16 GB card.  No capture may be open (cudaFree synchronises).
        const size_t idle = I.astats.reserved_bytes - std::min(I.astats.reserved_bytes, I.astats.live_bytes);
        if (idle > I.idle_after_trim + I.idle_slack && g_active_captures.load(std::memory_order_relaxed) == 0) {
            ReleaseStats rs;
            I.trim_arena(rs);
            ++I.astats.soft_trims;
            I.idle_after_trim = I.astats.reserved_bytes - std::min(I.astats.reserved_bytes, I.astats.live_bytes);
            cur = I.bump != 0 ? I.slabs.find(I.bump) : I.slabs.end();
        }
    }
    if (cur == I.slabs.end() || cur->second.used + bytes > cur->second.size) {
        // (slot-14 WP-16, audit M-7) re-tested after the soft trim: the bump may have moved
        // to a surviving slab whose tail fits.
        const size_t sz = std::max(bytes, Impl::kSlabBytes);
        void* base = nullptr;
        cudaError_t e = cudaMalloc(&base, sz);
        if (e == cudaErrorMemoryAllocation) {
            // (slot-14 WP-3) whole slabs of free-listed blocks may be what
            // fills the device: return them to the driver and retry once.
            (void)cudaGetLastError();
            ReleaseStats rs;
            I.trim_arena(rs);
            ++I.astats.oom_retries;
            I.idle_after_trim = I.astats.reserved_bytes - std::min(I.astats.reserved_bytes, I.astats.live_bytes);
            e = cudaMalloc(&base, sz);
        }
        CUDA_CALL(e);
        ++I.astats.cuda_mallocs;
        I.astats.reserved_bytes += sz;
        I.astats.peak_reserved_bytes = std::max(I.astats.peak_reserved_bytes, I.astats.reserved_bytes);
        const uintptr_t key = reinterpret_cast<uintptr_t>(base);
        I.slabs[key] = Impl::Slab{base, sz, 0, 0};
        I.bump = key;
        cur = I.slabs.find(key);
    }
    Impl::Slab& s = cur->second;
    void* p = static_cast<char*>(s.base) + s.used;
    s.used += bytes;
    s.live += bytes;
    I.astats.live_bytes += bytes;
    I.astats.peak_live_bytes = std::max(I.astats.peak_live_bytes, I.astats.live_bytes);
    I.trace_alloc(p, bytes);
    return p;
}
void CudaCtx::dfree(void* p, size_t bytes) {
    if (!p) return;
    std::lock_guard<std::mutex> lk(impl_->mu);
    if (bytes == 0) bytes = Impl::kAlign;
    bytes = (bytes + Impl::kAlign - 1) / Impl::kAlign * Impl::kAlign;
    Impl::Slab& s = impl_->slab_of(p);
    ENGINE_CHECK_MSG(s.live >= bytes, "CudaCtx::dfree: more bytes freed than are live in the slab");
    s.live -= bytes;
    impl_->free_lists[bytes].push_back(p);
    impl_->trace_free(p);
    impl_->astats.live_bytes -= std::min(bytes, impl_->astats.live_bytes);
}
ReleaseStats CudaCtx::release_cached(bool pinned) {
    ENGINE_CHECK_MSG(!capturing_now(), "CudaCtx::release_cached inside a graph capture");
    // (slot-14 WP-16, audit M-6) the same predicate as dalloc's soft trim: no capture open on ANY thread.
    ENGINE_CHECK_MSG(g_active_captures.load(std::memory_order_relaxed) == 0,
                     "CudaCtx::release_cached while another thread holds an open graph capture");
    std::lock_guard<std::mutex> lk(impl_->mu);
    ReleaseStats rs;
    impl_->trim_arena(rs);
    impl_->idle_after_trim = impl_->astats.reserved_bytes - std::min(impl_->astats.reserved_bytes, impl_->astats.live_bytes);
    if (pinned) impl_->release_pinned(rs);
    return rs;
}
ArenaStats CudaCtx::arena_stats() const { return impl_->astats; }
size_t CudaCtx::arena_idle_slack() const { return impl_->idle_slack; }

// ---- transfers ----------------------------------------------------------------
void CudaCtx::upload(const void* host, void* dev, size_t bytes, Stream s) {
    if (bytes == 0) return;
    ENGINE_CHECK_MSG(!capturing_now(), "CudaCtx::upload: staged copy inside a graph capture");
    std::lock_guard<std::mutex> lk(impl_->mu);
    Impl& I = *impl_;
    I.tstats.h2d_bytes += bytes; ++I.tstats.h2d_calls;
    const char* src = static_cast<const char*>(host);
    char* dst = static_cast<char*>(dev);
    size_t off = 0;
    while (off < bytes) {
        const size_t idx = I.acquire_chunk();
        if (idx == Impl::kNone) {
            // no pinned memory obtainable at all: pageable fallback, counted
            ++I.tstats.pageable_copies;
            CUDA_CALL(cudaMemcpyAsync(dst + off, src + off, bytes - off, cudaMemcpyHostToDevice, S(s)));
            CUDA_CALL(cudaStreamSynchronize(S(s)));   // pageable async is synchronous anyway
            return;
        }
        const size_t n = std::min(I.chunk_bytes, bytes - off);
        I.copiers->copy(I.chunks[idx].p, src + off, n);
        CUDA_CALL(cudaMemcpyAsync(dst + off, I.chunks[idx].p, n, cudaMemcpyHostToDevice, S(s)));
        I.mark_busy(idx, S(s));
        off += n;
    }
}

void CudaCtx::download(void* host, const void* dev, size_t bytes, Stream s) {
    if (bytes == 0) return;
    ENGINE_CHECK_MSG(!capturing_now(), "CudaCtx::download: staged copy inside a graph capture");
    std::lock_guard<std::mutex> lk(impl_->mu);
    Impl& I = *impl_;
    I.tstats.d2h_bytes += bytes; ++I.tstats.d2h_calls;
    const char* src = static_cast<const char*>(dev);
    char* dst = static_cast<char*>(host);
    // pipeline: issue D2H into chunks; drain (event sync + memcpy) in order
    struct Pending { size_t idx; size_t off; size_t n; };
    std::deque<Pending> q;
    std::vector<char> pending(I.chunks.size(), 0);
    auto drain_one = [&]() {
        const Pending pd = q.front(); q.pop_front();
        CUDA_CALL(cudaEventSynchronize(I.chunks[pd.idx].ev));
        I.copiers->copy(dst + pd.off, I.chunks[pd.idx].p, pd.n);
        I.chunks[pd.idx].busy = false;
        pending[pd.idx] = 0;
    };
    size_t off = 0;
    while (off < bytes) {
        pending.resize(I.chunks.size() + 1, 0);   // acquire may append one chunk
        size_t idx = I.acquire_chunk(&pending);
        while (idx == Impl::kMustDrain) {          // every busy chunk is ours: free one
            drain_one();
            idx = I.acquire_chunk(&pending);
        }
        if (idx == Impl::kNone) {
            while (!q.empty()) drain_one();
            ++I.tstats.pageable_copies;
            CUDA_CALL(cudaMemcpy(dst + off, src + off, bytes - off, cudaMemcpyDeviceToHost));
            return;
        }
        const size_t n = std::min(I.chunk_bytes, bytes - off);
        CUDA_CALL(cudaMemcpyAsync(I.chunks[idx].p, src + off, n, cudaMemcpyDeviceToHost, S(s)));
        I.mark_busy(idx, S(s));
        pending.resize(std::max(pending.size(), idx + 1), 0);
        pending[idx] = 1;
        q.push_back(Pending{idx, off, n});
        off += n;
    }
    while (!q.empty()) drain_one();
}

void CudaCtx::copy_d2d(void* dst, const void* src, size_t bytes, Stream s) {
    if (bytes == 0) return;
    impl_->tstats.d2d_bytes += bytes;
    CUDA_CALL(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, S(s)));
}
void CudaCtx::memset_d(void* dst, int value, size_t bytes, Stream s) {
    if (bytes == 0) return;
    CUDA_CALL(cudaMemsetAsync(dst, value, bytes, S(s)));
}
void CudaCtx::upload_pageable(const void* host, void* dev, size_t bytes) {
    ++impl_->tstats.pageable_copies;
    CUDA_CALL(cudaMemcpy(dev, host, bytes, cudaMemcpyHostToDevice));
}
void CudaCtx::download_pageable(void* host, const void* dev, size_t bytes) {
    ++impl_->tstats.pageable_copies;
    CUDA_CALL(cudaMemcpy(host, dev, bytes, cudaMemcpyDeviceToHost));
}
void* CudaCtx::pinned_alloc(size_t bytes) {
    void* p = nullptr;
    CUDA_CALL(cudaHostAlloc(&p, bytes, cudaHostAllocDefault));
    return p;
}
void CudaCtx::pinned_free(void* p) { if (p) CUDA_CALL(cudaFreeHost(p)); }

PinnedPoolStats CudaCtx::pinned_stats() const { return impl_->pstats; }
TransferStats   CudaCtx::transfer_stats() const { return impl_->tstats; }
void            CudaCtx::reset_transfer_stats() { impl_->tstats = TransferStats{}; }
size_t CudaCtx::pinned_cap_bytes() const { return impl_->cap_bytes; }
size_t CudaCtx::pinned_chunk_bytes() const { return impl_->chunk_bytes; }
int    CudaCtx::copy_threads() const { return impl_->copiers->size(); }

// ---------------------------------------------------------------------------
// GraphCapture
// ---------------------------------------------------------------------------
struct GraphCapture::Impl {
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    bool capturing = false;
    size_t nodes = 0;
};

GraphCapture::GraphCapture() : impl_(new Impl()) {}
GraphCapture::~GraphCapture() {
    if (impl_->capturing) {
        cudaGraph_t g = nullptr;
        (void)cudaStreamEndCapture(impl_->stream, &g);
        if (g) (void)cudaGraphDestroy(g);
        --t_capture_depth;
        g_active_captures.fetch_sub(1, std::memory_order_relaxed);
    }
    if (impl_->exec) (void)cudaGraphExecDestroy(impl_->exec);
    if (impl_->graph) (void)cudaGraphDestroy(impl_->graph);
    delete impl_;
}
void GraphCapture::begin(Stream s) {
    ENGINE_CHECK_MSG(!impl_->capturing, "GraphCapture::begin: already capturing");
    ENGINE_CHECK_MSG(s.valid(), "GraphCapture::begin: invalid stream");
    if (impl_->exec) { CUDA_CALL(cudaGraphExecDestroy(impl_->exec)); impl_->exec = nullptr; }
    if (impl_->graph) { CUDA_CALL(cudaGraphDestroy(impl_->graph)); impl_->graph = nullptr; }
    impl_->stream = S(s);
    CUDA_CALL(cudaStreamBeginCapture(impl_->stream, cudaStreamCaptureModeThreadLocal));
    g_active_captures.fetch_add(1, std::memory_order_relaxed);
    impl_->capturing = true;
    ++t_capture_depth;
}
void GraphCapture::end() {
    ENGINE_CHECK_MSG(impl_->capturing, "GraphCapture::end: not capturing");
    impl_->capturing = false;
    --t_capture_depth;
    g_active_captures.fetch_sub(1, std::memory_order_relaxed);
    CUDA_CALL(cudaStreamEndCapture(impl_->stream, &impl_->graph));
    CUDA_CALL(cudaGraphInstantiate(&impl_->exec, impl_->graph, 0));
    size_t n = 0;
    CUDA_CALL(cudaGraphGetNodes(impl_->graph, nullptr, &n));
    impl_->nodes = n;
}
void GraphCapture::launch(Stream s) const {
    ENGINE_CHECK_MSG(impl_->exec != nullptr, "GraphCapture::launch: not instantiated");
    CUDA_CALL(cudaGraphLaunch(impl_->exec, S(s)));
}
bool   GraphCapture::instantiated() const { return impl_->exec != nullptr; }
size_t GraphCapture::num_nodes() const { return impl_->nodes; }
Stream GraphCapture::captured_stream() const { return Stream{impl_->stream}; }
bool   GraphCapture::capturing() const { return impl_->capturing; }

} // namespace cuda
} // namespace engine
