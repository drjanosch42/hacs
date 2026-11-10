// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// =====================================================================
// CudaCtx.h -- the CUDA runtime context of the matrix engine (Phase 5)
// =====================================================================
//
// One process owns the GPU (WSL2 has no MPS, 04 Sec. 7.1) and multiplexes it
// internally, so the context is a process-wide singleton: device selection,
// a small set of streams, the pinned staging pool (<= 128 MB, cudaHostAlloc
// from the start -- cudaHostRegister is unsupported on WSL2), the device
// arena, and the transfer counters that EngineCudaMemoryValidation-style
// checks read ("no pageable D2H, ever").
//
// This header deliberately includes NO CUDA header: every handle is opaque
// (`Stream`, `Event`), so the tape / scheduler of Phase 4 can include it from
// a plain C++ TU without the toolkit's include path. The implementation
// (CudaCtx.cpp, cuda_kernels.cu) is the only place <cuda_runtime.h> appears.
//
// Availability model. `available()` is the PHYSICAL question (a usable device
// and a working runtime); it is answered once, never throws, and is false on
// a machine without a GPU -- every validation in this layer then reports
// "[SKIP]" and returns 0. `device_enabled()` is the POLICY question the tape's
// device assignment asks: available() AND ENGINE_DEVICE=gpu (the GPU is never
// enabled by default -- plan Phase 5, "unblocked-kernel trap").
//   ENGINE_DEVICE=cpu|gpu|hybrid cpu (default): device_enabled() == false;
//                                gpu: every eligible tape node on the device;
//                                hybrid: the cost model of pass 6 decides
//   ENGINE_CUDA_DEVICE=<index>   which device (default 0)
//   ENGINE_CUDA_DISABLE=1        makes available() false (tests the no-GPU path)
//   ENGINE_CUDA_ARENA_SLACK_MB   (slot-14 WP-3) idle arena bytes tolerated
//                                before a soft trim returns fully free slabs
//                                to the driver (default: a sixteenth of the
//                                device under ENGINE_GPU_LEAN's default, a
//                                quarter with ENGINE_GPU_LEAN=0 (slot-23 P7; the env
//                                arm retired in OFF builds by S25 LEVERS-25);
//                                read once; never changes a value)
//
// Every runtime error is thrown as engine::cuda::CudaError (std::runtime_error)
// carrying the CUDA error string and the call site.

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace engine {
namespace cuda {

class CudaError : public std::runtime_error {
public:
    explicit CudaError(const std::string& what) : std::runtime_error(what) {}
};

// ---- availability / policy ------------------------------------------------
bool        available();                 // physical: device present, runtime OK (cached)
const char* unavailable_reason();        // "" when available()
enum class Device : uint8_t { Cpu, Gpu, Hybrid };
Device      requested_device();          // ENGINE_DEVICE (default Cpu)
bool        device_enabled();            // available() && requested_device() != Cpu

struct DeviceInfo {
    std::string name;
    int    index = -1;
    int    cc_major = 0, cc_minor = 0;
    int    sms = 0;
    size_t total_mem = 0;
    size_t free_mem = 0;
    int    driver_version = 0;
    int    runtime_version = 0;
};
DeviceInfo device_info();                // throws CudaError when !available()

// ---- opaque handles --------------------------------------------------------
struct Stream { void* h = nullptr; bool valid() const { return h != nullptr; } };
struct Event  { void* h = nullptr; };

// ---- pinned staging pool ---------------------------------------------------
// Fixed-size chunks allocated with cudaHostAlloc, lazily, up to `cap_bytes`
// (default 128 MB). A transfer larger than the pool recycles chunks (waiting
// on their events), so the pool NEVER grows with the problem size: peak_bytes()
// <= cap_bytes at all times (CudaTransferValidation asserts it).
struct PinnedPoolStats {
    size_t   chunk_bytes = 0;
    size_t   cap_bytes = 0;
    size_t   allocated_bytes = 0;   // current pinned allocation
    size_t   peak_bytes = 0;        // high-water mark of allocated_bytes
    uint64_t acquisitions = 0;
    uint64_t waits = 0;             // times a caller had to wait for a busy chunk
};

// ---- transfer counters -----------------------------------------------------
struct TransferStats {
    uint64_t h2d_bytes = 0;
    uint64_t d2h_bytes = 0;
    uint64_t h2d_calls = 0;
    uint64_t d2h_calls = 0;
    uint64_t d2d_bytes = 0;
    uint64_t pageable_copies = 0;   // MUST stay 0 while the pool has chunks
};

// ---- device arena ----------------------------------------------------------
// A caching allocator over cudaMalloc: exact-size free lists + slab bumping.
// Nothing here is capture-safe: allocate every buffer / workspace BEFORE a
// graph capture begins (cudaMalloc inside a capture is a CUDA error).
struct ArenaStats {
    size_t reserved_bytes = 0;   // total cudaMalloc'ed and not yet returned to the driver
    size_t live_bytes = 0;       // handed out and not yet released
    size_t peak_live_bytes = 0;
    size_t peak_reserved_bytes = 0;   // high-water of reserved_bytes (slot-14 WP-3)
    uint64_t cuda_mallocs = 0;
    uint64_t reuses = 0;
    uint64_t cuda_frees = 0;          // slabs returned to the driver (slot-14 WP-3)
    uint64_t oom_retries = 0;         // cudaMalloc failures answered by a trim + one retry (WP-3)
    uint64_t soft_trims = 0;          // idle-growth trims before a new slab (WP-3, ENGINE_CUDA_ARENA_SLACK_MB)
};

// What one release returned to the driver (slot-14 WP-3).
struct ReleaseStats {
    size_t   arena_bytes_freed = 0;   // slab bytes cudaFree'd
    uint64_t slabs_freed = 0;
    uint64_t slabs_kept = 0;          // slabs still holding a live block (never touched)
    size_t   pinned_bytes_freed = 0;  // pinned staging chunks cudaFreeHost'ed
};

class CudaCtx {
public:
    // The singleton; throws CudaError when !available().
    static CudaCtx& get();

    int    num_streams() const;
    Stream stream(int i) const;        // i in [0, num_streams())
    // Private streams for a tape executor (Phase 5 Stage B): a graph capture
    // owns its stream, so concurrently running executors must not share one.
    Stream create_stream();            // non-blocking; destroy with destroy_stream
    void   destroy_stream(Stream s);
    void   sync(Stream s) const;       // cudaStreamSynchronize
    void   sync_all() const;           // cudaDeviceSynchronize

    Event  create_event() const;       // disable-timing event
    void   record(Event e, Stream s) const;
    void   wait(Stream s, Event e) const;     // stream waits on event
    void   sync(Event e) const;               // host waits on event
    void   destroy(Event e) const;
    // Timing events (for benchmarks only).
    Event  create_timing_event() const;
    float  elapsed_ms(Event a, Event b) const;

    // Device memory (arena).
    void*  dalloc(size_t bytes);
    void   dfree(void* p, size_t bytes);
    ArenaStats arena_stats() const;
    // (slot-23 P7) the soft-trim slack in bytes (ENGINE_CUDA_ARENA_SLACK_MB, else the lean / legacy default).
    size_t arena_idle_slack() const;

    // (slot-14 WP-3) Returns the memory the arena caches to the driver: every
    // slab none of whose blocks is live is cudaFree'd (its blocks leave the
    // free lists), and -- when `pinned` -- every pinned staging chunk is
    // cudaFreeHost'ed after its event completed (the pool regrows lazily, up
    // to its cap).  A slab holding a live block is never touched, so no
    // pointer anyone holds is invalidated.  Without it the arena's exact-size
    // free lists keep every freed block for the life of the process: the
    // monotone VRAM growth of a long `ENGINE_DEVICE=gpu` run (slot-14 [C4]
    // Sec. 2, [C7] headline 3).  Never changes a value: which device address
    // a buffer lives at is not an input of any kernel.  Not legal inside a
    // graph capture.
    ReleaseStats release_cached(bool pinned = true);

    // Staged transfers: pageable host memory <-> device through the pinned
    // pool. `upload` returns when every chunk has been handed to the stream
    // (the host side of the copy is done; the device side is stream-ordered).
    // `download` returns when the bytes are IN the host buffer (it must wait
    // for the D2H DMA): it is synchronous with respect to `stream`'s prior work.
    void upload(const void* host, void* dev, size_t bytes, Stream s);
    void download(void* host, const void* dev, size_t bytes, Stream s);
    // Direct pinned<->device / device<->device copies (no staging).
    void copy_d2d(void* dst, const void* src, size_t bytes, Stream s);
    void memset_d(void* dst, int value, size_t bytes, Stream s);
    // Unstaged pageable copies -- for the bandwidth benchmark ONLY; counted.
    void upload_pageable(const void* host, void* dev, size_t bytes);
    void download_pageable(void* host, const void* dev, size_t bytes);
    // A raw pinned allocation outside the pool (bench: pinned-vs-pageable).
    void* pinned_alloc(size_t bytes);
    void  pinned_free(void* p);

    PinnedPoolStats pinned_stats() const;
    TransferStats   transfer_stats() const;
    void            reset_transfer_stats();

    // Tunables (before first use of the pool): env ENGINE_CUDA_PINNED_MB caps
    // the pool (default 128), ENGINE_CUDA_PINNED_CHUNK_MB sets the chunk (8).
    size_t pinned_cap_bytes() const;
    size_t pinned_chunk_bytes() const;
    // Staging memcpys are sliced across this many persistent host threads
    // (ENGINE_CUDA_COPY_THREADS, default 4): a single thread reads pinned
    // memory at ~5 GB/s on WSL2, see CudaCtx.cpp. 0 = inline memcpy.
    int    copy_threads() const;

    struct Impl;
private:
    CudaCtx();
    ~CudaCtx();
    CudaCtx(const CudaCtx&) = delete;
    CudaCtx& operator=(const CudaCtx&) = delete;
    Impl* impl_;
};

// (slot-14 WP-3) The between-validations release: CudaCtx::release_cached()
// on the context if it EXISTS -- a no-op returning zeros when it was never
// created (this call never initialises CUDA) or when !available().
// ValidationRegistry calls it after every validation of a process whose
// effective device policy is not cpu.
ReleaseStats release_device_caches();
bool         context_created();

// (slot-23 P7) ENGINE_TRACE_VRAM attribution: the calling thread's current program family (e.g. a
// ShapeKey tag; the pointer must stay valid while it is set); the arena records it per live block while the
// trace is on (it copies the string). Diagnostic only; a no-op in the CPU-only stub.
void        set_trace_tag(const char* tag);
const char* trace_tag();

// ---- CUDA graph capture -------------------------------------------------------
// Captures every launch issued on `stream` between begin() and end() into an
// executable graph; launch() replays it (0.68 us per node vs 6-24 us per
// stream launch, 04 Sec. 1.4). Rules: no allocation inside the capture
// (dalloc / batch uploads happen before begin()), launches go to the captured
// stream only, and a replay reads whatever the operand buffers hold at replay
// time (the graph records POINTERS, not values).
class GraphCapture {
public:
    GraphCapture();
    ~GraphCapture();
    GraphCapture(const GraphCapture&) = delete;
    GraphCapture& operator=(const GraphCapture&) = delete;

    void begin(Stream s);           // cudaStreamBeginCapture (thread-local mode)
    void end();                     // cudaStreamEndCapture + instantiate
    void launch(Stream s) const;    // cudaGraphLaunch
    bool instantiated() const;
    size_t num_nodes() const;
    Stream captured_stream() const;
    bool capturing() const;

    struct Impl;
private:
    Impl* impl_;
};

// True while ANY GraphCapture on this thread is between begin() and end():
// the transfer helpers refuse to run then (a staged copy synchronises the
// host, which is illegal inside a capture).
bool capturing_now();

} // namespace cuda
} // namespace engine
