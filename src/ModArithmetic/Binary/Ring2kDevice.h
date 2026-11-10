// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once

// (slot-24 PAR-GPU) The device route of ring2k::echelon_blocked for the native u32 / u64 rungs (Z/2^32, Z/2^64).
//
// WHAT RUNS WHERE.  The blocked right-looking LU of Ring2k.h, with the working matrix RESIDENT on the device:
//   * the device holds a column window of width cw of A, all m rows: cw = n when it fits ("resident": one window);
//     else cw = the largest multiple of the panel width that fits ("tiled", in PHASES): a phase uploads the window
//     [c, c + cw) at the next panel column c, factors panels right-looking inside it until the window's panels are done or
//     the host store of the phase's L factors would pass its budget (store_budget(), ENGINE_GPU_SMITH2K_HOST_MB), downloads
//     the window, passes the columns right of it through the device in slabs that receive the phase's panels (swaps, TRSM,
//     GEMM replayed from the store), and gives the columns left of it the phase's swaps on the host (one row
//     permutation).  Every column receives the same operations in the same order as in the host code (each column's update
//     reads only that column and L), so the tiling cannot change a value;
//   * per panel the host fetches the panel's columns (rows r .. m-1, one 2D copy), factors them with the PAR-2K lazy panel
//     (the pivot search, the row swaps, the L slots: exactly the host code's arithmetic), and hands the device the
//     factored panel, the swap list and L; the device applies the swaps to its other columns, the unit-lower TRSM on the
//     pivot rows and the trailing GEMM C -= L U;
//   * LOOKAHEAD: the device updates the NEXT panel's columns first and copies them back on a second stream while the rest
//     of the trailing GEMM runs, so the host's panel factorisation overlaps the device GEMM;
//   * at the end the device columns come back in one 2D copy.
// Z/2^W is the device's native wraparound integer arithmetic: every entry receives the unique value of the host code
// (one exact answer per entry, independent of tiling and summation order), the pivots are read from those values, so
// the matrix, the pivots, the valuations and every Smith / Howell output are bit-identical to the CPU
// (DescentSmithGpuValidation).
//
// THE HOOK.  ModArithmetic links no CUDA: the device operations are a table of function pointers (Ops) that
// cohomology_app installs (Integral/Descent/DescentSmithGpu.cpp, over EngineCuda's CudaRing2k.h).  Without a table, or
// when plan() declines (the policy: ENGINE_DEVICE=gpu and ENGINE_GPU_SMITH2K not 0; the size floor; a device; the
// device memory), echelon_blocked runs the host code unchanged.  A CUDA error before the first host write falls back to
// the host code (same values); after it, it is rethrown (no silent recovery from a half-written matrix).

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine {
namespace ring2k {
namespace device {

// One panel step for the device (all asynchronous on the device's compute stream; the host buffers are copied into
// pinned staging before step() returns, so the caller may reuse them at once).
struct StepArgs {
    uint32_t m = 0;                       // rows of A
    uint32_t r = 0, np = 0;               // the panel's pivot rows r .. r + np - 1
    uint32_t pc0 = 0, pc1 = 0;            // the panel's columns (the swaps skip them: the factored panel carries its swaps)
    const void* panel = nullptr;          // the factored panel, rows r .. m-1, row-major, ld = ldp elements; null = host panel
    size_t ldp = 0;
    const uint32_t* swaps = nullptr;      // nswaps pairs (a, b) of global rows, swapped in this order
    uint32_t nswaps = 0;
    const void* L = nullptr;              // the L store from row r on (ld = ldl elements); rows r .. m-1, columns 0 .. np-1 read
    size_t ldl = 0;
    uint32_t u0 = 0;                      // the device's TRSM / GEMM columns [u0, cw) (u0 >= cw: no update)
    uint32_t la0 = 0, la1 = 0;            // lookahead: rows r + np .. m-1 of the columns [la0, la1) fetched after their update
    uint32_t w = 0;                       // the window columns in use, [0, w) (0 = the session's cw); every column above is window-relative
};

struct Ops {
    // The device column window cw for an m x n matrix of `elem`-byte entries (4 or 8) with this panel width:
    // n (resident), a multiple of `panel` below n (tiled), or 0 (the device route declines; the host code runs).
    uint32_t (*plan)(uint32_t elem, uint32_t m, uint32_t n, uint32_t panel) = nullptr;
    void* (*open)(uint32_t elem, uint32_t m, uint32_t cw, uint32_t panel) = nullptr;   // throws (std::runtime_error) on failure
    // Synchronous 2D copies between a host block (pointer to its first entry, ld = ldh elements) and the device window's
    // rows [r0, r0 + rows) x columns [c0, c0 + cols) (window-relative).
    void (*put)(void* s, const void* host, size_t ldh, uint32_t r0, uint32_t rows, uint32_t c0, uint32_t cols) = nullptr;
    void (*get)(void* s, void* host, size_t ldh, uint32_t r0, uint32_t rows, uint32_t c0, uint32_t cols) = nullptr;
    void (*step)(void* s, const StepArgs& a) = nullptr;
    // Waits for the last step's lookahead block and copies it to host (ld = ldh elements).
    void (*fetch)(void* s, void* host, size_t ldh) = nullptr;
    void (*close)(void* s) = nullptr;                    // never throws; frees everything
    // ENGINE_VERIFY_DEVICE twin: re-run every device LU on the host and compare entrywise (throws on a difference).
    bool (*verify)() = nullptr;
    // Tiled route: the host bytes the L store of one phase may hold (0 or null = unlimited).
    uint64_t (*store_budget)() = nullptr;
};
void       set_ops(const Ops* o);    // the table must outlive every call (a static)
const Ops* ops();

// Diagnostics (relaxed counters; never a value).
struct Stats {
    uint64_t calls = 0;          // echelon_blocked calls of the u32 / u64 rungs that asked plan()
    uint64_t device = 0;         // ... that ran on the device (resident + tiled)
    uint64_t resident = 0, tiled = 0;   // ... in one window / in several column slabs
    uint64_t slabs = 0;          // window / slab uploads of the tiled LUs (phases + replay slabs)
    uint64_t replays = 0;        // panel replays onto later slabs
    uint64_t fallbacks = 0;      // CUDA error before the first host write: the host code ran
    uint64_t panels = 0;         // device panel steps
    uint64_t lookaheads = 0;     // ... whose next panel came back by lookahead
    uint64_t verified = 0;       // device LUs re-run on the host and compared (verify())
    // wall nanoseconds on the calling thread (diagnostic): the session open + slab uploads, the panel fetches (lookahead
    // waits included), the host panel factorisations, the device step enqueues (staging copies), the panel replays onto
    // later slabs + the host row permutations of the earlier slabs, the slab downloads + close
    uint64_t ns_open = 0, ns_fetch = 0, ns_panel = 0, ns_step = 0, ns_host = 0, ns_close = 0;
};
Stats stats();
void  note(uint64_t Stats::*field, uint64_t v = 1);

// While one exists on this thread, echelon_blocked never asks the device (the verify twin's host run; validations).
struct ScopedHostOnly {
    ScopedHostOnly();
    ~ScopedHostOnly();
    ScopedHostOnly(const ScopedHostOnly&) = delete;
    ScopedHostOnly& operator=(const ScopedHostOnly&) = delete;
};
bool host_only();

} // namespace device
} // namespace ring2k
} // namespace engine
