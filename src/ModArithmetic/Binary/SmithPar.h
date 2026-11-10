// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once

// (slot-24 PAR-2K) Threads for the ring2k / ring3k valuation-level Smith forms (Ring2k.h, Ring3k.h; ENGINE_API.md Sec. 8.2 / 8.4).
//
// WHAT IS SPLIT.  Inside the blocked right-looking LU (ring2k::echelon_blocked, ring3k's echelon_blocked) the per-panel
//   (a) trailing update C -= L U (one GEMM) is split into disjoint tiles of C (row tiles; row x column tiles when the
//       trailing block has few rows), and
//   (b) the unit-lower TRSM on the panel's pivot rows is split into disjoint column tiles (every column is its own solve),
//   (c) the panel factorisation's row updates for one pivot column (rows below the pivot) are split into disjoint row tiles,
//       and the search for the next column's pivot (the FIRST odd / unit row) is fused into the same region as a per-tile
//       "first candidate" whose minimum over the tiles is the serial loop's answer.
// Every entry of the working matrix receives exactly the arithmetic of the serial code (Z/2^W wraparound, or canonical
// residues mod 3^k: one exact answer per entry, independent of tiling and summation order), the pivot rule reads the same
// values, so the pivots, the row swaps and every output are bit-identical for every thread count.
//
// WHO USES IT.  Opt-in per call site: the descent's Smith calls (Integral/Descent/) open a ScopedEnable; every other consumer
// of ring2k / ring3k (the census, the orbifold campaign, the L23 reader) stays serial exactly as before.  threads() is 1
// outside a ScopedEnable, so the serial code runs verbatim there.
//
// THE LEVER.  HECKE_DESCENT_SMITH_MT: default ON; "0" = serial everywhere (the pre-PAR-2K code path, verbatim); any other
// value warns once on stderr and stays on.  Read once per process.  ENGINE_THREADS=1 is the serial route as well.
//
// THE THREAD BUDGET (slot-24 PAR-UNIFY).  default_threads() is the ONE descent thread count of the process, shared by this
// file's ring2k / ring3k route and PAR-D2's maps::SmithForm (Integral/Descent/DescentSmith.cpp): ENGINE_THREADS when it is
// an integer in [1, kMaxThreads] (0 or unset = the default; a value above kMaxThreads is clamped; anything else -- "-1",
// "abc", "16x" -- warns once on stderr and takes the default), else the process's CPU affinity count (sched_getaffinity:
// a taskset / Slurm / numactl --cpunodebind allotment, NOT the machine's hardware_concurrency), capped at
// kDefaultThreadCap.  Kits that run several descents or shards per node must set ENGINE_THREADS (cores per process); on
// a multi-socket node run one process per socket under numactl --cpunodebind=k --membind=k (the pool is not NUMA-aware).
//
// THE POOL.  ONE private persistent pool per process (not engine::tape::ThreadPool: ModArithmetic does not link the
// engine, and the engine's pool is capped at 32 workers), shared by both Smith forms: parallel_for (this route; opt-in by
// ScopedEnable) and parallel_for_n (an explicit thread count; PAR-D2).  Workers spin briefly (~0.1-1 ms) after a region,
// then block on a condition variable, so an idle pool costs nothing.  One region at a time: a call from inside a task, or
// while another thread holds the pool (several descents in one process), runs serially on the calling thread (same
// values), so a process never runs more than default_threads() pool threads plus its own calling threads.
//
// THE WORK THRESHOLD.  min_work() (default 2^20 multiply-adds): a step whose work is below 2 min_work() stays serial (both
// Smith forms), so small cells pay no region overhead.  Deterministic: a function of the shape only, never of the timing,
// and never a value (every split is entrywise the serial arithmetic).
//
// THE PANEL (slot-24 PAR-UNIFY, size rule PANEL-IMPL).  The threaded route (threads() > 1) runs the blocked LU with a wider
// panel than the serial route's Params::panel (64): fewer, larger trailing GEMMs; identical values for exact Z/2^W and
// canonical Z/3^k (audit da38ffe0 Sec. 2), so no width changes a value.  The ring2k LU (route_panel_2k) takes kLargePanel = 512
// columns when the matrix it factors has >= kLargePanelRows = 12 000 rows (A.rows at that Smith level / Howell call), else
// kDefaultPanel = 128 (PANEL_report.md: 512 cuts the (1000, 20) LU by 17 %, but costs ~1 s on the small level-1 LUs, where the
// lazy panel's serial pivot search outweighs the GEMM saving); the ring3k LU keeps 128.  HECKE_DESCENT_SMITH_PANEL=<w>
// (read-only knob, 1..4096) is the override and the rollback: an explicit width wins for every size and both rings (=128
// restores the pre-PANEL-IMPL default, =64 the pre-PAR-UNIFY one).  The ring2k DEVICE LU (Ring2kDeviceLU.h) has its own panel,
// kDevicePanel = 128 on the threaded route (its panel is factored on the host, serially within the panel, so 512 costs it
// ~20 s at (1000, 20)); neither the size rule nor HECKE_DESCENT_SMITH_PANEL reaches it.
//
// THE EFFECTIVE LINE.  When HECKE_DESCENT_SMITH_MT, HECKE_DESCENT_SMITH_PANEL or (PAR-F2) HECKE_DESCENT_SMITH_F2_MT is set, one line
//   EFFECTIVE HECKE_DESCENT_SMITH_MT phase=start mode=threaded|serial lever=on|off source=env|default threads=<n>
//            threads_source=env|affinity|hardware|env-invalid panel=<w>|auto(128|512) panel_rows=12000
// is printed on stderr at start-up (so a run killed by a wall-clock limit still shows it), and at exit the same line with
// phase=exit and regions=parallel:<a>,serial:<b> (this ring2k / ring3k route) d2_regions=<e> (PAR-D2's maps::SmithForm on the
// shared pool, governed by HECKE_DESCENT_SMITH_BLOCKED, not by this lever) tasks=<c> panel_splits=<d> max_used=<m> scoped_regions=<s>
// lu_panels=128:<a>,512:<b>,other:<c> (PANEL-IMPL: the host width chosen per threaded ring2k smith / howell LU call; under
// ENGINE_DEVICE=gpu a call the device takes runs at kDevicePanel instead -- read EFFECTIVE GPU_SMITH2K panels=).  mode is
// what the descent's (unscoped) calls run: threaded iff the lever is on and threads > 1; scoped_regions counts the
// regions run under a validation's ScopedLever / ScopedThreads override; (PAR-F2) both lines also carry f2=threaded|serial
// f2_source=env|default, the exit line f2_lazy=<lazy echelons> f2_par_searches=<split searches / window updates> and
// t_s=<phase>:<wall seconds>,... (PhaseTimer below; diagnostic) (so a validation inside a HECKE_DESCENT_SMITH_MT=0
// process does not look like a broken lever).  Neither the lever, the thread count nor the panel can change a value
// (DescentSmithMtValidation).

#include "../ModArithmetic/EngineCheck.h"
#include "../ModArithmetic/EnvLever.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace engine {
namespace smithpar {

constexpr uint32_t kMaxThreads = 256;        // ENGINE_THREADS is clamped to this
constexpr uint32_t kDefaultThreadCap = 64;   // the default (affinity) count is capped to this: beyond, set ENGINE_THREADS

uint32_t default_threads();                   // the process's descent thread budget (see THE THREAD BUDGET); >= 1
const char* default_threads_source();         // "env", "affinity", "hardware" or "env-invalid"
// (PANEL-IMPL) THE PANEL above.  The threaded route's LU widths:
constexpr uint32_t kDefaultPanel = 128;       // ring3k; ring2k below kLargePanelRows rows
constexpr uint32_t kLargePanel = 512;         // ring2k at >= kLargePanelRows rows
constexpr uint32_t kLargePanelRows = 12000;
constexpr uint32_t kDevicePanel = 128;        // the ring2k device LU (Ring2kDeviceLU.h), whatever the size or the knob
uint32_t explicit_panel();                    // ScopedPanel, else a valid HECKE_DESCENT_SMITH_PANEL; 0 = auto
uint32_t descent_panel();                     // explicit_panel(), else kDefaultPanel (the ring3k width)
uint32_t descent_panel_2k(uint32_t rows);     // explicit_panel(), else 512 at rows >= kLargePanelRows, else 128

bool     lever_enabled();                     // HECKE_DESCENT_SMITH_MT (or the scoped override)
uint32_t threads();                           // threads a region on THIS thread would use (1 = serial)
uint64_t min_work();                          // the work (element multiply-adds) below which a step stays serial

// (slot-24 PAR-F2) The F_2 / F_3 echelon of each valuation level (BitMatrix / TritMatrix) and the per-level passes.  The
// threaded route (f2_route(): the lever HECKE_DESCENT_SMITH_F2_MT is on, default, AND threads() > 1) runs binary::echelon_lazy
// (the lazy blocked M4RI / Gauss: the final matrix, rank and pivots are binary::echelon's, bit for bit) and the vectorised,
// threaded level passes; HECKE_DESCENT_SMITH_F2_MT=0 (read-only) restores the PAR-2K code verbatim.  Never a value.
bool f2_lever_enabled();
bool f2_route();

// (PAR-F2) Per-phase wall clocks of the Smith forms (diagnostic; summed over threads that call them; printed in the EFFECTIVE
// exit line as t_s=...).  Phases: the F_2 / F_3 matrix build, the F_2 / F_3 echelon, the other per-level passes (zero test,
// column permutation, Schur halving / division and compaction), the blocked LU.
enum class Phase : uint32_t { F2Pack = 0, F2Echelon, F3Pack, F3Echelon, LevelPass, LU, EchSearch, EchFlush, EchTable, Perm, Schur };
constexpr uint32_t kPhaseCount = 11;
void   add_phase_ns(Phase p, uint64_t ns);
double phase_seconds(Phase p);
void   count_f2_lazy_call();     // one echelon_lazy call (validation non-vacuity)
void   count_f2_par_search();    // one pivot search that went parallel
struct PhaseTimer {
    explicit PhaseTimer(Phase p);
    ~PhaseTimer();
    PhaseTimer(const PhaseTimer&) = delete;
    PhaseTimer& operator=(const PhaseTimer&) = delete;
private:
    Phase p_;
    uint64_t t0_;
};

// The LU panel a ring3k call runs: descent_panel() on the threaded route, else the caller's Params panel.
inline uint32_t route_panel(uint32_t serial_panel) { return threads() > 1 ? descent_panel() : serial_panel; }
// (PANEL-IMPL) The host LU panel of a ring2k call on a matrix of `rows` rows: descent_panel_2k(rows) on the threaded route
// (counted in lu_panels=), else the caller's Params panel.
uint32_t route_panel_2k(uint32_t serial_panel, uint32_t rows);
// (PANEL-IMPL) The ring2k device LU's panel: kDevicePanel on the threaded route, else the caller's Params panel (as before).
inline uint32_t route_device_panel(uint32_t serial_panel) { return threads() > 1 ? kDevicePanel : serial_panel; }

// Runs fn(0) .. fn(ntasks - 1), each exactly once, on up to threads() threads (the caller included).  The tasks must
// write disjoint memory.  The first exception thrown by a task is rethrown after all tasks have stopped.
void parallel_for(uint32_t ntasks, const std::function<void(uint32_t)>& fn);
// The same on up to `nthreads` threads, without the ScopedEnable opt-in (PAR-D2's maps::SmithForm, which has its own lever
// and an explicit thread count).  Serial when nthreads <= 1, ntasks == 1, inside a task, or while the pool is busy.
void parallel_for_n(uint32_t nthreads, uint32_t ntasks, const std::function<void(uint32_t)>& fn);

// body(i0, i1) over contiguous row ranges covering [0, rows): one call when threads() == 1 or work < 2 min_work(), else
// up to 4 threads() tiles.  For row-independent passes (each row written only by its own range).
inline void for_rows(uint32_t rows, uint64_t work, const std::function<void(uint32_t, uint32_t)>& body) {
    const uint32_t T = threads();
    const uint64_t mw = min_work();
    if (T <= 1 || rows < 2 || work < 2 * mw) { body(0, rows); return; }
    uint64_t want = 4ull * T;
    if (mw > 0 && work / mw < want) want = work / mw;
    const uint32_t tiles = static_cast<uint32_t>(want < rows ? want : rows);
    if (tiles < 2) { body(0, rows); return; }
    parallel_for(tiles, [&](uint32_t t) {
        const uint32_t a = static_cast<uint32_t>(static_cast<uint64_t>(rows) * t / tiles);
        const uint32_t b = static_cast<uint32_t>(static_cast<uint64_t>(rows) * (t + 1) / tiles);
        if (b > a) body(a, b);
    });
}

// The opt-in: a Smith call inside this scope (on this thread) may use threads.  Nests.
struct ScopedEnable {
    ScopedEnable();
    ~ScopedEnable();
    ScopedEnable(const ScopedEnable&) = delete;
    ScopedEnable& operator=(const ScopedEnable&) = delete;
};

// Validation overrides (process-wide, not per thread).  ScopedThreads(0) = the default count.  ScopedLever(on) overrides
// HECKE_DESCENT_SMITH_MT.  ScopedMinWork(w) lowers the size threshold so that small matrices split too.
struct ScopedThreads {
    explicit ScopedThreads(uint32_t n);
    ~ScopedThreads();
    ScopedThreads(const ScopedThreads&) = delete;
    ScopedThreads& operator=(const ScopedThreads&) = delete;
private:
    uint32_t saved_;
};
// (S24 CL-5) the two on/off levers are engine::lever::EnvLever objects (EnvLever.h); the Scoped structs are its RAII override.
const lever::EnvLever& mt_lever();   // HECKE_DESCENT_SMITH_MT
const lever::EnvLever& f2_lever();   // HECKE_DESCENT_SMITH_F2_MT
struct ScopedLever : lever::EnvLever::Scoped {
    explicit ScopedLever(bool on) : Scoped(mt_lever(), on) {}
};
// (PAR-F2) ScopedF2Lever(on) overrides HECKE_DESCENT_SMITH_F2_MT.
struct ScopedF2Lever : lever::EnvLever::Scoped {
    explicit ScopedF2Lever(bool on) : Scoped(f2_lever(), on) {}
};
// ScopedPanel(w) overrides descent_panel() (0 = the default / HECKE_DESCENT_SMITH_PANEL).
struct ScopedPanel {
    explicit ScopedPanel(uint32_t w);
    ~ScopedPanel();
    ScopedPanel(const ScopedPanel&) = delete;
    ScopedPanel& operator=(const ScopedPanel&) = delete;
private:
    uint32_t saved_;
};
struct ScopedMinWork {
    explicit ScopedMinWork(uint64_t w);
    ~ScopedMinWork();
    ScopedMinWork(const ScopedMinWork&) = delete;
    ScopedMinWork& operator=(const ScopedMinWork&) = delete;
private:
    uint64_t saved_;
};

struct Stats {
    uint64_t regions_parallel = 0, regions_serial = 0, tasks = 0, panel_splits = 0;
    uint64_t regions_scoped = 0;   // parallel regions run under a ScopedLever / ScopedThreads override (validations)
    uint64_t regions_parallel_n = 0;   // of regions_parallel: those of parallel_for_n (PAR-D2's maps::SmithForm)
    uint64_t f2_lazy_calls = 0, f2_par_searches = 0;   // (PAR-F2) echelon_lazy calls; pivot searches split over the pool
    uint32_t max_threads = 1;
};
void  count_panel_split();      // one split panel update (diagnostic; the validation's non-vacuity check)
Stats stats();
std::string effective_text(const char* phase = "now");   // the EFFECTIVE line (without a trailing newline)

// Tiling helper: split [0, n) into `parts` contiguous ranges, range t = [n t / parts, n (t + 1) / parts), each boundary
// rounded down to a multiple of `align` (the last range ends at n).
inline uint32_t split_at(uint32_t n, uint32_t parts, uint32_t t, uint32_t align) {
    if (t >= parts) return n;
    uint32_t x = static_cast<uint32_t>((static_cast<uint64_t>(n) * t) / parts);
    if (align > 1) x -= x % align;
    return x;
}

// (slot-24 PAR-F2) The Schur compaction of a valuation level, threaded.  Row i (0 <= i < sr) of the result is the sc
// elements at base + (r0 + i) n0 (the front of source row r0 + i, already halved / divided in place), moved to base + i sc.
// Destination index <= source index, and dst(i) ends at (i + 1) sc <= (r0 + i + 1) n0 = the start of source row i + 1.
// Waves of rows [a, b) in ascending order: when b sc <= (r0 + a) n0 every destination of the wave lies below every unread
// source, and the rows are copied directly over row tiles; otherwise (few pivots: the shift is small) the wave's sources
// are first copied to a buffer (<= 32 MiB) and then to their destinations, both over row tiles -- a wave's destinations end
// at b sc <= the start of source row b, so no later source is touched.  The same bytes as the serial ascending memmove.
template<class E>
void compact_rows(E* base, uint32_t r0, uint32_t n0, uint32_t sr, uint32_t sc) {
    // (PAR-F2 audit 3.4) the destination-below-source argument needs sc <= n0 (both call sites pass sc = n0 - r0); O(1)
    ENGINE_CHECK_MSG(sc <= n0, "smithpar::compact_rows: the compacted width sc must not exceed the source stride n0");
    if (sr == 0 || sc == 0 || r0 == 0) return;   // r0 == 0: sc == n0, every row is already in place
    const size_t rowb = sizeof(E) * sc;
    const uint32_t bmax = static_cast<uint32_t>(std::max<size_t>(1, (size_t(32) << 20) / rowb));
    std::vector<E> buf;
    uint32_t a = 0;
    while (a < sr) {
        const uint64_t lim = (static_cast<uint64_t>(r0) + a) * n0 / sc;   // rows below lim: destinations under source row a
        if (lim >= static_cast<uint64_t>(a) + 64u) {
            const uint32_t b = static_cast<uint32_t>(std::min<uint64_t>(sr, lim));
            for_rows(b - a, static_cast<uint64_t>(b - a) * sc, [&](uint32_t i0, uint32_t i1) {
                for (uint32_t i = a + i0; i < a + i1; ++i)
                    std::memcpy(static_cast<void*>(base + static_cast<size_t>(i) * sc), base + static_cast<size_t>(r0 + i) * n0, rowb);
            });
            a = b;
            continue;
        }
        const uint32_t b = std::min<uint32_t>(sr, a + bmax);
        if (buf.size() < static_cast<size_t>(b - a) * sc) buf.resize(static_cast<size_t>(b - a) * sc);
        for_rows(b - a, static_cast<uint64_t>(b - a) * sc, [&](uint32_t i0, uint32_t i1) {
            for (uint32_t i = a + i0; i < a + i1; ++i)
                std::memcpy(static_cast<void*>(buf.data() + static_cast<size_t>(i - a) * sc), base + static_cast<size_t>(r0 + i) * n0, rowb);
        });
        for_rows(b - a, static_cast<uint64_t>(b - a) * sc, [&](uint32_t i0, uint32_t i1) {
            for (uint32_t i = a + i0; i < a + i1; ++i)
                std::memcpy(static_cast<void*>(base + static_cast<size_t>(i) * sc), buf.data() + static_cast<size_t>(i - a) * sc, rowb);
        });
        a = b;
    }
}

} // namespace smithpar
} // namespace engine
