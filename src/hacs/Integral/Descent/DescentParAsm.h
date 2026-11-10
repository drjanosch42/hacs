// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_DESCENT_DESCENT_PAR_ASM_H
#define INTEGRAL_DESCENT_DESCENT_PAR_ASM_H

// ===========================================================================
// PAR-ASM (hecke-extensions slot 24; docs/implementation/slot24/PARASM_report.md): the remaining serial stages of the integral
// descent made parallel, deterministic and value-identical.
//
//   (a) the tree pair's complex assembly (BuildTreePairComplex: one task per root matrix d, d_B, a^*, then the periodic copies; the
//       certificates streamed through maps::ProductsEqual; pair::MatMulZ through maps::MulMod) and the cone maps of RunPairLes / D2's
//       DensePiece: the identity-like cochain maps j, proj, dl are passed to maps::InducedMapEmbed (a column block of Y.coord times a
//       column block of X.gens, no dense c x c matrix);
//   (b) the ell >= 5 Bredon-tree primes of the level-1 groups descent (AssembleDescent) run concurrently under a JOINT RSS
//       prediction (rule 19): at most floor(0.9 cap / per-prime prediction) at a time;
//   (c) the ell >= 5 pieces of AssembleDescentPair (the shipped E8 pair at level 1, the Bredon pair at level N) concurrently under
//       the same rule, after the ell <= 3 pieces.
//   (d) (PAR-LN, PARLN_report.md) AssembleDescentN's sparse ell >= 5 cokernels: the sparse eliminations (EliminateSparseCokernel) of
//       up to floor(0.9 cap / PredictSparseElimRss) primes at a time, then the dense residuals one after another in the serial order.
// NOT parallel (PARASM_report.md Sec. 5): the tree pair's ell' rank source.
//
// DETERMINISM.  Every parallel region writes disjoint outputs (rows of a product, one result slot per prime / piece) by the same
// exact ring operations as the serial code, then the results are consumed in the serial order.  Shared counters are atomic
// (maps' Smith-form count, these counters); the validation captures (DescentSmith.h) are mutex-guarded and, while installed, make
// every region run in index order.  An exception is re-thrown from the
// LOWEST index that threw -- the one the serial loop would have thrown first.  So no value, refusal or message depends on the thread
// count or the schedule.
//
// THE LEVERS (read-only, value-preserving; each read ONCE per process, a scoped override wins):
//   HECKE_DESCENT_PAR_ASM=0     every threaded path of (a) and (c) back to the serial code of record (verbatim);
//   HECKE_DESCENT_PAR_ELL=0     the ell >= 5 primes of (b) one after another (verbatim);
//   HECKE_DESCENT_EMBED_MAP=0   the identity-like maps built as dense c x c matrices and passed to maps::InducedMap (verbatim).
//   HECKE_DESCENT_PAR_LN=0      the ell >= 5 primes of (d) one after another, each eliminated and finished before the next (serial).
// Exactly "0" turns a lever off; "1" or unset keeps it on; any other value warns once and keeps it on.
// Threads: ENGINE_THREADS (else the hardware concurrency), at most 256; ENGINE_THREADS=1 is the serial route.  With ENGINE_DEVICE
// set to anything but cpu, or a thread-local engine device override (ScopedThreadDevice) installed, the concurrent regions of (b),
// (c) run serially (the device route is not proven re-entrant).
//
// WORKERS (audit F3, F4).  Every worker installs the caller's full registered thread context (engine::tape::capture_thread_context():
// the mathematical settings, the consumer pool, the device override, elim::ScopedLegacy, the consumers' Scoped* options) and is
// marked by engine::parworker (src/ModArithmetic/Binary/ParWorker.h): on it every inner threaded layer -- nested regions, the descent
// Smith form, maps::MulMod / ProductsEqual, the ring2k / ring3k pool -- runs serially.  A failed thread creation shrinks the region
// to the workers started (at worst the caller alone); it is never fatal.
// ===========================================================================

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

#include "../../../ModArithmetic/EnvLever.h"

namespace integral_descent {
namespace par {

enum class Lever { Asm = 0, Ell = 1, Embed = 2, EllN = 3 };

bool Enabled(Lever l);
std::string Source(Lever l);          // "default", "env" or "scoped"

// (S24 CL-5) each lever is an engine::lever::EnvLever (EnvLever.h); ScopedLever is its RAII override
const engine::lever::EnvLever& LeverOf(Lever l);
struct ScopedLever : engine::lever::EnvLever::Scoped {
    ScopedLever(Lever l, bool on) : Scoped(LeverOf(l), on) {}
};

// The worker count: a scoped override (validation), else ENGINE_THREADS, else the hardware; at most 256, at least 1.
uint32_t Threads();
class ScopedThreads {
public:
    explicit ScopedThreads(uint32_t t);
    ~ScopedThreads();
    ScopedThreads(const ScopedThreads&) = delete;
    ScopedThreads& operator=(const ScopedThreads&) = delete;
private:
    uint32_t prev_;
};
// true when ENGINE_DEVICE names a device policy other than cpu, or a thread device override is installed on this thread (the
// concurrent regions of (b), (c) then stay serial)
bool DeviceRequested();

// true on a worker of a running ForEachIndex / ForRowBlocks region (= engine::parworker::active(); nested regions and every inner
// threaded layer then stay serial: no oversubscription)
bool InRegion();

// fn(i) for every i in [0, n), each exactly once, on min(workers, n) threads (the caller is one of them); workers <= 1: in order on
// the caller.  Indices are handed out in increasing order.  If any call throws, the exception of the LOWEST throwing index is
// re-thrown after every worker has stopped (indices above the lowest throwing one may or may not have run).
void ForEachIndex(size_t n, uint32_t workers, const std::function<void(size_t)>& fn);

// fn(r0, r1) on contiguous row blocks covering [0, rows); serial (one call) unless rows * work_per_row >= min_work and the Asm
// lever is on.  The blocks are disjoint, so a writer of rows [r0, r1) is deterministic.
void ForRowBlocks(uint32_t rows, uint64_t work_per_row, const std::function<void(uint32_t, uint32_t)>& fn,
                  uint64_t min_work = uint64_t(1) << 22);

// Counters (measurements; the validation reads them to see that the threaded paths ran).
struct Counters {
    uint64_t row_regions = 0;       // ForRowBlocks calls that ran on > 1 thread
    uint64_t index_regions = 0;     // ForEachIndex calls that ran on > 1 thread
    uint64_t embed_maps = 0;        // InducedMapEmbed calls on the structured path
    uint64_t ell_concurrent = 0;    // ell >= 5 loops run with > 1 prime at a time
    uint64_t ell_serial_rss = 0;    // ell >= 5 loops kept at 1 by the joint RSS prediction
    uint64_t ln_concurrent = 0;     // (PAR-LN audit A3) level-N sparse stages run with > 1 item at a time
    uint64_t ln_serial_rss = 0;     // (PAR-LN audit A3) level-N sparse stages limited by the joint RSS prediction
};
Counters GetCounters();
void NoteEmbedMap();
void NoteEll(bool concurrent, bool rss_limited);
// (PAR-LN (d)) the level-N sparse stage of AssembleDescentN: host-only items (no engine kernel, executor or device code), so it does
// not consult DeviceRequested() (audit A4); its joint rule counts the items' PredictSparseElimRss only -- not the base RSS nor the
// residuals retained until the serial consumption, which are small at current sizes (audit A5).  Counted apart (audit A3).
void NoteLn(bool concurrent, bool rss_limited);

// One `EFFECTIVE HeckeDescentParAsm ...` line (levers, sources, threads, counters).
std::string EffectiveLine();

// The joint RSS rule (b), (c): how many items may run at once.  per_item = the largest single-item prediction; cap = the ceiling;
// 0.9 cap / per_item, clamped to [1, min(threads, items)].
uint32_t JointWorkers(uint32_t items, uint64_t per_item_bytes, uint64_t cap_bytes, uint32_t threads);

}  // namespace par
}  // namespace integral_descent

#endif
