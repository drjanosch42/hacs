// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// PAR-ASM: the levers, the worker count and the two fork-join primitives of the descent's assembly stages.  See the header.
#include "DescentParAsm.h"

#include "DescentSmith.h"
#include "../../../ModArithmetic/Binary/ParWorker.h"
#include "../../../Engine/EngineRuntime.h"
#include "../../../Engine/Threads.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <sstream>
#include <system_error>
#include <thread>
#include <vector>

namespace integral_descent {
namespace par {

namespace {

// (S24 CL-5) the four levers on EnvLever: "" = unset; exactly 0 = off; 1 = on; anything else warns once and reads as the default
// (source=default).  All four are read together, in this order, at the first consultation of any (as before).
#define PARASM_LEVER(NAME)                                                                                              \
    {NAME, engine::lever::Policy{true, engine::lever::Empty::Unset, engine::lever::Match::Exact01, engine::lever::Cache::Once, \
                                 "WARNING: " NAME "=%s is neither 0 nor 1; the lever stays on (default)\n", "default", "env", "default"}}
constexpr int kLevers = 4;
engine::lever::EnvLever g_levers[kLevers] = {PARASM_LEVER("HECKE_DESCENT_PAR_ASM"), PARASM_LEVER("HECKE_DESCENT_PAR_ELL"),
                                             PARASM_LEVER("HECKE_DESCENT_EMBED_MAP"), PARASM_LEVER("HECKE_DESCENT_PAR_LN")};
#undef PARASM_LEVER
std::atomic<uint32_t> g_threads_scoped{0};

const engine::lever::EnvLever& read_all(int i) {
    static const bool once = [] {
        for (const auto& l : g_levers) l.env_state();
        return true;
    }();
    (void)once;
    return g_levers[i];
}

std::atomic<uint64_t> c_row{0}, c_index{0}, c_embed{0}, c_ell_conc{0}, c_ell_rss{0}, c_ln_conc{0}, c_ln_rss{0};
// (audit F4) the worker flag is engine::parworker (src/ModArithmetic/Binary/ParWorker.h), shared with every inner threaded layer:
// on a worker of a region, nested ForEachIndex / ForRowBlocks, the descent Smith form, maps::MulMod / ProductsEqual and the
// ring2k / ring3k pool all run serially (no oversubscription, no nested thread creation)
using DepthGuard = engine::parworker::Scope;

}  // namespace

bool InRegion() { return engine::parworker::active(); }

const engine::lever::EnvLever& LeverOf(Lever l) { return g_levers[static_cast<int>(l)]; }
bool Enabled(Lever l) { return read_all(static_cast<int>(l)).enabled(); }
std::string Source(Lever l) { return read_all(static_cast<int>(l)).source(); }

uint32_t Threads() {
    const uint32_t s = g_threads_scoped.load();
    if (s) return s;
    static const uint32_t t = engine::threads::configured(engine::threads::kDescentParAsm);   // (S24 CL-5, C-05) cap 256
    return t;
}
ScopedThreads::ScopedThreads(uint32_t t) : prev_(g_threads_scoped.exchange(std::max(1u, std::min(256u, t)))) {}
ScopedThreads::~ScopedThreads() { g_threads_scoped.store(prev_); }

bool DeviceRequested() {
    // (audit F3) a thread-local device override (ScopedThreadDevice, Cpu or Gpu) counts too: the caller asked for a specific device
    // route, which is not proven re-entrant, so the concurrent regions stay serial
    if (engine::tape::thread_device_override() != engine::tape::ThreadDevice::Inherit) return true;
    const char* e = std::getenv("ENGINE_DEVICE");
    return e && *e && std::strcmp(e, "cpu") != 0;
}

void ForEachIndex(size_t n, uint32_t workers, const std::function<void(size_t)>& fn) {
    if (n == 0) return;
    // (audit F2) while a validation capture is installed (maps::SetSmithCapture / SetCallerCapture) the region runs in index order on
    // the caller, so the captured sequence is the serial one (and the capture vectors are never pushed concurrently)
    const bool serial = engine::parworker::active() || maps::CaptureActive();
    const uint32_t w = serial ? 1u : static_cast<uint32_t>(std::min<size_t>(n, std::max(1u, workers)));
    if (w <= 1) {
        for (size_t i = 0; i < n; ++i) fn(i);
        return;
    }
    c_index.fetch_add(1);
    std::atomic<size_t> next{0};
    std::atomic<size_t> lowest_throw{n};
    std::vector<std::exception_ptr> errs(n);
    auto body = [&] {
        DepthGuard g;
        for (;;) {
            const size_t i = next.fetch_add(1);
            if (i >= n) return;
            if (i > lowest_throw.load()) return;   // the serial loop would not have reached i
            try {
                fn(i);
            } catch (...) {
                errs[i] = std::current_exception();
                size_t cur = lowest_throw.load();
                while (i < cur && !lowest_throw.compare_exchange_weak(cur, i)) {}
            }
        }
    };
    // (audit F3) every worker runs under the caller's full registered thread context (engine::tape::capture_thread_context(): the
    // mathematical settings, the consumer pool, the thread device override, elim::ScopedLegacy, the consumers' Scoped* options),
    // exactly as the serial loop on the caller would
    const std::vector<engine::tape::ThreadContextInstaller> ctx = engine::tape::capture_thread_context();
    std::vector<std::thread> th;
    th.reserve(w - 1);
    // (audit F4) a failed thread creation (std::system_error, e.g. a pids limit) is not fatal: the region goes on with the workers
    // already started (at worst the caller alone), which hand out the same indices -- never std::terminate on a joinable thread
    try {
        for (uint32_t k = 1; k < w; ++k)
            th.emplace_back([&] {
                std::vector<engine::tape::ThreadContextGuard> guards;
                guards.reserve(ctx.size());
                for (const auto& install : ctx) guards.push_back(install());
                body();
            });
    } catch (const std::system_error&) {
    }
    body();
    for (auto& t : th) t.join();
    const size_t lo = lowest_throw.load();
    if (lo < n) std::rethrow_exception(errs[lo]);
}

void ForRowBlocks(uint32_t rows, uint64_t work_per_row, const std::function<void(uint32_t, uint32_t)>& fn, uint64_t min_work) {
    if (rows == 0) return;
    const uint32_t T = Enabled(Lever::Asm) && !engine::parworker::active() ? Threads() : 1u;
    const uint64_t work = uint64_t(rows) * std::max<uint64_t>(1, work_per_row);
    if (T <= 1 || work < min_work || rows < 2) {
        fn(0, rows);
        return;
    }
    // blocks of at least min_work / 4 work, at most 4 per thread (load balance), at least one row
    const uint64_t per_block = std::max<uint64_t>(min_work / 4, (work + 4ull * T - 1) / (4ull * T));
    uint32_t brows = static_cast<uint32_t>(std::max<uint64_t>(1, per_block / std::max<uint64_t>(1, work_per_row)));
    brows = std::min(brows, rows);
    const uint32_t nb = (rows + brows - 1) / brows;
    if (nb <= 1) {
        fn(0, rows);
        return;
    }
    c_row.fetch_add(1);
    ForEachIndex(nb, T, [&](size_t b) {
        const uint32_t r0 = static_cast<uint32_t>(b) * brows;
        fn(r0, std::min(rows, r0 + brows));
    });
}

Counters GetCounters() {
    Counters c;
    c.row_regions = c_row.load();
    c.index_regions = c_index.load();
    c.embed_maps = c_embed.load();
    c.ell_concurrent = c_ell_conc.load();
    c.ell_serial_rss = c_ell_rss.load();
    c.ln_concurrent = c_ln_conc.load();
    c.ln_serial_rss = c_ln_rss.load();
    return c;
}
void NoteEmbedMap() { c_embed.fetch_add(1); }
void NoteEll(bool concurrent, bool rss_limited) {
    if (concurrent) c_ell_conc.fetch_add(1);
    if (rss_limited) c_ell_rss.fetch_add(1);
}
void NoteLn(bool concurrent, bool rss_limited) {
    if (concurrent) c_ln_conc.fetch_add(1);
    if (rss_limited) c_ln_rss.fetch_add(1);
}

std::string EffectiveLine() {
    const Counters c = GetCounters();
    std::ostringstream o;
    o << "EFFECTIVE HeckeDescentParAsm asm=" << (Enabled(Lever::Asm) ? "on" : "off") << "(" << Source(Lever::Asm) << ")"
      << " ell=" << (Enabled(Lever::Ell) ? "on" : "off") << "(" << Source(Lever::Ell) << ")"
      << " embed=" << (Enabled(Lever::Embed) ? "on" : "off") << "(" << Source(Lever::Embed) << ")"
      << " ln=" << (Enabled(Lever::EllN) ? "on" : "off") << "(" << Source(Lever::EllN) << ")"
      << " threads=" << Threads() << " device=" << (DeviceRequested() ? "requested(serial regions)" : "cpu")
      << " row_regions=" << c.row_regions << " index_regions=" << c.index_regions << " embed_maps=" << c.embed_maps
      << " ell_concurrent=" << c.ell_concurrent << " ell_rss_limited=" << c.ell_serial_rss
      << " ln_concurrent=" << c.ln_concurrent << " ln_rss_limited=" << c.ln_serial_rss;
    return o.str();
}

uint32_t JointWorkers(uint32_t items, uint64_t per_item_bytes, uint64_t cap_bytes, uint32_t threads) {
    if (items <= 1 || threads <= 1) return 1;
    uint64_t fit = per_item_bytes ? (cap_bytes / 10 * 9) / per_item_bytes : items;
    fit = std::max<uint64_t>(1, fit);
    return static_cast<uint32_t>(std::min<uint64_t>(fit, std::min(items, threads)));
}

}  // namespace par
}  // namespace integral_descent
