// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (slot-24 PAR-GPU) The policy of the ring2k device LU and the installation of its operations table.  See DescentSmithGpu.h.

#include "DescentSmithGpu.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../../../ModArithmetic/Binary/Ring2kDevice.h"
#include "../../../ModArithmetic/Binary/SmithPar.h"
#include "../../../Cuda/CudaCtx.h"
#include "../../../Cuda/CudaRing2k.h"
#include "../../../Engine/EngineRuntime.h"
#include "../../../Engine/Program.h"
#include "../../../ModArithmetic/EnvLever.h"

namespace integral_descent {
namespace smith_gpu {

namespace {

namespace dev = engine::ring2k::device;
namespace cr = engine::cuda::ring2k;

std::atomic<int> g_force{-1};          // -1: the policy; 0: never; 1: always (with a device)
std::atomic<uint32_t> g_max_cols{0};
std::atomic<bool> g_force_verify{false};
std::atomic<uint64_t> g_store_bytes{0};   // validation override of the tiled route's L-store budget (0 = the default)
std::atomic<int> g_simulate{0};
std::atomic<uint64_t> g_policy_calls{0}, g_declined_vram{0}, g_declined_host{0};

// MemAvailable of /proc/meminfo in bytes (0 = unknown: no check).
uint64_t host_mem_available() {
    std::FILE* f = std::fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    uint64_t kb = 0;
    while (std::fgets(line, sizeof line, f))
        if (std::sscanf(line, "MemAvailable: %llu kB", reinterpret_cast<unsigned long long*>(&kb)) == 1) break;
    std::fclose(f);
    return kb << 10;
}

// (S24 CL-5) the two on/off levers on EnvLever.  ENGINE_GPU_SMITH2K: "" = unset; exactly 0 = off; 1 = on; anything else warns
// once and stays on.  ENGINE_GPU_SMITH2K_VERIFY: opt-in, on iff the value starts with '1' (Match::FirstChar), silently.
engine::lever::EnvLever g_lever("ENGINE_GPU_SMITH2K",
                                engine::lever::Policy{true, engine::lever::Empty::Unset, engine::lever::Match::Exact01, engine::lever::Cache::Once,
                                                      "WARNING: ENGINE_GPU_SMITH2K=%s is neither 0 nor 1; the device Smith LU stays on (under "
                                                      "ENGINE_DEVICE=gpu)\n"});
bool lever_set() {   // set and non-empty (no reading: the WARNING stays at the first lever_on())
    const char* e = g_lever.raw();
    return e && *e;
}

bool env_verify() {
#if defined(ENGINE_VERIFY_DEVICE) && ENGINE_VERIFY_DEVICE
    return true;
#else
    static const engine::lever::EnvLever verify(
        "ENGINE_GPU_SMITH2K_VERIFY",
        engine::lever::Policy{false, engine::lever::Empty::Unset, engine::lever::Match::FirstChar, engine::lever::Cache::Once, nullptr});
    return verify.env_enabled();
#endif
}

// The tiled route's L-store budget per phase: ENGINE_GPU_SMITH2K_HOST_MB (default 4096 MiB).
uint64_t env_store_budget() {
    static const uint64_t v = [] {
        const char* e = std::getenv("ENGINE_GPU_SMITH2K_HOST_MB");
        const uint64_t mb = (e && *e) ? static_cast<uint64_t>(std::strtoull(e, nullptr, 10)) : uint64_t(4096);
        return (mb == 0 ? uint64_t(4096) : mb) << 20;
    }();
    return v;
}
uint64_t op_store_budget() {
    const uint64_t o = g_store_bytes.load(std::memory_order_relaxed);
    return o != 0 ? o : env_store_budget();
}

uint64_t env_vram_cap() {
    static const uint64_t v = [] {
        const char* e = std::getenv("ENGINE_GPU_SMITH2K_VRAM_MB");
        return (e && *e) ? static_cast<uint64_t>(std::strtoull(e, nullptr, 10)) << 20 : uint64_t(0);
    }();
    return v;
}

// The device bytes a session may take: free VRAM - 512 MiB - 1/32 of the card, capped by ENGINE_GPU_SMITH2K_VRAM_MB.  False when
// the device cannot be queried.
bool vram_budget(uint64_t& budget) {
    cr::MemInfo mi;
    try {
        mi = cr::mem_info();
    } catch (const std::exception&) {
        return false;
    }
    const uint64_t reserve = (uint64_t(512) << 20) + mi.total_bytes / 32;
    budget = mi.free_bytes > reserve ? mi.free_bytes - reserve : 0;
    if (env_vram_cap() != 0 && budget > env_vram_cap()) budget = env_vram_cap();
    return true;
}

uint32_t op_plan(uint32_t elem, uint32_t m, uint32_t n, uint32_t panel) {
    const int f = g_force.load(std::memory_order_relaxed);
    if (f == 0) return 0;
    if (f != 1) {
        if (static_cast<uint64_t>(m) * n < min_entries() || !lever_on()) return 0;
        if (engine::tape::elim_pass_options().effective_policy() != engine::tape::DevicePolicy::Gpu) return 0;
        g_policy_calls.fetch_add(1, std::memory_order_relaxed);
    }
    if (!engine::cuda::available()) return 0;
    uint64_t budget = 0;
    if (!vram_budget(budget)) return 0;
    uint32_t cw = 0;
    if (cr::session_bytes(elem, m, n, panel) <= budget) {
        cw = n;
    } else {
        const uint64_t base = cr::session_bytes(elem, m, 0, panel);
        const uint64_t per_col = static_cast<uint64_t>(m) * elem;
        uint64_t c = budget > base ? (budget - base) / per_col : 0;
        c -= c % panel;
        cw = c >= n ? n : static_cast<uint32_t>(c);
        if (cw < n && cw < 8 * panel) {   // fewer than eight panels per slab: too many replays; the host LU
            g_declined_vram.fetch_add(1, std::memory_order_relaxed);
            return 0;
        }
    }
    const uint32_t mc = g_max_cols.load(std::memory_order_relaxed);
    if (mc > 0 && mc < n && cw > mc) cw = mc - mc % panel < panel ? panel : mc - mc % panel;
    if (cw < n) {
        // the tiled route keeps one phase's L factors on the host: at most op_store_budget() bytes (plus one panel)
        const uint64_t R = std::min(m, n);
        const uint64_t store = std::min<uint64_t>((static_cast<uint64_t>(m) * R - R * R / 2) * elem,
                                                  op_store_budget() + static_cast<uint64_t>(m) * panel * elem);
        const uint64_t avail = host_mem_available();
        if (avail != 0 && avail < store + (uint64_t(2) << 30)) {
            g_declined_host.fetch_add(1, std::memory_order_relaxed);
            return 0;
        }
    }
    return cw;
}

void* op_open(uint32_t elem, uint32_t m, uint32_t cw, uint32_t panel) {
    if (g_simulate.load(std::memory_order_relaxed) > 0)
        throw engine::cuda::CudaError("simulated CudaError (DescentSmithGpu ScopedSimulateError)");
    return cr::open(elem, m, cw, panel);
}

bool op_verify() { return g_force_verify.load(std::memory_order_relaxed) || env_verify(); }

void print_effective_at_exit() { std::fprintf(stderr, "%s\n", effective_text().c_str()); }

const dev::Ops& table() {
    static const dev::Ops t = [] {
        dev::Ops o;
        o.plan = &op_plan;
        o.open = &op_open;
        o.put = &cr::put;
        o.get = &cr::get;
        o.step = &cr::step;
        o.fetch = &cr::fetch;
        o.close = &cr::close;
        o.verify = &op_verify;
        o.store_budget = &op_store_budget;
        return o;
    }();
    return t;
}

const bool g_installed = [] {
    dev::set_ops(&table());
    const char* d = std::getenv("ENGINE_DEVICE");
    if (lever_set() || (d && std::strcmp(d, "gpu") == 0)) std::atexit(&print_effective_at_exit);
    return true;
}();

} // namespace

bool installed() { return g_installed && dev::ops() == &table(); }

bool lever_on() { return g_lever.env_enabled(); }

uint64_t min_entries() {
    static const uint64_t v = [] {
        const char* e = std::getenv("ENGINE_GPU_SMITH2K_MIN");
        return (e && *e) ? static_cast<uint64_t>(std::strtoull(e, nullptr, 10)) : (uint64_t(1) << 24);
    }();
    return v;
}

std::string effective_text() {
    const dev::Stats s = dev::stats();
    char buf[512];
    std::snprintf(buf, sizeof buf,
                  "EFFECTIVE GPU_SMITH2K mode=%s policy_gpu_calls=%llu device=%llu resident=%llu tiled=%llu slabs=%llu replays=%llu "
                  "fallbacks=%llu declined_vram=%llu declined_host=%llu panels=%llu lookahead=%llu verified=%llu min_entries=%llu",
                  lever_on() ? "on" : "off", static_cast<unsigned long long>(g_policy_calls.load()),
                  static_cast<unsigned long long>(s.device), static_cast<unsigned long long>(s.resident),
                  static_cast<unsigned long long>(s.tiled), static_cast<unsigned long long>(s.slabs),
                  static_cast<unsigned long long>(s.replays), static_cast<unsigned long long>(s.fallbacks),
                  static_cast<unsigned long long>(g_declined_vram.load()), static_cast<unsigned long long>(g_declined_host.load()),
                  static_cast<unsigned long long>(s.panels),
                  static_cast<unsigned long long>(s.lookaheads), static_cast<unsigned long long>(s.verified),
                  static_cast<unsigned long long>(min_entries()));
    char t[200];
    auto sec = [](uint64_t ns) { return static_cast<double>(ns) * 1e-9; };
    std::snprintf(t, sizeof t, " t_s=open:%.2f,fetch:%.2f,panel:%.2f,step:%.2f,host:%.2f,close:%.2f", sec(s.ns_open), sec(s.ns_fetch),
                  sec(s.ns_panel), sec(s.ns_step), sec(s.ns_host), sec(s.ns_close));
    return std::string(buf) + t;
}

uint64_t PredictHostRssBytes(uint64_t m, uint64_t n, uint32_t elem) {
    if (m == 0 || n == 0) return 0;
    const int f = g_force.load(std::memory_order_relaxed);
    if (f == 0) return 0;
    if (f != 1) {
        if (m * n < min_entries() || !lever_on()) return 0;
        if (engine::tape::elim_pass_options().effective_policy() != engine::tape::DevicePolicy::Gpu) return 0;
    }
    if (!engine::cuda::available()) return 0;
    // (PANEL-IMPL) the device LU's panel (smithpar::kDevicePanel on the threaded route; the serial route's 64 is narrower), no
    // longer the host LU's: the host size rule (512 from 12 000 rows) and HECKE_DESCENT_SMITH_PANEL do not reach the device route.
    const uint64_t panel = engine::smithpar::kDevicePanel;
    const uint64_t R = std::min(m, n);
    const uint64_t whole_L = (m * R - R * R / 2) * elem;
    // (PANEL-IMPL, PANEL_report.md Sec. 6.3) the host L store is used by the tiled route only: when op_plan would keep this LU
    // resident (the whole window fits the VRAM budget now, no validation column cap), the store term is dropped.  The query is
    // made on the bound (m, n), at least as large as every LU it covers; a later tiling (less free VRAM at run time) is still
    // refused by op_plan's own MemAvailable check.
    bool resident = false;
    {
        const uint32_t mc = g_max_cols.load(std::memory_order_relaxed);
        uint64_t budget = 0;
        if (m <= UINT32_MAX && n <= UINT32_MAX && !(mc > 0 && mc < n) && vram_budget(budget))
            resident = cr::session_bytes(elem, static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<uint32_t>(panel)) <= budget;
    }
    const uint64_t store = resident ? 0 : std::min<uint64_t>(whole_L, op_store_budget() + m * panel * elem);
    const uint64_t buffers = 2 * m * panel * elem;
    const uint64_t staging = static_cast<uint64_t>(cr::staging_bytes(elem, static_cast<uint32_t>(std::min<uint64_t>(m, UINT32_MAX)),
                                                                      static_cast<uint32_t>(panel)));
    const uint64_t twin = op_verify() ? m * n * elem : 0;
    return store + buffers + staging + twin;
}

ScopedForce::ScopedForce(bool on, uint32_t max_cols, bool verify, uint64_t store_bytes)
    : saved_on_(g_force.exchange(on ? 1 : 0)), saved_cols_(g_max_cols.exchange(max_cols)), saved_verify_(g_force_verify.exchange(verify)),
      saved_store_(g_store_bytes.exchange(store_bytes)) {}
ScopedForce::~ScopedForce() {
    g_force.store(saved_on_);
    g_max_cols.store(saved_cols_);
    g_force_verify.store(saved_verify_);
    g_store_bytes.store(saved_store_);
}

ScopedSimulateError::ScopedSimulateError() { g_simulate.fetch_add(1); }
ScopedSimulateError::~ScopedSimulateError() { g_simulate.fetch_sub(1); }

} // namespace smith_gpu
} // namespace integral_descent
