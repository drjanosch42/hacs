// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (slot-24 PAR-GPU) The device-operations table of the ring2k LU and its counters.  See Ring2kDevice.h.

#include "Ring2kDevice.h"

#include <atomic>

namespace engine {
namespace ring2k {
namespace device {

namespace {
std::atomic<const Ops*> g_ops{nullptr};
thread_local uint32_t tl_host_only = 0;
struct Counters {
    std::atomic<uint64_t> calls{0}, device{0}, resident{0}, tiled{0}, slabs{0}, replays{0}, fallbacks{0}, panels{0}, lookaheads{0}, verified{0};
    std::atomic<uint64_t> ns_open{0}, ns_fetch{0}, ns_panel{0}, ns_step{0}, ns_host{0}, ns_close{0};
};
Counters& ctr() {
    static Counters c;
    return c;
}
} // namespace

void       set_ops(const Ops* o) { g_ops.store(o, std::memory_order_release); }
const Ops* ops() { return g_ops.load(std::memory_order_acquire); }

Stats stats() {
    Counters& c = ctr();
    Stats s;
    s.calls = c.calls.load();
    s.device = c.device.load();
    s.resident = c.resident.load();
    s.tiled = c.tiled.load();
    s.slabs = c.slabs.load();
    s.replays = c.replays.load();
    s.fallbacks = c.fallbacks.load();
    s.panels = c.panels.load();
    s.lookaheads = c.lookaheads.load();
    s.verified = c.verified.load();
    s.ns_open = c.ns_open.load();
    s.ns_fetch = c.ns_fetch.load();
    s.ns_panel = c.ns_panel.load();
    s.ns_step = c.ns_step.load();
    s.ns_host = c.ns_host.load();
    s.ns_close = c.ns_close.load();
    return s;
}

void note(uint64_t Stats::*field, uint64_t v) {
    Counters& c = ctr();
    std::atomic<uint64_t>* a = nullptr;
    if (field == &Stats::calls) a = &c.calls;
    else if (field == &Stats::device) a = &c.device;
    else if (field == &Stats::resident) a = &c.resident;
    else if (field == &Stats::tiled) a = &c.tiled;
    else if (field == &Stats::slabs) a = &c.slabs;
    else if (field == &Stats::replays) a = &c.replays;
    else if (field == &Stats::fallbacks) a = &c.fallbacks;
    else if (field == &Stats::panels) a = &c.panels;
    else if (field == &Stats::lookaheads) a = &c.lookaheads;
    else if (field == &Stats::verified) a = &c.verified;
    else if (field == &Stats::ns_open) a = &c.ns_open;
    else if (field == &Stats::ns_fetch) a = &c.ns_fetch;
    else if (field == &Stats::ns_panel) a = &c.ns_panel;
    else if (field == &Stats::ns_step) a = &c.ns_step;
    else if (field == &Stats::ns_host) a = &c.ns_host;
    else if (field == &Stats::ns_close) a = &c.ns_close;
    if (a) a->fetch_add(v, std::memory_order_relaxed);
}

ScopedHostOnly::ScopedHostOnly() { ++tl_host_only; }
ScopedHostOnly::~ScopedHostOnly() { --tl_host_only; }
bool host_only() { return tl_host_only > 0; }

} // namespace device
} // namespace ring2k
} // namespace engine
