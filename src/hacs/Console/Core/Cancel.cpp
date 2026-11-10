// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Cancel.cpp -- see Cancel.h.  (NBCONFIRM S5)
// ===========================================================================

#include "Cancel.h"

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <utility>

namespace console {

namespace {

static_assert(std::atomic<bool>::is_always_lock_free, "RequestCancel must be async-signal-safe");

std::atomic<bool> g_requested{false};

thread_local int t_armed = 0;        // CancelScope depth on this thread
thread_local int t_suppressed = 0;   // NoCancelScope depth on this thread

// The test interlock and the visit accounting (touched only by armed threads and the validation).
std::mutex g_mu;
std::atomic<bool> g_test_on{false};
CancelTest g_test;
bool g_test_fired = false;
std::function<void(std::uint64_t, const char*)> g_notifier;
std::uint64_t g_visits = 0;
std::uint64_t g_matched = 0;   // visits whose site matched the hook's prefix
bool g_trace = false;
std::vector<std::string> g_sites;

bool StartsWith(const char* s, const std::string& prefix) {
    if (prefix.empty()) return true;
    if (s == nullptr) return false;
    return std::string(s).rfind(prefix, 0) == 0;
}

bool ParseU64(const std::string& s, std::uint64_t& out) {
    if (s.empty() || s.size() > 18) return false;
    std::uint64_t v = 0;
    for (char c : s) {
        if (c < '0' || c > '9') return false;
        v = v * 10 + static_cast<std::uint64_t>(c - '0');
    }
    out = v;
    return true;
}

} // namespace

void RequestCancel() noexcept { g_requested.store(true, std::memory_order_relaxed); }
void ClearCancel() noexcept { g_requested.store(false, std::memory_order_relaxed); }
bool CancelRequested() noexcept { return g_requested.load(std::memory_order_relaxed); }

CancelScope::CancelScope() { ++t_armed; }
CancelScope::~CancelScope() { --t_armed; }
NoCancelScope::NoCancelScope() { ++t_suppressed; }
NoCancelScope::~NoCancelScope() { --t_suppressed; }

bool CancelArmedHere() noexcept { return t_armed > 0 && t_suppressed == 0; }

void PollCancel(const char* site) {
    if (!CancelArmedHere()) return;
    if (g_test_on.load(std::memory_order_relaxed)) {
        std::function<void(std::uint64_t, const char*)> notify;
        bool wait = false, hold = false;
        std::uint64_t k = 0;
        {
            std::lock_guard<std::mutex> lk(g_mu);
            ++g_visits;
            if (g_trace) g_sites.emplace_back(site == nullptr ? "" : site);
            if (!g_test_fired && g_test.mode != CancelTestMode::Off && StartsWith(site, g_test.prefix) && ++g_matched == g_test.k) {
                g_test_fired = true;
                g_test_on.store(g_trace, std::memory_order_relaxed);
                k = g_test.k;
                if (g_test.mode == CancelTestMode::After) RequestCancel();
                else {
                    wait = g_test.mode == CancelTestMode::Wait;
                    hold = g_test.mode == CancelTestMode::Hold;
                    notify = g_notifier;
                }
            }
        }
        if (wait || hold) {
            if (notify) notify(k, site);
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(wait ? 60 : 3);
            while ((hold || !CancelRequested()) && std::chrono::steady_clock::now() < deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }
    if (CancelRequested()) throw Interrupted{SourcePos(), site == nullptr ? std::string() : std::string(site)};
}

bool ParseCancelTest(const std::string& text, CancelTest& out) {
    out = CancelTest{};
    CancelTest t;
    std::string rest;
    if (text.rfind("after:", 0) == 0) {
        t.mode = CancelTestMode::After;
        rest = text.substr(6);
    } else if (text.rfind("wait:", 0) == 0) {
        t.mode = CancelTestMode::Wait;
        rest = text.substr(5);
    } else if (text.rfind("hold:", 0) == 0) {
        t.mode = CancelTestMode::Hold;
        rest = text.substr(5);
    } else {
        return false;
    }
    const std::size_t colon = rest.find(':');
    const std::string ks = rest.substr(0, colon);
    if (!ParseU64(ks, t.k) || t.k == 0) return false;
    if (colon != std::string::npos) {
        t.prefix = rest.substr(colon + 1);
        for (char c : t.prefix)
            if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '.')) return false;
    }
    out = t;
    return true;
}

void SetCancelTest(const CancelTest& t) {
    std::lock_guard<std::mutex> lk(g_mu);
    g_test = t;
    g_test_fired = false;
    g_matched = 0;
    g_test_on.store(t.mode != CancelTestMode::Off || g_trace, std::memory_order_relaxed);
}

CancelTest GetCancelTest() {
    std::lock_guard<std::mutex> lk(g_mu);
    return g_test;
}

bool CancelTestFired() {
    std::lock_guard<std::mutex> lk(g_mu);
    return g_test_fired;
}

void SetCancelWaitNotifier(std::function<void(std::uint64_t, const char*)> fn) {
    std::lock_guard<std::mutex> lk(g_mu);
    g_notifier = std::move(fn);
}

std::uint64_t CancelVisits() {
    std::lock_guard<std::mutex> lk(g_mu);
    return g_visits;
}

void ResetCancelVisits() {
    std::lock_guard<std::mutex> lk(g_mu);
    g_visits = 0;
    g_sites.clear();
}

void SetCancelTrace(bool on) {
    std::lock_guard<std::mutex> lk(g_mu);
    g_trace = on;
    g_test_on.store(on || (g_test.mode != CancelTestMode::Off && !g_test_fired), std::memory_order_relaxed);
}

std::vector<std::string> CancelTraceSites() {
    std::lock_guard<std::mutex> lk(g_mu);
    return g_sites;
}

} // namespace console
