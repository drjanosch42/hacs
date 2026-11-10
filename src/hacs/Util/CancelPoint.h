// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// CancelPoint.h -- (NBCONFIRM S5) the pipeline's neutral cancellation hook.
//
// The pipeline never depends on the console (check_console_includes.sh rule 2),
// so a long computation marks its cooperative cancellation points with
// oc_util::CancelPoint("<site>") and nothing else.  The hook is a process-wide
// function pointer, null by default (a no-op: campaigns, validations, --bench,
// the census drivers); the console bindings install console::PollCancel, which
// is itself a no-op unless the calling thread runs a REPL statement or a served
// cell (console::CancelScope) and a cancel was requested -- then it throws
// console::Interrupted.
//
// RULES for a point (Console/Core/Cancel.h): only on the thread that called the
// verb, between whole units of work; never inside a kernel, a parallel-region
// body, a destructor, a noexcept function or the builder of a cache entry, so
// that unwinding drops only stack-local partial results.  A point never reads
// or changes a value.
// ===========================================================================
#pragma once
#ifndef OC_UTIL_CANCEL_POINT_H
#define OC_UTIL_CANCEL_POINT_H

#include <atomic>

namespace oc_util {

using CancelPollFn = void (*)(const char* site);

inline std::atomic<CancelPollFn> g_cancel_poll{nullptr};

inline void SetCancelPoll(CancelPollFn fn) { g_cancel_poll.store(fn, std::memory_order_release); }

inline void CancelPoint(const char* site) {
    if (const CancelPollFn fn = g_cancel_poll.load(std::memory_order_acquire)) fn(site);
}

} // namespace oc_util

#endif // OC_UTIL_CANCEL_POINT_H
