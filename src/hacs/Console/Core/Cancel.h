// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Cancel.h -- (NBCONFIRM S5) cooperative cancellation of a running statement.
//
// [P] Sec. 7 (REPL Ctrl-C) and Sec. 18.6 (the served `interrupt`).  ONE
// process-wide request flag, raised by the REPL's SIGINT handler or the serve
// reader thread, and polled at CANCELLATION POINTS:
//
//   * the evaluator: every statement and every verb / method / lazy-member call
//     (Evaluator.cpp), and the end of every cost-gated admission (Confirm.cpp);
//   * the long verbs' own loop boundaries -- per Hecke prime, per torsion prime,
//     per level, per stage -- in the bindings and, through the neutral hook
//     Util/CancelPoint.h, in the pipeline (the list: NBCONFIRM_S5_report.md).
//
// A point is a plain function call on the thread that runs the statement.  It
// is armed only inside a CancelScope (the REPL's statement, the served cell)
// on THAT thread: on any other thread (an engine worker, a std::thread of a
// parallel region) and outside a scope (scripts, -c, validations, campaigns)
// PollCancel is a no-op and can never throw.  Points sit only between whole
// units of work (never inside a kernel, a parallel region, a destructor, a
// noexcept function or a cache builder), so unwinding drops only stack-local
// partial results: no cache entry, no program cache and no handle state is
// left half-written, and a later run computes exactly what an uncancelled run
// computes (the cancellation never changes a value).
//
// A cancelled call raises `Interrupted`.  Like serve's ServeInterrupted it is
// neither a std::exception nor a ConsoleError, so no `catch` of a verb, of the
// evaluator or of `expect_error` swallows it, and a lazy handle whose compute()
// it leaves stays `declared` (Object.cpp catches only those two), i.e. the
// next read retries -- nothing partial is bound.
//
// STL-ONLY (plan Sec. 2 C8).
// ===========================================================================
#pragma once
#ifndef CONSOLE_CANCEL_H
#define CONSOLE_CANCEL_H

#include "ConsoleError.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace console {

struct Interrupted {
    SourcePos pos;      // the innermost statement / call the evaluator knows (filled on the way out)
    std::string site;   // the cancellation point that saw the request
};

// The request flag.  RequestCancel() is async-signal-safe (one lock-free atomic store).
void RequestCancel() noexcept;
void ClearCancel() noexcept;
bool CancelRequested() noexcept;

// Arms the points on the CURRENT thread for its lifetime (nests).
class CancelScope {
public:
    CancelScope();
    ~CancelScope();
    CancelScope(const CancelScope&) = delete;
    CancelScope& operator=(const CancelScope&) = delete;
};

// Disarms the points on the current thread for its lifetime (nests): rendering (PrintJson),
// run_validation, anything whose own catch (...) must not see an Interrupted.
class NoCancelScope {
public:
    NoCancelScope();
    ~NoCancelScope();
    NoCancelScope(const NoCancelScope&) = delete;
    NoCancelScope& operator=(const NoCancelScope&) = delete;
};

bool CancelArmedHere() noexcept;

// The cancellation point: a no-op unless armed on this thread; throws Interrupted{.., site} when a
// cancel is requested.  `site` is a string literal ("descent.hecke_prime", ...).
void PollCancel(const char* site);

// ---------------------------------------------------------------------------
// The test interlock (read-only: it can only stop a computation, never change one).
//   after:K[:PREFIX]  raise the request at the K-th armed visit (of a point whose site starts with PREFIX)
//   wait:K[:PREFIX]   at that visit call the wait notifier, then wait (<= 60 s) for an outside request
//   hold:K[:PREFIX]   at that visit call the notifier, then stay 3 s at the point whatever happens (a long
//                     kernel between two points: the case of a second Ctrl-C before the first took effect)
// One-shot: the hook disarms itself when it fires.  Programmatic (validations) or from the env
// variable HACS_CANCEL_TEST (the REPL and --serve read it once at start; --serve only as the child of
// this same binary, as HACS_HOST_PROBE_FIXTURE).
// ---------------------------------------------------------------------------
enum class CancelTestMode { Off, After, Wait, Hold };
struct CancelTest {
    CancelTestMode mode = CancelTestMode::Off;
    std::uint64_t k = 0;
    std::string prefix;
};
// Parses "after:K[:PREFIX]" / "wait:K[:PREFIX]" / "hold:K[:PREFIX]" (K >= 1); false (and Off) on anything else.
bool ParseCancelTest(const std::string& text, CancelTest& out);
void SetCancelTest(const CancelTest& t);
CancelTest GetCancelTest();
bool CancelTestFired();
// Called (on the armed thread) when a `wait` / `hold` hook fires, before it waits.
void SetCancelWaitNotifier(std::function<void(std::uint64_t k, const char* site)> fn);

// Visit accounting (armed visits only), for the validation: the count and, when tracing, the sites in order.
std::uint64_t CancelVisits();
void ResetCancelVisits();
void SetCancelTrace(bool on);
std::vector<std::string> CancelTraceSites();

} // namespace console

#endif // CONSOLE_CANCEL_H
