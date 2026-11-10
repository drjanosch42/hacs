// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Serve.h -- `cohomology_app --serve`: the served console, protocol
// `oc-serve/1` ([P] v3.10 Sec. 18; hecke-extensions slot 19, NB-2).
//
// One process = one session = one console Session, driven by newline-delimited
// JSON on stdio: requests `eval` / `complete` / `help` / `describe` /
// `interrupt` / `close` in, frames `hello` / `out` / `progress` / `value` /
// `done` / `completions` / `help` / `described` / `bye` / `fatal` out.  Errors of
// a cell (exit codes 2-5 of Sec. 6) are `done` replies and the session
// survives them; 1 is a request-level refusal; the PROCESS exits 0 (EOF,
// `close`, SIGTERM), 1 (a launch refusal) or 70 (a `fatal` frame).
//
// READ-ONLY.  The session dispatches only the overloads whose `read_only` tag
// is set (ReadOnlyTags.h): the process registry is restricted IN PLACE
// (Registry::transform_overloads, so every TypeInfo keeps its address) to the
// tagged overloads (each wrapped for the cooperative interrupt), a refuser in
// place of an untagged overload whose verb has a tagged sibling
// (`save_state(x, path)`), and no entry at all for a verb with no tagged
// overload (refused statically, before anything of the cell runs).
// Defence in depth: the environment gate, RLIMIT_FSIZE = 0, no core dumps,
// PR_SET_NO_NEW_PRIVS, the RSS watchdog ([P] Sec. 18.9 / 18.10).
//
// Compiled only with OC_BUILD_NOTEBOOK=ON.  The driver uses the Core's public
// surface (Parse, Evaluator::exec / eval, Session, Registry) as ScriptRunner
// does; the one Core addition is the additive Registry::transform_overloads,
// which no other front end calls.
// ===========================================================================
#pragma once
#ifndef CONSOLE_BINDINGS_SERVE_H
#define CONSOLE_BINDINGS_SERVE_H

#include <functional>
#include <string>
#include <vector>

namespace console {
namespace serve {

constexpr const char* kProto = "oc-serve/1";
constexpr const char* kConsoleVersion = "3.10";
constexpr std::size_t kDefaultMaxReplyBytes = 1048576;
constexpr std::size_t kMaxRequestBytes = 8388608;
constexpr std::size_t kQueueLimit = 64;
constexpr int kWatchdogPollMs = 250;
constexpr int kFatalExit = 70;

// The process entry point.  `argv[0..argc)` are the tokens AFTER `--serve`.
// `register_verbs` populates console::Registry::global() (main.cpp passes the
// pipeline + extension registration); it runs AFTER the environment gate and
// the process hardening, so a refused launch registers nothing.  Never returns
// normally on a served session (the process `_exit`s); returns the exit code
// of a launch refusal.
int RunServe(int argc, char** argv, const std::function<void()>& register_verbs);

// The environment gate of Sec. 18.9, as data (the validation tests it too).
const std::vector<std::string>& EnvWhitelist();
// True when `name` must refuse a launch: it matches GOLDEN_*, CONSOLE_*,
// CLI_MANUAL_*, HECKE_*, ENGINE_*, OC_* or *SAN_OPTIONS and is not on the
// whitelist.
bool EnvNameRefused(const std::string& name);

} // namespace serve
} // namespace console

#endif // CONSOLE_BINDINGS_SERVE_H
