// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Session.h -- console C0: variables, settings, output and chatter capture.
//
// Plan Sec. 2 C4 (the session owns the AUTHORITATIVE settings; no RAII stack),
// Sec. 2 C5 (canonical output, `#~` annotations, the pipeline's own std::cout
// AND std::cerr redirected around a verb call, `--log`) and Sec. 9 (the TU
// layout: Session.h/.cpp = "variables, authoritative settings + snapshot,
// cout/cerr redirection, log").
//
// Scoping (plan Sec. 9): ONE global frame; a `for` loop shadows its variable
// for the duration of the loop and restores the previous binding afterwards.
//
// STL-ONLY (plan Sec. 2 C8).
// ===========================================================================
#pragma once
#ifndef CONSOLE_SESSION_H
#define CONSOLE_SESSION_H

#include <cstdint>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "Registry.h"
#include "Settings.h"
#include "Value.h"

namespace console {

// The two console-owned settings C0 registers.  Every ENGINE knob (threads,
// backend, device, lean, ...) is registered by C2 through the same mechanism,
// and every MATHEMATICAL one (final_mode, sturm.cap, locality, ... -- owner
// decision R1) by the extension engine; C0 registers neither, because the
// engine owns their defaults and their validation.
constexpr const char* kSettingPrintMaxEntries = "print_max_entries";
constexpr const char* kSettingVerbose = "verbose";

void RegisterCoreSettings(Settings& settings, bool verbose_default);

class Session {
public:
    explicit Session(Registry& registry, bool verbose_default = false);
    ~Session();

    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    Registry& registry() { return registry_; }
    const Registry& registry() const { return registry_; }
    Settings& settings() { return settings_; }
    const Settings& settings() const { return settings_; }

    // --- variables (one global frame) --------------------------------------
    void set_var(const std::string& name, Value v);
    const Value* find_var(const std::string& name) const;
    bool erase_var(const std::string& name);
    const std::vector<std::pair<std::string, Value>>& vars() const { return vars_; }

    // --- script parameters (`args.name`, plan Sec. 6) ----------------------
    void set_arg(const std::string& name, Value v) { args_.set(name, std::move(v)); }
    const Record& args() const { return args_; }

    // --- output ------------------------------------------------------------
    std::ostream& out();
    void set_output(std::ostream& os) { out_ = &os; }
    void print_line(const std::string& text);
    // A `#~`-prefixed line: timings and benchmark summaries, excluded from every
    // .expected comparison (plan Sec. 2 C5).
    void print_annotation(const std::string& text);

    // --- transcript log (`--log <file>`, plan Sec. 2 C5) -------------------
    void set_log(std::ostream* log) { log_ = log; }
    void log_input(const std::string& line);
    // (C2b, slot-7 WP-8) Whether a `--log` transcript is being written, so the
    // script drivers log their INPUT lines too (the REPL always did).
    bool logging() const { return log_ != nullptr; }

    // --- convenience for verbs / the evaluator -----------------------------
    std::int64_t print_max_entries() const;
    bool verbose() const;

    // The directory of the script currently running, for `include` resolution
    // (plan Sec. 3.2: "include is resolved relative to the including file").
    const std::string& current_dir() const { return current_dir_; }
    void set_current_dir(std::string dir) { current_dir_ = std::move(dir); }

    // The include stack, for cycle detection.
    std::vector<std::string>& include_stack() { return include_stack_; }

    // (NBCONFIRM S3) The stream a confirmation prompt reads its answer from, or null when this session has no prompt.  Only
    // RunRepl sets it, and only when the host probe reports that stdin and stderr are terminals and the REPL reads std::cin
    // (Confirm.h); every other mode -- --script, -c, -, a piped REPL, the script half of `--script f --console`, --serve --
    // leaves it null, so a call outside its silent envelope refuses exactly as before (design Sec. 2.2 correction, owner Q2).
    std::istream* confirm_input() const { return confirm_in_; }
    void set_confirm_input(std::istream* in) { confirm_in_ = in; }

    // (NBCONFIRM S4) The nesting depth of `expect_error` blocks being executed.  A block that EXPECTS an error never asks:
    // inside one, a call outside its silent envelope refuses exactly as in a prompt-less mode (today's text, exit 3) on a
    // terminal REPL and in a served session alike, so a script's expected refusal is machine-independent (Confirm.cpp).
    int expect_error_depth() const { return expect_error_depth_; }
    void enter_expect_error() { ++expect_error_depth_; }
    void leave_expect_error() { --expect_error_depth_; }

private:
    friend class ChatterCapture;

    Registry& registry_;
    Settings settings_;
    std::vector<std::pair<std::string, Value>> vars_;   // insertion order; no hash container
    Record args_;
    std::ostream* out_;
    std::ostream* log_ = nullptr;
    std::string current_dir_;
    std::vector<std::string> include_stack_;
    std::istream* confirm_in_ = nullptr;   // (NBCONFIRM S3) see confirm_input()
    int expect_error_depth_ = 0;           // (NBCONFIRM S4) see expect_error_depth()

    // Set while a ChatterCapture is active: console output goes here (an
    // ostream over the ORIGINAL std::cout buffer) instead of through the
    // swapped std::cout.
    std::unique_ptr<std::ostream> direct_out_;
};

// Swaps std::cout's AND std::cerr's stream buffers to an in-memory sink for the
// duration of the scope, so the pipeline's own chatter cannot reach a
// transcript (plan Sec. 2 C5).  While it is active the session keeps writing to
// the real terminal through the buffer captured at construction.
//
// The rdbuf swap is process-global (risk R17): a future concurrent verb must
// give each task its own sink rather than nesting this.
class ChatterCapture {
public:
    explicit ChatterCapture(Session& session);
    ~ChatterCapture();

    ChatterCapture(const ChatterCapture&) = delete;
    ChatterCapture& operator=(const ChatterCapture&) = delete;

    std::string text() const;

private:
    Session& session_;
    std::unique_ptr<std::ostringstream> sink_;
    std::streambuf* old_cout_ = nullptr;
    std::streambuf* old_cerr_ = nullptr;
    bool rebound_output_ = false;
};

} // namespace console

#endif // CONSOLE_SESSION_H
