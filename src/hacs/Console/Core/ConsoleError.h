// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// ConsoleError.h -- console C0 (language core): source positions, the error
// hierarchy and the two invariant-check macros.
//
// Binding design: docs/console/00-console-interface-plan.md v2, Sec. 2 C7
// ("Error model") and Sec. 8 (the origin -> behaviour table); work package
// WP-5 of docs/implementation/hecke_extensions_slot1_plan.md Sec. 3.
//
// STL-ONLY.  Console/Core/ is the analogue of HeckeCore: a static library that
// depends on NOTHING in the tree (plan Sec. 2 C8, "one-way dependency"), gated
// by docs/matrix-engine/tools/check_console_includes.sh.  That is why this
// header does NOT include src/ModArithmetic/EngineCheck.h and defines
// CONSOLE_CHECK / CONSOLE_DEBUG_CHECK instead, with exactly the semantics
// the assertion policy (EngineCheck.h) demands of ENGINE_CHECK /
// ENGINE_DEBUG_CHECK:
//
//   CONSOLE_CHECK(cond)        always compiled in, O(1) invariants; throws
//                              std::logic_error carrying file:line and the
//                              stringified condition.
//   CONSOLE_DEBUG_CHECK(cond)  compiled out under NDEBUG; for checks whose cost
//                              scales with the input.
//
// No bare `assert` appears anywhere in Console/ outside the validation TU.
// ===========================================================================
#pragma once
#ifndef CONSOLE_CONSOLE_ERROR_H
#define CONSOLE_CONSOLE_ERROR_H

#include <exception>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace console {

// --- Source positions ------------------------------------------------------
// Every token, AST node, error and object handle carries one.  `file` is the
// script path as the user spelled it ("<string>" for -c / in-process input,
// "<stdin>" for the REPL), so error text is reproducible.
struct SourcePos {
    std::string file;
    int line = 0;   // 1-based; 0 = unknown
    int col = 0;    // 1-based; 0 = unknown

    SourcePos() = default;
    SourcePos(std::string f, int l, int c) : file(std::move(f)), line(l), col(c) {}

    bool known() const { return line > 0; }
    // "file:line:col" (or "file:line", or "" when nothing is known).
    std::string to_string() const;
};

// --- The error hierarchy ---------------------------------------------------
// Plan Sec. 8.  Every console-visible failure is one of these; the console
// never lets a raw std::exception escape a verb call (Evaluator wraps it,
// naming the C++ class through the registry's exception translator so that
// Core needs no pipeline header -- the C7 catch order is installed by the C1
// bindings, not hard-coded here).
class ConsoleError : public std::exception {
public:
    ConsoleError(std::string kind, std::string message, SourcePos pos)
        : kind_(std::move(kind)), message_(std::move(message)), pos_(std::move(pos)) {
        rebuild();
    }
    ~ConsoleError() override = default;

    const char* what() const noexcept override { return formatted_.c_str(); }

    const std::string& kind() const { return kind_; }          // "parse error", "type error", ...
    const std::string& message() const { return message_; }
    const SourcePos& pos() const { return pos_; }
    const std::vector<std::string>& notes() const { return notes_; }

    // Secondary lines printed under the message, in insertion order:
    //   "  object T2 created at script.oc:9:7 by hecke(H, p = 2)"
    //   "  caused by: .../HeckeOperatorOnK.cpp:NNN: ..."
    void add_note(std::string note) { notes_.push_back(std::move(note)); rebuild(); }
    void set_pos_if_unknown(const SourcePos& p) {
        if (!pos_.known() && p.known()) { pos_ = p; rebuild(); }
    }

    // The full multi-line rendering used by what(), the REPL and scripts.
    const std::string& formatted() const { return formatted_; }

private:
    void rebuild();

    std::string kind_;
    std::string message_;
    SourcePos pos_;
    std::vector<std::string> notes_;
    std::string formatted_;
};

// Raised by the lexer and the parser.  A script is parsed in full before
// anything runs, so a ParseError is reported before any side effect (exit 2).
class ParseError : public ConsoleError {
public:
    ParseError(std::string message, SourcePos pos, std::string source_line = std::string());
    // The offending line and a caret, when the driver could supply it.
    const std::string& source_line() const { return source_line_; }
private:
    std::string source_line_;
};

// Arity / kind / unknown-name failures raised before any work happens: nothing
// is bound (plan Sec. 2 C2, "a verb-time error creates nothing").
class TypeError : public ConsoleError {
public:
    TypeError(std::string message, SourcePos pos)
        : ConsoleError("type error", std::move(message), std::move(pos)) {}
};

// Everything that fails while running: integer overflow, division by zero, a
// failed handle, a wrapped pipeline exception, a missing `args.name`.
class RuntimeError : public ConsoleError {
public:
    RuntimeError(std::string message, SourcePos pos)
        : ConsoleError("error", std::move(message), std::move(pos)) {}
};

// (console C1a) A legal request that THIS BUILD cannot execute -- plan Sec. 2
// C7's `hecke::NotImplementedInThisBuild` row, which the plan gives its OWN
// exit code, 5, so that a script or a CI matrix can tell "this build cannot do
// it" from "you asked for something wrong" (exit 1) and from "the computation
// failed" (exit 3).  It is NEVER reported as a bad value and the setting is
// never degraded to its default.
//
// The core raises it without naming a pipeline type: the C1 bindings install a
// Registry::set_exception_kind() classifier that returns kNotImplementedKind
// for the exceptions it recognises, and the evaluator / the handle wrapper
// preserve that kind through their own wrapping.
extern const char* const kNotImplementedKind;   // "not implemented in this build"

class NotImplementedError : public ConsoleError {
public:
    NotImplementedError(std::string message, SourcePos pos)
        : ConsoleError(kNotImplementedKind, std::move(message), std::move(pos)) {}
};

// `assert(cond, msg)` and `expect_error "..." { ... }`.  Exit code 4.
class AssertionFailure : public ConsoleError {
public:
    AssertionFailure(std::string message, SourcePos pos)
        : ConsoleError("assertion failed", std::move(message), std::move(pos)) {}
};

// --- Control-flow signals (NOT errors) -------------------------------------
// Deliberately outside ConsoleError so that `expect_error` cannot swallow them
// and a catch(const ConsoleError&) cannot mistake them for a failure.
struct ExitSignal { int code = 0; };
struct QuitSignal { int code = 0; };

// --- Invariant checks ------------------------------------------------------
namespace detail {
[[noreturn]] void ConsoleCheckFailed(const char* file, int line, const char* cond,
                                     const char* msg);
} // namespace detail

#define CONSOLE_CHECK(cond)                                                          \
    do {                                                                             \
        if (!(cond)) ::console::detail::ConsoleCheckFailed(__FILE__, __LINE__, #cond, nullptr); \
    } while (false)

#define CONSOLE_CHECK_MSG(cond, msg)                                                 \
    do {                                                                             \
        if (!(cond)) ::console::detail::ConsoleCheckFailed(__FILE__, __LINE__, #cond, (msg)); \
    } while (false)

#ifdef NDEBUG
#define CONSOLE_DEBUG_CHECK(cond)          do { } while (false)
#define CONSOLE_DEBUG_CHECK_MSG(cond, msg) do { } while (false)
#else
#define CONSOLE_DEBUG_CHECK(cond)          CONSOLE_CHECK(cond)
#define CONSOLE_DEBUG_CHECK_MSG(cond, msg) CONSOLE_CHECK_MSG(cond, msg)
#endif

} // namespace console

#endif // CONSOLE_CONSOLE_ERROR_H
