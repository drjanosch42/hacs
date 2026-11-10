// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// ScriptRunner.h -- console C0: the in-process drivers and the exit codes.
//
// Plan Sec. 6 ("Scripting and reproducibility"):
//
//   0  success
//   1  usage / file not found / undeclared --set key
//   2  parse error (reported before anything runs -- the whole file is parsed first)
//   3  runtime error (including every ENGINE_CHECK and a missing args.name)
//   4  a failed assert / expect_error
//   5  a legal mathematical setting this build cannot execute (C1a; plan
//      Sec. 2 C7 / Sec. 6 -- hecke::NotImplementedInThisBuild, kind
//      kNotImplementedKind).  It is never reported as a bad value (1) and
//      never as a failed computation (3).
//      exit(n) sets it explicitly; the REPL exits 0 on quit/EOF.
//
// SCOPE NOTE (a deliberate C0/C1 split, recorded in the WP-5 report): this
// header is the in-process driver only -- source string, file, and a REPL loop
// over an istream.  The COMMAND-LINE surface (`--script f.oc`, `-c "..."`,
// stdin, `--console`, `--log`, `name=value` parameters) is `ConsoleCli` in
// phase C1, together with its one `main.cpp` branch; slot 1's WP-5 changes no
// entry point, because the CLI is frozen by ConsoleCliPreservationValidation.
//
// STL-ONLY (plan Sec. 2 C8).
// ===========================================================================
#pragma once
#ifndef CONSOLE_SCRIPT_RUNNER_H
#define CONSOLE_SCRIPT_RUNNER_H

#include <cstdint>
#include <iosfwd>
#include <string>

#include "Evaluator.h"
#include "Session.h"

namespace console {

enum class ExitCode : int {
    Ok = 0,
    Usage = 1,
    Parse = 2,
    Runtime = 3,
    Assertion = 4,
    NotImplemented = 5
};

// Parses `source` in full, then runs it.  Errors are reported on the session's
// output stream, in the plan's format, and mapped to the exit codes above.
// `file` is the name that appears in every message.
int RunSource(Session& session, const std::string& source, const std::string& file,
              Evaluator::FileLoader loader = nullptr);

// Reads `path`, sets the session's current directory to its parent (so
// `include` resolves relative to it) and runs it.  Missing file => exit 1.
int RunFile(Session& session, const std::string& path,
            Evaluator::FileLoader loader = nullptr);

// A REPL over `in`: every complete statement runs, an error is printed and the
// session continues (plan Sec. 2 C7).  Returns 0 on quit/EOF, or the code of
// the last exit(n).
int RunRepl(Session& session, std::istream& in, std::ostream& prompt_out,
            bool show_prompts);

// The plan's rendering of an error, including the offending source line when
// one is available (plan Sec. 6).
std::string FormatError(const ConsoleError& e);

} // namespace console

#endif // CONSOLE_SCRIPT_RUNNER_H
