// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// ConsoleCli.h -- console C1a: the command-line surface.
//
// Plan (docs/console/00-console-interface-plan.md v3) Sec. 9: "ConsoleCli.h/.cpp
// RunConsole(argc, argv) -- the command-line surface  [C1a]".  It is the ONLY
// Core/ file that touches argv; the in-process drivers (RunSource / RunFile /
// RunRepl) are C0's ScriptRunner and are unchanged.
//
// main.cpp gains exactly ONE branch (plan Sec. 9), inserted after the --help
// test and before the validation-name loop:
//
//     if (arg1 == "--console" || arg1 == "-i" || arg1 == "--script" ||
//         arg1 == "-c" || arg1 == "-") {
//         console::RegisterPipelineVerbs();
//         return console::RunConsole(argc - 1, argv + 1);
//     }
//
// so argv[0] here is the FIRST console argument.  `print_usage` in main.cpp
// stays byte-identical (plan Sec. 5, Q8: console usage is printed by
// `--console --help`), which ConsoleCliPreservationValidation gates.
//
// Forms (plan Sec. 6, Sec. 7):
//
//     --console | -i                 REPL over std::cin (verbose ON)
//     --script <file> [args...]      run a file, then exit
//     -c <code>                      run one string
//     -                              run the script on stdin
//     --script <file> --console      run the file, then drop into the REPL
//                                    with its bindings
//
// Modifiers, in any position after the form:
//
//     --set key=value                a session setting applied BEFORE the first
//                                    statement; an undeclared key is exit 1
//     name=value                     a script parameter, read as `args.name`
//     --log <file>                   append the transcript (plan Sec. 2 C5)
//     --quiet / --verbose            the `verbose` presentation key
//     --help                         console usage on stdout, exit 0
//
// Exit codes are ScriptRunner's (plan Sec. 6): 0 ok, 1 usage / file not found /
// undeclared --set key, 2 parse, 3 runtime, 4 assertion, 5 a legal setting this
// build cannot execute.
//
// STL-ONLY (plan Sec. 2 C8): this file may include C++ standard headers and
// Console/Core/ headers and nothing else -- check_console_includes.sh rule 1.
// ===========================================================================
#pragma once
#ifndef CONSOLE_CONSOLE_CLI_H
#define CONSOLE_CONSOLE_CLI_H

#include <iosfwd>
#include <string>
#include <vector>

#include "Value.h"

namespace console {

class Registry;

// Runs the console over `argv[0..argc)`, on the process-wide registry
// (Registry::global()), which the caller has already populated.  Returns the
// exit code.
int RunConsole(int argc, char** argv);

// The same, on an explicit registry -- what the validations drive so that a
// test cannot depend on (or disturb) the global one.  `in` / `out` default to
// std::cin / std::cout when null.
int RunConsole(Registry& registry, const std::vector<std::string>& args,
               std::istream* in, std::ostream* out);

// The text `--console --help` prints.  Exposed so the validation can assert it
// rather than re-spell it.
std::string ConsoleUsage();

// Parses a command-line literal ("10", "-3", "true", "false", "cpu", "\"x\"")
// into a Value: Int when the whole token is an integer, Bool for true/false,
// String otherwise (with surrounding quotes stripped).  Plan Sec. 6,
// "`name=value` pairs are parsed as literals".
Value ParseCliLiteral(const std::string& text);

} // namespace console

#endif // CONSOLE_CONSOLE_CLI_H
