// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// ConsoleCli.cpp -- see ConsoleCli.h.
// ===========================================================================

#include "ConsoleCli.h"

#include "ConsoleError.h"
#include "Registry.h"
#include "ScriptRunner.h"
#include "Session.h"
#include "Settings.h"
#include "Value.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace console {

namespace {

bool IsInteger(const std::string& s) {
    if (s.empty()) return false;
    std::size_t i = (s[0] == '-' || s[0] == '+') ? 1 : 0;
    if (i >= s.size()) return false;
    for (; i < s.size(); ++i) {
        if (s[i] < '0' || s[i] > '9') return false;
    }
    return true;
}

// Splits "key=value" at the FIRST '=' ("--set device=cpu", "n=10").  Returns
// false when there is no '=' or the key is empty.
bool SplitAssignment(const std::string& text, std::string& key, std::string& value) {
    const std::size_t eq = text.find('=');
    if (eq == std::string::npos || eq == 0) return false;
    key = text.substr(0, eq);
    value = text.substr(eq + 1);
    return true;
}

// An identifier-shaped key, so that a stray path can never be mistaken for a
// parameter.  Dots are legal (the mathematical settings are dotted: sturm.cap).
bool IsSettingName(const std::string& s) {
    if (s.empty()) return false;
    for (std::size_t i = 0; i < s.size(); ++i) {
        const char c = s[i];
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' ||
                        (i > 0 && ((c >= '0' && c <= '9') || c == '.'));
        if (!ok) return false;
    }
    return true;
}

enum class Form { None, Repl, Script, Code, Stdin };

struct Invocation {
    Form form = Form::None;
    std::string path;                      // --script <file>
    std::string code;                      // -c <code>
    bool then_repl = false;                // --script f.oc --console
    bool help = false;
    std::string log_path;
    bool verbose_set = false;
    bool verbose = false;
    std::vector<std::pair<std::string, std::string>> settings;   // --set key=value
    std::vector<std::pair<std::string, std::string>> params;     // name=value
    std::string error;                     // non-empty => usage error (exit 1)
};

Invocation ParseArgs(const std::vector<std::string>& args) {
    Invocation inv;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        if (a == "--help" || a == "-h") {
            inv.help = true;
        } else if (a == "--console" || a == "-i") {
            if (inv.form == Form::None) inv.form = Form::Repl;
            else inv.then_repl = true;       // `--script f.oc --console`
        } else if (a == "--script") {
            if (i + 1 >= args.size()) { inv.error = "--script needs a file"; return inv; }
            if (inv.form != Form::None && inv.form != Form::Repl) {
                inv.error = "only one of --script / -c / - may be given";
                return inv;
            }
            inv.then_repl = (inv.form == Form::Repl);
            inv.form = Form::Script;
            inv.path = args[++i];
        } else if (a == "-c") {
            if (i + 1 >= args.size()) { inv.error = "-c needs a code string"; return inv; }
            if (inv.form != Form::None && inv.form != Form::Repl) {
                inv.error = "only one of --script / -c / - may be given";
                return inv;
            }
            inv.then_repl = (inv.form == Form::Repl);
            inv.form = Form::Code;
            inv.code = args[++i];
        } else if (a == "-") {
            if (inv.form != Form::None && inv.form != Form::Repl) {
                inv.error = "only one of --script / -c / - may be given";
                return inv;
            }
            inv.then_repl = false;   // stdin IS the script; a REPL after it would read nothing
            inv.form = Form::Stdin;
        } else if (a == "--set") {
            if (i + 1 >= args.size()) { inv.error = "--set needs key=value"; return inv; }
            std::string k, v;
            if (!SplitAssignment(args[++i], k, v) || !IsSettingName(k)) {
                inv.error = "--set expects key=value, got '" + args[i] + "'";
                return inv;
            }
            inv.settings.emplace_back(k, v);
        } else if (a == "--log") {
            if (i + 1 >= args.size()) { inv.error = "--log needs a file"; return inv; }
            inv.log_path = args[++i];
        } else if (a == "--quiet") {
            inv.verbose_set = true;
            inv.verbose = false;
        } else if (a == "--verbose") {
            inv.verbose_set = true;
            inv.verbose = true;
        } else {
            std::string k, v;
            if (SplitAssignment(a, k, v) && IsSettingName(k)) {
                inv.params.emplace_back(k, v);
            } else {
                inv.error = "unrecognised console argument '" + a + "'";
                return inv;
            }
        }
    }
    return inv;
}

} // namespace

// ---------------------------------------------------------------------------

Value ParseCliLiteral(const std::string& text) {
    if (text == "true") return Value::from_bool(true);
    if (text == "false") return Value::from_bool(false);
    if (IsInteger(text)) {
        // Out-of-range stays a String rather than silently wrapping.
        try {
            return Value::from_int(static_cast<std::int64_t>(std::stoll(text)));
        } catch (const std::exception&) {
            return Value::from_string(text);
        }
    }
    if (text.size() >= 2 && ((text.front() == '"' && text.back() == '"') ||
                             (text.front() == '\'' && text.back() == '\''))) {
        return Value::from_string(text.substr(1, text.size() - 2));
    }
    return Value::from_string(text);
}

std::string ConsoleUsage() {
    std::string u;
    u += "OC script console (console phase C1a).\n";
    u += "Usage:\n";
    u += "  hacs --console | -i          Interactive session (prompt 'oc> ').\n";
    u += "  hacs --script <file.oc> [name=value]...\n";
    u += "                                        Run a script; parameters are read as args.name.\n";
    u += "  hacs -c '<code>'             Run one line of OC script.\n";
    u += "  hacs -                       Run the script on standard input.\n";
    u += "Modifiers:\n";
    u += "  --set key=value                       Apply a session setting before the first statement.\n";
    u += "  --log <file>                          Append the transcript (inputs prefixed '> ').\n";
    u += "  --quiet | --verbose                   The pipeline's own chatter off / on.\n";
    u += "  --script <file> --console             Run the script, then drop into the REPL.\n";
    u += "  --help                                This text.\n";
    u += "Exit codes: 0 ok, 1 usage, 2 parse error, 3 runtime error, 4 failed assert,\n";
    u += "            5 a legal setting this build cannot execute.\n";
    u += "Language and verb reference: `help` inside the session, `?name` for one name.\n";
    return u;
}

int RunConsole(Registry& registry, const std::vector<std::string>& args, std::istream* in,
               std::ostream* out) {
    std::istream& input = (in != nullptr) ? *in : std::cin;
    std::ostream& output = (out != nullptr) ? *out : std::cout;

    const Invocation inv = ParseArgs(args);
    if (!inv.error.empty()) {
        output << "error: " << inv.error << "\n" << ConsoleUsage();
        return static_cast<int>(ExitCode::Usage);
    }
    if (inv.help) {
        output << ConsoleUsage();
        return static_cast<int>(ExitCode::Ok);
    }
    if (inv.form == Form::None) {
        output << "error: nothing to run\n" << ConsoleUsage();
        return static_cast<int>(ExitCode::Usage);
    }

    // `verbose` defaults ON in the REPL and OFF in script mode (plan Sec. 2 C4):
    // a transcript must be a pure function of the script.
    const bool interactive = (inv.form == Form::Repl);
    const bool verbose_default = inv.verbose_set ? inv.verbose : interactive;

    Session session(registry, verbose_default);
    session.set_output(output);

    std::ofstream log;
    if (!inv.log_path.empty()) {
        log.open(inv.log_path.c_str(), std::ios::app);
        if (!log) {
            output << "error: cannot open log file '" << inv.log_path << "'\n";
            return static_cast<int>(ExitCode::Usage);
        }
        session.set_log(&log);
    }

    // --set BEFORE the first statement; an undeclared key or an invalid value is
    // exit 1 (plan Sec. 6).  (C5a, slot-9 WP-12; decision 2 (ii) (A), a named
    // exception to [P] C5 (b)) A LEGAL value this build cannot execute -- the
    // installer's NotImplementedError, e.g. `--set coefficient_ring=Z` -- is
    // exit 5, as on the `set` statement.
    for (const auto& kv : inv.settings) {
        try {
            session.settings().set(kv.first, ParseCliLiteral(kv.second), SourcePos("<command line>", 0, 0));
        } catch (const ConsoleError& e) {
            output << FormatError(e) << "\n";
            if (e.kind() == kNotImplementedKind) return static_cast<int>(ExitCode::NotImplemented);
            return static_cast<int>(ExitCode::Usage);
        }
    }
    for (const auto& kv : inv.params) {
        session.set_arg(kv.first, ParseCliLiteral(kv.second));
    }

    int code = static_cast<int>(ExitCode::Ok);
    switch (inv.form) {
        case Form::Repl:
            code = RunRepl(session, input, output, /*show_prompts=*/true);
            break;
        case Form::Script:
            code = RunFile(session, inv.path);
            break;
        case Form::Code:
            code = RunSource(session, inv.code, "<string>");
            break;
        case Form::Stdin: {
            std::ostringstream buf;
            buf << input.rdbuf();
            code = RunSource(session, buf.str(), "<stdin>");
            break;
        }
        case Form::None:
            break;
    }

    // `--script f.oc --console`: explore the prepared state (plan Sec. 7).  A
    // failed script does not open a REPL over a half-built session.
    if (inv.then_repl && code == static_cast<int>(ExitCode::Ok)) {
        code = RunRepl(session, input, output, /*show_prompts=*/true);
    }
    return code;
}

int RunConsole(int argc, char** argv) {
    std::vector<std::string> args;
    args.reserve(static_cast<std::size_t>(argc > 0 ? argc : 0));
    for (int i = 0; i < argc; ++i) args.emplace_back(argv[i] == nullptr ? "" : argv[i]);
    return RunConsole(Registry::global(), args, nullptr, nullptr);
}

} // namespace console
