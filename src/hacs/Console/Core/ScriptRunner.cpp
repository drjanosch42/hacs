// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// ScriptRunner.cpp -- see ScriptRunner.h.
// ===========================================================================

#include "ScriptRunner.h"

#include "Cancel.h"
#include "Confirm.h"
#include "LineInput.h"
#include "Lexer.h"
#include "Parser.h"

#include <atomic>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <istream>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

namespace console {

std::string FormatError(const ConsoleError& e) {
    return e.formatted();
}

namespace {

int CodeFor(const ConsoleError& e) {
    if (e.kind() == "parse error") return static_cast<int>(ExitCode::Parse);
    if (e.kind() == "assertion failed") return static_cast<int>(ExitCode::Assertion);
    // (C1a) Plan Sec. 6: "5 a legal mathematical setting this build cannot
    // execute".  Its own code, never 1 and never 3.
    if (e.kind() == kNotImplementedKind) return static_cast<int>(ExitCode::NotImplemented);
    return static_cast<int>(ExitCode::Runtime);
}

// A ConsoleError raised at run time gets its source line attached here, where
// the whole script text is still in hand.
void ReportError(Session& session, const ConsoleError& e, const std::string& source) {
    session.print_line(FormatError(e));
    if (e.kind() != "parse error" && e.pos().line > 0) {
        const std::string line = SourceLine(source, e.pos().line);
        if (!line.empty()) session.print_line("  " + line);
    }
}

// (C2b, slot-7 WP-8) `--log` in script mode ([P] Sec. 2 C5, "Transcript":
// "`--log <file>` appends every input line and every output line, prefixed
// `> ` / nothing, so a REPL session is replayable as a script by stripping the
// prefix").  Through slot 6 only the REPL logged its input; a script, `-c` and
// stdin logged their OUTPUT alone.  Now each top-level statement's source lines
// are logged (prefixed) just before it runs, so the log interleaves inputs and
// the outputs they produced.  A statement's chunk runs from its first line to
// the line before the next statement, minus trailing blank and comment lines,
// which are logged in front of the NEXT statement instead -- so a comment
// stays next to the code it describes.  Stripping every `> ` line's prefix and
// dropping the unprefixed lines gives back the source, line for line.
bool IsBlankOrComment(const std::string& line) {
    for (char c : line) {
        if (c == ' ' || c == '\t' || c == '\r') continue;
        return c == '#';
    }
    return true;
}

void RunLogged(Session& session, Evaluator& ev, const Program& program,
               const std::string& source) {
    std::vector<std::string> lines;
    {
        std::istringstream in(source);
        std::string l;
        while (std::getline(in, l)) lines.push_back(l);
    }
    const int total = static_cast<int>(lines.size());
    int logged = 0;   // lines 1..logged are in the log
    const std::vector<StmtPtr>& st = program.statements;
    for (std::size_t i = 0; i < st.size(); ++i) {
        const int start = st[i]->pos.line;
        int end = (i + 1 < st.size()) ? st[i + 1]->pos.line - 1 : total;
        if (end < start) end = start;   // two statements on one line
        if (end > total) end = total;
        while (end > start && IsBlankOrComment(lines[static_cast<std::size_t>(end - 1)])) --end;
        for (int k = logged + 1; k <= end; ++k) {
            session.log_input(lines[static_cast<std::size_t>(k - 1)]);
        }
        if (end > logged) logged = end;
        ev.exec(st[i]);
    }
    for (int k = logged + 1; k <= total; ++k) {
        session.log_input(lines[static_cast<std::size_t>(k - 1)]);
    }
}

// (NBCONFIRM S5, [P] Sec. 7) The REPL's SIGINT handler.  While a statement runs, the FIRST Ctrl-C only raises the cancel
// request (one lock-free atomic store: async-signal-safe); the statement stops at its next cancellation point and the REPL
// prints `interrupted` and the next prompt.  At the idle prompt, and for a SECOND Ctrl-C that arrives before the first took
// effect (a computation between two points that the user wants to abandon), the handler restores the default action and
// re-raises: the process ends by SIGINT exactly as it did before S5 (shell status 130).  Installed only for the process's
// own interactive REPL (RunRepl on std::cin with prompts), never for a script, -c, - or a validation's string streams.
std::atomic<bool> g_repl_running{false};
static_assert(std::atomic<bool>::is_always_lock_free, "the SIGINT handler reads g_repl_running");

extern "C" void OnReplSigint(int) {
    if (g_repl_running.load(std::memory_order_relaxed) && !CancelRequested()) {
        RequestCancel();
        return;
    }
    std::signal(SIGINT, SIG_DFL);
    std::raise(SIGINT);
}

struct ReplSigint {
    bool installed = false;
    void (*old)(int) = SIG_DFL;
    explicit ReplSigint(bool on) {
        if (!on) return;
        old = std::signal(SIGINT, &OnReplSigint);
        installed = old != SIG_ERR;
    }
    ~ReplSigint() {
        if (installed) std::signal(SIGINT, old == SIG_ERR ? SIG_DFL : old);
    }
};

// `<stdin>:1:5: interrupted (at descent.hecke_prime)`
std::string InterruptedText(const Interrupted& x) {
    std::string t = x.pos.known() ? x.pos.to_string() + ": " : std::string();
    t += "interrupted";
    if (!x.site.empty()) t += " (at " + x.site + ")";
    return t;
}

} // namespace

int RunSource(Session& session, const std::string& source, const std::string& file,
              Evaluator::FileLoader loader) {
    // The WHOLE input is parsed before anything runs (plan Sec. 6), so a parse
    // error cannot leave half a script's side effects behind.
    Program program;
    try {
        program = Parse(source, file);
    } catch (const ConsoleError& e) {
        session.print_line(FormatError(e));
        return static_cast<int>(ExitCode::Parse);
    }

    Evaluator ev(session);
    if (loader) ev.set_file_loader(std::move(loader));
    try {
        if (session.logging()) RunLogged(session, ev, program, source);
        else ev.run(program);
    } catch (const ExitSignal& sig) {
        return sig.code;
    } catch (const QuitSignal& sig) {
        return sig.code;
    } catch (const ConsoleError& e) {
        ReportError(session, e, source);
        return CodeFor(e);
    } catch (const std::exception& e) {
        // Nothing below the evaluator should reach here; if it does, it is
        // still a runtime failure and must not abort the process.
        session.print_line(std::string(file) + ": error: " + e.what());
        return static_cast<int>(ExitCode::Runtime);
    }
    return static_cast<int>(ExitCode::Ok);
}

int RunFile(Session& session, const std::string& path, Evaluator::FileLoader loader) {
    std::string source;
    if (loader) {
        if (!loader(path, source)) {
            session.print_line("error: cannot read script '" + path + "'");
            return static_cast<int>(ExitCode::Usage);
        }
    } else {
        std::ifstream in(path.c_str());
        if (!in) {
            session.print_line("error: cannot read script '" + path + "'");
            return static_cast<int>(ExitCode::Usage);
        }
        std::ostringstream buf;
        buf << in.rdbuf();
        source = buf.str();
    }
    const std::string saved_dir = session.current_dir();
    session.set_current_dir(DirectoryOf(path));
    const int code = RunSource(session, source, path, std::move(loader));
    session.set_current_dir(saved_dir);
    return code;
}

int RunRepl(Session& session, std::istream& in, std::ostream& prompt_out, bool show_prompts) {
    LineInput input(in, prompt_out, show_prompts);
    Evaluator ev(session);
    // (NBCONFIRM S3, [P] Sec. 7.4) A cost confirmation may ask only here, only on the process's own std::cin, and only when
    // the host probe reports that stdin and stderr are terminals.  A REPL over a pipe or a validation's string stream keeps
    // the prompt-less policy (refuse outside the envelope, exactly as before).
    struct ConfirmInput {
        Session& s;
        std::istream* saved;
        ~ConfirmInput() { s.set_confirm_input(saved); }
    } confirm_input{session, session.confirm_input()};
    {
        const HostProbe& h = GetHostProbe();
        if (show_prompts && &in == &std::cin && h.prompt_tty && h.prompt_tty()) session.set_confirm_input(&in);
    }
    // (NBCONFIRM S5) Ctrl-C cancels the running statement (see OnReplSigint); the test interlock HACS_CANCEL_TEST (Cancel.h)
    // is read here, for this REPL only.
    const bool own_repl = show_prompts && &in == &std::cin;
    const ReplSigint sigint(own_repl);
    if (own_repl) {
        CancelTest t;
        const char* env = std::getenv("HACS_CANCEL_TEST");
        if (env != nullptr && ParseCancelTest(env, t)) {
            SetCancelTest(t);
            SetCancelWaitNotifier([&session](std::uint64_t k, const char* site) {
                session.print_annotation("cancel test: waiting at " + std::string(site ? site : "") + " (visit " + std::to_string(k) + ")");
                session.out().flush();
            });
        }
    }
    int last_code = static_cast<int>(ExitCode::Ok);
    std::string text;
    while (input.read_statement(text)) {
        session.log_input(text);
        Program program;
        try {
            program = Parse(text, "<stdin>");
        } catch (const ConsoleError& e) {
            session.print_line(FormatError(e));
            continue;   // the REPL survives every error (plan Sec. 2 C7)
        }
        struct Running {   // (NBCONFIRM S5) the statement's cancel window
            explicit Running(bool on) : on_(on) {
                if (!on_) return;
                ClearCancel();
                g_repl_running.store(true);
            }
            ~Running() {
                if (!on_) return;
                g_repl_running.store(false);
                ClearCancel();
            }
            bool on_;
        };
        try {
            const Running running(own_repl);
            const CancelScope armed;
            ev.run(program);
        } catch (const Interrupted& x) {
            ClearCancel();   // honoured (a validation's in-process REPL has no Running window to clear it)
            session.print_line(InterruptedText(x));   // the session goes on; the statement's result is not bound
        } catch (const ExitSignal& sig) {
            return sig.code;
        } catch (const QuitSignal& sig) {
            return sig.code;
        } catch (const ConfirmDeclined&) {
            session.print_line("not run");   // (NBCONFIRM S3) a declined confirmation: nothing is bound, the session goes on
        } catch (const ConsoleError& e) {
            if (e.kind() == kDeclinedKind) {   // the same, re-thrown as a plain ConsoleError by a wrapper
                session.print_line("not run");
                continue;
            }
            ReportError(session, e, text);
            last_code = static_cast<int>(ExitCode::Ok);   // the REPL does not fail on an error
        } catch (const std::exception& e) {
            session.print_line(std::string("error: ") + e.what());
        }
    }
    return last_code;
}

} // namespace console
