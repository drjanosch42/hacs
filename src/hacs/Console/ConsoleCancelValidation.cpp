// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ConsoleCancelValidation -- see the header.  (NBCONFIRM S5)

#include "ConsoleCancelValidation.h"

#include "../BuildSwitches.h"
#include "Bindings/HostProbe.h"
#include "Bindings/RegisterPipelineVerbs.h"
#include "Core/Cancel.h"
#include "Core/Registry.h"
#include "Core/ScriptRunner.h"
#include "Core/Session.h"
#include "../HeckeExt/RegisterHeckeExtensionVerbs.h"
#include "../Util/ValidationCheck.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

using console::Registry;
using console::Session;

oc_util::Checker Check("  [FAIL] ConsoleCancelValidation: ", true, 40);

Registry& Reg() {
    static Registry* reg = [] {
        Registry* r = new Registry();
        console::RegisterPipelineVerbs(*r);
        hecke_ext::RegisterHeckeExtensionVerbs(*r);
        return r;
    }();
    return *reg;
}

bool Contains(const std::string& h, const std::string& n) { return h.find(n) != std::string::npos; }

std::vector<std::string> Lines(const std::string& text) {
    std::vector<std::string> out;
    std::istringstream is(text);
    std::string l;
    while (std::getline(is, l)) {
        if (!l.empty() && l.back() == '\r') l.pop_back();
        if (l.rfind("#~", 0) == 0) continue;   // annotations are never compared ([P] Sec. 2 C5)
        out.push_back(l);
    }
    return out;
}

std::string LastLine(const std::string& text) {
    const std::vector<std::string> l = Lines(text);
    return l.empty() ? std::string() : l.back();
}

// An in-process REPL over string streams: armed (CancelScope) like the terminal REPL, but no SIGINT handler.
std::string Repl(const std::string& input) {
    Session s(Reg(), /*verbose=*/false);
    std::istringstream in(input);
    std::ostringstream out;
    s.set_output(out);
    console::RunRepl(s, in, out, /*show_prompts=*/false);
    return out.str();
}

// --- child processes -------------------------------------------------------
struct ChildOut {
    int status = -1;
    std::string out;
};

// Runs this binary with `args` and `env` added to a minimal environment; stdout + stderr captured.
ChildOut RunSelf(const std::vector<std::string>& args, const std::vector<std::string>& env_extra) {
    ChildOut r;
    const std::string exe = console::host::SelfExe();
    int pfd[2];
    if (exe.empty() || ::pipe(pfd) != 0) return r;
    std::vector<std::string> av{exe};
    av.insert(av.end(), args.begin(), args.end());
    std::vector<std::string> ev{"PATH=/usr/bin:/bin", "ENGINE_CUDA_DISABLE=1", "HACS_DEVICE_PROFILE=none"};
    if (const char* h = std::getenv("HOME")) ev.push_back(std::string("HOME=") + h);
    ev.insert(ev.end(), env_extra.begin(), env_extra.end());
    std::vector<char*> argv, envp;
    for (std::string& s : av) argv.push_back(&s[0]);
    argv.push_back(nullptr);
    for (std::string& s : ev) envp.push_back(&s[0]);
    envp.push_back(nullptr);
    const pid_t pid = ::fork();
    if (pid < 0) return r;
    if (pid == 0) {
        ::dup2(pfd[1], 1);
        ::dup2(pfd[1], 2);
        ::close(pfd[0]);
        ::close(pfd[1]);
        ::execve(exe.c_str(), argv.data(), envp.data());
        ::_exit(127);
    }
    ::close(pfd[1]);
    char buf[4096];
    for (;;) {
        const ssize_t n = ::read(pfd[0], buf, sizeof buf);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) break;
        r.out.append(buf, static_cast<std::size_t>(n));
    }
    ::close(pfd[0]);
    ::waitpid(pid, &r.status, 0);
    return r;
}

// --- (A) ---------------------------------------------------------------------
struct Case {
    const char* verb;
    std::string prefix;   // a setting line, or ""
    std::string make;     // binds `x`
    std::string read;     // prints the result (for descent: the lazy member read)
    bool lazy;            // `make` returns a lazy handle; `read` runs the work
};

std::vector<std::string> Trace(const std::string& input) {
    console::SetCancelTest(console::CancelTest{});
    console::SetCancelTrace(true);
    console::ResetCancelVisits();
    (void)Repl(input);
    std::vector<std::string> sites = console::CancelTraceSites();
    console::SetCancelTrace(false);
    console::ResetCancelVisits();
    return sites;
}

struct ArmAStats {
    int verbs = 0, cancels = 0, library_sites = 0;
    std::string sites_seen;
};

void ArmA(ArmAStats& st) {
    const std::vector<Case> cases = {
        {"descent", "set coefficient_ring = \"Z\"", "x = descent(11, 2)", "print(x.digest)", true},
        {"hecke_structure", "set coefficient_ring = \"Z\"", "x = hecke_structure(5, 10, \"anemic\")", "print(x.record_fnv1a64)", false},
        {"level_n", "", "x = level_n(37, 10, 101)", "print(x)", false},
        {"nebentypus", "", "x = nebentypus(13, 4, 10, 37)", "print(x)", false},
        {"hida_tower", "", "x = hida_tower(40, 5, 2)", "print(x)", false},
        {"hida_lambda", "", "x = hida_lambda(11, 100, 3)", "print(x)", false},
        {"forms_algebra", "", "x = forms_algebra(200, 7)", "print(x)", false},
    };
    for (const Case& c : cases) {
        ++st.verbs;
        const std::string pre = c.prefix.empty() ? std::string() : c.prefix + "\n";
        // the cold reference: a child process, no cancellation anywhere
        const std::string code = (c.prefix.empty() ? std::string() : c.prefix + "; ") + c.make + "; " + c.read;
        const ChildOut ref = RunSelf({"-c", code}, {});
        const std::string want = LastLine(ref.out);
        if (!Check(WIFEXITED(ref.status) && WEXITSTATUS(ref.status) == 0 && !want.empty(),
                   std::string("(A) ") + c.verb + ": the cold reference run failed: " + ref.out))
            continue;
        const std::vector<std::string> s0 = Trace(pre);
        const std::vector<std::string> s1 = Trace(pre + c.make + "\n");
        const std::vector<std::string> s2 = Trace(pre + c.make + "\n" + c.read + "\n");
        Check(s2.size() > s1.size() && s1.size() > s0.size(), std::string("(A) ") + c.verb + ": no cancellation point traced");
        bool lib = false;
        for (std::size_t k = s0.size(); k < s2.size(); ++k) {
            const std::string& site = s2[k];
            if (site != "statement" && site != "call") {
                lib = true;
                if (!Contains(st.sites_seen, site + ",")) st.sites_seen += site + ",";
            }
        }
        Check(lib, std::string("(A) ") + c.verb + ": no cancellation point inside the verb was visited");
        if (lib) ++st.library_sites;
        const std::string uname = "unknown name 'x'";
        for (std::size_t k = s0.size() + 1; k <= s2.size(); ++k) {
            const std::string& site = s2[k - 1];
            const bool in_read = k > s1.size();
            // in the read of a lazy handle: the SAME handle is read again; otherwise the binding is re-made
            const std::string prog = (c.lazy && in_read) ? pre + c.make + "\n" + c.read + "\n" + c.read + "\n"
                                                         : pre + c.make + "\n" + c.read + "\n" + c.make + "\n" + c.read + "\n";
            console::CancelTest t;
            t.mode = console::CancelTestMode::After;
            t.k = k;
            console::SetCancelTest(t);
            const std::string out = Repl(prog);
            const bool fired = console::CancelTestFired();
            console::SetCancelTest(console::CancelTest{});
            ++st.cancels;
            const std::string what = std::string("(A) ") + c.verb + " cancelled at visit " + std::to_string(k) + " (" + site + "): ";
            Check(fired, what + "the interlock did not fire");
            Check(Contains(out, "interrupted (at " + site + ")"), what + "no `interrupted` line: " + out);
            if (!in_read || !c.lazy) {
                // the statement stopped before its binding (make) or bound nothing new (read): `x` unbound after a stop in make
                if (!in_read) Check(Contains(out, uname), what + "the stopped statement bound `x`: " + out);
            } else {
                Check(!Contains(out, uname), what + "the lazy handle was lost: " + out);
            }
            Check(LastLine(out) == want, what + "the re-run differs from the cold run: '" + LastLine(out) + "' vs '" + want + "'");
        }
        // after every cancel: a fresh uncancelled session still prints the cold value (no cache entry was poisoned)
        const std::string fresh = Repl(pre + c.make + "\n" + c.read + "\n");
        Check(LastLine(fresh) == want && !Contains(fresh, "interrupted"),
              std::string("(A) ") + c.verb + ": a fresh session after the cancels differs from the cold run: " + fresh);
    }
}

// --- (B) the terminal REPL ---------------------------------------------------
struct Pty {
    int m = -1;
    pid_t pid = -1;
    std::string shown;
    int status = 0;
    bool reaped = false;

    bool start(const std::vector<std::string>& env) {
        m = ::posix_openpt(O_RDWR | O_NOCTTY);
        if (m < 0 || ::grantpt(m) != 0 || ::unlockpt(m) != 0) return false;
        const char* sn = ::ptsname(m);
        if (sn == nullptr) return false;
        const std::string slave = sn;
        const std::string exe = console::host::SelfExe();
        pid = ::fork();
        if (pid < 0) return false;
        if (pid == 0) {
            ::setsid();
            const int s = ::open(slave.c_str(), O_RDWR);   // the controlling terminal: Ctrl-C reaches this process group
            if (s < 0) ::_exit(126);
            ::dup2(s, 0);
            ::dup2(s, 1);
            ::dup2(s, 2);
            if (s > 2) ::close(s);
            ::close(m);
            ::signal(SIGINT, SIG_DFL);
            for (const std::string& e : env) ::putenv(const_cast<char*>(e.c_str()));
            ::execl(exe.c_str(), exe.c_str(), "--console", static_cast<char*>(nullptr));
            ::_exit(127);
        }
        return true;
    }
    void send(const std::string& s) {
        if (::write(m, s.data(), s.size()) != static_cast<ssize_t>(s.size())) {}
    }
    // Reads until `pattern` appears after `from` (true) or the timeout / EOF (false).
    bool until(const std::string& pattern, std::size_t from, int timeout_ms) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        for (;;) {
            if (shown.find(pattern, from) != std::string::npos) return true;
            const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count();
            if (left <= 0) return false;
            struct pollfd p{m, POLLIN, 0};
            const int r = ::poll(&p, 1, static_cast<int>(std::min<long long>(left, 200)));
            if (r > 0) {
                char buf[4096];
                const ssize_t n = ::read(m, buf, sizeof buf);
                if (n <= 0) return shown.find(pattern, from) != std::string::npos;   // EIO: the child closed the terminal
                shown.append(buf, static_cast<std::size_t>(n));
            }
        }
    }
    // Waits for the child (<= timeout); true when it was reaped.
    bool wait(int timeout_ms) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (std::chrono::steady_clock::now() < deadline) {
            (void)until("\x01never\x01", shown.size(), 50);
            if (::waitpid(pid, &status, WNOHANG) == pid) {
                reaped = true;
                return true;
            }
        }
        return false;
    }
    ~Pty() {
        if (pid > 0 && !reaped) {
            ::kill(pid, SIGKILL);
            ::waitpid(pid, &status, 0);
        }
        if (m >= 0) ::close(m);
    }
};

bool KilledBySigint(int st) { return WIFSIGNALED(st) && WTERMSIG(st) == SIGINT; }

void ArmB(const std::string& want_digest, bool& ok) {
    const uint32_t before = Check.fails;
    const std::vector<std::string> env = {"HACS_CANCEL_TEST=wait:1:descent.charpoly", "HACS_DEVICE_PROFILE=none", "ENGINE_CUDA_DISABLE=1"};
    {
        Pty p;
        if (!Check(p.start(env), "(B) cannot open a pseudo-terminal")) return;
        p.send("set coefficient_ring = \"Z\"\nr = descent(11, 2)\nprint(r.digest)\n");
        const bool waiting = p.until("cancel test: waiting at descent.charpoly", 0, 60000);
        Check(waiting, "(B) the REPL did not reach the descent's first charpoly point");
        const std::size_t mark = p.shown.size();
        const auto t0 = std::chrono::steady_clock::now();
        p.send("\x03");
        Check(p.until("interrupted (at descent.charpoly)", mark, 30000), "(B) Ctrl-C printed no `interrupted`");
        const double lat = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        Check(p.until("oc> ", mark, 10000), "(B) the prompt did not return after Ctrl-C");
        const std::size_t mark2 = p.shown.size();
        p.send("print(r.digest)\n");
        Check(p.until(want_digest, mark2, 60000) && p.until("oc> ", mark2 + 1, 10000),
              "(B) the re-read after Ctrl-C did not print the cold digest " + want_digest);
        p.send("\x03");   // at the idle prompt: today's behaviour, the process ends by SIGINT
        Check(p.wait(20000) && KilledBySigint(p.status), "(B) Ctrl-C at the idle prompt did not end the REPL by SIGINT");
        std::printf("  (B) Ctrl-C -> interrupted in %.3f s (the interlock waits at the point)\n", lat);
        if (Check.fails != before) std::cout << "  (B) terminal:\n" << p.shown << std::endl;
    }
    {   // a second Ctrl-C before the first took effect ends the process, as before S5: `hold` keeps the statement 3 s at the
        // point (a long kernel between two points); the two presses are 0.3 s apart (two signals in one write would coalesce)
        std::vector<std::string> henv = env;
        henv[0] = "HACS_CANCEL_TEST=hold:1:descent.charpoly";
        Pty p;
        if (!Check(p.start(henv), "(B) cannot open a pseudo-terminal")) return;
        p.send("set coefficient_ring = \"Z\"\nr = descent(11, 2)\nprint(r.digest)\n");
        Check(p.until("cancel test: waiting at descent.charpoly", 0, 60000), "(B2) the REPL did not reach the point");
        p.send("\x03");
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        p.send("\x03");
        Check(p.wait(20000) && KilledBySigint(p.status), "(B2) a second Ctrl-C before the first took effect did not end the REPL by SIGINT");
        Check(!Contains(p.shown, "interrupted (at"), "(B2) the statement was interrupted although the process should have ended");
    }
    ok = Check.fails == before;
}

// --- (C) the served console --------------------------------------------------
struct ServeChild {
    pid_t pid = -1;
    int in = -1, out = -1;
    std::string obuf;
    bool start(const std::vector<std::string>& env) {
        const std::string exe = console::host::SelfExe();
        int pi[2], po[2];
        if (exe.empty() || ::pipe(pi) != 0 || ::pipe(po) != 0) return false;
        std::vector<std::string> ev = env;
        std::vector<char*> envp;
        for (std::string& s : ev) envp.push_back(&s[0]);
        envp.push_back(nullptr);
        std::string a0 = exe, a1 = "--serve";
        char* argv[] = {&a0[0], &a1[0], nullptr};
        pid = ::fork();
        if (pid < 0) return false;
        if (pid == 0) {
            ::dup2(pi[0], 0);
            ::dup2(po[1], 1);
            const int dn = ::open("/dev/null", O_WRONLY);
            if (dn >= 0) ::dup2(dn, 2);
            ::close(pi[0]);
            ::close(pi[1]);
            ::close(po[0]);
            ::close(po[1]);
            ::execve(exe.c_str(), argv, envp.data());
            ::_exit(127);
        }
        ::close(pi[0]);
        ::close(po[1]);
        in = pi[1];
        out = po[0];
        return true;
    }
    void send(const std::string& line) {
        const std::string s = line + "\n";
        if (::write(in, s.data(), s.size()) != static_cast<ssize_t>(s.size())) {}
    }
    // Reads frames until one contains every needle (returned), or timeout.
    bool frame(const std::vector<std::string>& needles, std::string& got, int timeout_ms) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        for (;;) {
            std::size_t nl;
            while ((nl = obuf.find('\n')) != std::string::npos) {
                const std::string l = obuf.substr(0, nl);
                obuf.erase(0, nl + 1);
                bool all = true;
                for (const std::string& n : needles) all = all && Contains(l, n);
                if (all) {
                    got = l;
                    return true;
                }
            }
            const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count();
            if (left <= 0) return false;
            struct pollfd p{out, POLLIN, 0};
            if (::poll(&p, 1, static_cast<int>(std::min<long long>(left, 200))) > 0) {
                char buf[65536];
                const ssize_t n = ::read(out, buf, sizeof buf);
                if (n <= 0) return false;
                obuf.append(buf, static_cast<std::size_t>(n));
            }
        }
    }
    ~ServeChild() {
        if (in >= 0) ::close(in);
        if (pid > 0) {
            int st = 0;
            for (int i = 0; i < 100 && ::waitpid(pid, &st, WNOHANG) == 0; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(50));
            if (::waitpid(pid, &st, WNOHANG) == 0) {
                ::kill(pid, SIGKILL);
                ::waitpid(pid, &st, 0);
            }
        }
        if (out >= 0) ::close(out);
    }
};

std::string Eval(const std::string& id, const std::string& src) {
    std::string q;
    for (char c : src) {
        if (c == '"' || c == '\\') q += '\\';
        q += c;
    }
    return "{\"op\":\"eval\",\"id\":\"" + id + "\",\"src\":\"" + q + "\"}";
}

void ArmC(const std::string& want_digest, double& latency, bool& ok) {
    const uint32_t before = Check.fails;
    std::vector<std::string> env = {"PATH=/usr/bin:/bin", "ENGINE_CUDA_DISABLE=1", "HACS_CANCEL_TEST=wait:1:descent.charpoly"};
    if (const char* h = std::getenv("HOME")) env.push_back(std::string("HOME=") + h);
    ServeChild s;
    if (!Check(s.start(env), "(C) cannot start --serve")) return;
    std::string f;
    Check(s.frame({"\"kind\":\"hello\""}, f, 30000), "(C) no hello frame");
    s.send(Eval("c1", "set coefficient_ring = \"Z\"; r = descent(11, 2)"));
    Check(s.frame({"\"kind\":\"done\"", "\"id\":\"c1\"", "\"status\":\"ok\""}, f, 60000), "(C) cell c1 did not finish ok: " + f);
    s.send(Eval("c2", "print(r.digest)"));
    Check(s.frame({"\"kind\":\"progress\"", "cancel test: waiting at descent.charpoly"}, f, 60000), "(C) c2 did not reach the point");
    const auto t0 = std::chrono::steady_clock::now();
    s.send("{\"op\":\"interrupt\"}");
    Check(s.frame({"\"kind\":\"done\"", "\"id\":\"c2\"", "\"status\":\"interrupted\"", "\"kind\":\"interrupted\""}, f, 30000),
          "(C) the interrupt did not end c2 as interrupted: " + f);
    latency = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    s.send(Eval("c3", "print(r.digest)"));
    std::string o;
    Check(s.frame({"\"kind\":\"out\"", "\"id\":\"c3\"", want_digest}, o, 60000), "(C) c3 did not print the cold digest " + want_digest);
    Check(s.frame({"\"kind\":\"done\"", "\"id\":\"c3\"", "\"status\":\"ok\""}, f, 30000), "(C) c3 did not finish ok: " + f);
    ok = Check.fails == before;
}

} // namespace

uint32_t ConsoleCancelValidation() {
    std::cout << "--- ConsoleCancelValidation (NBCONFIRM S5: cooperative cancel at the long verbs' points; REPL Ctrl-C; serve interrupt) ---"
              << std::endl;
    Check.reset();
    console::SetCancelTest(console::CancelTest{});
    console::ClearCancel();

    ArmAStats st;
    ArmA(st);

    // the cold digest of the lazy arms
    const ChildOut ref = RunSelf({"-c", "set coefficient_ring = \"Z\"; r = descent(11, 2); print(r.digest)"}, {});
    const std::string want = LastLine(ref.out);
    Check(want.rfind("0x", 0) == 0, "the cold descent digest: " + ref.out);

    // (D) scripts and -c: no handler, the interlock is ignored (prompt-less modes are unchanged)
    const ChildOut d = RunSelf({"-c", "set coefficient_ring = \"Z\"; r = descent(11, 2); print(r.digest)"}, {"HACS_CANCEL_TEST=after:1"});
    Check(WIFEXITED(d.status) && WEXITSTATUS(d.status) == 0 && LastLine(d.out) == want && !Contains(d.out, "interrupted"),
          "(D) -c honoured the cancel interlock: " + d.out);

    bool pty_ok = false;
    ArmB(want, pty_ok);

    std::string serve = "skipped";
    double lat = -1;
    if (oc::kBuildNotebook) {
        bool sok = false;
        ArmC(want, lat, sok);
        serve = sok ? "OK" : "FAIL";
        char b[64];
        std::snprintf(b, sizeof b, "%.3f", lat);
        std::cout << "  (C) serve interrupt -> done interrupted in " << b << " s" << std::endl;
    } else {
        std::cout << "  [SKIP] ConsoleCancelValidation (C): OC_BUILD_NOTEBOOK=OFF" << std::endl;
    }
    console::SetCancelTest(console::CancelTest{});
    console::ClearCancel();

    std::cout << "  points: " << st.sites_seen << std::endl;
    std::cout << "  CONSOLE_CANCEL verbs=" << st.verbs << " cancels=" << st.cancels << " verbs_with_inner_points=" << st.library_sites
              << " pty=" << (pty_ok ? "OK" : "FAIL") << " serve=" << serve << " verdict=" << (Check.fails ? "FAIL" : "OK") << std::endl;
    std::cout << "  " << Check.summary("ConsoleCancelValidation") << std::endl;
    return Check.fails ? 1u : 0u;
}
