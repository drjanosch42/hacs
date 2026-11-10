// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// ConsoleServeValidation.cpp -- see ConsoleServeValidation.h.
// ===========================================================================

#include "ConsoleServeValidation.h"

#include "../BuildSwitches.h"

#if OC_BUILD_NOTEBOOK
#include "Bindings/ReadOnlyTags.h"
#include "Bindings/RegisterPipelineVerbs.h"
#include "Bindings/Serve.h"
#include "Bindings/ServeJson.h"
#include "Core/Ast.h"
#include "Core/Parser.h"
#include "Core/Registry.h"
#include "Core/ScriptRunner.h"
#include "Core/Session.h"
#include "Core/Settings.h"

#include "../Hecke/HeckeSettings.h"
#include "../HeckeExt/RegisterHeckeExtensionVerbs.h"
#include "../../ModArithmetic/Fnv.h"
#include "../Util/TextIo.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <climits>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <map>
#include <poll.h>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

extern char** environ;
#endif

#if OC_BUILD_NOTEBOOK
namespace {

using namespace console;
using serve_json::Json;
using serve_json::Quote;

uint32_t g_checks = 0;
uint32_t g_fail = 0;

void Check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        if (g_fail <= 80) std::cout << "  [FAIL] " << what << std::endl;
    }
}

// ---------------------------------------------------------------------------
// Files and paths
// ---------------------------------------------------------------------------
using oc_util::HexU64;
using oc_util::ReadFile;   // S24 CL-7 (F11): Util/TextIo.h

bool PathExists(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0;
}

std::vector<std::string> SplitLines(const std::string& s) {
    std::vector<std::string> out;
    std::istringstream in(s);
    std::string line;
    while (std::getline(in, line)) out.push_back(line);
    return out;
}

std::string FindRepoRoot() {
    const char* marker = "/src/hacs/Console/scripts/prelude.oc";
#ifdef OC_SOURCE_DIR
    {
        const std::string r(OC_SOURCE_DIR);
        if (!r.empty() && PathExists(r + marker)) return r;
    }
#endif
    char buf[PATH_MAX];
    if (::getcwd(buf, sizeof(buf)) != nullptr) {
        std::string dir(buf);
        for (int up = 0; up < 8; ++up) {
            if (PathExists(dir + marker)) return dir;
            const std::size_t slash = dir.find_last_of('/');
            if (slash == std::string::npos || slash == 0) break;
            dir = dir.substr(0, slash);
        }
    }
    return std::string();
}

std::string SelfExe() {
    char buf[PATH_MAX];
    const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) return std::string();
    buf[n] = '\0';
    return std::string(buf);
}

// Scratch: persistent, never /tmp (the project's disk-crash lesson).
std::string MakeScratch() {
    std::vector<std::string> parents;
    // (S26 INTEG, R1 audit S3) ${XDG_CACHE_HOME:-$HOME/.cache}/hacs/scratch, created if missing; else beside the binary.
    {
        std::string cache;
        if (const char* x = std::getenv("XDG_CACHE_HOME"); x != nullptr && *x != '\0') cache = x;
        else if (const char* h = std::getenv("HOME"); h != nullptr && *h != '\0') cache = std::string(h) + "/.cache";
        if (!cache.empty()) {
            ::mkdir(cache.c_str(), 0755);
            ::mkdir((cache + "/hacs").c_str(), 0755);
            ::mkdir((cache + "/hacs/scratch").c_str(), 0755);
            parents.push_back(cache + "/hacs/scratch");
        }
    }
    const std::string exe = SelfExe();
    if (!exe.empty()) parents.push_back(exe.substr(0, exe.find_last_of('/')));
    for (const std::string& p : parents) {
        if (!PathExists(p)) continue;
        std::string tmpl = p + "/serve_validation.XXXXXX";
        std::vector<char> b(tmpl.begin(), tmpl.end());
        b.push_back('\0');
        if (::mkdtemp(b.data()) != nullptr) return std::string(b.data());
    }
    return std::string();
}

void RemoveTree(const std::string& dir) {
    if (DIR* d = ::opendir(dir.c_str())) {
        while (struct dirent* e = ::readdir(d)) {
            const std::string n = e->d_name;
            if (n == "." || n == "..") continue;
            const std::string full = dir + "/" + n;
            struct stat st;
            if (::lstat(full.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) RemoveTree(full);
            else ::unlink(full.c_str());
        }
        ::closedir(d);
    }
    ::rmdir(dir.c_str());
}

std::size_t DirEntries(const std::string& dir) {
    std::size_t n = 0;
    if (DIR* d = ::opendir(dir.c_str())) {
        while (struct dirent* e = ::readdir(d)) {
            const std::string s = e->d_name;
            if (s != "." && s != "..") ++n;
        }
        ::closedir(d);
    }
    return n;
}

// ---------------------------------------------------------------------------
// The child process
// ---------------------------------------------------------------------------
std::vector<std::string> BaseEnv() {
    std::vector<std::string> env;
    for (const char* k : {"PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "ENGINE_THREADS"}) {
        if (const char* v = std::getenv(k)) env.push_back(std::string(k) + "=" + v);
    }
    env.push_back("ENGINE_CUDA_DISABLE=1");
    return env;
}

std::vector<std::string> WithEnv(std::vector<std::string> env, const std::string& kv) {
    const std::string key = kv.substr(0, kv.find('='));
    env.erase(std::remove_if(env.begin(), env.end(),
                             [&](const std::string& e) { return e.compare(0, key.size() + 1, key + "=") == 0; }),
              env.end());
    env.push_back(kv);
    return env;
}

class Child {
public:
    pid_t pid = -1;
    int in = -1, out = -1, err = -1;
    std::string obuf, ebuf;
    bool out_eof = false;
    bool exited = false;
    int status = 0;
    // stream invariants (A)
    std::int64_t next_seq = 0;
    bool seq_ok = true;
    bool all_json = true;
    bool no_blank = true;
    std::size_t max_frame = 0;
    std::size_t lines = 0;

    Child() = default;
    Child(const Child&) = delete;
    Child& operator=(const Child&) = delete;
    ~Child() {
        if (pid > 0 && !exited) {
            ::kill(pid, SIGKILL);
            ::waitpid(pid, &status, 0);
        }
        for (int fd : {in, out, err})
            if (fd >= 0) ::close(fd);
    }

    bool start(const std::string& exe, const std::vector<std::string>& args, const std::vector<std::string>& env,
               const std::string& cwd, const std::string& stdout_file = std::string()) {
        int pi[2], po[2], pe[2];
        if (::pipe2(pi, O_CLOEXEC) != 0 || ::pipe2(po, O_CLOEXEC) != 0 || ::pipe2(pe, O_CLOEXEC) != 0) return false;
        std::vector<std::string> av;
        av.push_back(exe);
        av.insert(av.end(), args.begin(), args.end());
        std::vector<char*> argv, envp;
        for (std::string& s : av) argv.push_back(&s[0]);
        argv.push_back(nullptr);
        std::vector<std::string> ev = env;
        for (std::string& s : ev) envp.push_back(&s[0]);
        envp.push_back(nullptr);
        const pid_t p = ::fork();
        if (p < 0) return false;
        if (p == 0) {
            ::dup2(pi[0], 0);
            if (!stdout_file.empty()) {
                const int fd = ::open(stdout_file.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
                if (fd < 0) ::_exit(126);
                ::dup2(fd, 1);
            } else {
                ::dup2(po[1], 1);
            }
            ::dup2(pe[1], 2);
            if (!cwd.empty() && ::chdir(cwd.c_str()) != 0) ::_exit(125);
            ::execve(exe.c_str(), argv.data(), envp.data());
            ::_exit(127);
        }
        pid = p;
        ::close(pi[0]);
        ::close(po[1]);
        ::close(pe[1]);
        in = pi[1];
        out = po[0];
        err = pe[0];
        ::fcntl(out, F_SETFL, ::fcntl(out, F_GETFL) | O_NONBLOCK);
        ::fcntl(err, F_SETFL, ::fcntl(err, F_GETFL) | O_NONBLOCK);
        return true;
    }

    bool send(const std::string& s) {
        if (in < 0) return false;
        std::size_t off = 0;
        while (off < s.size()) {
            const ssize_t n = ::write(in, s.data() + off, s.size() - off);
            if (n < 0) {
                if (errno == EINTR) continue;
                if (errno == EAGAIN) {
                    pump(10);
                    continue;
                }
                return false;
            }
            off += static_cast<std::size_t>(n);
        }
        return true;
    }
    bool send_line(const std::string& s) { return send(s + "\n"); }
    void close_in() {
        if (in >= 0) ::close(in);
        in = -1;
    }

    // Reads whatever is available on stdout / stderr within `ms`.
    void pump(int ms) {
        struct pollfd fds[2];
        int n = 0;
        if (!out_eof && out >= 0) fds[n++] = {out, POLLIN, 0};
        if (err >= 0) fds[n++] = {err, POLLIN, 0};
        if (n == 0) {
            if (ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(ms));
            return;
        }
        if (::poll(fds, static_cast<nfds_t>(n), ms) <= 0) return;
        char buf[65536];
        for (int k = 0; k < n; ++k) {
            if ((fds[k].revents & (POLLIN | POLLHUP | POLLERR)) == 0) continue;
            while (true) {
                const ssize_t r = ::read(fds[k].fd, buf, sizeof(buf));
                if (r > 0) {
                    (fds[k].fd == out ? obuf : ebuf).append(buf, static_cast<std::size_t>(r));
                    continue;
                }
                if (r == 0) {
                    if (fds[k].fd == out) out_eof = true;
                    else {
                        ::close(err);
                        err = -1;
                    }
                }
                break;
            }
        }
    }

    bool line(std::string& l, int timeout_ms) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (true) {
            const std::size_t nl = obuf.find('\n');
            if (nl != std::string::npos) {
                l = obuf.substr(0, nl);
                obuf.erase(0, nl + 1);
                ++lines;
                return true;
            }
            if (out_eof) return false;
            const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count();
            if (left <= 0) return false;
            pump(static_cast<int>(std::min<long long>(left, 200)));
        }
    }

    // The next frame; checks the stream invariants on the way.
    bool frame(Json& j, int timeout_ms, std::string* raw = nullptr) {
        std::string l;
        if (!line(l, timeout_ms)) return false;
        if (raw != nullptr) *raw = l;
        if (l.empty()) no_blank = false;
        max_frame = std::max(max_frame, l.size());
        std::string e;
        if (!serve_json::ParseJson(l, j, e) || !j.is_object()) {
            all_json = false;
            j = Json();
            return true;
        }
        if (const Json* s = j.get("seq")) {
            std::int64_t v = -1;
            if (!s->as_int64(v) || v != next_seq) seq_ok = false;
            next_seq = v + 1;
        } else {
            seq_ok = false;
        }
        return true;
    }

    // Waits for the exit; SIGKILL after `timeout_ms`.  Returns the exit code,
    // 128 + signal for a signal, -1 if it had to be killed.
    int wait_exit(int timeout_ms) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (!exited) {
            const pid_t r = ::waitpid(pid, &status, WNOHANG);
            if (r == pid) {
                exited = true;
                break;
            }
            if (std::chrono::steady_clock::now() > deadline) {
                ::kill(pid, SIGKILL);
                ::waitpid(pid, &status, 0);
                exited = true;
                pump(0);
                return -1;
            }
            pump(20);
        }
        for (int k = 0; k < 20 && (!out_eof || err >= 0); ++k) pump(10);
        if (WIFEXITED(status)) return WEXITSTATUS(status);
        if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
        return -1;
    }
};

const char* S(const Json* j) { return (j != nullptr && j->is_string()) ? j->text.c_str() : ""; }
std::int64_t I(const Json* j) {
    std::int64_t v = -999;
    if (j != nullptr) j->as_int64(v);
    return v;
}

// ---------------------------------------------------------------------------
// One request
// ---------------------------------------------------------------------------
struct Cell {
    bool done = false;
    Json done_frame;
    std::string out;      // stdout stream, concatenated
    std::string log;      // log stream
    std::vector<std::string> progress;
    std::vector<std::pair<std::size_t, Json>> values;   // (out length when it arrived, frame)
    std::vector<Json> frames;
    std::vector<std::string> raw;
    std::size_t out_frames = 0;
};

std::string EvalRequest(const std::string& id, const std::string& src, const std::string& args_json = std::string()) {
    std::string r = "{\"op\":\"eval\",\"id\":" + Quote(id) + ",\"src\":" + Quote(src);
    if (!args_json.empty()) r += ",\"args\":" + args_json;
    return r + "}";
}

// Collects frames until the `done` of `id` (frames of other ids are kept too).
bool Collect(Child& c, const std::string& id, Cell& cell, int timeout_ms) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (true) {
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count();
        if (left <= 0) return false;
        Json f;
        std::string raw;
        if (!c.frame(f, static_cast<int>(left), &raw)) return false;
        cell.frames.push_back(f);
        cell.raw.push_back(raw);
        const std::string kind = S(f.get("kind"));
        const Json* fid = f.get("id");
        const bool mine = fid != nullptr && fid->is_string() && fid->text == id;
        if (kind == "out" && mine) {
            ++cell.out_frames;
            (std::string(S(f.get("stream"))) == "log" ? cell.log : cell.out) += S(f.get("text"));
        } else if (kind == "progress" && mine) {
            cell.progress.push_back(S(f.get("text")));
        } else if (kind == "value" && mine) {
            cell.values.push_back({cell.out.size(), f});
        } else if (kind == "done" && mine) {
            cell.done = true;
            cell.done_frame = f;
            return true;
        } else if (kind == "fatal") {
            return false;
        }
    }
}

bool RunCell(Child& c, const std::string& id, const std::string& src, Cell& cell, int timeout_ms,
             const std::string& args_json = std::string()) {
    if (!c.send_line(EvalRequest(id, src, args_json))) return false;
    return Collect(c, id, cell, timeout_ms);
}

std::string Status(const Cell& c) { return S(c.done_frame.get("status")); }
std::int64_t Exit(const Cell& c) { return I(c.done_frame.get("exit")); }
std::string ErrKind(const Cell& c) {
    const Json* e = c.done_frame.get("error");
    return (e != nullptr && e->is_object()) ? S(e->get("kind")) : "";
}
std::string ErrMsg(const Cell& c) {
    const Json* e = c.done_frame.get("error");
    return (e != nullptr && e->is_object()) ? S(e->get("message")) + std::string("\n") + S(e->get("detail")) : "";
}

// ---------------------------------------------------------------------------
// Scripts
// ---------------------------------------------------------------------------
const char* const kExpectedHeader = "#! oc-format 1";

std::string TranscriptOf(const std::string& raw) {
    std::string out;
    for (const std::string& line : SplitLines(raw)) {
        if (line.rfind("#~", 0) == 0) continue;
        std::size_t end = line.size();
        while (end > 0 && (line[end - 1] == ' ' || line[end - 1] == '\r' || line[end - 1] == '\t')) --end;
        out += line.substr(0, end);
        out += '\n';
    }
    return out;
}

std::string ExpectedBody(const std::string& file_text) {
    const std::size_t nl = file_text.find('\n');
    return nl == std::string::npos ? std::string() : file_text.substr(nl + 1);
}

// Replaces every `include("x.oc");` line by the file's text (cycle-checked),
// which is what `include` does in the same global frame.
bool Flatten(const std::string& dir, const std::string& name, std::vector<std::string>& stack, std::string& out) {
    for (const std::string& s : stack)
        if (s == name) return false;
    std::string text;
    if (!ReadFile(dir + "/" + name, text)) return false;
    stack.push_back(name);
    for (const std::string& line : SplitLines(text)) {
        std::size_t b = line.find_first_not_of(" \t");
        if (b != std::string::npos && line.compare(b, 9, "include(\"") == 0) {
            const std::size_t q = line.find('"', b + 9);
            if (q == std::string::npos) return false;
            std::string inner;
            if (!Flatten(dir, line.substr(b + 9, q - b - 9), stack, inner)) return false;
            out += inner;
            continue;
        }
        out += line + "\n";
    }
    stack.pop_back();
    return true;
}

std::vector<std::string> NumberedScripts(const std::string& dir) {
    std::vector<std::string> out;
    if (DIR* d = ::opendir(dir.c_str())) {
        while (struct dirent* e = ::readdir(d)) {
            const std::string n = e->d_name;
            if (n.size() < 4 || n.substr(n.size() - 3) != ".oc") continue;
            if (n[0] < '0' || n[0] > '9') continue;
            out.push_back(n);
        }
        ::closedir(d);
    }
    std::sort(out.begin(), out.end());
    return out;
}

// The registry facts the harness needs (computed in process, never from the child).
struct RegistryFacts {
    std::set<std::string> unserved;                         // no tagged overload
    std::map<std::string, std::set<std::size_t>> refuser_arity;   // verb -> param counts of untagged overloads
    std::map<std::string, std::set<std::size_t>> served_arity;
    std::set<std::string> setting_keys;
    std::vector<std::pair<std::string, std::string>> tagged;  // (verb, sig) of every read-only overload, full order
    std::vector<bool> is_tagged;                              // per full-registry overload
    serve_tags::TagCensus census;
    std::size_t types = 0;
};

RegistryFacts Facts() {
    RegistryFacts f;
    Registry reg;
    Session session(reg, false);
    RegisterPipelineVerbs(reg);
    hecke_ext::RegisterHeckeExtensionVerbs(reg);
    reg.install_settings(session.settings());
    for (const std::string& k : session.settings().names()) f.setting_keys.insert(k);
    f.census = serve_tags::CensusOf(reg);
    f.types = reg.type_names().size();
    for (const std::string& name : reg.verb_names()) {
        const VerbEntry* v = reg.find_verb(name);
        bool any = false;
        for (const Overload& o : v->overloads) {
            const bool ro = serve_tags::IsReadOnly(*v, o);
            f.is_tagged.push_back(ro);
            if (ro) f.tagged.push_back({name, o.sig.to_string(name)});
            any = any || ro;
            (ro ? f.served_arity : f.refuser_arity)[name].insert(o.sig.params.size());
        }
        if (!any) f.unserved.insert(name);
    }
    return f;
}

struct Offender {
    std::string what;   // verb name or "set <key>"
    std::string kind;   // forbidden / forbidden_overload
};

void FirstUnserved(const ExprPtr& e, const RegistryFacts& f, std::string& hit) {
    if (!e || !hit.empty()) return;
    if (e->kind == ExprKind::Call && f.unserved.count(e->text)) {
        hit = e->text;
        return;
    }
    FirstUnserved(e->lhs, f, hit);
    FirstUnserved(e->rhs, f, hit);
    for (const ExprPtr& x : e->items) FirstUnserved(x, f, hit);
    for (const Argument& a : e->args) FirstUnserved(a.value, f, hit);
}
void FirstUnservedStmt(const StmtPtr& s, const RegistryFacts& f, std::string& hit) {
    if (!s || !hit.empty()) return;
    if (s->kind == StmtKind::Include) { hit = "include"; return; }
    if (s->kind == StmtKind::Quit) { hit = "quit"; return; }
    FirstUnserved(s->value, f, hit);
    for (const StmtPtr& b : s->body) FirstUnservedStmt(b, f, hit);
    for (const StmtPtr& b : s->orelse) FirstUnservedStmt(b, f, hit);
}
void FirstDynamic(const ExprPtr& e, const RegistryFacts& f, std::string& hit) {
    if (!e || !hit.empty()) return;
    if (e->kind == ExprKind::Call && f.refuser_arity.count(e->text)) {
        const std::size_t n = e->args.size();
        const auto& ra = f.refuser_arity.at(e->text);
        const auto it = f.served_arity.find(e->text);
        if (ra.count(n) && (it == f.served_arity.end() || !it->second.count(n))) {
            hit = e->text;
            return;
        }
    }
    FirstDynamic(e->lhs, f, hit);
    FirstDynamic(e->rhs, f, hit);
    for (const ExprPtr& x : e->items) FirstDynamic(x, f, hit);
    for (const Argument& a : e->args) FirstDynamic(a.value, f, hit);
}
void FirstDynamicStmt(const StmtPtr& s, const RegistryFacts& f, std::string& hit) {
    if (!s || !hit.empty()) return;
    if (s->kind == StmtKind::Set && f.setting_keys.count(s->name) && !serve_tags::IsServeSettable(s->name)) {
        hit = "set " + s->name;
        return;
    }
    FirstDynamic(s->value, f, hit);
    for (const StmtPtr& b : s->body) FirstDynamicStmt(b, f, hit);
    for (const StmtPtr& b : s->orelse) FirstDynamicStmt(b, f, hit);
}

// The first refused construct of a (flattened) script, as the server decides it.
Offender Classify(const std::string& src, const RegistryFacts& f) {
    Offender o;
    Program p;
    try {
        p = Parse(src, "<cell>");
    } catch (...) {
        o.what = "parse";
        o.kind = "parse";
        return o;
    }
    std::string hit;
    for (const StmtPtr& s : p.statements) FirstUnservedStmt(s, f, hit);
    if (!hit.empty()) {
        o.what = hit;
        o.kind = "forbidden";
        return o;
    }
    for (const StmtPtr& s : p.statements) FirstDynamicStmt(s, f, hit);
    if (!hit.empty()) {
        o.what = hit;
        o.kind = "forbidden_overload";
    }
    return o;
}

// The EXEMPT table of [P] Sec. 18.11 (B): {script, first untagged call, expected kind}.
struct Exempt {
    const char* script;
    const char* first;
    const char* kind;
};
const Exempt kExempt[] = {
    {"014_expect_error.oc", "report", "forbidden_overload"},
    {"021_settings_resume.oc", "set threads", "forbidden_overload"},
    {"022_harness.oc", "run_validation", "forbidden"},
    {"048_c5c_ring_campaign.oc", "run_campaign", "forbidden"},
};

std::string NormaliseDigits(const std::string& s) {
    std::string o;
    bool in = false;
    for (char c : s) {
        if (c >= '0' && c <= '9') {
            if (!in) o += '#';
            in = true;
        } else {
            in = false;
            o += c;
        }
    }
    return o;
}

// ---------------------------------------------------------------------------
// (C) envelope invariants
// ---------------------------------------------------------------------------
bool NumbersOk(const Json& j) {
    switch (j.kind) {
        case Json::Kind::Number: {
            std::int64_t v = 0;
            if (!j.as_int64(v)) return false;
            return v <= 9007199254740991LL && v >= -9007199254740991LL;
        }
        case Json::Kind::Array:
            for (const Json& x : j.items)
                if (!NumbersOk(x)) return false;
            return true;
        case Json::Kind::Object:
            for (const auto& m : j.members)
                if (!NumbersOk(m.second)) return false;
            return true;
        default:
            return true;
    }
}

// Structural checks of one envelope (recursively); returns "" or the reason.
std::string EnvelopeProblem(const Json& e, int depth = 0) {
    if (depth > 64) return "too deep";
    static const std::vector<std::string> head = {"t", "canonical", "hash", "v"};
    if (!e.is_object() || e.members.size() < 4) return "not an envelope";
    for (std::size_t i = 0; i < 4; ++i)
        if (e.members[i].first != head[i]) return "key order";
    const Json* t = e.get("t");
    const Json* h = e.get("hash");
    if (!t->is_string() || !e.get("canonical")->is_string()) return "t / canonical not strings";
    if (!(h->is_null() || (h->is_string() && h->text.size() == 18 && h->text.rfind("0x", 0) == 0))) return "hash form";
    const Json* v = e.get("v");
    const std::string& tn = t->text;
    if (tn == "List" && v->is_array()) {
        for (const Json& x : v->items) {
            const std::string p = EnvelopeProblem(x, depth + 1);
            if (!p.empty()) return "list element: " + p;
        }
    } else if (tn == "Record") {
        if (!v->is_array()) return "a Record v is not an array of pairs";
        for (const Json& pr : v->items) {
            if (!pr.is_array() || pr.items.size() != 2 || !pr.items[0].is_string()) return "a Record pair";
            const std::string p = EnvelopeProblem(pr.items[1], depth + 1);
            if (!p.empty()) return "record value: " + p;
        }
    } else if (tn == "Poly" && v->is_object()) {
        const Json* o = v->get("order");
        if (o == nullptr || o->text != "low_to_high") return "Poly without order low_to_high";
    } else if (tn == "Matrix" && v->is_object()) {
        const Json* en = v->get("entries");
        const std::int64_t rows = I(v->get("rows")), cols = I(v->get("cols"));
        if (en != nullptr) {
            if (!en->is_array() || static_cast<std::int64_t>(en->items.size()) != rows) return "Matrix rows";
            for (const Json& r : en->items)
                if (!r.is_array() || static_cast<std::int64_t>(r.items.size()) != cols) return "Matrix cols";
        } else {
            const Json* tr = e.get("truncated");
            if (tr == nullptr || !tr->b) return "Matrix without entries is not truncated";
        }
    }
    return std::string();
}

bool VolatileKey(const std::string& k) {
    return k.rfind("wall_", 0) == 0 || k.rfind("alloc", 0) == 0 || k.find("rss") != std::string::npos;
}

// Strips the volatile fields of an envelope ([P] Sec. 18.5); true when any was found.
bool StripVolatile(Json& e) {
    bool any = false;
    Json* v = nullptr;
    Json* t = nullptr;
    for (auto& m : e.members) {
        if (m.first == "v") v = &m.second;
        if (m.first == "t") t = &m.second;
    }
    if (v == nullptr || t == nullptr) return false;
    if (t->text == "Record" && v->is_array()) {
        std::vector<Json> kept;
        for (Json& pr : v->items) {
            if (pr.items.size() == 2 && VolatileKey(pr.items[0].text)) {
                any = true;
                continue;
            }
            if (pr.items.size() == 2) any = StripVolatile(pr.items[1]) || any;
            kept.push_back(pr);
        }
        v->items = kept;
    } else if (t->text == "List" && v->is_array()) {
        for (Json& x : v->items) any = StripVolatile(x) || any;
    }
    if (any)
        for (auto& m : e.members)
            if (m.first == "canonical") m.second.text.clear();
    return any;
}

// The determinism normal form of one frame (seq, perf, pid and progress dropped).
std::string NormalFrame(const Json& f) {
    const std::string kind = S(f.get("kind"));
    if (kind == "hello" || kind == "progress") return std::string();
    Json g;
    g.kind = Json::Kind::Object;
    for (const auto& m : f.members) {
        if (m.first == "seq" || m.first == "perf") continue;
        Json x = m.second;
        if (kind == "value" && m.first == "value") StripVolatile(x);
        g.members.push_back({m.first, x});
    }
    return serve_json::WriteJson(g);
}

// ---------------------------------------------------------------------------
// A pass over the script set (arm B, and the determinism arm)
// ---------------------------------------------------------------------------
struct ScriptRun {
    std::string name;
    bool started = false;
    bool hello = false;
    Cell cell;
    int exit_code = -2;
    std::string coverage;   // the child's coverage mask
    bool stream_ok = true;
    std::size_t max_frame = 0;
    std::string stderr_text;
};

struct PassResult {
    std::vector<ScriptRun> runs;
    std::uint64_t digest = engine::kFnvOffset64;
};

std::string CoverageOf(const std::string& err) {
    const std::string tag = "serve: coverage ";
    const std::size_t at = err.rfind(tag);
    if (at == std::string::npos) return std::string();
    const std::size_t m = err.find("mask=", at);
    if (m == std::string::npos) return std::string();
    const std::size_t e = err.find('\n', m);
    return err.substr(m + 5, e == std::string::npos ? std::string::npos : e - m - 5);
}

void RunOne(const std::string& exe, const std::vector<std::string>& env, const std::vector<std::string>& args,
            const std::string& cwd, const std::string& name, const std::string& src, ScriptRun& r, int timeout_ms) {
    r.name = name;
    Child c;
    r.started = c.start(exe, args, env, cwd);
    if (!r.started) return;
    Json hello;
    r.hello = c.frame(hello, 60000) && std::string(S(hello.get("kind"))) == "hello";
    if (r.hello) RunCell(c, "s", src, r.cell, timeout_ms);
    c.send_line("{\"op\":\"close\"}");
    Json bye;
    c.frame(bye, 10000);
    c.close_in();
    r.exit_code = c.wait_exit(10000);
    r.stream_ok = c.seq_ok && c.all_json && c.no_blank;
    r.max_frame = c.max_frame;
    r.stderr_text = c.ebuf;
    r.coverage = CoverageOf(c.ebuf);
}

PassResult RunPass(const std::string& exe, const std::vector<std::string>& env, const std::vector<std::string>& args,
                   const std::string& cwd, const std::vector<std::pair<std::string, std::string>>& scripts) {
    PassResult p;
    for (const auto& s : scripts) {
        ScriptRun r;
        RunOne(exe, env, args, cwd, s.first, s.second, r, 600000);
        std::string norm = s.first + "\n";
        for (const Json& f : r.cell.frames) norm += NormalFrame(f) + "\n";
        p.digest = engine::FnvBytes(p.digest, norm);
        p.runs.push_back(std::move(r));
    }
    return p;
}

// ---------------------------------------------------------------------------
// The arms
// ---------------------------------------------------------------------------
struct Ctx {
    std::string exe, root, scratch, cwd;
    std::vector<std::string> env;
    RegistryFacts facts;
    std::size_t cells = 0;
    std::vector<bool> covered;
};

std::vector<std::string> SettableSpec() { return serve_tags::ServeSettableKeys(); }

void ArmA(Ctx& x) {
    Child c;
    Check(c.start(x.exe, {"--serve"}, x.env, x.cwd), "(A) the served child starts");
    Json h;
    std::string raw;
    const bool got = c.frame(h, 60000, &raw);
    Check(got, "(A) a hello frame arrives: stderr '" + c.ebuf.substr(0, 300) + "'");
    if (!got) return;
    Check(std::string(S(h.get("kind"))) == "hello" && I(h.get("seq")) == 0, "(A) the first frame is hello, seq 0");
    Check(std::string(S(h.get("proto"))) == "oc-serve/1", "(A) proto oc-serve/1");
    Check(I(h.get("json_version")) == 1 && I(h.get("oc_format")) == 1, "(A) json_version 1, oc_format 1");
    Check(std::string(S(h.get("console"))) == "3.10", "(A) console 3.10");
    const Json* b = h.get("build");
    Check(b != nullptr && b->get("commit") != nullptr && b->get("config") != nullptr && b->get("cuda") != nullptr &&
              b->get("pid") != nullptr,
          "(A) build = {commit, config, cuda, pid}");
    if (b != nullptr && b->get("cuda") != nullptr) Check(!b->get("cuda")->b, "(A) cuda is false under ENGINE_CUDA_DISABLE=1");
    Check(h.get("read_only") != nullptr && h.get("read_only")->b, "(A) read_only true");
    const Json* n = h.get("counts");
    Check(n != nullptr && I(n->get("verbs")) == static_cast<std::int64_t>(x.facts.census.read_only_verbs) &&
              I(n->get("overloads")) == static_cast<std::int64_t>(x.facts.census.read_only) &&
              I(n->get("types")) == static_cast<std::int64_t>(x.facts.types),
          "(A) counts = the registry's tagged counts (" + std::to_string(x.facts.census.read_only_verbs) + " verbs, " +
              std::to_string(x.facts.census.read_only) + " overloads)");
    const Json* l = h.get("limits");
    Check(l != nullptr && I(l->get("rss_cap_mib")) == 0 && I(l->get("max_reply_bytes")) == 1048576 &&
              I(l->get("max_request_bytes")) == 8388608 && I(l->get("confirm_wall_s")) == 60,
          "(A) limits = {0, 1048576, 8388608, confirm_wall_s 60} by default");
    std::vector<std::string> settable;
    if (const Json* s = h.get("settable"))
        for (const Json& k : s->items) settable.push_back(k.text);
    Check(settable == SettableSpec(), "(A) settable equals [P] Sec. 18.8's list");
    const std::vector<std::string> order = {"kind", "seq", "proto", "json_version", "oc_format", "console", "build",
                                            "read_only", "counts", "limits", "settable"};
    std::vector<std::string> keys;
    for (const auto& m : h.members) keys.push_back(m.first);
    Check(keys == order, "(A) the hello keys are in the fixed order");
    Cell cell;
    Check(RunCell(c, "a1", "print(1 + 1)", cell, 30000) && cell.out == "2\n" && Status(cell) == "ok",
          "(A) a first eval answers");
    ++x.cells;
    c.send_line("{\"op\":\"close\"}");
    Json bye;
    Check(c.frame(bye, 10000) && std::string(S(bye.get("kind"))) == "bye", "(A) close answers bye");
    c.close_in();
    Check(c.wait_exit(10000) == 0, "(A) close exits 0");
    Check(c.seq_ok && c.all_json && c.no_blank, "(A) every frame parsed, no blank line, seq strictly +1");
}

void ArmBC(Ctx& x, std::size_t& n_scripts, std::size_t& n_exempt, std::size_t& n_values, std::string& digest_text) {
    const std::string dir = x.root + "/src/hacs/Console/scripts";
    const std::vector<std::string> names = NumberedScripts(dir);
    Check(names.size() >= 38, "(B) the numbered script set is present (" + std::to_string(names.size()) + ")");
    std::vector<std::pair<std::string, std::string>> runnable, exempt;
    std::map<std::string, Offender> exempt_found;
    for (const std::string& n : names) {
        std::string src;
        std::vector<std::string> stack;
        if (!Flatten(dir, n, stack, src)) {
            Check(false, "(B) " + n + ": its includes flatten");
            continue;
        }
        const Offender o = Classify(src, x.facts);
        if (o.kind.empty()) runnable.push_back({n, src});
        else {
            exempt.push_back({n, src});
            exempt_found[n] = o;
        }
    }
    // The exempt table is exactly the recomputed set.
    std::set<std::string> table;
    for (const Exempt& e : kExempt) {
        table.insert(e.script);
        const auto it = exempt_found.find(e.script);
        Check(it != exempt_found.end(), std::string("(B) the exempt script ") + e.script +
                                            " still calls an untagged overload (the table must stay minimal)");
        if (it != exempt_found.end())
            Check(it->second.what == e.first && it->second.kind == e.kind,
                  std::string("(B) ") + e.script + ": the first refused construct is " + it->second.what + " (" +
                      it->second.kind + "), the table says " + e.first + " (" + e.kind + ")");
    }
    for (const auto& kv : exempt_found)
        Check(table.count(kv.first) != 0, "(B) " + kv.first + " calls an untagged overload (" + kv.second.what +
                                              ") but is not in the exempt table");
    n_scripts = runnable.size();
    n_exempt = exempt.size();

    // --- (B) the runnable scripts, default configuration --------------------
    const PassResult pass = RunPass(x.exe, x.env, {"--serve"}, x.cwd, runnable);
    x.cells += runnable.size();
    for (std::size_t k = 0; k < pass.runs.size(); ++k) {
        const ScriptRun& r = pass.runs[k];
        const std::string& n = r.name;
        Check(r.started && r.hello, "(B) " + n + ": the child starts and says hello");
        Check(r.cell.done && Status(r.cell) == "ok" && Exit(r.cell) == 0,
              "(B) " + n + ": done ok / exit 0 (got " + Status(r.cell) + " / " + std::to_string(Exit(r.cell)) + ": " +
                  ErrMsg(r.cell).substr(0, 300) + ")");
        Check(r.exit_code == 0, "(B) " + n + ": the child exits 0 after close");
        Check(r.stream_ok, "(B) " + n + ": every frame parses, seq strictly +1, no blank line");
        Check(r.max_frame <= 1048576, "(B) " + n + ": no frame exceeds max_reply_bytes");
        Check(r.cell.log.empty(), "(B) " + n + ": no captured chatter (verbose = false)");
        std::string expected;
        const std::string ep = dir + "/" + n.substr(0, n.size() - 3) + ".expected";
        if (!ReadFile(ep, expected)) {
            Check(false, "(B) " + n + ": the .expected is readable");
        } else {
            Check(SplitLines(expected).size() > 0 && SplitLines(expected)[0] == kExpectedHeader,
                  "(B) " + n + ": the .expected carries the oc-format header");
            const std::string got = TranscriptOf(r.cell.out);
            const bool same = (got == ExpectedBody(expected));
            Check(same, "(B) " + n + ": the served out text is byte for byte the .expected body");
            if (!same) {
                const std::vector<std::string> a = SplitLines(ExpectedBody(expected)), b = SplitLines(got);
                std::size_t i = 0;
                while (i < a.size() && i < b.size() && a[i] == b[i]) ++i;
                std::cout << "    first difference at line " << (i + 1) << ": expected '"
                          << (i < a.size() ? a[i] : "<eof>") << "', served '" << (i < b.size() ? b[i] : "<eof>")
                          << "'" << std::endl;
            }
        }
        // Coverage (C2).
        if (r.coverage.size() == x.covered.size())
            for (std::size_t i = 0; i < r.coverage.size(); ++i) x.covered[i] = x.covered[i] || r.coverage[i] == '1';
        else
            Check(false, "(C2) " + n + ": the child reports its coverage mask (stderr '" + r.stderr_text.substr(0, 200) + "')");

        // (C) values.
        for (const auto& pv : r.cell.values) {
            ++n_values;
            const Json* env = pv.second.get("value");
            const std::string prob = env == nullptr ? std::string("no value") : EnvelopeProblem(*env);
            Check(prob.empty(), "(C) " + n + ": a value envelope is well formed: " + prob);
            Check(NumbersOk(pv.second), "(C) " + n + ": every number of a value frame obeys the 2^53 integer rule");
            if (env != nullptr && env->get("canonical") != nullptr) {
                const std::string canon = env->get("canonical")->text + "\n";
                const std::string before = r.cell.out.substr(0, pv.first);
                Check(before.size() >= canon.size() && before.compare(before.size() - canon.size(), canon.size(), canon) == 0,
                      "(C) " + n + ": a value's canonical is what its statement printed");
            }
        }
    }

    // --- (B) progress frames against the in-process run ----------------------
    for (std::size_t k = 0; k < runnable.size() && k < pass.runs.size(); ++k) {
        const hecke::ScopedMathSettings restore(hecke::math_settings());
        Registry reg;
        std::ostringstream out;
        Session session(reg, false);
        RegisterPipelineVerbs(reg);
        hecke_ext::RegisterHeckeExtensionVerbs(reg);
        reg.install_settings(session.settings());
        session.settings().set_baseline(session.settings().snapshot());
        session.set_output(out);
        const int code = RunSource(session, runnable[k].second, "<cell>");
        std::vector<std::string> want;
        for (const std::string& l : SplitLines(out.str()))
            if (l.rfind("#~", 0) == 0) {
                std::string t = l.substr(2);
                if (!t.empty() && t[0] == ' ') t.erase(0, 1);
                want.push_back(NormaliseDigits(t));
            }
        std::vector<std::string> got;
        for (const std::string& p : pass.runs[k].cell.progress) got.push_back(NormaliseDigits(p));
        Check(code == 0 && got == want, "(B) " + runnable[k].first + ": the progress frames are the #~ lines of the "
                                         "in-process run (" + std::to_string(got.size()) + " vs " +
                                         std::to_string(want.size()) + ")");
        ++x.cells;
    }

    // --- (B) the exempt scripts are rejected as the table states --------------
    for (const auto& s : exempt) {
        ScriptRun r;
        RunOne(x.exe, x.env, {"--serve"}, x.cwd, s.first, s.second, r, 600000);
        ++x.cells;
        const auto it = exempt_found.find(s.first);
        const std::string kind = it == exempt_found.end() ? std::string() : it->second.kind;
        Check(r.cell.done, "(B) exempt " + s.first + ": a done arrives");
        if (kind == "forbidden") {
            Check(Status(r.cell) == "rejected" && Exit(r.cell) == 1 && ErrKind(r.cell) == "forbidden" &&
                      I(r.cell.done_frame.get("statements")) == 0 && r.cell.out.empty(),
                  "(B) exempt " + s.first + ": rejected / forbidden / exit 1 before anything ran");
        } else {
            Check(Status(r.cell) == "error" && Exit(r.cell) == 3 && ErrKind(r.cell) == "forbidden_overload",
                  "(B) exempt " + s.first + ": error / forbidden_overload / exit 3 (got " + Status(r.cell) + " / " +
                      ErrKind(r.cell) + ")");
        }
        if (r.coverage.size() == x.covered.size())
            for (std::size_t i = 0; i < r.coverage.size(); ++i) x.covered[i] = x.covered[i] || r.coverage[i] == '1';
    }

    // --- (C) determinism: ENGINE_THREADS 1 / 2 / 8, program cache off ---------
    struct Variant {
        std::string label;
        std::vector<std::string> env;
        std::vector<std::string> args;
    };
    std::vector<Variant> variants = {
        {"threads=1", WithEnv(x.env, "ENGINE_THREADS=1"), {"--serve"}},
        {"threads=2", WithEnv(x.env, "ENGINE_THREADS=2"), {"--serve"}},
        {"threads=8", WithEnv(x.env, "ENGINE_THREADS=8"), {"--serve"}},
        {"program_cache=false", x.env, {"--serve", "--set", "program_cache=false"}},
    };
    std::vector<PassResult> results(variants.size());
    std::vector<std::thread> threads;
    for (std::size_t v = 0; v < variants.size(); ++v)
        threads.emplace_back([&, v] { results[v] = RunPass(x.exe, variants[v].env, variants[v].args, x.cwd, runnable); });
    for (std::thread& t : threads) t.join();
    digest_text = "default=" + HexU64(pass.digest);
    for (std::size_t v = 0; v < variants.size(); ++v) {
        x.cells += runnable.size();
        digest_text += " " + variants[v].label + "=" + HexU64(results[v].digest);
        bool all_ok = true;
        for (const ScriptRun& r : results[v].runs) all_ok = all_ok && r.cell.done && Status(r.cell) == "ok";
        Check(all_ok, "(C) determinism " + variants[v].label + ": every script ran ok");
        Check(results[v].digest == pass.digest, "(C) determinism: the frame stream at " + variants[v].label +
                                                    " hashes as the default run (" + HexU64(results[v].digest) + " vs " +
                                                    HexU64(pass.digest) + ")");
        if (results[v].digest != pass.digest) {
            for (std::size_t k = 0; k < results[v].runs.size() && k < pass.runs.size(); ++k) {
                const auto& a = pass.runs[k].cell.frames;
                const auto& b = results[v].runs[k].cell.frames;
                for (std::size_t i = 0; i < std::max(a.size(), b.size()); ++i) {
                    const std::string fa = i < a.size() ? NormalFrame(a[i]) : "<none>";
                    const std::string fb = i < b.size() ? NormalFrame(b[i]) : "<none>";
                    if (fa != fb) {
                        std::cout << "    " << pass.runs[k].name << " frame " << i << ": '" << fa.substr(0, 200)
                                  << "' vs '" << fb.substr(0, 200) << "'" << std::endl;
                        break;
                    }
                }
            }
        }
    }
}

void ArmCHash(Ctx& x) {
    Child c;
    if (!c.start(x.exe, {"--serve"}, x.env, x.cwd)) {
        Check(false, "(C) hash: the child starts");
        return;
    }
    Json h;
    c.frame(h, 60000);
    const std::vector<std::pair<std::string, std::string>> cases = {
        {"A = matrix(7, [[1, 2], [3, 4]])", "A"},
        {"n = 5", "n"},
        {"s = \"text\"", "s"},
        {"H = cohomology(symmetric_power(sl2z_model(), 10, 691))", "H"}};
    int k = 0;
    for (const auto& cs : cases) {
        Cell a, b, d;
        RunCell(c, "h" + std::to_string(k) + "a", cs.first, a, 60000);
        if (cs.second == "H") RunCell(c, "h" + std::to_string(k) + "c", "compute(H);", d, 300000);
        RunCell(c, "h" + std::to_string(k) + "b", cs.second, b, 60000);
        Cell p;
        RunCell(c, "h" + std::to_string(k) + "p", "print(hash(" + cs.second + "))", p, 60000);
        x.cells += 3;
        const Json* env = b.values.empty() ? nullptr : b.values[0].second.get("value");
        Check(env != nullptr && EnvelopeProblem(*env).empty() && NumbersOk(b.values[0].second),
              "(C) the value frame of " + cs.second + " is a well-formed envelope");
        Check(env != nullptr && env->get("canonical") != nullptr && env->get("canonical")->text + "\n" == b.out,
              "(C) the canonical of " + cs.second + " is what its echo printed");
        // hash(x) prints its text, or fails (no golden hash): the envelope says the same, or null.
        const bool has = p.done && Status(p) == "ok";
        Check(env != nullptr && env->get("hash") != nullptr &&
                  (has ? (env->get("hash")->is_string() && env->get("hash")->text + "\n" == p.out)
                       : env->get("hash")->is_null()),
              "(C) the envelope hash of " + cs.second + " equals hash(" + cs.second + ") (" +
                  (env != nullptr && env->get("hash") != nullptr ? serve_json::WriteJson(*env->get("hash")) : "none") +
                  " vs " + p.out + ")");
        Cell t;
        RunCell(c, "h" + std::to_string(k) + "t", "print(type(" + cs.second + "))", t, 60000);
        Check(env != nullptr && std::string(S(env->get("t"))) + "\n" == t.out, "(C) t of " + cs.second + " is type(x)");
        ++k;
    }
    c.close_in();
    c.wait_exit(10000);
}

// One served session for the error / survival arms.
struct Session1 {
    Child c;
    int n = 0;
    bool ok = false;
    Session1(Ctx& x, const std::vector<std::string>& args = {"--serve"}) {
        ok = c.start(x.exe, args, x.env, x.cwd);
        Json h;
        ok = ok && c.frame(h, 60000) && std::string(S(h.get("kind"))) == "hello";
    }
    Cell run(const std::string& src, int timeout_ms = 120000) {
        Cell cell;
        RunCell(c, "c" + std::to_string(n++), src, cell, timeout_ms);
        return cell;
    }
    bool survives() {
        Cell cell = run("print(1 + 1)");
        return cell.done && Status(cell) == "ok" && cell.out == "2\n";
    }
};

void ArmD(Ctx& x) {
    Session1 s(x);
    Check(s.ok, "(D) the child starts");
    if (!s.ok) return;
    struct Row {
        const char* label;
        const char* src;
        const char* status;
        int exit;
        const char* kind;
        int line;
        int created_line;
    };
    const std::string sentinel = x.scratch + "/sentinel_state.json";
    const std::vector<Row> rows = {
        {"parse", "x = (", "error", 2, "parse", 1, 0},
        {"runtime", "y = 1\nz = y / 0", "error", 3, "runtime", 2, 0},
        {"type", "abs(\"a\")", "error", 3, "type", 1, 0},
        {"lazy failure", "H5 = cohomology(s1_scalar(1, 5))\nT5 = hecke(H5, 2)\nprint(T5.dims)", "error", 3, "runtime", 3, 2},
        {"failed handle re-accessed", "print(T5.dims)", "error", 3, "runtime", 1, 2},
        {"assert", "assert(false, \"served assertion\")", "error", 4, "assertion", 1, 0},
        {"expect_error that did not fail", "expect_error \"zzz\" { print(1) }", "error", 4, "assertion", 1, 0},
        {"not implemented", "set coefficient_ring = \"Z/25\"", "error", 5, "not_implemented", 1, 0},   // (slot-24 D6) Z is un-reserved; Z/<m> still is
        {"include", "include(\"prelude.oc\")", "rejected", 1, "forbidden", 1, 0},
        {"quit", "print(1)\nquit", "rejected", 1, "forbidden", 2, 0},
        {"exit", "exit(0)", "rejected", 1, "forbidden", 1, 0},
        {"unserved verb", "print(\"never\")\nrun_campaign(1)", "rejected", 1, "forbidden", 2, 0},
    };
    for (const Row& r : rows) {
        const Cell c = s.run(r.src);
        x.cells += 2;
        const Json* e = c.done_frame.get("error");
        const std::string msg = (e != nullptr && e->is_object()) ? S(e->get("message")) : "";
        Check(c.done && Status(c) == r.status && Exit(c) == r.exit && ErrKind(c) == r.kind,
              std::string("(D) ") + r.label + ": " + r.status + " / exit " + std::to_string(r.exit) + " / " + r.kind +
                  " (got " + Status(c) + " / " + std::to_string(Exit(c)) + " / " + ErrKind(c) + ": " + msg + ")");
        Check(msg.rfind("<cell>:" + std::to_string(r.line) + ":", 0) == 0,
              std::string("(D) ") + r.label + ": the message carries the <cell>:" + std::to_string(r.line) +
                  ":col prefix ('" + msg + "')");
        if (e != nullptr && e->is_object()) {
            const Json* at = e->get("at");
            Check(at != nullptr && I(at->get("line")) == r.line, std::string("(D) ") + r.label + ": error.at.line");
            if (r.created_line > 0) {
                const Json* ca = e->get("created_at");
                Check(ca != nullptr && I(ca->get("line")) == r.created_line,
                      std::string("(D) ") + r.label + ": error.created_at names the creation site");
            }
        }
        if (std::string(r.kind) == "forbidden" || std::string(r.kind) == "parse")
            Check(c.out.empty() && I(c.done_frame.get("statements")) == 0, std::string("(D) ") + r.label + ": nothing ran");
        Check(s.survives(), std::string("(D) the session survives ") + r.label);
    }
    // forbidden_overload: an untagged overload under a tagged name; the call does not run.
    {
        const Cell c = s.run("print(\"before\")\nsave_state(1, \"" + sentinel + "\")");
        x.cells += 2;
        Check(c.done && Status(c) == "error" && Exit(c) == 3 && ErrKind(c) == "forbidden_overload" && c.out == "before\n",
              "(D) save_state(x, path): error / forbidden_overload / exit 3 after the print ran (got " + Status(c) +
                  " / " + ErrKind(c) + " / out '" + c.out + "')");
        Check(!PathExists(sentinel), "(D) the refused save_state(x, path) wrote nothing");
        Check(s.survives(), "(D) the session survives forbidden_overload");
    }
    std::cout << "EFFECTIVE ConsoleServeValidation arm=D bad_alloc=skipped" << std::endl;
    s.c.close_in();
    Check(s.c.wait_exit(10000) == 0, "(D) EOF exits 0");
}

void ArmE(Ctx& x) {
    Session1 s(x);
    Check(s.ok, "(E) the child starts");
    if (!s.ok) return;
    // An interrupt with no running eval is ignored: no frame.
    s.c.send_line("{\"op\":\"interrupt\"}");
    const Cell q = s.run("print(7)");
    Check(q.done && q.frames.size() == 2 && q.out == "7\n", "(E) an interrupt with no running eval produces no frame");
    const auto t0 = std::chrono::steady_clock::now();
    s.c.send_line(EvalRequest("loop", "for i in range(0, 4000) { for j in range(0, 4000) { x = abs(j) } }"));
    std::this_thread::sleep_for(std::chrono::milliseconds(400));
    s.c.send_line("{\"op\":\"interrupt\"}");
    Cell c;
    Collect(s.c, "loop", c, 60000);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    x.cells += 3;
    Check(c.done && Status(c) == "interrupted" && Exit(c) == 3 && ErrKind(c) == "interrupted",
          "(E) the interrupted cell answers interrupted / exit 3 (got " + Status(c) + " / " + ErrKind(c) + ")");
    Check(secs < 30.0, "(E) the interrupt is acted on within one statement (" + std::to_string(secs) + " s)");
    const Cell after = s.run("print(x >= 0)");
    Check(after.done && Status(after) == "ok" && after.out == "true\n", "(E) the next eval sees the bindings of the first");
    Check(s.survives(), "(E) the session survives the interrupt");
    s.c.close_in();
    s.c.wait_exit(10000);
}

void ArmF(Ctx& x) {
    {
        Session1 s(x);
        Check(s.ok, "(F) the child starts");
        if (!s.ok) return;
        auto reject = [&](const std::string& line, const char* id_want, const std::string& label) {
            s.c.send_line(line);
            Json f;
            const bool got = s.c.frame(f, 30000);
            const Json* e = f.get("error");
            const Json* id = f.get("id");
            const bool id_ok = id_want == nullptr ? (id != nullptr && id->is_null())
                                                  : (id != nullptr && id->is_string() && id->text == id_want);
            Check(got && std::string(S(f.get("kind"))) == "done" && std::string(S(f.get("status"))) == "rejected" &&
                      I(f.get("exit")) == 1 && e != nullptr && std::string(S(e->get("kind"))) == "protocol" && id_ok,
                  "(F) " + label + " => done rejected / protocol / exit 1");
            ++x.cells;
        };
        reject("this is not json", nullptr, "malformed JSON");
        reject("{\"op\":\"frobnicate\",\"id\":\"f1\"}", "f1", "an unknown op");
        reject("{\"op\":\"eval\",\"src\":\"print(1)\"}", nullptr, "a missing id");
        reject("{\"op\":\"eval\",\"id\":\"f2\",\"src\":5}", "f2", "a non-string src");
        reject("[1,2,3]", nullptr, "a non-object request");
        Check(s.survives(), "(F) the session survives the protocol errors");
        // The 65th queued request is `busy`: a long cell first (given time to
        // leave the queue), then 65 more in one write.
        s.c.send_line(EvalRequest("long", "for i in range(0, 4000) { for j in range(0, 4000) { x = abs(j) } }"));
        std::this_thread::sleep_for(std::chrono::milliseconds(400));
        std::string burst;
        for (int k = 0; k < 65; ++k) burst += EvalRequest("b" + std::to_string(k), "print(" + std::to_string(k) + ")") + "\n";
        s.c.send(burst);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        s.c.send_line("{\"op\":\"interrupt\"}");
        std::size_t busy = 0, ok = 0;
        std::string busy_id;
        bool long_interrupted = false;
        for (int k = 0; k < 66; ++k) {
            Cell dummy;
            Json f;
            bool done = false;
            while (!done) {
                if (!s.c.frame(f, 60000)) break;
                if (std::string(S(f.get("kind"))) != "done") continue;
                done = true;
            }
            if (!done) break;
            const std::string id = S(f.get("id"));
            const Json* e = f.get("error");
            if (e != nullptr && e->is_object() && std::string(S(e->get("kind"))) == "busy") {
                ++busy;
                busy_id = id;
            } else if (id == "long") {
                long_interrupted = std::string(S(f.get("status"))) == "interrupted";
            } else if (std::string(S(f.get("status"))) == "ok") {
                ++ok;
            }
        }
        x.cells += 66;
        Check(busy == 1 && busy_id == "b64", "(F) the 65th queued request is rejected busy (" + std::to_string(busy) +
                                                 " busy, id '" + busy_id + "')");
        Check(long_interrupted && ok == 64, "(F) the 64 queued requests run after the interrupted one (" +
                                                std::to_string(ok) + " ok)");
        s.c.send_line("{\"op\":\"close\"}");
        Json bye;
        Check(s.c.frame(bye, 10000) && std::string(S(bye.get("kind"))) == "bye", "(F) close => bye");
        s.c.close_in();
        Check(s.c.wait_exit(10000) == 0, "(F) close => exit 0");
        Check(s.c.seq_ok && s.c.all_json && s.c.no_blank, "(F) the stream stayed well formed");
    }
    {
        Session1 s(x);
        s.c.close_in();
        const int code = s.c.wait_exit(10000);
        std::string rest;
        Check(code == 0 && !s.c.line(rest, 100), "(F) stdin EOF => exit 0, no bye");
    }
    {
        Session1 s(x);
        std::string big(8388608 + 64, 'x');
        s.c.send(big);
        Json f;
        bool fatal = false;
        while (s.c.frame(f, 30000))
            if (std::string(S(f.get("kind"))) == "fatal") fatal = std::string(S(f.get("reason"))) == "protocol_overflow";
        const int code = s.c.wait_exit(10000);
        Check(fatal && code == 70, "(F) a request line over max_request_bytes => fatal protocol_overflow, exit 70 (got " +
                                       std::to_string(code) + ")");
        x.cells += 1;
    }
}

void ArmG(Ctx& x) {
    struct Case {
        std::string kv;
        std::string name;
    };
    const std::string golden_out = x.scratch + "/golden_out.inc";
    const std::vector<Case> cases = {{"GOLDEN_EMIT=1", "GOLDEN_EMIT"},
                                     {"CONSOLE_EMIT_EXPECTED=1", "CONSOLE_EMIT_EXPECTED"},
                                     {"HECKE_NEBENTYPUS_NB3_FULL=1", "HECKE_NEBENTYPUS_NB3_FULL"},
                                     {"ENGINE_DEVICE=gpu", "ENGINE_DEVICE"},
                                     {"ASAN_OPTIONS=detect_leaks=0", "ASAN_OPTIONS"}};
    for (const Case& cs : cases) {
        std::vector<std::string> env = WithEnv(x.env, cs.kv);
        if (cs.name == "GOLDEN_EMIT") env = WithEnv(env, "GOLDEN_OUT=" + golden_out);
        Child c;
        c.start(x.exe, {"--serve"}, env, x.cwd);
        const int code = c.wait_exit(30000);
        const std::string want = "serve: refused: environment variable " +
                                 (cs.name == "GOLDEN_EMIT" ? std::string() : cs.name);
        Check(code == 1 && c.obuf.empty() &&
                  c.ebuf.find("serve: refused: environment variable ") != std::string::npos &&
                  c.ebuf.find(" is not permitted in a served session") != std::string::npos &&
                  (cs.name == "GOLDEN_EMIT" || c.ebuf.find(want) != std::string::npos),
              "(G) " + cs.kv + ": exit 1, empty stdout, the refusal on stderr (got " + std::to_string(code) + ", '" +
                  c.ebuf.substr(0, 160) + "')");
        ++x.cells;
    }
    Check(!PathExists(golden_out), "(G) the emitter's would-be output path does not exist");
    // (NBCONFIRM S4 fix round 1, audit L2) the host-probe fixture is honoured only as a child of this binary (as in (L));
    // launched by another program (here /bin/sh, which does not exec the command since another one follows) it is refused.
    {
        Child c;
        c.start("/bin/sh", {"-c", "\"$0\" --serve; exit $?", x.exe}, WithEnv(x.env, "HACS_HOST_PROBE_FIXTURE=1"), x.cwd);
        const int code = c.wait_exit(30000);
        Check(code == 1 && c.obuf.empty() &&
                  c.ebuf.find("serve: refused: environment variable HACS_HOST_PROBE_FIXTURE is not permitted in a served "
                              "session") != std::string::npos,
              "(G) HACS_HOST_PROBE_FIXTURE=1 under a non-validation parent: exit 1, the refusal on stderr (got " +
                  std::to_string(code) + ", '" + c.ebuf.substr(0, 160) + "')");
        ++x.cells;
    }
    // Whitelist only: starts, with the limits in force.
    {
        Session1 s(x);
        Check(s.ok, "(G) the whitelist environment starts a session");
        std::string limits, status;
        ReadFile("/proc/" + std::to_string(s.c.pid) + "/limits", limits);
        ReadFile("/proc/" + std::to_string(s.c.pid) + "/status", status);
        auto row = [&](const std::string& label) {
            for (const std::string& l : SplitLines(limits))
                if (l.rfind(label, 0) == 0) return l;
            return std::string();
        };
        auto zero = [](const std::string& l) {
            std::istringstream in(l.substr(26));
            std::string soft, hard;
            in >> soft >> hard;
            return soft == "0" && hard == "0";
        };
        Check(zero(row("Max file size")), "(G) RLIMIT_FSIZE = 0 in the child ('" + row("Max file size") + "')");
        Check(zero(row("Max core file size")), "(G) RLIMIT_CORE = 0 in the child ('" + row("Max core file size") + "')");
        Check(status.find("NoNewPrivs:\t1") != std::string::npos, "(G) PR_SET_NO_NEW_PRIVS is set in the child");
        Check(s.survives(), "(G) the hardened session evaluates");
        ++x.cells;
        s.c.close_in();
        s.c.wait_exit(10000);
    }
    // A regular-file stdout is refused.
    {
        const std::string f = x.scratch + "/stdout_file.txt";
        Child c;
        c.start(x.exe, {"--serve"}, x.env, x.cwd, f);
        const int code = c.wait_exit(30000);
        struct stat st;
        const bool empty = ::stat(f.c_str(), &st) == 0 && st.st_size == 0;
        Check(code == 1 && empty && c.ebuf.find("stdout must be a pipe") != std::string::npos,
              "(G) a regular-file stdout is refused (exit 1, nothing written)");
        ::unlink(f.c_str());
    }
    // Other CLI tokens with --serve are exit 1.
    {
        Child c;
        c.start(x.exe, {"--serve", "--script", "x.oc"}, x.env, x.cwd);
        Check(c.wait_exit(30000) == 1 && c.obuf.empty(), "(G) --serve with another CLI token is exit 1");
        Child d;
        d.start(x.exe, {"--serve", "--set", "threads=1"}, x.env, x.cwd);
        Check(d.wait_exit(30000) == 1 && d.obuf.empty(), "(G) --serve --set of a non-settable key is exit 1");
    }
}

void ArmH(Ctx& x) {
    std::int64_t base = 0;
    {
        Session1 s(x);
        const Cell c = s.run("print(1)");
        if (const Json* p = c.done_frame.get("perf")) base = I(p->get("rss_mib"));
        s.c.close_in();
        s.c.wait_exit(10000);
    }
    Check(base > 0, "(H) the baseline resident set is reported (" + std::to_string(base) + " MiB)");
    const std::int64_t cap = base + 256;
    Session1 s(x, {"--serve", "--rss-cap-mib", std::to_string(cap)});
    Check(s.ok, "(H) a capped session starts");
    if (!s.ok) return;
    s.c.send_line(EvalRequest("big", "Z = random_matrix(7, 9000, 9000, 1)\nprint(1)"));
    const auto t0 = std::chrono::steady_clock::now();
    bool fatal = false, bad_alloc = false;
    Json f;
    while (s.c.frame(f, 60000)) {
        const std::string kind = S(f.get("kind"));
        if (kind == "fatal") fatal = std::string(S(f.get("reason"))) == "rss_cap";
        if (kind == "done") {
            const Json* e = f.get("error");
            bad_alloc = I(f.get("exit")) == 3 && e != nullptr && e->is_object() &&
                        std::string(S(e->get("message"))).find("memory") != std::string::npos;
            break;
        }
        if (fatal) break;
    }
    const int code = fatal ? s.c.wait_exit(10000) : -2;
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    Check((fatal && code == 70) || bad_alloc,
          "(H) over the cap: a fatal rss_cap frame and exit 70, or an ordinary bad_alloc (fatal=" +
              std::to_string(fatal) + " exit=" + std::to_string(code) + " bad_alloc=" + std::to_string(bad_alloc) + ")");
    Check(secs < 60.0, "(H) the cap acts within the stated interval (" + std::to_string(secs) + " s)");
    ++x.cells;
    std::cout << "EFFECTIVE ConsoleServeValidation arm=H cap_mib=" << cap << " outcome="
              << (fatal ? "fatal_rss_cap" : bad_alloc ? "bad_alloc" : "none") << std::endl;
}

void ArmI(Ctx& x, std::size_t& log_frames) {
    Session1 s(x);
    if (!s.ok) {
        Check(false, "(I) the child starts");
        return;
    }
    const Cell c = s.run("G = golden_subset([[10, 691]])\nprint(G.mismatches)\nH = cohomology(symmetric_power(sl2z_model(), 12, 691))\nprint(H.dims)", 600000);
    x.cells += 1;
    Check(c.done && Status(c) == "ok", "(I) a pipeline cell runs (" + ErrMsg(c).substr(0, 200) + ")");
    for (const Json& f : c.frames)
        if (std::string(S(f.get("kind"))) == "out" && std::string(S(f.get("stream"))) == "log") ++log_frames;
    Check(s.c.all_json && s.c.no_blank, "(I) every stdout line is a frame (no chatter on the protocol stream)");
    const Cell v = s.run("set verbose = true");
    Check(v.done && Status(v) == "error" && Exit(v) == 3 && ErrKind(v) == "forbidden_overload",
          "(I) set verbose = true is forbidden_overload (got " + Status(v) + " / " + ErrKind(v) + ")");
    Check(s.survives(), "(I) the session survives");
    s.c.close_in();
    s.c.wait_exit(10000);
}

void ArmJ(Ctx& x) {
    Session1 s(x);
    if (!s.ok) {
        Check(false, "(J) the child starts");
        return;
    }
    auto request = [&](const std::string& line, const std::string& id, Json& reply) {
        s.c.send_line(line);
        Cell c;
        Collect(s.c, id, c, 30000);
        for (const Json& f : c.frames)
            if (std::string(S(f.get("kind"))) != "done") reply = f;
        ++x.cells;
        return c.done && Status(c) == "ok";
    };
    Json r;
    Check(request("{\"op\":\"complete\",\"id\":\"j1\",\"prefix\":\"hec\"}", "j1", r), "(J) complete answers");
    bool has_hecke = false;
    if (const Json* it = r.get("items"))
        for (const Json& i : it->items) has_hecke = has_hecke || (i.get("text") != nullptr && i.get("text")->text == "hecke");
    Check(std::string(S(r.get("kind"))) == "completions" && has_hecke, "(J) complete(\"hec\") contains hecke");
    Json r2;
    request("{\"op\":\"complete\",\"id\":\"j2\",\"prefix\":\"run_\"}", "j2", r2);
    Check(serve_json::WriteJson(r2).find("run_campaign") == std::string::npos &&
              serve_json::WriteJson(r2).find("run_validation") == std::string::npos,
          "(J) no untagged verb is completed");
    Json r3;
    Check(request("{\"op\":\"help\",\"id\":\"j3\",\"topic\":\"\"}", "j3", r3), "(J) help(\"\") answers");
    const std::string h = serve_json::WriteJson(r3);
    std::size_t served = 0;
    if (const Json* v = r3.get("verbs")) served = v->items.size();
    Check(std::string(S(r3.get("kind"))) == "help" && served == x.facts.census.read_only_verbs &&
              h.find("\"name\":\"hecke\"") != std::string::npos && h.find("run_campaign") == std::string::npos &&
              h.find("\"name\":\"exit\"") == std::string::npos,
          "(J) help(\"\") lists the served verbs only (" + std::to_string(served) + ")");
    Json r4;
    request("{\"op\":\"help\",\"id\":\"j4\",\"topic\":\"save_state\"}", "j4", r4);
    std::size_t ov = 0;
    if (const Json* v = r4.get("verbs"))
        if (!v->items.empty() && v->items[0].get("overloads") != nullptr) ov = v->items[0].get("overloads")->items.size();
    Check(ov == 1, "(J) help(\"save_state\") shows the one served overload");
    const Cell vb = s.run("print(verbs())");
    Check(vb.out.find("run_campaign") == std::string::npos && vb.out.find("run_validation") == std::string::npos &&
              vb.out.find("hecke") != std::string::npos,
          "(J) verbs() shows the served view");
    s.run("Q = matrix(5, [[1, 0], [0, 1]])");
    Json r5;
    Check(request("{\"op\":\"describe\",\"id\":\"j5\",\"name\":\"Q\"}", "j5", r5) &&
              std::string(S(r5.get("kind"))) == "described" && r5.get("value") != nullptr &&
              std::string(S(r5.get("value")->get("t"))) == "Matrix",
          "(J) describe answers with the envelope");
    Json r6;
    request("{\"op\":\"complete\",\"id\":\"j6\",\"prefix\":\"Q.ro\"}", "j6", r6);
    Check(serve_json::WriteJson(r6).find("Q.rows") != std::string::npos, "(J) member completion");
    s.c.close_in();
    s.c.wait_exit(10000);
}

// (K) display(x, ...) (slot-24 NB-RENDER R3, [P] Sec. 18.4): in a served cell it forces a declared
// handle and sends, per argument, the canonical echo and a value frame at the call's position --
// exactly what the bare expression statement sends; outside a served session it prints show's text.
void ArmK(Ctx& x) {
    {
        // Outside --serve (an in-process session, no sink): one PrintFull line per argument.
        const hecke::ScopedMathSettings restore(hecke::math_settings());
        Registry reg;
        Session ses(reg, /*verbose_default=*/false);
        RegisterPipelineVerbs(reg);
        hecke_ext::RegisterHeckeExtensionVerbs(reg);
        reg.install_settings(ses.settings());
        std::ostringstream out;
        ses.set_output(out);
        const int code = RunSource(ses, "A = matrix(7, [[1, 2], [3, 4]]);\nH = cohomology(symmetric_power(sl2z_model(), 4, 7));\n"
                                        "display(A, H);\nshow(A);\nshow(H);\n", "<display>");
        const std::string o = out.str();
        const std::size_t half = o.size() / 2;
        Check(code == 0 && o.size() % 2 == 0 && o.substr(0, half) == o.substr(half),
              "(K) outside --serve, display(A, H) prints what show(A); show(H) prints, H forced first: " + o.substr(0, 160));
        Check(o.find("declared") == std::string::npos, "(K) display forced the declared Cohomology");
    }
    Session1 s(x);
    if (!s.ok) {
        Check(false, "(K) the child starts");
        return;
    }
    const Cell a = s.run("H = cohomology(symmetric_power(sl2z_model(), 10, 691))\nH");
    const Cell d = s.run("display(H, H.dims, 7)");
    const Cell b = s.run("H.dims");
    x.cells += 3;
    Check(a.done && Status(a) == "ok" && a.values.size() == 1 &&
              std::string(S(a.values[0].second.get("value")->get("state"))) == "declared",
          "(K) the bare echo of a fresh handle is declared");
    Check(d.done && Status(d) == "ok" && d.values.size() == 3, "(K) display(H, H.dims, 7) sends three value frames (" +
                                                                    std::to_string(d.values.size()) + ")");
    if (d.values.size() == 3) {
        const Json* v0 = d.values[0].second.get("value");
        Check(std::string(S(v0->get("t"))) == "Cohomology" && std::string(S(v0->get("state"))) == "computed",
              "(K) the declared handle was forced before its frame");
        Check(std::string(S(d.values[2].second.get("value")->get("canonical"))) == "7", "(K) a builtin argument is a frame too");
        bool pos_ok = true;
        for (const auto& fv : d.values)
            pos_ok = pos_ok && I(fv.second.get("stmt")) == 0 && I(fv.second.get("line")) == 1 && I(fv.second.get("col")) == 1;
        Check(pos_ok, "(K) every display frame carries the statement index and the call's line / col");
        std::string echo;
        for (const auto& fv : d.values) echo += std::string(S(fv.second.get("value")->get("canonical"))) + "\n";
        Check(d.out == echo, "(K) the stdout text is the canonical echo of each argument, as a bare expression prints it");
        Check(b.values.size() == 1 && serve_json::WriteJson(*b.values[0].second.get("value")) ==
                                          serve_json::WriteJson(*d.values[1].second.get("value")),
              "(K) display(H.dims) and the bare statement H.dims send the same envelope");
    }
    Check(s.survives(), "(K) the session survives");
    s.c.close_in();
    s.c.wait_exit(10000);
}

// (L) NBCONFIRM S4 ([P] Sec. 18.4 / 18.6 / 18.8 / 18.10): the per-cell cost confirmation of a served session.
// The child runs with HACS_DEVICE_PROFILE=none and HACS_HOST_PROBE_FIXTURE=1 (MemAvailable 4096 MiB, C: 255 GB), so the
// decisions do not depend on this host's memory, disk or device profile.  Cells: descent(1, 62) (just over n <= 60, ~1 s,
// < 1 GiB; normal under confirm_wall_s = 0), hida_tower(240, 5, 1) (~13 s at 32 threads, < 0.7 GiB: the 10 s heartbeat),
// level_n(30, 100, 7) (strong; never run), descent(1000000, 0) (hard refusal; never run).
bool IsToken(const std::string& t) {
    if (t.size() != 18 || t.compare(0, 2, "0x") != 0) return false;
    for (std::size_t k = 2; k < t.size(); ++k)
        if (!std::isxdigit(static_cast<unsigned char>(t[k])) || std::isupper(static_cast<unsigned char>(t[k]))) return false;
    return true;
}
std::string Token(const Cell& c) {
    const Json* f = c.done_frame.get("confirm");
    return (f != nullptr && f->is_object()) ? S(f->get("token")) : "";
}
double Num(const Json* j) { return (j != nullptr && j->is_number()) ? std::strtod(j->text.c_str(), nullptr) : -1.0; }
std::string ConfirmReq(const std::string& id, const std::string& src, const std::vector<std::string>& toks) {
    std::string r = "{\"op\":\"eval\",\"id\":" + Quote(id) + ",\"src\":" + Quote(src) + ",\"confirm\":[";
    for (std::size_t k = 0; k < toks.size(); ++k) r += (k ? "," : "") + Quote(toks[k]);
    return r + "]}";
}

void ArmL(Ctx& x, std::size_t& heartbeats) {
    Ctx y = x;
    y.env = WithEnv(WithEnv(x.env, "HACS_DEVICE_PROFILE=none"), "HACS_HOST_PROBE_FIXTURE=1");
    const std::string P = "set coefficient_ring = \"Z\"; set confirm_wall_s = 0; x = 7; r = descent(1, 62); print(r.status)";
    std::string T;
    double rss_hi = -1.0;
    {
        Session1 s(y, {"--serve"});
        Check(s.ok, "(L) the child starts");
        if (!s.ok) return;
        int k = 0;
        auto send = [&](const std::string& src, const std::vector<std::string>& toks) {
            Cell c;
            const std::string id = "l" + std::to_string(k++);
            s.c.send_line(ConfirmReq(id, src, toks));
            Collect(s.c, id, c, 300000);
            ++x.cells;
            return c;
        };
        // inside the envelope: unchanged, silent
        const Cell in = s.run("set coefficient_ring = \"Z\"; r0 = descent(1, 10); print(r0.status)");
        bool any_conf = false;
        for (const Json& f : in.frames) any_conf = any_conf || f.get("confirmed") != nullptr;
        Check(Status(in) == "ok" && in.out == "ok\n" && in.progress.empty() && !any_conf && in.done_frame.get("confirm") == nullptr,
              "(L) inside the silent envelope a call runs as before (no estimate, no frame)");
        // outside, normal: confirm_required
        const Cell a = s.run(P);
        T = Token(a);
        const Json* cf = a.done_frame.get("confirm");
        Check(Status(a) == "confirm_required" && Exit(a) == 3 && ErrKind(a) == "confirm_required" && IsToken(T),
              "(L) outside the envelope: done{status:confirm_required, exit 3, kind confirm_required, token 0x<16 hex>} (got " +
                  Status(a) + " / " + ErrKind(a) + " / '" + T + "')");
        if (cf != nullptr) {
            rss_hi = Num(cf->get("rss_hi"));
            const Json* site = cf->get("site");
            Check(std::string(S(cf->get("level"))) == "normal" && std::string(S(cf->get("verb"))) == "descent" &&
                      std::string(S(cf->get("call"))) == "descent(1, 62)" && Num(cf->get("wall_hi")) > 0 && rss_hi > 0 &&
                      std::string(S(cf->get("model"))).rfind("descent_cost/", 0) == 0 &&
                      std::string(S(cf->get("calibration"))) == "uncalibrated" && site != nullptr && I(site->get("line")) == 1 &&
                      I(site->get("col")) == 64,
                  "(L) confirm = {level normal, verb descent, call, site 1:64, wall_hi, rss_hi, model, calibration}");
        }
        Check(I(a.done_frame.get("statements")) == 3, "(L) the statements before the confirmation point ran (3)");
        const Cell xv = s.run("print(x)");
        Check(xv.out == "7\n", "(L) their bindings persist (x = 7)");
        const Cell ru = s.run("print(r)");
        Check(Status(ru) == "error", "(L) the stopped call bound nothing (r unknown)");
        // the same cell, the same token: deterministic
        const Cell a2 = s.run(P);
        Check(Token(a2) == T, "(L) the same cell in the same state offers the same token");
        // a forged token confirms nothing
        const Cell f1 = send(P, {"0x0000000000000000"});
        Check(Status(f1) == "confirm_required" && Token(f1) == T, "(L) a forged token is not accepted");
        // an edited cell
        const std::string P2 = "set coefficient_ring = \"Z\"; set confirm_wall_s = 0; x = 7;  r = descent(1, 62); print(r.status)";
        const Cell e1 = send(P2, {T});
        Check(Status(e1) == "confirm_required" && Token(e1) != T && IsToken(Token(e1)),
              "(L) a token never confirms an edited cell (a new token is offered)");
        // another argument value at the same site (a prior binding read by the call)
        const std::string PV = "set coefficient_ring = \"Z\"; set confirm_wall_s = 0; r = descent(1, nn); print(r.status)";
        s.run("nn = 62");
        const Cell v1 = s.run(PV);
        const std::string TV = Token(v1);
        s.run("nn = 64");
        const Cell v2 = send(PV, {TV});
        Check(Status(v2) == "confirm_required" && Token(v2) != TV,
              "(L) a token never confirms another argument value at the same call site");
        // the token confirms the call: the heartbeat's start frames (constructor + lazy work), the result
        const Cell ok = send(P, {T});
        std::size_t conf_frames = 0;
        bool wall_ok = true;
        for (const Json& f : ok.frames)
            if (std::string(S(f.get("kind"))) == "progress" && f.get("confirmed") != nullptr) {
                ++conf_frames;
                wall_ok = wall_ok && Num(f.get("confirmed")->get("wall_hi")) > 0;
            }
        Check(Status(ok) == "ok" && ok.out == "ok\n" && conf_frames >= 1 && wall_ok,
              "(L) with its token the cell runs: status ok, 'ok', >= 1 progress frame carrying confirmed.wall_hi (" +
                  std::to_string(conf_frames) + "; " + ErrMsg(ok).substr(0, 200) + ")");
        // a changed setting
        s.run("set print_max_entries = 77");
        const Cell s1 = send(P, {T});
        Check(Status(s1) == "confirm_required" && Token(s1) != T, "(L) a changed setting invalidates the token");
        // the read-only guarantee is unchanged: a token lets no unserved verb, overload or setting through
        const Cell ro1 = send("run_validation(\"X\")", {T});
        Check(Status(ro1) == "rejected" && ErrKind(ro1) == "forbidden", "(L) a token does not serve an unserved verb");
        const Cell ro2 = send("set confirm = \"yes\"", {T});
        Check(Status(ro2) == "error" && ErrKind(ro2) == "forbidden_overload", "(L) `confirm` is not settable in a served session");
        // protocol: the confirm list
        for (const std::string& bad : {std::string("\"0x1\""), std::string("[1]"), std::string("[\"\"]"),
                                       "[\"" + std::string(65, 'a') + "\"]", std::string("[\"a b\"]")}) {
            Cell c;
            const std::string id = "lp" + std::to_string(k++);
            s.c.send_line("{\"op\":\"eval\",\"id\":" + Quote(id) + ",\"src\":\"print(1)\",\"confirm\":" + bad + "}");
            Collect(s.c, id, c, 30000);
            Check(Status(c) == "rejected" && Exit(c) == 1 && ErrKind(c) == "protocol" && c.out.empty(),
                  "(L) a malformed confirm list is a protocol rejection and runs nothing: " + bad.substr(0, 20));
        }
        {
            std::vector<std::string> many(17, T);
            const Cell c = send("print(1)", many);
            Check(Status(c) == "rejected" && ErrKind(c) == "protocol", "(L) at most 16 tokens");
        }
        // strong: level_n(30, 100, 7) (extrapolated x > 2, wall_hi > 1 h) -- never run (over F_ell: the Gallery verbs refuse Z)
        s.run("set coefficient_ring = \"F_ell\"");
        const Cell st = s.run("level_n(30, 100, 7)");
        const Json* sc = st.done_frame.get("confirm");
        Check(Status(st) == "confirm_required" && sc != nullptr && std::string(S(sc->get("level"))) == "strong" &&
                  sc->get("reasons") != nullptr && !sc->get("reasons")->items.empty(),
              "(L) a strong warning is confirm_required with level strong and its reasons (got " + Status(st) + ": " +
                  ErrMsg(st).substr(0, 160) + ")");
        // a hard refusal has no token, in every mode
        const Cell hr = s.run("set coefficient_ring = \"Z\"; descent(1000000, 0)");
        Check(Status(hr) == "error" && Exit(hr) == 3 && ErrKind(hr) == "runtime" && hr.done_frame.get("confirm") == nullptr &&
                  ErrMsg(hr).find("refused before the run") != std::string::npos,
              "(L) a hard refusal (rule 19) is an error without a token: " + ErrMsg(hr).substr(0, 160));
        // the heartbeat of a confirmed run of > 10 s
        const std::string H = "set coefficient_ring = \"F_ell\"; set confirm_wall_s = 0; h = hida_tower(240, 5, 1); print(h.n)";
        const Cell h1 = s.run(H);
        const Cell h2 = send(H, {Token(h1)});
        std::size_t ticks = 0, starts = 0;
        for (const Json& f : h2.frames) {
            if (std::string(S(f.get("kind"))) != "progress" || f.get("confirmed") == nullptr) continue;
            if (Num(f.get("confirmed")->get("elapsed_s")) >= 9.0) ++ticks;
            else ++starts;
        }
        heartbeats = ticks;
        const double el = Num(h2.done_frame.get("perf") ? h2.done_frame.get("perf")->get("elapsed_ms") : nullptr) / 1000.0;
        Check(Status(h2) == "ok" && h2.out == "240\n" && starts == 1 && (el < 10.5 || ticks >= 1),
              "(L) a confirmed run shows one start frame and a heartbeat every 10 s (elapsed " + std::to_string(el) + " s, " +
                  std::to_string(ticks) + " heartbeat(s))");
        Check(s.survives(), "(L) the session survives");
        s.c.close_in();
        s.c.wait_exit(10000);
    }
    // a token is bound to its session: a restarted session does not accept it
    {
        Session1 s(y, {"--serve"});
        Cell c;
        s.c.send_line(ConfirmReq("n1", P, {T}));
        Collect(s.c, "n1", c, 60000);
        ++x.cells;
        Check(Status(c) == "confirm_required" && Token(c) != T, "(L) a token of another session is stale");
        s.c.close_in();
        s.c.wait_exit(10000);
    }
    // --set confirm_wall_s on the command line (now serve-settable) and --rss-cap-mib as R_cap of rule 1
    {
        Session1 s(y, {"--serve", "--set", "confirm_wall_s=0", "--rss-cap-mib", "300"});
        Check(s.ok, "(L) --serve --set confirm_wall_s=0 --rss-cap-mib 300 starts");
        if (s.ok) {
            const Cell c = s.run("set coefficient_ring = \"Z\"; descent(1, 62)");
            Check(rss_hi > 300 && Status(c) == "error" && Exit(c) == 3 && c.done_frame.get("confirm") == nullptr &&
                      ErrMsg(c).find("min(session RSS cap") != std::string::npos,
                  "(L) the session's --rss-cap-mib is R_cap: an estimate above it is refused before the run (rss_hi " +
                      std::to_string(rss_hi) + " MiB > 300): " + ErrMsg(c).substr(0, 160));
            Check(s.survives(), "(L) the capped session survives");
        }
        s.c.close_in();
        s.c.wait_exit(10000);
        ++x.cells;
    }
    std::cout << "EFFECTIVE ConsoleServeValidation arm=L tokens=ok heartbeats=" << heartbeats << " rss_hi=" << rss_hi << std::endl;
}

// (C2) the supplementary coverage cell.
void ArmC2(Ctx& x, std::size_t& covered, std::size_t& total) {
    std::string src;
    const std::string rel = "src/hacs/Console/serve/coverage.oc";
    if (!ReadFile(x.root + "/" + rel, src)) {
        Check(false, "(C2) " + rel + " is readable");
        return;
    }
    ScriptRun r;
    RunOne(x.exe, x.env, {"--serve"}, x.cwd, "coverage.oc", src, r, 600000);
    x.cells += 1;
    Check(r.cell.done && Status(r.cell) == "ok", "(C2) the coverage cell runs ok (" + ErrMsg(r.cell).substr(0, 300) + ")");
    if (r.coverage.size() == x.covered.size())
        for (std::size_t i = 0; i < r.coverage.size(); ++i) x.covered[i] = x.covered[i] || r.coverage[i] == '1';
    // Every tagged overload was called at least once.
    std::size_t k = 0;
    total = 0;
    covered = 0;
    Registry reg;
    Session session(reg, false);
    RegisterPipelineVerbs(reg);
    hecke_ext::RegisterHeckeExtensionVerbs(reg);
    for (const std::string& name : reg.verb_names()) {
        const VerbEntry* v = reg.find_verb(name);
        for (const Overload& o : v->overloads) {
            if (serve_tags::IsReadOnly(*v, o)) {
                ++total;
                if (k < x.covered.size() && x.covered[k]) ++covered;
                else Check(false, "(C2) the tagged overload " + name + " | " + o.sig.to_string(name) +
                                      " was never called in the conformance run");
            }
            ++k;
        }
    }
}

} // namespace
#endif

uint32_t ConsoleServeValidation() {
    if (!oc::kBuildNotebook) return oc::BuildSwitchSkip("ConsoleServeValidation", "OC_BUILD_NOTEBOOK");
#if OC_BUILD_NOTEBOOK
    std::cout << "=== ConsoleServeValidation ([P] v3.10 Sec. 18.11, oc-serve/1; slot-19 NB-2) ===" << std::endl;
    g_checks = 0;
    g_fail = 0;
    Ctx x;
    x.exe = SelfExe();
    x.root = FindRepoRoot();
    if (x.exe.empty() || x.root.empty()) {
        std::cout << "  [SKIP] ConsoleServeValidation: the executable or the source tree was not found" << std::endl;
        return 0;
    }
    x.scratch = MakeScratch();
    if (x.scratch.empty()) {
        Check(false, "a scratch directory can be made (${XDG_CACHE_HOME:-$HOME/.cache}/hacs/scratch or beside the binary)");
        return g_fail;
    }
    x.cwd = x.scratch + "/cwd";
    ::mkdir(x.cwd.c_str(), 0755);
    x.env = BaseEnv();
    x.facts = Facts();
    x.covered.assign(x.facts.is_tagged.size(), false);

    void (*old_pipe)(int) = std::signal(SIGPIPE, SIG_IGN);
    const auto t0 = std::chrono::steady_clock::now();

    // git status of the source tree, before (probed, not required).
    auto git_status = [&](std::string& out) {
        const std::string cmd = "git -C '" + x.root + "' status --porcelain 2>/dev/null";
        FILE* p = ::popen(cmd.c_str(), "r");
        if (p == nullptr) return false;
        char buf[4096];
        while (std::fgets(buf, sizeof(buf), p) != nullptr) out += buf;
        return ::pclose(p) == 0;
    };
    std::string git_before, git_after;
    const bool git = PathExists(x.root + "/.git") && git_status(git_before);
    Check(DirEntries(x.cwd) == 0, "(C2) the working directory is empty before the run");

    std::size_t n_scripts = 0, n_exempt = 0, n_values = 0, log_frames = 0, cov = 0, cov_total = 0;
    std::string digests;
    ArmA(x);
    ArmBC(x, n_scripts, n_exempt, n_values, digests);
    ArmCHash(x);
    ArmC2(x, cov, cov_total);
    ArmD(x);
    ArmE(x);
    ArmF(x);
    ArmG(x);
    ArmH(x);
    ArmI(x, log_frames);
    ArmJ(x);
    ArmK(x);
    std::size_t heartbeats = 0;
    ArmL(x, heartbeats);

    Check(DirEntries(x.cwd) == 0, "(C2) the working directory is empty after the run");
    if (git) {
        git_status(git_after);
        Check(git_after == git_before, "(C2) git status --porcelain of the source tree is unchanged");
    }
    std::signal(SIGPIPE, old_pipe);
    RemoveTree(x.scratch);

    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "EFFECTIVE ConsoleServeValidation arm=ALL cells=" << x.cells << " scripts=" << n_scripts
              << " exempt=" << n_exempt << " values=" << n_values << " coverage=" << cov << "/" << cov_total
              << " log_frames=" << log_frames << " bad_alloc=skipped git=" << (git ? "present" : "absent") << std::endl;
    std::cout << "SERVE_DETERMINISM " << digests << std::endl;
    std::cout << "ConsoleServeValidation: " << g_checks << " checks, " << g_fail << " failing, " << secs << " s"
              << std::endl;
    std::cout << "ConsoleServeValidation: " << (g_fail == 0 ? "PASSED" : "FAILED") << std::endl;
    return g_fail;
#else
    return 0;
#endif
}
