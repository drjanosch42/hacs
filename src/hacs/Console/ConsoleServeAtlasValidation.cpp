// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// ConsoleServeAtlasValidation.cpp -- see ConsoleServeAtlasValidation.h.
// ===========================================================================

#include "ConsoleServeAtlasValidation.h"

#include "../BuildSwitches.h"

#if OC_BUILD_NOTEBOOK
#include "Bindings/ServeJson.h"

#include "../HeckeExt/HeckeAtlas.h"
#include "../Util/AtlasIndex.h"
#include "../../ModArithmetic/Fnv.h"
#include "../Util/TextIo.h"

#include <algorithm>
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
#endif

#if OC_BUILD_NOTEBOOK
namespace {

using console::serve_json::Json;

uint32_t g_checks = 0;
uint32_t g_fail = 0;
uint32_t g_queries = 0;

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
using oc_util::ReadFile;   // S24 CL-7 (F11): Util/TextIo.h

bool WriteFile(const std::string& path, const std::string& text) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) return false;
    out << text;
    return static_cast<bool>(out);
}

bool PathExists(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0;
}

struct Stamp {
    bool ok = false;
    long long size = 0, mtime_s = 0, mtime_ns = 0;
    std::uint64_t fnv = 0;
};

Stamp StampOf(const std::string& path) {
    Stamp s;
    struct stat st;
    if (::stat(path.c_str(), &st) != 0) return s;
    std::string text;
    if (!ReadFile(path, text)) return s;
    s.ok = true;
    s.size = static_cast<long long>(st.st_size);
    s.mtime_s = static_cast<long long>(st.st_mtim.tv_sec);
    s.mtime_ns = static_cast<long long>(st.st_mtim.tv_nsec);
    s.fnv = engine::FnvBytes(engine::kFnvOffset64, text);
    return s;
}

bool SameStamp(const Stamp& a, const Stamp& b) {
    return a.ok && b.ok && a.size == b.size && a.mtime_s == b.mtime_s && a.mtime_ns == b.mtime_ns && a.fnv == b.fnv;
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

std::string RealPath(const std::string& p) {
    char buf[PATH_MAX];
    if (::realpath(p.c_str(), buf) == nullptr) return std::string();
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
        std::string tmpl = p + "/serve_atlas_validation.XXXXXX";
        std::vector<char> b(tmpl.begin(), tmpl.end());
        b.push_back('\0');
        if (::mkdtemp(b.data()) != nullptr) return RealPath(std::string(b.data()));
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

bool Contains(const std::string& hay, const std::string& needle) { return hay.find(needle) != std::string::npos; }

// ---------------------------------------------------------------------------
// The child process (the harness of ConsoleServeValidation, narrowed).
// ---------------------------------------------------------------------------
std::vector<std::string> BaseEnv() {
    std::vector<std::string> env;
    for (const char* k : {"PATH", "HOME", "LANG", "LC_ALL", "TMPDIR"}) {
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
    std::int64_t next_seq = 0;
    bool seq_ok = true;
    bool all_json = true;
    bool no_blank = true;
    std::size_t max_frame = 0;

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
                return true;
            }
            if (out_eof) return false;
            const auto left =
                std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count();
            if (left <= 0) return false;
            pump(static_cast<int>(std::min<long long>(left, 200)));
        }
    }

    bool frame(Json& j, int timeout_ms, std::string* raw = nullptr) {
        std::string l;
        if (!line(l, timeout_ms)) return false;
        if (raw != nullptr) *raw = l;
        if (l.empty()) no_blank = false;
        max_frame = std::max(max_frame, l.size());
        std::string e;
        if (!console::serve_json::ParseJson(l, j, e) || !j.is_object()) {
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
    std::int64_t v = -1;
    if (j == nullptr || !j->is_number() || !j->as_int64(v)) return -1;
    return v;
}
bool B(const Json* j, bool want) { return j != nullptr && j->is_bool() && j->b == want; }

std::string Hex(std::uint64_t v) { return atlas_index::Hex16(v); }

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------
struct StoreFx {
    std::string dir, path, text, name;
    hecke_ext::AtlasStore oracle;
    atlas_index::Index ix;          // the in-process scan of `text`
    std::string index_text;         // Serialize(ix)
    Stamp stamp;
};

struct Ctx {
    std::string exe, root, scratch, cwd;
    std::vector<std::string> env;
    StoreFx s23, s13, mem;         // mem: a copy of s23 with no sidecar
    std::size_t cells = 0, systems = 0;
};

// Runs `exe args` to its end (stdin closed).  Returns the exit code.
int RunToEnd(Ctx& x, const std::vector<std::string>& args, std::string& out, std::string& err,
             const std::vector<std::string>* env = nullptr, int timeout_ms = 120000) {
    Child c;
    if (!c.start(x.exe, args, env != nullptr ? *env : x.env, x.cwd)) return -2;
    c.close_in();
    const int code = c.wait_exit(timeout_ms);
    out = c.obuf;
    err = c.ebuf;
    return code;
}

// A served child, after its hello.
bool StartServed(Ctx& x, Child& c, const std::vector<std::string>& args, Json& hello,
                 const std::vector<std::string>* env = nullptr, int timeout_ms = 60000) {
    std::vector<std::string> a = {"--serve-atlas"};
    a.insert(a.end(), args.begin(), args.end());
    if (!c.start(x.exe, a, env != nullptr ? *env : x.env, x.cwd)) return false;
    if (!c.frame(hello, timeout_ms)) return false;
    return std::string(S(hello.get("kind"))) == "atlas_hello";
}

// One request, its one reply.
bool Ask(Child& c, const std::string& req, Json& rep, std::string* raw = nullptr, int timeout_ms = 60000) {
    ++g_queries;
    if (!c.send_line(req)) return false;
    if (!c.frame(rep, timeout_ms, raw)) return false;
    return std::string(S(rep.get("kind"))) == "atlas_reply";
}

std::string Status(const Json& r) { return S(r.get("status")); }
std::string ErrKind(const Json& r) {
    const Json* e = r.get("error");
    return (e != nullptr && e->is_object()) ? std::string(S(e->get("kind"))) : std::string();
}
const Json* Result(const Json& r) { return r.get("result"); }

// The reply stream without its volatile `perf`.
std::string Normal(const std::string& raw) {
    Json j;
    std::string e;
    if (!console::serve_json::ParseJson(raw, j, e) || !j.is_object()) return raw;
    j.members.erase(std::remove_if(j.members.begin(), j.members.end(),
                                   [](const std::pair<std::string, Json>& m) { return m.first == "perf"; }),
                    j.members.end());
    return console::serve_json::WriteJson(j);
}

// ---------------------------------------------------------------------------
// The oracle side
// ---------------------------------------------------------------------------
bool U32List(const Json* j, const std::vector<uint32_t>& want) {
    if (j == nullptr || !j->is_array() || j->items.size() != want.size()) return false;
    for (std::size_t i = 0; i < want.size(); ++i)
        if (I(&j->items[i]) != static_cast<std::int64_t>(want[i])) return false;
    return true;
}
bool U64List(const Json* j, const std::vector<uint64_t>& want) {
    if (j == nullptr || !j->is_array() || j->items.size() != want.size()) return false;
    for (std::size_t i = 0; i < want.size(); ++i)
        if (I(&j->items[i]) != static_cast<std::int64_t>(want[i])) return false;
    return true;
}

bool SystemMatches(const Json* s, const hecke_ext::AtlasSystem& o, std::string& why) {
    auto bad = [&](const std::string& w) {
        why = w;
        return false;
    };
    if (s == nullptr || !s->is_object()) return bad("not an object");
    if (I(s->get("id")) != o.id) return bad("id");
    if (I(s->get("r")) != o.r) return bad("r");
    const Json* ev = s->get("eigenvalues");
    if (ev == nullptr || !ev->is_array() || ev->items.size() != o.eigenvalues.size()) return bad("eigenvalues length");
    for (std::size_t k = 0; k < o.eigenvalues.size(); ++k)
        if (!U32List(&ev->items[k], o.eigenvalues[k])) return bad("eigenvalue " + std::to_string(k));
    if (std::string(S(s->get("type"))) != o.type) return bad("type");
    const Json* chi = s->get("chi");
    if (o.has_chi) {
        if (!U32List(chi, {o.chi_a, o.chi_b})) return bad("chi");
    } else if (chi == nullptr || !chi->is_null()) {
        return bad("chi (null)");
    }
    if (I(s->get("base")) != o.base || I(s->get("twist")) != o.twist) return bad("base / twist");
    if (!B(s->get("collected"), o.collected) || !B(s->get("in_H1"), o.in_H1)) return bad("collected / in_H1");
    const Json* oc = s->get("occurrences");
    if (oc == nullptr || !oc->is_array() || oc->items.size() != o.occurrences.size()) return bad("occurrences length");
    for (std::size_t k = 0; k < o.occurrences.size(); ++k) {
        const Json& a = oc->items[k];
        const hecke_ext::AtlasOccurrence& w = o.occurrences[k];
        if (I(a.get("n")) != w.n || I(a.get("ideal")) != w.ideal || std::string(S(a.get("group"))) != w.group ||
            I(a.get("dim")) != w.dim)
            return bad("occurrence " + std::to_string(k));
        const Json* ae = a.get("a_ell");
        if (ae == nullptr || std::string(S(ae->get("status"))) != (w.a_ell.known ? "known" : "unknown") ||
            std::string(S(ae->get("reason"))) != w.a_ell.reason)
            return bad("occurrence a_ell " + std::to_string(k));
        if (w.a_ell.known && !U32List(ae->get("value"), w.a_ell.value)) return bad("occurrence a_ell value");
    }
    return true;
}

struct Filter {
    std::string json;   // the request's filter object
    bool has_r = false, has_type = false, has_coll = false, has_base = false, has_twist = false, has_n = false;
    uint32_t r = 0, base = 0, twist = 0, n = 0;
    std::string type;
    bool coll = false;
};

std::vector<uint32_t> OracleFilter(const hecke_ext::AtlasStore& st, const Filter& f) {
    std::vector<uint32_t> ids;
    for (const hecke_ext::AtlasSystem& s : st.systems) {
        if (f.has_r && s.r != f.r) continue;
        if (f.has_type && s.type != f.type) continue;
        if (f.has_coll && s.collected != f.coll) continue;
        if (f.has_base && s.base != f.base) continue;
        if (f.has_twist && s.twist != f.twist) continue;
        if (f.has_n) {
            bool at = false;
            for (const hecke_ext::AtlasOccurrence& o : s.occurrences) at = at || o.n == f.n;
            if (!at) continue;
        }
        ids.push_back(s.id);
    }
    return ids;
}

std::vector<Filter> Filters(const hecke_ext::AtlasStore& st) {
    std::vector<Filter> out;
    auto add = [&](Filter f, const std::string& json) {
        f.json = json;
        out.push_back(f);
    };
    add(Filter(), "{}");
    std::set<uint32_t> rs;
    for (const auto& s : st.systems) rs.insert(s.r);
    for (uint32_t r : rs) {
        Filter f;
        f.has_r = true;
        f.r = r;
        add(f, "{\"r\":" + std::to_string(r) + "}");
    }
    for (const char* t : {"eisenstein", "cuspidal"}) {
        Filter f;
        f.has_type = true;
        f.type = t;
        add(f, std::string("{\"type\":\"") + t + "\"}");
    }
    for (bool c : {true, false}) {
        Filter f;
        f.has_coll = true;
        f.coll = c;
        add(f, std::string("{\"collected\":") + (c ? "true" : "false") + "}");
    }
    {
        Filter f;
        f.has_base = true;
        f.base = st.systems.empty() ? 0 : st.systems.back().base;
        add(f, "{\"base\":" + std::to_string(f.base) + "}");
    }
    for (uint32_t t : {0u, 1u, 7u}) {
        Filter f;
        f.has_twist = true;
        f.twist = t;
        add(f, "{\"twist\":" + std::to_string(t) + "}");
    }
    for (const auto& c : st.cells) {
        Filter f;
        f.has_n = true;
        f.n = c.n;
        add(f, "{\"n\":" + std::to_string(c.n) + "}");
    }
    {
        Filter f;
        f.has_type = true;
        f.type = "cuspidal";
        f.has_r = true;
        f.r = 2;
        add(f, "{\"type\":\"cuspidal\",\"r\":2}");
    }
    {
        Filter f;
        f.has_coll = true;
        f.coll = false;
        f.has_type = true;
        f.type = "eisenstein";
        f.has_twist = true;
        f.twist = 3;
        add(f, "{\"collected\":false,\"type\":\"eisenstein\",\"twist\":3}");
    }
    {
        Filter f;   // matches nothing
        f.has_r = true;
        f.r = 99;
        add(f, "{\"r\":99}");
    }
    return out;
}

// Pages through `systems` with `filter` at page size `limit`; ids in order.
bool PageSystems(Child& c, const std::string& filter, int limit, std::vector<uint32_t>& ids, std::uint64_t& total,
                 std::size_t& pages, std::string* log) {
    ids.clear();
    pages = 0;
    std::uint64_t offset = 0;
    for (int guard = 0; guard < 100000; ++guard) {
        Json rep;
        std::string raw;
        const std::string req = "{\"op\":\"systems\",\"id\":\"p" + std::to_string(guard) + "\",\"filter\":" + filter +
                                ",\"offset\":" + std::to_string(offset) + ",\"limit\":" + std::to_string(limit) + "}";
        if (!Ask(c, req, rep, &raw) || Status(rep) != "ok") return false;
        if (log != nullptr) *log += Normal(raw) + "\n";
        ++pages;
        const Json* res = Result(rep);
        if (res == nullptr || !res->is_object()) return false;
        total = static_cast<std::uint64_t>(I(res->get("total")));
        const Json* items = res->get("items");
        if (items == nullptr || !items->is_array()) return false;
        for (const Json& it : items->items) ids.push_back(static_cast<uint32_t>(I(it.get("id"))));
        const Json* next = res->get("next");
        if (next == nullptr) return false;
        if (next->is_null()) return true;
        const std::int64_t nx = I(next);
        if (nx <= static_cast<std::int64_t>(offset)) return false;
        offset = static_cast<std::uint64_t>(nx);
    }
    return false;
}

// ---------------------------------------------------------------------------
// The fixtures
// ---------------------------------------------------------------------------
bool MakeStore(Ctx& x, StoreFx& fx, const std::string& sub, const std::string& text) {
    fx.dir = x.scratch + "/" + sub;
    ::mkdir(fx.dir.c_str(), 0755);
    fx.text = text;
    fx.oracle = hecke_ext::ParseAtlasStore(text);
    fx.name = "hecke_atlas_ell" + std::to_string(fx.oracle.ell) + ".json";
    fx.path = fx.dir + "/" + fx.name;
    if (!WriteFile(fx.path, text)) return false;
    std::string err;
    if (!atlas_index::ScanStore(text.data(), text.size(), fx.name, atlas_index::FnvOf(text.data(), text.size()), fx.ix,
                                err)) {
        Check(false, "(A) the in-process scan of " + fx.name + " succeeds: " + err);
        return false;
    }
    fx.index_text = atlas_index::Serialize(fx.ix);
    fx.stamp = StampOf(fx.path);
    return fx.stamp.ok;
}

// ---------------------------------------------------------------------------
// (A) the sidecar writer
// ---------------------------------------------------------------------------
void ArmA(Ctx& x) {
    std::string out, err, idx;
    // Unit level: the scanner, the format round trip, FnvOf = FnvBytes.
    {
        const StoreFx& f = x.s23;
        Check(atlas_index::FnvOf(f.text.data(), f.text.size()) == engine::FnvBytes(engine::kFnvOffset64, f.text),
              "(A) FnvOf equals engine::FnvBytes on the store text");
        Check(f.ix.cells.size() == f.oracle.cells.size() && f.ix.systems.size() == f.oracle.systems.size(),
              "(A) the scan counts equal ParseAtlasStore's");
        bool attrs = true;
        std::size_t occ = 0;
        for (std::size_t i = 0; i < f.oracle.systems.size() && i < f.ix.systems.size(); ++i) {
            const auto& o = f.oracle.systems[i];
            const auto& r = f.ix.systems[i];
            attrs = attrs && r.r == o.r && r.type == (o.type == "eisenstein" ? 'e' : 'c') &&
                    (r.collected != 0) == o.collected && r.base == o.base && r.twist == o.twist;
            occ += o.occurrences.size();
            attrs = attrs && f.text.compare(r.off, 7 + std::to_string(i).size(), "{\"id\":" + std::to_string(i) + ",") == 0;
        }
        Check(attrs, "(A) every scanned system row's attributes and offset agree with ParseAtlasStore");
        Check(occ == f.ix.occ.size(), "(A) the scan's occurrences are the oracle's");
        atlas_index::Index back;
        std::string perr, diff;
        Check(atlas_index::Parse(f.index_text, back, perr) && atlas_index::Equal(back, f.ix, diff),
              "(A) Parse(Serialize(ix)) == ix " + perr + diff);
        atlas_index::Index junk;
        Check(!atlas_index::ScanStore("{}", 2, "x", 0, junk, perr), "(A) a non-store is not indexable");
    }
    // The writer, default output beside the store.
    const int c1 = RunToEnd(x, {"--atlas", "--index", x.s23.path}, out, err);
    Check(c1 == 0, "(A) --atlas --index <store> exits 0 (got " + std::to_string(c1) + "): " + out.substr(0, 300));
    Check(Contains(out, "ATLAS_INDEX store_bytes=" + std::to_string(x.s23.text.size()) + " store_fnv=" +
                            Hex(x.s23.ix.store_fnv) + " cells=" + std::to_string(x.s23.ix.cells.size()) +
                            " systems=" + std::to_string(x.s23.ix.systems.size())),
          "(A) the ATLAS_INDEX line carries the store's size, FNV and counts");
    Check(ReadFile(x.s23.path + ".ocidx", idx) && idx == x.s23.index_text,
          "(A) <store>.ocidx equals Serialize of the in-process scan, byte for byte");
    Check(SameStamp(StampOf(x.s23.path), x.s23.stamp), "(A) the store's bytes and mtime are unchanged by the writer");
    // --out.
    const std::string custom = x.s13.dir + "/custom.ocidx";
    const int c2 = RunToEnd(x, {"--atlas", "--index", x.s13.path, "--out", custom}, out, err);
    Check(c2 == 0 && ReadFile(custom, idx) && idx == x.s13.index_text, "(A) --out <path> writes the same index there");
    Check(!PathExists(x.s13.path + ".ocidx"), "(A) --out writes nothing beside the store");
    // Refusals.
    const int c3 = RunToEnd(x, {"--atlas", "--index", x.s23.path, "--out", x.s23.path}, out, err);
    Check(c3 == 1 && Contains(out, "is the store itself"), "(A) --out = the store is refused (exit 1)");
    Check(SameStamp(StampOf(x.s23.path), x.s23.stamp), "(A) ... and the store is untouched");
    const std::string in_tree = x.root + "/nb4_must_not_exist.ocidx";
    const int c4 = RunToEnd(x, {"--atlas", "--index", x.s23.path, "--out", in_tree}, out, err);
    Check(c4 == 1 && !PathExists(in_tree), "(A) an output inside the source tree is refused and not written");
    const int c5 = RunToEnd(x, {"--atlas", "--index", x.scratch + "/missing.json"}, out, err);
    Check(c5 == 3, "(A) a missing store exits 3");
    const std::string junk = x.scratch + "/junk/junk.json";
    ::mkdir((x.scratch + "/junk").c_str(), 0755);
    WriteFile(junk, "{\"schema\":\"hecke_atlas/v1\",\"ell\":5}\n");
    const int c6 = RunToEnd(x, {"--atlas", "--index", junk}, out, err);
    Check(c6 == 3 && Contains(out, "not indexable") && !PathExists(junk + ".ocidx"),
          "(A) a store that is not indexable exits 3 and writes nothing");
    const int c7 = RunToEnd(x, {"--atlas", "--index", x.s23.path, "--bogus"}, out, err);
    Check(c7 == 1, "(A) a stray token after --index is a usage error");
}

// ---------------------------------------------------------------------------
// (B) handshake
// ---------------------------------------------------------------------------
void CheckHello(const Json& h, const StoreFx& f, const std::string& source, const std::string& index_text,
                const std::string& tag) {
    Check(I(h.get("seq")) == 0, tag + " atlas_hello seq 0");
    Check(std::string(S(h.get("proto"))) == "oc-serve-atlas/1", tag + " proto oc-serve-atlas/1");
    Check(B(h.get("read_only"), true), tag + " read_only true");
    const Json* b = h.get("build");
    Check(b != nullptr && b->get("commit") != nullptr && b->get("config") != nullptr && I(b->get("pid")) > 0,
          tag + " build {commit, config, pid}");
    const Json* s = h.get("store");
    Check(s != nullptr && std::string(S(s->get("name"))) == f.name, tag + " store.name is the basename");
    Check(s != nullptr && I(s->get("bytes")) == static_cast<std::int64_t>(f.text.size()), tag + " store.bytes");
    Check(s != nullptr && std::string(S(s->get("fnv"))) == Hex(engine::FnvBytes(engine::kFnvOffset64, f.text)),
          tag + " store.fnv = FnvBytes of the text");
    Check(s != nullptr && std::string(S(s->get("schema"))) == "hecke_atlas/v1" && I(s->get("ell")) == f.oracle.ell &&
              I(s->get("n_max")) == f.oracle.n_max,
          tag + " store schema / ell / n_max");
    Check(s != nullptr && I(s->get("cells")) == static_cast<std::int64_t>(f.oracle.cells.size()) &&
              I(s->get("systems")) == static_cast<std::int64_t>(f.oracle.systems.size()) &&
              I(s->get("occurrences")) == static_cast<std::int64_t>(f.ix.occ.size()),
          tag + " store counts = ParseAtlasStore's");
    Check(!Contains(console::serve_json::WriteJson(h), f.dir), tag + " the store's path is not disclosed");
    const Json* ix = h.get("index");
    Check(ix != nullptr && std::string(S(ix->get("source"))) == source, tag + " index.source = " + source);
    Check(ix != nullptr && std::string(S(ix->get("fnv"))) == Hex(atlas_index::FnvOf(index_text.data(), index_text.size())),
          tag + " index.fnv = the FNV of the index text");
    const Json* l = h.get("limits");
    Check(l != nullptr && I(l->get("max_request_bytes")) == 65536 && I(l->get("page_max")) == 1000,
          tag + " limits.max_request_bytes / page_max");
    Check(h.get("open_ms") != nullptr, tag + " open_ms present");
}

// ---------------------------------------------------------------------------
// (C) the queries, logged (volatile `perf` stripped) for the sidecar <-> memory
// comparison
// ---------------------------------------------------------------------------
std::string ArmC(Ctx& x, Child& c, const StoreFx& f, const std::string& tag) {
    std::string log;
    Json rep;
    std::string raw;
    const hecke_ext::AtlasStore& st = f.oracle;
    // info
    if (Ask(c, "{\"op\":\"info\",\"id\":\"info\"}", rep, &raw) && Status(rep) == "ok") {
        log += Normal(raw) + "\n";
        const Json* res = Result(rep);
        const Json* hd = res != nullptr ? res->get("header") : nullptr;
        Check(hd != nullptr && std::string(S(hd->get("schema"))) == st.schema && I(hd->get("ell")) == st.ell &&
                  I(hd->get("n_max")) == st.n_max &&
                  std::string(S(hd->get("settings_fingerprint"))) == st.settings_fingerprint,
              tag + " info.header schema / ell / n_max / settings_fingerprint");
        Check(hd != nullptr && U64List(hd->get("primes"), st.primes), tag + " info.header.primes");
        const Json* flds = hd != nullptr ? hd->get("fields") : nullptr;
        bool fok = flds != nullptr && flds->is_array() && flds->items.size() == st.fields.size();
        for (std::size_t i = 0; fok && i < st.fields.size(); ++i)
            fok = I(flds->items[i].get("r")) == st.fields[i].r && U32List(flds->items[i].get("C"), st.fields[i].C);
        Check(fok, tag + " info.header.fields (r, C_{ell,r})");
        Check(hd != nullptr && I(hd->get("chi_count")) == st.chi_count && I(hd->get("chi_in_sigma")) == st.chi_in_sigma,
              tag + " info.header chi_count / chi_in_sigma");
        const Json* sep = hd != nullptr ? hd->get("s_sep") : nullptr;
        Check(sep != nullptr && U64List(sep->get("primes"), st.s_sep.primes) && B(sep->get("separates"), st.s_sep.separates),
              tag + " info.header.s_sep");
        const Json* s0 = hd != nullptr ? hd->get("s0_anemic") : nullptr;
        Check(s0 != nullptr && U64List(s0->get("primes"), st.s0_anemic.primes), tag + " info.header.s0_anemic");
        Check(hd != nullptr && hd->get("cells") == nullptr && hd->get("systems") == nullptr,
              tag + " info.header carries no cells / systems");
        std::int64_t eis = 0, cusp = 0, coll = 0;
        std::map<uint32_t, std::int64_t> prof;
        for (const auto& s : st.systems) {
            (s.type == "eisenstein" ? eis : cusp)++;
            coll += s.collected ? 1 : 0;
            ++prof[s.r];
        }
        Check(res != nullptr && I(res->get("cells")) == static_cast<std::int64_t>(st.cells.size()) &&
                  I(res->get("systems")) == static_cast<std::int64_t>(st.systems.size()) &&
                  I(res->get("collected")) == coll,
              tag + " info counts");
        const Json* ty = res != nullptr ? res->get("types") : nullptr;
        Check(ty != nullptr && I(ty->get("eisenstein")) == eis && I(ty->get("cuspidal")) == cusp, tag + " info.types");
        const Json* rp = res != nullptr ? res->get("r_profile") : nullptr;
        bool pok = rp != nullptr && rp->is_array() && rp->items.size() == prof.size();
        std::size_t k = 0;
        for (const auto& kv : prof) {
            pok = pok && U32List(&rp->items[k], {kv.first, static_cast<uint32_t>(kv.second)});
            ++k;
        }
        Check(pok, tag + " info.r_profile");
    } else {
        Check(false, tag + " info answers ok");
    }
    // cells, one page
    if (Ask(c, "{\"op\":\"cells\",\"id\":\"cells\",\"limit\":1000}", rep, &raw) && Status(rep) == "ok") {
        log += Normal(raw) + "\n";
        const Json* res = Result(rep);
        const Json* items = res != nullptr ? res->get("items") : nullptr;
        bool ok = items != nullptr && items->is_array() && items->items.size() == st.cells.size() &&
                  I(res->get("total")) == static_cast<std::int64_t>(st.cells.size()) && res->get("next") != nullptr &&
                  res->get("next")->is_null();
        for (std::size_t i = 0; ok && i < st.cells.size(); ++i) {
            const Json& it = items->items[i];
            const auto& w = st.cells[i];
            const Json* u = it.get("u_ell");
            ok = I(it.get("n")) == w.n && U32List(it.get("dims"), w.dims) && I(it.get("ideals")) == w.ideals &&
                 I(it.get("local_ideals")) == w.local_ideals && u != nullptr && B(u->get("included"), w.u_ell_included) &&
                 std::string(S(u->get("reason"))) == w.u_ell_reason;
        }
        Check(ok, tag + " cells: every cell row equals the oracle's, next null");
    } else {
        Check(false, tag + " cells answers ok");
    }
    // every cell with its occurrences
    bool cells_ok = true;
    for (const auto& w : st.cells) {
        if (!Ask(c, "{\"op\":\"cell\",\"id\":\"c\",\"n\":" + std::to_string(w.n) + "}", rep, &raw) || Status(rep) != "ok") {
            cells_ok = false;
            continue;
        }
        log += Normal(raw) + "\n";
        std::vector<std::pair<uint32_t, uint32_t>> want, got;
        for (const auto& s : st.systems)
            for (const auto& o : s.occurrences)
                if (o.n == w.n) want.emplace_back(o.ideal, s.id);
        std::sort(want.begin(), want.end());
        const Json* res = Result(rep);
        const Json* occ = res != nullptr ? res->get("occurrences") : nullptr;
        if (occ == nullptr || !occ->is_array() || res->get("cell") == nullptr || I(res->get("cell")->get("n")) != w.n) {
            cells_ok = false;
            continue;
        }
        for (const Json& o : occ->items)
            got.emplace_back(static_cast<uint32_t>(I(o.get("ideal"))), static_cast<uint32_t>(I(o.get("sid"))));
        cells_ok = cells_ok && got == want;
    }
    Check(cells_ok, tag + " cell n: every cell and its occurrences {ideal, sid} equal the oracle's");
    // every system by id
    std::size_t sys_bad = 0;
    std::string first_why;
    for (const auto& o : st.systems) {
        std::string why;
        if (!Ask(c, "{\"op\":\"system\",\"id\":\"s\",\"sid\":" + std::to_string(o.id) + "}", rep, &raw) ||
            Status(rep) != "ok" || Result(rep) == nullptr || !SystemMatches(Result(rep)->get("system"), o, why)) {
            if (sys_bad++ == 0) first_why = "sid " + std::to_string(o.id) + ": " + why;
            continue;
        }
        log += Normal(raw) + "\n";
    }
    Check(sys_bad == 0, tag + " system sid: every system equals the oracle's (" + std::to_string(sys_bad) +
                            " differ; first: " + first_why + ")");
    // systems under every filter, paged at 7 and at 1000
    for (const Filter& f : Filters(st)) {
        const std::vector<uint32_t> want = OracleFilter(st, f);
        std::vector<uint32_t> got7, got1000;
        std::uint64_t total7 = 0, total1000 = 0;
        std::size_t pages7 = 0, pages1000 = 0;
        const bool ok7 = PageSystems(c, f.json, 7, got7, total7, pages7, &log);
        const bool ok1000 = PageSystems(c, f.json, 1000, got1000, total1000, pages1000, nullptr);
        Check(ok7 && got7 == want && total7 == want.size() &&
                  pages7 == (want.empty() ? 1 : (want.size() + 6) / 7),
              tag + " systems filter " + f.json + ": pages of 7 concatenate to the oracle's " +
                  std::to_string(want.size()) + " ids (got " + std::to_string(got7.size()) + ")");
        Check(ok1000 && got1000 == want && pages1000 == 1, tag + " systems filter " + f.json + ": one page of 1000");
    }
    // the full row content of a filtered page equals `system`
    if (Ask(c, "{\"op\":\"systems\",\"id\":\"full\",\"filter\":{\"r\":2},\"limit\":3}", rep, &raw) && Status(rep) == "ok") {
        const Json* items = Result(rep) != nullptr ? Result(rep)->get("items") : nullptr;
        bool ok = items != nullptr && items->is_array();
        for (std::size_t i = 0; ok && i < items->items.size(); ++i) {
            const std::int64_t id = I(items->items[i].get("id"));
            std::string why;
            ok = id >= 0 && static_cast<std::size_t>(id) < st.systems.size() &&
                 SystemMatches(&items->items[i], st.systems[static_cast<std::size_t>(id)], why);
        }
        Check(ok, tag + " systems items are full system rows");
    }
    (void)x;
    return log;
}

// ---------------------------------------------------------------------------
// (D) errors and the process lifecycle
// ---------------------------------------------------------------------------
void ArmD(Ctx& x) {
    Child c;
    Json hello, rep;
    if (!StartServed(x, c, {"--store", x.s23.path}, hello)) {
        Check(false, "(D) a served child starts");
        return;
    }
    struct Row {
        std::string req, status, kind, id;
    };
    const std::string N = std::to_string(x.s23.oracle.systems.size());
    const std::vector<Row> rows = {
        {"this is not json", "rejected", "protocol", ""},
        {"[1,2]", "rejected", "protocol", ""},
        {"{\"id\":\"e1\"}", "rejected", "protocol", "e1"},
        {"{\"op\":7,\"id\":\"e2\"}", "rejected", "protocol", "e2"},
        {"{\"op\":\"bogus\",\"id\":\"e3\"}", "rejected", "protocol", "e3"},
        {"{\"op\":\"info\"}", "rejected", "protocol", ""},
        {"{\"op\":\"info\",\"id\":\"\"}", "rejected", "protocol", ""},
        {"{\"op\":\"info\",\"id\":\"e4\",\"extra\":1}", "rejected", "protocol", "e4"},
        {"{\"op\":\"cells\",\"id\":\"e5\",\"limit\":0}", "rejected", "protocol", "e5"},
        {"{\"op\":\"cells\",\"id\":\"e6\",\"limit\":1001}", "rejected", "protocol", "e6"},
        {"{\"op\":\"cells\",\"id\":\"e7\",\"offset\":-1}", "rejected", "protocol", "e7"},
        {"{\"op\":\"cells\",\"id\":\"e8\",\"limit\":\"5\"}", "rejected", "protocol", "e8"},
        {"{\"op\":\"cell\",\"id\":\"e9\"}", "rejected", "protocol", "e9"},
        {"{\"op\":\"cell\",\"id\":\"e10\",\"n\":1}", "error", "not_found", "e10"},
        {"{\"op\":\"system\",\"id\":\"e11\",\"sid\":" + N + "}", "error", "not_found", "e11"},
        {"{\"op\":\"system\",\"id\":\"e12\",\"sid\":-1}", "rejected", "protocol", "e12"},
        {"{\"op\":\"system\",\"id\":\"e13\",\"sid\":1.5}", "rejected", "protocol", "e13"},
        {"{\"op\":\"systems\",\"id\":\"e14\",\"filter\":{\"type\":\"bogus\"}}", "rejected", "protocol", "e14"},
        {"{\"op\":\"systems\",\"id\":\"e15\",\"filter\":{\"zzz\":1}}", "rejected", "protocol", "e15"},
        {"{\"op\":\"systems\",\"id\":\"e16\",\"filter\":3}", "rejected", "protocol", "e16"},
        {"{\"op\":\"systems\",\"id\":\"e17\",\"filter\":{\"collected\":1}}", "rejected", "protocol", "e17"},
        {"{\"op\":\"system\",\"id\":\"e18\",\"path\":\"/etc/passwd\",\"sid\":0}", "rejected", "protocol", "e18"},
    };
    for (const Row& r : rows) {
        const bool got = Ask(c, r.req, rep);
        Check(got && Status(rep) == r.status && ErrKind(rep) == r.kind,
              "(D) " + r.req + " -> " + r.status + " / " + r.kind + " (got " + Status(rep) + " / " + ErrKind(rep) + ")");
        const Json* idj = rep.get("id");
        Check(r.id.empty() ? (idj != nullptr && idj->is_null()) : std::string(S(idj)) == r.id,
              "(D) " + r.req + " -> the reply's id");
        Check(rep.get("result") != nullptr && rep.get("result")->is_null(), "(D) an error reply has result null");
        Json ok;
        Check(Ask(c, "{\"op\":\"system\",\"id\":\"alive\",\"sid\":0}", ok) && Status(ok) == "ok",
              "(D) the process survives " + r.req);
    }
    Check(Ask(c, "{\"op\":\"cells\",\"id\":\"past\",\"offset\":100000}", rep) && Status(rep) == "ok" &&
              Result(rep) != nullptr && Result(rep)->get("items") != nullptr &&
              Result(rep)->get("items")->items.empty() && Result(rep)->get("next")->is_null(),
          "(D) an offset past the end is an empty page, next null");
    Check(c.send_line("") && Ask(c, "{\"op\":\"info\",\"id\":\"after-blank\"}", rep) && Status(rep) == "ok",
          "(D) a blank line is ignored (no frame)");
    Check(c.send_line("{\"op\":\"close\"}"), "(D) close is sent");
    Json bye;
    Check(c.frame(bye, 30000) && std::string(S(bye.get("kind"))) == "atlas_bye", "(D) close -> atlas_bye");
    Check(c.wait_exit(30000) == 0, "(D) close -> exit 0");
    Check(c.seq_ok && c.all_json && c.no_blank, "(D) every frame is JSON, no blank line, seq strictly +1 from 0");
    {
        Child e;
        Json h;
        if (StartServed(x, e, {"--store", x.s23.path}, h)) {
            e.close_in();
            Json more;
            Check(!e.frame(more, 30000), "(D) stdin EOF: no further frame (no bye)");
            Check(e.wait_exit(30000) == 0, "(D) stdin EOF -> exit 0");
        } else {
            Check(false, "(D) the EOF child starts");
        }
    }
    {
        Child o;
        Json h;
        if (StartServed(x, o, {"--store", x.s23.path}, h)) {
            o.send(std::string(70000, 'a'));
            Json f;
            Check(o.frame(f, 30000) && std::string(S(f.get("kind"))) == "atlas_fatal" &&
                      std::string(S(f.get("reason"))) == "protocol_overflow",
                  "(D) a request line over 65536 bytes -> atlas_fatal protocol_overflow");
            Check(o.wait_exit(30000) == 70, "(D) ... exit 70");
        } else {
            Check(false, "(D) the overflow child starts");
        }
    }
}

// ---------------------------------------------------------------------------
// (E) the frame cap
// ---------------------------------------------------------------------------
void ArmE(Ctx& x) {
    Child c;
    Json hello;
    if (!StartServed(x, c, {"--store", x.s23.path, "--max-reply-bytes", "4096"}, hello)) {
        Check(false, "(E) a child with --max-reply-bytes 4096 starts");
        return;
    }
    Check(hello.get("limits") != nullptr && I(hello.get("limits")->get("max_reply_bytes")) == 4096,
          "(E) hello.limits.max_reply_bytes = 4096");
    std::vector<uint32_t> ids;
    std::uint64_t total = 0;
    std::size_t pages = 0;
    const bool ok = PageSystems(c, "{}", 1000, ids, total, pages, nullptr);
    std::vector<uint32_t> want(x.s23.oracle.systems.size());
    for (std::size_t i = 0; i < want.size(); ++i) want[i] = static_cast<uint32_t>(i);
    Check(ok && ids == want && pages >= 2, "(E) a 1000-item request is cut at the cap into " + std::to_string(pages) +
                                               " pages that concatenate to every id");
    Json rep;
    Check(Ask(c, "{\"op\":\"info\",\"id\":\"i\"}", rep) && (Status(rep) == "ok" || ErrKind(rep) == "too_large"),
          "(E) info answers ok or too_large under the cap, never an oversized frame");
    Check(c.max_frame <= 4096, "(E) no frame exceeds 4096 bytes (max " + std::to_string(c.max_frame) + ")");
    std::string out, err;
    Child small;
    Check(small.start(x.exe, {"--serve-atlas", "--store", x.s23.path, "--max-reply-bytes", "100"}, x.env, x.cwd) &&
              small.wait_exit(30000) == 1 && small.obuf.empty(),
          "(E) --max-reply-bytes below 4096 is refused");
}

// ---------------------------------------------------------------------------
// (F) stale / corrupt indexes, --expect-fnv
// ---------------------------------------------------------------------------
void Refused(Ctx& x, const std::vector<std::string>& args, const std::string& needle, const std::string& what,
             const std::vector<std::string>* env = nullptr) {
    Child c;
    std::vector<std::string> a = {"--serve-atlas"};
    a.insert(a.end(), args.begin(), args.end());
    if (!c.start(x.exe, a, env != nullptr ? *env : x.env, x.cwd)) {
        Check(false, what + ": the child starts");
        return;
    }
    c.close_in();
    const int code = c.wait_exit(60000);
    Check(code == 1, what + ": exit 1 (got " + std::to_string(code) + ")");
    Check(c.obuf.empty(), what + ": nothing on stdout");
    Check(Contains(c.ebuf, "serve-atlas: refused: ") && Contains(c.ebuf, needle),
          what + ": stderr names the reason '" + needle + "' (got '" + c.ebuf.substr(0, 200) + "')");
}

void ArmF(Ctx& x) {
    const std::string d = x.scratch + "/stale";
    ::mkdir(d.c_str(), 0755);
    // The index of the other store.
    Refused(x, {"--store", x.s23.path, "--index", x.s13.dir + "/custom.ocidx"}, "stale or corrupt index",
            "(F) the index of another store");
    // An edited S line with a re-computed end hash.
    {
        atlas_index::Index ix = x.s23.ix;
        ix.systems[0].type = ix.systems[0].type == 'e' ? 'c' : 'e';
        WriteFile(d + "/edited.ocidx", atlas_index::Serialize(ix));
        Refused(x, {"--store", x.s23.path, "--index", d + "/edited.ocidx"}, "differs from the store",
                "(F) an edited S line with a consistent end hash");
    }
    // A broken end hash.
    {
        std::string t = x.s23.index_text;
        const std::size_t at = t.find("\nC\t");
        if (at != std::string::npos) t[at + 3] = t[at + 3] == '9' ? '8' : '9';
        WriteFile(d + "/broken.ocidx", t);
        Refused(x, {"--store", x.s23.path, "--index", d + "/broken.ocidx"}, "end hash", "(F) a broken end hash");
    }
    // A store edited after its index was written (same length): the stale sidecar beside it is refused.
    {
        const std::string sd = d + "/edited_store";
        ::mkdir(sd.c_str(), 0755);
        std::string t = x.s23.text;
        const std::size_t ev = t.find("\"eigenvalues\":[[");
        const std::size_t dig = ev + 16;
        t[dig] = t[dig] == '1' ? '2' : '1';
        WriteFile(sd + "/" + x.s23.name, t);
        WriteFile(sd + "/" + x.s23.name + ".ocidx", x.s23.index_text);
        Refused(x, {"--store", sd + "/" + x.s23.name}, "stale or corrupt index", "(F) a stale sidecar beside an edited store");
    }
    Refused(x, {"--store", x.s23.path, "--expect-fnv", "0x0123456789abcdef"}, "differs from --expect-fnv",
            "(F) a wrong --expect-fnv");
    {
        Child c;
        Json h;
        Check(StartServed(x, c, {"--store", x.s23.path, "--expect-fnv", Hex(x.s23.ix.store_fnv)}, h),
              "(F) the right --expect-fnv starts");
    }
    Refused(x, {"--store", x.scratch + "/junk/junk.json"}, "not indexable", "(F) a store that is not indexable");
    Refused(x, {"--store", x.scratch + "/nothing.json"}, "cannot resolve", "(F) a missing store");
    Refused(x, {"--store", x.s23.path, "--index", x.scratch + "/nothing.ocidx"}, "the index", "(F) a missing --index");
    Refused(x, {}, "--store", "(F) no --store");
}

// ---------------------------------------------------------------------------
// (G) environment gate and hardening
// ---------------------------------------------------------------------------
std::string ProcFile(pid_t pid, const std::string& name) {
    std::string t;
    ReadFile("/proc/" + std::to_string(pid) + "/" + name, t);
    return t;
}

void ArmG(Ctx& x) {
    const std::string outdir = x.scratch + "/atlas_out";
    for (const std::string& kv : {std::string("GOLDEN_EMIT=1"), "HECKE_ATLAS_OUT=" + outdir, std::string("ENGINE_DEVICE=gpu"),
                                  std::string("ASAN_OPTIONS=detect_leaks=0"), std::string("OC_X=1"),
                                  std::string("CONSOLE_EMIT_EXPECTED=1"), std::string("HECKE_NEBENTYPUS_NB3_FULL=1")}) {
        const std::vector<std::string> env = WithEnv(x.env, kv);
        const std::string name = kv.substr(0, kv.find('='));
        Refused(x, {"--store", x.s23.path}, "environment variable " + name + " is not permitted in a served session",
                "(G) env " + name, &env);
    }
    Check(!PathExists(outdir), "(G) the refused HECKE_ATLAS_OUT directory was never created");
    Refused(x, {"--store", x.s23.path, "--set", "print_max_entries=5"}, "is not accepted", "(G) --set with --serve-atlas");
    Refused(x, {"--store", x.s23.path, "ConsoleServeAtlasValidation"}, "is not accepted", "(G) a validation name");
    Refused(x, {"--store", x.s23.path, "--store", x.s13.path}, "is not accepted", "(G) a second --store");
    Refused(x, {"--store"}, "is not accepted", "(G) --store without a value");
    {
        const std::string f = x.scratch + "/stdout_file.txt";
        Child c;
        if (c.start(x.exe, {"--serve-atlas", "--store", x.s23.path}, x.env, x.cwd, f)) {
            c.close_in();
            const int code = c.wait_exit(30000);
            std::string t;
            ReadFile(f, t);
            Check(code == 1 && t.empty() && Contains(c.ebuf, "not a regular file"), "(G) a regular-file stdout is refused");
        }
        ::unlink(f.c_str());
    }
    // The running process.
    Child c;
    Json h;
    if (!StartServed(x, c, {"--store", x.s23.path}, h)) {
        Check(false, "(G) a served child starts");
        return;
    }
    const std::string limits = ProcFile(c.pid, "limits");
    auto limit_zero = [&](const std::string& label) {
        const std::size_t at = limits.find(label);
        if (at == std::string::npos) return false;
        std::istringstream in(limits.substr(at + label.size()));
        std::string soft, hard;
        in >> soft >> hard;
        return soft == "0" && hard == "0";
    };
    Check(limit_zero("Max file size"), "(G) RLIMIT_FSIZE = 0 (/proc/<pid>/limits)");
    Check(limit_zero("Max core file size"), "(G) RLIMIT_CORE = 0 (/proc/<pid>/limits)");
    Check(Contains(ProcFile(c.pid, "status"), "NoNewPrivs:\t1"), "(G) NoNewPrivs 1");
    const std::string real = RealPath(x.s23.path);
    bool fd_found = false, fd_ro = true;
    const std::string fddir = "/proc/" + std::to_string(c.pid) + "/fd";
    if (DIR* d = ::opendir(fddir.c_str())) {
        while (struct dirent* e = ::readdir(d)) {
            const std::string n = e->d_name;
            if (n == "." || n == "..") continue;
            char buf[PATH_MAX];
            const ssize_t k = ::readlink((fddir + "/" + n).c_str(), buf, sizeof(buf) - 1);
            if (k <= 0) continue;
            buf[k] = '\0';
            if (std::string(buf) != real) continue;
            fd_found = true;
            const std::string info = ProcFile(c.pid, "fdinfo/" + n);
            const std::size_t at = info.find("flags:");
            if (at == std::string::npos) {
                fd_ro = false;
                continue;
            }
            const unsigned long flags = std::strtoul(info.c_str() + at + 6, nullptr, 8);
            fd_ro = fd_ro && (flags & O_ACCMODE) == O_RDONLY;
        }
        ::closedir(d);
    }
    Check(fd_found && fd_ro, "(G) the store's descriptor is open O_RDONLY (/proc/<pid>/fdinfo)");
    bool map_found = false, map_ro = true;
    {
        std::istringstream in(ProcFile(c.pid, "maps"));
        std::string l;
        while (std::getline(in, l)) {
            if (l.size() < real.size() || l.compare(l.size() - real.size(), real.size(), real) != 0) continue;
            map_found = true;
            std::istringstream ls(l);
            std::string range, perms;
            ls >> range >> perms;
            map_ro = map_ro && perms.size() >= 2 && perms[1] == '-';
        }
    }
    Check(map_found && map_ro, "(G) the store's mapping has no w permission (/proc/<pid>/maps)");
    const std::string env = ProcFile(c.pid, "environ");
    Check(!Contains(env, "HECKE_") && !Contains(env, "GOLDEN_"), "(G) the served child's environment is the whitelist");
}

// ---------------------------------------------------------------------------
// (R) the real Sigma(691) store, read-only, opt-in
// ---------------------------------------------------------------------------
// (Q) slot-22 VERB-3 (owner decision O13): `--atlas --query <store> '<json>'` is the
// served request handler run once.  Every request below is answered by a served
// child and by the one-shot; the frames agree in kind, id, op, status, error,
// store and result (seq and perf are volatile); exit 0 on `ok`, 3 otherwise;
// refusals (missing store, wrong arity, `close`, over-long) exit 1, stdout empty.
void ArmQ(Ctx& x) {
    const std::vector<std::string> reqs = {
        "{\"op\":\"info\",\"id\":\"q1\"}",
        "{\"op\":\"cells\",\"id\":\"q2\",\"limit\":4}",
        "{\"op\":\"cell\",\"id\":\"q3\",\"n\":2}",
        "{\"op\":\"system\",\"id\":\"q4\",\"sid\":0}",
        "{\"op\":\"systems\",\"id\":\"q5\",\"filter\":{\"r\":2},\"limit\":7}",
        "{\"op\":\"systems\",\"id\":\"q6\",\"filter\":{\"type\":\"cuspidal\"},\"offset\":1,\"limit\":3}",
        "{\"op\":\"cell\",\"id\":\"q7\",\"n\":999999}",
        "{\"op\":\"system\",\"id\":\"q8\",\"sid\":99999999}",
        "{\"op\":\"bogus\",\"id\":\"q9\"}",
        "{\"op\":\"info\"}",
        "not json"};
    auto core = [](const std::string& raw) {
        Json j;
        std::string e;
        if (!console::serve_json::ParseJson(raw, j, e) || !j.is_object()) return std::string("<unparsed>");
        j.members.erase(std::remove_if(j.members.begin(), j.members.end(),
                                       [](const std::pair<std::string, Json>& m) {
                                           return m.first == "perf" || m.first == "seq";
                                       }),
                        j.members.end());
        return console::serve_json::WriteJson(j);
    };
    Child c;
    Json h;
    if (!StartServed(x, c, {"--store", x.s23.path}, h)) {
        Check(false, "(Q) the served reference child starts: " + c.ebuf.substr(0, 300));
        return;
    }
    std::size_t agree = 0, exits = 0;
    for (const std::string& r : reqs) {
        Json rep;
        std::string raw_served;
        const bool a = Ask(c, r, rep, &raw_served);
        std::string out, err;
        const int code = RunToEnd(x, {"--atlas", "--query", x.s23.path, r}, out, err);
        while (!out.empty() && out.back() == '\n') out.pop_back();
        const bool one_line = out.find('\n') == std::string::npos;
        agree += (a && one_line && core(out) == core(raw_served)) ? 1 : 0;
        exits += (code == (Status(rep) == "ok" ? 0 : 3)) ? 1 : 0;
    }
    c.send_line("{\"op\":\"close\"}");
    c.wait_exit(30000);
    Check(agree == reqs.size(), "(Q) the one-shot reply equals the served reply for " + std::to_string(reqs.size()) +
                                    " requests (" + std::to_string(agree) + " agree)");
    Check(exits == reqs.size(),
          "(Q) exit 0 for an `ok` reply, 3 for `error` / `rejected` (" + std::to_string(exits) + ")");
    std::string out, err;
    Check(RunToEnd(x, {"--atlas", "--query", x.s23.path + ".absent", reqs[0]}, out, err) == 1 && out.empty(),
          "(Q) a missing store: exit 1, empty stdout");
    Check(RunToEnd(x, {"--atlas", "--query", x.s23.path}, out, err) == 1 && out.empty(),
          "(Q) one argument: exit 1, empty stdout");
    Check(RunToEnd(x, {"--atlas", "--query", x.s23.path, "{\"op\":\"close\"}"}, out, err) == 1 && out.empty(),
          "(Q) op close is refused: exit 1, empty stdout");
    Check(RunToEnd(x, {"--atlas", "--query", x.s23.path, std::string(70000, ' ')}, out, err) == 1 && out.empty(),
          "(Q) an over-long request: exit 1, empty stdout");
    Check(RunToEnd(x, {"--atlas", "--query", x.s23.path, "   "}, out, err) == 1 && out.empty(),
          "(Q) a blank request: exit 1, empty stdout");
    Check(RunToEnd(x, {"--atlas", "--query", x.s23.path, "{\"op\":\"info\",\n\"id\":\"q1\"}"}, out, err) == 1 &&
              out.empty(),
          "(Q) a request with a newline: exit 1, empty stdout");
    std::string info;
    RunToEnd(x, {"--atlas", "--query", x.s23.path, reqs[0]}, info, err);
    Json ij;
    std::string pe;
    const bool pj = console::serve_json::ParseJson(info, ij, pe) && ij.is_object();
    Check(pj && std::string(S(ij.get("store"))) == Hex(atlas_index::FnvOf(x.s23.text.data(), x.s23.text.size())),
          "(Q) the reply names the store's FNV");
}

std::string ArmR(Ctx& x) {
    const char* dir = std::getenv("HECKE_ATLAS_IN");
    if (dir == nullptr || *dir == '\0') return "absent";
    const std::string path = std::string(dir) + "/hecke_atlas_ell691.json";
    if (!PathExists(path)) {
        Check(false, "(R) HECKE_ATLAS_IN is set but " + path + " does not exist");
        return "absent";
    }
    struct stat before, after;
    ::stat(path.c_str(), &before);
    Child c;
    Json h;
    if (!StartServed(x, c, {"--store", path, "--expect-fnv", "0x73c7210047a81b97"}, h, nullptr, 600000)) {
        Check(false, "(R) --serve-atlas starts on the real store: " + c.ebuf.substr(0, 300));
        return "absent";
    }
    const Json* s = h.get("store");
    Check(s != nullptr && I(s->get("ell")) == 691 && I(s->get("cells")) == 346 && I(s->get("systems")) == 1002225 &&
              I(s->get("bytes")) == 451875793 && std::string(S(s->get("fnv"))) == "0x73c7210047a81b97",
          "(R) the real store: ell 691, 346 cells, 1002225 systems, 451875793 bytes, FNV 0x73c7210047a81b97");
    const std::string source = h.get("index") != nullptr ? S(h.get("index")->get("source")) : "";
    Json rep;
    Check(Ask(c, "{\"op\":\"system\",\"id\":\"r0\",\"sid\":0}", rep) && Status(rep) == "ok" &&
              std::string(S(Result(rep)->get("system")->get("type"))) == "eisenstein" &&
              U32List(Result(rep)->get("system")->get("chi"), {0, 1}),
          "(R) system 0 is the Eisenstein system chi_{0,1}");
    Check(Ask(c, "{\"op\":\"system\",\"id\":\"rN\",\"sid\":1002224}", rep) && Status(rep) == "ok" &&
              I(Result(rep)->get("system")->get("r")) == 15 && I(Result(rep)->get("system")->get("twist")) == 689,
          "(R) system 1002224: r 15, twist 689");
    Check(Ask(c, "{\"op\":\"cell\",\"id\":\"rc\",\"n\":690}", rep) && Status(rep) == "ok" &&
              I(Result(rep)->get("cell")->get("n")) == 690,
          "(R) cell 690");
    Check(Ask(c, "{\"op\":\"systems\",\"id\":\"rs\",\"filter\":{\"r\":54},\"limit\":5}", rep) && Status(rep) == "ok" &&
              I(Result(rep)->get("total")) == 1380,
          "(R) systems with r = 54: 1380");
    c.send_line("{\"op\":\"close\"}");
    c.wait_exit(30000);
    ::stat(path.c_str(), &after);
    Check(before.st_size == after.st_size && before.st_mtim.tv_sec == after.st_mtim.tv_sec &&
              before.st_mtim.tv_nsec == after.st_mtim.tv_nsec,
          "(R) the real store's size and mtime are unchanged");
    return source.empty() ? "absent" : source;
}

} // namespace
#endif

uint32_t ConsoleServeAtlasValidation() {
    if (!oc::kBuildNotebook) return oc::BuildSwitchSkip("ConsoleServeAtlasValidation", "OC_BUILD_NOTEBOOK");
#if OC_BUILD_NOTEBOOK
    std::cout << "=== ConsoleServeAtlasValidation ([P] v3.11 Sec. 18.16, oc-serve-atlas/1; slot-20 NB-4) ===" << std::endl;
    g_checks = 0;
    g_fail = 0;
    g_queries = 0;
    Ctx x;
    x.exe = SelfExe();
    x.root = RealPath(FindRepoRoot());
    if (x.exe.empty() || x.root.empty()) {
        std::cout << "  [SKIP] ConsoleServeAtlasValidation: the executable or the source tree was not found" << std::endl;
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
    void (*old_pipe)(int) = std::signal(SIGPIPE, SIG_IGN);
    const auto t0 = std::chrono::steady_clock::now();

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

    std::string real = "absent";
    try {
        const hecke_ext::Atlas a23 = hecke_ext::BuildAtlas(23);
        const hecke_ext::Atlas a13 = hecke_ext::BuildAtlas(13);
        const bool fx = MakeStore(x, x.s23, "s23", a23.store.to_json()) && MakeStore(x, x.s13, "s13", a13.store.to_json()) &&
                        MakeStore(x, x.mem, "mem", a23.store.to_json());
        Check(fx, "the fixtures Sigma(23), Sigma(13) are written into the scratch directory");
        if (fx) {
            x.cells = x.s23.oracle.cells.size() + x.s13.oracle.cells.size();
            x.systems = x.s23.oracle.systems.size() + x.s13.oracle.systems.size();
            bool r2 = false, cusp = false, twist = false;
            for (const auto& s : x.s23.oracle.systems) {
                r2 = r2 || s.r == 2;
                cusp = cusp || s.type == "cuspidal";
                twist = twist || s.twist != 0;
            }
            Check(r2 && cusp && twist, "the Sigma(23) fixture has r = 2, cuspidal and twisted systems");
            Check(DirEntries(x.cwd) == 0, "(H) the working directory is empty before the run");

            ArmA(x);
            // (B) + (C): sidecar beside the store, explicit --index, memory.
            std::string log_side, log_mem;
            {
                Child c;
                Json h;
                if (StartServed(x, c, {"--store", x.s23.path}, h)) {
                    CheckHello(h, x.s23, "sidecar", x.s23.index_text, "(B) sidecar");
                    log_side = ArmC(x, c, x.s23, "(C) sidecar");
                    Check(c.seq_ok && c.all_json && c.no_blank, "(C) sidecar: every frame JSON, seq +1, no blank");
                } else {
                    Check(false, "(B) the sidecar child starts: " + c.ebuf.substr(0, 300));
                }
            }
            {
                Child c;
                Json h;
                if (StartServed(x, c, {"--store", x.mem.path}, h)) {
                    CheckHello(h, x.mem, "memory", x.mem.index_text, "(B) memory");
                    log_mem = ArmC(x, c, x.mem, "(C) memory");
                } else {
                    Check(false, "(B) the memory child starts: " + c.ebuf.substr(0, 300));
                }
            }
            Check(!log_side.empty() && log_side == log_mem,
                  "(C) the sidecar and memory reply streams are byte-identical (perf stripped; " +
                      std::to_string(log_side.size()) + " bytes)");
            {
                Child c;
                Json h;
                if (StartServed(x, c, {"--store", x.s13.path, "--index", x.s13.dir + "/custom.ocidx"}, h)) {
                    CheckHello(h, x.s13, "sidecar", x.s13.index_text, "(B) --index");
                    ArmC(x, c, x.s13, "(C) Sigma(13)");
                } else {
                    Check(false, "(B) the --index child starts: " + c.ebuf.substr(0, 300));
                }
            }
            ArmD(x);
            ArmE(x);
            ArmF(x);
            ArmG(x);
            ArmQ(x);
            real = ArmR(x);
            // (H) nothing was written.
            Check(SameStamp(StampOf(x.s23.path), x.s23.stamp), "(H) the Sigma(23) store is byte- and mtime-unchanged");
            Check(SameStamp(StampOf(x.s13.path), x.s13.stamp), "(H) the Sigma(13) store is byte- and mtime-unchanged");
            Check(SameStamp(StampOf(x.mem.path), x.mem.stamp), "(H) the memory-index store is byte- and mtime-unchanged");
            Check(!PathExists(x.mem.path + ".ocidx"), "(H) no sidecar appeared beside the memory-index store");
            Check(DirEntries(x.cwd) == 0, "(H) the working directory is empty after the run");
        }
    } catch (const std::exception& e) {
        Check(false, std::string("no exception escapes: ") + e.what());
    }
    if (git) {
        git_status(git_after);
        Check(git_after == git_before, "(H) git status --porcelain of the source tree is unchanged");
    }
    std::signal(SIGPIPE, old_pipe);
    RemoveTree(x.scratch);

    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "EFFECTIVE ConsoleServeAtlasValidation arm=ALL cells=" << x.cells << " systems=" << x.systems
              << " queries=" << g_queries << " real=" << real << " git=" << (git ? "present" : "absent") << std::endl;
    std::cout << "ConsoleServeAtlasValidation: " << g_checks << " checks, " << g_fail << " failing, " << secs << " s"
              << std::endl;
    std::cout << "ConsoleServeAtlasValidation: " << (g_fail == 0 ? "PASSED" : "FAILED") << std::endl;
    return g_fail;
#else
    return 0;
#endif
}
