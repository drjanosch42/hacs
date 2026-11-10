// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Serve.cpp -- see Serve.h.  [P] v3.10 Sec. 18 is the specification; the
// section numbers below are its.
// ===========================================================================

#include "Serve.h"

#include "Bindings.h"
#include "Display.h"
#include "HostProbe.h"
#include "PrintJson.h"
#include "ReadOnlyTags.h"
#include "ServeJson.h"

#include "../Core/Ast.h"
#include "../Core/Cancel.h"
#include "../Core/ConsoleCli.h"
#include "../Core/Confirm.h"
#include "../Core/ConsoleError.h"
#include "../Core/CostModel.h"
#include "../Core/Evaluator.h"
#include "../Core/Object.h"
#include "../Core/Parser.h"
#include "../Core/Printer.h"
#include "../Core/Registry.h"
#include "../Core/Session.h"
#include "../Core/Settings.h"
#include "../Core/Value.h"

#include "../../Util/Rss.h"
#include "../../../ModArithmetic/Fnv.h"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <streambuf>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

extern char** environ;

#include "../../BuildInfo.h"
#ifndef OC_BUILD_CONFIG
#define OC_BUILD_CONFIG "unknown"
#endif
#ifndef OC_ENABLE_CUDA
#define OC_ENABLE_CUDA 0
#endif

namespace console {
namespace serve {

namespace {

using serve_json::IntJson;
using serve_json::Json;
using serve_json::Quote;

// ---------------------------------------------------------------------------
// Process-wide state (one session per process, Sec. 18.2).
// ---------------------------------------------------------------------------
int g_proto_fd = -1;                    // the protocol stream (the original stdout)
std::mutex g_out_mutex;                 // one whole frame at a time
std::uint64_t g_seq = 0;                // hello = 0
std::atomic<bool> g_interrupt{false};   // Sec. 18.6
std::atomic<bool> g_running{false};     // an `eval` is executing
volatile std::sig_atomic_t g_term = 0;  // SIGTERM seen
std::size_t g_max_reply = kDefaultMaxReplyBytes;
std::uint64_t g_rss_cap_mib = 0;        // 0 = off

// The coverage of the conformance run ([P] Sec. 18.11 (C2)): one flag per
// overload of the FULL registry, in registration order, set when its served
// wrapper ran.  Reported on stderr at a clean exit.
std::vector<std::atomic<bool>>* g_called = nullptr;

// A cooperative interrupt (Sec. 18.6) and a dynamic refusal of an untagged
// overload / non-settable `set` (Sec. 18.7 `forbidden_overload`).  Neither
// derives from std::exception or ConsoleError, so no `catch` of the evaluator,
// of a verb or of `expect_error` can swallow one: it unwinds to the cell.
struct ServeInterrupted {
    SourcePos pos;
};
struct ServeForbidden {
    std::string message;
    SourcePos pos;
    std::string verb;
};

// ---------------------------------------------------------------------------
// (NBCONFIRM S4) The per-cell confirmation ([P] Sec. 18.4, design Sec. 2.5).
//
// A call outside its silent envelope whose estimate needs a confirmation (rules
// 2-3; rule 1 is a hard refusal without a token, rule 4 runs silently) stops
// the cell with done{status:"confirm_required", exit:3, confirm:{token, ...}}.
// The client re-sends the SAME eval with "confirm":[token, ...]; the token is
// FNV-1a-64 (ModArithmetic/Fnv.h) over this session's nonce, the cell text,
// the call site, the call, the verb, the estimator's arguments and model, the
// threads, the level and every setting in force -- so it confirms that one
// call of that one cell text in this session and nothing else: an edited cell,
// another argument value, a changed setting or a restarted session all give a
// new token.  The token is not a setting and lets nothing else through: the
// served registry (read_only tags) and the static scan are unchanged.
// Like ServeInterrupted it is not a ConsoleError, so no catch can swallow it.
// ---------------------------------------------------------------------------
constexpr std::size_t kMaxConfirmTokens = 16;
constexpr std::size_t kMaxTokenChars = 64;
constexpr int kHeartbeatS = 10;

struct ServeConfirmRequired {
    ConfirmQuery q;
    std::string token;
};

std::uint64_t g_confirm_nonce = 0;              // per process (= per session)
const std::string* g_cell_src = nullptr;        // the running cell's text (null between cells)
std::vector<std::string> g_cell_confirm;        // the running cell's `confirm` tokens
std::string g_cell_id_json;                     // the running cell's id, as JSON
std::function<void()> g_cell_flush;             // flushes the running cell's pending `out` text (evaluator thread only)
std::function<std::string()> g_settings_text;   // every setting in force, one line

// The heartbeat of a confirmed run (design Sec. 2.5): armed by NoteConfirmedRun, disarmed at the cell's end.
struct Heartbeat {
    std::mutex m;
    bool active = false;
    std::string id_json, label;
    cost::CostEstimate e;
    std::chrono::steady_clock::time_point t0, last;
};
Heartbeat g_hb;

void LogErr(const std::string& line) {
    const std::string s = line + "\n";
    std::size_t off = 0;
    while (off < s.size()) {
        const ssize_t n = ::write(2, s.data() + off, s.size() - off);
        if (n < 0) {
            if (errno == EINTR) continue;
            return;
        }
        off += static_cast<std::size_t>(n);
    }
}

void WriteAllOrDie(const std::string& s) {
    std::size_t off = 0;
    while (off < s.size()) {
        const ssize_t n = ::write(g_proto_fd, s.data() + off, s.size() - off);
        if (n < 0) {
            if (errno == EINTR) continue;
            ::_exit(kFatalExit);   // the gateway went away: nothing left to tell it
        }
        off += static_cast<std::size_t>(n);
    }
}

// Writes one frame; `build` receives the frame's seq and returns its JSON.
template <class F>
void Frame(F build) {
    std::lock_guard<std::mutex> lock(g_out_mutex);
    const std::uint64_t seq = g_seq++;
    WriteAllOrDie(build(seq) + "\n");
}

std::uint64_t ResidentMiB() {
    FILE* f = std::fopen("/proc/self/statm", "r");
    if (f == nullptr) return 0;
    unsigned long long size = 0, resident = 0;
    const int got = std::fscanf(f, "%llu %llu", &size, &resident);
    std::fclose(f);
    if (got != 2) return 0;
    const long page = ::sysconf(_SC_PAGESIZE);
    return (resident * static_cast<unsigned long long>(page > 0 ? page : 4096)) >> 20;
}

[[noreturn]] void Fatal(const std::string& reason, const std::string& detail) {
    // A partially written frame of another thread is not completed (Sec.
    // 18.10): wait for the stream briefly, then cut it either way.
    bool locked = false;
    for (int i = 0; i < 200 && !locked; ++i) {
        locked = g_out_mutex.try_lock();
        if (!locked) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (locked) {
        const std::string f = "{\"kind\":\"fatal\",\"reason\":" + Quote(reason) + ",\"detail\":" + Quote(detail) +
                              ",\"seq\":" + std::to_string(g_seq++) + "}\n";
        std::size_t off = 0;
        while (off < f.size()) {
            const ssize_t n = ::write(g_proto_fd, f.data() + off, f.size() - off);
            if (n <= 0) {
                if (n < 0 && errno == EINTR) continue;
                break;
            }
            off += static_cast<std::size_t>(n);
        }
    }
    LogErr("serve: fatal: " + reason + ": " + detail);
    ::_exit(kFatalExit);
}

void CheckRssCap() {
    if (g_rss_cap_mib == 0) return;
    const std::uint64_t r = ResidentMiB();
    if (r > g_rss_cap_mib)
        Fatal("rss_cap", "resident " + std::to_string(r) + " MiB > cap " + std::to_string(g_rss_cap_mib) + " MiB");
}

// ---------------------------------------------------------------------------
// (NBCONFIRM S4) the confirmation token and the heartbeat.
// ---------------------------------------------------------------------------
std::string NumJson(double v) {
    if (!std::isfinite(v)) return "null";
    char b[64];
    std::snprintf(b, sizeof b, "%.6g", v);
    return b;
}

std::string Hex64(std::uint64_t v) {
    char b[24];
    std::snprintf(b, sizeof b, "0x%016llx", static_cast<unsigned long long>(v));
    return b;
}

std::string ConfirmToken(const ConfirmQuery& q, const std::string& src, const std::string& settings_text) {
    std::uint64_t h = engine::kFnvOffset64;
    using engine::FnvBytes;
    using engine::FnvU64;
    auto feed = [&](const std::string& x) {
        h = FnvU64(h, static_cast<std::uint64_t>(x.size()));   // length-prefixed: no two field lists collide by concatenation
        h = FnvBytes(h, x);
    };
    feed("oc-confirm/1");
    h = FnvU64(h, g_confirm_nonce);
    feed(src);
    h = FnvU64(h, static_cast<std::uint64_t>(q.pos.line));
    h = FnvU64(h, static_cast<std::uint64_t>(q.pos.col));
    feed(q.who);
    feed(q.call);
    feed(q.verb);
    feed(q.estimate.model);
    h = FnvU64(h, static_cast<std::uint64_t>(q.cost_args.size()));
    for (const std::string& a : q.cost_args) feed(a);
    h = FnvU64(h, static_cast<std::uint64_t>(q.estimate.T_eff));
    feed(cost::CostLevelName(q.decision.level));
    feed(settings_text);
    return Hex64(h);
}

std::string ConfirmedJson(const std::string& label, const cost::CostEstimate& e, double elapsed_s, std::uint64_t resident) {
    return "{\"call\":" + Quote(label) + ",\"verb\":" + Quote(e.verb) + ",\"model\":" + Quote(e.model) +
           ",\"wall_s\":" + NumJson(e.wall_s) + ",\"wall_hi\":" + NumJson(e.wall_hi) + ",\"rss_mib\":" + NumJson(e.rss_mib) +
           ",\"rss_hi\":" + NumJson(e.rss_hi) + ",\"elapsed_s\":" + NumJson(elapsed_s) +
           ",\"resident_mib\":" + std::to_string(resident) + "}";
}

// With g_hb.m held.
void HeartbeatFrameLocked(bool start) {
    const auto now = std::chrono::steady_clock::now();
    const double el = std::chrono::duration<double>(now - g_hb.t0).count();
    const std::uint64_t res = ResidentMiB();
    std::string text;
    if (start) {
        text = "running " + g_hb.label + ": confirmed outside its silent envelope; estimated " + DescribeEstimate(g_hb.e);
    } else {
        text = "running " + g_hb.label + ": " + cost::FormatSeconds(el) + " of ~ " + cost::FormatSeconds(g_hb.e.wall_s) +
               " (up to ~ " + cost::FormatSeconds(g_hb.e.wall_hi) + "), resident " +
               cost::FormatMiB(static_cast<double>(res));
    }
    const std::string conf = ConfirmedJson(g_hb.label, g_hb.e, el, res);
    const std::string id_json = g_hb.id_json;
    Frame([&](std::uint64_t seq) {
        return "{\"kind\":\"progress\",\"id\":" + id_json + ",\"seq\":" + std::to_string(seq) + ",\"text\":" + Quote(text) +
               ",\"confirmed\":" + conf + "}";
    });
    g_hb.last = now;
}

// The ConfirmedRunHook of a served session: on the evaluator thread, inside a cell.
void ArmHeartbeat(const std::string& label, const cost::CostEstimate& e) {
    if (g_cell_src == nullptr) return;   // not inside a cell: nothing to report to
    if (g_cell_flush) g_cell_flush();    // program order: the cell's earlier output first
    std::lock_guard<std::mutex> lk(g_hb.m);
    g_hb.active = true;
    g_hb.id_json = g_cell_id_json;
    g_hb.label = label;
    g_hb.e = e;
    g_hb.t0 = std::chrono::steady_clock::now();
    HeartbeatFrameLocked(/*start=*/true);
}

void DisarmHeartbeat() {
    std::lock_guard<std::mutex> lk(g_hb.m);
    g_hb.active = false;
}

void HeartbeatLoop() {
    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(kWatchdogPollMs));
        std::lock_guard<std::mutex> lk(g_hb.m);
        if (g_hb.active && std::chrono::steady_clock::now() - g_hb.last >= std::chrono::seconds(kHeartbeatS))
            HeartbeatFrameLocked(/*start=*/false);
    }
}

// The ConfirmChannel of a served session (Confirm.h): returns when the running cell carries this call's token.
void ServeConfirmChannel(const ConfirmQuery& q) {
    const std::string src = g_cell_src != nullptr ? *g_cell_src : std::string();
    const std::string tok = ConfirmToken(q, src, g_settings_text ? g_settings_text() : std::string());
    for (const std::string& t : g_cell_confirm)
        if (t == tok) return;
    throw ServeConfirmRequired{q, tok};
}

// ---------------------------------------------------------------------------
// The served registry (Sec. 18.8).
// ---------------------------------------------------------------------------
struct ServedView {
    std::set<std::string> unserved_verbs;                       // no tagged overload
    std::set<std::pair<std::string, std::string>> refusers;     // untagged overload, tagged sibling
    std::size_t served_verbs = 0;
    std::size_t served_overloads = 0;
    std::size_t full_overloads = 0;
};

VerbImpl Wrap(VerbImpl impl, std::size_t index) {
    return [impl, index](CallContext& ctx, const std::vector<Value>& a) -> Value {
        if (g_interrupt.load()) throw ServeInterrupted{ctx.pos};
        if (g_called != nullptr && index < g_called->size()) (*g_called)[index].store(true);
        return impl(ctx, a);
    };
}

VerbImpl Refuser(const std::string& verb, const std::string& sig) {
    return [verb, sig](CallContext& ctx, const std::vector<Value>&) -> Value {
        throw ServeForbidden{"'" + sig + "' is not served: this overload is not read-only (read_only = false)",
                             ctx.pos, verb};
    };
}

// Restricts the process registry to the served view, IN PLACE, so that every
// TypeInfo keeps its address (the extension layer keys its per-registry type
// tables by TypeInfo*, and `==` on objects goes through the TypeInfo): a
// tagged overload is wrapped (interrupt + coverage), an untagged overload of a
// verb with a tagged sibling becomes a refuser, a verb with no tagged overload
// disappears.  `raw` receives the ORIGINAL `hash` and `coeffs` impls, which
// PrintJson calls (rendering must neither count as a call nor be interrupted).
void BuildServed(Registry& reg, Registry& raw, ServedView& view) {
    for (const std::string& name : reg.verb_names()) {
        const VerbEntry* e = reg.find_verb(name);
        bool any = false;
        for (const Overload& o : e->overloads) any = any || serve_tags::IsReadOnly(*e, o);
        if (!any) view.unserved_verbs.insert(name);
        else ++view.served_verbs;
        if (name == "hash" || name == "coeffs")
            for (const Overload& o : e->overloads) raw.register_verb(name, o.sig, o.impl, o.doc, e->group);
    }
    bind::ForbidReportWrite();   // the pathless `report` is served: Report.write must not be
    std::size_t index = 0;
    reg.transform_overloads([&](const VerbEntry& e, Overload& o) {
        const std::size_t i = index++;
        ++view.full_overloads;
        if (view.unserved_verbs.count(e.name)) return false;
        const std::string sig = o.sig.to_string(e.name);
        if (serve_tags::LookupTag(e.name, sig) == serve_tags::TagState::ReadOnly) {
            o.impl = Wrap(o.impl, i);
            ++view.served_overloads;
        } else {
            o.impl = Refuser(e.name, sig);
            o.doc = "[not served: read_only = false] " + o.doc;
            view.refusers.insert({e.name, sig});
        }
        return true;
    });
}

// ---------------------------------------------------------------------------
// Static and per-statement checks.
// ---------------------------------------------------------------------------
struct Offence {
    bool found = false;
    std::string message;
    SourcePos pos;
};

void ScanExpr(const ExprPtr& e, const ServedView& view, Offence& off) {
    if (!e || off.found) return;
    if (e->kind == ExprKind::Call && view.unserved_verbs.count(e->text)) {
        off.found = true;
        off.pos = e->pos;
        off.message = e->text == "exit"
                          ? std::string("exit(...) is not available in a served session (end it with `close`)")
                          : "verb '" + e->text + "' is not served: no overload of it is read-only";
        return;
    }
    ScanExpr(e->lhs, view, off);
    ScanExpr(e->rhs, view, off);
    for (const ExprPtr& x : e->items) ScanExpr(x, view, off);
    for (const Argument& a : e->args) ScanExpr(a.value, view, off);
}

void ScanStmt(const StmtPtr& s, const ServedView& view, Offence& off) {
    if (!s || off.found) return;
    if (s->kind == StmtKind::Include) {
        off.found = true;
        off.pos = s->pos;
        off.message = "include is not available in a served session (it reads a file named by the request)";
        return;
    }
    if (s->kind == StmtKind::Quit) {
        off.found = true;
        off.pos = s->pos;
        off.message = "quit is not available in a served session (end it with `close`)";
        return;
    }
    ScanExpr(s->value, view, off);
    for (const StmtPtr& b : s->body) ScanStmt(b, view, off);
    for (const StmtPtr& b : s->orelse) ScanStmt(b, view, off);
}

// A `set` of an existing key that is not settable in a served session.
void CheckSets(const StmtPtr& s, const Settings& settings) {
    if (!s) return;
    if (s->kind == StmtKind::Set && settings.find_spec(s->name) != nullptr && !serve_tags::IsServeSettable(s->name)) {
        throw ServeForbidden{"set " + s->name +
                                 ": not settable in a served session (process-global or presentation key)",
                             s->pos, "set"};
    }
    for (const StmtPtr& b : s->body) CheckSets(b, settings);
    for (const StmtPtr& b : s->orelse) CheckSets(b, settings);
}

// ---------------------------------------------------------------------------
// Frames.
// ---------------------------------------------------------------------------
std::string PosJson(const SourcePos& p) {
    if (!p.known()) return "null";
    return "{\"line\":" + std::to_string(p.line) + ",\"col\":" + std::to_string(p.col) + "}";
}

struct ErrorInfo {
    std::string status;   // ok / error / interrupted / rejected
    int exit = 0;
    std::string kind;
    std::string message;
    std::string detail;
    SourcePos at;
    SourcePos created_at;
    std::string verb;

    std::string json() const {
        if (status == "ok") return "null";
        return "{\"kind\":" + Quote(kind) + ",\"message\":" + Quote(message) + ",\"detail\":" + Quote(detail) +
               ",\"at\":" + PosJson(at) + ",\"created_at\":" + PosJson(created_at) +
               ",\"verb\":" + (verb.empty() ? std::string("null") : Quote(verb)) + "}";
    }
};

ErrorInfo Rejected(const std::string& kind, const std::string& message, const SourcePos& at = SourcePos()) {
    ErrorInfo e;
    e.status = "rejected";
    e.exit = 1;
    e.kind = kind;
    e.message = message;
    e.at = at;
    return e;
}

std::string IdJson(const std::string* id) { return id == nullptr ? std::string("null") : Quote(*id); }

void Done(const std::string* id, const ErrorInfo& e, std::size_t statements,
          std::chrono::steady_clock::time_point t0, const std::string& extra = std::string()) {
    const long long ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    const std::uint64_t rss = ResidentMiB();
    const long long peak = static_cast<long long>(oc_util::PeakRssMiB());
    Frame([&](std::uint64_t seq) {
        return "{\"kind\":\"done\",\"id\":" + IdJson(id) + ",\"seq\":" + std::to_string(seq) +
               ",\"status\":" + Quote(e.status) + ",\"exit\":" + std::to_string(e.exit) + ",\"error\":" + e.json() + extra +
               ",\"statements\":" + std::to_string(statements) + ",\"perf\":{\"elapsed_ms\":" + std::to_string(ms) +
               ",\"rss_mib\":" + std::to_string(rss) + ",\"peak_rss_mib\":" + std::to_string(peak) + "}}";
    });
}

// The fixed part of an `out` frame, for the cap arithmetic.
std::size_t OutOverhead(const std::string& id_json) { return 96 + id_json.size(); }

std::string EscapedChar(char c) {
    const std::string one(1, c);
    return JsonEscape(one);
}

void EmitOutPiece(const std::string& id_json, const std::string& stream, const std::string& text, bool cont) {
    Frame([&](std::uint64_t seq) {
        return "{\"kind\":\"out\",\"id\":" + id_json + ",\"seq\":" + std::to_string(seq) + ",\"stream\":" +
               Quote(stream) + ",\"text\":" + Quote(text) + (cont ? ",\"cont\":true" : "") + "}";
    });
}

// Emits `text` (complete lines) as `out` frames of at most g_max_reply bytes:
// whole lines are packed; a line too long for one frame is split, `cont:true`
// on every piece but its last (Sec. 18.4).
void EmitOut(const std::string& id_json, const std::string& stream, const std::string& text) {
    if (text.empty()) return;
    const std::size_t budget = g_max_reply > OutOverhead(id_json) + 16 ? g_max_reply - OutOverhead(id_json) - 16 : 64;
    std::string pending;
    std::size_t pending_escaped = 0;
    std::size_t i = 0;
    while (i < text.size()) {
        std::size_t nl = text.find('\n', i);
        const std::size_t end = (nl == std::string::npos) ? text.size() : nl + 1;
        const std::string line = text.substr(i, end - i);
        i = end;
        const std::size_t esc = JsonEscape(line).size();
        if (pending_escaped + esc <= budget) {
            pending += line;
            pending_escaped += esc;
            continue;
        }
        if (!pending.empty()) {
            EmitOutPiece(id_json, stream, pending, false);
            pending.clear();
            pending_escaped = 0;
        }
        if (esc <= budget) {
            pending = line;
            pending_escaped = esc;
            continue;
        }
        // One line longer than a frame: split it at UTF-8 boundaries.
        std::string piece;
        std::size_t piece_esc = 0;
        for (std::size_t k = 0; k < line.size(); ++k) {
            const unsigned char c = static_cast<unsigned char>(line[k]);
            const std::size_t ce = EscapedChar(static_cast<char>(c)).size();
            const bool boundary = (c & 0xC0) != 0x80;
            if (boundary && piece_esc + ce + 8 > budget && !piece.empty()) {
                EmitOutPiece(id_json, stream, piece, true);
                piece.clear();
                piece_esc = 0;
            }
            piece += static_cast<char>(c);
            piece_esc += ce;
        }
        if (!piece.empty()) {
            const bool ends_line = !piece.empty() && piece.back() == '\n';
            EmitOutPiece(id_json, stream, piece, !ends_line);
        }
    }
    if (!pending.empty()) EmitOutPiece(id_json, stream, pending, false);
}

// The session's output stream for one cell: complete lines are classified as
// they arrive -- a `#~` line is a `progress` frame at once (after the pending
// `out` text, so program order holds); other lines accumulate and are flushed
// as `out` frames at every statement end (and when large).
class CellSink : public std::streambuf {
public:
    explicit CellSink(std::string id_json) : id_json_(std::move(id_json)) {}

    void flush_out() {
        EmitOut(id_json_, "stdout", pending_);
        pending_.clear();
    }
    // End of cell: a partial last line is flushed as it is.
    void finish() {
        if (!line_.empty()) {
            pending_ += line_;
            line_.clear();
        }
        flush_out();
    }

protected:
    int overflow(int c) override {
        if (c == traits_type::eof()) return traits_type::not_eof(c);
        put(static_cast<char>(c));
        return c;
    }
    std::streamsize xsputn(const char* s, std::streamsize n) override {
        for (std::streamsize k = 0; k < n; ++k) put(s[k]);
        return n;
    }

private:
    void put(char c) {
        line_ += c;
        if (c != '\n') return;
        if (line_.rfind("#~", 0) == 0) {
            flush_out();
            std::string text = line_.substr(2, line_.size() - 3);   // drop "#~" and "\n"
            if (!text.empty() && text[0] == ' ') text.erase(0, 1);
            Frame([&](std::uint64_t seq) {
                return "{\"kind\":\"progress\",\"id\":" + id_json_ + ",\"seq\":" + std::to_string(seq) +
                       ",\"text\":" + Quote(text) + "}";
            });
        } else {
            pending_ += line_;
            if (pending_.size() >= g_max_reply / 2) flush_out();
        }
        line_.clear();
    }

    std::string id_json_;
    std::string line_;
    std::string pending_;
};

// ---------------------------------------------------------------------------
// The session.
// ---------------------------------------------------------------------------
struct Server {
    Registry& served;   // the process registry, restricted in place to the served view
    Registry raw;       // the original `hash` / `coeffs` impls, for PrintJson
    ServedView view;
    std::unique_ptr<Session> session;
    std::ostringstream idle_out;   // the session's output between cells (nothing writes there)

    explicit Server(Registry& r) : served(r) {}

    PrintJsonContext json_ctx() {
        PrintJsonContext c;
        c.registry = &raw;
        c.session = session.get();
        c.max_entries = session->print_max_entries();
        return c;
    }

    void value_frame(const std::string& id_json, std::size_t index, const SourcePos& pos, const Value& v) {
        const PrintJsonContext ctx = json_ctx();
        auto build = [&](const std::string& env, std::uint64_t seq) {
            return "{\"kind\":\"value\",\"id\":" + id_json + ",\"seq\":" + std::to_string(seq) +
                   ",\"stmt\":" + std::to_string(index) + ",\"line\":" + std::to_string(pos.line) +
                   ",\"col\":" + std::to_string(pos.col) + ",\"value\":" + env + "}";
        };
        std::string env = PrintJson(v, ctx);
        const std::size_t overhead = 128 + id_json.size();
        if (env.size() + overhead > g_max_reply) env = PrintJsonTruncated(v, ctx, g_max_reply / 8);
        if (env.size() + overhead > g_max_reply) env = PrintJsonTruncated(v, ctx, 0);
        Frame([&](std::uint64_t seq) { return build(env, seq); });
    }

    // ----- eval ---------------------------------------------------------------
    void eval(const std::string& id, const Json& req) {
        const auto t0 = std::chrono::steady_clock::now();
        const std::string id_json = Quote(id);
        const Json* src = req.get("src");
        if (src == nullptr || !src->is_string()) {
            Done(&id, Rejected("protocol", "eval: `src` must be a string"), 0, t0);
            return;
        }
        if (src->text.size() > kMaxRequestBytes) {
            Done(&id, Rejected("protocol", "eval: `src` exceeds 8 MiB"), 0, t0);
            return;
        }
        std::vector<std::pair<std::string, Value>> args;
        if (const Json* a = req.get("args")) {
            if (!a->is_object()) {
                Done(&id, Rejected("protocol", "eval: `args` must be an object"), 0, t0);
                return;
            }
            for (const auto& kv : a->members) {
                const Json& x = kv.second;
                std::int64_t iv = 0;
                if (x.is_bool()) args.emplace_back(kv.first, Value::from_bool(x.b));
                else if (x.is_number() && x.as_int64(iv)) args.emplace_back(kv.first, Value::from_int(iv));
                else if (x.is_string()) args.emplace_back(kv.first, ParseCliLiteral(x.text));
                else {
                    Done(&id, Rejected("protocol", "eval: args." + kv.first + " must be a Bool, an integer or a String"),
                         0, t0);
                    return;
                }
            }
        }

        // (NBCONFIRM S4) `confirm`: the tokens of earlier confirm_required replies (Sec. 18.4).
        std::vector<std::string> confirm;
        if (const Json* c = req.get("confirm")) {
            bool ok = c->is_array() && c->items.size() <= kMaxConfirmTokens;
            if (ok)
                for (const Json& t : c->items) {
                    ok = ok && t.is_string() && !t.text.empty() && t.text.size() <= kMaxTokenChars;
                    if (ok)
                        for (char ch : t.text) ok = ok && ch > 0x20 && ch < 0x7f;
                    if (ok) confirm.push_back(t.text);
                }
            if (!ok) {
                Done(&id, Rejected("protocol", "eval: `confirm` must be a list of at most " + std::to_string(kMaxConfirmTokens) +
                                                   " tokens (strings of 1-" + std::to_string(kMaxTokenChars) +
                                                   " printable ASCII characters)"),
                     0, t0);
                return;
            }
        }

        // The whole cell is parsed first (Sec. 6): a parse error runs nothing.
        Program program;
        try {
            program = Parse(src->text, "<cell>");
        } catch (const ConsoleError& e) {
            ErrorInfo info = FromConsoleError(e);
            Done(&id, info, 0, t0);
            return;
        }
        // Static refusal (Sec. 18.3 / 18.7): include, quit, exit, unserved verbs.
        Offence off;
        for (const StmtPtr& s : program.statements) ScanStmt(s, view, off);
        if (off.found) {
            Done(&id, Rejected("forbidden", "<cell>:" + std::to_string(off.pos.line) + ":" +
                                                std::to_string(off.pos.col) + ": " + off.message,
                               off.pos),
                 0, t0);
            return;
        }

        for (const auto& kv : args) session->set_arg(kv.first, kv.second);

        CellSink sink(id_json);
        std::ostream cell_out(&sink);
        session->set_output(cell_out);
        std::ostringstream chatter;
        std::streambuf* old_cout = std::cout.rdbuf(chatter.rdbuf());
        std::streambuf* old_cerr = std::cerr.rdbuf(chatter.rdbuf());
        auto flush_all = [&]() {
            cell_out.flush();
            const std::string c = chatter.str();
            if (!c.empty()) {
                chatter.str(std::string());
                chatter.clear();
                sink.flush_out();
                EmitOut(id_json, "log", c.back() == '\n' ? c : c + "\n");
            }
            sink.flush_out();
        };

        g_interrupt.store(false);
        console::ClearCancel();
        g_running.store(true);
        // (NBCONFIRM S4) the cell the confirmation channel and the heartbeat refer to
        g_cell_src = &src->text;
        g_cell_confirm = confirm;
        g_cell_id_json = id_json;
        g_cell_flush = [&]() { flush_all(); };
        std::string confirm_extra;
        std::size_t executed = 0;
        ErrorInfo info;
        info.status = "ok";
        Evaluator ev(*session);
        // (slot-24 NB-RENDER R3) `display(x, ...)` in this cell: the canonical echo and a value
        // frame per argument, exactly what a bare expression statement `x` sends (Sec. 18.4).
        std::size_t current_stmt = 0;
        const bind::DisplaySinkScope display_sink([&](const Value& v, const SourcePos& pos) {
            const console::NoCancelScope rendering;   // (NBCONFIRM S5) rendering is never interrupted
            session->print_line(PrintCanonical(v, session->print_max_entries()));
            flush_all();
            value_frame(id_json, current_stmt, pos, v);
        });
        // No file system in a served cell: include is refused statically, and
        // the loader refuses again should one ever be reached.
        ev.set_file_loader([](const std::string&, std::string&) { return false; });
        try {
            const console::CancelScope armed;   // (NBCONFIRM S5) the cell's cancellation points are live on this thread
            for (std::size_t i = 0; i < program.statements.size(); ++i) {
                const StmtPtr& s = program.statements[i];
                current_stmt = i;
                if (g_interrupt.load() || g_term) throw ServeInterrupted{s->pos};
                CheckRssCap();
                CheckSets(s, session->settings());
                if (s->kind == StmtKind::ExprStmt) {
                    // Evaluator::exec_expr_stmt, verbatim, plus the value frame.
                    Value v = ev.eval(s->value);
                    if (!s->suppress_print && !v.is_nil()) {
                        const console::NoCancelScope rendering;   // (NBCONFIRM S5)
                        session->print_line(PrintCanonical(v, session->print_max_entries()));
                        flush_all();
                        value_frame(id_json, i, s->pos, v);
                    }
                } else {
                    ev.exec(s);
                }
                ++executed;
                flush_all();
            }
        } catch (const ConsoleError& e) {
            info = FromConsoleError(e);
        } catch (const ServeInterrupted& e) {
            info.status = "interrupted";
            info.exit = 3;
            info.kind = "interrupted";
            info.message = (e.pos.known() ? e.pos.to_string() + ": " : std::string()) + "interrupted";
            info.at = e.pos;
        } catch (const console::Interrupted& e) {
            // (NBCONFIRM S5) stopped INSIDE a statement, at a cancellation point: the same done (Sec. 18.6); the statement
            // made no binding of its own (earlier statements keep theirs), a lazy handle it was computing stays declared
            info.status = "interrupted";
            info.exit = 3;
            info.kind = "interrupted";
            info.message = (e.pos.known() ? e.pos.to_string() + ": " : std::string()) + "interrupted";
            info.at = e.pos;
        } catch (const ServeForbidden& e) {
            info.status = "error";
            info.exit = 3;
            info.kind = "forbidden_overload";
            info.message = (e.pos.known() ? e.pos.to_string() + ": " : std::string()) + "error: " + e.message;
            info.at = e.pos;
            info.verb = e.verb;
        } catch (const ServeConfirmRequired& e) {
            info = ConfirmRequiredInfo(e, confirm_extra);
        } catch (const ExitSignal&) {
            info = Rejected("forbidden", "exit is not available in a served session");
        } catch (const QuitSignal&) {
            info = Rejected("forbidden", "quit is not available in a served session");
        } catch (const std::bad_alloc&) {
            info.status = "error";
            info.exit = 3;
            info.kind = "runtime";
            info.message = "<cell>: error: out of memory (std::bad_alloc)";
        } catch (const std::exception& e) {
            info.status = "error";
            info.exit = 3;
            info.kind = "runtime";
            info.message = std::string("<cell>: error: ") + e.what();
        } catch (...) {
            std::cout.rdbuf(old_cout);
            std::cerr.rdbuf(old_cerr);
            Fatal("internal", "an exception of unknown type escaped the dispatcher");
        }
        DisarmHeartbeat();   // (S4) no heartbeat of this cell after its done
        g_cell_flush = nullptr;
        g_cell_src = nullptr;
        g_cell_confirm.clear();
        flush_all();
        sink.finish();
        std::cout.rdbuf(old_cout);
        std::cerr.rdbuf(old_cerr);
        session->set_output(idle_out);
        g_running.store(false);
        g_interrupt.store(false);
        console::ClearCancel();
        CheckRssCap();
        Done(&id, info, executed, t0, confirm_extra);
    }

    // (NBCONFIRM S4) done{status:"confirm_required", exit:3, error:{kind:"confirm_required", ...}, confirm:{...}}.
    static ErrorInfo ConfirmRequiredInfo(const ServeConfirmRequired& c, std::string& extra) {
        const ConfirmQuery& q = c.q;
        const cost::CostEstimate& e = q.estimate;
        const cost::CostDecision& d = q.decision;
        const bool strong = d.level == cost::CostLevel::Strong;
        ErrorInfo info;
        info.status = "confirm_required";
        info.exit = 3;
        info.kind = "confirm_required";
        info.message = (q.pos.known() ? q.pos.to_string() + ": " : std::string()) + "confirm: " + q.who +
                       " is outside its silent envelope (" + q.pending + "); it runs only when the cell is re-sent with " +
                       "its confirmation token";
        info.detail = "estimated " + DescribeEstimate(e);
        if (strong) {
            info.detail += "\nSTRONG WARNING:";
            for (std::size_t i = 0; i < d.reasons.size(); ++i) info.detail += (i ? "; " : " ") + d.reasons[i];
        }
        for (const std::string& n : q.notes) info.detail += "\nnote: " + n;
        info.at = q.pos;
        info.verb = q.verb;
        auto list = [](const std::vector<std::string>& v) {
            std::string o = "[";
            for (std::size_t k = 0; k < v.size(); ++k) o += (k ? "," : "") + Quote(v[k]);
            return o + "]";
        };
        extra = ",\"confirm\":{\"token\":" + Quote(c.token) + ",\"level\":" + Quote(cost::CostLevelName(d.level)) +
                ",\"verb\":" + Quote(q.verb) + ",\"who\":" + Quote(q.who) + ",\"call\":" + Quote(q.call) +
                ",\"site\":" + PosJson(q.pos) + ",\"bound\":" + Quote(q.pending) + ",\"wall_s\":" + NumJson(e.wall_s) +
                ",\"wall_hi\":" + NumJson(e.wall_hi) + ",\"rss_mib\":" + NumJson(e.rss_mib) + ",\"rss_hi\":" + NumJson(e.rss_hi) +
                ",\"mem_limit_mib\":" + (d.mem_limit_mib > 0 ? NumJson(d.mem_limit_mib) : std::string("null")) +
                ",\"threads\":" + std::to_string(e.T_eff) + ",\"extrapolated\":" + (e.extrapolated ? "true" : "false") +
                ",\"x_factor\":" + NumJson(e.x_factor) + ",\"low_confidence\":" + (e.low_confidence.empty() ? "false" : "true") +
                ",\"model\":" + Quote(e.model) + ",\"calibration\":" + Quote(e.calibration) +
                ",\"reasons\":" + list(d.reasons) + ",\"notes\":" + list(q.notes) +
                ",\"kill_wall_s\":" + (d.kill_s > 0 ? NumJson(d.kill_s) : std::string("null")) +
                ",\"text\":" + Quote(DescribeEstimate(e)) + "}";
        return info;
    }

    static ErrorInfo FromConsoleError(const ConsoleError& e) {
        ErrorInfo info;
        info.status = "error";
        const std::string& k = e.kind();
        if (k == "parse error") { info.kind = "parse"; info.exit = 2; }
        else if (k == "type error") { info.kind = "type"; info.exit = 3; }
        else if (k == "assertion failed") { info.kind = "assertion"; info.exit = 4; }
        else if (k == kNotImplementedKind) { info.kind = "not_implemented"; info.exit = 5; }
        else { info.kind = "runtime"; info.exit = 3; }
        const std::string& f = e.formatted();
        const std::size_t nl = f.find('\n');
        info.message = f.substr(0, nl);
        if (nl != std::string::npos) {
            std::string d = f.substr(nl + 1);
            // The notes are indented by two spaces in the formatted text.
            std::string out;
            std::istringstream in(d);
            std::string line;
            bool first = true;
            while (std::getline(in, line)) {
                if (line.rfind("  ", 0) == 0) line.erase(0, 2);
                if (!first) out += '\n';
                first = false;
                out += line;
            }
            info.detail = out;
        }
        info.at = e.pos();
        for (const std::string& note : e.notes()) {
            const std::string tag = "created at <cell>:";
            const std::size_t at = note.find(tag);
            if (at == std::string::npos) continue;
            int line = 0, col = 0;
            if (std::sscanf(note.c_str() + at + tag.size(), "%d:%d", &line, &col) >= 1) {
                info.created_at = SourcePos("<cell>", line, col);
            }
            break;
        }
        return info;
    }

    // ----- complete / help / describe ---------------------------------------
    void complete(const std::string& id, const Json& req) {
        const auto t0 = std::chrono::steady_clock::now();
        const Json* p = req.get("prefix");
        if (p == nullptr || !p->is_string()) {
            Done(&id, Rejected("protocol", "complete: `prefix` must be a string"), 0, t0);
            return;
        }
        const std::string& prefix = p->text;
        std::vector<std::string> items;
        auto item = [&](const std::string& text, const char* kind, const std::string& sig) {
            std::string o = "{\"text\":" + Quote(text) + ",\"kind\":" + Quote(kind);
            if (!sig.empty()) o += ",\"sig\":" + Quote(sig);
            items.push_back(o + "}");
        };
        auto starts = [&](const std::string& s, const std::string& pre) { return s.compare(0, pre.size(), pre) == 0; };
        const std::size_t dot = prefix.rfind('.');
        if (dot != std::string::npos) {
            const std::string base = prefix.substr(0, dot);
            const std::string part = prefix.substr(dot + 1);
            const Value* v = session->find_var(base);
            if (v != nullptr && v->is_object() && v->object_value() && v->object_value()->type() != nullptr) {
                const TypeInfo* t = v->object_value()->type();
                for (const TypeInfo::Member& m : t->members)
                    if (starts(m.name, part)) item(base + "." + m.name, "member", std::string());
                for (const TypeInfo::Method& m : t->methods) {
                    if (!starts(m.name, part)) continue;
                    std::string sig = m.name + "(";
                    for (std::size_t k = 0; k < m.params.size(); ++k) sig += (k ? ", " : "") + m.params[k];
                    item(base + "." + m.name, "member", sig + ")");
                }
            } else if (v != nullptr && v->is_record()) {
                for (const auto& kv : v->record_value().entries())
                    if (starts(kv.first, part)) item(base + "." + kv.first, "member", std::string());
            }
        } else {
            for (const std::string& name : served.verb_names()) {
                if (!starts(name, prefix)) continue;
                const VerbEntry* e = served.find_verb(name);
                std::string sig;
                for (const Overload& o : e->overloads) {
                    const std::string s = o.sig.to_string(name);
                    if (!view.refusers.count({name, s})) { sig = s; break; }
                }
                item(name, "verb", sig);
            }
            for (const std::string& t : served.type_names())
                if (starts(t, prefix)) item(t, "type", std::string());
            for (const std::string& k : serve_tags::ServeSettableKeys())
                if (starts(k, prefix) && session->settings().find_spec(k) != nullptr) item(k, "setting", std::string());
            for (const auto& kv : session->vars())
                if (starts(kv.first, prefix)) item(kv.first, "variable", std::string());
        }
        Frame([&](std::uint64_t seq) {
            std::string o = "{\"kind\":\"completions\",\"id\":" + Quote(id) + ",\"seq\":" + std::to_string(seq) +
                            ",\"items\":[";
            for (std::size_t k = 0; k < items.size(); ++k) o += (k ? "," : "") + items[k];
            return o + "]}";
        });
        ErrorInfo ok;
        ok.status = "ok";
        Done(&id, ok, 0, t0);
    }

    std::string verb_json(const std::string& name) {
        const VerbEntry* e = served.find_verb(name);
        std::string o = "{\"name\":" + Quote(name) + ",\"group\":" + Quote(e->group) + ",\"overloads\":[";
        bool first = true;
        for (const Overload& ov : e->overloads) {
            const std::string sig = ov.sig.to_string(name);
            if (view.refusers.count({name, sig})) continue;
            if (!first) o += ',';
            first = false;
            o += "{\"sig\":" + Quote(sig) + ",\"doc\":" + Quote(ov.doc) + ",\"read_only\":true}";
        }
        return o + "]}";
    }

    std::string type_json(const std::string& name, bool full_members) {
        const TypeInfo* t = served.find_type(name);
        std::string o = "{\"name\":" + Quote(name) + ",\"doc\":" + Quote(t->doc);
        if (full_members) {
            o += ",\"members\":[";
            bool first = true;
            for (const TypeInfo::Member& m : t->members) {
                if (!first) o += ',';
                first = false;
                o += "{\"name\":" + Quote(m.name) + ",\"computed\":" + (m.computed ? "true" : "false") +
                     ",\"doc\":" + Quote(m.doc) + "}";
            }
            o += "],\"methods\":[";
            first = true;
            for (const TypeInfo::Method& m : t->methods) {
                if (!first) o += ',';
                first = false;
                std::string sig = m.name + "(";
                for (std::size_t k = 0; k < m.params.size(); ++k) sig += (k ? ", " : "") + m.params[k];
                o += "{\"sig\":" + Quote(sig + ")") + ",\"computed\":" + (m.computed ? "true" : "false") +
                     ",\"doc\":" + Quote(m.doc) + "}";
            }
            o += "]";
        }
        return o + "}";
    }

    std::string setting_json(const std::string& name) {
        const SettingSpec* s = session->settings().find_spec(name);
        return "{\"name\":" + Quote(name) + ",\"doc\":" + Quote(s->doc) +
               ",\"settable\":" + (serve_tags::IsServeSettable(name) ? "true" : "false") + "}";
    }

    void help(const std::string& id, const Json& req) {
        const auto t0 = std::chrono::steady_clock::now();
        const Json* p = req.get("topic");
        if (p == nullptr || !p->is_string()) {
            Done(&id, Rejected("protocol", "help: `topic` must be a string"), 0, t0);
            return;
        }
        const std::string& topic = p->text;
        std::vector<std::string> verbs, types, settings;
        if (topic.empty()) {
            for (const std::string& n : served.verb_names()) verbs.push_back(verb_json(n));
            for (const std::string& n : served.type_names()) types.push_back(type_json(n, false));
            for (const std::string& n : session->settings().names()) settings.push_back(setting_json(n));
        } else {
            if (served.find_verb(topic) != nullptr) verbs.push_back(verb_json(topic));
            if (served.find_type(topic) != nullptr) types.push_back(type_json(topic, true));
            if (session->settings().find_spec(topic) != nullptr) settings.push_back(setting_json(topic));
        }
        auto arr = [](const std::vector<std::string>& v) {
            std::string o = "[";
            for (std::size_t k = 0; k < v.size(); ++k) o += (k ? "," : "") + v[k];
            return o + "]";
        };
        Frame([&](std::uint64_t seq) {
            return "{\"kind\":\"help\",\"id\":" + Quote(id) + ",\"seq\":" + std::to_string(seq) +
                   ",\"topic\":" + Quote(topic) + ",\"verbs\":" + arr(verbs) + ",\"types\":" + arr(types) +
                   ",\"settings\":" + arr(settings) + "}";
        });
        ErrorInfo ok;
        ok.status = "ok";
        Done(&id, ok, 0, t0);
    }

    void describe(const std::string& id, const Json& req) {
        const auto t0 = std::chrono::steady_clock::now();
        const Json* p = req.get("name");
        if (p == nullptr || !p->is_string()) {
            Done(&id, Rejected("protocol", "describe: `name` must be a string"), 0, t0);
            return;
        }
        const Value* v = session->find_var(p->text);
        if (v == nullptr) {
            ErrorInfo e;
            e.status = "error";
            e.exit = 3;
            e.kind = "runtime";
            e.message = "describe: unknown name '" + p->text + "'";
            Done(&id, e, 0, t0);
            return;
        }
        const PrintJsonContext ctx = json_ctx();
        std::string env = PrintJson(*v, ctx);
        if (env.size() + 512 > g_max_reply) env = PrintJsonTruncated(*v, ctx, g_max_reply / 8);
        std::string state = "computed";
        std::string created = "null";
        std::string members = "[";
        if (v->is_object() && v->object_value()) {
            const ObjectPtr& o = v->object_value();
            state = o->state() == ObjectState::Declared ? "declared"
                    : o->state() == ObjectState::Failed ? "failed"
                                                         : "computed";
            created = PosJson(o->creation_site());
            if (o->type() != nullptr) {
                bool first = true;
                for (const TypeInfo::Member& m : o->type()->members) {
                    if (!first) members += ',';
                    first = false;
                    members += "{\"name\":" + Quote(m.name) + ",\"computed\":" + (m.computed ? "true" : "false") + "}";
                }
            }
        }
        members += "]";
        Frame([&](std::uint64_t seq) {
            return "{\"kind\":\"described\",\"id\":" + Quote(id) + ",\"seq\":" + std::to_string(seq) +
                   ",\"value\":" + env + ",\"state\":" + Quote(state) + ",\"created_at\":" + created +
                   ",\"members\":" + members + "}";
        });
        ErrorInfo ok;
        ok.status = "ok";
        Done(&id, ok, 0, t0);
    }
};

// ---------------------------------------------------------------------------
// The request queue and the reader thread (Sec. 18.2).
// ---------------------------------------------------------------------------
struct Item {
    bool eof = false;
    bool malformed = false;
    std::string error;
    Json req;
};

std::mutex g_q_mutex;
std::condition_variable g_q_cv;
std::deque<Item> g_queue;

bool ValidId(const Json* id) {
    if (id == nullptr || !id->is_string()) return false;
    const std::string& s = id->text;
    if (s.empty() || s.size() > 64) return false;
    for (char c : s)
        if (c < 0x20 || c > 0x7e) return false;
    return true;
}

void Push(Item it) {
    {
        std::lock_guard<std::mutex> lock(g_q_mutex);
        if (!it.eof && g_queue.size() >= kQueueLimit) {
            const Json* id = it.malformed ? nullptr : it.req.get("id");
            const std::string idv = ValidId(id) ? id->text : std::string();
            const auto t0 = std::chrono::steady_clock::now();
            Done(ValidId(id) ? &idv : nullptr,
                 Rejected("busy", "the request queue is full (" + std::to_string(kQueueLimit) + " requests)"), 0, t0);
            return;
        }
        g_queue.push_back(std::move(it));
    }
    g_q_cv.notify_one();
}

void HandleLine(std::string line) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    bool blank = true;
    for (char c : line) blank = blank && (c == ' ' || c == '\t');
    if (blank) return;
    Item it;
    if (!serve_json::ParseJson(line, it.req, it.error)) {
        it.malformed = true;
        Push(std::move(it));
        return;
    }
    if (it.req.is_object()) {
        const Json* op = it.req.get("op");
        if (op != nullptr && op->is_string() && op->text == "interrupt") {
            // (NBCONFIRM S5) also the evaluator's request flag: the running cell stops at its next cancellation point
            // (Console/Core/Cancel.h), inside a long verb too, not only at the next statement
            if (g_running.load()) {
                g_interrupt.store(true);
                console::RequestCancel();
            }
            return;   // no frame (Sec. 18.6)
        }
    }
    Push(std::move(it));
}

void ReaderLoop() {
    std::string line;
    std::vector<char> buf(65536);
    while (true) {
        const ssize_t n = ::read(0, buf.data(), buf.size());
        if (n < 0) {
            if (errno == EINTR) continue;
            Fatal("stdin", std::string("read error: ") + std::strerror(errno));
        }
        if (n == 0) {
            if (!line.empty()) HandleLine(line);
            Item eof;
            eof.eof = true;
            Push(std::move(eof));
            return;
        }
        for (ssize_t k = 0; k < n; ++k) {
            const char c = buf[static_cast<std::size_t>(k)];
            if (c == '\n') {
                HandleLine(line);
                line.clear();
                continue;
            }
            line += c;
            if (line.size() > kMaxRequestBytes)
                Fatal("protocol_overflow", "a request line exceeds max_request_bytes (" +
                                               std::to_string(kMaxRequestBytes) + ")");
        }
    }
}

void WatchdogLoop() {
    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(kWatchdogPollMs));
        CheckRssCap();
    }
}

extern "C" void OnSigterm(int) {
    g_term = 1;
    g_interrupt.store(true);
    console::RequestCancel();   // (NBCONFIRM S5) async-signal-safe: one lock-free store
}

[[noreturn]] void CleanExit() {
    if (g_called != nullptr) {
        std::string mask;
        std::size_t k = 0;
        for (const std::atomic<bool>& b : *g_called) {
            mask += b.load() ? '1' : '0';
            if (b.load()) ++k;
        }
        LogErr("serve: coverage called=" + std::to_string(k) + " of=" + std::to_string(g_called->size()) +
               " mask=" + mask);
    }
    ::_exit(0);
}

bool ParseCount(const std::string& s, std::uint64_t& out) {
    if (s.empty() || s.size() > 18) return false;
    for (char c : s)
        if (c < '0' || c > '9') return false;
    out = std::strtoull(s.c_str(), nullptr, 10);
    return true;
}

} // namespace

const std::vector<std::string>& EnvWhitelist() {
    static const std::vector<std::string> w = {"ENGINE_THREADS", "ENGINE_CUDA_DISABLE", "ENGINE_CUDA_DEVICE",
                                               "ENGINE_HOST_THREADS", "PATH", "HOME", "LANG", "LC_ALL", "TMPDIR"};
    return w;
}

bool EnvNameRefused(const std::string& name) {
    for (const std::string& w : EnvWhitelist())
        if (w == name) return false;
    static const char* const kPrefixes[] = {"GOLDEN_", "CONSOLE_", "CLI_MANUAL_", "HECKE_", "ENGINE_", "OC_"};
    for (const char* p : kPrefixes)
        if (name.rfind(p, 0) == 0) return true;
    const std::string suffix = "SAN_OPTIONS";   // ASAN_ / UBSAN_ / TSAN_ / LSAN_OPTIONS
    return name.size() >= suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// (NBCONFIRM S4 fix round 1, audit L2) HACS_HOST_PROBE_FIXTURE is the validations' host-probe interlock (MemAvailable 4096
// MiB, C: 255 GB, HostProbe.cpp).  A served session honours it only as the child of this same binary, i.e. of a validation
// harness (ConsoleServeValidation (L) forks and execs it directly); a --serve launched by a shell, the gateway or any other
// program that carries it is refused at the environment gate, so a production session can never switch the C: rules off.
// Same binary = the parent's /proc/<ppid>/exe is the same file (st_dev, st_ino) as /proc/self/exe.
bool ParentIsThisBinary() {
    struct stat self_st {}, parent_st {};
    const std::string parent = "/proc/" + std::to_string(static_cast<long>(::getppid())) + "/exe";
    if (::stat("/proc/self/exe", &self_st) != 0 || ::stat(parent.c_str(), &parent_st) != 0) return false;
    return self_st.st_dev == parent_st.st_dev && self_st.st_ino == parent_st.st_ino;
}

bool EnvNameRefusedHere(const std::string& name) {
    if (name == "HACS_HOST_PROBE_FIXTURE") return !ParentIsThisBinary();
    // (NBCONFIRM S5) the cancel test interlock (Console/Core/Cancel.h): the same rule -- a validation's child only
    if (name == "HACS_CANCEL_TEST") return !ParentIsThisBinary();
    return EnvNameRefused(name);
}

int RunServe(int argc, char** argv, const std::function<void()>& register_verbs) {
    // --- 1. the environment gate (Sec. 18.9): refuse, never scrub ------------
    for (char** e = environ; e != nullptr && *e != nullptr; ++e) {
        const std::string entry(*e);
        const std::string name = entry.substr(0, entry.find('='));
        if (EnvNameRefusedHere(name)) {
            LogErr("serve: refused: environment variable " + name + " is not permitted in a served session");
            return 1;
        }
    }

    // --- 2. the command line (Sec. 18.2) ---------------------------------------
    std::vector<std::pair<std::string, std::string>> sets;
    for (int i = 0; i < argc; ++i) {
        const std::string a = argv[i] == nullptr ? std::string() : std::string(argv[i]);
        if (a == "--set" && i + 1 < argc) {
            const std::string kv = argv[++i];
            const std::size_t eq = kv.find('=');
            if (eq == std::string::npos || eq == 0) {
                LogErr("serve: usage: --set key=value (got '" + kv + "')");
                return 1;
            }
            sets.emplace_back(kv.substr(0, eq), kv.substr(eq + 1));
        } else if (a == "--rss-cap-mib" && i + 1 < argc) {
            if (!ParseCount(argv[++i], g_rss_cap_mib)) {
                LogErr("serve: usage: --rss-cap-mib <N>, N >= 0");
                return 1;
            }
        } else if (a == "--max-reply-bytes" && i + 1 < argc) {
            std::uint64_t n = 0;
            if (!ParseCount(argv[++i], n) || n < 4096) {
                LogErr("serve: usage: --max-reply-bytes <N>, N >= 4096");
                return 1;
            }
            g_max_reply = static_cast<std::size_t>(n);
        } else {
            LogErr("serve: refused: '" + a +
                   "' is not accepted with --serve (only --set key=value, --rss-cap-mib N, --max-reply-bytes N)");
            return 1;
        }
    }

    // --- 3. process hardening (Sec. 18.9) --------------------------------------
    struct stat st;
    if (::fstat(1, &st) != 0 || S_ISREG(st.st_mode)) {
        LogErr("serve: refused: stdout must be a pipe or a socket, not a regular file");
        return 1;
    }
    struct rlimit zero;
    zero.rlim_cur = 0;
    zero.rlim_max = 0;
    if (::setrlimit(RLIMIT_CORE, &zero) != 0) {
        LogErr(std::string("serve: refused: setrlimit(RLIMIT_CORE) failed: ") + std::strerror(errno));
        return 1;
    }
    if (::setrlimit(RLIMIT_FSIZE, &zero) != 0) {
        LogErr(std::string("serve: refused: setrlimit(RLIMIT_FSIZE) failed: ") + std::strerror(errno));
        return 1;
    }
    std::signal(SIGXFSZ, SIG_IGN);
    std::signal(SIGPIPE, SIG_IGN);
    if (::prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0) {
        LogErr(std::string("serve: refused: prctl(PR_SET_NO_NEW_PRIVS) failed: ") + std::strerror(errno));
        return 1;
    }
    // The protocol stream is a private copy of stdout; fd 1 itself becomes
    // stderr, so a stray printf / std::cout of the pipeline lands in the
    // process log and can never corrupt a frame (Sec. 18.1 item 1).
    g_proto_fd = ::fcntl(1, F_DUPFD_CLOEXEC, 3);
    if (g_proto_fd < 0 || ::dup2(2, 1) < 0) {
        LogErr("serve: refused: cannot set up the protocol stream");
        return 1;
    }
    std::signal(SIGTERM, OnSigterm);

    // --- 4. the registries and the session -------------------------------------
    register_verbs();
    static Server server(Registry::global());
    BuildServed(server.served, server.raw, server.view);
    static std::vector<std::atomic<bool>> called(server.view.full_overloads);
    for (std::atomic<bool>& b : called) b.store(false);
    g_called = &called;
    server.session.reset(new Session(server.served, /*verbose_default=*/false));
    server.session->set_output(server.idle_out);
    for (const auto& kv : sets) {
        if (!serve_tags::IsServeSettable(kv.first)) {
            LogErr("serve: refused: --set " + kv.first + " is not settable in a served session");
            return 1;
        }
        try {
            server.session->settings().set(kv.first, ParseCliLiteral(kv.second), SourcePos("<command line>", 0, 0));
        } catch (const ConsoleError& e) {
            LogErr("serve: " + e.formatted());
            return e.kind() == kNotImplementedKind ? 5 : 1;
        }
    }

    // --- 4b. (NBCONFIRM S4) the cost confirmation of a served session -----------
    // The host facts of Confirm.h's rules (MemAvailable, C:, CPUs, the device profile's staleness) and the session's
    // --rss-cap-mib as R_cap of rule 1; no terminal (the protocol is the prompt); the channel and the heartbeat hook.
    {
        HostProbe hp = host::RealHostProbe();
        hp.prompt_tty = [] { return false; };
        hp.write_prompt = [](const std::string& t) { LogErr("serve: " + t); };
        hp.rss_cap_mib = [] { return g_rss_cap_mib > 0 ? static_cast<double>(g_rss_cap_mib) : -1.0; };
        SetHostProbe(std::move(hp));
        std::random_device rd;
        g_confirm_nonce = (static_cast<std::uint64_t>(rd()) << 32) ^ static_cast<std::uint64_t>(rd()) ^
                          static_cast<std::uint64_t>(::getpid()) ^
                          static_cast<std::uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
        Session* sess = server.session.get();
        g_settings_text = [sess]() { return sess->settings().snapshot().to_string(); };
        SetConfirmChannel(ServeConfirmChannel);
        SetConfirmedRunHook(ArmHeartbeat);
    }

    // --- 4c. (NBCONFIRM S5) the cancel test interlock (a validation's child only, see the environment gate) ---
    {
        console::CancelTest t;
        const char* env = std::getenv("HACS_CANCEL_TEST");
        if (env != nullptr && console::ParseCancelTest(env, t)) {
            console::SetCancelTest(t);
            console::SetCancelWaitNotifier([](std::uint64_t k, const char* site) {
                if (g_cell_flush) g_cell_flush();
                const std::string id_json = g_cell_id_json;
                const std::string text = "cancel test: waiting at " + std::string(site ? site : "") + " (visit " + std::to_string(k) + ")";
                Frame([&](std::uint64_t seq) {
                    return "{\"kind\":\"progress\",\"id\":" + id_json + ",\"seq\":" + std::to_string(seq) + ",\"text\":" + Quote(text) + "}";
                });
            });
        }
    }

    // --- 5. hello (Sec. 18.2) ---------------------------------------------------
    {
        const char* cd = std::getenv("ENGINE_CUDA_DISABLE");
        const bool cuda = OC_ENABLE_CUDA != 0 && !(cd != nullptr && std::string(cd) != "0" && std::string(cd) != "");
        std::string settable = "[";
        bool first = true;
        for (const std::string& k : serve_tags::ServeSettableKeys()) {
            if (server.session->settings().find_spec(k) == nullptr) continue;
            if (!first) settable += ',';
            first = false;
            settable += Quote(k);
        }
        settable += "]";
        Frame([&](std::uint64_t seq) {
            return "{\"kind\":\"hello\",\"seq\":" + std::to_string(seq) + ",\"proto\":" + Quote(kProto) +
                   ",\"json_version\":" + std::to_string(kJsonVersion) + ",\"oc_format\":1,\"console\":" +
                   Quote(kConsoleVersion) + ",\"build\":{\"commit\":" + Quote(oc::GitCommit()) +
                   ",\"config\":" + Quote(OC_BUILD_CONFIG) + ",\"cuda\":" + (cuda ? "true" : "false") +
                   ",\"pid\":" + std::to_string(::getpid()) + "},\"read_only\":true,\"counts\":{\"verbs\":" +
                   std::to_string(server.view.served_verbs) + ",\"overloads\":" +
                   std::to_string(server.view.served_overloads) + ",\"types\":" +
                   std::to_string(server.served.type_names().size()) + "},\"limits\":{\"rss_cap_mib\":" +
                   std::to_string(g_rss_cap_mib) + ",\"max_reply_bytes\":" + std::to_string(g_max_reply) +
                   ",\"max_request_bytes\":" + std::to_string(kMaxRequestBytes) + ",\"confirm_wall_s\":" +
                   std::to_string(static_cast<long long>(SessionConfirmWallS(*server.session))) + "},\"settable\":" + settable + "}";
        });
    }

    std::thread(ReaderLoop).detach();
    if (g_rss_cap_mib > 0) std::thread(WatchdogLoop).detach();
    std::thread(HeartbeatLoop).detach();   // (NBCONFIRM S4) idle unless a confirmed run is armed

    // --- 6. requests, strictly one at a time, in arrival order -----------------
    while (true) {
        Item it;
        {
            std::unique_lock<std::mutex> lock(g_q_mutex);
            while (g_queue.empty()) {
                if (g_term) CleanExit();
                g_q_cv.wait_for(lock, std::chrono::milliseconds(100));
            }
            it = std::move(g_queue.front());
            g_queue.pop_front();
        }
        if (it.eof) CleanExit();
        const auto t0 = std::chrono::steady_clock::now();
        if (it.malformed) {
            Done(nullptr, Rejected("protocol", "malformed JSON: " + it.error), 0, t0);
        } else if (!it.req.is_object()) {
            Done(nullptr, Rejected("protocol", "a request must be a JSON object"), 0, t0);
        } else {
            const Json* op = it.req.get("op");
            const Json* idj = it.req.get("id");
            const bool id_ok = ValidId(idj);
            const std::string id = id_ok ? idj->text : std::string();
            const std::string* idp = id_ok ? &id : nullptr;
            if (op == nullptr || !op->is_string()) {
                Done(idp, Rejected("protocol", "a request needs a string `op`"), 0, t0);
            } else if (op->text == "close") {
                Frame([&](std::uint64_t seq) { return "{\"kind\":\"bye\",\"seq\":" + std::to_string(seq) + "}"; });
                CleanExit();
            } else if (op->text != "eval" && op->text != "complete" && op->text != "help" && op->text != "describe") {
                Done(idp, Rejected("protocol", "unknown op '" + op->text + "'"), 0, t0);
            } else if (!id_ok) {
                Done(nullptr, Rejected("protocol", "`id` must be a string of 1-64 printable ASCII characters"), 0, t0);
            } else {
                try {
                    if (op->text == "eval") server.eval(id, it.req);
                    else if (op->text == "complete") server.complete(id, it.req);
                    else if (op->text == "help") server.help(id, it.req);
                    else server.describe(id, it.req);
                } catch (const std::exception& e) {
                    Fatal("internal", std::string("an exception escaped the dispatcher: ") + e.what());
                } catch (...) {
                    Fatal("internal", "an exception of unknown type escaped the dispatcher");
                }
            }
        }
        if (g_term) CleanExit();
    }
}

} // namespace serve
} // namespace console
