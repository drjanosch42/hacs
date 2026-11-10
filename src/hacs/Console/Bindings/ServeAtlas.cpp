// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// ServeAtlas.cpp -- see ServeAtlas.h.  [P] v3.11 Sec. 18.16 is the
// specification; the letters below are its paragraphs.
//
// (CliManualCoverageValidation scans this file for option comparisons on the
// variable `a`: only the command-line options are compared that way.)
// ===========================================================================

#include "ServeAtlas.h"

#include "Serve.h"
#include "ServeJson.h"

#include "../../Util/AtlasIndex.h"

#include <algorithm>
#include <atomic>
#include <climits>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <map>
#include <mutex>
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

namespace console {
namespace serve_atlas {

namespace {

using serve_json::Json;
using serve_json::Quote;
using serve_json::UIntJson;

int g_proto_fd = -1;
std::mutex g_out_mutex;
std::uint64_t g_seq = 0;
std::size_t g_max_reply = kDefaultMaxReplyBytes;
std::uint64_t g_rss_cap_mib = 0;

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

int Refuse(const std::string& why) {
    LogErr("serve-atlas: refused: " + why);
    return 1;
}

void WriteAllOrDie(const std::string& s) {
    std::size_t off = 0;
    while (off < s.size()) {
        const ssize_t n = ::write(g_proto_fd, s.data() + off, s.size() - off);
        if (n < 0) {
            if (errno == EINTR) continue;
            ::_exit(kFatalExit);
        }
        off += static_cast<std::size_t>(n);
    }
}

template <class F>
void Frame(F build) {
    std::lock_guard<std::mutex> lock(g_out_mutex);
    const std::uint64_t seq = g_seq++;
    WriteAllOrDie(build(seq) + "\n");
}

[[noreturn]] void Fatal(const std::string& reason, const std::string& detail) {
    bool locked = false;
    for (int i = 0; i < 200 && !locked; ++i) {
        locked = g_out_mutex.try_lock();
        if (!locked) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (locked) {
        const std::string f = "{\"kind\":\"atlas_fatal\",\"reason\":" + Quote(reason) + ",\"detail\":" + Quote(detail) +
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
    LogErr("serve-atlas: fatal: " + reason + ": " + detail);
    ::_exit(kFatalExit);
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

void CheckRssCap() {
    if (g_rss_cap_mib == 0) return;
    const std::uint64_t r = ResidentMiB();
    if (r > g_rss_cap_mib)
        Fatal("rss_cap", "resident " + std::to_string(r) + " MiB > cap " + std::to_string(g_rss_cap_mib) + " MiB");
}

void WatchdogLoop() {
    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(kWatchdogPollMs));
        CheckRssCap();
    }
}

bool ParseCount(const std::string& s, std::uint64_t& out) {
    if (s.empty() || s.size() > 18) return false;
    for (char c : s)
        if (c < '0' || c > '9') return false;
    out = std::strtoull(s.c_str(), nullptr, 10);
    return true;
}

bool ReadWholeFile(const std::string& path, std::string& out, std::string& err) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        err = "cannot open " + path + ": " + std::strerror(errno);
        return false;
    }
    struct stat st;
    if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
        ::close(fd);
        err = path + " is not a regular file";
        return false;
    }
    out.clear();
    out.reserve(static_cast<std::size_t>(st.st_size));
    char buf[1 << 16];
    while (true) {
        const ssize_t n = ::read(fd, buf, sizeof(buf));
        if (n < 0) {
            if (errno == EINTR) continue;
            ::close(fd);
            err = "read error on " + path + ": " + std::strerror(errno);
            return false;
        }
        if (n == 0) break;
        out.append(buf, static_cast<std::size_t>(n));
    }
    ::close(fd);
    return true;
}

// ---------------------------------------------------------------------------
// The served store.
// ---------------------------------------------------------------------------
struct Store {
    atlas_index::MappedFile file;
    atlas_index::Index ix;
    std::string fnv_hex;
    std::string info_result;   // the `info` result, built once
};

Store* g_store = nullptr;
std::string g_last_status = "ok";   // the status of the last reply frame (one-shot exit code)

// One store row, re-emitted by the Sec. 18.4 writer.
std::string Row(std::uint64_t off, std::uint64_t len) {
    Json j;
    std::string err;
    if (!serve_json::ParseJson(std::string(g_store->file.data() + off, static_cast<std::size_t>(len)), j, err))
        Fatal("internal", "a store row at byte " + std::to_string(off) + " does not parse: " + err);
    return serve_json::WriteJson(j);
}

struct Reply {
    std::string status = "ok";   // ok | error | rejected
    std::string err_kind, err_msg;
    std::string result = "null";
};

Reply Err(const std::string& status, const std::string& kind, const std::string& msg) {
    Reply r;
    r.status = status;
    r.err_kind = kind;
    r.err_msg = msg;
    return r;
}
Reply Protocol(const std::string& msg) { return Err("rejected", "protocol", msg); }

std::string ReplyFrame(std::uint64_t seq, const std::string& id_json, const std::string& op_json, const Reply& r,
                       std::uint64_t elapsed_us) {
    std::string f = "{\"kind\":\"atlas_reply\",\"id\":" + id_json + ",\"seq\":" + std::to_string(seq) +
                    ",\"op\":" + op_json + ",\"status\":" + Quote(r.status) + ",\"error\":";
    if (r.err_kind.empty()) f += "null";
    else f += "{\"kind\":" + Quote(r.err_kind) + ",\"message\":" + Quote(r.err_msg) + "}";
    f += ",\"store\":" + Quote(g_store->fnv_hex) + ",\"result\":" + r.result + ",\"perf\":{\"elapsed_us\":" +
         std::to_string(elapsed_us) + "}}";
    return f;
}

// The bytes a reply frame adds around its `result` (plus a margin for the
// volatile seq / elapsed digits and the newline).
std::size_t FrameOverhead(const std::string& id_json, const std::string& op_json) {
    Reply r;
    r.result.clear();
    return ReplyFrame(0, id_json, op_json, r, 0).size() + 48;
}

// ---- field checks ----------------------------------------------------------
bool UIntField(const Json* j, std::uint64_t max, std::uint64_t& out) {
    std::int64_t v = 0;
    if (j == nullptr || !j->is_number() || !j->as_int64(v) || v < 0 || static_cast<std::uint64_t>(v) > max) return false;
    out = static_cast<std::uint64_t>(v);
    return true;
}

bool OnlyFields(const Json& req, std::initializer_list<const char*> allowed, std::string& bad) {
    for (const auto& m : req.members) {
        bool ok = false;
        for (const char* a : allowed) ok = ok || m.first == a;
        if (!ok) {
            bad = m.first;
            return false;
        }
    }
    return true;
}

bool Paging(const Json& req, std::uint64_t& offset, std::uint64_t& limit, std::string& msg) {
    offset = 0;
    limit = kPageDefault;
    if (const Json* o = req.get("offset")) {
        if (!UIntField(o, 0xffffffffull, offset)) {
            msg = "`offset` must be an integer >= 0";
            return false;
        }
    }
    if (const Json* l = req.get("limit")) {
        if (!UIntField(l, kPageMax, limit) || limit == 0) {
            msg = "`limit` must be an integer in 1 .. " + std::to_string(kPageMax);
            return false;
        }
    }
    return true;
}

// A page of items: at most `limit`, cut earlier at the frame cap.  `item(k)`
// renders the k-th match; `count` is the number of matches.
template <class ItemAt>
Reply Page(std::uint64_t total, std::uint64_t offset, std::uint64_t limit, std::size_t overhead, ItemAt item) {
    Reply r;
    const std::string head = "{\"total\":" + std::to_string(total) + ",\"offset\":" + std::to_string(offset) + ",\"items\":[";
    std::string items;
    std::uint64_t k = offset, taken = 0;
    const std::size_t budget = g_max_reply > overhead + head.size() + 32 ? g_max_reply - overhead - head.size() - 32 : 0;
    bool cut = false;
    while (k < total && taken < limit) {
        const std::string it = item(k);
        const std::size_t add = it.size() + (taken > 0 ? 1 : 0);
        if (items.size() + add > budget) {
            if (taken == 0)
                return Err("error", "too_large", "item " + std::to_string(k) + " alone exceeds max_reply_bytes (" +
                                                     std::to_string(g_max_reply) + ")");
            cut = true;
            break;
        }
        if (taken > 0) items += ',';
        items += it;
        ++taken;
        ++k;
    }
    (void)cut;
    const std::string next = k < total ? std::to_string(k) : std::string("null");
    r.result = head + items + "],\"next\":" + next + "}";
    return r;
}

// ---- the ops (Sec. 18.16 (e)) ----------------------------------------------
Reply OpInfo(const Json& req) {
    std::string bad;
    if (!OnlyFields(req, {"op", "id"}, bad)) return Protocol("unknown field `" + bad + "` for op info");
    Reply r;
    r.result = g_store->info_result;
    return r;
}

Reply OpCells(const Json& req, std::size_t overhead) {
    std::string bad, msg;
    if (!OnlyFields(req, {"op", "id", "offset", "limit"}, bad)) return Protocol("unknown field `" + bad + "` for op cells");
    std::uint64_t offset = 0, limit = 0;
    if (!Paging(req, offset, limit, msg)) return Protocol(msg);
    const auto& cells = g_store->ix.cells;
    return Page(cells.size(), offset, limit, overhead,
                [&](std::uint64_t k) { return Row(cells[static_cast<std::size_t>(k)].off, cells[static_cast<std::size_t>(k)].len); });
}

Reply OpCell(const Json& req) {
    std::string bad;
    if (!OnlyFields(req, {"op", "id", "n"}, bad)) return Protocol("unknown field `" + bad + "` for op cell");
    std::uint64_t n = 0;
    if (!UIntField(req.get("n"), 0xffffffffull, n)) return Protocol("op cell needs an integer `n` >= 0");
    const auto& cells = g_store->ix.cells;
    const auto it = std::lower_bound(cells.begin(), cells.end(), n,
                                     [](const atlas_index::CellRow& c, std::uint64_t v) { return c.n < v; });
    if (it == cells.end() || it->n != n) return Err("error", "not_found", "no cell n = " + std::to_string(n));
    std::string occ = "[";
    bool first = true;
    for (const atlas_index::OccRow& o : g_store->ix.occ) {
        if (o.n != n) continue;
        if (!first) occ += ',';
        first = false;
        occ += "{\"ideal\":" + std::to_string(o.ideal) + ",\"sid\":" + std::to_string(o.sid) + "}";
    }
    occ += "]";
    Reply r;
    r.result = "{\"cell\":" + Row(it->off, it->len) + ",\"occurrences\":" + occ + "}";
    return r;
}

Reply OpSystem(const Json& req) {
    std::string bad;
    if (!OnlyFields(req, {"op", "id", "sid"}, bad)) return Protocol("unknown field `" + bad + "` for op system");
    std::uint64_t sid = 0;
    if (!UIntField(req.get("sid"), 0xffffffffull, sid)) return Protocol("op system needs an integer `sid` >= 0");
    if (sid >= g_store->ix.systems.size())
        return Err("error", "not_found", "no system sid = " + std::to_string(sid) + " (the store has " +
                                             std::to_string(g_store->ix.systems.size()) + ")");
    const atlas_index::SysRow& s = g_store->ix.systems[static_cast<std::size_t>(sid)];
    Reply r;
    r.result = "{\"system\":" + Row(s.off, s.len) + "}";
    return r;
}

Reply OpSystems(const Json& req, std::size_t overhead) {
    std::string bad, msg;
    if (!OnlyFields(req, {"op", "id", "filter", "offset", "limit"}, bad))
        return Protocol("unknown field `" + bad + "` for op systems");
    std::uint64_t offset = 0, limit = 0;
    if (!Paging(req, offset, limit, msg)) return Protocol(msg);
    bool has_r = false, has_type = false, has_coll = false, has_base = false, has_twist = false, has_n = false;
    std::uint64_t r = 0, base = 0, twist = 0, n = 0;
    char type = 'c';
    bool coll = false;
    if (const Json* f = req.get("filter")) {
        if (!f->is_object()) return Protocol("`filter` must be an object");
        for (const auto& m : f->members) {
            const std::string& k = m.first;
            const Json& v = m.second;
            if (k == "r") {
                if (!UIntField(&v, 0xffffffffull, r)) return Protocol("filter.r must be an integer >= 0");
                has_r = true;
            } else if (k == "type") {
                if (!v.is_string() || (v.text != "eisenstein" && v.text != "cuspidal"))
                    return Protocol("filter.type must be \"eisenstein\" or \"cuspidal\"");
                type = v.text == "eisenstein" ? 'e' : 'c';
                has_type = true;
            } else if (k == "collected") {
                if (!v.is_bool()) return Protocol("filter.collected must be a Bool");
                coll = v.b;
                has_coll = true;
            } else if (k == "base") {
                if (!UIntField(&v, 0xffffffffull, base)) return Protocol("filter.base must be an integer >= 0");
                has_base = true;
            } else if (k == "twist") {
                if (!UIntField(&v, 0xffffffffull, twist)) return Protocol("filter.twist must be an integer >= 0");
                has_twist = true;
            } else if (k == "n") {
                if (!UIntField(&v, 0xffffffffull, n)) return Protocol("filter.n must be an integer >= 0");
                has_n = true;
            } else {
                return Protocol("unknown filter key `" + k + "`");
            }
        }
    }
    const auto& sys = g_store->ix.systems;
    std::vector<char> at_n;
    if (has_n) {
        at_n.assign(sys.size(), 0);
        for (const atlas_index::OccRow& o : g_store->ix.occ)
            if (o.n == n && o.sid < sys.size()) at_n[o.sid] = 1;
    }
    std::vector<std::uint32_t> match;
    for (std::size_t i = 0; i < sys.size(); ++i) {
        const atlas_index::SysRow& s = sys[i];
        if (has_r && s.r != r) continue;
        if (has_type && s.type != type) continue;
        if (has_coll && (s.collected != 0) != coll) continue;
        if (has_base && s.base != base) continue;
        if (has_twist && s.twist != twist) continue;
        if (has_n && !at_n[i]) continue;
        match.push_back(static_cast<std::uint32_t>(i));
    }
    return Page(match.size(), offset, limit, overhead, [&](std::uint64_t k) {
        const atlas_index::SysRow& s = sys[match[static_cast<std::size_t>(k)]];
        return Row(s.off, s.len);
    });
}

bool ValidId(const Json* id) {
    if (id == nullptr || !id->is_string()) return false;
    const std::string& s = id->text;
    if (s.empty() || s.size() > 64) return false;
    for (char c : s)
        if (c < 0x20 || c > 0x7e) return false;
    return true;
}

void HandleLine(std::string line) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    bool blank = true;
    for (char c : line) blank = blank && (c == ' ' || c == '\t');
    if (blank) return;
    const auto t0 = std::chrono::steady_clock::now();
    Json req;
    std::string perr;
    std::string id_json = "null", op_json = "null";
    Reply rep;
    if (!serve_json::ParseJson(line, req, perr)) {
        rep = Protocol("malformed JSON: " + perr);
    } else if (!req.is_object()) {
        rep = Protocol("a request must be a JSON object");
    } else {
        const Json* op = req.get("op");
        const Json* id = req.get("id");
        if (op != nullptr && op->is_string()) op_json = Quote(op->text);
        if (ValidId(id)) id_json = Quote(id->text);
        if (op == nullptr || !op->is_string()) {
            rep = Protocol("a request needs a string `op`");
        } else if (op->text == "close") {
            Frame([&](std::uint64_t seq) { return "{\"kind\":\"atlas_bye\",\"seq\":" + std::to_string(seq) + "}"; });
            ::_exit(0);
        } else if (!ValidId(id)) {
            rep = Protocol("`id` must be a string of 1-64 printable ASCII characters");
        } else {
            const std::size_t overhead = FrameOverhead(id_json, op_json);
            if (op->text == "info") rep = OpInfo(req);
            else if (op->text == "cells") rep = OpCells(req, overhead);
            else if (op->text == "cell") rep = OpCell(req);
            else if (op->text == "system") rep = OpSystem(req);
            else if (op->text == "systems") rep = OpSystems(req, overhead);
            else rep = Protocol("unknown op '" + op->text + "'");
            if (rep.status == "ok" && rep.result.size() + overhead > g_max_reply)
                rep = Err("error", "too_large",
                          "the reply exceeds max_reply_bytes (" + std::to_string(g_max_reply) + ")");
        }
    }
    const std::uint64_t us = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count());
    g_last_status = rep.status;
    Frame([&](std::uint64_t seq) { return ReplyFrame(seq, id_json, op_json, rep, us); });
    CheckRssCap();
}

// The `info` result (built once at open).
bool BuildInfo(Store& s, std::string& err) {
    const atlas_index::Index& ix = s.ix;
    const std::string doc = std::string(s.file.data() + ix.head_off, static_cast<std::size_t>(ix.head_len)) +
                            "\"cells\":[],\"systems\":[]" +
                            std::string(s.file.data() + ix.tail_off, static_cast<std::size_t>(ix.tail_len));
    Json j;
    if (!serve_json::ParseJson(doc, j, err) || !j.is_object()) {
        err = "the store header does not parse: " + err;
        return false;
    }
    Json header;
    header.kind = Json::Kind::Object;
    for (auto& m : j.members)
        if (m.first != "cells" && m.first != "systems") header.members.push_back(std::move(m));
    std::uint64_t eis = 0, cusp = 0, coll = 0;
    std::map<std::uint32_t, std::uint64_t> by_r;
    for (const atlas_index::SysRow& r : ix.systems) {
        if (r.type == 'e') ++eis;
        else ++cusp;
        if (r.collected) ++coll;
        ++by_r[r.r];
    }
    std::string prof = "[";
    for (const auto& kv : by_r) {
        if (prof.size() > 1) prof += ',';
        prof += "[" + std::to_string(kv.first) + "," + std::to_string(kv.second) + "]";
    }
    prof += "]";
    s.info_result = "{\"header\":" + serve_json::WriteJson(header) + ",\"cells\":" + std::to_string(ix.cells.size()) +
                    ",\"systems\":" + std::to_string(ix.systems.size()) + ",\"occurrences\":" +
                    std::to_string(ix.occ.size()) + ",\"collected\":" + std::to_string(coll) +
                    ",\"types\":{\"eisenstein\":" + std::to_string(eis) + ",\"cuspidal\":" + std::to_string(cusp) +
                    "},\"r_profile\":" + prof + "}";
    return true;
}

// (c)(4) the store: O_RDONLY, PROT_READ, its FNV; (c)(5) the index: derived from
// the store, the sidecar (if any) compared field by field.  Shared by the
// served process and the one-shot `--atlas --query`.  Returns 0 or the exit
// code of a refusal (1).
int OpenStoreOrRefuse(Store& store, const std::string& store_path, const std::string& index_path, bool has_index,
                      bool has_fnv, std::uint64_t expect_fnv, std::string& source, std::uint64_t& index_fnv,
                      std::uint64_t& open_ms) {
    const auto t_open = std::chrono::steady_clock::now();
    g_store = &store;
    std::string err;
    char real[PATH_MAX];
    if (::realpath(store_path.c_str(), real) == nullptr)
        return Refuse("cannot resolve the store " + store_path + ": " + std::strerror(errno));
    const std::string store_real(real);
    if (!store.file.open(store_real, err)) return Refuse(err);
    store.file.advise_sequential();
    const std::uint64_t fnv = atlas_index::FnvOf(store.file.data(), store.file.size());
    store.fnv_hex = atlas_index::Hex16(fnv);
    if (has_fnv && fnv != expect_fnv)
        return Refuse("the store's FNV-1a-64 " + store.fnv_hex + " differs from --expect-fnv " +
                      atlas_index::Hex16(expect_fnv));
    if (!atlas_index::ScanStore(store.file.data(), store.file.size(), atlas_index::BaseName(store_real), fnv, store.ix,
                                err))
        return Refuse("the store is not indexable: " + err);
    store.file.release_resident();
    source = "memory";
    index_fnv = 0;
    {
        std::string path = index_path;
        if (!has_index) {
            const std::string def = store_real + atlas_index::kIndexSuffix;
            struct stat ist;
            if (::stat(def.c_str(), &ist) == 0) path = def;
        }
        if (!path.empty()) {
            std::string text;
            if (!ReadWholeFile(path, text, err)) return Refuse("the index: " + err);
            atlas_index::Index side;
            if (!atlas_index::Parse(text, side, err)) return Refuse("stale or corrupt index " + path + ": " + err);
            std::string diff;
            if (!atlas_index::Equal(side, store.ix, diff))
                return Refuse("stale or corrupt index " + path + ": it differs from the store at " + diff);
            source = "sidecar";
            index_fnv = atlas_index::FnvOf(text.data(), text.size());
        } else {
            const std::string text = atlas_index::Serialize(store.ix);
            index_fnv = atlas_index::FnvOf(text.data(), text.size());
        }
    }
    if (!BuildInfo(store, err)) return Refuse(err);
    open_ms = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t_open).count());
    return 0;
}

} // namespace

int RunServeAtlas(int argc, char** argv) {
    // --- (c)(1) the environment gate of Sec. 18.9: refuse, never scrub --------
    for (char** e = environ; e != nullptr && *e != nullptr; ++e) {
        const std::string entry(*e);
        const std::string name = entry.substr(0, entry.find('='));
        if (serve::EnvNameRefused(name))
            return Refuse("environment variable " + name + " is not permitted in a served session");
    }

    // --- (c)(2) the command line -----------------------------------------------
    std::string store_path, index_path;
    bool has_store = false, has_index = false, has_fnv = false, has_reply = false, has_cap = false;
    std::uint64_t expect_fnv = 0;
    for (int i = 0; i < argc; ++i) {
        const std::string a = argv[i] == nullptr ? std::string() : std::string(argv[i]);
        const bool has_value = i + 1 < argc && argv[i + 1] != nullptr;
        if (a == "--store" && has_value && !has_store) {
            store_path = argv[++i];
            has_store = true;
        } else if (a == "--index" && has_value && !has_index) {
            index_path = argv[++i];
            has_index = true;
        } else if (a == "--expect-fnv" && has_value && !has_fnv) {
            if (!atlas_index::ParseHex64(argv[++i], expect_fnv))
                return Refuse("--expect-fnv takes 0x<1-16 hex digits>");
            has_fnv = true;
        } else if (a == "--max-reply-bytes" && has_value && !has_reply) {
            std::uint64_t n = 0;
            if (!ParseCount(argv[++i], n) || n < kMinMaxReplyBytes)
                return Refuse("--max-reply-bytes <N>, N >= " + std::to_string(kMinMaxReplyBytes));
            g_max_reply = static_cast<std::size_t>(n);
            has_reply = true;
        } else if (a == "--rss-cap-mib" && has_value && !has_cap) {
            if (!ParseCount(argv[++i], g_rss_cap_mib)) return Refuse("--rss-cap-mib <N>, N >= 0");
            has_cap = true;
        } else {
            return Refuse("'" + a +
                          "' is not accepted with --serve-atlas (only --store <f>, --index <f>, --expect-fnv 0x<hex>, "
                          "--max-reply-bytes N, --rss-cap-mib N, each at most once)");
        }
    }
    if (!has_store) return Refuse("--store <store.json> is required");

    // --- (c)(3) process hardening (Sec. 18.9) ----------------------------------
    struct stat st;
    if (::fstat(1, &st) != 0 || S_ISREG(st.st_mode))
        return Refuse("stdout must be a pipe or a socket, not a regular file");
    struct rlimit zero;
    zero.rlim_cur = 0;
    zero.rlim_max = 0;
    if (::setrlimit(RLIMIT_CORE, &zero) != 0)
        return Refuse(std::string("setrlimit(RLIMIT_CORE) failed: ") + std::strerror(errno));
    if (::setrlimit(RLIMIT_FSIZE, &zero) != 0)
        return Refuse(std::string("setrlimit(RLIMIT_FSIZE) failed: ") + std::strerror(errno));
    std::signal(SIGXFSZ, SIG_IGN);
    std::signal(SIGPIPE, SIG_IGN);
    if (::prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0)
        return Refuse(std::string("prctl(PR_SET_NO_NEW_PRIVS) failed: ") + std::strerror(errno));
    g_proto_fd = ::fcntl(1, F_DUPFD_CLOEXEC, 3);
    if (g_proto_fd < 0 || ::dup2(2, 1) < 0) return Refuse("cannot set up the protocol stream");

    // --- (c)(4), (c)(5) the store and its index (OpenStoreOrRefuse) -------------
    static Store store;
    std::string source;
    std::uint64_t index_fnv = 0, open_ms = 0;
    {
        const int rc = OpenStoreOrRefuse(store, store_path, index_path, has_index, has_fnv, expect_fnv, source,
                                         index_fnv, open_ms);
        if (rc != 0) return rc;
    }

    // --- (c)(6) atlas_hello ----------------------------------------------------
    const atlas_index::Index& ix = store.ix;
    Frame([&](std::uint64_t seq) {
        return "{\"kind\":\"atlas_hello\",\"seq\":" + std::to_string(seq) + ",\"proto\":" + Quote(kProto) +
               ",\"read_only\":true,\"build\":{\"commit\":" + Quote(oc::GitCommit()) + ",\"config\":" +
               Quote(OC_BUILD_CONFIG) + ",\"pid\":" + std::to_string(::getpid()) + "},\"store\":{\"name\":" +
               Quote(ix.store_name) + ",\"bytes\":" + UIntJson(ix.store_bytes) + ",\"fnv\":" + Quote(store.fnv_hex) +
               ",\"schema\":" + Quote(ix.schema) + ",\"ell\":" + std::to_string(ix.ell) + ",\"n_max\":" +
               std::to_string(ix.n_max) + ",\"cells\":" + std::to_string(ix.cells.size()) + ",\"systems\":" +
               std::to_string(ix.systems.size()) + ",\"occurrences\":" + std::to_string(ix.occ.size()) +
               "},\"index\":{\"source\":" + Quote(source) + ",\"fnv\":" + Quote(atlas_index::Hex16(index_fnv)) +
               "},\"limits\":{\"max_reply_bytes\":" + std::to_string(g_max_reply) + ",\"max_request_bytes\":" +
               std::to_string(kMaxRequestBytes) + ",\"page_max\":" + std::to_string(kPageMax) + ",\"rss_cap_mib\":" +
               std::to_string(g_rss_cap_mib) + "},\"open_ms\":" + std::to_string(open_ms) + "}";
    });
    if (g_rss_cap_mib > 0) std::thread(WatchdogLoop).detach();
    CheckRssCap();

    // --- requests, one at a time ------------------------------------------------
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
            ::_exit(0);
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
                Fatal("protocol_overflow",
                      "a request line exceeds max_request_bytes (" + std::to_string(kMaxRequestBytes) + ")");
        }
    }
}

// `cohomology_app --atlas --query <store.json> '<request JSON>'` (slot-22 VERB-3,
// owner decision O13): the request handler of the served process, run once.
// Prints exactly one `atlas_reply` frame (no hello) on stdout.  Exit 0 reply
// `ok`; 1 usage or a refusal of the store; 3 reply `error` / `rejected`.
// The store is opened O_RDONLY / PROT_READ exactly as in the served process; the
// path comes from the command line only; `close` is not an op here.  The
// environment gate of --serve is not applied (a one-shot CLI, not a served
// session); RLIMIT_CORE is zeroed.
int RunAtlasQuery(int argc, char** argv) {
    if (argc != 2 || argv[0] == nullptr || argv[1] == nullptr)
        return Refuse("usage: hacs --atlas --query <store.json> '<request JSON>'");
    const std::string request(argv[1]);
    if (request.size() > kMaxRequestBytes)
        return Refuse("the request exceeds " + std::to_string(kMaxRequestBytes) + " bytes");
    // (VERB-3 audit) one request = one NDJSON line: a blank request would be
    // skipped by HandleLine (no frame, exit 0) and an embedded newline would be
    // two requests to a served session, so both are usage errors here.
    if (request.find_first_not_of(" \t\r") == std::string::npos)
        return Refuse("the request is blank");
    if (request.find('\n') != std::string::npos)
        return Refuse("the request must be one line (no newline)");
    struct rlimit zero;
    zero.rlim_cur = 0;
    zero.rlim_max = 0;
    (void)::setrlimit(RLIMIT_CORE, &zero);
    std::signal(SIGPIPE, SIG_IGN);
    g_proto_fd = 1;
    static Store store;
    std::string source;
    std::uint64_t index_fnv = 0, open_ms = 0;
    const int rc = OpenStoreOrRefuse(store, argv[0], std::string(), false, false, 0, source, index_fnv, open_ms);
    if (rc != 0) return rc;
    {   // `close` ends a served session; here it is a request that does not exist.
        Json probe;
        std::string perr;
        if (serve_json::ParseJson(request, probe, perr) && probe.is_object()) {
            const Json* op = probe.get("op");
            if (op != nullptr && op->is_string() && op->text == "close")
                return Refuse("op close is not available in --atlas --query");
        }
    }
    HandleLine(request);
    return g_last_status == "ok" ? 0 : 3;
}

} // namespace serve_atlas
} // namespace console
