// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// AtlasIndex.cpp -- see AtlasIndex.h ([P] v3.11 Sec. 18.16 (b)).
// ===========================================================================

#include "AtlasIndex.h"

#include "../../ModArithmetic/Fnv.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <tuple>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace atlas_index {

namespace {

// ---------------------------------------------------------------------------
// A strict cursor over the store bytes.
// ---------------------------------------------------------------------------
struct Cur {
    const char* d;
    std::size_t n;
    std::size_t p;
    bool at(char ch) const { return p < n && d[p] == ch; }
};

bool Lit(Cur& c, const char* s) {
    const std::size_t L = std::strlen(s);
    if (c.p + L > c.n || std::memcmp(c.d + c.p, s, L) != 0) return false;
    c.p += L;
    return true;
}

bool U32(Cur& c, std::uint32_t& v) {
    const std::size_t s = c.p;
    std::uint64_t x = 0;
    while (c.p < c.n && c.d[c.p] >= '0' && c.d[c.p] <= '9') {
        x = x * 10 + static_cast<std::uint64_t>(c.d[c.p] - '0');
        ++c.p;
        if (c.p - s > 10 || x > 0xffffffffull) return false;
    }
    if (c.p == s) return false;
    if (c.p - s > 1 && c.d[s] == '0') return false;   // no leading zero
    v = static_cast<std::uint32_t>(x);
    return true;
}

bool Bool(Cur& c, bool& b) {
    if (Lit(c, "true")) {
        b = true;
        return true;
    }
    if (Lit(c, "false")) {
        b = false;
        return true;
    }
    return false;
}

// A nested array of non-negative integers (the eigenvalue lists): only digits,
// commas and brackets.
bool SkipNumArray(Cur& c) {
    if (!c.at('[')) return false;
    int depth = 0;
    while (c.p < c.n) {
        const char ch = c.d[c.p];
        if (ch == '[') {
            ++depth;
        } else if (ch == ']') {
            if (--depth == 0) {
                ++c.p;
                return true;
            }
        } else if (!((ch >= '0' && ch <= '9') || ch == ',')) {
            return false;
        }
        ++c.p;
    }
    return false;
}

// A balanced JSON object starting at '{' on ONE line (strings and their
// escapes skipped; a newline outside a string ends the scan unsuccessfully).
bool SkipObject(Cur& c) {
    if (!c.at('{')) return false;
    int depth = 0;
    bool in_str = false;
    while (c.p < c.n) {
        const char ch = c.d[c.p];
        if (in_str) {
            if (ch == '\\') ++c.p;
            else if (ch == '"') in_str = false;
            else if (ch == '\n') return false;
        } else if (ch == '"') {
            in_str = true;
        } else if (ch == '{' || ch == '[') {
            ++depth;
        } else if (ch == '}' || ch == ']') {
            if (--depth == 0) {
                ++c.p;
                return true;
            }
        } else if (ch == '\n') {
            return false;
        }
        ++c.p;
    }
    return false;
}

std::string Where(const Cur& c, const std::string& what) {
    return what + " at byte " + std::to_string(c.p);
}

bool ScanCell(Cur& c, CellRow& row, std::string& err) {
    const std::size_t start = c.p;
    if (!Lit(c, "{\"n\":") || !U32(c, row.n) || !c.at(',')) {
        err = Where(c, "a cell row does not start {\"n\":<n>,");
        return false;
    }
    c.p = start;
    if (!SkipObject(c)) {
        err = Where(c, "an unbalanced cell row");
        return false;
    }
    row.off = start;
    row.len = c.p - start;
    return true;
}

bool ScanSystem(Cur& c, std::uint32_t expect_id, SysRow& row, std::vector<OccRow>& occ, std::string& err) {
    const std::size_t start = c.p;
    std::uint32_t id = 0, a = 0, b = 0;
    bool collected = false, in_h1 = false;
    auto fail = [&](const char* what) {
        err = Where(c, std::string("system row ") + std::to_string(expect_id) + ": " + what);
        return false;
    };
    if (!Lit(c, "{\"id\":") || !U32(c, id)) return fail("does not start {\"id\":<id>");
    if (id != expect_id) return fail("its id is not its row index");
    if (!Lit(c, ",\"r\":") || !U32(c, row.r) || row.r == 0) return fail("no \"r\":<r >= 1>");
    if (!Lit(c, ",\"eigenvalues\":") || !SkipNumArray(c)) return fail("no \"eigenvalues\" integer array");
    if (!Lit(c, ",\"type\":\"")) return fail("no \"type\"");
    if (Lit(c, "eisenstein\"")) row.type = 'e';
    else if (Lit(c, "cuspidal\"")) row.type = 'c';
    else return fail("\"type\" is neither eisenstein nor cuspidal");
    if (!Lit(c, ",\"chi\":")) return fail("no \"chi\"");
    if (!Lit(c, "null")) {
        if (!Lit(c, "[") || !U32(c, a) || !Lit(c, ",") || !U32(c, b) || !Lit(c, "]"))
            return fail("\"chi\" is neither null nor [a,b]");
    }
    if (!Lit(c, ",\"base\":") || !U32(c, row.base)) return fail("no \"base\"");
    if (!Lit(c, ",\"twist\":") || !U32(c, row.twist)) return fail("no \"twist\"");
    if (!Lit(c, ",\"collected\":") || !Bool(c, collected)) return fail("no \"collected\"");
    if (!Lit(c, ",\"in_H1\":") || !Bool(c, in_h1)) return fail("no \"in_H1\"");
    if (!Lit(c, ",\"occurrences\":[")) return fail("no \"occurrences\"");
    row.collected = collected ? 1 : 0;
    if (!Lit(c, "]")) {
        while (true) {
            const std::size_t o0 = c.p;
            OccRow o;
            o.sid = id;
            if (!Lit(c, "{\"n\":") || !U32(c, o.n) || !Lit(c, ",\"ideal\":") || !U32(c, o.ideal))
                return fail("an occurrence does not start {\"n\":<n>,\"ideal\":<i>");
            c.p = o0;
            if (!SkipObject(c)) return fail("an unbalanced occurrence");
            occ.push_back(o);
            if (Lit(c, ",")) continue;
            if (Lit(c, "]")) break;
            return fail("an occurrence list that is not , or ] separated");
        }
    }
    if (!Lit(c, "}")) return fail("does not end after \"occurrences\"");
    row.off = start;
    row.len = c.p - start;
    return true;
}

// ---------------------------------------------------------------------------
// Sidecar lines.
// ---------------------------------------------------------------------------
std::vector<std::string> Fields(const char* b, const char* e) {
    std::vector<std::string> out;
    const char* s = b;
    for (const char* q = b; q <= e; ++q) {
        if (q == e || *q == '\t') {
            out.emplace_back(s, q);
            s = q + 1;
        }
    }
    return out;
}

bool ToU64(const std::string& s, std::uint64_t& v) {
    if (s.empty() || s.size() > 20) return false;
    if (s.size() > 1 && s[0] == '0') return false;
    std::uint64_t x = 0;
    for (char ch : s) {
        if (ch < '0' || ch > '9') return false;
        const std::uint64_t d = static_cast<std::uint64_t>(ch - '0');
        if (x > (UINT64_MAX - d) / 10) return false;
        x = x * 10 + d;
    }
    v = x;
    return true;
}

bool ToU32(const std::string& s, std::uint32_t& v) {
    std::uint64_t x = 0;
    if (!ToU64(s, x) || x > 0xffffffffull) return false;
    v = static_cast<std::uint32_t>(x);
    return true;
}

} // namespace

// ---------------------------------------------------------------------------
std::uint64_t FnvRange(std::uint64_t h, const char* p, std::size_t n) {
    const unsigned char* u = reinterpret_cast<const unsigned char*>(p);
    for (std::size_t i = 0; i < n; ++i) {
        h ^= u[i];
        h *= engine::kFnvPrime64;
    }
    return h;
}

std::uint64_t FnvOf(const char* p, std::size_t n) { return FnvRange(engine::kFnvOffset64, p, n); }

std::string Hex16(std::uint64_t v) {
    static const char* const kHex = "0123456789abcdef";
    std::string s = "0x0000000000000000";
    for (int i = 0; i < 16; ++i) s[17 - i] = kHex[(v >> (4 * i)) & 0xf];
    return s;
}

bool ParseHex64(const std::string& s, std::uint64_t& out) {
    if (s.size() < 3 || s.size() > 18 || s[0] != '0' || (s[1] != 'x' && s[1] != 'X')) return false;
    std::uint64_t v = 0;
    for (std::size_t i = 2; i < s.size(); ++i) {
        const char ch = s[i];
        int d;
        if (ch >= '0' && ch <= '9') d = ch - '0';
        else if (ch >= 'a' && ch <= 'f') d = ch - 'a' + 10;
        else if (ch >= 'A' && ch <= 'F') d = ch - 'A' + 10;
        else return false;
        v = (v << 4) | static_cast<std::uint64_t>(d);
    }
    out = v;
    return true;
}

std::string BaseName(const std::string& path) {
    const std::size_t slash = path.find_last_of('/');
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

bool ScanStore(const char* data, std::size_t size, const std::string& store_name, std::uint64_t store_fnv,
               Index& out, std::string& error) {
    Index ix;
    ix.store_name = store_name;
    ix.store_bytes = size;
    ix.store_fnv = store_fnv;
    ix.schema = kStoreSchema;
    Cur c{data, size, 0};
    if (!Lit(c, "{\"schema\":\"hecke_atlas/v1\",\"ell\":") || !U32(c, ix.ell) || !Lit(c, ",\"n_max\":") ||
        !U32(c, ix.n_max) || !Lit(c, ",")) {
        error = "not a hecke_atlas/v1 store: it does not start {\"schema\":\"hecke_atlas/v1\",\"ell\":<ell>,\"n_max\":<n>,";
        return false;
    }
    static const char kCells[] = "\n\"cells\":[";
    const char* hit = std::search(data + c.p, data + size, kCells, kCells + sizeof(kCells) - 1);
    if (hit == data + size || hit[-1] != ',') {
        error = "no line starting \"cells\":[ after the header";
        return false;
    }
    ix.head_off = 0;
    ix.head_len = static_cast<std::uint64_t>(hit - data) + 1;   // through the '\n'
    c.p = static_cast<std::size_t>(hit - data) + sizeof(kCells) - 1;

    if (!Lit(c, "]")) {
        while (true) {
            if (!Lit(c, "\n")) {
                error = Where(c, "a cell row does not start a line");
                return false;
            }
            CellRow row;
            if (!ScanCell(c, row, error)) return false;
            if (!ix.cells.empty() && row.n <= ix.cells.back().n) {
                error = Where(c, "cells are not strictly ascending in n");
                return false;
            }
            ix.cells.push_back(row);
            if (Lit(c, ",")) continue;
            if (Lit(c, "]")) break;
            error = Where(c, "the cell list is not , or ] separated");
            return false;
        }
    }
    if (!Lit(c, ",\n\"systems\":[")) {
        error = Where(c, "no line starting \"systems\":[ after the cells");
        return false;
    }
    if (!Lit(c, "]")) {
        while (true) {
            if (!Lit(c, "\n")) {
                error = Where(c, "a system row does not start a line");
                return false;
            }
            SysRow row;
            if (!ScanSystem(c, static_cast<std::uint32_t>(ix.systems.size()), row, ix.occ, error)) return false;
            ix.systems.push_back(row);
            if (Lit(c, ",")) continue;
            if (Lit(c, "]")) break;
            error = Where(c, "the system list is not , or ] separated");
            return false;
        }
    }
    ix.tail_off = c.p;
    ix.tail_len = size - c.p;
    {
        Cur t = c;
        if (!Lit(t, ",\n\"chi_count\":")) {
            error = Where(c, "the systems are not followed by ,\\n\"chi_count\":");
            return false;
        }
        std::size_t e = size;
        while (e > c.p && (data[e - 1] == '\n' || data[e - 1] == ' ' || data[e - 1] == '\r' || data[e - 1] == '\t')) --e;
        if (e == c.p || data[e - 1] != '}') {
            error = "the store does not end with }";
            return false;
        }
    }
    std::sort(ix.occ.begin(), ix.occ.end(), [](const OccRow& x, const OccRow& y) {
        return std::tie(x.n, x.ideal, x.sid) < std::tie(y.n, y.ideal, y.sid);
    });
    out = std::move(ix);
    return true;
}

std::string Serialize(const Index& ix) {
    std::string s;
    s.reserve(256 + ix.cells.size() * 32 + ix.systems.size() * 40 + ix.occ.size() * 24);
    auto kv = [&](const char* k, const std::string& v) {
        s += k;
        s += '\t';
        s += v;
        s += '\n';
    };
    s += kIndexMagic;
    s += '\n';
    kv("store_name", ix.store_name);
    kv("store_bytes", std::to_string(ix.store_bytes));
    kv("store_fnv", Hex16(ix.store_fnv));
    kv("schema", ix.schema);
    kv("ell", std::to_string(ix.ell));
    kv("n_max", std::to_string(ix.n_max));
    kv("cells", std::to_string(ix.cells.size()));
    kv("systems", std::to_string(ix.systems.size()));
    kv("occurrences", std::to_string(ix.occ.size()));
    kv("head", std::to_string(ix.head_off) + "\t" + std::to_string(ix.head_len));
    kv("tail", std::to_string(ix.tail_off) + "\t" + std::to_string(ix.tail_len));
    for (const CellRow& c : ix.cells)
        kv("C", std::to_string(c.n) + "\t" + std::to_string(c.off) + "\t" + std::to_string(c.len));
    for (const SysRow& r : ix.systems) {
        s += "S\t";
        s += std::to_string(r.off);
        s += '\t';
        s += std::to_string(r.len);
        s += '\t';
        s += std::to_string(r.r);
        s += '\t';
        s += r.type;
        s += '\t';
        s += r.collected ? '1' : '0';
        s += '\t';
        s += std::to_string(r.base);
        s += '\t';
        s += std::to_string(r.twist);
        s += '\n';
    }
    for (const OccRow& o : ix.occ)
        kv("O", std::to_string(o.n) + "\t" + std::to_string(o.ideal) + "\t" + std::to_string(o.sid));
    kv("end", Hex16(FnvOf(s.data(), s.size())));
    return s;
}

bool Parse(const std::string& text, Index& out, std::string& error) {
    Index ix;
    const char* b = text.data();
    const char* e = b + text.size();
    const char* p = b;
    std::size_t line_no = 0;
    // Next line [ls, le) without the '\n'; false at the end / no '\n'.
    auto next = [&](const char*& ls, const char*& le) {
        if (p >= e) return false;
        const char* nl = static_cast<const char*>(std::memchr(p, '\n', static_cast<std::size_t>(e - p)));
        if (nl == nullptr) return false;
        ls = p;
        le = nl;
        p = nl + 1;
        ++line_no;
        return true;
    };
    auto bad = [&](const std::string& what) {
        error = "index line " + std::to_string(line_no) + ": " + what;
        return false;
    };
    const char *ls = nullptr, *le = nullptr;
    if (!next(ls, le) || std::string(ls, le) != kIndexMagic) return bad(std::string("the magic line is not ") + kIndexMagic);
    auto key = [&](const char* k, std::vector<std::string>& f, std::size_t arity) {
        if (!next(ls, le)) return bad(std::string("missing line '") + k + "'");
        f = Fields(ls, le);
        if (f.size() != arity + 1 || f[0] != k) return bad(std::string("expected '") + k + "' with " + std::to_string(arity) + " field(s)");
        return true;
    };
    std::vector<std::string> f;
    std::uint64_t n_cells = 0, n_sys = 0, n_occ = 0;
    if (!key("store_name", f, 1)) return false;
    ix.store_name = f[1];
    if (!key("store_bytes", f, 1) || !ToU64(f[1], ix.store_bytes)) return bad("bad store_bytes");
    if (!key("store_fnv", f, 1) || f[1].size() != 18 || !ParseHex64(f[1], ix.store_fnv)) return bad("bad store_fnv");
    if (!key("schema", f, 1)) return false;
    ix.schema = f[1];
    if (!key("ell", f, 1) || !ToU32(f[1], ix.ell)) return bad("bad ell");
    if (!key("n_max", f, 1) || !ToU32(f[1], ix.n_max)) return bad("bad n_max");
    if (!key("cells", f, 1) || !ToU64(f[1], n_cells)) return bad("bad cells");
    if (!key("systems", f, 1) || !ToU64(f[1], n_sys)) return bad("bad systems");
    if (!key("occurrences", f, 1) || !ToU64(f[1], n_occ)) return bad("bad occurrences");
    if (!key("head", f, 2) || !ToU64(f[1], ix.head_off) || !ToU64(f[2], ix.head_len)) return bad("bad head");
    if (!key("tail", f, 2) || !ToU64(f[1], ix.tail_off) || !ToU64(f[2], ix.tail_len)) return bad("bad tail");
    if (n_cells > text.size() || n_sys > text.size() || n_occ > text.size()) return bad("a count larger than the index");
    ix.cells.reserve(static_cast<std::size_t>(n_cells));
    for (std::uint64_t i = 0; i < n_cells; ++i) {
        CellRow c;
        if (!key("C", f, 3) || !ToU32(f[1], c.n) || !ToU64(f[2], c.off) || !ToU64(f[3], c.len)) return bad("bad C line");
        ix.cells.push_back(c);
    }
    ix.systems.reserve(static_cast<std::size_t>(n_sys));
    for (std::uint64_t i = 0; i < n_sys; ++i) {
        SysRow r;
        if (!key("S", f, 7) || !ToU64(f[1], r.off) || !ToU64(f[2], r.len) || !ToU32(f[3], r.r) ||
            f[4].size() != 1 || (f[4][0] != 'e' && f[4][0] != 'c') || (f[5] != "0" && f[5] != "1") ||
            !ToU32(f[6], r.base) || !ToU32(f[7], r.twist))
            return bad("bad S line");
        r.type = f[4][0];
        r.collected = f[5] == "1" ? 1 : 0;
        ix.systems.push_back(r);
    }
    ix.occ.reserve(static_cast<std::size_t>(n_occ));
    for (std::uint64_t i = 0; i < n_occ; ++i) {
        OccRow o;
        if (!key("O", f, 3) || !ToU32(f[1], o.n) || !ToU32(f[2], o.ideal) || !ToU32(f[3], o.sid)) return bad("bad O line");
        ix.occ.push_back(o);
    }
    const std::size_t end_at = static_cast<std::size_t>(p - b);
    if (!key("end", f, 1)) return false;
    std::uint64_t h = 0;
    if (f[1].size() != 18 || !ParseHex64(f[1], h)) return bad("bad end hash");
    if (h != FnvOf(b, end_at)) return bad("the end hash does not match the index's bytes (a corrupt or edited index)");
    if (p != e) return bad("bytes after the end line");
    out = std::move(ix);
    return true;
}

bool Equal(const Index& a, const Index& b, std::string& diff) {
    auto d = [&](const std::string& what) {
        diff = what;
        return false;
    };
    if (a.store_name != b.store_name) return d("store_name " + a.store_name + " != " + b.store_name);
    if (a.store_bytes != b.store_bytes) return d("store_bytes " + std::to_string(a.store_bytes) + " != " + std::to_string(b.store_bytes));
    if (a.store_fnv != b.store_fnv) return d("store_fnv " + Hex16(a.store_fnv) + " != " + Hex16(b.store_fnv));
    if (a.schema != b.schema) return d("schema");
    if (a.ell != b.ell) return d("ell");
    if (a.n_max != b.n_max) return d("n_max");
    if (a.head_off != b.head_off || a.head_len != b.head_len) return d("head");
    if (a.tail_off != b.tail_off || a.tail_len != b.tail_len) return d("tail");
    if (a.cells.size() != b.cells.size()) return d("the number of cells");
    if (a.systems.size() != b.systems.size()) return d("the number of systems");
    if (a.occ.size() != b.occ.size()) return d("the number of occurrences");
    for (std::size_t i = 0; i < a.cells.size(); ++i) {
        const CellRow &x = a.cells[i], &y = b.cells[i];
        if (x.n != y.n || x.off != y.off || x.len != y.len) return d("cell line " + std::to_string(i));
    }
    for (std::size_t i = 0; i < a.systems.size(); ++i) {
        const SysRow &x = a.systems[i], &y = b.systems[i];
        if (x.off != y.off || x.len != y.len || x.r != y.r || x.type != y.type || x.collected != y.collected ||
            x.base != y.base || x.twist != y.twist)
            return d("system line " + std::to_string(i));
    }
    for (std::size_t i = 0; i < a.occ.size(); ++i) {
        const OccRow &x = a.occ[i], &y = b.occ[i];
        if (x.n != y.n || x.ideal != y.ideal || x.sid != y.sid) return d("occurrence line " + std::to_string(i));
    }
    return true;
}

// ---------------------------------------------------------------------------
MappedFile::~MappedFile() {
    if (data_ != nullptr) ::munmap(const_cast<char*>(data_), size_);
    if (fd_ >= 0) ::close(fd_);
}

bool MappedFile::open(const std::string& path, std::string& error) {
    if (fd_ >= 0) {
        error = "already open";
        return false;
    }
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        error = "cannot open " + path + " for reading: " + std::strerror(errno);
        return false;
    }
    struct stat st;
    if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
        ::close(fd);
        error = path + " is not a regular file";
        return false;
    }
    if (st.st_size <= 0) {
        ::close(fd);
        error = path + " is empty";
        return false;
    }
    void* m = ::mmap(nullptr, static_cast<std::size_t>(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
    if (m == MAP_FAILED) {
        ::close(fd);
        error = "cannot map " + path + ": " + std::strerror(errno);
        return false;
    }
    fd_ = fd;
    data_ = static_cast<const char*>(m);
    size_ = static_cast<std::size_t>(st.st_size);
    return true;
}

void MappedFile::advise_sequential() const {
    if (data_ != nullptr) ::madvise(const_cast<char*>(data_), size_, MADV_SEQUENTIAL);
}

void MappedFile::release_resident() const {
    if (data_ != nullptr) ::madvise(const_cast<char*>(data_), size_, MADV_DONTNEED);
}

} // namespace atlas_index
