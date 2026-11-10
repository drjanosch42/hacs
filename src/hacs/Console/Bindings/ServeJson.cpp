// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// ServeJson.cpp -- see ServeJson.h.
// ===========================================================================

#include "ServeJson.h"

#include "../Core/Printer.h"

#include <cerrno>
#include <cstdlib>

namespace console {
namespace serve_json {

namespace {

constexpr std::uint64_t kMaxSafe = 9007199254740991ull;   // 2^53 - 1

struct Parser {
    const std::string& s;
    std::size_t i = 0;
    std::string err;
    int max_depth;

    Parser(const std::string& text, int depth) : s(text), max_depth(depth) {}

    bool fail(const std::string& why) {
        if (err.empty()) err = why + " at byte " + std::to_string(i);
        return false;
    }
    void ws() {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
    }
    bool lit(const char* word) {
        std::size_t k = 0;
        while (word[k] != '\0') {
            if (i + k >= s.size() || s[i + k] != word[k]) return fail("invalid literal");
            ++k;
        }
        i += k;
        return true;
    }
    static void put_utf8(std::string& o, std::uint32_t cp) {
        if (cp < 0x80) {
            o += static_cast<char>(cp);
        } else if (cp < 0x800) {
            o += static_cast<char>(0xC0 | (cp >> 6));
            o += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            o += static_cast<char>(0xE0 | (cp >> 12));
            o += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            o += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            o += static_cast<char>(0xF0 | (cp >> 18));
            o += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            o += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            o += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }
    bool hex4(std::uint32_t& out) {
        if (i + 4 > s.size()) return fail("truncated \\u escape");
        out = 0;
        for (int k = 0; k < 4; ++k) {
            const char c = s[i++];
            out <<= 4;
            if (c >= '0' && c <= '9') out |= static_cast<std::uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f') out |= static_cast<std::uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') out |= static_cast<std::uint32_t>(c - 'A' + 10);
            else return fail("bad hex digit in \\u escape");
        }
        return true;
    }
    bool string(std::string& o) {
        ++i;   // opening quote
        while (true) {
            if (i >= s.size()) return fail("unterminated string");
            const unsigned char c = static_cast<unsigned char>(s[i]);
            if (c == '"') { ++i; return true; }
            if (c < 0x20) return fail("raw control character in string");
            if (c != '\\') { o += static_cast<char>(c); ++i; continue; }
            ++i;
            if (i >= s.size()) return fail("unterminated escape");
            const char e = s[i++];
            switch (e) {
                case '"': o += '"'; break;
                case '\\': o += '\\'; break;
                case '/': o += '/'; break;
                case 'b': o += '\b'; break;
                case 'f': o += '\f'; break;
                case 'n': o += '\n'; break;
                case 'r': o += '\r'; break;
                case 't': o += '\t'; break;
                case 'u': {
                    std::uint32_t cp = 0;
                    if (!hex4(cp)) return false;
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        if (i + 6 > s.size() || s[i] != '\\' || s[i + 1] != 'u') return fail("lone high surrogate");
                        i += 2;
                        std::uint32_t lo = 0;
                        if (!hex4(lo)) return false;
                        if (lo < 0xDC00 || lo > 0xDFFF) return fail("bad low surrogate");
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                        return fail("lone low surrogate");
                    }
                    put_utf8(o, cp);
                    break;
                }
                default: return fail("bad escape");
            }
        }
    }
    bool number(Json& out) {
        const std::size_t b = i;
        if (s[i] == '-') ++i;
        if (i >= s.size()) return fail("bad number");
        if (s[i] == '0') {
            ++i;
        } else if (s[i] >= '1' && s[i] <= '9') {
            while (i < s.size() && s[i] >= '0' && s[i] <= '9') ++i;
        } else {
            return fail("bad number");
        }
        if (i < s.size() && s[i] == '.') {
            ++i;
            if (i >= s.size() || s[i] < '0' || s[i] > '9') return fail("bad fraction");
            while (i < s.size() && s[i] >= '0' && s[i] <= '9') ++i;
        }
        if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
            ++i;
            if (i < s.size() && (s[i] == '+' || s[i] == '-')) ++i;
            if (i >= s.size() || s[i] < '0' || s[i] > '9') return fail("bad exponent");
            while (i < s.size() && s[i] >= '0' && s[i] <= '9') ++i;
        }
        out.kind = Json::Kind::Number;
        out.text = s.substr(b, i - b);
        return true;
    }
    bool value(Json& out, int depth) {
        if (depth > max_depth) return fail("nesting too deep");
        ws();
        if (i >= s.size()) return fail("unexpected end of input");
        const char c = s[i];
        if (c == '{') {
            out.kind = Json::Kind::Object;
            ++i;
            ws();
            if (i < s.size() && s[i] == '}') { ++i; return true; }
            while (true) {
                ws();
                if (i >= s.size() || s[i] != '"') return fail("expected a member name");
                std::string key;
                if (!string(key)) return false;
                ws();
                if (i >= s.size() || s[i] != ':') return fail("expected ':'");
                ++i;
                Json v;
                if (!value(v, depth + 1)) return false;
                out.members.emplace_back(std::move(key), std::move(v));
                ws();
                if (i < s.size() && s[i] == ',') { ++i; continue; }
                if (i < s.size() && s[i] == '}') { ++i; return true; }
                return fail("expected ',' or '}'");
            }
        }
        if (c == '[') {
            out.kind = Json::Kind::Array;
            ++i;
            ws();
            if (i < s.size() && s[i] == ']') { ++i; return true; }
            while (true) {
                Json v;
                if (!value(v, depth + 1)) return false;
                out.items.push_back(std::move(v));
                ws();
                if (i < s.size() && s[i] == ',') { ++i; continue; }
                if (i < s.size() && s[i] == ']') { ++i; return true; }
                return fail("expected ',' or ']'");
            }
        }
        if (c == '"') {
            out.kind = Json::Kind::String;
            return string(out.text);
        }
        if (c == 't') { out.kind = Json::Kind::Bool; out.b = true; return lit("true"); }
        if (c == 'f') { out.kind = Json::Kind::Bool; out.b = false; return lit("false"); }
        if (c == 'n') { out.kind = Json::Kind::Null; return lit("null"); }
        if (c == '-' || (c >= '0' && c <= '9')) return number(out);
        return fail("unexpected character");
    }
};

bool IsIntegerLiteral(const std::string& t) {
    if (t.empty()) return false;
    for (std::size_t k = (t[0] == '-' ? 1 : 0); k < t.size(); ++k)
        if (t[k] < '0' || t[k] > '9') return false;
    return t.size() > (t[0] == '-' ? 1u : 0u);
}

// |t| <= 2^53 - 1 for an integer literal of any length.
bool FitsSafe(const std::string& t) {
    const std::string mag = (t[0] == '-') ? t.substr(1) : t;
    if (mag.size() > 16) return false;
    return std::strtoull(mag.c_str(), nullptr, 10) <= kMaxSafe;
}

void Write(const Json& j, std::string& o) {
    switch (j.kind) {
        case Json::Kind::Null: o += "null"; return;
        case Json::Kind::Bool: o += j.b ? "true" : "false"; return;
        case Json::Kind::Number:
            if (IsIntegerLiteral(j.text) && !FitsSafe(j.text)) o += "\"" + j.text + "\"";
            else o += j.text;
            return;
        case Json::Kind::String: o += Quote(j.text); return;
        case Json::Kind::Array: {
            o += '[';
            for (std::size_t k = 0; k < j.items.size(); ++k) {
                if (k != 0) o += ',';
                Write(j.items[k], o);
            }
            o += ']';
            return;
        }
        case Json::Kind::Object: {
            o += '{';
            for (std::size_t k = 0; k < j.members.size(); ++k) {
                if (k != 0) o += ',';
                o += Quote(j.members[k].first);
                o += ':';
                Write(j.members[k].second, o);
            }
            o += '}';
            return;
        }
    }
}

} // namespace

const Json* Json::get(const std::string& key) const {
    for (const auto& m : members)
        if (m.first == key) return &m.second;
    return nullptr;
}

bool Json::as_int64(std::int64_t& out) const {
    if (kind != Kind::Number || !IsIntegerLiteral(text)) return false;
    errno = 0;
    char* end = nullptr;
    const long long v = std::strtoll(text.c_str(), &end, 10);
    if (errno != 0 || end == nullptr || *end != '\0') return false;
    out = static_cast<std::int64_t>(v);
    return true;
}

bool ParseJson(const std::string& text, Json& out, std::string& error, int max_depth) {
    Parser p(text, max_depth);
    out = Json();
    if (!p.value(out, 0)) {
        error = p.err;
        return false;
    }
    p.ws();
    if (p.i != text.size()) {
        p.fail("trailing characters");
        error = p.err;
        return false;
    }
    return true;
}

std::string WriteJson(const Json& j) {
    std::string o;
    Write(j, o);
    return o;
}

std::string Quote(const std::string& s) {
    return "\"" + JsonEscape(s) + "\"";
}

std::string IntJson(std::int64_t v) {
    const std::uint64_t mag = v < 0 ? (0ull - static_cast<std::uint64_t>(v)) : static_cast<std::uint64_t>(v);
    if (mag <= kMaxSafe) return std::to_string(v);
    return "\"" + std::to_string(v) + "\"";
}

std::string UIntJson(std::uint64_t v) {
    if (v <= kMaxSafe) return std::to_string(v);
    return "\"" + std::to_string(v) + "\"";
}

} // namespace serve_json
} // namespace console
