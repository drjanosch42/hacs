// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Lexer.cpp -- see Lexer.h.  EBNF reference: plan Sec. 3.1 / 3.2.
// ===========================================================================

#include "Lexer.h"

#include <limits>

namespace console {

const char* TokenKindName(TokenKind kind) {
    switch (kind) {
        case TokenKind::End:       return "end of input";
        case TokenKind::Newline:   return "end of line";
        case TokenKind::Semicolon: return "';'";
        case TokenKind::Int:       return "integer";
        case TokenKind::String:    return "string";
        case TokenKind::Ident:     return "identifier";
        case TokenKind::Keyword:   return "keyword";
        case TokenKind::LParen:    return "'('";
        case TokenKind::RParen:    return "')'";
        case TokenKind::LBracket:  return "'['";
        case TokenKind::RBracket:  return "']'";
        case TokenKind::LBrace:    return "'{'";
        case TokenKind::RBrace:    return "'}'";
        case TokenKind::Comma:     return "','";
        case TokenKind::Dot:       return "'.'";
        case TokenKind::Assign:    return "'='";
        case TokenKind::Question:  return "'?'";
        case TokenKind::Plus:      return "'+'";
        case TokenKind::Minus:     return "'-'";
        case TokenKind::Star:      return "'*'";
        case TokenKind::Slash:     return "'/'";
        case TokenKind::Percent:   return "'%'";
        case TokenKind::Caret:     return "'^'";
        case TokenKind::Eq:        return "'=='";
        case TokenKind::Ne:        return "'!='";
        case TokenKind::Lt:        return "'<'";
        case TokenKind::Le:        return "'<='";
        case TokenKind::Gt:        return "'>'";
        case TokenKind::Ge:        return "'>='";
    }
    return "token";
}

namespace {

// Plan Sec. 3.1, verbatim, in the order the plan lists them.
const char* const kReserved[] = {
    "for", "in", "if", "else", "set", "help", "quit", "true", "false",
    "and", "or", "not", "expect_error", "include", "args", "def"
};

bool IsIdentStart(char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_';
}
bool IsIdentChar(char c) {
    return IsIdentStart(c) || (c >= '0' && c <= '9');
}
bool IsDigit(char c) { return c >= '0' && c <= '9'; }
bool IsHexDigit(char c) {
    return IsDigit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}
int HexValue(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return c - 'A' + 10;
}

class Scanner {
public:
    Scanner(const std::string& src, const std::string& file) : src_(src), file_(file) {}

    std::vector<Token> run() {
        std::vector<Token> out;
        while (true) {
            skip_blanks_and_comments();
            if (at_end()) break;
            const char c = src_[i_];
            if (c == '\n') {
                const SourcePos p = here();
                advance();
                // A newline inside an unbalanced ( or [ is pure whitespace.
                if (depth_ == 0) push(out, TokenKind::Newline, p);
                continue;
            }
            scan_token(out);
        }
        Token end;
        end.kind = TokenKind::End;
        end.pos = here();
        out.push_back(end);
        return out;
    }

private:
    bool at_end() const { return i_ >= src_.size(); }
    char peek(std::size_t off = 0) const {
        return (i_ + off < src_.size()) ? src_[i_ + off] : '\0';
    }
    SourcePos here() const { return SourcePos(file_, line_, col_); }

    void advance() {
        if (at_end()) return;
        if (src_[i_] == '\n') {
            ++line_;
            col_ = 1;
        } else {
            ++col_;
        }
        ++i_;
    }

    [[noreturn]] void fail(const std::string& msg, const SourcePos& pos) const {
        throw ParseError(msg, pos, SourceLine(src_, pos.line));
    }

    void push(std::vector<Token>& out, TokenKind kind, const SourcePos& pos,
              std::string text = std::string(), std::int64_t iv = 0) {
        Token t;
        t.kind = kind;
        t.pos = pos;
        t.text = std::move(text);
        t.int_value = iv;
        out.push_back(std::move(t));
    }

    void skip_blanks_and_comments() {
        while (!at_end()) {
            const char c = src_[i_];
            if (c == ' ' || c == '\t' || c == '\r') {
                advance();
            } else if (c == '\\') {
                // Line continuation: a backslash followed (up to blanks) by a
                // newline swallows that newline.
                std::size_t j = i_ + 1;
                while (j < src_.size() && (src_[j] == ' ' || src_[j] == '\t' || src_[j] == '\r')) ++j;
                if (j < src_.size() && src_[j] == '\n') {
                    while (i_ <= j) advance();
                } else {
                    return;   // a stray backslash: let scan_token report it
                }
            } else if (c == '#') {
                while (!at_end() && src_[i_] != '\n') advance();
            } else if (c == '/' && peek(1) == '/') {
                while (!at_end() && src_[i_] != '\n') advance();
            } else {
                return;
            }
        }
    }

    void scan_token(std::vector<Token>& out) {
        const SourcePos pos = here();
        const char c = src_[i_];

        if (IsIdentStart(c)) {
            std::string word;
            while (!at_end() && IsIdentChar(src_[i_])) {
                word += src_[i_];
                advance();
            }
            push(out, IsReservedWord(word) ? TokenKind::Keyword : TokenKind::Ident, pos, word);
            return;
        }
        if (IsDigit(c)) {
            scan_int(out, pos);
            return;
        }
        if (c == '"') {
            scan_string(out, pos);
            return;
        }

        switch (c) {
            case ';': advance(); push(out, TokenKind::Semicolon, pos); return;
            case '(': advance(); ++depth_; push(out, TokenKind::LParen, pos); return;
            case ')': advance(); if (depth_ > 0) --depth_; push(out, TokenKind::RParen, pos); return;
            case '[': advance(); ++depth_; push(out, TokenKind::LBracket, pos); return;
            case ']': advance(); if (depth_ > 0) --depth_; push(out, TokenKind::RBracket, pos); return;
            case '{': advance(); push(out, TokenKind::LBrace, pos); return;
            case '}': advance(); push(out, TokenKind::RBrace, pos); return;
            case ',': advance(); push(out, TokenKind::Comma, pos); return;
            case '.': advance(); push(out, TokenKind::Dot, pos); return;
            case '?': advance(); push(out, TokenKind::Question, pos); return;
            case '+': advance(); push(out, TokenKind::Plus, pos); return;
            case '-': advance(); push(out, TokenKind::Minus, pos); return;
            case '*': advance(); push(out, TokenKind::Star, pos); return;
            case '/': advance(); push(out, TokenKind::Slash, pos); return;
            case '%': advance(); push(out, TokenKind::Percent, pos); return;
            case '^': advance(); push(out, TokenKind::Caret, pos); return;
            case '=':
                advance();
                if (peek() == '=') { advance(); push(out, TokenKind::Eq, pos); }
                else               { push(out, TokenKind::Assign, pos); }
                return;
            case '!':
                advance();
                if (peek() == '=') { advance(); push(out, TokenKind::Ne, pos); return; }
                fail("unexpected '!' (did you mean '!=' or 'not'?)", pos);
            case '<':
                advance();
                if (peek() == '=') { advance(); push(out, TokenKind::Le, pos); }
                else               { push(out, TokenKind::Lt, pos); }
                return;
            case '>':
                advance();
                if (peek() == '=') { advance(); push(out, TokenKind::Ge, pos); }
                else               { push(out, TokenKind::Gt, pos); }
                return;
            default:
                break;
        }
        std::string msg = "unexpected character '";
        msg += c;
        msg += "'";
        fail(msg, pos);
    }

    void scan_int(std::vector<Token>& out, const SourcePos& pos) {
        std::string digits;
        bool hex = false;
        if (src_[i_] == '0' && (peek(1) == 'x' || peek(1) == 'X')) {
            hex = true;
            advance();
            advance();
            while (!at_end() && (IsHexDigit(src_[i_]) || src_[i_] == '_')) {
                if (src_[i_] != '_') digits += src_[i_];
                advance();
            }
            if (digits.empty()) fail("hex literal has no digits after '0x'", pos);
        } else {
            while (!at_end() && (IsDigit(src_[i_]) || src_[i_] == '_')) {
                if (src_[i_] != '_') digits += src_[i_];
                advance();
            }
        }
        // A literal immediately followed by an identifier character is a typo,
        // not two tokens ("12abc").
        if (!at_end() && IsIdentStart(src_[i_])) {
            fail("invalid character in integer literal", here());
        }

        // Accumulate in uint64_t and reject anything above 2^63 - 1.  The
        // grammar has no negative literals (unary minus is an operator), so the
        // -2^63 end is reached as -(2^63) and is checked by the evaluator.
        const std::uint64_t limit = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
        std::uint64_t acc = 0;
        const std::uint64_t base = hex ? 16u : 10u;
        for (char d : digits) {
            const std::uint64_t dv = static_cast<std::uint64_t>(hex ? HexValue(d) : (d - '0'));
            if (acc > (limit - dv) / base) {
                fail("integer literal does not fit in Int (int64_t)", pos);
            }
            acc = acc * base + dv;
        }
        push(out, TokenKind::Int, pos, digits, static_cast<std::int64_t>(acc));
    }

    void scan_string(std::vector<Token>& out, const SourcePos& pos) {
        advance();   // the opening quote
        std::string body;
        while (true) {
            if (at_end() || src_[i_] == '\n') fail("unterminated string literal", pos);
            const char c = src_[i_];
            if (c == '"') {
                advance();
                break;
            }
            if (c == '\\') {
                const SourcePos esc = here();
                advance();
                if (at_end()) fail("unterminated string literal", pos);
                const char e = src_[i_];
                switch (e) {
                    case '"':  body += '"';  break;
                    case '\\': body += '\\'; break;
                    case 'n':  body += '\n'; break;
                    case 't':  body += '\t'; break;
                    default: {
                        std::string msg = "unknown escape '\\";
                        msg += e;
                        msg += "' (only \\\" \\\\ \\n \\t are defined)";
                        fail(msg, esc);
                    }
                }
                advance();
                continue;
            }
            body += c;
            advance();
        }
        push(out, TokenKind::String, pos, body);
    }

    const std::string& src_;
    const std::string& file_;
    std::size_t i_ = 0;
    int line_ = 1;
    int col_ = 1;
    int depth_ = 0;   // ( and [ only
};

} // namespace

bool IsReservedWord(const std::string& word) {
    for (const char* w : kReserved) {
        if (word == w) return true;
    }
    return false;
}

std::vector<Token> Tokenize(const std::string& source, const std::string& file) {
    Scanner s(source, file);
    return s.run();
}

std::string SourceLine(const std::string& source, int line) {
    if (line <= 0) return std::string();
    int current = 1;
    std::size_t start = 0;
    for (std::size_t i = 0; i <= source.size(); ++i) {
        if (i == source.size() || source[i] == '\n') {
            if (current == line) {
                std::string out = source.substr(start, i - start);
                if (!out.empty() && out.back() == '\r') out.pop_back();
                return out;
            }
            ++current;
            start = i + 1;
        }
    }
    return std::string();
}

bool NeedsContinuation(const std::string& source) {
    int depth = 0;
    bool in_string = false;
    bool trailing_backslash = false;
    for (std::size_t i = 0; i < source.size(); ++i) {
        const char c = source[i];
        if (in_string) {
            if (c == '\\' && i + 1 < source.size()) { ++i; continue; }
            if (c == '"') in_string = false;
            if (c == '\n') { in_string = false; }   // unterminated: stop tracking
            continue;
        }
        if (c == '#') {
            while (i < source.size() && source[i] != '\n') ++i;
            continue;
        }
        if (c == '/' && i + 1 < source.size() && source[i + 1] == '/') {
            while (i < source.size() && source[i] != '\n') ++i;
            continue;
        }
        if (c == '"') { in_string = true; continue; }
        if (c == '(' || c == '[' || c == '{') ++depth;
        else if (c == ')' || c == ']' || c == '}') { if (depth > 0) --depth; }
    }
    // A trailing backslash (possibly followed by blanks) continues the line.
    for (std::size_t j = source.size(); j > 0; --j) {
        const char c = source[j - 1];
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') continue;
        trailing_backslash = (c == '\\');
        break;
    }
    return depth > 0 || in_string || trailing_backslash;
}

} // namespace console
