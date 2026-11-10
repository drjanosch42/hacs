// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Lexer.h -- console C0: tokens and the scanner.
//
// Normative source: plan Sec. 3.1 ("Lexical").
//
//   * Whitespace separates tokens; NEWLINE terminates a statement UNLESS the
//     round/square brackets are unbalanced or the line ends in `\`.  Curly
//     braces do NOT suspend the rule -- statements inside a block are
//     newline-terminated -- so the scanner tracks only ( and [ depth.
//   * Comments `#` or `//` to end of line (the campaign spec's convention,
//     CampaignSpec.h:16).
//   * Identifiers [A-Za-z_][A-Za-z0-9_]*, case-sensitive.  Reserved:
//     for in if else set help quit true false and or not expect_error include
//     args def   (`def` reserved for owner question Q2).
//   * Integers: decimal, 0x hex, `_` separators (134_217_757).  Type Int is
//     int64_t; a literal outside [-2^63, 2^63-1] is a PARSE error.
//   * Strings: "..." with \" \\ \n \t.  No interpolation.
//
// STL-ONLY (plan Sec. 2 C8).
// ===========================================================================
#pragma once
#ifndef CONSOLE_LEXER_H
#define CONSOLE_LEXER_H

#include <cstdint>
#include <string>
#include <vector>

#include "ConsoleError.h"

namespace console {

enum class TokenKind : std::uint8_t {
    End = 0,     // end of input
    Newline,     // a statement terminator produced by a newline
    Semicolon,   // ';' -- terminator that also suppresses printing
    Int,
    String,
    Ident,
    Keyword,
    // punctuation / operators
    LParen, RParen, LBracket, RBracket, LBrace, RBrace,
    Comma, Dot, Assign, Question,
    Plus, Minus, Star, Slash, Percent, Caret,
    Eq, Ne, Lt, Le, Gt, Ge
};

const char* TokenKindName(TokenKind kind);

struct Token {
    TokenKind kind = TokenKind::End;
    std::string text;            // identifier / keyword spelling, or the decoded string body
    std::int64_t int_value = 0;
    SourcePos pos;
};

// True for the reserved words of plan Sec. 3.1.
bool IsReservedWord(const std::string& word);

// Scans `source` into tokens.  `file` is the name that appears in every
// position.  Throws ParseError (with the offending line and a caret) on an
// illegal character, an unterminated string, a bad escape or an integer
// literal outside int64_t.
std::vector<Token> Tokenize(const std::string& source, const std::string& file);

// The 1-based `line`-th line of `source`, for error carets.  Returns "" when
// the line does not exist.
std::string SourceLine(const std::string& source, int line);

// True when `source` cannot be a complete statement yet because a bracket is
// still open or the last line ends in a backslash -- the REPL's continuation
// test (plan Sec. 7, prompt "... ").  Never throws: an unterminated string
// counts as "incomplete".
bool NeedsContinuation(const std::string& source);

} // namespace console

#endif // CONSOLE_LEXER_H
