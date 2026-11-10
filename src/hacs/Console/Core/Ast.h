// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Ast.h -- console C0: the syntax tree of the grammar in plan Sec. 3.2.
//
// The EBNF this mirrors, verbatim:
//
//   program    := { statement }
//   statement  := assign | exprstmt | for | if | setstmt | include | help | expect
//   assign     := IDENT "=" expr TERM
//   exprstmt   := expr TERM
//   for        := "for" IDENT "in" expr block
//   if         := "if" expr block [ "else" ( block | if ) ]
//   setstmt    := "set" IDENT "=" expr TERM
//   include    := "include" "(" STRING ")" TERM
//   help       := ( "help" [ IDENT | "?" ] | "?" IDENT ) TERM
//   expect     := "expect_error" STRING block
//   block      := "{" { statement } "}"
//   TERM       := NEWLINE | ";"
//   expr       := orexpr
//   orexpr     := andexpr { "or" andexpr }
//   andexpr    := notexpr { "and" notexpr }
//   notexpr    := "not" notexpr | cmp
//   cmp        := arith [ ("==" | "!=" | "<" | "<=" | ">" | ">=") arith ]
//   arith      := term { ("+" | "-") term }
//   term       := power { ("*" | "/" | "%") power }
//   power      := unary [ "^" power ]                         (right-assoc)
//   unary      := "-" unary | postfix
//   postfix    := primary { "(" args ")" | "." IDENT | "[" expr "]" }
//   primary    := INT | STRING | "true" | "false" | IDENT | "args" | "(" expr ")"
//               | "[" [ expr { "," expr } ] "]"
//   args       := [ arg { "," arg } ]           arg := expr | IDENT "=" expr
//
// One deliberate addition, recorded as a C0 deviation in the WP-5 report:
// `quit` is parsed as a statement of its own.  Plan Sec. 3.1 reserves the word
// and Sec. 6 gives it a meaning ("The REPL exits 0 on quit/EOF"), but Sec. 3.2
// lists no production for it; since a reserved word can never be an identifier
// or a verb name, a statement is the only place it can go.
//
// STL-ONLY (plan Sec. 2 C8).
// ===========================================================================
#pragma once
#ifndef CONSOLE_AST_H
#define CONSOLE_AST_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ConsoleError.h"

namespace console {

// --- Expressions -----------------------------------------------------------
enum class ExprKind : std::uint8_t {
    IntLit, StringLit, BoolLit, Ident, Args, ListLit,
    Unary, Binary, Logical, Compare, Call, Member, MethodCall, Index
};

enum class BinaryOp : std::uint8_t { Add, Sub, Mul, Div, Mod, Pow };
enum class CompareOp : std::uint8_t { Eq, Ne, Lt, Le, Gt, Ge };
enum class LogicalOp : std::uint8_t { And, Or, Not };

const char* BinaryOpName(BinaryOp op);
const char* CompareOpName(CompareOp op);

struct Expr;
using ExprPtr = std::shared_ptr<const Expr>;

// A call argument: positional (name empty) or keyword (`IDENT "=" expr`).
struct Argument {
    std::string name;
    ExprPtr value;
    SourcePos pos;
};

struct Expr {
    ExprKind kind = ExprKind::IntLit;
    SourcePos pos;

    // Literals / names.
    std::int64_t int_value = 0;
    bool bool_value = false;
    std::string text;              // string body, identifier, member/method name

    // Operators.
    BinaryOp binary_op = BinaryOp::Add;
    CompareOp compare_op = CompareOp::Eq;
    LogicalOp logical_op = LogicalOp::And;

    ExprPtr lhs;                   // unary/binary/compare/logical operand, callee base
    ExprPtr rhs;
    std::vector<ExprPtr> items;    // list literal
    std::vector<Argument> args;    // call / method call

    // The source text of this expression, kept for error messages and for the
    // creation-site note of a handle ("created at f.oc:9:7 by hecke(H, p = 2)",
    // plan Sec. 2 C7).
    std::string source_text;
};

// --- Statements ------------------------------------------------------------
enum class StmtKind : std::uint8_t {
    Assign, ExprStmt, For, If, Set, Include, Help, ExpectError, Quit
};

struct Stmt;
using StmtPtr = std::shared_ptr<const Stmt>;

struct Stmt {
    StmtKind kind = StmtKind::ExprStmt;
    SourcePos pos;

    std::string name;              // assign target, for-variable, set key, help topic
    ExprPtr value;                 // assigned/printed/iterated/condition expression
    bool suppress_print = false;   // an exprstmt terminated by ';' prints nothing
    std::string text;              // include path, expect_error substring

    std::vector<StmtPtr> body;     // for/if-then/expect_error block
    std::vector<StmtPtr> orelse;   // if-else block (an `else if` is a one-statement block)
};

struct Program {
    std::vector<StmtPtr> statements;
    std::string file;
    std::string source;            // kept so a runtime error can show its line
};

} // namespace console

#endif // CONSOLE_AST_H
