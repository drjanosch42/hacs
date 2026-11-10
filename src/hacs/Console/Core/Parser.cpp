// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Parser.cpp -- see Parser.h.  Each function below carries the EBNF production
// it implements, so plan Sec. 3.2 and the code can be diffed line by line
// (Phase C0 acceptance (d)).
// ===========================================================================

#include "Parser.h"

#include "Lexer.h"

#include <string>
#include <vector>

namespace console {

const char* BinaryOpName(BinaryOp op) {
    switch (op) {
        case BinaryOp::Add: return "+";
        case BinaryOp::Sub: return "-";
        case BinaryOp::Mul: return "*";
        case BinaryOp::Div: return "/";
        case BinaryOp::Mod: return "%";
        case BinaryOp::Pow: return "^";
    }
    return "?";
}

const char* CompareOpName(CompareOp op) {
    switch (op) {
        case CompareOp::Eq: return "==";
        case CompareOp::Ne: return "!=";
        case CompareOp::Lt: return "<";
        case CompareOp::Le: return "<=";
        case CompareOp::Gt: return ">";
        case CompareOp::Ge: return ">=";
    }
    return "?";
}

namespace {

class ParserImpl {
public:
    ParserImpl(const std::string& source, const std::string& file)
        : source_(source), file_(file), tokens_(Tokenize(source, file)) {
        // Offsets of the first character of every line, so an expression's
        // source text can be sliced out for error messages / creation sites.
        line_start_.push_back(0);
        for (std::size_t i = 0; i < source_.size(); ++i) {
            if (source_[i] == '\n') line_start_.push_back(i + 1);
        }
    }

    // program := { statement }
    Program run() {
        Program prog;
        prog.file = file_;
        prog.source = source_;
        skip_terminators();
        while (!at(TokenKind::End)) {
            prog.statements.push_back(statement());
            skip_terminators();
        }
        return prog;
    }

private:
    // --- token helpers -----------------------------------------------------
    const Token& cur() const { return tokens_[i_]; }
    const Token& peek(std::size_t off = 1) const {
        const std::size_t j = i_ + off;
        return j < tokens_.size() ? tokens_[j] : tokens_.back();
    }
    bool at(TokenKind k) const { return cur().kind == k; }
    bool at_keyword(const char* w) const {
        return cur().kind == TokenKind::Keyword && cur().text == w;
    }
    void advance() {
        if (i_ + 1 < tokens_.size()) ++i_;
    }
    bool accept(TokenKind k) {
        if (!at(k)) return false;
        advance();
        return true;
    }
    bool accept_keyword(const char* w) {
        if (!at_keyword(w)) return false;
        advance();
        return true;
    }
    void expect(TokenKind k, const char* what) {
        if (!at(k)) {
            fail(std::string("expected ") + what + ", found " + describe(cur()));
        }
        advance();
    }
    std::string expect_ident(const char* what) {
        if (!at(TokenKind::Ident)) {
            if (cur().kind == TokenKind::Keyword) {
                fail(std::string("expected ") + what + ", found the reserved word '" +
                     cur().text + "'");
            }
            fail(std::string("expected ") + what + ", found " + describe(cur()));
        }
        const std::string name = cur().text;
        advance();
        return name;
    }

    static std::string describe(const Token& t) {
        switch (t.kind) {
            case TokenKind::Ident:   return "identifier '" + t.text + "'";
            case TokenKind::Keyword: return "keyword '" + t.text + "'";
            case TokenKind::Int:     return "integer " + std::to_string(t.int_value);
            case TokenKind::String:  return "string literal";
            default:                 return TokenKindName(t.kind);
        }
    }

    [[noreturn]] void fail(const std::string& msg) const {
        const SourcePos p = cur().pos;
        throw ParseError(msg, p, SourceLine(source_, p.line));
    }
    [[noreturn]] void fail_at(const std::string& msg, const SourcePos& p) const {
        throw ParseError(msg, p, SourceLine(source_, p.line));
    }

    std::size_t offset_of(const SourcePos& p) const {
        if (p.line <= 0 || static_cast<std::size_t>(p.line) > line_start_.size()) {
            return source_.size();
        }
        const std::size_t off = line_start_[static_cast<std::size_t>(p.line - 1)] +
                                static_cast<std::size_t>(p.col > 0 ? p.col - 1 : 0);
        return off < source_.size() ? off : source_.size();
    }

    // The raw text of tokens [from, i_), trimmed.  Used for `source_text`.
    std::string text_since(std::size_t from) const {
        const std::size_t a = offset_of(tokens_[from].pos);
        const std::size_t b = offset_of(cur().pos);
        if (b <= a) return std::string();
        std::string s = source_.substr(a, b - a);
        while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\n' ||
                              s.back() == '\r' || s.back() == ';')) {
            s.pop_back();
        }
        return s;
    }

    void skip_terminators() {
        while (at(TokenKind::Newline) || at(TokenKind::Semicolon)) advance();
    }

    // TERM := NEWLINE | ";"   (";" additionally suppresses printing)
    bool statement_terminator(const char* what) {
        if (at(TokenKind::Semicolon)) {
            advance();
            return true;   // suppressed
        }
        if (at(TokenKind::Newline)) {
            advance();
            return false;
        }
        if (at(TokenKind::End) || at(TokenKind::RBrace)) return false;
        fail(std::string("expected end of statement after ") + what + ", found " +
             describe(cur()));
    }

    // --- statements --------------------------------------------------------
    StmtPtr statement() {
        if (at_keyword("for"))          return for_stmt();
        if (at_keyword("if"))           return if_stmt();
        if (at_keyword("set"))          return set_stmt();
        if (at_keyword("include"))      return include_stmt();
        if (at_keyword("help"))         return help_stmt();
        if (at_keyword("expect_error")) return expect_stmt();
        if (at_keyword("quit"))         return quit_stmt();
        if (at(TokenKind::Question))    return help_question_stmt();
        if (at_keyword("def")) {
            fail("'def' is reserved for user-defined functions and is not implemented "
                 "(console plan Sec. 16, owner question Q2)");
        }
        // assign := IDENT "=" expr TERM      (an '=' that is not '==')
        if (at(TokenKind::Ident) && peek().kind == TokenKind::Assign) {
            auto s = std::make_shared<Stmt>();
            s->kind = StmtKind::Assign;
            s->pos = cur().pos;
            s->name = cur().text;
            advance();   // IDENT
            advance();   // '='
            s->value = expression();
            statement_terminator("an assignment");
            return s;
        }
        // exprstmt := expr TERM
        auto s = std::make_shared<Stmt>();
        s->kind = StmtKind::ExprStmt;
        s->pos = cur().pos;
        s->value = expression();
        s->suppress_print = statement_terminator("an expression");
        return s;
    }

    // for := "for" IDENT "in" expr block
    StmtPtr for_stmt() {
        auto s = std::make_shared<Stmt>();
        s->kind = StmtKind::For;
        s->pos = cur().pos;
        advance();   // 'for'
        s->name = expect_ident("a loop variable name");
        if (!accept_keyword("in")) fail("expected 'in' after the loop variable");
        s->value = expression();
        s->body = block();
        return s;
    }

    // if := "if" expr block [ "else" ( block | if ) ]
    StmtPtr if_stmt() {
        auto s = std::make_shared<Stmt>();
        s->kind = StmtKind::If;
        s->pos = cur().pos;
        advance();   // 'if'
        s->value = expression();
        s->body = block();
        // `else` may legally begin on the next line.
        const std::size_t save = i_;
        skip_terminators();
        if (at_keyword("else")) {
            advance();
            if (at_keyword("if")) {
                s->orelse.push_back(if_stmt());
            } else {
                s->orelse = block();
            }
        } else {
            i_ = save;
        }
        return s;
    }

    // setstmt := "set" IDENT "=" expr TERM
    StmtPtr set_stmt() {
        auto s = std::make_shared<Stmt>();
        s->kind = StmtKind::Set;
        s->pos = cur().pos;
        advance();   // 'set'
        // Setting names may contain dots ("sturm.cap", owner decision R1), so a
        // key is a dotted identifier chain.
        std::string key = expect_ident("a setting name");
        while (at(TokenKind::Dot)) {
            advance();
            key += ".";
            key += expect_ident("a setting name component");
        }
        s->name = key;
        expect(TokenKind::Assign, "'=' after the setting name");
        s->value = expression();
        statement_terminator("a 'set' statement");
        return s;
    }

    // include := "include" "(" STRING ")" TERM
    StmtPtr include_stmt() {
        auto s = std::make_shared<Stmt>();
        s->kind = StmtKind::Include;
        s->pos = cur().pos;
        advance();   // 'include'
        expect(TokenKind::LParen, "'(' after 'include'");
        if (!at(TokenKind::String)) fail("include() takes a string literal path");
        s->text = cur().text;
        advance();
        expect(TokenKind::RParen, "')' closing include(...)");
        statement_terminator("include(...)");
        return s;
    }

    // help := "help" [ IDENT | "?" ] TERM
    StmtPtr help_stmt() {
        auto s = std::make_shared<Stmt>();
        s->kind = StmtKind::Help;
        s->pos = cur().pos;
        advance();   // 'help'
        if (at(TokenKind::Ident) || at(TokenKind::Keyword)) {
            s->name = cur().text;
            advance();
        } else if (at(TokenKind::Question)) {
            s->name = "?";
            advance();
        }
        statement_terminator("'help'");
        return s;
    }

    // help := "?" IDENT TERM
    StmtPtr help_question_stmt() {
        auto s = std::make_shared<Stmt>();
        s->kind = StmtKind::Help;
        s->pos = cur().pos;
        advance();   // '?'
        if (!at(TokenKind::Ident) && !at(TokenKind::Keyword)) {
            fail("'?' must be followed by a name ('?x' describes a value, '?verb' its overloads)");
        }
        s->name = cur().text;
        advance();
        statement_terminator("'?name'");
        return s;
    }

    // expect := "expect_error" STRING block
    StmtPtr expect_stmt() {
        auto s = std::make_shared<Stmt>();
        s->kind = StmtKind::ExpectError;
        s->pos = cur().pos;
        advance();   // 'expect_error'
        if (!at(TokenKind::String)) {
            fail("expect_error takes a string literal: expect_error \"substring\" { ... }");
        }
        s->text = cur().text;
        advance();
        s->body = block();
        return s;
    }

    StmtPtr quit_stmt() {
        auto s = std::make_shared<Stmt>();
        s->kind = StmtKind::Quit;
        s->pos = cur().pos;
        advance();
        statement_terminator("'quit'");
        return s;
    }

    // block := "{" { statement } "}"
    std::vector<StmtPtr> block() {
        // The '{' may open on the next line.
        skip_terminators();
        if (!at(TokenKind::LBrace)) fail("expected '{' to open a block, found " + describe(cur()));
        advance();
        std::vector<StmtPtr> out;
        skip_terminators();
        while (!at(TokenKind::RBrace)) {
            if (at(TokenKind::End)) fail("unterminated block: '}' expected");
            out.push_back(statement());
            skip_terminators();
        }
        advance();   // '}'
        return out;
    }

    // --- expressions -------------------------------------------------------
    // expr := orexpr
    ExprPtr expression() {
        const std::size_t start = i_;
        ExprPtr e = or_expr();
        return with_text(e, start);
    }

    ExprPtr with_text(const ExprPtr& e, std::size_t start) {
        if (!e) return e;
        if (!e->source_text.empty()) return e;
        auto copy = std::make_shared<Expr>(*e);
        copy->source_text = text_since(start);
        return copy;
    }

    // orexpr := andexpr { "or" andexpr }
    ExprPtr or_expr() {
        ExprPtr lhs = and_expr();
        while (at_keyword("or")) {
            const SourcePos p = cur().pos;
            advance();
            auto e = std::make_shared<Expr>();
            e->kind = ExprKind::Logical;
            e->logical_op = LogicalOp::Or;
            e->pos = p;
            e->lhs = lhs;
            e->rhs = and_expr();
            lhs = e;
        }
        return lhs;
    }

    // andexpr := notexpr { "and" notexpr }
    ExprPtr and_expr() {
        ExprPtr lhs = not_expr();
        while (at_keyword("and")) {
            const SourcePos p = cur().pos;
            advance();
            auto e = std::make_shared<Expr>();
            e->kind = ExprKind::Logical;
            e->logical_op = LogicalOp::And;
            e->pos = p;
            e->lhs = lhs;
            e->rhs = not_expr();
            lhs = e;
        }
        return lhs;
    }

    // notexpr := "not" notexpr | cmp
    ExprPtr not_expr() {
        if (at_keyword("not")) {
            const SourcePos p = cur().pos;
            advance();
            auto e = std::make_shared<Expr>();
            e->kind = ExprKind::Logical;
            e->logical_op = LogicalOp::Not;
            e->pos = p;
            e->lhs = not_expr();
            return e;
        }
        return cmp_expr();
    }

    // cmp := arith [ ("==" | "!=" | "<" | "<=" | ">" | ">=") arith ]   (non-associative)
    ExprPtr cmp_expr() {
        ExprPtr lhs = arith_expr();
        CompareOp op = CompareOp::Eq;
        bool found = false;
        switch (cur().kind) {
            case TokenKind::Eq: op = CompareOp::Eq; found = true; break;
            case TokenKind::Ne: op = CompareOp::Ne; found = true; break;
            case TokenKind::Lt: op = CompareOp::Lt; found = true; break;
            case TokenKind::Le: op = CompareOp::Le; found = true; break;
            case TokenKind::Gt: op = CompareOp::Gt; found = true; break;
            case TokenKind::Ge: op = CompareOp::Ge; found = true; break;
            default: break;
        }
        if (!found) return lhs;
        const SourcePos p = cur().pos;
        advance();
        auto e = std::make_shared<Expr>();
        e->kind = ExprKind::Compare;
        e->compare_op = op;
        e->pos = p;
        e->lhs = lhs;
        e->rhs = arith_expr();
        // `a < b < c` is a grammar error, not a silently chained comparison.
        switch (cur().kind) {
            case TokenKind::Eq: case TokenKind::Ne: case TokenKind::Lt:
            case TokenKind::Le: case TokenKind::Gt: case TokenKind::Ge:
                fail("comparisons do not chain; use 'and' (a < b and b < c)");
            default: break;
        }
        return e;
    }

    // arith := term { ("+" | "-") term }
    ExprPtr arith_expr() {
        ExprPtr lhs = term_expr();
        while (at(TokenKind::Plus) || at(TokenKind::Minus)) {
            const BinaryOp op = at(TokenKind::Plus) ? BinaryOp::Add : BinaryOp::Sub;
            const SourcePos p = cur().pos;
            advance();
            auto e = std::make_shared<Expr>();
            e->kind = ExprKind::Binary;
            e->binary_op = op;
            e->pos = p;
            e->lhs = lhs;
            e->rhs = term_expr();
            lhs = e;
        }
        return lhs;
    }

    // term := power { ("*" | "/" | "%") power }
    ExprPtr term_expr() {
        ExprPtr lhs = power_expr();
        while (at(TokenKind::Star) || at(TokenKind::Slash) || at(TokenKind::Percent)) {
            BinaryOp op = BinaryOp::Mul;
            if (at(TokenKind::Slash)) op = BinaryOp::Div;
            else if (at(TokenKind::Percent)) op = BinaryOp::Mod;
            const SourcePos p = cur().pos;
            advance();
            auto e = std::make_shared<Expr>();
            e->kind = ExprKind::Binary;
            e->binary_op = op;
            e->pos = p;
            e->lhs = lhs;
            e->rhs = power_expr();
            lhs = e;
        }
        return lhs;
    }

    // power := unary [ "^" power ]      (right-associative)
    ExprPtr power_expr() {
        ExprPtr base = unary_expr();
        if (at(TokenKind::Caret)) {
            const SourcePos p = cur().pos;
            advance();
            auto e = std::make_shared<Expr>();
            e->kind = ExprKind::Binary;
            e->binary_op = BinaryOp::Pow;
            e->pos = p;
            e->lhs = base;
            e->rhs = power_expr();
            return e;
        }
        return base;
    }

    // unary := "-" unary | postfix
    ExprPtr unary_expr() {
        if (at(TokenKind::Minus)) {
            const SourcePos p = cur().pos;
            advance();
            auto e = std::make_shared<Expr>();
            e->kind = ExprKind::Unary;
            e->pos = p;
            e->lhs = unary_expr();
            return e;
        }
        return postfix_expr();
    }

    // postfix := primary { "(" args ")" | "." IDENT | "[" expr "]" }
    ExprPtr postfix_expr() {
        const std::size_t start = i_;
        ExprPtr e = primary_expr();
        while (true) {
            if (at(TokenKind::LParen)) {
                const SourcePos p = cur().pos;
                advance();
                std::vector<Argument> args = argument_list();
                expect(TokenKind::RParen, "')' closing a call");
                auto call = std::make_shared<Expr>();
                if (e->kind == ExprKind::Member) {
                    // x.f(...) is a method call on x, not a call of the value x.f
                    call->kind = ExprKind::MethodCall;
                    call->lhs = e->lhs;
                    call->text = e->text;
                } else if (e->kind == ExprKind::Ident) {
                    call->kind = ExprKind::Call;
                    call->text = e->text;
                } else {
                    fail_at("only a verb name or a method can be called", p);
                }
                // The call's position is where the EXPRESSION starts, not
                // where '(' is: it is what a handle records as its creation
                // site, and `stub(5)` should point at `stub`.
                call->pos = tokens_[start].pos;
                (void)p;
                call->args = std::move(args);
                call->source_text = text_since(start);
                e = call;
                continue;
            }
            if (at(TokenKind::Dot)) {
                const SourcePos p = cur().pos;
                advance();
                auto member = std::make_shared<Expr>();
                member->kind = ExprKind::Member;
                member->pos = p;
                member->lhs = e;
                member->text = expect_ident("a member name after '.'");
                member->source_text = text_since(start);
                e = member;
                continue;
            }
            if (at(TokenKind::LBracket)) {
                const SourcePos p = cur().pos;
                advance();
                auto idx = std::make_shared<Expr>();
                idx->kind = ExprKind::Index;
                idx->pos = p;
                idx->lhs = e;
                idx->rhs = expression();
                expect(TokenKind::RBracket, "']' closing an index");
                idx->source_text = text_since(start);
                e = idx;
                continue;
            }
            break;
        }
        return e;
    }

    // args := [ arg { "," arg } ]      arg := expr | IDENT "=" expr
    std::vector<Argument> argument_list() {
        std::vector<Argument> out;
        if (at(TokenKind::RParen)) return out;
        bool seen_keyword = false;
        while (true) {
            Argument a;
            a.pos = cur().pos;
            if (at(TokenKind::Ident) && peek().kind == TokenKind::Assign) {
                a.name = cur().text;
                advance();
                advance();
                seen_keyword = true;
            } else if (seen_keyword) {
                fail("a positional argument cannot follow a keyword argument");
            }
            a.value = expression();
            out.push_back(std::move(a));
            if (!accept(TokenKind::Comma)) break;
        }
        return out;
    }

    // primary := INT | STRING | "true" | "false" | IDENT | "args"
    //          | "(" expr ")" | "[" [ expr { "," expr } ] "]"
    ExprPtr primary_expr() {
        const Token& t = cur();
        switch (t.kind) {
            case TokenKind::Int: {
                auto e = std::make_shared<Expr>();
                e->kind = ExprKind::IntLit;
                e->pos = t.pos;
                e->int_value = t.int_value;
                e->source_text = std::to_string(t.int_value);
                advance();
                return e;
            }
            case TokenKind::String: {
                auto e = std::make_shared<Expr>();
                e->kind = ExprKind::StringLit;
                e->pos = t.pos;
                e->text = t.text;
                advance();
                return e;
            }
            case TokenKind::Ident: {
                auto e = std::make_shared<Expr>();
                e->kind = ExprKind::Ident;
                e->pos = t.pos;
                e->text = t.text;
                e->source_text = t.text;
                advance();
                return e;
            }
            case TokenKind::Keyword: {
                if (t.text == "true" || t.text == "false") {
                    auto e = std::make_shared<Expr>();
                    e->kind = ExprKind::BoolLit;
                    e->pos = t.pos;
                    e->bool_value = (t.text == "true");
                    e->source_text = t.text;
                    advance();
                    return e;
                }
                if (t.text == "args") {
                    auto e = std::make_shared<Expr>();
                    e->kind = ExprKind::Args;
                    e->pos = t.pos;
                    e->source_text = "args";
                    advance();
                    return e;
                }
                fail("the reserved word '" + t.text + "' cannot appear in an expression");
            }
            case TokenKind::LParen: {
                advance();
                ExprPtr inner = expression();
                expect(TokenKind::RParen, "')' closing a parenthesised expression");
                return inner;
            }
            case TokenKind::LBracket: {
                auto e = std::make_shared<Expr>();
                e->kind = ExprKind::ListLit;
                e->pos = t.pos;
                advance();
                if (!at(TokenKind::RBracket)) {
                    while (true) {
                        e->items.push_back(expression());
                        if (!accept(TokenKind::Comma)) break;
                        if (at(TokenKind::RBracket)) break;   // trailing comma
                    }
                }
                expect(TokenKind::RBracket, "']' closing a list literal");
                return e;
            }
            default:
                break;
        }
        fail("expected an expression, found " + describe(t));
    }

    const std::string& source_;
    const std::string& file_;
    std::vector<Token> tokens_;
    std::vector<std::size_t> line_start_;
    std::size_t i_ = 0;
};

} // namespace

Program Parse(const std::string& source, const std::string& file) {
    ParserImpl p(source, file);
    return p.run();
}

} // namespace console
