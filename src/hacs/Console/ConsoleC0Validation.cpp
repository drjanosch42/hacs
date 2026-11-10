// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// ConsoleC0Validation.cpp -- see ConsoleC0Validation.h.
//
// This TU is compiled into cohomology_app, NOT into the ConsoleCore static
// library: the library must stay STL-only (plan Sec. 2 C8, enforced by
// docs/matrix-engine/tools/check_console_includes.sh), and the tree-level arms
// of section 7 below need POSIX directory walking.  Keeping the test code out
// of the library is also why check_console_includes.sh needs no exception for
// it.
// ===========================================================================

#include "ConsoleC0Validation.h"

#include "Core/ConsoleError.h"
#include "Core/CoreVerbs.h"
#include "Core/Evaluator.h"
#include "Core/Lexer.h"
#include "Core/LineInput.h"
#include "Core/Object.h"
#include "Core/Parser.h"
#include "Core/Printer.h"
#include "Core/Registry.h"
#include "Core/ScriptRunner.h"
#include "Core/Session.h"
#include "Core/Settings.h"
#include "Core/Value.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {

using namespace console;

// --- check harness ---------------------------------------------------------

std::uint32_t g_checks = 0;
std::uint32_t g_failures = 0;

void Check(bool ok, const std::string& label) {
    ++g_checks;
    if (ok) return;
    ++g_failures;
    if (g_failures <= 40) {
        std::cout << "  [FAIL] " << label << std::endl;
    } else if (g_failures == 41) {
        std::cout << "  [FAIL] (further failures suppressed)" << std::endl;
    }
}

void CheckEq(const std::string& got, const std::string& want, const std::string& label) {
    const bool ok = (got == want);
    Check(ok, ok ? label : label + "  [got: <" + got + "> want: <" + want + ">]");
}

void CheckEqInt(long long got, long long want, const std::string& label) {
    const bool ok = (got == want);
    Check(ok, ok ? label : label + "  [got: " + std::to_string(got) +
                               " want: " + std::to_string(want) + "]");
}

bool Contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

// --- the stub type (plan Sec. 10, C0: "using a stub type registered by the
// validation itself, since the arithmetic types are bindings") --------------

struct StubPayload {
    std::int64_t value = 0;
    bool fail_on_compute = false;
    bool budget_cut = false;
    std::int64_t budget = 0;
    int computes = 0;
};

const TypeInfo* g_stub_type = nullptr;
int g_total_computes = 0;

StubPayload& Payload(const ObjectPtr& o) {
    return *static_cast<StubPayload*>(o->payload().get());
}

ObjectPtr MakeStub(CallContext& ctx, std::int64_t value, bool fail, bool budget_cut,
                   std::int64_t budget) {
    auto payload = std::make_shared<StubPayload>();
    payload->value = value;
    payload->fail_on_compute = fail;
    payload->budget_cut = budget_cut;
    payload->budget = budget;

    ObjectPtr obj = Object::create(
        g_stub_type, ctx.pos, ctx.call_text, ctx.session->settings().snapshot(), payload,
        [](Object& self) {
            StubPayload& p = *static_cast<StubPayload*>(self.payload().get());
            ++p.computes;
            ++g_total_computes;
            if (p.fail_on_compute) {
                throw std::runtime_error("stub: the computation was asked to fail");
            }
            // The value a computed member reports depends on the settings
            // snapshot in force, which is how the "settings captured at
            // creation" claim is observable.
            const Value* cap = self.snapshot().find(kSettingPrintMaxEntries);
            if (cap != nullptr && cap->is_int()) p.value += cap->int_value();
            if (p.budget_cut) {
                ResumableState rs;
                rs.resumable = true;
                rs.budget_setting = "stub.budget";
                rs.budget_value = Value::from_int(p.budget);
                rs.reason = "stub budget exhausted";
                Record progress;
                progress.set("iterations", Value::from_int(p.budget));
                rs.progress = progress;
                self.set_resumable_state(rs);
            }
        });
    return obj;
}

void RegisterStubType(Registry& reg) {
    TypeInfo t;
    t.name = "Stub";
    t.doc = "the C0 validation's stand-in for a pipeline handle";

    TypeInfo::Member value;
    value.name = "value";
    value.computed = true;
    value.doc = "the computed value";
    value.get = [](const ObjectPtr& o) { return Value::from_int(Payload(o).value); };
    t.members.push_back(value);

    TypeInfo::Member label;
    label.name = "label";
    label.computed = false;   // structural: readable while still `declared`
    label.doc = "a structural member";
    label.get = [](const ObjectPtr&) { return Value::from_string("stub"); };
    t.members.push_back(label);

    TypeInfo::Member computes;
    computes.name = "computes";
    computes.computed = false;
    computes.get = [](const ObjectPtr& o) {
        return Value::from_int(Payload(o).computes);
    };
    t.members.push_back(computes);

    TypeInfo::Method scaled;
    scaled.name = "scaled";
    scaled.params = {"k"};
    scaled.required = 1;
    scaled.computed = true;
    scaled.doc = "value * k";
    scaled.call = [](const ObjectPtr& o, const std::vector<Value>& args,
                     const SourcePos& pos) -> Value {
        if (args.empty() || !args[0].is_int()) throw TypeError("scaled(k) expects an Int", pos);
        return Value::from_int(Payload(o).value * args[0].int_value());
    };
    t.methods.push_back(scaled);

    t.print = [](const ObjectPtr& o, std::int64_t) -> std::string {
        return "Stub(" + std::to_string(Payload(o).value) + ")";
    };
    t.show = [](const ObjectPtr& o) -> std::string {
        return "Stub value=" + std::to_string(Payload(o).value) +
               " state=" + ObjectStateName(o->state());
    };
    t.equal = [](const ObjectPtr& a, const ObjectPtr& b) {
        return Payload(a).value == Payload(b).value;
    };
    t.resume = [](const ObjectPtr& o, const SettingsSnapshot& snap) -> ObjectPtr {
        auto payload = std::make_shared<StubPayload>();
        payload->value = Payload(o).value;
        const Value* b = snap.find("stub.budget");
        payload->budget = (b != nullptr && b->is_int()) ? b->int_value() : 0;
        payload->budget_cut = false;   // the continuation completes
        return Object::create(o->type(), o->creation_site(), "resume(" + o->creation_expr() + ")",
                              snap, payload, [](Object& self) {
                                  StubPayload& p = *static_cast<StubPayload*>(self.payload().get());
                                  p.value += p.budget;
                              });
    };

    g_stub_type = reg.register_type(std::move(t));
}

void RegisterStubVerbs(Registry& reg) {
    Signature s1;
    s1.params = {Param::of("value", ValueKind::Int)};
    reg.register_verb(
        "stub", s1,
        [](CallContext& ctx, const std::vector<Value>& args) -> Value {
            return Value::from_object(MakeStub(ctx, args[0].int_value(), false, false, 0));
        },
        "a lazily computed stub handle", "test");

    Signature s2;
    s2.params = {Param::of("value", ValueKind::Int)};
    reg.register_verb(
        "stub_failing", s2,
        [](CallContext& ctx, const std::vector<Value>& args) -> Value {
            return Value::from_object(MakeStub(ctx, args[0].int_value(), true, false, 0));
        },
        "a stub handle whose compute() throws", "test");

    Signature s3;
    s3.params = {Param::of("value", ValueKind::Int), Param::of("budget", ValueKind::Int)};
    reg.register_verb(
        "stub_budgeted", s3,
        [](CallContext& ctx, const std::vector<Value>& args) -> Value {
            return Value::from_object(
                MakeStub(ctx, args[0].int_value(), false, true, args[1].int_value()));
        },
        "a stub handle cut off by a budget (owner decision R3)", "test");

    // A handle that keeps another alive -- the plan Sec. 4.1 edge.
    Signature s4;
    s4.params = {Param::of("inner", ValueKind::Object)};
    reg.register_verb(
        "stub_wrap", s4,
        [](CallContext& ctx, const std::vector<Value>& args) -> Value {
            ObjectPtr inner = args[0].object_value();
            ObjectPtr outer = MakeStub(ctx, 0, false, false, 0);
            outer->keep_alive(inner);
            return Value::from_object(outer);
        },
        "a stub handle that keeps its argument alive", "test");

    // Overload-order probes: three same-arity overloads of one verb.
    Signature o1;
    o1.params = {Param::of("x", ValueKind::Int)};
    reg.register_verb(
        "which", o1,
        [](CallContext&, const std::vector<Value>&) { return Value::from_string("Int"); },
        "Int overload", "test");
    Signature o2;
    o2.params = {Param::of("x", ValueKind::String)};
    reg.register_verb(
        "which", o2,
        [](CallContext&, const std::vector<Value>&) { return Value::from_string("String"); },
        "String overload", "test");
    Signature o3;
    o3.params = {Param::any("x")};
    reg.register_verb(
        "which", o3,
        [](CallContext&, const std::vector<Value>&) { return Value::from_string("any"); },
        "catch-all overload", "test");

    // Keyword arguments and defaults.
    Signature k1;
    k1.params = {Param::of("a", ValueKind::Int),
                 Param::of("b", ValueKind::Int).optional(Value::from_int(10)),
                 Param::of("c", ValueKind::Int).optional(Value::from_int(100))};
    reg.register_verb(
        "combine", k1,
        [](CallContext&, const std::vector<Value>& args) {
            return Value::from_int(args[0].int_value() + args[1].int_value() * 2 +
                                   args[2].int_value() * 3);
        },
        "a + 2b + 3c", "test");

    // A verb that throws a plain std::exception, to exercise the wrapping of
    // plan Sec. 2 C7 (the catch order itself is installed by C1's translator).
    Signature t1;
    t1.params = {};
    reg.register_verb(
        "boom", t1,
        [](CallContext&, const std::vector<Value>&) -> Value {
            throw std::runtime_error("a pipeline-style failure");
        },
        "throws std::runtime_error", "test");
}

// --- run helpers -----------------------------------------------------------

struct Outcome {
    int code = 0;
    std::string out;
};

// A registry with the core verbs plus the stub surface.
struct Fixture {
    Registry registry;
    Fixture() {
        RegisterCoreVerbs(registry);
        RegisterStubType(registry);
        RegisterStubVerbs(registry);
    }
};

Evaluator::FileLoader MapLoader(const std::map<std::string, std::string>& files) {
    return [files](const std::string& path, std::string& out) -> bool {
        std::map<std::string, std::string>::const_iterator it = files.find(path);
        if (it == files.end()) return false;
        out = it->second;
        return true;
    };
}

Outcome Run(const std::string& source,
            const std::map<std::string, std::string>* files = nullptr,
            const std::vector<std::pair<std::string, Value>>* args = nullptr) {
    Fixture fx;
    Session session(fx.registry, /*verbose_default=*/false);
    std::ostringstream oss;
    session.set_output(oss);
    if (args != nullptr) {
        for (const std::pair<std::string, Value>& kv : *args) session.set_arg(kv.first, kv.second);
    }
    Outcome outcome;
    outcome.code = files != nullptr ? RunSource(session, source, "t.oc", MapLoader(*files))
                                    : RunSource(session, source, "t.oc");
    outcome.out = oss.str();
    return outcome;
}

// Runs and expects success, returning the printed text with the trailing
// newline stripped.
std::string Eval(const std::string& source, const std::string& label) {
    Outcome o = Run(source);
    Check(o.code == 0, label + "  [exit " + std::to_string(o.code) + ": " + o.out + "]");
    std::string text = o.out;
    while (!text.empty() && (text.back() == '\n')) text.pop_back();
    return text;
}

void ExpectValue(const std::string& source, const std::string& want, const std::string& label) {
    CheckEq(Eval(source, label), want, label);
}

void ExpectError(const std::string& source, int want_code, const std::string& needle,
                 const std::string& label) {
    Outcome o = Run(source);
    const bool code_ok = (o.code == want_code);
    Check(code_ok, code_ok ? label + " (exit code)"
                           : label + " (exit code)  [got " + std::to_string(o.code) + " want " +
                                 std::to_string(want_code) + "; output: " + o.out + "]");
    const bool msg_ok = needle.empty() || Contains(o.out, needle);
    Check(msg_ok, msg_ok ? label + " (message)"
                         : label + " (message)  [want <" + needle + "> in: " + o.out + "]");
}

// --- 1. lexing -------------------------------------------------------------

void TestLexing() {
    ExpectValue("1", "1", "lex: decimal literal");
    ExpectValue("0", "0", "lex: zero");
    ExpectValue("134_217_757", "134217757", "lex: underscore separators");
    ExpectValue("0xff", "255", "lex: hex literal");
    ExpectValue("0XFF", "255", "lex: upper-case hex");
    ExpectValue("0x_f_f", "255", "lex: hex with separators");
    ExpectValue("9223372036854775807", "9223372036854775807", "lex: int64 max");
    ExpectValue("\"abc\"", "\"abc\"", "lex: string literal prints quoted");
    ExpectValue("print(\"a\\nb\")", "a\nb", "lex: \\n escape");
    ExpectValue("print(\"a\\tb\")", "a\tb", "lex: \\t escape");
    ExpectValue("print(\"a\\\"b\")", "a\"b", "lex: \\\" escape");
    ExpectValue("print(\"a\\\\b\")", "a\\b", "lex: backslash escape");
    ExpectValue("true", "true", "lex: true");
    ExpectValue("false", "false", "lex: false");

    ExpectValue("1 # a comment\n", "1", "lex: '#' comment");
    ExpectValue("1 // a comment\n", "1", "lex: '//' comment");
    ExpectValue("# only a comment\n2", "2", "lex: comment-only line");
    ExpectValue("1 +\\\n 2", "3", "lex: backslash continuation");
    ExpectValue("(1 +\n 2)", "3", "lex: newline inside parentheses");
    ExpectValue("[1,\n 2][1]", "2", "lex: newline inside brackets");

    ExpectError("9223372036854775808", 2, "does not fit in Int", "lex: int64 overflow literal");
    ExpectError("\"unterminated", 2, "unterminated string", "lex: unterminated string");
    ExpectError("\"bad\\q\"", 2, "unknown escape", "lex: unknown escape");
    ExpectError("0x", 2, "no digits", "lex: empty hex literal");
    ExpectError("1abc", 2, "invalid character in integer", "lex: digit followed by letter");
    ExpectError("@", 2, "unexpected character", "lex: illegal character");
    ExpectError("1 ! 2", 2, "unexpected '!'", "lex: lone '!'");

    // Reserved words are reserved (plan Sec. 3.1).
    const char* const reserved[] = {"for", "in", "if", "else", "set", "help", "quit",
                                    "true", "false", "and", "or", "not", "expect_error",
                                    "include", "args", "def"};
    for (const char* w : reserved) {
        Check(IsReservedWord(w), std::string("lex: '") + w + "' is reserved");
    }
    Check(!IsReservedWord("hecke"), "lex: 'hecke' is not reserved");
    Check(!IsReservedWord("print"), "lex: 'print' is not reserved");

    // NeedsContinuation, the REPL's rule.
    Check(NeedsContinuation("f(1,"), "lex: continuation on open paren");
    Check(NeedsContinuation("[1, 2"), "lex: continuation on open bracket");
    Check(NeedsContinuation("for x in [1] {"), "lex: continuation on open brace");
    Check(NeedsContinuation("1 + \\"), "lex: continuation on trailing backslash");
    Check(NeedsContinuation("\"open"), "lex: continuation inside a string");
    Check(!NeedsContinuation("1 + 2"), "lex: complete statement needs no continuation");
    Check(!NeedsContinuation("f(1) # ("), "lex: a paren inside a comment does not continue");
    Check(!NeedsContinuation("\"a(\""), "lex: a paren inside a string does not continue");

    // Tokenize directly: positions.
    std::vector<Token> toks = Tokenize("x = 1\ny = 2", "f.oc");
    CheckEqInt(static_cast<long long>(toks.size()), 8, "lex: token count of two assignments");
    CheckEqInt(toks[0].pos.line, 1, "lex: first token line");
    CheckEqInt(toks[0].pos.col, 1, "lex: first token column");
    CheckEqInt(toks[4].pos.line, 2, "lex: token line after newline");
    CheckEqInt(toks[4].pos.col, 1, "lex: token column after newline");
    Check(toks.back().kind == TokenKind::End, "lex: input ends with End");

    CheckEq(SourceLine("a\nbb\nccc", 2), "bb", "lex: SourceLine picks the right line");
    CheckEq(SourceLine("a\nbb\nccc", 9), "", "lex: SourceLine out of range");
}

// --- 2. parsing ------------------------------------------------------------

void TestParsing() {
    // Precedence table (plan Sec. 3.2), one case per level.
    ExpectValue("1 + 2 * 3", "7", "parse: * binds tighter than +");
    ExpectValue("(1 + 2) * 3", "9", "parse: parentheses");
    ExpectValue("2 ^ 3 ^ 2", "512", "parse: ^ is right-associative");
    // EBNF: power := unary [ "^" power ], so the unary minus is part of the BASE:
    // -2 ^ 2 is (-2)^2 = 4.  The plan Sec. 3.2 grammar is normative here, and a
    // script that wants the mathematical convention writes 0 - 2 ^ 2.
    ExpectValue("-2 ^ 2", "4", "parse: unary minus is part of the base of ^ (EBNF)");
    ExpectValue("0 - 2 ^ 2", "-4", "parse: the mathematical convention, spelled out");
    ExpectValue("10 - 3 - 2", "5", "parse: - is left-associative");
    ExpectValue("100 / 10 / 2", "5", "parse: / is left-associative");
    ExpectValue("7 % 3", "1", "parse: %");
    ExpectValue("1 + 2 == 3", "true", "parse: comparison binds looser than arithmetic");
    ExpectValue("true and false or true", "true", "parse: 'and' binds tighter than 'or'");
    ExpectValue("not true and false", "false", "parse: 'not' binds tighter than 'and'");
    ExpectValue("not (true and false)", "true", "parse: parenthesised logical");
    ExpectValue("-3", "-3", "parse: unary minus");
    ExpectValue("- -3", "3", "parse: double unary minus");

    // Postfix chain.
    ExpectValue("[[1,2],[3,4]][1][0]", "3", "parse: chained indexing");
    ExpectValue("stub(1).label", "\"stub\"", "parse: member access");
    ExpectValue("stub(4).scaled(2)", "808", "parse: method call");
    ExpectValue("[1,2,3][len([1,2]) ]", "3", "parse: call inside an index");

    // Statements.
    ExpectValue("x = 41\nx + 1", "42", "parse: assignment then expression");
    ExpectValue("1;\n2", "2", "parse: ';' suppresses printing");
    ExpectValue("if 1 < 2 { 7 }", "7", "parse: if");
    ExpectValue("if 1 > 2 { 7 } else { 8 }", "8", "parse: if/else");
    ExpectValue("if 1 > 2 { 7 } else if 3 > 2 { 9 } else { 8 }", "9", "parse: else-if chain");
    ExpectValue("for x in [1,2,3] { x }", "1\n2\n3", "parse: for");
    ExpectValue("if 1 < 2 {\n 7\n}", "7", "parse: block across lines");
    ExpectValue("if 1 > 2 {\n 7\n}\nelse {\n 8\n}", "8", "parse: else on its own line");
    ExpectValue("set print_max_entries = 3\nsettings()[\"print_max_entries\"][\"value\"]", "3",
                "parse: set statement");
    ExpectValue("expect_error \"unknown name\" { nope }\n\"ok\"", "\"ok\"",
                "parse: expect_error block");
    ExpectValue("[1, 2, 3,]", "[1, 2, 3]", "parse: trailing comma in a list");
    ExpectValue("[]", "[]", "parse: empty list");

    // Parse errors, with the caret line.
    Outcome o = Run("1 +");
    CheckEqInt(o.code, 2, "parse: incomplete expression is exit 2");
    Check(Contains(o.out, "t.oc:1:4"), "parse: error carries file:line:col");
    Check(Contains(o.out, "^"), "parse: error prints a caret");
    ExpectError("if 1 < 2 7", 2, "expected '{'", "parse: block required after if");
    ExpectError("for x [1] { }", 2, "expected 'in'", "parse: 'in' required");
    ExpectError("for 1 in [1] { }", 2, "expected a loop variable", "parse: loop variable must be a name");
    ExpectError("if true { 1 ", 2, "unterminated block", "parse: unterminated block");
    ExpectError("f(1))", 2, "expected end of statement", "parse: stray close paren");
    ExpectError("[1, 2", 2, "expected ']'", "parse: unterminated list");
    ExpectError("1 < 2 < 3", 2, "comparisons do not chain", "parse: chained comparison rejected");
    ExpectError("def f(x) { x }", 2, "'def' is reserved", "parse: def is reserved, not implemented");
    ExpectError("for = 1", 2, "expected a loop variable name",
                "parse: a reserved word cannot be assigned to");
    ExpectError("include(1)", 2, "string literal path", "parse: include needs a string");
    ExpectError("expect_error { }", 2, "expect_error takes a string", "parse: expect_error needs a substring");
    ExpectError("combine(b = 1, 2)", 2, "positional argument cannot follow",
                "parse: positional after keyword");
    ExpectError("1 2", 2, "expected end of statement", "parse: two expressions on one line");
    ExpectError("(1 + 2)(3)", 2, "only a verb name or a method can be called",
                "parse: calling a non-name");

    // The AST records the source text of a call, which becomes a handle's
    // creation expression (plan Sec. 2 C7).
    Program p = Parse("T = stub(7)\n", "f.oc");
    CheckEqInt(static_cast<long long>(p.statements.size()), 1, "parse: one statement");
    Check(p.statements[0]->kind == StmtKind::Assign, "parse: statement is an assignment");
    CheckEq(p.statements[0]->value->source_text, "stub(7)", "parse: call source text captured");
}

// --- 3. evaluation ---------------------------------------------------------

void TestEvaluation() {
    // Exact integer arithmetic, overflow-checked.
    ExpectValue("2 + 3", "5", "eval: addition");
    ExpectValue("100 / 4", "25", "eval: exact division");
    ExpectValue("2 ^ 62", "4611686018427387904", "eval: power at the int64 edge");
    ExpectValue("0 ^ 0", "1", "eval: 0^0 is 1");
    ExpectValue("(0 - 9223372036854775807) - 1", "-9223372036854775808", "eval: int64 min");
    ExpectError("9223372036854775807 + 1", 3, "overflow in '+'", "eval: addition overflow");
    ExpectError("(0-9223372036854775807) - 2", 3, "overflow in '-'", "eval: subtraction overflow");
    ExpectError("4294967296 * 4294967296", 3, "overflow in '*'", "eval: multiplication overflow");
    ExpectError("2 ^ 64", 3, "overflow in '^'", "eval: power overflow");
    ExpectError("- (0 - 9223372036854775807 - 1)", 3, "overflow in unary", "eval: negation overflow");
    ExpectError("7 / 2", 3, "exact division", "eval: inexact division rejected");
    ExpectError("1 / 0", 3, "division by zero", "eval: division by zero");
    ExpectError("1 % 0", 3, "division by zero in '%'", "eval: modulo by zero");
    ExpectError("2 ^ (0-1)", 3, "non-negative exponent", "eval: negative exponent");
    ExpectValue("abs(0 - 5)", "5", "eval: abs");

    // Strings.
    ExpectValue("print(\"a\" + \"b\")", "ab", "eval: string concatenation");
    ExpectError("\"a\" + 1", 3, "not defined on String and Int", "eval: mixed + rejected");
    ExpectError("[1] + [2]", 3, "not defined on List and List", "eval: list + rejected");

    // Comparison and equality.
    ExpectValue("1 == 1", "true", "eval: int equality");
    ExpectValue("\"a\" == \"a\"", "true", "eval: string equality");
    ExpectValue("[1,[2]] == [1,[2]]", "true", "eval: deep list equality");
    ExpectValue("[1,2] == [2,1]", "false", "eval: list order matters");
    ExpectValue("1 == \"1\"", "false", "eval: cross-type equality is false, not an error");
    ExpectValue("\"a\" < \"b\"", "true", "eval: string ordering");
    ExpectValue("stub(3) == stub(3)", "true", "eval: objects compare through TypeInfo::equal");
    ExpectValue("stub(3) == stub(4)", "false", "eval: unequal stubs");
    ExpectError("[1] < [2]", 3, "not defined on List and List", "eval: list ordering rejected");

    // Logical short-circuit: the right side is never evaluated.
    ExpectValue("false and boom()", "false", "eval: 'and' short-circuits");
    ExpectValue("true or boom()", "true", "eval: 'or' short-circuits");
    ExpectError("true and 1", 3, "expects Bool", "eval: 'and' needs Bool");
    ExpectError("not 1", 3, "'not' expects Bool", "eval: 'not' needs Bool");
    ExpectError("if 1 { 2 }", 3, "needs a Bool condition", "eval: no truthiness");

    // for / shadowing (plan Sec. 9: one global frame + for-loop shadowing).
    ExpectValue("t = 0\nfor x in [1,2,3] { t = t + x }\nt", "6", "eval: for accumulates");
    ExpectValue("x = 99\nfor x in [1,2] { }\nx", "99", "eval: loop variable is restored");
    ExpectValue("for x in [1,2] { }\nexpect_error \"unknown name\" { x }\n\"ok\"", "\"ok\"",
                "eval: loop variable is unbound afterwards");
    ExpectValue("for x in [] { boom() }\n\"ok\"", "\"ok\"", "eval: empty loop body never runs");
    ExpectValue("t = 0\nfor a in [1,2] { for b in [10,20] { t = t + a*b } }\nt", "90",
                "eval: nested loops");
    ExpectError("for x in 1 { }", 3, "iterates over a List", "eval: for needs a List");

    // Indexing.
    ExpectValue("[10,20,30][2]", "30", "eval: list index");
    ExpectError("[1,2][5]", 3, "out of range", "eval: index out of range");
    ExpectError("[1,2][0-1]", 3, "out of range", "eval: negative index");
    ExpectError("[1,2][\"a\"]", 3, "index must be Int", "eval: wrong index type");
    ExpectError("1[0]", 3, "indexing is not defined", "eval: indexing a scalar");

    // Records.
    ExpectValue("len(settings())", "2", "eval: settings() has the two core keys");
    ExpectValue("keys(settings())[0]", "\"print_max_entries\"", "eval: record key order");
    ExpectError("settings().nope", 3, "has no key 'nope'", "eval: missing record key");

    // Unknown names.
    ExpectError("nope", 3, "unknown name 'nope'", "eval: unknown variable");
    ExpectError("nope()", 3, "unknown verb 'nope'", "eval: unknown verb");
    ExpectError("x = 1\nx()", 3, "is a variable, not a verb", "eval: calling a variable");
    ExpectError("print", 3, "is a verb; call it", "eval: naming a verb without calling it");
    ExpectError("x = print()", 3, "would be bound to nothing", "eval: binding a nil result");

    // assert / expect_error / exit.
    ExpectError("assert(1 == 2, \"one is not two\")", 4, "one is not two", "eval: assert fails with 4");
    ExpectValue("assert(1 == 1, \"fine\")\n\"ok\"", "\"ok\"", "eval: assert passes");
    ExpectError("assert(1)", 3, "expects Bool", "eval: assert needs a Bool");
    ExpectError("expect_error \"nope\" { 1 }", 4, "did not fail", "eval: expect_error with no error");
    ExpectError("expect_error \"apples\" { 1/0 }", 4, "not with", "eval: expect_error wrong message");
    ExpectValue("expect_error \"division by zero\" { 1/0 }\n\"ok\"", "\"ok\"",
                "eval: expect_error matches");
    Outcome ex = Run("exit(7)");
    CheckEqInt(ex.code, 7, "eval: exit(7)");
    Outcome ex0 = Run("1\nexit()\nboom()");
    CheckEqInt(ex0.code, 0, "eval: exit() stops the script with 0");
    CheckEq(ex0.out, "1\n", "eval: statements after exit() do not run");
    Outcome q = Run("quit\nboom()");
    CheckEqInt(q.code, 0, "eval: quit ends the script");

    // A wrapped pipeline-style exception (plan Sec. 2 C7).
    ExpectError("boom()", 3, "a pipeline-style failure", "eval: std::exception is wrapped");
    ExpectError("boom()", 3, "caused by", "eval: wrapped exception names its cause");

    // args (plan Sec. 6).
    std::vector<std::pair<std::string, Value>> args;
    args.emplace_back("n", Value::from_int(128));
    args.emplace_back("name", Value::from_string("run"));
    Outcome a1 = Run("args.n", nullptr, &args);
    CheckEq(a1.out, "128\n", "eval: args.n");
    Outcome a2 = Run("print(args.name)", nullptr, &args);
    CheckEq(a2.out, "run\n", "eval: args.name");
    Outcome a3 = Run("args.missing", nullptr, &args);
    CheckEqInt(a3.code, 3, "eval: missing args.name is exit 3");
    Check(Contains(a3.out, "no script parameter named 'missing'"),
          "eval: missing parameter names itself");
    Outcome a4 = Run("len(args)", nullptr, &args);
    CheckEq(a4.out, "2\n", "eval: args is a Record");

    // include (plan Sec. 3.2 / Sec. 6).
    std::map<std::string, std::string> files;
    files["prelude.oc"] = "base = 40\n";
    files["sub/deeper.oc"] = "deep = 2\n";
    files["sub/mid.oc"] = "include(\"deeper.oc\")\nmid = deep + 1\n";
    files["loop_a.oc"] = "include(\"loop_b.oc\")\n";
    files["loop_b.oc"] = "include(\"loop_a.oc\")\n";
    files["selfloop.oc"] = "include(\"selfloop.oc\")\n";
    files["broken.oc"] = "1 +\n";

    Outcome i1 = Run("include(\"prelude.oc\")\nbase + 2", &files);
    CheckEq(i1.out, "42\n", "eval: include shares the global frame");
    Outcome i2 = Run("include(\"sub/mid.oc\")\nmid", &files);
    CheckEq(i2.out, "3\n", "eval: nested include resolves relative to the including file");
    Outcome i3 = Run("include(\"loop_a.oc\")", &files);
    CheckEqInt(i3.code, 3, "eval: include cycle is a runtime error");
    Check(Contains(i3.out, "include cycle"), "eval: include cycle is named");
    Outcome i4 = Run("include(\"selfloop.oc\")", &files);
    CheckEqInt(i4.code, 3, "eval: self-include cycle");
    Outcome i5 = Run("include(\"nowhere.oc\")", &files);
    CheckEqInt(i5.code, 3, "eval: missing include is a runtime error");
    Check(Contains(i5.out, "cannot read"), "eval: missing include says so");
    Outcome i6 = Run("include(\"broken.oc\")", &files);
    CheckEqInt(i6.code, 2, "eval: a parse error inside an include is exit 2");
    Check(Contains(i6.out, "broken.oc"), "eval: include parse error names the included file");
    Outcome i7 = Run("include(\"prelude.oc\")\ninclude(\"prelude.oc\")\nbase", &files);
    CheckEq(i7.out, "40\n", "eval: including twice in sequence is not a cycle");

    // Path helpers.
    CheckEq(DirectoryOf("a/b/c.oc"), "a/b", "eval: DirectoryOf");
    CheckEq(DirectoryOf("c.oc"), "", "eval: DirectoryOf with no directory");
    CheckEq(ResolvePath("a/b", "c.oc"), "a/b/c.oc", "eval: ResolvePath relative");
    CheckEq(ResolvePath("a/b", "/abs/c.oc"), "/abs/c.oc", "eval: ResolvePath absolute");
    CheckEq(ResolvePath("", "c.oc"), "c.oc", "eval: ResolvePath with no directory");
}

// --- 4. printing -----------------------------------------------------------

void TestPrinting() {
    ExpectValue("[1, \"a\", true]", "[1, \"a\", true]", "print: list is canonical");
    ExpectValue("print(\"a\", 1, true)", "a 1 true", "print: print uses the display form");
    ExpectValue("print([\"a\"])", "[\"a\"]", "print: strings inside a list stay quoted");
    ExpectValue("show([1,2])", "[1, 2]", "print: show");
    ExpectValue("print(str(123))", "123", "print: str of an Int");
    ExpectValue("repr(\"a\")", "\"\\\"a\\\"\"", "print: repr quotes");
    ExpectValue("stub(1)", "Stub(1)", "print: a type's own printer (still declared)");
    ExpectValue("print(settings()[\"verbose\"][\"source\"])", "session",
                "print: a record field of settings()");

    // The entry cap.
    ExpectValue("set print_max_entries = 2\n[1,2,3,4]", "[1, 2, ... (2 more of 4)]",
                "print: print_max_entries caps a list");
    ExpectValue("set print_max_entries = 2\nshow([1,2,3,4])", "[1, 2, 3, 4]",
                "print: show ignores the cap");
    ExpectValue("set print_max_entries = 0\n[1,2]", "[... (2 more of 2)]",
                "print: a zero cap shows nothing");
    ExpectError("set print_max_entries = 0 - 1", 3, "must be >= 0", "print: negative cap rejected");

    // JSON with a proper escaper (gap G1).
    ExpectValue("print(json(1))", "1", "json: Int");
    ExpectValue("print(json(true))", "true", "json: Bool");
    ExpectValue("print(json(\"a\\\"b\"))", "\"a\\\"b\"", "json: quote escaped");
    ExpectValue("print(json(\"a\\nb\"))", "\"a\\nb\"", "json: newline escaped");
    ExpectValue("print(json([1, \"a\"]))", "[1, \"a\"]", "json: list");
    CheckEq(JsonEscape(std::string("\x01")), "\\u0001", "json: control character escaped");
    CheckEq(JsonEscape("a\\b"), "a\\\\b", "json: backslash escaped");
    CheckEq(JsonEscape("a\tb"), "a\\tb", "json: tab escaped");
    CheckEq(ToJson(Value::nil()), "null", "json: nil");

    // Annotations are the `#~` lines every transcript comparison ignores.
    Outcome ann = Run("annotate(\"time 1.5\")\n1");
    CheckEq(ann.out, "#~ time 1.5\n1\n", "print: annotate emits a '#~' line");
    CheckEq(std::string(kAnnotationPrefix), "#~ ", "print: the annotation prefix is '#~ '");

    // describe / ?x.
    Outcome d = Run("T = stub(5)\n?T");
    CheckEqInt(d.code, 0, "print: ?x runs");
    Check(Contains(d.out, "type:  Stub"), "print: ?x names the type");
    Check(Contains(d.out, "state: declared"), "print: ?x reports the state");
    Check(Contains(d.out, "born:  t.oc:1:5 by stub(5)"), "print: ?x reports the creation site");
    Check(Contains(d.out, "print_max_entries = 400"), "print: ?x reports the settings snapshot");
    Check(Contains(d.out, ".value"), "print: ?x lists members");
    Check(Contains(d.out, ".scaled(k)"), "print: ?x lists methods");
    Check(!Contains(d.out, "state: computed"), "print: ?x does not force the computation");

    Outcome dl = Run("L = [1,2,3]\n?L");
    Check(Contains(dl.out, "type:  List"), "print: describe a list");
    Check(Contains(dl.out, "size:  3"), "print: describe reports a list's size");

    // help.
    Outcome h = Run("help");
    Check(Contains(h.out, "OC script"), "help: banner");
    Check(Contains(h.out, "print"), "help: lists verbs");
    Check(Contains(h.out, "Stub"), "help: lists types");
    Outcome hv = Run("help range");
    Check(Contains(hv.out, "range(from: Int, to: Int)"), "help: one verb's first overload");
    Check(Contains(hv.out, "range(from: Int, to: Int, step: Int)"), "help: second overload");
    Outcome ht = Run("help Stub");
    Check(Contains(ht.out, "type Stub"), "help: a type");
    Outcome hs = Run("help print_max_entries");
    Check(Contains(hs.out, "setting print_max_entries"), "help: a setting");
    Outcome hn = Run("help nothing_at_all");
    Check(Contains(hn.out, "no help for"), "help: unknown topic");
    Outcome hq = Run("x = 3\n?x");
    Check(Contains(hq.out, "value: 3"), "help: ?x on a plain value");
}

// --- 5. registry -----------------------------------------------------------

void TestRegistry() {
    // Resolution order: first match in registration order (plan Sec. 2 C3).
    ExpectValue("print(which(1))", "Int", "registry: Int overload wins for an Int");
    ExpectValue("print(which(\"a\"))", "String", "registry: String overload for a String");
    ExpectValue("print(which([1]))", "any", "registry: the catch-all takes the rest");
    ExpectValue("print(which(true))", "any", "registry: Bool falls through to the catch-all");

    // Keyword arguments and defaults.
    ExpectValue("combine(1)", "321", "registry: defaults fill in");
    ExpectValue("combine(1, 2)", "305", "registry: a positional argument overrides its default");
    ExpectValue("combine(1, c = 0)", "21", "registry: a keyword argument by name");
    ExpectValue("combine(c = 0, a = 1, b = 2)", "5", "registry: keyword arguments in any order");
    ExpectError("combine()", 3, "missing required argument 'a'", "registry: missing required");
    ExpectError("combine(1, 2, 3, 4)", 3, "at most 3", "registry: too many arguments");
    ExpectError("combine(1, a = 2)", 3, "given twice", "registry: duplicate argument");
    ExpectError("combine(1, d = 2)", 3, "no parameter named 'd'", "registry: unknown keyword");
    ExpectError("combine(\"a\")", 3, "expects Int", "registry: wrong argument type");
    ExpectError("which()", 3, "no overload", "registry: arity mismatch lists the overloads");
    ExpectError("which()", 3, "which(x: Int)", "registry: the error prints each signature");

    // print is variadic.
    ExpectValue("print()", "", "registry: variadic verb with no arguments");
    ExpectValue("print(1,2,3,4,5)", "1 2 3 4 5", "registry: variadic verb with five arguments");

    // Members and methods.
    ExpectError("stub(1).nope", 3, "has no member 'nope'", "registry: unknown member");
    ExpectError("stub(1).nope", 3, "members: value, label, computes", "registry: member list in the error");
    ExpectError("stub(1).nope()", 3, "has no method 'nope'", "registry: unknown method");
    ExpectError("stub(1).scaled()", 3, "wrong number of arguments", "registry: method arity");
    ExpectError("stub(1).scaled(\"x\")", 3, "expects an Int", "registry: method argument type");
    ExpectValue("stub(1).scaled(k = 2)", "802", "registry: method keyword argument");
    ExpectError("stub(1).scaled(j = 2)", 3, "no parameter named 'j'", "registry: method keyword name");
    ExpectError("1.value", 3, "not defined on Int", "registry: member access on a scalar");

    // Direct API: type / verb tables.
    Fixture fx;
    Check(fx.registry.find_type("Stub") != nullptr, "registry: find_type");
    Check(fx.registry.find_type("Nope") == nullptr, "registry: find_type on an unknown name");
    Check(fx.registry.find_verb("print") != nullptr, "registry: find_verb");
    Check(fx.registry.find_verb("nope") == nullptr, "registry: find_verb on an unknown name");
    const VerbEntry* which = fx.registry.find_verb("which");
    Check(which != nullptr && which->overloads.size() == 3, "registry: three overloads of 'which'");
    const std::vector<std::string> verbs = fx.registry.verb_names();
    Check(verbs.size() >= CoreVerbNames().size(), "registry: every core verb is registered");
    for (const std::string& n : CoreVerbNames()) {
        Check(fx.registry.find_verb(n) != nullptr, "registry: core verb '" + n + "' present");
    }

    // Signature rendering.
    Signature s;
    s.params = {Param::of("a", ValueKind::Int),
                Param::of("b", ValueKind::String).optional(Value::from_string("x"))};
    CheckEq(s.to_string("f"), "f(a: Int, b: String = \"x\")", "registry: signature text");
    CheckEqInt(static_cast<long long>(s.required_count()), 1, "registry: required_count");

    // ParamAccepts.
    Check(ParamAccepts(Param::any("x"), Value::from_int(1)), "registry: 'any' accepts an Int");
    Check(!ParamAccepts(Param::of("x", ValueKind::Int), Value::from_string("a")),
          "registry: an Int parameter rejects a String");
    Check(!ParamAccepts(Param::object("x", "Other"), Value::from_int(1)),
          "registry: an object parameter rejects a scalar");
}

// --- 6. settings and handles ----------------------------------------------

void TestSettingsAndHandles() {
    // The R1 settings layer, exercised directly.
    Settings settings;
    RegisterCoreSettings(settings, false);

    SettingSpec mode;
    mode.name = "final_mode";
    mode.kind = SettingKind::String;
    mode.default_value = Value::from_string("closure");
    mode.family = SettingFamily::Mathematical;
    mode.allowed = {"closure", "sturm_heuristic"};
    mode.doc = "owner decision OQ8";
    settings.register_setting(mode);

    SettingSpec cap;
    cap.name = "sturm.cap";
    cap.kind = SettingKind::Int;
    cap.default_value = Value::from_int(12);
    cap.family = SettingFamily::Mathematical;
    int applied = 0;
    cap.apply = [&applied](const Value&) { ++applied; };
    cap.validate = [](const Value& v) -> std::string {
        return v.int_value() <= 0 ? std::string("must be positive") : std::string();
    };
    settings.register_setting(cap);

    SettingSpec env;
    env.name = "tile_mult";
    env.kind = SettingKind::Int;
    env.default_value = Value::from_int(1);
    env.source = SettingSource::EnvOnly;
    env.env_var = "ENGINE_TILE_MULT";
    settings.register_setting(env);

    settings.set_baseline(settings.snapshot());

    const SourcePos pos("t.oc", 1, 1);
    Check(settings.find_spec("final_mode") != nullptr, "settings: spec is found");
    Check(settings.get("final_mode")->string_value() == "closure", "settings: default value");

    settings.set("final_mode", Value::from_string("sturm_heuristic"), pos);
    CheckEq(settings.get("final_mode")->string_value(), "sturm_heuristic", "settings: set takes");

    settings.set("sturm.cap", Value::from_int(24), pos);
    CheckEqInt(settings.get("sturm.cap")->int_value(), 24, "settings: dotted key");
    CheckEqInt(applied, 1, "settings: the apply hook ran once");

    bool threw = false;
    try { settings.set("final_mode", Value::from_string("nonsense"), pos); }
    catch (const TypeError& e) { threw = Contains(e.formatted(), "accepts only"); }
    Check(threw, "settings: a value outside `allowed` is rejected");

    threw = false;
    try { settings.set("sturm.cap", Value::from_int(0), pos); }
    catch (const TypeError& e) { threw = Contains(e.formatted(), "must be positive"); }
    Check(threw, "settings: the engine's validate hook is honoured");

    threw = false;
    try { settings.set("sturm.cap", Value::from_string("x"), pos); }
    catch (const TypeError& e) { threw = Contains(e.formatted(), "expects Int"); }
    Check(threw, "settings: a wrong-kind value is rejected");

    threw = false;
    try { settings.set("tile_mult", Value::from_int(4), pos); }
    catch (const TypeError& e) {
        threw = Contains(e.formatted(), "environment-only") &&
                Contains(e.formatted(), "ENGINE_TILE_MULT");
    }
    Check(threw, "settings: an env-only key names its variable and is never silently accepted");

    threw = false;
    try { settings.set("no_such_key", Value::from_int(1), pos); }
    catch (const TypeError& e) { threw = Contains(e.formatted(), "unknown setting"); }
    Check(threw, "settings: an unknown key is an error naming it");

    // Snapshots: the mathematical subset, `with`, equality, install/restore.
    const SettingsSnapshot snap = settings.snapshot();
    CheckEqInt(static_cast<long long>(snap.values().size()), 5, "settings: snapshot size");
    const SettingsSnapshot math = snap.mathematical();
    CheckEqInt(static_cast<long long>(math.values().size()), 2,
               "settings: exactly the two mathematical settings (owner decision R1)");
    Check(math.find("final_mode") != nullptr, "settings: final_mode is mathematical");
    Check(math.find(kSettingPrintMaxEntries) == nullptr,
          "settings: a presentation key is not mathematical");
    Check(snap.find("tile_mult") != nullptr, "settings: the snapshot includes env-only keys");
    const SettingsSnapshot changed = snap.with("sturm.cap", Value::from_int(48));
    CheckEqInt(changed.find("sturm.cap")->int_value(), 48, "settings: snapshot.with replaces");
    CheckEqInt(snap.find("sturm.cap")->int_value(), 24, "settings: snapshot.with does not mutate");
    Check(!snap.equals(changed), "settings: snapshots compare");
    Check(snap.equals(settings.snapshot()), "settings: an unchanged snapshot is equal");

    {
        // Installing is idempotent and order-independent, which is why there is
        // no RAII stack (plan Sec. 2 C4 / audit amendment A2).
        SettingsScope scope(settings, changed);
        CheckEqInt(settings.get("sturm.cap")->int_value(), 48, "settings: scope installs");
        SettingsScope inner(settings, snap);
        CheckEqInt(settings.get("sturm.cap")->int_value(), 24, "settings: nested scope installs");
    }
    CheckEqInt(settings.get("sturm.cap")->int_value(), 24, "settings: scopes restore");

    settings.set("final_mode", Value::from_string("closure"), pos);
    settings.reset_to_baseline();
    CheckEq(settings.get("final_mode")->string_value(), "closure", "settings: reset to baseline");
    CheckEqInt(settings.get("sturm.cap")->int_value(), 12, "settings: baseline value restored");

    const Record described = settings.describe();
    Check(described.has("final_mode"), "settings: describe lists every key");
    const Value* row = described.find("tile_mult");
    Check(row != nullptr && row->record_value().find("source")->string_value() == "env",
          "settings: describe marks env-only keys as source 'env'");
    const Value* mrow = described.find("final_mode");
    Check(mrow != nullptr && mrow->record_value().find("family")->string_value() == "mathematical",
          "settings: describe marks the mathematical family");

    // A snapshot that mentions a key this store does not know installs cleanly.
    Settings other;
    RegisterCoreSettings(other, false);
    other.install(snap);
    CheckEqInt(other.get(kSettingPrintMaxEntries)->int_value(), kDefaultPrintMaxEntries,
               "settings: install ignores unknown keys");

    // --- handles ----------------------------------------------------------
    // Laziness: declaring costs nothing; a computed member triggers the work.
    ExpectValue("T = stub(1)\nT.computes", "0", "handle: declaring does not compute");
    ExpectValue("T = stub(1)\nT.label", "\"stub\"", "handle: a structural member needs no compute");
    ExpectValue("T = stub(1)\nT.value\nT.computes", "401\n1", "handle: a computed member computes once");
    ExpectValue("T = stub(1)\nT.value\nT.value\nT.computes", "401\n401\n1",
                "handle: compute() runs exactly once");
    ExpectValue("T = stub(1)\ncompute(T)\nT.computes", "Stub(401)\n1", "handle: compute() verb");
    ExpectValue("T = stub(1)\nprint(state(T)[\"state\"])", "declared", "handle: state() before use");
    ExpectValue("T = stub(1)\nT.value\nprint(state(T)[\"state\"])", "401\ncomputed",
                "handle: state() after use");

    // Settings are captured at CREATION, not at compute (plan Sec. 2 C2).
    ExpectValue("set print_max_entries = 7\nT = stub(0)\nset print_max_entries = 400\nT.value",
                "7", "handle: the creation-time snapshot is what computes");
    ExpectValue("T = stub(0)\nset print_max_entries = 7\nT.value", "400",
                "handle: a later 'set' does not reach an existing handle");
    ExpectValue("set print_max_entries = 7\nT = stub(0)\nT.value\n"
                "settings()[\"print_max_entries\"][\"value\"]",
                "7\n7", "handle: the session's value is restored after compute()");

    // Failure is a bound state and is never retried (plan Sec. 2 C2 / Sec. 4.2).
    Outcome f1 = Run("T = stub_failing(1)\nT.value");
    CheckEqInt(f1.code, 3, "handle: a failing compute is a runtime error");
    Check(Contains(f1.out, "compute() failed"), "handle: the failure says so");
    Check(Contains(f1.out, "created at t.oc:1:5 by stub_failing(1)"),
          "handle: a lazy failure names the CREATION site as well as the access site");
    Check(Contains(f1.out, "t.oc:2:2"), "handle: the failure names the access site");
    Check(Contains(f1.out, "asked to fail"), "handle: the underlying message survives");

    Outcome f2 = Run("T = stub_failing(1)\n"
                     "expect_error \"compute() failed\" { T.value }\n"
                     "expect_error \"never retried\" { T.value }\n"
                     "T.computes");
    CheckEqInt(f2.code, 0, "handle: a failed handle rethrows on every access");
    CheckEq(f2.out, "1\n", "handle: a failed handle is never recomputed");
    Outcome f3 = Run("T = stub_failing(1)\nexpect_error \"x\" { T.value }\nstr(state(T)[\"state\"])");
    Check(Contains(f3.out, "failed"), "handle: the state is 'failed'");

    // A verb-time error binds nothing (plan Sec. 2 C2).
    Outcome f4 = Run("T = stub(\"not an int\")\nT");
    CheckEqInt(f4.code, 3, "handle: a verb-time type error");
    Check(Contains(f4.out, "no overload"), "handle: the type error names the overloads");

    // Keep-alive edges (plan Sec. 4.1).
    Outcome ka = Run("W = stub_wrap(stub(3))\n?W");
    Check(Contains(ka.out, "keeps alive: Stub"), "handle: the keep-alive edge is recorded");
    ExpectValue("W = stub_wrap(stub(3))\nW.label", "\"stub\"",
                "handle: an unbound intermediate stays valid");

    // R3: resumable state and resume().
    ExpectValue("T = stub_budgeted(1, 5)\nT.value\nprint(state(T)[\"resumable\"][\"resumable\"])",
                "401\ntrue", "resume: a budget cut-off is recorded");
    ExpectValue("T = stub_budgeted(1, 5)\nT.value\n"
                "print(state(T)[\"resumable\"][\"budget_setting\"])",
                "401\nstub.budget", "resume: the budget setting is named");
    ExpectValue("T = stub_budgeted(1, 5)\nT.value\n"
                "state(T)[\"resumable\"][\"progress\"][\"iterations\"]",
                "401\n5", "resume: the iteration state is persisted");
    Outcome r1 = Run("T = stub_budgeted(1, 5)\nT.value\n?T");
    Check(Contains(r1.out, "resumable: yes"), "resume: ?x reports resumability");
    ExpectValue("T = stub_budgeted(1, 5)\nT.value\nU = resume(T, 64)\nU.value",
                "401\n465", "resume: a resumed handle continues with the new budget");
    ExpectValue("T = stub_budgeted(1, 5)\nT.value\nU = resume(T, 64)\n"
                "state(U)[\"settings\"][\"stub.budget\"]",
                "401\n64", "resume: the new budget is recorded in the resumed snapshot");
    Outcome r2 = Run("T = stub_budgeted(1, 5)\nT.value\nU = resume(T, 64)\n?U");
    Check(Contains(r2.out, "resumed from: stub_budgeted(1, 5)"),
          "resume: the resumed handle links back to its parent");
    ExpectError("T = stub(1)\nT.value\nresume(T)", 3, "is complete",
                "resume: a complete handle cannot be resumed");
    ExpectError("resume(1)", 3, "no overload", "resume: resume() needs an object");

    // Object::create invariants.
    Fixture fx;
    Session session(fx.registry, false);
    ObjectPtr plain = Object::create(fx.registry.find_type("Stub"), SourcePos("f", 1, 1), "x",
                                     session.settings().snapshot(),
                                     std::make_shared<StubPayload>(), nullptr);
    Check(plain->state() == ObjectState::Computed,
          "handle: an object with no compute step is born computed");
    CheckEq(std::string(ObjectStateName(ObjectState::Failed)), "failed", "handle: state names");
    bool caught = false;
    try { plain->keep_alive(plain); } catch (const std::logic_error&) { caught = true; }
    Check(caught, "handle: an object cannot keep itself alive (CONSOLE_CHECK)");
    CheckEq(CreationSiteNote(plain), "object of type Stub created at f:1:1 by x",
            "handle: the creation-site note");
    CheckEq(CreationSiteNote(nullptr), "", "handle: no note for a null handle");
}

// --- 7. the no-transcript-freezing guard ----------------------------------

bool PathExists(const std::string& path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0;
}

bool LooksLikeRepoRoot(const std::string& dir) {
    return PathExists(dir + "/docs/matrix-engine/tools") &&
           PathExists(dir + "/src/hacs");
}

// The two independent routes of ConsumerMigrationValidation.cpp:37-51 -- the
// CMake-baked path, and walking up from the executable.
std::string FindRepoRoot() {
#ifdef OC_SOURCE_DIR
    {
        const std::string cmake_root(OC_SOURCE_DIR);
        if (!cmake_root.empty() && LooksLikeRepoRoot(cmake_root)) return cmake_root;
    }
#endif
    char buf[4096];
    const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) {
        buf[n] = '\0';
        std::string dir(buf);
        for (int up = 0; up < 8; ++up) {
            const std::size_t slash = dir.find_last_of('/');
            if (slash == std::string::npos || slash == 0) break;
            dir = dir.substr(0, slash);
            if (LooksLikeRepoRoot(dir)) return dir;
        }
    }
    if (::getcwd(buf, sizeof(buf)) != nullptr) {
        std::string dir(buf);
        for (int up = 0; up < 8; ++up) {
            if (LooksLikeRepoRoot(dir)) return dir;
            const std::size_t slash = dir.find_last_of('/');
            if (slash == std::string::npos || slash == 0) break;
            dir = dir.substr(0, slash);
        }
    }
    return std::string();
}

void CollectFiles(const std::string& dir, std::vector<std::string>& out, int depth) {
    if (depth > 8) return;
    DIR* d = ::opendir(dir.c_str());
    if (d == nullptr) return;
    while (true) {
        struct dirent* e = ::readdir(d);
        if (e == nullptr) break;
        const std::string name(e->d_name);
        if (name == "." || name == "..") continue;
        const std::string full = dir + "/" + name;
        struct stat st;
        if (::stat(full.c_str(), &st) != 0) continue;
        if (S_ISDIR(st.st_mode)) CollectFiles(full, out, depth + 1);
        else out.push_back(full);
    }
    ::closedir(d);
}

bool EndsWith(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

void TestNoTranscriptFreezing() {
    // (a) In process: a C0-only registry cannot express a pin at all.  Slot-1
    // plan Sec. 4.1, lane C: "C1 freezes no decomposition transcript before
    // E0b", and the language core is where that is cheapest to guarantee.
    Fixture fx;
    for (const std::string& forbidden : VerbsForbiddenInC0()) {
        Check(fx.registry.find_verb(forbidden) == nullptr,
              "guard: verb '" + forbidden + "' must not exist in C0");
    }
    ExpectError("hash(1)", 3, "unknown verb 'hash'", "guard: hash() is not callable");
    ExpectError("golden(\"n=10 ell=691\", \"A(T_p)^1\")", 3, "unknown verb 'golden'",
                "guard: golden() is not callable");
    const TypeInfo* stub = fx.registry.find_type("Stub");
    Check(stub != nullptr && !stub->hash,
          "guard: no C0 type fills the TypeInfo::hash slot");

    // (b) In the tree: no frozen console transcript is checked in.
    const std::string root = FindRepoRoot();
    if (root.empty()) {
        std::cout << "  [SKIP] tree arm: could not locate the source tree" << std::endl;
        return;
    }
    const std::string console_dir = root + "/src/hacs/Console";
    std::vector<std::string> files;
    CollectFiles(console_dir, files, 0);
    Check(!files.empty(), "guard: the Console directory was found and is not empty");
    // (console C1b, slot-4 WP-5) NARROWED, exactly as plan Sec. 10 Phase C1b
    // item 5 prescribes.  The pre-E0b form of this arm was "no *.expected at
    // all under Console/"; C1b ships them, frozen against the POST-E0b
    // reference.  What survives -- and what E0b actually protects -- is that no
    // frozen transcript may name a PRE-re-baseline decomposition object, i.e.
    // carry a `hecke_report/v1` (or `hecke_module_decomposition/v1`) text.
    // ConsoleScriptValidation carries the same arm for the C1 registry; this
    // one keeps it alive for a C0-only build.
    std::uint32_t frozen = 0;
    std::uint32_t stale = 0;
    for (const std::string& f : files) {
        if (EndsWith(f, ".inc")) {
            ++stale;
            std::cout << "  [FAIL] a compiled-in reference under Console/: " << f << std::endl;
            continue;
        }
        if (!EndsWith(f, ".expected") && !EndsWith(f, ".transcript")) continue;
        ++frozen;
        std::ifstream in(f.c_str(), std::ios::binary);
        std::ostringstream buf;
        buf << in.rdbuf();
        const std::string text = buf.str();
        if (text.find("hecke_report/v1") != std::string::npos ||
            text.find("hecke_module_decomposition/v1") != std::string::npos) {
            ++stale;
            std::cout << "  [FAIL] frozen transcript names a PRE-E0b schema: " << f << std::endl;
        }
    }
    CheckEqInt(stale, 0,
               "guard: no frozen console transcript names a pre-E0b decomposition object");
    Check(frozen > 0, "C1b ships frozen transcripts (" + std::to_string(frozen) + " files)");

    // (c) The architecture rule: ConsoleCore includes only the STL.  The script
    // is probed, not required -- a tree without it reports SKIP rather than
    // failing, the same convention ConsumerMigrationValidation uses for
    // check_matrix_includes.sh.
    const std::string script = root + "/docs/matrix-engine/tools/check_console_includes.sh";
    if (!PathExists(script)) {
        std::cout << "  [SKIP] check_console_includes.sh is not present in this tree" << std::endl;
        return;
    }
    if (std::system(nullptr) == 0) {
        std::cout << "  [SKIP] no command processor; the include rule was not run" << std::endl;
        return;
    }
    Check(::access(script.c_str(), X_OK) == 0, "guard: check_console_includes.sh is executable");
    const std::string cmd = "cd '" + root + "' && '" + script + "'";
    std::cout << "  $ " << cmd << std::endl;
    const int status = std::system(cmd.c_str());
    const bool ran = (status != -1) && WIFEXITED(status);
    Check(ran, "guard: check_console_includes.sh ran");
    if (ran) {
        CheckEqInt(WEXITSTATUS(status), 0,
                   "guard: ConsoleCore includes only the STL, and nothing outside "
                   "Console/ includes a Console header");
    }
}

// --- 8. the drivers: REPL, file runner, chatter capture, line input -------

void TestDrivers() {
    Fixture fx;

    // LineInput: a statement is complete when NeedsContinuation() says so
    // (plan Sec. 2 C6 -- identical under a pipe and a terminal).
    {
        std::istringstream in("a = [1,\n 2]\nb = 3\n");
        std::ostringstream prompts;
        LineInput li(in, prompts, /*show_prompts=*/true);
        std::string text;
        Check(li.read_statement(text), "driver: LineInput reads a statement");
        CheckEq(text, "a = [1,\n 2]", "driver: LineInput joins a continued line");
        Check(li.read_statement(text), "driver: LineInput reads the next statement");
        CheckEq(text, "b = 3", "driver: LineInput reads a complete line as-is");
        Check(!li.read_statement(text), "driver: LineInput reports EOF");
        Check(Contains(prompts.str(), "oc> "), "driver: LineInput prints the primary prompt");
        Check(Contains(prompts.str(), "... "), "driver: LineInput prints the continuation prompt");
    }
    {
        std::istringstream in("1\n");
        std::ostringstream prompts;
        LineInput li(in, prompts, /*show_prompts=*/false);
        std::string text;
        Check(li.read_statement(text), "driver: LineInput without prompts");
        CheckEq(prompts.str(), "", "driver: no prompt text when prompts are off");
    }

    // RunRepl: every error leaves the session usable (plan Sec. 2 C7).
    {
        Session session(fx.registry, /*verbose_default=*/true);
        std::ostringstream out;
        session.set_output(out);
        std::istringstream in("x = 20\nx + 1\n1/0\nx + 2\n");
        std::ostringstream prompts;
        const int code = RunRepl(session, in, prompts, /*show_prompts=*/false);
        CheckEqInt(code, 0, "driver: the REPL ends with 0 at EOF");
        Check(Contains(out.str(), "21"), "driver: the REPL evaluates");
        Check(Contains(out.str(), "division by zero"), "driver: the REPL reports an error");
        Check(Contains(out.str(), "22"), "driver: the REPL continues after an error");
    }
    {
        Session session(fx.registry, true);
        std::ostringstream out;
        session.set_output(out);
        std::istringstream in("quit\n1\n");
        std::ostringstream prompts;
        CheckEqInt(RunRepl(session, in, prompts, false), 0, "driver: 'quit' ends the REPL");
        CheckEq(out.str(), "", "driver: nothing after 'quit' runs");
    }
    {
        Session session(fx.registry, true);
        std::ostringstream out;
        session.set_output(out);
        std::istringstream in("exit(5)\n");
        std::ostringstream prompts;
        CheckEqInt(RunRepl(session, in, prompts, false), 5, "driver: exit(n) sets the REPL's code");
    }
    {
        // A parse error in the REPL is printed, not fatal.
        Session session(fx.registry, true);
        std::ostringstream out;
        session.set_output(out);
        std::istringstream in("1 +\n7\n");
        std::ostringstream prompts;
        CheckEqInt(RunRepl(session, in, prompts, false), 0,
                   "driver: a REPL parse error is not fatal");
        Check(Contains(out.str(), "parse error"), "driver: the REPL prints the parse error");
        Check(Contains(out.str(), "7"), "driver: the REPL runs the next statement");
    }
    {
        // The transcript log records inputs and outputs (plan Sec. 2 C5).
        Session session(fx.registry, true);
        std::ostringstream out;
        std::ostringstream log;
        session.set_output(out);
        session.set_log(&log);
        std::istringstream in("6 * 7\n");
        std::ostringstream prompts;
        RunRepl(session, in, prompts, false);
        Check(Contains(log.str(), "> 6 * 7"), "driver: the log records the input line");
        Check(Contains(log.str(), "42"), "driver: the log records the output");
    }

    // RunFile: the script's own directory is what `include` resolves against.
    {
        std::map<std::string, std::string> files;
        files["scripts/main.oc"] = "include(\"helper.oc\")\nhelper + 1\n";
        files["scripts/helper.oc"] = "helper = 41\n";
        Session session(fx.registry, false);
        std::ostringstream out;
        session.set_output(out);
        const int code = RunFile(session, "scripts/main.oc", MapLoader(files));
        CheckEqInt(code, 0, "driver: RunFile runs a script");
        CheckEq(out.str(), "42\n", "driver: include resolves against the script's directory");
        CheckEq(session.current_dir(), "", "driver: RunFile restores the previous directory");
    }
    {
        std::map<std::string, std::string> files;
        Session session(fx.registry, false);
        std::ostringstream out;
        session.set_output(out);
        CheckEqInt(RunFile(session, "nowhere.oc", MapLoader(files)),
                   static_cast<int>(ExitCode::Usage), "driver: a missing script is exit 1");
        Check(Contains(out.str(), "cannot read script"), "driver: a missing script says so");
    }

    // The exit-code table of plan Sec. 6.
    CheckEqInt(static_cast<int>(ExitCode::Ok), 0, "driver: exit code 0 = success");
    CheckEqInt(static_cast<int>(ExitCode::Usage), 1, "driver: exit code 1 = usage");
    CheckEqInt(static_cast<int>(ExitCode::Parse), 2, "driver: exit code 2 = parse");
    CheckEqInt(static_cast<int>(ExitCode::Runtime), 3, "driver: exit code 3 = runtime");
    CheckEqInt(static_cast<int>(ExitCode::Assertion), 4, "driver: exit code 4 = assertion");

    // ChatterCapture: the pipeline's own stdout/stderr is swallowed while the
    // session keeps writing to its own stream (plan Sec. 2 C5; risk R17 notes
    // that the rdbuf swap is process-global).
    {
        Session session(fx.registry, false);
        std::ostringstream out;
        session.set_output(out);
        {
            ChatterCapture capture(session);
            std::cout << "pipeline chatter";
            std::cerr << " and a warning";
            session.print_line("console output");
            CheckEq(capture.text(), "pipeline chatter and a warning",
                    "driver: cout and cerr are both captured");
        }
        CheckEq(out.str(), "console output\n",
                "driver: console output is unaffected by the capture");
        std::ostringstream probe;
        std::streambuf* old = std::cout.rdbuf(probe.rdbuf());
        std::cout << "after";
        std::cout.rdbuf(old);
        CheckEq(probe.str(), "after", "driver: std::cout is restored afterwards");
    }

    // Session variables and the scoping rule, through the API.
    {
        Session session(fx.registry, false);
        Check(session.find_var("x") == nullptr, "session: an unbound name is not found");
        session.set_var("x", Value::from_int(1));
        Check(session.find_var("x") != nullptr, "session: set_var binds");
        session.set_var("x", Value::from_int(2));
        CheckEqInt(session.find_var("x")->int_value(), 2, "session: set_var rebinds");
        CheckEqInt(static_cast<long long>(session.vars().size()), 1,
                   "session: rebinding does not add a second entry");
        Check(session.erase_var("x"), "session: erase_var removes");
        Check(!session.erase_var("x"), "session: erase_var on an unbound name");
        CheckEqInt(session.print_max_entries(), kDefaultPrintMaxEntries,
                   "session: the default print cap");
        Check(!session.verbose(), "session: scripts are quiet by default");
        Session repl(fx.registry, true);
        Check(repl.verbose(), "session: the REPL is verbose by default");
    }
}

// --- 9. the C0 (c) performance floor --------------------------------------

void TestPerformanceFloor() {
    // Plan Sec. 10, C0 acceptance (c): "Parse + evaluate of a 10 000-statement
    // synthetic script <= 0.5 s (a sanity floor, not a benchmark)."
    std::string src;
    src.reserve(200000);
    src += "t = 0\n";
    for (int i = 0; i < 10000; ++i) {
        src += "t = t + " + std::to_string(i % 97) + "\n";
    }
    src += "t;\n";

    const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    Outcome o = Run(src);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    CheckEqInt(o.code, 0, "floor: the 10 000-statement script runs");
    char buf[128];
    std::snprintf(buf, sizeof(buf), "%.4f", seconds);
    std::cout << "  10 000 statements parsed + evaluated in " << buf << " s (floor 0.5 s)"
              << std::endl;
    Check(seconds <= 0.5, std::string("floor: 10 000 statements <= 0.5 s (took ") + buf + " s)");
}

} // namespace

uint32_t ConsoleC0Validation() {
    std::cout << "=== ConsoleC0Validation (console plan Sec. 10, Phase C0; slot-1 WP-5) ==="
              << std::endl;
    g_checks = 0;
    g_failures = 0;

    TestLexing();
    TestParsing();
    TestEvaluation();
    TestPrinting();
    TestRegistry();
    TestSettingsAndHandles();
    TestNoTranscriptFreezing();
    TestDrivers();
    TestPerformanceFloor();

    std::cout << "ConsoleC0Validation: " << g_checks << " checks, " << g_failures
              << " failure(s)." << std::endl;
    if (g_failures == 0) {
        std::cout << "ConsoleC0Validation: PASSED" << std::endl;
    } else {
        std::cout << "ConsoleC0Validation: FAILED" << std::endl;
    }
    return g_failures;
}
