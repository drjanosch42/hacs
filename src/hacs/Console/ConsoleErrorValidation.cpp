// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// ConsoleErrorValidation.cpp -- see ConsoleErrorValidation.h.
//
// Compiled into cohomology_app (like every console validation), never into
// ConsoleCore: it registers test-only verbs that throw pipeline exception
// types, runs the real HeckeModuleDecomposition, probes the CUDA device and
// writes script files -- none of which the STL-only rule allows in the library.
// ===========================================================================

#include "ConsoleErrorValidation.h"

#include "Bindings/Bindings.h"
#include "Bindings/RegisterPipelineVerbs.h"
#include "Core/ConsoleCli.h"
#include "Core/ConsoleError.h"
#include "Core/Object.h"
#include "Core/Registry.h"
#include "Core/ScriptRunner.h"
#include "Core/Session.h"
#include "Core/Settings.h"
#include "Core/Value.h"

#include "../Hecke/ExactInt.h"
#include "../Hecke/HeckeModuleDecomposition.h"
#include "../Hecke/HeckeSettings.h"
#include "../HeckeExt/RegisterHeckeExtensionVerbs.h"
#include "../../Engine/DeviceBackend.h"
#include "../../ModArithmetic/EngineCheck.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

using namespace console;

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

bool Contains(const std::string& text, const std::string& needle) {
    return text.find(needle) != std::string::npos;
}

std::string FirstLine(const std::string& text) {
    const std::size_t nl = text.find('\n');
    return nl == std::string::npos ? text : text.substr(0, nl);
}

// --- the test-only verbs ------------------------------------------------------
//
// Registered on THIS validation's registry only.  Each raises exactly the C++
// type its Sec. 8 row names, so the row tests the console's handling of that
// type and not the reachability of some pipeline corner.

// Counts the compute() calls of the `__lazy_counter` stub: the "never retried"
// rule of Sec. 2 C2 is that a failed handle's compute() runs ONCE.
int g_lazy_computes = 0;

// The non-commuting pair handed to the REAL certified decomposition: A =
// diag(1, 2) has the two primary blocks span(e1), span(e2), and B = [[1,1],
// [1,1]] maps e1 to e1 + e2, so the simultaneous refinement's exact
// invariance solve fails -- which is the decomposition's own runtime
// commutation certificate (HeckeModuleDecomposition.h: "an exact-solve failure
// during refinement ... certifies that some pair of supplied operators does
// NOT commute").
void RunNonCommutingDecomposition() {
    const uint32_t ell = 7;
    std::vector<std::pair<uint64_t, MatrixT<uint32_t>>> ops;
    ops.emplace_back(2u, MatrixT<uint32_t>(2, 2, ell, {{1, 0}, {0, 2}}));
    ops.emplace_back(3u, MatrixT<uint32_t>(2, 2, ell, {{1, 1}, {1, 1}}));
    hecke::HeckeModuleDecomposition D(std::move(ops), ell);
    D.compute();
}

void RegisterTestVerbs(Registry& registry) {
    const char* g = "test";
    registry.register_verb(
        "__overflow", bind::Sig({}),
        [](CallContext&, const std::vector<Value>&) -> Value {
            throw hecke::HeckeOverflow("hecke: exact multiply overflow (ConsoleErrorValidation)");
        },
        "test-only: throws hecke::HeckeOverflow", g);
    registry.register_verb(
        "__engine_check", bind::Sig({}),
        [](CallContext&, const std::vector<Value>&) -> Value {
            const bool test_only_invariant = false;
            ENGINE_CHECK_MSG(test_only_invariant, "ConsoleErrorValidation's induced ENGINE_CHECK");
            return Value();
        },
        "test-only: fails an ENGINE_CHECK (std::logic_error)", g);
    registry.register_verb(
        "__bad_alloc", bind::Sig({}),
        [](CallContext&, const std::vector<Value>&) -> Value { throw std::bad_alloc(); },
        "test-only: throws std::bad_alloc", g);
    registry.register_verb(
        "__noncommuting", bind::Sig({}),
        [](CallContext&, const std::vector<Value>&) -> Value {
            RunNonCommutingDecomposition();
            return Value::from_string("the decomposition accepted a non-commuting pair");
        },
        "test-only: hands a non-commuting pair to hecke::HeckeModuleDecomposition", g);

    // Two lazy stub types: reading `.value` forces compute().
    {
        TypeInfo t;
        t.name = "ErrLazyNotImpl";
        t.members.push_back({"value", true, "forces compute()",
                             [](const ObjectPtr&) { return Value::from_int(0); }});
        const TypeInfo* type = registry.register_type(std::move(t));
        registry.register_verb(
            "__lazy_not_implemented", bind::Sig({}),
            [type](CallContext& ctx, const std::vector<Value>&) {
                // The RAW pipeline type, thrown from inside compute() -- as the
                // pipeline would throw it from inside a lazy cohomology / hecke /
                // decompose handle.  Built exactly as the shipped bindings build
                // their handles, through bind::LazyCompute: the core's handle
                // wrapper alone would report it as exit 3 (row r06 found that,
                // before the wrapper existed), which is the C4 hardening item.
                ObjectPtr o = Object::create(
                    type, ctx.pos, ctx.call_text, ctx.session->settings().snapshot(),
                    std::shared_ptr<void>(), bind::LazyCompute([](Object&) {
                        throw hecke::NotImplementedInThisBuild("ConsoleErrorValidation lazy stub");
                    }));
                return Value::from_object(o);
            },
            "test-only: a lazy handle whose compute() throws hecke::NotImplementedInThisBuild",
            g);
    }
    {
        TypeInfo t;
        t.name = "ErrLazyCounter";
        t.members.push_back({"value", true, "forces compute()",
                             [](const ObjectPtr&) { return Value::from_int(0); }});
        const TypeInfo* type = registry.register_type(std::move(t));
        registry.register_verb(
            "__lazy_counter", bind::Sig({}),
            [type](CallContext& ctx, const std::vector<Value>&) {
                ObjectPtr o = Object::create(type, ctx.pos, ctx.call_text,
                                             ctx.session->settings().snapshot(),
                                             std::shared_ptr<void>(), [](Object&) {
                                                 ++g_lazy_computes;
                                                 throw std::runtime_error(
                                                     "ConsoleErrorValidation counting stub");
                                             });
                return Value::from_object(o);
            },
            "test-only: a lazy handle that counts its compute() calls and throws", g);
    }
    // (slot-10 WP-12, arm (W)) A non-lazy stub whose STRUCTURAL getters raise
    // each kind the core's member access must position ([P] v3.8 Sec. 2 C7
    // site 1), and whose type-level resume hook refuses with an unpositioned
    // ConsoleError (site 2).  The getters are raw -- no bind::GateTypeOnRing
    // conversion -- so the core wrap alone classifies them.
    {
        TypeInfo t;
        t.name = "ErrMember";
        t.members.push_back({"ok", false, "a plain read",
                             [](const ObjectPtr&) { return Value::from_int(7); }});
        t.members.push_back({"raw", false, "raises std::runtime_error",
                             [](const ObjectPtr&) -> Value {
                                 throw std::runtime_error("ErrMember raw getter failure");
                             }});
        t.members.push_back({"not_impl", false, "raises hecke::NotImplementedInThisBuild",
                             [](const ObjectPtr&) -> Value {
                                 throw hecke::NotImplementedInThisBuild("ErrMember not_impl getter");
                             }});
        t.members.push_back({"logic", false, "fails an ENGINE_CHECK (std::logic_error)",
                             [](const ObjectPtr&) -> Value {
                                 const bool test_only_invariant = false;
                                 ENGINE_CHECK_MSG(test_only_invariant, "ErrMember logic getter");
                                 return Value();
                             }});
        t.members.push_back({"unpositioned", false, "raises a console TypeError with no position",
                             [](const ObjectPtr&) -> Value {
                                 throw TypeError("ErrMember unpositioned console error", SourcePos());
                             }});
        t.members.push_back({"positioned", false, "raises a console TypeError that has a position",
                             [](const ObjectPtr&) -> Value {
                                 throw TypeError("ErrMember positioned console error",
                                                 SourcePos("elsewhere.oc", 99, 7));
                             }});
        t.resume = [](const ObjectPtr&, const SettingsSnapshot&) -> ObjectPtr {
            throw TypeError("ErrMember resume hook refusal", SourcePos());
        };
        const TypeInfo* type = registry.register_type(std::move(t));
        registry.register_verb(
            "__member_stub", bind::Sig({}),
            [type](CallContext& ctx, const std::vector<Value>&) {
                ObjectPtr o = Object::create(type, ctx.pos, ctx.call_text,
                                             ctx.session->settings().snapshot());
                ResumableState rs;
                rs.resumable = true;
                rs.budget_setting = "sturm";
                rs.reason = "ConsoleErrorValidation stub";
                o->set_resumable_state(rs);
                return Value::from_object(o);
            },
            "test-only: an object whose members raise each exception kind", g);
    }
}

// The validation's registry: the shipped surface (the pipeline verbs, the
// extension verbs) plus the test-only verbs above.
struct ErrRegistry {
    Registry registry;
    ErrRegistry() {
        RegisterPipelineVerbs(registry);
        hecke_ext::RegisterHeckeExtensionVerbs(registry);
        RegisterTestVerbs(registry);
    }
};

// --- the script-file driver -------------------------------------------------

std::string g_dir;   // the per-process temporary directory

std::string WriteScript(const std::string& name, const std::string& text) {
    const std::string path = g_dir + "/" + name + ".oc";
    std::ofstream out(path.c_str(), std::ios::binary);
    out << text;
    return path;
}

struct Outcome {
    int code = -1;
    std::string text;
};

// `cohomology_app --script <path> [extra...]`, in process, on `reg`.
Outcome RunCli(ErrRegistry& reg, const std::vector<std::string>& args,
               const std::string& stdin_text = std::string()) {
    const hecke::ScopedMathSettings restore(hecke::math_settings());
    std::istringstream in(stdin_text);
    std::ostringstream out;
    Outcome o;
    o.code = RunConsole(reg.registry, args, &in, &out);
    o.text = out.str();
    return o;
}

// One Sec. 8 row: a script file, its exit code, where its error must point,
// what the error line must say.
struct Row {
    std::string name;         // the file stem, and the label
    std::string origin;       // the Sec. 8 "Origin" cell the row covers
    std::string script;
    int code = 3;
    int line = 0;             // the ERROR's position (0: not asserted)
    int col = 0;
    std::string kind;         // "assertion failed", "error", "type error", ...
    std::vector<std::string> contains;
    std::vector<std::string> absent;
    int created_line = 0;     // a lazy failure's creation site (0: none)
    int created_col = 0;
};

void RunRow(ErrRegistry& reg, const Row& row) {
    const std::string path = WriteScript(row.name, row.script);
    const Outcome o = RunCli(reg, {"--script", path});
    const std::string label = "(R) " + row.name + " [" + row.origin + "]";
    Check(o.code == row.code, label + ": exit " + std::to_string(row.code) + " (got " +
                                  std::to_string(o.code) + ")");
    // The error line: the first line of the output that carries the prefix.
    std::string err_line;
    for (std::istringstream is(o.text); std::getline(is, err_line);) {
        if (err_line.rfind(path + ":", 0) == 0) break;
        err_line.clear();
    }
    if (row.line > 0) {
        const std::string want = path + ":" + std::to_string(row.line) + ":" +
                                 std::to_string(row.col) + ": " + row.kind + ": ";
        Check(err_line.rfind(want, 0) == 0,
              label + ": the error line starts '" + want.substr(g_dir.size() + 1) + "'");
    }
    for (const std::string& s : row.contains) {
        Check(Contains(o.text, s), label + ": the output names '" + s + "'");
    }
    for (const std::string& s : row.absent) {
        Check(!Contains(o.text, s), label + ": the output does NOT contain '" + s + "'");
    }
    if (row.created_line > 0) {
        const std::string want = "created at " + path + ":" + std::to_string(row.created_line) +
                                 ":" + std::to_string(row.created_col);
        Check(Contains(o.text, want),
              label + ": the lazy failure names its creation site '" +
                  want.substr(0, 11) + "<file>" + want.substr(11 + path.size()) + "'");
    }
    const bool ok = (o.code == row.code);
    std::cout << "  (R) " << std::left << std::setw(28) << row.name << " exit " << o.code
              << (ok ? "" : " (UNEXPECTED)") << "  " << row.origin << std::endl;
    if (!ok || (row.line > 0 && err_line.empty())) {
        std::cout << "    --- output ---\n" << o.text << std::flush;
    }
}

void TestRows(ErrRegistry& reg) {
    std::vector<Row> rows;
    {
        Row r;
        r.name = "r01_assert";
        r.origin = "assert -> AssertionFailure, exit 4";
        r.script = "x = 1;\nassert(x == 2, \"row one\");\nprint(\"not reached\");\n";
        r.code = 4; r.line = 2; r.col = 1; r.kind = "assertion failed";
        r.contains = {"row one"};
        r.absent = {"not reached"};
        rows.push_back(r);
    }
    {
        Row r;
        r.name = "r02_expect_error_none";
        r.origin = "expect_error, block raises nothing -> exit 4";
        r.script = "x = 1;\nexpect_error \"anything\" {\n    x + 1\n}\n";
        r.code = 4; r.line = 2; r.col = 1; r.kind = "assertion failed";
        rows.push_back(r);
    }
    {
        Row r;
        r.name = "r03_expect_error_other";
        r.origin = "expect_error, a different error -> exit 4";
        r.script = "\nexpect_error \"no such substring\" { rank(3) }\n";
        r.code = 4; r.line = 2; r.col = 1; r.kind = "assertion failed";
        rows.push_back(r);
    }
    {
        Row r;
        r.name = "r04_overflow";
        r.origin = "exact-integer overflow -> hecke::HeckeOverflow named, exit 3";
        r.script = "print(\"before\");\n  __overflow();\n";
        r.code = 3; r.line = 2; r.col = 3; r.kind = "error";
        r.contains = {"before", "hecke::HeckeOverflow", "exact multiply overflow"};
        rows.push_back(r);
    }
    {
        Row r;
        r.name = "r05_not_implemented";
        r.origin = "a legal setting this build cannot execute -> exit 5 (field_polynomial = conway)";
        r.script = "set field_polynomial = \"conway\";\nF = canonical_field_polynomial(691, 3);\n";
        r.code = 5; r.line = 2; r.col = 5; r.kind = "not implemented in this build";
        r.contains = {"hecke::NotImplementedInThisBuild", "field_polynomial = conway"};
        r.absent = {"invalid", "unknown field_polynomial value"};
        rows.push_back(r);
    }
    {
        Row r;
        r.name = "r06_not_implemented_lazy";
        r.origin = "exit 5 through a LAZY handle whose compute() throws the raw pipeline type";
        r.script = "x = __lazy_not_implemented();\nprint(x.value);\n";
        r.code = 5; r.line = 2; r.col = 8; r.kind = "not implemented in this build";
        r.contains = {"ConsoleErrorValidation lazy stub"};
        r.created_line = 1; r.created_col = 5;
        rows.push_back(r);
    }
    {
        Row r;
        r.name = "r07_engine_check";
        r.origin = "ENGINE_CHECK -> std::logic_error, message verbatim (file:line), exit 3";
        r.script = "__engine_check();\n";
        r.code = 3; r.line = 1; r.col = 1; r.kind = "error";
        r.contains = {"std::logic_error", "ENGINE_CHECK failed", "ConsoleErrorValidation.cpp:",
                      "induced ENGINE_CHECK"};
        rows.push_back(r);
    }
    {
        Row r;
        r.name = "r08_noncommuting";
        r.origin = "pipeline contract (std::runtime_error / invalid_argument), exit 3: "
                   "non-commuting operators";
        r.script = "__noncommuting();\n";
        r.code = 3; r.line = 1; r.col = 1; r.kind = "error";
        r.contains = {"caused by: std::"};
        r.absent = {"accepted a non-commuting pair"};
        rows.push_back(r);
    }
    {
        Row r;
        r.name = "r09_model_mismatch";
        r.origin = "pipeline contract through a LAZY handle, exit 3: model mismatch";
        r.script = "H = cohomology(s1_scalar(1, 5));\nT = hecke(H, 2);\nprint(T.dims);\n";
        r.code = 3; r.line = 3; r.col = 8; r.kind = "error";
        r.contains = {"compute() failed", "no SL(2,Z) tessellation model"};
        r.created_line = 2; r.created_col = 5;
        rows.push_back(r);
    }
    {
        Row r;
        r.name = "r10_invalid_value";
        r.origin = "pipeline contract, raised by HeckeSettings (a bad value), exit 3";
        r.script = "set locality = \"nonsense\";\n";
        r.code = 3; r.line = 1; r.col = 1; r.kind = "type error";
        r.contains = {"unknown locality value"};
        rows.push_back(r);
    }
    {
        Row r;
        r.name = "r11_bad_alloc";
        r.origin = "memory -> std::bad_alloc, exit 3";
        r.script = "\n\n__bad_alloc();\n";
        r.code = 3; r.line = 3; r.col = 1; r.kind = "error";
        r.contains = {"out of memory"};
        rows.push_back(r);
    }
    {
        Row r;
        r.name = "r12_type_error";
        r.origin = "console type / arity -> TypeError with the signature list, exit 3";
        r.script = "y = rank(3);\n";
        r.code = 3; r.line = 1; r.col = 5; r.kind = "type error";
        r.contains = {"no overload of 'rank'", "rank(A: Matrix)"};
        rows.push_back(r);
    }
    {
        Row r;
        r.name = "r13_arity";
        r.origin = "console arity -> TypeError, exit 3";
        r.script = "\nz = hecke();\n";
        r.code = 3; r.line = 2; r.col = 5; r.kind = "type error";
        r.contains = {"hecke("};
        rows.push_back(r);
    }
    {
        Row r;
        r.name = "r14_parse";
        r.origin = "parse -> ParseError BEFORE execution, exit 2";
        r.script = "print(\"side effect\");\na = = 1;\n";
        r.code = 2; r.line = 2; r.col = 5; r.kind = "parse error";
        r.absent = {"side effect"};
        rows.push_back(r);
    }
    {
        Row r;
        r.name = "r15_lazy_generic";
        r.origin = "compute-time failure of a handle -> access site + creation site, exit 3";
        r.script = "# a comment line\nc = __lazy_counter();\nprint(1);\nv = c.value;\n";
        r.code = 3; r.line = 4; r.col = 6; r.kind = "error";
        r.contains = {"compute() failed", "caused by: ", "counting stub"};
        r.created_line = 2; r.created_col = 5;
        rows.push_back(r);
    }
    {
        Row r;
        r.name = "r16_ok";
        r.origin = "success, exit 0 (the control)";
        r.script = "x = 2;\nassert(x * 3 == 6, \"ok\");\n";
        r.code = 0;
        rows.push_back(r);
    }
    for (const Row& row : rows) RunRow(reg, row);
}

// --- (U) the exit-1 rows ------------------------------------------------------

void TestUsage(ErrRegistry& reg) {
    const Outcome missing = RunCli(reg, {"--script", g_dir + "/no_such_script.oc"});
    Check(missing.code == 1, "(U) a missing script file is exit 1");
    Check(Contains(missing.text, "cannot read script"), "(U) and says so");

    const Outcome undeclared = RunCli(reg, {"--set", "no_such_key=1", "-c", "print(1)"});
    Check(undeclared.code == 1, "(U) an undeclared --set key is exit 1");
    Check(Contains(undeclared.text, "unknown setting 'no_such_key'"), "(U) and names the key");
    Check(!Contains(undeclared.text, "\n1\n") && FirstLine(undeclared.text) != "1",
          "(U) and nothing ran");

    const Outcome bogus = RunCli(reg, {"--bogus-flag"});
    Check(bogus.code == 1, "(U) an unrecognised console argument is exit 1");
    Check(Contains(bogus.text, "unrecognised console argument '--bogus-flag'"),
          "(U) and names the argument");

    const Outcome ok = RunCli(reg, {"-c", "print(6 * 7)"});
    Check(ok.code == 0 && ok.text == "42\n", "(U) the control: -c 'print(6 * 7)' is exit 0");
    std::cout << "  (U) exit 1: missing file " << missing.code << ", undeclared --set key "
              << undeclared.code << ", unrecognised argument " << bogus.code << std::endl;
}

// --- (L) a failed handle: bound, rethrown on every access, never retried -----

void TestLazyRules(ErrRegistry& reg) {
    g_lazy_computes = 0;
    const std::string repl =
        "c = __lazy_counter()\n"
        "c.value\n"
        "c.value\n"
        "?c\n";
    const Outcome o = RunCli(reg, {"--console", "--quiet"}, repl);
    Check(o.code == 0, "(L) the REPL survives both failures (exit 0)");
    Check(g_lazy_computes == 1, "(L) the failed handle's compute() ran exactly ONCE (never retried), "
                                "ran " + std::to_string(g_lazy_computes));
    Check(Contains(o.text, "compute() failed"), "(L) the first access reports compute() failed");
    Check(Contains(o.text, "computation failed earlier"),
          "(L) the second access rethrows the SAME failure");
    Check(Contains(o.text, "never retried"), "(L) and says it is never retried");
    Check(Contains(o.text, "failed"), "(L) `?c` shows the handle bound in state failed");
    std::size_t sites = 0;
    for (std::size_t at = o.text.find("created at <stdin>:1:5"); at != std::string::npos;
         at = o.text.find("created at <stdin>:1:5", at + 1)) {
        ++sites;
    }
    Check(sites >= 2, "(L) both accesses name the creation site <stdin>:1:5");
    std::cout << "  (L) failed handle: compute() ran " << g_lazy_computes
              << " time(s) over two accesses; creation site named " << sites << " time(s)"
              << std::endl;
}

// --- (P) the REPL survives every error ----------------------------------------

void TestReplSurvives(ErrRegistry& reg) {
    const std::string repl =
        "y = rank(3)\n"
        "print(y)\n"
        "__bad_alloc()\n"
        "__overflow()\n"
        "__engine_check()\n"
        "a = = 1\n"
        "print(\"alive\", 6 * 7)\n";
    const Outcome o = RunCli(reg, {"--console", "--quiet"}, repl);
    Check(o.code == 0, "(P) the REPL does not fail on an error (exit 0)");
    Check(Contains(o.text, "unknown name 'y'"), "(P) a verb-time error binds nothing");
    Check(Contains(o.text, "out of memory"), "(P) std::bad_alloc is reported");
    Check(Contains(o.text, "alive 42"), "(P) and the session is still usable after every error");
}

// --- (I) post-throw integrity -------------------------------------------------

// The induced throws, each in its own chunk on ONE session (every chunk must
// fail with its own code), then the golden-grid recomputation.
const char* const kThrows[][2] = {
    {"x = hecke(cohomology(s1_scalar(1, 5)), 2);\nprint(x.dims);\n", "3"},   // model mismatch
    {"__noncommuting();\n", "3"},                                          // non-commuting ops
    {"__overflow();\n", "3"},
    {"__engine_check();\n", "3"},
    {"__bad_alloc();\n", "3"},
    {"set field_polynomial = \"conway\";\nF = canonical_field_polynomial(691, 3);\n", "5"},
    {"set field_polynomial = \"least\";\nq = __lazy_not_implemented();\nprint(q.value);\n", "5"},
};

const char* const kIntegrity =
    "POINT_COH = \"n=10 ell=691\";\n"
    "POINT_HECKE = \"n=10 ell=691 p=2 r=1\";\n"
    "assert(fingerprint() == \"0x0000000000000000\", \"the defaults are back in force\");\n"
    "H = cohomology(symmetric_power(sl2z_model(), 10, 691));\n"
    "T = hecke(H, 2);\n"
    "n = 0;\n"
    "for q in H.degrees {\n"
    "    assert(H.dims[q] == golden(POINT_COH, \"dim_H^\" + str(q)).value, \"dim_H^\" + str(q));\n"
    "    assert(hash(T.A(q)) == golden(POINT_HECKE, \"A(T_p)^\" + str(q)).hash,\n"
    "           \"A(T_p)^\" + str(q));\n"
    "    assert(hash(T.Theta(q)) == golden(POINT_HECKE, \"Theta_flat^\" + str(q)).hash,\n"
    "           \"Theta_flat^\" + str(q));\n"
    "    assert(hash(T.Theta_eq(q)) == golden(POINT_HECKE, \"Theta_eq^\" + str(q)).hash,\n"
    "           \"Theta_eq^\" + str(q));\n"
    "    n = n + 4;\n"
    "}\n"
    "assert(hash(decompose(T, 1)) == golden(POINT_HECKE, \"decomp_A(T_p)^1_invariants\").hash,\n"
    "       \"decomp_A(T_p)^1_invariants\");\n"
    "print(\"integrity\", n + 1);\n";

void TestIntegrityUnder(ErrRegistry& reg, const std::string& device) {
    const hecke::ScopedMathSettings restore(hecke::math_settings());
    std::ostringstream out;
    Session session(reg.registry, /*verbose_default=*/false);
    session.set_output(out);
    const std::string arm = "(I) device = " + device;
    int code = RunSource(session, "set device = \"" + device + "\";\n", "<integrity>");
    Check(code == 0, arm + ": `set device` is accepted");
    uint32_t induced = 0;
    for (const auto& t : kThrows) {
        out.str(std::string());
        out.clear();
        code = RunSource(session, t[0], "<induced>");
        const int want = std::atoi(t[1]);
        Check(code == want, arm + ": the induced throw '" + FirstLine(t[0]) + "' is exit " +
                                t[1] + " (got " + std::to_string(code) + ")");
        if (code == want) ++induced;
    }
    out.str(std::string());
    out.clear();
    code = RunSource(session, kIntegrity, "<integrity>");
    const std::string text = out.str();
    Check(code == 0, arm + ": after " + std::to_string(induced) +
                         " induced throws the same session recomputes the golden objects of "
                         "(10, 691), p = 2, and every one matches golden(...)");
    Check(Contains(text, "integrity 13"), arm + ": all 13 objects were compared");
    std::cout << "  " << arm << ": " << induced << " induced throws, then "
              << (code == 0 ? "13 golden objects matched" : "MISMATCH") << std::endl;
    if (code != 0) std::cout << "    --- output ---\n" << text << std::flush;
}

void TestIntegrity(ErrRegistry& reg) {
    TestIntegrityUnder(reg, "cpu");
    if (engine::tape::DeviceBackend<uint32_t>::usable()) {
        TestIntegrityUnder(reg, "gpu");
    } else {
        std::cout << "  [SKIP] (I) device = gpu: no usable CUDA device" << std::endl;
    }
}

// --- (K) coefficient_ring, once slot-8 WP-3 registers it ----------------------

void TestCoefficientRing(ErrRegistry& reg) {
    const std::vector<std::string>& keys = hecke::MathSettings::keys();
    if (std::find(keys.begin(), keys.end(), std::string("coefficient_ring")) == keys.end()) {
        std::cout << "  [SKIP] (K) coefficient_ring is not registered in this tree ("
                  << keys.size() << " mathematical keys); slot-8 WP-3 carries its own arm"
                  << std::endl;
        return;
    }
    // [E] sec:ring (a): until slot 24 D6 the value `Z` was RESERVED and refused
    // by MathSettings::set() itself (exit 5, the set statement's file:line:col).
    // (slot-24 D6, DESCENT_plan.md Sec. 4 D6, OD-10 (a)) `Z` is UN-RESERVED: the
    // `set` statement succeeds and the integral descent honours it; every FIELD
    // construction site refuses it by name, exit 5, positioned at the refusing
    // statement (here `cohomology`, line 2) -- the same exit code as before, one
    // statement later.  The (K) arm keeps its pins on that refusal.
    const std::string p1 = WriteScript("k1_ring_reserved",
                                       "set coefficient_ring = \"Z\";\n"
                                       "H = cohomology(symmetric_power(sl2z_model(), 10, 691));\n");
    const Outcome reserved = RunCli(reg, {"--script", p1});
    Check(reserved.code == 5, "(K) cohomology(...) under coefficient_ring = \"Z\" is exit 5 (got " +
                                  std::to_string(reserved.code) + ")");
    Check(Contains(reserved.text, "coefficient_ring=Z"), "(K) and the refusal names the key and the value");
    Check(Contains(FirstLine(reserved.text), p1 + ":2:"),
          "(K) and it carries the refusing statement's file:line:col (slot-9 WP-12; got '" +
              FirstLine(reserved.text) + "')");
    const std::string p1s = WriteScript("k1_ring_z_set", "set coefficient_ring = \"Z\";\nprint(\"stored\");\n");
    const Outcome zset = RunCli(reg, {"--script", p1s});
    Check(zset.code == 0 && Contains(zset.text, "stored"), "(K) set coefficient_ring = \"Z\" is legal and stored (slot-24 D6; got exit " +
                                                               std::to_string(zset.code) + ")");
    // (b), as lifted by (h): `decompose` does not honour the ring and is
    // refused at its dispatch, exit 5 -- after symmetric_power, cohomology and
    // hecke(H, 2) honoured it.
    const std::string p2 = WriteScript(
        "k2_ring_refused",
        "set coefficient_ring = \"Z/ell^2\";\n"
        "H = cohomology(symmetric_power(sl2z_model(), 10, 691));\n"
        "D = decompose(hecke(H, 2), 1);\n"
        "print(D.blocks);\n");
    const Outcome refused = RunCli(reg, {"--script", p2});
    Check(refused.code == 5, "(K) decompose under coefficient_ring = Z/ell^2 is exit 5 (got " +
                                 std::to_string(refused.code) + ")");
    Check(Contains(refused.text, "coefficient_ring") && Contains(refused.text, "decompose"),
          "(K) and the refusal names the key and the verb");
    // (slot-8 WP-13, referee M-1 -> slot-9 WP-12, decision 2 (ii) (A)) The same
    // legal-but-unexecutable value as a `--set` argument: refused before any
    // statement runs.  PIN CHANGED 1 -> 5 in slot-9 WP-12's commit ([E]
    // v1.10.0-pre.1 sec:ring (a)): it is exit 5, as on the `set` statement.
    // (slot-24 D6) `--set coefficient_ring=Z` is legal now: `print(1)` runs (exit 0), and a field
    // construction under it is the exit-5 refusal.
    const Outcome cli_z = RunCli(reg, {"-c", "print(1)", "--set", "coefficient_ring=Z"});
    Check(cli_z.code == 0 && FirstLine(cli_z.text) == "1",
          "(K) --set coefficient_ring=Z is legal since slot-24 D6 (got " + std::to_string(cli_z.code) + ")");
    const Outcome cli = RunCli(reg, {"-c", "H = cohomology(symmetric_power(sl2z_model(), 10, 691));", "--set", "coefficient_ring=Z"});
    Check(cli.code == 5, "(K) --set coefficient_ring=Z, then cohomology(...), is exit 5 (got " +
                             std::to_string(cli.code) + ")");
    Check(Contains(cli.text, "coefficient_ring"), "(K) and the refusal names the key");
    // An INVALID value and an undeclared key stay usage errors, exit 1.
    const Outcome cli_bad = RunCli(reg, {"-c", "print(1)", "--set", "coefficient_ring=Z/ell^1"});
    Check(cli_bad.code == 1, "(K) --set coefficient_ring=Z/ell^1 (invalid) stays exit 1 (got " +
                                 std::to_string(cli_bad.code) + ")");
    const Outcome cli_ok = RunCli(reg, {"-c", "print(fingerprint() != \"0x0000000000000000\")",
                                        "--set", "coefficient_ring=Z/ell^2"});
    Check(cli_ok.code == 0 && FirstLine(cli_ok.text) == "true",
          "(K) --set coefficient_ring=Z/ell^2 (legal, executable) is accepted, exit 0");

    // (slot-9 WP-12) [E] sec:ring (h), member by member: under Z/ell^2 at
    // (10, 691) every honoured verb and member succeeds (exit 0), the field
    // members of a ring Cohomology raise the object's named logic_error (exit
    // 3), and everything else exits 5 -- each in its own script, so a code is
    // the code of that one access.
    const std::string kRingPrelude =
        "M = sl2z_model();\n"
        "LF = symmetric_power(M, 10, 691);\n"
        "HF = cohomology(LF);\n"
        "DF = dual(LF);\n"
        "DEC = decompose(hecke(HF, 2), 1);\n"
        "set coefficient_ring = \"Z/ell^2\";\n"
        "L = symmetric_power(M, 10, 691);\n"
        "H = cohomology(L);\n"
        "T = hecke(H, 2);\n";
    struct RingRow {
        const char* expr;
        int code;
    };
    const RingRow rows[] = {
        // honoured (h)
        {"L", 0}, {"H", 0}, {"T", 0},
        {"H.coefficient_ring", 0}, {"H.computed", 0}, {"H.q_max", 0}, {"H.truncated", 0},
        {"H.q_max_source", 0}, {"H.degrees", 0}, {"H.lengths", 0}, {"H.invariant_factors(1)", 0},
        {"H.ladic_type_certified", 0}, {"H.r_K(0)", 0}, {"H.r_K(1)", 0}, {"H.ladic_type(1)", 0},
        {"H.bockstein_rank", 0}, {"H.higher_torsion_count", 0}, {"H.red(1)", 0},
        {"state(H)", 0}, {"save_state(H)", 0},
        {"T.coefficient_ring", 0}, {"T.A(1)", 0}, {"T.charpoly_free", 0},
        // (slot-10 WP-12; [E] v1.11 (h), [S10] decision 6 (ii) (A)) the five
        // structural members: PINS CHANGED 5 -> 0 for L.n, L.d_V, T.p, T.r;
        // L.coefficient_ring is the one new member
        {"L.n", 0}, {"L.d_V", 0}, {"L.coefficient_ring", 0}, {"T.p", 0}, {"T.r", 0},
        {"hecke(H, 691).charpoly_free", 0}, {"hash(H.invariant_factors(1))", 0},
        // the field members: the object's named std::logic_error
        {"H.dims", 3}, {"H.C(1)", 3}, {"H.Z(1)", 3}, {"H.B(1)", 3}, {"H.H(1)", 3}, {"H.P(1)", 3},
        // a complete ladic_level state is not resumable; golden() refuses (R19)
        {"resume(H, \"Z/ell^3\")", 3}, {"golden(\"n=10 ell=691\", \"dim_H^1\")", 3},
        // not on (h)'s list: exit 5
        {"H.ell", 5}, {"H.d_V", 5}, {"H.dim_C(1)", 5},
        {"L.p", 5}, {"L.relations", 5},
        {"T.cosets", 5}, {"T.certificates", 5}, {"T.json", 5}, {"T.dims", 5},
        {"T.Theta(1)", 5}, {"T.Theta_eq(1)", 5}, {"T.apply(1, identity(691, 1))", 5},
        {"T.apply_route2(1, identity(691, 1))", 5},
        {"hecke(H, 2, 2)", 5}, {"family(H)", 5}, {"decompose(T, 1)", 5}, {"report(DEC)", 5},
        {"crosschecks(T)", 5}, {"dual(L)", 5}, {"pairing(LF, DF)", 5},
        {"symmetric_power(M, 4, 2)", 5}, {"symmetric_power(M, 10, 3)", 5},
        // a handle of one ring used under another: refused, naming both
        {"hecke(HF, 2)", 5}, {"cohomology(LF)", 5},
    };
    uint32_t honoured = 0, logic = 0, refusedn = 0, row_fail = 0;
    uint32_t idx = 0;
    for (const RingRow& r : rows) {
        const std::string path =
            WriteScript("k3_ring_row_" + std::to_string(idx++), kRingPrelude + "print(" + r.expr + ");\n");
        const Outcome o = RunCli(reg, {"--script", path});
        const bool ok = (o.code == r.code);
        Check(ok, std::string("(K) (h) under Z/ell^2: ") + r.expr + " is exit " + std::to_string(r.code) +
                      " (got " + std::to_string(o.code) + ")");
        if (!ok) {
            ++row_fail;
            if (row_fail <= 3) std::cout << "    --- output of " << r.expr << " ---\n" << o.text << std::flush;
        }
        if (r.code == 0) ++honoured;
        if (r.code == 3) ++logic;
        if (r.code == 5) {
            ++refusedn;
            Check(Contains(o.text, "not honoured in this build") || Contains(o.text, "not_implemented"),
                  std::string("(K) (h): the refusal of ") + r.expr + " is a build-capability refusal");
        }
    }
    // The uncertified level (30, 5) at K = 2: the K-mode's defined-only-where-
    // certified members raise their named logic_error, and the ladic_level
    // state is incomplete -- resume refuses by the non-budget rule.
    const std::string kUncert =
        "set coefficient_ring = \"Z/ell^2\";\n"
        "H = cohomology(symmetric_power(sl2z_model(), 30, 5));\n"
        "compute(H);\n";   // the core's state / resume read a COMPUTED handle's state
    const struct {
        const char* expr;
        int code;
        const char* needle;
    } urows[] = {
        {"H.ladic_type_certified", 0, "false"},
        {"state(H).resumable.resumable", 0, "true"},
        {"H.ladic_type(1)", 3, "logic_error"},
        {"hecke(H, 2).charpoly_free", 3, "logic_error"},
        {"resume(H, \"Z/ell^3\")", 3, "FRESH run at a larger k"},
    };
    for (const auto& u : urows) {
        const std::string path = WriteScript("k4_uncert_" + std::to_string(idx++), kUncert + "print(" + u.expr + ");\n");
        const Outcome o = RunCli(reg, {"--script", path});
        Check(o.code == u.code && Contains(o.text, u.needle),
              std::string("(K) (h) at the uncertified (30, 5), K = 2: ") + u.expr + " is exit " +
                  std::to_string(u.code) + " naming '" + u.needle + "' (got " + std::to_string(o.code) + ")");
    }
    // (slot-9 WP-12 x WP-13 merge follow-up) The C5b verbs that take a
    // Cohomology handle refuse a Z/ell^k HANDLE at dispatch (exit 5), even
    // once the setting is back at F_ell -- one row per distinct dispatch path:
    // check_identities, L_map (either argument), and the lazy iota / boundary /
    // cone constructor (forced by print, so a missing gate would surface as the
    // field accessor's exit 3).
    const std::string kHandlePrelude =
        "M = sl2z_model();\n"
        "HF = cohomology(symmetric_power(M, 10, 691));\n"
        "set coefficient_ring = \"Z/ell^2\";\n"
        "H = cohomology(symmetric_power(M, 10, 691));\n"
        "set coefficient_ring = \"F_ell\";\n";
    const char* hrows[] = {
        "check_identities(H)", "L_map(H, HF)", "L_map(HF, H)", "iota(H)", "boundary(H)", "cone(H)",
    };
    uint32_t handle_refused = 0;
    for (const char* e : hrows) {
        const std::string path =
            WriteScript("k5_ring_handle_" + std::to_string(idx++), kHandlePrelude + "print(" + e + ");\n");
        const Outcome o = RunCli(reg, {"--script", path});
        const bool ok = o.code == 5 && (Contains(o.text, "not honoured in this build") ||
                                        Contains(o.text, "not_implemented"));
        Check(ok, std::string("(K) a Z/ell^2 Cohomology handle under F_ell: ") + e +
                      " is a build-capability refusal, exit 5 (got " + std::to_string(o.code) + ")");
        if (ok) ++handle_refused;
        else std::cout << "    --- output of " << e << " ---\n" << o.text << std::flush;
    }
    std::cout << "  (K) C5b ring-handle gate: " << handle_refused << "/" << (sizeof(hrows) / sizeof(hrows[0]))
              << " dispatch rows exit 5" << std::endl;

    // (slot-11 WP-18c; [E] v1.12.0-pre.2 sec:ring (a), (h), the per-cell rule)
    // THE DIGIT CELLS, pinned at the new refusal points: under Z/ell^40 at
    // (10, 691) (691^40 >= 2^64: the digit route) the digit list honours
    // (exit 0), every other member exits 5; the per-cell refusals (ell = 3
    // beyond 2^64 -- the console's own ell in {2, 3} dispatch refusal; the
    // 8192-bit cap; odd n) exit 5; the parse cap 4096 (Z/ell^4097 exit 1,
    // Z/ell^64 accepted).
    const std::string kDigitPrelude =
        "set coefficient_ring = \"Z/ell^40\";\n"
        "H = cohomology(symmetric_power(sl2z_model(), 10, 691));\n"
        "T = hecke(H, 2);\n";
    struct DRow { const char* expr; int code; const char* needle; };
    const DRow drows[] = {
        {"H.lengths", 0, "[0, 120, 0]"},
        {"H.invariant_factors(1)", 0, "[40, 40, 40]"},
        {"H.precision.precision_certified", 0, "true"},
        {"H.ladic_type(1)", 0, "free_rank: 3"},
        {"T.charpoly_free.coeffs", 0, "-2001"},
        {"H.bockstein_rank", 5, "not honoured in this build"},
        {"H.red(0)", 5, "not honoured in this build"},
        {"H.dims", 3, "a field (F_ell) accessor"},        // WP-14 pass 2 M-1: exit 3 as on every ring handle
        {"H.C(1)", 3, "a field (F_ell) accessor"},
        {"T.A(1)", 5, "not honoured in this build"},
    };
    uint32_t digit_ok = 0;
    for (const DRow& r : drows) {
        const std::string path =
            WriteScript("k6_digit_" + std::to_string(idx++), kDigitPrelude + "print(" + r.expr + ");\n");
        const Outcome o = RunCli(reg, {"--script", path});
        const bool ok = o.code == r.code && Contains(o.text, r.needle);
        Check(ok, std::string("(K) digit cell Z/691^40: ") + r.expr + " is exit " + std::to_string(r.code) +
                      " (got " + std::to_string(o.code) + ")");
        if (ok) ++digit_ok;
        else std::cout << "    --- output of " << r.expr << " ---\n" << o.text << std::flush;
    }
    const struct { const char* script; const char* needle; } drefuse[] = {
        {"set coefficient_ring = \"Z/ell^41\";\nprint(symmetric_power(sl2z_model(), 10, 3));\n", "ell = 3"},
        {"set coefficient_ring = \"Z/ell^820\";\nprint(symmetric_power(sl2z_model(), 10, 691));\n", "8250 bits"},
        {"set coefficient_ring = \"Z/ell^40\";\nprint(symmetric_power(sl2z_model(), 11, 691));\n", "n > 0 even"},
    };
    for (const auto& r : drefuse) {
        const std::string path = WriteScript("k7_digit_refuse_" + std::to_string(idx++), r.script);
        const Outcome o = RunCli(reg, {"--script", path});
        const bool ok = o.code == 5 && Contains(o.text, r.needle);
        Check(ok, std::string("(K) the per-cell rule refuses (exit 5, naming ") + r.needle + "; got " +
                      std::to_string(o.code) + ")");
        if (ok) ++digit_ok;
        else std::cout << "    --- output ---\n" << o.text << std::flush;
    }
    const Outcome cap_bad = RunCli(reg, {"-c", "print(1)", "--set", "coefficient_ring=Z/ell^4097"});
    Check(cap_bad.code == 1, "(K) --set coefficient_ring=Z/ell^4097 (beyond the parse cap 4096) is exit 1 (got " +
                                 std::to_string(cap_bad.code) + ")");
    const Outcome cap_ok = RunCli(reg, {"-c", "print(1)", "--set", "coefficient_ring=Z/ell^64"});
    Check(cap_ok.code == 0, "(K) --set coefficient_ring=Z/ell^64 is legal since the cap 4096 (got " +
                                std::to_string(cap_ok.code) + ")");
    std::cout << "  (K) digit cells (slot-11 WP-18c): " << digit_ok << "/"
              << (sizeof(drows) / sizeof(drows[0]) + sizeof(drefuse) / sizeof(drefuse[0]))
              << " rows as pinned; Z/ell^4097 exit " << cap_bad.code << ", Z/ell^64 exit " << cap_ok.code << std::endl;
    std::cout << "  (K) coefficient_ring: `Z` set exit " << zset.code << ", cohomology under Z exit " << reserved.code
              << " (statement positioned), decompose under Z/ell^2 exit " << refused.code
              << ", --set coefficient_ring=Z exit " << cli_z.code << " / with cohomology " << cli.code
              << " (slot-24 D6: Z un-reserved)"
              << ", --set Z/ell^1 exit " << cli_bad.code << "; (h) closure: " << honoured
              << " honoured, " << logic << " logic_error (exit 3), " << refusedn << " refused (exit 5), "
              << row_fail << " rows failing" << std::endl;
    if (reserved.code != 5 || refused.code != 5 || cli.code != 5) {
        std::cout << "    --- output ---\n" << reserved.text << refused.text << cli.text << std::flush;
    }
}

// --- (M) a campaign spec REFUSED FOR A RING REASON is exit 5, not 3 ---------
//
// slot-9 WP-15 phase 2 finding 2 (docs/implementation/slot9/WP-15_report.md
// Sec. 6.5, "the exit-3 observation"): Harness.cpp's VerbCampaignSpec /
// VerbCampaignSpecText used to wrap EVERY std::exception escaping
// campaign::ParseCampaignSpec{,File} in a plain RuntimeError, so a spec
// refused for a RING reason (Campaign/CampaignSpec.cpp's two
// `hecke::RefuseCoefficientRing` call sites: `ext = true` under a non-default
// `coefficient_ring`, or a grid cell with `ell^k >= 2^64`) surfaced as exit 3
// -- [P] Sec. 8's own row for `hecke::NotImplementedInThisBuild` is exit 5,
// "this build cannot do it", never "the computation failed". Fixed in
// slot-9 WP-8c, Harness.cpp only (Console/Core/ untouched): both verbs now
// catch `hecke::NotImplementedInThisBuild` ahead of the generic
// `std::exception` handler and rethrow it typed as `NotImplementedError`. This
// arm pins both refusal sites, on both verbs, plus a control that an ORDINARY
// spec error (no ring involved) keeps its old code, exit 3.
void TestCampaignSpecRingRefusal(ErrRegistry& reg) {
    // (b4): ext = true under a non-default coefficient_ring, via
    // campaign_spec_text (a spec given as a string).
    const std::string p1 = WriteScript(
        "m1_campaign_ext_ring_text",
        "campaign_spec_text(\"coefficient_ring = Z/ell^2\\next = true\\n"
        "n_sym = 10\\nell = 691\\np = 2\\nr = 1\\n\");\n");
    const Outcome ext_ring = RunCli(reg, {"--script", p1});
    Check(ext_ring.code == 5,
          "(M) campaign_spec_text: ext = true under coefficient_ring = Z/ell^2 is exit 5 (got " +
              std::to_string(ext_ring.code) + ")");
    Check(Contains(ext_ring.text, "is not honoured in this build") &&
              Contains(ext_ring.text, "ext pass"),
          "(M) and the refusal names the ring and the ext pass");

    // (b3): a grid cell with ell^k >= 2^64, via campaign_spec_text.
    const std::string p2 = WriteScript(
        "m2_campaign_cell_ring_text",
        "campaign_spec_text(\"coefficient_ring = Z/ell^41\\nn_sym = 4\\nell = 3\\n"
        "p = 2\\nr = 1\\n\");\n");
    const Outcome cell_ring = RunCli(reg, {"--script", p2});
    Check(cell_ring.code == 5,
          "(M) campaign_spec_text: a grid cell with ell^k >= 2^64 is exit 5 (got " +
              std::to_string(cell_ring.code) + ")");
    Check(Contains(cell_ring.text, "is not honoured in this build") &&
              Contains(cell_ring.text, "ell = 3"),
          "(M) and the refusal names the ring and the cell");

    // The same ext-pass refusal, reached through campaign_spec(path) (a file)
    // instead of campaign_spec_text -- the OTHER verb Harness.cpp wraps.
    const std::string spec_path = WriteScript(
        "m3_campaign_ext_ring_file",
        "coefficient_ring = Z/ell^2\next = true\nn_sym = 10\nell = 691\np = 2\nr = 1\n");
    const std::string p3 = WriteScript("m3_campaign_ext_ring_file_driver",
                                       "campaign_spec(\"" + spec_path + "\");\n");
    const Outcome file_ring = RunCli(reg, {"--script", p3});
    Check(file_ring.code == 5,
          "(M) campaign_spec(path): ext = true under coefficient_ring = Z/ell^2 is exit 5 (got " +
              std::to_string(file_ring.code) + ")");
    Check(Contains(file_ring.text, "is not honoured in this build"),
          "(M) and the refusal names the ring");

    // Control: an ORDINARY spec error (no grid keys at all -- nothing to do
    // with the ring) is unaffected, and stays exit 3.
    const std::string p4 = WriteScript("m4_campaign_plain_bad", "campaign_spec_text(\"\");\n");
    const Outcome plain_bad = RunCli(reg, {"--script", p4});
    Check(plain_bad.code == 3,
          "(M) campaign_spec_text: an ordinary spec error (no grid keys) stays exit 3 (got " +
              std::to_string(plain_bad.code) + ")");
    Check(Contains(plain_bad.text, "no jobs"), "(M) and the control names the actual defect");

    std::cout << "  (M) campaign spec ring refusal: campaign_spec_text ext-pass exit "
              << ext_ring.code << ", cell exit " << cell_ring.code << ", campaign_spec(path) exit "
              << file_ring.code << "; control (ordinary spec error) exit " << plain_bad.code
              << std::endl;
}

// --- (W) the C5 leftovers (slot-10 WP-12) -------------------------------------
//
// [S10] decision 6 (A); [P] v3.8-pre.1 Sec. 2 C7 / Sec. 3.6 / Sec. 8 and the
// "C5 leftovers" row of Sec. 10; [E] v1.11.0-pre.1 sec:ring (h).
//   (i)   a member getter's error carries the member access's file:line:col,
//         and the type-level resume refusal the resume call's -- the one
//         named Console/Core exception, two sites; NO exit code moves (at the
//         base every row below had the same code and no position);
//   (ii)  LocalSystem.n / .d_V / .coefficient_ring (new) and HeckeOperator.p /
//         .r honour on a ring handle, forcing no computation; a member read
//         follows the HANDLE's ring; every other unlisted member stays exit 5;
//         a verb handed a handle of another ring is exit 5 naming both rings;
//   (iii) run_campaign / campaign_plan run inside the WP-8c handler: a
//         ConsoleError unchanged, NotImplementedInThisBuild exit 5, anything
//         else exit 3.  No input at the base reaches the exit-5 branch of
//         either verb (recorded; no test-only hook in production code), so
//         the arm pins the reachability facts and the exit-3 / type-error
//         controls.

// Runs `text` as a script file; checks its exit code and, when `line > 0`,
// that the error line starts `<path>:<line>:<col>: <kind>: `.
Outcome LeftoverRow(ErrRegistry& reg, const std::string& name, const std::string& text, int code,
                    int line, int col, const std::string& kind,
                    const std::vector<std::string>& contains = {}) {
    const std::string path = WriteScript(name, text);
    const Outcome o = RunCli(reg, {"--script", path});
    const std::string label = "(W) " + name;
    Check(o.code == code,
          label + ": exit " + std::to_string(code) + " (got " + std::to_string(o.code) + ")");
    bool ok = (o.code == code);
    if (line > 0) {
        const std::string want = path + ":" + std::to_string(line) + ":" + std::to_string(col) +
                                 ": " + kind + ": ";
        const bool at = FirstLine(o.text).rfind(want, 0) == 0;
        Check(at, label + ": the error line starts '" + want.substr(g_dir.size() + 1) + "'");
        ok = ok && at;
    }
    for (const std::string& s : contains) {
        const bool has = Contains(o.text, s);
        Check(has, label + ": the output names '" + s + "'");
        ok = ok && has;
    }
    if (!ok) std::cout << "    --- output of " << name << " ---\n" << o.text << std::flush;
    return o;
}

void TestLeftovers(ErrRegistry& reg) {
    uint32_t rows = 0;
    // ---- (i) site 1: the core member access, on the raw-getter stub --------
    const std::string stub = "S = __member_stub();\n";
    struct PosRow {
        const char* name;
        const char* member;
        int code;
        const char* kind;
        std::vector<std::string> contains;
        bool creation_site;
    };
    const PosRow prow[] = {
        {"w01_raw", "raw", 3, "error",
         {"ErrMember.raw failed", "ErrMember raw getter failure"}, true},
        {"w02_not_impl", "not_impl", 5, "not implemented in this build",
         {"ErrMember.not_impl failed", "hecke::NotImplementedInThisBuild", "ErrMember not_impl getter"},
         true},
        {"w03_logic", "logic", 3, "error",
         {"ErrMember.logic failed", "std::logic_error", "ErrMember logic getter"}, true},
        {"w04_unpositioned", "unpositioned", 3, "type error", {"ErrMember unpositioned console error"},
         false},
        {"w05_missing", "nope", 3, "type error", {"type ErrMember has no member 'nope'"}, false},
    };
    for (const PosRow& r : prow) {
        const Outcome o = LeftoverRow(reg, r.name, stub + "v = S." + r.member + ";\n", r.code, 2, 6,
                                      r.kind, r.contains);
        if (r.creation_site) {
            const std::string want = "created at " + g_dir + "/" + r.name + ".oc:1:5";
            Check(Contains(o.text, want), std::string("(W) ") + r.name +
                                              ": the wrapped getter error names the creation site");
        }
        ++rows;
    }
    {   // an error that already carries a position keeps it
        const Outcome o = LeftoverRow(reg, "w06_positioned", stub + "v = S.positioned;\n", 3, 0, 0, "",
                                      {"elsewhere.oc:99:7: type error: ErrMember positioned console error"});
        Check(!Contains(o.text, "w06_positioned.oc:2:6"),
              "(W) w06_positioned: a positioned ConsoleError is not re-positioned");
        ++rows;
    }
    // The wrapped getter error is a console error, so expect_error sees it
    // (a raw exception escaping the getter could not be caught by a script).
    LeftoverRow(reg, "w07_expect_error",
                stub + "expect_error \"ErrMember raw getter failure\" { S.raw }\n"
                       "expect_error \"ErrMember not_impl getter\" { S.not_impl }\n"
                       "assert(S.ok == 7, \"a plain read is unaffected\");\n",
                0, 0, 0, "");
    ++rows;
    // ---- (i) site 2: the core resume verb around the type's hook ----------
    LeftoverRow(reg, "w08_resume_hook", stub + "x = resume(S);\n", 3, 2, 5, "type error",
                {"ErrMember resume hook refusal"});
    ++rows;
    // ---- (i) on the shipped bindings: GuardedMember's errors, positioned ---
    const std::string ring = "M = sl2z_model();\n"
                             "set coefficient_ring = \"Z/ell^2\";\n"
                             "L = symmetric_power(M, 10, 691);\n"
                             "H = cohomology(L);\n"
                             "T = hecke(H, 2);\n";   // 5 lines
    LeftoverRow(reg, "w09_ring_member_refused", ring + "v = L.p;\n", 5, 6, 6,
                "not implemented in this build", {"LocalSystem.p", "(h)'s list"});
    LeftoverRow(reg, "w10_ring_field_member", ring + "v = H.dims;\n", 3, 6, 6, "error",
                {"Cohomology.dims failed", "std::logic_error"});
    LeftoverRow(reg, "w11_ring_operator_member", ring + "v = T.cosets;\n", 5, 6, 6,
                "not implemented in this build", {"HeckeOperator.cosets"});
    // The real type-level resume refusal: Cohomology's hook on an uncertified
    // ladic_level state ((30, 5) at K = 2 -- the (K) arm's uncertified point).
    LeftoverRow(reg, "w12_ring_resume_hook",
                "set coefficient_ring = \"Z/ell^2\";\n"
                "H = cohomology(symmetric_power(sl2z_model(), 30, 5));\n"
                "compute(H);\n"
                "x = resume(H, \"Z/ell^3\");\n",
                3, 4, 5, "type error", {"FRESH run at a larger k"});
    rows += 4;

    // ---- (ii) the five structural members on a ring handle ---------------
    LeftoverRow(
        reg, "w13_ring_members",
        "M = sl2z_model();\n"
        "LF = symmetric_power(M, 10, 691);\n"
        "set coefficient_ring = \"Z/ell^2\";\n"
        "L = symmetric_power(M, 10, 691);\n"
        "H = cohomology(L);\n"
        "T = hecke(H, 2);\n"
        "U = hecke(H, 691);\n"
        "assert(L.n == 10 and L.d_V == 11 and L.coefficient_ring == \"Z/ell^2\", \"LocalSystem\");\n"
        "assert(T.p == 2 and T.r == 1 and U.p == 691 and U.r == 1, \"HeckeOperator\");\n"
        "assert(state(L).state == \"declared\" and state(H).state == \"declared\" and "
        "state(T).state == \"declared\", \"the five reads force no computation\");\n"
        "assert(LF.coefficient_ring == \"F_ell\" and LF.p == 691, \"a field handle under the ring\");\n"
        "set coefficient_ring = \"F_ell\";\n"
        "assert(L.n == 10 and L.coefficient_ring == \"Z/ell^2\" and T.p == 2, \"a ring handle under "
        "F_ell\");\n"
        "assert(s1_scalar(1, 5).coefficient_ring == \"F_ell\", \"a preset LocalSystem\");\n",
        0, 0, 0, "");
    ++rows;
    // A member read follows the HANDLE's ring: under F_ell, a ring handle's
    // unlisted members still refuse.
    const std::string back = ring + "set coefficient_ring = \"F_ell\";\n";   // 6 lines
    LeftoverRow(reg, "w14_own_ring_p", back + "v = L.p;\n", 5, 7, 6, "not implemented in this build",
                {"LocalSystem.p"});
    LeftoverRow(reg, "w15_own_ring_relations", back + "v = L.relations;\n", 5, 7, 6,
                "not implemented in this build", {"LocalSystem.relations"});
    rows += 2;
    // ---- (ii) the ring-mismatch rule at a verb: exit 5, naming both rings --
    struct MisRow {
        const char* name;
        const char* tail;
        std::vector<std::string> contains;
    };
    const MisRow mrows[] = {
        {"w16_mismatch_hecke_k3", "set coefficient_ring = \"Z/ell^3\";\nX = hecke(H, 3);\n",
         {"Z/ell^3 is in force", "built over Z/ell^2"}},
        {"w17_mismatch_cohomology_k3", "set coefficient_ring = \"Z/ell^3\";\nX = cohomology(L);\n",
         {"Z/ell^3 is in force", "built over Z/ell^2"}},
        {"w18_mismatch_hecke_field", "set coefficient_ring = \"F_ell\";\nX = hecke(H, 2);\n",
         {"F_ell is in force", "built over Z/ell^2"}},
        {"w19_mismatch_field_handle",
         "set coefficient_ring = \"F_ell\";\nHF = cohomology(symmetric_power(M, 10, 691));\n"
         "set coefficient_ring = \"Z/ell^2\";\nX = hecke(HF, 2);\n",
         {"Z/ell^2 is in force", "built over F_ell"}},
    };
    for (const MisRow& m : mrows) {
        const std::string text = ring + m.tail;
        const int line = static_cast<int>(std::count(text.begin(), text.end(), '\n'));
        LeftoverRow(reg, m.name, text, 5, line, 5, "not implemented in this build", m.contains);
        ++rows;
    }

    // ---- (iii) the campaign verbs ----------------------------------------
    const std::string field_spec =
        "S = campaign_spec_text(\"name = wp12\\nn_sym = 10\\nell = 691\\np = 2\\nr = 1\\n\");\n";
    // campaign_plan: a console type error (here from overload resolution,
    // before the handler runs) is unchanged -- exit 3, at the call.
    LeftoverRow(reg, "w20_plan_type_error", "G = campaign_plan(sl2z_model());\n", 3, 1, 5, "type error",
                {"no overload of 'campaign_plan'", "CampaignSpec"});
    LeftoverRow(reg, "w21_plan_ok",
                field_spec + "G = campaign_plan(S);\nassert(len(G) == 1, \"one shape group\");\n", 0, 0,
                0, "");
    // run_campaign: an ordinary failure stays exit 3, named by the verb.
    LeftoverRow(reg, "w22_run_uncreatable",
                field_spec + "R = run_campaign(S, output = \"/proc/oc_wp12_no_such_dir/x\");\n", 3, 2,
                5, "error", {"run_campaign: ", "cannot create output directory"});
    // Reachability (recorded, not a hook): the only NotImplementedInThisBuild
    // a run can raise is caught and recorded INSIDE the driver -- here the ext
    // pass's `field_polynomial = conway` refusal (the cell is "unavailable",
    // the job itself runs), and the spec's own mathematical settings are in
    // force whatever the session's ring (`inherit_settings = true` under
    // Z/ell^2).  Both runs therefore exit 0.
    const std::string out1 = g_dir + "/w23_out";
    LeftoverRow(reg, "w23_run_conway_recorded",
                "S = campaign_spec_text(\"name = wp12c\\nfield_polynomial = conway\\next = true\\n"
                "n_sym = 10\\nell = 691\\np = 2\\nr = 1\\n\");\n"
                "R = run_campaign(S, output = \"" + out1 + "\");\n"
                "assert(R.ran == 1 and R.failed == 0, \"the job runs; the ext cell is recorded "
                "unavailable\");\n",
                0, 0, 0, "");
    Check(Contains([&] {
              std::ifstream in((out1 + "/summary.tsv").c_str());
              std::stringstream ss;
              ss << in.rdbuf();
              return ss.str();
          }(), "\tok\t"),
          "(W) w23: the campaign's summary.tsv records the job ok");
    const std::string out2 = g_dir + "/w24_out";
    LeftoverRow(reg, "w24_run_inherit_ring",
                "set coefficient_ring = \"Z/ell^2\";\n" + field_spec +
                    "R = run_campaign(S, output = \"" + out2 + "\", inherit_settings = true);\n"
                    "assert(R.ran == 1 and R.failed == 0, \"the spec's settings, not the session's\");\n",
                0, 0, 0, "");
    rows += 5;
    std::cout << "  (W) C5 leftovers: " << rows
              << " scripts -- member-getter and resume-hook errors positioned (codes unchanged), "
                 "the five ring-handle members honour, ring mismatch exit 5 naming both rings, "
                 "campaign verbs' handler (exit-5 branch unreachable at the base, recorded)"
              << std::endl;
}


// --- (T) theta and the Dickson LES ([P] v3.9 Sec. 8's theta rows; slot-12 WP-8)
//
// The two theta verbs add no Sec. 8 row and no exit code: a non-default
// coefficient_ring in force or a ring Cohomology handle is exit 5, a
// mismatched pair of handles a TypeError (exit 3) naming the three numbers,
// ell < 5 the ENGINE_CHECK naming lem:h2torsion (exit 3), and a failed
// DicksonLES compute binds the handle `failed`, never retried.  A DicksonLES
// created under F_ell computes under its creation-time snapshot even when the
// ring in force has changed since (Sec. 2 C2), and odd n is not refused.
void TestTheta(ErrRegistry& reg) {
    struct TRow {
        const char* name;
        std::string script;
        int code;
        std::vector<std::string> contains;
        int line = 0, col = 0;          // the error's position (0: not asserted)
        int created_line = 0, created_col = 0;
    };
    const std::string kM = "M = sl2z_model();\n";
    const std::vector<TRow> rows = {
        {"t01_les_ring_dispatch", "set coefficient_ring = \"Z/ell^2\";\nD = dickson_les(30, 5);\n", 5,
         {"dickson_les", "coefficient_ring"}, 2, 5},
        {"t02_theta_ring_in_force",
         kM + "H = cohomology(symmetric_power(M, 30, 5));\nHu = cohomology(symmetric_power(M, 36, 5));\n"
              "set coefficient_ring = \"Z/ell^2\";\nT = theta(H, Hu);\n",
         5, {"theta", "coefficient_ring"}, 5, 5},
        {"t03_theta_ring_handle_lower",
         kM + "Hu = cohomology(symmetric_power(M, 36, 5));\nset coefficient_ring = \"Z/ell^2\";\n"
              "H = cohomology(symmetric_power(M, 30, 5));\nset coefficient_ring = \"F_ell\";\nT = theta(H, Hu);\n",
         5, {"theta"}, 6, 5},
        {"t04_theta_ring_handle_upper",
         kM + "H = cohomology(symmetric_power(M, 30, 5));\nset coefficient_ring = \"Z/ell^2\";\n"
              "Hu = cohomology(symmetric_power(M, 36, 5));\nset coefficient_ring = \"F_ell\";\nT = theta(H, Hu);\n",
         5, {"theta"}, 6, 5},
        {"t05_theta_mismatched_n",
         kM + "T = theta(cohomology(symmetric_power(M, 30, 5)), cohomology(symmetric_power(M, 32, 5)));\n", 3,
         {"type error", "n = 30, n_up = 32, ell = 5"}, 2, 5},
        {"t06_theta_mismatched_ell",
         kM + "T = theta(cohomology(symmetric_power(M, 10, 5)), cohomology(symmetric_power(M, 16, 7)));\n", 3,
         {"type error", "differ in ell"}, 2, 5},
        {"t07_theta_small_ell",
         kM + "T = theta(cohomology(symmetric_power(M, 4, 3)), cohomology(symmetric_power(M, 8, 3)));\n", 3,
         {"ENGINE_CHECK failed: ell >= 5", "lem:h2torsion"}, 2, 5},
        {"t08_les_small_ell_lazy", "D = dickson_les(10, 3);\nprint(D.n);\nprint(D.dims);\n", 3,
         {"ENGINE_CHECK failed: ell >= 5", "lem:h2torsion", "compute() failed"}, 3, 8, 1, 5},
        {"t09_les_snapshot_rule",
         "D = dickson_les(16, 7);\nset coefficient_ring = \"Z/ell^2\";\nassert(D.exact and D.certificate == 0, "
         "\"the creation-time F_ell snapshot\");\n",
         0, {}},
        {"t10_theta_odd_n",
         kM + "T = theta(cohomology(symmetric_power(M, 11, 5)), cohomology(symmetric_power(M, 17, 5)));\n"
              "assert(T.ranks == [0, 0, 0] and T.twist_check == 0 and T.u_ell_zero, \"odd n: every term 0\");\n"
              "D = dickson_les(11, 5);\nassert(D.exact and D.dims.H1_n == 0, \"odd n is not refused\");\n",
         0, {}},
    };
    uint32_t ok_rows = 0;
    for (const TRow& r : rows) {
        const std::string path = WriteScript(r.name, r.script);
        const Outcome o = RunCli(reg, {"--script", path});
        const std::string label = std::string("(T) ") + r.name;
        bool ok = o.code == r.code;
        Check(o.code == r.code, label + ": exit " + std::to_string(r.code) + " (got " + std::to_string(o.code) + ")");
        for (const std::string& s : r.contains) {
            const bool c = Contains(o.text, s);
            ok = ok && c;
            Check(c, label + ": the output names '" + s + "'");
        }
        if (r.line > 0) {
            const std::string want = path + ":" + std::to_string(r.line) + ":" + std::to_string(r.col) + ": ";
            const bool c = Contains(o.text, want);
            ok = ok && c;
            Check(c, label + ": the error carries its file:" + std::to_string(r.line) + ":" + std::to_string(r.col));
        }
        if (r.created_line > 0) {
            const std::string want = "created at " + path + ":" + std::to_string(r.created_line) + ":" +
                                     std::to_string(r.created_col);
            const bool c = Contains(o.text, want);
            ok = ok && c;
            Check(c, label + ": the lazy failure names its creation site");
        }
        if (ok) ++ok_rows;
        else std::cout << "    --- output of " << r.name << " ---\n" << o.text << std::flush;
    }
    // A failed DicksonLES handle: bound, rethrown on every access, never retried.
    const std::string repl = "D = dickson_les(10, 3)\nD.dims\nD.exact\n?D\n";
    const Outcome o = RunCli(reg, {"--console", "--quiet"}, repl);
    Check(o.code == 0, "(T) the REPL survives both failures of a DicksonLES handle (exit 0)");
    Check(Contains(o.text, "compute() failed"), "(T) the first access reports compute() failed");
    Check(Contains(o.text, "computation failed earlier"), "(T) the second access rethrows the SAME failure");
    Check(Contains(o.text, "never retried"), "(T) and says it is never retried");
    std::cout << "  (T) theta / DicksonLES: " << ok_rows << "/" << rows.size()
              << " script rows as specified (ring exit 5, mismatch / ell < 5 exit 3, snapshot rule, odd n); "
                 "a failed handle never retried"
              << std::endl;
}

// --- (HR) hecke_ring_algebra (slot-16 WP-6) -------------------------------------
// Refusal codes of the Z/ell^K Hecke-algebra verb: a non-default coefficient_ring
// exit 5 (dispatch), ell < 5 / odd n / a non-prime ell / kmax out of range exit 3,
// ell^kmax >= 2^64 (the digit route, not bound here) exit 5; a good call exit 0.
void TestRingAlgebra(ErrRegistry& reg) {
    struct RRow {
        const char* name;
        std::string script;
        int code;
        std::vector<std::string> contains;
        int line = 0, col = 0;
    };
    const std::vector<RRow> rows = {
        {"r01_ring_in_force", "set coefficient_ring = \"Z/ell^2\";\nR = hecke_ring_algebra(10, 7);\n", 5,
         {"hecke_ring_algebra", "coefficient_ring"}, 2, 5},
        {"r02_small_ell", "R = hecke_ring_algebra(10, 3);\n", 3, {"ENGINE_CHECK failed: ell >= 5", "lem:h2torsion"}, 1, 5},
        {"r03_odd_n", "R = hecke_ring_algebra(11, 7);\n", 3, {"n >= 1 even"}, 1, 5},
        {"r04_not_prime", "R = hecke_ring_algebra(10, 9);\n", 3, {"type error", "ell must be a prime"}, 1, 5},
        {"r05_kmax_range", "R = hecke_ring_algebra(10, 7, 1);\n", 3, {"type error", "kmax must lie in 2..12"}, 1, 5},
        {"r06_digit_route", "R = hecke_ring_algebra(10, 691, 8);\n", 5, {"digit route"}, 1, 5},
        {"r07_good_call", "R = hecke_ring_algebra(16, 7, 2);\nassert(R.ell == 7 and len(R.blocks) == len(R.lines), \"ok\");\n", 0, {}},
        // slot-17 WP-8 (R17-2 finding): the module route's U_ell is gated by the frozen `u_ell` model -- a value-free exit-5 refusal
        // before the run (45.8 GB predicted at (608, 547)); `off` refuses too; a pinned budget that admits the cell runs.
        {"r08_u_ell_over_budget", "R = hecke_ring_algebra(608, 547, 2);\n", 5, {"u_ell_rss/v2", "over the `u_ell` budget"}, 1, 5},
        {"r09_u_ell_off", "set u_ell = \"off\";\nR = hecke_ring_algebra(16, 7, 2);\n", 5, {"`u_ell` setting is off"}, 2, 5},
        {"r10_u_ell_pinned_ok", "set u_ell = \"budget_mb=64\";\nR = hecke_ring_algebra(16, 7, 2);\nassert(R.ell == 7, \"ok\");\n", 0, {}},
    };
    uint32_t ok_rows = 0;
    for (const RRow& r : rows) {
        const std::string path = WriteScript(r.name, r.script);
        const Outcome o = RunCli(reg, {"--script", path});
        const std::string label = std::string("(HR) ") + r.name;
        bool ok = o.code == r.code;
        Check(o.code == r.code, label + ": exit " + std::to_string(r.code) + " (got " + std::to_string(o.code) + ")");
        for (const std::string& s : r.contains) {
            const bool c = Contains(o.text, s);
            ok = ok && c;
            Check(c, label + ": the output names '" + s + "'");
        }
        if (r.line > 0) {
            const std::string want = path + ":" + std::to_string(r.line) + ":" + std::to_string(r.col) + ": ";
            const bool c = Contains(o.text, want);
            ok = ok && c;
            Check(c, label + ": the error carries its file:" + std::to_string(r.line) + ":" + std::to_string(r.col));
        }
        if (ok) ++ok_rows;
        else std::cout << "    --- output of " << r.name << " ---\n" << o.text << std::flush;
    }
    std::cout << "  (HR) hecke_ring_algebra: " << ok_rows << "/" << rows.size()
              << " script rows as specified (ring exit 5, ell < 5 / odd n / not prime / kmax exit 3, digit route exit 5)"
              << std::endl;
}

} // namespace

// --- (Q9) the wide band 2^32 <= ell^k < 2^64, served (slot-12 WP-17) ----------
//
// [P] v3.9.3 Q9 decided (a) (owner, 2026-09-26): the console serves
// Z/ell^k with 2^32 <= ell^k < 2^64 at ell >= 5 through the native pipeline's
// uint64_t instantiation, exactly as the campaign job loop does
// (Campaign/CampaignRing.cpp, RunRingPipeline<uint64_t>).  Before WP-17 these
// cells refused (the uint32_t ring context, exit 5, "the console's uint32_t
// gap") while the campaign served them.  This arm pins (1) the routing at the
// four band edges on both primes (5^13 | 5^14, 5^27 | 5^28, 691^3 | 691^4,
// 691^6 | 691^7: IsWideCell, every cell served, `.precision` -- the digit-only
// member -- exit 5 below 2^64 and exit 0 beyond); (2) VALUE IDENTITY with the
// campaign on seven band cells: the console runs the campaign itself
// (run_campaign), then builds the same cell by the verbs, and every object of
// the job's hecke_ring_report/v1 the console can hash (lengths, invariant
// factors, r_K, the Bockstein pair, red_H^q) and the free charpoly are equal
// -- including Z/19^15 > 2^63, where entries print as decimal Strings; (3) the
// member closure on a wide handle: the field members exit 3, the refusals of
// GateTypeOnRing exit 5, exactly as on a uint32_t ring handle.

std::string ExtractAfter(const std::string& text, const std::string& key, std::size_t from, char stop) {
    const std::size_t a = text.find(key, from);
    if (a == std::string::npos) return std::string();
    const std::size_t b = text.find(stop, a + key.size());
    if (b == std::string::npos) return std::string();
    return text.substr(a + key.size(), b - a - key.size());
}

void TestWideBand(ErrRegistry& reg) {
    namespace fs = std::filesystem;
    uint32_t idx = 0;
    // (1) the routing at the band edges.
    struct Edge { uint32_t ell, k; bool wide, digit; };
    const Edge edges[] = {
        {5, 13, false, false}, {5, 14, true, false}, {5, 27, true, false}, {5, 28, false, true},
        {691, 3, false, false}, {691, 4, true, false}, {691, 6, true, false}, {691, 7, false, true},
    };
    uint32_t edge_ok = 0;
    for (const Edge& e : edges) {
        const std::string tag = std::to_string(e.ell) + "^" + std::to_string(e.k);
        const bool ok = console::bind::IsWideCell(e.ell, e.k) == e.wide;
        Check(ok, "(Q9) IsWideCell(" + tag + ") == " + (e.wide ? "true" : "false"));
        const std::string pre = "set coefficient_ring = \"Z/ell^" + std::to_string(e.k) + "\";\n"
                                "H = cohomology(symmetric_power(sl2z_model(), 10, " + std::to_string(e.ell) + "));\n";
        const Outcome served = RunCli(reg, {"--script", WriteScript("q1_edge_" + std::to_string(idx++),
                                                                    pre + "print(H.lengths);\n")});
        const bool s_ok = served.code == 0 && Contains(served.text, "[0, ");
        Check(s_ok, "(Q9) Z/" + tag + " at (10, " + std::to_string(e.ell) + "): H.lengths is served, exit 0 (got " +
                        std::to_string(served.code) + ")");
        if (!s_ok) std::cout << "    --- output ---\n" << served.text << std::flush;
        const Outcome prec = RunCli(reg, {"--script", WriteScript("q1_edge_" + std::to_string(idx++),
                                                                  pre + "print(H.precision.precision_certified);\n")});
        const bool p_ok = e.digit ? (prec.code == 0 && Contains(prec.text, "true")) : prec.code == 5;
        Check(p_ok, "(Q9) Z/" + tag + ": H.precision is exit " + (e.digit ? "0 (the digit route)" : "5 (native)") +
                        " (got " + std::to_string(prec.code) + ")");
        if (ok && s_ok && p_ok) ++edge_ok;
    }
    // The uint64_t route is really taken: an A(1) entry above 2^32 at Z/5^14.
    {
        const Outcome o = RunCli(reg, {"--script", WriteScript("q1_a64_" + std::to_string(idx++),
            "set coefficient_ring = \"Z/ell^14\";\n"
            "print(hecke(cohomology(symmetric_power(sl2z_model(), 10, 5)), 2).A(1));\n")});
        Check(o.code == 0 && Contains(o.text, "6103515601"),
              "(Q9) Z/5^14: A^(1)(T_2) carries -24 mod 5^14 = 6103515601 > 2^32 (got exit " +
                  std::to_string(o.code) + ")");
    }

    // (2) value identity with the campaign job loop.
    struct Cell { uint32_t n, ell, k, p; };
    const Cell cells[] = {
        {10, 5, 14, 2}, {10, 5, 20, 2}, {10, 5, 27, 2}, {10, 691, 4, 2},
        {10, 691, 6, 2}, {10, 19, 15, 2}, {30, 5, 14, 5},
    };
    uint32_t cell_ok = 0, objects_equal = 0;
    for (const Cell& c : cells) {
        const std::string tag = "(" + std::to_string(c.n) + ", " + std::to_string(c.ell) + ") over Z/" +
                                std::to_string(c.ell) + "^" + std::to_string(c.k) + ", p = " + std::to_string(c.p);
        const std::string out = g_dir + "/q2_campaign_" + std::to_string(idx++);
        const std::string K = std::to_string(c.k);
        const std::string script =
            "S = campaign_spec_text(\"coefficient_ring = Z/ell^" + K + "\\nn_sym = " + std::to_string(c.n) +
            "\\nell = " + std::to_string(c.ell) + "\\np = " + std::to_string(c.p) + "\\nr = 1\\n\");\n"
            "R = run_campaign(S, output = \"" + out + "\");\n"
            "print(\"RAN\", R.ran, R.failed);\n"
            "set coefficient_ring = \"Z/ell^" + K + "\";\n"
            "H = cohomology(symmetric_power(sl2z_model(), " + std::to_string(c.n) + ", " + std::to_string(c.ell) + "));\n"
            "for q in [0, 1, 2] {\n"
            "  print(\"OBJ lengths_H^\" + str(q), hash(H.lengths[q]));\n"
            "  print(\"OBJ invariant_factors_H^\" + str(q), hash(H.invariant_factors(q)));\n"
            "  print(\"OBJ red_H^\" + str(q), hash(H.red(q)));\n"
            "}\n"
            "print(\"OBJ r_K(d^0)\", hash(H.r_K(0)));\n"
            "print(\"OBJ r_K(d^1)\", hash(H.r_K(1)));\n"
            "print(\"OBJ bockstein_rank^0\", hash(H.bockstein_rank));\n"
            "print(\"OBJ higher_torsion_count\", hash(H.higher_torsion_count));\n"
            "C = hecke(H, " + std::to_string(c.p) + ").charpoly_free;\n"
            "S2 = \"\";\n"
            "for x in C.coeffs { S2 = S2 + str(x) + \",\"; }\n"
            "print(\"CP\", C.precision, S2);\n";
        const Outcome o = RunCli(reg, {"--script", WriteScript("q2_cell_" + std::to_string(idx++), script)});
        bool ok = o.code == 0 && Contains(o.text, "RAN 1 0");
        Check(ok, "(Q9) " + tag + ": the console ran the campaign job and the verbs, exit 0 (got " +
                      std::to_string(o.code) + ")");
        std::string job;
        std::error_code ec;
        if (fs::is_directory(out + "/jobs", ec)) {
            for (const auto& ent : fs::directory_iterator(out + "/jobs", ec)) {
                std::ifstream in(ent.path().string().c_str(), std::ios::binary);
                std::ostringstream ss;
                ss << in.rdbuf();
                job = ss.str();
            }
        }
        Check(!job.empty() && Contains(job, "\"status\":\"ok\""), "(Q9) " + tag + ": the campaign job record is ok");
        ok = ok && !job.empty();
        // Every OBJ line against the record's object of the same name.
        uint32_t seen = 0;
        std::istringstream lines(o.text);
        std::string line;
        std::string cp_console;
        while (std::getline(lines, line)) {
            if (line.rfind("CP ", 0) == 0) cp_console = line.substr(3);
            if (line.rfind("OBJ ", 0) != 0) continue;
            const std::size_t sp = line.rfind(' ');
            const std::string name = line.substr(4, sp - 4);
            const std::string h = line.substr(sp + 1);
            const std::size_t at = job.find("\"name\":\"" + name + "\"");
            const std::string want =
                at == std::string::npos ? std::string() : "0x" + ExtractAfter(job, "\"fnv1a64\":\"", at, '"');
            const bool eq = at != std::string::npos && want == h;
            Check(eq, "(Q9) " + tag + ": console " + name + " " + h + " == campaign " + want);
            if (eq) ++objects_equal;
            ok = ok && eq;
            ++seen;
        }
        Check(seen == 13, "(Q9) " + tag + ": 13 objects compared (got " + std::to_string(seen) + ")");
        // The free charpoly: precision and every coefficient's decimal text.
        const std::size_t at = job.find("\"charpoly_free\":{");
        std::string cp_campaign;
        if (at != std::string::npos) {
            cp_campaign = ExtractAfter(job, "\"precision\":", at, ',') + " ";
            std::string coeffs = ExtractAfter(job, "\"coeffs\":[", at, ']');
            coeffs.erase(std::remove(coeffs.begin(), coeffs.end(), '"'), coeffs.end());
            cp_campaign += coeffs + ",";
        }
        const bool cp_eq = !cp_campaign.empty() && cp_console == cp_campaign;
        Check(cp_eq, "(Q9) " + tag + ": charpoly_free console '" + cp_console + "' == campaign '" + cp_campaign + "'");
        ok = ok && cp_eq && seen == 13;
        if (!ok) std::cout << "    --- output ---\n" << o.text << std::flush;
        if (ok) ++cell_ok;
    }

    // (3) the member closure on a wide handle (Z/691^4 at (10, 691)).
    const std::string kWide =
        "M = sl2z_model();\n"
        "set coefficient_ring = \"Z/ell^4\";\n"
        "L = symmetric_power(M, 10, 691);\n"
        "H = cohomology(L);\n"
        "T = hecke(H, 2);\n";
    const struct { const char* expr; int code; } wrows[] = {
        {"H.lengths", 0}, {"H.invariant_factors(1)", 0}, {"H.q_max", 0}, {"H.degrees", 0},
        {"H.truncated", 0}, {"H.q_max_source", 0}, {"H.coefficient_ring", 0}, {"H.ladic_type_certified", 0},
        {"H.r_K(1)", 0}, {"H.ladic_type(1)", 0}, {"H.bockstein_rank", 0}, {"H.higher_torsion_count", 0},
        {"H.red(1)", 0}, {"state(H)", 0}, {"H", 0},
        {"L.n", 0}, {"L.d_V", 0}, {"L.coefficient_ring", 0},
        {"T.p", 0}, {"T.r", 0}, {"T.coefficient_ring", 0}, {"T.A(1)", 0}, {"T.charpoly_free", 0},
        {"hecke(H, 691).charpoly_free", 0},
        {"H.dims", 3}, {"H.C(1)", 3}, {"H.Z(1)", 3}, {"H.B(1)", 3}, {"H.H(1)", 3}, {"H.P(1)", 3},
        {"resume(H, \"Z/ell^5\")", 3},
        {"H.ell", 5}, {"H.d_V", 5}, {"H.dim_C(1)", 5}, {"H.precision", 5},
        {"L.p", 5}, {"L.relations", 5},
        {"T.cosets", 5}, {"T.certificates", 5}, {"T.json", 5}, {"T.dims", 5}, {"T.Theta(1)", 5},
        {"T.Theta_eq(1)", 5}, {"hecke(H, 2, 2)", 5}, {"family(H)", 5}, {"decompose(T, 1)", 5},
        {"crosschecks(T)", 5}, {"dual(L)", 5},
    };
    uint32_t wide_ok = 0;
    for (const auto& r : wrows) {
        const Outcome o = RunCli(reg, {"--script", WriteScript("q3_wide_" + std::to_string(idx++),
                                                               kWide + "print(" + r.expr + ");\n")});
        const bool ok = o.code == r.code;
        Check(ok, std::string("(Q9) wide handle Z/691^4: ") + r.expr + " is exit " + std::to_string(r.code) +
                      " (got " + std::to_string(o.code) + ")");
        if (ok) ++wide_ok;
        else std::cout << "    --- output of " << r.expr << " ---\n" << o.text << std::flush;
    }
    std::cout << "  (Q9) wide band (slot-12 WP-17): edges " << edge_ok << "/" << (sizeof(edges) / sizeof(edges[0]))
              << " routed; campaign identity " << cell_ok << "/" << (sizeof(cells) / sizeof(cells[0]))
              << " cells (" << objects_equal << " objects equal); wide-handle closure " << wide_ok << "/"
              << (sizeof(wrows) / sizeof(wrows[0])) << " rows as pinned" << std::endl;
}

uint32_t ConsoleErrorValidation() {
    g_checks = 0;
    g_failures = 0;
    const auto t0 = std::chrono::steady_clock::now();
    std::cout << "--- ConsoleErrorValidation (console C4, [P] Sec. 8 / Sec. 2 C7 / Sec. 10 C4 (b)) ---"
              << std::endl;
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path base = fs::temp_directory_path(ec);
    g_dir = ((ec ? fs::path(".") : base) / ("oc_console_c4_err_" + std::to_string(::getpid())))
                .string();
    fs::remove_all(g_dir, ec);
    fs::create_directories(g_dir, ec);

    {
        ErrRegistry reg;
        TestRows(reg);
        TestUsage(reg);
        TestLazyRules(reg);
        TestReplSurvives(reg);
        TestIntegrity(reg);
        TestCoefficientRing(reg);
        TestCampaignSpecRingRefusal(reg);
        TestLeftovers(reg);
        TestTheta(reg);
        TestRingAlgebra(reg);
        TestWideBand(reg);   // (slot-12 WP-17) [P] Q9 decided (a)
    }
    Check(hecke::math_settings().fingerprint() == 0,
          "the host's mathematical settings are at their defaults afterwards");

    fs::remove_all(g_dir, ec);
    const double wall =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "  ConsoleErrorValidation " << (g_failures == 0 ? "PASSED" : "FAILED") << " ("
              << g_checks << " checks, " << g_failures << " failing, " << std::fixed
              << std::setprecision(1) << wall << " s)" << std::endl;
    return g_failures;
}
