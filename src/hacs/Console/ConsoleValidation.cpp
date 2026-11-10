// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// ConsoleValidation.cpp -- see ConsoleValidation.h.
//
// Compiled into cohomology_app (like ConsoleC0Validation.cpp), never into
// ConsoleCore: it walks directories through POSIX <dirent.h>, shells out with
// std::system, and constructs matrices -- all three of which the STL-only rule
// forbids inside the library.
// ===========================================================================

#include "ConsoleValidation.h"

#include "Bindings/Bindings.h"
#include "Bindings/RegisterPipelineVerbs.h"
#include "Core/ConsoleCli.h"
#include "Core/ConsoleError.h"
#include "Core/CoreVerbs.h"
#include "Core/Object.h"
#include "Core/Printer.h"
#include "Core/Registry.h"
#include "Core/ScriptRunner.h"
#include "Core/Session.h"
#include "Core/Settings.h"
#include "Core/Value.h"

#include "../Golden/GoldenHarness.h"
#include "../HeckeExt/RegisterHeckeExtensionVerbs.h"
#include "../Hecke/ExactInt.h"
#include "../Hecke/HeckeSettings.h"
#include "../../ModArithmetic/Fnv.h"
#include "../Util/TextIo.h"

#include <algorithm>
#include <filesystem>
#include <cstdio>
#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {
// Termux has no writable /tmp: honour $TMPDIR (std::filesystem::temp_directory_path), falling back to /tmp.
std::string TmpDir() {
    std::error_code ec;
    const auto p = std::filesystem::temp_directory_path(ec);
    return ec ? std::string("/tmp") : p.string();
}


using namespace console;

// --- check harness (the ConsoleC0Validation shape) -------------------------

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

void CheckEqInt(long long got, long long want, const std::string& label) {
    const bool ok = (got == want);
    Check(ok, ok ? label : label + "  [got: " + std::to_string(got) +
                               " want: " + std::to_string(want) + "]");
}

bool Contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

// --- the source tree (the two independent routes of
//     ConsumerMigrationValidation.cpp:37-51) -------------------------------

bool PathExists(const std::string& path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0;
}

bool LooksLikeRepoRoot(const std::string& dir) {
    return PathExists(dir + "/docs/matrix-engine/tools") &&
           PathExists(dir + "/src/hacs");
}

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

std::string SelfPath() {
    char buf[4096];
    const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) return std::string();
    buf[n] = '\0';
    return std::string(buf);
}

using oc_util::ReadFile;   // S24 CL-7 (F11): Util/TextIo.h
std::vector<std::string> SplitLines(const std::string& text) {
    std::vector<std::string> out;
    std::string line;
    for (char c : text) {
        if (c == '\n') { out.push_back(line); line.clear(); }
        else line += c;
    }
    if (!line.empty()) out.push_back(line);
    return out;
}

// Under a sanitizer build LeakSanitizer appends its own exit code 1 to EVERY
// child process, and `--bench micro --help` reaches main.cpp's lazy CUDA hook
// probe, whose driver (`/usr/lib/wsl/lib/libcuda.so.1`) leaks ~51 KB in five
// allocations on this box -- a PRE-EXISTING finding in a third-party library,
// with no frame of this tree in it, which would otherwise turn every spawned
// exit code into 1.  Leak detection is therefore switched off FOR THE CHILDREN
// ONLY, and only on a sanitizer build: the console's own memory behaviour is
// covered in process by ConsoleScriptValidation, which runs every binding and
// every script under the full sanitizer.  On an ordinary build this adds
// nothing to the command line.
#if defined(__SANITIZE_ADDRESS__)
#define OC_CONSOLE_CHILD_SANITIZER_PREFIX "export ASAN_OPTIONS=detect_leaks=0; "
#else
#define OC_CONSOLE_CHILD_SANITIZER_PREFIX ""
#endif

// Runs `cmd`, capturing stdout (stderr is left alone, as the frozen files were
// captured with 2>/dev/null).  Returns the exit status, or -1.
int RunCapturing(const std::string& cmd, std::string& out) {
    out.clear();
    // `export` (not a `VAR=val` prefix): `cmd` may itself be a `cd ... && ...`
    // chain, and a command-local assignment would not survive the `&&`.
    const std::string full =
        std::string(OC_CONSOLE_CHILD_SANITIZER_PREFIX) + cmd + " 2>/dev/null";
    FILE* pipe = ::popen(full.c_str(), "r");
    if (pipe == nullptr) return -1;
    char buf[4096];
    while (std::fgets(buf, sizeof(buf), pipe) != nullptr) out += buf;
    const int status = ::pclose(pipe);
    if (status == -1 || !WIFEXITED(status)) return -1;
    return WEXITSTATUS(status);
}

std::string Shq(const std::string& s) { return "'" + s + "'"; }

// `print_usage` echoes argv[0] verbatim (main.cpp), so the frozen
// --help output depends on HOW the binary was invoked.  Every frozen case is
// therefore run the way the files were captured: from the executable's own
// directory, spelled `./hacs` (since R1; `./cohomology_app` before).
// (S26 INTEG, R1 audit M2) /proc/self/exe names the real FILE, which need not
// be called `hacs` (e.g. a copy published as a regular file `cohomology_app`).
// Then the child is started with argv[0] = "./hacs" via bash's `exec -a`, so
// the captured help is the frozen one whatever the file is called.
int RunAsFrozen(const std::string& exe, const std::string& args, std::string& out) {
    const std::size_t slash = exe.find_last_of('/');
    const std::string dir = (slash == std::string::npos) ? std::string(".") : exe.substr(0, slash);
    const std::string base = (slash == std::string::npos) ? exe : exe.substr(slash + 1);
    if (base == "hacs" || base.find('\'') != std::string::npos)
        return RunCapturing("cd " + Shq(dir) + " && ./" + base + " " + args, out);
    return RunCapturing("cd " + Shq(dir) + " && exec bash -c 'exec -a ./hacs ./" + base + " \"$@\"' oc " + args, out);
}

// ===========================================================================
// Part 1 -- ConsoleScriptValidation
// ===========================================================================

// A session over its OWN registry, so a test can register a stub type without
// touching the process-wide one.
struct Fixture {
    Registry registry;
    std::ostringstream out;
    Session session;

    Fixture() : session(registry, /*verbose_default=*/false) {
        RegisterPipelineVerbs(registry);
        // (C1b) The Session is a MEMBER, so it is constructed BEFORE this body
        // runs and therefore before the bindings install the engine-owned
        // settings on the registry (Registry.h, the settings-installer seam).
        // main.cpp's console branch has no such problem -- it registers first
        // and RunConsole builds the Session afterwards -- so the installer is
        // re-run here by hand rather than by weakening the seam.
        registry.install_settings(session.settings());
        session.settings().set_baseline(session.settings().snapshot());
        session.set_output(out);
    }
};

struct Outcome {
    int code = 0;
    std::string text;
};

Outcome Run(Fixture& fx, const std::string& src) {
    Outcome o;
    o.code = RunSource(fx.session, src, "<test>");
    o.text = fx.out.str();
    fx.out.str(std::string());
    fx.out.clear();
    return o;
}

// --- 1.1 every binding is registered ---------------------------------------

void TestRegistrationTable() {
    Fixture fx;

    for (const std::string& t : PipelineTypeNames()) {
        Check(fx.registry.find_type(t) != nullptr, "type '" + t + "' is registered");
    }
    for (const std::string& v : PipelineVerbNames()) {
        Check(fx.registry.find_verb(v) != nullptr, "verb '" + v + "' is registered");
    }
    // The core verbs survive the pipeline registration.
    for (const std::string& v : CoreVerbNames()) {
        Check(fx.registry.find_verb(v) != nullptr, "core verb '" + v + "' still resolves");
    }

    // C1a fills TypeInfo::hash for exactly the value types -- that is what
    // retires C0's "no type fills TypeInfo::hash" assertion for the C1 registry
    // (plan Sec. 10, Phase C1a).
    for (const char* n : {bind::kTypeMatrix, bind::kTypePoly, bind::kTypeFp}) {
        const TypeInfo* t = fx.registry.find_type(n);
        Check(t != nullptr && static_cast<bool>(t->hash),
              std::string("type ") + n + " fills TypeInfo::hash");
    }

    // (C1b) The decomposition surface is HERE now.  C1a's guard list is empty
    // (RegisterPipelineVerbs.cpp says why); the guard that replaces it is the
    // NEXT phase boundary, so "which phase are we in" stays checkable.
    Check(VerbsForbiddenBeforeC1b().empty(),
          "guard: nothing of the decomposition surface is withheld any more");
    for (const char* v : {"decompose", "report", "crosschecks"}) {
        Check(fx.registry.find_verb(v) != nullptr,
              std::string("C1b: verb '") + v + "' is registered");
    }
    for (const char* t : {"Decomposition", "Block", "Report"}) {
        Check(fx.registry.find_type(t) != nullptr,
              std::string("C1b: type '") + t + "' is registered");
    }
    Check(fx.registry.find_type("Sturm") == nullptr,
          "guard: the extension types are NOT on a bare pipeline registry");
    for (const std::string& v : VerbsForbiddenBeforeC2()) {
        Check(fx.registry.find_verb(v) == nullptr,
              "guard: verb '" + v + "' is phase C2/C5's, not C1b's");
    }
    // C1b fills TypeInfo::hash for the Decomposition too -- it is the golden
    // `decomp_*_invariants` recipe, which is what a transcript pins.
    {
        const TypeInfo* t = fx.registry.find_type("Decomposition");
        Check(t != nullptr && static_cast<bool>(t->hash),
              "C1b: Decomposition fills TypeInfo::hash (the _invariants recipe)");
    }
    // The extension layer registers its OWN verbs, from HeckeExt/ ([OD]
    // slot-3 close-out 1).  On this fixture's registry that is an explicit,
    // separate call -- exactly as main.cpp's console branch makes it.
    hecke_ext::RegisterHeckeExtensionVerbs(fx.registry);
    for (const char* v : {"sturm_bound", "u_ell_decision"}) {
        Check(fx.registry.find_verb(v) != nullptr,
              std::string("C1b: extension verb '") + v + "' is registered from HeckeExt/");
    }
    // (slot-5 WP-10) The two E6 prediction units, C1b's one recorded deferral
    // ([WP4-5] Sec. 5 item 1): both units shipped in slot-4 WP-7, so the verbs
    // are bound from the SAME TU and the guards' two exact-path entries still
    // cover the whole extension console surface.
    for (const char* v : {"bernoulli_mod", "glover_factors"}) {
        Check(fx.registry.find_verb(v) != nullptr,
              std::string("WP-10: extension verb '") + v + "' is registered from HeckeExt/");
    }
    Check(fx.registry.find_type("Sturm") != nullptr, "C1b: the Sturm type is registered");
    {
        // Idempotent per registry, like RegisterPipelineVerbs().
        hecke_ext::RegisterHeckeExtensionVerbs(fx.registry);
        const VerbEntry* sb = fx.registry.find_verb("sturm_bound");
        Check(sb != nullptr && sb->overloads.size() == 2,
              "C1b: a second extension registration is a no-op (2 overloads)");
    }

    // Registration is idempotent on the global registry, so an entry point
    // reached twice cannot double-register an overload.
    RegisterPipelineVerbs();
    RegisterPipelineVerbs();
    const VerbEntry* h = Registry::global().find_verb("hash");
    Check(h != nullptr && h->overloads.size() == 1,
          "global registration is idempotent (hash has one overload)");
}

// --- 1.2 GoldenLookup and the hash contract --------------------------------

void TestGoldenLookup() {
    Fixture fx;

    Check(GoldenReferenceSize() > 14000,
          "GoldenLookup sees the compiled-in reference (" +
              std::to_string(GoldenReferenceSize()) + " entries)");
    const GoldenEntry* e = GoldenLookup("n=10 ell=691", "dim_H^1");
    Check(e != nullptr, "GoldenLookup finds a known scalar entry");
    if (e != nullptr) {
        CheckEqInt(static_cast<long long>(e->rows), 3, "dim H^1 at (10, 691) is 3");
        CheckEqInt(e->tier, 'A', "it is a tier-A entry");
        CheckEqInt(static_cast<long long>(e->hash),
                   static_cast<long long>(engine::FnvU64(engine::kFnvOffset64, 3ull)),
                   "the frozen hash IS FnvU64(offset, value) -- the one FNV-1a-64");
    }
    Check(GoldenLookup("n=10 ell=691", "no_such_object") == nullptr,
          "GoldenLookup returns nullptr for an unknown object");

    // The verb reads the same bytes.
    Outcome o = Run(fx, "g = golden(\"n=10 ell=691\", \"dim_H^1\"); print(g.value)\n");
    CheckEqInt(o.code, 0, "golden(...) runs");
    Check(Contains(o.text, "3"), "golden(...).value is the frozen dimension");

    o = Run(fx, "golden(\"n=10 ell=691\", \"no_such_object\")\n");
    CheckEqInt(o.code, 3, "an unknown golden object is a runtime error");
    Check(Contains(o.text, "not in GoldenReference.inc"), "and it says so");

    // Rule R19 (plan Sec. 6): a pin may only be taken at the DEFAULT
    // mathematical configuration, so `golden(...)` refuses otherwise.
    o = Run(fx, "print(fingerprint())\n");
    Check(Contains(o.text, "0x0000000000000000"), "the fingerprint is 0 at the defaults");
    {
        hecke::ScopedLocality scoped(hecke::LocalityMode::Budget, 8);
        Check(hecke::math_settings().fingerprint() != 0,
              "a non-default locality moves the fingerprint");
        Outcome bad = Run(fx, "golden(\"n=10 ell=691\", \"dim_H^1\")\n");
        CheckEqInt(bad.code, 3, "golden(...) refuses at a non-zero fingerprint");
        Check(Contains(bad.text, "not at their defaults"), "and names the reason");
    }
    Check(hecke::math_settings().fingerprint() == 0, "the scope restored the defaults");

    // hash() is the harness's per-kind definition.
    o = Run(fx, "A = matrix(7, [[1, 2], [3, 4]]); print(hash(A))\n");
    CheckEqInt(o.code, 0, "hash(Matrix) runs");
    {
        FlatMatrixT<uint32_t> m(2, 2, 7);
        m.set_entry(0, 0, 1); m.set_entry(0, 1, 2);
        m.set_entry(1, 0, 3); m.set_entry(1, 1, 4);
        char buf[32];
        std::snprintf(buf, sizeof(buf), "0x%016llx",
                      static_cast<unsigned long long>(bind::HashMatrix(m)));
        Check(Contains(o.text, buf), "hash(Matrix) is rows, cols, then the entries, over Fnv.h");
    }
    o = Run(fx, "print(hash(7))\n");
    char sbuf[32];
    std::snprintf(sbuf, sizeof(sbuf), "0x%016llx",
                  static_cast<unsigned long long>(bind::HashScalarValue(7)));
    Check(Contains(o.text, sbuf), "hash(Int) is the scalar-invariant hash");
    o = Run(fx, "hash(true)\n");
    CheckEqInt(o.code, 3, "hash() refuses a kind the harness does not define one for");
}

// --- 1.3 the exit-code table, 0..5 -----------------------------------------

// A verb that throws exactly what plan Sec. 2 C7's own row names.  It is
// registered on the TEST registry only, so the arm is independent of whether
// any particular mathematical setting is implemented in this build -- which is
// the point: slot-3 WP-3 is landing `locality = strict` concurrently, and this
// check must not depend on that.
Value VerbThrowNotImplemented(CallContext&, const std::vector<Value>&) {
    throw hecke::NotImplementedInThisBuild("locality = strict");
}
Value VerbThrowOverflow(CallContext&, const std::vector<Value>&) {
    throw hecke::HeckeOverflow("hecke: exact multiply overflow");
}

void TestExitCodes() {
    Fixture fx;
    fx.registry.register_verb("__not_implemented", bind::Sig({}), &VerbThrowNotImplemented,
                              "test-only: throws hecke::NotImplementedInThisBuild", "test");
    fx.registry.register_verb("__overflow", bind::Sig({}), &VerbThrowOverflow,
                              "test-only: throws hecke::HeckeOverflow", "test");

    CheckEqInt(Run(fx, "print(1 + 1)\n").code, 0, "exit 0: success");
    CheckEqInt(Run(fx, "a = = 1\n").code, 2, "exit 2: parse error");
    CheckEqInt(Run(fx, "rank(3)\n").code, 3, "exit 3: a type error at run time");
    CheckEqInt(Run(fx, "assert(false, \"nope\")\n").code, 4, "exit 4: a failed assert");
    CheckEqInt(Run(fx, "exit(7)\n").code, 7, "exit(n) sets the code explicitly");

    // Exit code 5 -- the plan's own outcome for a legal setting this build
    // cannot execute.  It is NEVER 1 (a bad value) and NEVER 3 (a failed
    // computation).
    Outcome ni = Run(fx, "__not_implemented()\n");
    CheckEqInt(ni.code, 5, "exit 5: hecke::NotImplementedInThisBuild");
    Check(Contains(ni.text, "not implemented in this build"), "the message names the outcome");
    Check(Contains(ni.text, "hecke::NotImplementedInThisBuild"),
          "the C++ class is named through the translator");
    Check(Contains(ni.text, "locality = strict"), "and the detail survives");

    // The catch ORDER: HeckeOverflow derives from std::runtime_error, so it must
    // be tested first or it would never be named -- and it is an ordinary
    // runtime failure, exit 3.
    Outcome ov = Run(fx, "__overflow()\n");
    CheckEqInt(ov.code, 3, "HeckeOverflow is an ordinary runtime error (exit 3)");
    Check(Contains(ov.text, "hecke::HeckeOverflow"), "and is named as its own class");

    // The same outcome through a LAZY HANDLE: the kind must survive the
    // handle wrapper, which is a different code path (Object::ensure_computed).
    Fixture fx2;
    TypeInfo t;
    t.name = "NotImplStub";
    t.members.push_back({"value", true, "forces compute()", [](const ObjectPtr&) {
                             return Value::from_int(0);
                         }});
    const TypeInfo* stub = fx2.registry.register_type(std::move(t));
    fx2.registry.register_verb(
        "__stub", bind::Sig({}),
        [stub](CallContext& ctx, const std::vector<Value>&) {
            ObjectPtr o = Object::create(stub, ctx.pos, ctx.call_text,
                                         ctx.session->settings().snapshot(),
                                         std::shared_ptr<void>(), [](Object&) {
                                             throw NotImplementedError(
                                                 "D.blocks: locality = strict", SourcePos());
                                         });
            return Value::from_object(o);
        },
        "test-only", "test");
    Outcome lazy = Run(fx2, "x = __stub()\nprint(x.value)\n");
    CheckEqInt(lazy.code, 5, "exit 5 survives the lazy-handle wrapper");
    Check(Contains(lazy.text, "created at"), "and the error still names the creation site");
}

// --- 1.4 the R3 verbs: state / resume / save_state -------------------------

void TestResumableVerbs() {
    Fixture fx;

    // A budgeted stub: incomplete at budget 8, complete at 64 -- the shape of
    // HeckeSturmBound and of the commutant state, the two shipped R3 instances.
    struct StubPayload {
        std::int64_t budget = 0;
    };
    TypeInfo t;
    t.name = "Budgeted";
    t.members.push_back({"reached", false, "the budget this handle reached",
                         [](const ObjectPtr& o) {
                             return Value::from_int(
                                 static_cast<StubPayload*>(o->payload().get())->budget);
                         }});
    // `TypeInfo::resume` is deliberately left UNSET: a type with no continuation
    // must make `resume(x, ...)` a typed error naming the type, which is the
    // last arm below.  A type that CAN continue is HeckeSturmBound's job, and
    // binding it needs the HeckeExt/ include guard widened -- see the WP-7
    // report, deviation 6.
    const TypeInfo* stub_type = fx.registry.register_type(std::move(t));
    auto make = [](const TypeInfo* type, CallContext& ctx, std::int64_t budget) {
        auto p = std::make_shared<StubPayload>();
        p->budget = budget;
        ObjectPtr o = Object::create(type, ctx.pos, ctx.call_text,
                                     ctx.session->settings().snapshot(), p, [](Object&) {});
        ResumableState rs;
        rs.resumable = (budget < 64);
        rs.budget_setting = "sturm.cap";
        rs.budget_value = Value::from_int(budget);
        rs.reason = rs.resumable ? "closure not reached within the budget" : "";
        Record progress;
        progress.set("kind", Value::from_string("sturm_bound"));
        progress.set("reached", Value::from_int(budget));
        std::vector<Value> hist;
        hist.push_back(Value::from_int(budget));
        progress.set("budget_history", Value::from_list(std::move(hist)));
        rs.progress = progress;
        o->set_resumable_state(rs);
        return o;
    };
    fx.registry.register_verb(
        "__budgeted", bind::Sig({Param::of("budget", ValueKind::Int)}),
        [stub_type, make](CallContext& ctx, const std::vector<Value>& a) {
            return Value::from_object(make(stub_type, ctx, a[0].int_value()));
        },
        "test-only", "test");

    Outcome o = Run(fx, "x = __budgeted(8)\ns = state(x)\nprint(s.resumable)\n");
    CheckEqInt(o.code, 0, "state(x) runs on a budgeted handle");
    Check(Contains(o.text, "true"), "and reports the handle as resumable");

    // save_state with no stem: the object is inlined and returned as JSON text.
    o = Run(fx, "print(save_state(__budgeted(8)))\n");
    CheckEqInt(o.code, 0, "save_state(x) runs");
    Check(Contains(o.text, "\"kind\": \"sturm_bound\""), "save_state names the kind");
    Check(Contains(o.text, "\"complete\": false"), "save_state records completeness");
    Check(Contains(o.text, "\"budget\": 8"), "save_state records the budget");
    Check(Contains(o.text, "\"budget_history\""), "save_state records the budget history");
    Check(Contains(o.text, "\"state\""), "save_state carries the iteration state");

    // save_state to a path.
    const std::string path = TmpDir() + "/oc_console_c1a_state.json";
    ::unlink(path.c_str());
    o = Run(fx, "r = save_state(__budgeted(8), \"" + path + "\")\nprint(r.inline_state)\n");
    CheckEqInt(o.code, 0, "save_state(x, path) runs");
    std::string written;
    Check(ReadFile(path, written), "save_state(x, path) wrote the file");
    Check(Contains(written, "\"sturm_bound\""), "the written object carries the kind");
    Check(Contains(o.text, "true"), "a small state is inlined, not side-carred");
    ::unlink(path.c_str());

    // A handle with NO resumable state is a typed error, not a silent empty
    // object -- a computation that is not budgeted has nothing to continue.
    o = Run(fx, "save_state(matrix(7, [[1]]))\n");
    CheckEqInt(o.code, 3, "save_state on a non-resumable handle is a typed error");
    Check(Contains(o.text, "carries no resumable state"), "and says why");

    // `state` and `resume` are C0's and still work through the C1 registry.
    o = Run(fx, "resume(__budgeted(8), 64)\n");
    CheckEqInt(o.code, 3, "resume() on a type with no continuation is a typed error");
    Check(Contains(o.text, "resume"), "and names the verb");
}

// --- 1.5 the smoke script set ----------------------------------------------

void CollectScripts(const std::string& dir, std::vector<std::string>& out) {
    DIR* d = ::opendir(dir.c_str());
    if (d == nullptr) return;
    while (struct dirent* e = ::readdir(d)) {
        const std::string name = e->d_name;
        if (name.size() < 4 || name.substr(name.size() - 3) != ".oc") continue;
        // Only the numbered scripts run in process: `prelude.oc` is included by
        // them, and `cli_*.oc` needs command-line parameters, which only the CLI
        // arm supplies.
        if (name[0] < '0' || name[0] > '9') continue;
        out.push_back(dir + "/" + name);
    }
    ::closedir(d);
    std::sort(out.begin(), out.end());
}

// --- 1.5b the `.expected` transcripts (C1b) ---------------------------------
//
// Plan Sec. 10, Phase C1b acceptance: "every `.expected` carries
// `#! oc-format 1`; CONSOLE_EMIT_EXPECTED=1 ./cohomology_app ConsoleScriptEmit
// regenerates them and is a no-op otherwise".
//
// WHAT IS COMPARED.  A script's transcript is its session output with every
// `#~` ANNOTATION line removed (plan Sec. 2 C5: annotations carry timings and
// are excluded from every comparison), and with trailing whitespace stripped.
// The `.expected` file is that text under one header line.
//
// WHY THE FILES ARE SAFE TO FREEZE (audit finding F4).  No script in the set
// prints a list a sibling package extends -- not the verb list, not `help`,
// not the settings registry, not a validation roster.  Each prints only values
// it has itself pinned through `golden(...)` or values that are theorems about
// the object (the preset dimensions).  A later package that appends a verb, a
// settings key or a validation name therefore cannot move one of these files.

const char* const kExpectedHeader = "#! oc-format 1";

std::string TranscriptOf(const std::string& raw) {
    std::string out;
    for (const std::string& line : SplitLines(raw)) {
        if (line.rfind("#~", 0) == 0) continue;   // annotation: never compared
        std::size_t end = line.size();
        while (end > 0 && (line[end - 1] == ' ' || line[end - 1] == '\r' ||
                           line[end - 1] == '\t')) {
            --end;
        }
        out += line.substr(0, end);
        out += '\n';
    }
    return out;
}

std::string ExpectedBody(const std::string& file_text) {
    // Everything after the first line, which is the header.
    const std::size_t nl = file_text.find('\n');
    if (nl == std::string::npos) return std::string();
    return file_text.substr(nl + 1);
}

std::string ExpectedPathFor(const std::string& script_path) {
    return script_path.substr(0, script_path.size() - 3) + ".expected";
}

void CompareExpected(const std::string& script_path, const std::string& transcript) {
    const std::string expected_path = ExpectedPathFor(script_path);
    const std::string name = expected_path.substr(expected_path.find_last_of('/') + 1);
    std::string file_text;
    if (!ReadFile(expected_path, file_text)) {
        Check(false, "the frozen transcript " + name + " is present");
        return;
    }
    const std::vector<std::string> lines = SplitLines(file_text);
    Check(!lines.empty() && lines[0] == kExpectedHeader,
          name + " carries the `" + kExpectedHeader + "` header");
    const std::string want = ExpectedBody(file_text);
    const bool same = (want == transcript);
    Check(same, "the transcript of " + name + " is byte-identical to the frozen one");
    if (!same) {
        std::cout << "    --- expected ---" << std::endl << want;
        std::cout << "    --- got ---" << std::endl << transcript;
    }
}

void TestScripts(const std::string& root) {
    if (root.empty()) {
        std::cout << "  [SKIP] script arm: could not locate the source tree" << std::endl;
        return;
    }
    const std::string dir = root + "/src/hacs/Console/scripts";
    std::vector<std::string> scripts;
    CollectScripts(dir, scripts);
    Check(!scripts.empty(), "the smoke script set is present");
    for (const std::string& path : scripts) {
        // (C1b) A script may LEGALLY change a mathematical setting -- 015 runs
        // the whole strict-locality arm -- so the settings in force are
        // restored around every script and the "left at the defaults"
        // assertion below is about the SCRIPT, not about the harness.
        const hecke::ScopedMathSettings restore(hecke::math_settings());
        Fixture fx;
        // (C1b) The extension verbs, registered from HeckeExt/ exactly as
        // main.cpp's console branch registers them, so a script can call
        // `sturm_bound` / `u_ell_decision`.
        hecke_ext::RegisterHeckeExtensionVerbs(fx.registry);
        // The pipeline's own chatter must not reach a transcript (plan Sec. 2
        // C5); `verbose` is off by default in script mode, so this is also a
        // test of that.
        const int code = RunFile(fx.session, path);
        const std::string text = fx.out.str();
        const std::string name = path.substr(path.find_last_of('/') + 1);
        CheckEqInt(code, 0, "script " + name + " exits 0");
        if (code != 0) {
            std::cout << "    --- " << name << " output ---" << std::endl;
            std::cout << text << std::endl;
            continue;
        }
        for (const std::string& line : SplitLines(text)) {
            if (line.rfind("#~", 0) == 0) std::cout << "  " << line << std::endl;
        }
        // Every pin in the set is taken at the default configuration (R19).
        Check(hecke::math_settings().fingerprint() == 0,
              "script " + name + " left the mathematical settings at their defaults");
        CompareExpected(path, TranscriptOf(text));
    }
}

// --- 1.5c ConsoleScriptEmit (C1b) -------------------------------------------
//
// Owner-gated exactly as GoldenEmit is: a no-op unless CONSOLE_EMIT_EXPECTED=1
// is set, so `--all` stays read-only.  The FIRST emission is part of the C1b
// package (a new artefact: nothing is re-baselined); any later re-emission is
// owner-gated, and its diff must be explained file by file.
uint32_t EmitExpected(const std::string& root) {
    const char* gate = std::getenv("CONSOLE_EMIT_EXPECTED");
    if (gate == nullptr || std::string(gate) != "1") {
        std::cout << "  [SKIP] ConsoleScriptEmit is a no-op unless CONSOLE_EMIT_EXPECTED=1"
                  << std::endl;
        return 0;
    }
    if (root.empty()) {
        std::cout << "  [FAIL] could not locate the source tree" << std::endl;
        return 1;
    }
    const std::string dir = root + "/src/hacs/Console/scripts";
    std::vector<std::string> scripts;
    CollectScripts(dir, scripts);
    if (scripts.empty()) {
        std::cout << "  [FAIL] no scripts found under " << dir << std::endl;
        return 1;
    }
    uint32_t failures = 0;
    for (const std::string& path : scripts) {
        const hecke::ScopedMathSettings restore(hecke::math_settings());
        Fixture fx;
        hecke_ext::RegisterHeckeExtensionVerbs(fx.registry);
        const int code = RunFile(fx.session, path);
        const std::string name = path.substr(path.find_last_of('/') + 1);
        if (code != 0) {
            std::cout << "  [FAIL] " << name << " exits " << code
                      << " -- refusing to freeze a failing script" << std::endl;
            std::cout << fx.out.str() << std::endl;
            ++failures;
            continue;
        }
        const std::string out_path = ExpectedPathFor(path);
        std::ofstream os(out_path.c_str(), std::ios::binary | std::ios::trunc);
        if (!os) {
            std::cout << "  [FAIL] cannot write " << out_path << std::endl;
            ++failures;
            continue;
        }
        os << kExpectedHeader << "\n" << TranscriptOf(fx.out.str());
        os.close();
        std::cout << "  wrote " << out_path.substr(out_path.find_last_of('/') + 1) << std::endl;
    }
    return failures;
}

// --- 1.6 the narrowed transcript guard and the include rule ----------------

void CollectFiles(const std::string& dir, std::vector<std::string>& out, int depth) {
    if (depth > 6) return;
    DIR* d = ::opendir(dir.c_str());
    if (d == nullptr) return;
    while (struct dirent* e = ::readdir(d)) {
        const std::string name = e->d_name;
        if (name == "." || name == "..") continue;
        const std::string full = dir + "/" + name;
        struct stat st;
        if (::stat(full.c_str(), &st) != 0) continue;
        if (S_ISDIR(st.st_mode)) CollectFiles(full, out, depth + 1);
        else out.push_back(full);
    }
    ::closedir(d);
}

void TestTreeGuards(const std::string& root) {
    if (root.empty()) {
        std::cout << "  [SKIP] tree arm: could not locate the source tree" << std::endl;
        return;
    }
    // (C1b) The guard is NARROWED, not dropped (plan Sec. 10, Phase C1b
    // item 5; [WP3-7] Sec. 6.4): C1b ships `.expected` files, so the statement
    // that survives is the one E0b actually protects -- no frozen transcript
    // may name a decomposition object of the PRE-re-baseline schema.  A
    // `hecke_report/v1` text in a frozen file is exactly that, and is what is
    // forbidden now.
    std::vector<std::string> files;
    CollectFiles(root + "/src/hacs/Console", files, 0);
    Check(!files.empty(), "the Console directory was found");
    std::uint32_t frozen = 0;
    std::uint32_t stale = 0;
    for (const std::string& f : files) {
        const bool is_expected = f.size() > 9 && f.substr(f.size() - 9) == ".expected";
        const bool is_transcript = f.size() > 11 && f.substr(f.size() - 11) == ".transcript";
        if (!is_expected && !is_transcript) continue;
        ++frozen;
        std::string text;
        if (!ReadFile(f, text)) continue;
        if (Contains(text, "hecke_report/v1") ||
            Contains(text, "hecke_module_decomposition/v1")) {
            ++stale;
            std::cout << "  [FAIL] frozen transcript names a PRE-E0b schema: " << f << std::endl;
        }
        if (text.rfind(kExpectedHeader, 0) != 0) {
            ++stale;
            std::cout << "  [FAIL] frozen transcript lacks the `" << kExpectedHeader
                      << "` header: " << f << std::endl;
        }
    }
    CheckEqInt(stale, 0,
               "guard: no frozen console transcript names a pre-E0b decomposition schema");
    Check(frozen > 0, "C1b ships frozen transcripts (" + std::to_string(frozen) + " files)");

    // Plan Sec. 2 C8: "C1a re-runs the same arm from ConsoleValidation.cpp so
    // the guard survives if C0's validation is ever narrowed."  Probed, not
    // required, exactly as ConsumerMigrationValidation probes
    // check_matrix_includes.sh.
    const std::string script = root + "/docs/matrix-engine/tools/check_console_includes.sh";
    if (!PathExists(script) || std::system(nullptr) == 0) {
        std::cout << "  [SKIP] check_console_includes.sh is not runnable in this tree"
                  << std::endl;
        return;
    }
    const std::string cmd = "cd " + Shq(root) + " && " + Shq(script);
    const int status = std::system(cmd.c_str());
    const bool ran = (status != -1) && WIFEXITED(status);
    Check(ran, "guard: check_console_includes.sh ran");
    if (ran) {
        CheckEqInt(WEXITSTATUS(status), 0,
                   "guard: ConsoleCore is still STL-only with the C1a bindings in the tree");
    }
}

// ===========================================================================
// Part 2 -- ConsoleCliPreservationValidation
// ===========================================================================

struct FrozenCase {
    const char* file;      // under Console/scripts/cli_frozen/
    const char* args;      // appended to the binary
    int exit_code;
    bool modulo_appended;  // --list only (plan Sec. 11)
};

void CompareFrozen(const std::string& exe, const std::string& frozen_dir, const FrozenCase& c,
                   const std::string& root) {
    std::string want;
    const std::string want_path = frozen_dir + "/" + c.file;
    if (!ReadFile(want_path, want)) {
        Check(false, std::string("frozen file ") + c.file + " is present");
        return;
    }
    std::string args = c.args;
    // The campaign case names a spec file inside the tree.
    const std::string marker = "%SPEC%";
    const std::size_t at = args.find(marker);
    if (at != std::string::npos) {
        args = args.substr(0, at) + Shq(frozen_dir + "/campaign_smoke.spec") +
               args.substr(at + marker.size());
    }
    (void)root;
    std::string got;
    const int code = RunAsFrozen(exe, args, got);
    CheckEqInt(code, c.exit_code, std::string("`") + c.args + "` exits " +
                                      std::to_string(c.exit_code));
#ifdef ENGINE_SHADOW_LEGACY
    // (S24 CL-L1) A shadow build also runs the legacy cohomology arm
    // (legacy/EquivariantCohomologyOnKShadow.inc), which prints its
    // `[TIMING] A_Cq kernel` / `[TIMING] A_Zq kernel` lines on stdout; they are
    // the shadow's own output, not the frozen CLI's, so they are dropped here.
    {
        std::string kept;
        for (const std::string& line : SplitLines(got))
            if (line.rfind("[TIMING] A_Cq kernel q=", 0) != 0 && line.rfind("[TIMING] A_Zq kernel q=", 0) != 0)
                kept += line + "\n";
        if (!got.empty() && got.back() != '\n' && !kept.empty()) kept.pop_back();
        got = kept;
    }
#endif
    if (!c.modulo_appended) {
        const bool same = (got == want);
        Check(same, std::string("`") + c.args + "` stdout is byte-identical to the frozen file");
        if (!same) {
            const std::vector<std::string> a = SplitLines(want);
            const std::vector<std::string> b = SplitLines(got);
            for (std::size_t i = 0; i < a.size() || i < b.size(); ++i) {
                const std::string wa = i < a.size() ? a[i] : std::string("<missing>");
                const std::string wb = i < b.size() ? b[i] : std::string("<missing>");
                if (wa != wb) {
                    std::cout << "    line " << (i + 1) << " frozen: " << wa << std::endl;
                    std::cout << "    line " << (i + 1) << " live  : " << wb << std::endl;
                }
            }
        }
        return;
    }
    // --list: every frozen line must appear in the live listing, IN THE SAME
    // RELATIVE ORDER and with the same tier; extra lines are allowed and are
    // reported, so an unexpected growth is visible (plan Sec. 11).
    const std::vector<std::string> frozen_lines = SplitLines(want);
    const std::vector<std::string> live_lines = SplitLines(got);
    std::size_t j = 0;
    std::uint32_t missing = 0;
    for (const std::string& line : frozen_lines) {
        bool found = false;
        while (j < live_lines.size()) {
            if (live_lines[j] == line) { found = true; ++j; break; }
            ++j;
        }
        if (!found) {
            ++missing;
            std::cout << "    [FAIL] frozen --list line lost or reordered: " << line << std::endl;
        }
    }
    CheckEqInt(missing, 0, "--list: every frozen name is still there, in order, with its tier");
    const long long added =
        static_cast<long long>(live_lines.size()) - static_cast<long long>(frozen_lines.size());
    std::cout << "  --list: " << frozen_lines.size() << " frozen names, " << live_lines.size()
              << " live (" << (added >= 0 ? "+" : "") << added << ")" << std::endl;
    Check(added >= 0, "--list only ever grows");
}

void TestFrozenCli(const std::string& exe, const std::string& root) {
    const std::string frozen_dir =
        root + "/src/hacs/Console/scripts/cli_frozen";
    if (!PathExists(frozen_dir)) {
        std::cout << "  [SKIP] the frozen CLI outputs are not in this tree" << std::endl;
        return;
    }
    static const FrozenCase cases[] = {
        {"help.stdout", "--help", 0, false},
        // (S24 CL-7, owner ruling 4) no argument: the same usage text, exit 1.
        {"help.stdout", "", 1, false},
        {"bench_help.stdout", "--bench --help", 0, false},
        {"bench_micro_help.stdout", "--bench micro --help", 0, false},
        {"campaign_help.stdout", "--campaign --help", 0, false},
        {"campaign_dryrun.stdout", "--campaign %SPEC% --dry-run", 0, false},
        {"validation_HeckeIdentityValidation.stdout", "HeckeIdentityValidation", 0, false},
        {"list.stdout", "--list", 0, true},
    };
    for (const FrozenCase& c : cases) CompareFrozen(exe, frozen_dir, c, root);
}

void TestConsoleCli(const std::string& exe, const std::string& root) {
    std::string out;

    // (d) of the plan's C1a acceptance.
    CheckEqInt(RunCapturing(Shq(exe) + " -c 'print(1+1)'", out), 0, "-c exits 0");
    Check(out == "2\n", "-c 'print(1+1)' prints 2");

    CheckEqInt(RunCapturing(Shq(exe) + " -c 'a = 1' -c 'b = 2'", out), 1,
               "two -c forms is a usage error (exit 1)");

    CheckEqInt(RunCapturing(Shq(exe) + " --console --help", out), 0, "--console --help exits 0");
    Check(Contains(out, "OC script console"), "and prints the console usage");
    Check(Contains(out, "5 a legal setting this build cannot execute"),
          "whose exit-code table includes 5");

    // A parse error on line 7 must be reported BEFORE line 1 runs (plan (d)).
    {
        const std::string script = TmpDir() + "/oc_console_c1a_parse.oc";
        std::ofstream f(script.c_str());
        f << "print(\"line 1 must not run\")\n";
        for (int i = 2; i <= 6; ++i) f << "x" << i << " = " << i << "\n";
        f << "y = = 7\n";
        f.close();
        const int code = RunCapturing(Shq(exe) + " --script " + Shq(script), out);
        CheckEqInt(code, 2, "a parse error is exit 2");
        Check(!Contains(out, "line 1 must not run"),
              "the whole file is parsed before anything runs");
        Check(Contains(out, ":7:"), "and the error names line 7");
        ::unlink(script.c_str());
    }

    // stdin.
    CheckEqInt(RunCapturing("echo 'print(6*7)' | " + Shq(exe) + " -", out), 0, "stdin exits 0");
    Check(out == "42\n", "and runs the piped script");

    // A missing file is exit 1.
    CheckEqInt(RunCapturing(Shq(exe) + " --script /tmp/oc_no_such_script.oc", out), 1,
               "a missing script file is exit 1");

    // The modifiers follow the FORM keyword, because main.cpp only enters the
    // console when argv[1] is one of --console / -i / --script / -c / - (so a
    // stray `--set` in front is still an unknown validation name, unchanged).
    CheckEqInt(RunCapturing(Shq(exe) + " -c 'print(1)' --set no_such_key=1", out), 1,
               "an undeclared --set key is exit 1");
    Check(Contains(out, "no_such_key"), "and names the key");
    // A declared one is applied before the first statement.
    CheckEqInt(RunCapturing(Shq(exe) + " -c 'print(1)' --set print_max_entries=2", out), 0,
               "a declared --set key is accepted");
    // `--set ... -c ...` in front is four unknown validation NAMES, i.e. the
    // pre-console behaviour, unchanged: exit code = the failure count.
    CheckEqInt(RunCapturing(Shq(exe) + " --set print_max_entries=2 -c 'print(1)'", out), 4,
               "a --set BEFORE the form keyword is not a console invocation at all");

    // Parameters: `--script f.oc n=10` binds args.n, and a missing one is
    // a runtime error (exit 3).
    if (!root.empty()) {
        const std::string script =
            root + "/src/hacs/Console/scripts/cli_params.oc";
        if (PathExists(script)) {
            const int code =
                RunCapturing(Shq(exe) + " --script " + Shq(script) + " n=10 ell=691", out);
            CheckEqInt(code, 0, "--script f.oc n=10 ell=691 exits 0");
            Check(Contains(out, "dims [0, 3, 0]"), "and the golden dimensions came back");
            CheckEqInt(RunCapturing(Shq(exe) + " --script " + Shq(script) + " n=10", out), 3,
                       "a missing args.name is a runtime error (exit 3)");
        }
    }

    // assert / expect_error.
    CheckEqInt(RunCapturing(Shq(exe) + " -c 'assert(false, \"boom\")'", out), 4,
               "a failed assert is exit 4");
    Check(Contains(out, "boom"), "and prints the message");
    CheckEqInt(RunCapturing(Shq(exe) + " -c 'expect_error \"unknown verb\" { nope() }'", out), 0,
               "expect_error consumes the error it names");

    // The console branch did not eat a validation name: `-` and `-c` change
    // meaning (the plan's one stated behavioural change), everything else does
    // not.
    CheckEqInt(RunCapturing(Shq(exe) + " NoSuchValidationName", out), 1,
               "an unknown validation name is still exit 1");
    Check(Contains(out, "Unknown validation"), "with the pre-existing message");
}

} // namespace

// ===========================================================================

uint32_t ConsoleScriptValidation() {
    std::cout << "=== ConsoleScriptValidation (console plan v3 Sec. 10, Phase C1a; slot-3 WP-7) "
                 "==="
              << std::endl;
    g_checks = 0;
    g_failures = 0;

    const std::string root = FindRepoRoot();
    TestRegistrationTable();
    TestGoldenLookup();
    TestExitCodes();
    TestResumableVerbs();
    TestScripts(root);
    TestTreeGuards(root);

    std::cout << "ConsoleScriptValidation: " << g_checks << " checks, " << g_failures
              << " failure(s)." << std::endl;
    std::cout << "ConsoleScriptValidation: " << (g_failures == 0 ? "PASSED" : "FAILED")
              << std::endl;
    return g_failures;
}

uint32_t ConsoleScriptEmit() {
    std::cout << "=== ConsoleScriptEmit (console plan v3.1 Sec. 10, Phase C1b; slot-4 WP-5) ==="
              << std::endl;
    const uint32_t failures = EmitExpected(FindRepoRoot());
    std::cout << "ConsoleScriptEmit: " << (failures == 0 ? "PASSED" : "FAILED") << std::endl;
    return failures;
}

uint32_t ConsoleCliPreservationValidation() {
    std::cout << "=== ConsoleCliPreservationValidation (console plan v3 Sec. 10 C1a (c)/(d), "
                 "Sec. 11) ==="
              << std::endl;
    g_checks = 0;
    g_failures = 0;

    const std::string exe = SelfPath();
    const std::string root = FindRepoRoot();
    if (exe.empty() || std::system(nullptr) == 0) {
        std::cout << "  [SKIP] no command processor, or the executable could not locate itself"
                  << std::endl;
        std::cout << "ConsoleCliPreservationValidation: PASSED (skipped)" << std::endl;
        return 0;
    }
    if (root.empty()) {
        std::cout << "  [SKIP] could not locate the source tree" << std::endl;
        std::cout << "ConsoleCliPreservationValidation: PASSED (skipped)" << std::endl;
        return 0;
    }

    TestFrozenCli(exe, root);
    TestConsoleCli(exe, root);

    std::cout << "ConsoleCliPreservationValidation: " << g_checks << " checks, " << g_failures
              << " failure(s)." << std::endl;
    std::cout << "ConsoleCliPreservationValidation: " << (g_failures == 0 ? "PASSED" : "FAILED")
              << std::endl;
    return g_failures;
}
