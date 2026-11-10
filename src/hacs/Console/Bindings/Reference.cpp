// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Reference.cpp -- console phase C4 (hecke-extensions slot 8, WP-10): the
// generated reference manual, docs/console/console.md.
//
// [P] Sec. 3.6 / Sec. 4.4: "the language reference in docs/console/console.md
// is generated from the registry (`help --markdown`), not written by hand", and
// it "must state in words" two things (Sec. 4.4): that `-2 ^ 2 == 4`, and that
// print/str render a top-level string bare while every other position quotes it.
//
// SPELLING.  [P] names the entry `help --markdown`.  `help` is a statement of
// the C0 grammar, parsed and executed inside Console/Core/, which is frozen for
// slot 8 (the shared-file rule: "Console/Core/: nobody").  So C4 ships the
// generator as a BINDING verb, `help_markdown()`, returning the manual as one
// String; `print(help_markdown())` prints it bare.  The reconciliation with
// [P]'s spelling is handed to [P] v3.6 (WP-13).  Regenerate with
//     docs/console/gen_console_md.sh [path/to/cohomology_app]
//
// DETERMINISM.  The text is a pure function of the registry and the settings
// SPECS -- never of a value read from the environment or the machine: an
// engine knob's default is the environment's (threads = the hardware's), so the
// manual names its variable instead of printing a value; only the mathematical
// settings, whose defaults the engine owns, print a default.  Verbs, types and
// settings appear in REGISTRATION order (Registry keeps it), which is also
// overload-resolution order.
// ===========================================================================

#include "Bindings.h"

#include "../Core/ConsoleCli.h"
#include "../Core/Printer.h"
#include "../Core/Settings.h"

#include <map>
#include <string>
#include <vector>

namespace console {
namespace bind {

namespace {

// Markdown table cells: a `|` would end the cell, a newline the row.
std::string Cell(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '|') out += "\\|";
        else if (c == '\n') out += ' ';
        else out += c;
    }
    return out;
}

const char* const kLanguage = R"MD(## Language

OC script is a small, deterministic expression language. The normative specification is
[P] `docs/console/00-console-interface-plan.md` Sec. 3; this section restates it for the reader.

**Lexical.** A newline ends a statement unless brackets are open or the line ends in `\`; `;` also
ends one and, after an expression statement, suppresses its printing. Comments run from `#` or
`//` to the end of the line. Identifiers are `[A-Za-z_][A-Za-z0-9_]*`, case-sensitive; reserved:
`for in if else set help quit true false and or not expect_error include args def`. Integers are
decimal or `0x` hex, with `_` separators (`134_217_757`), of type `Int` = `int64_t` (overflow is an
error, never a wrap). Strings are `"..."` with `\" \\ \n \t`; `str(x)` and `+` concatenate.

**Grammar.**

```
program    := { statement }
statement  := assign | exprstmt | for | if | setstmt | include | help | expect | quit
assign     := IDENT "=" expr TERM
quit       := "quit" TERM
exprstmt   := expr TERM
for        := "for" IDENT "in" expr block
if         := "if" expr block [ "else" ( block | if ) ]
setstmt    := "set" IDENT "=" expr TERM
include    := "include" "(" STRING ")" TERM
help       := ( "help" [ IDENT | "?" ] | "?" IDENT ) TERM
expect     := "expect_error" STRING block
block      := "{" { statement } "}"
TERM       := NEWLINE | ";"
expr       := orexpr
orexpr     := andexpr { "or" andexpr }
andexpr    := notexpr { "and" notexpr }
notexpr    := "not" notexpr | cmp
cmp        := arith [ ("==" | "!=" | "<" | "<=" | ">" | ">=") arith ]
arith      := term { ("+" | "-") term }
term       := power { ("*" | "/" | "%") power }
power      := unary [ "^" power ]                         (right-assoc)
unary      := "-" unary | postfix
postfix    := primary { "(" args ")" | "." IDENT | "[" expr "]" }
primary    := INT | STRING | "true" | "false" | IDENT | "args" | "(" expr ")" | "[" [ expr { "," expr } ] "]"
args       := [ arg { "," arg } ]           arg := expr | IDENT "=" expr
```

**Two rules to know.**

* **The unary minus binds inside the base of `^`**: `power := unary [ "^" power ]`, so
  `-2 ^ 2 == 4` and `0 - 2 ^ 2 == -4`. This is *not* the school convention.
* **Strings print bare only at the top of `print` / `str`**: `print("dim")` prints `dim`, while a
  bare expression statement `"dim"`, and a string inside a list or record anywhere, prints quoted
  (`"dim"`, `["a", "b"]`).

**Semantics in brief.** `/` on `Int` is exact and an error on a non-zero remainder; on `Fp` it is
field division. `^` on `Int` is exact and overflow-checked. Comparison of objects is mathematical
equality (matrices entrywise). `include("f.oc")` resolves relative to the including file, is
cycle-checked and runs in the same global frame. Pipeline objects are **lazy handles**: a verb
returns at once, the work runs on first access to a computed member, under the settings
**snapshot taken when the handle was created**; a handle whose computation failed stays bound in
state `failed` and is never retried. `?x` describes a value (type, state, settings snapshot,
creation site, members); `help` lists everything, `help <name>` one verb, type or setting.
`expect_error "substring" { ... }` succeeds iff the block raises an error containing the
substring; it is the only error-consuming construct (there is no `try`). Lines a script prints
with the prefix `#~ ` are annotations (timings), excluded from every transcript comparison.
)MD";

const char* const kErrors = R"MD(## Exit codes and the error model

| Exit | Meaning |
|---|---|
| 0 | success |
| 1 | usage: an unrecognised console argument, a script file not found, an undeclared `--set` key or an invalid `--set` value |
| 2 | parse error -- reported before anything runs, so a script with one has no side effect |
| 3 | runtime error: a console type / arity error, a pipeline contract violation, every `ENGINE_CHECK`, `hecke::HeckeOverflow`, out of memory, a failed handle |
| 4 | a failed `assert(cond, msg)` or `expect_error` |
| 5 | a legal setting this build cannot execute (`hecke::NotImplementedInThisBuild`) -- on a `set` statement, as a `--set` argument, or at a verb or member that does not honour the setting in force (e.g. [E] `sec:ring` (h)'s list under `coefficient_ring = Z/ell^<k>`); never reported as a bad value, never degraded to the default |

Every error is printed as `file:line:col: kind: message`, followed by the offending source line;
a C++ exception names its class (`caused by: hecke::HeckeOverflow: ...`, `std::logic_error`, ...),
and an `ENGINE_CHECK` keeps its own `file:line`. A lazy failure names **both** the access site and
the handle's creation site (`object of type ... created at file:line:col by <expression>`). In a
script every error is fatal; in the REPL every error leaves the session usable, and a verb-time
error binds nothing. `ConsoleErrorValidation` gates every row of this table.
)MD";

void AppendVerbs(const Registry& reg, std::string& out) {
    out += "## Verbs\n\n";
    out += "In registration order, which is also the overload-resolution order (the first "
           "overload that accepts the arguments wins). `x: Kind = v` is an optional parameter "
           "with its default.\n";
    std::string group;
    for (const std::string& name : reg.verb_names()) {
        const VerbEntry* v = reg.find_verb(name);
        if (v == nullptr) continue;
        if (v->group != group) {
            group = v->group;
            out += "\n### Group `" + group + "`\n\n";
        }
        for (const Overload& o : v->overloads) {
            out += "* `" + o.sig.to_string(v->name) + "`";
            if (!o.doc.empty()) out += " -- " + o.doc;
            out += "\n";
        }
    }
    out += "\n";
}

void AppendTypes(const Registry& reg, std::string& out) {
    out += "## Types\n\n";
    out += "Members marked *computed* force the handle's computation; *structural* members are "
           "available while it is still `declared`.\n";
    for (const std::string& name : reg.type_names()) {
        const TypeInfo* t = reg.find_type(name);
        if (t == nullptr) continue;
        out += "\n### `" + t->name + "`\n\n";
        if (!t->doc.empty()) out += t->doc + "\n\n";
        if (t->members.empty() && t->methods.empty()) {
            out += "(no members)\n";
            continue;
        }
        out += "| member | | description |\n|---|---|---|\n";
        for (const TypeInfo::Member& m : t->members) {
            out += "| `." + m.name + "` | " + (m.computed ? "computed" : "structural") + " | " +
                   Cell(m.doc) + " |\n";
        }
        for (const TypeInfo::Method& m : t->methods) {
            std::string sig = "." + m.name + "(";
            for (std::size_t i = 0; i < m.params.size(); ++i) {
                if (i != 0) sig += ", ";
                sig += m.params[i];
            }
            sig += ")";
            out += "| `" + sig + "` | " + (m.computed ? "computed" : "structural") + " | " +
                   Cell(m.doc) + " |\n";
        }
    }
    out += "\n";
}

void AppendSettings(const Settings& settings, std::string& out) {
    out += "## Settings\n\n";
    out += "`set key = value` changes a setting for the rest of the session; `settings()` shows "
           "every key with its value and source; `reset_settings()` returns to the session's "
           "baseline. **Mathematical** settings may change a published value, enter the "
           "fingerprint (`fingerprint()`, `0x0000000000000000` at the defaults) and are the "
           "engine's own (validated by `hecke::MathSettings`); `golden(...)` refuses at a "
           "non-zero fingerprint. **Engine** settings change how fast a value is produced, never "
           "the value; their defaults are the environment's, so this manual names the variable "
           "instead of a value. *env-only* keys are shown by `settings()` but cannot be `set`.\n\n";
    out += "| key | family | kind | default / source | description |\n|---|---|---|---|---|\n";
    for (const std::string& name : settings.names()) {
        const SettingSpec* s = settings.find_spec(name);
        if (s == nullptr) continue;
        const bool math = (s->family == SettingFamily::Mathematical);
        const char* kind = s->kind == SettingKind::Int    ? "Int"
                           : s->kind == SettingKind::Bool ? "Bool"
                                                          : "String";
        std::string def;
        if (math) {
            def = "`" + PrintCanonical(s->default_value, kDefaultPrintMaxEntries) + "`";
        } else if (!s->env_var.empty()) {
            def = "env `" + s->env_var + "`";
        } else {
            def = "session";
        }
        if (s->source == SettingSource::EnvOnly) def += " (env-only)";
        std::string doc = s->doc;
        if (!s->allowed.empty()) {
            doc += " (one of:";
            for (std::size_t i = 0; i < s->allowed.size(); ++i) {
                doc += (i == 0 ? " " : ", ");
                doc += s->allowed[i];
            }
            doc += ")";
        }
        out += "| `" + s->name + "` | " + (math ? "mathematical" : "engine") + " | " + kind +
               " | " + Cell(def) + " | " + Cell(doc) + " |\n";
    }
    out += "\n";
}

Value VerbHelpMarkdown(CallContext& ctx, const std::vector<Value>&) {
    CONSOLE_CHECK(ctx.registry != nullptr && ctx.session != nullptr);
    std::string out;
    out += "# OC script -- the console reference\n\n";
    out += "<!-- GENERATED by docs/console/gen_console_md.sh, i.e. `hacs -c "
           "'print(help_markdown())'`, from the registry of the build it runs in. Do not edit "
           "by hand: regenerate when a verb, type or setting is registered. -->\n\n";
    out += "The console of `hacs`: a small, deterministic, hand-rolled expression "
           "language over the project's pipeline (cohomology of `SL(2,Z)` with `Sym^n` "
           "coefficients, Hecke operators, their decomposition, the extension layer, and the "
           "benchmark / campaign / validation harness). Binding design: [P] "
           "`docs/console/00-console-interface-plan.md`. This manual is generated; its verb, "
           "type and settings tables are the registry itself.\n\n";
    out += "## Running the console\n\n```\n" + ConsoleUsage() + "```\n\n";
    out += kLanguage;
    out += "\n";
    out += kErrors;
    out += "\n";
    AppendVerbs(*ctx.registry, out);
    AppendTypes(*ctx.registry, out);
    AppendSettings(ctx.session->settings(), out);
    return Value::from_string(out);
}

} // namespace

void RegisterReferenceBindings(Registry& registry) {
    registry.register_verb("help_markdown", Sig({}), &VerbHelpMarkdown,
                           "the console reference manual (docs/console/console.md) as Markdown, "
                           "generated from this registry -- [P]'s `help --markdown`",
                           "reference");
}

} // namespace bind
} // namespace console
