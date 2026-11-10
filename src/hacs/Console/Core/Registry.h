// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Registry.h -- console C0: the type and verb registry.
//
// Plan Sec. 2 C3: "The evaluator knows NO verb and NO type.  Both come from a
// registry", populated by EXPLICIT registration functions -- never by
// self-registering statics, which the linker drops across a static-library
// boundary (the reason ValidationRegistry.h:43-50 records).  C0 registers only
// the core verbs (RegisterCoreVerbs, CoreVerbs.h); C1 adds
// RegisterPipelineVerbs, and the extension engine its own function, WITHOUT
// editing the evaluator.
//
// Overload resolution (plan Sec. 2 C3): by arity, then by argument type, FIRST
// MATCH IN REGISTRATION ORDER -- so an extension can add charpoly(MatrixQ)
// next to charpoly(Matrix) without touching either.  Keyword arguments are
// bound per overload before its types are checked, because the parameter names
// belong to the overload.
//
// STL-ONLY (plan Sec. 2 C8).
// ===========================================================================
#pragma once
#ifndef CONSOLE_REGISTRY_H
#define CONSOLE_REGISTRY_H

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "ConsoleError.h"
#include "Object.h"
#include "Value.h"

namespace console {

class Session;
class Registry;
class Settings;

// What a verb implementation is handed.  The evaluator owns it; a verb may use
// `session` to print, to read settings, to create handles (which need the
// session's settings snapshot) and to reach the registry.
struct CallContext {
    Session* session = nullptr;
    Registry* registry = nullptr;
    SourcePos pos;              // the call site
    std::string call_text;      // "hecke(H, p = 2)" -- the creation expression of any handle made here
};

// A formal parameter.  `kind == ValueKind::Nil` means "any value"; for
// ValueKind::Object, a non-empty `type_name` additionally pins the console type.
struct Param {
    std::string name;
    ValueKind kind = ValueKind::Nil;
    std::string type_name;
    bool required = true;
    Value default_value;
    std::string doc;

    static Param any(std::string n);
    static Param of(std::string n, ValueKind k);
    static Param object(std::string n, std::string type);
    Param& optional(Value dflt);
    Param& described(std::string d);
};

struct Signature {
    std::vector<Param> params;
    // A variadic signature repeats its LAST parameter zero or more times
    // (print(a, b, c)).  Its other parameters are matched normally.
    bool variadic = false;

    std::size_t required_count() const;
    // "print(value, ...)" / "range(a, b, step = 1)"
    std::string to_string(const std::string& verb) const;
};

using VerbImpl = std::function<Value(CallContext&, const std::vector<Value>&)>;

struct Overload {
    Signature sig;
    VerbImpl impl;
    std::string doc;
};

struct VerbEntry {
    std::string name;
    std::string group;               // "core", "arithmetic", "pipeline", "harness"
    std::vector<Overload> overloads; // registration order == resolution order
};

class Registry {
public:
    Registry() = default;
    Registry(const Registry&) = delete;
    Registry& operator=(const Registry&) = delete;

    // --- types -------------------------------------------------------------
    // Returns a stable pointer: TypeInfos live in unique_ptrs for the life of
    // the registry, so a handle may hold a raw TypeInfo* (plan Sec. 3.6).
    const TypeInfo* register_type(TypeInfo info);
    const TypeInfo* find_type(const std::string& name) const;
    std::vector<std::string> type_names() const;

    // --- verbs -------------------------------------------------------------
    void register_verb(const std::string& name, Signature sig, VerbImpl impl,
                       std::string doc, std::string group = "core");
    const VerbEntry* find_verb(const std::string& name) const;
    std::vector<std::string> verb_names() const;

    // (slot-19 NB-2, [P] v3.10 Sec. 18.8) Restricts THIS registry to a view:
    // `f` is called on every overload in registration order and may rewrite
    // it (its impl / doc); returning false removes it, and a verb left with
    // no overload is removed.  Types are untouched (so every TypeInfo keeps
    // its address).  The served console (`--serve`, one session per
    // process) calls it, once, after registration, to build its view.
    // (slot-24 D6 audit m1) The bindings' field-verb gate
    // (bind::GateVerbsOnField) calls it at registration, on every front end,
    // to wrap -- never remove -- the overloads of the verbs it names.
    using OverloadTransform = std::function<bool(const VerbEntry&, Overload&)>;
    void transform_overloads(const OverloadTransform& f);

    // Binds `positional` / `keyword` arguments to the first overload of `entry`
    // that accepts them, and returns the bound argument vector in parameter
    // order.  Throws TypeError listing every overload when nothing matches --
    // plan Sec. 8, "console type / arity -> error with the verb's signature
    // list; nothing bound".
    const Overload* resolve(const VerbEntry& entry,
                            const std::vector<Value>& positional,
                            const std::vector<std::pair<std::string, Value>>& keyword,
                            const SourcePos& pos,
                            std::vector<Value>& bound_out) const;

    // C1 installs the catch-order naming of plan Sec. 2 C7 here (HeckeOverflow
    // before std::runtime_error, ENGINE_CHECK's std::logic_error verbatim, ...)
    // so that Core needs no pipeline header.  Returns true and fills
    // `class_name` when it recognises the exception.
    using ExceptionTranslator = std::function<bool(const std::exception&, std::string& class_name)>;
    void set_exception_translator(ExceptionTranslator t) { translator_ = std::move(t); }
    bool translate_exception(const std::exception& e, std::string& class_name) const {
        return translator_ ? translator_(e, class_name) : false;
    }

    // (C1a) The second half of the same seam: which console error KIND -- and
    // therefore which exit code -- a recognised exception becomes.  An empty
    // string (the default, and what an unrecognised exception gets) means the
    // ordinary runtime error of exit code 3.  The C1 bindings install a
    // classifier returning `kNotImplementedKind` for
    // hecke::NotImplementedInThisBuild (plan Sec. 2 C7, exit code 5), so the
    // core still names no pipeline type.
    using ExceptionKindFn = std::function<std::string(const std::exception&)>;
    void set_exception_kind(ExceptionKindFn f) { kind_fn_ = std::move(f); }
    std::string exception_kind(const std::exception& e) const {
        return kind_fn_ ? kind_fn_(e) : std::string();
    }

    // (C1b) The THIRD seam of the same shape: which settings the ENGINE owns.
    //
    // Plan Sec. 2 C4 / owner decision R1: "the engine owns the defaults and the
    // validation of values -- the console only *sets* them, by key".  C0
    // registers the two console-owned presentation keys and nothing else; the
    // MATHEMATICAL keys live in hecke::MathSettings, which Console/Core/ may
    // not name (Sec. 2 C8, STL-only).  The bindings therefore install an
    // installer here, and Session's constructor runs it on its own Settings
    // store right after RegisterCoreSettings -- so `set locality = strict`
    // works on every session the CLI or a validation builds, and the core
    // still names no pipeline type.
    //
    // Unset (the default, and what a C0-only registry has) means "no settings
    // beyond the core ones", which is exactly C0's behaviour.
    using SettingsInstaller = std::function<void(Settings&)>;
    void set_settings_installer(SettingsInstaller f) { settings_installer_ = std::move(f); }
    // (C2a, slot-6 WP-13) A SECOND family registers through the same seam: the
    // engine knobs (Console/Bindings/EngineSettings.cpp).  An added installer
    // runs after the one `set_settings_installer` holds, in the order added,
    // so `settings()` lists the families in registration order.
    void add_settings_installer(SettingsInstaller f) {
        extra_installers_.push_back(std::move(f));
    }
    void install_settings(Settings& settings) const {
        if (settings_installer_) settings_installer_(settings);
        for (const SettingsInstaller& f : extra_installers_) {
            if (f) f(settings);
        }
    }

    // The process-wide registry the shipped console uses.  Validations build
    // their own instance instead, so registering a stub type cannot leak
    // between tests.
    static Registry& global();

private:
    std::vector<std::unique_ptr<TypeInfo>> types_;   // stable addresses
    // unique_ptr so that a VerbEntry* handed out by find_verb stays valid
    // when a later registration grows the table.
    std::vector<std::unique_ptr<VerbEntry>> verbs_;  // registration order; no hash container
    ExceptionTranslator translator_;
    ExceptionKindFn kind_fn_;
    SettingsInstaller settings_installer_;
    std::vector<SettingsInstaller> extra_installers_;
};

// True when `v` can be passed as `p`.
bool ParamAccepts(const Param& p, const Value& v);

} // namespace console

#endif // CONSOLE_REGISTRY_H
