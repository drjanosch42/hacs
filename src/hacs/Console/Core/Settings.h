// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Settings.h -- console C0: the settings layer and the snapshot every handle
// carries.
//
// Two binding sources:
//
//  * plan Sec. 2 C4 ("Session settings instead of environment variables"):
//    the session holds the AUTHORITATIVE value of every setting and applies it
//    through a plain setter.  There is deliberately NO RAII stack (audit
//    amendment A2): a stack of Scoped* guards destroyed out of LIFO order
//    restores stale values, so a snapshot is installed by re-applying every
//    value, which is idempotent and order-independent.
//
//  * hecke_extensions_owner_decisions.md R1 ("A configuration / settings layer
//    for mathematical conventions and bounds") and R3 ("Resumable budgeted
//    computations").  R1 adds a SECOND family of settings -- final_mode,
//    sturm.cap, locality, field_polynomial, coefficient_ring, ... -- which are
//    part of what a result MEANS, so every record and every console handle
//    must store the values in force.  That is what SettingSpec::mathematical
//    marks and what SettingsSnapshot::mathematical() extracts.
//
// C0 registers only the two CONSOLE-OWNED presentation keys (print_max_entries,
// verbose; see Session.h).  It registers no engine knob and no mathematical
// setting: the engine owns their defaults and their validation (R1, "The engine
// owns the defaults ...  The console sets them through the registry and never
// re-implements them"), C2 registers the engine knobs and the extension engine
// the mathematical ones.  What C0 owns is the MECHANISM: the spec, the
// registry, the authoritative store, the snapshot, and the env-only diagnosis.
//
// STL-ONLY (plan Sec. 2 C8).
// ===========================================================================
#pragma once
#ifndef CONSOLE_SETTINGS_H
#define CONSOLE_SETTINGS_H

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "ConsoleError.h"
#include "Value.h"

namespace console {

enum class SettingKind : std::uint8_t { Int, Bool, String };

enum class SettingSource : std::uint8_t {
    Session = 0,   // settable by `set key = value`
    EnvOnly = 1    // read from the environment once; `set` on it is an error
                   // that names the variable (plan Sec. 2 C4, last table row)
};

// R1's "family" column: an ENGINE setting changes how fast a result is
// computed, a MATHEMATICAL setting changes what the result MEANS.  Only the
// latter enters a reproducibility record's hash input ("A setting that changes
// a published value must change the record's hash input, so golden objects pin
// the default configuration").
enum class SettingFamily : std::uint8_t { Engine = 0, Mathematical = 1 };

struct SettingSpec {
    std::string name;
    SettingKind kind = SettingKind::String;
    Value default_value;
    SettingSource source = SettingSource::Session;
    SettingFamily family = SettingFamily::Engine;
    std::string env_var;                      // the environment variable this mirrors, if any
    std::vector<std::string> allowed;         // enumerated String settings ("closure" | "sturm_heuristic")
    std::string doc;

    // Engine-owned hooks.  `validate` returns an empty string when the value is
    // acceptable and the reason otherwise; `apply` is the plain setter (C2
    // binds set_consumer_pool_override, elim::ScopedLegacy, ...).  Both may be
    // empty: C0 exercises the layer with pure value semantics.
    std::function<std::string(const Value&)> validate;
    std::function<void(const Value&)> apply;
};

// An immutable, ordered copy of every setting in force at one instant.  Every
// Object snapshots one at CREATION (plan Sec. 2 C2, "Settings are captured at
// creation, not at compute") and installs it around its own compute().
class SettingsSnapshot {
public:
    SettingsSnapshot() = default;

    const std::vector<std::pair<std::string, Value>>& values() const { return values_; }
    const Value* find(const std::string& name) const;
    bool empty() const { return values_.empty(); }

    // The R1 subset: the settings that change what a result means.
    SettingsSnapshot mathematical() const;

    Record to_record() const;
    // "threads = 4, backend = legacy" -- one line, for `?x`.
    std::string to_string() const;

    // A snapshot with `name` replaced (used by Object::resume, R3: "a resume
    // with a changed budget is a new setting and must be recorded").
    SettingsSnapshot with(const std::string& name, Value value) const;

    bool equals(const SettingsSnapshot& other) const;

private:
    friend class Settings;
    std::vector<std::pair<std::string, Value>> values_;
    std::vector<SettingFamily> families_;
};

// The session's authoritative store.
class Settings {
public:
    Settings() = default;

    // Registration is explicit and additive (the ValidationRegistry pattern).
    // Re-registering a name replaces the spec and re-seeds the value from the
    // new default.
    void register_setting(SettingSpec spec);
    const SettingSpec* find_spec(const std::string& name) const;
    std::vector<std::string> names() const;

    // `set key = value`.  Throws TypeError naming the key (and, for an env-only
    // key, the environment variable) exactly as plan Sec. 2 C4 requires.
    void set(const std::string& name, const Value& value, const SourcePos& pos);

    const Value* get(const std::string& name) const;

    SettingsSnapshot snapshot() const;
    // Re-apply every value of `snap` that this store knows (plan Sec. 2 C4,
    // "applies them through the plain setters").  Unknown names are ignored --
    // a snapshot taken before a type registered its settings must still
    // install.
    void install(const SettingsSnapshot& snap);

    // Re-applies the snapshot taken when the session started (the environment's
    // values): the verb `reset_settings()`.
    void reset_to_baseline();
    void set_baseline(SettingsSnapshot snap) { baseline_ = std::move(snap); }
    const SettingsSnapshot& baseline() const { return baseline_; }

    // Every setting with its value, source and family -- the verb `settings()`.
    Record describe() const;

private:
    struct Entry {
        SettingSpec spec;
        Value value;
    };
    std::vector<Entry> entries_;   // registration order; no hash container
    SettingsSnapshot baseline_;

    Entry* find_entry(const std::string& name);
    const Entry* find_entry(const std::string& name) const;
};

// Installs a snapshot for the duration of a scope and re-installs the previous
// values afterwards.  Because installing means "apply every value", nesting is
// safe in any order -- unlike a stack of per-setting guards (A2).
class SettingsScope {
public:
    SettingsScope(Settings& settings, const SettingsSnapshot& snap)
        : settings_(settings), previous_(settings.snapshot()) {
        settings_.install(snap);
    }
    ~SettingsScope() { settings_.install(previous_); }

    SettingsScope(const SettingsScope&) = delete;
    SettingsScope& operator=(const SettingsScope&) = delete;

private:
    Settings& settings_;
    SettingsSnapshot previous_;
};

} // namespace console

#endif // CONSOLE_SETTINGS_H
