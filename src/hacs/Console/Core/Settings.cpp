// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Settings.cpp -- see Settings.h.
// ===========================================================================

#include "Settings.h"

#include <algorithm>

namespace console {

namespace {

const char* SettingKindName(SettingKind kind) {
    switch (kind) {
        case SettingKind::Int:    return "Int";
        case SettingKind::Bool:   return "Bool";
        case SettingKind::String: return "String";
    }
    return "String";
}

bool KindMatches(SettingKind kind, const Value& v) {
    switch (kind) {
        case SettingKind::Int:    return v.is_int();
        case SettingKind::Bool:   return v.is_bool();
        case SettingKind::String: return v.is_string();
    }
    return false;
}

std::string ValueText(const Value& v) {
    switch (v.kind()) {
        case ValueKind::Int:    return std::to_string(v.int_value());
        case ValueKind::Bool:   return v.bool_value() ? "true" : "false";
        case ValueKind::String: return v.string_value();
        default:                return std::string(ValueKindName(v.kind()));
    }
}

} // namespace

// --- SettingsSnapshot ------------------------------------------------------

const Value* SettingsSnapshot::find(const std::string& name) const {
    for (const std::pair<std::string, Value>& e : values_) {
        if (e.first == name) return &e.second;
    }
    return nullptr;
}

SettingsSnapshot SettingsSnapshot::mathematical() const {
    SettingsSnapshot out;
    for (std::size_t i = 0; i < values_.size(); ++i) {
        if (i < families_.size() && families_[i] == SettingFamily::Mathematical) {
            out.values_.push_back(values_[i]);
            out.families_.push_back(SettingFamily::Mathematical);
        }
    }
    return out;
}

Record SettingsSnapshot::to_record() const {
    Record r;
    for (const std::pair<std::string, Value>& e : values_) r.set(e.first, e.second);
    return r;
}

std::string SettingsSnapshot::to_string() const {
    std::string out;
    for (std::size_t i = 0; i < values_.size(); ++i) {
        if (i != 0) out += ", ";
        out += values_[i].first;
        out += " = ";
        out += ValueText(values_[i].second);
    }
    return out;
}

SettingsSnapshot SettingsSnapshot::with(const std::string& name, Value value) const {
    SettingsSnapshot out = *this;
    for (std::pair<std::string, Value>& e : out.values_) {
        if (e.first == name) {
            e.second = std::move(value);
            return out;
        }
    }
    out.values_.emplace_back(name, std::move(value));
    out.families_.push_back(SettingFamily::Mathematical);
    return out;
}

bool SettingsSnapshot::equals(const SettingsSnapshot& other) const {
    if (values_.size() != other.values_.size()) return false;
    for (std::size_t i = 0; i < values_.size(); ++i) {
        if (values_[i].first != other.values_[i].first) return false;
        if (!values_[i].second.equals(other.values_[i].second)) return false;
    }
    return true;
}

// --- Settings --------------------------------------------------------------

Settings::Entry* Settings::find_entry(const std::string& name) {
    for (Entry& e : entries_) {
        if (e.spec.name == name) return &e;
    }
    return nullptr;
}

const Settings::Entry* Settings::find_entry(const std::string& name) const {
    for (const Entry& e : entries_) {
        if (e.spec.name == name) return &e;
    }
    return nullptr;
}

void Settings::register_setting(SettingSpec spec) {
    CONSOLE_CHECK_MSG(!spec.name.empty(), "a setting must have a name");
    CONSOLE_CHECK_MSG(KindMatches(spec.kind, spec.default_value),
                      "a setting's default value must match its declared kind");
    Entry* existing = find_entry(spec.name);
    if (existing != nullptr) {
        existing->value = spec.default_value;
        existing->spec = std::move(spec);
        return;
    }
    Entry e;
    e.value = spec.default_value;
    e.spec = std::move(spec);
    entries_.push_back(std::move(e));
}

const SettingSpec* Settings::find_spec(const std::string& name) const {
    const Entry* e = find_entry(name);
    return e != nullptr ? &e->spec : nullptr;
}

std::vector<std::string> Settings::names() const {
    std::vector<std::string> out;
    out.reserve(entries_.size());
    for (const Entry& e : entries_) out.push_back(e.spec.name);
    return out;
}

void Settings::set(const std::string& name, const Value& value, const SourcePos& pos) {
    Entry* e = find_entry(name);
    if (e == nullptr) {
        std::string msg = "unknown setting '" + name + "'";
        if (!entries_.empty()) {
            msg += " (known settings: ";
            for (std::size_t i = 0; i < entries_.size(); ++i) {
                if (i != 0) msg += ", ";
                msg += entries_[i].spec.name;
            }
            msg += ")";
        } else {
            msg += " (no settings are registered in this session)";
        }
        throw TypeError(msg, pos);
    }
    if (e->spec.source == SettingSource::EnvOnly) {
        // Plan Sec. 2 C4: "A knob with no in-process override is documented as
        // env-only, never silently accepted."
        std::string msg = "setting '" + name + "' is environment-only";
        if (!e->spec.env_var.empty()) {
            msg += "; set the environment variable " + e->spec.env_var +
                   " before starting the process";
        }
        throw TypeError(msg, pos);
    }
    if (!KindMatches(e->spec.kind, value)) {
        throw TypeError("setting '" + name + "' expects " +
                            std::string(SettingKindName(e->spec.kind)) + ", got " +
                            value.type_name(),
                        pos);
    }
    if (!e->spec.allowed.empty() && value.is_string()) {
        if (std::find(e->spec.allowed.begin(), e->spec.allowed.end(), value.string_value()) ==
            e->spec.allowed.end()) {
            std::string msg = "setting '" + name + "' accepts only ";
            for (std::size_t i = 0; i < e->spec.allowed.size(); ++i) {
                if (i != 0) msg += " | ";
                msg += e->spec.allowed[i];
            }
            msg += "; got '" + value.string_value() + "'";
            throw TypeError(msg, pos);
        }
    }
    if (e->spec.validate) {
        const std::string reason = e->spec.validate(value);
        if (!reason.empty()) {
            throw TypeError("setting '" + name + "': " + reason, pos);
        }
    }
    e->value = value;
    if (e->spec.apply) e->spec.apply(value);
}

const Value* Settings::get(const std::string& name) const {
    const Entry* e = find_entry(name);
    return e != nullptr ? &e->value : nullptr;
}

SettingsSnapshot Settings::snapshot() const {
    SettingsSnapshot snap;
    snap.values_.reserve(entries_.size());
    snap.families_.reserve(entries_.size());
    for (const Entry& e : entries_) {
        snap.values_.emplace_back(e.spec.name, e.value);
        snap.families_.push_back(e.spec.family);
    }
    return snap;
}

void Settings::install(const SettingsSnapshot& snap) {
    for (const std::pair<std::string, Value>& kv : snap.values()) {
        Entry* e = find_entry(kv.first);
        if (e == nullptr) continue;            // unknown here: ignore, never throw
        if (e->spec.source == SettingSource::EnvOnly) continue;
        e->value = kv.second;
        if (e->spec.apply) e->spec.apply(kv.second);
    }
}

void Settings::reset_to_baseline() {
    install(baseline_);
}

Record Settings::describe() const {
    Record out;
    for (const Entry& e : entries_) {
        Record row;
        row.set("value", e.value);
        row.set("kind", Value::from_string(SettingKindName(e.spec.kind)));
        row.set("source", Value::from_string(e.spec.source == SettingSource::EnvOnly
                                                 ? "env"
                                                 : "session"));
        row.set("family", Value::from_string(e.spec.family == SettingFamily::Mathematical
                                                 ? "mathematical"
                                                 : "engine"));
        if (!e.spec.env_var.empty()) row.set("env_var", Value::from_string(e.spec.env_var));
        if (!e.spec.doc.empty()) row.set("doc", Value::from_string(e.spec.doc));
        out.set(e.spec.name, Value::from_record(std::move(row)));
    }
    return out;
}

} // namespace console
