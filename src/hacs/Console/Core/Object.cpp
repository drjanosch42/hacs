// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Object.cpp -- see Object.h.
// ===========================================================================

#include "Object.h"

namespace console {

const char* ObjectStateName(ObjectState state) {
    switch (state) {
        case ObjectState::Declared: return "declared";
        case ObjectState::Computed: return "computed";
        case ObjectState::Failed:   return "failed";
    }
    return "declared";
}

const TypeInfo::Member* TypeInfo::find_member(const std::string& n) const {
    for (const Member& m : members) {
        if (m.name == n) return &m;
    }
    return nullptr;
}

const TypeInfo::Method* TypeInfo::find_method(const std::string& n) const {
    for (const Method& m : methods) {
        if (m.name == n) return &m;
    }
    return nullptr;
}

Record ResumableState::to_record() const {
    Record r;
    r.set("resumable", Value::from_bool(resumable));
    r.set("budget_setting", Value::from_string(budget_setting));
    r.set("budget_value", budget_value);
    r.set("reason", Value::from_string(reason));
    r.set("progress", Value::from_record(progress));
    return r;
}

ObjectPtr Object::create(const TypeInfo* type, SourcePos creation_site,
                         std::string creation_expr, SettingsSnapshot snapshot,
                         std::shared_ptr<void> payload, ComputeFn compute) {
    CONSOLE_CHECK(type != nullptr);
    // std::make_shared cannot reach the private constructor; the explicit
    // `new` is the standard workaround and is immediately owned.
    ObjectPtr obj(new Object());
    obj->type_ = type;
    obj->creation_site_ = std::move(creation_site);
    obj->creation_expr_ = std::move(creation_expr);
    obj->snapshot_ = std::move(snapshot);
    obj->payload_ = std::move(payload);
    obj->compute_ = std::move(compute);
    // An object with no compute step is born computed: its payload is already
    // there (a wrapped matrix, a record-like result).
    obj->state_ = obj->compute_ ? ObjectState::Declared : ObjectState::Computed;
    return obj;
}

const std::string& Object::type_name() const {
    static const std::string unknown = "Object";
    return type_ != nullptr ? type_->name : unknown;
}

void Object::keep_alive(ObjectPtr other) {
    if (!other) return;
    CONSOLE_CHECK_MSG(other.get() != this, "an object cannot keep itself alive");
    keep_alive_.push_back(std::move(other));
}

void Object::ensure_computed(Settings& settings, const SourcePos& access) {
    if (state_ == ObjectState::Computed) return;

    if (state_ == ObjectState::Failed) {
        // Never retried (plan Sec. 2 C2, "Failure is a bound state").
        RuntimeError err(creation_expr_.empty()
                             ? std::string("this object's computation failed earlier")
                             : creation_expr_ + ": computation failed earlier",
                         access);
        err.add_note(CreationSiteNote(shared_from_this()));
        err.add_note("caused by: " + error_text_);
        err.add_note("a failed handle is never retried; construct a new object");
        throw err;
    }

    CONSOLE_CHECK(compute_ != nullptr);
    // The snapshot taken at CREATION, not the session's current values.
    SettingsScope scope(settings, snapshot_);
    try {
        compute_(*this);
    } catch (const ConsoleError& e) {
        state_ = ObjectState::Failed;
        error_text_ = e.formatted();
        const std::string message =
            (creation_expr_.empty() ? std::string("object") : creation_expr_) +
            ": compute() failed";
        // (C1a) A legal setting this build cannot execute keeps its OWN kind --
        // and therefore exit code 5 -- through the handle wrapper (plan
        // Sec. 2 C7).  The binding's compute() is what recognises it, so the
        // core still names no pipeline type.
        if (e.kind() == kNotImplementedKind) {
            NotImplementedError err(message, access);
            err.add_note(CreationSiteNote(shared_from_this()));
            err.add_note("caused by: " + error_text_);
            throw err;
        }
        RuntimeError err(message, access);
        err.add_note(CreationSiteNote(shared_from_this()));
        err.add_note("caused by: " + error_text_);
        throw err;
    } catch (const std::exception& e) {
        state_ = ObjectState::Failed;
        error_text_ = e.what();
        RuntimeError err((creation_expr_.empty() ? std::string("object") : creation_expr_) +
                             ": compute() failed",
                         access);
        err.add_note(CreationSiteNote(shared_from_this()));
        err.add_note("caused by: " + error_text_);
        throw err;
    }
    state_ = ObjectState::Computed;
}

std::string CreationSiteNote(const ObjectPtr& obj) {
    if (!obj) return std::string();
    std::string note = "object of type " + obj->type_name();
    if (obj->creation_site().known()) {
        note += " created at " + obj->creation_site().to_string();
    }
    if (!obj->creation_expr().empty()) {
        note += " by " + obj->creation_expr();
    }
    return note;
}

} // namespace console
