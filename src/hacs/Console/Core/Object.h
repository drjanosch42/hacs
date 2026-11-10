// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Object.h -- console C0: the handle model and the type registry entry.
//
// Plan Sec. 2 C2 ("Handles, not copies"), Sec. 4.1-4.4 and Sec. 3.6.  The core
// language knows NO pipeline type: `Object` is a type-erased handle carrying
//
//   * a TypeInfo* (name, members, methods, printers) supplied by the bindings;
//   * `payload`, a shared_ptr<void> holding the C++ instance;
//   * keep-alive shared_ptrs to every handle the payload may dereference --
//     the pipeline's own contract is raw pointers with "must outlive this
//     object" (HeckeOperatorOnK.h:199-207), so the edges are what make
//     `T = hecke(cohomology(L), p = 2)` safe (plan Sec. 4.1);
//   * the settings snapshot taken at CREATION, installed around compute()
//     (plan Sec. 2 C2, "Settings are captured at creation, not at compute");
//   * the creation site, so a LAZY failure still names the constructing line
//     (plan Sec. 2 C7, and the audit's A3(c) counter-proposal);
//   * the R3 resumable state: a budget cut-off must store its iteration counts
//     so the user can resume it with a higher budget
//     (hecke_extensions_owner_decisions.md R3, "Designed now, implemented
//     later" -- C0 ships the contract and the settings bookkeeping, the
//     budgeted computations themselves arrive with the extension engine).
//
// State machine (plan Sec. 4.2): declared -> computed | failed.  `failed` is a
// BOUND state: the error is stored, rethrown on every later access, and the
// computation is NEVER retried, because nothing in the pipeline specifies the
// object state after a throw.
//
// STL-ONLY (plan Sec. 2 C8).
// ===========================================================================
#pragma once
#ifndef CONSOLE_OBJECT_H
#define CONSOLE_OBJECT_H

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "ConsoleError.h"
#include "Settings.h"
#include "Value.h"

namespace console {

class Object;
class Session;
using ObjectPtr = std::shared_ptr<Object>;

enum class ObjectState : std::uint8_t { Declared = 0, Computed = 1, Failed = 2 };

const char* ObjectStateName(ObjectState state);

// --- TypeInfo --------------------------------------------------------------
// One per console type, registered by an explicit registration function
// (plan Sec. 2 C3) -- never by a self-registering static, which the linker
// drops across a static-library boundary (ValidationRegistry.h:43-50).
struct TypeInfo {
    // A member of a value.  `computed == true` means reading it triggers
    // compute(); `computed == false` is structural and available while the
    // handle is still `declared` (plan Sec. 4.2).
    struct Member {
        std::string name;
        bool computed = true;
        std::string doc;
        std::function<Value(const ObjectPtr&)> get;
    };

    // A method call `x.f(...)`.  Arguments arrive already evaluated and
    // already bound to the method's parameter names (see Registry.h).
    struct Method {
        std::string name;
        std::vector<std::string> params;      // documentation / keyword binding
        std::size_t required = 0;
        bool computed = true;
        std::string doc;
        std::function<Value(const ObjectPtr&, const std::vector<Value>&, const SourcePos&)> call;
    };

    std::string name;
    std::string doc;
    std::vector<Member> members;
    std::vector<Method> methods;

    // Canonical one-line / full print forms (plan Sec. 4.3).  `print` receives
    // the entry cap; an object bigger than the cap prints its summary form.
    std::function<std::string(const ObjectPtr&, std::int64_t max_entries)> print;
    std::function<std::string(const ObjectPtr&)> show;

    // Mathematical equality for `==` (plan Sec. 3.2).  Unset => handle identity.
    std::function<bool(const ObjectPtr&, const ObjectPtr&)> equal;

    // The per-kind golden hash of plan Sec. 2 C5.  C0 registers NO type that
    // fills this in and NO `hash` verb: the console must freeze no transcript
    // before the E0b re-baseline (slot-1 plan Sec. 4.1, lane C).  The slot
    // exists so that C1 can fill it without touching the core.
    std::function<std::uint64_t(const ObjectPtr&)> hash;

    // Extra `?x` lines beyond the standard ones.
    std::function<std::vector<std::pair<std::string, std::string>>(const ObjectPtr&)> describe_extra;

    // R3: build a fresh handle continuing this one under a changed budget.
    // Unset => `resume(x, ...)` reports a typed error naming the type.
    std::function<ObjectPtr(const ObjectPtr&, const SettingsSnapshot&)> resume;

    const Member* find_member(const std::string& n) const;
    const Method* find_method(const std::string& n) const;
};

// --- R3 resumable state ----------------------------------------------------
// The state a budget-cut computation persists so the console can resume it.
// Filled in by the computation itself; C0 defines the shape and the accounting.
struct ResumableState {
    bool resumable = false;         // true <=> the result is incomplete because a budget was hit
    std::string budget_setting;     // which setting was the budget ("locality", "sturm.cap", "u_ell")
    Value budget_value;             // its value when the cut happened
    std::string reason;             // human-readable ("closure not reached after 64 witnesses")
    Record progress;                // iteration counts / partial state, engine-defined

    Record to_record() const;
};

// --- Object ----------------------------------------------------------------
class Object : public std::enable_shared_from_this<Object> {
public:
    // The work a handle performs on first access of a computed member.  It runs
    // exactly once, under the handle's settings snapshot.
    using ComputeFn = std::function<void(Object&)>;

    static ObjectPtr create(const TypeInfo* type, SourcePos creation_site,
                            std::string creation_expr, SettingsSnapshot snapshot,
                            std::shared_ptr<void> payload = nullptr,
                            ComputeFn compute = nullptr);

    const TypeInfo* type() const { return type_; }
    const std::string& type_name() const;

    ObjectState state() const { return state_; }
    bool computed() const { return state_ == ObjectState::Computed; }
    bool failed() const { return state_ == ObjectState::Failed; }
    const std::string& error_text() const { return error_text_; }

    const SourcePos& creation_site() const { return creation_site_; }
    const std::string& creation_expr() const { return creation_expr_; }
    const SettingsSnapshot& snapshot() const { return snapshot_; }

    const std::shared_ptr<void>& payload() const { return payload_; }
    void set_payload(std::shared_ptr<void> p) { payload_ = std::move(p); }

    // Keep-alive edges (plan Sec. 4.1).  A handle keeps alive everything the
    // C++ object may dereference, transitively.
    void keep_alive(ObjectPtr other);
    const std::vector<ObjectPtr>& kept_alive() const { return keep_alive_; }

    // R3.
    const ResumableState& resumable_state() const { return resumable_; }
    void set_resumable_state(ResumableState s) { resumable_ = std::move(s); }
    bool is_resumable() const { return resumable_.resumable; }
    // The chain a resume produced: "resumed from <expr> at <site>".
    const ObjectPtr& resumed_from() const { return resumed_from_; }
    void set_resumed_from(ObjectPtr from) { resumed_from_ = std::move(from); }

    // Runs compute() once, under the settings snapshot, through `settings`.
    // Idempotent after success; after a failure it rethrows the STORED error
    // and never re-runs the computation.  `access` is the site that triggered
    // the work, so the message can name both sites.
    void ensure_computed(Settings& settings, const SourcePos& access);

    // Type-safe access to the payload.  Throws CONSOLE_CHECK on a mismatch --
    // a binding bug, not a user error.
    template <typename T>
    T* payload_as() const {
        return static_cast<T*>(payload_.get());
    }

private:
    Object() = default;

    const TypeInfo* type_ = nullptr;
    ObjectState state_ = ObjectState::Declared;
    std::string error_text_;
    SourcePos creation_site_;
    std::string creation_expr_;
    SettingsSnapshot snapshot_;
    std::shared_ptr<void> payload_;
    ComputeFn compute_;
    std::vector<ObjectPtr> keep_alive_;
    ResumableState resumable_;
    ObjectPtr resumed_from_;
};

// "object T2 created at script.oc:9:7 by hecke(H, p = 2)" -- the note a lazy
// failure adds to its error (plan Sec. 2 C7).
std::string CreationSiteNote(const ObjectPtr& obj);

} // namespace console

#endif // CONSOLE_OBJECT_H
