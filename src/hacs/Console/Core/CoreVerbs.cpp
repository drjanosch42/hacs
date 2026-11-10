// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// CoreVerbs.cpp -- see CoreVerbs.h.
// ===========================================================================

#include "CoreVerbs.h"

#include "Object.h"
#include "Printer.h"
#include "Registry.h"
#include "Session.h"
#include "Value.h"

#include <limits>

namespace console {

namespace {

Signature Sig(std::vector<Param> params, bool variadic = false) {
    Signature s;
    s.params = std::move(params);
    s.variadic = variadic;
    return s;
}

// --- printing --------------------------------------------------------------

Value VerbPrint(CallContext& ctx, const std::vector<Value>& args) {
    std::string line;
    for (std::size_t i = 0; i < args.size(); ++i) {
        if (i != 0) line += " ";
        line += PrintDisplay(args[i], ctx.session->print_max_entries());
    }
    ctx.session->print_line(line);
    return Value::nil();
}

Value VerbShow(CallContext& ctx, const std::vector<Value>& args) {
    ctx.session->print_line(PrintFull(args[0]));
    return Value::nil();
}

Value VerbDescribe(CallContext& ctx, const std::vector<Value>& args) {
    ctx.session->print_line(Describe(args[0]));
    return Value::nil();
}

Value VerbStr(CallContext& ctx, const std::vector<Value>& args) {
    return Value::from_string(PrintDisplay(args[0], ctx.session->print_max_entries()));
}

Value VerbRepr(CallContext& ctx, const std::vector<Value>& args) {
    return Value::from_string(PrintCanonical(args[0], ctx.session->print_max_entries()));
}

Value VerbJson(CallContext&, const std::vector<Value>& args) {
    return Value::from_string(ToJson(args[0]));
}

Value VerbAnnotate(CallContext& ctx, const std::vector<Value>& args) {
    // A `#~` line: excluded from every .expected comparison (plan Sec. 2 C5),
    // which is how a script says something that is true but not reproducible.
    ctx.session->print_annotation(args[0].string_value());
    return Value::nil();
}

// --- containers ------------------------------------------------------------

Value VerbLen(CallContext& ctx, const std::vector<Value>& args) {
    const Value& v = args[0];
    switch (v.kind()) {
        case ValueKind::List:   return Value::from_int(static_cast<std::int64_t>(v.list_value().size()));
        case ValueKind::String: return Value::from_int(static_cast<std::int64_t>(v.string_value().size()));
        case ValueKind::Record: return Value::from_int(static_cast<std::int64_t>(v.record_value().size()));
        default:
            throw TypeError("len() expects List, String or Record, got " + v.type_name(), ctx.pos);
    }
}

Value VerbKeys(CallContext&, const std::vector<Value>& args) {
    std::vector<Value> out;
    for (const std::string& k : args[0].record_value().keys()) {
        out.push_back(Value::from_string(k));
    }
    return Value::from_list(std::move(out));
}

Value VerbRange(CallContext& ctx, const std::vector<Value>& args) {
    const std::int64_t from = args[0].int_value();
    const std::int64_t to = args[1].int_value();
    const std::int64_t step = args.size() > 2 ? args[2].int_value() : 1;
    if (step == 0) throw RuntimeError("range() step must not be zero", ctx.pos);
    std::vector<Value> out;
    // Half-open [from, to), like every other range in this tree.
    if (step > 0) {
        for (std::int64_t i = from; i < to; i += step) {
            out.push_back(Value::from_int(i));
            if (i > std::numeric_limits<std::int64_t>::max() - step) break;
        }
    } else {
        for (std::int64_t i = from; i > to; i += step) {
            out.push_back(Value::from_int(i));
            if (i < std::numeric_limits<std::int64_t>::min() - step) break;
        }
    }
    return Value::from_list(std::move(out));
}

// --- scalars ---------------------------------------------------------------

Value VerbAbs(CallContext& ctx, const std::vector<Value>& args) {
    const std::int64_t v = args[0].int_value();
    if (v == std::numeric_limits<std::int64_t>::min()) {
        throw RuntimeError("Int overflow in abs()", ctx.pos);
    }
    return Value::from_int(v < 0 ? -v : v);
}

Value VerbMin(CallContext&, const std::vector<Value>& args) {
    const std::int64_t a = args[0].int_value();
    const std::int64_t b = args[1].int_value();
    return Value::from_int(a < b ? a : b);
}

Value VerbMax(CallContext&, const std::vector<Value>& args) {
    const std::int64_t a = args[0].int_value();
    const std::int64_t b = args[1].int_value();
    return Value::from_int(a > b ? a : b);
}

Value VerbType(CallContext&, const std::vector<Value>& args) {
    return Value::from_string(args[0].type_name());
}

// --- control ---------------------------------------------------------------

Value VerbAssert(CallContext& ctx, const std::vector<Value>& args) {
    if (!args[0].is_bool()) {
        throw TypeError("assert() expects a Bool condition, got " + args[0].type_name(), ctx.pos);
    }
    if (args[0].bool_value()) return Value::nil();
    const std::string msg = args.size() > 1 ? args[1].string_value() : std::string();
    throw AssertionFailure(msg.empty() ? std::string("assert(...)") : msg, ctx.pos);
}

Value VerbExit(CallContext&, const std::vector<Value>& args) {
    const std::int64_t code = args.empty() ? 0 : args[0].int_value();
    ExitSignal sig;
    sig.code = static_cast<int>(code);
    throw sig;
}

// --- introspection ---------------------------------------------------------

Value VerbVars(CallContext& ctx, const std::vector<Value>&) {
    Record r;
    for (const std::pair<std::string, Value>& kv : ctx.session->vars()) {
        r.set(kv.first, kv.second);
    }
    return Value::from_record(std::move(r));
}

Value VerbVerbs(CallContext& ctx, const std::vector<Value>&) {
    std::vector<Value> out;
    for (const std::string& n : ctx.registry->verb_names()) out.push_back(Value::from_string(n));
    return Value::from_list(std::move(out));
}

Value VerbTypes(CallContext& ctx, const std::vector<Value>&) {
    std::vector<Value> out;
    for (const std::string& n : ctx.registry->type_names()) out.push_back(Value::from_string(n));
    return Value::from_list(std::move(out));
}

Value VerbSettings(CallContext& ctx, const std::vector<Value>&) {
    return Value::from_record(ctx.session->settings().describe());
}

Value VerbResetSettings(CallContext& ctx, const std::vector<Value>&) {
    ctx.session->settings().reset_to_baseline();
    return Value::nil();
}

Value VerbArgs(CallContext& ctx, const std::vector<Value>&) {
    return Value::from_record(ctx.session->args());
}

// --- handles (plan Sec. 2 C2 / Sec. 4.2; owner decision R3) ----------------

Value VerbCompute(CallContext& ctx, const std::vector<Value>& args) {
    const ObjectPtr& obj = args[0].object_value();
    if (!obj) throw TypeError("compute() expects an object", ctx.pos);
    obj->ensure_computed(ctx.session->settings(), ctx.pos);
    return args[0];
}

Value VerbState(CallContext& ctx, const std::vector<Value>& args) {
    const ObjectPtr& obj = args[0].object_value();
    if (!obj) throw TypeError("state() expects an object", ctx.pos);
    Record r;
    r.set("type", Value::from_string(obj->type_name()));
    r.set("state", Value::from_string(ObjectStateName(obj->state())));
    r.set("created_at", Value::from_string(obj->creation_site().to_string()));
    r.set("created_by", Value::from_string(obj->creation_expr()));
    r.set("settings", Value::from_record(obj->snapshot().to_record()));
    // Owner decision R1: only the MATHEMATICAL settings are part of what the
    // result means, so they are reported separately -- this is the subset a
    // reproducibility record must carry into its hash input.
    r.set("mathematical_settings",
          Value::from_record(obj->snapshot().mathematical().to_record()));
    r.set("resumable", Value::from_record(obj->resumable_state().to_record()));
    if (obj->failed()) r.set("error", Value::from_string(obj->error_text()));
    return Value::from_record(std::move(r));
}

Value VerbResume(CallContext& ctx, const std::vector<Value>& args) {
    const ObjectPtr& obj = args[0].object_value();
    if (!obj) throw TypeError("resume() expects an object", ctx.pos);
    const ResumableState& rs = obj->resumable_state();
    if (!rs.resumable) {
        throw TypeError("resume(): this " + obj->type_name() +
                            " is complete -- only a computation cut off by a budget can be "
                            "resumed (owner decision R3)",
                        ctx.pos);
    }
    const TypeInfo* type = obj->type();
    if (type == nullptr || !type->resume) {
        throw TypeError("resume(): type " + obj->type_name() +
                            " records a budget cut-off but does not implement resume yet "
                            "(R3: designed now, implemented later)",
                        ctx.pos);
    }
    if (rs.budget_setting.empty()) {
        throw TypeError("resume(): this object does not name the budget setting it hit", ctx.pos);
    }
    // (C2a, slot-6 WP-13; plan Sec. 2 C5 "Resumed handles", Sec. 10 C2 (h))
    // A resume changes EXACTLY ONE key -- the budget.  If the session has
    // since moved any other MATHEMATICAL setting, continuing the old state
    // would splice two configurations into one result, so it is refused and
    // the user is told to start a new computation.  The budget key itself is
    // exempt, and so is its root when it is spelled dotted (`sturm.cap` is the
    // cap VALUE of the registered key `sturm`).  Engine knobs change no value
    // and are not compared.
    {
        const std::string& budget = rs.budget_setting;
        const std::string root = budget.substr(0, budget.find('.'));
        const SettingsSnapshot created = obj->snapshot().mathematical();
        const SettingsSnapshot now = ctx.session->settings().snapshot().mathematical();
        auto text = [](const Value& v) -> std::string {
            if (v.is_string()) return v.string_value();
            if (v.is_int()) return std::to_string(v.int_value());
            if (v.is_bool()) return v.bool_value() ? "true" : "false";
            return v.type_name();
        };
        for (const std::pair<std::string, Value>& kv : created.values()) {
            if (kv.first == budget || kv.first == root) continue;
            const Value* cur = now.find(kv.first);
            if (cur == nullptr || cur->equals(kv.second)) continue;
            throw TypeError("resume under a different setting -> start a new computation: `" +
                                kv.first + "` was '" + text(kv.second) +
                                "' when this handle was created and is '" + text(*cur) +
                                "' now (a resume may change the budget `" + budget +
                                "` and nothing else)",
                            ctx.pos);
        }
    }
    // R3: "a resume with a changed budget is a NEW SETTING and must be
    // recorded" -- so the resumed handle gets a snapshot that differs from its
    // parent's in exactly that key, and keeps a link to the parent.
    SettingsSnapshot snap = obj->snapshot();
    if (args.size() > 1 && !args[1].is_nil()) {
        snap = snap.with(rs.budget_setting, args[1]);
    }
    // (C5 leftovers, slot-10 WP-12; [P] v3.8 Sec. 2 C7 site 2.)  The type's
    // resume hook refuses with a ConsoleError that carries no position of its
    // own; back-fill the call's file:line:col.  Any other exception is wrapped,
    // with the position, by the verb call's own try (Evaluator::eval_call).
    ObjectPtr resumed;
    try {
        resumed = type->resume(obj, snap);
    } catch (ConsoleError& ce) {
        ce.set_pos_if_unknown(ctx.pos);
        throw;
    }
    if (!resumed) {
        throw RuntimeError("resume(): " + obj->type_name() + " produced no continuation", ctx.pos);
    }
    resumed->set_resumed_from(obj);
    return Value::from_object(resumed);
}

std::vector<std::string>& MutableCoreVerbNames() {
    static std::vector<std::string> names;
    return names;
}

} // namespace

void RegisterCoreVerbs(Registry& registry) {
    std::vector<std::string>& names = MutableCoreVerbNames();
    names.clear();

    auto add = [&registry, &names](const std::string& name, Signature sig, VerbImpl impl,
                                   const char* doc) {
        registry.register_verb(name, std::move(sig), std::move(impl), doc, "core");
        bool seen = false;
        for (const std::string& n : names) {
            if (n == name) { seen = true; break; }
        }
        if (!seen) names.push_back(name);
    };

    // printing
    add("print", Sig({Param::any("value")}, /*variadic=*/true), &VerbPrint,
        "prints its arguments on one line, separated by spaces (a String prints bare)");
    add("show", Sig({Param::any("value")}), &VerbShow,
        "prints the full canonical form, ignoring print_max_entries");
    add("describe", Sig({Param::any("value")}), &VerbDescribe,
        "prints type, state, settings snapshot, creation site and members (same as ?x)");
    add("str", Sig({Param::any("value")}), &VerbStr, "the display form as a String");
    add("repr", Sig({Param::any("value")}), &VerbRepr,
        "the canonical form as a String (strings quoted)");
    add("json", Sig({Param::any("value")}), &VerbJson,
        "JSON for Int / Bool / String / List / Record, with a proper escaper");
    add("annotate", Sig({Param::of("text", ValueKind::String)}), &VerbAnnotate,
        "prints a '#~ ' line, which every transcript comparison ignores");

    // containers
    add("len", Sig({Param::any("value")}), &VerbLen, "length of a List, String or Record");
    add("keys", Sig({Param::of("record", ValueKind::Record)}), &VerbKeys,
        "the keys of a Record, in order");
    add("range", Sig({Param::of("from", ValueKind::Int), Param::of("to", ValueKind::Int)}),
        &VerbRange, "the half-open range [from, to) as a List of Int");
    add("range",
        Sig({Param::of("from", ValueKind::Int), Param::of("to", ValueKind::Int),
             Param::of("step", ValueKind::Int)}),
        &VerbRange, "the half-open range [from, to) with the given step");

    // scalars
    add("abs", Sig({Param::of("value", ValueKind::Int)}), &VerbAbs, "absolute value");
    add("min", Sig({Param::of("a", ValueKind::Int), Param::of("b", ValueKind::Int)}), &VerbMin,
        "the smaller of two Ints");
    add("max", Sig({Param::of("a", ValueKind::Int), Param::of("b", ValueKind::Int)}), &VerbMax,
        "the larger of two Ints");
    add("type", Sig({Param::any("value")}), &VerbType, "the type name as a String");

    // control
    add("assert", Sig({Param::of("condition", ValueKind::Bool)}), &VerbAssert,
        "fails the script (exit code 4) when the condition is false");
    add("assert",
        Sig({Param::of("condition", ValueKind::Bool), Param::of("message", ValueKind::String)}),
        &VerbAssert, "fails the script (exit code 4) with the given message");
    add("exit", Sig({}), &VerbExit, "ends the script with exit code 0");
    add("exit", Sig({Param::of("code", ValueKind::Int)}), &VerbExit,
        "ends the script with the given exit code");

    // introspection
    add("vars", Sig({}), &VerbVars, "every bound variable as a Record");
    add("verbs", Sig({}), &VerbVerbs, "every registered verb name");
    add("types", Sig({}), &VerbTypes, "every registered type name");
    add("settings", Sig({}), &VerbSettings,
        "every setting with its value, source (session / env) and family (engine / mathematical)");
    add("reset_settings", Sig({}), &VerbResetSettings,
        "re-applies the settings snapshot taken when the session started");
    add("script_args", Sig({}), &VerbArgs, "the script parameters, same as `args`");

    // handles
    add("compute", Sig({Param::of("object", ValueKind::Object)}), &VerbCompute,
        "forces a lazily declared handle and returns it");
    add("state", Sig({Param::of("object", ValueKind::Object)}), &VerbState,
        "state, creation site, settings snapshot and resumable state of a handle");
    add("resume", Sig({Param::of("object", ValueKind::Object)}), &VerbResume,
        "continues a computation that a budget cut off, with the same budget");
    add("resume",
        Sig({Param::of("object", ValueKind::Object), Param::any("budget").optional(Value::nil())}),
        &VerbResume, "continues a budget-cut computation with a new budget, recorded in the "
                     "resumed handle's settings snapshot");
}

const std::vector<std::string>& CoreVerbNames() {
    return MutableCoreVerbNames();
}

const std::vector<std::string>& VerbsForbiddenInC0() {
    // The transcript-freezing guard (CoreVerbs.h): none of these may exist in a
    // C0-only registry.  `hash` and `golden` are the pins themselves; the four
    // pipeline verbs are what a pin would be taken of.
    static const std::vector<std::string> names = {
        "hash", "golden", "cohomology", "hecke", "decompose", "report"
    };
    return names;
}

} // namespace console
