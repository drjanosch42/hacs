// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Reproducibility.cpp -- console phase C1a: `hash`, `golden`, the mathematical
// fingerprint, `save_state`, and the C7 exception seam.
//
// Plan (docs/console/00-console-interface-plan.md v3) Sec. 2 C5 and Sec. 10
// "Phase C1a".  This is the TU that makes a console session a reproducibility
// instrument rather than a calculator:
//
//   hash(x)             the per-kind FNV-1a-64 of Golden/GoldenHarness.cpp,
//                       over the ONE src/ModArithmetic/Fnv.h.
//   golden(point, obj)  the READ-ONLY lookup into GoldenReference.inc through
//                       Golden/GoldenHarness.h's GoldenLookup (gap G13).  It
//                       computes nothing and pins nothing: a console pin is a
//                       comparison against the EXISTING reference, never a new
//                       one, which is why C1a cannot move a golden value.
//   fingerprint()       hecke::math_settings().fingerprint() -- 0 at the
//                       defaults.  Rule R19 (plan Sec. 6): a hash may only be
//                       pinned at the default mathematical configuration, so
//                       `golden(...)` REFUSES at a non-zero fingerprint rather
//                       than hand back a number that means something else.
//   save_state(x[,path])  R3.1 item 6's object (plan Sec. 2 C5, "Resumed
//                       handles"): inline when its JSON is <= 64 KiB, else a
//                       sidecar referenced by relative path AND its Fnv.h
//                       FNV-1a-64.  Named save_state, NOT save, because
//                       Sec. 4.3 already gives save(x, path) the meaning
//                       "write the canonical printed form".
//
// The C7 catch order is installed here as the registry's two seams, so that
// Console/Core/ still names no pipeline type (plan Sec. 2 C8): a translator
// that NAMES the C++ class, and a classifier that decides which console error
// kind -- and therefore which EXIT CODE -- a recognised exception becomes.
// ===========================================================================

#include "Bindings.h"
#include "Payloads.h"   // (C5a) HandleRingK

#include "../Core/Printer.h"
#include "../Core/ScriptRunner.h"

#include "../../Golden/GoldenHarness.h"
#include "../../Hecke/ExactInt.h"
#include "../../Hecke/HeckeSettings.h"
#include "../../../ModArithmetic/Fnv.h"
#include "../../Util/TextIo.h"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace console {
namespace bind {

namespace {

using oc_util::HexU64;   // S24 CL-7 (F11): Util/TextIo.h
// --- hash ------------------------------------------------------------------

uint64_t HashOfValue(const Value& v, const SourcePos& pos) {
    switch (v.kind()) {
        case ValueKind::Int:
            // A scalar invariant: FnvU64(offset, value) -- Collector::scalar.
            if (v.int_value() < 0) {
                throw TypeError("hash(x): a negative Int is not a golden scalar invariant", pos);
            }
            return HashScalarValue(static_cast<uint64_t>(v.int_value()));
        case ValueKind::String:
            // A text object: FnvBytes(offset, s) -- Collector::text.  Note that
            // the harness normalises JSON whitespace BEFORE hashing, so a raw
            // `hash(T.json)` is not the `op_json_full` entry; the console does
            // not claim it is.
            return engine::FnvBytes(engine::kFnvOffset64, v.string_value());
        case ValueKind::Object: {
            const ObjectPtr& o = v.object_value();
            if (!o || o->type() == nullptr || !o->type()->hash) {
                throw TypeError("hash(x): type " + v.type_name() + " has no golden hash", pos);
            }
            return o->type()->hash(o);
        }
        case ValueKind::List: {
            // A basis: FnvU64(offset, count), then each matrix -- Collector::basis.
            const std::vector<Value>& items = v.list_value();
            // (C5a, slot-9 WP-12) A non-empty list of non-negative Ints: the
            // golden LIST recipe of [E] sec:ring (e) -- its length, then each
            // entry as a uint64 (GoldenHarness.cpp HashU32List), which is what
            // `invariant_factors_H^q` hashes.  The empty list needs no branch:
            // both recipes hash it as FnvU64(offset, 0).
            bool all_int = !items.empty();
            for (const Value& item : items) all_int = all_int && item.is_int() && item.int_value() >= 0;
            if (all_int) {
                uint64_t h = engine::FnvU64(engine::kFnvOffset64, items.size());
                for (const Value& item : items) h = engine::FnvU64(h, static_cast<uint64_t>(item.int_value()));
                return h;
            }
            std::vector<FlatMatrixT<uint32_t>> ms;
            ms.reserve(items.size());
            for (const Value& item : items) {
                ms.push_back(AsMatrix(item, pos, "hash(list)").m);
            }
            return HashMatrixList(ms);
        }
        default:
            throw TypeError("hash(x): " + v.type_name() + " has no golden hash", pos);
    }
}

Value VerbHash(CallContext& ctx, const std::vector<Value>& a) {
    const uint64_t h = HashOfValue(a[0], ctx.pos);
    // Int is int64_t, so a hash with the top bit set would not round-trip; it
    // is handed back as its canonical 0x... text, which is what a transcript
    // and a `==` comparison both want.
    return Value::from_string(HexU64(h));
}

Value VerbHashU64(CallContext& ctx, const std::vector<Value>& a) {
    const uint64_t h = HashOfValue(a[0], ctx.pos);
    if (h > static_cast<uint64_t>(9223372036854775807ull)) {
        throw RuntimeError("hash_int(x): this hash does not fit an Int (use hash(x), which "
                           "returns the canonical 0x... text)",
                           ctx.pos);
    }
    return Value::from_int(static_cast<std::int64_t>(h));
}

// --- golden ----------------------------------------------------------------

Value VerbGolden(CallContext& ctx, const std::vector<Value>& a) {
    // Rule R19 (plan Sec. 6 / Sec. 2 C4 rule 1): the golden objects pin the
    // DEFAULT mathematical configuration and nothing else, so a pin taken under
    // a non-default setting is refused rather than silently compared.
    const uint64_t fp = hecke::math_settings().fingerprint();
    if (fp != 0) {
        throw RuntimeError(
            "golden(point, object): the mathematical settings are not at their defaults "
            "(fingerprint " + HexU64(fp) +
                "); the golden reference pins the default configuration only",
            ctx.pos);
    }
    const std::string point = a[0].string_value();
    const std::string object = a[1].string_value();
    const GoldenEntry* e = GoldenLookup(point, object);
    if (e == nullptr) {
        throw RuntimeError("golden(point, object): '" + point + "' / '" + object +
                               "' is not in GoldenReference.inc",
                           ctx.pos);
    }
    Record r;
    r.set("point", Value::from_string(point));
    r.set("object", Value::from_string(object));
    r.set("tier", Value::from_string(std::string(1, e->tier)));
    r.set("rows", Value::from_int(static_cast<std::int64_t>(e->rows)));
    r.set("cols", Value::from_int(static_cast<std::int64_t>(e->cols)));
    // For a scalar invariant (cols == 0) `rows` IS the value -- the harness's
    // own convention (GoldenHarness.h).  It is exposed under both names so a
    // script reads what it means.
    r.set("value", Value::from_int(static_cast<std::int64_t>(e->rows)));
    r.set("scalar", Value::from_bool(e->cols == 0));
    r.set("hash", Value::from_string(HexU64(e->hash)));
    return Value::from_record(std::move(r));
}

Value VerbGoldenHas(CallContext& ctx, const std::vector<Value>& a) {
    (void)ctx;
    return Value::from_bool(GoldenLookup(a[0].string_value(), a[1].string_value()) != nullptr);
}

Value VerbGoldenSize(CallContext& ctx, const std::vector<Value>&) {
    (void)ctx;
    return Value::from_int(static_cast<std::int64_t>(GoldenReferenceSize()));
}

// --- the mathematical settings (READ-ONLY in C1a) --------------------------

Value VerbFingerprint(CallContext& ctx, const std::vector<Value>&) {
    (void)ctx;
    return Value::from_string(HexU64(hecke::math_settings().fingerprint()));
}

Value VerbMathSettings(CallContext& ctx, const std::vector<Value>&) {
    (void)ctx;
    const hecke::MathSettings& s = hecke::math_settings();
    Record r;
    r.set("locality", Value::from_string(s.locality.to_text()));
    r.set("locality_source", Value::from_string(hecke::to_string(s.locality_source)));
    r.set("is_default", Value::from_bool(s.is_default()));
    r.set("fingerprint", Value::from_string(HexU64(s.fingerprint())));
    r.set("canonical_text", Value::from_string(s.canonical_text()));
    std::vector<Value> keys;
    for (const std::string& k : hecke::MathSettings::keys()) keys.push_back(Value::from_string(k));
    r.set("keys", Value::from_list(std::move(keys)));
    return Value::from_record(std::move(r));
}

// --- save_state (R3.1 item 6) ----------------------------------------------

constexpr std::size_t kInlineStateLimit = 64u * 1024u;

Record StateObject(const ObjectPtr& o, bool& has_state) {
    const ResumableState& rs = o->resumable_state();
    has_state = rs.resumable || !rs.budget_setting.empty();
    Record out;
    const Value* kind = rs.progress.find("kind");
    out.set("kind", kind != nullptr && kind->is_string() ? *kind
                                                         : Value::from_string(rs.budget_setting));
    out.set("complete", Value::from_bool(!rs.resumable));
    out.set("budget", rs.budget_value);
    const Value* reached = rs.progress.find("reached");
    out.set("reached", reached != nullptr ? *reached : Value::nil());
    const Value* hist = rs.progress.find("budget_history");
    out.set("budget_history", hist != nullptr ? *hist : Value::from_list({}));
    if (!rs.reason.empty()) out.set("reason", Value::from_string(rs.reason));
    out.set("state", Value::from_record(rs.progress));
    return out;
}

Value VerbSaveState(CallContext& ctx, const std::vector<Value>& a) {
    const Value& v = a[0];
    if (!v.is_object() || !v.object_value()) {
        throw TypeError("save_state(x): expected a handle, got " + v.type_name(), ctx.pos);
    }
    const ObjectPtr& o = v.object_value();
    // (C5a, slot-9 WP-12; [E] sec:ring (h)) A Z/ell^k Cohomology's ladic_level
    // state exists once the handle has computed, and `save_state(H)` returns
    // it: so the verb forces a DECLARED ring Cohomology first, under its own
    // snapshot.  Every other type is untouched.
    if (o->type() != nullptr && o->type()->name == kTypeCohomology && HandleRingK(o) >= 2) {
        o->ensure_computed(ctx.session->settings(), ctx.pos);
    }
    bool has_state = false;
    Record obj = StateObject(o, has_state);
    if (!has_state) {
        throw TypeError("save_state(x): " + o->type_name() +
                            " carries no resumable state (only a budgeted computation does)",
                        ctx.pos);
    }
    const std::string state_json = ToJson(*obj.find("state"));

    if (a.size() < 2 || a[1].is_nil()) {
        // No stem: the state is inlined regardless of size and FLAGGED, which
        // is the shipped behaviour of HeckeSturmBound ([WP3]), R3 instance 1.
        if (state_json.size() > kInlineStateLimit) {
            obj.set("state_oversize", Value::from_bool(true));
        }
        return Value::from_string(ToJson(Value::from_record(obj)));
    }
    if (!a[1].is_string()) throw TypeError("save_state(x, path): path must be a String", ctx.pos);
    const std::string path = a[1].string_value();

    Record written;
    if (state_json.size() > kInlineStateLimit) {
        // A sidecar, referenced by RELATIVE path and by its FNV-1a-64.
        const Value* kind = obj.find("kind");
        const std::string kind_text = (kind != nullptr && kind->is_string()) ? kind->string_value()
                                                                            : std::string("state");
        const std::size_t dot = path.find_last_of('.');
        const std::string stem = (dot == std::string::npos) ? path : path.substr(0, dot);
        const std::string sidecar = stem + ".state." + kind_text + ".json";
        std::ofstream side(sidecar.c_str());
        if (!side) throw RuntimeError("save_state: cannot write '" + sidecar + "'", ctx.pos);
        side << state_json;
        side.close();
        const std::size_t slash = sidecar.find_last_of('/');
        const std::string rel = (slash == std::string::npos) ? sidecar : sidecar.substr(slash + 1);
        Record ref;
        ref.set("path", Value::from_string(rel));
        ref.set("fnv64", Value::from_string(HexU64(engine::FnvBytes(engine::kFnvOffset64, state_json))));
        obj.set("state", Value::nil());
        obj.set("state_ref", Value::from_record(std::move(ref)));
        written.set("sidecar", Value::from_string(sidecar));
    }
    std::ofstream out(path.c_str());
    if (!out) throw RuntimeError("save_state: cannot write '" + path + "'", ctx.pos);
    const std::string text = ToJson(Value::from_record(obj));
    out << text;
    if (!out) throw RuntimeError("save_state: write failure on '" + path + "'", ctx.pos);
    written.set("path", Value::from_string(path));
    written.set("bytes", Value::from_int(static_cast<std::int64_t>(text.size())));
    written.set("inline_state", Value::from_bool(state_json.size() <= kInlineStateLimit));
    return Value::from_record(std::move(written));
}

} // namespace

// ===========================================================================
// The C7 exception seam (plan Sec. 2 C7 / Sec. 8).
// ===========================================================================

void InstallExceptionSeam(Registry& registry) {
    // The catch ORDER is what matters: HeckeOverflow and NotImplementedInThisBuild
    // both DERIVE from std::runtime_error, so they must be tested before it or
    // they would never be named.
    registry.set_exception_translator([](const std::exception& e, std::string& class_name) {
        if (dynamic_cast<const hecke::HeckeOverflow*>(&e) != nullptr) {
            class_name = "hecke::HeckeOverflow";
            return true;
        }
        if (dynamic_cast<const hecke::NotImplementedInThisBuild*>(&e) != nullptr) {
            class_name = "hecke::NotImplementedInThisBuild";
            return true;
        }
        if (dynamic_cast<const std::logic_error*>(&e) != nullptr) {
            // Every ENGINE_CHECK / CONSOLE_CHECK; its message already carries
            // file:line and the stringified condition (EngineCheck.h:40-56).
            class_name = "std::logic_error";
            return true;
        }
        if (dynamic_cast<const std::invalid_argument*>(&e) != nullptr) {
            class_name = "std::invalid_argument";
            return true;
        }
        if (dynamic_cast<const std::out_of_range*>(&e) != nullptr) {
            class_name = "std::out_of_range";
            return true;
        }
        if (dynamic_cast<const std::runtime_error*>(&e) != nullptr) {
            class_name = "std::runtime_error";
            return true;
        }
        return false;
    });

    registry.set_exception_kind([](const std::exception& e) -> std::string {
        // "not_implemented_in_this_build is its own outcome, not a bad value"
        // (plan Sec. 2 C7): exit code 5, never 1 and never 3.  Both the typed
        // test and the literal tag are accepted, because the tag is the
        // contract HeckeSettings.h publishes for callers that cannot see the
        // type (hecke::kNotImplementedInThisBuildTag).
        if (dynamic_cast<const hecke::NotImplementedInThisBuild*>(&e) != nullptr) {
            return kNotImplementedKind;
        }
        const char* tag = hecke::kNotImplementedInThisBuildTag;
        const std::string what = e.what() == nullptr ? std::string() : std::string(e.what());
        if (tag != nullptr && what.rfind(tag, 0) == 0) return kNotImplementedKind;
        return std::string();
    });
}

void RegisterReproducibilityBindings(Registry& registry) {
    const char* g = "reproducibility";
    auto add = [&registry, g](const char* name, Signature sig, VerbImpl impl, const char* doc) {
        registry.register_verb(name, std::move(sig), std::move(impl), doc, g);
    };

    add("hash", Sig({Param::any("x")}), &VerbHash,
        "the per-kind FNV-1a-64 of the golden harness, as canonical 0x... text");
    add("hash_int", Sig({Param::any("x")}), &VerbHashU64,
        "the same hash as an Int, when it fits one");
    add("golden",
        Sig({Param::of("point", ValueKind::String), Param::of("object", ValueKind::String)}),
        &VerbGolden,
        "the frozen GoldenReference.inc entry {tier, rows, cols, value, hash} -- READ-ONLY");
    add("golden_has",
        Sig({Param::of("point", ValueKind::String), Param::of("object", ValueKind::String)}),
        &VerbGoldenHas, "whether the reference has that (point, object)");
    add("golden_size", Sig({}), &VerbGoldenSize, "the number of entries in the reference");
    add("fingerprint", Sig({}), &VerbFingerprint,
        "the mathematical-settings fingerprint (0x0... at the defaults)");
    add("math_settings", Sig({}), &VerbMathSettings,
        "the mathematical settings in force, with their sources and the fingerprint (read-only "
        "in C1a; `set` on them is phase C2)");
    add("save_state", Sig({Param::any("x")}), &VerbSaveState,
        "the R3 resumable-state object of a handle, as JSON text");
    add("save_state", Sig({Param::any("x"), Param::of("path", ValueKind::String)}), &VerbSaveState,
        "the same, written to a file (a state over 64 KiB goes to a sidecar referenced by "
        "relative path and FNV-1a-64)");
}

} // namespace bind
} // namespace console
