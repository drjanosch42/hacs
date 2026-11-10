// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Bindings.cpp -- the helpers of Bindings.h (console phase C1a).
// ===========================================================================

#include "Bindings.h"

#include "../../../ModArithmetic/Fnv.h"
#include "../../Hecke/HeckeSettings.h"

#include <exception>
#include <limits>
#include <stdexcept>
#include <set>
#include <string>

namespace console {
namespace bind {

Signature Sig(std::vector<Param> params, bool variadic) {
    Signature s;
    s.params = std::move(params);
    s.variadic = variadic;
    return s;
}

const TypeInfo* TypeByName(CallContext& ctx, const char* name) {
    CONSOLE_CHECK(ctx.registry != nullptr);
    const TypeInfo* t = ctx.registry->find_type(name);
    CONSOLE_CHECK_MSG(t != nullptr, "a binding asked for a type its own registration did not add");
    return t;
}

ObjectPtr MakeValueObject(CallContext& ctx, const char* type_name,
                          std::shared_ptr<void> payload) {
    CONSOLE_CHECK(ctx.session != nullptr);
    ObjectPtr obj = Object::create(TypeByName(ctx, type_name), ctx.pos, ctx.call_text,
                                   ctx.session->settings().snapshot(), std::move(payload),
                                   // A value-like object has no lazy work: it is
                                   // born computed, so `?x` never says `declared`
                                   // about a matrix that is already in hand.
                                   [](Object&) {});
    obj->ensure_computed(ctx.session->settings(), ctx.pos);
    return obj;
}

ObjectPtr MakeValueObjectFrom(const TypeInfo* type, const ObjectPtr& owner,
                              std::shared_ptr<void> payload) {
    CONSOLE_CHECK(type != nullptr && owner != nullptr);
    ObjectPtr obj = Object::create(type, owner->creation_site(), owner->creation_expr(),
                                   owner->snapshot(), std::move(payload), [](Object&) {});
    // A value object has no lazy work, so it is born computed.  A throwaway
    // Settings is enough to run the empty compute: install() ignores names it
    // does not know, so the snapshot round-trip is a no-op on it.
    Settings scratch;
    obj->ensure_computed(scratch, owner->creation_site());
    return obj;
}

Object::ComputeFn LazyCompute(Object::ComputeFn body) {
    CONSOLE_CHECK(static_cast<bool>(body));
    return [body = std::move(body)](Object& self) {
        try {
            body(self);
        } catch (const ConsoleError&) {
            throw;
        } catch (const hecke::NotImplementedInThisBuild& e) {
            throw NotImplementedError(std::string("hecke::NotImplementedInThisBuild: ") + e.what(),
                                      SourcePos());
        } catch (const std::exception& e) {
            const char* tag = hecke::kNotImplementedInThisBuildTag;
            const char* what = e.what();
            if (tag != nullptr && what != nullptr && std::string(what).rfind(tag, 0) == 0) {
                throw NotImplementedError(what, SourcePos());
            }
            throw;
        }
    };
}

namespace {

// The core's member access (Evaluator::object_member) -- unlike a method call
// -- does not wrap a C++ exception raised by a getter, so a raw pipeline
// exception would escape every `expect_error` and always exit 3.  A gated
// member therefore converts it HERE, with the kinds of the C7 seam: a
// hecke::NotImplementedInThisBuild keeps exit 5, anything else is a runtime
// error (exit 3) naming the C++ class.  The member getter receives no source
// position, so the error carries none of its own; since slot-10 WP-12 the
// core's Evaluator::object_member back-fills the member access's file:line:col
// ([P] v3.8 Sec. 2 C7 site 1).  (That core wrap would now also classify a raw
// exception by itself; this conversion is kept so the error texts and codes
// of a gated member do not move.)
Value GuardedMember(const std::function<Value(const ObjectPtr&)>& get, const ObjectPtr& o,
                    const std::string& who) {
    try {
        return get(o);
    } catch (const ConsoleError&) {
        throw;
    } catch (const hecke::NotImplementedInThisBuild& e) {
        NotImplementedError err(who + ": not implemented in this build", SourcePos());
        err.add_note(std::string("caused by: hecke::NotImplementedInThisBuild: ") + e.what());
        throw err;
    } catch (const std::logic_error& e) {
        RuntimeError err(who + " failed", SourcePos());
        err.add_note(std::string("caused by: std::logic_error: ") + e.what());
        throw err;
    } catch (const std::exception& e) {
        const char* tag = hecke::kNotImplementedInThisBuildTag;
        if (tag != nullptr && std::string(e.what()).rfind(tag, 0) == 0) {
            NotImplementedError err(who + ": not implemented in this build", SourcePos());
            err.add_note(std::string("caused by: ") + e.what());
            throw err;
        }
        RuntimeError err(who + " failed", SourcePos());
        err.add_note(std::string("caused by: ") + e.what());
        throw err;
    }
}

} // namespace

void GateTypeOnRing(TypeInfo& t, const std::function<RingRule(const std::string&)>& rule,
                    const std::function<uint32_t(const ObjectPtr&)>& ring_k) {
    for (TypeInfo::Member& m : t.members) {
        auto inner = m.get;
        const std::string name = t.name + "." + m.name;
        if (rule(m.name) != RingRule::Refuse) {
            m.get = [inner, name](const ObjectPtr& o) { return GuardedMember(inner, o, name); };
            continue;
        }
        const std::string who = "console " + name;
        m.get = [inner, who, name, ring_k](const ObjectPtr& o) {
            return GuardedMember(
                [&](const ObjectPtr& x) {
                    const uint32_t k = ring_k(x);
                    if (k >= 2) {
                        hecke::RefuseCoefficientRing(who, k,
                                                     " -- the member is not on [E] sec:ring (h)'s list");
                    }
                    return inner(x);
                },
                o, name);
        };
    }
    for (TypeInfo::Method& m : t.methods) {
        if (rule(m.name) != RingRule::Refuse) continue;
        auto inner = m.call;
        const std::string who = "console " + t.name + "." + m.name + "(...)";
        m.call = [inner, who, ring_k](const ObjectPtr& o, const std::vector<Value>& args,
                                      const SourcePos& pos) {
            const uint32_t k = ring_k(o);
            if (k >= 2) {
                hecke::RefuseCoefficientRing(who, k, " -- the member is not on [E] sec:ring (h)'s list");
            }
            return inner(o, args, pos);
        };
    }
}

void GateTypeOnDigit(
    TypeInfo& t, const std::set<std::string>& pass_through, const std::set<std::string>& field,
    const std::map<std::string, std::function<Value(const ObjectPtr&)>>& members,
    const std::map<std::string, std::function<Value(const ObjectPtr&, const std::vector<Value>&, const SourcePos&)>>&
        methods,
    const std::function<bool(const ObjectPtr&)>& is_digit, const std::function<uint32_t(const ObjectPtr&)>& ring_k) {
    const std::string why = " at a digit cell (ell^k >= 2^64) -- the member is not on [E] v1.12.0-pre.2 sec:ring (h)'s "
                            "digit list";
    // The field accessor's own error text (EquivariantCohomologyOnK::require_field),
    // so a digit handle's field member is exit 3 exactly as a native ring handle's.
    auto field_error = [](const std::string& what, uint32_t k) {
        return std::logic_error(what + ": a field (F_ell) accessor, but this object is over " +
                                hecke::CoefficientRingText(k) +
                                " -- over Z/ell^k there is no dim; use lengths / invariant_factors "
                                "([E] sec:ring (b3), (h))");
    };
    for (TypeInfo::Member& m : t.members) {
        if (pass_through.count(m.name) != 0) continue;
        auto inner = m.get;
        const std::string name = t.name + "." + m.name;
        const std::string who = "console " + name;
        auto it = members.find(m.name);
        std::function<Value(const ObjectPtr&)> digit;
        if (it != members.end()) digit = it->second;
        const bool is_field = field.count(m.name) != 0;
        m.get = [inner, digit, name, who, why, is_digit, ring_k, is_field, field_error](const ObjectPtr& o) {
            if (!is_digit(o)) return inner(o);
            return GuardedMember(
                [&](const ObjectPtr& x) {
                    if (is_field) throw field_error(name, ring_k(x));
                    if (!digit) hecke::RefuseCoefficientRing(who, ring_k(x), why);
                    return digit(x);
                },
                o, name);
        };
    }
    for (TypeInfo::Method& m : t.methods) {
        if (pass_through.count(m.name) != 0) continue;
        auto inner = m.call;
        const std::string who = "console " + t.name + "." + m.name + "(...)";
        auto it = methods.find(m.name);
        std::function<Value(const ObjectPtr&, const std::vector<Value>&, const SourcePos&)> digit;
        if (it != methods.end()) digit = it->second;
        const bool is_field = field.count(m.name) != 0;
        const std::string fname = t.name + "." + m.name + "(q)";
        m.call = [inner, digit, who, why, is_digit, ring_k, is_field, fname, field_error](
                     const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
            if (!is_digit(o)) return inner(o, args, pos);
            if (is_field) throw field_error(fname, ring_k(o));
            if (!digit) hecke::RefuseCoefficientRing(who, ring_k(o), why);
            return digit(o, args, pos);
        };
    }
}

void GateTypeOnWide(
    TypeInfo& t, const std::map<std::string, std::function<Value(const ObjectPtr&)>>& members,
    const std::map<std::string, std::function<Value(const ObjectPtr&, const std::vector<Value>&, const SourcePos&)>>&
        methods,
    const std::function<bool(const ObjectPtr&)>& is_wide, const std::function<uint32_t(const ObjectPtr&)>& ring_k) {
    // The digit gate's mechanics with an empty field set and every member not
    // given here passed through: no "digit cell" refusal text can arise.
    std::set<std::string> pass_through;
    for (const TypeInfo::Member& m : t.members)
        if (members.count(m.name) == 0) pass_through.insert(m.name);
    for (const TypeInfo::Method& m : t.methods)
        if (methods.count(m.name) == 0) pass_through.insert(m.name);
    GateTypeOnDigit(t, pass_through, {}, members, methods, is_wide, ring_k);
}

void GateVerbsOnField(Registry& registry, const std::vector<std::string>& names) {
    const std::set<std::string> gated(names.begin(), names.end());
    std::set<std::string> seen;
    registry.transform_overloads([&](const VerbEntry& e, Overload& o) {
        if (gated.count(e.name) == 0) return true;
        seen.insert(e.name);
        const std::string who = "console verb " + e.name;
        o.impl = [inner = o.impl, who](CallContext& ctx, const std::vector<Value>& a) {
            // CoefficientRingK(who) itself refuses Z, naming `who`.
            const uint32_t k = hecke::CoefficientRingK(who);
            if (k != 1)
                hecke::RefuseCoefficientRing(who, k, " (the verb computes over F_ell only; [E] sec:ring (h))");
            return inner(ctx, a);
        };
        return true;
    });
    for (const std::string& n : gated)
        if (seen.count(n) == 0)
            throw std::logic_error("GateVerbsOnField: verb '" + n + "' is not registered");
}

bool IsWideCell(uint64_t ell, uint32_t k) {
    if (k < 2 || ell < 2) return false;
    unsigned __int128 acc = 1;
    for (uint32_t i = 0; i < k; ++i) {
        acc *= ell;
        if (acc >> 64) return false;   // ell^k >= 2^64: the digit route
    }
    return acc > 0xffffffffull;
}

Value WideIntValue(uint64_t x) {
    if (x <= static_cast<uint64_t>(std::numeric_limits<std::int64_t>::max()))
        return Value::from_int(static_cast<std::int64_t>(x));
    return Value::from_string(std::to_string(x));
}

const ObjectPtr& AsObjectOf(const Value& v, const char* type_name, const SourcePos& pos,
                            const char* who) {
    if (!v.is_object() || !v.object_value() || v.object_value()->type() == nullptr ||
        v.object_value()->type()->name != type_name) {
        throw TypeError(std::string(who) + ": expected a " + type_name + ", got " + v.type_name(),
                        pos);
    }
    return v.object_value();
}

const MatrixVal& AsMatrix(const Value& v, const SourcePos& pos, const char* who) {
    const ObjectPtr& o = AsObjectOf(v, kTypeMatrix, pos, who);
    return *o->payload_as<MatrixVal>();
}

const PolyVal& AsPoly(const Value& v, const SourcePos& pos, const char* who) {
    const ObjectPtr& o = AsObjectOf(v, kTypePoly, pos, who);
    return *o->payload_as<PolyVal>();
}

const FpVal& AsFp(const Value& v, const SourcePos& pos, const char* who) {
    const ObjectPtr& o = AsObjectOf(v, kTypeFp, pos, who);
    return *o->payload_as<FpVal>();
}

Value MatrixValue(CallContext& ctx, FlatMatrixT<uint32_t> m) {
    auto p = std::make_shared<MatrixVal>();
    p->m = std::move(m);
    return Value::from_object(MakeValueObject(ctx, kTypeMatrix, std::move(p)));
}

Value PolyValue(CallContext& ctx, hecke::FpPolyT<uint32_t> f) {
    auto p = std::make_shared<PolyVal>();
    p->f = std::move(f);
    return Value::from_object(MakeValueObject(ctx, kTypePoly, std::move(p)));
}

Value FpValue(CallContext& ctx, uint32_t prime, uint32_t a) {
    auto p = std::make_shared<FpVal>();
    p->p = prime;
    p->a = a % prime;
    return Value::from_object(MakeValueObject(ctx, kTypeFp, std::move(p)));
}

Value IntList(const std::vector<uint32_t>& xs) {
    std::vector<Value> out;
    out.reserve(xs.size());
    for (uint32_t x : xs) out.push_back(Value::from_int(static_cast<std::int64_t>(x)));
    return Value::from_list(std::move(out));
}

uint32_t U32Arg(const Value& v, const SourcePos& pos, const char* who) {
    if (!v.is_int()) {
        throw TypeError(std::string(who) + ": expected Int, got " + v.type_name(), pos);
    }
    const std::int64_t x = v.int_value();
    if (x < 0 || x > static_cast<std::int64_t>(std::numeric_limits<uint32_t>::max())) {
        throw TypeError(std::string(who) + ": " + std::to_string(x) +
                            " is out of range (0 .. 2^32-1)",
                        pos);
    }
    return static_cast<uint32_t>(x);
}

uint64_t U64Arg(const Value& v, const SourcePos& pos, const char* who) {
    if (!v.is_int()) {
        throw TypeError(std::string(who) + ": expected Int, got " + v.type_name(), pos);
    }
    if (v.int_value() < 0) {
        throw TypeError(std::string(who) + ": expected a non-negative Int, got " +
                            std::to_string(v.int_value()),
                        pos);
    }
    return static_cast<uint64_t>(v.int_value());
}

void RequireSameModulus(const FlatMatrixT<uint32_t>& a, const FlatMatrixT<uint32_t>& b,
                        const SourcePos& pos, const char* who) {
    if (a.get_p() != b.get_p()) {
        throw TypeError(std::string(who) + ": the two matrices live over different fields (F_" +
                            std::to_string(a.get_p()) + " and F_" + std::to_string(b.get_p()) + ")",
                        pos);
    }
}

// --- the per-kind hash ------------------------------------------------------
// GoldenHarness.cpp's HashFlatInto / HashPolyInto / HashScalar, verbatim, over
// the one FNV-1a-64 of src/ModArithmetic/Fnv.h.  Any drift here would
// silently turn every `hash(x) == golden(...)` pin into a tautology, so the two
// definitions are pinned against each other by ConsoleScriptValidation, which
// compares real objects against the real reference.

uint64_t HashMatrix(const FlatMatrixT<uint32_t>& m) {
    uint64_t h = engine::kFnvOffset64;
    h = engine::FnvU64(h, m.get_rows());
    h = engine::FnvU64(h, m.get_cols());
    for (uint32_t r = 0; r < m.get_rows(); ++r)
        for (uint32_t c = 0; c < m.get_cols(); ++c)
            h = engine::FnvU64(h, static_cast<uint64_t>(m.get_entry(r, c)));
    return h;
}

uint64_t HashPoly(const hecke::FpPolyT<uint32_t>& f) {
    uint64_t h = engine::kFnvOffset64;
    h = engine::FnvU64(h, f.coeffs.size());
    for (uint32_t c : f.coeffs) h = engine::FnvU64(h, static_cast<uint64_t>(c));
    return h;
}

uint64_t HashScalarValue(uint64_t v) { return engine::FnvU64(engine::kFnvOffset64, v); }

uint64_t HashMatrixList(const std::vector<FlatMatrixT<uint32_t>>& ms) {
    uint64_t h = engine::FnvU64(engine::kFnvOffset64, ms.size());
    for (const FlatMatrixT<uint32_t>& m : ms) {
        h = engine::FnvU64(h, m.get_rows());
        h = engine::FnvU64(h, m.get_cols());
        for (uint32_t r = 0; r < m.get_rows(); ++r)
            for (uint32_t c = 0; c < m.get_cols(); ++c)
                h = engine::FnvU64(h, static_cast<uint64_t>(m.get_entry(r, c)));
    }
    return h;
}

} // namespace bind
} // namespace console
