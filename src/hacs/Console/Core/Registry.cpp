// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Registry.cpp -- see Registry.h.
// ===========================================================================

#include "Registry.h"

#include "Lexer.h"

namespace console {

Param Param::any(std::string n) {
    Param p;
    p.name = std::move(n);
    p.kind = ValueKind::Nil;
    return p;
}

Param Param::of(std::string n, ValueKind k) {
    Param p;
    p.name = std::move(n);
    p.kind = k;
    return p;
}

Param Param::object(std::string n, std::string type) {
    Param p;
    p.name = std::move(n);
    p.kind = ValueKind::Object;
    p.type_name = std::move(type);
    return p;
}

Param& Param::optional(Value dflt) {
    required = false;
    default_value = std::move(dflt);
    return *this;
}

Param& Param::described(std::string d) {
    doc = std::move(d);
    return *this;
}

std::size_t Signature::required_count() const {
    std::size_t n = 0;
    for (const Param& p : params) {
        if (p.required) ++n;
    }
    return n;
}

namespace {

std::string ValueTextForDoc(const Value& v) {
    switch (v.kind()) {
        case ValueKind::Int:    return std::to_string(v.int_value());
        case ValueKind::Bool:   return v.bool_value() ? "true" : "false";
        case ValueKind::String: return "\"" + v.string_value() + "\"";
        case ValueKind::Nil:    return "nil";
        default:                return v.type_name();
    }
}

std::string ParamTypeText(const Param& p) {
    if (p.kind == ValueKind::Nil) return "any";
    if (p.kind == ValueKind::Object) {
        return p.type_name.empty() ? std::string("Object") : p.type_name;
    }
    return ValueKindName(p.kind);
}

} // namespace

std::string Signature::to_string(const std::string& verb) const {
    std::string out = verb + "(";
    for (std::size_t i = 0; i < params.size(); ++i) {
        if (i != 0) out += ", ";
        out += params[i].name;
        out += ": ";
        out += ParamTypeText(params[i]);
        if (!params[i].required) {
            out += " = ";
            out += ValueTextForDoc(params[i].default_value);
        }
        if (variadic && i + 1 == params.size()) out += ", ...";
    }
    out += ")";
    return out;
}

bool ParamAccepts(const Param& p, const Value& v) {
    if (p.kind == ValueKind::Nil) return true;
    if (p.kind != v.kind()) return false;
    if (p.kind == ValueKind::Object && !p.type_name.empty()) {
        const ObjectPtr& o = v.object_value();
        if (!o) return false;
        return o->type_name() == p.type_name;
    }
    return true;
}

const TypeInfo* Registry::register_type(TypeInfo info) {
    CONSOLE_CHECK_MSG(!info.name.empty(), "a console type must have a name");
    for (std::unique_ptr<TypeInfo>& t : types_) {
        if (t->name == info.name) {
            *t = std::move(info);
            return t.get();
        }
    }
    types_.push_back(std::unique_ptr<TypeInfo>(new TypeInfo(std::move(info))));
    return types_.back().get();
}

const TypeInfo* Registry::find_type(const std::string& name) const {
    for (const std::unique_ptr<TypeInfo>& t : types_) {
        if (t->name == name) return t.get();
    }
    return nullptr;
}

std::vector<std::string> Registry::type_names() const {
    std::vector<std::string> out;
    out.reserve(types_.size());
    for (const std::unique_ptr<TypeInfo>& t : types_) out.push_back(t->name);
    return out;
}

void Registry::register_verb(const std::string& name, Signature sig, VerbImpl impl,
                             std::string doc, std::string group) {
    CONSOLE_CHECK_MSG(!name.empty(), "a verb must have a name");
    CONSOLE_CHECK_MSG(static_cast<bool>(impl), "a verb must have an implementation");
    CONSOLE_CHECK_MSG(!IsReservedWord(name),
                      "a verb may not be named after a reserved word");
    Overload o;
    o.sig = std::move(sig);
    o.impl = std::move(impl);
    o.doc = std::move(doc);
    for (std::unique_ptr<VerbEntry>& e : verbs_) {
        if (e->name == name) {
            e->overloads.push_back(std::move(o));
            return;
        }
    }
    verbs_.push_back(std::unique_ptr<VerbEntry>(new VerbEntry()));
    VerbEntry& e = *verbs_.back();
    e.name = name;
    e.group = std::move(group);
    e.overloads.push_back(std::move(o));
}

const VerbEntry* Registry::find_verb(const std::string& name) const {
    for (const std::unique_ptr<VerbEntry>& e : verbs_) {
        if (e->name == name) return e.get();
    }
    return nullptr;
}

std::vector<std::string> Registry::verb_names() const {
    std::vector<std::string> out;
    out.reserve(verbs_.size());
    for (const std::unique_ptr<VerbEntry>& e : verbs_) out.push_back(e->name);
    return out;
}

void Registry::transform_overloads(const OverloadTransform& f) {
    std::vector<std::unique_ptr<VerbEntry>> kept;
    for (std::unique_ptr<VerbEntry>& e : verbs_) {
        std::vector<Overload> keep;
        for (Overload& o : e->overloads) {
            if (f(*e, o)) keep.push_back(std::move(o));
        }
        e->overloads = std::move(keep);
        if (!e->overloads.empty()) kept.push_back(std::move(e));
    }
    verbs_ = std::move(kept);
}

namespace {

// Binds one overload.  Returns true and fills `bound` on success; otherwise
// fills `why` with the first reason it failed.
bool BindOverload(const Signature& sig, const std::vector<Value>& positional,
                  const std::vector<std::pair<std::string, Value>>& keyword,
                  std::vector<Value>& bound, std::string& why) {
    bound.clear();

    if (sig.variadic) {
        // Variadic: the last parameter repeats.  Keyword arguments are not
        // accepted for the repeating tail (there is no name to bind them to).
        if (!keyword.empty()) {
            why = "this overload takes no keyword arguments";
            return false;
        }
        const std::size_t fixed = sig.params.empty() ? 0 : sig.params.size() - 1;
        if (positional.size() < fixed) {
            why = "expected at least " + std::to_string(fixed) + " argument(s), got " +
                  std::to_string(positional.size());
            return false;
        }
        for (std::size_t i = 0; i < positional.size(); ++i) {
            const Param& p = sig.params[i < fixed ? i : sig.params.size() - 1];
            if (!ParamAccepts(p, positional[i])) {
                why = "argument " + std::to_string(i + 1) + " ('" + p.name + "') expects " +
                      ParamTypeText(p) + ", got " + positional[i].type_name();
                return false;
            }
            bound.push_back(positional[i]);
        }
        return true;
    }

    if (positional.size() > sig.params.size()) {
        why = "expected at most " + std::to_string(sig.params.size()) + " argument(s), got " +
              std::to_string(positional.size());
        return false;
    }

    std::vector<Value> slots(sig.params.size());
    std::vector<bool> filled(sig.params.size(), false);

    for (std::size_t i = 0; i < positional.size(); ++i) {
        slots[i] = positional[i];
        filled[i] = true;
    }
    for (const std::pair<std::string, Value>& kv : keyword) {
        std::size_t idx = sig.params.size();
        for (std::size_t i = 0; i < sig.params.size(); ++i) {
            if (sig.params[i].name == kv.first) { idx = i; break; }
        }
        if (idx == sig.params.size()) {
            why = "no parameter named '" + kv.first + "'";
            return false;
        }
        if (filled[idx]) {
            why = "parameter '" + kv.first + "' given twice";
            return false;
        }
        slots[idx] = kv.second;
        filled[idx] = true;
    }
    for (std::size_t i = 0; i < sig.params.size(); ++i) {
        const Param& p = sig.params[i];
        if (!filled[i]) {
            if (p.required) {
                why = "missing required argument '" + p.name + "'";
                return false;
            }
            slots[i] = p.default_value;
            continue;
        }
        if (!ParamAccepts(p, slots[i])) {
            why = "argument '" + p.name + "' expects " + ParamTypeText(p) + ", got " +
                  slots[i].type_name();
            return false;
        }
    }
    bound = std::move(slots);
    return true;
}

} // namespace

const Overload* Registry::resolve(const VerbEntry& entry,
                                  const std::vector<Value>& positional,
                                  const std::vector<std::pair<std::string, Value>>& keyword,
                                  const SourcePos& pos,
                                  std::vector<Value>& bound_out) const {
    std::vector<std::string> reasons;
    for (const Overload& o : entry.overloads) {
        std::vector<Value> bound;
        std::string why;
        if (BindOverload(o.sig, positional, keyword, bound, why)) {
            bound_out = std::move(bound);
            return &o;   // first match in registration order
        }
        reasons.push_back(o.sig.to_string(entry.name) + " -- " + why);
    }
    std::string msg = "no overload of '" + entry.name + "' accepts these arguments";
    for (const std::string& r : reasons) {
        msg += "\n    " + r;
    }
    throw TypeError(msg, pos);
}

Registry& Registry::global() {
    static Registry instance;
    return instance;
}

} // namespace console
