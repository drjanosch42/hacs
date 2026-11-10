// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// RegisterHeckeExtensionVerbsValues.cpp -- one area of the extension layer's console
// bindings (S24 CL-8 split of RegisterHeckeExtensionVerbs.cpp; the map is in
// RegisterHeckeExtensionVerbsDetail.h): Atlas (E6), GF / GFElt (the canonical split field), Settings (R1).
// ===========================================================================

#include "RegisterHeckeExtensionVerbsDetail.h"

namespace hecke_ext {
namespace verbs_detail {

namespace {

// ===========================================================================
// Atlas (E6), GF / GFElt (the canonical split field), Settings (R1).
// ===========================================================================

struct AtlasVal {
    uint32_t ell = 0;
    std::shared_ptr<const Atlas> atlas;
    void ensure() {
        if (atlas) return;
        RequireField("atlas");
        atlas = InProcessAtlas(ell);
        CONSOLE_CHECK(atlas != nullptr);
    }
};
AtlasVal& AtlasOf(const ObjectPtr& o) { return *o->payload_as<AtlasVal>(); }

struct GFVal {
    std::shared_ptr<CanonicalField> F;
};
struct GFEltVal {
    std::shared_ptr<CanonicalField> F;
    ObjectPtr field;     // the GF handle (keep-alive)
    FieldElt a;
};
GFVal& GFOf(const ObjectPtr& o) { return *o->payload_as<GFVal>(); }
GFEltVal& EltOf(const ObjectPtr& o) { return *o->payload_as<GFEltVal>(); }

} // namespace

Value VerbAtlas(CallContext& ctx, const std::vector<Value>& a) {
    static const char* kWho = "atlas(ell)";
    RequireField("atlas");
    const uint32_t ell = console::bind::U32Arg(a[0], ctx.pos, kWho);
    if (!IsInProcessAtlasEll(ell)) {
        throw RuntimeError(std::string(kWho) + ": atlas_unavailable -- Sigma(ell) is built in "
                                               "process only at ell in {5, 7, 11, 13} (larger "
                                               "atlases are built outside the suite by "
                                               "`hacs --atlas`)",
                           ctx.pos);
    }
    auto v = std::make_shared<AtlasVal>();
    v->ell = ell;
    return Value::from_object(Object::create(console::bind::TypeByName(ctx, kTypeAtlas), ctx.pos,
                                             ctx.call_text, ctx.session->settings().snapshot(), v,
                                             console::bind::LazyCompute([v](Object&) { v->ensure(); })));
}

namespace {

Value FieldEltValue(const TypeInfo* type, const ObjectPtr& field, const std::shared_ptr<CanonicalField>& F,
                    FieldElt x) {
    auto p = std::make_shared<GFEltVal>();
    p->F = F;
    p->field = field;
    p->a = std::move(x);
    return Value::from_object(ChildOf(type, field, p));
}

} // namespace

Value VerbGF(CallContext& ctx, const std::vector<Value>& a) {
    static const char* kWho = "GF(ell, r)";
    const uint32_t ell = console::bind::U32Arg(a[0], ctx.pos, kWho);
    const uint32_t r = console::bind::U32Arg(a[1], ctx.pos, kWho);
    if (ell < 2) throw TypeError(std::string(kWho) + ": ell must be a prime >= 2", ctx.pos);
    if (r < 1) throw TypeError(std::string(kWho) + ": the degree r must be at least 1", ctx.pos);
    auto p = std::make_shared<GFVal>();
    // The `field_polynomial` setting in force selects C_{ell,r} ([E]
    // def:canon): `least` by default, `conway` a legal value this build cannot
    // execute (exit 5 through the seam).
    p->F = std::make_shared<CanonicalField>(ell, FieldPolynomialFromSettings(ell, r));
    return Value::from_object(console::bind::MakeValueObject(ctx, kTypeGF, p));
}

namespace {

const GFEltVal& EltArg(const Value& v, const SourcePos& pos, const char* who) {
    return *console::bind::AsObjectOf(v, kTypeGFElt, pos, who)->payload_as<GFEltVal>();
}

void SameField(const GFEltVal& x, const GFEltVal& y, const SourcePos& pos, const char* who) {
    if (x.F->ell() != y.F->ell() || !(x.F->defining_poly() == y.F->defining_poly()))
        throw TypeError(std::string(who) + ": the two elements lie in different fields", pos);
}

Value GFBinary(CallContext& ctx, const std::vector<Value>& a, const char* who, int op) {
    const GFEltVal& x = EltArg(a[0], ctx.pos, who);
    const GFEltVal& y = EltArg(a[1], ctx.pos, who);
    SameField(x, y, ctx.pos, who);
    FieldElt z = op == 0 ? x.F->add(x.a, y.a) : op == 1 ? x.F->sub(x.a, y.a) : x.F->mul(x.a, y.a);
    return FieldEltValue(TypesOf(x.field).gfelt, x.field, x.F, std::move(z));
}

} // namespace

Value VerbGFAdd(CallContext& ctx, const std::vector<Value>& a) { return GFBinary(ctx, a, "gf_add(x, y)", 0); }
Value VerbGFSub(CallContext& ctx, const std::vector<Value>& a) { return GFBinary(ctx, a, "gf_sub(x, y)", 1); }
Value VerbGFMul(CallContext& ctx, const std::vector<Value>& a) { return GFBinary(ctx, a, "gf_mul(x, y)", 2); }
Value VerbGFInv(CallContext& ctx, const std::vector<Value>& a) {
    const GFEltVal& x = EltArg(a[0], ctx.pos, "gf_inv(x)");
    if (x.F->is_zero(x.a)) throw RuntimeError("gf_inv(x): x is zero", ctx.pos);
    return FieldEltValue(TypesOf(x.field).gfelt, x.field, x.F, x.F->inv(x.a));
}
Value VerbGFPow(CallContext& ctx, const std::vector<Value>& a) {
    const GFEltVal& x = EltArg(a[0], ctx.pos, "gf_pow(x, e)");
    return FieldEltValue(TypesOf(x.field).gfelt, x.field, x.F,
                         x.F->pow(x.a, console::bind::U64Arg(a[1], ctx.pos, "gf_pow(x, e)")));
}
Value VerbGFFrobenius(CallContext& ctx, const std::vector<Value>& a) {
    const GFEltVal& x = EltArg(a[0], ctx.pos, "gf_frobenius(x)");
    return FieldEltValue(TypesOf(x.field).gfelt, x.field, x.F, x.F->frobenius(x.a));
}
Value VerbGFEval(CallContext& ctx, const std::vector<Value>& a) {
    static const char* kWho = "gf_eval(f, x)";
    const console::bind::PolyVal& f = console::bind::AsPoly(a[0], ctx.pos, kWho);
    const GFEltVal& x = EltArg(a[1], ctx.pos, kWho);
    if (f.f.ell != x.F->ell()) throw TypeError(std::string(kWho) + ": f is not over the field's prime", ctx.pos);
    return FieldEltValue(TypesOf(x.field).gfelt, x.field, x.F, x.F->eval(f.f, x.a));
}

namespace {

// The Settings value: a snapshot of hecke::MathSettings.
struct SettingsVal {
    hecke::MathSettings s;
};
SettingsVal& SetOf(const ObjectPtr& o) { return *o->payload_as<SettingsVal>(); }

} // namespace

Value SettingsHandleFrom(const ObjectPtr& owner, const hecke::MathSettings& s) {
    auto p = std::make_shared<SettingsVal>();
    p->s = s;
    return Value::from_object(ChildOf(TypesOf(owner).settings, owner, p));
}

Value VerbSettingsSnapshot(CallContext& ctx, const std::vector<Value>&) {
    auto p = std::make_shared<SettingsVal>();
    p->s = hecke::math_settings();
    return Value::from_object(console::bind::MakeValueObject(ctx, kTypeSettings, p));
}

void RegisterValueTypes(Registry& registry) {
    // --- Atlas ------------------------------------------------------------
    {
        TypeInfo t;
        t.name = kTypeAtlas;
        t.doc = "Sigma(ell): the anemic level-1 eigensystems over weights <= ell + 1, their "
                "theta-twists, and the separating set S_sep(ell) (HeckeAtlas, in process at "
                "ell in {5, 7, 11, 13}); lazy";
        auto S = [](const ObjectPtr& o) -> const AtlasStore& { return AtlasOf(o).atlas->store; };
        t.members.push_back({"ell", false, "the prime", [](const ObjectPtr& o) { return Value::from_int(AtlasOf(o).ell); }});
        t.members.push_back({"n_max", true, "the largest even weight cell n <= ell - 1", [S](const ObjectPtr& o) { return Value::from_int(S(o).n_max); }});
        t.members.push_back({"primes", true, "the anemic pool T", [S](const ObjectPtr& o) { return U64List(S(o).primes); }});
        t.members.push_back({"num_systems", true, "|Sigma(ell)|", [S](const ObjectPtr& o) { return Value::from_int(static_cast<std::int64_t>(S(o).systems.size())); }});
        t.members.push_back({"chi_count", true, "distinct chi_{a,b} as T-systems", [S](const ObjectPtr& o) { return Value::from_int(S(o).chi_count); }});
        t.members.push_back({"chi_in_sigma", true, "of which in Sigma", [S](const ObjectPtr& o) { return Value::from_int(S(o).chi_in_sigma); }});
        t.members.push_back({"separating_primes", true, "the greedy S_sep(ell)", [S](const ObjectPtr& o) { return U64List(S(o).s_sep.primes); }});
        t.members.push_back({"separates", true, "S_sep separates Sigma u chi", [S](const ObjectPtr& o) { return Value::from_bool(S(o).s_sep.separates); }});
        t.members.push_back({"s0_separates", true, "{2,3,5,7} \\ {ell} already separates", [S](const ObjectPtr& o) { return Value::from_bool(S(o).s0_anemic.separates); }});
        t.members.push_back({"cells", true, "the weight cells {n, dims, ideals, local_ideals}",
                             [S](const ObjectPtr& o) {
                                 std::vector<Value> out;
                                 for (const AtlasCell& c : S(o).cells) {
                                     Record r;
                                     r.set("n", Value::from_int(c.n));
                                     r.set("dims", U32List(c.dims));
                                     r.set("ideals", Value::from_int(c.ideals));
                                     r.set("local_ideals", Value::from_int(c.local_ideals));
                                     out.push_back(Value::from_record(std::move(r)));
                                 }
                                 return Value::from_list(std::move(out));
                             }});
        t.members.push_back({"json", true, "the hecke_atlas/v1 store text", [S](const ObjectPtr& o) { return Value::from_string(S(o).to_json()); }});
        {
            TypeInfo::Method m;
            m.name = "system";
            m.params = {"i"};
            m.required = 1;
            m.doc = "the i-th system {id, r, type, base, twist, chi, eigenvalues, collected, in_H1}";
            m.call = [S](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                const uint32_t i = console::bind::U32Arg(args[0], pos, "Atlas.system(i)");
                if (i >= S(o).systems.size()) throw RuntimeError("Atlas.system(i): no system " + std::to_string(i), pos);
                const AtlasSystem& s = S(o).systems[i];
                Record r;
                r.set("id", Value::from_int(s.id));
                r.set("r", Value::from_int(s.r));
                r.set("type", Value::from_string(s.type));
                r.set("base", Value::from_int(s.base));
                r.set("twist", Value::from_int(s.twist));
                r.set("chi", s.has_chi ? U32List({s.chi_a, s.chi_b}) : Value::nil());
                std::vector<Value> eig;
                for (const FieldElt& e : s.eigenvalues) eig.push_back(s.r == 1 ? Value::from_int(e[0]) : U32List(e));
                r.set("eigenvalues", Value::from_list(std::move(eig)));
                r.set("collected", Value::from_bool(s.collected));
                r.set("in_H1", Value::from_bool(s.in_H1));
                return Value::from_record(std::move(r));
            };
            t.methods.push_back(std::move(m));
        }
        {
            TypeInfo::Method m;
            m.name = "lookup";
            m.params = {"eigenvalues"};
            m.required = 1;
            m.doc = "the id of the system with these eigenvalues on `primes` (Ints at r = 1, "
                    "coefficient lists at r > 1; any Frobenius conjugate), or -1";
            m.call = [S](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                static const char* kW = "Atlas.lookup(eigenvalues)";
                if (!args[0].is_list()) throw TypeError(std::string(kW) + ": expected a list", pos);
                const std::vector<Value>& xs = args[0].list_value();
                if (xs.size() != S(o).primes.size())
                    throw TypeError(std::string(kW) + ": one eigenvalue per prime of `primes`", pos);
                uint32_t r = 1;
                std::vector<FieldElt> v;
                for (const Value& x : xs) {
                    if (x.is_int()) {
                        v.push_back(FieldElt{console::bind::U32Arg(x, pos, kW)});
                    } else if (x.is_list()) {
                        FieldElt e;
                        for (const Value& c : x.list_value()) e.push_back(console::bind::U32Arg(c, pos, kW));
                        v.push_back(std::move(e));
                    } else {
                        throw TypeError(std::string(kW) + ": an eigenvalue is an Int or a list", pos);
                    }
                    if (v.size() == 1) r = static_cast<uint32_t>(v.back().size());
                    if (v.back().size() != r) throw TypeError(std::string(kW) + ": mixed field degrees", pos);
                }
                const CanonicalField F(AtlasOf(o).ell, r);
                const std::vector<FieldElt> key = AtlasOrbitKey(F, v);
                return Value::from_int(AtlasFind(S(o), r, key));
            };
            t.methods.push_back(std::move(m));
        }
        t.print = [](const ObjectPtr& o, std::int64_t) {
            std::ostringstream os;
            os << "Atlas Sigma(" << AtlasOf(o).ell << ")";
            if (!o->computed()) os << " (declared, not yet computed)";
            else os << ": " << AtlasOf(o).atlas->store.systems.size() << " systems";
            return os.str();
        };
        registry.register_type(std::move(t));
    }
    // --- GF ---------------------------------------------------------------
    {
        TypeInfo t;
        t.name = kTypeGF;
        t.doc = "the project-canonical field F_ell[Y]/(C_{ell,r}) ([E] def:canon; "
                "HeckeCanonicalField), C chosen by the `field_polynomial` setting";
        auto F = [](const ObjectPtr& o) -> const CanonicalField& { return *GFOf(o).F; };
        t.members.push_back({"ell", false, "the prime", [F](const ObjectPtr& o) { return Value::from_int(F(o).ell()); }});
        t.members.push_back({"degree", false, "r = [F : F_ell]", [F](const ObjectPtr& o) { return Value::from_int(F(o).degree()); }});
        t.members.push_back({"polynomial", false, "C_{ell,r}", [F](const ObjectPtr& o) { return PolyFrom(o, F(o).defining_poly()); }});
        t.members.push_back({"canonical", false, "C is the CanonicalLess-least monic irreducible", [F](const ObjectPtr& o) { return Value::from_bool(F(o).is_canonical()); }});
        t.members.push_back({"zero", false, "0", [F](const ObjectPtr& o) { return FieldEltValue(TypesOf(o).gfelt, o, GFOf(o).F, F(o).zero()); }});
        t.members.push_back({"one", false, "1", [F](const ObjectPtr& o) { return FieldEltValue(TypesOf(o).gfelt, o, GFOf(o).F, F(o).one()); }});
        t.members.push_back({"gen", false, "Y, the class of the variable", [F](const ObjectPtr& o) { return FieldEltValue(TypesOf(o).gfelt, o, GFOf(o).F, F(o).generator()); }});
        {
            TypeInfo::Method m;
            m.name = "elt";
            m.params = {"coeffs"};
            m.required = 1;
            m.computed = false;
            m.doc = "the element sum c_i Y^i (coefficients low -> high, reduced mod C)";
            m.call = [F](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                if (!args[0].is_list()) throw TypeError("GF.elt(coeffs): expected a list of Int", pos);
                std::vector<uint32_t> c;
                for (const Value& x : args[0].list_value())
                    c.push_back(static_cast<uint32_t>(console::bind::U64Arg(x, pos, "GF.elt(coeffs)") % F(o).ell()));
                return FieldEltValue(TypesOf(o).gfelt, o, GFOf(o).F,
                                     F(o).from_poly(hecke::FpPoly(F(o).ell(), c)));
            };
            t.methods.push_back(std::move(m));
        }
        {
            TypeInfo::Method m;
            m.name = "element_at";
            m.params = {"i"};
            m.required = 1;
            m.computed = false;
            m.doc = "the i-th element of the field's fixed enumeration";
            m.call = [F](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                return FieldEltValue(TypesOf(o).gfelt, o, GFOf(o).F,
                                     F(o).element_at(console::bind::U64Arg(args[0], pos, "GF.element_at(i)")));
            };
            t.methods.push_back(std::move(m));
        }
        t.print = [](const ObjectPtr& o, std::int64_t) {
            const CanonicalField& f = *GFOf(o).F;
            // FpPoly prints its variable as X; the field's generator is Y.
            return "GF(" + std::to_string(f.ell()) + "^" + std::to_string(f.degree()) +
                   "), C_{ell,r} = " + f.defining_poly().to_string();
        };
        t.hash = [](const ObjectPtr& o) {
            return console::bind::HashPoly(GFOf(o).F->defining_poly());
        };
        registry.register_type(std::move(t));
    }
    // --- GFElt ------------------------------------------------------------
    {
        TypeInfo t;
        t.name = kTypeGFElt;
        t.doc = "an element of a GF field (coefficient vector in the basis 1, Y, ..., Y^{r-1})";
        t.members.push_back({"coeffs", false, "the coefficients, low -> high, width r",
                             [](const ObjectPtr& o) { return U32List(EltOf(o).a); }});
        t.members.push_back({"is_zero", false, "x == 0", [](const ObjectPtr& o) { return Value::from_bool(EltOf(o).F->is_zero(EltOf(o).a)); }});
        t.members.push_back({"field", false, "the GF handle", [](const ObjectPtr& o) { return Value::from_object(EltOf(o).field); }});
        t.members.push_back({"poly", false, "the representative polynomial in Y", [](const ObjectPtr& o) { return PolyFrom(o, EltOf(o).F->to_poly(EltOf(o).a)); }});
        t.print = [](const ObjectPtr& o, std::int64_t) { return EltOf(o).F->to_text(EltOf(o).a); };
        t.equal = [](const ObjectPtr& x, const ObjectPtr& y) {
            const GFEltVal& a = EltOf(x);
            const GFEltVal& b = EltOf(y);
            return a.F->ell() == b.F->ell() && a.F->defining_poly() == b.F->defining_poly() && a.a == b.a;
        };
        t.hash = [](const ObjectPtr& o) {
            // FNV-1a-64 (Fnv.h) over the field's prime, degree and the coefficients.
            const GFEltVal& e = EltOf(o);
            engine::FnvHasher h;
            h.feed(e.F->ell());
            h.feed(e.F->degree());
            for (uint32_t c : e.a) h.feed(c);
            return h.value();
        };
        registry.register_type(std::move(t));
    }
    // --- Settings ---------------------------------------------------------
    {
        TypeInfo t;
        t.name = kTypeSettings;
        t.doc = "a snapshot of the MATHEMATICAL settings (hecke::MathSettings, [X] Sec. 1.5): "
                "every registered key with its value and source, and the fingerprint";
        t.members.push_back({"fingerprint", false, "FNV-1a-64 of the non-default entries (0x0 at the defaults)",
                             [](const ObjectPtr& o) { return Value::from_string(oc_util::HexU64(SetOf(o).s.fingerprint())); }});
        t.members.push_back({"is_default", false, "every key at its default", [](const ObjectPtr& o) { return Value::from_bool(SetOf(o).s.is_default()); }});
        t.members.push_back({"keys", false, "the registered keys, in registry order",
                             [](const ObjectPtr&) { return StringList(hecke::MathSettings::keys()); }});
        t.members.push_back({"values", false, "{key: value text}", [](const ObjectPtr& o) {
                                 Record r;
                                 for (const std::string& k : hecke::MathSettings::keys())
                                     r.set(k, Value::from_string(SetOf(o).s.text(k)));
                                 return Value::from_record(std::move(r));
                             }});
        t.members.push_back({"text", false, "the canonical text", [](const ObjectPtr& o) { return Value::from_string(SetOf(o).s.canonical_text()); }});
        t.members.push_back({"json", false, "the settings JSON (values and sources)", [](const ObjectPtr& o) { return Value::from_string(SetOf(o).s.to_json()); }});
        {
            TypeInfo::Method m;
            m.name = "get";
            m.params = {"key"};
            m.required = 1;
            m.computed = false;
            m.doc = "the value text of one key";
            m.call = [](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                if (!args[0].is_string()) throw TypeError("Settings.get(key): expected a String", pos);
                if (!hecke::MathSettings::known_key(args[0].string_value()))
                    throw RuntimeError("Settings.get(key): no mathematical setting '" + args[0].string_value() + "'", pos);
                return Value::from_string(SetOf(o).s.text(args[0].string_value()));
            };
            t.methods.push_back(std::move(m));
        }
        t.print = [](const ObjectPtr& o, std::int64_t) {
            return "Settings (fingerprint " + oc_util::HexU64(SetOf(o).s.fingerprint()) + ")";
        };
        t.hash = [](const ObjectPtr& o) { return SetOf(o).s.fingerprint(); };
        registry.register_type(std::move(t));
    }
}

} // namespace verbs_detail
} // namespace hecke_ext
