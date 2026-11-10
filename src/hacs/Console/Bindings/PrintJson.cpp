// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// PrintJson.cpp -- see PrintJson.h.
// ===========================================================================

#include "PrintJson.h"

#include "ServeJson.h"

#include "../Core/Object.h"
#include "../Core/Printer.h"
#include "../Core/Registry.h"
#include "../Core/Session.h"

#include <exception>
#include <string>
#include <utility>
#include <vector>

namespace console {
namespace serve {

namespace {

using serve_json::IntJson;
using serve_json::Quote;

// Nested OBJECTS are rendered in full down to this depth; below it an object is
// a reference envelope (t, canonical, hash: null, v: null, truncated) -- the
// member graph of the extension types is not a tree (an ideal names its
// algebra, the algebra its ideals).
constexpr int kMaxObjectDepth = 3;

// Members never read by the generic form: a re-verification, not data.
bool SkippedMember(const std::string& name) { return name == "verify"; }

std::string StateName(const ObjectPtr& o) {
    if (!o) return "failed";
    switch (o->state()) {
        case ObjectState::Declared: return "declared";
        case ObjectState::Computed: return "computed";
        case ObjectState::Failed: return "failed";
    }
    return "failed";
}

// Reads member `name` without ever forcing: a computed member of a handle that
// is not yet computed is "absent".  Any exception is "absent" as well.
bool ReadMember(const ObjectPtr& o, const std::string& name, Value& out) {
    if (!o || o->type() == nullptr) return false;
    const TypeInfo::Member* m = o->type()->find_member(name);
    if (m == nullptr || !m->get) return false;
    if (m->computed && !o->computed()) return false;
    try {
        out = m->get(o);
        return true;
    } catch (...) {
        return false;
    }
}

bool CallMethod(const ObjectPtr& o, const std::string& name, const std::vector<Value>& args, Value& out) {
    if (!o || o->type() == nullptr) return false;
    const TypeInfo::Method* m = o->type()->find_method(name);
    if (m == nullptr || !m->call) return false;
    if (m->computed && !o->computed()) return false;
    try {
        out = m->call(o, args, SourcePos("<print_json>", 0, 0));
        return true;
    } catch (...) {
        return false;
    }
}

// Calls a registered verb of the context's registry (`hash`, `coeffs`) on one
// argument.  The served registry's wrappers are bypassed only in the sense that
// the context points at whichever registry the caller chose.
bool CallVerb1(const PrintJsonContext& ctx, const char* verb, const Value& arg, Value& out) {
    if (ctx.registry == nullptr) return false;
    const VerbEntry* e = ctx.registry->find_verb(verb);
    if (e == nullptr) return false;
    try {
        std::vector<Value> bound;
        const Overload* ov = ctx.registry->resolve(*e, {arg}, {}, SourcePos("<print_json>", 0, 0), bound);
        CallContext cc;
        cc.session = ctx.session;
        cc.registry = const_cast<Registry*>(ctx.registry);
        cc.pos = SourcePos("<print_json>", 0, 0);
        cc.call_text = std::string(verb) + "(...)";
        out = ov->impl(cc, bound);
        return true;
    } catch (...) {
        return false;
    }
}

bool IntOf(const Value& v, std::int64_t& out) {
    if (!v.is_int()) return false;
    out = v.int_value();
    return true;
}

// A plain JSON rendering of a member value (not an envelope): Int / Bool /
// String / List / Record recursively, an object as its canonical text.  Used
// inside the dedicated `v` trees, where the schema says "int", "[int,...]".
std::string Plain(const Value& v, const PrintJsonContext& ctx) {
    switch (v.kind()) {
        case ValueKind::Nil: return "null";
        case ValueKind::Int: return IntJson(v.int_value());
        case ValueKind::Bool: return v.bool_value() ? "true" : "false";
        case ValueKind::String: return Quote(v.string_value());
        case ValueKind::List: {
            std::string o = "[";
            bool first = true;
            for (const Value& x : v.list_value()) {
                if (!first) o += ',';
                first = false;
                o += Plain(x, ctx);
            }
            return o + "]";
        }
        case ValueKind::Record: {
            std::string o = "{";
            bool first = true;
            for (const auto& kv : v.record_value().entries()) {
                if (!first) o += ',';
                first = false;
                o += Quote(kv.first) + ":" + Plain(kv.second, ctx);
            }
            return o + "}";
        }
        case ValueKind::Object:
            return Quote(PrintCanonical(v, ctx.max_entries));
    }
    return "null";
}

// A JSON text the engine already emits, as a tree (no re-encoding; integer
// literals follow the 2^53 rule).  An unparsable text stays a String.
std::string EmbeddedJson(const std::string& text) {
    serve_json::Json j;
    std::string err;
    if (serve_json::ParseJson(text, j, err)) return serve_json::WriteJson(j);
    return Quote(text);
}

// ---------------------------------------------------------------------------
// (slot-24 NB-RENDER R1) Render hints ([P] Sec. 18.5 "hint").  One row per
// (record schema or type name, path): the value at that path is mathematically
// a polynomial / q-series / abelian group / partition / layer profile, which its
// own type (a List of Int or String) cannot say.  The twin of
// Console/serve/render_hints.tsv (ConsoleReadOnlyTagValidation arm (H) compares
// them).  A hint never changes `v` or `canonical`; it is one more key after
// `state`.  Paths: `.`-separated keys / member names, `[]` = every element of a
// List, `*` = any key of a Record, `name()` = the calls of a member-form object.
// Ring attributes `ring=Z/ell^<x>` / `ring=F_ell` are resolved against the Int
// fields of the enclosing Records (nearest first), then the object's members;
// so is any attribute written `key=@name` (slot-24 R3); an unresolved
// parameter is simply left out.
// ---------------------------------------------------------------------------
const std::vector<RenderHintRow> kHintRows = {
    {"level_n/v1", "ops[].charpoly", "poly", "order=low_to_high var=X ring=F_ell"},
    {"nebentypus/v1", "ops[].charpoly", "poly", "order=low_to_high var=X ring=F_ell"},
    {"forms_algebra/v1", "ops[].charpoly", "poly", "order=low_to_high var=X ring=F_ell"},
    {"forms_algebra/v1", "basis[]", "qexp", "ring=F_ell"},
    {"hida_lambda/v1", "ops[].charpoly", "poly", "order=low_to_high var=X ring=Z/ell^W"},
    {"hida_tower/v1", "levels[].charpoly.*", "poly", "order=low_to_high var=X ring=Z/ell^e"},
    {"hida_tower/v1", "levels[].free_charpoly.*", "poly", "order=low_to_high var=X ring=Z/ell^e"},
    {"kummer_unit/v1", "coefficients", "poly", "order=low_to_high var=zeta ring=Z"},
    {"lfunction/v1", "coefficients", "poly", "order=low_to_high var=T ring=Z/ell^R"},
    {"DescentRecord", "groups_inv[]", "abgroup", ""},
    {"DescentRecord", "pair_inv.*[]", "abgroup", ""},
    {"DescentRecord", "charpoly[].coefficients", "poly", "order=high_to_low var=X ring=Z"},
    // (slot-24 NB-TOURFIX) char_Z(T_p) factored over Z: a List of {coefficients (high -> low), degree, multiplicity, irreducible}
    {"DescentRecord", "charpoly_factored[].factors", "factored", "order=high_to_low var=X ring=Z"},
    {"HeckeOperator", "charpoly_free.coeffs", "poly", "order=low_to_high var=X ring=Z/ell^precision"},
    {"HeckeAlgebra", "hilbert", "hilbert", ""},
    {"LoewyProfile", "loewy", "layers", "units=f"},
    {"LoewyProfile", "socle", "layers", "units=f"},
    {"LoewyProfile", "jordan()", "partition", "units=f"},
    // (slot-24 NB-RENDER R3) the Z/ell^K Hecke-algebra route, the characters of a maximal ideal, the Dickson LES maps
    {"hecke_ring_algebra/v1", "blocks[].torsion", "cyclic", "ell=@ell"},
    {"hecke_ring_algebra/v1", "blocks[].A1_hilbert", "hilbert", ""},
    {"hecke_ring_algebra/v1", "blocks[].AK_hilbert", "hilbert", ""},
    {"MaximalIdeal", "character()", "eigen", "var=w f=@residue_degree"},
    {"DicksonLES", "maps.theta0", "lesmap", "map=theta0 n=@n n_up=@n_up"},
    {"DicksonLES", "maps.pi0", "lesmap", "map=pi0 n=@n n_up=@n_up"},
    {"DicksonLES", "maps.delta", "lesmap", "map=delta n=@n n_up=@n_up"},
    {"DicksonLES", "maps.theta1", "lesmap", "map=theta1 n=@n n_up=@n_up"},
    {"DicksonLES", "maps.pi1", "lesmap", "map=pi1 n=@n n_up=@n_up"},
};

std::vector<std::string> HintSegs(const std::string& path) {
    std::vector<std::string> segs;
    std::string cur;
    auto flush = [&] {
        if (!cur.empty()) segs.push_back(cur);
        cur.clear();
    };
    for (std::size_t i = 0; i < path.size(); ++i) {
        const char c = path[i];
        if (c == '.') {
            flush();
        } else if (c == '[' && i + 1 < path.size() && path[i + 1] == ']') {
            flush();
            segs.push_back("[]");
            ++i;
        } else {
            cur += c;
        }
    }
    flush();
    return segs;
}

bool HasHintRows(const std::string& name) {
    for (const RenderHintRow& r : kHintRows)
        if (name == r.name) return true;
    return false;
}

// The types rendered by the member form D-m ([P] Sec. 18.5): the generic
// members PLUS the per-type calls below, without `generic: true`.
const std::vector<std::string> kMemberFormTypes = {
    "HeckeOperator", "Pairing", "MaximalIdeal", "HeckeAlgebra", "LoewyProfile", "ExtensionClass",
    "BoundaryCohomology", "ConeCohomology", "DicksonLES", "DescentRecord",
};
bool MemberForm(const std::string& t) {
    for (const std::string& x : kMemberFormTypes)
        if (x == t) return true;
    return false;
}

// The calls of a member-form type: the method, and the List member its
// arguments come from ("" = the degrees 0 .. |dims| - 1, or 0 .. 2 when `dims`
// is not readable).  A call is made only where CallMethod allows it, so a
// declared handle makes none.
struct CallSpec {
    const char* type;
    const char* method;
    const char* args_from;
};
const std::vector<CallSpec> kCalls = {
    {"HeckeOperator", "A", ""},
    {"Pairing", "perfect", ""},
    {"MaximalIdeal", "character", "operator_primes"},
    {"MaximalIdeal", "t", "operator_primes"},
    {"LoewyProfile", "jordan", "primes"},
    {"ExtensionClass", "matrix", "active"},
};
constexpr std::size_t kMaxCalls = 16;
constexpr std::int64_t kMaxCharpolyDim = 64;

struct Builder {
    const PrintJsonContext& ctx;

    // The hint scope in force: a schema'd Record or a typed object, the path
    // below it, the enclosing Records (for the ring parameters).
    struct Scope {
        bool active = false;
        std::string name;
        std::vector<std::string> segs;
        std::vector<const Record*> recs;
        ObjectPtr obj;
    };
    Scope sc;

    bool param_of(const std::string& name, std::int64_t& out) {
        for (auto it = sc.recs.rbegin(); it != sc.recs.rend(); ++it) {
            const Value* x = (*it)->find(name);
            if (x != nullptr && x->is_int()) {
                out = x->int_value();
                return true;
            }
        }
        Value v;
        if (sc.obj && ReadMember(sc.obj, name, v) && v.is_int()) {
            out = v.int_value();
            return true;
        }
        return false;
    }

    // The hint JSON at the current path, or "".
    std::string hint_here() {
        if (!sc.active) return std::string();
        for (const RenderHintRow& r : kHintRows) {
            if (sc.name != r.name) continue;
            const std::vector<std::string> rs = HintSegs(r.path);
            if (rs.size() != sc.segs.size()) continue;
            bool ok = true;
            for (std::size_t i = 0; i < rs.size() && ok; ++i)
                ok = rs[i] == sc.segs[i] || (rs[i] == "*" && sc.segs[i] != "[]");
            if (!ok) continue;
            std::string h = "{\"kind\":" + Quote(r.kind);
            std::vector<std::string> toks;
            std::string tok;
            for (const char c : std::string(r.attrs) + " ") {
                if (c == ' ') {
                    if (!tok.empty()) toks.push_back(tok);
                    tok.clear();
                } else {
                    tok += c;
                }
            }
            for (const std::string& kv : toks) {
                const std::size_t eq = kv.find('=');
                if (eq == std::string::npos) continue;
                const std::string k = kv.substr(0, eq), val = kv.substr(eq + 1);
                std::int64_t x = 0;
                if (k != "ring") {
                    // (slot-24 NB-RENDER R3) `key=@name`: the Int field / member `name` (nearest first), else left out.
                    if (!val.empty() && val[0] == '@') {
                        if (param_of(val.substr(1), x)) h += "," + Quote(k) + ":" + IntJson(x);
                    } else {
                        h += "," + Quote(k) + ":" + Quote(val);
                    }
                    continue;
                }
                if (val == "Z") {
                    h += ",\"ring\":\"Z\"";
                } else if (val == "F_ell") {
                    h += ",\"ring\":\"F_ell\"";
                    if (param_of("ell", x)) h += ",\"ell\":" + IntJson(x);
                } else if (val.rfind("Z/ell^", 0) == 0) {
                    h += ",\"ring\":\"Z/ell^k\"";
                    if (param_of("ell", x)) h += ",\"ell\":" + IntJson(x);
                    if (param_of(val.substr(6), x)) h += ",\"k\":" + IntJson(x);
                } else {
                    h += ",\"ring\":" + Quote(val);
                }
            }
            return h + "}";
        }
        return std::string();
    }

    // `out` receives `"field":<json>` pairs, comma-separated, for an object.
    void field(std::string& out, const char* name, const std::string& json) {
        if (out.size() > 1) out += ',';
        out += Quote(name) + ":" + json;
    }
    bool member_field(std::string& out, const ObjectPtr& o, const char* member, const char* as = nullptr) {
        Value v;
        if (!ReadMember(o, member, v)) return false;
        field(out, as != nullptr ? as : member, Plain(v, ctx));
        return true;
    }

    std::string poly_tree(const ObjectPtr& o) {
        std::string t = "{";
        member_field(t, o, "ell");
        Value coeffs;
        std::int64_t deg = -1;
        Value d;
        if (ReadMember(o, "deg", d)) IntOf(d, deg);
        const bool have = CallVerb1(ctx, "coeffs", Value::from_object(o), coeffs) && coeffs.is_list();
        if (have && coeffs.list_value().empty()) deg = -1;
        field(t, "deg", std::to_string(deg));
        field(t, "order", Quote("low_to_high"));
        field(t, "coeffs", have ? Plain(coeffs, ctx) : std::string("[]"));
        return t + "}";
    }

    // The per-type tree; `truncated` is set when the tree is shape-only.
    std::string dedicated(const ObjectPtr& o, const std::string& t, bool& truncated) {
        std::string v = "{";
        if (t == "Fp") {
            member_field(v, o, "p");
            member_field(v, o, "value");
        } else if (t == "Poly") {
            return poly_tree(o);
        } else if (t == "Matrix") {
            Value r, c;
            std::int64_t rows = 0, cols = 0;
            if (ReadMember(o, "rows", r)) IntOf(r, rows);
            if (ReadMember(o, "cols", c)) IntOf(c, cols);
            field(v, "rows", std::to_string(rows));
            field(v, "cols", std::to_string(cols));
            member_field(v, o, "p");
            if (rows * cols <= ctx.max_entries) {
                std::string e = "[";
                for (std::int64_t i = 0; i < rows; ++i) {
                    if (i != 0) e += ',';
                    e += '[';
                    for (std::int64_t j = 0; j < cols; ++j) {
                        if (j != 0) e += ',';
                        Value x;
                        if (CallMethod(o, "entry", {Value::from_int(i), Value::from_int(j)}, x)) e += Plain(x, ctx);
                        else e += "null";
                    }
                    e += ']';
                }
                field(v, "entries", e + "]");
            } else {
                truncated = true;
            }
        } else if (t == "Model") {
            member_field(v, o, "name");
            member_field(v, o, "dims");
            member_field(v, o, "vertices");
        } else if (t == "Complex") {
            member_field(v, o, "name");
            Value d;
            std::int64_t dim = -1;
            if (ReadMember(o, "dim", d)) IntOf(d, dim);
            field(v, "dim", std::to_string(dim));
            member_field(v, o, "euler");
            member_field(v, o, "dims");
            std::string s = "[";
            std::int64_t entries = 0;
            bool ok = true;
            for (std::int64_t q = 0; q <= dim && ok; ++q) {
                Value sq;
                if (!CallMethod(o, "simplices", {Value::from_int(q)}, sq) || !sq.is_list()) { ok = false; break; }
                for (const Value& x : sq.list_value()) entries += x.is_list() ? static_cast<std::int64_t>(x.list_value().size()) : 1;
                if (q != 0) s += ',';
                s += Plain(sq, ctx);
            }
            if (ok && entries <= ctx.max_entries) field(v, "simplices", s + "]");
            else truncated = true;
        } else if (t == "LocalSystem") {
            member_field(v, o, "n");
            member_field(v, o, "p");
            member_field(v, o, "d_V");
            member_field(v, o, "coefficient_ring");
            if (o->computed()) member_field(v, o, "relations");
        } else if (t == "Cohomology") {
            member_field(v, o, "ell");
            member_field(v, o, "d_V");
            member_field(v, o, "coefficient_ring");
            if (o->computed()) {
                member_field(v, o, "dims");
                member_field(v, o, "degrees");
                member_field(v, o, "q_max");
                member_field(v, o, "truncated");
                Value ring;
                const bool field_ring = ReadMember(o, "coefficient_ring", ring) && ring.is_string() &&
                                        ring.string_value().rfind("F_", 0) == 0;
                if (!field_ring) {
                    member_field(v, o, "lengths");
                    member_field(v, o, "ladic_type_certified");
                    // (slot-24 NB-RENDER R1) H^q = (+)_i Z/ell^{e_i}: the exponents per degree (left out when not readable).
                    Value dims;
                    if ((ReadMember(o, "degrees", dims) && dims.is_list()) || (ReadMember(o, "lengths", dims) && dims.is_list())) {
                        std::string e = "[";
                        bool ok = true;
                        for (std::size_t q = 0; q < dims.list_value().size() && ok; ++q) {
                            Value f;
                            ok = CallMethod(o, "invariant_factors", {Value::from_int(static_cast<std::int64_t>(q))}, f) &&
                                 f.is_list();
                            if (ok) e += (q != 0 ? "," : "") + Plain(f, ctx);
                        }
                        if (ok) field(v, "invariant_factors", e + "]");
                    }
                }
            }
        } else if (t == "GF") {
            member_field(v, o, "ell");
            member_field(v, o, "degree", "r");
            Value poly;
            if (ReadMember(o, "polynomial", poly) && poly.is_object()) field(v, "polynomial", poly_tree(poly.object_value()));
        } else if (t == "GFElt") {
            Value fieldv;
            if (ReadMember(o, "field", fieldv) && fieldv.is_object()) {
                member_field(v, fieldv.object_value(), "ell");
                member_field(v, fieldv.object_value(), "degree", "r");
            }
            member_field(v, o, "coeffs");
            field(v, "order", Quote("low_to_high"));
        } else if (t == "Block") {
            // A Block has no `.json` text of its own (it is a fragment of its
            // Decomposition's record): `json` is the record of its members,
            // the basis matrix excepted (it is the Decomposition's data).
            std::string j = "{";
            for (const TypeInfo::Member& m : o->type()->members) {
                if (m.name == "basis") continue;
                Value x;
                if (ReadMember(o, m.name, x)) field(j, m.name.c_str(), Plain(x, ctx));
            }
            field(v, "json", j + "}");
        } else if (t == "Decomposition") {
            member_field(v, o, "q");
            member_field(v, o, "num_blocks", "blocks");
            Value j;
            if (ReadMember(o, "json", j) && j.is_string()) field(v, "json", EmbeddedJson(j.string_value()));
        } else if (t == "Report") {
            field(v, "schema", Quote("hecke_report/v2"));
            Value j;
            if (ReadMember(o, "json", j) && j.is_string()) field(v, "json", EmbeddedJson(j.string_value()));
        } else if (t == "ExtRecord") {
            Value s;
            field(v, "schema", ReadMember(o, "schema", s) && s.is_string() ? Quote(s.string_value())
                                                                             : Quote("hecke_ext/v5"));
            Value j;
            if (ReadMember(o, "json", j) && j.is_string()) field(v, "json", EmbeddedJson(j.string_value()));
        } else if (t == "Settings") {
            Value vals;
            std::string e = "[";
            if (ReadMember(o, "values", vals) && vals.is_record()) {
                bool first = true;
                for (const auto& kv : vals.record_value().entries()) {
                    if (!first) e += ',';
                    first = false;
                    e += "[" + Quote(kv.first) + "," + envelope(kv.second, 1) + "]";
                }
            }
            field(v, "entries", e + "]");
            member_field(v, o, "fingerprint");
        }
        return v + "}";
    }

    std::string members_json(const ObjectPtr& o, int depth) {
        std::string m = "{";
        if (o && o->type() != nullptr) {
            for (const TypeInfo::Member& mem : o->type()->members) {
                if (SkippedMember(mem.name)) continue;
                Value x;
                if (!ReadMember(o, mem.name, x)) continue;
                sc.segs.push_back(mem.name);
                field(m, mem.name.c_str(), envelope(x, depth + 1));
                sc.segs.pop_back();
            }
        }
        return m + "}";
    }

    std::string generic(const ObjectPtr& o, int depth) {
        return "{\"members\":" + members_json(o, depth) + "}";
    }

    // (slot-24 NB-RENDER R1) The member form D-m:
    //   {"members":{<name>:<envelope>,...},"calls":[[<method>,[<arg>],<envelope>],...]}
    std::string member_form(const ObjectPtr& o, const std::string& t, int depth) {
        std::string out = "{\"members\":" + members_json(o, depth) + ",\"calls\":[";
        std::size_t made = 0;
        auto emit = [&](const std::string& name, const Value& arg, const Value& result) {
            if (made != 0) out += ',';
            sc.segs.push_back(name + "()");
            out += "[" + Quote(name) + ",[" + Plain(arg, ctx) + "]," + envelope(result, depth + 1) + "]";
            sc.segs.pop_back();
            ++made;
        };
        for (const CallSpec& c : kCalls) {
            if (t != c.type) continue;
            std::vector<Value> args;
            if (*c.args_from == 0) {
                Value d;
                std::size_t nq = 3;
                if (ReadMember(o, "dims", d) && d.is_list()) nq = d.list_value().size();
                for (std::size_t q = 0; q < nq; ++q) args.push_back(Value::from_int(static_cast<std::int64_t>(q)));
            } else {
                Value l;
                if (ReadMember(o, c.args_from, l) && l.is_list()) args = l.list_value();
            }
            for (const Value& a : args) {
                if (made >= kMaxCalls) break;
                Value r;
                if (!CallMethod(o, c.method, {a}, r)) continue;
                emit(c.method, a, r);
                // HeckeOperator: the characteristic polynomial of A(q) beside it (the `charpoly` verb, a field Matrix only).
                if (t == "HeckeOperator" && made < kMaxCalls && r.is_object() && r.object_value() &&
                    r.object_value()->type() != nullptr && r.object_value()->type()->name == "Matrix") {
                    Value rows;
                    std::int64_t n = 0;
                    if (ReadMember(r.object_value(), "rows", rows)) IntOf(rows, n);
                    Value cp;
                    if (n > 0 && n <= kMaxCharpolyDim && CallVerb1(ctx, "charpoly", r, cp)) emit("charpoly(A)", a, cp);
                }
            }
        }
        return out + "]}";
    }

    std::string hash_field(const Value& v) {
        if (v.is_object()) {
            const ObjectPtr& o = v.object_value();
            if (!o || !o->computed() || o->type() == nullptr || !o->type()->hash) return "null";
        }
        if (v.is_nil() || v.is_bool() || v.is_record()) return "null";
        Value h;
        if (CallVerb1(ctx, "hash", v, h) && h.is_string()) return Quote(h.string_value());
        return "null";
    }

    std::string envelope(const Value& v, int depth) {
        // (slot-24 R3) a nil at a hinted path (an optional field left out) carries no hint: it is not the hinted object.
        const std::string hint = v.is_nil() ? std::string() : hint_here();
        const std::string t = v.is_nil() ? std::string("Nil") : v.type_name();
        std::string out = "{\"t\":" + Quote(t) + ",\"canonical\":" + Quote(PrintCanonical(v, ctx.max_entries)) +
                          ",\"hash\":" + hash_field(v) + ",\"v\":";
        bool truncated = false;
        std::int64_t length = -1;
        bool generic_form = false;
        std::string state;
        switch (v.kind()) {
            case ValueKind::Nil: out += "null"; break;
            case ValueKind::Bool: out += v.bool_value() ? "true" : "false"; break;
            case ValueKind::Int: out += IntJson(v.int_value()); break;
            case ValueKind::String: out += Quote(v.string_value()); break;
            case ValueKind::List: {
                const std::vector<Value>& items = v.list_value();
                const std::int64_t cap = ctx.max_entries;
                out += '[';
                for (std::size_t i = 0; i < items.size(); ++i) {
                    if (static_cast<std::int64_t>(i) >= cap) {
                        truncated = true;
                        length = static_cast<std::int64_t>(items.size());
                        break;
                    }
                    if (i != 0) out += ',';
                    sc.segs.push_back("[]");
                    out += envelope(items[i], depth);
                    sc.segs.pop_back();
                }
                out += ']';
                break;
            }
            case ValueKind::Record: {
                const Record& rec = v.record_value();
                const Scope saved = sc;
                const Value* schema = rec.find("schema");
                if (schema != nullptr && schema->is_string() && HasHintRows(schema->string_value())) {
                    sc = Scope();
                    sc.active = true;
                    sc.name = schema->string_value();
                }
                if (sc.active) sc.recs.push_back(&rec);
                out += '[';
                bool first = true;
                for (const auto& kv : rec.entries()) {
                    if (!first) out += ',';
                    first = false;
                    sc.segs.push_back(kv.first);
                    out += "[" + Quote(kv.first) + "," + envelope(kv.second, depth) + "]";
                    sc.segs.pop_back();
                }
                out += ']';
                sc = saved;
                break;
            }
            case ValueKind::Object: {
                const ObjectPtr& o = v.object_value();
                state = StateName(o);
                if (!o || o->type() == nullptr) {
                    out += "null";
                } else if (depth >= kMaxObjectDepth) {
                    out += "null";
                    truncated = true;
                } else {
                    const Scope saved = sc;
                    sc = Scope();
                    if (HasHintRows(t)) {
                        sc.active = true;
                        sc.name = t;
                        sc.obj = o;
                    }
                    if (MemberForm(t)) {
                        out += member_form(o, t, depth);
                    } else if (HasDedicatedSchema(t)) {
                        out += dedicated(o, t, truncated);
                    } else {
                        out += generic(o, depth);
                        generic_form = true;
                    }
                    sc = saved;
                }
                break;
            }
        }
        if (truncated) out += ",\"truncated\":true";
        if (length >= 0) out += ",\"length\":" + std::to_string(length);
        if (generic_form) out += ",\"generic\":true";
        if (!state.empty()) out += ",\"state\":" + Quote(state);
        if (!hint.empty()) out += ",\"hint\":" + hint;
        return out + "}";
    }
};

// Cuts `s` to at most `cap` bytes without splitting a UTF-8 sequence.
std::string CutUtf8(const std::string& s, std::size_t cap) {
    if (s.size() <= cap) return s;
    std::size_t n = cap;
    while (n > 0 && (static_cast<unsigned char>(s[n]) & 0xC0) == 0x80) --n;
    return s.substr(0, n);
}

} // namespace

const std::vector<std::string>& DedicatedSchemaTypes() {
    static const std::vector<std::string> types = {
        "Nil", "Bool", "Int", "String", "List", "Record",
        "Fp", "Poly", "Matrix", "Model", "Complex", "LocalSystem", "Cohomology",
        "GF", "GFElt", "Block", "Decomposition", "Report", "ExtRecord", "Settings",
        // (slot-24 NB-RENDER R1) the member form D-m
        "HeckeOperator", "Pairing", "MaximalIdeal", "HeckeAlgebra", "LoewyProfile", "ExtensionClass",
        "BoundaryCohomology", "ConeCohomology", "DicksonLES", "DescentRecord",
    };
    return types;
}

const std::vector<std::string>& MemberFormTypes() { return kMemberFormTypes; }

const std::vector<RenderHintRow>& RenderHintRows() { return kHintRows; }

bool HasDedicatedSchema(const std::string& type_name) {
    for (const std::string& t : DedicatedSchemaTypes())
        if (t == type_name) return true;
    return false;
}

std::string PrintJson(const Value& v, const PrintJsonContext& ctx) {
    Builder b{ctx, {}};
    return b.envelope(v, 0);
}

std::string HashField(const Value& v, const PrintJsonContext& ctx) {
    Builder b{ctx, {}};
    return b.hash_field(v);
}

std::string PrintJsonTruncated(const Value& v, const PrintJsonContext& ctx, std::size_t canonical_cap) {
    Builder b{ctx, {}};
    const std::string t = v.is_nil() ? std::string("Nil") : v.type_name();
    std::string out = "{\"t\":" + Quote(t) + ",\"canonical\":" +
                      Quote(CutUtf8(PrintCanonical(v, ctx.max_entries), canonical_cap)) +
                      ",\"hash\":" + b.hash_field(v) + ",\"v\":null,\"truncated\":true";
    if (v.is_object()) out += ",\"state\":" + Quote(StateName(v.object_value()));
    return out + "}";
}

} // namespace serve
} // namespace console
