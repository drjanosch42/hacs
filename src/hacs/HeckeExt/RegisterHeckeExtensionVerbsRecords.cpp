// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// RegisterHeckeExtensionVerbsRecords.cpp -- one area of the extension layer's console
// bindings (S24 CL-8 split of RegisterHeckeExtensionVerbs.cpp; the map is in
// RegisterHeckeExtensionVerbsDetail.h): the Stage 1-5 objects of one cell: ExtRecord / MaximalIdeal /
// HeckeAlgebra / LoewyProfile / ExtensionClass / Presentation.
// ===========================================================================

#include "RegisterHeckeExtensionVerbsDetail.h"

namespace hecke_ext {
namespace verbs_detail {

namespace {

// ===========================================================================
// ExtRecord / MaximalIdeal / HeckeAlgebra / LoewyProfile / ExtensionClass /
// Presentation -- the Stage 1-5 objects of one cell (HeckeExt/HeckeExtDriver).
// ===========================================================================

struct ExtRecordVal {
    uint32_t n = 0, ell = 0;
    ExtDriverOptions opts;
    std::unique_ptr<HeckeExtDriver> driver;
    void ensure() {
        if (driver) return;
        RequireField("ext_record");
        driver.reset(new HeckeExtDriver(n, ell, opts));
        driver->run();
    }
    const HeckeExtRecord& record(int side) const {
        CONSOLE_CHECK(driver != nullptr);
        if (side == 0) return driver->record();
        return driver->sign_record(side == 1);
    }
};

struct IdealVal {
    std::shared_ptr<ExtRecordVal> rec;
    int side = 0;              // 0 = the unsplit record, 1 = plus, 2 = minus
    uint32_t index = 0;        // into that record's `ideals`
    const IdealRecord& ideal() const { return rec->record(side).ideals[index]; }
};

} // namespace

struct HeckeAlgebraVal {
    std::shared_ptr<ExtRecordVal> rec;
    uint32_t index = 0;
    const LocalAlgebra* L = nullptr;
};

namespace {

struct LoewyVal {
    std::shared_ptr<ExtRecordVal> rec;
    std::string group;
    const ModuleStructure* M = nullptr;
};

struct ExtensionVal {
    std::shared_ptr<ExtRecordVal> rec;
    const StructureMap* S = nullptr;
    std::vector<uint64_t> primes;   // index-aligned with S->matrices
};

struct PresentationVal {
    std::shared_ptr<ExtRecordVal> rec;
    Presentation P;
};

ExtRecordVal& Rec(const ObjectPtr& o) { return *o->payload_as<ExtRecordVal>(); }
IdealVal& Ideal(const ObjectPtr& o) { return *o->payload_as<IdealVal>(); }
HeckeAlgebraVal& Alg(const ObjectPtr& o) { return *o->payload_as<HeckeAlgebraVal>(); }
LoewyVal& Loewy(const ObjectPtr& o) { return *o->payload_as<LoewyVal>(); }
ExtensionVal& Ext(const ObjectPtr& o) { return *o->payload_as<ExtensionVal>(); }
PresentationVal& Pres(const ObjectPtr& o) { return *o->payload_as<PresentationVal>(); }

} // namespace

const HeckeAlgebraVal* AsHeckeAlgebra(const Value& v) {
    if (!v.is_object() || !v.object_value() || v.object_value()->type() == nullptr ||
        v.object_value()->type()->name != kTypeHeckeAlgebra) {
        return nullptr;
    }
    return v.object_value()->payload_as<HeckeAlgebraVal>();
}

std::vector<ExtMatrix> HeckeAlgebraGenerators(const HeckeAlgebraVal& a) { return a.L->t; }

namespace {

Value IdealHandle(const ObjectPtr& owner, const std::shared_ptr<ExtRecordVal>& rec, int side,
                  uint32_t index) {
    auto p = std::make_shared<IdealVal>();
    p->rec = rec;
    p->side = side;
    p->index = index;
    return Value::from_object(ChildOf(TypesOf(owner).ideal, owner, p));
}

Value IdealList(const ObjectPtr& owner, const std::shared_ptr<ExtRecordVal>& rec, int side) {
    std::vector<Value> out;
    const HeckeExtRecord& r = rec->record(side);
    for (uint32_t i = 0; i < r.ideals.size(); ++i) out.push_back(IdealHandle(owner, rec, side, i));
    return Value::from_list(std::move(out));
}

Value LoewyHandle(const ObjectPtr& owner, const std::shared_ptr<ExtRecordVal>& rec,
                  const ModuleRecordEntry& e) {
    auto p = std::make_shared<LoewyVal>();
    p->rec = rec;
    p->group = e.group;
    p->M = &e.structure;
    return Value::from_object(ChildOf(TypesOf(owner).loewy, owner, p));
}

Value ExtensionHandle(const ObjectPtr& owner, const std::shared_ptr<ExtRecordVal>& rec,
                      const StructureMap& s, const std::vector<uint64_t>& primes) {
    auto p = std::make_shared<ExtensionVal>();
    p->rec = rec;
    p->S = &s;
    p->primes = primes;
    return Value::from_object(ChildOf(TypesOf(owner).extension, owner, p));
}

// The unsplit ideal's local algebra -- the driver's own (index-aligned with
// the record's ideals).  A per-sign ideal has no stored LocalAlgebra.
const LocalAlgebra& AlgebraOfIdeal(const IdealVal& iv, const SourcePos& pos, const char* who) {
    if (iv.side != 0) {
        throw RuntimeError(std::string(who) + ": the per-sign records keep no LocalAlgebra; take "
                                              "it from the unsplit record's ideal",
                           pos);
    }
    const LocalAlgebra& L = iv.rec->driver->local_algebra(iv.index);
    if (!L.ok) {
        throw RuntimeError(std::string(who) + ": ideal " + std::to_string(iv.index) +
                               " is not certified local, so it has no local algebra",
                           pos);
    }
    return L;
}

Value AlgebraHandle(const ObjectPtr& owner, const IdealVal& iv, const SourcePos& pos, const char* who) {
    auto p = std::make_shared<HeckeAlgebraVal>();
    p->rec = iv.rec;
    p->index = iv.index;
    p->L = &AlgebraOfIdeal(iv, pos, who);
    return Value::from_object(ChildOf(TypesOf(owner).algebra, owner, p));
}

Value PresentationHandle(const ObjectPtr& owner, const IdealVal& iv, const SourcePos& pos,
                         const char* who) {
    const LocalAlgebra& L = AlgebraOfIdeal(iv, pos, who);
    auto p = std::make_shared<PresentationVal>();
    p->rec = iv.rec;
    p->P = ComputePresentation(L);
    return Value::from_object(ChildOf(TypesOf(owner).presentation, owner, p));
}

Value ResumableRecordValue(const ResumableRecord& r) {
    Record out;
    out.set("present", Value::from_bool(r.present));
    out.set("kind", Value::from_string(r.kind));
    out.set("complete", Value::from_bool(r.complete));
    out.set("budget", Value::from_string(r.budget_json));
    out.set("reached", Value::from_string(r.reached_json));
    out.set("budget_history", Value::from_string(r.budget_history_json));
    return Value::from_record(std::move(out));
}

} // namespace

Value VerbExtRecord(CallContext& ctx, const std::vector<Value>& a) {
    static const char* kWho = "ext_record(n, ell, cone, sign_split, presentation, sturm, commutant)";
    RequireField("ext_record");
    auto v = std::make_shared<ExtRecordVal>();
    v->n = console::bind::U32Arg(a[0], ctx.pos, kWho);
    v->ell = console::bind::U32Arg(a[1], ctx.pos, kWho);
    if (v->ell < 2 || !hecke::IsPrimeU64(v->ell)) throw TypeError(std::string(kWho) + ": ell must be a prime >= 2", ctx.pos);
    auto flag = [&](size_t i, bool dflt) {
        return (a.size() > i && a[i].is_bool()) ? a[i].bool_value() : dflt;
    };
    v->opts.with_cone = flag(2, false);
    v->opts.with_sign_split = flag(3, false);
    v->opts.with_presentation = flag(4, false);
    v->opts.run_sturm = flag(5, true);
    v->opts.run_commutant = flag(6, true);
    ObjectPtr o = Object::create(console::bind::TypeByName(ctx, kTypeExtRecord), ctx.pos,
                                 ctx.call_text, ctx.session->settings().snapshot(), v,
                                 console::bind::LazyCompute([v](Object&) { v->ensure(); }));
    return Value::from_object(o);
}

Value VerbLocalAlgebra(CallContext& ctx, const std::vector<Value>& a) {
    static const char* kWho = "local_algebra(m)";
    const ObjectPtr& m = console::bind::AsObjectOf(a[0], kTypeMaximalIdeal, ctx.pos, kWho);
    return AlgebraHandle(m, Ideal(m), ctx.pos, kWho);
}

Value VerbModuleStructure(CallContext& ctx, const std::vector<Value>& a) {
    static const char* kWho = "module_structure(m, group)";
    const ObjectPtr& m = console::bind::AsObjectOf(a[0], kTypeMaximalIdeal, ctx.pos, kWho);
    const IdealVal& iv = Ideal(m);
    const IdealRecord& id = iv.ideal();
    if (id.modules.empty()) {
        throw RuntimeError(std::string(kWho) + ": ideal " + std::to_string(id.id) +
                               " carries no module structure (it is not certified local)",
                           ctx.pos);
    }
    if (a.size() < 2 || a[1].is_nil()) return LoewyHandle(m, iv.rec, id.modules.front());
    if (!a[1].is_string()) throw TypeError(std::string(kWho) + ": `group` is a String", ctx.pos);
    for (const ModuleRecordEntry& e : id.modules) {
        if (e.group == a[1].string_value()) return LoewyHandle(m, iv.rec, e);
    }
    throw RuntimeError(std::string(kWho) + ": ideal " + std::to_string(id.id) +
                           " has no module structure in group '" + a[1].string_value() + "'",
                       ctx.pos);
}

Value VerbPresentation(CallContext& ctx, const std::vector<Value>& a) {
    static const char* kWho = "presentation(m)";
    const ObjectPtr& m = console::bind::AsObjectOf(a[0], kTypeMaximalIdeal, ctx.pos, kWho);
    return Guard(kWho, ctx.pos, [&] { return PresentationHandle(m, Ideal(m), ctx.pos, kWho); });
}

void RegisterExtRecordTypes(Registry& registry) {
    // --- ExtRecord --------------------------------------------------------
    {
        TypeInfo t;
        t.name = kTypeExtRecord;
        t.doc = "one cell's hecke_ext/v5 record (HeckeExtDriver, Stages 1-5): its maximal ideals, "
                "groups, u_ell verdict and settings; lazy";
        auto R = [](const ObjectPtr& o) -> const HeckeExtRecord& { return Rec(o).record(0); };
        auto shared = [](const ObjectPtr& o) {
            return std::shared_ptr<ExtRecordVal>(o->payload(), &Rec(o));
        };
        t.members.push_back({"n", false, "the symmetric power n", [](const ObjectPtr& o) {
                                 return Value::from_int(Rec(o).n);
                             }});
        t.members.push_back({"ell", false, "the coefficient prime", [](const ObjectPtr& o) {
                                 return Value::from_int(Rec(o).ell);
                             }});
        t.members.push_back({"schema", true, "the record schema", [R](const ObjectPtr& o) {
                                 return Value::from_string(R(o).schema);
                             }});
        t.members.push_back({"weight", true, "k = n + 2", [R](const ObjectPtr& o) {
                                 return Value::from_int(R(o).weight);
                             }});
        t.members.push_back({"coefficient_ring", true, "the record's ring (F_ell)",
                             [R](const ObjectPtr& o) {
                                 return Value::from_string(R(o).coefficient_ring);
                             }});
        t.members.push_back({"dims", true, "dim H^q of the bulk, every degree",
                             [](const ObjectPtr& o) {
                                 return U32List(Rec(o).driver->H_dimensions());
                             }});
        t.members.push_back({"degrees_run", true, "the bulk degrees Stages 2-5 ran on",
                             [](const ObjectPtr& o) {
                                 return U32List(Rec(o).driver->degrees_run());
                             }});
        t.members.push_back({"operators", true, "the primes whose operators Stage 1 computed",
                             [R](const ObjectPtr& o) { return U64List(R(o).operators_computed); }});
        t.members.push_back({"anemic_pool", true, "the anemic pool {q <= 31} \\ {ell}",
                             [R](const ObjectPtr& o) { return U64List(R(o).anemic_pool); }});
        t.members.push_back({"s_sep", true, "S_sep(ell) as Stage 2(3) ran it (empty if not computed)",
                             [R](const ObjectPtr& o) { return U64List(R(o).s_sep.primes); }});
        t.members.push_back({"dim_M_k", true, "dim M_k", [R](const ObjectPtr& o) {
                                 return Value::from_int(R(o).dim_M_k);
                             }});
        t.members.push_back({"dim_S_k", true, "dim S_k", [R](const ObjectPtr& o) {
                                 return Value::from_int(R(o).dim_S_k);
                             }});
        t.members.push_back({"dim_dickson", true, "the Dickson count", [R](const ObjectPtr& o) {
                                 return Value::from_int(R(o).dim_dickson);
                             }});
        t.members.push_back({"H1_shriek", true, "dim H^1_! (cone = true), else nil",
                             [R](const ObjectPtr& o) {
                                 return R(o).has_dim_H1_shriek ? Value::from_int(R(o).dim_H1_shriek)
                                                               : Value::nil();
                             }});
        t.members.push_back({"les_exact", true, "the cone's LES exact (cone = true), else nil",
                             [R](const ObjectPtr& o) {
                                 return R(o).les_available ? Value::from_bool(R(o).les_exact)
                                                           : Value::nil();
                             }});
        t.members.push_back({"groups", true, "the record's groups, {name, available, dim}",
                             [R](const ObjectPtr& o) {
                                 std::vector<Value> out;
                                 for (const GroupRecord& g : R(o).groups) {
                                     Record gr;
                                     gr.set("name", Value::from_string(g.name));
                                     gr.set("available", Value::from_bool(g.available));
                                     gr.set("dim", Value::from_int(g.dim));
                                     out.push_back(Value::from_record(std::move(gr)));
                                 }
                                 return Value::from_list(std::move(out));
                             }});
        t.members.push_back({"u_ell", true, "the cell's U_ell verdict and its resumable state",
                             [R](const ObjectPtr& o) {
                                 const UEllRecord& u = R(o).u_ell;
                                 Record ur;
                                 ur.set("included", Value::from_bool(u.included));
                                 ur.set("reason", Value::from_string(u.reason));
                                 ur.set("budget_mb", Value::from_int(u.budget_mb));
                                 ur.set("resumable", ResumableRecordValue(u.resumable));
                                 return Value::from_record(std::move(ur));
                             }});
        t.members.push_back({"ideals", true, "the maximal ideals, as MaximalIdeal handles",
                             [shared](const ObjectPtr& o) { return IdealList(o, shared(o), 0); }});
        t.members.push_back({"num_ideals", true, "the number of maximal ideals",
                             [R](const ObjectPtr& o) {
                                 return Value::from_int(static_cast<std::int64_t>(R(o).ideals.size()));
                             }});
        t.members.push_back({"has_sign_records", true, "whether sign_split = true built H^+ / H^-",
                             [](const ObjectPtr& o) {
                                 return Value::from_bool(Rec(o).driver->has_sign_records());
                             }});
        t.members.push_back({"ideals_plus", true, "the ideals of the H^+ record (sign_split = true)",
                             [shared](const ObjectPtr& o) {
                                 if (!Rec(o).driver->has_sign_records())
                                     throw RuntimeError("ExtRecord.ideals_plus: build with sign_split = true",
                                                        SourcePos());
                                 return IdealList(o, shared(o), 1);
                             }});
        t.members.push_back({"ideals_minus", true, "the ideals of the H^- record (sign_split = true)",
                             [shared](const ObjectPtr& o) {
                                 if (!Rec(o).driver->has_sign_records())
                                     throw RuntimeError("ExtRecord.ideals_minus: build with sign_split = true",
                                                        SourcePos());
                                 return IdealList(o, shared(o), 2);
                             }});
        t.members.push_back({"settings", true, "the MathSettings the record was computed under",
                             [R](const ObjectPtr& o) { return SettingsHandleFrom(o, R(o).settings); }});
        t.members.push_back({"verify", true, "HeckeExtRecord::verify() (0 = consistent)",
                             [R](const ObjectPtr& o) { return Value::from_int(R(o).verify()); }});
        t.members.push_back({"json", true, "the hecke_ext/v5 record text (carries the build's commit)",
                             [R](const ObjectPtr& o) {
                                 return Value::from_string(BuildHeckeExtReportJson(R(o)));
                             }});
        t.members.push_back({"digest", true,
                             "FNV-1a-64 of the record text with `commit` redacted (NOT the golden "
                             "ext_json_* hash)",
                             [R](const ObjectPtr& o) {
                                 HeckeExtRecord copy = R(o);
                                 copy.commit = "<redacted>";
                                 return Value::from_string(oc_util::HexU64(
                                     engine::FnvBytes(engine::kFnvOffset64, BuildHeckeExtReportJson(copy))));
                             }});
        {
            TypeInfo::Method m;
            m.name = "ideal";
            m.params = {"i"};
            m.required = 1;
            m.doc = "the i-th maximal ideal";
            m.call = [R, shared](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                const uint32_t i = console::bind::U32Arg(args[0], pos, "ExtRecord.ideal(i)");
                if (i >= R(o).ideals.size())
                    throw RuntimeError("ExtRecord.ideal(i): no ideal " + std::to_string(i), pos);
                return IdealHandle(o, shared(o), 0, i);
            };
            t.methods.push_back(std::move(m));
        }
        t.print = [](const ObjectPtr& o, std::int64_t) {
            const ExtRecordVal& v = Rec(o);
            std::ostringstream os;
            os << "ExtRecord (" << v.n << ", " << v.ell << ")";
            if (!o->computed()) os << " (declared, not yet computed)";
            else os << ": " << v.record(0).ideals.size() << " maximal ideal(s)";
            return os.str();
        };
        registry.register_type(std::move(t));
    }
    // --- MaximalIdeal -----------------------------------------------------
    {
        TypeInfo t;
        t.name = kTypeMaximalIdeal;
        t.doc = "one maximal ideal m of a cell's Hecke algebra (an ideal of the ExtRecord)";
        auto I = [](const ObjectPtr& o) -> const IdealRecord& { return Ideal(o).ideal(); };
        t.members.push_back({"id", true, "the ideal's id in its record", [I](const ObjectPtr& o) {
                                 return Value::from_int(I(o).id);
                             }});
        t.members.push_back({"sign", true, "all / plus / minus", [I](const ObjectPtr& o) {
                                 return Value::from_string(I(o).sign);
                             }});
        t.members.push_back({"dim", true, "dim_F_ell of the block H_m", [I](const ObjectPtr& o) {
                                 return Value::from_int(I(o).dim);
                             }});
        t.members.push_back({"group", true, "the ambient group of the block", [I](const ObjectPtr& o) {
                                 for (const IdealGroupMatch& g : I(o).groups)
                                     if (g.own) return Value::from_string(g.group);
                                 return I(o).modules.empty() ? Value::nil()
                                                             : Value::from_string(I(o).modules.front().group);
                             }});
        t.members.push_back({"local", true, "certified local", [I](const ObjectPtr& o) {
                                 return Value::from_bool(I(o).local);
                             }});
        t.members.push_back({"residue_degree", true, "f = [T/m : F_ell]", [I](const ObjectPtr& o) {
                                 return Value::from_int(I(o).residue_degree);
                             }});
        t.members.push_back({"residue_degree_certified", true, "whether f is certified",
                             [I](const ObjectPtr& o) {
                                 return Value::from_bool(I(o).residue_degree_certified);
                             }});
        t.members.push_back({"witness_poly", true, "pi_w, the witness's minimal polynomial",
                             [I](const ObjectPtr& o) {
                                 const IdealRecord& id = I(o);
                                 return PolyFrom(o, hecke::FpPoly(static_cast<uint32_t>(Ideal(o).rec->ell),
                                                                  id.witness_poly));
                             }});
        t.members.push_back({"w_hat_exponent", true, "N of the Teichmueller lift (local only)",
                             [I](const ObjectPtr& o) {
                                 return I(o).coefficient_field.present
                                            ? Value::from_int(I(o).coefficient_field.w_hat_exponent)
                                            : Value::nil();
                             }});
        t.members.push_back({"final", true, "true / false / heuristic", [I](const ObjectPtr& o) {
                                 return Value::from_string(ToString(I(o).final.value));
                             }});
        t.members.push_back({"final_reason", true, "why not final ('' when final)",
                             [I](const ObjectPtr& o) { return Value::from_string(I(o).final.reason); }});
        t.members.push_back({"certificate", true,
                             "cyclic / commutant / sturm(B') / dickson / boundary / unavailable",
                             [I](const ObjectPtr& o) { return Value::from_string(I(o).certificate); }});
        t.members.push_back({"certificate_routes", true,
                             "the routes' outcomes {a, b, c, d0, d1} (d0 / d1: the finality certificates, hecke_ext/v4)",
                             [I](const ObjectPtr& o) {
                                 Record r;
                                 r.set("a", Value::from_string(I(o).certificate_routes.a));
                                 r.set("b", Value::from_string(I(o).certificate_routes.b));
                                 r.set("c", Value::from_string(I(o).certificate_routes.c));
                                 r.set("d0", Value::from_string(I(o).certificate_routes.d0));
                                 r.set("d1", Value::from_string(I(o).certificate_routes.d1));
                                 return Value::from_record(std::move(r));
                             }});
        t.members.push_back({"generators_certified", true, "the certified generating primes",
                             [I](const ObjectPtr& o) { return U64List(I(o).generators_certified); }});
        t.members.push_back({"boundary_reps", true,
                             "R(ell), the class representatives (d1) tested (empty unless it reached them)",
                             [I](const ObjectPtr& o) { return U64List(I(o).boundary_reps); }});
        t.members.push_back({"operator_primes", true, "the primes of the restricted operators",
                             [I](const ObjectPtr& o) { return U64List(I(o).operator_primes); }});
        t.members.push_back({"tangent_active", true, "the primes whose n_p is active in m/m^2",
                             [I](const ObjectPtr& o) {
                                 std::vector<uint64_t> out;
                                 const IdealRecord& id = I(o);
                                 for (size_t k = 0; k < id.tangent_primes.size(); ++k)
                                     if (k < id.tangent_active.size() && id.tangent_active[k])
                                         out.push_back(id.tangent_primes[k]);
                                 return U64List(out);
                             }});
        t.members.push_back({"embdim", true, "dim_K m/m^2 (local only)", [I](const ObjectPtr& o) {
                                 return I(o).algebra_present ? Value::from_int(I(o).embdim) : Value::nil();
                             }});
        t.members.push_back({"nilpotency", true, "the nilpotency index nu (local only)",
                             [I](const ObjectPtr& o) {
                                 return I(o).algebra_present ? Value::from_int(I(o).nilpotency)
                                                             : Value::nil();
                             }});
        t.members.push_back({"gorenstein", true, "dim soc = f (local only)", [I](const ObjectPtr& o) {
                                 return I(o).algebra_present ? Value::from_bool(I(o).gorenstein)
                                                             : Value::nil();
                             }});
        t.members.push_back({"label", true, "the atlas label {type, system, theta_twist, ...} or nil",
                             [I](const ObjectPtr& o) {
                                 const LabelRecord& l = I(o).label;
                                 if (!l.computed) return Value::nil();
                                 Record r;
                                 r.set("type", Value::from_string(l.type));
                                 r.set("system", Value::from_int(l.system));
                                 r.set("r", Value::from_int(l.r));
                                 r.set("base", Value::from_int(l.base));
                                 r.set("theta_twist", Value::from_int(l.theta_twist));
                                 r.set("chi", l.has_chi ? U32List({l.chi_a, l.chi_b}) : Value::nil());
                                 return Value::from_record(std::move(r));
                             }});
        t.members.push_back({"predicted", true, "the predictions attached to the ideal",
                             [I](const ObjectPtr& o) { return StringList(I(o).predicted); }});
        t.members.push_back({"unpredicted", true, "what was found and not predicted",
                             [I](const ObjectPtr& o) { return StringList(I(o).unpredicted); }});
        t.members.push_back({"modules", true, "the LoewyProfile of every group the ideal lives in",
                             [](const ObjectPtr& o) {
                                 const IdealVal& iv = Ideal(o);
                                 std::vector<Value> out;
                                 for (const ModuleRecordEntry& e : iv.ideal().modules)
                                     out.push_back(LoewyHandle(o, iv.rec, e));
                                 return Value::from_list(std::move(out));
                             }});
        t.members.push_back({"algebra", true, "the local Hecke algebra T_m (HeckeAlgebra)",
                             [](const ObjectPtr& o) {
                                 return AlgebraHandle(o, Ideal(o), SourcePos(), "MaximalIdeal.algebra");
                             }});
        {
            TypeInfo::Method m;
            m.name = "t";
            m.params = {"p"};
            m.required = 1;
            m.doc = "the restricted operator t_p = T_p|_{H_m}";
            m.call = [I](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                const uint64_t p = console::bind::U64Arg(args[0], pos, "MaximalIdeal.t(p)");
                const IdealRecord& id = I(o);
                for (size_t k = 0; k < id.operator_primes.size(); ++k)
                    if (id.operator_primes[k] == p && k < id.t_p.size()) return MatrixFrom(o, id.t_p[k]);
                throw RuntimeError("MaximalIdeal.t(p): p = " + std::to_string(p) +
                                       " is not an operator prime of this ideal",
                                   pos);
            };
            t.methods.push_back(std::move(m));
        }
        {
            TypeInfo::Method m;
            m.name = "character";
            m.params = {"p"};
            m.required = 1;
            m.doc = "P_p, the eigenvalue of T_p as a polynomial in the witness (local only)";
            m.call = [I](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                const uint64_t p = console::bind::U64Arg(args[0], pos, "MaximalIdeal.character(p)");
                const CoefficientFieldRecord& cf = I(o).coefficient_field;
                for (size_t k = 0; k < cf.primes.size(); ++k)
                    if (cf.primes[k] == p)
                        return PolyFrom(o, hecke::FpPoly(static_cast<uint32_t>(Ideal(o).rec->ell), cf.P[k]));
                throw RuntimeError("MaximalIdeal.character(p): no character at p = " +
                                       std::to_string(p) + " (the ideal is not local, or p was not run)",
                                   pos);
            };
            t.methods.push_back(std::move(m));
        }
        t.print = [](const ObjectPtr& o, std::int64_t) {
            const IdealRecord& id = Ideal(o).ideal();
            std::ostringstream os;
            os << "MaximalIdeal #" << id.id << " (" << id.sign << "): dim " << id.dim << ", f = "
               << id.residue_degree << (id.local ? ", local" : ", NOT certified local");
            return os.str();
        };
        registry.register_type(std::move(t));
    }
    // --- HeckeAlgebra -----------------------------------------------------
    {
        TypeInfo t;
        t.name = kTypeHeckeAlgebra;
        t.doc = "the local Hecke algebra T_m = A_S of a certified-local ideal (HeckeLocalAlgebra; "
                "[X] Sec. 7's `LocalAlgebra`)";
        auto L = [](const ObjectPtr& o) -> const LocalAlgebra& { return *Alg(o).L; };
        t.members.push_back({"ell", true, "the prime", [L](const ObjectPtr& o) { return Value::from_int(L(o).ell); }});
        t.members.push_back({"D", true, "dim_F_ell of the block", [L](const ObjectPtr& o) { return Value::from_int(L(o).D); }});
        t.members.push_back({"f", true, "the residue degree", [L](const ObjectPtr& o) { return Value::from_int(L(o).f); }});
        t.members.push_back({"dim", true, "dim_F_ell A_S", [L](const ObjectPtr& o) { return Value::from_int(L(o).dim_A); }});
        t.members.push_back({"dim_m", true, "dim_F_ell m_S", [L](const ObjectPtr& o) { return Value::from_int(L(o).dim_m_S); }});
        t.members.push_back({"nilpotency", true, "nu: m^nu = 0 != m^{nu-1}", [L](const ObjectPtr& o) { return Value::from_int(L(o).nu); }});
        t.members.push_back({"hilbert", true, "dim(m^i/m^{i+1})/f", [L](const ObjectPtr& o) { return U32List(L(o).hilbert); }});
        t.members.push_back({"socle_dim", true, "dim_F_ell soc A_S", [L](const ObjectPtr& o) { return Value::from_int(L(o).dim_socle); }});
        t.members.push_back({"gorenstein", true, "dim soc A_S == f", [L](const ObjectPtr& o) { return Value::from_bool(L(o).gorenstein); }});
        t.members.push_back({"embdim", true, "dim_K m/m^2", [L](const ObjectPtr& o) { return Value::from_int(L(o).embdim); }});
        t.members.push_back({"primes", true, "S, ascending", [L](const ObjectPtr& o) { return U64List(L(o).primes); }});
        t.members.push_back({"active_set", true, "the primes whose n_p-bar is nonzero",
                             [L](const ObjectPtr& o) {
                                 std::vector<uint64_t> out;
                                 for (size_t k = 0; k < L(o).primes.size(); ++k)
                                     if (L(o).active[k]) out.push_back(L(o).primes[k]);
                                 return U64List(out);
                             }});
        t.members.push_back({"minimal_active", true, "a K-spanning subset, greedy in ascending p",
                             [L](const ObjectPtr& o) { return U64List(L(o).minimal_active); }});
        t.members.push_back({"min_generators", true, "dim(V/m V)/f", [L](const ObjectPtr& o) { return Value::from_int(L(o).min_generators); }});
        t.members.push_back({"cyclic", true, "certificate (a): V cyclic over A_S", [L](const ObjectPtr& o) { return Value::from_bool(L(o).cyclicity.cyclic); }});
        t.members.push_back({"verify", true, "certificate (a)'s verify() (0 = re-derives)",
                             [L](const ObjectPtr& o) { return Value::from_int(L(o).cyclicity.verify()); }});
        t.members.push_back({"basis", true, "an F_ell-basis of A_S (canonical RREF)",
                             [L](const ObjectPtr& o) { return MatrixListFrom(o, L(o).basis); }});
        t.members.push_back({"radical", true, "an F_ell-basis of m_S",
                             [L](const ObjectPtr& o) {
                                 return MatrixListFrom(o, L(o).m_powers.empty() ? std::vector<ExtMatrix>()
                                                                                : L(o).m_powers[0]);
                             }});
        t.members.push_back({"w_hat", true, "the Teichmueller lift of the witness", [L](const ObjectPtr& o) { return MatrixFrom(o, L(o).w_hat); }});
        t.members.push_back({"invariants", true,
                             "the SAME invariants by the certificate-free route of "
                             "algebra_invariants(t_p) -- a cross-check",
                             [L](const ObjectPtr& o) {
                                 return InvariantsRecord(ComputeAlgebraInvariants(L(o).t, L(o).ell));
                             }});
        t.members.push_back({"json", true, "the LocalAlgebra JSON", [L](const ObjectPtr& o) { return Value::from_string(L(o).to_json()); }});
        auto per_prime = [](const char* name, const char* doc, bool nil_part) {
            TypeInfo::Method m;
            m.name = name;
            m.params = {"p"};
            m.required = 1;
            m.doc = doc;
            m.call = [nil_part, name](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                const LocalAlgebra& A = *Alg(o).L;
                const uint64_t p = console::bind::U64Arg(args[0], pos, name);
                for (size_t k = 0; k < A.primes.size(); ++k)
                    if (A.primes[k] == p) return MatrixFrom(o, nil_part ? A.n[k] : A.t[k]);
                throw RuntimeError(std::string("HeckeAlgebra.") + name + "(p): p = " +
                                       std::to_string(p) + " is not in S",
                                   pos);
            };
            return m;
        };
        t.methods.push_back(per_prime("t", "t_p = T_p restricted to the block", false));
        t.methods.push_back(per_prime("n", "n_p = t_p - P_p(w_hat), the nilpotent generator", true));
        t.print = [](const ObjectPtr& o, std::int64_t) {
            const LocalAlgebra& A = *Alg(o).L;
            std::ostringstream os;
            os << "HeckeAlgebra: dim " << A.dim_A << " over F_" << A.ell << ", f = " << A.f
               << ", embdim " << A.embdim << ", nu = " << A.nu
               << (A.gorenstein ? ", Gorenstein" : ", not Gorenstein");
            return os.str();
        };
        registry.register_type(std::move(t));
    }
    // --- LoewyProfile -----------------------------------------------------
    {
        TypeInfo t;
        t.name = kTypeLoewyProfile;
        t.doc = "the Stage-5 module structure of a block at m (HeckeModuleStructure; [X] Sec. 7's "
                "`ModuleStructure`): Loewy / socle series, Jordan types, structure maps";
        auto M = [](const ObjectPtr& o) -> const ModuleStructure& { return *Loewy(o).M; };
        t.members.push_back({"group", false, "the ambient group", [](const ObjectPtr& o) { return Value::from_string(Loewy(o).group); }});
        t.members.push_back({"D", true, "dim_F_ell V", [M](const ObjectPtr& o) { return Value::from_int(M(o).D); }});
        t.members.push_back({"f", true, "the residue degree", [M](const ObjectPtr& o) { return Value::from_int(M(o).f); }});
        t.members.push_back({"loewy", true, "Loewy layer dimensions in units of f", [M](const ObjectPtr& o) { return U32List(M(o).loewy); }});
        t.members.push_back({"socle", true, "socle layer dimensions in units of f", [M](const ObjectPtr& o) { return U32List(M(o).socle); }});
        t.members.push_back({"m_dims", true, "dim_F_ell m^j V, j = 0 .. nu_V", [M](const ObjectPtr& o) { return U32List(M(o).m_dims); }});
        t.members.push_back({"socle_dims", true, "dim_F_ell soc^j V", [M](const ObjectPtr& o) { return U32List(M(o).socle_dims); }});
        t.members.push_back({"primes", true, "the generators' primes", [M](const ObjectPtr& o) { return U64List(M(o).primes); }});
        t.members.push_back({"cyclic", true, "V cyclic over T_m", [M](const ObjectPtr& o) { return Value::from_bool(M(o).cyclic); }});
        t.members.push_back({"free", true, "V free of rank one", [M](const ObjectPtr& o) { return Value::from_bool(M(o).free_rank_one); }});
        t.members.push_back({"min_generators", true, "dim(V/mV)/f", [M](const ObjectPtr& o) { return Value::from_int(M(o).min_generators); }});
        t.members.push_back({"semisimple", true, "mV = 0 (no extension at m)", [M](const ObjectPtr& o) { return Value::from_bool(M(o).loewy.size() <= 1); }});
        t.members.push_back({"verify", true, "ModuleStructure::verify() (0 = re-derives)", [M](const ObjectPtr& o) { return Value::from_int(M(o).verify()); }});
        t.members.push_back({"json", true, "the module structure JSON", [M](const ObjectPtr& o) { return Value::from_string(M(o).to_json()); }});
        t.members.push_back({"structure_maps", true, "the ExtensionClass of every consecutive layer pair",
                             [M](const ObjectPtr& o) {
                                 std::vector<Value> out;
                                 for (const StructureMap& s : M(o).structure_maps)
                                     out.push_back(ExtensionHandle(o, Loewy(o).rec, s, M(o).primes));
                                 return Value::from_list(std::move(out));
                             }});
        {
            TypeInfo::Method m;
            m.name = "jordan";
            m.params = {"p"};
            m.required = 1;
            m.doc = "the Jordan type of n_p on V, in units of f";
            m.call = [M](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                const uint64_t p = console::bind::U64Arg(args[0], pos, "LoewyProfile.jordan(p)");
                for (size_t k = 0; k < M(o).primes.size(); ++k)
                    if (M(o).primes[k] == p) return U32List(M(o).jordan[k]);
                throw RuntimeError("LoewyProfile.jordan(p): p = " + std::to_string(p) +
                                       " is not a generator prime",
                                   pos);
            };
            t.methods.push_back(std::move(m));
        }
        {
            TypeInfo::Method m;
            m.name = "layer";
            m.params = {"j"};
            m.required = 1;
            m.doc = "the canonical basis of the Loewy layer m^j V / m^{j+1} V (columns)";
            m.call = [M](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                const uint32_t j = DegreeArgC5(args[0], pos, "LoewyProfile.layer(j)", M(o).layer_basis.size());
                return MatrixFrom(o, M(o).layer_basis[j]);
            };
            t.methods.push_back(std::move(m));
        }
        {
            TypeInfo::Method m;
            m.name = "structure_map";
            m.params = {"j"};
            m.required = 1;
            m.doc = "the ExtensionClass of the layer pair (j, j+1)";
            m.call = [M](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                const uint32_t j = DegreeArgC5(args[0], pos, "LoewyProfile.structure_map(j)",
                                               M(o).structure_maps.size());
                return ExtensionHandle(o, Loewy(o).rec, M(o).structure_maps[j], M(o).primes);
            };
            t.methods.push_back(std::move(m));
        }
        t.print = [](const ObjectPtr& o, std::int64_t) {
            const ModuleStructure& m = *Loewy(o).M;
            std::ostringstream os;
            os << "LoewyProfile on " << Loewy(o).group << ": loewy [";
            for (size_t i = 0; i < m.loewy.size(); ++i) os << (i ? ", " : "") << m.loewy[i];
            os << "], socle [";
            for (size_t i = 0; i < m.socle.size(); ++i) os << (i ? ", " : "") << m.socle[i];
            os << "]" << (m.cyclic ? ", cyclic" : ", not cyclic");
            return os.str();
        };
        registry.register_type(std::move(t));
    }
    // --- ExtensionClass ---------------------------------------------------
    {
        TypeInfo t;
        t.name = kTypeExtension;
        t.doc = "the structure map theta_j between two consecutive Loewy layers ([E] D4): the "
                "extension class the generators n_p glue the layers by";
        auto S = [](const ObjectPtr& o) -> const StructureMap& { return *Ext(o).S; };
        t.members.push_back({"top_layer", true, "j", [S](const ObjectPtr& o) { return Value::from_int(S(o).top_layer); }});
        t.members.push_back({"bottom_layer", true, "j + 1", [S](const ObjectPtr& o) { return Value::from_int(S(o).bottom_layer); }});
        t.members.push_back({"top_dim", true, "the top layer's K-dimension", [S](const ObjectPtr& o) { return Value::from_int(S(o).top_dim); }});
        t.members.push_back({"bottom_dim", true, "the bottom layer's K-dimension", [S](const ObjectPtr& o) { return Value::from_int(S(o).bottom_dim); }});
        t.members.push_back({"rank", true, "F_ell rank of the stacked map", [S](const ObjectPtr& o) { return Value::from_int(S(o).rank); }});
        t.members.push_back({"rank_over_f", true, "its K-rank", [S](const ObjectPtr& o) { return Value::from_int(S(o).rank_over_f); }});
        t.members.push_back({"active", true, "the primes whose own map is nonzero", [S](const ObjectPtr& o) { return U64List(S(o).active); }});
        t.members.push_back({"rank_p", true, "the F_ell rank per generator", [S](const ObjectPtr& o) { return U32List(S(o).rank_p); }});
        t.members.push_back({"split", true, "rank == 0: the two layers are not glued",
                             [S](const ObjectPtr& o) { return Value::from_bool(S(o).rank == 0); }});
        {
            TypeInfo::Method m;
            m.name = "matrix";
            m.params = {"p"};
            m.required = 1;
            m.doc = "the matrix of n_p between the layers (rows bottom, columns top)";
            m.call = [S](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                const uint64_t p = console::bind::U64Arg(args[0], pos, "ExtensionClass.matrix(p)");
                const ExtensionVal& e = Ext(o);
                for (size_t k = 0; k < e.primes.size(); ++k)
                    if (e.primes[k] == p && k < S(o).matrices.size()) return MatrixFrom(o, S(o).matrices[k]);
                throw RuntimeError("ExtensionClass.matrix(p): p = " + std::to_string(p) +
                                       " is not a generator prime",
                                   pos);
            };
            t.methods.push_back(std::move(m));
        }
        t.print = [](const ObjectPtr& o, std::int64_t) {
            const StructureMap& s = *Ext(o).S;
            std::ostringstream os;
            os << "ExtensionClass layers (" << s.top_layer << ", " << s.bottom_layer << "): rank "
               << s.rank << (s.rank == 0 ? " (split)" : " (non-split)");
            return os.str();
        };
        registry.register_type(std::move(t));
    }
    // --- Presentation -----------------------------------------------------
    {
        TypeInfo t;
        t.name = kTypePresentation;
        t.doc = "generators and a minimal relation set of a certified-local block over T_m "
                "(HeckePresentation, [X] Sec. 2.2 Stage 5, OQ11)";
        auto P = [](const ObjectPtr& o) -> const Presentation& { return Pres(o).P; };
        t.members.push_back({"available", false, "whether the block was local", [P](const ObjectPtr& o) { return Value::from_bool(P(o).available); }});
        t.members.push_back({"min_generators", false, "t, over A_S", [P](const ObjectPtr& o) { return Value::from_int(P(o).min_generators); }});
        t.members.push_back({"min_relations", false, "over A_S", [P](const ObjectPtr& o) { return Value::from_int(P(o).min_relations); }});
        t.members.push_back({"dim_relation_module", false, "t dim A_S - D", [P](const ObjectPtr& o) { return Value::from_int(P(o).dim_relation_module); }});
        t.members.push_back({"free", false, "no relations", [P](const ObjectPtr& o) { return Value::from_bool(P(o).is_free()); }});
        t.members.push_back({"generator_indices", false, "the generators' standard indices", [P](const ObjectPtr& o) { return U32List(P(o).generator_indices); }});
        t.members.push_back({"generators", false, "the generators as columns", [P](const ObjectPtr& o) { return MatrixFrom(o, P(o).generators); }});
        t.members.push_back({"relations", false, "the relations as t-tuples over the A_S basis",
                             [P](const ObjectPtr& o) {
                                 std::vector<Value> out;
                                 for (const auto& rel : P(o).relations) out.push_back(U32List(rel));
                                 return Value::from_list(std::move(out));
                             }});
        t.members.push_back({"verify", false, "Presentation::verify() (0 = re-derives)", [P](const ObjectPtr& o) { return Value::from_int(P(o).verify()); }});
        t.members.push_back({"json", false, "the presentation JSON", [P](const ObjectPtr& o) { return Value::from_string(P(o).to_json()); }});
        t.print = [](const ObjectPtr& o, std::int64_t) {
            const Presentation& p = Pres(o).P;
            std::ostringstream os;
            os << "Presentation: " << p.min_generators << " generator(s), " << p.min_relations
               << " relation(s)" << (p.is_free() ? " (free)" : "");
            return os.str();
        };
        registry.register_type(std::move(t));
    }
}

} // namespace verbs_detail
} // namespace hecke_ext
