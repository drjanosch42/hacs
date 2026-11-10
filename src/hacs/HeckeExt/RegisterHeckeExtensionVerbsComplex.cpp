// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// RegisterHeckeExtensionVerbsComplex.cpp -- one area of the extension layer's console
// bindings (S24 CL-8 split of RegisterHeckeExtensionVerbs.cpp; the map is in
// RegisterHeckeExtensionVerbsDetail.h): Involution (E3), BoundaryCohomology (E4), ConeCohomology (E5),
// and theta / the Dickson long exact sequence (DicksonLES).
// ===========================================================================

#include "RegisterHeckeExtensionVerbsDetail.h"

namespace hecke_ext {
namespace verbs_detail {

namespace {

// ---------------------------------------------------------------------------
// (slot-12 WP-8; [P] v3.9 Sec. 3.6 "theta and the Dickson long exact
// sequence", built as specified.)  theta is SOURCE-indexed ([E]
// sec:thetaobj):  theta_n : H^q(M_n (x) det) -> H^q(M_{n+ell+1}), the same map
// as L_map(H_lower, H) (target-indexed).  Both verbs bind the slot-11 WP-5 /
// WP-6 objects (HeckeExt/HeckeDicksonTheta, HeckeDicksonThetaHecke); nothing
// here is a second implementation of theta.
//   dickson_les(n, ell) -> DicksonLES   a lazy handle over DicksonThetaCell;
//   theta(H, H_up)      -> Record      DicksonInducedMaps + the Hecke checks
//                                      (twist_check over {2,3,5,7} \ {ell},
//                                      u_ell_zero) on two Cohomology handles.
// Errors ([P] Sec. 8, v3.9-pre.1 rows): a non-default coefficient_ring exit 5
// (at dispatch and in the lazy compute), a ring handle exit 5, mismatched
// handles TypeError exit 3, ell < 5 the ENGINE_CHECK_MSG naming
// lem:h2torsion, exit 3.
// ---------------------------------------------------------------------------
struct DicksonLESVal {
    uint32_t n = 0, ell = 0;
    std::unique_ptr<DicksonThetaCell> cell;
    void ensure() {
        if (cell) return;
        RequireField("dickson_les");
        // DicksonThetaCell refuses ell < 5 itself (ENGINE_CHECK_MSG naming
        // lem:h2torsion -> std::logic_error -> exit 3; [E] sec:thetaobj (h)).
        cell.reset(new DicksonThetaCell(n, ell));
    }
};
DicksonLESVal& Les(const ObjectPtr& o) { return *o->payload_as<DicksonLESVal>(); }

} // namespace

Value VerbDicksonLES(CallContext& ctx, const std::vector<Value>& a) {
    static const char* kWho = "dickson_les(n, ell)";
    RequireField("dickson_les");
    auto v = std::make_shared<DicksonLESVal>();
    v->n = console::bind::U32Arg(a[0], ctx.pos, kWho);
    v->ell = console::bind::U32Arg(a[1], ctx.pos, kWho);
    if (v->ell < 2 || !hecke::IsPrimeU64(v->ell)) throw TypeError(std::string(kWho) + ": ell must be a prime >= 2", ctx.pos);
    ObjectPtr o = Object::create(console::bind::TypeByName(ctx, kTypeDicksonLES), ctx.pos,
                                 ctx.call_text, ctx.session->settings().snapshot(), v,
                                 console::bind::LazyCompute([v](Object&) { v->ensure(); }));
    return Value::from_object(o);
}

namespace {

Value ThetaMatValue(CallContext& ctx, const ThetaMat& m) {
    return console::bind::MatrixValue(ctx, Flat(m.to_matrix()));
}

} // namespace

Value VerbTheta(CallContext& ctx, const std::vector<Value>& a) {
    static const char* kWho = "theta(H, H_up)";
    RequireField("theta");
    const ObjectPtr& H = console::bind::AsObjectOf(a[0], console::bind::kTypeCohomology, ctx.pos, kWho);
    const ObjectPtr& Hu = console::bind::AsObjectOf(a[1], console::bind::kTypeCohomology, ctx.pos, kWho);
    console::bind::RequireFieldHandle(H, "console extension verb theta");
    console::bind::RequireFieldHandle(Hu, "console extension verb theta");
    const CohSource lo = SourceOf(H, ctx.pos, kWho);
    const CohSource hi = SourceOf(Hu, ctx.pos, kWho);
    if (lo.ell != hi.ell) {
        throw TypeError(std::string(kWho) + ": the two cohomologies differ in ell (H: ell = " +
                            std::to_string(lo.ell) + ", H_up: ell = " + std::to_string(hi.ell) + ")",
                        ctx.pos);
    }
    const uint32_t ell = lo.ell, n = lo.n, n_up = hi.n;
    if (n_up != n + ell + 1) {
        throw TypeError(std::string(kWho) + ": needs H_up at n_up = n + ell + 1 (here n = " +
                            std::to_string(n) + ", n_up = " + std::to_string(n_up) + ", ell = " +
                            std::to_string(ell) + ")",
                        ctx.pos);
    }
    // [E] sec:thetaobj (h): a mathematical precondition, not a build
    // capability -- the same refusal DicksonThetaCell makes (exit 3).
    ENGINE_CHECK_MSG(ell >= 5,
                     "theta(H, H_up) requires ell >= 5 ([C] lem:h2torsion; rmk:thetasmallell): at ell in "
                     "{2, 3} the equivariant cochain functor is not exact on the model, so the long sequence "
                     "of 0 -> Sym^n -> Sym^{n+ell+1} -> Q_n -> 0 is not computed there");
    const std::vector<FlatMatrix> induced = DicksonInducedMaps(*hi.coh, *lo.coh, *hi.K, n_up, ell);
    std::vector<ThetaMat> th;
    for (const FlatMatrix& f : induced) th.push_back(ThetaFromMatrix(f.to_matrix()));
    Record r;
    r.set("n", Value::from_int(n));
    r.set("n_up", Value::from_int(n_up));
    r.set("ell", Value::from_int(ell));
    std::vector<Value> maps;
    std::vector<uint32_t> ranks;
    for (const FlatMatrix& f : induced) {
        maps.push_back(console::bind::MatrixValue(ctx, f));
        Flat c(f);
        ranks.push_back(c.rank());
    }
    r.set("induced", Value::from_list(std::move(maps)));
    r.set("ranks", U32List(ranks));
    ENGINE_CHECK_MSG(th.size() >= 2, "theta(H, H_up): the model has degrees 0..2");
    const ThetaMat& t1 = th[1];
    r.set("ker", ThetaMatValue(ctx, ThetaKernelBasis(t1)));
    r.set("image", ThetaMatValue(ctx, ThetaImageBasis(t1)));
    // The Hecke checks ([C] prop:thetatwist, thm:thetales(2)), on every degree
    // where theta is a non-empty map.
    hecke::HeckeOperatorFamily fam_n(lo.coh, *lo.K, *lo.model, ell, n);
    hecke::HeckeOperatorFamily fam_up(hi.coh, *hi.K, *hi.model, ell, n_up);
    auto nonempty = [](const ThetaMat& m) { return m.rows > 0 && m.cols > 0; };
    uint32_t twist_fail = 0;
    for (uint64_t p : {uint64_t{2}, uint64_t{3}, uint64_t{5}, uint64_t{7}}) {
        if (p == ell) continue;
        const auto& An = fam_n.T_p(p);
        const auto& Au = fam_up.T_p(p);
        bool ok = true;
        for (size_t q = 0; q < th.size(); ++q) {
            if (!nonempty(th[q])) continue;
            const ThetaMat lhs = ThetaMul(ThetaFromMatrix(Au.at(q)), th[q]);
            const ThetaMat rhs =
                ThetaScaled(ThetaMul(th[q], ThetaFromMatrix(An.at(q))), static_cast<uint32_t>(p % ell));
            if (!ThetaEqual(lhs, rhs)) ok = false;
        }
        if (!ok) ++twist_fail;
    }
    r.set("twist_check", Value::from_int(twist_fail));
    bool u_zero = true;
    {
        const auto& Uu = fam_up.T_p(ell);   // U_ell: T_p at p = ell, the same pipeline
        for (size_t q = 0; q < th.size(); ++q) {
            if (!nonempty(th[q])) continue;
            if (!ThetaMul(ThetaFromMatrix(Uu.at(q)), th[q]).is_zero()) u_zero = false;
        }
    }
    r.set("u_ell_zero", Value::from_bool(u_zero));
    return Value::from_record(std::move(r));
}

namespace {

// ===========================================================================
// Involution (E3), BoundaryCohomology (E4), ConeCohomology (E5) -- built on a
// Cohomology handle H of the 7-vertex model, each keeping H alive.
// ===========================================================================

std::vector<uint64_t> GoodSmallPrimes(uint32_t ell) {
    std::vector<uint64_t> out;
    for (uint64_t p : {2u, 3u, 5u, 7u})
        if (p % ell != 0) out.push_back(p);
    return out;
}

struct InvolutionVal {
    ObjectPtr H;
    std::unique_ptr<hecke::HeckeInvolution> inv;
    std::map<uint32_t, std::unique_ptr<HeckeSignSplit>> splits;
    uint32_t n = 0, ell = 0;
    // (slot-20 WP-LV3; [E] sec:ring (h)) iota^* over Z/ell^k: the handle's ring (1 = F_ell) and the module involution.
    uint32_t ring_k = 1;
    std::unique_ptr<hecke::RingInvolution> ring;
    void ensure() {
        if (inv || ring) return;
        if (ring_k >= 2) {
            ensure_ring();
            return;
        }
        RequireField("iota");
        const CohSource s = SourceOf(H, SourcePos(), "iota(H)");
        n = s.n;
        ell = s.ell;
        inv = BuildProductionInvolution(*s.coh, *s.K, *s.model, s.n, s.ell, "console");
    }
    void ensure_ring() {
        console::bind::CohomologyVal* cv = H->payload_as<console::bind::CohomologyVal>();
        CONSOLE_CHECK(cv != nullptr && cv->L != nullptr);
        const console::bind::LocalSystemVal* lv = cv->L->payload_as<console::bind::LocalSystemVal>();
        n = lv->n_sym;
        ell = lv->ell;
        if (cv->digit)
            throw hecke::NotImplementedInThisBuild("console iota(H) over Z/ell^" + std::to_string(ring_k) +
                                                   " at a digit cell (ell^k >= 2^64): the digit route stores no module basis");
        cv->ensure();
        const console::bind::ModelVal& mv = console::bind::ModelOf(*cv);
        const hecke::IotaSimplicialMap im = hecke::BuildIotaSimplicialMapOnModel(mv.K, *mv.model);
        const hecke::ScopedCoefficientRing scope(ring_k, hecke::SettingSource::Console);
        ring.reset(new hecke::RingInvolution(cv->wide ? hecke::BuildRingInvolution(*cv->coh64, im, n, ell, "console")
                                                      : hecke::BuildRingInvolution(*cv->coh, im, n, ell, "console")));
    }
    const HeckeSignSplit* split(uint32_t q) {
        auto it = splits.find(q);
        if (it != splits.end()) return it->second.get();
        const CohSource s = SourceOf(H, SourcePos(), "Involution.sign_split(q)");
        std::unique_ptr<HeckeSignSplit> sp =
            BuildSignSplit(*s.coh, *s.K, *s.model, *inv, s.n, s.ell, q, GoodSmallPrimes(s.ell));
        const HeckeSignSplit* raw = sp.get();
        splits[q] = std::move(sp);
        return raw;
    }
};
InvolutionVal& Inv(const ObjectPtr& o) { return *o->payload_as<InvolutionVal>(); }

struct BoundaryVal {
    ObjectPtr H;
    const hecke::SL2ZTessellationTables* model = nullptr;
    std::unique_ptr<hecke::SL2ZBoundaryPair> pair;
    std::map<uint64_t, std::unique_ptr<BoundaryActionAtPrime>> acts;
    uint32_t n = 0, ell = 0;
    void ensure() {
        if (pair) return;
        RequireField("boundary");
        const CohSource s = SourceOf(H, SourcePos(), "boundary(H)");
        n = s.n;
        ell = s.ell;
        model = s.model;
        pair.reset(new hecke::SL2ZBoundaryPair(s.coh, *s.K));
    }
    const BoundaryActionAtPrime& at(uint64_t p) {
        auto it = acts.find(p);
        if (it == acts.end()) it = acts.emplace(p, BuildBoundaryActionAtPrime(*pair, *model, p)).first;
        return *it->second;
    }
};
BoundaryVal& Bd(const ObjectPtr& o) { return *o->payload_as<BoundaryVal>(); }

struct ConeVal {
    ObjectPtr H;
    const hecke::SL2ZTessellationTables* model = nullptr;
    std::unique_ptr<hecke::SL2ZBoundaryPair> pair;
    std::unique_ptr<EquivariantMappingCone> cone;
    std::map<uint64_t, std::unique_ptr<ConeActionAtPrime>> acts;
    uint32_t n = 0, ell = 0;
    void ensure() {
        if (cone) return;
        RequireField("cone");
        const CohSource s = SourceOf(H, SourcePos(), "cone(H)");
        n = s.n;
        ell = s.ell;
        model = s.model;
        pair.reset(new hecke::SL2ZBoundaryPair(s.coh, *s.K));
        cone = pair->make_cone();
    }
    const ConeActionAtPrime& at(uint64_t p) {
        auto it = acts.find(p);
        if (it == acts.end()) it = acts.emplace(p, BuildConeActionAtPrime(*pair, *cone, *model, p)).first;
        return *it->second;
    }
};
ConeVal& Cone(const ObjectPtr& o) { return *o->payload_as<ConeVal>(); }

template <class Val>
Value LazyOnCohomology(CallContext& ctx, const std::vector<Value>& a, const char* type_name,
                       const char* who, const char* verb) {
    RequireField(verb);
    const ObjectPtr& H = console::bind::AsObjectOf(a[0], console::bind::kTypeCohomology, ctx.pos, who);
    // (slot-9 WP-12 x WP-13) a Z/ell^k handle is refused at dispatch, exit 5, not
    // when the lazy handle is forced onto a field accessor.
    console::bind::RequireFieldHandle(H, std::string("console extension verb ") + verb);
    {
        // Refuse a preset local system at dispatch (it carries no model).
        const console::bind::CohomologyVal* cv = H->payload_as<console::bind::CohomologyVal>();
        const console::bind::LocalSystemVal* lv = cv->L->payload_as<console::bind::LocalSystemVal>();
        if (!lv->preset.empty() || lv->model == nullptr) {
            throw TypeError(std::string(who) + ": needs a cohomology on the SL(2,Z) model; the "
                                               "preset local system '" + lv->preset +
                                "' carries no tessellation model",
                            ctx.pos);
        }
    }
    auto v = std::make_shared<Val>();
    v->H = H;
    ObjectPtr o = Object::create(console::bind::TypeByName(ctx, type_name), ctx.pos, ctx.call_text,
                                 ctx.session->settings().snapshot(), v,
                                 console::bind::LazyCompute([v](Object&) { v->ensure(); }));
    o->keep_alive(H);
    return Value::from_object(o);
}

} // namespace

Value VerbIota(CallContext& ctx, const std::vector<Value>& a) {
    // (slot-20 WP-LV3; [E] sec:ring (h)) iota HONOURS a Z/ell^k Cohomology under the same ring (ell >= 5; the uint32_t
    // and wide bands; a digit handle refuses when forced).  A mismatch between the ring in force and the handle's is
    // refused at dispatch (exit 5), exactly as `hecke(H, p)`.  An F_ell handle is unchanged (LazyOnCohomology).
    const ObjectPtr& H = console::bind::AsObjectOf(a[0], console::bind::kTypeCohomology, ctx.pos, "iota(H)");
    const uint32_t k = hecke::CoefficientRingK("console verb iota");
    const uint32_t kH = console::bind::HandleRingK(H);
    if (kH < 2) return LazyOnCohomology<InvolutionVal>(ctx, a, kTypeInvolution, "iota(H)", "iota");
    if (k != kH) {
        hecke::RefuseCoefficientRing("console extension verb iota", k >= 2 ? k : kH,
                                     " (coefficient_ring = " + hecke::CoefficientRingText(k) +
                                         " is in force, but the Cohomology was built over " + hecke::CoefficientRingText(kH) + ")");
    }
    {
        const console::bind::CohomologyVal* cv = H->payload_as<console::bind::CohomologyVal>();
        const console::bind::LocalSystemVal* lv = cv->L->payload_as<console::bind::LocalSystemVal>();
        if (!lv->preset.empty() || lv->model == nullptr) {
            throw TypeError("iota(H): needs a cohomology on the SL(2,Z) model; the preset local system '" + lv->preset +
                                "' carries no tessellation model",
                            ctx.pos);
        }
    }
    auto v = std::make_shared<InvolutionVal>();
    v->H = H;
    v->ring_k = kH;
    ObjectPtr o = Object::create(console::bind::TypeByName(ctx, kTypeInvolution), ctx.pos, ctx.call_text,
                                 ctx.session->settings().snapshot(), v,
                                 console::bind::LazyCompute([v](Object&) { v->ensure(); }));
    o->keep_alive(H);
    return Value::from_object(o);
}
Value VerbBoundary(CallContext& ctx, const std::vector<Value>& a) {
    return LazyOnCohomology<BoundaryVal>(ctx, a, kTypeBoundaryCoh, "boundary(H)", "boundary");
}
Value VerbCone(CallContext& ctx, const std::vector<Value>& a) {
    return LazyOnCohomology<ConeVal>(ctx, a, kTypeConeCoh, "cone(H)", "cone");
}

namespace {

// The four per-complex degree members ([P] Sec. 3.6) + coefficient_ring.
void AddDegreeMembers(TypeInfo& t, std::function<ComplexDegreeFields(const ObjectPtr&)> fields) {
    t.members.push_back({"q_max", true, "the top degree of THIS complex (derived)",
                         [fields](const ObjectPtr& o) { return Value::from_int(fields(o).q_max); }});
    t.members.push_back({"truncated", true, "false on a finite model",
                         [fields](const ObjectPtr& o) { return Value::from_bool(fields(o).truncated); }});
    t.members.push_back({"q_max_source", true, "'derived' unless truncated (E9)",
                         [fields](const ObjectPtr& o) { return Value::from_string(fields(o).q_max_source); }});
    t.members.push_back({"degrees", true, "[0 .. q_max], derived",
                         [fields](const ObjectPtr& o) { return U32List(fields(o).degrees); }});
    t.members.push_back({"coefficient_ring", true, "the complex's ring (the extension layer is F_ell)",
                         [](const ObjectPtr&) { return Value::from_string("F_ell"); }});
}

TypeInfo::Method DegreeMatrixMethod(const char* name, const char* doc,
                                    std::function<size_t(const ObjectPtr&)> count,
                                    std::function<Value(const ObjectPtr&, uint32_t, const SourcePos&)> get) {
    TypeInfo::Method m;
    m.name = name;
    m.params = {"q"};
    m.required = 1;
    m.doc = doc;
    m.call = [name, count, get](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
        const std::string who = std::string(name) + "(q)";
        const uint32_t q = DegreeArgC5(args[0], pos, who.c_str(), count(o));
        return Guard(who.c_str(), pos, [&] { return get(o, q, pos); });
    };
    return m;
}

TypeInfo::Method PrimeDegreeMethod(const char* name, const char* doc,
                                   std::function<Value(const ObjectPtr&, uint64_t, uint32_t, const SourcePos&)> get) {
    TypeInfo::Method m;
    m.name = name;
    m.params = {"p", "q"};
    m.required = 2;
    m.doc = doc;
    m.call = [name, get](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
        const std::string who = std::string(name) + "(p, q)";
        const uint64_t p = console::bind::U64Arg(args[0], pos, who.c_str());
        const uint32_t q = console::bind::U32Arg(args[1], pos, who.c_str());
        if (p < 2) throw TypeError(who + ": p must be a prime", pos);
        return Guard(who.c_str(), pos, [&] { return get(o, p, q, pos); });
    };
    return m;
}

} // namespace

void RegisterComplexTypes(Registry& registry) {
    // --- Involution -------------------------------------------------------
    {
        TypeInfo t;
        t.name = kTypeInvolution;
        t.doc = "the production involution iota^* = -iota^*_geo on H ([C] def:iota(iv); "
                "HeckeInvolution) with the +/- split and, per degree, the sign split of the Hecke "
                "action and the Eisenstein line (HeckeSignSplit); lazy";
        // (slot-20 WP-LV3) On a Z/ell^k handle (Inv(o).ring_k >= 2) every member below dereferencing the FIELD object
        // raises first: the field members (dims, dims_plus, dims_minus) a named std::logic_error (exit 3: there is no dim
        // over Z/ell^k), the field methods (matrix, plus, minus, sign_split, eisenstein_line) hecke::NotImplementedInThisBuild
        // (exit 5, not on [E] sec:ring (h)'s list).  n, ell, status, certificate, json and the new module members honour.
        auto I = [](const ObjectPtr& o) -> const hecke::HeckeInvolution& {
            InvolutionVal& v = Inv(o);
            if (v.ring_k >= 2)
                throw std::logic_error("Involution: a field (F_ell) accessor, but this object is over " + hecke::CoefficientRingText(v.ring_k) +
                                       " -- over Z/ell^k there is no dim; use invariant_factors_plus / invariant_factors_minus / "
                                       "lengths_plus / lengths_minus ([E] sec:ring (b3), (h))");
            return *v.inv;
        };
        auto RefuseOnRing = [](const ObjectPtr& o, const char* who) {
            if (Inv(o).ring_k >= 2)
                hecke::RefuseCoefficientRing(std::string("console Involution.") + who, Inv(o).ring_k,
                                             " -- the member is not on [E] sec:ring (h)'s list");
        };
        t.members.push_back({"n", true, "the symmetric power", [](const ObjectPtr& o) { return Value::from_int(Inv(o).n); }});
        t.members.push_back({"ell", true, "the prime", [](const ObjectPtr& o) { return Value::from_int(Inv(o).ell); }});
        t.members.push_back({"status", true, "the split's status", [I](const ObjectPtr& o) {
                                 if (Inv(o).ring_k >= 2) return Value::from_string(hecke::to_string(hecke::SignSplitStatus::Split));
                                 return Value::from_string(hecke::to_string(I(o).sign_split_status()));
                             }});
        t.members.push_back({"coefficient_ring", true, "F_ell, or Z/ell^k (slot 20: iota honours the ring)", [](const ObjectPtr& o) {
                                 return Value::from_string(Inv(o).ring_k >= 2 ? "Z/ell^" + std::to_string(Inv(o).ring_k) : std::string("F_ell"));
                             }});
        // the module members: H^q_+- as Z/ell^k-modules (over F_ell: k = 1, every factor 1)
        auto SignFactors = [I](const ObjectPtr& o, uint32_t q, bool plus) {
            InvolutionVal& v = Inv(o);
            if (v.ring_k >= 2) {
                CONSOLE_CHECK(v.ring != nullptr);
                if (q >= v.ring->degrees.size()) return std::vector<uint32_t>{};
                return plus ? v.ring->degrees[q].plus_factors : v.ring->degrees[q].minus_factors;
            }
            return std::vector<uint32_t>(plus ? I(o).dim_H_plus(q) : I(o).dim_H_minus(q), 1u);
        };
        auto ndegrees = [I](const ObjectPtr& o) -> uint32_t {
            InvolutionVal& v = Inv(o);
            if (v.ring_k >= 2) return static_cast<uint32_t>(v.ring->degrees.size());
            return I(o).num_degrees();
        };
        t.members.push_back({"lengths_plus", true, "length H^q_+ per degree (= dims_plus over F_ell)", [SignFactors, ndegrees](const ObjectPtr& o) {
                                 return Guard("Involution.lengths_plus", SourcePos(), [&] {
                                     std::vector<uint32_t> d;
                                     for (uint32_t q = 0; q < ndegrees(o); ++q) {
                                         uint32_t s = 0;
                                         for (uint32_t e : SignFactors(o, q, true)) s += e;
                                         d.push_back(s);
                                     }
                                     return U32List(d);
                                 });
                             }});
        t.members.push_back({"lengths_minus", true, "length H^q_- per degree (= dims_minus over F_ell)", [SignFactors, ndegrees](const ObjectPtr& o) {
                                 return Guard("Involution.lengths_minus", SourcePos(), [&] {
                                     std::vector<uint32_t> d;
                                     for (uint32_t q = 0; q < ndegrees(o); ++q) {
                                         uint32_t s = 0;
                                         for (uint32_t e : SignFactors(o, q, false)) s += e;
                                         d.push_back(s);
                                     }
                                     return U32List(d);
                                 });
                             }});
        t.methods.push_back(DegreeMatrixMethod("invariant_factors_plus", "H^q_+ = (+)_i Z/ell^{e_i}: the e_i, descending (F_ell: all 1)",
            [ndegrees](const ObjectPtr& o) { return static_cast<size_t>(ndegrees(o)); },
            [SignFactors](const ObjectPtr& o, uint32_t q, const SourcePos&) { return U32List(SignFactors(o, q, true)); }));
        t.methods.push_back(DegreeMatrixMethod("invariant_factors_minus", "H^q_- = (+)_i Z/ell^{e_i}: the e_i, descending (F_ell: all 1)",
            [ndegrees](const ObjectPtr& o) { return static_cast<size_t>(ndegrees(o)); },
            [SignFactors](const ObjectPtr& o, uint32_t q, const SourcePos&) { return U32List(SignFactors(o, q, false)); }));
        t.members.push_back({"dims", true, "dim H^q", [I](const ObjectPtr& o) {
                                 std::vector<uint32_t> d;
                                 for (uint32_t q = 0; q < I(o).num_degrees(); ++q) d.push_back(I(o).dim_H(q));
                                 return U32List(d);
                             }});
        t.members.push_back({"dims_plus", true, "dim H^q_+", [I](const ObjectPtr& o) {
                                 return Guard("Involution.dims_plus", SourcePos(), [&] {
                                     std::vector<uint32_t> d;
                                     for (uint32_t q = 0; q < I(o).num_degrees(); ++q) d.push_back(I(o).dim_H_plus(q));
                                     return U32List(d);
                                 });
                             }});
        t.members.push_back({"dims_minus", true, "dim H^q_-", [I](const ObjectPtr& o) {
                                 return Guard("Involution.dims_minus", SourcePos(), [&] {
                                     std::vector<uint32_t> d;
                                     for (uint32_t q = 0; q < I(o).num_degrees(); ++q) d.push_back(I(o).dim_H_minus(q));
                                     return U32List(d);
                                 });
                             }});
        t.members.push_back({"certificate", true, "InvolutionCertificate::verify() (0 = obl:iota discharged; over Z/ell^k: iota^2 = id and H = H^+ (+) H^- were ENGINE_CHECKed)",
                             [I](const ObjectPtr& o) {
                                 if (Inv(o).ring_k >= 2) return Value::from_int(0);
                                 return Value::from_int(I(o).certificate().verify());
                             }});
        t.members.push_back({"json", true, "hecke_involution/v1 (over Z/ell^k: hecke_involution_ring/v1)", [I](const ObjectPtr& o) {
                                 if (Inv(o).ring_k >= 2) return Value::from_string(Inv(o).ring->to_json());
                                 return Value::from_string(I(o).to_json());
                             }});
        auto ndeg = [I, RefuseOnRing](const ObjectPtr& o) {
            RefuseOnRing(o, "matrix / plus / minus / sign_split / eisenstein_line");
            return static_cast<size_t>(I(o).num_degrees());
        };
        t.methods.push_back(DegreeMatrixMethod("matrix", "iota^* on H^q", ndeg,
            [I](const ObjectPtr& o, uint32_t q, const SourcePos&) { return FlatFrom(o, I(o).matrix(q)); }));
        t.methods.push_back(DegreeMatrixMethod("plus", "a basis of H^q_+ (columns, in H^q coordinates)", ndeg,
            [I](const ObjectPtr& o, uint32_t q, const SourcePos&) { return FlatFrom(o, I(o).H_plus_basis(q)); }));
        t.methods.push_back(DegreeMatrixMethod("minus", "a basis of H^q_- (columns, in H^q coordinates)", ndeg,
            [I](const ObjectPtr& o, uint32_t q, const SourcePos&) { return FlatFrom(o, I(o).H_minus_basis(q)); }));
        t.methods.push_back(DegreeMatrixMethod("sign_split", "the Hecke action's sign split at {2,3,5,7} \\ {ell} and the Eisenstein line", ndeg,
            [](const ObjectPtr& o, uint32_t q, const SourcePos& pos) {
                const HeckeSignSplit* s = Inv(o).split(q);
                if (s == nullptr) throw RuntimeError("Involution.sign_split(q): dim H^q = 0", pos);
                Record r;
                r.set("degree", Value::from_int(q));
                r.set("status", Value::from_string(ToString(s->status())));
                r.set("dim_plus", Value::from_int(s->dim_plus()));
                r.set("dim_minus", Value::from_int(s->dim_minus()));
                uint32_t commuting = 0;
                std::vector<uint64_t> primes;
                for (size_t i = 0; i < s->num_operators(); ++i) {
                    primes.push_back(s->op(i).p);
                    if (s->op(i).commutes) ++commuting;
                }
                r.set("primes", U64List(primes));
                r.set("operators_commuting", Value::from_int(commuting));
                const EisensteinLine& e = s->eisenstein_line();
                Record er;
                er.set("status", Value::from_string(e.status));
                er.set("defined", Value::from_bool(e.defined));
                er.set("f", Value::from_int(e.f));
                er.set("dim_socle", Value::from_int(e.dim_socle));
                er.set("socle_equals_mV", Value::from_bool(e.socle_equals_mV));
                r.set("eisenstein", Value::from_record(std::move(er)));
                r.set("certificate", Value::from_int(s->certificate().verify()));
                return Value::from_record(std::move(r));
            }));
        t.methods.push_back(DegreeMatrixMethod("eisenstein_line", "the Eisenstein line in the stored H^q basis (defined only)", ndeg,
            [](const ObjectPtr& o, uint32_t q, const SourcePos& pos) {
                const HeckeSignSplit* s = Inv(o).split(q);
                if (s == nullptr || !s->eisenstein_line().defined)
                    throw RuntimeError("Involution.eisenstein_line(q): no Eisenstein line is defined in degree " +
                                           std::to_string(q), pos);
                return MatrixFrom(o, s->eisenstein_line().line_in_H);
            }));
        t.print = [](const ObjectPtr& o, std::int64_t) {
            std::ostringstream os;
            os << "Involution iota^*";
            if (!o->computed()) os << " (declared, not yet computed)";
            else if (Inv(o).ring_k >= 2) os << " on H(Sym^" << Inv(o).n << ", Z/" << Inv(o).ell << "^" << Inv(o).ring_k << ")";
            else os << " on H(Sym^" << Inv(o).n << ", F_" << Inv(o).ell << ")";
            return os.str();
        };
        registry.register_type(std::move(t));
    }
    // --- BoundaryCohomology -----------------------------------------------
    {
        TypeInfo t;
        t.name = kTypeBoundaryCoh;
        t.doc = "H^*_bd on the boundary circle (SL2ZBoundaryPair), with the restriction r and T_p^bd "
                "(HeckeBoundaryOperator); lazy";
        auto P = [](const ObjectPtr& o) -> const hecke::SL2ZBoundaryPair& { return *Bd(o).pair; };
        t.members.push_back({"n", true, "the symmetric power", [](const ObjectPtr& o) { return Value::from_int(Bd(o).n); }});
        t.members.push_back({"ell", true, "the prime", [](const ObjectPtr& o) { return Value::from_int(Bd(o).ell); }});
        t.members.push_back({"model", true, "the boundary model", [P](const ObjectPtr& o) { return Value::from_string(P(o).boundary_model()); }});
        t.members.push_back({"is_group_cohomology", true, "2 in R^x (the circle computes group cohomology)",
                             [P](const ObjectPtr& o) { return Value::from_bool(P(o).boundary_is_group_cohomology()); }});
        t.members.push_back({"dims", true, "dim H^q_bd", [P](const ObjectPtr& o) { return U32List(P(o).boundary_cohomology().get_H_eq_dimensions()); }});
        t.members.push_back({"bulk_dims", true, "dim H^q of the bulk", [P](const ObjectPtr& o) { return U32List(P(o).bulk_cohomology().get_H_eq_dimensions()); }});
        AddDegreeMembers(t, [P](const ObjectPtr& o) { return ComplexDegreeFields::derived_from(P(o).boundary_cohomology()); });
        auto nr = [P](const ObjectPtr& o) { return P(o).restriction().get_induced_H_eq_maps().size(); };
        t.methods.push_back(DegreeMatrixMethod("r", "the restriction H^q -> H^q_bd", nr,
            [P](const ObjectPtr& o, uint32_t q, const SourcePos&) { return FlatFrom(o, P(o).restriction().get_induced_H_eq_maps()[q]); }));
        t.methods.push_back(PrimeDegreeMethod("A", "T_p^bd on H^q_bd",
            [](const ObjectPtr& o, uint64_t p, uint32_t q, const SourcePos& pos) {
                const BoundaryActionAtPrime& a = Bd(o).at(p);
                if (q >= a.bd->num_degrees()) throw RuntimeError("BoundaryCohomology.A(p, q): no degree " + std::to_string(q), pos);
                return MatrixFrom(o, a.bd->A(q));
            }));
        t.print = [](const ObjectPtr& o, std::int64_t) {
            std::ostringstream os;
            os << "BoundaryCohomology";
            if (!o->computed()) os << " (declared, not yet computed)";
            else os << " of Sym^" << Bd(o).n << " over F_" << Bd(o).ell << " on the circle";
            return os.str();
        };
        registry.register_type(std::move(t));
    }
    // --- ConeCohomology ---------------------------------------------------
    {
        TypeInfo t;
        t.name = kTypeConeCoh;
        t.doc = "H^*_c = H^*(Cone(r)) with its long exact sequence (EquivariantMappingCone) and, per "
                "prime, the cone action A_c(T_p) with h DERIVED from the certified pair homotopy "
                "eta over Z (HeckeConeAction); lazy";
        auto C = [](const ObjectPtr& o) -> const EquivariantMappingCone& { return *Cone(o).cone; };
        t.members.push_back({"n", true, "the symmetric power", [](const ObjectPtr& o) { return Value::from_int(Cone(o).n); }});
        t.members.push_back({"ell", true, "the prime", [](const ObjectPtr& o) { return Value::from_int(Cone(o).ell); }});
        t.members.push_back({"name", true, "the cone's name", [C](const ObjectPtr& o) { return Value::from_string(C(o).name()); }});
        t.members.push_back({"dims", true, "dim H^q_c", [C](const ObjectPtr& o) { return U32List(C(o).dim_Hc_all()); }});
        t.members.push_back({"bulk_dims", true, "dim H^q of the bulk", [C](const ObjectPtr& o) { return U32List(C(o).bulk().get_H_eq_dimensions()); }});
        t.members.push_back({"boundary_dims", true, "dim H^q_bd", [C](const ObjectPtr& o) { return U32List(C(o).boundary().get_H_eq_dimensions()); }});
        AddDegreeMembers(t, [C](const ObjectPtr& o) { return C(o).cone_fields(); });
        t.members.push_back({"certificate", true, "ConeCertificate::verify() (0 = re-derives)",
                             [C](const ObjectPtr& o) { return Value::from_int(C(o).certificate().verify()); }});
        t.members.push_back({"H1_shriek", true, "dim H^1_! = dim ker r^1 (nil if not derivable)",
                             [C](const ObjectPtr& o) {
                                 uint32_t s = 0;
                                 return ConeDimH1Shriek(C(o), &s) ? Value::from_int(s) : Value::nil();
                             }});
        t.members.push_back({"les", true, "the long exact sequence spots {term, q, dim, rank_in, rank_out, asserted, exact}",
                             [C](const ObjectPtr& o) {
                                 std::vector<Value> out;
                                 for (const ConeLesSpot& s : C(o).les_spots()) {
                                     Record r;
                                     r.set("term", Value::from_string(s.term));
                                     r.set("q", Value::from_int(s.q));
                                     r.set("dim", Value::from_int(s.dim_middle));
                                     r.set("rank_in", Value::from_int(s.rank_in));
                                     r.set("rank_out", Value::from_int(s.rank_out));
                                     r.set("asserted", Value::from_bool(s.asserted));
                                     r.set("exact", Value::from_bool(s.exact));
                                     out.push_back(Value::from_record(std::move(r)));
                                 }
                                 return Value::from_list(std::move(out));
                             }});
        t.members.push_back({"les_exact", true, "every asserted spot of the LES is exact",
                             [C](const ObjectPtr& o) {
                                 bool ok = true;
                                 for (const ConeLesSpot& s : C(o).les_spots())
                                     if (s.asserted && !s.exact) ok = false;
                                 return Value::from_bool(ok);
                             }});
        auto nc = [C](const ObjectPtr& o) { return static_cast<size_t>(C(o).num_degrees()); };
        t.methods.push_back(DegreeMatrixMethod("pi", "H^q_c -> H^q", nc,
            [C](const ObjectPtr& o, uint32_t q, const SourcePos& pos) {
                if (!C(o).pi_built(q)) throw RuntimeError("ConeCohomology.pi(q): not built in degree " + std::to_string(q), pos);
                return FlatFrom(o, C(o).pi(q));
            }));
        t.methods.push_back(DegreeMatrixMethod("r", "H^q -> H^q_bd", nc,
            [C](const ObjectPtr& o, uint32_t q, const SourcePos& pos) {
                if (!C(o).r_built(q)) throw RuntimeError("ConeCohomology.r(q): not built in degree " + std::to_string(q), pos);
                return FlatFrom(o, C(o).r(q));
            }));
        t.methods.push_back(DegreeMatrixMethod("iota", "H^q_bd -> H^{q+1}_c (the connecting map)", nc,
            [C](const ObjectPtr& o, uint32_t q, const SourcePos& pos) {
                if (!C(o).iota_built(q)) throw RuntimeError("ConeCohomology.iota(q): not built in degree " + std::to_string(q), pos);
                return FlatFrom(o, C(o).iota(q));
            }));
        t.methods.push_back(PrimeDegreeMethod("A", "A_c(T_p) on H^q_c",
            [](const ObjectPtr& o, uint64_t p, uint32_t q, const SourcePos& pos) {
                const ConeActionAtPrime& a = Cone(o).at(p);
                if (q >= a.act->num_degrees()) throw RuntimeError("ConeCohomology.A(p, q): no degree " + std::to_string(q), pos);
                return MatrixFrom(o, a.act->A_c(q));
            }));
        t.methods.push_back(PrimeDegreeMethod("h", "the cochain homotopy h at p in degree q, DERIVED from eta (not stored)",
            [](const ObjectPtr& o, uint64_t p, uint32_t q, const SourcePos&) {
                return FlatFrom(o, Cone(o).at(p).act->h_flat(q));
            }));
        {
            TypeInfo::Method m;
            m.name = "pair_homotopy";
            m.params = {"p"};
            m.required = 1;
            m.doc = "the certified pair homotopy eta at p: {eta_zero, slots, certify, solve_regime, ...}";
            m.call = [](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                const uint64_t p = console::bind::U64Arg(args[0], pos, "ConeCohomology.pair_homotopy(p)");
                return Guard("ConeCohomology.pair_homotopy(p)", pos, [&] {
                    const hecke::HeckePairHomotopy& e = *Cone(o).at(p).eta;
                    Record r;
                    r.set("p", Value::from_int(static_cast<std::int64_t>(p)));
                    r.set("eta_zero", Value::from_bool(e.eta_zero()));
                    r.set("nonzero_slots", Value::from_int(e.num_nonzero_slots()));
                    r.set("slots", Value::from_int(e.num_slots()));
                    r.set("constructed_slots", Value::from_int(e.num_constructed_slots()));
                    r.set("derived_slots", Value::from_int(e.num_derived_slots()));
                    r.set("fills", Value::from_int(e.num_fills()));
                    r.set("terms", Value::from_int(e.num_terms()));
                    r.set("max_abs_coeff", Value::from_int(e.max_abs_coeff()));
                    r.set("solve_regime", Value::from_string(e.solve_regime()));
                    r.set("integral", Value::from_bool(e.integral()));
                    r.set("q_max", Value::from_int(e.q_max()));
                    r.set("degrees", U32List(e.degrees()));
                    r.set("certify", Value::from_int(e.certify()));
                    r.set("table_fnv64", Value::from_string(oc_util::HexU64(e.table_fnv64())));
                    return Value::from_record(std::move(r));
                });
            };
            t.methods.push_back(std::move(m));
        }
        {
            TypeInfo::Method m;
            m.name = "action_certificate";
            m.params = {"p"};
            m.required = 1;
            m.doc = "ConeActionCertificate::verify() of the action at p (0 = re-derives)";
            m.call = [](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                const uint64_t p = console::bind::U64Arg(args[0], pos, "ConeCohomology.action_certificate(p)");
                return Guard("ConeCohomology.action_certificate(p)", pos, [&] {
                    return Value::from_int(Cone(o).at(p).act->certificate().verify());
                });
            };
            t.methods.push_back(std::move(m));
        }
        t.print = [](const ObjectPtr& o, std::int64_t) {
            std::ostringstream os;
            os << "ConeCohomology H^*_c";
            if (!o->computed()) os << " (declared, not yet computed)";
            else {
                os << " of Sym^" << Cone(o).n << " over F_" << Cone(o).ell << ": dims [";
                const auto& d = Cone(o).cone->dim_Hc_all();
                for (size_t i = 0; i < d.size(); ++i) os << (i ? ", " : "") << d[i];
                os << "]";
            }
            return os.str();
        };
        registry.register_type(std::move(t));
    }
    // --- DicksonLES (slot-12 WP-8; [P] v3.9 Sec. 3.6) ----------------------
    {
        TypeInfo t;
        t.name = kTypeDicksonLES;
        t.doc = "the long exact sequence of 0 -> Sym^n (x) det -> Sym^{n+ell+1} -> Q_n -> 0 on the model "
                "over F_ell, ell >= 5 ([E] sec:thetaobj; HeckeDicksonTheta): the six terms, the five maps "
                "in the stored bases (theta = L_* source-indexed, delta by the snake construction), "
                "exactness at all six spots as subspaces, H^0 the Dickson ring; lazy";
        auto C = [](const ObjectPtr& o) -> const DicksonThetaCell& { return *Les(o).cell; };
        t.members.push_back({"n", false, "the symmetric power of the source term", [](const ObjectPtr& o) { return Value::from_int(Les(o).n); }});
        t.members.push_back({"ell", false, "the prime", [](const ObjectPtr& o) { return Value::from_int(Les(o).ell); }});
        t.members.push_back({"n_up", false, "n + ell + 1, the middle term's symmetric power",
                             [](const ObjectPtr& o) { return Value::from_int(Les(o).n + Les(o).ell + 1); }});
        t.members.push_back({"dims", true, "{H0_n, H0_up, H0_Q, H1_n, H1_up, H1_Q}", [C](const ObjectPtr& o) {
                                 const DicksonThetaCell& c = C(o);
                                 Record r;
                                 r.set("H0_n", Value::from_int(c.h(ThetaTerm::SymN, 0)));
                                 r.set("H0_up", Value::from_int(c.h(ThetaTerm::SymM, 0)));
                                 r.set("H0_Q", Value::from_int(c.h(ThetaTerm::Quot, 0)));
                                 r.set("H1_n", Value::from_int(c.h(ThetaTerm::SymN, 1)));
                                 r.set("H1_up", Value::from_int(c.h(ThetaTerm::SymM, 1)));
                                 r.set("H1_Q", Value::from_int(c.h(ThetaTerm::Quot, 1)));
                                 return Value::from_record(std::move(r));
                             }});
        t.members.push_back({"ranks", true, "{theta0, pi0, delta, theta1, pi1}", [C](const ObjectPtr& o) {
                                 const DicksonThetaCell& c = C(o);
                                 Record r;
                                 r.set("theta0", Value::from_int(ThetaRank(c.theta(0))));
                                 r.set("pi0", Value::from_int(ThetaRank(c.pi(0))));
                                 r.set("delta", Value::from_int(ThetaRank(c.delta(0))));
                                 r.set("theta1", Value::from_int(ThetaRank(c.theta(1))));
                                 r.set("pi1", Value::from_int(ThetaRank(c.pi(1))));
                                 return Value::from_record(std::move(r));
                             }});
        t.members.push_back({"maps", true, "{theta0, pi0, delta, theta1, pi1} as Matrix values in the stored bases (column convention)",
                             [C](const ObjectPtr& o) {
                                 const DicksonThetaCell& c = C(o);
                                 Record r;
                                 r.set("theta0", MatrixFrom(o, c.theta(0).to_matrix()));
                                 r.set("pi0", MatrixFrom(o, c.pi(0).to_matrix()));
                                 r.set("delta", MatrixFrom(o, c.delta(0).to_matrix()));
                                 r.set("theta1", MatrixFrom(o, c.theta(1).to_matrix()));
                                 r.set("pi1", MatrixFrom(o, c.pi(1).to_matrix()));
                                 return Value::from_record(std::move(r));
                             }});
        t.members.push_back({"exact", true, "exactness certified at all six spots as subspaces", [C](const ObjectPtr& o) {
                                 const auto& spots = C(o).certificate().spots;
                                 bool ok = spots.size() == 6;
                                 for (const ExactnessSpot& s : spots) ok = ok && s.exact();
                                 return Value::from_bool(ok);
                             }});
        t.members.push_back({"ker_theta", true, "a basis of ker theta^1 (columns, in H1_n's stored coordinates)",
                             [C](const ObjectPtr& o) { return MatrixFrom(o, ThetaKernelBasis(C(o).theta(1)).to_matrix()); }});
        t.members.push_back({"im_delta", true, "a basis of im delta (columns, in H1_n's stored coordinates)",
                             [C](const ObjectPtr& o) { return MatrixFrom(o, ThetaImageBasis(C(o).delta(0)).to_matrix()); }});
        t.members.push_back({"ker_equals_im", true, "ker theta^1 = im delta as subspaces of H1_n", [C](const ObjectPtr& o) {
                                 const ThetaMat K = ThetaKernelBasis(C(o).theta(1));
                                 const ThetaMat I = ThetaImageBasis(C(o).delta(0));
                                 return Value::from_bool(K.cols == I.cols && ThetaSpanContains(K, I) &&
                                                         ThetaSpanContains(I, K));
                             }});
        t.members.push_back({"dickson_count", false, "#{(a, b) : a(ell + 1) + b(ell^2 - ell) = n}",
                             [](const ObjectPtr& o) {
                                 return Value::from_int(static_cast<std::int64_t>(DicksonExponents(Les(o).n, Les(o).ell).size()));
                             }});
        t.members.push_back({"certificate", true, "DicksonThetaCertificate code (0 = every check of [E] (6a)-(6d) held)",
                             [C](const ObjectPtr& o) { return Value::from_int(C(o).certificate().code); }});
        t.members.push_back({"coefficient_ring", false, "the complex's ring (theta is F_ell only)",
                             [](const ObjectPtr&) { return Value::from_string("F_ell"); }});
        t.print = [](const ObjectPtr& o, std::int64_t) {
            std::ostringstream os;
            os << "DicksonLES (n = " << Les(o).n << ", ell = " << Les(o).ell << ")";
            if (!o->computed()) os << " (declared, not yet computed)";
            else os << " " << Les(o).cell->summary();
            return os.str();
        };
        registry.register_type(std::move(t));
    }
}

} // namespace verbs_detail
} // namespace hecke_ext
