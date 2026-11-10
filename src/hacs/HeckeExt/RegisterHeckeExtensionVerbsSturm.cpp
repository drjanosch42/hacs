// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// RegisterHeckeExtensionVerbsSturm.cpp -- one area of the extension layer's console
// bindings (S24 CL-8 split of RegisterHeckeExtensionVerbs.cpp; the map is in
// RegisterHeckeExtensionVerbsDetail.h): the slot-4/5/6 verbs (C1b, WP-10, C2a): sturm_bound, u_ell_decision,
// bernoulli_mod, glover_factors, canonical_field_polynomial, and the `Sturm` type.
// ===========================================================================

#include "RegisterHeckeExtensionVerbsDetail.h"

namespace hecke_ext {
namespace verbs_detail {

namespace {

// The payload: the computation plus the inputs a `resume` needs to rebuild it.
struct SturmVal {
    uint32_t k = 0;
    uint32_t ell = 0;
    uint32_t cap = 0;                       // 0 = the engine's default cap
    std::unique_ptr<SturmBound> bound;
    // A resume carries the parent's reached state forward through resume(cap)
    // rather than re-running, which is what makes budget_history [c1, c2].
    std::unique_ptr<SturmBound> from_parent;
    uint32_t resume_cap = 0;

    void ensure() {
        if (bound) return;
        if (from_parent) {
            bound = std::move(from_parent);
            bound->resume(resume_cap);
            return;
        }
        bound.reset(new SturmBound(k, ell, cap));
        bound->run();
    }
};

SturmVal& Sturm(const ObjectPtr& o) { return *o->payload_as<SturmVal>(); }

const SturmBound& Bound(const ObjectPtr& o) {
    SturmVal& sv = Sturm(o);
    sv.ensure();
    return *sv.bound;
}

// The R3 state every budgeted handle reports (plan Sec. 2 C5, [X] Sec. 1.3.1).
void FillResumableState(Object& self, const SturmVal& sv) {
    const SturmState& st = sv.bound->state();
    ResumableState rs;
    rs.budget_setting = "sturm.cap";
    rs.budget_value = Value::from_int(st.cap);
    rs.resumable = (st.outcome == SturmOutcome::NoneAtCap);
    if (rs.resumable) {
        rs.reason = "the algebra closure was still proper when the cap " +
                    std::to_string(st.cap) + " was passed";
    }
    Record pr;
    pr.set("kind", Value::from_string("sturm_bound"));
    pr.set("outcome", Value::from_string(ToString(st.outcome)));
    pr.set("reached", Value::from_int(st.B_reached));
    pr.set("closure_dim", Value::from_int(st.closure_dim));
    pr.set("dim_M_k", Value::from_int(st.dim_M_k));
    std::vector<Value> hist;
    for (uint32_t b : st.budget_history) hist.push_back(Value::from_int(b));
    pr.set("budget_history", Value::from_list(std::move(hist)));
    std::vector<Value> primes;
    for (uint32_t q : st.primes_done) primes.push_back(Value::from_int(q));
    pr.set("primes_done", Value::from_list(std::move(primes)));
    rs.progress = pr;
    self.set_resumable_state(rs);
}

ObjectPtr MakeSturmHandle(const TypeInfo* type, const SourcePos& pos, const std::string& expr,
                          const console::SettingsSnapshot& snap, std::shared_ptr<SturmVal> v) {
    return Object::create(type, pos, expr, snap, v, console::bind::LazyCompute([v](Object& self) {
        v->ensure();
        FillResumableState(self, *v);
    }));
}

} // namespace

Value VerbSturmBound(CallContext& ctx, const std::vector<Value>& a) {
    static const char* kWho = "sturm_bound(k, ell, cap)";
    auto v = std::make_shared<SturmVal>();
    v->k = console::bind::U32Arg(a[0], ctx.pos, kWho);
    v->ell = console::bind::U32Arg(a[1], ctx.pos, kWho);
    if (a.size() > 2 && !a[2].is_nil()) v->cap = console::bind::U32Arg(a[2], ctx.pos, kWho);
    if (v->k < 2) throw TypeError(std::string(kWho) + ": the weight k must be at least 2", ctx.pos);
    if (v->ell < 2) throw TypeError(std::string(kWho) + ": ell must be a prime >= 2", ctx.pos);
    return Value::from_object(MakeSturmHandle(console::bind::TypeByName(ctx, kTypeSturm), ctx.pos,
                                              ctx.call_text, ctx.session->settings().snapshot(),
                                              v));
}

Value VerbUEllDecision(CallContext& ctx, const std::vector<Value>& a) {
    static const char* kWho = "u_ell_decision(n, ell, budget_mb)";
    const uint32_t n = console::bind::U32Arg(a[0], ctx.pos, kWho);
    const uint64_t ell = console::bind::U64Arg(a[1], ctx.pos, kWho);
    UEllVerdict verdict;
    if (a.size() > 2 && !a[2].is_nil()) {
        verdict = UEllDecision(n, ell, console::bind::U32Arg(a[2], ctx.pos, kWho));
    } else {
        // No explicit budget: the `u_ell` SETTING in force decides, which is
        // what Stage 1 of HeckeExt/HeckeSturmSweep does ([X] Sec. 2.2).
        verdict = UEllDecision(n, ell, hecke::math_settings().u_ell);
    }
    Record r;
    r.set("included", Value::from_bool(verdict.included));
    r.set("reason", Value::from_string(to_string(verdict.reason)));
    r.set("budget_mb", Value::from_int(verdict.budget_mb));
    r.set("model_version", Value::from_string(verdict.model_version));
    // predicted_mb is a double and the language has no Float: it is reported in
    // hundredths of a megabyte as an Int, and as its canonical text.
    std::ostringstream os;
    os.setf(std::ios::fixed);
    os.precision(2);
    os << verdict.predicted_mb;
    r.set("predicted_mb", Value::from_string(os.str()));
    r.set("predicted_centi_mb",
          Value::from_int(static_cast<std::int64_t>(verdict.predicted_mb * 100.0 + 0.5)));
    return Value::from_record(std::move(r));
}

// ---------------------------------------------------------------------------
// (slot-5 WP-10) The two E6 prediction units.
// ---------------------------------------------------------------------------

// B_w in F_ell ([X] Sec. 2.1; HeckeExt/HeckePredictions).  The unit REFUSES
// outside 0 <= w < ell - 1 rather than returning a wrong value, and the verb
// passes that refusal through as an ordinary console runtime error carrying
// the unit's own message -- which is what makes `expect_error` able to pin it.
Value VerbBernoulliMod(CallContext& ctx, const std::vector<Value>& a) {
    static const char* kWho = "bernoulli_mod(w, ell)";
    const uint32_t w = console::bind::U32Arg(a[0], ctx.pos, kWho);
    const uint32_t ell = console::bind::U32Arg(a[1], ctx.pos, kWho);
    if (ell < 2) throw TypeError(std::string(kWho) + ": ell must be a prime >= 2", ctx.pos);
    return Value::from_int(static_cast<std::int64_t>(bernoulli_mod(w, ell)));
}

// The Glover / Ash-Stevens composition factors of Sym^n(F_ell^2)
// (HeckeExt/GloverFactors; [E] Sec. identities item 6).
//
// THE det-TWIST CONVENTION IS THE PIPELINE'S STANDARD-MODULE ONE ([WP4-7] F3):
// `Sym^n` here is the module `ComputeSymmetricPowerMatrixFM` acts on, i.e. the
// symmetric power of the STANDARD representation and not of its dual.  The two
// differ by det^{-n}, which would shift every reported `det_twist`, so the
// convention is part of what the verb returns and is stated in its doc string.
// It is the same convention slot-5 WP-11 is writing into [E] Sec. 2.3 item 6.
Value VerbGloverFactors(CallContext& ctx, const std::vector<Value>& a) {
    static const char* kWho = "glover_factors(n, ell)";
    const uint32_t n = console::bind::U32Arg(a[0], ctx.pos, kWho);
    const uint32_t ell = console::bind::U32Arg(a[1], ctx.pos, kWho);
    if (ell < 2) throw TypeError(std::string(kWho) + ": ell must be a prime >= 2", ctx.pos);
    const GloverDecomposition d = GloverFactors(n, ell);
    Record r;
    r.set("n", Value::from_int(d.n));
    r.set("ell", Value::from_int(d.ell));
    r.set("dimension", Value::from_int(d.dimension()));
    r.set("length", Value::from_int(d.length()));
    std::vector<Value> factors;
    for (const GloverFactor& f : d.factors) {
        Record fr;
        fr.set("sym", Value::from_int(f.sym));
        fr.set("det_twist", Value::from_int(f.det_twist));
        fr.set("multiplicity", Value::from_int(f.multiplicity));
        fr.set("dimension", Value::from_int(f.dimension()));
        factors.push_back(Value::from_record(std::move(fr)));
    }
    r.set("factors", Value::from_list(std::move(factors)));
    std::vector<Value> sl2;
    for (uint32_t m : d.sl2_multiplicities()) sl2.push_back(Value::from_int(m));
    r.set("sl2_multiplicities", Value::from_list(std::move(sl2)));
    r.set("text", Value::from_string(d.to_string()));
    r.set("json", Value::from_string(d.to_json()));
    return Value::from_record(std::move(r));
}

// (C2a, slot-6 WP-13) C_{ell,r} as the `field_polynomial` SETTING in force
// selects it -- HeckeExt/HeckeCanonicalField's FieldPolynomialFromSettings,
// the key's one consumer.  This is the use that makes `set field_polynomial =
// "conway"` observable ([P] Sec. 10 C2 (g), the exit-5 arm): the setting
// itself is LEGAL (it parses, validates and moves the fingerprint), and the
// consumer raises hecke::NotImplementedInThisBuild, which the C7 seam maps to
// exit code 5 -- never a bad-value error and never a silent fall-back to
// `least`.  Before this verb no console path reached the consumer at all.
Value VerbCanonicalFieldPolynomial(CallContext& ctx, const std::vector<Value>& a) {
    static const char* kWho = "canonical_field_polynomial(ell, r)";
    const uint32_t ell = console::bind::U32Arg(a[0], ctx.pos, kWho);
    const uint32_t r = console::bind::U32Arg(a[1], ctx.pos, kWho);
    if (ell < 2) throw TypeError(std::string(kWho) + ": ell must be a prime >= 2", ctx.pos);
    if (r < 1) throw TypeError(std::string(kWho) + ": the degree r must be at least 1", ctx.pos);
    return console::bind::PolyValue(ctx, FieldPolynomialFromSettings(ell, r));
}

void RegisterSturmTypes(Registry& registry) {
    TypeInfo t;
    t.name = kTypeSturm;
    t.doc = "the computed Sturm bound B'(k, ell) of [E] def:bprime: the least prime bound at "
            "which the F_ell-ALGEBRA generated by T_q, q <= B, reaches dim M_k";
    t.members.push_back({"k", false, "the weight k = n + 2", [](const ObjectPtr& o) {
                             return Value::from_int(o->payload_as<SturmVal>()->k);
                         }});
    t.members.push_back({"ell", false, "the coefficient prime", [](const ObjectPtr& o) {
                             return Value::from_int(o->payload_as<SturmVal>()->ell);
                         }});
    t.members.push_back({"outcome", true, "found / none_at_cap / in_progress",
                         [](const ObjectPtr& o) {
                             return Value::from_string(ToString(Bound(o).outcome()));
                         }});
    t.members.push_back({"found", true, "whether B' exists at or below the cap",
                         [](const ObjectPtr& o) {
                             return Value::from_bool(Bound(o).has_B_prime());
                         }});
    t.members.push_back({"B_prime", true, "B', meaningful iff `found`", [](const ObjectPtr& o) {
                             return Value::from_int(Bound(o).B_prime());
                         }});
    t.members.push_back({"cap", true, "the cap in force (12*B(k) by default)",
                         [](const ObjectPtr& o) { return Value::from_int(Bound(o).cap()); }});
    t.members.push_back({"dim_M_k", true, "dim M_k(F_ell)", [](const ObjectPtr& o) {
                             return Value::from_int(Bound(o).dim_M_k());
                         }});
    t.members.push_back({"closure_dim", true, "the dimension the algebra closure reached",
                         [](const ObjectPtr& o) {
                             return Value::from_int(Bound(o).closure_dim());
                         }});
    t.members.push_back({"primes_done", true, "the primes processed, ascending",
                         [](const ObjectPtr& o) {
                             std::vector<Value> out;
                             for (uint32_t q : Bound(o).primes_done())
                                 out.push_back(Value::from_int(q));
                             return Value::from_list(std::move(out));
                         }});
    t.members.push_back({"certified", true,
                         "SturmBoundCertificate::verify() == 0, re-derived from the reported data",
                         [](const ObjectPtr& o) {
                             return Value::from_bool(Bound(o).certificate().verify() == 0);
                         }});
    t.members.push_back({"verify", true, "the certificate's code (0 = every clause re-derives)",
                         [](const ObjectPtr& o) {
                             return Value::from_int(
                                 static_cast<std::int64_t>(Bound(o).certificate().verify()));
                         }});
    t.members.push_back({"json", true, "the certificate's JSON, `resumable` object included",
                         [](const ObjectPtr& o) {
                             return Value::from_string(Bound(o).certificate().to_json());
                         }});
    // R3: `resume(x, budget)` -- the new handle continues from THIS one's state
    // under a higher cap, so budget_history is [old, new] and never [new].
    t.resume = [](const ObjectPtr& parent,
                  const console::SettingsSnapshot& snap) -> ObjectPtr {
        SturmVal& pv = Sturm(parent);
        pv.ensure();
        const Value* cap_v = snap.find("sturm.cap");
        if (cap_v == nullptr || !cap_v->is_int()) {
            throw RuntimeError("resume(x, budget): a Sturm handle is resumed by its `sturm.cap` "
                               "budget, which the new snapshot does not carry",
                               parent->creation_site());
        }
        // (C2a, slot-6 WP-13) A cap below the reached prime is refused HERE,
        // eagerly and with the engine's own named text ("resume: budget below
        // reached state", SturmBound::resume), rather than as a lazy compute
        // failure of the new handle ([P] Sec. 10 C2 (h)).
        const std::int64_t want = cap_v->int_value();
        if (want < 0 || static_cast<std::uint64_t>(want) < pv.bound->state().B_reached) {
            throw RuntimeError("resume: budget below reached state (this Sturm handle reached "
                               "B = " + std::to_string(pv.bound->state().B_reached) +
                                   ", asked for cap = " + std::to_string(want) + ")",
                               parent->creation_site());
        }
        auto nv = std::make_shared<SturmVal>();
        nv->k = pv.k;
        nv->ell = pv.ell;
        nv->cap = pv.cap;
        nv->from_parent.reset(new SturmBound(*pv.bound));
        nv->resume_cap = static_cast<uint32_t>(cap_v->int_value());
        ObjectPtr o = MakeSturmHandle(parent->type(), parent->creation_site(),
                                      parent->creation_expr(), snap, nv);
        o->keep_alive(parent);
        return o;
    };
    t.print = [](const ObjectPtr& o, std::int64_t) {
        const SturmVal* sv = o->payload_as<SturmVal>();
        std::ostringstream os;
        os << "Sturm bound B'(" << sv->k << ", " << sv->ell << ")";
        if (!o->computed()) {
            os << " (declared, not yet computed)";
        } else if (sv->bound->has_B_prime()) {
            os << " = " << sv->bound->B_prime();
        } else {
            os << " = " << ToString(sv->bound->outcome());
        }
        return os.str();
    };
    registry.register_type(std::move(t));
}

} // namespace verbs_detail
} // namespace hecke_ext
