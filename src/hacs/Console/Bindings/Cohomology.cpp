// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Cohomology.cpp -- console phase C1a: `symmetric_power`, `cohomology` and the
// Cohomology handle (plan Sec. 3.6, rows LocalSystem and Cohomology), plus the
// four PER-COMPLEX degree members and `dickson_exceptional` (inventory gap G9).
//
// The two handles are LAZY (plan Sec. 4.2): `cohomology(L)` binds a declared
// handle and `compute_cohomology()` runs on the first access of a COMPUTED
// member, under the handle's creation-time settings snapshot.  So a script can
// build a grid of objects and pay only for the ones it reads.
//
// THE FOUR DEGREE MEMBERS ARE PER COMPLEX, NEVER PER RECORD (plan Sec. 3.6,
// [C] sec:recordschema item 5, slot-2 owner confirmation 1): `.q_max`,
// `.truncated`, `.q_max_source` and `.degrees` belong to THIS complex and are
// DERIVED from its own degree count -- the very `get_H_eq_dimensions().size()`
// the pipeline reports -- so no literal `2` is ever written.
//
// (C5a, slot-9 WP-12; [E] v1.10.0-pre.1 sec:ring (h), decision 2 (A)) THE RING
// LIFT.  Under `coefficient_ring = Z/ell^k` at ell >= 5, `symmetric_power` and
// `cohomology` HONOUR the ring and return handles over Z/ell^k (E8's type); at
// ell in {2, 3} they keep exit 5 (the console lifts nothing there in slot 9).
// The ring is read ONCE, at the verb, and stored in the payload (ring_k); the
// payload's ensure() installs exactly that ring around the construction, so a
// handle's ring never depends on who forces it, and a verb handed a handle of
// another ring refuses at dispatch (hecke::NotImplementedInThisBuild, exit 5 --
// the pipeline's own classification of a ring mismatch, HeckeOperatorOnK.h).
// On a Z/ell^k Cohomology the members of (h)'s list honour; the FIELD members
// (dims, C, Z, B, H, P) raise the object's own named std::logic_error (exit
// 3; there is no dim over Z/ell^k); every other member refuses with exit 5,
// naming itself.  Nothing changes on an F_ell handle except the new members.
// ===========================================================================

#include "Bindings.h"
#include "Payloads.h"

#include "../../Hecke/CoefficientRing.h"
#include "../../Hecke/DicksonCount.h"
#include "../../Hecke/HeckeReduction.h"
#include "../../LocalSystemSL2Z.h"
#include "../../Hecke/HeckeSettings.h"

#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace console {
namespace bind {

// --- the payload builders (Payloads.h) -------------------------------------

void LocalSystemVal::ensure() {
    if (lssc || lssc64) return;
    if (digit) return;   // (slot-11 WP-18c) the digit route builds no local system object
    CONSOLE_CHECK(model != nullptr);
    const ModelVal* mv = model->payload_as<ModelVal>();
    if (wide) {
        // (slot-12 WP-17; [P] v3.9.3 Q9 decided (a)) 2^32 <= ell^k < 2^64: the
        // native pipeline's uint64_t instantiation -- the campaign job loop's
        // own recipe (Campaign/CampaignRing.cpp, RunRingPipeline<uint64_t>).
        CONSOLE_CHECK(ring_k >= 2);
        const hecke::ScopedCoefficientRing ring(ring_k, hecke::SettingSource::Console);
        const engine::ModCtx<uint64_t> ctx =
            hecke::CoefficientRingContext<uint64_t>("console pipeline verb symmetric_power", ell, ring_k);
        lssc64.reset(new LocalSystemSimplicialComplex64(
            mv->K, hecke::GenerateSL2ZRelationsRing<uint64_t>(mv->K, n_sym, ctx), n_sym + 1, ctx.p));
        return;
    }
    if (ring_k >= 2) {
        // (C5a, slot-9 WP-12; [E] sec:ring (h)) The ring's own relation list,
        // GenerateSL2ZRelationsRing over Z/ell^k -- built under exactly the ring
        // this handle was dispatched under.
        const hecke::ScopedCoefficientRing ring(ring_k, hecke::SettingSource::Console);
        const engine::ModCtx<uint32_t> ctx =
            hecke::CoefficientRingContext<uint32_t>("console pipeline verb symmetric_power", ell, ring_k);
        lssc.reset(new LocalSystemSimplicialComplex(
            mv->K, hecke::GenerateSL2ZRelationsRing<uint32_t>(mv->K, n_sym, ctx), n_sym + 1, ctx.p));
        return;
    }
    // (slot-8 WP-3) [E] sec:ring (b4): an F_ell handle refuses a non-default
    // coefficient_ring (exit 5).
    hecke::RequireFieldCoefficientRing("console pipeline verbs (symmetric_power)");
    // BenchDriver.cpp:252-258's recipe: the relation list with its group data,
    // every rho_n(g_r) DERIVED (invariant I3), never entered by hand.
    std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels;
    for (const auto& d : GenerateSL2ZRelationsWithGroupData(mv->K, n_sym, ell)) {
        rels.push_back(d.rel);
    }
    lssc.reset(new LocalSystemSimplicialComplex(mv->K, std::move(rels), n_sym + 1, ell));
}

void CohomologyVal::ensure() {
    if (coh || digit_rec || coh64) return;
    CONSOLE_CHECK(L != nullptr);
    LocalSystemVal* lv = L->payload_as<LocalSystemVal>();
    if (wide) {
        // (slot-12 WP-17) the wide band, under exactly the handle's ring.
        CONSOLE_CHECK_MSG(lv->wide && lv->ring_k == ring_k, "a wide Cohomology over a non-wide LocalSystem");
        lv->ensure();
        const hecke::ScopedCoefficientRing ring(ring_k, hecke::SettingSource::Console);
        coh64.reset(new EquivariantCohomologyOnK64(lv->lssc64.get()));
        coh64->compute_cohomology();
        return;
    }
    if (digit) {
        // (slot-11 WP-18c; [E] v1.12.0-pre.2 (h)) the digit route: the cohomology
        // and its certified precision bound (PrecisionInsufficient is exit 5).
        CONSOLE_CHECK_MSG(lv->digit && lv->ring_k == ring_k, "a digit Cohomology over a non-digit LocalSystem");
        hecke::ladic_cell::Request rq;
        rq.n_sym = lv->n_sym;
        rq.ell = lv->ell;
        rq.k = ring_k;
        rq.width = hecke::ladic_cell::EngineLadicWidth();
        digit_rec = std::make_shared<const hecke::ladic_cell::CellRecord>(hecke::ladic_cell::RunCell(rq));
        return;
    }
    if (ring_k >= 2) {
        CONSOLE_CHECK_MSG(lv->ring_k == ring_k, "a ring Cohomology over a LocalSystem of another ring");
        lv->ensure();
        const hecke::ScopedCoefficientRing ring(ring_k, hecke::SettingSource::Console);
        coh.reset(new EquivariantCohomologyOnK(lv->lssc.get()));
        coh->compute_cohomology();
        return;
    }
    hecke::RequireFieldCoefficientRing("console pipeline verbs (cohomology)");   // (slot-8 WP-3)
    lv->ensure();
    coh.reset(new EquivariantCohomologyOnK(lv->lssc.get()));
    coh->compute_cohomology();
}

// (C5a) The F_ell object at the same (n, ell), with F_ell installed for its
// construction only -- exactly the object an F_ell `cohomology(L)` builds.
const EquivariantCohomologyOnK& CohomologyVal::field_comparison() {
    if (coh_field) return *coh_field;
    CONSOLE_CHECK(L != nullptr);
    const LocalSystemVal* lv = L->payload_as<LocalSystemVal>();
    CONSOLE_CHECK(lv->model != nullptr);
    const ModelVal* mv = lv->model->payload_as<ModelVal>();
    const hecke::ScopedCoefficientRing field(1);
    std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels;
    for (const auto& d : GenerateSL2ZRelationsWithGroupData(mv->K, lv->n_sym, lv->ell)) {
        rels.push_back(d.rel);
    }
    lssc_field.reset(new LocalSystemSimplicialComplex(mv->K, std::move(rels), lv->n_sym + 1, lv->ell));
    coh_field.reset(new EquivariantCohomologyOnK(lssc_field.get()));
    coh_field->compute_cohomology();
    return *coh_field;
}

// (C5a) The Z/ell^2 object the (mod-ell) Bockstein is computed from: this
// handle's own object at k = 2, else one built with Z/ell^2 installed for its
// construction only ([E] sec:ring (g), the RunE8Point precedent).
const EquivariantCohomologyOnK& CohomologyVal::ring2_comparison() {
    CONSOLE_CHECK(ring_k >= 2 && coh != nullptr);
    if (ring_k == 2) return *coh;
    if (coh_ring2) return *coh_ring2;
    const LocalSystemVal* lv = L->payload_as<LocalSystemVal>();
    CONSOLE_CHECK(lv->model != nullptr);
    const ModelVal* mv = lv->model->payload_as<ModelVal>();
    const hecke::ScopedCoefficientRing ring(2, hecke::SettingSource::Console);
    const engine::ModCtx<uint32_t> ctx =
        hecke::CoefficientRingContext<uint32_t>("console Cohomology.bockstein_rank", lv->ell, 2);
    lssc_ring2.reset(new LocalSystemSimplicialComplex(
        mv->K, hecke::GenerateSL2ZRelationsRing<uint32_t>(mv->K, lv->n_sym, ctx), lv->n_sym + 1, ctx.p));
    coh_ring2.reset(new EquivariantCohomologyOnK(lssc_ring2.get()));
    coh_ring2->compute_cohomology();
    return *coh_ring2;
}

// (slot-12 WP-17) The wide band's two comparison objects, in the uint64_t
// width, exactly as the campaign job loop builds them (field_object /
// ring2_object of RunRingPipeline<uint64_t>): F_ell at (n, ell), and Z/ell^2.
const EquivariantCohomologyOnK64& CohomologyVal::field_comparison64() {
    if (coh_field64) return *coh_field64;
    CONSOLE_CHECK(L != nullptr);
    const LocalSystemVal* lv = L->payload_as<LocalSystemVal>();
    CONSOLE_CHECK(lv->model != nullptr);
    const ModelVal* mv = lv->model->payload_as<ModelVal>();
    const hecke::ScopedCoefficientRing field(1);
    const uint64_t ell = lv->ell;
    lssc_field64.reset(new LocalSystemSimplicialComplex64(
        mv->K, GenerateSL2ZRelationsElt<uint64_t>(mv->K, lv->n_sym, ell), lv->n_sym + 1, ell));
    coh_field64.reset(new EquivariantCohomologyOnK64(lssc_field64.get()));
    coh_field64->compute_cohomology();
    return *coh_field64;
}

const EquivariantCohomologyOnK64& CohomologyVal::ring2_comparison64() {
    CONSOLE_CHECK(wide && ring_k >= 2 && coh64 != nullptr);
    if (coh_ring264) return *coh_ring264;
    const LocalSystemVal* lv = L->payload_as<LocalSystemVal>();
    CONSOLE_CHECK(lv->model != nullptr);
    const ModelVal* mv = lv->model->payload_as<ModelVal>();
    const hecke::ScopedCoefficientRing ring(2, hecke::SettingSource::Console);
    const engine::ModCtx<uint64_t> ctx =
        hecke::CoefficientRingContext<uint64_t>("console Cohomology.bockstein_rank", lv->ell, 2);
    lssc_ring264.reset(new LocalSystemSimplicialComplex64(
        mv->K, hecke::GenerateSL2ZRelationsRing<uint64_t>(mv->K, lv->n_sym, ctx), lv->n_sym + 1, ctx.p));
    coh_ring264.reset(new EquivariantCohomologyOnK64(lssc_ring264.get()));
    coh_ring264->compute_cohomology();
    return *coh_ring264;
}

uint32_t HandleRingK(const ObjectPtr& handle) {
    if (!handle || handle->type() == nullptr) return 1;
    const std::string& n = handle->type()->name;
    if (n == kTypeLocalSys) return handle->payload_as<LocalSystemVal>()->ring_k;
    if (n == kTypeCohomology) return handle->payload_as<CohomologyVal>()->ring_k;
    if (n == kTypeHecke) return handle->payload_as<HeckeVal>()->ring_k;
    return 1;
}

void RequireFieldHandle(const ObjectPtr& handle, const std::string& who) {
    const uint32_t k = HandleRingK(handle);
    if (k >= 2) {
        hecke::RefuseCoefficientRing(who, k,
                                     " (the " + handle->type_name() + " handle was built over Z/ell^" +
                                         std::to_string(k) + "; [E] sec:ring (h) lists what honours it)");
    }
}

const ModelVal& ModelOf(const CohomologyVal& coh) {
    CONSOLE_CHECK(coh.L != nullptr);
    const LocalSystemVal* lv = coh.L->payload_as<LocalSystemVal>();
    CONSOLE_CHECK(lv->model != nullptr);
    return *lv->model->payload_as<ModelVal>();
}

namespace {

using Flat = FlatMatrixT<uint32_t>;

CohomologyVal& Coh(const ObjectPtr& o) { return *o->payload_as<CohomologyVal>(); }

const TypeInfo* g_matrix_type = nullptr;

Value MatrixValueFrom(const ObjectPtr& owner, Flat m) {
    auto p = std::make_shared<MatrixVal>();
    p->m = std::move(m);
    return Value::from_object(MakeValueObjectFrom(g_matrix_type, owner, std::move(p)));
}

// A basis (a list of N_q x d_V cochains) as a console List of Matrix.  That is
// the shape Golden/GoldenHarness.cpp's `Collector::basis` hashes -- count, then
// each matrix -- so `hash(H.C(0))` is comparable with
// `golden("n=... ell=...", "C_eq^0").hash` (plan Sec. 6, "every pin through
// golden(...)").
Value BasisList(const ObjectPtr& owner, const std::vector<Flat>& basis) {
    std::vector<Value> out;
    out.reserve(basis.size());
    for (const Flat& f : basis) out.push_back(MatrixValueFrom(owner, f));
    return Value::from_list(std::move(out));
}

uint32_t DegreeArg(const std::vector<Value>& args, const SourcePos& pos, const char* who,
                   std::size_t n_degrees) {
    const uint32_t q = U32Arg(args[0], pos, who);
    if (n_degrees == 0 || q >= n_degrees) {
        throw RuntimeError(std::string(who) + ": degree " + std::to_string(q) +
                               " is not a degree of this complex (it has " +
                               std::to_string(n_degrees) + ")",
                           pos);
    }
    return q;
}

// --- (C5a) the ring lift ------------------------------------------------------

// (slot-11 WP-18c) the precision bound's keys as a console Record.
Value PrecisionRecord(const hecke::ladic_cell::Precision& P) {
    Record r;
    r.set("precision_requested", Value::from_int(P.requested));
    r.set("precision_working", Value::from_int(P.working));
    r.set("e_max", Value::from_int(P.e_max));
    r.set("word_level", Value::from_int(P.word_level));
    r.set("precision_certified", Value::from_bool(P.certified));
    r.set("digit_width", Value::from_int(P.digit_width));
    r.set("digits_L", Value::from_int(P.digits_L));
    return Value::from_record(std::move(r));
}

bool IsRing(const ObjectPtr& o) { return Coh(o).ring_k >= 2; }

uint32_t CohEll(const ObjectPtr& o) {
    const CohomologyVal& cv = Coh(o);
    CONSOLE_CHECK(cv.L != nullptr);
    return cv.L->payload_as<LocalSystemVal>()->ell;
}

// "Z/691^2" / "F_691": the ring with its prime, for printers.
std::string RingWithPrime(uint32_t ell, uint32_t k) {
    return k >= 2 ? "Z/" + std::to_string(ell) + "^" + std::to_string(k) : "F_" + std::to_string(ell);
}

// A module (Z/ell^k) member read on an F_ell handle: the object's kind of
// error, a named std::logic_error (exit 3).
[[noreturn]] void ThrowModuleOnField(const std::string& who) {
    throw std::logic_error(who + ": a Z/ell^k (module) member, but this Cohomology is over F_ell "
                                 "([E] sec:ring (h))");
}

// [E] sec:ring (h): the Cohomology members a Z/ell^k handle honours, the FIELD
// members (the object's own named logic_error, exit 3), and -- by elimination --
// the members that refuse with exit 5.  A member added later refuses until (h)
// lists it.
RingRule CohomologyRingRule(const std::string& name) {
    static const std::set<std::string> honour = {
        "computed", "q_max", "truncated", "q_max_source", "degrees", "coefficient_ring",
        "lengths", "invariant_factors", "ladic_type_certified", "r_K", "ladic_type",
        "bockstein_rank", "higher_torsion_count", "red"};
    static const std::set<std::string> field = {"dims", "C", "Z", "B", "H", "P"};
    if (honour.count(name) != 0) return RingRule::Honour;
    if (field.count(name) != 0) return RingRule::Field;
    return RingRule::Refuse;
}

// The ladic_level state of a Z/ell^k handle as the R3 resumable state the
// core's `state` / `save_state` read ([E] sec:ring (d), (h)).  `resumable` means
// "incomplete" (the level is uncertified); its continuation is a FRESH run at a
// larger k, so the type's `resume` refuses (RegisterTypes).
template<class CohT>
void FillLadicState(Object& self, const CohomologyVal& cv, const CohT& coh) {
    if (!coh.ladic_applicable()) return;
    const typename CohT::LadicLevelState st = coh.ladic_level_state();
    ResumableState rs;
    rs.resumable = !st.complete;
    rs.budget_setting = "coefficient_ring";
    rs.budget_value = Value::from_string(hecke::CoefficientRingText(cv.ring_k));
    if (!st.complete) {
        rs.reason = "the K-mode is uncertified at K = " + std::to_string(st.budget_K) + ": r_K(d^0) = " +
                    std::to_string(st.reached_r0) + " (needs " + std::to_string(st.need_r0) +
                    "), r_K(d^1) = " + std::to_string(st.reached_r1) + " (needs " +
                    std::to_string(st.need_r1) + "); the continuation is a fresh run at a larger k";
    }
    Record budget;
    budget.set("K", Value::from_int(st.budget_K));
    Record reached;
    reached.set("r_K0", Value::from_int(st.reached_r0));
    reached.set("r_K1", Value::from_int(st.reached_r1));
    Record need;
    need.set("r_K0", Value::from_int(st.need_r0));
    need.set("r_K1", Value::from_int(st.need_r1));
    Record progress;
    progress.set("kind", Value::from_string(st.kind));
    progress.set("budget", Value::from_record(std::move(budget)));
    progress.set("reached", Value::from_record(std::move(reached)));
    progress.set("need", Value::from_record(std::move(need)));
    progress.set("complete", Value::from_bool(st.complete));
    rs.progress = std::move(progress);
    self.set_resumable_state(std::move(rs));
}

// --- verbs -----------------------------------------------------------------

Value VerbSymmetricPower(CallContext& ctx, const std::vector<Value>& a) {
    const ObjectPtr& model = AsObjectOf(a[0], kTypeModel, ctx.pos, "symmetric_power(M, n, ell)");
    const uint32_t n = U32Arg(a[1], ctx.pos, "symmetric_power(M, n, ell)");
    const uint32_t ell = U32Arg(a[2], ctx.pos, "symmetric_power(M, n, ell)");
    // (slot-8 WP-3 -> C5a, slot-9 WP-12) [E] sec:ring (h): the verb HONOURS a
    // non-default coefficient_ring at ell >= 5, and refuses it AT DISPATCH at
    // ell in {2, 3} (exit 5 -- the console lifts nothing there in slot 9).
    const uint32_t k = hecke::CoefficientRingK("console verb symmetric_power");
    if (k >= 2 && hecke::IsBredonPrime(ell)) {
        hecke::RefuseCoefficientRing("console pipeline verb symmetric_power", k,
                                     " at ell = " + std::to_string(ell) +
                                         " ([E] sec:ring (h): the console lifts nothing at ell in {2, 3})");
    }
    // (slot-14 WP-14b, A14 at symmetric_power) the other two sites of the
    // "composite ell accepted at dispatch" finding (dickson_les, ext_record)
    // were fixed at slot 13 with hecke::IsPrimeU64; this is the third,
    // re-carried because it needed CoefficientRing.h, which this file
    // already includes (see above) -- so the fix is now the same one line.
    if (ell < 2 || !hecke::IsPrimeU64(ell)) throw TypeError("symmetric_power(M, n, ell): ell must be a prime >= 2", ctx.pos);
    // (slot-11 WP-18c; [E] v1.12.0-pre.2 (h), the per-cell rule) ell^k >= 2^64 at
    // ell >= 5 is the DIGIT route; a cell it does not execute refuses here.
    const bool digit = k >= 2 && !hecke::ladic_cell::FitsU64(ell, k);
    if (digit)
        hecke::ladic_cell::RequireDigitCell("console pipeline verb symmetric_power", n, ell,
                                            k + hecke::ladic_cell::WordLevelMax(ell) - 1,   // K's bound (m-3)
                                            hecke::ladic_cell::EngineLadicWidth());

    auto p = std::make_shared<LocalSystemVal>();
    p->n_sym = n;
    p->ell = ell;
    p->model = model;
    p->ring_k = k;
    p->digit = digit;
    // (slot-12 WP-17; [P] v3.9.3 Q9 decided (a)) 2^32 <= ell^k < 2^64 at ell >= 5:
    // served by the uint64_t instantiation (was: refused by the uint32_t
    // context at the first computed access -- the console's uint32_t gap).
    p->wide = IsWideCell(ell, k);
    ObjectPtr obj = Object::create(TypeByName(ctx, kTypeLocalSys), ctx.pos, ctx.call_text,
                                   ctx.session->settings().snapshot(), p,
                                   LazyCompute([p](Object&) { p->ensure(); }));
    obj->keep_alive(model);
    return Value::from_object(obj);
}

Value VerbCohomology(CallContext& ctx, const std::vector<Value>& a) {
    const ObjectPtr& L = AsObjectOf(a[0], kTypeLocalSys, ctx.pos, "cohomology(L)");
    // (slot-8 WP-3 -> C5a) [E] sec:ring (h): honours a Z/ell^k LocalSystem under
    // the same ring.  A mismatch between the ring in force and L's is refused
    // at dispatch, exit 5, naming both.
    const uint32_t k = hecke::CoefficientRingK("console verb cohomology");
    const uint32_t kL = L->payload_as<LocalSystemVal>()->ring_k;
    if (k != kL) {
        hecke::RefuseCoefficientRing("console pipeline verb cohomology", k >= 2 ? k : kL,
                                     " (coefficient_ring = " + hecke::CoefficientRingText(k) +
                                         " is in force, but the LocalSystem was built over " +
                                         hecke::CoefficientRingText(kL) +
                                         "; build the local system under the ring of its cohomology)");
    }
    auto p = std::make_shared<CohomologyVal>();
    p->L = L;
    p->ring_k = k;
    p->digit = L->payload_as<LocalSystemVal>()->digit;
    p->wide = L->payload_as<LocalSystemVal>()->wide;   // (slot-12 WP-17)
    ObjectPtr obj = Object::create(TypeByName(ctx, kTypeCohomology), ctx.pos, ctx.call_text,
                                   ctx.session->settings().snapshot(), p,
                                   LazyCompute([p](Object& self) {
                                       p->ensure();
                                       if (p->ring_k >= 2 && p->wide) FillLadicState(self, *p, *p->coh64);
                                       else if (p->ring_k >= 2 && !p->digit) FillLadicState(self, *p, *p->coh);
                                   }));
    obj->keep_alive(L);
    return Value::from_object(obj);
}

Value VerbDickson(CallContext& ctx, const std::vector<Value>& a) {
    const uint32_t n = U32Arg(a[0], ctx.pos, "dickson_exceptional(n, ell)");
    const uint32_t ell = U32Arg(a[1], ctx.pos, "dickson_exceptional(n, ell)");
    return Value::from_bool(hecke::IsDicksonExceptionalCount(n, ell));
}

Value VerbDicksonCount(CallContext& ctx, const std::vector<Value>& a) {
    const uint32_t n = U32Arg(a[0], ctx.pos, "dickson_count(n, ell)");
    const uint32_t ell = U32Arg(a[1], ctx.pos, "dickson_count(n, ell)");
    return Value::from_int(static_cast<std::int64_t>(hecke::DicksonInvariantCount(n, ell)));
}

// --- types -----------------------------------------------------------------

using BasisGetter = const std::vector<std::vector<Flat>>& (EquivariantCohomologyOnK::*)() const;

TypeInfo::Method BasisMethod(const char* name, const char* doc, BasisGetter get) {
    TypeInfo::Method m;
    m.name = name;
    m.params = {"q"};
    m.required = 1;
    m.computed = true;
    m.doc = doc;
    m.call = [get, name](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
        const auto& bases = (Coh(o).coh.get()->*get)();
        const std::string who = std::string("Cohomology.") + name + "(q)";
        const uint32_t q = DegreeArg(args, pos, who.c_str(), bases.size());
        return BasisList(o, bases[q]);
    };
    return m;
}

void RegisterTypes(Registry& registry) {
    g_matrix_type = registry.find_type(kTypeMatrix);
    CONSOLE_CHECK_MSG(g_matrix_type != nullptr,
                      "RegisterCoreMathBindings must run before RegisterCohomologyBindings");
    {
        TypeInfo t;
        t.name = kTypeLocalSys;
        t.doc = "Sym^n(F_ell^2) on a model, with its rho_n(g_r) relations (lazy)";
        t.members.push_back({"n", false, "the symmetric power", [](const ObjectPtr& o) {
                                 return Value::from_int(o->payload_as<LocalSystemVal>()->n_sym);
                             }});
        t.members.push_back({"p", false, "the coefficient prime ell", [](const ObjectPtr& o) {
                                 return Value::from_int(o->payload_as<LocalSystemVal>()->ell);
                             }});
        t.members.push_back({"d_V", false, "dim Sym^n = n + 1", [](const ObjectPtr& o) {
                                 return Value::from_int(o->payload_as<LocalSystemVal>()->n_sym + 1);
                             }});
        // (C5 leftovers, slot-10 WP-12; [E] v1.11 sec:ring (h), [P] v3.8 Sec. 3.6)
        // The one member slot 10 adds: the ring's canonical text, as on
        // Cohomology and HeckeOperator -- a read of the handle's stored ring.
        t.members.push_back({"coefficient_ring", false,
                             "the ring's canonical text: F_ell, or Z/ell^<k> ([E] sec:ring (a))",
                             [](const ObjectPtr& o) {
                                 return Value::from_string(hecke::CoefficientRingText(
                                     o->payload_as<LocalSystemVal>()->ring_k));
                             }});
        t.members.push_back({"relations", true, "the number of local-system relations",
                             [](const ObjectPtr& o) {
                                 LocalSystemVal* v = o->payload_as<LocalSystemVal>();
                                 v->ensure();
                                 return Value::from_int(static_cast<std::int64_t>(
                                     v->lssc->get_user_defined_simplex_relations().size()));
                             }});
        t.print = [](const ObjectPtr& o, std::int64_t) {
            const LocalSystemVal* v = o->payload_as<LocalSystemVal>();
            std::ostringstream os;
            if (v->ring_k >= 2) {
                os << "LocalSystem Sym^" << v->n_sym << "((" << RingWithPrime(v->ell, v->ring_k)
                   << ")^2), d_V = " << (v->n_sym + 1);
                return os.str();
            }
            os << "LocalSystem Sym^" << v->n_sym << "(F_" << v->ell << "^2), d_V = "
               << (v->n_sym + 1);
            return os.str();
        };
        // (C5a) [E] sec:ring (h): on a Z/ell^k handle every member not on the
        // list refuses (exit 5).  (slot-10 WP-12; [E] v1.11 (h), [S10] decision
        // 6 (ii) (A)) The list now holds three structural reads of the stored
        // data -- n, d_V (= n + 1, the rank of the free module Sym^n(R^2)) and
        // coefficient_ring -- none forcing a computation; `p` (it would read
        // ell under Z/ell^k) and `relations` keep refusing.
        GateTypeOnRing(
            t,
            [](const std::string& name) {
                return (name == "n" || name == "d_V" || name == "coefficient_ring")
                           ? RingRule::Honour
                           : RingRule::Refuse;
            },
            [](const ObjectPtr& o) { return o->payload_as<LocalSystemVal>()->ring_k; });
        registry.register_type(std::move(t));
    }
    {
        TypeInfo t;
        t.name = kTypeCohomology;
        t.doc = "H^q_eq(K, R; L) with its C / Z / B / H bases and projectors (lazy); over Z/ell^k "
                "(coefficient_ring, ell >= 5) its lengths, invariant factors and K-mode";

        t.members.push_back({"computed", false, "whether compute_cohomology() has run",
                             [](const ObjectPtr& o) { return Value::from_bool(o->computed()); }});
        t.members.push_back({"dims", true, "dim H^q for every degree q", [](const ObjectPtr& o) {
                                 return IntList(Coh(o).coh->get_H_eq_dimensions());
                             }});
        t.members.push_back({"ell", true, "the coefficient prime", [](const ObjectPtr& o) {
                                 return Value::from_int(Coh(o).coh->get_prime());
                             }});
        t.members.push_back({"d_V", true, "the representation dimension", [](const ObjectPtr& o) {
                                 return Value::from_int(Coh(o).coh->get_representation_dim());
                             }});
        // The four per-complex degree members (plan Sec. 3.6).  (C5a) Over
        // Z/ell^k the degree count is the module route's own, num_degrees().
        t.members.push_back({"q_max", true, "the top degree of THIS complex (derived)",
                             [](const ObjectPtr& o) {
                                 if (IsRing(o)) {
                                     return Value::from_int(
                                         static_cast<std::int64_t>(Coh(o).coh->num_degrees()) - 1);
                                 }
                                 const auto& d = Coh(o).coh->get_H_eq_dimensions();
                                 return Value::from_int(static_cast<std::int64_t>(d.size()) - 1);
                             }});
        t.members.push_back({"truncated", true, "false on a finite model", [](const ObjectPtr& o) {
                                 (void)Coh(o);
                                 return Value::from_bool(false);
                             }});
        t.members.push_back({"q_max_source", true, "'derived' unless truncated (E9)",
                             [](const ObjectPtr& o) {
                                 (void)Coh(o);
                                 return Value::from_string("derived");
                             }});
        t.members.push_back({"degrees", true, "[0 .. q_max], derived", [](const ObjectPtr& o) {
                                 const std::size_t n = IsRing(o)
                                                           ? Coh(o).coh->num_degrees()
                                                           : Coh(o).coh->get_H_eq_dimensions().size();
                                 std::vector<uint32_t> qs;
                                 for (uint32_t q = 0; q < n; ++q) qs.push_back(q);
                                 return IntList(qs);
                             }});

        t.methods.push_back(BasisMethod("C", "a basis of C^q_eq, one N_q x d_V cochain per element",
                                        &EquivariantCohomologyOnK::get_C_eq_bases_fm));
        t.methods.push_back(BasisMethod("Z", "a basis of Z^q_eq",
                                        &EquivariantCohomologyOnK::get_Z_eq_bases_fm));
        t.methods.push_back(BasisMethod("B", "a basis of B^q_eq",
                                        &EquivariantCohomologyOnK::get_B_eq_bases_fm));
        t.methods.push_back(BasisMethod("H", "a basis of H^q_eq",
                                        &EquivariantCohomologyOnK::get_H_eq_bases_fm));
        {
            TypeInfo::Method m;
            m.name = "P";
            m.params = {"q"};
            m.required = 1;
            m.computed = true;
            m.doc = "the projector C^q_flat -> H^q coordinates (legacy shape conventions)";
            m.call = [](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                const auto& P = Coh(o).coh->get_H_eq_projectors_from_C_eq_flat_fm();
                const uint32_t q = DegreeArg(args, pos, "Cohomology.P(q)", P.size());
                return MatrixValueFrom(o, P[q]);
            };
            t.methods.push_back(std::move(m));
        }
        {
            TypeInfo::Method m;
            m.name = "dim_C";
            m.params = {"q"};
            m.required = 1;
            m.computed = true;
            m.doc = "dim C^q_eq (available even in the memory-lean mode)";
            m.call = [](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                const auto& d = Coh(o).coh->get_H_eq_dimensions();
                const uint32_t q = DegreeArg(args, pos, "Cohomology.dim_C(q)", d.size());
                return Value::from_int(Coh(o).coh->C_eq_dimension(q));
            };
            t.methods.push_back(std::move(m));
        }

        // --- (C5a, slot-9 WP-12) E8: the Z/ell^k members of [E] sec:ring (h) ---
        t.members.push_back({"coefficient_ring", false,
                             "the ring's canonical text: F_ell, or Z/ell^<k> ([E] sec:ring (a))",
                             [](const ObjectPtr& o) {
                                 return Value::from_string(hecke::CoefficientRingText(Coh(o).ring_k));
                             }});
        t.members.push_back({"lengths", true,
                             "the composition length of H^q over Z/ell^k, q = 0, 1, 2 (lengths_H^q)",
                             [](const ObjectPtr& o) { return IntList(Coh(o).coh->H_lengths()); }});
        {
            TypeInfo::Method m;
            m.name = "invariant_factors";
            m.params = {"q"};
            m.required = 1;
            m.computed = true;
            m.doc = "H^q = (+)_i Z/ell^{e_i}: the exponents e_i, descending (invariant_factors_H^q)";
            m.call = [](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                if (!IsRing(o)) (void)Coh(o).coh->H_lengths();   // the object's own module-accessor error
                const uint32_t q = DegreeArg(args, pos, "Cohomology.invariant_factors(q)",
                                             Coh(o).coh->num_degrees());
                return IntList(Coh(o).coh->H_invariant_factors(q));
            };
            t.methods.push_back(std::move(m));
        }
        t.members.push_back({"ladic_type_certified", true,
                             "whether the K-mode certifies the Z_(ell)-type at K = k ([E] sec:ring (d))",
                             [](const ObjectPtr& o) {
                                 return Value::from_bool(Coh(o).coh->ladic_type_certified());
                             }});
        {
            TypeInfo::Method m;
            m.name = "r_K";
            m.params = {"q"};
            m.required = 1;
            m.computed = true;
            m.doc = "r_K(d^q) for q in {0, 1}: the K-mode's rank of the coboundary";
            m.call = [](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                const uint32_t q = U32Arg(args[0], pos, "Cohomology.r_K(q)");
                if (q > 1) {
                    throw RuntimeError("Cohomology.r_K(q): q must be 0 or 1 (r_K(d^0), r_K(d^1)), got " +
                                           std::to_string(q),
                                       pos);
                }
                return Value::from_int(Coh(o).coh->r_K(q));
            };
            t.methods.push_back(std::move(m));
        }
        {
            TypeInfo::Method m;
            m.name = "ladic_type";
            m.params = {"q"};
            m.required = 1;
            m.computed = true;
            m.doc = "{free_rank, torsion} of H^q(C)_(ell); defined only at a certified level (else "
                    "the object's named std::logic_error)";
            m.call = [](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                if (!IsRing(o)) (void)Coh(o).coh->H_lengths();
                const uint32_t q = DegreeArg(args, pos, "Cohomology.ladic_type(q)",
                                             Coh(o).coh->num_degrees());
                const EquivariantCohomologyOnK::LadicType lt = Coh(o).coh->ladic_type(q);
                Record r;
                r.set("free_rank", Value::from_int(lt.free_rank));
                r.set("torsion", IntList(lt.torsion));
                return Value::from_record(std::move(r));
            };
            t.methods.push_back(std::move(m));
        }
        // (slot-11 WP-18c; [E] v1.12.0-pre.2 (h)) the precision bound of a DIGIT
        // handle; every other handle refuses (exit 5).
        t.members.push_back({"precision", true,
                             "on a digit handle (ell^k >= 2^64): the certified precision bound -- "
                             "{precision_requested, precision_working, e_max, word_level, "
                             "precision_certified, digit_width, digits_L}",
                             [](const ObjectPtr& o) -> Value {
                                 throw hecke::NotImplementedInThisBuild(
                                     "Cohomology.precision: defined on a digit handle only (coefficient_ring = " +
                                     hecke::CoefficientRingText(Coh(o).ring_k) +
                                     " with ell^k < 2^64 is the native route; [E] v1.12.0-pre.2 sec:ring (h))");
                             }});
        t.members.push_back({"bockstein_rank", true,
                             "bockstein_rank^0: the rank of beta^0 on H^0(F_ell), from the Z/ell^2 "
                             "and F_ell objects ([E] sec:ring (d), (g))",
                             [](const ObjectPtr& o) {
                                 if (!IsRing(o)) ThrowModuleOnField("Cohomology.bockstein_rank");
                                 CohomologyVal& cv = Coh(o);
                                 const hecke::BocksteinBookkeeping bk =
                                     hecke::BocksteinCounts(cv.ring2_comparison(), cv.field_comparison());
                                 return Value::from_int(bk.bockstein_rank0);
                             }});
        t.members.push_back({"higher_torsion_count", true,
                             "dim H^0(F_ell) - bockstein_rank^0 ([E] sec:ring (d))",
                             [](const ObjectPtr& o) {
                                 if (!IsRing(o)) ThrowModuleOnField("Cohomology.higher_torsion_count");
                                 CohomologyVal& cv = Coh(o);
                                 const hecke::BocksteinBookkeeping bk =
                                     hecke::BocksteinCounts(cv.ring2_comparison(), cv.field_comparison());
                                 return Value::from_int(bk.higher_torsion_count);
                             }});
        {
            TypeInfo::Method m;
            m.name = "red";
            m.params = {"q"};
            m.required = 1;
            m.computed = true;
            m.doc = "red_H^q: H^q(Z/ell^k) -> H^q(F_ell), the (c5) matrix from the cyclic summands "
                    "to the F_ell object's D5b basis, over F_ell";
            m.call = [](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                if (!IsRing(o)) ThrowModuleOnField("Cohomology.red(q)");
                CohomologyVal& cv = Coh(o);
                const uint32_t q = DegreeArg(args, pos, "Cohomology.red(q)", cv.coh->num_degrees());
                const engine::DenseBuf<uint32_t> R = hecke::ReduceHToField(*cv.coh, cv.field_comparison(), q);
                const uint32_t ell = CohEll(o);
                Flat F(R.rows, R.cols, ell);
                for (uint32_t i = 0; i < R.rows; ++i)
                    for (uint32_t j = 0; j < R.cols; ++j) F.set_entry(i, j, R.at(i, j));
                return MatrixValueFrom(o, std::move(F));
            };
            t.methods.push_back(std::move(m));
        }
        // The ladic_level state's continuation is a FRESH run at a larger k:
        // `coefficient_ring` is not a budget ([E] sec:ring (d), (h); sec:schema's
        // rule for a changed non-budget setting).
        t.resume = [](const ObjectPtr& o, const SettingsSnapshot&) -> ObjectPtr {
            throw TypeError("resume under a different setting -> start a new computation: `coefficient_ring` "
                            "is not a budget -- the ladic_level state of this " +
                                hecke::CoefficientRingText(Coh(o).ring_k) +
                                " cohomology continues as a FRESH run at a larger k (set coefficient_ring = "
                                "\"Z/ell^<k+1>\" and build a new cohomology), never as a resume",
                            SourcePos());
        };

        t.print = [](const ObjectPtr& o, std::int64_t) {
            if (IsRing(o)) {
                const std::string ring = RingWithPrime(CohEll(o), Coh(o).ring_k);
                if (!o->computed()) return "Cohomology over " + ring + " (declared, not yet computed)";
                std::vector<uint32_t> len;
                if (Coh(o).digit_rec) {   // (slot-11 WP-18c) a digit handle
                    for (const auto& d : Coh(o).digit_rec->degrees) len.push_back(d.length_k);
                } else if (Coh(o).coh64) {   // (slot-12 WP-17) a wide handle
                    len = Coh(o).coh64->H_lengths();
                } else {
                    len = Coh(o).coh->H_lengths();
                }
                std::ostringstream os;
                os << "Cohomology over " << ring << ", lengths = [";
                for (std::size_t i = 0; i < len.size(); ++i) os << (i ? ", " : "") << len[i];
                os << "]";
                return os.str();
            }
            if (!o->computed()) return std::string("Cohomology (declared, not yet computed)");
            const auto& d = Coh(o).coh->get_H_eq_dimensions();
            std::ostringstream os;
            os << "Cohomology dims = [";
            for (std::size_t i = 0; i < d.size(); ++i) os << (i ? ", " : "") << d[i];
            os << "]";
            return os.str();
        };
        GateTypeOnRing(t, &CohomologyRingRule,
                       [](const ObjectPtr& o) { return Coh(o).ring_k; });
        // (slot-11 WP-18c; [E] v1.12.0-pre.2 (h)) a digit handle: the record's values.
        {
            auto rec = [](const ObjectPtr& o) -> const hecke::ladic_cell::CellRecord& {
                CONSOLE_CHECK(Coh(o).digit_rec != nullptr);
                return *Coh(o).digit_rec;
            };
            std::map<std::string, std::function<Value(const ObjectPtr&)>> dm;
            dm["q_max"] = [rec](const ObjectPtr& o) {
                return Value::from_int(static_cast<std::int64_t>(rec(o).degrees.size()) - 1);
            };
            dm["degrees"] = [rec](const ObjectPtr& o) {
                std::vector<uint32_t> qs;
                for (uint32_t q = 0; q < rec(o).degrees.size(); ++q) qs.push_back(q);
                return IntList(qs);
            };
            dm["lengths"] = [rec](const ObjectPtr& o) {
                std::vector<uint32_t> len;
                for (const auto& d : rec(o).degrees) len.push_back(d.length_k);
                return IntList(len);
            };
            dm["ladic_type_certified"] = [rec](const ObjectPtr& o) {
                return Value::from_bool(rec(o).precision.certified);
            };
            dm["precision"] = [rec](const ObjectPtr& o) { return PrecisionRecord(rec(o).precision); };
            std::map<std::string, std::function<Value(const ObjectPtr&, const std::vector<Value>&, const SourcePos&)>> dmm;
            dmm["invariant_factors"] = [rec](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                const uint32_t q = DegreeArg(args, pos, "Cohomology.invariant_factors(q)", rec(o).degrees.size());
                return IntList(rec(o).degrees[q].invariant_factors_k);
            };
            dmm["r_K"] = [rec](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                const uint32_t q = U32Arg(args[0], pos, "Cohomology.r_K(q)");
                if (q > 1) {
                    throw RuntimeError("Cohomology.r_K(q): q must be 0 or 1 (r_K(d^0), r_K(d^1)), got " +
                                           std::to_string(q),
                                       pos);
                }
                return Value::from_int(q == 0 ? rec(o).r_K0 : rec(o).r_K1);
            };
            dmm["ladic_type"] = [rec](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                const uint32_t q = DegreeArg(args, pos, "Cohomology.ladic_type(q)", rec(o).type.size());
                Record r;
                r.set("free_rank", Value::from_int(rec(o).type[q].free_rank));
                r.set("torsion", IntList(rec(o).type[q].torsion));
                return Value::from_record(std::move(r));
            };
            GateTypeOnDigit(t, {"computed", "truncated", "q_max_source", "coefficient_ring"},
                            {"dims", "C", "Z", "B", "H", "P"},   // RingRule::Field: exit 3 (M-1)
                            dm, dmm,
                            [](const ObjectPtr& o) { return Coh(o).digit; },
                            [](const ObjectPtr& o) { return Coh(o).ring_k; });
        }
        // (slot-12 WP-17; [P] v3.9.3 Q9 decided (a)) a WIDE handle (2^32 <=
        // ell^k < 2^64): the same members as a uint32_t ring handle, read off
        // the uint64_t objects -- the campaign job loop's own (RunRingPipeline
        // <uint64_t>); the field members call the object's own field accessor
        // (its named std::logic_error, exit 3); the refusals are GateTypeOnRing's.
        {
            auto c64 = [](const ObjectPtr& o) -> CohomologyVal& {
                CONSOLE_CHECK(Coh(o).wide && Coh(o).coh64 != nullptr);
                return Coh(o);
            };
            std::map<std::string, std::function<Value(const ObjectPtr&)>> wm;
            wm["q_max"] = [c64](const ObjectPtr& o) {
                return Value::from_int(static_cast<std::int64_t>(c64(o).coh64->num_degrees()) - 1);
            };
            wm["degrees"] = [c64](const ObjectPtr& o) {
                std::vector<uint32_t> qs;
                for (uint32_t q = 0; q < c64(o).coh64->num_degrees(); ++q) qs.push_back(q);
                return IntList(qs);
            };
            wm["lengths"] = [c64](const ObjectPtr& o) { return IntList(c64(o).coh64->H_lengths()); };
            wm["ladic_type_certified"] = [c64](const ObjectPtr& o) {
                return Value::from_bool(c64(o).coh64->ladic_type_certified());
            };
            wm["bockstein_rank"] = [c64](const ObjectPtr& o) {
                CohomologyVal& cv = c64(o);
                return Value::from_int(
                    hecke::BocksteinCounts(cv.ring2_comparison64(), cv.field_comparison64()).bockstein_rank0);
            };
            wm["higher_torsion_count"] = [c64](const ObjectPtr& o) {
                CohomologyVal& cv = c64(o);
                return Value::from_int(
                    hecke::BocksteinCounts(cv.ring2_comparison64(), cv.field_comparison64()).higher_torsion_count);
            };
            wm["dims"] = [c64](const ObjectPtr& o) -> Value {
                (void)c64(o).coh64->get_H_eq_dimensions();   // raises the object's field-accessor error
                throw std::logic_error("Cohomology.dims: a field (F_ell) accessor on a Z/ell^k object");
            };
            using WMethod = std::function<Value(const ObjectPtr&, const std::vector<Value>&, const SourcePos&)>;
            std::map<std::string, WMethod> wmm;
            wmm["invariant_factors"] = [c64](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                const uint32_t q =
                    DegreeArg(args, pos, "Cohomology.invariant_factors(q)", c64(o).coh64->num_degrees());
                return IntList(c64(o).coh64->H_invariant_factors(q));
            };
            wmm["r_K"] = [c64](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                const uint32_t q = U32Arg(args[0], pos, "Cohomology.r_K(q)");
                if (q > 1) {
                    throw RuntimeError("Cohomology.r_K(q): q must be 0 or 1 (r_K(d^0), r_K(d^1)), got " +
                                           std::to_string(q),
                                       pos);
                }
                return Value::from_int(c64(o).coh64->r_K(q));
            };
            wmm["ladic_type"] = [c64](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                const uint32_t q = DegreeArg(args, pos, "Cohomology.ladic_type(q)", c64(o).coh64->num_degrees());
                const auto lt = c64(o).coh64->ladic_type(q);
                Record r;
                r.set("free_rank", Value::from_int(lt.free_rank));
                r.set("torsion", IntList(lt.torsion));
                return Value::from_record(std::move(r));
            };
            wmm["red"] = [c64](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                CohomologyVal& cv = c64(o);
                const uint32_t q = DegreeArg(args, pos, "Cohomology.red(q)", cv.coh64->num_degrees());
                const engine::DenseBuf<uint64_t> R = hecke::ReduceHToField(*cv.coh64, cv.field_comparison64(), q);
                const uint32_t ell = CohEll(o);
                Flat F(R.rows, R.cols, ell);
                for (uint32_t i = 0; i < R.rows; ++i)
                    for (uint32_t j = 0; j < R.cols; ++j) F.set_entry(i, j, static_cast<uint32_t>(R.at(i, j)));
                return MatrixValueFrom(o, std::move(F));
            };
            // The field methods: the uint64_t object's own accessor raises.
            auto field_method = [c64](const char* name, int which) {
                return [c64, name, which](const ObjectPtr& o, const std::vector<Value>&, const SourcePos&) -> Value {
                    const EquivariantCohomologyOnK64& c = *c64(o).coh64;
                    switch (which) {
                        case 0: (void)c.get_C_eq_bases_fm(); break;
                        case 1: (void)c.get_Z_eq_bases_fm(); break;
                        case 2: (void)c.get_B_eq_bases_fm(); break;
                        case 3: (void)c.get_H_eq_bases_fm(); break;
                        default: (void)c.get_H_eq_projectors_from_C_eq_flat_fm(); break;
                    }
                    throw std::logic_error(std::string("Cohomology.") + name +
                                           "(q): a field (F_ell) accessor on a Z/ell^k object");
                };
            };
            wmm["C"] = field_method("C", 0);
            wmm["Z"] = field_method("Z", 1);
            wmm["B"] = field_method("B", 2);
            wmm["H"] = field_method("H", 3);
            wmm["P"] = field_method("P", 4);
            GateTypeOnWide(t, wm, wmm, [](const ObjectPtr& o) { return Coh(o).wide; },
                           [](const ObjectPtr& o) { return Coh(o).ring_k; });
        }
        registry.register_type(std::move(t));
    }
}

} // namespace

void RegisterCohomologyBindings(Registry& registry) {
    RegisterTypes(registry);

    const char* g = "pipeline";
    auto add = [&registry, g](const char* name, Signature sig, VerbImpl impl, const char* doc) {
        registry.register_verb(name, std::move(sig), std::move(impl), doc, g);
    };

    add("symmetric_power",
        Sig({Param::object("model", kTypeModel), Param::of("n", ValueKind::Int),
             Param::of("ell", ValueKind::Int)}),
        &VerbSymmetricPower, "the local system Sym^n(F_ell^2) on the model");
    add("cohomology", Sig({Param::object("L", kTypeLocalSys)}), &VerbCohomology,
        "H^*_eq(K, R; L) -- lazy: the work runs on the first computed access");
    add("dickson_exceptional",
        Sig({Param::of("n", ValueKind::Int), Param::of("ell", ValueKind::Int)}), &VerbDickson,
        "whether (n, ell) is on the Dickson-invariant exceptional locus");
    add("dickson_count",
        Sig({Param::of("n", ValueKind::Int), Param::of("ell", ValueKind::Int)}), &VerbDicksonCount,
        "#{(a, b) >= 0 : a(ell+1) + b(ell^2-ell) = n}");
}

} // namespace bind
} // namespace console
