// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Hecke.cpp -- console phase C1a: `hecke` and `family` (plan Sec. 3.6, rows
// HeckeOperator and Family).
//
// It registers NO `decompose` and NO `report` verb, deliberately: those are
// exactly the objects the extension stage's single golden re-baseline moves, so
// they are phase C1b's (plan Sec. 10, "Why the split"; [X] Sec. 8, "no
// decomposition transcript before E0b").  `crosschecks` waits with them.
//
// The pipeline recipe is BenchDriver.cpp:299-326's, unchanged and through the
// public API only: HeckeCosetTables -> HeckeTauTable (which CERTIFIES
// d tau = tau d exactly in its constructor) -> HeckeOperatorOnK::compute(),
// which itself runs the runtime assertions (i)/(ii)/(iii) and the O12(a)
// two-route bit-identity on every call.  A console `hecke(H, p)` is therefore
// as certified as a `--bench hecke` run, and `.certificates` reports the
// counters it already produced.
//
// tau construction takes hecke::tau_construction_mutex() (the reason is at
// CampaignDriver.h:73-76: tau construction is first-match-wins over a fixed
// order and drives the NON-ATOMIC PredicateStats counters), even though the
// console is single-threaded at the script level, so a future concurrent verb
// cannot race it.
// ===========================================================================

#include "Bindings.h"
#include "Payloads.h"
#include "../../Hecke/HeckeSettings.h"   // (slot-8 WP-3) the refusal at dispatch

#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace console {
namespace bind {

// --- the payload builders (Payloads.h) -------------------------------------

// (C1b) The Hecke action is defined on the 7-vertex SL(2,Z) model; a preset
// local system (LocalSystemPresets, bound in Presets.cpp) has no tessellation
// model at all, so `hecke` / `family` on it is a USER error with a named
// message, not an internal CONSOLE_CHECK.  This is the console's form of the
// model mismatch HeckeModelDescriptor.h:76-80 raises one level down.
void RequireSL2ZModel(const CohomologyVal& cv, const char* who) {
    CONSOLE_CHECK(cv.L != nullptr);
    const LocalSystemVal* lv = cv.L->payload_as<LocalSystemVal>();
    if (lv->model == nullptr) {
        throw RuntimeError(std::string(who) +
                           ": this cohomology was built from the preset local system '" +
                           lv->preset +
                           "', which carries no SL(2,Z) tessellation model -- the Hecke action is "
                               "defined on the 7-vertex model only",
                           SourcePos{});
    }
}

void HeckeVal::ensure() {
    if (op || digit_rec || op64) return;
    CONSOLE_CHECK(H != nullptr);
    CohomologyVal* cv = H->payload_as<CohomologyVal>();
    if (digit) {
        // (slot-11 WP-18c; [E] v1.12.0-pre.2 (h)) the digit route with the prime.
        const LocalSystemVal* lv = cv->L->payload_as<LocalSystemVal>();
        hecke::ladic_cell::Request rq;
        rq.n_sym = lv->n_sym;
        rq.ell = lv->ell;
        rq.k = ring_k;
        rq.primes = {p};
        rq.hida = false;
        rq.width = hecke::ladic_cell::EngineLadicWidth();
        // (S13 WP-12) The record is a pure function of (n, ell, k, width, p): a second hecke(H, p)
        // with the same key -- another handle, or the same call again -- reuses it (a bounded
        // process-wide cache of the last four).  (S18 WP-19b) A different p shares the cell: the
        // prime-independent steps 1-2 live in a hecke::ladic_cell::DigitCellSession (a bounded
        // process-wide cache of the last two cells), and only the Hecke step runs per p.
        // session.run({p}) is RunCell(rq) by construction (RunCell is that call).
        using Key = std::tuple<uint32_t, uint64_t, uint32_t, int, uint64_t, bool, int>;
        static std::mutex mu;
        static std::vector<std::pair<Key, std::shared_ptr<const hecke::ladic_cell::CellRecord>>> cache;
        // (S13 WP-14, A2) the whole Request: hida and precision_source too.  The cache outlives a
        // Session by design (value-neutral: the record is a pure function of the key).
        const Key key{rq.n_sym, rq.ell, rq.k, static_cast<int>(rq.width), p, rq.hida,
                      static_cast<int>(rq.precision_source)};
        // The cell key: the record key without p.
        using CellKey = std::tuple<uint32_t, uint64_t, uint32_t, int, bool, int>;
        static std::vector<std::pair<CellKey, std::shared_ptr<hecke::ladic_cell::DigitCellSession>>> cells;
        const CellKey ckey{rq.n_sym, rq.ell, rq.k, static_cast<int>(rq.width), rq.hida,
                           static_cast<int>(rq.precision_source)};
        std::shared_ptr<hecke::ladic_cell::DigitCellSession> session;
        {
            std::lock_guard<std::mutex> lk(mu);
            for (const auto& e : cache)
                if (e.first == key) { digit_rec = e.second; return; }
            for (const auto& e : cells)
                if (e.first == ckey) session = e.second;
        }
        if (!session) {
            session = std::make_shared<hecke::ladic_cell::DigitCellSession>(rq);   // throws as RunCell does
            std::lock_guard<std::mutex> lk(mu);
            if (cells.size() >= 2) cells.erase(cells.begin());
            cells.emplace_back(ckey, session);
        }
        digit_rec = std::make_shared<const hecke::ladic_cell::CellRecord>(session->run(rq.primes));
        {
            std::lock_guard<std::mutex> lk(mu);
            if (cache.size() >= 4) cache.erase(cache.begin());
            cache.emplace_back(key, digit_rec);
        }
        return;
    }
    cv->ensure();
    RequireSL2ZModel(*cv, "hecke(H, p)");
    const ModelVal& mv = ModelOf(*cv);
    // (C5a, slot-9 WP-12) (n, ell) from the LocalSystem, not from the
    // cohomology object: over Z/ell^k the object's modulus is ell^k.  On an
    // F_ell handle they are the same numbers.
    const LocalSystemVal* lv = cv->L->payload_as<LocalSystemVal>();
    const uint32_t ell = lv->ell;
    const uint32_t n_sym = lv->n_sym;
    // (C5a) A Z/ell^k operator -- tables, tau and the operator -- is built
    // under exactly its cohomology's ring (the golden harness's
    // AddE8Operators); an F_ell operator exactly as before.
    std::unique_ptr<hecke::ScopedCoefficientRing> ring;
    if (ring_k >= 2) {
        CONSOLE_CHECK_MSG(cv->ring_k == ring_k, "a ring HeckeOperator over a Cohomology of another ring");
        ring.reset(new hecke::ScopedCoefficientRing(ring_k, hecke::SettingSource::Console));
    }
    if (wide) {
        // (slot-12 WP-17; [P] v3.9.3 Q9 decided (a)) the wide band: the
        // uint64_t tables, tau and operator -- the campaign job loop's recipe
        // (RunRingPipeline<uint64_t>, r = 1), Theta both ways asserted equal.
        CONSOLE_CHECK_MSG(cv->wide && cv->coh64 != nullptr, "a wide HeckeOperator over a non-wide Cohomology");
        {
            std::lock_guard<std::mutex> guard(hecke::tau_construction_mutex());
            cosets64.reset(new hecke::HeckeCosetTables64(p, r, static_cast<uint64_t>(ell), n_sym));
            tau64.reset(new hecke::HeckeTauTable64(mv.K, *mv.model, *cosets64, static_cast<uint64_t>(ell), n_sym));
        }
        op64.reset(new hecke::HeckeOperatorOnK64(cv->coh64.get(), cosets64.get(), tau64.get()));
        op64->compute();
        return;
    }
    {
        std::lock_guard<std::mutex> guard(hecke::tau_construction_mutex());
        cosets.reset(new hecke::HeckeCosetTables(p, r, ell, n_sym));
        tau.reset(new hecke::HeckeTauTable(mv.K, *mv.model, *cosets, ell, n_sym));
    }
    op.reset(new hecke::HeckeOperatorOnK(cv->coh.get(), cosets.get(), tau.get()));
    op->compute();
}

void FamilyVal::ensure() {
    if (fam) return;
    CONSOLE_CHECK(H != nullptr);
    CohomologyVal* cv = H->payload_as<CohomologyVal>();
    cv->ensure();
    RequireSL2ZModel(*cv, "family(H)");
    const ModelVal& mv = ModelOf(*cv);
    fam.reset(new hecke::HeckeOperatorFamily(cv->coh.get(), mv.K, *mv.model,
                                             cv->coh->get_prime(),
                                             cv->coh->get_representation_dim() - 1));
}

namespace {

using Flat = FlatMatrixT<uint32_t>;

HeckeVal& Op(const ObjectPtr& o) { return *o->payload_as<HeckeVal>(); }
FamilyVal& Fam(const ObjectPtr& o) { return *o->payload_as<FamilyVal>(); }

const TypeInfo* g_matrix_type = nullptr;

Value MatrixValueFrom(const ObjectPtr& owner, Flat m) {
    auto p = std::make_shared<MatrixVal>();
    p->m = std::move(m);
    return Value::from_object(MakeValueObjectFrom(g_matrix_type, owner, std::move(p)));
}

// A per-degree operator list (A^{(0)}, A^{(1)}, ...) as a console List of
// Matrix.  The golden harness hashes each degree SEPARATELY (`A(T_p)^q`), so a
// pin indexes into this list rather than hashing it whole.
Value DegreeList(const ObjectPtr& owner, const std::vector<MatrixT<uint32_t>>& A) {
    std::vector<Value> out;
    out.reserve(A.size());
    for (const MatrixT<uint32_t>& m : A) out.push_back(MatrixValueFrom(owner, Flat(m)));
    return Value::from_list(std::move(out));
}

uint32_t DegreeArg(const Value& v, const SourcePos& pos, const char* who, std::size_t n) {
    const uint32_t q = U32Arg(v, pos, who);
    if (n == 0 || q >= n) {
        throw RuntimeError(std::string(who) + ": degree " + std::to_string(q) +
                               " is not a degree of this complex (it has " + std::to_string(n) +
                               ")",
                           pos);
    }
    return q;
}

// --- verbs -----------------------------------------------------------------

Value VerbHecke(CallContext& ctx, const std::vector<Value>& a) {
    const ObjectPtr& H = AsObjectOf(a[0], kTypeCohomology, ctx.pos, "hecke(H, p, r)");
    const uint64_t p = U64Arg(a[1], ctx.pos, "hecke(H, p, r)");
    uint32_t r = 1;
    if (a.size() > 2 && !a[2].is_nil()) r = U32Arg(a[2], ctx.pos, "hecke(H, p, r)");
    // (slot-8 WP-3 -> C5a, slot-9 WP-12) [E] sec:ring (h): `hecke(H, p)` with
    // r = 1 HONOURS a Z/ell^k Cohomology under the same ring (p = ell
    // included: U_ell).  Refused AT DISPATCH, exit 5: r >= 2 under a ring, and
    // any mismatch between the ring in force and the Cohomology's.
    const uint32_t k = hecke::CoefficientRingK("console verb hecke");
    const uint32_t kH = HandleRingK(H);
    if (k != kH) {
        hecke::RefuseCoefficientRing("console pipeline verb hecke", k >= 2 ? k : kH,
                                     " (coefficient_ring = " + hecke::CoefficientRingText(k) +
                                         " is in force, but the Cohomology was built over " +
                                         hecke::CoefficientRingText(kH) + ")");
    }
    if (k >= 2 && r >= 2) {
        hecke::RefuseCoefficientRing("console pipeline verb hecke(H, p, r) at r = " + std::to_string(r), k,
                                     " -- only r = 1 honours the ring ([E] sec:ring (h))");
    }
    if (p < 2) throw TypeError("hecke(H, p, r): p must be a prime >= 2", ctx.pos);
    if (r < 1) throw TypeError("hecke(H, p, r): r must be at least 1", ctx.pos);

    auto v = std::make_shared<HeckeVal>();
    v->H = H;
    v->p = p;
    v->r = r;
    v->ring_k = k;
    v->digit = H->payload_as<CohomologyVal>()->digit;
    v->wide = H->payload_as<CohomologyVal>()->wide;   // (slot-12 WP-17)
    ObjectPtr obj = Object::create(TypeByName(ctx, kTypeHecke), ctx.pos, ctx.call_text,
                                   ctx.session->settings().snapshot(), v,
                                   LazyCompute([v](Object&) { v->ensure(); }));
    obj->keep_alive(H);
    return Value::from_object(obj);
}

Value VerbFamily(CallContext& ctx, const std::vector<Value>& a) {
    hecke::RequireFieldCoefficientRing("console pipeline verb family");   // (slot-8 WP-3), exit 5
    const ObjectPtr& H = AsObjectOf(a[0], kTypeCohomology, ctx.pos, "family(H)");
    RequireFieldHandle(H, "console pipeline verb family");                // (C5a) a ring handle too
    auto v = std::make_shared<FamilyVal>();
    v->H = H;
    ObjectPtr obj = Object::create(TypeByName(ctx, kTypeFamily), ctx.pos, ctx.call_text,
                                   ctx.session->settings().snapshot(), v,
                                   LazyCompute([v](Object&) { v->ensure(); }));
    obj->keep_alive(H);
    return Value::from_object(obj);
}

// --- types -----------------------------------------------------------------

void RegisterTypes(Registry& registry) {
    g_matrix_type = registry.find_type(kTypeMatrix);
    CONSOLE_CHECK_MSG(g_matrix_type != nullptr,
                      "RegisterCoreMathBindings must run before RegisterHeckeBindings");
    {
        TypeInfo t;
        t.name = kTypeHecke;
        t.doc = "T_p (or U_ell, the same path) on H^*: the coset tables, the certified tau datum "
                "and the assembled Theta / A^{(q)}";
        t.members.push_back({"p", false, "the Hecke prime", [](const ObjectPtr& o) {
                                 return Value::from_int(
                                     static_cast<std::int64_t>(o->payload_as<HeckeVal>()->p));
                             }});
        t.members.push_back({"r", false, "the exponent of N = p^r", [](const ObjectPtr& o) {
                                 return Value::from_int(o->payload_as<HeckeVal>()->r);
                             }});
        t.members.push_back({"cosets", true, "the number of cosets m", [](const ObjectPtr& o) {
                                 return Value::from_int(static_cast<std::int64_t>(
                                     Op(o).cosets->num_cosets()));
                             }});
        t.members.push_back(
            {"certificates", true,
             "the in-process certificate bundle compute() already ran (O12(a) and the counters)",
             [](const ObjectPtr& o) {
                 const auto c = Op(o).op->certificates();
                 Record r;
                 r.set("routes_bit_identical", Value::from_bool(c.routes_bit_identical));
                 r.set("relation_checks",
                       Value::from_int(static_cast<std::int64_t>(c.relation_checks)));
                 r.set("cocycle_checks",
                       Value::from_int(static_cast<std::int64_t>(c.cocycle_checks)));
                 r.set("theta_eq_solves",
                       Value::from_int(static_cast<std::int64_t>(c.theta_eq_solves)));
                 r.set("theta_eq_residual_max",
                       Value::from_int(static_cast<std::int64_t>(c.theta_eq_residual_max)));
                 r.set("theta_eq_certified", Value::from_bool(c.theta_eq_certified));
                 return Value::from_record(std::move(r));
             }});
        t.members.push_back({"json", true, "the operator's own report JSON (levels (a) + (b))",
                             [](const ObjectPtr& o) {
                                 return Value::from_string(Op(o).op->to_json());
                             }});
        t.members.push_back({"dims", true, "dim H^q of the underlying complex",
                             [](const ObjectPtr& o) {
                                 std::vector<uint32_t> d;
                                 for (const auto& m : Op(o).op->get_H_matrices())
                                     d.push_back(m.rows);
                                 return IntList(d);
                             }});
        {
            TypeInfo::Method m;
            m.name = "A";
            m.params = {"q"};
            m.required = 1;
            m.computed = true;
            m.doc = "A^{(q)}(T_p), the cohomology-level matrix in degree q (over Z/ell^k: the (c5) "
                    "matrix on the cyclic summands, a List of rows, entry (i, j) in [0, ell^{e_i}))";
            m.call = [](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                if (Op(o).ring_k >= 2) {
                    // (C5a) Rows over different moduli ell^{e_i}: a List of Int
                    // rows, never a Matrix -- a Matrix's field algorithms
                    // (rank, charpoly, ...) would be meaningless on it.
                    CohomologyVal* cv = Op(o).H->payload_as<CohomologyVal>();
                    const uint32_t q = DegreeArg(args[0], pos, "HeckeOperator.A(q)", cv->coh->num_degrees());
                    const engine::DenseBuf<uint32_t>& M = Op(o).op->H_matrix_module(q);
                    std::vector<Value> rows;
                    for (uint32_t i = 0; i < M.rows; ++i) {
                        std::vector<uint32_t> row;
                        for (uint32_t j = 0; j < M.cols; ++j) row.push_back(M.at(i, j));
                        rows.push_back(IntList(row));
                    }
                    return Value::from_list(std::move(rows));
                }
                const auto& A = Op(o).op->get_H_matrices();
                const uint32_t q = DegreeArg(args[0], pos, "HeckeOperator.A(q)", A.size());
                return MatrixValueFrom(o, Flat(A[q]));
            };
            t.methods.push_back(std::move(m));
        }
        {
            TypeInfo::Method m;
            m.name = "Theta";
            m.params = {"q"};
            m.required = 1;
            m.computed = true;
            m.doc = "Theta^{(q)}_flat on the flat cochains C^q(K; V)";
            m.call = [](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                const auto& T = Op(o).op->get_Theta_flat_matrices();
                const uint32_t q = DegreeArg(args[0], pos, "HeckeOperator.Theta(q)", T.size());
                return MatrixValueFrom(o, T[q]);
            };
            t.methods.push_back(std::move(m));
        }
        {
            TypeInfo::Method m;
            m.name = "Theta_eq";
            m.params = {"q"};
            m.required = 1;
            m.computed = true;
            m.doc = "Theta^{(q)} restricted to C^q_eq in the stored C_eq basis";
            m.call = [](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                const auto& T = Op(o).op->get_Theta_eq_matrices();
                const uint32_t q = DegreeArg(args[0], pos, "HeckeOperator.Theta_eq(q)", T.size());
                return MatrixValueFrom(o, T[q]);
            };
            t.methods.push_back(std::move(m));
        }
        {
            TypeInfo::Method m;
            m.name = "apply";
            m.params = {"q", "v"};
            m.required = 2;
            m.computed = true;
            m.doc = "Theta^{(q)} applied to a flat cochain (the fused route)";
            m.call = [](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                const auto& T = Op(o).op->get_Theta_flat_matrices();
                const uint32_t q = DegreeArg(args[0], pos, "HeckeOperator.apply(q, v)", T.size());
                const MatrixVal& v = AsMatrix(args[1], pos, "HeckeOperator.apply(q, v)");
                return MatrixValueFrom(o, Op(o).op->apply_Theta(q, v.m));
            };
            t.methods.push_back(std::move(m));
        }
        {
            TypeInfo::Method m;
            m.name = "apply_route2";
            m.params = {"q", "v"};
            m.required = 2;
            m.computed = true;
            m.doc = "the INDEPENDENT second route of [C, eq:heckechain] on the same cochain "
                    "(matrix-free; it does not read the fused tables)";
            m.call = [](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                const auto& T = Op(o).op->get_Theta_flat_matrices();
                const uint32_t q =
                    DegreeArg(args[0], pos, "HeckeOperator.apply_route2(q, v)", T.size());
                const MatrixVal& v = AsMatrix(args[1], pos, "HeckeOperator.apply_route2(q, v)");
                return MatrixValueFrom(o, Op(o).op->apply_route2(q, v.m));
            };
            t.methods.push_back(std::move(m));
        }
        // --- (C5a, slot-9 WP-12) [E] sec:ring (h): the ring members -------------
        t.members.push_back({"coefficient_ring", false,
                             "the ring's canonical text: F_ell, or Z/ell^<k> ([E] sec:ring (a))",
                             [](const ObjectPtr& o) {
                                 return Value::from_string(
                                     hecke::CoefficientRingText(o->payload_as<HeckeVal>()->ring_k));
                             }});
        t.members.push_back(
            {"charpoly_free", true,
             "over Z/ell^k: {precision, coeffs} -- the charpoly of A^{(1)}(T_p) on the free quotient, "
             "over Z/ell^precision, coefficients low -> high; defined only at a certified level",
             [](const ObjectPtr& o) {
                 if (Op(o).ring_k < 2) {
                     throw std::logic_error("HeckeOperator.charpoly_free: a Z/ell^k (module) member, but this "
                                            "operator is over F_ell ([E] sec:ring (h))");
                 }
                 const auto fc = Op(o).op->free_charpoly_H1();
                 std::vector<Value> coeffs;
                 for (uint32_t c : fc.coeffs) coeffs.push_back(Value::from_int(static_cast<std::int64_t>(c)));
                 Record r;
                 r.set("precision", Value::from_int(fc.precision));
                 r.set("coeffs", Value::from_list(std::move(coeffs)));
                 return Value::from_record(std::move(r));
             }});
        t.print = [](const ObjectPtr& o, std::int64_t) {
            const HeckeVal* v = o->payload_as<HeckeVal>();
            std::ostringstream os;
            os << "HeckeOperator T_" << v->p;
            if (v->r != 1) os << "^" << v->r << " (primitive double coset)";
            if (v->ring_k >= 2) {
                const LocalSystemVal* lv =
                    v->H->payload_as<CohomologyVal>()->L->payload_as<LocalSystemVal>();
                os << " over Z/" << lv->ell << "^" << v->ring_k;
            }
            if (!o->computed()) os << " (declared, not yet computed)";
            return os.str();
        };
        // (C5a) On a Z/ell^k operator only (h)'s members honour; every other
        // member refuses, exit 5, naming itself.  (slot-10 WP-12; [E] v1.11
        // (h), [S10] decision 6 (ii) (A)) The list gains the two structural
        // reads `p` and `r` (always 1 on a ring handle: hecke(H, p, r) at
        // r >= 2 is refused at dispatch), which force no computation.
        GateTypeOnRing(
            t,
            [](const std::string& name) {
                return (name == "coefficient_ring" || name == "A" || name == "charpoly_free" ||
                        name == "p" || name == "r")
                           ? RingRule::Honour
                           : RingRule::Refuse;
            },
            [](const ObjectPtr& o) { return o->payload_as<HeckeVal>()->ring_k; });
        // (slot-11 WP-18c; [E] v1.12.0-pre.2 (h)) a digit handle: charpoly_free
        // from the route's record (coefficients as Strings in (g)'s residue
        // form); every member but coefficient_ring / p / r refuses (exit 5).
        {
            std::map<std::string, std::function<Value(const ObjectPtr&)>> dm;
            dm["charpoly_free"] = [](const ObjectPtr& o) {
                const HeckeVal& v = Op(o);
                CONSOLE_CHECK(v.digit_rec != nullptr && v.digit_rec->hecke.size() == 1);
                const hecke::ladic_cell::HeckeOut& h = v.digit_rec->hecke.front();
                std::vector<Value> coeffs;
                for (const auto& c : h.charpoly)
                    coeffs.push_back(Value::from_string(hecke::ladic_cell::ResidueText(c, v.digit_rec->ell)));
                Record r;
                r.set("precision", Value::from_int(h.precision));
                r.set("coeffs", Value::from_list(std::move(coeffs)));
                return Value::from_record(std::move(r));
            };
            GateTypeOnDigit(t, {"coefficient_ring", "p", "r"}, {}, dm, {},
                            [](const ObjectPtr& o) { return o->payload_as<HeckeVal>()->digit; },
                            [](const ObjectPtr& o) { return o->payload_as<HeckeVal>()->ring_k; });
        }
        // (slot-12 WP-17; [P] v3.9.3 Q9 decided (a)) a WIDE operator (2^32 <=
        // ell^k < 2^64): A(q) and charpoly_free read off the uint64_t operator;
        // an entry that does not fit int64_t (ell^k > 2^63) is its decimal
        // String.  Every other member: GateTypeOnRing's (p, r, coefficient_ring
        // honour, the rest exit 5).
        {
            auto op64 = [](const ObjectPtr& o) -> const hecke::HeckeOperatorOnK64& {
                CONSOLE_CHECK(Op(o).wide && Op(o).op64 != nullptr);
                return *Op(o).op64;
            };
            std::map<std::string, std::function<Value(const ObjectPtr&)>> wm;
            wm["charpoly_free"] = [op64](const ObjectPtr& o) {
                const auto fc = op64(o).free_charpoly_H1();
                std::vector<Value> coeffs;
                for (uint64_t c : fc.coeffs) coeffs.push_back(WideIntValue(c));
                Record r;
                r.set("precision", Value::from_int(fc.precision));
                r.set("coeffs", Value::from_list(std::move(coeffs)));
                return Value::from_record(std::move(r));
            };
            std::map<std::string, std::function<Value(const ObjectPtr&, const std::vector<Value>&, const SourcePos&)>> wmm;
            wmm["A"] = [op64](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                CohomologyVal* cv = Op(o).H->payload_as<CohomologyVal>();
                const uint32_t q = DegreeArg(args[0], pos, "HeckeOperator.A(q)", cv->coh64->num_degrees());
                const engine::DenseBuf<uint64_t>& M = op64(o).H_matrix_module(q);
                std::vector<Value> rows;
                for (uint32_t i = 0; i < M.rows; ++i) {
                    std::vector<Value> row;
                    for (uint32_t j = 0; j < M.cols; ++j) row.push_back(WideIntValue(M.at(i, j)));
                    rows.push_back(Value::from_list(std::move(row)));
                }
                return Value::from_list(std::move(rows));
            };
            GateTypeOnWide(t, wm, wmm, [](const ObjectPtr& o) { return o->payload_as<HeckeVal>()->wide; },
                           [](const ObjectPtr& o) { return o->payload_as<HeckeVal>()->ring_k; });
        }
        registry.register_type(std::move(t));
    }
    {
        TypeInfo t;
        t.name = kTypeFamily;
        t.doc = "the Hecke operator family on one cohomology: T_p, T_{p^r} (recursion and the "
                "independent direct route), T_n and the diamond T(p,p)";
        auto degree_list_method = [](const char* name, const char* doc, int n_args) {
            TypeInfo::Method m;
            m.name = name;
            m.required = static_cast<std::size_t>(n_args);
            m.computed = true;
            m.doc = doc;
            return m;
        };
        {
            TypeInfo::Method m = degree_list_method("T", "T_p per degree (this IS U_ell at p = ell)", 1);
            m.params = {"p"};
            m.call = [](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                return DegreeList(o, Fam(o).fam->T_p(U64Arg(args[0], pos, "Family.T(p)")));
            };
            t.methods.push_back(std::move(m));
        }
        {
            TypeInfo::Method m = degree_list_method(
                "T_pow", "the FULL T_{p^r} by the algebra recursion of [C, thm:relations]", 2);
            m.params = {"p", "r"};
            m.call = [](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                return DegreeList(o, Fam(o).fam->T_p_pow(U64Arg(args[0], pos, "Family.T_pow(p, r)"),
                                                         U32Arg(args[1], pos, "Family.T_pow(p, r)")));
            };
            t.methods.push_back(std::move(m));
        }
        {
            TypeInfo::Method m = degree_list_method(
                "T_pow_direct",
                "the same operator by the INDEPENDENT double-coset route -- what T_pow is "
                "cross-checked against",
                2);
            m.params = {"p", "r"};
            m.call = [](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                return DegreeList(
                    o, Fam(o).fam->T_p_pow_direct(U64Arg(args[0], pos, "Family.T_pow_direct(p, r)"),
                                                  U32Arg(args[1], pos, "Family.T_pow_direct(p, r)")));
            };
            t.methods.push_back(std::move(m));
        }
        {
            TypeInfo::Method m = degree_list_method(
                "T_pow_primitive",
                "the PRIMITIVE double-coset operator T'_{p^r} (equal to T_p at r = 1; it differs "
                "from the standard T_{p^r} for r >= 2)",
                2);
            m.params = {"p", "r"};
            m.call = [](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                return DegreeList(o,
                                  Fam(o).fam->T_p_pow_primitive_direct(
                                      U64Arg(args[0], pos, "Family.T_pow_primitive(p, r)"),
                                      U32Arg(args[1], pos, "Family.T_pow_primitive(p, r)")));
            };
            t.methods.push_back(std::move(m));
        }
        {
            TypeInfo::Method m = degree_list_method("T_n", "T_n for a composite index n", 1);
            m.params = {"n"};
            m.call = [](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                return DegreeList(o, Fam(o).fam->T_n(U64Arg(args[0], pos, "Family.T_n(n)")));
            };
            t.methods.push_back(std::move(m));
        }
        {
            TypeInfo::Method m =
                degree_list_method("T_pp", "the diamond T(p,p) = p^n I ([C, prop:diamond])", 1);
            m.params = {"p"};
            m.call = [](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                return DegreeList(o, Fam(o).fam->T_pp(U64Arg(args[0], pos, "Family.T_pp(p)")));
            };
            t.methods.push_back(std::move(m));
        }
        {
            TypeInfo::Method m;
            m.name = "theta";
            m.params = {"p", "q"};
            m.required = 2;
            m.computed = true;
            m.doc = "Theta^{(q)}_flat of the family's r = 1 pipeline for p";
            m.call = [](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                const uint64_t p = U64Arg(args[0], pos, "Family.theta(p, q)");
                const uint32_t q = U32Arg(args[1], pos, "Family.theta(p, q)");
                return MatrixValueFrom(o, Fam(o).fam->theta_flat_fm(p, q));
            };
            t.methods.push_back(std::move(m));
        }
        t.print = [](const ObjectPtr& o, std::int64_t) {
            if (!o->computed()) return std::string("Family (declared, not yet computed)");
            return std::string("Family on H^* (T_p, T_{p^r}, T_n, T(p,p))");
        };
        registry.register_type(std::move(t));
    }
}

} // namespace

void RegisterHeckeBindings(Registry& registry) {
    RegisterTypes(registry);

    const char* g = "pipeline";
    auto add = [&registry, g](const char* name, Signature sig, VerbImpl impl, const char* doc) {
        registry.register_verb(name, std::move(sig), std::move(impl), doc, g);
    };

    add("hecke", Sig({Param::object("H", kTypeCohomology), Param::of("p", ValueKind::Int)}),
        &VerbHecke, "T_p on H^* (this IS U_ell when p == ell -- the same code path)");
    add("hecke",
        Sig({Param::object("H", kTypeCohomology), Param::of("p", ValueKind::Int),
             Param::of("r", ValueKind::Int).optional(Value::from_int(1))}),
        &VerbHecke, "the primitive double-coset pipeline for N = p^r");
    add("family", Sig({Param::object("H", kTypeCohomology)}), &VerbFamily,
        "the Hecke operator family on H^* (32-bit only)");
}

} // namespace bind
} // namespace console
