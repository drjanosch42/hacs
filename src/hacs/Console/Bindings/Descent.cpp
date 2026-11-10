// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// DESCENT milestone D6: the console type DescentRecord and the verb descent(N, n, primes) -- see Descent.h and
// DescentRecord.h.  [P] = docs/console/00-console-interface-plan.md (Sec. 3.6 types, Sec. 8 errors, Sec. 18.8 read_only).
//
// ERRORS ([P] Sec. 8).  Under a ring other than Z: a RuntimeError at dispatch, exit 3 (the record is an object over Z and the
// setting selects it; it is never implied).  N = 0, n or N outside the cost bound, a primes List that is not a list of primes
// <= 97 avoiding N's divisors (at most 6, sum <= kMaxPrimeSum): TypeError, exit 3.  Inside the lazy compute: a refused part is NOT an error (the
// handle reports status "refused" and the reason); an ENGINE_CHECK (a consistency defect) is a runtime error, exit 3.

#include "Descent.h"

#include "../../Integral/Descent/DescentRecord.h"
#include "../../Integral/Descent/DescentModuleExport.h"
#include "../../Integral/Descent/DescentModuleExportZ.h"

#include "Bindings.h"
#include "../Core/Confirm.h"
#include "../Core/ConsoleError.h"
#include "../Core/Object.h"
#include "../Core/Registry.h"
#include "../Core/Session.h"
#include "../Core/Value.h"
#include "../../Hecke/HeckeSettings.h"
#include "../../NumberField/IntPolyFactor.h"
#include "../../Util/TextIo.h"

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace integral_descent {

namespace {

using console::CallContext;
using console::Object;
using console::ObjectPtr;
using console::Param;
using console::Record;
using console::Registry;
using console::RuntimeError;
using console::SourcePos;
using console::TypeError;
using console::TypeInfo;
using console::Value;
using console::ValueKind;
namespace rec = integral_descent::record;

constexpr const char* kTypeDescentRecord = "DescentRecord";
// The console cost bound of one cell: psi(N) (n + 1) <= kMaxModuleRank, n <= kMaxN, N <= kMaxLevel, and (D6 audit m2) the sum of the
// Hecke primes <= kMaxPrimeSum.  The cost model (DESCENT6_report.md Sec. 9, measured): wall ~ t0(N, n) + c * sum_{p in P} p with
// c ~ 0.1 .. 0.2 s per unit of p nearly independent of D = psi(N) (n + 1) (the T_p transports have p + 1 cosets on every piece and
// every torsion read), and the peak RSS grows with p * psi(N).  At the bound the measured worst cases are ~40 s ((23, 10), [97]) and
// ~1.4 GiB ((293, 0), [97]); the default primes have sum <= 31 inside the D bound.
constexpr uint64_t kMaxModuleRank = 300;
constexpr uint32_t kMaxN = 60;
constexpr uint32_t kMaxLevel = 300;
constexpr uint32_t kMaxPrimeSum = 100;

struct DescentRecordVal {
    uint32_t N = 1, n = 0;
    std::vector<uint32_t> ps;
    uint64_t rss_cap_bytes = rec::RecordOptions{}.rss_cap_bytes;   // (NBCONFIRM S3) the piece cap, see VerbDescent
    // (NBCONFIRM S4) set when the constructor was admitted OUTSIDE the silent envelope (confirmed or confirm = yes): the lazy
    // work below re-announces the estimate, so a served session shows its heartbeat (and the gateway its wall limit) also
    // when the first member read happens in a later cell.  Advisory only; never part of the record.
    std::shared_ptr<const console::cost::CostEstimate> confirmed;
    std::string confirmed_call;
    std::unique_ptr<rec::DescentRecordZ> R;
    std::unique_ptr<Value> factored;   // (NB-TOURFIX) the member charpoly_factored, computed once on first read
    // (slot-25 EXPORT) the method module(ell, K): one export per ell at the route's width, read at every K <= K_max
    std::map<uint32_t, std::shared_ptr<integral_descent::modexport::ModuleExport>> exports;
    // (slot-25 EXZ2-P3) the method module_Z(): EXPORT-Z step 2, computed once (its exports run in the global basis)
    std::shared_ptr<const integral_descent::modexport::ModuleZ> module_z2;
    void ensure() {
        if (R) return;
        if (confirmed) console::NoteConfirmedRun(confirmed_call + " (lazy work)", *confirmed);
        rec::RecordOptions o;
        o.ps = ps;
        o.rss_cap_bytes = rss_cap_bytes;   // (NBCONFIRM S3) = the RecordOptions default unless a confirmed cell lowered it
        R = std::make_unique<rec::DescentRecordZ>(rec::BuildDescentRecord(N, n, o));
    }
    // (S3 Fix round 1, audit m3) the exports of module / module_Z refuse at the same lowered cap as the record's pieces; inside
    // the envelope rss_cap_bytes is the 40 GiB default and min() leaves ExportOptions{} unchanged.
    integral_descent::modexport::ExportOptions export_options() const {
        integral_descent::modexport::ExportOptions eo;
        eo.rss_cap_bytes = std::min<uint64_t>(eo.rss_cap_bytes, rss_cap_bytes);
        return eo;
    }
};
DescentRecordVal& Val(const ObjectPtr& o) { return *o->payload_as<DescentRecordVal>(); }
const rec::DescentRecordZ& Rec(const ObjectPtr& o) { return *Val(o).R; }

using oc_util::HexU64;   // S24 CL-7 (F11): Util/TextIo.h
Value IntList(const std::vector<uint32_t>& v) {
    std::vector<Value> out;
    for (uint32_t x : v) out.push_back(Value::from_int(x));
    return Value::from_list(std::move(out));
}
Value GroupList(const std::vector<integral_model::AbelianGroupType>& G) {
    std::vector<Value> out;
    for (const auto& g : G) out.push_back(Value::from_string(g.to_string()));
    return Value::from_list(std::move(out));
}
// (slot-24 NB-RENDER R1) The SAME group as a structure, read from the AbelianGroupType directly (never parsed back from
// its text): {free_rank, torsion: [[p, k, mult], ...]} -- Z^free_rank (+) (+)_{(p, k)} (Z/p^k)^mult, primes ascending,
// exponents ascending within a prime (the order of AbelianGroupType::to_string()).  Zero group: free_rank 0, torsion [].
Value GroupInv(const integral_model::AbelianGroupType& g) {
    std::vector<Value> tors;
    for (const auto& [p, stored] : g.torsion) {
        std::vector<uint32_t> exps = stored;   // stored descending; listed ascending, as to_string() prints them
        std::sort(exps.begin(), exps.end());
        std::size_t i = 0;
        while (i < exps.size()) {
            std::size_t j = i;
            while (j < exps.size() && exps[j] == exps[i]) ++j;
            tors.push_back(Value::from_list({Value::from_int(p), Value::from_int(exps[i]),
                                             Value::from_int(static_cast<std::int64_t>(j - i))}));
            i = j;
        }
    }
    Record r;
    r.set("free_rank", Value::from_int(g.free_rank));
    r.set("torsion", Value::from_list(std::move(tors)));
    return Value::from_record(std::move(r));
}
Value GroupInvList(const std::vector<integral_model::AbelianGroupType>& G) {
    std::vector<Value> out;
    for (const auto& g : G) out.push_back(GroupInv(g));
    return Value::from_list(std::move(out));
}
Value ComplexRecord(uint32_t q_max, const std::vector<integral_model::AbelianGroupType>& G,
                    const std::vector<std::string>& derived_from = {}) {
    Record r;
    r.set("q_max", Value::from_int(q_max));
    r.set("truncated", Value::from_bool(true));
    r.set("q_max_source", Value::from_string("derived"));
    std::vector<uint32_t> deg;
    for (uint32_t q = 0; q <= q_max; ++q) deg.push_back(q);
    r.set("degrees", IntList(deg));
    r.set("groups", GroupList(G));
    // ([C] sec:recordschema items 5, 7 (f)) a derived complex names the complexes it is derived from
    if (!derived_from.empty()) {
        std::vector<Value> names;
        for (const std::string& s : derived_from) names.push_back(Value::from_string(s));
        r.set("derived_from", Value::from_list(std::move(names)));
    }
    return Value::from_record(std::move(r));
}

// (slot-24 NB-TOURFIX) char_Z(T_p) factored over Z: Yun + Zassenhaus (NumberField/IntPolyFactor.h), exact, and verified by
// multiplying back.  Per p: {p, status, expanded, factored, factors: [{coefficients (high -> low, String), degree,
// multiplicity, irreducible}], complete, verified, note}.  A coefficient beyond 2^63 is read exactly from the record's
// faithful field (CharpolyEntry::exact, FMTMP).  status "unavailable" (factors []) when char_Z is refused or a coefficient is
// undetermined ("?").  Presentation only: the record, its JSON, digest and golden points do not see it.
bool PlainInteger(const std::string& s) {
    std::size_t i = (!s.empty() && s[0] == '-') ? 1 : 0;
    if (i >= s.size()) return false;
    for (; i < s.size(); ++i)
        if (!std::isdigit(static_cast<unsigned char>(s[i]))) return false;
    return true;
}
// (FMTMP) the record's faithful value of a coefficient beyond 2^63 (CharpolyEntry::exact, canonical signed hex; [C]
// sec:recordschema item 7 (l)) -- the same field the JSON carries -- as an nf::BigInt, by Horner in base 16.
bool ExactBig(const std::string& hex, nf::BigInt& out) {
    if (!rec::ExactHexCanonical(hex)) return false;
    const bool neg = hex[0] == '-';
    nf::BigInt acc, sixteen(16);
    for (std::size_t j = neg ? 3 : 2; j < hex.size(); ++j) {
        const int64_t v = hex[j] <= '9' ? hex[j] - '0' : hex[j] - 'a' + 10;
        acc = acc * sixteen + nf::BigInt(v);
    }
    out = neg ? -acc : acc;
    return true;
}
// the coefficients as exact integers (constant term first), or false if one is undetermined / unreadable
bool ExactCoefficients(const rec::CharpolyEntry& c, std::vector<nf::BigInt>& f) {
    f.clear();
    for (std::size_t i = c.coefficients.size(); i-- > 0;) {
        const std::string& s = c.coefficients[i];
        nf::BigInt v;
        if (PlainInteger(s)) v = nf::BigInt::from_string(s);
        else if (i < c.exact.size() && !c.exact[i].empty()) {
            if (!ExactBig(c.exact[i], v)) return false;
        } else return false;
        f.push_back(v);
    }
    return true;
}
Value FactoredEntry(const rec::CharpolyEntry& c) {
    Record r;
    r.set("p", Value::from_int(c.p));
    std::vector<nf::BigInt> f;   // constant term first
    bool known = !c.refused && c.determined && !c.coefficients.empty() && ExactCoefficients(c, f);
    if (known) known = f.size() >= 2 && f.back() == nf::BigInt(1);
    if (!known) {
        r.set("status", Value::from_string("unavailable"));
        r.set("expanded", Value::from_string(""));
        r.set("factored", Value::from_string(""));
        r.set("factors", Value::from_list({}));
        r.set("complete", Value::from_bool(false));
        r.set("verified", Value::from_bool(false));
        r.set("note", Value::from_string(c.refused ? "char_Z refused" : (c.r == 0 ? "r = 0: the constant polynomial 1"
                                                                                  : "a coefficient is undetermined")));
        return Value::from_record(std::move(r));
    }
    const nf::ZFactorisation F = nf::factor_monic_over_z(f);
    std::vector<Value> fs;
    for (const nf::ZFactor& z : F.factors) {
        Record x;
        std::vector<Value> cs;
        for (auto it = z.poly.rbegin(); it != z.poly.rend(); ++it) cs.push_back(Value::from_string(it->to_string()));
        x.set("coefficients", Value::from_list(std::move(cs)));
        x.set("degree", Value::from_int(static_cast<std::int64_t>(z.poly.size()) - 1));
        x.set("multiplicity", Value::from_int(z.multiplicity));
        x.set("irreducible", Value::from_bool(z.irreducible));
        fs.push_back(Value::from_record(std::move(x)));
    }
    r.set("status", Value::from_string("ok"));
    r.set("expanded", Value::from_string(nf::zpoly_text(f)));
    r.set("factored", Value::from_string(F.text()));
    r.set("factors", Value::from_list(std::move(fs)));
    r.set("complete", Value::from_bool(F.complete));
    r.set("verified", Value::from_bool(F.verified));
    r.set("note", Value::from_string(F.complete ? "irreducible factors over Z (Zassenhaus), product verified"
                                                : "a factor marked irreducible = false is a product of irreducibles not "
                                                  "separated (recombination cap); product verified"));
    return Value::from_record(std::move(r));
}

Value VerbDescent(CallContext& ctx, const std::vector<Value>& a) {
    static const char* kWho = "descent(N, n, primes)";
    if (!hecke::CoefficientRingIsZ())
        throw RuntimeError(std::string(kWho) + ": the descent record is an object over Z; set coefficient_ring = \"Z\" first (in force: " +
                               hecke::math_settings().coefficient_ring.to_text() + ")",
                           ctx.pos);
    auto v = std::make_shared<DescentRecordVal>();
    // (NBCONFIRM S3) The cost bound below is the verb's SILENT ENVELOPE: inside it the verb is unchanged; outside it the
    // confirmation policy of Console/Core/Confirm.h decides (prompt-less modes refuse with the same text as before).  The
    // checks run in the pre-S3 order; N >= 1, p prime, p not dividing N and a non-empty list stay refusals (well-definedness).
    // descent is LAZY, so the confirmation sits here, in the constructor (S1 Sec. 4.2), not at the first member read.
    console::CostGate gate(ctx, kWho, "descent");
    gate.validate([&] {
        v->N = console::bind::U32Arg(a[0], ctx.pos, kWho);
        v->n = console::bind::U32Arg(a[1], ctx.pos, kWho);
        const std::string n_msg = std::string(kWho) + ": N must satisfy 1 <= N <= " + std::to_string(kMaxLevel);
        if (v->N < 1) gate.refuse(n_msg);
        if (v->N > kMaxLevel) gate.outside(n_msg);
        if (v->n > kMaxN) gate.outside(std::string(kWho) + ": n must satisfy 0 <= n <= " + std::to_string(kMaxN) + " (the cost bound of this verb)");
        if (rec::RecordModuleRank(v->N, v->n) > kMaxModuleRank)
            gate.outside(std::string(kWho) + ": index(N) * (n + 1) must be <= " + std::to_string(kMaxModuleRank) + " (the cost bound of this verb)");
        if (a.size() > 2 && !a[2].is_nil()) {
            if (!a[2].is_list()) gate.refuse(std::string(kWho) + ": `primes` is a List of primes");
            for (const Value& x : a[2].list_value()) {
                const uint32_t p = console::bind::U32Arg(x, ctx.pos, kWho);
                bool prime = p >= 2;
                for (uint64_t d = 2; prime && d * d <= p; ++d) prime = p % d != 0;   // (S3) 64-bit d: no wrap near 2^32
                const std::string p_msg = std::string(kWho) + ": `primes` must be primes <= 97";
                if (!prime) gate.refuse(p_msg);
                if (p > 97) gate.outside(p_msg);
                if (v->N % p == 0) gate.refuse(std::string(kWho) + ": `primes` must not divide N (U_p is deferred)");
                if (std::find(v->ps.begin(), v->ps.end(), p) == v->ps.end()) v->ps.push_back(p);
            }
            const std::string c_msg = std::string(kWho) + ": `primes` holds 1 .. 6 primes";
            if (v->ps.empty()) gate.refuse(c_msg);
            if (v->ps.size() > 6) gate.outside(c_msg);
            uint64_t sum = 0;
            for (uint32_t p : v->ps) sum += p;
            if (sum > kMaxPrimeSum)
                gate.outside(std::string(kWho) + ": the sum of `primes` must be <= " + std::to_string(kMaxPrimeSum) +
                             " (the cost bound of this verb: T_p costs about p)");
            std::sort(v->ps.begin(), v->ps.end());   // (D6 audit N3) the record's P is a set: ascending, as BuildDescentRecord hashes it
        } else {
            v->ps = rec::DefaultRecordPrimes(v->N);
        }
    });
    {
        std::vector<std::string> cost_args{std::to_string(v->N), std::to_string(v->n)};
        if (a.size() > 2 && !a[2].is_nil()) {
            std::string pl;
            for (uint32_t p : v->ps) pl += (pl.empty() ? "" : ":") + std::to_string(p);
            cost_args.push_back(pl);
        }
        const console::CostAdmission adm =
            gate.admit(cost_args, {"descent is lazy: the work runs at the first member read (r.digest, r.groups, ...)",
                                   "a part refused by its own limits comes back as status \"refused\" (a result, not an error)"});
        // (S2 brief item 6) a confirmed cell's pieces refuse at the console's memory limit, not at the 40 GiB default, so an
        // under-estimate ends as status "refused" instead of in the OOM killer.  Inside the envelope nothing changes.
        if (adm.outside && adm.decision.mem_limit_mib > 0) {
            const uint64_t lim = static_cast<uint64_t>(adm.decision.mem_limit_mib * 1024.0 * 1024.0);
            v->rss_cap_bytes = std::min<uint64_t>(v->rss_cap_bytes, lim);
        }
        if (adm.outside && adm.decision.level != console::cost::CostLevel::Silent) {
            v->confirmed = std::make_shared<const console::cost::CostEstimate>(adm.estimate);
            v->confirmed_call = ctx.call_text.empty() ? std::string(kWho) : ctx.call_text;
        }
    }
    ObjectPtr o = Object::create(console::bind::TypeByName(ctx, kTypeDescentRecord), ctx.pos, ctx.call_text,
                                 ctx.session->settings().snapshot(), v, console::bind::LazyCompute([v](Object&) { v->ensure(); }));
    return Value::from_object(o);
}

void RegisterDescentType(Registry& registry) {
    TypeInfo t;
    t.name = kTypeDescentRecord;
    t.doc = "the integral descent record integral_descent_record/v1 of H^q(Gamma_0(N); Sym^n Z^2) over Z ([C] thm:descentcert; "
            "slot-24 D6): the groups to q_max = 3 (object = group, truncated, periodic from q = 2), the bad primes, the certified "
            "precision K per prime, char_Z(T_p | H^1/tors), the torsion Hecke modules by their invariants, the boundary pair "
            "(sec:descentpair); only under coefficient_ring = Z; lazy";
    using M = TypeInfo::Member;
    t.members.push_back(M{"N", false, "the level", [](const ObjectPtr& o) { return Value::from_int(Val(o).N); }});
    t.members.push_back(M{"n", false, "the symmetric power", [](const ObjectPtr& o) { return Value::from_int(Val(o).n); }});
    t.members.push_back(M{"primes", false, "the Hecke primes P (none dividing N)", [](const ObjectPtr& o) { return IntList(Val(o).ps); }});
    t.members.push_back(M{"coefficient_ring", false, "Z (the record exists only under coefficient_ring = Z)",
                          [](const ObjectPtr&) { return Value::from_string("Z"); }});
    t.members.push_back(M{"object", false, "group (group cohomology, never Bredon; OD-10 (a))",
                          [](const ObjectPtr&) { return Value::from_string("group"); }});
    t.members.push_back(M{"q_max", false, "3, derived from thm:descentcert(6)", [](const ObjectPtr&) { return Value::from_int(rec::kRecordQMax); }});
    t.members.push_back(M{"truncated", false, "true: degrees above q_max are not reported (they are periodic, thm:descentcert(6))",
                          [](const ObjectPtr&) { return Value::from_bool(true); }});
    t.members.push_back(M{"q_max_source", false, "derived", [](const ObjectPtr&) { return Value::from_string("derived"); }});
    t.members.push_back(M{"degrees", false, "[0, 1, 2, 3]", [](const ObjectPtr&) { return IntList({0, 1, 2, 3}); }});
    t.members.push_back(M{"groups", true, "H^0 .. H^3 as abelian groups (\"Z^3 + Z/2 + Z/4\"); empty when refused",
                          [](const ObjectPtr& o) { return GroupList(Rec(o).H); }});
    t.members.push_back(M{"bad_primes", true, "B(M_N) of thm:descentcert(1): List of [ell, count] (prop:badprimesN(1)), ell >= 5",
                          [](const ObjectPtr& o) {
                              std::vector<Value> out;
                              for (const auto& [ell, c] : Rec(o).bad_primes) out.push_back(IntList({ell, c}));
                              return Value::from_list(std::move(out));
                          }});
    t.members.push_back(M{"torsion_primes", true, "the primes whose torsion is read: 2, 3 (n even) and B(M_N)",
                          [](const ObjectPtr& o) { return IntList(Rec(o).torsion_primes); }});
    t.members.push_back(M{"K", true, "per prime of the groups' run: {ell, route, K, W, certified} (certification provenance, not hashed)",
                          [](const ObjectPtr& o) {
                              std::vector<Value> out;
                              for (const rec::PrimeK& k : Rec(o).K) {
                                  Record r;
                                  r.set("ell", Value::from_int(k.ell));
                                  r.set("route", Value::from_string(k.route));
                                  r.set("K", Value::from_int(k.K));
                                  r.set("W", Value::from_int(k.W));
                                  r.set("certified", Value::from_bool(k.certified));
                                  out.push_back(Value::from_record(std::move(r)));
                              }
                              return Value::from_list(std::move(out));
                          }});
    t.members.push_back(M{"charpoly", true,
                          "per p: {p, r, coefficients (high -> low, String: an integer, \"[sign,D digits]\" beyond 2^63, \"?\" undetermined), "
                          "exact (high -> low, String: every coefficient as an exact integer, \"?\" undetermined), "
                          "determined, digest, refused} -- char_Z(T_p | H^1/tors), prop:crtcharpolyN's output rule",
                          [](const ObjectPtr& o) {
                              std::vector<Value> out;
                              for (const rec::CharpolyEntry& c : Rec(o).charpoly) {
                                  Record r;
                                  r.set("p", Value::from_int(c.p));
                                  r.set("r", Value::from_int(c.r));
                                  std::vector<Value> cs;
                                  for (const std::string& s : c.coefficients) cs.push_back(Value::from_string(s));
                                  r.set("coefficients", Value::from_list(std::move(cs)));
                                  // (FMTMP) every coefficient as an exact decimal integer (the record's faithful field beyond 2^63)
                                  std::vector<Value> ex;
                                  std::vector<nf::BigInt> f;
                                  if (ExactCoefficients(c, f))
                                      for (auto it = f.rbegin(); it != f.rend(); ++it) ex.push_back(Value::from_string(it->to_string()));
                                  else
                                      for (const std::string& s : c.coefficients) ex.push_back(Value::from_string(PlainInteger(s) ? s : "?"));
                                  r.set("exact", Value::from_list(std::move(ex)));
                                  r.set("determined", Value::from_bool(c.determined));
                                  r.set("digest", Value::from_string(HexU64(c.digest)));
                                  r.set("refused", Value::from_bool(c.refused));
                                  out.push_back(Value::from_record(std::move(r)));
                              }
                              return Value::from_list(std::move(out));
                          }});
    t.members.push_back(M{"charpoly_factored", true,
                          "per p: {p, status (ok | unavailable), expanded, factored (\"(X + 24)^2 (X - 2049)\"), factors: "
                          "[{coefficients (high -> low, String), degree, multiplicity, irreducible}], complete, verified, note} -- "
                          "char_Z(T_p | H^1/tors) factored over Z (Yun + Zassenhaus, exact; the product of the factors is "
                          "checked against char_Z); presentation only, not in the record's JSON or digest",
                          [](const ObjectPtr& o) {
                              DescentRecordVal& V = Val(o);
                              if (!V.factored) {
                                  std::vector<Value> out;
                                  for (const rec::CharpolyEntry& c : V.R->charpoly) out.push_back(FactoredEntry(c));
                                  V.factored = std::make_unique<Value>(Value::from_list(std::move(out)));
                              }
                              return *V.factored;
                          }});
    t.members.push_back(M{"torsion", true,
                          "per (q, ell): {q, ell, exponents, module_certified, ops: [{p, invariants}], refused} -- the torsion Hecke "
                          "module by its choice-free invariants (prop:torsionhecke(1))",
                          [](const ObjectPtr& o) {
                              std::vector<Value> out;
                              for (const rec::TorsionModule& m : Rec(o).torsion) {
                                  Record r;
                                  r.set("q", Value::from_int(m.q));
                                  r.set("ell", Value::from_int(m.ell));
                                  r.set("exponents", IntList(m.exponents));
                                  r.set("module_certified", Value::from_bool(m.module_certified));
                                  std::vector<Value> ops;
                                  for (const auto& [p, canon] : m.invariants) {
                                      Record x;
                                      x.set("p", Value::from_int(p));
                                      x.set("invariants", Value::from_string(canon));
                                      ops.push_back(Value::from_record(std::move(x)));
                                  }
                                  r.set("ops", Value::from_list(std::move(ops)));
                                  r.set("refused", Value::from_bool(m.refused));
                                  out.push_back(Value::from_record(std::move(r)));
                              }
                              return Value::from_list(std::move(out));
                          }});
    t.members.push_back(M{"pair", true,
                          "the boundary pair (sec:descentpair): {bulk, boundary, compact} each {q_max, truncated, q_max_source, degrees, "
                          "groups}, q = 0 .. 4 (compact, the cone, also derived_from = [bulk, boundary]); refused",
                          [](const ObjectPtr& o) {
                              const rec::PairPart& P = Rec(o).pair;
                              Record r;
                              r.set("bulk", ComplexRecord(P.q_max, P.H));
                              r.set("boundary", ComplexRecord(P.q_max, P.Hd));
                              r.set("compact", ComplexRecord(P.q_max, P.Hc, rec::kCompactDerivedFrom));
                              r.set("refused", Value::from_bool(P.refused));
                              return Value::from_record(std::move(r));
                          }});
    // (slot-24 NB-RENDER R1, owner decision O3 of NOTEBOOK_render_audit_1029.md) structural twins of `groups` and of
    // `pair.*.groups`; the String members stay unchanged.
    t.members.push_back(M{"groups_inv", true,
                          "H^0 .. H^3 as structures: per q {free_rank, torsion: [[p, k, mult], ...]} = Z^free_rank + "
                          "(+) (Z/p^k)^mult (the same groups as `groups`); empty when refused",
                          [](const ObjectPtr& o) { return GroupInvList(Rec(o).H); }});
    t.members.push_back(M{"pair_inv", true,
                          "the boundary pair's groups as structures: {bulk, boundary, compact}, each a List over q = 0 .. 4 of "
                          "{free_rank, torsion: [[p, k, mult], ...]} (the same groups as `pair`)",
                          [](const ObjectPtr& o) {
                              const rec::PairPart& P = Rec(o).pair;
                              Record r;
                              r.set("bulk", GroupInvList(P.H));
                              r.set("boundary", GroupInvList(P.Hd));
                              r.set("compact", GroupInvList(P.Hc));
                              return Value::from_record(std::move(r));
                          }});
    t.members.push_back(M{"status", true, "ok | refused", [](const ObjectPtr& o) { return Value::from_string(Rec(o).status); }});
    t.members.push_back(M{"refusal", true, "the first refused part's reason (\"\" when ok)",
                          [](const ObjectPtr& o) { return Value::from_string(Rec(o).refusal); }});
    t.members.push_back(M{"digest", true, "record_fnv1a64: FNV-1a-64 of the record's invariants and the settings' canonical text",
                          [](const ObjectPtr& o) { return Value::from_string(HexU64(Rec(o).digest())); }});
    t.members.push_back(M{"json", true, "the record as one JSON line (schema integral_descent_record/v1)",
                          [](const ObjectPtr& o) { return Value::from_string(Rec(o).to_json()); }});
    // (slot-25 EXPORT, docs/implementation/slot25/00-slot25-plan.md Sec. 2.1) H^1 (x) Z/ell^K with the T_p of `primes` as an object:
    // Integral/Descent/DescentModuleExport.h.  The handle's own bounds (N <= 300, n <= 60, index(N) (n + 1) <= 300, sum of primes <=
    // 100) were checked by the verb; the method adds ell (a prime < 2^31) and K >= 1.  It reads none of the record's computed parts
    // (computed = false: it does not force the record) and no setting.  A refused export is not an error: status "blocked(export)".
    {
        TypeInfo::Method m;
        m.name = "module";
        m.params = {"ell", "K"};
        m.required = 2;
        m.computed = false;
        m.doc = "H^1(Gamma_0(N); Sym^n Z^2) (x) Z/ell^K with T_p (p in primes) by cochain-level lifts (slot-25 EXPORT; [C] "
                "prop:torsionhecke, prop:strictliftN): {N, n, ell, K, K_max, route, carrier, status (ok | blocked(export) | refused), "
                "refusal, free_rank, torsion (exponents at K), invariant_factors, generators: [{kind, exponent, cocycle (a cocycle mod "
                "ell^K)}], ops: [{p, "
                "matrix (column convention, row i mod ell^exponent_i), charpoly_free (high -> low, mod ell^K)}]}; one run at the "
                "route's width serves every K <= K_max (computed on the first call per ell; it does not force the record); ell in {2, 3}: "
                "the tree model, ell >= 5: the cokernel presentation (K_max <= 1 for ell >= 46341)";
        m.call = [](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) -> Value {
            static const char* kWho = "DescentRecord.module(ell, K)";
            namespace mx = integral_descent::modexport;
            const uint32_t ell = console::bind::U32Arg(args[0], pos, kWho);
            const uint32_t K = console::bind::U32Arg(args[1], pos, kWho);
            bool prime = ell >= 2;
            for (uint32_t d = 2; prime && uint64_t(d) * d <= ell; ++d) prime = ell % d != 0;
            if (!prime || ell >= (1u << 31)) throw TypeError(std::string(kWho) + ": ell must be a prime < 2^31", pos);
            if (K < 1) throw TypeError(std::string(kWho) + ": K must be >= 1", pos);
            DescentRecordVal& V = Val(o);
            std::shared_ptr<mx::ModuleExport>& E = V.exports[ell];
            if (!E) E = std::make_shared<mx::ModuleExport>(mx::ExportModule(V.N, V.n, ell, V.ps, V.export_options()));
            Record r;
            r.set("N", Value::from_int(V.N));
            r.set("n", Value::from_int(V.n));
            r.set("ell", Value::from_int(ell));
            r.set("K", Value::from_int(K));
            r.set("K_max", Value::from_int(E->K_max));
            r.set("route", Value::from_string(E->route));
            r.set("carrier", Value::from_string(E->carrier));
            r.set("primes", IntList(V.ps));
            const bool served = E->ok && K <= E->K_max;
            r.set("status", Value::from_string(!E->ok ? E->status() : (served ? "ok" : "refused")));
            r.set("refusal", Value::from_string(!E->ok ? E->refusal
                                                      : served ? std::string()
                                                               : "REFUSED [Precision] K = " + std::to_string(K) + " > K_max = " +
                                                                     std::to_string(E->K_max) + " (the width of this export)"));
            if (!served) {
                r.set("free_rank", Value::from_int(0));
                r.set("torsion", IntList({}));
                r.set("invariant_factors", IntList({}));
                r.set("generators", Value::from_list({}));
                r.set("ops", Value::from_list({}));
                return Value::from_record(std::move(r));
            }
            const mx::ModuleAtK M = E->at(K);
            r.set("free_rank", Value::from_int(M.free_rank));
            r.set("torsion", IntList(std::vector<uint32_t>(M.exps.begin(), M.exps.begin() + M.torsion_count)));
            r.set("invariant_factors", IntList(M.invariant_factors()));
            uint64_t Q = 1;
            for (uint32_t i = 0; i < K; ++i) Q *= ell;
            std::vector<Value> gens;
            for (uint32_t j = 0; j < M.rank(); ++j) {
                Record g;
                g.set("kind", Value::from_string(j < M.torsion_count ? "torsion" : "free"));
                g.set("exponent", Value::from_int(M.exps[j]));
                std::vector<Value> cv;
                for (uint32_t i = 0; i < E->cocycles.r; ++i) cv.push_back(Value::from_int(static_cast<std::int64_t>(E->cocycles.at(i, j) % Q)));
                g.set("cocycle", Value::from_list(std::move(cv)));
                gens.push_back(Value::from_record(std::move(g)));
            }
            r.set("generators", Value::from_list(std::move(gens)));
            std::vector<Value> ops;
            for (size_t k = 0; k < M.ops.size(); ++k) {
                Record x;
                x.set("p", Value::from_int(M.ps[k]));
                std::vector<Value> rows;
                for (uint32_t i = 0; i < M.rank(); ++i) {
                    std::vector<Value> row;
                    for (uint32_t j = 0; j < M.rank(); ++j) row.push_back(Value::from_int(static_cast<std::int64_t>(M.ops[k][size_t(i) * M.rank() + j])));
                    rows.push_back(Value::from_list(std::move(row)));
                }
                x.set("matrix", Value::from_list(std::move(rows)));
                x.set("charpoly_free", IntList(M.free_charpoly(k)));
                ops.push_back(Value::from_record(std::move(x)));
            }
            r.set("ops", Value::from_list(std::move(ops)));
            return Value::from_record(std::move(r));
        };
        t.methods.push_back(std::move(m));
    }
    // (slot-25 EXPORT-Z, docs/implementation/slot25/EXPORTZ_design.md, EXPORTZ2_design.md Sec. 5) H^1 over Z as a presented
    // Z[T_P]-module.  module_Z("local") is STEP 1, byte for byte the value of slot-25 EXPORT-Z step 1: tau (the torsion block) and M
    // (free -> torsion, per ell against the export's local free generators) from the exports at K = sigma_max(ell) of every torsion
    // prime, shared with `module` (the same per-ell cache), free_block = "pending(step 2)".  module_Z() is STEP 2 (EXZ2-P3): the same
    // fields with F EXACT over Z (decimal integer strings) in the picked basis iso_dixon/v3 (certified det(Phi_cert^T B) = +-1), M in
    // that basis -- against the classes [b_j] themselves (fix round 1, audit F2: T_p b_j = sum_i F_ij b_i + sum_a M_aj g_a, M = M_local
    // C_ell - pi_B F + tau pi_B, pi_B = pi_ell(B) printed per torsion prime in `retractions`) --, the change of basis C_ell per torsion
    // prime and the basis' key (rule, q0, FNVs); computed
    // once per handle (its exports run in the global basis and are not shared with `module`).  computed = true: both force the record,
    // whose basis-free invariants (char_Z, the torsion canon) are printed beside the matrices.
    {
        TypeInfo::Method m;
        m.name = "module_Z";
        m.params = {"mode"};
        m.required = 0;
        m.computed = true;
        m.doc = "H^1(Gamma_0(N); Sym^n Z^2) over Z as a presented Z[T_p]-module ([C] prop:torsionhecke): T_p = [[tau, M], [0, F]] in "
                "generators torsion first, then free; module_Z() = EXPORT-Z step 2 (F exact over Z), module_Z(\"local\") = step 1 "
                "(slot-25 EXPORT-Z, F pending). {N, n, primes, status (ok | blocked(export)), refusal, "
                "splitting (export_retraction/v1), retractions: [{ell, route, K_run, K, fnv, [step 2: pi_B (t_ell x r, row a mod d_a: the "
                "retraction on B, the torsion coordinates of the classes [b_j]), pi_B_fnv]}], torsion_primes, largest_torsion_prime, "
                "invariant_factors (the elementary divisors d_i = ell^sigma_i, ell ascending; key name kept), torsion_generators: "
                "[{ell, exponent, d}], free_rank (a blocked value prints these four as \"blocked\"), free_block (step 2: \"exact\", or "
                "\"refused\" with free_refusal; local: \"pending(step 2)\"), M_basis (step 2: global, M in the basis B; local: M's ell-block "
                "is against module(ell, K)'s free generators), entry_encoding (step 2: exact, F as decimal integer strings; local: word), "
                "split (unknown), presentation, [step 2: basis: {rule (iso_dixon/v3), q0, saturation (det_pm1), basis_fnv, phi_fnv, bits_B, "
                "bits_Phi, bound (phi0 | phi_cert by the Auto size rule; - at rank 0)}, free_refusal], ops: [{p, tau (row i mod d_i, "
                "block-diagonal over ell), M: [{ell, K, matrix}] (step 2: against the classes [b_j] of B, T_p b_j = sum_i F_ij b_i + sum_a "
                "M_aj g_a, i.e. M = M_local C - pi_B F + tau pi_B; local: against module(ell, K)'s free generators), F (step 2: "
                "r x r decimal strings; local: \"pending(step 2)\"), F_local: [{ell, modulus, matrix}] (the export's free block mod ell^K, not "
                "F), charpoly_Z (the record's char_Z(T_p | H^1/tors), high -> low), [step 2: F_fnv, F_proof (apriori | apost | rank0), "
                "F_primes, M_fnv (FNV-1a-64 of M over every ell, span convention), C: [{ell, modulus, matrix}] (B in the export's local free "
                "generators: C F = F_local C mod ell^K)]}], "
                "torsion_canon: [{ell, ops: [{p, invariants}]}] (the record's)}; unique up to [[Aut T, Hom(L, T)], [0, GL_r(Z)]]";
        m.call = [](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) -> Value {
            namespace mx = integral_descent::modexport;
            static const char* kWho = "DescentRecord.module_Z(mode)";
            bool local = false;
            if (!args.empty()) {
                if (!args[0].is_string() || args[0].string_value() != "local")
                    throw TypeError(std::string(kWho) + ": mode must be \"local\" (step 1); module_Z() is step 2", pos);
                local = true;
            }
            DescentRecordVal& V = Val(o);
            const rec::DescentRecordZ& R = *V.R;
            mx::ModuleZ Zl;
            if (local) {
                Zl = mx::ExportModuleZ(V.N, V.n, V.ps, {}, [&V](uint32_t ell) -> std::shared_ptr<const mx::ModuleExport> {
                    std::shared_ptr<mx::ModuleExport>& E = V.exports[ell];
                    if (!E) E = std::make_shared<mx::ModuleExport>(mx::ExportModule(V.N, V.n, ell, V.ps, V.export_options()));
                    return E;
                });
            } else if (!V.module_z2) {
                mx::ModuleZ2Options z2;   // (S3 Fix round 1, audit m3) the lowered piece cap of a confirmed cell, else the defaults
                z2.step1.export_opt = V.export_options();
                z2.rss_cap_bytes = std::min<uint64_t>(z2.rss_cap_bytes, V.rss_cap_bytes);
                V.module_z2 = std::make_shared<const mx::ModuleZ>(mx::ExportModuleZ2(V.N, V.n, V.ps, z2));
            }
            const mx::ModuleZ& Z = local ? Zl : *V.module_z2;
            const bool step2 = !local;
            const bool exact = step2 && Z.ok && Z.free_block == "exact";
            Record r;
            r.set("N", Value::from_int(V.N));
            r.set("n", Value::from_int(V.n));
            r.set("primes", IntList(V.ps));
            r.set("status", Value::from_string(Z.status()));
            r.set("refusal", Value::from_string(Z.refusal));
            r.set("splitting", Value::from_string(Z.splitting));
            std::vector<Value> retr;
            for (const mx::LocalPartZ& P : Z.local) {
                Record x;
                x.set("ell", Value::from_int(P.ell));
                x.set("route", Value::from_string(P.E->route));
                x.set("K_run", Value::from_int(P.E->K_run));
                x.set("K", Value::from_int(P.K));
                x.set("fnv", Value::from_string(HexU64(P.retraction_fnv)));
                if (exact) {   // (fix round 1, audit F2) pi_ell(B): the torsion coordinates of the classes [b_j] (the change of complement)
                    std::vector<Value> rows;
                    for (uint32_t i = 0; i < P.t; ++i) {
                        std::vector<Value> row;
                        for (uint32_t j = 0; j < Z.free_rank; ++j)
                            row.push_back(Value::from_int(static_cast<std::int64_t>(P.pi_B[size_t(i) * Z.free_rank + j])));
                        rows.push_back(Value::from_list(std::move(row)));
                    }
                    x.set("pi_B", Value::from_list(std::move(rows)));
                    x.set("pi_B_fnv", Value::from_string(P.t > 0 ? HexU64(P.pi_B_fnv) : std::string()));
                }
                retr.push_back(Value::from_record(std::move(x)));
            }
            r.set("retractions", Value::from_list(std::move(retr)));
            r.set("torsion_primes", IntList(Z.torsion_primes));
            // audit F1: a blocked value has no torsion data; it must not read as T = 0 or as an earlier export's rank
            const Value blocked = Value::from_string("blocked");
            r.set("largest_torsion_prime", Z.ok ? Value::from_int(Z.largest_torsion_prime()) : blocked);
            std::vector<Value> dv, tg;
            for (uint32_t i = 0; i < Z.torsion_rank(); ++i) {
                dv.push_back(Value::from_int(static_cast<std::int64_t>(Z.d[i])));
                Record g;
                g.set("ell", Value::from_int(Z.tor_ell[i]));
                g.set("exponent", Value::from_int(Z.tor_sigma[i]));
                g.set("d", Value::from_int(static_cast<std::int64_t>(Z.d[i])));
                tg.push_back(Value::from_record(std::move(g)));
            }
            r.set("invariant_factors", Z.ok ? Value::from_list(std::move(dv)) : blocked);
            r.set("torsion_generators", Z.ok ? Value::from_list(std::move(tg)) : blocked);
            r.set("free_rank", Z.ok ? Value::from_int(Z.free_rank) : blocked);
            r.set("free_block", Value::from_string(Z.free_block));
            r.set("M_basis", Value::from_string(Z.M_basis));
            r.set("entry_encoding", Value::from_string(Z.entry_encoding));
            r.set("split", Value::from_string(Z.split));
            r.set("presentation", Value::from_string("unique up to the block-triangular group [[Aut T, Hom(L, T)], [0, GL_r(Z)]]; "
                                                     "basis-free: invariant_factors, free_rank, charpoly_Z, torsion_canon"));
            if (step2) {
                const mx::FreeBlockZ& G = Z.global;
                Record b;
                b.set("rule", Value::from_string(G.basis_rule));
                b.set("q0", Value::from_string(G.q0_key));
                b.set("saturation", Value::from_string(G.saturation));
                b.set("basis_fnv", Value::from_string(exact ? HexU64(G.basis_fnv) : std::string()));
                b.set("phi_fnv", Value::from_string(exact ? HexU64(G.phi_fnv) : std::string()));
                b.set("bits_B", Value::from_int(G.bits_B));
                b.set("bits_Phi", Value::from_int(G.bits_Phi));
                b.set("bound", Value::from_string(G.bound_basis));
                r.set("basis", Value::from_record(std::move(b)));
                r.set("free_refusal", Value::from_string(Z.free_refusal));
            }
            auto Rows = [](const std::vector<uint64_t>& A, uint32_t rows, uint32_t cols) {
                std::vector<Value> out;
                for (uint32_t i = 0; i < rows; ++i) {
                    std::vector<Value> row;
                    for (uint32_t j = 0; j < cols; ++j) row.push_back(Value::from_int(static_cast<std::int64_t>(A[size_t(i) * cols + j])));
                    out.push_back(Value::from_list(std::move(row)));
                }
                return Value::from_list(std::move(out));
            };
            std::vector<Value> ops;
            for (size_t k = 0; Z.ok && k < Z.ps.size(); ++k) {
                Record x;
                x.set("p", Value::from_int(Z.ps[k]));
                x.set("tau", Rows(Z.tau[k], Z.torsion_rank(), Z.torsion_rank()));
                std::vector<Value> Ms, Fs, Cs;
                for (size_t l = 0; l < Z.local.size(); ++l) {
                    const mx::LocalPartZ& P = Z.local[l];
                    if (P.t == 0) continue;
                    const mx::FreeView fv = Z.free_view(k, l);   // M in the object's basis (global in step 2)
                    Record mb;
                    mb.set("ell", Value::from_int(P.ell));
                    mb.set("K", Value::from_int(P.K));
                    mb.set("matrix", Rows(fv.M_block, P.t, Z.free_rank));
                    Ms.push_back(Value::from_record(std::move(mb)));
                    Record fb;
                    fb.set("ell", Value::from_int(P.ell));
                    uint64_t Q = 1;
                    for (uint32_t i = 0; i < P.K; ++i) Q *= P.ell;
                    fb.set("modulus", Value::from_int(static_cast<std::int64_t>(Q)));
                    fb.set("matrix", Rows(Z.F_local(k, l), Z.free_rank, Z.free_rank));
                    Fs.push_back(Value::from_record(std::move(fb)));
                    if (exact) {
                        std::vector<uint64_t> Cq(P.C.size());
                        for (size_t e = 0; e < Cq.size(); ++e) Cq[e] = P.C[e] % Q;
                        Record cb;
                        cb.set("ell", Value::from_int(P.ell));
                        cb.set("modulus", Value::from_int(static_cast<std::int64_t>(Q)));
                        cb.set("matrix", Rows(Cq, Z.free_rank, Z.free_rank));
                        Cs.push_back(Value::from_record(std::move(cb)));
                    }
                }
                x.set("M", Value::from_list(std::move(Ms)));
                if (exact) {
                    std::vector<Value> rows;
                    for (uint32_t i = 0; i < Z.free_rank; ++i) {
                        std::vector<Value> row;
                        for (uint32_t j = 0; j < Z.free_rank; ++j) row.push_back(Value::from_string(Z.global.F[k][size_t(i) * Z.free_rank + j].to_string()));
                        rows.push_back(Value::from_list(std::move(row)));
                    }
                    x.set("F", Value::from_list(std::move(rows)));
                } else {
                    x.set("F", Value::from_string(Z.free_block));
                }
                x.set("F_local", Value::from_list(std::move(Fs)));
                std::vector<Value> cz;
                for (const rec::CharpolyEntry& c : R.charpoly) {
                    if (c.p != Z.ps[k]) continue;
                    std::vector<nf::BigInt> f;
                    if (ExactCoefficients(c, f))
                        for (auto it = f.rbegin(); it != f.rend(); ++it) cz.push_back(Value::from_string(it->to_string()));
                    else
                        for (auto it = c.coefficients.rbegin(); it != c.coefficients.rend(); ++it)
                            cz.push_back(Value::from_string(PlainInteger(*it) ? *it : "?"));
                }
                x.set("charpoly_Z", Value::from_list(std::move(cz)));
                if (step2) {
                    x.set("F_fnv", Value::from_string(exact ? HexU64(Z.global.F_fnv[k]) : std::string()));
                    x.set("M_fnv", Value::from_string(exact ? HexU64(Z.M_fnv[k]) : std::string()));
                    x.set("F_proof", Value::from_string(exact ? Z.global.proof[k] : std::string()));
                    x.set("F_primes", Value::from_int(exact ? Z.global.primes_used[k] : 0));
                    x.set("C", Value::from_list(std::move(Cs)));
                }
                ops.push_back(Value::from_record(std::move(x)));
            }
            r.set("ops", Value::from_list(std::move(ops)));
            std::vector<Value> canon;
            for (const rec::TorsionModule& tm : R.torsion) {
                if (tm.q != 1) continue;
                Record x;
                x.set("ell", Value::from_int(tm.ell));
                std::vector<Value> co;
                for (const auto& [p, s] : tm.invariants) {
                    Record y;
                    y.set("p", Value::from_int(p));
                    y.set("invariants", Value::from_string(s));
                    co.push_back(Value::from_record(std::move(y)));
                }
                x.set("ops", Value::from_list(std::move(co)));
                canon.push_back(Value::from_record(std::move(x)));
            }
            r.set("torsion_canon", Value::from_list(std::move(canon)));
            return Value::from_record(std::move(r));
        };
        t.methods.push_back(std::move(m));
    }
    t.print = [](const ObjectPtr& o, std::int64_t) {
        std::ostringstream os;
        os << "DescentRecord (N = " << Val(o).N << ", n = " << Val(o).n << ")";
        if (!o->computed()) os << " (declared, not yet computed)";
        else {
            const rec::DescentRecordZ& R = Rec(o);
            os << " " << R.status << ": ";
            for (size_t q = 0; q < R.H.size(); ++q) os << (q ? ", " : "") << "H^" << q << " = " << R.H[q].to_string();
        }
        return os.str();
    };
    registry.register_type(std::move(t));
}

}  // namespace

void RegisterDescentVerbs(Registry& registry) {
    RegisterDescentType(registry);
    registry.register_verb("descent",
                           console::bind::Sig({Param::of("N", ValueKind::Int), Param::of("n", ValueKind::Int),
                                               Param::of("primes", ValueKind::List).optional(Value::nil())}),
                           &VerbDescent,
                           "the integral descent record of H^q(Gamma_0(N); Sym^n Z^2) over Z ([C] thm:descentcert, sec:descentpair; "
                           "slot-24 D6), a lazy DescentRecord: groups to q_max = 3, bad primes, K per prime, char_Z(T_p), the torsion "
                           "Hecke modules, the boundary pair; only under coefficient_ring = \"Z\"; primes = List of 1 .. 6 primes <= 97 "
                           "not dividing N, sum <= 100 (nil = the first three of 2, 3, 5, 7, 11, 13 not dividing N); N <= 300, "
                           "n <= 60, index(N) * (n + 1) <= 300 (at the bound about 40 s and 1.4 GiB)",
                           "extension");
}

}  // namespace integral_descent
