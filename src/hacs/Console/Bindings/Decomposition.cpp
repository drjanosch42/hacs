// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Decomposition.cpp -- console phase C1b: the DECOMPOSITION SURFACE
// (plan docs/console/00-console-interface-plan.md v3.1 Sec. 10 "Phase C1b",
// Sec. 3.6 rows Decomposition / Block / Report / CrossChecks).
//
// C1a deliberately registered no `decompose` and no `report` verb: those are
// exactly the objects the extension stage's single golden re-baseline moved
// (slot 2 WP-2, phase E0b -> `hecke_report/v2`), and a transcript frozen
// against the old schema would have been re-baselined indistinguishably from a
// bug.  E0b has merged, so C1b binds them -- against the POST-E0b reference,
// and pinned to it only through `golden(...)`, never through a copied number.
//
// WHAT IS BOUND
//   decompose(ops, q)      HeckeModuleDecomposition over the family `ops`
//                          (one HeckeOperator handle, or a list of them), in
//                          cohomological degree q.  q is MANDATORY (gap G12).
//   report(D [, level_a, path])
//                          the hecke_report/v2 JSON; `ell`, `n_sym`, `degree`
//                          and `num_degrees` come from the kept handles, so a
//                          console report cannot disagree with its source.
//   crosschecks(T)         HeckeCrossChecks::run_all + the three individual
//                          codes, as a Record.
//
// THE THREE HASHES A PIN NEEDS.  Golden/GoldenHarness.cpp's
// CollectDecomposition writes four entries per decomposition:
//   <name>_invariants   raw(blocks.size(), 0, h)   -- TypeInfo::hash here
//   <name>_certify      scalar(D.certify())        -- `.certify()` here
//   <name>_U            matrix(D.change_of_basis())-- hash(D.U) here
//   <name>_witness      raw(blocks.size(), 0, hw)  -- `.witness_hash` here
// The two raw hashes are reproduced BYTE FOR BYTE below (the recipes are
// transcribed, not re-invented), which is what makes
// `hash(D) == golden(point, "decomp_A(T_p)^1_invariants").hash` meaningful.
//
// NOT C1b's (recorded, [P] Sec. 3.5 / Sec. 10): `solve` for under-determined
// systems and the general pi-primary `jordan_type` are C2; the local-algebra
// invariants are C5; `bernoulli_mod` is E6's.
//
// THE MATHEMATICAL SETTINGS are registered here too, through the registry's
// settings-installer seam (Registry.h): `set locality = strict` is what the
// C1b strict-mode script needs, and R1 says the ENGINE owns the defaults and
// the validation -- so every key comes from hecke::MathSettings::keys() and
// every value goes through hecke::MathSettings::set().  The console
// re-implements neither.
// ===========================================================================

#include "Bindings.h"
#include "Payloads.h"

#include "../Core/Printer.h"
#include "../Core/Settings.h"

#include "../../Hecke/HeckeCrossChecks.h"
#include "../../Hecke/HeckeModuleDecomposition.h"
#include "../../Hecke/HeckeReport.h"
#include "../../Hecke/HeckeSettings.h"
#include "../../../ModArithmetic/Fnv.h"
#include "../../Util/TextIo.h"

#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace console {
namespace bind {

namespace {

// Set once by `--serve` ([P] v3.10.2 Sec. 18.14): the pathless `report` is served,
// so `Report.write(path)` must stay unreachable there.
bool g_report_write_forbidden = false;

using Flat = FlatMatrixT<uint32_t>;

const TypeInfo* g_matrix_type = nullptr;
const TypeInfo* g_poly_type = nullptr;
const TypeInfo* g_block_type = nullptr;

// --- payloads --------------------------------------------------------------

// The decomposition handle.  It keeps its source HeckeOperator handles alive
// (plan Sec. 4.1) AND the data `report` needs -- ell, n_sym, the degree and
// the complex's own degree COUNT, so that the record's four per-complex fields
// are derived from the complex and never from a literal.
struct DecompositionVal {
    std::vector<ObjectPtr> sources;                 // HeckeOperator handles
    std::vector<std::pair<uint64_t, MatrixT<uint32_t>>> ops;
    uint32_t ell = 0;
    uint32_t n_sym = 0;
    uint32_t q = 0;
    uint32_t num_degrees = 0;
    std::unique_ptr<hecke::HeckeModuleDecomposition> D;

    // (C2a, slot-6 WP-13) A RESUMED handle: its decomposition is a COPY of the
    // parent's, continued by resume_locality(resume_budget) -- never a fresh
    // run, which is what makes budget_history [b1, b2] rather than [b2]
    // ([X] Sec. 1.3.1 R3.1 items 2-4).
    ObjectPtr resume_parent;
    uint32_t resume_budget = 0;

    void ensure();
};

DecompositionVal& Dec(const ObjectPtr& o) { return *o->payload_as<DecompositionVal>(); }

void DecompositionVal::ensure() {
    if (D) return;
    if (resume_parent) {
        DecompositionVal& pv = Dec(resume_parent);
        pv.ensure();
        std::unique_ptr<hecke::HeckeModuleDecomposition> next(
            new hecke::HeckeModuleDecomposition(*pv.D));
        next->resume_locality(resume_budget);
        D = std::move(next);
        return;
    }
    D.reset(new hecke::HeckeModuleDecomposition(ops, ell));
    D->compute();
}

// A block is a VIEW on its parent's block list: it owns nothing and keeps the
// parent handle alive, which is the only edge the plan's ownership graph needs
// (PrimaryBlock lives inside HeckeModuleDecomposition's vector).
struct BlockVal {
    ObjectPtr parent;
    std::size_t index = 0;
};

struct ReportVal {
    ObjectPtr D;                                    // the Decomposition handle
    bool level_a = false;
    std::string path;                               // "" = not written to a file
    std::string json;
    void ensure();
};

const hecke::PrimaryBlock& BlockOf(const ObjectPtr& o) {
    const BlockVal* bv = o->payload_as<BlockVal>();
    CONSOLE_CHECK(bv != nullptr && bv->parent != nullptr);
    DecompositionVal& dv = Dec(bv->parent);
    dv.ensure();
    CONSOLE_CHECK(bv->index < dv.D->blocks().size());
    return dv.D->blocks()[bv->index];
}

Value MatrixValueFrom(const ObjectPtr& owner, Flat m) {
    auto p = std::make_shared<MatrixVal>();
    p->m = std::move(m);
    return Value::from_object(MakeValueObjectFrom(g_matrix_type, owner, std::move(p)));
}

Value PolyValueFrom(const ObjectPtr& owner, hecke::FpPolyT<uint32_t> f) {
    auto p = std::make_shared<PolyVal>();
    p->f = std::move(f);
    return Value::from_object(MakeValueObjectFrom(g_poly_type, owner, std::move(p)));
}

// --- the two raw golden hashes, transcribed from CollectDecomposition -------
//
// GoldenHarness.cpp:636-657.  HashPolyInto is the harness's polynomial recipe,
// which Bindings.cpp's HashPoly already reproduces; it is spelled out again
// here in the INTO form because the decomposition hash chains it.

uint64_t HashPolyInto(uint64_t h, const hecke::FpPolyT<uint32_t>& f) {
    h = engine::FnvU64(h, f.coeffs.size());
    for (uint32_t c : f.coeffs) h = engine::FnvU64(h, c);
    return h;
}

uint64_t DecompositionInvariantsHash(const hecke::HeckeModuleDecomposition& D) {
    uint64_t h = engine::FnvU64(engine::kFnvOffset64, D.blocks().size());
    for (const hecke::PrimaryBlock& b : D.blocks()) {
        h = engine::FnvU64(h, b.dim);
        h = engine::FnvU64(h, b.residue_degree);
        h = engine::FnvU64(h, b.char_poly.size());
        for (const auto& kv : b.char_poly) {
            h = engine::FnvU64(h, kv.first);
            h = HashPolyInto(h, kv.second);
        }
        for (const auto& kv : b.min_poly) {
            h = engine::FnvU64(h, kv.first);
            h = HashPolyInto(h, kv.second);
        }
        for (const auto& kv : b.jordan_partition) {
            h = engine::FnvU64(h, kv.first);
            h = engine::FnvU64(h, kv.second.size());
            for (uint32_t part : kv.second) h = engine::FnvU64(h, part);
        }
    }
    return h;
}

uint64_t DecompositionWitnessHash(const hecke::HeckeModuleDecomposition& D) {
    uint64_t hw = engine::FnvU64(engine::kFnvOffset64, D.blocks().size());
    for (const hecke::PrimaryBlock& b : D.blocks()) {
        hw = engine::FnvU64(hw, b.residue_degree_certified ? 1 : 0);
        hw = engine::FnvU64(hw, b.residue_witness.size());
        for (const auto& w : b.residue_witness) {
            hw = engine::FnvU64(hw, w.first);
            hw = engine::FnvU64(hw, w.second);
        }
    }
    return hw;
}

using oc_util::HexU64;   // S24 CL-7 (F11): Util/TextIo.h
// --- report ----------------------------------------------------------------

void ReportVal::ensure() {
    if (!json.empty()) return;
    CONSOLE_CHECK(D != nullptr);
    DecompositionVal& dv = Dec(D);
    dv.ensure();

    hecke::HeckeReportInput in;
    in.ell = dv.ell;
    in.n_sym = dv.n_sym;
    in.degree = dv.q;
    // The four per-complex record fields are DERIVED from the complex's own
    // degree count ([C] sec:recordschema item 5, [P] Sec. 3.6): the console
    // passes `num_degrees` and writes no literal `2`.
    in.num_degrees = dv.num_degrees;
    in.decomp = dv.D.get();
    for (const auto& pa : dv.ops) in.level_b[pa.first] = pa.second;
    // Level (a) is the (N_q d_V)^2 cochain matrix.  It is opt-in for exactly
    // the reason HeckeReport.h gives: the caller owns the size gate.
    std::vector<Flat> keep;
    if (level_a) {
        for (const ObjectPtr& src : dv.sources) {
            HeckeVal* hv = src->payload_as<HeckeVal>();
            hv->ensure();
            const auto& Tf = hv->op->get_Theta_flat_matrices();
            if (dv.q < Tf.size()) in.level_a_theta_flat[hv->p] = Tf[dv.q];
        }
    }
    json = hecke::BuildHeckeReportJson(in);
    if (!path.empty()) hecke::WriteHeckeReport(json, path);
}

ReportVal& Rep(const ObjectPtr& o) { return *o->payload_as<ReportVal>(); }

// --- verbs -----------------------------------------------------------------

// The decomposition handle's compute step.  R3 instance 2: the locality loop
// is a budgeted computation, so an incomplete state is REPORTED, never hidden.
void FillLocalityResumable(Object& self, DecompositionVal& v) {
    v.ensure();
    const hecke::LocalityState& ls = v.D->locality_state();
    ResumableState rs;
    rs.budget_setting = "locality";
    rs.budget_value = Value::from_int(ls.budget);
    rs.resumable = !ls.complete;
    if (!ls.complete) {
        rs.reason = "the locality budget was exhausted before every block was certified";
    }
    Record pr;
    pr.set("kind", Value::from_string("locality"));
    pr.set("complete", Value::from_bool(ls.complete));
    pr.set("splits", Value::from_int(static_cast<std::int64_t>(ls.splits)));
    pr.set("interrupted", Value::from_bool(ls.interrupted));
    pr.set("phase_reached", Value::from_string(hecke::to_string(ls.phase_reached)));
    std::vector<Value> hist;
    for (uint32_t b : ls.budget_history) hist.push_back(Value::from_int(b));
    pr.set("budget_history", Value::from_list(std::move(hist)));
    rs.progress = pr;
    self.set_resumable_state(rs);
}

ObjectPtr MakeDecompositionHandle(const TypeInfo* type, const SourcePos& pos,
                                  const std::string& expr, const SettingsSnapshot& snap,
                                  std::shared_ptr<DecompositionVal> v) {
    return Object::create(type, pos, expr, snap, v,
                          LazyCompute([v](Object& self) { FillLocalityResumable(self, *v); }));
}

// `ops` is either one HeckeOperator handle or a list of them.
std::vector<ObjectPtr> OperatorList(const Value& v, const SourcePos& pos, const char* who) {
    std::vector<ObjectPtr> out;
    if (v.is_object()) {
        out.push_back(AsObjectOf(v, kTypeHecke, pos, who));
        return out;
    }
    if (!v.is_list()) {
        throw TypeError(std::string(who) + ": expected a HeckeOperator or a list of them, got " +
                            v.type_name(),
                        pos);
    }
    const std::vector<Value>& items = v.list_value();
    if (items.empty()) {
        throw TypeError(std::string(who) + ": the operator family is empty", pos);
    }
    for (const Value& item : items) out.push_back(AsObjectOf(item, kTypeHecke, pos, who));
    return out;
}

Value VerbDecompose(CallContext& ctx, const std::vector<Value>& a) {
    static const char* kWho = "decompose(ops, q)";
    std::vector<ObjectPtr> sources = OperatorList(a[0], ctx.pos, kWho);
    const uint32_t q = U32Arg(a[1], ctx.pos, kWho);
    // (C5a, slot-9 WP-12) [E] sec:ring (h): `decompose` keeps exit 5 under a
    // non-default ring -- refused AT DISPATCH, for the ring in force and for a
    // Z/ell^k operator handed in under F_ell alike.  No-ops at the default.
    hecke::RequireFieldCoefficientRing("console pipeline verb decompose");
    for (const ObjectPtr& src : sources) RequireFieldHandle(src, "console pipeline verb decompose");

    auto v = std::make_shared<DecompositionVal>();
    v->sources = sources;
    v->q = q;

    // Everything below forces the source operators, which is what the user
    // asked for: `decompose` has no meaningful declared state of its own.
    for (const ObjectPtr& src : sources) {
        HeckeVal* hv = src->payload_as<HeckeVal>();
        hv->ensure();
        const auto& A = hv->op->get_H_matrices();
        if (q >= A.size()) {
            throw RuntimeError(std::string(kWho) + ": degree " + std::to_string(q) +
                                   " is not a degree of this complex (it has " +
                                   std::to_string(A.size()) + ")",
                               ctx.pos);
        }
        CohomologyVal* cv = hv->H->payload_as<CohomologyVal>();
        cv->ensure();
        const uint32_t ell = cv->coh->get_prime();
        const uint32_t n_sym = cv->coh->get_representation_dim() - 1;
        if (v->ops.empty()) {
            v->ell = ell;
            v->n_sym = n_sym;
            v->num_degrees = static_cast<uint32_t>(A.size());
        } else if (v->ell != ell || v->n_sym != n_sym) {
            throw TypeError(std::string(kWho) +
                                ": every operator must live on the SAME cohomology (got "
                                "(n, ell) = (" +
                                std::to_string(n_sym) + ", " + std::to_string(ell) + ") after (" +
                                std::to_string(v->n_sym) + ", " + std::to_string(v->ell) + "))",
                            ctx.pos);
        }
        for (const auto& already : v->ops) {
            if (already.first == hv->p) {
                throw TypeError(std::string(kWho) + ": two operators with the same prime p = " +
                                    std::to_string(hv->p) +
                                    " (the decomposition keys its family by p)",
                                ctx.pos);
            }
        }
        v->ops.emplace_back(hv->p, A[q]);
    }

    ObjectPtr obj = MakeDecompositionHandle(TypeByName(ctx, kTypeDecomposition), ctx.pos,
                                            ctx.call_text, ctx.session->settings().snapshot(), v);
    for (const ObjectPtr& src : sources) obj->keep_alive(src);
    return Value::from_object(obj);
}

Value VerbReport(CallContext& ctx, const std::vector<Value>& a) {
    static const char* kWho = "report(D, level_a, path)";
    const ObjectPtr& D = AsObjectOf(a[0], kTypeDecomposition, ctx.pos, kWho);
    hecke::RequireFieldCoefficientRing("console pipeline verb report");   // (C5a) [E] sec:ring (h), exit 5
    auto v = std::make_shared<ReportVal>();
    v->D = D;
    if (a.size() > 1 && !a[1].is_nil()) {
        if (!a[1].is_bool()) throw TypeError(std::string(kWho) + ": level_a must be a Bool", ctx.pos);
        v->level_a = a[1].bool_value();
    }
    if (a.size() > 2 && !a[2].is_nil()) {
        if (!a[2].is_string())
            throw TypeError(std::string(kWho) + ": path must be a String", ctx.pos);
        v->path = a[2].string_value();
    }
    ObjectPtr obj = Object::create(TypeByName(ctx, kTypeReport), ctx.pos, ctx.call_text,
                                   ctx.session->settings().snapshot(), v,
                                   LazyCompute([v](Object&) { v->ensure(); }));
    obj->keep_alive(D);
    return Value::from_object(obj);
}

Value VerbCrossChecks(CallContext& ctx, const std::vector<Value>& a) {
    static const char* kWho = "crosschecks(T)";
    const ObjectPtr& T = AsObjectOf(a[0], kTypeHecke, ctx.pos, kWho);
    // (C5a, slot-9 WP-12) [E] sec:ring (h): keeps exit 5 under a ring, at dispatch.
    hecke::RequireFieldCoefficientRing("console pipeline verb crosschecks");
    RequireFieldHandle(T, "console pipeline verb crosschecks");
    HeckeVal* hv = T->payload_as<HeckeVal>();
    hv->ensure();
    CohomologyVal* cv = hv->H->payload_as<CohomologyVal>();
    cv->ensure();
    hecke::HeckeCrossChecks xc(cv->coh.get(), hv->cosets.get(), hv->tau.get());
    Record r;
    const uint32_t i = xc.check_beta_equivariance();
    const uint32_t ii = xc.check_beta_alpha_equals_Xi();
    const uint32_t iii = xc.check_triple_factorization(*hv->op);
    r.set("beta_equivariance", Value::from_int(i));
    r.set("beta_alpha_equals_Xi", Value::from_int(ii));
    r.set("triple_factorization", Value::from_int(iii));
    r.set("all", Value::from_int(static_cast<std::int64_t>(i | ii | iii)));
    r.set("ok", Value::from_bool(i == 0 && ii == 0 && iii == 0));
    return Value::from_record(std::move(r));
}

// --- types -----------------------------------------------------------------

Value PolyMapMember(const ObjectPtr& o, const std::map<uint64_t, hecke::FpPolyT<uint32_t>>& m,
                    const std::vector<Value>& args, const SourcePos& pos, const char* who) {
    const uint64_t p = U64Arg(args[0], pos, who);
    const auto it = m.find(p);
    if (it == m.end()) {
        throw RuntimeError(std::string(who) + ": p = " + std::to_string(p) +
                               " is not in this block's operator family",
                           pos);
    }
    return PolyValueFrom(o, it->second);
}

void RegisterTypes(Registry& registry) {
    g_matrix_type = registry.find_type(kTypeMatrix);
    g_poly_type = registry.find_type(kTypePoly);
    CONSOLE_CHECK_MSG(g_matrix_type != nullptr && g_poly_type != nullptr,
                      "RegisterCoreMathBindings must run before RegisterDecompositionBindings");

    // --- Block -------------------------------------------------------------
    {
        TypeInfo t;
        t.name = kTypeBlock;
        t.doc = "one primary block of a decomposition: its dimension, residue degree, the "
                "per-prime minimal / characteristic polynomials and generalized-Jordan "
                "partition, and the E0b locality certificate";
        t.members.push_back({"dim", true, "dim_F of the block", [](const ObjectPtr& o) {
                                 return Value::from_int(BlockOf(o).dim);
                             }});
        t.members.push_back({"residue_degree", true, "f = lcm_j deg(pi_j)",
                             [](const ObjectPtr& o) {
                                 return Value::from_int(BlockOf(o).residue_degree);
                             }});
        t.members.push_back({"certified", true,
                             "whether f is CERTIFIED (a witness proves it), not merely a lower "
                             "bound",
                             [](const ObjectPtr& o) {
                                 return Value::from_bool(BlockOf(o).residue_degree_certified);
                             }});
        t.members.push_back({"local", true,
                             "the E0b locality certificate: this block is ONE maximal ideal",
                             [](const ObjectPtr& o) {
                                 return Value::from_bool(BlockOf(o).local_certified);
                             }});
        t.members.push_back({"phase", true,
                             "which phase reached the verdict (linear / berlekamp / "
                             "residue_search)",
                             [](const ObjectPtr& o) {
                                 return Value::from_string(
                                     hecke::to_string(BlockOf(o).phase_reached));
                             }});
        t.members.push_back({"w_hat_exponent", true, "N: the Teichmueller exponent of the witness",
                             [](const ObjectPtr& o) {
                                 return Value::from_int(BlockOf(o).w_hat_exponent);
                             }});
        t.members.push_back({"witness_poly", true, "pi_w, the certifying witness's minimal polynomial",
                             [](const ObjectPtr& o) {
                                 return PolyValueFrom(o, BlockOf(o).witness_poly);
                             }});
        t.members.push_back({"basis", true, "the d x dim matrix whose columns span the block",
                             [](const ObjectPtr& o) {
                                 return MatrixValueFrom(o, Flat(BlockOf(o).basis_columns));
                             }});
        t.members.push_back({"primes", true, "the family's primes, ascending",
                             [](const ObjectPtr& o) {
                                 std::vector<Value> out;
                                 for (const auto& kv : BlockOf(o).char_poly)
                                     out.push_back(Value::from_int(
                                         static_cast<std::int64_t>(kv.first)));
                                 return Value::from_list(std::move(out));
                             }});
        t.members.push_back({"witness", true,
                             "the linear residue witness as [[p, c], ...] (empty in strict mode, "
                             "where the witness is a matrix)",
                             [](const ObjectPtr& o) {
                                 std::vector<Value> out;
                                 for (const auto& w : BlockOf(o).residue_witness) {
                                     out.push_back(Value::from_list(
                                         {Value::from_int(static_cast<std::int64_t>(w.first)),
                                          Value::from_int(w.second)}));
                                 }
                                 return Value::from_list(std::move(out));
                             }});
        {
            TypeInfo::Method m;
            m.name = "min_poly";
            m.params = {"p"};
            m.required = 1;
            m.doc = "the minimal polynomial of T_p on this block";
            m.call = [](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                return PolyMapMember(o, BlockOf(o).min_poly, args, pos, "Block.min_poly(p)");
            };
            t.methods.push_back(std::move(m));
        }
        {
            TypeInfo::Method m;
            m.name = "char_poly";
            m.params = {"p"};
            m.required = 1;
            m.doc = "the characteristic polynomial of T_p on this block";
            m.call = [](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                return PolyMapMember(o, BlockOf(o).char_poly, args, pos, "Block.char_poly(p)");
            };
            t.methods.push_back(std::move(m));
        }
        {
            TypeInfo::Method m;
            m.name = "character";
            m.params = {"p"};
            m.required = 1;
            m.doc = "P_p, the character polynomial of the locality certificate";
            m.call = [](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                return PolyMapMember(o, BlockOf(o).character, args, pos, "Block.character(p)");
            };
            t.methods.push_back(std::move(m));
        }
        {
            TypeInfo::Method m;
            m.name = "jordan";
            m.params = {"p"};
            m.required = 1;
            m.doc = "the generalized-Jordan partition of T_p on this block, parts descending";
            m.call = [](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                const uint64_t p = U64Arg(args[0], pos, "Block.jordan(p)");
                const auto& jp = BlockOf(o).jordan_partition;
                const auto it = jp.find(p);
                if (it == jp.end()) {
                    throw RuntimeError("Block.jordan(p): p = " + std::to_string(p) +
                                           " is not in this block's operator family",
                                       pos);
                }
                return IntList(it->second);
            };
            t.methods.push_back(std::move(m));
        }
        t.print = [](const ObjectPtr& o, std::int64_t) {
            const hecke::PrimaryBlock& b = BlockOf(o);
            std::ostringstream os;
            os << "Block dim " << b.dim << ", f " << b.residue_degree
               << (b.local_certified ? ", local" : ", locality UNCERTIFIED");
            return os.str();
        };
        g_block_type = registry.register_type(std::move(t));
    }

    // --- Decomposition -----------------------------------------------------
    {
        TypeInfo t;
        t.name = kTypeDecomposition;
        t.doc = "the certified primary decomposition of H^q as a module over the commutative "
                "F_ell-algebra generated by a commuting family of Hecke matrices";
        t.members.push_back({"q", false, "the cohomological degree", [](const ObjectPtr& o) {
                                 return Value::from_int(Dec(o).q);
                             }});
        t.members.push_back({"ell", false, "the coefficient prime", [](const ObjectPtr& o) {
                                 return Value::from_int(Dec(o).ell);
                             }});
        t.members.push_back({"n", false, "n_sym of the coefficient system",
                             [](const ObjectPtr& o) { return Value::from_int(Dec(o).n_sym); }});
        t.members.push_back({"primes", false, "the family's primes, in the order given",
                             [](const ObjectPtr& o) {
                                 std::vector<Value> out;
                                 for (const auto& kv : Dec(o).ops)
                                     out.push_back(
                                         Value::from_int(static_cast<std::int64_t>(kv.first)));
                                 return Value::from_list(std::move(out));
                             }});
        t.members.push_back({"blocks", true, "the primary blocks, in canonical order",
                             [](const ObjectPtr& o) {
                                 DecompositionVal& dv = Dec(o);
                                 dv.ensure();
                                 std::vector<Value> out;
                                 for (std::size_t i = 0; i < dv.D->blocks().size(); ++i) {
                                     auto bv = std::make_shared<BlockVal>();
                                     bv->parent = o;
                                     bv->index = i;
                                     ObjectPtr b = MakeValueObjectFrom(g_block_type, o, bv);
                                     b->keep_alive(o);
                                     out.push_back(Value::from_object(b));
                                 }
                                 return Value::from_list(std::move(out));
                             }});
        t.members.push_back({"num_blocks", true, "how many primary blocks there are",
                             [](const ObjectPtr& o) {
                                 DecompositionVal& dv = Dec(o);
                                 dv.ensure();
                                 return Value::from_int(
                                     static_cast<std::int64_t>(dv.D->blocks().size()));
                             }});
        t.members.push_back({"U", true, "the change of basis: the block bases, in block order",
                             [](const ObjectPtr& o) {
                                 DecompositionVal& dv = Dec(o);
                                 dv.ensure();
                                 return MatrixValueFrom(o, Flat(dv.D->change_of_basis()));
                             }});
        t.members.push_back(
            {"witness_hash", true,
             "the golden `decomp_*_witness` hash (the residue-witness data), as 0x... text",
             [](const ObjectPtr& o) {
                 DecompositionVal& dv = Dec(o);
                 dv.ensure();
                 return Value::from_string(HexU64(DecompositionWitnessHash(*dv.D)));
             }});
        t.members.push_back({"locality", true,
                             "the E0b locality loop's state: mode, budget, splits, completeness",
                             [](const ObjectPtr& o) {
                                 DecompositionVal& dv = Dec(o);
                                 dv.ensure();
                                 const hecke::LocalityState& ls = dv.D->locality_state();
                                 Record r;
                                 r.set("mode", Value::from_string(
                                                   ls.mode == hecke::LocalityMode::Strict
                                                       ? "strict"
                                                       : "budget"));
                                 r.set("budget", Value::from_int(ls.budget));
                                 r.set("complete", Value::from_bool(ls.complete));
                                 r.set("splits", Value::from_int(
                                                     static_cast<std::int64_t>(ls.splits)));
                                 r.set("interrupted", Value::from_bool(ls.interrupted));
                                 r.set("phase_reached",
                                       Value::from_string(hecke::to_string(ls.phase_reached)));
                                 return Value::from_record(std::move(r));
                             }});
        t.members.push_back({"json", true, "the decomposition's own JSON record",
                             [](const ObjectPtr& o) {
                                 DecompositionVal& dv = Dec(o);
                                 dv.ensure();
                                 return Value::from_string(dv.D->to_json());
                             }});
        {
            TypeInfo::Method m;
            m.name = "certify";
            m.required = 0;
            m.doc = "the UNCONDITIONAL certificate: 0 iff every claim re-derives (never trusting "
                    "the randomized factor search)";
            m.call = [](const ObjectPtr& o, const std::vector<Value>&, const SourcePos&) {
                DecompositionVal& dv = Dec(o);
                dv.ensure();
                return Value::from_int(static_cast<std::int64_t>(dv.D->certify()));
            };
            t.methods.push_back(std::move(m));
        }
        {
            TypeInfo::Method m;
            m.name = "conjugated";
            m.params = {"p"};
            m.required = 1;
            m.doc = "U^{-1} A^{(q)}(T_p) U, verified exactly block-diagonal";
            m.call = [](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                DecompositionVal& dv = Dec(o);
                dv.ensure();
                const uint64_t p = U64Arg(args[0], pos, "Decomposition.conjugated(p)");
                return MatrixValueFrom(o, Flat(dv.D->conjugated(p)));
            };
            t.methods.push_back(std::move(m));
        }
        // (C2a, slot-6 WP-13) R3: `resume(D, budget)` continues the locality
        // loop from THIS handle's state (HeckeModuleDecomposition::
        // resume_locality) under a higher budget.  The budget may be given as
        // an Int N or as the setting's own text "budget=<N>"; it is parsed by
        // hecke::MathSettings::set -- the one validation point -- and the
        // resumed handle's snapshot carries it as the `locality` SETTING text,
        // so `state(R).settings.locality` reads exactly what `set` would have
        // stored.  A budget below the one reached is refused HERE, eagerly,
        // with the engine's named error text.
        t.resume = [](const ObjectPtr& parent, const SettingsSnapshot& snap) -> ObjectPtr {
            DecompositionVal& pv = Dec(parent);
            pv.ensure();
            const hecke::LocalityState& ls = pv.D->locality_state();
            const Value* b = snap.find("locality");
            std::string text;
            if (b != nullptr && b->is_int()) {
                if (b->int_value() < 0) {
                    throw RuntimeError("resume(D, budget): a locality budget is a positive Int",
                                       parent->creation_site());
                }
                text = "budget=" + std::to_string(b->int_value());
            } else if (b != nullptr && b->is_string()) {
                text = b->string_value();
            } else {
                throw RuntimeError("resume(D, budget): a Decomposition is resumed by its "
                                   "`locality` budget (an Int N or \"budget=<N>\")",
                                   parent->creation_site());
            }
            hecke::MathSettings probe = hecke::math_settings();
            try {
                probe.set("locality", text, hecke::SettingSource::Console);
            } catch (const std::invalid_argument& e) {
                throw RuntimeError(std::string("resume(D, budget): ") + e.what(),
                                   parent->creation_site());
            }
            if (probe.locality.mode != hecke::LocalityMode::Budget) {
                throw RuntimeError("resume(D, budget): a locality resume continues the BUDGET "
                                   "loop, so the new value must be budget=<N>, not '" +
                                       text + "'",
                                   parent->creation_site());
            }
            const uint32_t n = probe.locality.budget;
            if (n < ls.budget) {
                throw RuntimeError("resume: budget below reached state (this handle reached "
                                   "budget=" + std::to_string(ls.budget) + ", asked for budget=" +
                                       std::to_string(n) + ")",
                                   parent->creation_site());
            }
            auto nv = std::make_shared<DecompositionVal>();
            nv->sources = pv.sources;
            nv->ops = pv.ops;
            nv->ell = pv.ell;
            nv->n_sym = pv.n_sym;
            nv->q = pv.q;
            nv->num_degrees = pv.num_degrees;
            nv->resume_parent = parent;
            nv->resume_budget = n;
            const SettingsSnapshot rsnap =
                snap.with("locality", Value::from_string(probe.locality.to_text()));
            ObjectPtr o = MakeDecompositionHandle(parent->type(), parent->creation_site(),
                                                  parent->creation_expr(), rsnap, nv);
            o->keep_alive(parent);
            return o;
        };
        // The golden `decomp_*_invariants` hash, byte for byte.
        t.hash = [](const ObjectPtr& o) {
            DecompositionVal& dv = Dec(o);
            dv.ensure();
            return DecompositionInvariantsHash(*dv.D);
        };
        t.print = [](const ObjectPtr& o, std::int64_t) {
            const DecompositionVal* dv = o->payload_as<DecompositionVal>();
            std::ostringstream os;
            os << "Decomposition of H^" << dv->q << " over " << dv->ops.size() << " operator(s)";
            if (!o->computed()) {
                os << " (declared, not yet computed)";
            } else {
                os << ": " << dv->D->blocks().size() << " block(s)";
            }
            return os.str();
        };
        registry.register_type(std::move(t));
    }

    // --- Report ------------------------------------------------------------
    {
        TypeInfo t;
        t.name = kTypeReport;
        t.doc = "the hecke_report/v2 record of one degree: level (b) A^{(q)}(T_p), level (c) the "
                "decomposition, and optionally level (a) Theta^{(q)}_flat";
        t.members.push_back({"schema", false, "the frozen schema name",
                             [](const ObjectPtr&) {
                                 return Value::from_string("hecke_report/v2");
                             }});
        t.members.push_back({"level_a", false, "whether level (a) was included",
                             [](const ObjectPtr& o) {
                                 return Value::from_bool(o->payload_as<ReportVal>()->level_a);
                             }});
        t.members.push_back({"path", false, "the file it was written to, or \"\"",
                             [](const ObjectPtr& o) {
                                 return Value::from_string(o->payload_as<ReportVal>()->path);
                             }});
        t.members.push_back({"degree", true, "the degree q this record is about",
                             [](const ObjectPtr& o) {
                                 return Value::from_int(Dec(Rep(o).D).q);
                             }});
        // The four PER-COMPLEX degree fields, derived exactly as the writer
        // derives them ([C] sec:recordschema item 5).
        t.members.push_back({"q_max", true, "the top degree of THIS complex",
                             [](const ObjectPtr& o) {
                                 return Value::from_int(Dec(Rep(o).D).num_degrees - 1);
                             }});
        t.members.push_back({"truncated", true, "false on a finite model",
                             [](const ObjectPtr&) { return Value::from_bool(false); }});
        t.members.push_back({"q_max_source", true, "\"derived\" unless a setting pinned q_max",
                             [](const ObjectPtr&) { return Value::from_string("derived"); }});
        t.members.push_back({"degrees", true, "[0 .. q_max]", [](const ObjectPtr& o) {
                                 std::vector<uint32_t> d;
                                 for (uint32_t i = 0; i < Dec(Rep(o).D).num_degrees; ++i)
                                     d.push_back(i);
                                 return IntList(d);
                             }});
        t.members.push_back({"json", true, "the report text", [](const ObjectPtr& o) {
                                 ReportVal& rv = Rep(o);
                                 rv.ensure();
                                 return Value::from_string(rv.json);
                             }});
        {
            TypeInfo::Method m;
            m.name = "write";
            m.params = {"path"};
            m.required = 1;
            m.doc = "write the report to a file (an empty path writes to stdout)";
            m.call = [](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                if (!args[0].is_string())
                    throw TypeError("Report.write(path): path must be a String", pos);
                if (g_report_write_forbidden)
                    throw RuntimeError("Report.write(path): not available in a served session", pos);
                ReportVal& rv = Rep(o);
                rv.ensure();
                hecke::WriteHeckeReport(rv.json, args[0].string_value());
                return Value::from_string(args[0].string_value());
            };
            t.methods.push_back(std::move(m));
        }
        {
            TypeInfo::Method m;
            m.name = "has_key";
            m.params = {"key"};
            m.required = 1;
            m.doc = "whether the record carries a given JSON key -- the schema assertion a "
                    "transcript can make without freezing the whole text";
            m.call = [](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                if (!args[0].is_string())
                    throw TypeError("Report.has_key(key): key must be a String", pos);
                ReportVal& rv = Rep(o);
                rv.ensure();
                const std::string needle = "\"" + args[0].string_value() + "\":";
                return Value::from_bool(rv.json.find(needle) != std::string::npos);
            };
            t.methods.push_back(std::move(m));
        }
        t.print = [](const ObjectPtr& o, std::int64_t) {
            const ReportVal* rv = o->payload_as<ReportVal>();
            std::ostringstream os;
            os << "Report hecke_report/v2 (level_a = " << (rv->level_a ? "true" : "false") << ")";
            if (!o->computed()) os << " (declared, not yet computed)";
            return os.str();
        };
        registry.register_type(std::move(t));
    }
}

// --- the engine-owned MATHEMATICAL settings --------------------------------
//
// [OD] R1: "The engine owns the defaults ... The console sets them through the
// registry and never re-implements them."  So the key LIST comes from
// hecke::MathSettings::keys(), the default TEXT from a default-constructed
// MathSettings, and both validation and application go through
// MathSettings::set() -- the one validation point.  A key a later slot appends
// therefore becomes a console `set` key with no console edit at all.
//
// The default text comes from hecke::MathSettings::text(key) (slot-4 WP-1/WP-5
// integration).  Until then this TU carried a per-key switch of its own, which
// returned "" for a key appended after it was written: slot-4 WP-1's
// `commutant_budget` and `final_mode` therefore displayed an empty default and,
// worse, made the no-op guard below compare against "" -- so a console round
// trip back to the shipped default stamped `console` where a virgin session has
// `default`, which is exactly the divergence that guard exists to prevent.
// A second copy of the key list was the defect; asking the engine removes it.

void InstallMathSettings(Settings& settings) {
    const hecke::MathSettings& defaults = hecke::MathSettings::defaults();
    for (const std::string& key : hecke::MathSettings::keys()) {
        SettingSpec spec;
        spec.name = key;
        spec.kind = SettingKind::String;
        spec.default_value = Value::from_string(defaults.text(key));
        spec.source = SettingSource::Session;
        spec.family = SettingFamily::Mathematical;
        spec.doc = "mathematical setting `" + key +
                   "` (hecke::MathSettings -- it changes what a result MEANS, so a non-default "
                   "value moves the fingerprint and `golden(...)` refuses)";
        spec.validate = [key](const Value& v) -> std::string {
            if (!v.is_string()) return "expected a String value";
            hecke::MathSettings probe = hecke::math_settings();
            try {
                if (!probe.set(key, v.string_value(), hecke::SettingSource::Console)) {
                    return "unknown mathematical setting '" + key + "'";
                }
            } catch (const std::invalid_argument& e) {
                return e.what();
            } catch (const hecke::NotImplementedInThisBuild& e) {
                // (C4, slot-8 WP-10; reported by WP-3) A value the settings
                // layer itself refuses as legal-but-not-executable ([E]
                // sec:ring (a): `coefficient_ring = Z`) is exit 5, not a bad
                // value.  The `set` statement reaches Settings::set directly,
                // never the registry's kind classifier, so the raw type would
                // surface as exit 3; the console's own NotImplementedError
                // keeps the kind.  (Its position is not known here: the core
                // statement does not back-fill it -- a Console/Core/ item.)
                throw NotImplementedError(
                    std::string("hecke::NotImplementedInThisBuild: ") + e.what(), SourcePos());
            }
            return std::string();
        };
        const std::string default_text = spec.default_value.string_value();
        spec.apply = [key, default_text](const Value& v) {
            if (!v.is_string()) return;
            const std::string& want = v.string_value();
            // A no-op must stay a no-op.  Settings::install() re-applies EVERY
            // value (plan Sec. 2 C4: there is no RAII stack), so without this
            // guard every handle's compute() would re-`set` `locality` and
            // stamp its source `console` -- which the record PRINTS
            // (`level_c.locality.source`), so a console report of a default
            // session would differ from the golden `report_*` text.
            if (want == hecke::math_settings().text(key)) return;
            hecke::MathSettings next = hecke::math_settings();
            // Coming BACK to the shipped default restores the `default` source
            // too, so the round trip `set k = x` / `set k = <default>` leaves
            // the record byte-identical to a session that never set anything.
            const hecke::SettingSource src = (want == default_text)
                                                 ? hecke::SettingSource::Default
                                                 : hecke::SettingSource::Console;
            // A value this build cannot honour raises
            // hecke::NotImplementedInThisBuild, which the C7 seam maps to exit
            // code 5.  It is deliberately NOT caught here.
            if (next.set(key, want, src)) {
                hecke::set_math_settings(next);
            }
        };
        settings.register_setting(std::move(spec));
    }
}

} // namespace

void ForbidReportWrite() { g_report_write_forbidden = true; }

void RegisterDecompositionBindings(Registry& registry) {
    RegisterTypes(registry);
    registry.set_settings_installer(&InstallMathSettings);

    const char* g = "pipeline";
    auto add = [&registry, g](const char* name, Signature sig, VerbImpl impl, const char* doc) {
        registry.register_verb(name, std::move(sig), std::move(impl), doc, g);
    };

    add("decompose", Sig({Param::any("ops"), Param::of("q", ValueKind::Int)}), &VerbDecompose,
        "the certified primary decomposition of H^q over a commuting Hecke family (one operator "
        "or a list of them); q is mandatory");
    // Two overloads ([P] v3.10.2 Sec. 18.14): the pathless one only computes the
    // record (read_only); the path one WRITES a file (stays untagged).  The
    // pathless one is registered first, so every legal call resolves as before.
    add("report",
        Sig({Param::object("D", kTypeDecomposition),
             Param::of("level_a", ValueKind::Bool).optional(Value::from_bool(false))}),
        &VerbReport, "the hecke_report/v2 record of the decomposition's degree");
    add("report",
        Sig({Param::object("D", kTypeDecomposition),
             Param::of("level_a", ValueKind::Bool).optional(Value::from_bool(false)),
             Param::of("path", ValueKind::String).optional(Value::from_string(""))}),
        &VerbReport,
        "the hecke_report/v2 record of the decomposition's degree, also written to a file");
    add("crosschecks", Sig({Param::object("T", kTypeHecke)}), &VerbCrossChecks,
        "the classical alpha/beta/omega factorization checks (i)/(ii)/(iii) against the fused "
        "pipeline -- expensive in p");
}

// --- (C2a, slot-6 WP-13) the two test seams of Bindings.h -------------------

ObjectPtr MakeMatrixDecomposition(CallContext& ctx,
                                  std::vector<std::pair<uint64_t, MatrixT<uint32_t>>> ops,
                                  uint32_t ell) {
    CONSOLE_CHECK_MSG(!ops.empty(), "MakeMatrixDecomposition: the family is empty");
    auto v = std::make_shared<DecompositionVal>();
    v->ops = std::move(ops);
    v->ell = ell;
    v->q = 0;
    v->num_degrees = 0;
    return MakeDecompositionHandle(TypeByName(ctx, kTypeDecomposition), ctx.pos, ctx.call_text,
                                   ctx.session->settings().snapshot(), v);
}

const hecke::HeckeModuleDecomposition* DecompositionOf(const ObjectPtr& o) {
    if (!o || o->type() == nullptr || o->type()->name != kTypeDecomposition) return nullptr;
    if (!o->computed()) return nullptr;
    return Dec(o).D.get();
}

} // namespace bind
} // namespace console
