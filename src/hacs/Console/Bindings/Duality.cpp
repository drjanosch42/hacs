// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Duality.cpp -- console phase C5a (hecke-extensions slot 9, WP-12): E7's two
// reserved types ([P] Sec. 3.6, "Reserved additionally by [X] v1.5.1"; [P]
// Sec. 10 C5, "The binding list, extended"):
//
//   ContragredientLocalSystem  `dual(L)` -- D_n = Sym^n(F_ell^2)^vee on the
//       7-vertex model with rho_n^vee(h) = rho_n(adj h)^T ([C] def:contragredient),
//       its cohomology, the map Phi_n : M_n -> D_n and H^*(Phi_n), and its Hecke
//       action A(D_n)(T_p)^q -- hecke::ContragredientCell (Hecke/HeckeContragredient).
//   Pairing  `pairing(L, D)` -- the cone-level pairing H^t_c(M_n) x H^{2-t}(D_n)
//       -> F_ell and the circle pairing, their Grams and perfectness, and the
//       PD adjointness gate A_c^{(t)T} G^t == G^t A(D_n)^{(2-t)} (factor 1, no
//       determinant twist) as a Record of booleans -- hecke::PoincareDualityPairing
//       (Hecke/HeckePairing).  [C] cl:pd is OPEN: the gate is a TEST of it, and
//       the console reports what the shipped code computes.
//
// GOLDEN IDENTITY.  Every object the golden harness freezes at an E7 point
// (`n=<n> ell=<ell> e7`, GoldenHarness.cpp RunE7Cell32) is reachable as a
// value whose console hash is byte-identical to the harness's:
//   dim_H_Dn^q             hash(D.dims[q])                     (a scalar)
//   charpoly_A_Dn(T_p)^q   hash(charpoly(D.A(p, q)))           (a polynomial; 1 at 0x0)
//   pairing_perfect^t      P.perfect(t) == (golden value == 1)
//   pd_adjoint^t           P.pd_gate([2, 3, 5]).t<t>, the harness's primes
//   pairing_matrix^t       hash(P.gram(t))                     (a matrix, tier B)
// R19 still refuses `golden(...)` at a non-zero fingerprint.
//
// THE RING.  Neither verb is on [E] sec:ring (h)'s honouring list, so both
// refuse a non-default coefficient_ring AT DISPATCH (exit 5), and refuse a
// LocalSystem handle built over Z/ell^k.
// ===========================================================================

#include "Bindings.h"
#include "Payloads.h"

#include "../../Hecke/HeckeBoundaryDatum.h"
#include "../../Hecke/HeckeBoundaryOperator.h"
#include "../../Hecke/HeckeConeAction.h"
#include "../../Hecke/HeckeContragredient.h"
#include "../../Hecke/HeckePairHomotopy.h"
#include "../../Hecke/HeckePairing.h"
#include "../../Hecke/HeckeSettings.h"
#include "../../Hecke/SL2ZBoundaryPair.h"

#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace console {
namespace bind {

namespace {

using Flat = FlatMatrixT<uint32_t>;

const TypeInfo* g_matrix_type = nullptr;

Value MatrixValueFrom(const ObjectPtr& owner, Flat m) {
    auto p = std::make_shared<MatrixVal>();
    p->m = std::move(m);
    return Value::from_object(MakeValueObjectFrom(g_matrix_type, owner, std::move(p)));
}

// --- payloads ------------------------------------------------------------------

struct ContragredientVal {
    uint32_t n_sym = 0;
    uint32_t ell = 0;
    std::unique_ptr<hecke::ContragredientCell> cell;
    // One computed pair (op_M, op_D) per Hecke prime, built on first use.
    std::map<uint32_t, hecke::ContragredientCell::HeckePair> heckes;

    void ensure() {
        if (cell) return;
        hecke::RequireFieldCoefficientRing("console verb dual (ContragredientLocalSystem)");
        cell.reset(new hecke::ContragredientCell(n_sym, ell));
    }
    const hecke::ContragredientCell::HeckePair& pair(uint32_t p) {
        CONSOLE_CHECK(cell != nullptr);
        auto it = heckes.find(p);
        if (it != heckes.end()) return it->second;
        hecke::ContragredientCell::HeckePair hp;
        {
            // tau construction is first-match-wins over a fixed order and drives
            // non-atomic counters (Hecke.cpp); the same lock as `hecke(H, p)`.
            std::lock_guard<std::mutex> guard(hecke::tau_construction_mutex());
            hp = cell->hecke_pair(p);
        }
        return heckes.emplace(p, std::move(hp)).first->second;
    }
};

struct PairingVal {
    ObjectPtr D;                                   // keep-alive: the pairing borrows the cell
    std::unique_ptr<hecke::PoincareDualityPairing> pd;
    // Per Hecke prime, the cone action the adjointness gate needs -- exactly the
    // harness's E7Op (GoldenHarness.cpp RunE7Cell32).
    struct GateOp {
        std::unique_ptr<hecke::HeckeBoundaryTauTable> taub;
        std::unique_ptr<hecke::HeckePairHomotopy> eta;
        std::unique_ptr<hecke::HeckeBoundaryOperator> bd;
        std::unique_ptr<hecke::HeckeConeAction> act;
        std::vector<char> adjoint;                 // t = 0..2; t = 0 vacuous (H^0_c = 0), recorded 1
    };
    std::map<uint32_t, std::unique_ptr<GateOp>> gates;

    ContragredientVal& dv() { return *D->payload_as<ContragredientVal>(); }

    void ensure() {
        if (pd) return;
        hecke::RequireFieldCoefficientRing("console verb pairing (Pairing)");
        dv().ensure();
        pd.reset(new hecke::PoincareDualityPairing(*dv().cell));
    }
    const std::vector<char>& gate(uint32_t p) {
        CONSOLE_CHECK(pd != nullptr);
        auto it = gates.find(p);
        if (it != gates.end()) return it->second->adjoint;
        const hecke::ContragredientCell& cell = *dv().cell;
        const hecke::ContragredientCell::HeckePair& hp = dv().pair(p);
        auto g = std::make_unique<GateOp>();
        g->taub.reset(new hecke::HeckeBoundaryTauTable(cell.K(), cell.model(), *hp.cosets));
        g->eta.reset(new hecke::HeckePairHomotopy(*hp.tau, *g->taub));
        g->bd.reset(new hecke::HeckeBoundaryOperator(&pd->pair_M().boundary_cohomology(), hp.cosets.get(),
                                                     g->taub.get()));
        g->bd->compute();
        g->act.reset(new hecke::HeckeConeAction(&pd->cone(), hp.op_M.get(), g->bd.get(), g->eta.get()));
        g->act->compute();
        const auto& A_D = hp.op_D->get_H_matrices();
        CONSOLE_CHECK_MSG(A_D.size() == 3, "A(D_n)(T_p) must have three degrees");
        g->adjoint.assign(3, 1);
        for (uint32_t t = 1; t < 3; ++t) {
            if (!hecke::PairingAdjoint(g->act->A_c(t), pd->gram(t), A_D[2 - t], 1u, cell.ell())) {
                g->adjoint[t] = 0;
            }
        }
        return gates.emplace(p, std::move(g)).first->second->adjoint;
    }
};

ContragredientVal& Dual(const ObjectPtr& o) { return *o->payload_as<ContragredientVal>(); }
PairingVal& Pair(const ObjectPtr& o) { return *o->payload_as<PairingVal>(); }

uint32_t IndexArg(const Value& v, const SourcePos& pos, const std::string& who, uint32_t n) {
    const uint32_t i = U32Arg(v, pos, who.c_str());
    if (i >= n) {
        throw RuntimeError(who + ": index " + std::to_string(i) + " is out of range (0 .. " +
                               std::to_string(n - 1) + ")",
                           pos);
    }
    return i;
}

uint32_t PrimeArg(const Value& v, const SourcePos& pos, const char* who) {
    const uint32_t p = U32Arg(v, pos, who);
    if (p < 2) throw TypeError(std::string(who) + ": p must be a prime >= 2", pos);
    return p;
}

// --- verbs ---------------------------------------------------------------------

Value VerbDual(CallContext& ctx, const std::vector<Value>& a) {
    static const char* kWho = "dual(L)";
    const ObjectPtr& L = AsObjectOf(a[0], kTypeLocalSys, ctx.pos, kWho);
    // [E] sec:ring (h): not on the honouring list -- exit 5 under a ring.
    hecke::RequireFieldCoefficientRing("console verb dual");
    RequireFieldHandle(L, "console verb dual");
    const LocalSystemVal* lv = L->payload_as<LocalSystemVal>();
    if (lv->model == nullptr) {
        throw RuntimeError(std::string(kWho) + ": the preset local system '" + lv->preset +
                               "' carries no SL(2,Z) tessellation model -- the contragredient is "
                               "built on the 7-vertex model only",
                           ctx.pos);
    }
    auto v = std::make_shared<ContragredientVal>();
    v->n_sym = lv->n_sym;
    v->ell = lv->ell;
    ObjectPtr obj = Object::create(TypeByName(ctx, "ContragredientLocalSystem"), ctx.pos, ctx.call_text,
                                   ctx.session->settings().snapshot(), v,
                                   LazyCompute([v](Object&) { v->ensure(); }));
    obj->keep_alive(L);
    return Value::from_object(obj);
}

Value VerbPairing(CallContext& ctx, const std::vector<Value>& a) {
    static const char* kWho = "pairing(L, D)";
    const ObjectPtr& L = AsObjectOf(a[0], kTypeLocalSys, ctx.pos, kWho);
    const ObjectPtr& D = AsObjectOf(a[1], "ContragredientLocalSystem", ctx.pos, kWho);
    hecke::RequireFieldCoefficientRing("console verb pairing");
    RequireFieldHandle(L, "console verb pairing");
    const LocalSystemVal* lv = L->payload_as<LocalSystemVal>();
    const ContragredientVal& dv = Dual(D);
    if (lv->model == nullptr || lv->n_sym != dv.n_sym || lv->ell != dv.ell) {
        throw TypeError(std::string(kWho) + ": D must be the contragredient of L (dual(L)); got L = Sym^" +
                            std::to_string(lv->n_sym) + " over F_" + std::to_string(lv->ell) +
                            ", D of Sym^" + std::to_string(dv.n_sym) + " over F_" + std::to_string(dv.ell),
                        ctx.pos);
    }
    auto v = std::make_shared<PairingVal>();
    v->D = D;
    ObjectPtr obj = Object::create(TypeByName(ctx, "Pairing"), ctx.pos, ctx.call_text,
                                   ctx.session->settings().snapshot(), v,
                                   LazyCompute([v](Object&) { v->ensure(); }));
    obj->keep_alive(L);
    obj->keep_alive(D);
    return Value::from_object(obj);
}

// --- types ---------------------------------------------------------------------

void RegisterTypes(Registry& registry) {
    g_matrix_type = registry.find_type(kTypeMatrix);
    CONSOLE_CHECK_MSG(g_matrix_type != nullptr,
                      "RegisterCoreMathBindings must run before RegisterDualityBindings");
    {
        TypeInfo t;
        t.name = "ContragredientLocalSystem";
        t.doc = "D_n = Sym^n(F_ell^2)^vee on the 7-vertex model (rho_n^vee(h) = rho_n(adj h)^T), its "
                "cohomology, Phi_n : M_n -> D_n and the Hecke action A(D_n)(T_p) (E7; lazy)";
        t.members.push_back({"n", false, "the symmetric power", [](const ObjectPtr& o) {
                                 return Value::from_int(Dual(o).n_sym);
                             }});
        t.members.push_back({"p", false, "the coefficient prime ell", [](const ObjectPtr& o) {
                                 return Value::from_int(Dual(o).ell);
                             }});
        t.members.push_back({"d_V", false, "dim D_n = n + 1", [](const ObjectPtr& o) {
                                 return Value::from_int(Dual(o).n_sym + 1);
                             }});
        t.members.push_back({"coefficient_ring", false, "F_ell (E7 is a field object)",
                             [](const ObjectPtr&) { return Value::from_string("F_ell"); }});
        t.members.push_back({"phi_is_isomorphism", false,
                             "whether Phi_n (x) F_ell is an isomorphism, i.e. n < ell ([C] lem:symdual)",
                             [](const ObjectPtr& o) {
                                 return Value::from_bool(hecke::PhiNIsIsomorphism(Dual(o).n_sym, Dual(o).ell));
                             }});
        t.members.push_back({"dims", true, "dim H^q(Gamma; D_n) for every degree q (golden dim_H_Dn^q)",
                             [](const ObjectPtr& o) {
                                 return IntList(Dual(o).cell->coh_D().get_H_eq_dimensions());
                             }});
        t.members.push_back({"dims_M", true, "dim H^q(Gamma; M_n), the standard module, for comparison",
                             [](const ObjectPtr& o) {
                                 return IntList(Dual(o).cell->coh_M().get_H_eq_dimensions());
                             }});
        t.members.push_back({"Phi", true, "Phi_n : M_n -> D_n, (n+1) x (n+1) over F_ell",
                             [](const ObjectPtr& o) { return MatrixValueFrom(o, Dual(o).cell->Phi()); }});
        {
            TypeInfo::Method m;
            m.name = "induced_Phi";
            m.params = {"q"};
            m.required = 1;
            m.computed = true;
            m.doc = "P^q = H^q(Phi_n) : H^q(M_n) -> H^q(D_n)";
            m.call = [](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                const auto& P = Dual(o).cell->induced_Phi();
                const uint32_t q = IndexArg(args[0], pos, "ContragredientLocalSystem.induced_Phi(q)",
                                            static_cast<uint32_t>(P.size()));
                return MatrixValueFrom(o, P[q]);
            };
            t.methods.push_back(std::move(m));
        }
        {
            TypeInfo::Method m;
            m.name = "A";
            m.params = {"p", "q"};
            m.required = 2;
            m.computed = true;
            m.doc = "A(D_n)(T_p)^q, the Hecke action on H^q(D_n) (U_ell at p = ell); "
                    "charpoly(D.A(p, q)) is the golden charpoly_A_Dn(T_p)^q";
            m.call = [](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                const uint32_t p = PrimeArg(args[0], pos, "ContragredientLocalSystem.A(p, q)");
                const auto& A = Dual(o).pair(p).op_D->get_H_matrices();
                const uint32_t q = IndexArg(args[1], pos, "ContragredientLocalSystem.A(p, q)",
                                            static_cast<uint32_t>(A.size()));
                return MatrixValueFrom(o, Flat(A[q]));
            };
            t.methods.push_back(std::move(m));
        }
        {
            TypeInfo::Method m;
            m.name = "A_M";
            m.params = {"p", "q"};
            m.required = 2;
            m.computed = true;
            m.doc = "A(M_n)(T_p)^q on the standard module, built from the SAME coset / tau data";
            m.call = [](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                const uint32_t p = PrimeArg(args[0], pos, "ContragredientLocalSystem.A_M(p, q)");
                const auto& A = Dual(o).pair(p).op_M->get_H_matrices();
                const uint32_t q = IndexArg(args[1], pos, "ContragredientLocalSystem.A_M(p, q)",
                                            static_cast<uint32_t>(A.size()));
                return MatrixValueFrom(o, Flat(A[q]));
            };
            t.methods.push_back(std::move(m));
        }
        {
            TypeInfo::Method m;
            m.name = "intertwines";
            m.params = {"p"};
            m.required = 1;
            m.computed = true;
            m.doc = "{q0, q1, q2}: whether A(D_n)(T_p) P^q == P^q A(M_n)(T_p) exactly (UNTWISTED, "
                    "factor 1; [C] prop:pairing)";
            m.call = [](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                const uint32_t p = PrimeArg(args[0], pos, "ContragredientLocalSystem.intertwines(p)");
                ContragredientVal& dv = Dual(o);
                const auto& hp = dv.pair(p);
                const auto& AD = hp.op_D->get_H_matrices();
                const auto& AM = hp.op_M->get_H_matrices();
                const auto& P = dv.cell->induced_Phi();
                Record r;
                for (uint32_t q = 0; q < AD.size() && q < AM.size() && q < P.size(); ++q) {
                    r.set("q" + std::to_string(q),
                          Value::from_bool(hecke::HeckeIntertwines(AD[q], P[q], AM[q], 1u, dv.ell)));
                }
                return Value::from_record(std::move(r));
            };
            t.methods.push_back(std::move(m));
        }
        t.print = [](const ObjectPtr& o, std::int64_t) {
            const ContragredientVal& v = Dual(o);
            std::ostringstream os;
            os << "ContragredientLocalSystem Sym^" << v.n_sym << "(F_" << v.ell << "^2)^vee, d_V = "
               << (v.n_sym + 1);
            if (!o->computed()) os << " (declared, not yet computed)";
            return os.str();
        };
        registry.register_type(std::move(t));
    }
    {
        TypeInfo t;
        t.name = "Pairing";
        t.doc = "the cone-level pairing H^t_c(M_n) x H^{2-t}(D_n) -> F_ell and the circle pairing: "
                "Grams, perfectness, the PD adjointness gate of [C] cl:pd (E7; lazy)";
        t.members.push_back({"coefficient_ring", false, "F_ell (E7 is a field object)",
                             [](const ObjectPtr&) { return Value::from_string("F_ell"); }});
        t.members.push_back({"certificate", true,
                             "the pairing certificate re-derived: {failing, failing_items}",
                             [](const ObjectPtr& o) {
                                 const hecke::PairingCertificate& c = Pair(o).pd->certificate();
                                 Record r;
                                 r.set("failing", Value::from_int(c.verify()));
                                 r.set("failing_items", IntList(c.failing_items()));
                                 return Value::from_record(std::move(r));
                             }});
        auto index_method = [](const char* name, const char* doc, uint32_t n,
                               std::function<Value(const ObjectPtr&, uint32_t)> f) {
            TypeInfo::Method m;
            m.name = name;
            m.params = {n == 3 ? "t" : "s"};
            m.required = 1;
            m.computed = true;
            m.doc = doc;
            const std::string who = std::string("Pairing.") + name + (n == 3 ? "(t)" : "(s)");
            m.call = [f, n, who](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                return f(o, IndexArg(args[0], pos, who, n));
            };
            return m;
        };
        t.methods.push_back(index_method("perfect", "whether the cone pairing in H_c degree t is perfect "
                                                    "(golden pairing_perfect^t)",
                                         3, [](const ObjectPtr& o, uint32_t i) {
                                             return Value::from_bool(Pair(o).pd->perfect(i));
                                         }));
        t.methods.push_back(index_method("gram", "G^t: dim H^t_c x dim H^{2-t}(D_n) (golden pairing_matrix^t)",
                                         3, [](const ObjectPtr& o, uint32_t i) {
                                             return MatrixValueFrom(o, Pair(o).pd->gram(i));
                                         }));
        t.methods.push_back(index_method("boundary_perfect", "whether the circle pairing in degree s is perfect",
                                         2, [](const ObjectPtr& o, uint32_t i) {
                                             return Value::from_bool(Pair(o).pd->boundary_perfect(i));
                                         }));
        t.methods.push_back(index_method("boundary_gram", "the circle Gram in degree s", 2,
                                         [](const ObjectPtr& o, uint32_t i) {
                                             return MatrixValueFrom(o, Pair(o).pd->boundary_gram(i));
                                         }));
        {
            TypeInfo::Method m;
            m.name = "pd_gate";
            m.params = {"primes"};
            m.required = 1;
            m.computed = true;
            m.doc = "{t0, t1, t2}: A_c^{(t)T} G^t == G^t A(D_n)^{(2-t)} for EVERY p in `primes` (factor 1; "
                    "t = 0 vacuous and true); at primes [2, 3, 5] it is the golden pd_adjoint^t. A TEST of "
                    "[C] cl:pd, which is open";
            m.call = [](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
                if (!args[0].is_list() || args[0].list_value().empty()) {
                    throw TypeError("Pairing.pd_gate(primes): expected a non-empty List of primes", pos);
                }
                std::vector<char> all(3, 1);
                for (const Value& pv : args[0].list_value()) {
                    const std::vector<char>& g = Pair(o).gate(PrimeArg(pv, pos, "Pairing.pd_gate(primes)"));
                    for (uint32_t t = 0; t < 3; ++t) all[t] = static_cast<char>(all[t] && g[t]);
                }
                Record r;
                for (uint32_t t = 0; t < 3; ++t) r.set("t" + std::to_string(t), Value::from_bool(all[t] != 0));
                return Value::from_record(std::move(r));
            };
            t.methods.push_back(std::move(m));
        }
        t.print = [](const ObjectPtr& o, std::int64_t) {
            const ContragredientVal& d = Pair(o).dv();
            std::ostringstream os;
            os << "Pairing H^t_c(Sym^" << d.n_sym << ") x H^{2-t}(D_" << d.n_sym << ") -> F_" << d.ell;
            if (!o->computed()) os << " (declared, not yet computed)";
            return os.str();
        };
        registry.register_type(std::move(t));
    }
}

} // namespace

void RegisterDualityBindings(Registry& registry) {
    RegisterTypes(registry);
    const char* g = "pipeline";
    registry.register_verb("dual", Sig({Param::object("L", kTypeLocalSys)}), &VerbDual,
                           "the contragredient D_n of a model local system (E7; lazy)", g);
    registry.register_verb("pairing",
                           Sig({Param::object("L", kTypeLocalSys),
                                Param::object("D", "ContragredientLocalSystem")}),
                           &VerbPairing,
                           "the Poincare-duality pairing between H_c(L) and H(D), D = dual(L) (E7; lazy)", g);
}

} // namespace bind
} // namespace console
