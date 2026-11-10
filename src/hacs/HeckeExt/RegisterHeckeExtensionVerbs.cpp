// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// RegisterHeckeExtensionVerbs.cpp -- see RegisterHeckeExtensionVerbs.h.
//
// (S24 CL-8, CODEAUDIT A10) This TU is split by area: it keeps the two public
// entry points, RegisterC5 (every C5 verb row, in its old order) and the C5
// helpers; the areas live in the six RegisterHeckeExtensionVerbs<Area>.cpp
// siblings, which share the private RegisterHeckeExtensionVerbsDetail.h (the
// map is there).  These eight files are the ONLY files in HeckeExt/ that
// include a Console/ header.  Both guards carry exactly these paths as rule-2
// exceptions, with the reason "[OD] slot-3 close-out 1":
//   * docs/matrix-engine/tools/check_heckeext_includes.sh -- the
//     ALLOW_EXT_CONSOLE array admitting Console/ includes ONLY (Cuda/, Engine/
//     and BlockFlatMatrix.h stay banned in these files too, and rules 1, 3 and
//     4 are untouched);
//   * docs/matrix-engine/tools/check_console_includes.sh -- one ALLOW_CONSOLE
//     line each.  Rule 1 (ConsoleCore is STL-only) is untouched: these TUs are
//     compiled into cohomology_app, never into the library.
//
// WHAT IT BINDS (phase C1b, slot-4 WP-5 item 6):
//   sturm_bound(k, ell [, cap])   HeckeExt/HeckeSturmBound -- a `Sturm` handle
//                                 whose state / resume / save_state run on the
//                                 REAL SturmState (C1a's stub arm stays as a
//                                 unit test of the mechanism).
//   u_ell_decision(n, ell [, budget_mb])
//                                 HeckeExt/HeckeUEllBudget's pure verdict.
//
// ADDED IN SLOT 5 (WP-10 item 1; [WP4-5] Sec. 5 item 1, the one C1b deferral):
//   bernoulli_mod(w, ell)         HeckeExt/HeckePredictions -- B_w in F_ell by
//                                 the F_ell recursion, refused (by the named
//                                 BernoulliOutOfRange) at w >= ell - 1.
//   glover_factors(n, ell)        HeckeExt/GloverFactors -- the Glover /
//                                 Ash-Stevens composition factors of
//                                 Sym^n(F_ell^2) as a Record of (sym,
//                                 det_twist, multiplicity) triples.
// C1b deferred both only because slot-4 WP-7 had not merged when it froze its
// transcripts; the two units shipped in that package and nothing about them
// moved.
//
// ADDED IN SLOT 9 (WP-13, console phase C5b; [P] Sec. 10 C5): the extension-
// layer bindings -- the ExtRecord and its MaximalIdeal / HeckeAlgebra /
// LoewyProfile / ExtensionClass / Presentation objects, Atlas, GF / GFElt,
// Settings, Involution, BoundaryCohomology, ConeCohomology, L_map,
// check_identities and the local-algebra verbs.  See the "C5b" section below;
// the pipeline plumbing is HeckeExt/HeckeExtBindingSupport and the
// certificate-free local-algebra route HeckeExt/HeckeAlgebraInvariants, both
// console-free.  E7's and E8's console types are C5a's (slot-9 WP-12), not
// this TU's.
// ===========================================================================

#include "RegisterHeckeExtensionVerbs.h"
#include "RegisterHeckeExtensionVerbsDetail.h"
#include "RegisterHeckeZRecordVerbs.h"

namespace hecke_ext {
namespace verbs_detail {

// ===========================================================================
// (slot-9 WP-13, console phase C5b) THE EXTENSION-LAYER BINDINGS.
//
// [P] Sec. 10 C5's goal list as far as the engine's API exists at the package
// base (docs/implementation/slot9/WP-13_report.md Sec. 1 is the inventory:
// each listed name -> the engine type it binds, or "not landed").  Everything
// here is a TRANSLATION: the pipeline plumbing lives in the console-free
// HeckeExt/HeckeExtBindingSupport and the new local-algebra route in
// HeckeExt/HeckeAlgebraInvariants, so this TU (the one HeckeExt/ file admitted
// to include Console/ headers) stays a thin layer of payloads, getters and
// printers.
//
// Types:  ExtRecord, MaximalIdeal, HeckeAlgebra, LoewyProfile, ExtensionClass,
//         Presentation, Atlas, GF, GFElt, Settings, Involution,
//         BoundaryCohomology, ConeCohomology.
// Verbs:  ext_record, local_algebra, module_structure, presentation, atlas,
//         GF / canonical_field, gf_* arithmetic, math_settings, iota,
//         boundary, cone, L_map, check_identities, and the local-algebra
//         verbs nilpotent_part / radical / tangent_space / socle /
//         nilpotency_index / loewy / socle_series / is_local /
//         algebra_invariants.
//
// RING.  The extension layer computes over F_ell only.  Every verb that runs
// the pipeline refuses a non-default `coefficient_ring` AT DISPATCH and again
// inside its lazy compute (hecke::RequireFieldCoefficientRing -> exit 5 through
// the C7 seam / bind::LazyCompute), exactly as C1a's pipeline verbs do; the
// pure linear-algebra verbs (on Matrix values) and GF do not read the setting.
// Every complex type reports `.coefficient_ring == "F_ell"`.
//
// NO GOLDEN CLAIM.  None of these types defines a `hash` that claims equality
// with a golden object: the ext record's golden lines go through the golden
// harness's file-local normalisation (Golden/GoldenHarness.cpp, WP-4's file),
// which is not public.  ExtRecord therefore offers `.digest` (FNV-1a-64 of the
// record text with `commit` redacted) and no `hash`; the value-like types
// (GFElt, Settings) hash their own canonical text.
// ===========================================================================

std::mutex& TypesMutex() {
    static std::mutex m;
    return m;
}
std::map<const TypeInfo*, std::shared_ptr<const C5Types>>& TypesMap() {
    static std::map<const TypeInfo*, std::shared_ptr<const C5Types>> m;
    return m;
}
const C5Types& TypesOf(const ObjectPtr& owner) {
    CONSOLE_CHECK(owner != nullptr && owner->type() != nullptr);
    std::lock_guard<std::mutex> lock(TypesMutex());
    const auto it = TypesMap().find(owner->type());
    CONSOLE_CHECK_MSG(it != TypesMap().end(), "a C5b handle whose type was not registered by RegisterC5");
    return *it->second;
}

Value U32List(const std::vector<uint32_t>& xs) { return console::bind::IntList(xs); }

Value U64List(const std::vector<uint64_t>& xs) {
    std::vector<Value> out;
    for (uint64_t x : xs) out.push_back(Value::from_int(static_cast<std::int64_t>(x)));
    return Value::from_list(std::move(out));
}

Value StringList(const std::vector<std::string>& xs) {
    std::vector<Value> out;
    for (const std::string& x : xs) out.push_back(Value::from_string(x));
    return Value::from_list(std::move(out));
}

// A matrix as a console Matrix value owned by `owner` (a member getter).
Value MatrixFrom(const ObjectPtr& owner, const ExtMatrix& m) {
    auto p = std::make_shared<console::bind::MatrixVal>();
    p->m = Flat(m);
    ObjectPtr o = console::bind::MakeValueObjectFrom(TypesOf(owner).matrix, owner, std::move(p));
    o->keep_alive(owner);
    return Value::from_object(o);
}
Value FlatFrom(const ObjectPtr& owner, const Flat& m) {
    auto p = std::make_shared<console::bind::MatrixVal>();
    p->m = m;
    ObjectPtr o = console::bind::MakeValueObjectFrom(TypesOf(owner).matrix, owner, std::move(p));
    o->keep_alive(owner);
    return Value::from_object(o);
}
Value PolyFrom(const ObjectPtr& owner, const hecke::FpPoly& f) {
    auto p = std::make_shared<console::bind::PolyVal>();
    p->f = f;
    ObjectPtr o = console::bind::MakeValueObjectFrom(TypesOf(owner).poly, owner, std::move(p));
    o->keep_alive(owner);
    return Value::from_object(o);
}
Value MatrixListFrom(const ObjectPtr& owner, const std::vector<ExtMatrix>& ms) {
    std::vector<Value> out;
    for (const ExtMatrix& m : ms) out.push_back(MatrixFrom(owner, m));
    return Value::from_list(std::move(out));
}
// A child handle of `type` over `payload`, born computed, keeping `owner` alive.
ObjectPtr ChildOf(const TypeInfo* type, const ObjectPtr& owner, std::shared_ptr<void> payload) {
    ObjectPtr o = console::bind::MakeValueObjectFrom(type, owner, std::move(payload));
    o->keep_alive(owner);
    return o;
}

uint32_t DegreeArgC5(const Value& v, const SourcePos& pos, const char* who, size_t n) {
    const uint32_t q = console::bind::U32Arg(v, pos, who);
    if (q >= n) {
        throw RuntimeError(std::string(who) + ": degree " + std::to_string(q) +
                               " is not a degree of this complex (it has " + std::to_string(n) +
                               ")",
                           pos);
    }
    return q;
}

void RequireField(const char* who) {
    hecke::RequireFieldCoefficientRing(std::string("console extension verb ") + who);
}

CohSource SourceOf(const ObjectPtr& H, const SourcePos& pos, const char* who) {
    console::bind::CohomologyVal* cv = H->payload_as<console::bind::CohomologyVal>();
    CONSOLE_CHECK(cv != nullptr && cv->L != nullptr);
    const console::bind::LocalSystemVal* lv = cv->L->payload_as<console::bind::LocalSystemVal>();
    if (!lv->preset.empty() || lv->model == nullptr) {
        throw TypeError(std::string(who) + ": needs a cohomology on the SL(2,Z) model; the preset "
                                           "local system '" + lv->preset +
                            "' carries no tessellation model",
                        pos);
    }
    cv->ensure();
    const console::bind::ModelVal& mv = console::bind::ModelOf(*cv);
    CohSource s;
    s.coh = cv->coh.get();
    s.K = &mv.K;
    s.model = mv.model.get();
    s.ell = cv->coh->get_prime();
    s.n = cv->coh->get_representation_dim() - 1;
    return s;
}

namespace {

// ===========================================================================
// Registration.
// ===========================================================================
void RegisterC5(Registry& registry) {
    using console::bind::Sig;
    RegisterValueTypes(registry);
    RegisterExtRecordTypes(registry);
    RegisterComplexTypes(registry);

    auto Tp = std::make_shared<C5Types>();
    C5Types& T = *Tp;
    T.matrix = registry.find_type(console::bind::kTypeMatrix);
    T.poly = registry.find_type(console::bind::kTypePoly);
    T.ideal = registry.find_type(kTypeMaximalIdeal);
    T.algebra = registry.find_type(kTypeHeckeAlgebra);
    T.loewy = registry.find_type(kTypeLoewyProfile);
    T.extension = registry.find_type(kTypeExtension);
    T.presentation = registry.find_type(kTypePresentation);
    T.gf = registry.find_type(kTypeGF);
    T.gfelt = registry.find_type(kTypeGFElt);
    T.settings = registry.find_type(kTypeSettings);
    CONSOLE_CHECK_MSG(T.matrix != nullptr && T.poly != nullptr,
                      "RegisterHeckeExtensionVerbs must run after RegisterPipelineVerbs");
    {
        std::lock_guard<std::mutex> lock(TypesMutex());
        for (const char* name : {kTypeExtRecord, kTypeMaximalIdeal, kTypeHeckeAlgebra, kTypeLoewyProfile,
                                 kTypeExtension, kTypePresentation, kTypeAtlas, kTypeGF, kTypeGFElt,
                                 kTypeSettings, kTypeInvolution, kTypeBoundaryCoh, kTypeConeCoh,
                                 kTypeDicksonLES}) {
            const TypeInfo* ti = registry.find_type(name);
            CONSOLE_CHECK(ti != nullptr);
            TypesMap()[ti] = Tp;
        }
    }

    const char* g = "extension";
    const Value nil = Value::nil();
    auto obj = [](const char* n, const char* type) { return Param::object(n, type); };

    // --- the local-algebra verbs ([P] Sec. 3.5, C5 rows) ---------------------
    registry.register_verb("nilpotent_part", Sig({obj("A", console::bind::kTypeMatrix)}),
                           &VerbNilpotentPartOne,
                           "the nilpotent part n of A = s + n (Jordan-Chevalley over F_ell: "
                           "s = A^(ell^N))", g);
    registry.register_verb("nilpotent_part", Sig({Param::of("ops", ValueKind::List)}),
                           &VerbNilpotentPartList,
                           "the nilpotent parts of a COMMUTING family (the generators of its "
                           "radical ideal)", g);
    registry.register_verb("radical", Sig({Param::any("alg")}), &VerbRadical,
                           "rad A = the nilpotent elements of A = F_ell[ops] (ker Frob^e), "
                           "{dim, basis}; `alg` is a list of commuting operators, the Record of "
                           "algebra(ops) or a HeckeAlgebra", g);
    registry.register_verb("tangent_space", Sig({Param::any("alg")}), &VerbTangentSpace,
                           "m/m^2 = rad/rad^2: {dim (over F_ell), local, f, embdim, basis}", g);
    registry.register_verb("socle", Sig({Param::any("alg")}), &VerbSocle,
                           "soc A = Ann_A(rad A): {dim, local, f, gorenstein, basis}", g);
    registry.register_verb("nilpotency_index", Sig({Param::any("alg")}), &VerbNilpotencyIndex,
                           "the least nu with rad^nu = 0 (1 iff A is reduced)", g);
    registry.register_verb("is_local", Sig({Param::any("alg")}), &VerbIsLocal,
                           "whether A has exactly one maximal ideal (Berlekamp: dim{x : x^ell - x "
                           "in rad} - dim rad == 1) -- decided, never searched", g);
    registry.register_verb("algebra_invariants", Sig({Param::any("alg")}), &VerbAlgebraInvariants,
                           "every invariant above in one Record", g);
    registry.register_verb("loewy",
                           Sig({Param::any("alg"), obj("V", console::bind::kTypeMatrix).optional(nil)}),
                           &VerbLoewy,
                           "the Loewy (radical) series rad^j V of an A-stable subspace V (columns; "
                           "the whole space when omitted): {dims, layers, f, layers_over_f, length}", g);
    registry.register_verb("socle_series",
                           Sig({Param::any("alg"), obj("V", console::bind::kTypeMatrix).optional(nil)}),
                           &VerbSocleSeries, "the socle series soc^j V, same shape as loewy", g);
    registry.register_verb(
        "check_identities",
        Sig({obj("H", console::bind::kTypeCohomology),
             Param::of("primes", ValueKind::List).optional(nil),
             Param::of("items", ValueKind::List).optional(nil)}),
        &VerbCheckIdentities,
        "every item HeckeIdentityCertificate defines, at H's cell, as {items: [{item, name, "
        "status, detail}], code, k_max, ...}; primes = [p, p'] (default [2, 3]); items 6 and 7 are "
        "assembled only when listed in `items`, and an item whose inputs were not assembled "
        "reports `unavailable`", g);
    registry.register_verb(
        "L_map",
        Sig({obj("H_lower", console::bind::kTypeCohomology), obj("H", console::bind::kTypeCohomology)}),
        &VerbLMap,
        "the Dickson map L_* : H^*(M_m) -> H^*(M_n), m = n - ell - 1: {L, certificate, induced, "
        "ranks}", g);

    // --- the Stage 1-5 objects ---------------------------------------------
    registry.register_verb(
        "ext_record",
        Sig({Param::of("n", ValueKind::Int), Param::of("ell", ValueKind::Int),
             Param::of("cone", ValueKind::Bool).optional(Value::from_bool(false)),
             Param::of("sign_split", ValueKind::Bool).optional(Value::from_bool(false)),
             Param::of("presentation", ValueKind::Bool).optional(Value::from_bool(false)),
             Param::of("sturm", ValueKind::Bool).optional(Value::from_bool(true)),
             Param::of("commutant", ValueKind::Bool).optional(Value::from_bool(true))}),
        &VerbExtRecord,
        "the cell's hecke_ext/v5 record (Stages 1-5) under the mathematical settings in force "
        "(lazy); cone / sign_split add the E4b-E5 / E3 objects, sturm / commutant select the "
        "Stage-4 routes", g);
    registry.register_verb("local_algebra", Sig({obj("m", kTypeMaximalIdeal)}), &VerbLocalAlgebra,
                           "the local Hecke algebra T_m of a certified-local maximal ideal", g);
    registry.register_verb("module_structure",
                           Sig({obj("m", kTypeMaximalIdeal),
                                Param::of("group", ValueKind::String).optional(nil)}),
                           &VerbModuleStructure,
                           "the LoewyProfile of m in `group` (default: m's own group)", g);
    registry.register_verb("presentation", Sig({obj("m", kTypeMaximalIdeal)}), &VerbPresentation,
                           "generators and a minimal relation set of H_m over T_m", g);
    registry.register_verb("atlas", Sig({Param::of("ell", ValueKind::Int)}), &VerbAtlas,
                           "Sigma(ell) and S_sep(ell), in process at ell in {5, 7, 11, 13} (lazy)", g);

    // --- the split field ------------------------------------------------------
    registry.register_verb("GF", Sig({Param::of("ell", ValueKind::Int), Param::of("r", ValueKind::Int)}),
                           &VerbGF, "the canonical field F_ell[Y]/(C_{ell,r}) of degree r", g);
    registry.register_verb("canonical_field",
                           Sig({Param::of("ell", ValueKind::Int), Param::of("r", ValueKind::Int)}),
                           &VerbGF, "the same as GF(ell, r) ([X] Sec. 7's name)", g);
    registry.register_verb("gf_add", Sig({obj("x", kTypeGFElt), obj("y", kTypeGFElt)}), &VerbGFAdd, "x + y in GF", g);
    registry.register_verb("gf_sub", Sig({obj("x", kTypeGFElt), obj("y", kTypeGFElt)}), &VerbGFSub, "x - y in GF", g);
    registry.register_verb("gf_mul", Sig({obj("x", kTypeGFElt), obj("y", kTypeGFElt)}), &VerbGFMul, "x * y in GF", g);
    registry.register_verb("gf_inv", Sig({obj("x", kTypeGFElt)}), &VerbGFInv, "x^-1 in GF (x != 0)", g);
    registry.register_verb("gf_pow", Sig({obj("x", kTypeGFElt), Param::of("e", ValueKind::Int)}), &VerbGFPow, "x^e in GF", g);
    registry.register_verb("gf_frobenius", Sig({obj("x", kTypeGFElt)}), &VerbGFFrobenius, "x^ell", g);
    registry.register_verb("gf_eval", Sig({obj("f", console::bind::kTypePoly), obj("x", kTypeGFElt)}),
                           &VerbGFEval, "f(x) for f over F_ell and x in GF", g);

    // --- settings and the complexes -------------------------------------------
    registry.register_verb("settings_snapshot", Sig({}), &VerbSettingsSnapshot,
                           "the mathematical settings in force, as a Settings value (math_settings() is the C1a Record form)", g);
    registry.register_verb("iota", Sig({obj("H", console::bind::kTypeCohomology)}), &VerbIota,
                           "the production involution iota^* on H and its +/- split (lazy)", g);
    registry.register_verb("boundary", Sig({obj("H", console::bind::kTypeCohomology)}), &VerbBoundary,
                           "H^*_bd on the boundary circle with r and T_p^bd (lazy)", g);
    registry.register_verb("cone", Sig({obj("H", console::bind::kTypeCohomology)}), &VerbCone,
                           "H^*_c = H^*(Cone(r)) with its LES and A_c(T_p) (lazy)", g);

    // --- theta and the Dickson long exact sequence (slot-12 WP-8; [P] v3.9 Sec. 3.6)
    registry.register_verb("dickson_les", Sig({Param::of("n", ValueKind::Int), Param::of("ell", ValueKind::Int)}),
                           &VerbDicksonLES,
                           "the long exact sequence of 0 -> Sym^n (x) det -> Sym^{n+ell+1} -> Q_n -> 0 over F_ell, "
                           "ell >= 5, as a DicksonLES handle (lazy)", g);
    registry.register_verb("hecke_ring_algebra",
                           Sig({Param::of("n", ValueKind::Int), Param::of("ell", ValueKind::Int),
                                Param::of("kmax", ValueKind::Int).optional(Value::from_int(3))}),
                           &VerbHeckeRingAlgebra,
                           "the Z/ell^K Hecke-algebra route ([C] cor:stablecert; HeckeRingHeckeAlgebra), ell >= 5, "
                           "n even, kmax = 3 by default, ell^max(kmax, K_cls) < 2^64, U_ell within the `u_ell` budget: {schema = hecke_ring_algebra/v1, n, ell, kmax, primes, e_max, K_cls, K_run, Lv_run, level_mode, blocks, lines} "
                           "(K_run = the levels read, Lv_run = the levels at which the module route ran, level_mode = reuse:<policy> | per_K:<reason>; "
                           "[C] obl:precisionreuse); per "
                           "block {system, D, free_rank, torsion, rank_T, A1_hilbert, AK_hilbert, K_stab, eps, "
                           "verdict, stable_cert, stable, cf9, hidden}; a ring exit 5, ell < 5 exit 3", g);
    registry.register_verb("theta",
                           Sig({obj("H", console::bind::kTypeCohomology), obj("H_up", console::bind::kTypeCohomology)}),
                           &VerbTheta,
                           "Dickson theta = L_* : H^q(M_n (x) det) -> H^q(M_{n+ell+1}), source-indexed (the same map "
                           "as L_map(H, H_up)): {n, n_up, ell, induced, ranks, ker, image, twist_check, u_ell_zero}", g);
    // --- the one-variable truncated ell-adic L-function (slot-20 stretch S1; [E] sec:lvlfunction; [P] v3.11.1 Sec. 3.6)
    registry.register_verb("lfunction",
                           Sig({Param::of("n", ValueKind::Int), Param::of("ell", ValueKind::Int), Param::of("a", ValueKind::Int),
                                Param::of("M", ValueKind::Int).optional(Value::from_int(2)),
                                Param::of("R", ValueKind::Int).optional(Value::from_int(4)),
                                Param::of("sign", ValueKind::String).optional(Value::from_string("minus")),
                                Param::of("sep", ValueKind::List).optional(Value::nil())}),
                           &VerbLFunction,
                           "the record lfunction/v1 of branch a: L_a(T) in Lambda/(ell^R, omega_{M-1}(T)) of the ell-stabilised MTT "
                           "measure ([E] def:lvmtt) of the ordinary eigensymbol picked by sep (List of [p, c]; nil = T_2+3T_3+7T_5; "
                           "the Eisenstein root if it occurs, else the first root mod ell), sign \"minus\" (r^-) or \"plus\" (r^+); "
                           "ell >= 5 prime, n even, M >= 2, ell^M <= 4*10^6: {schema, n, k, ell, R, M, G, u, sign, geometric, form, "
                           "normalisation, a, vanishes, mu, lambda, readable, lambda_mod_ell, coefficients, precision, values_modulus, "
                           "critical_values, hash, line} ([E] lem:lvlfunprec: prec(0) = R, prec(t) = min(R, M-1-floor(log_ell t)))", g);
    // --- slot-21 verbs package ([PG] sec:pg-missing)
    registry.register_verb("kummer_unit",
                           Sig({Param::of("ell", ValueKind::Int), Param::of("i", ValueKind::Int)}),
                           &VerbKummerUnit,
                           "the cyclotomic unit eta_i of layer 1 ([E] sec:lvnf) and its Kummer l-th-power test, ell prime, 5 <= ell <= 65521, "
                           "i even, 2 <= i <= ell - 3: {schema, ell, i, exponents, coefficients, max_digits, v_pi, unramified, lambda_split, "
                           "not_lth_power, eigen_ok, unit_ok, ell_divides_Bi, oracle_agrees, split_primes}", g);
    registry.register_verb("hida_tower",
                           Sig({Param::of("n", ValueKind::Int), Param::of("ell", ValueKind::Int), Param::of("m", ValueKind::Int)}),
                           &VerbHidaTower,
                           "the Hida weight-tower cell at class n ([C] def:hidatower), ell >= 5 prime, n even <= 200, m <= 3: both readings of "
                           "the ordinary part at levels e = 1 .. m, {schema, n, ell, m, e_max, K, h, exps, levels: [{e, ord_rank, image_free, "
                           "charpoly, free_ord_rank, free_charpoly, readings_agree}], lines}", g);

    // --- slot-22 WP-VERB: the remaining [PG] sec:pg-missing verbs (thin bindings; each refuses by name past its cost bound)
    registry.register_verb("level_n",
                           Sig({Param::of("N", ValueKind::Int), Param::of("n", ValueKind::Int), Param::of("ell", ValueKind::Int),
                                Param::of("primes", ValueKind::List).optional(Value::nil())}),
                           &VerbLevelN,
                           "H^1(Gamma_0(N); Sym^n F_ell) by Shapiro's lemma ([C] sec:levelN; the sparse cokernel of the induced module) "
                           "and the Heilbronn operators T_p (p not dividing N) / U_p (p | N) on it, primes = List of at most 6 primes "
                           "<= 97, none equal to ell (nil = 2, 3, 5 without ell); ell >= 5 prime, n even, N <= 100, index(N) * (n + 1) <= 1000: "
                           "{schema, N, n, ell, index, D, dim_S, dim_ST, h0, h1, cusps, dim_cusp_forms, ops: [{label, p, bad, charpoly}]}", g);
    registry.register_verb("nebentypus",
                           Sig({Param::of("N", ValueKind::Int), Param::of("m", ValueKind::Int), Param::of("n", ValueKind::Int),
                                Param::of("ell", ValueKind::Int), Param::of("primes", ValueKind::List).optional(Value::nil())}),
                           &VerbNebentypus,
                           "H^1(Gamma_0(N); Sym^n F_ell (x) chi) for the Dirichlet character of Conrey label N.m reduced into F_ell "
                           "([C] sec:nebentypus), and the Heilbronn operators on it; ell prime not dividing N with order(chi) | ell - 1, "
                           "n >= 0, N <= 100, index(N) * (n + 1) <= 1000; primes = List of at most 4 primes <= 13 (nil = 2, 3, 5 without ell): {schema, label, N, m, order, parity_ok, "
                           "n, ell, D, h0, h1, ops: [{label, p, bad, charpoly}]} (h0 = h1 = 0 and no ops when chi(-1) != (-1)^n)", g);
    registry.register_verb("forms_algebra",
                           Sig({Param::of("k", ValueKind::Int), Param::of("ell", ValueKind::Int),
                                Param::of("primes", ValueKind::List).optional(Value::nil()),
                                Param::of("terms", ValueKind::Int).optional(Value::from_int(8))}),
                           &VerbFormsAlgebra,
                           "the forms side mod ell: S_k over F_ell with its Miller q-expansion basis and T_p on it ([E] sec:forms; the "
                           "characteristic polynomials), ell >= 5 prime, k even, 12 <= k <= 400, primes (at most 6, <= 97, none equal to ell; "
                           "nil = 2, 3, 5 without ell), terms <= 64: {schema, k, ell, d, k_prime, precision, basis, ops: [{p, charpoly}]}", g);
    registry.register_verb("hida_lambda",
                           Sig({Param::of("ell", ValueKind::Int), Param::of("k", ValueKind::Int), Param::of("W", ValueKind::Int),
                                Param::of("primes", ValueKind::List).optional(Value::nil())}),
                           &VerbHidaLambda,
                           "the forms side of the Lambda-adic Hecke algebra at one weight ([U] sec:u-lambda): the ordinary parts e M_k and "
                           "e S_k over Z/ell^W and the characteristic polynomials of T_p and U = T_ell on e S_k; ell >= 5 prime <= 101, k even "
                           "with 4 <= k <= 200 and k - 1 >= W, W <= 6, ell^W < 2^62; primes as for forms_algebra (nil = 2, 3, 5 without ell): "
                           "{schema, ell, k, W, dim_M, dim_S, rank_M, rank_S, comm_ok, stable_ok, eisenstein, ops: [{name, charpoly}]}", g);
}


} // namespace
} // namespace verbs_detail

using namespace verbs_detail;

void RegisterHeckeExtensionVerbs(Registry& registry) {
    if (registry.find_type(kTypeSturm) != nullptr) return;  // idempotent per registry
    RegisterSturmTypes(registry);

    const char* group = "extension";
    registry.register_verb(
        "sturm_bound",
        console::bind::Sig({Param::of("k", ValueKind::Int), Param::of("ell", ValueKind::Int)}),
        &VerbSturmBound, "the computed Sturm bound B'(k, ell) at the default cap 12*B(k)", group);
    registry.register_verb(
        "sturm_bound",
        console::bind::Sig({Param::of("k", ValueKind::Int), Param::of("ell", ValueKind::Int),
                            Param::of("cap", ValueKind::Int).optional(Value::from_int(0))}),
        &VerbSturmBound, "the same with the cap pinned", group);
    registry.register_verb(
        "u_ell_decision",
        console::bind::Sig({Param::of("n", ValueKind::Int), Param::of("ell", ValueKind::Int)}),
        &VerbUEllDecision, "whether U_ell fits the `u_ell` budget in force, by the frozen RSS model",
        group);
    registry.register_verb(
        "u_ell_decision",
        console::bind::Sig({Param::of("n", ValueKind::Int), Param::of("ell", ValueKind::Int),
                            Param::of("budget_mb", ValueKind::Int)}),
        &VerbUEllDecision, "the same against an explicit budget in MB", group);
    // (slot-5 WP-10) The two E6 prediction units, C1b's one deferral.
    registry.register_verb(
        "bernoulli_mod",
        console::bind::Sig({Param::of("w", ValueKind::Int), Param::of("ell", ValueKind::Int)}),
        &VerbBernoulliMod,
        "B_w in F_ell by the F_ell recursion (von Staudt-Clausen: defined exactly for "
        "0 <= w < ell - 1, refused outside it)",
        group);
    registry.register_verb(
        "glover_factors",
        console::bind::Sig({Param::of("n", ValueKind::Int), Param::of("ell", ValueKind::Int)}),
        &VerbGloverFactors,
        "the Glover / Ash-Stevens composition factors Sym^{n_i} (x) det^{j_i} of Sym^n(F_ell^2) "
        "over F_ell[GL_2(F_ell)], with multiplicities, in the PIPELINE's convention: Sym^n of the "
        "STANDARD module (not its dual), the one rho_n acts on -- the dual would shift every "
        "det_twist by -n",
        group);
    // (slot-6 WP-13, console C2a) The `field_polynomial` setting's one consumer.
    registry.register_verb(
        "canonical_field_polynomial",
        console::bind::Sig({Param::of("ell", ValueKind::Int), Param::of("r", ValueKind::Int)}),
        &VerbCanonicalFieldPolynomial,
        "C_{ell,r}, the defining polynomial of the degree-r split field, as the "
        "`field_polynomial` setting in force selects it: `least` (the default) gives the "
        "CanonicalLess-least monic irreducible of [E] def:canon; `conway` is a legal value this "
        "build cannot execute (exit code 5)",
        group);
    // (slot-9 WP-13, console C5b) the extension-layer bindings and the
    // local-algebra verbs -- see the C5 section above.
    RegisterC5(registry);
    // (slot-24 D6) the integral descent record: the type DescentRecord and the verb `descent` (Console/Bindings/
    // Descent.h), last, so no existing overload moves and the read_only twin rows are the tail of both tables.
    integral_descent::RegisterDescentVerbs(registry);
    // (slot-25 HZ-REC) the record integral_hecke_structure/v1: the verb `hecke_structure` (HeckeExt/RegisterHeckeZRecordVerbs.h),
    // after `descent`, so no existing overload moves and the read_only twin rows stay the tail of both tables.
    heckez::record::RegisterHeckeZRecordVerbs(registry);
    // (slot-24 D6 audit m1; [C] sec:recordschema item 7 (a), [E] sec:ring (h)) The scalar-argument extension
    // verbs that compute over F_ell and never read the ring refuse `Z` and `Z/ell^k` at their statement
    // (exit 5), instead of returning an F_ell result under a snapshot naming the other ring.  The handle verbs
    // (iota, boundary, cone, theta, L_map, check_identities, ...) and ext_record / atlas / dickson_les /
    // hecke_ring_algebra refuse inside their own impls (RequireField) and are not listed; `descent` honours Z.
    console::bind::GateVerbsOnField(registry, {"sturm_bound", "u_ell_decision", "bernoulli_mod", "glover_factors",
                                               "lfunction", "kummer_unit", "hida_tower", "level_n", "nebentypus",
                                               "forms_algebra", "hida_lambda"});
}

void RegisterHeckeExtensionVerbs() { RegisterHeckeExtensionVerbs(console::Registry::global()); }

} // namespace hecke_ext
