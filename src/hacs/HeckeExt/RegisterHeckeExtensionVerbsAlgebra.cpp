// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// RegisterHeckeExtensionVerbsAlgebra.cpp -- one area of the extension layer's console
// bindings (S24 CL-8 split of RegisterHeckeExtensionVerbs.cpp; the map is in
// RegisterHeckeExtensionVerbsDetail.h): the local-algebra verbs ([P] Sec. 3.5's C5 rows), check_identities,
// L_map and hecke_ring_algebra.
// ===========================================================================

#include "RegisterHeckeExtensionVerbsDetail.h"

namespace hecke_ext {
namespace verbs_detail {

namespace {

// ===========================================================================
// The local-algebra verbs ([P] Sec. 3.5's C5 rows) on HeckeAlgebraInvariants.
// ===========================================================================

// `alg` is a List of commuting Matrix operators, the Record `algebra(ops)`
// returns (its `basis` spans the algebra), or a HeckeAlgebra handle (its t_p).
std::vector<ExtMatrix> ReadGenerators(const Value& alg, const SourcePos& pos, const char* who) {
    std::vector<ExtMatrix> gens;
    const std::vector<Value>* items = nullptr;
    if (alg.is_list()) {
        items = &alg.list_value();
    } else if (alg.is_record()) {
        const Value* b = alg.record_value().find("basis");
        if (b == nullptr || !b->is_list()) {
            throw TypeError(std::string(who) + ": a Record argument must carry the `basis` of "
                                               "algebra(ops)",
                            pos);
        }
        items = &b->list_value();
    } else if (const HeckeAlgebraVal* h = AsHeckeAlgebra(alg)) {
        return HeckeAlgebraGenerators(*h);
    } else {
        throw TypeError(std::string(who) + ": expected a list of commuting operators, the Record "
                                           "of algebra(ops), or a HeckeAlgebra, got " +
                            alg.type_name(),
                        pos);
    }
    if (items->empty()) throw TypeError(std::string(who) + ": the operator list is empty", pos);
    for (const Value& v : *items) gens.push_back(console::bind::AsMatrix(v, pos, who).m.to_matrix());
    return gens;
}

AlgebraInvariants InvariantsOf(const Value& alg, const SourcePos& pos, const char* who) {
    const std::vector<ExtMatrix> gens = ReadGenerators(alg, pos, who);
    try {
        return ComputeAlgebraInvariants(gens, gens.front().p);
    } catch (const std::invalid_argument& e) {
        throw TypeError(std::string(who) + ": " + e.what(), pos);
    }
}

Value MatrixListValue(CallContext& ctx, const std::vector<ExtMatrix>& ms) {
    std::vector<Value> out;
    for (const ExtMatrix& m : ms) out.push_back(console::bind::MatrixValue(ctx, Flat(m)));
    return Value::from_list(std::move(out));
}

} // namespace

Value VerbNilpotentPartOne(CallContext& ctx, const std::vector<Value>& a) {
    static const char* kWho = "nilpotent_part(A)";
    const ExtMatrix t = console::bind::AsMatrix(a[0], ctx.pos, kWho).m.to_matrix();
    try {
        return console::bind::MatrixValue(ctx, Flat(JordanChevalleySplit(t).nilpotent));
    } catch (const std::invalid_argument& e) {
        throw TypeError(std::string(kWho) + ": " + e.what(), ctx.pos);
    }
}

Value VerbNilpotentPartList(CallContext& ctx, const std::vector<Value>& a) {
    static const char* kWho = "nilpotent_part(ops)";
    const std::vector<ExtMatrix> gens = ReadGenerators(a[0], ctx.pos, kWho);
    // The family must commute: the nilpotent parts of a COMMUTING family are
    // the generators of its radical ideal ([E] alg:stage3 (2)).
    try {
        (void)ComputeAlgebraInvariants(gens, gens.front().p);
    } catch (const std::invalid_argument& e) {
        throw TypeError(std::string(kWho) + ": " + e.what(), ctx.pos);
    }
    std::vector<ExtMatrix> ns;
    for (const ExtMatrix& t : gens) ns.push_back(JordanChevalleySplit(t).nilpotent);
    return MatrixListValue(ctx, ns);
}

Value VerbRadical(CallContext& ctx, const std::vector<Value>& a) {
    const AlgebraInvariants A = InvariantsOf(a[0], ctx.pos, "radical(alg)");
    Record r;
    r.set("dim", Value::from_int(A.dim_rad));
    r.set("basis", MatrixListValue(ctx, A.radical));
    return Value::from_record(std::move(r));
}

Value VerbTangentSpace(CallContext& ctx, const std::vector<Value>& a) {
    const AlgebraInvariants A = InvariantsOf(a[0], ctx.pos, "tangent_space(alg)");
    Record r;
    r.set("dim", Value::from_int(A.dim_tangent));
    r.set("local", Value::from_bool(A.local));
    r.set("f", Value::from_int(A.residue_degree));
    r.set("embdim", A.local ? Value::from_int(A.dim_tangent / A.residue_degree) : Value::nil());
    r.set("basis", MatrixListValue(ctx, A.tangent));
    return Value::from_record(std::move(r));
}

Value VerbSocle(CallContext& ctx, const std::vector<Value>& a) {
    const AlgebraInvariants A = InvariantsOf(a[0], ctx.pos, "socle(alg)");
    Record r;
    r.set("dim", Value::from_int(A.dim_socle));
    r.set("local", Value::from_bool(A.local));
    r.set("f", Value::from_int(A.residue_degree));
    r.set("gorenstein", Value::from_bool(A.gorenstein));
    r.set("basis", MatrixListValue(ctx, A.socle));
    return Value::from_record(std::move(r));
}

Value VerbNilpotencyIndex(CallContext& ctx, const std::vector<Value>& a) {
    return Value::from_int(InvariantsOf(a[0], ctx.pos, "nilpotency_index(alg)").nilpotency_index);
}

Value VerbIsLocal(CallContext& ctx, const std::vector<Value>& a) {
    return Value::from_bool(InvariantsOf(a[0], ctx.pos, "is_local(alg)").local);
}

Value InvariantsRecord(const AlgebraInvariants& A) {
    Record r;
    r.set("ell", Value::from_int(A.ell));
    r.set("ambient_dim", Value::from_int(A.D));
    r.set("dim", Value::from_int(A.dim_A));
    r.set("dim_radical", Value::from_int(A.dim_rad));
    r.set("components", Value::from_int(A.components));
    r.set("local", Value::from_bool(A.local));
    r.set("residue_degree", Value::from_int(A.residue_degree));
    r.set("radical_power_dims", U32List(A.radical_power_dims));
    r.set("nilpotency_index", Value::from_int(A.nilpotency_index));
    r.set("dim_tangent", Value::from_int(A.dim_tangent));
    r.set("embdim", A.local ? Value::from_int(A.dim_tangent / A.residue_degree) : Value::nil());
    r.set("dim_socle", Value::from_int(A.dim_socle));
    r.set("gorenstein", Value::from_bool(A.gorenstein));
    r.set("hilbert", A.local ? U32List(A.hilbert) : Value::nil());
    return Value::from_record(std::move(r));
}

Value VerbAlgebraInvariants(CallContext& ctx, const std::vector<Value>& a) {
    return InvariantsRecord(InvariantsOf(a[0], ctx.pos, "algebra_invariants(alg)"));
}

namespace {

Value SeriesRecord(const AlgebraInvariants& A, const ModuleSeries& s) {
    Record r;
    r.set("dims", U32List(s.dims));
    r.set("layers", U32List(s.layers));
    r.set("f", Value::from_int(A.residue_degree));
    if (A.local) {
        std::vector<uint32_t> over;
        for (uint32_t x : s.layers) over.push_back(x / A.residue_degree);
        r.set("layers_over_f", U32List(over));
    } else {
        r.set("layers_over_f", Value::nil());
    }
    r.set("length", Value::from_int(static_cast<std::int64_t>(s.layers.size())));
    return Value::from_record(std::move(r));
}

ExtMatrix SubspaceArg(const std::vector<Value>& a, const AlgebraInvariants& A, const SourcePos& pos,
                      const char* who) {
    if (a.size() > 1 && !a[1].is_nil()) {
        const ExtMatrix V = console::bind::AsMatrix(a[1], pos, who).m.to_matrix();
        if (V.p != A.ell) throw TypeError(std::string(who) + ": V is not over F_ell", pos);
        return V;
    }
    return ExtMatrix(A.D, A.ell);   // the whole space
}

} // namespace

Value VerbLoewy(CallContext& ctx, const std::vector<Value>& a) {
    static const char* kWho = "loewy(alg, V)";
    const AlgebraInvariants A = InvariantsOf(a[0], ctx.pos, kWho);
    try {
        return SeriesRecord(A, LoewySeries(A, SubspaceArg(a, A, ctx.pos, kWho)));
    } catch (const std::invalid_argument& e) {
        throw TypeError(std::string(kWho) + ": " + e.what(), ctx.pos);
    }
}

Value VerbSocleSeries(CallContext& ctx, const std::vector<Value>& a) {
    static const char* kWho = "socle_series(alg, V)";
    const AlgebraInvariants A = InvariantsOf(a[0], ctx.pos, kWho);
    try {
        return SeriesRecord(A, SocleSeries(A, SubspaceArg(a, A, ctx.pos, kWho)));
    } catch (const std::invalid_argument& e) {
        throw TypeError(std::string(kWho) + ": " + e.what(), ctx.pos);
    }
}

// ---------------------------------------------------------------------------
// check_identities(H [, primes [, items]]) -- [P] Sec. 3.5's C5 row, on
// HeckeExt/HeckeIdentityCertificate: every item the certificate defines, in
// its own verify() order, each `passed` / `failed` / `not_applicable` as the
// certificate reports it, or `unavailable` where this binding did not hand the
// certificate its inputs (items 6 and 7 are assembled on request only).
// ---------------------------------------------------------------------------
Value VerbCheckIdentities(CallContext& ctx, const std::vector<Value>& a) {
    static const char* kWho = "check_identities(H, primes, items)";
    RequireField("check_identities");
    const ObjectPtr& H = console::bind::AsObjectOf(a[0], console::bind::kTypeCohomology, ctx.pos, kWho);
    // (slot-9 WP-12 x WP-13) a Z/ell^k handle is refused here, exit 5, not on a field accessor.
    console::bind::RequireFieldHandle(H, "console extension verb check_identities");
    uint64_t p = 2, p_prime = 3;
    if (a.size() > 1 && !a[1].is_nil()) {
        if (!a[1].is_list() || a[1].list_value().size() != 2) {
            throw TypeError(std::string(kWho) + ": `primes` is a list of two distinct good primes",
                            ctx.pos);
        }
        p = console::bind::U64Arg(a[1].list_value()[0], ctx.pos, kWho);
        p_prime = console::bind::U64Arg(a[1].list_value()[1], ctx.pos, kWho);
    }
    std::vector<bool> want(kIdentityItems + 1, false);
    if (a.size() > 2 && !a[2].is_nil()) {
        if (!a[2].is_list()) throw TypeError(std::string(kWho) + ": `items` is a list of Int", ctx.pos);
        for (const Value& v : a[2].list_value()) {
            const uint32_t k = console::bind::U32Arg(v, ctx.pos, kWho);
            if (k < 1 || k > kIdentityItems) {
                throw TypeError(std::string(kWho) + ": the certificate defines items 1.." +
                                    std::to_string(kIdentityItems),
                                ctx.pos);
            }
            want[k] = true;
        }
    }
    const CohSource s = SourceOf(H, ctx.pos, kWho);
    IdentityAssembly as;
    try {
        as = AssembleIdentityCertificate(*s.coh, *s.K, *s.model, s.n, s.ell, p, p_prime, want);
    } catch (const std::invalid_argument& e) {
        throw TypeError(std::string(kWho) + ": " + e.what(), ctx.pos);
    }
    Record r;
    r.set("n", Value::from_int(s.n));
    r.set("ell", Value::from_int(s.ell));
    r.set("p", Value::from_int(static_cast<std::int64_t>(p)));
    r.set("p_prime", Value::from_int(static_cast<std::int64_t>(p_prime)));
    r.set("code", Value::from_int(as.code));
    std::vector<Value> items;
    uint32_t n_pass = 0, n_fail = 0, n_na = 0, n_unav = 0;
    uint32_t k = 0;
    for (const auto& it : IdentityItems(as.cert)) {
        ++k;
        Record ir;
        ir.set("item", Value::from_int(k));
        ir.set("name", Value::from_string(it.first));
        std::string status;
        std::string detail;
        if (!as.supplied[k]) {
            status = "unavailable";
            detail = as.why[k];
            ++n_unav;
        } else {
            status = ToString(it.second->status);
            detail = it.second->detail;
            if (it.second->status == IdentityStatus::kPassed) ++n_pass;
            else if (it.second->status == IdentityStatus::kFailed) ++n_fail;
            else ++n_na;
        }
        ir.set("status", Value::from_string(status));
        ir.set("detail", Value::from_string(detail));
        items.push_back(Value::from_record(std::move(ir)));
    }
    r.set("items", Value::from_list(std::move(items)));
    r.set("k_max", Value::from_int(k));
    r.set("passed", Value::from_int(n_pass));
    r.set("failed", Value::from_int(n_fail));
    r.set("not_applicable", Value::from_int(n_na));
    r.set("unavailable", Value::from_int(n_unav));
    return Value::from_record(std::move(r));
}

// ---------------------------------------------------------------------------
// L_map(H_lower, H) -- the Dickson map L_* : H^*(Ga; M_m) -> H^*(Ga; M_n),
// m = n - ell - 1 ([E] identity 6's `L.` half, [I G12];
// HeckeExt/HeckeDicksonMultiplication).
// ---------------------------------------------------------------------------
Value VerbLMap(CallContext& ctx, const std::vector<Value>& a) {
    static const char* kWho = "L_map(H_lower, H)";
    RequireField("L_map");
    const ObjectPtr& Hl = console::bind::AsObjectOf(a[0], console::bind::kTypeCohomology, ctx.pos, kWho);
    const ObjectPtr& H = console::bind::AsObjectOf(a[1], console::bind::kTypeCohomology, ctx.pos, kWho);
    // (slot-9 WP-12 x WP-13) a Z/ell^k handle is refused here, exit 5, not on a field accessor.
    console::bind::RequireFieldHandle(Hl, "console extension verb L_map");
    console::bind::RequireFieldHandle(H, "console extension verb L_map");
    const CohSource lo = SourceOf(Hl, ctx.pos, kWho);
    const CohSource hi = SourceOf(H, ctx.pos, kWho);
    if (lo.ell != hi.ell) throw TypeError(std::string(kWho) + ": the two cohomologies differ in ell", ctx.pos);
    const uint32_t ell = hi.ell, n = hi.n;
    if (!DicksonMultiplicationApplicable(n, ell) || lo.n != DicksonSourceDegree(n, ell)) {
        throw TypeError(std::string(kWho) + ": needs n >= ell + 1 and H_lower at m = n - ell - 1 "
                                            "(here n = " + std::to_string(n) + ", m = " +
                            std::to_string(lo.n) + ", ell = " + std::to_string(ell) + ")",
                        ctx.pos);
    }
    const DicksonMultiplicationCertificate cert = CertifyDicksonMultiplication(n, ell);
    const std::vector<FlatMatrix> induced = DicksonInducedMaps(*hi.coh, *lo.coh, *hi.K, n, ell);
    Record r;
    r.set("n", Value::from_int(n));
    r.set("m", Value::from_int(lo.n));
    r.set("ell", Value::from_int(ell));
    r.set("L", console::bind::MatrixValue(ctx, Flat(cert.L)));
    r.set("certificate", Value::from_int(cert.verify()));
    std::vector<Value> maps;
    std::vector<uint32_t> ranks;
    for (const FlatMatrix& f : induced) {
        maps.push_back(console::bind::MatrixValue(ctx, f));
        Flat c(f);
        ranks.push_back(c.rank());
    }
    r.set("induced", Value::from_list(std::move(maps)));
    r.set("ranks", U32List(ranks));
    return Value::from_record(std::move(r));
}

// ---------------------------------------------------------------------------
// (slot-16 WP-6.)  hecke_ring_algebra(n, ell, kmax = 3): the Z/ell^K Hecke-algebra
// route of HeckeExt/HeckeRingHeckeAlgebra (slot-14 WP-6, certified verdict slot-15
// WP-5; [C] cor:stablecert), one eager Record.  Binding only: it calls
// ringalg::ComputeRingLoewyCell and reads its LoewyBlocks.
//   {schema, n, ell, kmax, primes, e_max, K_cls, K_run, Lv_run, level_mode, blocks, lines}  (schema = "hecke_ring_algebra/v1", slot-24 NB-RENDER R3)
// ((slot-23 G1, [C, obl:precisionreuse](vi)) K_run = the levels READ, the list it always had; Lv_run = the levels at
// which the module route was RUN, the levels below the latest run being derived, [C, lem:precisionreduction];
// level_mode (G1 audit F5) = the cell's level route, "reuse:<policy>" or "per_K:<reason>" -- a fallback is visible)
// per block {system, D, free_rank, torsion, rank_T, A1_hilbert, AK_hilbert,
// K_stab, eps, verdict, stable_cert, stable, cf9, hidden}; `lines` are the
// LOEWY_RING lines.  Errors: a non-default coefficient_ring exit 5 (dispatch);
// ell < 5, odd n, n < 1 exit 3 (ENGINE_CHECK); a non-prime ell or kmax outside
// 2..12 TypeError exit 3; ell^kmax >= 2^64 (the digit route) exit 5.
// ---------------------------------------------------------------------------
Value VerbHeckeRingAlgebra(CallContext& ctx, const std::vector<Value>& a) {
    static const char* kWho = "hecke_ring_algebra(n, ell, kmax)";
    RequireField("hecke_ring_algebra");
    const uint32_t n = console::bind::U32Arg(a[0], ctx.pos, kWho);
    const uint32_t ell = console::bind::U32Arg(a[1], ctx.pos, kWho);
    const uint32_t kmax = console::bind::U32Arg(a[2], ctx.pos, kWho);
    if (ell < 2 || !hecke::IsPrimeU64(ell)) throw TypeError(std::string(kWho) + ": ell must be a prime", ctx.pos);
    if (kmax < 2 || kmax > 12) throw TypeError(std::string(kWho) + ": kmax must lie in 2..12", ctx.pos);
    ENGINE_CHECK_MSG(ell >= 5, "hecke_ring_algebra(n, ell, kmax) requires ell >= 5 ([C] lem:h2torsion; the "
                               "Z/ell^K module route is the field route's lift at ell >= 5)");
    ENGINE_CHECK_MSG(n >= 1 && n % 2 == 0, "hecke_ring_algebra(n, ell, kmax) requires n >= 1 even (odd n vanishes on this model)");
    // slot-17 WP-8 (W6-2): the exact boundary -- ell^k fits uint64_t iff the u128 power stays <= 2^64 - 1 (no floating log2 sum).
    auto pow_fits = [](uint32_t base, uint32_t k) {
        unsigned __int128 r = 1;
        for (uint32_t i = 0; i < k; ++i) {
            r *= base;
            if (r > static_cast<unsigned __int128>(~uint64_t(0))) return false;
        }
        return true;
    };
    if (!pow_fits(ell, kmax))
        throw console::NotImplementedError(std::string(kWho) + ": ell^kmax >= 2^64 is the digit route, not bound to this verb", ctx.pos);
    // slot-17 WP-8 (W6-1): the working level is max(kmax, K_cls), K_cls = max(2, e_max + 1) (the cheap torsion pre-pass,
    // [C, prop:torsionprepass]); if ell^that overflows the word the cell is the digit route, a designed exit 5 -- not an exit 3.
    // (slot-18 WP-5 (e)) the pre-pass runs once: this run is handed to ComputeRingLoewyCell below (LoewyOptions::prepass).
    const hecke::ordinary::TorsionPrepass prepass = hecke::ordinary::TorsionExponentPrepass(n, ell);
    {
        const uint32_t kcls = std::max<uint32_t>(2, prepass.e_max + 1);
        if (!pow_fits(ell, kcls))
            throw console::NotImplementedError(std::string(kWho) + ": ell^K_cls >= 2^64 (K_cls = " + std::to_string(kcls) +
                                                   ") is the digit route, not bound to this verb", ctx.pos);
    }
    // slot-17 WP-8 (R17-2 finding): the module route builds U_ell (ell is in the default prime list) with no `u_ell` gate; the
    // frozen RSS model (HeckeUEllBudget.h) predicted 45.8 GB at (608, 547) against 3.7 GB measured with the levers. Gate it the way
    // every other route does -- the `u_ell` setting in force decides, by the frozen model -- as a value-free REFUSAL (the verb does
    // not silently drop U_ell, which would change the record): `set u_ell = budget_mb=<M>` admits it.
    {
        const UEllVerdict uv = UEllDecision(n, ell, hecke::math_settings().u_ell);
        if (uv.reason == UEllReason::Off)
            throw console::NotImplementedError(std::string(kWho) + ": the module route builds U_ell, and the `u_ell` setting is off", ctx.pos);
        if (!uv.included)
            throw console::NotImplementedError(
                std::string(kWho) + ": U_ell at (n, ell) = (" + std::to_string(n) + ", " + std::to_string(ell) + ") is predicted " +
                    std::to_string(static_cast<uint64_t>(uv.predicted_mb)) + " MB by the frozen model " + uv.model_version +
                    ", over the `u_ell` budget of " + std::to_string(uv.budget_mb) + " MB (refused before the run; raise it with `set u_ell = budget_mb=<M>`)",
                ctx.pos);
    }
    ringalg::LoewyOptions opts;
    opts.kmax = kmax;
    opts.prepass = &prepass;
    ringalg::LoewyCell c;
    try {
        c = ringalg::ComputeRingLoewyCell(n, ell, opts);
    } catch (const hecke::NotImplementedInThisBuild& e) {
        throw console::NotImplementedError(std::string(kWho) + ": " + e.what(), ctx.pos);
    }
    auto u64list = [](const std::vector<uint64_t>& xs) {
        std::vector<Value> v;
        for (uint64_t x : xs) v.push_back(Value::from_int(static_cast<std::int64_t>(x)));
        return Value::from_list(std::move(v));
    };
    Record r;
    // (slot-24 NB-RENDER R3) the record's schema name, the key of its render hints ([P] Sec. 18.5); no value moves.
    r.set("schema", Value::from_string("hecke_ring_algebra/v1"));
    r.set("n", Value::from_int(c.n));
    r.set("ell", Value::from_int(static_cast<std::int64_t>(c.ell)));
    r.set("kmax", Value::from_int(c.kmax));
    r.set("primes", u64list(c.primes));
    r.set("e_max", Value::from_int(c.e_max));
    r.set("K_cls", Value::from_int(c.K_cls));
    r.set("K_run", U32List(c.K_run));     // (slot-23 G1) the levels read
    r.set("Lv_run", U32List(c.Lv_run));   // (slot-23 G1, [C, obl:precisionreuse](vi)) the levels run
    r.set("level_mode", Value::from_string(c.level_mode));   // (G1 audit F5) "reuse:<policy>" | "per_K:<reason>"
    std::vector<Value> blocks, lines;
    for (const ringalg::LoewyBlock& b : c.blocks) {
        Record br;
        br.set("system", u64list(b.system));
        br.set("D", Value::from_int(b.D));
        br.set("free_rank", Value::from_int(b.free_rank));
        br.set("torsion", U32List(b.torsion));
        br.set("rank_T", Value::from_int(b.rank_T));
        br.set("A1_hilbert", U32List(b.A1.hilbert));
        const ringalg::AlgebraReading* T = b.last_T();
        br.set("AK_hilbert", T ? U32List(T->hilbert) : Value::nil());
        br.set("K_stab", Value::from_int(b.K_stab));
        br.set("eps", Value::from_int(b.eps));
        br.set("verdict", Value::from_string(b.verdict));
        br.set("stable_cert", Value::from_string(b.stable_cert));
        br.set("stable", Value::from_bool(b.stable_family()));
        br.set("cf9", Value::from_string(!b.nonreduced_A1() ? "trivial" : (b.cf9_holds() ? "held" : "FAIL")));
        br.set("hidden", Value::from_bool(b.hidden()));
        blocks.push_back(Value::from_record(std::move(br)));
        lines.push_back(Value::from_string(ringalg::LoewyRingLine(c, b)));
    }
    r.set("blocks", Value::from_list(std::move(blocks)));
    r.set("lines", Value::from_list(std::move(lines)));
    return Value::from_record(std::move(r));
}

} // namespace verbs_detail
} // namespace hecke_ext
