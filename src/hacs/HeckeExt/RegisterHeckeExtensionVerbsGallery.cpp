// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// RegisterHeckeExtensionVerbsGallery.cpp -- one area of the extension layer's console
// bindings (S24 CL-8 split of RegisterHeckeExtensionVerbs.cpp; the map is in
// RegisterHeckeExtensionVerbsDetail.h): lfunction (slot-20 S1) and the [PG] sec:pg-missing verbs: kummer_unit,
// hida_tower, level_n, nebentypus, forms_algebra, hida_lambda.
// ===========================================================================

#include "RegisterHeckeExtensionVerbsDetail.h"
#include "../Console/Core/Cancel.h"    // (NBCONFIRM S5) PollCancel
#include "../Console/Core/Confirm.h"   // (NBCONFIRM S3) CostGate

namespace hecke_ext {
namespace verbs_detail {

// ---------------------------------------------------------------------------
// (slot-20 stretch S1; [E] sec:lvlfunction, [P] v3.11.1 Sec. 3.6)  lfunction(n, ell, a, M = 2, R = 4,
// sign = "minus", sep = nil): the record lfunction/v1 of branch a -- L_a(T) in Lambda/(ell^R, omega_{M-1}(T))
// of the ell-stabilised MTT measure of the ordinary eigensymbol picked by the separating operator `sep`
// (a List of [p, c] pairs; nil = T_2 + 3T_3 + 7T_5), the Eisenstein root if it occurs, else the first root
// mod ell.  One eager Record; binding only (LValue/LFunctionRecord: RunSymbolCell + BuildMeasure, the measure
// that BirchModularSymbolRoute::measure() returns).  Read-only: pure in-memory arithmetic, no I/O.  Refusals
// (ell, n, M, R, sign, ell^M > 4*10^6 balls, a non-simple root, non-ordinary) exit 3.
// ---------------------------------------------------------------------------
Value VerbLFunction(CallContext& ctx, const std::vector<Value>& a) {
    static const char* kWho = "lfunction(n, ell, a, M, R, sign, sep)";
    lvalue::LFunctionRequest rq;
    rq.n = console::bind::U32Arg(a[0], ctx.pos, kWho);
    rq.ell = console::bind::U64Arg(a[1], ctx.pos, kWho);
    const uint32_t br = console::bind::U32Arg(a[2], ctx.pos, kWho);
    if (a.size() > 3 && !a[3].is_nil()) rq.M = console::bind::U32Arg(a[3], ctx.pos, kWho);
    if (a.size() > 4 && !a[4].is_nil()) rq.R = console::bind::U32Arg(a[4], ctx.pos, kWho);
    if (a.size() > 5 && !a[5].is_nil()) {
        if (!a[5].is_string()) throw TypeError(std::string(kWho) + ": `sign` is the String \"minus\" or \"plus\"", ctx.pos);
        rq.sign = a[5].string_value();
    }
    if (a.size() > 6 && !a[6].is_nil()) {
        if (!a[6].is_list()) throw TypeError(std::string(kWho) + ": `sep` is a List of [p, c] pairs", ctx.pos);
        rq.sep.clear();
        for (const Value& pc : a[6].list_value()) {
            if (!pc.is_list() || pc.list_value().size() != 2)
                throw TypeError(std::string(kWho) + ": `sep` is a List of [p, c] pairs", ctx.pos);
            const uint64_t p = console::bind::U64Arg(pc.list_value()[0], ctx.pos, kWho);
            const uint64_t c = console::bind::U64Arg(pc.list_value()[1], ctx.pos, kWho);
            if (p < 2 || p > 97 || !hecke::IsPrimeU64(p)) throw TypeError(std::string(kWho) + ": `sep` primes must be primes <= 97", ctx.pos);
            rq.sep[p] = (rq.sep.count(p) ? rq.sep[p] : 0) + c;
        }
    }
    rq.branches = {br};
    return Guard(kWho, ctx.pos, [&] {
        const lvalue::LFunctionRecord rec = lvalue::ComputeLFunction(rq);
        const lvalue::LFunctionBranch& b = rec.branches.front();
        auto u64list = [](const std::vector<uint64_t>& xs) {
            std::vector<Value> v;
            for (uint64_t x : xs) v.push_back(Value::from_int(static_cast<std::int64_t>(x)));
            return Value::from_list(std::move(v));
        };
        Record form;
        form.set("separator", Value::from_string(rec.form.separator));
        form.set("target_root", Value::from_int(static_cast<std::int64_t>(rec.form.target_root)));
        form.set("eisenstein_congruent", Value::from_bool(rec.form.eisenstein_congruent));
        Record ap;
        for (const auto& kv : rec.form.a_p) ap.set(std::to_string(kv.first), Value::from_int(static_cast<std::int64_t>(kv.second)));
        form.set("a_p", Value::from_record(std::move(ap)));
        form.set("a_ell", Value::from_int(static_cast<std::int64_t>(rec.a_ell)));
        form.set("alpha", Value::from_int(static_cast<std::int64_t>(rec.alpha)));
        form.set("symbol", Value::from_string(rec.form.symbol_normalisation));
        Record r;
        r.set("schema", Value::from_string(lvalue::LFunctionRecord::kSchema));
        r.set("n", Value::from_int(rec.n));
        r.set("k", Value::from_int(rec.k));
        r.set("ell", Value::from_int(static_cast<std::int64_t>(rec.ell)));
        r.set("R", Value::from_int(rec.R));
        r.set("M", Value::from_int(rec.M));
        r.set("G", Value::from_int(static_cast<std::int64_t>(rec.G)));
        r.set("u", Value::from_int(static_cast<std::int64_t>(rec.u)));
        r.set("sign", Value::from_string(rec.form.sign));
        r.set("geometric", Value::from_string(rec.geometric));
        r.set("form", Value::from_record(std::move(form)));
        r.set("normalisation", Value::from_string(rec.normalisation));
        r.set("a", Value::from_int(b.a));
        r.set("vanishes", Value::from_bool(b.vanishes));
        r.set("mu", Value::from_int(b.mu));
        r.set("lambda", Value::from_int(b.lambda));
        r.set("readable", Value::from_bool(b.readable));
        r.set("lambda_mod_ell", Value::from_int(b.lambda_mod_ell));
        r.set("coefficients", u64list(b.coefficients));
        r.set("precision", U32List(b.precision));
        r.set("values_modulus", Value::from_int(static_cast<std::int64_t>(b.values_modulus)));
        r.set("critical_values", u64list(b.critical_values));
        std::ostringstream h;
        h << "0x" << std::hex << std::setw(16) << std::setfill('0') << rec.hash();
        r.set("hash", Value::from_string(h.str()));
        r.set("line", Value::from_string(rec.line(b)));
        return Value::from_record(std::move(r));
    });
}

// ---------------------------------------------------------------------------
// kummer_unit(ell, i) -- slot-21 verbs package ([PG] sec:pg-missing): the exact
// cyclotomic unit eta_i of [E] sec:lvnf layer 1 on the power basis of Z[zeta_ell]
// and the Kummer l-th-power test of nf::kummer_layer1 (a thin binding; no new
// mathematics).  ell prime, 5 <= ell <= 65521, i even, 2 <= i <= ell - 3.
// ---------------------------------------------------------------------------
Value VerbKummerUnit(CallContext& ctx, const std::vector<Value>& a) {
    static const char* kWho = "kummer_unit(ell, i)";
    const uint32_t ell = console::bind::U32Arg(a[0], ctx.pos, kWho);
    const uint32_t i = console::bind::U32Arg(a[1], ctx.pos, kWho);
    if (ell < 5 || ell > 65521 || !hecke::IsPrimeU64(ell))
        throw TypeError(std::string(kWho) + ": ell must be a prime with 5 <= ell <= 65521", ctx.pos);
    if (i < 2 || i % 2 != 0 || i + 3 > ell)
        throw TypeError(std::string(kWho) + ": i must be even with 2 <= i <= ell - 3", ctx.pos);
    return Guard(kWho, ctx.pos, [&] {
        const nf::ExactEta eta = nf::exact_eta_layer1(ell, i);
        const nf::Layer1Result t = nf::kummer_layer1(ell, i);
        std::vector<Value> coeffs, exps, sp;
        for (const nf::BigInt& c : eta.coeffs) coeffs.push_back(Value::from_string(c.to_string()));
        for (const auto& kv : eta.exponents) exps.push_back(Value::from_list({Value::from_int(kv.first), Value::from_int(kv.second)}));
        for (uint32_t q : t.split_primes) sp.push_back(Value::from_int(q));
        Record r;
        r.set("schema", Value::from_string("kummer_unit/v1"));
        r.set("ell", Value::from_int(ell));
        r.set("i", Value::from_int(i));
        r.set("exponents", Value::from_list(std::move(exps)));
        r.set("coefficients", Value::from_list(std::move(coeffs)));
        r.set("max_digits", Value::from_int(static_cast<std::int64_t>(eta.max_digits)));
        r.set("v_pi", Value::from_int(t.v_pi));
        r.set("unramified", Value::from_bool(t.unramified));
        r.set("lambda_split", Value::from_bool(t.lambda_split));
        r.set("not_lth_power", Value::from_bool(t.not_lth_power));
        r.set("eigen_ok", Value::from_bool(t.eigen_ok));
        r.set("unit_ok", Value::from_bool(t.unit_ok));
        r.set("ell_divides_Bi", Value::from_bool(t.ell_divides_Bi));
        r.set("oracle_agrees", Value::from_bool(t.oracle_agrees));
        r.set("split_primes", Value::from_list(std::move(sp)));
        return Value::from_record(std::move(r));
    });
}

// ---------------------------------------------------------------------------
// hida_tower(n, ell, m) -- slot-21 verbs package: hida::ComputeTowerCell at
// class n (even), ell >= 5 prime, levels e' = 1 .. m; both readings of the
// ordinary part ([C] cor:ordprecision), per level: ord_rank, charpolys of T_p.
// Guard: ell^(m + e_max) is the working modulus, so m <= 3 and n <= 200.
// ---------------------------------------------------------------------------
Value VerbHidaTower(CallContext& ctx, const std::vector<Value>& a) {
    static const char* kWho = "hida_tower(n, ell, m)";
    // (NBCONFIRM S3) n <= 200 is the silent envelope (cost: T ~ n^3.3); ell <= 23170 (lem:bounds), m <= 3, n even >= 2 stay
    // refusals.  The working modulus ell^(m + e_max) is guarded below the binding by CoefficientRingContext (exit 5 at
    // ell^K >= 2^64; docs/implementation/slot25/POW64_investigation.md); K = m + e_max < 2 is refused by name after the pre-pass.
    console::CostGate gate(ctx, kWho, "hida_tower");
    uint32_t n = 0, m = 0;
    uint64_t ell = 0;
    gate.validate([&] {
        n = console::bind::U32Arg(a[0], ctx.pos, kWho);
        ell = console::bind::U64Arg(a[1], ctx.pos, kWho);
        m = console::bind::U32Arg(a[2], ctx.pos, kWho);
        if (ell < 5 || ell > 23170 || !hecke::IsPrimeU64(ell)) gate.refuse(std::string(kWho) + ": ell must be a prime with 5 <= ell <= 23170");
        const std::string n_msg = std::string(kWho) + ": n must be even with 2 <= n <= 200";
        if (n < 2 || n % 2 != 0) gate.refuse(n_msg);
        if (n > 200) gate.outside(n_msg);
        if (m < 1 || m > 3) gate.refuse(std::string(kWho) + ": m must be 1, 2 or 3");
    });
    gate.admit({std::to_string(n), std::to_string(ell), std::to_string(m)},
               {"the run may still refuse after the torsion pre-pass (K = m + e_max must be >= 2 and ell^K < 2^64)"});
    return Guard(kWho, ctx.pos, [&] {
        hida::TowerCell c;
        try {
            c = hida::ComputeTowerCell(n, ell, m);
        } catch (const hida::TowerPreconditionError& e) {
            throw TypeError(std::string(kWho) + ": " + e.what(), ctx.pos);   // (NBCONFIRM S3) K < 2: a typed refusal
        }
        auto polys = [](const hida::Reading& rd) {
            Record out;
            for (std::size_t j = 0; j < rd.primes.size(); ++j) {
                std::vector<Value> co;
                for (uint64_t x : rd.charpoly[j]) co.push_back(Value::from_int(static_cast<std::int64_t>(x)));
                out.set(std::to_string(rd.primes[j]), Value::from_list(std::move(co)));
            }
            return Value::from_record(std::move(out));
        };
        std::vector<Value> levels, lines;
        for (uint32_t e = 1; e <= m; ++e) {
            const hida::Reading& rd = c.module_read[e - 1];
            const hida::Reading& fr = c.free_read[e - 1];
            Record lv;
            lv.set("e", Value::from_int(e));
            lv.set("ord_rank", Value::from_int(rd.ord_rank));
            lv.set("image_free", Value::from_bool(rd.image_free));
            lv.set("charpoly", polys(rd));
            lv.set("free_ord_rank", Value::from_int(fr.ord_rank));
            lv.set("free_charpoly", polys(fr));
            lv.set("readings_agree", Value::from_bool(rd.ord_rank == fr.ord_rank && rd.charpoly == fr.charpoly));
            levels.push_back(Value::from_record(std::move(lv)));
            lines.push_back(Value::from_string(hida::HidaTowerLine(c, e)));
        }
        Record r;
        r.set("schema", Value::from_string("hida_tower/v1"));
        r.set("n", Value::from_int(n));
        r.set("ell", Value::from_int(static_cast<std::int64_t>(ell)));
        r.set("m", Value::from_int(m));
        r.set("e_max", Value::from_int(c.e_max));
        r.set("K", Value::from_int(c.K));
        r.set("h", Value::from_int(c.h));
        r.set("exps", U32List(c.exps));
        r.set("levels", Value::from_list(std::move(levels)));
        r.set("lines", Value::from_list(std::move(lines)));
        return Value::from_record(std::move(r));
    });
}

namespace {

// ---------------------------------------------------------------------------
// Slot-22 WP-VERB: level_n, nebentypus, forms_algebra, hida_lambda.  Thin
// read-only bindings over the public Hecke / HeckeExt APIs (no new
// mathematics).  Each refused by name past a cost bound; since NBCONFIRM S3 that
// bound is the verb's silent envelope (Console/Core/Confirm.h).  Re-measured in
// S1 (<= 3.4 s, <= 66 MB at the bound; the WP-VERB report is lost) and fitted
// beyond it in S2 (docs/console/NBCONFIRM_S2_calibration.md).
// ---------------------------------------------------------------------------
// (NBCONFIRM S3) The size limits (p <= pmax, at most `maxcount` primes) are part of each verb's silent envelope (S1 row 8):
// crossing one is recorded on the gate; p prime and p != ell stay refusals.
std::vector<uint64_t> VerbPrimesArg(console::CostGate& gate, const std::vector<Value>& a, size_t idx, uint64_t ell, const char* who,
                                    const SourcePos& pos, uint64_t pmax = 97, size_t maxcount = 6) {
    std::vector<uint64_t> out;
    if (idx < a.size() && !a[idx].is_nil()) {
        if (!a[idx].is_list()) gate.refuse(std::string(who) + ": `primes` is a List of primes");
        for (const Value& v : a[idx].list_value()) {
            const uint64_t p = console::bind::U64Arg(v, pos, who);
            const std::string p_msg = std::string(who) + ": `primes` must be primes <= " + std::to_string(pmax);
            if (!hecke::IsPrimeU64(p)) gate.refuse(p_msg);
            if (p > pmax) gate.outside(p_msg);
            if (p == ell) gate.refuse(std::string(who) + ": `primes` must avoid ell");
            if (std::find(out.begin(), out.end(), p) == out.end()) out.push_back(p);
        }
        if (out.size() > maxcount) gate.outside(std::string(who) + ": at most " + std::to_string(maxcount) + " primes");
        std::sort(out.begin(), out.end());
    } else {
        for (uint64_t p : {2u, 3u, 5u})
            if (p != ell) out.push_back(p);
    }
    return out;
}

// The estimator's spelling of a prime list ("2:3:5"), or no entry for the default list.
void PushPrimes(std::vector<std::string>& args, const std::vector<Value>& a, size_t idx, const std::vector<uint64_t>& primes) {
    if (idx < a.size() && !a[idx].is_nil()) {
        std::string pl;
        for (uint64_t p : primes) pl += (pl.empty() ? "" : ":") + std::to_string(p);
        args.push_back(pl);
    }
}

} // namespace

Value VerbLevelN(CallContext& ctx, const std::vector<Value>& a) {
    static const char* kWho = "level_n(N, n, ell, primes)";
    // (NBCONFIRM S3) N <= 100, n <= 200, index(N) (n + 1) <= 1000 and the prime-list size are the silent envelope; ell in
    // [5, 65521] prime (u32 products), N >= 1 and n even stay refusals.
    console::CostGate gate(ctx, kWho, "level_n");
    uint64_t N = 0;
    uint32_t n = 0, ell = 0;
    std::vector<uint64_t> primes;
    gate.validate([&] {
        N = console::bind::U64Arg(a[0], ctx.pos, kWho);
        n = console::bind::U32Arg(a[1], ctx.pos, kWho);
        ell = console::bind::U32Arg(a[2], ctx.pos, kWho);
        if (ell < 5 || ell > 65521 || !hecke::IsPrimeU64(ell)) gate.refuse(std::string(kWho) + ": ell must be a prime with 5 <= ell <= 65521");
        const std::string N_msg = std::string(kWho) + ": N must satisfy 1 <= N <= 100";
        if (N < 1) gate.refuse(N_msg);
        if (N > 100) gate.outside(N_msg);
        // (S3 Fix round 1, audit m1) a word-size precondition of hecke::level (FactorLevel: N < 2^31), checked after the envelope
        // bound so that every prompt-less mode keeps its pre-S3 text; in `yes` / terminal modes it replaces the leaked ENGINE_CHECK.
        if (N >= (1ull << 31)) gate.refuse(std::string(kWho) + ": N must be < 2^31 (word size of the level code)");
        const std::string n_msg = std::string(kWho) + ": n must be even with 0 <= n <= 200";
        if (n % 2 != 0) gate.refuse(n_msg);
        if (n > 200) gate.outside(n_msg);
        if (hecke::level::IndexGamma0(N) * (static_cast<uint64_t>(n) + 1) > 1000)
            gate.outside(std::string(kWho) + ": index(N) * (n + 1) must be <= 1000 (the cost bound of this verb)");
        primes = VerbPrimesArg(gate, a, 3, ell, kWho, ctx.pos);
    });
    {
        std::vector<std::string> ca{std::to_string(N), std::to_string(n), std::to_string(ell)};
        PushPrimes(ca, a, 3, primes);
        gate.admit(ca);
    }
    return Guard(kWho, ctx.pos, [&] {
        const engine::ModCtx<uint32_t> mc(ell);
        const hecke::level::Gamma0LevelN lvl(N);
        console::PollCancel("level_n.cokernel");   // (NBCONFIRM S5)
        const hecke::level::SparseCokernel Q = hecke::level::SparseCokernel::Build(lvl, n, mc);
        std::vector<Value> ops;
        for (uint64_t p : primes) {
            console::PollCancel("level_n.operator");   // per Hecke prime
            const hecke::level::HeilbronnOperator H(lvl, p, n, mc);
            const hecke::level::HMat M = hecke::level::OnCokernelSparse(Q, H);
            std::vector<Value> cp;
            for (uint64_t x : hecke::level::HCharpoly(M)) cp.push_back(Value::from_int(static_cast<std::int64_t>(x)));
            Record o;
            o.set("label", Value::from_string(H.label()));
            o.set("p", Value::from_int(static_cast<std::int64_t>(p)));
            o.set("bad", Value::from_bool(H.bad()));
            o.set("charpoly", Value::from_list(std::move(cp)));
            ops.push_back(Value::from_record(std::move(o)));
        }
        Record r;
        r.set("schema", Value::from_string("level_n/v1"));
        r.set("N", Value::from_int(static_cast<std::int64_t>(N)));
        r.set("n", Value::from_int(n));
        r.set("ell", Value::from_int(ell));
        r.set("index", Value::from_int(lvl.m()));
        r.set("D", Value::from_int(Q.D()));
        r.set("dim_S", Value::from_int(Q.stats().dim_S));
        r.set("dim_ST", Value::from_int(Q.stats().dim_ST));
        r.set("h0", Value::from_int(Q.h0()));
        r.set("h1", Value::from_int(Q.h1()));
        r.set("cusps", Value::from_int(static_cast<std::int64_t>(hecke::level::CuspCount(N))));
        r.set("dim_cusp_forms", Value::from_int(hecke::level::DimCuspForms(N, n + 2)));
        r.set("ops", Value::from_list(std::move(ops)));
        return Value::from_record(std::move(r));
    });
}

Value VerbNebentypus(CallContext& ctx, const std::vector<Value>& a) {
    static const char* kWho = "nebentypus(N, m, n, ell, primes)";
    // (NBCONFIRM S3) N <= 100, n <= 100, index(N) (n + 1) <= 1000 and the prime list (p <= 13, at most 4) are the silent
    // envelope; ell, N >= 1, ell not dividing N and the Conrey index stay refusals.
    console::CostGate gate(ctx, kWho, "nebentypus");
    uint64_t N = 0, m = 0;
    uint32_t n = 0, ell = 0;
    std::vector<uint64_t> primes;
    gate.validate([&] {
        N = console::bind::U64Arg(a[0], ctx.pos, kWho);
        m = console::bind::U64Arg(a[1], ctx.pos, kWho);
        n = console::bind::U32Arg(a[2], ctx.pos, kWho);
        ell = console::bind::U32Arg(a[3], ctx.pos, kWho);
        if (ell < 5 || ell > 65521 || !hecke::IsPrimeU64(ell)) gate.refuse(std::string(kWho) + ": ell must be a prime with 5 <= ell <= 65521");
        const std::string N_msg = std::string(kWho) + ": N must satisfy 1 <= N <= 100 and ell must not divide N";
        if (N < 1 || N % ell == 0) gate.refuse(N_msg);
        if (N > 100) gate.outside(N_msg);
        // (S3 Fix round 1, audit m1) a word-size precondition of hecke::level (FactorLevel: N < 2^31), checked after the envelope
        // bound so that every prompt-less mode keeps its pre-S3 text; in `yes` / terminal modes it replaces the leaked ENGINE_CHECK.
        if (N >= (1ull << 31)) gate.refuse(std::string(kWho) + ": N must be < 2^31 (word size of the level code)");
        if (m < 1 || std::gcd(m, N) != 1) gate.refuse(std::string(kWho) + ": the Conrey index m must be a unit mod N");
        if (n > 100) gate.outside(std::string(kWho) + ": n must satisfy 0 <= n <= 100");
        if (hecke::level::IndexGamma0(N) * (static_cast<uint64_t>(n) + 1) > 1000)
            gate.outside(std::string(kWho) + ": index(N) * (n + 1) must be <= 1000 (the cost bound of this verb)");
        primes = VerbPrimesArg(gate, a, 4, ell, kWho, ctx.pos, 13, 4);
    });
    {
        std::vector<std::string> ca{std::to_string(N), std::to_string(m), std::to_string(n), std::to_string(ell)};
        PushPrimes(ca, a, 4, primes);
        gate.admit(ca);
    }
    return Guard(kWho, ctx.pos, [&] {
        namespace nb = hecke::neb;
        const nb::DirichletCharacter chi = nb::DirichletCharacter::Conrey(N, m);
        const nb::Fq F = nb::Fq::Prime(ell);
        if ((ell - 1) % chi.order() != 0)
            throw std::runtime_error("the order " + std::to_string(chi.order()) + " of chi does not divide ell - 1: chi does not reduce into F_ell");
        const nb::CharacterValues psi = nb::ReduceCharacter(chi, F);
        const hecke::level::Gamma0LevelN lvl(N);
        console::PollCancel("nebentypus.cokernel");   // (NBCONFIRM S5)
        const nb::NebCokernel ck(lvl, n, F, psi);
        std::vector<Value> ops;
        if (!ck.parity_zero()) {
            for (uint64_t p : primes) {
                console::PollCancel("nebentypus.operator");   // per Hecke prime
                const nb::HeilbronnChi H(lvl, p, n, F, psi);
                const nb::FqMat X = ck.OnH1(H);
                std::vector<Value> cp;
                for (uint32_t x : nb::FqCharpoly(F, X)) cp.push_back(Value::from_int(x));
                Record o;
                o.set("label", Value::from_string(H.label()));
                o.set("p", Value::from_int(static_cast<std::int64_t>(p)));
                o.set("bad", Value::from_bool(H.bad()));
                o.set("charpoly", Value::from_list(std::move(cp)));
                ops.push_back(Value::from_record(std::move(o)));
            }
        }
        Record r;
        r.set("schema", Value::from_string("nebentypus/v1"));
        r.set("label", Value::from_string(chi.label()));
        r.set("N", Value::from_int(static_cast<std::int64_t>(N)));
        r.set("m", Value::from_int(static_cast<std::int64_t>(m)));
        r.set("order", Value::from_int(static_cast<std::int64_t>(chi.order())));
        r.set("parity_ok", Value::from_bool(!ck.parity_zero()));
        r.set("n", Value::from_int(n));
        r.set("ell", Value::from_int(ell));
        r.set("D", Value::from_int(ck.D()));
        r.set("h0", Value::from_int(ck.h0()));
        r.set("h1", Value::from_int(ck.h1()));
        r.set("ops", Value::from_list(std::move(ops)));
        return Value::from_record(std::move(r));
    });
}

Value VerbFormsAlgebra(CallContext& ctx, const std::vector<Value>& a) {
    static const char* kWho = "forms_algebra(k, ell, primes, terms)";
    // (NBCONFIRM S3) k <= 400, terms <= 64 and the prime-list size are the silent envelope; ell and k even >= 12 stay refusals.
    console::CostGate gate(ctx, kWho, "forms_algebra");
    uint32_t k = 0, ell = 0, terms = 8;
    std::vector<uint64_t> primes;
    gate.validate([&] {
        k = console::bind::U32Arg(a[0], ctx.pos, kWho);
        ell = console::bind::U32Arg(a[1], ctx.pos, kWho);
        if (ell < 5 || ell > 65521 || !hecke::IsPrimeU64(ell)) gate.refuse(std::string(kWho) + ": ell must be a prime with 5 <= ell <= 65521");
        const std::string k_msg = std::string(kWho) + ": k must be even with 12 <= k <= 400";
        if (k < 12 || k % 2 != 0) gate.refuse(k_msg);
        if (k > 400) gate.outside(k_msg);
        primes = VerbPrimesArg(gate, a, 2, ell, kWho, ctx.pos);
        terms = a.size() > 3 && !a[3].is_nil() ? console::bind::U32Arg(a[3], ctx.pos, kWho) : 8;
        if (terms > 64) gate.outside(std::string(kWho) + ": terms must be <= 64");
    });
    {
        std::vector<std::string> ca{std::to_string(k), std::to_string(ell)};
        PushPrimes(ca, a, 2, primes);
        gate.admit(ca);
    }
    return Guard(kWho, ctx.pos, [&] {
        namespace fm = hecke_ext::forms;
        const uint32_t d0 = (k / 12) + 1;
        const uint32_t pmax = static_cast<uint32_t>(primes.empty() ? 2 : primes.back());
        const uint32_t N = std::max(2 * d0 * pmax + 2, terms + 1);
        console::PollCancel("forms_algebra.space");   // (NBCONFIRM S5)
        const fm::CuspSpace S = fm::BuildCuspSpace(k, ell, N);
        const lambda::ZMod R(ell, 1);
        std::vector<Value> basis, ops;
        for (const fm::QSeries& f : S.basis) {
            std::vector<Value> c;
            for (uint32_t i = 0; i < terms && i < f.size(); ++i) c.push_back(Value::from_int(f[i]));
            basis.push_back(Value::from_list(std::move(c)));
        }
        for (uint64_t p : primes) {
            console::PollCancel("forms_algebra.operator");   // (NBCONFIRM S5) per Hecke prime
            const ExtMatrix T = fm::HeckeOnCusp(S, static_cast<uint32_t>(p));
            lambda::Mat M(S.d, lambda::Vec(S.d, 0));
            for (uint32_t i = 0; i < S.d; ++i)
                for (uint32_t j = 0; j < S.d; ++j) M[i][j] = T.data[i][j];
            std::vector<Value> cp;
            for (lambda::u128 x : lambda::Charpoly(M, R)) cp.push_back(Value::from_int(static_cast<std::int64_t>(x)));
            Record o;
            o.set("p", Value::from_int(static_cast<std::int64_t>(p)));
            o.set("charpoly", Value::from_list(std::move(cp)));
            ops.push_back(Value::from_record(std::move(o)));
        }
        Record r;
        r.set("schema", Value::from_string("forms_algebra/v1"));
        r.set("k", Value::from_int(k));
        r.set("ell", Value::from_int(ell));
        r.set("d", Value::from_int(S.d));
        r.set("k_prime", Value::from_int(S.k_prime));
        r.set("precision", Value::from_int(S.precision));
        r.set("basis", Value::from_list(std::move(basis)));
        r.set("ops", Value::from_list(std::move(ops)));
        return Value::from_record(std::move(r));
    });
}

Value VerbHidaLambda(CallContext& ctx, const std::vector<Value>& a) {
    static const char* kWho = "hida_lambda(ell, k, W, primes)";
    // (NBCONFIRM S3) ell <= 101, k <= 200, W <= 6 and the prime-list size are the silent envelope (S1 row 6: mixed); ell >= 5
    // prime, k even >= 4, W >= 1, W <= k - 1 and the word size ell^W < 2^62 stay refusals.  The word-size check moved here from
    // inside the computation (S2 brief item 3), before the estimate; inside the envelope it is vacuous (101^6 < 2^40).
    console::CostGate gate(ctx, kWho, "hida_lambda");
    uint32_t ell = 0, k = 0, W = 0;
    std::vector<uint64_t> primes;
    gate.validate([&] {
        ell = console::bind::U32Arg(a[0], ctx.pos, kWho);
        k = console::bind::U32Arg(a[1], ctx.pos, kWho);
        W = console::bind::U32Arg(a[2], ctx.pos, kWho);
        const std::string ell_msg = std::string(kWho) + ": ell must be a prime with 5 <= ell <= 101";
        if (ell < 5 || !hecke::IsPrimeU64(ell)) gate.refuse(ell_msg);
        if (ell > 101) gate.outside(ell_msg);
        const std::string k_msg = std::string(kWho) + ": k must be even with 4 <= k <= 200";
        if (k < 4 || k % 2 != 0) gate.refuse(k_msg);
        if (k > 200) gate.outside(k_msg);
        const std::string W_msg = std::string(kWho) + ": W must satisfy 1 <= W <= 6 and W <= k - 1";
        if (W < 1 || k - 1 < W) gate.refuse(W_msg);
        if (W > 6) gate.outside(W_msg);
        primes = VerbPrimesArg(gate, a, 3, ell, kWho, ctx.pos);
        unsigned __int128 M = 1;
        for (uint32_t i = 0; i < W && M < (static_cast<unsigned __int128>(1) << 62); ++i) M *= ell;
        if (M >= (static_cast<unsigned __int128>(1) << 62)) gate.refuse(std::string(kWho) + ": ell^W must be < 2^62 (word size)");
    });
    {
        std::vector<std::string> ca{std::to_string(ell), std::to_string(k), std::to_string(W)};
        PushPrimes(ca, a, 3, primes);
        gate.admit(ca);
    }
    return Guard(kWho, ctx.pos, [&] {
        namespace lm = hecke_ext::lambda;
        const lm::ZMod R(ell, W);
        if (R.M >= (static_cast<lm::u128>(1) << 62)) throw std::runtime_error("ell^W must be < 2^62");
        const lm::FormsWeight fw = lm::ComputeFormsWeight(ell, k, W, primes);
        std::vector<Value> ops;
        for (const std::string& nm : fw.S.names) {
            std::vector<Value> cp;
            for (lm::u128 x : lm::Charpoly(fw.S.mats.at(nm), R)) cp.push_back(Value::from_int(static_cast<std::int64_t>(x)));
            Record o;
            o.set("name", Value::from_string(nm));
            o.set("charpoly", Value::from_list(std::move(cp)));
            ops.push_back(Value::from_record(std::move(o)));
        }
        Record eis;
        for (const auto& kv : fw.eis) eis.set(kv.first, Value::from_int(static_cast<std::int64_t>(kv.second)));
        Record r;
        r.set("schema", Value::from_string("hida_lambda/v1"));
        r.set("ell", Value::from_int(ell));
        r.set("k", Value::from_int(k));
        r.set("W", Value::from_int(W));
        r.set("dim_M", Value::from_int(fw.dim_M));
        r.set("dim_S", Value::from_int(fw.dim_S));
        r.set("rank_M", Value::from_int(fw.M.s));
        r.set("rank_S", Value::from_int(fw.S.s));
        r.set("comm_ok", Value::from_bool(fw.S.comm_ok && fw.M.comm_ok));
        r.set("stable_ok", Value::from_bool(fw.stable_ok));
        r.set("eisenstein", Value::from_record(std::move(eis)));
        r.set("ops", Value::from_list(std::move(ops)));
        return Value::from_record(std::move(r));
    });
}

} // namespace verbs_detail
} // namespace hecke_ext
