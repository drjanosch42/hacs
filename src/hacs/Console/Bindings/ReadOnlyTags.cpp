// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// ReadOnlyTags.cpp -- see ReadOnlyTags.h.
//
// THE TABLE.  One row per registered verb overload, in registration order (the
// order of docs/latex/cli_manual/verbs.tex).  `true` = the overload performs
// pure computation on in-memory objects and session state: it opens no file by
// a request-supplied name and none for writing, reads no environment variable
// it did not read at process start, spawns nothing, runs no validation /
// benchmark / campaign and reaches no emitter ([P] v3.10 Sec. 18.8).  The audit
// of every row is in Console/serve/read_only_verbs.tsv.  NB-2 audited the
// forecast of [P] Sec. 18.8 row by row; NB-report ([P] v3.10.2 Sec. 18.14) split
// `report`: 158 tagged, 9 not (exit x2, report(D, level_a, path), save_state(x,
// path), run_validation, bench, micro, campaign_spec, run_campaign).
// ===========================================================================

#include "ReadOnlyTags.h"

#include "../Core/Registry.h"

#include <iterator>
#include <set>
#include <utility>

namespace console {
namespace serve_tags {

namespace {

const ReadOnlyTag kTags[] = {
    {"print", "print(value: any, ...)", true},
    {"show", "show(value: any)", true},
    {"describe", "describe(value: any)", true},
    {"str", "str(value: any)", true},
    {"repr", "repr(value: any)", true},
    {"json", "json(value: any)", true},
    {"annotate", "annotate(text: String)", true},
    {"len", "len(value: any)", true},
    {"keys", "keys(record: Record)", true},
    {"range", "range(from: Int, to: Int)", true},
    {"range", "range(from: Int, to: Int, step: Int)", true},
    {"abs", "abs(value: Int)", true},
    {"min", "min(a: Int, b: Int)", true},
    {"max", "max(a: Int, b: Int)", true},
    {"type", "type(value: any)", true},
    {"assert", "assert(condition: Bool)", true},
    {"assert", "assert(condition: Bool, message: String)", true},
    {"exit", "exit()", false},
    {"exit", "exit(code: Int)", false},
    {"vars", "vars()", true},
    {"verbs", "verbs()", true},
    {"types", "types()", true},
    {"settings", "settings()", true},
    {"reset_settings", "reset_settings()", true},
    {"script_args", "script_args()", true},
    {"compute", "compute(object: Object)", true},
    {"state", "state(object: Object)", true},
    {"resume", "resume(object: Object)", true},
    {"resume", "resume(object: Object, budget: any = nil)", true},
    {"display", "display(value: any, ...)", true},
    {"Fp", "Fp(p: Int, a: Int)", true},
    {"poly", "poly(p: Int, coeffs: List)", true},
    {"X", "X(p: Int)", true},
    {"coeffs", "coeffs(f: Poly)", true},
    {"degree", "degree(f: Poly)", true},
    {"factor", "factor(f: Poly)", true},
    {"gcd", "gcd(f: Poly, g: Poly)", true},
    {"powmod", "powmod(g: Poly, e: Int, f: Poly)", true},
    {"eval", "eval(f: Poly, x: Int)", true},
    {"matrix", "matrix(p: Int, rows: List)", true},
    {"identity", "identity(p: Int, n: Int)", true},
    {"zero", "zero(p: Int, rows: Int, cols: Int)", true},
    {"mul", "mul(a: any, b: any)", true},
    {"add", "add(a: any, b: any)", true},
    {"sub", "sub(a: any, b: any)", true},
    {"scale", "scale(A: Matrix, c: Int)", true},
    {"rank", "rank(A: Matrix)", true},
    {"nullity", "nullity(A: Matrix)", true},
    {"rref", "rref(A: Matrix)", true},
    {"kernel", "kernel(A: Matrix)", true},
    {"image", "image(A: Matrix)", true},
    {"column_space", "column_space(A: Matrix)", true},
    {"inverse", "inverse(A: Matrix)", true},
    {"transpose", "transpose(A: Matrix)", true},
    {"pow", "pow(A: Matrix, e: Int)", true},
    {"solve", "solve(A: Matrix, B: Matrix)", true},
    {"hstack", "hstack(parts: List)", true},
    {"vstack", "vstack(parts: List)", true},
    {"submatrix", "submatrix(A: Matrix, r0: Int, r1: Int, c0: Int, c1: Int)", true},
    {"columns", "columns(A: Matrix, idx: List)", true},
    {"vec", "vec(A: Matrix)", true},
    {"unvec", "unvec(v: Matrix, rows: Int, cols: Int)", true},
    {"sum_spaces", "sum_spaces(U: Matrix, V: Matrix)", true},
    {"intersect", "intersect(U: Matrix, V: Matrix)", true},
    {"contains", "contains(U: Matrix, v: Matrix)", true},
    {"in_span", "in_span(list: List, A: Matrix)", true},
    {"restrict", "restrict(A: Matrix, basis: Matrix)", true},
    {"algebra", "algebra(ops: List)", true},
    {"commutant", "commutant(ops: List)", true},
    {"is_cyclic", "is_cyclic(ops: List, v: Matrix)", true},
    {"cyclic_vector", "cyclic_vector(ops: List, seed: Int)", true},
    {"random_vector", "random_vector(p: Int, n: Int, seed: Int)", true},
    {"random_matrix", "random_matrix(p: Int, rows: Int, cols: Int, seed: Int)", true},
    {"jordan_type", "jordan_type(A: Matrix)", true},
    {"charpoly", "charpoly(A: Matrix)", true},
    {"minpoly", "minpoly(A: Matrix)", true},
    {"trace", "trace(A: Matrix)", true},
    {"eval_at", "eval_at(f: Poly, A: Matrix)", true},
    {"sl2z_model", "sl2z_model()", true},
    {"sl2z_model", "sl2z_model(name: String)", true},
    {"models", "models()", true},
    {"sphere", "sphere(n: Int)", true},
    {"cylinder", "cylinder()", true},
    {"moebius", "moebius()", true},
    {"projective_plane", "projective_plane()", true},
    {"complex", "complex(simplices: List, name: String)", true},
    {"symmetric_power", "symmetric_power(model: Model, n: Int, ell: Int)", true},
    {"cohomology", "cohomology(L: LocalSystem)", true},
    {"dickson_exceptional", "dickson_exceptional(n: Int, ell: Int)", true},
    {"dickson_count", "dickson_count(n: Int, ell: Int)", true},
    {"hecke", "hecke(H: Cohomology, p: Int)", true},
    {"hecke", "hecke(H: Cohomology, p: Int, r: Int = 1)", true},
    {"family", "family(H: Cohomology)", true},
    {"dual", "dual(L: LocalSystem)", true},
    {"pairing", "pairing(L: LocalSystem, D: ContragredientLocalSystem)", true},
    {"s1_scalar", "s1_scalar(scalar: Int, ell: Int)", true},
    {"s1_unipotent", "s1_unipotent(ell: Int)", true},
    {"cylinder_scalar", "cylinder_scalar(scalar: Int, ell: Int)", true},
    {"cylinder_unipotent", "cylinder_unipotent(ell: Int)", true},
    {"moebius_scalar", "moebius_scalar(scalar: Int, ell: Int)", true},
    {"moebius_unipotent", "moebius_unipotent(ell: Int)", true},
    {"decompose", "decompose(ops: any, q: Int)", true},
    {"report", "report(D: Decomposition, level_a: Bool = false)", true},
    {"report", "report(D: Decomposition, level_a: Bool = false, path: String = \"\")", false},
    {"crosschecks", "crosschecks(T: HeckeOperator)", true},
    {"hash", "hash(x: any)", true},
    {"hash_int", "hash_int(x: any)", true},
    {"golden", "golden(point: String, object: String)", true},
    {"golden_has", "golden_has(point: String, object: String)", true},
    {"golden_size", "golden_size()", true},
    {"fingerprint", "fingerprint()", true},
    {"math_settings", "math_settings()", true},
    {"save_state", "save_state(x: any)", true},
    {"save_state", "save_state(x: any, path: String)", false},
    {"validations", "validations()", true},
    {"validations", "validations(tier: String)", true},
    {"run_validation", "run_validation(name: String, inherit_settings: Bool = false)", false},
    {"bench", "bench(mode: String, n: Int, ell: Int, p: Int = 0, r: Int = 1, family: List = List, repeat: Int = 1, lean: Bool = false, cross_checks: Bool = false, count_allocs: Bool = false, inherit_settings: Bool = false)", false},
    {"micro", "micro(op: String, a: any = nil, b: any = nil, c: any = nil, d: any = nil, backend: String = \"\", repeat: Int = 1, inherit_settings: Bool = false)", false},
    {"campaign_spec", "campaign_spec(path: String)", false},
    {"campaign_spec_text", "campaign_spec_text(text: String)", true},
    {"campaign_plan", "campaign_plan(spec: CampaignSpec)", true},
    {"run_campaign", "run_campaign(spec: CampaignSpec, output: String = \"\", inherit_settings: Bool = false)", false},
    {"golden_subset", "golden_subset(cells: List, inherit_settings: Bool = false)", true},
    {"timed", "timed(x: any)", true},
    {"bench_line", "bench_line(record: Record)", true},
    {"bench_line", "bench_line(record: Record, mode: String)", true},
    {"rss", "rss()", true},
    {"help_markdown", "help_markdown()", true},
    {"sturm_bound", "sturm_bound(k: Int, ell: Int)", true},
    {"sturm_bound", "sturm_bound(k: Int, ell: Int, cap: Int = 0)", true},
    {"u_ell_decision", "u_ell_decision(n: Int, ell: Int)", true},
    {"u_ell_decision", "u_ell_decision(n: Int, ell: Int, budget_mb: Int)", true},
    {"bernoulli_mod", "bernoulli_mod(w: Int, ell: Int)", true},
    {"glover_factors", "glover_factors(n: Int, ell: Int)", true},
    {"canonical_field_polynomial", "canonical_field_polynomial(ell: Int, r: Int)", true},
    {"nilpotent_part", "nilpotent_part(A: Matrix)", true},
    {"nilpotent_part", "nilpotent_part(ops: List)", true},
    {"radical", "radical(alg: any)", true},
    {"tangent_space", "tangent_space(alg: any)", true},
    {"socle", "socle(alg: any)", true},
    {"nilpotency_index", "nilpotency_index(alg: any)", true},
    {"is_local", "is_local(alg: any)", true},
    {"algebra_invariants", "algebra_invariants(alg: any)", true},
    {"loewy", "loewy(alg: any, V: Matrix = nil)", true},
    {"socle_series", "socle_series(alg: any, V: Matrix = nil)", true},
    {"check_identities", "check_identities(H: Cohomology, primes: List = nil, items: List = nil)", true},
    {"L_map", "L_map(H_lower: Cohomology, H: Cohomology)", true},
    {"ext_record", "ext_record(n: Int, ell: Int, cone: Bool = false, sign_split: Bool = false, presentation: Bool = false, sturm: Bool = true, commutant: Bool = true)", true},
    {"local_algebra", "local_algebra(m: MaximalIdeal)", true},
    {"module_structure", "module_structure(m: MaximalIdeal, group: String = nil)", true},
    {"presentation", "presentation(m: MaximalIdeal)", true},
    {"atlas", "atlas(ell: Int)", true},
    {"GF", "GF(ell: Int, r: Int)", true},
    {"canonical_field", "canonical_field(ell: Int, r: Int)", true},
    {"gf_add", "gf_add(x: GFElt, y: GFElt)", true},
    {"gf_sub", "gf_sub(x: GFElt, y: GFElt)", true},
    {"gf_mul", "gf_mul(x: GFElt, y: GFElt)", true},
    {"gf_inv", "gf_inv(x: GFElt)", true},
    {"gf_pow", "gf_pow(x: GFElt, e: Int)", true},
    {"gf_frobenius", "gf_frobenius(x: GFElt)", true},
    {"gf_eval", "gf_eval(f: Poly, x: GFElt)", true},
    {"settings_snapshot", "settings_snapshot()", true},
    {"iota", "iota(H: Cohomology)", true},
    {"boundary", "boundary(H: Cohomology)", true},
    {"cone", "cone(H: Cohomology)", true},
    {"dickson_les", "dickson_les(n: Int, ell: Int)", true},
    {"hecke_ring_algebra", "hecke_ring_algebra(n: Int, ell: Int, kmax: Int = 3)", true},
    {"theta", "theta(H: Cohomology, H_up: Cohomology)", true},
    {"lfunction", "lfunction(n: Int, ell: Int, a: Int, M: Int = 2, R: Int = 4, sign: String = \"minus\", sep: List = nil)", true},
    {"kummer_unit", "kummer_unit(ell: Int, i: Int)", true},
    {"hida_tower", "hida_tower(n: Int, ell: Int, m: Int)", true},
    {"level_n", "level_n(N: Int, n: Int, ell: Int, primes: List = nil)", true},
    {"nebentypus", "nebentypus(N: Int, m: Int, n: Int, ell: Int, primes: List = nil)", true},
    {"forms_algebra", "forms_algebra(k: Int, ell: Int, primes: List = nil, terms: Int = 8)", true},
    {"hida_lambda", "hida_lambda(ell: Int, k: Int, W: Int, primes: List = nil)", true},
    // (slot-24 D6) the integral descent record (Console/Bindings/Descent.cpp): in-process, no I/O.
    {"descent", "descent(N: Int, n: Int, primes: List = nil)", true},
    // (slot-25 HZ-REC) the record integral_hecke_structure/v1 (HeckeExt/RegisterHeckeZRecordVerbs.cpp): in-process, no I/O.
    {"hecke_structure", "hecke_structure(N: Int, n: Int, algebra: String = \"full\")", true},
};

} // namespace

const std::vector<ReadOnlyTag>& ReadOnlyTagTable() {
    static const std::vector<ReadOnlyTag> table(std::begin(kTags), std::end(kTags));
    return table;
}

TagState LookupTag(const std::string& verb, const std::string& signature) {
    for (const ReadOnlyTag& t : ReadOnlyTagTable()) {
        if (verb == t.verb && signature == t.signature)
            return t.read_only ? TagState::ReadOnly : TagState::NotReadOnly;
    }
    return TagState::Untagged;
}

TagState LookupTag(const VerbEntry& entry, const Overload& overload) {
    return LookupTag(entry.name, overload.sig.to_string(entry.name));
}

const std::vector<std::string>& ServeSettableKeys() {
    static const std::vector<std::string> keys = {
        "print_max_entries", "lean", "program_cache",
        // (NBCONFIRM S4) the confirmation threshold of [P] Sec. 7.1: per session, policy only (never a value); `confirm`
        // itself is NOT settable here -- in a served session every expensive call goes through its per-cell token
        "confirm_wall_s",
        // the eleven mathematical settings (the switches that may change a value)
        "locality", "u_ell", "sturm", "sturm_unavailable", "commutant_budget", "final_mode",
        "field_polynomial", "split_field_report", "presentation", "predictions", "coefficient_ring",
    };
    return keys;
}

bool IsServeSettable(const std::string& key) {
    for (const std::string& k : ServeSettableKeys())
        if (k == key) return true;
    return false;
}

TagCensus CensusOf(const Registry& registry) {
    TagCensus c;
    std::set<std::pair<std::string, std::string>> seen;
    for (const std::string& name : registry.verb_names()) {
        const VerbEntry* v = registry.find_verb(name);
        if (v == nullptr) continue;
        bool any_ro = false;
        for (const Overload& o : v->overloads) {
            ++c.overloads;
            const std::string sig = o.sig.to_string(v->name);
            seen.insert({v->name, sig});
            const TagState s = LookupTag(v->name, sig);
            if (s == TagState::Untagged) {
                c.untagged.push_back(v->name + " | " + sig);
                continue;
            }
            ++c.tagged;
            if (s == TagState::ReadOnly) {
                ++c.read_only;
                any_ro = true;
            }
        }
        if (any_ro) ++c.read_only_verbs;
    }
    for (const ReadOnlyTag& t : ReadOnlyTagTable()) {
        if (!seen.count({t.verb, t.signature})) c.stale.push_back(std::string(t.verb) + " | " + t.signature);
    }
    return c;
}

} // namespace serve_tags
} // namespace console
