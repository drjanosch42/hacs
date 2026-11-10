// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// RegisterHeckeZRecordVerbs -- slot-25 HZ-REC: the console verb `hecke_structure` (see the header).

#include "RegisterHeckeZRecordVerbs.h"

#include "HeckeZRecordSource.h"
#include "../Console/Bindings/Bindings.h"
#include "../Console/Core/Confirm.h"
#include "../Console/Core/ConsoleError.h"
#include "../Console/Core/Registry.h"
#include "../Console/Core/Value.h"
#include "../Hecke/HeckeSettings.h"
#include "../Integral/Descent/DescentRecord.h"

#include <string>
#include <string_view>
#include <vector>

namespace heckez {
namespace record {

namespace {

using console::CallContext;
using console::Param;
using console::Record;
using console::RuntimeError;
using console::TypeError;
using console::Value;
using console::ValueKind;

// The cost bound of the verb (the descent verb's: the record runs the descent of its cell, HZ-RAT and the local reads):
// N <= 300, n <= 60, psi(N) (n + 1) <= 300.  (NBCONFIRM S2) Measured up to 94 s at (5, 40, "anemic") inside it; it is the
// silent envelope of the confirmation (Console/Core/Confirm.h), never a prompt.
constexpr uint32_t kMaxLevel = 300, kMaxN = 60;
constexpr uint64_t kMaxModuleRank = 300;

Value VerbHeckeStructure(CallContext& ctx, const std::vector<Value>& a) {
    static const char* kWho = "hecke_structure(N, n, algebra)";
    if (!hecke::CoefficientRingIsZ())
        throw RuntimeError(std::string(kWho) + ": the record is an object over Z; set coefficient_ring = \"Z\" first (in force: " +
                               hecke::math_settings().coefficient_ring.to_text() + ")",
                           ctx.pos);
    // (NBCONFIRM S3) The old cost bound is the verb's SILENT ENVELOPE (owner ruling 2026-11-06, the S1 follow-up: no prompt
    // inside it, even where a cell takes a minute or more); outside it the policy of Console/Core/Confirm.h decides.  n even,
    // N >= 1, the algebra name and the N = 1 rule stay refusals; (1, 0) is refused by name (it leaked an ENGINE_CHECK before).
    console::CostGate gate(ctx, kWho, "hecke_structure");
    uint32_t N = 0, n = 0;
    std::string alg;
    gate.validate([&] {
        N = console::bind::U32Arg(a[0], ctx.pos, kWho);
        n = console::bind::U32Arg(a[1], ctx.pos, kWho);
        alg = a.size() > 2 && a[2].is_string() ? a[2].string_value() : std::string("full");
        const std::string N_msg = std::string(kWho) + ": N must satisfy 1 <= N <= " + std::to_string(kMaxLevel);
        if (N < 1) gate.refuse(N_msg);
        if (N > kMaxLevel) gate.outside(N_msg);
        const std::string n_msg = std::string(kWho) + ": n must be even, 0 <= n <= " + std::to_string(kMaxN) + " (the cost bound of this verb)";
        if (n % 2 != 0) gate.refuse(n_msg);
        if (n > kMaxN) gate.outside(n_msg);
        if (integral_descent::record::RecordModuleRank(N, n) > kMaxModuleRank)
            gate.outside(std::string(kWho) + ": index(N) * (n + 1) must be <= " + std::to_string(kMaxModuleRank) + " (the cost bound of this verb)");
        if (alg != "full" && alg != "anemic") gate.refuse(std::string(kWho) + ": algebra is \"full\" or \"anemic\"");
        if (N == 1 && alg != "full") gate.refuse(std::string(kWho) + ": at N = 1 only \"full\" (= the anemic algebra)");
        // (NBCONFIRM S3, S1 Sec. 4 item 5) value-neutral: this cell used to fail RationalStep's ENGINE_CHECK d_target >= 1
        if (N == 1 && n == 0) gate.refuse(std::string(kWho) + ": at N = 1, n = 0 the cusp space is 0 (no Hecke structure to compute)");
    });
    gate.admit({std::to_string(N), std::to_string(n), alg});
    const HeckeStructureRecord R = BuildHeckeStructureRecord(N, n, alg, RecordSettings{}, DefaultRecordSources());
    {   // (schema audit O2 / F4) the console scans every record it returns, as the campaign does
        const ScanResult sc = ScanRecord(R);
        if (sc.values_outside_lists || sc.exact_under_bound || !sc.overall_rule_ok)
            throw RuntimeError(std::string(kWho) + ": the record fails its scan (outside=" + std::to_string(sc.values_outside_lists) +
                                   " under=" + std::to_string(sc.exact_under_bound) + "): " +
                                   (sc.problems.empty() ? std::string("overall status rule") : sc.problems.front()),
                               ctx.pos);
    }
    Record r;
    // (EXZ2-P3 fix round 1, audit F8) the schema as a literal, so ConsoleReadOnlyTagValidation (R) sees it beside its render_coverage row;
    // pinned equal to the record's kSchema at compile time
    static_assert(std::string_view(kSchema) == std::string_view("integral_hecke_structure/v1"), "the HeckeZ record schema");
    r.set("schema", Value::from_string("integral_hecke_structure/v1"));
    r.set("id", Value::from_string(R.id));
    r.set("N", Value::from_int(N));
    r.set("n", Value::from_int(n));
    r.set("algebra", Value::from_string(alg));
    r.set("status", Value::from_string(R.status));
    r.set("golden_eligible", Value::from_bool(R.golden_eligible()));
    r.set("record_fnv1a64", Value::from_string(R.record_fnv1a64));
    r.set("cert_sha256", Value::from_string(R.cert_sha256));
    r.set("settings_fingerprint", Value::from_string(R.settings_fingerprint));
    r.set("descent_record_fnv1a64", Value::from_string(R.descent.record_fnv1a64));
    r.set("signature", Value::from_string(StatusSignature(R)));
    // the per-ell generation statuses (G2) and the congruence module (G3)
    std::vector<Value> gen;
    for (const Field& f : R.fields) {
        const std::string suf = ".generation.status";
        if (f.group != "G2" || f.path.size() <= suf.size() || f.path.compare(f.path.size() - suf.size(), suf.size(), suf) != 0) continue;
        const std::string ell = f.path.substr(7, f.path.size() - 7 - suf.size());
        const Field* rt = R.find("G2.ell=" + ell + ".generation.route");
        gen.push_back(Value::from_string(ell + ":" + f.value + (rt ? "(" + rt->value + ")" : "")));
    }
    r.set("generation", Value::from_list(std::move(gen)));
    const Field* cts = R.find("G3.C_T.status");
    const Field* cte = R.find("G3.C_T.exact");
    r.set("C_T", Value::from_string(std::string(cts ? cts->value : "-") + (cte ? ":" + cte->value : "")));
    std::vector<Value> body;
    for (const Field& f : R.fields)
        if (f.group != "G9" && f.group != "G10" && f.group != "G0x") body.push_back(Value::from_string(f.path + "=" + f.value));
    r.set("body", Value::from_list(std::move(body)));
    return Value::from_record(std::move(r));
}

}  // namespace

void RegisterHeckeZRecordVerbs(console::Registry& registry) {
    registry.register_verb("hecke_structure",
                           console::bind::Sig({Param::of("N", ValueKind::Int), Param::of("n", ValueKind::Int),
                                               Param::of("algebra", ValueKind::String).optional(Value::from_string("full"))}),
                           &VerbHeckeStructure,
                           "the record integral_hecke_structure/v1 of the cell (N, n, algebra) ([C] sec:recordschema item 8; slot-25 "
                           "HZ-REC): status, record_fnv1a64, cert_sha256, the status signature and the hashed body (G0 .. G8) as a "
                           "Record; only under coefficient_ring = \"Z\"; algebra = \"full\" | \"anemic\" (N > 1); N <= 300, n even "
                           "<= 60, index(N) * (n + 1) <= 300 (the descent verb's bound; beyond it a terminal REPL asks first, "
                           "elsewhere it refuses unless confirm = \"yes\")",
                           "extension");
}

}  // namespace record
}  // namespace heckez
