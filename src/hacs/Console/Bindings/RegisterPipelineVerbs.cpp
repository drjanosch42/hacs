// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// RegisterPipelineVerbs.cpp -- see RegisterPipelineVerbs.h.
// ===========================================================================

#include "RegisterPipelineVerbs.h"

#include "Bindings.h"
#include "Display.h"

#include "../Core/Cancel.h"
#include "../Core/CoreVerbs.h"
#include "../Core/Registry.h"
#include "../../Util/CancelPoint.h"

namespace console {

void RegisterPipelineVerbs(Registry& registry) {
    // (NBCONFIRM S5) the pipeline's cancellation points (Util/CancelPoint.h) poll the console's request flag; a no-op
    // unless the calling thread runs a REPL statement or a served cell (Console/Core/Cancel.h).
    oc_util::SetCancelPoll(&PollCancel);
    // ORDER MATTERS.  Overload resolution is "first match in registration
    // order" (plan Sec. 2 C3), and the later TUs look their argument types up
    // by name, so CoreMath -- which registers Matrix, Poly and Fp -- must run
    // before anything that takes a Matrix.
    RegisterCoreVerbs(registry);
    // (slot-24 NB-RENDER R3) `display`, beside the core printing verbs -- one new name, so its place moves no overload.
    bind::RegisterDisplayBindings(registry);
    bind::RegisterCoreMathBindings(registry);       // Fp / Poly / Matrix + Sec. 3.5
    bind::RegisterModelBindings(registry);          // sl2z_model, complexes, presets
    bind::RegisterCohomologyBindings(registry);     // symmetric_power, cohomology
    bind::RegisterHeckeBindings(registry);          // hecke, family
    // (C5a, slot-9 WP-12) E7's two types and their verbs `dual` / `pairing`,
    // beside the pipeline verbs they extend -- new names only, so their place
    // in the order moves no existing overload.
    bind::RegisterDualityBindings(registry);
    // (C1b) The decomposition surface -- registered AFTER RegisterHeckeBindings
    // because `decompose` takes HeckeOperator handles and looks the type up by
    // name.  It also installs the engine-owned MATHEMATICAL settings.
    bind::RegisterPresetBindings(registry);         // the S^1 / cylinder / Moebius presets
    bind::RegisterDecompositionBindings(registry);  // decompose, report, crosschecks
    bind::RegisterReproducibilityBindings(registry);// hash, golden, fingerprint, save_state
    // (C2a, slot-6 WP-13) The engine knobs, as a second settings installer
    // AFTER the mathematical one, so `settings()` lists them after those.
    bind::RegisterEngineSettingsBindings(registry);
    // (C2b, slot-7 WP-8) The harness verbs -- new names only, so their place in
    // the order moves no existing overload.
    bind::RegisterHarnessBindings(registry);
    // (C4, slot-8 WP-10) The generated reference manual -- one new name, so
    // its place in the order moves no existing overload.
    bind::RegisterReferenceBindings(registry);
    // (slot-24 D6 audit m1) The scalar-argument verbs of this table that compute
    // over F_ell and never read the ring refuse any other ring at their
    // statement (exit 5); see bind::GateVerbsOnField.  Wrapping moves no overload.
    bind::GateVerbsOnField(registry, {"dickson_exceptional", "dickson_count", "s1_scalar", "s1_unipotent",
                                      "cylinder_scalar", "cylinder_unipotent", "moebius_scalar",
                                      "moebius_unipotent"});
    bind::InstallExceptionSeam(registry);           // the C7 catch order
}

void RegisterPipelineVerbs() {
    static bool done = false;
    if (done) return;
    done = true;
    RegisterPipelineVerbs(Registry::global());
}

const std::vector<std::string>& PipelineTypeNames() {
    static const std::vector<std::string> names = {
        bind::kTypeFp,      bind::kTypePoly,       bind::kTypeMatrix,
        bind::kTypeModel,   bind::kTypeComplex,    bind::kTypeLocalSys,
        bind::kTypeCohomology, bind::kTypeHecke,   bind::kTypeFamily,
        // (C1b)
        bind::kTypeDecomposition, bind::kTypeBlock, bind::kTypeReport,
        // (C2b)
        bind::kTypeCampaignSpec,
        // (C5a, slot-9 WP-12) E7
        bind::kTypeContragredient, bind::kTypePairing,
    };
    return names;
}

const std::vector<std::string>& PipelineVerbNames() {
    static const std::vector<std::string> names = {
        // arithmetic (plan Sec. 3.4)
        "Fp", "poly", "X", "coeffs", "degree", "factor", "gcd", "powmod", "eval",
        "matrix", "identity", "zero", "mul", "add", "sub", "scale",
        // the Sec. 3.5 linear-algebra surface
        "rank", "nullity", "rref", "kernel", "image", "column_space", "inverse", "transpose",
        "solve", "hstack", "vstack", "submatrix", "columns", "vec", "unvec",
        "sum_spaces", "intersect", "contains", "in_span",
        "restrict", "algebra", "commutant", "is_cyclic", "cyclic_vector",
        "random_vector", "random_matrix", "jordan_type",
        "charpoly", "minpoly", "trace", "eval_at",
        // the pipeline (plan Sec. 3.6)
        "sl2z_model", "models", "sphere", "cylinder", "moebius", "projective_plane", "complex",
        "symmetric_power", "cohomology", "dickson_exceptional", "dickson_count",
        "hecke", "family",
        // the C1b local-system presets ([P] Sec. 3.6, row LocalSystem)
        "s1_scalar", "s1_unipotent", "cylinder_scalar", "cylinder_unipotent",
        "moebius_scalar", "moebius_unipotent",
        // the C1b decomposition surface (plan Sec. 10, Phase C1b)
        "decompose", "report", "crosschecks",
        // reproducibility (plan Sec. 2 C5)
        "hash", "hash_int", "golden", "golden_has", "golden_size", "fingerprint",
        "math_settings", "save_state",
        // (C2b, slot-7 WP-8) the harness verbs ([P] Sec. 5, Sec. 10 C2b) --
        // the same list as bind::HarnessVerbNames(), spelled out here so this
        // table stays the one place a reader finds every pipeline verb.
        "validations", "run_validation", "bench", "micro", "campaign_spec",
        "campaign_spec_text", "campaign_plan", "run_campaign", "golden_subset", "timed",
        "bench_line", "rss",
        // (C4, slot-8 WP-10) the generated reference manual ([P] Sec. 4.4).
        "help_markdown",
        // (C5a, slot-9 WP-12) E7's reserved verbs ([P] Sec. 3.6).
        "dual", "pairing",
    };
    return names;
}

const std::vector<std::string>& VerbsForbiddenBeforeC1b() {
    // (C1b) EMPTY since phase C1b: `decompose` / `report` / `crosschecks` are
    // registered above.  The accessor is kept -- and kept empty rather than
    // deleted -- because ConsoleC0Validation still asserts that a C0-ONLY
    // registry has none of the C1 surface, and because an empty list is the
    // checkable statement "nothing of the decomposition surface is withheld
    // any more".  The successor guard is VerbsForbiddenBeforeC2().
    static const std::vector<std::string> names = {};
    return names;
}

const std::vector<std::string>& VerbsForbiddenBeforeC2() {
    // The next phase's reserved names ([P] Sec. 3.6 "Reserved names for the
    // extension engine", Sec. 10 Phase C2/C5).  None of them is registered by
    // C1b; the list is what keeps the phase boundary checkable now that C1b's
    // own list is empty.
    //
    // (C2b, slot-7 WP-8) The C2 half of the list is REGISTERED now --
    // `validations`, `run_validation`, `bench`, `golden_subset` and the
    // campaign verb, which [P] Sec. 5 and acceptance (d) spell `run_campaign`
    // (the placeholder `campaign_run` was never a plan name) -- so they left
    // the list.  What remains are phase C5's reserved names.
    static const std::vector<std::string> names = {
        "boundary_cohomology", "compact_support", "iota", "sign_split", "atlas",
    };
    return names;
}

} // namespace console
