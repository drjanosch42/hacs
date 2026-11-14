// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#include <iostream>
#include <string>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>

#include "ValidationRegistry.h"
#include "ValidationShard.h"   // (S24 SHARD) --all --shard i/N
#include "Bench/BenchDriver.h"
#include "Campaign/CampaignDriver.h"   // Phase 6: --campaign <spec-file>
#include "Campaign/ProgramCacheRelease.h"  // slot-8 WP-5: the --atlas between-cells hook
#include "../Cuda/CudaBench.h"   // Phase 5: installs the --backend gpu hooks
#include "Console/Bindings/RegisterPipelineVerbs.h"  // console C1a: the one CLI branch
#include "Console/Core/ConsoleCli.h"
#include "Console/Bindings/Calibrate.h"  // NBCONFIRM S3: --calibrate
#include "Console/Bindings/HostProbe.h"  // NBCONFIRM S3: the device facts behind the cost confirmation
#include "HeckeExt/RegisterHeckeExtensionVerbs.h"  // console C1b: the ext verbs
#include "HeckeExt/HeckeAtlasCli.h"  // hecke-ext slot 7 WP-5: --atlas
#if OC_BUILD_NOTEBOOK
#include "Console/Bindings/Serve.h"  // slot-19 NB-2: --serve (oc-serve/1)
#include "Console/Bindings/ServeAtlas.h"  // slot-20 NB-4: --serve-atlas (oc-serve-atlas/1)
#endif
#include "../ModArithmetic/EngineConfig.h"  // S24 CL-L0: the ENGINE_BACKEND=legacy refusal
#include "../ModArithmetic/RetiredDeviceLevers.h"  // S24 CL-R2: the retired GPU levers' refusal
#include "../ModArithmetic/DenseElim.h"  // S25 LEVERS-25: the module levers (env arm retired)
#include "../ModArithmetic/FlatMatrix.h"  // S25 LEVERS-25: HECKE_RHO_SIMD
#include "Hecke/HeckeOperatorOnK.h"                     // S25 LEVERS-25: HECKE_RHO2_STREAM
#include "Hecke/HeckeInvolution.h"                      // S25 LEVERS-25: HECKE_IOTA_STREAM
#include "HeckeExt/HeckeRingHeckeAlgebra.h"             // S25 LEVERS-25: HECKE_PRECISION_REUSE
#include "HeckeExt/HeckeFormsHeckeAlgebra.h"            // S25 LEVERS-25: HECKE_FORMS_FASTMUL
#include "Orbifold/L23/AmalgamReader.h"                 // S25 LEVERS-25: L23_TERNARY_F3, L23_BIN3O1
#include "BuildInfo.h"                                  // R1 / X1: oc::HacsVersionLine()
#include "../ModArithmetic/Kernels.h"                   // S26 LEVERS-26: ENGINE_VNNI
#include "../Engine/EngineRuntime.h"                   // S27 LEV-K: ENGINE_HOST_ELIM_HOOK's EFFECTIVE line
#include "../ModArithmetic/Binary/SmithPar.h"           // S26 LEVERS-26: HECKE_DESCENT_SMITH_MT / _F2_MT
#include "Util/FieldDiet.h"                             // S26 LEVERS-26: HECKE_FIELD_DIET
#include "Integral/Descent/DescentTree.h"               // S26 LEVERS-26: HECKE_DESCENT_TREE
#include "Integral/Descent/DescentTreePair.h"           // S26 LEVERS-26: HECKE_DESCENT_TREE_PAIR
#include "Integral/Descent/DescentAssembly.h"           // S26 LEVERS-26: HECKE_DESCENT_GOOD_SPARSE / _GOOD_DENSE
#include "Integral/Descent/DescentSmith.h"              // S26 LEVERS-26: HECKE_DESCENT_SMITH_BLOCKED
#include "Integral/Descent/DescentParAsm.h"             // S26 LEVERS-26: HECKE_DESCENT_PAR_ASM / _PAR_ELL / _EMBED_MAP / _PAR_LN

// Prints usage for the validation-harness CLI (Stage S0).  Frozen by
// ConsoleCliPreservationValidation (Console/scripts/cli_frozen/help.stdout) and
// mirrored by CliManualCoverageValidation's kTopUsage.  (S24 CL-7, owner ruling 4
// of 2026-10-30) Also printed, with exit 1, when no argument is given; the
// legacy no-argument benchmark is gone.  The --serve line is printed in every
// build so the frozen text does not depend on OC_BUILD_NOTEBOOK.
static void print_usage(const char* argv0)
{
    std::cout << "Usage:" << std::endl;
    std::cout << "  " << argv0 << "                 With no argument: print this usage text; exit 1." << std::endl;
    std::cout << "  " << argv0 << " --help          Print this usage text; exit 0." << std::endl;
    std::cout << "  " << argv0 << " --version       Print the version line (`hacs <version> (commit ...; contract ...)`); exit 0." << std::endl;
    std::cout << "  " << argv0 << " --list          List all registered validation names with their tier." << std::endl;
    std::cout << "  " << argv0 << " --all           Run the fast + standard tiers; exit code = number of failures." << std::endl;
    std::cout << "  " << argv0 << " --all-full      Run every tier including slow; exit code = number of failures." << std::endl;
    std::cout << "  " << argv0 << " <Name>          Run one registered validation by name; exit code = its own return code." << std::endl;
    std::cout << "  " << argv0 << " --bench <mode> <args...>" << std::endl;
    std::cout << "                                  Run the benchmark CLI (`--bench --help` for modes)." << std::endl;
    std::cout << "  " << argv0 << " --campaign <spec-file> [options]" << std::endl;
    std::cout << "                                  Run a parameter sweep (`--campaign --help`; docs/matrix-engine/campaign.md)." << std::endl;
    std::cout << "  " << argv0 << " --atlas <ell> [options]" << std::endl;
    std::cout << "                                  Build, close, verify or index a residue atlas (`--atlas --help`)." << std::endl;
    std::cout << "  " << argv0 << " --console | -i | --script <file.oc> | -c '<code>' | - [modifiers]" << std::endl;
    std::cout << "                                  The OC script console (`--console --help`; docs/console/console.md)." << std::endl;
    std::cout << "  " << argv0 << " --serve [options]" << std::endl;
    std::cout << "                                  The served read-only console, NDJSON on stdio (OC_BUILD_NOTEBOOK=ON builds only)." << std::endl;
}

// (S25 LEVERS-25) The slot-23 levers whose ENV ARM is retired (EnvLever.h, EnvArmRetiredLever; the old routes stay
// in every build as oracles / production routes, reached through their Scoped overrides): in an OC_ENABLE_LEGACY=OFF
// build the env reading that selects the old route is refused here (exit 1), for every entry point, instead of at
// whichever read comes first (or never, in a run that does not reach the route: the vacuous-arm lesson).  ON build:
// nothing is read here, so the ON build behaves exactly as before.  ENGINE_GPU_LEAN is read by
// refuse_retired_device_levers_at_startup().  docs/implementation/slot25/LEVERS25_report.md.
// (S26 LEVERS-26) + the eleven retired slot-24 descent env arms, HECKE_FIELD_DIET and ENGINE_VNNI (slot26/LEVERS26_plan.md).
static void refuse_retired_env_arms_at_startup() {
#if !OC_ENABLE_LEGACY
    (void)engine::elim::module_stream_lever().env_state();     // HECKE_MODULE_STREAM=0
    (void)engine::elim::module_eval_lever().env_state();       // HECKE_MODULE_EVAL=0
    (void)engine::elim::module_parallel_lever().env_state();   // HECKE_MODULE_PARALLEL=0
    (void)engine::elim::route2_shared_memo_lever().env_state();  // HECKE_ROUTE2_SHARED_MEMO=0
    rho_fill_env_check();                                      // HECKE_RHO_SIMD=0 / scalar (raw env only: no ENGINE_KERNEL resolution)
    (void)hecke::HeckeOperatorOptions::rho2_streaming();       // HECKE_RHO2_STREAM=0 (this thread's initial state)
    (void)hecke::HeckeInvolution::iota_streaming();            // HECKE_IOTA_STREAM=0 (this thread's initial state)
    (void)hecke_ext::ringalg::PrecisionReuseEnabled();         // HECKE_PRECISION_REUSE=0
    (void)hecke_ext::forms::FormsFastMulEnabled();             // HECKE_FORMS_FASTMUL=0
    (void)l23::ternary_f3_reader_enabled();                    // L23_TERNARY_F3=0
    (void)l23::bin3o1_enabled();                               // L23_BIN3O1=0
    // (S26 LEVERS-26) the slot-24 descent levers, HECKE_FIELD_DIET and ENGINE_VNNI (slot26/LEVERS26_plan.md Sec. 1)
    (void)integral_descent::DescentTreeLever().env_state();        // HECKE_DESCENT_TREE=0
    (void)integral_descent::DescentTreePairLever().env_state();    // HECKE_DESCENT_TREE_PAIR=0
    (void)integral_descent::DescentGoodSparseLever().env_state();  // HECKE_DESCENT_GOOD_SPARSE=0
    (void)integral_descent::DescentGoodDenseLever().env_state();   // HECKE_DESCENT_GOOD_DENSE=0
    (void)integral_descent::maps::DescentSmithBlockedLever().env_state();  // HECKE_DESCENT_SMITH_BLOCKED=0
    (void)engine::smithpar::mt_lever().env_state();                // HECKE_DESCENT_SMITH_MT=0
    (void)engine::smithpar::f2_lever().env_state();                // HECKE_DESCENT_SMITH_F2_MT=0
    for (const auto l : {integral_descent::par::Lever::Asm, integral_descent::par::Lever::Ell, integral_descent::par::Lever::Embed,
                         integral_descent::par::Lever::EllN})
        (void)integral_descent::par::LeverOf(l).env_state();       // HECKE_DESCENT_PAR_ASM / _PAR_ELL / _EMBED_MAP / _PAR_LN =0
    (void)oc_util::field_diet_env_on();                            // HECKE_FIELD_DIET=0
    engine::kernels::vnni_env_check();                             // ENGINE_VNNI=0 / off
#endif
}

int main(int argc, char** argv) {

    // (R1 / X1) `--version`: one line, exit 0; answered before anything else is read or registered.
    if (argc == 2) {
        const std::string arg1 = argv[1];
        if (arg1 == "--version") {
            std::cout << oc::HacsVersionLine() << std::endl;
            return 0;
        }
    }

    // (S24 CL-L0, CLEANUP plan Sec. 2.4) Read ENGINE_BACKEND once, first: in an OC_ENABLE_LEGACY=OFF build
    // `ENGINE_BACKEND=legacy` is refused here (exit 1) for every entry point, instead of at whichever read comes first.
    (void)engine::backend_is_legacy();
    // (S24 CL-R2) The same for the three retired GPU rollback levers (ENGINE_ELIM_SESSION, ENGINE_GPU_IMMA,
    // ENGINE_GPU_ELIM; + ENGINE_GPU_LEAN since S25 LEVERS-25): in an OFF build a value starting with '0' is refused here,
    // whatever the device policy.
    engine::lever::refuse_retired_device_levers_at_startup();
    // (S25 LEVERS-25) and the slot-23 levers whose env arm is retired (ENGINE_GPU_LEAN is in the call above).
    refuse_retired_env_arms_at_startup();

    if (argc <= 1) {
        // (S24 CL-7, owner ruling 4 of 2026-10-30) No arguments: the usage text
        // and exit 1, the console's usage-error code.  Nothing is registered.
        print_usage(argv[0]);
        return 1;
    }

    // (NBCONFIRM S3, slot 25; owner addendum of 2026-11-06) `--calibrate[=thorough]`: the per-device speed calibration of
    // the console cost estimators (Console/Bindings/Calibrate.h).  On demand only; it runs its probes as child processes of
    // this binary and writes the device profile.  Dispatched before the validation registry: it registers nothing.  Not in
    // the frozen help text (cli_frozen/help.stdout); documented in the manual's token table.
    {
        const std::string arg1 = argv[1];
        if (arg1 == "--calibrate" || arg1 == "--calibrate=thorough") {
            console::host::InstallRealHostProbe();
            return console::calibrate::RunCalibrate(argc - 1, argv + 1);
        }
    }

#if OC_BUILD_NOTEBOOK
    // (hecke-extensions slot 19, NB-2; [P] v3.10 Sec. 18) The served console:
    // NDJSON on stdio, read-only.  Dispatched BEFORE the validation registry
    // and the device probe below, so a served process registers no validation
    // and touches no device; the environment gate and the process hardening
    // run inside RunServe, before the verbs are registered.  With
    // OC_BUILD_NOTEBOOK=OFF the token is unknown (the validation-name loop
    // reports it, exit 1).
    {
        const std::string arg1 = argv[1];
        if (arg1 == "--serve") {
            return console::serve::RunServe(argc - 2, argv + 2, [] {
                console::RegisterPipelineVerbs();
                hecke_ext::RegisterHeckeExtensionVerbs();
            });
        }
        // (hecke-extensions slot 20, NB-4; [P] v3.11 Sec. 18.16) read-only
        // queries on a hecke_atlas/v1 store (oc-serve-atlas/1): no verb, no
        // validation, no device; the same env gate and hardening as --serve.
        if (arg1 == "--serve-atlas") {
            return console::serve_atlas::RunServeAtlas(argc - 2, argv + 2);
        }
        // (slot-22 VERB-3, owner decision O13) `--atlas --query <store> '<json>'`:
        // the same request handler, once, for scripts; no verb, no validation.
        if (arg1 == "--atlas" && argc >= 3 && std::string(argv[2]) == "--query") {
            return console::serve_atlas::RunAtlasQuery(argc - 3, argv + 3);
        }
    }
#endif

    // One or more CLI arguments: name-addressable validation harness (Stage S0).
    RegisterAllValidations();
    // matrix-engine Phase 5: the `--backend gpu` micro arm (a no-op without a
    // device).  Installed LAZILY (memory profile 05 Sec. 0): the probe inside
    // creates a CUDA context, +110 MB of peak RSS on every CPU-only run, so it
    // is reached only by `--bench micro ...` or when a GPU device policy /
    // ENGINE_CUDA_PROBE=1 is requested in the environment.
    {
        const char* dev = std::getenv("ENGINE_DEVICE");
        const bool gpu_env = (dev != nullptr && (std::strcmp(dev, "gpu") == 0 || std::strcmp(dev, "hybrid") == 0)) ||
                             std::getenv("ENGINE_CUDA_PROBE") != nullptr;
        const bool micro = (argc > 2 && std::string(argv[1]) == "--bench" && std::string(argv[2]) == "micro");
        if (gpu_env || micro) engine::cuda::InstallMicroHooks();
    }

    const std::string arg1 = argv[1];

    if (arg1 == "--list") {
        for (const auto& entry : ValidationRegistry::list_with_tiers()) {
            std::cout << entry.first << "\t[" << ValidationTierName(entry.second) << "]" << std::endl;
        }
        return 0;
    }

    if (arg1 == "--all" || arg1 == "--all-full") {
        // Phase-0 owner decision Q13: `--all` runs the fast + standard tiers,
        // `--all-full` every tier.  The exit-code contract is unchanged: exit
        // code = failure count.
        const bool include_slow = (arg1 == "--all-full");
        // (S24 SHARD, VALSTREAM item 5) optional modifiers: `--shard i/N`
        // (1-based) runs shard i of N of that roster (ValidationShard.h);
        // `--list` then prints the shard's names instead of running them.  Any
        // other argument, or a malformed i/N, is a usage error (exit 1).  Not
        // in the frozen help text (cli_frozen/help.stdout); documented in the
        // manual's token table and docs/validation-catalogue.md.
        bool sharded = false, list_only = false;
        ShardSpec spec;
        for (int k = 2; k < argc; ++k) {
            const std::string mod = argv[k];
            std::string err;
            if (mod == "--shard" && k + 1 < argc && !sharded) {
                if (!ParseShardSpec(argv[++k], spec, err)) {
                    std::cerr << "error: " << err << std::endl;
                    return 1;
                }
                sharded = true;
            } else if (mod == "--list" && !list_only) {
                list_only = true;
            } else {
                std::cerr << "error: " << arg1 << " takes only [--shard i/N] [--list], got '" << mod << "'"
                          << std::endl;
                return 1;
            }
        }
        if (list_only && !sharded) spec = ShardSpec{};  // `--all --list`: the whole roster as shard 1/1
        if (sharded || list_only)
            return static_cast<int>(ValidationRegistry::run_shard(include_slow, spec.index, spec.count, list_only));
        uint32_t failures = ValidationRegistry::run_all(include_slow);
        return static_cast<int>(failures);
    }

    if (arg1 == "--bench") {
        // Dispatch straight into the plain-C++ bench TU; main.cpp stays thin
        // (see Bench/BenchDriver.h for why the driver is not an nvcc TU).
        return bench::RunBench(argc - 2, argv + 2);
    }

    if (arg1 == "--campaign") {
        // matrix-engine Phase 6: the sweep driver. Same reason as --bench for
        // living in a plain C++ TU; main.cpp only dispatches.
        const int rc = campaign::RunCampaign(argc - 2, argv + 2);
        // (S27 LEV-K) ENGINE_HOST_ELIM_HOOK's check line (EngineRuntime.h): on stderr, after a campaign that bound the
        // host elimination hook at least once, or with the variable set; never after --help / --dry-run.  Counters only.
        bool help_or_dry = false;
        for (int i = 2; i < argc; ++i)
            if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--dry-run") == 0)
                help_or_dry = true;
        if (!help_or_dry && (engine::tape::host_elim_hook_binds() > 0 || std::getenv("ENGINE_HOST_ELIM_HOOK") != nullptr))
            std::fprintf(stderr, "%s\n", engine::tape::host_elim_hook_effective_line().c_str());
        return rc;
    }

    if (arg1 == "--atlas") {
        // (hecke-extensions slot 7, WP-5) the residue-atlas build outside the
        // suite (HeckeExt/HeckeAtlasCli.h); `--atlas --help` prints its usage.
        // (slot-8 WP-5, [KI] Sec. 4c) the between-cells release of the engine's
        // program caches, injected as a callback: the build's cell loop lives
        // in HeckeExt/, which may not name the engine.
        return hecke_ext::RunAtlasCli(argc - 2, argv + 2,
                                      [] { campaign::ReleaseEngineProgramCaches(
                                          campaign::kCellLoopReleaseThreshold); });
    }

    if (arg1 == "--help" || arg1 == "-h") {
        print_usage(argv[0]);
        return 0;
    }

    // Console phase C1a (docs/console/00-console-interface-plan.md v3 Sec. 9):
    // the ONE branch the console adds, inserted AFTER the --help test and
    // BEFORE the validation-name loop.  The console usage is printed by
    // `--console --help` (plan Q8); print_usage only points at it, and is gated
    // by ConsoleCliPreservationValidation.
    if (arg1 == "--console" || arg1 == "-i" || arg1 == "--script" || arg1 == "-c" ||
        arg1 == "-") {
        console::RegisterPipelineVerbs();
        // (console C1b, slot-4 WP-5 item 6; [OD] slot-3 close-out 1) The
        // extension layer registers its OWN verbs, from HeckeExt/, so that no
        // Console/Bindings/ TU has to include a HeckeExt/ header (which
        // check_heckeext_includes.sh rule 1 forbids).  It must run after
        // RegisterPipelineVerbs, whose types it looks up by name.
        hecke_ext::RegisterHeckeExtensionVerbs();
        // (NBCONFIRM S3) the terminal / memory / disk / CPU facts the cost confirmation reads (Console/Core/Confirm.h)
        console::host::InstallRealHostProbe();
        return console::RunConsole(argc - 1, argv + 1);
    }

    // One or more validation names: run each in THIS process, in order; the
    // exit code is the number of failures.  Several names in one process is
    // how a state-dependent failure (program cache, thread pool, pool ids
    // carried across validations) is reproduced without a full --all.
    uint32_t failures = 0;
    for (int i = 1; i < argc; ++i)
        if (ValidationRegistry::run(argv[i]) != 0) ++failures;
    return static_cast<int>(failures);
}
