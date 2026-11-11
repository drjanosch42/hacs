// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#include "ValidationRegistry.h"
#include "Util/DeviceRelease.h"
#include "ValidationShard.h"  // (S24 SHARD) --shard i/N
#include "../Cuda/CudaCtx.h"  // (S24 SHARD) the device-shard guard: context_created()

#include <exception>
#include <iostream>
#include <utility>

// Headers of the six pre-existing validations (Stage S0 preconditions).
#include "../ModArithmetic/ModArithmeticValidation.h"
#include "../ModArithmetic/MatrixValidation.h"
#include "../ModArithmetic/FlatMatrixValidation.h"
#include "../ModArithmetic/legacy/BlockFlatMatrixValidation.h"   // (S24 CL-L2) legacy-gated
#include "SimplicialCohomology.h"
#include "LocalSystemCohomologyValidation.h"

// Hecke validations (Stage S1+).
#include "Hecke/HeckeExactIntSmokeValidation.h"
#include "Hecke/HeckeFpFilterValidation.h"
#include "Hecke/HeckeTessellationSmokeValidation.h"
#include "Hecke/HeckeTessellationValidation.h"
#include "Hecke/HeckeCosetTablesSelfCheckValidation.h"
#include "Hecke/HeckeCosetTablesValidation.h"
#include "Hecke/HeckeTauTableSelfCheckValidation.h"
#include "Hecke/HeckeTauTableValidation.h"
#include "Hecke/HeckeIotaTauSymmetryValidation.h"
#include "Hecke/HeckeIotaSignValidation.h"
#include "Hecke/HeckeInducedLocalSystemSmokeValidation.h"
#include "Hecke/HeckeShapiroValidation.h"
#include "Hecke/HeckeMorphismGeneralizationValidation.h"
#include "Hecke/HeckeMorphismRegressionValidation.h"
#include "Hecke/HeckeOperatorSmokeValidation.h"
#include "Hecke/HeckeOracleValidation.h"
#include "Hecke/HeckeCrossCheckValidation.h"
#include "Hecke/HeckeIndependenceValidation.h"
#include "Hecke/HeckeOperatorFamilySmokeValidation.h"
#include "Hecke/HeckeRelationsValidation.h"
#include "Hecke/HeckeDecompositionSmokeValidation.h"
#include "Hecke/HeckeBlockFormValidation.h"
#include "Hecke/HeckeSimultaneousSmokeValidation.h"
#include "Hecke/HeckeSimultaneousValidation.h"
#include "Hecke/HeckeEndToEndValidation.h"
#include "Hecke/HeckeExtendedRegimeValidation.h"
// hecke-extensions phase E0a -- the read-only locality certificate; E0b (slot
// 2 WP-1) -- the same certificate inside the refinement loop, plus HeckeSettings;
// E0b-S (slot 3 WP-3) -- strict mode (arm 7).
#include "Hecke/HeckeLocalityValidation.h"
// hecke-extensions slot 1, WP-4a -- the certified boundary datum tau_bd.
#include "Hecke/HeckeBoundaryTauValidation.h"
// hecke-extensions slot 3, WP-1 (E1b-0) -- the U_ell exact-integer reach and
// the frozen U_ell RSS budget behind the `u_ell` mathematical setting.
#include "Hecke/HeckeUEllReachValidation.h"
#include "HeckeExt/HeckeUEllBudgetValidation.h"
#include "HeckeExt/HeckeExtSturmSweepValidation.h"
// (hecke-extensions slot 4, WP-1 / E2a) the extension record.
#include "HeckeExt/HeckeExtRecordValidation.h"
// hecke-extensions phase E1s (slot 1, WP-3) -- the computed Sturm bound
// B'(k, ell) and the HeckeExt/ one-way include guard.
#include "HeckeExt/HeckeSturmBoundValidation.h"
#include "HeckeExt/HeckeExtArchitectureValidation.h"
#include "HeckeExt/HeckeExtAlgebraValidation.h"
#include "HeckeExt/HeckeInvolutionValidation.h"
// hecke-extensions slot 1, WP-2 -- the E0a-subset identity certificate.
#include "HeckeExt/HeckeIdentityValidation.h"
// hecke-extensions slot 1, WP-4b -- the boundary Hecke operator Theta^bd.
#include "Hecke/HeckeBoundaryOperatorValidation.h"
// hecke-extensions slot 2, WP-7 (phase E4b) -- H^1_!, the Eisenstein line,
// Harder's criterion, the Harder split and the two recorded diagnostics.
#include "HeckeExt/HeckeBoundaryParabolicValidation.h"
#include "Hecke/HeckePairHomotopyValidation.h"
#include "Hecke/HeckeConeValidation.h"
#include "Hecke/HeckeConeActionValidation.h"

// Phase 1 (matrix-engine) -- item 7.4 vacuous-relation short-circuit.
#include "VacuousRelationValidation.h"
// Phase 0 -- substrate differential harness (01 Sec. 7 gap #9).
#include "../ModArithmetic/SubstrateDifferentialValidation.h"
// Phase 1 (matrix-engine) -- rho_n closed forms + recurrence (items 7.0/7.5).
#include "../ModArithmetic/SymmetricPowerValidation.h"
// matrix-engine Phase 2 (arithmetic primitive layer)
#include "../ModArithmetic/ModContextValidation.h"
#include "../ModArithmetic/KernelBitIdentityValidation.h"
#include "../ModArithmetic/DenseElimValidation.h"
#include "DenseElimPipelineValidation.h"
// matrix-engine Phase 3 -- static consumer-migration gate (tool scripts).
#include "ConsumerMigrationValidation.h"
// Console phase C0 (hecke-extensions slot 1, WP-5) -- the language core.
#include "Console/ConsoleC0Validation.h"
#include "Console/ConsoleValidation.h"
#include "Console/ConsoleSettingsValidation.h"
// (slot-7 WP-8 / console C2b) ConsoleDeterminismValidation.
#include "Console/ConsoleDeterminismValidation.h"
// (slot-8 WP-5, [KI] Sec. 4c) ProgramCacheReleaseValidation.
#include "Campaign/ProgramCacheReleaseValidation.h"
// (slot-8 WP-10 / console C4) ConsoleErrorValidation.
#include "Console/ConsoleErrorValidation.h"
// matrix-engine Phase 3 (block-sparse data model + block elimination, plan D3/D5).
#include "../ModArithmetic/BlockMatValidation.h"
#include "../ModArithmetic/BlockEliminationValidation.h"
#include "BlockModelPipelineValidation.h"
#include "Registry/LegacyValidation.h"   // (S24 CL-L2) OC_LEGACY_VALIDATION
#include "legacy/CohomologyBlockModelValidation.h"   // (S24 CL-L2) legacy-gated
// Phase 2 (matrix-engine) -- arithmetic micro-benchmarks + performance smoke.
#include "../ModArithmetic/PerformanceSmokeValidation.h"
// matrix-engine Phase 5 -- the CUDA device layer (skips cleanly without a GPU).
#include "../Cuda/CudaValidation.h"
// Phase 1 (matrix-engine) -- multi-RHS Theta_eq solve (item 7.1).
#include "Hecke/ThetaEqSolveValidation.h"
// matrix-engine Phase 3 Stage B -- BarCocycleH1 solver absorption (conditional per R2).
#include "Hecke/legacy/BarCocycleSolverValidation.h"   // (S24 CL-L2) legacy-gated
// Phase 3 Stage B (matrix-engine) -- HeckeOperatorOnK on the block model.
#include "Hecke/legacy/HeckeThetaBlockValidation.h"   // (S24 CL-L2) legacy-gated
// Phase 0 (matrix-engine) -- golden-output regression harness.
#include "Golden/GoldenHarness.h"
// Phase 3 (matrix-engine) -- the Matrix.h include guard.
#include "MatrixRetirementValidation.h"
// hecke-extensions slot 3, WP-10 -- the [C, <label>] / \HA{<label>} citation census.
#include "ContractCensusValidation.h"
// matrix-engine Phase 4 -- the tape engine (src/Engine/).
#include "../Engine/EngineTapeValidation.h"
#include "../Engine/EngineTapeEliminationValidation.h"
#include "../Engine/EngineDeterminismCoreValidation.h"
#include "../Engine/EngineArenaValidation.h"
#include "../Engine/EngineCoreScalingValidation.h"
// matrix-engine Phase 4, Stage B -- the consumers on the tape.
#include "EngineDeterminismValidation.h"
#include "EngineTapeConsumerValidation.h"
// matrix-engine Phase 6 -- the campaign driver.
#include "Campaign/CampaignValidation.h"
// matrix-engine Phase 5 Stage B -- the tape's device arm (src/Engine/DeviceBackend.*).
#include "../Engine/EngineCrossDeviceValidation.h"
#include "../Engine/EngineCudaMemoryValidation.h"
#include "../Engine/EngineDeviceTrapValidation.h"
// hecke-extensions slot 4, WP-7 (phase E6) -- Bernoulli numbers mod ell with
// the Kummer / Harder predicate, and the Glover composition factors.
#include "HeckeExt/HeckePredictionsValidation.h"
// (hecke-extensions slot 4, WP-2 / phase E2b)
#include "HeckeExt/HeckeExtRamanujanValidation.h"
// hecke-extensions slot 5, WP-7 (phase E6) -- the project-canonical split field.
#include "HeckeExt/HeckeCanonicalFieldValidation.h"
#include "HeckeExt/HeckeAtlasValidation.h"
#include "HeckeExt/HeckeExtEndToEndValidation.h"
#include "HeckeExtDeterminismValidation.h"
#include "Campaign/CampaignExtValidation.h"
// hecke-extensions slot 5, WP-8 -- identity class 5 (the Dickson L.) and M1.
#include "HeckeExt/HeckeExtKummerValidation.h"
#include "HeckeExt/HeckeExtSmallEllValidation.h"
#include "HeckeExt/HeckeExtSeparationValidation.h"
#include "Hecke/HeckeDualityValidation.h"   // slot-6 WP-1 (E7a)
#include "HeckeExt/HeckePresentationValidation.h"   // slot-6 WP-2 (E2 (a), OQ11)
#include "Hecke/HeckeReductionValidation.h"   // slot-8 WP-1 (E8 item 2)
#include "Hecke/HeckeReductionCoverageSlowValidation.h"   // slot-8 WP-4 (E8 item 4a)
#include "Integral/HeckeBredonTreeValidation.h"      // slot-8 WP-8 (Line 2, V-B)
#include "Integral/HeckeIntegralCharpolyValidation.h"   // slot-8 WP-7 (Line 2, M-0)

namespace {

struct RegistryEntry {
    std::string name;
    ValidationFn fn;
    ValidationTier tier;
};

// Explicit static table, populated only by RegisterAllValidations() below.
// A function-local static avoids static-initialization-order issues; it is
// still a single, explicit table (not self-registering statics scattered
// across translation units, which the linker can drop across the
// ModArithmetic static-library boundary -- see ValidationRegistry.h).
std::vector<RegistryEntry>& registry_table() {
    static std::vector<RegistryEntry> table;
    return table;
}

// A few of the pre-existing validations take an int verbosity parameter
// rather than being zero-argument. Adapt them to ValidationFn with the same
// default verbosity (0) used elsewhere in the codebase (see the commented
// calls in the historical kernel.cu main()).
uint32_t Wrap_mod_arithmetic_validation() { return mod_arithmetic_validation(0); }
uint32_t Wrap_FlatMatrixValidation() { return FlatMatrixValidation(0); }
#if OC_ENABLE_LEGACY
uint32_t Wrap_BlockFlatMatrixValidation() { return BlockFlatMatrixValidation(0); }
#endif

} // namespace

const char* ValidationTierName(ValidationTier tier) {
    switch (tier) {
        case ValidationTier::Fast: return "fast";
        case ValidationTier::Standard: return "standard";
        case ValidationTier::Slow: return "slow";
    }
    return "standard";
}

void ValidationRegistry::register_validation(const std::string& name, ValidationFn fn,
                                             ValidationTier tier) {
    for (auto& entry : registry_table()) {
        if (entry.name == name) {
            entry.fn = fn;
            entry.tier = tier;
            return;
        }
    }
    registry_table().push_back(RegistryEntry{name, fn, tier});
}

// (S13 WP-16) Every registered validation runs behind this guard: an exception
// escaping a validation (an ENGINE_CHECK's std::logic_error, a HeckeOverflow,
// ...) is reported as that validation's FAILURE and the suite goes on --
// ENGINE_CHECK's contract ("the validation harness reports a failure instead
// of aborting the suite").  Before this guard such an exception reached
// std::terminate (SIGABRT, exit 134).  A validation that returns normally is
// unaffected: its return code is passed through unchanged.
namespace {
// (S14 WP-3) After every validation -- normal return or exception -- device
// memory the CUDA arena caches is returned to the driver, iff the effective
// device policy is not cpu (Util/DeviceRelease.h; a no-op under cpu).
struct DeviceReleaseOnExit {
    const std::string& name;
    ~DeviceReleaseOnExit() { (void)ReleaseDeviceMemoryAfterValidation(name); }
};
// (slot-14 WP-2) The catch-path primitive itself, factored out of RunGuarded
// so it can be driven directly by name+fn (ValidationRegistry::run_guarded_standalone)
// as well as by a RegistryEntry, without duplicating the try/catch text.
uint32_t RunGuardedFn(const std::string& name, ValidationFn fn) {
    const DeviceReleaseOnExit release_on_exit{name};
    try {
        return fn();
    } catch (const std::exception& e) {
        std::cout << "[EXCEPTION] " << name << ": uncaught exception escaped the validation: " << e.what()
                  << std::endl;
    } catch (...) {
        std::cout << "[EXCEPTION] " << name << ": uncaught non-std exception escaped the validation"
                  << std::endl;
    }
    return 1;
}
uint32_t RunGuarded(const RegistryEntry& entry) { return RunGuardedFn(entry.name, entry.fn); }
} // namespace

uint32_t ValidationRegistry::run_guarded_standalone(const std::string& name, ValidationFn fn) {
    const uint32_t result = RunGuardedFn(name, fn);
    if (result == 0) {
        std::cout << "[PASS] " << name << std::endl;
    } else {
        std::cout << "[FAIL] " << name << " (code " << result << ")" << std::endl;
    }
    return result;
}

uint32_t ValidationRegistry::run(const std::string& name) {
    for (const auto& entry : registry_table()) {
        if (entry.name == name) {
            uint32_t result = RunGuarded(entry);
            if (result == 0) {
                std::cout << "[PASS] " << name << std::endl;
            } else {
                std::cout << "[FAIL] " << name << " (code " << result << ")" << std::endl;
            }
            return result;
        }
    }
    std::cout << "[FAIL] Unknown validation: " << name << std::endl;
    return 1;
}

uint32_t ValidationRegistry::run_all(bool include_slow) {
    uint32_t failures = 0;
    uint32_t ran = 0;
    uint32_t skipped = 0;
    for (const auto& entry : registry_table()) {
        if (!include_slow && entry.tier == ValidationTier::Slow) {
            std::cout << "[SKIP] " << entry.name << " (tier slow; use --all-full)" << std::endl;
            ++skipped;
            continue;
        }
        ++ran;
        uint32_t result = RunGuarded(entry);
        if (result == 0) {
            std::cout << "[PASS] " << entry.name << std::endl;
        } else {
            std::cout << "[FAIL] " << entry.name << " (code " << result << ")" << std::endl;
            ++failures;
        }
    }
    std::cout << (failures == 0 ? "All validations PASSED (" : "Some validations FAILED (")
               << ran - failures << "/" << ran << " passed";
    if (skipped != 0) std::cout << ", " << skipped << " slow-tier skipped";
    std::cout << ")." << std::endl;
    return failures;
}

uint32_t ValidationRegistry::run_shard(bool include_slow, uint32_t index, uint32_t count, bool list_only) {
    namespace cu = engine::cuda;
    std::vector<const RegistryEntry*> roster;
    std::vector<std::string> names;
    for (const auto& entry : registry_table()) {
        if (!include_slow && entry.tier == ValidationTier::Slow) continue;
        roster.push_back(&entry);
        names.push_back(entry.name);
    }
    const ShardPartition part = PartitionShards(names, count);
    uint32_t k = 0;
    for (uint32_t s : part.shard_of) k += (s == index) ? 1u : 0u;
    // A device policy other than cpu sends every name to the GPU: the per-shard device guard is moot then.
    const bool guard = count > 1 && index != part.device_shard && cu::requested_device() == cu::Device::Cpu;
    std::cout << "SHARD " << index << "/" << count << " names=" << k << " of " << names.size()
              << " roster=" << (include_slow ? "all-full" : "all") << " expected_s="
              << (part.load_ms[index - 1] + 500) / 1000 << " device_shard=" << part.device_shard
              << " device_guard=" << (guard ? "on" : "off") << std::endl;
    if (list_only) {
        for (size_t pos = 0; pos < roster.size(); ++pos)
            if (part.shard_of[pos] == index)
                std::cout << roster[pos]->name << "\t[" << ValidationTierName(roster[pos]->tier) << "]" << std::endl;
        return 0;
    }
    uint32_t failures = 0;
    uint32_t ran = 0;
    for (size_t pos = 0; pos < roster.size(); ++pos) {
        if (part.shard_of[pos] != index) continue;
        const RegistryEntry& entry = *roster[pos];
        const bool had_context = cu::context_created();
        ++ran;
        uint32_t result = RunGuarded(entry);
        if (guard && !had_context && cu::context_created()) {
            std::cout << "[SHARD-DEVICE] " << entry.name << " created the CUDA context outside the device shard "
                      << part.device_shard << "/" << count
                      << ": add it to src/hacs/ValidationShardDevice.inc" << std::endl;
            if (result == 0) result = 1;
        }
        if (result == 0) {
            std::cout << "[PASS] " << entry.name << std::endl;
        } else {
            std::cout << "[FAIL] " << entry.name << " (code " << result << ")" << std::endl;
            ++failures;
        }
    }
    std::cout << (failures == 0 ? "All validations PASSED (" : "Some validations FAILED (")
              << ran - failures << "/" << ran << " passed, shard " << index << "/" << count << ")." << std::endl;
    return failures;
}

std::vector<std::string> ValidationRegistry::list() {
    std::vector<std::string> names;
    names.reserve(registry_table().size());
    for (const auto& entry : registry_table()) {
        names.push_back(entry.name);
    }
    return names;
}

std::vector<std::pair<std::string, ValidationTier>> ValidationRegistry::list_with_tiers() {
    std::vector<std::pair<std::string, ValidationTier>> out;
    out.reserve(registry_table().size());
    for (const auto& entry : registry_table()) {
        out.emplace_back(entry.name, entry.tier);
    }
    return out;
}

ValidationTier ValidationRegistry::tier_of(const std::string& name) {
    for (const auto& entry : registry_table()) {
        if (entry.name == name) return entry.tier;
    }
    return ValidationTier::Standard;
}

// Slot-9 per-package registration fragments (Registry/Slot9/WP<n>.cpp), slot-9 WP-1.

// Slot-10 per-package registration fragments (Registry/Slot10/WP<n>.cpp), slot-10 WP-1.

// Slot-11 per-package registration fragments (Registry/Slot11/WP<n>.cpp), slot-11 WP-1.

// Slot-12 per-package registration fragments (Registry/Slot12/WP<n>.cpp), slot-12 WP-1.
// Slot-13 per-package registration fragments (Registry/Slot13/WP<n>.cpp), slot-13 WP-1.

// Slot-14 per-package registration fragments (Registry/Slot14/WP<n>.cpp), slot-14 WP-1.

// Slot-15 per-package registration fragments (Registry/Slot15/WP<n>.cpp), slot-15 WP-1.

// Slot-16 per-package registration fragments (Registry/Slot16/<name>.cpp), slot-16 WP-1.

// Closed slots 9..24: one file Registry/Slot<S>.cpp each (S24 CL-10).
void RegisterSlot9();
void RegisterSlot10();
void RegisterSlot11();
void RegisterSlot12();
void RegisterSlot13();
void RegisterSlot14();
void RegisterSlot15();
void RegisterSlot16();
void RegisterSlot17();
void RegisterSlot18();
void RegisterSlot19();
void RegisterSlot20();
void RegisterSlot21();
void RegisterSlot22();
void RegisterSlot23();
void RegisterSlot24();
void RegisterSlot25();
void RegisterSlot26HeckeZUp();   // slot 26 (open): Registry/Slot26/HeckeZUp.cpp
void RegisterSlot26HeckeZLocalPar();   // slot 26 (open): Registry/Slot26/HeckeZLocalPar.cpp
void RegisterSlot26HeckeZEisUp();   // slot 26 (open): Registry/Slot26/HeckeZEisUp.cpp
void RegisterSlot26HeckeHidaDiscLambda();   // slot 26 (open): Registry/Slot26/HeckeHidaDiscLambda.cpp

void RegisterAllValidations() {
    // ===============================================================    // TIER CLASSIFICATION (Phase-0 owner decision Q13).
    //
    // Thresholds: fast < 0.5 s, standard 0.5 s .. 60 s, slow > 60 s.
    // Times measured 2026-09-05 on the doc-03 Sec. 1.1 machine (Ryzen 9
    // 7950X, WSL2), build `cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo .`
    // (-O2 -g -DNDEBUG, GCC 11.4), ONE PROCESS PER VALIDATION
    // (`./cohomology_app <Name>`), wall clock. They are recorded next to
    // each registration below as a `// x.xx s` comment and are the evidence
    // for the tier chosen -- re-measure and re-tier if a validation's cost
    // changes materially.
    //
    // Of the 32 PRE-EXISTING validations: 21 fast (sum ~2 s) + 10 standard
    // (sum ~73 s) + 1 slow (HeckeBenchmarkSmoke, 64.9 s). Adding Phase 0's
    // SubstrateDifferentialValidation (standard, 0.007 s) makes 33. Measured
    // end to end at RelWithDebInfo: `--all` 1m20s, `--all-full` 2m25s.
    //
    // The only `slow` entry is HeckeBenchmarkSmoke -- it is a pure
    // performance smoke ((ell,n) = (134217757,100), the Q8 baseline), it is
    // the single dominant cost of the suite, and nothing else exceeds 60 s
    // (the next largest are HeckeMorphismRegressionValidation at 23.8 s,
    // BlockFlatMatrixValidation at 23.3 s and HeckeEndToEndValidation at
    // 18.1 s -- all `standard`). NOTE that the Hecke ACCEPTANCE GATE is
    // HeckeEndToEndValidation + HeckeBadPrimeValidation + HeckeBenchmarkSmoke
    // (HeckeBenchmarkSmoke is slow), so the gate must be run as `--all-full` (or by naming the
    // three), never as plain `--all`.
    //
    // CONVENTION FOR THE REST OF PHASE 0 (integrator): the golden-output
    // regression `GoldenRegressionValidation` registers as
    // ValidationTier::Standard; the determinism / parameter grids register
    // as ValidationTier::Slow.
    // ========================================================    // The six pre-existing validations (Stage S0 preconditions).
    ValidationRegistry::register_validation("mod_arithmetic_validation", &Wrap_mod_arithmetic_validation,
                                            ValidationTier::Fast);  // measured 0.061 s
    ValidationRegistry::register_validation("MatrixValidation", &MatrixValidation,
                                            ValidationTier::Fast);  // measured 0.003 s
    ValidationRegistry::register_validation("FlatMatrixValidation", &Wrap_FlatMatrixValidation,
                                            ValidationTier::Fast);  // measured 0.003 s
    // (S24 CL-L2) legacy-gated whole name: [SKIP] in an OC_ENABLE_LEGACY=OFF build.
#if OC_ENABLE_LEGACY
    ValidationRegistry::register_validation("BlockFlatMatrixValidation", &Wrap_BlockFlatMatrixValidation,
                                            ValidationTier::Standard);  // measured 23.3 s
#else
    ValidationRegistry::register_validation("BlockFlatMatrixValidation", OC_LEGACY_VALIDATION(BlockFlatMatrixValidation),
                                            ValidationTier::Standard);  // measured 23.3 s
#endif
    ValidationRegistry::register_validation("SimplicialCohomologyValidation", &SimplicialCohomologyValidation,
                                            ValidationTier::Fast);  // measured 0.003 s
    ValidationRegistry::register_validation("LocalSystemCohomologyValidation", &LocalSystemCohomologyValidation,
                                            ValidationTier::Fast);  // measured 0.004 s

    // matrix-engine Phase 1 -- item 7.4: the vacuous-relation short-circuit is
    // exactly value-preserving, and the odd-n H^* = 0 semantics survive it.
    ValidationRegistry::register_validation("VacuousRelationValidation", &VacuousRelationValidation,
                                            ValidationTier::Standard);

    // matrix-engine Phase 2 -- arithmetic primitive layer (plan D2/D3/D6).
    // ModContextValidation: every ModCtx primitive vs the legacy mod_* /
    // mod_*64 at all regime boundaries (1e6 fixed-seed samples per prime) plus
    // the k_max tightness / overflow-guard proof.  KernelBitIdentityValidation:
    // gemm/gemv/trmm/transpose/... vs the naive RefLinAlg oracle and vs the OLD
    // FlatMatrix::operator*, both element widths, all layouts, both microkernel
    // paths (AVX-512 and portable) bit-identical.
    ValidationRegistry::register_validation("ModContextValidation", &ModContextValidation,
                                            ValidationTier::Standard);  // measured 0.83 s
    ValidationRegistry::register_validation("KernelBitIdentityValidation", &KernelBitIdentityValidation,
                                            ValidationTier::Standard);  // measured 14.9 s
    // DenseElimValidation (plan D5/D5b): the blocked in-buffer elimination --
    // RREF, pivots, kernel/image bases, transform U, inverse, multi-RHS solve,
    // complement/projector, block-elimination step -- ENTRYWISE against the
    // RefLinAlg oracle and against the OLD FlatMatrix / factor_basis_block code
    // (reached through engine::elim::ScopedLegacy), both element widths, all
    // regime boundary primes, plus a sweep over the blocking parameters.
    ValidationRegistry::register_validation("DenseElimValidation", &DenseElimValidation,
                                            ValidationTier::Standard);  // measured 37 s
    // DenseElimPipelineValidation: the same entrywise comparison on the REAL
    // matrices -- A_Cq of the SL(2,Z) model at n_sym = 128 (A_C1 is 1677 x
    // 1419 of rank 258) through FlatMatrix, BlockFlatMatrix::flat_kernel and
    // RefLinAlg, legacy vs engine -- plus the whole compute_cohomology() run
    // (C/Z/B/H_eq bases and projectors) legacy vs engine at n_sym = 32.
    ValidationRegistry::register_validation("DenseElimPipelineValidation", &DenseElimPipelineValidation,
                                            ValidationTier::Standard);

    // matrix-engine Phase 3 -- the block-sparse data model (plan D3) and the
    // block structured elimination (plan D5 / D5b).  BlockMatValidation: pool
    // dedup / refcount / collisions, kinds, CSR determinism, builder folding,
    // non-uniform + zero-width grids, views, block GEMM / gemm_dense / add /
    // scale / equal vs dense arithmetic, the to_flat / from_flat L1 identity.
    // BlockEliminationValidation: block RREF / kernel / image / factor_basis /
    // solve ENTRYWISE vs engine::elim on the flattened matrix and vs the
    // legacy BlockFlatMatrix / factor_basis_block / change_of_basis_fm, on
    // structural-shortcut inputs and real-shaped grids.
    ValidationRegistry::register_validation("BlockMatValidation", &BlockMatValidation,
                                            ValidationTier::Standard);
    ValidationRegistry::register_validation("BlockEliminationValidation", &BlockEliminationValidation,
                                            ValidationTier::Standard);
    // BlockModelPipelineValidation: the block model on the real A_C1 (n_sym =
    // 128, both short-circuit modes) and the real W^1 / H^1 factor problems,
    // entrywise vs legacy BlockFlatMatrix::flat_kernel / factor_basis_block,
    // with the model-alone timings printed next to the legacy ones.
    ValidationRegistry::register_validation("BlockModelPipelineValidation", &BlockModelPipelineValidation,
                                            ValidationTier::Standard);
    // Phase 3 Stage B: compute_cohomology() itself on the block model.  Engine
    // arm vs the kept legacy arm (ScopedLegacyCohomologyPath) entrywise on a
    // small SL(2,Z) grid (odd n, ell = 2, 2^32-5, the uint64_t regime, both
    // short-circuit settings, a relation-free system), the *_block accessors
    // against the FlatMatrix ones (the L1 to_flat contract, plan 7.20), and
    // algebraic certificates A_Cq X = 0, P_q H = I, P_q B = 0.
    ValidationRegistry::register_validation("CohomologyBlockModelValidation", OC_LEGACY_VALIDATION(CohomologyBlockModelValidation),
                                            ValidationTier::Standard);
    // matrix-engine Phase 3 Stage B -- BarCocycleH1.cpp's hand-rolled
    // RREF-with-RHS solver was absorbed into engine::elim::solve_multi_rhs.
    // The plan makes that absorption CONDITIONAL on the engine reproducing the
    // old pivot rule and free-variable convention exactly; this compares the
    // frozen pre-Stage-B body against the engine route over a fixed-seed grid
    // that includes rank-deficient column sets and out-of-span right-hand
    // sides (which the two production call sites themselves never reach).
    ValidationRegistry::register_validation("BarCocycleSolverValidation", OC_LEGACY_VALIDATION(BarCocycleSolverValidation),
                                            ValidationTier::Fast);

    // matrix-engine Phase 5 -- device kernels vs engine::kernels (memcmp) and
    // the residency / pinned-pool contract. Both return 0 with "[SKIP]" when
    // engine::cuda::available() is false.
    ValidationRegistry::register_validation("CudaBitIdentityValidation", &CudaBitIdentityValidation,
                                            ValidationTier::Standard);
    ValidationRegistry::register_validation("CudaTransferValidation", &CudaTransferValidation,
                                            ValidationTier::Standard);

    // matrix-engine Phase 4 -- the tape engine (plan D4): Program / Recorder,
    // passes 1-6, wavefront Executor on the persistent thread pool, arena,
    // program cache, panel-driven elimination driver.  EngineTapeValidation:
    // record -> passes -> execute equals direct blk:: / kernels:: execution on
    // pipeline-shaped programs, every pass toggled, cache replay == fresh
    // recording, a changed pattern hash misses.  EngineTapeEliminationValidation:
    // tape RREF / kernel / image / factor_basis / solve entrywise vs
    // engine::blk on BlockEliminationValidation's grids.
    // EngineDeterminismCoreValidation: the same programs under ENGINE_THREADS
    // in {1, 4, 16, 32} x ENGINE_BATCH in {0, 1}, FNV-64 hashes identical.
    // EngineArenaValidation: lifetimes, arena vs owned storage, poisoned
    // arena, replay reuse (run it on the Check build for ASan/UBSan).
    // EngineCoreScalingValidation: the core's own scaling table (reported,
    // identity gated).
    ValidationRegistry::register_validation("EngineTapeValidation", &EngineTapeValidation,
                                            ValidationTier::Standard);
    ValidationRegistry::register_validation("EngineTapeEliminationValidation", &EngineTapeEliminationValidation,
                                            ValidationTier::Standard);
    ValidationRegistry::register_validation("EngineDeterminismCoreValidation", &EngineDeterminismCoreValidation,
                                            ValidationTier::Standard);
    ValidationRegistry::register_validation("EngineArenaValidation", &EngineArenaValidation,
                                            ValidationTier::Standard);
    ValidationRegistry::register_validation("EngineCoreScalingValidation", &EngineCoreScalingValidation,
                                            ValidationTier::Standard);
    // Phase 4 Stage B (consumer wiring): EngineTapeConsumerValidation -- the
    // consumers on the tape vs the direct engine::blk path on their own stored
    // objects, entrywise; program-cache replay == fresh recording == forced
    // fresh compile, also on another pool size; a changed tau pattern (another
    // Hecke prime; a mod-ell triangle-fill fallback point when the probe finds
    // one) MISSES the cache.  EngineDeterminismValidation -- the golden grid's
    // small/medium cells IN PROCESS under ENGINE_THREADS in {1,4,16,32} x
    // ENGINE_BATCH in {0,1}: every golden object == the reference (both tiers)
    // and every arm's digest identical.  Both `standard` (measured ~12 s and
    // ~45 s on the default build).
    ValidationRegistry::register_validation("EngineTapeConsumerValidation", &EngineTapeConsumerValidation,
                                            ValidationTier::Standard);
    ValidationRegistry::register_validation("EngineDeterminismValidation", &EngineDeterminismValidation,
                                            ValidationTier::Standard);

    // matrix-engine Phase 5 Stage B -- the tape's device arm.  All three
    // print "[SKIP]" and return 0 when engine::cuda::available() is false.
    // EngineCrossDeviceValidation: the core's pipeline-shaped programs
    // (Theta-like, block x block, the elimination driver, gather / scatter /
    // COMPARE, the op set, k > k_max GEMMs, every regime) under
    // ENGINE_DEVICE in {cpu, gpu, hybrid} x threads x batching x graph
    // capture, FNV-64 hashes identical, with ENGINE_VERIFY_DEVICE re-execution
    // on every GPU node.  EngineCudaMemoryValidation: the pinned pool never
    // exceeds its cap, pageable_copies stays 0 across a run, the device arena
    // returns to its baseline after the executor's device state is released.
    // EngineDeviceTrapValidation (slow): the "unblocked-kernel trap" evidence
    // table -- tape elimination / Theta-like batches / dense 4096^3 under
    // cpu vs gpu vs hybrid, k-blocking chunk count and the k_max assertion,
    // the Phase 5 exit-criterion figures; gated on identity only.
    ValidationRegistry::register_validation("EngineCrossDeviceValidation", &EngineCrossDeviceValidation,
                                            ValidationTier::Standard);
    ValidationRegistry::register_validation("EngineCudaMemoryValidation", &EngineCudaMemoryValidation,
                                            ValidationTier::Standard);
    ValidationRegistry::register_validation("EngineDeviceTrapValidation", &EngineDeviceTrapValidation,
                                            ValidationTier::Slow);

    // Hecke validations (Stage S1+).
    ValidationRegistry::register_validation("HeckeExactIntSmokeValidation", &HeckeExactIntSmokeValidation,
                                            ValidationTier::Fast);  // measured 0.006 s
    ValidationRegistry::register_validation("HeckeFpFilterValidation", &HeckeFpFilterValidation,
                                            ValidationTier::Standard);  // measured 1.28 s
    ValidationRegistry::register_validation("HeckeTessellationSmokeValidation", &HeckeTessellationSmokeValidation,
                                            ValidationTier::Fast);  // measured 0.011 s

    // Stage S2V -- independent validation of Stage S2.
    ValidationRegistry::register_validation("HeckeTessellationValidation", &HeckeTessellationValidation,
                                            ValidationTier::Fast);  // measured 0.077 s

    // Stage S3 -- HeckeCosetTables Definition-of-Done battery (self_check()
    // over the DoD parameter grid). NOTE: this is distinct from the
    // Stage S3V validation `HeckeCosetTablesValidation` (oracle O16, the
    // worked p=2 contract tables of [C, sec:example]), which is written and
    // registered separately by the independent S3V validation agent.
    ValidationRegistry::register_validation("HeckeCosetTablesSelfCheckValidation",
                                             &HeckeCosetTablesSelfCheckValidation,
                                            ValidationTier::Fast);  // measured 0.192 s

    // Stage S3V -- INDEPENDENT validation of Stage S3 against the worked
    // p=2 contract example ([C, sec:example], oracle O16). See
    // Hecke/HeckeCosetTablesValidation.h for the independence statement.
    ValidationRegistry::register_validation("HeckeCosetTablesValidation",
                                             &HeckeCosetTablesValidation,
                                            ValidationTier::Fast);  // measured 0.005 s

    // Stage S4 -- HeckeTauTable Definition-of-Done battery: constructs and
    // certifies the g-twisted chain datum tau across the DoD grid (the
    // constructor already enforces the MANDATORY certification of
    // [C, obl:tau] by throwing on failure). NOTE: distinct from the Stage
    // S4V validation `HeckeTauTableValidation` (the independent checker-only
    // re-implementation of [C, obl:tau]), which is written and registered
    // separately by the independent S4V validation agent.
    ValidationRegistry::register_validation("HeckeTauTableSelfCheckValidation",
                                             &HeckeTauTableSelfCheckValidation,
                                            ValidationTier::Fast);  // measured 0.218 s

    // Stage S4V -- INDEPENDENT validation of Stage S4 ([C, obl:tau] /
    // [C, obl:backend]). See Hecke/HeckeTauTableValidation.h for the
    // independence statement: re-derives the boundary operator and the (T1)
    // twisted-equivariance translation from the contract, never calls
    // HeckeTauTable::certify() or its private helpers.
    ValidationRegistry::register_validation("HeckeTauTableValidation",
                                             &HeckeTauTableValidation,
                                            ValidationTier::Fast);  // measured 0.13 s

    // gl2_contract.tex chapter 9 (the involution iota, v0.12.0): the two
    // clauses of obl:iota that are predicates on finite exact integer data.
    // (A) obl:iota(i) -- the shipped exact vertex coordinates realise
    // z -> -conj(z) as lem:iotamodel's vertex permutation, with its sign table
    // reproduced entry for entry -- is GATING. (B) obl:iota(iv) -- whether the
    // canonical tau of alg:tau satisfies iota_X tau iota_X == tau, the
    // hypothesis of prop:iotacommutes(3b) and the only route to
    // [iota^*, T_g] = 0 at ell in {2,3} -- is RECORDED per that clause, and
    // since the owner's re-baseline of 2026-09-16 (iota_K-equivariant
    // tie-break + tree-routed path templates) it holds on EVERY table of the
    // grid and is therefore ALSO GATING for the binding policy: a single
    // differing slot on the canonical tau is a failure. The legacy policy
    // (the pre-re-baseline tau) is run alongside and its verdict -- false,
    // 3600 of 10764 binding slots -- is recorded, never a failure.
    ValidationRegistry::register_validation("HeckeIotaTauSymmetryValidation",
                                             &HeckeIotaTauSymmetryValidation,
                                            ValidationTier::Fast);  // measured 0.5 s (three tau policies)

    // gl2_contract.tex chapter 9 (v1.2.0), the remaining clauses obl:iota(ii),
    // (iii), (v): iota^* is BUILT as a cochain map from def:iota alone (live
    // rho_n of diag(-1,1), iota_K with lem:iotamodel's signs re-derived, and
    // the global sign of def:iota(iv): iota^* := -iota^*_geo, the geometric
    // pullback negated -- owner decision OQ3 of 2026-09-16,
    // docs/implementation/oq3_resolution.md) and checked -- GATING -- to be a
    // local-system automorphism entrywise on all 31 relations, to commute
    // with d, to square to the identity on C^q and on H^q_eq, to commute with
    // A^{(1)}(T_p), to equal -iota^*_geo entrywise, and to be -1 on
    // H^0(Gamma; F_ell) at n = 0 (the degree-0 anchor that pins the global
    // sign). The definition-consistency oracle "the Eisenstein class lies in
    // H^1_{eq,+}" (obl:iota(v), rmk:iotasign; TRUE BY DEFINITION since v1.2.0)
    // is evaluated by two independent routes -- the T_p-eigenline of
    // eigenvalue 1+p^{n+1} and the class visible under restriction to the
    // cusp -- and is GATING on every cell where a route is defined, together
    // with the agreement of the routes: a '-' anywhere would mean the
    // geometric sign is not uniform across cells and the definition is
    // ill-posed. RECORDED: the geometric sign (`minus`, the v1.1.1 finding of
    // docs/implementation/oq3_iota_sign_verification.md, now read as a
    // statement about iota^*_geo) and the dim H^1_+/- split against
    // (d_{n+2}, d_{n+2}-1), the mod-ell Eichler-Shimura prediction.
    // Prints one `IOTA_SIGN eisenstein=plus (geometric=minus) ...` line.
    ValidationRegistry::register_validation("HeckeIotaSignValidation",
                                             &HeckeIotaSignValidation,
                                            ValidationTier::Standard);

    // Stage S5 -- InducedLocalSystem smoke validation: builds L_{Gamma_0(2)}
    // ([C, def:inducedls] / [C, eq:MH]) for (n,ell) in {(0,5), (2,5)} and
    // runs EquivariantCohomologyOnK UNCHANGED on it ([C, prop:lhcompliant]).
    // NOTE: distinct from the Stage S5V validation `HeckeShapiroValidation`
    // (the independent Shapiro dimension oracle, [C, obl:shapiro]), which is
    // written and registered separately by the independent S5V validation
    // agent.
    ValidationRegistry::register_validation("HeckeInducedLocalSystemSmokeValidation",
                                             &HeckeInducedLocalSystemSmokeValidation,
                                            ValidationTier::Fast);  // measured 0.023 s

    // Stage S5V -- INDEPENDENT Shapiro validation of Stage S5 ([C, obl:shapiro],
    // [C, prop:lhcompliant], oracle O11, and the Q2 ruling of the plan's
    // Sec. 0.1). Written by the S5V validation agent directly from
    // docs/latex/hecke_action.tex and classical genus/cusp data of
    // X_0(2)/X_0(3), without reading HeckeInducedLocalSystemSmokeValidation.cpp's
    // expected values. See Hecke/HeckeShapiroValidation.h/.cpp for the
    // independence statement and the full derivation.
    ValidationRegistry::register_validation("HeckeShapiroValidation",
                                             &HeckeShapiroValidation,
                                            ValidationTier::Fast);  // measured 0.029 s

    // Stage S6 -- self-check that the per-simplex sparse-block-row generalization
    // of EquivariantCochainComplexMorphismOnK (the code-level v5 fix, [C, rmk:nogo])
    // reproduces the legacy constant-fiber-matrix induced maps byte-for-byte. See
    // Hecke/HeckeMorphismGeneralizationValidation.h. (The INDEPENDENT bit-identity
    // regression against the pre-S6 commit is Stage S6V's HeckeMorphismRegressionValidation.)
    ValidationRegistry::register_validation("HeckeMorphismGeneralizationValidation",
                                            &HeckeMorphismGeneralizationValidation,
                                            ValidationTier::Fast);  // measured 0.006 s

    // Stage S6V -- INDEPENDENT validation agent for Stage S6: bit-identity
    // regression of get_induced_H_eq_maps() (all cases, all degrees) against
    // reference values captured on the pre-S6 parent commit
    // 41df5e278460f4c1a9f7ee3270878052a0802983, plus a 100-fixed-seed
    // legacy-ctor-vs-FiberBlockTable equivalence check on the S6 tree. See
    // Hecke/HeckeMorphismRegressionValidation.h for the full oracle statement
    // and the capture method/commit hashes.
    ValidationRegistry::register_validation("HeckeMorphismRegressionValidation",
                                             &HeckeMorphismRegressionValidation,
                                            ValidationTier::Standard);  // measured 23.8 s

    // Stage S7 -- HeckeOperatorOnK Definition-of-Done smoke: the fused
    // pipeline Theta_g ([C, def:xi]/[C, thm:xi]) with matrix levels (a)+(b)
    // ([C, alg:assembly]); asserts A^(0)(T_2) = (3) at (p,ell,n) = (2,5,0)
    // and A^(1)(T_2) = (4) at (2,5,2), the O12(a) two-route bit-identity,
    // active runtime assertions (i)/(ii)/(iii), and parseable to_json()
    // output. NOTE: distinct from the Stage S7V validation
    // `HeckeOracleValidation` (the independent primary oracle battery
    // O1-O8/O14), which is written and registered separately by the
    // independent S7V validation agent.
    ValidationRegistry::register_validation("HeckeOperatorSmokeValidation",
                                             &HeckeOperatorSmokeValidation,
                                            ValidationTier::Fast);  // measured 0.008 s

    // Stage S7V -- INDEPENDENT primary oracle battery for HeckeOperatorOnK
    // (Stage S7): oracles O1,O2,O3,O4,O5,O6(hard part),O7,O8(charpoly-only),
    // O14, gated per plan amendment A-F4 (Sec 0.1). Written by the S7V
    // validation agent directly from docs/latex/hecke_action.tex and
    // docs/implementation/hecke_action_implementation_plan.md, without
    // reading HeckeOperatorSmokeValidation.cpp's expected values -- see
    // Hecke/HeckeOracleValidation.h/.cpp for the independence statement and
    // the full derivation of every expected char-poly/dimension.
    ValidationRegistry::register_validation("HeckeOracleValidation",
                                             &HeckeOracleValidation,
                                            ValidationTier::Standard);  // measured 1.87 s

    // Stage S8 -- the two independent cross-check routes of [C, obl:agreement]:
    // (i) the three-map factorization omega.beta.alpha == Theta_g asserted
    // BIT-identically at cochain level on all of C^q(K;V), q = 0,1,2 (with
    // the A-F3 localizing assertions active first: beta maps L'-equivariant
    // cochains to L''-equivariant cochains, and beta.alpha == Xi_g), and
    // (ii) the corrected bar-cocycle H^1 route ([C, prop:barformula]) whose
    // Hecke matrix must match charpoly AND minpoly of A^(1)(T_p), across
    // (p, ell, n) in {2,3} x {5,13,691} x {0,2,10}. NOTE: distinct from the
    // Stage S8V validation `HeckeIndependenceValidation` (oracle O12 in
    // full), which is written and registered separately by the independent
    // S8V validation agent.
    ValidationRegistry::register_validation("HeckeCrossCheckValidation",
                                             &HeckeCrossCheckValidation,
                                            ValidationTier::Standard);  // measured 1.07 s

    // Stage S8V -- INDEPENDENT agreement & independence validation: oracle
    // O12 in full -- (a) route bit-identity (re-exercised via apply_Theta on
    // the full standard basis), (b) custom Gamma_0(p) coset representatives
    // (bit-identical Theta_flat, [C, thm:welldef](a)), (c) TauVertexPolicy
    // FixedV6 and GeometricNearestLegacy vs GeometricNearest (equal A^(q),
    // [C, thm:welldef](b)), (d) the
    // double-coset representative g -- documented as NOT fully reachable via
    // the public API, with the explicitly-permitted weaker reachable form
    // asserted instead (uniform left-translate of every s_i by a single
    // fixed gamma in Gamma_0(N)), and (e) HECKE_PURE_INTEGER=ON vs OFF
    // (verified externally by diffing this validation's stdout across both
    // CMake configs). See Hecke/HeckeIndependenceValidation.h for the full
    // statement and the precise documentation of what is/isn't reachable.
    ValidationRegistry::register_validation("HeckeIndependenceValidation",
                                             &HeckeIndependenceValidation,
                                            ValidationTier::Standard);  // measured 0.564 s

    // Stage S9 -- HeckeOperatorFamily Definition-of-Done smoke: the Hecke
    // algebra structure ([C, thm:relations] recursion T_{p^{r+1}} = T_p
    // T_{p^r} - p^{n+1} T_{p^{r-1}}, T_{p^0} = I; [C, prop:diamond]'s
    // central scalar T(p,p) = p^n I) built on top of Stage S7's
    // HeckeOperatorOnK pipeline. Asserts T_p_pow(p,1) == T_p(p) and
    // T_pp(p) == p^n*I for p in {2,3}, ell=13, n in {0,2,10}, plus the
    // independent recursion-vs-direct-pipeline identity
    // T_p_pow(2,2) == T_p_pow_direct(2,2) at (ell,n)=(13,10) (an O9-style
    // check, valid for ell >= 5, [C, obl:algebra] hypothesis (H3)). NOTE:
    // distinct from the (separately authored) Stage S9V validation
    // `HeckeRelationsValidation` (the full independent O9/O10 battery,
    // including the p == ell bad-prime cases gated per amendment A-F1),
    // which is written and registered separately by the independent S9V
    // validation agent.
    ValidationRegistry::register_validation("HeckeOperatorFamilySmokeValidation",
                                             &HeckeOperatorFamilySmokeValidation,
                                            ValidationTier::Fast);  // measured 0.475 s

    // Stage S9V -- INDEPENDENT Hecke-relations validation: oracles O9 (the
    // internal-consistency Hecke-algebra identities -- commutativity,
    // T_4 = T_2^2 - 2^{n+1} I via the FULL, independently-recomputed
    // T_p_pow_direct route, T_6 = T_2 T_3, T(p,p) = p^n I, and the
    // primitive-vs-full decomposition witness -- contract-backed for ALL
    // ell >= 5, [C, obl:algebra]) and O10 (the bad-prime p == ell case,
    // gated per plan amendment A-F1: hard-asserted at p=ell=5, recorded
    // ONLY -- not asserted -- at p=ell in {2,3}). Written by the S9V
    // validation agent directly from docs/latex/hecke_action.tex and the
    // implementation plan's Sec 0.1 (A-F1) / Sec 0.2 (the T_{p^r}
    // full-operator ruling) / "S9V -- Hecke-relations validation" stage,
    // without reading HeckeOperatorFamilySmokeValidation.cpp's expected
    // values -- see Hecke/HeckeRelationsValidation.h for the full
    // independence statement and derivation.
    ValidationRegistry::register_validation("HeckeRelationsValidation",
                                             &HeckeRelationsValidation,
                                            ValidationTier::Standard);  // measured 0.524 s

    // Stage S10 -- F_ell[X] toolkit + single-operator level-(c)
    // block-diagonalization Definition-of-Done smoke: 200 fixed-seed
    // pseudorandom d <= 8 matrices over ell in {5, 13, 101, 134217757}
    // assembled from known block forms by random conjugation (recovered
    // factor multiset / Jordan partitions / residue degrees must match the
    // constructed truth, certify() == 0, conjugated(p) exactly
    // block-diagonal), the companion matrix of (X^2+X+1)^2 over F_5 (one
    // block, partition (2), residue degree 2), and FpPoly corner batteries
    // (division-free Berkowitz at ell <= d, the inseparable f' = 0
    // ell-th-root path, the ell = 2 trace-map equal-degree split). All
    // factorizations/decompositions carry deterministic UNCONDITIONAL
    // certificates -- the randomized Cantor-Zassenhaus search is never
    // trusted (plan Sec. 3 "Certification"). NOTE: distinct from the Stage
    // S10V validation `HeckeBlockFormValidation` (oracles O6/O8), which is
    // written and registered separately by the independent S10V validation
    // agent.
    ValidationRegistry::register_validation("HeckeDecompositionSmokeValidation",
                                             &HeckeDecompositionSmokeValidation,
                                            ValidationTier::Fast);  // measured 0.022 s

    // Stage S10V -- INDEPENDENT decomposition validation of Stage S10's
    // HeckeModuleDecomposition (level-(c) block-diagonalization) on the two
    // REAL non-semisimple / non-split cases: oracle O6 (n=10, ell=691, the
    // Ramanujan-congruence non-semisimplicity, amendment A-F2: HARD-asserts
    // rank(A+24I)=1, i.e. Jordan partition (2,1) not (3)) and oracle O8
    // (n=22, ell in {101,10007}, the weight-24 Galois-conjugate-pair
    // split-vs-nonsplit branch, selected at runtime via the Legendre symbol
    // of 144169 computed by Euler's criterion). Written by the S10V
    // validation agent directly from docs/latex/hecke_action.tex and the
    // implementation plan's amendment A-F2 / "S10V -- Decomposition
    // validation" stage, without reading HeckeDecompositionSmokeValidation.cpp's
    // expected values, and RE-VERIFYING every certificate (reconstruction,
    // irreducibility) with its own from-scratch F_ell[X] toolkit rather than
    // calling HeckeModuleDecomposition::certify() -- see
    // Hecke/HeckeBlockFormValidation.h/.cpp for the full independence
    // statement and derivation.
    ValidationRegistry::register_validation("HeckeBlockFormValidation",
                                             &HeckeBlockFormValidation,
                                            ValidationTier::Standard);  // measured 0.876 s

    // Stage S11 -- HeckeModuleDecomposition's multi-operator path
    // (ComputeMultiOp(), the simultaneous MaxSpec(T) refinement for a
    // commuting family) plus Hecke/HeckeReport's JSON writer,
    // Definition-of-Done smoke: at (ell,n)=(13,10), family {T_2,T_3}: dim
    // H^1 = 3, T_2/T_3 charpolys cross-checked against the embedded
    // eigenvalue data, decomposition = {dim 2 cuspidal (Delta system,
    // isotypic multiplicity 2, RECORDED not chased for a further split),
    // dim 1 Eisenstein (eigen-system T_2|->8, T_3|->10, ASSERTED)}; at
    // (ell,n)=(691,10): a single dim-3 block re-deriving the S10V O6 result
    // (char_poly=(X+24)^3, min_poly=(X+24)^2) now jointly with T_3. The JSON
    // report round-trips through an embedded, from-scratch mini JSON parser.
    // NOTE: distinct from the Stage S11V validation `HeckeSimultaneousValidation`
    // (an independent re-check reading only the JSON, plus the
    // (10007,22)/{T_2,T_3} case), which is written and registered separately
    // by the independent S11V validation agent.
    ValidationRegistry::register_validation("HeckeSimultaneousSmokeValidation",
                                             &HeckeSimultaneousSmokeValidation,
                                            ValidationTier::Fast);  // measured 0.157 s

    // Stage S11V -- INDEPENDENT validation of the simultaneous (MaxSpec(T))
    // decomposition and its HeckeReport JSON export: re-derives the expected
    // eigen-systems/block structures from classical modular-forms theory
    // (Ramanujan's Delta, the classical Ramanujan congruence mod 691, and the
    // weight-24 LMFDB data), and re-asserts them by reading ONLY the JSON
    // string emitted by hecke::BuildHeckeReportJson() through its OWN
    // from-scratch mini-parser and its OWN from-scratch F_ell[X] toolkit --
    // never reading HeckeSimultaneousSmokeValidation.cpp's expected values,
    // and never calling hecke::HeckeModuleDecomposition::certify() as an
    // oracle. Covers (ell,n) in {(13,10), (691,10), (10007,22), (101,22)},
    // the last two selected/confirmed via a runtime Legendre-symbol
    // computation (Euler's criterion) that also branches the block-structure
    // assertion (two dim-2 blocks when (144169|ell)=+1, one dim-4
    // residue-degree-2 CERTIFIED block when (144169|ell)=-1). See
    // Hecke/HeckeSimultaneousValidation.h for the full independence
    // statement and derivation.
    ValidationRegistry::register_validation("HeckeSimultaneousValidation",
                                             &HeckeSimultaneousValidation,
                                            ValidationTier::Standard);  // measured 1.57 s

    // Stage S12 -- the ACCEPTANCE GATE: three aggregated validations exercising
    // every contract obligation and oracle with the correct gating, plus a
    // performance smoke.
    //   * HeckeEndToEndValidation -- the full oracle matrix over
    //     p x ell x n x q, with A-F4 (Dickson-loci-recorded) / FIX-negI (odd-n
    //     H^*=0) / A-F1 (bad-prime) / auditor-verified-constant gating; prints
    //     an obligation->test-IDs coverage table; return value = aggregate
    //     failure count.
    //   * HeckeBadPrimeValidation -- O10 bad-prime settings (p = ell in {2,5}).
    //   * HeckeBenchmarkSmoke -- (ell,n)=(134217757,100), p=2 performance smoke
    //     (no wall-time assertion; wall time printed as the Q8 baseline).
    ValidationRegistry::register_validation("HeckeEndToEndValidation",
                                             &HeckeEndToEndValidation,
                                            ValidationTier::Standard);  // measured 18.1 s
    ValidationRegistry::register_validation("HeckeBadPrimeValidation",
                                             &HeckeBadPrimeValidation,
                                            ValidationTier::Fast);  // measured 0.205 s
    ValidationRegistry::register_validation("HeckeBenchmarkSmoke",
                                             &HeckeBenchmarkSmoke,
                                            ValidationTier::Slow);  // measured 64.9 s

    // Stage S13 -- extended coefficient regime (ell < 2^64): the full pipeline
    // over the __int128-backed 64-bit backend at two verified large primes
    // (~2^34 and 2^61-1), asserting O1 / dims / charpoly identities and the
    // two-route bit-identity + tau certification at 64-bit ell.
    ValidationRegistry::register_validation("HeckeExtendedRegimeValidation",
                                             &HeckeExtendedRegimeValidation,
                                            ValidationTier::Fast);  // measured 0.206 s

    // hecke-extensions phases E0a + E0b -- the locality certificate of
    // [E] Cert. 5.2 / Claims 3.1, 3.3, read-only (E0a) and INSIDE THE
    // REFINEMENT LOOP (E0b, slot 2 WP-1), plus the mathematical settings layer
    // Hecke/HeckeSettings ([X] Sec. 1.5 `locality`).
    //   arm 1  Claim 3.1's non-local families: at a forced budget of ONE
    //          candidate the block is not split and exits BOUNDED (a FROZEN
    //          copy of the pre-fix acceptance predicate still accepts it, the
    //          certificate still rejects it, and compute() honestly flags
    //          local_certified = false with a resumable state); at the default
    //          budget compute() SPLITS it into its two maximal ideals and
    //          certifies both; and run(1) + resume_locality(64) equals a fresh
    //          run(64) entrywise modulo budget_history ([X] Sec. 1.3.1 R3).
    //   arm 2  genuinely local algebras, accepted, with the loop's stored
    //          certificate equal to the read-only verdict.
    //   arm 3/4 the Claim 3.3 Teichmueller trap and the k = 1 ANALYTIC path
    //          (w_hat never formed), matched field for field.
    //   arm 6  HeckeSettings: defaults, parsing/validation, the canonical text
    //          and its fingerprint, `locality = strict` COMPUTING (it is
    //          implemented since slot 3 WP-3; the named
    //          not_implemented_in_this_build error survives for the next value
    //          a build cannot honour), and the thread-local propagation onto
    //          run_concurrently() tasks.
    //   arm 5  THE GRID ARM: every decomposition the golden harness builds is
    //          recomputed, every block certified, the loop's own certificate
    //          compared with the read-only verdict, and all four golden
    //          decomposition hashes re-checked against GoldenReference.inc.
    //   arm 7  (slot 3 WP-3, E0b-S) STRICT MODE -- [X] Sec. 3.1 steps 1-6,
    //          [E] Cert. `sec:localcert` "Strict mode": the (2, 30) covering
    //          example built from a CERTIFIED degree-30 irreducible (budget
    //          mode `local = false` / bound 30, strict `local = true` / f = 30
    //          / phase_reached = residue_search, in ~0.1 s), the Berlekamp
    //          split path at a forced budget of one candidate, GRID IDENTITY
    //          (the whole sweep re-run under `locality = strict`: the same
    //          four hashes, phase_reached = linear on every block),
    //          interruption + resume-invariance, the fingerprint, the
    //          deterministic lexicographic fallback, a five-way tamper arm on
    //          certify()'s codes 20 / 23 / 25 / 26 / 30, and a concurrent
    //          budget-vs-strict arm (what the CheckThread run exercises).
    // `standard`: the grid arm is a full golden pass, and arm 7 re-runs it
    // under strict (~8 s in total) -- which is exactly what it is for: its
    // verdict is what makes E0b value-preserving and E0b-S byte-identical to
    // it wherever budget mode certifies.
    ValidationRegistry::register_validation("HeckeLocalityValidation",
                                             &HeckeLocalityValidation,
                                            ValidationTier::Standard);
    // The (160, 691) probe is RECORDED, NOT ASSERTED ([Q17]), and expensive, so
    // it is BOTH `slow` (skipped by `--all`) and interlocked on
    // HECKE_LOCALITY_PROBE=1 (so `--all-full`, `Check` and cross_config.sh only
    // see the SKIP line). It is run once, by hand, on an optimised binary.
    ValidationRegistry::register_validation("HeckeLocalityProbe",
                                             &HeckeLocalityProbe,
                                            ValidationTier::Slow);  // interlocked

    // hecke-extensions slot 1, WP-2 -- hecke_ext::IdentityCertificate (the
    // E0a subset of [X Sec.1.3]'s IdentityCertificate: Eichler-Shimura in
    // its dim-H^0-robust form, Hecke commutativity, d_2 = 0, and the p^2
    // recursion against the INDEPENDENT double-coset construction).
    // `fast`: ten small pipeline points plus four tamper cases, well under
    // a second.
    ValidationRegistry::register_validation("HeckeIdentityValidation",
                                             &HeckeIdentityValidation,
                                            ValidationTier::Fast);

    // Phase 0 -- SubstrateDifferentialValidation (01 Sec. 7 gap #9): the
    // three matrix substrates (Matrix / FlatMatrix / BlockFlatMatrix) are
    // differentially compared against each other on RREF / rank / kernel /
    // image, plus cache-coherence sequences and algebraic identities, at a
    // prime ladder that spans the uint32 regime and (via the 64-bit types)
    // 2^61-1. `standard`: measured ~2 s.
    ValidationRegistry::register_validation("SubstrateDifferentialValidation",
                                            &SubstrateDifferentialValidation,
                                            ValidationTier::Standard);

    // Phase 1 (matrix-engine), items 7.0/7.5: rho_n was replaced by closed
    // forms for the structured generators plus an O(d_V^2) two-term column
    // recurrence. rho_n(A) is a unique mathematical object, so the contract is
    // BIT-IDENTITY with the pre-Phase-1 implementation, which is kept as
    // ComputeSymmetricPowerMatrixFM_reference and used as the oracle here.
    ValidationRegistry::register_validation("SymmetricPowerValidation",
                                            &SymmetricPowerValidation,
                                            ValidationTier::Standard);
    ValidationRegistry::register_validation("SymmetricPowerValidationSlow",
                                            &SymmetricPowerValidationSlow,
                                            ValidationTier::Slow);

    // Phase 2 (matrix-engine): coarse performance floors on the arithmetic
    // substrate (gemm 512, rref 1677x1419) against the backend named by
    // ENGINE_BACKEND (default legacy). The floors are deliberately loose --
    // they are set from the measured LEGACY baseline with >= 2x margin so the
    // test is a regression guard today, and Stage B raises them to the plan's
    // values (>= 4 Gmodmul/s, <= 3 s) when the engine backend becomes the
    // default. Measurements come from the same code as `--bench micro`.
    ValidationRegistry::register_validation("PerformanceSmokeValidation",
                                            &PerformanceSmokeValidation,
                                            ValidationTier::Standard);

    // Phase 0 (matrix-engine): golden-output regression harness. GoldenEmit
    // REGENERATES Golden/GoldenReference.inc (owner-gated re-baseline).
    //
    // TIER (integrator decision, 2026-09-05): GoldenRegressionValidation is
    // `standard` even though it measures 71 s at RelWithDebInfo and the
    // convention above puts anything over ~60 s in `slow`. Reasons: (i) it is
    // the convention this file already records for it; (ii) unlike the one
    // genuine `slow` entry (HeckeBenchmarkSmoke, a pure PERFORMANCE smoke) it
    // is a CORRECTNESS gate -- the bit-exactness contract of every phase of the
    // matrix-engine programme -- and a correctness gate that only runs under an
    // opt-in flag is a gate nobody runs; (iii) it keeps `--all` at ~2m30s,
    // still inside Phase 0's "`--all` exit 0 in <= 4 min" exit criterion.
    // Re-tier to `slow` if `--all` ever approaches that budget.
    ValidationRegistry::register_validation("GoldenRegressionValidation",
                                            &GoldenRegressionValidation,
                                            ValidationTier::Standard);  // measured 71.0 s
    // `fast`: under `--all` GoldenEmit hits its GOLDEN_EMIT safety interlock and
    // returns immediately without computing or writing anything; the real
    // (minutes-long) emission only happens under `GOLDEN_EMIT=1`.
    ValidationRegistry::register_validation("GoldenEmit", &GoldenEmit,
                                            ValidationTier::Fast);  // measured 0.000 s (interlocked)
    // hecke-extensions phase E0a: the "before"/"after" object dump that makes
    // E0b's re-baseline explainable object by object ([X] Sec. 1.6, steps 1-4).
    // `fast` for the same reason as GoldenEmit: under `--all` it hits its
    // GOLDEN_DUMP safety interlock and returns immediately, writing nothing;
    // the real (minutes-long) dump only happens under GOLDEN_DUMP=<dir>.
    ValidationRegistry::register_validation("GoldenDump", &GoldenDump,
                                            ValidationTier::Fast);  // interlocked

    // Phase 1 (matrix-engine), item 7.1: the multi-right-hand-side Theta_eq
    // solve is compared column by column against the per-vector
    // change_of_basis_fm it replaces (bit-identity), against the published
    // Theta_eq, and against the A-N3 residual certificate, on small/medium
    // points of the golden grid plus synthetic systems.
    ValidationRegistry::register_validation("ThetaEqSolveValidation",
                                            &ThetaEqSolveValidation,
                                            ValidationTier::Standard);  // measured 8.1 s

    // Phase 3 (matrix-engine), Stage B: HeckeOperatorOnK::compute() on the
    // block model -- engine arm vs the forced legacy arm entrywise (Theta_flat,
    // Theta_eq, A^{(q)}, certificate JSON), Theta_block(q) vs Theta_flat and vs
    // the tau-table pattern, apply_Theta / apply_route2 on full standard bases;
    // both element widths.
    ValidationRegistry::register_validation("HeckeThetaBlockValidation",
                                            OC_LEGACY_VALIDATION(HeckeThetaBlockValidation),
                                            ValidationTier::Standard);

    // Phase 3 (matrix-engine), Stage A: the STATIC consumer-migration gate.
    // Runs docs/matrix-engine/tools/raw_mod_calls.sh (and
    // check_matrix_includes.sh once the retirement work lands it) and fails on
    // any call site outside the tool's explicit allow-list. `fast`: it is two
    // greps over ~90 files, measured 0.05 s.
    ValidationRegistry::register_validation("ConsumerMigrationValidation",
                                            &ConsumerMigrationValidation,
                                            ValidationTier::Fast);  // measured 0.05 s
    // Phase 3 (matrix-engine): the `Matrix` retirement guard. Shells out to
    // docs/matrix-engine/tools/check_matrix_includes.sh, which owns the
    // allow-list; a new direct `#include "Matrix.h"` outside it fails here.
    ValidationRegistry::register_validation("MatrixRetirementValidation",
                                            &MatrixRetirementValidation,
                                            ValidationTier::Fast);  // measured 0.02 s

    // matrix-engine Phase 6 -- the campaign driver (`--campaign <spec-file>`).
    // CampaignSmokeValidation runs a 20-job sweep through the driver AND the
    // same 20 points as individual single-job campaigns, compares them hash for
    // hash, cross-checks every shared (point, object) key against the frozen
    // Golden/GoldenReference.inc, and pins memory-lean mode as value-identical
    // to the resident mode; `standard`. CampaignCertificateValidation is the
    // certify/resume contract and is cheap; `fast`.
    ValidationRegistry::register_validation("CampaignSmokeValidation",
                                            &CampaignSmokeValidation,
                                            ValidationTier::Standard);
    ValidationRegistry::register_validation("CampaignCertificateValidation",
                                            &CampaignCertificateValidation,
                                            ValidationTier::Fast);

    // Console phase C0 -- the language core of the CAS-style console
    // (docs/console/00-console-interface-plan.md v2 Sec. 10, Phase C0; work
    // package WP-5 of docs/implementation/hecke_extensions_slot1_plan.md).
    // Lexer / parser / evaluator / printer / registry / settings / handles over
    // ConsoleCore ONLY -- it constructs no matrix and touches no pipeline
    // arithmetic, so it can change no golden value.  It also carries the
    // slot-1 "no transcript freezing before E0b" guard and runs
    // docs/matrix-engine/tools/check_console_includes.sh over the checkout.
    // `fast`: measured 0.03 s.
    ValidationRegistry::register_validation("ConsoleC0Validation",
                                            &ConsoleC0Validation,
                                            ValidationTier::Fast);

    // Console phase C1a -- the bindings, the CLI branch, GoldenLookup and the
    // first pins (docs/console/00-console-interface-plan.md v3 Sec. 10, Phase
    // C1a; work package WP-7 of
    // docs/implementation/hecke_extensions_slot3_plan.md Sec. 3).
    //
    // ConsoleScriptValidation runs the smoke script set IN PROCESS on a fresh
    // Session.  Every pin in those scripts is read from GoldenReference.inc
    // through the ONE read-only lookup Golden/GoldenHarness.h now exposes
    // (GoldenLookup, inventory gap G13) and compared at the DEFAULT
    // mathematical configuration (fingerprint 0, plan rule R19) -- so the
    // console PROVES it reproduces the golden objects and freezes nothing of
    // its own.  It also carries the exit-code table 0/1/2/3/4/5 (5 = the typed
    // hecke::NotImplementedInThisBuild outcome, tested through a test-only verb
    // so the arm cannot depend on which mathematical settings this build
    // implements), the R3 state / resume / save_state verbs, the narrowed
    // "no decomposition transcript before C1b" guard, and a re-run of
    // check_console_includes.sh.  `fast`.
    ValidationRegistry::register_validation("ConsoleScriptValidation",
                                            &ConsoleScriptValidation,
                                            ValidationTier::Fast);

    // ConsoleCliPreservationValidation drives THIS binary through std::system:
    // the stdout and exit code of --help, --bench --help, --bench micro --help,
    // --campaign --help, --campaign <spec> --dry-run and one validation name
    // are byte-compared against files frozen BEFORE the console branch landed,
    // and --list is compared MODULO APPENDED NAMES (plan Sec. 11: freezing it
    // byte for byte would fail at the next registration by anyone).  Then the
    // console's own forms -- -c, --script with parameters, stdin, --console
    // --help -- and their exit codes.  `fast`.
    ValidationRegistry::register_validation("ConsoleCliPreservationValidation",
                                            &ConsoleCliPreservationValidation,
                                            ValidationTier::Fast);

    // Console phase C1b (hecke-extensions slot 4, WP-5) -- ConsoleScriptEmit,
    // which regenerates the `.expected` transcripts of Console/scripts/.
    // OWNER-GATED exactly like GoldenEmit: a no-op unless run as
    // `CONSOLE_EMIT_EXPECTED=1 ./cohomology_app ConsoleScriptEmit`, so `--all`
    // stays read-only and a re-baseline is always a deliberate act whose diff
    // has to be explained file by file. The FIRST emission was part of the C1b
    // package -- a new artefact, nothing re-baselined -- and every hash line in
    // those files is cross-checked by ConsoleScriptValidation against
    // `golden(...)` at fingerprint 0. `fast`.
    ValidationRegistry::register_validation("ConsoleScriptEmit", &ConsoleScriptEmit,
                                            ValidationTier::Fast);

    // hecke-extensions slot 1, WP-4a (phase E4a) -- the certified Gamma-level
    // boundary datum tau_bd of gl2_contract.tex chapter sec:boundarydatum
    // (Hecke/HeckeBoundaryDatum). GATING, over Z, on every table of the grid
    // (p, nu) in {(2,1),(3,1),(5,1),(7,1),(11,1),(2,2),(3,2)} x ell in
    // {2, 3, 5, 7, 691, 134217757} x n in {0, 2, 10} at uint32_t and at the
    // u64 primes 2^61-1, 2^32+15: obl:taubdry(i) (single component; the
    // component map is c -> g^{-1} c recomputed from g alone; degree 0 is one
    // simplex of coefficient 1), (ii) (T2bd) through an independent boundary
    // operator and an independent (T1bd) translation, (iii) the (T1bd)
    // bookkeeping across i(v3) = T i(v0) and on random words, (iv) every
    // coefficient in {+-1} with solve regime Z for EVERY ell; plus
    // prop:bdrycosets(1)-(3) (the B_N / Gamma_inf table, the nu = 1
    // representatives and identities, the bijection with the bulk cosets,
    // the two double cosets), the ell/n/width independence of the table, the
    // rejection of seven tampered chains, and the (B1)-(B3) boundary
    // subcomplex data of SL2ZTessellationTables. Prints the p = 2 table and a
    // `BOUNDARY_TAU tables=... checks=... failures=...` line. WP-4b extends
    // the boundary with Theta^bd (clauses (v)-(vii)) in its own validation.
    ValidationRegistry::register_validation("HeckeBoundaryTauValidation",
                                             &HeckeBoundaryTauValidation,
                                            ValidationTier::Standard);  // measured ~1 s

    // hecke-extensions phase E1s (slot 1, WP-3): the computed Sturm bound
    // B'(k, ell) of [E] def:bprime / cl:bprime -- the LEAST prime bound B such
    // that the F_ell-ALGEBRA generated by the T_q, q <= B prime, on the Miller
    // basis of M_k(F_ell) fills End's image, with the cap 12*B(k) and the
    // outcome recorded either way.  The unit is PIPELINE-FREE (no cohomology,
    // no coset table: pure modular-form arithmetic mod ell), so this is a
    // `fast` entry.  Arms: the q-expansions against EXACT-integer sigma_3 /
    // sigma_5 and the published tau(2..7), and against E_4^3 - E_6^2 = 1728 D;
    // dim M_k against the monomial count #{(a,b) : 4a + 6b = k}; the echelon
    // Miller basis and its stability in the precision; T_q on M_12(F_691) with
    // the congruent eigenvalues sigma_11(q) == tau(q) of [E] cl:ramanujan; the
    // twenty pinned bounds (B' = 2 on all ten Kummer cells of [E] Claim 1.8 and
    // at (12,691), (34,131), (162,691), (200,691); 5 at (24,5), (30,5), (32,5),
    // (36,5), (60,5); 7 at (28,7); 3 at (36,7)) with SturmBoundCertificate
    // verify() == 0 and two tampered variants failing; the PROPER subalgebra
    // <T_2, T_3> of dimension 2 < 3 at (24,5) and (28,7) -- the naive B(k)
    // sweep's counterexample; and the R3 resumable-state contract of
    // [X] Sec. 1.3.1 (instance 1, of which this is the first implementation):
    // `none_at_cap` under a forced cap = 3 RECORDED not thrown,
    // resume(cap = 12*B(k)) reaching B' = 5 with a state and certificate
    // entrywise equal to a fresh run's, resume(cap = 2) refused with the named
    // error, a resume on a complete state a strict no-op, and the
    // budget_history [3, 36].  Plus both `sturm.unavailable` modes as the pure
    // (outcome, mode) -> (final, reason) map -- E1b owns the heuristic sweep.
    ValidationRegistry::register_validation("HeckeSturmBoundValidation",
                                            &HeckeSturmBoundValidation,
                                            ValidationTier::Fast);  // measured ~0.1 s
    // hecke-extensions phase E1s (slot 1, WP-3): the HeckeExt/ include guard.
    // HeckeExt/ is compiled into cohomology_app rather than into a static
    // library ([X] Sec. 1.1), so the layer's ONE-WAY dependency is a property of
    // the source tree; it is checked by running
    // docs/matrix-engine/tools/check_heckeext_includes.sh over the checkout via
    // std::system, the MatrixRetirementValidation pattern.  Four rules: no
    // include of HeckeExt/ outside HeckeExt/ / Campaign/ / Golden/ / Bench/ /
    // kernel.cu / *Validation.* / ValidationRegistry.cpp; no Console/, Cuda/,
    // Engine/ or BlockFlatMatrix.h inside HeckeExt/; HeckeSturmBound.* limited
    // to SubspaceOps / FpPoly / Matrix / ModArithmetic / EngineCheck; and no
    // `unordered_` anywhere in the layer.
    ValidationRegistry::register_validation("HeckeExtArchitectureValidation",
                                            &HeckeExtArchitectureValidation,
                                            ValidationTier::Fast);  // measured ~0.02 s
    // hecke-extensions slot 2, WP-4 (phase E1a) -- the LOCAL ALGEBRA A_S of a
    // certified-local block and generation certificate (a)
    // (HeckeExt/HeckeLocalAlgebra; [E] sec:localalg, alg:stage3, cl:resfield,
    // cert:cyclic; [X] Sec. 2.2 Stage 3 and Stage 4(a); [C] obl:algebra /
    // thm:relations is where the commutativity the closure rests on is stated,
    // and [C] delegates the local pieces T_m to [E]).  Constructed truths only:
    // monomial algebras F_{ell^f}[x_1..x_v]/(monomial ideal) whose dim A_S,
    // ideal filtration, nu, Hilbert function, socle, Gorenstein verdict and
    // embedding dimension are read off the monomial set -- F[eps]/eps^2,
    // F[x,y]/(x,y)^2, F[x]/(x^3), F_{ell^2}[eps]/eps^2 (embdim 1 BY THE K-SPAN
    // TEST, with the witness's own direction kept for t = alpha + eps),
    // F_{ell^2}[x,y]/(x,y)^2, a non-cyclic A + A and the degenerate field
    // (m_S = 0, nu = 1) -- each run plain and conjugated by a seeded
    // invertible U, over ell in {5, 13, 691, 134217757}.  Plus: certificate
    // (a) re-derived by verify() on every case; cyclicity against an EXPLICIT
    // A_S v closure computed in the validation; the [XA B2] negative control
    // (P(w) in place of P(w_hat) gives n = 0, the lost direction); the [XA M5]
    // cross-block evaluation test (a Frobenius-conjugate character matches, a
    // same-ideal pair of DIFFERENT dimensions matches with N' taken for the
    // tested block, a distinct character does not); and a ten-way tamper arm in
    // which each corruption fails exactly its own clause.
    //
    // slot 2, WP-5 (phase E1a-2) added the GENERATION CERTIFICATE (b)
    // (HeckeExt/HeckeGenerationCertificates; [E] cert:comm; [X] Sec. 2.2
    // Stage 4(b), Sec. 1.3 CommutantCertificate, Sec. 1.3.1 R3.3 item 3):
    // the socle bound dim C >= dim A_S + f*t*s - dim_F soc A_S checked against
    // an INDEPENDENT naive dim C and a closed form on every case (all D <= 24);
    // `excluded_by_socle_bound` on both Gorenstein non-cyclic cases, including
    // the (10, 691) shape A + A/m where the bound 5 is EXACT ([XA B1]); the
    // positive case V = A^dual for A = F[x,y]/(x,y)^2, where (b) PASSES and (a)
    // fails; structured == naive == closed form wherever all three run, with
    // the invariant factors re-derived from the nullity sequence of pi_a(a)^j;
    // the R3 arms -- the budget interrupted at EVERY reachable row count and
    // resumed to a state and certificate ENTRYWISE equal to a fresh run's, the
    // budget histories differing, a lower budget refused with the named error,
    // a resume on a complete state a strict no-op; the OQ8 `final = false`
    // emission (reason `closure_unreachable` / `budget_exhausted:commutant`,
    // with the resumable object and `complete = false`); and a sixteen-way
    // tamper arm with a positive control (a rescaled cyclic block is another
    // valid RCF basis and must still verify -- the certificate re-derives
    // dim C, which is basis-independent).  The REAL-BLOCK arms of [X] Sec. 5's
    // row (Sturm membership at (10, 691), the joint X_1 + X_4 decomposition)
    // are E1b's, not here.
    ValidationRegistry::register_validation("HeckeExtAlgebraValidation",
                                            &HeckeExtAlgebraValidation,
                                            ValidationTier::Standard);  // measured ~0.1 s
    // hecke-extensions slot 1, WP-4b (phase E4b) -- the boundary Hecke
    // operator Theta^bd_g / T^bd_p on the circle model
    // (Hecke/HeckeBoundaryOperator), gl2_contract.tex sec:boundarydatum:
    // obl:taubdry(v) (route A, the Shapiro transcription on the infinity-line
    // with the canonical left cosets, and route B, the closed forms
    // sum_delta rho_n(delta) / eq:bdrygroupform, both == eq:bdrychain on
    // cohomology; route B re-derived here), (vi) the chi_b(p) = p^b +
    // p^{n+1-b} eigenvalues on x^{n-b ell} Nm^b (own polynomial arithmetic)
    // and 1 + p^{n+1} on [y^n] for n < ell, ASSERTED; the general-n H^1
    // multiset of cl:chi(2) RECORDED; (vii) r^q A^{(q)} = A_bd^{(q)} r^q
    // against HeckeOperatorOnK through EquivariantCochainComplexMorphismOnK,
    // ASSERTED at ell >= 5 and RECORDED at ell in {2, 3}; the beta_j
    // right-coset NEGATIVE control (prop:bdrycosets(4), prop:bdrychar(2)):
    // chi_{b+1} in place of chi_b, and it BREAKS (vii); the model recording
    // of owner confirmation 11 (boundary_model = circle,
    // boundary_is_group_cohomology = (ell != 2), q_max derived, never the
    // lem:ell2unbounded dimensions). Grid: (p, nu) in {(2,1),(3,1),(5,1),
    // (7,1),(2,2),(3,2)} x ell in {2,3,5,7,11,13,101,691,134217757,2^31-1}
    // x n in {0,2,...,16}, plus a u64 arm at 2^61-1. Prints one
    // `BOUNDARY_OP cells=... checks=... failures=...` line.
    ValidationRegistry::register_validation("HeckeBoundaryOperatorValidation",
                                             &HeckeBoundaryOperatorValidation,
                                            ValidationTier::Standard);  // measured ~5 s

    // hecke-extensions slot 2, WP-7 (phase E4b) -- HeckeExt/HeckeBoundary, the
    // ext-layer consumer of the boundary datum and operator: H^1_! = ker r^1
    // ([C] cl:les(2)) as a T_p-stable submodule with its action A_! (exact
    // solve), the Eisenstein line by HeckeIotaSignValidation's route (a) rule
    // (the joint 1 + p^{n+1} eigenline over p in {2,3,5} \ {ell}), HARDER's
    // criterion ([E] cl:harder: r^1 = 0 on the Eisenstein class iff
    // ell | num(B_k/k), n < ell - 1) ASSERTED on the ten Kummer cells of [E]
    // Claim 1.8 and on every non-Kummer cell of the grid, RECORDED at (10, 5)
    // and (10, 7) ([XA m5]); the Harder split (charpoly(A) = charpoly(A_!) *
    // charpoly(A_bd | im r^1), chi_0 on the quotient at n < ell, H^1 = H^1_!
    // (+) Eis where the line restricts non-trivially); the classical
    // dim H^1_! = 2 dim S_{n+2} at ell > n + 1 non-Kummer, RECORDED at the
    // Dickson loci (20, 5), (24, 5) and at ell <= n + 1; the certificate's
    // verify() with a seven-way tamper arm; the golden collectors (14 records
    // per (p, nu), never emitted) digest-stable in-process and across child
    // processes at ENGINE_THREADS = 1 / 16; and the two diagnostics of
    // Hecke/HeckeBoundaryDiagnostics -- cochain_commutes (r Theta_g =
    // Theta_g^bd r on C^q_eq(K), p in {2, 3, 5, 7}) and eta_zero (the
    // cusp-support walk of the bulk tau against tau_bd) -- RECORDED, never
    // branched on ([C] rmk:conestrict), with the one theorem eta_zero =>
    // cochain_commutes asserted and the p = 2 walk pinned to
    // tau_table_p2.txt's recorded shape. Grid: ell in {2, 3, 5, 7, 691,
    // 134217757, 2^31-1} x even n <= 16 x (p, nu) in {(2,1),(3,1),(5,1),(7,1),
    // (2,2)}, plus (20,5), (24,5), (10,13) and the nine off-grid Kummer cells.
    // Prints `BOUNDARY_DIAG ...` and `BOUNDARY_PARABOLIC ...`.
    ValidationRegistry::register_validation("HeckeBoundaryParabolicValidation",
                                             &HeckeBoundaryParabolicValidation,
                                            ValidationTier::Standard);  // measured 4.7 s (75 cells, 339 pipelines)

    // hecke-extensions slot 2, WP-8 (phase E5-1) -- the certified pair
    // homotopy eta of gl2_contract.tex sec:cone (Hecke/HeckePairHomotopy):
    // boundary eta + eta boundary = j tau_bd - tau j over Z, built by
    // lem:etaexists against the shipped tau and WP-4a's tau_bd on ONE coset
    // table, certified in the constructor. GATING on every table of the grid
    // (p, nu) in {(2,1),(3,1),(5,1),(7,1),(11,1),(2,2),(3,2)} x ell in
    // {2, 3, 5, 691, 134217757} x n in {0, 2} at uint32_t and at 2^61-1 /
    // 2^32+15 at u64, both vertex-chain policies: obl:eta(i) (det 1, the
    // right dimension, canonical form, the same coset datum and model, tau
    // integral), (ii) the homotopy identity on every stored slot AND on
    // random Gamma-translates through an OWN (T1) translation and an OWN
    // boundary operator, (iii) the (T1) bookkeeping across every stored
    // transporter and the entry-bound recount, (iv) solve regime Z for EVERY
    // ell with no mod-m path; the ell / n / width independence of the table;
    // eight tampered chains rejected by certify_with_override; eta_zero
    // agreeing slot by slot with WP-7's cusp-support walk (both false: the
    // eta-table is genuinely needed); lem:etaunique -- the two policies'
    // tables differ by boundary L - L boundary for the UNIQUE degree-2 L
    // (computed by unique_fill, verified by an own boundary); and the
    // degrees built per open:ell2 rider (i) recorded ([0, 1], q_max = 1
    // derived, truncated = false). Clause (v) and the cone are slot 3's.
    // Prints the p = 2 table and one `PAIR_HOMOTOPY tables=... checks=...
    // failures=...` line.
    ValidationRegistry::register_validation("HeckePairHomotopyValidation",
                                             &HeckePairHomotopyValidation,
                                            ValidationTier::Standard);  // measured: see docs/implementation/slot2/WP-8_report.md

    // hecke-extensions slot 3, WP-5 (phase E5-2) -- the equivariant mapping
    // cone Cone(r) of gl2_contract.tex sec:cone (EquivariantMappingCone,
    // Hecke/SL2ZBoundaryPair): the pair (7-vertex model, circle model) with
    // the pipeline-side restriction r = j^* (def:pair), the cone
    // eq:conediff in H_c-indexing on the flat layout of alg:assembly, Z / B /
    // H / P through engine::blk / engine::elim (D5b), the three map families
    // of eq:coneles in stored bases, the per-complex degree fields of
    // open:ell2 (derived; a set q_max on the finite model rejected) and
    // ConeCertificate (obl:coneexact(iv)-(v) + Dcone^2 = 0) with a
    // re-deriving verify().  The validation re-derives everything
    // independently: an own dense eq:conediff assembly and own D^2 = 0, an
    // own zig-zag connecting map == the stored r^q == the morphism's induced
    // map == HeckeExt/HeckeBoundary::r(q) entrywise, exactness at every
    // asserted spot (every spot on the model complexes) by own ranks with
    // im(pi^1) = ker r^1 explicitly, H^0_c = 0 at EVERY ell and n
    // (lem:conerelative(4)), the first eq:conedims identity at every ell and
    // the second where H^2_eq = 0 (asserted at ell >= 5, recorded at ell in
    // {2, 3}), dim H^1_c - dim coker r^0 = dim H^1_! against HeckeBoundary,
    // the measured dim H^1_c = 3 at (10, 691) and 1 + 2 dim S_{n+2} with
    // H^2_c = 0 at all ten Kummer cells of [E] Claim 1.8, the record fields,
    // the truncated branch of (iv) on a declared truncation of the cylinder
    // rel. both boundary circles (E9's shape as a test double), a tamper arm
    // (a corrupted cone basis entry fails the basis items only; a corrupted
    // r^1 entry fails the map item), and the golden collectors' digest
    // (defined, never emitted) equal across two in-process builds and two
    // ENGINE_THREADS = 1 / 16 child processes.  Grid: ell in {2, 3, 5, 7,
    // 691, 134217757, 2^31-1} x even n <= 16, plus (20, 5), (10, 13) and the
    // nine off-grid Kummer cells.  Prints one `CONE cells=... checks=...
    // failures=...` line.  WP-6 (the cone action) extends this TU.
    ValidationRegistry::register_validation("HeckeConeValidation",
                                             &HeckeConeValidation,
                                            ValidationTier::Standard);  // measured 1.9 s (74 cells, 5350 checks); docs/implementation/slot3/WP-5_report.md

    // hecke-extensions slot 3, WP-10 -- the fifth grep-level migration guard
    // ([C] sec:machinery item 2, rule L1), next to ConsumerMigrationValidation /
    // MatrixRetirementValidation / ConsoleC0Validation / HeckeExtArchitectureValidation.
    // Every `[C, <label>]` citation in the tree (docs, C++ comments,
    // CMakeLists.txt, READMEs, plans) and every `\HA{<label>}` cross-reference
    // in docs/latex/hecke_extensions.tex must name a label docs/latex/gl2_contract.tex
    // actually `\label{}`s. Unlike the other four guards a citation payload is
    // comma-separated, sometimes clause-suffixed, sometimes multi-label, so the
    // parser lives in docs/matrix-engine/tools/label_census.py; this validation
    // runs the bash entry point docs/matrix-engine/tools/label_census.sh via
    // std::system, the MatrixRetirementValidation pattern, and regenerates
    // docs/latex/label_census.tsv (one row per citing file, plus TOTAL)
    // unconditionally. SITE COUNT IS NOT GATED (it grows every slot); only the
    // UNRESOLVED count, after label_census.allow, must be 0.
    ValidationRegistry::register_validation("ContractCensusValidation",
                                            &ContractCensusValidation,
                                            ValidationTier::Fast);  // measured ~0.2 s (python3 census over the checkout)
    // (hecke-extensions slot 3, WP-6 / phase E5-3) The Hecke ACTION on the
    // mapping cone: the derived cochain homotopy h = eta~ of [C, eq:derivedh]
    // (the coset transport of slot-2 WP-8's eta-table, never chosen), the cone
    // cochain map Phi_g = (Theta_g, Theta^bd_g + h) of [C, eq:conemap], the
    // residual bracket of [C, lem:conesign] computed and asserted identically
    // zero, A_c^{(q)} on H^q_c by [C, alg:assembly] on Cone(r), and the two
    // certificates: ConeActionCertificate ([C, obl:eta](v),(vi) and
    // [C, obl:coneexact](i)-(iii)) and ConeRelationsCertificate
    // ([C, obl:coneexact](vi): commutativity and the nu = 2 recursion with the
    // FULL T_{p^2} = T'_{p^2} + p^n I).  Arms: the own coset transport, the own
    // dense residual, the matrix-free second route on every standard basis
    // vector, the LES intertwinings against the pipeline's own A^{(q)} /
    // A_bd^{(q)} and against HeckeOperatorFamily's independent double-coset
    // route, [C, obl:coneexact](viii)(a) on the four policy combinations
    // (EtaVertexPolicy x TauBoundaryVertexPolicy) with the NON-VACUITY
    // assertion that both table hashes move, (viii)(b)'s negative control
    // h' = h + xi failing EXACTLY obl:eta(v) and changing A_c^{(1)} at
    // (10, 691), and a two-way tamper arm.  Grid: ell in {2, 3, 5, 7, 691,
    // 134217757, 2^31-1} x even n <= 16 with (p, nu) in {(2,1),(3,1)}, a wider
    // (5,1) / (2,2) sub-grid and U_ell at ell <= 7.  Prints one
    // `CONE_ACTION cells=... checks=... failures=...` line.
    ValidationRegistry::register_validation("HeckeConeActionValidation",
                                             &HeckeConeActionValidation,
                                            ValidationTier::Standard);  // docs/implementation/slot3/WP-6_report.md

    // ======================= hecke-extensions slot 3 =========================

    // slot 3, WP-1 (phase E1b-0) -- THE U_ell EXACT-INTEGER REACH
    // (docs/KNOWN_ISSUES.md Sec. 3, closed by this package; [C] thm:backend's
    // escalation ladder, lem:bounds(3), obl:tau / obl:backend).
    //
    // DIAGNOSIS.  [C, alg:tau] step 1 forms z = (g^{-1} s_i) . pos(v) through
    // the projective adj(g) = diag(N, 1) trick and reduces it to the
    // fundamental domain.  The POINT's cleared coordinates reach
    // ||z|| ~ ||pos||^2 N^2 (measured 144.8 N^2 on the shipped 7-vertex model)
    // and QuadPoint::moebius squares them, so the int64 tier dies at
    // N >= ~4580 -- exactly the measured bracket 3617 < N < 5003 of [KI] Sec. 3.
    // The site is QuadPoint::moebius called from
    // HeckeTauTableT::reduce_to_fundamental_domain, NOT the coset construction
    // ([KI]'s original attribution; the coset tables build and self-check at
    // ell = 43867), and NOT a failure of [C, lem:bounds]: the certified bound
    // B = max(B_tab, 4 N^2) is 1.0e8 at N = 5003 with 2 B^2 = 2.0e16 << 2^63,
    // and certify() clause (e) passes.  The overflowing quantity is a step-1
    // INTERMEDIATE that no clause of lem:bounds covers.
    //
    // REPAIR: QuadPointT<__int128> (QuadPoint.h), thm:backend's own third
    // tier, reached by hecke::escalate_on_overflow -- int64 first, widened
    // ONLY after a HeckeOverflow, so wherever int64 completes nothing changes
    // and the golden reference cannot move.  Nothing STORED widens: the
    // reduction's outputs (gamma_red, a vertex index) stay int64-sized.
    //
    // STANDARD arm: QuadPointW vs QuadPoint on 4000 seeded transforms of the
    // shipped vertex positions (identical wherever int64 completes, correct
    // past it); coset tables self-checked at ell = 3617 and (p, r) = (71, 2)
    // with N = 5041 past the old limit -- the audit's F13 vacuity check, i.e.
    // context, not the reach proof; the FORCED-WIDE arm at ell in {5, 691} and
    // (71, 2), where tau's chains, its certified bound B and A^{(q)}(U_ell) are
    // compared ENTRYWISE between ExactTier::ForceWide and the default;
    // escalation accounting (Auto never widens below the threshold, so every
    // golden cell runs the pre-WP-1 path); and the NEGATIVE CONTROL -- under
    // ExactTier::ForceNarrow ell = 5003 still throws the named HeckeOverflow.
    ValidationRegistry::register_validation("HeckeUEllReachValidation",
                                            &HeckeUEllReachValidation,
                                            ValidationTier::Standard);  // measured ~11 s (6160 checks)
    // The same package's arms past ~10 s (slot-3 plan WP-1 scope 3: "if an arm
    // exceeds ~10 s it goes into a slow name"): the forced-wide arm at
    // ell = 3617; the reach proof itself -- the certified tau table at
    // ell = 5003 and 10007 -- U_5003 = T_p(5003) at n = 2 END TO END with
    // HeckeBadPrimeValidation's oracle rows (well-typedness, rank(U^r)
    // non-increasing and stationary by r = d_q; U_{ell^2} is not attempted,
    // N = ell^2 would be 25 M cosets); and the ell = 43867 coset tables.
    // U_43867 END TO END is deliberately NOT asserted: u_ell = auto refuses it
    // and HeckeUEllBudgetValidation pins that refusal.
    ValidationRegistry::register_validation("HeckeUEllReachSlow",
                                            &HeckeUEllReachSlow,
                                            ValidationTier::Slow);  // measured ~10 min

    // slot 3, WP-1 -- the FROZEN U_ell RSS model and the `u_ell` mathematical
    // setting ([X] Sec. 1.5 table, Sec. 2.2 Stage 1, [XA B3]; [OD] slot-1
    // confirmation 4: u_ell = auto, budget_mb = 2048).  HeckeExt/
    // HeckeUEllBudget.h is a PURE header -- PredictUEllRssMb(n, ell) and
    // UEllDecision(n, ell, budget_mb) with a model-version string, refitted by
    // this package on a 17-point grid.  Arms: the fit's own residuals on the
    // measured grid; the admission table at M = 2048 re-derived against
    // [X] Sec. 1.5's prose, with every change reported; the pinned refusal
    // UEllDecision(16, 43867, 2048) = {false, u_ell_over_budget, ...}; the
    // `u_ell` key's parsing, round-trip, canonical text, fingerprint (0 at the
    // defaults) and thread-local propagation onto run_concurrently() tasks.
    ValidationRegistry::register_validation("HeckeUEllBudgetValidation",
                                            &HeckeUEllBudgetValidation,
                                            ValidationTier::Fast);  // measured ~0.01 s

    // slot 3, WP-2 (E1b) -- GENERATION CERTIFICATE (c), the STURM SWEEP on the
    // pipeline's own blocks, plus Stage 1's `u_ell` budget check
    // ([E] cert:sturm on the range of cl:sturmmodl, [X] Sec. 2.2 Stage 1 and
    // Stage 4(c), Sec. 1.3.1 R3).  The real-block arms cannot live in
    // HeckeSturmBoundValidation, which consumes no pipeline by design.
    //
    // Arms: Stage 1 (S_0 = {2,3,5,7} + ell iff the frozen RSS model admits it;
    // the pinned refusal at (16, 43867); `off` removing ell even at ell <= 7;
    // the two new mathematical keys `sturm` / `sturm_unavailable` with their
    // parsing, canonical text and fingerprint); (10, 691) -- A_S =
    // F_691[eps]/eps^2, (a) fails, (b) `excluded_by_socle_bound` with the
    // INDEPENDENT naive dim C = 5, (c) sturm(B' = 2) with
    // generators_certified = [2] and final = true, plus the two real-block
    // arms deferred from slot-2 WP-5; the B(k) counterexamples (22, 5) and
    // (26, 7) -- the naive {2, 3} sweep REJECTED and the B' sweep accepted;
    // both `sturm.unavailable` modes on the forced-cap (24, 5) case with the
    // real A^{(1)}(T_q), with the R3 resume-invariance arms; a tampered
    // certificate failing exactly its own clause; and the recorded ext-layer
    // wall time at n = 256 against `hecke 256` (acceptance (c), never
    // asserted).
    ValidationRegistry::register_validation("HeckeExtSturmSweepValidation",
                                            &HeckeExtSturmSweepValidation,
                                            ValidationTier::Standard);  // measured ~2.4 s (274 checks)

    // slot 4, WP-1 (E2a) -- THE EXTENSION RECORD `hecke_ext/v1`, the one
    // tri-state `Final`, the mathematical-settings fingerprint in the record
    // and in the campaign job id, the `commutant_budget` key, and the R3.1
    // item-6 sidecar ([E] `sec:schema`, [X] Sec. 1.3 / Sec. 1.3.1 R3.1-R3.2 /
    // Sec. 1.4 / Sec. 1.5, [C] `sec:recordschema`).
    //
    // Arms: the `Final` migration (`SturmFinal` is an ALIAS of it now) and the
    // fixed reason list of R3.2, accepted and rejected; the two new
    // mathematical keys `commutant_budget` / `final_mode` (parse, canonical
    // text, round trip, registry ORDER, fingerprint 0 at the defaults, the
    // scoped override); `is_final`'s truth table over
    // final_mode x sturm_unavailable x {proof range, heuristic range, none}
    // with EVERY reason of the fixed list, plus a 9216-point grid on which the
    // DEFAULT configuration never produces "heuristic"; the per-complex degree
    // fields of [C] `sec:recordschema` item 5 (bulk 2 / false / derived /
    // [0,1,2]; a `set` q_max on an untruncated complex and a complex with no
    // degree count both REFUSED); the [E] Sec. 7 key set and `not_applicable`
    // at n = 0; the sidecar round trip on a forced-oversize state (relative
    // path, FNV-1a-64, reload byte for byte, the no-stem `state_oversize`
    // rule, the same-kind disambiguator); `commutant_budget` driving
    // CommutantRoute with the R3 resume (a lower budget refused, a raise equal
    // to a fresh run modulo budget_history); the real (10, 691) bulk record
    // built from the shipped pieces under `u_ell = off` (dim H^1 = 3, one
    // local block, A_S = F_691[eps]/eps^2, Stage 5's loewy/socle = [2,1] and
    // the [0,1] structure map of [E] Sec. 7's worked example, (b) excluded by
    // the socle bound, (c) sturm(B' = 2), final = true); and the campaign
    // job-id arm (`n16_ell5_p2_r1` byte-unchanged, `_s<16 hex>` on a
    // non-default setting, distinct ids / shape signatures / resume paths).
    //
    // Its three golden collectors (`ext_ideals^q`, `ext_certify^q`,
    // `ext_json_canonical^q`) are file-local to its TU and are called from
    // NOWHERE else: slot 4 emits no golden entry.
    ValidationRegistry::register_validation("HeckeExtRecordValidation",
                                            &HeckeExtRecordValidation,
                                            ValidationTier::Fast);  // measured ~0.05 s (381 checks)

    // (hecke-extensions slot 4, WP-3 / phase E3) The PRODUCTION involution
    // iota^* of [C, sec:involution] (Hecke/HeckeInvolution) and the sign split
    // (HeckeExt/HeckeSignSplit).  Until slot 4 iota^* existed only inside two
    // VALIDATIONS ([C, sec:codemap]: "no production iota^* exists"); this is
    // the object [C] specified for it, and its acceptance is an ENTRYWISE
    // comparison against the involution HeckeIotaSignValidation builds from
    // [C, def:iota] alone -- exported by this package as hecke::iota_oracle,
    // value-preservingly (that validation's printed lines are byte-identical).
    //
    // It asserts: production iota_C^{(q)} == oracle in every degree on the
    // 77-cell grid, iota^* == -iota^*_geo ([C, def:iota](iv), owner decision
    // OQ3), iota^2 = I, iota d = d iota, [iota, A^{(1)}(T_p)] = 0 for
    // p in {2,3,5}, the split dim H^1_+- = (d_{n+2}, d_{n+2}-1) off the
    // Dickson locus, the GATING Eisenstein sign by both routes reproducing
    // "eisenstein=plus (geometric=minus)" FROM THE PRODUCTION OBJECT, the
    // degree-0 anchor iota^* = -1 on H^0 at n = 0, (10, 691)'s
    // dim H^1_+- = (2, 1) with cyclic on both signs and the Eisenstein line
    // soc(X^+_{4,m}) = m_S X^+_{4,m} DEFINED (undefined on the unsplit H^1,
    // whose socle is 2-dimensional), the boundary naturality
    // r iota^* = iota^*_bdry r and the cone identity D_c iota_c = iota_c D_c,
    // a tamper arm, and the ell = 3 (recorded) / ell = 2 (not_applicable)
    // cells the 77-cell grid does not have.
    ValidationRegistry::register_validation("HeckeInvolutionValidation",
                                            &HeckeInvolutionValidation,
                                            ValidationTier::Standard);  // measured ~2.1 s (2818 checks, 77 + 11 cells)

    // slot 4, WP-7 (E6) -- the two E6 PRE-RECORD units: Bernoulli numbers
    // mod ell with the Kummer / Harder divisibility predicate
    // (HeckeExt/HeckePredictions, [X] Sec. 2.1 and Sec. 2.2 Stage 6) and the
    // Glover / Ash-Stevens composition factors of Sym^n(F_ell^2)
    // (HeckeExt/GloverFactors, [E] Sec. identities item 6).  `fast`: neither
    // unit builds a cohomology or a coset table, and the only pipeline object
    // the validation touches is ComputeSymmetricPowerMatrixFM on 2 x 2 inputs.
    //
    // WHAT IT RETIRES.  Slot-2 WP-7 shipped Harder's criterion against a PINNED
    // num(B_k/k) table limited to k <= 26.  Arm A reconstructs that table AS AN
    // INTEGER out of bernoulli_mod alone (CRT over four ~2^30 primes, with von
    // Staudt-Clausen supplying den(B_k)) and compares it entrywise -- 13 of 13
    // agree -- and arm B carries the same reconstruction to k = 42.  The
    // REWIRING of HeckeExt/HeckeBoundary is deliberately NOT done here (it would
    // move that validation's recorded `harder_applicable` for k > 26); it is
    // slot 5's, with the record integration.
    //
    // Arms: (A) the pin, (B) beyond it, (C) the Worpitzky/Stirling closed form
    // as an independent second formula for every w < ell - 1 at every prime
    // ell <= 131 and sampled to w = 3615, (D) the Kummer congruence that
    // warrants kummer_verdict's third branch, (E) the predicate against integer
    // divisibility on 21 x 35 cells plus the four named Kummer divisors, the
    // regularity of every prime <= 31 and the irregular pair (37, 32),
    // (F) the named refusal at w >= ell - 1, (G) Glover's structural invariants,
    // (H) Glover against the BRAUER CHARACTER of the real rho_n on every
    // ell-regular class of GL_2(F_ell) -- eigenvalues read off the
    // characteristic polynomial of the matrix ComputeSymmetricPowerMatrixFM
    // returns, the Hessenberg routine itself cross-checked against
    // det(xI - rho_n(g)) -- for every n <= 3 ell at ell in {5, 7} and beyond at
    // ell in {2, 3, 11}.
    ValidationRegistry::register_validation("HeckePredictionsValidation",
                                            &HeckePredictionsValidation,
                                            ValidationTier::Fast);  // measured ~0.7 s (113 664 checks)

    // slot 5, WP-7 (E6) -- the PROJECT-CANONICAL SPLIT FIELD of [X] Sec. 2.2
    // Stage 6 and [E] def:canon / cl:canon / alg:split (planned as slot-4
    // WP-10, which did not run).  HeckeExt/HeckeCanonicalField: C_{ell,r} =
    // the CanonicalLess-least monic irreducible of degree r, by enumeration in
    // that order + hecke::CertifyIrreducible; the arithmetic of
    // F = F_ell[Y]/(C) and of univariate polynomials over it (the only new
    // arithmetic Stage 6 needs); and the root finding of alg:split step 1 with
    // its FIXED candidate sequence -- the elements of F in CanonicalLess order
    // -- so that a block's eigenvalues become comparable across blocks and
    // across runs.  It also registers the two settings keys `field_polynomial`
    // (`least` | `conway`, the latter parsed and raising
    // hecke::NotImplementedInThisBuild) and `split_field_report`
    // (`canonical` | `own_field`), both at their defaults, so the all-defaults
    // fingerprint is still 0.
    //
    // THERE IS NO REPORT CONSUMER: nothing on any pipeline path calls the unit,
    // and no record gains a key -- the integration is E6's, in slot 6.  The
    // package measures E6's performance gate (`--bench ext`'s
    // canonical_field_s= / own_field_s= columns) and deliberately does not flip
    // the default on it.
    //
    // `fast`: no cohomology, no coset table, no matrix -- the whole validation
    // is FpPoly arithmetic.  Its oracles are independent of the unit: an
    // exhaustive trial-division irreducibility test (no gcd, no Frobenius), the
    // Moebius count of monic irreducibles, the FpPoly-mod-C model of F, and an
    // EXHAUSTIVE root search over F wherever |F| is small enough.  Seven arms:
    // (A) minimality of C_{ell,r}, (B) the field arithmetic, (C) polynomials
    // over F, (D) root finding against exhaustion, (E) the split with a
    // nine-way tamper on its certificate, (F) the (2, 30) stress case -- the
    // char-2 trace-map branch at the largest residue degree the repository
    // produces -- and (G) the two settings keys.
    ValidationRegistry::register_validation("HeckeCanonicalFieldValidation",
                                            &HeckeCanonicalFieldValidation,
                                            ValidationTier::Fast);

    // (hecke-extensions slot 4, WP-2 / phase E2b) [E] `cl:ramanujan` (Claim
    // 8.1) at (10, 691) THROUGH HeckeExt/HeckeExtDriver -- the Stage 1-5
    // orchestration and the first acceptance test of E2.  Six arms: the
    // primary, basis-INDEPENDENT checks (the four matrix identities in
    // M_3(F_691) including 3(U_691 - 1) = -542(T_2 + 24), the ratios
    // c'_q/c'_2 == c_q/c_2, the eigenvalues, the Jordan types (2,1), the
    // algebra invariants, Loewy/socle (2,1), (a) fails / (b)
    // excluded_by_socle_bound with the naive dim C = 5 / (c) sturm(B' = 2),
    // final = true) with U_691 INSIDE the frozen u_ell budget; then the
    // basis-DEPENDENT pins T_q = a_q I + c'_q E_31, labelled as such; the
    // STREAMED-operator route at (22, 5) and (26, 7), where the sweep needs a
    // prime outside S_0 and the streamed restriction must equal the
    // family-built one entrywise with a byte-identical record; the `u_ell`
    // RESUME of [X] Sec. 1.3.1 R3.3 item 4 (budget_mb=64 refuses U_691,
    // resume(2048) reproduces the fresh `auto` record byte for byte,
    // resume(32) is refused, a resume on an included verdict is a no-op); the
    // Stage-3 fast path (consuming the block's certified locality data) equal
    // ENTRYWISE to re-deriving it; and the n = 0 / odd-n edges.
    ValidationRegistry::register_validation("HeckeExtRamanujanValidation",
                                            &HeckeExtRamanujanValidation,
                                            ValidationTier::Standard);

    // (hecke-extensions slot 5, WP-2 / phase E2 cont.) THE CAMPAIGN SEAM.
    // Six arms: the per-cell grouping of [X] Sec. 1.5 / [XA m14] as a property
    // of plan_campaign() alone; acceptance (a), `ext = false` byte-identity
    // (the 21-column summary.tsv header, the id `n16_ell5_p2_r1` and the
    // default shape signature as literals, and no `extensions` key anywhere);
    // the attachment (one record per (n_sym, ell) cell, on the cell's FIRST
    // job, in the standalone `ext/` file AND verbatim inside that job's JSON,
    // with the eight ext columns APPENDED to summary.tsv); DETERMINISM across
    // four `jobs x threads` splits, ext record files byte-identical and
    // summary.tsv byte-identical outside its two timing columns; the
    // record-level `u_ell` RESUME of [X] Sec. 1.3.1 R3.3 item 4 at (10, 691)
    // -- refused at budget_mb=64, a lower budget refused BY NAME, the resume
    // to the shipped default byte-identical to a fresh `u_ell = auto`
    // campaign's record, a resume over an included verdict a strict no-op --
    // and the R3.1 item-6 state sidecar, on a record forced oversize in a
    // scratch directory (found, fnv-verified, and refused by name when
    // deleted or altered).
    ValidationRegistry::register_validation("HeckeExtCampaignValidation",
                                            &HeckeExtCampaignValidation,
                                            ValidationTier::Standard);

    // (hecke-extensions slot 5, WP-2 / phase E2 cont.) THE RECORD'S
    // DETERMINISM GATE ([X] Sec. 5, Sec. 2.4 risk R10).  The FNV-1a-64 of the
    // whole `hecke_ext/v2` record text -- the unsplit one and both per-sign
    // ones -- recomputed on (10, 691), (34, 5) and (24, 7) under
    // ENGINE_THREADS in {1, 4, 16, 32} x ENGINE_BATCH in {0, 1}, plus the
    // large (128, 5) cell on the threads {1, 16} arms and two `device = gpu`
    // arms where a device exists (`[SKIP]` otherwise).  The digest input is
    // the record TEXT and not a hand-picked invariant list, because a
    // hand-picked list is what an order-dependence bug slips past.
    ValidationRegistry::register_validation("HeckeExtDeterminismValidation",
                                            &HeckeExtDeterminismValidation,
                                            ValidationTier::Standard);

    // (hecke-extensions slot 5, WP-8) M1 -- the programme's first
    // owner-visible deliverable: for EVERY case of [E] Sec. 8 that lives in
    // H^0/H^1, a record on X_1 (+) X_4, each ideal `final = true` or
    // `final = false` WITH A REASON FROM THE FIXED LIST of [X] Sec. 1.3.1
    // R3.2, and B'(k, ell) on every row.  The three validations below are
    // the three slices of [E] Sec. 8; each emits its rows through
    // HeckeExt/HeckeM1Table, which is the ONE definition of a well-formed M1
    // row.  None of them emits or reads a golden entry.

    // [E] Sec. 8.3's Kummer list: all ten cells of [E] Claim 1.8 with
    // n < ell - 1 and k <= 26, at the DEFAULT mathematical settings.  Arm 1
    // checks the list itself against slot-4 WP-7's `kummer_divides` /
    // `bernoulli_mod` -- including a completeness sweep over every odd prime
    // <= 20000 -- so the ten cells are not merely confirmed but are the only
    // ones in that range.  Arm 2 runs each cell through HeckeExtDriver:
    // dim X_1 = 0, dim X_4 = dim M_k + dim S_k, the frozen-model U_ell
    // admission (included at ell in {691, 3617, 283, 617, 131, 593, 103},
    // refused at {43867, 2294797, 657931}), EXACTLY ONE non-split
    // Eisenstein-cuspidal ideal with nu = 2 / embdim 1 / Gorenstein / Loewy
    // = socle = (2,1) / D = 3, every other block semisimple with A_S the
    // field F_{ell^f}, and B' = 2 on every cell with the non-split block
    // final ONLY via sturm(B') ([XA B1]).
    //
    // TIER `slow` -- measured, not assumed: the ten cells at the default
    // settings cost 37 s wall, of which U_3617 at (14, 3617) alone is 34 s.
    // [X] Sec. 5 books this validation `slow, ~40 s`; the slot-5 plan's
    // "(standard)" is not reachable without moving `u_ell` off its default,
    // which would make M1 a statement about a non-default configuration.
    // See docs/implementation/slot5/WP-8_report.md Sec. 3.
    ValidationRegistry::register_validation("HeckeExtKummerValidation",
                                            &HeckeExtKummerValidation,
                                            ValidationTier::Slow);

    // [E] Sec. 8.3's small-ell cases.  ASSERTED: the Dickson loci (6,5),
    // (8,7), (12,11) -- dim X_1 == Hecke/DicksonCount.h's count, the H^0
    // character chi_1(p) = p + p^n of [C, cl:chi], dim X_4 = dim M_k +
    // dim S_k + dim X_1; tau(5) == 0 mod 5 and tau(7) == 0 mod 7;
    // B'(60, 5) = 5 = ell with U_5 admitted, the one case whose proof sweep
    // contains U_ell; THE STURM COUNTEREXAMPLES on the pipeline's own H^1 at
    // (22, 5) and (26, 7) -- dim H^1 = 5, dim <A(T_2), A(T_3)> = 2 <
    // dim <A(T_q) : q <= 13> = 3 = dim M_k, so the naive B(k) = 3 sweep would
    // have certified falsely; and B'(34, 131) = 2.  RECORDED (rule A6):
    // (12, 5)'s H^0 character multiset (chi_2 == chi_3 as functions mod 5, so
    // an index LABEL is not distinguishable), the Delta ideal's split verdict
    // at (34, 5) / (58, 7), and the T_ell Jordan data at (58, 5) / (82, 7).
    ValidationRegistry::register_validation("HeckeExtSmallEllValidation",
                                            &HeckeExtSmallEllValidation,
                                            ValidationTier::Standard);

    // [E] Sec. 8.2's two further ell = 691 rows.  (160, 691): the seven
    // blocks 2:1 1:1 2:1 2:1 4:2 6:3 10:5 that the slot-1 probe RECORDED and
    // [X] Sec. 5 says this validation "must assert"; plus the separation
    // claim itself, measured -- the driver is re-run with good_primes {2},
    // {3}, {5}, {2,3}, {2,3,5} and no single T_q separates what {T_2, T_3}
    // does, while {T_2, T_3} already separates as far as {T_2, T_3, T_5}.
    // (198, 691): dim H^1 = 33, blocks 2:1 3:1 4:2 4:2 20:10, exactly one
    // non-split block (the 3:1, Loewy (2,1), Gorenstein) and exactly one
    // 20:10 block (one Galois orbit of residue degree 10, multiplicity 2),
    // with 691 | num(B_200/200) recomputed from `bernoulli_mod`; the index-one
    // "||" and the uniqueness of the congruent eigenform are RECORDED, not
    // asserted ([Q13]).
    ValidationRegistry::register_validation("HeckeExtSeparationValidation",
                                            &HeckeExtSeparationValidation,
                                            ValidationTier::Standard);

    // (slot 6, WP-1 / phase E7a) The contragredient side of E7: rho_n^vee =
    // rho_n(adj .)^T on Mat22::adj, the D_n relation list relation by
    // relation, Phi_n of [C, lem:symdual] (the permutation-definition Gram,
    // both equivariances over Z, rank iff n < ell), the evaluation pairing of
    // [C, prop:pairing] over Z, H^*(D_n) and A(D_n)(T_p)^q for p in
    // {2,3,5,7}; the untwisted relation A(D_n) P = P A(M_n) -- exponent 0,
    // pre-stated from lem:symdual -- asserted for n < ell and n >= ell, with
    // the p^{+n} / p^{-n} readings asserted to FAIL on the non-vacuous cells;
    // the memo stamp's representation field; the golden objects defined, not
    // emitted.  (slot 6, WP-3 / phase E7b, same name and tier) arms A10-A15:
    // the cohomological pairings of Hecke/HeckePairing on 20 cells (WP-1's
    // ten, (10, 691) and the Kummer cells), perfectness ([C, cl:pd](1), OPEN:
    // this is its oracle), the PD gate for X_3 ([C, cl:pd](2),
    // obl:coneexact(vii) third sentence), socle <-> Loewy (identity item 7)
    // and the iota-pairing sign, recorded.
    ValidationRegistry::register_validation("HeckeDualityValidation",
                                            &HeckeDualityValidation,
                                            ValidationTier::Standard);

    // (slot 6, WP-2 / phase E2 (a), OQ11) The on-request presentation of
    // [X] Sec. 2.2 Stage 5: constructed algebras (A^dual over F[x,y]/(x,y)^2
    // with its three relations, cyclic F[x]/(x^3), A + A, the (10, 691) shape
    // A + A/m, the f = 2 analogue, a field) against their known presentations
    // in the stored basis and conjugated, over ell in {5, 13, 691, 134217757};
    // a tamper arm on verify(); at (10, 691) through
    // ExtDriverOptions::with_presentation: H^1_+ free of rank one, the
    // unsplit H^1_m with t = 2 and one minimal relation, and the record texts
    // byte-identical with the option on and off.  The settings-key arm is
    // DEFERRED to slot 7's re-baseline.
    ValidationRegistry::register_validation("HeckePresentationValidation",
                                            &HeckePresentationValidation,
                                            ValidationTier::Fast);
    // (hecke-extensions slot 6, WP-8 / E6) THE RESIDUE ATLAS.  The skeleton
    // gate at ell in {5, 7}: Sigma closed under the theta-twist on the keys
    // AND on the polynomial data (P_p |-> p^j P_p re-split in the canonical
    // field), S_sep separating Sigma u chi, the hecke_atlas/v1 store
    // round-tripping and byte-deterministic across two builds, and the
    // HECKE_ATLAS_OUT write gate closed.  Writes nothing.
    ValidationRegistry::register_validation("HeckeAtlasSkeletonValidation",
                                            &HeckeAtlasSkeletonValidation,
                                            ValidationTier::Fast);
    // The full slot-6 scope (owner ruling 4): Sigma(5), Sigma(7), Sigma(11),
    // Sigma(13) with the skeleton's arms plus the Eisenstein systems and the
    // chi family, Delta mod 11 / mod 13 (tau(q) on the pool, weight ell + 1
    // included, not a lower-weight twist), the Kummer table and the
    // Kummer/Ribet dichotomy on the atlas, the Glover/Brauer arm for
    // n <= 3 ell at ell in {5, 7}, and the per-cell canonical_field_s/total_s
    // column.  `slow` per [X] Sec. 5; writes its stores only under
    // HECKE_ATLAS_OUT=<dir>.
    ValidationRegistry::register_validation("HeckeAtlasValidation",
                                            &HeckeAtlasValidation,
                                            ValidationTier::Slow);
    // (slot 6, WP-13 / console phase C2a) Settings and resume, [P] Sec. 10 C2
    // (f), (g), (h): the engine knobs as `set` keys against their getters
    // (env-only keys refused, reset_settings(), run_concurrently, snapshot at
    // creation, restoration at session end); every hecke::MathSettings::keys()
    // entry round-tripped with its source, errors raised by HeckeSettings, the
    // count PRINTED and reconciled against [X] Sec. 1.5; the exit-5 arm on
    // `field_polynomial = conway`; resume-invariance on Decomposition (locality
    // budget=8 -> 64) and Sturm (cap 3 -> 36).
    ValidationRegistry::register_validation("ConsoleSettingsValidation",
                                            &ConsoleSettingsValidation,
                                            ValidationTier::Fast);
    // (hecke-extensions slot 6, WP-9 / E6) [X] Sec. 5's end-to-end gate, as
    // far as slot 6 reaches: IDENTITY 6 (IdentityCertificate item 6, with
    // items 1-5) on ell in {5, 7, 11}, n <= 200, under [E] cl:glover's
    // standard-module convention, and its NEGATIVE CONTROL (the
    // contragredient reading must fail on the named cell (22, 5)); item 7
    // (socle <-> Loewy under PD, WP-3's) asserted on its cells; the S_sep / Stage 2(3) table (opt-in, default
    // empty) and the proof_range comparison on the ext subgrid and the M1
    // list, RECORDED only.  `slow`.  Writes nothing.
    ValidationRegistry::register_validation("HeckeExtEndToEndValidation",
                                            &HeckeExtEndToEndValidation,
                                            ValidationTier::Slow);
    // (hecke-extensions slot 7, WP-1 / E7 acceptance (b)) [C] cl:pd(1)-(2) --
    // perfectness and the PD gate for X_3 -- on the M1 cells (58, 7), (58, 5),
    // (82, 7), whose cones over D_n are too costly for HeckeDualityValidation,
    // which covers the rest of the E2 case list (arm A16); (160, 691) and
    // (198, 691) (~6 min) only under HECKE_DUALITY_COVERAGE_LARGE=1.  `slow`
    // (~30 s, above the standard tier's cost for one name).  Writes nothing.
    ValidationRegistry::register_validation("HeckeDualityCoverageSlowValidation",
                                            &HeckeDualityCoverageSlowValidation,
                                            ValidationTier::Slow);
    // (hecke-extensions slot 7, WP-8 / console phase C2b) [P] Sec. 10 C2 (a)-(e)
    // and Sec. 11's ConsoleDeterminismValidation, its FIRST arm (threads): every
    // numbered script's transcript under `set threads = 1` and `= 16` identical;
    // the C1 set (001-016) under `set backend = "legacy"` equal to the engine
    // arm; run_validation("GoldenRegressionValidation") == 0 from a script with
    // the chatter captured and the session's threads / device lifted; run_campaign
    // on the CampaignSmokeValidation spec == the driver's summary.tsv outside the
    // timing columns; bench("hecke", 48, ...).dims == cohomology(...).dims; plus
    // `--log` in script mode and gap G9.  `standard`: it runs the script set three
    // times, GoldenRegressionValidation once and the smoke campaign twice.
    ValidationRegistry::register_validation("ConsoleDeterminismValidation",
                                            &ConsoleDeterminismValidation,
                                            ValidationTier::Standard);
    // (hecke-extensions slot 8, WP-5; docs/KNOWN_ISSUES.md Sec. 4c) the
    // between-cells program-cache release (Campaign/ProgramCacheRelease.h):
    // (A) both global caches cleared above the threshold, the count returned;
    // (B) a recompile after a clear is byte-identical (a pipeline point and a
    // uint64_t tape program); (C) one thread recomputes while another clears
    // in a loop -- byte-identical, and the CheckThread proof that a clear
    // concurrent with run_cached() is race-free.  `fast` (~1 s).  Writes nothing.
    ValidationRegistry::register_validation("ProgramCacheReleaseValidation",
                                            &ProgramCacheReleaseValidation,
                                            ValidationTier::Fast);
    // (hecke-extensions slot 8, WP-1 / E8 item 2) HeckeReductionValidation's
    // ELIMINATION arms, ENGINE_API.md Sec. 13.9: the Z/ell^k ring
    // (ModCtx::prime_power at (2,2) (2,3) (3,2) (5,2) (5,3) (691,2) u32,
    // (134217757,2) u64 Montgomery, (2,40) u64 U64_Wide; 10^6 samples each),
    // every Sec. 13.3 entry point refusing a composite context and every
    // Sec. 13.6 entry point a field, the Sec. 13.7 replay check; elim::howell ==
    // ref::howell entrywise (random + structured, uniqueness), kernel_module /
    // image_module / solve_multi_rhs_module / invariant_factors against the
    // oracle and their identities, the k = 1 field path == RREF, and review
    // M1's arm (invariant_factors(Z/4, [[2],[1]]) = (2), image_module pivot
    // valuations (1, 1)).  Seeds and counts printed.  `standard` (target
    // <= 10 s).  Writes nothing.  E8 items 3-4 append the pipeline arms.
    ValidationRegistry::register_validation("HeckeReductionValidation",
                                            &HeckeReductionValidation,
                                            ValidationTier::Standard);
    // (hecke-extensions slot 8, WP-8 / Line 2, proposal V-B; [IB] Sec. 6.2 V-B,
    // Sec. 2.5(a)) the Bredon-identification validation: for ell in {2, 3},
    // dim H^q of the shipped 7-vertex pipeline equals the independent
    // Bass-Serre tree model (V^S (+) V^U -> V^{-I}, S = [[0,-1],[1,0]],
    // U = [[0,-1],[1,1]]), checked first against the frozen 16-golden-cell
    // oracle of the design brief's B-1 table (stopping without adjustment on
    // any disagreement), then over every n in {0, ..., 40} including odd n.
    // `fast` (indicative wall < 1 s). Writes nothing.
    ValidationRegistry::register_validation("HeckeBredonTreeValidation",
                                            &HeckeBredonTreeValidation,
                                            ValidationTier::Fast);
    // (hecke-extensions slot 8, WP-7 / Line 2, milestone M-0; [IB] Sec. 6.2
    // M-0) char_Z(T_p | Lambda^1) by CRT over existing F_ell runs: certified
    // equality over Z with the frozen O5 table for n in {10, ..., 20}, p = 2
    // (and n = 10, p = 3), every residue compared digit-wise, one held-out
    // prime per cell; at n = 64 (and n = 128 under HECKE_INTEGRAL_CAMPAIGN=1,
    // read-only) the residue vectors and the certified bound, recorded, with
    // the oracle-free checks (O3 degree, Eisenstein factor, square cofactor).
    // `standard`. Writes nothing.
    ValidationRegistry::register_validation("HeckeIntegralCharpolyValidation",
                                            &HeckeIntegralCharpolyValidation,
                                            ValidationTier::Standard);
    // (hecke-extensions slot 8, WP-10 / console phase C4) [P] Sec. 10 C4 (b)
    // and Sec. 11: every row of [P] Sec. 8 reaches its exit code with
    // file:line:col (and the creation site for a lazy failure), through real
    // script files on the --script path; the exit-1 rows; a failed handle is
    // rethrown and never retried; the REPL survives every error; and after the
    // induced throws (model mismatch, non-commuting operators, ...) the same
    // session recomputes golden objects of (10, 691) and matches golden(...),
    // under device = cpu and, with a usable device, gpu.  Test-only verbs live
    // on its own registry.  `fast`.  Writes only a temporary directory it removes.
    // (The same package extends the already-registered ConsoleDeterminismValidation
    // with C4 (a), the threads x program_cache x device product; its line and
    // tier do not move.)
    ValidationRegistry::register_validation("ConsoleErrorValidation",
                                            &ConsoleErrorValidation,
                                            ValidationTier::Fast);

    // (hecke-extensions slot 8, WP-4 / E8 item 4a (d) on the bulk) the reduction
    // and Bockstein equivariances -- A(T_p) o red = red o A(T_p) on H^0..H^2,
    // beta o A(T_p) = A(T_p) o beta, the coefficient long exact sequence with
    // its length bookkeeping and bockstein_rank^0 + higher_torsion_count ==
    // dim H^0(F_ell) -- at Z/ell^2 on the E2 case list at ell >= 5 (the 31
    // ext-subgrid cells and the M1 cells, 52 distinct), p in {2, 3, 5};
    // (160, 691) and (198, 691) only under the read-only interlock
    // HECKE_REDUCTION_COVERAGE_LARGE=1.  `slow` (above the standard tier's
    // cost for one name; the five pre-stated cells stay in the standard
    // HeckeReductionValidation).  Writes nothing.
    ValidationRegistry::register_validation("HeckeReductionCoverageSlowValidation",
                                            &HeckeReductionCoverageSlowValidation,
                                            ValidationTier::Slow);

    // --- Insertion point for future Hecke*Validation entries -----------------
    // Later stages (...) add their own
    // `uint32_t Hecke*Validation()` functions under src/hacs/Hecke/
    // and register them here.
    // ---------------------------------------------------------------------

    // (slot-9 WP-1) per-package fragments, in package-number order, after every
    // existing registration; a package registers ONLY in its own fragment.
    RegisterSlot9();
    RegisterSlot10();
    RegisterSlot11();
    RegisterSlot12();
    RegisterSlot13();
    RegisterSlot14();
    RegisterSlot15();
    RegisterSlot16();
    RegisterSlot17();
    RegisterSlot18();
    RegisterSlot19();
    RegisterSlot20();
    RegisterSlot21();
    RegisterSlot22();
    RegisterSlot23();
    RegisterSlot24();
    RegisterSlot25();
    RegisterSlot26HeckeZUp();
    RegisterSlot26HeckeZLocalPar();
    RegisterSlot26HeckeZEisUp();
    RegisterSlot26HeckeHidaDiscLambda();
}
