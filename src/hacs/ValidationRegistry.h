// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef VALIDATION_REGISTRY_H
#define VALIDATION_REGISTRY_H

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

// A validation is a zero-argument function returning 0 on success (I9: the
// codebase-wide "functions returning 0" convention for validations).
using ValidationFn = uint32_t (*)();

// Phase-0 owner decision Q13 (docs/matrix-engine/00-implementation-plan.md
// Sec. 7): every registration carries a TIER, so the acceptance gate can be
// split into a cheap edit-loop suite and a full one.
//
//   Fast     -- well under a second; trivially runnable on every save.
//   Standard -- the default; seconds to a few tens of seconds.
//   Slow     -- over ~60 s at RelWithDebInfo, or an explicit benchmark.
//
// `--all` runs Fast + Standard (the exit-code contract is unchanged: exit
// code = number of failures). `--all-full` runs every tier. `--list` prints
// the tier next to each name.
//
// CONVENTION FOR LATER PHASE-0 WORK (for the integrator): the golden-output
// regression `GoldenRegressionValidation` registers as `Standard`; the
// determinism/parameter grids register as `Slow`.
enum class ValidationTier : uint8_t {
    Fast = 0,
    Standard = 1,
    Slow = 2
};

// "fast" / "standard" / "slow" -- the spelling used by --list and --tier.
const char* ValidationTierName(ValidationTier tier);

// Name-addressable registry of validation functions, so any registered
// validation can be run standalone via `./cohomology_app <Name>` without
// editing main() (see docs/implementation/hecke_action_implementation_plan.md,
// Stage S0).
//
// Registration is an EXPLICIT static table populated by RegisterAllValidations()
// (declared below), not self-registering static objects: self-registering
// statics defined in the ModArithmetic/cohomology_app translation units that
// are never referenced elsewhere can be dropped by the linker across the
// static-library boundary. RegisterAllValidations() is defined in one
// translation unit (ValidationRegistry.cpp) with explicit
// register_validation(...) calls, and must be invoked once (e.g. from main())
// before run()/run_all()/list() are used.
class ValidationRegistry {
public:
    // Adds (or replaces, if the name already exists) an entry in the table.
    // `tier` defaults to Standard so an un-tiered registration keeps the
    // pre-Phase-0 behaviour (it runs under plain `--all`).
    static void register_validation(const std::string& name, ValidationFn fn,
                                    ValidationTier tier = ValidationTier::Standard);

    // Runs the validation with the given name, printing PASS/FAIL.
    // Returns the validation's own return code (0 = success); returns a
    // nonzero code if no validation with that name is registered.
    static uint32_t run(const std::string& name);

    // Runs registered validations in registration order, printing PASS/FAIL
    // for each. Returns the number of validations that failed (0 = all
    // passed) -- the exit-code contract of `--all`, unchanged.
    // include_slow == false (the default, `--all`) skips ValidationTier::Slow
    // entries, reporting them as SKIPped; they never count as failures.
    // include_slow == true (`--all-full`) runs everything.
    static uint32_t run_all(bool include_slow = false);

    // (S24 SHARD) `--all [--full roster] --shard i/N`: runs shard `index` (1-based) of `count` of the
    // tier's roster (ValidationShard.h: weight-balanced LPT, device names in one shard), in registration
    // order.  Prints `SHARD i/N names=<k> of <total> ...` first and the run_all summary line last; slow
    // names outside the roster are not printed.  Returns the failure count.  `list_only`: print the
    // header and the shard's `name<TAB>[tier]` lines instead of running them (exit 0).
    // In a shard that is not the device shard, a validation that creates the CUDA context fails
    // ("[SHARD-DEVICE] ..."), unless ENGINE_DEVICE selects a device for every name anyway.
    static uint32_t run_shard(bool include_slow, uint32_t index, uint32_t count, bool list_only);

    // Names of all registered validations, in registration order.
    static std::vector<std::string> list();

    // (name, tier) of all registered validations, in registration order.
    static std::vector<std::pair<std::string, ValidationTier>> list_with_tiers();

    // Tier of a registered validation; ValidationTier::Standard if unknown.
    static ValidationTier tier_of(const std::string& name);

    // (slot-14 WP-2; S13 WP-16 catch-path test) Runs a single validation
    // FUNCTION -- not a registered NAME -- through exactly the same
    // exception-catching path run()/run_all() use: an exception escaping
    // `fn` (std::exception or not) prints "[EXCEPTION] <name>: ..." then
    // "[FAIL] <name> (code 1)" and this returns 1; a normal return prints
    // "[PASS] <name>" (0) or "[FAIL] <name> (code N)" (N != 0) as usual, and
    // this returns that code. `fn` is never added to the global table, so
    // this is how ValidationHarnessCatchValidation exercises the guard on a
    // deliberately-throwing function without registering a throwing name.
    static uint32_t run_guarded_standalone(const std::string& name, ValidationFn fn);
};

// Populates the registry with all known validations: the six pre-existing
// validations plus (as later Hecke stages land) each new Hecke*Validation.
// Defined in ValidationRegistry.cpp; safe to call more than once.
void RegisterAllValidations();

#endif // VALIDATION_REGISTRY_H
