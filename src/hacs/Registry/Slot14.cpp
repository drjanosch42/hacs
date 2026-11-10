// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Slot-14 registration (closed slot): the former per-package fragments Registry/Slot14/<pkg>.cpp folded verbatim
// (S24 CL-10), each package function kept; RegisterSlot14() calls them in the old RegisterAllValidations() order.

// ===== Slot 14 WP2 ===========================================
// Slot-14 WP-2 registration fragment (hecke-extensions slot 14). Owned by
// package WP-2 ONLY: register this package's validations here and nowhere else
// in the shared structure. Called explicitly from RegisterAllValidations()
// (never a static initialiser); its sources are listed in cmake/slot14/WP2.cmake.
//
// (hecke-extensions slot 14, WP-2) Harness and kit hygiene. This file carries
// the plan's item 4: "the WP-16 catch path gets its test" -- S13 WP-16 made an
// exception escaping a validation into that validation's [FAIL] instead of
// std::terminate (ValidationRegistry.cpp's RunGuardedFn / DeviceReleaseOnExit),
// but nothing exercised the catch path itself: no validation was engineered to
// throw and checked to come back [FAIL] rather than abort. ValidationHarnessCatchValidation
// closes that gap WITHOUT registering a throwing name in the global table --
// it drives two private, unregistered throwing functions (one std::exception,
// one not) through ValidationRegistry::run_guarded_standalone(), the exact
// same guard run()/run_all() use, capturing stdout to assert on the printed
// [EXCEPTION]/[FAIL] lines, that a THIRD (clean) function still runs
// afterwards (the suite really continues), and that the failure tally is
// exactly one per thrower.
#include "../ValidationRegistry.h"
#include "../ValidationShard.h"  // (S24 SHARD) the partition self-check arm

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

// Side-effect flags: the only way to observe that a plain ValidationFn
// (uint32_t(*)(), no captures) actually ran.
bool g_ran_thrower_std = false;
bool g_ran_thrower_nonstd = false;
bool g_ran_next = false;

uint32_t PrivHarnessCatchThrowsLogicError() {
    g_ran_thrower_std = true;
    throw std::logic_error(
        "ValidationHarnessCatchValidation: engineered std::logic_error (not a real failure)");
}

// A deliberately non-std::exception thrown type, to exercise the guard's
// catch (...) branch as well as its catch (const std::exception&) branch.
struct PrivHarnessCatchNonStdException {};

uint32_t PrivHarnessCatchThrowsNonStd() {
    g_ran_thrower_nonstd = true;
    throw PrivHarnessCatchNonStdException{};
}

// Runs after the two throwers: if the suite really "goes on" (an escaping exception is that validation's failure),
// this must still execute and return 0.
uint32_t PrivHarnessCatchNextPasses() {
    g_ran_next = true;
    return 0;
}

} // namespace

uint32_t ValidationHarnessCatchValidation() {
    std::cout << "=== ValidationHarnessCatchValidation (slot-14 WP-2; the S13 WP-16 catch path) ==="
              << std::endl;

    g_ran_thrower_std = false;
    g_ran_thrower_nonstd = false;
    g_ran_next = false;

    // (slot-14 WP-16, audit M-15) Only std::cout is captured (its rdbuf is
    // swapped): a thrower that wrote to std::cerr, or through printf / the C
    // stdout, would escape the capture.  Both throwers here are private and
    // write nothing themselves; whoever extends this must keep it that way or
    // capture those streams too.
    // Capture stdout for the duration of the three guarded calls so the
    // printed [EXCEPTION]/[FAIL]/[PASS] lines can be asserted on, not just
    // trusted. Restored unconditionally (even if the guard itself somehow
    // let something escape, which would itself be a finding).
    std::ostringstream captured;
    std::streambuf* const old_buf = std::cout.rdbuf(captured.rdbuf());
    uint32_t r1 = 1, r2 = 1, r3 = 1;
    bool guard_escaped = false;
    try {
        r1 = ValidationRegistry::run_guarded_standalone("PrivHarnessCatchThrowsLogicError",
                                                        &PrivHarnessCatchThrowsLogicError);
        r2 = ValidationRegistry::run_guarded_standalone("PrivHarnessCatchThrowsNonStd",
                                                        &PrivHarnessCatchThrowsNonStd);
        r3 = ValidationRegistry::run_guarded_standalone("PrivHarnessCatchNextPasses",
                                                        &PrivHarnessCatchNextPasses);
    } catch (...) {
        guard_escaped = true;
    }
    std::cout.rdbuf(old_buf);
    const std::string text = captured.str();
    // Keep the captured lines visible in the real log too, but INDENTED under a
    // `captured:` prefix (slot-14 WP-16, audit S-7): the two engineered `[FAIL]`
    // lines must not appear at column 0, where `grep '^\[FAIL\]'` -- the triage
    // habit of every report -- would count them.  The assertions below search
    // `text`, the raw capture, so they are unaffected.
    {
        std::istringstream lines(text);
        std::string line;
        while (std::getline(lines, line)) std::cout << "  captured: " << line << "\n";
        std::cout << std::flush;
    }

    uint32_t checks = 0, fails = 0;
    auto check = [&](bool cond, const char* what) {
        ++checks;
        if (!cond) {
            ++fails;
            std::cout << "  [CHECK-FAIL] " << what << std::endl;
        }
    };

    check(!guard_escaped, "run_guarded_standalone() itself never lets an exception escape");
    check(g_ran_thrower_std, "the std::logic_error thrower actually ran (not short-circuited)");
    check(g_ran_thrower_nonstd, "the non-std thrower actually ran (not short-circuited)");
    check(g_ran_next, "the validation AFTER both throwers still ran -- the harness does not abort");
    check(r1 == 1, "the std::logic_error thrower's guarded result is 1");
    check(r2 == 1, "the non-std thrower's guarded result is 1");
    check(r3 == 0, "the clean validation after the throwers returns 0");

    const auto pos_std_exc = text.find("[EXCEPTION] PrivHarnessCatchThrowsLogicError: ");
    const auto pos_std_fail = text.find("[FAIL] PrivHarnessCatchThrowsLogicError (code 1)");
    const auto pos_nonstd_exc = text.find("[EXCEPTION] PrivHarnessCatchThrowsNonStd: ");
    const auto pos_nonstd_fail = text.find("[FAIL] PrivHarnessCatchThrowsNonStd (code 1)");
    const auto pos_next_pass = text.find("[PASS] PrivHarnessCatchNextPasses");

    check(pos_std_exc != std::string::npos,
          "std::logic_error prints \"[EXCEPTION] <name>: ...\"");
    check(pos_std_fail != std::string::npos && (pos_std_exc == std::string::npos || pos_std_exc < pos_std_fail),
          "... then \"[FAIL] <name> (code 1)\", in that order");
    check(pos_nonstd_exc != std::string::npos,
          "the non-std throw ALSO prints \"[EXCEPTION] <name>: ...\" (the catch (...) branch)");
    check(pos_nonstd_fail != std::string::npos &&
              (pos_nonstd_exc == std::string::npos || pos_nonstd_exc < pos_nonstd_fail),
          "... then \"[FAIL] <name> (code 1)\", in that order");
    check(pos_next_pass != std::string::npos, "the next validation prints its own \"[PASS] <name>\"");
    check(pos_std_fail != std::string::npos && pos_nonstd_exc != std::string::npos &&
              pos_std_fail < pos_nonstd_exc && pos_nonstd_fail != std::string::npos &&
              pos_nonstd_fail < pos_next_pass,
          "the three ran in registration order: thrower 1, thrower 2, then the clean one");

    const uint32_t failure_count = r1 + r2 + r3; // 0/1 codes here, so this IS the tally
    check(failure_count == 2, "the failure count is exactly 1 per thrower (2 total), never an abort");

    // (S24 SHARD) The `--shard i/N` partition is harness too: union = roster, disjoint, the device group in
    // one shard, reproducible, malformed specs refused, for several N on both rosters (ValidationShard.h).
    // An arm of this validation rather than a new name, so the roster stays at 267.
    fails += ShardPartitionSelfCheck(checks);

    std::cout << "ValidationHarnessCatchValidation: " << checks << " checks, " << fails << " failing"
              << std::endl;
    return fails;
}

void RegisterSlot14WP2() {
    ValidationRegistry::register_validation("ValidationHarnessCatchValidation",
                                            &ValidationHarnessCatchValidation, ValidationTier::Fast);
}

// ===== Slot 14 WP3 ===========================================
// Slot-14 WP-3 registration fragment (hecke-extensions slot 14). Owned by
// package WP-3 ONLY: register this package's validations here and nowhere else
// in the shared structure. Called explicitly from RegisterAllValidations()
// (never a static initialiser); its sources are listed in cmake/slot14/WP3.cmake.
#include "../ValidationRegistry.h"

void RegisterSlot14WP3() {
    // (hecke-extensions slot 14, WP-3) Engine guards G1 / G2 and device-memory
    // release between validations. WP-3 added arms only -- FlatMatrixValidation
    // (G1, 112 -> 119 tests) and EngineCudaMemoryValidation (H, 17 -> 31
    // checks) -- so no name is registered here, per [S14] Sec. 4. The release
    // hook itself is Util/DeviceRelease.cpp (listed in cmake/slot14/WP3.cmake),
    // called from ValidationRegistry's RunGuarded.
}

// ===== Slot 14 WP6 ===========================================
// Slot-14 WP-6 registration fragment (hecke-extensions slot 14). Owned by
// package WP-6 ONLY: register this package's validations here and nowhere else
// in the shared structure. Called explicitly from RegisterAllValidations()
// (never a static initialiser); its sources are listed in cmake/slot14/WP6.cmake.
#include "../ValidationRegistry.h"
#include "../HeckeExt/HeckeRingLoewyValidation.h"   // slot-14 WP-6

void RegisterSlot14WP6() {
    // (hecke-extensions slot 14, WP-6) The Z/ell^K Hecke-algebra route in C++
    // ([M1] route R1: A_K (x) F_ell, eps_K, Loewy layers, torsion blocks included).
    // `standard`; HECKE_RING_LOEWY_CELLS="n:ell[:KMAX],..." runs the route over a
    // cell list instead (read-only, not an emitter; LOEWY_RING lines).
    ValidationRegistry::register_validation("HeckeRingLoewyValidation",
                                            &HeckeRingLoewyValidation,
                                            ValidationTier::Standard);
}

// ===== Slot 14 WP10 ===========================================
// Slot-14 WP-10 registration fragment (hecke-extensions slot 14). Owned by
// package WP-10 ONLY: register this package's validations here and nowhere
// else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser); its sources are
// listed in cmake/slot14/WP10.cmake.
#include "../ValidationRegistry.h"
#include "../Hecke/HeckeUEllStreamValidation.h"   // slot-14 WP-10
#include "LegacyValidation.h"                     // OC_LEGACY_VALIDATION (S24 CL-R1)

void RegisterSlot14WP10() {
    // (hecke-extensions slot 14, WP-10) The coset streaming lever
    // (HeckeOperatorOptions::ScopedCosetStreaming) and the serial phase on the
    // pool (ScopedParallelSetup), both DEFAULT ON since the owner decision of
    // 2026-09-28 (Q8), each with an opt-out: bit-identity with and without
    // each lever, alone and composed with the slot-13 U_ell lever.  (S24 CL-R1)
    // The levers are retired: an OC_ENABLE_LEGACY=OFF build prints [SKIP].
    ValidationRegistry::register_validation("HeckeUEllStreamValidation",
                                            OC_LEGACY_VALIDATION(HeckeUEllStreamValidation),
                                            ValidationTier::Slow);  // slow since S24 VAL-IMPL (VALSTREAM_audit_1030.md row 5)
}

// ===== Slot 14 WP12 ===========================================
// Slot-14 WP-12 registration fragment (hecke-extensions slot 14). Owned by
// package WP-12 ONLY: register this package's validations here and nowhere
// else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser); its sources are
// listed in cmake/slot14/WP12.cmake. GATED (owner Q4, answered yes 2026-09-29).
#include "../ValidationRegistry.h"
#include "../Hecke/HeckeLevelNValidation.h"   // slot-14 WP-12

void RegisterSlot14WP12() {
    // (hecke-extensions slot 14, WP-12) Level N on the fixed model by Shapiro,
    // first build -- [C] v1.14.0 sec:levelN (def:cosetrepsN, def:levelNmodel,
    // rmk:levelNcheckable (LN1), (LN2), the tau half of (LN3), (LN4)-(LN6) at the
    // R3 cells) with an independent group-cocycle route.  Writes nothing;
    // LEVEL_N / LEVEL_N_R3 lines on stdout.
    ValidationRegistry::register_validation("HeckeLevelNValidation",
                                            &HeckeLevelNValidation,
                                            ValidationTier::Standard);
}

// ===== Slot 14 WP14c ===========================================
// Slot-14 WP-14c registration fragment (hecke-extensions slot 14). Owned by
// package WP-14c ONLY: register this package's validations here and nowhere
// else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser); its sources are
// listed in cmake/slot14/WP14c.cmake. STRETCH (gated on WP-9 ACCEPT): may
// stay empty at the close.
#include "../ValidationRegistry.h"

void RegisterSlot14WP14c() {
    // (hecke-extensions slot 14, WP-14c) [U]'s reading code
    // (HeckeExt/HeckeHidaComponents): empty at day zero. WP-14c registers
    // HeckeHidaComponentsValidation here, if it runs.
}

void RegisterSlot14() {
    RegisterSlot14WP2();
    RegisterSlot14WP3();
    RegisterSlot14WP6();
    RegisterSlot14WP10();
    RegisterSlot14WP12();
    RegisterSlot14WP14c();
}
