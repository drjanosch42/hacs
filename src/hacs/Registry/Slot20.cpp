// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Slot-20 registration (closed slot): the former per-package fragments Registry/Slot20/<pkg>.cpp folded verbatim
// (S24 CL-10), each package function kept; RegisterSlot20() calls them in the old RegisterAllValidations() order.

// ===== Slot 20 WPLV3 ===========================================
// Slot-20 WPLV3 registration fragment (hecke-extensions slot 20). Owned by
// package WPLV3 ONLY: register this package's validations here and nowhere
// else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser); its sources are
// listed in cmake/slot20/WPLV3.cmake.
#include "../ValidationRegistry.h"

void RegisterSlot20WPLV3() {
    // empty at WP-1 (day zero)
}

// ===== Slot 20 WPMEAS ===========================================
// Slot-20 WPMEAS registration fragment (hecke-extensions slot 20). Owned by
// package WPMEAS ONLY: register this package's validations here and nowhere
// else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser); its sources are
// listed in cmake/slot20/WPMEAS.cmake.
#include "../ValidationRegistry.h"
#include "../HeckeExt/LValue/HeckeLValueMeasureValidation.h"

void RegisterSlot20WPMEAS() {
    // (WP-MEAS) the measure-valued evaluation of the linear route ([E] sec:lvmeasure, gate chk:lvmeas): the ell-stabilised
    // Mazur-Tate-Teitelbaum measure of an ordinary eigensymbol against R20-1's frozen values, the PROVED identities, the
    // Eisenstein congruence of measures against two Kubota-Leopoldt group-ring routes (E_{1,c}, NumberField's CwLayer), the
    // route on the C++ module cone at (30,37) (arm M0, kept standard), negative controls. HECKE_LVALUE_MEASURE_HEAVY=1 (read-only)
    // adds (226,103) at level 3, (282,37) at R = 5 and every character.
    ValidationRegistry::register_validation("HeckeLValueMeasureValidation", &HeckeLValueMeasureValidation, ValidationTier::Standard);
}

// ===== Slot 20 NB4 ===========================================
// Slot-20 NB4 registration fragment (hecke-extensions slot 20). Owned by
// package NB4 ONLY: register this package's validations here and nowhere
// else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser); its sources are
// listed in cmake/slot20/NB4.cmake.
#include "../ValidationRegistry.h"
#include "../Console/ConsoleServeAtlasValidation.h"

void RegisterSlot20NB4() {
    // (NB-4, [P] v3.11 Sec. 18.16 (h)) The atlas sidecar writer
    // (`--atlas --index`) and `--serve-atlas` (oc-serve-atlas/1), driven as
    // children over pipes on Sigma(23) and Sigma(13) built in process (each
    // written into a scratch directory under
    // a scratch directory, removed afterwards): the
    // index against the in-process index, every reply against
    // ParseAtlasStore, sidecar <-> memory byte-identical, paging and the frame
    // cap, errors, stale / corrupt index refused, the env gate and hardening,
    // the stores never written.  Optional read-only real-store arm under
    // HECKE_ATLAS_IN=<dir>.  Always registered (the --list count is
    // switch-independent); `[SKIP] ConsoleServeAtlasValidation:
    // OC_BUILD_NOTEBOOK=OFF` in a default build.
    ValidationRegistry::register_validation("ConsoleServeAtlasValidation", &ConsoleServeAtlasValidation,
                                            ValidationTier::Standard);
}

void RegisterSlot20() {
    RegisterSlot20WPLV3();
    RegisterSlot20WPMEAS();
    RegisterSlot20NB4();
}
