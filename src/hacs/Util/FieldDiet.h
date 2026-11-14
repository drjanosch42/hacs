// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Util/FieldDiet.h -- the ONE switch of the slot-24 G6 field-route RSS diet.
//
// Header-only; STL only apart from ModArithmetic/LegacyConfig.h (itself STL
// only, but it #errors unless the target defines OC_ENABLE_LEGACY).  A thread-local flag whose initial state on every
// thread is read from the environment: HECKE_FIELD_DIET=0 (exactly "0"; retired in
// OC_ENABLE_LEGACY=OFF builds by S26 LEVERS-26, see field_diet_env_on())
// selects the rollback, anything else or unset leaves the diet on.  Every G6
// lever reads it -- HeckeOperatorOptions::field_diet() (program 1 in parts,
// the family release, malloc_trim), ConeOptions::unit_row_span() (the cone
// certificate's span route) and the boundary morphism's lean basis read
// (EquivariantCochainComplexMorphismOnK) -- so one ScopedFieldDiet switches
// them together, and HeckeOperatorOptions' registered thread context carries
// the flag into the engine's host fan-out threads.  None of the levers can
// change a value (HeckeOperatorOnK.h, HeckeOperatorOptions (6)).
// ===========================================================================
#pragma once
#ifndef OC_UTIL_FIELD_DIET_H
#define OC_UTIL_FIELD_DIET_H

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include "../../ModArithmetic/LegacyConfig.h"   // (S26 LEVERS-26) the env arm's refusal

namespace oc_util {

// (S26 LEVERS-26) The env arm HECKE_FIELD_DIET=0 is RETIRED in OC_ENABLE_LEGACY=OFF builds: exactly "0" is refused
// (`REFUSED HECKE_FIELD_DIET=0: compiled out (OC_ENABLE_LEGACY=OFF)`, stderr, exit 1), here at the first read and at
// start-up in main(); every other value reads as before.  The undieted route stays in every build through
// ScopedFieldDiet(false) / set_field_diet_on(false) (HeckeThetaRssDietValidation, HeckeExtDeterminismValidation).
// Evidence: (R-a) of slot26/00-slot26-plan.md Sec. 1 item 7 (slot26/LEVERS26_plan.md Sec. 1).  ON builds: unchanged.
inline bool field_diet_env_on() {
    const char* v = std::getenv("HECKE_FIELD_DIET");
    const bool on = v == nullptr || std::strcmp(v, "0") != 0;
#if !OC_ENABLE_LEGACY
    if (!on) oc::legacy::refuse_compiled_out(std::string("HECKE_FIELD_DIET=") + v);
#endif
    return on;
}

// (G6 audit F2) The EFFECTIVE line of the slot-14 convention, printed by the
// FIRST reader of the flag in the process -- whichever lever that is (a cone
// or a boundary morphism built without any Hecke operator reads the flag too)
// -- and only when the variable is set: what the variable selects, not a
// scoped override.  One magic static: once per process, thread-safe.
inline void field_diet_announce() {
    static const bool announced = []() {
        if (const char* v = std::getenv("HECKE_FIELD_DIET"))
            std::cout << "EFFECTIVE HECKE_FIELD_DIET field_diet=" << (field_diet_env_on() ? "on" : "off")
                      << " (HECKE_FIELD_DIET=" << v << ")" << std::endl;
        return true;
    }();
    (void)announced;
}

inline bool& field_diet_flag() {
    static thread_local bool flag = (field_diet_announce(), field_diet_env_on());
    return flag;
}

inline bool field_diet_on() { return field_diet_flag(); }
inline void set_field_diet_on(bool on) { field_diet_flag() = on; }

} // namespace oc_util

#endif // OC_UTIL_FIELD_DIET_H
