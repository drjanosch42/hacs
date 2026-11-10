// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Util/FieldDiet.h -- the ONE switch of the slot-24 G6 field-route RSS diet.
//
// Header-only, STL only.  A thread-local flag whose initial state on every
// thread is read from the environment: HECKE_FIELD_DIET=0 (exactly "0")
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

namespace oc_util {

inline bool field_diet_env_on() {
    const char* v = std::getenv("HECKE_FIELD_DIET");
    return v == nullptr || std::strcmp(v, "0") != 0;
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
