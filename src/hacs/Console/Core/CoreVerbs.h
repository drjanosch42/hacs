// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// CoreVerbs.h -- console C0: the verbs ConsoleCore itself provides.
//
// Plan Sec. 2 C3: "populated by EXPLICIT registration functions called from
// main.cpp -- console::RegisterCoreVerbs (ConsoleCore's own types: Int, Bool,
// String, List, Record) ...".  Everything that needs a matrix, a polynomial or
// a pipeline object is a BINDING (phase C1) and is registered by
// RegisterPipelineVerbs, which this library never sees.
//
// Deliberately ABSENT from C0, and the reason:
//
//   hash(x) / golden(point, object)   the reproducibility pins of plan
//                                     Sec. 2 C5.  Slot 1 lane C (see
//                                     hecke_extensions_slot1_plan.md Sec. 4.1)
//                                     freezes NO decomposition transcript
//                                     before the E0b re-baseline, and the
//                                     cheapest way to guarantee that is for the
//                                     language core to be unable to express a
//                                     pin at all.  ConsoleC0Validation asserts
//                                     that neither verb exists.
//   timed / bench / capture           need a clock, RSS and lazily evaluated
//                                     arguments; C2.
//
// STL-ONLY (plan Sec. 2 C8).
// ===========================================================================
#pragma once
#ifndef CONSOLE_CORE_VERBS_H
#define CONSOLE_CORE_VERBS_H

#include <string>
#include <vector>

namespace console {

class Registry;

// Registers every core verb on `registry`.  Idempotent in effect but not in
// count: calling it twice would register each overload twice, so the shipped
// console calls it exactly once per registry.
void RegisterCoreVerbs(Registry& registry);

// The names RegisterCoreVerbs installs, in registration order.  Used by
// ConsoleC0Validation (and by `help`) so the list cannot drift from the code.
const std::vector<std::string>& CoreVerbNames();

// The names C0 must NOT provide -- the transcript-freezing guard above.
const std::vector<std::string>& VerbsForbiddenInC0();

} // namespace console

#endif // CONSOLE_CORE_VERBS_H
