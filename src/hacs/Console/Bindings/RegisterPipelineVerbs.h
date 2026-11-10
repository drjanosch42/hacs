// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// RegisterPipelineVerbs.h -- console phase C1a: the ONE explicit registration
// table (plan Sec. 2 C3, Sec. 9).
//
// Registration is EXPLICIT and never by a self-registering static, which the
// linker drops across a static-library boundary -- the reason
// ValidationRegistry.h:43-50 records, and the reason ConsoleCore's registry
// takes the same shape.
//
// main.cpp's one console branch calls the no-argument form; the validations
// call the Registry& form on their OWN registry, so a test cannot disturb (or
// be disturbed by) the process-wide one.
// ===========================================================================
#pragma once
#ifndef CONSOLE_REGISTER_PIPELINE_VERBS_H
#define CONSOLE_REGISTER_PIPELINE_VERBS_H

#include <string>
#include <vector>

namespace console {

class Registry;

// Registers the core verbs and every C1a binding on `registry`, in the fixed
// order the overload resolution of plan Sec. 2 C3 depends on, and installs the
// C7 exception seam.
void RegisterPipelineVerbs(Registry& registry);

// The same on Registry::global() -- the form main.cpp calls.  Idempotent: a
// second call on the same registry is a no-op, so an entry point that is
// reached twice cannot double-register an overload.
void RegisterPipelineVerbs();

// The console TYPE names C1a registers, in registration order.  Used by the
// validation (and by `help`) so the list cannot drift from the code.
const std::vector<std::string>& PipelineTypeNames();

// The verb names C1a adds ON TOP of RegisterCoreVerbs, in registration order.
const std::vector<std::string>& PipelineVerbNames();

// The verbs that must NOT exist before phase C1b -- the narrowed form of C0's
// "no transcript freezing before E0b" guard (plan Sec. 10, Phase C1a): the
// decomposition surface and the report.  `hash`, `golden`, `cohomology` and
// `hecke` ARE registered by C1a and are no longer forbidden.
const std::vector<std::string>& VerbsForbiddenBeforeC1b();

// (C1b) The successor guard: the phase C2 / C5 reserved names that must NOT be
// registered yet.  VerbsForbiddenBeforeC1b() is empty since C1b shipped the
// decomposition surface, so this is what keeps the phase boundary checkable.
const std::vector<std::string>& VerbsForbiddenBeforeC2();

} // namespace console

#endif // CONSOLE_REGISTER_PIPELINE_VERBS_H
