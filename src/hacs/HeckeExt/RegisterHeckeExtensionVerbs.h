// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// RegisterHeckeExtensionVerbs.h -- the extension layer's OWN console verbs
// ([OD] hecke-extensions slot-3 close-out 1; slot-4 plan Sec. 2.3 item 4 and
// Sec. 6 confirmation 4; [P] v3.1 Sec. 3.6 "The extension layer's namespace is
// hecke_ext, so its registration TU is HeckeExt/RegisterHeckeExtensionVerbs.cpp").
//
// WHY HERE AND NOT IN Console/Bindings/.  check_heckeext_includes.sh RULE 1
// forbids anything outside HeckeExt/, Campaign/, Golden/, Bench/, main.cpp,
// *Validation.* and ValidationRegistry.cpp from including a HeckeExt/ header --
// the pipeline never depends on the extension layer.  A `sturm_bound` verb
// written in Console/Bindings/ would need exactly that include, so the owner
// chose the reverse direction: HeckeExt/ registers its own verbs.  That costs
// ONE exact-path exception in each of the two guards' rule 2, both with the
// reason "[OD] slot-3 close-out 1", and nothing wider.
//
// THIS HEADER INCLUDES NO Console/ HEADER.  It forward-declares
// `console::Registry` and nothing else, so the guards' exceptions name the .cpp
// ALONE (slot-4 plan Sec. 2.3 item 4, "[audit] Only the .cpp may carry the
// entry").  Its includers are main.cpp (admitted by heckeext rule 1 and
// console rule 2) and Console/ConsoleValidation.cpp (a *Validation.* TU,
// admitted by both).
// ===========================================================================
#pragma once
#ifndef HECKE_EXT_REGISTER_HECKE_EXTENSION_VERBS_H
#define HECKE_EXT_REGISTER_HECKE_EXTENSION_VERBS_H

namespace console {
class Registry;
}

namespace hecke_ext {

// Registers the extension layer's console types and verbs on `registry`.
// Mirrors console::RegisterPipelineVerbs' two forms (RegisterPipelineVerbs.h):
// the Registry& form is what a validation calls on its OWN registry, the
// no-argument form is what main.cpp's console branch calls on the
// process-wide one.  Idempotent per registry in the no-argument form.
//
// It must run AFTER console::RegisterPipelineVerbs on the same registry: the
// types it registers are looked up by name from the C1a/C1b set.
void RegisterHeckeExtensionVerbs(console::Registry& registry);
void RegisterHeckeExtensionVerbs();

} // namespace hecke_ext

#endif // HECKE_EXT_REGISTER_HECKE_EXTENSION_VERBS_H
