// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef CONSOLE_BINDINGS_DESCENT_H
#define CONSOLE_BINDINGS_DESCENT_H

// DESCENT milestone D6 (slot 24): the console surface of the integral descent record (Integral/Descent/DescentRecord.h).
// A Console/Bindings/ TU (so check_console_includes.sh and ConsoleReadOnlyTagValidation (C1) cover it); it includes no HeckeExt/ header.
//   descent(N, n, primes = nil) -> DescentRecord     a lazy handle over integral_descent::record::BuildDescentRecord, under
//                                                     coefficient_ring = Z (the verb refuses at dispatch under any other ring)
// Registered at the tail of hecke_ext::RegisterHeckeExtensionVerbs(registry) (so every registry fixture that registers the
// extension verbs registers these), after its types.  read_only: the twin rows in Console/Bindings/ReadOnlyTags.cpp and
// Console/serve/read_only_verbs.tsv.

namespace console {
class Registry;
}

namespace integral_descent {

void RegisterDescentVerbs(console::Registry& registry);

}  // namespace integral_descent

#endif
