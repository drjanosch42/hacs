// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// ConsoleReadOnlyTagValidation.h -- [P] v3.10 Sec. 18.11, `fast`
// (hecke-extensions slot 19, NB-2).
//
//   (A) registry <-> Console/serve/read_only_verbs.tsv, both directions, tag
//       for tag (an overload with no row, a row with no overload, or a
//       differing tag fails and names the overload);
//   (B) completeness of the default: registered == explicitly tagged;
//   (C1) the source audit: the registration TUs of the console verbs
//       (Console/Core/CoreVerbs.cpp, Console/Bindings/*.cpp,
//       HeckeExt/RegisterHeckeExtensionVerbs*.cpp) contain no writer / env /
//       process entry point outside the content-keyed allow-list
//       Console/serve/read_only_sources.allow (an entry matching nothing is
//       STALE and fails);  (C2), the dynamic audit, is ConsoleServeValidation's;
//   (D) the manual: the read_only marks of docs/latex/cli_manual/verbs.tex and
//       \CMNumReadOnly of counts.tex equal the registry;
//   (E) PrintJson type coverage: every registered type has a dedicated schema
//       or the generic form, the dedicated set equals [P] Sec. 18.5's table,
//       and a set of in-process values renders to valid envelopes (the 2^53
//       integer rule, Poly low_to_high, Matrix entries, never forcing a
//       declared handle);
//   (U) the units of the served console: the JSON reader / writer and the
//       environment gate's pattern rule.
// Prints `EFFECTIVE ConsoleReadOnlyTagValidation arm=ALL cells=<overloads> ...`.
// With OC_BUILD_NOTEBOOK=OFF: `[SKIP] ConsoleReadOnlyTagValidation:
// OC_BUILD_NOTEBOOK=OFF`, exit 0.
// ===========================================================================
#pragma once

#include <cstdint>

uint32_t ConsoleReadOnlyTagValidation();
