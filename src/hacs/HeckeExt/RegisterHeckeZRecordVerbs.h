// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_REGISTER_HECKEZ_RECORD_VERBS_H
#define HECKE_EXT_REGISTER_HECKEZ_RECORD_VERBS_H

// The console verb `hecke_structure(N, n, algebra = "full")` (slot-25 HZ-REC; [P] Sec. "hecke_structure"): the record
// integral_hecke_structure/v1 of one cell as a console Record.  Registered from HeckeExt/ (check_heckeext_includes.sh rule 1 forbids
// Console/Bindings/ from reaching the HeckeExt/ source of the record builder; ALLOW_EXT_CONSOLE entry with that reason).  This header
// includes no Console/ header (the .cpp alone carries the exception).
namespace console {
class Registry;
}

namespace heckez {
namespace record {
void RegisterHeckeZRecordVerbs(console::Registry& registry);
}  // namespace record
}  // namespace heckez

#endif
