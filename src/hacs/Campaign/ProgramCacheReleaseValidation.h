// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef OC_CAMPAIGN_PROGRAM_CACHE_RELEASE_VALIDATION_H
#define OC_CAMPAIGN_PROGRAM_CACHE_RELEASE_VALIDATION_H

#include <cstdint>

// (hecke-extensions slot 8, WP-5; [KI] Sec. 4c) FAST tier.  The contract of
// campaign::ReleaseEngineProgramCaches (Campaign/ProgramCacheRelease.h):
//   (A) the policy -- it clears BOTH global caches (uint32_t and uint64_t),
//       returns the number of programs it dropped, and leaves them alone
//       while their combined size is <= the threshold;
//   (B) no value moves -- a pipeline point (cohomology + A^{(q)}(T_2)) and a
//       uint64_t tape program recomputed after a clear (a forced recompile,
//       counted as a cache miss) are byte-identical to the first run;
//   (C) clear-while-running -- one thread repeats (B)'s computations while a
//       second thread clears both caches in a tight loop; every repeat is
//       byte-identical to the reference, and at least one clear released
//       programs mid-run.  Under the CheckThread configuration this arm is the
//       ThreadSanitizer statement that a clear concurrent with run_cached()
//       is race-free (the slot-8 WP-6 parallel ext pass relies on it).
// Writes nothing.  SKIPs under ENGINE_BACKEND=legacy (no program is recorded).
uint32_t ProgramCacheReleaseValidation();

#endif // OC_CAMPAIGN_PROGRAM_CACHE_RELEASE_VALIDATION_H
