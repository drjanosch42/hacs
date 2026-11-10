// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// ConsoleServeAtlasValidation.h -- the conformance test of the atlas sidecar
// (`cohomology_app --atlas --index`) and of `cohomology_app --serve-atlas`,
// protocol `oc-serve-atlas/1` ([P] v3.11 Sec. 18.16 (h); hecke-extensions
// slot 20, NB-4).  Tier `standard`.
//
// Builds Sigma(23) (cuspidal and Eisenstein systems, r in {1, 2}, twists) and
// Sigma(13) in process (BuildAtlas), writes their store texts into a scratch
// directory under ${XDG_CACHE_HOME:-$HOME/.cache}/hacs/scratch (removed
// afterwards) and drives the real binary as a child over pipes
// (ENGINE_CUDA_DISABLE=1, an empty working directory; no Python, no network):
//   (A) the sidecar writer: exit 0, the ATLAS_INDEX line, the file equal to
//       atlas_index::Serialize of the in-process scan, the store's bytes and
//       mtime unchanged; the refusals (output = the store, inside the source
//       tree, a missing or not indexable store);
//   (B) atlas_hello with the sidecar, an explicit --index, and the memory index;
//   (C) every query against the oracle ParseAtlasStore(text): info, the cells,
//       every cell with its occurrences, every system, `systems` under every
//       filter key and combinations, paged; the sidecar and the memory runs give
//       byte-identical reply streams (volatile `perf` stripped);
//   (D) errors, one per row of (e), the process surviving each; close / EOF /
//       an over-long line (atlas_fatal protocol_overflow, exit 70);
//   (E) the frame cap (--max-reply-bytes 4096): pages cut, `next` continues,
//       no frame over the cap;
//   (F) stale and corrupt indexes and a wrong --expect-fnv refused;
//   (G) the environment gate and the hardening (limits, O_RDONLY descriptor,
//       a mapping without `w`, NoNewPrivs) read back from /proc;
//   (H) the stores' bytes and mtimes, the empty working directory and
//       `git status --porcelain` (probed) unchanged;
//   (R) optional, read-only: HECKE_ATLAS_IN=<dir> -> the real Sigma(691) store.
// Prints `EFFECTIVE ConsoleServeAtlasValidation arm=ALL cells=<k> systems=<k>
// queries=<k> real=<absent|sidecar|memory>`.  With OC_BUILD_NOTEBOOK=OFF:
// `[SKIP] ConsoleServeAtlasValidation: OC_BUILD_NOTEBOOK=OFF`, exit 0.
// ===========================================================================
#pragma once

#include <cstdint>

uint32_t ConsoleServeAtlasValidation();
