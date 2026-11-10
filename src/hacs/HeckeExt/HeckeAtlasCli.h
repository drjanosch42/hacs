// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_HECKE_ATLAS_CLI_H
#define HECKE_EXT_HECKE_ATLAS_CLI_H

// ===========================================================================
// (hecke-extensions slot 7, WP-5 / phase E6 cont.) `--atlas`: the one entry
// point that builds a residue atlas Sigma(ell) outside the validation suite.
// The ell = 691 build (346 driver runs, of the order of an hour) is NOT a
// registered validation; docs/matrix-engine/tools/atlas_build.sh drives this
// entry point under a `git status` guard.
//
//   cohomology_app --atlas <ell> [--n-max N] [--collect-only] [--quiet]
//   cohomology_app --atlas --close <store.json>
//   cohomology_app --atlas --verify <store.json>
//   cohomology_app --atlas --index <store.json> [--out <path>]
//
//   <ell>            build Sigma(ell) (every even n <= min(ell - 1, N));
//                    one ATLAS_CELL line per cell unless --quiet;
//   --collect-only   stop after the collection (no closure, no S_sep), and
//                    write the store as hecke_atlas_ell<ell>.collected.json;
//   --close <f>      re-derive the closure and S_sep of a store (collected or
//                    closed) without the pipeline;
//   --verify <f>     re-close a CLOSED store and compare byte for byte
//                    (exit 0 iff identical) -- the integrity of a store whose
//                    build is too long to repeat.
//   --index <f>      (slot-20 NB-4, [P] v3.11 Sec. 18.16 (b)) write the
//                    oc-atlas-index/1 sidecar of a store (default <f>.ocidx,
//                    or --out <path>; never the store itself, never inside the
//                    source tree or docs/matrix-engine/results/); the store is
//                    opened read-only.  Exit 0 / 1 refused / 3 not indexable.
//
// Output: ATLAS_SUMMARY (sizes, the residue-degree profile, S_sep, the wall
// and peak RSS) and ATLAS_STORE (bytes and the FNV-1a-64 of the store text).
// The store is written ONLY under HECKE_ATLAS_OUT=<dir>, by
// WriteAtlasStoreIfRequested, which refuses docs/matrix-engine/results/.
// Exit codes: 0 ok, 1 usage, 2 verify mismatch, 3 exception.
// ===========================================================================

#include <functional>

namespace hecke_ext {

// `between_cells` (slot-8 WP-5, [KI] Sec. 4c; slot-8 plan Sec. 2.3 item 9) is
// called once after every cell of a build (AtlasOptions::between_cells).  The
// layer may not name the engine (check_heckeext_includes.sh rule 2), so the
// caller injects the release of the engine's process-wide program caches
// here: main.cpp's `--atlas` branch passes campaign::ReleaseEngineProgramCaches.
// STL only; the empty default calls nothing, which keeps every existing
// caller (and every store byte -- the hook never enters the store).
int RunAtlasCli(int argc, char** argv, std::function<void()> between_cells = {});

} // namespace hecke_ext

#endif // HECKE_EXT_HECKE_ATLAS_CLI_H
