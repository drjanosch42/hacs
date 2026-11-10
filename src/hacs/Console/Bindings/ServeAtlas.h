// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// ServeAtlas.h -- `cohomology_app --serve-atlas`: read-only queries on a
// `hecke_atlas/v1` store, protocol `oc-serve-atlas/1` ([P] v3.11 Sec. 18.16;
// hecke-extensions slot 20, NB-4).
//
// One process answers NDJSON requests on stdin (`info`, `cells`, `cell`,
// `system`, `systems`, `close`) with exactly one `atlas_reply` frame each on
// stdout, after one `atlas_hello`.  The store's path comes from the command
// line (the gateway's launch configuration), never from a request.
//
// READ-ONLY.  The environment gate is serve::EnvNameRefused (the same rule as
// `--serve`, refuse, never scrub); the process hardening is that of
// `--serve` (RLIMIT_FSIZE = 0, RLIMIT_CORE = 0, PR_SET_NO_NEW_PRIVS, stdout not
// a regular file, fd 1 re-pointed to stderr); the store is opened O_RDONLY and
// mapped PROT_READ / MAP_PRIVATE (Util/AtlasIndex.h).  The sidecar index, if
// any, is compared field by field with the index the scanner derives from the
// store at every open; a difference refuses the launch.
//
// Compiled only with OC_BUILD_NOTEBOOK=ON (cmake/slot20/NB4.cmake).  Registers
// no verb and no validation, probes no device.
// ===========================================================================
#pragma once
#ifndef CONSOLE_BINDINGS_SERVE_ATLAS_H
#define CONSOLE_BINDINGS_SERVE_ATLAS_H

#include <cstddef>

namespace console {
namespace serve_atlas {

constexpr const char* kProto = "oc-serve-atlas/1";
constexpr std::size_t kDefaultMaxReplyBytes = 1048576;
constexpr std::size_t kMinMaxReplyBytes = 4096;
constexpr std::size_t kMaxRequestBytes = 65536;
constexpr std::size_t kPageMax = 1000;
constexpr std::size_t kPageDefault = 50;
constexpr int kWatchdogPollMs = 250;
constexpr int kFatalExit = 70;

// `argv[0..argc)` are the tokens AFTER `--serve-atlas`.  Returns the exit code
// of a launch refusal (1); a served process `_exit`s (0 or 70).
int RunServeAtlas(int argc, char** argv);

// `argv[0..argc)` are the tokens AFTER `--atlas --query`: `<store.json>
// '<request JSON>'` (slot-22 VERB-3).  One `atlas_reply` frame on stdout; exit 0
// on `ok`, 1 usage / refusal, 3 on an `error` / `rejected` reply.
int RunAtlasQuery(int argc, char** argv);

} // namespace serve_atlas
} // namespace console

#endif // CONSOLE_BINDINGS_SERVE_ATLAS_H
