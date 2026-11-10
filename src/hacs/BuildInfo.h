// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (slot-24 BUILD-SLIM, F7) The commit baked in at CONFIGURE time lives in ONE generated translation unit
// (<build>/generated/OcBuildInfo.cpp, from cmake/OcBuildInfo.cpp.in), not in a compile definition: a definition is part of
// the target's flags.make, which every object depends on, so a reconfigure at a new HEAD recompiled the whole application.
// Now a new HEAD recompiles that one file and relinks.  Metadata only: no golden object, digest or job id reads it.
#pragma once

namespace oc {
// `git rev-parse HEAD` at configure time, or "unknown" (no git, or not a checkout).
const char* GitCommit();
// (R1 / X1) The HACS release version, from the tracked VERSION file at configure time (e.g. "0.1.0").  Metadata only.
const char* HacsVersion();
// The one-line text printed by `hacs --version`: "hacs <version> (commit <sha12>; contract [C] x, [E] y, [U] z)".
const char* HacsVersionLine();
}  // namespace oc
