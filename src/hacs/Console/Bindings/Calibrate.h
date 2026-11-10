// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Calibrate.h -- NBCONFIRM S3: `cohomology_app --calibrate[=thorough]`, the
// per-device speed calibration of the owner addendum of 2026-11-06 (S2 report
// Sec. 4, Fix round 1 F4/F8/F9; prototype docs/console/tools/nbconfirm_calibrate.py).
//
//   cohomology_app --calibrate [--repeat R] [--threads T] [--taskset CPUS]
//                              [--profile PATH | --no-write] [--emit-reference]
//   cohomology_app --calibrate=thorough ...        (= --repeat 3)
//
// Runs the fixed probe set of the embedded `calib` rows (calib/v2: the six verb
// anchors zdesc zrec u128 heil ntt ladic, inside today's silent envelopes, and
// the four fingerprints elim gemm bw hecke) as CHILD PROCESSES of this binary
// (one fork + exec of /proc/self/exe per probe, so that each probe's peak RSS
// is its own: getrusage of the child), at 1 thread and at T = min(16, the
// affinity CPU count) threads, the minimum of R repeats per probe.  One
// progress line per probe on stderr.  Writes the device profile
// ${HACS_DEVICE_PROFILE:-${XDG_CONFIG_HOME:-~/.config}/hacs/device_profile.tsv}
// atomically (.tmp + rename): factor = t_device(T) / t_reference(T) at EQUAL
// thread count, the ELF build id of the binary, the affinity CPU count.
//
// Prints `NBCONFIRM_CALIB set=... verdict=OK|digest_mismatch|probe_failed` and
// `COST_MODEL_DRIFT set=... worst_wall=<class>:<f> worst_rss=<class>:<r>
// verdict=OK|DRIFT|NO_REFERENCE_RSS` (the drift gate of S2 FR.6: a verb
// anchor's wall factor or RSS ratio outside [0.75, 1.33]; meaningful on the
// reference box only, so it is a close-window step and never part of --all).
//
// Never changes a value, a record or a hash: the probes' outputs are
// discarded except two digests / checksums, checked against the reference rows
// as a smoke test (a mismatch writes no profile).  On demand only.
// ===========================================================================
#pragma once
#ifndef CONSOLE_BINDINGS_CALIBRATE_H
#define CONSOLE_BINDINGS_CALIBRATE_H

namespace console {
namespace calibrate {

// argv[0] is the token itself ("--calibrate" or "--calibrate=thorough").  Exit 0 ok, 1 usage / failure, 2 digest mismatch.
int RunCalibrate(int argc, char** argv);

} // namespace calibrate
} // namespace console

#endif // CONSOLE_BINDINGS_CALIBRATE_H
