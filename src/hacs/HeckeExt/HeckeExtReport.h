// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_REPORT_H
#define HECKE_EXT_REPORT_H

// ===========================================================================
// hecke-extensions phases E2a / E2 cont. (slot 4 WP-1; slot 5 WP-1) -- the
// `hecke_ext/v2` WRITER.
//
// [X] Sec. 1.4: "`BuildHeckeExtReportJson(const HeckeExtRecord&)` and a
// streamed `StreamHeckeExtReportJson(std::ostream&, ...)`, hand-rolled exactly
// like HeckeReport.cpp (no escaper needed: the writer emits only literals,
// numbers and enumerated strings), schema `"hecke_ext/v2"`, keys and nesting
// verbatim from [E] Sec. 7", with `hecke_report/v1`'s conventions kept:
//
//     poly_coeff_order   low -> high degree (A-N4)
//     matrices           arrays of ROWS (row-major), entries in [0, ell)
//     residue_degree     vs residue_degree_lower_bound, never conflated
//
// Both conventions are stated IN BAND, as `hecke_report` states them.
//
// A SEPARATE WRITER, DELIBERATELY.  Hecke/HeckeReport.{h,cpp} is NOT touched:
// Golden/GoldenHarness.cpp hashes its text, so a new key there is a golden
// re-baseline and this package makes none ([X] Sec. 1.6; slot-5 plan Sec. 4.4,
// whose per-package row for WP-1 ends `git diff <base> --
// Hecke/HeckeReport.{h,cpp}` empty).
//
// THE TWO CONE FLAGS ARE EMITTED SINCE v2 (slot-5 WP-1 scope 4).  Slot 4
// reserved them by name and did not emit them, because `pair` was
// `{"status": "unavailable"}`.  Inside an AVAILABLE `pair` object this writer
// now emits, from [WP3-6] finding 1:
//
//     "phi_flat_identity"   -- Phi_g's flat identity, which holds on the
//                              stored C^q_eq basis always and FLAT only in
//                              29 of 424 degrees;
//     "residual_flat_zero"  -- the lem:conesign residual's flat form.
//
// Both are diagnostics under [C] rule A6: recorded, never branched on, and a
// `false` is conformance ([C] `sec:recordschema` item 6(e)).
//
// THE SIDECAR ([X] Sec. 1.3.1 R3.1 item 6).  "The record stem is set by the
// record layer (E2), the only place that knows it."  That is
// `WriteHeckeExtReport(record, path)`: it derives the stem from `path` (the
// trailing ".json" removed), and every resumable state whose JSON exceeds
// 64 KiB is written to `<stem>.state.<kind>.json` and referenced from the
// record by RELATIVE PATH (the sidecar's basename, so the pair of files moves
// together) and FNV-1a-64 (src/ModArithmetic/Fnv.h, the one
// byte-wise FNV from its standard offset basis).  With NO stem -- which is
// what `BuildHeckeExtReportJson` has -- an oversize state is INLINED anyway
// and flagged `"state_oversize": true` with its hash, rather than emitting a
// path that names no file (the slot-1 WP-3 rule, HeckeSturmBound.cpp:655).
//
// DISAMBIGUATION, a deviation from [X]'s literal `<record-stem>.state.<kind>
// .json`: a record may carry several oversize states of the SAME kind (one
// commutant per ideal).  The FIRST of a kind keeps [X]'s name; the i-th
// (i >= 1) is `<stem>.state.<kind>.<i>.json`.  Reported as a WP-1 finding for
// [C] `sec:recordschema`.
// ===========================================================================

#include "HeckeExtRecord.h"

#include <iosfwd>
#include <string>

namespace hecke_ext {

struct ExtReportOptions {
    // The record stem.  Empty (the default) means "no stem": an oversize
    // state is inlined and flagged, never referenced by a path.
    std::string sidecar_stem;
    // The directory the relative sidecar path is taken against; set by
    // WriteHeckeExtReport from `path`.  Empty means the stem IS the prefix.
    std::string sidecar_dir;
};

// The whole record as one string (no stem: an oversize state is inlined).
std::string BuildHeckeExtReportJson(const HeckeExtRecord& record);
std::string BuildHeckeExtReportJson(const HeckeExtRecord& record,
                                    const ExtReportOptions& opts);

// The streamed form; it writes the sidecars as it goes when a stem is set.
void StreamHeckeExtReportJson(std::ostream& os, const HeckeExtRecord& record,
                              const ExtReportOptions& opts = ExtReportOptions());

// Writes the record to `path`, sets the stem from it, and writes any sidecar
// beside it.  Throws std::runtime_error if a file cannot be written.  The
// default path of [X] Sec. 1.4 is `hecke_ext_ell<ell>_n<n>.json`.
void WriteHeckeExtReport(const HeckeExtRecord& record, const std::string& path);


} // namespace hecke_ext

#endif // HECKE_EXT_REPORT_H
