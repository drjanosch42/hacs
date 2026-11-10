// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef CAMPAIGN_RING_H
#define CAMPAIGN_RING_H

// ===========================================================================
// (hecke-extensions slot 9, WP-8; C5c; decision 6 (A)) THE CAMPAIGN'S JOB LOOP
// UNDER A NON-DEFAULT `coefficient_ring` -- the job-level schema
// `hecke_ring_report/v1`.  Binding text: [E] v1.10.0-pre.2 sec:ring (g)
// (docs/latex/hecke_extensions.tex), members (g1)-(g16), the summary.tsv /
// index.json changes; (b1)'s campaign row; (b4)'s campaign surface.  User
// documentation: docs/matrix-engine/campaign.md Sec. 15.
//
// WHEN.  Only under `coefficient_ring = Z/ell^<k>` (k >= 2): RunOneJob()
// dispatches here before anything of the field path runs, and writes this
// schema INSTEAD of `campaign_job/v1`.  A fingerprint-0 campaign never reaches
// this file, so its job files, summary.tsv and index.json are byte for byte
// what they were (the acceptance (b) of the slot-9 plan's WP-8).
//
// WHAT A RING JOB COMPUTES, per job (n_sym, ell, p, r, degrees) at K = k:
//  * always: EquivariantCohomologyOnK over Z/ell^k (the module route at
//    ell >= 5, the Bredon flat-coordinate route at ell in {2, 3}) -- lengths
//    and invariant factors per degree;
//  * at ell >= 5: the K-mode (r_K, ladic_type_certified, ladic_type^q, the
//    ladic_level state) on the applicability range n > 0 even; the mod-ell
//    Bockstein counts from the cell's Z/ell^2 object and the F_ell object,
//    each built with its own ring installed for its construction only (the
//    golden harness's RunE8Point precedent); red_H^q against the F_ell
//    object; and, for r = 1 with q = 1 reported, A(T_p) over the ring with its
//    free characteristic polynomial at a certified level (U_ell at p = ell);
//  * refusals, recorded and never silently skipped ((g14)): at ell in {2, 3}
//    the Hecke part (HeckeOperatorOnK), the Bockstein and the reduction
//    (HeckeReduction), each obtained from the REAL guard it hits -- so a later
//    lift that removes a guard makes the job fail loudly here rather than
//    silently change its record; at r >= 2 the Hecke part at this job loop's
//    own dispatch (T_{p^r} over the ring is not validated in this build).
//
// Spec validation (CampaignSpec.cpp) has already refused `ext = true` under
// the ring and every grid cell outside the per-cell rule, before any job runs;
// this file never refuses a whole job.  (slot-11 WP-18c; [E] v1.12.0-pre.2
// sec:ring (g)) A cell with ell^k >= 2^64 at ell >= 5 is a DIGIT job
// (Hecke/LadicCell): lengths and invariant factors of H^q(Z/ell^k) read from
// the certified type, the K-mode at the working level K = k + e_max, the free
// charpoly at precision k, the ordinary idempotent at p = ell; `route`,
// `precision` and a textual `modulus` in the record; the Bockstein and the
// reduction recorded as refusals; a PrecisionInsufficient is recorded as the
// refusal `precision` with its resumable state (status ok_partial).
// ===========================================================================

#include "CampaignDriver.h"

#include <iosfwd>
#include <string>

namespace campaign {

// One job under a non-default ring: writes jobs/<id>.json as
// hecke_ring_report/v1 (the whole file is assembled in memory -- it is small
// -- and written once, so a job killed mid-way leaves no trailer and `resume`
// re-runs it).  Same contract as the field RunOneJob: never throws for a
// per-job failure (status "failed", trailer false), throws only for an I/O
// error on the job file.
JobResult RunRingJob(const CampaignSpec& spec, const Job& job, const std::string& path, bool verbose,
                     bool allow_rss_reset, std::ostream* log);

// summary.tsv of a ring campaign ([E] sec:ring (g)): the 15 appended column
// names, each preceded by a tab, and one job's 15 values (same form).
std::string RingSummaryHeader();
std::string RingSummaryColumns(const CampaignSpec& spec, const JobResult& r);

} // namespace campaign

#endif // CAMPAIGN_RING_H
