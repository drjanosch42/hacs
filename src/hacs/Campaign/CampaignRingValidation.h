// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef CAMPAIGN_RING_VALIDATION_H
#define CAMPAIGN_RING_VALIDATION_H

#include <cstdint>

// (hecke-extensions slot 9, WP-8; C5c; decision 6 (A)) STANDARD tier.  The
// campaign's job loop under `coefficient_ring = Z/ell^<k>` and its job-level
// schema hecke_ring_report/v1 ([E] sec:ring (g); Campaign/CampaignRing.h;
// campaign.md Sec. 15).  Arms:
//  (A) the E8 points of [E] (e) at ell = 5 over Z/25 and Z/125, and (10, 691)
//      over Z/691^2, run as two ring campaigns (jobs = 2, per-job pools): every
//      (d) object a record carries equals the frozen golden entry at the E8
//      point, tier / rows / cols / hash, and every frozen 4a entry of the point
//      (charpoly_* for the job's own p) is carried -- a set equality; the (e)
//      table's pre-stated values (certified, invariant factors, Bockstein
//      counts) literally;
//  (T) weight 12 over Z/ell^2 at (10, 691) and (10, 134217757) (the uint64_t
//      job): charpoly_free = (x - (1 + 2^11))(x - tau(2))^2 mod ell^2, and the
//      modulus string;
//  (B) ell in {2, 3}: lengths and invariant factors against the frozen O7
//      rows of HeckeBredonTreeRingValidation, kmode `not_applicable`, the
//      three refusals (hecke by HeckeOperatorOnK, bockstein / reduction by
//      HeckeReduction) with the `not_implemented_in_this_build:` tag, status
//      `ok_partial`, no ladic / bockstein / hecke member;
//  (R) r = 2 at ell = 5: the Hecke part refused at the job loop's dispatch,
//      everything else computed;
//  (S) the record's member order (g1)-(g16), no `dim_H` / `reports` member,
//      summary.tsv's 21 + 15 columns with the ten field columns `-`,
//      index.json's `coefficient_ring`, `resume` skipping every ring job, and
//      jobs = 1 == jobs = 2 (objects and summary columns);
//  (X) spec validation: `ext = true` under the ring and a cell with
//      ell^k >= 2^64 refused by name before any job file exists (through the
//      CLI too: exit 1, no output directory); ell^k < 2^64 admitted.
// Writes only under the system temp directory, removed at the end.
uint32_t CampaignRingValidation();

#endif // CAMPAIGN_RING_VALIDATION_H
