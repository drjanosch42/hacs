// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef CAMPAIGN_DESCENT_H
#define CAMPAIGN_DESCENT_H

// ===========================================================================
// (hecke-extensions slot 24, DESCENT D6; DESCENT_plan.md v0.4 Sec. 4 D6, [X] OD-10 (a)) THE CAMPAIGN UNDER
// `coefficient_ring = Z`: one `integral_descent_record/v1` record (Integral/Descent/DescentRecord.h) per
// (level, n_sym) CELL -- the integral descent of H^q(Gamma_0(N); Sym^n Z^2) with its Hecke invariants and its boundary
// pair.  User documentation: docs/matrix-engine/campaign.md Sec. 17.
//
// THE SPEC.  `coefficient_ring = Z` (a mathematical setting: the fingerprint is non-zero, so every cell id carries
// `_s<16 hex>`); per section the grid keys `n_sym` (required) and `level` (default 1) and, optionally, `p` -- the
// cell's Hecke primes P (a p dividing a level of its section is refused at spec validation -- D6 audit N2; `p` absent: the record's
// default, the first three of 2, 3, 5, 7, 11, 13 not dividing N).  Refused at spec validation, by name, before any
// cell runs: `ell`, `r`, `q` (the record has no coefficient prime, no T_{p^r}, and its own derived degrees),
// `orbifold`, `ext` (refused earlier, as for every non-default ring), `jobs > 1`, N = 0, a non-prime p.
//
// IDENTITY.  cell id `desc_N<N>_n<n>[_p<p1>-<p2>-...]_s<16 hex>[_partsGH|_partsG]`: the `_p` tag only when P is not the default of N, the
// `_parts` tail only for `parts = groups,hecke` / `groups` (ICENSUS-P2; the default `all` adds nothing),
// `_s` the FNV-1a-64 fingerprint of the non-default mathematical settings (hecke::MathSettings, never 0 under Z).  The
// record's own `record_fnv1a64` hashes the settings' canonical text as well, so a non-default setting changes both
// the id and the hash input.  Timings and RSS are outside the hash.
//
// OUTPUT.  <output>/cells/<id>.json: one line {"schema":"integral_descent_cell/v1","id",..,"campaign",..,
// "record":<the record>,"complete":true}; summary.tsv (one row per cell); index.json.  The last line printed is
// `EFFECTIVE campaign arm=descent cells=<k> ran=<r> resumed=<s> refused=<x> failed=<f>`.  `resume = true` skips a
// cell whose file ends with the completion trailer.  A cell whose record is refused (a cap, a precision limit) is
// written with status "refused" and counted; a cell that throws is `failed` (exit 2).
// ===========================================================================

#include "CampaignSpec.h"

#include <cstdint>
#include <string>
#include <vector>

namespace campaign {

struct DescentCellPlan {
    uint32_t N = 1, n = 0;
    std::vector<uint32_t> ps;     // the cell's Hecke primes (none dividing N)
    std::string id;
};
// The fingerprint every cell id carries (hecke::MathSettings::fingerprint() of the spec's settings; non-zero under Z).
uint64_t DescentFingerprint(const CampaignSpec& spec);
std::string DescentCellId(const CampaignSpec& spec, uint32_t N, uint32_t n, const std::vector<uint32_t>& ps);
// Union of the sections' level x n_sym products, de-duplicated on (N, n, P) (first occurrence wins), ordered by (N, n, P).
std::vector<DescentCellPlan> PlanDescentCells(const CampaignSpec& spec);

struct DescentCampaignResult {
    uint32_t cells = 0, ran = 0, resumed = 0, refused = 0, failed = 0;
    std::vector<std::string> lines;   // the summary rows written
};
DescentCampaignResult RunDescentCampaign(const CampaignSpec& spec, bool verbose);
int RunDescentCampaignCli(const CampaignSpec& spec, bool dry_run, bool verbose);

}  // namespace campaign

#endif
