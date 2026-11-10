// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef CAMPAIGN_HECKEZ_H
#define CAMPAIGN_HECKEZ_H

// ===========================================================================
// (hecke-extensions slot 25, HZ-REC; [C] sec:recordschema item 8 (a), 04-record-census.md Sec. 1.5) THE CAMPAIGN UNDER
// `coefficient_ring = Z` WITH `structure = hecke`: one `integral_hecke_structure/v1` record (Integral/HeckeZRecord.h) per
// (level, n_sym, algebra) CELL.  User documentation: docs/matrix-engine/campaign.md Sec. 18.
//
// THE SPEC.  `coefficient_ring = Z`, `structure = hecke`; per section `n_sym` (required, even), `level` (default 1) and `algebra`
// (`full` / `anemic` / both; default `full` at every N, as the console verb (schema audit F15); `anemic` at N = 1 is a spec error).  Refused at spec
// validation: `ell`, `r`, `q`, `p` (the record's operator set is its P_rule, a cap of the settings text), `parts` (the descent_ref
// is always the groups,hecke descent), `orbifold`, `jobs > 1`, N = 0, an odd n.  `hz_cert = on` also writes the certificate files
// into <output>/hzcert/ (HZ-EMIT; an output knob).
//
// IDENTITY.  cell id = RecordId: `hz_N<N>_n<n>_<alg>_s<16 hex>`, the hex the FNV-1a-64 of the record's settings text (the math
// settings' canonical text + the HeckeZ caps), so a non-default setting or cap changes the id and the hash input (item 8 (j)).
//
// OUTPUT.  <output>/cells/<id>.json: one line {"schema":"integral_hecke_structure_cell/v1","id",..,"campaign",..,"record":<the
// record>,"complete":true}; summary.tsv; index.json.  The record's descent_ref is joined to integral census ed.1 (rows.tsv, through the
// source path baked into the binary).  Last line: `EFFECTIVE campaign arm=hecke_structure cells=<k> ran=<r> resumed=<s> refused=<x>
// pending=<p> final=<f> failed=<e> outside=<o> under=<u> ed1_mismatch=<m>` (and `rejected=<r>`).  Every written record is
// scanned (ScanRecord): a value outside the closed lists or an exact value under a bound fails the cell.  `resume = true` re-reads a
// cell file that ends with the completion trailer and accepts it only if VerifyCellText holds (id, fingerprint, both hashes and the
// status recompute, the scan is clean); a rejected file is counted and the cell recomputed (schema audit F4).  An ed.1 mismatch of
// the descent_ref is counted and makes the exit code 2.
// ===========================================================================

#include "CampaignSpec.h"

#include <cstdint>
#include <string>
#include <vector>

namespace campaign {

struct HeckeZCellPlan {
    uint32_t N = 1, n = 0;
    std::string algebra;
    std::string id;
};
std::vector<HeckeZCellPlan> PlanHeckeZCells(const CampaignSpec& spec);
int RunHeckeZCampaignCli(const CampaignSpec& spec, bool dry_run, bool verbose);

}  // namespace campaign

#endif
