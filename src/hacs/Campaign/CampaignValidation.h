// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef CAMPAIGN_VALIDATION_H
#define CAMPAIGN_VALIDATION_H

#include <cstdint>

// matrix-engine Phase 6 -- the two campaign validations of the plan's Phase 6
// "Validations" bullet. Both return 0 on success (repo convention I9).

// STANDARD tier. A 20-job spec (n_sym <= 16, ell in {2, 5, 134217757},
// p in {2, 3}, r in {1, 2}) driven through the campaign driver must produce,
// hash for hash, what 20 INDIVIDUAL single-job campaigns produce -- so nothing
// carried between jobs in one process can perturb a result. It then asserts
// that every campaign object whose (point, object) key also occurs in the
// frozen Golden/GoldenReference.inc carries the SAME FNV-1a-64 hash (the
// campaign reuses the golden harness's hashing and vocabulary verbatim), and
// finally that memory-lean mode is value-identical to the resident mode.
uint32_t CampaignSmokeValidation();

// FAST tier. The certify contract: the per-job JSON records the certify mode;
// under `sampled` the IN-PROCESS certificates C3-C6 still ran on EVERY job
// (counted, not assumed) while the external cross-checks ran only on the
// sampled subset; and `resume` skips completed jobs while re-running a job
// whose output lacks the completion trailer.
uint32_t CampaignCertificateValidation();

// (hecke-extensions slot 7, WP-8 / console C2b) The spec text
// CampaignSmokeValidation runs, exposed read-only so that the console's
// acceptance (d) -- `run_campaign` on the CampaignSmokeValidation spec produces
// the same summary.tsv outside its timing columns ([P] Sec. 10 C2) -- runs
// exactly THIS spec rather than a copy that could drift from it.
const char* CampaignSmokeSpecText();

#endif // CAMPAIGN_VALIDATION_H
