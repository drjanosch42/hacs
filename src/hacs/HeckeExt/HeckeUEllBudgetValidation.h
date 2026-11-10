// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_U_ELL_BUDGET_VALIDATION_H
#define HECKE_EXT_U_ELL_BUDGET_VALIDATION_H

#include <cstdint>

// hecke-extensions slot 3, WP-1 (phase E1b-0).  Returns 0 on success
// (invariant I9); registered in ValidationRegistry::RegisterAllValidations()
// as a `fast` entry -- it builds nothing, it evaluates a closed form.
//
// Arms:
//   (a) the frozen model against the 16 MEASURED grid points of
//       docs/implementation/slot3/WP-1_report.md Sec. 4, pinned in the TU:
//       every residual inside the declared band, and the declared band tight
//       (a re-fit that leaves it must bump kUEllRssModelVersion);
//   (b) monotonicity of PredictUEllRssMb in n and in ell, and the purity of
//       UEllDecision (same inputs -> same verdict, no dependence on the
//       settings in force);
//   (c) the ADMISSION TABLE at M = 2048, re-deriving [X] Sec. 1.5's prose
//       ("every ell <= 13 at n <= 1000, ell = 691 ..., ell = 10007 at
//       n <= 38") and REPORTING every change -- the n-ranges moved, the ten
//       Kummer verdicts of [E] Claim 1.8 did not;
//   (d) the pin UEllDecision(16, 43867, 2048) =
//       {included = false, reason = u_ell_over_budget, predicted_mb, 2048};
//   (e) `off` short-circuits without a prediction, and budget_mb = 0 admits
//       nothing;
//   (f) the `u_ell` MATHEMATICAL SETTING (Hecke/HeckeSettings.h): parsing of
//       every legal spelling and rejection of every illegal one, the
//       round trip through to_text(), the registry order, canonical_text(),
//       the fingerprint (0 at the defaults, non-zero and distinct otherwise),
//       to_json(), set() through the one validation point, and the
//       thread-local scoped override's propagation onto
//       engine::tape::run_concurrently() tasks.
uint32_t HeckeUEllBudgetValidation();

#endif // HECKE_EXT_U_ELL_BUDGET_VALIDATION_H
