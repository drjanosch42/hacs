// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef CONTRACT_CENSUS_VALIDATION_H
#define CONTRACT_CENSUS_VALIDATION_H

#include <cstdint>

// hecke-extensions slot 3, WP-10 -- the fifth grep-level migration guard, next
// to ConsumerMigrationValidation / MatrixRetirementValidation / ConsoleC0Validation
// (check_console_includes.sh) / HeckeExtArchitectureValidation
// (check_heckeext_includes.sh), all in docs/matrix-engine/tools/.
//
// [C] `sec:machinery` item 2 (rule L1): every `[C, <label>]` citation in the
// tree -- docs, C++ comments, CMakeLists.txt, READMEs, plans -- and every
// `\HA{<label>}` cross-reference in docs/latex/hecke_extensions.tex must name
// a label that docs/latex/gl2_contract.tex actually `\label{}`s. Unlike the
// other four guards this one cannot be a line-level grep (a citation payload
// is comma-separated, sometimes clause-suffixed, sometimes multi-label), so
// the parsing lives in docs/matrix-engine/tools/label_census.py; this
// validation runs the bash entry point docs/matrix-engine/tools/label_census.sh
// over the checkout via std::system, EXACTLY the pattern
// HeckeExtArchitectureValidation / MatrixRetirementValidation use for their
// scripts.
//
// The script regenerates docs/latex/label_census.tsv (one row per citing
// file, plus a TOTAL row) UNCONDITIONALLY on every run -- unlike GoldenEmit
// this is not owner-gated, because the census reports on the tree as it
// stands, not a frozen baseline. Per hecke_extensions_slot3_plan.md WP-10,
// the SITE COUNT is deliberately never pinned (it grows every
// slot); only the UNRESOLVED count -- after docs/matrix-engine/tools/
// label_census.allow -- must be 0, both in the freshly regenerated census
// and (recorded, not gated -- see label_census.py's `committed_unresolved=`
// line) in whatever TSV was already committed.
//
// Since hecke-extensions slot 4 WP-6, label_census.allow is CONTENT-KEYED
// (`relpath|<verbatim fragment of the site's source line>`, not
// `relpath:line`), so inserting or deleting lines above a cited passage
// cannot silently move an allow-list entry off its site, and an entry that
// matches zero citation sites is itself flagged STALE and fails the census.
// This validation exercises both properties directly, against scratch
// roots + scratch allow-lists (never the real
// docs/matrix-engine/tools/label_census.allow or docs/latex/label_census.tsv):
//   1. an allow-listed site survives a line inserted above it (still
//      unresolved=0 after the citation's line number moves);
//   2. a planted, genuinely undefined citation still fails;
//   3. an allow-list entry matching nothing is reported STALE and fails;
//   4. the retired `relpath:line` form is rejected with a named error, not
//      silently accepted or silently ignored.
// These scratch arms are skipped (not failed) if python3 is unavailable,
// matching the SKIP-on-missing-capability rule below.
//
// A checkout without the script is a SKIP, not a failure (the binary may be
// run outside the tree, or python3 may be unavailable -- label_census.sh
// itself reports that as its own exit 2, which this validation also treats
// as SKIP rather than FAIL, matching a missing script).
//
// Returns 0 on success, or the number of failing checks (the main census
// plus up to four scratch arms) otherwise.
uint32_t ContractCensusValidation(void);

#endif // CONTRACT_CENSUS_VALIDATION_H
