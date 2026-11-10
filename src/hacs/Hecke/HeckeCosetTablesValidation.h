// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_COSET_TABLES_VALIDATION_H
#define HECKE_COSET_TABLES_VALIDATION_H

#include <cstdint>

// Stage S3V -- INDEPENDENT validation of Stage S3 (HeckeCosetTables) against
// the worked contract example, oracle O16
// (docs/implementation/hecke_action_implementation_plan.md, "### S3V --
// Coset-table validation vs. the worked contract example").
//
// Independence statement: every expected numeric value asserted here (the
// p = 2 representatives s_i, the deltas delta_i = s_i^{-1} g, the P^1(F_2)
// classes, the 4x3 permutation table, and all twelve cocycles h_i(gamma))
// is TRANSCRIBED VERBATIM from docs/latex/hecke_action.tex,
// \section{Worked micro-example ($p=2$) and validation plan}
// (\label{sec:example}), NOT read from HeckeCosetTables.cpp or
// HeckeCosetTablesSelfCheckValidation.cpp. The p=3,5 and p=2,r=2
// self-checks are re-derived here from the contract's general
// representative recipe ([C, prop:cosets] contract box) and from
// elementary group theory (the cocycle identity h_i(gamma) =
// s_i*gamma*s_{i.gamma}^{-1}), independently re-implemented against the
// public hecke::Mat22 exact-arithmetic primitives -- never by reading
// HeckeCosetTables' internal tables as ground truth.
//
// Returns the number of discrepancies found (0 == success), matching the
// ValidationFn convention (invariant I9).
uint32_t HeckeCosetTablesValidation();

#endif // HECKE_COSET_TABLES_VALIDATION_H
