// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef SUBSTRATE_DIFFERENTIAL_VALIDATION_H
#define SUBSTRATE_DIFFERENTIAL_VALIDATION_H

#include <cstdint>

// ===========================================================================
// Phase 0 -- substrate differential harness
// (docs/matrix-engine/01-matrix-substrate-audit.md Sec. 7, gap #9;
//  docs/matrix-engine/00-implementation-plan.md, Phase 0 scope).
//
// The three matrix substrates (MatrixT / FlatMatrixT / BlockFlatMatrixT) are
// three independent implementations of the SAME linear algebra over F_p, and
// nothing in the tree currently compares them against each other. This
// validation does exactly that, on fixed-seed pseudorandom inputs, so that
// Phase 3's retirement of BlockFlatMatrix and Phase 2's arithmetic rewrite
// have a differential oracle to run against.
//
// Returns the number of FAILING cases (0 = all agree); each case prints its
// own PASS/FAIL line, so a partial failure localises immediately.
// ===========================================================================
uint32_t SubstrateDifferentialValidation();

#endif // SUBSTRATE_DIFFERENTIAL_VALIDATION_H
