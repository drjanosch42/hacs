// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_ALGEBRA_VALIDATION_H
#define HECKE_EXT_ALGEBRA_VALIDATION_H

// hecke-extensions phase E1a (slot 2, WP-4) -- the validation of
// HeckeExt/HeckeLocalAlgebra: the local algebra A_S of a certified-local
// block, its invariants, generation certificate (a), and the cross-block
// evaluation test.
//
// It runs on CONSTRUCTED TRUTHS -- local F_ell-algebras whose every invariant
// is known in closed form, conjugated by a seeded invertible U so that no
// result can depend on a basis -- over ell in {5, 13, 691, 134217757}
// ([X] Sec. 5, row `HeckeExtAlgebraValidation`).  The commutant arms and the
// real-block (Sturm-membership, joint-decomposition) arms of that row belong
// to WP-5 and E1b and are NOT here.
//
// Returns 0 on success; nonzero = number of failing checks.  Standard tier.

#include <cstdint>

uint32_t HeckeExtAlgebraValidation(void);

#endif // HECKE_EXT_ALGEBRA_VALIDATION_H
