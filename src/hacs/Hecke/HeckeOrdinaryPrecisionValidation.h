// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 12, WP-4) HeckeOrdinaryPrecisionValidation -- the
// torsion pre-pass and the ordinary projector U^{N_ord} of Hecke/HeckeOrdinaryPart
// against slot-11 WP-18c's run-derived objects: [C] v1.12.0-pre.2
// rmk:hidacheckable (Hi1), (Hi2); prop:torsionprepass, cor:ordprecision,
// lem:ordexponent.  Standard tier; writes nothing.
#pragma once
#ifndef HECKE_ORDINARY_PRECISION_VALIDATION_H
#define HECKE_ORDINARY_PRECISION_VALIDATION_H

#include <cstdint>

uint32_t HeckeOrdinaryPrecisionValidation();

#endif
