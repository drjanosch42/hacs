// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_FORMS_NTT3_VALIDATION_H
#define HECKE_EXT_FORMS_NTT3_VALIDATION_H

#include <cstdint>

// (hecke-extensions slot 24, FNTT3) HeckeFormsNtt3Validation (standard): the forms cross-route's three-prime chunked
// engine (P1, P2, P3 = 45 * 2^23 + 1, Garner CRT) against independent oracles -- the naive product, the two-prime
// engine wherever it applies (byte for byte), exact u128 dot products and closed forms past the two-prime CRT bound,
// the E_4^2 = E_8 / E_4 E_6 = E_10 identities, the HeckeSturmBound construction and the HeckeOnCusp residual check on
// whole spaces -- and the lifting of the FNTT refusal (FormsSpaceSupported, BuildCuspSpace, the sat pre-run refusal),
// with the rollback HECKE_FORMS_NTT3=0 (ScopedFormsNtt3(false)) restoring it.  See the .cpp for the grid.
uint32_t HeckeFormsNtt3Validation();

#endif // HECKE_EXT_FORMS_NTT3_VALIDATION_H
