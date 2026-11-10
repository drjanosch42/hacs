// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_FORMS_FAST_MUL_VALIDATION_H
#define HECKE_EXT_FORMS_FAST_MUL_VALIDATION_H

#include <cstdint>

// (hecke-extensions slot 23, G2-X) HeckeFormsFastMulValidation (standard): the forms cross-route's fast q-series
// product (NTT over two NTT primes + CRT, the delayed-reduction schoolbook below the threshold) and the E_4 / E_6
// divisor sieve against the naive O(N^2) routes (the oracle and the rollback arm HECKE_FORMS_FASTMUL=0), byte for
// byte: random and extreme inputs at lengths around the thresholds, moduli up to the word limit 2^16, the fallback;
// an independent long-length identity (E_4^2 = E_8) beyond the oracle's reach; the real cross-route spaces and
// operators (BuildModularSpace / BuildCuspSpace / HeckeOnCusp at the sat route's precision) and whole SAT_DEFECT
// cells under both scopes.  (slot-24 FNTT) Arms (N1)-(N5): the FNTT route of BuildCuspSpace / BuildModularSpace /
// HeckeOnCusp against the schoolbook / naive products, the G2-X route (rollback HECKE_FORMS_NTT_CACHE=0) and the independent
// HeckeSturmBound series, SIMD and scalar butterflies, ell up to 1000003 (the forms range past 2^16).  See the .cpp for the grid.
uint32_t HeckeFormsFastMulValidation();

#endif // HECKE_EXT_FORMS_FAST_MUL_VALIDATION_H
