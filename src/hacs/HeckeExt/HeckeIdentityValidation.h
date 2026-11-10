// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_IDENTITY_VALIDATION_H
#define HECKE_EXT_IDENTITY_VALIDATION_H

#include <cstdint>

// hecke-extensions slot 1, WP-2 -- validation of `hecke_ext::IdentityCertificate`
// (HeckeIdentityCertificate.h), the E0a subset of [X Sec.1.3]. Returns 0 on
// success. `fast` tier (docs/implementation/hecke_extensions_slot1_plan.md
// Sec.3 WP-2 scope 3).
//
// ARMS.
//  (1) Real pipeline points spanning: the classical case (dim H^0 = 0), a
//      Dickson-exceptional locus (dim H^0 != 0, where the ROBUST form of
//      Eichler-Shimura is the one that survives), n = 0 (not applicable),
//      odd n (not applicable), and ell in {2, 3} (Eichler-Shimura / d_2 = 0 /
//      the p^2 recursion not applicable there, commutativity still is) --
//      `IdentityCertificate::verify()` returns 0 at every point, with the
//      expected per-item applicability.
//  (2) A tampered-certificate arm on a real, otherwise-passing certificate:
//      independently corrupting dim_H1, one entry of A_p_prime (breaking
//      commutativity) and one entry of A_p2 each makes verify() fail with
//      the corresponding first-failure code (1, 2, 4) and leaves the other
//      applicable items passing.
uint32_t HeckeIdentityValidation();

#endif // HECKE_EXT_IDENTITY_VALIDATION_H
