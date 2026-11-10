// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_ALGEBRA_INVARIANTS_VALIDATION_H
#define HECKE_EXT_ALGEBRA_INVARIANTS_VALIDATION_H

// hecke-extensions slot 9, WP-13 (console phase C5b) -- the validation of the
// two console-free units behind the C5b bindings:
//
//   * HeckeExt/HeckeAlgebraInvariants (the certificate-free local-algebra
//     route of the verbs `radical`, `tangent_space`, `socle`,
//     `nilpotency_index`, `is_local`, `loewy`, `socle_series`,
//     `nilpotent_part`):
//       (A) against HeckeLocalAlgebra / HeckeModuleStructure on every
//           certified-local ideal of a set of driver cells -- the algebra
//           basis entrywise, dim rad == dim m_S, the nilpotency index, the
//           Hilbert function, the socle, the embedding dimension, the
//           Jordan-Chevalley nilpotent part == LocalAlgebra::n entrywise, and
//           the Loewy / socle series == ModuleStructure's m_dims / socle_dims;
//       (B) constructed algebras with known invariants: F[x]/(x^3), the
//           non-Gorenstein F[x,y]/(x,y)^2, a split semisimple family, and the
//           GALOIS-CONJUGATE TRAP (two commuting operators, each with an
//           irreducible minimal polynomial, generating F_49 x F_49 -- not
//           local; a per-operator primary test would call it local);
//       (C) the Jordan-Chevalley split on a matrix with a known split, and
//           the refusals (non-commuting family, non-stable subspace);
//   * HeckeExt/HeckeExtBindingSupport (the pipeline plumbing of the complex
//     and identity bindings), at (12, 5):
//       (D) the cone action at p = 2 certifies, the boundary operator's
//           degrees, the production involution's certificate, the sign
//           split, the Dickson map at (12, 5) <- (6, 5), and the identity
//           certificate with every item 1-7 assembled: code 0, items 6 and 7
//           passed, none unavailable.
//
// Returns the number of failing checks (0 = success).  Standard tier.

#include <cstdint>

uint32_t HeckeExtAlgebraInvariantsValidation(void);

#endif // HECKE_EXT_ALGEBRA_INVARIANTS_VALIDATION_H
