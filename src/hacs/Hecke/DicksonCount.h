// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_DICKSON_COUNT_H
#define HECKE_DICKSON_COUNT_H

// hecke-extensions slot 1, WP-2 -- Hecke::DicksonInvariantCount, [X Sec.2.1].
//
// The Dickson-invariant exceptional locus of Sym^n(F_ell^2) as an
// SL_2(F_ell)-module ([E], the classical Dickson invariants of degree
// ell+1 and ell^2-ell for the polynomial ring F_ell[x,y]): n is exceptional
// iff Sym^n(F_ell^2) contains a copy of the trivial (or a Dickson-twisted)
// invariant, which happens exactly when n is a non-negative combination
//
//     n = a*(ell+1) + b*(ell^2 - ell),   a, b >= 0 integers.
//
// Before this header, THREE file-local copies of the BOOLEAN predicate
// existed (docs/implementation/hecke_extensions_slot1_plan.md Sec.3 WP-2
// scope 1): `HeckeOracleValidation.cpp:237` checked only the FIRST
// generator, (ell+1) | n -- an under-approximation that happens to miss the
// second Dickson degree ell*(ell-1) (e.g. n=20, ell=5); the other two
// (`HeckeEndToEndValidation.cpp:98`, `Golden/GoldenHarness.cpp:382`) already
// looped over b and checked both generators, which is the correct
// predicate. `DicksonInvariantCount` generalizes the correct form to a
// COUNT (needed by `HeckeIdentityCertificate`'s identity 1, [X Sec.1.3]),
// and `n > 0 && count > 0` reproduces the correct predicate exactly for all
// three former call sites (a strict refinement at the one call site that
// under-approximated).
//
// Pure counting, no search: for each b with b*(ell^2-ell) <= n, at most one
// a can work, decided by one mod check -- O(n / (ell^2-ell) + 1) time, the
// same loop shape the two correct copies already used.

#include <cstdint>

namespace hecke {

// #{(a, b) : a, b >= 0 integers, a*(ell+1) + b*(ell^2-ell) = n}. Returns 0
// for n == 0 (the trivial local system is not on this locus by convention,
// matching every pre-existing call site) and for ell < 2 (a malformed
// modulus, no representation possible).
inline uint64_t DicksonInvariantCount(uint32_t n, uint32_t ell) {
    if (n == 0 || ell < 2) return 0;
    const uint64_t d1 = static_cast<uint64_t>(ell) + 1u;
    const uint64_t d2 = static_cast<uint64_t>(ell) * (static_cast<uint64_t>(ell) - 1u);
    uint64_t count = 0;
    for (uint64_t b = 0; b * d2 <= n; ++b) {
        if ((static_cast<uint64_t>(n) - b * d2) % d1 == 0) ++count;
    }
    return count;
}

// The boolean predicate every former call site actually needs, spelled so
// the three copies collapse to this one call.
inline bool IsDicksonExceptionalCount(uint32_t n, uint32_t ell) {
    return n > 0 && DicksonInvariantCount(n, ell) > 0;
}

// (hecke-extensions slot 7, WP-8 / console C2b, gap G9 of
// docs/console/00-console-interface-plan.md Sec. 14.2) The predicate under the
// name the gap table and [ME Sec. 5 R8] use -- "the predicate, never a list".
// The same function as IsDicksonExceptionalCount; HeckeEndToEndValidation's
// last file-local wrapper is retired onto it.
inline bool IsDicksonExceptional(uint32_t n, uint32_t ell) {
    return IsDicksonExceptionalCount(n, ell);
}

} // namespace hecke

#endif // HECKE_DICKSON_COUNT_H
