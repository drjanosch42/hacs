// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_GOLDEN_OBJECT_H
#define HECKE_EXT_GOLDEN_OBJECT_H

// ===========================================================================
// hecke-extensions slot 5, WP-3 -- the extension layer's golden-collector POD,
// in ONE place.
//
// WHY THIS FILE EXISTS.  `hecke_ext::GoldenObject` was defined -- identically,
// field for field -- in BOTH HeckeExt/HeckeBoundary.h (slot 2, WP-7) and
// HeckeExt/HeckeSignSplit.h (slot 4, WP-3).  Two definitions of one class in
// one namespace are legal only while no translation unit includes both, and
// until slot 5 none did: HeckeBoundaryParabolicValidation.cpp includes the
// first, HeckeExtRecordValidation.cpp the second.  WP-3 is the first TU that
// CALLS both collector families -- Golden/GoldenHarness.cpp -- and it does not
// compile against two definitions.
//
// The struct is moved here VERBATIM (same name, same namespace, same members,
// same order, same defaults), and both headers now include this one, so no
// declaration, no hash and no emitted value changes.  The two comment blocks
// that documented it are merged below.
// ===========================================================================

#include <cstdint>
#include <string>

namespace hecke_ext {

// One golden record a unit of the extension layer offers the harness.  The
// hash is FNV-1a-64 exactly as Golden/GoldenHarness.cpp forms it, so a record
// pushed through Collector::raw() is indistinguishable from one the harness
// computed itself.
struct GoldenObject {
    std::string object;   // e.g. "rank_r^1", "A_bd^1", "dim_Hplus^1"
    char tier = 'A';      // 'A' = gating invariant, 'B' = basis-dependent
    uint64_t rows = 0, cols = 0;
    uint64_t hash = 0;    // FNV-1a-64 exactly as Golden/GoldenHarness.cpp
};

} // namespace hecke_ext

#endif // HECKE_EXT_GOLDEN_OBJECT_H
