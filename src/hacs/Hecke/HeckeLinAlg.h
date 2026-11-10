// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Hecke/HeckeLinAlg.h -- the decomposition's small exact linear-algebra
// helpers, public.
//
// (hecke-extensions slot 7, WP-8 / console C2b, gap G14 of
// docs/console/00-console-interface-plan.md Sec. 14.2.)  The gap named two
// functions file-local to HeckeModuleDecomposition.cpp:
//
//  * SolveColumnsExact -- ALREADY public since hecke-extensions E0a, in
//    Hecke/SubspaceOps.h (moved there verbatim so E0b / E1a could share it);
//    this header re-exports it by including SubspaceOps.h, so a caller that
//    wants "the decomposition's linear algebra" needs one include.
//  * PartitionFromNullities -- moved here VERBATIM from the decomposition's
//    anonymous namespace (same body, same checks, same exception type and
//    text), so the decomposition's partitions -- and every golden object built
//    on them -- cannot move.  Its first consumer outside the decomposition is
//    the console's `jordan_type(A)` (Console/Bindings/CoreMath.cpp), which
//    used to carry its own copy of the same conjugate-partition arithmetic.
//
// [X] Sec. 2.2 Stage 5 names PartitionFromNullities as the Jordan-type step
// "(reused)"; HeckeExt/HeckeModuleStructure.cpp re-derived it only because it
// was file-local, and can now route through this header.
// ===========================================================================
#pragma once
#ifndef HECKE_HECKE_LIN_ALG_H
#define HECKE_HECKE_LIN_ALG_H

#include "SubspaceOps.h"   // SolveColumnsExact, RankOf, KernelAsMatrix, ...

#include <algorithm>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace hecke {

// Generalized-Jordan partition of a pi-primary operator from the nullities of
// pi(A)^r: n_r = (null_r - null_{r-1}) / deg(pi) counts the parts >= r; the
// partition is its conjugate. `nullities` has entries for r = 0..e (with
// nullities[e] the stabilized value). All divisibility/monotonicity facts are
// asserted.  `f` = deg(pi) (1 for a nilpotent operator); `who` names the
// caller in the error text.  Parts are returned in non-increasing order.
inline std::vector<uint32_t> PartitionFromNullities(const std::vector<uint32_t>& nullities,
                                                    uint32_t f, const char* who) {
    if (nullities.empty() || nullities[0] != 0 || f == 0) {
        throw std::runtime_error(std::string(who) + ": malformed nullity sequence");
    }
    const uint32_t e = static_cast<uint32_t>(nullities.size()) - 1;
    std::vector<uint32_t> n_r(e + 2, 0); // n_r[r] = #parts >= r, r = 1..e
    for (uint32_t r = 1; r <= e; ++r) {
        const uint32_t drop = nullities[r] - nullities[r - 1];
        if (nullities[r] < nullities[r - 1] || drop % f != 0) {
            throw std::runtime_error(std::string(who) + ": nullity drops not multiples of deg(pi)");
        }
        n_r[r] = drop / f;
        if (r > 1 && n_r[r] > n_r[r - 1]) {
            throw std::runtime_error(std::string(who) + ": non-monotone Jordan filtration");
        }
    }
    std::vector<uint32_t> partition;
    for (uint32_t r = e; r >= 1; --r) {
        const uint32_t count = n_r[r] - n_r[r + 1];
        for (uint32_t c = 0; c < count; ++c) partition.push_back(r);
    }
    std::sort(partition.begin(), partition.end(), std::greater<uint32_t>());
    return partition;
}

} // namespace hecke

#endif // HECKE_HECKE_LIN_ALG_H
