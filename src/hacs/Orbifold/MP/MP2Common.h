// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef ORBIFOLD_MP_MP2_COMMON_H
#define ORBIFOLD_MP_MP2_COMMON_H

// ===========================================================================
// (hecke-extensions slot 22, MP-2) Shared harness pieces of the four MP-2
// validations (OrbifoldMPComparisonValidation, OrbifoldMPRefinementValidation,
// OrbifoldMPAllDegreesValidation, OrbifoldMPNerveH3Slow): the tally, the frozen
// oracle row, and the two sides of a cohomology comparison --
//   * SourceH: the cocycles Z^q and the length of H^q of a small DENSE complex
//     (C_I, the hybrid with its projector, the Bredon complex's own basis);
//   * MPSide: the MP cochain complex of one InternedMPSet over Z/ell^k, Sym^n:
//     the coboundaries d^0 .. d^{q_top} and the lengths of H^0 .. H^{q_top}.
// Not a validation; no output of its own beyond what the callers print.
// ===========================================================================

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "AmalgamOracle.h"
#include "CochainMaps.h"
#include "Nerve.h"
#include "SparseLength.h"
#include "../../../ModArithmetic/DenseBuf.h"
#include "../../../ModArithmetic/ModCtx.h"

namespace orbifold_mp {

struct MPTally {
    uint32_t checks = 0, failing = 0;
    bool check(bool ok, const std::string& what) {
        ++checks;
        if (!ok) {
            ++failing;
            std::cout << "  [FAIL] " << what << "\n";
        }
        return ok;
    }
};

// The frozen mv_check.out (1) row (MvCheckOracle.inc); ENGINE_CHECK: present.
const MvCheckRow& FrozenRow(uint32_t ell, uint32_t n);

struct SourceH {
    std::vector<Vec> cocycles;   // generators of Z^q
    uint64_t len_Z = 0, len_B = 0;
    uint64_t lenH() const { return len_Z - len_B; }
};
// delta = delta^q (rows C^{q+1}), prev = delta^{q-1} (rows C^q; nullptr at q = 0), proj = the projector P_q whose
// image is C^q (nullptr: all of the layout).  Z^q = ker delta ∩ im P = ker [delta; 1 - P].
SourceH Source(const engine::ModCtx<uint32_t>& ctx, const engine::DenseBuf<uint32_t>& delta,
               const engine::DenseBuf<uint32_t>* prev, const engine::DenseBuf<uint32_t>* proj);

struct MPSide {
    uint32_t n = 0, ell = 0, k = 1, m = 2;
    std::vector<SparseRows> d;        // d^0 .. d^{q_top}
    std::vector<uint64_t> len_im;     // length im d^q
    std::vector<LengthStats> stats;   // RowSpanLength's statistics of d^q (fill-in, wall)
    std::vector<uint64_t> dimC;       // dim C^q (number of coordinates), q = 0 .. q_top + 1
    MPSide(const InternedMPSet& S, uint32_t n, uint32_t ell, uint32_t k, uint32_t q_top);
    int64_t lenH(uint32_t q) const;   // q <= q_top
    // The image of the given source cocycles (already mapped into C^q coordinates) in H^q.
    ImageInH image(uint32_t q, const std::vector<Vec>& mapped, bool cocycle_check = true) const;
};

// Apply a sparse map (rows = target coordinates) to every vector.
std::vector<Vec> MapAll(const SparseRows& phi, const std::vector<Vec>& v, uint32_t m);
std::vector<Vec> MapAllDense(const engine::DenseBuf<uint32_t>& phi, const std::vector<Vec>& v, uint32_t m);

std::string JoinV(const std::vector<int64_t>& v);

}  // namespace orbifold_mp

#endif
