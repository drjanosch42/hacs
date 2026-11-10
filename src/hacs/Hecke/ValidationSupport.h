// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Hecke/ValidationSupport.h -- the helpers the Hecke validations used to copy
// (S24 CL-3; CODEAUDIT B2; docs/implementation/slot24/CLEANUP_plan_1030.md)
// ===========================================================================
//
//   HeckeDatum / MakeHeckeDatum : the level-1 coset + tau pair of the primitive
//        double coset Gamma diag(1, p^nu) Gamma on the fixed model (7 identical
//        8-line MakeDatum copies; each validation keeps a one-line wrapper that
//        supplies its own `Model`);
//   find_simplex_index          : the sorted-vertex lookup (4 copies; each keeps
//        its own exception text, passed as `err`).
// Header-only; the scalar helpers live in ModArithmetic/ScalarMod.h and the
// stopwatch in Util/Timing.h.
// ===========================================================================
#pragma once
#ifndef HECKE_VALIDATION_SUPPORT_H
#define HECKE_VALIDATION_SUPPORT_H

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "HeckeCosetTables.h"
#include "HeckeOperatorFamily.h"   // hecke::tau_construction_mutex()
#include "HeckeTauTable.h"
#include "SL2ZTessellationTables.h"

namespace hecke {
namespace vsupport {

struct HeckeDatum {
    std::unique_ptr<hecke::HeckeCosetTables> cosets;
    std::unique_ptr<hecke::HeckeTauTable> tau;
};

inline HeckeDatum MakeHeckeDatum(const SimplicialComplex& K, const hecke::SL2ZTessellationTables& tess, uint64_t p,
                                 uint32_t nu, uint32_t coef, uint32_t n) {
    HeckeDatum d;
    std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
    d.cosets.reset(new hecke::HeckeCosetTables(p, nu, coef, n));
    d.tau.reset(new hecke::HeckeTauTable(K, tess, *d.cosets, coef, n));
    return d;
}

using SimplexTable = std::vector<std::vector<std::vector<uint32_t>>>;

inline uint32_t find_simplex_index(const SimplexTable& simplices, uint32_t q, std::vector<uint32_t> verts,
                                   const char* err) {
    std::sort(verts.begin(), verts.end());
    const auto& list = simplices.at(q);
    for (uint32_t k = 0; k < list.size(); ++k) {
        if (list[k] == verts) return k;
    }
    throw std::runtime_error(err);
}

} // namespace vsupport
} // namespace hecke

#endif // HECKE_VALIDATION_SUPPORT_H
