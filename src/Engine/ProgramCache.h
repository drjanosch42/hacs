// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// =====================================================================
// ProgramCache.h -- compiled programs keyed by shape signature (Phase 4, D4)
// =====================================================================
//
// A program is a function of the SHAPE of a problem, not of its data: the
// block grid, the block pattern with its kinds and pool-id classes, d_V,
// the prime regime and the element width.  ShapeKey carries exactly that,
// plus the caller's `pattern_hash` of the concrete pattern it recorded
// against (for the Hecke assembly: the tau-table pattern of the concrete
// (n_sym, p, r, q, ell) -- plan D4's cache-poisoning caveat: the mod-ell
// fallback of the triangle-fill solve can change the support, so the key
// MUST be computed from the actual pattern before lookup, and a miss falls
// back to recording, never to replaying a stale program).
//
// pattern_hash_of(BlockMat) is the reference way to compute that value for
// a block matrix: grid, CSR, kinds, coefficients, transposition flags and the
// pool ids in first-occurrence numbering (so two pools with the same
// structure hash alike, and a block that changes identity does not).

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "../ModArithmetic/BlockMat.h"
#include "Program.h"

namespace engine {
namespace tape {

struct ShapeKey {
    std::vector<uint32_t> row_dims, col_dims;
    uint64_t pattern_hash = 0;
    uint32_t d_V = 0;
    uint8_t  regime = 0;
    uint8_t  elt_width = 0;
    uint64_t extra = 0;         // free for the consumer (e.g. q, route)
    std::string tag;            // program family ("theta", "solve", ...)

    bool operator<(const ShapeKey& o) const;
    bool operator==(const ShapeKey& o) const;
    uint64_t hash() const;
    std::string to_string() const;
};

template<class Elt>
uint64_t pattern_hash_of(const BlockMat<Elt>& A);

template<class Elt>
class ProgramCache {
public:
    struct Stats { uint64_t hits = 0, misses = 0, inserts = 0; };

    std::shared_ptr<const Program<Elt>> find(const ShapeKey& key);
    std::shared_ptr<const Program<Elt>> insert(const ShapeKey& key, Program<Elt>&& prog);
    size_t size() const;
    void clear();
    Stats stats() const;

    static ProgramCache& global();

private:
    mutable std::mutex m_;
    std::map<ShapeKey, std::shared_ptr<const Program<Elt>>> table_;
    Stats stats_;
};

} // namespace tape
} // namespace engine
