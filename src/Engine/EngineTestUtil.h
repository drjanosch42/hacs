// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// EngineTestUtil.h -- shared helpers of the Phase 4 engine validations
// (fixed-seed generators shaped like BlockEliminationValidation's, FNV-64
// hashes of views, a check counter).  Validation-only; never linked into a
// result path.
#include <chrono>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "../ModArithmetic/BlockMat.h"
#include "../ModArithmetic/DenseBuf.h"
#include "../ModArithmetic/DenseView.h"
#include "Program.h"

namespace engine {
namespace tape {
namespace test {

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ull) {}
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    uint32_t below(uint32_t n) { return n == 0 ? 0 : static_cast<uint32_t>(next() % n); }
    template<class Elt> Elt elt(Elt p) { return static_cast<Elt>(next() % p); }
    template<class Elt> Elt nonzero(Elt p) { Elt v = elt<Elt>(p); return v == 0 ? Elt(1 % p) : v; }
};

// The raw SplitMix64 stream of the three kernel bit-identity validations (KernelBitIdentity, KernelAvx2BitIdentity,
// VnniGemmBitIdentity; CL-4 / C-11): the seed is taken AS IS (unlike Rng, a zero seed stays zero), so every stream is
// byte-identical to the per-file copies it replaced.  Each validation derives its own `below` (its argument/result width
// differs and the case lists depend on it).
struct SplitMix64 {
    uint64_t s;
    explicit SplitMix64(uint64_t seed) : s(seed) {}
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
};

// The bit-identity case driver: one check bumps `checks`; a failure bumps `fails` and prints the first 40 as "  [FAIL] <what>".
inline void bit_check(uint32_t& checks, uint32_t& fails, bool ok, const std::string& what) {
    ++checks;
    if (!ok && ++fails <= 40) std::cout << "  [FAIL] " << what << std::endl;
}
template<class T> std::string str(T v) { return std::to_string(v); }
// GemmOp labels (Set / Add / Sub) and their numeric tag in the printed case lines.
template<class Op> const char* opname(Op op) { return int(op) == 0 ? "Set" : int(op) == 1 ? "Add" : "Sub"; }
template<class Op> int opint(Op op) { return int(op); }

template<class Elt>
DenseBuf<Elt> random_buf(Rng& rng, uint32_t r, uint32_t c, Elt p, Layout layout = Layout::RowMajor) {
    DenseBuf<Elt> b(r, c, layout);
    for (uint32_t i = 0; i < r; ++i)
        for (uint32_t j = 0; j < c; ++j) b.at(i, j) = rng.elt<Elt>(p);
    return b;
}

template<class E>
uint64_t hash_view(const DenseView<E>& v) {
    uint64_t h = fnv_init();
    h = fnv_mix(h, v.rows()); h = fnv_mix(h, v.cols());
    for (uint32_t i = 0; i < v.rows(); ++i)
        for (uint32_t j = 0; j < v.cols(); ++j) h = fnv_mix(h, static_cast<uint64_t>(v.at(i, j)));
    return h;
}

enum class Shape { CqLike, ZqLike, Generic, Dependent };
inline const char* shape_name(Shape s) {
    return s == Shape::CqLike ? "Cq" : s == Shape::ZqLike ? "Zq" : s == Shape::Generic ? "generic" : "dependent";
}

// The input generator of BlockEliminationValidation (same shapes: relation
// rows I / -M / I-M, pencils, generic kinds incl. transposed windows,
// rank-deficient duplicates), with `max_shared` distinct dense blocks per
// (dr, dc) so the pool-id sharing the CSE relies on is exercised.
template<class Elt>
BlockMat<Elt> make_input(Rng& rng, std::shared_ptr<BlockPool<Elt>> pool, Elt p, Shape shape,
                         const std::vector<uint32_t>& rd, const std::vector<uint32_t>& cd, uint32_t density_pct,
                         uint32_t max_shared = 0) {
    const ModCtx<Elt> ctx(p);
    const uint32_t nbr = static_cast<uint32_t>(rd.size()), nbc = static_cast<uint32_t>(cd.size());
    BlockBuilder<Elt> b(pool, rd, cd, p);
    std::vector<typename BlockPool<Elt>::Id> shared_ids;
    std::vector<std::pair<uint32_t, uint32_t>> shared_dims;
    auto pick = [&](uint32_t r, uint32_t c) {
        uint32_t have = 0;
        for (size_t s = 0; s < shared_ids.size(); ++s)
            if (shared_dims[s].first == r && shared_dims[s].second == c) {
                ++have;
                if (rng.below(3) != 0 || (max_shared != 0 && have >= max_shared)) return shared_ids[s];
            }
        typename BlockPool<Elt>::Id id = pool->intern(random_buf<Elt>(rng, r, c, p));
        shared_ids.push_back(id); shared_dims.emplace_back(r, c);
        return id;
    };
    const Elt neg1 = ctx.neg(Elt(1));
    if (shape == Shape::CqLike || shape == Shape::Dependent) {
        for (uint32_t br = 0; br < nbr; ++br) {
            if (rd[br] == 0) continue;
            uint32_t lhs = rng.below(nbc), rhs = rng.below(nbc);
            if (cd[lhs] != rd[br] || cd[rhs] != rd[br]) {
                std::vector<uint32_t> ok;
                for (uint32_t c = 0; c < nbc; ++c) if (cd[c] == rd[br]) ok.push_back(c);
                if (ok.empty()) continue;
                lhs = ok[rng.below(static_cast<uint32_t>(ok.size()))];
                rhs = ok[rng.below(static_cast<uint32_t>(ok.size()))];
            }
            const auto id = pick(rd[br], rd[br]);
            if (lhs == rhs) b.set(br, lhs, BlockRef<Elt>::affine(Elt(1), neg1, id));
            else { b.set(br, lhs, BlockRef<Elt>::identity()); b.set(br, rhs, BlockRef<Elt>::dense(id, neg1)); }
            if (shape == Shape::Dependent && br + 1 < nbr && rd[br + 1] == rd[br] && rng.below(2) == 0) {
                const Elt s = rng.elt<Elt>(p);
                if (lhs == rhs) b.set(br + 1, lhs, b.compose(BlockRef<Elt>::affine(Elt(1), neg1, id), Elt(0), s));
                else { b.set(br + 1, lhs, BlockRef<Elt>::scalar(s)); b.set(br + 1, rhs, BlockRef<Elt>::dense(id, ctx.mul(s, neg1))); }
                ++br;
            }
        }
    } else if (shape == Shape::ZqLike) {
        for (uint32_t br = 0; br < nbr; ++br)
            for (uint32_t bc = 0; bc < nbc; ++bc) {
                if (rd[br] == 0 || cd[bc] == 0 || rng.below(100) >= density_pct) continue;
                const Elt bl = rng.elt<Elt>(p), brh = rng.elt<Elt>(p);
                if (rd[br] != cd[bc]) { b.set(br, bc, BlockRef<Elt>::dense(pick(rd[br], cd[bc]), brh == 0 ? Elt(1) : brh)); continue; }
                const auto id = pick(rd[br], cd[bc]);
                b.set(br, bc, BlockRef<Elt>::affine(bl, ctx.neg(brh), id));
            }
    } else {
        for (uint32_t br = 0; br < nbr; ++br)
            for (uint32_t bc = 0; bc < nbc; ++bc) {
                if (rd[br] == 0 || cd[bc] == 0 || rng.below(100) >= density_pct) continue;
                const bool sq = rd[br] == cd[bc];
                switch (rng.below(sq ? 6 : 3)) {
                    case 0: b.set(br, bc, BlockRef<Elt>::dense(pick(rd[br], cd[bc]))); break;
                    case 1: b.set(br, bc, BlockRef<Elt>::dense(pick(rd[br], cd[bc]), neg1)); break;
                    case 2: b.set(br, bc, BlockRef<Elt>::dense(pick(cd[bc], rd[br]), rng.nonzero<Elt>(p), 0, 0, true)); break;
                    case 3: b.set(br, bc, BlockRef<Elt>::identity()); break;
                    case 4: b.set(br, bc, BlockRef<Elt>::scalar(rng.nonzero<Elt>(p))); break;
                    default: b.set(br, bc, BlockRef<Elt>::affine(Elt(1), neg1, pick(rd[br], cd[bc]))); break;
                }
            }
    }
    return b.finalize();
}

// The same pattern as A with every dense block replaced by a fresh random
// block (pool ids in first-occurrence order): a "new data, same shape"
// instance for replay tests.
template<class Elt>
BlockMat<Elt> resample_data(Rng& rng, const BlockMat<Elt>& A, std::shared_ptr<BlockPool<Elt>> pool) {
    std::map<uint32_t, uint32_t> ids;
    BlockBuilder<Elt> b(pool, A.row_dims(), A.col_dims(), A.p());
    for (uint32_t br = 0; br < A.nbr(); ++br)
        for (uint32_t k = A.row_begin(br); k < A.row_end(br); ++k) {
            BlockRef<Elt> r = A.ref_at(k);
            if (r.has_data()) {
                auto it = ids.find(r.pool_id);
                if (it == ids.end()) {
                    const DenseBuf<Elt>& src = A.pool()->buf(r.pool_id);
                    it = ids.emplace(r.pool_id, pool->intern(random_buf<Elt>(rng, src.rows, src.cols, A.p()))).first;
                }
                r.pool_id = it->second;
            }
            b.set(br, A.col_at(k), r);
        }
    return b.finalize();
}

struct Checker {
    uint64_t checks = 0, fails = 0;
    void check(bool ok, const std::string& what) {
        ++checks;
        if (!ok) { ++fails; std::cout << "  FAIL: " << what << std::endl; }
    }
    template<class E1, class E2>
    void same(const DenseView<E1>& a, const DenseView<E2>& b, const std::string& what) {
        check(same_entries(a, b), what + " (entrywise)");
    }
};

inline double seconds_since(const std::chrono::steady_clock::time_point& t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

} // namespace test
} // namespace tape
} // namespace engine
