// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once

// FNV-1a-64 -- the ONE definition, shared by every golden/campaign/benchmark
// hash in the tree (hecke-extensions slot 1, WP-2; docs/implementation/
// hecke_extensions_slot1_plan.md Sec.3 WP-2 scope 4; [X, "hecke_extensions_
// implementation_plan.md"] Sec.1.1 "One `src/ModArithmetic/Fnv.h`").
//
// Before this header existed, the SAME byte-wise function was defined three
// times: src/hacs/Golden/GoldenHarness.cpp (`FnvU64` /
// `FnvBytes`, offset 0xcbf29ce484222325, prime 0x100000001b3),
// src/hacs/Campaign/CampaignDriver.cpp (byte-for-byte the
// same, by its own comment), and src/ModArithmetic/
// PerformanceSmokeValidation.cpp (the `Fnv` class, `feed`/`value`, same
// constants and byte order). The audit ([SPA] Sec. "Q3") verified all three
// compute the same function before unifying them here, so this header is
// value-preserving by construction: every golden hash, the campaign hashes
// and the `EngineDeterminismValidation` digest are bit-identical before and
// after.
//
// `engine::BlockPool.h`'s word-wise hash and `engine::tape::fnv_mix`
// (src/Engine/Program.h) are DIFFERENT functions (different byte
// order / mixing) and are explicitly NOT folded into this header -- see
// [X Sec.1.1].
//
// Placement: `src/ModArithmetic/PerformanceSmokeValidation.cpp` is
// compiled into the `ModArithmetic` static library (CMakeLists.txt), so a
// header under `src/hacs/` would be an upward include.
// This header has NO dependencies beyond <cstdint>/<string>, so it is
// reachable from ModArithmetic, from the Hecke/Golden/Campaign layer above
// it, and from a future console consumer, without violating any include
// direction in the tree.

#include <cstdint>
#include <string>

namespace engine {

// The FNV-1a-64 offset basis and prime, exactly as used at every pre-
// unification call site.
inline constexpr uint64_t kFnvOffset64 = 0xcbf29ce484222325ull;
inline constexpr uint64_t kFnvPrime64 = 0x100000001b3ull;

// Feed the 8 bytes of `v`, least-significant byte first, into running hash
// `h`. This is the free-function form used by GoldenHarness.cpp and
// CampaignDriver.cpp to build up matrix/record hashes incrementally.
inline uint64_t FnvU64(uint64_t h, uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        h ^= (v >> (8 * i)) & 0xffu;
        h *= kFnvPrime64;
    }
    return h;
}

// Feed every byte of `s` into running hash `h`.
inline uint64_t FnvBytes(uint64_t h, const std::string& s) {
    for (unsigned char c : s) {
        h ^= c;
        h *= kFnvPrime64;
    }
    return h;
}

// Hash a single scalar starting from the offset basis -- the common
// `FnvU64(kFnvOffset, v)` idiom.
inline uint64_t HashScalar64(uint64_t v) { return FnvU64(kFnvOffset64, v); }

// Stateful wrapper equivalent to PerformanceSmokeValidation.cpp's
// pre-unification `Fnv` class: `feed()` one 64-bit word at a time from the
// offset basis, read back with `value()`. Matrix/buffer-shaped feeders
// (`feed_matrix`, `feed_buf`) stay at the call site because they depend on
// types (`FlatMatrix`, `engine::DenseBuf<uint32_t>`) this dependency-free
// header must not include.
class FnvHasher {
public:
    void feed(uint64_t x) { h_ = FnvU64(h_, x); }
    uint64_t value() const { return h_; }

private:
    uint64_t h_ = kFnvOffset64;
};

} // namespace engine
