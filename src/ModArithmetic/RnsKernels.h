// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef ENGINE_RNS_KERNELS_H
#define ENGINE_RNS_KERNELS_H
// =====================================================================
// RnsKernels.h -- the residue-number-system kernels of the multimodular
// lifts (slot-25 EXZ2-P2; docs/implementation/slot25/EXPORTZ2_design.md
// v2 Sec. 2 "Entries", EXZ2P2_report.md).  Word-only (uint32 vectors):
// the exact big integers live in the caller (nf::BigInt in
// Integral/Descent/), this layer never sees one.
//
// THE LAYOUT.  An RNS value set is E integers x_e, held as L residue
// PLANES r_k[e] = x_e mod q_k (one plane per word prime, the "lane" axis
// of the parallelisation principle: the integer length becomes the plane
// count).  Garner's mixed radix x = d_0 + d_1 q_0 + d_2 q_0 q_1 + ... is
// computed plane by plane: digit plane k needs ONE shared modulus q_k for
// all E entries, so it is a vector operation over the entry axis --
//
//   garner_step:  d_k[e] = (r_k[e] - sum_{i<k} d_i[e] * c_{k,i}) * inv_k   (mod q_k)
//                 c_{k,i} = prod_{j<i} q_j mod q_k,  inv_k = (prod_{j<k} q_j)^{-1} mod q_k
//
// with every product a Shoup multiplication by a per-step constant
// (w' = floor(w 2^32 / q_k); needs q_k <= 2^31, ModCtx::shoup_available())
// and every sum a canonical mod_add.  Three bodies: portable, AVX2 (8 lanes,
// vpmuludq on even / odd halves + vpmulld + vpminud) and AVX-512 (16
// lanes, the same instruction set on zmm).  Each step of each body produces
// the CANONICAL residue of the same exact integer, so the three are
// bit-identical by construction; KernelBitIdentityValidation compares
// them entry for entry (and against a scalar oracle).  Selection as the
// GEMM: the forced path, else kernels::isa_auto under ENGINE_KERNEL.
//
// The digit planes d_k are what the caller assembles into big integers
// (Horner over k, at the very end: "lift via CRT / mixed radix only at the
// end"); the symmetric sign is read from the digits (garner_negative).
// =====================================================================

#include <cstddef>
#include <cstdint>
#include <vector>

#include "CpuFeatures.h"

namespace engine {
namespace rns {

using IsaPath = cpu::IsaPath;

// The precomputed constants of a Garner plan over the primes q_0 .. q_{L-1} (distinct, odd, 2 < q <= 2^31).
struct GarnerPlan {
    std::vector<uint32_t> primes;
    std::vector<uint32_t> coef, coef_shoup;   // row k: c_{k,i}, i < k (row-major L x L, upper part unused)
    std::vector<uint32_t> inv, inv_shoup;     // inv_k
    explicit GarnerPlan(const std::vector<uint32_t>& primes);
    size_t lanes() const { return primes.size(); }
};

// One digit plane (see the header): d_k[0..E) from the residue plane r_k and the digit planes d_0 .. d_{k-1}.
// digits[i] points at plane i (each of E entries, d_i[e] < q_i); out may alias res.  path Auto = the best body under the ceiling.
void garner_step(const GarnerPlan& plan, uint32_t k, const uint32_t* const* digits, const uint32_t* res, uint32_t* out, size_t E,
                 IsaPath path = IsaPath::Auto);
// All digit planes: res[k] = the residue plane of prime k (E entries each); digits resized to L planes of E.
void garner_digits(const GarnerPlan& plan, const std::vector<const uint32_t*>& res, size_t E, std::vector<std::vector<uint32_t>>& digits,
                   IsaPath path = IsaPath::Auto);
// The sign of the symmetric representative in (-P/2, P/2] of entry e: true iff the mixed-radix value exceeds (P - 1)/2 (P odd; the
// digits of (P - 1)/2 are (q_k - 1)/2 for every k).
bool garner_negative(const GarnerPlan& plan, const std::vector<std::vector<uint32_t>>& digits, size_t e);

// The body that `path` resolves to ("portable" / "avx2" / "avx512"), and the bodies this build + CPU can run.
IsaPath garner_resolved_path(IsaPath path);
bool garner_avx2_available();
bool garner_avx512_available();
// The scalar oracle of one step (no Shoup: plain 64-bit % arithmetic), for the validation.
void garner_step_oracle(const GarnerPlan& plan, uint32_t k, const uint32_t* const* digits, const uint32_t* res, uint32_t* out, size_t E);

}  // namespace rns
}  // namespace engine

#endif
