// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_LATTICE_MODULE_H
#define HECKE_EXT_LATTICE_MODULE_H

// ===========================================================================
// (hecke-extensions slot 19, WP-LC) THE LATTICE COMPARISON: H^1(Gamma; Sym^n Z/ell^K)_m as a module over the Hecke
// algebra, on the shipped Z/ell^K route -- the code of [E] sec:block258, Check chk:latcmp.
//
// Inputs are shipped objects only (no new elimination):
//   * H = H^1(Gamma; Sym^n Z/ell^K) = (+)_i Z/ell^{a_i} with its cyclic-summand basis b_i and T_p on it
//     (EquivariantCohomologyOnKT<uint64_t> under ScopedCoefficientRing(K), HeckeOperatorOnKT's H_matrix_module(1)) --
//     the recipe of hecke_ext::ringalg::RunModule (Native) on uint64_t;
//   * iota^* on H by [C, def:iota](iv) in flat 1-cochain coordinates,
//     (iota^* f)(sigma_s) = -eps_s rho_n(iota) f(iota_K sigma_s), rho_n(iota) = diag((-1)^{n-j}), read back by
//     H_class_coordinates (the construction of WP-LV's EisensteinBernoulliRoute, re-derived here; iota^2 = 1 and
//     [iota^*, T_p] = 0 are ENGINE_CHECKed);
//   * the Howell / Smith layer of [E] sec:ring: engine::elim::cyclic_decomposition / cyclic_coordinates (the block and
//     its sign parts), engine::elim::invariant_factors (Smith valuations of an operator = the cokernel's exponents).
//
// What a cell computes, for a system m = (ell, T_p - a_p : p in system primes):
//   (i)   e_m on H: per system prime the F_ell idempotent polynomial E_{p,a} (1 mod (x - a)^mult, 0 mod the cofactor
//         of chi_p mod ell), E(T_p) over Z/ell^K (Horner), the product, Newton-lifted e <- 3e^2 - 2e^3; e^2 = e and
//         e T_p = T_p e are ENGINE_CHECKed;
//   (ii)  the block e_m H, embedded in (Z/ell^K)^r by g_i -> ell^{K - a_i} g_i, its cyclic decomposition (exponents:
//         the block is FREE iff all equal K), the T_p and iota^* restricted to it (every restricted column
//         re-verified by cyclic_coordinates);
//   (iii) the sign parts H_m^+- = image of (1 +- iota^*) / 2 on the block, again by cyclic decomposition, the
//         operators restricted.
// The readings (Smith valuations, mu, P(t) = 0, T_p = Q_p(t)) are made on those restricted matrices by the caller
// (HeckeLatticeModuleValidation).
//
// Pipeline side only, console-free; writes nothing, emits no record and no golden object.  Value-free.
// ===========================================================================

#include <cstdint>
#include <string>
#include <vector>

#include "../Hecke/HeckeOrdinaryPart.h"       // RingMat
#include "ZellK.h"

namespace hecke_ext {
namespace latmod {

// A square matrix over Z/ell^K on a FREE module (Z/ell^K)^r, column convention (X b_j = sum_i a[i*r+j] b_i), and its
// arithmetic: the shared kit (ZellK.h; S24 CL-G1).
using FreeMat = zk::FreeMat;
using zk::Identity;
using zk::Mul;
using zk::Add;
using zk::Scale;                                                 // c * A (c reduced mod ell^K)
using zk::AddScalar;                                             // A + c * 1
using zk::ReduceTo;                                              // the same matrix over Z/ell^M, M <= K
using zk::DirectSum;
using zk::EvalPoly;                                              // P(X) by Horner, coefficients low -> high

// The Smith valuations <d_1, ..., d_r>_K of X on (Z/ell^K)^r (ascending; d = K means "0 mod ell^K", i.e. >= K):
// the cokernel is (+)_i Z/ell^{d_i}.  By engine::elim::invariant_factors(I, X) (the Smith rule of [E] sec:ring).
std::vector<uint32_t> SmithValuations(const FreeMat& X);
// rank over F_ell of the horizontal concatenation [X_1 | X_2 | ...] mod ell (the F_ell-dimension of sum_i im X_i
// on H / ell H); so dim H / (ell, X_1, X_2, ...) H = r - RankModEllJoint.  In-house F_ell elimination (independent of
// the Smith rule).
uint32_t RankModEllJoint(const std::vector<FreeMat>& Xs);

// One sign part or the whole block, with its operators (aligned with LatticeBlock::primes) and iota^*.
struct FreePart {
    uint32_t rank = 0;
    std::vector<uint32_t> exps;        // the cyclic-decomposition exponents (all == K: free)
    bool free() const;
    std::vector<FreeMat> T;            // T_p restricted
    FreeMat iota;                      // iota^* restricted (on a sign part: +-1)
};

struct LatticeBlock {
    uint32_t n = 0;
    uint64_t ell = 0;
    uint32_t K = 0;
    std::vector<uint64_t> primes;      // ascending, the T_p computed
    std::vector<uint64_t> sys_primes;  // the primes of the system
    std::vector<uint64_t> sys_a;       // a_p in [0, ell)
    std::vector<uint32_t> H_exps;      // H^1(Gamma; Sym^n Z/ell^K) cyclic exponents (the whole module)
    uint32_t D = 0;                    // dim_F_ell of the block mod ell (rank of e mod ell)
    FreePart block;                    // e_m H
    FreePart plus, minus;              // the iota^* = +1 / -1 parts of e_m H
    double seconds = 0.0;
    const FreeMat& op(const FreePart& part, uint64_t p) const;   // ENGINE_CHECKs p in primes
};

// The whole module H^1(Gamma; Sym^n Z/ell^K) = (+)_i Z/ell^{a_i} with T_p (p in primes, ascending) and iota^*.
struct LatticeModule {
    uint32_t n = 0;
    uint64_t ell = 0;
    uint32_t K = 0;
    std::vector<uint64_t> primes;
    std::vector<uint32_t> exps;
    std::vector<hecke::ordinary::RingMat> T;
    hecke::ordinary::RingMat iota;
    double seconds = 0.0;
};
// n >= 2 even, ell >= 5 prime, K >= 2, ell^K < 2^62.  Throws std::logic_error (ENGINE_CHECK) if iota^2 != 1 or
// [iota^*, T_p] != 0.
LatticeModule RunLatticeModule(uint32_t n, uint64_t ell, uint32_t K, const std::vector<uint64_t>& primes);

// The block of the system (sys_primes[i] -> sys_a[i], each a computed T_p) and its sign parts.  Throws
// std::logic_error (ENGINE_CHECK) if some a_p is not an eigenvalue of T_p mod ell, e^2 != e, e does not commute with
// the T_p or iota^*, or a restricted column does not re-verify.
LatticeBlock LocalizeBlock(const LatticeModule& N, const std::vector<uint64_t>& sys_primes, const std::vector<uint64_t>& sys_a);

// The block reduced to precision M <= K (H_M = H_K / ell^M on a free block, [E] chk:latcmp (a)).
LatticeBlock ReduceBlock(const LatticeBlock& B, uint32_t M);

std::string JoinU32(const std::vector<uint32_t>& v);

} // namespace latmod
} // namespace hecke_ext

#endif // HECKE_EXT_LATTICE_MODULE_H
