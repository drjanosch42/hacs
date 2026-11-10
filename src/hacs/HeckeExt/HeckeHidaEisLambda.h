// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_HECKE_HIDA_EIS_LAMBDA_H
#define HECKE_EXT_HECKE_HIDA_EIS_LAMBDA_H

// ===========================================================================
// (hecke-extensions slot 25, package H1) THE EISENSTEIN COMPONENT OVER LAMBDA AT
// IRREGULAR PAIRS -- the finite-weight readings of [U] part 6 (sec:u-eislambda,
// docs/latex/hida_hecke_algebra.tex 0.7.0-pre.2): Lemma H0 (lem:u-H0), the
// generator test of prop:u-H11, the data (D1), (D3) of prop:u-H12, the rung
// prediction (U23) of pr:u-H1, the Lean checker K9's integers, and the G8 source of
// the record integral_hecke_structure/v1 under the non-default settings value
// heckez.G8 = hida ([C, sec:recordschema] item 8 (i), (j)).  Card: docs/
// implementation/slot24/heckez/06-work-packages.md "H1"; plan 05-hida-targets.md
// Sec. 2; report docs/implementation/slot25/H1_report.md.
//
// Everything is the q-expansion arm of HeckeHidaLambda (ComputeFormsWeight on the
// Miller basis, the ordinary projector by T_ell, the certified Fitting split of the
// full residual system, EisensteinBlock): pure host C++, mod ell^W < 2^84, no new
// arithmetic.  Writes nothing, no golden object, no digest of record.
//
// PARALLEL STRUCTURE.  Every reading is a function of one (ell, k): the callers run
// pairs, weights and operators independently (05 Sec. 2.8); nothing here shares state.
// ===========================================================================

#include <cstdint>
#include <string>
#include <vector>

#include "HeckeHidaLambda.h"
#include "../Integral/HeckeZRecord.h"

namespace hecke_ext {
namespace eislambda {

using lambda::u128;
using lambda::Vec;

// The T_p read at weight k: every prime p != ell with p <= max(k/12 + 1, 7) (the Sturm bound of M_k at level 1, with 2, 3, 5, 7
// always present: the Fitting split uses the first four), plus `extra` (primes != ell), ascending, unique.
std::vector<uint64_t> SturmPrimes(uint64_t ell, uint32_t k, const std::vector<uint64_t>& extra = {});

// One weight's ordinary Eisenstein block (the m_Eis-part of e S_k, the full residual system: EisensteinBlock).
struct BlockReading {
    uint64_t ell = 0;
    uint32_t k = 0, W = 0;
    uint32_t s_M = 0;                 // rank of e M_k
    uint32_t g = 0, rank_M = 0;       // rank of the block in e S_k (= g_a, [U, sec:u-eislambda] notation) and in e M_k (= 1 + g)
    std::vector<uint64_t> primes;     // the T_q read (q != ell), ascending
    std::vector<int32_t> v_det;       // per q: v_ell det(T_q - e_q | block) mod ell^W (-1: >= W); g = 1: v(a_q(f_k) - (1 + q^{k-1}))
    std::vector<bool> gen;            // per q: T_q - e_q generates the block's algebra mod ell (Lemma H0(3))
    int32_t v_Bk = -1;                // v_ell(B_k / k) mod ell^W (-1: >= W)
    int32_t L_forms = -1;             // g = 1: min_q v_det (the congruence length on the forms side); -1 otherwise
    uint64_t I0_q = 0;                // the first q with gen and v_det = v_Bk (prop:u-H11 at g = 1; Step 2 of prop:u-H12 at g >= 2); 0 = none
    bool lambda_one = false;          // v(G(x_k) - G(x_{k+ell-1})) = 1 ((D3'), so lambda(G_a) = 1, mu = 0)
    Vec f2;                           // chi(T_2 | block) mod ell^W, low -> high (the Hensel factor f_k of (D1) where they agree)
    u128 e2 = 0;                      // 1 + 2^{k-1} mod ell^W
    u128 Bk_l2 = 0, Gk_l2 = 0, Gnext_l2 = 0;   // B_k/k, G(x_k), G(x_{k+ell-1}) mod ell^2 (K9)
    lambda::FormsWeight fw;           // the forms data (the operators of the block are re-derived from it on demand)
    double secs = 0;
};
// ell >= 5, k >= 4 even, (ell - 1) !| k, k - 1 >= W >= 2, ell^W < 2^84 (ENGINE_CHECK, so a refusal is an exception).
BlockReading ReadBlock(uint64_t ell, uint32_t k, uint32_t W, const std::vector<uint64_t>& extra_primes = {});
// The same from forms data already computed (fw.primes must contain 2, 3, 5, 7 unless one is ell; `primes` = the T_q of fw).
BlockReading ReadBlockFrom(const lambda::FormsWeight& fw, const std::vector<uint64_t>& primes);

// "HIDA_K9 ell= k= g= fk_mod_l2=[..] ek_mod_l2= Bk_over_k_mod_l2= G_k_mod_l2= G_next_mod_l2= lambda_residue= checks=<list>" -- the
// integers of the Lean checker K9 (05 Sec. 2.9, 07 v0.3 Sec. 5.9), all below ell^2, with the four kernel checks evaluated here:
// f_k == (x - e_k)^g mod ell; v(f_k(e_k)) = v(B_k/k) (both < 2, so read mod ell^2); g = 2: v(disc f_k) = 1; the lambda-test residue
// (G(x_k) - G(x_{k+ell-1}))/ell mod ell != 0.  *all_ok: every check that applies held.
std::string K9Line(const BlockReading& b, bool* all_ok);

// The G8 source (heckez.G8 = hida; Integral/HeckeZRecord.h RecordSources::hida_block): theta = sum theta[i] T_{ops[i]} (T_ell = U on
// e M_k), its charpoly on e M_k mod ell, and the block reading at (ell, k, W).  served = false (with the cap) where ReadBlock refuses.
heckez::record::RecordSources::HidaBlockOut HidaBlockAt(uint64_t ell, uint32_t k, uint32_t W, const std::vector<uint32_t>& ops,
                                                         const std::vector<uint32_t>& theta);

// The rung prediction (U23): F_S(Y) = Y^g + ... + c on the cuspidal block, its coefficient polynomials in Z = X - x_{centre} known
// mod (ell^K, Z^m) (the truncation's F on e M_k is Y F_S(Y), RunClass's Eisenstein factor), evaluated at Z = x_{k*} - x_{centre}
// (exact mod ell^K when m >= K, as v(Z) >= 1) and shifted to x = Y + (1 + 2^{k*-1}).  Returns f_{k*}(x) mod ell^K, low -> high.
Vec RungPrediction(uint64_t ell, uint32_t K, uint32_t m, uint32_t centre_k, uint64_t k_star, const std::vector<Vec>& F_M);

}  // namespace eislambda
}  // namespace hecke_ext

#endif  // HECKE_EXT_HECKE_HIDA_EIS_LAMBDA_H
