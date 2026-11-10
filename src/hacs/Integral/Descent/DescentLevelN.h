// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_DESCENT_DESCENT_LEVEL_N_H
#define INTEGRAL_DESCENT_DESCENT_LEVEL_N_H

// ===========================================================================
// DESCENT milestone D1 (hecke-extensions slot 24; docs/implementation/slot24/DESCENT_plan.md v0.3 Sec. 4 "D1", Sec. 2,
// Sec. 3.5; referee docs/implementation/slot24/C-D1_referee.md F1, F3, F4).  [C] = docs/latex/gl2_contract.tex
// (v1.23.0-pre.17), cited by label: sec:descentN, prop:descentN, prop:descentNthree, prop:badprimesN, lem:dicksonSL2, prop:levelorbit,
// prop:certifiedeell.
//
// WHAT IS COMPUTED.  H^q(Gamma_0(N); Sym^n Z^2) = H^q(SL_2(Z); M_N), M_N = Ind_{Gamma_0(N)}^Gamma Sym^n Z^2, trivial
// character, q = 0 .. 5, as an AbelianGroupType per q, glued by [C, prop:descentN](2) (free rank + one primary part
// per prime) from:
//   (1) the free ranks b_0, b_1 in CLOSED FORM: b_0 = [n = 0], b_1 - b_0 = the right-hand side of [C, prop:levelorbit](3)
//       over Q (m_N, nu_2, nu_3, d_S, d_ST; d_S, d_ST exact character sums), b_q = 0 for q >= 2; cross-checked against the
//       SHIPPED sparse cokernel (hecke::level::SparseCokernel) at one good prime ell' >= 5, ell' > n + 1, ell' !| 6N
//       ([C, prop:descentN](6); the rule rests on Claims A, B alone, F1(a)).
//   (2) at each ell >= 5 of the closed-form list B(n, N) of [C, prop:badprimesN](2) (Dickson count delta_ell(n) for
//       ell !| N, Borel count beta_ell(n) for ell | N, up to ell = n + 1): the ell-primary torsion of H^1 by ONE Smith
//       over Z/ell^K of the sparse cokernel presentation of [C, prop:descentN](7)(b) -- rows generating M^S + M^{ST} in
//       the basis of M^{-I} = M, the generators the transported blocks on free orbits and the averaging-idempotent
//       columns at fixed points (prop:levelorbit (1); F4: admissible with no certificate), eliminated SPARSELY with
//       unit pivots, the leftover (all entries divisible by ell) by SmithLevelsEll; certified iff s = rank T^0 - b_0
//       (closed form), and the summand count must equal s_ell of prop:badprimesN(1),(2).  Every other prime >= 5 carries
//       no torsion BY THEOREM ([C, prop:badprimesN](2): Claims A, B at ell | N or ell > n + 1; at ell !| N the Dickson
//       count through [C, cl:dimH1](2) and [C, lem:dicksonSL2], since pre.16).  The direct Smith of d^0 of C_eq(M_N) (x) Z/ell^K ((7)(a), RunBredonDirectSmith) is the
//       measured alternative (dense; chosen against by measurement, DESCENT1_report.md).
//   (3) the 3-primary part -- the shortcut below is [C, prop:descentNthree] (v1.23.0-pre.17, not yet refereed; the D1 code
//       audit's M1): its hypothesis is nu_3(N) = 0, i.e. 9 | N or some prime p = 2 mod 3 divides N (p = 2 included), the
//       code's gate exactly (arm (S) checks Nu3(N) == 0 <=> that condition for N <= 5000); checked against Iso^(3) by
//       arm (S): if Gamma_0(N) has no element of order 3 (nu_3(N) = 0, i.e. x^2 + x + 1 = 0 has no root mod
//       N -- [C, prop:modeltorsion]), the SAME sparse route at ell = 3 (U acts freely on P^1(Z/N), so M_N restricted
//       to <U> is induced from <-I> and the tree's Bredon complex is 3-adically exact: no division by 3 is needed, the
//       S-fixed points divide by 2 only); else Iso^(3) with the Ind module on ring3k (D0's level-N probe route),
//       degrees <= 6, behind an RSS prediction.
//   (4) the 2-primary part: Iso^(2) with the Ind module on ring2k (D0's probe route), degrees <= 4, H^5 := H^3 by
//       [C, prop:descentN](5) (cup with res(e)), behind an RSS prediction.  Above the RSS ceiling the descent REFUSES
//       (never a partial answer).  rank_Q d^q on Iso^(ell) from the certified Betti numbers (source (4)(c)(ii)).
// For n odd: 2 kills H^* ([C, prop:badprimesN](2)); only (4) runs.
// RSS (D1 audit m3): the dense pieces (Iso^(2), Iso^(3), the DirectBredon alternative at 424 D^2 B) and the dense residual of
// the sparse route (rows x cols x 4 B) are predicted and refused above rss_cap_bytes; the SPARSE fill itself and the shipped
// good-prime SparseCokernel are not guarded (measured <= 1.2 GiB at D = 37 800).
// (PAR-LN, PARLN_report.md) The good prime, the ell >= 5 eliminations and the ell = 3 shortcut's elimination run CONCURRENTLY, at most
// floor(0.9 rss_cap_bytes / PredictSparseElimRss) at a time (the per-item prediction, calibrated with a margin on the measured
// growth); their dense residuals follow one after another in the serial order under the ceiling rule above.  The stage ends before the
// 2- and 3-adic pieces start, so the call's peak is the larger of the stage's joint term and those pieces' predictions.  Lever
// HECKE_DESCENT_PAR_LN=0 restores the serial order (no value, refusal or message depends on it; IntegralDescentParLnValidation).
// A result with ANY uncertified piece, a summand count off prop:badprimesN, a closed form off the good prime, or a
// piece above the ceiling is REFUSED: refused = true, the reason, H EMPTY.
// Not built: the nebentypus M_N(chi) over O_chi ([C, prop:descentN](8); not cheap: DESCENT1_report.md Sec. 6).
// ===========================================================================

#include <cstdint>
#include <string>
#include <vector>

#include "DescentAssembly.h"

namespace integral_descent {

// ---- (1) closed forms ----------------------------------------------------------------
struct LevelNCounts {
    uint32_t N = 1, n = 0, m = 1, d = 1, D = 1;
    uint32_t nu2 = 0, nu3 = 0;
    uint32_t dS = 0, dST = 0;          // dim (Sym^n Q^2)^{<S>}, ^{<ST>} (exact character sums)
    uint32_t dimMS = 0, dimMST = 0;    // dim M_N^{<S>}, M_N^{<ST>} over Q (prop:levelorbit (1))
    uint32_t b0 = 0, b1 = 0;           // Betti numbers (n odd: 0)
    uint32_t rank_T0 = 0;              // dimMS + dimMST: rank of T^0
};
LevelNCounts LevelNClosedForm(uint32_t N, uint32_t n);
// dim (Sym^n Q^2)^{<g>} for g = S (order 4) or ST (order 6), from the eigenvalues zeta^{+-1}: (1/o) sum_i tr g^i.
uint32_t SymInvariantDim(uint32_t n, uint32_t order);
// [C, prop:badprimesN]: delta_ell(n), beta_ell(n, t).
uint32_t DeltaCount(uint32_t ell, uint32_t n);
uint32_t BetaCount(uint32_t ell, uint32_t n, uint32_t t = 0);
// s_ell(M_N) for ell >= 5 (n > 0 even: delta at ell !| N, beta at ell | N; n = 0 or odd: 0).
uint32_t PredictedSummands(uint32_t N, uint32_t n, uint32_t ell);
// B(n, N) cap [5, inf): (ell, s_ell) with s_ell > 0, ascending.
std::vector<std::pair<uint32_t, uint32_t>> BadPrimesN(uint32_t N, uint32_t n);
// The good prime: the least prime >= 10007, > n + 1, prime to 6N.
uint32_t GoodPrimeN(uint32_t N, uint32_t n);

// ---- (2), (3) the sparse cokernel over Z/ell^K ([C, prop:descentN](7)(b)) -------------------
struct SparseSmithRun {
    uint32_t ell = 0, K = 0, D = 0;
    uint64_t generators = 0, generator_nnz = 0, dependent = 0, deferred = 0, final_nnz = 0, peak_nnz = 0;
    uint64_t merge_ops = 0;
    uint32_t units = 0;                // unit pivots of the sparse phase
    uint32_t residual_rows = 0, residual_cols = 0;
    uint32_t rank_Q = 0, r_K = 0;      // rank_Q = rank T^0 - b_0 (closed form)
    bool certified = false;
    AbelianGroupType tors1;            // certified only
    double gen_seconds = 0, elim_seconds = 0, residual_seconds = 0, seconds = 0;
    bool residual_refused = false;
};
// n even.  ell >= 5, or ell = 3 with nu_3(N) = 0 (ENGINE_CHECK; ell = 3 is [C, prop:descentNthree]).  K = 0: the
// budget (ell^K < 2^31).  A dense residual above residual_cap_bytes is not started: residual_refused, not certified.
SparseSmithRun RunSparseCokernelSmith(uint32_t N, uint32_t n, uint32_t ell, uint32_t K = 0,
                                      uint64_t residual_cap_bytes = 40ull << 30);
// (PAR-LN) the same route in its two stages: RunSparseCokernelSmith = FinishSparseCokernelSmith(EliminateSparseCokernel(...), cap).
//   EliminateSparseCokernel: the generators and the sparse unit-pivot elimination (single-threaded, host only); its result holds the
//     run's fields known after the elimination (units, nnz counters, residual_rows / _cols, gen_ / elim_seconds) and the residual rows
//     (every entry divisible by ell) with their columns already numbered densely.  The elimination's own structures are freed on return,
//     so only the residual survives to the second stage.
//   FinishSparseCokernelSmith: the ceiling rule (a dense residual above residual_cap_bytes is not started: residual_refused), the dense
//     residual's SmithLevelsEll, the certificate.  Same values, same order of operations as before the split.
struct SparseElimination {
    SparseSmithRun run;
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> residual;   // (dense column, value) per residual row, ascending columns
    uint64_t residual_nnz = 0;   // run.seconds holds the elimination stage's wall; FinishSparseCokernelSmith adds the residual's
};
SparseElimination EliminateSparseCokernel(uint32_t N, uint32_t n, uint32_t ell, uint32_t K = 0);
SparseSmithRun FinishSparseCokernelSmith(SparseElimination&& e, uint64_t residual_cap_bytes = 40ull << 30);
// (PAR-LN, rule 19) the predicted peak RSS of ONE EliminateSparseCokernel at (N, n): the generators, the pivot rows at their peak fill,
// the per-column work arrays and the monomial action (see DescentLevelN.cpp for the terms and the measured calibration).
uint64_t PredictSparseElimRss(uint32_t N, uint32_t n);
// The dense residual stage: the residual (rows x cols uint32) and SmithLevelsEll's working copy.
uint64_t PredictSparseResidualRss(uint32_t rows, uint32_t cols);
// The same generators stacked densely (the validation's oracle for the sparse elimination): Smith valuations.
std::vector<uint32_t> DenseCokernelSmith(uint32_t N, uint32_t n, uint32_t ell, uint32_t K);
// (D2) the same generators as rows (generators x D) over ctx, for any ctx in which the orders 2, 3 at the fixed points are
// units (ENGINE_CHECK): rows generating M_N^{<S>} + M_N^{<ST>} ([C, prop:descentN](7)(b), [C, prop:descentNthree](3)); n even.
engine::DenseBuf<uint32_t> FixedGeneratorRows(uint32_t N, uint32_t n, const engine::ModCtx<uint32_t>& ctx);

// ---- the assembly ----------------------------------------------------------------------
enum class EllRoute { Sparse, DirectBredon };
struct LevelNOptions {
    uint64_t rss_cap_bytes = 40ull << 30;   // per dense piece; above it the descent REFUSES (rule 19)
    EllRoute ell_route = EllRoute::Sparse;
    bool force_dense_three = false;          // run Iso^(3) even when nu_3 = 0 (arm (S))
    bool rank_from_betti = true;             // Iso^(ell) rank source (4)(c)(ii); false: (4)(c)(i) at the good prime
    uint32_t K_ell = 0;                      // 0: budget; smaller: the negative control
    bool skip_two = false, skip_three = false;   // measurement only (the result is then REFUSED)
    // VALIDATION ONLY (D1 audit m4, negative controls): perturb the expected summand count of prop:badprimesN and the
    // closed-form b_1 compared with the good prime; any non-zero value must REFUSE.
    int32_t test_count_offset = 0, test_b1_offset = 0;
};
struct LevelNResult {
    LevelNCounts c;
    uint32_t good_prime = 0;
    uint32_t good_h0 = 0, good_h1 = 0;       // the shipped SparseCokernel at the good prime
    std::vector<std::pair<uint32_t, uint32_t>> bad;   // (ell, s_ell) of prop:badprimesN
    std::vector<SparseSmithRun> sparse;      // per ell in bad (ell_route = Sparse)
    std::vector<BredonSmithRun> bredon;      // per ell in bad (ell_route = DirectBredon)
    bool three_shortcut = false;
    SparseSmithRun three_sparse;
    LocalPiece two, three;
    bool ran_two = false, ran_three = false;
    uint64_t rss_pred_two = 0, rss_pred_three = 0, rss_meas_two = 0, rss_meas_three = 0, rss_meas_ell = 0;
    std::vector<AbelianGroupType> H;         // q = 0 .. 5 (empty when refused)
    double t_good = 0, t_ell = 0, t_three = 0, t_two = 0, t_total = 0;
    // (PAR-LN; lever HECKE_DESCENT_PAR_LN, on by default) the concurrent sparse stage: ell_items = [the good prime (n even)] + the
    // ell >= 5 eliminations (sparse route) + [the ell = 3 shortcut's elimination (nu_3 = 0)], up to ell_workers at a time (joint RSS
    // rule); then the dense residuals one after another in the serial order.  ell_rss_pred_one = PredictSparseElimRss(N, n); _joint =
    // ell_workers x one; ell_rss_growth = the stage's sampled peak over the RSS before it.  t_ell_elim = the concurrent stage's wall,
    // t_ell_residual = the ell >= 5 residuals' wall.  With the lever on, t_good is the good prime's own wall inside the stage, t_ell
    // covers the ell >= 5 residuals only and t_three the shortcut's residual (the stage's wall is t_ell_elim); t_total is the call's wall.
    uint32_t ell_items = 0;
    uint32_t ell_workers = 1;
    bool ell_concurrent = false;
    uint64_t ell_rss_pred_one = 0, ell_rss_pred_joint = 0, ell_rss_growth = 0, ell_residual_pred_max = 0;
    double t_ell_elim = 0, t_ell_residual = 0;
    bool refused = false;
    std::string refusal;
};
LevelNResult AssembleDescentN(uint32_t N, uint32_t n, const LevelNOptions& opt = {});

// ---- the measured comparator for gate G-T: the free tree resolution at 2 with the Ind module -------------------------
// D0's TreeDifferentials (cochains 2D, 3D, 3D, ...) on ring2k mod 2^30, to degree 4, with rank_Q from the closed-form
// Betti numbers (rank d^0 = 2D - b_0, rank d^1 = 3D - b_1 - rank d^0, rank d^q = 3D - rank d^{q-1}); MEASURED ONLY,
// it feeds no result.  RSS: PredictTreeRss(D).
struct TreeTwoRun {
    bool ran = false, certified = false;
    std::vector<AbelianGroupType> tors;   // 2-primary tors H^q, q = 0 .. 4
    double seconds = 0;
    uint64_t rss_pred = 0, rss_meas = 0;
};
TreeTwoRun RunTreeTwoN(uint32_t N, uint32_t n, uint64_t rss_cap_bytes);

}  // namespace integral_descent

#endif
