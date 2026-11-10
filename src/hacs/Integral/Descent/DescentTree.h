// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_DESCENT_DESCENT_TREE_H
#define INTEGRAL_DESCENT_DESCENT_TREE_H

// ===========================================================================
// DESCENT milestone D5 (hecke-extensions slot 24; docs/implementation/slot24/DESCENT_plan.md v0.4 Sec. 4 "D5", the owner's
// ruling at gate G-T of 2026-10-25).  [C] = docs/latex/gl2_contract.tex, cited by label: def:isomodeltree,
// prop:isomodeltree (C-D6), prop:certifiedeell, prop:descent, prop:descentN; [C] v1.23.0-pre.23.
//
// THE TREE ROUTE FOR THE ell-LOCAL BULK PIECES OF D0 (level 1) AND D1 (level N).  [C, prop:isomodeltree](5): the ell-local
// pieces of [C, prop:descentN](2) (and of prop:descent(2) at level 1) may be read on the tree models at ell = 2 and 3,
// with no change to the glueing datum.  RunTreeLocalPiece reads the ell-primary torsion of H^q(Gamma; M), q = 1 ..
// q_top, on C^._{T,ell}(M_{O_ell}) (x) Z/ell^K, in the closed form of (2) (Integral/Model/TreeCochains.h), by the
// certified-exponent rule of [C, prop:certifiedeell](2)-(6), which holds VERBATIM on this complex ((1)(d)):
//   * Smith valuations over Z/ell^K on the native ring engines (ring2k mod 2^30, ring3k mod 3^19); certified iff the
//     number of non-zero diagonal entries s equals rank_Q d^q;
//   * rank_Q d^q from exactly the two sources (1)(d) admits: (i) the reduction mod a qualifying ell' (prime to q_ell) of
//     THIS complex (TreeLocalForm over F_ell'; at q = 0 the (4)(b) comparison rank P_0 - rank delta^0 = b_0), or (ii)
//     the Betti recursion c_q - b_q - rank d^{q-1} from certified Betti numbers.  Whichever is used, the other (when
//     computed) is an ENGINE_CHECKed cross-check.  The rank of a differential of Iso^(ell), of Iso or of the free tree is
//     never used for it;
//   * the a-priori bound sigma <= v_ell(24) for q >= 1 (ENGINE_CHECKed consistency check of the read-out);
//   * the stopping rule: at 2 a differential not certified at K = 30 is rerun at W = 61 through ModCtx<uint64_t> (Sym^n
//     only, the code map's W <= 61); at 3, K = 19 is ring3k's cap; uncertified = a REFUSAL by the caller.
// Periodicity ((2)(d)): at 2, d^{q+2} = d^q for q >= 1, at 3 for q >= 2, as MATRICES on the same cochains, so a run to
// q_top reads H^4, H^5 off d^3 = d^1, d^4 = d^2 (2) and d^4 = d^2, d^5 = d^3 (3) -- "read on the same cochains", not the
// cup-product identification (which (2)(d) does not assert).  explicit_periodic builds every d^q through TreeLocalForm
// (whose closed form selects the branch by the parity of q, so it re-runs the copy's branch: not a test of (2)(d); the
// independent test is the validation's arm (E), TreeLocalForm against the chain model TreeModelChain, D5 audit m3).
//
// THE SWITCH.  HECKE_DESCENT_TREE (read once): unset or 1 = the tree route (default); 0 = the dense Iso^(ell) route of
// D0/D1 (rollback, read-only, value-preserving: both routes compute the same groups, [C, prop:isomodeltree](1)(c)); any
// other value warns and keeps the default.  ScopedDescentTree overrides it process-wide (validation use).  Where the
// tree replaces a dense piece: the 2-adic piece at level 1 and level N, and the 3-adic dense piece (Iso^(3); level 1, and
// level N when nu_3(N) > 0), and the level-1 ell >= 5 piece (the Bredon tree in place of the direct Smith of C_eq) -- each
// switched on measurement (DESCENT5_report.md).  The nu_3(N) = 0 shortcut at 3 ([C, prop:descentNthree]) and the level-N ell >= 5
// route are unchanged (sparse, cheaper than any dense tree).
//
// Bulk only: nothing here concerns Xbdry, Isob, the pair or the cusp (C-D8, D5c).  No settings key, no record key, no
// golden object.
// ===========================================================================

#include <cstdint>
#include <string>
#include <vector>

#include "DescentAssembly.h"
#include "../Model/TreeCochains.h"
#include "../../../ModArithmetic/EnvLever.h"

namespace integral_descent {

// HECKE_DESCENT_TREE (read once; process-wide).  It never changes a value.
bool DescentTreeEnabled();
std::string DescentTreeSource();   // "default" | "env" | "scoped"
const engine::lever::EnvLever& DescentTreeLever();   // (S24 CL-5) the EnvLever behind it (EnvLever.h)
struct ScopedDescentTree : engine::lever::EnvLever::Scoped {
    explicit ScopedDescentTree(bool on) : Scoped(DescentTreeLever(), on) {}
};

// Predicted peak RSS (bytes) of RunTreeLocalPiece on a module of dim D with blocks of size d (Sym^n: d = D; Ind: d = n + 1):
// one reduced differential (uint32_t) and its Smith copy, the memoised block-monomial rho's, + 0.2 GiB.
uint64_t PredictTreeLocalRss(uint32_t ell, uint32_t D, uint32_t d);

// Predicted peak RSS GROWTH (bytes) of the whole level-1 descent AssembleDescent(n) on the tree route (D5 audit m2; DESCENT5_predictions.md
// amendment 2, filed before the cells that test it): the shipped good-prime run (ShippedFieldDims on C_7 at F_ell', dense in D = n + 1,
// 2304 D^2 bytes = 576 words per D^2: calibrated on n = 100 .. 700 at <= 1 940 bytes per D^2, >= 15 % headroom) plus the largest
// tree piece (PredictTreeLocalRss(2, D, D), incl. its 0.2 GiB base).  The pieces run one after the other; the sum is the margin.
// (PAR-ASM (b)) plus the joint prediction of the concurrent ell >= 5 primes (JointWorkers x PredictTreeBredonRss(D)) when they run concurrently.
uint64_t PredictDescentLevel1Rss(uint32_t n);

// ell in {2, 3}; ctx = Z/ell^K at the model's ell.  h = rational Betti numbers b_q (size >= q_top + 1).  rank_prime = 0:
// rank_Q from the Betti recursion (source (ii)) alone; else source (i) on THIS complex over F_rank_prime with M_rank (the
// same module over F_rank_prime), cross-checked against (ii).  sym_n >= 0: M is Sym^{sym_n} (enables the W = 61 route at
// 2); force_wide: that route only.  The result has the shape of RunLocalPiece's (differentials d^0 .. d^{q_top-1}, tors
// H^1 .. H^{q_top}); route = "tree".
LocalPiece RunTreeLocalPiece(uint32_t ell, const integral_model::TreeModule<uint32_t>& M, const engine::ModCtx<uint32_t>& ctx,
                             uint32_t q_top, const std::vector<uint32_t>& h, uint32_t rank_prime,
                             const integral_model::TreeModule<uint32_t>* M_rank, int sym_n, bool force_wide = false,
                             bool explicit_periodic = false);
// Sym^n at level 1 (ctx built here; K = 0: the rung's full width).
LocalPiece RunTreeLocalPieceSym(uint32_t ell, uint32_t n, uint32_t q_top, const std::vector<uint32_t>& h, uint32_t rank_prime,
                                uint32_t K = 0, bool force_wide = false, bool explicit_periodic = false);
// Ind_{Gamma_0(N)}^Gamma Sym^n (rank_prime = 0: Betti; else source (i) at that prime).
LocalPiece RunTreeLocalPieceN(uint32_t ell, uint32_t N, uint32_t n, uint32_t q_top, const std::vector<uint32_t>& h,
                              uint32_t rank_prime = 0, uint32_t K = 0, bool explicit_periodic = false);

// The Bredon tree at ell >= 5 ([C, prop:isomodeltree](2)(c)) on Sym^n: the ell-primary torsion of H^1 by one Smith of d^0
// over Z/ell^K, rank_Q d^0 = c_0 - b_0.  At level 1 it replaces the direct Smith of C_eq (measured useful); at level N the sparse
// route of [C, prop:descentN](7)(b) stays (dense tree Bredon is not cheaper there).
BredonSmithRun RunTreeBredonSym(uint32_t n, uint32_t ell, uint32_t K, uint32_t b0);
// The level-1 ell >= 5 piece of AssembleDescent on the Bredon tree (the default there unless HECKE_DESCENT_TREE=0, switched on the
// measurement of DESCENT5_report.md Sec. 4.4): one Smith of d^0 at the budget K; the torsion lies in H^1 only because the tree's
// cochains vanish in degrees >= 2 ((2)(c)).
BadPrimeRun RunTreeBredonLevel1(uint32_t n, uint32_t ell, uint32_t count);

}  // namespace integral_descent

#endif
