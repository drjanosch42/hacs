// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_DESCENT_DESCENT_PAIR_H
#define INTEGRAL_DESCENT_DESCENT_PAIR_H

// ===========================================================================
// DESCENT milestone D4 (hecke-extensions slot 24; docs/implementation/slot24/DESCENT_plan.md v0.4 Sec. 4 "D4").  [C] =
// docs/latex/gl2_contract.tex (v1.23.0-pre.27, sec:descentpair, clause C-D7, refereed C-D7_referee.md), cited by label:
// def:descentpair, lem:pairlifting, prop:descentpair, eq:descentpairles, prop:descentpairhecke, rmk:descentpairscope and the
// CODE MAP after it, whose obligations (a)-(h) are this unit's specification; also prop:strictpair, prop:strictlift,
// prop:bdrycohom, prop:descentmaps (2)(g) (the precision rule) and (2)(b) (signs), prop:strictliftN (sec:descenthecke).
//
// WHAT IS COMPUTED.  For A = Gamma_0(N), V = Sym^n Z^2, trivial character, M = M_N = Ind_A^Gamma V, over Z:
//   * the three terms of the Borel--Serre pair sequence eq:descentpairles,
//         ... -> H^q_c --pi_*--> H^q(A; V) --res--> H^q(Gamma_d; M_N) --iota_*--> H^{q+1}_c -> ...,
//     H^q_c := H^{q-1}(Cone(rres)) (def:descentpair), q = 0 .. q_max (q_max >= 4, code map (g); cochains to q_max + 1, the
//     brutal truncation of prop:descentpair(6)), each as an AbelianGroupType glued from one primary part per prime
//     (prop:descentpair(2)(a)-(c), prop:cohglue);
//   * the three maps res, iota_* (here "j") and pi_* (here "proj") as maps of finitely generated abelian groups: the types of
//     kernel, image and cokernel (D2's InducedMap, prop:descentmaps(2)(g)'s precision rule, every Smith form certified);
//   * exactness at EVERY reported spot, AS SUBGROUPS (D2's ExactAt: the lattice quotient ker / im = 0), on each piece
//     (code map (e); prop:descentpair(2)(b): exactness of the glued sequence at a spot = exactness of the local ones);
//   * the boundary checked against the closed form from the cusp widths (code map (d), prop:descentpair(3), (4)), and the
//     statements of prop:descentpair(5) (H^0_c = 0, H^1_c torsion-free, rational ranks, 48 H^q_c = 0 for q >= 3, the 3-local
//     identity H^q_c = H^q for q >= 3, (5)(e): tors H^2_c (x) Z_(ell) = tors H_0(Gamma; M) (x) Z_(ell) at ell >= 5).
//
// THE PIECES (code map (a), (b); one computation per prime of S, nothing shared between primes):
//   (D5c) BY DEFAULT ell = 2 and ell = 3 are read on the TREE CUSP PAIR instead ([C] sec:treepair, Integral/Descent/DescentTreePair:
//             the cone of the axis map a^(ell)* on the tree cochains and the cusp model, RunPairLes verbatim); HECKE_DESCENT_TREE_PAIR=0
//             (or HECKE_DESCENT_TREE=0) is the read-only rollback to the dense pieces described next.  PairResult::route_two says which.
//   ell = 2:  the reduced complex of the DENSE Iso^(2) (D0/D1's SelectBlockJR / ClauseMatrixReduced, in J-coordinates on both
//             sides as D2's BuildRed) and its boundary sub-sum -- the boundary summands of the SAME O_2-model, a block
//             selection (no lift: rres is a restriction); at 2 the boundary is FREE (prop:descentpair(1)(d)), non-zero in every
//             degree; mod 2^30.  The pair at level N stays on dense Iso^(2) (plan D4; the tree cusp pair is D5c, not built);
//             PredictPairRss2 is checked BEFORE anything is built and the piece REFUSES above 0.9 x the ceiling.
//   ell = 3:  the same on Iso^(3), mod 3^19; its boundary is checked to be the circle model (code map (a): every boundary
//             column non-periodic, and its coboundary is the t = 0 block of the free model's, i.e. Hom_Gamma(C(Xbdry), M)).
//   ell >= 5: the Z[1/6]-piece.  Level 1: the SHIPPED E8 item-4b pair (code map (b); SL2ZBoundaryPair over Z/ell^K: C_eq of
//             the 7-vertex model and the circle model, the restriction in their module bases); level N: the Bredon pair with
//             fibre M_N, i.e. LocalModel(desc, ell) (every stabiliser order prime to ell: C_eq(M_N) in degrees <= 2) with its
//             boundary sub-sum (the circle model) -- a new object (the sparse cokernel route of D1 carries no cone).
//   S = {2, 3} cup {5 <= ell <= n + 1} cup {ell >= 5 : ell | N, n > 0} cup B(n, N) (n even); n odd: S = {2}
//   (prop:badprimesN(2); prop:descentpair(4): tors H^1(Gamma_d) has order prod_c n! w_c^n; (5)(c), (5)(e): H^q_c has no
//   ell >= 5 torsion for q != 2, and at q = 2 none for ell > n + 1; every kernel / image / cokernel of a map of the sequence
//   is a subquotient of a term).  Options add primes outside S (each must then give torsion-free terms).
//   Model (a), the free Iso over Z/ell^K itself, is a selectable route at 2 and 3 (PairRoute::IsoFree) -- the model-independence
//   arm (lem:pairlifting(3)).
//
// CERTIFICATION (code map (c), prop:descentpair(6)).  rank_Q of every differential: the bulk by the Betti recursion from the
// closed-form b_0, b_1 (prop:levelorbit(3), D1); the boundary by its closed form (b_0(d) = b_1(d) = c(N) for n even, 0 else;
// (6)(a)); the cone: D^{-1} = rank C^0 ((6)(b)), D^0 by the Betti recursion with dim H^1_c = b_1 ((5)(b)), D^q (q >= 1) with
// b_1(Cone) = dim H^2_c = b_0; AND, by default (prime_rank_check), every cone differential D^q, q >= 0, on the SAME O_ell-model
// reduced modulo a prime ell' >= 5, ell' > n + 1, ell' prime to q_ell (the code map's (c) for D^0 and D^q, q >= 1), computed on
// the full layout (rank of D P) and required EQUAL to the recursion (refused otherwise).  ell' in {2, 3}, or ell' <= n + 1 for
// D^0, is refused by rule (never used; the code map's "never at ell' in {2, 3}").  The E8 route (level 1, ell >= 5) certifies the
// identity of complexes with the Bredon model (E8Identity, prop:descentpair(1)(d); refused unless all()) inside the assembly on
// every call and runs the ell' ranks on that Bredon model (audit m3).  prime_rank_check = false makes the result REFUSED (a
// measurement, audit m4).  The
// a-priori exponents are consistency checks (refusal on violation): tors H^q(d) exponent <= v_ell(2) for q >= 2; the
// v_ell-order of tors H^1(d) = sum_c v_ell(n! w_c^n) (n even); H^1_c torsion-free; exponents of tors H^q_c <= v_ell(48), q >= 3;
// at ell >= 5 tors H^2_c = tors H_0(Gamma; M)_(ell) (H_0 = coker [rho(S) - 1 | rho(T) - 1], Smith over Z/ell^K).
// Precision: D2's BuildPres / InducedMap / ExactAt track the precision left at every step and REFUSE when it is exhausted
// (prop:descentmaps(2)(g), prop:descentpair(2)(c)).
//
// SIGNS (prop:descentpair(1)(c), referee m4).  proj = pi_* : [(c, c_d)] |-> [c] is MINUS the connecting map delta of the
// Ext-sequence of 0 -> St -> Z[P^1] -> Z -> 0; iota_* = j_St^*.  Immaterial for groups, kernels, images, exactness and T_p;
// nothing here compares a map entrywise with an independently computed delta.
//
// HECKE ON THE PAIR (code map (f); prop:descentpairhecke(1)-(3)).  ConeHecke builds, on the FREE model's full layout
// (LocalModel(desc, 0) = Iso, cell * D + component, over any ModCtx), the level-N operator
//   (T~^{(N)}_{Iso,p} phi)(x)_i = sum_t rho_n(delta_{i,t}) phi(tau_Iso(s_{j(i,t)} x))_{a(i,t)}            (eq:levelcorr)
// from the PAIR LIFT tau_Iso of Integral/Model/StrictLift (prop:strictlift(1): boundary step first, certified with its boundary
// items), r_t the transversal of Gamma_0(pN) \ Gamma_0(N) (hecke::level::Gamma0SubTransversal), and T~^{(N)}_{Isob,p}
// SEPARATELY, by the same formula on Isob's own layout from tau^d_Iso = the Isob-component of tau_Iso on the boundary cells (the
// values the lift's boundary step built; a term off Isob is dropped there and counted); the strict identity
// rres T~_Iso = T~_Isob rres is then checked ENTRYWISE between these two separately assembled operators (audit m2: it fails on any
// term of tau_Iso(Isob) off Isob and on any defect of either assembly; a planted off-boundary term with the closure test ignored
// is refused by it alone, ConeDatum::TestOffBoundaryTerm); every term of tau_Iso(Isob) is also checked to stay in Isob
// (bdry_closed); Phi_p = (T~_Iso[1], T~_Isob) on the cone.  For EVERY prime p !| N (owner scope addition 2026-10-25, beyond
// pre.27's code map (f), which said p not in {2, 3, 5} is refused -- DESCENT4_report.md lists the contract sentences to amend):
// the pair lift is PRODUCED on first request by the terminating construction (prop:descentpairhecke(1), prop:strictpair,
// prop:strictlift(1)), to degree 6, certified by StrictLift::certify() including its boundary items, and cached per p; the
// operator is REFUSED only when that certificate fails or a cap is hit (an exact product leaving int64, or the RSS prediction
// PredictPairLiftRss(p) above kPairLiftRssCap, read before building -- rule 19).  p | N is refused (U_p,
// [R-Q4]), and the bulk-only lift (ConeDatum::BulkOnly) is REFUSED -- never used on the cone (prop:descentpairhecke(5),
// rmk:conestrict).
//
// DEPENDENCIES: Integral/Descent/ (DescentAssembly, DescentLevelN, DescentMaps -- public API only, no edit), Integral/Model/
// (IsotropyModel, IsotropyModelEll, StrictLift), Hecke/ (HeckeLevelN, HeckeLevelEll's transversal, SL2ZBoundaryPair, the shipped
// pipeline for the E8 route).  No bignum, no settings key, no record key, no golden object.
// ===========================================================================

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "DescentAssembly.h"
#include "DescentMaps.h"
#include "../Model/StrictLift.h"

namespace integral_descent {
namespace pair {

using maps::Mat;
using maps::MapRes;
using maps::MapTypes;
using maps::Pres;
using maps::ZMod;

// ---- cusps and the closed form of the boundary (prop:descentpair(3), (4)) -----------------------------------------------
// The cusps of Gamma_0(N) as the <T>-orbits on Gamma_0(N)\Gamma = P^1(Z/N) (bottom rows, right action), from the shipped
// hecke::level::Gamma0LevelN table; the widths w_c (sum = m_N), ascending.
std::vector<uint32_t> CuspWidths(uint32_t N);
// F_{n,w} = tors of the coinvariants (Sym^n Z^2)_{T^w} at ell: the ell-exponents of the Smith form of rho_n(T^w) - 1 (an
// (n+1) x (n+1) integer matrix) over Z/ell^k; `certified` iff its rank mod ell^k is n (the free part Z x^n... is one line).
std::vector<uint32_t> CoinvariantTorsionExponents(uint32_t n, uint32_t w, uint32_t ell, uint32_t k, bool& certified);
// dim_F2 (Sym^n F_2^2)^{T^w}: n + 1 for w even, floor(n/2) + 1 for w odd (prop:descentpair(4)).
uint32_t Mod2InvariantDim(uint32_t n, uint32_t w);
// The ell-primary part plus free rank of H^q(Gamma_d; M_N) in closed form, from the widths alone; the F_{n,w} parts from
// CoinvariantTorsionExponents at precision k (ok = false if one was not certified).
AbelianGroupType BoundaryClosedForm(const std::vector<uint32_t>& widths, uint32_t n, uint32_t ell, uint32_t q, uint32_t k, bool& ok);
// v_ell(n! w^n).
uint32_t ValuationFactorialPower(uint32_t n, uint32_t w, uint32_t ell);

// ---- the rational data ---------------------------------------------------------------------------------------------------
struct PairBetti {
    uint32_t b0 = 0, b1 = 0;     // H^q(A; V (x) Q), closed form (D1's LevelNClosedForm)
    uint32_t c = 0;              // c(N)
    uint32_t bd0 = 0, bd1 = 0;   // H^0, H^1 of the boundary over Q: c(N) for n even, 0 for n odd
    uint32_t b(uint32_t q) const { return q == 0 ? b0 : (q == 1 ? b1 : 0u); }
    uint32_t bd(uint32_t q) const { return q == 0 ? bd0 : (q == 1 ? bd1 : 0u); }
    // dim H^q_c (prop:descentpair(5)(b) for M_N: dim H^1_c = b_1, dim H^2_c = b_0)
    uint32_t bc(int q) const { return q == 1 ? b1 : (q == 2 ? b0 : 0u); }
    // rational ranks of the maps on cohomology
    uint32_t rank_res(uint32_t q) const { return q == 0 ? b0 : (q == 1 ? bd1 - b0 : 0u); }
    uint32_t rank_j(uint32_t q) const { return q == 0 ? bd0 - b0 : (q == 1 ? b0 : 0u); }
    uint32_t rank_proj(uint32_t q) const { return q == 1 ? b1 - (bd0 - b0) : 0u; }
};
PairBetti PairBettiOf(uint32_t N, uint32_t n);

// ---- one local piece ------------------------------------------------------------------------------------------------------
// The pair (bulk, boundary, restriction) over Z/ell^k in some basis: d[q] (c[q+1] x c[q]), db[q] (cb[q+1] x cb[q]),
// r[q] (cb[q] x c[q]), q = 0 .. T (cochains), differentials q = 0 .. T - 1.
struct PairComplex {
    uint32_t ell = 0;
    ZMod Z;
    std::string route;
    uint32_t T = 0;
    std::vector<uint32_t> c, cb;
    std::vector<Mat> d, db, r;
    // (BuildModelPair with_embedding only) per degree q: emb[q] (full layout cells * D x c[q]) the inclusion of the reduced
    // cochains, column block of a cell = P_cell[:, J_cell]; coords[q] (c[q] x full) the J-coordinates on im P,
    // G_cell^{-1} pi_R per cell (coords emb = 1).
    std::vector<Mat> emb, coords;
    uint64_t bytes() const;
};
// The reduced pair of an O_ell-model L (Iso^(2), Iso^(3), or the Bredon model at ell >= 5) with fibre M over ctx = Z/ell^K,
// cochains to degree T: J-coordinates on both sides (D2's BuildRed), the boundary the sub-sum of the boundary cells.
PairComplex BuildModelPair(const LocalModel& L, const engine::ModCtx<uint32_t>& ctx, const CoefficientModule<uint32_t>& M,
                           uint32_t T, bool with_embedding = false);
// The free model Iso = LocalModel(Descriptor5(), 0), built once per process (the cells of StrictLift's IsotropyModel).
const LocalModel& FreeIsoModel();
// The free model Iso (LocalModel(desc, 0)) on the full layout over Z/ell^K (model (a) of def:descentpair).
PairComplex BuildFreePair(const engine::ModCtx<uint32_t>& ctx, const CoefficientModule<uint32_t>& M, uint32_t T);
// The shipped E8 item-4b pair at level 1 over Z/ell^K, ell >= 5 (SL2ZBoundaryPair), in the module bases of C_eq(K) and of the
// circle model; also returns the shipped module cone's invariant factors of H^q_c, q = 0 .. 2 (comparison only) and the
// identity-of-complexes certificate of arm (Q) (prop:descentpair(1)(d)): the comparison cmpI^* = eps^* (the selection of the
// representative simplices, on bulk and boundary) intertwines the shipped coboundaries and restriction with those of the
// Bredon model LocalModel(desc, ell) and is onto its cochains (im P).
struct E8Identity {
    bool bulk_chain_map = false, bdry_chain_map = false, res_compatible = false, onto = false;
    uint64_t entries_checked = 0;
    std::vector<std::vector<uint32_t>> shipped_Hc_invariant_factors;   // q = 0 .. 2 (shipped ModuleCone, over Z/ell^K)
    std::string why;
    bool all() const { return bulk_chain_map && bdry_chain_map && res_compatible && onto; }
};
PairComplex BuildE8Pair(uint32_t n, uint32_t ell, uint32_t K, uint32_t T, E8Identity* id = nullptr);

struct PieceReport {
    uint32_t ell = 0;
    std::string route;
    uint32_t K = 0, T = 0;
    std::vector<uint32_t> c, cb;              // cochain ranks (bulk, boundary)
    uint32_t exact_spots = 0, checks = 0, rank_checks = 0;
    uint32_t min_prec = 0;
    uint32_t rank_prime = 0;                  // 0: Betti recursion only
    uint64_t rss_pred = 0, rss_meas = 0, rss_base = 0, matrix_bytes = 0;
    double seconds = 0, build_seconds = 0, les_seconds = 0, rank_seconds = 0;
    bool ok = true;
    std::string why;
};

// The presentations and maps of one piece: H, Hd, Hc for q = 0 .. qm; res, j for q = 0 .. qm (j^q needs Hc^{q+1}: q < qm);
// proj for q = 0 .. qm.
struct PairLocal {
    uint32_t ell = 0, qm = 0;
    std::vector<Pres> H, Hd, Hc;
    std::vector<MapRes> res, j, proj;
};

// Runs the sequence on a pair complex: presentations (every Smith certified against the Betti data), maps, exactness at every
// reported spot, the a-priori exponent checks.  qm = the top reported degree (<= T - 1).  test_rank_offset (validation
// only) perturbs the claimed rational rank of res^1 (must refuse).
bool RunPairLes(const PairComplex& X, const PairBetti& bt, uint32_t qm, PairLocal& L, PieceReport& rep,
                int32_t test_rank_offset = 0);

// ---- the assembly ---------------------------------------------------------------------------------------------------------
enum class PairRoute { IsoEll, IsoFree };
struct PairOptions {
    uint32_t q_max = 4;                       // >= 4 (code map (g); below: throws); <= 5 (the descriptor's degree 6; above: REFUSED)
    uint64_t rss_cap_bytes = 40ull << 30;
    PairRoute route = PairRoute::IsoEll;      // at ell in {2, 3}
    bool level1_e8 = true;                    // level 1, ell >= 5: the shipped E8 pair (code map (b)); false: the Bredon model
    bool prime_rank_check = true;             // code map (c): every cone differential also mod ell' (refused if unequal); false:
                                              // a measurement only, the result is then REFUSED (audit m4)
    uint32_t rank_prime = 0;                  // 0: the least prime >= 10007 (> n + 1)
    std::vector<uint32_t> extra_primes;       // primes outside S to run as well (their terms must be torsion-free)
    std::vector<uint32_t> only_primes;        // non-empty: run only these primes of S (measurement; the result is then REFUSED)
    // VALIDATION ONLY (negative controls; each must REFUSE): perturb one entry of the restriction in every piece; claim a wrong
    // rank of res^1; run the 2-adic piece at a small K; skip the ell' rule (ell' <= n + 1 or in {2, 3} for D^0).
    bool test_break_res = false;
    int32_t test_rank_offset = 0;
    uint32_t test_K_two = 0;
    bool test_skip_prime_rule = false;
};
struct PairResult {
    uint32_t N = 1, n = 0, q_max = 4, D = 0;
    std::vector<uint32_t> widths;
    PairBetti bt;
    std::vector<AbelianGroupType> H, Hd, Hc;  // q = 0 .. q_max (empty when refused)
    std::vector<MapTypes> res, j, proj;       // res, proj: 0 .. q_max; j: 0 .. q_max - 1
    std::vector<PieceReport> pieces;
    std::vector<uint32_t> primes;             // S (+ extra)
    uint32_t exact_spots = 0, checks = 0;
    std::vector<std::string> closed_form;     // per q: "H^q(d) = ... (closed form ...)"
    uint64_t rss_pred = 0;                    // the whole call: max(the ell <= 3 pieces, the ell >= 5 stage's ell_rss_pred_joint) (PAR-ASM F6)
    // (PAR-ASM (c)) the ell >= 5 pieces: how many ran at once, one piece's and the joint RSS prediction, the measured joint growth
    uint32_t ell_workers = 1;
    uint64_t ell_rss_pred_one = 0, ell_rss_pred_joint = 0, ell_rss_growth = 0;
    double t_total = 0;
    std::string route_two;                    // the route of the ell in {2, 3} pieces: "tree" (D5c, the default), "Iso2" (dense), "IsoFree"
    bool refused = false;
    std::string refusal;
};
PairResult AssembleDescentPair(uint32_t N, uint32_t n, const PairOptions& opt = {});

// Predicted peak RSS (bytes) of the dense pair piece at ell in {2, 3} with module dimension D, cochains to degree T: the
// structural count (the reduced d, d_d, the cone's D^j, three c x c of the largest Smith form, the rho memo) times the
// calibrated factor plus 256 MiB (DESCENT4_predictions.md).
uint64_t PairRssStructural(uint32_t ell, uint32_t D, uint32_t T);
uint64_t PredictPairRss(uint32_t ell, uint32_t D, uint32_t T);

// ---- Hecke on the pair (code map (f)) ---------------------------------------------------------------------------------------
// TestPerturbedLift: a perturbed lift, certified (must fail its certificate).  TestOffBoundaryTerm: one term off Isob planted in
// tau_Iso(s_1 e_b) of the first boundary cell b in degree 1, the certificate and the closure test bdry_closed IGNORED -- the
// strict identity alone must refuse it (the audit's F7 made detectable).  TestHomotopyIntoIsob / TestHomotopyIntoBulk (C-D7p
// referee m4): tau + dK + Kd with K(s_1 e_v) = e_e, v the first t = 0 boundary cell of degree 0, e the first t = 0 degree-1 cell in
// Isob / outside Isob -- (T1)-(T3) hold, (S) / (B) must refuse it.  All validation only.
enum class ConeDatum { PairLift, BulkOnly, TestPerturbedLift, TestOffBoundaryTerm, TestHomotopyIntoIsob, TestHomotopyIntoBulk };
// THE REFUSAL CODES of T_p on the pair (C-D7p referee M1).  Every refusal carries exactly one, printed as "REFUSED [<name>] ..."; an
// exception that is none of these is a genuine error and PROPAGATES (a validation failure, never a refusal):
//   NotPrime            p is not a prime;
//   PDividesN           p | N (U_p, [R-Q4]);
//   BulkOnly            ConeDatum::BulkOnly asked for (never on the cone);
//   DegreeRange         q_top > kPairLiftTop = 6 (the lift is built to degree 6; the pair is served for q_max <= 5);
//   RssCeiling          PredictPairLiftRss(p) > kPairLiftRssCap, before anything is built (rule 19);
//   IntegerCap          hecke::HeckeOverflow while the lift is built or the operator assembled: an exact product outside int64, or a
//                       tau-table chain coefficient outside int32 -- NOT its subclass hecke::HeckeInternalError (a violated
//                       precondition or internal check), which is a genuine error and propagates;
//   DomainCap           std::overflow_error: p (or p^r) outside the coset tables' uint32 domain;
//   InputTableCertificate  an input table's mandatory self-certification failed (HeckeTauTable: "certification FAILED", obl:tau);
//   LiftCertificate     StrictLift::certify() fails ((T1), (T2), (T3), (S), (B)-preserved, (B)-eps);
//   IsobClosure         a term of tau_Iso(Isob) off Isob on the operator's layout;
//   StrictIdentity      rres T~_Iso != T~_Isob rres entrywise (the two separately assembled operators).
enum class PairRefusal { None, NotPrime, PDividesN, BulkOnly, DegreeRange, RssCeiling, IntegerCap, DomainCap, InputTableCertificate,
                         LiftCertificate, IsobClosure, StrictIdentity };
const char* PairRefusalName(PairRefusal r);
// IntegerCap / DomainCap / InputTableCertificate for the exceptions above, None for every other (a genuine error, rethrown).
PairRefusal ClassifyPairException(const std::exception& e);
constexpr uint32_t kPairLiftTop = 6;
// The cost of building and certifying the pair lift at p (built once per process, on first request).
struct PairLiftCost {
    uint32_t p = 0;
    double build_seconds = 0, certify_seconds = 0;
    uint64_t rss_pred = 0, rss_growth = 0, terms = 0;
    PairRefusal code = PairRefusal::None;   // None iff ok
    int64_t max_abs_coefficient = 0, max_abs_entry = 0;
    bool ok = false;
    std::string why;
};
// Callers at the same p wait while that p is built and certified (one construction per p); other p are not blocked.
const PairLiftCost& PairLiftCostOf(uint32_t p);
// Rule 19 for the pair lift (audit m5): an upper bound of the RSS growth of building and certifying the lift at p, from the measured
// stored-term count (~ p^{1.4}, DESCENT4_report.md Sec. 4); the lift is refused before building above kPairLiftRssCap.
uint64_t PredictPairLiftRss(uint32_t p);
constexpr uint64_t kPairLiftRssCap = 4ull << 30;
struct ConeHecke {
    uint32_t N = 1, n = 0, p = 0, q_top = 0, D = 0;
    std::vector<engine::DenseBuf<uint32_t>> bulk, bdry;   // T~_Iso, T~_Isob per degree q = 0 .. q_top (full layout)
    bool strict_identity = false;      // rres T~_Iso = T~_Isob rres entrywise, every degree (two separately assembled operators)
    bool bdry_closed = false;          // every term of tau_Iso(s_j x), x in Isob, lies in Isob
    uint64_t terms = 0, bdry_terms = 0, bdry_dropped = 0, entries_checked = 0;
    bool refused = false;
    PairRefusal code = PairRefusal::None;
    std::string refusal;
};
// The level-N operator on the free model's pair over ctx (cochains to q_top <= 6).  Refuses p not prime, p | N, a pair lift whose
// certificate fails or whose construction hit a cap, ConeDatum::BulkOnly (never on the cone), and a failed strict identity: exactly
// the PairRefusal codes above.
ConeHecke BuildConeHecke(uint32_t N, uint32_t n, uint32_t p, const engine::ModCtx<uint32_t>& ctx, uint32_t q_top,
                         ConeDatum datum = ConeDatum::PairLift);
// The free model's pair at level N on its full layout over ctx (the same cells as ConeHecke): d, db, r.
PairComplex BuildFreePairN(uint32_t N, uint32_t n, const engine::ModCtx<uint32_t>& ctx, uint32_t T);
// The certificate of the pair lift at a prime p (StrictLift::certify(): (T1)-(T3), (S), and the boundary items (B) --
// tau_Iso(Isob) in Isob and eps tau^d_Iso = tau_bd eps: the plan's arm (S), rmk:descentpairscope(iii)).
const integral_model::StrictLiftCertificate& PairLiftCertificate(uint32_t p);
const integral_model::StrictLift& PairLift(uint32_t p);

// ---- small helpers shared with the validation -------------------------------------------------------------------------------
Mat FromDenseBuf(const engine::DenseBuf<uint32_t>& D);
Mat MatMulZ(const ZMod& Z, const Mat& A, const Mat& B);
// The matrix of the map induced by the cochain map f on the presentations X -> Y (Y.coord f X.gens^T, at the smaller precision).
Mat InducedMatrix(const Pres& X, const Pres& Y, const Mat& f, ZMod& Zout);
// The cone differential D^j : Cone^j = C^{j+1} (+) C_d^j -> Cone^{j+1}, (c, c_d) |-> (-d c, d_d c_d + r c) (eq:conediff).
Mat ConeDifferential(const PairComplex& X, int j);
uint32_t ConeRank(const PairComplex& X, int j);   // the cochain rank of Cone^j

}  // namespace pair
}  // namespace integral_descent

#endif
