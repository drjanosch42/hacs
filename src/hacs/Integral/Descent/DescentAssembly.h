// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_DESCENT_DESCENT_ASSEMBLY_H
#define INTEGRAL_DESCENT_DESCENT_ASSEMBLY_H

// ===========================================================================
// DESCENT milestone D0 (hecke-extensions slot 24; docs/implementation/slot24/DESCENT_plan.md v0.2 Sec. 4 "D0",
// Sec. 2.1-2.4, Sec. 3; audit DESCENT_plan_audit.md M2, M3, M6, M7).  [C] = docs/latex/gl2_contract.tex, cited by
// label.
//
// WHAT IS COMPUTED.  The level-1 integral groups H^q(SL_2(Z); Sym^n Z^2), q = 0 .. 5, as an AbelianGroupType per q,
// glued by the fibre product of [C, prop:descent](2) (equivalently [C, prop:cohglue]: a finitely generated abelian
// group is its free rank plus the direct sum of its ell-primary parts, each read at ONE prime, no extension problem
// between primes) from four inputs and nothing else:
//
//   (1) the good-prime rank: dim H^q(C_7 (x) F_ell') of the SHIPPED 7-vertex pipeline at one prime ell' >= 5,
//       ell' > n + 1 ([C, prop:certifiede](4)); C_7 computes group cohomology once 6 is inverted ([C, prop:bredon]),
//       so these are the ranks of H^q(Gamma; Sym^n Q).  H^q (x) Q = 0 for q >= 2 (vcd 1), checked on the object.
//   (2) the ell-primary torsion at each bad prime ell >= 5 of the closed-form list [C, prop:badprimes](1) (the
//       "Dickson primes" with a non-zero count; zero-count primes carry no torsion BY THEOREM, not by omission), from a
//       certified Z/ell^K run of the shipped C_7 pipeline (the E8 K-mode, ladic_type_certified(); [C, prop:certifiede]
//       on C_eq(Z) at ell >= 5, scope item (5)).  Its torsion lies in H^1 only.
//   (3) the 3-primary torsion from Iso^(3) ([C, def:isomodelell], [C, prop:ellthreeclosed]: the Bredon cochains plus
//       ONE periodic column at v_4) with coefficients mod 3^19, on the native ring engine::ring3k (Smith by valuation
//       levels, one run, no K search);
//   (4) the 2-primary torsion from Iso^(2) with coefficients mod 2^30, on engine::ring2k (rung uint32_t; a differential
//       not certified there is rerun at W = 61 through ModCtx<uint64_t>, the stopping rule of [C, prop:certifiedeell](6)),
//       built to degree 4 in production (H^4 read directly from d^3, H^5 := H^3 by [C, lem:bulkperiodic],
//       q >= 2), to degree 6 in the check arm (P) at n <= 60 so that the periodicity is tested, not assumed (M7).
//
// THE CERTIFIED-EXPONENT RULE ON Iso^(ell) -- the clause C-D0, [C, prop:certifiedeell] (v1.23.0-pre.12, referee fixes at
// pre.13).  Iso^(ell) is one complex over O_ell = Z[1/q_ell], q_ell = lcm |Q_sigma| (q_2 = 3, |Q| in {1, 3}; q_3 = 4,
// |Q| in {2, 4}, Q_{v6} = <S>), and the code builds its base changes to Z/ell^K and F_ell'; every denominator is checked at
// build time to be a product of powers of the |Q_sigma| involved, dividing a power of q_ell (CheckDenominators; the code
// map).  Per differential: the Smith valuations sigma_1 <= .. <= sigma_s < K over Z/ell^K, read on the native ring
// engines (the DenseBuf -> ring2k / ring3k hand-off of the code map); certified iff s == rank_Q d^q
// ([C, prop:certifiedeell](6)); then tors H^{q+1}_(ell) = (+)_{sigma > 0} Z/ell^sigma.  rank_Q d^q comes from source
// (4)(c)(i): rank over F_ell' of THIS LocalModel(desc, ell) reduced mod ell' (the clause's delta^q = P delta P on the full
// layout, the same module over F_ell'; ell' = the good prime, which qualifies by (4)(a) for q >= 1 and by (4)(b) for
// q = 0); source (4)(c)(ii) (c_q - b_q - rank d^{q-1}, b_q from input (1)) is an ENGINE_CHECKed cross-check.  The model's
// ell is tied to the context it is reduced into (a mismatch is refused).  The stopping rule: at 2 a differential not
// certified at K = 30 is rerun at W = 61 through ModCtx<uint64_t> (the code map's W <= 61); at 3, K = 19 is ring3k's cap
// and an uncertified differential is a REFUSAL.  By the a-priori bound a_t <= v_ell(24) for q >= 1 (an ENGINE_CHECKed
// consistency check of the read-out) the caps can bind only at d^0.
//
// THE REDUCED COMPLEX ([C, prop:certifiedeell](5)).  The cochains are C^q = (+)_cells M^{Q_cell} = im P_q,
// P_q = diag rho(e_cell).  Per cell the image of P_cell is a free direct summand over Z_(ell); a basis is the columns J of
// P_cell whose reductions mod ell are a column rank profile (Nakayama), and the coordinate projection pi_R onto rows R (a
// row rank profile of P_cell[:, J] mod ell) is an isomorphism on im P_cell.  The matrix handed to the ring engine is
// A = pi_R P_{q+1} delta_code P_q[:, J], the clause's delta^q in these coordinates (the outer projectors are applied
// explicitly, whatever normalisation LocalModel's stored coefficients carry), so the elimination runs at exactly the
// cochain ranks of the plan's Sec. 3.1 (Iso^(2): 4.33, 13.33, 18.33, ... d; Iso^(3): 4.5, 10, 6, 1, ... d).
//
// THE MEASURED COMPARATORS (DESCENT_plan Sec. 2.5; MEASURED ONLY -- none of them feeds a DescentResult):
//   (a)  the dense free model C_I: integral_model::AssembleFromModel (IntegralGroups) on IsotropyModel(q_max = 3), i.e.
//        to degree 4 on equal terms, at 2, 3 and every Dickson prime plus the good prime (its linear K search on a
//        general ModCtx, as built);
//   (b)  column collapse: IS the descent ((b) computes on exactly the complexes of the descent); reported as an
//        identity, not run;
//   (c)  the free tree resolution of [C, rmk:treeconeres] in closed form (cochains V^2 -> V^3 -> V^3 -> ...; the
//        cone of P^{<-I>} -> P^{<S>} (+) P^{<U>}), at every prime: ring2k at 2, ring3k at 3, Z/ell^K (linear K) at the
//        Dickson primes, F_ell' at the good prime; any CoefficientModule (Sym^n, or Ind_{Gamma_0(N)} for the level-N
//        probe);
//   (c') the tree Bredon complex M^S (+) M^U -> M^{-I} ([C, prop:bredon](2); the object of HeckeBredonTreeRing's (T)
//        arm) at the good prime and at the Dickson primes -- the comparator of the Z[1/6] piece.
//
// DEPENDENCIES: Integral/Model/ (IntegralGroups, IsotropyModel, IsotropyModelEll), the shipped 7-vertex pipeline
// (EquivariantCohomologyOnK, LocalSystemSL2Z, Hecke/CoefficientRing for the K-mode), Hecke/HeckeLevelN (MonomialOf, the
// level-N probe only), ModArithmetic/ (DenseElim, Binary/Ring2k, Binary/Ring3k).  No bignum, no settings key, no
// record key, no golden object.
// ===========================================================================

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "../../../ModArithmetic/EnvLever.h"
#include "../Model/IntegralGroups.h"
#include "../Model/IsotropyModel.h"
#include "../Model/IsotropyModelEll.h"

namespace integral_descent {

using integral_model::AbelianGroupType;
using integral_model::CoefficientModule;
using integral_model::LocalModel;

bool IsPrime(uint32_t v);

// The process's current RSS in bytes (0 if /proc is unreadable).  Nothing here resets the kernel's
// high-water mark (D0 audit m4: /proc/self/clear_refs also resets what ru_maxrss and /usr/bin/time report).
uint64_t CurrentRssBytes();
// Per-piece peak RSS by SAMPLING: a background thread reads VmRSS every 5 ms between construction and stop(); stop()
// returns the maximum seen (absolute process RSS, the retained heap of earlier work included).
class RssSampler {
public:
    RssSampler();
    ~RssSampler();
    uint64_t stop();
    RssSampler(const RssSampler&) = delete;
    RssSampler& operator=(const RssSampler&) = delete;
private:
    struct Impl;
    Impl* impl_;
};
// Process CPU seconds (user + system, all threads; getrusage RUSAGE_SELF).
double ProcessCpuSeconds();

// The shared descriptors (built once per process): IsotropyModel(5) (C_I to degree 6), IsotropyModel(3) (to degree 4),
// and the ell-local models Iso^(2), Iso^(3) on the degree-6 descriptor.
const integral_model::IsotropyModel& Descriptor5();
const integral_model::IsotropyModel& Descriptor3();
const LocalModel& Iso2();
const LocalModel& Iso3();

// ---- (1) the good prime ------------------------------------------------------------
// dim H^q(C_7 (x) F_ell), q = 0..2, of the shipped pipeline (any prime ell; at ell in {2, 3} it is the Bredon object).
std::vector<uint32_t> ShippedFieldDims(uint32_t n, uint32_t ell);
uint32_t DefaultGoodPrime(uint32_t n);   // 10007 for n <= 10000
// (slot-24 PAR-L1) AssembleDescent's good-prime step at EVEN n: by default the shipped SPARSE cokernel of Gamma_0(1) over
// F_good (the record route's step, AssembleDescentN (1); n = 2000: 22.8 s serial against 296 s for the serial C_7, 55 s on 16 threads), its
// h_0 / h_1 checked against the closed form b_0 / b_1 (LevelNClosedForm(1, n)) and h_2 := 0 (vcd 1; prop:descentN(7)(b)).
// Odd n keeps ShippedFieldDims.  Read-only rollback lever HECKE_DESCENT_GOOD_SPARSE: exactly 0 restores ShippedFieldDims
// (the C_7 pipeline on the caller's consumer pool) verbatim; 1 or unset = sparse; any other value warns and keeps sparse.
// Read once per process; ScopedDescentGoodSparse overrides it (validation use).  Cannot change a value: both routes compute
// dim H^q(Gamma; Sym^n F_good), q = 0, 1, 2, and the closed-form check throws on any disagreement.
// The good-prime dims by the route in force (sparse / dense: {h_0, h_1, 0}); *sparse (optional) reports whether the route is off
// the shipped C_7.  (slot-24 PAR-GP) Since PAR-GP the default even-n route reads the same three numbers as DENSE ranks (below).
std::vector<uint32_t> GoodPrimeFieldDims(uint32_t n, uint32_t good, bool* sparse = nullptr);
bool DescentGoodSparseEnabled();
std::string DescentGoodSparseSource();   // "default" | "env" | "scoped"
const engine::lever::EnvLever& DescentGoodSparseLever();   // (S24 CL-5) the EnvLever behind it (EnvLever.h)
struct ScopedDescentGoodSparse : engine::lever::EnvLever::Scoped {
    explicit ScopedDescentGoodSparse(bool on) : Scoped(DescentGoodSparseLever(), on) {}
};
// (slot-24 PAR-GP) The sparse route's three numbers as DENSE ranks.  At level 1 (one coset, a fixed point of S and of ST) every
// generator of Rel = M^<S> + M^<ST> is a column of an averaging idempotent e_S = 1 + rho(S), e_ST = 1 + rho(ST) + rho(ST)^2
// (prop:levelorbit (1); the scalar 1/|gamma| does not change a span), so the sparse cokernel's dim_S, dim_ST and dim Rel are
// rank e_S, rank e_ST and rank [e_S | e_ST]: unique numbers, so no route can move a value.  They are read off by the engine's
// blocked PLU (engine::elim::rref of e_S^T and e_ST^T, then e_ST's RREF rows reduced against e_S's (eliminate_against, row tiles)
// and the column rank profile of the residual), threaded over the ONE descent pool (maps::DescentSmithDefaultThreads():
// ENGINE_THREADS; 1 on a PAR-ASM worker, engine::parworker::active()) through the elimination's ParallelFor hook (disjoint rows,
// exact residues: ENGINE_API.md Sec. 8 / D4).  threads = 0: that budget; 1: serial.
struct GoodPrimeRanks {
    uint32_t dim_S = 0, dim_ST = 0, dim_rel = 0;
    uint32_t threads = 1;           // the descent-pool thread count (the eliminate_against tiles; rref's hook unless caller_hook)
    bool caller_hook = false;       // (PAR-GP audit F4) a caller's elimination hook was in force: rref ran on it, not on `threads`
    double seconds = 0;
};
GoodPrimeRanks GoodPrimeDenseRanks(uint32_t n, uint32_t good, uint32_t threads = 0);
uint64_t GoodPrimeDenseHookRegions();   // regions the dense route's elimination hook opened, process total (diagnostic)
enum class GoodPrimeRoute : uint8_t { Shipped = 0, Sparse = 1, Dense = 2 };
const char* GoodPrimeRouteName(GoodPrimeRoute r);   // "shipped" / "sparse" / "dense"
// The route GoodPrimeFieldDims takes at n (the levers below, read once per call).
GoodPrimeRoute GoodPrimeRouteFor(uint32_t n);
// The good-prime dims with the route reported (GoodPrimeFieldDims is this with route ignored).  Default at even n: Dense; read-only
// rollback lever HECKE_DESCENT_GOOD_DENSE: exactly 0 restores PAR-L1's sparse cokernel verbatim; 1 or unset = dense; any other value
// warns once and keeps dense.  HECKE_DESCENT_GOOD_SPARSE=0 (the shipped C_7) takes precedence over both.  Every route checks
// h_0 / h_1 against the closed form (the shipped C_7 route through AssembleDescent's own checks), so a disagreement throws.
std::vector<uint32_t> GoodPrimeFieldDimsRoute(uint32_t n, uint32_t good, GoodPrimeRoute* route);
bool DescentGoodDenseEnabled();
std::string DescentGoodDenseSource();   // "default" | "env" | "env-invalid(default kept)" | "scoped"
const engine::lever::EnvLever& DescentGoodDenseLever();   // (S24 CL-5) the EnvLever behind it (EnvLever.h)
struct ScopedDescentGoodDense : engine::lever::EnvLever::Scoped {
    explicit ScopedDescentGoodDense(bool on) : Scoped(DescentGoodDenseLever(), on) {}
};

// ---- (2) the bad primes ell >= 5: the shipped C_7 K-mode ------------------------------
struct BadPrimeRun {
    uint32_t ell = 0;
    uint32_t count = 0;        // [C, prop:badprimes](1): the number of cyclic summands predicted
    uint32_t K = 0;            // the certified K (0 if none within budget)
    bool certified = false;
    std::vector<uint32_t> K_tried;
    std::vector<double> run_seconds;   // wall seconds of each K run, in K_tried's order
    AbelianGroupType tors1;    // the ell-primary torsion of H^1
    uint32_t free_rank1 = 0;   // rank of H^1(C_eq(Z)_(ell)), cross-checked against the good prime
    bool torsion_only_in_H1 = false;
    double seconds = 0;
};
// The K search of the shipped K-mode (D0 audit, "the K search").  The rule ([C, prop:certifiede](3)) certifies at ANY K
// with r_K = rank_Q, and certification is monotone in K, so no minimal K is needed: Doubling runs K = 2, 4, 8, ... (capped
// at the budget ell^K < 2^31) and stops at the first certified run; Linear (K = 2, 3, ...) is the D0 first draft's search,
// kept for the run count of the report.  An uncertified run at the budget is a refusal.
enum class KSearch { Doubling, Linear };
BadPrimeRun RunShippedKMode(uint32_t n, uint32_t ell, uint32_t count, KSearch search = KSearch::Doubling);

// ---- (D1) the direct Smith form at ell >= 5 ([C, prop:descentN](7)(a); DESCENT0_report.md Sec. 1's recommendation) ----
// The Smith valuations sigma (ascending, each < K) of a dense matrix over Z/ell^K, ell an odd prime, by valuation levels:
// at each level the F_ell column / row rank profile (engine::elim::rref over F_ell) gives an invertible-mod-ell block A11,
// A11 Y = A12 is solved ell-adically (Hensel / Dixon lifting from the F_ell inverse), the Schur complement
// A22 - A21 Y is 0 mod ell (ENGINE_CHECKed) and is divided by ell exactly.  One F_ell elimination per level, the rest
// kernels::gemm; a unique function of its input (the multiset of Smith valuations), equal to elim::invariant_factors'
// K - e (checked by the D1 validations).
std::vector<uint32_t> SmithLevelsEll(uint32_t ell, uint32_t K, engine::DenseBuf<uint32_t> A);
// The route of the ell >= 5 torsion in AssembleDescent: the shipped C_7 K-mode (D0, kept for the measurement), or ONE
// direct Smith of d^0 of C_eq (x) Z/ell^K at the budget K (D1, the default).
enum class BadPrimeRoute { KMode, DirectSmith };
// [C, prop:descentN](7)(a): C_eq(M) is LocalModel(desc, ell) at ell >= 5 (every stabiliser order 2, 4, 6 is prime to ell,
// so every column is non-periodic and the cochains are the Bredon cochains M^{Gamma_sigma}, degrees 0, 1, 2).  One Smith of
// d^0 over Z/ell^K in bases of the images of the averaging idempotents (the columns J of the rank profile of e_sigma mod
// ell, lifted by Nakayama -- (7)(a)'s "any bases"); certified iff s == rank_Q d^0 = c_0 - b_0 (b_0 the closed form).
// d^1 is onto by [C, prop:bredon](2) ((7)(a)): rank d^1 = c_2, so rank H^1 = c_1 - (c_0 - b_0) - c_2; verify_d1 also runs
// the Smith of d^1 and checks every valuation is 0 (the theorem tested).  K = 0: the budget DefaultKBudget(ell).
struct BredonSmithRun {
    uint32_t ell = 0, K = 0;
    uint32_t c0 = 0, c1 = 0, c2 = 0;   // the Bredon cochain ranks (reduced)
    uint32_t rank_Q = 0, r_K = 0;
    bool certified = false;
    bool d1_checked = false, d1_onto = false;
    AbelianGroupType tors1;            // the ell-primary torsion of H^1 (certified only)
    uint32_t free_rank1 = 0;
    double build_seconds = 0, smith_seconds = 0, seconds = 0;
    uint64_t peak_matrix_bytes = 0;
};
const LocalModel& BredonModel(uint32_t ell);   // LocalModel(Descriptor5(), ell), ell >= 5 prime, built once per ell
BredonSmithRun RunBredonDirectSmith(const CoefficientModule<uint32_t>& M, uint32_t ell, uint32_t K, uint32_t b0,
                                    bool verify_d1 = false);
// (PAR-ASM (b)) the RSS of one Bredon-tree prime at level 1 (D = n + 1), bytes: the per-prime term of the joint prediction.
uint64_t PredictTreeBredonRss(uint32_t D);
BadPrimeRun RunDirectSmithLevel1(uint32_t n, uint32_t ell, uint32_t count, bool verify_d1 = false);

// ---- (3), (4) the ell-local pieces ------------------------------------------------------
struct LocalDifferential {
    uint32_t q = 0;
    uint32_t rows = 0, cols = 0;   // the REDUCED matrix
    uint32_t rank_Q = 0;         // rank_{F_ell'} of the full-layout delta^q ([C, prop:certifiedeell](4),(5))
    uint32_t rank_derived = 0;   // c_q - rank d^{q-1} - h_q (cross-check; must equal rank_Q)
    uint32_t r_K = 0;
    uint32_t W = 0;              // the width the valuations were read at (K of the context, or 61)
    bool escalated = false;      // the K-route was not certified and the W = 61 route ran (ell = 2)
    bool certified = false;
    std::vector<uint32_t> nonunit;   // sigma >= 1, descending (certified only)
    int32_t copy_of = -1;        // (D5, tree route) d^q == d^{copy_of} as matrices ([C, prop:isomodeltree](2)(d)); not rebuilt
};
struct LocalPiece {
    uint32_t ell = 0, K = 0;
    uint32_t q_top = 0;                       // cochains built in degrees 0 .. q_top (differentials d^0 .. d^{q_top-1})
    std::vector<uint32_t> full_rank;          // cells_q * dim, q = 0 .. q_top
    std::vector<uint32_t> reduced_rank;       // c_q, q = 0 .. q_top
    std::vector<LocalDifferential> d;         // q = 0 .. q_top - 1
    std::vector<AbelianGroupType> tors;       // ell-primary tors H^q, q = 0 .. q_top (index 0 always zero)
    uint32_t rank_prime = 0;
    uint32_t dimH0_rank_prime = 0;            // rank P_0 - rank delta^0 over F_ell' (the (4)(b) comparison, printed)
    uint64_t denominators_checked = 0;        // coefficients of d and e_sigma whose denominators divide q_ell
    double build_seconds = 0, smith_seconds = 0, rank_seconds = 0, seconds = 0;
    uint64_t peak_matrix_bytes = 0;
    std::string route = "dense";              // (D5) "dense" = Iso^(ell) (RunLocalPiece), "tree" = Iso^(ell)_T (DescentTree.h)
    bool all_certified() const;
};
// Iso^(ell) (ell in {2, 3}) on the module M (dim D), differentials d^0 .. d^{q_top-1}, against the rational cohomology
// dims `h` (size >= q_top + 1; h[q] = dim H^q(Gamma; M_Q)).  K = 0: the rung's full width (30 at 2, 19 at 3); a smaller
// K is the negative control of arm (N).
// rank_Q d^q is read as rank_{F_ell'} of delta^q at `rank_prime` (C-D0 (4)); the derived rank is an ENGINE_CHECKed
// cross-check.  At ell = 2 with M = Sym^{sym_n} (sym_n >= 0) a differential the K-route does not certify is re-read on
// the O_2 -> Z/2^61 reduction through ModCtx<uint64_t> (ring2k rung uint64_t, read mod 2^61; the code map's W <= 61);
// force_wide runs that route only (the (W) check arm).  At ell = 3
// K = 19 is ring3k's cap: an exponent >= 19 leaves the differential uncertified, i.e. a REFUSAL (never a guess).
LocalPiece RunLocalPiece(const LocalModel& L, const CoefficientModule<uint32_t>& M, const engine::ModCtx<uint32_t>& ctx,
                         uint32_t q_top, const std::vector<uint32_t>& h, uint32_t rank_prime,
                         const CoefficientModule<uint32_t>& M_rank /* the same module over F_rank_prime */, int sym_n,
                         bool force_wide, bool rank_from_betti = false);
// (D1) rank_from_betti: rank_Q d^q from source (4)(c)(ii) ALONE (c_q - b_q - rank d^{q-1}, the Betti numbers h certified:
// the closed forms of [C, prop:descentN](4), (7)(b)), skipping the full-layout F_ell' rank of source (i); M_rank is then
// unused.  [C, prop:certifiedeell](4)(c) admits either source; the D1 validation runs both on its small cells.
// The same with M = Sym^n and the context built here.
LocalPiece RunLocalPieceSym(const LocalModel& L, uint32_t n, uint32_t q_top, const std::vector<uint32_t>& h,
                            uint32_t rank_prime, uint32_t K = 0, bool force_wide = false);
// q_ell = lcm |Q_sigma| ([C, prop:certifiedeell]): 3 at ell = 2, 4 at ell = 3.
uint32_t QEll(uint32_t ell);
// (C-D0 gap (2)) Build-time check that every coefficient of d and every e_sigma has denominators dividing q_ell
// (ENGINE_CHECK); returns the number of group-ring elements checked.
uint64_t CheckDenominators(const LocalModel& L, uint32_t q_ell);
uint32_t LocalWidth(uint32_t ell);   // 30 at 2, 19 at 3
// (D2) the building blocks of RunLocalPiece's reduced complex, exposed unchanged for DescentMaps: the per-column selection
// (J: basis columns of im P_cell, R: rows on which pi_R is an isomorphism on im P_cell) and the clause's matrix
// A = pi_R P_{q+1} delta P_q[:, J] ([C, prop:certifiedeell](5)).
void SelectBlockJR(const engine::ModCtx<uint32_t>& ctx, const integral_model::ColumnSpec& col, const CoefficientModule<uint32_t>& M,
                   std::map<integral_model::GKey, engine::DenseBuf<uint32_t>>& memo, std::vector<uint32_t>& J,
                   std::vector<uint32_t>& R);
engine::DenseBuf<uint32_t> ClauseMatrixReduced(const LocalModel& L, const engine::ModCtx<uint32_t>& ctx,
                                               const CoefficientModule<uint32_t>& M, uint32_t q,
                                               const std::vector<std::vector<uint32_t>>& Jq,
                                               const std::vector<std::vector<uint32_t>>& Rq1,
                                               const std::vector<const integral_model::ColumnSpec*>& colq,
                                               const std::vector<const integral_model::ColumnSpec*>& colq1);

// ---- the glue ------------------------------------------------------------------------
struct DescentOptions {
    uint32_t good_prime = 0;   // 0: DefaultGoodPrime(n)
    uint32_t q_top_two = 4;    // Iso^(2) cochains to this degree (4 production; 6 = arm (P))
    uint32_t q_top_three = 6;  // Iso^(3) (its cochains above degree 2 are one M^{-I}: the closed form, built)
    bool force_wide_two = false;   // Iso^(2) on the W = 61 route only (check arm (W))
    uint32_t K_three = 0;          // 0: ring3k's width 19; smaller: the refusal negative control (arm (R))
    KSearch k_search = KSearch::Doubling;
    BadPrimeRoute bad_route = BadPrimeRoute::DirectSmith;   // D1: the default; KMode = D0's route (measurement)
    bool verify_d1 = false;                                 // DirectSmith: also check d^1 of C_eq onto
    uint64_t rss_cap_bytes = 40ull << 30;   // (PAR-ASM (b)) the ceiling of the concurrent ell >= 5 primes (joint prediction)
};
struct DescentResult {
    uint32_t n = 0, good_prime = 0;
    std::vector<uint32_t> rational_dims;     // h_q, q = 0..2 (the good prime: sparse cokernel at even n, else C_7)
    bool good_sparse = false;                // (PAR-L1) the good prime ran off the shipped C_7 (sparse or dense route)
    GoodPrimeRoute good_route = GoodPrimeRoute::Shipped;   // (PAR-GP) which good-prime route ran
    std::vector<AbelianGroupType> H;         // q = 0 .. 5
    std::vector<std::pair<uint32_t, uint32_t>> dickson;   // (ell, count) of [C, prop:badprimes](1)
    std::vector<BadPrimeRun> bad;
    LocalPiece two, three;
    bool H5_from_periodicity = true;         // H^5's 2-part := H^3's (q_top_two < 6)
    double t_good = 0, t_bad = 0, t_two = 0, t_three = 0, t_total = 0;
    double cpu_good = 0, cpu_bad = 0, cpu_two = 0, cpu_three = 0, cpu_total = 0;   // process CPU seconds per piece
    // (PAR-ASM (b)) the ell >= 5 primes: how many ran at once, the per-prime and joint RSS predictions, the measured growth
    uint32_t ell_workers = 1;
    uint64_t ell_rss_pred_one = 0, ell_rss_pred_joint = 0, ell_rss_growth = 0;
    // (D0 audit M3) a result with ANY uncertified piece is REFUSED: refused = true, the reason, and H EMPTY -- never a
    // partial group list marked only by a flag ([C, prop:certifiedeell](6): never report a group from s < rank_Q).
    bool refused = false;
    std::string refusal;
    bool certified() const;
};
DescentResult AssembleDescent(uint32_t n, const DescentOptions& opt = {});

// ---- the comparators (measured only) -------------------------------------------------
struct ComparatorRun {
    std::string route;                  // "a", "c", "c'"
    std::vector<AbelianGroupType> H;    // whatever the route computes (q = 0 .. 4 for a / c; q = 0..1 for c')
    bool certified = false;
    double seconds = 0;
    double cpu_seconds = 0;
    uint64_t peak_rss = 0;
};
// (a) IntegralGroups on IsotropyModel(3) at {2, 3} + Dickson(n), good prime: H^0..H^3 and tors H^4.
ComparatorRun RunComparatorA(uint32_t n, uint32_t good_prime);
// (c) the free tree resolution at every prime, to degree q_top (H^0 .. H^{q_top-1} and tors H^{q_top}).
ComparatorRun RunComparatorTree(uint32_t n, uint32_t good_prime, uint32_t q_top = 4);
// (c') the tree Bredon complex (n even) at the good prime and the Dickson primes: H^0, H^1 of the Z[1/6] piece.
ComparatorRun RunComparatorTreeBredon(uint32_t n, uint32_t good_prime);
// dim H^0, H^1 of the tree Bredon complex over F_ell (ell >= 5, n even): the Betti-number comparison of the (Z) arm at the
// cost cells (D0 audit m5), an invariant independent of prop:badprimes' list.
std::vector<uint32_t> TreeBredonFieldDims(uint32_t n, uint32_t ell);

// The tree complex's differentials d^0 .. d^{q_top-1} for a module M over ctx (layout: S, U, then the <-I> block of
// degree q - 1), the closed form of [C, rmk:treeconeres] (as integral_line_probe.py's tree_complex).
std::vector<engine::DenseBuf<uint32_t>> TreeDifferentials(const engine::ModCtx<uint32_t>& ctx,
                                                          const CoefficientModule<uint32_t>& M, uint32_t q_top);
// The ell-primary torsion of H^1 .. H^{q_top} of a complex given by its differentials over Z/ell^K (ring2k at 2,
// ring3k at 3, elim::invariant_factors otherwise), certified against rank_Q (size q_top).  Returns false if some
// differential is not certified.
bool TorsionFromDifferentials(const std::vector<engine::DenseBuf<uint32_t>>& D, const engine::ModCtx<uint32_t>& ctx,
                              const std::vector<uint32_t>& rank_Q, std::vector<AbelianGroupType>& tors_out);

// ---- the level-N probe ---------------------------------------------------------------
// Ind_{Gamma_0(N)}^Gamma Sym^n over ctx through the shipped Gamma0LevelN table (MonomialOf).
CoefficientModule<uint32_t> InducedSymModule(uint32_t N, uint32_t n, const engine::ModCtx<uint32_t>& ctx);
uint32_t PsiIndex(uint32_t N);   // [SL_2(Z) : Gamma_0(N)]
struct LevelNProbe {
    uint32_t N = 0, n = 0, D = 0, good_prime = 0;
    std::vector<uint32_t> rational_dims;     // shipped level-N route at the good prime (ell' > n + 1, ell' prime to 6N)
    double t_shipped = 0, t_two = 0, t_three = 0, t_tree2 = 0, t_tree3 = 0;
    uint64_t rss_pred_two = 0, rss_meas_two = 0, rss_pred_three = 0, rss_meas_three = 0, rss_pred_tree = 0,
             rss_meas_tree = 0, rss_meas_shipped = 0;
    bool ran_two = false;                    // false: Iso^(2) skipped (prediction above the cap)
    LocalPiece two, three;
    AbelianGroupType tree_tors2[5], tree_tors3[5];
    bool tree_certified = false;
    bool agree = false;                      // Iso^(ell) torsion == tree torsion at 2 and 3 (where both ran)
};
// Predicted peak RSS (bytes) of RunLocalPiece on Iso^(ell) with a module of dim D to degree q_top (dense uint32_t:
// the full coboundary, the reduced working copy, the rho memo), and of the tree comparator.
uint64_t PredictLocalRss(uint32_t ell, uint32_t D, uint32_t q_top);
uint64_t PredictTreeRss(uint32_t D);
LevelNProbe RunLevelNProbe(uint32_t N, uint32_t n, uint64_t rss_cap_bytes);

}  // namespace integral_descent

#endif
