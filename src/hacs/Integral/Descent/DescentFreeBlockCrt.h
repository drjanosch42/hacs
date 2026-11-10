// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_DESCENT_FREE_BLOCK_CRT_H
#define INTEGRAL_DESCENT_FREE_BLOCK_CRT_H

// ===========================================================================
// EXPORT-Z step 2, package EXZ2-P2 (hecke-extensions slot 25): the free block F_p over Z by RNS lanes and a PROVEN multimodular CRT.
// docs/implementation/slot25/EXPORTZ2_design.md v2 Sec. 1.5, 1.6, 2, 7.4; EXZ2P1a_audit.md F6, F7, F11, F13; EXZ2P2_report.md.
// [C] = docs/latex/gl2_contract.tex, cited by label: prop:isomodeltree (4), prop:strictliftN (3), def:rhon.
//
// INPUT.  The carrier Iso_T (IsoTreeCarrierZ: exact d^0, d^1; T~_T = Psi^* T~_Iso Phi^* on RNS lanes), a Z-basis B of the free class
// lattice L = Z^1 / B^1_sat (r cocycle columns; P1b's job -- here FIXTURE bases, see the report), the pair-0 index sets R0 (rows of d^0,
// independent over Q) and P (the r free positions) of iso_dixon (pair 0 is unchanged from v2 to v3), and pair 0's integer functionals
// Phi_0 (for the bound only).
//
// THE LANE (one word prime q; design Sec. 1.5 with audit F6: the per-lane LU route for Phi_0).  The normalised pair-0 functionals
//     Psi y = y_P - d^0[P, .] z,   M_0 z = y_{R0},   M_0 = d^0[R0, .]           (Phi_0 = Psi^T diag(den): Phi_0 is NEVER reduced)
// are applied by one sparse elimination of the rows R0 of d^0 mod q per lane (EchelonModQ, fill <= 1.1 nnz d^0) and a transposed
// solve per column.  Per tile of w columns of B (reduced mod q ON THE FLY from the BigInt master, never resident):
//     G''  += Psi B_tile,  H''_p += Psi T~_T B_tile  (Phi^*, then T~_Iso as GEMMs  W2[rb] += (mul rho_n(E)) W[cb]  per integer term of
//     IsoHeckeN::terms_Z, then Psi^*),   and at the end  F_p mod q = G''^{-1} H''_p  (one r x r elimination mod q for every p).
// q is ADMISSIBLE iff q !| 6N, the rows R0 stay independent mod q and G'' is invertible mod q; an inadmissible q is skipped and
// recorded (the order of the primes is fixed: CachedCrtPrimes, descending from 2^30).  F = (Psi B)^{-1} Psi T~ B = G'^{-1} Phi_0^T T~ B
// is the same rational matrix for every admissible q; it is INTEGRAL because B is a Z-basis of L and T_p preserves L.
//
// THE PROOF (design Sec. 1.6; no heuristic stop).  M := |Psi^*| (sum_terms |mul| rho_n(|E|)) |Phi^*| (the entrywise majorant,
// [C, def:rhon]: |rho_n(E)| <= rho_n(|E|)), evaluated on |B| upward-rounded (UpReal: 64-bit mantissa, binary exponent, every operation
// rounded up), gives beta_H = |Phi_0|^T M |B| >= |H|, H = Phi_0^T T~ B.  With G' = Phi_0^T B (exact, r x r):
//   a-priori:      Q = prod q > 2 beta_F,  beta_F = |G'^{-1}| beta_H  (evaluated as  Q |det G'| > 2 |adj G'| beta_H, no division);
//   a-posteriori:  Q > max ( |G'| |F^| + beta_H ),  F^ the symmetric lift: G' F^ == H mod Q and |G' F^ - H| < Q  =>  G' F^ = H.
// The a-posteriori test needs no integrality premise but needs det G' != 0; the a-priori one needs B to be a Z-basis of L.
// (Fix rounds 1-2, audit F2 / F3, re-audit N1.)  EXACTLY WHAT IS CHECKED inside FreeBlockCrt (everything else is the caller's input):
//   * on EVERY proof route (not lanes_only), exactly over Z on the sparse columns and before any lane: d^1 B = 0 and Phi_0^T d^0 = 0;
//     a failure is refused [Uncertified].  Phi_0 is NOT required to be the pair-0 functionals (the lanes use Psi from R0 / P only):
//     with d^1 B = 0 the lanes return the Hecke matrix F (T~ B = B F + d^0 x), and Phi_0^T d^0 = 0 makes G' F = H = Phi_0^T T~ B, so beta_H
//     bounds G' F;  a Phi_0 that does not kill d^0 (e.g. +-B) is refused;
//   * det G' != 0: by the exact fraction-free inverse (a-priori route), else by a WITNESS prime q with det(G' mod q) != 0; if every prime
//     tried divides det G' and their product exceeds the Hadamard-type bound prod_i sum_j |G'_ij|, then det G' = 0 is PROVEN and the run
//     is refused [SingularG];
//   * a-priori route only: the Z-basis certificate Phi_cert (FreeBlockInput::PhiCert) is re-checked exactly: Phi_cert^T d^0 = 0 and
//     det(Phi_cert^T B) = +-1 (with r = b_1 this proves B is a Z-basis of L/tors); without it or with one that fails: [Uncertified].
//     After an a-priori stop the r^2 entries are checked against the bound, |det G'| |F^_ij| <= ceil(sum_k |adj_ik| beta_H,kj), and a
//     violation is refused [BoundViolated] (a faulty lane; never "proven").
//   NOT checked (assumed from the caller): that R0 / P are pair 0 of iso_dixon (only shapes and index sets), that the carrier is the model.
//   The a-posteriori route needs no integrality premise; the a-priori one needs B to be a Z-basis of L (the certificate).
// The lanes run in SWEEPS: when the a-priori prefix is known (inverse + certificate), all its lanes run at once on `threads` workers;
// otherwise rounds of `batch` primes.  The proofs are evaluated at the checkpoints L = batch, 2 batch, .. (and the a-priori prefix), on
// the digit planes of the first L lanes (garner_step, ENGINE_KERNEL bodies; the planes are prefix-stable, so each plane is computed
// once); each operator stops at the first checkpoint where ONE of the two inequalities holds -- output-sensitive, proven either way,
// and the checkpoints depend only on the candidate order and admissibility, never on `threads`.  Only a test that can close lifts the
// mixed-radix digits to nf::BigInt.
//
// PARALLELISATION (the owner's principle).  Primes: one lane per prime, lanes concurrently on `threads` workers (std::thread, an
// atomic lane counter, each lane owns its echelon, tile buffers and rho memo; the carrier's lane TreeModule is released when the lane
// ends, audit F7, also when it throws).  THE CARRIER IS SHARED SAFELY (fix round 1, audit F9): the carrier holds each lane's
// TreeModule by shared_ptr and every user keeps its own reference for the duration of a call, so release_lane(q) by one user never
// frees a module another thread is using (it only drops the carrier's cached reference: the other user's next call rebuilds it -- time,
// not correctness); the exact memos are cleared only when no majorant pass holds them.  So P1b's verification lanes may use the same
// carrier and the same primes concurrently.  Rank: tiles of w columns.  Entries: the Garner digit planes are vector operations over the r^2 entries
// (engine::rns::garner_step: portable / AVX2 / AVX-512).  GEMM: engine::kernels::gemm (portable / AVX2 / AVX-512 by ENGINE_KERNEL or
// gemm_path).  CUDA (opt-in, route = Gpu, needs ENGINE_DEVICE=gpu): the T~_Iso terms of a lane as ONE batched u32 GEMM with that
// prime's modulus on its own stream (engine::cuda::GemmBatch, no device-kernel change), the fixed-order sum and everything else on the
// CPU.  Every route produces the canonical residues of the same integers: F (and its FNV) is bit-identical across routes and threads.
//
// (EXZ2-P3) ONE COLUMN FORMAT AND ONE PROOF LOGIC.  The input columns are P1b's PackedZ (IsoTreeCarrierZ.h; the masters of
// BuildFreeBasis), so the lanes reduce B mod q straight from the limb arena (in-word, no BigInt per entry).  SparseColsZ is only the
// fixture file format (to_packed).  The premise checks call IsoTreeCarrierZ's FirstNonCocycle / FirstNotKillingD0 and the a-priori
// certificate is P1b's CertifyBasis (DescentFreeBasisZ.h: d^1 B = 0, Phi_cert^T d^0 = 0, det(Phi_cert^T B) = +-1 by residues beyond
// 2 x Hadamard, and the rank witness rank L <= r) -- there is no second implementation here.
// (EXZ2-P3) THE BOUND WITH Phi_cert (FreeBlockOptions::bound_with; P2 report rec. 2).  Once the certificate holds, G_c = Phi_cert^T B is
// unimodular, so G_c^{-1} is INTEGRAL: it is found by q-adic lifting with symmetric digits, which terminates exactly (residual 0 => G_c X
// = I over Z), output-sensitively, in place of the fraction-free inverse of G' = Phi_0^T B.  Phi_cert kills d^0, so Phi_cert^T T~ B =
// G_c F, and beta_H = |Phi_cert|^T M |B|, beta_F = |G_c^{-1}| beta_H prove the CRT exactly as with Phi_0 (det = +-1: the a-priori test
// is Q > 2 beta_F); the a-posteriori test reads |G_c| |F^| + beta_H.  BESIDE it the a-posteriori test also runs with Phi_0 (G'_0 =
// Phi_0^T B, det != 0 by a witness prime) on the SAME majorant pass (|Phi_0|^T V costs nnz(Phi_0) r): each test proves on its own, the
// first that holds stops the operator (FreeBlockOp::proof_detail = "apost_phi0" when Phi_0's closed it).  The lanes do not change (they
// never use Phi): F and its FNV are the same integers; only the prime count of the proof can differ.  (As measured, EXZ2P3_report.md:
// the bound costs 1.3 s instead of 2.6 s at (1, 200), but the a-priori prefix is ~ 50 % longer than with Phi_0, |G_c^{-1}| |Phi_cert|^T
// being a product of absolute values; the exact dual functionals Phi_cert G_c^{-T} were tried and were no tighter, at r nnz(Phi_cert)
// BigInt products.)
//
// No record key, no golden object, no console surface (P3: DescentModuleExportZ.h, ExportModuleZ2).
// ===========================================================================

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "IsoTreeCarrierZ.h"
#include "../../../ModArithmetic/CpuFeatures.h"

namespace integral_descent {
namespace freez {

struct Certificate;  // DescentFreeBasisZ.h

// Sparse integer columns (rows x cols): column j = (row, value) with ascending rows, values non-zero.
struct SparseColsZ {
    uint32_t rows = 0;
    std::vector<std::vector<std::pair<uint32_t, BigInt>>> cols;
    uint32_t ncols() const { return static_cast<uint32_t>(cols.size()); }
    uint64_t nnz() const;
    uint32_t max_bits() const;
    std::vector<std::vector<BigInt>> dense() const;   // column-major dense copy (validation / small cells)
    uint64_t fnv() const;                             // MatrixFnv of dense() without materialising it
    static SparseColsZ from_dense(const std::vector<std::vector<BigInt>>& cols, uint32_t rows);
    // (EXZ2-P3) the production column format (FreeBlockInput) and back
    PackedZ to_packed() const;
    static SparseColsZ from_packed(const PackedZ& A);
};

// ---- the refusals (fix round 1, audit F5 e: data-dependent outcomes are refusals, not exceptions) ----------------------------------
//   NoDevice        route = Gpu without ENGINE_DEVICE=gpu / a device;
//   Terms           IsoHeckeN refused an operator (its text);
//   RssCeiling      the predicted bytes exceed rss_cap_bytes (O-3), before any lane;
//   Uncertified     use_apriori without a certificate Phi_cert, or the certificate fails (d^1 B, Phi_cert^T d^0, det(Phi_cert^T B) = +-1);
//   SingularG       det G' = 0 (G' = Phi_0^T B): proven by the fraction-free inverse or by the witness search;
//   NoProof         no proof within max_primes lanes (e.g. F not integral in B);
//   Inadmissible    64 inadmissible primes in a row;
//   BoundViolated   after an a-priori stop |det G'| |F^_ij| exceeds the bound (a faulty lane or a false premise).
// A violated PRECONDITION (shapes, indices, sortedness; ENGINE_CHECK) and a lane that throws are genuine errors and still throw.
enum class FreeBlockRefusal { None, NoDevice, Terms, RssCeiling, Uncertified, SingularG, NoProof, Inadmissible, BoundViolated };
const char* FreeBlockRefusalName(FreeBlockRefusal r);

// ---- upward-rounded non-negative reals (the bound arithmetic) ---------------------------------------------------------------------
struct UpReal {
    uint64_t m = 0;   // value m * 2^e, m normalised (top bit set) or 0
    int32_t e = 0;
    static UpReal of_u64(uint64_t v);
    static UpReal of_big(const BigInt& x);   // >= |x|
    bool is_zero() const { return m == 0; }
    BigInt ceil_big() const;                 // the least integer >= the value
};
UpReal UpAdd(const UpReal& a, const UpReal& b);
UpReal UpMul(const UpReal& a, const UpReal& b);

// ---- the fixture (P2 runs beside P1b: fixed Z-bases from the model, re-certified by the validation) -------------------------------
struct FreeBlockFixture {
    uint32_t N = 0, n = 0, r = 0, dim1 = 0;
    std::vector<uint32_t> ps, R0, P;
    SparseColsZ B, Phi0, PhiCert;
    std::map<uint32_t, std::vector<BigInt>> F_model;   // p -> r x r row-major (the model's exact F_p in this B)
};
// Every read is checked (fix round 1, audit F6): a malformed line, an index out of range, a duplicate (i, j), a zero entry, a bad
// integer, a bad F_<p> name or a matrix of the wrong shape (B, Phi0, PhiCert: dim1 x r; F_p: r x r) returns false with the reason.
bool LoadFreeBlockFixture(const std::string& path, FreeBlockFixture& fx, std::string& why);

// ---- the CRT ------------------------------------------------------------------------------------------------------------------------
enum class LaneRoute { Cpu, Gpu };
const char* LaneRouteName(LaneRoute r);

// (EXZ2-P3) which functionals carry the Sec. 1.6 bound: Phi_0 (P2: G' = Phi_0^T B, fraction-free inverse) or Phi_cert (G_c unimodular,
// integral inverse by q-adic lifting).  PhiCert: Phi_cert when the a-priori route runs with a certificate, else Phi_0.  Auto: the same,
// but Phi_0 while r (bits Phi_0 + bits B) <= kAutoBoundBits (its inverse is then cheap and its a-priori bound the tighter one).
enum class BoundBasis { Auto, Phi0, PhiCert };
const char* BoundBasisName(BoundBasis b);
// Auto: Phi_cert iff r (bits Phi_0 + bits B) > kAutoBoundBits (the fraction-free inverse of G' grows to ~ that many bits per entry)
constexpr uint64_t kAutoBoundBits = 20000;

struct FreeBlockOptions {
    uint32_t threads = 0;                      // 0: ENGINE_THREADS, else hardware_concurrency, capped at 16
    uint32_t tile = 128;                       // columns of B per tile (the rank axis)
    uint32_t batch = 4;                        // primes per round of the output-sensitive stop (FIXED: independent of threads)
    uint32_t max_primes = 4096;
    engine::cpu::IsaPath gemm_path = engine::cpu::IsaPath::Auto;     // != Auto: forwarded to kernels::set_kernel_path for the run and
                                                                     // restored on every exit (RAII).  The override is PROCESS-WIDE:
                                                                     // a test / measurement knob, not for concurrent callers.  Auto
                                                                     // leaves the caller's path untouched.
    engine::cpu::IsaPath garner_path = engine::cpu::IsaPath::Auto;
    LaneRoute route = LaneRoute::Cpu;          // Gpu: refused unless engine::cuda::device_enabled()
    bool use_apriori = true, use_apost = true; // which proofs may stop the run (at least one)
    uint32_t lanes_only = 0;                   // MEASUREMENT: run exactly this many admissible lanes, no proof, no lift (F may be
                                               // rational); with Phi_0 the bound is still evaluated (bits, primes_apriori)
    uint64_t rss_cap_bytes = 8ull << 30;       // the predicted-memory refusal [RssCeiling] (O-3)
    // (slot-26 Z1) p | N in ps is the free-model U_p of `prop:strictliftN` item (9) ([C] 1.26.0-pre.3): IsoHeckeN(.., allow_bad_primes).
    // Default false: p | N refused (PDividesN) as before.
    bool allow_bad_primes = false;
    uint64_t rho_memo_cap_bytes = 64ull << 20; // per lane
    uint32_t skip_first = 0;                   // TEST ONLY: start the prime list this many admissible primes later (q-independence)
    BoundBasis bound_with = BoundBasis::Phi0;  // (EXZ2-P3) P2's bound by default (its pins); ExportModuleZ2 passes Auto (ModuleZ2Options::bound)
};

struct FreeBlockOp {
    uint32_t p = 0;
    std::vector<BigInt> F;                     // r x r row-major (empty if not proven)
    bool proven = false;
    std::string proof;                         // "apriori" | "apost" | ""
    std::string proof_detail;                  // (EXZ2-P3) "apost_phi0": the a-posteriori test closed with Phi_0 beside the dual Xi
    uint32_t primes_used = 0;                  // lanes in the proof
    uint32_t primes_apriori = 0;               // the smallest prefix the a-priori bound needs (0: not computed)
    uint32_t bits_F = 0, bits_beta_H = 0, bits_beta_F = 0;
    uint64_t fnv = 0;                          // MatrixFnv of F (rows ';', entries ',')
    // (fix round 1, audit F4) the bound itself, so that a validation can check it against the exact H = G' F:
    std::vector<BigInt> beta_H_ceil;           // r x r: ceil(beta_H,ij) >= |H_ij|
    std::vector<BigInt> apriori_rhs;           // r x r: ceil(sum_k |adj_ik| beta_H,kj) >= |det G'| |F_ij| (empty without the inverse)
};

struct FreeBlockResult {
    bool ok = false;
    bool refused = false;
    FreeBlockRefusal code = FreeBlockRefusal::None;
    std::string refusal;                       // "REFUSED [<code>] ..." (FreeBlockRefusalName)
    uint32_t N = 0, n = 0, r = 0, threads = 0;
    // (fix round 1) the premises as established: G' = Phi_0^T B exactly, its determinant (when the inverse was formed), how det G' != 0
    // was shown ("inverse" | "witness" | "" when no proof ran), the witness prime, and whether the a-priori certificate was checked
    std::vector<BigInt> Gp;
    BigInt detG;
    std::string nonsingular_by;
    uint32_t det_witness = 0;
    bool certified = false;
    bool cert_reused = false;                  // (Z3a) the certificate was BuildFreeBasis' (PriorCert), not re-run
    // (EXZ2-P3) the certificate (P1b's CertifyBasis): its rank witness prime and det primes; the bound's basis "phi0" | "phi_cert"
    uint32_t cert_rank_q = 0, cert_primes = 0;
    std::string bound_basis;
    double t_cert = 0, t_inverse = 0;
    uint32_t inverse_steps = 0;                // q-adic steps of G_c^{-1} (phi_cert)
    std::string route, gemm_path, garner_path;
    std::vector<FreeBlockOp> ops;
    std::vector<uint32_t> primes;              // admissible lanes used, in order
    std::vector<uint32_t> skipped;             // inadmissible primes met (in order)
    uint32_t rounds = 0;
    uint64_t rss_pred = 0;
    double t_bound = 0, t_lanes = 0, t_garner = 0, t_lift = 0, t_total = 0;
    double lane_seconds_max = 0, lane_seconds_sum = 0;
    uint64_t lane_rho_memo_peak = 0;
    size_t carrier_lanes_left = 0;             // carrier lane TreeModules alive after the run (F7: must be 0)
    // lanes_only: the residues of F mod q per lane (ops x r x r), for the route / thread comparison
    std::vector<std::vector<uint32_t>> lane_residues;
};

// Preconditions (ENGINE_CHECK, fix round 1 audit F6): B is dim C^1 x r with r = b_1; Phi0 (when given) and PhiCert (when given) are
// dim C^1 x r; every column has ascending distinct rows < dim C^1 and non-zero values; R0 has rank d^0 = dim C^0 - b_0 distinct rows
// < dim C^1; P has r distinct positions < dim C^1.  The admissibility of a lane (R0 independent mod q) is data, not a precondition.
struct FreeBlockInput {
    const IsoTreeCarrierZ* car = nullptr;
    const PackedZ* B = nullptr;                // (EXZ2-P3) P1b's packed masters (BuildFreeBasis: B, pairs[0].Phi, Phi_cert)
    const PackedZ* Phi0 = nullptr;
    const PackedZ* PhiCert = nullptr;          // the Z-basis certificate (audit F3): required for use_apriori, re-checked exactly
    // (Z3a, EXZ2P3 F10) the certificate CertifyBasis already established for exactly this (B, Phi_cert), so that it runs once: used ONLY when
    // PriorCert->ok / rank_ok / cocycles / kills_d0 all hold AND its own stored fnv_B / fnv_Phi equal PackedFnv(*B) / PackedFnv(*PhiCert) AND its stored carrier key (N, n, dims) equals car's; otherwise
    // (null, not ok, or a fingerprint mismatch) FreeBlockCrt runs CertifyBasis itself.  Every path that produces F is certified either way.
    const Certificate* PriorCert = nullptr;
    std::vector<uint32_t> R0, P;
    std::vector<uint32_t> ps;
};

// Predicted peak RSS (bytes) of FreeBlockCrt; terms = sum over p of |terms_Z(1)| (bounds each lane's rho memo by terms d^2 4 B;
// 0 = unknown: the memo cap is charged).
uint64_t PredictFreeBlockRss(const FreeBlockInput& in, const FreeBlockOptions& opt, uint64_t terms = 0);
// Rank r = 0 (b_1 = 0: every odd n, and e.g. (1, 0)) is legal: ok, every operator proven ("rank0") with the empty F, no lane run.
FreeBlockResult FreeBlockCrt(const FreeBlockInput& in, const FreeBlockOptions& opt = {});

// One lane alone (validation): F_p mod q for every p of `ps` (ops x r x r row-major); false iff q is inadmissible.
bool FreeBlockLane(const FreeBlockInput& in, uint32_t q, const FreeBlockOptions& opt, std::vector<uint32_t>& out, std::string* why = nullptr);

// (EXZ2-P3) the inverse of a unimodular integer matrix (det G = +-1, r x r row-major) by q-adic lifting with symmetric digits at the
// first CrtPrimes prime where G is invertible: X = sum q^i X_i, R_{i+1} = (R_i - G X_i) / q exactly; R = 0 proves G X = I over Z
// (q^i R_i = I - G X_{<i} is exact).  false when G is singular mod each of 8 primes, or the expansion has not ended after max_steps
// (0: 2 + the Hadamard bits of G / 29 -- more than any unimodular G needs).
bool InverseUnimodular(const std::vector<BigInt>& G, uint32_t r, std::vector<BigInt>& X, uint32_t* steps = nullptr, uint32_t max_steps = 0);

// det(x I - F) over Z by Berkowitz (division-free), high -> low (monic first), F r x r row-major.
std::vector<BigInt> CharpolyBerkowitz(const std::vector<BigInt>& F, uint32_t r);
// Exact r x r products (row-major).
std::vector<BigInt> MatMulZ(const std::vector<BigInt>& A, const std::vector<BigInt>& B, uint32_t r);

}  // namespace freez
}  // namespace integral_descent

#endif
