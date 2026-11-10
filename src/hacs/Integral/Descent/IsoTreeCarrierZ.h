// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_DESCENT_ISO_TREE_CARRIER_Z_H
#define INTEGRAL_DESCENT_ISO_TREE_CARRIER_Z_H

// ===========================================================================
// EXPORT-Z step 2, package EXZ2-P1a (hecke-extensions slot 25; docs/implementation/slot25/EXPORTZ2_design.md v2 Sec. 1.0, 1.1,
// 2, 3, 7; audit EXPORTZ2_design_audit.md F2, F4, F8, F11, F12, F14).  [C] = docs/latex/gl2_contract.tex, cited by label:
// def:isomodeltree, prop:isomodeltree (4) (the integral comparison Phi^Z_T, Psi^Z_T), prop:strictliftN (3), def:rhon.
//
// THE GLOBAL CARRIER (design Sec. 1.0).  The free tree model Iso_T (TreeModelChain(0, .), ranks 2, 3, 3 in degrees 0, 1, 2) is a free
// Z[Gamma]-resolution, so C^q = Hom_Gamma(Iso_T,q, M_N) = M_N^{cells(q)} with M_N = Ind_{Gamma_0(N)}^Gamma Sym^n Z^2 (D = psi(N)(n+1))
// computes H^q(Gamma_0(N); Sym^n Z^2) at EVERY prime at once.  IsoTreeCarrierZ holds
//   * d^0 : C^0 -> C^1 (3D x 2D) and d^1 : C^1 -> C^2 (3D x 3D) EXACTLY, as sparse integer matrices (nf::BigInt entries: binomials
//     of rho_n, 2^55 at n = 58, 2^197 at n = 200), built from the chain-level differential of TreeModelChain(0, 3) with the
//     integral block-monomial action rho(g) = (perm of Gamma_0(N)\Gamma) x rho_n(h_j(g)) (h_j = w_j g w_{j.g}^{-1} in int64, exactly;
//     rho_n over Z by the division recurrence on the columns, [C, def:rhon]);
//   * per word prime (an RNS lane) the residues of d^0, d^1 and the transported Hecke operator
//     T~_T = (Psi^Z_T)^* o T~_Iso o (Phi^Z_T)^*  ([C, prop:isomodeltree](4), (6); T~_Iso of prop:strictliftN(3) = hecke3::IsoHeckeN)
//     applied to a batch of columns (never densified);
//   * the integer term list of T~_Iso (IsoHeckeN::terms_Z(): (row block, column block, E in M_2(Z), mul in Z)) and its entrywise
//     majorant rho_n(|E|) |mul| (exact integers; design Sec. 1.6 -- P1a measures the bits, P2 proves with it).
// Nothing at ell in {2, 3} is used: TreeTransport keeps refusing ell outside {2, 3}; this class needs no ell-local half.
//
// THE MOD-q0 ELIMINATION AND THE DIXON PAIRS (design Sec. 1.1; the pinned rule is iso_dixon/v3 since EXZ2-P1b, see kBasisRule
// below: v2's rotation by floor(s * 3D / K) was cell-aligned, audit EXZ2P1a_audit.md F1).  Modulo q0 = 1073741789 (the first CrtPrimes
// prime; the next one when q0 | 6N or a rank profile mod q0 is below the rational one, F10):
//   * Pc_s = the first-pivot (greedy) column rank profile of d^1 in the order pi_s, Fr_s = the rest (in pi_s order);
//   * R0_s = the greedy row rank profile of d^0 on the rows Fr_s (in that order), P_s = Fr_s \ R0_s; |P_s| = r = b_1 is CHECKED;
//   * B_s column j = the primitive integer vector of ker d^1 supported on Pc_s + {P_s[j]} with positive entry den_j at P_s[j];
//     Phi_s column i = the primitive integer vector v with v^T d^0 = 0 supported on R0_s + {P_s[i]}, positive at P_s[i].
// Both are UNIQUE (the columns Pc_s of d^1 and the rows R0_s of d^0 are independent over Q, checked mod q0), so B_s, Phi_s depend on
// pi_s alone -- not on q0 once the rank profiles are right, not on the pivot rows, not on the order of the solve (the validation
// recomputes them at a second prime).  They are found by Dixon lifting on the SPARSE ECHELON factorisation mod q0 (zero-skipping
// left-looking elimination of the inserted vectors, fill measured) with the residual update R <- (R - M X_i) / q0 in exact integers:
// M X_i is evaluated in RNS lanes (one lane per word prime, sparse, enough lanes for |M| X_i < prod q / 2 entrywise) and lifted by
// Garner; the stopping rule is output-sensitive per column since EXZ2-P1b (P1a: the model's exz2v2_model.py schedule).  The verification is EXACT: d^1 b = 0 (Phi^T d^0 = 0) modulo word primes whose
// product exceeds 2 |d^1| |b| entrywise (bit bound, rounded up).  No int64 / __int128 path and no IntegerCap on this stage.
//
// THE SIZE CAP (O-3, the lead's decision): a predicted-memory refusal RssCeiling in measured bits (PredictPairsRss, the calibrated
// formula of the P1a audit, printed with the ceiling) decided before the pairs are built; nothing else caps the stage.
//
// The sums, saturation and certificate (design Sec. 1.2-1.4) are in DescentFreeBasisZ.h (EXZ2-P1b).  No console surface, no record
// key, no golden object, no settings key.
// ===========================================================================

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "../../Hecke/ExactInt.h"
#include "../../NumberField/BigInt.h"
#include "../../../ModArithmetic/ModCtx.h"

namespace integral_descent {
namespace hecke3 {
class IsoHeckeN;
}
namespace freez {

using nf::BigInt;

// ---- exact rho_n ----------------------------------------------------------------------------------------------------------------
// rho_n(g) over Z, row-major (n+1) x (n+1), in the convention of [C, def:rhon] (column j = coefficients of (a x + c y)^{n-j} (b x + d y)^j,
// entry (t, j) = the coefficient of x^{n-t} y^t), the same matrix as integral_model::rho_n reduced mod p.  Any g in M_2(Z) with
// (a, c) != (0, 0); O(n^2) operations by the exact division recurrence f_{j+1} = f_j (b + d y) / (a + c y).
std::vector<BigInt> RhoNZ(const hecke::Mat22& g, uint32_t n);
// The entrywise majorant rho_n(|g|) (|a|, |b|, |c|, |d|): |rho_n(g)| <= rho_n(|g|) entrywise (design Sec. 1.6).
std::vector<BigInt> RhoNAbsZ(const hecke::Mat22& g, uint32_t n);

// ---- sparse integer matrices ----------------------------------------------------------------------------------------------------
struct SparseZ {
    uint32_t rows = 0, cols = 0;
    std::vector<uint64_t> rowptr;     // CSR, rows + 1
    std::vector<uint32_t> col;        // ascending within a row
    std::vector<BigInt> val;          // non-zero
    uint64_t nnz() const { return val.size(); }
    uint32_t max_bits() const;
    // the largest row sum of |entries| (bits, rounded up) and of a column sum
    // (S25 EXZ2-P2, audit F11) no mutable cache: an unsealed matrix computes the value on every call WITHOUT writing anything (safe
    // under concurrent readers); seal() stores it once, before the matrix is shared (the carrier's ctor and transpose() seal).
    uint32_t max_row_abs_bits() const;
    void seal();
    SparseZ transpose() const;
    int32_t row_abs_bits_ = -1;
};
// residues of a SparseZ modulo a word prime q (the same pattern; zeros kept so the pattern is shared by every lane)
std::vector<uint32_t> Residues(const SparseZ& A, uint32_t q);

// ---- word-prime helpers -----------------------------------------------------------------------------------------------------
uint32_t BitsOf(const BigInt& x);                     // bit length of |x| (0 for 0)
long double ToLongDouble(const BigInt& x);            // the top 128 bits, scaled (measurement / majorants only)
// The lanes: CrtPrimes order (descending from 2^30), skipping `skip` (q0) and divisors of 6N, until sum(bitlen - 1) >= bits.
// (S25 EXZ2-P2, audit F13) the CrtPrimes list is computed once per process (CachedCrtPrimes, thread-safe) and only grown on demand.
std::vector<uint32_t> LanePrimes(uint64_t bits, uint32_t skip, uint32_t N);
// The first `count` CrtPrimes (descending from 2^30), from the process-wide cache (grown under a mutex, a copy returned).
std::vector<uint32_t> CachedCrtPrimes(uint32_t count);
// Garner over the lanes to the symmetric representative in (-P/2, P/2], exact (BigInt); the context holds the prefix products and
// inverses (built once per lane set).
struct GarnerCtx {
    explicit GarnerCtx(const std::vector<uint32_t>& primes);
    std::vector<uint32_t> primes, pre, inv;
    BigInt P;
};
BigInt GarnerSigned(const std::vector<uint32_t>& residues, const GarnerCtx& G);
BigInt GarnerSigned(const std::vector<uint32_t>& residues, const std::vector<uint32_t>& primes);
// FNV-1a-64 of a column-major integer matrix (rows x cols) in the model's canonical text: rows joined by ';', entries by ','
// (exz2v2_model.py fnv()).
uint64_t MatrixFnv(const std::vector<std::vector<BigInt>>& cols_major, uint32_t rows);

// =================================================================================================================================
// The carrier
// =================================================================================================================================
class IsoTreeCarrierZ {
public:
    IsoTreeCarrierZ(uint32_t N, uint32_t n);
    ~IsoTreeCarrierZ();
    uint32_t N() const { return N_; }
    uint32_t n() const { return n_; }
    uint32_t D() const { return D_; }
    uint32_t m() const { return m_; }                   // [Gamma : Gamma_0(N)]
    uint32_t cells(uint32_t q) const;                   // tree cells in degree q (2, 3, 3)
    uint32_t dim(uint32_t q) const { return cells(q) * D_; }
    uint32_t b0() const { return b0_; }
    uint32_t b1() const { return b1_; }                 // the rational Betti number = r
    // d^q : C^q -> C^{q+1}, q in {0, 1}, rows = C^{q+1} coordinates (cell x coset x Sym^n), cols = C^q coordinates.
    const SparseZ& d(uint32_t q) const { return q == 0 ? d0_ : d1_; }
    double build_seconds = 0;

    // ---- one RNS lane (a word prime q, q !| 6N, q > n + 1) ------------------------------------------------------------------------
    // Y = T~_T X mod q on C^deg (deg <= 2); X, Y row-major dim(deg) x k; T = the operator IsoHeckeN(N, n, p, ctx_q, deg) (its terms
    // only; built by the caller ONCE per lane and reused).
    void apply_hecke_mod(const hecke3::IsoHeckeN& T, uint32_t deg, const std::vector<uint32_t>& X, uint32_t k, std::vector<uint32_t>& Y) const;
    // Phi^*: C^deg_T -> C^deg_Iso and Psi^*: C^deg_Iso -> C^deg_T mod q (row-major, k columns); iso_dim = Iso cells(deg) * D.
    void phi_star_mod(uint32_t q, uint32_t deg, const std::vector<uint32_t>& V, uint32_t k, std::vector<uint32_t>& W) const;
    void psi_star_mod(uint32_t q, uint32_t deg, const std::vector<uint32_t>& W, uint32_t k, std::vector<uint32_t>& V) const;
    uint32_t iso_dim(uint32_t deg) const;
    // the cochain differential of the tree in degree deg mod q applied to X (C^deg -> C^{deg+1}), any deg <= 2 (deg 2 for the chain-map
    // check; built mod q on demand, not stored exactly)
    void d_mod(uint32_t q, uint32_t deg, const std::vector<uint32_t>& X, uint32_t k, std::vector<uint32_t>& Y) const;

    // ---- the entrywise majorant of T~_T on C^1 applied to non-negative column data (long double, measurement only) --------------
    // Y = |Psi^*| (sum_terms |mul| rho_n(|E|)) |Phi^*| X on C^1 (X >= 0, row-major dim(1) x k).
    void apply_hecke_majorant(const hecke3::IsoHeckeN& T, const std::vector<long double>& X, uint32_t k, std::vector<long double>& Y) const;

    // ---- (S25 EXZ2-P2, audit F7) the per-lane caches are lane-scoped ----------------------------------------------------------------
    // release_lane(q): drop the carrier's reference to lane q's TreeModule (its rho memo of dense d x d blocks); the next use of q
    // rebuilds it.  (Fix round 1, audit F9) SAFE under concurrent users: each lane call holds its own shared reference, so a module
    // in use by another thread (P1b's verification lanes share the carrier and the CrtPrimes list) lives until that call ends -- the
    // cost of a release by one user is a rebuild by the other, never a dangling module.  release_exact_memo(): drop the exact rho_n /
    // block-monomial / majorant memos (rebuilt on demand); while apply_hecke_majorant runs the clear is deferred to its end.
    // lane_count(): the lanes held now (diagnostics).
    void release_lane(uint32_t q) const;
    void release_exact_memo() const;
    size_t lane_count() const;

    struct Impl;

private:
    uint32_t N_ = 1, n_ = 0, D_ = 0, m_ = 1, b0_ = 0, b1_ = 0;
    SparseZ d0_, d1_;
    std::unique_ptr<Impl> impl_;
};

// =================================================================================================================================
// The mod-q0 sparse echelon (left-looking, zero-skipping) and the Dixon pairs
// =================================================================================================================================
// Sparse vectors over F_q (index ascending).
struct SpVec {
    std::vector<uint32_t> idx;
    std::vector<uint32_t> val;
};
class EchelonModQ {
public:
    EchelonModQ(uint32_t q, uint32_t length);
    uint32_t q() const { return q_; }
    uint32_t length() const { return L_; }
    uint32_t rank() const { return static_cast<uint32_t>(lead_.size()); }
    // Reduce v against the pivots; independent: becomes the next pivot (true), else dropped (false).
    bool insert(const SpVec& v);
    // x with sum_k x_k v_k = b for the INSERTED INDEPENDENT vectors v_k (insertion order); false if b is not in their span.
    // (S25 EXZ2-P2, audit F11) const and thread-safe: the reduction's scratch is local, nothing of the echelon is written.
    bool solve(const std::vector<uint32_t>& b, std::vector<uint32_t>& x) const;
    // (S25 EXZ2-P2) the transposed system: z (length L) with v_k . z = u_k for every inserted independent vector v_k (k = 0 .. rank-1),
    // z zero off the pivot leads (V = W T: T^T forward, then W^T z = u' backwards in descending lead order).  Const, thread-safe.
    void solve_transposed(const std::vector<uint32_t>& u, std::vector<uint32_t>& z) const;
    uint64_t fill() const;           // stored non-zeros of the reduced pivots
    uint64_t t_nnz() const;          // stored non-zeros of the triangular factor
    uint64_t bytes() const;

private:
    uint32_t q_, L_;
    std::vector<int32_t> pivot_of_;                  // coordinate -> pivot index or -1
    std::vector<uint32_t> lead_, lead_inv_;
    std::vector<SpVec> w_;                           // reduced pivot vectors
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> tcol_;   // T column k: (k' < k, coefficient)
    std::vector<uint32_t> acc_;     // insert()'s accumulator (insert is non-const)
    std::vector<uint8_t> mark_;     // insert()'s marks; solve() passes its own
    bool reduce(std::vector<uint32_t>& a, std::vector<uint32_t>& touched, std::vector<std::pair<uint32_t, uint32_t>>* coef,
                std::vector<uint8_t>& mark) const;
};

// ---- packed integer columns (EXZ2-P1b, audit EXZ2P1a_audit.md F5 / Sec. 4.2) -------------------------------------------------------
// CSC over one limb arena: per column a colptr into row[] / off[]; per non-zero the row index and a u64 offset into `limbs`, where
// limbs[off] = (limb count << 1) | sign and the magnitude's u32 limbs (little endian) follow.  Zeros are not stored.  Lane residues are
// read from the arena on the fly (mod_u32), so no resident copy per prime exists.
struct PackedZ {
    uint32_t rows = 0, cols = 0;
    std::vector<uint64_t> colptr = std::vector<uint64_t>(1, 0);
    std::vector<uint32_t> row;
    std::vector<uint64_t> off;
    std::vector<uint32_t> limbs;
    explicit PackedZ(uint32_t rows_ = 0) : rows(rows_) {}
    uint64_t nnz() const { return row.size(); }
    // entries: row STRICTLY ascending (ENGINE_CHECKed, fix round 1 F12), zeros skipped
    void append_col(const std::vector<std::pair<uint32_t, BigInt>>& entries);
    void append_dense_col(const std::vector<BigInt>& v);
    BigInt value(uint64_t k) const;
    uint32_t mod_u32(uint64_t k, uint32_t q) const;
    uint32_t bits(uint64_t k) const;
    std::vector<BigInt> col_dense(uint32_t j) const;
    std::vector<std::pair<uint32_t, BigInt>> col_sparse(uint32_t j) const;
    std::vector<std::vector<BigInt>> dense() const;   // cols x rows (small cells / validation only)
    uint32_t max_bits() const;
    uint64_t bytes() const;
    static PackedZ FromDense(const std::vector<std::vector<BigInt>>& cols, uint32_t rows);
};
// FNV-1a-64 of the canonical text (identical to MatrixFnv of the dense columns: rows joined by ';', entries by ',', zeros "0").
uint64_t PackedFnv(const PackedZ& A);

// The pinned rule iso_dixon/v3 (EXZ2-P1b; audit EXZ2P1a_audit.md F1, Sec. 3.3, "alt2"): with rot(k) the natural order of the 3D
// coordinates of C^1 started at k cyclically, t = floor(s / 2) and rho = (0, floor(D/2), floor(D/4))[t],
//     pi_s = rot(rho_t) for s even,   pi_s = reversed(rot(rho_t)) for s odd,      s = 0 .. K - 1, K = 3 (escalation K <= 5).
// Pair 0 is v2's pair 0 (B_0, Phi_0 and their FNVs unchanged).  Everything else of v2 stands: q0, the greedy rank profiles, the
// primitive positive columns, the exactness proof.  The Dixon stop is output-sensitive (audit F6, report finding 2): every column
// tries its vector rational reconstruction after every q0-adic step from step 2 on and leaves the batch when it verifies; columns are
// solved in batches of PairsOptions::dixon_batch (F5: the r-wide state does not exist).  B_s, Phi_s are unique (they do not depend
// on the schedule); the step counts are per column (the stats report the maximum).
constexpr uint32_t kQ0 = 1073741789u;
constexpr uint32_t kPairsK = 3;
constexpr uint32_t kPairsKMax = 5;
const char* const kBasisRule = "iso_dixon/v3";
// the order offset rho_t and the reversal flag of pair s (D = the cell size)
uint32_t PairRotation(uint32_t s, uint32_t D);
inline bool PairReversed(uint32_t s) { return (s & 1u) != 0; }
std::vector<uint32_t> PairOrder(uint32_t s, uint32_t D, uint32_t nv);

struct DixonStats {
    uint32_t steps = 0;               // q0-adic steps (the maximum over the columns)
    uint64_t col_steps = 0;           // sum over the columns
    uint32_t attempts = 0;            // reconstruction attempts (all columns)
    uint32_t lanes = 0;               // RNS lanes of the residual product
    uint32_t verify_lanes = 0;        // word primes of the exactness proof
    uint32_t batches = 0;
    double seconds = 0;
};
struct Pair {
    uint32_t s = 0, rot = 0;
    bool reversed = false;
    std::vector<uint32_t> Pc, R0, P;  // in pi_s order
    PackedZ B, Phi;                   // r columns each, rows = dim C^1 (packed, F5)
    DixonStats dixB, dixPhi;
    uint32_t bits_B = 0, bits_Phi = 0, bits_denB = 0, bits_denPhi = 0;
    uint64_t fill_d1 = 0, fill_d0 = 0, tnnz_d1 = 0, tnnz_d0 = 0;   // echelon fill (reduced pivots) and the triangular factor
    double t_elim = 0;
    uint64_t fnv_B = 0, fnv_Phi = 0;
    uint32_t support_Phi = 0;         // |R0 + P| (the Phi storage of design Sec. 1.5)
};
struct PairsOptions {
    uint32_t K = kPairsK;
    uint32_t q0 = 0;                   // 0: kQ0 (then the next CrtPrimes primes on q0 | 6N or a rank drop, F10)
    uint64_t rss_cap_bytes = 3ull << 30;
    uint32_t dixon_batch = 32;         // right-hand sides per Dixon batch (F5); the output does not depend on it
};
namespace testing {
// Validation-only hooks (EXZ2-P1b fix round 1, audit EXZ2P1b_audit.md F10: not in the public options).  nullptr = production.
struct PairsHooks {
    uint32_t rank_drop_q0 = 0;         // treat this prime as rank-dropping (exercises the F10 path)
};
}  // namespace testing
struct PairsResult {
    bool refused = false;
    std::string refusal;              // "REFUSED [RssCeiling] ..." etc.
    uint32_t q0 = 0, r = 0, rank_d1 = 0, rank_d0 = 0;
    uint32_t q0_confirm = 0;          // the second prime at which every pair's greedy profile (Pc, R0, P) was confirmed (fix round 1, F7)
    std::vector<uint32_t> q0_skipped; // primes passed over (q0 | 6N, a rank drop, or a profile not confirmed at q0_confirm: F10, F7)
    std::vector<Pair> pairs;
    uint64_t rss_pred = 0;
    double seconds = 0;
};
// The calibrated predictor of audit EXZ2P1a_audit.md Sec. 4.1 (copied exactly; bytes; h(b) = max(32, 16 ceil((4 ceil(b/32) + 8)/16)),
// b_d = bits(d), K pairs):
//   RSS_pairs = 16 MiB + 64 K r dim C^1                               (dense BigInt headers of B_s, Phi_s; 0 once packed)
//             + K r (24 h(b_d) + 0.8 (dim C^0 + r) h(beta_Phi))     (their non-zeros)
//             + 3 r dim C^0 (32 + h(30 sigma_Phi))                    (Phi-side Dixon: X, R and the freed generation)
//             + (nnz d^0 + nnz d^1) (72 + h(b_d))                     (transposes, q0 residues, echelons at <= 1.25 nnz)
//   a priori: beta_Phi = max(2 b_d + 8, ceil(n^1.6 / 3)),  sigma_Phi = ceil(beta_Phi / 10) + 2.
// The two layout substitutions the audit names: the header term is 0 (the masters are packed, PackedZ), and the Dixon term's r is the
// batch width min(r, dixon_batch) (the column-batched Dixon).  `packed = false, batch = r` gives the audit's formula for the P1a layout.
uint64_t PredictPairsRss(const IsoTreeCarrierZ& c, uint32_t K = kPairsK, bool packed = true, uint32_t batch = 32);
// (Fix round 1, audit F7) every pair's greedy profile is CONFIRMED at a second prime (the first CrtPrimes prime != q0 prime to 6N): a
// prime can only delay pivots, so equal ranks at q0 do not imply the rational profile; a disagreement moves q0 like a rank drop.
PairsResult BuildPairs(const IsoTreeCarrierZ& c, const PairsOptions& opt = {}, const testing::PairsHooks* hooks = nullptr);
// Append pair s = res.pairs.size() (the K-escalation of design Sec. 1.3 (iv)); same q0, same rule.  (Fix round 1, audit F5) the
// RssCeiling is re-evaluated for K = s + 1 first: above it, res.refused / res.refusal are set (REFUSED [RssCeiling]) and nothing is
// built; the F10 fallback's refusal is returned the same way.
void ExtendPairs(const IsoTreeCarrierZ& c, PairsResult& res, const PairsOptions& opt = {});

// Exact checks used by the validation and the measurement (BigInt arithmetic, no lanes): d^1 B = 0, Phi^T d^0 = 0.
bool ExactCocycles(const IsoTreeCarrierZ& c, const std::vector<std::vector<BigInt>>& B);
bool ExactKillsD0(const IsoTreeCarrierZ& c, const std::vector<std::vector<BigInt>>& Phi);
bool ExactCocycles(const IsoTreeCarrierZ& c, const PackedZ& B);
bool ExactKillsD0(const IsoTreeCarrierZ& c, const PackedZ& Phi);
// (S25 EXZ2-P3) the same checks with the first failing column (-1: every column holds); ExactCocycles / ExactKillsD0 on PackedZ call these
int64_t FirstNonCocycle(const IsoTreeCarrierZ& c, const PackedZ& B);
int64_t FirstNotKillingD0(const IsoTreeCarrierZ& c, const PackedZ& Phi);
// det of a square integer matrix (row-major) by Bareiss, exact.
BigInt DetBareiss(std::vector<BigInt> A, uint32_t r);
// G = Phi^T B (r x r, row-major)
std::vector<BigInt> PairingMatrix(const std::vector<std::vector<BigInt>>& Phi, const std::vector<std::vector<BigInt>>& B);
// G = Phi^T B for packed columns (rows = Phi.cols, cols = B.cols, row-major), exact
std::vector<BigInt> PairingMatrix(const PackedZ& Phi, const PackedZ& B);

}  // namespace freez
}  // namespace integral_descent

#endif
