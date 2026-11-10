// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Stage S10/S11 -- level-(c) block-diagonalization: single-operator (S10)
// and simultaneous MaxSpec(T) refinement for a commuting family (S11).
// See HeckeModuleDecomposition.h for the binding definition (plan Sec. 3)
// and the certification contract.

#include "HeckeModuleDecomposition.h"

// (hecke-extensions E0a) SolveColumnsExact / HStack / KernelAsMatrix / RankOf /
// NilpotencyExponent / SplitMix64 / kRcfSeed / PrimaryRCFBasis moved VERBATIM
// out of this file's anonymous namespace into Hecke/SubspaceOps.{h,cpp}, so
// that E0b's locality refinement and E1a's local-algebra layer can share them
// without `Hecke/` having to include `HeckeExt/` (slot-1 plan Sec. 2.3 item 5).
// Pure refactor: same bodies, same seeds, same search order -- the golden
// reference (tier B's basis-dependent decomp_*_U included) is unchanged.
#include "SubspaceOps.h"
#include "HeckeLinAlg.h"   // PartitionFromNullities (G14)

#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/ModArithmetic.h"
#include "HeckeSettings.h"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace hecke {

namespace {

using ModArithmetic::mod_add;
using ModArithmetic::mod_minus;
using ModArithmetic::mod_mul;

// Stage S11 -- deterministic seed for the residue-degree witness search
// (deterministic-seed combinations sum c_j T_{p_j}; search only, every
// witness is re-certified via CertifyIrreducible before being trusted).
constexpr uint64_t kResidueWitnessSeed = 0x510C0FFEE5EED042ull;
constexpr uint32_t kResidueWitnessMaxAttempts = 64;

// (E0b-S, slot 3 WP-3) The deterministic, TERMINATING fallback of [X] Sec. 3.1
// step 3 enumerates Abar's coordinate vectors lexicographically.  It is
// mathematically guaranteed to find a generator (a finite field is never the
// union of its proper subfields) and, on the evidence of the failure
// probability of the seeded phase -- at most omega(f) * ell^(-f/2) per
// candidate -- is never entered.  The cap exists so that a defect elsewhere
// surfaces as a NAMED error in bounded time instead of a hang; exceeding it is
// a contradiction with the mathematics, i.e. a bug, and says so.
constexpr uint64_t kResidueLexCap = 1u << 20;

uint32_t Gcd32(uint32_t a, uint32_t b) {
    while (b != 0) {
        const uint32_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

uint32_t Lcm32(uint32_t a, uint32_t b) {
    if (a == 0 || b == 0) return 0;
    return (a / Gcd32(a, b)) * b;
}

// (E0a/E0b) Least multiple of f with ell^N >= D (f >= 1, ell >= 2) -- the
// Teichmueller exponent N of [E] Cert. 5.2 clause (2).  Arithmetic only; the
// k = 1 analytic path of [X] Sec. 3.1 computes N and never forms w_hat.
uint32_t TeichmuellerExponent(uint32_t ell, uint32_t f, uint32_t D) {
    uint32_t t = 0;
    uint64_t acc = 1;
    while (acc < static_cast<uint64_t>(D)) { acc *= static_cast<uint64_t>(ell); ++t; }
    if (f == 0) return t;
    return ((t + f - 1u) / f) * f;
}

// (E0a/E0b) The unique irreducible factor of a pi-primary block's min-poly
// pi^e.
bool PrimaryIrreducible(const FpPoly& min_poly, FpPoly& pi_out) {
    if (min_poly.degree() < 1) return false;
    const FpPolyFactorization fac = FactorPoly(min_poly);
    if (fac.factors.size() != 1) return false;
    pi_out = fac.factors[0].pi;
    return true;
}

// ---------------------------------------------------------------------------
// (E0a/E0b) CLAUSES (2) AND (3) of [E] Cert. 5.2 on ONE candidate witness `W`
// whose char-poly's squarefree part is the irreducible `pi_w` of degree `f`
// (clause (1), checked by the caller).
//
//   (2) w_hat := W^(ell^N) is the Teichmueller lift of W's residue, and
//       pi_w(w_hat) = 0 is CHECKED, not assumed;
//   (3) for every operator t_p there is P_p in F_ell[X] with deg P_p < f such
//       that n_p := t_p - P_p(w_hat) is NILPOTENT.
//
// P_p is FOUND on U := ker pi_p(t_p) INTERSECT ker pi_w(W) (one kernel, via
// VStack), where t_p and P_p(w_hat) are commuting semisimple operators
// differing by a nilpotent and 1, w_hat, ..., w_hat^{f-1} are independent --
// but the CERTIFICATE is clause (3) on the WHOLE block, re-derived from the
// returned (pi_w, N, P_p) alone.  The solve is never trusted by itself.
//
// Returns kLocalityOk, or the first failing clause's code.  THIS IS THE ONE
// IMPLEMENTATION: the read-only CertifyLocality (E0a) and the refinement loop
// (E0b) both call it, so the certificate the loop enforces and the certificate
// the grid sweep verified cannot drift apart.
// ---------------------------------------------------------------------------
uint32_t CertifyWitnessClauses23(const Matrix& W, const FpPoly& pi_w, uint32_t f, uint32_t N,
                                 const std::vector<std::pair<uint64_t, Matrix>>& ops,
                                 const std::map<uint64_t, FpPoly>& pi_by_p,
                                 const std::map<uint64_t, Matrix>& Asub_by_p,
                                 uint32_t D, uint32_t ell,
                                 std::map<uint64_t, FpPoly>& character_out,
                                 Matrix* w_hat_out) {
    // (2) the Teichmueller lift.
    const Matrix w_hat = FrobeniusPower(W, ell, N);
    if (!EvalPolyAtMatrix(pi_w, w_hat).is_zero()) return kLocalityWHatNotInResidue;

    // (3) the character P_p and the nilpotency of t_p - P_p(w_hat).
    const Matrix PiW = EvalPolyAtMatrix(pi_w, W);
    std::map<uint64_t, FpPoly> character;
    for (const auto& op : ops) {
        const uint64_t p = op.first;
        const Matrix& Asub = Asub_by_p.at(p);
        const Matrix PiP = EvalPolyAtMatrix(pi_by_p.at(p), Asub);
        // U = ker pi_p(t_p) INTERSECT ker pi_w(W), as one kernel.
        const Matrix U = KernelAsMatrix(VStack(PiP, PiW));
        if (U.cols == 0) return kLocalityEmptyIntersection;
        Matrix w_U(0, 0, ell), t_U(0, 0, ell);
        try {
            w_U = SolveColumnsExact(U, w_hat * U, "CertifyLocality(w_hat|_U)");
            t_U = SolveColumnsExact(U, Asub * U, "CertifyLocality(t_p|_U)");
        } catch (const std::exception&) {
            return kLocalityCharacterUnsolvable;
        }
        // Solve sum_{i<f} c_i (w_hat|_U)^i = t_p|_U in the c_i.
        const uint32_t u = U.cols;
        Matrix Bsys(u * u, f, ell), rhs(u * u, 1, ell);
        Matrix pw(u, ell); // identity
        for (uint32_t i = 0; i < f; ++i) {
            for (uint32_t a = 0; a < u; ++a)
                for (uint32_t b = 0; b < u; ++b) Bsys.data[a * u + b][i] = pw.data[a][b];
            if (i + 1 < f) pw = pw * w_U;
        }
        for (uint32_t a = 0; a < u; ++a)
            for (uint32_t b = 0; b < u; ++b) rhs.data[a * u + b][0] = t_U.data[a][b];
        Matrix c(0, 0, ell);
        try {
            c = SolveColumnsExact(Bsys, rhs, "CertifyLocality(character)");
        } catch (const std::exception&) {
            return kLocalityCharacterUnsolvable;
        }
        std::vector<uint32_t> coeffs(f);
        for (uint32_t i = 0; i < f; ++i) coeffs[i] = c.data[i][0];
        const FpPoly P(ell, coeffs);
        const Matrix n_p = Asub - EvalPolyAtMatrix(P, w_hat);
        if (!IsNilpotent(n_p, D)) return kLocalityNotNilpotent;
        character.emplace(p, P);
    }

    character_out = std::move(character);
    if (w_hat_out) *w_hat_out = w_hat;
    return kLocalityOk;
}

Matrix SubBlock(const Matrix& C, uint32_t off, uint32_t size) {
    Matrix D(size, size, C.p);
    for (uint32_t i = 0; i < size; ++i) {
        for (uint32_t j = 0; j < size; ++j) D.data[i][j] = C.data[off + i][off + j];
    }
    return D;
}

bool IsBlockDiagonal(const Matrix& C, const std::vector<uint32_t>& dims) {
    uint32_t off = 0;
    std::vector<std::pair<uint32_t, uint32_t>> ranges;
    for (uint32_t s : dims) {
        ranges.emplace_back(off, off + s);
        off += s;
    }
    if (off != C.rows || C.rows != C.cols) return false;
    for (uint32_t i = 0; i < C.rows; ++i) {
        // block containing row i
        uint32_t blo = 0, bhi = 0;
        for (const auto& rg : ranges) {
            if (i >= rg.first && i < rg.second) { blo = rg.first; bhi = rg.second; break; }
        }
        for (uint32_t j = 0; j < C.cols; ++j) {
            if ((j < blo || j >= bhi) && C.data[i][j] != 0) return false;
        }
    }
    return true;
}

// Companion matrix of a monic q (deg k >= 1) in the cyclic basis
// v, Av, ..., A^{k-1} v: subdiagonal 1's, last column -q_i.
Matrix CompanionOfPoly(const FpPoly& q) {
    if (!q.is_monic() || q.degree() < 1) {
        throw std::invalid_argument("CompanionOfPoly: needs a monic polynomial of degree >= 1");
    }
    const uint32_t k = static_cast<uint32_t>(q.degree());
    Matrix C(k, k, q.ell);
    for (uint32_t i = 1; i < k; ++i) C.data[i][i - 1] = 1;
    for (uint32_t i = 0; i < k; ++i) C.data[i][k - 1] = mod_minus(q.coeffs[i], q.ell);
    return C;
}

// (slot-7 WP-8, gap G14) PartitionFromNullities moved VERBATIM to the public
// Hecke/HeckeLinAlg.h; the call sites below resolve to hecke::PartitionFromNullities.

void emit_matrix_json(std::ostringstream& oss, const Matrix& M) {
    oss << "[";
    for (uint32_t r = 0; r < M.rows; ++r) {
        if (r > 0) oss << ",";
        oss << "[";
        for (uint32_t c = 0; c < M.cols; ++c) {
            if (c > 0) oss << ",";
            oss << M.data[r][c];
        }
        oss << "]";
    }
    oss << "]";
}

void emit_poly_json(std::ostringstream& oss, const FpPoly& P) {
    // LOW -> HIGH degree (A-N4).
    oss << "[";
    for (size_t i = 0; i < P.coeffs.size(); ++i) {
        if (i > 0) oss << ",";
        oss << P.coeffs[i];
    }
    oss << "]";
}

std::string indent(uint32_t offset) { return std::string(offset, ' '); }

} // namespace

// ---------------------------------------------------------------------------
// (E0b-S, slot 3 WP-3) The phase names and the test-only cancellation hook.
// ---------------------------------------------------------------------------

const char* to_string(LocalityPhase ph) {
    switch (ph) {
        case LocalityPhase::Linear: return "linear";
        case LocalityPhase::Berlekamp: return "berlekamp";
        case LocalityPhase::ResidueSearch: return "residue_search";
    }
    return "linear";
}

namespace {
thread_local uint32_t g_locality_step_limit = 0;   // 0 = off; TEST-ONLY
thread_local uint32_t g_locality_steps = 0;
thread_local uint32_t g_strict_seeded_trials = kStrictSeededFromBudget;  // TEST-ONLY
}  // namespace

void SetLocalityStepLimit(uint32_t limit) { g_locality_step_limit = limit; }
uint32_t LocalityStepLimit() { return g_locality_step_limit; }
uint32_t LocalityStepsTaken() { return g_locality_steps; }
void ResetLocalitySteps() { g_locality_steps = 0; }
void SetStrictResidueSeededTrials(uint32_t n) { g_strict_seeded_trials = n; }
uint32_t StrictResidueSeededTrials() { return g_strict_seeded_trials; }

namespace {
// One step boundary.  Returns true when the run must stop cooperatively.
bool LocalityStepAndCheck() {
    ++g_locality_steps;
    return g_locality_step_limit != 0 && g_locality_steps >= g_locality_step_limit;
}
}  // namespace

// ---------------------------------------------------------------------------
// HeckeModuleDecomposition
// ---------------------------------------------------------------------------

HeckeModuleDecomposition::HeckeModuleDecomposition(
    std::vector<std::pair<uint64_t, Matrix>> ops, uint32_t ell)
    : ops_(std::move(ops)), ell_(ell) {
    // (slot-8 WP-3) [E] sec:ring (b): F_ell[X] and a primary decomposition are field notions.
    RequireFieldCoefficientRing("HeckeModuleDecomposition");
}

void HeckeModuleDecomposition::require_computed(const char* who) const {
    if (!computed_) {
        throw std::runtime_error(std::string("HeckeModuleDecomposition::") + who +
                                 ": compute() has not been run");
    }
}

const Matrix& HeckeModuleDecomposition::op_matrix(uint64_t p) const {
    for (const auto& op : ops_) {
        if (op.first == p) return op.second;
    }
    throw std::invalid_argument("HeckeModuleDecomposition: unknown operator prime p = " +
                                std::to_string(p));
}

void HeckeModuleDecomposition::compute() {
    if (computed_) return;
    if (ops_.empty()) {
        throw std::invalid_argument("HeckeModuleDecomposition: empty operator family");
    }
    // (E0b, slot 2 WP-1) The `locality` setting decides how the refinement
    // loop certifies its blocks ([X] Sec. 1.5).
    //
    // (E0b-S, slot 3 WP-3) STRICT MODE IS NOW IMPLEMENTED ([X] Sec. 3.1
    // "Strict mode" steps 1-6, [E] Cert. `sec:localcert` "Strict mode"):
    // after the SAME bounded linear witness family budget mode uses, a block
    // that is still undecided enters the terminating Berlekamp route, which
    // either certifies it or splits it -- no search, and every split strictly
    // reduces the dimension.  The two modes therefore agree BYTE FOR BYTE
    // wherever budget mode certifies, which is every block of the golden grid
    // (`HeckeLocalityValidation` arms 5 and 7 assert exactly that), and
    // `locality = strict` is never the default and never runs on the golden
    // path.
    {
        const MathSettings& S = math_settings();
        loc_ = LocalityState();
        loc_.mode = S.locality.mode;
        loc_.budget = S.locality.budget;
        loc_.budget_history.push_back(S.locality.budget);
    }
    if (ops_.size() == 1) {
        ComputeSingleOp();
    } else {
        ComputeMultiOp();
    }
}

// ---------------------------------------------------------------------------
// Stage S10 -- single-operator primary decomposition (k = 1). Body unchanged
// from S10 (moved verbatim into this method; see HeckeModuleDecomposition.h).
// ---------------------------------------------------------------------------
void HeckeModuleDecomposition::ComputeSingleOp() {
    blocks_.clear(); // in case a previous compute() attempt threw midway
    loc_.blocks.clear();
    const uint64_t p = ops_[0].first;
    const Matrix& A = ops_[0].second;
    if (A.rows != A.cols) {
        throw std::invalid_argument("HeckeModuleDecomposition: operator matrix not square");
    }
    if (A.p != ell_) {
        throw std::invalid_argument("HeckeModuleDecomposition: operator modulus != ell");
    }
    d_ = A.rows;

    if (d_ == 0) { // H^q = 0: empty decomposition, trivially certified.
        U_ = Matrix(0, 0, ell_);
        U_inv_ = Matrix(0, 0, ell_);
        computed_ = true;
        return;
    }

    // charpoly(A) = prod pi_j^{m_j}; FactorPoly certifies (product identity +
    // deterministic irreducibility proofs) before returning.
    const FpPoly cp = CharPolyBerkowitz(A);
    const FpPolyFactorization fac = FactorPoly(cp);

    std::vector<Matrix> block_bases;
    for (const FpPolyFactor& factor : fac.factors) {
        const FpPoly& pi = factor.pi;
        const uint32_t m = factor.multiplicity;
        const uint32_t f = static_cast<uint32_t>(pi.degree());
        const uint32_t dim_j = m * f;

        // Nullities of pi(A)^r on the full space equal those on the primary
        // component (pi(A) is invertible on every other component).
        const Matrix Bpi = EvalPolyAtMatrix(pi, A);
        std::vector<Matrix> pows;
        pows.reserve(m + 1);
        pows.emplace_back(d_, ell_); // identity
        for (uint32_t r = 1; r <= m; ++r) pows.push_back(pows.back() * Bpi);
        std::vector<uint32_t> nullities(m + 1);
        for (uint32_t r = 0; r <= m; ++r) nullities[r] = d_ - RankOf(pows[r]);
        if (nullities[m] != dim_j) {
            throw std::runtime_error("HeckeModuleDecomposition: primary component of "
                                     "unexpected dimension");
        }
        // Nilpotency exponent e on the component = first stabilization.
        uint32_t e = m;
        while (e > 1 && nullities[e - 1] == nullities[e]) --e;
        std::vector<uint32_t> trimmed(nullities.begin(), nullities.begin() + e + 1);
        std::vector<uint32_t> partition =
            PartitionFromNullities(trimmed, f, "HeckeModuleDecomposition(partition)");
        uint32_t part_sum = 0;
        for (uint32_t s : partition) part_sum += s;
        if (part_sum != m || partition.empty() || partition[0] != e) {
            throw std::runtime_error("HeckeModuleDecomposition: partition inconsistent with "
                                     "multiplicity/exponent");
        }

        // Primary component: ker pi(A)^{m_j} via the existing Matrix kernel.
        Matrix K = KernelAsMatrix(pows[m]);
        if (K.cols != dim_j) {
            throw std::runtime_error("HeckeModuleDecomposition: kernel dimension mismatch");
        }
        // Restriction of A to the component (exact solve = invariance check).
        Matrix Aj = SolveColumnsExact(K, A * K, "HeckeModuleDecomposition(component)");

        // Within-block cyclic-vector / quotient-chain basis.
        std::vector<uint32_t> sizes;
        Matrix R = PrimaryRCFBasis(Aj, pi, sizes);
        if (sizes != partition) {
            throw std::runtime_error("HeckeModuleDecomposition: cyclic-chain heights disagree "
                                     "with the rank-derived Jordan partition");
        }

        PrimaryBlock block;
        block.dim = dim_j;
        block.residue_degree = f;
        block.basis_columns = K * R;
        block.char_poly.emplace(p, FpPoly::pow(pi, m));
        block.min_poly.emplace(p, FpPoly::pow(pi, e));
        block.jordan_partition.emplace(p, partition);
        // Stage S11 fields: for a single operator the residue degree f is
        // trivially certified by the operator itself (witness = {(p, 1)}).
        block.residue_degree_certified = true;
        block.residue_witness = { {p, 1u} };

        // (E0b, slot 2 WP-1) THE k = 1 ANALYTIC PATH ([X] Sec. 3.1, [XA m17]).
        // F_ell[t]/(pi^e) is a local ring, so the block is local for free and
        // NOTHING is computed: w_hat -- which would cost N*ceil(log2 ell)
        // dim x dim products on EVERY single-operator decomposition of the
        // campaign, the golden harness and `--bench decomp` -- is never
        // formed.  The character is X mod pi_w, the REDUCED representative of
        // degree < f (the constant alpha at f = 1), which is what makes the
        // analytic and the matrix path agree at f = 1 (v1.5.1, slot-1 WP-1).
        block.local_certified = true;
        block.witness_poly = pi;
        block.w_hat_exponent = TeichmuellerExponent(ell_, f, dim_j);
        block.character.emplace(p, FpPoly::mod(FpPoly::X(ell_), pi));

        LocalityBlockState bs;
        bs.dim = block.dim;
        bs.basis_columns = block.basis_columns;
        bs.local = true;
        bs.attempts = 1;
        bs.residue_degree_lower_bound = f;
        bs.witness_poly = block.witness_poly;
        bs.w_hat_exponent = block.w_hat_exponent;
        bs.character = block.character;
        bs.witness = block.residue_witness;
        bs.combos_tried.push_back(block.residue_witness);
        bs.residue_degree_certified = true;
        bs.residue_witness = block.residue_witness;
        loc_.blocks.push_back(std::move(bs));

        block_bases.push_back(block.basis_columns);
        blocks_.push_back(std::move(block));
    }
    loc_.complete = true;

    uint32_t total = 0;
    for (const PrimaryBlock& b : blocks_) total += b.dim;
    if (total != d_) {
        throw std::runtime_error("HeckeModuleDecomposition: block dimensions do not sum to d");
    }

    U_ = HStack(block_bases, d_, ell_);
    Matrix Ucopy = U_;
    if (!Ucopy.is_invertible()) {
        throw std::runtime_error("HeckeModuleDecomposition: change of basis not invertible");
    }
    U_inv_ = Ucopy.inverse();

    // Internal structural assertion (compute-time; certify() re-derives all
    // of this independently): the conjugated matrix is exactly block-diagonal
    // and each diagonal block is the direct sum of companion(pi^{s_t}).
    const Matrix C = U_inv_ * A * U_;
    std::vector<uint32_t> dims;
    for (const PrimaryBlock& b : blocks_) dims.push_back(b.dim);
    if (!IsBlockDiagonal(C, dims)) {
        throw std::runtime_error("HeckeModuleDecomposition: conjugated matrix not block-diagonal");
    }
    uint32_t off = 0;
    for (size_t j = 0; j < blocks_.size(); ++j) {
        const FpPoly& pi = fac.factors[j].pi;
        const std::vector<uint32_t>& partition = blocks_[j].jordan_partition.at(p);
        uint32_t sub_off = off;
        for (uint32_t s : partition) {
            const FpPoly pis = FpPoly::pow(pi, s);
            const Matrix comp = CompanionOfPoly(pis);
            for (uint32_t i = 0; i < comp.rows; ++i) {
                for (uint32_t jj = 0; jj < comp.cols; ++jj) {
                    if (C.data[sub_off + i][sub_off + jj] != comp.data[i][jj]) {
                        throw std::runtime_error("HeckeModuleDecomposition: conjugated block is "
                                                 "not in primary rational canonical form");
                    }
                }
            }
            sub_off += comp.rows;
        }
        off += blocks_[j].dim;
    }

    computed_ = true;
}

// ---------------------------------------------------------------------------
// Stage S11 -- simultaneous MaxSpec(T) refinement for a commuting family
// (k > 1). See the header comment (HeckeModuleDecomposition.h) for the full
// algorithm statement and the residue-degree witness search.
// ---------------------------------------------------------------------------
void HeckeModuleDecomposition::ComputeMultiOp() {
    blocks_.clear();

    // ---- Shape / modulus validation across the whole family. ----
    const Matrix& A0 = ops_[0].second;
    if (A0.rows != A0.cols) {
        throw std::invalid_argument("HeckeModuleDecomposition: operator matrix not square");
    }
    if (A0.p != ell_) {
        throw std::invalid_argument("HeckeModuleDecomposition: operator modulus != ell");
    }
    d_ = A0.rows;
    {
        std::vector<uint64_t> seen;
        for (const auto& op : ops_) {
            const Matrix& A = op.second;
            if (A.rows != A.cols || A.rows != d_) {
                throw std::invalid_argument("HeckeModuleDecomposition: mismatched operator "
                                            "dimensions in the family");
            }
            if (A.p != ell_) {
                throw std::invalid_argument("HeckeModuleDecomposition: operator modulus != ell");
            }
            if (std::find(seen.begin(), seen.end(), op.first) != seen.end()) {
                throw std::invalid_argument("HeckeModuleDecomposition: duplicate Hecke prime p = " +
                                            std::to_string(op.first) + " in the family");
            }
            seen.push_back(op.first);
        }
    }

    if (d_ == 0) { // H^q = 0: empty decomposition, trivially certified.
        U_ = Matrix(0, 0, ell_);
        U_inv_ = Matrix(0, 0, ell_);
        loc_.complete = true;
        computed_ = true;
        return;
    }

    // ---- The partition: start with the whole space, refine to the MaxSpec(T)
    // fixed point, then run the E0b locality stage (which may split further
    // and restart). ----
    std::vector<Matrix> bases;
    bases.push_back(Matrix(d_, ell_)); // basis = I_d, i.e. the whole space
    RunLocalityStage(bases, std::vector<LocalityBlockState>(), loc_.budget);

    FinalizeChangeOfBasis();
    computed_ = true;
}

// ---------------------------------------------------------------------------
// The op-by-op refinement passes.  Mathematically ONE pass through the family
// already reaches the MaxSpec(T) fixed point (a subspace of a pi-primary space
// stays pi-primary under further invariant-subspace refinement, by
// Cayley-Hamilton -- see the header comment); the outer loop over full passes
// is a DEFENSIVE measure honoring the plan's "repeat until a fixed point ...
// with a defensive pass cap" language.
//
// It is BASIS-PRESERVING on a partition that is already at the fixed point:
// there charpoly(t_p|block) = pi^m, so Ppow = pi(Asub)^m = 0 by
// Cayley-Hamilton and KernelAsMatrix(0) is the identity.  That is what lets
// the E0b stage restart the whole loop after a split without disturbing the
// blocks it did not touch.
// ---------------------------------------------------------------------------
void HeckeModuleDecomposition::RefineToFixedPoint(std::vector<Matrix>& bases) const {
    const uint32_t kPassCap = d_ + static_cast<uint32_t>(ops_.size()) + 2;
    bool reached_fixed_point = false;
    for (uint32_t pass = 0; pass < kPassCap && !reached_fixed_point; ++pass) {
        bool any_split_this_pass = false;
        for (const auto& op : ops_) {
            std::vector<Matrix> refined;
            for (const Matrix& B : bases) {
                const uint32_t m_dim = B.cols;
                // Exact solve: restricts T_{p_j} to the current block. Its
                // solvability IS the runtime certificate that T_{p_j}
                // commutes with everything used to build this block so far
                // (an operator that failed to commute would generically make
                // the system inconsistent, and SolveColumnsExact throws).
                Matrix Asub = SolveColumnsExact(B, op.second * B,
                    "HeckeModuleDecomposition(S11 refinement)");
                const FpPoly cp = CharPolyBerkowitz(Asub);
                const FpPolyFactorization fac = FactorPoly(cp);
                if (fac.factors.size() > 1) any_split_this_pass = true;
                for (const FpPolyFactor& factor : fac.factors) {
                    const uint32_t mj = factor.multiplicity;
                    const uint32_t fj = static_cast<uint32_t>(factor.pi.degree());
                    Matrix Ppow(m_dim, ell_); // identity
                    const Matrix Bpi = EvalPolyAtMatrix(factor.pi, Asub);
                    for (uint32_t r = 0; r < mj; ++r) Ppow = Ppow * Bpi;
                    Matrix K = KernelAsMatrix(Ppow);
                    if (K.cols != mj * fj) {
                        throw std::runtime_error("HeckeModuleDecomposition: S11 refinement "
                                                 "primary component has unexpected dimension");
                    }
                    refined.push_back(B * K);
                }
            }
            bases = std::move(refined);
        }
        if (!any_split_this_pass) reached_fixed_point = true;
    }
    if (!reached_fixed_point) {
        throw std::runtime_error("HeckeModuleDecomposition: simultaneous refinement did not "
                                 "reach a fixed point within the defensive pass cap (a genuinely "
                                 "commuting exact family should never hit this)");
    }
}

// ---------------------------------------------------------------------------
// (hecke-extensions phase E0b, slot 2 WP-1) THE LOCALITY STAGE
// ([X] Sec. 3.1 steps 1-3; [E] Cert. 5.2, BOTH clauses).
//
// BEFORE E0b the loop stopped at the MaxSpec(T) fixed point and accepted a
// block as soon as SOME witness w had a char-poly whose squarefree part was
// ONE irreducible of degree f = lcm_j deg(pi_j).  That predicate certifies the
// RESIDUE DEGREE and says nothing about LOCALITY: [E] Claim 3.1's pair
// t_1 = (alpha, alpha^ell), t_2 = (alpha, alpha) on F_{ell^2}^2 passes it with
// T_S/rad = F_{ell^2} x F_{ell^2}, i.e. the reported "block" is really two.
//
// AFTER E0b, per block, over ONE fixed candidate sequence (the sequence the
// R3.3 `attempt index` indexes -- see LocalityState in the header):
//   * a candidate whose char-poly has >= 2 DISTINCT irreducible factors SPLITS
//     the block, by the primary decomposition of that candidate, sub-blocks
//     inserted AT THE PARENT'S POSITION in CanonicalLess order (which is the
//     order FactorPoly returns, exactly as the refinement loop above uses for
//     its own splits) -- and the whole stage restarts.  Every split strictly
//     reduces block dimensions, so this terminates.
//   * a candidate passing clause (1) is subjected to clauses (2) and (3):
//     the Teichmueller lift w_hat = w^(ell^N) with pi_w(w_hat) = 0, and a
//     character P_p of degree < f with t_p - P_p(w_hat) NILPOTENT for every
//     operator -- the trivial single-operator witness included (there
//     n_j = t_j - w_hat, not 0).  Only then is the block local.
//   * when the budget ([X] Sec. 1.5 `locality`, default 64) is exhausted with
//     neither a certificate nor a split, the block is emitted
//     local_certified = false with residue_degree as a proven LOWER BOUND, the
//     NOTE names locality, and the state is resumable.  THE LOOP NEVER SPINS.
//
// WHAT DOES NOT MOVE.  `residue_degree_certified` / `residue_witness` keep
// their pre-E0b semantics EXACTLY: they are set by the FIRST candidate that
// passes clause (1), whether or not the block turns out local, and are never
// overwritten by a later candidate.  So on a grid where every block is local
// and the first candidate already carries a full certificate -- which is the
// measured slot-1 verdict, `nonlocal = 0`, `witness_changes = 0` over all 348
// golden blocks -- this stage takes NO new branch and every golden hash is
// unchanged ([X] Sec. 3.3's Claim).
// ---------------------------------------------------------------------------
void HeckeModuleDecomposition::RunLocalityStage(std::vector<Matrix>& bases,
                                                std::vector<LocalityBlockState> carried,
                                                uint32_t budget) {
    ENGINE_CHECK_MSG(budget >= 1, "HeckeModuleDecomposition: locality budget must be >= 1");
    // Every split strictly increases the block count, which is bounded by d_.
    const uint32_t kSplitCap = d_ + 1;
    const bool strict = (loc_.mode == LocalityMode::Strict);
    // (E0b-S) Cooperative cancellation: once it fires, no further candidate is
    // evaluated on ANY block, every undecided block is emitted `local = false`
    // and the state is left resumable ([X] Sec. 1.3.1 R3.3 item 5).
    bool interrupted = false;

    for (uint32_t round = 0;; ++round) {
        ENGINE_CHECK_MSG(round <= kSplitCap,
                         "HeckeModuleDecomposition: locality splitting did not terminate");
        const size_t before = bases.size();
        RefineToFixedPoint(bases);
        // A refinement that changed the partition invalidates every carried
        // attempt index (the blocks are new objects).
        if (bases.size() != before || carried.size() != bases.size()) carried.clear();

        blocks_.clear();
        loc_.blocks.clear();
        bool split_happened = false;

        for (size_t bi = 0; bi < bases.size(); ++bi) {
            PrimaryBlock block;
            block.dim = bases[bi].cols;
            block.basis_columns = bases[bi];

            std::map<uint64_t, FpPoly> pi_by_p;
            std::map<uint64_t, Matrix> Asub_by_p;

            // ---- Per-block, per-operator char-poly / min-poly / Jordan data,
            // recomputed directly from the (basis-independent) block subspace.
            // Unchanged from the pre-E0b finalize loop. ----
            for (const auto& op : ops_) {
                const uint64_t p = op.first;
                Matrix Asub = SolveColumnsExact(block.basis_columns,
                    op.second * block.basis_columns,
                    "HeckeModuleDecomposition(S11 finalize)");

                const FpPoly cp = CharPolyBerkowitz(Asub);
                const FpPolyFactorization fac = FactorPoly(cp);
                if (fac.factors.size() != 1) {
                    throw std::runtime_error("HeckeModuleDecomposition: block not T_" +
                                             std::to_string(p) + "-primary at the claimed fixed "
                                             "point (internal algorithm defect)");
                }
                const FpPoly& pi = fac.factors[0].pi;
                const uint32_t mult = fac.factors[0].multiplicity;
                const uint32_t f = static_cast<uint32_t>(pi.degree());
                if (mult * f != block.dim) {
                    throw std::runtime_error("HeckeModuleDecomposition: block dimension "
                                             "inconsistent with T_" + std::to_string(p) +
                                             "'s char-poly");
                }

                // Nullity sequence of pi(Asub)^r -> nilpotency exponent e ->
                // Jordan partition (identical technique to the S10 single-op path).
                const Matrix Bpi = EvalPolyAtMatrix(pi, Asub);
                std::vector<Matrix> pows;
                pows.reserve(mult + 1);
                pows.emplace_back(block.dim, ell_); // identity
                for (uint32_t r = 1; r <= mult; ++r) pows.push_back(pows.back() * Bpi);
                std::vector<uint32_t> nullities(mult + 1);
                for (uint32_t r = 0; r <= mult; ++r) nullities[r] = block.dim - RankOf(pows[r]);
                if (nullities[mult] != block.dim) {
                    throw std::runtime_error("HeckeModuleDecomposition: primary block not fully "
                                             "annihilated by pi_p^mult for T_" +
                                             std::to_string(p));
                }
                uint32_t e = mult;
                while (e > 1 && nullities[e - 1] == nullities[e]) --e;
                std::vector<uint32_t> trimmed(nullities.begin(), nullities.begin() + e + 1);
                std::vector<uint32_t> partition =
                    PartitionFromNullities(trimmed, f, "HeckeModuleDecomposition(S11 partition)");

                block.char_poly.emplace(p, FpPoly::pow(pi, mult));
                block.min_poly.emplace(p, FpPoly::pow(pi, e));
                block.jordan_partition.emplace(p, partition);

                pi_by_p.emplace(p, pi);
                Asub_by_p.emplace(p, std::move(Asub));
            }

            // ---- Residue degree f = lcm_j(f_j). ----
            uint32_t f_lcm = 1;
            for (const auto& kv : pi_by_p) {
                f_lcm = Lcm32(f_lcm, static_cast<uint32_t>(kv.second.degree()));
            }
            block.residue_degree = f_lcm;
            block.residue_degree_certified = false;

            // ---- The E0b certified witness search over the one fixed
            // candidate sequence. ----
            LocalityBlockState st;
            st.dim = block.dim;
            st.basis_columns = block.basis_columns;
            st.residue_degree_lower_bound = f_lcm;
            uint32_t start_at = 0;
            if (bi < carried.size() && carried[bi].dim == block.dim &&
                carried[bi].basis_columns == block.basis_columns) {
                // A resume: continue at the stored attempt index, carrying the
                // residue-degree claim the prefix already established so that
                // the continuation reproduces a fresh run exactly.
                start_at = carried[bi].attempts;
                st.attempts = carried[bi].attempts;
                st.combos_tried = carried[bi].combos_tried;
                st.residue_degree_certified = carried[bi].residue_degree_certified;
                st.residue_witness = carried[bi].residue_witness;
                // (E0b-S, slot 3 WP-3) ... AND the certificate itself, where
                // the prefix already found one.  Before this the continuation
                // restarted the search of an ALREADY CERTIFIED block at its
                // stored attempt index and so skipped the very candidate that
                // had certified it, turning a local block non-local -- latent
                // whenever a partial state mixed certified and uncertified
                // blocks, which is exactly what strict mode's interruption
                // produces (WP-3 report, finding 1).  A carried certificate is
                // what a fresh run would re-derive, by determinism, so this is
                // resume-invariance, not a shortcut.
                st.local = carried[bi].local;
                st.witness = carried[bi].witness;
                st.witness_poly = carried[bi].witness_poly;
                st.w_hat_exponent = carried[bi].w_hat_exponent;
                st.character = carried[bi].character;
                st.witness_matrix = carried[bi].witness_matrix;
                st.phase_reached = carried[bi].phase_reached;
            }

            const uint32_t N = TeichmuellerExponent(ell_, f_lcm, block.dim);
            std::vector<Matrix> split_bases;   // in the BLOCK's own coordinates

            // The candidate sequence, generated in order from index 0; entries
            // before `start_at` are generated (so the seeded stream advances
            // identically) but not evaluated.
            uint32_t idx = 0;
            const auto consider =
                [&](const std::vector<std::pair<uint64_t, uint32_t>>& combo) -> bool {
                // Returns true when the search must stop (certified, split, or
                // out of budget).
                const uint32_t here = idx++;
                if (here < start_at) return false;
                if (here >= budget) return true;
                // (E0b-S) A step boundary: BETWEEN two witness candidates,
                // never inside an elimination.
                if (LocalityStepAndCheck()) { interrupted = true; return true; }
                ++st.attempts;
                st.combos_tried.push_back(combo);

                Matrix W(block.dim, block.dim, ell_); // zero
                for (const auto& term : combo) W = W + Asub_by_p.at(term.first) * term.second;

                FpPoly cpW;
                std::vector<std::pair<FpPoly, uint32_t>> sfd;
                try {
                    cpW = CharPolyBerkowitz(W);
                    sfd = SquarefreeDecomposition(cpW);
                } catch (const std::exception&) {
                    return false;
                }
                const bool clause1 =
                    (sfd.size() == 1 &&
                     static_cast<uint32_t>(sfd[0].first.degree()) == f_lcm &&
                     CertifyIrreducible(sfd[0].first) == 0);

                if (clause1) {
                    // The pre-E0b acceptance predicate: the FIRST candidate
                    // passing it fixes residue_degree_certified /
                    // residue_witness, and nothing later overwrites them.
                    if (!st.residue_degree_certified) {
                        st.residue_degree_certified = true;
                        st.residue_witness = combo;
                    }
                    const FpPoly& pi_w = sfd[0].first;
                    std::map<uint64_t, FpPoly> character;
                    if (CertifyWitnessClauses23(W, pi_w, f_lcm, N, ops_, pi_by_p, Asub_by_p,
                                                block.dim, ell_, character,
                                                /*w_hat_out=*/nullptr) == kLocalityOk) {
                        st.local = true;
                        st.witness = combo;
                        st.witness_poly = pi_w;
                        st.w_hat_exponent = N;
                        st.character = std::move(character);
                        return true;
                    }
                    return false;   // clause (1) but not local: try the next candidate
                }

                // Not a witness.  Does it SPLIT the block?  Only now is the
                // full factorization worth its cost, which is why the golden
                // path -- where candidate 0 certifies -- never pays it.
                try {
                    const FpPolyFactorization facW = FactorPoly(cpW);
                    if (facW.factors.size() > 1) {
                        for (const FpPolyFactor& fw : facW.factors) {   // CanonicalLess order
                            const uint32_t mj = fw.multiplicity;
                            const uint32_t fj = static_cast<uint32_t>(fw.pi.degree());
                            Matrix Ppow(block.dim, ell_); // identity
                            const Matrix Wpi = EvalPolyAtMatrix(fw.pi, W);
                            for (uint32_t r = 0; r < mj; ++r) Ppow = Ppow * Wpi;
                            Matrix K = KernelAsMatrix(Ppow);
                            if (K.cols != mj * fj) {
                                throw std::runtime_error("HeckeModuleDecomposition: locality "
                                                         "split component has unexpected "
                                                         "dimension");
                            }
                            split_bases.push_back(std::move(K));
                        }
                        return true;
                    }
                } catch (const std::exception&) {
                    split_bases.clear();
                }
                return false;
            };

            // A block whose certificate a resume carried in is already
            // decided; the search must not run again (see the carry above).
            if (!st.local && !interrupted) {
                bool stop = false;
                for (const auto& op : ops_) {
                    if (static_cast<uint32_t>(pi_by_p.at(op.first).degree()) != f_lcm) continue;
                    const std::vector<std::pair<uint64_t, uint32_t>> trivial = { {op.first, 1u} };
                    if (consider(trivial)) { stop = true; break; }
                }
                if (!stop) {
                    SplitMix64 rng(kResidueWitnessSeed ^
                                   (0x9E3779B97F4A7C15ull * (block.dim + 1u)) ^ ell_);
                    std::vector<std::pair<uint64_t, uint32_t>> combo;
                    for (const auto& op : ops_) combo.emplace_back(op.first, 1u); // all-ones
                    for (uint32_t attempt = 0; attempt < kResidueWitnessMaxAttempts; ++attempt) {
                        if (attempt > 0) {
                            for (auto& term : combo) {
                                term.second = (ell_ > 1) ? (1u + rng.next_mod(ell_ - 1)) : 0u;
                            }
                        }
                        if (consider(combo)) break;
                        if (idx >= budget) break;
                    }
                }
            }

            // ---- (E0b-S, slot 3 WP-3) STRICT MODE'S TERMINATING ROUTE.
            // Entered ONLY when the linear family of `budget` candidates has
            // been exhausted with neither a certificate nor a split -- so on
            // every block budget mode decides, strict mode has already
            // produced the SAME bytes and this branch is dead ([X] Sec. 3.1
            // step 4, [E] Cert. `sec:localcert` "Strict mode"). ----
            if (strict && !interrupted && !st.local && split_bases.empty()) {
                const uint32_t out = StrictLocalityRoute(block, pi_by_p, Asub_by_p, f_lcm,
                                                         budget, st, split_bases);
                if (out == kStrictInterrupted) {
                    interrupted = true;
                    split_bases.clear();
                } else if (out == kStrictSplit) {
                    ENGINE_CHECK_MSG(split_bases.size() >= 2,
                                     "HeckeModuleDecomposition: the Berlekamp split produced "
                                     "fewer than two sub-blocks");
                    if (loc_.phase_reached < LocalityPhase::Berlekamp) {
                        loc_.phase_reached = LocalityPhase::Berlekamp;
                    }
                } else {
                    ENGINE_CHECK_MSG(st.local,
                                     "HeckeModuleDecomposition: the strict locality route "
                                     "returned `certified` without a certificate");
                    if (loc_.phase_reached < LocalityPhase::ResidueSearch) {
                        loc_.phase_reached = LocalityPhase::ResidueSearch;
                    }
                }
            }

            if (!split_bases.empty()) {
                // Replace this block by its sub-blocks AT THE PARENT'S
                // POSITION and restart the stage on the new partition.
                std::vector<Matrix> next;
                for (size_t j = 0; j < bi; ++j) next.push_back(bases[j]);
                for (const Matrix& K : split_bases) next.push_back(bases[bi] * K);
                for (size_t j = bi + 1; j < bases.size(); ++j) next.push_back(bases[j]);
                bases = std::move(next);
                carried.clear();
                ++loc_.splits;
                split_happened = true;
                break;
            }

            block.residue_degree_certified = st.residue_degree_certified;
            block.residue_witness = st.residue_witness;
            block.local_certified = st.local;
            block.witness_poly = st.witness_poly;
            block.w_hat_exponent = st.w_hat_exponent;
            block.character = st.character;
            block.witness_matrix = st.witness_matrix;      // (E0b-S) empty in budget mode
            block.phase_reached = st.phase_reached;

            if (!block.residue_degree_certified) {
                std::cout << "HeckeModuleDecomposition: NOTE -- residue degree of a dim-"
                          << block.dim << " block could only be bounded below (f = " << f_lcm
                          << "), not certified equal to the true residue field degree, within the "
                             "fixed witness-search budget (" << budget
                          << " attempts). Reported as residue_degree_lower_bound in to_json(); "
                             "interval [" << f_lcm << ", " << block.dim << "]. Flagged for human "
                             "review." << std::endl;
            }
            if (!block.local_certified) {
                // [X] Sec. 3.1 step 3: never silently.  The block's LOCALITY
                // -- not merely its residue degree -- is unproven under this
                // budget; the state is resumable with a higher one.
                //
                // (E0b-S) Under STRICT mode the route terminates, so the only
                // way here is a cooperative cancellation: the NOTE says so,
                // and the state is resumable at the SAME budget.
                if (strict) {
                    std::cout << "HeckeModuleDecomposition: NOTE -- LOCALITY of a dim-"
                              << block.dim << " block is NOT certified because the strict "
                                 "locality route was INTERRUPTED at a step boundary ([X] "
                                 "Sec. 1.3.1 R3.3 item 5). Emitted with \"local\": false and "
                                 "residue_degree_lower_bound = " << f_lcm
                              << "; the locality state is resumable and completes to the same "
                                 "partition and certificate as an uninterrupted run."
                              << std::endl;
                } else {
                    std::cout << "HeckeModuleDecomposition: NOTE -- LOCALITY of a dim-"
                              << block.dim << " block is NOT certified within the locality budget ("
                              << budget << " witness candidates, [E] Cert. 5.2): neither a full "
                                 "certificate nor a splitting witness was found. Emitted with "
                                 "\"local\": false and residue_degree_lower_bound = " << f_lcm
                              << "; the locality state is resumable with a higher budget. Flagged "
                                 "for human review." << std::endl;
                }
            }

            blocks_.push_back(std::move(block));
            loc_.blocks.push_back(std::move(st));
        }

        if (!split_happened) break;
    }

    loc_.complete = true;
    for (const LocalityBlockState& b : loc_.blocks) {
        if (!b.local) { loc_.complete = false; break; }
    }
    // (E0b-S) Recomputed by every run, so a resume that completes clears it.
    loc_.interrupted = interrupted;
}


// ---------------------------------------------------------------------------
// (hecke-extensions phase E0b-S, slot 3 WP-3) STRICT MODE'S TERMINATING ROUTE
// -- [X] Sec. 3.1 "Strict mode" steps 1-3, [E] Cert. `sec:localcert` "Strict
// mode" (1)-(3).  NO SEARCH decides anything here: locality is a kernel
// computation, and only the *witness* (which certifies an already-decided
// local block) is drawn from a fixed-seed sequence with a deterministic
// lexicographic fallback.
//
// WHY IT TERMINATES.
//   (1) rad A_S = ker(Frob^N), ell^N >= D.  On a COMMUTATIVE F_ell-algebra
//       x -> x^ell is F_ell-linear, so this is one kernel of one dim A_S
//       square matrix -- no search, no factorisation.
//   (2) Abar := A_S / rad A_S = prod_i F_{ell^{f_i}} and the Frobenius-fixed
//       subalgebra B := ker(Frob_Abar - id) = F_ell^m has dim B = m
//       (Berlekamp).  dim B = 1 IS locality.  dim B >= 2 splits the block by
//       the primary decomposition of a lift of the first NON-SCALAR RREF
//       basis element of B -- a fixed choice -- into >= 2 non-zero pieces
//       (V is faithful over A_S, so no primitive idempotent of Abar lifts to
//       an element acting as 0), and every split strictly reduces dim V.
//   (3) With dim B = 1, Abar is a field of degree f, and a finite field is
//       never the union of its proper subfields, so a witness whose residue
//       generates Abar EXISTS in A_S even when the F_ell-span of the
//       generators' residues lies in a union of proper subfields (that is
//       exactly the (2, 30) covering example).  Candidates are Abar's own
//       coordinate vectors, lifted by the same coordinates; the failure
//       probability of one candidate is <= omega(f) ell^{-f/2}, and the
//       lexicographic enumeration of all ell^f of them is the deterministic
//       fallback.
//
// AN IDENTITY, ASSERTED.  On a block with dim B = 1 the residues of the
// generators generate Abar as a FIELD, so Abar = F_ell(tbar_p : p) is the
// compositum of the F_{ell^{deg pi_p}} and therefore
//      dim_{F_ell} Abar = lcm_p deg pi_p = f_lcm,
// the very number budget mode reports as `residue_degree_lower_bound`.  The
// bound is thus SHARP on a local block, which is why strict mode never has to
// revise `residue_degree` and why certify()'s code 19 keeps holding.
// ---------------------------------------------------------------------------
uint32_t HeckeModuleDecomposition::StrictLocalityRoute(
    const PrimaryBlock& block,
    const std::map<uint64_t, FpPoly>& pi_by_p,
    const std::map<uint64_t, Matrix>& Asub_by_p,
    uint32_t f_lcm, uint32_t budget,
    LocalityBlockState& st,
    std::vector<Matrix>& split_bases) {
    const uint32_t D = block.dim;
    ENGINE_CHECK_MSG(D >= 1, "StrictLocalityRoute: empty block");

    // ---- step 1: A_S, the Frobenius matrix, and rad A_S = ker Frob^N. ----
    std::vector<Matrix> gens;
    gens.reserve(ops_.size());
    for (const auto& op : ops_) gens.push_back(Asub_by_p.at(op.first));
    uint32_t dimA = 0;
    const Matrix closure = algebra_closure(gens, D, ell_, dimA);
    const std::vector<uint32_t> pivots = RrefPivots(closure);
    // The least N with ell^N >= D (TeichmuellerExponent with f = 0 is exactly
    // that; the Teichmueller exponent of the WITNESS is taken below with its
    // own f).
    const uint32_t N_rad = TeichmuellerExponent(ell_, 0, D);
    const Matrix F = frobenius_matrix(closure, D, ell_);
    const Matrix rad = frobenius_kernel(F, N_rad);

    // ---- the quotient Abar = A_S / rad, in coordinates. ----
    RrefRowSpan radspan(dimA, ell_);
    for (uint32_t c = 0; c < rad.cols; ++c) {
        std::vector<uint32_t> v(dimA, 0u);
        for (uint32_t r = 0; r < dimA; ++r) v[r] = rad.data[r][c];
        radspan.insert(std::move(v));
    }
    const uint32_t dim_rad = radspan.dim();
    ENGINE_CHECK_MSG(dim_rad == rad.cols,
                     "StrictLocalityRoute: the radical's spanning columns are dependent");
    std::vector<uint32_t> noncols;              // Abar's coordinates inside A_S's
    {
        std::vector<char> is_piv(dimA, 0);
        for (uint32_t q : radspan.pivots()) is_piv[q] = 1;
        for (uint32_t j = 0; j < dimA; ++j) {
            if (!is_piv[j]) noncols.push_back(j);
        }
    }
    const uint32_t fbar = static_cast<uint32_t>(noncols.size());   // dim Abar
    ENGINE_CHECK_MSG(fbar >= 1, "StrictLocalityRoute: A_S / rad A_S is the zero algebra");

    // Reduction mod rad, and the quotient map Q: F^dimA -> F^fbar.  A vector
    // supported on `noncols` is already reduced, so e_{noncols[j]} is a
    // SECTION of Q -- which is what makes the witness lift free of a solve.
    const std::vector<std::vector<uint32_t>>& rrows = radspan.rows();
    const std::vector<uint32_t>& rpiv = radspan.pivots();
    const auto reduce_mod_rad = [&](std::vector<uint32_t> v) {
        for (size_t i = 0; i < rrows.size(); ++i) {
            const uint32_t c = v[rpiv[i]];
            if (c == 0) continue;
            for (uint32_t j = rpiv[i]; j < dimA; ++j) {
                if (rrows[i][j] == 0) continue;
                v[j] = ModArithmetic::mod_sub(v[j],
                                              ModArithmetic::mod_mul(c, rrows[i][j], ell_), ell_);
            }
        }
        return v;
    };
    const auto Q = [&](const std::vector<uint32_t>& v) {
        const std::vector<uint32_t> r = reduce_mod_rad(v);
        std::vector<uint32_t> out(fbar, 0u);
        for (uint32_t j = 0; j < fbar; ++j) out[j] = r[noncols[j]];
        return out;
    };
    // The algebra element with the given A_S-coordinates.
    const auto elt_of_coords = [&](const std::vector<uint32_t>& c) {
        Matrix M(D, D, ell_);   // zero
        for (uint32_t i = 0; i < dimA; ++i) {
            if (c[i] == 0) continue;
            M = M + MatrixOfVec(closure.data[i], D, D, ell_) * c[i];
        }
        return M;
    };

    // ---- step 2: B = ker(Frob_Abar - id); dim B = m decides. ----
    Matrix Mq(fbar, dimA, ell_);
    for (uint32_t i = 0; i < dimA; ++i) {
        std::vector<uint32_t> col(dimA, 0u);
        for (uint32_t r = 0; r < dimA; ++r) col[r] = F.data[r][i];
        col[i] = ModArithmetic::mod_sub(col[i], 1u, ell_);
        const std::vector<uint32_t> qc = Q(col);
        for (uint32_t r = 0; r < fbar; ++r) Mq.data[r][i] = qc[r];
    }
    const Matrix Wk = KernelAsMatrix(Mq);
    ENGINE_CHECK_MSG(Wk.cols >= dim_rad,
                     "StrictLocalityRoute: rad A_S is not contained in the Berlekamp preimage");
    const uint32_t m = Wk.cols - dim_rad;
    ENGINE_CHECK_MSG(m >= 1, "StrictLocalityRoute: the Berlekamp subalgebra does not contain 1");

    if (m >= 2) {
        // --- the split.  The RREF of a span is unique, so "the first
        // non-scalar RREF basis element" is a fixed choice; the transform is
        // carried alongside so the lift needs no (rank-deficient) solve. ---
        std::vector<std::vector<uint32_t>> brows;   // RREF basis of B, in Abar coords
        std::vector<std::vector<uint32_t>> bcoef;   // each row as a combination of Wk's columns
        std::vector<uint32_t> bpiv;
        for (uint32_t c = 0; c < Wk.cols; ++c) {
            std::vector<uint32_t> v(dimA, 0u);
            for (uint32_t r = 0; r < dimA; ++r) v[r] = Wk.data[r][c];
            std::vector<uint32_t> row = Q(v);
            std::vector<uint32_t> coef(Wk.cols, 0u);
            coef[c] = 1u;
            // Reduce against the current rows (the RrefRowSpan::insert body,
            // with the transform carried in parallel).
            for (size_t i = 0; i < brows.size(); ++i) {
                const uint32_t f_i = row[bpiv[i]];
                if (f_i == 0) continue;
                for (uint32_t j = bpiv[i]; j < fbar; ++j) {
                    if (brows[i][j] == 0) continue;
                    row[j] = ModArithmetic::mod_sub(
                        row[j], ModArithmetic::mod_mul(f_i, brows[i][j], ell_), ell_);
                }
                for (uint32_t j = 0; j < Wk.cols; ++j) {
                    if (bcoef[i][j] == 0) continue;
                    coef[j] = ModArithmetic::mod_sub(
                        coef[j], ModArithmetic::mod_mul(f_i, bcoef[i][j], ell_), ell_);
                }
            }
            uint32_t piv = fbar;
            for (uint32_t j = 0; j < fbar; ++j) {
                if (row[j] != 0) { piv = j; break; }
            }
            if (piv == fbar) continue;
            const uint32_t inv = ModArithmetic::mod_inv(row[piv], ell_);
            if (inv != 1) {
                for (uint32_t j = piv; j < fbar; ++j) {
                    if (row[j] != 0) row[j] = ModArithmetic::mod_mul(row[j], inv, ell_);
                }
                for (uint32_t j = 0; j < Wk.cols; ++j) {
                    if (coef[j] != 0) coef[j] = ModArithmetic::mod_mul(coef[j], inv, ell_);
                }
            }
            for (size_t i = 0; i < brows.size(); ++i) {
                const uint32_t f_i = brows[i][piv];
                if (f_i == 0) continue;
                for (uint32_t j = piv; j < fbar; ++j) {
                    if (row[j] == 0) continue;
                    brows[i][j] = ModArithmetic::mod_sub(
                        brows[i][j], ModArithmetic::mod_mul(f_i, row[j], ell_), ell_);
                }
                for (uint32_t j = 0; j < Wk.cols; ++j) {
                    if (coef[j] == 0) continue;
                    bcoef[i][j] = ModArithmetic::mod_sub(
                        bcoef[i][j], ModArithmetic::mod_mul(f_i, coef[j], ell_), ell_);
                }
            }
            size_t at = brows.size();
            for (size_t i = 0; i < bpiv.size(); ++i) {
                if (bpiv[i] > piv) { at = i; break; }
            }
            const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(at);
            brows.insert(brows.begin() + off, std::move(row));
            bcoef.insert(bcoef.begin() + off, std::move(coef));
            bpiv.insert(bpiv.begin() + off, piv);
        }
        ENGINE_CHECK_MSG(brows.size() == m,
                         "StrictLocalityRoute: the Berlekamp subalgebra's rank disagrees with "
                         "dim ker(Frob_Abar - id) - dim rad");

        // The residue of 1, against which "non-scalar" is decided.
        const Matrix Id(D, ell_);
        const std::vector<uint32_t> qone =
            Q(AlgebraCoords(closure, pivots, Id, ell_, "StrictLocalityRoute(unit)"));
        uint32_t p0 = fbar;
        for (uint32_t j = 0; j < fbar; ++j) {
            if (qone[j] != 0) { p0 = j; break; }
        }
        ENGINE_CHECK_MSG(p0 < fbar, "StrictLocalityRoute: the unit has zero residue");
        size_t chosen = brows.size();
        for (size_t i = 0; i < brows.size(); ++i) {
            const uint32_t lam = ModArithmetic::mod_mul(
                brows[i][p0], ModArithmetic::mod_inv(qone[p0], ell_), ell_);
            bool scalar = true;
            for (uint32_t j = 0; j < fbar && scalar; ++j) {
                if (brows[i][j] != ModArithmetic::mod_mul(lam, qone[j], ell_)) scalar = false;
            }
            if (!scalar) { chosen = i; break; }
        }
        ENGINE_CHECK_MSG(chosen < brows.size(),
                         "StrictLocalityRoute: dim B >= 2 but every RREF basis element of B is "
                         "a scalar");

        // Lift: x = Wk * bcoef[chosen] in A_S's coordinates.
        std::vector<uint32_t> x(dimA, 0u);
        for (uint32_t r = 0; r < dimA; ++r) {
            uint32_t acc = 0;
            for (uint32_t c = 0; c < Wk.cols; ++c) {
                if (bcoef[chosen][c] == 0 || Wk.data[r][c] == 0) continue;
                acc = ModArithmetic::mod_add(
                    acc, ModArithmetic::mod_mul(bcoef[chosen][c], Wk.data[r][c], ell_), ell_);
            }
            x[r] = acc;
        }
        const Matrix w = elt_of_coords(x);

        // The primary decomposition of w on V -- the SAME machinery the linear
        // phase's splitting branch uses, in CanonicalLess factor order.
        const FpPolyFactorization facW = FactorPoly(CharPolyBerkowitz(w));
        ENGINE_CHECK_MSG(facW.factors.size() >= 2,
                         "StrictLocalityRoute: a non-scalar element of the Berlekamp subalgebra "
                         "has a single irreducible char-poly factor");
        for (const FpPolyFactor& fw : facW.factors) {
            const uint32_t mj = fw.multiplicity;
            const uint32_t fj = static_cast<uint32_t>(fw.pi.degree());
            Matrix Ppow(D, ell_);   // identity
            const Matrix Wpi = EvalPolyAtMatrix(fw.pi, w);
            for (uint32_t r = 0; r < mj; ++r) Ppow = Ppow * Wpi;
            Matrix K = KernelAsMatrix(Ppow);
            ENGINE_CHECK_MSG(K.cols == mj * fj,
                             "StrictLocalityRoute: a Berlekamp split component has unexpected "
                             "dimension");
            split_bases.push_back(std::move(K));
        }
        st.phase_reached = LocalityPhase::Berlekamp;
        return kStrictSplit;
    }

    // ---- step 3: dim B = 1, so V IS local; find a witness over Abar. ----
    ENGINE_CHECK_MSG(fbar == f_lcm,
                     "StrictLocalityRoute: dim(A_S / rad A_S) != lcm_p deg(pi_p) on a block with "
                     "dim B = 1 (the compositum identity of [E] Cert. `sec:localcert`)");
    const uint32_t f = fbar;
    const uint32_t N = TeichmuellerExponent(ell_, f, D);

    const auto try_candidate = [&](const std::vector<uint32_t>& v) -> bool {
        std::vector<uint32_t> c(dimA, 0u);
        for (uint32_t j = 0; j < f; ++j) c[noncols[j]] = v[j];
        const Matrix w = elt_of_coords(c);
        // deg min_Abar(wbar) = f ?  (Krylov on the residues; the first
        // dependence is the degree.)
        RrefRowSpan sp(f, ell_);
        Matrix pw(D, ell_);   // identity
        uint32_t deg = f + 1;
        for (uint32_t i = 0; i <= f; ++i) {
            const std::vector<uint32_t> q =
                Q(AlgebraCoords(closure, pivots, pw, ell_, "StrictLocalityRoute(krylov)"));
            if (!sp.insert(q)) { deg = i; break; }
            pw = pw * w;
        }
        if (deg != f) return false;
        // Clause (1) of [E] Cert. 5.2, then clauses (2) and (3) through the
        // ONE implementation the budget loop and the read-only certificate
        // also use.
        FpPoly cpW;
        std::vector<std::pair<FpPoly, uint32_t>> sfd;
        try {
            cpW = CharPolyBerkowitz(w);
            sfd = SquarefreeDecomposition(cpW);
        } catch (const std::exception&) {
            return false;
        }
        if (!(sfd.size() == 1 && static_cast<uint32_t>(sfd[0].first.degree()) == f &&
              CertifyIrreducible(sfd[0].first) == 0)) {
            return false;
        }
        const FpPoly& pi_w = sfd[0].first;
        std::map<uint64_t, FpPoly> character;
        if (CertifyWitnessClauses23(w, pi_w, f, N, ops_, pi_by_p, Asub_by_p, D, ell_, character,
                                    /*w_hat_out=*/nullptr) != kLocalityOk) {
            return false;
        }
        st.local = true;
        st.witness_poly = pi_w;
        st.w_hat_exponent = N;
        st.character = std::move(character);
        st.witness_matrix = w;                  // the witness is NOT linear in the t_p
        st.phase_reached = LocalityPhase::ResidueSearch;
        // The strict witness certifies the residue degree too, and f = f_lcm
        // was asserted above, so `residue_degree` does not move.
        st.residue_degree_certified = true;
        return true;
    };

    const uint32_t seeded = (StrictResidueSeededTrials() == kStrictSeededFromBudget)
                                ? budget
                                : StrictResidueSeededTrials();
    SplitMix64 rng(kResidueWitnessSeed ^ (0x9E3779B97F4A7C15ull * (f + 1u)) ^ ell_);
    for (uint32_t attempt = 0; attempt < seeded; ++attempt) {
        if (LocalityStepAndCheck()) return kStrictInterrupted;
        std::vector<uint32_t> v(f, 0u);
        for (uint32_t j = 0; j < f; ++j) v[j] = rng.next_mod(ell_);
        if (try_candidate(v)) return kStrictCertified;
    }
    // The deterministic, terminating fallback: Abar's coordinate vectors in
    // lexicographic order (low coordinate fastest).  Guaranteed to contain a
    // generator; in practice never entered.
    {
        std::vector<uint32_t> v(f, 0u);
        for (uint64_t counter = 0; counter < kResidueLexCap; ++counter) {
            if (LocalityStepAndCheck()) return kStrictInterrupted;
            if (try_candidate(v)) return kStrictCertified;
            uint32_t j = 0;
            while (j < f) {
                if (++v[j] < ell_) break;
                v[j] = 0;
                ++j;
            }
            if (j == f) break;   // all ell^f coordinate vectors exhausted
        }
    }
    throw std::runtime_error(
        "HeckeModuleDecomposition: strict locality mode found no element of A_S whose residue "
        "generates A_S / rad A_S, although dim(Berlekamp subalgebra) = 1 proves the block local "
        "and a finite field is never the union of its proper subfields -- this is an "
        "implementation defect, not a mathematical outcome");
}

// ---- Change of basis + block-diagonal verification for EVERY operator in the
// family (unchanged; shared by compute() and resume_locality()). ----
void HeckeModuleDecomposition::FinalizeChangeOfBasis() {
    uint32_t total = 0;
    for (const PrimaryBlock& b : blocks_) total += b.dim;
    if (total != d_) {
        throw std::runtime_error("HeckeModuleDecomposition: multi-op block dimensions do not "
                                 "sum to d");
    }

    std::vector<Matrix> block_bases;
    for (const PrimaryBlock& b : blocks_) block_bases.push_back(b.basis_columns);
    U_ = HStack(block_bases, d_, ell_);
    Matrix Ucopy = U_;
    if (!Ucopy.is_invertible()) {
        throw std::runtime_error("HeckeModuleDecomposition: multi-op change of basis not "
                                 "invertible");
    }
    U_inv_ = Ucopy.inverse();

    std::vector<uint32_t> dims;
    for (const PrimaryBlock& b : blocks_) dims.push_back(b.dim);
    for (const auto& op : ops_) {
        const Matrix C = U_inv_ * op.second * U_;
        if (!IsBlockDiagonal(C, dims)) {
            throw std::runtime_error("HeckeModuleDecomposition: conjugated matrix for T_" +
                                     std::to_string(op.first) + " is not block-diagonal");
        }
    }
}

const std::vector<PrimaryBlock>& HeckeModuleDecomposition::blocks() const {
    require_computed("blocks");
    return blocks_;
}

// TEST-ONLY (slot 3 WP-3); see the header.
std::vector<PrimaryBlock>& HeckeModuleDecomposition::blocks_for_tamper_test() {
    require_computed("blocks_for_tamper_test");
    return blocks_;
}

// (hecke-extensions E0a) Read-only view of the inputs, for CertifyLocality.
const std::vector<std::pair<uint64_t, Matrix>>& HeckeModuleDecomposition::ops() const {
    return ops_;
}

uint32_t HeckeModuleDecomposition::ell() const { return ell_; }

// ---------------------------------------------------------------------------
// (hecke-extensions phase E0b, slot 2 WP-1) The resumable locality state.
// ---------------------------------------------------------------------------

bool LocalityBlockState::equals(const LocalityBlockState& o) const {
    return dim == o.dim && basis_columns == o.basis_columns && local == o.local &&
           attempts == o.attempts &&
           residue_degree_lower_bound == o.residue_degree_lower_bound &&
           witness_poly == o.witness_poly && w_hat_exponent == o.w_hat_exponent &&
           character == o.character && witness == o.witness &&
           combos_tried == o.combos_tried &&
           residue_degree_certified == o.residue_degree_certified &&
           residue_witness == o.residue_witness &&
           witness_matrix == o.witness_matrix && phase_reached == o.phase_reached;
}

bool LocalityState::equals(const LocalityState& o, bool ignore_budget_history) const {
    if (mode != o.mode || budget != o.budget || splits != o.splits || complete != o.complete ||
        phase_reached != o.phase_reached || interrupted != o.interrupted) {
        return false;
    }
    if (!ignore_budget_history && budget_history != o.budget_history) return false;
    if (blocks.size() != o.blocks.size()) return false;
    for (size_t i = 0; i < blocks.size(); ++i) {
        if (!blocks[i].equals(o.blocks[i])) return false;
    }
    return true;
}

std::string LocalityState::to_json() const {
    std::ostringstream oss;
    oss << "{\"kind\":\"locality\","
        << "\"mode\":\"" << (mode == LocalityMode::Strict ? "strict" : "budget") << "\","
        << "\"budget\":" << budget << ",\"complete\":" << (complete ? "true" : "false")
        << ",\"splits\":" << splits
        << ",\"phase_reached\":\"" << to_string(phase_reached) << "\""
        << ",\"interrupted\":" << (interrupted ? "true" : "false")
        << ",\"budget_history\":[";
    for (size_t i = 0; i < budget_history.size(); ++i) {
        oss << (i ? "," : "") << budget_history[i];
    }
    oss << "],\"blocks\":[";
    for (size_t i = 0; i < blocks.size(); ++i) {
        const LocalityBlockState& b = blocks[i];
        oss << (i ? "," : "") << "{\"dim\":" << b.dim
            << ",\"local\":" << (b.local ? "true" : "false")
            << ",\"attempts\":" << b.attempts
            << ",\"residue_degree_lower_bound\":" << b.residue_degree_lower_bound
            << ",\"residue_degree_certified\":" << (b.residue_degree_certified ? "true" : "false")
            << ",\"phase_reached\":\"" << to_string(b.phase_reached) << "\""
            << ",\"w_hat_exponent\":" << b.w_hat_exponent << ",\"witness_poly\":";
        emit_poly_json(oss, b.witness_poly);
        oss << ",\"witness\":[";
        for (size_t j = 0; j < b.witness.size(); ++j) {
            oss << (j ? "," : "") << "[" << b.witness[j].first << "," << b.witness[j].second << "]";
        }
        oss << "],\"residue_witness\":[";
        for (size_t j = 0; j < b.residue_witness.size(); ++j) {
            oss << (j ? "," : "") << "[" << b.residue_witness[j].first << ","
                << b.residue_witness[j].second << "]";
        }
        oss << "],\"character\":[";
        {
            bool first = true;
            for (const auto& kv : b.character) {
                oss << (first ? "" : ",") << "{\"p\":" << kv.first << ",\"P\":";
                emit_poly_json(oss, kv.second);
                oss << "}";
                first = false;
            }
        }
        oss << "],\"combos_tried\":[";
        for (size_t j = 0; j < b.combos_tried.size(); ++j) {
            oss << (j ? "," : "") << "[";
            for (size_t t = 0; t < b.combos_tried[j].size(); ++t) {
                oss << (t ? "," : "") << "[" << b.combos_tried[j][t].first << ","
                    << b.combos_tried[j][t].second << "]";
            }
            oss << "]";
        }
        oss << "],\"witness_matrix\":";
        emit_matrix_json(oss, b.witness_matrix);
        oss << ",\"basis_columns\":";
        emit_matrix_json(oss, b.basis_columns);
        oss << "}";
    }
    oss << "]}";
    return oss.str();
}

const LocalityState& HeckeModuleDecomposition::locality_state() const {
    require_computed("locality_state");
    return loc_;
}

void HeckeModuleDecomposition::resume_locality(uint32_t new_budget) {
    require_computed("resume_locality");
    // R3.1 item 2: a budget below what the state has already reached is
    // refused with a named error; a resume on a complete state is a strict
    // no-op, so neither the budget nor the history moves.
    ENGINE_CHECK_MSG(new_budget >= loc_.budget,
                     "HeckeModuleDecomposition::resume_locality: budget below reached state");
    if (loc_.complete) return;
    if (ops_.size() == 1 || d_ == 0) return;   // the analytic path is always complete

    loc_.budget = new_budget;
    loc_.budget_history.push_back(new_budget);

    std::vector<Matrix> bases;
    std::vector<LocalityBlockState> carried = loc_.blocks;
    for (const LocalityBlockState& b : carried) bases.push_back(b.basis_columns);
    RunLocalityStage(bases, std::move(carried), new_budget);
    FinalizeChangeOfBasis();
}

const Matrix& HeckeModuleDecomposition::change_of_basis() const {
    require_computed("change_of_basis");
    return U_;
}

Matrix HeckeModuleDecomposition::conjugated(uint64_t p) const {
    require_computed("conjugated");
    const Matrix& A = op_matrix(p);
    Matrix C = U_inv_ * A * U_;
    std::vector<uint32_t> dims;
    for (const PrimaryBlock& b : blocks_) dims.push_back(b.dim);
    if (!IsBlockDiagonal(C, dims)) {
        throw std::runtime_error("HeckeModuleDecomposition::conjugated: not block-diagonal");
    }
    return C;
}

uint32_t HeckeModuleDecomposition::certify() const {
    // Unconditional certificate (plan Sec. 3): everything below is re-derived
    // from (ops_, ell_, U_, blocks_) by deterministic algorithms only --
    // never from the randomized factor search. Return 0 iff all pass.
    if (!computed_) return 1;

    // Sum of block dimensions = d.
    uint32_t total = 0;
    for (const PrimaryBlock& b : blocks_) total += b.dim;
    if (total != d_) return 2;

    // det U != 0, via a FRESH inverse (not the cached U_inv_), plus U U^{-1} = I.
    if (U_.rows != d_ || U_.cols != d_) return 3;
    Matrix Ucopy = U_;
    if (!Ucopy.is_invertible()) return 3;
    const Matrix Uinv = Ucopy.inverse();
    if (!(U_ * Uinv == Matrix(d_, ell_))) return 4;

    std::vector<uint32_t> dims;
    for (const PrimaryBlock& b : blocks_) dims.push_back(b.dim);

    // Collects, per block, per operator, the re-extracted irreducible pi_j --
    // used AFTER the loop below to independently recompute the residue
    // degree f = lcm_j(deg pi_j) and cross-check it against the reported
    // PrimaryBlock::residue_degree (Stage S11; for a single operator, k = 1,
    // this reduces to exactly the pre-existing S10 check `residue_degree ==
    // deg(pi)`, since lcm of one element is itself).
    std::vector<std::map<uint64_t, FpPoly>> pi_by_block(blocks_.size());

    for (const auto& op : ops_) {
        const uint64_t p = op.first;
        const Matrix& A = op.second;

        // U^{-1} A U exactly block-diagonal with the reported sizes.
        const Matrix C = Uinv * A * U_;
        if (!IsBlockDiagonal(C, dims)) return 5;

        // Factorization certificate on the ORIGINAL matrix: the product of
        // the reported per-block characteristic polynomials equals
        // charpoly(A), recomputed by the division-free Berkowitz algorithm.
        const FpPoly cpA = CharPolyBerkowitz(A);
        FpPoly product = FpPoly::one(ell_);

        uint32_t off = 0;
        for (size_t bi = 0; bi < blocks_.size(); ++bi) {
            const PrimaryBlock& b = blocks_[bi];
            const auto it_cp = b.char_poly.find(p);
            const auto it_mp = b.min_poly.find(p);
            const auto it_jp = b.jordan_partition.find(p);
            if (it_cp == b.char_poly.end() || it_mp == b.min_poly.end() ||
                it_jp == b.jordan_partition.end()) {
                return 6;
            }
            const FpPoly& cp_blk = it_cp->second;
            const FpPoly& mp_blk = it_mp->second;
            const std::vector<uint32_t>& partition = it_jp->second;
            if (cp_blk.ell != ell_ || mp_blk.ell != ell_) return 6;
            product = product * cp_blk;

            // Re-extract pi from the reported block charpoly by the
            // DETERMINISTIC squarefree machinery (no randomization): a
            // primary block must have squarefree decomposition pi^m with a
            // single part.
            std::vector<std::pair<FpPoly, uint32_t>> sfd;
            try {
                sfd = SquarefreeDecomposition(cp_blk);
            } catch (const std::exception&) {
                return 7;
            }
            if (sfd.size() != 1) return 7;
            const FpPoly& pi = sfd[0].first;
            const uint32_t m = sfd[0].second;
            const uint32_t f = static_cast<uint32_t>(pi.degree());

            // pi deterministically irreducible (gcd + Frobenius proof).
            if (CertifyIrreducible(pi) != 0) return 8;
            pi_by_block[bi].emplace(p, pi);
            // Dimension bookkeeping (residue-degree cross-operator check --
            // f = lcm_j(deg pi_j), NOT deg(pi) for this p alone when k > 1 --
            // is done in a separate pass below, after every operator's pi has
            // been collected for every block).
            if (b.dim != m * f) return 10;
            if (FpPoly::pow(pi, m) != cp_blk) return 11;

            // min poly is pi^e with 1 <= e <= m.
            if (mp_blk.degree() < 1 || mp_blk.degree() % f != 0) return 12;
            const uint32_t e = static_cast<uint32_t>(mp_blk.degree()) / f;
            if (e < 1 || e > m || FpPoly::pow(pi, e) != mp_blk) return 12;

            // On the conjugated diagonal block D: recomputed charpoly(D)
            // equals the reported pi^m; pi(D)^e = 0 and pi(D)^{e-1} != 0.
            const Matrix D = SubBlock(C, off, b.dim);
            if (CharPolyBerkowitz(D) != cp_blk) return 13;
            const Matrix PiD = EvalPolyAtMatrix(pi, D);
            std::vector<Matrix> pows;
            pows.reserve(e + 1);
            pows.emplace_back(b.dim, ell_); // identity
            for (uint32_t r = 1; r <= e; ++r) pows.push_back(pows.back() * PiD);
            if (!pows[e].is_zero()) return 14;
            if (pows[e - 1].is_zero()) return 15;

            // Jordan partition re-derived from the ranks of pi(D)^r on the block.
            std::vector<uint32_t> nullities(e + 1);
            for (uint32_t r = 0; r <= e; ++r) nullities[r] = b.dim - RankOf(pows[r]);
            std::vector<uint32_t> rederived;
            try {
                rederived = PartitionFromNullities(nullities, f, "certify");
            } catch (const std::exception&) {
                return 16;
            }
            if (rederived != partition) return 16;

            off += b.dim;
        }

        if (product != cpA) return 17;
    }

    // ---- Residue-degree cross-operator certificate (Stage S11; A-N4).
    // Every block must have collected a pi_j for every op in the family
    // (already enforced by the "missing per-op data" check, return 6, above);
    // f = lcm_j(deg pi_j) must equal the reported residue_degree EXACTLY,
    // whether reported certified or only as a lower bound. If reported
    // certified, the stored witness must independently re-verify: evaluated
    // on the block, it must have a squarefree min-poly part with a SINGLE
    // irreducible factor of degree exactly f. ----
    for (size_t bi = 0; bi < blocks_.size(); ++bi) {
        const PrimaryBlock& b = blocks_[bi];
        const std::map<uint64_t, FpPoly>& pis = pi_by_block[bi];
        if (pis.size() != ops_.size()) return 18; // missing pi for some op (should be unreachable)

        uint32_t f_lcm = 1;
        for (const auto& kv : pis) {
            f_lcm = Lcm32(f_lcm, static_cast<uint32_t>(kv.second.degree()));
        }
        if (b.residue_degree != f_lcm) return 19;

        if (b.residue_degree_certified) {
            // (E0b-S, slot 3 WP-3) The witness is EITHER a linear combination
            // of the t_p (`residue_witness`, all budget mode ever produces)
            // OR, under strict mode's step-3 route, the matrix itself
            // (`witness_matrix`, a polynomial in the t_p that no (p, c) list
            // can express).  Exactly one of the two is used, and which one is
            // a property of the reported data, never of how it was found.
            const bool have_linear = !b.residue_witness.empty();
            const bool have_matrix = (b.witness_matrix.rows == b.dim &&
                                      b.witness_matrix.cols == b.dim);
            if (!have_linear && !have_matrix) return 20;
            Matrix W(b.dim, b.dim, ell_); // zero
            if (have_linear) {
                for (const auto& term : b.residue_witness) {
                    if (pis.find(term.first) == pis.end()) return 20; // cites an unknown p
                    const Matrix& Ap = op_matrix(term.first);
                    Matrix Asub = SolveColumnsExact(b.basis_columns, Ap * b.basis_columns,
                        "certify(residue witness)");
                    W = W + Asub * term.second;
                }
            } else {
                W = b.witness_matrix;
            }
            std::vector<std::pair<FpPoly, uint32_t>> sfd;
            try {
                const FpPoly cpW = CharPolyBerkowitz(W);
                sfd = SquarefreeDecomposition(cpW);
            } catch (const std::exception&) {
                return 21;
            }
            if (sfd.size() != 1) return 22;
            if (static_cast<uint32_t>(sfd[0].first.degree()) != b.residue_degree) return 23;
            if (CertifyIrreducible(sfd[0].first) != 0) return 24;
        }

        // ---- (hecke-extensions phase E0b, slot 2 WP-1) THE LOCALITY
        // CERTIFICATE, codes 25-29 ([X] Sec. 3.1).  Re-derived from the
        // reported (local_certified, witness_poly, w_hat_exponent, character,
        // residue_witness) alone -- never from the search that produced them.
        //
        //  25  local_certified is inconsistent with the reported certificate
        //      data (a block flagged non-local must carry none of it; a block
        //      flagged local must carry a pi_w irreducible of degree f, a
        //      non-empty witness over known primes, and a character covering
        //      every operator)
        //  26  f does not divide N, or ell^N < dim
        //  27  pi_w(w_hat) != 0   (k = 1: pi_w is not the block's own pi, so
        //      the analytic Teichmueller statement does not hold)
        //  28  deg P_p >= f for some p
        //  29  n_p = t_p - P_p(w_hat) is not nilpotent for some p   (k = 1:
        //      the character is not the reduced representative X mod pi_w, so
        //      n = t - w_hat is not the block's nilpotent part)
        //  30  (E0b-S, slot 3 WP-3) a reported MATRIX witness (strict mode's
        //      step-3 route) does not commute with the operator family --
        //      without that, clauses (2)/(3) do not imply locality
        //
        // For k = 1 the block is local by construction (F_ell[t]/(pi^e) is a
        // local ring) and w_hat is NEVER formed -- clauses 27 and 29 are
        // re-checked ARITHMETICALLY, exactly as compute() derives them, so a
        // single-operator decomposition costs no matrix power here either.
        //
        // WHY `residue_witness` IS THE LOCALITY WITNESS.  The certificate's
        // clauses (2) and (3) are statements about a witness w, so certify()
        // needs one, and the only witness a PrimaryBlock reports is
        // `residue_witness`.  That is the right one, and not by coincidence:
        // on a LOCAL block clause (1) already implies the rest ([E] Claim 3.3
        // -- a w whose char-poly's squarefree part is irreducible of degree f
        // has a residue generating T/m, hence a Teichmueller lift and
        // characters), so the first clause-(1) candidate, which is exactly
        // what fixes `residue_witness`, is also the first fully certifying
        // one; and on a block that is NOT local nothing is claimed and this
        // branch is not entered.  certify() does not TRUST that argument: it
        // re-derives (2) and (3) on the reported witness and fails with 27 or
        // 29 if they do not hold -- which is the correct verdict for a
        // certificate that cannot be re-derived from the reported data.
        const uint32_t f_here = b.residue_degree;
        if (!b.local_certified) {
            if (!b.witness_poly.coeffs.empty() || b.w_hat_exponent != 0 ||
                !b.character.empty()) {
                return 25;  // flagged non-local but carrying a certificate
            }
            continue;
        }
        if (b.witness_poly.coeffs.empty()) return 25;
        if (static_cast<uint32_t>(b.witness_poly.degree()) != f_here) return 25;
        if (CertifyIrreducible(b.witness_poly) != 0) return 25;
        if (b.character.size() != ops_.size()) return 25;
        // (E0b-S) A local block reports a witness: a linear one, or strict
        // mode's matrix one.  A block that reports NEITHER has no witness for
        // clauses (2)/(3) to be re-derived on, and a block that reports BOTH
        // would leave which one is certified ambiguous.
        {
            const bool have_linear = !b.residue_witness.empty();
            const bool have_matrix = (b.witness_matrix.rows == b.dim &&
                                      b.witness_matrix.cols == b.dim && b.dim > 0);
            if (have_linear == have_matrix) return 25;
        }
        for (const auto& op : ops_) {
            if (b.character.find(op.first) == b.character.end()) return 25;
        }
        {
            const uint32_t N = b.w_hat_exponent;
            if (f_here == 0 || N % f_here != 0) return 26;
            uint64_t acc = 1;
            bool reaches = false;
            for (uint32_t i = 0; i < N; ++i) {
                acc *= static_cast<uint64_t>(ell_);
                if (acc >= static_cast<uint64_t>(b.dim)) { reaches = true; break; }
            }
            if (!reaches && b.dim > 1) return 26;
        }
        for (const auto& kv : b.character) {
            if (!kv.second.coeffs.empty() &&
                static_cast<uint32_t>(kv.second.degree()) >= f_here) {
                return 28;
            }
        }
        if (ops_.size() == 1) {
            // The analytic path: pi_w must BE the block's own irreducible, and
            // the character the reduced representative X mod pi_w.
            const uint64_t p = ops_[0].first;
            if (!(b.witness_poly == pis.at(p))) return 27;
            if (!(b.character.at(p) == FpPoly::mod(FpPoly::X(ell_), b.witness_poly))) return 29;
            continue;
        }
        {
            // The matrix path: form w_hat once and re-derive both clauses.
            Matrix W(b.dim, b.dim, ell_); // zero
            const bool matrix_witness = b.residue_witness.empty();
            if (matrix_witness) {
                // (E0b-S, slot 3 WP-3) STRICT MODE'S WITNESS.  A linear
                // witness lies in A_S by construction; a reported MATRIX does
                // not, so certify() checks the one thing the sufficiency
                // argument of [E] Cert. 5.2 needs of it beyond clauses (2)
                // and (3): that it COMMUTES with the family.  With that,
                // t_p = P_p(w_hat) + n_p inside the commutative algebra
                // K[n_p] with K = F_ell[w_hat] a field and the n_p commuting
                // nilpotents, so A_S sits in a local ring, meets its maximal
                // ideal in nilpotents only, and A_S / rad A_S embeds in K as
                // a finite domain -- a field.  Hence the block is local.
                // Code 30 is what a reported witness outside the commutant
                // fails on.
                W = b.witness_matrix;
                for (const auto& op : ops_) {
                    Matrix Asub = SolveColumnsExact(b.basis_columns, op.second * b.basis_columns,
                        "certify(locality witness commutation)");
                    if (!((W * Asub) == (Asub * W))) return 30;
                }
            } else {
                for (const auto& term : b.residue_witness) {
                    const Matrix& Ap = op_matrix(term.first);
                    Matrix Asub = SolveColumnsExact(b.basis_columns, Ap * b.basis_columns,
                        "certify(locality witness)");
                    W = W + Asub * term.second;
                }
            }
            const Matrix w_hat = FrobeniusPower(W, ell_, b.w_hat_exponent);
            if (!EvalPolyAtMatrix(b.witness_poly, w_hat).is_zero()) return 27;
            for (const auto& op : ops_) {
                Matrix Asub = SolveColumnsExact(b.basis_columns, op.second * b.basis_columns,
                    "certify(locality nilpotency)");
                const Matrix n_p = Asub - EvalPolyAtMatrix(b.character.at(op.first), w_hat);
                if (!IsNilpotent(n_p, b.dim)) return 29;
            }
        }
    }

    return 0;
}

std::string HeckeModuleDecomposition::to_string_offset(uint32_t offset) const {
    const std::string ind = indent(offset);
    std::ostringstream oss;
    oss << ind << "HeckeModuleDecomposition (level (c), "
        << (ops_.size() == 1 ? "single-operator primary decomposition"
                             : "simultaneous MaxSpec(T) refinement")
        << ")\n";
    oss << ind << "  ell = " << ell_ << ", dim H = " << d_ << ", operators:";
    for (const auto& op : ops_) oss << " T_" << op.first;
    oss << "\n";
    if (!computed_) {
        oss << ind << "  <compute() not yet run>\n";
        return oss.str();
    }
    oss << ind << "  blocks: " << blocks_.size() << "\n";
    for (size_t j = 0; j < blocks_.size(); ++j) {
        const PrimaryBlock& b = blocks_[j];
        oss << ind << "  block " << j << ": dim = " << b.dim << ", residue degree ";
        if (b.residue_degree_certified) {
            oss << "= " << b.residue_degree;
        } else {
            oss << ">= " << b.residue_degree << " (LOWER BOUND, not certified -- "
                                                "flagged for human review)";
        }
        // (hecke-extensions phase E0b, slot 2 WP-2) The corrected note, the
        // same three statements HeckeReport's `multiplicity_note` makes:
        // multiplicity, semisimplicity, single-operator family or not. The
        // pre-E0b text ("genuinely irreducible as a T-module, NOT further
        // separable by more Hecke operators") was false on both counts
        // ([I G1]).
        if (b.residue_degree_certified && b.residue_degree > 0 &&
            b.dim % b.residue_degree == 0 && b.dim > b.residue_degree) {
            oss << " (isotypic multiplicity " << (b.dim / b.residue_degree)
                << " over the computed operator family)";
        }
        {
            bool semisimple = true;
            for (const auto& kv : b.jordan_partition) {
                for (uint32_t part : kv.second) {
                    if (part != 1) semisimple = false;
                }
            }
            oss << (semisimple ? ", semisimple" : ", NOT semisimple");
            oss << (b.local_certified ? ", local (certified)" : ", locality NOT certified");
            oss << (ops_.size() == 1
                        ? " [single operator: a larger commuting family may refine this block]"
                        : " [no operator of the computed family refines this block further]");
        }
        oss << "\n";
        for (const auto& op : ops_) {
            const uint64_t p = op.first;
            oss << ind << "    T_" << p << ": charpoly = "
                << b.char_poly.at(p).to_string() << "\n";
            oss << ind << "    T_" << p << ": minpoly  = "
                << b.min_poly.at(p).to_string() << "\n";
            oss << ind << "    T_" << p << ": Jordan partition = (";
            const std::vector<uint32_t>& partition = b.jordan_partition.at(p);
            for (size_t t = 0; t < partition.size(); ++t) {
                if (t > 0) oss << ", ";
                oss << partition[t];
            }
            oss << ")\n";
        }
    }
    oss << ind << "  certified: " << (certify() == 0 ? "yes" : "NO") << "\n";
    return oss.str();
}

std::string HeckeModuleDecomposition::to_json() const {
    require_computed("to_json");
    std::ostringstream oss;
    oss << "{";
    oss << "\"schema\":\"hecke_module_decomposition/v2\",";
    oss << "\"ell\":" << ell_ << ",";
    oss << "\"operators\":[";
    for (size_t i = 0; i < ops_.size(); ++i) {
        if (i > 0) oss << ",";
        oss << "{\"p\":" << ops_[i].first << "}";
    }
    oss << "],";
    oss << "\"cohomology_dim\":" << d_ << ",";
    // Documented conventions (amendment A-N4; see Hecke/README.md):
    oss << "\"poly_coeff_order\":\"low->high degree (A-N4)\",";
    oss << "\"matrix_entry_order\":\"arrays of rows (row-major); entries are "
           "canonical residues in [0, ell)\",";
    oss << "\"blocks\":[";
    for (size_t j = 0; j < blocks_.size(); ++j) {
        const PrimaryBlock& b = blocks_[j];
        if (j > 0) oss << ",";
        oss << "{\"dim\":" << b.dim << ",";
        // A-N4: an unproven residue degree is NEVER emitted under the
        // unqualified "residue_degree" key -- only "residue_degree_lower_bound".
        if (b.residue_degree_certified) {
            oss << "\"residue_degree\":" << b.residue_degree << ",";
        } else {
            oss << "\"residue_degree_lower_bound\":" << b.residue_degree << ",";
        }
        oss << "\"residue_degree_certified\":" << (b.residue_degree_certified ? "true" : "false") << ",";
        // (hecke-extensions phase E0b, slot 2 WP-2) The locality certificate,
        // the same three keys hecke_report/v2 emits and with the same rule:
        // `witness_poly` / `coefficient_field` appear exactly when `local` is
        // true. This object is hashed NOWHERE ([X] Sec. 1.6 / [XA M1]) -- the
        // golden report texts come from HeckeReport.cpp.
        oss << "\"local\":" << (b.local_certified ? "true" : "false") << ",";
        if (b.local_certified) {
            oss << "\"witness_poly\":";
            emit_poly_json(oss, b.witness_poly);
            oss << ",\"coefficient_field\":{\"w_hat_exponent\":" << b.w_hat_exponent
                << ",\"P\":{";
            bool first_chi = true;
            for (const auto& kv : b.character) {
                if (!first_chi) oss << ",";
                first_chi = false;
                oss << "\"" << kv.first << "\":";
                emit_poly_json(oss, kv.second);
            }
            oss << "}},";
        }
        oss << "\"residue_witness\":[";
        for (size_t t = 0; t < b.residue_witness.size(); ++t) {
            if (t > 0) oss << ",";
            oss << "{\"p\":" << b.residue_witness[t].first
                << ",\"c\":" << b.residue_witness[t].second << "}";
        }
        oss << "],";
        oss << "\"min_polys\":{";
        {
            bool first = true;
            for (const auto& kv : b.min_poly) {
                if (!first) oss << ",";
                first = false;
                oss << "\"" << kv.first << "\":";
                emit_poly_json(oss, kv.second);
            }
        }
        oss << "},\"char_polys\":{";
        {
            bool first = true;
            for (const auto& kv : b.char_poly) {
                if (!first) oss << ",";
                first = false;
                oss << "\"" << kv.first << "\":";
                emit_poly_json(oss, kv.second);
            }
        }
        oss << "},\"jordan_partitions\":{";
        {
            bool first = true;
            for (const auto& kv : b.jordan_partition) {
                if (!first) oss << ",";
                first = false;
                oss << "\"" << kv.first << "\":[";
                for (size_t t = 0; t < kv.second.size(); ++t) {
                    if (t > 0) oss << ",";
                    oss << kv.second[t];
                }
                oss << "]";
            }
        }
        oss << "},\"basis_columns\":";
        emit_matrix_json(oss, b.basis_columns);
        oss << "}";
    }
    oss << "],";
    oss << "\"change_of_basis\":";
    emit_matrix_json(oss, U_);
    oss << ",";
    oss << "\"certified\":" << (certify() == 0 ? "true" : "false");
    oss << "}";
    return oss.str();
}

// ===========================================================================
// (hecke-extensions phase E0a) CertifyLocality -- [E] Cert. 5.2 / Claim 3.3.
// Read-only: nothing in compute() calls it.  See HeckeModuleDecomposition.h
// for the mathematics and for what each failure code means.
// ===========================================================================

LocalityVerdict CertifyLocality(const PrimaryBlock& block,
                                const std::vector<std::pair<uint64_t, Matrix>>& ops,
                                uint32_t ell,
                                const LocalityOptions& opts) {
    LocalityVerdict v;
    if (ell < 2) throw std::invalid_argument("CertifyLocality: ell must be a prime >= 2");
    if (ops.empty()) throw std::invalid_argument("CertifyLocality: empty operator family");
    for (const auto& op : ops) {
        if (op.second.rows != op.second.cols) {
            throw std::invalid_argument("CertifyLocality: non-square operator");
        }
        if (op.second.p != ell) {
            throw std::invalid_argument("CertifyLocality: operator modulus != ell");
        }
    }
    const uint32_t D = block.dim;
    if (D == 0 || block.basis_columns.cols != D) {
        v.failure = kLocalityMalformedBlock;
        v.note = "block dimension does not match its basis";
        return v;
    }

    // --- pi_p per operator, and f = lcm_j deg(pi_j), re-derived from the
    // block's own min-polys (never taken on trust from residue_degree). ---
    std::map<uint64_t, FpPoly> pi_by_p;
    uint32_t f = 1;
    for (const auto& op : ops) {
        auto it = block.min_poly.find(op.first);
        if (it == block.min_poly.end()) {
            v.failure = kLocalityMalformedBlock;
            v.note = "no min-poly recorded for T_" + std::to_string(op.first);
            return v;
        }
        FpPoly pi;
        if (!PrimaryIrreducible(it->second, pi)) {
            v.failure = kLocalityMalformedBlock;
            v.note = "min-poly of T_" + std::to_string(op.first) + " is not pi^e";
            return v;
        }
        f = Lcm32(f, static_cast<uint32_t>(pi.degree()));
        pi_by_p.emplace(op.first, pi);
    }
    v.residue_degree = f;
    if (block.residue_degree != f) {
        v.failure = kLocalityDegreeMismatch;
        v.note = "reported residue degree " + std::to_string(block.residue_degree) +
                 " != re-derived lcm " + std::to_string(f);
        return v;
    }
    const uint32_t N = TeichmuellerExponent(ell, f, D);
    v.w_hat_exponent = N;

    // --- k = 1: F_ell[t]/(pi^e) is local by construction; w_hat is NEVER
    // formed ([X Sec. 3.1], [XA m17]). ---
    if (ops.size() == 1 && !opts.force_matrix_path) {
        const uint64_t p = ops[0].first;
        v.local = true;
        v.analytic = true;
        v.attempts = 1;
        v.witness_poly = pi_by_p.at(p);
        v.witness = { {p, 1u} };
        v.character.emplace(p, FpPoly::mod(FpPoly::X(ell), v.witness_poly));
        return v;
    }

    // --- Restrict every operator to the block (solvability = commutation). ---
    std::map<uint64_t, Matrix> Asub_by_p;
    try {
        for (const auto& op : ops) {
            Asub_by_p.emplace(op.first,
                SolveColumnsExact(block.basis_columns, op.second * block.basis_columns,
                                  "CertifyLocality(restriction)"));
        }
    } catch (const std::exception& e) {
        v.failure = kLocalityMalformedBlock;
        v.note = std::string("block is not invariant: ") + e.what();
        return v;
    }

    // --- The certificate, for one witness combination. ---
    auto try_witness = [&](const std::vector<std::pair<uint64_t, uint32_t>>& combo,
                           LocalityVerdict& out) -> uint32_t {
        Matrix W(D, D, ell); // zero
        for (const auto& term : combo) W = W + Asub_by_p.at(term.first) * term.second;

        FpPoly pi_w;
        try {
            const FpPoly cpW = CharPolyBerkowitz(W);
            const std::vector<std::pair<FpPoly, uint32_t>> sfd = SquarefreeDecomposition(cpW);
            if (sfd.size() != 1) return kLocalityNoWitness;
            if (static_cast<uint32_t>(sfd[0].first.degree()) != f) return kLocalityNoWitness;
            if (CertifyIrreducible(sfd[0].first) != 0) return kLocalityNoWitness;
            pi_w = sfd[0].first;
        } catch (const std::exception&) {
            return kLocalityNoWitness;
        }

        // (2) + (3), through the ONE shared implementation (E0b, WP-1: the
        // refinement loop enforces literally this certificate).
        std::map<uint64_t, FpPoly> character;
        Matrix w_hat(0, 0, ell);
        const uint32_t code = CertifyWitnessClauses23(W, pi_w, f, N, ops, pi_by_p, Asub_by_p,
                                                      D, ell, character, &w_hat);
        if (code != kLocalityOk) return code;

        out.local = true;
        out.failure = kLocalityOk;
        out.witness_poly = pi_w;
        out.character = std::move(character);
        out.w_hat = w_hat;
        out.witness_matrix = W;
        return kLocalityOk;
    };

    // --- Witness enumeration, in the SHIPPED order: first every single
    // operator whose deg(pi_p) == f (the shipped search takes the first of
    // them), then the all-ones combination, then the deterministic-seed
    // random combinations with the shipped seed.  For a genuinely local block
    // the FIRST candidate is both the shipped witness and a full certificate
    // ([X Sec. 3.3] Claim) -- which is exactly what the grid arm asserts. ---
    const uint32_t budget =
        opts.max_attempts ? opts.max_attempts : kResidueWitnessMaxAttempts;
    // `last` keeps the STRONGEST evidence seen, not the most recent one: a
    // candidate that fails clause (1) ("this combination is not a witness at
    // all") says nothing about the block, whereas one that passes (1) and (2)
    // and then fails (3) is a PROOF of non-locality.  So the first non-
    // kLocalityNoWitness code wins and is never overwritten.
    uint32_t last = kLocalityNoWitness;
    std::string last_note = "no witness with an irreducible squarefree part of degree f";
    const auto record = [&last](uint32_t code) {
        if (last == kLocalityNoWitness) last = code;
    };

    for (const auto& op : ops) {
        if (static_cast<uint32_t>(pi_by_p.at(op.first).degree()) != f) continue;
        const std::vector<std::pair<uint64_t, uint32_t>> combo = { {op.first, 1u} };
        ++v.attempts;
        LocalityVerdict cand;
        const uint32_t code = try_witness(combo, cand);
        if (code == kLocalityOk) {
            cand.residue_degree = f;
            cand.w_hat_exponent = N;
            cand.witness = combo;
            cand.attempts = v.attempts;
            return cand;
        }
        record(code);
    }

    SplitMix64 rng(kResidueWitnessSeed ^ (0x9E3779B97F4A7C15ull * (block.dim + 1u)) ^ ell);
    std::vector<std::pair<uint64_t, uint32_t>> combo;
    for (const auto& op : ops) combo.emplace_back(op.first, 1u); // first attempt: all-ones
    for (uint32_t attempt = 0; attempt < budget; ++attempt) {
        if (attempt > 0) {
            for (auto& term : combo) {
                term.second = (ell > 1) ? (1u + rng.next_mod(ell - 1)) : 0u;
            }
        }
        ++v.attempts;
        LocalityVerdict cand;
        const uint32_t code = try_witness(combo, cand);
        if (code == kLocalityOk) {
            cand.residue_degree = f;
            cand.w_hat_exponent = N;
            cand.witness = combo;
            cand.attempts = v.attempts;
            return cand;
        }
        record(code);
    }

    v.failure = last ? last : kLocalityNoWitness;
    switch (v.failure) {
        case kLocalityWHatNotInResidue:
            last_note = "pi_w(w_hat) != 0 for every accepted witness"; break;
        case kLocalityEmptyIntersection:
            last_note = "ker pi_p(t_p) & ker pi_w(w) = 0 -- the block carries no "
                        "common residue line"; break;
        case kLocalityCharacterUnsolvable:
            last_note = "no P_p with deg < f and t_p = P_p(w_hat) on the common "
                        "kernel -- the block is NOT local ([E] Claim 3.1)"; break;
        case kLocalityNotNilpotent:
            last_note = "t_p - P_p(w_hat) is not nilpotent -- the block is NOT local"; break;
        default: break;
    }
    v.note = last_note;
    return v;
}

} // namespace hecke
