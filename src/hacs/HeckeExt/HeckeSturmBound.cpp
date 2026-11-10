// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions phase E1s -- the computed Sturm bound B'(k, ell).
// See HeckeSturmBound.h for the mathematical contract ([E] `def:bprime`,
// `cl:bprime`, `cl:sturmmodl`) and the resumable-state contract
// ([X] Sec. 1.3.1 R3, instance 1).
//
// INCLUDE DISCIPLINE ([X] Sec. 1.1, enforced by
// docs/matrix-engine/tools/check_heckeext_includes.sh): this TU may include
// Hecke/SubspaceOps.h, Hecke/FpPoly.h, ModArithmetic's Matrix.h /
// ModArithmetic.h / EngineCheck.h / Fnv.h, its own header, and C++ standard headers --
// nothing else.  In particular no cohomology, no coset table, no tape, no CUDA.

#include "HeckeSturmBound.h"

#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/Fnv.h"

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace hecke_ext {

namespace {

using Mat = MatrixT<uint32_t>;

// --- small deterministic helpers -------------------------------------------

bool IsPrime(uint32_t n) {
    if (n < 2u) return false;
    if (n % 2u == 0u) return n == 2u;
    for (uint64_t d = 3; d * d <= static_cast<uint64_t>(n); d += 2) {
        if (n % static_cast<uint32_t>(d) == 0u) return false;
    }
    return true;
}

uint32_t NextPrimeAfter(uint32_t n) {
    uint32_t q = (n < 2u) ? 2u : n + 1u;
    while (!IsPrime(q)) ++q;
    return q;
}

// Byte-wise FNV-1a-64 hash of the R3.1 item 6 `state_ref.fnv64` /
// `state_fnv64` integrity field (never on a result path): the ONE definition
// in src/ModArithmetic/Fnv.h (WP-2), from its STANDARD offset basis
// engine::kFnvOffset64 = 0xcbf29ce484222325 -- the same function, seed
// included, as every golden / campaign / benchmark hash in the tree.
//
// History: WP-3's former file-local copy seeded with 1469598103934665603
// (0x14650fb0739d0383, the seed of BlockPool.h / tape::fnv_init, a DIFFERENT
// hash family), and the WP-2/WP-3 integration kept that seed to stay
// value-preserving.  WP-7 (slot 1 close, lead's decision) switched it to the
// standard basis: nothing had pinned a state hash (no golden entry, no
// validation constant, no recorded campaign JSON carries one), and a recorded
// integrity hash must be computable by a third party with the one documented
// function ([X] Sec. 1.3.1 R3.1 item 6: "FNV-1a-64, Fnv.h").
uint64_t Fnv1a64(const std::string& s) {
    return engine::FnvBytes(engine::kFnvOffset64, s);
}

std::string HexU64(uint64_t v) {
    static const char* kDigits = "0123456789abcdef";
    std::string out = "0x";
    for (int shift = 60; shift >= 0; shift -= 4) {
        out.push_back(kDigits[(v >> shift) & 0xFull]);
    }
    return out;
}

// --- truncated power series over F_ell -------------------------------------

// c = a * b, truncated to L coefficients.
//
// (slot-22 PERF1, P9) Delayed reduction: the exact integer sums
// sum_{i+j=m} a_i b_j are accumulated in uint64 and reduced mod ell only every
// `batch` rows of the schoolbook product, instead of one mod_mul + mod_add per
// term.  Every input is a canonical residue (< ell), so a product is at most
// (ell-1)^2; between two reductions an accumulator holds a residue (< ell) plus
// at most `batch` such products, and `batch` is chosen so that this cannot
// exceed 2^64 - 1.  The final reduction leaves the canonical residue of the
// same integer sum, i.e. exactly what the per-term mod_add chain produced --
// the coefficients are bit-identical, only the number of divisions changes.
std::vector<uint32_t> SeriesMul(const std::vector<uint32_t>& a,
                                const std::vector<uint32_t>& b,
                                uint32_t L, uint32_t ell) {
    ENGINE_CHECK_MSG(a.size() >= L && b.size() >= L, "SeriesMul: an operand is shorter than the precision");
    // The batch bound below needs canonical operands (every entry < ell); compiled out under NDEBUG.
    ENGINE_DEBUG_CHECK_MSG(std::all_of(a.begin(), a.begin() + L, [ell](uint32_t x) { return x < ell; }) &&
                           std::all_of(b.begin(), b.begin() + L, [ell](uint32_t x) { return x < ell; }),
                           "SeriesMul: non-canonical operand");
    const uint64_t e = ell;
    const uint64_t maxprod = (e - 1u) * (e - 1u);
    // batch * maxprod + (ell - 1) <= 2^64 - 1, at least 1 (ell < 2^32).
    uint64_t batch = (maxprod == 0u) ? static_cast<uint64_t>(L) + 1u : (~0ull - (e - 1u)) / maxprod;
    if (batch == 0u) batch = 1u;
    std::vector<uint64_t> acc(L, 0u);
    uint64_t since = 0;
    for (uint32_t i = 0; i < L; ++i) {
        const uint64_t ai = a[i];
        if (ai == 0u) continue;
        if (since == batch) {
            for (uint32_t m = i; m < L; ++m) acc[m] %= e;   // rows < i are final: no later row reaches them
            since = 0;
        }
        uint64_t* dst = acc.data() + i;
        const uint32_t* src = b.data();
        const uint32_t n = L - i;
        for (uint32_t j = 0; j < n; ++j) dst[j] += ai * src[j];
        ++since;
    }
    std::vector<uint32_t> c(L, 0u);
    for (uint32_t m = 0; m < L; ++m) c[m] = static_cast<uint32_t>(acc[m] % e);
    return c;
}

// A * B over F_ell with the same delayed reduction (slot-22 PERF1, P9): the
// value of MatrixT::operator* entry for entry (both are the canonical residue
// of the exact integer sum_k A_ik B_kj), without a division per term.
Mat MatMulExact(const Mat& A, const Mat& B) {
    ENGINE_CHECK_MSG(A.cols == B.rows && A.p == B.p, "MatMulExact: shapes or moduli disagree");
    // Canonical operands (entries < p) are the precondition of the batch bound; compiled out under NDEBUG.
    ENGINE_DEBUG_CHECK_MSG(
        [&] { for (const auto* M : {&A, &B}) for (const auto& row : M->data) for (uint32_t x : row) if (x >= M->p) return false; return true; }(),
        "MatMulExact: non-canonical operand");
    const uint64_t e = A.p;
    const uint64_t maxprod = (e - 1u) * (e - 1u);
    uint64_t batch = (maxprod == 0u) ? static_cast<uint64_t>(A.cols) + 1u : (~0ull - (e - 1u)) / maxprod;
    if (batch == 0u) batch = 1u;
    Mat C(A.rows, B.cols, A.p);
    std::vector<uint64_t> acc(B.cols, 0u);
    for (uint32_t i = 0; i < A.rows; ++i) {
        std::fill(acc.begin(), acc.end(), 0u);
        uint64_t since = 0;
        for (uint32_t k = 0; k < A.cols; ++k) {
            const uint64_t a = A.data[i][k];
            if (a == 0u) continue;
            if (since == batch) {
                for (uint64_t& x : acc) x %= e;
                since = 0;
            }
            const uint32_t* b = B.data[k].data();
            for (uint32_t j = 0; j < B.cols; ++j) acc[j] += a * b[j];
            ++since;
        }
        for (uint32_t j = 0; j < B.cols; ++j) C.data[i][j] = static_cast<uint32_t>(acc[j] % e);
    }
    return C;
}

std::vector<uint32_t> SeriesOne(uint32_t L) {
    std::vector<uint32_t> s(L, 0u);
    if (L > 0) s[0] = 1u;
    return s;
}

// sigma_j(m) mod ell for m = 0 .. L-1 (entry 0 unused), by a divisor sieve.
std::vector<uint32_t> SigmaTable(uint32_t j, uint32_t L, uint32_t ell) {
    std::vector<uint32_t> sig(L, 0u);
    for (uint32_t d = 1; d < L; ++d) {
        const uint32_t dj = ModArithmetic::mod_pow(d % ell, j, ell);
        if (dj == 0u) continue;
        for (uint32_t m = d; m < L; m += d) {
            sig[m] = ModArithmetic::mod_add(sig[m], dj, ell);
        }
    }
    return sig;
}

// The monomial (a, b) with 4a + 6b = r, b in {0, 1}, for the admissible
// weights r in {0, 4, 6, 8, 10, 14, ...}: b = 1 iff r = 2 (mod 4).
void MonomialExponents(uint32_t r, uint32_t& a, uint32_t& b) {
    b = (r % 4u == 2u) ? 1u : 0u;
    ENGINE_CHECK_MSG(r >= 6u * b, "EchelonMillerBasis: negative E_4 exponent");
    a = (r - 6u * b) / 4u;
}

} // namespace

// ---------------------------------------------------------------------------
// Enum names
// ---------------------------------------------------------------------------

const char* ToString(SturmOutcome outcome) {
    switch (outcome) {
        case SturmOutcome::InProgress: return "in_progress";
        case SturmOutcome::Found:      return "found";
        case SturmOutcome::NoneAtCap:  return "none_at_cap";
    }
    return "in_progress";
}

const char* ToString(SturmUnavailableMode mode) {
    return mode == SturmUnavailableMode::Heuristic ? "heuristic" : "nonfinal";
}

// ToString(SturmFinal) moved to HeckeExt/HeckeExtFinal.h as the inline
// ToString(Final) (slot-4 WP-1 / E2a): SturmFinal IS Final now, and a second
// out-of-line definition here would be a redefinition.  Same three strings.

// ---------------------------------------------------------------------------
// q-expansions and the Miller basis
// ---------------------------------------------------------------------------

std::vector<uint32_t> EisensteinSeries(uint32_t weight, uint32_t L, uint32_t ell) {
    ENGINE_CHECK_MSG(weight == 4u || weight == 6u, "EisensteinSeries: only E_4 and E_6");
    ENGINE_CHECK_MSG(L > 0u, "EisensteinSeries: empty precision");
    ENGINE_CHECK_MSG(ell >= 2u, "EisensteinSeries: ell must be at least 2");

    const uint32_t j    = weight - 1u;           // sigma_3 for E_4, sigma_5 for E_6
    const uint32_t coef = (weight == 4u) ? 240u : 504u;
    const std::vector<uint32_t> sig = SigmaTable(j, L, ell);

    std::vector<uint32_t> e(L, 0u);
    e[0] = 1u % ell;
    const uint32_t c = coef % ell;
    for (uint32_t m = 1; m < L; ++m) {
        const uint32_t t = ModArithmetic::mod_mul(c, sig[m], ell);
        e[m] = (weight == 4u) ? t : ModArithmetic::mod_minus(t, ell);
    }
    return e;
}

std::vector<uint32_t> DeltaSeries(uint32_t L, uint32_t ell) {
    ENGINE_CHECK_MSG(L > 0u, "DeltaSeries: empty precision");
    ENGINE_CHECK_MSG(ell >= 2u, "DeltaSeries: ell must be at least 2");
    if (L == 1u) return std::vector<uint32_t>(1, 0u);

    // prod_{n>=1} (1 - q^n) by the pentagonal number theorem:
    //   sum_{j in Z} (-1)^j q^{j(3j-1)/2}.
    const uint32_t M = L - 1u;                  // the coefficients Delta needs, shifted by q
    std::vector<uint32_t> eta(M, 0u);
    eta[0] = 1u % ell;
    for (int64_t j = 1;; ++j) {
        const int64_t g1 = j * (3 * j - 1) / 2;
        const int64_t g2 = j * (3 * j + 1) / 2;
        if (g1 >= static_cast<int64_t>(M)) break;
        const bool sign_is_minus = (j % 2 != 0);
        const uint32_t one = 1u % ell;
        const uint32_t val = sign_is_minus ? ModArithmetic::mod_minus(one, ell) : one;
        eta[static_cast<uint32_t>(g1)] = ModArithmetic::mod_add(eta[static_cast<uint32_t>(g1)], val, ell);
        if (g2 < static_cast<int64_t>(M)) {
            eta[static_cast<uint32_t>(g2)] =
                ModArithmetic::mod_add(eta[static_cast<uint32_t>(g2)], val, ell);
        }
    }

    // eta^24 by square-and-multiply on the bits of 24 = 0b11000.
    std::vector<uint32_t> acc = SeriesOne(M);
    std::vector<uint32_t> base = eta;
    uint32_t e = 24u;
    while (e > 0u) {
        if (e & 1u) acc = SeriesMul(acc, base, M, ell);
        e >>= 1;
        if (e > 0u) base = SeriesMul(base, base, M, ell);
    }

    std::vector<uint32_t> delta(L, 0u);
    for (uint32_t m = 0; m < M; ++m) delta[m + 1u] = acc[m];
    return delta;
}

std::vector<std::vector<uint32_t>> EchelonMillerBasis(uint32_t k, uint32_t ell, uint32_t L) {
    const uint32_t d = SturmBound::dimension_M_k(k);
    ENGINE_CHECK_MSG(d >= 1u, "EchelonMillerBasis: M_k is zero for this weight");
    ENGINE_CHECK_MSG(L >= d, "EchelonMillerBasis: precision below dim M_k");

    const std::vector<uint32_t> e4 = EisensteinSeries(4u, L, ell);
    const std::vector<uint32_t> e6 = EisensteinSeries(6u, L, ell);
    const std::vector<uint32_t> dl = DeltaSeries(L, ell);

    // The exponents, j = 0 .. d-1 with r = k - 12j; the skipped r == 2 case can
    // only be the LAST j (r = 2 forces j = (k-2)/12 = floor(k/12)), so the
    // surviving leading exponents are exactly 0 .. d-1.
    uint32_t max_a = 0;
    std::vector<uint32_t> av(d, 0u), bv(d, 0u);
    for (uint32_t j = 0; j < d; ++j) {
        const uint32_t r = k - 12u * j;
        ENGINE_CHECK_MSG(r != 2u, "EchelonMillerBasis: weight-2 monomial inside the range");
        MonomialExponents(r, av[j], bv[j]);
        max_a = std::max(max_a, av[j]);
    }

    // (slot-22 PERF1, P9) Only the d powers E_4^{a_j} the rows use are built,
    // not all max_a + 1 of them (max_a ~ k/4, d ~ k/12): r_j = k - 12 j has the
    // residue of k mod 4 for every j, so b_j is constant and a_j = a_{j+1} + 3.
    // E_4^{a_{d-1}} is built by plain multiplication and each lower row's power
    // by one product with E_4^3.  Truncated series multiplication mod ell is
    // commutative and associative and commutes with truncation, so every power
    // -- hence every row -- is the same series as before, coefficient for
    // coefficient.
    for (uint32_t j = 0; j + 1 < d; ++j) {
        ENGINE_CHECK_MSG(bv[j] == bv[j + 1] && av[j] == av[j + 1] + 3u,
                         "EchelonMillerBasis: the monomial exponents are not a_j = a_{j+1} + 3");
    }
    (void)max_a;
    std::vector<std::vector<uint32_t>> pow_e4(d);
    {
        std::vector<uint32_t> p = SeriesOne(L);
        for (uint32_t i = 0; i < av[d - 1u]; ++i) p = SeriesMul(p, e4, L, ell);
        pow_e4[d - 1u] = std::move(p);
        if (d > 1u) {
            const std::vector<uint32_t> e4_cubed = SeriesMul(SeriesMul(e4, e4, L, ell), e4, L, ell);
            for (uint32_t j = d - 1u; j-- > 0;) pow_e4[j] = SeriesMul(pow_e4[j + 1u], e4_cubed, L, ell);
        }
    }

    std::vector<std::vector<uint32_t>> rows;
    rows.reserve(d);
    std::vector<uint32_t> pow_dl = SeriesOne(L);
    for (uint32_t j = 0; j < d; ++j) {
        if (j > 0) pow_dl = SeriesMul(pow_dl, dl, L, ell);
        std::vector<uint32_t> f = SeriesMul(pow_e4[j], pow_dl, L, ell);
        if (bv[j] == 1u) f = SeriesMul(f, e6, L, ell);
        rows.push_back(std::move(f));
        std::vector<uint32_t>().swap(pow_e4[j]);
    }

    // The monomial coefficient matrix is unitriangular over Z (Delta^j has
    // leading term q^j with coefficient 1, E_4^a E_6^b has constant term 1), so
    // its reduction is unitriangular for EVERY ell and a plain back-substitution
    // -- no pivoting, no inverse -- turns it into the reduced echelon form.
    for (uint32_t j = 0; j < d; ++j) {
        for (uint32_t m = 0; m < j; ++m) {
            ENGINE_CHECK_MSG(rows[j][m] == 0u, "EchelonMillerBasis: monomial matrix not unitriangular");
        }
        ENGINE_CHECK_MSG(rows[j][j] == 1u % ell,
                         "EchelonMillerBasis: monomial leading coefficient is not 1");
    }
    for (uint32_t j = d; j-- > 0;) {
        for (uint32_t i = 0; i < j; ++i) {
            const uint32_t c = rows[i][j];
            if (c == 0u) continue;
            for (uint32_t m = j; m < L; ++m) {
                if (rows[j][m] == 0u) continue;
                rows[i][m] = ModArithmetic::mod_sub(
                    rows[i][m], ModArithmetic::mod_mul(c, rows[j][m], ell), ell);
            }
        }
    }
    return rows;
}

MatrixT<uint32_t> HeckeOperatorOnMk(uint32_t q, uint32_t k, uint32_t ell,
                                    const std::vector<std::vector<uint32_t>>& miller) {
    const uint32_t d = static_cast<uint32_t>(miller.size());
    ENGINE_CHECK_MSG(d >= 1u, "HeckeOperatorOnMk: empty Miller basis");
    ENGINE_CHECK_MSG(q >= 2u, "HeckeOperatorOnMk: q must be at least 2");
    const uint32_t L = static_cast<uint32_t>(miller[0].size());
    // The precision bound of [X] Sec. 2.2 Stage 4(c) / R3.3 item 1, asserted at
    // EVERY use (the plan's own risk note: too small a precision would silently
    // truncate the closure).
    ENGINE_CHECK_MSG(static_cast<uint64_t>(q) * (d - 1u) + 1u <= L,
                     "HeckeOperatorOnMk: precision below q * (dim M_k - 1) + 1");

    // a_m(T_q f) = a_{qm}(f) + q^{k-1} a_{m/q}(f)   (second term iff q | m;
    // at m = 0 it contributes, giving the Eisenstein constant term 1 + q^{k-1}).
    const uint32_t qk1 = ModArithmetic::mod_pow(q % ell, k - 1u, ell);

    Mat T(d, d, ell);
    for (uint32_t i = 0; i < d; ++i) {
        for (uint32_t j = 0; j < d; ++j) {
            uint32_t v = miller[j][q * i];
            if (i % q == 0u) {
                v = ModArithmetic::mod_add(v, ModArithmetic::mod_mul(qk1, miller[j][i / q], ell), ell);
            }
            T.data[i][j] = v;
        }
    }
    return T;
}

// ---------------------------------------------------------------------------
// SturmState
// ---------------------------------------------------------------------------

bool SturmState::equals(const SturmState& o, bool ignore_budget_history) const {
    if (k != o.k || ell != o.ell || dim_M_k != o.dim_M_k) return false;
    if (cap != o.cap) return false;
    if (!ignore_budget_history && budget_history != o.budget_history) return false;
    if (B_reached != o.B_reached || primes_done != o.primes_done) return false;
    if (precision != o.precision || miller != o.miller) return false;
    if (closure_basis != o.closure_basis || closure_dim != o.closure_dim) return false;
    if (outcome != o.outcome || B_prime != o.B_prime) return false;
    if (tq_cache.size() != o.tq_cache.size()) return false;
    for (size_t i = 0; i < tq_cache.size(); ++i) {
        if (tq_cache[i].rows != o.tq_cache[i].rows) return false;
        if (tq_cache[i].cols != o.tq_cache[i].cols) return false;
        if (tq_cache[i].p != o.tq_cache[i].p) return false;
        if (tq_cache[i].data != o.tq_cache[i].data) return false;
    }
    return true;
}

namespace {

void AppendU32List(std::ostringstream& os, const std::vector<uint32_t>& v) {
    os << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) os << ",";
        os << v[i];
    }
    os << "]";
}

void AppendU32Table(std::ostringstream& os, const std::vector<std::vector<uint32_t>>& t) {
    os << "[";
    for (size_t i = 0; i < t.size(); ++i) {
        if (i) os << ",";
        AppendU32List(os, t[i]);
    }
    os << "]";
}

void AppendMatrix(std::ostringstream& os, const Mat& M) {
    os << "[";
    for (uint32_t i = 0; i < M.rows; ++i) {
        if (i) os << ",";
        AppendU32List(os, M.data[i]);
    }
    os << "]";
}

} // namespace

std::string SturmState::to_json() const {
    std::ostringstream os;
    os << "{\"kind\":\"sturm_bound\"";
    os << ",\"k\":" << k;
    os << ",\"ell\":" << ell;
    os << ",\"dim_M_k\":" << dim_M_k;
    os << ",\"cap\":" << cap;
    os << ",\"budget_history\":";  AppendU32List(os, budget_history);
    os << ",\"B_reached\":" << B_reached;
    os << ",\"primes_done\":";     AppendU32List(os, primes_done);
    os << ",\"precision\":" << precision;
    os << ",\"miller\":";          AppendU32Table(os, miller);
    os << ",\"T_q\":[";
    for (size_t i = 0; i < tq_cache.size(); ++i) {
        if (i) os << ",";
        os << "{\"q\":" << (i < primes_done.size() ? primes_done[i] : 0u) << ",\"matrix\":";
        AppendMatrix(os, tq_cache[i]);
        os << "}";
    }
    os << "]";
    os << ",\"closure_dim\":" << closure_dim;
    os << ",\"closure_basis\":";   AppendU32Table(os, closure_basis);
    os << ",\"outcome\":\"" << ToString(outcome) << "\"";
    os << ",\"B_prime\":";
    if (outcome == SturmOutcome::Found) os << B_prime; else os << "null";
    os << "}";
    return os.str();
}

// ---------------------------------------------------------------------------
// SturmBound
// ---------------------------------------------------------------------------

uint32_t SturmBound::weight_bound(uint32_t k) { return k / 12u + 1u; }

uint32_t SturmBound::default_cap(uint32_t k) { return 12u * weight_bound(k); }

uint32_t SturmBound::dimension_M_k(uint32_t k) {
    if (k % 2u != 0u) return 0u;
    if (k == 2u) return 0u;
    if (k % 12u == 2u) return k / 12u;
    return k / 12u + 1u;
}

SturmBound::SturmBound(uint32_t k, uint32_t ell, uint32_t cap) {
    ENGINE_CHECK_MSG(k >= 4u && k % 2u == 0u,
                     "SturmBound: the weight k = n + 2 must be even and at least 4 "
                     "(M_k = 0 otherwise, so B'(k, ell) is not defined)");
    ENGINE_CHECK_MSG(ell >= 2u, "SturmBound: ell must be at least 2");
    st_.k       = k;
    st_.ell     = ell;
    st_.dim_M_k = dimension_M_k(k);
    ENGINE_CHECK_MSG(st_.dim_M_k >= 1u, "SturmBound: dim M_k is zero");
    st_.cap = (cap == 0u) ? default_cap(k) : cap;
    ENGINE_CHECK_MSG(st_.cap >= 2u, "SturmBound: the cap must be at least 2");
}

void SturmBound::grow_precision_(uint32_t q) {
    // L >= q * dim M_k covers the a_{qm} reads for m <= dim M_k - 1 with room to
    // spare ([X] Sec. 1.3.1 R3.3 item 1).
    const uint64_t want = static_cast<uint64_t>(q) * st_.dim_M_k + 1u;
    ENGINE_CHECK_MSG(want <= 0xFFFFFFFFull, "SturmBound: precision overflow");
    const uint32_t need = static_cast<uint32_t>(want);
    if (st_.precision >= need && !st_.miller.empty()) return;
    st_.precision = need;
    // Regenerating is deterministic and stable in L: the echelon basis's
    // coefficient at index m does not depend on the precision it was computed
    // at (the back-substitution is driven by the columns m < dim M_k only), so
    // the already-cached T_q stay valid.
    st_.miller = EchelonMillerBasis(st_.k, st_.ell, st_.precision);
}

// (slot-22 PERF1, P9) The closure is EXTENDED by the newly appended generators
// rather than recomputed from the identity at every prime (which made the
// closure 95 % of the bound's wall at k = 702: the full re-closure is repeated
// for each of the B' primes, and the bound for each non-cyclic ideal).
//
// Invariant: closure_span_[0] is the span of the unital algebra generated by
// st_.tq_cache[0 .. closure_gens_), i.e. it contains I and is closed under
// right multiplication by each of those generators.  Adding generators g:
//  (1) right-multiply every element of a basis of the OLD span by each new g;
//  (2) right-multiply every vector that (1) or (2) added by EVERY generator.
// The result contains I, contains the old span, and is closed under right
// multiplication by every generator (an element is a combination of the old
// basis and the added vectors, and each of those has had each relevant product
// inserted), so it is the span of the generated algebra -- the same SET as
// hecke::algebra_closure's.  RrefRowSpan keeps its rows the REDUCED row echelon
// form of their span, which is unique, so closure_basis / closure_dim are
// identical to hecke::algebra_closure's output entry for entry (the
// certificate's clauses 7 and 9 still recompute them with algebra_closure from
// scratch, so this is checked on every certificate).
void SturmBound::recompute_closure_() {
    const uint32_t D = st_.dim_M_k, ell = st_.ell;
    if (closure_span_.empty() || closure_gens_ > st_.tq_cache.size()) {
        closure_span_.clear();
        closure_span_.emplace_back(D * D, ell);
        ENGINE_CHECK_MSG(closure_span_[0].insert(hecke::VecOfMatrix(Mat(D, ell))),
                         "SturmBound: the identity did not enter the empty closure span");
        closure_gens_ = 0;
    }
    hecke::RrefRowSpan& span = closure_span_[0];
    const std::vector<Mat>& gens = st_.tq_cache;
    for (; closure_gens_ < gens.size(); ++closure_gens_) {
        const Mat& g = gens[closure_gens_];
        ENGINE_CHECK_MSG(g.rows == D && g.cols == D && g.p == ell, "SturmBound: a T_q is not dim M_k x dim M_k over ell");
        if (span.dim() == D * D) continue;   // already everything: nothing can be added
        std::vector<Mat> frontier;
        {
            // (1) a snapshot of the old basis: the rows move under insert().
            const std::vector<std::vector<uint32_t>> old_rows = span.rows();
            for (const std::vector<uint32_t>& r : old_rows) {
                Mat prod = MatMulExact(hecke::MatrixOfVec(r, D, D, ell), g);
                if (span.insert(hecke::VecOfMatrix(prod))) frontier.push_back(std::move(prod));
            }
        }
        // (2) the added vectors, by every generator up to and including g.
        for (size_t head = 0; head < frontier.size(); ++head) {
            const Mat cur = frontier[head];
            for (size_t gi = 0; gi <= closure_gens_; ++gi) {
                Mat prod = MatMulExact(cur, gens[gi]);
                if (span.insert(hecke::VecOfMatrix(prod))) frontier.push_back(std::move(prod));
            }
        }
    }
    st_.closure_basis = span.rows();
    st_.closure_dim = span.dim();
}

void SturmBound::advance_() {
    if (st_.outcome == SturmOutcome::Found) return;
    uint32_t q = NextPrimeAfter(st_.B_reached);
    while (q <= st_.cap) {
        grow_precision_(q);
        st_.primes_done.push_back(q);
        st_.tq_cache.push_back(HeckeOperatorOnMk(q, st_.k, st_.ell, st_.miller));
        st_.B_reached = q;
        recompute_closure_();
        if (st_.closure_dim == st_.dim_M_k) {
            st_.outcome = SturmOutcome::Found;
            st_.B_prime = q;
            return;
        }
        q = NextPrimeAfter(q);
    }
    st_.outcome = SturmOutcome::NoneAtCap;
    st_.B_prime = 0u;
}

SturmOutcome SturmBound::run() {
    ENGINE_CHECK_MSG(st_.budget_history.empty(),
                     "SturmBound::run: already run -- use resume(new_cap)");
    st_.budget_history.push_back(st_.cap);
    advance_();
    return st_.outcome;
}

SturmOutcome SturmBound::resume(uint32_t new_cap) {
    ENGINE_CHECK_MSG(!st_.budget_history.empty(),
                     "SturmBound::resume: nothing to resume -- run() first");
    ENGINE_CHECK_MSG(new_cap >= st_.B_reached,
                     "resume: budget below reached state");
    // R3.3 item 1: on a complete (found) state resume is a no-op -- it returns
    // the same outcome AND the same certificate, so it must not move the cap or
    // append to the budget history either.
    if (st_.outcome == SturmOutcome::Found) return st_.outcome;
    st_.cap = new_cap;
    st_.budget_history.push_back(new_cap);
    st_.outcome = SturmOutcome::InProgress;
    advance_();
    return st_.outcome;
}

const MatrixT<uint32_t>& SturmBound::T(uint32_t q) const {
    for (size_t i = 0; i < st_.primes_done.size(); ++i) {
        if (st_.primes_done[i] == q) return st_.tq_cache[i];
    }
    ENGINE_CHECK_MSG(false, "SturmBound::T: q is not among the processed primes");
    return st_.tq_cache.at(0);
}

SturmBoundCertificate SturmBound::certificate() const {
    SturmBoundCertificate c;
    c.k           = st_.k;
    c.ell         = st_.ell;
    c.dim_M_k     = st_.dim_M_k;
    c.cap         = st_.cap;
    c.has_B_prime = (st_.outcome == SturmOutcome::Found);
    c.B_prime     = c.has_B_prime ? st_.B_prime : 0u;
    c.closure_dim = st_.closure_dim;
    c.primes      = st_.primes_done;
    c.T_q         = st_.tq_cache;
    c.state       = st_;
    return c;
}

SturmFinality sturm_finality(const SturmBound& bound, SturmUnavailableMode mode) {
    SturmFinality out;
    switch (bound.outcome()) {
        case SturmOutcome::Found:
            // The bound exists: it does not obstruct finality.  Whether the
            // IDEAL is final is E2's is_final (locality + a generation
            // certificate + the proof range of [E] `cl:sturmmodl`).
            out.value = SturmFinal::True;
            out.reason.clear();
            return out;
        case SturmOutcome::NoneAtCap:
            if (mode == SturmUnavailableMode::Heuristic) {
                out.value  = SturmFinal::Heuristic;
                out.reason = "sturm_heuristic";
            } else {
                out.value  = SturmFinal::False;
                out.reason = "sturm_unavailable";
            }
            return out;
        case SturmOutcome::InProgress:
        default:
            out.value  = SturmFinal::False;
            out.reason = "budget_exhausted:sturm_bound";
            return out;
    }
}

// ---------------------------------------------------------------------------
// SturmBoundCertificate
// ---------------------------------------------------------------------------

const char* SturmBoundCertificate::verify_message(uint32_t code) {
    switch (code) {
        case 0:  return "ok";
        case 1:  return "the weight/modulus/dimension header is inconsistent";
        case 2:  return "the cap is below 2, or B' exceeds it";
        case 3:  return "the prime list is not a strictly ascending list of primes, "
                        "or is not aligned with the matrix list";
        case 4:  return "the prime list is not exactly the primes <= B' (resp. <= cap)";
        case 5:  return "a stored T_q differs from the matrix re-derived from the q-expansions";
        case 6:  return "the stored matrices do not commute";
        case 7:  return "the recomputed closure dimension differs from the stored closure_dim";
        case 8:  return "the closure dimension does not match the claimed outcome";
        case 9:  return "B' is not minimal: the primes below it already generate everything";
        case 10: return "the resumable state does not agree with the certificate";
        default: return "unknown code";
    }
}

uint32_t SturmBoundCertificate::verify() const {
    // Clause 1 -- the header.
    if (k < 4u || k % 2u != 0u) return 1u;
    if (ell < 2u) return 1u;
    if (dim_M_k == 0u || dim_M_k != SturmBound::dimension_M_k(k)) return 1u;

    // Clause 2 -- the budget.
    if (cap < 2u) return 2u;
    if (has_B_prime && (B_prime < 2u || B_prime > cap)) return 2u;

    // Clause 3 -- the prime list.
    if (primes.size() != T_q.size()) return 3u;
    for (size_t i = 0; i < primes.size(); ++i) {
        if (!IsPrime(primes[i])) return 3u;
        if (i > 0 && primes[i] <= primes[i - 1]) return 3u;
        if (T_q[i].rows != dim_M_k || T_q[i].cols != dim_M_k || T_q[i].p != ell) return 3u;
    }

    // Clause 4 -- it must be EXACTLY the primes up to the claimed reach, so a
    // dropped T_q cannot pass.
    {
        const uint32_t reach = has_B_prime ? B_prime : cap;
        std::vector<uint32_t> expected;
        for (uint32_t q = 2; q <= reach; ++q) {
            if (IsPrime(q)) expected.push_back(q);
        }
        if (expected != primes) return 4u;
    }

    // Clause 5 -- re-derive every matrix from the q-expansions mod ell.  This is
    // a deterministic recomputation, not a search.
    if (!primes.empty()) {
        const uint32_t q_max = primes.back();
        const uint64_t want  = static_cast<uint64_t>(q_max) * dim_M_k + 1u;
        if (want > 0xFFFFFFFFull) return 5u;
        const std::vector<std::vector<uint32_t>> miller =
            EchelonMillerBasis(k, ell, static_cast<uint32_t>(want));
        for (size_t i = 0; i < primes.size(); ++i) {
            const Mat ref = HeckeOperatorOnMk(primes[i], k, ell, miller);
            if (ref.data != T_q[i].data) return 5u;
        }
    }

    // Clause 6 -- the Hecke operators commute (a cheap structural re-check).
    for (size_t i = 0; i < T_q.size(); ++i) {
        for (size_t j = i + 1; j < T_q.size(); ++j) {
            const Mat ab = T_q[i] * T_q[j];
            const Mat ba = T_q[j] * T_q[i];
            if (ab.data != ba.data) return 6u;
        }
    }

    // Clause 7 -- recompute the closure dimension; never read closure_dim.
    uint32_t dim = 0;
    (void)hecke::algebra_closure(T_q, dim_M_k, ell, dim);
    if (dim != closure_dim) return 7u;

    // Clause 8 -- the outcome.
    if (has_B_prime) {
        if (dim != dim_M_k) return 8u;
    } else {
        if (dim >= dim_M_k) return 8u;
    }

    // Clause 9 -- minimality of B' (the primes strictly below it must NOT
    // already generate M_k's endomorphism image).
    if (has_B_prime && primes.size() >= 2u) {
        std::vector<Mat> below(T_q.begin(), T_q.end() - 1);
        uint32_t dim_below = 0;
        (void)hecke::algebra_closure(below, dim_M_k, ell, dim_below);
        if (dim_below >= dim_M_k) return 9u;
    }

    // Clause 10 -- the resumable state agrees with what the certificate claims.
    if (state.k != k || state.ell != ell || state.dim_M_k != dim_M_k) return 10u;
    if (state.cap != cap) return 10u;
    if (state.primes_done != primes) return 10u;
    if (state.closure_dim != closure_dim) return 10u;
    if (has_B_prime) {
        if (state.outcome != SturmOutcome::Found || state.B_prime != B_prime) return 10u;
    } else if (state.outcome != SturmOutcome::NoneAtCap) {
        return 10u;
    }
    if (state.budget_history.empty()) return 10u;
    if (state.budget_history.back() != cap) return 10u;

    return 0u;
}

std::string SturmBoundCertificate::resumable_json() const {
    const std::string state_json = state.to_json();
    std::ostringstream os;
    os << "{\"kind\":\"sturm_bound\"";
    os << ",\"complete\":" << (has_B_prime ? "true" : "false");
    os << ",\"budget\":{\"cap\":" << cap << "}";
    os << ",\"reached\":{\"B_reached\":" << state.B_reached
       << ",\"primes\":" << state.primes_done.size()
       << ",\"precision\":" << state.precision
       << ",\"closure_dim\":" << closure_dim
       << ",\"dim_M_k\":" << dim_M_k << "}";
    os << ",\"budget_history\":";
    AppendU32List(os, state.budget_history);

    // R3.1 item 6: inline up to 64 KiB, sidecar beyond it.  Only the record
    // layer (E2) knows the record stem, so without one we inline anyway and say
    // so rather than emitting a path that names no file.
    const bool oversize = state_json.size() > 65536u;
    if (!oversize) {
        os << ",\"state\":" << state_json;
    } else if (!state_sidecar_stem.empty()) {
        const std::string path = state_sidecar_stem + ".state.sturm_bound.json";
        std::ofstream out(path.c_str(), std::ios::binary);
        ENGINE_CHECK_MSG(out.good(), "SturmBoundCertificate: cannot write the state sidecar");
        out << state_json;
        out.close();
        os << ",\"state_ref\":{\"path\":\"" << path
           << "\",\"fnv64\":\"" << HexU64(Fnv1a64(state_json))
           << "\",\"bytes\":" << state_json.size() << "}";
    } else {
        os << ",\"state_oversize\":true";
        os << ",\"state_fnv64\":\"" << HexU64(Fnv1a64(state_json)) << "\"";
        os << ",\"state\":" << state_json;
    }
    os << "}";
    return os.str();
}

std::string SturmBoundCertificate::to_json() const {
    std::ostringstream os;
    os << "{\"certificate\":\"sturm_bound\"";
    os << ",\"k\":" << k;
    os << ",\"ell\":" << ell;
    os << ",\"dim_M_k\":" << dim_M_k;
    os << ",\"cap\":" << cap;
    os << ",\"B_prime\":";
    if (has_B_prime) os << B_prime; else os << "null";
    os << ",\"closure_dim\":" << closure_dim;
    os << ",\"primes\":";
    AppendU32List(os, primes);
    os << ",\"T_q\":[";
    for (size_t i = 0; i < T_q.size(); ++i) {
        if (i) os << ",";
        os << "{\"q\":" << primes[i] << ",\"matrix\":";
        AppendMatrix(os, T_q[i]);
        os << "}";
    }
    os << "]";
    os << ",\"verify\":" << verify();
    os << ",\"resumable\":" << resumable_json();
    os << "}";
    return os.str();
}

} // namespace hecke_ext
