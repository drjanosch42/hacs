// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// CwLayer.h -- the Coates--Wiles measure from the layer Q_ell(zeta_{ell^m})
// (slot-19 WP-NF2: the C++ port of R19-7's level2.py / levelm.py;
// docs/implementation/slot19/R19-7_report.md sec. 1.1-1.3, 2, 8).
//
// One Iwasawa logarithm lambda_m = log_ell(1 - zeta_{ell^m}) = n^{-1} log(pi^n / ell)
// in R_m = Z_ell[z]/Phi_{ell^m} mod ell^W serves every class: the measure is
//   nu(a + ell^m Z_ell) = ell^{-m} sum_{j=1}^m Tr_j(zeta_{ell^j}^{-a} lambda_j),  lambda_j = Tr_{m/j} lambda_m,
// projected to omega^w on the ell^{m-1} representatives (1+ell)^i, and
//   Psi_k = (ell-1) sum_i (1+ell)^{ik} nu_i == (1 - ell^{k-1}) B_k / k   (mod ell^m),  k == w (ell - 1).
// No Bernoulli number enters the route; bernoulli_ref() is the independent reference.
//
// Arithmetic: coefficients mod ell^W in base-ell^d limbs (LadicInt.h); products by
// an exact multi-prime convolution (Ntt64.h, 62-bit primes, transform length
// f * 2^a), folded mod z^{ell^m} - 1 and Phi_{ell^m} per prime, Garner into limbs.
// pi^n comes from the binomial coefficients mod ell^{W+1} (no ring products).
// Memory: two live ring elements + two transform buffers + P residue vectors.
#include <cstdint>
#include <string>
#include <vector>

#include "BigRational.h"
#include "LadicInt.h"
#include "Ntt64.h"

namespace nf {

struct CwRingStats {
    uint64_t squarings = 0, products = 0, ntts = 0;
    double ntt_secs = 0, io_secs = 0, garner_secs = 0;
};

class CwRing {
public:
    using Elt = std::vector<uint32_t>;  // n * K limbs, coefficient-major

    // direct_log: power-of-two sub-transforms above 2^direct_log go four-step (tests lower it).
    CwRing(uint32_t ell, unsigned m, unsigned W, unsigned threads, unsigned direct_log = 15);

    const LadicCtx& ctx() const { return cx_; }
    uint32_t ell() const { return ell_; }
    unsigned m() const { return m_; }
    unsigned W() const { return cx_.W; }
    uint64_t n() const { return n_; }      // phi(ell^m)
    uint64_t N() const { return N_; }      // ell^m
    uint64_t blk() const { return blk_; }  // ell^{m-1}
    unsigned threads() const { return threads_; }
    std::size_t ntt_length() const { return L_; }
    unsigned prime_count() const { return static_cast<unsigned>(plans_.size()); }

    Elt zero() const { return Elt(n_ * cx_.K, 0); }
    Elt one() const;
    // From signed integer coefficients of any length (reduced mod z^N - 1, Phi, ell^W).
    Elt from_i64(const std::vector<int64_t>& c) const;

    void mul_inplace(Elt& a, const Elt& b);  // a <- a b; &a == &b squares
    void sqr_inplace(Elt& a) { mul_inplace(a, a); }
    void pow_inplace(Elt& a, uint64_t e);    // left-to-right: live a, the base copy
    void add_inplace(Elt& a, const Elt& b) const;
    void sub_inplace(Elt& a, const Elt& b) const;
    void scal_inplace(Elt& a, const LVal& c) const;
    Elt sigma(const Elt& a, uint64_t c) const;  // z -> z^c (small rings only: a dense length-N temporary)
    uint32_t residue(const Elt& a) const;       // a mod pi, in F_ell
    unsigned content_val(const Elt& a) const;   // min_i v_ell(a_i) (W if a == 0)
    Elt schoolbook_mul(const Elt& a, const Elt& b) const;  // O(n^2) reference (tests)

    // sha256 (first 16 hex) of the (W, n) int64 little-endian ell-adic digit array (R19-7's numpy layout).
    std::string digits_sha16(const Elt& a) const;

    // The planned peak of the product stage in bytes (two elements, two buffers, P residue vectors).
    static uint64_t planned_product_peak_bytes(uint32_t ell, unsigned m, unsigned W);
    static unsigned planned_prime_count(uint32_t ell, unsigned m, unsigned W);

    CwRingStats stats;

private:
    void fill_residues(const Elt& a, unsigned j, std::vector<uint64_t>& buf);

    uint32_t ell_;
    unsigned m_;
    uint64_t n_, N_, blk_;
    unsigned threads_;
    LadicCtx cx_;
    std::size_t L_;
    std::vector<Ntt64Plan> plans_;
    std::vector<std::vector<uint64_t>> betapow_m_;  // per prime: beta^t R mod p (t < K)
    std::vector<uint64_t> off_mod_p_, scale_m_;     // per prime: OFF mod p; L^{-1} R^2 mod p
    std::vector<std::vector<uint64_t>> garner_pk_m_;  // [j][k] = (p_k mod p_j) R mod p_j
    std::vector<uint64_t> garner_inv_m_;              // [j] = (p_0..p_{j-1})^{-1} R mod p_j
    std::vector<LVal> garner_C_;                      // [j] = p_0..p_{j-1} mod ell^W
    unsigned tdigits_ = 3;
};

// pi^n / ell mod ell^W (n = phi(ell^m), pi = 1 - z). binomial = true: from C(n, i) mod ell^{W+1};
// false: by ring products at precision W + 1 (the cross-check).
CwRing::Elt cw_pi_n_over_ell(CwRing& R, bool binomial);

struct CwLogOptions {
    unsigned r = 0;       // ell^r-th power before the series (0: m + 1, R19-7's default)
    bool binomial_pi = true;
    int perturb = 0;      // negative controls: 1 = unit times u_2, 2 = one digit of pi^n/ell changed
};
struct CwLog {
    CwRing::Elt Lg;       // lambda_m = Lg / ell^s
    unsigned r = 0, s = 0, cy = 0, terms = 0;
    double secs_pi = 0, secs_pow = 0, secs_series = 0;
};
CwLog cw_lambda(CwRing& R, const CwLogOptions& opt);
// ell^r log_ell(v) mod ell^W for a unit v (Teichmueller-normalised, then (w^{ell^r} - 1) and the series).
CwRing::Elt cw_log_unit(CwRing& R, CwRing::Elt v, unsigned r, unsigned* cy = nullptr, unsigned* terms = nullptr,
                        double* secs_pow = nullptr, double* secs_series = nullptr);

// The omega^w part of the level-j measure of y (an element of R_j mod ell^W on its power basis,
// n_j coefficients, ctx limbs), divided by ell^scale: nu_i at a_i = (1+ell)^i mod ell^j, i < ell^{j-1},
// returned mod ell^Mout; integral = every projected sum divisible by ell^scale.
struct CwMeasure {
    uint32_t w = 0;
    unsigned level = 0;
    bool integral = false;
    std::vector<uint32_t> nu;  // mod ell^Mout
};
std::vector<CwMeasure> cw_measure(const LadicCtx& cx, unsigned level, const uint32_t* y, unsigned scale,
                                  const std::vector<uint32_t>& classes, unsigned Mout, unsigned threads);

// Psi(k_t) = (ell - 1) sum_i nu_i X_t^i mod ell^M, X_t = (1+ell)^{k_t}, k_t = w + (ell-1) t, t < rows.
std::vector<uint32_t> cw_moments(const std::vector<uint32_t>& nu, uint32_t ell, unsigned M, uint32_t w, uint64_t rows,
                                 unsigned threads);
// k_t with R19-7's convention (k < 2 is replaced by k + n_m).
inline uint64_t cw_row_k(uint32_t ell, unsigned m, uint32_t w, uint64_t t) {
    uint64_t k = w + static_cast<uint64_t>(ell - 1) * t;
    if (k < 2) {
        uint64_t nm = ell - 1;
        for (unsigned i = 1; i < m; ++i) nm *= ell;
        k += nm;
    }
    return k;
}

// B_{2k} / (2k) exactly for 2k <= kmax (tangent numbers; index by k = 2, 4, ...: out[k] for even k).
std::vector<BigRational> bernoulli_over_k(unsigned kmax);
// (1 - ell^{k-1}) B_k / k mod ell^M for the rows of class w: exact for k <= kexact, else the degree-(M-1)
// polynomial in t through the first M exact weights (R19-7 levelm.py's Kummer-law reference).
// law_checked / law_held: further exact weights on which the polynomial was compared.
struct CwBernoulliRef {
    std::vector<uint32_t> ref;  // per row t
    uint64_t exact_rows = 0, law_checked = 0, law_held = 0;
};
CwBernoulliRef cw_bernoulli_ref(uint32_t ell, unsigned m, unsigned M, uint32_t w, uint64_t rows,
                                const std::vector<BigRational>& Bk, unsigned kexact);

// ---------------------------------------------------------------- one cell
struct CwCellSpec {
    uint32_t ell = 0;
    unsigned m = 2, r = 0, W = 0;  // r = 0: m + 1; W = 0: r + 3m
    std::vector<uint32_t> classes;
    bool control = true;     // the level-(m-1) pushforward (m >= 2)
    bool hash = true;
    unsigned bern_kmax = 0;  // exact Bernoulli reference up to this k (0: none)
    const std::vector<BigRational>* bern_table = nullptr;  // optional precomputed bernoulli_over_k(>= bern_kmax)
    CwLogOptions log;        // r / perturb (r overridden by the field above)
};
struct CwClassResult {
    uint32_t w = 0;
    bool integral = false;
    std::vector<uint32_t> psi;            // per row t, mod ell^m
    std::vector<uint64_t> zeros_count;    // [j] = rows with Psi == 0 mod ell^j, j = 1..m
    std::vector<uint64_t> zeros_top;      // the rows' k with Psi == 0 mod ell^m (at most 16)
    int64_t alpha = -1;                   // x_kappa mod ell^m at the unique zero, else -1
    uint32_t leopoldt = 0;                // -(ell-1) sum nu_i mod ell^m = L_ell(1, omega^w) mod ell^m
    bool control_run = false, control_integral = false;
    uint64_t control_right_low = 0, control_right_high = 0;  // Psi_low == Psi mod ell^{m-1} / mod ell^m
    bool delta_law = false;               // m-th differences of Psi in t vanish mod ell^m (moment self-check)
    bool bern_run = false;
    uint64_t bern_rows_ok = 0, bern_exact_rows = 0, bern_law_checked = 0, bern_law_held = 0;
};
struct CwCellResult {
    CwCellSpec spec;
    unsigned r = 0, W = 0, s = 0, cy = 0, terms = 0, P = 0, K = 0;
    uint64_t n = 0, L = 0, rows = 0;
    std::string sha16;
    CwRingStats stats;
    double secs_pi = 0, secs_pow = 0, secs_series = 0, secs_measure = 0, secs_total = 0;
    uint64_t planned_peak_bytes = 0;
    std::vector<CwClassResult> classes;
};
// NF_CHECK refuses W < r + 3m - 1 (nu would be known mod less than ell^m).
CwCellResult cw_run_cell(const CwCellSpec& spec, unsigned threads);

// The cost model (fitted on this box, docs/implementation/slot19/wpnf2/README.md): per-product NTT work
// and the O(n) stages; returns predicted seconds at `threads` and the planned peak bytes.
struct CwCostPrediction {
    uint64_t n = 0, L = 0;
    unsigned P = 0, K = 0, r = 0, W = 0;
    uint64_t squarings = 0, products = 0, ntts = 0;
    double ntt_work = 0;         // sum over transforms of L log2 L
    double secs = 0;             // predicted wall
    uint64_t peak_bytes = 0;     // planned peak (max of the binomial stage and the product stage)
};
CwCostPrediction cw_predict(uint32_t ell, unsigned m, unsigned r, unsigned W, unsigned nclasses, unsigned threads);

}  // namespace nf
