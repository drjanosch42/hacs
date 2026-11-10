// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// Kummer.h -- the local ell-th-power / unramifiedness tests of R18-2 ([E]
// sec:lvnf, PROVED in docs/implementation/slot18/R18-2_report.md section 1)
// and its Z/ell^2 layer by the equivariant route of slot-19 R19-3 (slot-19 WP-NF).
//
// Layer 1 (over Q(zeta_ell)): eta_i = prod_{c=2}^{ell-1} u_c^{f_c}, f_c = c^{-i}
// mod ell in (-ell/2, ell/2). For a unit eta with v := v_pi(eta^{ell-1} - 1):
// K(eta^{1/ell})/K is unramified at lambda = (1 - zeta) iff v >= ell, and lambda
// splits iff v >= ell + 1 ((U^(1))^ell = U^(ell+1)).
//
// Layer 2 (over Q(zeta_{ell^2}), e = ell(ell - 1)): the ell lifts kappa = w mod
// (ell - 1) of omega^w; eta_kappa == A * B^{-kappa ell} modulo ell^2-th powers
// (Theta_kappa = E_w (S - kappa ell T) in (Z/ell^2)[G]); local test
// W^{ell^2} = x_0^{ell^2 Z_ell} (W cap U^(thr2)), thr2 = w + (ell - 1) + 2e,
// evaluated greedily (one a mod ell cancels the leading term at
// v(x_0^{ell^2} - 1)). For units: unramified <=> local ell^2-th power.
// Ring: (Z/ell^3)[z]/Phi_{ell^2} (precision pi^{3e}).
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "BigInt.h"
#include "Cyclotomic.h"

namespace nf {

// ell | B_i  <=>  ell^2 | sum_{a<ell} a^i  ((ell - 1) not dividing i, i even): the Kummer-criterion oracle.
bool ell_divides_bernoulli(uint32_t ell, uint32_t i);

// The exact eta_i of layer 1 on the power basis of Z[zeta_ell] (BigInt coefficients).
struct ExactEta {
    uint32_t ell = 0, i = 0;
    std::map<uint32_t, int64_t> exponents;  // f_c, c = 2 .. ell - 1
    std::vector<BigInt> coeffs;             // a_0 .. a_{ell-2}
    std::size_t max_digits = 0;             // decimal digits of max |a_j|
};
ExactEta exact_eta_layer1(uint32_t ell, uint32_t i);

struct Layer1Result {
    uint32_t ell = 0, i = 0;
    unsigned v_pi = 0;        // v_pi(eta^{ell-1} - 1), computed mod ell^prec (>= prec (ell-1) if 0)
    bool unramified = false;  // v >= ell
    bool lambda_split = false;  // v >= ell + 1
    bool not_lth_power = false;  // eta(r)^{(q-1)/ell} != 1 at every split prime used
    bool eigen_ok = false;       // eta(r^t)^h == (eta(r)^h)^{t^i} at every split prime, t = 2..ell-1 (sampled for big ell)
    bool unit_ok = false;        // N(eta) = +-1 at every split prime used (only for ell <= 101; else true)
    bool ell_divides_Bi = false;
    bool oracle_agrees = false;  // (v > i) == ell | B_i
    std::vector<uint32_t> split_primes;
    uint64_t products = 0;
};
// The test on the product form mod ell^prec (prec >= 2).
Layer1Result kummer_layer1(uint32_t ell, uint32_t i, unsigned prec = 2, unsigned n_split = 3);
// The same local valuation for an arbitrary unit given exactly (e.g. eta^ell, a negative control).
unsigned layer1_vpi_of(uint32_t ell, const std::vector<BigInt>& eta, unsigned prec = 3);

struct Layer2Row {
    uint32_t t = 0;
    uint64_t kappa = 0;  // mod ell(ell - 1)
    uint32_t u = 0;      // (1 - kappa) mod ell
    unsigned v_w = 0, best_lth = 0, best_l2th = 0;
    uint32_t a_l2 = 0;
    bool local_lth_power = false, unramified = false;
    bool brute_checked = false, greedy_eq_brute = true;
    unsigned brute_lth = 0, brute_l2th = 0;
};

struct Layer2Global {
    uint64_t kappa = 0;
    bool not_global_lth_power = false, eigen_eps_kappa = false;
    int64_t sum_f = 0, sumabs_f = 0;
    std::vector<uint32_t> split_primes;
};

struct Layer2Options {
    std::vector<uint32_t> brute_ts;   // lifts t at which R18-2's full scan over a mod ell is compared
    bool grouping_identity = true;    // check Theta_kappa == E_w (S - kappa ell T) coefficientwise (all lifts)
    bool global_tests = true;         // on the canonical generator of every unramified lift
};

struct Layer2Result {
    uint32_t ell = 0, w = 0, g = 0;
    std::vector<uint32_t> delta_factors;  // the q^a || ell - 1
    unsigned v_x0 = 0, L1 = 0, L2 = 0, thr1 = 0, thr2 = 0;
    bool grouping_identity = false;
    std::vector<Layer2Row> rows;
    std::vector<uint64_t> unramified_kappas;
    std::vector<Layer2Global> global;
    uint64_t products = 0;
    std::string summary() const;
};
Layer2Result kummer_layer2_equivariant(uint32_t ell, uint32_t w, const Layer2Options& opt = Layer2Options());

}  // namespace nf
