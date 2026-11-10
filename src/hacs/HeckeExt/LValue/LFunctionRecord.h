// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_LVALUE_LFUNCTION_RECORD_H
#define HECKE_EXT_LVALUE_LFUNCTION_RECORD_H

// ===========================================================================
// (hecke-extensions slot 20, stretch S1; owner-Q3 roadmap P1) THE RECORD `lfunction/v1`: the one-variable truncated
// ell-adic L-function of an ordinary eigensymbol, read off the ell-stabilised MTT measure of [E] def:lvmtt (WP-MEAS).
// Contract: [E] 1.20.0-pre.5 sec:lvlfunction (def:lvlfun, lem:lvlfunprec, rmk:lvlfunschema); console verb `lfunction`
// ([P] v3.11.1 Sec. 3.6, read_only).  No new mathematics: per branch a in Z/(ell-1),
//
//     L_a(T) = int omega^a(x) (1+T)^{l_u(x)} dmu   in   Lambda / (ell^R, omega_{M-1}(T)),  omega_r = (1+T)^{ell^r} - 1,
//
// is [E] prop:lvmtt(5)'s branch element g in Z/ell^R[Gamma/Gamma^{ell^{M-1}}] written in the basis T^t, t < G = ell^{M-1}
// (c_t = sum_i g_i C(i, t), the unique representative of degree < G), with the CERTIFIED PRECISION of each coefficient
// against the true series (lem:lvlfunprec):  prec(0) = R,  prec(t) = min(R, M - 1 - floor(log_ell t))  (1 <= t < G).
//
// Pure host arithmetic (STL + the WP-MEAS headers); writes nothing; no golden object.  The record is identified by the
// FNV-1a-64 (Fnv.h) of its canonical JSON text.
// ===========================================================================

#include "ModularSymbolMeasure.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace hecke_ext {
namespace lvalue {

// The branch element g of branch a at level `level` (1 <= level <= me.M), from the level-`level` balls:
// g_i = sum_{delta in mu_{ell-1}} omega^a(delta) mu(delta u^i + ell^level Z_ell), i < ell^{level-1}.  At level == me.M it is me.branch(a).
std::vector<uint64_t> BranchElementAtLevel(const ModularSymbolMeasure& me, uint32_t a, uint32_t level);
// c_t = sum_i g_i C(i, t) mod q, t < g.size() (the Taylor shift X = 1 + T of sum_i g_i X^i).
std::vector<uint64_t> TCoefficients(const std::vector<uint64_t>& g, uint64_t q);
// lem:lvlfunprec: prec(0) = R, prec(t) = min(R, level - 1 - floor(log_ell t)), t < G = ell^{level-1}.
std::vector<uint32_t> CertifiedTPrecision(uint64_t ell, uint32_t R, uint32_t level);
// sum_t c_t x^t mod q.
uint64_t EvalTPolynomial(const std::vector<uint64_t>& c, uint64_t x, uint64_t q);
// The image of g in Z/q[C_{Gsmall}] (Gsmall | g.size()): the reduction modulo omega_r, Gsmall = ell^r.
std::vector<uint64_t> FoldBranchElement(const std::vector<uint64_t>& g, size_t Gsmall, uint64_t q);

struct LFunctionBranch {
    uint32_t a = 0;
    bool vanishes = false;                   // the wrong parity ((-1)^a != geometric sign): 0 identically
    std::vector<uint64_t> coefficients;      // c_t mod ell^R, t < G
    std::vector<uint32_t> precision;         // lem:lvlfunprec, t < G
    uint32_t mu = 0, lambda = 0;             // ModularSymbolMeasure::invariants(a) (prop:lvmtt(5))
    bool readable = false;
    int64_t lambda_mod_ell = -1;             // the first t < G with c_t a unit (certified for every t < G), -1 if none
    uint64_t values_modulus = 0;             // ell^min(M, R)
    std::vector<uint64_t> critical_values;   // L_a(u^j - 1) = int omega^{a-j}(x) x^j dmu mod ell^min(M,R), j = 0..n
};

// How the eigensymbol was chosen (the inputs of RunSymbolCell) and its eigen-system.
struct LFunctionForm {
    std::string separator;                   // e.g. "T_2+3T_3+7T_5"
    uint64_t target_root = 0;                // the root of charpoly(sep | W^-) mod ell that was used
    bool eisenstein_congruent = false;       // that root is the Eisenstein root sum c_p (1 + p^{k-1}) mod ell
    std::map<uint64_t, uint64_t> a_p;        // mod ell^R, p in {2,3,5,7,ell} and the separator's primes
    std::string sign;                        // "minus" (r^-, W^- = odd coordinates) | "plus" (r^+, even coordinates)
    std::string symbol_normalisation;        // "r^- primitive, first ell-unit coordinate 1" | "r^+_0 = 1"
};

struct LFunctionRecord {
    static constexpr const char* kSchema = "lfunction/v1";
    uint32_t n = 0, k = 0;
    uint64_t ell = 0;
    uint32_t R = 0, M = 0;
    uint64_t G = 0, u = 0;
    uint64_t a_ell = 0, alpha = 0;
    std::string geometric;                   // "odd" | "even" (the eigenvalue of rho_n(diag(-1,1)))
    LFunctionForm form;
    std::string normalisation;               // ModularSymbolMeasure::normalisation().to_string()
    std::vector<LFunctionBranch> branches;
    std::string canonical() const;           // canonical JSON (fixed key order, no whitespace)
    uint64_t hash() const;                   // FNV-1a-64 of canonical()
    std::string line(const LFunctionBranch& b) const;   // one LFUNCTION text line
};
// The record of the listed branches of a measure (every branch a in [0, ell-1) if `branches` is empty).
LFunctionRecord MakeLFunctionRecord(const ModularSymbolMeasure& me, const LFunctionForm& form, const std::vector<uint32_t>& branches);

// The symbol-side cell the console verb runs: RunSymbolCell(n, ell, R, sep, control = true) (the Eisenstein root if it occurs,
// else the first root mod ell), UnitRoot, BuildMeasure to level M -- the measure that BirchModularSymbolRoute::measure() returns on
// the class with this symbol ([E] chk:lvmeas (M0)).  Refusals (std::invalid_argument): ell < 5 or not prime, n odd or < 2, M < 2,
// R < 1, ell^R >= 2^62, ell^M > kMaxBalls, a non-simple root, no r^+ for sign "plus", a non-ordinary eigen-system.
struct LFunctionRequest {
    uint32_t n = 0;
    uint64_t ell = 0;
    uint32_t M = 2, R = 4;
    std::string sign = "minus";
    std::map<uint64_t, uint64_t> sep = {{2, 1}, {3, 3}, {5, 7}};
    std::vector<uint32_t> branches;
    static constexpr uint64_t kMaxBalls = 4000000;   // ell^M (the console's read-only budget: <= ~2 s, <= 0.2 GB)
};
LFunctionRecord ComputeLFunction(const LFunctionRequest& req);

} // namespace lvalue
} // namespace hecke_ext

#endif // HECKE_EXT_LVALUE_LFUNCTION_RECORD_H
