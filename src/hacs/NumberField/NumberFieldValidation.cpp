// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// NumberFieldValidation (fast) -- the gate of the NumberField library
// (hecke-extensions slot 19, WP-NF; [E] 1.19.0-pre.2 sec:lvnf; plan
// docs/implementation/slot19/00-slot19-plan.md "WP-NF").
//
// Oracle: slot-18 R18-2 (docs/implementation/slot18/R18-2_report.md, r182/:
// layer1.py, layer1_big.py, layer2.py, kummer_polys_l37.txt) and the
// slot-19 R19-3 prototype eqlayer2.py (equivariant layer 2; its 59 / 67 runs
// reproduce R18-2 section 2). Every expected value below is copied from
// those reports, never from this code's own output:
//   (1) BigInt / BigRational / NTT / mulmod64 self-tests (portable fallback
//       compared with the 128-bit path), exact Bernoulli numbers;
//   (2) exact eta_32, eta_6 in Z[zeta_37]: FNV-1a-64 of the 36 coefficients
//       (decimal, one per line), 127 / 31 digits, the exponents f_c, and the
//       36 coefficient strings against r182/kummer_polys_l37.txt when present;
//       v_pi(eta^36 - 1) = 38 / 6 exactly and mod 37^3;
//   (3) layer 1 at 37: all 17 even eigenspaces, v_i = i except v_32 = 38,
//       Kummer's criterion against the exact B_i;
//   (4) layer 2 (equivariant) at (37,32), (59,44), (67,58): the unique
//       unramified lift kappa = 284 / 914 / 3292 (u = y_0 = 13 / 31 / 59), the
//       levels, thresholds and best valuations, greedy == R18-2's full scan at
//       two lifts of 37, the global tests on the canonical generator;
//   (5) layer 1 at 691: v = 692 at i = 12, 200 (unramified, lambda split);
//   (6) negative controls: the regular index (37, 30) has no unramified lift;
//       eta_6^37 passes the LOCAL test but fails "not an ell-th power"; a
//       tampered eta_32 changes the FNV and is not a unit; i = 2 at 691 and the
//       15 regular indices at 37 are ramified;
//   (7) Q(alpha), Newton polygon, Hensel;
//   (8) check_numberfield_includes.sh (STL-only; no lower library includes it);
//   (9) slot-19 WP-NF2: the Coates--Wiles layer route (CwLayer.h) against
//       R19-7 (CwLayerValidation.cpp: lambda_m hashes, the level-2 rows, the
//       rungs 284 / 37580, negative controls; read-only env arms
//       NUMBERFIELD_LAYER_ALL=1 and NUMBERFIELD_LAYER_CELLS=...).
// Read-only, no golden object, no digest.
#include "NumberFieldValidation.h"

#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "../../ModArithmetic/Fnv.h"
#include "BigInt.h"
#include "BigRational.h"
#include "Cyclotomic.h"
#include "CwLayerValidation.h"
#include "Kummer.h"
#include "ModUtil.h"
#include "Ntt.h"
#include "PolyAlg.h"

namespace {

uint32_t g_checks = 0;
uint32_t g_fail = 0;

void Check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        std::cout << "  [FAIL] " << what << std::endl;
    }
}

template <typename A, typename B>
void CheckEq(const A& got, const B& want, const std::string& what) {
    ++g_checks;
    if (!(got == want)) {
        ++g_fail;
        std::ostringstream os;
        os << "  [FAIL] " << what << ": got " << got << ", want " << want;
        std::cout << os.str() << std::endl;
    }
}

double Now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool PathExists(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0;
}

std::string FindRepoRoot() {
    auto looks = [](const std::string& d) {
        return PathExists(d + "/docs/matrix-engine/tools") && PathExists(d + "/src/hacs");
    };
#ifdef OC_SOURCE_DIR
    {
        const std::string r(OC_SOURCE_DIR);
        if (!r.empty() && looks(r)) return r;
    }
#endif
    char buf[4096];
    if (::getcwd(buf, sizeof(buf)) != nullptr) {
        std::string dir(buf);
        for (int up = 0; up < 8; ++up) {
            if (looks(dir)) return dir;
            const std::size_t s = dir.find_last_of('/');
            if (s == std::string::npos || s == 0) break;
            dir = dir.substr(0, s);
        }
    }
    return std::string();
}

uint64_t FnvCoeffs(const std::vector<nf::BigInt>& c) {
    std::string s;
    for (const auto& x : c) s += x.to_string() + "\n";
    return engine::FnvBytes(engine::kFnvOffset64, s);
}

// ------------------------------------------------------------------ (1)
void SelfTests() {
    using nf::BigInt;
    CheckEq(BigInt::pow(BigInt(2), 127).operator-=(BigInt(1)).to_string(),
            std::string("170141183460469231731687303715884105727"), "BigInt 2^127 - 1");
    CheckEq((BigInt(-7) / BigInt(2)).to_string(), std::string("-3"), "BigInt truncated quotient -7/2");
    CheckEq((BigInt(-7) % BigInt(2)).to_string(), std::string("-1"), "BigInt remainder sign of dividend");
    CheckEq(BigInt(-7).floor_mod(BigInt(2)).to_string(), std::string("1"), "BigInt floor_mod");
    CheckEq(BigInt::gcd(BigInt::pow(BigInt(2), 100) * BigInt::pow(BigInt(3), 5),
                        BigInt::pow(BigInt(2), 60) * BigInt::pow(BigInt(3), 9) * BigInt(7)),
            BigInt::pow(BigInt(2), 60) * BigInt::pow(BigInt(3), 5), "BigInt gcd");
    CheckEq((BigInt::pow(BigInt(37), 5) * BigInt(2)).valuation(37), 5u, "BigInt valuation");
    CheckEq(BigInt::pow(BigInt(10), 1000).decimal_digits(), std::size_t(1001), "BigInt decimal digits");
    CheckEq(BigInt(INT64_MIN).to_string(), std::string("-9223372036854775808"), "BigInt INT64_MIN");
    std::mt19937_64 rng(20261003);
    auto rand_big = [&](std::size_t limbs, bool neg) {
        BigInt::Mag m(limbs);
        for (auto& x : m) x = static_cast<uint32_t>(rng());
        if (!m.empty() && m.back() == 0) m.back() = 1;
        return BigInt::from_magnitude(m, neg);
    };
    bool kara = true, divok = true, strok = true;
    for (std::size_t la : {33u, 64u, 100u, 257u}) {
        for (std::size_t lb : {32u, 70u, 257u}) {
            const BigInt a = rand_big(la, false), b = rand_big(lb, false);
            if (BigInt::mul_mag(a.magnitude(), b.magnitude()) != BigInt::mul_schoolbook(a.magnitude(), b.magnitude())) kara = false;
            for (int sg = 0; sg < 4; ++sg) {
                const BigInt x = (sg & 1) ? -a : a, y = (sg & 2) ? -b : b;
                BigInt q, r;
                BigInt::divmod(x, y, q, r);
                if (q * y + r != x || r.abs() >= y.abs() || (!r.is_zero() && r.is_neg() != x.is_neg())) divok = false;
            }
            if (BigInt::from_string(a.to_string()) != a || BigInt::from_string((-b).to_string()) != -b) strok = false;
        }
    }
    Check(kara, "BigInt Karatsuba == schoolbook (12 operand pairs, 33..257 limbs)");
    Check(divok, "BigInt Knuth-D divmod identity, all sign combinations");
    Check(strok, "BigInt decimal round trip");
    bool mm = true;
    for (int t = 0; t < 2000; ++t) {
        const uint64_t m = (rng() | 1u), a = rng(), b = rng();
        if (nf::mulmod64(a, b, m) != nf::mulmod64_portable(a, b, m)) mm = false;
    }
    Check(mm, "mulmod64 == portable fallback (2000 random 64-bit triples)");
    bool ntt = true;
    for (uint32_t M : {50653u, 300763u, 329939371u, 4294967291u}) {  // 37^3, 67^3, 691^3, a 32-bit prime
        std::vector<uint32_t> a(1500), b(1100);
        for (auto& x : a) x = static_cast<uint32_t>(rng() % M);
        for (auto& x : b) x = static_cast<uint32_t>(rng() % M);
        if (nf::convolve_mod(a, b, M) != nf::convolve_mod_schoolbook(a, b, M)) ntt = false;
        if (nf::convolve_mod(a, a, M) != nf::convolve_mod_schoolbook(a, a, M)) ntt = false;
    }
    Check(ntt, "NTT (2 and 3 primes, Garner) == schoolbook mod 37^3, 67^3, 691^3, 2^32-5");
    const auto B = nf::bernoulli_numbers(36);
    CheckEq(B[12].to_string(), std::string("-691/2730"), "B_12 exact");
    CheckEq(B[2].to_string(), std::string("1/6"), "B_2 exact");
    CheckEq(B[32].valuation(37), 1, "v_37(B_32) = 1 (37 irregular at 32)");
    CheckEq(B[12].valuation(691), 1, "v_691(B_12) = 1");
    bool kum = true;
    for (uint32_t i = 2; i <= 34; i += 2) {
        if ((B[i].valuation(37) >= 1) != nf::ell_divides_bernoulli(37, i)) kum = false;
    }
    Check(kum, "37 | B_i: exact Bernoulli == power-sum criterion, i = 2..34");
}

// ------------------------------------------------------------------ (2), (3)
void ExactEtaAndLayer1At37(const std::string& root) {
    const std::vector<std::pair<uint32_t, std::pair<uint64_t, std::size_t>>> pins = {
        {32, {0xcba3d1938a721f7bull, 127}},  // R18-2 kummer_polys_l37.txt, class c
        {6, {0xef292f5a4e321b01ull, 31}},    // class b
    };
    std::map<uint32_t, std::vector<std::string>> oracle;  // i -> 36 coefficient strings
    std::map<uint32_t, std::map<uint32_t, int64_t>> oracle_f;
    const std::string path = root.empty() ? std::string() : root + "/docs/implementation/slot18/r182/kummer_polys_l37.txt";
    if (!path.empty() && PathExists(path)) {
        std::ifstream in(path);
        std::string line;
        uint32_t cur = 0;
        bool next_is_f = false;
        while (std::getline(in, line)) {
            if (line.rfind("## class ", 0) == 0) {
                cur = static_cast<uint32_t>(std::stoul(line.substr(line.find("i = ") + 4)));
            } else if (line.rfind("# exponents", 0) == 0) {
                next_is_f = true;
            } else if (next_is_f) {
                std::istringstream is(line);
                std::string tok;
                while (is >> tok) {
                    const std::size_t c = tok.find(':');
                    oracle_f[cur][static_cast<uint32_t>(std::stoul(tok.substr(0, c)))] = std::stoll(tok.substr(c + 1));
                }
                next_is_f = false;
            } else if (line.rfind("a_", 0) == 0) {
                oracle[cur].push_back(line.substr(line.find(" = ") + 3));
            }
        }
    }
    nf::CycloZ Z(37, 1);
    for (const auto& p : pins) {
        const uint32_t i = p.first;
        const nf::ExactEta e = nf::exact_eta_layer1(37, i);
        const uint64_t h = FnvCoeffs(e.coeffs);
        std::cout << "NUMBERFIELD_ETA ell=37 i=" << i << " coeffs=" << e.coeffs.size() << " max_digits=" << e.max_digits
                  << " fnv=0x" << std::hex << h << std::dec << std::endl;
        CheckEq(e.coeffs.size(), std::size_t(36), "eta_" + std::to_string(i) + ": 36 coefficients");
        CheckEq(h, p.second.first, "eta_" + std::to_string(i) + ": FNV of the 36 exact coefficients (R18-2)");
        CheckEq(e.max_digits, p.second.second, "eta_" + std::to_string(i) + ": max coefficient digits (R18-2)");
        if (!oracle.empty()) {
            const auto& o = oracle[i];
            bool same = o.size() == 36;
            for (std::size_t j = 0; same && j < 36; ++j) same = (o[j] == e.coeffs[j].to_string());
            Check(same, "eta_" + std::to_string(i) + ": the 36 coefficients == r182/kummer_polys_l37.txt");
            bool fsame = true;
            for (const auto& kv : e.exponents) fsame = fsame && oracle_f[i].count(kv.first) && oracle_f[i][kv.first] == kv.second;
            Check(fsame && oracle_f[i].size() == 36, "eta_" + std::to_string(i) + ": exponents f_c == the oracle file");
        }
        // v_pi(eta^36 - 1): exactly (BigInt binomial transform) and mod 37^3
        auto e36 = Z.pow(e.coeffs, 36);
        e36[0] -= nf::BigInt(1);
        const long vex = Z.vpi_exact(e36);
        const unsigned vmod = nf::layer1_vpi_of(37, e.coeffs, 3);
        const long want = i == 32 ? 38 : 6;
        CheckEq(vex, want, "eta_" + std::to_string(i) + ": v_pi(eta^36 - 1) exact");
        CheckEq(static_cast<long>(vmod), want, "eta_" + std::to_string(i) + ": v_pi(eta^36 - 1) mod 37^3");
    }
    if (oracle.empty()) std::cout << "  [SKIP] r182/kummer_polys_l37.txt not found; FNV pins only" << std::endl;
    {
        const auto f = nf::kummer_exponents(37, 1, 32);
        const int64_t want[] = {16, 7, -3, -4, 1, -4, -11, 12, 10};  // R18-2 section 1: f = (2:16, 3:7, ..., 10:10)
        bool ok = true;
        for (uint32_t c = 2; c <= 10; ++c) ok = ok && f.at(c) == want[c - 2];
        Check(ok, "f_c for eta_32, c = 2..10 (R18-2 section 1)");
    }
    // layer 1, all 17 even eigenspaces
    std::ostringstream vs;
    bool all = true;
    for (uint32_t i = 2; i <= 34; i += 2) {
        const nf::Layer1Result r = nf::kummer_layer1(37, i, 3);
        vs << (i == 2 ? "" : ",") << r.v_pi;
        const unsigned want = i == 32 ? 38 : i;
        all = all && r.v_pi == want && r.unit_ok && r.not_lth_power && r.eigen_ok && r.oracle_agrees &&
              r.unramified == (i == 32) && r.lambda_split == (i == 32);
    }
    std::cout << "NUMBERFIELD_LAYER1 ell=37 v_i(i=2..34)=" << vs.str() << std::endl;
    Check(all, "layer 1 at 37: v_i = i except v_32 = 38; unit, not a 37th power, eigen omega^i, Kummer oracle (R18-2 table)");
}

// ------------------------------------------------------------------ (4)
struct L2Pin {
    uint32_t ell, w;
    uint64_t kappa0;
    uint32_t y0;
    unsigned L1, L2, thr1, thr2, best_unr, best_other;
    int64_t sum_f, sumabs_f;
};

void Layer2() {
    const std::vector<L2Pin> pins = {
        // R18-2 section 2 (layer2_37.out) / R19-3 eqlayer2 (59, 67; = R18-2's layer2_59/67.out thresholds and kappa)
        {37, 32, 284, 13, 1184, 2516, 1400, 2732, 2732, 2696, 5475, 455835},
        {59, 44, 914, 31, 2596, 6018, 3524, 6946, 6946, 6888, 20885, 2978009},
        {67, 58, 3292, 59, 3886, 8308, 4546, 8968, 8968, 8902, 8977, 4962605},
    };
    for (const L2Pin& p : pins) {
        const double t0 = Now();
        nf::Layer2Options opt;
        if (p.ell == 37) opt.brute_ts = {0, 7};  // t = 7 is kappa = 284
        const nf::Layer2Result r = nf::kummer_layer2_equivariant(p.ell, p.w, opt);
        const double secs = Now() - t0;
        std::cout << "NUMBERFIELD_LAYER2 " << r.summary() << " secs=" << secs << std::endl;
        const std::string tag = "layer 2 (" + std::to_string(p.ell) + "," + std::to_string(p.w) + ")";
        CheckEq(r.rows.size(), std::size_t(p.ell), tag + ": ell lifts");
        Check(r.grouping_identity, tag + ": Theta_kappa == E_w (S - kappa ell T) mod ell^2, all lifts");
        CheckEq(r.v_x0, p.w, tag + ": v(x_0 - 1) = w");
        CheckEq(r.L1, p.L1, tag + ": v(x_0^ell - 1)");
        CheckEq(r.L2, p.L2, tag + ": v(x_0^{ell^2} - 1)");
        CheckEq(r.thr1, p.thr1, tag + ": threshold ell");
        CheckEq(r.thr2, p.thr2, tag + ": threshold ell^2");
        CheckEq(r.unramified_kappas.size(), std::size_t(1), tag + ": exactly one unramified lift");
        if (r.unramified_kappas.size() == 1) CheckEq(r.unramified_kappas[0], p.kappa0, tag + ": the unramified kappa");
        bool rows_ok = true, brute_ok = true;
        unsigned brute_n = 0;
        for (const auto& row : r.rows) {
            const bool is0 = row.kappa == p.kappa0;
            rows_ok = rows_ok && row.local_lth_power && row.v_w == p.L2 && row.best_l2th == (is0 ? p.best_unr : p.best_other);
            if (is0) CheckEq(row.u, p.y0, tag + ": u = 1 - kappa_0 = y_0");
            if (row.brute_checked) {
                ++brute_n;
                brute_ok = brute_ok && row.greedy_eq_brute;
            }
        }
        Check(rows_ok, tag + ": every lift a local ell-th power, v_w = L2, best v = " + std::to_string(p.best_unr) +
                           " at kappa_0 and " + std::to_string(p.best_other) + " elsewhere");
        if (p.ell == 37) {
            CheckEq(brute_n, 2u, tag + ": R18-2 full scan run at two lifts");
            Check(brute_ok, tag + ": greedy test == R18-2's full scan over a mod ell");
        }
        CheckEq(r.global.size(), std::size_t(1), tag + ": global tests on the canonical generator");
        if (r.global.size() == 1) {
            Check(r.global[0].not_global_lth_power, tag + ": eta_kappa0 not a global ell-th power (3 split primes)");
            Check(r.global[0].eigen_eps_kappa, tag + ": eigencharacter eps^kappa0 mod ell^2");
            CheckEq(r.global[0].sum_f, p.sum_f, tag + ": sum f_c of the canonical exponents");
            CheckEq(r.global[0].sumabs_f, p.sumabs_f, tag + ": sum |f_c|");
        }
    }
}

// ------------------------------------------------------------------ (5)
void Layer1At691() {
    const double t0 = Now();
    for (uint32_t i : {12u, 200u, 2u}) {
        const nf::Layer1Result r = nf::kummer_layer1(691, i, 2);
        std::cout << "NUMBERFIELD_LAYER1 ell=691 i=" << i << " v_pi=" << r.v_pi << " unramified=" << r.unramified
                  << " lambda_split=" << r.lambda_split << " not_lth_power=" << r.not_lth_power
                  << " kummer_oracle=" << r.oracle_agrees << " products=" << r.products << std::endl;
        const unsigned want = i == 2 ? 2 : 692;  // R18-2 layer1_691.out
        const std::string tag = "691 layer 1, i = " + std::to_string(i);
        CheckEq(r.v_pi, want, tag + ": v_pi(eta^690 - 1)");
        CheckEq(r.unramified, i != 2, tag + ": unramified");
        CheckEq(r.lambda_split, i != 2, tag + ": lambda split");
        Check(r.not_lth_power && r.eigen_ok && r.oracle_agrees, tag + ": not a 691th power, eigen, Kummer oracle");
        CheckEq(r.ell_divides_Bi, i != 2, tag + ": 691 | B_i");
    }
    std::cout << "NUMBERFIELD_LAYER1 ell=691 secs=" << Now() - t0 << std::endl;
}

// ------------------------------------------------------------------ (6)
void NegativeControls() {
    {
        const nf::Layer2Result r = nf::kummer_layer2_equivariant(37, 30);
        std::cout << "NUMBERFIELD_CONTROL regular " << r.summary() << std::endl;
        Check(r.unramified_kappas.empty(), "control: (37, 30) regular -- no lift is a local 37^2-th power");
    }
    nf::CycloZ Z(37, 1);
    const nf::ExactEta e6 = nf::exact_eta_layer1(37, 6);
    {
        // eta_6^37 is a global 37th power: the LOCAL test passes, the global test must not.
        const auto p = Z.pow(e6.coeffs, 37);
        const unsigned v = nf::layer1_vpi_of(37, p, 3);
        Check(v >= 38, "control: eta_6^37 passes the local test (v_pi >= 38), got " + std::to_string(v));
        bool all_one = true;
        for (uint32_t q : nf::split_primes(37, 3)) {
            const uint32_t rt = nf::root_of_order(q, 37);
            all_one = all_one && nf::powmod32(nf::eval_at(p, rt, q), (q - 1) / 37, q) == 1;
        }
        Check(all_one, "control: eta_6^37 is detected as a 37th power at every split prime");
    }
    {
        nf::ExactEta e32 = nf::exact_eta_layer1(37, 32);
        e32.coeffs[0] += nf::BigInt(1);
        Check(FnvCoeffs(e32.coeffs) != 0xcba3d1938a721f7bull, "control: a tampered eta_32 changes the FNV");
        const uint32_t q = nf::split_primes(37, 1)[0];
        const uint32_t rt = nf::root_of_order(q, 37);
        uint64_t nm = 1;
        for (uint32_t t = 1; t < 37; ++t) nm = nm * nf::eval_at(e32.coeffs, nf::powmod32(rt, t, q), q) % q;
        Check(nm != 1 && nm != q - 1, "control: the tampered eta_32 is not a unit (norm mod 149 != +-1)");
    }
}

// ------------------------------------------------------------------ (7)
void Algebra() {
    using nf::BigInt;
    using nf::BigRational;
    {
        nf::NumberFieldQ K({BigRational(-2), BigRational(0), BigRational(1)});  // Q(sqrt 2)
        const auto a = K.alpha();
        const auto x = K.add(K.from_int(1), a);
        Check(K.equal(K.inv(x), K.sub(a, K.from_int(1))), "Q(sqrt 2): (1 + a)^{-1} = a - 1");
    }
    {
        nf::NumberFieldQ K({BigRational(-2), BigRational(0), BigRational(0), BigRational(1)});  // Q(2^{1/3})
        const auto a = K.alpha();
        Check(K.equal(K.pow(a, 3), K.from_int(2)), "Q(2^{1/3}): a^3 = 2");
        auto half_a2 = K.mul(a, a);
        for (auto& c : half_a2) c /= BigRational(2);
        Check(K.equal(K.inv(a), half_a2), "Q(2^{1/3}): a^{-1} = a^2 / 2");
    }
    {
        // Q(zeta_37): x = u_2 u_5^3 u_7^{-2} is a unit, so its inverse is integral and equals
        // u_2^{-1} u_5^{-3} u_7^2 (the product form). (eta_6 itself takes ~1 s by Euclid over Q -- not in a fast gate.)
        nf::NumberFieldQ K(std::vector<BigRational>(37, BigRational(1)));
        const std::map<uint32_t, int64_t> f = {{2, 1}, {5, 3}, {7, -2}};
        std::map<uint32_t, int64_t> fneg;
        for (const auto& kv : f) fneg[kv.first] = -kv.second;
        nf::CycloZ Z(37, 1);
        const auto xe = Z.product_unit(f);
        nf::NumberFieldQ::Elt x(36);
        for (int j = 0; j < 36; ++j) x[j] = BigRational(xe[j]);
        const auto xi = K.inv(x);
        const auto ei = Z.product_unit(fneg);
        bool same = true;
        for (int j = 0; j < 36; ++j) same = same && xi[j].den() == BigInt(1) && xi[j].num() == ei[j];
        Check(same, "Q(zeta_37): (u_2 u_5^3 u_7^-2)^{-1} by extended Euclid over Q == the product form, integral");
    }
    {
        // Newton polygon of Phi_37(x + 1) = sum_j C(37, j + 1) x^j at 37: one segment of slope -1/36 (pi = 1 - zeta).
        std::vector<BigInt> f(37);
        BigInt c(1);
        for (int j = 0; j <= 36; ++j) {
            c = c * BigInt(37 - j) / BigInt(j + 1);  // C(37, j + 1)
            f[j] = c;
        }
        const auto segs = nf::newton_polygon(f, 37);
        Check(segs.size() == 1 && segs[0].length() == 36 && segs[0].slope() == BigRational(BigInt(-1), BigInt(36)),
              "Newton polygon of Phi_37(x+1) at 37: one segment, slope -1/36, length 36");
        // x^4 + 9 x^2 + 27 x + 81 = (x^2 ...) at 3: points (0,4),(1,3),(2,2),(4,0): one segment slope -1
        const auto s2 = nf::newton_polygon({BigInt(81), BigInt(27), BigInt(9), BigInt(0), BigInt(1)}, 3);
        Check(s2.size() == 1 && s2[0].slope() == BigRational(-1), "Newton polygon x^4+9x^2+27x+81 at 3: slope -1");
        const auto s3 = nf::newton_polygon({BigInt(3), BigInt(1), BigInt(1)}, 3);  // x^2 + x + 3: slopes -1, 0
        Check(s3.size() == 2 && s3[0].slope() == BigRational(-1) && s3[1].slope() == BigRational(0),
              "Newton polygon x^2+x+3 at 3: slopes -1, 0");
    }
    {
        // Hensel: x^2 + 1 == (x - 2)(x - 3) mod 5, lifted to 5^10; the root r = -G(0) has r^2 = -1 mod 5^10.
        const auto hp = nf::hensel_lift({BigInt(1), BigInt(0), BigInt(1)}, {3, 1}, {2, 1}, 5, 10);
        const uint64_t N = hp.modulus;
        const uint64_t r = (N - hp.G[0]) % N;
        Check(N == 9765625u && hp.G.size() == 2 && r * r % N == N - 1, "Hensel: sqrt(-1) in Z/5^10");
        // Phi_37 mod 149 has the root 149's 37th roots of unity; lift x - r0 against the cofactor to 149^3.
        std::vector<BigInt> phi(37, BigInt(1));
        const uint32_t q = 149, r0 = nf::root_of_order(q, 37);
        nf::PolyMod g = {q - r0, 1}, qq, rem;
        nf::poly_divrem(nf::poly_from_bigint(phi, q), g, q, qq, rem);
        const auto h3 = nf::hensel_lift(phi, g, qq, q, 3);
        const uint64_t M3 = h3.modulus, root = (M3 - h3.G[0]) % M3;
        uint64_t v = 0;
        for (int j = 36; j >= 0; --j) v = (v * root + 1) % M3;  // Phi_37(root)
        Check(rem.empty() && v == 0, "Hensel: a root of Phi_37 in Z/149^3");
    }
}

// ------------------------------------------------------------------ (8)
void IncludeGuard(const std::string& root) {
    if (root.empty()) {
        std::cout << "  [SKIP] repository root not found; include guard not run" << std::endl;
        return;
    }
    const std::string script = root + "/docs/matrix-engine/tools/check_numberfield_includes.sh";
    Check(PathExists(script), "guard: check_numberfield_includes.sh present");
    if (!PathExists(script)) return;
    Check(::access(script.c_str(), X_OK) == 0, "guard: check_numberfield_includes.sh is executable");
    const std::string cmd = "cd '" + root + "' && bash '" + script + "'";   // via bash: no /usr/bin/env on Termux
    std::cout << "  $ " << cmd << std::endl;
    const int status = std::system(cmd.c_str());
    const bool ran = status != -1 && WIFEXITED(status);
    Check(ran, "guard: check_numberfield_includes.sh ran");
    if (ran) CheckEq(WEXITSTATUS(status), 0, "guard: NumberField is STL-only and no lower library includes it");
}

}  // namespace

uint32_t NumberFieldValidation() {
    g_checks = 0;
    g_fail = 0;
    const double t0 = Now();
    const std::string root = FindRepoRoot();
#if defined(__SIZEOF_INT128__) && !defined(NF_NO_INT128)
    const int int128 = 1;
#else
    const int int128 = 0;
#endif
    std::cout << "NumberFieldValidation (slot-19 WP-NF; [E] sec:lvnf; oracle R18-2 / R19-3) int128_path=" << int128
              << std::endl;
    double t = Now();
    SelfTests();
    std::cout << "  (1) self-tests secs=" << Now() - t << std::endl;
    t = Now();
    ExactEtaAndLayer1At37(root);
    std::cout << "  (2)(3) exact eta, layer 1 at 37 secs=" << Now() - t << std::endl;
    t = Now();
    Layer2();
    std::cout << "  (4) layer 2 secs=" << Now() - t << std::endl;
    t = Now();
    Layer1At691();
    t = Now();
    NegativeControls();
    std::cout << "  (6) controls secs=" << Now() - t << std::endl;
    t = Now();
    Algebra();
    std::cout << "  (7) algebra secs=" << Now() - t << std::endl;
    t = Now();
    NumberFieldCwLayerGate(root, &g_checks, &g_fail);
    std::cout << "  (9) CW layer secs=" << Now() - t << std::endl;
    IncludeGuard(root);
    std::cout << "NUMBERFIELD_GATE checks=" << g_checks << " failing=" << g_fail << " secs=" << Now() - t0 << std::endl;
    return g_fail;
}
