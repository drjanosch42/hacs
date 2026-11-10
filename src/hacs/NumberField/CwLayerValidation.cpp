// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// CwLayerValidation.cpp -- part (9) of NumberFieldValidation: the Coates--Wiles
// layer route of slot-19 WP-NF2 (NumberField/CwLayer.h), the C++ port of R19-7's
// level2.py / levelm.py. Oracle: docs/implementation/slot19/R19-7_report.md and
// its committed outputs r197/out/level2_l{37,59,67,103,691}.json (every expected
// row, hash, zero and alpha below is R19-7's, or [U] prop:u-eistrunc's PROVED rung,
// never this code's own output).
//
// Default arm (fast): primitives (128-bit product == its portable fallback,
// Montgomery, Miller--Rabin, the 62-bit NTT primes, NTT convolution == schoolbook
// for f * 2^a lengths direct and four-step, SHA-256 vectors, base-ell^d limbs ==
// BigInt); the ring (products == schoolbook, pi^n/ell from binomials == by products,
// P6: log(1+z) == (sigma_2 - 1) lambda_2 in every digit); level 2 at 37 (all 35
// classes, 1295 rows), 59, 67, 103 (rows == the JSON, == the exact Bernoulli /
// Kummer-law reference, lambda_2 SHA-256 == R19-7's); level 3 at (37, 32) (sha,
// the zero 37580, alpha 28305, the level-2 control 1369 / 37); the lean parameters
// r = m, W = 4m - 1 give the same rows; negative controls (W below the floor is
// refused, a perturbed unit is caught, the level-(m-1) datum is wrong mod ell^m).
//
// NUMBERFIELD_LAYER_ALL=1 (read-only): + 691 level 2 (3455 rows, five classes),
// 103 level 3 (914250), 37 level 4 (1072544, alpha 281570), each also with the lean
// parameters; about a minute at 8 threads, < 0.3 GB.
// NUMBERFIELD_LAYER_CELLS="ell:m:w1/w2[:r[:W[:bern]]],..." (read-only): any cell,
// e.g. "691:3:12/200:3:11" (the 691 level-3 run: predicted ~32 GB, see
// docs/implementation/slot19/wpnf2/README.md); a PREDICT line precedes each run;
// rows mod ell^2 are compared with the level-2 JSON, the zeros with the rungs.
// Threads: ENGINE_THREADS if set, else min(8, hardware threads); the values never
// depend on it.
#include "CwLayerValidation.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

#include "BigInt.h"
#include "CwLayer.h"
#include "LadicInt.h"
#include "Mont.h"
#include "Ntt64.h"
#include "Sha256.h"

namespace {

uint32_t* g_c = nullptr;
uint32_t* g_f = nullptr;

void Check(bool ok, const std::string& what) {
    ++*g_c;
    if (!ok) {
        ++*g_f;
        std::cout << "  [FAIL] " << what << std::endl;
    }
}

template <typename A, typename B>
void CheckEq(const A& got, const B& want, const std::string& what) {
    ++*g_c;
    if (!(got == want)) {
        ++*g_f;
        std::ostringstream os;
        os << "  [FAIL] " << what << ": got " << got << ", want " << want;
        std::cout << os.str() << std::endl;
    }
}

double Now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

unsigned Threads() {
    if (const char* e = std::getenv("ENGINE_THREADS")) {
        const int t = std::atoi(e);
        if (t > 0) return static_cast<unsigned>(t);
    }
    const unsigned hw = std::thread::hardware_concurrency();
    return std::max(1u, std::min(8u, hw ? hw : 1u));
}

double PeakRssGb() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmHWM:", 0) == 0) {
            std::istringstream is(line.substr(6));
            double kb = 0;
            is >> kb;
            return kb / 1048576.0;
        }
    }
    return -1;
}

uint64_t Upow(uint64_t b, unsigned e) {
    uint64_t r = 1;
    for (unsigned i = 0; i < e; ++i) r *= b;
    return r;
}

// ------------------------------------------------------------------ a minimal JSON reader (R19-7's outputs)
struct JVal {
    enum Kind { Null, Bool, Num, Str, Arr, Obj } kind = Null;
    double num = 0;
    std::string str;
    std::vector<JVal> arr;
    std::vector<std::pair<std::string, JVal>> obj;
    const JVal* get(const std::string& k) const {
        for (const auto& kv : obj)
            if (kv.first == k) return &kv.second;
        return nullptr;
    }
};

struct JParser {
    const std::string& s;
    std::size_t i = 0;
    explicit JParser(const std::string& t) : s(t) {}
    void ws() {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) ++i;
    }
    std::string str() {
        std::string out;
        ++i;  // "
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\') ++i;
            out += s[i++];
        }
        ++i;
        return out;
    }
    JVal val() {
        ws();
        JVal v;
        if (i >= s.size()) throw std::runtime_error("json: eof");
        const char c = s[i];
        if (c == '{') {
            v.kind = JVal::Obj;
            ++i;
            ws();
            if (s[i] == '}') {
                ++i;
                return v;
            }
            for (;;) {
                ws();
                std::string k = str();
                ws();
                ++i;  // :
                v.obj.emplace_back(k, val());
                ws();
                if (s[i] == ',') {
                    ++i;
                    continue;
                }
                ++i;  // }
                break;
            }
        } else if (c == '[') {
            v.kind = JVal::Arr;
            ++i;
            ws();
            if (s[i] == ']') {
                ++i;
                return v;
            }
            for (;;) {
                v.arr.push_back(val());
                ws();
                if (s[i] == ',') {
                    ++i;
                    continue;
                }
                ++i;
                break;
            }
        } else if (c == '"') {
            v.kind = JVal::Str;
            v.str = str();
        } else if (s.compare(i, 4, "null") == 0) {
            i += 4;
        } else if (s.compare(i, 4, "true") == 0) {
            v.kind = JVal::Bool;
            v.num = 1;
            i += 4;
        } else if (s.compare(i, 5, "false") == 0) {
            v.kind = JVal::Bool;
            i += 5;
        } else {
            v.kind = JVal::Num;
            std::size_t j = i;
            while (j < s.size() && (std::isdigit(static_cast<unsigned char>(s[j])) || s[j] == '-' || s[j] == '+' ||
                                    s[j] == '.' || s[j] == 'e' || s[j] == 'E'))
                ++j;
            v.str = s.substr(i, j - i);
            v.num = std::atof(v.str.c_str());
            i = j;
        }
        return v;
    }
};

struct L2Class {
    std::map<uint64_t, uint32_t> psi;  // k -> Psi_k mod ell^2
    int64_t alpha = -1;
    uint32_t leopoldt = 0;
    bool has_leopoldt = false;
};
struct L2Json {
    bool ok = false;
    std::string sha;
    unsigned W = 0, r = 0;
    std::map<uint32_t, L2Class> classes;
    uint64_t rows = 0;
};

L2Json LoadLevel2(const std::string& root, uint32_t ell) {
    L2Json out;
    if (root.empty()) return out;
    std::ifstream f(root + "/docs/implementation/slot19/r197/out/level2_l" + std::to_string(ell) + ".json");
    if (!f) return out;
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string text = ss.str();
    JParser p(text);
    const JVal top = p.val();
    if (const JVal* v = top.get("lambda_sha")) out.sha = v->str;
    if (const JVal* v = top.get("W")) out.W = static_cast<unsigned>(v->num);
    if (const JVal* v = top.get("r")) out.r = static_cast<unsigned>(v->num);
    const JVal* cls = top.get("classes");
    if (!cls) return out;
    for (const auto& kv : cls->obj) {
        L2Class c;
        if (const JVal* ps = kv.second.get("Psi")) {
            for (const auto& e : ps->obj) c.psi[std::stoull(e.first)] = static_cast<uint32_t>(std::stoull(e.second.str));
        }
        if (const JVal* a = kv.second.get("alpha_mod_l2"))
            if (a->kind == JVal::Num) c.alpha = std::stoll(a->str);
        if (const JVal* l = kv.second.get("leopoldt_mod_l2")) {
            if (l->kind == JVal::Num) {
                c.leopoldt = static_cast<uint32_t>(std::stoull(l->str));
                c.has_leopoldt = true;
            }
        }
        out.rows += c.psi.size();
        out.classes[static_cast<uint32_t>(std::stoul(kv.first))] = c;
    }
    out.ok = !out.classes.empty();
    return out;
}

// ------------------------------------------------------------------ the rungs (R19-7 sec. 3-4; [U] prop:u-eistrunc)
// depth-m rung of (ell, w): the unique k mod ell^{m-1}(ell-1) with Psi_k == 0 mod ell^m.
std::vector<uint64_t> Rungs(uint32_t ell, uint32_t w) {
    if (ell == 37 && w == 32) return {284, 37580, 1072544};
    if (ell == 103 && w == 24) return {228, 914250};
    if (ell == 59 && w == 44) return {914};
    if (ell == 67 && w == 58) return {3292};
    if (ell == 691 && w == 12) return {342252, 49928412};
    if (ell == 691 && w == 200) return {342440, 159113510};
    return {};
}
bool IrregularPair(uint32_t ell, uint32_t w) { return !Rungs(ell, w).empty(); }

// ------------------------------------------------------------------ (9a) primitives
void Primitives() {
    std::mt19937_64 rng(20261004);
    bool m128 = true, mont = true, m32 = true;
    for (int i = 0; i < 20000; ++i) {
        const uint64_t a = rng(), b = rng();
        uint64_t h1, l1, h2, l2;
        nf::mul128(a, b, h1, l1);
        nf::mul128_portable(a, b, h2, l2);
        m128 = m128 && h1 == h2 && l1 == l2;
    }
    Check(m128, "mul128 == mul128_portable on 20000 random pairs (the aarch64 / no-__int128 fallback)");
    for (std::size_t j = 0; j < 4; ++j) {
        const uint64_t p = nf::ntt64_prime(j);
        const nf::Mont64 mt(p);
        for (int i = 0; i < 2000; ++i) {
            const uint64_t a = rng() % p, b = rng() % p;
            mont = mont && mt.from(mt.mul(mt.to(a), mt.to(b))) == nf::mulmod_u64(a, b, p);
        }
    }
    Check(mont, "Mont64 product == mulmod_u64 (4 primes x 2000)");
    {
        const nf::Mont32 mt(37u * 37u * 37u * 37u);
        for (int i = 0; i < 2000; ++i) {
            const uint32_t a = rng() % mt.M, b = rng() % mt.M;
            m32 = m32 && mt.from(mt.mul(mt.to(a), mt.to(b))) == static_cast<uint32_t>(uint64_t(a) * b % mt.M);
        }
    }
    Check(m32, "Mont32 product == % (mod 37^4)");
    Check(nf::is_prime_u64(2305843009213693951ull) && nf::is_prime_u64(1000000007ull) && !nf::is_prime_u64(561) &&
              !nf::is_prime_u64(3215031751ull) && !nf::is_prime_u64(2305843009213693953ull) &&
              !nf::is_prime_u64(uint64_t(4611686014132420609ull)),
          "Miller--Rabin: 2^61-1, 10^9+7 prime; 561, 3215031751 (spsp to 2,3,5,7), 2^61+1, (2^31-1)^2 composite");
    bool primes_ok = true;
    for (std::size_t j = 0; j < 8; ++j) {
        const uint64_t p = nf::ntt64_prime(j);
        primes_ok = primes_ok && p < (uint64_t(1) << 62) && (p - 1) % (uint64_t(15) << 32) == 0 && nf::is_prime_u64(p) &&
                    (j == 0 || p < nf::ntt64_prime(j - 1));
    }
    Check(primes_ok, "the NTT primes: p = c 15 2^32 + 1 < 2^62, prime, decreasing");
    // NTT convolution == schoolbook (direct and four-step, f = 1, 3, 5).
    bool conv = true;
    std::string lens;
    for (std::size_t L : {48ul, 80ul, 1024ul, 3072ul, 5120ul, std::size_t(1) << 12}) {
        for (unsigned dl : {15u, 3u}) {
            const uint64_t p = nf::ntt64_prime(1);
            const nf::Ntt64Plan pl(p, L, dl);
            const nf::Mont64& mt = pl.mont();
            const std::size_t na = L / 2 - 1;
            std::vector<uint64_t> a(L, 0), b(L, 0);
            for (std::size_t i = 0; i < na; ++i) {
                a[i] = rng() % p;
                b[i] = rng() % p;
            }
            std::vector<uint64_t> A = a, B = b;
            pl.forward(A.data(), 3);
            pl.forward(B.data(), 3);
            for (std::size_t i = 0; i < L; ++i) A[i] = mt.mul(A[i], B[i]);
            pl.inverse(A.data(), 3);
            const uint64_t sc = mt.to(mt.to(mt.pow(L % p, p - 2)));
            for (int s = 0; s < 64; ++s) {
                const std::size_t k = (s < 8) ? static_cast<std::size_t>(s) : rng() % (2 * na - 1);
                uint64_t acc = 0;
                for (std::size_t i = 0; i <= k; ++i)
                    if (i < na && k - i < na) acc = mt.add(acc, nf::mulmod_u64(a[i], b[k - i], p));
                conv = conv && mt.mul(A[k], sc) == acc;
            }
            if (dl == 3) lens += std::to_string(L) + (pl.four_step() ? "(4-step,f=" : "(f=") + std::to_string(pl.radix()) + ") ";
        }
    }
    Check(conv, "NTT64 convolution == schoolbook at L = " + lens);
    {
        nf::Sha256 h0;
        nf::Sha256 h1;
        h1.update("abc", 3);
        CheckEq(h0.hex(), std::string("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"), "SHA-256('')");
        CheckEq(h1.hex(), std::string("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"), "SHA-256('abc')");
        nf::Sha256 h2;
        const std::string m = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        h2.update(m.data(), m.size());
        CheckEq(h2.hex(), std::string("248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"), "SHA-256 (two blocks)");
    }
    {
        // base-ell^d limbs == BigInt arithmetic mod ell^W
        bool ok = true;
        for (auto lw : std::vector<std::pair<uint32_t, unsigned>>{{37, 17}, {691, 11}, {103, 12}, {67, 9}}) {
            const nf::LadicCtx cx(lw.first, lw.second);
            const nf::BigInt Q = cx.modulus();
            for (int i = 0; i < 200; ++i) {
                const nf::BigInt x = (nf::BigInt::from_u64(rng()) * nf::BigInt::from_u64(rng()) * nf::BigInt::from_u64(rng())).floor_mod(Q);
                const nf::BigInt y = (nf::BigInt::from_u64(rng()) * nf::BigInt::from_u64(rng())).floor_mod(Q);
                const nf::LVal X = cx.from_bigint(x), Y = cx.from_bigint(y);
                nf::LVal Z{};
                cx.mul(X.data(), Y.data(), Z.data());
                ok = ok && cx.to_bigint(Z.data()) == (x * y).floor_mod(Q);
                cx.sub(X.data(), Y.data(), Z.data());
                ok = ok && cx.to_bigint(Z.data()) == (x - y).floor_mod(Q);
                if (x.mod_u32(lw.first) != 0) {
                    const nf::LVal I = cx.inverse(X.data());
                    cx.mul(X.data(), I.data(), Z.data());
                    ok = ok && cx.to_bigint(Z.data()) == nf::BigInt(1);
                }
                cx.div_lpow(X.data(), 3, Z.data());
                ok = ok && cx.to_bigint(Z.data()) == x / nf::BigInt::pow(nf::BigInt(static_cast<int64_t>(lw.first)), 3);
            }
        }
        Check(ok, "LadicCtx (base ell^d limbs) == BigInt: mul, sub, inverse, div by ell^3 (37^17, 691^11, 103^12, 67^9)");
    }
}

// ------------------------------------------------------------------ (9b) the ring
void RingChecks(unsigned T) {
    std::mt19937_64 rng(4102026);
    struct RC {
        uint32_t ell;
        unsigned m, W, dl;
    };
    for (const RC& rc : {RC{37, 2, 9, 15}, RC{5, 3, 20, 3}, RC{7, 2, 40, 3}, RC{3, 4, 12, 3}}) {
        nf::CwRing R(rc.ell, rc.m, rc.W, T, rc.dl);
        const nf::LadicCtx& cx = R.ctx();
        const nf::BigInt Q = cx.modulus();
        nf::CwRing::Elt a = R.zero(), b = R.zero();
        for (uint64_t i = 0; i < R.n(); ++i) {
            const nf::LVal x = cx.from_bigint(nf::BigInt::from_u64(rng()) * nf::BigInt::from_u64(rng()) * nf::BigInt::from_u64(rng()));
            const nf::LVal y = cx.from_bigint(nf::BigInt::from_u64(rng()) * nf::BigInt::from_u64(rng()) * nf::BigInt::from_u64(rng()));
            std::copy(x.begin(), x.begin() + cx.K, a.begin() + i * cx.K);
            std::copy(y.begin(), y.begin() + cx.K, b.begin() + i * cx.K);
        }
        const std::string tag = "ring (Z/" + std::to_string(rc.ell) + "^" + std::to_string(rc.W) + ")[z]/Phi_" +
                                std::to_string(rc.ell) + "^" + std::to_string(rc.m) + " (P=" + std::to_string(R.prime_count()) +
                                ", L=" + std::to_string(R.ntt_length()) + ")";
        nf::CwRing::Elt c = a;
        R.mul_inplace(c, b);
        Check(c == R.schoolbook_mul(a, b), tag + ": NTT product == schoolbook");
        nf::CwRing::Elt d = a;
        R.sqr_inplace(d);
        Check(d == R.schoolbook_mul(a, a), tag + ": NTT square == schoolbook");
        Check(nf::cw_pi_n_over_ell(R, true) == nf::cw_pi_n_over_ell(R, false),
              tag + ": pi^n/ell from the binomials == by ring products");
        // sigma_c is a ring automorphism
        Check(R.sigma(c, 2) == R.schoolbook_mul(R.sigma(a, 2), R.sigma(b, 2)), tag + ": sigma_2 multiplicative");
    }
    {
        // P6: ell^s log(1 + z) == ell^s (sigma_2 - 1) lambda_2 in all W digits (R19-7: "agree mod ell^5, digits 9/9").
        nf::CwRing R(37, 2, 9, T);
        nf::CwLogOptions o;
        o.r = 3;
        const nf::CwLog L = nf::cw_lambda(R, o);
        nf::CwRing::Elt rhs = R.sigma(L.Lg, 2);
        R.sub_inplace(rhs, L.Lg);
        nf::CwRing::Elt lu = nf::cw_log_unit(R, R.from_i64({1, 1}), o.r);
        const nf::LadicCtx& cx = R.ctx();
        for (uint64_t i = 0; i < R.n(); ++i) cx.mul_lpow(lu.data() + i * cx.K, L.s - o.r, lu.data() + i * cx.K);
        Check(lu == rhs, "P6 at 37: ell^s log(1+z) == ell^s (sigma_2 - 1) lambda_2 in all 9 digits (the unit route == the Iwasawa log)");
    }
}

// ------------------------------------------------------------------ (9c) cells
struct GateCell {
    uint32_t ell;
    unsigned m, r, W;
    std::vector<uint32_t> classes;  // empty: from the JSON
    std::string sha;                // expected lambda sha16 ("" = none)
    unsigned bern;                  // exact Bernoulli reference up to this k
    bool lean;                      // the lean parameters (r = m, W = 4m - 1): rows must equal `ref_rows`
};

struct CellOut {
    std::map<uint32_t, std::vector<uint32_t>> rows;
    nf::CwCellResult res;
};

CellOut RunGateCell(const GateCell& gc, const L2Json* js, const std::vector<nf::BigRational>& Bk, unsigned T,
                    const std::map<uint32_t, std::vector<uint32_t>>* ref_rows, uint64_t* json_rows_ok) {
    nf::CwCellSpec sp;
    sp.ell = gc.ell;
    sp.m = gc.m;
    sp.r = gc.r;
    sp.W = gc.W;
    sp.classes = gc.classes;
    if (sp.classes.empty() && js)
        for (const auto& kv : js->classes) sp.classes.push_back(kv.first);
    sp.hash = !gc.sha.empty();
    sp.bern_kmax = gc.bern;
    sp.bern_table = &Bk;
    const double t0 = Now();
    CellOut out;
    out.res = nf::cw_run_cell(sp, T);
    const nf::CwCellResult& R = out.res;
    const std::string tag = "CW layer (" + std::to_string(gc.ell) + ", m=" + std::to_string(gc.m) + ", r=" +
                            std::to_string(R.r) + ", W=" + std::to_string(R.W) + ")";
    std::cout << "NUMBERFIELD_CWLAYER ell=" << gc.ell << " m=" << gc.m << " r=" << R.r << " W=" << R.W << " n=" << R.n
              << " L=" << R.L << " P=" << R.P << " K=" << R.K << " sha=" << (R.sha16.empty() ? "-" : R.sha16)
              << " cy=" << R.cy << " terms=" << R.terms << " squarings=" << R.stats.squarings
              << " products=" << R.stats.products << " ntts=" << R.stats.ntts << " classes=" << R.classes.size()
              << " rows=" << R.rows << " threads=" << T << " secs=" << Now() - t0 << std::endl;
    if (!gc.sha.empty()) CheckEq(R.sha16, gc.sha, tag + ": lambda_m digit-array SHA-256 == R19-7's");
    const uint64_t mod = Upow(gc.ell, gc.m);
    const uint64_t mod2 = Upow(gc.ell, 2);
    uint64_t jtot = 0, jok = 0;
    for (const nf::CwClassResult& c : R.classes) {
        out.rows[c.w] = c.psi;
        const std::string ct = tag + " w=" + std::to_string(c.w);
        const bool even = c.w % 2 == 0;
        Check(c.integral, ct + ": nu integral after e_w");
        Check(c.delta_law, ct + ": m-th differences of Psi in t vanish mod ell^m");
        // the JSON (level 2: equal rows; level >= 3: equal mod ell^2 at the same k mod ell(ell-1))
        uint64_t cj = 0, cjok = 0;
        if (js) {
            auto it = js->classes.find(c.w);
            if (it != js->classes.end()) {
                const uint64_t per = static_cast<uint64_t>(gc.ell) * (gc.ell - 1);
                for (uint64_t t = 0; t < c.psi.size(); ++t) {
                    const uint64_t k = nf::cw_row_k(gc.ell, gc.m, c.w, t);
                    uint64_t k2 = (k - c.w) % per + c.w;
                    if (k2 < 2) k2 += static_cast<uint64_t>(gc.ell) * (gc.ell - 1);
                    auto jt = it->second.psi.find(k2);
                    if (jt == it->second.psi.end()) continue;
                    ++cj;
                    cjok += (c.psi[t] % mod2 == jt->second);
                }
                if (gc.m == 2) {
                    CheckEq(cj, static_cast<uint64_t>(c.psi.size()), ct + ": every row has a JSON row");
                    if (it->second.has_leopoldt) CheckEq(c.leopoldt, it->second.leopoldt, ct + ": L_ell(1, omega^w) mod ell^2 == JSON");
                }
                CheckEq(cjok, cj, ct + (gc.m == 2 ? ": Psi_k == R19-7's row" : ": Psi_k mod ell^2 == R19-7's level-2 row"));
            }
        }
        jtot += cj;
        jok += cjok;
        if (c.bern_run) {
            CheckEq(c.bern_rows_ok, static_cast<uint64_t>(c.psi.size()),
                    ct + ": Psi_k == (1 - ell^(k-1)) B_k / k (exact rows " + std::to_string(c.bern_exact_rows) + ", Kummer law)");
            CheckEq(c.bern_law_held, c.bern_law_checked, ct + ": Kummer law held on every further exact weight");
        }
        const std::vector<uint64_t> rungs = Rungs(gc.ell, c.w);
        if (!even) {
            CheckEq(c.zeros_count[gc.m], static_cast<uint64_t>(c.psi.size()), ct + ": odd class, Psi == 0");
        } else if (IrregularPair(gc.ell, c.w) && rungs.size() >= gc.m - 1) {
            const uint64_t want = rungs[gc.m - 2];
            CheckEq(c.zeros_count[gc.m], uint64_t(1), ct + ": a unique zero mod ell^m");
            CheckEq(c.zeros_top.empty() ? uint64_t(0) : c.zeros_top[0], want, ct + ": the depth-m rung");
            const int64_t a = static_cast<int64_t>((nf::powmod_u64(1 + gc.ell, want, mod) + mod - 1) % mod);
            CheckEq(c.alpha, a, ct + ": alpha mod ell^m = x_kappa");
        } else if (!IrregularPair(gc.ell, c.w)) {
            CheckEq(c.zeros_count[1], uint64_t(0), ct + ": regular class, no zero mod ell");
        }
        if (c.control_run && even) {
            CheckEq(c.control_right_low, static_cast<uint64_t>(c.psi.size()), ct + ": level-(m-1) control right mod ell^(m-1)");
            CheckEq(c.control_right_high, static_cast<uint64_t>(c.psi.size() / gc.ell), ct + ": level-(m-1) control right mod ell^m in 1/ell of the rows");
        }
        if (ref_rows) {
            auto it = ref_rows->find(c.w);
            if (it != ref_rows->end()) Check(it->second == c.psi, ct + ": lean parameters give the same rows as R19-7's");
        }
        std::cout << "NUMBERFIELD_CWLAYER_CLASS ell=" << gc.ell << " m=" << gc.m << " w=" << c.w << " integral=" << c.integral
                  << " json_rows=" << cjok << "/" << cj << " bern_rows=" << c.bern_rows_ok << "/" << c.psi.size()
                  << " zeros=[";
        for (unsigned j = 1; j <= gc.m; ++j) std::cout << c.zeros_count[j] << (j < gc.m ? "," : "");
        std::cout << "] zero_k=" << (c.zeros_count[gc.m] == 1 ? std::to_string(c.zeros_top[0]) : std::string("-"))
                  << " alpha=" << c.alpha << " leopoldt=" << c.leopoldt << " control=" << c.control_right_low << "/"
                  << c.control_right_high << std::endl;
    }
    if (json_rows_ok) *json_rows_ok += jok;
    if (js && gc.m == 2 && !gc.lean && gc.classes.empty()) CheckEq(jok, js->rows, tag + ": all of R19-7's rows reproduced");
    return out;
}

void GateCellsDefault(const std::string& root, unsigned T) {
    const double t0 = Now();
    const std::vector<nf::BigRational> Bk = nf::bernoulli_over_k(1400);
    {
        // the tangent-number Bernoulli == Akiyama--Tanigawa (BigRational.h) for k <= 60
        const auto B = nf::bernoulli_numbers(60);
        bool ok = true;
        for (unsigned k = 2; k <= 60; k += 2) ok = ok && Bk[k] == B[k] / nf::BigRational(static_cast<int64_t>(k));
        Check(ok, "B_k / k from tangent numbers == Akiyama--Tanigawa, k <= 60");
    }
    std::cout << "  (9c) Bernoulli table k <= 1400 secs=" << Now() - t0 << std::endl;
    uint64_t rows_ok = 0, rows_total = 0;
    for (uint32_t ell : {37u, 59u, 67u, 103u}) {
        const L2Json js = LoadLevel2(root, ell);
        Check(js.ok, "R19-7's level2_l" + std::to_string(ell) + ".json present and parsed");
        if (!js.ok) continue;
        rows_total += js.rows;
        RunGateCell(GateCell{ell, 2, js.r, js.W, {}, js.sha, 1400u, false}, &js, Bk, T, nullptr, &rows_ok);
    }
    CheckEq(rows_total, uint64_t(1295 + 236 + 268 + 515), "level-2 JSON rows at 37 / 59 / 67 / 103");
    std::cout << "NUMBERFIELD_CWLAYER_ROWS level=2 ell=37,59,67,103 rows_ok=" << rows_ok << "/" << rows_total << std::endl;
    const L2Json j37 = LoadLevel2(root, 37);
    const CellOut c3 = RunGateCell(GateCell{37, 3, 4, 12, {32}, "bfe0068dbec3fba7", 1400, false}, &j37, Bk, T, nullptr, nullptr);
    RunGateCell(GateCell{37, 3, 3, 11, {32}, "", 1400, true}, &j37, Bk, T, &c3.rows, nullptr);
}

void NegativeControls(const std::string& root, unsigned T) {
    {
        nf::CwCellSpec sp;
        sp.ell = 37;
        sp.m = 2;
        sp.r = 3;
        sp.W = 7;  // floor r + 3m - 1 = 8
        sp.classes = {32};
        bool refused = false;
        try {
            nf::cw_run_cell(sp, T);
        } catch (const std::logic_error&) {
            refused = true;
        }
        Check(refused, "control: W = 7 < r + 3m - 1 = 8 at (37, m=2, r=3) is refused (nu would be known mod 37 only)");
        sp.W = 8;
        const nf::CwCellResult ok = nf::cw_run_cell(sp, T);
        CheckEq(ok.classes.at(0).zeros_top.empty() ? uint64_t(0) : ok.classes.at(0).zeros_top[0], uint64_t(284),
                "control: W = 8 (exactly the floor) is accepted and still finds 284");
    }
    const L2Json js = LoadLevel2(root, 37);
    if (!js.ok) return;
    for (int pert : {1, 2}) {
        nf::CwCellSpec sp;
        sp.ell = 37;
        sp.m = 2;
        sp.r = 3;
        sp.W = 9;
        sp.classes = {32, 2};
        sp.hash = true;
        sp.log.perturb = pert;
        const nf::CwCellResult R = nf::cw_run_cell(sp, T);
        uint64_t bad = 0;
        for (const auto& c : R.classes) {
            const auto& jc = js.classes.at(c.w);
            for (uint64_t t = 0; t < c.psi.size(); ++t)
                if (jc.psi.at(nf::cw_row_k(37, 2, c.w, t)) != c.psi[t]) ++bad;
        }
        std::cout << "NUMBERFIELD_CWLAYER_CONTROL perturb=" << (pert == 1 ? "unit_times_u2" : "one_digit")
                  << " sha=" << R.sha16 << " rows_differing=" << bad << "/74" << std::endl;
        Check(bad > 0 && R.sha16 != "796a44bad240a894",
              std::string("control: a perturbed unit (") + (pert == 1 ? "pi^n/ell times u_2" : "one digit of pi^n/ell") +
                  ") is caught (hash and rows differ from R19-7's)");
    }
}

void AllArm(const std::string& root, unsigned T) {
    std::cout << "EFFECTIVE NumberFieldValidation arm=LAYER_ALL cells=6" << std::endl;
    const std::vector<nf::BigRational> Bk = nf::bernoulli_over_k(3000);
    const L2Json j691 = LoadLevel2(root, 691), j103 = LoadLevel2(root, 103), j37 = LoadLevel2(root, 37);
    Check(j691.ok, "R19-7's level2_l691.json present and parsed");
    uint64_t ok691 = 0;
    const CellOut a = RunGateCell(GateCell{691, 2, 3, 9, {}, "1cc99c56e175f32f", 2800, false}, j691.ok ? &j691 : nullptr, Bk, T,
                                  nullptr, &ok691);
    std::cout << "NUMBERFIELD_CWLAYER_ROWS level=2 ell=691 rows_ok=" << ok691 << "/" << j691.rows << " rss_hwm_gb=" << PeakRssGb() << std::endl;
    RunGateCell(GateCell{691, 2, 2, 7, {}, "", 2800, true}, j691.ok ? &j691 : nullptr, Bk, T, &a.rows, nullptr);
    const CellOut b = RunGateCell(GateCell{103, 3, 4, 12, {24}, "71ca74ef9863f524", 3000, false}, &j103, Bk, T, nullptr, nullptr);
    RunGateCell(GateCell{103, 3, 3, 11, {24}, "", 3000, true}, &j103, Bk, T, &b.rows, nullptr);
    const CellOut c = RunGateCell(GateCell{37, 4, 5, 17, {32}, "91277ba0d2e2defe", 3000, false}, &j37, Bk, T, nullptr, nullptr);
    RunGateCell(GateCell{37, 4, 4, 15, {32}, "", 3000, true}, &j37, Bk, T, &c.rows, nullptr);
    std::cout << "NUMBERFIELD_CWLAYER_ALL rss_hwm_gb=" << PeakRssGb() << std::endl;
}

void CellsArm(const std::string& root, const std::string& spec, unsigned T) {
    std::vector<GateCell> cells;
    std::stringstream ss(spec);
    std::string item;
    unsigned pos = 0;
    while (std::getline(ss, item, ',')) {
        ++pos;
        std::vector<std::string> f;
        std::stringstream is(item);
        std::string x;
        while (std::getline(is, x, ':')) f.push_back(x);
        GateCell gc{0, 0, 0, 0, {}, "", 0, false};
        bool good = f.size() >= 3 && f.size() <= 6;
        try {
            if (good) {
                gc.ell = static_cast<uint32_t>(std::stoul(f[0]));
                gc.m = static_cast<unsigned>(std::stoul(f[1]));
                std::stringstream ws(f[2]);
                std::string w;
                while (std::getline(ws, w, '/')) gc.classes.push_back(static_cast<uint32_t>(std::stoul(w)));
                if (f.size() > 3) gc.r = static_cast<unsigned>(std::stoul(f[3]));
                if (f.size() > 4) gc.W = static_cast<unsigned>(std::stoul(f[4]));
                if (f.size() > 5) gc.bern = static_cast<unsigned>(std::stoul(f[5]));
                good = gc.ell >= 3 && nf::is_prime_u64(gc.ell) && gc.m >= 2 && gc.m < gc.ell && !gc.classes.empty();
                for (uint32_t w2 : gc.classes) good = good && w2 >= 1 && w2 < gc.ell - 1;
            }
        } catch (...) {
            good = false;
        }
        if (!good) {
            std::cout << "  [REFUSED] NUMBERFIELD_LAYER_CELLS entry " << pos << " '" << item
                      << "' (want ell:m:w1/w2[:r[:W[:bern]]], ell an odd prime, 2 <= m < ell, 1 <= w <= ell-2)" << std::endl;
            Check(false, "NUMBERFIELD_LAYER_CELLS entry " + std::to_string(pos) + " well-formed");
            continue;
        }
        if (gc.bern == 0) {
            uint32_t wmax = *std::max_element(gc.classes.begin(), gc.classes.end());
            gc.bern = std::min<unsigned>(4000, wmax + (gc.ell - 1) * (gc.m + 2));
        }
        cells.push_back(gc);
    }
    std::cout << "EFFECTIVE NumberFieldValidation arm=LAYER_CELLS cells=" << cells.size() << std::endl;
    for (const GateCell& gc : cells) {
        const nf::CwCostPrediction pr = nf::cw_predict(gc.ell, gc.m, gc.r, gc.W, static_cast<unsigned>(gc.classes.size()), T);
        std::cout << "NUMBERFIELD_CWLAYER_PREDICT ell=" << gc.ell << " m=" << gc.m << " r=" << pr.r << " W=" << pr.W
                  << " n=" << pr.n << " L=" << pr.L << " P=" << pr.P << " K=" << pr.K << " squarings=" << pr.squarings
                  << " products=" << pr.products << " ntts=" << pr.ntts << " threads=" << T << " pred_secs=" << pr.secs
                  << " pred_peak_gb=" << pr.peak_bytes / 1073741824.0 << std::endl;
        const std::vector<nf::BigRational> Bk = nf::bernoulli_over_k(gc.bern);
        const L2Json js = LoadLevel2(root, gc.ell);
        GateCell g2 = gc;
        g2.sha = "";
        const double t0 = Now();
        RunGateCell(g2, js.ok ? &js : nullptr, Bk, T, nullptr, nullptr);
        std::cout << "NUMBERFIELD_CWLAYER_CELL_DONE ell=" << gc.ell << " m=" << gc.m << " secs=" << Now() - t0
                  << " rss_hwm_gb=" << PeakRssGb() << std::endl;
    }
}

}  // namespace

void NumberFieldCwLayerGate(const std::string& root, uint32_t* checks, uint32_t* fails) {
    g_c = checks;
    g_f = fails;
    const unsigned T = Threads();
    double t = Now();
    Primitives();
    std::cout << "  (9a) CW layer primitives secs=" << Now() - t << std::endl;
    t = Now();
    RingChecks(T);
    std::cout << "  (9b) CW layer ring secs=" << Now() - t << std::endl;
    t = Now();
    GateCellsDefault(root, T);
    std::cout << "  (9c) CW layer gate cells secs=" << Now() - t << std::endl;
    t = Now();
    NegativeControls(root, T);
    std::cout << "  (9d) CW layer controls secs=" << Now() - t << std::endl;
    const char* all = std::getenv("NUMBERFIELD_LAYER_ALL");
    if (all && std::string(all) == "1") {
        t = Now();
        AllArm(root, T);
        std::cout << "  (9e) CW layer LAYER_ALL secs=" << Now() - t << std::endl;
    }
    const char* cells = std::getenv("NUMBERFIELD_LAYER_CELLS");
    if (cells && *cells) {
        t = Now();
        CellsArm(root, cells, T);
        std::cout << "  (9f) CW layer LAYER_CELLS secs=" << Now() - t << std::endl;
    }
}
