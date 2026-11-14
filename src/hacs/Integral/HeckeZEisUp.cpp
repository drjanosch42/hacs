// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HeckeZEisUp -- slot-26 HZ-EISUP: the Eisenstein U_p systems of the full algebra at N > 1 (see the header and HZEISUP_design.md v0.2).

#include "HeckeZEisUp.h"

#include "HeckeZCharpoly.h"
#include "IntegralCharpolyCrt.h"
#include "../NumberField/BigInt.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/Fnv.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <functional>
#include <map>
#include <set>
#include <sstream>
#include <thread>

namespace heckez {
namespace eisup {

namespace {

using u64 = uint64_t;
using nf::BigInt;
using ZPoly = std::vector<BigInt>;   // low to high

// ---- exact integers -------------------------------------------------------------------------------------------------------------
BigInt FromHex(const std::string& s) {
    size_t i = 0;
    const bool neg = !s.empty() && s[0] == '-';
    if (neg) i = 1;
    ENGINE_CHECK_MSG(s.size() >= i + 3 && s[i] == '0' && s[i + 1] == 'x', "HeckeZEisUp: not a canonical hex integer");
    BigInt v(0);
    const BigInt sixteen(16);
    for (size_t j = i + 2; j < s.size(); ++j) {
        const char c = s[j];
        int d = -1;
        if (c >= '0' && c <= '9') d = c - '0';
        else if (c >= 'a' && c <= 'f') d = c - 'a' + 10;
        ENGINE_CHECK_MSG(d >= 0, "HeckeZEisUp: not a canonical hex integer");
        v = v * sixteen + BigInt(d);
    }
    return neg ? -v : v;
}
std::string ToHex(const BigInt& v) {
    if (v.is_zero()) return "0x0";
    const BigInt::Mag& m = v.magnitude();
    std::ostringstream o;
    if (v.is_neg()) o << "-";
    o << "0x" << std::hex << m.back();
    for (size_t i = m.size() - 1; i-- > 0;) {
        o.width(8);
        o.fill('0');
        o << m[i];
    }
    return o.str();
}
std::string ToDec(const BigInt& v) { return v.to_string(); }
ZPoly PolyFromHex(const std::vector<std::string>& h) {
    ZPoly p;
    for (const auto& s : h) p.push_back(FromHex(s));
    return p;
}
std::vector<std::string> PolyHex(const ZPoly& p) {
    std::vector<std::string> o;
    for (const auto& c : p) o.push_back(ToHex(c));
    return o;
}
// p(x) / (x - r) with remainder (synthetic division); p of degree >= 1
ZPoly DivLinear(const ZPoly& p, const BigInt& r, BigInt* rem) {
    const size_t dg = p.size() - 1;
    ZPoly q(dg);
    BigInt acc = p[dg];
    for (size_t i = dg; i-- > 0;) {
        q[i] = acc;
        acc = p[i] + acc * r;
    }
    *rem = acc;
    return q;
}
BigInt EvalZ(const ZPoly& p, const BigInt& x) {
    BigInt acc(0);
    for (size_t i = p.size(); i-- > 0;) acc = acc * x + p[i];
    return acc;
}
ZPoly ProdLinear(const std::vector<BigInt>& roots) {
    ZPoly p = {BigInt(1)};
    for (const BigInt& r : roots) {
        ZPoly q(p.size() + 1, BigInt(0));
        for (size_t i = 0; i < p.size(); ++i) {
            q[i + 1] += p[i];
            q[i] -= p[i] * r;
        }
        p = std::move(q);
    }
    return p;
}
uint64_t BitLen(const BigInt& v) { return v.is_zero() ? 0 : static_cast<uint64_t>(v.bit_length()); }
u64 ModOf(const BigInt& v, u64 q) {
    const BigInt r = v.floor_mod(BigInt::from_u64(q));
    return r.is_zero() ? 0 : static_cast<u64>(r.to_int64());
}

// ---- linear algebra mod q (q < 2^31: every product < 2^62) ------------------------------------------------------------------------
u64 PowMod(u64 a, u64 e, u64 q) {
    u64 r = 1 % q;
    for (a %= q; e; e >>= 1, a = a * a % q)
        if (e & 1) r = r * a % q;
    return r;
}
u64 InvMod(u64 a, u64 q) { return PowMod(a, q - 2, q); }
// the right kernel of the b x b matrix A (row-major) mod q: basis vectors v_t (one per free column f_t, v_t[f_t] = 1, 0 at the other free
// columns); *free receives the free columns
std::vector<std::vector<u64>> Kernel(std::vector<u64> A, size_t rows, size_t cols, u64 q, std::vector<size_t>* free) {
    std::vector<size_t> pivc;
    size_t r = 0;
    for (size_t c = 0; c < cols && r < rows; ++c) {
        size_t piv = r;
        while (piv < rows && A[piv * cols + c] == 0) ++piv;
        if (piv == rows) continue;
        if (piv != r)
            for (size_t j = 0; j < cols; ++j) std::swap(A[piv * cols + j], A[r * cols + j]);
        const u64 inv = InvMod(A[r * cols + c], q);
        for (size_t j = 0; j < cols; ++j) A[r * cols + j] = A[r * cols + j] * inv % q;
        for (size_t i = 0; i < rows; ++i) {
            if (i == r) continue;
            const u64 f = A[i * cols + c];
            if (!f) continue;
            for (size_t j = 0; j < cols; ++j) A[i * cols + j] = (A[i * cols + j] + (q - f) * A[r * cols + j]) % q;
        }
        pivc.push_back(c);
        ++r;
    }
    std::vector<bool> isp(cols, false);
    for (size_t c : pivc) isp[c] = true;
    free->clear();
    for (size_t c = 0; c < cols; ++c)
        if (!isp[c]) free->push_back(c);
    std::vector<std::vector<u64>> basis;
    for (size_t f : *free) {
        std::vector<u64> v(cols, 0);
        v[f] = 1;
        for (size_t i = 0; i < pivc.size(); ++i) v[pivc[i]] = (q - A[i * cols + f]) % q;
        basis.push_back(std::move(v));
    }
    return basis;
}
std::vector<u64> MatVec(const std::vector<u64>& X, size_t b, const std::vector<u64>& v, u64 q) {
    std::vector<u64> w(b, 0);
    for (size_t i = 0; i < b; ++i) {
        u64 s = 0;
        for (size_t j = 0; j < b; ++j)
            if (v[j]) s = (s + X[i * b + j] * v[j]) % q;
        w[i] = s;
    }
    return w;
}

// the per-prime read: the restrictions of every operator to K_q (e x e, row-major, column convention: column t = X v_t in the basis)
struct PrimeRead {
    u64 q = 0;
    std::vector<std::vector<u64>> R;   // per operator
    std::vector<u64> theta;            // the restriction of theta
};

// the run at one word prime: K_q and the restrictions (nullopt-like: ok = false when dim K_q != e)
bool ReadAtPrime(const PrimeRun& run, const std::vector<uint32_t>& ops, uint32_t k, size_t e, const std::vector<uint32_t>& theta,
                 PrimeRead* out) {
    const u64 q = run.ell;
    const size_t b = run.h1;
    const u64 c0 = (1 + PowMod(ops[0], k - 1, q)) % q;
    std::vector<u64> A = run.mats[0];
    for (size_t i = 0; i < b; ++i) A[i * b + i] = (A[i * b + i] + q - c0) % q;
    std::vector<size_t> fr;
    const auto basis = Kernel(A, b, b, q, &fr);
    if (basis.size() != e) return false;
    out->q = q;
    out->R.assign(ops.size(), std::vector<u64>(e * e, 0));
    for (size_t o = 0; o < ops.size(); ++o) {
        for (size_t t = 0; t < e; ++t) {
            const std::vector<u64> w = MatVec(run.mats[o], b, basis[t], q);
            // the coordinates of w at the free columns; w must lie in K_q (E_Z / q is T-stable, prop:heckeZeisup (2))
            std::vector<u64> back(b, 0);
            for (size_t i = 0; i < e; ++i) {
                const u64 c = w[fr[i]];
                out->R[o][i * e + t] = c;
                if (c)
                    for (size_t x = 0; x < b; ++x) back[x] = (back[x] + c * basis[i][x]) % q;
            }
            ENGINE_CHECK_MSG(back == w, "HeckeZEisUp: K_q is not stable under an operator of the list (a defect: T is commutative)");
        }
    }
    out->theta.assign(e * e, 0);
    for (size_t o = 0; o < ops.size(); ++o) {
        if (!theta[o]) continue;
        const u64 c = theta[o] % q;
        for (size_t x = 0; x < e * e; ++x) out->theta[x] = (out->theta[x] + c * out->R[o][x]) % q;
    }
    return true;
}

// runs at primes[from, to) in parallel, index-ordered (RunAtPrime: the full step's own operator route)
void RunPrimes(uint32_t N, uint32_t n, const std::vector<uint32_t>& ops, const std::vector<uint32_t>& primes, size_t from, size_t to,
               uint32_t threads, std::vector<PrimeRun>& out) {
    out.resize(std::max(out.size(), to));
    std::atomic<size_t> next(from);
    const uint32_t T = std::max<uint32_t>(1, std::min<uint32_t>(threads, static_cast<uint32_t>(to - from)));
    std::vector<std::exception_ptr> errs(T);
    auto work = [&](uint32_t w) {
        try {
            for (;;) {
                const size_t i = next.fetch_add(1);
                if (i >= to) break;
                out[i] = RunAtPrime(N, n, ops, primes[i]);
            }
        } catch (...) {
            errs[w] = std::current_exception();
        }
    };
    std::vector<std::thread> pool;
    for (uint32_t w = 1; w < T; ++w) pool.emplace_back(work, w);
    work(0);
    for (auto& t : pool) t.join();
    for (auto& ex : errs)
        if (ex) std::rethrow_exception(ex);
}

bool IsSquarefree(uint32_t N) {
    for (uint32_t p = 2; static_cast<uint64_t>(p) * p <= N; ++p)
        if (N % (p * p) == 0) return false;
    return true;
}
std::vector<uint32_t> PrimesOf(uint32_t N) {
    std::vector<uint32_t> o;
    uint32_t x = N;
    for (uint32_t p = 2; static_cast<uint64_t>(p) * p <= x; ++p)
        if (x % p == 0) {
            o.push_back(p);
            while (x % p == 0) x /= p;
        }
    if (x > 1) o.push_back(x);
    return o;
}
bool TupleLess(const std::vector<BigInt>& a, const std::vector<BigInt>& b) {
    for (size_t i = 0; i < a.size() && i < b.size(); ++i)
        if (a[i] != b[i]) return a[i] < b[i];
    return a.size() < b.size();
}

}  // namespace

bool InScope(uint32_t N, uint32_t n) { return N > 1 && n % 2 == 0 && EisAnemicCount(n + 2, N) == 1; }

std::vector<uint32_t> EisUpResult::Uprimes() const { return std::vector<uint32_t>(ops.end() - nU, ops.end()); }

std::string EisUpResult::text() const {
    std::ostringstream o;
    o << "EISUP N=" << N << " n=" << n << " ops=";
    for (size_t i = 0; i < ops.size(); ++i) o << (i ? "," : "") << (i + nU >= ops.size() ? "U_" : "T_") << ops[i];
    o << " theta=(";
    for (size_t i = 0; i < theta.size(); ++i) o << (i ? " " : "") << theta[i];
    o << ") e_k=" << e_k << " d=" << d << " q0=" << q0 << " primes=" << primes_used << " skipped=[";
    for (size_t i = 0; i < skipped.size(); ++i) o << (i ? "," : "") << skipped[i];
    o << "] q*=" << id_prime << "\n";
    auto poly = [&](const std::vector<std::string>& p) {
        o << "[";
        for (size_t i = 0; i < p.size(); ++i) o << (i ? "," : "") << p[i];
        o << "]";
    };
    o << "e_theta=";
    poly(e_theta);
    o << "\n";
    const std::vector<uint32_t> U = Uprimes();
    for (size_t i = 0; i < g.size(); ++i) {
        o << "g_U" << U[i] << "=";
        poly(g[i]);
        o << "\n";
    }
    for (size_t j = 0; j < lines.size(); ++j) {
        o << "line " << j << ":";
        for (size_t i = 0; i < lines[j].U.size(); ++i) o << " U_" << U[i] << "=" << lines[j].U[i];
        o << " lambda_theta=" << lines[j].lambda_theta << "\n";
    }
    o << "c_theta=";
    poly(c_theta);
    o << "\n";
    return o.str();
}

uint64_t EisUpResult::digest() const { return engine::FnvBytes(engine::kFnvOffset64, text()); }

EisUpResult EisUpStep(const EisUpInput& in) {
    const auto t0 = std::chrono::steady_clock::now();
    ENGINE_CHECK_MSG(InScope(in.N, in.n), "HeckeZEisUp: outside the scope (N > 1, e^an_k(N) = 1)");
    ENGINE_CHECK_MSG(in.nU >= 1 && in.nU < in.ops.size() && in.theta.size() == in.ops.size(), "HeckeZEisUp: the operator list");
    ENGINE_CHECK_MSG(in.e_k >= 1 && in.d >= in.e_k && in.f.size() == static_cast<size_t>(in.d) + 1, "HeckeZEisUp: e_k, d, f_theta");
    ENGINE_CHECK_MSG(in.N % in.ops[0] != 0, "HeckeZEisUp: q0 = the first operator must be a T_p, p !| N");
    const EisUpTamper tz = in.tamper ? *in.tamper : EisUpTamper{};
    const uint32_t N = in.N, n = in.n, k = n + 2;
    const size_t m = in.ops.size(), nT = m - in.nU;
    const size_t e = static_cast<size_t>(in.e_k + tz.ek_delta);
    ENGINE_CHECK_MSG(e >= 1, "HeckeZEisUp: e_k >= 1");
    EisUpResult R;
    R.N = N;
    R.n = n;
    R.nU = in.nU;
    R.ops = in.ops;
    R.theta = in.theta;
    R.e_k = in.e_k;
    R.d = in.d;
    R.q0 = in.ops[0];
    const std::vector<uint32_t> U = R.Uprimes();
    ENGINE_CHECK_MSG(U == PrimesOf(N), "HeckeZEisUp: the U_p must be the primes dividing N, ascending");

    // ---- the proved bounds (prop:heckeZeisup (1), (3)) ----
    std::vector<BigInt> eisT(m);   // 1 + p^{k-1} for a T_p
    BigInt Rb(0);
    for (size_t o = 0; o < m; ++o) {
        const BigInt pk = BigInt::pow(BigInt::from_u64(in.ops[o]), k - 1);
        if (o < nT) {
            eisT[o] = pk + BigInt(1);
            Rb += BigInt::from_u64(in.theta[o]) * eisT[o];
        } else {
            Rb += BigInt::from_u64(in.theta[o]) * pk;
        }
    }
    const uint64_t E = static_cast<uint64_t>(e);
    uint64_t bound = E + E * std::max<uint64_t>(1, BitLen(Rb));                 // e_theta: binom(e, i) R^i
    std::vector<uint64_t> bU(in.nU), bT(nT);
    for (size_t i = 0; i < in.nU; ++i) {
        bU[i] = E + E * BitLen(BigInt::pow(BigInt::from_u64(U[i]), k - 1));      // g_p: binom(e, i) (p^{k-1})^i
        bound = std::max(bound, bU[i]);
    }
    for (size_t o = 0; o < nT; ++o) {
        bT[o] = E + E * BitLen(eisT[o]);                                          // charpoly(T_p | E)
        bound = std::max(bound, bT[o]);
    }
    const uint64_t need = bound + 1 + 64;   // slack

    // ---- the word primes: the retained runs first, then the SAME list by RunAtPrime (the fallback) ----
    const std::vector<PrimeRun>* kept = (in.runs && !in.force_recompute) ? in.runs : nullptr;
    std::vector<uint32_t> primes;
    if (kept)
        for (const PrimeRun& r : *kept) primes.push_back(r.ell);
    {
        std::vector<uint32_t> chk;
        ExtendWordPrimes(chk, primes.size(), N);
        ENGINE_CHECK_MSG(chk == primes, "HeckeZEisUp: the retained runs are not a prefix of the word-prime list");
    }
    std::vector<PrimeRun> extra;   // index-aligned with primes beyond the kept ones
    const size_t nkept = primes.size();
    const uint32_t T = std::max<uint32_t>(1, in.rs.threads);
    std::vector<PrimeRead> reads;
    std::vector<uint32_t> used;
    uint64_t covered = 0;
    size_t idx = 0;
    auto run_at = [&](size_t i) -> const PrimeRun& {
        if (i < nkept) {
            ++R.reused;
            return (*kept)[i];
        }
        if (i - nkept >= extra.size() || extra[i - nkept].mats.empty()) {
            const size_t to = std::min<size_t>(std::max<size_t>(i + 1, i + T), in.rs.word_prime_cap);
            ExtendWordPrimes(primes, to, N);
            std::vector<uint32_t> tail(primes.begin() + nkept, primes.end());
            RunPrimes(N, n, in.ops, tail, i - nkept, to - nkept, T, extra);
            R.computed += static_cast<uint32_t>(to - i);
        }
        return extra[i - nkept];
    };
    auto next_read = [&]() -> bool {   // reads the next prime of the list; false when it was skipped
        if (idx >= in.rs.word_prime_cap) throw WordPrimeCapRefusal(in.rs.word_prime_cap);
        const PrimeRun& run = run_at(idx);
        ENGINE_CHECK_MSG(run.h1 == in.b1 && run.mats.size() == m, "HeckeZEisUp: dim H^1 at a word prime differs from b1");
        PrimeRead pr;
        const bool ok = ReadAtPrime(run, in.ops, k, e, in.theta, &pr);
        const uint32_t q = run.ell;
        if (idx >= nkept) extra[idx - nkept] = PrimeRun();   // a fallback run's matrices are dropped once read
        ++idx;
        if (!ok) {
            R.skipped.push_back(q);
            return false;
        }
        reads.push_back(std::move(pr));
        used.push_back(q);
        covered += BitLen64(q) - 1u;
        return true;
    };
    while (covered < need) next_read();

    // ---- CRT: e_theta, g_p, charpoly(T_p | E) ----
    auto crt = [&](const std::vector<std::vector<uint64_t>>& per_prime, uint64_t bits) {
        ZPoly out;
        const size_t len = per_prime[0].size();
        for (size_t c = 0; c < len; ++c) {
            std::vector<uint32_t> res;
            for (const auto& v : per_prime) res.push_back(static_cast<uint32_t>(v[c]));
            out.push_back(FromHex(ZFromResidues(used, res, bits).hex()));
        }
        return out;
    };
    auto charpolys = [&](const std::function<const std::vector<u64>&(const PrimeRead&)>& sel) {
        std::vector<std::vector<uint64_t>> pp;
        for (const PrimeRead& r : reads) pp.push_back(CharpolyHessenberg(sel(r), static_cast<uint32_t>(e), r.q));
        return pp;
    };
    ZPoly eth = crt(charpolys([](const PrimeRead& r) -> const std::vector<u64>& { return r.theta; }), bound);
    std::vector<ZPoly> gp(in.nU);
    for (size_t i = 0; i < in.nU; ++i) {
        const size_t o = nT + i;
        gp[i] = crt(charpolys([o](const PrimeRead& r) -> const std::vector<u64>& { return r.R[o]; }), bound);
    }
    // C-E3: charpoly(T_p | E) = (x - (1 + p^{k-1}))^{e} for every T_p of the list ((F2); T_ell is not in the list, audit m6)
    for (size_t o = 0; o < nT; ++o) {
        const ZPoly ct = crt(charpolys([o](const PrimeRead& r) -> const std::vector<u64>& { return r.R[o]; }), bound);
        ENGINE_CHECK_MSG(ct == ProdLinear(std::vector<BigInt>(e, eisT[o])),
                         "HeckeZEisUp C-E3: charpoly(T_p | E) is not (x - (1 + p^{k-1}))^{e_k} (contradicts prop:heckeZeisup (1))");
    }

    // ---- C-E4: the integer roots of every g_p from C_p, certified by exact division ----
    std::vector<std::vector<BigInt>> roots(in.nU), distinct(in.nU);
    for (size_t i = 0; i < in.nU; ++i) {
        std::vector<BigInt> cand = {BigInt(0), BigInt(1), BigInt(-1)};
        BigInt pj(1);
        for (uint32_t j = 1; j + 1 <= k - 1 + (tz.drop_top_candidate ? 0 : 1); ++j) {
            pj = pj * BigInt::from_u64(U[i]);
            cand.push_back(pj);
            cand.push_back(-pj);
        }
        ZPoly g = gp[i];
        for (const BigInt& c : cand) {
            bool any = false;
            while (g.size() > 1) {
                BigInt rem;
                const ZPoly q = DivLinear(g, c, &rem);
                if (!rem.is_zero()) break;
                g = q;
                roots[i].push_back(c);
                any = true;
            }
            if (any) distinct[i].push_back(c);
        }
        ENGINE_CHECK_MSG(g.size() == 1 && g[0] == BigInt(1),
                         "HeckeZEisUp C-E4: g_p = charpoly(U_p | E) does not split over C_p = {0, +-1, +-p^j} (prop:heckeZeisup (4))");
    }

    // ---- q*: the first used prime with e_theta mod q* separable and the distinct roots of every g_p pairwise incongruent ----
    auto eligible = [&](u64 q) {
        std::vector<u64> ev;   // the roots of e_theta are not known yet: separability by the gcd with the derivative
        for (const BigInt& c : eth) ev.push_back(ModOf(c, q));
        // separable mod q iff gcd(e, e') = 1 over F_q
        auto trim = [](std::vector<u64>& p) { while (!p.empty() && p.back() == 0) p.pop_back(); };
        std::vector<u64> a = ev, b;
        for (size_t i = 1; i < ev.size(); ++i) b.push_back(ev[i] * (i % q) % q);
        trim(a);
        trim(b);
        while (!b.empty()) {
            while (a.size() >= b.size() && !a.empty()) {
                const u64 f = a.back() * InvMod(b.back(), q) % q;
                const size_t sh = a.size() - b.size();
                for (size_t i = 0; i < b.size(); ++i) a[sh + i] = (a[sh + i] + (q - f) * b[i]) % q;
                trim(a);
            }
            std::swap(a, b);
        }
        if (a.size() != 1) return false;
        for (const auto& ds : distinct) {
            std::set<u64> s;
            for (const BigInt& r : ds) s.insert(ModOf(r, q));
            if (s.size() != ds.size()) return false;
        }
        return true;
    };
    size_t star = reads.size();
    for (size_t i = 0; i < reads.size(); ++i)
        if (eligible(reads[i].q)) {
            star = i;
            break;
        }
    while (star == reads.size()) {   // finitely many primes are not eligible: read further primes of the same list
        if (next_read() && eligible(reads.back().q)) star = reads.size() - 1;
    }
    const PrimeRead& S = reads[star];
    const u64 qs = S.q;
    R.id_prime = static_cast<uint32_t>(qs);
    R.primes_used = static_cast<uint32_t>(used.size());

    // ---- the identification at q* (prop:heckeZeisup (4)) ----
    // every tuple t in prod_p distinct(g_p): mu_t = lambda_t(theta) mod q*; a theta-eigenline of K_q* is ker(theta - mu) of dimension 1
    std::vector<std::vector<BigInt>> lineU;   // per line, lambda(U_p)
    std::vector<u64> lineMu;
    {
        std::vector<size_t> pos(in.nU, 0);
        std::set<u64> seen;
        BigInt base(0);
        for (size_t o = 0; o < nT; ++o) base += BigInt::from_u64(in.theta[o]) * eisT[o];
        for (;;) {
            BigInt lam = base;
            for (size_t i = 0; i < in.nU; ++i) lam += BigInt::from_u64(in.theta[nT + i]) * distinct[i][pos[i]];
            const u64 mu = ModOf(lam, qs);
            if (seen.insert(mu).second) {
                std::vector<u64> A = S.theta;
                for (size_t x = 0; x < e; ++x) A[x * e + x] = (A[x * e + x] + qs - mu) % qs;
                std::vector<size_t> fr;
                const auto ker = Kernel(A, e, e, qs, &fr);
                ENGINE_CHECK_MSG(ker.size() <= 1, "HeckeZEisUp: theta mod q* has a repeated eigenvalue on K_q* (e_theta mod q* is separable)");
                if (ker.size() == 1) {
                    const std::vector<u64>& v = ker[0];
                    size_t piv = 0;
                    while (v[piv] == 0) ++piv;
                    std::vector<BigInt> tup(in.nU);
                    for (size_t i = 0; i < in.nU; ++i) {
                        // the scalar of U_p on the line (column convention: U v = R v)
                        std::vector<u64> w(e, 0);
                        const std::vector<u64>& M = S.R[nT + i];
                        for (size_t r = 0; r < e; ++r)
                            for (size_t c = 0; c < e; ++c) w[r] = (w[r] + M[r * e + c] * v[c]) % qs;
                        const u64 s = w[piv] * InvMod(v[piv], qs) % qs;
                        for (size_t r = 0; r < e; ++r)
                            ENGINE_CHECK_MSG(w[r] == s * v[r] % qs, "HeckeZEisUp: U_p does not act by a scalar on a theta-eigenline of K_q*");
                        int found = -1;
                        for (size_t t = 0; t < distinct[i].size(); ++t)
                            if (ModOf(distinct[i][t], qs) == s) found = static_cast<int>(t);
                        ENGINE_CHECK_MSG(found >= 0, "HeckeZEisUp: the scalar of U_p on a line is no root of g_p mod q*");
                        tup[i] = distinct[i][static_cast<size_t>(found)];
                    }
                    lineU.push_back(tup);
                    lineMu.push_back(mu);
                }
            }
            size_t i = 0;
            while (i < in.nU && ++pos[i] == distinct[i].size()) pos[i++] = 0;
            if (i == in.nU) break;
        }
    }
    ENGINE_CHECK_MSG(lineU.size() == e, "HeckeZEisUp C-E5: the theta-eigenlines of K_q* are not e_k lines with a tuple of roots each");

    // (tamper (f), validation only) swap one U_p value between two lines
    if (tz.swap_p) {
        size_t ip = 0;
        while (ip < in.nU && U[ip] != tz.swap_p) ++ip;
        ENGINE_CHECK_MSG(ip < in.nU, "HeckeZEisUp tamper: swap_p is no prime of N");
        auto find_line = [&](const std::vector<std::string>& t) {
            for (size_t j = 0; j < lineU.size(); ++j) {
                bool eq = t.size() == in.nU;
                for (size_t i = 0; eq && i < in.nU; ++i) eq = ToDec(lineU[j][i]) == t[i];
                if (eq) return j;
            }
            return lineU.size();
        };
        const size_t a = find_line(tz.swap_a), b = find_line(tz.swap_b);
        ENGINE_CHECK_MSG(a < lineU.size() && b < lineU.size(), "HeckeZEisUp tamper: the swapped tuples are not lines");
        std::swap(lineU[a][ip], lineU[b][ip]);
    }

    // ---- the exact closure: C-E5, C-E4 (g_p = prod), C-E2 (e_theta | f_theta) ----
    std::vector<uint32_t> th = in.theta;
    if (tz.theta_after) th[0] += 1;
    std::vector<BigInt> lam(e);
    for (size_t j = 0; j < e; ++j) {
        BigInt s(0);
        for (size_t o = 0; o < nT; ++o) s += BigInt::from_u64(th[o]) * eisT[o];
        for (size_t i = 0; i < in.nU; ++i) s += BigInt::from_u64(th[nT + i]) * lineU[j][i];
        lam[j] = s;
        ENGINE_CHECK_MSG(ModOf(s, qs) == lineMu[j], "HeckeZEisUp C-E5: lambda_j(theta) is not the theta-eigenvalue of line j mod q*");
    }
    for (size_t a = 0; a < e; ++a)
        for (size_t b = a + 1; b < e; ++b)
            ENGINE_CHECK_MSG(lam[a] != lam[b], "HeckeZEisUp C-E5: two Eisenstein systems with one theta-value (theta separates them under (C1))");
    ENGINE_CHECK_MSG(ProdLinear(lam) == eth, "HeckeZEisUp C-E5: prod_j (x - lambda_j(theta)) != e_theta");
    for (size_t i = 0; i < in.nU; ++i) {
        std::vector<BigInt> col;
        for (size_t j = 0; j < e; ++j) col.push_back(lineU[j][i]);
        ENGINE_CHECK_MSG(ProdLinear(col) == gp[i], "HeckeZEisUp C-E4: g_p != prod_j (x - lambda_j(U_p))");
    }
    ZPoly f = PolyFromHex(in.f);
    if (tz.f_plus_one) f[0] += BigInt(1);
    ENGINE_CHECK_MSG(f.back() == BigInt(1), "HeckeZEisUp: f_theta is not monic");
    ZPoly c = f;
    for (size_t j = 0; j < e; ++j) {
        BigInt rem;
        c = DivLinear(c, lam[j], &rem);
        ENGINE_CHECK_MSG(rem.is_zero(), "HeckeZEisUp C-E2: e_theta does not divide f_theta exactly over Z");
    }
    ENGINE_CHECK_MSG(c.size() == static_cast<size_t>(in.d - in.e_k) + 1, "HeckeZEisUp C-E2: deg c_theta != d - e_k");

    // ---- the result, lines ordered by the tuple ----
    std::vector<size_t> ord(e);
    for (size_t j = 0; j < e; ++j) ord[j] = j;
    std::sort(ord.begin(), ord.end(), [&](size_t a, size_t b) { return TupleLess(lineU[a], lineU[b]); });
    R.e_theta = PolyHex(eth);
    for (const ZPoly& g : gp) R.g.push_back(PolyHex(g));
    for (size_t j : ord) {
        EisLine L;
        for (const BigInt& x : lineU[j]) L.U.push_back(ToHex(x));
        L.lambda_theta = ToHex(lam[j]);
        R.lines.push_back(std::move(L));
    }
    R.c_theta = PolyHex(c);
    R.secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return R;
}

std::vector<std::vector<std::string>> PredictedEisUp(uint32_t N, uint32_t k, bool* predicted) {
    std::vector<std::vector<std::string>> out;
    *predicted = N > 1 && IsSquarefree(N);
    if (!*predicted) return out;
    const std::vector<uint32_t> ps = PrimesOf(N);
    std::vector<std::vector<BigInt>> tuples;
    for (size_t mask = 0; mask < (size_t(1) << ps.size()); ++mask) {
        std::vector<BigInt> t;
        bool all_top = true;
        for (size_t i = 0; i < ps.size(); ++i) {
            const bool top = (mask >> i) & 1;
            all_top = all_top && top;
            t.push_back(top ? BigInt::pow(BigInt::from_u64(ps[i]), k - 1) : BigInt(1));
        }
        if (k == 2 && all_top) continue;
        tuples.push_back(t);
    }
    std::sort(tuples.begin(), tuples.end(), TupleLess);
    for (const auto& t : tuples) {
        std::vector<std::string> s;
        for (const BigInt& x : t) s.push_back(ToHex(x));
        out.push_back(s);
    }
    return out;
}

std::string LambdaOf(const EisUpResult& R, size_t j, const std::vector<uint32_t>& thP, const std::vector<uint32_t>& coeffs) {
    ENGINE_CHECK_MSG(j < R.lines.size() && coeffs.size() == thP.size(), "HeckeZEisUp LambdaOf: arguments");
    const std::vector<uint32_t> U = R.Uprimes();
    const uint32_t k = R.n + 2;
    BigInt s(0);
    for (size_t i = 0; i < thP.size(); ++i) {
        if (!coeffs[i]) continue;
        const uint32_t p = thP[i];
        BigInt v;
        if (R.N % p != 0) {
            v = BigInt::pow(BigInt::from_u64(p), k - 1) + BigInt(1);   // T_p ((F2), p !| N; T_ell included)
        } else {
            const auto it = std::find(U.begin(), U.end(), p);
            ENGINE_CHECK_MSG(it != U.end(), "HeckeZEisUp LambdaOf: U_p with p | N not in the list");
            v = FromHex(R.lines[j].U[static_cast<size_t>(it - U.begin())]);
        }
        s += BigInt::from_u64(coeffs[i]) * v;
    }
    return ToHex(s);
}

uint64_t HexModEll(const std::string& hex, uint64_t ell) { return ModOf(FromHex(hex), ell); }

std::string HexPlus(const std::string& hex, int64_t delta) { return ToHex(FromHex(hex) + BigInt(delta)); }

Attribution AttributeAtEll(const std::vector<uint64_t>& fbar, const std::vector<uint64_t>& lam_mod, uint64_t ell) {
    // (slot-27 EXW) ell < 2^63: the (C3) read on the free block certifies generation at ell >= 2^31 (class C of EXW_design.md Sec. 1); the
    // products are taken in 128 bits there (below 2^31 the 64-bit route as before, byte for byte)
    ENGINE_CHECK_MSG(!fbar.empty() && fbar.back() == 1 % ell && ell >= 2 && ell < (1ull << 63), "HeckeZEisUp AttributeAtEll: f-bar monic, ell < 2^63");
    ENGINE_CHECK_MSG(lam_mod.size() + 1 <= fbar.size(), "HeckeZEisUp AttributeAtEll: more Eisenstein components than deg f-bar");
    std::vector<uint64_t> c = fbar;
    for (uint64_t lm : lam_mod) {
        const uint64_t l = lm % ell;
        const size_t dg = c.size() - 1;
        std::vector<uint64_t> q(dg, 0);
        uint64_t acc = c[dg];
        for (size_t i = dg; i-- > 0;) {
            q[i] = acc;
            acc = ell < (1ull << 31) ? (c[i] + acc * l) % ell
                                     : static_cast<uint64_t>((static_cast<unsigned __int128>(acc) * l + c[i]) % ell);
        }
        ENGINE_CHECK_MSG(acc == 0, "HeckeZEisUp C-E7: prod_j (x - lambda_j(theta')) does not divide f-bar_{theta'} mod ell (remainder != 0)");
        c = std::move(q);
    }
    Attribution A;
    A.cbar = std::move(c);
    return A;
}

namespace {
// (slot-27 EXW) v_ell(x) for x != 0 and any ell < 2^63: BigInt::valuation below 2^32, else repeated exact division
uint32_t ValuationU64(const BigInt& x, uint64_t ell) {
    if (ell < (1ull << 32)) return x.valuation(static_cast<uint32_t>(ell));
    const BigInt L = BigInt::from_u64(ell);
    BigInt y = x, q, r;
    uint32_t v = 0;
    for (;;) {
        BigInt::divmod(y, L, q, r);
        if (!r.is_zero()) return v;
        y = q;
        ++v;
    }
}
}  // namespace

uint32_t ValuationOfDifference(const std::string& a, const std::string& b, uint64_t ell) {
    const BigInt d = FromHex(a) - FromHex(b);
    ENGINE_CHECK_MSG(!d.is_zero(), "HeckeZEisUp: two Eisenstein components with one theta'-value (theta' generates T (x) Z_ell)");
    ENGINE_CHECK_MSG(ell >= 2 && ell < (1ull << 63), "HeckeZEisUp: ell < 2^63");
    return ValuationU64(d, ell);
}

uint32_t CuspResultantValuation(const std::vector<std::string>& f, const std::vector<std::string>& lambdas, size_t j, uint64_t ell) {
    ZPoly c = PolyFromHex(f);
    for (const std::string& l : lambdas) {
        BigInt rem;
        c = DivLinear(c, FromHex(l), &rem);
        ENGINE_CHECK_MSG(rem.is_zero(), "HeckeZEisUp: prod_j (x - lambda_j(theta')) does not divide f_theta' over Z");
    }
    const BigInt v = EvalZ(c, FromHex(lambdas.at(j)));
    ENGINE_CHECK_MSG(!v.is_zero(), "HeckeZEisUp: c_theta'(lambda_j) = 0 (f_theta' is separable under (C1))");
    ENGINE_CHECK_MSG(ell >= 2 && ell < (1ull << 63), "HeckeZEisUp: ell < 2^63");
    return ValuationU64(v, ell);
}

}  // namespace eisup
}  // namespace heckez
