// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// DESCENT milestone D3, part (A): char_Z(T_p | H^1(Gamma_0(N); Sym^n Z^2)/tors) by residues, the certified bound and Garner's mixed
// radix in words ([C] prop:crtcharpolyN (1)-(5), v1.23.0-pre.31; code-map obligation (d)).  See DescentHecke.h.
#include "DescentHecke.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <thread>

#if defined(__GLIBC__)
#include <malloc.h>   // malloc_trim
#endif

#include "DescentLevelN.h"
#include "../IntegralCharpolyCrt.h"
#include "../../Hecke/HeckeLevelN.h"
#include "../../Hecke/ExactInt.h"
#include "../../Hecke/HeckeLevelNSparse.h"
#include "../../../ModArithmetic/EngineCheck.h"
#include "../../../ModArithmetic/ScalarMod.h"
#include "../../../ModArithmetic/Fnv.h"
#include "../../../Engine/Threads.h"

namespace integral_descent {
namespace hecke3 {

namespace {
std::atomic<uint32_t> g_fault{0};
double since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}
// (S24 CL-G3, C-07) the shared scalar helpers (ScalarMod.h).
uint64_t addmod(uint64_t a, uint64_t b, uint64_t Q) { return engine::scalar::addmod<uint64_t>(a, b, Q); }   // a, b < Q (Q > 0)
uint32_t powmod32(uint64_t a, uint64_t e, uint32_t m) { return engine::scalar::powmod<uint32_t>(static_cast<uint32_t>(a % m), e, m); }
uint32_t DefaultThreads() { return engine::threads::configured(engine::threads::kCap16); }   // (S24 CL-5, C-05) ENGINE_THREADS, cap 16
}  // namespace

// ---------------------------------------------------------------------------------------------------------------------------------
// refusals, faults
// ---------------------------------------------------------------------------------------------------------------------------------
const char* D3RefusalName(D3Refusal r) {
    switch (r) {
        case D3Refusal::None: return "None";
        case D3Refusal::NotPrime: return "NotPrime";
        case D3Refusal::PDividesN: return "PDividesN";
        case D3Refusal::Lift: return "Lift";
        case D3Refusal::DegreeRange: return "DegreeRange";
        case D3Refusal::RssCeiling: return "RssCeiling";
        case D3Refusal::GroupUncertified: return "GroupUncertified";
        case D3Refusal::ModulePrecision: return "ModulePrecision";
        case D3Refusal::NotEndomorphism: return "NotEndomorphism";
        case D3Refusal::NotCocycle: return "NotCocycle";
        case D3Refusal::IntegerCap: return "IntegerCap";
        case D3Refusal::DomainCap: return "DomainCap";
        case D3Refusal::KerCokerPrecision: return "KerCokerPrecision";
    }
    return "?";
}

D3Refusal ClassifyD3Exception(const std::exception& e) {
    switch (pair::ClassifyPairException(e)) {
        case pair::PairRefusal::IntegerCap: return D3Refusal::IntegerCap;
        case pair::PairRefusal::DomainCap: return D3Refusal::DomainCap;
        default: return D3Refusal::None;   // InputTableCertificate cannot arise in the operator's assembly; everything else propagates
    }
}

D3Fault ActiveD3Fault() { return static_cast<D3Fault>(g_fault.load()); }
ScopedD3Fault::ScopedD3Fault(D3Fault f) : saved_(g_fault.exchange(static_cast<uint32_t>(f))) {}
ScopedD3Fault::~ScopedD3Fault() { g_fault.store(saved_); }

// ---------------------------------------------------------------------------------------------------------------------------------
// the bound and the word arithmetic of (3), (4)
// ---------------------------------------------------------------------------------------------------------------------------------
uint32_t BitLen(uint64_t x) {
    uint32_t b = 0;
    while (x) {
        ++b;
        x >>= 1;
    }
    return b;
}

uint64_t CoefficientBitBound(uint32_t r, uint32_t n, uint32_t p, uint32_t i) {
    if (ActiveD3Fault() == D3Fault::NoEisensteinBound)   // VALIDATION ONLY: the cusp half 2 p^{(n+1)/2} without the Eisenstein term
        return uint64_t(r) + uint64_t(i) * ((uint64_t(n + 1) * BitLen(p) + 1) / 2);
    return uint64_t(r) + uint64_t(i) * uint64_t(n + 1) * BitLen(p);
}

uint64_t WordMulMod(uint64_t a, uint64_t b, uint64_t Q) {
    if (Q == 0) return a * b;   // 2^64: the native wrap-around
    ENGINE_CHECK_MSG(a < Q, "WordMulMod: a < Q");
    uint64_t r = 0;
    for (int bit = 63; bit >= 0; --bit) {
        r = addmod(r, r, Q);
        if ((b >> bit) & 1u) r = addmod(r, a, Q);
    }
    return r;
}

std::vector<uint32_t> GarnerInverses(const std::vector<uint32_t>& primes) {
    std::vector<uint32_t> inv(primes.size(), 1);
    for (size_t k = 0; k < primes.size(); ++k) {
        const uint64_t lk = primes[k];
        ENGINE_CHECK_MSG(lk > 2 && lk % 2 == 1, "GarnerInverses: odd primes");
        uint64_t prod = 1;
        for (size_t j = 0; j < k; ++j) prod = prod * (primes[j] % lk) % lk;
        ENGINE_CHECK_MSG(prod != 0, "GarnerInverses: the primes must be pairwise distinct");
        inv[k] = powmod32(prod, lk - 2, static_cast<uint32_t>(lk));
    }
    return inv;
}

std::vector<uint32_t> GarnerDigits(const std::vector<uint32_t>& res, const std::vector<uint32_t>& primes, const std::vector<uint32_t>& inv) {
    const size_t K = res.size();
    ENGINE_CHECK_MSG(K <= primes.size() && K <= inv.size(), "GarnerDigits: sizes");
    std::vector<uint32_t> d(K, 0);
    for (size_t k = 0; k < K; ++k) {
        const uint64_t lk = primes[k];
        // the partial mixed-radix number (digits 0 .. k-1) mod ell_k, by Horner from the top; every product < ell_k^2 < 2^64
        uint64_t acc = 0;
        for (size_t m = k; m-- > 0;) acc = (acc * (primes[m] % lk) + d[m]) % lk;
        const uint64_t diff = (res[k] % lk + lk - acc) % lk;
        d[k] = static_cast<uint32_t>(diff * inv[k] % lk);
    }
    return d;
}

bool MixedRadixNegative(const std::vector<uint32_t>& digits, const std::vector<uint32_t>& primes) {
    for (size_t k = digits.size(); k-- > 0;) {
        const uint32_t h = (primes[k] - 1) / 2;
        if (digits[k] != h) return digits[k] > h;
    }
    return false;   // c^+ = (P - 1)/2 exactly: c = c^+ >= 0
}

uint64_t MixedRadixModQ(const std::vector<uint32_t>& digits, const std::vector<uint32_t>& primes, bool negative, uint64_t Q) {
    auto red = [&](uint64_t x) { return Q == 0 ? x : x % Q; };
    auto add = [&](uint64_t a, uint64_t b) { return Q == 0 ? a + b : addmod(a, b, Q); };
    uint64_t acc = 0;
    for (size_t k = digits.size(); k-- > 0;) acc = add(WordMulMod(acc, primes[k], Q), red(digits[k]));
    if (!negative) return acc;
    uint64_t P = red(1);
    for (size_t k = 0; k < digits.size(); ++k) P = WordMulMod(P, primes[k], Q);
    // (c^+ - P) mod Q
    if (Q == 0) return acc - P;
    return acc >= P ? acc - P : acc + (Q - P);
}

bool MixedRadixToInt64(const std::vector<uint32_t>& digits, const std::vector<uint32_t>& primes, bool negative, int64_t* value) {
    std::vector<uint32_t> e = digits;
    if (negative) {
        // |c| = P - c^+: digit-wise subtraction from P = (0, ..., 0) with a final carry
        uint32_t borrow = 0;
        for (size_t k = 0; k < e.size(); ++k) {
            const int64_t t = -int64_t(digits[k]) - borrow;
            if (t < 0) {
                e[k] = static_cast<uint32_t>(t + primes[k]);
                borrow = 1;
            } else {
                e[k] = static_cast<uint32_t>(t);
                borrow = 0;
            }
        }
    }
    const uint64_t lim = (uint64_t(1) << 63) - 1;
    uint64_t acc = 0;
    for (size_t k = e.size(); k-- > 0;) {
        if (acc > (lim - e[k]) / primes[k]) return false;
        acc = acc * primes[k] + e[k];
    }
    *value = negative ? -static_cast<int64_t>(acc) : static_cast<int64_t>(acc);
    return true;
}

// ---------------------------------------------------------------------------------------------------------------------------------
// residues on the O-piece: the shipped H_{N,p} on M_N / Rel_N over F_ell'
// ---------------------------------------------------------------------------------------------------------------------------------
std::vector<uint32_t> CharpolyFp(std::vector<uint32_t> H, uint32_t n, uint32_t q) {
    ENGINE_CHECK_MSG(H.size() == size_t(n) * n, "CharpolyFp: square");
    auto at = [&](uint32_t i, uint32_t j) -> uint32_t& { return H[size_t(i) * n + j]; };
    const uint64_t Q = q;
    for (uint32_t c = 0; c + 2 < n; ++c) {
        uint32_t piv = n;
        for (uint32_t i = c + 1; i < n; ++i)
            if (at(i, c)) {
                piv = i;
                break;
            }
        if (piv == n) continue;
        if (piv != c + 1) {
            for (uint32_t j = 0; j < n; ++j) std::swap(at(piv, j), at(c + 1, j));
            for (uint32_t i = 0; i < n; ++i) std::swap(at(i, piv), at(i, c + 1));
        }
        const uint64_t inv = powmod32(at(c + 1, c), Q - 2, q);
        for (uint32_t i = c + 2; i < n; ++i) {
            if (!at(i, c)) continue;
            const uint64_t f = at(i, c) * inv % Q;
            // row_i -= f row_{c+1}; col_{c+1} += f col_i
            uint32_t* ri = &at(i, 0);
            const uint32_t* rc = &at(c + 1, 0);
            const uint64_t nf = (Q - f) % Q;
            for (uint32_t j = 0; j < n; ++j)
                if (rc[j]) ri[j] = static_cast<uint32_t>((ri[j] + nf * rc[j]) % Q);
            for (uint32_t r = 0; r < n; ++r) {
                const uint32_t v = at(r, i);
                if (v) at(r, c + 1) = static_cast<uint32_t>((at(r, c + 1) + f * v) % Q);
            }
        }
    }
    // p_k = (x - h_kk) p_{k-1} - sum_{i<k} h_{ik} prod_{j=i+1}^{k} h_{j,j-1} p_{i-1}   (low -> high internally)
    std::vector<std::vector<uint64_t>> P(n + 1);
    P[0] = {1};
    for (uint32_t k = 1; k <= n; ++k) {
        std::vector<uint64_t> pk(k + 1, 0);
        const uint64_t hkk = at(k - 1, k - 1);
        for (uint32_t t = 0; t < k; ++t) {
            pk[t + 1] = (pk[t + 1] + P[k - 1][t]) % Q;
            pk[t] = (pk[t] + (Q - hkk) % Q * P[k - 1][t]) % Q;
        }
        uint64_t prod = 1;
        for (uint32_t i = k - 1; i >= 1; --i) {
            prod = prod * at(i, i - 1) % Q;
            if (!prod) break;
            const uint64_t f = prod * at(i - 1, k - 1) % Q;
            if (f)
                for (uint32_t t = 0; t < P[i - 1].size(); ++t) pk[t] = (pk[t] + (Q - f) * P[i - 1][t]) % Q;
        }
        P[k] = std::move(pk);
    }
    std::vector<uint32_t> out(n + 1);
    for (uint32_t t = 0; t <= n; ++t) out[n - t] = static_cast<uint32_t>(P[n][t]);
    return out;
}

std::vector<uint32_t> ShippedResidueCharpoly(uint32_t N, uint32_t n, uint32_t p, uint32_t ellp, uint32_t* h1_out, bool check_rel) {
    ENGINE_CHECK_MSG(ellp >= 5 && ellp > n + 1 && IsPrime(ellp), "ShippedResidueCharpoly: ell' >= 5 prime, ell' > n + 1 (s_ell' = 0)");
    if (n % 2 == 1) {
        if (h1_out) *h1_out = 0;
        return {1};   // r_N = 0 (prop:crtcharpolyN(3), t1)
    }
    ENGINE_CHECK_MSG(N % p != 0, "ShippedResidueCharpoly: p !| N (T_p)");
    const hecke::level::Gamma0LevelN lvl(N);
    const U32Ctx F(ellp);
    // the shipped sparse route of CokernelHeckeMatrices (value-identical to its dense route, HeckeLevelNSparseValidation (S)), its pieces
    // called directly: the shipped wrapper re-checks prop:heilbronn(1) on EVERY generator of Rel_N at every prime (~ 0.83 D applications of
    // H_{N,p}, cubic in d at level 1 -- five times the operator itself); a theorem, so the record checks it at two primes (the first
    // residue prime and the held-out prime, check_rel) and not at the others.
    const hecke::level::SparseCokernel Q = hecke::level::SparseCokernel::Build(lvl, n, F, check_rel);
    const hecke::level::HeilbronnOperator H(lvl, p, n, F);
    if (check_rel)
        ENGINE_CHECK_MSG(hecke::level::RelEscapesOnGenerators(Q, H) == 0, "ShippedResidueCharpoly: H(Rel) not in Rel (prop:heilbronn (1))");
    // [H] on the free-column basis, column by column: H e_c is the sum over the terms with source block b(c) of the c-th COLUMN of their
    // block (O(d) per term; the shipped OnCokernelSparse multiplies whole blocks, O(d^2) per term); the same matrix, entry for entry --
    // checked against OnCokernelSparse at every check_rel prime
    const uint32_t d = n + 1, Dm = H.D(), h = Q.h1();
    std::vector<std::vector<uint32_t>> by_src(H.m() + 1);
    for (uint32_t t = 0; t < H.terms().size(); ++t) by_src[H.terms()[t].src].push_back(t);
    hecke::level::HMat T;
    T.rows = T.cols = h;
    T.modulus = T.ell = ellp;
    T.field = true;
    T.row_exp.assign(h, 1);
    T.a.assign(size_t(h) * h, 0);
    std::vector<uint64_t> acc(Dm);
    std::vector<uint32_t> v(Dm);
    for (uint32_t j = 0; j < h; ++j) {
        const uint32_t fc = Q.free_columns()[j], b = fc / d + 1, comp = fc % d;
        std::fill(acc.begin(), acc.end(), 0);
        for (uint32_t t : by_src[b]) {
            const hecke::level::HeilbronnTerm& tm = H.terms()[t];
            const std::vector<uint32_t>& blk = H.blocks()[tm.block].dense;
            uint64_t* out = acc.data() + size_t(tm.dst - 1) * d;
            for (uint32_t r = 0; r < d; ++r) out[r] += blk[size_t(r) * d + comp];
        }
        for (uint32_t i = 0; i < Dm; ++i) v[i] = static_cast<uint32_t>(acc[i] % ellp);
        const std::vector<uint32_t> c = Q.coords_dense(v);
        for (uint32_t i = 0; i < h; ++i) T.a[size_t(i) * h + j] = c[i];
    }
    if (check_rel) {
        const hecke::level::HMat S = hecke::level::OnCokernelSparse(Q, H);
        ENGINE_CHECK_MSG(S.rows == T.rows && S.a == T.a, "ShippedResidueCharpoly: the column evaluation differs from the shipped OnCokernelSparse");
    }
    ENGINE_CHECK_MSG(T.rows == T.cols && T.field && T.modulus == ellp, "ShippedResidueCharpoly: a square field matrix");
    if (h1_out) *h1_out = T.rows;
    std::vector<uint32_t> A(size_t(T.rows) * T.cols);
    for (size_t i = 0; i < A.size(); ++i) A[i] = static_cast<uint32_t>(T.a[i] % ellp);
    return CharpolyFp(std::move(A), T.rows, ellp);
}

uint64_t PredictCharpolyRss(uint32_t N, uint32_t n, uint32_t threads, uint32_t p) {
    // per concurrent prime: the shipped sparse cokernel (calibrated at D1: <= 1.2 GiB at D = 37 800, ~ 34 kB per unit of D at d > 1
    // -- taken as 48 D bytes x d), the Heilbronn blocks (one d x d block per (epsilon, j): |Mer_p| m d^2 x 4 B, x 1.5 for the term list
    // and the allocator; DESCENT3_predictions.md: 85 MiB per thread measured at (1, 600), p = 13), the r x r HMat (8 B), its Hessenberg
    // copy (4 B) and the recurrence's polynomials (8 B r^2 / 2); plus 256 MiB
    const LevelNCounts c = LevelNClosedForm(N, n);
    const uint64_t D = c.D, d = n + 1, r = c.b1, m = c.m;
    const uint64_t mer = p ? hecke::level::MerelSet(p).size() : 64;
    const uint64_t per = 48 * D * d + 6 * mer * m * d * d + 64 * D + 16 * r * r + (32ull << 20);
    return per * threads + (256ull << 20);
}

std::string CharpolyZ::text() const {
    std::ostringstream o;
    for (size_t i = 0; i < c.size(); ++i) {
        if (i) o << ",";
        const CrtCoefficient& x = c[i];
        if (!x.certified)
            o << "?";
        else if (x.fits)
            o << x.value;
        else
            o << "[" << (x.negative ? "-" : "+") << "," << x.digits.size() << " digits]";
    }
    o << ";fnv=0x" << std::hex << std::setw(16) << std::setfill('0') << digest() << std::dec << ";undetermined=" << undetermined();
    return o.str();
}

uint64_t CharpolyZ::digest() const {
    engine::FnvHasher h;
    h.feed(r);
    for (const CrtCoefficient& x : c) {
        h.feed(x.i);
        if (x.certified) {
            h.feed(1u + (x.negative ? 1u : 0u));
            h.feed(x.mod_2_64);
            h.feed(x.mod_3_40);
        } else {
            h.feed(0);
            h.feed(x.residues.size());
            for (uint32_t v : x.residues) h.feed(v);
        }
    }
    return h.value();
}

uint32_t CharpolyZ::undetermined() const {
    uint32_t k = 0;
    for (const CrtCoefficient& x : c) k += !x.certified;
    return k;
}

CharpolyZ CharpolyOverZ(uint32_t N, uint32_t n, uint32_t p, const CrtOptions& opt) {
    const auto t0 = std::chrono::steady_clock::now();
    CharpolyZ R;
    R.N = N;
    R.n = n;
    R.p = p;
    auto refuse = [&](D3Refusal code, const std::string& w) {
        R.refused = true;
        R.code = code;
        R.refusal = "REFUSED [" + std::string(D3RefusalName(code)) + "] char_Z(T_" + std::to_string(p) + ") at (N, n) = (" + std::to_string(N) +
                    ", " + std::to_string(n) + "): " + w;
        R.seconds = since(t0);
        return R;
    };
    if (p < 2 || !IsPrime(p)) return refuse(D3Refusal::NotPrime, "p is not a prime");
    if (N % p == 0) return refuse(D3Refusal::PDividesN, "p | N: U_p is deferred ([R-Q4])");
    const LevelNCounts cnt = LevelNClosedForm(N, n);
    R.r = (n % 2 == 1) ? 0u : cnt.b1;   // prop:crtcharpolyN(1): r_N = b_1(M_N); 0 for n odd
    const uint32_t r = R.r;
    R.c.resize(r + 1);
    for (uint32_t i = 0; i <= r; ++i) {
        R.c[i].i = i;
        R.c[i].bits = CoefficientBitBound(r, n, p, i);
    }
    R.c[0].certified = R.c[0].fits = true;
    R.c[0].value = 1;
    R.c[0].mod_2_64 = R.c[0].mod_3_40 = 1;
    if (r == 0) {
        R.determined = true;
        R.seconds = since(t0);
        return R;
    }
    // the prime list: the shortest prefix of CrtPrimes with sum (bitlen - 1) >= b_{p,r} + 1, capped by max_primes
    const uint64_t need = R.c[r].bits + 1;
    uint32_t count = static_cast<uint32_t>((need + 28) / 29) + 2;
    if (opt.max_primes) count = std::min(count, opt.max_primes);
    if (ActiveD3Fault() == D3Fault::SinglePrime) count = 1;   // VALIDATION ONLY
    std::vector<uint32_t> all = integral::CrtPrimes(count + 1);
    R.held_out_prime = opt.held_out ? all.back() : 0u;
    all.pop_back();
    {
        uint64_t cov = 0;
        uint32_t k = 0;
        while (k < all.size() && cov < need) cov += BitLen(all[k++]) - 1;
        all.resize(k);   // drop the unneeded tail (the +2 slack)
    }
    R.primes = all;
    const uint32_t K = static_cast<uint32_t>(all.size());
    const uint32_t threads = opt.threads ? opt.threads : DefaultThreads();
    R.rss_pred = PredictCharpolyRss(N, n, threads, p);
    if (R.rss_pred > opt.rss_cap_bytes)
        return refuse(D3Refusal::RssCeiling, "rule 19: predicted " + std::to_string(R.rss_pred >> 20) + " MiB above the ceiling " +
                                                 std::to_string(opt.rss_cap_bytes >> 20) + " MiB (refused before building)");
    // residues, one prime per task (each task is a separate shipped computation over F_ell'); the held-out prime as task K
    std::vector<std::vector<uint32_t>> res(K + (R.held_out_prime ? 1 : 0));
    std::vector<uint32_t> h1(res.size(), 0);
    std::atomic<uint32_t> next{0};
    std::mutex emu;
    std::exception_ptr err;
    const auto t1 = std::chrono::steady_clock::now();
    RssSampler rs;
    auto worker = [&]() {
        for (;;) {
            const uint32_t k = next.fetch_add(1);
            if (k >= res.size()) return;
            const uint32_t ellp = k < K ? all[k] : R.held_out_prime;
            try {
                res[k] = ShippedResidueCharpoly(N, n, p, ellp, &h1[k], /*check_rel=*/k == 0 || k >= K);
            } catch (...) {
                std::lock_guard<std::mutex> lk(emu);
                if (!err) err = std::current_exception();
                next.store(static_cast<uint32_t>(res.size()));
                return;
            }
        }
    };
    {
        std::vector<std::thread> pool;
        for (uint32_t t = 0; t < std::min<uint32_t>(threads, static_cast<uint32_t>(res.size())); ++t) pool.emplace_back(worker);
        for (auto& th : pool) th.join();
    }
    R.rss_meas = rs.stop();
#if defined(__GLIBC__)
    // (the D3 audit's N10) the residue tasks' freed cokernels sit in up to 16 per-thread arenas (about 1 GiB stayed resident after the first
    // p at (10007, 0)); return them, so the next char_Z starts from the base its prediction assumes.  Frees only memory no live object owns.
    malloc_trim(0);
#endif
    if (err) std::rethrow_exception(err);
    R.residue_seconds = since(t1);
    R.b1_shipped = h1[0];
    // a defect, not a refusal (the D3 audit's m2): every ell' > n + 1 has s_ell' = 0, so dim H^1 = b_1 by prop:crtcharpolyN(2)
    for (size_t k = 0; k < res.size(); ++k)
        if (h1[k] != r || res[k].size() != r + 1u || res[k][0] != 1u)
            throw hecke::HeckeInternalError(("[ResidueRank] char_Z(T_" + std::to_string(p) + ") at (N, n) = (" + std::to_string(N) + ", " + std::to_string(n) +
                                             "): the shipped presentation at ell' = " + std::to_string(k < K ? all[k] : R.held_out_prime) + " > n + 1 has dim H^1 = " +
                                             std::to_string(h1[k]) + " != r_N = b_1 = " + std::to_string(r) + " (closed form, prop:descentN(7)(b)): a defect")
                                                .c_str());
    if (ActiveD3Fault() == D3Fault::PerturbedResidue) res[0][1] = (res[0][1] + 1) % all[0];   // VALIDATION ONLY
    if (ActiveD3Fault() == D3Fault::PerturbedDigits) {   // VALIDATION ONLY: c_r's residue at the middle prime of its prefix
        uint64_t cov = 0;
        uint32_t k = 0;
        while (k < K && cov < R.c[r].bits + 1) cov += BitLen(all[k++]) - 1;
        res[k / 2][r] = (res[k / 2][r] + 1) % all[k / 2];
    }
    // Garner per coefficient, on the prefix its bound needs
    const auto t2 = std::chrono::steady_clock::now();
    const std::vector<uint32_t> inv = GarnerInverses(all);
    std::vector<uint64_t> prefix_bits(K + 1, 0);
    for (uint32_t k = 0; k < K; ++k) prefix_bits[k + 1] = prefix_bits[k] + BitLen(all[k]) - 1;
    std::atomic<uint32_t> nexti{1};
    const uint64_t Q340 = [] {
        uint64_t q = 1;
        for (int i = 0; i < 40; ++i) q *= 3;
        return q;
    }();
    auto gworker = [&]() {
        for (;;) {
            const uint32_t i = nexti.fetch_add(1);
            if (i > r) return;
            CrtCoefficient& C = R.c[i];
            const uint64_t want = C.bits + 1;
            uint32_t k = 0;
            while (k < K && prefix_bits[k] < want) ++k;
            if (prefix_bits[k] < want) {
                C.certified = false;
                C.primes_used = K;
                C.residues.resize(K);
                for (uint32_t t = 0; t < K; ++t) C.residues[t] = res[t][i];
                continue;
            }
            std::vector<uint32_t> rr(k);
            for (uint32_t t = 0; t < k; ++t) rr[t] = res[t][i];
            const std::vector<uint32_t> pr(all.begin(), all.begin() + k);
            C.digits = GarnerDigits(rr, pr, inv);
            C.primes_used = k;
            C.certified = true;
            C.negative = MixedRadixNegative(C.digits, pr);
            C.fits = MixedRadixToInt64(C.digits, pr, C.negative, &C.value);
            C.mod_2_64 = MixedRadixModQ(C.digits, pr, C.negative, 0);
            C.mod_3_40 = MixedRadixModQ(C.digits, pr, C.negative, Q340);
            if (C.fits) C.digits.clear();
        }
    };
    {
        std::vector<std::thread> pool;
        for (uint32_t t = 0; t < std::min<uint32_t>(threads, r); ++t) pool.emplace_back(gworker);
        for (auto& th : pool) th.join();
    }
    R.garner_seconds = since(t2);
    R.determined = true;
    for (const auto& C : R.c) R.determined = R.determined && C.certified;
    // the held-out prime: every certified coefficient against its residue there
    if (R.held_out_prime) {
        const std::vector<uint32_t>& hr = res[K];
        for (uint32_t i = 1; i <= r && R.held_out_ok; ++i) {
            const CrtCoefficient& C = R.c[i];
            if (!C.certified) continue;
            uint64_t v;
            if (C.fits) {
                const int64_t m = static_cast<int64_t>(R.held_out_prime);
                int64_t x = C.value % m;
                if (x < 0) x += m;
                v = static_cast<uint64_t>(x);
            } else {
                const std::vector<uint32_t> pr(all.begin(), all.begin() + C.primes_used);
                v = MixedRadixModQ(C.digits, pr, C.negative, R.held_out_prime);
            }
            if (v != hr[i]) R.held_out_ok = false;
        }
        // a defect, not a refusal (the D3 audit's m2): the bound of (3) and the certificate of (4) make a mismatch impossible
        if (!R.held_out_ok)
            throw hecke::HeckeInternalError(("[HeldOutMismatch] char_Z(T_" + std::to_string(p) + ") at (N, n) = (" + std::to_string(N) + ", " + std::to_string(n) +
                                             "): a certified coefficient disagrees with the held-out prime " + std::to_string(R.held_out_prime) + ": a defect")
                                                .c_str());
    }
    R.seconds = since(t0);
    return R;
}

}  // namespace hecke3
}  // namespace integral_descent
