// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// EXPORT-Z step 2, package EXZ2-P1a: the global carrier Iso_T over Z, the mod-q0 sparse echelon and the Dixon pairs
// (IsoTreeCarrierZ.h; docs/implementation/slot25/EXPORTZ2_design.md v2 Sec. 1.0, 1.1).

#include "IsoTreeCarrierZ.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <map>
#include <mutex>
#include <queue>
#include <sstream>

#include "DescentAssembly.h"
#include "DescentHecke.h"
#include "DescentLevelN.h"
#include "../IntegralCharpolyCrt.h"
#include "../Model/IsotropyModelTree.h"
#include "../Model/TreeCochains.h"
#include "../../Hecke/HeckeLevelN.h"
#include "../../../ModArithmetic/EngineCheck.h"
#include "../../../ModArithmetic/Fnv.h"

namespace integral_descent {
namespace freez {

using hecke::Mat22;
using integral_model::GKey;
using integral_model::QGroupRingElt;
using integral_model::Rat;
using clk = std::chrono::steady_clock;

namespace {
double since(clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); }

uint64_t PowMod(uint64_t a, uint64_t e, uint64_t m) {
    uint64_t r = 1 % m;
    a %= m;
    while (e) {
        if (e & 1) r = r * a % m;
        a = a * a % m;
        e >>= 1;
    }
    return r;
}
uint32_t InvModPrime(uint32_t a, uint32_t q) {
    ENGINE_CHECK_MSG(a % q != 0, "freez: inverse of 0 mod q");
    return static_cast<uint32_t>(PowMod(a, q - 2, q));
}
uint32_t RedI64(int64_t v, uint32_t q) {
    int64_t r = v % static_cast<int64_t>(q);
    if (r < 0) r += q;
    return static_cast<uint32_t>(r);
}
uint32_t RedRat(const Rat& r, uint32_t q) {
    const uint32_t a = RedI64(r.num, q);
    if (r.den == 1) return a;
    return static_cast<uint32_t>(uint64_t(a) * InvModPrime(RedI64(r.den, q), q) % q);
}
BigInt DivExactI64(const BigInt& x, int64_t a) {
    ENGINE_CHECK_MSG(a != 0, "freez: exact division by 0");
    const uint64_t ua = a < 0 ? uint64_t(-(a + 1)) + 1 : uint64_t(a);
    BigInt y = ua <= 0xffffffffull ? x.div_exact_u32(static_cast<uint32_t>(ua)) : x / BigInt::from_u64(ua);
    return a < 0 ? -y : y;
}
long double ToLd(const BigInt& x) {
    const auto& mg = x.magnitude();
    long double v = 0;
    const size_t L = mg.size();
    const size_t lo = L > 4 ? L - 4 : 0;
    for (size_t i = L; i-- > lo;) v = v * 4294967296.0L + mg[i];
    v = std::ldexp(v, static_cast<int>(32 * lo));
    return x.is_neg() ? -v : v;
}
}  // namespace

uint32_t BitsOf(const BigInt& x) { return static_cast<uint32_t>(x.bit_length()); }
long double ToLongDouble(const BigInt& x) { return ToLd(x); }

// ---------------------------------------------------------------------------------------------------------------------------------
// rho_n over Z
// ---------------------------------------------------------------------------------------------------------------------------------
std::vector<BigInt> RhoNZ(const Mat22& g, uint32_t n) {
    const int64_t a = g.a, b = g.b, c = g.c, d = g.d;
    ENGINE_CHECK_MSG(a != 0 || c != 0, "RhoNZ: the first column of g is zero");
    const uint32_t D = n + 1;
    std::vector<BigInt> M(size_t(D) * D);
    // f_0 = (a + c y)^n
    std::vector<BigInt> f(D, BigInt(0));
    f[0] = BigInt(1);
    const BigInt A(a), B(b), C(c), Dd(d);
    for (uint32_t e = 0; e < n; ++e) {
        for (uint32_t t = e + 1; t >= 1; --t) f[t] = f[t] * A + f[t - 1] * C;
        f[0] = f[0] * A;
    }
    for (uint32_t j = 0; j <= n; ++j) {
        for (uint32_t t = 0; t <= n; ++t) M[size_t(t) * D + j] = f[t];
        if (j == n) break;
        // h = f (b + d y), degree n + 1
        std::vector<BigInt> h(D + 1, BigInt(0));
        for (uint32_t t = 0; t <= n; ++t) {
            if (f[t].is_zero()) continue;
            h[t] += f[t] * B;
            h[t + 1] += f[t] * Dd;
        }
        // f = h / (a + c y), exact
        if (a != 0) {
            BigInt prev(0);
            for (uint32_t t = 0; t <= n; ++t) {
                BigInt num = h[t];
                if (t > 0 && c != 0) num -= C * prev;
                prev = DivExactI64(num, a);
                f[t] = prev;
            }
            ENGINE_CHECK_MSG(h[D] == C * f[n], "RhoNZ: the division by (a + c y) is not exact");
        } else {
            ENGINE_CHECK_MSG(h[0].is_zero(), "RhoNZ: the division by c y is not exact");
            for (uint32_t t = 0; t <= n; ++t) f[t] = DivExactI64(h[t + 1], c);
        }
    }
    return M;
}

std::vector<BigInt> RhoNAbsZ(const Mat22& g, uint32_t n) {
    Mat22 h = g;
    h.a = h.a < 0 ? -h.a : h.a;
    h.b = h.b < 0 ? -h.b : h.b;
    h.c = h.c < 0 ? -h.c : h.c;
    h.d = h.d < 0 ? -h.d : h.d;
    return RhoNZ(h, n);
}

// ---------------------------------------------------------------------------------------------------------------------------------
// sparse integer matrices, lanes, Garner, FNV
// ---------------------------------------------------------------------------------------------------------------------------------
uint32_t SparseZ::max_bits() const {
    uint32_t b = 0;
    for (const auto& v : val) b = std::max(b, BitsOf(v));
    return b;
}
uint32_t SparseZ::max_row_abs_bits() const {
    if (row_abs_bits_ >= 0) return static_cast<uint32_t>(row_abs_bits_);
    uint32_t best = 0;
    for (uint32_t i = 0; i < rows; ++i) {
        BigInt s(0);
        for (uint64_t k = rowptr[i]; k < rowptr[i + 1]; ++k) s += val[k].abs();
        best = std::max(best, BitsOf(s));
    }
    return best;
}
void SparseZ::seal() {
    row_abs_bits_ = -1;
    row_abs_bits_ = static_cast<int32_t>(max_row_abs_bits());
}
SparseZ SparseZ::transpose() const {
    SparseZ T;
    T.rows = cols;
    T.cols = rows;
    T.rowptr.assign(cols + 1, 0);
    for (uint32_t c : col) ++T.rowptr[c + 1];
    for (uint32_t i = 0; i < cols; ++i) T.rowptr[i + 1] += T.rowptr[i];
    T.col.resize(col.size());
    T.val.resize(val.size());
    std::vector<uint64_t> pos(T.rowptr.begin(), T.rowptr.end() - 1);
    for (uint32_t i = 0; i < rows; ++i)
        for (uint64_t k = rowptr[i]; k < rowptr[i + 1]; ++k) {
            const uint64_t p = pos[col[k]]++;
            T.col[p] = i;
            T.val[p] = val[k];
        }
    T.seal();
    return T;
}
std::vector<uint32_t> Residues(const SparseZ& A, uint32_t q) {
    std::vector<uint32_t> r(A.val.size());
    for (size_t k = 0; k < A.val.size(); ++k) r[k] = A.val[k].mod_u32(q);
    return r;
}

std::vector<uint32_t> CachedCrtPrimes(uint32_t count) {
    static std::mutex mu;
    static std::vector<uint32_t> cache;
    std::lock_guard<std::mutex> lk(mu);
    if (cache.size() < count) cache = integral::CrtPrimes(std::max<uint32_t>(count, std::max<uint32_t>(256, 2 * uint32_t(cache.size()))));
    return std::vector<uint32_t>(cache.begin(), cache.begin() + count);
}

std::vector<uint32_t> LanePrimes(uint64_t bits, uint32_t skip, uint32_t N) {
    std::vector<uint32_t> out;
    uint64_t have = 0;
    uint32_t batch = 64;
    while (have < bits) {
        const std::vector<uint32_t> cand = CachedCrtPrimes(batch);
        out.clear();
        have = 0;
        for (uint32_t q : cand) {
            if (q == skip || (6ull * N) % q == 0) continue;
            out.push_back(q);
            have += 29;   // bitlen(q) - 1 = 29 for every q in (2^29, 2^30)
            if (have >= bits) break;
        }
        batch *= 2;
        ENGINE_CHECK_MSG(batch < (1u << 22), "LanePrimes: too many lanes requested");
    }
    return out;
}

GarnerCtx::GarnerCtx(const std::vector<uint32_t>& p) : primes(p) {
    const size_t L = p.size();
    pre.assign(L * L, 0);
    inv.assign(L, 0);
    P = BigInt(1);
    for (size_t k = 0; k < L; ++k) {
        uint64_t pw = 1;
        for (size_t i = 0; i < k; ++i) {
            pre[k * L + i] = static_cast<uint32_t>(pw);
            pw = pw * (p[i] % p[k]) % p[k];
        }
        inv[k] = InvModPrime(static_cast<uint32_t>(pw), p[k]);
        P = P * BigInt(int64_t(p[k]));
    }
}

BigInt GarnerSigned(const std::vector<uint32_t>& res, const std::vector<uint32_t>& primes) { return GarnerSigned(res, GarnerCtx(primes)); }

BigInt GarnerSigned(const std::vector<uint32_t>& res, const GarnerCtx& G) {
    const std::vector<uint32_t>& primes = G.primes;
    const size_t L = primes.size();
    ENGINE_CHECK_MSG(res.size() == L && L > 0, "GarnerSigned: shapes");
    bool allzero = true;
    for (uint32_t x : res)
        if (x) {
            allzero = false;
            break;
        }
    if (allzero) return BigInt(0);
    // mixed radix digits: d_k = (res_k - sum_{i<k} d_i prod_{j<i} q_j) / prod_{j<k} q_j  mod q_k
    std::vector<uint64_t> dg(L);
    for (size_t k = 0; k < L; ++k) {
        const uint64_t qk = primes[k];
        uint64_t acc = 0;
        for (size_t i = 0; i < k; ++i) acc = (acc + dg[i] % qk * G.pre[k * L + i]) % qk;
        const uint64_t diff = (res[k] % qk + qk - acc) % qk;
        dg[k] = diff * G.inv[k] % qk;
    }
    // sign: c^+ > (P - 1)/2 iff the digits exceed those of (P-1)/2 from the top
    bool neg = false;
    {
        // compare digits with those of (P-1)/2: (P-1)/2 = sum_k h_k prod_{j<k} q_j with h_k = (q_k - 1)/2 for all k (q odd)
        for (size_t k = L; k-- > 0;) {
            const uint64_t h = (primes[k] - 1) / 2;
            if (dg[k] > h) { neg = true; break; }
            if (dg[k] < h) { neg = false; break; }
        }
    }
    BigInt x(0);
    for (size_t k = L; k-- > 0;) {
        x = x * BigInt(int64_t(primes[k]));
        x += BigInt(int64_t(dg[k]));
    }
    if (neg) x -= G.P;
    return x;
}

uint64_t MatrixFnv(const std::vector<std::vector<BigInt>>& cols, uint32_t rows) {
    uint64_t h = engine::kFnvOffset64;
    for (uint32_t i = 0; i < rows; ++i) {
        std::string line;
        if (i) line += ";";
        for (size_t j = 0; j < cols.size(); ++j) {
            if (j) line += ",";
            line += cols[j][i].to_string();
        }
        h = engine::FnvBytes(h, line);
    }
    return h;
}

// =================================================================================================================================
// The carrier
// =================================================================================================================================
struct IsoTreeCarrierZ::Impl {
    const integral_model::IntegralTreeComparison* cmp = nullptr;
    std::shared_ptr<const hecke::level::Gamma0LevelN> A;
    // exact block-monomial action: per g, per coset j (0-based): column block col[j], rho_n(h_j) (index into rho memo)
    struct MonoZ {
        std::vector<uint32_t> col;
        std::vector<const std::vector<BigInt>*> blk;
    };
    std::mutex mu;
    std::map<GKey, std::vector<BigInt>> rho_memo;         // rho_n(h) exact
    std::map<GKey, MonoZ> mono_memo;
    std::map<GKey, std::vector<long double>> abs_memo;    // rho_n(|E|) as long double (majorant)
    // (S25 EXZ2-P2 fix round 1, audit F9) lane modules are SHARED: the map holds one reference, every user holds its own for the duration
    // of a call, so release_lane(q) never frees a module in use.  The exact memos (mono_memo, rho_memo, abs_memo) are pointed into
    // outside the lock by apply_hecke_majorant: release_exact_memo() clears them only when no majorant pass runs (else it defers the
    // clear to the last pass's end).
    std::map<uint32_t, std::shared_ptr<const integral_model::TreeModule<uint32_t>>> lane_mod;
    uint32_t majorant_users = 0;
    bool exact_clear_pending = false;
    void clear_exact_locked() {
        mono_memo.clear();
        rho_memo.clear();
        abs_memo.clear();
        exact_clear_pending = false;
    }
    const MonoZ& mono(const Mat22& g, uint32_t n) {
        const GKey k = integral_model::key_of(g);
        auto it = mono_memo.find(k);
        if (it != mono_memo.end()) return it->second;
        MonoZ M;
        const uint32_t m = A->m();
        M.col.resize(m);
        M.blk.resize(m);
        for (uint32_t j = 1; j <= m; ++j) {
            const Mat22 x = A->rep(j) * g;
            const uint32_t jj = A->class_of(x);
            const Mat22 h = x * A->rep(jj).inv_unimodular();
            ENGINE_CHECK_MSG(A->contains(h), "IsoTreeCarrierZ: h_j(g) not in Gamma_0(N)");
            M.col[j - 1] = jj - 1;
            const GKey hk = integral_model::key_of(h);
            auto r = rho_memo.find(hk);
            if (r == rho_memo.end()) r = rho_memo.emplace(hk, RhoNZ(h, n)).first;
            M.blk[j - 1] = &r->second;
        }
        return mono_memo.emplace(k, std::move(M)).first->second;
    }
    std::shared_ptr<const integral_model::TreeModule<uint32_t>> lane(uint32_t q, uint32_t N, uint32_t n) {
        std::lock_guard<std::mutex> lk(mu);
        auto& e = lane_mod[q];
        if (!e) e = std::make_shared<const integral_model::TreeModule<uint32_t>>(integral_model::TreeInducedSymModule(N, n, engine::ModCtx<uint32_t>(q)));
        return e;
    }
};

IsoTreeCarrierZ::~IsoTreeCarrierZ() = default;

uint32_t IsoTreeCarrierZ::cells(uint32_t q) const { return impl_->cmp->tree().rank(q); }
uint32_t IsoTreeCarrierZ::iso_dim(uint32_t deg) const { return static_cast<uint32_t>(impl_->cmp->iso().cells(deg).size()) * D_; }

IsoTreeCarrierZ::IsoTreeCarrierZ(uint32_t N, uint32_t n) : N_(N), n_(n), impl_(std::make_unique<Impl>()) {
    const auto t0 = clk::now();
    ENGINE_CHECK_MSG(N >= 1, "IsoTreeCarrierZ: N >= 1");
    Impl& I = *impl_;
    I.cmp = &hecke3::SharedTreeComparison();
    const integral_model::IsoTreeCertificates cert = I.cmp->certify();
    ENGINE_CHECK_MSG(cert.all() && cert.integral, "IsoTreeCarrierZ: the integral comparison of prop:isomodeltree(4) is not certified");
    const integral_model::TreeModelChain& T = I.cmp->tree();
    ENGINE_CHECK_MSG(T.ell() == 0 && T.q_top() >= 3, "IsoTreeCarrierZ: the comparison's tree is not the free model to degree 3");
    I.A = std::make_shared<const hecke::level::Gamma0LevelN>(N);
    m_ = I.A->m();
    const uint32_t d = n + 1;
    D_ = m_ * d;
    const LevelNCounts cnt = LevelNClosedForm(N, n);
    b0_ = n % 2 ? 0u : cnt.b0;
    b1_ = n % 2 ? 0u : cnt.b1;
    // the exact coboundaries d^0, d^1
    for (uint32_t q = 0; q < 2; ++q) {
        const uint32_t rows = T.rank(q + 1) * D_, cols = T.rank(q) * D_;
        std::map<std::pair<uint32_t, uint32_t>, BigInt> acc;
        for (const integral_model::LEntry& e : T.differential(q + 1)) {
            for (const auto& [gk, r] : e.coeff.terms) {
                ENGINE_CHECK_MSG(r.den == 1, "IsoTreeCarrierZ: a non-integral coefficient in the free tree's differential");
                if (r.num == 0) continue;
                const Impl::MonoZ& M = I.mono(integral_model::mat_of(gk), n);
                const BigInt coef(r.num);
                for (uint32_t j = 0; j < m_; ++j) {
                    const std::vector<BigInt>& blk = *M.blk[j];
                    const uint32_t r0 = e.col * D_ + j * d, c0 = e.row * D_ + M.col[j] * d;
                    for (uint32_t x = 0; x < d; ++x)
                        for (uint32_t y = 0; y < d; ++y) {
                            const BigInt& v = blk[size_t(x) * d + y];
                            if (v.is_zero()) continue;
                            BigInt& slot = acc[{r0 + x, c0 + y}];
                            slot += coef * v;
                        }
                }
            }
        }
        SparseZ& S = q == 0 ? d0_ : d1_;
        S.rows = rows;
        S.cols = cols;
        S.rowptr.assign(rows + 1, 0);
        for (auto& [key, v] : acc) {
            if (v.is_zero()) continue;
            ++S.rowptr[key.first + 1];
            S.col.push_back(key.second);
            S.val.push_back(std::move(v));
        }
        for (uint32_t i = 0; i < rows; ++i) S.rowptr[i + 1] += S.rowptr[i];
        S.seal();
    }
    build_seconds = since(t0);
}

void IsoTreeCarrierZ::release_lane(uint32_t q) const {
    std::lock_guard<std::mutex> lk(impl_->mu);
    impl_->lane_mod.erase(q);
}
void IsoTreeCarrierZ::release_exact_memo() const {
    std::lock_guard<std::mutex> lk(impl_->mu);
    if (impl_->majorant_users) impl_->exact_clear_pending = true;   // (F9) a majorant pass holds pointers into the memos
    else impl_->clear_exact_locked();
}
size_t IsoTreeCarrierZ::lane_count() const {
    std::lock_guard<std::mutex> lk(impl_->mu);
    return impl_->lane_mod.size();
}

namespace {
// out[block j] += coef * rho(g) in (block-monomial, mod q, k columns row-major)
void AddRhoMod(uint32_t q, const integral_model::BlockMonomial<uint32_t>& B, uint32_t coef, const uint32_t* in, uint32_t* out, uint32_t k) {
    const uint32_t d = B.d;
    for (uint32_t j = 0; j < B.m; ++j) {
        const std::vector<uint32_t>& blk = B.blocks[B.blk[j]];
        const uint32_t* src = in + size_t(B.col[j]) * d * k;
        uint32_t* dst = out + size_t(j) * d * k;
        for (uint32_t x = 0; x < d; ++x)
            for (uint32_t y = 0; y < d; ++y) {
                const uint64_t f = uint64_t(coef) * blk[size_t(x) * d + y] % q;
                if (!f) continue;
                uint32_t* yr = dst + size_t(x) * k;
                const uint32_t* xr = src + size_t(y) * k;
                for (uint32_t c = 0; c < k; ++c)
                    if (xr[c]) yr[c] = static_cast<uint32_t>((yr[c] + f * xr[c]) % q);
            }
    }
}
void AddQRhoMod(uint32_t q, const integral_model::TreeModule<uint32_t>& TM, const QGroupRingElt& a, const uint32_t* in, uint32_t* out,
                uint32_t k) {
    for (const auto& [g, r] : a.terms) {
        const uint32_t coef = RedRat(r, q);
        if (!coef) continue;
        AddRhoMod(q, TM.rho(integral_model::mat_of(g)), coef, in, out, k);
    }
}
}  // namespace

void IsoTreeCarrierZ::phi_star_mod(uint32_t q, uint32_t deg, const std::vector<uint32_t>& V, uint32_t k, std::vector<uint32_t>& W) const {
    const auto TMref = impl_->lane(q, N_, n_);   // (F9) this call's own reference
    const auto& TM = *TMref;
    ENGINE_CHECK_MSG(V.size() == size_t(dim(deg)) * k, "IsoTreeCarrierZ::phi_star_mod: shape");
    W.assign(size_t(iso_dim(deg)) * k, 0);
    const auto& rows = impl_->cmp->phi().value.at(deg);   // Iso cell y -> chain on the tree
    for (uint32_t y = 0; y < rows.size(); ++y)
        for (const auto& [x, b] : rows[y]) AddQRhoMod(q, TM, b, V.data() + size_t(x) * D_ * k, W.data() + size_t(y) * D_ * k, k);
}

void IsoTreeCarrierZ::psi_star_mod(uint32_t q, uint32_t deg, const std::vector<uint32_t>& W, uint32_t k, std::vector<uint32_t>& V) const {
    const auto TMref = impl_->lane(q, N_, n_);   // (F9) this call's own reference
    const auto& TM = *TMref;
    ENGINE_CHECK_MSG(W.size() == size_t(iso_dim(deg)) * k, "IsoTreeCarrierZ::psi_star_mod: shape");
    V.assign(size_t(dim(deg)) * k, 0);
    const auto& rows = impl_->cmp->psi().value.at(deg);   // tree cell x -> chain on Iso
    for (uint32_t x = 0; x < rows.size(); ++x)
        for (const auto& [y, b] : rows[x]) AddQRhoMod(q, TM, b, W.data() + size_t(y) * D_ * k, V.data() + size_t(x) * D_ * k, k);
}

void IsoTreeCarrierZ::apply_hecke_mod(const hecke3::IsoHeckeN& T, uint32_t deg, const std::vector<uint32_t>& X, uint32_t k,
                                      std::vector<uint32_t>& Y) const {
    ENGINE_CHECK_MSG(!T.refused() && T.N() == N_ && T.n() == n_ && T.q_top() >= deg, "IsoTreeCarrierZ::apply_hecke_mod: the operator");
    std::vector<uint32_t> W, W2;
    const uint64_t q = T.modulus();   // the operator's own modulus is the lane
    ENGINE_CHECK_MSG(q < (1ull << 31) && (6ull * N_) % q != 0, "IsoTreeCarrierZ::apply_hecke_mod: the lane is not a word prime prime to 6N");
    phi_star_mod(static_cast<uint32_t>(q), deg, X, k, W);
    T.apply(deg, W, k, W2);
    psi_star_mod(static_cast<uint32_t>(q), deg, W2, k, Y);
}

void IsoTreeCarrierZ::d_mod(uint32_t q, uint32_t deg, const std::vector<uint32_t>& X, uint32_t k, std::vector<uint32_t>& Y) const {
    const auto TMref = impl_->lane(q, N_, n_);   // (F9) this call's own reference
    const auto& TM = *TMref;
    const integral_model::TreeModelChain& T = impl_->cmp->tree();
    ENGINE_CHECK_MSG(deg + 1 <= T.q_top() && X.size() == size_t(dim(deg)) * k, "IsoTreeCarrierZ::d_mod: shape");
    Y.assign(size_t(T.rank(deg + 1)) * D_ * k, 0);
    for (const integral_model::LEntry& e : T.differential(deg + 1))
        AddQRhoMod(q, TM, e.coeff, X.data() + size_t(e.row) * D_ * k, Y.data() + size_t(e.col) * D_ * k, k);
}

void IsoTreeCarrierZ::apply_hecke_majorant(const hecke3::IsoHeckeN& T, const std::vector<long double>& X, uint32_t k,
                                           std::vector<long double>& Y) const {
    Impl& I = *impl_;
    const uint32_t d = n_ + 1;
    ENGINE_CHECK_MSG(X.size() == size_t(dim(1)) * k, "apply_hecke_majorant: shape");
    // (F9) hold the exact memos for the pass: release_exact_memo() defers its clear while a pass runs
    struct MajorantUse {
        Impl& I;
        explicit MajorantUse(Impl& i) : I(i) {
            std::lock_guard<std::mutex> lk(I.mu);
            ++I.majorant_users;
        }
        ~MajorantUse() {
            std::lock_guard<std::mutex> lk(I.mu);
            if (--I.majorant_users == 0 && I.exact_clear_pending) I.clear_exact_locked();
        }
    } use_guard(I);
    // |rho(g)| of the transport maps: the exact block-monomial blocks, absolute values
    auto add_abs_rho = [&](const QGroupRingElt& a, const long double* in, long double* out) {
        for (const auto& [g, r] : a.terms) {
            const long double coef = std::fabs(static_cast<long double>(r.num) / static_cast<long double>(r.den));
            if (coef == 0) continue;
            const Impl::MonoZ* M;
            {
                std::lock_guard<std::mutex> lk(I.mu);
                M = &I.mono(integral_model::mat_of(g), n_);
            }
            for (uint32_t j = 0; j < m_; ++j) {
                const std::vector<BigInt>& blk = *M->blk[j];
                const long double* src = in + size_t(M->col[j]) * d * k;
                long double* dst = out + size_t(j) * d * k;
                for (uint32_t x = 0; x < d; ++x)
                    for (uint32_t y = 0; y < d; ++y) {
                        const BigInt& v = blk[size_t(x) * d + y];
                        if (v.is_zero()) continue;
                        const long double f = coef * std::fabs(ToLd(v));
                        for (uint32_t c = 0; c < k; ++c) dst[size_t(x) * k + c] += f * src[size_t(y) * k + c];
                    }
            }
        }
    };
    // |Phi^*| X
    std::vector<long double> W(size_t(iso_dim(1)) * k, 0), W2(size_t(iso_dim(1)) * k, 0);
    const auto& prow = I.cmp->phi().value.at(1);
    for (uint32_t y = 0; y < prow.size(); ++y)
        for (const auto& [x, b] : prow[y]) add_abs_rho(b, X.data() + size_t(x) * D_ * k, W.data() + size_t(y) * D_ * k);
    // sum |mul| rho_n(|E|)
    for (const auto& t : T.terms_Z(1)) {
        if (t.mul == 0) continue;
        const GKey ek = integral_model::key_of(t.E);
        const std::vector<long double>* R;
        {
            std::lock_guard<std::mutex> lk(I.mu);
            auto it = I.abs_memo.find(ek);
            if (it == I.abs_memo.end()) {
                const std::vector<BigInt> Z = RhoNAbsZ(t.E, n_);
                std::vector<long double> v(Z.size());
                for (size_t i = 0; i < Z.size(); ++i) v[i] = ToLd(Z[i]);
                it = I.abs_memo.emplace(ek, std::move(v)).first;
            }
            R = &it->second;
        }
        const long double mul = std::fabs(static_cast<long double>(t.mul));
        for (uint32_t x = 0; x < d; ++x)
            for (uint32_t y = 0; y < d; ++y) {
                const long double f = mul * (*R)[size_t(x) * d + y];
                if (f == 0) continue;
                long double* yr = W2.data() + (size_t(t.rb) * d + x) * k;
                const long double* xr = W.data() + (size_t(t.cb) * d + y) * k;
                for (uint32_t c = 0; c < k; ++c) yr[c] += f * xr[c];
            }
    }
    // |Psi^*|
    Y.assign(size_t(dim(1)) * k, 0);
    const auto& srow = I.cmp->psi().value.at(1);
    for (uint32_t x = 0; x < srow.size(); ++x)
        for (const auto& [y, b] : srow[x]) add_abs_rho(b, W2.data() + size_t(y) * D_ * k, Y.data() + size_t(x) * D_ * k);
}

// =================================================================================================================================
// EchelonModQ
// =================================================================================================================================
EchelonModQ::EchelonModQ(uint32_t q, uint32_t length) : q_(q), L_(length), pivot_of_(length, -1), acc_(length, 0), mark_(length, 0) {}

bool EchelonModQ::reduce(std::vector<uint32_t>& a, std::vector<uint32_t>& touched,
                         std::vector<std::pair<uint32_t, uint32_t>>* coef, std::vector<uint8_t>& mark_) const {
    // a: dense accumulator (length L), touched: its non-zero support (unsorted, may hold zeros); ascending sweep through a min-heap
    // every coordinate ever pushed is marked (mark_ = 1) and listed in `marked`; all are cleared at the end
    std::priority_queue<uint32_t, std::vector<uint32_t>, std::greater<uint32_t>> heap;
    std::vector<uint32_t> marked;
    for (uint32_t i : touched) {
        heap.push(i);
        mark_[i] = 1;
        marked.push_back(i);
    }
    std::vector<uint32_t> out;
    uint32_t last = UINT32_MAX;
    while (!heap.empty()) {
        const uint32_t c = heap.top();
        heap.pop();
        if (c == last) continue;
        last = c;
        if (!a[c]) continue;
        const int32_t k = pivot_of_[c];
        if (k < 0) {
            out.push_back(c);
            continue;
        }
        const uint64_t alpha = uint64_t(a[c]) * lead_inv_[k] % q_;
        if (coef) coef->push_back({static_cast<uint32_t>(k), static_cast<uint32_t>(alpha)});
        const SpVec& w = w_[k];
        const uint64_t neg = q_ - alpha;
        for (size_t t = 0; t < w.idx.size(); ++t) {
            const uint32_t i = w.idx[t];
            const uint32_t before = a[i];
            a[i] = static_cast<uint32_t>((a[i] + neg * w.val[t]) % q_);
            if (i > c && !before && !mark_[i]) {
                mark_[i] = 1;
                marked.push_back(i);
                heap.push(i);
            }
        }
    }
    touched.swap(out);
    for (uint32_t i : marked) mark_[i] = 0;
    return !touched.empty();
}

bool EchelonModQ::insert(const SpVec& v) {
    std::vector<uint32_t> touched;
    for (size_t t = 0; t < v.idx.size(); ++t) {
        const uint32_t x = v.val[t] % q_;
        if (!x) continue;
        acc_[v.idx[t]] = x;
        mark_[v.idx[t]] = 1;
        touched.push_back(v.idx[t]);
    }
    std::vector<std::pair<uint32_t, uint32_t>> coef;
    const bool indep = reduce(acc_, touched, &coef, mark_);
    // touched now holds the surviving support (ascending); clear the accumulator
    if (!indep) {
        for (uint32_t i : touched) acc_[i] = 0;
        // the accumulator may hold zeros only: nothing else to clear
        return false;
    }
    SpVec w;
    std::sort(touched.begin(), touched.end());
    for (uint32_t i : touched) {
        if (acc_[i]) {
            w.idx.push_back(i);
            w.val.push_back(acc_[i]);
        }
        acc_[i] = 0;
    }
    const uint32_t k = static_cast<uint32_t>(lead_.size());
    lead_.push_back(w.idx.front());
    lead_inv_.push_back(InvModPrime(w.val.front(), q_));
    pivot_of_[w.idx.front()] = static_cast<int32_t>(k);
    w_.push_back(std::move(w));
    tcol_.push_back(std::move(coef));
    return true;
}

bool EchelonModQ::solve(const std::vector<uint32_t>& b, std::vector<uint32_t>& x) const {
    ENGINE_CHECK_MSG(b.size() == L_, "EchelonModQ::solve: length");
    std::vector<uint32_t> a(b);
    std::vector<uint32_t> touched;
    std::vector<uint8_t> mark(L_, 0);   // (S25 EXZ2-P2, F11) the call's own marks: solve() writes nothing shared
    for (uint32_t i = 0; i < L_; ++i)
        if (a[i]) {
            touched.push_back(i);
            mark[i] = 1;
        }
    std::vector<std::pair<uint32_t, uint32_t>> coef;
    const bool rest = reduce(a, touched, &coef, mark);
    if (rest) return false;
    const uint32_t K = rank();
    std::vector<uint32_t> y(K, 0);
    for (const auto& [k, al] : coef) y[k] = static_cast<uint32_t>((y[k] + al) % q_);
    // M = W T with T unit upper triangular, column k: T_{k k} = 1, T_{k' k} = tcol_[k] entries; solve T x = y backwards
    x.assign(K, 0);
    for (uint32_t k = K; k-- > 0;) {
        const uint32_t xk = y[k];
        x[k] = xk;
        if (!xk) continue;
        for (const auto& [kp, t] : tcol_[k]) y[kp] = static_cast<uint32_t>((y[kp] + uint64_t(q_ - t) * xk) % q_);
    }
    return true;
}

void EchelonModQ::solve_transposed(const std::vector<uint32_t>& u, std::vector<uint32_t>& z) const {
    const uint32_t K = rank();
    ENGINE_CHECK_MSG(u.size() == K, "EchelonModQ::solve_transposed: length");
    // V = W T, T unit upper triangular (T_{k' k} = tcol_[k]); V^T z = u  <=>  W^T z = u' with T^T u' = u (forward, k ascending)
    std::vector<uint32_t> up(u);
    for (uint32_t k = 0; k < K; ++k) {
        uint64_t s = up[k];
        for (const auto& [kp, t] : tcol_[k]) s = (s + uint64_t(q_ - t) * up[kp]) % q_;
        up[k] = static_cast<uint32_t>(s);
    }
    // W^T z = u': w_k has its lead at lead_[k] and every other entry at a larger index; z = 0 off the leads, so process the pivots in
    // descending lead order (a larger index is a lead fixed earlier or a zero).  (Fix round 1, audit F8 e) the order is read off
    // pivot_of_ (lead -> k) by one descending scan: no sort, no per-call order vector; the same order, so the same z.
    z.assign(L_, 0);
    for (uint32_t lead = L_; lead-- > 0;) {
        if (pivot_of_[lead] < 0) continue;
        const uint32_t k = static_cast<uint32_t>(pivot_of_[lead]);
        const SpVec& w = w_[k];
        uint64_t s = up[k];
        for (size_t t = 1; t < w.idx.size(); ++t) {
            const uint32_t zi = z[w.idx[t]];
            if (zi) s = (s + uint64_t(q_ - w.val[t]) * zi) % q_;
        }
        z[lead_[k]] = static_cast<uint32_t>(s * lead_inv_[k] % q_);
    }
}

uint64_t EchelonModQ::fill() const {
    uint64_t s = 0;
    for (const auto& w : w_) s += w.idx.size();
    return s;
}
uint64_t EchelonModQ::t_nnz() const {
    uint64_t s = 0;
    for (const auto& t : tcol_) s += t.size();
    return s;
}
uint64_t EchelonModQ::bytes() const { return fill() * 8 + t_nnz() * 8 + uint64_t(L_) * 9; }

// =================================================================================================================================
// Dixon lifting
// =================================================================================================================================
namespace {

BigInt Isqrt(const BigInt& m) {
    if (m.sign() <= 0) return BigInt(0);
    // Newton from above: x0 = 2^{ceil(bits/2)}
    BigInt x = BigInt::pow(BigInt(2), (m.bit_length() + 1) / 2 + 1);
    while (true) {
        BigInt y = (x + m / x) / BigInt(2);
        if (y >= x) break;
        x = y;
    }
    while (x * x > m) x -= BigInt(1);
    while ((x + BigInt(1)) * (x + BigInt(1)) <= m) x += BigInt(1);
    return x;
}

// the model's ratrec(a, m): (denominator t1 > 0) or false
bool RatRecDen(const BigInt& a, const BigInt& m, const BigInt& Bnd, BigInt& den) {
    BigInt r0 = m, r1 = a.floor_mod(m), t0(0), t1(1);
    while (r1 > Bnd) {
        BigInt qq = r0 / r1;
        BigInt r2 = r0 - qq * r1;
        r0 = std::move(r1);
        r1 = std::move(r2);
        BigInt t2 = t0 - qq * t1;
        t0 = std::move(t1);
        t1 = std::move(t2);
    }
    if (t1.is_zero() || t1.abs() > Bnd) return false;
    if (t1.is_neg()) {
        t1 = -t1;
        r1 = -r1;
    }
    if (BigInt::gcd(r1, t1) != BigInt(1)) return false;
    den = t1;
    return true;
}

// the model's vec_ratrec(col, m): common denominator + numerators (symmetric residues of a * den mod m)
bool VecRatRec(const std::vector<BigInt>& col, const BigInt& m, std::vector<BigInt>& nums, BigInt& den) {
    const BigInt Bnd = Isqrt(m / BigInt(2));
    const BigInt half = m / BigInt(2);
    den = BigInt(1);
    for (const BigInt& a : col) {
        BigInt t;
        // shortcut, exactly equivalent to the model's ratrec: when the symmetric residue x of a * den has |x| <= Bnd, (x, 1) is a
        // reconstruction within the bounds, hence THE reconstruction (unique for m > 2 Bnd^2): the extended Euclid returns t1 = 1
        BigInt x = (a * den).floor_mod(m);
        if (x > half) x -= m;
        if (x.abs() <= Bnd) continue;
        if (!RatRecDen((a * den).floor_mod(m), m, Bnd, t)) return false;
        den = den * t;
        if (den * den > m) return false;
    }
    nums.resize(col.size());
    for (size_t i = 0; i < col.size(); ++i) {
        BigInt v = (col[i] * den).floor_mod(m);
        if (v > half) v -= m;
        nums[i] = std::move(v);
    }
    return true;
}

// A sparse integer vector of length L (exact) and its lane residues.
struct SpZ {
    std::vector<uint32_t> idx;
    std::vector<BigInt> val;
};

// A x = 0 exactly, by lanes whose product exceeds 2 |A| |x| entrywise (bits rounded up): A sparse (CSR, exact), x dense.
bool LaneZero(const SparseZ& A, const std::vector<BigInt>& x, uint32_t q0, uint32_t N, uint32_t* lanes_used) {
    uint32_t bx = 0;
    std::vector<uint32_t> supp;
    for (uint32_t i = 0; i < x.size(); ++i)
        if (!x[i].is_zero()) {
            supp.push_back(i);
            bx = std::max(bx, BitsOf(x[i]));
        }
    if (supp.empty()) return true;
    const uint64_t bits = uint64_t(A.max_row_abs_bits()) + bx + 2;
    const std::vector<uint32_t> L = LanePrimes(bits, q0, N);
    if (lanes_used) *lanes_used = static_cast<uint32_t>(L.size());
    for (uint32_t q : L) {
        std::vector<uint32_t> xr(x.size(), 0);
        for (uint32_t i : supp) xr[i] = x[i].mod_u32(q);
        for (uint32_t i = 0; i < A.rows; ++i) {
            uint64_t s = 0;
            for (uint64_t k = A.rowptr[i]; k < A.rowptr[i + 1]; ++k) {
                const uint32_t xv = xr[A.col[k]];
                if (!xv) continue;
                s = (s + uint64_t(A.val[k].mod_u32(q)) * xv) % q;
            }
            if (s) return false;
        }
    }
    return true;
}

// Dixon: solve sum_k x_k v_k = b_j over Q for the independent system vectors V (length L, the pivots of E in order) and the RHS list.
// Output per RHS: numerators (k of them) and the common denominator.  verify(j, nums, den) is the exact check.  EXZ2-P1b: the RHS are
// solved in batches of `batch` columns (audit F5: the state is batch x L, not r x L), and the stop is output-sensitive per column (audit
// F6): from step 2 on, every active column tries its vector rational reconstruction after every step and leaves the batch once it
// verifies.  The result does not depend on the schedule (the solution supported on the independent set + {P_j} is unique, and den is
// the lcm of its reduced denominators), only the step counts do.
struct DixonOut {
    std::vector<std::vector<BigInt>> nums;
    std::vector<BigInt> den;
    DixonStats st;
};
DixonOut Dixon(const std::vector<SpZ>& V, const EchelonModQ& E, const std::vector<SpZ>& rhs, uint32_t L, uint32_t N,
               const std::function<bool(uint32_t, const std::vector<BigInt>&, const BigInt&)>& verify, uint32_t batch) {
    const auto t0 = clk::now();
    const uint32_t q0 = E.q();
    const uint32_t k = static_cast<uint32_t>(V.size()), m = static_cast<uint32_t>(rhs.size());
    ENGINE_CHECK_MSG(E.rank() == k, "Dixon: the echelon's rank != the system size");
    DixonOut out;
    out.nums.assign(m, {});
    out.den.assign(m, BigInt(1));
    if (m == 0) return out;
    if (k == 0) {
        for (uint32_t j = 0; j < m; ++j) ENGINE_CHECK_MSG(verify(j, {}, BigInt(1)), "Dixon: an empty system with a failing check");
        out.st.seconds = since(t0);
        return out;
    }
    if (batch == 0) batch = m;
    // the row-wise view of M (L x k) for the lane products: per row the (system vector, entry) pairs
    std::vector<std::vector<std::pair<uint32_t, const BigInt*>>> rowsM(L);
    for (uint32_t c = 0; c < k; ++c)
        for (size_t t = 0; t < V[c].idx.size(); ++t) rowsM[V[c].idx[t]].push_back({c, &V[c].val[t]});
    uint32_t brow = 0;
    for (uint32_t i = 0; i < L; ++i) {
        BigInt s(0);
        for (const auto& pr : rowsM[i]) s += pr.second->abs();
        brow = std::max(brow, BitsOf(s));
    }
    const std::vector<uint32_t> lanes = LanePrimes(uint64_t(brow) + 30 + 2, q0, N);
    const GarnerCtx gctx(lanes);
    out.st.lanes = static_cast<uint32_t>(lanes.size());
    std::vector<std::vector<uint32_t>> Mres(lanes.size());
    for (size_t l = 0; l < lanes.size(); ++l) {
        auto& r = Mres[l];
        for (uint32_t i = 0; i < L; ++i)
            for (const auto& pr : rowsM[i]) r.push_back(pr.second->mod_u32(lanes[l]));
    }
    const BigInt Q0{int64_t(q0)};
    std::vector<uint32_t> rq(L), xi, res(lanes.size());
    std::vector<std::vector<uint32_t>> xr(lanes.size(), std::vector<uint32_t>(k));
    for (uint32_t b0 = 0; b0 < m; b0 += batch) {
        ++out.st.batches;
        const uint32_t b1 = std::min(m, b0 + batch);
        std::vector<uint32_t> active;
        std::vector<std::vector<BigInt>> R(b1 - b0), X(b1 - b0);
        for (uint32_t j = b0; j < b1; ++j) {
            active.push_back(j);
            R[j - b0].assign(L, BigInt(0));
            for (size_t t = 0; t < rhs[j].idx.size(); ++t) R[j - b0][rhs[j].idx[t]] = rhs[j].val[t];
            X[j - b0].assign(k, BigInt(0));
        }
        BigInt qpow(1);
        uint32_t steps = 0;
        while (!active.empty()) {
            for (uint32_t j : active) {
                std::vector<BigInt>& Rj = R[j - b0];
                std::vector<BigInt>& Xj = X[j - b0];
                for (uint32_t i = 0; i < L; ++i) rq[i] = Rj[i].is_zero() ? 0u : Rj[i].mod_u32(q0);
                ENGINE_CHECK_MSG(E.solve(rq, xi), "Dixon: the residual is not in the span mod q0");
                for (uint32_t c = 0; c < k; ++c)
                    if (xi[c]) Xj[c] += qpow * BigInt(int64_t(xi[c]));
                for (size_t l = 0; l < lanes.size(); ++l)
                    for (uint32_t c = 0; c < k; ++c) xr[l][c] = xi[c] % lanes[l];
                size_t pos = 0;
                for (uint32_t i = 0; i < L; ++i) {
                    const size_t nr = rowsM[i].size();
                    bool any = false;
                    for (size_t l = 0; l < lanes.size(); ++l) {
                        const uint64_t q = lanes[l];
                        uint64_t s = 0;
                        const uint32_t* mr = Mres[l].data() + pos;
                        for (size_t t = 0; t < nr; ++t) {
                            const uint32_t xv = xr[l][rowsM[i][t].first];
                            if (xv) s = (s + uint64_t(mr[t]) * xv) % q;
                        }
                        res[l] = static_cast<uint32_t>(s);
                        any = any || s;
                    }
                    pos += nr;
                    BigInt& Ri = Rj[i];
                    if (any) Ri -= GarnerSigned(res, gctx);
                    if (!Ri.is_zero()) Ri = Ri.div_exact_u32(q0);
                }
            }
            qpow = qpow * Q0;
            ++steps;
            ENGINE_CHECK_MSG(steps < 100000, "Dixon: no reconstruction after 10^5 steps");
            if (steps < 2) continue;
            std::vector<uint32_t> still;
            for (uint32_t j : active) {
                ++out.st.attempts;
                std::vector<BigInt> nums;
                BigInt den;
                if (VecRatRec(X[j - b0], qpow, nums, den) && verify(j, nums, den)) {
                    out.nums[j] = std::move(nums);
                    out.den[j] = std::move(den);
                    out.st.steps = std::max(out.st.steps, steps);
                    out.st.col_steps += steps;
                    std::vector<BigInt>().swap(R[j - b0]);
                    std::vector<BigInt>().swap(X[j - b0]);
                } else {
                    still.push_back(j);
                }
            }
            active.swap(still);
        }
    }
    out.st.seconds = since(t0);
    return out;
}

}  // namespace

bool ExactCocycles(const IsoTreeCarrierZ& c, const std::vector<std::vector<BigInt>>& B) {
    const SparseZ& A = c.d(1);
    for (const auto& b : B) {
        ENGINE_CHECK_MSG(b.size() == A.cols, "ExactCocycles: length");
        for (uint32_t i = 0; i < A.rows; ++i) {
            BigInt s(0);
            for (uint64_t k = A.rowptr[i]; k < A.rowptr[i + 1]; ++k)
                if (!b[A.col[k]].is_zero()) s += A.val[k] * b[A.col[k]];
            if (!s.is_zero()) return false;
        }
    }
    return true;
}
bool ExactKillsD0(const IsoTreeCarrierZ& c, const std::vector<std::vector<BigInt>>& Phi) {
    const SparseZ& A = c.d(0);   // rows C^1, cols C^0: v^T A = 0
    for (const auto& v : Phi) {
        ENGINE_CHECK_MSG(v.size() == A.rows, "ExactKillsD0: length");
        std::vector<BigInt> s(A.cols, BigInt(0));
        for (uint32_t i = 0; i < A.rows; ++i) {
            if (v[i].is_zero()) continue;
            for (uint64_t k = A.rowptr[i]; k < A.rowptr[i + 1]; ++k) s[A.col[k]] += v[i] * A.val[k];
        }
        for (const auto& x : s)
            if (!x.is_zero()) return false;
    }
    return true;
}

BigInt DetBareiss(std::vector<BigInt> A, uint32_t r) {
    if (r == 0) return BigInt(1);
    BigInt prev(1);
    int sign = 1;
    for (uint32_t k = 0; k + 1 < r; ++k) {
        if (A[size_t(k) * r + k].is_zero()) {
            uint32_t p = k + 1;
            while (p < r && A[size_t(p) * r + k].is_zero()) ++p;
            if (p == r) return BigInt(0);
            for (uint32_t j = 0; j < r; ++j) std::swap(A[size_t(k) * r + j], A[size_t(p) * r + j]);
            sign = -sign;
        }
        const BigInt piv = A[size_t(k) * r + k];
        for (uint32_t i = k + 1; i < r; ++i)
            for (uint32_t j = k + 1; j < r; ++j) {
                BigInt v = A[size_t(i) * r + j] * piv - A[size_t(i) * r + k] * A[size_t(k) * r + j];
                A[size_t(i) * r + j] = v / prev;
            }
        prev = piv;
    }
    BigInt d = A[size_t(r - 1) * r + (r - 1)];
    return sign < 0 ? -d : d;
}

std::vector<BigInt> PairingMatrix(const std::vector<std::vector<BigInt>>& Phi, const std::vector<std::vector<BigInt>>& B) {
    const uint32_t r = static_cast<uint32_t>(B.size());
    ENGINE_CHECK_MSG(Phi.size() == r, "PairingMatrix: shapes");
    std::vector<BigInt> G(size_t(r) * r, BigInt(0));
    for (uint32_t i = 0; i < r; ++i)
        for (uint32_t j = 0; j < r; ++j) {
            BigInt s(0);
            for (size_t t = 0; t < B[j].size(); ++t)
                if (!Phi[i][t].is_zero() && !B[j][t].is_zero()) s += Phi[i][t] * B[j][t];
            G[size_t(i) * r + j] = std::move(s);
        }
    return G;
}

// ---------------------------------------------------------------------------------------------------------------------------------
// PackedZ (EXZ2-P1b, audit F5)
// ---------------------------------------------------------------------------------------------------------------------------------
void PackedZ::append_col(const std::vector<std::pair<uint32_t, BigInt>>& entries) {
    // (fix round 1, audit F12) the precondition "rows strictly ascending" is checked (a duplicate row would corrupt PairingMatrix /
    // PackedFnv silently); O(entries)
    int64_t last = -1;
    for (const auto& [i, v] : entries) {
        ENGINE_CHECK_MSG(static_cast<int64_t>(i) > last, "PackedZ::append_col: rows not strictly ascending");
        last = i;
        if (v.is_zero()) continue;
        ENGINE_CHECK_MSG(i < rows, "PackedZ::append_col: row index out of range");
        const auto& mg = v.magnitude();
        row.push_back(i);
        off.push_back(limbs.size());
        limbs.push_back((static_cast<uint32_t>(mg.size()) << 1) | (v.is_neg() ? 1u : 0u));
        limbs.insert(limbs.end(), mg.begin(), mg.end());
    }
    colptr.push_back(row.size());
    ++cols;
}
void PackedZ::append_dense_col(const std::vector<BigInt>& v) {
    ENGINE_CHECK_MSG(v.size() == rows, "PackedZ::append_dense_col: length");
    std::vector<std::pair<uint32_t, BigInt>> e;
    for (uint32_t i = 0; i < rows; ++i)
        if (!v[i].is_zero()) e.push_back({i, v[i]});
    append_col(e);
}
BigInt PackedZ::value(uint64_t k) const {
    const uint64_t o = off[k];
    const uint32_t h = limbs[o];
    BigInt::Mag mg(limbs.begin() + static_cast<std::ptrdiff_t>(o + 1), limbs.begin() + static_cast<std::ptrdiff_t>(o + 1 + (h >> 1)));
    return BigInt::from_magnitude(std::move(mg), (h & 1u) != 0);
}
uint32_t PackedZ::mod_u32(uint64_t k, uint32_t q) const {
    const uint64_t o = off[k];
    const uint32_t h = limbs[o], cnt = h >> 1;
    uint64_t r = 0;
    for (uint32_t i = cnt; i-- > 0;) r = ((r << 32) | limbs[o + 1 + i]) % q;
    if ((h & 1u) && r) r = q - r;
    return static_cast<uint32_t>(r);
}
uint32_t PackedZ::bits(uint64_t k) const {
    const uint64_t o = off[k];
    const uint32_t cnt = limbs[o] >> 1;
    if (!cnt) return 0;
    uint32_t top = limbs[o + cnt], b = 0;
    while (top) {
        ++b;
        top >>= 1;
    }
    return 32 * (cnt - 1) + b;
}
std::vector<BigInt> PackedZ::col_dense(uint32_t j) const {
    std::vector<BigInt> v(rows, BigInt(0));
    for (uint64_t k = colptr[j]; k < colptr[j + 1]; ++k) v[row[k]] = value(k);
    return v;
}
std::vector<std::pair<uint32_t, BigInt>> PackedZ::col_sparse(uint32_t j) const {
    std::vector<std::pair<uint32_t, BigInt>> v;
    for (uint64_t k = colptr[j]; k < colptr[j + 1]; ++k) v.push_back({row[k], value(k)});
    return v;
}
std::vector<std::vector<BigInt>> PackedZ::dense() const {
    std::vector<std::vector<BigInt>> out;
    for (uint32_t j = 0; j < cols; ++j) out.push_back(col_dense(j));
    return out;
}
uint32_t PackedZ::max_bits() const {
    uint32_t b = 0;
    for (uint64_t k = 0; k < nnz(); ++k) b = std::max(b, bits(k));
    return b;
}
uint64_t PackedZ::bytes() const { return colptr.size() * 8 + row.size() * 4 + off.size() * 8 + limbs.size() * 4; }
PackedZ PackedZ::FromDense(const std::vector<std::vector<BigInt>>& cols_, uint32_t rows_) {
    PackedZ A(rows_);
    for (const auto& c : cols_) A.append_dense_col(c);
    return A;
}
uint64_t PackedFnv(const PackedZ& A) {
    std::vector<std::vector<std::pair<uint32_t, uint64_t>>> byrow(A.rows);
    for (uint32_t j = 0; j < A.cols; ++j)
        for (uint64_t k = A.colptr[j]; k < A.colptr[j + 1]; ++k) byrow[A.row[k]].push_back({j, k});
    uint64_t h = engine::kFnvOffset64;
    for (uint32_t i = 0; i < A.rows; ++i) {
        std::string line;
        if (i) line += ";";
        size_t t = 0;
        for (uint32_t j = 0; j < A.cols; ++j) {
            if (j) line += ",";
            if (t < byrow[i].size() && byrow[i][t].first == j) line += A.value(byrow[i][t++].second).to_string();
            else line += "0";
        }
        h = engine::FnvBytes(h, line);
    }
    return h;
}

// (S25 EXZ2-P3) the one exact vanishing logic for packed columns (P2's FreeBlockCrt premises and P1b's certificate both call these):
// the index of the first column that fails, -1 when all hold.
int64_t FirstNonCocycle(const IsoTreeCarrierZ& c, const PackedZ& B) {
    const SparseZ& A = c.d(1);
    ENGINE_CHECK_MSG(B.rows == A.cols, "ExactCocycles: length");
    const SparseZ At = A.transpose();   // rows = columns of d^1
    for (uint32_t j = 0; j < B.cols; ++j) {
        std::map<uint32_t, BigInt> y;
        for (uint64_t k = B.colptr[j]; k < B.colptr[j + 1]; ++k) {
            const BigInt b = B.value(k);
            const uint32_t t = B.row[k];
            for (uint64_t e = At.rowptr[t]; e < At.rowptr[t + 1]; ++e) y[At.col[e]] += At.val[e] * b;
        }
        for (const auto& kv : y)
            if (!kv.second.is_zero()) return j;
    }
    return -1;
}
int64_t FirstNotKillingD0(const IsoTreeCarrierZ& c, const PackedZ& Phi) {
    const SparseZ& A = c.d(0);
    ENGINE_CHECK_MSG(Phi.rows == A.rows, "ExactKillsD0: length");
    for (uint32_t j = 0; j < Phi.cols; ++j) {
        std::vector<BigInt> s(A.cols, BigInt(0));
        for (uint64_t k = Phi.colptr[j]; k < Phi.colptr[j + 1]; ++k) {
            const BigInt v = Phi.value(k);
            const uint32_t i = Phi.row[k];
            for (uint64_t e = A.rowptr[i]; e < A.rowptr[i + 1]; ++e) s[A.col[e]] += v * A.val[e];
        }
        for (const auto& x : s)
            if (!x.is_zero()) return j;
    }
    return -1;
}
bool ExactCocycles(const IsoTreeCarrierZ& c, const PackedZ& B) { return FirstNonCocycle(c, B) < 0; }
bool ExactKillsD0(const IsoTreeCarrierZ& c, const PackedZ& Phi) { return FirstNotKillingD0(c, Phi) < 0; }
std::vector<BigInt> PairingMatrix(const PackedZ& Phi, const PackedZ& B) {
    ENGINE_CHECK_MSG(Phi.rows == B.rows, "PairingMatrix: shapes");
    const uint32_t a = Phi.cols, b = B.cols;
    std::vector<BigInt> G(size_t(a) * b, BigInt(0));
    std::vector<BigInt> Phv(Phi.nnz());
    for (uint64_t k = 0; k < Phi.nnz(); ++k) Phv[k] = Phi.value(k);
    std::vector<int64_t> pos(B.rows, -1);
    std::vector<BigInt> bv;
    for (uint32_t j = 0; j < b; ++j) {
        bv.clear();
        for (uint64_t k = B.colptr[j]; k < B.colptr[j + 1]; ++k) {
            pos[B.row[k]] = static_cast<int64_t>(bv.size());
            bv.push_back(B.value(k));
        }
        for (uint32_t i = 0; i < a; ++i) {
            BigInt s(0);
            for (uint64_t k = Phi.colptr[i]; k < Phi.colptr[i + 1]; ++k) {
                const int64_t p = pos[Phi.row[k]];
                if (p >= 0) s += Phv[k] * bv[size_t(p)];
            }
            G[size_t(i) * b + j] = std::move(s);
        }
        for (uint64_t k = B.colptr[j]; k < B.colptr[j + 1]; ++k) pos[B.row[k]] = -1;
    }
    return G;
}

// ---------------------------------------------------------------------------------------------------------------------------------
// iso_dixon/v3: the orders, the predictor, the pairs
// ---------------------------------------------------------------------------------------------------------------------------------
uint32_t PairRotation(uint32_t s, uint32_t D) {
    const uint32_t t = s / 2;
    ENGINE_CHECK_MSG(t < 3, "PairRotation: iso_dixon/v3 defines s = 0 .. 5 (K <= 5)");
    const uint32_t rho[3] = {0u, D / 2, D / 4};
    return rho[t];
}
std::vector<uint32_t> PairOrder(uint32_t s, uint32_t D, uint32_t nv) {
    const uint32_t k = nv ? PairRotation(s, D) % nv : 0;
    std::vector<uint32_t> o(nv);
    for (uint32_t i = 0; i < nv; ++i) o[i] = (i + k) % nv;
    if (PairReversed(s)) std::reverse(o.begin(), o.end());
    return o;
}

uint64_t PredictPairsRss(const IsoTreeCarrierZ& c, uint32_t K, bool packed, uint32_t batch) {
    // audit EXZ2P1a_audit.md Sec. 4.1, copied exactly (see the header); the two layout substitutions the audit names
    auto h = [](double b) -> double {
        const double w = std::ceil(b / 32.0);
        return std::max(32.0, 16.0 * std::ceil((4.0 * w + 8.0) / 16.0));
    };
    const double nv = c.dim(1), n0 = c.dim(0), r = c.b1();
    const double bd = std::max<uint32_t>(c.d(0).max_bits(), c.d(1).max_bits());
    const double nn = c.n();
    const double beta = std::max(2.0 * bd + 8.0, std::ceil(std::pow(nn, 1.6) / 3.0));
    const double sigma = std::ceil(beta / 10.0) + 2.0;
    const double rb = batch == 0 ? r : std::min<double>(r, batch);
    const double nnz = static_cast<double>(c.d(0).nnz() + c.d(1).nnz());
    double b = 16.0 * 1048576.0;
    if (!packed) b += 64.0 * K * r * nv;
    b += K * r * (24.0 * h(bd) + 0.8 * (n0 + r) * h(beta));
    b += 3.0 * rb * n0 * (32.0 + h(30.0 * sigma));
    b += nnz * (72.0 + h(bd));
    return static_cast<uint64_t>(std::ceil(b));
}

namespace {
struct PairsCtx {
    uint32_t q0 = 0, q1 = 0;          // q1: the profile-confirmation prime (fix round 1, F7)
    SparseZ d1t, d0t;
    std::vector<uint32_t> d1t_q, d0_q, d1t_q1, d0_q1;
};
// the confirmation prime for q0: the first CrtPrimes prime != q0 prime to 6N
uint32_t ConfirmPrime(uint32_t q0, uint32_t N) {
    for (uint32_t q : integral::CrtPrimes(64))
        if (q != q0 && (6ull * N) % q != 0) return q;
    ENGINE_CHECK_MSG(false, "ConfirmPrime: no CrtPrimes prime prime to 6N");
    return 0;
}
void MakeCtx(const IsoTreeCarrierZ& c, uint32_t q0, PairsCtx& X) {
    X.q0 = q0;
    X.q1 = ConfirmPrime(q0, c.N());
    if (X.d1t.rows == 0 && X.d1t.cols == 0) {
        X.d1t = c.d(1).transpose();
        X.d0t = c.d(0).transpose();
    }
    X.d1t_q = Residues(X.d1t, q0);
    X.d0_q = Residues(c.d(0), q0);
    X.d1t_q1 = Residues(X.d1t, X.q1);
    X.d0_q1 = Residues(c.d(0), X.q1);
}
// the greedy profiles under `order` modulo E1.q() = E0.q(): Pc (columns of d^1), then on the rest Fr the rows R0 of d^0 and P = Fr - R0
void GreedyProfile(const IsoTreeCarrierZ& c, const PairsCtx& X, const std::vector<uint32_t>& d1t_q, const std::vector<uint32_t>& d0_q,
                   const std::vector<uint32_t>& order, EchelonModQ& E1, EchelonModQ& E0, std::vector<uint32_t>& Pc,
                   std::vector<uint32_t>& R0, std::vector<uint32_t>& P) {
    const SparseZ& d0 = c.d(0);
    std::vector<uint32_t> Fr;
    for (uint32_t j : order) {
        SpVec v;
        for (uint64_t k = X.d1t.rowptr[j]; k < X.d1t.rowptr[j + 1]; ++k) {
            if (!d1t_q[k]) continue;
            v.idx.push_back(X.d1t.col[k]);
            v.val.push_back(d1t_q[k]);
        }
        if (E1.insert(v)) Pc.push_back(j);
        else Fr.push_back(j);
    }
    for (uint32_t j : Fr) {
        SpVec v;
        for (uint64_t k = d0.rowptr[j]; k < d0.rowptr[j + 1]; ++k) {
            if (!d0_q[k]) continue;
            v.idx.push_back(d0.col[k]);
            v.val.push_back(d0_q[k]);
        }
        if (E0.insert(v)) R0.push_back(j);
        else P.push_back(j);
    }
}
// false: a rank profile mod q0 below the rational one (F10: the caller moves to the next prime)
bool BuildOnePair(const IsoTreeCarrierZ& c, const PairsCtx& X, uint32_t s, uint32_t batch, Pair& P, uint32_t* rk1, uint32_t* rk0) {
    const auto te = clk::now();
    const SparseZ& d0 = c.d(0);
    const SparseZ& d1 = c.d(1);
    const uint32_t nv = c.dim(1), n0 = c.dim(0), n2 = c.dim(2), q0 = X.q0;
    P = Pair();
    P.s = s;
    P.rot = PairRotation(s, c.D());
    P.reversed = PairReversed(s);
    const std::vector<uint32_t> order = PairOrder(s, c.D(), nv);
    EchelonModQ E1(q0, n2);
    EchelonModQ E0(q0, n0);
    GreedyProfile(c, X, X.d1t_q, X.d0_q, order, E1, E0, P.Pc, P.R0, P.P);
    P.t_elim = since(te);
    P.fill_d1 = E1.fill();
    P.fill_d0 = E0.fill();
    P.tnnz_d1 = E1.t_nnz();
    P.tnnz_d0 = E0.t_nnz();
    if (rk1) *rk1 = E1.rank();
    if (rk0) *rk0 = E0.rank();
    const uint32_t exp_rk1 = c.D() + c.b0() - c.b1(), exp_rk0 = n0 - c.b0();
    if (E1.rank() != exp_rk1 || E0.rank() != exp_rk0 || P.P.size() != c.b1()) return false;
    {
        // (fix round 1, audit F7) the profile confirmed at the second prime q1 (one more elimination of each matrix; the echelons at q1
        // are dropped at once).  A disagreement: the caller moves q0 like a rank drop.
        std::vector<uint32_t> Pc1, R01, P1;
        EchelonModQ F1(X.q1, n2), F0(X.q1, n0);
        GreedyProfile(c, X, X.d1t_q1, X.d0_q1, order, F1, F0, Pc1, R01, P1);
        P.t_elim = since(te);
        if (Pc1 != P.Pc || R01 != P.R0 || P1 != P.P) return false;
    }
    const uint32_t r = c.b1();
    // ---- B side: the system vectors are the columns Pc of d^1 (length n2), rhs_j = - column P_j
    std::vector<SpZ> V1(P.Pc.size()), rhs1(r);
    auto colZ = [&](uint32_t j, bool neg) {
        SpZ v;
        for (uint64_t k = X.d1t.rowptr[j]; k < X.d1t.rowptr[j + 1]; ++k) {
            v.idx.push_back(X.d1t.col[k]);
            v.val.push_back(neg ? -X.d1t.val[k] : X.d1t.val[k]);
        }
        return v;
    };
    for (size_t t = 0; t < P.Pc.size(); ++t) V1[t] = colZ(P.Pc[t], false);
    for (uint32_t j = 0; j < r; ++j) rhs1[j] = colZ(P.P[j], true);
    auto bcol = [&](uint32_t j, const std::vector<BigInt>& nums, const BigInt& den) {
        std::vector<BigInt> v(nv, BigInt(0));
        v[P.P[j]] = den;
        for (size_t t = 0; t < P.Pc.size(); ++t) v[P.Pc[t]] = nums[t];
        return v;
    };
    uint32_t vl = 0;
    DixonOut ob = Dixon(V1, E1, rhs1, n2, c.N(), [&](uint32_t j, const std::vector<BigInt>& nums, const BigInt& den) {
        uint32_t l = 0;
        const bool ok = LaneZero(d1, bcol(j, nums, den), q0, c.N(), &l);
        vl = std::max(vl, l);
        return ok;
    }, batch);
    P.dixB = ob.st;
    P.dixB.verify_lanes = vl;
    // ---- Phi side: system vectors = the rows R0 of d^0 (length n0), rhs_i = - row P_i
    std::vector<SpZ> V0(P.R0.size()), rhs0(r);
    auto rowZ = [&](uint32_t j, bool neg) {
        SpZ v;
        for (uint64_t k = d0.rowptr[j]; k < d0.rowptr[j + 1]; ++k) {
            v.idx.push_back(d0.col[k]);
            v.val.push_back(neg ? -d0.val[k] : d0.val[k]);
        }
        return v;
    };
    for (size_t t = 0; t < P.R0.size(); ++t) V0[t] = rowZ(P.R0[t], false);
    for (uint32_t i = 0; i < r; ++i) rhs0[i] = rowZ(P.P[i], true);
    auto pcol = [&](uint32_t i, const std::vector<BigInt>& nums, const BigInt& den) {
        std::vector<BigInt> v(nv, BigInt(0));
        v[P.P[i]] = den;
        for (size_t t = 0; t < P.R0.size(); ++t) v[P.R0[t]] = nums[t];
        return v;
    };
    vl = 0;
    DixonOut op = Dixon(V0, E0, rhs0, n0, c.N(), [&](uint32_t i, const std::vector<BigInt>& nums, const BigInt& den) {
        uint32_t l = 0;
        const bool ok = LaneZero(X.d0t, pcol(i, nums, den), q0, c.N(), &l);
        vl = std::max(vl, l);
        return ok;
    }, batch);
    P.dixPhi = op.st;
    P.dixPhi.verify_lanes = vl;
    // stream the columns into the packed masters (sparse: the supports Pc + {P_j}, R0 + {P_i})
    P.B = PackedZ(nv);
    P.Phi = PackedZ(nv);
    for (uint32_t j = 0; j < r; ++j) {
        std::vector<std::pair<uint32_t, BigInt>> e;
        e.push_back({P.P[j], ob.den[j]});
        for (size_t t = 0; t < P.Pc.size(); ++t)
            if (!ob.nums[j][t].is_zero()) e.push_back({P.Pc[t], ob.nums[j][t]});
        std::sort(e.begin(), e.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        P.B.append_col(e);
        P.bits_denB = std::max(P.bits_denB, BitsOf(ob.den[j]));
        std::vector<BigInt>().swap(ob.nums[j]);
        e.clear();
        e.push_back({P.P[j], op.den[j]});
        for (size_t t = 0; t < P.R0.size(); ++t)
            if (!op.nums[j][t].is_zero()) e.push_back({P.R0[t], op.nums[j][t]});
        std::sort(e.begin(), e.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        P.Phi.append_col(e);
        P.bits_denPhi = std::max(P.bits_denPhi, BitsOf(op.den[j]));
        std::vector<BigInt>().swap(op.nums[j]);
    }
    P.bits_B = P.B.max_bits();
    P.bits_Phi = P.Phi.max_bits();
    P.fnv_B = PackedFnv(P.B);
    P.fnv_Phi = PackedFnv(P.Phi);
    P.support_Phi = static_cast<uint32_t>(P.R0.size() + P.P.size());
    return true;
}
// q0 candidates: the requested (or pinned) q0, then the CrtPrimes primes below it in CrtPrimes order (all of them when the requested
// prime is not a CrtPrimes prime, e.g. a small test prime)
std::vector<uint32_t> Q0Candidates(uint32_t first) {
    const std::vector<uint32_t> cp = integral::CrtPrimes(64);
    std::vector<uint32_t> out{first};
    const bool listed = std::find(cp.begin(), cp.end(), first) != cp.end();
    for (uint32_t q : cp)
        if (listed ? q < first : q != first) out.push_back(q);
    return out;
}
}  // namespace

PairsResult BuildPairs(const IsoTreeCarrierZ& c, const PairsOptions& opt, const testing::PairsHooks* hooks) {
    const auto t0 = clk::now();
    PairsResult res;
    res.r = c.b1();
    ENGINE_CHECK_MSG(opt.K >= 1 && opt.K <= kPairsKMax, "BuildPairs: 1 <= K <= 5 (iso_dixon/v3)");
    res.rss_pred = PredictPairsRss(c, opt.K, true, opt.dixon_batch);
    if (res.rss_pred > opt.rss_cap_bytes) {
        res.refused = true;
        std::ostringstream o;
        o << "REFUSED [RssCeiling] the Dixon pairs at (N, n) = (" << c.N() << ", " << c.n() << "): predicted " << res.rss_pred
          << " bytes > the ceiling " << opt.rss_cap_bytes << " bytes";
        res.refusal = o.str();
        return res;
    }
    const uint32_t nv = c.dim(1), n0 = c.dim(0), n2 = c.dim(2);
    ENGINE_CHECK_MSG(c.d(1).cols == nv && c.d(1).rows == n2 && c.d(0).rows == nv && c.d(0).cols == n0, "BuildPairs: carrier shapes");
    PairsCtx X;
    bool done = false;
    for (uint32_t q0 : Q0Candidates(opt.q0 ? opt.q0 : kQ0)) {
        if ((6ull * c.N()) % q0 == 0) {
            res.q0_skipped.push_back(q0);
            continue;
        }
        MakeCtx(c, q0, X);
        res.pairs.clear();
        bool ok = true;
        for (uint32_t s = 0; s < opt.K && ok; ++s) {
            Pair P;
            uint32_t rk1 = 0, rk0 = 0;
            ok = !(hooks && q0 == hooks->rank_drop_q0) && BuildOnePair(c, X, s, opt.dixon_batch, P, &rk1, &rk0);
            if (s == 0) {
                res.rank_d1 = rk1;
                res.rank_d0 = rk0;
            }
            if (ok) res.pairs.push_back(std::move(P));
        }
        if (!ok) {
            res.q0_skipped.push_back(q0);
            continue;
        }
        res.q0 = q0;
        res.q0_confirm = X.q1;
        done = true;
        break;
    }
    ENGINE_CHECK_MSG(done, "BuildPairs: no admissible q0 among 64 CrtPrimes primes (every one drops a rank)");
    res.seconds = since(t0);
    return res;
}

void ExtendPairs(const IsoTreeCarrierZ& c, PairsResult& res, const PairsOptions& opt) {
    const auto t0 = clk::now();
    const uint32_t s = static_cast<uint32_t>(res.pairs.size());
    ENGINE_CHECK_MSG(!res.refused && s < kPairsKMax, "ExtendPairs: K <= 5 (iso_dixon/v3)");
    // (fix round 1, audit F5) the ceiling for K = s + 1, before anything is built
    const uint64_t pred = PredictPairsRss(c, s + 1, true, opt.dixon_batch);
    if (pred > opt.rss_cap_bytes) {
        res.refused = true;
        res.rss_pred = pred;
        std::ostringstream o;
        o << "REFUSED [RssCeiling] the K-escalation to K = " << (s + 1) << " at (N, n) = (" << c.N() << ", " << c.n() << "): predicted "
          << pred << " bytes > the ceiling " << opt.rss_cap_bytes << " bytes";
        res.refusal = o.str();
        return;
    }
    res.rss_pred = pred;
    PairsCtx X;
    MakeCtx(c, res.q0, X);
    Pair P;
    if (BuildOnePair(c, X, s, opt.dixon_batch, P, nullptr, nullptr)) {
        res.pairs.push_back(std::move(P));
        res.seconds += since(t0);
        return;
    }
    // F10: a rank drop at the current q0 for the new order: rebuild every pair at the next admissible prime
    PairsOptions o = opt;
    o.K = s + 1;
    const std::vector<uint32_t> cand = Q0Candidates(res.q0);
    ENGINE_CHECK_MSG(cand.size() > 1, "ExtendPairs: no further q0");
    o.q0 = cand[1];
    std::vector<uint32_t> skipped = res.q0_skipped;
    skipped.push_back(res.q0);
    res = BuildPairs(c, o);
    skipped.insert(skipped.end(), res.q0_skipped.begin(), res.q0_skipped.end());
    res.q0_skipped = skipped;
}

}  // namespace freez
}  // namespace integral_descent
