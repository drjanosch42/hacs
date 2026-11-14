// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Slot-26 card Z3b: the crossed-homomorphism model and the operator [C, eq:barcorrected] through Shapiro.  See the header and
// docs/implementation/slot26/Z3b_design.md v0.2.
#include "HeckeZCrossedHomOracle.h"

#include <string>

#include "../../Hecke/HeckeLevelN.h"
#include "../../../ModArithmetic/EngineCheck.h"

namespace integral_descent {
namespace xhom {

namespace {
uint64_t Red(int64_t v, uint64_t Q) {
    const int64_t q = static_cast<int64_t>(Q);
    int64_t r = v % q;
    return static_cast<uint64_t>(r < 0 ? r + q : r);
}
uint64_t MulMod(uint64_t a, uint64_t b, uint64_t Q) { return static_cast<uint64_t>((unsigned __int128)a * b % Q); }
// Y (d x k) += B (d x d) X (d x k) mod Q
void BlockMulAdd(const std::vector<uint64_t>& B, const uint64_t* X, uint64_t* Y, uint32_t d, uint32_t k, uint64_t Q) {
    for (uint32_t s = 0; s < d; ++s)
        for (uint32_t c = 0; c < k; ++c) {
            unsigned __int128 acc = Y[size_t(s) * k + c];
            const uint64_t* br = B.data() + size_t(s) * d;
            for (uint32_t t = 0; t < d; ++t)
                if (br[t]) acc += (unsigned __int128)br[t] * X[size_t(t) * k + c];
            Y[size_t(s) * k + c] = static_cast<uint64_t>(acc % Q);
        }
}
}  // namespace

// ---- rho_n ----------------------------------------------------------------------------------------------------------------------
std::vector<uint64_t> SymRhoMod(const Mat22& g, uint32_t n, uint64_t Q) {
    ENGINE_CHECK_MSG(Q >= 2 && Q < (1ull << 32), "SymRhoMod: 2 <= Q < 2^32");
    const uint32_t d = n + 1;
    const uint64_t a = Red(g.a, Q), b = Red(g.b, Q), c = Red(g.c, Q), dd = Red(g.d, Q);
    std::vector<uint64_t> pa(d, 1), pb(d, 1), pc(d, 1), pd(d, 1);
    for (uint32_t i = 1; i < d; ++i) {
        pa[i] = MulMod(pa[i - 1], a, Q);
        pb[i] = MulMod(pb[i - 1], b, Q);
        pc[i] = MulMod(pc[i - 1], c, Q);
        pd[i] = MulMod(pd[i - 1], dd, Q);
    }
    // binomials mod Q by Pascal (Q need not be prime)
    std::vector<std::vector<uint64_t>> C(d, std::vector<uint64_t>(d, 0));
    for (uint32_t i = 0; i < d; ++i) {
        C[i][0] = 1 % Q;
        for (uint32_t j = 1; j <= i; ++j) C[i][j] = (C[i - 1][j - 1] + (j < i ? C[i - 1][j] : 0)) % Q;
    }
    std::vector<uint64_t> R(size_t(d) * d, 0);
    std::vector<uint64_t> A(d), B(d);
    for (uint32_t j = 0; j < d; ++j) {
        const uint32_t e = n - j;
        for (uint32_t s = 0; s <= e; ++s) A[s] = MulMod(MulMod(C[e][s], pa[e - s], Q), pc[s], Q);
        for (uint32_t u = 0; u <= j; ++u) B[u] = MulMod(MulMod(C[j][u], pb[j - u], Q), pd[u], Q);
        for (uint32_t s = 0; s <= e; ++s) {
            if (!A[s]) continue;
            for (uint32_t u = 0; u <= j; ++u) {
                uint64_t& r = R[size_t(s + u) * d + j];
                r = (r + MulMod(A[s], B[u], Q)) % Q;
            }
        }
    }
    return R;
}

std::vector<nf::BigInt> SymRhoExact(const Mat22& g, uint32_t n) {
    using nf::BigInt;
    const uint32_t d = n + 1;
    std::vector<BigInt> pa(d, BigInt(1)), pb(d, BigInt(1)), pc(d, BigInt(1)), pd(d, BigInt(1));
    for (uint32_t i = 1; i < d; ++i) {
        pa[i] = pa[i - 1] * BigInt(g.a);
        pb[i] = pb[i - 1] * BigInt(g.b);
        pc[i] = pc[i - 1] * BigInt(g.c);
        pd[i] = pd[i - 1] * BigInt(g.d);
    }
    std::vector<std::vector<BigInt>> C(d, std::vector<BigInt>(d, BigInt(0)));
    for (uint32_t i = 0; i < d; ++i) {
        C[i][0] = BigInt(1);
        for (uint32_t j = 1; j <= i; ++j) C[i][j] = C[i - 1][j - 1] + (j < i ? C[i - 1][j] : BigInt(0));
    }
    std::vector<BigInt> R(size_t(d) * d, BigInt(0));
    std::vector<BigInt> A(d), B(d);
    for (uint32_t j = 0; j < d; ++j) {
        const uint32_t e = n - j;
        for (uint32_t s = 0; s <= e; ++s) A[s] = C[e][s] * pa[e - s] * pc[s];
        for (uint32_t u = 0; u <= j; ++u) B[u] = C[j][u] * pb[j - u] * pd[u];
        for (uint32_t s = 0; s <= e; ++s) {
            if (A[s].is_zero()) continue;
            for (uint32_t u = 0; u <= j; ++u) R[size_t(s + u) * d + j] += A[s] * B[u];
        }
    }
    return R;
}

// ---- words ------------------------------------------------------------------------------------------------------------------------
Mat22 LetterMat(uint8_t x) {
    switch (x) {
        case kS: return Mat22{0, -1, 1, 0};
        case kU: return Mat22{0, -1, 1, 1};
        case kT: return Mat22{1, 1, 0, 1};
        case kTi: return Mat22{1, -1, 0, 1};
    }
    ENGINE_CHECK_MSG(false, "LetterMat: letter");
    return Mat22{1, 0, 0, 1};
}
Mat22 Adj(const Mat22& g) { return Mat22{g.d, -g.b, -g.c, g.a}; }

std::vector<uint8_t> Word(const Mat22& g) {
    ENGINE_CHECK_MSG(g.det() == 1, "Word: g is not in SL_2(Z)");
    std::vector<uint8_t> w;
    auto pushT = [&w](int64_t q) {
        const uint8_t x = q >= 0 ? kT : kTi;
        for (int64_t i = 0; i < (q >= 0 ? q : -q); ++i) w.push_back(x);
    };
    int64_t a = g.a, b = g.b, c = g.c, d = g.d;
    while (c != 0) {
        int64_t q = a / c;
        if ((a % c != 0) && ((a < 0) != (c < 0))) --q;   // floor
        pushT(q);
        const int64_t a1 = a - q * c, b1 = b - q * d;    // T^{-q} g
        w.push_back(kS);                                 // = S g2, g2 = S^{-1} T^{-q} g
        a = c;
        b = d;
        c = -a1;
        d = -b1;
    }
    ENGINE_CHECK_MSG((a == 1 && d == 1) || (a == -1 && d == -1), "Word: Euclid did not end in +-T^b");
    if (a == 1) pushT(b);
    else {
        w.push_back(kS);
        w.push_back(kS);   // -I = S^2
        pushT(-b);
    }
    Mat22 P{1, 0, 0, 1};
    for (uint8_t x : w) P = P * LetterMat(x);
    ENGINE_CHECK_MSG(P == g, "Word: the word does not multiply back to g");
    return w;
}

std::vector<Mat22> StandardReps(uint32_t p) {
    std::vector<Mat22> r;
    for (uint32_t j = 0; j < p; ++j) r.push_back(Mat22{int64_t(p), -int64_t(j), 0, 1});
    r.push_back(Mat22{1, 0, 0, int64_t(p)});
    return r;
}

// ---- the model ------------------------------------------------------------------------------------------------------------------
CrossedModel::CrossedModel(uint32_t N_, uint32_t n_, uint64_t Q_) : N(N_), n(n_), Q(Q_) {
    ENGINE_CHECK_MSG(Q >= 2 && Q < (1ull << 32), "CrossedModel: 2 <= Q < 2^32");
    const hecke::level::Gamma0LevelN A(N);
    m = A.m();
    d = n + 1;
    D = m * d;
    ENGINE_CHECK_MSG(A.rep(1) == (Mat22{1, 0, 0, 1}), "CrossedModel: w_1 != I (the Shapiro base class)");
    std::map<std::array<int64_t, 4>, std::vector<uint64_t>> memo;
    for (uint8_t x = 0; x < 4; ++x) {
        col[x].resize(m);
        h[x].resize(m);
        blk[x].resize(m);
        const Mat22 X = LetterMat(x);
        for (uint32_t j = 1; j <= m; ++j) {
            const Mat22 y = A.rep(j) * X;
            const uint32_t jj = A.class_of(y);
            const Mat22 hh = y * A.rep(jj).inv_unimodular();
            ENGINE_CHECK_MSG(A.contains(hh), "CrossedModel: h_j(x) not in Gamma_0(N)");
            col[x][j - 1] = jj - 1;
            h[x][j - 1] = hh;
            const std::array<int64_t, 4> key{hh.a, hh.b, hh.c, hh.d};
            auto it = memo.find(key);
            if (it == memo.end()) it = memo.emplace(key, SymRhoMod(hh, n, Q)).first;
            blk[x][j - 1] = it->second;
        }
    }
}

void CrossedModel::act(uint8_t x, const std::vector<uint64_t>& v, uint32_t k, std::vector<uint64_t>& out) const {
    ENGINE_CHECK_MSG(v.size() == size_t(D) * k, "CrossedModel::act: shape");
    out.assign(size_t(D) * k, 0);
    for (uint32_t j = 0; j < m; ++j)
        BlockMulAdd(blk[x][j], v.data() + size_t(col[x][j]) * d * k, out.data() + size_t(j) * d * k, d, k, Q);
}

void CrossedModel::kappa(const std::vector<uint64_t>& c, uint32_t k, std::vector<uint64_t>& fS, std::vector<uint64_t>& fU) const {
    ENGINE_CHECK_MSG(c.size() == size_t(3) * D * k, "CrossedModel::kappa: shape");
    const size_t B = size_t(D) * k;
    fS.assign(c.begin(), c.begin() + B);
    std::vector<uint64_t> z(c.begin() + 2 * B, c.end()), Uz;
    act(kU, z, k, Uz);
    fU.resize(B);
    for (size_t e = 0; e < B; ++e) fU[e] = ((c[B + e] + z[e]) % Q + Q - Uz[e]) % Q;
}

std::vector<uint64_t> CrossedModel::kappa_prime(const std::vector<uint64_t>& fS, const std::vector<uint64_t>& fU, uint32_t k) const {
    const size_t B = size_t(D) * k;
    ENGINE_CHECK_MSG(fS.size() == B && fU.size() == B, "CrossedModel::kappa_prime: shape");
    std::vector<uint64_t> y(3 * B, 0);
    std::copy(fS.begin(), fS.end(), y.begin());
    std::copy(fU.begin(), fU.end(), y.begin() + B);
    return y;
}

// ---- the operator ---------------------------------------------------------------------------------------------------------------
std::vector<uint64_t> HeckeCross(const CrossedModel& X, const std::vector<uint64_t>& fS, const std::vector<uint64_t>& fU, uint32_t k, uint32_t p,
                                 const std::vector<Mat22>& reps, bool iota, HeckeStats* st) {
    const uint32_t d = X.d, D = X.D;
    const uint64_t Q = X.Q;
    const size_t B = size_t(D) * k;
    ENGINE_CHECK_MSG(p >= 2 && X.N % p != 0, "HeckeCross: p !| N");
    ENGINE_CHECK_MSG(reps.size() == size_t(p) + 1, "HeckeCross: p + 1 representatives");
    ENGINE_CHECK_MSG(fS.size() == B && fU.size() == B, "HeckeCross: shape");
    // the four letter values: f(T) = f(S^3 U) = fS + S(fS + S(fS + S fU)), f(T^{-1}) = -T^{-1} f(T)
    std::array<std::vector<uint64_t>, 4> f;
    f[kS] = fS;
    f[kU] = fU;
    {
        std::vector<uint64_t> v = fU, w;
        for (int i = 0; i < 3; ++i) {
            X.act(kS, v, k, w);
            for (size_t e = 0; e < B; ++e) w[e] = (w[e] + fS[e]) % Q;
            v.swap(w);
        }
        f[kT] = v;
        X.act(kTi, f[kT], k, w);
        for (size_t e = 0; e < B; ++e) w[e] = (Q - w[e]) % Q;
        f[kTi] = w;
    }
    // phi(t) = f(t)_{j0}, t in A: Horner along the coset path (only block j0 = 0)
    std::map<std::array<int64_t, 4>, std::vector<uint64_t>> memo;
    HeckeStats local;
    HeckeStats& S = st ? *st : local;
    auto phi = [&](const Mat22& t) -> const std::vector<uint64_t>& {
        const std::array<int64_t, 4> key{t.a, t.b, t.c, t.d};
        auto it = memo.find(key);
        if (it != memo.end()) {
            ++S.memo_hits;
            return it->second;
        }
        ENGINE_CHECK_MSG(t.c % int64_t(X.N) == 0, "HeckeCross: phi evaluated outside Gamma_0(N)");
        const std::vector<uint8_t> w = Word(t);
        ++S.words;
        S.letters += w.size();
        std::vector<uint32_t> path(w.size() + 1, 0);
        for (size_t i = 0; i < w.size(); ++i) path[i + 1] = X.col[w[i]][path[i]];
        ENGINE_CHECK_MSG(path.back() == 0, "HeckeCross: t does not fix the base class j0");
        std::vector<uint64_t> v(size_t(d) * k, 0), u;
        for (size_t i = w.size(); i-- > 0;) {
            const uint8_t x = w[i];
            const uint32_t c = path[i];
            u.assign(f[x].begin() + size_t(c) * d * k, f[x].begin() + size_t(c + 1) * d * k);
            BlockMulAdd(X.blk[x][c], v.data(), u.data(), d, k, Q);
            v.swap(u);
        }
        return memo.emplace(key, std::move(v)).first->second;
    };
    std::vector<std::vector<uint64_t>> rho_rep;
    for (const Mat22& dl : reps) rho_rep.push_back(SymRhoMod(iota ? Adj(dl) : dl, X.n, Q));
    std::vector<uint64_t> fS2(B, 0), fU2(B, 0);
    for (uint8_t x : {uint8_t(kS), uint8_t(kU)}) {
        std::vector<uint64_t>& out = x == kS ? fS2 : fU2;
        for (uint32_t j = 0; j < X.m; ++j) {
            const Mat22& gam = X.h[x][j];
            uint64_t* dst = out.data() + size_t(j) * d * k;
            for (size_t jr = 0; jr < reps.size(); ++jr) {
                const Mat22 gd = gam * reps[jr];
                int hit = -1;
                Mat22 kap{1, 0, 0, 1};
                for (size_t kr = 0; kr < reps.size(); ++kr) {
                    const Mat22 Y = Adj(reps[kr]) * gd;
                    const int64_t P = p;
                    if (Y.a % P || Y.b % P || Y.c % P || Y.d % P) continue;
                    const Mat22 K{Y.a / P, Y.b / P, Y.c / P, Y.d / P};
                    if (K.c % int64_t(X.N) != 0) continue;
                    ENGINE_CHECK_MSG(hit < 0, "HeckeCross: two representatives hit (gamma delta_j)");
                    hit = static_cast<int>(kr);
                    kap = K;
                }
                ENGINE_CHECK_MSG(hit >= 0, "HeckeCross: no representative hit (gamma delta_j)");
                ENGINE_CHECK_MSG(kap.det() == 1, "HeckeCross: kappa_j(gamma) not in SL_2(Z)");
                ++S.evals;
                BlockMulAdd(rho_rep[size_t(hit)], phi(kap).data(), dst, d, k, Q);
            }
        }
    }
    return X.kappa_prime(fS2, fU2, k);
}

}  // namespace xhom
}  // namespace integral_descent
