// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 22, MP-3) Subgroups on the MP model: Shapiro, res / cor, forget-coset -- see SubgroupMP.h.
#include "SubgroupMP.h"

#include <map>
#include <set>

#include "../../../ModArithmetic/EngineCheck.h"

namespace orbifold_mp {

using integral_model::key_of;

namespace {

// The double cosets B\Gamma/H on the right cosets: dc[j] = the least class j' with B w_j H = B w_j' H (1-based).
std::vector<uint32_t> DoubleCosets(const CosetLevel& B, const std::vector<Mat22>& H) {
    const uint32_t m = B.m();
    std::vector<uint32_t> dc(m + 1, 0);
    for (uint32_t j = 1; j <= m; ++j) {
        if (dc[j]) continue;
        // closure of {j} under right multiplication by H (H a group: one pass suffices)
        std::set<uint32_t> orb;
        for (const Mat22& u : H) orb.insert(B.class_of(B.rep(j) * u));
        const uint32_t jmin = *orb.begin();
        for (uint32_t x : orb) {
            ENGINE_CHECK_MSG(dc[x] == 0 || dc[x] == jmin, "orbifold_mp::DoubleCosets: inconsistent double cosets");
            dc[x] = jmin;
        }
    }
    return dc;
}

Vec RhoSym(const engine::ModCtx<uint32_t>& ctx, const Mat22& g, uint32_t n, uint32_t m) {
    const uint32_t d = n + 1;
    const engine::DenseBuf<uint32_t> R = integral_model::rho_n(ctx, g, n);
    Vec out(static_cast<size_t>(d) * d);
    for (uint32_t a = 0; a < d; ++a)
        for (uint32_t b = 0; b < d; ++b) out[a * d + b] = R.at(a, b) % m;
    return out;
}

void PutBlock(Vec& M, uint32_t cols, uint32_t bi, uint32_t bj, const Vec& blk, uint32_t d, uint32_t m) {
    for (uint32_t a = 0; a < d; ++a)
        for (uint32_t b = 0; b < d; ++b) {
            uint32_t& x = M[static_cast<size_t>(bi * d + a) * cols + bj * d + b];
            x = (x + blk[a * d + b]) % m;
        }
}

}  // namespace

SubgroupModel SubgroupDescriptor(const Descriptor& D, const CosetLevel& B, const std::string& name) {
    SubgroupModel S;
    S.name = name;
    S.D.name = D.name + "/" + name;
    S.minus_I_in = B.contains(Mat22::I().neg());
    uint32_t next_label = 0;
    // Vertex B-orbits, per Gamma-orbit o: the double cosets B\Gamma/Gamma_o.
    struct OrbitInfo {
        std::vector<uint32_t> dc;           // class -> least class of its double coset
        std::map<uint32_t, uint32_t> borb;  // least class -> B-orbit index
        std::vector<Mat22> stab;            // Gamma_o elements
    };
    std::vector<OrbitInfo> info(D.orbit_stab.size());
    for (uint32_t o = 0; o < D.orbit_stab.size(); ++o) {
        info[o].stab = Elements(D.orbit_stab[o]);
        info[o].dc = DoubleCosets(B, info[o].stab);
        for (uint32_t j = 1; j <= B.m(); ++j) {
            if (info[o].dc[j] != j) continue;
            const Mat22 c = B.rep(j);
            const Mat22 ci = c.inv_unimodular();
            const CyclicGroup& G = D.orbit_stab[o];
            Mat22 x = G.gen;
            uint32_t i0 = 0;
            for (uint32_t i = 1; i <= G.order; ++i) {
                if (B.contains(c * x * ci)) {
                    i0 = i;
                    break;
                }
                x = x * G.gen;
            }
            ENGINE_CHECK_MSG(i0 >= 1 && G.order % i0 == 0, "orbifold_mp::SubgroupDescriptor: B-stabiliser not of index dividing |Gamma_v|");
            const uint32_t idx = static_cast<uint32_t>(S.D.orbit_stab.size());
            info[o].borb[j] = idx;
            S.D.orbit_stab.push_back(CyclicGroup{c * x * ci, G.order / i0});
            S.D.orbit_label.push_back(next_label);
            S.D.point[next_label++] = Point{idx, Mat22::I()};
        }
    }
    S.vertex_orbits = static_cast<uint32_t>(S.D.orbit_stab.size());
    // A Gamma-point (o, g) as a B-point: g u = b c with u in Gamma_o, c the double-coset representative.
    auto toB = [&](uint32_t o, const Mat22& g) -> Point {
        const OrbitInfo& I = info[o];
        const uint32_t jmin = I.dc[B.class_of(g)];
        for (const Mat22& u : I.stab) {
            const Mat22 gu = g * u;
            if (B.class_of(gu) != jmin) continue;
            const Mat22 b = gu * B.rep(jmin).inv_unimodular();
            ENGINE_CHECK_MSG(B.contains(b), "orbifold_mp::SubgroupDescriptor: a point element is not in B");
            return Point{I.borb.at(jmin), b};
        }
        ENGINE_CHECK_MSG(false, "orbifold_mp::SubgroupDescriptor: no u in Gamma_v reaches the double-coset representative");
        return Point{};
    };
    // Top B-orbits (sigma, r), r in B\Gamma/Gamma_sigma.
    for (uint32_t s = 0; s < D.top.size(); ++s) {
        const std::vector<Point> P = D.lifted(s);
        // the pointwise stabiliser of iota(sigma): the elements of Gamma_{p_0} fixing every vertex
        std::vector<Mat22> st;
        const Mat22 g0 = P[0].g, g0i = g0.inv_unimodular();
        for (const Mat22& u : Elements(D.orbit_stab[P[0].orbit])) {
            const Mat22 x = g0 * u * g0i;
            bool fixes = true;
            for (const Point& p : P) fixes = fixes && D.key(Point{p.orbit, x * p.g}) == D.key(p);
            if (fixes) st.push_back(x);
        }
        S.stab_sigma.push_back(st);
        const std::vector<uint32_t> dc = DoubleCosets(B, st);
        for (uint32_t j = 1; j <= B.m(); ++j) {
            if (dc[j] != j) continue;
            const Mat22 r = B.rep(j);
            std::vector<uint32_t> labels;
            for (const Point& p : P) {
                const uint32_t lab = next_label++;
                S.D.point[lab] = toB(p.orbit, r * p.g);
                labels.push_back(lab);
            }
            S.D.top.push_back(labels);
            S.sigma.push_back(s);
            S.r.push_back(r);
            S.r_class.push_back(j);
        }
    }
    ENGINE_CHECK_MSG(S.D.top.size() < 256, "orbifold_mp::SubgroupDescriptor: too many top simplices for InternedMPSet");
    return S;
}

bool SameSimplex(const Descriptor& D, const SubgroupModel& Bm, uint32_t t, const SubgroupModel& Am, uint32_t tA, const Mat22& h) {
    if (Bm.sigma.at(t) != Am.sigma.at(tA)) return false;
    std::set<PointKey> a, b;
    for (const Point& p : D.lifted(Bm.sigma[t])) b.insert(D.key(Point{p.orbit, Bm.r[t] * p.g}));
    for (const Point& p : D.lifted(Am.sigma[tA])) a.insert(D.key(Point{p.orbit, h * Am.r[tA] * p.g}));
    return a == b;
}

RefinementLabels ForgetCosetLabels(const Descriptor& D, const SubgroupModel& Bm, const CosetLevel& B, const SubgroupModel& Am,
                                   const CosetLevel& A, uint32_t* exact_out) {
    (void)B;
    RefinementLabels L;
    uint32_t exact = 0;
    for (uint32_t t = 0; t < Bm.D.top.size(); ++t) {
        const uint32_t s = Bm.sigma[t];
        bool found = false;
        for (uint32_t tA = 0; tA < Am.D.top.size() && !found; ++tA) {
            if (Am.sigma[tA] != s) continue;
            const uint32_t cA = A.class_of(Am.r[tA]);
            // prefer u = I (the first element of the stabiliser list is I: Elements() starts at I).  u = -I is needed only
            // when -I is not in A (else A r^B = A r^B (-I)); MP-3b's Gamma_1(8) -> Gamma_1(4) reaches it (F-MP3 F3, P7.9).
            for (const Mat22& u : Am.stab_sigma.at(s)) {
                const Mat22 ru = Bm.r[t] * u;
                if (A.class_of(ru) != cA) continue;
                const Mat22 h = ru * Am.r[tA].inv_unimodular();
                ENGINE_CHECK_MSG(A.contains(h), "orbifold_mp::ForgetCosetLabels: a label is not in A");
                ENGINE_CHECK_MSG(SameSimplex(D, Bm, t, Am, tA, h), "orbifold_mp::ForgetCosetLabels: iota_B != h iota_A");
                if (u == Mat22::I()) ++exact;
                L.coarse.push_back(tA);
                L.h.push_back(h);
                found = true;
                break;
            }
        }
        ENGINE_CHECK_MSG(found, "orbifold_mp::ForgetCosetLabels: no A-orbit contains the B-orbit");
    }
    if (exact_out) *exact_out = exact;
    return L;
}

RefinementLabels LabelsToGamma(const SubgroupModel& Bm) {
    RefinementLabels L;
    L.coarse = Bm.sigma;
    L.h = Bm.r;
    return L;
}

ModuleRep CoindRep(const CosetLevel& B, uint32_t n, uint32_t ell, uint32_t e) {
    const uint32_t d = n + 1, mB = B.m(), dim = mB * d, m = PowU(ell, e);
    const engine::ModCtx<uint32_t> ctx = RingCtx(ell, e);
    return ModuleRep("Coind_" + B.name() + " Sym^" + std::to_string(n), dim, ell, e, [&B, ctx, n, d, mB, dim, m](const Mat22& g) {
        Vec M(static_cast<size_t>(dim) * dim, 0);
        for (uint32_t j = 1; j <= mB; ++j) {
            const Mat22 x = B.rep(j) * g;
            const uint32_t jj = B.class_of(x);
            const Mat22 h = x * B.rep(jj).inv_unimodular();
            ENGINE_CHECK_MSG(B.contains(h), "orbifold_mp::CoindRep: h_j not in B");
            PutBlock(M, dim, j - 1, jj - 1, RhoSym(ctx, h, n, m), d, m);
        }
        return M;
    });
}

ResCor ResCorMaps(const CosetLevel& A, const CosetLevel& B, uint32_t n, uint32_t ell, uint32_t e, int64_t wrong_rep, bool drop_rho) {
    const uint32_t d = n + 1, mA = A.m(), mB = B.m(), m = PowU(ell, e);
    const engine::ModCtx<uint32_t> ctx = RingCtx(ell, e);
    ENGINE_CHECK_MSG(mB % mA == 0, "orbifold_mp::ResCorMaps: [Gamma : B] not a multiple of [Gamma : A]");
    ResCor R;
    R.index = mB / mA;
    R.res.assign(static_cast<size_t>(mB) * d * mA * d, 0);
    R.cor.assign(static_cast<size_t>(mA) * d * mB * d, 0);
    const Vec Id = IdentitySmall(d, 1, m);
    for (uint32_t i = 1; i <= mB; ++i) {
        Mat22 v = B.rep(i);
        const uint32_t a = A.class_of(v);
        if (static_cast<int64_t>(i) == wrong_rep) v = v * Mat22::T();
        const Mat22 h = v * A.rep(a).inv_unimodular();
        if (static_cast<int64_t>(i) != wrong_rep) ENGINE_CHECK_MSG(A.contains(h), "orbifold_mp::ResCorMaps: B not contained in A");
        R.fibre.push_back(a - 1);
        PutBlock(R.res, mA * d, i - 1, a - 1, drop_rho ? Id : RhoSym(ctx, h, n, m), d, m);
        PutBlock(R.cor, mB * d, a - 1, i - 1, drop_rho ? Id : RhoSym(ctx, h.inv_unimodular(), n, m), d, m);
    }
    return R;
}

Vec EvaluationMatrix(const CosetLevel& B, const Mat22& h, uint32_t n, uint32_t ell, uint32_t e) {
    const uint32_t d = n + 1, mB = B.m(), m = PowU(ell, e);
    const engine::ModCtx<uint32_t> ctx = RingCtx(ell, e);
    const uint32_t t = B.class_of(h);
    const Mat22 b = h * B.rep(t).inv_unimodular();
    ENGINE_CHECK_MSG(B.contains(b), "orbifold_mp::EvaluationMatrix: h w_t^{-1} not in B");
    Vec E(static_cast<size_t>(d) * mB * d, 0);
    const Vec R = RhoSym(ctx, b, n, m);
    for (uint32_t a = 0; a < d; ++a)
        for (uint32_t c = 0; c < d; ++c) E[static_cast<size_t>(a) * mB * d + (t - 1) * d + c] = R[a * d + c];
    return E;
}

Vec MatMulSmall(const Vec& A, const Vec& B, uint32_t r, uint32_t k, uint32_t c, uint32_t m) {
    ENGINE_CHECK_MSG(A.size() == static_cast<size_t>(r) * k && B.size() == static_cast<size_t>(k) * c, "orbifold_mp::MatMulSmall: shapes");
    Vec C(static_cast<size_t>(r) * c, 0);
    for (uint32_t i = 0; i < r; ++i)
        for (uint32_t t = 0; t < k; ++t) {
            const uint64_t a = A[static_cast<size_t>(i) * k + t];
            if (!a) continue;
            for (uint32_t j = 0; j < c; ++j)
                C[static_cast<size_t>(i) * c + j] = static_cast<uint32_t>((C[static_cast<size_t>(i) * c + j] + a * B[static_cast<size_t>(t) * c + j]) % m);
        }
    return C;
}

Vec IdentitySmall(uint32_t d, uint32_t scalar, uint32_t m) {
    Vec I(static_cast<size_t>(d) * d, 0);
    for (uint32_t a = 0; a < d; ++a) I[a * d + a] = scalar % m;
    return I;
}

}  // namespace orbifold_mp
