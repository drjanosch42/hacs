// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 22, P10 (a)) The wild character models -- see HeckeNebentypusWild.h and
// docs/implementation/slot22/P10_spec.md.
#include "HeckeNebentypusWild.h"

#include <algorithm>
#include <map>
#include <stdexcept>

#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/ScalarMod.h"

namespace hecke {
namespace level {
namespace wild {

namespace {

using engine::scalar::gcd;
using engine::scalar::pow_wrap;


// <d>_ell = u^{L} mod ell^r, L in [0, ell^{r-1})
uint64_t LogUmod(uint64_t d, uint32_t ell, uint32_t r) {
    const ZK Rr(ell, r);
    const uint32_t om = Rr.teich(d);
    const uint32_t x = Rr.mul(static_cast<uint32_t>(d % Rr.Q), Rr.inv(om));
    const uint64_t L = pow_wrap(ell, r - 1);
    uint32_t cur = static_cast<uint32_t>(1 % Rr.Q);
    for (uint64_t s = 0; s < L; ++s) {
        if (cur == x) return s;
        cur = Rr.mul(cur, static_cast<uint32_t>((1 + ell) % Rr.Q));
    }
    throw std::logic_error("wild LogUmod: <d> is not a power of u");
}

class ExpandedHeilbronn final : public tower::ChiOperator {
public:
    ExpandedHeilbronn(const GammaHLevelN& G, uint64_t n, const std::vector<uint32_t>& idt, const WildRing& O, const WildCharacter& chi)
        : base_(G, n, idt, ZK(O.ell, 1)), O_(O), chi_(&chi), m_(G.m()) {}
    uint32_t m() const override { return m_ * O_.phi; }
    std::string label() const override { return base_.label(); }
    void image(uint32_t c1, std::vector<std::pair<uint32_t, uint32_t>>& out) const override {
        const uint32_t col = c1 - 1, J = col / O_.phi + 1, i = col % O_.phi;
        std::vector<std::pair<uint32_t, uint32_t>> img;
        base_.image(J, img);
        const OElem zi = O_.zpow(i);
        for (const auto& e : img) {
            const OElem s = O_.mul(chi_->table[e.second], zi);
            for (uint32_t k = 0; k < O_.phi; ++k)
                if (s[k]) out.push_back({(e.first - 1) * O_.phi + k + 1, s[k]});
        }
    }
private:
    tower::ChiHeilbronn base_;
    WildRing O_;
    const WildCharacter* chi_;
    uint32_t m_;
};

class ExpandedScalar final : public tower::ChiOperator {
public:
    ExpandedScalar(uint32_t m, const WildRing& O, const OElem& s) : m_(m), O_(O), s_(s) {}
    uint32_t m() const override { return m_ * O_.phi; }
    std::string label() const override { return "scalar"; }
    void image(uint32_t c1, std::vector<std::pair<uint32_t, uint32_t>>& out) const override {
        const uint32_t col = c1 - 1, J = col / O_.phi + 1, i = col % O_.phi;
        const OElem v = O_.mul(s_, O_.zpow(i));
        for (uint32_t k = 0; k < O_.phi; ++k)
            if (v[k]) out.push_back({(J - 1) * O_.phi + k + 1, v[k]});
    }
private:
    uint32_t m_;
    WildRing O_;
    OElem s_;
};

} // namespace

// ============================================================================
// the ring
// ============================================================================
OElem WildRing::one() const { OElem a(phi, 0); a[0] = static_cast<uint32_t>(1 % R.Q); return a; }
OElem WildRing::scalar(uint32_t c) const { OElem a(phi, 0); a[0] = static_cast<uint32_t>(c % R.Q); return a; }
OElem WildRing::add(const OElem& a, const OElem& b) const { OElem c(phi); for (uint32_t i = 0; i < phi; ++i) c[i] = R.add(a[i], b[i]); return c; }
OElem WildRing::sub(const OElem& a, const OElem& b) const { OElem c(phi); for (uint32_t i = 0; i < phi; ++i) c[i] = R.sub(a[i], b[i]); return c; }
OElem WildRing::mul(const OElem& a, const OElem& b) const {
    std::vector<uint64_t> p(2 * phi - 1, 0);
    for (uint32_t i = 0; i < phi; ++i) {
        if (!a[i]) continue;
        for (uint32_t k = 0; k < phi; ++k) p[i + k] = (p[i + k] + static_cast<uint64_t>(a[i]) * b[k]) % R.Q;
    }
    for (uint32_t n = 2 * phi - 1; n-- > phi;) {
        const uint64_t c = p[n] % R.Q;
        if (!c) continue;
        // z^n = z^{n - phi} z^phi, z^phi = -(Phi - z^phi)
        for (uint32_t k = 0; k < phi; ++k) p[n - phi + k] = (p[n - phi + k] + (R.Q - c) * Phi[k]) % R.Q;
        p[n] = 0;
    }
    OElem out(phi);
    for (uint32_t i = 0; i < phi; ++i) out[i] = static_cast<uint32_t>(p[i] % R.Q);
    return out;
}
OElem WildRing::zpow(uint64_t n) const {
    OElem r = one(), b(phi, 0);
    if (phi == 1) { b[0] = static_cast<uint32_t>(1 % R.Q); }   // z = 1 in Z_ell[z]/(z - 1) is not used (j >= 1: phi >= ell - 1 >= 4)
    else b[1] = 1;
    while (n) {
        if (n & 1) r = mul(r, b);
        b = mul(b, b);
        n >>= 1;
    }
    return r;
}
uint32_t WildRing::mod_pi(const OElem& a) const {
    uint64_t s = 0;
    for (uint32_t x : a) s += x;
    return static_cast<uint32_t>(s % ell);
}

WildRing MakeWildRing(uint32_t ell, uint32_t j, uint32_t e) {
    ENGINE_CHECK_MSG(ell >= 5 && j >= 1 && j <= 3, "MakeWildRing: ell >= 5, j in [1, 3]");
    WildRing O;
    O.ell = ell;
    O.j = j;
    O.phi = static_cast<uint32_t>(pow_wrap(ell, j - 1) * (ell - 1));
    O.e = e;
    if (e == 0 || e % O.phi != 0 || e / O.phi > 3)
        throw std::logic_error("wild_precision: e = " + std::to_string(e) + " is not K phi_j with K in [1, 3] (phi_j = " +
                               std::to_string(O.phi) + "; other e are not built)");
    O.K = e / O.phi;
    O.R = ZK(ell, O.K);
    O.Phi.assign(O.phi + 1, 0);
    const uint64_t step = pow_wrap(ell, j - 1);
    for (uint32_t k = 0; k < ell; ++k) O.Phi[k * step] = 1;
    return O;
}

WildCharacter MakeWildCharacter(const WildRing& O, uint64_t N, uint32_t i, uint64_t t) {
    uint64_t x = N;
    uint32_t r = 0;
    while (x % O.ell == 0) { x /= O.ell; ++r; }
    if (x != 1 || r == 0)
        throw std::logic_error("wild_tame_level: N = " + std::to_string(N) + " is not a power of ell = " + std::to_string(O.ell) +
                               " (tame level N > 1 is P11, slot 23)");
    const uint64_t Lr = pow_wrap(O.ell, r - 1), lj = pow_wrap(O.ell, O.j);
    if ((t * Lr) % lj != 0)
        throw std::logic_error("wild_ring_order: ell^j = " + std::to_string(lj) + " does not divide t ell^{r-1} = " + std::to_string(t * Lr) +
                               " (zeta^{t L(d)} is not well defined on (Z/ell^r)^x)");
    WildCharacter c;
    c.r = r;
    c.i = i;
    c.t = t;
    c.N = N;
    c.table.assign(N, O.zero());
    for (uint64_t d = 1; d < N; ++d) {
        if (gcd<uint64_t>(d, N) != 1) continue;
        const uint32_t om = O.R.pow(O.R.teich(d), i % (O.ell - 1));
        const uint64_t L = LogUmod(d, O.ell, r);
        c.table[d] = O.mul(O.scalar(om), O.zpow(t * L % lj));
    }
    return c;
}

uint32_t SpanLength(uint32_t D, const std::vector<SparseVec>& gens, const ZK& R, std::vector<uint32_t>* tors_pval) {
    const tower::ChainCokernel Q(D, gens, R, 0, true);
    uint32_t len = R.K * static_cast<uint32_t>(Q.pivots().size());
    const auto& T = Q.torsion_relations();
    if (!T.empty()) {
        std::map<uint32_t, uint32_t> cmap;
        for (const SparseVec& v : T) for (uint32_t c : v.col) cmap.emplace(c, 0);
        uint32_t k = 0;
        for (auto& kv : cmap) kv.second = k++;
        std::vector<std::vector<uint32_t>> rows;
        for (const SparseVec& v : T) {
            std::vector<uint32_t> row(k, 0);
            for (size_t i = 0; i < v.col.size(); ++i) row[cmap[v.col[i]]] = v.val[i];
            rows.push_back(std::move(row));
        }
        const tower::HowellZK H(R, std::move(rows), k);
        len += H.length();
        if (tors_pval) *tors_pval = H.pivot_valuations();
    } else if (tors_pval) {
        tors_pval->clear();
    }
    return len;
}

// ============================================================================
// the model
// ============================================================================
WildModel::WildModel(const WildRing& O, const WildCharacter& chi) : O_(O), chi_(chi) {
    const uint64_t N = chi.N;
    std::vector<uint64_t> units;
    for (uint64_t u = 1; u < N; ++u) if (gcd<uint64_t>(u, N) == 1) units.push_back(u);
    G_ = std::make_unique<GammaHLevelN>(N, units);   // H = G_N: Gamma_0(N)
    ENGINE_CHECK_MSG(G_->index_H() == 1, "WildModel: H = G_N");
    idt_.resize(N);
    for (uint64_t x = 0; x < N; ++x) idt_[x] = static_cast<uint32_t>(x);
    const tower::MonomialChi MS = tower::MonomialChiOf(*G_, Mat22::S(), idt_), MU = tower::MonomialChiOf(*G_, Mat22::ST(), idt_);
    const uint32_t m = G_->m(), phi = O.phi;
    std::vector<SparseVec> gS, gU;
    auto norm = [&](const tower::MonomialChi& M, uint32_t ord, std::vector<SparseVec>& out) {
        for (uint32_t Kc = 1; Kc <= m; ++Kc) {
            std::map<uint32_t, OElem> acc;   // class -> coefficient
            uint32_t cur = Kc;
            OElem c = O.one();
            for (uint32_t k = 0; k < ord; ++k) {
                auto it = acc.find(cur);
                if (it == acc.end()) acc[cur] = c; else it->second = O.add(it->second, c);
                const uint32_t J = M.src[cur];
                c = O.mul(c, chi.table[M.scal[J]]);
                cur = J;
            }
            ENGINE_CHECK_MSG(cur == Kc && c == O.one(), "WildModel: M(gamma)^ord != 1 (chi(-1) != 1?)");
            for (uint32_t i = 0; i < phi; ++i) {
                const OElem zi = O.zpow(i);
                SparseVec v;
                for (const auto& kv : acc) {
                    const OElem s = O.mul(kv.second, zi);
                    for (uint32_t k = 0; k < phi; ++k)
                        if (s[k]) { v.col.push_back((kv.first - 1) * phi + k); v.val.push_back(s[k]); }
                }
                if (!v.col.empty()) out.push_back(std::move(v));
            }
        }
    };
    norm(MS, 2, gS);
    norm(MU, 3, gU);
    gens_ = gS;
    gens_.insert(gens_.end(), gU.begin(), gU.end());
    const uint32_t D = m * phi;
    len_rel_ = SpanLength(D, gens_, O.R, &tors_pval_);
    const uint32_t lenS = SpanLength(D, gS, O.R), lenU = SpanLength(D, gU, O.R);
    len_H1_ = O.K * D - len_rel_;
    len_H0_ = lenS + lenU - len_rel_;
    Q_ = std::make_unique<tower::ChainCokernel>(D, gens_, O.R, 0, true);
}

std::unique_ptr<tower::ChiOperator> WildModel::heilbronn(uint64_t n) const {
    return std::make_unique<ExpandedHeilbronn>(*G_, n, idt_, O_, chi_);
}
std::unique_ptr<tower::ChiOperator> WildModel::scalar(const OElem& s) const { return std::make_unique<ExpandedScalar>(G_->m(), O_, s); }

} // namespace wild
} // namespace level
} // namespace hecke
