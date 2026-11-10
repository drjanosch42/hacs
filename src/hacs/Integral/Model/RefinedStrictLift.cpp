// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 18, WP-7: the strict lift on a synthetic refined descriptor.
// See RefinedStrictLift.h; [C, sec:isomodelell] code map item (v), [C, IL5];
// report docs/implementation/slot18/WP-7_report.md.
#include "RefinedStrictLift.h"

#include <algorithm>
#include <utility>

#include "PeriodicColumn.h"
#include "../../../ModArithmetic/EngineCheck.h"

namespace integral_model {

using hecke::Mat22;

namespace {

int parity(const Simplex& from, const Simplex& to) {
    ENGINE_CHECK(from.size() == to.size());
    std::vector<uint32_t> pos(from.size());
    for (size_t i = 0; i < to.size(); ++i) {
        const auto it = std::find(from.begin(), from.end(), to[i]);
        ENGINE_CHECK_MSG(it != from.end(), "RefinedStrictLift: not a permutation of the stored simplex");
        pos[i] = static_cast<uint32_t>(it - from.begin());
    }
    int sign = 1;
    for (size_t i = 0; i < pos.size(); ++i)
        for (size_t j = i + 1; j < pos.size(); ++j)
            if (pos[i] > pos[j]) sign = -sign;
    return sign;
}

Simplex sorted(Simplex s) {
    std::sort(s.begin(), s.end());
    return s;
}

// Every simplex of a model's K' by vertex set: (stored list, representative, transport g; stored = g . rep).
struct KLookup {
    std::map<Simplex, std::pair<Simplex, std::pair<uint32_t, Mat22>>> by_set;
    explicit KLookup(const IsotropyModel& M) {
        for (uint32_t s = 0; s < 3; ++s)
            for (uint32_t r = 0; r < M.reps(s).size(); ++r)
                by_set[sorted(M.reps(s)[r])] = {M.reps(s)[r], {r, Mat22::I()}};
        for (const auto& nr : M.non_representatives()) by_set[sorted(nr.face)] = {nr.face, {nr.rep, nr.g}};
    }
    // The simplex x (as an oriented vertex list) as +- g . rep.
    RepTerm find(const Simplex& x, int64_t coeff) const {
        const auto it = by_set.find(sorted(x));
        ENGINE_CHECK_MSG(it != by_set.end(), "BuildSubdivisionMaps: a simplex is not in K'");
        return RepTerm{it->second.second.first, it->second.second.second, coeff * parity(it->second.first, x)};
    }
};

std::vector<Mat22> powers(const IsotropyModel& M, uint32_t s, uint32_t rep) {
    std::vector<Mat22> p;
    Mat22 x = Mat22::I();
    for (uint32_t i = 0; i < M.stabiliser_order(s, rep); ++i) {
        p.push_back(x);
        x = x * M.stabiliser_generator(s, rep);
    }
    return p;
}

void canon_add(CanonChain& c, const std::pair<uint32_t, GKey>& k, const Rat& r) {
    auto& v = c[k];
    v = v + r;
    if (v.is_zero()) c.erase(k);
}

RepChain translate(const Mat22& g, const RepChain& x) {
    RepChain out;
    for (const auto& t : x) out.push_back({t.rep, g * t.g, t.coeff});
    return out;
}

RepChain apply_map(const RepChainMap& f, uint32_t s, const RepChain& x) {
    RepChain out;
    for (const auto& t : x)
        for (const auto& u : f[s].at(t.rep)) out.push_back({u.rep, t.g * u.g, hecke::checked_mul<int64_t>(t.coeff, u.coeff)});
    return out;
}

RepChain boundary(const IsotropyModel& M, uint32_t s, const RepChain& x) {
    RepChain out;
    for (const auto& t : x)
        for (const auto& [sign, ft] : M.faces(s, t.rep)) out.push_back({ft.rep, t.g * ft.g, t.coeff * sign});
    return out;
}

void chain_add(LChain& x, uint32_t cell, const QGroupRingElt& v, const Rat& scale = Rat::of(1)) {
    if (v.is_zero()) return;
    auto& slot = x[cell];
    slot.add(v, scale);
    if (slot.is_zero()) x.erase(cell);
}
void chain_add(LChain& x, const LChain& y, const Rat& scale = Rat::of(1)) {
    for (const auto& [c, v] : y) chain_add(x, c, v, scale);
}
bool chains_equal(const LChain& a, const LChain& b) {
    LChain d = a;
    chain_add(d, b, Rat::of(-1));
    for (const auto& [c, v] : d)
        if (!v.is_zero()) return false;
    return true;
}
LChain left_mult(const Mat22& u, const LChain& x) {
    LChain out;
    const QGroupRingElt U = QGroupRingElt::unit(u);
    for (const auto& [c, v] : x) chain_add(out, c, U * v);
    return out;
}

// The t = 0 section of an orbit chain of dst's descriptor: g . sigma_rep |-> g e'_{rep, 0}.
LChain section(const LocalModel& dst, uint32_t s, const RepChain& x) {
    LChain y;
    for (const auto& t : x) {
        if (t.coeff == 0) continue;
        const int64_t idx = dst.cell_index(s, t.rep, 0);
        ENGINE_CHECK_MSG(idx >= 0, "section: a t = 0 cell is missing");
        chain_add(y, static_cast<uint32_t>(idx), dst.normalise(s, static_cast<uint32_t>(idx), QGroupRingElt::unit(t.g)),
                  Rat::of(t.coeff));
    }
    return y;
}

}  // namespace

CanonChain Canonical(const IsotropyModel& M, uint32_t s, const RepChain& x) {
    CanonChain c;
    std::map<uint32_t, std::vector<Mat22>> pw;
    for (const auto& t : x) {
        if (t.coeff == 0) continue;
        auto it = pw.find(t.rep);
        if (it == pw.end()) it = pw.emplace(t.rep, powers(M, s, t.rep)).first;
        canon_add(c, {t.rep, decompose(key_of(t.g), it->second).first}, Rat::of(t.coeff));
    }
    return c;
}

// ---------------------------------------------------------------------------
// The subdivision chain maps
// ---------------------------------------------------------------------------
SubdivisionMaps BuildSubdivisionMaps(const IsotropyModel& X, const IsotropyModel& Y) {
    ENGINE_CHECK_MSG(X.data().collapse.empty() && X.data().refines.empty(),
                     "BuildSubdivisionMaps: X must be the root (unrefined) descriptor");
    const KLookup kx(X), ky(Y);
    const auto& col = Y.data().collapse;
    SubdivisionMaps m;
    for (uint32_t s = 0; s < 3; ++s) {
        m.collapse[s].resize(Y.reps(s).size());
        for (uint32_t r = 0; r < Y.reps(s).size(); ++r) {
            Simplex img;
            for (uint32_t v : Y.reps(s)[r]) {
                const auto it = col.find(v);
                img.push_back(it == col.end() ? v : it->second);
            }
            const Simplex srt = sorted(img);
            if (std::adjacent_find(srt.begin(), srt.end()) != srt.end()) continue;   // degenerate (a repeated vertex): 0
            m.collapse[s][r].push_back(kx.find(img, 1));
        }
        m.subdivide[s].resize(X.reps(s).size());
        for (uint32_t r = 0; r < X.reps(s).size(); ++r) {
            const Simplex& sig = X.reps(s)[r];
            const auto it = Y.data().refines.find(sig);
            if (it == Y.data().refines.end()) {
                m.subdivide[s][r].push_back(ky.find(sig, 1));
            } else {
                for (const auto& [sign, piece] : it->second) m.subdivide[s][r].push_back(ky.find(piece, sign));
            }
        }
    }
    return m;
}

SubdivisionCertificate CertifySubdivisionMaps(const IsotropyModel& X, const IsotropyModel& Y, const SubdivisionMaps& m) {
    SubdivisionCertificate cert;
    cert.collapse_chain_map = cert.subdivide_chain_map = cert.collapse_subdivide_id = true;
    auto fail = [&](const std::string& w) {
        if (cert.first_failure.empty()) cert.first_failure = w;
    };
    for (uint32_t s = 0; s < 3; ++s) {
        for (uint32_t r = 0; r < Y.reps(s).size(); ++r) {
            ++cert.checks;
            const RepChain sig{{r, Mat22::I(), 1}};
            if (s >= 1 && Canonical(X, s - 1, boundary(X, s, apply_map(m.collapse, s, sig))) !=
                              Canonical(X, s - 1, apply_map(m.collapse, s - 1, boundary(Y, s, sig)))) {
                cert.collapse_chain_map = false;
                fail("d c != c d at s=" + std::to_string(s) + " rep " + std::to_string(r));
            }
        }
        for (uint32_t r = 0; r < X.reps(s).size(); ++r) {
            ++cert.checks;
            const RepChain sig{{r, Mat22::I(), 1}};
            if (s >= 1 && Canonical(Y, s - 1, boundary(Y, s, apply_map(m.subdivide, s, sig))) !=
                              Canonical(Y, s - 1, apply_map(m.subdivide, s - 1, boundary(X, s, sig)))) {
                cert.subdivide_chain_map = false;
                fail("d sd != sd d at s=" + std::to_string(s) + " rep " + std::to_string(r));
            }
            if (Canonical(X, s, apply_map(m.collapse, s, apply_map(m.subdivide, s, sig))) != Canonical(X, s, sig)) {
                cert.collapse_subdivide_id = false;
                fail("c sd != id at s=" + std::to_string(s) + " rep " + std::to_string(r));
            }
        }
    }
    return cert;
}

// ---------------------------------------------------------------------------
// Comparison maps over a base chain map
// ---------------------------------------------------------------------------
ComparisonMap StrictComparisonOver(const LocalModel& src, const LocalModel& dst, const RepChainMap& base, LiftAveraging avg) {
    ENGINE_CHECK_MSG(src.q_max() == dst.q_max(), "StrictComparisonOver: the two models must share q_max");
    for (uint32_t s = 0; s < 3; ++s)
        ENGINE_CHECK_MSG(base[s].size() == src.num_reps(s), "StrictComparisonOver: the base map is not over src's representatives");
    ComparisonMap f;
    f.value.assign(src.q_top() + 1, {});
    for (uint32_t q = 0; q <= src.q_top(); ++q) {
        f.value[q].assign(src.rank(q), {});
        for (uint32_t xi = 0; xi < src.rank(q); ++xi) {
            const LCell& x = src.cells(q)[xi];
            LChain y;
            if (x.t == 0) y = section(dst, x.s, base[x.s][x.rep]);
            if (q >= 1) {
                LChain z;
                for (const LEntry& e : src.differential(q)) {
                    if (e.col != xi) continue;
                    for (const auto& [c, v] : f.value[q - 1][e.row]) chain_add(z, c, e.coeff * v);
                }
                chain_add(z, dst.apply_d(q, y), Rat::of(-1));
                bool zero = true;
                for (const auto& [c, v] : z) zero = zero && v.is_zero();
                if (!zero) chain_add(y, dst.contract(q - 1, z));
            }
            ++f.stats.lifts;
            const ColumnSpec& cs = src.column(x.s, x.rep);
            if (avg == LiftAveraging::On && cs.q_order > 1) {
                LChain ya;
                for (const auto& [c, v] : y) chain_add(ya, c, cs.e * v);
                if (!chains_equal(ya, y)) ++f.stats.changed_by_average;
                y = std::move(ya);
            }
            f.value[q][xi] = std::move(y);
        }
    }
    return f;
}

ComparisonCertificate CertifyComparisonOver(const LocalModel& src, const LocalModel& dst, const RepChainMap& base,
                                            const ComparisonMap& f) {
    ComparisonCertificate cert;
    cert.chain_map = cert.strict = cert.well_defined = true;
    auto fail = [&](const std::string& why) {
        if (cert.first_failure.empty()) cert.first_failure = why;
    };
    for (uint32_t q = 0; q <= src.q_top(); ++q)
        for (uint32_t xi = 0; xi < src.rank(q); ++xi) {
            ++cert.cells_checked;
            const LCell& x = src.cells(q)[xi];
            const LChain& fx = f.value.at(q).at(xi);
            const ColumnSpec& cs = src.column(x.s, x.rep);
            LChain ea;
            for (const auto& [c, v] : fx) chain_add(ea, c, cs.e * v);
            if (!chains_equal(ea, fx)) {
                cert.well_defined = false;
                ++cert.not_well_defined;
                fail("f(x) not Q_x-invariant at degree " + std::to_string(q) + " cell " + std::to_string(xi));
            }
            const CanonChain want = (x.t == 0) ? Canonical(dst.descriptor(), x.s, base[x.s][x.rep]) : CanonChain{};
            if (dst.augment(q, fx) != want) {
                cert.strict = false;
                fail("eps_dst f != base eps_src at degree " + std::to_string(q) + " cell " + std::to_string(xi));
            }
            if (q >= 1) {
                LChain rhs;
                for (const LEntry& e : src.differential(q)) {
                    if (e.col != xi) continue;
                    for (const auto& [c, v] : f.value[q - 1][e.row]) chain_add(rhs, c, e.coeff * v);
                }
                if (!chains_equal(dst.apply_d(q, fx), rhs)) {
                    cert.chain_map = false;
                    fail("d f != f d at degree " + std::to_string(q) + " cell " + std::to_string(xi));
                }
            }
        }
    return cert;
}

// ---------------------------------------------------------------------------
// The strict lift on the refined model
// ---------------------------------------------------------------------------
RefinedStrictLift::RefinedStrictLift(const LocalModel& A, const StrictLift& tau, const SubdivisionMaps& maps)
    : A_(&A), tau_(&tau), maps_(&maps) {
    ENGINE_CHECK_MSG(!tau.central(), "RefinedStrictLift: the double coset diag(1, p^nu) only");
    const IsotropyModel& X = tau.model();
    const IsotropyModel& Y = A.descriptor();
    for (uint32_t s = 0; s < 3; ++s)
        ENGINE_CHECK_MSG(maps.collapse[s].size() == Y.reps(s).size() && maps.subdivide[s].size() == X.reps(s).size(),
                         "RefinedStrictLift: the subdivision maps are not between the lift's model and A's descriptor");
    ENGINE_CHECK_MSG(A.q_top() <= tau.q_top(), "RefinedStrictLift: A is built beyond the shipped lift's degrees");
    m_ = tau.num_cosets();
    build();
}

LChain RefinedStrictLift::tilde(uint32_t q, const Mat22& gamma, uint32_t cell) const {
    uint32_t j = 0;
    const Mat22 u = tau_->twisted_decomposition(gamma, j);   // gamma = h s_j, u = g^{-1} h g
    return left_mult(u, raw_.at(q).at(j - 1).at(cell));
}

LChain RefinedStrictLift::apply(uint32_t q, const LChain& x) const {
    LChain out;
    for (const auto& [cell, a] : x)
        for (const auto& [g, r] : a.terms) chain_add(out, tilde(q, mat_of(g), cell), r);
    return out;
}

LChain RefinedStrictLift::value(uint32_t q, uint32_t j, uint32_t cell) const {
    LChain gen;
    gen[cell] = A_->normalise(q, cell, QGroupRingElt::unit(tau_->coset_representative(j)));
    return apply(q, gen);
}

namespace {
// f(g . sigma) = sd o tau^s o c (g . sigma) as an orbit chain of Y.
RepChain base_chain(const StrictLift& tau, const SubdivisionMaps& maps, uint32_t s, uint32_t rep, const Mat22& g) {
    const IsotropyModel& X = tau.model();
    IsoChain x(X.rank(s));
    for (const auto& t : translate(g, maps.collapse[s].at(rep))) x.c[X.cell_index(s, t.rep, 0)].add(key_of(t.g), t.coeff);
    const IsoChain y = tau.apply(s, x);
    RepChain xs;   // eps of tau_Iso(x) = tau^s(c(g . sigma)) (the shipped lift's (S) identity)
    for (uint32_t ci = 0; ci < X.rank(s); ++ci) {
        const Cell& c = X.cells(s)[ci];
        if (c.t != 0) continue;
        for (const auto& [gk, mult] : y.c[ci].terms) xs.push_back({c.rep, mat_of(gk), mult});
    }
    return apply_map(maps.subdivide, s, xs);
}
}  // namespace

CanonChain RefinedStrictLift::base(uint32_t s, uint32_t rep, const Mat22& g) const {
    return Canonical(A_->descriptor(), s, base_chain(*tau_, *maps_, s, rep, g));
}

LChain RefinedStrictLift::base_section(uint32_t s, uint32_t rep, const Mat22& g) const {
    return section(*A_, s, base_chain(*tau_, *maps_, s, rep, g));
}

void RefinedStrictLift::build() {
    const LocalModel& A = *A_;
    raw_.assign(A.q_top() + 1, {});
    for (uint32_t q = 0; q <= A.q_top(); ++q) {
        raw_[q].assign(m_, std::vector<LChain>(A.rank(q)));
        for (uint32_t c = 0; c < A.rank(q); ++c) {
            const LCell& cell = A.cells(q)[c];
            for (uint32_t j = 1; j <= m_; ++j) {
                const Mat22 sj = tau_->coset_representative(j);
                LChain y;
                if (cell.t == 0) y = base_section(cell.s, cell.rep, sj);
                if (q >= 1) {
                    LChain gen;
                    gen[c] = A.normalise(q, c, QGroupRingElt::unit(sj));
                    LChain z = apply(q - 1, A.apply_d(q, gen));
                    chain_add(z, A.apply_d(q, y), Rat::of(-1));
                    if (!z.empty()) chain_add(y, A.contract(q - 1, z));
                }
                raw_[q][j - 1][c] = std::move(y);
            }
        }
    }
}

RefinedLiftCertificate RefinedStrictLift::certify() const {
    const LocalModel& A = *A_;
    RefinedLiftCertificate cert;
    cert.t1_twisted_equivariance = cert.t2_chain_map = cert.t3_augmentation = cert.strict = true;
    auto fail = [&](const std::string& w) {
        if (cert.first_failure.empty()) cert.first_failure = w;
    };
    cert.terms.assign(q_top() + 1, 0);
    std::vector<std::vector<std::vector<LChain>>> val(q_top() + 1);
    for (uint32_t q = 0; q <= q_top(); ++q) {
        val[q].assign(m_, std::vector<LChain>(A.rank(q)));
        for (uint32_t c = 0; c < A.rank(q); ++c)
            for (uint32_t j = 1; j <= m_; ++j) {
                LChain v = value(q, j, c);
                if (!chains_equal(v, raw(q, j, c))) ++cert.changed_by_average;
                for (const auto& [cc, a] : v)
                    for (const auto& [g, r] : a.terms) {
                        ++cert.terms[q];
                        cert.max_abs_numerator = std::max(cert.max_abs_numerator, r.num < 0 ? -r.num : r.num);
                        cert.max_denominator = std::max(cert.max_denominator, r.den);
                    }
                val[q][j - 1][c] = std::move(v);
            }
    }
    for (uint32_t q = 0; q <= q_top(); ++q)
        for (uint32_t c = 0; c < A.rank(q); ++c) {
            const LCell& cell = A.cells(q)[c];
            for (uint32_t j = 1; j <= m_; ++j) {
                const LChain& v = val[q][j - 1][c];
                const Mat22 sj = tau_->coset_representative(j);
                LChain gen;
                gen[c] = A.normalise(q, c, QGroupRingElt::unit(sj));
                const std::string at = " q=" + std::to_string(q) + " cell=" + std::to_string(c) + " j=" + std::to_string(j);
                if (q >= 1) {
                    ++cert.t2_checks;
                    if (!chains_equal(A.apply_d(q, v), apply(q - 1, A.apply_d(q, gen)))) {
                        cert.t2_chain_map = false;
                        fail("(T2) d tau' != tau' d" + at);
                    }
                }
                if (q == 0) {
                    ++cert.t3_checks;
                    Rat sum = Rat::of(0);
                    for (const auto& [k, r] : A.augment(0, v)) sum = sum + r;
                    if (sum != Rat::of(1)) {
                        cert.t3_augmentation = false;
                        fail("(T3) aug eps' tau' != aug eps'" + at);
                    }
                }
                if (q <= 2) {
                    ++cert.strict_checks;
                    const CanonChain want = (cell.t == 0) ? base(cell.s, cell.rep, sj) : CanonChain{};
                    if (A.augment(q, v) != want) {
                        cert.strict = false;
                        fail("(S) eps' tau' != f eps'" + at);
                    }
                }
            }
        }
    // (T1) on the Schreier generators of Gamma_0(N), the conjugate formed independently of the helper.
    const int64_t N = static_cast<int64_t>(tau_->N());
    const Mat22 g = tau_->g();
    ENGINE_CHECK_MSG(g.a == 1 && g.b == 0 && g.c == 0 && g.d == N, "RefinedStrictLift: g = diag(1, N) expected");
    std::vector<std::pair<Mat22, Mat22>> gens;   // (h, g^{-1} h g)
    for (uint32_t i = 1; i <= m_; ++i)
        for (const Mat22& x : {Mat22::T(), Mat22::S()}) {
            const Mat22 gamma = tau_->coset_representative(i) * x;
            uint32_t j2 = 0;
            const Mat22 u = tau_->twisted_decomposition(gamma, j2);
            const Mat22 h = gamma * tau_->coset_representative(j2).inv_unimodular();
            if (h == Mat22::I() || h == Mat22::I().neg()) continue;
            bool ok = h.det() == 1 && h.c % N == 0;
            const Mat22 conj{h.a, hecke::checked_mul<int64_t>(N, h.b), ok ? h.c / N : 0, h.d};
            ok = ok && conj == u;
            if (!ok) {
                cert.t1_twisted_equivariance = false;
                fail("(T1) a Schreier generator outside Gamma_0(N), or the helper's conjugate differs");
                continue;
            }
            bool dup = false;
            for (const auto& [h2, u2] : gens) dup = dup || h2 == h;
            if (!dup) gens.push_back({h, conj});
        }
    cert.schreier_generators = static_cast<uint32_t>(gens.size());
    for (const auto& [h, conj] : gens)
        for (uint32_t q = 0; q <= q_top(); ++q)
            for (uint32_t c = 0; c < A.rank(q); ++c)
                for (uint32_t j = 1; j <= m_; ++j) {
                    ++cert.t1_checks;
                    LChain gen;
                    gen[c] = A.normalise(q, c, QGroupRingElt::unit(h * tau_->coset_representative(j)));
                    if (!chains_equal(apply(q, gen), left_mult(conj, val[q][j - 1][c]))) {
                        cert.t1_twisted_equivariance = false;
                        fail("(T1) tau'(h y) != (g^-1 h g) tau'(y) at q=" + std::to_string(q) + " cell=" +
                             std::to_string(c) + " j=" + std::to_string(j));
                    }
                }
    return cert;
}

void RefinedStrictLift::perturb(uint32_t q, uint32_t j, uint32_t cell) {
    chain_add(raw_.at(q).at(j - 1).at(cell), cell, A_->normalise(q, cell, QGroupRingElt::unit(Mat22::I())));
}

template<class Elt>
engine::DenseBuf<Elt> RefinedStrictLift::theta_hat(const engine::ModCtx<Elt>& ctx, const CoefficientModule<Elt>& M,
                                                   uint32_t q) const {
    ENGINE_CHECK_MSG(q <= q_top(), "RefinedStrictLift::theta_hat: q > q_top");
    const uint32_t D = M.dim;
    const uint32_t w = A_->rank(q);
    engine::DenseBuf<Elt> out(w * D, w * D);
    std::map<GKey, engine::DenseBuf<Elt>> memo;
    for (uint32_t j = 1; j <= m_; ++j) {
        const Mat22 delta = tau_->coset_delta_matrix(j);
        for (uint32_t c = 0; c < w; ++c) {
            const LChain v = value(q, j, c);
            for (const auto& [c2, a] : v) {
                QGroupRingElt dw;
                for (const auto& [g, r] : a.terms) dw.add(key_of(delta * mat_of(g)), r);   // exact, det N, never inverted
                const auto R = RhoQ(ctx, dw, M, memo);
                for (uint32_t i = 0; i < D; ++i)
                    for (uint32_t k = 0; k < D; ++k) {
                        Elt& x = out.at(c * D + i, c2 * D + k);
                        x = ctx.add(x, R.at(i, k));
                    }
            }
        }
    }
    return out;
}

template engine::DenseBuf<uint32_t> RefinedStrictLift::theta_hat<uint32_t>(const engine::ModCtx<uint32_t>&,
                                                                          const CoefficientModule<uint32_t>&, uint32_t)
    const;

}  // namespace integral_model
