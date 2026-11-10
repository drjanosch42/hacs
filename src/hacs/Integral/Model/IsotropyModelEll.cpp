// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 16, WP-16 -- the ell-local isotropy model of
// [C, def:isomodelell].  See IsotropyModelEll.h for the contract.
#include "IsotropyModelEll.h"

#include <algorithm>
#include <numeric>
#include <sstream>
#include <utility>

#include "../../../ModArithmetic/DenseElim.h"
#include "../../../ModArithmetic/DenseView.h"
#include "../../../ModArithmetic/EngineCheck.h"
#include "../../../ModArithmetic/ScalarMod.h"
#include "PeriodicColumn.h"

namespace integral_model {

using hecke::Mat22;

// ---------------------------------------------------------------------------
// Rationals
// ---------------------------------------------------------------------------
namespace {
int64_t gcd_abs(int64_t a, int64_t b) {   // (S24 CL-G3, C-07) ScalarMod.h's gcd on the magnitudes
    return static_cast<int64_t>(engine::scalar::gcd<uint64_t>(hecke::uabs64(a), hecke::uabs64(b)));
}
}  // namespace

Rat Rat::of(int64_t n, int64_t d) {
    ENGINE_CHECK_MSG(d != 0, "Rat: zero denominator");
    if (n == 0) return Rat{0, 1};
    if (d < 0) {
        n = hecke::checked_mul<int64_t>(n, -1);
        d = hecke::checked_mul<int64_t>(d, -1);
    }
    const int64_t g = gcd_abs(n, d);
    return Rat{n / g, d / g};
}

Rat operator+(const Rat& a, const Rat& b) {
    if (a.num == 0) return b;
    if (b.num == 0) return a;
    const int64_t g = gcd_abs(a.den, b.den);
    const int64_t den = hecke::checked_mul<int64_t>(a.den / g, b.den);
    const int64_t num = hecke::checked_add<int64_t>(hecke::checked_mul<int64_t>(a.num, b.den / g),
                                                    hecke::checked_mul<int64_t>(b.num, a.den / g));
    return Rat::of(num, den);
}

Rat operator*(const Rat& a, const Rat& b) {
    if (a.num == 0 || b.num == 0) return Rat{0, 1};
    const int64_t g1 = gcd_abs(a.num, b.den), g2 = gcd_abs(b.num, a.den);
    return Rat::of(hecke::checked_mul<int64_t>(a.num / g1, b.num / g2),
                   hecke::checked_mul<int64_t>(a.den / g2, b.den / g1));
}

Rat operator-(const Rat& a) { return Rat{hecke::checked_mul<int64_t>(a.num, -1), a.den}; }

// ---------------------------------------------------------------------------
// Q[Gamma]
// ---------------------------------------------------------------------------
QGroupRingElt QGroupRingElt::unit(const Mat22& g, Rat r) {
    QGroupRingElt x;
    x.add(key_of(g), r);
    return x;
}

QGroupRingElt QGroupRingElt::from(const GroupRingElt& z) {
    QGroupRingElt x;
    for (const auto& [g, n] : z.terms) x.add(g, Rat::of(n));
    return x;
}

void QGroupRingElt::add(const GKey& g, const Rat& r) {
    if (r.is_zero()) return;
    auto it = terms.find(g);
    if (it == terms.end()) {
        terms.emplace(g, r);
        return;
    }
    it->second = it->second + r;
    if (it->second.is_zero()) terms.erase(it);
}

void QGroupRingElt::add(const QGroupRingElt& o, const Rat& scale) {
    if (scale.is_zero()) return;
    for (const auto& [g, r] : o.terms) add(g, r * scale);
}

QGroupRingElt QGroupRingElt::operator*(const QGroupRingElt& o) const {
    QGroupRingElt out;
    for (const auto& [g1, r1] : terms) {
        const Mat22 m1 = mat_of(g1);
        for (const auto& [g2, r2] : o.terms) out.add(key_of(m1 * mat_of(g2)), r1 * r2);
    }
    return out;
}

int64_t QGroupRingElt::max_den() const {
    int64_t m = 1;
    for (const auto& [g, r] : terms) m = std::max(m, r.den);
    return m;
}

std::string QGroupRingElt::to_string() const {
    if (terms.empty()) return "0";
    std::ostringstream os;
    bool first = true;
    for (const auto& [g, r] : terms) {
        if (!first) os << " + ";
        first = false;
        os << r.num;
        if (r.den != 1) os << "/" << r.den;
        os << "*[" << g[0] << "," << g[1] << ";" << g[2] << "," << g[3] << "]";
    }
    return os.str();
}

// ---------------------------------------------------------------------------
// Columns ([C, def:isomodelell])
// ---------------------------------------------------------------------------
QGroupRingElt ColumnCoefficient(const ColumnSpec& c, uint32_t t) {
    ENGINE_CHECK_MSG(t >= 1 && c.periodic(), "ColumnCoefficient: only a periodic column has t >= 1");
    QGroupRingElt x;
    if (t % 2 == 1) {
        x.add(key_of(c.pow[1 % c.m]), Rat::of(1));
        x.add(key_of(c.pow[0]), Rat::of(-1));
    } else {
        for (uint32_t i = 0; i < c.p_order; ++i) x.add(key_of(c.pow[i]), Rat::of(1));
    }
    return x * c.e;
}

QGroupRingElt ColumnHomotopyEll(const ColumnSpec& c, const QGroupRingElt& a, uint32_t t) {
    // Coordinates of a e in the R-basis c0 g^i e (c0 a canonical left coset
    // representative of Gamma_sigma, 0 <= i < |P|): g^{|P|} lies in Q, so
    // g^j e = g^{j mod |P|} e.
    std::map<GKey, std::map<uint32_t, Rat>> coords;
    for (const auto& [g, r] : a.terms) {
        const auto [c0, j] = decompose(g, c.pow);
        auto& v = coords[c0][j % c.p_order];
        v = v + r;
    }
    QGroupRingElt out;
    for (const auto& [c0, row] : coords) {
        const Mat22 C = mat_of(c0);
        for (const auto& [i, r] : row) {
            if (r.is_zero()) continue;
            if (t % 2 == 0) {
                for (uint32_t i2 = 0; i2 < i; ++i2) out.add(key_of(C * c.pow[i2]), r);
            } else if (i + 1 == c.p_order) {
                out.add(c0, r);
            }
        }
    }
    return out * c.e;
}

namespace {

// Is a e (a column element of degree t) a cycle of the augmented column?
bool column_cycle(const ColumnSpec& c, const QGroupRingElt& a, uint32_t t) {
    if (t >= 1) return (a * ColumnCoefficient(c, t)).is_zero();
    std::map<GKey, Rat> cosets;
    for (const auto& [g, r] : a.terms) {
        auto& v = cosets[decompose(g, c.pow).first];
        v = v + r;
    }
    for (const auto& [k, v] : cosets)
        if (!v.is_zero()) return false;
    return true;
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

bool chain_is_zero(const LChain& x) {
    for (const auto& [c, v] : x)
        if (!v.is_zero()) return false;
    return true;
}

bool chains_equal(const LChain& a, const LChain& b) {
    LChain d = a;
    chain_add(d, b, Rat::of(-1));
    return chain_is_zero(d);
}

}  // namespace

// ---------------------------------------------------------------------------
// LocalModel
// ---------------------------------------------------------------------------
LocalModel::LocalModel(const IsotropyModel& desc, uint32_t ell, LiftAveraging avg)
    : desc_(&desc), ell_(ell), q_max_(desc.q_max()), avg_(avg) {
    // (slot-16 WP-17, audit WP-16 F-5) a prime, not merely >= 2: ell = 4 or 6 would build a wrong "Sylow" subgroup.
    auto is_prime = [](uint32_t x) {
        if (x < 2) return false;
        for (uint32_t d = 2; static_cast<uint64_t>(d) * d <= x; ++d)
            if (x % d == 0) return false;
        return true;
    };
    ENGINE_CHECK_MSG(ell == 0 || is_prime(ell), "LocalModel: ell must be 0 (no localisation) or a prime");
    build_columns();
    build_differentials();
}

void LocalModel::build_columns() {
    for (uint32_t s = 0; s < 3; ++s) {
        const uint32_t nr = static_cast<uint32_t>(desc_->reps(s).size());
        cols_[s].assign(nr, {});
        for (uint32_t r = 0; r < nr; ++r) {
            ColumnSpec& c = cols_[s][r];
            c.m = desc_->stabiliser_order(s, r);
            const Mat22 g = desc_->stabiliser_generator(s, r);
            Mat22 p = Mat22::I();
            for (uint32_t i = 0; i < c.m; ++i) {
                c.pow.push_back(p);
                p = p * g;
            }
            ENGINE_CHECK_MSG(p == Mat22::I(), "LocalModel: stabiliser generator order");
            if (ell_ == 0) {
                c.p_order = c.m;
            } else {
                c.p_order = 1;
                uint32_t rest = c.m;
                while (rest % ell_ == 0) {
                    rest /= ell_;
                    c.p_order *= ell_;
                }
            }
            c.q_order = c.m / c.p_order;
            for (uint32_t i = 0; i < c.q_order; ++i)
                c.e.add(key_of(c.pow[(c.p_order * i) % c.m]), Rat::of(1, c.q_order));
        }
    }
}

int64_t LocalModel::cell_index(uint32_t s, uint32_t rep, uint32_t t) const {
    if (s > 2 || rep >= cols_[s].size()) return -1;
    const uint32_t q = s + t;
    if (q >= cells_.size()) return -1;
    if (t >= 1 && !cols_[s][rep].periodic()) return -1;
    // Cells of degree q are ordered by s, then representative, existing ones only.
    int64_t idx = 0;
    for (uint32_t s2 = 0; s2 < s; ++s2) {
        if (s2 > q) break;
        const uint32_t t2 = q - s2;
        for (uint32_t r2 = 0; r2 < cols_[s2].size(); ++r2)
            if (t2 == 0 || cols_[s2][r2].periodic()) ++idx;
    }
    for (uint32_t r2 = 0; r2 < rep; ++r2)
        if (t == 0 || cols_[s][r2].periodic()) ++idx;
    return idx;
}

QGroupRingElt LocalModel::normalise(uint32_t q, uint32_t cell, const QGroupRingElt& c) const {
    const LCell& lc = cells_.at(q).at(cell);
    const ColumnSpec& col = cols_[lc.s][lc.rep];
    if (col.q_order == 1) return c;
    return c * col.e;
}

void LocalModel::build_differentials() {
    const uint32_t Q = q_top();
    cells_.assign(Q + 1, {});
    for (uint32_t q = 0; q <= Q; ++q)
        for (uint32_t s = 0; s <= std::min<uint32_t>(q, 2); ++s)
            for (uint32_t r = 0; r < cols_[s].size(); ++r)
                if (q == s || cols_[s][r].periodic())
                    cells_[q].push_back(LCell{s, r, q - s, desc_->is_boundary_rep(s, r)});

    auto average = [&](uint32_t s, uint32_t r, QGroupRingElt& y) {
        ++stats_.lifts;
        if (avg_ == LiftAveraging::Off || cols_[s][r].q_order == 1) return;
        QGroupRingElt a = cols_[s][r].e * y;
        if (a != y) ++stats_.changed_by_average;
        y = std::move(a);
    };
    auto exists = [&](uint32_t s, uint32_t r, uint32_t t) { return cell_index(s, r, t) >= 0; };

    // D1[s][t][rep][k]: component at cell (s-1, k, t);  D2[t][rep][k]: at (0, k, t+1).
    using Col = std::vector<QGroupRingElt>;
    std::array<std::vector<std::vector<Col>>, 3> D1;
    std::vector<std::vector<Col>> D2;
    for (uint32_t s = 1; s <= 2; ++s) {
        if (Q < s) continue;
        const uint32_t tmax = Q - s;
        D1[s].assign(tmax + 1, std::vector<Col>(cols_[s].size()));
        std::vector<uint32_t> order;   // boundary summands first ([C, prop:isomodelell](1))
        for (uint32_t r = 0; r < cols_[s].size(); ++r) if (desc_->is_boundary_rep(s, r)) order.push_back(r);
        for (uint32_t r = 0; r < cols_[s].size(); ++r) if (!desc_->is_boundary_rep(s, r)) order.push_back(r);
        for (uint32_t t = 0; t <= tmax; ++t) {
            for (uint32_t r : order) {
                if (!exists(s, r, t)) continue;
                Col& out = D1[s][t][r];
                out.assign(cols_[s - 1].size(), {});
                if (t == 0) {
                    for (const auto& [sign, ft] : desc_->faces(s, r)) out[ft.rep].add(key_of(ft.g), Rat::of(sign));
                    for (uint32_t k = 0; k < out.size(); ++k)
                        if (!out[k].is_zero()) out[k] = out[k] * cols_[s - 1][k].e;
                } else {
                    const QGroupRingElt x = ColumnCoefficient(cols_[s][r], t);
                    for (uint32_t k = 0; k < out.size(); ++k) {
                        const QGroupRingElt& prev = D1[s][t - 1][r][k];
                        if (prev.is_zero()) continue;
                        const QGroupRingElt z = (x * prev) * cols_[s - 1][k].e;
                        if (z.is_zero()) continue;
                        ENGINE_CHECK_MSG(exists(s - 1, k, t - 1), "LocalModel: d^(1) input at a missing cell");
                        ENGINE_CHECK_MSG(column_cycle(cols_[s - 1][k], z, t - 1),
                                         "LocalModel: homotopy input is not a cycle (d^(1))");
                        out[k] = ColumnHomotopyEll(cols_[s - 1][k], z, t - 1);
                        ENGINE_CHECK_MSG(out[k].is_zero() || exists(s - 1, k, t),
                                         "LocalModel: d^(1) lift lands in a missing cell");
                    }
                }
                for (uint32_t k = 0; k < out.size(); ++k) {
                    if (out[k].is_zero()) continue;
                    average(s, r, out[k]);
                    if (desc_->is_boundary_rep(s, r))
                        ENGINE_CHECK_MSG(desc_->is_boundary_rep(s - 1, k),
                                         "LocalModel: d^(1) of a boundary summand left the boundary");
                }
            }
        }
    }
    if (Q >= 2) {
        const uint32_t tmax = Q - 2;
        D2.assign(tmax + 1, std::vector<Col>(cols_[2].size()));
        for (uint32_t t = 0; t <= tmax; ++t) {
            for (uint32_t r = 0; r < cols_[2].size(); ++r) {
                if (!exists(2, r, t)) continue;
                Col w(cols_[0].size());
                const Col& c1 = D1[2][t][r];
                for (uint32_t e = 0; e < c1.size(); ++e) {
                    if (c1[e].is_zero()) continue;
                    ENGINE_CHECK(exists(1, e, t));
                    const Col& c0 = D1[1][t][e];
                    for (uint32_t v = 0; v < c0.size(); ++v)
                        if (!c0[v].is_zero()) w[v].add(c1[e] * c0[v], Rat::of(-1));
                }
                if (t >= 1) {
                    const QGroupRingElt x = ColumnCoefficient(cols_[2][r], t);
                    for (uint32_t v = 0; v < w.size(); ++v)
                        if (!D2[t - 1][r][v].is_zero()) w[v].add(x * D2[t - 1][r][v], Rat::of(-1));
                }
                Col& out = D2[t][r];
                out.assign(cols_[0].size(), {});
                for (uint32_t v = 0; v < w.size(); ++v) {
                    const QGroupRingElt z = w[v] * cols_[0][v].e;
                    if (z.is_zero()) continue;
                    ENGINE_CHECK_MSG(exists(0, v, t), "LocalModel: d^(2) input at a missing cell");
                    ENGINE_CHECK_MSG(column_cycle(cols_[0][v], z, t),
                                     "LocalModel: homotopy input is not a cycle (d^(2))");
                    out[v] = ColumnHomotopyEll(cols_[0][v], z, t);
                    ENGINE_CHECK_MSG(out[v].is_zero() || exists(0, v, t + 1),
                                     "LocalModel: d^(2) lift lands in a missing cell");
                    if (!out[v].is_zero()) average(2, r, out[v]);
                }
            }
        }
    }

    // Assemble d_q : Iso'_q -> Iso'_{q-1}.
    diff_.assign(Q + 1, {});
    by_col_.assign(Q + 1, {});
    for (uint32_t q = 1; q <= Q; ++q) {
        auto& dq = diff_[q];
        for (uint32_t ci = 0; ci < cells_[q].size(); ++ci) {
            const LCell& c = cells_[q][ci];
            std::vector<LEntry> ce;
            if (c.t >= 1) {
                QGroupRingElt x = ColumnCoefficient(cols_[c.s][c.rep], c.t);
                if (c.s % 2 == 1) {
                    QGroupRingElt nx;
                    nx.add(x, Rat::of(-1));
                    x = nx;
                }
                ce.push_back({static_cast<uint32_t>(cell_index(c.s, c.rep, c.t - 1)), ci, x});
            }
            if (c.s >= 1) {
                const Col& z = D1[c.s][c.t][c.rep];
                for (uint32_t k = 0; k < z.size(); ++k)
                    if (!z[k].is_zero()) ce.push_back({static_cast<uint32_t>(cell_index(c.s - 1, k, c.t)), ci, z[k]});
            }
            if (c.s == 2) {
                const Col& z = D2[c.t][c.rep];
                for (uint32_t k = 0; k < z.size(); ++k)
                    if (!z[k].is_zero()) ce.push_back({static_cast<uint32_t>(cell_index(0, k, c.t + 1)), ci, z[k]});
            }
            std::sort(ce.begin(), ce.end(), [](const LEntry& a, const LEntry& b) { return a.row < b.row; });
            for (auto& e : ce) dq.push_back(std::move(e));
        }
        by_col_[q].assign(cells_[q].size(), {});
        for (const auto& e : dq) by_col_[q][e.col].push_back(&e);
    }
}

LChain LocalModel::apply_d(uint32_t q, const LChain& x) const {
    ENGINE_CHECK_MSG(q >= 1 && q <= q_top(), "LocalModel::apply_d: degree out of range");
    LChain out;
    for (const auto& [col, v] : x) {
        if (v.is_zero()) continue;
        for (const LEntry* e : by_col_[q].at(col)) chain_add(out, e->row, v * e->coeff);
    }
    return out;
}

std::map<std::pair<uint32_t, GKey>, Rat> LocalModel::augment(uint32_t q, const LChain& x) const {
    std::map<std::pair<uint32_t, GKey>, Rat> out;
    for (const auto& [ci, v] : x) {
        const LCell& c = cells_.at(q).at(ci);
        if (c.t != 0) continue;
        for (const auto& [g, r] : v.terms) {
            auto& slot = out[{c.rep, decompose(g, cols_[c.s][c.rep].pow).first}];
            slot = slot + r;
        }
    }
    for (auto it = out.begin(); it != out.end();) it = it->second.is_zero() ? out.erase(it) : std::next(it);
    return out;
}

LChain LocalModel::contract(uint32_t q, const LChain& z) const {
    ENGINE_CHECK_MSG(q + 1 <= q_top(), "LocalModel::contract: degree beyond the built range");
    LChain zz;
    chain_add(zz, z);
    LChain k;
    for (int s = static_cast<int>(std::min<uint32_t>(q, 2)); s >= 0; --s) {
        const uint32_t t = q - static_cast<uint32_t>(s);
        LChain ks;
        for (uint32_t r = 0; r < cols_[s].size(); ++r) {
            const int64_t idx = cell_index(static_cast<uint32_t>(s), r, t);
            if (idx < 0) continue;
            auto it = zz.find(static_cast<uint32_t>(idx));
            if (it == zz.end() || it->second.is_zero()) continue;
            ENGINE_CHECK_MSG(column_cycle(cols_[s][r], it->second, t),
                             "LocalModel::contract: the top column component is not a column cycle");
            QGroupRingElt h = ColumnHomotopyEll(cols_[s][r], it->second, t);
            if (h.is_zero()) continue;
            const int64_t idx2 = cell_index(static_cast<uint32_t>(s), r, t + 1);
            ENGINE_CHECK_MSG(idx2 >= 0, "LocalModel::contract: lift lands in a missing cell");
            chain_add(ks, static_cast<uint32_t>(idx2), h, Rat::of((s % 2 == 0) ? 1 : -1));
        }
        if (ks.empty()) continue;
        chain_add(k, ks);
        chain_add(zz, apply_d(q + 1, ks), Rat::of(-1));
    }
    ENGINE_CHECK_MSG(chain_is_zero(zz), "LocalModel::contract: descent left a non-zero remainder (z not a cycle of ker eps)");
    return k;
}

LocalCertificates LocalModel::certify() const {
    LocalCertificates cert;
    cert.d_squared_zero = cert.well_defined = cert.boundary_subcomplex = cert.eps_chain_map = true;
    auto fail = [&](const std::string& why) {
        if (cert.first_failure.empty()) cert.first_failure = why;
    };
    // d o d = 0.
    for (uint32_t q = 2; q <= q_top(); ++q)
        for (uint32_t b = 0; b < cells_[q].size(); ++b) {
            std::map<uint32_t, QGroupRingElt> acc;
            for (const LEntry* e1 : by_col_[q][b])
                for (const LEntry* e2 : by_col_[q - 1][e1->row]) {
                    acc[e2->row].add(e1->coeff * e2->coeff);
                    ++cert.d_squared_terms_checked;
                }
            for (const auto& [row, v] : acc)
                if (!v.is_zero()) {
                    cert.d_squared_zero = false;
                    fail("d_" + std::to_string(q - 1) + " d_" + std::to_string(q) + " != 0 at cell " + std::to_string(b) +
                         " -> " + std::to_string(row) + ": " + v.to_string());
                }
        }
    // Well-definedness and IM2.
    for (uint32_t q = 1; q <= q_top(); ++q)
        for (const auto& e : diff_[q]) {
            ++cert.entries_checked;
            const LCell& c = cells_[q][e.col];
            const LCell& r = cells_[q - 1][e.row];
            const QGroupRingElt w = (cols_[c.s][c.rep].e * e.coeff) * cols_[r.s][r.rep].e;
            if (w != e.coeff) {
                cert.well_defined = false;
                fail("coefficient of d_" + std::to_string(q) + " at (" + std::to_string(e.row) + ", " +
                     std::to_string(e.col) + ") is not in e_col Q[Gamma] e_row");
            }
            if (c.boundary && !r.boundary) {
                cert.boundary_subcomplex = false;
                fail("d of boundary cell " + std::to_string(e.col) + " in degree " + std::to_string(q) +
                     " has a non-boundary component");
            }
        }
    // IM1: eps' d = d eps' on canonical cosets.
    for (uint32_t q = 1; q <= q_top(); ++q)
        for (uint32_t b = 0; b < cells_[q].size(); ++b) {
            const LCell& c = cells_[q][b];
            LChain gen;
            gen[b] = cols_[c.s][c.rep].e;
            const auto lhs = augment(q - 1, apply_d(q, gen));
            std::map<std::pair<uint32_t, GKey>, Rat> rhs;
            if (c.t == 0 && c.s >= 1)
                for (const auto& [sign, ft] : desc_->faces(c.s, c.rep)) {
                    auto& slot = rhs[{ft.rep, decompose(key_of(ft.g), cols_[c.s - 1][ft.rep].pow).first}];
                    slot = slot + Rat::of(sign);
                }
            for (auto it = rhs.begin(); it != rhs.end();) it = it->second.is_zero() ? rhs.erase(it) : std::next(it);
            ++cert.eps_cells_checked;
            if (lhs != rhs) {
                cert.eps_chain_map = false;
                fail("eps' d != d eps' at degree " + std::to_string(q) + " cell " + std::to_string(b));
            }
        }
    return cert;
}

// ---------------------------------------------------------------------------
// Comparison maps ([C, prop:isomodelell](4))
// ---------------------------------------------------------------------------
ComparisonMap StrictComparison(const LocalModel& src, const LocalModel& dst, LiftAveraging avg) {
    ENGINE_CHECK_MSG(&src.descriptor() == &dst.descriptor() && src.q_max() == dst.q_max(),
                     "StrictComparison: the two models must share the descriptor and q_max");
    ComparisonMap f;
    f.value.assign(src.q_top() + 1, {});
    for (uint32_t q = 0; q <= src.q_top(); ++q) {
        f.value[q].assign(src.rank(q), {});
        for (uint32_t xi = 0; xi < src.rank(q); ++xi) {
            const LCell& x = src.cells(q)[xi];
            LChain y;
            if (x.t == 0) {
                const int64_t idx = dst.cell_index(x.s, x.rep, 0);
                ENGINE_CHECK(idx >= 0);
                y[static_cast<uint32_t>(idx)] = dst.column(x.s, x.rep).e;   // y_0' = 1 (x) e
            }
            if (q >= 1) {
                LChain z;
                for (const LEntry& e : src.differential(q)) {
                    if (e.col != xi) continue;
                    for (const auto& [c, v] : f.value[q - 1][e.row]) chain_add(z, c, e.coeff * v);
                }
                chain_add(z, dst.apply_d(q, y), Rat::of(-1));
                if (!chain_is_zero(z)) chain_add(y, dst.contract(q - 1, z));
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

ComparisonCertificate CertifyComparison(const LocalModel& src, const LocalModel& dst, const ComparisonMap& f) {
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
            // well defined: e_x f(x) = f(x)
            LChain ea;
            for (const auto& [c, v] : fx) chain_add(ea, c, cs.e * v);
            if (!chains_equal(ea, fx)) {
                cert.well_defined = false;
                ++cert.not_well_defined;
                fail("f(x) not Q_x-invariant at degree " + std::to_string(q) + " cell " + std::to_string(xi));
            }
            // strict over Xbulk
            LChain gen;
            gen[xi] = cs.e;
            if (dst.augment(q, fx) != src.augment(q, gen)) {
                cert.strict = false;
                fail("eps f != eps at degree " + std::to_string(q) + " cell " + std::to_string(xi));
            }
            // chain map
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
// Cochains over a ModCtx
// ---------------------------------------------------------------------------
namespace {

// a^{-1} mod m (gcd(a, m) = 1); (S24 CL-G3, C-07) ScalarMod.h's extended Euclid.
uint64_t inverse_mod(uint64_t a, uint64_t m) {
    uint64_t x = 0;
    ENGINE_CHECK_MSG(engine::scalar::try_invmod<uint64_t>(a, m, x), "RhoQ: a denominator is not a unit of the coefficient ring (ell divides it)");
    return x;
}

template<class Elt>
Elt reduce_rat(const engine::ModCtx<Elt>& ctx, const Rat& r) {
    const uint64_t m = static_cast<uint64_t>(ctx.p);
    ENGINE_CHECK_MSG(m < (uint64_t(1) << 62), "RhoQ: modulus too large");
    const int64_t pm = static_cast<int64_t>(m);
    int64_t a = r.num % pm;
    if (a < 0) a += pm;
    const uint64_t inv = inverse_mod(static_cast<uint64_t>(r.den), m);
    return ctx.mul(static_cast<Elt>(a), static_cast<Elt>(inv));
}

template<class Elt>
uint32_t module_length(const engine::ModCtx<Elt>& ctx, const engine::DenseBuf<Elt>& M) {
    if (M.rows == 0 || M.cols == 0) return 0;
    if (ctx.is_field()) {
        engine::DenseBuf<Elt> R = M;
        return engine::elim::rref(ctx, engine::view(R)).rank;
    }
    return engine::elim::length(ctx, engine::cview(M));
}

}  // namespace

template<class Elt>
CoefficientModule<Elt> SymModule(const engine::ModCtx<Elt>& ctx, uint32_t n) {
    CoefficientModule<Elt> M;
    M.name = "Sym^" + std::to_string(n);
    M.dim = n + 1;
    M.rho = [ctx, n](const Mat22& g) { return rho_n(ctx, g, n); };
    return M;
}

template<class Elt>
engine::DenseBuf<Elt> RhoQ(const engine::ModCtx<Elt>& ctx, const QGroupRingElt& a, const CoefficientModule<Elt>& M,
                           std::map<GKey, engine::DenseBuf<Elt>>& memo) {
    const uint32_t D = M.dim;
    engine::DenseBuf<Elt> out(D, D);
    for (const auto& [g, r] : a.terms) {
        const Elt c = reduce_rat(ctx, r);
        if (c == 0) continue;
        auto it = memo.find(g);
        if (it == memo.end()) {
            engine::DenseBuf<Elt> R = M.rho(mat_of(g));
            ENGINE_CHECK_MSG(R.rows == D && R.cols == D, "RhoQ: the coefficient module returned a block of the wrong size");
            it = memo.emplace(g, std::move(R)).first;
        }
        const auto& R = it->second;
        for (uint32_t i = 0; i < D; ++i)
            for (uint32_t j = 0; j < D; ++j) out.at(i, j) = ctx.add(out.at(i, j), ctx.mul(c, R.at(i, j)));
    }
    return out;
}

template<class Elt>
engine::DenseBuf<Elt> RhoQ(const engine::ModCtx<Elt>& ctx, const QGroupRingElt& a, uint32_t n,
                           std::map<GKey, engine::DenseBuf<Elt>>& memo) {
    return RhoQ(ctx, a, SymModule(ctx, n), memo);
}

template<class Elt>
engine::DenseBuf<Elt> LocalModel::coboundary_impl(const engine::ModCtx<Elt>& ctx, const CoefficientModule<Elt>& Mod,
                                                  uint32_t q, bool bdry) const {
    ENGINE_CHECK_MSG(q <= q_max_, "LocalModel::coboundary: q > q_max");
    const uint32_t D = Mod.dim;
    auto local_index = [&](uint32_t deg) {
        std::vector<int64_t> idx(cells_[deg].size(), -1);
        int64_t k = 0;
        for (uint32_t i = 0; i < cells_[deg].size(); ++i)
            if (!bdry || cells_[deg][i].boundary) idx[i] = k++;
        return std::make_pair(idx, static_cast<uint32_t>(k));
    };
    const auto [src, ncols] = local_index(q);
    const auto [dst, nrows] = local_index(q + 1);
    engine::DenseBuf<Elt> M(nrows * D, ncols * D);
    std::map<GKey, engine::DenseBuf<Elt>> memo;
    for (const auto& e : diff_[q + 1]) {
        const int64_t bi = dst[e.col], bj = src[e.row];
        if (bi < 0 || bj < 0) {
            ENGINE_CHECK_MSG(bi < 0, "LocalModel: boundary cochain differential leaves the boundary");
            continue;
        }
        const auto R = RhoQ(ctx, e.coeff, Mod, memo);
        for (uint32_t i = 0; i < D; ++i)
            for (uint32_t j = 0; j < D; ++j) {
                Elt& v = M.at(static_cast<uint32_t>(bi) * D + i, static_cast<uint32_t>(bj) * D + j);
                v = ctx.add(v, R.at(i, j));
            }
    }
    return M;
}

template<class Elt>
engine::DenseBuf<Elt> LocalModel::projector_impl(const engine::ModCtx<Elt>& ctx, const CoefficientModule<Elt>& Mod,
                                                 uint32_t q, bool bdry) const {
    const uint32_t D = Mod.dim;
    uint32_t k = 0;
    for (const auto& c : cells_.at(q)) k += (!bdry || c.boundary) ? 1 : 0;
    engine::DenseBuf<Elt> P(k * D, k * D);
    std::map<GKey, engine::DenseBuf<Elt>> memo;
    uint32_t b = 0;
    for (const auto& c : cells_.at(q)) {
        if (bdry && !c.boundary) continue;
        const auto R = RhoQ(ctx, cols_[c.s][c.rep].e, Mod, memo);
        for (uint32_t i = 0; i < D; ++i)
            for (uint32_t j = 0; j < D; ++j) P.at(b * D + i, b * D + j) = R.at(i, j);
        ++b;
    }
    return P;
}

template<class Elt>
engine::DenseBuf<Elt> LocalModel::coboundary(const engine::ModCtx<Elt>& ctx, uint32_t n, uint32_t q) const {
    return coboundary_impl(ctx, SymModule(ctx, n), q, false);
}
template<class Elt>
engine::DenseBuf<Elt> LocalModel::boundary_coboundary(const engine::ModCtx<Elt>& ctx, uint32_t n, uint32_t q) const {
    return coboundary_impl(ctx, SymModule(ctx, n), q, true);
}
template<class Elt>
engine::DenseBuf<Elt> LocalModel::projector(const engine::ModCtx<Elt>& ctx, uint32_t n, uint32_t q) const {
    return projector_impl(ctx, SymModule(ctx, n), q, false);
}
template<class Elt>
engine::DenseBuf<Elt> LocalModel::boundary_projector(const engine::ModCtx<Elt>& ctx, uint32_t n, uint32_t q) const {
    return projector_impl(ctx, SymModule(ctx, n), q, true);
}
template<class Elt>
engine::DenseBuf<Elt> LocalModel::coboundary(const engine::ModCtx<Elt>& ctx, const CoefficientModule<Elt>& M,
                                             uint32_t q) const {
    return coboundary_impl(ctx, M, q, false);
}
template<class Elt>
engine::DenseBuf<Elt> LocalModel::boundary_coboundary(const engine::ModCtx<Elt>& ctx, const CoefficientModule<Elt>& M,
                                                      uint32_t q) const {
    return coboundary_impl(ctx, M, q, true);
}
template<class Elt>
engine::DenseBuf<Elt> LocalModel::projector(const engine::ModCtx<Elt>& ctx, const CoefficientModule<Elt>& M,
                                            uint32_t q) const {
    return projector_impl(ctx, M, q, false);
}
template<class Elt>
engine::DenseBuf<Elt> LocalModel::boundary_projector(const engine::ModCtx<Elt>& ctx, const CoefficientModule<Elt>& M,
                                                     uint32_t q) const {
    return projector_impl(ctx, M, q, true);
}

template<class Elt>
std::vector<uint32_t> LocalModel::cohomology_lengths(const engine::ModCtx<Elt>& ctx, const CoefficientModule<Elt>& M,
                                                     bool bdry) const {
    std::vector<uint32_t> im(q_max_ + 1, 0), lp(q_max_ + 1, 0), H(q_max_ + 1, 0);
    for (uint32_t q = 0; q <= q_max_; ++q) {
        im[q] = module_length(ctx, coboundary_impl(ctx, M, q, bdry));
        lp[q] = module_length(ctx, projector_impl(ctx, M, q, bdry));
    }
    for (uint32_t q = 0; q <= q_max_; ++q) {
        const uint32_t sub = im[q] + (q > 0 ? im[q - 1] : 0);
        ENGINE_CHECK_MSG(sub <= lp[q], "LocalModel::cohomology_lengths: negative length");
        H[q] = lp[q] - sub;
    }
    return H;
}

template<class Elt>
std::vector<uint32_t> LocalModel::cohomology_lengths(const engine::ModCtx<Elt>& ctx, uint32_t n, bool bdry) const {
    return cohomology_lengths(ctx, SymModule(ctx, n), bdry);
}

template<class Elt>
engine::DenseBuf<Elt> Pullback(const engine::ModCtx<Elt>& ctx, const CoefficientModule<Elt>& Mod, uint32_t q,
                               const LocalModel& src, const LocalModel& dst, const ComparisonMap& f) {
    const uint32_t D = Mod.dim;
    engine::DenseBuf<Elt> M(src.rank(q) * D, dst.rank(q) * D);
    std::map<GKey, engine::DenseBuf<Elt>> memo;
    for (uint32_t x = 0; x < src.rank(q); ++x)
        for (const auto& [c, v] : f.value.at(q).at(x)) {
            const auto R = RhoQ(ctx, v, Mod, memo);
            for (uint32_t i = 0; i < D; ++i)
                for (uint32_t j = 0; j < D; ++j) {
                    Elt& w = M.at(x * D + i, c * D + j);
                    w = ctx.add(w, R.at(i, j));
                }
        }
    return M;
}

template<class Elt>
engine::DenseBuf<Elt> Pullback(const engine::ModCtx<Elt>& ctx, uint32_t n, uint32_t q, const LocalModel& src,
                               const LocalModel& dst, const ComparisonMap& f) {
    return Pullback(ctx, SymModule(ctx, n), q, src, dst, f);
}

#define ISOMODEL_ELL_INSTANTIATE(Elt)                                                                          \
    template CoefficientModule<Elt> SymModule<Elt>(const engine::ModCtx<Elt>&, uint32_t);                       \
    template engine::DenseBuf<Elt> RhoQ<Elt>(const engine::ModCtx<Elt>&, const QGroupRingElt&, uint32_t,        \
                                             std::map<GKey, engine::DenseBuf<Elt>>&);                          \
    template engine::DenseBuf<Elt> RhoQ<Elt>(const engine::ModCtx<Elt>&, const QGroupRingElt&,                  \
                                             const CoefficientModule<Elt>&,                                    \
                                             std::map<GKey, engine::DenseBuf<Elt>>&);                          \
    template engine::DenseBuf<Elt> LocalModel::coboundary<Elt>(const engine::ModCtx<Elt>&, uint32_t, uint32_t)   \
        const;                                                                                                 \
    template engine::DenseBuf<Elt> LocalModel::boundary_coboundary<Elt>(const engine::ModCtx<Elt>&, uint32_t,   \
                                                                        uint32_t) const;                       \
    template engine::DenseBuf<Elt> LocalModel::projector<Elt>(const engine::ModCtx<Elt>&, uint32_t, uint32_t)    \
        const;                                                                                                 \
    template engine::DenseBuf<Elt> LocalModel::boundary_projector<Elt>(const engine::ModCtx<Elt>&, uint32_t,    \
                                                                       uint32_t) const;                        \
    template engine::DenseBuf<Elt> LocalModel::coboundary<Elt>(const engine::ModCtx<Elt>&,                      \
                                                               const CoefficientModule<Elt>&, uint32_t) const; \
    template engine::DenseBuf<Elt> LocalModel::boundary_coboundary<Elt>(                                       \
        const engine::ModCtx<Elt>&, const CoefficientModule<Elt>&, uint32_t) const;                           \
    template engine::DenseBuf<Elt> LocalModel::projector<Elt>(const engine::ModCtx<Elt>&,                       \
                                                              const CoefficientModule<Elt>&, uint32_t) const;  \
    template engine::DenseBuf<Elt> LocalModel::boundary_projector<Elt>(                                        \
        const engine::ModCtx<Elt>&, const CoefficientModule<Elt>&, uint32_t) const;                           \
    template std::vector<uint32_t> LocalModel::cohomology_lengths<Elt>(const engine::ModCtx<Elt>&, uint32_t,     \
                                                                       bool) const;                            \
    template std::vector<uint32_t> LocalModel::cohomology_lengths<Elt>(                                        \
        const engine::ModCtx<Elt>&, const CoefficientModule<Elt>&, bool) const;                               \
    template engine::DenseBuf<Elt> Pullback<Elt>(const engine::ModCtx<Elt>&, uint32_t, uint32_t,                \
                                                 const LocalModel&, const LocalModel&, const ComparisonMap&);  \
    template engine::DenseBuf<Elt> Pullback<Elt>(const engine::ModCtx<Elt>&, const CoefficientModule<Elt>&,     \
                                                 uint32_t, const LocalModel&, const LocalModel&,               \
                                                 const ComparisonMap&);

ISOMODEL_ELL_INSTANTIATE(uint32_t)
ISOMODEL_ELL_INSTANTIATE(uint64_t)
#undef ISOMODEL_ELL_INSTANTIATE

}  // namespace integral_model
