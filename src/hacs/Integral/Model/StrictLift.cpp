// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Line 2, L2-1b (1) (hecke-extensions slot 10, WP-10) -- the strict lift
// tau_Iso of [C, prop:strictlift], its certificates and the coset transport
// Theta^_g.  See StrictLift.h for the contract.
#include "StrictLift.h"

#include <algorithm>
#include <map>
#include <sstream>
#include <tuple>
#include <utility>

#include "../../Hecke/HeckeBoundaryDatum.h"
#include "../../Hecke/HeckeCosetTables.h"
#include "../../Hecke/HeckeTauTable.h"
#include "../../Hecke/SL2ZTessellationTables.h"
#include "../../LocalSystemSL2Z.h"
#include "../../SimplicialCohomology.h"
#include "../../../ModArithmetic/EngineCheck.h"
#include "PeriodicColumn.h"

namespace integral_model {

using hecke::Mat22;

// ---------------------------------------------------------------------------
// IsoChain
// ---------------------------------------------------------------------------
bool IsoChain::is_zero() const {
    for (const auto& e : c)
        if (!e.is_zero()) return false;
    return true;
}

void IsoChain::add(const IsoChain& o, int64_t scale) {
    ENGINE_CHECK_MSG(c.size() == o.c.size(), "IsoChain::add: degree mismatch");
    for (size_t i = 0; i < c.size(); ++i) c[i].add(o.c[i], scale);
}

uint64_t IsoChain::num_terms() const {
    uint64_t t = 0;
    for (const auto& e : c) t += e.terms.size();
    return t;
}

int64_t IsoChain::max_abs_coefficient() const {
    int64_t m = 0;
    for (const auto& e : c) m = std::max(m, e.max_abs_coefficient());
    return m;
}

int64_t IsoChain::max_abs_entry() const {
    int64_t m = 0;
    for (const auto& e : c) m = std::max(m, e.max_abs_entry());
    return m;
}

uint64_t IsoChain::l1_norm() const {
    uint64_t s = 0;
    for (const auto& e : c)
        for (const auto& [g, n] : e.terms) s = hecke::checked_add<uint64_t>(s, hecke::uabs64(n));
    return s;
}

std::string IsoChain::to_string() const {
    std::ostringstream os;
    bool first = true;
    for (size_t i = 0; i < c.size(); ++i) {
        if (c[i].is_zero()) continue;
        if (!first) os << " + ";
        first = false;
        os << "(" << c[i].to_string() << ") e_" << i;
    }
    if (first) os << "0";
    return os.str();
}

IsoChain left_multiply(const Mat22& u, const IsoChain& x) {
    IsoChain out(x.c.size());
    const GroupRingElt U = GroupRingElt::unit(u);
    for (size_t i = 0; i < x.c.size(); ++i)
        if (!x.c[i].is_zero()) out.c[i] = U * x.c[i];
    return out;
}

// ---------------------------------------------------------------------------
// decompose / dF_coeff / column_homotopy: shared with PeriodicityClass.cpp
// and IsotropyModel.cpp since slot-14 WP-14b (review finding slot-10 m-2) --
// see PeriodicColumn.h (found by ordinary enclosing-namespace lookup, this
// file being inside `namespace integral_model` throughout).
// ---------------------------------------------------------------------------
namespace {

// A column cycle: t = 0 -- the multiplicities of every coset gamma <g> sum to
// zero; t >= 1 -- a * x_t = 0.
bool is_column_cycle(const GroupRingElt& a, const std::vector<Mat22>& powg, uint32_t t) {
    if (t == 0) {
        std::map<GKey, int64_t> cosets;
        for (const auto& [g, n] : a.terms) {
            const GKey c = decompose(g, powg).first;
            cosets[c] = hecke::checked_add<int64_t>(cosets[c], n);
        }
        for (const auto& [c, n] : cosets)
            if (n != 0) return false;
        return true;
    }
    return (a * dF_coeff(powg, t)).is_zero();
}

// A canonical chain of Xbulk: (sigma_hat, gamma_canon) -> signed multiplicity.
using CanonChain = std::map<std::pair<uint32_t, GKey>, int64_t>;

void canon_add(CanonChain& acc, const hecke::SL2ZTessellationTables& tess, const Mat22& gamma, uint32_t q,
               uint32_t sigma, int64_t n) {
    if (n == 0) return;
    const hecke::KuSimplexKey k = tess.canonical_key(gamma, q, sigma);
    const auto key = std::make_pair(k.sigma_hat, key_of(k.gamma_canon));
    int64_t& v = acc[key];
    v = hecke::checked_add<int64_t>(v, hecke::checked_mul<int64_t>(n, static_cast<int64_t>(k.sign)));
    if (v == 0) acc.erase(key);
}

CanonChain canon_of_terms(const hecke::SL2ZTessellationTables& tess, uint32_t q,
                          const std::vector<hecke::ChainTerm>& terms) {
    CanonChain acc;
    for (const auto& t : terms) canon_add(acc, tess, t.gamma, q, t.sigma_idx, t.coeff);
    return acc;
}

}  // namespace

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------
StrictLift::StrictLift(const IsotropyModel& R, const hecke::HeckeCosetTablesT<uint32_t>& cosets,
                       const hecke::HeckeTauTableT<uint32_t>& tau_s,
                       const hecke::HeckeBoundaryTauTableT<uint32_t>& taub)
    : R_(&R), cosets_(&cosets), tau_s_(&tau_s), taub_(&taub), tess_(&tau_s.model()) {
    ENGINE_CHECK_MSG(tau_s.is_strict_pair(),
                     "StrictLift: tau_s must be the BoundaryStrict table of [C, prop:strictpair]");
    ENGINE_CHECK_MSG(tau_s.strict_boundary_table() == &taub,
                     "StrictLift: tau_s was not built on this boundary table");
    ENGINE_CHECK_MSG(&tau_s.cosets() == &cosets && &taub.cosets() == &cosets,
                     "StrictLift: tau_s / tau_bd were not built on these coset tables");
    ENGINE_CHECK_MSG(std::string(tau_s.solve_regime()) == "Z", "StrictLift: tau_s is not a datum over Z");
    // Slot-11 WP-11: g = diag(1, p^nu), nu in {1, 2, 3} (was: nu = 1 only).
    ENGINE_CHECK_MSG(cosets.r() >= 1 && cosets.r() <= 3, "StrictLift: g = diag(1, p^nu), nu in {1, 2, 3} only");
    m_ = cosets.num_cosets();
    N_ = cosets.N();
    g_ = Mat22{1, 0, 0, static_cast<int64_t>(N_)};
    ENGINE_CHECK(cosets.g() == g_);
    init_representatives();
    build();
}

// Slot-11 WP-11: g = p I, Gamma'' = Gamma, m = 1, tau = id ([C, prop:diamond]).
StrictLift::StrictLift(const IsotropyModel& R, const hecke::SL2ZTessellationTables& tess, uint64_t p_central)
    : R_(&R), cosets_(nullptr), tau_s_(nullptr), taub_(nullptr), tess_(&tess) {
    ENGINE_CHECK_MSG(p_central >= 2 && p_central < (1ull << 31), "StrictLift: central p out of range");
    central_ = true;
    m_ = 1;
    N_ = hecke::checked_mul<uint64_t>(p_central, p_central);   // det g
    const int64_t pp = static_cast<int64_t>(p_central);
    g_ = Mat22{pp, 0, 0, pp};
    init_representatives();
    build();
}

std::string DoubleCoset::name() const {
    if (central) return "diag(p,p)";
    return nu == 1 ? std::string("diag(1,p)") : "diag(1,p^" + std::to_string(nu) + ")";
}

DoubleCoset StrictLift::double_coset() const {
    return central_ ? DoubleCoset::scalar() : DoubleCoset::diag_one(cosets_->r());
}

const hecke::HeckeCosetTablesT<uint32_t>& StrictLift::cosets() const {
    ENGINE_CHECK_MSG(cosets_ != nullptr, "StrictLift::cosets: the central lift has no coset tables");
    return *cosets_;
}
const hecke::HeckeTauTableT<uint32_t>& StrictLift::tau_s() const {
    ENGINE_CHECK_MSG(tau_s_ != nullptr, "StrictLift::tau_s: the central lift has no tau table (tau = id)");
    return *tau_s_;
}
const hecke::HeckeBoundaryTauTableT<uint32_t>& StrictLift::taub() const {
    ENGINE_CHECK_MSG(taub_ != nullptr, "StrictLift::taub: the central lift has no boundary table (tau_bd = id)");
    return *taub_;
}

Mat22 StrictLift::coset_s(uint32_t j) const { return central_ ? Mat22::I() : cosets_->s(j); }
Mat22 StrictLift::coset_delta(uint32_t j) const { return central_ ? g_ : cosets_->delta(j); }
Mat22 StrictLift::twisted(const Mat22& gamma, uint32_t& j) const {
    if (central_) {   // gamma = gamma * s_1, conj_g = id
        j = 1;
        return gamma;
    }
    return cosets_->twisted_conjugate(gamma, j);
}
std::vector<hecke::ChainTerm> StrictLift::tau_s_chain(uint32_t j, uint32_t q, uint32_t kidx) const {
    if (central_) return {hecke::ChainTerm{1, Mat22::I(), kidx}};
    return tau_s_->tau(j, q, kidx);
}
std::vector<hecke::ChainTerm> StrictLift::taub_chain(uint32_t j, uint32_t q, uint32_t kidx) const {
    if (central_) return {hecke::ChainTerm{1, Mat22::I(), kidx}};
    return taub_->tau(j, q, kidx);
}

void StrictLift::init_representatives() {
    const IsotropyModel& R = *R_;
    // Representatives, their K indices, and every K simplex's transport.
    const auto& Ks = IsotropyModel::K_simplices();
    rep_kidx_.assign(3, {});
    kidx_to_rep_.assign(3, {});
    for (uint32_t s = 0; s < 3; ++s) {
        for (const Simplex& rep : R.reps(s)) {
            const auto it = std::find(Ks[s].begin(), Ks[s].end(), rep);
            ENGINE_CHECK(it != Ks[s].end());
            rep_kidx_[s].push_back(static_cast<uint32_t>(it - Ks[s].begin()));
        }
        kidx_to_rep_[s].assign(Ks[s].size(), {UINT32_MAX, Mat22::I()});
        for (uint32_t r = 0; r < rep_kidx_[s].size(); ++r) kidx_to_rep_[s][rep_kidx_[s][r]] = {r, Mat22::I()};
    }
    for (const auto& nr : R.non_representatives()) {
        const uint32_t s = static_cast<uint32_t>(nr.face.size()) - 1;
        const auto it = std::find(Ks[s].begin(), Ks[s].end(), nr.face);
        ENGINE_CHECK(it != Ks[s].end());
        kidx_to_rep_[s][static_cast<uint32_t>(it - Ks[s].begin())] = {nr.rep, nr.g};
    }
    for (uint32_t s = 0; s < 3; ++s)
        for (const auto& pr : kidx_to_rep_[s]) ENGINE_CHECK_MSG(pr.first != UINT32_MAX, "StrictLift: an unplaced K simplex");

    pow_.assign(3, {});
    for (uint32_t s = 0; s < 3; ++s) {
        pow_[s].resize(R.reps(s).size());
        for (uint32_t r = 0; r < R.reps(s).size(); ++r) {
            Mat22 p = Mat22::I();
            for (uint32_t i = 0; i < R.stabiliser_order(s, r); ++i) {
                pow_[s][r].push_back(p);
                p = p * R.stabiliser_generator(s, r);
            }
        }
    }
}

const IsoChain& StrictLift::value(uint32_t q, uint32_t j, uint32_t cell) const {
    ENGINE_CHECK_MSG(q < table_.size() && j >= 1 && j <= m_ && cell < table_[q][j - 1].size(),
                     "StrictLift::value: index out of range");
    return table_[q][j - 1][cell];
}

IsoChain StrictLift::on_translate(const Mat22& gamma, uint32_t q, uint32_t cell) const {
    uint32_t j = 0;
    const Mat22 u = twisted(gamma, j);   // gamma = h s_j, u = g^{-1} h g
    return left_multiply(u, value(q, j, cell));
}

IsoChain StrictLift::apply(uint32_t q, const IsoChain& x) const {
    ENGINE_CHECK(x.c.size() == R_->rank(q));
    IsoChain out(R_->rank(q));
    for (uint32_t c = 0; c < x.c.size(); ++c)
        for (const auto& [g, n] : x.c[c].terms) out.add(on_translate(mat_of(g), q, c), n);
    return out;
}

IsoChain StrictLift::boundary(uint32_t q, const IsoChain& x) const {
    ENGINE_CHECK(q >= 1 && x.c.size() == R_->rank(q));
    IsoChain out(R_->rank(q - 1));
    for (const DiffEntry& e : R_->differential(q))
        if (!x.c[e.col].is_zero()) out.c[e.row].add(x.c[e.col] * e.coeff);
    return out;
}

IsoChain StrictLift::basis_boundary(uint32_t q, uint32_t j, uint32_t cell) const {
    IsoChain x(R_->rank(q));
    x.c[cell] = GroupRingElt::unit(coset_s(j));
    return boundary(q, x);
}

// A stored tau^s chain (canonical terms gamma . i(sigma), sigma a K index) as
// an element of Iso_q at t = 0: gamma . i(sigma) = gamma g_t . i(rep) =
// eps(gamma g_t e_{q, rep, 0}), the transports orientation-preserving
// ([C, lem:model7coincidences] (2); IsotropyModel's descriptor).
IsoChain StrictLift::chain_to_iso(uint32_t q, const std::vector<hecke::ChainTerm>& terms) const {
    ENGINE_CHECK(q <= 2);
    IsoChain y(R_->rank(q));
    for (const auto& t : terms) {
        ENGINE_CHECK(t.sigma_idx < kidx_to_rep_[q].size());
        const auto& [rep, tr] = kidx_to_rep_[q][t.sigma_idx];
        y.c[R_->cell_index(q, rep, 0)].add(key_of(t.gamma * tr), t.coeff);
    }
    return y;
}

// The descent through the columns: z in Iso_qz a cycle of ker eps; returns k
// in Iso_{qz+1} supported at t >= 1 with d k = z.
IsoChain StrictLift::contract(uint32_t qz, const IsoChain& z) const {
    IsoChain k(R_->rank(qz + 1));
    IsoChain zz = z;
    for (int s = static_cast<int>(std::min<uint32_t>(qz, 2)); s >= 0; --s) {
        const uint32_t su = static_cast<uint32_t>(s);
        const uint32_t t = qz - su;
        IsoChain ks(R_->rank(qz + 1));
        bool any = false;
        for (uint32_t r = 0; r < R_->reps(su).size(); ++r) {
            const GroupRingElt& a = zz.c[R_->cell_index(su, r, t)];
            if (a.is_zero()) continue;
            ENGINE_CHECK_MSG(is_column_cycle(a, pow_[su][r], t),
                             "StrictLift: a column component of tau_Iso(dx) - dy0 is not a column cycle");
            GroupRingElt h = column_homotopy(a, pow_[su][r], t);
            if (su % 2 == 1) {
                GroupRingElt nh;
                nh.add(h, -1);
                h = nh;
            }
            ks.c[R_->cell_index(su, r, t + 1)] = h;
            any = true;
        }
        if (!any) continue;
        zz.add(boundary(qz + 1, ks), -1);
        for (uint32_t r = 0; r < R_->reps(su).size(); ++r)
            ENGINE_CHECK_MSG(zz.c[R_->cell_index(su, r, t)].is_zero(),
                             "StrictLift: the column step did not clear its column");
        k.add(ks);
    }
    ENGINE_CHECK_MSG(zz.is_zero(), "StrictLift: the descent through the columns left a non-zero remainder");
    return k;
}

void StrictLift::build() {
    const uint32_t Q = R_->q_top();
    table_.assign(Q + 1, {});
    for (uint32_t q = 0; q <= Q; ++q) {
        const auto& cells = R_->cells(q);
        table_[q].assign(m_, std::vector<IsoChain>(cells.size()));
        // Boundary summands first ([C, prop:strictlift] (1)): each value uses
        // only degree q - 1, so the order fixes no value; it is kept as the
        // lemma's order, and (B) certifies the boundary restriction.
        std::vector<uint32_t> order;
        for (uint32_t c = 0; c < cells.size(); ++c) if (cells[c].boundary) order.push_back(c);
        for (uint32_t c = 0; c < cells.size(); ++c) if (!cells[c].boundary) order.push_back(c);
        for (uint32_t j = 1; j <= m_; ++j) {
            for (uint32_t c : order) {
                const Cell& cell = cells[c];
                IsoChain y(cells.size());
                if (cell.t == 0) {
                    // y_0: f(x) = tau^s(eps x) (on R_d this is tau_bd(eps x), by (g) of the table).
                    y = chain_to_iso(q, tau_s_chain(j, q, rep_kidx_[q][cell.rep]));
                }
                if (q >= 1) {
                    IsoChain z = apply(q - 1, basis_boundary(q, j, c));
                    z.add(boundary(q, y), -1);
                    y.add(contract(q - 1, z));
                }
                table_[q][j - 1][c] = std::move(y);
            }
        }
    }
}

StrictLift StrictLift::perturbed(uint32_t q, uint32_t j, uint32_t cell, int64_t scale) const {
    StrictLift out = *this;
    ENGINE_CHECK(q < out.table_.size() && j >= 1 && j <= m_ && cell < out.table_[q][j - 1].size());
    out.table_[q][j - 1][cell].c[cell].add(key_of(Mat22::I()), scale);
    return out;
}

StrictLift StrictLift::with_homotopy(uint32_t q, uint32_t j, uint32_t cell, const IsoChain& k) const {
    StrictLift out = *this;
    ENGINE_CHECK(q + 1 < table_.size() && j >= 1 && j <= m_ && cell < R_->rank(q) && k.c.size() == R_->rank(q + 1));
    // K on an element of Iso_q: each term gamma e_cell, gamma = h s_j' -> (g^{-1} h g) k when j' = j
    auto K = [&](const IsoChain& x) {
        IsoChain o(R_->rank(q + 1));
        for (const auto& [gk, n] : x.c[cell].terms) {
            uint32_t jj = 0;
            const Mat22 u = twisted(mat_of(gk), jj);
            if (jj == j) o.add(left_multiply(u, k), n);
        }
        return o;
    };
    // degree q: tau'(s_j e_cell) = tau + d k (K vanishes on Iso_{q-1}); degree q + 1: tau'(y) = tau(y) + K(d y)
    out.table_[q][j - 1][cell].add(boundary(q + 1, k));
    for (uint32_t jj = 1; jj <= m_; ++jj)
        for (uint32_t c = 0; c < R_->rank(q + 1); ++c) out.table_[q + 1][jj - 1][c].add(K(basis_boundary(q + 1, jj, c)));
    return out;
}

// ---------------------------------------------------------------------------
// Certificates
// ---------------------------------------------------------------------------
StrictLiftCertificate StrictLift::certify() const {
    StrictLiftCertificate C;
    C.t1_twisted_equivariance = C.t2_chain_map = C.t3_augmentation = true;
    C.strict_eps = C.boundary_preserved = C.boundary_eps = true;
    auto fail = [&](bool& flag, const std::string& why) {
        flag = false;
        if (C.first_failure.empty()) C.first_failure = why;
    };
    const uint32_t Q = q_top();
    const auto tag = [](uint32_t q, uint32_t j, uint32_t c) {
        return " (q = " + std::to_string(q) + ", j = " + std::to_string(j) + ", cell = " + std::to_string(c) + ")";
    };

    // Sizes.
    std::map<GKey, bool> elems;
    C.terms.assign(Q + 1, 0);
    C.max_terms.assign(Q + 1, 0);
    C.max_l1.assign(Q + 1, 0);
    for (uint32_t q = 0; q <= Q; ++q)
        for (uint32_t j = 1; j <= m_; ++j)
            for (uint32_t c = 0; c < R_->rank(q); ++c) {
                const IsoChain& v = value(q, j, c);
                C.terms[q] += v.num_terms();
                C.max_terms[q] = std::max(C.max_terms[q], v.num_terms());
                C.max_l1[q] = std::max(C.max_l1[q], v.l1_norm());
                C.max_abs_coefficient = std::max(C.max_abs_coefficient, v.max_abs_coefficient());
                C.max_abs_entry = std::max(C.max_abs_entry, v.max_abs_entry());
                for (const auto& e : v.c)
                    for (const auto& [g, n] : e.terms) elems[g] = true;
            }
    C.distinct_group_elements = elems.size();

    // (T2) d tau_Iso(s_j e_c) = tau_Iso(d(s_j e_c)).
    for (uint32_t q = 1; q <= Q; ++q)
        for (uint32_t j = 1; j <= m_; ++j)
            for (uint32_t c = 0; c < R_->rank(q); ++c) {
                ++C.t2_checks;
                if (boundary(q, value(q, j, c)) != apply(q - 1, basis_boundary(q, j, c)))
                    fail(C.t2_chain_map, "(T2) d tau_Iso != tau_Iso d" + tag(q, j, c));
            }

    // (T3) aug eps tau_Iso(s_j e_c) = 1 in degree 0 (every degree-0 cell is (0, rep, 0)).
    for (uint32_t j = 1; j <= m_; ++j)
        for (uint32_t c = 0; c < R_->rank(0); ++c) {
            ++C.t3_checks;
            int64_t aug = 0;
            for (const auto& e : value(0, j, c).c)
                for (const auto& [g, n] : e.terms) aug = hecke::checked_add<int64_t>(aug, n);
            if (aug != 1) fail(C.t3_augmentation, "(T3) aug eps tau_Iso != 1" + tag(0, j, c));
        }

    // (S) eps tau_Iso = tau^s eps, on canonical chains of Xbulk (q <= 2; above
    // degree 2 both sides vanish, C_q(Xbulk) = 0), and (B).
    auto eps_canon = [&](uint32_t q, const IsoChain& x) {
        CanonChain acc;
        for (uint32_t c = 0; c < x.c.size(); ++c) {
            const Cell& cell = R_->cells(q)[c];
            if (cell.t != 0) continue;
            for (const auto& [g, n] : x.c[c].terms) canon_add(acc, *tess_, mat_of(g), q, rep_kidx_[q][cell.rep], n);
        }
        return acc;
    };
    for (uint32_t q = 0; q <= std::min<uint32_t>(Q, 2); ++q)
        for (uint32_t j = 1; j <= m_; ++j)
            for (uint32_t c = 0; c < R_->rank(q); ++c) {
                const Cell& cell = R_->cells(q)[c];
                ++C.strict_checks;
                const CanonChain lhs = eps_canon(q, value(q, j, c));
                const CanonChain rhs =
                    (cell.t == 0) ? canon_of_terms(*tess_, q, tau_s_chain(j, q, rep_kidx_[q][cell.rep])) : CanonChain{};
                if (lhs != rhs) fail(C.strict_eps, "(S) eps tau_Iso != tau^s eps" + tag(q, j, c));
                if (cell.boundary && cell.t == 0) {
                    ++C.boundary_checks;
                    const CanonChain bd = canon_of_terms(*tess_, q, taub_chain(j, q, rep_kidx_[q][cell.rep]));
                    if (lhs != bd) fail(C.boundary_eps, "(B) eps tau^d_Iso != tau_bd eps" + tag(q, j, c));
                }
            }
    for (uint32_t q = 0; q <= Q; ++q)
        for (uint32_t j = 1; j <= m_; ++j)
            for (uint32_t c = 0; c < R_->rank(q); ++c) {
                if (!R_->cells(q)[c].boundary) continue;
                ++C.boundary_checks;
                const IsoChain& v = value(q, j, c);
                for (uint32_t c2 = 0; c2 < v.c.size(); ++c2)
                    if (!v.c[c2].is_zero() && !R_->cells(q)[c2].boundary) {
                        fail(C.boundary_preserved, "(B) tau_Iso(R_d) leaves R_d" + tag(q, j, c));
                        break;
                    }
            }

    // (T1) on the Schreier generators h_i(x) = s_i x s_{i.x}^{-1}, x in {T, S}.
    // The conjugate g^{-1} h g = [[a, N b], [c / N, d]] is formed here, not by
    // the helper.
    // Slot-11 WP-11: g = diag(1, N) with N = p^nu in general; in the central
    // case Gamma'' = Gamma, m = 1, s_1 = I, the Schreier generators are T and
    // S themselves and conj_g = id.
    const int64_t N = static_cast<int64_t>(N_);
    std::vector<Mat22> gens;
    if (central_) {
        gens = {Mat22::T(), Mat22::S()};
    } else {
        if (!(cosets_->s(1) == Mat22::I()))   // Schreier's lemma needs s_1 = I (slot-10 WP-14 m-3)
            fail(C.t1_twisted_equivariance, "(T1) s_1 != I: the Schreier set need not generate Gamma''");
    }
    for (uint32_t gen : {0u, 1u})
        for (uint32_t i = 1; i <= m_ && !central_; ++i) {
            const Mat22 h = cosets_->cocycle(gen, i);
            const Mat22 x = (gen == 0) ? Mat22::T() : Mat22::S();
            // Schreier: h = s_i x s_{i.x}^{-1}, in Gamma_0(N).
            const Mat22 expect = cosets_->s(i) * x * cosets_->s_inv(cosets_->perm(gen, i));
            if (!(h == expect) || h.det() != 1 || h.c % N != 0)
                fail(C.t1_twisted_equivariance, "(T1) a Schreier generator is not s_i x s_{i.x}^{-1} in Gamma_0(N)");
            gens.push_back(h);
        }
    C.schreier_generators = static_cast<uint32_t>(gens.size());
    const Mat22 us[2] = {Mat22::T(), Mat22::S()};
    for (const Mat22& h : gens) {
        const Mat22 conj = central_ ? h : Mat22{h.a, hecke::checked_mul<int64_t>(N, h.b), h.c / N, h.d};
        for (uint32_t q = 0; q <= Q; ++q)
            for (uint32_t j = 1; j <= m_; ++j)
                for (uint32_t c = 0; c < R_->rank(q); ++c) {
                    ++C.t1_checks;
                    if (on_translate(h * coset_s(j), q, c) != left_multiply(conj, value(q, j, c)))
                        fail(C.t1_twisted_equivariance, "(T1) tau_Iso(h s_j e) != (g^-1 h g) tau_Iso(s_j e)" + tag(q, j, c));
                    for (const Mat22& u : us) {
                        ++C.t1_checks;
                        const Mat22 y = coset_s(j) * u;
                        if (on_translate(h * y, q, c) != left_multiply(conj, on_translate(y, q, c)))
                            fail(C.t1_twisted_equivariance,
                                 "(T1) tau_Iso(h s_j u e) != (g^-1 h g) tau_Iso(s_j u e)" + tag(q, j, c));
                    }
                }
    }
    return C;
}

// ---------------------------------------------------------------------------
// The coset transport Theta^_g ([C, eq:cosettransport])
// ---------------------------------------------------------------------------
template<class Elt>
engine::DenseBuf<Elt> StrictLift::theta_hat(const engine::ModCtx<Elt>& ctx, uint32_t n, uint32_t q) const {
    ENGINE_CHECK_MSG(q <= q_top(), "StrictLift::theta_hat: q > q_top");
    const uint32_t D = n + 1;
    const uint32_t w = R_->rank(q);
    engine::DenseBuf<Elt> M(w * D, w * D);
    std::map<GKey, engine::DenseBuf<Elt>> memo;
    const int64_t pm = static_cast<int64_t>(ctx.p);
    for (uint32_t j = 1; j <= m_; ++j) {
        const Mat22 delta = coset_delta(j);
        for (uint32_t c = 0; c < w; ++c) {
            const IsoChain& v = value(q, j, c);
            for (uint32_t c2 = 0; c2 < w; ++c2)
                for (const auto& [g, nmul] : v.c[c2].terms) {
                    const Mat22 prod = delta * mat_of(g);   // exact, det N, never inverted
                    const GKey key = key_of(prod);
                    auto it = memo.find(key);
                    if (it == memo.end()) it = memo.emplace(key, rho_n(ctx, prod, n)).first;
                    int64_t r = nmul % pm;
                    if (r < 0) r += pm;
                    const Elt coef = static_cast<Elt>(r);
                    if (coef == 0) continue;
                    const auto& Rm = it->second;
                    for (uint32_t a = 0; a < D; ++a)
                        for (uint32_t b = 0; b < D; ++b) {
                            Elt& dst = M.at(c * D + a, c2 * D + b);
                            dst = ctx.add(dst, ctx.mul(coef, Rm.at(a, b)));
                        }
                }
        }
    }
    return M;
}

template engine::DenseBuf<uint32_t> StrictLift::theta_hat<uint32_t>(const engine::ModCtx<uint32_t>&, uint32_t,
                                                                    uint32_t) const;
template engine::DenseBuf<uint64_t> StrictLift::theta_hat<uint64_t>(const engine::ModCtx<uint64_t>&, uint32_t,
                                                                    uint32_t) const;

// ---------------------------------------------------------------------------
bool SameIntegralDatum(const hecke::HeckeTauTableT<uint32_t>& a, const hecke::HeckeTauTableT<uint32_t>& b) {
    if (a.num_cosets() != b.num_cosets()) return false;
    const auto& Ks = IsotropyModel::K_simplices();
    for (uint32_t q = 0; q < 3; ++q)
        for (uint32_t s = 0; s < Ks[q].size(); ++s)
            for (uint32_t i = 1; i <= a.num_cosets(); ++i) {
                const auto& x = a.tau(i, q, s);
                const auto& y = b.tau(i, q, s);
                if (x.size() != y.size()) return false;
                for (size_t k = 0; k < x.size(); ++k)
                    if (x[k].coeff != y[k].coeff || !(x[k].gamma == y[k].gamma) || x[k].sigma_idx != y[k].sigma_idx)
                        return false;
            }
    return true;
}

// ---------------------------------------------------------------------------
StrictLiftBundle::StrictLiftBundle(uint64_t p_hecke, uint32_t q_max, uint32_t ell_ref, uint32_t n_ref)
    : StrictLiftBundle(p_hecke, DoubleCoset::diag_one(1), q_max, ell_ref, n_ref) {}

StrictLiftBundle::StrictLiftBundle(uint64_t p_hecke, DoubleCoset dc, uint32_t q_max, uint32_t ell_ref, uint32_t n_ref)
    : p_(p_hecke), dc_(dc) {
    K_ = std::make_unique<SimplicialComplex>(CreateSL2ZFundamentalDomainTriangulation());
    tess_ = std::make_unique<hecke::SL2ZTessellationTables>(*K_);
    if (dc.central) {
        R_ = std::make_unique<IsotropyModel>(q_max);
        lift_ = std::make_unique<StrictLift>(*R_, *tess_, p_hecke);
        return;
    }
    cosets_ = std::make_unique<hecke::HeckeCosetTablesT<uint32_t>>(p_hecke, dc.nu, ell_ref, n_ref);
    taub_ = std::make_unique<hecke::HeckeBoundaryTauTableT<uint32_t>>(*K_, *tess_, *cosets_);
    tau_s_ = std::make_unique<hecke::HeckeTauTableT<uint32_t>>(*K_, *tess_, *cosets_, ell_ref, n_ref, *taub_);
    R_ = std::make_unique<IsotropyModel>(q_max);
    lift_ = std::make_unique<StrictLift>(*R_, *cosets_, *tau_s_, *taub_);
}

StrictLiftBundle::~StrictLiftBundle() = default;

const hecke::HeckeCosetTablesT<uint32_t>& StrictLiftBundle::cosets() const {
    ENGINE_CHECK_MSG(cosets_ != nullptr, "StrictLiftBundle::cosets: diag(p,p) has no coset tables");
    return *cosets_;
}
const hecke::HeckeBoundaryTauTableT<uint32_t>& StrictLiftBundle::taub() const {
    ENGINE_CHECK_MSG(taub_ != nullptr, "StrictLiftBundle::taub: diag(p,p) has no boundary table");
    return *taub_;
}
const hecke::HeckeTauTableT<uint32_t>& StrictLiftBundle::tau_s() const {
    ENGINE_CHECK_MSG(tau_s_ != nullptr, "StrictLiftBundle::tau_s: diag(p,p) has no tau table");
    return *tau_s_;
}

}  // namespace integral_model
