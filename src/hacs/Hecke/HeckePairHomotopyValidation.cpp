// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 2, WP-8 -- validation of the certified pair homotopy
// eta. See HeckePairHomotopyValidation.h for the independence statement and
// the gate list G1-G11. Every predicate below is re-derived from
// gl2_contract.tex ([C], by label) and evaluated through public accessors.

#include "HeckePairHomotopyValidation.h"

#include "ExactInt.h"
#include "HeckeBoundaryDatum.h"
#include "HeckeBoundaryDiagnostics.h"
#include "HeckeCosetTables.h"
#include "HeckePairHomotopy.h"
#include "HeckeTauTable.h"
#include "SL2ZTessellationTables.h"
#include "../LocalSystemSL2Z.h"      // CreateSL2ZFundamentalDomainTriangulation
#include "../SimplicialCohomology.h" // SimplicialComplex

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>
#include "ValidationSupport.h"

namespace {

using hecke::ChainTerm;
using hecke::EtaVertexPolicy;
using hecke::HeckeBoundaryTauTableT;
using hecke::HeckeCosetTablesT;
using hecke::HeckePairHomotopyT;
using hecke::HeckeTauTableT;
using hecke::KuSimplexKey;
using hecke::Mat22;
using hecke::SL2ZTessellationTables;

using SimplexTable = std::vector<std::vector<std::vector<uint32_t>>>;
using CanonKey = std::array<int64_t, 6>; // {q, sigma_hat, a, b, c, d}
using ChainAcc = std::map<CanonKey, int64_t>;

// =========================================================================
// Section 0: own bookkeeping (nothing here calls a library helper).
// =========================================================================

uint32_t find_simplex_index(const SimplexTable& simplices, uint32_t q, std::vector<uint32_t> verts) {
    return hecke::vsupport::find_simplex_index(simplices, q, std::move(verts),
                                               "HeckePairHomotopyValidation: simplex not found.");
}

uint32_t face_index(const SimplexTable& simplices, uint32_t q, uint32_t sigma, uint32_t f) {
    std::vector<uint32_t> verts = simplices.at(q).at(sigma);
    verts.erase(verts.begin() + static_cast<std::ptrdiff_t>(f));
    return find_simplex_index(simplices, q - 1, verts);
}

void add_term(ChainAcc& acc, const SL2ZTessellationTables& model, uint32_t q, const Mat22& gamma,
              uint32_t sigma, int64_t coeff) {
    if (coeff == 0) return;
    const KuSimplexKey key = model.canonical_key(gamma, q, sigma);
    const CanonKey k = { static_cast<int64_t>(q), static_cast<int64_t>(key.sigma_hat),
                         key.gamma_canon.a, key.gamma_canon.b, key.gamma_canon.c, key.gamma_canon.d };
    acc[k] = hecke::checked_add<int64_t>(acc[k], hecke::checked_mul<int64_t>(coeff, static_cast<int64_t>(key.sign)));
}

ChainAcc acc_of_terms(const SL2ZTessellationTables& model, uint32_t q, const std::vector<ChainTerm>& terms,
                      int64_t scale = 1, const Mat22* left = nullptr) {
    ChainAcc acc;
    for (const auto& t : terms) {
        const Mat22 g = left ? (*left) * t.gamma : t.gamma;
        add_term(acc, model, q, g, t.sigma_idx, scale * static_cast<int64_t>(t.coeff));
    }
    return acc;
}

// The boundary of a q-chain given as a canonical map, as a (q-1)-map (own
// faces: delete the f-th vertex of the increasing vertex list, sign (-1)^f).
ChainAcc own_boundary(const SL2ZTessellationTables& model, const SimplexTable& simplices, uint32_t q,
                      const ChainAcc& chain) {
    ChainAcc out;
    for (const auto& kv : chain) {
        if (kv.second == 0) continue;
        if (kv.first[0] != static_cast<int64_t>(q)) throw std::runtime_error("own_boundary: degree mismatch.");
        const Mat22 gamma{ kv.first[2], kv.first[3], kv.first[4], kv.first[5] };
        const uint32_t sigma = static_cast<uint32_t>(kv.first[1]);
        for (uint32_t f = 0; f <= q; ++f) {
            const int64_t sgn = (f % 2 == 0) ? 1 : -1;
            add_term(out, model, q - 1, gamma, face_index(simplices, q, sigma, f), sgn * kv.second);
        }
    }
    return out;
}

bool acc_equal(const ChainAcc& x, const ChainAcc& y) {
    for (const auto& kv : x) {
        const auto it = y.find(kv.first);
        if (kv.second != ((it == y.end()) ? 0 : it->second)) return false;
    }
    for (const auto& kv : y) {
        if (kv.second != 0 && x.find(kv.first) == x.end()) return false;
    }
    return true;
}

bool acc_is_zero(const ChainAcc& x) {
    for (const auto& kv : x) if (kv.second != 0) return false;
    return true;
}

ChainAcc acc_add(const ChainAcc& x, const ChainAcc& y, int64_t sy) {
    ChainAcc out = x;
    for (const auto& kv : y) out[kv.first] = hecke::checked_add<int64_t>(out[kv.first], sy * kv.second);
    return out;
}

std::vector<ChainTerm> terms_of_acc(const ChainAcc& acc, uint32_t q) {
    std::vector<ChainTerm> out;
    for (const auto& kv : acc) {
        if (kv.second == 0) continue;
        if (kv.first[0] != static_cast<int64_t>(q)) throw std::runtime_error("terms_of_acc: degree mismatch.");
        out.push_back(ChainTerm{ static_cast<int32_t>(kv.second),
                                 Mat22{ kv.first[2], kv.first[3], kv.first[4], kv.first[5] },
                                 static_cast<uint32_t>(kv.first[1]) });
    }
    return out;
}

// Own (T1) translation of a stored table: value(gamma . i(sigma)) from the
// stored slot of its Gamma''-class, gamma = h s_j with j = class_index_of(gamma),
// h = gamma s_j^{-1} in Gamma_0(N), conjugate g^{-1} h g = [[h.a, N h.b],
// [h.c / N, h.d]] written out here; the stored slot's chain lives in degree
// q_store. `fetch(j)` returns the stored terms of slot (j, sigma).
template<class Elt, class Fetch>
ChainAcc own_translate(const SL2ZTessellationTables& model, const HeckeCosetTablesT<Elt>& cosets,
                       const Mat22& gamma, uint32_t q_store, Fetch fetch) {
    const uint32_t j = cosets.class_index_of(gamma);
    const Mat22 h = gamma * cosets.s_inv(j);
    const int64_t N = static_cast<int64_t>(cosets.N());
    if (h.det() != 1 || (h.c % N) != 0) throw std::runtime_error("own_translate: (P1) split failed.");
    const Mat22 conj{ h.a, hecke::checked_mul<int64_t>(N, h.b), h.c / N, h.d };
    if (conj.det() != 1) throw std::runtime_error("own_translate: conjugate not unimodular.");
    return acc_of_terms(model, q_store, fetch(j), 1, &conj);
}

int8_t own_parity(const std::array<uint8_t, 3>& perm, uint32_t q) {
    int inv = 0;
    for (uint32_t i = 0; i + 1 <= q; ++i)
        for (uint32_t j = i + 1; j <= q; ++j)
            if (perm[i] > perm[j]) ++inv;
    return (inv % 2 == 0) ? int8_t(1) : int8_t(-1);
}

// Fixed-seed words in T, T^{-1}, S (a tiny LCG; deterministic across runs).
std::vector<Mat22> random_words(uint32_t count, uint32_t seed) {
    std::vector<Mat22> out;
    uint64_t s = 0x9E3779B97F4A7C15ull ^ (static_cast<uint64_t>(seed) * 0x100000001B3ull);
    const auto next = [&]() { s = s * 6364136223846793005ull + 1442695040888963407ull; return static_cast<uint32_t>(s >> 33); };
    const Mat22 gens[3] = { Mat22::T(), Mat22::T().inv_unimodular(), Mat22::S() };
    for (uint32_t w = 0; w < count; ++w) {
        const uint32_t len = 1 + next() % 6;
        Mat22 g = Mat22::I();
        for (uint32_t k = 0; k < len; ++k) g = g * gens[next() % 3];
        out.push_back(g);
    }
    return out;
}

struct Ctx {
    uint64_t p;
    uint32_t nu;
    uint64_t ell;
    uint32_t n;
    const char* width;
    EtaVertexPolicy policy;
    std::string str() const {
        std::ostringstream oss;
        oss << "p=" << p << " nu=" << nu << " ell=" << ell << " n=" << n << " " << width << " policy="
            << hecke::to_string(policy);
        return oss.str();
    }
};

struct Stats {
    uint32_t tables = 0;
    uint32_t checked = 0;
    uint32_t slots = 0;
    uint32_t nonzero = 0;
    uint32_t eta_zero_true = 0;
    uint32_t fills = 0;
    uint32_t max_rings = 0;
    int64_t max_coeff = 0;
    uint32_t policies_differ_slots = 0;
    uint32_t walk_compared = 0;
    uint32_t tamper_cases = 0;
};

void fail(const Ctx& ctx, const std::string& what, uint32_t& failures) {
    std::cout << "HeckePairHomotopyValidation: FAILED [" << ctx.str() << "] " << what << std::endl;
    ++failures;
}

// =========================================================================
// Section 1: G1-G6, G9, G11 on one table.
// =========================================================================

template<class Elt>
uint32_t check_table(const SimplexTable& simplices, const SL2ZTessellationTables& model,
                     const HeckeCosetTablesT<Elt>& cosets, const HeckeTauTableT<Elt>& tau,
                     const HeckeBoundaryTauTableT<Elt>& taub, const HeckePairHomotopyT<Elt>& eta, const Ctx& ctx,
                     Stats& st) {
    uint32_t failures = 0;
    uint32_t& checked = st.checked;
    const uint32_t m = cosets.num_cosets();
    const int64_t N = static_cast<int64_t>(cosets.N());

    // G1: the mandatory certification, re-asserted.
    ++checked;
    if (eta.certify() != 0) fail(ctx, "certify() != 0 on the completed object", failures);

    // G2: shape, identity of the tables, record fields, degrees.
    ++checked;
    if (&eta.cosets() != &cosets || &eta.model() != &model || &eta.tau() != &tau || &eta.tau_boundary() != &taub ||
        &tau.cosets() != &cosets || &taub.cosets() != &cosets) {
        fail(ctx, "the eta-table does not refer to the given tau / tau_bd / coset datum (obl:eta(i))", failures);
    }
    ++checked;
    if (eta.N() != cosets.N() || eta.num_cosets() != m || tau.used_mod_ell_solve()) {
        fail(ctx, "N / m mismatch or a non-integral tau", failures);
    }
    ++checked;
    if (std::strcmp(eta.solve_regime(), "Z") != 0 || !eta.integral() || std::strcmp(eta.solve_regime_tau(), "Z") != 0 ||
        std::strcmp(eta.solve_regime_tau_boundary(), "Z") != 0) {
        fail(ctx, "a solve regime is not Z (obl:eta(iv), (vi))", failures);
    }
    ++checked;
    if (eta.num_vertex_slots_multi_path() != 0) fail(ctx, "a vertex slot needed more than one path on the shipped tables", failures);
    ++checked;
    {
        const std::string js = eta.to_json();
        if (js.find("\"schema\":\"hecke_pair_homotopy/v1\"") == std::string::npos || js.find("\"built_against\":{\"tau_fnv64\":\"0x") == std::string::npos ||
            js.find("\"regimes\":{\"tau\":\"Z\",\"tau_bd\":\"Z\",\"eta\":\"Z\"}") == std::string::npos ||
            js.find("\"degrees\":[0,1]") == std::string::npos) {
            fail(ctx, "to_json() lacks the schema / built_against hashes / regimes / degrees", failures);
        }
    }
    ++checked;
    const uint32_t dim_X = static_cast<uint32_t>(simplices.size()) - 1;
    const uint32_t expect_deg = std::min(taub.num_degrees(), dim_X);
    if (eta.num_degrees() != expect_deg || eta.num_degrees() != 2 || eta.q_max() != 1 || eta.truncated() ||
        std::strcmp(eta.q_max_source(), "derived") != 0 || eta.degrees() != std::vector<uint32_t>{ 0, 1 }) {
        fail(ctx, "degree bookkeeping (open:ell2 rider (i)): expected degrees [0, 1], q_max = 1 derived", failures);
    }

    // G2 (structural) and G11 (counts), own recount.
    uint32_t own_slots = 0, own_nonzero = 0, own_terms = 0;
    int64_t own_max_coeff = 0;
    uint64_t own_B_tab = 0;
    for (uint32_t q = 0; q < eta.num_degrees(); ++q) {
        for (uint32_t sigma : model.boundary_simplices(q)) {
            for (uint32_t j = 1; j <= m; ++j) {
                const auto& terms = eta.eta(j, q, sigma);
                ++own_slots;
                if (!terms.empty()) ++own_nonzero;
                for (const auto& t : terms) {
                    ++own_terms;
                    ++checked;
                    if (t.coeff == 0) fail(ctx, "zero coefficient stored", failures);
                    if (t.gamma.det() != 1) fail(ctx, "det gamma_t != 1 (obl:eta(i))", failures);
                    if (t.sigma_idx >= simplices[q + 1].size()) fail(ctx, "sigma_t is not a (q+1)-simplex (obl:eta(i))", failures);
                    else {
                        const KuSimplexKey key = model.canonical_key(t.gamma, q + 1, t.sigma_idx);
                        if (key.sigma_hat != t.sigma_idx || !(key.gamma_canon == t.gamma) || key.sign != 1)
                            fail(ctx, "term not in canonical form (obl:eta(i))", failures);
                    }
                    const int64_t ac = (t.coeff < 0) ? -static_cast<int64_t>(t.coeff) : t.coeff;
                    own_max_coeff = std::max(own_max_coeff, ac);
                    own_B_tab = std::max(own_B_tab, static_cast<uint64_t>(t.gamma.max_abs()));
                }
            }
        }
    }
    ++checked;
    const uint32_t expect_slots = m * static_cast<uint32_t>(model.boundary_simplices(0).size() + model.boundary_simplices(1).size());
    if (own_slots != expect_slots || eta.num_slots() != own_slots ||
        eta.num_constructed_slots() + eta.num_derived_slots() != own_slots || eta.num_nonzero_slots() != own_nonzero ||
        eta.num_terms() != own_terms || eta.max_abs_coeff() != own_max_coeff || eta.eta_zero() != (own_nonzero == 0) ||
        eta.num_nonzero_slots(0) + eta.num_nonzero_slots(1) != own_nonzero) {
        fail(ctx, "slot / term / coefficient counts disagree with an own recount (G11)", failures);
    }
    // G5 (bounds): recount, B, the int64 tier.
    ++checked;
    const uint64_t four_N_sq = 4ull * static_cast<uint64_t>(N) * static_cast<uint64_t>(N);
    const uint64_t own_B = std::max(own_B_tab, four_N_sq);
    if (eta.certified_bound_B_tab() != own_B_tab || eta.certified_bound_B() != own_B || own_B > 2147483647ull) {
        fail(ctx, "entry bound recount / int64 tier (obl:eta(iii), lem:bounds(5))", failures);
    }

    // G3: the homotopy identity on every stored slot, own boundary and faces.
    for (uint32_t q = 0; q < eta.num_degrees(); ++q) {
        for (uint32_t sigma : model.boundary_simplices(q)) {
            for (uint32_t j = 1; j <= m; ++j) {
                ChainAcc lhs = own_boundary(model, simplices, q + 1, acc_of_terms(model, q + 1, eta.eta(j, q, sigma)));
                if (q >= 1) {
                    for (uint32_t f = 0; f <= q; ++f) {
                        const uint32_t face = face_index(simplices, q, sigma, f);
                        const int64_t sgn = (f % 2 == 0) ? 1 : -1;
                        lhs = acc_add(lhs, acc_of_terms(model, q, eta.eta(j, q - 1, face)), sgn);
                    }
                }
                const ChainAcc rhs = acc_add(acc_of_terms(model, q, taub.tau(j, q, sigma)),
                                             acc_of_terms(model, q, tau.tau(j, q, sigma)), -1);
                ++checked;
                if (!acc_equal(lhs, rhs)) {
                    std::ostringstream oss;
                    oss << "eq:pairhomotopy fails on the stored slot (j=" << j << ", q=" << q << ", sigma=" << sigma << ")";
                    fail(ctx, oss.str(), failures);
                }
                ++checked;
                if (!acc_equal(rhs, acc_of_terms(model, q, eta.defect(j, q, sigma))))
                    fail(ctx, "defect() disagrees with the own (j tau_bd - tau j)", failures);
            }
        }
    }

    // G4: the identity on random Gamma-translates, all three tables own-translated;
    // the library helpers cross-checked against the own translation.
    {
        const std::vector<Mat22> words = random_words(12, static_cast<uint32_t>(cosets.N() * 7919u + ctx.n));
        for (uint32_t q = 0; q < eta.num_degrees(); ++q) {
            for (uint32_t sigma : model.boundary_simplices(q)) {
                for (const Mat22& gamma : words) {
                    const auto fetch_eta = [&](uint32_t s) { return [&, s](uint32_t j) { return eta.eta(j, q, s); }; };
                    const ChainAcc eg = own_translate(model, cosets, gamma, q + 1, fetch_eta(sigma));
                    ChainAcc lhs = own_boundary(model, simplices, q + 1, eg);
                    if (q >= 1) {
                        for (uint32_t f = 0; f <= q; ++f) {
                            const uint32_t face = face_index(simplices, q, sigma, f);
                            const int64_t sgn = (f % 2 == 0) ? 1 : -1;
                            const ChainAcc ef = own_translate(model, cosets, gamma, q,
                                                              [&, face](uint32_t j) { return eta.eta(j, q - 1, face); });
                            lhs = acc_add(lhs, ef, sgn);
                        }
                    }
                    const ChainAcc tb = own_translate(model, cosets, gamma, q, [&](uint32_t j) { return taub.tau(j, q, sigma); });
                    const ChainAcc tt = own_translate(model, cosets, gamma, q, [&](uint32_t j) { return tau.tau(j, q, sigma); });
                    const ChainAcc rhs = acc_add(tb, tt, -1);
                    ++checked;
                    if (!acc_equal(lhs, rhs)) fail(ctx, "eq:pairhomotopy fails on a random translate (own translation)", failures);
                    ++checked;
                    if (!acc_equal(eg, acc_of_terms(model, q + 1, eta.eta_on_translate(gamma, q, sigma))))
                        fail(ctx, "eta_on_translate disagrees with the own (T1) translation", failures);
                    ++checked;
                    if (!acc_equal(rhs, acc_of_terms(model, q, eta.defect_on_translate(gamma, q, sigma))))
                        fail(ctx, "defect_on_translate disagrees with the own (T1) translation", failures);
                }
            }
        }
    }

    // G5: the (T1) bookkeeping over every stored transporter between boundary
    // simplices, own translation and own parity.
    for (uint32_t q = 0; q < eta.num_degrees(); ++q) {
        const auto& list = model.boundary_simplices(q);
        for (uint32_t sigma : list) {
            for (uint32_t sigma2 : list) {
                if (!model.same_groupoid_orbit(q, sigma, sigma2)) continue;
                const auto& ts = model.transporter(q, sigma, sigma2);
                const auto& perms = model.transporter_vertex_perms(q, sigma, sigma2);
                for (size_t k = 0; k < ts.size(); ++k) {
                    for (uint32_t j = 1; j <= m; ++j) {
                        const Mat22 gamma = cosets.s(j) * ts[k];
                        const ChainAcc rhs = own_translate(model, cosets, gamma, q + 1,
                                                           [&](uint32_t jj) { return eta.eta(jj, q, sigma2); });
                        const int8_t eps = own_parity(perms[k], q);
                        const ChainAcc lhs = acc_of_terms(model, q + 1, eta.eta(j, q, sigma), (eps < 0) ? -1 : 1);
                        ++checked;
                        if (!acc_equal(lhs, rhs)) fail(ctx, "(T1) bookkeeping fails across a stored transporter (obl:eta(iii))", failures);
                    }
                }
            }
        }
    }

    // G9: eta_zero against WP-7's cusp-support walk (uint32_t tables only:
    // the walk is declared on the u32 aliases).
    if constexpr (std::is_same<Elt, uint32_t>::value) {
        const hecke::CuspSupportWalk walk = hecke::cusp_support_walk(tau, taub);
        ++checked;
        if (walk.eta_zero() != eta.eta_zero()) fail(ctx, "eta_zero disagrees with the cusp-support walk's verdict", failures);
        ++checked;
        if (walk.differ_per_degree.empty() || walk.differ_per_degree[0] != eta.num_nonzero_slots(0))
            fail(ctx, "the walk's differing vertex slots != the nonzero vertex slots of eta", failures);
        for (uint32_t v : model.boundary_simplices(0)) {
            for (uint32_t j = 1; j <= m; ++j) {
                const bool differ = !acc_equal(acc_of_terms(model, 0, tau.tau(j, 0, v)), acc_of_terms(model, 0, taub.tau(j, 0, v)));
                ++checked;
                if (differ != !eta.eta(j, 0, v).empty()) fail(ctx, "a vertex slot's eta is nonzero iff tau != tau_bd there: violated", failures);
            }
        }
        ++st.walk_compared;
    }

    st.slots += own_slots;
    st.nonzero += own_nonzero;
    if (eta.eta_zero()) ++st.eta_zero_true;
    st.fills += eta.num_fills();
    st.max_rings = std::max(st.max_rings, eta.max_rings_used());
    st.max_coeff = std::max(st.max_coeff, own_max_coeff);
    return failures;
}

// =========================================================================
// Section 2: G10 -- lem:etaunique on the two policies' tables.
// =========================================================================

template<class Elt>
uint32_t check_uniqueness(const SimplexTable& simplices, const SL2ZTessellationTables& model,
                          const HeckeCosetTablesT<Elt>& cosets, const HeckePairHomotopyT<Elt>& eta,
                          const HeckePairHomotopyT<Elt>& eta2, const Ctx& ctx, Stats& st) {
    uint32_t failures = 0;
    const uint32_t m = cosets.num_cosets();
    ++st.checked;
    if (eta2.certify() != 0 || eta2.policy() != EtaVertexPolicy::LegacyBfsPath) fail(ctx, "the second policy's table does not certify", failures);
    // L(x) on every vertex slot: the unique 2-chain with boundary xi(x).
    std::map<std::pair<uint32_t, uint32_t>, ChainAcc> L; // (j, v) -> 2-map
    for (uint32_t v : model.boundary_simplices(0)) {
        for (uint32_t j = 1; j <= m; ++j) {
            const ChainAcc xi = acc_add(acc_of_terms(model, 1, eta2.eta(j, 0, v)), acc_of_terms(model, 1, eta.eta(j, 0, v)), -1);
            if (!acc_is_zero(xi)) ++st.policies_differ_slots;
            ++st.checked;
            if (!acc_is_zero(own_boundary(model, simplices, 1, xi))) fail(ctx, "(eta' - eta)(x) is not a 1-cycle", failures);
            std::vector<ChainTerm> Lx;
            try {
                Lx = eta.unique_fill(1, terms_of_acc(xi, 1));
            } catch (const std::exception& e) {
                fail(ctx, std::string("unique_fill failed on (eta' - eta)(x): ") + e.what(), failures);
                continue;
            }
            const ChainAcc Lacc = acc_of_terms(model, 2, Lx);
            ++st.checked;
            if (!acc_equal(own_boundary(model, simplices, 2, Lacc), xi)) fail(ctx, "boundary L(x) != (eta' - eta)(x) (lem:etaunique)", failures);
            L[{ j, v }] = Lacc;
        }
    }
    // Edge slots: (eta' - eta)(e) == -L(boundary e) = -(L(s_j v_b) - L(s_j v_a)), forced.
    for (uint32_t e : model.boundary_simplices(1)) {
        const uint32_t va = simplices[1][e][0];
        const uint32_t vb = simplices[1][e][1];
        for (uint32_t j = 1; j <= m; ++j) {
            const ChainAcc xi = acc_add(acc_of_terms(model, 2, eta2.eta(j, 1, e)), acc_of_terms(model, 2, eta.eta(j, 1, e)), -1);
            if (!acc_is_zero(xi)) ++st.policies_differ_slots;
            const ChainAcc Lb = acc_add(L[{ j, vb }], L[{ j, va }], -1);
            ++st.checked;
            if (!acc_equal(xi, acc_add(ChainAcc{}, Lb, -1))) fail(ctx, "(eta' - eta)(e) != -L(boundary e) (lem:etaunique, degree 1)", failures);
        }
    }
    return failures;
}

// =========================================================================
// Section 3: G8 -- the tamper arm.
// =========================================================================

template<class Elt>
uint32_t check_tampering(const SL2ZTessellationTables& model, const HeckeCosetTablesT<Elt>& cosets,
                         const HeckePairHomotopyT<Elt>& eta, const Ctx& ctx, Stats& st) {
    uint32_t failures = 0;
    const uint32_t m = cosets.num_cosets();
    // A nonzero vertex slot and a nonzero edge slot.
    uint32_t jv = 0, v = 0, je = 0, e = 0;
    for (uint32_t vv : model.boundary_simplices(0))
        for (uint32_t j = 1; j <= m && jv == 0; ++j)
            if (!eta.eta(j, 0, vv).empty()) { jv = j; v = vv; }
    for (uint32_t ee : model.boundary_simplices(1))
        for (uint32_t j = 1; j <= m && je == 0; ++j)
            if (!eta.eta(j, 1, ee).empty()) { je = j; e = ee; }
    ++st.checked;
    if (jv == 0 || je == 0) {
        fail(ctx, "tamper arm needs a nonzero vertex slot and a nonzero edge slot", failures);
        return failures;
    }
    const std::vector<ChainTerm> orig_v = eta.eta(jv, 0, v);
    const std::vector<ChainTerm> orig_e = eta.eta(je, 1, e);
    ++st.checked;
    if (eta.certify_with_override(jv, 0, v, orig_v) != 0 || eta.certify_with_override(je, 1, e, orig_e) != 0)
        fail(ctx, "the untampered override does not certify", failures);

    struct Case { const char* name; uint32_t j, q, s; std::vector<ChainTerm> chain; uint32_t expect; };
    std::vector<Case> cases;
    { auto c = orig_v; c.erase(c.begin()); cases.push_back({ "term dropped (vertex)", jv, 0, v, c, 0 }); }
    { auto c = orig_v; c[0].coeff += 1; cases.push_back({ "coefficient changed", jv, 0, v, c, 0 }); }
    cases.push_back({ "nonzero vertex chain zeroed", jv, 0, v, {}, 30 });
    cases.push_back({ "nonzero edge chain zeroed", je, 1, e, {}, 31 });
    { auto c = orig_v; c[0].gamma = c[0].gamma * Mat22::T(); cases.push_back({ "transporter moved by T", jv, 0, v, c, 0 }); }
    { auto c = orig_v; c[0].gamma = Mat22{ 1, 1, 0, 2 }; cases.push_back({ "det != 1", jv, 0, v, c, 11 }); }
    { auto c = orig_v; c[0].gamma = c[0].gamma.neg(); cases.push_back({ "non-canonical sign", jv, 0, v, c, 12 }); }
    { auto c = orig_v; c[0].sigma_idx = 99; cases.push_back({ "wrong-dimension simplex", jv, 0, v, c, 14 }); }
    // A chain moved to a Gamma''-related slot without conjugation: slot
    // (j', v') := eta(jv, v) for the first other slot of the same orbit whose
    // stored chain differs from it.
    {
        bool done = false;
        for (uint32_t v2 : model.boundary_simplices(0)) {
            if (done) break;
            if (!model.same_groupoid_orbit(0, v, v2)) continue;
            for (uint32_t j2 = 1; j2 <= m && !done; ++j2) {
                if (j2 == jv && v2 == v) continue;
                if (!acc_equal(acc_of_terms(model, 1, eta.eta(j2, 0, v2)), acc_of_terms(model, 1, orig_v))) {
                    cases.push_back({ "chain moved to a Gamma''-related slot", j2, 0, v2, orig_v, 0 });
                    done = true;
                }
            }
        }
    }
    for (const Case& c : cases) {
        ++st.checked;
        ++st.tamper_cases;
        const uint32_t code = eta.certify_with_override(c.j, c.q, c.s, c.chain);
        if (code == 0 || (c.expect != 0 && code != c.expect)) {
            std::ostringstream oss;
            oss << "tamper '" << c.name << "' returned code " << code << " (expected " << (c.expect ? std::to_string(c.expect) : "nonzero") << ")";
            fail(ctx, oss.str(), failures);
        }
    }
    return failures;
}

// =========================================================================
// Section 4: the grid driver (G7 signatures across ell / n / width).
// =========================================================================

template<class Elt>
std::string signature_of(const SL2ZTessellationTables& model, const HeckePairHomotopyT<Elt>& eta) {
    std::ostringstream oss;
    for (uint32_t q = 0; q < eta.num_degrees(); ++q)
        for (uint32_t sigma : model.boundary_simplices(q))
            for (uint32_t j = 1; j <= eta.num_cosets(); ++j) {
                oss << q << ":" << sigma << ":" << j << "|";
                for (const auto& t : eta.eta(j, q, sigma))
                    oss << t.coeff << "*[" << t.gamma.a << "," << t.gamma.b << "," << t.gamma.c << "," << t.gamma.d << "]." << t.sigma_idx << ";";
                oss << "\n";
            }
    oss << "fnv=" << eta.table_fnv64();
    return oss.str();
}

template<class Elt>
uint32_t run_width(const SimplicialComplex& K, const SimplexTable& simplices, const SL2ZTessellationTables& model,
                   const std::vector<std::pair<uint64_t, uint32_t>>& prs, const std::vector<uint64_t>& ells,
                   const std::vector<uint32_t>& ns, const char* width,
                   std::map<std::pair<uint64_t, int>, std::string>& sigs, Stats& st, bool tamper, bool print_p2) {
    uint32_t failures = 0;
    for (const auto& pr : prs) {
        for (uint64_t ell : ells) {
            for (uint32_t n : ns) {
                const Ctx ctx{ pr.first, pr.second, ell, n, width, EtaVertexPolicy::CanonicalTreePath };
                try {
                    const HeckeCosetTablesT<Elt> cosets(pr.first, pr.second, static_cast<Elt>(ell), n);
                    const HeckeTauTableT<Elt> tau(K, model, cosets, static_cast<Elt>(ell), n);
                    const HeckeBoundaryTauTableT<Elt> taub(K, model, cosets);
                    const HeckePairHomotopyT<Elt> eta(tau, taub, EtaVertexPolicy::CanonicalTreePath);
                    const HeckePairHomotopyT<Elt> eta2(tau, taub, EtaVertexPolicy::LegacyBfsPath);
                    st.tables += 2;
                    failures += check_table(simplices, model, cosets, tau, taub, eta, ctx, st);
                    Ctx ctx2 = ctx;
                    ctx2.policy = EtaVertexPolicy::LegacyBfsPath;
                    failures += check_table(simplices, model, cosets, tau, taub, eta2, ctx2, st);
                    failures += check_uniqueness(simplices, model, cosets, eta, eta2, ctx, st);
                    // G7: ell / n / width independence, keyed by (N, policy).
                    for (int pol = 0; pol < 2; ++pol) {
                        const std::string sig = signature_of(model, pol == 0 ? eta : eta2);
                        const auto key = std::make_pair(cosets.N(), pol);
                        const auto it = sigs.find(key);
                        ++st.checked;
                        if (it == sigs.end()) sigs.emplace(key, sig);
                        else if (it->second != sig) fail(pol == 0 ? ctx : ctx2, "table differs from the first table at the same N (ell/n/width dependence)", failures);
                    }
                    if (tamper && ell == ells.front() && n == ns.front() && pr.first <= 5 && pr.second == 1) {
                        failures += check_tampering(model, cosets, eta, ctx, st);
                    }
                    if (print_p2 && pr.first == 2 && pr.second == 1 && ell == ells.front() && n == ns.front()) {
                        std::cout << eta.to_string();
                        std::cout << "PAIR_HOMOTOPY_RECORD " << eta.to_json() << std::endl;
                        std::cout << "PAIR_HOMOTOPY_RECORD " << eta2.to_json() << std::endl;
                    }
                    if (print_p2 && ell == ells.front() && n == ns.front()) {
                        // One recorded line per level (the table is ell/n-independent).
                        std::cout << "PAIR_HOMOTOPY_CELL p=" << pr.first << " nu=" << pr.second << " m=" << cosets.num_cosets()
                                  << " slots=" << eta.num_slots() << " derived=" << eta.num_derived_slots()
                                  << " nonzero=" << eta.num_nonzero_slots() << " nonzero_q0=" << eta.num_nonzero_slots(0)
                                  << " nonzero_q1=" << eta.num_nonzero_slots(1) << " fills=" << eta.num_fills()
                                  << " max_rings=" << eta.max_rings_used() << " terms=" << eta.num_terms()
                                  << " max_coeff=" << eta.max_abs_coeff() << " B_tab=" << eta.certified_bound_B_tab()
                                  << " B=" << eta.certified_bound_B() << " fnv=0x" << std::hex << eta.table_fnv64()
                                  << " fnv_legacy=0x" << eta2.table_fnv64() << std::dec << std::endl;
                    }
                } catch (const std::exception& e) {
                    std::cout << "HeckePairHomotopyValidation: EXCEPTION [" << ctx.str() << "]: " << e.what() << std::endl;
                    ++failures;
                }
            }
        }
    }
    return failures;
}

} // namespace

uint32_t HeckePairHomotopyValidation() {
    const auto t0 = std::chrono::steady_clock::now();
    uint32_t failures = 0;
    Stats st;
    try {
        const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
        const SL2ZTessellationTables model(K);
        const SimplexTable& simplices = K.get_simplices();

        std::map<std::pair<uint64_t, int>, std::string> sigs; // (N, policy) -> table (shared across widths)
        const std::vector<std::pair<uint64_t, uint32_t>> prs32 = {
            { 2, 1 }, { 3, 1 }, { 5, 1 }, { 7, 1 }, { 11, 1 }, { 2, 2 }, { 3, 2 }
        };
        const std::vector<uint64_t> ells32 = { 2, 3, 5, 691, 134217757ull };
        const std::vector<uint32_t> ns = { 0, 2 };
        failures += run_width<uint32_t>(K, simplices, model, prs32, ells32, ns, "u32", sigs, st,
                                        /*tamper=*/true, /*print_p2=*/true);

        const std::vector<std::pair<uint64_t, uint32_t>> prs64 = { { 2, 1 }, { 3, 1 }, { 2, 2 } };
        const std::vector<uint64_t> ells64 = { 2305843009213693951ull /* 2^61-1 */, 4294967311ull /* 2^32+15 */ };
        const std::vector<uint32_t> ns64 = { 2 };
        failures += run_width<uint64_t>(K, simplices, model, prs64, ells64, ns64, "u64", sigs, st,
                                        /*tamper=*/false, /*print_p2=*/false);

        // G10 is not vacuous: the two policies differ somewhere on the grid.
        ++st.checked;
        if (st.policies_differ_slots == 0) {
            std::cout << "HeckePairHomotopyValidation: FAILED the two vertex-chain policies never differ (G10 vacuous)" << std::endl;
            ++failures;
        }
    } catch (const std::exception& e) {
        std::cout << "HeckePairHomotopyValidation: EXCEPTION during setup: " << e.what() << std::endl;
        return 1000;
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "PAIR_HOMOTOPY tables=" << st.tables << " checks=" << st.checked << " failures=" << failures
              << " slots=" << st.slots << " nonzero=" << st.nonzero << " eta_zero=" << st.eta_zero_true << "/" << st.tables
              << " fills=" << st.fills << " max_rings=" << st.max_rings << " max_coeff=" << st.max_coeff
              << " policies_differ=" << st.policies_differ_slots << " walk_compared=" << st.walk_compared
              << " tamper_cases=" << st.tamper_cases << " (" << secs << " s)" << std::endl;
    return failures;
}
