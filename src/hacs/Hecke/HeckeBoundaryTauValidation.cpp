// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions WP-4a -- validation of the certified boundary datum tau_bd.
// See HeckeBoundaryTauValidation.h for the independence statement and the
// arm list. Every predicate below is re-derived from gl2_contract.tex
// ([C], by label) and evaluated through public accessors only.

#include "HeckeBoundaryTauValidation.h"

#include "HeckeBoundaryDatum.h"
#include "HeckeCosetTables.h"
#include "SL2ZTessellationTables.h"
#include "ExactInt.h"
#include "../LocalSystemSL2Z.h"      // CreateSL2ZFundamentalDomainTriangulation
#include "../SimplicialCohomology.h" // SimplicialComplex

#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include "ValidationSupport.h"

namespace {

using hecke::BoundaryCoset;
using hecke::ChainTerm;
using hecke::CuspKey;
using hecke::HeckeBoundaryTauTableT;
using hecke::HeckeCosetTablesT;
using hecke::KuSimplexKey;
using hecke::Mat22;
using hecke::SL2ZTessellationTables;

using SimplexTable = std::vector<std::vector<std::vector<uint32_t>>>;

// =========================================================================
// Section 0: our own bookkeeping (nothing here calls a private helper).
// =========================================================================

uint32_t find_simplex_index(const SimplexTable& simplices, uint32_t q, std::vector<uint32_t> verts) {
    return hecke::vsupport::find_simplex_index(simplices, q, std::move(verts),
                                               "HeckeBoundaryTauValidation: simplex not found.");
}

uint32_t face_index(const SimplexTable& simplices, uint32_t q, uint32_t sigma, uint32_t f) {
    std::vector<uint32_t> verts = simplices.at(q).at(sigma);
    verts.erase(verts.begin() + static_cast<std::ptrdiff_t>(f));
    return find_simplex_index(simplices, q - 1, verts);
}

using CanonKey = std::array<int64_t, 6>; // {q, sigma_hat, a, b, c, d}
using ChainAcc = std::map<CanonKey, int64_t>;

void add_term(ChainAcc& acc, const SL2ZTessellationTables& model, uint32_t q, const Mat22& gamma,
              uint32_t sigma, int64_t coeff) {
    if (coeff == 0) return;
    const KuSimplexKey key = model.canonical_key(gamma, q, sigma);
    const CanonKey k{ static_cast<int64_t>(key.q), static_cast<int64_t>(key.sigma_hat),
                      key.gamma_canon.a, key.gamma_canon.b, key.gamma_canon.c, key.gamma_canon.d };
    acc[k] += coeff * static_cast<int64_t>(key.sign);
}

void prune_zero(ChainAcc& acc) {
    for (auto it = acc.begin(); it != acc.end();) {
        if (it->second == 0) it = acc.erase(it); else ++it;
    }
}

bool chains_equal(ChainAcc a, ChainAcc b) {
    prune_zero(a);
    prune_zero(b);
    return a == b;
}

ChainAcc terms_to_acc(const SL2ZTessellationTables& model, uint32_t q, const std::vector<ChainTerm>& terms) {
    ChainAcc acc;
    for (const auto& t : terms) add_term(acc, model, q, t.gamma, t.sigma_idx, t.coeff);
    return acc;
}

std::string acc_to_string(const ChainAcc& acc) {
    if (acc.empty()) return "0";
    std::ostringstream oss;
    bool first = true;
    for (const auto& kv : acc) {
        if (!first) oss << " + ";
        first = false;
        oss << kv.second << "*(q=" << kv.first[0] << ",sigma_hat=" << kv.first[1] << ",gamma=[["
            << kv.first[2] << "," << kv.first[3] << "],[" << kv.first[4] << "," << kv.first[5] << "]])";
    }
    return oss.str();
}

// Our own P^1(Q) arithmetic (independent of hecke::CuspKey).
struct Cusp {
    int64_t a, c;
    bool operator==(const Cusp& r) const { return a == r.a && c == r.c; }
    bool operator!=(const Cusp& r) const { return !(*this == r); }
};

Cusp own_normalise(int64_t a, int64_t c) {
    const int64_t g = std::gcd(a, c);
    if (g == 0) throw std::runtime_error("HeckeBoundaryTauValidation: (0:0).");
    a /= g;
    c /= g;
    if (c < 0 || (c == 0 && a < 0)) { a = -a; c = -c; }
    return Cusp{ a, c };
}

Cusp own_cusp_of(const Mat22& gamma) { return own_normalise(gamma.a, gamma.c); } // gamma . infinity

// g^{-1} = diag(1, 1/N) acts on H by z -> N z, so g^{-1} (a : c) = (N a : c).
Cusp own_g_inverse(uint64_t N, const Cusp& c) {
    return own_normalise(hecke::checked_mul<int64_t>(static_cast<int64_t>(N), c.a), c.c);
}

// Our own (T1bd) translation ([C, def:taubdry]): gamma = h s_j with
// j = class_index_of(gamma), h = gamma s_j^{-1} in Gamma_0(N), and
// tau_bd(gamma . i(sigma)) = (g^{-1} h g) . tau_bd(s_j . i(sigma)) with the
// exact integral conjugate g^{-1} h g = [[h.a, N h.b],[h.c / N, h.d]].
template<class Elt>
ChainAcc own_translate(const SL2ZTessellationTables& model, const HeckeCosetTablesT<Elt>& cosets,
                       const HeckeBoundaryTauTableT<Elt>& table, const Mat22& gamma, uint32_t q,
                       uint32_t sigma, uint32_t& failures) {
    const int64_t N = static_cast<int64_t>(cosets.N());
    const uint32_t j = cosets.class_index_of(gamma);
    const Mat22 h = gamma * cosets.s_inv(j);
    ChainAcc out;
    if (h.det() != 1 || h.c % N != 0) {
        std::cout << "HeckeBoundaryTauValidation: (P1) split failed for gamma." << std::endl;
        ++failures;
        return out;
    }
    const Mat22 conj{ h.a, hecke::checked_mul<int64_t>(N, h.b), h.c / N, h.d };
    for (const auto& t : table.tau(j, q, sigma)) {
        add_term(out, model, q, conj * t.gamma, t.sigma_idx, t.coeff);
    }
    return out;
}

// Fixed-seed xorshift64 (deterministic across runs and builds).
struct XorShift64 {
    uint64_t state;
    explicit XorShift64(uint64_t seed) : state(seed) {}
    uint64_t next() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    }
    uint64_t below(uint64_t n) { return next() % n; }
};

struct Ctx {
    uint64_t p;
    uint32_t nu;
    uint64_t ell;
    uint32_t n;
    const char* width;
    std::string str() const {
        std::ostringstream oss;
        oss << "p=" << p << ",nu=" << nu << ",ell=" << ell << ",n=" << n << "," << width;
        return oss.str();
    }
};

// A width-independent signature of a table: (q, sigma, j, terms...) in order.
using Signature = std::vector<std::array<int64_t, 9>>;

template<class Elt>
Signature signature_of(const SL2ZTessellationTables& model, const HeckeBoundaryTauTableT<Elt>& table) {
    Signature sig;
    for (uint32_t q = 0; q < table.num_degrees(); ++q) {
        for (uint32_t sigma : model.boundary_simplices(q)) {
            for (uint32_t j = 1; j <= table.num_cosets(); ++j) {
                for (const auto& t : table.tau(j, q, sigma)) {
                    sig.push_back({ q, sigma, j, t.coeff, t.gamma.a, t.gamma.b, t.gamma.c, t.gamma.d,
                                    static_cast<int64_t>(t.sigma_idx) });
                }
                sig.push_back({ q, sigma, j, 0, 0, 0, 0, 0, -1 }); // slot terminator
            }
        }
    }
    return sig;
}

// =========================================================================
// Section 1: the per-table battery ([C, obl:taubdry](i)-(iv), prop:bdrycosets).
// =========================================================================

template<class Elt>
uint32_t check_table(const SimplexTable& simplices, const SL2ZTessellationTables& model,
                     const HeckeCosetTablesT<Elt>& cosets, const HeckeBoundaryTauTableT<Elt>& table,
                     const Ctx& ctx, uint32_t& checked) {
    uint32_t failures = 0;
    const uint32_t m = cosets.num_cosets();
    const uint64_t N = cosets.N();
    const std::string cs = ctx.str();
    const auto fail = [&](const std::string& what) {
        std::cout << "HeckeBoundaryTauValidation: FAILED [" << cs << "] " << what << std::endl;
        ++failures;
    };

    // The table's own certificate, re-asserted (never relied on below).
    ++checked;
    if (table.certify() != 0) fail("certify() != 0 on a constructed table");
    ++checked;
    if (std::string(table.solve_regime()) != "Z" || !table.integral()) {
        fail("solve regime is not Z / integral is not true ([C, obl:taubdry](iv))");
    }
    ++checked;
    if (table.num_degrees() != model.boundary_dim() + 1) fail("num_degrees != boundary_dim + 1");

    // (a), (iv), (i): structure, coefficients, components, (T3bd).
    uint64_t recount = 0;
    for (uint32_t q = 0; q < table.num_degrees(); ++q) {
        for (uint32_t sigma : model.boundary_simplices(q)) {
            for (uint32_t j = 1; j <= m; ++j) {
                const auto& terms = table.tau(j, q, sigma);
                ++checked;
                bool first = true;
                Cusp comp{ 1, 0 };
                bool single = true;
                for (const auto& t : terms) {
                    if (!(t.coeff == 1 || t.coeff == -1)) {
                        fail("coefficient not in {+-1} ([C, obl:taubdry](iv)) at (j=" + std::to_string(j) +
                             ",q=" + std::to_string(q) + ",sigma=" + std::to_string(sigma) + ")");
                    }
                    if (t.gamma.det() != 1) fail("det(gamma_k) != 1");
                    if (!model.is_boundary_simplex(q, t.sigma_idx)) fail("sigma_k is not a boundary simplex");
                    const KuSimplexKey key = model.canonical_key(t.gamma, q, t.sigma_idx);
                    if (key.sigma_hat != t.sigma_idx || !(key.gamma_canon == t.gamma) || key.sign != 1) {
                        fail("term not in canonical form");
                    }
                    const uint64_t norm = static_cast<uint64_t>(t.gamma.max_abs());
                    if (norm > recount) recount = norm;
                    const Cusp c = own_cusp_of(t.gamma);
                    if (first) { comp = c; first = false; }
                    else if (c != comp) single = false;
                }
                if (!single) fail("chain supported in more than one component ([C, obl:taubdry](i))");
                if (!terms.empty()) {
                    const Cusp expected = own_g_inverse(N, own_cusp_of(cosets.s(j)));
                    if (comp != expected) {
                        fail("component map != c -> g^{-1} c ([C, lem:bdrycomponent]) at (j=" +
                             std::to_string(j) + ",q=" + std::to_string(q) + ",sigma=" +
                             std::to_string(sigma) + ")");
                    }
                    // The table's own component accessors agree with ours.
                    bool s2 = false;
                    const CuspKey ck = HeckeBoundaryTauTableT<Elt>::chain_component(terms, s2);
                    const CuspKey ex = HeckeBoundaryTauTableT<Elt>::component_map(cosets.g(), table.slot_component(j));
                    if (!s2 || ck.a != comp.a || ck.c != comp.c || ex.a != expected.a || ex.c != expected.c) {
                        fail("chain_component / component_map disagree with the independent cusp arithmetic");
                    }
                }
                if (q == 0 && (terms.size() != 1 || terms[0].coeff != 1)) {
                    fail("degree-0 chain is not one simplex with coefficient 1 ((T3bd))");
                }
            }
        }
    }
    ++checked;
    if (recount != table.certified_bound_B_tab()) fail("certified_bound_B_tab != recount");
    ++checked;
    if (table.certified_bound_B() < recount || table.certified_bound_B() < 4u * N * N ||
        table.certified_bound_B() > 2147483647ull) {
        fail("certified_bound_B inconsistent or outside the int64 tier");
    }

    // (ii) (T2bd): OUR boundary operator on the stored chain vs OUR
    // evaluation of tau_bd(boundary(s_j sigma)) = sum_f (-1)^f tau_bd(s_j . i(d_f sigma)),
    // each face through OUR (T1bd) translation at gamma = s_j (which must
    // coincide with the directly stored face slot) .
    for (uint32_t q = 1; q < table.num_degrees(); ++q) {
        for (uint32_t sigma : model.boundary_simplices(q)) {
            for (uint32_t j = 1; j <= m; ++j) {
                ++checked;
                ChainAcc lhs;
                for (const auto& t : table.tau(j, q, sigma)) {
                    for (uint32_t f = 0; f <= q; ++f) {
                        const uint32_t fidx = face_index(simplices, q, t.sigma_idx, f);
                        add_term(lhs, model, q - 1, t.gamma, fidx, ((f % 2 == 0) ? 1 : -1) * static_cast<int64_t>(t.coeff));
                    }
                }
                ChainAcc rhs, rhs_direct;
                for (uint32_t f = 0; f <= q; ++f) {
                    const uint32_t fidx = face_index(simplices, q, sigma, f);
                    const int64_t sgn = (f % 2 == 0) ? 1 : -1;
                    const ChainAcc tr = own_translate(model, cosets, table, cosets.s(j), q - 1, fidx, failures);
                    for (const auto& kv : tr) rhs[kv.first] += sgn * kv.second;
                    for (const auto& t : table.tau(j, q - 1, fidx)) {
                        add_term(rhs_direct, model, q - 1, t.gamma, t.sigma_idx, sgn * t.coeff);
                    }
                }
                if (!chains_equal(lhs, rhs) || !chains_equal(rhs, rhs_direct)) {
                    ChainAcc l = lhs, r = rhs;
                    prune_zero(l); prune_zero(r);
                    fail("(T2bd) chain-map identity at (j=" + std::to_string(j) + ",q=" + std::to_string(q) +
                         ",sigma=" + std::to_string(sigma) + "):\n    d(tau_bd(x)) = " + acc_to_string(l) +
                         "\n    tau_bd(d(x)) = " + acc_to_string(r));
                }
            }
        }
    }

    // (iii) (T1bd) bookkeeping. Route 1: the model's relation i(v3) = T i(v0)
    // through HeckeCosetTables::perm/cocycle (gen 0 = T): s_j . i(v3) =
    // h_j(T) . (s_{j.T} . i(v0)), so tau_bd(j, v3) == (g^{-1} h_j(T) g) . tau_bd(j.T, v0).
    {
        const uint32_t v3 = 3, v0 = 0;
        const int64_t Ni = static_cast<int64_t>(N);
        for (uint32_t j = 1; j <= m; ++j) {
            ++checked;
            const uint32_t j2 = cosets.perm(0, j);
            const Mat22 h = cosets.cocycle(0, j);
            if (h.c % Ni != 0) { fail("cocycle h_j(T) not in Gamma_0(N)"); continue; }
            const Mat22 conj{ h.a, hecke::checked_mul<int64_t>(h.b, Ni), h.c / Ni, h.d };
            ChainAcc predicted;
            for (const auto& t : table.tau(j2, 0, v0)) add_term(predicted, model, 0, conj * t.gamma, t.sigma_idx, t.coeff);
            const ChainAcc actual = terms_to_acc(model, 0, table.tau(j, 0, v3));
            if (!chains_equal(predicted, actual)) fail("(T1bd) bookkeeping across i(v3) = T i(v0) at j=" + std::to_string(j));
        }
    }
    // Route 2: 40 fixed-seed random gamma per table, every boundary slot:
    // OUR translation == the table's public helper, and the translated
    // chain lives in the component g^{-1} (gamma . infinity).
    {
        XorShift64 rng(0x42644261'75546175ULL ^ (N * 1315423911ULL) ^ ctx.ell);
        const Mat22 gens[4] = { Mat22::T(), Mat22::S(), Mat22::ST(), Mat22::TS() };
        for (int trial = 0; trial < 40; ++trial) {
            const int len = 1 + static_cast<int>(rng.below(6));
            Mat22 w = Mat22::I();
            for (int t = 0; t < len; ++t) w = w * gens[rng.below(4)];
            if (rng.below(2) == 1) w = w.neg();
            const Cusp expected = own_g_inverse(N, own_cusp_of(w));
            for (uint32_t q = 0; q < table.num_degrees(); ++q) {
                for (uint32_t sigma : model.boundary_simplices(q)) {
                    ++checked;
                    const ChainAcc ours = own_translate(model, cosets, table, w, q, sigma, failures);
                    const ChainAcc theirs = terms_to_acc(model, q, table.tau_on_translate(w, q, sigma));
                    if (!chains_equal(ours, theirs)) {
                        fail("tau_on_translate differs from the independent (T1bd) translation");
                    }
                    for (const auto& kv : ours) {
                        if (kv.second == 0) continue;
                        const Cusp c = own_normalise(kv.first[2], kv.first[4]);
                        if (c != expected) { fail("translated chain in the wrong component"); break; }
                    }
                }
            }
        }
    }

    // [C, prop:bdrycosets](1)-(3).
    {
        const auto& bc = table.boundary_cosets();
        ++checked;
        if (bc.size() != m) fail("|B_N / Gamma_inf| != m");
        const int64_t p = static_cast<int64_t>(ctx.p);
        // (1): canonical, primitive, upper triangular, det N, pairwise distinct.
        std::vector<int64_t> pw(ctx.nu + 1, 1);
        for (uint32_t u = 1; u <= ctx.nu; ++u) pw[u] = pw[u - 1] * p;
        for (size_t k = 0; k < bc.size(); ++k) {
            ++checked;
            const Mat22& d = bc[k].delta;
            bool ok = d.c == 0 && d.det() == static_cast<int64_t>(N) && bc[k].u <= ctx.nu &&
                      d.a == pw[bc[k].u] && d.d == pw[ctx.nu - bc[k].u] && d.b == bc[k].b &&
                      bc[k].b >= 0 && bc[k].b < d.a;
            if (ok) {
                const int64_t gabc = std::gcd(std::gcd(d.a, d.b), d.d);
                ok = (gabc == 1);
            }
            for (size_t k2 = 0; ok && k2 < k; ++k2) ok = !(bc[k2].delta == d);
            if (!ok) fail("boundary coset representative " + std::to_string(k) + " is not canonical/primitive/distinct");
        }
        if (ctx.nu == 1) {
            ++checked;
            bool ok = bc.size() == static_cast<size_t>(p) + 1 && bc[0].delta == Mat22{ 1, 0, 0, p };
            for (int64_t t = 0; ok && t < p; ++t) ok = bc[static_cast<size_t>(1 + t)].delta == Mat22{ p, t, 0, 1 };
            if (!ok) fail("nu = 1 representatives are not delta_inf = diag(1,p), delta_t = [[p,t],[0,1]]");
            // The three identities of (1), as exact matrix products.
            ++checked;
            const Mat22 T = Mat22::T();
            Mat22 Tp = Mat22::I();
            for (int64_t t = 0; t < p; ++t) Tp = Tp * T;
            const Mat22 d_inf = Mat22{ 1, 0, 0, p };
            bool ids = (T * d_inf == d_inf * Tp);
            for (int64_t t = 0; t + 1 < p; ++t) ids = ids && (T * Mat22{ p, t, 0, 1 } == Mat22{ p, t + 1, 0, 1 });
            ids = ids && (T * Mat22{ p, p - 1, 0, 1 } == Mat22{ p, 0, 0, 1 } * T);
            if (!ids) fail("the three identities of [C, prop:bdrycosets](1) fail");
            // (3): left multiplication by T fixes the u = 0 coset and permutes
            // the u = 1 cosets in one p-cycle (two double cosets).
            ++checked;
            bool dc = table.boundary_coset_index_of(T * bc[0].delta) == 0;
            std::vector<bool> seen(static_cast<size_t>(p), false);
            uint32_t idx = 1;
            for (int64_t t = 0; dc && t < p; ++t) {
                if (seen[idx - 1]) { dc = false; break; }
                seen[idx - 1] = true;
                const uint32_t nxt = table.boundary_coset_index_of(T * bc[idx].delta);
                if (nxt == 0 || nxt > static_cast<uint32_t>(p)) { dc = false; break; }
                idx = nxt;
            }
            if (dc) { if (idx != 1) dc = false; for (bool s : seen) dc = dc && s; }
            if (!dc) fail("[C, prop:bdrycosets](3): T does not act as one p-cycle on the u = 1 cosets");
        }
        // (2): the bijection j -> delta_j gamma_j Gamma_inf, with our own
        // canonical form (sign, then b mod a).
        std::vector<bool> hit(m, false);
        for (uint32_t j = 1; j <= m; ++j) {
            ++checked;
            const Mat22& gj = table.gamma_j(j);
            bool ok = gj.det() == 1 && own_cusp_of(gj) == own_g_inverse(N, own_cusp_of(cosets.s(j)));
            Mat22 dg = cosets.delta(j) * gj;
            ok = ok && dg.c == 0;
            if (ok) {
                if (dg.a < 0) dg = dg.neg();
                int64_t b = dg.b % dg.a;
                if (b < 0) b += dg.a;
                const uint32_t idx = table.boundary_coset_of_bulk(j);
                ok = idx < m && bc[idx].delta.a == dg.a && bc[idx].delta.d == dg.d && bc[idx].b == b && !hit[idx];
                if (ok) hit[idx] = true;
            }
            if (!ok) fail("[C, prop:bdrycosets](2) bijection fails at j=" + std::to_string(j));
        }
        ++checked;
        for (bool h : hit) if (!h) { fail("[C, prop:bdrycosets](2): not surjective"); break; }
    }
    return failures;
}

// =========================================================================
// Section 2: tampering (certify_with_override must reject).
// =========================================================================

template<class Elt>
uint32_t check_tampering(const SL2ZTessellationTables& model, const HeckeCosetTablesT<Elt>& cosets,
                         const HeckeBoundaryTauTableT<Elt>& table, const Ctx& ctx, uint32_t& checked) {
    uint32_t failures = 0;
    const std::string cs = ctx.str();
    const auto expect = [&](const char* what, uint32_t code, bool want_zero) {
        ++checked;
        if ((code == 0) != want_zero) {
            std::cout << "HeckeBoundaryTauValidation: FAILED [" << cs << "] tamper arm '" << what
                      << "': certify_with_override returned " << code << std::endl;
            ++failures;
        } else {
            std::cout << "HeckeBoundaryTauValidation: [" << cs << "] tamper '" << what << "' -> code "
                      << code << (want_zero ? " (expected 0)" : " (rejected, as required)") << std::endl;
        }
    };
    // A nonempty edge chain.
    uint32_t je = 0, ee = 0;
    for (uint32_t e : model.boundary_simplices(1)) {
        for (uint32_t j = 1; j <= cosets.num_cosets() && je == 0; ++j) {
            if (!table.tau(j, 1, e).empty()) { je = j; ee = e; }
        }
        if (je != 0) break;
    }
    if (je == 0) {
        std::cout << "HeckeBoundaryTauValidation: FAILED [" << cs << "] no nonempty edge chain to tamper with."
                  << std::endl;
        return 1;
    }
    const std::vector<ChainTerm> edge = table.tau(je, 1, ee);
    expect("identity", table.certify_with_override(je, 1, ee, edge), true);
    {
        std::vector<ChainTerm> t = edge;
        t[0].coeff = -t[0].coeff;
        expect("negated coefficient", table.certify_with_override(je, 1, ee, t), false);
    }
    {
        std::vector<ChainTerm> t = edge;
        t.erase(t.begin());
        expect("dropped term", table.certify_with_override(je, 1, ee, t), false);
    }
    {
        std::vector<ChainTerm> t = edge;
        t.push_back(edge[0]);
        expect("duplicated term", table.certify_with_override(je, 1, ee, t), false);
    }
    // Vertex slots (j = 1, v0).
    const uint32_t v0 = model.boundary_simplices(0)[0];
    const std::vector<ChainTerm> vert = table.tau(1, 0, v0);
    expect("identity (vertex)", table.certify_with_override(1, 0, v0, vert), true);
    {
        std::vector<ChainTerm> t = vert; // moved along its own line: T-shift
        const KuSimplexKey key = model.canonical_key(Mat22::T() * t[0].gamma, 0, t[0].sigma_idx);
        t[0].gamma = key.gamma_canon;
        t[0].sigma_idx = key.sigma_hat;
        expect("vertex shifted by T", table.certify_with_override(1, 0, v0, t), false);
    }
    {
        std::vector<ChainTerm> t = vert; // moved to another component: S-translate
        const KuSimplexKey key = model.canonical_key(Mat22::S() * t[0].gamma, 0, t[0].sigma_idx);
        t[0].gamma = key.gamma_canon;
        t[0].sigma_idx = key.sigma_hat;
        expect("vertex in another component", table.certify_with_override(1, 0, v0, t), false);
    }
    {
        std::vector<ChainTerm> t = vert;
        t[0].coeff = 2;
        expect("coefficient 2", table.certify_with_override(1, 0, v0, t), false);
    }
    {
        std::vector<ChainTerm> t = vert;
        ChainTerm u = vert[0];
        const KuSimplexKey key = model.canonical_key(Mat22::T() * u.gamma, 0, u.sigma_idx);
        u.gamma = key.gamma_canon;
        u.sigma_idx = key.sigma_hat;
        t.push_back(u);
        expect("two-term degree-0 chain", table.certify_with_override(1, 0, v0, t), false);
    }
    {
        expect("empty degree-0 chain", table.certify_with_override(1, 0, v0, {}), false);
    }
    return failures;
}

// =========================================================================
// Section 3: the boundary subcomplex and the cusp arithmetic.
// =========================================================================

uint32_t check_model(const SimplexTable& simplices, const SL2ZTessellationTables& model, uint32_t& checked) {
    uint32_t failures = 0;
    const auto fail = [&](const std::string& what) {
        std::cout << "HeckeBoundaryTauValidation: FAILED [model] " << what << std::endl;
        ++failures;
    };
    ++checked;
    if (model.validate_boundary() != 0) fail("validate_boundary() != 0");
    ++checked;
    if (model.boundary_dim() != 1 || model.boundary_period() != 3) fail("boundary_dim / period");
    ++checked;
    if (model.boundary_simplices(0) != std::vector<uint32_t>{ 0, 1, 2, 3 }) fail("boundary vertices != {0,1,2,3}");
    ++checked;
    {
        std::vector<uint32_t> es = { find_simplex_index(simplices, 1, { 0, 1 }),
                                     find_simplex_index(simplices, 1, { 1, 2 }),
                                     find_simplex_index(simplices, 1, { 2, 3 }) };
        std::sort(es.begin(), es.end());
        if (model.boundary_simplices(1) != es) fail("boundary edges != {e01, e12, e23}");
        for (uint32_t f = 0; f < simplices[2].size(); ++f) {
            if (model.is_boundary_simplex(2, f)) fail("a 2-simplex is marked boundary");
        }
        if (!model.boundary_simplices(2).empty()) fail("boundary_simplices(2) nonempty");
    }
    ++checked;
    if (!(model.boundary_translation() == Mat22::T())) fail("boundary translation != T");
    // Line coordinates: positions 0..3 are v0, v1, v2, T v0.
    ++checked;
    {
        bool ok = true;
        for (int64_t P = 0; P < 3; ++P) {
            const auto vt = model.boundary_line_vertex(P);
            ok = ok && (vt.first == Mat22::I()) && vt.second == static_cast<uint32_t>(P);
        }
        const auto v3 = model.boundary_line_vertex(3);
        ok = ok && (v3.first == Mat22::T()) && v3.second == 0;
        ok = ok && model.boundary_line_position(Mat22::I(), 3) == 3;
        ok = ok && model.boundary_line_position(Mat22::T().neg(), 1) == 4;
        const auto le = model.boundary_line_edge(2); // e23, +1
        ok = ok && le.edge_idx == find_simplex_index(simplices, 1, { 2, 3 }) && le.coeff == 1 && (le.gamma == Mat22::I());
        const auto lem = model.boundary_line_edge(-1); // T^{-1} e23
        ok = ok && lem.edge_idx == le.edge_idx && (lem.gamma == Mat22{ 1, -1, 0, 1 });
        if (!ok) fail("line coordinates");
    }
    // Nearest-vertex rule at hand-computed points (real parts -1/2, -1/6, 1/6, 1/2 + k):
    //   x = 0     : midpoint of v1, v2 -> tie -> larger -> v2 (P = 2)
    //   x = -1    : midpoint of T^{-1} v1 (-7/6) and T^{-1} v2 (-5/6) -> P = -1
    //   x = 1/2   : v3 exactly -> P = 3
    //   x = 1/3   : tie between 1/6 and 1/2 -> P = 3
    //   x = -0.4  : nearest -1/2 (dist 0.1) vs -1/6 (0.233) -> P = 0
    //   x = 1.3   : 7/6 = T v2 (dist 0.133) vs 3/2 = T v3 (0.2) -> P = 3*1 + 2 = 5
    ++checked;
    {
        bool ok = model.boundary_line_nearest(0, 1) == 2 && model.boundary_line_nearest(-1, 1) == -1 &&
                  model.boundary_line_nearest(1, 2) == 3 && model.boundary_line_nearest(1, 3) == 3 &&
                  model.boundary_line_nearest(-2, 5) == 0 && model.boundary_line_nearest(13, 10) == 5;
        if (!ok) fail("nearest-vertex rule at hand-computed points");
    }
    // Cusp arithmetic: reduce_to_infinity, apply(adj g), normalise.
    ++checked;
    {
        bool ok = true;
        const int64_t samples[][2] = { { 1, 0 }, { 0, 1 }, { 1, 1 }, { -1, 1 }, { 2, 3 }, { 3, -2 }, { 5, 7 },
                                       { -7, 5 }, { 11, 4 }, { 4, 11 }, { 6, 4 }, { 100, 37 }, { -37, 100 } };
        for (const auto& s : samples) {
            const CuspKey c = CuspKey::normalise(s[0], s[1]);
            const Cusp o = own_normalise(s[0], s[1]);
            ok = ok && c.a == o.a && c.c == o.c;
            const Mat22 gred = CuspKey::reduce_to_infinity(c);
            ok = ok && gred.det() == 1 && CuspKey::apply(gred, c) == CuspKey::infinity();
            // our own check of gred . c = infinity: bottom row kills the column
            ok = ok && (gred.c * c.a + gred.d * c.c == 0);
            for (uint64_t N : { 2ull, 4ull, 9ull, 25ull }) {
                const Mat22 g{ 1, 0, 0, static_cast<int64_t>(N) };
                const CuspKey img = HeckeBoundaryTauTableT<uint32_t>::component_map(g, c);
                const Cusp o2 = own_g_inverse(N, o);
                ok = ok && img.a == o2.a && img.c == o2.c;
            }
        }
        if (!ok) fail("CuspKey arithmetic vs the independent cusp arithmetic");
    }
    return failures;
}

// =========================================================================
// Grid driver.
// =========================================================================

template<class Elt>
uint32_t run_width(const SimplicialComplex& K, const SimplexTable& simplices, const SL2ZTessellationTables& model,
                   const std::vector<std::pair<uint64_t, uint32_t>>& prs, const std::vector<uint64_t>& ells,
                   const std::vector<uint32_t>& ns, const char* width, std::map<uint64_t, Signature>& sigs,
                   uint32_t& tables, uint32_t& checked, bool tamper, bool print_p2) {
    uint32_t failures = 0;
    for (const auto& pr : prs) {
        for (uint64_t ell : ells) {
            for (uint32_t n : ns) {
                const Ctx ctx{ pr.first, pr.second, ell, n, width };
                try {
                    const HeckeCosetTablesT<Elt> cosets(pr.first, pr.second, static_cast<Elt>(ell), n);
                    const HeckeBoundaryTauTableT<Elt> table(K, model, cosets);
                    ++tables;
                    failures += check_table(simplices, model, cosets, table, ctx, checked);
                    // (B) ell / n / width independence, keyed by N (p^nu).
                    const Signature sig = signature_of(model, table);
                    const auto it = sigs.find(cosets.N());
                    ++checked;
                    if (it == sigs.end()) {
                        sigs.emplace(cosets.N(), sig);
                    } else if (it->second != sig) {
                        std::cout << "HeckeBoundaryTauValidation: FAILED [" << ctx.str()
                                  << "] table differs from the first table at the same N (ell/n/width dependence)."
                                  << std::endl;
                        ++failures;
                    }
                    if (tamper && ell == ells.front() && n == ns.front() && pr.first <= 5 && pr.second == 1) {
                        failures += check_tampering(model, cosets, table, ctx, checked);
                    }
                    if (print_p2 && pr.first == 2 && pr.second == 1 && ell == ells.front() && n == ns.front()) {
                        std::cout << table.to_string();
                    }
                } catch (const std::exception& e) {
                    std::cout << "HeckeBoundaryTauValidation: EXCEPTION [" << ctx.str() << "]: " << e.what()
                              << std::endl;
                    ++failures;
                }
            }
        }
    }
    return failures;
}

} // namespace

uint32_t HeckeBoundaryTauValidation() {
    const auto t0 = std::chrono::steady_clock::now();
    uint32_t failures = 0;
    uint32_t checked = 0;
    uint32_t tables = 0;
    try {
        const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
        const SL2ZTessellationTables model(K);
        const SimplexTable& simplices = K.get_simplices();

        failures += check_model(simplices, model, checked);

        std::map<uint64_t, Signature> sigs; // N -> chains (shared across widths)
        const std::vector<std::pair<uint64_t, uint32_t>> prs32 = {
            { 2, 1 }, { 3, 1 }, { 5, 1 }, { 7, 1 }, { 11, 1 }, { 2, 2 }, { 3, 2 }
        };
        const std::vector<uint64_t> ells32 = { 2, 3, 5, 7, 691, 134217757ull };
        const std::vector<uint32_t> ns = { 0, 2, 10 };
        failures += run_width<uint32_t>(K, simplices, model, prs32, ells32, ns, "u32", sigs, tables, checked,
                                        /*tamper=*/true, /*print_p2=*/true);

        const std::vector<std::pair<uint64_t, uint32_t>> prs64 = { { 2, 1 }, { 3, 1 }, { 2, 2 } };
        const std::vector<uint64_t> ells64 = { 2305843009213693951ull /* 2^61-1 */, 4294967311ull /* 2^32+15 */ };
        const std::vector<uint32_t> ns64 = { 0, 2 };
        failures += run_width<uint64_t>(K, simplices, model, prs64, ells64, ns64, "u64", sigs, tables, checked,
                                        /*tamper=*/false, /*print_p2=*/false);
    } catch (const std::exception& e) {
        std::cout << "HeckeBoundaryTauValidation: EXCEPTION during setup: " << e.what() << std::endl;
        return 1000;
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "BOUNDARY_TAU tables=" << tables << " checks=" << checked << " failures=" << failures
              << " (" << secs << " s)" << std::endl;
    return failures;
}
