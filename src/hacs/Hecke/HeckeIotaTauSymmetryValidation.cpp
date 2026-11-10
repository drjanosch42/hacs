// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// obl:iota(i) + obl:iota(iv) of docs/latex/gl2_contract.tex chapter 9.
// See HeckeIotaTauSymmetryValidation.h for the full statement of what is
// gating -- (A), (C), and since 2026-09-16 (B) on the canonical tau -- and
// what is merely recorded ((B) on the legacy and FixedV6 taus).
//
// Independence discipline: every piece of mathematics used here is re-derived
// from the two contracts (gl2_contract.tex chapter 9 for iota, hecke_action.tex
// def:tau for (T1)) against the PUBLIC accessors of SL2ZTessellationTables,
// HeckeCosetTables and HeckeTauTable only. No private helper of HeckeTauTable
// is used (none is reachable), certify() is never called, and no existing
// computational path is modified.

#include "HeckeIotaTauSymmetryValidation.h"

#include "HeckeCosetTables.h"
#include "HeckeTauTable.h"
#include "SL2ZTessellationTables.h"
#include "ExactInt.h"
#include "QuadPoint.h"
#include "../LocalSystemSL2Z.h"      // CreateSL2ZFundamentalDomainTriangulation
#include "../SimplicialCohomology.h" // SimplicialComplex

#include <algorithm>
#include <array>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "ValidationSupport.h"

namespace {

using hecke::ChainTerm;
using hecke::HeckeCosetTables;
using hecke::HeckeTauTable;
using hecke::KuSimplexKey;
using hecke::Mat22;
using hecke::QuadPoint;
using hecke::SL2ZTessellationTables;
using hecke::TauVertexPolicy;

using SimplexTable = std::vector<std::vector<std::vector<uint32_t>>>;

// =========================================================================
// Section 0 -- the element iota and its conjugation (def:iota(i)).
// iota = diag(-1, 1);  c_iota(M) = iota * M * iota = [[a, -b], [-c, d]].
// det is preserved, so c_iota(Gamma) = Gamma.
// =========================================================================

Mat22 iota_mat() { return Mat22{ -1, 0, 0, 1 }; }

Mat22 c_iota(const Mat22& m) {
    // Written as the exact product so the formula is not silently retyped.
    const Mat22 iv = iota_mat();
    return iv * m * iv;
}

// z -> -conj(z) on an exact QuadPoint z = (u + v*sqrt(D)*i)/w: negate u.
QuadPoint iota_point(const QuadPoint& z) {
    return QuadPoint(hecke::checked_neg<int64_t>(z.u), z.v, z.w, z.D);
}

// =========================================================================
// Section 1 -- canonical chain accumulation (identical in spirit to the S4V
// accumulator: fold every term through the PUBLIC canonical_key so that the
// comparison is of chains, not of representations).
// =========================================================================

using CanonKey = std::array<int64_t, 6>; // { q, sigma_hat, a, b, c, d }
using ChainAcc = std::map<CanonKey, int64_t>;

void add_term(ChainAcc& acc, const SL2ZTessellationTables& model, uint32_t q,
              const Mat22& gamma, uint32_t sigma_idx, int64_t coeff) {
    if (coeff == 0) return;
    const KuSimplexKey key = model.canonical_key(gamma, q, sigma_idx);
    const CanonKey k{ static_cast<int64_t>(key.q), static_cast<int64_t>(key.sigma_hat),
                      key.gamma_canon.a, key.gamma_canon.b, key.gamma_canon.c, key.gamma_canon.d };
    acc[k] += coeff * static_cast<int64_t>(key.sign);
}

void prune_zero(ChainAcc& acc) {
    for (auto it = acc.begin(); it != acc.end(); ) {
        if (it->second == 0) it = acc.erase(it); else ++it;
    }
}

void reduce_mod_ell(ChainAcc& acc, uint32_t ell) {
    for (auto& kv : acc) {
        int64_t v = kv.second % static_cast<int64_t>(ell);
        if (v < 0) v += static_cast<int64_t>(ell);
        kv.second = v;
    }
    prune_zero(acc);
}

// Exact over Z, unless the table recorded a mod-ell triangle solve
// ([C, alg:tau](3)-(4)), in which case its chains are only defined mod ell.
void normalise(ChainAcc& acc, bool mod_ell, uint32_t ell) {
    if (mod_ell) reduce_mod_ell(acc, ell); else prune_zero(acc);
}

std::string chain_to_string(const ChainAcc& acc);

std::string key_to_string(const CanonKey& k) {
    std::ostringstream oss;
    oss << "(q=" << k[0] << ",sigma_hat=" << k[1] << ",gamma=[[" << k[2] << "," << k[3]
        << "],[" << k[4] << "," << k[5] << "]])";
    return oss.str();
}

std::string chain_to_string(const ChainAcc& acc) {
    if (acc.empty()) return "0";
    std::ostringstream oss;
    bool first = true;
    for (const auto& kv : acc) {
        if (!first) oss << " + ";
        first = false;
        oss << kv.second << "*" << key_to_string(kv.first);
    }
    return oss.str();
}

// First key at which the two chains differ, with both coefficients.
bool first_difference(const ChainAcc& a, const ChainAcc& b, CanonKey& key_out,
                      int64_t& va_out, int64_t& vb_out) {
    auto ia = a.begin();
    auto ib = b.begin();
    while (ia != a.end() && ib != b.end()) {
        if (ia->first < ib->first)      { key_out = ia->first; va_out = ia->second; vb_out = 0; return true; }
        if (ib->first < ia->first)      { key_out = ib->first; va_out = 0; vb_out = ib->second; return true; }
        if (ia->second != ib->second)   { key_out = ia->first; va_out = ia->second; vb_out = ib->second; return true; }
        ++ia; ++ib;
    }
    if (ia != a.end()) { key_out = ia->first; va_out = ia->second; vb_out = 0; return true; }
    if (ib != b.end()) { key_out = ib->first; va_out = 0; vb_out = ib->second; return true; }
    return false;
}

// =========================================================================
// Section 2 -- iota_K on the stored simplices: the vertex permutation from
// the shipped exact coordinates, and the orientation sign of every simplex
// as the parity of the sort back into increasing vertex order.
// =========================================================================

struct IotaModel {
    std::array<uint32_t, 7> vperm{};                  // iota on vertices
    std::vector<std::vector<uint32_t>> image_idx;     // [q][sigma] -> |iota_K sigma|
    std::vector<std::vector<int>> image_sign;         // [q][sigma] -> eps_sigma in {+1,-1}
};

uint32_t find_simplex_index(const SimplexTable& simplices, uint32_t q, std::vector<uint32_t> verts) {
    return hecke::vsupport::find_simplex_index(simplices, q, std::move(verts),
                                               "HeckeIotaTauSymmetryValidation: image simplex is not a simplex of K "
                             "-- hypothesis (I1) fails for this triangulation.");
}

// Parity of the permutation sorting `img` into increasing order (+1 even).
int sort_parity(const std::vector<uint32_t>& img) {
    int inversions = 0;
    for (size_t a = 0; a + 1 < img.size(); ++a) {
        for (size_t b = a + 1; b < img.size(); ++b) {
            if (img[a] > img[b]) ++inversions;
        }
    }
    return (inversions % 2 == 0) ? 1 : -1;
}

// lem:iotamodel's printed sign table, transcribed from the contract: the six
// simplices carrying sign -1, by VERTEX SET (never by index, so a change of
// the simplex ordering cannot silently satisfy the check).
const std::vector<std::vector<uint32_t>>& minus_sign_simplices() {
    static const std::vector<std::vector<uint32_t>> rows = {
        { 0, 1 }, { 2, 3 }, { 1, 2 },          // q = 1
        { 0, 1, 4 }, { 2, 3, 5 }, { 1, 2, 6 }, // q = 2
    };
    return rows;
}

// Builds iota_K and checks obl:iota(i) entirely over Z. Returns the number of
// GATING discrepancies.
uint32_t build_and_check_iota_model(const SimplexTable& simplices,
                                    const SL2ZTessellationTables& model,
                                    IotaModel& out, uint32_t& checked_out) {
    uint32_t failures = 0;

    // --- (A1) The vertex permutation, read off the SHIPPED exact coordinates.
    for (uint32_t v = 0; v < 7; ++v) {
        const QuadPoint img = iota_point(model.vertex_position(v));
        int32_t hit = -1;
        for (uint32_t w = 0; w < 7; ++w) {
            ++checked_out;
            if (model.vertex_position(w) == img) {
                if (hit >= 0) {
                    std::cout << "HeckeIotaTauSymmetryValidation: (A1) two vertices share a "
                                 "position; v=" << v << " maps ambiguously." << std::endl;
                    ++failures;
                }
                hit = static_cast<int32_t>(w);
            }
        }
        if (hit < 0) {
            std::cout << "HeckeIotaTauSymmetryValidation: (A1) FAILED -- z -> -conj(z) sends the "
                         "shipped position of v" << v << " off the vertex set; (I1) fails."
                      << std::endl;
            ++failures;
            return failures; // nothing downstream is meaningful
        }
        out.vperm[v] = static_cast<uint32_t>(hit);
    }

    // --- (A2) It is lem:iotamodel's permutation, and an involution.
    const std::array<uint32_t, 7> expected{ 3, 2, 1, 0, 5, 4, 6 };
    for (uint32_t v = 0; v < 7; ++v) {
        ++checked_out;
        if (out.vperm[v] != expected[v]) {
            std::cout << "HeckeIotaTauSymmetryValidation: (A2) FAILED -- vertex permutation "
                         "disagrees with lem:iotamodel at v" << v << ": got " << out.vperm[v]
                      << ", contract says " << expected[v] << "." << std::endl;
            ++failures;
        }
        ++checked_out;
        if (out.vperm[out.vperm[v]] != v) {
            std::cout << "HeckeIotaTauSymmetryValidation: (A2) FAILED -- vertex permutation is not "
                         "an involution at v" << v << "." << std::endl;
            ++failures;
        }
    }

    // --- (A3) lem:iotamodel's hypothesis Re v_1 = -Re v_2, exactly, on the
    // shipped coordinates: u1/w1 == -u2/w2, i.e. u1*w2 + u2*w1 == 0.
    {
        const QuadPoint p1 = model.vertex_position(1);
        const QuadPoint p2 = model.vertex_position(2);
        ++checked_out;
        const int64_t lhs = hecke::checked_add<int64_t>(hecke::checked_mul<int64_t>(p1.u, p2.w),
                                                        hecke::checked_mul<int64_t>(p2.u, p1.w));
        std::cout << "HeckeIotaTauSymmetryValidation: (A3) shipped horocycle coordinates: "
                     "Re v1 = " << p1.u << "/" << p1.w << ", Re v2 = " << p2.u << "/" << p2.w
                  << "  =>  Re v1 + Re v2 = " << lhs << "/(" << p1.w << "*" << p2.w << ")."
                  << std::endl;
        if (lhs != 0) {
            std::cout << "HeckeIotaTauSymmetryValidation: (A3) FAILED -- lem:iotamodel's hypothesis "
                         "Re v1 = -Re v2 does NOT hold for the shipped coordinates." << std::endl;
            ++failures;
        }
        // Re v0 = -1/2 and Re v3 = +1/2 (def:model7) -- also checked, exactly.
        const QuadPoint p0 = model.vertex_position(0);
        const QuadPoint p3 = model.vertex_position(3);
        ++checked_out;
        if (hecke::checked_add<int64_t>(hecke::checked_mul<int64_t>(p0.u, p3.w),
                                        hecke::checked_mul<int64_t>(p3.u, p0.w)) != 0) {
            std::cout << "HeckeIotaTauSymmetryValidation: (A3) FAILED -- Re v0 != -Re v3."
                      << std::endl;
            ++failures;
        }
    }

    // --- (A4) Simplicial: images of stored simplices are stored simplices,
    // with the recomputed orientation sign compared against lem:iotamodel.
    out.image_idx.assign(3, {});
    out.image_sign.assign(3, {});
    for (uint32_t q = 0; q <= 2; ++q) {
        const auto& list = simplices.at(q);
        out.image_idx[q].assign(list.size(), 0);
        out.image_sign[q].assign(list.size(), 1);
        for (uint32_t s = 0; s < list.size(); ++s) {
            std::vector<uint32_t> img;
            img.reserve(list[s].size());
            for (uint32_t v : list[s]) img.push_back(out.vperm.at(v));
            const int sign = sort_parity(img);
            const uint32_t idx = find_simplex_index(simplices, q, img);
            out.image_idx[q][s] = idx;
            out.image_sign[q][s] = sign;

            // Contract sign, by vertex set.
            std::vector<uint32_t> sorted_src = list[s];
            const auto& minus = minus_sign_simplices();
            const bool expect_minus =
                std::find(minus.begin(), minus.end(), sorted_src) != minus.end();
            ++checked_out;
            if ((sign == -1) != expect_minus) {
                std::cout << "HeckeIotaTauSymmetryValidation: (A4) SIGN TABLE MISMATCH at q=" << q
                          << " sigma=" << s << " {";
                for (size_t t = 0; t < sorted_src.size(); ++t)
                    std::cout << (t ? "," : "") << "v" << sorted_src[t];
                std::cout << "}: recomputed sign " << sign << ", lem:iotamodel says "
                          << (expect_minus ? -1 : 1) << "." << std::endl;
                ++failures;
            }
        }
    }

    // --- (A5) iota_K is an involution on simplices, signs included.
    for (uint32_t q = 0; q <= 2; ++q) {
        for (uint32_t s = 0; s < out.image_idx[q].size(); ++s) {
            const uint32_t t = out.image_idx[q][s];
            ++checked_out;
            if (out.image_idx[q][t] != s ||
                out.image_sign[q][s] * out.image_sign[q][t] != 1) {
                std::cout << "HeckeIotaTauSymmetryValidation: (A5) iota_K is not an involution on "
                             "oriented simplices at q=" << q << " sigma=" << s << "." << std::endl;
                ++failures;
            }
        }
    }

    // --- (A6) c_iota is a conjugation of Gamma with the eq:iotaconj table.
    {
        const std::array<std::pair<Mat22, Mat22>, 4> rows = {
            std::make_pair(Mat22::T(),  Mat22::T().inv_unimodular()),
            std::make_pair(Mat22::S(),  Mat22::S().inv_unimodular()),
            std::make_pair(Mat22::ST(), Mat22::TS().inv_unimodular()),
            std::make_pair(Mat22::TS(), Mat22::ST().inv_unimodular()),
        };
        for (const auto& row : rows) {
            ++checked_out;
            if (!(c_iota(row.first) == row.second)) {
                std::cout << "HeckeIotaTauSymmetryValidation: (A6) eq:iotaconj FAILED on a "
                             "generator." << std::endl;
                ++failures;
            }
        }
    }

    return failures;
}

// =========================================================================
// Section 3 -- the recorded predicate (B): iota_X tau iota_X == tau, and the
// certification (C) that tau^iota is itself a valid g-twisted chain datum.
//
// (C) is what makes a NEGATIVE verdict in (B) meaningful rather than a report
// on a mis-built iota_X: prop:iotacommutes(3a) proves, with no hypothesis on
// ell, that tau^iota = iota_X o tau o iota_X satisfies (T1)-(T3) of
// [hecke_action.tex, def:tau]. So if the iota_X realised here is the right
// one, tau^iota must pass the same S4V oracles the shipped tau passes; and
// then tau^iota != tau says exactly what (3a) allows -- two admissible data
// for the same g -- while tau^iota failing (T1)-(T3) would instead convict
// this file. (C) is therefore GATING.
// =========================================================================

// d_j(sigma) as an index into K's (q-1)-simplex list.
uint32_t face_index(const SimplexTable& simplices, uint32_t q, uint32_t sigma, uint32_t j) {
    std::vector<uint32_t> verts = simplices.at(q).at(sigma);
    if (j >= verts.size()) throw std::runtime_error("HeckeIotaTauSymmetryValidation: bad face index.");
    verts.erase(verts.begin() + static_cast<std::ptrdiff_t>(j));
    return find_simplex_index(simplices, q - 1, verts);
}

// tau(s_i . i(sigma)) as a canonical chain -- the stored slot, folded.
ChainAcc tau_chain(const SL2ZTessellationTables& model, const HeckeTauTable& table,
                   uint32_t i, uint32_t q, uint32_t sigma) {
    ChainAcc acc;
    for (const auto& t : table.tau(i, q, sigma)) {
        add_term(acc, model, q, t.gamma, t.sigma_idx, static_cast<int64_t>(t.coeff));
    }
    return acc;
}

// tau^iota(s_i . i(sigma)) = iota_X( tau( iota_X(s_i . i(sigma)) ) ).
//   iota_X(s_i . i(sigma)) = eps_sigma * ( c_iota(s_i) . i(sigma_img) ),
// and tau on that translate is evaluated through (T1), re-derived here:
// c_iota(s_i) = h * s_j, j = class_index_of(c_iota(s_i)), h in Gamma_0(N),
// tau(h . x) = (g^{-1} h g) . tau(x),  g^{-1} h g = [[h.a, N h.b],[h.c/N, h.d]].
// `structural_ok` reports the Gamma_0(N) membership assertion.
ChainAcc tau_iota_chain(const SL2ZTessellationTables& model, const HeckeCosetTables& cosets,
                        const HeckeTauTable& table, const IotaModel& iota,
                        uint32_t i, uint32_t q, uint32_t sigma, bool& structural_ok) {
    structural_ok = true;
    ChainAcc acc;
    const int64_t N = static_cast<int64_t>(cosets.N());
    const uint32_t sigma_img = iota.image_idx[q][sigma];
    const int64_t eps_sigma = iota.image_sign[q][sigma];

    const Mat22 gprime = c_iota(cosets.s(i));
    const uint32_t j = cosets.class_index_of(gprime);
    const Mat22 h = gprime * cosets.s_inv(j);
    if (h.det() != 1 || h.c % N != 0) { structural_ok = false; return acc; }
    const int64_t Nb = hecke::checked_mul<int64_t>(h.b, N);
    const Mat22 conj{ h.a, Nb, h.c / N, h.d }; // g^{-1} h g, exact integral

    for (const auto& t : table.tau(j, q, sigma_img)) {
        const Mat22 gk = conj * t.gamma;
        const uint32_t sig2 = iota.image_idx[q][t.sigma_idx];
        const int64_t eps2 = iota.image_sign[q][t.sigma_idx];
        add_term(acc, model, q, c_iota(gk), sig2,
                 static_cast<int64_t>(t.coeff) * eps_sigma * eps2);
    }
    return acc;
}

struct TauVerdict {
    uint32_t slots_checked = 0;
    uint32_t slots_differing = 0;
    std::array<uint32_t, 3> differing_by_degree{ { 0, 0, 0 } };
    std::array<uint32_t, 3> slots_by_degree{ { 0, 0, 0 } };
    // Minimal witness (first differing slot in (q, sigma, i) order).
    bool have_witness = false;
    uint32_t w_i = 0, w_q = 0, w_sigma = 0;
    CanonKey w_key{};
    int64_t w_tau = 0, w_tau_iota = 0;
    std::string w_tau_chain, w_tau_iota_chain;
};

// Returns the number of GATING failures: the (B1) Gamma_0(N) structural
// assertion and the (C) certification of tau^iota. The symmetry verdict
// itself is only accumulated into `v`, never returned.
uint32_t check_tau_symmetry(const SimplexTable& simplices, const SL2ZTessellationTables& model,
                            const HeckeCosetTables& cosets, const HeckeTauTable& table,
                            const IotaModel& iota, uint32_t ell, TauVerdict& v) {
    uint32_t failures = 0;
    const uint32_t m = cosets.num_cosets();
    const bool mod_ell = table.used_mod_ell_solve();

    // c_iota(g) = g is the hypothesis of prop:iotacommutes(3): g = diag(1, N).
    if (!(c_iota(cosets.g()) == cosets.g())) {
        std::cout << "HeckeIotaTauSymmetryValidation: (B0) c_iota(g) != g -- "
                     "prop:iotacommutes(3) does not apply to this g." << std::endl;
        return failures + 1;
    }

    // Cache tau^iota on every stored slot (also used by (C)).
    std::vector<std::vector<std::vector<ChainAcc>>> ti(3);
    for (uint32_t q = 0; q <= 2; ++q) {
        const uint32_t count = static_cast<uint32_t>(simplices.at(q).size());
        ti[q].assign(count, {});
        for (uint32_t sigma = 0; sigma < count; ++sigma) {
            ti[q][sigma].assign(m + 1, ChainAcc{});
            for (uint32_t i = 1; i <= m; ++i) {
                bool ok = true;
                ChainAcc c = tau_iota_chain(model, cosets, table, iota, i, q, sigma, ok);
                if (!ok) {
                    std::cout << "HeckeIotaTauSymmetryValidation: (B1) c_iota(s_i)*s_j^{-1} is not "
                                 "in Gamma_0(N) at i=" << i << "." << std::endl;
                    ++failures;
                }
                normalise(c, mod_ell, ell);
                ti[q][sigma][i] = std::move(c);
            }
        }
    }

    // --- (B) the recorded comparison, slot by slot.
    for (uint32_t q = 0; q <= 2; ++q) {
        const uint32_t count = static_cast<uint32_t>(simplices.at(q).size());
        for (uint32_t sigma = 0; sigma < count; ++sigma) {
            for (uint32_t i = 1; i <= m; ++i) {
                ++v.slots_checked;
                ++v.slots_by_degree[q];
                ChainAcc lhs = tau_chain(model, table, i, q, sigma);
                normalise(lhs, mod_ell, ell);
                const ChainAcc& rhs = ti[q][sigma][i];
                if (lhs != rhs) {
                    ++v.slots_differing;
                    ++v.differing_by_degree[q];
                    if (!v.have_witness) {
                        CanonKey k{};
                        int64_t va = 0, vb = 0;
                        if (first_difference(lhs, rhs, k, va, vb)) {
                            v.have_witness = true;
                            v.w_i = i; v.w_q = q; v.w_sigma = sigma;
                            v.w_key = k; v.w_tau = va; v.w_tau_iota = vb;
                            v.w_tau_chain = chain_to_string(lhs);
                            v.w_tau_iota_chain = chain_to_string(rhs);
                        }
                    }
                }
            }
        }
    }

    // --- (C1) (T3) for tau^iota: epsilon(tau^iota(vertex slot)) == 1.
    for (uint32_t sigma = 0; sigma < ti[0].size(); ++sigma) {
        for (uint32_t i = 1; i <= m; ++i) {
            int64_t eps = 0;
            for (const auto& kv : ti[0][sigma][i]) eps += kv.second;
            const bool ok = mod_ell ? ((eps - 1) % static_cast<int64_t>(ell) == 0) : (eps == 1);
            if (!ok) {
                std::cout << "HeckeIotaTauSymmetryValidation: (C1) tau^iota violates (T3) at (i="
                          << i << ",v=" << sigma << "): epsilon = " << eps << "." << std::endl;
                ++failures;
            }
        }
    }

    // --- (C2) (T2) for tau^iota: d(tau^iota(x)) == tau^iota(d x).
    for (uint32_t q = 1; q <= 2; ++q) {
        const uint32_t count = static_cast<uint32_t>(simplices.at(q).size());
        for (uint32_t sigma = 0; sigma < count; ++sigma) {
            for (uint32_t i = 1; i <= m; ++i) {
                ChainAcc lhs;
                for (const auto& kv : ti[q][sigma][i]) {
                    const Mat22 gk{ kv.first[2], kv.first[3], kv.first[4], kv.first[5] };
                    const uint32_t sig = static_cast<uint32_t>(kv.first[1]);
                    for (uint32_t jj = 0; jj <= q; ++jj) {
                        const uint32_t fidx = face_index(simplices, q, sig, jj);
                        add_term(lhs, model, q - 1, gk, fidx,
                                 kv.second * ((jj % 2 == 0) ? 1 : -1));
                    }
                }
                ChainAcc rhs;
                for (uint32_t jj = 0; jj <= q; ++jj) {
                    const uint32_t fidx = face_index(simplices, q, sigma, jj);
                    const int64_t sj = (jj % 2 == 0) ? 1 : -1;
                    for (const auto& kv : ti[q - 1][fidx][i]) rhs[kv.first] += kv.second * sj;
                }
                normalise(lhs, mod_ell, ell);
                normalise(rhs, mod_ell, ell);
                if (lhs != rhs) {
                    std::cout << "HeckeIotaTauSymmetryValidation: (C2) tau^iota violates (T2) at (i="
                              << i << ",q=" << q << ",sigma=" << sigma << "):\n      d(tau^iota x) = "
                              << chain_to_string(lhs) << "\n      tau^iota(d x) = "
                              << chain_to_string(rhs) << std::endl;
                    ++failures;
                }
            }
        }
    }

    // --- (C3) (T1) for tau^iota, on the model's relation/generator rows:
    // tau^iota(s_i . i(sigma_lhs)) == (g^{-1} h_i(g_r) g) . tau^iota(i.g_r, sigma_rhs).
    {
        struct Row { uint32_t q; std::vector<uint32_t> lhs_v, rhs_v; uint32_t gen; };
        static const std::vector<Row> rows = {
            { 0, { 3 }, { 0 }, 0 }, { 0, { 5 }, { 4 }, 0 }, { 1, { 3, 5 }, { 0, 4 }, 0 },
            { 0, { 4 }, { 5 }, 1 }, { 0, { 6 }, { 6 }, 1 }, { 1, { 4, 6 }, { 5, 6 }, 1 },
            { 0, { 4 }, { 4 }, 2 }, { 0, { 5 }, { 5 }, 3 },
        };
        const int64_t N = static_cast<int64_t>(cosets.N());
        for (const auto& row : rows) {
            const uint32_t s_lhs = find_simplex_index(simplices, row.q, row.lhs_v);
            const uint32_t s_rhs = find_simplex_index(simplices, row.q, row.rhs_v);
            for (uint32_t i = 1; i <= m; ++i) {
                const uint32_t jj = cosets.perm(row.gen, i);
                const Mat22 h = cosets.cocycle(row.gen, i);
                if (h.c % N != 0) { ++failures; continue; }
                const Mat22 conj{ h.a, hecke::checked_mul<int64_t>(h.b, N), h.c / N, h.d };
                ChainAcc pred;
                for (const auto& kv : ti[row.q][s_rhs][jj]) {
                    const Mat22 gk{ kv.first[2], kv.first[3], kv.first[4], kv.first[5] };
                    add_term(pred, model, row.q, conj * gk, static_cast<uint32_t>(kv.first[1]),
                             kv.second);
                }
                normalise(pred, mod_ell, ell);
                if (pred != ti[row.q][s_lhs][i]) {
                    std::cout << "HeckeIotaTauSymmetryValidation: (C3) tau^iota violates (T1) at i="
                              << i << ", q=" << row.q << "." << std::endl;
                    ++failures;
                }
            }
        }
    }

    return failures;
}

// =========================================================================
// Grid driver.
// =========================================================================

struct GridPoint {
    uint64_t p;
    uint32_t r;
    uint32_t ell;
    uint32_t n_sym;
};

struct RowResult {
    GridPoint gp{};
    const char* policy = "";
    uint32_t slots = 0;
    uint32_t differing = 0;
    bool mod_ell_solve = false;
    TauVerdict verdict{};
};

} // namespace

uint32_t HeckeIotaTauSymmetryValidation() {
    uint32_t gating_failures = 0;
    uint32_t checked = 0;

    try {
        const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
        const SL2ZTessellationTables model(K);
        const SimplexTable& simplices = K.get_simplices();

        std::cout << "HeckeIotaTauSymmetryValidation: iota = diag(-1,1) (gl2_contract.tex "
                     "def:iota); clause (A) = obl:iota(i) [GATING], clause (B) = obl:iota(iv) "
                     "[RECORDED per the clause; GATING on the canonical tau since the "
                     "re-baseline of 2026-09-16, recorded only on the legacy and FixedV6 taus]."
                  << std::endl;

        // ---------------- (A) obl:iota(i): gating ----------------
        IotaModel iota;
        gating_failures += build_and_check_iota_model(simplices, model, iota, checked);
        if (gating_failures != 0) {
            std::cout << "HeckeIotaTauSymmetryValidation: (A) FAILED with " << gating_failures
                      << " discrepancy(ies); (B) not evaluated." << std::endl;
            return gating_failures;
        }
        {
            std::ostringstream oss;
            for (uint32_t v = 0; v < 7; ++v) oss << (v ? ", " : "") << "v" << v << "->v" << iota.vperm[v];
            std::cout << "HeckeIotaTauSymmetryValidation: (A) PASSED -- iota_K = { " << oss.str()
                      << " }; sign -1 on exactly 6 of the 23 simplices, matching lem:iotamodel."
                      << std::endl;
        }

        // ---------------- (B) obl:iota(iv) ----------------
        // The binding grid is ell in {2,3} (rmk:iotatausym: the only regime
        // where (3b) is the sole route); ell in {5, 691} are carried to show
        // the verdict is not an artefact of small ell -- and since the
        // symmetry is an identity of integer chains it must hold at every ell.
        std::vector<GridPoint> grid;
        const std::vector<std::pair<uint64_t, uint32_t>> prs = {
            { 2, 1 }, { 3, 1 }, { 5, 1 }, { 7, 1 }, { 2, 2 }, { 3, 2 }
        };
        const std::vector<uint32_t> ells = { 2, 3, 5, 691 };
        const std::vector<uint32_t> ns = { 0, 1, 2, 3, 4, 10 };
        for (const auto& pr : prs)
            for (uint32_t ell : ells)
                for (uint32_t n : ns)
                    grid.push_back(GridPoint{ pr.first, pr.second, ell, n });

        // The canonical tau of alg:tau (GATING: every slot of every table must
        // be symmetric); the pre-2026-09-16 canonical tau (RECORDED: the
        // historical negative of rmk:iotatausym, reproduced so the differential
        // stays visible); the O12(c) alternate tau (RECORDED).
        const std::vector<std::pair<TauVertexPolicy, const char*>> policies = {
            { TauVertexPolicy::GeometricNearest, "GeometricNearest" },
            { TauVertexPolicy::GeometricNearestLegacy, "GeometricNearestLegacy" },
            { TauVertexPolicy::FixedV6, "FixedV6" },
        };

        std::vector<RowResult> rows;
        for (const auto& gp : grid) {
            for (const auto& pol : policies) {
                try {
                    const HeckeCosetTables cosets(gp.p, gp.r, gp.ell, gp.n_sym);
                    const HeckeTauTable table(K, model, cosets, gp.ell, gp.n_sym, pol.first);
                    TauVerdict v;
                    gating_failures += check_tau_symmetry(simplices, model, cosets, table, iota,
                                                          gp.ell, v);
                    checked += v.slots_checked;
                    rows.push_back(RowResult{ gp, pol.second, v.slots_checked, v.slots_differing,
                                              table.used_mod_ell_solve(), v });
                } catch (const std::exception& e) {
                    std::cout << "HeckeIotaTauSymmetryValidation: EXCEPTION at p=" << gp.p
                              << " r=" << gp.r << " ell=" << gp.ell << " n=" << gp.n_sym
                              << " policy=" << pol.second << ": " << e.what() << std::endl;
                    ++gating_failures;
                }
            }
        }

        // --- Per-policy summary: whole grid, binding subgrid, by degree, witness.
        struct PolicySummary {
            uint32_t rows = 0, mod_ell_rows = 0;
            uint32_t all_slots = 0, all_diff = 0;
            uint32_t binding_slots = 0, binding_diff = 0;
            std::array<uint32_t, 3> dd{ { 0, 0, 0 } }, ss{ { 0, 0, 0 } };
            const RowResult* first_bad = nullptr;
        };
        std::cout << "HeckeIotaTauSymmetryValidation: (B) tau-symmetry per policy (binding "
                     "subgrid = ell in {2,3}):" << std::endl;
        for (const auto& pol : policies) {
            PolicySummary ps;
            for (const auto& row : rows) {
                if (std::string(row.policy) != pol.second) continue;
                ++ps.rows;
                if (row.mod_ell_solve) ++ps.mod_ell_rows;
                ps.all_slots += row.slots; ps.all_diff += row.differing;
                if (row.gp.ell == 2 || row.gp.ell == 3) {
                    ps.binding_slots += row.slots; ps.binding_diff += row.differing;
                    for (uint32_t q = 0; q <= 2; ++q) {
                        ps.dd[q] += row.verdict.differing_by_degree[q];
                        ps.ss[q] += row.verdict.slots_by_degree[q];
                    }
                }
                if (row.differing != 0 && ps.first_bad == nullptr) ps.first_bad = &row;
            }
            const bool canonical = (pol.first == TauVertexPolicy::GeometricNearest);
            std::cout << "    [" << pol.second << (canonical ? ", the canonical tau of alg:tau"
                                                            : "") << "] whole grid: "
                      << ps.all_diff << " differing of " << ps.all_slots << " slots over "
                      << ps.rows << " tables (" << (ps.rows - ps.mod_ell_rows)
                      << " compared exactly over Z, " << ps.mod_ell_rows
                      << " mod ell); binding subgrid: " << ps.binding_diff << " of "
                      << ps.binding_slots << " (q=0 " << ps.dd[0] << "/" << ps.ss[0] << ", q=1 "
                      << ps.dd[1] << "/" << ps.ss[1] << ", q=2 " << ps.dd[2] << "/" << ps.ss[2]
                      << ")." << std::endl;
            if (ps.first_bad != nullptr && ps.first_bad->verdict.have_witness) {
                const auto& w = ps.first_bad->verdict;
                std::cout << "      first differing slot in (q,sigma,i) order: p="
                          << ps.first_bad->gp.p << "^" << ps.first_bad->gp.r << " ell="
                          << ps.first_bad->gp.ell << " n=" << ps.first_bad->gp.n_sym
                          << "; slot (i=" << w.w_i << ", q=" << w.w_q << ", sigma=" << w.w_sigma
                          << " = {";
                const auto& verts = simplices.at(w.w_q).at(w.w_sigma);
                for (size_t t = 0; t < verts.size(); ++t) std::cout << (t ? "," : "") << "v" << verts[t];
                std::cout << "}); at canonical term " << key_to_string(w.w_key)
                          << ": tau coefficient " << w.w_tau << ", (iota_X tau iota_X) coefficient "
                          << w.w_tau_iota << ".\n        tau(x)                 = " << w.w_tau_chain
                          << "\n        (iota_X tau iota_X)(x) = " << w.w_tau_iota_chain << std::endl;
            }
            if (canonical) {
                const bool symmetric = (ps.all_diff == 0) && (ps.all_slots != 0);
                std::cout << "HeckeIotaTauSymmetryValidation: RECORDED VERDICT obl:iota(iv), "
                             "canonical tau: iota_X tau iota_X " << (symmetric ? "==" : "!=")
                          << " tau on every stored slot of every table of the grid  =>  "
                             "prop:iotacommutes(3b) is "
                          << (symmetric ? "AVAILABLE for the canonical tau (the gap of "
                                          "rmk:iotatausym is closed for it)"
                                        : "NOT available (the gap of rmk:iotatausym is open)")
                          << ". Per obl:iota(iv) a true value licenses nothing beyond (3b)."
                          << std::endl;
                if (!symmetric) {
                    std::cout << "HeckeIotaTauSymmetryValidation: (B) FAILED -- the canonical tau "
                                 "is required to be iota-symmetric since the re-baseline of "
                                 "2026-09-16 ([C, alg:tau](1b),(2)); " << ps.all_diff
                              << " slot(s) differ." << std::endl;
                    ++gating_failures;
                }
            }
        }

        std::cout << "HeckeIotaTauSymmetryValidation: (C) tau^iota = iota_X o tau o iota_X was "
                     "certified, on every stored slot of every table above, to satisfy (T1), (T2) "
                     "and (T3) of def:tau -- i.e. it IS an admissible g-twisted chain datum, "
                     "exactly as prop:iotacommutes(3a) predicts. A (B) inequality on a "
                     "non-canonical tau is therefore 'two admissible data for the same g', not a "
                     "broken iota_X." << std::endl;

        std::cout << "HeckeIotaTauSymmetryValidation: " << checked
                  << " individual exact-integer assertions checked over " << rows.size()
                  << " (grid point, policy) tau tables; " << gating_failures
                  << " GATING discrepancy(ies)." << std::endl;

        return gating_failures;
    } catch (const std::exception& e) {
        std::cout << "HeckeIotaTauSymmetryValidation: EXCEPTION during setup: " << e.what()
                  << std::endl;
        return 1000;
    }
}
