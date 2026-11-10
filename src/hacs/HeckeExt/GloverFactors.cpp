// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 4, WP-7 (phase E6) -- HeckeExt/GloverFactors.
// See the header for the module, the convention and the recursion's derivation.

#include "GloverFactors.h"

#include "../../ModArithmetic/EngineCheck.h"

#include <algorithm>
#include <sstream>

namespace hecke_ext {

uint32_t GloverDecomposition::dimension() const {
    uint32_t d = 0;
    for (const GloverFactor& f : factors) d += f.dimension();
    return d;
}

uint32_t GloverDecomposition::length() const {
    uint32_t L = 0;
    for (const GloverFactor& f : factors) L += f.multiplicity;
    return L;
}

std::vector<uint32_t> GloverDecomposition::sl2_multiplicities() const {
    ENGINE_CHECK_MSG(ell >= 2, "GloverDecomposition::sl2_multiplicities: ell < 2");
    std::vector<uint32_t> m(ell, 0u);
    for (const GloverFactor& f : factors) {
        ENGINE_CHECK_MSG(f.sym < ell, "GloverDecomposition::sl2_multiplicities: sym >= ell");
        m[f.sym] += f.multiplicity;
    }
    return m;
}

std::string GloverDecomposition::to_string() const {
    std::ostringstream os;
    os << "Sym^" << n << "(F_" << ell << "^2) =";
    if (factors.empty()) os << " 0";
    for (size_t i = 0; i < factors.size(); ++i) {
        const GloverFactor& f = factors[i];
        os << (i ? " +" : "");
        if (f.multiplicity != 1u) os << " " << f.multiplicity << "*";
        os << " Sym^" << f.sym;
        if (f.det_twist != 0u) os << "(x)det^" << f.det_twist;
    }
    os << "   [dim " << dimension() << ", length " << length() << "]";
    return os.str();
}

std::string GloverDecomposition::to_json() const {
    std::ostringstream os;
    os << "{\"schema\":\"hecke_glover_factors/v1\",\"n\":" << n << ",\"ell\":" << ell
       << ",\"dimension\":" << dimension() << ",\"length\":" << length() << ",\"factors\":[";
    for (size_t i = 0; i < factors.size(); ++i) {
        const GloverFactor& f = factors[i];
        os << (i ? "," : "") << "{\"sym\":" << f.sym << ",\"det\":" << f.det_twist
           << ",\"multiplicity\":" << f.multiplicity << "}";
    }
    os << "]}";
    return os.str();
}

GloverDecomposition GloverFactors(uint32_t n, uint32_t ell) {
    ENGINE_CHECK_MSG(ell >= 2, "hecke_ext::GloverFactors: ell < 2");

    // The det-twist lives in Z/(ell - 1); for ell = 2 the determinant is
    // identically 1 and the group of twists is trivial.
    const uint32_t period = (ell > 2u) ? (ell - 1u) : 1u;

    // (sym, det_twist) pairs, one per step of the recursion, merged afterwards.
    std::vector<std::pair<uint32_t, uint32_t>> raw;
    raw.reserve(static_cast<size_t>(2u * (n / (ell + 1u)) + 2u));

    // k is signed: the last step of the recursion at k = ell leaves k = -1,
    // which ends it (Sym^{-1} = 0).
    int64_t k = static_cast<int64_t>(n); // the current symmetric power
    int64_t twist = 0;                   // the accumulated det-twist
    while (k >= 0) {
        if (k <= static_cast<int64_t>(ell) - 1) {
            raw.emplace_back(static_cast<uint32_t>(k), static_cast<uint32_t>(twist % period));
            break;
        }
        const int64_t r = k % static_cast<int64_t>(period);
        raw.emplace_back(static_cast<uint32_t>(r), static_cast<uint32_t>(twist % period));
        raw.emplace_back(static_cast<uint32_t>(static_cast<int64_t>(ell) - 1 - r),
                         static_cast<uint32_t>((twist + k) % period));
        k -= static_cast<int64_t>(ell) + 1;
        twist += 1;
    }

    std::sort(raw.begin(), raw.end());

    GloverDecomposition out;
    out.n = n;
    out.ell = ell;
    for (size_t i = 0; i < raw.size();) {
        size_t j = i;
        while (j < raw.size() && raw[j] == raw[i]) ++j;
        GloverFactor f;
        f.sym = raw[i].first;
        f.det_twist = raw[i].second;
        f.multiplicity = static_cast<uint32_t>(j - i);
        out.factors.push_back(f);
        i = j;
    }

    // --- The invariants the header states, all O(length) ---------------------
    ENGINE_CHECK_MSG(out.dimension() == n + 1u,
                     "hecke_ext::GloverFactors: the factors do not sum to dim Sym^n");
    for (const GloverFactor& f : out.factors) {
        ENGINE_CHECK_MSG(f.sym <= ell - 1u,
                         "hecke_ext::GloverFactors: a factor has sym > ell - 1");
        ENGINE_CHECK_MSG(f.det_twist < period,
                         "hecke_ext::GloverFactors: a det-twist is out of Z/(ell - 1)");
        // Central character: sym + 2 * det == n  (mod ell - 1).
        ENGINE_CHECK_MSG(((f.sym + 2u * f.det_twist) % period) == (n % period),
                         "hecke_ext::GloverFactors: a factor violates sym + 2 det == n mod ell - 1");
    }
    return out;
}

} // namespace hecke_ext
