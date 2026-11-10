// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// (hecke-extensions slot 7, WP-5 / phase E6 cont.) HeckeExt/HeckeTangentPair
// -- see the header for what is computed and why.
// ===========================================================================

#include "HeckeTangentPair.h"

#include "HeckeExtDriver.h"
#include "HeckeLocalAlgebra.h"

#include "../Hecke/HeckeSettings.h"
#include "../Hecke/SubspaceOps.h"
#include "../../ModArithmetic/EngineCheck.h"

#include <algorithm>
#include <map>
#include <sstream>

namespace hecke_ext {

namespace {

void U64List(std::ostringstream& o, const std::vector<uint64_t>& v) {
    o << '[';
    for (size_t i = 0; i < v.size(); ++i) o << (i ? "," : "") << v[i];
    o << ']';
}

// The weight of a collected system: the least n + 2 over its H1 occurrences
// (0 if it has none).
uint32_t WeightOf(const AtlasSystem& s) {
    uint32_t w = 0;
    for (const AtlasOccurrence& o : s.occurrences)
        if (o.group == "H1" && (w == 0 || o.n + 2u < w)) w = o.n + 2u;
    return w;
}

// Runs the driver at (n, ell) on the anemic pool with U_ell off and fills the
// computed fields of `t` for the system `sys`.
void ComputeOne(const AtlasStore& s, const AtlasSystem& sys, TangentPair& t) {
    hecke::UEllSetting off;
    off.mode = hecke::UEllMode::Off;
    const hecke::ScopedUEll scoped(off);

    ExtDriverOptions opts;
    opts.good_primes = s.primes;
    opts.run_commutant = false;
    opts.run_sturm = false;
    // (slot-7 merge of WP-5 onto WP-3's stage 2) Exactly the store's pool:
    // no in-process atlas.  Left at its default, the driver would build a
    // second Sigma(ell) under the u_ell = off fingerprint at ell <= 13 and
    // add its S_sep to Stage 1 -- not what this computation was written for.
    opts.consult_atlas = false;
    HeckeExtDriver d(t.n, s.ell, opts);
    d.run();

    const std::vector<AtlasIdealCharacter> chars = AtlasIdealCharacters(d);
    const CanonicalField F(s.ell, sys.r);
    int64_t found = -1;
    for (size_t i = 0; i < chars.size(); ++i) {
        const AtlasIdealCharacter& c = chars[i];
        if (c.group != "H1" || !c.local || c.r != sys.r || c.orbit.empty()) continue;
        if (c.primes != s.primes) continue;
        if (AtlasOrbitKey(F, c.orbit[0]) == sys.eigenvalues) {
            found = static_cast<int64_t>(i);
            break;
        }
    }
    if (found < 0) {
        // Not local, or not there: say which.
        bool any_nonlocal = false;
        for (const AtlasIdealCharacter& c : chars)
            if (c.group == "H1" && !c.local) any_nonlocal = true;
        t.status = any_nonlocal ? "not_local" : "not_found";
        t.reason = any_nonlocal ? "an H1 ideal of the cell is not certified local"
                                : "no H1 ideal of the cell carries the system";
        return;
    }
    const LocalAlgebra& la = d.local_algebra(static_cast<uint32_t>(found));
    ENGINE_CHECK_MSG(la.ok, "ComputeTangentPairs: the matched ideal is not local");
    t.D = la.D;
    t.r = la.f;
    t.multiplicity = (la.f > 0) ? la.D / (2u * la.f) : 0u;
    t.embdim = la.embdim;
    for (size_t j = 0; j < la.primes.size(); ++j) {
        ENGINE_CHECK_MSG(la.primes[j] != s.ell, "ComputeTangentPairs: U_ell present with u_ell = off");
        if (la.active[j]) t.active.push_back(la.primes[j]);
    }

    // The greedy K-spanning subset of the anemic primes, ascending: the
    // K-span of n_p-bar is the F_ell-span of w_hat^i n_p, 0 <= i < f, modulo
    // m_S^2 (LocalAlgebra's own tangent test, [E] alg:stage3(3)).
    const uint32_t width = la.D * la.D;
    hecke::RrefRowSpan chosen(width, s.ell);
    if (la.m_powers.size() >= 2)
        for (const ExtMatrix& b : la.m_powers[1]) chosen.insert(hecke::VecOfMatrix(b));
    const uint32_t dim_m2 = chosen.dim();
    for (size_t j = 0; j < la.n.size(); ++j) {
        if (chosen.dim() == la.dim_m_S) break;
        if (chosen.contains(hecke::VecOfMatrix(la.n[j]))) continue;
        t.generating.push_back(la.primes[j]);
        ExtMatrix pw(la.D, s.ell);   // identity
        for (uint32_t i = 0; i < la.f; ++i) {
            chosen.insert(hecke::VecOfMatrix(pw * la.n[j]));
            if (i + 1 < la.f) pw = pw * la.w_hat;
        }
    }
    ENGINE_CHECK_MSG((chosen.dim() - dim_m2) == la.embdim * la.f,
                     "ComputeTangentPairs: the anemic n_p do not K-span m/m^2");
    ENGINE_CHECK_MSG(t.generating == la.minimal_active,
                     "ComputeTangentPairs: the anemic greedy differs from minimal_active");
    // Each greedy prime adds exactly one K-dimension (the K-span of one n_p-bar).
    ENGINE_CHECK_MSG(t.generating.size() == la.embdim,
                     "ComputeTangentPairs: |generating| != embdim");
    t.pair = t.generating.size() <= 2u;
    t.status = "computed";
}

} // namespace

std::string TangentPair::to_json() const {
    std::ostringstream o;
    o << "{\"schema\":\"" << kTangentPairSchema << "\",\"ell\":" << ell << ",\"system\":" << system
      << ",\"base\":" << base << ",\"twist\":" << twist << ",\"weight\":" << weight
      << ",\"n\":" << n << ",\"status\":\"" << status << "\"";
    if (!reason.empty()) o << ",\"reason\":\"" << reason << "\"";
    if (status == "computed" || status == "inherited") {
        o << ",\"D\":" << D << ",\"r\":" << r << ",\"multiplicity\":" << multiplicity
          << ",\"embdim\":" << embdim << ",\"active\":";
        U64List(o, active);
        o << ",\"generating\":";
        U64List(o, generating);
        o << ",\"pair\":" << (pair ? "true" : "false");
    }
    o << '}';
    return o.str();
}

std::vector<TangentPair> ComputeTangentPairs(const AtlasStore& s, uint32_t n_reach) {
    std::vector<TangentPair> out;
    std::map<uint32_t, TangentPair> by_base;   // computed / deferred base results
    const uint64_t shift = static_cast<uint64_t>(s.ell) * s.ell - 1u;
    for (const AtlasSystem& sys : s.systems) {
        if (sys.type != "cuspidal") continue;
        TangentPair t;
        t.ell = s.ell;
        t.system = sys.id;
        t.base = sys.base;
        t.twist = sys.twist;
        ENGINE_CHECK_MSG(sys.base < s.systems.size() && s.systems[sys.base].collected,
                         "ComputeTangentPairs: a system whose base is not collected");
        const AtlasSystem& base = s.systems[sys.base];
        t.weight = WeightOf(base);
        const uint64_t n = t.weight >= 2u ? (t.weight - 2u) + shift : 0u;
        t.n = static_cast<uint32_t>(std::min<uint64_t>(n, 0xffffffffu));

        auto it = by_base.find(base.id);
        if (it == by_base.end()) {
            TangentPair b = t;
            b.system = base.id;
            b.base = base.id;
            b.twist = 0;
            if (t.weight < 2u) {
                b.status = "not_found";
                b.reason = "the base system has no H1 occurrence";
            } else if (n > n_reach) {
                b.status = "deferred";
                b.reason = "n = " + std::to_string(n) + " > " + std::to_string(n_reach);
            } else {
                ComputeOne(s, base, b);
            }
            it = by_base.emplace(base.id, b).first;
        }
        const TangentPair& b = it->second;
        if (sys.twist == 0u) {
            out.push_back(b);
            continue;
        }
        t.reason = b.reason;
        if (b.status == "computed") {
            t.status = "inherited";
            t.reason = "theta^" + std::to_string(sys.twist) + " of system " + std::to_string(base.id);
            t.D = b.D;
            t.r = b.r;
            t.multiplicity = b.multiplicity;
            t.embdim = b.embdim;
            t.active = b.active;
            t.generating = b.generating;
            t.pair = b.pair;
        } else {
            t.status = b.status;
        }
        out.push_back(t);
    }
    return out;
}

} // namespace hecke_ext
