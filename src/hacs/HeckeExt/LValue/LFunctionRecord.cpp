// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 20, stretch S1) the record lfunction/v1 -- see LFunctionRecord.h.
#include "LFunctionRecord.h"

#include "Bernoulli.h"

#include "../../../ModArithmetic/EngineCheck.h"
#include "../../../ModArithmetic/ScalarMod.h"
#include "../../../ModArithmetic/Fnv.h"

#include <algorithm>
#include <cstdio>
#include <sstream>
#include <stdexcept>

namespace hecke_ext {
namespace lvalue {

namespace {
using u128 = unsigned __int128;

using engine::scalar::pow_wrap;
bool IsPrimeSmall(uint64_t x) {
    if (x < 2) return false;
    for (uint64_t d = 2; d * d <= x; ++d)
        if (x % d == 0) return false;
    return true;
}
std::string JsonEscape(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (char ch : s) {
        const unsigned char c = static_cast<unsigned char>(ch);
        if (c == '"' || c == '\\') {
            o += '\\';
            o += ch;
        } else if (c < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof buf, "\\u%04x", c);
            o += buf;
        } else {
            o += ch;
        }
    }
    return o;
}
template<class T>
void JsonList(std::ostringstream& os, const std::vector<T>& v) {
    os << '[';
    for (size_t i = 0; i < v.size(); ++i) os << (i ? "," : "") << v[i];
    os << ']';
}
std::string SeparatorText(const std::map<uint64_t, uint64_t>& sep) {
    std::ostringstream os;
    bool first = true;
    for (const auto& pc : sep) {
        if (!first) os << '+';
        first = false;
        if (pc.second != 1) os << pc.second;
        os << "T_" << pc.first;
    }
    return os.str();
}
} // namespace

std::vector<uint64_t> BranchElementAtLevel(const ModularSymbolMeasure& me, uint32_t a, uint32_t level) {
    ENGINE_CHECK_MSG(level >= 1 && level <= me.M, "BranchElementAtLevel: 1 <= level <= M");
    if (level == me.M) return me.branch(a);
    const uint64_t ell = me.ell, MM = pow_wrap(ell, level), G = pow_wrap(ell, level - 1), u = 1 + ell, q = me.q;
    std::vector<uint64_t> deltas(ell), wts(ell);
    for (uint64_t d = 1; d < ell; ++d) {
        deltas[d] = TeichmullerPowMod(d, 1, ell, level) % MM;
        wts[d] = TeichmullerPowMod(d, static_cast<uint32_t>(a % (ell - 1)), ell, me.R) % q;
    }
    std::vector<uint64_t> g(G, 0);
    uint64_t ui = 1 % MM;
    for (uint64_t i = 0; i < G; ++i) {
        u128 s = 0;
        for (uint64_t d = 1; d < ell; ++d) s = (s + static_cast<u128>(wts[d]) * me.mu[level][static_cast<uint64_t>(static_cast<u128>(deltas[d]) * ui % MM)]) % q;
        g[i] = static_cast<uint64_t>(s);
        ui = static_cast<uint64_t>(static_cast<u128>(ui) * u % MM);
    }
    return g;
}

std::vector<uint64_t> TCoefficients(const std::vector<uint64_t>& g, uint64_t q) {
    const size_t G = g.size();
    std::vector<uint64_t> acc(G, 0);
    // Horner in X = 1 + T: acc <- acc * (1 + T) + g_i, i = G-1 .. 0.  Before the step of i, acc has degree <= G - 2 - i,
    // so the product by (1 + T) touches t <= G - 1 - i and the degree never reaches G.
    for (size_t i = G; i-- > 0;) {
        for (size_t t = G - 1 - i; t >= 1; --t) {
            const uint64_t s = acc[t] + acc[t - 1];
            acc[t] = s >= q ? s - q : s;
        }
        const uint64_t s = acc[0] + g[i] % q;
        acc[0] = s >= q ? s - q : s;
    }
    return acc;
}

std::vector<uint32_t> CertifiedTPrecision(uint64_t ell, uint32_t R, uint32_t level) {
    ENGINE_CHECK_MSG(level >= 1, "CertifiedTPrecision: level >= 1");
    const uint64_t G = pow_wrap(ell, level - 1);
    std::vector<uint32_t> p(G, 0);
    if (G == 0) return p;
    p[0] = R;
    uint32_t lg = 0;          // floor(log_ell t)
    uint64_t next = ell;      // ell^{lg+1}
    for (uint64_t t = 1; t < G; ++t) {
        while (t >= next) {
            ++lg;
            next *= ell;
        }
        const int64_t e = static_cast<int64_t>(level) - 1 - static_cast<int64_t>(lg);
        p[t] = static_cast<uint32_t>(std::max<int64_t>(0, std::min<int64_t>(R, e)));
    }
    return p;
}

uint64_t EvalTPolynomial(const std::vector<uint64_t>& c, uint64_t x, uint64_t q) {
    u128 acc = 0;
    for (size_t t = c.size(); t-- > 0;) acc = (acc * (x % q) + c[t] % q) % q;
    return static_cast<uint64_t>(acc);
}

std::vector<uint64_t> FoldBranchElement(const std::vector<uint64_t>& g, size_t Gsmall, uint64_t q) {
    ENGINE_CHECK_MSG(Gsmall >= 1 && g.size() % Gsmall == 0, "FoldBranchElement: Gsmall | |g|");
    std::vector<uint64_t> o(Gsmall, 0);
    for (size_t i = 0; i < g.size(); ++i) {
        const uint64_t s = o[i % Gsmall] + g[i] % q;
        o[i % Gsmall] = s >= q ? s - q : s;
    }
    return o;
}

namespace {
LFunctionBranch LFunctionOfBranch(const ModularSymbolMeasure& me, uint32_t a) {
    ENGINE_CHECK_MSG(a + 1 < me.ell, "LFunctionOfBranch: branch a in [0, ell - 1)");
    LFunctionBranch b;
    b.a = a;
    const std::vector<uint64_t> g = me.branch(a);
    b.vanishes = std::all_of(g.begin(), g.end(), [](uint64_t x) { return x == 0; });
    b.coefficients = TCoefficients(g, me.q);
    b.precision = CertifiedTPrecision(me.ell, me.R, me.M);
    const IwasawaInvariants I = me.invariants(a);
    b.mu = I.mu;
    b.lambda = I.lambda;
    b.readable = I.readable;
    for (size_t t = 0; t < b.coefficients.size(); ++t)
        if (b.coefficients[t] % me.ell != 0) {
            b.lambda_mod_ell = static_cast<int64_t>(t);
            break;
        }
    b.values_modulus = pow_wrap(me.ell, std::min(me.M, me.R));
    const uint64_t u = (1 + me.ell) % me.q;
    uint64_t uj = 1 % me.q;
    for (uint32_t j = 0; j <= me.n; ++j) {
        const uint64_t x = (uj + me.q - 1 % me.q) % me.q;   // u^j - 1
        b.critical_values.push_back(EvalTPolynomial(b.coefficients, x, me.q) % b.values_modulus);
        uj = static_cast<uint64_t>(static_cast<u128>(uj) * u % me.q);
    }
    return b;
}
} // namespace

std::string LFunctionRecord::canonical() const {
    std::ostringstream os;
    os << "{\"schema\":\"" << kSchema << "\",\"n\":" << n << ",\"k\":" << k << ",\"ell\":" << ell << ",\"R\":" << R << ",\"M\":" << M
       << ",\"G\":" << G << ",\"u\":" << u << ",\"sign\":\"" << JsonEscape(form.sign) << "\",\"geometric\":\"" << geometric << "\"";
    os << ",\"form\":{\"separator\":\"" << JsonEscape(form.separator) << "\",\"target_root\":" << form.target_root
       << ",\"eisenstein_congruent\":" << (form.eisenstein_congruent ? "true" : "false") << ",\"a_p\":{";
    bool first = true;
    for (const auto& kv : form.a_p) {
        os << (first ? "" : ",") << "\"" << kv.first << "\":" << kv.second;
        first = false;
    }
    os << "},\"a_ell\":" << a_ell << ",\"alpha\":" << alpha << ",\"symbol\":\"" << JsonEscape(form.symbol_normalisation) << "\"}";
    os << ",\"truncation\":{\"ring\":\"Z/ell^R[T]/(omega_{M-1}(T))\",\"omega\":\"(1+T)^{ell^{M-1}}-1\","
          "\"basis\":\"T^t, t < G = ell^{M-1}; c_t = sum_i g_i C(i,t), g_i = int_{delta u^i + ell^M} omega^a dmu\","
          "\"precision_rule\":\"prec(0) = R; prec(t) = min(R, M-1-floor(log_ell t)), 1 <= t < G (E lem:lvlfunprec)\"}";
    os << ",\"normalisation\":\"" << JsonEscape(normalisation) << "\",\"branches\":[";
    for (size_t i = 0; i < branches.size(); ++i) {
        const LFunctionBranch& b = branches[i];
        os << (i ? "," : "") << "{\"a\":" << b.a << ",\"vanishes\":" << (b.vanishes ? "true" : "false") << ",\"mu\":" << b.mu
           << ",\"lambda\":" << b.lambda << ",\"readable\":" << (b.readable ? "true" : "false") << ",\"lambda_mod_ell\":" << b.lambda_mod_ell
           << ",\"coefficients\":";
        JsonList(os, b.coefficients);
        os << ",\"precision\":";
        JsonList(os, b.precision);
        os << ",\"values_modulus\":" << b.values_modulus << ",\"critical_values\":";
        JsonList(os, b.critical_values);
        os << "}";
    }
    os << "]}";
    return os.str();
}

uint64_t LFunctionRecord::hash() const { return engine::FnvBytes(engine::kFnvOffset64, canonical()); }

std::string LFunctionRecord::line(const LFunctionBranch& b) const {
    std::ostringstream os;
    os << "LFUNCTION schema=" << kSchema << " n=" << n << " ell=" << ell << " R=" << R << " M=" << M << " sign=" << form.sign << " a=" << b.a
       << " vanishes=" << (b.vanishes ? 1 : 0) << " mu=" << b.mu << " lambda=" << b.lambda << " readable=" << (b.readable ? 1 : 0)
       << " lambda_mod_ell=" << b.lambda_mod_ell << " c0=" << (b.coefficients.empty() ? 0 : b.coefficients[0])
       << " c1=" << (b.coefficients.size() > 1 ? b.coefficients[1] : 0);
    return os.str();
}

LFunctionRecord MakeLFunctionRecord(const ModularSymbolMeasure& me, const LFunctionForm& form, const std::vector<uint32_t>& branches) {
    ENGINE_CHECK_MSG(me.M >= 1 && !me.mu.empty(), "MakeLFunctionRecord: an empty measure");
    LFunctionRecord r;
    r.n = me.n;
    r.k = me.n + 2;
    r.ell = me.ell;
    r.R = me.R;
    r.M = me.M;
    r.G = pow_wrap(me.ell, me.M - 1);
    r.u = 1 + me.ell;
    r.a_ell = me.a_ell;
    r.alpha = me.alpha;
    r.geometric = me.geometric_sign > 0 ? "even" : me.geometric_sign < 0 ? "odd" : "mixed";
    r.form = form;
    r.normalisation = me.normalisation().to_string();
    std::vector<uint32_t> bs = branches;
    if (bs.empty())
        for (uint32_t a = 0; a + 1 < me.ell; ++a) bs.push_back(a);
    for (uint32_t a : bs) r.branches.push_back(LFunctionOfBranch(me, a));
    return r;
}

LFunctionRecord ComputeLFunction(const LFunctionRequest& rq) {
    using birch::RunSymbolCell;
    using birch::SymbolCell;
    if (rq.ell < 5 || rq.ell >= (uint64_t(1) << 31) || !IsPrimeSmall(rq.ell))
        throw std::invalid_argument("lfunction: ell must be a prime, 5 <= ell < 2^31 (the domain of [E] rmk:lvshapes (B))");
    if (rq.n < 2 || rq.n % 2 != 0) throw std::invalid_argument("lfunction: n must be even and >= 2 (odd n vanishes on this model)");
    if (rq.M < 2) throw std::invalid_argument("lfunction: the level M must be >= 2 (G = ell^{M-1} >= ell)");
    if (rq.R < 1) throw std::invalid_argument("lfunction: R >= 1");
    if (rq.sign != "minus" && rq.sign != "plus") throw std::invalid_argument("lfunction: sign is \"minus\" or \"plus\"");
    if (rq.sep.empty()) throw std::invalid_argument("lfunction: an empty separating operator");
    {
        u128 x = 1;
        for (uint32_t i = 0; i < rq.R; ++i) x *= rq.ell;
        if (x >= (u128(1) << 62)) throw std::invalid_argument("lfunction: ell^R >= 2^62");
        u128 b = 1;
        for (uint32_t i = 0; i < rq.M; ++i) {
            b *= rq.ell;
            if (b > LFunctionRequest::kMaxBalls) break;
        }
        if (b > LFunctionRequest::kMaxBalls)
            throw std::invalid_argument("lfunction: ell^M = " + std::to_string(rq.ell) + "^" + std::to_string(rq.M) + " exceeds the budget of " +
                                        std::to_string(LFunctionRequest::kMaxBalls) + " balls");
    }
    for (uint32_t a : rq.branches)
        if (a + 1 >= rq.ell) throw std::invalid_argument("lfunction: a branch must lie in [0, ell - 1)");
    const SymbolCell C = RunSymbolCell(rq.n, rq.ell, rq.R, rq.sep, true);
    if (!C.simple) throw std::invalid_argument("lfunction: the symbol cell refused: " + C.refusal);
    const bool plus = rq.sign == "plus";
    if (plus && !C.have_r_plus) throw std::invalid_argument("lfunction: no r^+ with r^+_0 a unit for this eigen-system");
    const uint64_t a_ell = C.a_p.at(rq.ell);
    if (a_ell % rq.ell == 0) throw std::invalid_argument("lfunction: not ordinary (a_ell == 0 mod ell)");
    const birch::Ring& Rg = C.ring;
    const birch::SymRho rho(rq.n, Rg);
    const uint64_t alpha = birch::UnitRoot(Rg, a_ell, rq.n + 2);
    const ModularSymbolMeasure me = birch::BuildMeasure(rho, plus ? C.r_plus : C.r_minus, a_ell, alpha, rq.M, true);
    LFunctionForm form;
    form.separator = SeparatorText(rq.sep);
    form.target_root = C.target;
    form.eisenstein_congruent = C.eisenstein_congruent;
    form.a_p = C.a_p;
    form.sign = rq.sign;
    form.symbol_normalisation = plus ? "r^+ = the W^+ eigensymbol of the same system, r^+_0 = 1"
                                     : "r^- = the W^- eigensymbol, primitive, first ell-unit coordinate 1";
    return MakeLFunctionRecord(me, form, rq.branches);
}

} // namespace lvalue
} // namespace hecke_ext
