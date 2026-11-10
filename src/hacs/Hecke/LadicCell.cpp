// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 11, WP-18c) The digit pipeline route -- see
// LadicCell.h and [E] v1.12.0-pre.2 sec:ring.

#include "LadicCell.h"
#include "HeckeOrdinaryPart.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "CoefficientRing.h"
#include "ExactInt.h"
#include "HeckeCosetTables.h"
#include "HeckeOperatorFamily.h"   // tau_construction_mutex()
#include "HeckeTauTable.h"
#include "SL2ZTessellationTables.h"
#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/LadicInduction.h"

namespace hecke {
namespace ladic_cell {

using engine::u128;

// ===========================================================================
// The switch, the per-cell rule, residues
// ===========================================================================
Width EngineLadicWidth() {
    const char* v = std::getenv("ENGINE_LADIC_WIDTH");
    if (v == nullptr || *v == '\0') return Width::W64;
    const std::string s(v);
    if (s == "32") return Width::W32;
    if (s == "64") return Width::W64;
    if (s == "128") return Width::W128;
    throw std::invalid_argument("ENGINE_LADIC_WIDTH: expected 32, 64 or 128, got \"" + s +
                                "\" (an engine-family switch: it selects the digit word of the (C2) route "
                                "and cannot change a value)");
}

const char* to_text(CellRoute r) {
    switch (r) {
        case CellRoute::Native: return "native";
        case CellRoute::Digit: return "digit";
        case CellRoute::RefusedSmallEll: return "refused_small_ell";
        case CellRoute::RefusedBits: return "refused_bits";
    }
    return "?";
}

bool FitsU64(uint64_t ell, uint32_t k) {
    ENGINE_CHECK(ell >= 2);
    u128 acc = 1;
    for (uint32_t i = 0; i < k; ++i) {
        acc *= ell;
        if (acc >> 64) return false;
    }
    return true;
}

uint32_t CeilLog2(uint64_t ell) {
    ENGINE_CHECK(ell >= 2);
    uint32_t b = 0;
    while (b < 64 && (uint64_t(1) << b) < ell) ++b;
    return b;
}

uint32_t WordLevelMax(uint64_t ell) {
    uint32_t k = 1;
    while (FitsU64(ell, k + 1)) ++k;
    return k;
}

CellRoute ClassifyCell(uint64_t ell, uint32_t k) {
    if (FitsU64(ell, k)) return CellRoute::Native;
    if (ell == 2 || ell == 3) return CellRoute::RefusedSmallEll;
    if (static_cast<uint64_t>(k) * CeilLog2(ell) > kMaxBits) return CellRoute::RefusedBits;
    return CellRoute::Digit;
}

namespace {
uint32_t WidthBits(Width w) { return static_cast<uint32_t>(w); }
uint32_t DigitExp(uint64_t ell, Width w) {
    switch (w) {
        case Width::W32: return engine::ladic::DigitLayout<uint32_t>::max_digit_exp(ell);
        case Width::W64: return engine::ladic::DigitLayout<uint64_t>::max_digit_exp(ell);
        case Width::W128: return engine::ladic::DigitLayout<u128>::max_digit_exp(ell);
    }
    return 1;
}
std::string CellText(uint32_t n, uint64_t ell) {
    return "(n = " + std::to_string(n) + ", ell = " + std::to_string(ell) + ")";
}
} // namespace

double PredictedBytes(uint32_t n, uint64_t ell, uint32_t K, Width w) {
    // The dominant objects are the two dense Theta^{(1)} builds and the A_C1 /
    // elimination working copies: <= 8 dense (11 d_V)^2 matrices of L digits.
    const double dim = 11.0 * (static_cast<double>(n) + 1.0);
    const uint32_t m = DigitExp(ell, w);
    const double L = std::ceil(static_cast<double>(K) / m);
    return 8.0 * dim * dim * L * (WidthBits(w) / 8.0);
}

void RequireDigitCell(const std::string& who, uint32_t n, uint64_t ell, uint32_t K, Width w) {
    const std::string cell = CellText(n, ell);
    const CellRoute r = ClassifyCell(ell, K);
    if (r == CellRoute::RefusedSmallEll)
        throw NotImplementedInThisBuild(
            who + ": coefficient_ring=" + CoefficientRingText(K) + " is not honoured in this build at the cell " +
            cell + ": ell = " + std::to_string(ell) + " with ell^k >= 2^64 (ell in {2, 3}: C^q_eq is not free, the "
            "precision-split lemma does not apply; [E] v1.12.0-pre.2 sec:ring, the per-cell rule (iii))");
    if (r == CellRoute::RefusedBits)
        throw NotImplementedInThisBuild(
            who + ": coefficient_ring=" + CoefficientRingText(K) + " is not honoured in this build at the cell " +
            cell + ": the working precision needs " + std::to_string(static_cast<uint64_t>(K) * CeilLog2(ell)) +
            " bits > " + std::to_string(kMaxBits) + " ([E] v1.12.0-pre.2 sec:ring, the per-cell rule (iv))");
    if (r == CellRoute::Native)
        throw std::logic_error(who + ": the digit route was asked for a native cell " + cell + " at K = " +
                               std::to_string(K) + " (ell^K < 2^64 keeps the shipped route)");
    if (ell >= (uint64_t(1) << 31))
        throw NotImplementedInThisBuild(
            who + ": coefficient_ring=" + CoefficientRingText(K) + " is not honoured in this build at the cell " +
            cell + ": ell^k >= 2^64 with ell >= 2^31, outside the digit route's domain 2 <= ell < 2^31 "
            "(ENGINE_API.md Sec. 13.11.1; [E] v1.12.0-pre.2 sec:ring, the per-cell rule)");
    const double bytes = PredictedBytes(n, ell, K, w);
    if (bytes > kMaxPredictedBytes) {
        std::ostringstream o;
        o << who << ": coefficient_ring=" << CoefficientRingText(K) << " is not honoured in this build at the cell "
          << cell << ": the predicted working set is " << (bytes / (1024.0 * 1024.0 * 1024.0))
          << " GB > 40 GB ([E] v1.12.0-pre.2 sec:ring, the per-cell rule (iv); slot-11 rule 19)";
        throw NotImplementedInThisBuild(o.str());
    }
    if (n == 0 || n % 2 != 0)
        throw NotImplementedInThisBuild(
            who + ": coefficient_ring=" + CoefficientRingText(K) + " is not honoured in this build at the cell " +
            cell + ": the digit route needs the K-mode's range n > 0 even for its certified precision bound "
            "([E] sec:ring (d); v1.12.0-pre.2)");
}

void RefuseMember(const std::string& member, uint32_t n, uint64_t ell, uint32_t k) {
    RefuseCoefficientRing("LadicCell (" + member + ")", k,
                          " at the digit cell " + CellText(n, ell) +
                              " ([E] v1.12.0-pre.2 sec:ring: outside the digit route's member list in this build)");
}

std::string ResidueText(const Residue& r, uint64_t ell) {
    // value < 2^63 -> decimal; ell^j - value < 2^63 -> "-" decimal; else digits.
    auto fits = [&](const Residue& d, u128& out) {
        u128 v = 0;
        for (size_t i = d.size(); i-- > 0;) {
            if (v > ((~u128(0)) - d[i]) / ell) return false;
            v = v * ell + d[i];
            if (v >> 63) return false;
        }
        out = v;
        return true;
    };
    u128 v = 0;
    if (fits(r, v)) return std::to_string(static_cast<uint64_t>(v));
    // the negative: ell^j - x, digitwise (borrow)
    Residue neg(r.size());
    uint32_t borrow = 0;
    bool nonzero = false;
    for (size_t i = 0; i < r.size(); ++i) {
        const uint64_t need = static_cast<uint64_t>(r[i]) + borrow;
        if (need == 0) { neg[i] = 0; borrow = 0; }
        else { neg[i] = static_cast<uint32_t>((ell - need % ell) % ell); borrow = (need > 0) ? 1 : 0; nonzero = true; }
    }
    (void)nonzero;
    if (fits(neg, v)) return "-" + std::to_string(static_cast<uint64_t>(v));
    std::string s = "ell-adic:[";
    for (size_t i = 0; i < r.size(); ++i) s += (i ? "," : "") + std::to_string(r[i]);
    return s + "]";
}

Residue ResidueOfInt(int64_t v, uint64_t ell, uint32_t j) {
    // |v| in base ell, then negate digitwise if v < 0.
    uint64_t mag = v < 0 ? static_cast<uint64_t>(-(v + 1)) + 1u : static_cast<uint64_t>(v);
    Residue d(j, 0);
    for (uint32_t i = 0; i < j; ++i) { d[i] = static_cast<uint32_t>(mag % ell); mag /= ell; }
    if (v < 0) {
        uint32_t borrow = 0;
        for (uint32_t i = 0; i < j; ++i) {
            const uint64_t need = static_cast<uint64_t>(d[i]) + borrow;
            if (need == 0) { d[i] = 0; borrow = 0; }
            else { d[i] = static_cast<uint32_t>(ell - need); borrow = 1; }
        }
    }
    return d;
}

namespace {
template<class N>
std::vector<uint32_t> SerialiseNativeT(const engine::DenseBuf<N>& M, uint64_t ell, uint32_t K) {
    std::vector<uint32_t> out;
    out.reserve(static_cast<size_t>(M.rows) * M.cols * K + 2);
    out.push_back(M.rows);
    out.push_back(M.cols);
    for (uint32_t i = 0; i < M.rows; ++i)
        for (uint32_t j = 0; j < M.cols; ++j) {
            uint64_t v = static_cast<uint64_t>(M.at(i, j));
            for (uint32_t e = 0; e < K; ++e) { out.push_back(static_cast<uint32_t>(v % ell)); v /= ell; }
        }
    return out;
}
} // namespace
std::vector<uint32_t> SerialiseNative(const engine::DenseBuf<uint64_t>& M, uint64_t ell, uint32_t K) {
    return SerialiseNativeT(M, ell, K);
}
std::vector<uint32_t> SerialiseNative(const engine::DenseBuf<uint32_t>& M, uint64_t ell, uint32_t K) {
    return SerialiseNativeT(M, ell, K);
}

std::string CellRecord::canonical_text() const {
    std::ostringstream o;
    auto list = [&](const std::vector<uint32_t>& v) {
        o << "[";
        for (size_t i = 0; i < v.size(); ++i) o << (i ? "," : "") << v[i];
        o << "]";
    };
    auto res = [&](const std::vector<Residue>& v) {
        o << "[";
        for (size_t i = 0; i < v.size(); ++i) { if (i) o << ","; list(v[i]); }
        o << "]";
    };
    o << "cell n=" << n_sym << " ell=" << ell << " k=" << precision.requested << " K=" << precision.working
      << " e_max=" << precision.e_max << " word_level=" << precision.word_level << " certified=" << precision.certified
      << " word=" << precision.word_certified << " digit=" << precision.digit_certified << " uct=" << precision.uct_match;
    // (slot-12 WP-11, A8) the opt-in pre-pass route says so; appended ONLY when
    // set, so every default-route text (prepass = false) is byte-identical.
    if (precision.prepass) o << " prepass=1";
    o << "\n";
    for (size_t q = 0; q < degrees.size(); ++q) {
        const Degree& d = degrees[q];
        o << "H" << q << " C=" << d.rank_C << " invK=";
        list(d.invariant_factors_K);
        o << " dK=";
        list(d.dK);
        o << " invk=";
        list(d.invariant_factors_k);
        o << " lenK=" << d.length_K << " lenk=" << d.length_k << "\n";
    }
    o << "rK " << r_K0 << " " << r_K1 << " kmode=" << kmode_certified << "\n";
    for (size_t q = 0; q < type.size(); ++q) {
        o << "type" << q << " free=" << type[q].free_rank << " tors=";
        list(type[q].torsion);
        o << "\n";
    }
    for (const HeckeOut& h : hecke) {
        o << "T" << h.p << " uell=" << h.u_ell << " routes=" << h.routes_equal << " checks=" << h.relation_checks << "/"
          << h.cocycle_checks << "/" << h.theta_eq_solves << " r=" << h.rank_free << " prec=" << h.precision << " A=";
        res(h.free_block);
        o << " chi=";
        res(h.charpoly);
        o << "\n";
    }
    if (hida.present) {
        o << "hida prec=" << hida.precision << " r=" << hida.rank << " W=" << hida.W_exponent << " a=" << hida.nilpotent_dim
          << " rank_e=" << hida.rank_e_mod_ell << " ord=" << hida.ordinary_rank_mod_ell << " cert=" << hida.certified
          << " e=";
        res(hida.e);
        o << "\n";
    }
    auto objs = [&](const char* name, const std::vector<std::vector<uint32_t>>& v) {
        for (size_t i = 0; i < v.size(); ++i) { o << name << i << "="; list(v[i]); o << "\n"; }
    };
    objs("C", objects.C);
    objs("X", objects.X);
    objs("Zc", objects.Zc);
    objs("H", objects.H);
    objs("decB", objects.dec_basis);
    objs("decG", objects.dec_Gamma);
    objs("decU", objects.dec_U);
    objs("decE", objects.dec_exponents);
    objs("decP", objects.dec_position);
    for (size_t p = 0; p < objects.Theta.size(); ++p) objs(("Theta" + std::to_string(p) + "_").c_str(), objects.Theta[p]);
    for (size_t p = 0; p < objects.A.size(); ++p) objs(("A" + std::to_string(p) + "_").c_str(), objects.A[p]);
    return o.str();
}

// ===========================================================================
// The word-size K-mode (step 1): the SHIPPED native route at K0.
// ===========================================================================
namespace {

struct WordRun {
    uint32_t K0 = 0;
    bool certified = false;
    uint32_t r0 = 0, r1 = 0, need0 = 0, need1 = 0;
    std::vector<TypeQ> type;
};

template<class Elt>
WordRun RunWordLevelT(uint32_t n, uint64_t ell, uint32_t K0) {
    WordRun w;
    w.K0 = K0;
    const ScopedCoefficientRing ring(K0);
    const engine::ModCtx<Elt> ctx = CoefficientRingContext<Elt>("LadicCell (word-size K-mode)", ell, K0);
    std::unique_ptr<const SimplicialComplex> K;
    {
        std::lock_guard<std::mutex> lk(tau_construction_mutex());
        K.reset(new SimplicialComplex(CreateSL2ZFundamentalDomainTriangulation()));
    }
    LocalSystemSimplicialComplexT<Elt> lssc(*K, GenerateSL2ZRelationsRing<Elt>(*K, n, ctx), n + 1, ctx.p);
    EquivariantCohomologyOnKT<Elt> coh(&lssc);
    coh.compute_cohomology();
    ENGINE_CHECK_MSG(coh.ladic_applicable(), "LadicCell: the word-size K-mode is not applicable (n > 0 even)");
    w.r0 = coh.r_K(0);
    w.r1 = coh.r_K(1);
    w.need0 = coh.C_eq_rank(0);
    w.need1 = coh.C_eq_rank(2);
    w.certified = coh.ladic_type_certified();
    if (w.certified)
        for (uint32_t q = 0; q < coh.num_degrees(); ++q) {
            const auto t = coh.ladic_type(q);
            w.type.push_back(TypeQ{t.free_rank, t.torsion});
        }
    return w;
}

WordRun RunWordLevel(uint32_t n, uint64_t ell, uint32_t K0) {
    ENGINE_CHECK(FitsU64(ell, K0) && K0 >= 2);
    u128 a = 1;
    for (uint32_t i = 0; i < K0; ++i) a *= ell;
    // CoefficientRingContext's own element rule: uint32_t below 2^32.
    return a <= 0xffffffffull ? RunWordLevelT<uint32_t>(n, ell, K0) : RunWordLevelT<uint64_t>(n, ell, K0);
}

// [L] Sec. 2.5's observed bound e_max <= 1 + log_ell(n / (ell + 1)) -- the
// STARTING level only; the certificate decides.
uint32_t StartingWordLevel(uint32_t n, uint64_t ell) {
    int64_t b = 0;   // floor(1 + log_ell(n / (ell + 1))), by exact integer steps
    if (static_cast<uint64_t>(n) >= ell + 1) {
        b = 1;
        u128 t = ell + 1;
        while (t * ell <= n) { t *= ell; ++b; }
    }
    return static_cast<uint32_t>(std::max<int64_t>(2, b + 1));
}

// (slot-12 WP-11, A9) THE escalation policy of the word-size K-mode, shared by
// RunCell and CertifiedWordKMode ((Hi1)'s oracle): min(K_w, starting level),
// then K_w.
std::vector<uint32_t> DefaultWordLevels(uint32_t n, uint64_t ell) {
    const uint32_t Kw = WordLevelMax(ell);
    std::vector<uint32_t> levels{std::min(Kw, StartingWordLevel(n, ell))};
    if (levels.back() < Kw) levels.push_back(Kw);
    return levels;
}

// UCT: H^q(C (x) Z/ell^j) = (Z/ell^j)^{free_q} (+) tors H^q (+) tors H^{q+1},
// every certified exponent < the word level <= j.
std::vector<uint32_t> UctFactors(const std::vector<TypeQ>& t, uint32_t q, uint32_t j) {
    std::vector<uint32_t> e(t[q].free_rank, j);
    for (uint32_t a : t[q].torsion) e.push_back(std::min(a, j));
    if (q + 1 < t.size())
        for (uint32_t a : t[q + 1].torsion) e.push_back(std::min(a, j));
    std::sort(e.begin(), e.end(), [](uint32_t a, uint32_t b) { return a > b; });
    return e;
}

// ===========================================================================
// The digit route (steps 2 - 4), over a digit word W.
// ===========================================================================
template<class W>
class DigitCell {
public:
    using LM = engine::ladic::LadicMat<W>;
    using WK = engine::ladic::detail::Work<W>;
    using Vec = std::vector<W>;

    DigitCell(uint32_t n, uint64_t ell, uint32_t K)
        : lay_(engine::ladic::DigitLayout<W>::make(ell, K)), R_(lay_), L_(lay_.L), K_(K), ell_(ell), n_(n), d_(n + 1) {}

    const engine::ladic::DigitLayout<W>& layout() const { return lay_; }

    // ---- scalars -----------------------------------------------------------
    void set_i64(W* z, int64_t v) const {
        const uint64_t mag = v < 0 ? static_cast<uint64_t>(-(v + 1)) + 1u : static_cast<uint64_t>(v);
        R_.from_u64(z, mag);
        if (v < 0) R_.neg(z, z);
    }
    Vec vec_i64(int64_t v) const { Vec z(L_); set_i64(z.data(), v); return z; }
    Residue residue(const W* x, uint32_t precision) const {
        std::vector<uint32_t> a(K_);
        R_.to_ell_digits(a.data(), x);
        a.resize(precision);
        return a;
    }

    // ---- rho_n over Z/ell^K of an exact matrix (Sec. 13.10 F4, digits) -------
    WK rho(const Mat22& g) const {
        WK out(d_, d_, L_);
        rho_fill(vec_i64(g.a), vec_i64(g.b), vec_i64(g.c), vec_i64(g.d), out, 0);
        return out;
    }

    // ---- the relation list (the 7-vertex model; seeds' exact g_r and -I) -----
    struct Rel { uint32_t q, lhs, rhs; WK M; bool identity_self; };

    void build_model() {
        {
            std::lock_guard<std::mutex> lk(tau_construction_mutex());
            K_owned_.reset(new SimplicialComplex(CreateSL2ZFundamentalDomainTriangulation()));
            model_.reset(new SL2ZTessellationTables(*K_owned_));
        }
        const SimplicialComplex& Kc = *K_owned_;
        const uint32_t dimK = Kc.get_dimension();
        nq_ = (dimK == static_cast<uint32_t>(-1)) ? 0 : dimK + 1;
        ENGINE_CHECK_MSG(nq_ == 3, "LadicCell: the 7-vertex model has three degrees");
        std::vector<SL2ZRelationDatum> data;
        {
            const ScopedCoefficientRing field(1);   // the F_ell generator reads the ring; the row structure only
            data = GenerateSL2ZRelationsWithGroupData(Kc, n_, static_cast<uint32_t>(ell_));
        }
        std::vector<std::pair<Mat22, WK>> cache;
        for (const SL2ZRelationDatum& dd : data) {
            const WK* M = nullptr;
            for (const auto& e : cache) if (e.first == dd.g_r) M = &e.second;
            if (M == nullptr) { cache.emplace_back(dd.g_r, rho(dd.g_r)); M = &cache.back().second; }
            Rel r{dd.rel.q_dim, dd.rel.s_lhs_idx_in_Kq, dd.rel.s_rhs_idx_in_Kq, *M, false};
            if (r.lhs == r.rhs) r.identity_self = is_identity(r.M);
            rels_.push_back(std::move(r));
        }
        // integer incidences of the simplicial coboundary (+-1 / 0)
        const uint32_t P = 2147483647u;
        inc_.assign(nq_, {});
        for (uint32_t q = 0; q + 1 < nq_; ++q) {
            const FlatMatrixT<uint32_t> D = Kc.template coboundary_operator_elt<uint32_t>(q, P);
            std::vector<std::vector<int>>& I = inc_[q];
            I.assign(D.get_rows(), std::vector<int>(D.get_cols(), 0));
            for (uint32_t i = 0; i < D.get_rows(); ++i)
                for (uint32_t j = 0; j < D.get_cols(); ++j) {
                    const uint32_t v = D.get_entry(i, j);
                    ENGINE_CHECK_MSG(v == 0 || v == 1 || v == P - 1, "LadicCell: a coboundary incidence is not +-1");
                    I[i][j] = (v == 0) ? 0 : (v == 1 ? 1 : -1);
                }
        }
    }

    // ---- step 2: the cohomology --------------------------------------------
    struct DegreeData {
        LM C, X, Zc, H;
        engine::ladic::Cyclic<W> dec;
        std::vector<uint32_t> dK;
        uint32_t length = 0;
    };
    std::vector<DegreeData> deg_;

    void compute_cohomology() {
        const SimplicialComplex& Kc = *K_owned_;
        deg_.assign(nq_, DegreeData());
        for (uint32_t q = 0; q < nq_; ++q) {
            const uint32_t N = Kc.get_num_q_simplices(q), dim = N * d_;
            // A_Cq: one block row per non-vacuous relation.
            std::vector<const Rel*> live;
            for (const Rel& r : rels_) if (r.q == q && !r.identity_self) live.push_back(&r);
            WK A(static_cast<uint32_t>(live.size()) * d_, dim, L_);
            Vec t(L_);
            for (size_t b = 0; b < live.size(); ++b) {
                const Rel& r = *live[b];
                for (uint32_t a = 0; a < d_; ++a) {
                    const uint32_t row = static_cast<uint32_t>(b) * d_ + a;
                    for (uint32_t c = 0; c < d_; ++c) {
                        R_.neg(t.data(), r.M.at(a, c));                            // -M
                        W* dst = A.at(row, r.rhs * d_ + c);
                        R_.add(dst, dst, t.data());
                    }
                    W* diag = A.at(row, r.lhs * d_ + a);                          // + I
                    Vec one(L_);
                    R_.one(one.data());
                    R_.add(diag, diag, one.data());
                }
            }
            const LM Am = engine::ladic::detail::from_work(lay_, A);
            DegreeData& D = deg_[q];
            if (dim == 0) { D.C = LM(lay_, 0, 0); continue; }
            if (live.empty()) {
                D.C = identity(dim);
                continue;
            }
            const engine::ladic::Howell<W> Kh = engine::ladic::kernel_module(Am);
            for (uint32_t v : Kh.pivot_v)
                ENGINE_CHECK_MSG(v == 0, "LadicCell: C^q_eq is not free -- a Howell pivot of ker A_Cq is not a unit "
                                         "([C, prop:coneintegral](2) needs ell >= 5)");
            D.C = Kh.H.transpose();
        }
        // X_q: d^q in C_eq coordinates.
        for (uint32_t q = 0; q < nq_; ++q) {
            DegreeData& D = deg_[q];
            const uint32_t cq = D.C.cols();
            if (q + 1 >= nq_) { D.X = LM(lay_, 0, cq); continue; }
            const LM DC = apply_coboundary(q, D.C);
            const LM& C1 = deg_[q + 1].C;
            if (cq > 0 && C1.cols() > 0) {
                D.X = engine::ladic::solve_multi_rhs_module(C1, DC).transpose();   // c_{q+1} x c_q
                // the exact residual (WP-14 pass 2 m-5): C_{q+1} X_q == D C_q
                ENGINE_CHECK_MSG(engine::ladic::multiply(C1, D.X) == DC,
                                 "LadicCell: the X_q residual C_{q+1} X_q - D C_q is not zero");
            } else {
                D.X = LM(lay_, C1.cols(), cq);
                ENGINE_CHECK_MSG(is_zero(DC), "LadicCell: d C^q_eq not in C^{q+1}_eq = 0");
            }
        }
        for (uint32_t q = 1; q + 1 < nq_; ++q)
            if (deg_[q].X.rows() > 0 && deg_[q - 1].X.cols() > 0)
                ENGINE_CHECK_MSG(is_zero(engine::ladic::multiply(deg_[q].X, deg_[q - 1].X)), "LadicCell: d o d != 0");
        for (uint32_t q = 0; q < nq_; ++q) {
            DegreeData& D = deg_[q];
            const uint32_t cq = D.C.cols();
            if (cq == 0) {
                D.Zc = LM(lay_, 0, 0);
                D.H = LM(lay_, D.C.rows(), 0);
                continue;
            }
            if (D.X.rows() == 0) D.Zc = identity(cq);
            else D.Zc = engine::ladic::kernel_module(D.X).H.transpose();
            const LM none(lay_, cq, 0);
            const LM& Rl = (q > 0) ? deg_[q - 1].X : none;
            if (D.Zc.cols() == 0) {
                ENGINE_CHECK_MSG(Rl.cols() == 0 || is_zero(Rl), "LadicCell: B^q not in Z^q = 0");
                D.dec.basis = LM(lay_, cq, 0);
                D.dec.Gamma = LM(lay_, cq, 0);
                D.dec.U = LM(lay_, 0, 0);
            } else {
                D.dec = engine::ladic::cyclic_decomposition(D.Zc, Rl);
            }
            D.length = 0;
            for (uint32_t e : D.dec.exponents) D.length += e;
            D.H = D.dec.rows() > 0 ? engine::ladic::multiply(D.C, D.dec.basis) : LM(lay_, D.C.rows(), 0);
            if (D.X.rows() > 0) D.dK = engine::ladic::invariant_factors(D.X);
        }
    }

    // ---- step 3: one Hecke operator ------------------------------------------
    struct HeckeData {
        std::vector<LM> Theta, A;
        HeckeOut out;
        LM free_block;   // r x r over Z/ell^K
    };

    HeckeData hecke(uint64_t p) {
        HeckeData H;
        H.out.p = p;
        H.out.u_ell = (p == ell_);
        std::unique_ptr<HeckeCosetTablesT<uint32_t>> cosets;
        std::unique_ptr<HeckeTauTableT<uint32_t>> tau;
        {
            const ScopedCoefficientRing field(1);   // the tables are keyed on ell; only their exact data is read
            std::lock_guard<std::mutex> lk(tau_construction_mutex());
            cosets.reset(new HeckeCosetTablesT<uint32_t>(p, 1, static_cast<uint32_t>(ell_), n_));
            tau.reset(new HeckeTauTableT<uint32_t>(*K_owned_, *model_, *cosets, static_cast<uint32_t>(ell_), n_));
        }
        if (tau->used_mod_ell_solve())
            throw NotImplementedInThisBuild("LadicCell: the tau datum at " + CellText(n_, ell_) + ", p = " +
                                            std::to_string(p) + " used the mod-ell triangle fallback, so it is not "
                                            "integral ([E] sec:ring (b3))");
        const uint32_t m = cosets->num_cosets();
        std::vector<WK> omega(m + 1);
        for (uint32_t i = 1; i <= m; ++i) omega[i] = rho(cosets->s_inv(i));
        std::map<std::vector<W>, WK> rho_memo;
        auto rho_m = [&](const Mat22& g) -> const WK& {
            std::vector<W> key(4 * L_);
            set_i64(key.data(), g.a);
            set_i64(key.data() + L_, g.b);
            set_i64(key.data() + 2 * L_, g.c);
            set_i64(key.data() + 3 * L_, g.d);
            auto it = rho_memo.find(key);
            if (it == rho_memo.end()) it = rho_memo.emplace(key, rho(g)).first;
            return it->second;
        };
        std::map<std::pair<uint32_t, std::vector<W>>, WK> fused_memo;
        bool routes_equal = true;
        const SimplicialComplex& Kc = *K_owned_;
        Vec c(L_), tmp(L_);
        for (uint32_t q = 0; q < nq_; ++q) {
            const uint32_t N = Kc.get_num_q_simplices(q), dim = N * d_;
            WK T(dim, dim, L_), T2(dim, dim, L_);
            auto add_block = [&](WK& dst, uint32_t sigma, uint32_t sk, const W* cc, const WK& B) {
                for (uint32_t a = 0; a < d_; ++a)
                    for (uint32_t b = 0; b < d_; ++b) {
                        const W* v = B.at(a, b);
                        if (R_.is_zero(v)) continue;
                        R_.addmul(dst.at(sigma * d_ + a, sk * d_ + b), cc, v);
                    }
            };
            for (uint32_t sigma = 0; sigma < N; ++sigma)
                for (uint32_t i = 1; i <= m; ++i)
                    for (const ChainTerm& ct : tau->tau(i, q, sigma)) {
                        set_i64(c.data(), ct.coeff);
                        if (R_.is_zero(c.data())) continue;
                        ENGINE_CHECK_MSG(ct.sigma_idx < N, "LadicCell: tau term names a simplex out of range");
                        const Mat22 gg = cosets->g() * ct.gamma;              // exact (I6); never inverted (I7)
                        std::vector<W> key(4 * L_);
                        set_i64(key.data(), gg.a);
                        set_i64(key.data() + L_, gg.b);
                        set_i64(key.data() + 2 * L_, gg.c);
                        set_i64(key.data() + 3 * L_, gg.d);
                        const auto fk = std::make_pair(i, key);
                        auto it = fused_memo.find(fk);
                        if (it == fused_memo.end()) it = fused_memo.emplace(fk, matmul(omega[i], rho_m(gg))).first;
                        add_block(T, sigma, ct.sigma_idx, c.data(), it->second);
                        const Mat22 dg = cosets->delta(i) * ct.gamma;         // route 2: the exact product first
                        add_block(T2, sigma, ct.sigma_idx, c.data(), rho_m(dg));
                    }
            if (T.d != T2.d) routes_equal = false;
            if (!routes_equal)
                throw std::runtime_error("LadicCell: the fused route and route 2 differ at q=" + std::to_string(q) +
                                         " (O12(a) over Z/ell^K) at " + CellText(n_, ell_));
            const LM Tm = engine::ladic::detail::from_work(lay_, T);
            const DegreeData& D = deg_[q];
            const uint32_t t = D.H.cols();
            const LM Y = t > 0 ? engine::ladic::multiply(Tm, D.H) : LM(lay_, dim, 0);
            // (i) every relation of degree q holds on Theta z
            for (const Rel& r : rels_) {
                if (r.q != q) continue;
                for (uint32_t col = 0; col < t; ++col) {
                    for (uint32_t a = 0; a < d_; ++a) {
                        Y.get(r.lhs * d_ + a, col, tmp.data());
                        Vec y(L_);
                        for (uint32_t b = 0; b < d_; ++b) {
                            const W* mv = r.M.at(a, b);
                            if (R_.is_zero(mv)) continue;
                            Y.get(r.rhs * d_ + b, col, y.data());
                            R_.submul(tmp.data(), mv, y.data());
                        }
                        if (!R_.is_zero(tmp.data()))
                            throw std::runtime_error("LadicCell: runtime assertion (i) failed -- Theta z violates a "
                                                     "relation at " + CellText(n_, ell_));
                    }
                    ++H.out.relation_checks;
                }
            }
            // (ii) d(Theta z) = 0
            if (q + 1 < nq_) {
                if (t > 0) ENGINE_CHECK_MSG(is_zero(apply_coboundary(q, Y)),
                                            "LadicCell: runtime assertion (ii) failed -- d(Theta z) != 0");
            }
            H.out.cocycle_checks += t;
            // (iii) Theta C^q_eq in C^q_eq, the exact residual
            const uint32_t cq = D.C.cols();
            if (cq > 0) {
                const LM Wm = engine::ladic::multiply(Tm, D.C);
                LM TE;
                try {
                    TE = engine::ladic::solve_multi_rhs_module(D.C, Wm).transpose();
                } catch (const std::runtime_error& e) {
                    throw std::runtime_error(std::string("LadicCell: runtime assertion (iii) failed -- ") + e.what());
                }
                if (!(engine::ladic::multiply(D.C, TE) == Wm))
                    throw std::runtime_error("LadicCell: the Theta_eq residual is not identically zero");
                H.out.theta_eq_solves += cq;
            }
            // A^{(q)} on the cyclic summands (c5)
            LM A(lay_, t, t);
            if (t > 0) {
                const LM Yc = engine::ladic::solve_multi_rhs_module(D.C, Y).transpose();   // cq x t
                const LM co = engine::ladic::cyclic_coordinates(D.dec, Yc);                 // t x t, row = column
                A = co.transpose();
            }
            H.Theta.push_back(Tm);
            H.A.push_back(A);
        }
        H.out.routes_equal = routes_equal;
        return H;
    }

    // The free block of A^{(1)} and its division-free charpoly.
    void free_part(HeckeData& H, uint32_t precision) const {
        const DegreeData& D = deg_[1];
        uint32_t r = 0;
        while (r < D.dec.exponents.size() && D.dec.exponents[r] == K_) ++r;
        H.out.rank_free = r;
        H.out.precision = precision;
        H.free_block = LM(lay_, r, r);
        if (precision == 0) return;
        Vec x(L_);
        for (uint32_t i = 0; i < r; ++i)
            for (uint32_t j = 0; j < r; ++j) {
                H.A[1].get(i, j, x.data());
                H.free_block.set(i, j, x.data());
                H.out.free_block.push_back(residue(x.data(), precision));
            }
        const std::vector<Vec> chi = charpoly(R_, H.free_block);
        for (const Vec& cc : chi) H.out.charpoly.push_back(residue(cc.data(), precision));
    }

    // ---- step 4: the ordinary idempotent over Z/ell^k -----------------------
    HidaOut hida(const HeckeData& U, const std::vector<const HeckeData*>& others, uint32_t k) const {
        HidaOut h;
        h.present = true;
        h.precision = k;
        const uint32_t r = U.free_block.rows();
        h.rank = r;
        if (r == 0) {
            h.idempotent = h.commutes = h.complement_nilpotent = h.rank_match = h.certified = true;
            return h;
        }
        const LM Uk = U.free_block.truncate(k);
        const engine::ladic::Ring<W> Rk(Uk.layout());
        const uint32_t Lk = Uk.layout().L;
        h.W_exponent = r * k;
        LM Wm = mpow(Uk, h.W_exponent);
        const std::vector<Vec> chi = charpoly(Rk, Wm);   // r + 1 coefficients, low -> high
        uint32_t a = 0;
        while (a < r && Rk.valuation(chi[a].data()) != 0) {
            ENGINE_CHECK_MSG(Rk.is_zero(chi[a].data()),
                             "LadicCell (hida): chi_W has a non-zero non-unit coefficient below its first unit one "
                             "(Fitting fails -- impossible for W = U^{rk})");
            ++a;
        }
        h.nilpotent_dim = a;
        ENGINE_CHECK_MSG(Rk.valuation(chi[a].data()) == 0, "LadicCell (hida): p(0) is not a unit");
        Vec p0inv(Lk);
        Rk.inv(p0inv.data(), chi[a].data());
        // Q = sum_{j >= 1} c_{a+j} W^j  (Horner), e = -p0^{-1} Q
        LM Q(Uk.layout(), r, r);
        for (uint32_t j = r - a; j >= 1; --j) {
            LM S = Q;
            Vec x(Lk);
            for (uint32_t i = 0; i < r; ++i) {
                S.get(i, i, x.data());
                Rk.add(x.data(), x.data(), chi[a + j].data());
                S.set(i, i, x.data());
            }
            Q = engine::ladic::multiply(S, Wm);
        }
        LM e(Uk.layout(), r, r);
        Vec x(Lk), y(Lk);
        for (uint32_t i = 0; i < r; ++i)
            for (uint32_t jj = 0; jj < r; ++jj) {
                Q.get(i, jj, x.data());
                Rk.mul(y.data(), x.data(), p0inv.data());
                Rk.neg(y.data(), y.data());
                e.set(i, jj, y.data());
                std::vector<uint32_t> dg = e.ell_digits(i, jj);
                h.e.push_back(dg);
            }
        h.idempotent = (engine::ladic::multiply(e, e) == e);
        h.commutes = true;
        for (const HeckeData* o : others) {
            if (o->free_block.rows() != r) { h.commutes = false; continue; }
            const LM Tk = o->free_block.truncate(k);
            if (!(engine::ladic::multiply(e, Tk) == engine::ladic::multiply(Tk, e))) h.commutes = false;
        }
        if (!(engine::ladic::multiply(e, Uk) == engine::ladic::multiply(Uk, e))) h.commutes = false;
        // (U (1 - e))^r = 0 mod ell
        LM ImE(Uk.layout(), r, r);
        for (uint32_t i = 0; i < r; ++i)
            for (uint32_t jj = 0; jj < r; ++jj) {
                e.get(i, jj, x.data());
                Rk.neg(y.data(), x.data());
                if (i == jj) { Vec one(Lk); Rk.one(one.data()); Rk.add(y.data(), y.data(), one.data()); }
                ImE.set(i, jj, y.data());
            }
        const LM N = mpow(engine::ladic::multiply(Uk, ImE), r);
        h.complement_nilpotent = true;
        for (uint32_t i = 0; i < r; ++i)
            for (uint32_t jj = 0; jj < r; ++jj)
                if (N.ell_digits(i, jj)[0] != 0) h.complement_nilpotent = false;
        h.rank_e_mod_ell = rank_mod_ell(e);
        h.ordinary_rank_mod_ell = rank_mod_ell(mpow(Uk, r));
        h.rank_match = (h.rank_e_mod_ell == h.ordinary_rank_mod_ell);
        h.certified = h.idempotent && h.commutes && h.complement_nilpotent && h.rank_match;
        return h;
    }

    // (slot-15 WP-5) the Hecke matrices on the cyclic summands and the summand exponents only.
    void hecke_objects(Objects& ob, const std::vector<HeckeData>& hk) const {
        for (const DegreeData& D : deg_) ob.dec_exponents.push_back(D.dec.exponents);
        for (const HeckeData& H : hk) {
            std::vector<std::vector<uint32_t>> a;
            for (const LM& t : H.A) a.push_back(t.serialise());
            ob.A.push_back(a);
        }
    }

    void objects(Objects& ob, const std::vector<HeckeData>& hk) const {
        for (const DegreeData& D : deg_) {
            ob.C.push_back(D.C.serialise());
            ob.X.push_back(D.X.serialise());
            ob.Zc.push_back(D.Zc.serialise());
            ob.H.push_back(D.H.serialise());
            ob.dec_basis.push_back(D.dec.basis.serialise());
            ob.dec_Gamma.push_back(D.dec.Gamma.serialise());
            ob.dec_U.push_back(D.dec.U.serialise());
            ob.dec_exponents.push_back(D.dec.exponents);
            ob.dec_position.push_back(D.dec.position);
        }
        for (const HeckeData& H : hk) {
            std::vector<std::vector<uint32_t>> th, a;
            for (const LM& t : H.Theta) th.push_back(t.serialise());
            for (const LM& t : H.A) a.push_back(t.serialise());
            ob.Theta.push_back(th);
            ob.A.push_back(a);
        }
    }

    uint32_t nq() const { return nq_; }

private:
    const engine::ladic::DigitLayout<W> lay_;
    const engine::ladic::Ring<W> R_;
    const uint32_t L_, K_;
    const uint64_t ell_;
    const uint32_t n_, d_;
    uint32_t nq_ = 0;
    std::unique_ptr<const SimplicialComplex> K_owned_;
    std::unique_ptr<const SL2ZTessellationTables> model_;
    std::vector<Rel> rels_;
    std::vector<std::vector<std::vector<int>>> inc_;

    bool is_identity(const WK& M) const {
        for (uint32_t a = 0; a < d_; ++a)
            for (uint32_t b = 0; b < d_; ++b) {
                const W* v = M.at(a, b);
                if (a == b ? !R_.is_one(v) : !R_.is_zero(v)) return false;
            }
        return true;
    }
    bool is_zero(const LM& M) const {
        for (uint32_t dg = 0; dg < lay_.L; ++dg) {
            const W* p = M.plane(dg);
            for (size_t i = 0; i < static_cast<size_t>(M.rows()) * M.cols(); ++i) if (p[i]) return false;
        }
        return true;
    }
    LM identity(uint32_t n) const {
        LM I(lay_, n, n);
        Vec one(L_);
        R_.one(one.data());
        for (uint32_t i = 0; i < n; ++i) I.set(i, i, one.data());
        return I;
    }
    // (D^q (x) I_d) M for a dim_q x c multi-vector M.
    LM apply_coboundary(uint32_t q, const LM& M) const {
        const std::vector<std::vector<int>>& I = inc_[q];
        const uint32_t N1 = static_cast<uint32_t>(I.size());
        LM out(lay_, N1 * d_, M.cols());
        Vec acc(L_), x(L_);
        for (uint32_t s1 = 0; s1 < N1; ++s1)
            for (uint32_t a = 0; a < d_; ++a)
                for (uint32_t col = 0; col < M.cols(); ++col) {
                    R_.zero(acc.data());
                    for (uint32_t s0 = 0; s0 < I[s1].size(); ++s0) {
                        const int v = I[s1][s0];
                        if (v == 0) continue;
                        M.get(s0 * d_ + a, col, x.data());
                        if (v > 0) R_.add(acc.data(), acc.data(), x.data());
                        else R_.sub(acc.data(), acc.data(), x.data());
                    }
                    out.set(s1 * d_ + a, col, acc.data());
                }
        return out;
    }
    WK matmul(const WK& A, const WK& B) const {
        WK C(A.rows, B.cols, L_);
        for (uint32_t i = 0; i < A.rows; ++i)
            for (uint32_t k = 0; k < A.cols; ++k) {
                const W* a = A.at(i, k);
                if (R_.is_zero(a)) continue;
                for (uint32_t j = 0; j < B.cols; ++j) {
                    const W* b = B.at(k, j);
                    if (!R_.is_zero(b)) R_.addmul(C.at(i, j), a, b);
                }
            }
        return C;
    }
    static LM mpow(const LM& A, uint32_t e) {
        const uint32_t r = A.rows();
        LM result(A.layout(), r, r);
        {
            const engine::ladic::Ring<W> Rr(A.layout());
            std::vector<W> one(A.layout().L);
            Rr.one(one.data());
            for (uint32_t i = 0; i < r; ++i) result.set(i, i, one.data());
        }
        LM base = A;
        while (e > 0) {
            if (e & 1u) result = engine::ladic::multiply(result, base);
            e >>= 1;
            if (e) base = engine::ladic::multiply(base, base);
        }
        return result;
    }
    uint32_t rank_mod_ell(const LM& A) const {
        const uint32_t r = A.rows(), c = A.cols();
        std::vector<std::vector<uint64_t>> M(r, std::vector<uint64_t>(c));
        for (uint32_t i = 0; i < r; ++i)
            for (uint32_t j = 0; j < c; ++j) M[i][j] = A.ell_digits(i, j)[0];
        uint32_t rank = 0;
        for (uint32_t col = 0; col < c && rank < r; ++col) {
            uint32_t piv = rank;
            while (piv < r && M[piv][col] == 0) ++piv;
            if (piv == r) continue;
            std::swap(M[piv], M[rank]);
            const uint64_t inv = engine::ladic::Ring<W>::inv_mod_small(M[rank][col], ell_);
            for (uint32_t i = 0; i < r; ++i) {
                if (i == rank || M[i][col] == 0) continue;
                const uint64_t f = (M[i][col] * inv) % ell_;
                for (uint32_t j = col; j < c; ++j) M[i][j] = (M[i][j] + (ell_ - f) * M[rank][j]) % ell_;
            }
            ++rank;
        }
        return rank;
    }
    // Berkowitz (division-free), det(xI - A), low -> high -- the shipped
    // charpoly_division_free, in digits.
    static std::vector<Vec> charpoly(const engine::ladic::Ring<W>& Rr, const LM& Am) {
        const uint32_t n = Am.rows(), Ld = Am.layout().L;
        auto A = [&](uint32_t i, uint32_t j) { Vec v(Ld); Am.get(i, j, v.data()); return v; };
        std::vector<Vec> pc(1, Vec(Ld));
        Rr.one(pc[0].data());
        Vec acc(Ld);
        for (uint32_t r = 1; r <= n; ++r) {
            const uint32_t m = r - 1;
            std::vector<Vec> t(r + 1, Vec(Ld));
            Rr.one(t[0].data());
            Rr.neg(t[1].data(), A(m, m).data());
            std::vector<Vec> v(m, Vec(Ld));
            for (uint32_t i = 0; i < m; ++i) v[i] = A(i, m);
            for (uint32_t kk = 2; kk <= r; ++kk) {
                Rr.zero(acc.data());
                for (uint32_t i = 0; i < m; ++i) Rr.addmul(acc.data(), A(m, i).data(), v[i].data());
                Rr.neg(t[kk].data(), acc.data());
                std::vector<Vec> nv(m, Vec(Ld));
                for (uint32_t i = 0; i < m; ++i)
                    for (uint32_t j = 0; j < m; ++j) Rr.addmul(nv[i].data(), A(i, j).data(), v[j].data());
                v.swap(nv);
            }
            std::vector<Vec> np(r + 1, Vec(Ld));
            for (uint32_t i = 0; i <= r; ++i)
                for (uint32_t j = 0; j < r && j <= i; ++j) Rr.addmul(np[i].data(), t[i - j].data(), pc[j].data());
            pc.swap(np);
        }
        std::reverse(pc.begin(), pc.end());
        return pc;
    }

    void rho_fill(const Vec& a, const Vec& b, const Vec& c, const Vec& dd, WK& out, uint32_t depth) const {
        const uint32_t n = n_, dV = d_;
        const bool a_unit = R_.valuation(a.data()) == 0, b_unit = R_.valuation(b.data()) == 0;
        if (a_unit || b_unit) {
            // [U_a]: P_0 = (a + c y)^n, P_{j+1} = P_j (b + d y) / (a + c y); [U_b] mirrored.
            const Vec& u = a_unit ? a : b;
            const Vec& w = a_unit ? c : dd;
            const Vec& s = a_unit ? b : a;
            const Vec& t = a_unit ? dd : c;
            Vec uinv(L_);
            R_.inv(uinv.data(), u.data());
            std::vector<Vec> bin(dV, Vec(L_));      // Pascal row n by additions
            R_.one(bin[0].data());
            for (uint32_t r = 1; r <= n; ++r)
                for (uint32_t j = r; j >= 1; --j) R_.add(bin[j].data(), bin[j].data(), bin[j - 1].data());
            std::vector<Vec> pu(dV, Vec(L_)), pw(dV, Vec(L_));
            R_.one(pu[0].data());
            R_.one(pw[0].data());
            for (uint32_t kk = 1; kk < dV; ++kk) {
                R_.mul(pu[kk].data(), pu[kk - 1].data(), u.data());
                R_.mul(pw[kk].data(), pw[kk - 1].data(), w.data());
            }
            std::vector<Vec> cur(dV, Vec(L_)), nxt(dV, Vec(L_));
            Vec q(L_), rr(L_), tmp(L_), r_prev(L_), cur_prev(L_);
            for (uint32_t kk = 0; kk < dV; ++kk) {
                R_.mul(tmp.data(), pu[n - kk].data(), pw[kk].data());
                R_.mul(cur[kk].data(), bin[kk].data(), tmp.data());
            }
            const uint32_t j0 = a_unit ? 0 : n;
            for (uint32_t kk = 0; kk < dV; ++kk) R_.copy(out.at(kk, j0), cur[kk].data());
            for (uint32_t step = 0; step < n; ++step) {
                R_.zero(r_prev.data());
                R_.zero(cur_prev.data());
                for (uint32_t kk = 0; kk < dV; ++kk) {
                    R_.mul(q.data(), s.data(), cur[kk].data());
                    R_.addmul(q.data(), t.data(), cur_prev.data());
                    R_.submul(q.data(), w.data(), r_prev.data());
                    R_.mul(rr.data(), q.data(), uinv.data());
                    R_.copy(nxt[kk].data(), rr.data());
                    R_.copy(cur_prev.data(), cur[kk].data());
                    R_.copy(r_prev.data(), rr.data());
                }
                cur.swap(nxt);
                const uint32_t j = a_unit ? step + 1 : n - step - 1;
                for (uint32_t kk = 0; kk < dV; ++kk) R_.copy(out.at(kk, j), cur[kk].data());
            }
            return;
        }
        if (R_.is_zero(a.data()) && R_.is_zero(b.data())) {
            // [Z] the first row vanishes: column j is c^(n-j) d^j y^n.
            std::vector<Vec> pc(dV, Vec(L_)), pd(dV, Vec(L_));
            R_.one(pc[0].data());
            R_.one(pd[0].data());
            for (uint32_t kk = 1; kk < dV; ++kk) {
                R_.mul(pc[kk].data(), pc[kk - 1].data(), c.data());
                R_.mul(pd[kk].data(), pd[kk - 1].data(), dd.data());
            }
            for (uint32_t j = 0; j < dV; ++j) R_.mul(out.at(n, j), pc[n - j].data(), pd[j].data());
            return;
        }
        // [S] ell | a, ell | b, not both zero: g = diag(ell, 1) g'.
        ENGINE_CHECK_MSG(depth < K_, "LadicCell: rho_n split recursion deeper than K");
        Vec a1(L_), b1(L_);
        R_.shr(a1.data(), a.data(), 1);
        R_.shr(b1.data(), b.data(), 1);
        rho_fill(a1, b1, c, dd, out, depth + 1);
        for (uint32_t kk = 0; kk < dV; ++kk) {
            const uint32_t v = n - kk;
            if (v == 0) continue;
            for (uint32_t j = 0; j < dV; ++j) R_.shl(out.at(kk, j), out.at(kk, j), v);
        }
    }
};

// (slot-18 WP-19b) RunDigit split at the Hecke step: PrepareDigit is step 2
// and the certificates (prime-independent), HeckeDigit steps 3-4 for a prime
// list.  The bodies are the slot-11 RunDigit's, verbatim, cut in two.
template<class W>
void PrepareDigit(DigitCell<W>& cell, const Request& req, CellRecord& rec, const std::vector<TypeQ>& type) {
    const uint32_t K = rec.precision.working, k = req.k;
    rec.precision.digit_width = static_cast<uint32_t>(sizeof(W) * 8);
    rec.precision.digits_L = cell.layout().L;
    rec.precision.digit_m = cell.layout().m;
    cell.compute_cohomology();
    // the K-mode at K (the second certificate) and the UCT check
    const auto& D = cell.deg_;
    rec.r_K0 = static_cast<uint32_t>(D[0].dK.size());
    rec.r_K1 = static_cast<uint32_t>(D[1].dK.size());
    rec.kmode_certified = (rec.r_K0 == D[0].C.cols()) && (rec.r_K1 == D[2].C.cols());
    rec.precision.digit_certified = rec.kmode_certified;
    // UCT holds at EVERY K (tors Z/ell^a (x) Z/ell^K = Tor(Z/ell^a, Z/ell^K) = Z/ell^min(a,K));
    // the certificates are separate.
    bool uct = !type.empty();
    for (uint32_t q = 0; q < cell.nq(); ++q) {
        Degree dg;
        dg.rank_C = D[q].C.cols();
        dg.invariant_factors_K = D[q].dec.exponents;
        dg.length_K = D[q].length;
        dg.dK = D[q].dK;
        if (!type.empty()) {
            if (UctFactors(type, q, K) != dg.invariant_factors_K) uct = false;
            dg.invariant_factors_k = UctFactors(type, q, k);
            for (uint32_t e : dg.invariant_factors_k) dg.length_k += e;
        }
        rec.degrees.push_back(dg);
    }
    // The digit run's own type (shipped ladic_type's formula) must be the word type.
    if (rec.kmode_certified && !type.empty()) {
        for (uint32_t q = 0; q < cell.nq(); ++q) {
            const uint32_t r_here = static_cast<uint32_t>(D[q].dK.size());
            const uint32_t r_prev = q > 0 ? static_cast<uint32_t>(D[q - 1].dK.size()) : 0u;
            TypeQ t;
            t.free_rank = D[q].C.cols() - r_here - r_prev;
            if (q > 0) for (uint32_t e : D[q - 1].dK) if (e < K) t.torsion.push_back(K - e);
            std::sort(t.torsion.begin(), t.torsion.end(), [](uint32_t a, uint32_t b) { return a > b; });
            if (t.free_rank != type[q].free_rank || t.torsion != type[q].torsion) uct = false;
        }
    }
    rec.precision.uct_match = uct;
    rec.precision.certified = (rec.precision.word_certified || rec.precision.prepass) && rec.precision.digit_certified &&
                              rec.precision.uct_match;
    rec.type = type;
}

template<class W>
void HeckeDigit(DigitCell<W>& cell, const Request& req, const std::vector<uint64_t>& primes, CellRecord& rec) {
    const uint32_t K = rec.precision.working;
    // Hecke
    // (the fixed_K hook may run at K <= e_max: then no free datum has a precision)
    const uint32_t precision = K > rec.precision.e_max ? K - rec.precision.e_max : 0u;
    std::vector<typename DigitCell<W>::HeckeData> hk;
    const auto hecke_t0 = std::chrono::steady_clock::now();   // (S13 WP-12) the Hecke + Hida share, timing only
    for (uint64_t p : primes) {
        hk.push_back(cell.hecke(p));
        cell.free_part(hk.back(), precision);
        rec.hecke.push_back(hk.back().out);
    }
    if (req.hida && precision >= 2) {
        for (size_t i = 0; i < hk.size(); ++i)
            if (primes[i] == req.ell) {
                std::vector<const typename DigitCell<W>::HeckeData*> others;
                for (size_t j = 0; j < hk.size(); ++j) if (j != i) others.push_back(&hk[j]);
                rec.hida = cell.hida(hk[i], others, precision);
            }
    }
    rec.hecke_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - hecke_t0).count();
    if (req.keep_objects) cell.objects(rec.objects, hk);
    else if (req.keep_hecke) cell.hecke_objects(rec.objects, hk);   // (slot-15 WP-5) the module consumer
}

} // namespace

// (slot-18 WP-19b) the session: the prepared cell at one width.
struct DigitCellSession::Impl {
    Request req;
    CellRecord base;
    double cell_seconds = 0.0;
    std::mutex mu;
    virtual ~Impl() = default;
    virtual CellRecord run(const std::vector<uint64_t>& primes) = 0;
};

namespace {

template<class W>
struct SessionW final : DigitCellSession::Impl {
    DigitCell<W> cell;
    SessionW(const Request& rq, CellRecord rec, const std::vector<TypeQ>& type)
        : cell(rq.n_sym, rq.ell, rec.precision.working) {
        req = rq;
        req.primes.clear();
        cell.build_model();
        PrepareDigit<W>(cell, req, rec, type);
        base = std::move(rec);
    }
    CellRecord run(const std::vector<uint64_t>& primes) override {
        std::lock_guard<std::mutex> lk(mu);
        CellRecord rec = base;
        HeckeDigit<W>(cell, req, primes, rec);
        return rec;
    }
};

// Step 1 of RunCell (the per-cell rule, the certified precision bound, K):
// fills rec.precision / type and returns the width to run at.
Width StepOne(const Request& req, CellRecord& rec, std::vector<TypeQ>& type);

} // namespace

DigitCellSession::DigitCellSession(const Request& req) {
    const auto t0 = std::chrono::steady_clock::now();
    CellRecord rec;
    std::vector<TypeQ> type;
    const Width w = StepOne(req, rec, type);
    switch (w) {
        case Width::W32: impl_.reset(new SessionW<uint32_t>(req, std::move(rec), type)); break;
        case Width::W64: impl_.reset(new SessionW<uint64_t>(req, std::move(rec), type)); break;
        case Width::W128: impl_.reset(new SessionW<u128>(req, std::move(rec), type)); break;
    }
    if (!impl_) throw std::logic_error("LadicCell: unknown width");
    impl_->cell_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

DigitCellSession::~DigitCellSession() = default;

CellRecord DigitCellSession::run(const std::vector<uint64_t>& primes) { return impl_->run(primes); }

const CellRecord& DigitCellSession::base() const { return impl_->base; }

double DigitCellSession::cell_seconds() const { return impl_->cell_seconds; }

CellRecord RunCell(const Request& req) {
    DigitCellSession s(req);
    return s.run(req.primes);
}

namespace {

Width StepOne(const Request& req, CellRecord& rec, std::vector<TypeQ>& type) {
    const std::string who = "LadicCell";
    ENGINE_CHECK_MSG(req.k >= 2 && req.k <= kMaxK, "LadicCell: the requested precision must satisfy 2 <= k <= 4096");
    ENGINE_CHECK_MSG(req.module_K == 0 || (req.module_K >= req.k && req.module_K <= kMaxK && req.fixed_K == 0),
                     "LadicCell: module_K needs 2 <= k <= module_K <= 4096 and no fixed_K");
    const Width w = req.width;
    const std::string cell = CellText(req.n_sym, req.ell);
    // The per-cell rule on the requested precision (the working one is checked below).
    if (!req.allow_native_range) RequireDigitCell(who, req.n_sym, req.ell, req.module_K ? req.module_K : req.k, w);
    else if (req.ell < 5 || req.n_sym == 0 || req.n_sym % 2 != 0)
        throw NotImplementedInThisBuild(who + ": the TEST-ONLY native-range run needs ell >= 5 and n > 0 even at " + cell);

    rec.n_sym = req.n_sym;
    rec.ell = req.ell;
    rec.precision.requested = req.k;
    if (req.precision_source == PrecisionSource::Prepass) {
        // (slot-12 WP-4, OPT-IN) [C] v1.12.0-pre.2 cor:ordprecision: e_max and the
        // Z_(ell)-type of H^1 from the torsion pre-pass (prop:torsionprepass),
        // before any cochain complex.  H^0(Gamma; V) = 0 (n >= 1), H^2 has no
        // ell-torsion and no free part at ell >= 5; rank H^1 = 2 d_k - 1.
        const uint32_t K_target = req.fixed_K ? req.fixed_K : req.module_K ? req.module_K : req.k;
        hecke::ordinary::RequirePrecisionUse(req.n_sym, req.ell, K_target);   // the ell >= 5 refusal (A10: one pre-pass)
        const hecke::ordinary::TorsionPrepass pp = hecke::ordinary::TorsionExponentPrepass(req.n_sym, req.ell);
        type.push_back(TypeQ{0, {}});
        type.push_back(TypeQ{hecke::ordinary::FreeRankH1(req.n_sym), pp.torsion});
        type.push_back(TypeQ{0, {}});
        rec.precision.prepass = true;
        rec.precision.e_max = pp.e_max;
    } else {
        // Step 1: the word-size certified K-mode.
        const uint32_t Kw = WordLevelMax(req.ell);
        const std::vector<uint32_t> levels =
            req.word_level_override ? std::vector<uint32_t>{req.word_level_override} : DefaultWordLevels(req.n_sym, req.ell);
        WordRun wr;
        for (uint32_t K0 : levels) {
            ENGINE_CHECK_MSG(K0 >= 2 && K0 <= Kw, "LadicCell: a word level must satisfy 2 <= K0 <= K_w");
            wr = RunWordLevel(req.n_sym, req.ell, K0);
            rec.precision.word_levels_tried.push_back(K0);
            if (wr.certified) break;
        }
        if (!wr.certified) {
            std::ostringstream o;
            o << who << ": precision insufficient at the cell " << cell << ": the K-mode is uncertified at the word level K = "
              << wr.K0 << " (r_K(d^0) = " << wr.r0 << " needs " << wr.need0 << ", r_K(d^1) = " << wr.r1 << " needs "
              << wr.need1 << "); e_max is not certified and is never guessed -- refused, resumable (kind ladic_precision; "
              << "[E] v1.12.0-pre.2 sec:ring, the per-cell certified precision bound)";
            throw PrecisionInsufficient(o.str(), wr.K0, wr.r0, wr.r1, wr.need0, wr.need1);
        }
        rec.precision.word_certified = true;
        rec.precision.word_level = wr.K0;
        ENGINE_CHECK_MSG(wr.type.size() == 3 && wr.type[2].torsion.empty(),
                         "LadicCell: ladic_type^2 carries torsion at ell >= 5 ([C] lem:h2torsion)");
        ENGINE_CHECK_MSG(wr.type[2].free_rank == 0,
                         "LadicCell: ladic_type^2 has free rank > 0 (H^2 = 0 at ell >= 5; WP-14 pass 2 m-5)");
        rec.precision.e_max = wr.type[1].torsion.empty() ? 0u : wr.type[1].torsion.front();
        type = wr.type;
    }
    uint32_t K = req.fixed_K ? req.fixed_K : req.module_K ? req.module_K : req.k + rec.precision.e_max;
    ENGINE_CHECK_MSG(K >= 2 && (req.fixed_K != 0 || req.module_K != 0 || K > rec.precision.e_max),
                     "LadicCell: the working precision must exceed e_max");
    rec.precision.working = K;
    if (!req.allow_native_range) RequireDigitCell(who, req.n_sym, req.ell, K, w);
    return w;
}

} // namespace

WordKMode CertifiedWordKMode(uint32_t n_sym, uint64_t ell) {
    ENGINE_CHECK_MSG(ell >= 5 && n_sym > 0 && n_sym % 2 == 0, "LadicCell::CertifiedWordKMode needs ell >= 5 and n > 0 even");
    WordKMode out;
    for (uint32_t K0 : DefaultWordLevels(n_sym, ell)) {
        const WordRun wr = RunWordLevel(n_sym, ell, K0);
        out.levels_tried.push_back(K0);
        if (wr.certified) {
            out.certified = true;
            out.word_level = K0;
            out.type = wr.type;
            out.e_max = wr.type.size() > 1 && !wr.type[1].torsion.empty() ? wr.type[1].torsion.front() : 0u;
            break;
        }
    }
    return out;
}

} // namespace ladic_cell
} // namespace hecke
