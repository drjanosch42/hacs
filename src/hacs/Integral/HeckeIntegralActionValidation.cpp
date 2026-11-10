// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 10, WP-11 (Line 2, M1(a)): HeckeIntegralActionValidation.
// [C] = docs/latex/gl2_contract.tex v1.9.1, cited by label; [IB] =
// docs/implementation/integral_line_design_brief.md.  See the header for the arms.
#include "HeckeIntegralActionValidation.h"

#include "IntegralCharpolyCrt.h"
#include "Model/IntegralGroups.h"
#include "Model/IntegralHecke.h"
#include "Model/IsotropyModel.h"
#include "Model/PeriodicityClass.h"
#include "Model/StrictLift.h"

#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"
#include "../Hecke/CoefficientRing.h"
#include "../Hecke/FpPoly.h"
#include "../Hecke/HeckeCosetTables.h"
#include "../Hecke/HeckeOperatorOnK.h"
#include "../Hecke/HeckeSettings.h"
#include "../Hecke/HeckeTauTable.h"
#include "../Hecke/SL2ZTessellationTables.h"
#include "../../ModArithmetic/DenseElim.h"
#include "../../ModArithmetic/ModCtx.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace {

using engine::DenseBuf;
using engine::ModCtx;
using integral_model::AbelianGroupType;
using integral_model::ColumnLength;
using integral_model::InducedOperator;
using integral_model::IsotropyModel;
using integral_model::ModelCohomology;
using integral_model::PeriodicityClass;
using integral_model::StrictLiftBundle;
using U32 = uint32_t;

struct Fails {
    uint32_t n = 0;
    uint64_t checks = 0;
    uint32_t printed = 0;
    void check(bool ok, const std::string& what) {
        ++checks;
        if (!ok) {
            ++n;
            if (printed++ < 80) std::cout << "  [FAIL] " << what << std::endl;
        }
    }
};

std::string S(uint64_t v) { return std::to_string(v); }
std::string ring_name(U32 ell, U32 k) { return k == 1 ? "F_" + S(ell) : "Z/" + S(ell) + "^" + S(k); }

uint64_t ipow(uint64_t b, uint32_t e) {
    uint64_t r = 1;
    for (uint32_t i = 0; i < e; ++i) r *= b;
    return r;
}
uint32_t valuation(int64_t v, uint32_t ell) {
    ENGINE_CHECK(v != 0);
    uint64_t m = static_cast<uint64_t>(v < 0 ? -v : v);
    uint32_t e = 0;
    while (m % ell == 0) { m /= ell; ++e; }
    return e;
}

// ---- small matrix helpers on DenseBuf over a context ----------------------
DenseBuf<U32> scalar_shift(const ModCtx<U32>& ctx, const DenseBuf<U32>& A, int64_t c) {
    return integral_model::AddScalar(ctx, A, integral_model::ReduceSigned(ctx, c));
}
DenseBuf<U32> reduce_mod(const DenseBuf<U32>& A, U32 m) {
    DenseBuf<U32> B(A.rows, A.cols);
    for (U32 i = 0; i < A.rows; ++i)
        for (U32 j = 0; j < A.cols; ++j) B.at(i, j) = A.at(i, j) % m;
    return B;
}
bool is_zero(const DenseBuf<U32>& A) {
    for (U32 i = 0; i < A.rows; ++i)
        for (U32 j = 0; j < A.cols; ++j)
            if (A.at(i, j) != 0) return false;
    return true;
}
bool equal(const DenseBuf<U32>& A, const DenseBuf<U32>& B) {
    if (A.rows != B.rows || A.cols != B.cols) return false;
    for (U32 i = 0; i < A.rows; ++i)
        for (U32 j = 0; j < A.cols; ++j)
            if (A.at(i, j) != B.at(i, j)) return false;
    return true;
}
DenseBuf<U32> mul(const ModCtx<U32>& ctx, const DenseBuf<U32>& A, const DenseBuf<U32>& B) {
    return integral_model::MatMul(ctx, A, B);
}
DenseBuf<U32> lin(const ModCtx<U32>& ctx, int64_t a, const DenseBuf<U32>& A, int64_t b, const DenseBuf<U32>& B) {
    return integral_model::LinComb(ctx, integral_model::ReduceSigned(ctx, a), A, integral_model::ReduceSigned(ctx, b),
                                   B);
}
DenseBuf<U32> scaled(const ModCtx<U32>& ctx, const DenseBuf<U32>& A, uint64_t c) {
    DenseBuf<U32> Z(A.rows, A.cols);
    return integral_model::LinComb(ctx, static_cast<U32>(c % ctx.p), A, 0u, Z);
}

// An endomorphism of (+)_i Z/ell^{e_i} in coordinates (column j = image of b_j,
// entry i mod ell^{e_i}) is zero iff every entry is 0 mod ell^{e_i}.
bool module_endo_zero(const DenseBuf<U32>& A, const std::vector<U32>& ex, U32 ell) {
    for (U32 i = 0; i < A.rows; ++i) {
        const uint64_t m = ipow(ell, ex[i]);
        for (U32 j = 0; j < A.cols; ++j)
            if (A.at(i, j) % m != 0) return false;
    }
    return true;
}

hecke::FpPoly charpoly_mod_ell(const DenseBuf<U32>& A, U32 ell) {
    std::vector<std::vector<U32>> rows(A.rows, std::vector<U32>(A.cols, 0));
    for (U32 i = 0; i < A.rows; ++i)
        for (U32 j = 0; j < A.cols; ++j) rows[i][j] = A.at(i, j) % ell;
    if (A.rows == 0) return hecke::FpPoly::one(ell);
    return hecke::CharPolyBerkowitz(MatrixT<U32>(A.rows, A.cols, ell, rows));
}

// prod_i (x - r_i)^{m_i} mod M, low -> high (roots as signed decimals).
std::vector<U32> root_product_mod(const std::vector<std::pair<std::string, U32>>& roots, U32 M) {
    std::vector<uint64_t> c = {1};
    for (const auto& [r, mult] : roots) {
        const uint64_t rv = integral::ReduceDecimalMod(r, M);
        for (U32 t = 0; t < mult; ++t) {
            std::vector<uint64_t> d(c.size() + 1, 0);
            for (size_t i = 0; i < c.size(); ++i) {
                d[i + 1] = (d[i + 1] + c[i]) % M;
                d[i] = (d[i] + (M - rv) % M * c[i]) % M;
            }
            c = d;
        }
    }
    std::vector<U32> out(c.begin(), c.end());
    return out;
}
std::string poly_str(const std::vector<U32>& c) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < c.size(); ++i) os << (i ? "," : "") << c[i];
    os << "]";
    return os.str();
}
std::string exps_str(const std::vector<U32>& e, U32 ell) {
    if (e.empty()) return "0";
    std::ostringstream os;
    for (size_t i = 0; i < e.size(); ++i) os << (i ? " + " : "") << "Z/" << ipow(ell, e[i]);
    return os.str();
}

// ---- one ring cell of the model: the complex, T_p for p in {2, 3, 5}, E_q ---
struct ModelRing {
    U32 ell = 0, k = 0, n = 0;
    ModCtx<U32> ctx;
    std::unique_ptr<ModelCohomology<U32>> mc;
    std::map<U32, std::vector<DenseBuf<U32>>> th;   // p -> Theta^(q), q = 0..q_max
    std::vector<DenseBuf<U32>> E;                    // E_q^*, q = 0..3
};

std::unique_ptr<ModelRing> make_ring(const std::map<U32, std::unique_ptr<StrictLiftBundle>>& B,
                                     const PeriodicityClass* pc, U32 ell, U32 k, U32 n, bool with_ops = true) {
    auto r = std::make_unique<ModelRing>();
    r->ell = ell;
    r->k = k;
    r->n = n;
    r->ctx = integral_model::RingContext(ell, k);
    const IsotropyModel& M = B.at(2)->model();
    r->mc = std::make_unique<ModelCohomology<U32>>(M, r->ctx, n);
    if (!with_ops) return r;
    for (const auto& [p, b] : B) {
        auto& v = r->th[p];
        for (U32 q = 0; q <= M.q_max(); ++q) v.push_back(b->lift().theta_hat(r->ctx, n, q));
    }
    if (pc != nullptr)
        for (U32 q = 0; q + 2 <= M.q_max(); ++q) r->E.push_back(pc->matrix(r->ctx, n, q));
    return r;
}

// ---- line 1: the shipped E8 route (canonical tau) over Z/ell^k -------------
struct Line1Cell {
    SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    std::unique_ptr<hecke::SL2ZTessellationTables> tess;
    std::unique_ptr<LocalSystemSimplicialComplexT<U32>> lssc;
    std::unique_ptr<EquivariantCohomologyOnKT<U32>> coh;
    std::map<U32, std::unique_ptr<hecke::HeckeCosetTablesT<U32>>> cosets;
    std::map<U32, std::unique_ptr<hecke::HeckeTauTableT<U32>>> tau;
    std::map<U32, std::unique_ptr<hecke::HeckeOperatorOnKT<U32>>> op;
};

std::unique_ptr<Line1Cell> build_line1(U32 n, U32 ell, U32 k, const std::vector<U32>& ps) {
    ENGINE_CHECK(k >= 2);
    hecke::ScopedCoefficientRing ring(k);
    auto c = std::make_unique<Line1Cell>();
    c->tess = std::make_unique<hecke::SL2ZTessellationTables>(c->K);
    const ModCtx<U32> ctx = hecke::CoefficientRingContext<U32>("HeckeIntegralActionValidation", ell, k);
    auto rels = hecke::GenerateSL2ZRelationsRing<U32>(c->K, n, ctx);
    c->lssc = std::make_unique<LocalSystemSimplicialComplexT<U32>>(c->K, std::move(rels), n + 1, ctx.p);
    c->coh = std::make_unique<EquivariantCohomologyOnKT<U32>>(c->lssc.get());
    c->coh->compute_cohomology();
    for (U32 p : ps) {
        c->cosets[p] = std::make_unique<hecke::HeckeCosetTablesT<U32>>(p, 1, ell, n);
        c->tau[p] = std::make_unique<hecke::HeckeTauTableT<U32>>(c->K, *c->tess, *c->cosets[p], ell, n);
        c->op[p] = std::make_unique<hecke::HeckeOperatorOnKT<U32>>(c->coh.get(), c->cosets[p].get(), c->tau[p].get());
        c->op[p]->compute();
    }
    return c;
}

// ---- the congruence index of a free Z/ell^m[T]-module with char (x-a)(x-b)^{r-1} ----
// L a lattice of rank r with T acting semisimply over Q_ell with eigenvalues a
// (multiplicity 1) and b (r - 1); L_a, L_b the saturated eigenlattices.  The
// congruence module C = L / (L_a (+) L_b) is cyclic of order ell^{v - c_b}
// (reading 1: (T - b)L = ell^{c_b} L_a, so length((T - b)M) = m - c_b) and of
// order ell^{(r-1)v - c_a} (reading 2: length((T - a)M) = (r-1)m - c_a), v =
// v_ell(a - b), M = L / ell^m, valid for m >= v.  Returns {reading 1, reading 2}
// (-1 when a reading is outside its range).
std::pair<int, int> congruence_index(const DenseBuf<U32>& A, U32 ell, U32 m, int64_t a, int64_t b) {
    const ModCtx<U32> cm = integral_model::RingContext(ell, m);
    const U32 r = A.rows;
    const int v = static_cast<int>(valuation(a - b, ell));
    const DenseBuf<U32> Ar = reduce_mod(A, cm.p);
    const int lb = static_cast<int>(ColumnLength(cm, scalar_shift(cm, Ar, -b)));
    const int la = static_cast<int>(ColumnLength(cm, scalar_shift(cm, Ar, -a)));
    const int cb = static_cast<int>(m) - lb;
    const int ca = static_cast<int>((r - 1) * m) - la;
    const int c1 = v - cb, c2 = static_cast<int>(r - 1) * v - ca;
    return {(c1 >= 0 && cb >= 0) ? c1 : -1, (c2 >= 0 && ca >= 0) ? c2 : -1};
}

}  // namespace

uint32_t HeckeIntegralActionValidation() {
    const auto t0 = std::chrono::steady_clock::now();
    Fails F;
    std::cout << "=== HeckeIntegralActionValidation (slot-10 WP-11, Line 2 M1(a): the integral Hecke action, "
                 "basis-free; [IB] Sec. 6.2 M1(a)) ===\n";

    // ---- (P) preconditions ------------------------------------------------------
    std::cout << "--- (P) the lifts for T_2, T_3, T_5 and the class e on ONE model complex ---\n";
    std::map<U32, std::unique_ptr<StrictLiftBundle>> B;
    for (U32 p : {2u, 3u, 5u}) {
        try {
            B[p] = std::make_unique<StrictLiftBundle>(p, 5);
        } catch (const std::exception& e) {
            F.check(false, "(P) the strict lift for p = " + S(p) + " could not be built: " + e.what());
        }
    }
    if (B.size() != 3) {
        std::cout << "HeckeIntegralActionValidation: FAILED (" << F.n << ")\n";
        return F.n;
    }
    const IsotropyModel& M = B.at(2)->model();
    {
        // The three bundles each own a model; they must be the same complex
        // (their cochain spaces are identified below).
        const ModCtx<U32> c(10007);
        bool same = true;
        for (U32 p : {3u, 5u}) {
            const IsotropyModel& Mp = B.at(p)->model();
            for (U32 q = 0; q <= M.q_top(); ++q) same = same && (Mp.rank(q) == M.rank(q));
            for (U32 n : {0u, 5u, 10u})
                for (U32 q = 0; q <= M.q_max(); ++q) same = same && equal(Mp.coboundary(c, n, q), M.coboundary(c, n, q));
        }
        F.check(same, "(P) the models of the three bundles are not one complex");
        for (U32 p : {2u, 3u, 5u}) {
            const auto C = B.at(p)->lift().certify();
            F.check(C.all(), "(P) p = " + S(p) + ": StrictLift::certify() " + C.first_failure);
        }
        std::cout << "  three bundles, one complex: " << (same ? "yes" : "NO")
                  << "; StrictLift::certify() for p = 2, 3, 5 re-run (WP-10's (T1)/(T2)/(T3)/(S)/(B))\n";
    }
    std::unique_ptr<PeriodicityClass> pc;
    try {
        pc = std::make_unique<PeriodicityClass>(M);
        const auto C = pc->certify();
        F.check(C.all(), "(P) PeriodicityClass::certify() " + C.first_failure);
        std::cout << "  PeriodicityClass (WP-13) on the same model: certify() "
                  << (C.all() ? "ALL HOLD" : "FAILED") << "\n";
    } catch (const std::exception& e) {
        F.check(false, std::string("(P) PeriodicityClass could not be built: ") + e.what());
    }
    std::cout << "  'strictly eps^*-compatible' ([IB] Sec. 6.2 M1(a) item 3): CITED from WP-10 -- (C) over F_ell "
                 "(2,3,5,7) and Z/ell^k (ell 5, 7); over Z/ell^K at 2, 3 it follows from the exact Z-certificate (S) "
                 "re-run above; no Theta^s_g on the Bredon object is built (plan B-1, decision 4)\n";

    // ---- (1) O6 at n = 10 at good primes; O5 minimal identities at n = 12, 16 ----
    std::cout << "--- (1) O6 on Lambda^1 (x) F_ell = H^1(C_I (x) F_ell) at good primes, through Theta^_g ---\n";
    {
        struct Id {
            U32 n;
            std::vector<U32> ells;
        };
        const std::vector<Id> ids = {{10, {5, 7, 11, 13, 17, 691, 10007, 1000003}},
                                     {12, {13, 17, 691, 10007}},
                                     {16, {5, 11, 13, 691, 10007}}};
        for (const Id& id : ids) {
            uint32_t ok = 0, total = 0, neg_caught = 0;
            for (U32 ell : id.ells) {
                auto R = make_ring(B, nullptr, ell, 1, id.n);
                const ModCtx<U32>& c = R->ctx;
                const auto& T2 = R->th[2][1];
                const auto& T3 = R->th[3][1];
                const U32 r = integral::RankLambda1(id.n);
                F.check(R->mc->length_H(1) == r, "(1) n=" + S(id.n) + " ell=" + S(ell) + ": dim H^1 = " +
                                                     S(R->mc->length_H(1)) + " != rank Lambda^1 = " + S(r));
                for (U32 q = 2; q <= M.q_max(); ++q)
                    F.check(R->mc->length_H(q) == 0, "(1) n=" + S(id.n) + " ell=" + S(ell) + ": H^" + S(q) +
                                                          " != 0 at a good prime ([C, cl:amalgam](3))");
                bool good = true;
                bool neg = false;
                if (id.n == 10) {
                    const auto P1 = mul(c, scalar_shift(c, T2, -2049), scalar_shift(c, T2, 24));
                    const auto P2 = lin(c, 3, scalar_shift(c, T3, -252), -256, scalar_shift(c, T2, 24));
                    const bool i1 = R->mc->induces_zero(1, 1, P1), i2 = R->mc->induces_zero(1, 1, P2);
                    F.check(i1, "(1) n=10 ell=" + S(ell) + ": (T_2 - 2049)(T_2 + 24) != 0 on H^1");
                    F.check(i2, "(1) n=10 ell=" + S(ell) + ": 3(T_3 - 252) != 256(T_2 + 24) on H^1");
                    good = i1 && i2;
                    // negative control: a wrong constant must be seen
                    const auto N1 = mul(c, scalar_shift(c, T2, -2049), scalar_shift(c, T2, 23));
                    const auto N2 = lin(c, 3, scalar_shift(c, T3, -252), -255, scalar_shift(c, T2, 24));
                    neg = !R->mc->induces_zero(1, 1, N1) && !R->mc->induces_zero(1, 1, N2);
                } else if (id.n == 12) {
                    good = R->mc->induces_zero(1, 1, scalar_shift(c, T2, -8193));
                    F.check(good, "(1) n=12 ell=" + S(ell) + ": T_2 != 8193 on H^1");
                    neg = !R->mc->induces_zero(1, 1, scalar_shift(c, T2, -8192));
                } else {
                    good = R->mc->induces_zero(1, 1, mul(c, scalar_shift(c, T2, -131073), scalar_shift(c, T2, 528)));
                    F.check(good, "(1) n=16 ell=" + S(ell) + ": (T_2 - 131073)(T_2 + 528) != 0 on H^1");
                    neg = !R->mc->induces_zero(1, 1, mul(c, scalar_shift(c, T2, -131073), scalar_shift(c, T2, 527)));
                }
                F.check(neg, "(1) n=" + S(id.n) + " ell=" + S(ell) + ": the negative control (wrong constant) passed");
                ++total;
                ok += good ? 1 : 0;
                neg_caught += neg ? 1 : 0;
            }
            std::cout << "  n=" << id.n << ": "
                      << (id.n == 10 ? "(T_2 - 2049)(T_2 + 24) = 0 and 3(T_3 - 252) = 256(T_2 + 24)"
                                     : id.n == 12 ? "T_2 = 8193" : "(T_2 - 131073)(T_2 + 528) = 0")
                      << " on H^1 at " << ok << " / " << total << " good primes; wrong constant refused at "
                      << neg_caught << " / " << total << "\n";
        }
    }

    // ---- (2) the 691-congruence module at n = 10 --------------------------------
    std::cout << "--- (2) the 691-congruence module of H^1_+ (x) Z_691 at n = 10 (O6: Z/691) ---\n";
    {
        struct Op {
            U32 p;
            int64_t a, b;
        };
        const std::vector<Op> ops = {{2, 2049, -24}, {3, 177148, 252}};
        const U32 ell = 691;
        for (U32 k : {2u, 3u}) {
            const U32 Mk = static_cast<U32>(ipow(ell, k));
            const auto want2 = root_product_mod({{"2049", 1}, {"-24", 2}}, Mk);
            const auto want3 = root_product_mod({{"177148", 1}, {"252", 2}}, Mk);
            // line 1: the shipped E8 route (canonical tau).
            std::map<U32, DenseBuf<U32>> A1;
            std::vector<U32> ex1;
            try {
                auto c = build_line1(10, ell, k, {2, 3});
                ex1 = c->coh->H_invariant_factors(1);
                for (U32 p : {2u, 3u}) A1[p] = c->op[p]->H_matrix_module(1);
            } catch (const std::exception& e) {
                F.check(false, std::string("(2) line 1 at Z/691^") + S(k) + " threw: " + e.what());
                continue;
            }
            // line 2: the model.
            auto R = make_ring(B, nullptr, ell, k, 10);
            const InducedOperator<U32> N = integral_model::Subquotient(R->ctx, R->mc->Z(1), R->mc->B(1));
            std::map<U32, DenseBuf<U32>> A2;
            for (U32 p : {2u, 3u}) A2[p] = integral_model::OperatorMatrix(R->ctx, N, R->mc->B(1), R->th[p][1]);
            const std::vector<U32> free3(3, k);
            F.check(ex1 == free3, "(2) line 1 H^1(Z/691^" + S(k) + ") is not free of rank 3");
            F.check(N.exponents == free3, "(2) line 2 H^1(C_I (x) Z/691^" + S(k) + ") is not free of rank 3");
            if (ex1 != free3 || N.exponents != free3) continue;
            const ModCtx<U32> cm = integral_model::RingContext(ell, k);
            for (const auto& [line, A] : {std::make_pair(std::string("line 1"), &A1), std::make_pair(std::string("line 2"), &A2)}) {
                F.check(integral_model::CharpolyDivisionFree(cm, (*A)[2]) == want2,
                        "(2) " + line + " Z/691^" + S(k) + ": charpoly(T_2) != (x - 2049)(x + 24)^2");
                F.check(integral_model::CharpolyDivisionFree(cm, (*A)[3]) == want3,
                        "(2) " + line + " Z/691^" + S(k) + ": charpoly(T_3) != (x - 177148)(x - 252)^2");
                F.check(is_zero(mul(cm, scalar_shift(cm, (*A)[2], -2049), scalar_shift(cm, (*A)[2], 24))),
                        "(2) " + line + " Z/691^" + S(k) + ": (T_2 - 2049)(T_2 + 24) != 0");
                F.check(is_zero(lin(cm, 3, scalar_shift(cm, (*A)[3], -252), -256, scalar_shift(cm, (*A)[2], 24))),
                        "(2) " + line + " Z/691^" + S(k) + ": 3(T_3 - 252) != 256(T_2 + 24)");
                for (const Op& o : ops) {
                    const auto [c1, c2] = congruence_index((*A)[o.p], ell, k, o.a, o.b);
                    F.check(c1 == 1 && c2 == 1, "(2) " + line + " Z/691^" + S(k) + " via T_" + S(o.p) +
                                                    ": congruence module Z/691^(" + std::to_string(c1) + ", " +
                                                    std::to_string(c2) + ") != Z/691");
                    std::cout << "  " << line << ", Z/691^" << k << ", T_" << o.p << ": H^1 = (Z/691^" << k
                              << ")^3, charpoly = O5 mod 691^" << k << ", congruence module Z/691^" << c1
                              << " (via T - b) = Z/691^" << c2 << " (via T - a)\n";
                }
            }
        }
        std::cout << "  (the congruence module L / (L_a + L_b) of H^1 (x) Z_691 equals that of H^1_+ (x) Z_691: "
                     "L_- lies in the cusp eigenlattice L_b; no iota over Z/691^k is needed)\n";
    }

    // ---- (3) + (4) + (5): the bad primes 2 and 3 ---------------------------------
    std::cout << "--- (3) T_p on the 2- and 3-primary torsion, degrees <= 5; X6, X7, test (5) ---\n";
    uint64_t x6_nonzero_E = 0, x6_neg_caught = 0, x7_chain_noncommuting = 0;
    // The M1(a) cells n in {10, 12, 16} run every arm; the sweep over the other
    // n <= 20 (odd n included) runs (3) and (5)'s C_I half only.
    std::vector<U32> ns = {10, 12, 16};
    for (U32 n = 0; n <= 20; ++n)
        if (n != 10 && n != 12 && n != 16) ns.push_back(n);
    uint32_t sweep_cells = 0;
    for (U32 n : ns) {
        const bool core = (n == 10 || n == 12 || n == 16);
        sweep_cells += core ? 0 : 1;
        const integral_model::ModelIntegralCohomology G = integral_model::AssembleFromModel(M, n, {2, 3}, 10007);
        F.check(G.all_certified(), "(3) n=" + S(n) + ": M0's assembly is not certified");
        for (U32 ell : {2u, 3u}) {
            // exponents e_q of T^q_ell, q = 0 .. 6
            std::vector<U32> tors_ex[7];
            U32 E = 0;
            for (U32 q = 0; q <= 6; ++q) {
                const AbelianGroupType& g = (q <= G.q_max) ? G.H[q] : G.tors_H_top;
                auto it = g.torsion.find(ell);
                if (it != g.torsion.end()) tors_ex[q] = it->second;
                if (!tors_ex[q].empty()) E = std::max(E, tors_ex[q][0]);
            }
            const U32 e1 = tors_ex[1].empty() ? 0 : tors_ex[1][0];
            const U32 K = std::max(E, e1 + 4), K2 = K + E;
            F.check(ipow(ell, K2) < (1ull << 31), "(3) ell^K' fits a word");
            auto R = make_ring(B, pc.get(), ell, K, n);
            auto R2 = make_ring(B, nullptr, ell, K2, n, false);
            auto R1 = make_ring(B, pc.get(), ell, 1, n);
            const ModCtx<U32>& c = R->ctx;
            std::cout << "  n=" << n << " ell=" << ell << ": K=" << K << " K'=" << K2 << "; M0: T^q =";
            for (U32 q = 1; q <= 6; ++q) std::cout << " [" << q << "] " << exps_str(tors_ex[q], ell);
            std::cout << "\n";

            // (3a) X7: commutativity on H^q, q = 0..5, over Z/ell^K and F_ell
            {
                const std::vector<std::pair<U32, U32>> pairs = {{2, 3}, {2, 5}, {3, 5}};
                uint32_t ok = 0, tot = 0;
                for (ModelRing* X : {R.get(), R1.get()})
                    for (U32 q = 0; q <= M.q_max(); ++q)
                        for (const auto& [a, b] : pairs) {
                            const auto AB = mul(X->ctx, X->th[a][q], X->th[b][q]);
                            const auto BA = mul(X->ctx, X->th[b][q], X->th[a][q]);
                            if (!equal(AB, BA)) ++x7_chain_noncommuting;
                            const bool z = X->mc->induces_zero(q, q, lin(X->ctx, 1, AB, -1, BA));
                            F.check(z, "(3a) X7 n=" + S(n) + " " + ring_name(ell, X->k) + " q=" + S(q) + ": T_" + S(a) +
                                           " T_" + S(b) + " != T_" + S(b) + " T_" + S(a));
                            ++tot;
                            ok += z ? 1 : 0;
                        }
                std::cout << "    X7: T_p T_p' = T_p' T_p on H^q, q = 0..5, pairs (2,3) (2,5) (3,5), Z/" << ell << "^"
                          << K << " and F_" << ell << ": " << ok << " / " << tot << "\n";
            }

            // (3b) the torsion modules, two readings; T_p on T^q / ell
            {
                std::map<U32, std::map<U32, hecke::FpPoly>> sub_cp, quot_cp;   // q -> p -> charpoly mod ell on T^q/ell
                std::map<U32, std::vector<U32>> sub_ex, quot_ex;
                for (U32 q = 0; q <= M.q_max(); ++q) {
                    const DenseBuf<U32> Red = integral_model::ReducedCocycles(*R2->mc, q, c);
                    F.check(R->mc->are_cocycles(q, Red), "(3b) reduced cocycles are cocycles");
                    // (sub) q >= 2: IM(red) = T^q
                    if (q >= 2) {
                        const auto N = integral_model::Subquotient(c, Red, R->mc->B(q));
                        sub_ex[q] = N.exponents;
                        F.check(N.exponents == tors_ex[q], "(3b) n=" + S(n) + " ell=" + S(ell) + " q=" + S(q) +
                                                               ": IM(red) = " + exps_str(N.exponents, ell) +
                                                               " != M0's T^q = " + exps_str(tors_ex[q], ell));
                        std::map<U32, DenseBuf<U32>> Ap;
                        for (U32 p : {2u, 3u, 5u}) {
                            Ap[p] = integral_model::OperatorMatrix(c, N, R->mc->B(q), R->th[p][q]);
                            sub_cp[q][p] = charpoly_mod_ell(Ap[p], ell);
                        }
                        if (N.rank() > 0) {
                            const auto D = lin(c, 1, mul(c, Ap[2], Ap[3]), -1, mul(c, Ap[3], Ap[2]));
                            // (slot-14 WP-14b, m-18): this is IMPLIED by (3a)'s
                            // basis-free X7 check plus the torsion reading
                            // (T^q's cyclic coordinates are read off the same
                            // module the (3a) H^q operators act on) -- a
                            // consistency check on the coordinate reading, not
                            // an independent test of X7.
                            F.check(module_endo_zero(D, N.exponents, ell),
                                    "(3b) X7 on T^" + S(q) + " (coordinates, consistency check)");
                        }
                    }
                    // (quot) H^q(Z/ell^K) / IM(red) = T^{q+1}
                    {
                        const DenseBuf<U32> Rl = integral_model::HStack(R->mc->B(q), Red);
                        const auto N = integral_model::Subquotient(c, R->mc->Z(q), Rl);
                        quot_ex[q + 1] = N.exponents;
                        F.check(N.exponents == tors_ex[q + 1], "(3b) n=" + S(n) + " ell=" + S(ell) + " H^" + S(q) +
                                                                   "/IM(red) = " + exps_str(N.exponents, ell) +
                                                                   " != M0's T^" + S(q + 1) + " = " +
                                                                   exps_str(tors_ex[q + 1], ell));
                        for (U32 p : {2u, 3u, 5u})
                            quot_cp[q + 1][p] =
                                charpoly_mod_ell(integral_model::OperatorMatrix(c, N, Rl, R->th[p][q]), ell);
                    }
                }
                std::ostringstream line;
                for (U32 q = 1; q <= 6; ++q) {
                    if (tors_ex[q].empty()) continue;
                    line << "    T^" << q << " = " << exps_str(tors_ex[q], ell) << ":";
                    for (U32 p : {2u, 3u, 5u}) {
                        const hecke::FpPoly& qc = quot_cp[q][p];
                        line << " T_" << p << " mod " << ell << " char " << qc.to_string();
                        if (q >= 2 && q <= M.q_max()) {
                            const bool same = (sub_cp[q][p] == qc);
                            F.check(same, "(3b) n=" + S(n) + " ell=" + S(ell) + " T^" + S(q) + " p=" + S(p) +
                                              ": the two readings' T_p mod ell differ: sub " +
                                              sub_cp[q][p].to_string() + " vs quot " + qc.to_string());
                        }
                        line << (p == 5 ? "" : ";");
                    }
                    line << (q >= 2 && q <= M.q_max() ? "  [two readings agree]" : "  [Tor reading]") << "\n";
                }
                // periodicity of the torsion Hecke modules (q >= 2): T^q ~= T^{q+2}
                for (U32 q : {2u, 3u, 4u}) {
                    F.check(quot_ex[q] == quot_ex[q + 2], "(3b) T^" + S(q) + " !~= T^" + S(q + 2));
                    for (U32 p : {2u, 3u, 5u})
                        F.check(quot_cp[q][p] == quot_cp[q + 2][p],
                                "(3b) n=" + S(n) + " ell=" + S(ell) + " p=" + S(p) + ": T_p mod ell on T^" + S(q) +
                                    " != on T^" + S(q + 2));
                }
                std::cout << line.str();
            }

            // (3c) X6 and its char-poly form; test (5)
            if (pc) {
                uint32_t ok = 0, tot = 0;
                for (ModelRing* X : {R.get(), R1.get()})
                    for (U32 q = 0; q + 2 <= M.q_max(); ++q) {
                        if (!X->mc->induces_zero(q, q + 2, X->E[q])) ++x6_nonzero_E;
                        for (U32 p : {2u, 3u, 5u}) {
                            const auto D = lin(X->ctx, 1, mul(X->ctx, X->th[p][q + 2], X->E[q]), -1,
                                               mul(X->ctx, X->E[q], X->th[p][q]));
                            const bool z = X->mc->induces_zero(q, q + 2, D);
                            F.check(z, "(3c) X6 n=" + S(n) + " " + ring_name(ell, X->k) + " q=" + S(q) + " p=" + S(p) +
                                           ": T_p E_q != E_q T_p on H^q");
                            ++tot;
                            ok += z ? 1 : 0;
                        }
                        // negative control: mixing T_2 and T_3
                        const auto Dn = lin(X->ctx, 1, mul(X->ctx, X->th[2][q + 2], X->E[q]), -1,
                                            mul(X->ctx, X->E[q], X->th[3][q]));
                        if (!X->mc->induces_zero(q, q + 2, Dn)) ++x6_neg_caught;
                    }
                std::cout << "    X6: T_p E_q = E_q T_p on H^q -> H^{q+2}, q = 0..3, p = 2, 3, 5, Z/" << ell << "^" << K
                          << " and F_" << ell << ": " << ok << " / " << tot << "\n";
            } else {
                F.check(false, "(3c) X6 not run: no PeriodicityClass");
            }
            {
                // over F_ell: the matrices of T_p on H^q
                std::map<U32, std::map<U32, hecke::FpPoly>> cp;   // q -> p
                for (U32 q = 0; q <= M.q_max(); ++q) {
                    const auto N = integral_model::Subquotient(R1->ctx, R1->mc->Z(q), R1->mc->B(q));
                    for (U32 p : {2u, 3u, 5u})
                        cp[q][p] = charpoly_mod_ell(integral_model::OperatorMatrix(R1->ctx, N, R1->mc->B(q), R1->th[p][q]),
                                                    ell);
                }
                bool t4 = true, t5 = true;
                for (U32 p : {2u, 3u, 5u}) {
                    for (U32 q : {2u, 3u}) t4 = t4 && (cp[q][p] == cp[q + 2][p]);
                    t5 = t5 && (hecke::FpPoly::mod(cp[1][p], cp[3][p]).degree() == -1);
                }
                F.check(t4, "(3c) n=" + S(n) + " F_" + S(ell) + ": charpoly A^(q)(T_p) != charpoly A^(q+2)(T_p), q = 2, 3");
                F.check(t5, "(3d) n=" + S(n) + " F_" + S(ell) + ": test (5) charpoly A^(3)(T_p) does not divide A^(1)(T_p)");
                std::cout << "    over F_" << ell << ": charpoly A^(q)(T_p) = A^(q+2)(T_p), q = 2, 3: "
                          << (t4 ? "yes" : "NO") << "; test (5) A^(3) | A^(1) for p = 2, 3, 5: " << (t5 ? "yes" : "NO")
                          << "  (A^(1)(T_2) char " << cp[1][2].to_string() << ", A^(3)(T_2) char "
                          << cp[3][2].to_string() << ")\n";
            }

            // (4) Lambda^1 (x) Z/ell^m, m = K - e1, and at m - 1 (X1's Hecke half)
            if (core) {
                std::vector<U32> cps_prev;
                for (U32 KK : {K, K - 1}) {
                    std::unique_ptr<ModelRing> RK, RK2;
                    const ModelRing* Rk = R.get();
                    const ModelRing* Rk2 = R2.get();
                    if (KK != K) {
                        RK = make_ring(B, nullptr, ell, KK, n);
                        RK2 = make_ring(B, nullptr, ell, KK + E, n, false);
                        Rk = RK.get();
                        Rk2 = RK2.get();
                    }
                    const ModCtx<U32>& ck = Rk->ctx;
                    const U32 m = KK - e1;
                    const DenseBuf<U32> G1 = scaled(ck, integral_model::ReducedCocycles(*Rk2->mc, 1, ck), ipow(ell, e1));
                    const auto N = integral_model::Subquotient(ck, G1, Rk->mc->B(1));
                    const U32 r = integral::RankLambda1(n);
                    F.check(N.exponents == std::vector<U32>(r, m),
                            "(4) n=" + S(n) + " ell=" + S(ell) + ": ell^e1 IM(red) = " + exps_str(N.exponents, ell) +
                                " is not (Z/ell^" + S(m) + ")^" + S(r));
                    if (N.exponents != std::vector<U32>(r, m)) continue;
                    std::map<U32, DenseBuf<U32>> A;
                    for (U32 p : {2u, 3u}) A[p] = integral_model::OperatorMatrix(ck, N, Rk->mc->B(1), Rk->th.at(p)[1]);
                    const U32 Mm = static_cast<U32>(ipow(ell, m));
                    const ModCtx<U32> cm = integral_model::RingContext(ell, m);
                    for (const integral::O5Row& row : integral::O5Table()) {
                        if (row.n != n || (row.p != 2 && row.p != 3)) continue;
                        std::vector<std::pair<std::string, U32>> roots = {{row.eis_root, 1}};
                        if (row.cusp_root != nullptr && row.cusp_mult > 0) roots.push_back({row.cusp_root, row.cusp_mult});
                        const auto want = root_product_mod(roots, Mm);
                        const auto got = integral_model::CharpolyDivisionFree(cm, reduce_mod(A[row.p], Mm));
                        F.check(got == want, "(4) n=" + S(n) + " ell=" + S(ell) + " m=" + S(m) + ": charpoly(T_" +
                                                 S(row.p) + " | Lambda^1) " + poly_str(got) + " != O5 " + poly_str(want));
                        if (KK == K && row.p == 2) cps_prev = got;
                        if (KK != K && row.p == 2) {
                            // X1 (Hecke half): the charpoly at m - 1 is the reduction of the one at m
                            bool red = true;
                            const std::vector<U32>& hi = cps_prev;
                            for (size_t i = 0; i < got.size() && i < hi.size(); ++i) red = red && (hi[i] % Mm == got[i]);
                            F.check(red && !hi.empty(), "(5) X1 n=" + S(n) + " ell=" + S(ell) +
                                                            ": charpoly(T_2 | Lambda^1) mod ell^(m-1) is not the reduction");
                        }
                        if (KK == K)
                            std::cout << "    Lambda^1 (x) Z/" << ell << "^" << m << " (rank " << r << "): charpoly(T_"
                                      << row.p << ") = O5 " << row.verbatim << " mod " << ell << "^" << m << ": "
                                      << (got == want ? "yes" : "NO") << "\n";
                    }
                    if (KK != K) continue;
                    if (n == 10) {
                        F.check(is_zero(mul(cm, scalar_shift(cm, reduce_mod(A[2], Mm), -2049),
                                            scalar_shift(cm, reduce_mod(A[2], Mm), 24))),
                                "(4) n=10 ell=" + S(ell) + ": O6 (T_2 - 2049)(T_2 + 24) != 0 on Lambda^1 (x) Z/ell^m");
                        F.check(is_zero(lin(cm, 3, scalar_shift(cm, reduce_mod(A[3], Mm), -252), -256,
                                            scalar_shift(cm, reduce_mod(A[2], Mm), 24))),
                                "(4) n=10 ell=" + S(ell) + ": O6 3(T_3 - 252) != 256(T_2 + 24) on Lambda^1 (x) Z/ell^m");
                        std::cout << "    O6 on Lambda^1 (x) Z/" << ell << "^" << m << " (the local form at a bad prime): checked\n";
                    }
                    // the local type: the eigenvalues, their congruence, and the congruence index
                    for (const integral::O5Row& row : integral::O5Table()) {
                        if (row.n != n || row.p != 2) continue;
                        std::ostringstream os;
                        os << "    local type of Lambda^1 (x) Z_" << ell << " as a Z_" << ell << "[T_2]-module: ";
                        if (row.cusp_root == nullptr) {
                            os << "Z_" << ell << " with T_2 = " << row.eis_root << "\n";
                        } else {
                            const int64_t a = std::stoll(row.eis_root), b = std::stoll(row.cusp_root);
                            const U32 v = valuation(a - b, ell);
                            const auto [c1, c2] = congruence_index(reduce_mod(A[2], Mm), ell, m, a, b);
                            F.check(m >= v && c1 >= 0 && c1 == c2, "(4) n=" + S(n) + " ell=" + S(ell) +
                                                                      ": the two congruence readings differ (" +
                                                                      std::to_string(c1) + ", " + std::to_string(c2) + ")");
                            const auto fa = engine::elim::invariant_factors(cm, engine::cview(scalar_shift(cm, reduce_mod(A[2], Mm), -a)));
                            const auto fb = engine::elim::invariant_factors(cm, engine::cview(scalar_shift(cm, reduce_mod(A[2], Mm), -b)));
                            os << "a = " << row.eis_root << " (Eisenstein), b = " << row.cusp_root << " (x" << row.cusp_mult
                               << "), v_" << ell << "(a - b) = " << v << "; L / (L_a + L_b) = "
                               << (c1 <= 0 ? std::string("0 (split: L = L_a + L_b)") : "Z/" + S(ipow(ell, c1)))
                               << "; colspan(T - a) ~ " << exps_str(fa, ell) << ", colspan(T - b) ~ " << exps_str(fb, ell)
                               << " (over Z/" << ell << "^" << m << ")\n";
                        }
                        std::cout << os.str();
                    }
                }
            }

            // (5) X1: the UCT prediction against H^q(C_I (x) Z/ell^k), k = 1 .. K - 1
            {
                uint32_t ok = 0, tot = 0;
                for (U32 k = 1; k < K; ++k) {
                    const auto pred = integral_model::PredictedLengths(G, ell, k);
                    const auto got = M.cohomology_lengths(integral_model::RingContext(ell, k), n);
                    const bool eq = (pred == got);
                    F.check(eq, "(5) X1 n=" + S(n) + " " + ring_name(ell, k) + ": UCT prediction != lengths");
                    ++tot;
                    ok += eq ? 1 : 0;
                }
                std::cout << "    X1 (consistency): UCT prediction = lengths of H^q(C_I (x) Z/" << ell << "^k), k = 1.."
                          << K - 1 << ": " << ok << " / " << tot << "\n";
            }
        }
        // X1 at the Dickson primes: C_I and line 1
        for (const auto& [ell, cnt] : core ? integral_model::DicksonPrimes(n)
                                           : std::vector<std::pair<uint32_t, uint32_t>>{}) {
            (void)cnt;
            const integral_model::ModelIntegralCohomology Gd = integral_model::AssembleFromModel(M, n, {ell}, 10007);
            uint32_t ok = 0, tot = 0;
            for (U32 k = 1; k <= 3; ++k) {
                const auto pred = integral_model::PredictedLengths(Gd, ell, k);
                const auto got = M.cohomology_lengths(integral_model::RingContext(ell, k), n);
                const bool eq = (pred == got);
                F.check(eq, "(5) X1 n=" + S(n) + " " + ring_name(ell, k) + ": UCT != C_I lengths");
                ++tot;
                ok += eq ? 1 : 0;
                if (k >= 2) {
                    auto c = build_line1(n, ell, k, {});
                    const auto L1 = c->coh->H_lengths();
                    bool eq1 = L1.size() >= 3;
                    for (U32 q = 0; eq1 && q <= 2; ++q) eq1 = (L1[q] == pred[q]);
                    F.check(eq1, "(5) X1 n=" + S(n) + " " + ring_name(ell, k) + ": UCT != line-1 pipeline lengths");
                    ++tot;
                    ok += eq1 ? 1 : 0;
                }
            }
            std::cout << "  n=" << n << " Dickson prime " << ell << ": X1 (consistency), UCT = C_I lengths (k = 1..3) and = "
                         "line-1 pipeline lengths (k = 2, 3): "
                      << ok << " / " << tot << "\n";
        }
    }
    std::cout << "  sweep: arms (3) and (5) also ran at " << sweep_cells << " further n <= 20 (odd n included)\n";
    F.check(x6_nonzero_E > 0, "(3c) X6 is vacuous: E_q induced zero everywhere");
    F.check(x6_neg_caught > 0, "(3c) the X6 negative control (T_2 E_q vs E_q T_3) was never refused");
    F.check(x7_chain_noncommuting > 0, "(3a) X7 is vacuous at the cochain level (every pair commutes on the nose)");
    std::cout << "  non-vacuity: E_q non-zero on cohomology in " << x6_nonzero_E << " (n, ring, q) cells; the mixed "
              << "control T_2 E_q - E_q T_3 refused in " << x6_neg_caught << "; cochain-level T_p T_p' != T_p' T_p in "
              << x7_chain_noncommuting << " (ring, q, pair) cases (X7 holds on cohomology only)\n";

    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "HeckeIntegralActionValidation: " << F.checks << " checks, " << F.n << " failing, " << secs << " s\n";
    std::cout << (F.n == 0 ? "HeckeIntegralActionValidation: PASSED\n" : "HeckeIntegralActionValidation: FAILED\n");
    return F.n;
}
