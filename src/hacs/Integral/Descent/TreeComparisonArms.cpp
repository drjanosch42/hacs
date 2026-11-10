// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// DESCENT D5: the arms (M), (C), (N) of HeckeIsotropyModelTreeValidation.  See TreeComparisonArms.h.
#include "TreeComparisonArms.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "DescentAssembly.h"
#include "../Model/IsotropyModelTree.h"
#include "../Model/TreeCochains.h"
#include "../../../ModArithmetic/DenseElim.h"
#include "../../../ModArithmetic/DenseView.h"
#include "../../../ModArithmetic/EngineCheck.h"
#include "../../../ModArithmetic/Kernels.h"

namespace integral_descent {

namespace {

using engine::DenseBuf;
using U32Ctx = engine::ModCtx<uint32_t>;
using integral_model::CoefficientModule;
using clk = std::chrono::steady_clock;

DenseBuf<uint32_t> mul(const U32Ctx& F, const DenseBuf<uint32_t>& A, const DenseBuf<uint32_t>& B) {
    ENGINE_CHECK(A.cols == B.rows);
    DenseBuf<uint32_t> C(A.rows, B.cols);
    if (A.rows && B.cols && A.cols) engine::kernels::gemm<uint32_t>(F, engine::cview(A), engine::cview(B), engine::view(C));
    return C;
}
uint32_t rank_of(const U32Ctx& F, const DenseBuf<uint32_t>& A) {
    if (A.rows == 0 || A.cols == 0) return 0;
    DenseBuf<uint32_t> W = A;
    return engine::elim::rref(F, engine::view(W)).rank;
}
// kernel of A (columns of the result span ker A)
DenseBuf<uint32_t> kernel_cols(const U32Ctx& F, const DenseBuf<uint32_t>& A) {
    DenseBuf<uint32_t> W = A;
    std::vector<uint32_t> piv;
    if (A.rows && A.cols) piv = engine::elim::rref(F, engine::view(W)).pivots;
    DenseBuf<uint32_t> K;
    if (A.rows == 0) {
        K = DenseBuf<uint32_t>(A.cols, A.cols);
        for (uint32_t i = 0; i < A.cols; ++i) K.at(i, i) = 1;
        return K;
    }
    const DenseBuf<uint32_t> rows = engine::elim::kernel_basis(F, engine::cview(W), piv);
    DenseBuf<uint32_t> T(rows.cols, rows.rows);
    for (uint32_t i = 0; i < rows.rows; ++i)
        for (uint32_t j = 0; j < rows.cols; ++j) T.at(j, i) = rows.at(i, j);
    return T;
}
DenseBuf<uint32_t> hcat(const DenseBuf<uint32_t>& A, const DenseBuf<uint32_t>& B) {
    const uint32_t r = A.cols ? A.rows : B.rows;
    DenseBuf<uint32_t> C(r, A.cols + B.cols);
    for (uint32_t i = 0; i < r; ++i) {
        for (uint32_t j = 0; j < A.cols; ++j) C.at(i, j) = A.at(i, j);
        for (uint32_t j = 0; j < B.cols; ++j) C.at(i, A.cols + j) = B.at(i, j);
    }
    return C;
}
DenseBuf<uint32_t> vcat(const DenseBuf<uint32_t>& A, const DenseBuf<uint32_t>& B) {
    const uint32_t c = A.rows ? A.cols : B.cols;
    DenseBuf<uint32_t> C(A.rows + B.rows, c);
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint32_t j = 0; j < c; ++j) C.at(i, j) = A.at(i, j);
    for (uint32_t i = 0; i < B.rows; ++i)
        for (uint32_t j = 0; j < c; ++j) C.at(A.rows + i, j) = B.at(i, j);
    return C;
}
DenseBuf<uint32_t> minus_identity(const U32Ctx& F, DenseBuf<uint32_t> A) {
    for (uint32_t i = 0; i < A.rows; ++i) A.at(i, i) = F.sub(A.at(i, i), 1u);
    return A;
}
DenseBuf<uint32_t> id_minus(const U32Ctx& F, const DenseBuf<uint32_t>& P) {
    DenseBuf<uint32_t> A(P.rows, P.cols);
    for (uint32_t i = 0; i < P.rows; ++i)
        for (uint32_t j = 0; j < P.cols; ++j) A.at(i, j) = F.sub(i == j ? 1u : 0u, P.at(i, j));
    return A;
}

// One cochain complex in degrees q - 1, q, q + 1 over F_p: d^{q-1}, d^q and (optionally) the projector of the cochains.
struct Cx {
    DenseBuf<uint32_t> dprev, d, P, Pprev;   // dprev: C^{q-1} -> C^q (empty at q = 0); P: projector on C^q (empty = id)
    uint32_t dim = 0;
};
DenseBuf<uint32_t> cocycles(const U32Ctx& F, const Cx& X) {
    DenseBuf<uint32_t> A = X.d;
    if (X.P.rows) A = vcat(A, id_minus(F, X.P));
    if (A.rows == 0) {
        DenseBuf<uint32_t> I(X.dim, X.dim);
        for (uint32_t i = 0; i < X.dim; ++i) I.at(i, i) = 1;
        return I;
    }
    return kernel_cols(F, A);
}
DenseBuf<uint32_t> coboundaries(const U32Ctx& F, const Cx& X) {
    if (X.dprev.rows == 0) return DenseBuf<uint32_t>(X.dim, 0);
    return X.Pprev.rows ? mul(F, X.dprev, X.Pprev) : X.dprev;
}
uint32_t hdim(const U32Ctx& F, const Cx& X) {
    const DenseBuf<uint32_t> Z = cocycles(F, X), B = coboundaries(F, X);
    return Z.cols - rank_of(F, B);
}
// A o B induces the identity on H^q(X): (A B - 1) Z in im B_X
bool inverse_on_H(const U32Ctx& F, const Cx& X, const DenseBuf<uint32_t>& A, const DenseBuf<uint32_t>& B) {
    const DenseBuf<uint32_t> Z = cocycles(F, X), Bx = coboundaries(F, X);
    if (Z.cols == 0) return true;
    const DenseBuf<uint32_t> W = minus_identity(F, mul(F, A, B));
    const DenseBuf<uint32_t> WZ = mul(F, W, Z);
    const uint32_t rb = rank_of(F, Bx);
    return rank_of(F, hcat(Bx, WZ)) == rb;
}

CoefficientModule<uint32_t> module_of(uint32_t N, uint32_t n, const U32Ctx& F) {
    return N == 1 ? integral_model::SymModule(F, n) : InducedSymModule(N, n, F);
}

DenseBuf<uint32_t> adopt(std::vector<uint32_t> a, uint32_t rows, uint32_t cols) {
    return DenseBuf<uint32_t>::adopt(std::move(a), rows, cols);
}
bool is_zero(const DenseBuf<uint32_t>& A) {
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint32_t j = 0; j < A.cols; ++j)
            if (A.at(i, j)) return false;
    return true;
}

// G_q = block diag over the cells x of C^q of rho(e_x)[R_x, J_x] (identity where e_x = 1): TreeLocalForm's d^q is
// pi_R(C^{q+1}) d^q B_J(C^q), so the composite in one basis is d^{q+1} G_{q+1}^{-1} d^q (BuildRedTree's G^{-1}, D2).
DenseBuf<uint32_t> middle_G(const U32Ctx& F, const integral_model::TreeLocalForm<uint32_t>& T, const integral_model::TreeModule<uint32_t>& M,
                            uint32_t q) {
    using namespace integral_model::tree_elt;
    std::vector<std::pair<const integral_model::ImageSelection*, integral_model::QGroupRingElt>> cells;
    if (T.ell == 2) {
        cells = {{&T.sel_S, one()}, {&T.sel_U, eU2()}};
        if (q >= 1) cells.push_back({&T.sel_E, one()});
    } else {
        ENGINE_CHECK(T.ell == 3);
        if (q == 0) cells = {{&T.sel_S, eS4()}, {&T.sel_E, ec()}};
        else if (q == 1) cells = {{&T.sel_U, ec()}, {&T.sel_E, ec()}};
        else cells = {{&T.sel_U, ec()}};
    }
    const uint32_t c = T.cochain_rank(q);
    std::vector<uint32_t> G(size_t(c) * c, 0);
    uint32_t o = 0;
    for (const auto& [sel, e] : cells) {
        integral_model::AddRhoBlock(F, G, c, o, o, M, e, *sel, *sel);
        o += sel->rank();
    }
    ENGINE_CHECK_MSG(o == c, "middle_G: the cell layout disagrees with the cochain rank");
    return DenseBuf<uint32_t>::adopt(std::move(G), c, c);
}
// G^{-1} B over F_p (G invertible: the selection is a rank profile over F_p)
DenseBuf<uint32_t> solve(const U32Ctx& F, const DenseBuf<uint32_t>& G, const DenseBuf<uint32_t>& B) {
    DenseBuf<uint32_t> W = hcat(G, B);
    const auto r = engine::elim::rref(F, engine::view(W));
    ENGINE_CHECK_MSG(r.rank >= G.rows && (G.rows == 0 || r.pivots[G.rows - 1] == G.rows - 1), "solve: G not invertible over F_p");
    DenseBuf<uint32_t> X(G.rows, B.cols);
    for (uint32_t i = 0; i < G.rows; ++i)
        for (uint32_t j = 0; j < B.cols; ++j) X.at(i, j) = W.at(i, G.cols + j);
    return X;
}

}  // namespace

TreeFormCheck CheckTreeFormAgainstChain(uint32_t ell, uint32_t N, uint32_t n, uint32_t p, uint32_t q_max) {
    using namespace integral_model;
    const uint32_t qell = ell == 2 ? 3u : (ell == 3 ? 4u : 12u);
    ENGINE_CHECK_MSG(qell % p != 0, "CheckTreeFormAgainstChain: p must be prime to q_ell");
    if (ell >= 5) q_max = std::min<uint32_t>(q_max, 1);
    TreeFormCheck out;
    const U32Ctx F(p);
    const TreeModule<uint32_t> TM = N == 1 ? TreeSymModule(F, n) : TreeInducedSymModule(N, n, F);
    const CoefficientModule<uint32_t> M = module_of(N, n, F);
    const uint32_t D = M.dim;
    const uint32_t qt = ell >= 5 ? 1u : q_max + 1;   // d^0 .. d^{qt-1} addressable
    const TreeLocalForm<uint32_t> T = MakeTreeLocalForm(ell, F, TM, qt);
    std::vector<DenseBuf<uint32_t>> d;
    std::vector<uint32_t> rk;
    for (uint32_t q = 0; q < qt; ++q) {
        d.push_back(adopt(T.differential(F, TM, q), T.cochain_rank(q + 1), T.cochain_rank(q)));
        rk.push_back(rank_of(F, d.back()));
    }
    const std::string at = " (ell=" + std::to_string(ell) + " N=" + std::to_string(N) + " n=" + std::to_string(n) + " F_" + std::to_string(p) + ")";
    auto fail = [&](bool& flag, const std::string& what) {
        flag = false;
        if (out.first.empty()) out.first = what + at;
    };
    for (uint32_t q = 0; q + 1 < qt; ++q) {
        ++out.checks;
        if (!is_zero(mul(F, d[q + 1], solve(F, middle_G(F, T, TM, q + 1), d[q]))))
            fail(out.dd_zero, "d^" + std::to_string(q + 1) + " G^-1 d^" + std::to_string(q) + " != 0");
    }
    const TreeModelChain L(ell, q_max + 2);
    for (uint32_t q = 0; q <= q_max; ++q) {
        Cx X;
        X.d = L.coboundary(F, M, q);
        X.P = L.projector(F, M, q);
        X.dim = L.rank(q) * D;
        if (q >= 1) {
            X.dprev = L.coboundary(F, M, q - 1);
            X.Pprev = L.projector(F, M, q - 1);
        }
        const uint32_t cq = T.cochain_rank(q);
        ++out.checks;
        if (rank_of(F, X.P) != cq) fail(out.ranks, "c_" + std::to_string(q) + " != rank of the chain model's projector");
        const uint32_t rq = q < qt ? rk[q] : 0u, rp = q >= 1 ? rk[q - 1] : 0u;
        const int64_t hT = int64_t(cq) - rq - rp;
        const uint32_t hL = hdim(F, X);
        ++out.checks;
        if (hT != int64_t(hL))
            fail(out.dims, "dim H^" + std::to_string(q) + " of the form " + std::to_string(hT) + " != the chain model's " + std::to_string(hL));
    }
    return out;
}

bool TreeFieldDimsEqual(uint32_t ell, uint32_t n, uint32_t p, const std::vector<uint32_t>& h, uint32_t q_max, std::string& why) {
    using namespace integral_model;
    const U32Ctx F(p);
    const TreeModule<uint32_t> TM = TreeSymModule(F, n);
    const TreeLocalForm<uint32_t> T = MakeTreeLocalForm(ell, F, TM, q_max + 1);
    uint32_t prev = 0;
    for (uint32_t q = 0; q <= q_max; ++q) {
        const uint32_t r = rank_of(F, adopt(T.differential(F, TM, q), T.cochain_rank(q + 1), T.cochain_rank(q)));
        const int64_t dim = int64_t(T.cochain_rank(q)) - r - prev;
        const int64_t want = q < h.size() ? int64_t(h[q]) : 0;
        if (dim != want) {
            why = "dim H^" + std::to_string(q) + " = " + std::to_string(dim) + " != " + std::to_string(want) + " (ell=" + std::to_string(ell) +
                  " n=" + std::to_string(n) + " F_" + std::to_string(p) + ")";
            return false;
        }
        prev = r;
    }
    return true;
}

void RunTreeComparisonArms(uint32_t q_top, bool slow, TreeCheckFn check) {
    using namespace integral_model;
    const auto t0 = clk::now();
    // (M) the tree models
    for (uint32_t ell : {0u, 2u, 3u, 5u, 7u}) {
        const TreeCertificates c = TreeModelChain(ell, q_top + 2).certify();
        check(c.all(), "(M) def:isomodeltree: the " + (ell ? "ell=" + std::to_string(ell) + " local" : std::string("free")) +
                           " tree model: d o d = 0, well defined, the lift over the boundary -- " + c.first_failure);
    }
    // (M) (3) and (4) on the code's Iso
    const IsotropyModel desc(q_top);
    const IntegralTreeComparison cmp(desc, q_top);
    const IsoTreeCertificates ic = cmp.certify();
    check(ic.tables, "(M) (4)(a): r_T, G_T, i_T on the 19 representatives (chain maps, r i = id, id - i r = dG + Gd) -- " + ic.first_failure);
    check(ic.iso_is_code_model, "(M) the Iso of (4) is the code's IsotropyModel (LocalModel(desc, 0) == IsotropyModel)");
    check(ic.phi_chain && ic.psi_chain, "(M) Phi^Z_T, Psi^Z_T built by (L) are chain maps (degree <= " + std::to_string(q_top) + ")");
    check(ic.phi_over_r && ic.psi_over_iota, "(M) Phi^Z_T over r_T o cmpI, Psi^Z_T over i_T o eps_T");
    check(ic.integral, "(M) Phi^Z_T, Psi^Z_T, H_T, H_X are integral");
    check(ic.H_T && ic.H_X, "(M) (L'): d H_T + H_T d = Phi Psi - id, d H_X + H_X d = id - Psi Phi");
    std::vector<uint32_t> ells = {2, 3, 5};
    if (slow) ells.push_back(7);
    for (uint32_t ell : ells) {
        const auto r = cmp.check_ell(ell);
        check(r.phi_psi_strict, "(M) ell=" + std::to_string(ell) + ": (3) Phi^T_ell Psi^T_ell = id strictly -- " + r.first_failure);
        check(r.phi_T_chain && r.psi_T_chain, "(M) ell=" + std::to_string(ell) + ": Phi_T, Psi_T of (4)(d) are chain maps -- " + r.first_failure);
        check(r.H_ell && r.K && r.K_prime,
              "(M) ell=" + std::to_string(ell) + ": H_ell, K, K' homotopy identities of (4)(c), (d) -- " + r.first_failure);
    }
    std::cout << "  (M) tree models certified; (4)(a) tables; Phi^Z, Psi^Z, H_T, H_X, and per ell H_ell, K, K' to degree " << q_top
              << " (" << ic.checked << " chain-level checks); " << std::chrono::duration<double>(clk::now() - t0).count() << " s" << std::endl;

    // (C) cohomology over F_p: dims equal and Phi^*, Psi^* mutually inverse on H^q, q <= q_top - 1
    std::vector<std::pair<uint32_t, uint32_t>> cells = {{1, 0}, {1, 4}, {1, 12}, {2, 2}, {3, 0}, {5, 0}};
    if (slow) cells = {{1, 0}, {1, 2}, {1, 4}, {1, 6}, {1, 12}, {1, 20}, {2, 0}, {2, 2}, {3, 0}, {3, 2}, {4, 0}, {5, 0}, {7, 2}};
    uint32_t n_checks = 0;
    const TreeModelChain& Tr = cmp.tree();
    const LocalModel& Iso = cmp.iso();
    for (const auto& [N, n] : cells) {
        for (uint32_t p : {2u, 3u, 5u}) {
            const U32Ctx F(p);
            const auto M = module_of(N, n, F);
            const uint32_t D = M.dim;
            // the free tree against C_I: Phi^Z*, Psi^Z*
            for (uint32_t q = 0; q + 1 <= q_top - 1; ++q) {
                Cx XI, XT;
                XI.d = Iso.coboundary(F, M, q);
                XT.d = Tr.coboundary(F, M, q);
                XI.dim = Iso.rank(q) * D;
                XT.dim = Tr.rank(q) * D;
                if (q >= 1) {
                    XI.dprev = Iso.coboundary(F, M, q - 1);
                    XT.dprev = Tr.coboundary(F, M, q - 1);
                }
                const auto Ph = PullbackMap(F, M, cmp.phi(), q, Iso.rank(q), Tr.rank(q));   // C_T -> C_I
                const auto Ps = PullbackMap(F, M, cmp.psi(), q, Tr.rank(q), Iso.rank(q));   // C_I -> C_T
                const uint32_t hI = hdim(F, XI), hT = hdim(F, XT);
                check(hI == hT, "(C) dim H^" + std::to_string(q) + " C_I == C_T over F_" + std::to_string(p) + " at " + M.name);
                check(inverse_on_H(F, XI, Ph, Ps) && inverse_on_H(F, XT, Ps, Ph),
                      "(C) Phi^Z*, Psi^Z* mutually inverse on H^" + std::to_string(q) + " over F_" + std::to_string(p) + " at " + M.name);
                n_checks += 2;
            }
            // the ell-local tree against C_I: Phi_T*, Psi_T* (p prime to q_ell)
            for (uint32_t ell : {2u, 3u, 5u}) {
                const uint32_t qell = ell == 2 ? 3u : (ell == 3 ? 4u : 12u);
                if (qell % p == 0) continue;
                const TreeModelChain L(ell, q_top + 1);
                const TreeMap PhT = cmp.phi_T(ell), PsT = cmp.psi_T(ell);
                for (uint32_t q = 0; q + 1 <= q_top - 1; ++q) {
                    Cx XI, XL;
                    XI.d = Iso.coboundary(F, M, q);
                    XI.dim = Iso.rank(q) * D;
                    XL.d = L.coboundary(F, M, q);
                    XL.P = L.projector(F, M, q);
                    XL.dim = L.rank(q) * D;
                    if (q >= 1) {
                        XI.dprev = Iso.coboundary(F, M, q - 1);
                        XL.dprev = L.coboundary(F, M, q - 1);
                        XL.Pprev = L.projector(F, M, q - 1);
                    }
                    const auto Ph = PullbackMap(F, M, PhT, q, Iso.rank(q), L.rank(q));   // C_T,ell -> C_I
                    const auto Ps = PullbackMap(F, M, PsT, q, L.rank(q), Iso.rank(q));   // C_I -> C_T,ell
                    const uint32_t hI = hdim(F, XI), hL = hdim(F, XL);
                    const std::string at = " H^" + std::to_string(q) + " over F_" + std::to_string(p) + " at " + M.name + ", ell=" + std::to_string(ell);
                    check(hI == hL, "(C) dim C_I == dim C_T,ell:" + at);
                    check(inverse_on_H(F, XI, Ph, Ps) && inverse_on_H(F, XL, Ps, Ph), "(C) Phi_T*, Psi_T* mutually inverse on" + at);
                    n_checks += 2;
                    if (q == 1 && N == 1 && n == 4) {
                        // (6) the hook: T = 2 * id on C_I transports to 2 Psi_T* Phi_T*
                        DenseBuf<uint32_t> T2(XI.dim, XI.dim);
                        for (uint32_t i = 0; i < XI.dim; ++i) T2.at(i, i) = 2 % p;
                        const auto Tt = TransportToTree(F, M, cmp, ell, q, T2);
                        const auto ref = mul(F, Ps, Ph);
                        bool same = Tt.rows == ref.rows && Tt.cols == ref.cols;
                        for (uint32_t i = 0; same && i < ref.rows; ++i)
                            for (uint32_t j = 0; same && j < ref.cols; ++j) same = Tt.at(i, j) == F.mul(2 % p, ref.at(i, j));
                        check(same, "(C) (6) the transport hook: Psi_T* (2 id) Phi_T* == 2 Psi_T* Phi_T* at" + at);
                    }
                }
            }
        }
    }
    std::cout << "  (C) " << cells.size() << " cells x F_2, F_3, F_5: dim H^q(C_I) == dim H^q(C_T), == dim H^q(C_T,ell), Phi*, Psi* mutually "
              << "inverse on H^q, q <= " << q_top - 2 << " (" << n_checks << " checks); "
              << std::chrono::duration<double>(clk::now() - t0).count() << " s" << std::endl;

    // (N) negative controls: a mutated table is caught (an exception from the lifting, or a failing certificate)
    struct Mut {
        const char* what;
        void (*apply)(RetractionTables&);
    };
    const Mut muts[] = {
        {"a sign of r_T ({v2,v3} -> +S e)", [](RetractionTables& t) { t.r1[{2, 3}] = integral_model::tree_elt::S(); }},
        {"a value of r_T (v0 -> v_S)", [](RetractionTables& t) { t.r0[{0}] = std::make_pair('S', integral_model::tree_elt::one()); }},
        {"a value of G_T (G(v1) = 0)", [](RetractionTables& t) { t.G[{1}] = {}; }},
        {"a value of i_T (E -> {v0,v4})", [](RetractionTables& t) { t.iota['E'] = {0, 4}; }},
    };
    for (const Mut& m : muts) {
        RetractionTables t = DefaultRetractionTables();
        m.apply(t);
        bool caught = false;
        try {
            const IntegralTreeComparison bad(desc, std::min<uint32_t>(q_top, 4), t);
            caught = !bad.certify().all();
        } catch (const std::logic_error&) {
            caught = true;
        }
        check(caught, std::string("(N) the mutation is caught: ") + m.what);
    }
    std::cout << "  (N) four mutations of the (4)(a) tables are each caught" << std::endl;

    if (slow) {
        // (S) the coefficient sizes (the re-referee's N2, C-D6_referee.md: "finite but not periodic", linear growth in q; a remark, not a
        // statement of [C]).  Checked: (4) built to degree 17 is certified, Psi^Z_T and H_T keep coefficients in {-1, 0, 1}, and the
        // growth of Phi^Z_T and H_X is at most linear (max |coeff| <= q).  The per-degree maxima are printed (DESCENT5_report.md compares
        // them with N2's table).
        const auto t1 = clk::now();
        const IsotropyModel desc17(17);
        const IntegralTreeComparison big(desc17, 17);
        auto maxc = [](const TreeMap& m, uint32_t q) {
            int64_t x = 0;
            for (const auto& ch : m.value.at(q))
                for (const auto& [c, v] : ch)
                    for (const auto& kv : v.terms) x = std::max<int64_t>(x, kv.second.num < 0 ? -kv.second.num : kv.second.num);
            return x;
        };
        bool ok = big.certify().all();
        std::ostringstream o;
        std::ostringstream per;
        for (uint32_t q = 1; q <= 16; ++q) {
            ok = ok && maxc(big.psi(), q) <= 1 && maxc(big.H_T(), q) <= 1;
            per << (q > 1 ? "," : "") << maxc(big.phi(), q) << "/" << maxc(big.H_X(), q);
        }
        std::cout << "  (S) max |coeff| of Phi^Z / H_X per degree 1..16: " << per.str() << std::endl;
        for (uint32_t q = 1; q <= 16; ++q) ok = ok && maxc(big.phi(), q) <= q && maxc(big.H_X(), q) <= q;
        o << "Phi^Z(16)=" << maxc(big.phi(), 16) << " H_X(16)=" << maxc(big.H_X(), 16) << " Psi^Z(16)=" << maxc(big.psi(), 16)
          << " H_T(16)=" << maxc(big.H_T(), 16);
        check(ok, "(S) (4) to degree 17 certified; Psi^Z, H_T in {-1, 0, 1}; Phi^Z, H_X grow at most linearly (" + o.str() + ")");
        std::cout << "  (S) (4) built to degree 17 and certified; max |coeff| " << o.str() << " (N2's table: 7, 9, 1, 1 -- see DESCENT5_report.md); "
                  << std::chrono::duration<double>(clk::now() - t1).count() << " s" << std::endl;
    }
}

}  // namespace integral_descent
