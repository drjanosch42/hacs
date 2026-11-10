// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 11, WP-6 -- see HeckeDicksonThetaHecke.h.

#include "HeckeDicksonThetaHecke.h"

#include "../Hecke/HeckeModuleDecomposition.h"
#include "../Hecke/SubspaceOps.h"
#include "../../ModArithmetic/EngineCheck.h"

#include <algorithm>
#include <sstream>

namespace hecke_ext {

namespace {

inline uint32_t mulm(uint32_t a, uint32_t b, uint32_t p) {
    return static_cast<uint32_t>((static_cast<uint64_t>(a) * b) % p);
}
inline uint32_t subm(uint32_t a, uint32_t b, uint32_t p) { return a >= b ? a - b : a + p - b; }
inline uint32_t powm(uint64_t b, uint64_t e, uint32_t p) {
    uint64_t r = 1 % p;
    b %= p;
    while (e) {
        if (e & 1u) r = (r * b) % p;
        b = (b * b) % p;
        e >>= 1;
    }
    return static_cast<uint32_t>(r);
}

ThetaMat Identity(uint32_t d, uint32_t p) {
    ThetaMat I(d, d, p);
    for (uint32_t i = 0; i < d; ++i) I.at(i, i) = 1u % p;
    return I;
}

ThetaMat MinusScalar(const ThetaMat& A, uint32_t c) {
    ENGINE_CHECK_MSG(A.rows == A.cols, "MinusScalar: square matrix expected");
    ThetaMat B = A;
    for (uint32_t i = 0; i < A.rows; ++i) B.at(i, i) = subm(B.at(i, i), c % A.p, A.p);
    return B;
}

ThetaMat Power(const ThetaMat& A, uint32_t e) {
    ThetaMat R = Identity(A.rows, A.p), B = A;
    while (e) {
        if (e & 1u) R = ThetaMul(R, B);
        e >>= 1;
        if (e) B = ThetaMul(B, B);
    }
    return R;
}

ThetaMat VStack(const std::vector<ThetaMat>& parts, uint32_t cols, uint32_t p) {
    uint32_t rows = 0;
    for (const ThetaMat& P : parts) {
        ENGINE_CHECK_MSG(P.cols == cols, "VStack: column counts differ");
        rows += P.rows;
    }
    ThetaMat S(rows, cols, p);
    uint32_t r0 = 0;
    for (const ThetaMat& P : parts) {
        for (uint32_t i = 0; i < P.rows; ++i)
            for (uint32_t j = 0; j < cols; ++j) S.at(r0 + i, j) = P.at(i, j);
        r0 += P.rows;
    }
    return S;
}

ThetaMat HCat(const ThetaMat& A, const ThetaMat& B) {
    ENGINE_CHECK_MSG(A.rows == B.rows, "HCat: row counts differ");
    ThetaMat C(A.rows, A.cols + B.cols, A.p);
    for (uint32_t i = 0; i < A.rows; ++i) {
        for (uint32_t j = 0; j < A.cols; ++j) C.at(i, j) = A.at(i, j);
        for (uint32_t j = 0; j < B.cols; ++j) C.at(i, A.cols + j) = B.at(i, j);
    }
    return C;
}

// The (t, s) block of a flat (N d_r) x (N d_c) matrix as a ThetaMat; `nonzero`
// reports whether any entry is non-zero.
ThetaMat FlatBlock(const FlatMatrix& F, uint32_t t, uint32_t s, uint32_t dr, uint32_t dc, bool& nonzero) {
    ThetaMat B(dr, dc, F.get_p());
    nonzero = false;
    for (uint32_t i = 0; i < dr; ++i)
        for (uint32_t j = 0; j < dc; ++j) {
            const uint32_t v = F.get_entry(t * dr + i, s * dc + j);
            B.at(i, j) = v;
            if (v) nonzero = true;
        }
    return B;
}

std::string Cellname(const DicksonThetaCell& c) {
    return "(" + std::to_string(c.n()) + "," + std::to_string(c.ell()) + ")";
}

} // namespace

// ===========================================================================
// ThetaMat helpers
// ===========================================================================
ThetaMat ThetaFromMatrix(const MatrixT<uint32_t>& M) {
    ThetaMat T(M.rows, M.cols, M.p);
    for (uint32_t r = 0; r < M.rows; ++r)
        for (uint32_t c = 0; c < M.cols; ++c) T.at(r, c) = M.data[r][c] % M.p;
    return T;
}

ThetaMat ThetaScaled(const ThetaMat& A, uint32_t c) {
    ThetaMat B = A;
    for (uint32_t& v : B.a) v = mulm(v, c % A.p, A.p);
    return B;
}

bool ThetaEqual(const ThetaMat& A, const ThetaMat& B) {
    return A.rows == B.rows && A.cols == B.cols && A.a == B.a;
}

ThetaMat ThetaGeneralisedEigenspace(const std::vector<ThetaMat>& ops, const std::vector<uint32_t>& eig) {
    ENGINE_CHECK_MSG(!ops.empty() && ops.size() == eig.size(), "ThetaGeneralisedEigenspace: ops / eigenvalues");
    const uint32_t D = ops[0].rows, p = ops[0].p;
    std::vector<ThetaMat> parts;
    for (size_t i = 0; i < ops.size(); ++i) parts.push_back(Power(MinusScalar(ops[i], eig[i]), std::max<uint32_t>(D, 1)));
    return ThetaKernelBasis(VStack(parts, D, p));
}

uint32_t ThetaIntersectionDim(const ThetaMat& A, const ThetaMat& B) {
    const uint32_t ra = ThetaRank(A), rb = ThetaRank(B);
    if (ra == 0 || rb == 0) return 0;
    return ra + rb - ThetaRank(HCat(A, B));
}

std::vector<uint64_t> ThetaDefaultPrimes(uint32_t ell) {
    std::vector<uint64_t> ps;
    for (uint64_t p : { 2u, 3u, 5u, 7u })
        if (p != ell) ps.push_back(p);
    ps.push_back(ell);
    std::sort(ps.begin(), ps.end());
    return ps;
}

// ===========================================================================
// ThetaHeckeRun
// ===========================================================================
const ThetaMat& ThetaHeckeRun::op(ThetaTerm t, uint64_t p, uint32_t q) const {
    const auto it = A.find(p);
    ENGINE_CHECK_MSG(it != A.end() && q < 3, "ThetaHeckeRun::op: prime not run or degree out of range");
    return it->second[static_cast<uint32_t>(t)][q];
}

uint32_t ThetaHeckeRun::failures() const {
    uint32_t f = 0;
    for (const auto& c : checks)
        if (!c.ok) ++f;
    return f;
}

std::string ThetaHeckeRun::first_failure() const {
    for (const auto& c : checks)
        if (!c.ok) return c.name + (c.detail.empty() ? "" : ": " + c.detail);
    return "";
}

ThetaHeckeRun RunThetaHecke(const DicksonThetaCell& cell, const ThetaHeckeOptions& opts) {
    ThetaHeckeRun R;
    R.n = cell.n();
    R.ell = cell.ell();
    R.m = cell.m();
    R.primes = opts.primes;
    const uint32_t ell = R.ell;
    const std::string tag = Cellname(cell);
    auto check = [&](const std::string& name, bool ok, const std::string& detail = std::string()) {
        R.checks.push_back({ tag + " " + name, ok, ok ? std::string() : detail });
    };
    ENGINE_CHECK_MSG(cell.certificate().code == 0, "RunThetaHecke: the WP-5 certificate of the cell failed");

    const ThetaMat& th0 = cell.theta(0);
    const ThetaMat& th1 = cell.theta(1);
    const ThetaMat& pi0 = cell.pi(0);
    const ThetaMat& pi1 = cell.pi(1);
    const ThetaMat& de0 = cell.delta(0);
    R.rank_delta = ThetaRank(de0);
    R.rank_theta1 = ThetaRank(th1);
    const uint32_t N0 = cell.K().get_num_q_simplices(0);

    // The classes of the Dickson monomials of degree d on H^0 of a Sym term.
    auto dickson_classes = [&](ThetaTerm t, uint32_t d) {
        std::vector<std::pair<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>>> out;
        for (const auto& ab : DicksonExponents(d, ell)) {
            const std::vector<uint32_t> f = DicksonMonomial(ab.first, ab.second, ell);
            std::vector<uint32_t> y(static_cast<size_t>(N0) * (d + 1));
            for (uint32_t s = 0; s < N0; ++s) std::copy(f.begin(), f.end(), y.begin() + static_cast<size_t>(s) * (d + 1));
            out.push_back({ ab, cell.class_of(t, 0, y) });
        }
        return out;
    };
    auto apply = [&](const ThetaMat& A, const std::vector<uint32_t>& v) {
        std::vector<uint32_t> w(A.rows, 0);
        for (uint32_t i = 0; i < A.rows; ++i) {
            uint64_t acc = 0;
            for (uint32_t k = 0; k < A.cols; ++k) acc = (acc + static_cast<uint64_t>(A.at(i, k)) * v[k]) % ell;
            w[i] = static_cast<uint32_t>(acc);
        }
        return w;
    };

    for (const uint64_t p : opts.primes) {
        DicksonThetaCell::HeckeTriple tr = cell.hecke_triple(p);
        const std::string pt = "p=" + std::to_string(p);
        const bool is_u = (p == ell);
        const uint32_t detg = static_cast<uint32_t>(p % ell);   // det diag(1, p) mod ell

        // ---- (T) the datum is not keyed on n -------------------------------
        if (opts.tau_arm) {
            bool same_s = tr.cosets_n->num_cosets() == tr.cosets_m->num_cosets();
            for (uint32_t i = 1; same_s && i <= tr.cosets_n->num_cosets(); ++i)
                if (!(tr.cosets_n->s(i) == tr.cosets_m->s(i))) same_s = false;
            check("(T) " + pt + " s_i(n) == s_i(m)", same_s);
            bool same_tau = same_s;
            std::string where;
            for (uint32_t i = 1; same_tau && i <= tr.cosets_n->num_cosets(); ++i)
                for (uint32_t q = 0; same_tau && q < 3; ++q)
                    for (uint32_t s = 0; same_tau && s < cell.K().get_num_q_simplices(q); ++s) {
                        const auto& a = tr.tau_n->tau(i, q, s);
                        const auto& b = tr.tau_m->tau(i, q, s);
                        ++R.tau_slots_compared;
                        bool eq = a.size() == b.size();
                        for (size_t k = 0; eq && k < a.size(); ++k)
                            eq = a[k].coeff == b[k].coeff && a[k].gamma == b[k].gamma && a[k].sigma_idx == b[k].sigma_idx;
                        if (!eq) {
                            same_tau = false;
                            where = "i=" + std::to_string(i) + " q=" + std::to_string(q) + " sigma=" + std::to_string(s);
                        }
                    }
            check("(T) " + pt + " tau(n) == tau(m) on every slot", same_tau, where);
        }

        // ---- (F) cochain level ---------------------------------------------
        if (opts.cochain_arm) {
            const FlatMatrix& Lf = cell.L_fibre();
            const FlatMatrix& Pf = cell.pi_fibre();
            const ThetaMat L = [&] { ThetaMat X(Lf.get_rows(), Lf.get_cols(), ell);
                for (uint32_t i = 0; i < X.rows; ++i) for (uint32_t j = 0; j < X.cols; ++j) X.at(i, j) = Lf.get_entry(i, j);
                return X; }();
            const ThetaMat P = [&] { ThetaMat X(Pf.get_rows(), Pf.get_cols(), ell);
                for (uint32_t i = 0; i < X.rows; ++i) for (uint32_t j = 0; j < X.cols; ++j) X.at(i, j) = Pf.get_entry(i, j);
                return X; }();
            const uint32_t dn = cell.n() + 1, dm = cell.m() + 1, dq = ell + 1;
            for (uint32_t q = 0; q < 3; ++q) {
                const FlatMatrix& Tn = tr.op_n->get_Theta_flat_matrices().at(q);
                const FlatMatrix& Tm = tr.op_m->get_Theta_flat_matrices().at(q);
                const FlatMatrix& Tq = tr.op_q->get_Theta_flat_matrices().at(q);
                const uint32_t Nq = cell.K().get_num_q_simplices(q);
                bool ok_theta = true, ok_pi = true;
                std::string where;
                for (uint32_t t = 0; t < Nq && ok_theta && ok_pi; ++t)
                    for (uint32_t s = 0; s < Nq && ok_theta && ok_pi; ++s) {
                        bool nzn = false, nzm = false, nzq = false;
                        const ThetaMat Bn = FlatBlock(Tn, t, s, dn, dn, nzn);
                        const ThetaMat Bm = FlatBlock(Tm, t, s, dm, dm, nzm);
                        const ThetaMat Bq = FlatBlock(Tq, t, s, dq, dq, nzq);
                        if (nzm || nzn) {
                            const ThetaMat lhs = ThetaMul(Bm, L);
                            const ThetaMat rhs = ThetaScaled(ThetaMul(L, Bn), detg);
                            if (!ThetaEqual(lhs, rhs)) { ok_theta = false; where = "t=" + std::to_string(t) + " s=" + std::to_string(s); }
                        }
                        if (nzm || nzq) {
                            if (!ThetaEqual(ThetaMul(Bq, P), ThetaMul(P, Bm))) { ok_pi = false; where = "t=" + std::to_string(t) + " s=" + std::to_string(s); }
                        }
                    }
                check("(F) " + pt + " q=" + std::to_string(q) + " Theta_m (1 x L) = det(g) (1 x L) Theta_n on flat cochains", ok_theta, where);
                check("(F) " + pt + " q=" + std::to_string(q) + " Theta_Q (1 x pi) = (1 x pi) Theta_m on flat cochains", ok_pi, where);
            }
        }

        // ---- the matrices on cohomology --------------------------------------
        auto& slot = R.A[p];
        for (uint32_t q = 0; q < 3; ++q) {
            slot[0][q] = ThetaFromMatrix(tr.op_n->get_H_matrices().at(q));
            slot[1][q] = ThetaFromMatrix(tr.op_m->get_H_matrices().at(q));
            slot[2][q] = ThetaFromMatrix(tr.op_q->get_H_matrices().at(q));
        }
        const auto& An = slot[0];
        const auto& Am = slot[1];
        const auto& AQ = slot[2];
        const ThetaMat* th[2] = { &th0, &th1 };
        const ThetaMat* pim[2] = { &pi0, &pi1 };

        if (!is_u) {
            for (uint32_t q = 0; q < 2; ++q) {
                const std::string qs = " q=" + std::to_string(q);
                const ThetaMat lhs = ThetaMul(Am[q], *th[q]);
                const ThetaMat tA = ThetaMul(*th[q], An[q]);
                check("(E) " + pt + qs + " A_m theta = p theta A_n", ThetaEqual(lhs, ThetaScaled(tA, detg)));
                if (!tA.is_zero() && detg != 1u)
                    // (S13 WP-12, slot-11 m-5) Given the twisted identity this is a tautology (p X = X <=> (p-1) X = 0
                    // <=> X = 0, p != 1 mod ell): it does not pin the exponent by itself; the twisted check does.
                    check("(E) " + pt + qs + " the untwisted A_m theta = theta A_n FAILS (exponent pinned)", !ThetaEqual(lhs, tA));
                check("(E) " + pt + qs + " pi A_m = A_Q pi", ThetaEqual(ThetaMul(*pim[q], Am[q]), ThetaMul(AQ[q], *pim[q])));
            }
            check("(E) " + pt + " delta A_Q = p A_n delta",
                  ThetaEqual(ThetaMul(de0, AQ[0]), ThetaScaled(ThetaMul(An[1], de0), detg)));

            // (H0) T_p on the Dickson classes of Sym^n and Sym^m; H^0(Q_n).
            for (uint32_t which = 0; which < 2; ++which) {
                const ThetaTerm t = which == 0 ? ThetaTerm::SymN : ThetaTerm::SymM;
                const uint32_t d = which == 0 ? cell.n() : cell.m();
                for (const auto& cls : dickson_classes(t, d)) {
                    const uint32_t ev = mulm((1u + detg) % ell, powm(detg, cls.first.first, ell), ell);
                    std::vector<uint32_t> want = cls.second;
                    for (uint32_t& v : want) v = mulm(v, ev, ell);
                    check("(H0) " + pt + (which == 0 ? " Sym^n" : " Sym^m") + " T_p L^" + std::to_string(cls.first.first) + " D^" +
                              std::to_string(cls.first.second) + " = (1+p) p^a",
                          apply(slot[which][0], cls.second) == want);
                }
            }
            check("(H0) " + pt + " T_p = p + 1 on H^0(Q_n)",
                  ThetaEqual(AQ[0], ThetaScaled(Identity(AQ[0].rows, ell), (detg + 1u) % ell)));

            // (K) the kernel of theta on H^1 is the Eisenstein line (1 + p^{-1}).
            if (R.rank_delta == 1) {
                std::vector<uint32_t> v(de0.rows);
                for (uint32_t i = 0; i < de0.rows; ++i) v[i] = de0.at(i, 0);
                std::vector<uint32_t> want = v;
                const uint32_t ev = (1u + powm(detg, ell - 2, ell)) % ell;
                for (uint32_t& x : want) x = mulm(x, ev, ell);
                check("(K) " + pt + " T_p delta(1_n) = (1 + p^{-1}) delta(1_n)", apply(An[1], v) == want);
            }
        } else {
            for (uint32_t q = 0; q < 2; ++q) {
                const std::string qs = " q=" + std::to_string(q);
                check("(U)" + qs + " A_m(U) theta = 0", ThetaMul(Am[q], *th[q]).is_zero());
                check("(U)" + qs + " pi A_m(U) = A_Q(U) pi", ThetaEqual(ThetaMul(*pim[q], Am[q]), ThetaMul(AQ[q], *pim[q])));
            }
            check("(U) delta A_Q(U) = 0", ThetaMul(de0, AQ[0]).is_zero());
            check("(U) U = 0 on H^0(Q_n)", AQ[0].is_zero());
            for (uint32_t which = 0; which < 2; ++which) {
                const uint32_t d = which == 0 ? cell.n() : cell.m();
                const ThetaMat& U0 = slot[which][0];
                const uint32_t rk = ThetaRank(U0);
                (which == 0 ? R.u_rank_h0_n : R.u_rank_h0_m) = rk;
                const std::string ws = which == 0 ? " Sym^n" : " Sym^m";
                if (d == 0) {
                    check("(U)" + ws + " d=0: U = 1 on H^0(F_ell)", ThetaEqual(U0, Identity(U0.rows, ell)));
                    continue;
                }
                check("(U)" + ws + " rank U <= 1 on H^0", rk <= 1, "rank " + std::to_string(rk));
                check("(U)" + ws + " U^2 = 0 on H^0", ThetaMul(U0, U0).is_zero());
                const bool exception = (d == 120 && ell == 5);
                check("(U)" + ws + (exception ? " U != 0 on H^0 at (120, 5)" : " U = 0 on H^0"),
                      exception ? rk == 1 : U0.is_zero());
                if (exception) {
                    const auto cls = dickson_classes(which == 0 ? ThetaTerm::SymN : ThetaTerm::SymM, d);
                    std::vector<uint32_t> L20, L10D3, D6;
                    for (const auto& c : cls) {
                        if (c.first == std::make_pair(20u, 0u)) L20 = c.second;
                        if (c.first == std::make_pair(10u, 3u)) L10D3 = c.second;
                        if (c.first == std::make_pair(0u, 6u)) D6 = c.second;
                    }
                    std::vector<uint32_t> mL20 = L20;
                    for (uint32_t& x : mL20) x = subm(0, x, ell);
                    const std::vector<uint32_t> zero(U0.rows, 0);
                    check("(U) (120,5) U D^6 = -L^20", !D6.empty() && apply(U0, D6) == mL20);
                    check("(U) (120,5) U L^20 = 0", !L20.empty() && apply(U0, L20) == zero);
                    check("(U) (120,5) U L^10 D^3 = 0", !L10D3.empty() && apply(U0, L10D3) == zero);
                }
            }
        }
    }

    // ---- (B) the multiplicity inequality on primary blocks -----------------
    if (opts.block_arm) {
        std::vector<uint64_t> anemic;
        for (uint64_t p : opts.primes)
            if (p != ell) anemic.push_back(p);
        auto decompose = [&](uint32_t term) {
            std::vector<ThetaBlock> out;
            const uint32_t d = R.A.at(anemic.at(0))[term][1].rows;
            if (d == 0) return out;
            std::vector<std::pair<uint64_t, MatrixT<uint32_t>>> ops;
            for (uint64_t p : anemic) ops.push_back({ p, R.A.at(p)[term][1].to_matrix() });
            hecke::HeckeModuleDecomposition dec(ops, ell);
            dec.compute();
            for (const hecke::PrimaryBlock& b : dec.blocks()) {
                ThetaBlock tb;
                tb.basis = ThetaFromMatrix(b.basis_columns);
                tb.dim = b.dim;
                tb.residue_degree = b.residue_degree;
                out.push_back(std::move(tb));
            }
            return out;
        };
        if (!anemic.empty()) {
            R.blocks_n = decompose(0);
            R.blocks_m = decompose(1);
            const ThetaMat kerth = ThetaKernelBasis(th1);
            // The Eisenstein system (1 + p^{-1})_p on H^1_n.
            std::vector<ThetaMat> opsn;
            std::vector<uint32_t> eis;
            for (uint64_t p : anemic) {
                opsn.push_back(R.A.at(p)[0][1]);
                eis.push_back((1u + powm(p % ell, ell - 2, ell)) % ell);
            }
            const ThetaMat Eeis = cell.h(ThetaTerm::SymN, 1) ? ThetaGeneralisedEigenspace(opsn, eis) : ThetaMat(0, 0, ell);
            uint32_t sum_n = 0, sum_m = 0;
            for (const auto& b : R.blocks_n) sum_n += b.dim;
            for (const auto& c : R.blocks_m) sum_m += c.dim;
            check("(B) the blocks of H^1_n and H^1_m exhaust them",
                  sum_n == cell.h(ThetaTerm::SymN, 1) && sum_m == cell.h(ThetaTerm::SymM, 1));
            for (size_t bi = 0; bi < R.blocks_n.size(); ++bi) {
                const ThetaBlock& B = R.blocks_n[bi];
                const std::string bs = "block " + std::to_string(bi) + " (dim " + std::to_string(B.dim) + ")";
                const ThetaMat img = ThetaMul(th1, B.basis);
                const uint32_t rimg = ThetaRank(img);
                const uint32_t kdim = ThetaIntersectionDim(B.basis, kerth);
                check("(B) " + bs + " rank theta|_B = dim B - dim(B cap ker theta)", rimg + kdim == B.dim);
                if (kdim > 0) {
                    const bool in_eis = Eeis.cols > 0 && ThetaSpanContains(Eeis, B.basis);
                    check("(B) " + bs + " B cap ker theta != 0 only on the (1 + p^{-1}) block", in_eis);
                    check("(B) " + bs + " dim(B cap ker theta) <= rank delta", kdim <= R.rank_delta);
                }
                if (rimg == 0) continue;
                uint32_t hits = 0, target_dim = 0;
                for (const auto& C : R.blocks_m)
                    if (ThetaSpanContains(C.basis, img)) { ++hits; target_dim = C.dim; }
                check("(B) " + bs + " theta(B) lies in exactly one block of H^1_m", hits == 1, std::to_string(hits) + " blocks");
                if (hits == 1)
                    check("(B) " + bs + " dim C >= dim B - dim(B cap ker theta) (thm:thetales(5))", target_dim >= rimg);
            }
        }
    }
    return R;
}

} // namespace hecke_ext
