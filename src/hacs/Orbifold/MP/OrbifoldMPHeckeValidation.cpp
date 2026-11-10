// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// OrbifoldMPHeckeValidation -- slot 22, MP-4 (MP4_predictions.md P7).  See the header.
#include "OrbifoldMPHeckeValidation.h"

#include "CochainMaps.h"
#include "Comparison.h"
#include "MP2Common.h"
#include "MPFill.h"
#include "MPHecke.h"
#include "Models.h"
#include "ModuleCochains.h"
#include "Nerve.h"
#include "../../EquivariantCohomologyOnK.h"
#include "../../LocalSystemSL2Z.h"
#include "../../LocalSystemSimplicialComplex.h"
#include "../../SimplicialCohomology.h"
#include "../../Hecke/HeckeCosetTables.h"
#include "../../Hecke/HeckeOperatorOnK.h"
#include "../../Hecke/HeckeTauTable.h"
#include "../../Hecke/SL2ZTessellationTables.h"
#include "../../Integral/Model/IsotropyModel.h"
#include "../../Integral/Model/PeriodicityClass.h"
#include "../../Integral/Model/StrictLift.h"
#include "../../../ModArithmetic/DenseBuf.h"
#include "../../../ModArithmetic/DenseElim.h"
#include "../../../ModArithmetic/DenseView.h"
#include "../../../ModArithmetic/FlatMatrix.h"
#include "../../../ModArithmetic/ModCtx.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct MP4OracleRow {
    uint32_t q, ell, n, p, dim;
    int64_t coef[16];
};

#include "MP4Oracle.inc"

using engine::DenseBuf;
using engine::ModCtx;
using orbifold_mp::Descriptor;
using orbifold_mp::InternedMPSet;
using orbifold_mp::MPHeckeDatum;
using orbifold_mp::MPSide;
using orbifold_mp::PhiLift;
using orbifold_mp::SourceH;
using orbifold_mp::SparseRows;
using orbifold_mp::Vec;

const MP4OracleRow* OracleRow(uint32_t q, uint32_t ell, uint32_t n, uint32_t p) {
    for (const auto& r : kMP4Oracle)
        if (r.q == q && r.ell == ell && r.n == n && r.p == p) return &r;
    return nullptr;
}

std::string PolyStr(const std::vector<uint32_t>& c) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < c.size(); ++i) os << (i ? "," : "") << c[i];
    os << "]";
    return os.str();
}

// det(x I - A) over F_ell, coefficients low -> high (Hessenberg reduction; a port of mp4_oracle.py's).
std::vector<uint32_t> Charpoly(std::vector<std::vector<uint64_t>> H, uint32_t l) {
    const size_t n = H.size();
    auto inv = [&](uint64_t a) {
        uint64_t r = 1, b = a % l, e = l - 2;
        while (e) {
            if (e & 1) r = r * b % l;
            b = b * b % l;
            e >>= 1;
        }
        return r;
    };
    for (size_t k = 0; k + 2 < n; ++k) {
        size_t pr = n;
        for (size_t i = k + 1; i < n; ++i)
            if (H[i][k] % l) {
                pr = i;
                break;
            }
        if (pr == n) continue;
        if (pr != k + 1) {
            std::swap(H[pr], H[k + 1]);
            for (auto& r : H) std::swap(r[pr], r[k + 1]);
        }
        const uint64_t iv = inv(H[k + 1][k]);
        for (size_t i = k + 2; i < n; ++i) {
            const uint64_t f = H[i][k] * iv % l;
            if (!f) continue;
            for (size_t c = 0; c < n; ++c) H[i][c] = (H[i][c] + l * l - f * H[k + 1][c] % l) % l;
            for (auto& r : H) r[k + 1] = (r[k + 1] + f * r[i]) % l;
        }
    }
    using Poly = std::vector<uint64_t>;
    auto pmul = [&](const Poly& a, const Poly& b) {
        Poly o(a.size() + b.size() - 1, 0);
        for (size_t i = 0; i < a.size(); ++i)
            for (size_t j = 0; j < b.size(); ++j) o[i + j] = (o[i + j] + a[i] * b[j]) % l;
        return o;
    };
    auto padd = [&](const Poly& a, const Poly& b) {
        Poly o(std::max(a.size(), b.size()), 0);
        for (size_t i = 0; i < o.size(); ++i) o[i] = ((i < a.size() ? a[i] : 0) + (i < b.size() ? b[i] : 0)) % l;
        return o;
    };
    std::vector<Poly> P{{1}};
    for (size_t k = 1; k <= n; ++k) {
        Poly pk = pmul(Poly{(l - H[k - 1][k - 1] % l) % l, 1}, P[k - 1]);
        uint64_t prod = 1;
        for (size_t i = 1; i < k; ++i) {
            prod = prod * H[k - i][k - i - 1] % l;
            const uint64_t t = prod * H[k - i - 1][k - 1] % l;
            Poly term = pmul(Poly{t}, P[k - i - 1]);
            for (auto& x : term) x = (l - x) % l;
            pk = padd(pk, term);
        }
        P.push_back(pk);
    }
    std::vector<uint32_t> out(P[n].begin(), P[n].end());
    out.resize(n + 1, 0);
    return out;
}

uint64_t RankRows(const ModCtx<uint32_t>& ctx, const std::vector<Vec>& rows, uint32_t ncols) {
    if (rows.empty()) return 0;
    DenseBuf<uint32_t> A(static_cast<uint32_t>(rows.size()), ncols);
    for (uint32_t i = 0; i < rows.size(); ++i)
        for (uint32_t j = 0; j < ncols; ++j) A.at(i, j) = rows[i][j];
    return orbifold_mp::DenseLength(ctx, A);
}

// The columns of a CSR matrix (rows = target coordinates), as vectors.
std::vector<Vec> ColumnsOf(const SparseRows& A) {
    std::vector<Vec> cols(A.ncols, Vec(A.rows(), 0));
    for (uint64_t r = 0; r < A.rows(); ++r)
        for (uint64_t t = A.start[r]; t < A.start[r + 1]; ++t) cols[A.col[t]][r] = A.val[t];
    return cols;
}

std::vector<Vec> ColumnsOfDense(const DenseBuf<uint32_t>& A, uint32_t m) {
    std::vector<Vec> cols(A.cols, Vec(A.rows, 0));
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint32_t j = 0; j < A.cols; ++j) cols[j][i] = A.at(i, j) % m;
    return cols;
}

// Coordinates (column convention) of the images Y_i in the basis b of H^q, modulo the coboundary columns Bc (field).
// Returns false if a Y_i is not in span(b) + span(Bc).
bool Coordinates(const ModCtx<uint32_t>& F, const std::vector<Vec>& b, const std::vector<Vec>& Bc, const std::vector<Vec>& Y,
                 std::vector<std::vector<uint64_t>>* A) {
    const uint32_t h = static_cast<uint32_t>(b.size());
    if (h == 0) {
        A->clear();
        return true;
    }
    const uint32_t dim = static_cast<uint32_t>(b[0].size());
    const uint32_t nc = h + static_cast<uint32_t>(Bc.size());
    DenseBuf<uint32_t> M(dim, nc), R(dim, static_cast<uint32_t>(Y.size()));
    for (uint32_t j = 0; j < h; ++j)
        for (uint32_t i = 0; i < dim; ++i) M.at(i, j) = b[j][i];
    for (uint32_t j = 0; j < Bc.size(); ++j)
        for (uint32_t i = 0; i < dim; ++i) M.at(i, h + j) = Bc[j][i];
    for (uint32_t j = 0; j < Y.size(); ++j)
        for (uint32_t i = 0; i < dim; ++i) R.at(i, j) = Y[j][i];
    std::vector<std::vector<uint32_t>> sol;
    try {
        sol = engine::elim::solve_multi_rhs(F, engine::cview(M), engine::cview(R));
    } catch (const std::runtime_error&) {
        return false;
    }
    A->assign(h, std::vector<uint64_t>(Y.size(), 0));
    for (uint32_t i = 0; i < Y.size(); ++i)
        for (uint32_t j = 0; j < h; ++j) (*A)[j][i] = sol[i][j];
    return true;
}

Vec Sub(const Vec& a, const Vec& b, uint32_t m) {
    Vec o(a.size());
    for (size_t i = 0; i < a.size(); ++i) o[i] = static_cast<uint32_t>((a[i] + m - b[i] % m) % m);
    return o;
}

Vec Flatten(const FlatMatrix& f) {
    Vec v(static_cast<size_t>(f.get_rows()) * f.get_cols());
    for (uint32_t s = 0; s < f.get_rows(); ++s)
        for (uint32_t c = 0; c < f.get_cols(); ++c) v[static_cast<size_t>(s) * f.get_cols() + c] = f.get_entry(s, c);
    return v;
}

std::string Mod(uint32_t ell, uint32_t k) { return k == 1 ? "F_" + std::to_string(ell) : "Z/" + std::to_string(orbifold_mp::PowU(ell, k)); }

struct Tri {
    std::string name;
    const Descriptor* D;
    std::unique_ptr<InternedMPSet> S;
    std::unique_ptr<PhiLift> Phi;
    std::map<uint32_t, std::unique_ptr<MPHeckeDatum>> H;
};

}  // namespace

uint32_t OrbifoldMPHeckeValidation() {
    const auto t0 = std::chrono::steady_clock::now();
    orbifold_mp::MPTally T;
    uint32_t oracle_cells = 0, oracle_agree = 0;
    std::cout << "OrbifoldMPHeckeValidation: MP-4, MP4_predictions.md P7 (model=mp)\n";
    const std::vector<uint32_t> primes = {2, 3, 5, 7};

    const orbifold_mp::MPModels MM = orbifold_mp::BuildM7AndRZ();
    const integral_model::IsotropyModel CI(3);
    const integral_model::PeriodicityClass P(CI);
    std::map<uint32_t, std::unique_ptr<integral_model::StrictLiftBundle>> bundle;
    for (uint32_t p : primes) {
        bundle[p] = std::make_unique<integral_model::StrictLiftBundle>(p, 3);
        bool same = true;
        for (uint32_t q = 0; q <= 3; ++q) same = same && bundle[p]->model().rank(q) == CI.rank(q);
        T.check(same, "P7.0 the bundle's Iso_. has the layout of C_I (p=" + std::to_string(p) + ")");
    }

    // ---- P7.1: the twisted chain datum on the MP resolution, built by the MP filling, certified ------------------
    Tri M7{"M7", &MM.m7, nullptr, nullptr, {}}, RZ{"R_Z", &MM.rzd, nullptr, nullptr, {}};
    M7.S = std::make_unique<InternedMPSet>(MM.m7, 3);
    RZ.S = std::make_unique<InternedMPSet>(MM.rzd, 2);
    M7.Phi = std::make_unique<PhiLift>(*M7.S, MM.m7, P, 2);
    RZ.Phi = std::make_unique<PhiLift>(*RZ.S, MM.rzd, P, 2);
    for (Tri* t : {&M7, &RZ}) T.check(t->Phi->certify().all(), "P7.1 Phi (MP-2) certified on " + t->name);
    for (Tri* t : {&M7, &RZ})
        for (uint32_t p : primes) {
            if (t == &RZ && p > 3) continue;
            const uint32_t ktop = (t == &M7) ? 2 : 1;
            t->H[p] = std::make_unique<MPHeckeDatum>(*t->S, *t->D, bundle[p]->cosets(), p, ktop);
            const MPHeckeDatum& H = *t->H[p];
            const orbifold_mp::MPHeckeCertificate c = H.certify();
            uint64_t gens = 0;
            for (uint32_t k = 0; k <= ktop; ++k) gens += t->S->count_nondegenerate(k);
            gens *= (p + 1);
            const auto& fs = H.fill_stats();
            std::cout << "  HECKE_DATUM model=mp tri=" << t->name << " p=" << p << " m=" << H.m() << " degrees=0.." << ktop
                      << " generators=" << c.generators << " terms=";
            for (uint32_t k = 0; k <= ktop; ++k) std::cout << (k ? "/" : "") << H.terms(k);
            std::cout << " max_abs_coeff=" << H.max_abs_coefficient() << " fills=" << fs.fills << " x0=" << fs.x0_fills
                      << " x1=" << fs.x1_fills << " x1_max_tris=" << fs.x1_max_triangles << " x1_max_rounds=" << fs.x1_max_rounds
                      << " bfs_max=" << fs.bfs_max_visited << " tot_max=" << fs.max_tot_terms << " build_s=" << std::fixed
                      << std::setprecision(2) << H.build_seconds() << std::defaultfloat << " certify: aug=" << c.aug
                      << " chain_map=" << c.chain_map << " schreier=" << c.schreier << "\n";
            T.check(c.all(), "P7.1 tau certified (T1 Schreier, T2 chain map, T3 aug) on " + t->name + " p=" + std::to_string(p) + ": " +
                                 c.first_failure);
            T.check(c.generators == gens, "P7.1 tau certified on every generator s_j e_s of degree <= " + std::to_string(ktop) + " (" +
                                              t->name + " p=" + std::to_string(p) + ", " + std::to_string(c.generators) + ")");
        }

    // ---- P7.2: T_p a cochain map -----------------------------------------------------------------------------------
    {
        uint32_t ok = 0, total = 0;
        const std::vector<std::pair<uint32_t, uint32_t>> rings = {{2, 1}, {3, 1}, {5, 1}, {7, 1}, {2, 2}, {3, 2}};
        for (Tri* t : {&M7, &RZ})
            for (const auto& [ell, k] : rings) {
                const uint32_t m = orbifold_mp::PowU(ell, k);
                for (uint32_t n = 0; n <= 4; ++n) {
                    const uint32_t qmax = (t == &M7) ? 1 : 0;
                    std::vector<SparseRows> d;
                    for (uint32_t q = 0; q <= qmax; ++q) d.push_back(orbifold_mp::Coboundary(*t->S, q, n, m));
                    for (const auto& [p, H] : t->H) {
                        std::vector<SparseRows> Tq;
                        for (uint32_t q = 0; q <= qmax + 1; ++q) Tq.push_back(H->Operator(q, n, ell, k));
                        for (uint32_t q = 0; q <= qmax; ++q) {
                            const bool eq = orbifold_mp::CommutesExactly(Tq[q + 1], d[q], d[q], Tq[q], m);
                            T.check(eq, "P7.2 T_{q+1} d^q = d^q T_q, " + t->name + " " + Mod(ell, k) + " n=" + std::to_string(n) +
                                            " p=" + std::to_string(p) + " q=" + std::to_string(q));
                            ok += eq ? 1 : 0;
                            ++total;
                        }
                    }
                }
            }
        std::cout << "  HECKE_COCHAIN_MAP model=mp (M7: q<=1, R_Z: q=0) x (F_2,F_3,F_5,F_7,Z/4,Z/9) x n<=4 x p: " << ok << "/" << total
                  << " exact\n";
    }

    // ---- P7.3: T_p = cor o (conj_g o res), and conj_g o res a cochain map into C(S; Coind) --------------------------
    {
        uint32_t ok = 0, total = 0;
        for (const auto& [ell, k] : std::vector<std::pair<uint32_t, uint32_t>>{{3, 1}, {2, 2}, {5, 1}})
            for (uint32_t n : {1u, 2u}) {
                const uint32_t m = orbifold_mp::PowU(ell, k);
                std::vector<SparseRows> dM;
                for (uint32_t q = 0; q <= 1; ++q) dM.push_back(orbifold_mp::Coboundary(*M7.S, q, n, m));
                for (const auto& [p, H] : M7.H) {
                    const orbifold_mp::ModuleRep C = H->Coind(n, ell, k);
                    std::vector<SparseRows> A, dC;
                    for (uint32_t q = 0; q <= 2; ++q) A.push_back(H->CgRes(q, n, ell, k));
                    for (uint32_t q = 0; q <= 1; ++q) dC.push_back(orbifold_mp::CoboundaryRep(*M7.S, q, C));
                    for (uint32_t q = 0; q <= 1; ++q) {
                        const std::string at = " " + Mod(ell, k) + " n=" + std::to_string(n) + " p=" + std::to_string(p) + " q=" +
                                               std::to_string(q);
                        const bool f = orbifold_mp::SpEqual(H->Operator(q, n, ell, k),
                                                            orbifold_mp::SpMul(H->Cor(q, n, ell, k), A[q], m));
                        const bool c = orbifold_mp::CommutesExactly(A[q + 1], dM[q], dC[q], A[q], m);
                        T.check(f, "P7.3 T_q = Cor_q o CgRes_q exactly" + at);
                        T.check(c, "P7.3 CgRes (conj_g o res, label factor rho(g)) a cochain map into C(S; Coind)" + at);
                        ok += (f ? 1 : 0) + (c ? 1 : 0);
                        total += 2;
                    }
                }
            }
        std::cout << "  HECKE_FACTOR model=mp tri=M7 T = cor o conj_g o res, conj_g o res cochain map: " << ok << "/" << total << " exact\n";
    }

    // ---- P7.4 + P7.5: T_p o Phi^* = Phi^* o Theta^ on H^q; the intrinsic matrices vs the oracle -------------------
    struct Cell {
        Tri* t;
        uint32_t ell, k, n, qtop;
        bool intrinsic;
    };
    std::vector<Cell> cells;
    for (uint32_t ell : {2u, 3u, 5u, 7u})
        for (uint32_t n = 0; n <= 12; ++n) cells.push_back({&M7, ell, 1, n, (ell <= 3 && n <= 3) ? 2u : 1u, true});
    for (uint32_t ell : {2u, 3u})
        for (uint32_t k : {2u, 3u})
            for (uint32_t n : {0u, 2u}) cells.push_back({&M7, ell, k, n, 2, false});
    for (uint32_t ell : {2u, 3u, 5u, 7u})
        for (uint32_t n : {0u, 2u, 10u}) cells.push_back({&RZ, ell, 1, n, 1, true});
    uint32_t inter_ok = 0, inter_total = 0, mat_ok = 0, mat_total = 0;
    // F3 (audit): cells where the operator acts on a nonzero space (dim H^q(MP) > 0), by family.
    uint32_t inter_nt = 0, inter_nt_m7fl = 0, inter_total_m7fl = 0, orc_nt_m7 = 0, orc_total_m7 = 0, orc_nt_rz = 0, orc_total_rz = 0;
    std::map<std::string, std::string> charpoly_lines;
    for (const Cell& c : cells) {
        const ModCtx<uint32_t> ctx = orbifold_mp::RingCtx(c.ell, c.k);
        const uint32_t m = orbifold_mp::PowU(c.ell, c.k);
        const MPSide side(*c.t->S, c.n, c.ell, c.k, c.qtop);
        for (uint32_t q = 0; q <= c.qtop; ++q) {
            const DenseBuf<uint32_t> dq = CI.coboundary(ctx, c.n, q);
            DenseBuf<uint32_t> dp;
            if (q) dp = CI.coboundary(ctx, c.n, q - 1);
            const SourceH src = orbifold_mp::Source(ctx, dq, q ? &dp : nullptr, nullptr);
            const SparseRows PhiQ = c.t->Phi->Pullback(q, c.n, c.ell, c.k);
            // an H^q basis of C_I (field cells): greedy from the cocycle generators modulo B^q(C_I)
            std::vector<Vec> basis, Bci;
            if (c.intrinsic) {
                if (q) Bci = ColumnsOfDense(dp, m);
                std::vector<Vec> cur = Bci;
                uint64_t r = RankRows(ctx, cur, static_cast<uint32_t>(dq.cols));
                for (const Vec& z : src.cocycles) {
                    cur.push_back(z);
                    const uint64_t r2 = RankRows(ctx, cur, static_cast<uint32_t>(dq.cols));
                    if (r2 > r) {
                        basis.push_back(z);
                        r = r2;
                    } else {
                        cur.pop_back();
                    }
                }
            }
            for (const auto& [p, H] : c.t->H) {
                const std::string at = " " + c.t->name + " " + Mod(c.ell, c.k) + " n=" + std::to_string(c.n) + " p=" + std::to_string(p) +
                                       " q=" + std::to_string(q);
                const SparseRows Tq = H->Operator(q, c.n, c.ell, c.k);
                const DenseBuf<uint32_t> Th = bundle[p]->lift().theta_hat(ctx, c.n, q);
                std::vector<Vec> diffs;
                for (const Vec& z : src.cocycles) {
                    const Vec a = orbifold_mp::SpMatVec(Tq, orbifold_mp::SpMatVec(PhiQ, z, m), m);
                    const Vec b = orbifold_mp::SpMatVec(PhiQ, orbifold_mp::DenseMatVec(Th, z, m), m);
                    diffs.push_back(Sub(a, b, m));
                }
                const auto img = side.image(q, diffs);
                const bool inter = img.cocycles && img.image() == 0;
                T.check(inter, "P7.4 T_p Phi^* = Phi^* Theta^_g on H^q" + at + " (cocycles " + std::to_string(img.cocycles) +
                                   ", class length of the differences " + std::to_string(img.image()) + ")");
                inter_ok += inter ? 1 : 0;
                ++inter_total;
                {
                    const bool nt = side.lenH(q) > 0;
                    inter_nt += nt ? 1 : 0;
                    if (c.t == &M7 && c.k == 1) {
                        ++inter_total_m7fl;
                        inter_nt_m7fl += nt ? 1 : 0;
                    }
                }
                if (!c.intrinsic || q > 1) continue;
                // P7.5: the intrinsic matrix of T_p on H^q(MP) in the basis Phi^* z_i, and Theta^'s on H^q(C_I) in z_i.
                std::vector<Vec> b, Tb, Thz;
                for (const Vec& z : basis) {
                    b.push_back(orbifold_mp::SpMatVec(PhiQ, z, m));
                    Thz.push_back(orbifold_mp::DenseMatVec(Th, z, m));
                }
                for (const Vec& x : b) Tb.push_back(orbifold_mp::SpMatVec(Tq, x, m));
                std::vector<Vec> Bmp;
                if (q) Bmp = ColumnsOf(side.d.at(q - 1));
                std::vector<std::vector<uint64_t>> A, Aci;
                const bool solved = Coordinates(ctx, b, Bmp, Tb, &A);
                const bool solved_ci = Coordinates(ctx, basis, Bci, Thz, &Aci);
                const std::vector<uint32_t> cp = solved ? Charpoly(A, c.ell) : std::vector<uint32_t>{};
                const MP4OracleRow* O = OracleRow(q, c.ell, c.n, p);
                bool orc = O && solved && O->dim == basis.size() && static_cast<int64_t>(O->dim) == side.lenH(q) && cp.size() == O->dim + 1;
                if (orc)
                    for (uint32_t i = 0; i <= O->dim; ++i) {
                        int64_t w = O->coef[i] % static_cast<int64_t>(c.ell);
                        if (w < 0) w += c.ell;
                        orc = orc && static_cast<int64_t>(cp[i]) == w;
                    }
                ++oracle_cells;
                if (c.t == &M7) {
                    ++orc_total_m7;
                    orc_nt_m7 += (O && O->dim > 0) ? 1 : 0;
                } else {
                    ++orc_total_rz;
                    orc_nt_rz += (O && O->dim > 0) ? 1 : 0;
                }
                oracle_agree += orc ? 1 : 0;
                T.check(orc, "P7.5 charpoly of T_p on H^q(MP) (intrinsic) = mp4_oracle.out" + at + " (MP " + PolyStr(cp) + ", dim " +
                                 std::to_string(basis.size()) + ")");
                const bool same = solved && solved_ci && A == Aci;
                T.check(same, "P7.5 the intrinsic matrix on H^q(MP) = Theta^'s matrix on H^q(C_I) in the basis Phi^* z_i" + at);
                mat_ok += same ? 1 : 0;
                ++mat_total;
                std::ostringstream key;
                key << c.t->name << " F_" << c.ell << " q=" << q << " p=" << p;
                charpoly_lines[key.str()] += " " + std::to_string(c.n) + ":" + PolyStr(cp);
                // tau / Eichler--Shimura (n = 10, ell in {5, 7}): (x - tau(p))^2 (x - 1 - p^11) mod ell.
                if (q == 1 && c.n == 10 && c.ell >= 5 && c.t == &M7) {
                    const std::map<uint32_t, int64_t> tau = {{2, -24}, {3, 252}, {5, 4830}, {7, -16744}};
                    const int64_t l = c.ell;
                    auto md = [&](int64_t v) { return ((v % l) + l) % l; };
                    int64_t e = 1;
                    for (int i = 0; i < 11; ++i) e = e * static_cast<int64_t>(p) % l;
                    e = md(1 + e);
                    const int64_t tp = md(tau.at(p));
                    // (x - t)^2 (x - e) = x^3 - (2t + e) x^2 + (t^2 + 2 t e) x - t^2 e
                    const std::vector<uint32_t> want = {static_cast<uint32_t>(md(-tp * tp % l * e)),
                                                        static_cast<uint32_t>(md(tp * tp + 2 * tp * e)),
                                                        static_cast<uint32_t>(md(-(2 * tp + e))), 1u};
                    T.check(cp == want, "P7.5 tau/ES: charpoly T_p on H^1(Sym^10 F_ell) = (x - tau(p))^2 (x - 1 - p^11)" + at);
                    std::cout << "  TAU_ES model=mp tri=M7 ring=F_" << c.ell << " n=10 p=" << p << " tau(p)=" << tau.at(p)
                              << " charpoly=" << PolyStr(cp) << " (x-tau)^2(x-1-p^11)=" << PolyStr(want) << "\n";
                }
            }
        }
    }
    for (const auto& [k, v] : charpoly_lines) std::cout << "  HECKE_CHARPOLY model=mp " << k << " n:charpoly" << v << "\n";
    std::cout << "  HECKE_INTERTWINE model=mp T_p Phi^* = Phi^* Theta^_g: " << inter_ok << "/" << inter_total
              << "; intrinsic matrix = Theta^ matrix: " << mat_ok << "/" << mat_total << "; nontrivial=" << inter_nt << " (M7 over F_ell: "
              << inter_nt_m7fl << "/" << inter_total_m7fl << ")\n";
    T.check(inter_nt_m7fl == 240 && inter_total_m7fl == 448,
            "P7.4 nontrivial cells (dim H^q(MP) > 0) over F_ell on M7 = 240 of 448 (audit F3), got " + std::to_string(inter_nt_m7fl) +
                " of " + std::to_string(inter_total_m7fl));

    // ---- P7.6: the shipped Bredon pipeline's T_p, transported by eps^* = Phi^* eps_I^* -------------------------------
    {
        const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
        const hecke::SL2ZTessellationTables model(K);
        struct BCell {
            uint32_t ell, n;
        };
        std::vector<BCell> bc;
        for (uint32_t ell : {5u, 7u})
            for (uint32_t n = 0; n <= 12; n += 2) bc.push_back({ell, n});
        bc.push_back({2, 0});
        bc.push_back({2, 2});
        bc.push_back({3, 0});
        bc.push_back({3, 2});
        uint32_t ok = 0, total = 0, nontrivial = 0, nt_big = 0, tot_big = 0, nt_small = 0, tot_small = 0;
        for (const BCell& c : bc) {
            const ModCtx<uint32_t> ctx(c.ell);
            std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels;
            for (const auto& datum : GenerateSL2ZRelationsWithGroupData(K, c.n, c.ell)) rels.push_back(datum.rel);
            LocalSystemSimplicialComplex lssc(K, rels, c.n + 1, c.ell);
            EquivariantCohomologyOnK coh(&lssc);
            coh.compute_cohomology();
            const auto dims = coh.get_H_eq_dimensions();
            const auto& Hb = coh.get_H_eq_bases_fm();
            const MPSide side(*M7.S, c.n, c.ell, 1, 1);
            for (uint32_t p : primes) {
                if (p == c.ell) continue;
                std::vector<MatrixT<uint32_t>> A;
                std::string err;
                try {
                    hecke::HeckeCosetTables cosets(p, 1, c.ell, c.n);
                    hecke::HeckeTauTable tau(K, model, cosets, c.ell, c.n);
                    hecke::HeckeOperatorOnK op(&coh, &cosets, &tau);
                    op.compute();
                    A = op.get_H_matrices();
                } catch (const std::exception& ex) {
                    err = ex.what();
                }
                for (uint32_t q = 0; q <= 1; ++q) {
                    const std::string at = " F_" + std::to_string(c.ell) + " n=" + std::to_string(c.n) + " p=" + std::to_string(p) +
                                           " q=" + std::to_string(q);
                    if (!err.empty() || A.size() <= q) {
                        T.check(false, "P7.6 the shipped Bredon T_p" + at + ": " + (err.empty() ? "no matrix" : err));
                        ++total;
                        continue;
                    }
                    const uint32_t h = q < dims.size() ? dims[q] : 0;
                    const DenseBuf<uint32_t> E = CI.comparison(ctx, c.n, q);
                    const SparseRows PhiQ = M7.Phi->Pullback(q, c.n, c.ell, 1);
                    const SparseRows Tq = M7.H.at(p)->Operator(q, c.n, c.ell, 1);
                    std::vector<Vec> e;
                    for (uint32_t i = 0; i < h; ++i)
                        e.push_back(orbifold_mp::SpMatVec(PhiQ, orbifold_mp::DenseMatVec(E, Flatten(Hb[q][i]), c.ell), c.ell));
                    std::vector<Vec> diffs;
                    for (uint32_t i = 0; i < h; ++i) {
                        Vec rhs(e.empty() ? 0 : e[0].size(), 0);
                        for (uint32_t j = 0; j < h; ++j) {
                            const uint64_t a = A[q].data[j][i] % c.ell;
                            for (size_t r = 0; r < rhs.size(); ++r) rhs[r] = static_cast<uint32_t>((rhs[r] + a * e[j][r]) % c.ell);
                        }
                        diffs.push_back(Sub(orbifold_mp::SpMatVec(Tq, e[i], c.ell), rhs, c.ell));
                    }
                    const auto rk = side.image(q, e);
                    const auto img = side.image(q, diffs);
                    const bool good = rk.cocycles && rk.image() == h && img.cocycles && img.image() == 0;
                    T.check(good, "P7.6 T_p eps^* = eps^* T_p^Bredon on H^q" + at + " (Bredon dim " + std::to_string(h) + ", rank eps^* " +
                                      std::to_string(rk.image()) + ", difference class length " + std::to_string(img.image()) + ")");
                    ok += good ? 1 : 0;
                    ++total;
                    nontrivial += h > 0 ? 1 : 0;
                    (c.ell >= 5 ? tot_big : tot_small) += 1;
                    (c.ell >= 5 ? nt_big : nt_small) += h > 0 ? 1 : 0;
                }
            }
        }
        // F3: F_2, F_3 is the audit's figure (15 one-eigenvalue checks of 24); F_5, F_7 measured 51 of 84 (the audit estimated 48: 33, not 36, vacuous).
        T.check(nt_small == 15 && tot_small == 24 && nt_big == 51 && tot_big == 84,
                "P7.6 nontrivial checks (Bredon dim h > 0): F_2,F_3 15 of 24 and F_5,F_7 51 of 84 (audit F3), got " + std::to_string(nt_small) +
                    " of " + std::to_string(tot_small) + " and " + std::to_string(nt_big) + " of " + std::to_string(tot_big));
        std::cout << "  HECKE_BREDON model=mp tri=M7 (F_5, F_7: eps^* iso, n even <= 12; F_2, F_3: on im eps^*, n in {0, 2}) "
                     "T_p eps^* = eps^* T_p^shipped: "
                  << ok << "/" << total << "; nontrivial=" << nontrivial << " (F_5, F_7: " << nt_big << "/" << tot_big << "; F_2, F_3: " << nt_small << "/" << tot_small
                  << ")\n";
    }

    // ---- P7.7: negative controls -----------------------------------------------------------------------------------
    {
        const MPHeckeDatum& H2 = *M7.H.at(2);
        // (a), (b): the label factor dropped / a label factor mismatched with the datum (gT for g): not a cochain map at F_3, n = 2, q = 0.
        for (const auto& [f, name] : std::vector<std::pair<MPHeckeDatum::Factor, std::string>>{
                 {MPHeckeDatum::Factor::Dropped, "label factor rho(g) dropped"}, {MPHeckeDatum::Factor::WrongRep, "label factor mismatched with the datum (gT for g)"}}) {
            const uint32_t m = 3;
            const SparseRows d0 = orbifold_mp::Coboundary(*M7.S, 0, 2, m), d1 = orbifold_mp::Coboundary(*M7.S, 1, 2, m);
            const SparseRows T0 = H2.Operator(0, 2, 3, 1, f), T1 = H2.Operator(1, 2, 3, 1, f), T2 = H2.Operator(2, 2, 3, 1, f);
            const bool c0 = orbifold_mp::CommutesExactly(T1, d0, d0, T0, m);
            const bool c1 = orbifold_mp::CommutesExactly(T2, d1, d1, T1, m);
            // on H^1 at F_5, n = 10: the intertwining with Theta^ fails
            const ModCtx<uint32_t> ctx(5);
            const DenseBuf<uint32_t> dq = CI.coboundary(ctx, 10, 1), dp = CI.coboundary(ctx, 10, 0);
            const SourceH src = orbifold_mp::Source(ctx, dq, &dp, nullptr);
            const SparseRows PhiQ = M7.Phi->Pullback(1, 10, 5, 1);
            const SparseRows Tq = H2.Operator(1, 10, 5, 1, f);
            const DenseBuf<uint32_t> Th = bundle[2]->lift().theta_hat(ctx, 10, 1);
            const MPSide side(*M7.S, 10, 5, 1, 1);
            std::vector<Vec> diffs;
            for (const Vec& z : src.cocycles)
                diffs.push_back(Sub(orbifold_mp::SpMatVec(Tq, orbifold_mp::SpMatVec(PhiQ, z, 5), 5),
                                    orbifold_mp::SpMatVec(PhiQ, orbifold_mp::DenseMatVec(Th, z, 5), 5), 5));
            const auto img = side.image(1, diffs);
            const bool inter = img.cocycles && img.image() == 0;
            std::cout << "  NC P7.7" << (f == MPHeckeDatum::Factor::Dropped ? "a" : "b") << " model=mp " << name
                      << ": F_3 n=2 p=2 cochain map q=0:" << c0 << " q=1:" << c1 << "; F_5 n=10 p=2 H^1 T Phi^* = Phi^* Theta^:" << inter
                      << " (cocycles " << img.cocycles << ", difference length " << img.image() << ")"
                      << ((!c0 && !inter) ? " -> NC fires" : " -> NC SILENT") << "\n";
            T.check(!c0, "P7.7 NC " + name + ": T_1 d^0 != d^0 T_0 at F_3 n=2 p=2");
            T.check(!inter, "P7.7 NC " + name + ": T Phi^* != Phi^* Theta^ on H^1 at F_5 n=10 p=2");
        }
        // (a2) CgRes without rho(g) is not a cochain map into C(S; Coind).
        {
            const orbifold_mp::ModuleRep C = H2.Coind(2, 3, 1);
            const bool c = orbifold_mp::CommutesExactly(H2.CgRes(1, 2, 3, 1, true), orbifold_mp::Coboundary(*M7.S, 0, 2, 3),
                                                        orbifold_mp::CoboundaryRep(*M7.S, 0, C), H2.CgRes(0, 2, 3, 1, true), 3);
            std::cout << "  NC P7.7a2 model=mp CgRes without the label factor, F_3 n=2 p=2 q=0: cochain map=" << c
                      << (c ? " -> NC SILENT" : " -> NC fires") << "\n";
            T.check(!c, "P7.7 NC CgRes without rho(g) is not a cochain map into C(S; Coind) (F_3 n=2 p=2 q=0)");
        }
        // (c) a perturbed tau_1 fails the certificate.
        {
            MPHeckeDatum Hp = H2;
            Hp.perturb_for_negative_control(1, 1, 0);
            const auto cc = Hp.certify();
            std::cout << "  NC P7.7c model=mp perturbed tau_1(s_1 e_0) by + e_0: certify chain_map=" << cc.chain_map
                      << (cc.chain_map ? " -> NC SILENT" : " -> NC fires") << "\n";
            T.check(!cc.chain_map, "P7.7 NC a perturbed tau_1 fails the chain-map certificate");
        }
    }

    T.check(orc_nt_m7 == 220 && orc_total_m7 == 416,
            "P7.5 nontrivial oracle rows (dim H^q > 0) on M7 = 220 of 416 (audit F3), got " + std::to_string(orc_nt_m7) + " of " +
                std::to_string(orc_total_m7));
    std::cout << "ORACLE mp4_oracle.out cells=" << oracle_cells << " agree=" << oracle_agree << " nontrivial=" << (orc_nt_m7 + orc_nt_rz)
              << " (M7 " << orc_nt_m7 << "/" << orc_total_m7 << ", R_Z " << orc_nt_rz << "/" << orc_total_rz << ")" << " (sha256 " << kMP4OracleSha256 << ")\n";
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "ORBIFOLD_MP OrbifoldMPHeckeValidation checks=" << T.checks << " failing=" << T.failing << " wall_s=" << std::fixed
              << std::setprecision(2) << wall << std::defaultfloat << "\n";
    return T.failing == 0 ? 0 : 1;
}
