// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// hecke-extensions slot 13, WP-9 -- HeckeExt/HeckeCensusLattice.  See the header.
// ===========================================================================

#include "HeckeCensusLattice.h"

#include "GloverFactors.h"
#include "HeckeExtDriver.h"

#include "../Hecke/HeckeInvolution.h"
#include "../Hecke/HeckeModuleDecomposition.h"
#include "../Hecke/HeckeSettings.h"
#include "../Hecke/SubspaceOps.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/ScalarMod.h"

#include <algorithm>
#include <chrono>
#include <memory>
#include <sstream>

namespace hecke_ext {

namespace {

using Clock = std::chrono::steady_clock;
using hecke::ordinary::RingMat;

ExtDriverOptions LatticeDriverOptions(bool sign) {
    ExtDriverOptions o;
    o.with_cone = false;
    o.with_sign_split = sign;
    o.run_commutant = false;
    o.run_sturm = false;
    return o;
}

std::string OwnGroupOf(const IdealRecord& id) {
    for (const IdealGroupMatch& g : id.groups)
        if (g.own) return g.group;
    return std::string();
}

bool SameEntries(const ExtMatrix& a, const ExtMatrix& b) {
    if (a.rows != b.rows || a.cols != b.cols) return false;
    for (uint32_t i = 0; i < a.rows; ++i)
        for (uint32_t j = 0; j < a.cols; ++j)
            if (a.data[i][j] != b.data[i][j]) return false;
    return true;
}

uint64_t PowModU(uint64_t b, uint64_t e, uint64_t q) { return engine::scalar::powmod<uint64_t>(b, e, q); }

ExtMatrix ScalarShift(const ExtMatrix& T, uint32_t c) {   // T - c I
    ExtMatrix out = T;
    const uint32_t ell = T.p;
    for (uint32_t i = 0; i < T.rows; ++i) out.data[i][i] = (out.data[i][i] + ell - c % ell) % ell;
    return out;
}

bool IsZeroExt(const ExtMatrix& M) {
    for (const auto& r : M.data)
        for (uint32_t x : r)
            if (x) return false;
    return true;
}

// Independent columns of M spanning its column space (F_ell).
ExtMatrix ColBasis(const ExtMatrix& M) {
    const uint32_t ell = M.p;
    std::vector<std::vector<uint64_t>> A(M.rows, std::vector<uint64_t>(M.cols));
    for (uint32_t i = 0; i < M.rows; ++i)
        for (uint32_t j = 0; j < M.cols; ++j) A[i][j] = M.data[i][j];
    std::vector<uint32_t> piv;
    uint32_t r = 0;
    for (uint32_t j = 0; j < M.cols && r < M.rows; ++j) {
        uint32_t p = r;
        while (p < M.rows && A[p][j] == 0) ++p;
        if (p == M.rows) continue;
        std::swap(A[p], A[r]);
        const uint64_t iv = PowModU(A[r][j], ell - 2, ell);
        for (uint32_t c = j; c < M.cols; ++c) A[r][c] = A[r][c] * iv % ell;
        for (uint32_t i = 0; i < M.rows; ++i) {
            if (i == r || A[i][j] == 0) continue;
            const uint64_t f = A[i][j];
            for (uint32_t c = j; c < M.cols; ++c) A[i][c] = (A[i][c] + ell * ell - f * A[r][c] % ell) % ell;
        }
        piv.push_back(j);
        ++r;
    }
    ExtMatrix out(M.rows, static_cast<uint32_t>(piv.size()), ell);
    for (uint32_t k = 0; k < piv.size(); ++k)
        for (uint32_t i = 0; i < M.rows; ++i) out.data[i][k] = M.data[i][piv[k]];
    return out;
}

// Basis of span(B) meet span(U) (both full column rank), in ambient coordinates.
ExtMatrix Intersect(const ExtMatrix& B, const ExtMatrix& U) {
    const uint32_t ell = B.p;
    if (B.cols == 0 || U.cols == 0) return ExtMatrix(B.rows, 0, ell);
    const ExtMatrix K = hecke::KernelAsMatrix(hecke::HStack({B, U}, B.rows, ell));
    ExtMatrix top(B.cols, K.cols, ell);
    for (uint32_t i = 0; i < B.cols; ++i)
        for (uint32_t j = 0; j < K.cols; ++j) top.data[i][j] = K.data[i][j];
    if (K.cols == 0) return ExtMatrix(B.rows, 0, ell);
    return ColBasis(B * top);
}

// Loewy length of span(W0) under the nilpotent family X (ambient operators).
uint32_t Loewy(const ExtMatrix& W0, const std::vector<ExtMatrix>& X) {
    ExtMatrix W = ColBasis(W0);
    uint32_t L = 0;
    const uint32_t cap = W0.rows + 2;
    while (W.cols > 0) {
        ENGINE_CHECK_MSG(++L <= cap, "HeckeCensusLattice: the family is not nilpotent on the block");
        std::vector<ExtMatrix> parts;
        for (const ExtMatrix& x : X) parts.push_back(x * W);
        if (parts.empty()) break;
        W = ColBasis(hecke::HStack(parts, W0.rows, W0.p));
    }
    return L;
}

// The eigenvalue of t (residue degree 1): the a with t - a nilpotent; false if none.
bool Eigenvalue(const ExtMatrix& t, uint32_t& a) {
    const uint32_t ell = t.p, d = t.rows;
    if (d == 0) return false;
    if (d % ell != 0) {
        uint64_t tr = 0;
        for (uint32_t i = 0; i < d; ++i) tr = (tr + t.data[i][i]) % ell;
        const uint32_t c = static_cast<uint32_t>(tr * PowModU(d % ell, ell - 2, ell) % ell);
        if (hecke::IsNilpotent(ScalarShift(t, c), d)) { a = c; return true; }
        return false;
    }
    for (uint32_t c = 0; c < ell; ++c)
        if (hecke::IsNilpotent(ScalarShift(t, c), d)) { a = c; return true; }
    return false;
}

// dim of the joint generalized eigenspace of {q^b T_q - a_q} and whether each is 0 on it.
void JointPart(const std::map<uint64_t, ExtMatrix>& ops, uint32_t dim, uint32_t ell,
               const std::vector<uint64_t>& primes, const std::vector<uint32_t>& a, uint32_t b,
               uint32_t& out_dim, bool& scalar) {
    out_dim = 0;
    scalar = true;
    if (dim == 0) return;
    std::vector<ExtMatrix> M;
    for (size_t k = 0; k < primes.size(); ++k) {
        const ExtMatrix& T = ops.at(primes[k]);
        const uint32_t tw = static_cast<uint32_t>(PowModU(primes[k] % ell, b, ell));
        M.push_back(ScalarShift(T * tw, a[k]));
    }
    ExtMatrix S;
    bool first = true;
    for (const ExtMatrix& m : M) {
        ExtMatrix P = m;
        // m^dim by squaring up to >= dim
        uint32_t e = 1;
        while (e < dim) { P = P * P; e *= 2; }
        S = first ? P : hecke::VStack(S, P);
        first = false;
    }
    const ExtMatrix K = hecke::KernelAsMatrix(S);
    out_dim = K.cols;
    for (const ExtMatrix& m : M)
        if (K.cols > 0 && !IsZeroExt(m * K)) scalar = false;
}

RingMat ShiftRing(const RingMat& T, uint64_t c) {   // T - c I over (+) Z/ell^{a_i}
    RingMat I = hecke::ordinary::Identity(T.ell, T.row_exp);
    for (uint32_t i = 0; i < I.n(); ++i) {
        uint64_t q = 1;
        for (uint32_t k = 0; k < I.row_exp[i]; ++k) q *= T.ell;
        I.a[static_cast<size_t>(i) * I.n() + i] = c % q;
    }
    return hecke::ordinary::Sub(T, I);
}

std::string VecText(const std::vector<uint64_t>& v) {
    std::ostringstream s;
    for (size_t i = 0; i < v.size(); ++i) s << (i ? "," : "") << v[i];
    return s.str();
}
std::string VecText32(const std::vector<uint32_t>& v) {
    std::ostringstream s;
    for (size_t i = 0; i < v.size(); ++i) s << (i ? "," : "") << v[i];
    return s.str();
}

} // namespace

const CensusWeightCache::Entry& CensusWeightCache::get(uint32_t a, uint32_t ell) {
    auto it = cache_.find({a, ell});
    if (it != cache_.end()) return it->second;
    Entry e;
    if (a == 0 || (a % 2) == 1) {
        e.ok = true;   // H^1(SL_2(Z); Sym^a F_ell) = 0 at ell >= 5 for a = 0 and for odd a (-I)
        e.dim = 0;
    } else {
        HeckeExtDriver d(a, ell, LatticeDriverOptions(false));
        d.run();
        const std::vector<uint32_t>& dims = d.H_dimensions();
        e.dim = dims.size() > 1 ? dims[1] : 0;
        for (const auto& op : d.cell_operators(1)) e.ops.emplace(op.first, op.second);
        e.ok = true;
    }
    return cache_.emplace(std::make_pair(a, ell), e).first->second;
}

bool PairProductsInEllM(const std::vector<RingMat>& X, const RingMat& e, std::string* first_fail, uint32_t* pairs) {
    bool held = true;
    uint32_t np = 0;
    std::vector<RingMat> Xe;
    for (const RingMat& x : X) Xe.push_back(hecke::ordinary::Mul(x, e));
    for (size_t i = 0; i < X.size(); ++i)
        for (size_t j = i; j < X.size(); ++j) {
            ++np;
            const RingMat Y = hecke::ordinary::Mul(X[i], Xe[j]);
            if (hecke::ordinary::RankModEll(Y) != 0 && held) {
                held = false;
                if (first_fail) *first_fail = std::to_string(i) + "," + std::to_string(j);
            }
        }
    if (pairs) *pairs = np;
    return held;
}

CensusLatticeCell ComputeCensusLattice(uint32_t n, uint32_t ell, const CensusLatticeOptions& opts,
                                       CensusWeightCache* weights) {
    ENGINE_CHECK_MSG(ell >= 5, "HeckeCensusLattice: ell >= 5 (H^2(SL_2(Z); -) = 0 needs ell prime to 12)");
    ENGINE_CHECK_MSG(static_cast<uint64_t>(ell) * ell < (uint64_t(1) << 32),
                     "HeckeCensusLattice: ell^2 < 2^32 (the shipped Z/ell^2 module route)");
    const Clock::time_point t0 = Clock::now();
    CensusLatticeCell cell;
    cell.n = n;
    cell.ell = ell;

    std::unique_ptr<hecke::ScopedUEll> u_guard;
    if (opts.u_ell_budget_mb != 0) {
        hecke::UEllSetting u;
        u.mode = hecke::UEllMode::BudgetMb;
        u.budget_mb = opts.u_ell_budget_mb;
        u_guard.reset(new hecke::ScopedUEll(u, hecke::SettingSource::Spec));
    }

    // --- the F_ell side: the driver, its blocks, matched to the record ----
    HeckeExtDriver d(n, ell, LatticeDriverOptions(true));
    d.run();
    cell.driver_s = std::chrono::duration<double>(Clock::now() - t0).count();
    const HeckeExtRecord& rec = d.record();
    const std::vector<uint32_t>& dims = d.H_dimensions();
    cell.dim_H1 = dims.size() > 1 ? dims[1] : 0;
    std::vector<uint32_t> h1_ideals;
    for (uint32_t i = 0; i < rec.ideals.size(); ++i)
        if (OwnGroupOf(rec.ideals[i]) == "H1") h1_ideals.push_back(i);
    if (cell.dim_H1 == 0 || h1_ideals.empty()) {
        cell.seconds = std::chrono::duration<double>(Clock::now() - t0).count();
        return cell;
    }

    std::vector<std::pair<uint64_t, ExtMatrix>> ops = d.cell_operators(1);
    std::sort(ops.begin(), ops.end(),
              [](const std::pair<uint64_t, ExtMatrix>& a, const std::pair<uint64_t, ExtMatrix>& b) {
                  return a.first < b.first;
              });
    std::map<uint64_t, ExtMatrix> opmap;
    for (const auto& op : ops) opmap.emplace(op.first, op.second);
    hecke::HeckeModuleDecomposition decomp(ops, ell);
    decomp.compute();
    ENGINE_CHECK_MSG(decomp.certify() == 0, "HeckeCensusLattice: the re-run decomposition does not certify");
    const std::vector<hecke::PrimaryBlock>& blocks = decomp.blocks();
    ENGINE_CHECK_MSG(blocks.size() == h1_ideals.size(),
                     "HeckeCensusLattice: the re-run decomposition has a different number of H^1 blocks");

    const hecke::HeckeInvolution* inv = d.involution();
    ENGINE_CHECK_MSG(inv != nullptr, "HeckeCensusLattice: the driver built no involution");
    ExtMatrix Hp(cell.dim_H1, 0, ell), Hm(cell.dim_H1, 0, ell);
    {
        const FlatMatrix& P = inv->H_plus_basis(1);
        const FlatMatrix& M = inv->H_minus_basis(1);
        Hp = ExtMatrix(P.get_rows(), P.get_cols(), ell);
        for (uint32_t i = 0; i < P.get_rows(); ++i)
            for (uint32_t j = 0; j < P.get_cols(); ++j) Hp.data[i][j] = P.get_entry(i, j);
        Hm = ExtMatrix(M.get_rows(), M.get_cols(), ell);
        for (uint32_t i = 0; i < M.get_rows(); ++i)
            for (uint32_t j = 0; j < M.get_cols(); ++j) Hm.data[i][j] = M.get_entry(i, j);
    }
    ENGINE_CHECK_MSG(Hp.rows == cell.dim_H1 && Hm.rows == cell.dim_H1,
                     "HeckeCensusLattice: sign bases of the wrong height");

    // --- the lattice side: H^1(Gamma; Sym^n Z/ell^2) with T_q, U_ell first ---
    std::vector<uint64_t> S;
    for (const auto& op : ops) S.push_back(op.first);
    std::vector<uint64_t> mod_primes = {ell};
    for (uint64_t q : S)
        if (q != ell) mod_primes.push_back(q);
    const Clock::time_point tm = Clock::now();
    const hida::ModuleCell mc = opts.join_only ? hida::ModuleCell() : hida::RunModuleCell(n, ell, 2, mod_primes);
    cell.module_s = std::chrono::duration<double>(Clock::now() - tm).count();
    cell.module_gens = opts.join_only ? cell.dim_H1 : static_cast<uint32_t>(mc.exps.size());
    if (cell.module_gens != cell.dim_H1)
        cell.failures.push_back("module generators " + std::to_string(cell.module_gens)
                                + " != dim H^1(F_ell) " + std::to_string(cell.dim_H1) + " (UCT with H^2 = 0)");

    // (W) the cell's own Sym^n when n < ell is its only Glover factor.
    if (weights && n < ell && n % 2 == 0 && n > 0) {
        CensusWeightCache::Entry e;
        e.ok = true;
        e.dim = cell.dim_H1;
        e.ops = opmap;
        weights->seed(n, ell, e);
    }
    const GloverDecomposition glover = GloverFactors(n, ell);

    for (size_t k = 0; k < blocks.size(); ++k) {
        const hecke::PrimaryBlock& b = blocks[k];
        const IdealRecord& id = rec.ideals[h1_ideals[k]];
        CensusLatticeRow row;
        row.ideal_id = id.id;
        row.dim_E = b.dim;
        row.primes = S;
        row.witness_poly = id.witness_poly;

        // exact match with the record ideal (as HeckeCensusSplit)
        std::vector<ExtMatrix> t;
        for (const auto& op : ops)
            t.push_back(hecke::SolveColumnsExact(b.basis_columns, op.second * b.basis_columns,
                                                 "HeckeCensusLattice: block restriction"));
        ENGINE_CHECK_MSG(b.witness_poly.coeffs == id.witness_poly,
                         "HeckeCensusLattice: a re-run block's witness_poly is not the record ideal's");
        ENGINE_CHECK_MSG(S == id.operator_primes && t.size() == id.t_p.size(),
                         "HeckeCensusLattice: a re-run block's operator primes are not the record ideal's");
        for (size_t j = 0; j < t.size(); ++j)
            ENGINE_CHECK_MSG(SameEntries(t[j], id.t_p[j]),
                             "HeckeCensusLattice: a re-run block's restricted t_p is not the record ideal's");

        // eigenvalues
        row.eigen_ok = true;
        for (const ExtMatrix& tq : t) {
            uint32_t a = 0;
            if (!Eigenvalue(tq, a)) { row.eigen_ok = false; break; }
            row.a.push_back(a);
        }
        if (!row.eigen_ok) {
            row.status = "not_evaluated";
            row.failures.push_back("residue degree > 1: no eigenvalue in F_ell");
            cell.rows.push_back(row);
            continue;
        }

        // signs, Loewy lengths (F_ell)
        const ExtMatrix Vp = Intersect(b.basis_columns, Hp);
        const ExtMatrix Vm = Intersect(b.basis_columns, Hm);
        row.dim_plus = Vp.cols;
        row.dim_minus = Vm.cols;
        if (row.dim_plus + row.dim_minus != row.dim_E) row.failures.push_back("dim_plus + dim_minus != dim_E");
        std::vector<ExtMatrix> X, Xoff;
        for (size_t j = 0; j < S.size(); ++j) {
            X.push_back(ScalarShift(opmap.at(S[j]), row.a[j]));
            if (S[j] != ell) Xoff.push_back(X.back());
        }
        row.loewy_V = Loewy(b.basis_columns, X);
        row.loewy_Vplus = Loewy(Vp, X);
        row.loewy_Vminus = Loewy(Vm, X);
        row.loewy_V_off = Loewy(b.basis_columns, Xoff);
        row.loewy_Vplus_off = Loewy(Vp, Xoff);
        row.loewy_Vminus_off = Loewy(Vm, Xoff);
        if (opts.join_only) {
            row.status = "join_only";
            cell.rows.push_back(row);
            continue;
        }

        // (F) the pair products on V over F_ell
        row.cf9_field = true;
        for (size_t i = 0; i < X.size() && row.cf9_field; ++i)
            for (size_t j = i; j < X.size(); ++j)
                if (!IsZeroExt(X[i] * (X[j] * b.basis_columns))) { row.cf9_field = false; break; }

        // (L) the lattice mod ell^2
        std::vector<RingMat> XR;
        RingMat e = hecke::ordinary::Identity(ell, mc.exps);
        for (size_t j = 0; j < S.size(); ++j) {
            XR.push_back(ShiftRing(mc.at(S[j]), row.a[j]));
            const RingMat P = hecke::ordinary::OrdinaryProjector(XR.back());
            e = hecke::ordinary::Mul(e, hecke::ordinary::Sub(hecke::ordinary::Identity(ell, mc.exps), P));
        }
        if (hecke::ordinary::Mul(e, e) != e) row.failures.push_back("e_m is not idempotent");
        row.lattice_rank = hecke::ordinary::RankModEll(e);
        row.top_rank = hida::TopLayerRank(e, 2);
        row.free_ = (row.top_rank == row.lattice_rank);
        row.torsion_gens = row.lattice_rank - std::min(row.lattice_rank, row.top_rank);
        if (row.lattice_rank != row.dim_E)
            row.failures.push_back("rank_F_ell e_m = " + std::to_string(row.lattice_rank) + " != dim_E");
        std::string ff;
        row.cf9_lattice = PairProductsInEllM(XR, e, &ff, &row.pairs);
        if (!ff.empty()) {
            const size_t c = ff.find(',');
            row.first_fail = std::to_string(S[std::stoul(ff.substr(0, c))]) + ","
                             + std::to_string(S[std::stoul(ff.substr(c + 1))]);
        }
        for (const RingMat& x : XR)
            if (hecke::ordinary::RankModEll(hecke::ordinary::Mul(x, e)) != 0) row.m_nonzero = true;
        if (row.cf9_field != row.cf9_lattice)
            row.failures.push_back("the F_ell and the Z/ell^2 routes disagree on the pair products");

        // (W) the Serre-weight mechanism
        if (opts.serre_weights && weights) {
            std::vector<uint64_t> Soff;
            std::vector<uint32_t> aoff;
            for (size_t j = 0; j < S.size(); ++j)
                if (S[j] != ell) { Soff.push_back(S[j]); aoff.push_back(row.a[j]); }
            bool scalar_all = true;
            JointPart(opmap, cell.dim_H1, ell, Soff, aoff, 0, row.dim_mprime, scalar_all);
            row.sw_evaluated = true;
            row.sw_scalar = true;
            std::ostringstream fs;
            for (const GloverFactor& f : glover.factors) {
                const CensusWeightCache::Entry& we = weights->get(f.sym, ell);
                if (!we.ok) { row.sw_evaluated = false; row.sw_why = we.why; break; }
                if (we.dim == 0) continue;
                bool have = true;
                for (uint64_t q : Soff) have = have && we.ops.count(q) != 0;
                if (!have) { row.sw_evaluated = false; row.sw_why = "weight cell lacks a prime"; break; }
                uint32_t dc = 0;
                bool sc = true;
                JointPart(we.ops, we.dim, ell, Soff, aoff, f.det_twist, dc, sc);
                if (dc == 0) continue;
                row.sw_count += f.multiplicity;
                ++row.sw_distinct;
                row.sw_dim += f.multiplicity * dc;
                row.sw_scalar = row.sw_scalar && sc;
                fs << (fs.tellp() > 0 ? ";" : "") << f.sym << ":" << f.det_twist << ":" << f.multiplicity << ":" << dc;
            }
            row.sw_factors = fs.str();
            if (row.sw_evaluated) {
                row.mech_dim_ok = row.dim_mprime <= row.sw_dim;
                row.mech_loewy_ok = !row.sw_scalar || row.loewy_V_off <= row.sw_count;
                if (!row.mech_dim_ok)
                    row.failures.push_back("dim H^1_{m'} = " + std::to_string(row.dim_mprime)
                                           + " > sum over the Glover factors " + std::to_string(row.sw_dim)
                                           + " (right exactness violated: convention or bug)");
            }
        }

        row.status = row.cf9_lattice ? "held" : "failed";
        cell.rows.push_back(row);
    }
    cell.seconds = std::chrono::duration<double>(Clock::now() - t0).count();
    return cell;
}

std::string CensusLatticeHeader() {
    return "n\tell\tideal_id\tdim_E\tdim_plus\tdim_minus\tprimes\ta_q\tstatus\tcf9_lattice\tcf9_field\tpairs\t"
           "first_fail\tm_nonzero\tlattice_rank\tfree\ttorsion_gens\tloewy_V\tloewy_Vplus\tloewy_Vminus\t"
           "loewy_V_off\tloewy_Vplus_off\tloewy_Vminus_off\tsw_count\tsw_distinct\tsw_dim\tsw_scalar\tdim_mprime\t"
           "mech_dim\tmech_loewy\tsw_factors\tfailures\tcell_s\tdriver_s\tmodule_s";
}

std::string CensusLatticeLine(const CensusLatticeCell& cell, const CensusLatticeRow& r) {
    auto tf = [](bool b) { return b ? "true" : "false"; };
    std::ostringstream s;
    s << "CENSUS_LATTICE\t" << cell.n << "\t" << cell.ell << "\t" << r.ideal_id << "\t" << r.dim_E << "\t"
      << r.dim_plus << "\t" << r.dim_minus << "\t" << VecText(r.primes) << "\t" << VecText32(r.a) << "\t"
      << r.status << "\t" << tf(r.cf9_lattice) << "\t" << tf(r.cf9_field) << "\t" << r.pairs << "\t"
      << (r.first_fail.empty() ? "-" : r.first_fail) << "\t" << tf(r.m_nonzero) << "\t" << r.lattice_rank << "\t"
      << tf(r.free_) << "\t" << r.torsion_gens << "\t" << r.loewy_V << "\t" << r.loewy_Vplus << "\t"
      << r.loewy_Vminus << "\t" << r.loewy_V_off << "\t" << r.loewy_Vplus_off << "\t" << r.loewy_Vminus_off
      << "\t";
    if (r.sw_evaluated) {
        s << r.sw_count << "\t" << r.sw_distinct << "\t" << r.sw_dim << "\t" << tf(r.sw_scalar) << "\t" << r.dim_mprime << "\t"
          << (r.mech_dim_ok ? "ok" : "FAIL") << "\t"
          << (!r.sw_scalar ? "na" : (r.mech_loewy_ok ? "ok" : "FAIL")) << "\t"
          << (r.sw_factors.empty() ? "-" : r.sw_factors);
    } else {
        s << "-\t-\t-\t-\t-\tnot_evaluated\tnot_evaluated\t" << (r.sw_why.empty() ? "-" : r.sw_why);
    }
    s << "\t";
    std::vector<std::string> f = cell.failures;
    f.insert(f.end(), r.failures.begin(), r.failures.end());
    if (f.empty()) s << "-";
    for (size_t i = 0; i < f.size(); ++i) s << (i ? " | " : "") << f[i];
    s.setf(std::ios::fixed);
    s.precision(2);
    s << "\t" << cell.seconds << "\t" << cell.driver_s << "\t" << cell.module_s;
    return s.str();
}

} // namespace hecke_ext
