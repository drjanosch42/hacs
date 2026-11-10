// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// See the header.
#include "CohomologyBlockModelValidation.h"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "../../ModArithmetic/BlockElim.h"
#include "../../ModArithmetic/BlockKernels.h"
#include "../../ModArithmetic/BlockMat.h"
#include "../../ModArithmetic/DenseBuf.h"
#include "../../ModArithmetic/DenseView.h"
#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../SimplicialCohomology.h"

namespace {

using engine::BlockMat;
using engine::DenseBuf;
using engine::DenseView;
using engine::Layout;
using engine::cview;
using engine::view;
using Clock = std::chrono::steady_clock;

uint32_t g_fail = 0;
uint64_t g_checks = 0;
void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        if (g_fail <= 60) std::cout << "  [FAIL] " << what << std::endl;
    }
}
template<class T> std::string str(T v) { return std::to_string(v); }

template<class Elt>
bool same_fm(const FlatMatrixT<Elt>& a, const FlatMatrixT<Elt>& b) {
    if (a.get_rows() != b.get_rows() || a.get_cols() != b.get_cols()) return false;
    for (uint32_t i = 0; i < a.get_rows(); ++i)
        for (uint32_t j = 0; j < a.get_cols(); ++j)
            if (a.get_entry(i, j) != b.get_entry(i, j)) return false;
    return true;
}

template<class Elt>
bool same_lists(const std::vector<std::vector<FlatMatrixT<Elt>>>& a,
                const std::vector<std::vector<FlatMatrixT<Elt>>>& b) {
    if (a.size() != b.size()) return false;
    for (size_t q = 0; q < a.size(); ++q) {
        if (a[q].size() != b[q].size()) return false;
        for (size_t t = 0; t < a[q].size(); ++t)
            if (!same_fm<Elt>(a[q][t], b[q][t])) return false;
    }
    return true;
}

// Y = A * X (X a multi-vector), dense column-major.
template<class Elt>
DenseBuf<Elt> block_times(const BlockMat<Elt>& A, const BlockMat<Elt>& X) {
    DenseBuf<Elt> scratch;
    const DenseView<const Elt> Xv = engine::blk::multivector_view<Elt>(X, scratch);
    DenseBuf<Elt> Y(A.rows(), Xv.cols(), Layout::ColMajor);
    engine::blk::gemm_dense<Elt>(A, Xv, view(Y));
    return Y;
}
template<class Elt>
bool is_zero(const DenseBuf<Elt>& Y) { return BlockMat<Elt>::is_zero_view(cview(Y)); }
template<class Elt>
bool is_identity(const DenseBuf<Elt>& Y) {
    if (Y.rows != Y.cols) return false;
    for (uint32_t i = 0; i < Y.rows; ++i)
        for (uint32_t j = 0; j < Y.cols; ++j)
            if (Y.at(i, j) != (i == j ? Elt(1) : Elt(0))) return false;
    return true;
}

// (b): the multi-vector against its FlatMatrix list -- shape, partition, and
// column t (L1 order) == row-major cochain t.
template<class Elt>
void check_multivector(const BlockMat<Elt>& mv, const std::vector<FlatMatrixT<Elt>>& list,
                       uint32_t N_q, uint32_t d, const std::string& tag) {
    check(mv.nbc() == 1, tag + " nbc == 1");
    check(mv.rows() == N_q * d && mv.nbr() == N_q, tag + " row partition (N_q, d_V)");
    for (uint32_t s = 0; s < mv.nbr(); ++s)
        if (mv.row_dim(s) != d) { check(false, tag + " block height != d_V"); break; }
    check(mv.cols() == list.size(), tag + " column count == basis size");
    if (mv.cols() != list.size()) return;
    const DenseBuf<Elt> F = mv.to_flat(Layout::ColMajor);
    bool ok = true;
    for (uint32_t t = 0; ok && t < F.cols; ++t) {
        const FlatMatrixT<Elt>& c = list[t];
        if (c.get_rows() != N_q || c.get_cols() != d) { ok = false; break; }
        for (uint32_t s = 0; ok && s < N_q; ++s)
            for (uint32_t k = 0; k < d; ++k)
                if (F.at(s * d + k, t) != c.get_entry(s, k)) { ok = false; break; }
    }
    check(ok, tag + " to_flat column t == cochain t (L1)");
}

template<class Elt>
void check_object(const EquivariantCohomologyOnKT<Elt>& coh, const std::string& tag) {
    const SimplicialComplex* K = coh.get_lssc()->get_base_K_complex();
    const uint32_t d = coh.get_representation_dim();
    const auto& dims = coh.get_H_eq_dimensions();
    const auto& C = coh.get_C_eq_bases_fm();
    const auto& Z = coh.get_Z_eq_bases_fm();
    const auto& B = coh.get_B_eq_bases_fm();
    const auto& H = coh.get_H_eq_bases_fm();
    const auto& P = coh.get_H_eq_projectors_from_C_eq_flat_fm();
    check(coh.block_pool() != nullptr, tag + " block_pool");
    const uint32_t nq = static_cast<uint32_t>(dims.size());
    check(C.size() == nq && Z.size() == nq && B.size() == nq && H.size() == nq && P.size() == nq,
          tag + " per-degree list sizes");
    for (uint32_t q = 0; q < nq; ++q) {
        const std::string tq = tag + " q=" + str(q);
        const uint32_t N_q = K->get_num_q_simplices(q);
        const uint32_t ambient = N_q * d;
        check(coh.block_row_dims(q) == std::vector<uint32_t>(N_q, d), tq + " block_row_dims");
        check_multivector<Elt>(coh.C_eq_block(q), C[q], N_q, d, tq + " C_eq");
        check_multivector<Elt>(coh.Z_eq_block(q), Z[q], N_q, d, tq + " Z_eq");
        check_multivector<Elt>(coh.B_eq_block(q), B[q], N_q, d, tq + " B_eq");
        check_multivector<Elt>(coh.H_eq_block(q), H[q], N_q, d, tq + " H_eq");
        check(dims[q] == H[q].size() && dims[q] == coh.H_eq_block(q).cols(), tq + " dim H^q");

        // projector
        const BlockMat<Elt>& Pb = coh.P_block(q);
        check(Pb.rows() == dims[q] && Pb.cols() == ambient && Pb.nbr() == 1 && Pb.nbc() == N_q,
              tq + " P_block grid {dim H} x (N_q, d_V)");
        if (Z[q].empty()) {
            check(P[q].get_rows() == 0 && P[q].get_cols() == 0, tq + " legacy 0x0 projector when Z^q is empty");
        } else {
            check(P[q].get_rows() == dims[q] && P[q].get_cols() == ambient, tq + " projector shape");
            bool ok = P[q].get_rows() == Pb.rows() && P[q].get_cols() == Pb.cols();
            for (uint32_t i = 0; ok && i < Pb.rows(); ++i)
                for (uint32_t j = 0; j < Pb.cols(); ++j)
                    if (Pb.get_entry(i, j) != P[q].get_entry(i, j)) { ok = false; break; }
            check(ok, tq + " P_block == projector entrywise");
        }

        // constraint matrices + certificates
        const BlockMat<Elt>& AC = coh.A_C_block(q);
        const BlockMat<Elt>& AZ = coh.A_Z_block(q);
        const uint32_t k_q = static_cast<uint32_t>(K->template get_trivial_cocycle_basis_Z1_elt<Elt>(q).size());
        check(AC.cols() == ambient && AC.nbc() == N_q, tq + " A_C_block columns");
        check(AZ.cols() == k_q * d && AZ.nbc() == k_q, tq + " A_Z_block columns");
        check(AC.nbr() == AZ.nbr(), tq + " A_C / A_Z share the live-relation rows");
        check(is_zero<Elt>(block_times<Elt>(AC, coh.C_eq_block(q))), tq + " A_Cq * C_eq == 0");
        check(is_zero<Elt>(block_times<Elt>(AC, coh.Z_eq_block(q))), tq + " A_Cq * Z_eq == 0");
        check(is_zero<Elt>(block_times<Elt>(AC, coh.B_eq_block(q))), tq + " A_Cq * B_eq == 0");
        check(is_zero<Elt>(block_times<Elt>(AC, coh.H_eq_block(q))), tq + " A_Cq * H_eq == 0");
        check(is_identity<Elt>(block_times<Elt>(Pb, coh.H_eq_block(q))), tq + " P_q * H_eq == I");
        check(is_zero<Elt>(block_times<Elt>(Pb, coh.B_eq_block(q))), tq + " P_q * B_eq == 0");
        // With the 7.4 short-circuit on, an identity kernel means no live
        // relation (with it off, vacuous rows are kept as zero rows).
        if (C[q].size() == ambient && EquivariantCohomologyOptions::vacuous_relation_shortcircuit_enabled())
            check(AC.nbr() == 0, tq + " identity kernel <=> no live relation");
    }
}

// (a) + (b) + (c) for one local system.
template<class Elt>
void check_instance(const LocalSystemSimplicialComplexT<Elt>& lssc, const std::string& tag) {
    EquivariantCohomologyOnKT<Elt> eng(&lssc);
    eng.compute_cohomology();
    EquivariantCohomologyOnKT<Elt> leg(&lssc);
    {
        EquivariantCohomologyOptions::ScopedLegacyCohomologyPath force(true);
        leg.compute_cohomology();
    }
    check(eng.get_H_eq_dimensions() == leg.get_H_eq_dimensions(), tag + " dims engine == legacy");
    check(same_lists<Elt>(eng.get_C_eq_bases_fm(), leg.get_C_eq_bases_fm()), tag + " C_eq engine == legacy");
    check(same_lists<Elt>(eng.get_Z_eq_bases_fm(), leg.get_Z_eq_bases_fm()), tag + " Z_eq engine == legacy");
    check(same_lists<Elt>(eng.get_B_eq_bases_fm(), leg.get_B_eq_bases_fm()), tag + " B_eq engine == legacy");
    check(same_lists<Elt>(eng.get_H_eq_bases_fm(), leg.get_H_eq_bases_fm()), tag + " H_eq engine == legacy");
    {
        const auto& Pe = eng.get_H_eq_projectors_from_C_eq_flat_fm();
        const auto& Pl = leg.get_H_eq_projectors_from_C_eq_flat_fm();
        bool ok = Pe.size() == Pl.size();
        for (size_t q = 0; ok && q < Pe.size(); ++q) ok = same_fm<Elt>(Pe[q], Pl[q]);
        check(ok, tag + " projectors engine == legacy (incl. shape conventions)");
    }
    check_object<Elt>(eng, tag + " [engine]");
    check_object<Elt>(leg, tag + " [legacy]");
}

template<class Elt>
void sl2z_instance(const SimplicialComplex& K, uint32_t n_sym, Elt ell, const std::string& tag) {
    const auto rels = GenerateSL2ZRelationsElt<Elt>(K, n_sym, ell);
    const LocalSystemSimplicialComplexT<Elt> lssc(K, rels, n_sym + 1, ell);
    check_instance<Elt>(lssc, tag);
}

} // namespace

uint32_t CohomologyBlockModelValidation() {
    const auto t0 = Clock::now();
    g_fail = 0; g_checks = 0;
    std::cout << "=== CohomologyBlockModelValidation (compute_cohomology on the block model: engine vs legacy arm, "
                 "block accessors vs FlatMatrix, algebraic certificates) ===" << std::endl;
    const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();

    const std::vector<uint32_t> ns = {0, 1, 2, 3, 4, 5, 6, 8, 12, 16};
    const std::vector<uint32_t> ells = {2u, 3u, 7u, 134217757u, 4294967291u};
    uint32_t instances = 0;
    for (uint32_t ell : ells)
        for (uint32_t n : ns) {
            sl2z_instance<uint32_t>(K, n, ell, "n=" + str(n) + " ell=" + str(ell));
            ++instances;
        }
    // The pre-7.4 assembly (every relation kept) on a subset.
    {
        EquivariantCohomologyOptions::ScopedVacuousShortcircuit off(false);
        for (uint32_t n : {4u, 5u, 8u}) {
            sl2z_instance<uint32_t>(K, n, 134217757u, "n=" + str(n) + " ell=134217757 shortcircuit=off");
            ++instances;
        }
    }
    sl2z_instance<uint32_t>(K, 32, 134217757u, "n=32 ell=134217757");
    ++instances;
    // No relations at all: C_eq is the identity, A_Cq has no rows.
    {
        const std::vector<LocalSystemSimplicialComplex::SimplexRelation> none;
        const LocalSystemSimplicialComplex lssc(K, none, 3, 7u);
        check_instance<uint32_t>(lssc, "no-relations d_V=3 ell=7");
        ++instances;
    }
    // uint64_t instantiation, extended regime.
    for (uint32_t n : {4u, 7u}) {
        sl2z_instance<uint64_t>(K, n, 2305843009213693951ull, "u64 n=" + str(n) + " ell=2^61-1");
        ++instances;
    }

    const double secs = std::chrono::duration<double>(Clock::now() - t0).count();
    std::cout << "CohomologyBlockModelValidation: " << instances << " local systems, " << g_checks << " checks, "
              << g_fail << " failed, " << secs << " s" << std::endl;
    return g_fail == 0 ? 0 : 1;
}
