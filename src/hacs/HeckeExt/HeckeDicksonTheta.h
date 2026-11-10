// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_DICKSON_THETA_H
#define HECKE_EXT_DICKSON_THETA_H

// ===========================================================================
// hecke-extensions slot 11, WP-5 -- THE DICKSON / THETA LINE (1): the
// quotient local system Q_n, the long sequence of the short sequence of
// local systems
//
//     0 -> Sym^n --L.--> Sym^m --pi--> Q_n -> 0,      m = n + ell + 1,
//
// on the 7-vertex model over F_ell, its maps on cohomology, the certificates
// of the sequence, and the Dickson monomials L^a D^b on H^0.
//
// Plan: docs/implementation/hecke_extensions_slot11_plan.md, "### WP-5".
// The binding statements are [C, sec:dicksontheta] (v1.11.0-pre.1): Q_n and
// its basis [C, def:Qn], the twist (*) [C, lem:dicksontwist], the sequence
// [C, thm:thetales], H^0 [C, prop:dicksonH0], H^0(Q_n) [C, lem:Qnfunctions].
// This unit COMPUTES the objects and CHECKS them -- every check below is a
// computation on the stored objects, and a check that fails is reported with
// its code, never repaired.
//
// THE OBJECTS (all over F_ell, ell >= 5; one DicksonThetaCell per (n, ell)):
//   * three computed EquivariantCohomologyOnK objects on the shipped model:
//       Term::SymN   Sym^n        (GenerateSL2ZRelationsWithGroupData(K, n, ell))
//       Term::SymM   Sym^m        (GenerateSL2ZRelationsWithGroupData(K, m, ell))
//       Term::Quot   Q_n          (GenerateSL2ZQuotientLRelationsWithGroupData(K, n, ell),
//                                  FibreRepresentation::QuotientL, fibre dim ell + 1)
//   * the fibre maps  L. = DicksonMultiplicationMatrix(m, ell)   ((m+1) x (n+1))
//                     pi = QuotientLProjectionMatrixFM(n, ell)    ((ell+1) x (m+1));
//   * the maps on cohomology in the stored H bases, COLUMN convention
//     (column j = coordinates of the image of basis class j), q = 0, 1, 2:
//       iota(q)  = H^q(L.)  : H^q(Sym^n) -> H^q(Sym^m)     (theta(q) is the same matrix)
//       pi(q)    = H^q(pi)  : H^q(Sym^m) -> H^q(Q_n)
//       delta(q) : H^q(Q_n) -> H^{q+1}(Sym^n), q = 0, 1 -- the connecting map
//                  COMPUTED on cochains: lift a cocycle z of Q_n to an
//                  equivariant cochain y of Sym^m (an exact solve on the
//                  C^q_eq basis), apply the coboundary, divide by L. simplex
//                  by simplex (an exact back-substitution, re-multiplied and
//                  compared), project the result to H^{q+1}(Sym^n).
//
// THE CERTIFICATE (DicksonThetaCertificate; code 0 = every check held):
//   the fibre level  (W)  pi L. = 0, rank pi = ell + 1, pi E = I, L. certified
//                         (CertifyDicksonMultiplication), and on every sample g
//                         -- seeded, singular (U_ell's representatives mod ell
//                         among them) and every relation matrix g_r --
//                         pi rho_m(g) L. = 0 entrywise, rho_Q(g) pi = pi rho_m(g),
//                         and rho_Q(gh) = rho_Q(g) rho_Q(h) on consecutive pairs;
//   the cochain level     every image cochain satisfies the target's relations
//                         and is a cocycle; the lift exists; the coboundary of
//                         the lift is divisible by L.;
//   the pipeline objects  P_q H_q = I and P_q B_q = 0 on each term (so that P_q
//                         reads the class of any cocycle);
//   the sequence          ExactnessCheck at every spot (below) and H^2 = 0 on
//                         all three terms, and the alternating dimension sum;
//   H^0                   the constant cochains of L^a D^b, a(ell+1) +
//                         b(ell^2 - ell) = n (resp. m), are cocycles of Sym^n
//                         (resp. Sym^m) and their classes span a space of
//                         dimension #{(a, b)} = dim H^0.
//
// ExactnessCheck: for a composable pair  A : X -> Y,  B : Y -> Z  it computes a
// basis of im A (pivot columns) and of ker B (the RREF null space), and checks
// im A in ker B and ker B in im A as subspaces (ranks of the concatenations),
// besides the ranks.  The six spots of the sequence, in order:
//   0  H^0(Sym^n)  ker iota(0) = 0
//   1  H^0(Sym^m)  im iota(0) = ker pi(0)
//   2  H^0(Q_n)    im pi(0)   = ker delta(0)
//   3  H^1(Sym^n)  im delta(0) = ker iota(1)
//   4  H^1(Sym^m)  im iota(1) = ker pi(1)
//   5  H^1(Q_n)    im pi(1)   = ker delta(1) (delta(1) lands in H^2(Sym^n))
//
// REFUSALS: ell < 5 by ENGINE_CHECK_MSG naming [C] lem:h2torsion (a
// std::logic_error, console exit 3: a mathematical precondition, [E]
// sec:thetaobj (h); slot-11 WP-17, F-2), and a non-default coefficient_ring
// by name (RequireFieldCoefficientRing, hecke::NotImplementedInThisBuild).
//
// PLACEMENT.  HeckeExt/, namespace hecke_ext; compiled into cohomology_app
// (cmake/slot11/WP5.cmake).  No Console/, Cuda/, Engine/, BlockFlatMatrix.h
// include, no `unordered_` (check_heckeext_includes.sh).
// ===========================================================================

#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemUtils.h"
#include "../SimplicialCohomology.h"
#include "../Hecke/HeckeCosetTables.h"
#include "../Hecke/HeckeOperatorOnK.h"
#include "../Hecke/HeckeTauTable.h"
#include "../Hecke/SL2ZTessellationTables.h"
#include "../../ModArithmetic/FlatMatrix.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace hecke_ext {

// A small dense matrix over F_ell, row-major rows x cols (the level-(b) maps
// and the exactness algebra; dimensions are cohomology dimensions).
struct ThetaMat {
    uint32_t rows = 0, cols = 0;
    uint32_t p = 0;
    std::vector<uint32_t> a;   // rows * cols
    ThetaMat() = default;
    ThetaMat(uint32_t r, uint32_t c, uint32_t mod) : rows(r), cols(c), p(mod), a(static_cast<size_t>(r) * c, 0) {}
    uint32_t& at(uint32_t r, uint32_t c) { return a[static_cast<size_t>(r) * cols + c]; }
    uint32_t at(uint32_t r, uint32_t c) const { return a[static_cast<size_t>(r) * cols + c]; }
    bool is_zero() const;
    MatrixT<uint32_t> to_matrix() const;   // for HeckeIntertwines-style consumers
};

uint32_t ThetaRank(const ThetaMat& A);
ThetaMat ThetaMul(const ThetaMat& A, const ThetaMat& B);
ThetaMat ThetaKernelBasis(const ThetaMat& A);       // cols x k, columns a basis of ker A
ThetaMat ThetaImageBasis(const ThetaMat& A);        // rows x r, the pivot columns of A
bool ThetaSpanContains(const ThetaMat& big, const ThetaMat& small);   // colspan(small) in colspan(big)

// One spot of an exact sequence  X --A--> Y --B--> Z  (A: dim Y x dim X,
// B: dim Z x dim Y; either may be absent as a 0-column / 0-row matrix).
struct ExactnessSpot {
    std::string name;
    uint32_t dim = 0;              // dim Y
    uint32_t rank_in = 0;          // rank A = dim im A
    uint32_t dim_ker_out = 0;      // dim ker B
    bool composite_zero = false;   // B A = 0
    bool im_in_ker = false;        // im A in ker B
    bool ker_in_im = false;        // ker B in im A
    bool exact() const { return composite_zero && im_in_ker && ker_in_im && rank_in == dim_ker_out; }
};

// The six spots for given maps (iota0, pi0, delta0, iota1, pi1, delta1) --
// the maps are arguments so that a caller (the negative seam) can pass a
// perturbed one.  dims: h0n, h0m, h0q, h1n, h1m, h1q.
std::vector<ExactnessSpot> CheckSequenceExactness(const ThetaMat& iota0, const ThetaMat& pi0, const ThetaMat& delta0,
                                                  const ThetaMat& iota1, const ThetaMat& pi1, const ThetaMat& delta1,
                                                  const std::vector<uint32_t>& dims);



// The coefficient vector (length deg + 1, basis u_k = x^{deg-k} y^k) of
// L^a D^b, deg = a (ell+1) + b (ell^2 - ell), with L = x^ell y - x y^ell and
// D = sum_{i=0..ell} x^{(ell-i)(ell-1)} y^{i(ell-1)}.
std::vector<uint32_t> DicksonMonomial(uint32_t a, uint32_t b, uint32_t ell);
// Every (a, b) with a (ell+1) + b (ell^2 - ell) = n, a ascending.
std::vector<std::pair<uint32_t, uint32_t>> DicksonExponents(uint32_t n, uint32_t ell);

struct DicksonThetaCertificate {
    // 0 = ok; otherwise the first failing clause (verify_message).
    uint32_t code = 0;
    std::string detail;
    // Fibre level.
    uint32_t dickson_cert_code = 0;      // CertifyDicksonMultiplication(m, ell).verify()
    uint32_t num_samples = 0;            // samples the fibre checks ran on (relations included)
    int64_t xelly_first_failure = -1;    // the negative seam: must be >= 0 (a sample breaks x^ell y . Sym^n)
    // Cochain level.
    uint64_t relation_checks = 0, cocycle_checks = 0, lifts = 0, divisions = 0;
    // The sequence.
    std::vector<ExactnessSpot> spots;    // six
    bool h2_zero = false;                // H^2 = 0 on Sym^n, Sym^m, Q_n
    int64_t euler_sum = 0;               // h0n - h0m + h0q - h1n + h1m - h1q
    // H^0.
    uint32_t dickson_count_n = 0, dickson_rank_n = 0;   // #{(a,b)} and rank of their classes, Sym^n
    uint32_t dickson_count_m = 0, dickson_rank_m = 0;   // the same for Sym^m
    static const char* verify_message(uint32_t code);
};

// Which of the three terms.
enum class ThetaTerm : uint32_t { SymN = 0, SymM = 1, Quot = 2 };

class DicksonThetaCell {
public:
    // Builds and computes the three cohomology objects, the maps and the
    // certificate.  Refuses ell < 5 (std::logic_error naming lem:h2torsion) and
    // a non-default coefficient ring by name (hecke::NotImplementedInThisBuild).
    // num_random_samples: pseudo-
    // random 2x2 samples of the fibre checks on top of the fixed ones.
    DicksonThetaCell(uint32_t n, uint32_t ell, uint32_t num_random_samples = 18);

    uint32_t n() const { return n_; }
    uint32_t ell() const { return ell_; }
    uint32_t m() const { return m_; }

    const SimplicialComplex& K() const { return *K_; }
    const hecke::SL2ZTessellationTables& model() const { return *model_; }
    const EquivariantCohomologyOnK& coh(ThetaTerm t) const { return *coh_[static_cast<uint32_t>(t)]; }
    const LocalSystemSimplicialComplex& local_system(ThetaTerm t) const { return *lssc_[static_cast<uint32_t>(t)]; }
    FibreRepresentation representation(ThetaTerm t) const;
    uint32_t fibre_dim(ThetaTerm t) const;
    // dim H^q of a term.
    uint32_t h(ThetaTerm t, uint32_t q) const;

    const FlatMatrix& L_fibre() const { return L_; }     // (m+1) x (n+1)
    const FlatMatrix& pi_fibre() const { return Pi_; }   // (ell+1) x (m+1)

    // The maps on cohomology (column convention), q = 0, 1, 2 for iota / pi,
    // q = 0, 1 for delta.
    const ThetaMat& iota(uint32_t q) const;
    const ThetaMat& theta(uint32_t q) const { return iota(q); }
    const ThetaMat& pi(uint32_t q) const;
    const ThetaMat& delta(uint32_t q) const;

    const DicksonThetaCertificate& certificate() const { return cert_; }

    // --- Cochain-level helpers (flat layout s * d + c, simplex-major) --------
    // The class coordinates P_q y of a cocycle y of a term (length dim H^q).
    std::vector<uint32_t> class_of(ThetaTerm t, uint32_t q, const std::vector<uint32_t>& y) const;
    // Whether y satisfies every relation of the term in degree q and (q < 2) d y = 0.
    bool is_equivariant_cocycle(ThetaTerm t, uint32_t q, const std::vector<uint32_t>& y) const;
    // The stored basis class j of H^q(term) as a flat cochain.
    std::vector<uint32_t> basis_cochain(ThetaTerm t, uint32_t q, uint32_t j) const;
    // A fibre map applied simplex by simplex (F: r x c; y has N_q * c entries).
    std::vector<uint32_t> apply_fibre(const FlatMatrix& F, uint32_t q, const std::vector<uint32_t>& y) const;
    // The connecting map on one cocycle z of Q_n (degree q, q in {0, 1}):
    // returns the flat cocycle x of Sym^n in degree q + 1 (throws
    // std::runtime_error when a step of the construction fails).
    std::vector<uint32_t> connecting_cochain(uint32_t q, const std::vector<uint32_t>& z) const;

    // --- For the Hecke line (WP-6) -------------------------------------------
    // The three operators for g = diag(1, p) (p = ell gives U_ell), computed.
    // Sym^n and Q_n share one coset / tau table pair built at n; Sym^m has its
    // own at m.
    struct HeckeTriple {
        uint64_t p = 0;
        std::unique_ptr<hecke::HeckeCosetTables> cosets_n, cosets_m;
        std::unique_ptr<hecke::HeckeTauTable> tau_n, tau_m;
        std::unique_ptr<hecke::HeckeOperatorOnK> op_n, op_m, op_q;
    };
    HeckeTriple hecke_triple(uint64_t p) const;

    // One-line summary (dimensions, ranks, certificate code).
    std::string summary() const;

private:
    uint32_t n_ = 0, ell_ = 0, m_ = 0;
    std::unique_ptr<SimplicialComplex> K_;
    std::unique_ptr<hecke::SL2ZTessellationTables> model_;
    std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels_[3];
    std::unique_ptr<LocalSystemSimplicialComplex> lssc_[3];
    std::unique_ptr<EquivariantCohomologyOnK> coh_[3];
    FlatMatrix L_, Pi_;
    std::vector<FlatMatrix> D_;              // simplicial coboundaries D_q (N_{q+1} x N_q), q = 0, 1
    std::vector<ThetaMat> iota_, pi_, delta_;
    DicksonThetaCertificate cert_;

    uint32_t N(uint32_t q) const { return K_->get_num_q_simplices(q); }
    bool satisfies_relations(ThetaTerm t, uint32_t q, const std::vector<uint32_t>& y) const;
    std::vector<uint32_t> coboundary(uint32_t q, uint32_t d, const std::vector<uint32_t>& y) const;
    bool is_zero_vec(const std::vector<uint32_t>& y) const;
    void fail(uint32_t code, const std::string& detail);
    void certify_fibre(uint32_t num_random_samples);
    void certify_projectors();
    ThetaMat induced_map(ThetaTerm src, ThetaTerm dst, const FlatMatrix& F, uint32_t q);
    void compute_delta();
    void certify_sequence();
    void certify_h0();
};

} // namespace hecke_ext

#endif // HECKE_EXT_DICKSON_THETA_H
