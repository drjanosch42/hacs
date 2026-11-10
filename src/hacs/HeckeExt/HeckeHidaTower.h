// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_HECKE_HIDA_TOWER_H
#define HECKE_EXT_HECKE_HIDA_TOWER_H

// ===========================================================================
// (hecke-extensions slot 12, WP-5) HIDA WEIGHT TOWERS AT LEVEL 1, AND THE
// NON-ORDINARY CELL (10, 2411).  Binding text: [C] v1.12.0-pre.2 sec:hida --
// def:hidatower (the tower Tw_m(ell; n_0) and its datum), cor:ordprecision
// (the two readings of the ordinary part), rmk:ordtorsionfree,
// lem:ordexponent, and the predictions (Hi2)-(Hi6) of rmk:hidacheckable,
// which HeckeHidaTowerValidation / HeckeHidaTowerSlowValidation test.
//
// WHAT IT COMPUTES, per cell (n, ell), ell >= 5, n >= 2 even, at a working
// precision K (normally K = m_max + e_max, e_max from Hecke/HeckeOrdinaryPart's
// pre-pass, i.e. cor:ordprecision(2) before the run):
//
//  * H_K = H^1(Gamma; Sym^n (Z/ell^K)^2) by the shipped module route
//    (EquivariantCohomologyOnKT<uint64_t> under coefficient_ring = Z/ell^K) as
//    (+)_i Z/ell^{a_i}, and U_ell, T_2, T_3 on it (HeckeOperatorOnKT's
//    H_matrix_module, the shipped convention);
//  * for each target e' <= K - e_max, TWO readings of the ordinary part:
//     (1) on H_{e'} = H_K / ell^{e'} H_K (hecke::ordinary::Reduce): the
//         projector P = U^{N_ord(g, e')} (lem:ordexponent), its rank over
//         F_ell, and the characteristic polynomial of T_p on the free
//         Z/ell^{e'}-module P H_{e'} (cor:ordprecision(1));
//     (2) on the free block F' (the summands of order ell^K,
//         cor:ordprecision(3)), reduced mod ell^{e'}: the same three things
//         (cor:ordprecision(2)-(4)).
//    (Hi2)'s last sentence -- equal rank and equal characteristic polynomials
//    mod ell^{e'} -- is the comparison of (1) with (2);
//  * the characteristic polynomial of U_ell itself on H_K when H_K is free
//    (the (10, 2411) cell, (Hi6)).
//
// The characteristic polynomial of T on the free module P M (M any
// (+)_i Z/ell^{b_i}, P an idempotent commuting with T) is computed in an
// explicit basis: columns of P independent mod ell (Nakayama), coordinates by
// an ell-adic digit solve (P M ∩ ell M = ell P M), then Berkowitz's
// division-free algorithm over Z/ell^{e'}.  Every coordinate solve is
// re-verified (ENGINE_CHECK).
//
// Pipeline side only; writes nothing, emits no record, no golden object.
// ===========================================================================

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "../Hecke/HeckeOrdinaryPart.h"
#include "../LocalSystemUtils.h"   // FibreRepresentation (slot-13 WP-5: Hida's lattice)

namespace hecke_ext {
namespace hida {

using Poly = std::vector<uint64_t>;   // ascending coefficients mod ell^{e'}, monic

// One module run: H_K and the operators on it.
struct ModuleCell {
    uint32_t n_sym = 0;
    uint64_t ell = 0;
    uint32_t K = 0;
    std::vector<uint32_t> exps;                 // H^1 cyclic exponents a_i
    std::vector<uint64_t> primes;               // the Hecke primes run (ell first)
    std::vector<hecke::ordinary::RingMat> op;   // op[i] = the operator of primes[i]
    const hecke::ordinary::RingMat& at(uint64_t p) const;
};
// n >= 1, ell >= 5 prime, K >= 2 (the module route; K = 1 is F_ell and is read
// as a reduction of K >= 2).  primes must start with ell.
// (slot-13 WP-5) rep = FibreRepresentation::HidaLatticeH computes the same on
// Hida's lattice L_n ([C] def:hidalattice variant (H): the local system of
// GenerateSL2ZRelationsRingHida, the operators with rho^H_n); SymN is the
// slot-12 call byte for byte.
ModuleCell RunModuleCell(uint32_t n_sym, uint64_t ell, uint32_t K, const std::vector<uint64_t>& hecke_primes,
                         FibreRepresentation rep = FibreRepresentation::SymN);

// One reading of the ordinary part at target e'.
struct Reading {
    uint32_t e_target = 0;
    uint32_t generators = 0;                    // generator count of the module read
    uint32_t ord_rank = 0;                      // rank_F_ell P (= rank_{Z/ell^e'} P M)
    bool idempotent = false, commutes = false;
    // (slot-12 WP-11, A4) P M free of rank ord_rank over Z/ell^{e'}
    // (cor:ordprecision(1) / rmk:ordtorsionfree): the F_ell-dimension of
    // ell^{e'-1} P M, TopLayerRank(P, e'), equals ord_rank.  CharpolyOnImage
    // ENGINE_CHECKs it; the validation counts it per reading (ord_torsion = 0).
    bool image_free = false;
    std::vector<uint64_t> primes;               // the T_p read (p != ell)
    std::vector<Poly> charpoly;                 // charpoly[i] of T_{primes[i]} on P M
};
// (1): on H_{e'} = Reduce(H_K, e').
Reading ReadOnModule(const ModuleCell& c, uint32_t e_target);
// (2): on the free block F' of H_K (summands of order ell^K), mod ell^{e'};
// requires K >= e' + e_max (ENGINE_CHECKs that exactly h summands have order ell^K
// and the others order <= ell^{K - e'}).
Reading ReadOnFreeBlock(const ModuleCell& c, uint32_t e_target, uint32_t free_rank_h);

// charpoly of the full operator on a FREE module (all a_i == e'), Berkowitz.
Poly CharpolyFree(const hecke::ordinary::RingMat& A);
// dim_F_ell ell^{e'-1} P M = rank over F_ell of P's rows with a_i = e', mod ell
// (rows with a_i < e' are killed by ell^{e'-1}).  P M is free of rank
// RankModEll(P) iff this equals RankModEll(P) (slot-12 WP-11, A4).
uint32_t TopLayerRank(const hecke::ordinary::RingMat& P, uint32_t e_target);
// charpoly of T on P M, P idempotent commuting with T (see the header comment);
// ENGINE_CHECKs that P M is free (TopLayerRank(P, e') == RankModEll(P)).
Poly CharpolyOnImage(const hecke::ordinary::RingMat& P, const hecke::ordinary::RingMat& T, uint32_t e_target);

// ---- (slot-13 WP-5) the module datum ([C] def:hidalattice) -------------------
using Mat = std::vector<std::vector<uint64_t>>;   // square, entries mod ell^{e'} (row i, column j)

// The free module P M over Z/ell^{e'} with its explicit basis (the columns of P
// independent mod ell, as CharpolyOnImage has always chosen them) and the
// ell-adic digit coordinate solve (re-verified).  CharpolyOnImage is
// Berkowitz(MatrixOnImage(FreeImage(P, e'), T)) -- the same code, factored.
struct FreeImage {
    hecke::ordinary::RingMat P;
    uint32_t e = 0;
    std::vector<std::vector<uint64_t>> B;             // basis vectors (elements of M)
    uint32_t rank() const { return static_cast<uint32_t>(B.size()); }
    std::vector<uint64_t> coords(const std::vector<uint64_t>& x) const;   // x in P M
};
FreeImage MakeFreeImage(const hecke::ordinary::RingMat& P, uint32_t e_target);
Mat MatrixOnImage(const FreeImage& F, const hecke::ordinary::RingMat& T);   // column j = coords(T b_j)

// The lattice datum of a cell at level m: (M_m(n), (T_p)_p) in the basis of
// FreeImage(U^{N_ord}, m) on H_m = Reduce(H_K, m) (cor:ordprecision(1)).
struct ModuleDatum {
    uint64_t ell = 0;
    uint32_t m = 0, h = 0;
    std::vector<uint64_t> primes;                     // p != ell
    std::vector<Mat> A;                               // A[i] = T_{primes[i]} on M_m(n)
    std::vector<Poly> charpoly;                       // Berkowitz(A[i]) mod ell^m
};
ModuleDatum ReadModuleDatum(const ModuleCell& c, uint32_t m);

// [C] rmk:hidacheckable2 (Hi8)'s decision procedure (v1.13.0-pre.2, m4): the
// solution module Hom = {P : P A_p = A'_p P for all p} over Z/ell^m by a module
// solve (engine::elim::kernel_module; over F_ell a nullspace at m = 1); an
// isomorphism exists iff det is not identically zero on the reduction Hom-bar,
// a polynomial of degree h, decided on the grid {0..h}^{dim Hom-bar} (h < ell;
// exhaustively over F_ell otherwise, refused by name above 2^22 points).  The
// outcome is isomorphic -- with the certificate P (verified: P A_p = A'_p P over
// Z/ell^m, det P a unit) -- or not isomorphic; there is no third outcome.  The
// pre.1 invariants run first as necessary conditions (charpolys, the lengths of
// Hom(M, M), Hom(M, M'), Hom(M', M')); a pre-check that already decides "not
// isomorphic" is recorded as such (decided_by_precheck).
struct IsoDecision {
    bool isomorphic = false;
    bool decided_by_precheck = false;
    uint32_t len_hom_aa = 0, len_hom_ab = 0, len_hom_bb = 0;   // lengths over Z/ell^m
    uint32_t dim_hom_bar = 0;                                  // dim_F_ell of Hom-bar(M, M')
    uint64_t grid_points = 0;                                  // det evaluations made
    Mat certificate;                                           // P, when isomorphic
    std::string text() const;
};
IsoDecision DecideIsomorphic(const ModuleDatum& a, const ModuleDatum& b);
// The digest of a datum's class, for the HIDA_TOWER line: FNV of (h, m, the
// charpolys) -- class-invariant; the class itself is decided pairwise above.
uint64_t ModuleClassDigest(const ModuleDatum& d);

// ---- (slot-13 WP-5) arm (c): D_* : M^H_m(n) -> M_m(n) ------------------------
// Both lattices computed over Z/ell^K on the same model; D = diag(binom(n, j))
// applied fibrewise to Hida-lattice H^1 class representatives, read in the
// Sym^n cohomology's class coordinates (a cochain map, since D rho^H = rho D).
struct LatticePair {
    uint32_t n_sym = 0;
    uint64_t ell = 0;
    uint32_t K = 0;
    ModuleCell sym, hida;
    // D_* on H^1 over Z/ell^K: column j = the sym-coordinates of D (hida class j).
    std::vector<std::vector<uint64_t>> Dstar;   // sym.exps.size() x hida.exps.size()
    bool Dstar_equivariant = false;             // D A^H_p = A_p D for every p run (entrywise, mod ell^{a_i})
};
LatticePair RunLatticePair(uint32_t n_sym, uint64_t ell, uint32_t K, const std::vector<uint64_t>& hecke_primes);
// D_* restricted to the ordinary parts at level m, in the FreeImage bases: an
// h x h matrix over Z/ell^m; the image check (D_* P^H M^H inside P M) is part of it.
struct DstarReading {
    uint32_t m = 0, h = 0;
    bool image_in_ordinary = false;
    bool isomorphism = false;                   // det Delta a unit mod ell
    uint32_t rank_mod_ell = 0;                  // rank of Delta mod ell
};
DstarReading ReadDstar(const LatticePair& lp, uint32_t m);

uint64_t PowMod(uint64_t b, uint64_t e, uint64_t q);
uint64_t EvalMod(const Poly& f, uint64_t x, uint64_t q);
Poly ReducePoly(const Poly& f, uint64_t q);
std::string PolyText(const Poly& f);

// (Hi3)'s committed table: h^1_ord(n) = 1 at ell in {5, 7}; 3 at ell = 11 for
// n = 0 mod 10; 3 at ell = 13 for n = 10 mod 12; 1 otherwise.
uint32_t PredictedOrdinaryRank(uint32_t n_sym, uint64_t ell);

// One tower cell: pre-pass, K = m_max + e_max, the module run, both readings at
// e' = 1 .. m_max.
struct TowerCell {
    uint32_t n_sym = 0;
    uint64_t ell = 0;
    uint32_t e_max = 0, K = 0, h = 0;
    std::vector<uint32_t> exps;
    std::vector<Reading> module_read;           // index e' - 1
    std::vector<Reading> free_read;             // index e' - 1
};
// (NBCONFIRM S3, KNOWN_ISSUES "hida_tower(n, ell, 1) with e_max = 0") A cell whose K = m_max + e_max is below 2 is refused
// right after the pre-pass with this error (K >= 2 is a precondition of def:hidatower's module route); before S3 it reached
// RunModuleCell's ENGINE_CHECK.  Value-neutral: no cell that ran before is refused.  An std::invalid_argument, so a caller
// catching std::logic_error still sees it.
struct TowerPreconditionError : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};
TowerCell ComputeTowerCell(uint32_t n_sym, uint64_t ell, uint32_t m_max);

// "HIDA_TOWER ell=.. class=.. m=.. n=.. ord_rank=.. charpoly_T2=[..] charpoly_T3=[..]"
std::string HidaTowerLine(const TowerCell& c, uint32_t m);

} // namespace hida
} // namespace hecke_ext

#endif // HECKE_EXT_HECKE_HIDA_TOWER_H
