// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_HECKE_LAMBDA_TOWER_H
#define HECKE_EXT_HECKE_LAMBDA_TOWER_H
// ===========================================================================
// (hecke-extensions slot 22, P10) THE REAL HIDA TOWER: the ordinary part of H^1(Gamma^t_r; Z/ell^K(omega^{a-2})),
// Gamma^t_r = Gamma_{H_r}(ell^r), H_r = mu_{ell-1} (the Teichmueller subgroup), as a module over
// Lambda^{(r)}_K = Lambda/(ell^K, Omega_r) = (Z/ell^K)[G_r], G_r = <g>, g = <u>, u = 1 + ell, 1 + X -> u^2 g.
// Spec: docs/implementation/slot22/P10_spec.md (v0.3).  Text: [U] 0.6.0-pre.1 part 5 (sec:u-tower: def:u-towerlambda,
// lem:u-omegar, def:u-towerobj, prop:u-towerorient, thm:u-towercontrol, prop:u-towereis, prop:u-towerfree,
// cor:u-towertrunc, pr:u-tower (U18)-(U21)); [C] v1.22.0-pre.1 sec:nebwildsec.
//
// HOW (spec section 3): the table and the tame character model on WAKE-1's GammaHLevelN (Hecke/HeckeGammaHTower), the
// cokernel over Z/ell^K with unit pivots (lem:nebchainrref), U = U_ell and T_p (p <= 29) by prop:nebgammaHheilbronn, the
// diamonds <g>, <p>; the ordinary part by a certified block Krylov iteration of U (G5; deterministic U^T on the whole
// module at r <= 2, G5'); the split by the residual Eisenstein system of T_2 (BF'); on each part a basis over
// (Z/ell^K)[G_r] by Nakayama (G6) and the operators as elements of the group ring (G7, G9); the Weierstrass
// representatives A_p(X).
//
// Pure host C++ (HeckeExt -> Hecke one way).  Writes nothing; no golden object; no digest of record.  Refusals are
// ENGINE_CHECKs / std::logic_errors whose message starts with the name: krylov_unstable, tower_rss_cap, chainring_pivot,
// tower_split (the residual split is not a direct sum over Z/ell^K), fibre_precision.
// ===========================================================================

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "../Hecke/HeckeGammaHTower.h"

namespace hecke_ext {
namespace tower {

using hecke::level::tower::ZK;

// ---- dense matrices over Z/ell^K (row-major) ---------------------------------------------------------------
struct DM {
    uint32_t r = 0, c = 0;
    std::vector<uint32_t> a;
    DM() = default;
    DM(uint32_t rows, uint32_t cols) : r(rows), c(cols), a(static_cast<size_t>(rows) * cols, 0) {}
    uint32_t& at(uint32_t i, uint32_t j) { return a[static_cast<size_t>(i) * c + j]; }
    uint32_t at(uint32_t i, uint32_t j) const { return a[static_cast<size_t>(i) * c + j]; }
    bool operator==(const DM& o) const { return r == o.r && c == o.c && a == o.a; }
};
DM Ident(uint32_t n, const ZK& R);
DM MatMul(const DM& A, const DM& B, const ZK& R);
DM MatSub(const DM& A, const DM& B, const ZK& R);
DM MatPow(DM A, uint64_t e, const ZK& R);
DM ScalarSub(const DM& A, uint32_t lambda, const ZK& R);    // A - lambda I
uint32_t RankModEll(const DM& A, uint32_t ell);
std::vector<uint32_t> IndepColsModEll(const DM& A, uint32_t ell);   // greedy, left to right
std::vector<uint32_t> PivotRowsModEll(const DM& A, uint32_t ell);   // rows of a maximal minor invertible mod ell, smallest first
DM Inverse(const DM& A, const ZK& R);                                // ENGINE_CHECK invertible mod ell
DM Cols(const DM& A, const std::vector<uint32_t>& cols);
DM Rows(const DM& A, const std::vector<uint32_t>& rows);
std::vector<uint32_t> Charpoly(const DM& A, const ZK& R);           // Berkowitz, monic, low -> high
// Kernel and image of a square A over Z/ell^K when Z/ell^K^n = ker A (+) im A is not assumed: the unit-pivot RREF of A;
// ok = false when a non-unit residual remains (A is not "split": its Smith form has a non-unit, non-zero entry).
struct KerIm { bool ok = false; DM ker, im; };
KerIm KernelImage(const DM& A, const ZK& R);

// A free submodule of (Z/ell^K)^n with a basis independent mod ell: coordinates by the rows of an invertible minor.
class Subspace {
public:
    Subspace() = default;
    Subspace(const DM& basis, const ZK& R);                  // columns = the basis
    uint32_t dim() const { return B_.c; }
    const DM& basis() const { return B_; }
    const std::vector<uint32_t>& pivot_rows() const { return P_; }
    // x in the span: coordinates c with B c = x; false if x is not in the span.
    bool coords(const std::vector<uint32_t>& x, std::vector<uint32_t>& c) const;
    // The matrix of an operator given its images of the basis vectors (columns of XB): false if one leaves the span.
    bool restrict(const DM& XB, DM& out) const;
    DM canonical() const;                                   // B (B[P])^{-1}: unique for the module
private:
    ZK R_;
    DM B_, Ainv_;
    std::vector<uint32_t> P_;
};

// ---- the tower ----------------------------------------------------------------------------------------------
struct TowerCell {
    uint32_t ell = 0, a = 0, r = 1, K = 3;
    std::string text() const;                                // "(ell, a, r)"
};

struct TowerOptions {
    bool fl_crosscheck = true;           // the F_ell RREF of WAKE-1's SparseCokernel against the chain-ring pivots and tails
    bool deterministic = false;          // G5': e = U^T on the whole module (r <= 2)
    bool rel_escapes = true;             // G3: Rel preserved by every operator
    bool have_tame_override = false;     // NEGATIVE CONTROLS only: the tame exponent e instead of a - 2
    int64_t tame_override = 0;
    int64_t krylov_fixed_T = -1;         // NEGATIVE CONTROL only: T fixed, no doubling, no refusal (the certificate is reported)
    uint32_t R_pred = 0;                 // the predicted ordinary rank (sets the block size s = R_pred + 16)
    uint64_t byte_cap = 0;               // tower_rss_cap for the elimination (0: none)
    std::vector<uint64_t> primes;        // the T_p read (p != ell); default 2 .. 29
    uint64_t seed = 0x5032325031300aull; // the Krylov block's seed (fixed)
};

// An element of (Z/ell^K)[G_r]: coefficients of g^0 .. g^{L-1}.
using GR = std::vector<uint32_t>;

struct TowerPart {
    std::string kind;                    // "eis" | "cusp"
    uint32_t rank = 0;                   // over Z/ell^K
    uint32_t lambda_rank = 0;            // dim_{F_ell} part / (ell, g - 1)  (G6)
    bool basis_ok = false;               // the g-orbits of the lifted generators form a basis (unit determinant mod ell; G6)
    bool scalar_ok = false;              // every operator is a scalar of (Z/ell^K)[G_r] on this basis (G7 / G9)
    std::map<std::string, GR> scalars;   // "T2", ..., "U": the scalar (the (1,1) entry when !scalar_ok)
    std::map<std::string, std::vector<uint32_t>> weierstrass;   // A_p(X), deg < L
    std::vector<uint32_t> residual;      // per p in {2, 3, 5, 7}: the residual eigenvalue (UINT32_MAX if not unique)
    std::map<std::string, std::vector<uint32_t>> charpoly;      // "T2", "T3", "U" on the part (Z/ell^K)
};

struct TowerReading {
    TowerCell cell;
    uint64_t N = 0;
    uint32_t m = 0, h = 0, L = 1, D = 0;
    uint32_t teich_gen = 0;              // the generator of H_r used
    // G1 / G2 / G3
    bool table_r1_ok = true;             // r = 1: w_J = def:cosetrepsN's w_j entry for entry (and Gamma0LevelN's)
    bool fl_pivots_ok = true, fl_tails_ok = true;
    hecke::level::tower::ChainCokernelStats elim;
    std::map<std::string, uint32_t> rel_escapes;
    bool commute_ok = false, diamond_p_ok = false, diamond_H_ok = false, tpp_ok = false;
    std::map<std::string, uint64_t> nnz;
    // G5
    uint32_t krylov_T = 0, krylov_s = 0, rank = 0;
    std::vector<uint32_t> krylov_ranks;  // the F_ell rank of the block at T0, 2T0, 4T0, ...
    bool krylov_stable = false, E_free = false, E_Ustable = false, U_invertible = false;
    bool g5prime_run = false, g5prime_ok = false;
    // operators on E (E-coordinates)
    std::map<std::string, DM> onE;
    std::map<std::string, std::vector<uint32_t>> charpolyE;     // "T2", "T3", "U" on E
    std::map<std::string, std::vector<uint32_t>> eigsE;         // "T2", "T3": multiset of residual eigenvalues in F_ell
    // the split
    bool split_ok = false;
    std::vector<TowerPart> parts;        // eis first, then cusp (if any)
    std::vector<uint32_t> s_of_p;        // s(p) for p in primes (<p>_ell = u^{s(p)} mod ell^r)
    std::vector<uint64_t> primes;
    double seconds = 0, setup_s = 0, elim_s = 0, ops_s = 0, krylov_s_wall = 0;
    double peak_rss_mib = 0;
    const TowerPart* part(const std::string& kind) const;
};

// The full reading of one cell (spec section 3 steps 1-7).  Throws the named refusals.
TowerReading BuildTower(const TowerCell& cell, const TowerOptions& opt);
// (slot 23, G-EVT23) The default-arm set-up of HeckeLambdaTowerValidation, moved verbatim so the golden harness's
// `TOWER ell=<l> a=<a> r=<r>` points use the same recipe (value-neutral): d(a) of [U] thm:u-ell13 at ell <= 13, the
// predicted ordinary rank R_pred = ell^{r-1} (2 d(a) + 1) (it sets only the Krylov block size s = R_pred + 16; the rank
// itself is read, never assumed), and the default options (R_pred, deterministic = r <= 2; every other field default).
uint32_t TowerDofA(uint32_t ell, uint32_t a);
uint32_t TowerRankPred(uint32_t ell, uint32_t a, uint32_t r);
TowerOptions DefaultTowerOptions(const TowerCell& cell);

// <d>_ell = u^{s} mod ell^r: s in [0, ell^{r-1}) (brute force; ell^{r-1} small).
uint32_t LogU(uint64_t d, uint32_t ell, uint32_t r);
// The Weierstrass representative of c(g) in (Z/ell^K)[X]/(Omega_r): sum_i c_i u^{-2i} (1 + X)^i (deg < L).
std::vector<uint32_t> Weierstrass(const GR& c, const ZK& R);
// c(g) at g -> u^{k-2}, mod ell^{prec} (prec <= K): sum_i c_i u^{(k-2) i}.
uint32_t FibreAt(const GR& c, uint32_t ell, int64_t k, uint32_t prec);
// c(g) reduced to G_{r'}: g^{L'} -> 1 (L' = ell^{r'-1}).
GR ReduceTo(const GR& c, uint32_t Lp, const ZK& R);
// The Eisenstein scalar of prop:u-towereis: 1 + p omega^{a-2}(p) g^{s(p)} in (Z/ell^K)[G_r].  untwisted: the NEGATIVE
// CONTROL 1 + X -> <u> (the u^2 dropped): 1 + omega(p)^{a-1} <p>^{-1} g^{s(p)}.
GR EisensteinScalar(uint32_t ell, uint32_t a, uint32_t r, uint64_t p, const ZK& R, bool untwisted = false);

// ---- the Gamma_1 oracle (NB10, G4) ---------------------------------------------------------------------------
struct Gamma1OracleClass {
    uint32_t a = 0;
    uint32_t trace_e = 0;                // tr(e_chi | H^1(Gamma_1)) mod ell^K (= rank of the chi-part)
    uint32_t rank = 0;                   // the ordinary rank of e_chi H^1
    bool certified = false;              // U-stable, U invertible, in the chi-eigenspace of the diamonds
    std::map<std::string, std::vector<uint32_t>> charpoly;      // "T2", "T3", "U"
};
struct Gamma1Oracle {
    uint32_t ell = 0, r = 1, K = 3;
    uint32_t m = 0, h = 0;
    std::vector<Gamma1OracleClass> classes;
    double seconds = 0;
};
// Gamma_1(ell^r) (H = {1}; the general relations of order 4 / 6), trivial coefficients; for every class a in `classes`
// the tame projector e_chi (chi = omega^{a-2}), its ordinary part by the same certified Krylov, and the charpolys.
// R_pred[a] sets the Krylov block size.
Gamma1Oracle BuildGamma1Oracle(uint32_t ell, uint32_t r, uint32_t K, const std::vector<uint32_t>& classes,
                               const std::map<uint32_t, uint32_t>& R_pred);

// The record hecke_lambda_tower/v1 (spec section 9), one line per (cell, component); FNV-1a-64 of the line before
// " digest=".  Not pinned.
std::vector<std::string> TowerRecordLines(const TowerReading& t, const std::map<std::string, std::string>& extra);

} // namespace tower
} // namespace hecke_ext

#endif // HECKE_EXT_HECKE_LAMBDA_TOWER_H
