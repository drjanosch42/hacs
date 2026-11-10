// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_LVALUE_EISENSTEIN_BERNOULLI_ROUTE_H
#define HECKE_EXT_LVALUE_EISENSTEIN_BERNOULLI_ROUTE_H

// ===========================================================================
// (hecke-extensions slot 19, WP-LV) THE EISENSTEIN / BERNOULLI INSTANCE (E) of the
// L-value interface -- [E] Definition def:lveis, Claim cl:lveis (ring form (1),
// finite precision (3)), Remark rmk:lvgates; kind = congruence (v1.19.0-pre.2, F1 M-2).
//
// D: H := H^1(Gamma; Sym^n Z_ell) (mod torsion; refused when H^1 has torsion) on the
// 7-vertex SL_2(Z) model, read as H_K := H^1(Gamma; Sym^n Z/ell^K), K = m_max + 1;
// H' = H^- (the minus sign of iota^*, [C, def:iota](iv): "+" is the Eisenstein side)
// localised at m_Eis = (ell, T_p - 1 - p^{k-1}), k = n + 2.  The character
// e : T_p -> e_p := 1 + p^{k-1} (values in O = Z_ell); I_e = (T_p - e_p : p in S);
// Q_e = T(H') / I_e (Ann I_e = 0: e does not occur in H^-, and T(H^-) is
// torsion-free).  At finite precision m (cl:lveis(3)): A_m := the algebra the T_p
// (p in S) generate in End(H'_m), H'_m := H'/ell^m, and Q_m := A_m / (T_p - e_p).
//
// WHAT IS COMPUTED (one module run per cell, every reading exact over Z/ell^K):
//  (a) H_K = (+) Z/ell^{a_i} and T_p (p in S, S = {2, 3, 5, 7, ell} by default) by the
//      shipped module route (EquivariantCohomologyOnKT<uint64_t> under
//      ScopedCoefficientRing(K), HeckeOperatorOnKT's H_matrix_module(1)) -- the recipe
//      of ringalg::RunModule; refused unless every a_i = K and the count is
//      rank H = 2 d_k - 1 (hecke::ordinary::FreeRankH1): H is then torsion-free and
//      H_K = H / ell^K;
//  (b) iota^* on H_K: the PRODUCTION involution's cochain formula ([C, def:iota](iv),
//      Hecke/HeckeInvolution.h): (iota^* f)(sigma) = -eps_sigma rho_n(iota) f(iota_K sigma),
//      rho_n(iota) = diag((-1)^{n-j}), iota_K and eps from
//      hecke::BuildIotaSimplicialMapOnModel, applied to the module basis cocycles and
//      read back by H_class_coordinates (c5).  ENGINE_CHECKed: iota^2 = 1,
//      iota T_p = T_p iota, rank H^+ = d_k, rank H^- = d_k - 1 (this pins the global
//      sign: the Eisenstein class is on "+");
//  (c) H^{+-} = (1 +- iota)/2 H_K (free direct summands), the T_p restricted;
//  (d) on each sign the joint idempotent e_Eis of the system a_p = 1 + p^{k-1} mod ell
//      (per-p idempotent polynomials, Newton-lifted; a polynomial in the T_p, so in
//      T' = Z_ell[T_p : p in S]) and the block e_Eis H^{+-}, free of rank f;
//  (e) the reading at m: A_m (Howell spans over Z/ell^K of the ell^{K-m}-scaled
//      monomial generators: (Z/ell^m)^N embeds in (Z/ell^K)^N by ell^{K-m}),
//      dim_F A_1, the ideal J_m = sum_p (T_p - e_p) A_m, Q_m = A_m / J_m with its
//      elementary divisors (Smith rule, engine::elim::invariant_factors);
//  (f) SATURATION: dim_F A_1 == f.  Then T' (the Z_ell-algebra of S) is saturated in
//      End(H') of rank f = rank T(H') (on one sign every eigensystem occurs once), so
//      T' = T(H') (T(H') lies in T' (x) Q cap End = T') -- the stable verdict at
//      K = 1 and the generation certificate of cl:lveis(3) at once;
//      automatic on a rank-1 block;
//  (g) the inequality of cl:lveis(3): for p in S, p != ell, the characteristic
//      polynomial c_{k,p} of T_p on H'_m, v(c_{k,p}(e_p)) read mod ell^m.
//
// Pipeline side only, console-free; writes nothing, no golden object, no record.
// ===========================================================================

#include "LValueRoute.h"
#include "Bernoulli.h"
#include "../ZellK.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace hecke_ext {
namespace lvalue {

// A square endomorphism of a FREE Z/ell^K-module (column convention:
// T b_j = sum_i a[i*r + j] b_i), entries in [0, ell^K): the shared kit (../ZellK.h; S24 CL-G1).
using FreeMat = zk::FreeMat;

enum class Sign : uint8_t { Plus = 0, Minus = 1 };
const char* ToString(Sign s);

// The m_Eis block of one sign.
struct SignBlock {
    Sign sign = Sign::Minus;
    uint32_t side_rank = 0;              // rank H^{+-}
    uint32_t f = 0;                      // rank of the m_Eis block
    std::vector<FreeMat> ops;            // T_p on the block, aligned with the cell's primes
};

// One module run (a)-(d).
struct EisensteinCell {
    uint32_t n = 0;
    uint64_t ell = 0;
    uint32_t K = 0;
    uint32_t k() const { return n + 2; }
    std::vector<uint64_t> primes;        // S, ascending, contains ell
    std::vector<uint32_t> exps;          // H^1(Z/ell^K) exponents
    uint32_t rank = 0;                   // rank H = 2 d_k - 1
    std::vector<FreeMat> T;              // on H_K
    FreeMat iota;                        // on H_K
    SignBlock plus, minus;
    std::vector<uint64_t> e_values;      // e_p = 1 + p^{k-1} (twisted: 1 + chi(p) p^{k-1}) mod ell^K, aligned with primes
    double seconds = 0.0;
    // (slot-19 WP-LV2) the free part at finite precision and the twisted instance
    uint32_t K_run = 0;                  // the precision the cohomology was computed at (K_run >= K + torsion_t)
    uint32_t torsion_t = 0;              // the largest exponent of a non-free summand of H^1(Z/ell^{K_run}); 0: none
    uint32_t torsion_summands = 0;       // the number of non-free summands
    uint64_t level = 1;                  // 1, or N = ell on the character model ([C] def:nebmodel)
    uint32_t twist_j = 0;                // chi = omega^j (level ell); 0 at level 1
    uint32_t side_plus = 0, side_minus = 0;   // rank H^+, rank H^-
};

struct EisensteinOptions {
    uint32_t m_max = 3;                  // K = m_max + 1
    std::vector<uint64_t> primes;        // S; empty = {2, 3, 5, 7, ell}
    // (slot-19 WP-LV2) read H/tors on a cell whose H^1 has ell-torsion (the free part at finite precision,
    // RunEisensteinCell's note); false keeps WP-LV's refusal.
    bool torsion_aware = false;
    uint32_t max_extra_precision = 6;    // K_run <= K + this
    uint32_t min_extra_precision = 0;    // start at K_run = K + this (a cross-check: the free part must not depend on it)
};

// n >= 2 even, ell >= 5 prime, (ell - 1) not dividing k = n + 2, ell^{K_run} < 2^62.
// Throws hecke::NotImplementedInThisBuild on a cell whose H^1 has ell-torsion unless opt.torsion_aware.
EisensteinCell RunEisensteinCell(uint32_t n, uint64_t ell, const EisensteinOptions& opt);

// (slot-19 WP-LV2) THE TWISTED INSTANCE ([E] cl:lvtwist) on the character model H^1(Gamma_0(ell); V(chi)),
// chi = omega^j (omega the Teichmuller character mod ell, j even, j != 0 mod ell - 1), over Z/ell^K = W_K(F_ell)
// (slot-18 WP-6's model route: InducedRelationsChi with the Teichmuller-lifted table, T_p (p != ell) / U_ell by
// eq:nebcorr), read on the free part as above (rank h^1 - h^0 of [C] prop:neborbit(4) in characteristic 0).
// iota^* is the level-one cochain formula with rho_n(iota) replaced by the induced action
// (iota_M f)_j = chi(a_h) rho_n(iota) rho_n(h) f_{j'}, iota w_j iota = h w_{j'}, h in Gamma_0(ell).
// The character e_p = 1 + chi(p) p^{k-1} (that of E_k(1, chi)), e_ell = 1.  ell^{K_run} < 2^32.
EisensteinCell RunTwistedEisensteinCell(uint32_t n, uint64_t ell, uint32_t j, const EisensteinOptions& opt);

// (b)-(d) on a cell whose T (aligned with primes) and iota are filled on the free part mod ell^K:
// the iota / T_p consistency checks, the sign split (rank checked against expect_plus / expect_minus when
// both >= 0), the m_Eis blocks of the system e_values mod ell.
void FinishEisensteinCell(EisensteinCell& cell, const std::vector<uint64_t>& e_values, int64_t expect_plus, int64_t expect_minus);

// The reading (e)-(g) of one sign at precision m (1 <= m <= K - 1... m <= K).
// `character` overrides e (a negative control); empty = the Eisenstein e_p.
struct SignReading {
    Sign sign = Sign::Minus;
    uint32_t m = 0;
    CongruenceModule Q;
    std::vector<uint32_t> charpoly_valuations;   // (g): v(c_{k,p}(e_p)) mod ell^m, per p != ell (aligned)
    std::vector<uint64_t> charpoly_primes;
};
SignReading ReadSign(const EisensteinCell& cell, Sign sign, uint32_t m, const std::vector<uint64_t>& character = {});

// (h) [E] cl:lveis(2), the module form, in its lattice reading [H^+ : H^+_! (+) Z_ell E] = ell^c, on the
// m_Eis block of H^+ when it has rank <= 2 (rank 1: the Eisenstein line alone, c = 0).  On a rank-2 block
// X_p := T_p - e_p kills E and acts on the cusp line by a_p - e_p = tr(T_p) - 2 e_p, so X_p = (a_p - e_p) pi,
// pi the cuspidal projector, and c = min{j : ell^j pi integral} = v(a_p - e_p) - v(X_p) (v of a matrix: the
// minimum over its entries), for every p != ell with a_p != e_p mod ell^K -- all such p must agree.
// H^+_! = H^+ cap (cusp (x) Q): restriction to the boundary kills exactly the cuspidal part rationally.
struct ModuleFormReading {
    bool applicable = false;            // f_plus <= 2 and (rank 2) some p with v(a_p - e_p) < K
    uint32_t f_plus = 0;
    uint32_t c = 0;
    std::vector<uint64_t> primes;       // the p read (rank 2)
    std::vector<uint32_t> c_per_p;
};
ModuleFormReading ReadModuleForm(const EisensteinCell& cell);

// The planted negative control of (f): the saturation predicate on an explicit
// commuting family (free rank r over Z/ell^K).  Returns dim_F A_1.
uint32_t PlantedDimA1(const std::vector<FreeMat>& ops);

// ---- the route ---------------------------------------------------------------------
class EisensteinBernoulliRoute final : public LValueRoute {
public:
    // The cell is shared (one module run serves m = 1 .. m_max).  m <= cell.K.
    EisensteinBernoulliRoute(std::shared_ptr<const EisensteinCell> cell, uint32_t m);
    // A domain on another model is refused by name (hecke::NotImplementedInThisBuild).
    static void RequireSupportedDomain(const Domain& d);

    std::string name() const override { return "EisensteinBernoulli"; }
    RouteKind kind() const override { return RouteKind::Congruence; }
    const Domain& domain() const override { return d_; }
    NormalisationRecord normalisation() const override;
    CongruenceModule congruence_module() const override;
    IndexStatement index_statement() const override;

    const SignReading& reading() const;
    const BernoulliReading& bernoulli() const { return bern_; }
    const EisensteinCell& cell() const { return *cell_; }

private:
    std::shared_ptr<const EisensteinCell> cell_;
    uint32_t m_ = 0;
    Domain d_;
    BernoulliReading bern_;
    mutable std::unique_ptr<SignReading> reading_;
};

// "LVALUE route=EisensteinBernoulli kind=congruence n=.. ell=.. k=.. m=.. f=.. dim_A1=..
//  saturated=.. Q=[..] length=.. v(B_k/k)=.. expected=[lo,hi] holds=.. N={...}"
std::string LValueLine(const EisensteinBernoulliRoute& r);

} // namespace lvalue
} // namespace hecke_ext

#endif // HECKE_EXT_LVALUE_EISENSTEIN_BERNOULLI_ROUTE_H
