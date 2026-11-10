// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_LVALUE_ROUTE_H
#define HECKE_EXT_LVALUE_ROUTE_H

// ===========================================================================
// (hecke-extensions slot 19, WP-LV) THE L-VALUE INTERFACE -- [E] sec:lvalue,
// Definition def:lvroute (v1.19.0-pre.2), Remark rmk:lvshapes, the
// normalisation table sec:lvnorm.  Owner requirement of 2026-10-03
// (docs/implementation/slot18/lead_carry.md C9): one abstract interface that
// carries, WITHOUT CHANGE, three shapes of route:
//
//   kind = linear      a functional lambda : H' -> R           (Birch / Manin-Stevens
//                      modular symbols, cycles {r, oo} with twists; (B))
//   kind = bilinear    a pairing <,> : H' x H'' -> R with a     (Hida's GL2 x GL2 pairing,
//                      stated adjunction T_p^* of the T_p       L(1, Ad f); (H))
//   kind = congruence  a Hecke character e = (e_p)_p, I_e = (T_p - e_p) in T(H')_O,
//                      Q_e = T(H')_O / (I_e + Ann I_e)          (the Eisenstein / Bernoulli
//                                                                instance (E), implemented)
//
// A route is the tuple (D, kappa, N, ev, I) of def:lvroute: the DOMAIN D (a named
// model of [C] -- a parameter, never hard-wired: the MP constraint of the slot-19
// plan -- the cohomology, the coefficient ring Z_ell or Z/ell^m, the named
// Hecke-stable subquotient H'), the KIND kappa, the NORMALISATION RECORD N (fixed
// before anything is computed, rule 20; printed with every value), the EVALUATION
// ev (one entry point per kind; the two that do not match kind() refuse) and the
// INDEX STATEMENT I with its own status tag.
//
// NOT USED: Harder's route to rationality through the boundary spectral
// sequence (owner, 2026-10-03).
//
// Both declared shapes are IMPLEMENTED: the (B) instance (BirchModularSymbolRoute,
// kind linear) since slot-19 WP-LIN (BirchModularSymbol.{h,cpp}), the (H) instance
// (HidaPairingRoute, kind bilinear) since slot-19 WP-BIL (HidaPairingRoute.cpp).
// Pipeline side only, console-free, writes nothing,
// no golden object, no record.
// ===========================================================================

#include "BirchModularSymbol.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace hecke_ext {
namespace lvalue {

enum class RouteKind : uint8_t { Linear = 0, Bilinear = 1, Congruence = 2 };
const char* ToString(RouteKind k);

// The named models of [C] a domain may live on.  New models (the refined
// Moerdijk-Pronk triangulation, the Gamma_0(N) Shapiro model, the character model of
// [C] def:nebmodel) are added HERE, as values; a route that does not support a
// model refuses it by name (NotImplementedInThisBuild).
enum class ModelId : uint8_t {
    SL2ZSevenVertex = 0,    // [C]'s 7-vertex SL_2(Z)\H model (LocalSystemSL2Z)
    Gamma0NShapiro = 1,     // [C] level N by Shapiro on the fixed model
    CharacterModel = 2,     // [C, def:nebmodel] H^1(Gamma_0(N); V_R(chi))
    MoerdijkPronk = 3       // the refined MP-conformant model (future)
};
const char* ToString(ModelId m);

// D of def:lvroute(1).
struct Domain {
    ModelId model = ModelId::SL2ZSevenVertex;
    std::string cohomology = "H^1";     // "H^1" | "H^1_c" | "H^1(Gamma_0(N);V(chi))"
    uint32_t n_sym = 0;                 // the fibre Sym^n (k = n + 2)
    uint64_t level = 1;                 // N
    uint64_t ell = 0;
    uint32_t m = 0;                     // R = Z/ell^m; 0 means Z_ell
    std::string subquotient;            // H', e.g. "H^-_{m_Eis}"
    std::string to_string() const;
};

// One removed Euler factor or normalising constant, as an exact rational (decimal
// strings) with its ell-adic valuation; text when not rational (a Gauss sum, a period).
struct NormalisingConstant {
    std::string name;                   // e.g. "Euler factor at ell", "Gauss sum"
    std::string value;                  // exact text: "1 - ell^{k-1}", "-B_k/k", ...
    int32_t ell_valuation = 0;          // its v_ell (exact; 0 for a unit)
    bool valuation_known = true;
};

// N of def:lvroute(3).
struct NormalisationRecord {
    std::string l_function;             // "zeta", "L(f,chi,s)", "L(s,Ad f)"
    std::string s0;                     // the point
    std::string twist;                  // "chi = 1" | "chi = omega^j, conductor ell" | ...
    char sign_at_infinity = '-';        // the iota-eigenvalue ('+' = the Eisenstein side)
    std::vector<NormalisingConstant> euler_factors_removed;
    std::string period;                 // "none" | "Omega_f^{+-}" | "Petersson norm"
    std::vector<NormalisingConstant> constants;
    std::string ambiguity;              // "Z_ell^x": index statements are equalities of ideals
    std::string to_string() const;      // one line, printed with every value
};

// A class of the domain: coordinates on its cyclic-summand basis (+)_i Z/ell^{e_i}.
struct CohomologyClass {
    std::vector<uint32_t> exponents;
    std::vector<uint64_t> coordinates;
};

// A Hecke character e = (e_p)_{p in S}: values in O = R (residues mod ell^precision);
// a named finite extension O is carried by `extension` (empty = R itself).
struct HeckeCharacter {
    std::string name;                   // e.g. "Eisenstein 1 + p^{k-1}"
    std::vector<uint64_t> primes;       // S
    std::vector<uint64_t> values;       // e_p mod ell^precision
    uint32_t precision = 0;
    std::string extension;
};

// The finite module Q of an evaluation, with its saturation reading.
struct CongruenceModule {
    HeckeCharacter character;
    uint32_t m = 0;                     // R = Z/ell^m
    uint32_t block_rank = 0;            // rank of H' (free), the rank of T(H')
    uint32_t dim_A1 = 0;                // dim_F of the algebra generated mod ell
    bool saturated = false;             // [E] cl:lveis(3): stable at K = 1 (+ generation)
    uint32_t length_A = 0;              // length of A_m = T(H') / ell^m (as read)
    uint32_t length_I = 0;              // length of the ideal generated by the T_p - e_p
    uint32_t length = 0;                // length(Q) = length_A - length_I
    std::vector<uint32_t> elementary_divisors;   // Q = (+) Z/ell^{d_i}
    bool ann_term_zero = true;          // Ann(I_e) = 0 is the route's claim (e not in H'), not computed
    // (slot-19 WP-LV2) [E] cl:lveis(3) without saturation: c_sat with ell^{c_sat} T^sat in T (T^sat = T (x) Q cap End),
    // read from End_K / T_K (the largest elementary divisor below K, when exactly f^2 - f divisors equal K);
    // 0 when saturated, -1 when not determined at precision K.
    int32_t c_sat = -1;
};

// I of def:lvroute(5).
struct IndexStatement {
    std::string label;                  // e.g. "[E] cl:lveis(3)"
    std::string status;                 // \PROVED | \STD | \CONJ | ... (the tag of [E])
    std::string module_name;            // Q
    std::string l_value;                // L^alg, exact text
    uint32_t l_valuation = 0;           // v(L^alg)
    bool l_valuation_exact = true;
    // the assertion as computed: expected and found lengths (or bounds)
    uint32_t expected_lo = 0, expected_hi = 0;
    uint32_t found = 0;
    bool holds = false;
    std::string to_string() const;
};

// (slot 20, WP-MEAS; [E] def:lvmeasure, def:lvmtt) the measure-valued evaluation of a linear route (owner ruling O5:
// no fourth kind).  Complete types in ModularSymbolMeasure.h; LValueRoute::measure is defined in ModularSymbolMeasure.cpp.
struct MeasureRequest;
struct ModularSymbolMeasure;

// THE INTERFACE.  Every route answers kind(), domain(), normalisation() and
// index_statement(); exactly one of the three evaluations matches kind(), the
// other two refuse with std::logic_error (a wrong call, not a build capability).
class LValueRoute {
public:
    virtual ~LValueRoute();
    virtual std::string name() const = 0;
    virtual RouteKind kind() const = 0;
    virtual const Domain& domain() const = 0;
    virtual NormalisationRecord normalisation() const = 0;

    // kind = linear: lambda(c) for each class, in R.
    virtual std::vector<uint64_t> evaluate_functional(const std::vector<CohomologyClass>& classes) const;
    // kind = bilinear: <x, y> in R, and the adjoint T_p^* named.
    virtual uint64_t evaluate_pairing(const CohomologyClass& x, const CohomologyClass& y) const;
    virtual std::string adjunction() const;
    // kind = congruence: Q_e.
    virtual CongruenceModule congruence_module() const;

    // The index statement of the route at its domain, computed.
    virtual IndexStatement index_statement() const = 0;

    // (slot 20, WP-MEAS) kind = linear: the measure-valued evaluation x |-> mu_x(a + ell^M Z_ell) ([E] def:lvmeasure).
    // Default: refuse_kind("measure") for kind() != Linear; hecke::NotImplementedInThisBuild for a Linear route that
    // does not implement it.  Include ModularSymbolMeasure.h to call it.
    virtual ModularSymbolMeasure measure(const CohomologyClass& x, const MeasureRequest& req) const;

protected:
    [[noreturn]] void refuse_kind(const char* entry) const;
};

// ---- (B) Birch / Manin-Stevens modular symbols: IMPLEMENTED (slot-19 WP-LIN) -----------
// D = H^1_c of [C]'s 7-vertex model (the C++ mapping cone over F_ell or Z/ell^m), kind linear,
// lambda_{chi,j}(phi) = ev_{chi,j}(phi) = sum_{a mod M} conj(chi)(a) < rho_n([[M,-a],[0,1]]) Phi({a/M, oo}), u_{n-j}^vee >
// (= the pairing of phi({a/M} - {oo}) with rho_n^vee(adj B_{a,M}) u_{n-j}^vee, [E] rmk:lvshapes (B)), giving
// L(f, chi, j+1) / Omega_f^{+-} = chi(-1)(-2 pi i)^{j+1} tau(chi) / (j! C(n,j) M^{j+1}) * lambda, +- the GEOMETRIC sign
// chi(-1)(-1)^j ([C]: -chi(-1)(-1)^j).  The modular symbol of a cone class is Phi({0,oo}) = (rho(S)-1)(c(v0v4) +
// c(v4v6) - c_d(v0)) (BirchModularSymbol.h).  The model is a parameter: SL2ZSevenVertex (level one, the cone) and, since
// slot-22 P6, Gamma0NShapiro (level N by Shapiro on the same model: the symbol of Ind V = R[P^1(Z/N)] (x) Sym^n, the
// domain presented by W(Ind V), classes on its generators; Gamma0NShapiro.h); any other model is refused by name
// (NotImplementedInThisBuild); ell >= 5; ell does not divide M; gcd(M, N) = 1.  index_statement() is R19-8's (L1):
// the Eisenstein congruence r^+ == unit * p_0 on H^1_c to depth v(B_k/k) (level one only; refused at Gamma0NShapiro).
namespace shapiro { struct Gamma0NCell; }
class BirchModularSymbolRoute final : public LValueRoute {
public:
    // The declared constructor (kept): chi = the quadratic primitive character of conductor M (unique sign of
    // +-M a fundamental discriminant; M = 1: chi = 1); the cone is built lazily over R = Z/ell^m (F_ell at m = 1).
    BirchModularSymbolRoute(Domain d, uint64_t twist_conductor, uint32_t j);
    // On a cone cell already built (shared, not copied), with an explicit character.
    BirchModularSymbolRoute(std::shared_ptr<const birch::BirchConeCell> cell, birch::DirichletCharacter chi, uint32_t j);
    std::string name() const override { return "BirchModularSymbol"; }
    RouteKind kind() const override { return RouteKind::Linear; }
    const Domain& domain() const override { return d_; }
    NormalisationRecord normalisation() const override;
    std::vector<uint64_t> evaluate_functional(const std::vector<CohomologyClass>& classes) const override;
    IndexStatement index_statement() const override;
    // (slot 20, WP-MEAS) the ell-stabilised Mazur-Tate-Teitelbaum measure of an ordinary T_ell-eigenclass ([E] def:lvmtt);
    // defined in ModularSymbolMeasure.cpp.  Refuses "not a T_ell eigenclass" / "not ordinary" (std::invalid_argument).
    ModularSymbolMeasure measure(const CohomologyClass& x, const MeasureRequest& req) const override;

    // The whole j-vector ev_{chi,0..n} per class (R19-8 Sec. 6, step 4's second entry).
    std::vector<std::vector<uint64_t>> evaluate_all_j(const std::vector<CohomologyClass>& classes) const;
    // P = Phi({0, oo}) of a class (the Manin relations are ENGINE_CHECKed on it: G2).
    birch::Vec symbol(const CohomologyClass& x) const;            // at Gamma0NShapiro: the flattened (P_x)_x
    const birch::BirchConeCell& cell() const;                     // SL2ZSevenVertex only
    const shapiro::Gamma0NCell& shapiro_cell() const;              // (slot-22 P6) Gamma0NShapiro only: W(Ind V), sign 0
    const birch::DirichletCharacter& character() const { return chi_; }
    uint32_t j() const { return j_; }
    // The domain refusal (model, cohomology, ell, n, M); throws NotImplementedInThisBuild / std::invalid_argument.
    static void RequireSupportedDomain(const Domain& d, uint64_t twist_conductor);

private:
    Domain d_;
    uint64_t M_ = 1;
    uint32_t j_ = 0;
    birch::DirichletCharacter chi_;
    mutable std::shared_ptr<const birch::BirchConeCell> cell_;
    mutable std::shared_ptr<const shapiro::Gamma0NCell> ncell_;
};

// ---- (H) Hida's GL2 x GL2 pairing: IMPLEMENTED (slot-19 WP-BIL, HidaPairingRoute.cpp on HidaPairing.{h,cpp}) --------------
// <Phi, c> := <P, y_c> on H^1_c(V) x H^1(D) = W x D/(D^S + D^U) over Z/ell^m (R19-9 N3; perfect for every even n, ell >= 5;
// = -1 x Hecke/HeckePairing.h's cone cup form), T_p^* = T_p at level 1, iota anti-isometric.  D.cohomology names the lattice:
//   "H^1_c(M_n) x H^1(D_n)" (also "H^1", "H^1_c")  the mixed perfect form, x on the W basis, y on the dual basis y_j;
//   "H^1_! x H^1_!" (also "H^1_!")                 Hida's self-dual form G_!(P, Q) = <P, theta iota_!(Q)>, n + 1 < ell;
//   "H^1_c(M_n) x H^1_!(D_n)"                       P9, y on the basis of W_D = H^1_c(D), read through iota_!.
// disc = the congruence number = L(1, Ad f) / (Omega^+ Omega^-) up to Hida's Thm 5.1 constant ([E] rmk:lvshapes (H)); the
// index statement compares the bilinear evaluation with the congruence module of the lattice read from A_c(T_2) alone (R19-9
// Sec. 3 lemma) -- a route may carry two kinds.  Other models / levels refuse (hecke::NotImplementedInThisBuild).
namespace hidapair { class HidaPairingCell; }
class HidaPairingRoute final : public LValueRoute {
public:
    explicit HidaPairingRoute(Domain d);
    std::string name() const override { return "HidaPairing"; }
    RouteKind kind() const override { return RouteKind::Bilinear; }
    const Domain& domain() const override { return d_; }
    NormalisationRecord normalisation() const override;
    uint64_t evaluate_pairing(const CohomologyClass& x, const CohomologyClass& y) const override;
    std::string adjunction() const override;
    IndexStatement index_statement() const override;

private:
    Domain d_;
    mutable std::shared_ptr<const hidapair::HidaPairingCell> cell_;   // built on first evaluation
    const hidapair::HidaPairingCell& cell() const;
};

} // namespace lvalue
} // namespace hecke_ext

#endif // HECKE_EXT_LVALUE_ROUTE_H
