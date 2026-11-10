// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef ORBIFOLD_MP_MP_HECKE_H
#define ORBIFOLD_MP_MP_HECKE_H

// ===========================================================================
// (hecke-extensions slot 22, MP-4) Hecke operators ON THE MP MODEL
// ([C] 1.22.0-pre.6 rmk:mphecke, def:mpmorphism, prop:mprescor;
// MP_impl_plan.md v0.2.1 Sec. 2.5; MP_math_plan.md v0.2.1 Sec. 7.3).
//
// THE DOUBLE COSET.  g = diag(1, p), Gamma'' = Gamma_0(p) = Gamma n g Gamma g^{-1},
// Gamma^0(p) = Gamma n g^{-1} Gamma g, conj : Gamma'' -> Gamma^0(p), h |-> g^{-1} h g;
// s_1 .. s_m (m = p + 1) the SHIPPED representatives of Gamma'' \ Gamma
// (hecke::HeckeCosetTables, read-only), delta_j = s_j^{-1} g, and the (T1) split
// gamma = h s_j -> g^{-1} h g of HeckeCosetTables::twisted_conjugate.
//
// THE TWISTED CHAIN DATUM ON THE MP RESOLUTION (Def. tau of [C], lem:tauexists, on
// P = N Z[F'_.] -- free over Z[Gamma''] on { s_j e_s : j, s in S_k^nd }):
//   tau(h s_j e_s) := (g^{-1} h g) tau(s_j e_s)      (h in Gamma'', (T1) by construction),
//   tau_0(s_j e_s) := e_s (aug 1),
//   tau_k(s_j e_s) := MPFill::fill(tau_{k-1}(d (s_j e_s)))   (k >= 1),
// a chain map over conj, unique up to homotopy (comparison theorem: P is projective
// over Gamma'' and acyclic over Gamma^0(p)).  NOTHING of the shipped tau tables or of
// Iso_. enters: the filling is the MP model's own (MPFill.h), the twist is the group
// homomorphism.  This is the route of [C, rmk:mphecke] that needs no common
// refinement of X and gX (gap G3): X stays one Gamma-complex, the double coset acts
// through the comparison theorem on ITS resolution.
//
// THE OPERATOR  T_p = cor^Gamma_{Gamma''} o (conj_g) o res^Gamma_{Gamma^0(p)}  on
// C^q(S; Sym^n) = Hom_Gamma(N Z[F'_q], M), in three explicit cochain maps:
//   res    : f |-> f restricted to Gamma^0(p) (the inclusion of invariants);
//   conj_g : F |-> F',  F'(y) := rho_n(g) F(tau(y))  -- Gamma''-equivariant BECAUSE of the
//            factor rho_n(g) (rho_n(g) rho_n(g^{-1} h g) = rho_n(h) rho_n(g)): THE LABEL
//            FACTOR of the g-twist ([C, def:mpmorphism] (3), "the g-twist alike");
//   cor    : F' |-> sum_j rho_n(s_j^{-1}) F'(s_j .)  (the transfer, [C, prop:mprescor] (1)).
// In Shapiro coordinates (C^q(S; Coind_{Gamma''} M), phi_j(s) = F'(s_j e_s), the module
// Coind with blocks (j, j.gamma) = rho_n(s_j gamma s_{j.gamma}^{-1})):
//   CgRes_q : C^q(S; M) -> C^q(S; Coind M),  (CgRes f)_j(s) = sum c rho_n(g gamma) f(e_{s'})
//             over the terms c gamma e_{s'} of tau_q(s_j e_s)  -- a cochain map;
//   Cor_q   : the coefficient map (phi_j) |-> sum_j rho_n(s_j^{-1}) phi_j;
//   T_q = Cor_q o CgRes_q,  (T f)(e_s) = sum_j sum c rho_n(delta_j gamma) f(e_{s'}),
// the [C, eq:cosettransport] form with delta_j gamma formed exactly (det p, never
// inverted).  Factor::Dropped replaces rho_n(g) by the identity (the NC of
// thm:mpnatural (4) for the g-twist: no longer Gamma''-equivariant, no cochain map);
// Factor::WrongRep uses g' = g T (another element of Gamma g Gamma) with the datum
// built for g (the NC "label factor mismatched with the datum", audit F4: gT is a fine element of Gamma g Gamma,
// the failure is the factor rho_n(gT) in front of f o tau with tau built for g).
//
// Exact int64 (hecke overflow checks); every operator entry reduced mod ell^e <= 255;
// no golden object; no setting.  Pure host C++.
// ===========================================================================

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "MPFill.h"
#include "ModuleCochains.h"
#include "Nerve.h"
#include "SparseLength.h"

namespace hecke {
template<class Elt> class HeckeCosetTablesT;
}

namespace orbifold_mp {

struct MPHeckeCertificate {
    bool aug = false;          // (T3) aug tau_0 = aug on every generator
    bool chain_map = false;    // (T2) d tau_k(s_j e_s) = tau_{k-1}(d (s_j e_s)), exactly, every generator
    bool schreier = false;     // (T1) helper: gamma = h s_j with h in Gamma_0(p), conj = [[a, p b], [c / p, d]]
    uint64_t generators = 0, schreier_checks = 0;
    std::string first_failure;
    bool all() const { return aug && chain_map && schreier; }
};

class MPHeckeDatum {
public:
    enum class Factor : uint8_t { Label = 0, Dropped = 1, WrongRep = 2 };

    MPHeckeDatum(const InternedMPSet& S, const Descriptor& D, const hecke::HeckeCosetTablesT<uint32_t>& cosets, uint64_t p,
                 uint32_t k_top);

    uint64_t p() const { return p_; }
    uint32_t m() const { return m_; }
    uint32_t k_top() const { return static_cast<uint32_t>(start_.size()) - 1; }
    // tau_k(s_j e_s), j in [1, m], s the nd index, as a concrete chain.
    StrChain value(uint32_t k, uint32_t j, uint64_t s) const;
    // tau_k(gamma e_s) for arbitrary gamma in Gamma (through (T1)).
    StrChain on_translate(const Mat22& gamma, uint32_t k, uint64_t s) const;

    MPHeckeCertificate certify() const;
    // NC: add one unit term (the identity translate of e_s) to tau_k(s_j e_s).
    void perturb_for_negative_control(uint32_t k, uint32_t j, uint64_t s);

    // T_q on C^q(S; Sym^n) over Z/ell^e: rows, columns S_q^nd x (n + 1).
    SparseRows Operator(uint32_t q, uint32_t n, uint32_t ell, uint32_t e, Factor f = Factor::Label) const;
    // The factorisation (header): Coind_{Gamma_0(p)} Sym^n with the shipped s_j; CgRes_q; Cor_q.
    ModuleRep Coind(uint32_t n, uint32_t ell, uint32_t e) const;
    SparseRows CgRes(uint32_t q, uint32_t n, uint32_t ell, uint32_t e, bool drop_factor = false) const;
    SparseRows Cor(uint32_t q, uint32_t n, uint32_t ell, uint32_t e) const;

    // Statistics (the report, rule 19).
    uint64_t total_terms() const;
    uint64_t terms(uint32_t k) const { return terms_.at(k).size(); }
    int64_t max_abs_coefficient() const;
    double build_seconds() const { return build_s_; }
    const FillStats& fill_stats() const { return fill_.stats(); }

private:
    struct Term {
        uint64_t s;      // nd index in S_k
        uint32_t g;      // index into elems_
        int64_t c;
    };
    const InternedMPSet* S_;
    const hecke::HeckeCosetTablesT<uint32_t>* cosets_;
    uint64_t p_;
    uint32_t m_;
    MPFill fill_;
    std::vector<GKey> elems_;
    std::map<GKey, uint32_t> index_;
    // start_[k][(j - 1) * N_k + s] .. +1: the terms of tau_k(s_j e_s).
    std::vector<std::vector<uint64_t>> start_;
    std::vector<std::vector<Term>> terms_;
    double build_s_ = 0.0;

    uint32_t intern(const GKey& g);
    void store(uint32_t k, const StrChain& c);
    // tau_{k-1}(d (s_j e_s)) from the stored values, concrete.
    StrChain boundary_image(uint32_t k, uint32_t j, uint64_t s) const;
    // Dense rows of an operator: the coefficient matrix L(gamma) (d_out x (n+1)) of each term.
    SparseRows Assemble(uint32_t q, uint32_t n, uint32_t ell, uint32_t e, uint32_t d_out,
                        const std::function<Mat22(uint32_t j, const Mat22& gamma)>& left, bool coind_rows) const;
};

}  // namespace orbifold_mp

#endif
