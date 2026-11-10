// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_NEBENTYPUS_WILD_H
#define HECKE_NEBENTYPUS_WILD_H
// ===========================================================================
// (hecke-extensions slot 22, P10 (a)) THE WILD CHARACTER MODELS over O_{j,e} = Z_ell[zeta_{ell^j}] / pi^e.
// Spec: docs/implementation/slot22/P10_spec.md (v0.3) section 3 step 9, gates W1 - W4.  Text: [C] v1.22.0-pre.1
// def:nebwild, lem:nebwild, lem:nebwildtors, rmk:nebwildcheckable (NB7) - (NB9).
//
// Built for e = K phi_j (K in [1, 3]): O_{j,e} = (Z/ell^K)[z]/(Phi_{ell^j}(z)) on the power basis (def:nebwild "as data"),
// a free Z/ell^K-module of rank phi_j; every O-module of the model is read as a Z/ell^K-module (the coefficient
// z^i of class J is the column (J - 1) phi_j + i).  Lengths over O_{j,e} equal lengths over Z/ell^K (residue field F_ell
// for both, lem:nebwild (1)).  The cokernel is computed by the unit-pivot elimination of Hecke/HeckeGammaHTower with
// the torsion kept (at i = 0 the quotient has the torsion O/pi^c of lem:nebwildtors: there lem:nebchainrref does not
// apply) and a Howell form of the residual relations.  The F_q / W_K(F_q) route's refusal character_wild
// (Hecke/HeckeNebentypus ReduceCharacter) is untouched.
//
// Refusals (std::logic_error, message starts with the name): wild_precision (e not K phi_j with K in [1, 3], or
// ell^K >= 2^32), wild_ring_order (ell^j does not divide t ell^{r-1}: zeta^{t L(d)} is not well defined),
// wild_tame_level (N not a power of ell).
// ===========================================================================

#include <cstdint>
#include <memory>
#include <vector>

#include "HeckeGammaH.h"
#include "HeckeGammaHTower.h"

namespace hecke {
namespace level {
namespace wild {

using tower::ZK;
using OElem = std::vector<uint32_t>;   // phi_j coefficients of 1, z, ..., z^{phi_j - 1}, each mod ell^K

struct WildRing {
    uint32_t ell = 0, j = 1, phi = 1, e = 1, K = 1;
    ZK R;
    std::vector<uint32_t> Phi;         // Phi_{ell^j}, monic, low -> high (phi + 1 entries)
    OElem zero() const { return OElem(phi, 0); }
    OElem one() const;
    OElem scalar(uint32_t c) const;
    OElem zpow(uint64_t n) const;      // z^n reduced
    OElem add(const OElem& a, const OElem& b) const;
    OElem sub(const OElem& a, const OElem& b) const;
    OElem mul(const OElem& a, const OElem& b) const;
    uint32_t mod_pi(const OElem& a) const;   // a(1) mod ell: the residue in O/pi = F_ell
};
WildRing MakeWildRing(uint32_t ell, uint32_t j, uint32_t e);

// chi_{i,t}(d) = omega(d)^i zeta^{t L(d)} (def:nebwild), as a table of size N = ell^r (zero vector at the non-units).
struct WildCharacter {
    uint32_t r = 1, i = 0;
    uint64_t t = 1, N = 1;
    std::vector<OElem> table;
};
WildCharacter MakeWildCharacter(const WildRing& O, uint64_t N, uint32_t i, uint64_t t);

// The chi_{i,t}-model of Gamma_0(N) over O_{j,e}, expanded over Z/ell^K.
class WildModel {
public:
    WildModel(const WildRing& O, const WildCharacter& chi);
    const WildRing& ring() const { return O_; }
    uint32_t m() const { return G_->m(); }
    uint32_t D() const { return G_->m() * O_.phi; }
    const tower::ChainCokernel& cokernel() const { return *Q_; }
    uint32_t len_H1() const { return len_H1_; }
    uint32_t len_H0() const { return len_H0_; }
    uint32_t len_rel() const { return len_rel_; }
    // the minimal pi-valuation... the Howell pivot valuations of the torsion relations over Z/ell^K (empty: no torsion)
    const std::vector<uint32_t>& torsion_pivot_valuations() const { return tors_pval_; }
    // operators on the expanded module, as ChiOperators on columns 1 .. D
    std::unique_ptr<tower::ChiOperator> heilbronn(uint64_t n) const;   // T_n / U_n with the scalar chi(d_h)
    std::unique_ptr<tower::ChiOperator> scalar(const OElem& s) const;  // multiplication by s in O
    const std::vector<SparseVec>& generators() const { return gens_; }
private:
    WildRing O_;
    WildCharacter chi_;
    std::unique_ptr<GammaHLevelN> G_;
    std::vector<uint32_t> idt_;        // the identity table x -> x (so that ChiHeilbronn / MonomialChiOf report a_h, d_h)
    std::vector<SparseVec> gens_;
    std::unique_ptr<tower::ChainCokernel> Q_;
    uint32_t len_H1_ = 0, len_H0_ = 0, len_rel_ = 0;
    std::vector<uint32_t> tors_pval_;
};

// The length of the Z/ell^K-span of `gens` in (Z/ell^K)^D (unit pivots + the Howell form of the residuals).
uint32_t SpanLength(uint32_t D, const std::vector<SparseVec>& gens, const ZK& R, std::vector<uint32_t>* tors_pval = nullptr);

} // namespace wild
} // namespace level
} // namespace hecke

#endif // HECKE_NEBENTYPUS_WILD_H
