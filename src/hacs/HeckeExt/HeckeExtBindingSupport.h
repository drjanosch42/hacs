// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_BINDING_SUPPORT_H
#define HECKE_EXT_BINDING_SUPPORT_H

// ===========================================================================
// hecke-extensions slot 9, WP-13 (console phase C5b) -- the CONSOLE-FREE
// builders behind the extension layer's C5 bindings.
//
// WHY A SEPARATE TU.  HeckeExt/RegisterHeckeExtensionVerbs.cpp is the ONE
// HeckeExt/ file admitted to include Console/ headers (check_heckeext_includes.sh
// rule 2's one exception, [OD] slot-3 close-out 1).  Everything a binding needs
// that is PIPELINE plumbing -- owning the coset / tau / operator / boundary /
// eta / cone-action stack at one prime, the production involution, the induced
// Dickson map L_*, the assembly of HeckeIdentityCertificate's inputs -- lives
// here instead, with no console type in sight, so the registration TU stays a
// thin translation layer and every object built here is testable without a
// session (HeckeExtAlgebraInvariantsValidation does).
//
// NOTHING HERE IS A NEW ROUTE.  Each builder composes public constructors in
// exactly the order an existing validation already does:
//   * ConeActionAtPrime       HeckeExtEndToEndValidation.cpp's FillItem7 /
//                             HeckeDualityValidation's arm A13 (cosets -> tau ->
//                             HeckeOperatorOnK; tau_bd -> eta -> T_p^bd ->
//                             HeckeConeAction);
//   * BuildProductionInvolution  HeckeExtDriver::build_sign_split_ (the
//                             QuadPoint-proposed, lem:iotamodel-certified
//                             iota_K, the exact relation data, compute());
//   * DicksonInducedMaps      HeckeExtEndToEndValidation.cpp's InducedDickson;
//   * AssembleIdentityCertificate  the same file's item 1-7 input assembly.
//
// PLACEMENT.  HeckeExt/, single-threaded per object, no hash container, no
// Console/ / Cuda/ / Engine/ include (guard rules 2 and 4).  Every returned
// object OWNS what it built; the pointers it holds into the caller's objects
// (a cohomology, a complex, the tessellation tables) must outlive it -- which
// the console discharges with keep-alive edges.
// ===========================================================================

#include "HeckeIdentityCertificate.h"
#include "HeckeSignSplit.h"

#include "../EquivariantCohomologyOnK.h"
#include "../EquivariantMappingCone.h"
#include "../SimplicialCohomology.h"
#include "../Hecke/HeckeBoundaryDatum.h"
#include "../Hecke/HeckeBoundaryOperator.h"
#include "../Hecke/HeckeConeAction.h"
#include "../Hecke/HeckeCosetTables.h"
#include "../Hecke/HeckeInvolution.h"
#include "../Hecke/HeckeOperatorOnK.h"
#include "../Hecke/HeckePairHomotopy.h"
#include "../Hecke/HeckeTauTable.h"
#include "../Hecke/SL2ZBoundaryPair.h"
#include "../Hecke/SL2ZTessellationTables.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace hecke_ext {

// ---------------------------------------------------------------------------
// T_p on the boundary circle, and on the cone, at one prime.
// ---------------------------------------------------------------------------

// T_p^bd on `pair`'s boundary cohomology: HeckeCosetTables(p, 1, ell, n) ->
// the certified Gamma-level tau_bd table -> HeckeBoundaryOperator::compute().
struct BoundaryActionAtPrime {
    uint64_t p = 0;
    std::unique_ptr<hecke::HeckeCosetTables>      cosets;
    std::unique_ptr<hecke::HeckeBoundaryTauTable> taub;
    std::unique_ptr<hecke::HeckeBoundaryOperator> bd;
};
std::unique_ptr<BoundaryActionAtPrime> BuildBoundaryActionAtPrime(
    const hecke::SL2ZBoundaryPair& pair, const hecke::SL2ZTessellationTables& model, uint64_t p);

// A_c(T_p) on H^*_c = Cone(r): the bulk operator, T_p^bd, the certified pair
// homotopy eta over Z and the cone action (whose h is DERIVED from eta).
struct ConeActionAtPrime {
    uint64_t p = 0;
    std::unique_ptr<hecke::HeckeCosetTables>      cosets;
    std::unique_ptr<hecke::HeckeTauTable>         tau;
    std::unique_ptr<hecke::HeckeOperatorOnK>      op;
    std::unique_ptr<hecke::HeckeBoundaryTauTable> taub;
    std::unique_ptr<hecke::HeckePairHomotopy>     eta;
    std::unique_ptr<hecke::HeckeBoundaryOperator> bd;
    std::unique_ptr<hecke::HeckeConeAction>       act;
};
std::unique_ptr<ConeActionAtPrime> BuildConeActionAtPrime(
    const hecke::SL2ZBoundaryPair& pair, const EquivariantMappingCone& cone,
    const hecke::SL2ZTessellationTables& model, uint64_t p);

// ---------------------------------------------------------------------------
// The production involution iota^* = -iota^*_geo ([C] def:iota(iv)).
// ---------------------------------------------------------------------------
std::unique_ptr<hecke::HeckeInvolution> BuildProductionInvolution(
    const EquivariantCohomologyOnK& coh, const SimplicialComplex& K,
    const hecke::SL2ZTessellationTables& model, uint32_t n_sym, uint32_t ell,
    const std::string& label);

// The sign split of degree q under `primes` (each != ell), the family's
// A^{(q)}(T_p) restricted to H^q_+ / H^q_-, the Eisenstein line
// (HeckeExt/HeckeSignSplit).  Null when dim H^q = 0.
std::unique_ptr<HeckeSignSplit> BuildSignSplit(
    const EquivariantCohomologyOnK& coh, const SimplicialComplex& K,
    const hecke::SL2ZTessellationTables& model, const hecke::HeckeInvolution& iota,
    uint32_t n_sym, uint32_t ell, uint32_t q, const std::vector<uint64_t>& primes);

// ---------------------------------------------------------------------------
// The Dickson map L_* : H^*(Ga; M_m) -> H^*(Ga; M_n), m = n - ell - 1.
// ---------------------------------------------------------------------------
// `target` is the cohomology of M_n and `source` of M_m, both on the SAME
// complex `K`.  Returns the induced map per degree (dim H^q(M_n) x dim H^q(M_m)).
std::vector<FlatMatrix> DicksonInducedMaps(const EquivariantCohomologyOnK& target,
                                           const EquivariantCohomologyOnK& source,
                                           const SimplicialComplex& K, uint32_t n, uint32_t ell);

// ---------------------------------------------------------------------------
// HeckeIdentityCertificate's inputs, assembled at one cell.
// ---------------------------------------------------------------------------
//
// Items 1-5 are assembled whenever their inputs exist at the cell; items 6
// (the residue atlas) and 7 (socle <-> Loewy under PD) are expensive and are
// assembled only when `want_item[6]` / `want_item[7]` is set.  `supplied[k]`
// says whether item k's inputs were handed to the certificate; an item whose
// inputs were not supplied is reported "unavailable" by the binding (with
// `why[k]`), never "passed".  Indices 1..kIdentityItems; index 0 unused.
inline constexpr uint32_t kIdentityItems = 7;   // == the certificate's verify() order
struct IdentityAssembly {
    IdentityCertificate cert;
    uint32_t code = 0;                       // cert.verify()
    std::vector<bool> supplied;              // [0 .. kIdentityItems]
    std::vector<std::string> why;            // [0 .. kIdentityItems], "" iff supplied
};
IdentityAssembly AssembleIdentityCertificate(const EquivariantCohomologyOnK& coh,
                                             const SimplicialComplex& K,
                                             const hecke::SL2ZTessellationTables& model,
                                             uint32_t n_sym, uint32_t ell, uint64_t p,
                                             uint64_t p_prime,
                                             const std::vector<bool>& want_item);

// The certificate's items in its verify() order: {name, item}.
std::vector<std::pair<const char*, const IdentityItem*>> IdentityItems(const IdentityCertificate& c);
const char* ToString(IdentityStatus s);

} // namespace hecke_ext

#endif // HECKE_EXT_BINDING_SUPPORT_H
