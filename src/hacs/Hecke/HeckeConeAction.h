// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_CONE_ACTION_H
#define HECKE_CONE_ACTION_H

// hecke-extensions slot 3, WP-6 (phase E5-3) -- the HECKE ACTION ON THE
// MAPPING CONE: the derived cochain homotopy h = eta~, the cone cochain map
// Phi_g, the operator A_c^{(q)} on H^q_c, and ConeActionCertificate.
// Binding mathematics: docs/latex/gl2_contract.tex ([C], cited by label)
// chapter sec:cone --
//
//   * [C, lem:conesign]      for ANY degree-(-1) map h and any cochain maps
//                            Th, Th^bd, with Phi(c, c_bd) := (Th c, Th^bd c_bd + h c),
//
//                              (Dcone Phi - Phi Dcone)(c, c_bd)
//                                  = (0, (d h + h d + r Th - Th^bd r) c),
//
//                            so Phi is a cochain map IFF d h + h d = Th^bd r - r Th.
//                            The two intertwinings Phi iota = iota Th^bd and
//                            pi Phi = Th pi hold for EVERY h.  (err:e13 is the
//                            opposite orientation; this unit computes the
//                            residual bracket and asserts it is 0.)
//   * [C, def:coneaction],   h := eta~, the COSET TRANSPORT of the pair
//     [C, eq:derivedh]       homotopy eta ([C, def:pairhomotopy], slot-2 WP-8's
//                            Hecke/HeckePairHomotopy) and never chosen:
//
//                              (h phi)(y) = sum_{j=1}^m rho_n(delta_j) phi(eta(s_j . y)),
//                                           y in C_{q-1}(X_bd),
//
//                            which on the stored table eta(s_j . i(sigma)) =
//                            sum_t n_t gamma_t . sigma_t and a Gamma-equivariant
//                            phi is  sum_j sum_t n_t rho_n(delta_j gamma_t) phi(sigma_t)
//                            -- the SAME shape as [C, eq:heckechain] /
//                            [C, eq:bdrychain], with eta's table in place of
//                            tau / tau_bd and the target one degree lower.
//   * [C, eq:conemap]        Phi_g(c, c_bd) := (Th_g c, Th^bd_g c_bd + h c).
//   * [C, lem:derivedhomotopy]  d h + h d = Th^bd_g r - r Th_g, no hypothesis on ell.
//   * [C, thm:conewelldef](a),(b),(d),(e)  Phi_g is a cochain map in every
//                            degree; every map of [C, eq:coneles] intertwines
//                            (T^bd, T^c, T); T^c is independent of eta over Z
//                            for every ell, and (for 6 in R^x) of tau, tau_bd,
//                            s_j and g ([C, prop:coneindep]).
//   * [C, alg:assembly] on Cone(r):  column i of A_c^{(q)} is
//                            P_c(q) . Phi^q(z_i), z_i the stored H^q_c basis,
//                            the bulk's column convention T[z_i] = sum_j A_{ji} [z_j].
//   * [C, obl:eta](v),(vi) and [C, obl:coneexact](i)-(iii),(vi):
//                            ConeActionCertificate / ConeRelationsCertificate below.
//
// DEGREES AND SHAPES (the cone's H_c-indexing, EquivariantMappingCone):
//   Cc^q = C^q(K; V) (+) C^{q-1}(K_bd; V),   flat_dim(q) = bulk_part_dim(q) + boundary_part_dim(q),
//   Phi^q = [[ Theta^q ,        0            ],
//            [ h^q     ,  Theta^bd_{q-1}     ]]      (block lower-triangular),
//   h^q : C^q(K; V) -> C^{q-1}(K_bd; V), built from eta in BOUNDARY degree q-1.
// On the shipped model h is non-zero in cone degrees 1 and 2 only and
// h^0 = h^3 = 0, because X_bd is one-dimensional ((B2)) -- but nothing below
// is written for q in {0, 1, 2}: the loop runs over every degree the cone
// reports and the boundary degree q-1 is read from eta's / Theta^bd's own
// degree fields (degree-generic, [X] Sec. 1.2).
//
// TWO BRACKETINGS (the bulk's O12(a), the boundary operator's verbatim).  Every
// block of h is formed in two ways that must be bit-identical:
//   fused   rho_n(s_j^{-1}) * rho_n((g gamma_t) mod ell)   (the stored S3 block
//           times a freshly reduced rho_n);
//   direct  rho_n((delta_j gamma_t) mod ell), the EXACT integer product
//           delta_j * gamma_t (overflow-checked Mat22, invariant I6) reduced
//           once (invariant I2).  det-N matrices are NEVER inverted (I7).
// They agree because delta_j gamma_t = s_j^{-1} (g gamma_t) over Z and rho_n is
// multiplicative on the mod-ell monoid ([C, prop:symconv](2)); the memcmp is
// recorded in routes_bit_identical and asserted by compute().
//
// WHY C^*_eq AND NOT THE FULL FLAT SPACE.  [C, eq:cosettransport] is defined
// on Hom_{Z[Gamma]}(-, V), and the step that turns phi(gamma_t . sigma_t) into
// rho_n(gamma_t) phi(sigma_t) -- the step that makes Theta_g, Theta^bd_g and h
// finite matrices at all -- IS the Gamma-equivariance of phi.  The stored
// matrices are therefore the transports composed with that identification:
// they are honest linear maps of the flat cochain spaces, they agree with the
// matrix-free routes on EVERY flat vector, and the pipeline's own certificates
// say the same thing about Theta_g (HeckeOperatorOnK runtime assertion (iii),
// HeckeBoundaryOperator (iii)/(iv): "d Theta = Theta d ON THE C_eq BASIS").
// [C, lem:derivedhomotopy] and [C, lem:conesign] are statements about the
// complexes Ceq(K, Rel; L) and C^*_bd of [C, def:pair], and this unit asserts
// them there.  The full flat identity happens to hold in cone degree 0 and
// fails in degree 1 on the shipped model (measured), which is why it is a
// RECORDED diagnostic (phi_flat_identity) and not a clause.
//
// A SECOND, MATRIX-FREE ROUTE.  apply_Phi_direct(q, x) evaluates Phi^q on one
// flat cone cochain without reading Phi_flat: the bulk part through
// HeckeOperatorOnK::apply_route2 ([C, eq:heckechain] from the exact tau-terms),
// the boundary part through HeckeBoundaryOperator::apply_direct
// ([C, eq:bdrychain]) plus an own matrix-free evaluation of [C, eq:derivedh].
// compute() asserts apply_Phi(q, e_b) == Phi_flat[:, b] == apply_Phi_direct(q, e_b)
// on the whole standard basis of every degree (the HeckeThetaBlockValidation
// pattern), so the stored matrix and the two independent evaluations agree.
//
// WHAT compute() ASSERTS (any failure throws std::runtime_error and the object
// stays "not computed"):
//   * the two bracketings of h, entrywise;
//   * the RESIDUAL of [C, eq:conesign], (d h + h d + r Th - Th^bd r), computed
//     as a matrix in every degree and asserted identically zero ON THE
//     EQUIVARIANT SUB-COMPLEX -- which is [C, lem:derivedhomotopy] re-derived
//     on the emitted data and is the exact form of [C, obl:coneexact](i);
//   * Dcone Phi = Phi Dcone on the equivariant sub-complex; the FULL FLAT
//     identity is recorded in phi_flat_identity and is NOT asserted (see the
//     note below);
//   * the matrix-free routes above;
//   * ConeActionCertificate::verify() on the freshly built data.
//
// PLACEMENT (slot-3 plan Sec. 2.3 item 4 with its [audit] note): Hecke/, NOT
// HeckeExt/ -- the cone at n = 256 is pipeline-sized, HeckeOperatorOnK /
// EquivariantMappingCone reach the block model and (through the pipeline) the
// tape, which rule 2 of check_heckeext_includes.sh forbids in HeckeExt/, and
// nothing in the pipeline may depend on the extension layer (rule 1).
// uint32_t only, as EquivariantMappingCone and the restriction morphism are.
// No Matrix.h include (MatrixT reaches this header through FlatMatrix.h, as in
// HeckeOperatorOnK.h); no raw mod_* call (ModCtx); no unordered_ container;
// ENGINE_CHECK / ENGINE_DEBUG_CHECK, never a bare assert.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "HeckeBoundaryOperator.h"
#include "HeckeCosetTables.h"
#include "HeckeOperatorOnK.h"
#include "HeckePairHomotopy.h"
#include "RhoMemo.h"
#include "../EquivariantMappingCone.h"
#include "../../ModArithmetic/FlatMatrix.h"

namespace hecke {

// ===========================================================================
// ConeActionCertificate -- [C, obl:eta](v),(vi) and [C, obl:coneexact](i)-(iii).
// A plain copyable struct holding the stored matrices; verify() RE-DERIVES
// every item from them (item 1 additionally from the eta-table, which is what
// the clause asks for: "the stored h equals eq:derivedh recomputed from the
// eta-table") and never re-runs a search.  The table of [C, sec:coneobligations]
// says what this certificate pins that ConeCertificate does not: the MODULE,
// not the sequence -- a tampered h' = h + xi with xi a degree-(-1) cochain map
// passes every clause of obl:coneexact and fails item 1
// ([C, prop:coneambiguity], [C, obl:coneexact](viii)(b)).
//
// Items (failing_items() returns these numbers):
//   1  (v)      h^q == eq:derivedh recomputed from the eta-table, entrywise,
//               in every computed degree.
//   2  (i)      d h + h d == Theta^bd r - r Theta on the equivariant
//               sub-complex C^q_eq(K, Rel; L) (the contract's complex, see the
//               note above); equivalently the residual bracket of
//               [C, eq:conesign] kills the stored C_eq basis.  The FULL FLAT
//               identity is recorded in phi_flat_identity and is never an item.
//   3  (i)      Dcone^2 = 0 in every built degree pair.
//   4  (i)      Dcone Phi Ceq == Phi Dcone Ceq.
//   5  (ii)     Dcone Phi z == 0 on every stored cocycle column of Z^q_c.
//   6  (ii)     A_c^{(q)} == P_c Phi H_c recomputed ([C, alg:assembly] step 2).
//   7  (iii)    pi^q A_c^{(q)} == A^{(q)} pi^q.
//   8  (iii)    r^q A^{(q)} == A_bd^{(q)} r^q.  ASSERTED at EVERY ell since
//               slot-5 WP-4 (OI-3): clause (i) is unconditional in ell and
//               proves this identity on cohomology with no hypothesis on ell,
//               so the former six_invertible carve-out -- which mirrored
//               [C, obl:taubdry](vii), an obligation with no h/eta and so no
//               proof at ell in {2, 3} -- is gone.  [C] v1.7.0 asserts it at
//               every ell too, so code and contract agree.
//   9  (iii)    iota^q A_bd^{(q)} == A_c^{(q+1)} iota^q.
//  10  structural: Phi^q is exactly [[Theta^q, 0], [h^q, Theta^bd_{q-1}]] and
//               every stored shape matches the cone's flat layout.
//  11  (obl:eta(vi))  eta_zero => cochain_commutes (a contradiction in the
//               stored data if it fails); nothing else about the two is
//               asserted and no code path branches on them (rule A6).
//  12  RECORD HONESTY ([C] rule A6; [OD] 2026-09-22 Q2).  The emitted
//               `restriction_equivariance_asserted` must report the assertion
//               item 8 ACTUALLY performs.  Item 8 carries no ell guard, so a
//               certificate claiming the r-half was NOT asserted misreports
//               its own run and is refused here.  The check is unconditional:
//               no shape gate stands in front of it, so it fires on any
//               certificate, however little else it carries.
// ===========================================================================
struct ConeActionCertificate {
    // --- parameters -------------------------------------------------------
    uint32_t ell = 0, d_V = 0, n_sym = 0;
    uint64_t p_hecke = 0, N = 0;
    uint32_t nu = 0, m = 0;
    uint32_t q_max = 0;                 // of H^*_c (the cone's)
    uint32_t eta_num_degrees = 0;       // boundary degrees eta is built in
    bool six_invertible = false;        // 6 in R^x: ell >= 5
    // [C, obl:coneexact](iii)'s r-half, item 8: asserted at EVERY ell since
    // slot-5 WP-4, so this is `true` on every certificate compute() fills.  It
    // is what the record emits -- NOT `six_invertible`, which reported a
    // carve-out the code had already dropped ([OD] 2026-09-22 Q2) -- and
    // item 12 refuses a certificate that says otherwise.  `six_invertible`
    // stays correct where it is still read: ConeRelationsCertificate's R1/R2.
    bool restriction_equivariance_asserted = false;
    bool ring_is_field = true;

    // --- the stored matrices, per cone degree q = 0 .. q_max --------------
    std::vector<uint32_t> flat_dim, bulk_part_dim, boundary_part_dim;
    std::vector<FlatMatrix> Theta;      // bulk_part_dim(q) x bulk_part_dim(q)
    std::vector<FlatMatrix> Theta_bd;   // boundary_part_dim(q)^2 (boundary degree q-1)
    std::vector<FlatMatrix> h;          // boundary_part_dim(q) x bulk_part_dim(q)
    std::vector<FlatMatrix> Phi;        // flat_dim(q)^2
    std::vector<FlatMatrix> D;          // flat_dim(q+1) x flat_dim(q), from the cone
    std::vector<FlatMatrix> Ceq;        // flat_dim(q) x dim Cc^q_eq, from the cone
    std::vector<FlatMatrix> Z, H, P;    // the cone's stored bases and projector
    std::vector<FlatMatrix> r_flat;     // N_q(K_bd) d_V x N_q(K) d_V, per bulk degree
    std::vector<FlatMatrix> d_bulk;     // d_q (x) I of K   (bulk_part_dim(q+1) x bulk_part_dim(q))
    std::vector<FlatMatrix> d_bd;       // d_q (x) I of K_bd (boundary degrees)
    // The BULK's stored C^q_eq basis (flat columns), the sub-complex on which
    // [C, lem:derivedhomotopy] and [C, lem:conesign] are statements.
    std::vector<FlatMatrix> Ceq_bulk;   // bulk_part_dim(q) x dim C^q_eq(K)

    // --- level (b) --------------------------------------------------------
    std::vector<MatrixT<uint32_t>> A_c, A_bulk, A_bd;
    std::vector<FlatMatrix> pi, r, iota;
    std::vector<char> pi_built, r_built, iota_built;

    // --- the eta-table item 1 re-derives from (non-owning) ----------------
    const HeckePairHomotopyT<uint32_t>* eta_table = nullptr;
    const HeckeCosetTablesT<uint32_t>* cosets = nullptr;
    // [qb] -> K index of the qb-th boundary simplex, in boundary_simplices order.
    std::vector<std::vector<uint32_t>> boundary_simplices;

    // --- recorded diagnostics ([C, obl:eta](vi); rule A6: they gate nothing) ---
    bool eta_zero = false;
    bool cochain_commutes = false;       // r Theta == Theta^bd r entrywise
    bool phi_flat_identity = false;      // Dcone Phi == Phi Dcone as FULL flat matrices (recorded, expected false)
    bool residual_flat_zero = false;     // the eq:conesign bracket vanishes on the FULL flat space (recorded)
    bool routes_bit_identical = false;   // the two bracketings of h
    bool restriction_equivariance = false;  // item 8's predicate, whether asserted or not
    uint64_t eta_table_fnv64 = 0, tau_table_fnv64 = 0, tau_boundary_table_fnv64 = 0;
    std::string solve_regime_eta = "Z", solve_regime_tau = "Z", solve_regime_tau_boundary = "Z";
    std::string eta_policy, tau_boundary_policy;

    // 0 iff every item holds; a misshaped item fails, never throws.
    uint32_t verify() const;
    uint32_t verify(std::vector<std::string>& failures) const;
    std::vector<uint32_t> failing_items() const;
};

// ===========================================================================
// ConeRelationsCertificate -- [C, obl:coneexact](vi), the theorems re-checked
// on H^*_c.  It needs SEVERAL operators, so it is its own object: given the
// A_c^{(q)}(T_p) of a family of primes and, for one of them, the PRIMITIVE
// A_c^{(q)}(T_{Delta'_{p^2}}) of g = diag(1, p^2), verify() re-derives
//   R1  [A_c^{(q)}(T_p), A_c^{(q)}(T_{p'})] = 0 for every pair;
//   R2  the nu = 2 recursion of [C, thm:relations](1) with the FULL T_{p^2},
//         A_c(T_{p^2}) = A_c(T_p)^2 - p^{n+1} I,   T_{p^2} = T_{Delta'_{p^2}} + p^n I
//       ([C, sec:primfull]; err:e1 / err:e6 are exactly the two slips this
//       spelling rules out -- the primitive operator of g = diag(1, p^2)
//       differs from the full one by the central p^n I, and the recursion's
//       coefficient is p^{n+1}, not (p+1) p^n).
// Both are ASSERTED at 6 in R^x and RECORDED at ell in {2, 3}
// ([C, rmk:conetorsion]; the clause says so itself).
// ===========================================================================
struct ConeRelationsCertificate {
    uint32_t ell = 0, n_sym = 0, q_max = 0;
    bool six_invertible = false;
    std::vector<uint64_t> primes;                       // p_1 < p_2 < ...
    // A_c[i][q] = A_c^{(q)}(T_{p_i}).
    std::vector<std::vector<MatrixT<uint32_t>>> A_c;
    // The nu = 2 arm, present iff recursion_present.
    bool recursion_present = false;
    uint64_t recursion_prime = 0;
    std::vector<MatrixT<uint32_t>> A_c_primitive_p2;               // A_c^{(q)}(T_{Delta'_{p^2}})
    std::vector<MatrixT<uint32_t>> A_c_p;                          // A_c^{(q)}(T_p) for recursion_prime

    // Recorded outcomes (set by verify(); meaningful after a call).
    mutable uint32_t commutator_pairs = 0, commutator_failures = 0;
    mutable uint32_t recursion_degrees = 0, recursion_failures = 0;

    uint32_t verify() const;
    uint32_t verify(std::vector<std::string>& failures) const;
    std::vector<uint32_t> failing_items() const;        // {1} = R1, {2} = R2
};

// ===========================================================================
// (hecke-extensions slot 9, WP-4; E8 item 4b; [E] v1.10.0-pre.1 sec:ring (f))
// ModuleConeActionT -- the cone action over Z/ell^k, ROUTE 1 ONLY (slot-9
// decision 3 (iii) (A)): the strict-pair and relative cross-routes keep
// refusing a composite normative action, and obl:coneexact(vii)'s composite
// half is open.  Given the dense module cone (ModuleConeT), the bulk operator
// and the boundary operator over the SAME Z/ell^k (their module routes) and
// the certified eta-table (integral, ring-free), compute() builds
//   h^q    [C, eq:derivedh] over Z/ell^k in BOTH bracketings (fused
//          rho(s_j^{-1}) rho(g gamma), direct rho(delta_j gamma) of the exact
//          product -- the composite rho_n, E8-P5), asserted equal;
//   Phi^q  [[Theta^q, 0], [h^q, Theta^bd_{q-1}]] on the flat cone cochains;
//   A_c^q  on the cone's cyclic summands ([E] (c5));
// and asserts, exactly, the ring reading of the ConeActionCertificate items
// (numbers kept): 2 the residual d h + h d + r Theta - Theta^bd r kills the
// bulk C^q_eq basis (obl:coneexact(i)); 4 D_c Phi = Phi D_c on Cc^q_eq; 5
// D_c Phi z = 0 on the cone cocycles; 6 A_c = the class coordinates of
// Phi b_i; 7 pi A_c = A pi; 8 r A = A_bd r (every ell, as over F_ell); 9
// iota A_bd = A_c iota -- each (c5) identity compared row by row in
// [0, ell^{e_i}) of its target (obl:coneexact(ii)-(iii)).  Any failure throws
// std::runtime_error.  Templated so that the U64_Wide cell runs it too.
// ===========================================================================
template<class Elt>
class ModuleConeActionT {
public:
    using Buf = engine::DenseBuf<Elt>;
    struct Record {
        bool routes_bit_identical = false;       // the two bracketings of h
        bool eta_zero = false;                   // every eta slot empty (recorded; gates nothing)
        uint64_t residual_checks = 0, chain_map_checks = 0, cocycle_checks = 0;
        uint64_t intertwining_checks = 0;        // items 7-9, entries compared
    };
    ModuleConeActionT(const ModuleConeT<Elt>* cone, const HeckeOperatorOnKT<Elt>* bulk_op,
                      const HeckeBoundaryOperatorT<Elt>* bd_op, const HeckePairHomotopyT<Elt>* eta);
    void compute();
    bool computed() const { return computed_; }
    const ModuleConeT<Elt>& cone() const { return *cone_; }
    const Buf& h(uint32_t q) const;              // boundary_part_dim(q) x bulk_part_dim(q)
    const Buf& Phi(uint32_t q) const;            // flat_dim(q)^2
    const Buf& A_c(uint32_t q) const;            // t_q x t_q, (c5)
    const Record& record() const { return rec_; }

private:
    const ModuleConeT<Elt>* cone_;
    const HeckeOperatorOnKT<Elt>* bulk_;
    const HeckeBoundaryOperatorT<Elt>* bd_;
    const HeckePairHomotopyT<Elt>* eta_;
    engine::ModCtx<Elt> ctx_;
    bool computed_ = false;
    std::vector<Buf> h_, Phi_, A_;
    Record rec_;
    void require_computed(const char* who) const;
};

// ===========================================================================
// HeckeConeAction -- the object.
// ===========================================================================
class HeckeConeAction {
public:
    // cone:    a COMPUTED EquivariantMappingCone of the pair (its bulk
    //          cohomology must be the one bulk_op was built on, its boundary
    //          cohomology the one bd_op was built on; checked).
    // bulk_op: a COMPUTED HeckeOperatorOnK for g = diag(1, N).
    // bd_op:   a COMPUTED HeckeBoundaryOperator for the SAME coset tables.
    // eta:     the certified eta-table for the same (tau, tau_bd) -- its
    //          tau_bd must be bd_op's (checked by pointer identity of the
    //          coset tables and by the tau_bd table hash) AND its tau must be
    //          bulk_op's (checked by pointer identity -- the eta-PROVENANCE
    //          check, [X] v1.10 open item 9 / hecke-extensions slot 7 WP-11:
    //          a certified table for SOME tau, i.e. a cochain with "the right
    //          boundary" (obl:eta(i)-(iv)), is refused here unless it is also
    //          eta~ relative to THIS bulk_op).
    // All four must outlive this object.  Throws std::invalid_argument /
    // std::runtime_error on any inconsistency.
    HeckeConeAction(const EquivariantMappingCone* cone, const HeckeOperatorOnKT<uint32_t>* bulk_op,
                    const HeckeBoundaryOperatorT<uint32_t>* bd_op, const HeckePairHomotopyT<uint32_t>* eta);
    ~HeckeConeAction();
    HeckeConeAction(const HeckeConeAction&) = delete;
    HeckeConeAction& operator=(const HeckeConeAction&) = delete;

    // Builds h, Phi, A_c and the certificate, and runs every assertion of the
    // header note.  Throws std::runtime_error on any failure.  Idempotent.
    void compute();
    bool computed() const { return computed_; }

    // --- shape / record fields -------------------------------------------
    uint32_t ell() const { return ell_; }
    uint32_t d_V() const { return d_V_; }
    uint32_t n_sym() const { return n_sym_; }
    uint64_t p_hecke() const;
    uint32_t nu() const;
    uint64_t N() const;
    uint32_t q_max() const { return q_max_; }             // of H^*_c
    uint32_t num_degrees() const { return q_max_ + 1; }
    const EquivariantMappingCone& cone() const { return *cone_; }
    const HeckeOperatorOnKT<uint32_t>& bulk_operator() const { return *bulk_; }
    const HeckeBoundaryOperatorT<uint32_t>& boundary_operator() const { return *bd_; }
    const HeckePairHomotopyT<uint32_t>& eta() const { return *eta_; }

    // --- level (a) --------------------------------------------------------
    // h^q : C^q(K; V) -> C^{q-1}(K_bd; V), boundary_part_dim(q) x bulk_part_dim(q).
    const FlatMatrix& h_flat(uint32_t q) const;
    // Phi^q on the flat cone cochains Cc^q, flat_dim(q) x flat_dim(q).
    const FlatMatrix& Phi_flat(uint32_t q) const;
    // The residual (d h + h d + r Theta - Theta^bd r)^q of [C, eq:conesign] as
    // a matrix on the full flat C^q(K; V); after compute() it annihilates the
    // bulk C^q_eq basis (the contract's complex), which is what compute()
    // asserts.  Kept so a validation can print it and re-check it itself.
    const FlatMatrix& residual(uint32_t q) const;
    // The same restricted to the stored C^q_eq basis: identically zero after
    // compute().  This is [C, obl:coneexact](i)'s cone identity.
    FlatMatrix residual_on_Ceq(uint32_t q) const;
    // The bulk's stored C^q_eq basis in the flat layout (columns).
    FlatMatrix Ceq_bulk(uint32_t q) const;
    // The memo table of the rho_n blocks a matrix-free evaluation needs, keyed
    // by (j, gamma) exactly as the per-call table of assemble_h / apply_h_direct
    // was (slot-4 WP-4).  It is a memo of the pure function
    // gamma |-> rho_n(delta_j gamma mod ell), so sharing one table across the
    // calls of a sweep is value-preserving by construction; an empty table
    // reproduces the pre-slot-4 per-call behaviour exactly.
    // (slot-5 WP-6) The type is now the shared ::hecke::RhoMemoT of
    // Hecke/RhoMemo.h, which adds the O(1) identity stamp; the key, the value
    // and the consuming arithmetic are unchanged.
    using RhoMemo = ::hecke::RhoMemoT<FlatMatrix>;

    // (slot-5 WP-6) apply_Phi_direct evaluates THREE matrix-free routines,
    // each over its OWN coset table -- the bulk's route 2, the boundary's
    // direct route and h -- so it carries three tables, never one shared one:
    // the (j, gamma) key determines a block only relative to a fixed coset
    // table, and hecke::RhoMemoT::bind() enforces exactly that.
    struct RhoMemos {
        RhoMemo h;                                       // eta's coset table
        HeckeOperatorOnKT<uint32_t>::RhoMemo bulk;       // tau's  (route 2)
        HeckeBoundaryOperatorT<uint32_t>::RhoMemo bd;    // tau_bd's
    };

    // Phi_flat(q) * x (one dense product) and the matrix-free second route.
    FlatMatrix apply_Phi(uint32_t q, const FlatMatrix& x) const;
    FlatMatrix apply_Phi_direct(uint32_t q, const FlatMatrix& x) const;
    FlatMatrix apply_Phi_direct(uint32_t q, const FlatMatrix& x, RhoMemos& memo) const;
    // h^q evaluated matrix-free on one flat bulk q-cochain ([C, eq:derivedh]).
    FlatMatrix apply_h_direct(uint32_t q, const FlatMatrix& c_flat) const;
    FlatMatrix apply_h_direct(uint32_t q, const FlatMatrix& c_flat, RhoMemo& memo) const;
    // The identity a RhoMemo of h (eta's coset table) is stamped with.
    RhoMemo::Identity rho_memo_identity() const;

    // --- level (b): A_c^{(q)} on H^q_c ([C, alg:assembly] on Cone(r)) -----
    const MatrixT<uint32_t>& A_c(uint32_t q) const;
    const std::vector<MatrixT<uint32_t>>& get_H_matrices() const;
    uint32_t dim_Hc(uint32_t q) const;

    // --- the certificate --------------------------------------------------
    const ConeActionCertificate& certificate() const;

    // --- golden collectors (DEFINED, NEVER EMITTED; slot-3 plan Sec. 4.4) --
    // tier A: char-poly / rank / trace of A_c^{(q)}; tier B: the raw matrix
    // A_c^{(q)} itself (stored-basis dependent -- [X] Sec. 1.6 / owner
    // decision OP7; labelled 'B' here since slot-6 WP-5, which is the tier
    // the golden harness has always published it under), h_flat,
    // Theta^bd_flat and the three table hashes.  Called from
    // HeckeConeActionValidation and (the ext subgrid) Golden/GoldenHarness.cpp.
    std::vector<ConeGoldenObject> golden_objects() const;
    uint64_t golden_digest() const;

    std::string to_string() const { return to_string_offset(0); }
    std::string to_string_offset(uint32_t offset) const;
    // Hand-rolled JSON, schema "hecke_cone_action/v1": the per-complex degree
    // fields (the cone's, the bulk's, the boundary's and -- for h, a cochain
    // map of degree -1 -- its SOURCE complex's, per [C, sec:recordschema]
    // item 5 as WP-5 finding 2 states it), the boundary labels, the three
    // table hashes, eta_zero / cochain_commutes / the solve regimes, per
    // degree the shapes and A_c, and the certificate's bookkeeping.
    std::string to_json() const;

    // --- (slot-9 WP-4; E8 item 4b) Z/ell^k, route 1 ----------------------
    // Honoured when the cone and both operators are over the Z/ell^k IN FORCE
    // (the dense module route, ModuleConeActionT above); every FIELD accessor
    // of this class then throws a named std::logic_error.
    uint32_t coefficient_ring_k() const { return ring_k_; }
    bool is_composite() const { return ring_k_ >= 2; }
    const ModuleConeActionT<uint32_t>& module_action() const;

    // The two policies this object was built against (for the record and for
    // [C, obl:coneexact](viii)(a)'s independence arm).
    EtaVertexPolicy eta_policy() const { return eta_->policy(); }
    TauBoundaryVertexPolicy tau_boundary_policy() const { return bd_->taub().policy(); }

private:
    const EquivariantMappingCone* cone_;
    const HeckeOperatorOnKT<uint32_t>* bulk_;
    const HeckeBoundaryOperatorT<uint32_t>* bd_;
    const HeckePairHomotopyT<uint32_t>* eta_;

    uint32_t ell_ = 0, d_V_ = 0, n_sym_ = 0, m_ = 0, q_max_ = 0;
    bool computed_ = false;

    std::vector<FlatMatrix> h_;          // [q]
    std::vector<FlatMatrix> Phi_;        // [q]
    std::vector<FlatMatrix> resid_;      // [q]
    std::vector<MatrixT<uint32_t>> A_;              // [q]
    ConeActionCertificate cert_;
    uint32_t ring_k_ = 1;                                         // (slot-9 WP-4)
    std::unique_ptr<ModuleConeActionT<uint32_t>> module_;         // (slot-9 WP-4) the Z/ell^k result
    void require_field(const char* who) const;

    void check_degree(uint32_t q) const;
    void require_computed(const char* who) const;
    FlatMatrix rho_of_exact(const Mat22& gamma_exact) const;
    // The boundary degree feeding cone degree q (q - 1), and whether it exists.
    bool boundary_degree(uint32_t q, uint32_t& qb_out) const;
    // [C, eq:derivedh] assembled in one bracketing.
    FlatMatrix assemble_h(uint32_t q, bool fused) const;
    // d_q (x) I of a complex, in the flat layout.
    static FlatMatrix coboundary_tensor(const SimplicialComplex& C, uint32_t q, uint32_t dV, uint32_t ell);
};

// ===========================================================================
// HeckeConeStrictPairRoute -- hecke-extensions slot 6, WP-4 (E5 cross-route 1):
// the STRICT-PAIR route of [C, prop:coneroutes](1), the first sentence of
// [C, obl:coneexact](vii) (owner decision OP3: a committed validation arm,
// a CROSS-CHECK of [C, def:coneaction], never a definition -- the cone of
// [C, eq:conediff] with the shipped (tau, tau_bd, eta) stays normative).
//
// Given a COMPUTED normative HeckeConeAction, compute() builds, beside it and
// without touching it:
//   * tau^s   -- a second certified integer table, TauVertexPolicy::
//                BoundaryStrict ([C, prop:strictpair]): tau^s o j = j o tau_bd
//                for the normative tau_bd, its own TauCertificate run
//                (HeckeTauTable::certify() (a)-(g)) and its own fill search
//                (int64, then the widened tier; never mod ell, [OD] OP2);
//   * eta^s   -- HeckePairHomotopy on (tau^s, tau_bd), certified as always;
//                by strictness every defect (j tau_bd - tau^s j) vanishes, so
//                the certified table IS eta == 0 (asserted: eta_zero());
//   * Theta^s -- HeckeOperatorOnK on the same cohomology and coset tables;
//   * Phi^s   -- HeckeConeAction on the SAME cone, the SAME boundary operator
//                and eta^s (so h^s = 0 and Phi^s = (Theta^s, Theta^bd)), with
//                its ConeActionCertificate run by compute() as always;
// and compares A_c^{(q)}(Phi^s) with the normative A_c^{(q)} ENTRYWISE in every
// computed degree q = 0 .. q_max (both are in the cone's one stored H^q_c
// basis, so equality of the operator is equality of the matrices).
//
// Abort vs record ([C, obl:coneexact](vii), [C, prop:coneroutes](1)): the
// agreement is a theorem only for 6 in R^x (Proposition coneindep(2) needs the
// uniqueness half of lem:tauexists).  So:
//   * ell >= 5: a disagreement in any degree, or a failure to build or certify
//     any of the four objects, THROWS std::runtime_error -- a disagreement
//     would contradict the independence theorem, a build failure the existence
//     claim of prop:strictpair (a stop and a finding, never relaxed);
//   * ell in {2, 3}: both are RECORDED (record().disagreeing_degrees,
//     record().build_failure) and compute() returns normally.
// The record is a plain struct; nothing here is emitted into any golden
// object, report or campaign record.
//
// OVER Z/ell^k (hecke-extensions slot 10, WP-3; slot-10 decision 3 (A);
// [C, obl:coneexact](vii) over the ring).  Honoured when the normative action
// is over the Z/ell^k IN FORCE (HeckeConeAction::is_composite(), route 1 =
// ModuleConeActionT); every other combination keeps slot 8's named refusal.
// tau^s and eta^s are integral (ring-free) exactly as over F_ell; Theta^s is
// HeckeOperatorOnK's module route on the SAME composite bulk cohomology (the
// reduction mod ell^k of the integral tables); Phi^s is HeckeConeAction on the
// SAME composite cone and boundary operator, i.e. route 1 on (tau^s, tau_bd,
// eta^s == 0).  The comparison is A_c^{(q)} of the two ModuleConeActionT's,
// entrywise, in the cone's ONE stored cyclic-summand basis (both are the
// (c5) matrices of [E] sec:ring, rows canonical in [0, ell^{e_i})).  Only
// ell >= 5 reaches this path (the cone refuses ell in {2, 3}), so every
// failure THROWS.  cochain_commutes is re-derived over the ring on the bulk
// C^q_eq basis (r Theta^s C == Theta^bd r C).  THE REGIME (slot-10 plan
// M-3): the class is uint32_t-only, so it honours ell^k < 2^32 and refuses
// above it by the ring context's named refusal (RequireRingRegime); only
// route 1 (the templated ModuleConeActionT) runs the U64 cells.
// ===========================================================================
class HeckeConeStrictPairRoute {
public:
    struct Record {
        uint32_t ell = 0, n_sym = 0, nu = 0, q_max = 0;
        uint64_t p_hecke = 0;
        bool six_invertible = false;
        bool asserted = false;              // == six_invertible: the agreement is ASSERTED
        bool built = false;                 // tau^s, eta^s, Theta^s, Phi^s all built and certified
        std::string build_failure;          // the exception text when !built (recorded at ell in {2,3})
        // The strict table's record fields ([C, obl:eta](iv) vocabulary).
        std::string solve_regime;           // "Z" (never anything else for BoundaryStrict)
        uint32_t fills_int64 = 0, fills_widened = 0, strict_boundary_slots = 0;
        bool eta_zero = false;              // eta^s == 0 (asserted when built)
        uint32_t eta_nonzero_slots = 0;
        uint64_t tau_strict_fnv64 = 0, tau_normative_fnv64 = 0;
        bool tau_differs = false;           // the strict table differs from the shipped one (non-vacuity)
        bool cochain_commutes = false;      // r Theta^s == Theta^bd r entrywise (recorded)
        uint32_t ring_k = 1;                // (slot-10 WP-3) 1 = F_ell; k = the normative action's Z/ell^k
        bool external_table = false;        // (slot-10 WP-3) tau^s supplied by the validation seam (re-certified)
        // The comparison, per degree 0 .. q_max.
        uint32_t degrees_compared = 0;
        std::vector<uint32_t> disagreeing_degrees;
        bool agrees() const { return built && disagreeing_degrees.empty(); }
    };

    // normative: a COMPUTED HeckeConeAction; it (and everything it points to)
    // must outlive this object.  Throws std::invalid_argument otherwise.
    explicit HeckeConeStrictPairRoute(const HeckeConeAction* normative);
    // (slot-10 WP-3) VALIDATION SEAM: compare against the given strict table
    // instead of building one (non-owning; it must outlive this object).  The
    // table must be a BoundaryStrict table on the normative coset tables and
    // tau_bd; compute() RE-RUNS its certify() (a table mutated after its
    // construction is refused like any other build failure).  Used by the
    // negative control of HeckeConeCrossRouteRingValidation only.
    HeckeConeStrictPairRoute(const HeckeConeAction* normative, const HeckeTauTableT<uint32_t>* strict_table);
    ~HeckeConeStrictPairRoute();
    HeckeConeStrictPairRoute(const HeckeConeStrictPairRoute&) = delete;
    HeckeConeStrictPairRoute& operator=(const HeckeConeStrictPairRoute&) = delete;

    // Builds the strict datum and compares (see the class note).  Idempotent.
    void compute();
    bool computed() const { return computed_; }
    const Record& record() const { return rec_; }

    // (slot-10 WP-3) The uint32_t regime rule: refuses (hecke::NotImplementedInThisBuild,
    // naming the route) unless Z/ell^k fits the uint32_t instantiation of the ring context
    // (ell >= 5, ell^k < 2^32).  k == 1 is a no-op.
    static void RequireRingRegime(uint64_t ell, uint32_t k);

    // The strict-route objects (null when !record().built).
    const HeckeTauTableT<uint32_t>* tau_strict() const { return tau_s_ ? tau_s_.get() : tau_ext_; }
    const HeckePairHomotopyT<uint32_t>* eta_strict() const { return eta_s_.get(); }
    const HeckeConeAction* action_strict() const { return act_s_.get(); }
    const HeckeConeAction& normative() const { return *norm_; }

    std::string to_string() const;

private:
    const HeckeConeAction* norm_;
    bool computed_ = false;
    Record rec_;
    const HeckeTauTableT<uint32_t>* tau_ext_ = nullptr;   // (slot-10 WP-3) the validation seam
    void compute_composite();                              // (slot-10 WP-3) over Z/ell^k
    std::unique_ptr<HeckeTauTableT<uint32_t>> tau_s_;
    std::unique_ptr<HeckePairHomotopyT<uint32_t>> eta_s_;
    std::unique_ptr<HeckeOperatorOnKT<uint32_t>> bulk_s_;
    std::unique_ptr<HeckeConeAction> act_s_;
};

// ===========================================================================
// HeckeConeRelativeRoute -- hecke-extensions slot 6, WP-5 (E5 cross-route 2):
// the RELATIVE-COMPLEX route, "route 3", of [C, prop:coneroutes](2), the
// second sentence of [C, obl:coneexact](vii) (owner decision OP3: a committed
// validation arm and a CROSS-CHECK of [C, def:coneaction], never a definition
// -- the cone of [C, eq:conediff] stays normative).  No hypothesis on ell.
//
// Given a COMPUTED normative HeckeConeAction, compute() builds, beside it and
// from objects E5 already holds (its Theta, its h, the cone's r_flat and the
// bulk's local system), without touching any of them:
//   * ext_0   -- the extension by zero of [C, lem:conerelative](1),
//                C^q(K_bd; V) -> C^q(K; V), read off the stored flat
//                restriction r_flat (an identity-fibre block selection on the
//                shipped pair; anything else is refused): the section with
//                r ext_0 = I (asserted), mapping C^q_eq(K_bd) into C^q_eq(K)
//                (asserted through the bulk's own A_C rows);
//   * E       -- E^q := ext_0 o h^q : C^q(K; V) -> C^{q-1}(K; V);
//   * Theta'  -- Theta'^q := Theta^q + d E^q + E^{q+1} d  ([C, eq:route3]);
//   * ker r   -- the shipped EquivariantCohomologyOnK on the bulk model with
//                the boundary simplices' cochain values forced to zero (the
//                additive relative sub-complex constructor): its C/Z/B/H/P;
//   * A'^{(q)} = P_rel Theta'^q H_rel, the operator on H^q(ker r) in its
//                stored basis;
//   * M^q     -- the basis correspondence H^q(ker r) -> H^q_c of
//                [C, lem:conerelative](3), c |-> (c, 0), in the stored bases:
//                column i = P_c(q) (z_i, 0), z_i the i-th stored H^q(ker r)
//                cocycle;
//   * N^q     -- the inverse map H^q_c -> H^q(ker r), [(c, c_bd)] |-> [c +
//                d ext_0 c_bd] (R3.5 below asserts M N = N M = I on the
//                stored bases). WELL DEFINED ON CLASSES (slot-14 WP-14b,
//                review finding slot-10 m-11): for (c, c_bd) = D_c(x, x_bd) a
//                cone coboundary, put c' = d(ext_0 r x - x) + d(ext_0 d_bd
//                x_bd - d ext_0 x_bd); both bracketed terms lie in ker r
//                (R3.1's r ext_0 = I gives r(ext_0 r x - x) = 0, and applying
//                r to the second bracket uses r ext_0 = I and r d = d_bd r
//                again), so c' is a coboundary of ker r -- N sends a cone
//                coboundary input to the same class of c + d ext_0 c_bd,
//                hence is well defined on H^q_c. A (c5) product (R3.5, R3.6)
//                is only defined for matrices of homomorphisms, so this is
//                what licenses computing with N at all, not only its role as
//                M's inverse.
// and compares M A' M^{-1} with the normative A_c^{(q)} ENTRYWISE in every
// computed degree q = 0 .. q_max.
//
// EVERY failure THROWS std::runtime_error at EVERY ell (route 3 "aborts for
// every ell", [C, obl:coneexact](vii)): the disagreement itself, and each of
// the route's own certificate clauses, which are what makes the comparison
// mean something --
//   R3.1  r ext_0 = I, and ext_0 C^q_eq(K_bd) inside C^q_eq(K)  ([C, lem:conerelative](1));
//   R3.2  dim C^q_rel = dim C^q_eq(K) - dim C^q_eq(K_bd)  (r onto, (1) again);
//   R3.3  on C^q_eq(K): d Theta' = Theta' d (a cochain map), r Theta' = Theta^bd r
//         ON THE NOSE, and P_eq Theta' H_eq = A^{(q)} (the bulk operator is
//         unchanged: Theta' - Theta = dE + Ed is a homotopy)   ([C, prop:coneroutes](2));
//   R3.4  Theta' H_rel inside Z^q_rel  (Theta' preserves ker r; A_C_rel and d);
//   R3.5  (z_i, 0) is a cone cocycle, and M^q is square and invertible
//         ([C, lem:conerelative](3): the quasi-isomorphism);
//   R3.6  M A' M^{-1} == A_c^{(q)} entrywise.
// The record is a plain struct; nothing here is emitted into any golden
// object, report or campaign record.
//
// OVER Z/ell^k (hecke-extensions slot 10, WP-3; [C, obl:coneexact](vii) over
// the ring).  Honoured when the normative action is over the Z/ell^k IN FORCE
// (the rule of HeckeConeStrictPairRoute above, and its uint32_t regime rule).
// The same route on the dense module objects: ext_0 from ModuleConeT's flat
// restriction; E = ext_0 h with h = ModuleConeActionT::h; Theta' = Theta + dE
// + Ed with Theta = HeckeOperatorOnK::Theta_module; ker r = the relative
// sub-complex constructor over Z/ell^k (its MODULE route, ell >= 5: C^q_rel is
// free because r has the equivariant section ext_0); A' = the (c5) matrix of
// Theta' on H^q(ker r)'s cyclic summands.  The clauses, over the ring:
//   R3.1  r ext_0 = I; ext_0 C^q_eq(K_bd) inside C^q_eq(K) (a module solve);
//   R3.2  rank C^q_rel = rank C^q_eq(K) - rank C^q_eq(K_bd);
//   R3.3  d Theta' = Theta' d and r Theta' = Theta^bd r on the bulk C^q_eq
//         basis; Theta' induces the bulk's A^{(q)} (class coordinates);
//   R3.4  Theta' H_rel lies in the cocycles of ker r (a class-coordinate solve);
//   R3.5  M: z_i |-> [(z_i, 0)] and N: [(c, c_bd)] |-> [c + d ext_0 c_bd]
//         (the explicit inverse of lem:conerelative(3)'s quasi-isomorphism:
//         with the cone's D_c(c, c_bd) = (-d c, d_bd c_bd + r c),
//         (c, c_bd) - (c + d ext_0 c_bd, 0) = D_c(ext_0 c_bd, 0)) are (c5)
//         matrices with M N = I and N M = I, so no inverse is ever taken over
//         the ring, and the invariant factors of H^q(ker r) and H^q_c agree;
//   R3.6  M A' N == A_c^{(q)} entrywise, rows reduced in [0, ell^{e_i}).
// Every failure THROWS.  The FIELD accessors (ext0_flat, E_flat,
// Theta_prime_flat, A_rel, correspondence, A_c_transported) throw a named
// std::logic_error on a composite route; the *_module accessors are its twins.
// ===========================================================================
class HeckeConeRelativeRoute {
public:
    struct Record {
        uint32_t ell = 0, n_sym = 0, nu = 0, q_max = 0;
        uint64_t p_hecke = 0;
        uint32_t vanishing_simplices = 0;         // the forced-zero simplices of K, all degrees (7 on the model)
        std::vector<uint32_t> dim_C_rel, dim_H_rel;   // per degree 0 .. q_max
        bool E_nonzero = false;                   // NON-VACUITY: E != 0 in some degree (i.e. h != 0 reaches K)
        bool theta_prime_differs = false;         // NON-VACUITY: Theta' != Theta on C_eq in some degree
        bool r_commutes_normative = false;        // r Theta == Theta^bd r on C_eq (recorded; false on the shipped tau)
        uint32_t degrees_compared = 0;
        std::vector<uint32_t> disagreeing_degrees;   // always empty after compute() returned
        // (slot-10 WP-3) Over Z/ell^k: dim_C_rel holds rank C^q_rel (free), dim_H_rel the
        // number of cyclic summands of H^q(ker r), and these their exponents.
        uint32_t ring_k = 1;
        std::vector<std::vector<uint32_t>> H_rel_invariant_factors;
        uint64_t c5_checks = 0;                   // entries compared in R3.5 / R3.6
        bool agrees() const { return degrees_compared == q_max + 1 && disagreeing_degrees.empty(); }
    };

    // normative: a COMPUTED HeckeConeAction whose cone's bulk cohomology kept
    // its intermediates (A_C_block / C_eq_block are read); it (and everything
    // it points to) must outlive this object.  Throws std::invalid_argument otherwise.
    explicit HeckeConeRelativeRoute(const HeckeConeAction* normative);
    ~HeckeConeRelativeRoute();
    HeckeConeRelativeRoute(const HeckeConeRelativeRoute&) = delete;
    HeckeConeRelativeRoute& operator=(const HeckeConeRelativeRoute&) = delete;

    // Builds route 3 and compares (see the class note).  Idempotent.
    void compute();
    bool computed() const { return computed_; }
    const Record& record() const { return rec_; }
    const HeckeConeAction& normative() const { return *norm_; }

    // (slot-10 WP-3) The uint32_t regime rule, as HeckeConeStrictPairRoute::RequireRingRegime.
    static void RequireRingRegime(uint64_t ell, uint32_t k);

    // The route's objects (require compute(); q in 0 .. q_max).
    const EquivariantCohomologyOnK& relative_cohomology() const;   // ker r
    const FlatMatrix& ext0_flat(uint32_t q) const;       // N_q(K) d_V x N_q(K_bd) d_V (0-row/col where absent)
    const FlatMatrix& E_flat(uint32_t q) const;          // bulk_part_dim(q-1) x bulk_part_dim(q)
    const FlatMatrix& Theta_prime_flat(uint32_t q) const;
    const MatrixT<uint32_t>& A_rel(uint32_t q) const;    // A'^{(q)} on H^q(ker r)
    const FlatMatrix& correspondence(uint32_t q) const;  // M^q, dim H^q_c x dim H^q(ker r)
    const MatrixT<uint32_t>& A_c_transported(uint32_t q) const;   // M A' M^{-1}

    // (slot-10 WP-3) The Z/ell^k twins (require compute() on a composite route;
    // a logic_error on an F_ell one).  (c5) matrices, rows in [0, ell^{e_i}).
    bool is_composite() const { return rec_.ring_k >= 2; }
    const engine::DenseBuf<uint32_t>& Theta_prime_module(uint32_t q) const;
    const engine::DenseBuf<uint32_t>& A_rel_module(uint32_t q) const;             // A' on H^q(ker r)
    const engine::DenseBuf<uint32_t>& correspondence_module(uint32_t q) const;    // M: H^q(ker r) -> H^q_c
    const engine::DenseBuf<uint32_t>& inverse_correspondence_module(uint32_t q) const;   // N: H^q_c -> H^q(ker r)
    const engine::DenseBuf<uint32_t>& A_c_transported_module(uint32_t q) const;   // M A' N

    std::string to_string() const;

private:
    const HeckeConeAction* norm_;
    bool computed_ = false;
    Record rec_;
    std::unique_ptr<EquivariantCohomologyOnK> rel_;
    std::vector<FlatMatrix> ext0_, E_, Thp_, M_;
    std::vector<MatrixT<uint32_t>> Arel_, Atr_;
    std::vector<engine::DenseBuf<uint32_t>> Thp_m_, Arel_m_, M_m_, N_m_, Atr_m_;   // (slot-10 WP-3)
    void require_computed(const char* who, uint32_t q) const;
    void require_field_route(const char* who) const;      // (slot-10 WP-3)
    void require_module_route(const char* who) const;     // (slot-10 WP-3)
    void compute_composite();                             // (slot-10 WP-3)
};

} // namespace hecke

#endif // HECKE_CONE_ACTION_H
