// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_BOUNDARY_OPERATOR_H
#define HECKE_BOUNDARY_OPERATOR_H

// hecke-extensions slot 1, WP-4b (phase E4b) -- the boundary Hecke operator
// Theta^bd_g and T^bd_p of docs/latex/gl2_contract.tex ([C], cited by label),
// chapter sec:boundarydatum, on the CIRCLE MODEL of the boundary:
//
//   * [C, prop:circlemodel]  C^*_bd = Hom_{Z[Gamma]}(C_*(X_bd), V) is, on the
//                            7-vertex model, C^*_eq(K_bd, Rel_bd; L) -- the
//                            circle model on the path v0-v1-v2-v3 with the
//                            single relation f(v3) = rho_n(T) f(v0)
//                            (LocalSystemSL2Z: CreateSL2ZBoundaryPathTriangulation
//                            / GenerateSL2ZBoundaryRelationsElt). (2): H^0 =
//                            V^T, H^1 = V_T, H^q = 0 for q >= 2, at EVERY ell;
//                            (3): it computes H^*(bd; V) = H^*(Gamma_inf; V)
//                            exactly when 2 in R^x.
//   * [C, def:bdryop],       (Theta^bd_g phi)(x) = sum_j rho_n(delta_j)
//     [C, eq:bdrychain]      phi(tau_bd(s_j . x)), x in C_q(X_bd) -- the bulk's
//                            [C, eq:heckechain] with tau_bd in place of tau,
//                            the SAME s_j, delta_j = s_j^{-1} g (HeckeCosetTables)
//                            and the certified tau_bd of WP-4a
//                            (HeckeBoundaryTauTable). NORMATIVE (owner decision
//                            OP1). T^bd_p is the induced map on
//                            H^*(C^*_bd), assembled per [C, alg:assembly]:
//                            column i of A_bd^{(q)} is P_q * flatten(Theta z_i),
//                            column convention T[z_i] = sum_j A_{ji} [z_j].
//   * [C, prop:bdryop]       (1) Theta^bd_g preserves C^*_bd and commutes with
//                            d, for every ell; (2) on H^0 = V^{Gamma_inf} it is
//                            the classical sum sum_{delta in B_N/Gamma_inf}
//                            rho_n(delta).
//   * [C, prop:bdrychar]     the boundary characters chi_b(p) = p^b + p^{n+1-b}
//                            on the basis x^{n-b ell} Nm^b of V^T (1), the
//                            right-coset det-twist chi_{b+1} (2), the n < ell
//                            Eisenstein value 1 + p^{n+1} on [y^n] and the
//                            H^1 group form [C, eq:bdrygroupform] (3).
//   * [C, obl:taubdry](v)    ROUTE A, the Shapiro transcription: the operator
//                            sum_{delta in B_N/Gamma_inf} rho_n(delta)(.) written
//                            in the coordinates of [C, prop:circlemodel](1),
//                            asserted to agree with [C, eq:bdrychain] on
//                            H^*(C^*_bd) as matrices in every computed degree
//                            (a validation ROUTE, never a redefinition).
//   * [C, obl:taubdry](vi)   the oracle support (invariant_monomial_basis, chi);
//     [C, obl:taubdry](vii)  is evaluated by HeckeBoundaryOperatorValidation
//                            against HeckeOperatorOnK through the restriction
//                            morphism EquivariantCochainComplexMorphismOnK.
//
// ===========================================================================
// THE ROUTES AND THEIR MANDATORY AGREEMENT (compute() asserts all of them and
// THROWS on any failure; the object then stays "not computed").
//
// NORMATIVE (level (a), Theta_flat(q)): for each boundary q-simplex slot
// sigma (a K_bd simplex; N_0 = 4, N_1 = 3) accumulate over j = 1..m and each
// stored ChainTerm (n_t, gamma_t, sigma_t) of tau_bd(s_j . i(sigma)) the
// d_V x d_V block  n_t * rho_n(delta_j gamma_t)  into block column sigma_t,
// in TWO bracketings that must be bit-identical (the bulk's O12(a)):
//   fused    rho_n(s_j^{-1}) * rho_n((g gamma_t) mod ell)   (stored S3 block
//            times a freshly reduced rho_n);
//   direct   rho_n((delta_j gamma_t) mod ell), the EXACT integer product
//            delta_j * gamma_t (overflow-checked Mat22, invariant I6) reduced
//            once (invariant I2). det-N matrices are NEVER inverted (I7).
// They agree entry for entry because delta_j gamma_t = s_j^{-1} (g gamma_t)
// over Z and rho_n is multiplicative on the mod-ell monoid
// ([C, prop:symconv](2)); routes_bit_identical records the memcmp.
//
// ROUTE A -- the Shapiro transcription (Theta_shapiro_flat(q), A_shapiro(q);
// nu = 1 only, where [C, prop:bdrycosets](1),(3) give the double-coset
// structure B_p = Gamma_inf delta_inf Gamma_inf  u  Gamma_inf delta_0 Gamma_inf).
// A Gamma_inf-LEVEL twisted chain datum on the infinity-line X_bd,inf, built
// from the line coordinates of SL2ZTessellationTables alone -- no tau_bd, no
// s_j, no (P1) class map, no delta_j -- with the CANONICAL left-coset
// representatives delta_inf = diag(1, p), delta_t = [[p, t],[0, 1]]:
//   D_inf: Gamma'' = Gamma_inf, one transporter s = I, delta_inf^{-1} T
//          delta_inf = T^p; tau_inf is delta_inf-twisted, tau_inf(T x) =
//          T^p tau_inf(x): vertex at position P |-> the line vertex nearest
//          to p * Re(v_P) (dilation, delta_inf^{-1} z = p z), edges the walk
//          between the endpoint images;
//   D_0:   Gamma'' = <T^p> x <-I>, transporters s_t = T^{-t} (t = 0..p-1),
//          delta_t = s_t^{-1} delta_0 = T^t delta_0 (asserted equal to the
//          WP-4a table), delta_0^{-1} T^p delta_0 = T; tau_0 is
//          delta_0-twisted, tau_0(T^p x) = T tau_0(x): vertex at P |-> the
//          line vertex nearest to Re(v_P) / p (compression), edges the walk.
//   (T_A phi)(x) = rho_n(delta_inf) phi(tau_inf(x)) + sum_t rho_n(delta_t)
//                  phi(tau_0(T^{-t} x)),   phi evaluated on the line through
//   Gamma_inf-equivariance phi(T^k . slot) = rho_n(T^k) phi(slot).
// The twisted equivariances are asserted exactly (ENGINE_CHECK) on a window of
// positions. Route A is compared with the normative operator ON COHOMOLOGY
// (A_shapiro(q) == A(q) entrywise in every degree), which is what (v) states;
// at cochain level the two are only homotopic ([C, lem:taubdryexists](3)).
//
// ROUTE B -- the closed forms (A_groupform(q)): on H^0 = V^T the sum
// M_0 = sum_{delta in B_N/Gamma_inf} rho_n(delta) over WP-4a's canonical
// representatives (every nu; [C, prop:bdryop](2)); on H^1 = V_T, at nu = 1,
// [C, eq:bdrygroupform]  M_1 = rho_n(delta_inf) (sum_{a<p} rho_n(T)^a) +
// rho_n(delta_0). Both are transported to the stored H_eq basis through the
// coordinates of [C, prop:circlemodel](1)/(2) -- class_representative /
// lift_class below -- and the projector P_q, and asserted equal to A(q).
//
// The coordinates ([C, prop:circlemodel](2), degree by degree):
//   H^0 -> V^T : a 0-cocycle in C^0_eq is constant on the path, phi = (m, m,
//                m, rho_n(T) m) with m in V^T; class_representative = m,
//                lift_class(m) = (m, m, m, rho_n(T) m).
//   H^1 -> V_T : phi |-> phi([v0, T v0]) = the signed sum of phi over the
//                three edges of the fundamental path (the group 1-cocycle
//                f(T) = phi([v0, T v0]) of the proof of [C, prop:bdrychar](3));
//                lift_class(m) = m on the edge slot e01 (times its
//                orientation sign), 0 elsewhere, so that class_representative
//                (lift_class(m)) = m.
// matrix_on_H_from_fiber_operator(q, M) is the general transport [m] |-> [M m]
// into the stored basis; it also reports whether the map DESCENDS to classes
// (lifts are cocycles in C_eq and coboundaries go to 0) -- the negative
// control of HeckeBoundaryOperatorValidation relies on that flag.
// ===========================================================================
//
// THE MODEL IS RECORDED, NOT REFUSED (owner confirmation 11 of the slot-1
// plan, 2026-09-16; [C, rmk:ell2window](1), OP5(i)): the object is built for
// EVERY ell, ell in {2, 3} included, and carries
//   boundary_model()                 = "circle"
//   boundary_is_group_cohomology()   = (2 in R^x) = (ell != 2)
// i.e. whether H^*(C^*_bd) is H^*(bd; V) = H^*(Gamma_inf; V)
// ([C, prop:circlemodel](3)) or, at ell = 2, the bounded orbifold-model
// boundary analogue of [C, rmk:ell2window](1) -- the circle model's dim H^q
// are the prop:circlemodel(2) values dim V^T / dim V_T / 0, never the
// [C, lem:ell2unbounded] values. [C, open:ell2] (OC-1a, riders (i)-(iii)):
// the circle model is finite, so q_max() is DERIVED as the top degree of the
// complex (boundary_dim = 1), truncated() = false, and every degree
// <= q_max is reported; nothing is truncated.
//
// FLAT COCHAIN LAYOUT (binding, the same as HeckeOperatorOnK /
// EquivariantCochainComplexMorphismOnK): a q-cochain on K_bd is the column of
// length N_q * d_V with flat index s*d_V + k = value at K_bd simplex s, fiber
// coordinate k, in the basis u_0 = x^n, ..., u_n = y^n; the K_bd simplex order
// is SL2ZTessellationTables::boundary_simplices(q)'s order (asserted at
// construction by vertex-list comparison).
//
// Placement: like HeckeOperatorOnK, compiled into cohomology_app (consumes
// ModArithmetic and the cohomology layer), linking HeckeCore for Mat22. No
// Matrix.h include (MatrixT reaches this header through FlatMatrix.h, as in
// HeckeOperatorOnK.h); no raw mod_* call (ModCtx throughout).

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "ExactInt.h"
#include "HeckeBoundaryDatum.h"
#include "HeckeCosetTables.h"
#include "RhoMemo.h"
#include "SL2ZTessellationTables.h"
#include "../EquivariantCohomologyOnK.h"
#include "../../ModArithmetic/FlatMatrix.h"

#include "../../ModArithmetic/DenseBuf.h"
#include "../../ModArithmetic/ModCtx.h"

namespace hecke {

template<class Elt>
class HeckeBoundaryOperatorT {
public:
    // cohomology_of_bd: the COMPUTED circle model C^*_eq(K_bd, Rel_bd; L)
    // (compute_cohomology() called; checked), built on
    // CreateSL2ZBoundaryPathTriangulation() with
    // GenerateSL2ZBoundaryRelationsElt. cosets: the S3 tables for
    // g = diag(1, N); taub: the WP-4a certified tau_bd built against the SAME
    // cosets object (its model() supplies the line coordinates). All three
    // must outlive this object. The constructor cross-checks ell / d_V / N
    // consistency and that K_bd's simplex lists are K's boundary simplices in
    // boundary_simplices(q) order; throws std::runtime_error /
    // std::invalid_argument on any mismatch.
    HeckeBoundaryOperatorT(const EquivariantCohomologyOnKT<Elt>* cohomology_of_bd,
                           const HeckeCosetTablesT<Elt>* cosets,
                           const HeckeBoundaryTauTableT<Elt>* taub);

    // Builds every matrix below and runs the assertions of the header note:
    // the two-bracketing bit-identity, (i) Theta z_i satisfies every
    // relation, (ii) d(Theta z_i) = 0, (iii) Theta C_eq c_j in C^q_eq for
    // every C_eq basis vector (relation closure -- [C, prop:bdryop](1)),
    // (iv) d Theta = Theta d on the C^0_eq basis, route A == normative on
    // cohomology (nu = 1), route B == normative on cohomology. THROWS
    // std::runtime_error on any failure. Idempotent.
    void compute();

    // --- shape ---------------------------------------------------------------
    uint32_t num_degrees() const { return num_degrees_; } // boundary_dim + 1 = 2
    Elt ell() const { return ell_; }
    uint32_t n_sym() const { return n_sym_; }
    uint32_t d_V() const { return d_V_; }
    uint64_t N() const { return cosets_->N(); }
    uint64_t p_hecke() const { return cosets_->p_hecke(); }
    uint32_t nu() const { return cosets_->r(); }
    uint32_t num_cosets() const { return m_; }
    uint32_t N_q(uint32_t q) const;             // number of K_bd q-simplices
    const EquivariantCohomologyOnKT<Elt>& cohomology() const { return *coh_; }
    const HeckeCosetTablesT<Elt>& cosets() const { return *cosets_; }
    const HeckeBoundaryTauTableT<Elt>& taub() const { return *taub_; }
    const SL2ZTessellationTables& model() const { return taub_->model(); }

    // --- record fields (see the header note) --------------------------------
    const char* boundary_model() const { return "circle"; }
    bool boundary_is_group_cohomology() const { return ell_ != 2; } // 2 in R^x
    uint32_t q_max() const { return num_degrees_ - 1; }          // derived: top degree
    const char* q_max_source() const { return "derived"; }
    bool truncated() const { return false; }
    std::vector<uint32_t> reported_degrees() const;              // {0, ..., q_max}

    // --- level (a): Theta^bd_g on the flat cochains C^q(K_bd; V) -----------
    const FlatMatrixT<Elt>& Theta_flat(uint32_t q) const;   // (N_q d_V)^2, requires compute()
    // Matrix-free direct evaluation of [C, eq:bdrychain] on one flat cochain
    // (rho_n((delta_j gamma_t) mod ell) from exact products); usable before
    // compute(), reads no stored Theta.
    FlatMatrixT<Elt> apply_direct(uint32_t q, const FlatMatrixT<Elt>& c_flat) const;

    // (slot-5 WP-6) The memo-taking form.  `memo` caches the PURE function
    // (j, gamma) |-> rho_n((delta_j gamma) mod ell) of THIS operator -- see
    // Hecke/RhoMemo.h for the dependency argument -- so a caller applying the
    // direct route to many operands pays for each distinct block once instead
    // of once per call.  The arithmetic is term for term the same whether the
    // table arrives empty or warm, and the one-argument overload above keeps
    // creating a fresh local table, so every external caller sees exactly the
    // pre-package behaviour.
    using RhoMemo = ::hecke::RhoMemoT<FlatMatrixT<Elt>>;
    FlatMatrixT<Elt> apply_direct(uint32_t q, const FlatMatrixT<Elt>& c_flat, RhoMemo& memo) const;
    // The identity a RhoMemo of THIS operator is stamped with (Hecke/RhoMemo.h).
    typename RhoMemo::Identity rho_memo_identity() const;
    FlatMatrixT<Elt> apply_Theta(uint32_t q, const FlatMatrixT<Elt>& c_flat) const; // Theta_flat(q) * c

    // --- level (b): A_bd^{(q)} = T^bd_p on H^q(C^*_bd) in the stored basis --
    const MatrixT<Elt>& A(uint32_t q) const;
    const std::vector<MatrixT<Elt>>& get_H_matrices() const;
    uint32_t dim_H(uint32_t q) const;

    // --- route A (the Shapiro transcription; nu = 1) ------------------------
    bool shapiro_available() const { return cosets_->r() == 1; }
    const FlatMatrixT<Elt>& Theta_shapiro_flat(uint32_t q) const; // the Gamma_inf-level cochain operator
    const MatrixT<Elt>& A_shapiro(uint32_t q) const;
    // The line datum route A is built from (public so that a validation can
    // re-derive it): the image POSITION of the line vertex at position P under
    // tau_inf (which = 0) and tau_0 (which = 1).
    int64_t shapiro_vertex_image(uint32_t which, int64_t P) const;

    // --- route B (the closed forms) -----------------------------------------
    bool groupform_available(uint32_t q) const; // q = 0: every nu; q = 1: nu = 1
    const MatrixT<Elt>& A_groupform(uint32_t q) const;
    // The fiber operators behind route B: M_0 = sum_{delta} rho_n(delta)
    // (d_V x d_V, every nu) and M_1 = rho_n(delta_inf)(sum_{a<p} rho_n(T)^a) +
    // rho_n(delta_0) (nu = 1; throws otherwise).
    FlatMatrixT<Elt> groupform_fiber_operator(uint32_t q) const;

    // --- the coordinates of [C, prop:circlemodel](1)/(2) -------------------
    // class_representative: the d_V x 1 representative m of the class of a
    // flat q-cochain (q = 0: its value at v0; q = 1: its signed sum over the
    // fundamental path); lift_class: the flat q-cochain of the header note.
    FlatMatrixT<Elt> class_representative(uint32_t q, const FlatMatrixT<Elt>& c_flat) const;
    FlatMatrixT<Elt> lift_class(uint32_t q, const FlatMatrixT<Elt>& m) const;
    // The matrix of [m] |-> [M m] on H^q in the stored basis (column i =
    // P_q lift(M class_rep(z_i))). descends_out = true iff every lift is a
    // cocycle in C^q_eq AND the B^q_eq basis is sent to 0, i.e. the recipe is
    // well defined on classes. Requires the cohomology only (not compute()).
    MatrixT<Elt> matrix_on_H_from_fiber_operator(uint32_t q, const FlatMatrixT<Elt>& M, bool& descends_out) const;
    // The class coordinates P_q lift_class(q, m) (dim H^q x 1).
    FlatMatrixT<Elt> class_coordinates(uint32_t q, const FlatMatrixT<Elt>& m) const;

    // --- oracle support ([C, prop:bdrychar](1), [C, obl:taubdry](vi)) --------
    // The basis x^{n - b ell} Nm^b, b = 0..floor(n/ell), of V^T (Nm = y^ell -
    // x^{ell-1} y, [C, prop:bdrychar](1)) as d_V x 1 columns in the fiber
    // basis u_k = x^{n-k} y^k; and chi_b(p) = p^b + p^{n+1-b} mod ell.
    static std::vector<FlatMatrixT<Elt>> invariant_monomial_basis(uint32_t n_sym, Elt ell);
    static Elt chi(Elt ell, uint64_t p, uint32_t n_sym, uint32_t b);

    // --- the NEGATIVE CONTROL ([C, prop:bdrycosets](4), [C, prop:bdrychar](2)) --
    // The RIGHT-coset representatives beta_b = [[1, b],[0, p]] (0 <= b < p)
    // and beta_inf = diag(p, 1) of [C, prop:cosets](1) substituted NAIVELY for
    // the left-coset ones: sum_b rho_n(beta_b) + rho_n(beta_inf) (the H^0
    // formula) and rho_n(beta_inf)(sum_{a<p} rho_n(T)^a) + rho_n(beta_0) (the
    // H^1 group form). NEVER used by the pipeline; exposed so that
    // HeckeBoundaryOperatorValidation can show it gives chi_{b+1} in place of
    // chi_b and breaks [C, obl:taubdry](vii). nu = 1 only.
    FlatMatrixT<Elt> beta_control_fiber_operator(uint32_t q) const;

    // --- (slot-9 WP-4; E8 item 4b; [E] v1.10.0-pre.1 sec:ring (b1), (f)) Z/ell^k ---
    // The operator's ring is its cohomology object's; it is honoured when that
    // object is over the Z/ell^k IN FORCE (ell >= 5; the default SmallEllPolicy
    // refuses 2 and 3 by name), and every other combination keeps slot 8's
    // refusal.  Over Z/ell^k compute() runs the MODULE route: Theta^bd over
    // Z/ell^k, dense, built BOTH ways (fused rho(s_j^{-1}) rho(g gamma) and
    // direct rho(delta_j gamma), the composite rho_n of the EXACT coset / tau_bd
    // data, asserted equal), runtime assertions (i)-(iv) exactly as the field
    // route (the closure (iii) by solve_multi_rhs_module with the residual
    // recomputed), and T_p^bd on the cyclic summands of H^q(bd) ([E] (c5)).
    // Routes A and B are field notions and are not run there; every FIELD
    // accessor (Theta_flat, A, get_H_matrices, apply_*, the route matrices,
    // to_json, to_string) throws a named std::logic_error.
    uint32_t coefficient_ring_k() const { return ring_k_; }
    bool is_composite() const { return ring_k_ >= 2; }
    const engine::DenseBuf<Elt>& Theta_module(uint32_t q) const;     // (N_q d_V)^2 mod ell^k
    const engine::DenseBuf<Elt>& H_matrix_module(uint32_t q) const;  // t_q x t_q, (c5)

    // --- certificate bundle ----------------------------------------------------
    struct CertificateBundle {
        bool routes_bit_identical = false;  // fused vs direct bracketing, all entries, all degrees
        uint64_t relation_checks = 0;       // (i): #(z_i, relation) pairs
        uint64_t cocycle_checks = 0;        // (ii): #z_i with d(Theta z_i) = 0
        uint64_t relation_closure_checks = 0; // (iii): #(c_j, relation) pairs on the C_eq basis
        uint64_t chain_map_checks = 0;      // (iv): #C^0_eq basis vectors with d Theta c = Theta d c
        bool shapiro_agrees = false;        // route A == normative on H^* (false if nu > 1: not available)
        bool shapiro_available = false;
        bool groupform_agrees = false;      // route B == normative on H^* (every available degree)
    };
    CertificateBundle certificates() const;

    std::string to_string() const { return to_string_offset(0); }
    std::string to_string_offset(uint32_t offset) const;
    // Hand-rolled JSON (schema "hecke_boundary_operator/v1"): the record
    // fields, per degree N_q / dim_C_eq / dim_H / Theta_flat / A (rows,
    // canonical residues, column convention as the bulk), the route matrices
    // where available, and the certificate bundle.
    std::string to_json() const;

private:
    const EquivariantCohomologyOnKT<Elt>* coh_;
    const HeckeCosetTablesT<Elt>* cosets_;
    const HeckeBoundaryTauTableT<Elt>* taub_;

    Elt ell_ = 0;
    uint32_t d_V_ = 0;
    uint32_t n_sym_ = 0;
    uint32_t m_ = 0;
    uint32_t num_degrees_ = 0;
    bool computed_ = false;

    // K_bd simplex index <-> K boundary simplex index (both q-indexed).
    std::vector<std::vector<uint32_t>> bd_to_K_;          // [q][s] = K index
    std::vector<std::map<uint32_t, uint32_t>> K_to_bd_;   // [q]: K index -> s

    std::vector<FlatMatrixT<Elt>> Theta_flat_;   // level (a)
    std::vector<MatrixT<Elt>> A_;                // level (b)
    std::vector<FlatMatrixT<Elt>> Theta_shapiro_;
    std::vector<MatrixT<Elt>> A_shapiro_;
    std::vector<MatrixT<Elt>> A_groupform_;
    std::vector<char> groupform_available_;

    CertificateBundle cert_;

    // (slot-9 WP-4) The coefficient ring and the module route's results.
    uint32_t ring_k_ = 1;
    engine::ModCtx<Elt> ring_ctx_;
    std::vector<engine::DenseBuf<Elt>> Theta_module_, A_module_;
    void compute_module();
    void require_field_op(const char* what) const;

    // rho_n of an exact integer matrix reduced mod ell (invariant I2).
    FlatMatrixT<Elt> rho_of_exact(const Mat22& gamma_exact) const;
    FlatMatrixT<Elt> rho_T_pow(int64_t k) const; // rho_n(T^k), k in Z
    void check_degree(uint32_t q) const;
    void require_computed() const;
    uint32_t local_index(uint32_t q, uint32_t K_sigma) const;

    // Assembly of the normative operator in one bracketing.
    FlatMatrixT<Elt> assemble_theta(uint32_t q, bool fused) const;
    // Column i = P_q * y_i from the flat images of the H_eq basis.
    MatrixT<Elt> assemble_A_from_flat_images(uint32_t q, const std::vector<FlatMatrixT<Elt>>& y_flat) const;
    // Runtime assertions (i)-(iv) on a cochain operator; throws with `what`.
    void certify_cochain_operator(uint32_t q, const FlatMatrixT<Elt>& Theta, const char* what,
                                  uint64_t& relation_checks, uint64_t& cocycle_checks,
                                  uint64_t& closure_checks, uint64_t& chain_map_checks) const;
    // The K_bd coboundary d_q (x) I in flat coordinates: (N_{q+1} d_V) x (N_q d_V).
    FlatMatrixT<Elt> flat_coboundary(uint32_t q) const;
    FlatMatrixT<Elt> flatten(const FlatMatrixT<Elt>& cochain_rows) const; // N_q x d_V -> (N_q d_V) x 1

    // Route A.
    struct LineChainTerm { int64_t coeff; int64_t position; }; // vertex or edge at a line position
    std::vector<LineChainTerm> line_walk(int64_t from, int64_t to) const;
    FlatMatrixT<Elt> assemble_theta_shapiro(uint32_t q) const;
    void certify_line_datum() const;
};

using HeckeBoundaryOperator = HeckeBoundaryOperatorT<uint32_t>;
using HeckeBoundaryOperator64 = HeckeBoundaryOperatorT<uint64_t>;

} // namespace hecke

#endif // HECKE_BOUNDARY_OPERATOR_H
