// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef EQUIVARIANT_MAPPING_CONE_H
#define EQUIVARIANT_MAPPING_CONE_H

// hecke-extensions slot 3, WP-5 (phase E5-2) -- the equivariant MAPPING CONE
// of a restriction morphism, its long exact sequence and ConeCertificate.
// Binding mathematics: docs/latex/gl2_contract.tex ([C], cited by label)
// chapter sec:cone -- def:pair (the pair, r = j^*, the cone eq:conediff in
// the monodromy_v5 Sec. 8 convention), lem:conerelative(2)-(5) (Dcone^2 = 0,
// the short exact sequence 0 -> C_bd -> Cone(r) -> C_eq[1] -> 0 with
// connecting map +r^{q+1}, the long exact sequence eq:coneles, H^0_c = 0,
// the six-term window and eq:conedims), open:ell2 (the truncation policy,
// riders (i)-(iii)), obl:coneexact(iv)-(v) (the certificate clauses of this
// package; (i)-(iii), (vi)-(viii) are the cone ACTION's, slot-3 WP-6).
// Plan: hecke_extensions_slot3_plan.md Sec. 3 WP-5 with its [audit] notes.
//
// WHAT IT IS.  For a restriction morphism r : C^*_eq(K, Rel; L) -> C^*_bd (an
// EquivariantCochainComplexMorphismOnK whose DOMAIN is the boundary complex
// -- the target of the induced map on cohomology, in the class's contravariant
// naming -- and whose CODOMAIN is the bulk), the cone complex
//
//     Cone(r)^q = C^{q+1}_eq(K) (+) C^q_bd,     D(c, c_bd) = (-d c, d_bd c_bd + r c)
//
// and its cohomology H^q_c := H^{q-1}(Cone(r)) ([C, def:pair]).  EVERYTHING
// HERE IS INDEXED BY THE DEGREE OF H^*_c: the stored cochain group in degree
// q is  Cc^q := Cone(r)^{q-1} = C^q(K; V) (+) C^{q-1}(K_bd; V),  so that
// H^q_c = H^q(Cc) and no re-indexing appears anywhere else.  The maps of the
// long exact sequence
//
//     ... -> H^q_c --pi_*--> H^q_eq(K) --r^q--> H^q_bd --iota_*--> H^{q+1}_c -> ...
//
// are pi_*[(c, c_bd)] = [c] (the projection; sign-free because C_eq[1]
// carries -d, [C, def:pair]), r^q the induced map of the morphism, and
// iota_*[c_bd] = [(0, c_bd)] -- the connecting homomorphism of the short
// exact sequence is +r^{q+1} ([C, lem:conerelative](2)), which the validation
// re-derives by the zig-zag.
//
// STORAGE (the flat layout of [C, alg:assembly] on Cone(r)).  A cone q-cochain
// is the column of length N_q(K) d_V + N_{q-1}(K_bd) d_V: first the bulk
// q-cochain in the pipeline's L1 layout (flat index s*d_V + k), then the
// boundary (q-1)-cochain in ITS L1 layout.  Every stored object is an
// engine::BlockMat over that row partition ((d_V)^{N_q(K)} ++ (d_V)^{N_{q-1}(K_bd)}):
//
//   D_block(q)      Cc^q -> Cc^{q+1}, Scalar refs only (-+1 from the two
//                   simplicial coboundaries, the fibre blocks of r interned by
//                   content -- I for the identity-fibre restriction), so the
//                   differential carries no dense data at any n;
//   Ceq_block(q)    the equivariant cone cochains C^q_eq(K) (+) C^{q-1}_eq(K_bd)
//                   as ONE block-diagonal multi-vector (nbc == 1);
//   Z_block(q)      Ceq . ker(D . Ceq)                        (blk::kernel_basis, D5b)
//   B_block(q)      the D5b image basis of D_{q-1} . Ceq_{q-1}  (the pipeline's
//                   own Phase-4 route for B^q_eq: the pivot columns of D.C)
//   H_block(q),     blk::factor_basis(B, Z): the chosen Z columns and the
//   P_block(q)      projector Cc^q_flat -> H^q_c coordinates (dim H^q_c x flat_dim(q))
//   pi(q), r(q),    the three map families in the stored bases (FlatMatrix,
//   iota(q)         dim(target) x dim(source), column convention as the bulk:
//                   map[z_i] = sum_j M[j][i] [z_j]), built for every degree in
//                   which BOTH end terms have a stored basis.
//
// DEGREES ([C, open:ell2], riders (i)-(iii); sec:recordschema item 5 per
// complex).  The two input complexes carry ComplexDegreeFields, DERIVED by
// default (q_max = the top degree of the finite complex, truncated = false,
// q_max_source = "derived", degrees = [0..q_max]); on a finite model a SET
// q_max with truncated = false is rejected (rider (i)).  A declaration
// {truncated = true, q_max < top} is the shape of E9's truncated free model
// (the complex is built to degree q_max + 1 and that degree is scaffolding);
// before E9 only a validation uses it, on a finite pair, as a test double.
// The cone's fields are derived from its inputs' and never set:
//   finite inputs      q_max_c = max(top_K, top_bd + 1), truncated = false
//   truncated input    q_max_c = min(q_max_K, q_max_bd + 1), truncated = true
// so that Cc^{q_max_c + 1} exists (the scaffolding degree) and H^{q_max_c+1}_c
// has no stored basis.  A term H^q(F) is REPORTED iff q < 0, or q <= q_max(F),
// or F is not truncated (an untruncated finite complex reports its vanishing
// above its top degree; the box forbids that for truncated = true only).  A
// spot of the sequence is ASSERTED EXACT iff all three of its terms are
// reported -- which reproduces open:ell2's spelled-out clause verbatim when
// every q_max agrees -- and its two composites are asserted zero under the
// same condition; at a spot whose target term is unreported the rank of the
// incoming map and the dimension of its cokernel are RECORDED (the latter a
// lower bound for the unreported group, rmk:ell2window(2)).  Degree-generic:
// no `q < 3` literal anywhere in this unit.
//
// ConeCertificate (obl:coneexact(iv)-(v) + the Dcone^2 = 0 half of (i)): a
// plain copyable struct holding the stored matrices; verify() RE-DERIVES
// every item from them alone (never by calling back into the pipeline):
//   C1  D_{q+1} D_q = 0 in every degree pair built;
//   C2  the equivariant cone cochains form a subcomplex (D Ceq_q inside span Ceq_{q+1});
//   C3  Z: dim Z_q = dim Ceq_q - rank(D Ceq_q) recomputed, D Z = 0, full column rank, Z inside span Ceq;
//   C4  B: B_{q+1} == D_q Ceq_q[:, pivots] entrywise, full column rank == rank(D Ceq_q);
//   C5  H, P: D H = 0, P H = I, P B = 0, H inside span Z, rank [B | H] = rank B + dim H,
//       dim H = dim Z - rank B;
//   C6  the three map families recomputed from the stored bases / projectors == the stored maps;
//   C7  composites zero at every asserted spot ((iv));
//   C8  ker = im at every asserted spot, over the field by the rank identity ((iv); a
//       non-field ring is recorded unchecked, never passed);
//   C9  im(pi_*^q) = ker r^q EXPLICITLY (both inclusions by solves) in every fully
//       reported degree ((iv)'s "checked explicitly", degree-generic);
//   C10 H^0_c = 0 unconditionally ((v), lem:conerelative(4));
//   C11 eq:conedims where H^2_eq = 0 holds in the stored data ((v));
//   C12 H^2_eq = 0 asserted at 6 in R^x, recorded otherwise ((v));
//   C13 the degree fields consistent with the stored data and with the rules above.
//
// (hecke-extensions slot 23, LES-2; [C] sec:nebles def:levelcone, lem:levelconeexact) UNEQUAL FIBRES.  On the F_ell
// route the boundary complex may carry a fibre of another dimension d_bd (d_V_boundary()): the boundary part of Cc^q is
// then partitioned into blocks of d_bd, and r^q has d_bd x d_V blocks.  This is the level cone Cone(alpha_{B/A}) of a
// level pair B <= A (bulk = L_A, d_V = m_A (n + 1); boundary = L_B, d_bd = m_B (n + 1); r = alpha_*), assembled by
// Hecke/HeckeLevelCone.  Audit (LES-2 step 0, docs/implementation/slot23/LES2_predictions.md section 0): no clause
// C1-C13 uses that r is onto or that the boundary is a circle; C10 (H^0_c = 0), C11 and C12 hold for the level cone by
// lem:levelconeexact; the equal-fibre convention was the only obstruction.  With d_bd == d_V every statement is the
// pre-LES-2 one (texts, JSON and golden_digest unchanged).  The Z/ell^k module route keeps the equal-fibre refusal.
//
// Placement: generic (no Hecke include), compiled into cohomology_app next to
// EquivariantCochainComplexMorphismOnK; uint32_t only, as the restriction
// morphism is.  FlatMatrix / BlockMat / engine::blk / engine::elim only -- no
// Matrix.h (the Phase-3 exit criterion), no raw mod_* call (ModCtx), no
// unordered_ container, ENGINE_CHECK for invariants.  The Hecke assembly of
// the SL(2,Z) pair is Hecke/SL2ZBoundaryPair; the cone ACTION Phi_g is WP-6's
// Hecke/HeckeConeAction, which consumes D_block / Ceq_block / Z_block /
// P_block and the flat layout above.

#include "EquivariantCochainComplexMorphismOnK.h"
#include "EquivariantCohomologyOnK.h"
#include "../ModArithmetic/BlockMat.h"
#include "../ModArithmetic/DenseBuf.h"
#include "../ModArithmetic/FlatMatrix.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// The per-complex record fields of [C, sec:recordschema] item 5 (slot-2 owner
// confirmation 1: per complex, never a maximum).
struct ComplexDegreeFields {
    uint32_t q_max = 0;
    bool truncated = false;
    std::string q_max_source = "derived";   // "derived" | "set"
    std::vector<uint32_t> degrees;          // the reported degrees, 0..q_max

    // Derived from a computed complex: q_max = its top degree (one less than
    // the number of degrees it reports), truncated = false.
    static ComplexDegreeFields derived_from(const EquivariantCohomologyOnK& coh);
    // The E9 shape: a complex built to degree q_max + 1 whose top degree is
    // scaffolding.  Legal only with q_max < the complex's top degree
    // (checked by the cone's constructor; open:ell2 rider (i)).
    static ComplexDegreeFields declared_truncation(uint32_t q_max);

    // A term of this complex in degree q is reported (see the header note).
    bool reported(int64_t q) const { return q < 0 || q <= static_cast<int64_t>(q_max) || !truncated; }
    std::string to_json() const;
};

// One golden record (the shape of HeckeExt/HeckeBoundary's, re-declared here
// because the generic layer may not include HeckeExt/): hashed EXACTLY as
// Golden/GoldenHarness.cpp hashes a scalar / a matrix (engine::FnvU64 over
// rows, cols, entries row-major).  NOTHING here writes GoldenReference.inc;
// the collectors are called from HeckeConeValidation only (slot-3 plan
// Sec. 4.4: "defined, never emitted" means "never called from the golden
// harness").
struct ConeGoldenObject {
    std::string object;
    char tier = 'A';
    uint64_t rows = 0, cols = 0;
    uint64_t hash = 0;
};

// The status of one spot of eq:coneles.
struct ConeLesSpot {
    std::string term;                 // "Hc" | "H_bulk" | "H_bd"
    int32_t q = 0;                    // the degree of the middle term
    bool in_built = false, out_built = false;   // the two adjacent maps exist
    bool asserted = false;            // all three terms reported: composite + exactness asserted
    uint32_t dim_middle = 0, rank_in = 0, rank_out = 0;
    bool composite_zero = false;      // meaningful iff in_built && out_built
    bool exact = false;               // meaningful iff asserted and the ring is a field
    uint32_t coker_in_dim = 0;        // dim_middle - rank_in: at an unasserted spot the recorded lower bound
};

struct ConeCertificate {
    using BlockMat = engine::BlockMat<uint32_t>;

    uint32_t ell = 0;
    uint32_t d_V = 0;
    bool ring_is_field = true;        // F_ell; a non-field ring records C8 / C9 unchecked
    bool six_invertible = false;      // 6 in R^x: ell >= 5
    ComplexDegreeFields bulk, boundary, cone;

    // Per cone degree q = 0 .. q_max_c + 1 (the last one is the scaffolding
    // cochain group: D, Ceq stored; no Z / B / H / P).
    std::vector<uint32_t> flat_dim;           // N_q(K) d_V + N_{q-1}(K_bd) d_V
    std::vector<uint32_t> bulk_part_dim;      // N_q(K) d_V
    std::vector<BlockMat> D;                  // [q] Cc^q -> Cc^{q+1}, q <= q_max_c
    std::vector<BlockMat> Ceq;                // [q] q <= q_max_c + 1
    std::vector<BlockMat> Z, B, H, P;         // [q] q <= q_max_c
    std::vector<std::vector<uint32_t>> B_pivots;   // [q] indices into Ceq_{q-1}'s columns (empty at q = 0)
    std::vector<uint32_t> dim_Hc;             // [q] q <= q_max_c

    // The inputs' stored cohomology bases and projectors (flat, L1) and the
    // flat restriction, per degree of the respective complex.
    std::vector<BlockMat> H_bulk, P_bulk;     // [q] q <= top_K
    std::vector<BlockMat> H_bd, P_bd;         // [q] q <= top_bd
    std::vector<BlockMat> r_flat;             // [q] N_q(K_bd) d_V x N_q(K) d_V, q <= min(top_K, top_bd)
    std::vector<uint32_t> dim_H_bulk, dim_H_bd;

    // The three map families (dim target x dim source); built[q] says whether
    // both end terms have a stored basis.
    std::vector<FlatMatrix> pi, r, iota;
    std::vector<char> pi_built, r_built, iota_built;

    std::vector<ConeLesSpot> spots;           // as recorded at construction

    // 0 iff every item C1..C13 holds; a misshaped item fails, never throws.
    uint32_t verify() const;
    uint32_t verify(std::vector<std::string>& failures) const;
    // The item numbers (1..13) that fail, for the tamper arm.
    std::vector<uint32_t> failing_items() const;
};

// ===========================================================================
// (hecke-extensions slot 9, WP-4; E8 item 4b; [E] v1.10.0-pre.1 sec:ring (f);
// slot-9 plan WP-4 scope 1-3) THE DENSE MODULE ROUTE OF THE CONE OVER Z/ell^k.
//
// Selected by the ring alone (is_field() false; ENGINE_API.md Sec. 13.6-13.7:
// there is no block or tape Howell form, so the route is dense).  The F_ell
// cone above -- the tape route -- is untouched.  The route inherits the ring
// context's element type and regime (u32 below 2^32, the uint64_t
// instantiation with U64_Montgomery / U64_Wide above it) and uses Sec. 13.6's
// exact entry points only; no new regime and no new bound.
//
// Inputs: the bulk and the boundary complex over the SAME Z/ell^k (computed by
// EquivariantCohomologyOnKT's module route, ell >= 5: both C^q_eq free), and
// the flat restriction r^q : C^q(K; V) -> C^q(K_bd; V) per degree q <=
// min(top_K, top_bd), as dense matrices mod ell^k.  Everything is indexed by
// the degree of H^*_c, exactly as the F_ell class: Cc^q = C^q(K) (+) C^{q-1}(K_bd),
// bulk rows first.  Per cone degree q:
//   C_q    the Howell basis of Cc^q_eq: the two Howell bases concatenated, bulk
//          first (asserted once per degree to BE the Howell form of the block
//          sum -- Sec. 13.5's uniqueness);
//   X_q    D_c^q (c, c_bd) = (-d c, d_bd c_bd + r c) in C coordinates, by
//          solve_multi_rhs_module ([E] (c1) with the cone's own differential);
//   Zc_q   kernel_module(X_q); H^q_c = Zc_q / colspan X_{q-1} by
//          cyclic_decomposition ([E] (c2)-(c4)); H_q the b_i as flat cochains.
// The eight maps of eq:coneles, in [C]'s order pi^0, r^0, iota^0, pi^1, r^1,
// iota^1, pi^2, r^2, are (c5) matrices between the cyclic-summand bases
// (H^q(bd) := 0 above the boundary's top degree, so r^2 lands in 0).
//
// THE CERTIFICATE (obl:coneexact(iv)-(v) over the ring; ConeModuleCertificate):
// the F_ell ConeCertificate's item numbers are kept where the item has a ring
// reading, and ONE item is appended (14, never renumbered):
//   1   X_{q+1} X_q = 0 in coordinates (D_c^2 = 0 on Cc_eq);
//   10  H^0_c = 0 (lem:conerelative(4)), as a length;
//   11  (v) in lengths: at every term of eq:coneles, length(term) =
//       length(im in) + length(im out) (the length identity, the sequence's
//       exactness in lengths), and the two eq:conedims identities --
//       l(H^1_c) = l(H^0(bd)) - l(H^0) + l(ker r^1) unconditionally, and
//       l(H^2_c) = l(H^1(bd)) - l(im r^1) where H^2 = 0;
//   12  H^2 = 0 asserted at ell >= 5 (cl:amalgam; always so here);
//   14  (iv)'s NON-FIELD HALF: at every spot, the composite of consecutive maps
//       is zero (each column of the incoming map's image lies in the relations
//       of the target of the outgoing one, BY A SOLVE), and each Howell
//       generator x of ker(out) is exhibited as x = in(w1) + D_M w2 by
//       solve_multi_rhs_module, the witness stored and its residual
//       re-derived by verify().  At the three bulk spots this is im pi_* =
//       ker r with both inclusions by solves.
// A failure aborts compute() (std::runtime_error); verify() re-derives every
// item from the stored matrices and witnesses alone.
// ===========================================================================
template<class Elt>
struct ConeModuleCertificate {
    using Buf = engine::DenseBuf<Elt>;
    engine::ModCtx<Elt> ctx;                 // prime_power(ell, k): verify() re-derives over it
    std::vector<Buf> X;                      // [q] q = 0 .. q_max_c: D_c in coordinates
    // The sequence: terms T_0 .. T_{n-1} (Hc^0, H^0, H^0(bd), Hc^1, ...), their
    // exponent lists, and the maps T_i -> T_{i+1} (n - 1 of them) in (c5) form.
    std::vector<std::string> term_names;
    std::vector<std::vector<uint32_t>> term_exps;
    std::vector<std::string> map_names;
    std::vector<Buf> maps;                   // maps[i] : T_i -> T_{i+1}, rows = #summands of T_{i+1}
    std::vector<uint32_t> im_lengths;        // length of im maps[i]
    // Item 14's witnesses: at term i, generator x of ker(maps[i]) (in T_i's
    // coordinates) = maps[i-1] w1 + D_{T_i} w2 (w1: T_{i-1} coordinates).
    struct Witness { uint32_t term = 0; std::vector<Elt> x, w1, w2; };
    std::vector<Witness> witnesses;
    bool six_invertible = true;

    uint32_t verify() const;
    uint32_t verify(std::vector<std::string>& failures) const;
    std::vector<uint32_t> failing_items() const;
};

// The identity-fibre flat restriction C^q(K; V) -> C^q(K_bd; V) of an inclusion
// K_bd -> K whose simplices carry K's vertex labels (the circle model's path):
// (N_q(K_bd) d_V) x (N_q(K) d_V), entry 1 at (s_bd d_V + a, s_K d_V + a) with
// s_K = K.find_simplex_index(q, K_bd simplex s_bd).  Throws if a simplex of K_bd
// is not one of K's.
template<class Elt>
engine::DenseBuf<Elt> RestrictionFlatRing(const SimplicialComplex& K, const SimplicialComplex& K_bd, uint32_t q,
                                          uint32_t d_V, const engine::ModCtx<Elt>& ctx);

template<class Elt>
class ModuleConeT {
public:
    using Buf = engine::DenseBuf<Elt>;
    struct Degree {
        Buf C;                                   // flat_dim(q) x (cK + cb): Howell basis of Cc^q_eq
        uint32_t cK = 0, cb = 0;                 // its bulk / boundary column counts
        Buf X;                                   // c_{q+1} x c_q
        Buf Zc;                                  // c_q x z
        engine::elim::CyclicDecomposition<Elt> dec;
        Buf H;                                   // flat_dim(q) x t: the b_i
        uint32_t length_H = 0;
    };

    // bulk, boundary: COMPUTED complexes over the same Z/ell^k (ell >= 5);
    // r_flat[q] (q <= min(top_K, top_bd)) the flat restriction.  Both must
    // outlive this object.  Throws hecke::NotImplementedInThisBuild at ell in
    // {2, 3} (the default SmallEllPolicy: the cone is not honoured there) and
    // std::logic_error / std::invalid_argument on a ring or shape mismatch.
    ModuleConeT(const EquivariantCohomologyOnKT<Elt>* bulk, const EquivariantCohomologyOnKT<Elt>* boundary,
                std::vector<Buf> r_flat);
    void compute();                              // idempotent; throws on a failing item
    bool computed() const { return computed_; }

    const engine::ModCtx<Elt>& ring_context() const { return ctx_; }
    uint32_t q_max() const { return q_max_; }                 // of H^*_c (2 on the model)
    uint32_t num_degrees() const { return q_max_ + 1; }
    uint32_t d_V() const { return d_V_; }
    uint32_t top_bulk() const { return top_K_; }
    uint32_t top_boundary() const { return top_bd_; }
    uint32_t flat_dim(uint32_t q) const { return bulk_part_dim(q) + boundary_part_dim(q); }
    uint32_t bulk_part_dim(uint32_t q) const;
    uint32_t boundary_part_dim(uint32_t q) const;
    const EquivariantCohomologyOnKT<Elt>& bulk() const { return *bulk_; }
    const EquivariantCohomologyOnKT<Elt>& boundary() const { return *bd_; }
    const Buf& r_flat(uint32_t q) const;                      // q <= min(top_K, top_bd)
    Buf D_flat(uint32_t q) const;                             // Cc^q -> Cc^{q+1}, dense mod ell^k

    // --- H^*_c (require compute(); q <= q_max) ---
    const Degree& degree(uint32_t q) const;
    std::vector<uint32_t> H_lengths() const;                  // cone_lengths_Hc^q
    const std::vector<uint32_t>& H_invariant_factors(uint32_t q) const;
    const Buf& H_module_basis(uint32_t q) const;
    // (c5) coordinates of flat cone cocycles (columns of Y, flat_dim(q) rows).
    std::vector<std::vector<Elt>> H_class_coordinates(uint32_t q, engine::DenseView<const Elt> Y) const;

    // --- eq:coneles (the (c5) matrices; target rows reduced in [0, ell^{e_i})) ---
    bool pi_built(uint32_t q) const { return q < pi_.size(); }
    bool r_built(uint32_t q) const { return q < r_.size(); }
    bool iota_built(uint32_t q) const { return q < iota_.size(); }
    const Buf& pi(uint32_t q) const;                          // H^q_c -> H^q(K), q <= min(q_max, top_K)
    const Buf& r(uint32_t q) const;                           // H^q(K) -> H^q(bd) (0 rows above top_bd)
    const Buf& iota(uint32_t q) const;                        // H^q(bd) -> H^{q+1}_c, q <= top_bd, q + 1 <= q_max
    // The eight image lengths in [C]'s order (cone_les_im_lengths).
    std::vector<uint32_t> les_im_lengths() const;
    bool coneexact_iv_certified() const { return iv_certified_; }
    const ConeModuleCertificate<Elt>& certificate() const;

private:
    const EquivariantCohomologyOnKT<Elt>* bulk_;
    const EquivariantCohomologyOnKT<Elt>* bd_;
    std::vector<Buf> r_flat_;
    engine::ModCtx<Elt> ctx_;
    uint32_t d_V_ = 0, top_K_ = 0, top_bd_ = 0, q_max_ = 0;
    bool computed_ = false, iv_certified_ = false;
    std::vector<Degree> deg_;                    // q = 0 .. q_max + 1 (the last: C only)
    std::vector<Buf> pi_, r_, iota_;
    ConeModuleCertificate<Elt> cert_;
    void require_computed(const char* who) const;
};

// (slot-24 G6) The cone certificate's span tests (C2 / C3 / C5) by the
// UNIT-ROW route when every column of the basis has a unit row (a kernel
// basis always has): Y in span(B) iff Y[r, :] == B[r, :] . Y[u, :] on the
// non-unit rows r -- one product instead of the RREF of [B | Y] (the cone's
// binding transient, 1.4 GB at n_sym = 1000).  Verdict-identical; a basis
// without unit rows takes the elimination route as before.  Default ON;
// it IS the G6 switch (Util/FieldDiet.h: thread-local, HECKE_FIELD_DIET=0 --
// exactly "0" -- selects the rollback, the elimination route on every call),
// so ScopedUnitRowSpan and HeckeOperatorOptions::ScopedFieldDiet set the same
// flag.
namespace ConeOptions {
bool unit_row_span();
void set_unit_row_span(bool on);
class ScopedUnitRowSpan {
    bool saved_;
public:
    explicit ScopedUnitRowSpan(bool on) : saved_(unit_row_span()) { set_unit_row_span(on); }
    ~ScopedUnitRowSpan() { set_unit_row_span(saved_); }
    ScopedUnitRowSpan(const ScopedUnitRowSpan&) = delete;
    ScopedUnitRowSpan& operator=(const ScopedUnitRowSpan&) = delete;
};
// Validation entry (HeckeThetaRssDietValidation's G6 arm): the certificate's
// span predicate "every column of Y lies in span(B)" on dense operands, by the
// route the calling thread's switch selects (`verdict`), and by the dense
// reference RREF (`dense`); which route ran (the unit-row route, or a fallback
// to the elimination because some column of B has no unit row).
struct SpanProbe { bool verdict = false, dense = false, unit_row_route = false, fell_back = false; };
SpanProbe span_test_for_validation(uint32_t ell, const engine::DenseBuf<uint32_t>& B, const engine::DenseBuf<uint32_t>& Y);
// (G6 audit F3) Process-wide monotone counters (diagnostic, never a value):
// span tests decided by the unit-row route, and unit-row attempts that fell
// back to the elimination route (some basis column without a unit row).
struct UnitRowStats { uint64_t calls = 0, fallbacks = 0; };
UnitRowStats unit_row_stats();
}  // namespace ConeOptions

struct EquivariantMappingConeOptions {
    // Overrides of the derived degree fields (E9 / the validation's test
    // double); left unset, both are derived.
    bool override_bulk_fields = false;
    ComplexDegreeFields bulk_fields;
    bool override_boundary_fields = false;
    ComplexDegreeFields boundary_fields;
    // Record labels the generic cone cannot know (Hecke/SL2ZBoundaryPair sets them).
    std::string boundary_model;                   // e.g. "circle"; empty = unspecified
    bool boundary_is_group_cohomology_known = false;
    bool boundary_is_group_cohomology = false;
    std::string name = "Cone(r)";
};

class EquivariantMappingCone {
public:
    using BlockMat = engine::BlockMat<uint32_t>;
    using BlockPool = engine::BlockPool<uint32_t>;

    // bulk / boundary: COMPUTED cohomologies (the bulk with its intermediates
    // retained: C_eq_block(q) is read in every degree); restriction: the
    // computed morphism with domain == boundary and codomain == bulk.  All
    // three must outlive this object.  Throws std::invalid_argument /
    // std::runtime_error on any inconsistency (prime, d_V, the morphism's
    // ends, an illegal degree-field override).
    EquivariantMappingCone(const EquivariantCohomologyOnK* bulk, const EquivariantCohomologyOnK* boundary,
                           const EquivariantCochainComplexMorphismOnK* restriction,
                           EquivariantMappingConeOptions options = EquivariantMappingConeOptions());
    ~EquivariantMappingCone();
    EquivariantMappingCone(const EquivariantMappingCone&) = delete;
    EquivariantMappingCone& operator=(const EquivariantMappingCone&) = delete;

    // Builds every object of the header note and checks the certificate on
    // the freshly built data (throws std::runtime_error on a failing asserted
    // item).  Idempotent.
    void compute();
    bool computed() const { return computed_; }

    // --- parameters and the record fields ---
    uint32_t ell() const { return ell_; }
    uint32_t d_V() const { return d_V_; }
    // (hecke-extensions slot 23, LES-2) the boundary's fibre dimension: == d_V() for every boundary pair; the level cone of
    // [C, def:levelcone] (Hecke/HeckeLevelCone) has d_V() = m_A (n + 1) and d_V_boundary() = m_B (n + 1).  F_ell route only.
    uint32_t d_V_boundary() const { return d_bd_; }
    const std::string& name() const { return options_.name; }
    const ComplexDegreeFields& bulk_fields() const { return bulk_fields_; }
    const ComplexDegreeFields& boundary_fields() const { return boundary_fields_; }
    const ComplexDegreeFields& cone_fields() const { return cone_fields_; }
    uint32_t q_max() const { return cone_fields_.q_max; }          // of H^*_c, derived
    bool truncated() const { return cone_fields_.truncated; }
    uint32_t num_degrees() const { return cone_fields_.q_max + 1; } // reported degrees 0..q_max
    const std::string& boundary_model() const { return options_.boundary_model; }
    bool boundary_is_group_cohomology_known() const { return options_.boundary_is_group_cohomology_known; }
    bool boundary_is_group_cohomology() const { return options_.boundary_is_group_cohomology; }
    const EquivariantCohomologyOnK& bulk() const { return *bulk_; }
    const EquivariantCohomologyOnK& boundary() const { return *boundary_; }
    const EquivariantCochainComplexMorphismOnK& restriction() const { return *res_; }
    const std::shared_ptr<BlockPool>& block_pool() const { return pool_; }

    // --- the flat layout ---
    uint32_t flat_dim(uint32_t q) const;          // q <= q_max + 1
    uint32_t bulk_part_dim(uint32_t q) const;     // N_q(K) d_V (0 above the bulk's top)
    uint32_t boundary_part_dim(uint32_t q) const; // N_{q-1}(K_bd) d_V (0 at q = 0 and above the boundary's top + 1)
    std::vector<uint32_t> row_dims(uint32_t q) const;   // the row partition of Cc^q

    // --- the stored objects (requires compute(); q out of range throws) ---
    const BlockMat& D_block(uint32_t q) const;    // q <= q_max
    const BlockMat& Ceq_block(uint32_t q) const;  // q <= q_max + 1
    const BlockMat& Z_block(uint32_t q) const;    // q <= q_max
    const BlockMat& B_block(uint32_t q) const;
    const BlockMat& H_block(uint32_t q) const;
    const BlockMat& P_block(uint32_t q) const;
    const BlockMat& r_flat_block(uint32_t q) const;   // q <= min(top_K, top_bd)
    uint32_t dim_Hc(uint32_t q) const;            // q <= q_max
    const std::vector<uint32_t>& dim_Hc_all() const { require_field("dim_Hc_all"); return cert_.dim_Hc; }

    // --- the maps of eq:coneles in the stored bases ---
    bool pi_built(uint32_t q) const;    // H^q_c -> H^q_eq
    bool r_built(uint32_t q) const;     // H^q_eq -> H^q_bd
    bool iota_built(uint32_t q) const;  // H^q_bd -> H^{q+1}_c
    const FlatMatrix& pi(uint32_t q) const;
    const FlatMatrix& r(uint32_t q) const;
    const FlatMatrix& iota(uint32_t q) const;
    uint32_t rank_pi(uint32_t q) const;
    uint32_t rank_r(uint32_t q) const;
    uint32_t rank_iota(uint32_t q) const;
    const std::vector<ConeLesSpot>& les_spots() const { require_field("les_spots"); return cert_.spots; }

    // --- the certificate (by value; verify() re-derives everything) ---
    const ConeCertificate& certificate() const;

    // --- the golden collectors (never emitted here) ---
    // tier A: dim_Hc^q, rank_pi^q, rank_r^q, rank_iota^q per built degree;
    // tier B: the cone bases Hc^q and projectors Pc^q, and the maps pi^q, r^q, iota^q.
    std::vector<ConeGoldenObject> golden_objects() const;
    uint64_t golden_digest() const;

    // --- (slot-9 WP-4; E8 item 4b) Z/ell^k -------------------------------
    // The cone's ring is its inputs' (both over the same Z/ell^k, which must
    // also be the ring in force at construction).  Over Z/ell^k compute() runs
    // the dense module route (ModuleConeT above), and every FIELD accessor of
    // this class (the *_block, dim_Hc, the three map families as FlatMatrix,
    // the certificate, the golden collectors, to_json) throws a named
    // std::logic_error; module_cone() is the result.
    uint32_t coefficient_ring_k() const { return ring_k_; }
    bool is_composite() const { return ring_k_ >= 2; }
    const ModuleConeT<uint32_t>& module_cone() const;

    std::string to_string() const { return to_string_offset(0); }
    std::string to_string_offset(uint32_t offset) const;
    // Hand-rolled JSON, schema "equivariant_mapping_cone/v1": the per-complex
    // degree fields (bulk, boundary, cone; the cone's derived_from), the
    // boundary labels, per degree the dimensions and ranks, the LES spots
    // with their status, and the certificate's bookkeeping.
    std::string to_json() const;

    // (slot-24 G6) Releases the cochain-level objects after compute(): D, Ceq,
    // Z, B, H, P, the inputs' copied bases / projectors and r_flat, and their
    // pool (≈ 2.4 GB at n_sym = 1000 for the C1a cone).  Afterwards the
    // dimensions (dim_Hc), the maps pi / r / iota with their ranks, the LES
    // spots and the degree fields are served as before; the *_block accessors,
    // certificate(), golden_objects() / golden_digest(), to_json() and
    // block_pool() throw std::logic_error (block_pool() returns null).  F_ell
    // route only.  Opt-in: HeckeExtDriver calls it when
    // ExtDriverOptions::release_cone_cochains is set (the census campaign, under
    // the HECKE_FIELD_DIET lever), so the cone's objects are not alive during the
    // iota phase.  Memory only.
    void release_cochain_objects();
    bool cochains_released() const { return cochains_released_; }

private:
    bool cochains_released_ = false;
    void require_cochains(const char* who) const;
    const EquivariantCohomologyOnK* bulk_;
    const EquivariantCohomologyOnK* boundary_;
    const EquivariantCochainComplexMorphismOnK* res_;
    EquivariantMappingConeOptions options_;
    uint32_t ell_ = 0, d_V_ = 0;
    uint32_t d_bd_ = 0;                   // (LES-2) the boundary's fibre dimension (== d_V_ for a boundary pair)
    uint32_t top_K_ = 0, top_bd_ = 0;     // top degrees of the two finite complexes
    ComplexDegreeFields bulk_fields_, boundary_fields_, cone_fields_;
    bool computed_ = false;

    std::shared_ptr<BlockPool> pool_;
    ConeCertificate cert_;
    uint32_t ring_k_ = 1;                                // (slot-9 WP-4) 1 = F_ell
    std::unique_ptr<ModuleConeT<uint32_t>> module_;       // (slot-9 WP-4) the Z/ell^k result
    void require_field(const char* who) const;           // a field accessor on a composite cone: logic_error
    void init_composite();                               // (slot-9 WP-4) the Z/ell^k constructor tail
    std::vector<uint32_t> rank_pi_, rank_r_, rank_iota_;

    void require_computed(const char* who) const;
    void check_cone_degree(uint32_t q, uint32_t max_q, const char* who) const;
    BlockMat assemble_D(uint32_t q) const;
    BlockMat assemble_Ceq(uint32_t q) const;
    BlockMat assemble_r_flat(uint32_t q) const;
    void build_maps();
    void build_spots();
};

#endif // EQUIVARIANT_MAPPING_CONE_H
