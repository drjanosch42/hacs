// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_DESCENT_DESCENT_TREE_PAIR_H
#define INTEGRAL_DESCENT_DESCENT_TREE_PAIR_H

// ===========================================================================
// DESCENT milestone D5c (hecke-extensions slot 24; docs/implementation/slot24/DESCENT_plan.md v0.4 Sec. 4 "D5c", clause C-D8).
// [C] = docs/latex/gl2_contract.tex (v1.23.0-pre.34, sec:treepair; refereed C-D8_referee.md, ACCEPT after fixes m1-m4, N1-N8), cited
// by label: def:treepair, lem:treepaircyl, prop:treepair (1)-(6), rmk:treepairscope and the CODE MAP after it, whose obligations
// (a)-(g) are this unit's specification; def:descentpair, prop:descentpair (1)(c) (signs), (4), (5), (6) (precision), prop:descentpairhecke
// (the pair lift, its refusal codes), prop:isomodeltree (2), (3), (4), (6), prop:torsionhecke (3).
//
// THE TREE CUSP PAIR AS THE ROUTE OF THE DENSE ell-LOCAL PAIR PIECES (code map (a)-(c)).  D4's pair piece at 2 (and its dense
// 3-adic piece) is the cone of the restriction (Iso^(ell), Isob^(ell)): the dense bulk Iso^(2) inside the cone walls it at D = rank M_N
// ~ 1060 - 1240 (DESCENT4_report.md Sec. 4).  Here the same groups and maps are read on the TREE CONE Cone(a^(ell)*) of def:treepair(3):
//     Cone^q = C^{q+1}_{T,ell}(M) (+) B^q_ell(M),   D(alpha, c) = (-d alpha, d c + a^* alpha),
// the tree cochains of prop:isomodeltree(2) (TreeCochains, in J-coordinates on both sides: G^{-1} pi_R d B_J, D2's BuildRedTree rule),
// the cusp cochains B^q_ell(M) = Hom(B^(ell)_T, M) and the axis map a^(ell)* in the closed form of prop:treepair(2)(a), (b):
//     ell = 2:  B^0 = M, B^q = M (+) M (u on e_q, w on z_{q-1}); d^0 u = ((1 - c) u, (1 - T) u),
//               d^q (u, w) = (eps_{q+1} u, (1 - T) u - eps_q w);  a^*(x, y) = x (q = 0),
//               a^*(x, y, z) = (a_q x, [q odd] S^{-1} (y - x) + (S^{-1} - T) z), a_q = 1 (even), 1 + S (odd);
//     ell = 3:  B^0 = B^1 = M^{-I}, d^0 = 1 - T; a^*(x, y) = x, a^*(y, z) = S^{-1} y + (S^{-1} - T) z on C^1, 0 above.
// It is a PairComplex (c = the tree's ranks, cb = the cusp model's, d, db, r = a^*), so D4's RunPairLes reads it VERBATIM: the three
// terms of eq:descentpairles with the signs of prop:descentpair(1)(c) (pi_* = coneproj_* = -delta, iota_* = coneinc_*; the connecting
// map is H(a^*), no sign: lem:treepaircyl(3)), the maps, exactness as subgroups, the a-priori exponents; the boundary against
// BoundaryClosedForm (prop:descentpair(4)), the rational ranks by the Betti recursion and, by default, every cone differential mod a
// qualifying ell' on THIS tree cone (prop:treepair(5)(a): never a rank of (Iso, Isob)'s cone, of Cone(rres_Cyl) or of the bulk tree).
// Cone ranks ((2)(d), ell = 2): d + r, 3d + r, 4d + r, ... (r = rank M^{U^2} ~ d / 3) against ~ 16 1/3 d, 24 1/3 d on (Iso^(2), Isob^(2)).
//
// CERTIFICATES OF CONSTRUCTED MAPS (prop:treepair(6); a failure is hecke::HeckeInternalError, never a refusal): d_B o d_B = 0 and
// d_B a^* = a^* d_T entrywise on the built matrices (BuildTreePairComplex, every call); the chain-level identities of (1)(b), (3)(a)-(c)
// exactly in Q[Gamma] (SharedTreePairComparison, once per ell and process); the cone maps and the transported operator below.
//
// THE SWITCH (code map (c)): HECKE_DESCENT_TREE_PAIR, read-only, read once per process and once per call: unset or 1 = the tree cusp
// pair (the default route of the 2-adic pair piece and of the dense 3-adic one), 0 = D4's dense (Iso^(ell), Isob^(ell)) pair; any other
// value warns and keeps the default.  HECKE_DESCENT_TREE=0 (D5's global rollback) also selects the dense pair.  ScopedDescentTreePair
// overrides it process-wide (validation use).  VALUE-PRESERVING in the sense of the record's invariants (the C-D8 referee's N6): the
// group types, the map types and the choice-free Hecke invariants (prop:torsionhecke) agree; Smith generators and coordinates do not.
//
// THE COMPARISONS AND HECKE BY TRANSPORT (prop:treepair(3)(d), (4); code map (d), (e)).  TreePairCochains carries, at one (ell, N, n),
// the tree pair, the free Iso pair of D4 (full layout, BuildFreePairN), D3's TreeTransport (Phi_T^*, Psi_T^* and the J-coordinates),
// and the chain-level TreePairComparison (Phi_d, Psi_d, Y_Phi, Y_Psi); every map is applied to VECTORS (blocks of the block-monomial
// action, linear in D per term), never formed as a dense Iso-sized matrix:
//     j_Phi(alpha, c)   = (Phi_T^* alpha, Phi_d^* c - Y_Phi^* alpha) : Cone(a^*)^j -> Cone(rres_Iso)^j,
//     j_Psi(alpha', c') = (Psi_T^* alpha', Psi_d^* c' - Y_Psi^* alpha') : back;
//     T^c_T = j_Psi o (T~_Iso[1], T~_Isob) o j_Phi = (T'_T alpha, T'_d c + eta_T alpha),
// T~ = D3's IsoHeckeN (eq:levelcorr with D4's certified pair lift: StrictLift / pair::PairLiftCostOf, reused, not duplicated) and
// T~_Isob its boundary rows (apply_boundary: the closure (B) on the operator's layout, else IsobClosure).  The certificates (code map
// (e)): on the operator APPLIED (apply_hecke), its blocks = (T'_T, eta_T; 0, T'_d) assembled separately (the strict blocks), the eta identity d eta + eta d = T'_d a^* - a^* T'_T -- ENTRYWISE (eta and
// both sides as matrices on every basis vector of C^{j+1}_T) when D <= kEtaEntrywiseMaxD, else on the columns handed in (the
// presentation's generators; the C-D8 referee's N5: entrywise costs ~ 2.3 D applications of T~ per degree), and D T^c g = 0 for every
// generator g handed in.  A failure throws hecke::HeckeInternalError.
//
// REFUSALS (prop:treepair(6)): the groups refuse on RSS (PredictTreePairRss above 0.9 x the ceiling, before building; ReadTreePairHecke
// adds the transport's TreePairCochains::transport_bytes on the built layout, before any operator: the D5c audit's N2), on an exhausted
// precision (RunPairLes / BuildPres), on a failed rank certification; T_p refuses with exactly D4's PairRefusal codes (NotPrime,
// PDividesN, DegreeRange (q_max not in {4, 5}), RssCeiling, IntegerCap, DomainCap, InputTableCertificate, LiftCertificate, IsobClosure).
// No settings key, no record key, no golden object; no env variable but the read-only switch.
// ===========================================================================

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "DescentHecke.h"
#include "DescentPair.h"
#include "../Model/TreeCochains.h"
#include "../Model/TreePair.h"
#include "../../../ModArithmetic/EnvLever.h"

namespace integral_descent {

// HECKE_DESCENT_TREE_PAIR (read once; process-wide) AND DescentTreeEnabled().  It never changes a value.
bool DescentTreePairEnabled();
std::string DescentTreePairSource();   // "default" | "env" | "scoped" | "tree-rollback" (HECKE_DESCENT_TREE=0)
const engine::lever::EnvLever& DescentTreePairLever();   // (S24 CL-5) the EnvLever behind it (EnvLever.h)
struct ScopedDescentTreePair : engine::lever::EnvLever::Scoped {
    explicit ScopedDescentTreePair(bool on) : Scoped(DescentTreePairLever(), on) {}
};

namespace treepair {

using maps::Mat;
using maps::ZMod;
using U32Ctx = engine::ModCtx<uint32_t>;

// ---- (a) the tree pair complex ----------------------------------------------------------------------------------------------
// ell in {2, 3}; ctx = Z/ell^K (the Smith route) or a field F_ell' with ell' prime to q_ell (the rank source (i), on THIS complex); M the
// module by its block-monomial action; cochains to degree T.  J-coordinates on both sides.  Throws hecke::HeckeInternalError when d_B o
// d_B != 0 or d_B a^* != a^* d_T (entrywise); route = "tree".
pair::PairComplex BuildTreePairComplex(uint32_t ell, const U32Ctx& ctx, const integral_model::TreeModule<uint32_t>& M, uint32_t T);
// The ranks over F_ell' of the tree cone's differentials D^j, j = -1 .. T - 2 (the code map's (b); ell' >= 5 prime, prime to q_ell).
std::vector<uint32_t> TreeConeRanksModPrime(uint32_t ell, uint32_t N, uint32_t n, uint32_t ellp, uint32_t T);
// Rule 19 for the tree pair piece at module dimension D, cochains to T: the structural count (the stored d, d_B, a^*, two cone
// differentials, the largest Smith form with its transforms) times the calibrated factor plus 256 MiB (DESCENT5c_predictions.md).
uint64_t TreePairRssStructural(uint32_t ell, uint32_t D, uint32_t T);
uint64_t PredictTreePairRss(uint32_t ell, uint32_t D, uint32_t T);

// The chain-level comparison of prop:treepair(3) at ell (on D3's SharedTreeComparison(), degree kTreeComparisonTop), built and
// certified once per process; a failed certificate throws hecke::HeckeInternalError.  Thread-safe.
const integral_model::TreePairComparison& SharedTreePairComparison(uint32_t ell);

// ---- (d), (e) the cone maps and the transport --------------------------------------------------------------------------------
constexpr uint32_t kEtaEntrywiseMaxD = 1300;   // the eta identity entrywise up to D4's wall (the C-D8 referee's N5)
// Above kEtaEntrywiseMaxD (or with force_generators), the generators' checks see only cocycles; the blocks of the applied operator and
// the eta identity are then ALSO checked on kFreivaldsVectors random vectors of C^{j+1}_T (and of B^j for the boundary block), uniform in
// (Z/ell^K), deterministically seeded by (ell, N, n, j) (the C-D8b referee's Arm (G)): for a non-zero error matrix E over Z/ell^K and a
// uniform v, Pr[E v = 0] <= 1/ell (reduce E / ell^{v(E)} mod ell), so one check misses with probability <= ell^{-k}, k = 64: <= 2^-64.
constexpr uint32_t kFreivaldsVectors = 64;

// VALIDATION ONLY (the C-D8b referee's Arm (G), ported): a planted error in a SHARED primitive -- Y_Psi^* at chain degree q gains
// alpha |-> (r . alpha) z, r seeded, z a d_B-cocycle of B^q handed in -- so the applied operator AND the separately assembled eta_T carry
// it (the block comparison cannot see it), its values are d_B-cocycles (D T^c g = 0 and the identity on the generators cannot see it);
// only the eta identity one cone degree lower on all of C^q (entrywise, or the random vectors) does.  Process-wide; never set in
// production.
struct ScopedPlantedCocycleEta {
    ScopedPlantedCocycleEta(uint32_t q, std::vector<uint32_t> z, uint64_t seed);
    ~ScopedPlantedCocycleEta();
    ScopedPlantedCocycleEta(const ScopedPlantedCocycleEta&) = delete;
    ScopedPlantedCocycleEta& operator=(const ScopedPlantedCocycleEta&) = delete;
};

// A cochain endomorphism T = (T_b[1], T_d) of Cone(rres_Iso)(M_N (x) R) with rres T_b = T_d rres strictly (prop:treepair(4)'s
// hypothesis), on the free model's full layout (cell * D + component): apply = T_b on C^q_I; apply_boundary = T_d on Isob^q (bpos: the
// position of each Iso cell among the boundary cells, -1 off Isob), returning false iff T_b's boundary rows reach off Isob (then T is
// not a pair datum on this layout: IsobClosure).
class IsoCochainOp {
public:
    virtual ~IsoCochainOp() = default;
    virtual uint32_t q_top() const = 0;
    virtual void apply(uint32_t q, const std::vector<uint32_t>& X, uint32_t k, std::vector<uint32_t>& Y) const = 0;
    virtual bool apply_boundary(uint32_t q, const std::vector<int32_t>& bpos, const std::vector<uint32_t>& X, uint32_t k,
                                std::vector<uint32_t>& Y) const = 0;
};
// The level-N Hecke operator Phi^(N)_p of prop:descentpairhecke(2): D3's IsoHeckeN (D4's certified pair lift), applied sparsely.
class IsoHeckeOp final : public IsoCochainOp {
public:
    explicit IsoHeckeOp(const hecke3::IsoHeckeN& T) : T_(T) {}
    uint32_t q_top() const override { return T_.q_top(); }
    void apply(uint32_t q, const std::vector<uint32_t>& X, uint32_t k, std::vector<uint32_t>& Y) const override { T_.apply(q, X, k, Y); }
    bool apply_boundary(uint32_t q, const std::vector<int32_t>& bpos, const std::vector<uint32_t>& X, uint32_t k,
                        std::vector<uint32_t>& Y) const override {
        return T_.apply_boundary(q, bpos, X, k, Y);
    }
private:
    const hecke3::IsoHeckeN& T_;
};
// A dense operator per degree on the full layout (validation scale): T_d = the boundary block of T_b; closure checked.
class DenseIsoOp final : public IsoCochainOp {
public:
    DenseIsoOp(const ZMod& Z, std::vector<Mat> Tb, uint32_t D) : Z_(Z), Tb_(std::move(Tb)), D_(D) {}
    uint32_t q_top() const override { return static_cast<uint32_t>(Tb_.size()) - 1; }
    void apply(uint32_t q, const std::vector<uint32_t>& X, uint32_t k, std::vector<uint32_t>& Y) const override;
    bool apply_boundary(uint32_t q, const std::vector<int32_t>& bpos, const std::vector<uint32_t>& X, uint32_t k,
                        std::vector<uint32_t>& Y) const override;
    const Mat& bulk(uint32_t q) const { return Tb_.at(q); }
private:
    ZMod Z_;
    std::vector<Mat> Tb_;
    uint32_t D_;
};

struct TreePairCertCounts {
    uint64_t eta_entries = 0;       // entries of the eta identity compared
    uint64_t strict_entries = 0;    // entries of the strict block T^c(0, c)_1 = 0 compared
    uint64_t block_entries = 0;     // entries of the APPLIED operator's blocks compared with T'_T, eta_T, T'_d (the D5c audit's m1)
    uint64_t cocycle_columns = 0;   // generators g with D T^c g = 0 checked
    uint64_t random_columns = 0;    // random vectors the blocks and the eta identity were checked on (above kEtaEntrywiseMaxD)
    bool eta_entrywise = false;     // the eta identity was checked on every basis vector
    uint32_t eta_degrees = 0;       // cone degrees whose eta identity was checked
    uint32_t eta_skipped = 0;       // cone degrees whose eta identity was NOT checked (the comparisons stop at kTreeComparisonTop: at
    int eta_skipped_top = -1;       // q_max = 5 the top cone degree 4, H^5_c; the D5c audit's m2) and the highest such degree
    double eta_seconds = 0;
};

class TreePairCochains {
public:
    // ell in {2, 3}; cochains to degree T (<= kTreeComparisonTop + 1 = 6; the cone maps to cone degree T - 1, the transport to T - 2);
    // ctx = Z/ell^K (K = 0: LocalWidth(ell)).
    TreePairCochains(uint32_t ell, uint32_t N, uint32_t n, uint32_t T, uint32_t K = 0);
    ~TreePairCochains();
    uint32_t ell() const { return ell_; }
    uint32_t N() const { return N_; }
    uint32_t n() const { return n_; }
    uint32_t D() const { return D_; }
    uint32_t T() const { return T_; }
    const ZMod& Z() const { return Z_; }
    const U32Ctx& ctx() const { return ctx_; }
    const pair::PairComplex& tree() const { return X_; }   // the tree pair (J-coordinates)
    // The free Iso pair of D4 at this level on its full layout (BuildFreePairN; built on first use: Iso-sized, validation scale).
    const pair::PairComplex& iso() const;

    // j_Phi : Cone(a^*)^j -> Cone(rres_Iso)^j and j_Psi back, on columns (X: cone rank x k); j = -1 .. min(T - 1, kTreeComparisonTop - 1).
    Mat j_phi(int j, const Mat& X) const;
    Mat j_psi(int j, const Mat& Y) const;
    // T^c_T = j_Psi (T~_Iso[1], T~_Isob) j_Phi on Cone(a^*)^j (columns), and its blocks; refused_closure is set when T~ is not closed on
    // Isob (the IsobClosure refusal: the composite is still returned, but it is not a pair operator and the caller must refuse).
    Mat apply_hecke(const IsoCochainOp& T, int j, const Mat& X, bool& refused_closure) const;
    Mat bulk_hecke(const IsoCochainOp& T, uint32_t q, const Mat& X) const;                  // T'_T on C^q_T (J-coordinates)
    Mat bdry_hecke(const IsoCochainOp& T, uint32_t q, const Mat& X, bool& closed) const;    // T'_d on B^q
    Mat eta(const IsoCochainOp& T, int j, const Mat& A, bool& closed) const;                // eta_T on C^{j+1}_T -> B^j
    // The certificates of code map (e) on cone degree j, ON THE OPERATOR apply_hecke APPLIES (the D5c audit's m1): its blocks equal the
    // separately assembled ones, apply_hecke(alpha, 0) = (T'_T alpha, eta_T alpha) and apply_hecke(0, c) = (0, T'_d c) (alpha on every basis
    // vector of C^{j+1}_T when D <= kEtaEntrywiseMaxD or `entrywise`, else the alpha-parts of X; c the c-parts of X); D T^c g = 0 for the
    // columns of X (cocycles, checked mod ell^{prec}); and the eta identity on the same alphas -- when the comparisons reach degree j + 2,
    // else the degree is counted in cnt.eta_skipped.  Returns false (nothing certified past the closure test) when T~ is not closed on Isob
    // (the caller refuses IsobClosure); throws hecke::HeckeInternalError on a failed certificate.
    // Above kEtaEntrywiseMaxD (or force_generators: the same mode at any D, validation) the blocks and the identity are checked on the
    // generators AND on `random_vectors` seeded random vectors (kFreivaldsVectors; 0 = none, validation only).
    bool certify_transport(const IsoCochainOp& T, int j, const Mat& X, uint32_t prec, bool entrywise, TreePairCertCounts& cnt,
                           bool force_generators = false, uint32_t random_vectors = kFreivaldsVectors) const;

    // pullbacks on the full layouts (validation and the maps above): Phi_d^* : B^q -> Isob^q, Psi_d^* : Isob^q -> B^q, Y_Phi^* : C^{q+1}_T
    // (full) -> Isob^q, Y_Psi^* : C^{q+1}_I -> B^q.
    Mat phi_d_star(uint32_t q, const Mat& C) const;
    Mat psi_d_star(uint32_t q, const Mat& C) const;
    Mat y_phi_star(uint32_t q, const Mat& Alpha_full) const;
    Mat y_psi_star(uint32_t q, const Mat& Alpha_iso) const;
    // Phi_T^* : C^q_T (J) -> C^q_I (full) and Psi_T^* back (D3's TreeTransport maps, J-coordinates on the tree)
    Mat phi_T_star(uint32_t q, const Mat& U) const;
    Mat psi_T_star(uint32_t q, const Mat& A) const;
    // validation: the pullbacks of the CHAIN-LEVEL a^(ell) (C^q_T (J) -> B^q) and of the cusp model's chain differential (B^q -> B^{q+1})
    // -- to compare with the closed form of prop:treepair(2) entry by entry
    Mat axis_star_chain(uint32_t q, const Mat& U) const;
    Mat cusp_d_chain(uint32_t q, const Mat& C) const;
    const hecke3::TreeTransport& transport() const { return *tt_; }
    uint32_t isob_rank(uint32_t q) const;   // Isob cells(q) * D
    // Rule 19 for the transport (the D5c audit's N2, code map (g)): the Iso-sized vectors of one apply_hecke / eta on `cols` columns at
    // the widest cone degree <= q_max - 1 (Phi_T^* alpha, T~ of it, the boundary blocks, the results), six live at once, in bytes.  The operator T~'s own terms are the pair lift's (its rule 19 refuses through Lift[RssCeiling]).
    uint64_t transport_bytes(uint32_t q_max, uint32_t cols) const;
    double build_seconds = 0;

private:
    uint32_t ell_, N_, n_, T_, D_ = 0;
    ZMod Z_;
    U32Ctx ctx_;
    integral_model::TreeModule<uint32_t> TM_;
    pair::PairComplex X_;
    mutable std::unique_ptr<pair::PairComplex> iso_;
    std::unique_ptr<hecke3::TreeTransport> tt_;
    const integral_model::TreePairComparison* cmp_ = nullptr;
    std::vector<std::vector<int32_t>> bpos_;   // [q][Iso cell] -> position among Isob cells, -1 off Isob
};

// ---- the read of T_p on the three terms through the tree (code map (e), (f)) -----------------------------------------------------
struct TreePairHeckeOptions {
    uint32_t q_max = 4;                       // {4, 5}; else DegreeRange
    bool eta_entrywise = false;               // force the entrywise eta identity above kEtaEntrywiseMaxD
    bool force_generators = false;            // validation: the above-kEtaEntrywiseMaxD mode (generators + random vectors) at any D
    uint32_t random_vectors = kFreivaldsVectors;   // that mode's random vectors (0 = generators only: validation, the referee's gap)
    uint64_t rss_cap_bytes = 40ull << 30;
};
struct TreePairHeckeOp {
    uint32_t p = 0;
    bool refused = false;
    pair::PairRefusal code = pair::PairRefusal::None;
    std::string refusal;
    // induced matrices on the tree pair's presentations, q = 0 .. q_max (Hc: q = 1 .. q_max, index q; Hc[0] empty), with precision
    std::vector<Mat> on_H, on_Hd, on_Hc;
    std::vector<ZMod> Z_H, Z_Hd, Z_Hc;
    TreePairCertCounts cert;
    double seconds = 0;
};
struct TreePairHeckeRead {
    uint32_t ell = 0, N = 1, n = 0, q_max = 4, D = 0;
    pair::PairLocal L;                        // the tree pair's presentations (the groups of the piece)
    pair::PieceReport rep;
    std::vector<TreePairHeckeOp> ops;
    bool refused = false;
    std::string refusal;
    uint64_t rss_pred = 0;                    // the groups (PredictTreePairRss) + the transport (TreePairCochains::transport_bytes)
    uint64_t rss_pred_transport = 0;
    double seconds = 0;
};
// The tree pair's sequence at ell with T_p, p in ps, on H^q, H^q(d), H^q_c (q <= q_max) by transport.  The pair cochains to q_max + 1.
TreePairHeckeRead ReadTreePairHecke(uint32_t ell, uint32_t N, uint32_t n, const std::vector<uint32_t>& ps,
                                    const TreePairHeckeOptions& opt = {});
// The transport of any pair operator T (IsoCochainOp) onto the tree pair's presentations L (RunPairLes on C.tree(), q <= q_max): the
// induced matrices on H^q (T'_T), H^q(d) (T'_d) and H^q_c (T^c_T), every H^q_c read certified by certify_transport.  Refuses (op.code)
// IsobClosure and the named caps; a failed certificate throws hecke::HeckeInternalError.
void TransportOnPresentations(const TreePairCochains& C, const pair::PairLocal& L, uint32_t q_max, const IsoCochainOp& T, bool eta_entrywise,
                              TreePairHeckeOp& op, bool force_generators = false, uint32_t random_vectors = kFreivaldsVectors);
// The matrix of a map on presentations from its images of X's generators (Y.c x g_X), as pair::InducedMatrix.
Mat InducedFromImages(const pair::Pres& X, const pair::Pres& Y, const Mat& images, ZMod& Zout);

}  // namespace treepair
}  // namespace integral_descent

#endif
