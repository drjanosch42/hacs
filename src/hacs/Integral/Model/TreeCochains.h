// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_MODEL_TREE_COCHAINS_H
#define INTEGRAL_MODEL_TREE_COCHAINS_H

// ===========================================================================
// DESCENT milestone D5 (hecke-extensions slot 24; docs/implementation/slot24/DESCENT_plan.md v0.4 Sec. 4 "D5", brought
// forward by the owner's ruling at gate G-T of 2026-10-25).  [C] = docs/latex/gl2_contract.tex, cited by label:
// def:isomodeltree, prop:isomodeltree (clause C-D6, v1.23.0-pre.23 (C-D6 refereed ACCEPT at pre.21, editorial N1-N4 at pre.22), C-D6_referee.md).
//
// THE PRODUCTION HALF OF THE TREE MODELS: the ell-local cochain complexes C^._{T,ell}(M) = Hom(Iso^(ell)_T, M) of
// [C, prop:isomodeltree](2) in CLOSED FORM, in reduced coordinates, for any Z[Gamma]-module M free over Z given by
// its block-monomial action (Sym^n at level 1: one dense block; Ind_{Gamma_0(N)}^Gamma Sym^n: the coset-monomial
// action of hecke::level::MonomialOf, never densified).  The cochains are C^q = (+)_x M^{Q_x} over the tree cells x
// of degree q ([C, prop:isomodeltree](1)(c)); M^{Q_x} = im rho(e_x) is a free direct summand over Z_(ell) and its
// basis is the columns J of rho(e_x) whose reductions mod ell are a column rank profile (Nakayama; the same rule as
// DescentAssembly's select_block, [C, prop:certifiedeell](5)), the coordinate projection pi_R onto the rows R of a row
// rank profile of rho(e_x)[:, J] mod ell being an isomorphism on im rho(e_x).  The matrix of d^q handed to a Smith
// engine is pi_R(target) (d^q) B_J(source) -- every block is rows R / columns J of rho(a) for an element a of
// O_ell[Gamma] (the idempotent folded into a in the group ring), so no D x D product is ever formed.  The ranks
// in these coordinates are [C, prop:isomodeltree](2)(e)'s.
//
//   ell = 2:   C^0 = M (+) M^{U^2},  C^q = M (+) M^{U^2} (+) M (q >= 1), (2)(a); d^{q+2} = d^q for q >= 1 ((2)(d)), so
//              the distinct differentials are d^0, d^1, d^2;
//   ell = 3:   C^0 = M^{<S>} (+) M^{-I}, C^1 = M^{-I} (+) M^{-I}, C^q = M^{-I} (q >= 2), (2)(b); d^{q+2} = d^q for q >= 2;
//   ell >= 5:  C^0 = M^{<S>} (+) M^{<U>}, C^1 = M^{-I}, d^0(x, y) = x - y, (2)(c) (the Bredon tree).
//
// The block-monomial grouping also makes the image selection cheap at level N: rho(e_x) is block diagonal over the
// orbits of Q_x on the cosets, and the F_ell rank profiles are taken component by component (union-find on the
// block pattern), each component a dense |orbit| d x |orbit| d matrix (d = n + 1).
//
// No bignum, no settings key, no record key, no golden object; nothing here reads an env variable.
// ===========================================================================

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "IsotropyModelEll.h"
#include "../../Hecke/ExactInt.h"
#include "../../../ModArithmetic/DenseBuf.h"
#include "../../../ModArithmetic/ModCtx.h"

namespace integral_model {

// rho(g) as a block-monomial matrix: row block j (of m) carries one d x d block at column block col[j].
template<class Elt>
struct BlockMonomial {
    uint32_t m = 1, d = 0;
    std::vector<uint32_t> col;                 // [j] = column block of row block j (0-based)
    std::vector<uint32_t> blk;                 // [j] = index into blocks
    std::vector<std::vector<Elt>> blocks;      // d x d, row-major, reduced mod ctx.p
};

// A Z[Gamma]-module free over Z (reduced into a ModCtx), by its block-monomial action; rho() is memoised.
template<class Elt>
class TreeModule {
public:
    std::string name;
    uint32_t m = 1, d = 0;
    std::function<BlockMonomial<Elt>(const hecke::Mat22&)> make;
    uint32_t dim() const { return m * d; }
    // The memo is shared by copies and filled from const methods: one mutex guards it (D5 audit N6; std::map references stay
    // valid across inserts, so the returned reference outlives the lock).
    const BlockMonomial<Elt>& rho(const hecke::Mat22& g) const {
        const GKey k = key_of(g);
        std::lock_guard<std::mutex> lock(*mutex_);
        auto it = cache_->find(k);
        if (it == cache_->end()) it = cache_->emplace(k, make(g)).first;
        return it->second;
    }

private:
    std::shared_ptr<std::map<GKey, BlockMonomial<Elt>>> cache_ = std::make_shared<std::map<GKey, BlockMonomial<Elt>>>();
    std::shared_ptr<std::mutex> mutex_ = std::make_shared<std::mutex>();
};

// Sym^n Z^2 over ctx (m = 1, d = n + 1; rho_n of IsotropyModel.h).
template<class Elt>
TreeModule<Elt> TreeSymModule(const engine::ModCtx<Elt>& ctx, uint32_t n);
// Ind_{Gamma_0(N)}^Gamma Sym^n over ctx through the shipped Gamma0LevelN table (hecke::level::MonomialOf), the module
// of [C, sec:descentN] (the same module as DescentAssembly's InducedSymModule, which densifies it).
TreeModule<uint32_t> TreeInducedSymModule(uint32_t N, uint32_t n, const engine::ModCtx<uint32_t>& ctx);
// Any CoefficientModule as one dense block (m = 1).
template<class Elt>
TreeModule<Elt> TreeModuleFromDense(const CoefficientModule<Elt>& M);

// A basis of im rho(e) over Z_(ell) (columns J) and an isomorphism onto coordinates (rows R); pos maps a global
// index to its place in J resp. R (-1 if absent).
struct ImageSelection {
    bool all = false;                   // e = 1: J = R = everything
    std::vector<uint32_t> J, R;
    std::vector<int32_t> jpos, rpos;
    uint32_t rank() const { return static_cast<uint32_t>(J.size()); }
};
// e any element of O_ell[Gamma] with rho(e) idempotent (ctx Z/ell^K or a field F_p, ell = ctx.ell the residue prime).
template<class Elt>
ImageSelection SelectImage(const engine::ModCtx<Elt>& ctx, const TreeModule<Elt>& M, const QGroupRingElt& e);

// out[r0 + rows.rpos[i], c0 + cols.jpos[j]] += rho(a)[i, j] over rows R and columns J (out row-major, leading dim ld).
template<class Elt>
void AddRhoBlock(const engine::ModCtx<Elt>& ctx, std::vector<Elt>& out, uint32_t ld, uint32_t r0, uint32_t c0,
                 const TreeModule<Elt>& M, const QGroupRingElt& a, const ImageSelection& rows, const ImageSelection& cols);

// The canonical group-ring elements of def:isomodeltree: S, U = ST, c = -I and the idempotents.
namespace tree_elt {
QGroupRingElt one();
QGroupRingElt g(const hecke::Mat22& m, int64_t num = 1, int64_t den = 1);
QGroupRingElt S();
QGroupRingElt U();
QGroupRingElt c();
QGroupRingElt eU2();   // (1 + U^2 + U^4) / 3   (e_U at ell = 2)
QGroupRingElt eS4();   // N_S / 4               (e_S at ell = 3, ell >= 5)
QGroupRingElt eU6();   // N_U / 6               (e_U at ell >= 5)
QGroupRingElt ec();    // (1 + c) / 2           (e_e at ell = 3, ell >= 5; e_U at ell = 3)
QGroupRingElt NS();    // 1 + S + S^2 + S^3
QGroupRingElt sum(const QGroupRingElt& a, const QGroupRingElt& b, int64_t sb = 1);
}  // namespace tree_elt

// The ell-local tree cochains of [C, prop:isomodeltree](2) in reduced coordinates.  local_ell = 2, 3, or any prime
// >= 5 (the Bredon tree); ctx may be Z/ell^K (the Smith route) or a field F_ell' with ell' prime to q_ell (the rank
// source (4)(c)(i) of [C, prop:certifiedeell], read on THIS complex, [C, prop:isomodeltree](1)(d)).
template<class Elt>
struct TreeLocalForm {
    uint32_t ell = 0;            // the model's prime (2, 3, or >= 5)
    uint32_t D = 0;
    ImageSelection sel_S, sel_U, sel_E;   // M^{Q_S}, M^{Q_U}, M^{Q_e} (ell = 3: sel_U = sel_E = M^{-I})
    uint32_t q_top = 0;          // differentials d^0 .. d^{q_top - 1} are addressable
    uint32_t cochain_rank(uint32_t q) const;
    // The q' <= q with d^q == d^{q'} as matrices ([C, prop:isomodeltree](2)(d)); q itself below the period's start.
    uint32_t representative(uint32_t q) const;
    // d^q : C^q -> C^{q+1}, row-major, rows = cochain_rank(q + 1), cols = cochain_rank(q).
    std::vector<Elt> differential(const engine::ModCtx<Elt>& ctx, const TreeModule<Elt>& M, uint32_t q) const;
};
template<class Elt>
TreeLocalForm<Elt> MakeTreeLocalForm(uint32_t local_ell, const engine::ModCtx<Elt>& ctx, const TreeModule<Elt>& M,
                                     uint32_t q_top);

// The tree cochain ranks of [C, prop:isomodeltree](2)(e) from the fixed-space ranks (for RSS predictions).
uint32_t TreeCochainRank(uint32_t ell, uint32_t q, uint32_t r1, uint32_t rS, uint32_t rU, uint32_t rE);

// Validation-only fault injection (D5 audit m3 / m4: negative controls that reach the production tree route).  One
// process-wide selector, None by default; set ONLY through ScopedTreeFault by HeckeIsotropyModelTreeValidation's (NT) and
// (E) arms -- never from an env variable, a setting or a CLI token, so no run outside those arms can see a fault.  Each
// fault is one term of [C, prop:isomodeltree](2) changed (TreeLocalForm::differential) or one G^{-1} of D2's tree
// reduction dropped (DescentMaps.cpp BuildRedTree / ReduceFibreTree); the numbering is the audit's (D5_code_audit.md N2).
enum class TreeFault : uint32_t {
    None = 0,
    Ell2OddEdgeSign = 1,     // ell = 2, odd d^q: the edge row's (1 + S) x -> (1 - S) x
    Ell2EvenZ = 2,           // ell = 2, even d^q (q >= 2): - (1 + c) z -> - (1 - c) z
    Ell3OddNorm = 3,         // ell = 3, odd d^q (q >= 3): (1 + U + U^2) -> (1 - U)
    Ell2D0DropY = 4,         // ell = 2, d^0: the edge row's - y dropped
    BredonDropY = 5,         // ell >= 5, d^0: the - y term dropped
    Ell2OddZSign = 6,        // ell = 2, odd d^q: the sign of the z column (a unit column scaling; d o d != 0)
    Ell3D0EdgeIdem = 7,      // ell = 3, d^0: e_S not folded into the edge's x (e_c in place of N_S / 4)
    Ell2OddNoEU = 8,         // ell = 2, odd d^q: (1 + c) y without e_U folded in
    D2FibreNoGinv = 9,       // D2 ReduceFibreTree: no G_Y^{-1} on the target
    D2DiffNoGinv = 10,       // D2 BuildRedTree: no G^{-1} on the differential
};
constexpr uint32_t kTreeFaultCount = 10;
TreeFault ActiveTreeFault();
struct ScopedTreeFault {
    explicit ScopedTreeFault(TreeFault f);
    ~ScopedTreeFault();
    ScopedTreeFault(const ScopedTreeFault&) = delete;
    ScopedTreeFault& operator=(const ScopedTreeFault&) = delete;
private:
    uint32_t saved_;
};

}  // namespace integral_model

#endif
