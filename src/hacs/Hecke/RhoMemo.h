// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_RHO_MEMO_H
#define HECKE_RHO_MEMO_H

// hecke-extensions slot 5, WP-6 -- THE rho_n MEMO AS A PIPELINE-WIDE PATTERN.
//
// Every matrix-free apply routine of the Hecke layer evaluates the same shape
// ([C, eq:heckechain], [C, eq:bdrychain], [C, eq:derivedh]):
//
//     out(sigma) += n_k * rho_n( (delta_j gamma_k) mod ell ) * c(sigma_k),
//
// where (j, gamma_k) walks a chain table (tau / tau_bd / eta) and delta_j is
// the j-th entry of the operator's OWN coset table.  The rho_n block recurs
// across sigma and across calls, so each of these routines already built a
// per-call std::map keyed by (j, gamma).  Slot-4 WP-4 hoisted exactly that
// table out of HeckeConeAction::apply_h_direct and shared it across the calls
// of a standard-basis sweep; this header is that pattern, made explicit and
// reused by HeckeOperatorOnK::apply_route2 and
// HeckeBoundaryOperator::apply_direct.
//
// WHY THE HOIST CANNOT MOVE A VALUE.  The table memoises
//
//     (j, gamma)  |-->  rho_n( (D[j] * gamma) mod ell )
//                  =  ComputeSymmetricPowerMatrixFM( reduce(D[j] * gamma, ell), n_sym ),
//
// whose result depends on EXACTLY five things: the coset table D, the index j,
// the exact integer matrix gamma, the modulus ell and the symmetric power
// n_sym.  ComputeSymmetricPowerMatrixFM is a pure function of its two
// arguments (its Pascal / power tables are memoised per (n, ell) and are
// themselves pure), Mat22::operator* is exact integer arithmetic, and
// reduce_mat22 is a pure reduction.  Nothing in the chain reads a thread-local
// setting, a device, a thread count or an evaluation order.  (j, gamma) is the
// KEY; (D, ell, n_sym) is FIXED for the lifetime of the operator object that
// owns the table -- all three are set in that object's constructor and are
// never reassigned.  bind() below turns that last sentence from an assumption
// into an O(1) checked invariant: a table is stamped with the identity of the
// object that first filled it and ENGINE_CHECKs it on every later use, so a
// table handed to the wrong operator (or to the same operator after a
// different coset table) fails loudly instead of returning a wrong block.
//
// The arithmetic CONSUMING a block is untouched: it reads the same block, in
// the same order, whether the table arrives empty or warm.  A default-
// constructed table therefore reproduces the pre-package per-call behaviour
// exactly, which is why every PUBLIC one-argument overload keeps creating a
// fresh local table.
//
// THE ONE HAZARD THIS TYPE DOES NOT COVER, STATED SO IT IS NOT FORGOTTEN.  A
// table may be shared only between calls that memoise the SAME function of the
// key.  Two routines of this layer key a DIFFERENT function on the same
// (j, gamma): HeckeBoundaryOperator::assemble_theta and
// HeckeConeAction::assemble_h both take a `fused` flag and store
//
//     fused ? rho_n(s_j^{-1}) * rho_n((g gamma) mod ell)
//           : rho_n((delta_j gamma) mod ell)
//
// -- mathematically equal, deliberately computed in two bracketings and
// asserted bit-identical, but NOT the same expression, which is the whole
// point of that assertion.  Their tables are therefore strictly CALL-LOCAL and
// are deliberately not migrated to this type: the identity stamp would not
// catch a share across the two bracketings, because owner, coset table, ell,
// n_sym and m are all the same there.  Only the three MATRIX-FREE apply
// routines -- HeckeOperatorOnK::apply_route2, HeckeBoundaryOperator::
// apply_direct and HeckeConeAction::apply_h_direct, each of which computes the
// delta_j bracketing and only that one -- take a caller-owned table.
//
// std::map, NEVER unordered_*: the memo is a result path and this codebase's
// standing audit rule is that no result path may depend on the iteration order
// of a hash container (BlockPool's hand-rolled chained vector and
// engine::kernels' std::map CSE caches are the other two instances).  The
// order here is in fact never observed -- the table is only ever point-queried
// by key -- but the rule is kept structurally rather than argued away.

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>

#include "../../ModArithmetic/EngineCheck.h"

namespace hecke {

// `Block` is whatever the consuming routine wants to hold a rho_n block in
// (FlatMatrixT<Elt> for the boundary operator and the cone action, an owned
// engine::DenseBuf<Elt> for route 2, which feeds it straight to a gemv).
template<class Block>
class RhoMemoT {
public:
    // (coset index j, gamma.a, gamma.b, gamma.c, gamma.d).
    using Key = std::array<int64_t, 5>;
    using Table = std::map<Key, Block>;

    // The part of the value's dependency set that is NOT in the key.  `owner`
    // is the operator object, `delta_table` the address of the coset table it
    // reads delta_j from (a distinct object may share a coset table, and the
    // same address may be reused by a later object -- both are covered by
    // comparing the pair together with ell / n_sym / m).
    struct Identity {
        const void* owner = nullptr;
        const void* delta_table = nullptr;
        uint64_t ell = 0;
        uint64_t n_sym = 0;
        uint64_t m = 0;
        // (slot-6 WP-1) The fibre representation the blocks are built in
        // (FibreRepresentation of LocalSystemUtils.h as its integer: 0 =
        // rho_n, 1 = rho_n^vee).  A rho_n and a rho_n^vee operator share
        // (delta table, ell, n_sym, m) and can even share an owner address
        // after a destroy/rebuild, but their (j, gamma) keys name DIFFERENT
        // blocks -- so the representation is part of the stamp and a memo
        // shared across the two is caught STRUCTURALLY by bind().  0 is the
        // default, so every pre-slot-6 identity is unchanged.
        uint64_t representation = 0;
        bool operator==(const Identity& o) const {
            return owner == o.owner && delta_table == o.delta_table && ell == o.ell
                && n_sym == o.n_sym && m == o.m && representation == o.representation;
        }
        bool operator!=(const Identity& o) const { return !(*this == o); }
    };

    // Stamp the table on first use; re-check it on every later use.  O(1).
    void bind(const Identity& id) {
        if (!bound_) { id_ = id; bound_ = true; return; }
        ENGINE_CHECK_MSG(id_ == id,
            "hecke::RhoMemoT: a rho_n memo was reused across two different operator identities "
            "((owner, coset table, ell, n_sym, m, representation)); its (j, gamma) keys do not determine the same "
            "blocks there.");
    }

    Table& table() { return table_; }
    const Table& table() const { return table_; }
    bool bound() const { return bound_; }
    std::size_t size() const { return table_.size(); }
    void clear() { table_.clear(); bound_ = false; id_ = Identity{}; }

private:
    Table table_;
    Identity id_{};
    bool bound_ = false;
};

} // namespace hecke

#endif // HECKE_RHO_MEMO_H
