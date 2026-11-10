// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef LOCAL_SYSTEM_SIMPLICIAL_COMPLEX_H
#define LOCAL_SYSTEM_SIMPLICIAL_COMPLEX_H

#include "SimplicialCohomology.h" // Includes SimplicialComplex.h
#include "../ModArithmetic/FlatMatrix.h"           // SimplexRelation::M is a FlatMatrix
#include "../ModArithmetic/helper.h"               // For offset_string

#include <vector>
#include <map>
#include <utility> // For std::pair
#include <string>
#include <sstream>
#include <stdexcept>
#include <algorithm> // For std::min, std::max
#include <cstdint>

// Stage S13: templated on the coefficient element type Elt. The default alias
// LocalSystemSimplicialComplex = LocalSystemSimplicialComplexT<uint32_t> keeps
// all existing callers (and the nested SimplexRelation type) source-compatible
// and byte-identical; LocalSystemSimplicialComplex64 supports ell < 2^64.
template<class Elt>
class LocalSystemSimplicialComplexT {
public:
    // Represents a user-defined relation: f(s_lhs_oriented_idx) = M * f(s_rhs_oriented_idx)
    // s_lhs_oriented_idx and s_rhs_oriented_idx are indices into K.get_simplices()[q_dim]
    // These are assumed to be the *canonically oriented* simplices from K.
    struct SimplexRelation {
        uint32_t q_dim;                     // Dimension of the simplices s_lhs, s_rhs
        uint32_t s_lhs_idx_in_Kq;           // Index of the LHS simplex in K's q-simplices list
        uint32_t s_rhs_idx_in_Kq;           // Index of the RHS simplex in K's q-simplices list
        // (matrix-engine Phase 3) The structural pin of the deprecated `Matrix`
        // (audit 01 Sec. 5.1): every relation matrix of every local system was a
        // MatrixT, which forced Matrix on to every producer and consumer of a
        // local system. It is a FlatMatrixT now. Producers that still hand in a
        // MatrixT keep compiling through FlatMatrixT's converting constructor,
        // and the STORED VALUES are unchanged (the conversion is entrywise).
        FlatMatrixT<Elt> M;                 // The matrix for this relation
        std::string tag;                    // Optional tag for user reference

        // Constructor for SimplexRelation
        SimplexRelation(uint32_t q, uint32_t lhs_idx, uint32_t rhs_idx, const FlatMatrixT<Elt>& matrix, std::string rel_tag = "")
            : q_dim(q), s_lhs_idx_in_Kq(lhs_idx), s_rhs_idx_in_Kq(rhs_idx), M(matrix), tag(std::move(rel_tag)) {}
    };

private:
    SimplicialComplex base_complex; // K (the "contractible" complex)
    uint32_t representation_dim;
    Elt prime;

    // Stores all user-provided (s_lhs, s_rhs, M) relations.
    std::vector<SimplexRelation> user_defined_simplex_relations;

    // (slot-11 WP-12, F-2) The declared coefficient ring Z/ell^k, or none
    // (declared_ell_ == 0).  Read only by EquivariantCohomologyOnK's ring
    // selection; never on a value path.
    uint64_t declared_ell_ = 0;
    uint32_t declared_k_ = 1;

    // Common initialization logic
    void initialize_local_system_relations(std::vector<SimplexRelation>&& explicit_relations);

public:
    // Constructor 1: From raw simplex data for K and explicit SimplexRelations
    LocalSystemSimplicialComplexT(
        const std::vector<std::vector<std::vector<uint32_t>>>& simplices_data_K,
        const std::string& complex_name_K,
        const std::vector<SimplexRelation>& explicit_relations,
        uint32_t representation_dim,
        Elt prime);

    // Constructor 2: From an existing SimplicialComplex object K (makes a copy) and explicit SimplexRelations
    LocalSystemSimplicialComplexT(
        const SimplicialComplex& existing_K,
        const std::vector<SimplexRelation>& explicit_relations,
        uint32_t representation_dim,
        Elt prime);

    // Constructor 2b (matrix-engine Phase 4, memory profile 05 Sec. 7 item 4):
    // as 2, but ADOPTING the relation list instead of copying it, so the
    // rho_n(g) matrices are held once, not twice.
    LocalSystemSimplicialComplexT(
        const SimplicialComplex& existing_K,
        std::vector<SimplexRelation>&& explicit_relations,
        uint32_t representation_dim,
        Elt prime);

    ~LocalSystemSimplicialComplexT();

    // Getters
    const SimplicialComplex* get_base_K_complex() const { return &base_complex; }
    const std::vector<SimplexRelation>& get_user_defined_simplex_relations() const {
        return user_defined_simplex_relations;
    }
    uint32_t get_representation_dim() const { return representation_dim; }
    Elt get_prime() const { return prime; }

    // (slot-11 WP-12, F-2; ENGINE_API.md Sec. 13.12.3) The ring travels as
    // (ell, k): a producer that built the relations over Z/ell^k (with
    // hecke::GenerateSL2ZRelationsRing and a ModCtx::prime_power(ell, k))
    // declares it here, and EquivariantCohomologyOnK then takes ell from the
    // declaration instead of re-deriving it from the modulus (ExactKthRoot).
    // Throws std::logic_error unless k >= 2, ell >= 2 and ell^k == get_prime()
    // exactly (overflow-checked).  The primality of ell is a PRECONDITION,
    // exactly as for ModCtx::prime_power -- the ring is declared, not inferred
    // (Sec. 13.1).  Undeclared (the default) keeps the slot-8 derivation.
    void declare_coefficient_ring(uint64_t ell, uint32_t k) {
        if (k < 2 || ell < 2)
            throw std::logic_error("LocalSystemSimplicialComplex::declare_coefficient_ring: need ell >= 2 and k >= 2");
        unsigned __int128 acc = 1;
        for (uint32_t i = 0; i < k; ++i) {
            acc *= ell;
            if (acc > static_cast<unsigned __int128>(prime))
                throw std::logic_error("LocalSystemSimplicialComplex::declare_coefficient_ring: ell^k != the modulus");
        }
        if (acc != static_cast<unsigned __int128>(prime))
            throw std::logic_error("LocalSystemSimplicialComplex::declare_coefficient_ring: ell^k != the modulus");
        declared_ell_ = ell;
        declared_k_ = k;
    }
    bool has_declared_coefficient_ring() const { return declared_ell_ != 0; }
    uint64_t declared_ring_ell() const { return declared_ell_; }   // 0 when undeclared
    uint32_t declared_ring_k() const { return declared_k_; }       // 1 when undeclared

    // String conversion
    std::string to_string() const;
    std::string to_string_offset(const uint32_t offset) const;
};

using LocalSystemSimplicialComplex = LocalSystemSimplicialComplexT<uint32_t>;
using LocalSystemSimplicialComplex64 = LocalSystemSimplicialComplexT<uint64_t>;

#endif // LOCAL_SYSTEM_SIMPLICIAL_COMPLEX_H
