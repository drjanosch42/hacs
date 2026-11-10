// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

/*
 * LocalSystemSimplicialComplex.cpp
 *
 * Implements the LocalSystemSimplicialComplexT<Elt> data container (Stage S13
 * element-width generalization). See the header for the design note. The
 * uint32_t instantiation (alias LocalSystemSimplicialComplex) is byte-identical
 * to the pre-S13 code.
 */

#include "LocalSystemSimplicialComplex.h"

#include <utility>
#include <algorithm> // For std::sort, std::unique (if needed for relations later)
#include <iostream>  // For std::cerr (optional, for verbose errors)

// Constructor 1: From raw simplex data for K
template<class Elt>
LocalSystemSimplicialComplexT<Elt>::LocalSystemSimplicialComplexT(
    const std::vector<std::vector<std::vector<uint32_t>>>& simplices_data_K,
    const std::string& complex_name_K,
    const std::vector<SimplexRelation>& explicit_relations,
    uint32_t representation_dim,
    Elt prime)
    : base_complex(simplices_data_K, complex_name_K),
    representation_dim(representation_dim),
    prime(prime)
{
    if (this->representation_dim == 0) {
        throw std::runtime_error("LocalSystemSimplicialComplex: Representation dimension must be positive.");
    }
    if (this->prime < 2) {
        throw std::runtime_error("LocalSystemSimplicialComplex: Prime modulus must be at least 2.");
    }
    initialize_local_system_relations(std::vector<SimplexRelation>(explicit_relations));
}

// Constructor 2: From an existing SimplicialComplex object K (makes a copy)
template<class Elt>
LocalSystemSimplicialComplexT<Elt>::LocalSystemSimplicialComplexT(
    const SimplicialComplex& existing_K,
    const std::vector<SimplexRelation>& explicit_relations,
    uint32_t representation_dim,
    Elt prime)
    : base_complex(existing_K),
    representation_dim(representation_dim),
    prime(prime)
{
    if (this->representation_dim == 0) {
        throw std::runtime_error("LocalSystemSimplicialComplex: Representation dimension must be positive.");
    }
    if (this->prime < 2) {
        throw std::runtime_error("LocalSystemSimplicialComplex: Prime modulus must be at least 2.");
    }
    initialize_local_system_relations(std::vector<SimplexRelation>(explicit_relations));
}

template<class Elt>
LocalSystemSimplicialComplexT<Elt>::LocalSystemSimplicialComplexT(
    const SimplicialComplex& existing_K,
    std::vector<SimplexRelation>&& explicit_relations,
    uint32_t representation_dim,
    Elt prime)
    : base_complex(existing_K),
    representation_dim(representation_dim),
    prime(prime)
{
    if (this->representation_dim == 0) {
        throw std::runtime_error("LocalSystemSimplicialComplex: Representation dimension must be positive.");
    }
    if (this->prime < 2) {
        throw std::runtime_error("LocalSystemSimplicialComplex: Prime modulus must be at least 2.");
    }
    initialize_local_system_relations(std::move(explicit_relations));
}

// Common initialization logic for relations
template<class Elt>
void LocalSystemSimplicialComplexT<Elt>::initialize_local_system_relations(
    std::vector<SimplexRelation>&& explicit_relations)
{
    this->user_defined_simplex_relations.clear();
    const auto& K_simplices_by_dim = this->base_complex.get_simplices();

    for (auto& rel : explicit_relations) {
        // 1. Validate q_dim
        if (rel.q_dim > this->base_complex.get_dimension()) {
            std::ostringstream oss;
            oss << "LSSC Init (Relation Tag: " << rel.tag << "): SimplexRelation q_dim (" << rel.q_dim
                << ") exceeds base complex K dimension (" << this->base_complex.get_dimension() << ").";
            throw std::runtime_error(oss.str());
        }
        if (rel.q_dim >= K_simplices_by_dim.size() && this->base_complex.get_dimension() != static_cast<uint32_t>(-1) /*not empty complex*/) {
            std::ostringstream oss;
            oss << "LSSC Init (Relation Tag: " << rel.tag << "): SimplexRelation q_dim (" << rel.q_dim
                << ") is invalid as base_complex K has no simplices at this dimension (max_dim_K_data="
                << K_simplices_by_dim.size() - 1 << ").";
            throw std::runtime_error(oss.str());
        }

        // 2. Validate s_lhs_idx_in_Kq and s_rhs_idx_in_Kq
        if (rel.q_dim < K_simplices_by_dim.size()) {
            const auto& q_simplices_K = K_simplices_by_dim[rel.q_dim];
            if (rel.s_lhs_idx_in_Kq >= q_simplices_K.size()) {
                std::ostringstream oss;
                oss << "LSSC Init (Relation Tag: " << rel.tag << ", q=" << rel.q_dim
                    << "): SimplexRelation s_lhs_idx_in_Kq (" << rel.s_lhs_idx_in_Kq
                    << ") is out of bounds for " << q_simplices_K.size() << " simplices.";
                throw std::runtime_error(oss.str());
            }
            if (rel.s_rhs_idx_in_Kq >= q_simplices_K.size()) {
                std::ostringstream oss;
                oss << "LSSC Init (Relation Tag: " << rel.tag << ", q=" << rel.q_dim
                    << "): SimplexRelation s_rhs_idx_in_Kq (" << rel.s_rhs_idx_in_Kq
                    << ") is out of bounds for " << q_simplices_K.size() << " simplices.";
                throw std::runtime_error(oss.str());
            }
        }
        else if (this->base_complex.get_num_q_simplices(rel.q_dim) == 0) {
            if (rel.s_lhs_idx_in_Kq != 0 || rel.s_rhs_idx_in_Kq != 0 || this->base_complex.get_num_q_simplices(rel.q_dim) > 0) {
                std::ostringstream oss;
                oss << "LSSC Init (Relation Tag: " << rel.tag << ", q=" << rel.q_dim
                    << "): SimplexRelation indices are non-zero but K has no simplices at this dimension.";
                throw std::runtime_error(oss.str());
            }
        }


        // 3. Validate Matrix M
        if (rel.M.get_rows() != this->representation_dim || rel.M.get_cols() != this->representation_dim) {
            std::ostringstream oss;
            oss << "LSSC Init (Relation Tag: " << rel.tag << ", q=" << rel.q_dim
                << "): SimplexRelation Matrix M (" << rel.M.get_rows() << "x" << rel.M.get_cols()
                << ") has incorrect dimensions for representation_dim=" << this->representation_dim << ".";
            throw std::runtime_error(oss.str());
        }
        if (rel.M.get_p() != this->prime) {
            std::ostringstream oss;
            oss << "LSSC Init (Relation Tag: " << rel.tag << ", q=" << rel.q_dim
                << "): SimplexRelation Matrix M has prime " << rel.M.get_p()
                << " but LSSC expects prime " << this->prime << ".";
            throw std::runtime_error(oss.str());
        }

        this->user_defined_simplex_relations.push_back(std::move(rel));
    }
}

// Destructor Implementation
template<class Elt>
LocalSystemSimplicialComplexT<Elt>::~LocalSystemSimplicialComplexT() {
}

// to_string Implementation
template<class Elt>
std::string LocalSystemSimplicialComplexT<Elt>::to_string() const {
    return to_string_offset(0);
}

// to_string_offset Implementation
template<class Elt>
std::string LocalSystemSimplicialComplexT<Elt>::to_string_offset(const uint32_t offset) const {
    std::ostringstream oss;
    std::string indent = offset_string(offset);
    std::string indent2 = offset_string(offset + 2);
    std::string indent4 = offset_string(offset + 4);

    oss << indent << "Local System Simplicial Complex (LSSC):\n";
    oss << indent2 << "Representation Dimension (d_V): " << representation_dim << "\n";
    oss << indent2 << "Prime Modulus (p): " << prime << "\n";

    oss << indent2 << "Base Simplicial Complex K (name: " << base_complex.get_name() << "):\n"
        << base_complex.to_string_offset(offset + 4);

    oss << indent2 << "User-Defined Simplex Relations (f(s_lhs_idx) = M * f(s_rhs_idx)):\n";
    if (user_defined_simplex_relations.empty()) {
        oss << indent4 << "(none)\n";
    }
    else {
        for (const auto& rel : user_defined_simplex_relations) {
            oss << indent4 << "Tag: \"" << rel.tag << "\", Dim q=" << rel.q_dim << ":\n";
            oss << indent4 << "  f(K[" << rel.q_dim << "][" << rel.s_lhs_idx_in_Kq << "]) = M * f(K["
                << rel.q_dim << "][" << rel.s_rhs_idx_in_Kq << "])\n";
            oss << indent4 << "  Matrix M (" << rel.M.get_rows() << "x" << rel.M.get_cols() << ", p=" << rel.M.get_p() << "):\n"
                << rel.M.to_string_offset(offset + 6);
        }
    }
    return oss.str();
}

// Stage S13 explicit instantiations.
template class LocalSystemSimplicialComplexT<uint32_t>;
template class LocalSystemSimplicialComplexT<uint64_t>;
