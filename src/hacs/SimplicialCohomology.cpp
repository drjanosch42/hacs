// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

/*
 * SimplicialCohomology.cpp
 *
 * This file implements the core data structures and algorithms for computing the
 * simplicial cohomology of a given topological space. It provides classes for
 * representing simplicial complexes, their corresponding cochain complexes, and
 * morphisms between them.
 *
 * Class Implementations:
 * - SimplicialComplex: Represents a topological space as a collection of
 *   simplices organized by dimension. It is responsible for storing the combinatorial
 *   structure and can generate the coboundary operator matrices for a given prime p,
 *   which form the basis of the cochain complex. It can also pre-compute and cache
 *   bases for trivial cocycles (Z_1).
 *
 * - CochainComplex: Encapsulates the full pipeline for computing cohomology with
 *   trivial F_p coefficients. It takes a SimplicialComplex, builds the sequence
 *   of coboundary operators, and then systematically computes:
 *   1. Bases for k-cocycles (Z^k = ker d^k).
 *   2. Bases for k-coboundaries (B^k = im d^{k-1}).
 *   3. Bases for the k-th cohomology group (H^k = Z^k / B^k) by using the
 *      `factor_basis` function from the Matrix class.
 *   4. The dimensions of the cohomology groups (Betti numbers).
 *
 * - CochainComplexMorphism: Represents a map between two cochain complexes,
 *   typically induced by a map of the underlying simplicial complexes. Its primary
 *   function is to compute the induced maps on the level of cohomology. It also
 *   includes advanced functionality for constructing the mapping cone (`ConeComplex`)
 *   and the associated canonical maps, which are essential tools for verifying
*    the long exact sequence in cohomology and other theoretical results.
 *
 * The file also includes factory functions for creating standard example spaces
 * (like spheres and projective planes) and a validation function to test the
 * long exact sequence of a pair (X, A), demonstrating the correctness of the
 * implemented homological algebra.
 */

#include "SimplicialCohomology.h"
#include "EngineCheck.h" // ENGINE_CHECK / ENGINE_DEBUG_CHECK (NDEBUG policy)

// (matrix-engine Phase 3 Stage B) The trivial-coefficient layer's remaining
// hand-rolled arithmetic goes through the engine (ENGINE_API.md Sec. 1-3).
// Everything else here already runs on retrofitted FlatMatrix methods --
// kernel() / image() / operator* / change_of_basis_fm / factor_basis_block --
// so this file has no elimination of its own to migrate.
#include "../ModArithmetic/ModCtx.h"
#include "../ModArithmetic/Kernels.h"
#include "../ModArithmetic/DenseElim.h"   // (S24 CL-L2) engine::elim::complement_basis
#include <tuple>

// (matrix-engine Phase 3) The trivial-coefficient layer stores FlatMatrix now
// (see the header). This is the FlatMatrix counterpart of Matrix.h's deprecated
// `basis_to_string`, used only by the to_string() dumps below.
static std::string flat_basis_to_string(const std::vector<FlatMatrix>& basis, const uint32_t offset) {
    std::ostringstream oss;
    if (basis.empty()) {
        oss << offset_string(offset) << "empty\n";
    }
    for (uint32_t i = 0; i < basis.size(); i++) {
        oss << basis[i].to_string_transposed_offset(offset);
    }
    return oss.str();
}


// Helper function to convert a simplex to a string
std::string simplex_to_string(const std::vector<uint32_t>& simplex) {
    std::ostringstream oss;
    oss << "(";
    for (uint32_t i = 0; i < simplex.size(); ++i) {
        oss << simplex[i];
        if (i < simplex.size() - 1) {
            oss << ",";
        }
    }
    oss << ")";
    return oss.str();
}

// --- SimplicialComplex Implementation ---

// Default constructor for SimplicialComplex
SimplicialComplex::SimplicialComplex()
    : num_vertices(0), dim(static_cast<uint32_t>(-1)), euler_characteristic(0), name("Default_Empty_Complex"), trivial_cocycle_bases_computed(false) {
    // simplices will be an empty vector of vectors of vectors
    // trivial_cocycle_bases_Z1 will be empty
}

SimplicialComplex::SimplicialComplex(const std::vector<std::vector<std::vector<uint32_t>>>& simplices, const std::string name)
    : simplices(simplices), name(name), trivial_cocycle_bases_computed(false) { // Initialize new flag
    num_vertices = (simplices.size() > 0 && simplices[0].size() > 0) ? simplices[0].size() : 0; // Assuming 0-simplices define num_vertices
                                                                                              // More robust: find max vertex index across all simplices
    if (simplices.empty()) {
        dim = -1; // Or some indicator of empty complex
        num_vertices = 0;
    }
    else {
        dim = static_cast<uint32_t>(simplices.size() - 1);
        // Calculate num_vertices more robustly
        uint32_t max_v_idx = 0;
        bool found_vertex = false;
        for (const auto& dim_simplices_list : simplices) {
            for (const auto& s : dim_simplices_list) {
                for (uint32_t v_idx : s) {
                    if (v_idx > max_v_idx) max_v_idx = v_idx;
                    found_vertex = true;
                }
            }
        }
        num_vertices = found_vertex ? (max_v_idx + 1) : 0;
    }

    euler_characteristic = 0;
    for (size_t i = 0; i < simplices.size(); i++) {
        euler_characteristic += (i % 2 == 0) ? static_cast<int32_t>(simplices[i].size()) : -static_cast<int32_t>(simplices[i].size());
    }
    validate_simplices();
}

// Move-aware constructor for CochainComplex
CochainComplex::CochainComplex(std::vector<FlatMatrix>&& coboundary_operators_moved, uint32_t prime, const std::string name)
    : simplicial_complex(nullptr),
    coboundary_operators(std::move(coboundary_operators_moved)), // Move the vector in
    cochain_bases(),
    coboundary_bases(),
    cohomology_bases(),
    cohomology_dimensions(),
    p(prime),
    name(name)
{
    // The constructor takes ownership efficiently without copies.
    compute_cohomology();
}

SimplicialComplex::~SimplicialComplex() {}

void SimplicialComplex::validate_simplices() {
    for (const auto& dim_simplices : simplices) {
        for (const auto& simplex : dim_simplices) {
            if (!std::is_sorted(simplex.begin(), simplex.end())) {
                throw std::runtime_error("Each simplex must be given in strictly increasing order");
            }
        }
    }
}

// (matrix-engine Phase 3; audit 02 Sec. 7.17) --------------------------------
// Ordered simplex index. Built once per dimension on first use and cached; the
// simplex lists are immutable after construction, so no invalidation hook is
// needed. `emplace` keeps the FIRST insertion for a duplicated key, which is
// exactly `std::find`'s first-match-wins semantics -- that is what makes the
// three call-site rewrites value-preserving.
const std::map<std::vector<uint32_t>, uint32_t>& SimplicialComplex::simplex_index(uint32_t q) const {
    ENGINE_CHECK_MSG(q < simplices.size(), "SimplicialComplex::simplex_index: q out of range");
    if (simplex_index_.size() != simplices.size()) {
        simplex_index_.assign(simplices.size(), std::map<std::vector<uint32_t>, uint32_t>());
        simplex_index_built_.assign(simplices.size(), 0);
    }
    if (!simplex_index_built_[q]) {
        const auto& list = simplices[q];
        auto& idx = simplex_index_[q];
        for (uint32_t i = 0; i < list.size(); ++i) {
            idx.emplace(list[i], i); // first occurrence wins, as std::find does
        }
        simplex_index_built_[q] = 1;
    }
    return simplex_index_[q];
}

int64_t SimplicialComplex::find_simplex_index(uint32_t q, const std::vector<uint32_t>& sorted_vertices) const {
    if (q >= simplices.size()) return -1;
    const auto& idx = simplex_index(q);
    auto it = idx.find(sorted_vertices);
    if (it == idx.end()) return -1;
    return static_cast<int64_t>(it->second);
}

std::string SimplicialComplex::to_string() const {
    return to_string_offset(0);
}

std::string SimplicialComplex::to_string_offset(const uint32_t offset) const {
    std::ostringstream oss;
    oss << offset_string(offset);
    oss << "Simplicial Complex:\n";
    oss << offset_string(offset + 2);
    oss << "Name: " << name << "\n";
    oss << offset_string(offset + 2);
    oss << "Number of vertices: " << num_vertices << "\n";
    oss << offset_string(offset + 2);
    oss << "Dimension: " << dim << "\n";
    oss << offset_string(offset + 2);
    oss << "Euler Characteristic: " << euler_characteristic << "\n";
    oss << offset_string(offset + 2);
    oss << "Simplices:\n";
    for (uint32_t d = 0; d < simplices.size(); ++d) {
        oss << offset_string(offset + 4);
        oss << "dim " << d << ": ";
        for (const auto& simplex : simplices[d]) {
            oss << simplex_to_string(simplex) << " ";
        }
        oss << "\n";
    }
    return oss.str();
}

// compute F_p-linear coboundary operator
// Stage S13: the uint32_t method delegates to the element-width-generic
// implementation below (single source of truth; <uint32_t> is byte-identical).
// (matrix-engine Phase 0) The non-template `coboundary_operator_block` wrapper
// that used to follow was dead code -- zero callers; consumers go straight to
// coboundary_operator_block_elt<Elt>. Deleted, declaration included.
FlatMatrix SimplicialComplex::coboundary_operator(uint32_t q, uint32_t p) const {
    return coboundary_operator_elt<uint32_t>(q, p);
}

template<class Elt>
FlatMatrixT<Elt> SimplicialComplex::coboundary_operator_elt(uint32_t q, Elt p) const {
    if (q > dim) {
        return FlatMatrixT<Elt>::CreateZero(0, 0, p); // Empty matrix for out-of-range q
    }

    uint32_t source_dim = simplices[q].size();
    uint32_t target_dim = (q < dim) ? simplices[q + 1].size() : 0;

    if (source_dim == 0 && target_dim == 0) {
        return FlatMatrixT<Elt>::CreateZero(0, 0, p);
    }
    else if (source_dim == 0) {
        return FlatMatrixT<Elt>::CreateZero(target_dim, 0, p);
    }
    else if (target_dim == 0) {
        return FlatMatrixT<Elt>::CreateZero(0, source_dim, p);
    }

    // (matrix-engine Phase 3) Same entries as the vector<vector> build this
    // replaced, laid out row-major straight into the FlatMatrix buffer.
    std::vector<Elt> coboundary_data(static_cast<size_t>(target_dim) * source_dim, 0);

    for (uint32_t j = 0; j < target_dim; ++j) {
        const auto& target_simplex = simplices[q + 1][j];
        for (uint32_t i = 0; i < target_simplex.size(); ++i) {
            std::vector<uint32_t> source_simplex = target_simplex;
            source_simplex.erase(source_simplex.begin() + i);

            // Find index of source simplex in the previous dimension.
            // (matrix-engine Phase 3, 02 Sec. 7.17) O(log N) ordered-map lookup
            // instead of the O(N) std::find this replaces; same index, because
            // find_simplex_index is first-match-wins. `source_simplex` inherits
            // ascending order from `target_simplex`.
            const int64_t k_signed = find_simplex_index(q, source_simplex);
            if (k_signed >= 0) {
                uint32_t k = static_cast<uint32_t>(k_signed);
                coboundary_data[static_cast<size_t>(j) * source_dim + k] =
                    (i % 2 == 0) ? Elt(1) : (p - Elt(1)); // (-1)^i mod p
            }
        }
    }

    return FlatMatrixT<Elt>(target_dim, source_dim, p, coboundary_data);
}

template FlatMatrixT<uint32_t> SimplicialComplex::coboundary_operator_elt<uint32_t>(uint32_t, uint32_t) const;
template FlatMatrixT<uint64_t> SimplicialComplex::coboundary_operator_elt<uint64_t>(uint32_t, uint64_t) const;
#if OC_ENABLE_LEGACY
#include "legacy/SimplicialCohomologyLegacy.inc"   // (S24 CL-L2) coboundary_operator_block_elt + its instantiations
#endif

// --- CochainComplex Implementation ---

CochainComplex::CochainComplex(const SimplicialComplex* simplicial_complex, uint32_t prime)
    : simplicial_complex(simplicial_complex),
    coboundary_operators(),
    cochain_bases(),
    coboundary_bases(),
    cohomology_bases(),
    cohomology_dimensions(),
    p(prime),
    name(simplicial_complex->get_name())
{
    // Initialize coboundary operators
    compute_coboundary_operators();
    // Compute cohomology
    compute_cohomology();
}

CochainComplex::CochainComplex(const std::vector<FlatMatrix>* coboundary_operators, uint32_t prime, const std::string name)
    : coboundary_operators(*coboundary_operators),
    cochain_bases(),
    coboundary_bases(),
    cohomology_bases(),
    cohomology_dimensions(),
    p(prime), name(name)
{
    // Compute cohomology
    compute_cohomology();
}

// Second constructor for CochainComplexMorphism using a vector of matrices as input.
CochainComplexMorphism::CochainComplexMorphism(const int32_t degree, const std::vector<FlatMatrix>* operators,
    CochainComplex* domain_cochain_complex,
    CochainComplex* codomain_cochain_complex)
    : simplicial_morphism(nullptr), // No simplicial morphism in this case
    cochain_complex_morphism(nullptr),
    domain_cochain_complex(domain_cochain_complex),
    codomain_cochain_complex(codomain_cochain_complex),
    cochain_maps(*operators), // Directly assign the provided cochain maps
    cohomology_maps(), // Initialize cohomology maps as empty
    degree(degree)
{
    // Compute cohomology maps based on the given cochain maps.
    compute_cohomology_maps();
}


CochainComplex::~CochainComplex() {}

void CochainComplex::compute_coboundary_operators() {
    coboundary_operators.clear();
    for (uint32_t q = 0; q <= simplicial_complex->dim; ++q) {
        coboundary_operators.push_back(simplicial_complex->coboundary_operator(q, p));
    }
}

void CochainComplex::compute_cochains() {
    cochain_bases.clear();
    // this for loop contains all edge cases for cochains
    for (uint32_t q = 0; q < coboundary_operators.size(); ++q) {
        cochain_bases.push_back(coboundary_operators[q].kernel());
    }
}

void CochainComplex::compute_coboundaries() {
    coboundary_bases.clear();
    // this for loop does contain all edge cases for cochains
    for (uint32_t q = 0; q < coboundary_operators.size(); ++q) {
        if (q == 0) {
            coboundary_bases.push_back(std::vector<FlatMatrix>()); // Image of d_0 is 0
        }
        else {
            coboundary_bases.push_back(coboundary_operators[q - 1].image()); // Use .image() here
        }
    }
}

/*void CochainComplex::compute_cohomology_bases() {
    cohomology_bases.clear();
    cohomology_base_indices.clear();
    // this for loop does contain all edge cases for cochains
    for (uint32_t q = 0; q < coboundary_operators.size(); ++q) {
        //std::cout << name << ":\n";
        //std::cout << "computing cohomology in degree " << q << "\n";
        //std::cout << "  coboundary_bases[q].size() = " << coboundary_bases[q].size() << "\n";
        //std::cout << basis_to_string(coboundary_bases[q], 4);
        //std::cout << "  cochain_bases[q].size() = " << cochain_bases[q].size() << "\n";
        //std::cout << basis_to_string(cochain_bases[q], 4);
        std::vector<uint32_t> indices = {};
        cohomology_bases.push_back({});
        cohomology_projectors.push_back(factor_basis(coboundary_bases[q], cochain_bases[q], indices, &(cohomology_bases[q])));
        cohomology_base_indices.push_back(indices);
        //cohomology_projectors.push_back(projector);
        //std::cout << "cohomology in degree " << q << " computed.\n";
    }
}*/

void CochainComplex::compute_cohomology_bases() {
    cohomology_bases.clear();
    cohomology_base_indices.clear();
    cohomology_projectors.clear(); // Clear before populating

    for (uint32_t q = 0; q < coboundary_operators.size(); ++q) {
        // cochain_bases[q] is a basis for the cocycles Z^q.
        // coboundary_bases[q] is a basis for the coboundaries B^q.
        // Both are currently expressed in the ambient basis of C^q.
        const auto& Z_q_basis = cochain_bases[q];
        const auto& B_q_basis = coboundary_bases[q];

        if (Z_q_basis.empty()) {
            // If Z^q is the zero space, then H^q is also zero.
            cohomology_bases.push_back({});
            cohomology_base_indices.push_back({});
            cohomology_projectors.push_back(FlatMatrix::CreateZero(0, 0, p)); // Projector from {0} to {0}
            continue;
        }

        // We must express the relations (B^q) in terms of the generators (Z^q).
        // Since B^q is a subspace of Z^q, this is always possible.
        std::vector<FlatMatrix> B_q_in_Z_q_basis;
        if (!B_q_basis.empty()) {
            // Use change_of_basis to get the coefficient vectors.  (Phase 3:
            // change_of_basis_fm builds the SAME augmented matrix, reads the
            // SAME canonical RREF and applies the SAME pivot rule as the
            // deprecated Matrix overload it replaces.)
            auto B_q_coeffs_vecs = change_of_basis_fm(Z_q_basis, B_q_basis);

            // Convert each coefficient vector into a column matrix.
            for (const auto& coeff_vec : B_q_coeffs_vecs) {
                B_q_in_Z_q_basis.push_back(
                    coeff_vec.empty()
                        ? FlatMatrix::CreateZero(static_cast<uint32_t>(coeff_vec.size()), 1, p)
                        : FlatMatrix(static_cast<uint32_t>(coeff_vec.size()), 1, p, coeff_vec));
            }
        }

        // Now, call factor_basis with compatible inputs.
        // The generators are the basis vectors of Z^q, which we can represent
        // by the standard basis vectors of the coordinate space R^dim(Z^q).
        std::vector<FlatMatrix> Z_q_standard_basis;
        const uint32_t dim_Z_q = static_cast<uint32_t>(Z_q_basis.size());
        for (uint32_t i = 0; i < dim_Z_q; ++i) {
            std::vector<uint32_t> e_i(dim_Z_q, 0);
            e_i[i] = 1;
            Z_q_standard_basis.push_back(FlatMatrix(dim_Z_q, 1, p, e_i));
        }

        std::vector<FlatMatrix> H_q_coeff_basis; // This will be a basis of H^q in Z^q coordinates.

        // (matrix-engine Phase 3) factor_basis_block with ONE ROW PER AMBIENT
        // COORDINATE (block_row_dims = {1, ..., 1}) is the flat factor_basis:
        // it builds the same augmented [relations | generators | I], reads the
        // same canonical RREF, classifies pivots by the same rule and extracts
        // the projector from the same identity region.
        // (S24 CL-L2) factor_basis_block lives in the legacy BlockFlatMatrix.cpp, so the default route is
        // engine::elim::complement_basis -- factor_basis_block's rule on the same augmented [Rl | G | I]
        // (DenseElim.h; pinned entrywise against factor_basis_block, pivots and projector, by
        // DenseElimValidation) -- with factor_basis_block's return conventions; under engine::elim::use_legacy()
        // (ON builds) the factor_basis_block call itself runs (legacy/SimplicialCohomologyLegacy.inc).
        FlatMatrix projector;
        std::vector<uint32_t> H_q_indices;
#if OC_ENABLE_LEGACY
        if (engine::elim::use_legacy()) {
            std::tie(projector, H_q_indices) =
                legacy_trivial_factor_basis(dim_Z_q, B_q_in_Z_q_basis, Z_q_standard_basis, &H_q_coeff_basis, p);
        } else
#endif
        {
            const engine::ModCtx<uint32_t> ctx(p);
            const uint32_t n_rel = static_cast<uint32_t>(B_q_in_Z_q_basis.size());
            engine::DenseBuf<uint32_t> Rl(dim_Z_q, n_rel), G(dim_Z_q, dim_Z_q), proj;
            for (uint32_t j = 0; j < n_rel; ++j)
                for (uint32_t r = 0; r < dim_Z_q; ++r) Rl.at(r, j) = B_q_in_Z_q_basis[j].get_entry(r, 0);
            for (uint32_t i = 0; i < dim_Z_q; ++i) G.at(i, i) = 1u;   // the generators e_i
            H_q_indices = engine::elim::complement_basis<uint32_t>(ctx, engine::cview(Rl), engine::cview(G), &proj);
            for (uint32_t gen_idx : H_q_indices) H_q_coeff_basis.push_back(Z_q_standard_basis[gen_idx]);
            const uint32_t dim_H = static_cast<uint32_t>(H_q_indices.size());
            bool all_zero = true;
            std::vector<uint32_t> projector_buf(static_cast<size_t>(dim_H) * dim_Z_q);
            for (uint32_t i = 0; i < dim_H; ++i)
                for (uint32_t j = 0; j < dim_Z_q; ++j) {
                    const uint32_t v = proj.at(i, j);
                    projector_buf[static_cast<size_t>(i) * dim_Z_q + j] = v;
                    if (v != 0u) all_zero = false;
                }
            projector = (dim_H == 0 || all_zero) ? FlatMatrix::CreateZero(dim_H, dim_Z_q, p)
                                                 : FlatMatrix(dim_H, dim_Z_q, p, projector_buf);
        }

        cohomology_projectors.push_back(projector);
        cohomology_base_indices.push_back(H_q_indices);

        // The final basis for H^q must be expanded back into the ambient C^q space.
        // `H_q_coeff_basis` gives the coefficients with respect to `Z_q_basis`.
        // (matrix-engine Phase 3 Stage B) ONE gemm instead of the
        // dim_H x dim_Z x ambient scalar `mod_mul`/`mod_add` triple loop:
        // the expansion is exactly  Z * Coeff, with Z the (ambient x dim_Z)
        // matrix whose columns are the Z^q basis vectors and Coeff the
        // (dim_Z x dim_H) matrix of coefficient columns. Bit-identical: every
        // term is an exact product of canonical residues and the engine
        // reduces each k-block once (ENGINE_API.md Sec. 5), so neither the
        // `coeff == 0` skip nor the summation order of the old loop can move a
        // bit. Zero-dimension cases keep the old shapes exactly.
        std::vector<FlatMatrix> H_q_ambient_basis;
        const uint32_t ambient_rows = Z_q_basis[0].get_rows(); // Assuming Z_q not empty
        const uint32_t dim_H_q = static_cast<uint32_t>(H_q_coeff_basis.size());
        if (dim_H_q > 0 && ambient_rows > 0) {
            const engine::ModCtx<uint32_t> ctx(p);
            engine::DenseBuf<uint32_t> Zb(ambient_rows, dim_Z_q);   // ambient x dim_Z
            for (uint32_t i = 0; i < dim_Z_q; ++i)
                for (uint32_t j = 0; j < ambient_rows; ++j)
                    Zb.at(j, i) = Z_q_basis[i].get_entry(j, 0);
            engine::DenseBuf<uint32_t> Cf(dim_Z_q, dim_H_q);        // dim_Z x dim_H
            for (uint32_t t = 0; t < dim_H_q; ++t) {
                const FlatMatrix& h_coeffs = H_q_coeff_basis[t];
                ENGINE_CHECK(h_coeffs.get_rows() == dim_Z_q);
                for (uint32_t i = 0; i < dim_Z_q; ++i) Cf.at(i, t) = h_coeffs.get_entry(i, 0);
            }
            engine::DenseBuf<uint32_t> Out(ambient_rows, dim_H_q);
            if (dim_Z_q > 0) {
                engine::kernels::gemm<uint32_t>(ctx, engine::cview(Zb), engine::cview(Cf),
                                                engine::view(Out), engine::kernels::GemmOp::Set);
            }
            for (uint32_t t = 0; t < dim_H_q; ++t) {
                std::vector<uint32_t> ambient_buf(ambient_rows, 0);
                for (uint32_t j = 0; j < ambient_rows; ++j) ambient_buf[j] = Out.at(j, t);
                H_q_ambient_basis.push_back(FlatMatrix(ambient_rows, 1, p, ambient_buf));
            }
        } else {
            for (uint32_t t = 0; t < dim_H_q; ++t) {
                H_q_ambient_basis.push_back(FlatMatrix::CreateZero(0, 1, p));
            }
        }
        cohomology_bases.push_back(H_q_ambient_basis);
    }
}

void CochainComplex::compute_cohomology_dimensions() {
    cohomology_dimensions.clear();
    euler_characteristic = 0;
    for (uint32_t q = 0; q < coboundary_operators.size(); ++q) {
        //uint32_t kernel_dim = cochain_bases[q].size();
        //uint32_t image_dim = coboundary_bases[q].size();
        int32_t coh_dim = cohomology_bases[q].size();
        cohomology_dimensions.push_back(coh_dim);
        euler_characteristic += (q % 2 == 0) ? coh_dim : -coh_dim;
    }
}

void CochainComplex::compute_cohomology() {
    //std::cout << name << ":\n";
    //std::cout << "computing cochains\n";
    compute_cochains();
    //std::cout << "computing coboundaries\n";
    compute_coboundaries();
    //std::cout << "computing cohomology bases\n";
    compute_cohomology_bases();
    //std::cout << "computing cohomology dimensions\n";
    compute_cohomology_dimensions();
    //std::cout << "cohomology computed\n";
}

// --- NEW METHODS for trivial cocycle bases ---
// Stage S13: the uint32_t methods delegate to the element-width-generic
// implementations (single source of truth; <uint32_t> is byte-identical).
void SimplicialComplex::compute_trivial_cocycle_bases(uint32_t prime_for_coboundary) {
    compute_trivial_cocycle_bases_elt<uint32_t>(prime_for_coboundary);
}

const std::vector<FlatMatrix>& SimplicialComplex::get_trivial_cocycle_basis_Z1(uint32_t q) const {
    return get_trivial_cocycle_basis_Z1_elt<uint32_t>(q);
}

template<class Elt>
void SimplicialComplex::compute_trivial_cocycle_bases_elt(Elt prime_for_coboundary) {
    // Select the per-width cache (SimplicialComplex is combinatorial; only these
    // value caches are element-typed).
    auto& cache = [this]() -> std::vector<std::vector<FlatMatrixT<Elt>>>& {
        if constexpr (std::is_same_v<Elt, uint32_t>) return trivial_cocycle_bases_Z1;
        else return trivial_cocycle_bases_Z1_64;
    }();
    bool& computed = [this]() -> bool& {
        if constexpr (std::is_same_v<Elt, uint32_t>) return trivial_cocycle_bases_computed;
        else return trivial_cocycle_bases_computed_64;
    }();

    if (computed) return;

    cache.clear();
    // Coboundary D^q goes from C^q to C^{q+1}. Kernel of D^q gives Z_1^q.
    for (uint32_t q = 0; q <= this->dim; ++q) {
        if (q < simplices.size() && simplices[q].empty() && (q + 1 >= simplices.size() || simplices[q + 1].empty())) {
            if (simplices[q].empty()) { // C^q = 0
                cache.push_back({}); // Empty basis for Z_1^q as C^q itself is 0
            }
            else {
                FlatMatrixT<Elt> D_q = coboundary_operator_elt<Elt>(q, prime_for_coboundary);
                cache.push_back(D_q.kernel());
            }
        }
        else if (q < simplices.size() && simplices[q].empty()) {
            cache.push_back({});
        }
        else {
            FlatMatrixT<Elt> D_q = coboundary_operator_elt<Elt>(q, prime_for_coboundary);
            cache.push_back(D_q.kernel());
        }
    }

    computed = true;
}

template<class Elt>
const std::vector<FlatMatrixT<Elt>>& SimplicialComplex::get_trivial_cocycle_basis_Z1_elt(uint32_t q) const {
    const auto& cache = [this]() -> const std::vector<std::vector<FlatMatrixT<Elt>>>& {
        if constexpr (std::is_same_v<Elt, uint32_t>) return trivial_cocycle_bases_Z1;
        else return trivial_cocycle_bases_Z1_64;
    }();
    bool computed = [this]() -> bool {
        if constexpr (std::is_same_v<Elt, uint32_t>) return trivial_cocycle_bases_computed;
        else return trivial_cocycle_bases_computed_64;
    }();
    if (!computed) {
        throw std::runtime_error("SimplicialComplex: Trivial cocycle bases (Z1) not yet computed. Call compute_trivial_cocycle_bases() first with the desired prime.");
    }
    if (q >= cache.size()) {
        std::ostringstream oss;
        oss << "SimplicialComplex: Requested trivial cocycle basis Z1 for q=" << q
            << " is out of bounds. Max q is " << cache.size() - 1 << ".";
        throw std::out_of_range(oss.str());
    }
    return cache[q];
}

template void SimplicialComplex::compute_trivial_cocycle_bases_elt<uint32_t>(uint32_t);
template void SimplicialComplex::compute_trivial_cocycle_bases_elt<uint64_t>(uint64_t);
template const std::vector<FlatMatrixT<uint32_t>>& SimplicialComplex::get_trivial_cocycle_basis_Z1_elt<uint32_t>(uint32_t) const;
template const std::vector<FlatMatrixT<uint64_t>>& SimplicialComplex::get_trivial_cocycle_basis_Z1_elt<uint64_t>(uint32_t) const;

uint32_t SimplicialComplex::get_trivial_cocycle_basis_Z1_dim(uint32_t q) const {
    // Ensures bases are computed before accessing size
    return static_cast<uint32_t>(get_trivial_cocycle_basis_Z1(q).size());
}

std::string CochainComplex::to_string() const {
    return to_string_offset(0);
}

std::string CochainComplex::to_string_offset(const uint32_t offset) const {
    std::ostringstream oss;
    oss << offset_string(offset);
    oss << "Cochain Complex:\n";
    oss << offset_string(offset + 2);
    oss << "Name: " << name << "\n";
    oss << offset_string(offset + 2);
    oss << "Coeffients: F_" << p << "\n";
    oss << offset_string(offset + 2);
    oss << "Euler Characteristic: " << euler_characteristic << "\n";

    for (uint32_t q = 0; q < coboundary_operators.size(); ++q) {
        oss << offset_string(offset + 2);
        oss << "Cohomology in degree " << q << ":\n";
        oss << offset_string(offset + 4);
        oss << q << "-Betti number: " << cohomology_dimensions[q] << "\n";

        oss << offset_string(offset + 4);
        oss << "complex dimension: " << coboundary_operators[q].get_cols() << "\n";

        oss << offset_string(offset + 4);
        oss << q << "-cochain basis:\n";
        oss << flat_basis_to_string(cochain_bases[q], offset + 6);

        oss << offset_string(offset + 4);
        oss << q << "-coboundary basis:\n";
        oss << flat_basis_to_string(coboundary_bases[q], offset + 6);

        oss << offset_string(offset + 4);
        oss << q << "-cohomology basis:\n";
        oss << flat_basis_to_string(cohomology_bases[q], offset + 6);

        oss << offset_string(offset + 4);
        oss << "Coboundary operator in degree " << q << ":\n";
        oss << coboundary_operators[q].to_string_offset(offset + 6);

    }
    return oss.str();
}

// --- SimplicialComplexMorphism Implementation ---

SimplicialComplexMorphism::SimplicialComplexMorphism(const SimplicialComplex* domain, const SimplicialComplex* codomain)
    : domain(domain), codomain(codomain) {}

SimplicialComplexMorphism::~SimplicialComplexMorphism() {}

bool SimplicialComplexMorphism::is_valid() const {
    // Check if every simplex in the domain is also in the codomain
    for (uint32_t d = 0; d < domain->simplices.size(); ++d) {
        for (const auto& simplex : domain->simplices[d]) {
            bool found = false;
            if (d < codomain->simplices.size()) {
                for (const auto& codomain_simplex : codomain->simplices[d]) {
                    if (simplex == codomain_simplex) {
                        found = true;
                        break;
                    }
                }
            }
            if (!found) {
                return false;
            }
        }
    }
    return true;
}

std::string SimplicialComplexMorphism::to_string() const {
    return to_string_offset(0);
}

std::string SimplicialComplexMorphism::to_string_offset(const uint32_t offset) const {
    std::ostringstream oss;
    oss << offset_string(offset);
    oss << "Simplicial Complex Morphism:\n";
    oss << offset_string(offset + 2);
    oss << "Domain:\n" << domain->to_string_offset(offset + 4);
    oss << offset_string(offset + 2);
    oss << "Codomain:\n" << codomain->to_string_offset(offset + 4);
    oss << offset_string(offset + 2);
    oss << "Is valid: " << (is_valid() ? "true" : "false") << "\n";
    return oss.str();
}

// --- CochainComplexMorphism Implementation ---

// CAREFUL: The underylying object of the domain cochain complex is the
// codomain simplicial complex and vice versa
CochainComplexMorphism::CochainComplexMorphism(const SimplicialComplexMorphism* simplicial_morphism,
    CochainComplex* domain_cochain_complex,
    CochainComplex* codomain_cochain_complex)
    : simplicial_morphism(simplicial_morphism),
    cochain_complex_morphism(nullptr),
    domain_cochain_complex(domain_cochain_complex),
    codomain_cochain_complex(codomain_cochain_complex),
    cochain_maps(),
    cohomology_maps(),
    degree(0)
{
    //std::cout << "CochainComplexMorphism: compute cochain maps\n";
    compute_cochain_maps();
    //std::cout << "CochainComplexMorphism: compute cohomology maps\n";
    compute_cohomology_maps();
}

CochainComplexMorphism::~CochainComplexMorphism() {}

// given a splicial complex morphism, we compute the F_p-linearized cochain maps
void CochainComplexMorphism::compute_cochain_maps() {
    cochain_maps.clear();
    uint32_t p = domain_cochain_complex->get_prime();

    for (int q = 0; q <= (int)simplicial_morphism->codomain->get_dimension(); ++q) 
    {
        uint32_t domain_dim_simplices = (q <= (int)simplicial_morphism->domain->get_dimension()) ? \
            simplicial_morphism->domain->get_simplices()[q].size() : 0;
        uint32_t codomain_dim_simplices = simplicial_morphism->codomain->get_simplices()[q].size();

        // our functor into cochain complexes is contravariant
        std::vector<uint32_t> map_buf(static_cast<size_t>(domain_dim_simplices) * codomain_dim_simplices, 0);

        // Construct the map by finding corresponding simplices
        for (uint32_t i = 0; i < codomain_dim_simplices; ++i) {
            const auto& codomain_simplex = simplicial_morphism->codomain->get_simplices()[q][i];

            // Search for the same simplex in the domain
            if (q <= (int)simplicial_morphism->domain->get_dimension()) {
                auto it = std::find(simplicial_morphism->domain->get_simplices()[q].begin(),
                    simplicial_morphism->domain->get_simplices()[q].end(),
                    codomain_simplex);

                if (it != simplicial_morphism->domain->get_simplices()[q].end()) {
                    uint32_t j = std::distance(simplicial_morphism->domain->get_simplices()[q].begin(), it);
                    map_buf[static_cast<size_t>(j) * codomain_dim_simplices + i] = 1; // Found the simplex, set entry to 1
                }
            }
            // Otherwise, the simplex is not in the codomain, and the map remains 0 for that entry
        }

        //std::cout << "cochain map in dimension " << q << ":\n";
        //std::cout << map.to_string();

        cochain_maps.push_back(
            map_buf.empty() ? FlatMatrix::CreateZero(domain_dim_simplices, codomain_dim_simplices, p)
                            : FlatMatrix(domain_dim_simplices, codomain_dim_simplices, p, map_buf));
    }
}

void CochainComplexMorphism::compute_cohomology_maps() {
    cohomology_maps.clear();
    for (int q = 0; q < (int)domain_cochain_complex->cohomology_bases.size(); ++q) {
        
        //std::cout << "computing morphism of degree " << degree << " in dimension q = " << q << "\n";

        std::vector<FlatMatrix> codomain_cohomology_basis;
        std::vector<FlatMatrix> domain_cohomology_basis = domain_cochain_complex->cohomology_bases[q];

        if (q + degree >= 0 && q + degree < (int32_t)codomain_cochain_complex->cohomology_bases.size()) {
            codomain_cohomology_basis = codomain_cochain_complex->cohomology_bases[q + degree];
        }
        else {
            codomain_cohomology_basis = std::vector<FlatMatrix> {};
        }
        if (codomain_cohomology_basis.size() == 0) {
            cohomology_maps.push_back(FlatMatrix::CreateZero(0, static_cast<uint32_t>(domain_cohomology_basis.size()), domain_cochain_complex->p));
        }
        else {
            // if codomain_cohomology_basis.size() != 0 then q + degree
            // is in a valid range for the codomain cochain complex
            const uint32_t map_rows = static_cast<uint32_t>(codomain_cohomology_basis.size());
            const uint32_t map_cols = static_cast<uint32_t>(domain_cohomology_basis.size());
            std::vector<uint32_t> map_buf(static_cast<size_t>(map_rows) * map_cols, 0);
            for (uint32_t i = 0; i < domain_cohomology_basis.size(); i++)
            {
                //std::cout << "  computing mapping element as a product of\n";
                //std::cout << cochain_maps[q].to_string_offset(4) << "  and\n";
                //std::cout << domain_cohomology_basis[i].to_string_transposed_offset(4);
                FlatMatrix mapped_element = cochain_maps[q] * domain_cohomology_basis[i];

                // FIX (Option A): mapped_element is a RAW ambient cocycle living
                // in C^{q+degree}(codomain). cohomology_projectors[q+degree] has
                // shape (dim H^{q+degree}) x (dim Z^{q+degree}) and expects its
                // input expressed in Z^{q+degree} coordinates, i.e. coefficients
                // w.r.t. cochain_bases[q+degree] (which, despite the name, is a
                // basis of the cocycle space Z^{q+degree} = ker d^{q+degree}).
                // Re-express mapped_element in those coordinates first.
                // change_of_basis throws "Vector is not in image of basis." if
                // mapped_element is not a codomain cocycle -> correctness tripwire.
                std::vector<uint32_t> mapped_element_coeffs =
                    change_of_basis_fm(codomain_cochain_complex->cochain_bases[q + degree], mapped_element);
                const uint32_t n_coeffs = static_cast<uint32_t>(mapped_element_coeffs.size());
                FlatMatrix mapped_element_in_Z =
                    n_coeffs == 0 ? FlatMatrix::CreateZero(0, 1, codomain_cochain_complex->p)
                                  : FlatMatrix(n_coeffs, 1, codomain_cochain_complex->p, mapped_element_coeffs);

                // O(1) dimension agreement -- always on (NDEBUG policy, EngineCheck.h).
                ENGINE_CHECK(codomain_cochain_complex->cohomology_projectors[q + degree].get_cols()
                             == codomain_cochain_complex->cochain_bases[q + degree].size());
                ENGINE_CHECK(codomain_cochain_complex->cohomology_projectors[q + degree].get_cols()
                             == mapped_element_in_Z.get_rows());
                ENGINE_CHECK(codomain_cochain_complex->cohomology_projectors[q + degree].get_rows()
                             == codomain_cohomology_basis.size());

                FlatMatrix projected_mapped_element = codomain_cochain_complex->cohomology_projectors[q + degree] * mapped_element_in_Z;
                //std::cout << "  projected mapped element:\n";
                //std::cout << projected_mapped_element.to_string_transposed_offset(4);
                for (uint32_t j = 0; j < codomain_cohomology_basis.size(); j++)
                {
                    //std::cout << "  i,j:" << i << ", " << j << "\n";
                    map_buf[static_cast<size_t>(j) * map_cols + i] = projected_mapped_element.get_entry(j, 0);
                }
            }
            cohomology_maps.push_back(
                map_buf.empty() ? FlatMatrix::CreateZero(map_rows, map_cols, domain_cochain_complex->p)
                                : FlatMatrix(map_rows, map_cols, domain_cochain_complex->p, map_buf));
        }
    }
}

CochainComplex CochainComplexMorphism::ConeComplex(void)
{
    std::vector<FlatMatrix> cone_coboundaries = {};

    std::vector<FlatMatrix> dA = domain_cochain_complex->get_coboundary_operators();
    std::vector<FlatMatrix> dB = codomain_cochain_complex->get_coboundary_operators();
    
    long int max_dim = (long int)(dA.size()-1 > dB.size()) ? dA.size()-1 : dB.size();

    //std::cout << "Constructing cone complex:\n  dA.size = " << dA.size() << "\n  dB.size = " << dB.size() << "\n  max_dim = " << max_dim << "\n";

    for (long int q = -1; q < max_dim; ++q)
    {
        //std::cout << "    q = " << q << "\n";

        long int qA = q + 1;
        long int qB = q;
        long int colsA, rowsA, colsB, rowsB;

        // q = -1 gives qB = -1, which wraps to SIZE_MAX in the signed/unsigned comparison below: intentional
        // (selects the zero-size branch), so the -Wsign-compare warnings here are kept.
        if (qA >= dA.size())
        {
            rowsA = 0;
            colsA = 0;
        }
        else
        {
            rowsA = dA[qA].get_rows();
            colsA = dA[qA].get_cols();
        }
        if (qB < 0)
        {
            colsB = 0;
            if (cochain_maps.size() > 0)
            {
                rowsB = cochain_maps[qA].get_rows();
            }
            else
            {
                rowsB = 0;
            }
        }
        else if (qB >= dB.size())
        {
            rowsB = 0;
            colsB = 0;
        }
        else
        {
            rowsB = dB[qB].get_rows();
            colsB = dB[qB].get_cols();
        }

        const long int cd_rows = rowsA + rowsB, cd_cols = colsA + colsB;
        std::vector<uint32_t> cd_buf(static_cast<size_t>(cd_rows) * cd_cols, 0);
        //std::cout << "filling first " << rowsA << " x " << colsA << " block with qA[" << qA << "]:\n";
        //std::cout << dA[qA].to_string_offset(4);
       
        // (matrix-engine Phase 3 Stage B) The cone's first block is -d_A. Copy
        // the block in, then negate it with ONE engine `scal` by p-1 over a
        // strided view of the destination window: for a canonical entry a,
        // a*(p-1) mod p is p-a when a != 0 and 0 otherwise, i.e. exactly the
        // `mod_minus` this replaces, entry for entry.
        for (long int i = 0; i < rowsA; ++i)
            for (long int  j = 0; j < colsA; ++j)
            {
                cd_buf[static_cast<size_t>(i) * cd_cols + j] = dA[qA].get_entry(i, j);
            }
        if (rowsA > 0 && colsA > 0)
        {
            const uint32_t cone_p = domain_cochain_complex->p;
            const engine::ModCtx<uint32_t> ctx(cone_p);
            engine::DenseBuf<uint32_t> blockA(cd_buf.data(), cd_buf.size(),
                                              static_cast<uint32_t>(rowsA), static_cast<uint32_t>(colsA),
                                              engine::Layout::RowMajor, static_cast<uint32_t>(cd_cols));
            engine::kernels::scal<uint32_t>(ctx, cone_p - 1, engine::view(blockA));
        }
        //std::cout << "Intermediate codifferential:\n";
        //std::cout << codifferential.to_string_offset(4);
        //std::cout << "filling second " << rowsB << " x " << colsB << " block with:\n";
        //if(qB >= 0)
        //    std::cout << dB[qB].to_string_offset(4);
        for (long int  i = 0; i < rowsB; ++i)
            for (long int  j = 0; j < colsB; ++j)
            {
                cd_buf[static_cast<size_t>(rowsA + i) * cd_cols + (colsA + j)] = dB[qB].get_entry(i, j);
            }
        //std::cout << "filling third " << rowsB << " x " << colsA << " block with:\n";
        //std::cout << cochain_maps[qA].to_string_offset(4);
        for (long int i = 0; i < rowsB; ++i)
            for (long int j = 0; j < colsA; ++j)
            {
                cd_buf[static_cast<size_t>(rowsA + i) * cd_cols + j] = cochain_maps[qA].get_entry(i, j);
            }

        cone_coboundaries.push_back(
            cd_buf.empty()
                ? FlatMatrix::CreateZero(static_cast<uint32_t>(cd_rows), static_cast<uint32_t>(cd_cols), domain_cochain_complex->p)
                : FlatMatrix(static_cast<uint32_t>(cd_rows), static_cast<uint32_t>(cd_cols), domain_cochain_complex->p, cd_buf));
    }
    std::ostringstream oss;
    oss << "Cone of (" << domain_cochain_complex->get_name() << " -> " << codomain_cochain_complex->get_name() << ")";
    return CochainComplex(&cone_coboundaries, domain_cochain_complex->p, oss.str());
}

// Constructs the canonical cochain complex morphism from the cone complex to the domain cochain complex.
CochainComplexMorphism CochainComplexMorphism::ConeToDomainMorphism(const CochainComplex& cone, const CochainComplex& domain)
{
    // The cone complex construction uses a sign convention: -A in degree q for the domain's coboundary map A.
    // To ensure a valid cochain map, we need to incorporate a sign factor of (-1)^q.

    std::vector<FlatMatrix> cochain_maps;
    uint32_t p = domain.get_prime();

    const std::vector<FlatMatrix>& cone_coboundary_operators = cone.get_coboundary_operators();
    const std::vector<FlatMatrix>& domain_coboundary_operators = domain.get_coboundary_operators();

    for (int32_t q = -1; q < (int)cone_coboundary_operators.size() - 1; ++q)
    {
        //std::cout << "ConeToDomainMorphism: q = " << q << "\n";
        int32_t qA = q + 1;

        // Dimensions of the domain's coboundary operator in degree qA
        int32_t colsA = (qA < (int32_t)domain_coboundary_operators.size()) ? domain_coboundary_operators[qA].get_cols() : 0;

        // Dimensions of the cone's coboundary operator in degree q
        int32_t cols_cone = cone_coboundary_operators[q + 1].get_cols();

        // The cochain map in degree qA is the projection onto the domain's component in the cone, adjusted by (-1)^qA.
        std::vector<uint32_t> cm_buf(static_cast<size_t>(colsA) * cols_cone, 0);
        for (int32_t j = 0; j < colsA; ++j)
        {
            // Sign convention: (-1)^qA.
            cm_buf[static_cast<size_t>(j) * cols_cone + j] = ((qA % 2 == 0) ? 1 : (p - 1));
        }
        FlatMatrix cochain_map =
            cm_buf.empty() ? FlatMatrix::CreateZero(colsA, cols_cone, p)
                           : FlatMatrix(colsA, cols_cone, p, cm_buf);
        
        //std::cout << "  cochain_map Cone to Domain in degree " << q << ":\n";
        //std::cout << cochain_map.to_string_offset(4);

        cochain_maps.push_back(cochain_map);
    }

    // Create the cochain complex morphism.
    return CochainComplexMorphism(0, &cochain_maps, const_cast<CochainComplex*>(&cone), const_cast<CochainComplex*>(&domain));
}

// Constructs the canonical cochain complex morphism from the codomain cochain complex to the cone complex.
CochainComplexMorphism CochainComplexMorphism::CodomainToConeMorphism(const CochainComplex& codomain, const CochainComplex& cone)
{
    std::vector<FlatMatrix> cochain_maps;
    uint32_t p = codomain.get_prime();

    const std::vector<FlatMatrix>& cone_coboundary_operators = cone.get_coboundary_operators();
    const std::vector<FlatMatrix>& codomain_coboundary_operators = codomain.get_coboundary_operators();

    for (int32_t q = 0; q < (int32_t)codomain_coboundary_operators.size(); ++q)
    {
        //std::cout << "CodomainToConeMorphism: q = " << q << "\n";

        int32_t qC = q + 1; // we construct a morphism of degree 1

        // Dimensions of the codomain's coboundary operator in degree qB
        int32_t colsB = codomain_coboundary_operators[q].get_cols(); //(q >= 0 && q < codomain_coboundary_operators.size()) ? codomain_coboundary_operators[q].cols : 0;

        // Dimensions of the cone's coboundary operator in degree q
        int32_t cols_cone = (qC < (int32_t)cone_coboundary_operators.size()) ? cone_coboundary_operators[qC].get_cols() : 0;

        // The cochain map in degree q is the inclusion of the codomain's component into the cone shifted by 1.
        std::vector<uint32_t> cm_buf(static_cast<size_t>(cols_cone) * colsB, 0);
        for (int32_t i = 0; i < colsB; ++i)
        {
            cm_buf[static_cast<size_t>(cols_cone - colsB + i) * colsB + i] = 1;
        }
        FlatMatrix cochain_map =
            cm_buf.empty() ? FlatMatrix::CreateZero(cols_cone, colsB, p)
                           : FlatMatrix(cols_cone, colsB, p, cm_buf);

        //std::cout << "  cochain_map Codomain to Cone in degree " << q << ":\n";
        //std::cout << cochain_map.to_string_offset(4);

        cochain_maps.push_back(cochain_map);
    }

    // Create the cochain complex morphism.
    return CochainComplexMorphism(1, &cochain_maps, const_cast<CochainComplex*>(&codomain), const_cast<CochainComplex*>(&cone));
}

std::string CochainComplexMorphism::to_string() const {
    return to_string_offset(0);
}

std::string CochainComplexMorphism::to_string_offset(const uint32_t offset) const {
    std::ostringstream oss;
    oss << offset_string(offset);
    oss << "Cochain Complex Morphism:\n";
    if (simplicial_morphism != nullptr)
    {
        oss << offset_string(offset + 2);
        oss << "Simplicial Morphism:\n" << simplicial_morphism->to_string_offset(offset + 4) << "\n";
    }
    else if (cochain_complex_morphism != nullptr)
    {
        oss << offset_string(offset + 2);
        oss << "Cochain Complex Morphism:\n" << cochain_complex_morphism->to_string_offset(offset + 4) << "\n";
    }
    oss << offset_string(offset + 2);
    oss << "Degree: " << degree << "\n";
    oss << offset_string(offset + 2);
    oss << "Domain Cochain Complex: " << domain_cochain_complex->name_string() << "\n"; // domain_cochain_complex->to_string_offset(offset + 4) << "\n";
    oss << offset_string(offset + 2);
    oss << "Codomain Cochain Complex: " << codomain_cochain_complex->name_string() << "\n";

    //for (uint32_t q = 0; q < cochain_maps.size(); ++q) {
    //    oss << offset_string(offset + 2);
    //    oss << "Cochain Map in degree " << q << ":\n" << cochain_maps[q].to_string_offset(offset + 4) << "\n";
    //}
    for (uint32_t q = 0; q < cohomology_maps.size(); ++q) {
        oss << offset_string(offset + 2);
        oss << "Cohomology Map in degree " << q << ":\n" << cohomology_maps[q].to_string_offset(offset + 4) + "\n";
    }

    return oss.str();
}

// Example: 1-sphere
SimplicialComplex SimplicialOneSphere(void)
{
    std::vector<std::vector<std::vector<uint32_t>>> simplices = {
        {{0}, {1}, {2}},  // 0-simplices
        {{0,1}, {0,2}, {1,2}} // 1-simplices
    };

    // Create and return the simplicial complex
    return SimplicialComplex(simplices, "1-sphere");
}

// Example: 2-sphere
SimplicialComplex SimplicialTwoSphere(void)
{
    std::vector<std::vector<std::vector<uint32_t>>> simplices = {
        {{0}, {1}, {2}, {3}},  // 0-simplices
        {{0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}}, // 1-simplices
        {{0,1,2}, {0,1,3}, {0,2,3}, {1,2,3}}  // 2-simplices
    };

    // Create and return the simplicial complex
    return SimplicialComplex(simplices, "2-sphere");
}

// Example: 3-sphere
SimplicialComplex SimplicialThreeSphere(void)
{
    std::vector<std::vector<std::vector<uint32_t>>> simplices = {
        // 0-simplices
        {{0}, {1}, {2}, {3}, {4}},
        // 1-simplices
        {{0,1}, {0,2}, {0,3}, {0,4},  {1,2}, {1,3}, {1,4},  {2,3}, {2,4}, {3,4}},
        // 2-simplices
        {{0,1,2}, {0,1,3}, {0,1,4}, {0,2,3}, {0,2,4}, {0,3,4}, {1,2,3}, {1,2,4}, {1,3,4}, {2,3,4}},
        // 3-simplices
        {{0,1,2,3}, {0,1,2,4}, {0,1,3,4}, {0,2,3,4}, {1,2,3,4}}
    };

    // Create and return the simplicial complex
    return SimplicialComplex(simplices, "3-sphere");
}

// Example: cylinder
SimplicialComplex SimplicialCylinder(void)
{
    std::vector<std::vector<std::vector<uint32_t>>> simplices = {
        // 0-simplices
        {{0}, {1}, {2}, {3}, {4}, {5}},
        // 1-simplices
        {{0,1}, {0,2}, {0,3}, {0,4}, {1,3}, {1,4}, {1,5}, {2,3}, {2,4}, {2,5}, {3,5}, {4,5}},
        // 2-simplices
        {{0,1,3}, {0,2,3}, {1,4,5}, {0,1,4}, {2,3,5}, {2,4,5}}
    };

    // Create and return the simplicial complex
    return SimplicialComplex(simplices, "Cylinder");
}

// Example: Moebius band
SimplicialComplex SimplicialMoebiusBand(void)
{
    std::vector<std::vector<std::vector<uint32_t>>> simplices = {
        // 0-simplices
        {{0}, {1}, {2}, {3}, {4}, {5}},
        // 1-simplices
        {{0,1}, {0,2}, {0,3}, {0,4}, {1,3}, {1,4}, {0,5}, {2,3}, {2,4}, {2,5}, {3,5}, {4,5}},
        // 2-simplices
        {{0,1,3}, {0,2,3}, {0,4,5}, {0,1,4}, {2,3,5}, {2,4,5}}
    };

    // Create and return the simplicial complex
    return SimplicialComplex(simplices, "Moebius Band");
}

// Example: A triangulation of the real projective plane
SimplicialComplex SimplicialProjectivePlane(void)
{
    std::vector<std::vector<std::vector<uint32_t>>> simplices = {
        // 0-simplices
        {{1}, {2}, {3}, {4}, {5}, {6}},
        // 1-simplices
        {{1,2}, {1,3}, {1,4}, {1,5}, {1,6}, {2,3}, {2,4}, {2,5}, {2,6}, {3,4}, {3,5}, {3,6}, {4,5}, {4,6}, {5,6}},
        // 2-simplices
        {{1,2,5}, {1,2,6}, {1,3,4}, {1,3,6}, {1,4,5}, {2,3,4}, {2,3,5}, {2,4,6}, {3,5,6}, {4,5,6}}
    };

    // Create and return the simplicial complex
    return SimplicialComplex(simplices, "Projective Plane");
}

SimplicialComplex SimplicialProjectivePlaneBoundary(void)
{
    std::vector<std::vector<std::vector<uint32_t>>> simplices = {
        // 0-simplices
        {{1}, {2}, {3}},
        // 1-simplices
        {{1,2}, {1,3}, {2,3}},
    };

    // Create and return the simplicial complex
    return SimplicialComplex(simplices, "Boundary of Projective Plane");
}

uint32_t SimplicialCohomologyValidation(void)
{
    uint32_t p = 2;

    // Create two simplicial complexes, the first embedding into the second:
    
    //SimplicialComplex S1 = SimplicialThreeSphere();
    
    //SimplicialComplex S2 = SimplicialCylinder();
    
    //SimplicialComplex S2 = SimplicialMoebiusBand();
    
    SimplicialComplex S1 = SimplicialProjectivePlaneBoundary();
    SimplicialComplex S2 = SimplicialProjectivePlane();

    // Create cochain complexes for the simplicial complexes
    CochainComplex cochain_S1(&S1, p);
    CochainComplex cochain_S2(&S2, p);

    std::cout << cochain_S1.to_string();
    std::cout << cochain_S2.to_string();

    // Create a simplicial morphism (assuming S1 is a subcomplex of S2)
    SimplicialComplexMorphism morph(&S1, &S2);

    std::cout << morph.to_string() << std::endl;

    // Create a cochain complex morphism
    CochainComplexMorphism cochain_morph(&morph, &cochain_S2, &cochain_S1);

    // Output information
    std::cout << cochain_morph.to_string() << std::endl;

    // Create cone complex
    CochainComplex cone = cochain_morph.ConeComplex();
    std::cout << cone.to_string() << std::endl;

    CochainComplexMorphism cone_to_domain = cochain_morph.ConeToDomainMorphism(cone, cochain_S2);
    std::cout << cone_to_domain.to_string() << std::endl;

    CochainComplexMorphism codomain_to_cone = cochain_morph.CodomainToConeMorphism(cochain_S1, cone);
    std::cout << codomain_to_cone.to_string() << std::endl;

    std::vector<uint32_t> dimAs = cone.get_cohomology_dimensions();
    std::vector<uint32_t> dimBs = cochain_S2.get_cohomology_dimensions();
    std::vector<uint32_t> dimCs = cochain_S1.get_cohomology_dimensions();
    std::vector<FlatMatrix>* f = cone_to_domain.get_cohomology_maps();
    std::vector<FlatMatrix>* g = cochain_morph.get_cohomology_maps();
    std::vector<FlatMatrix>* h = codomain_to_cone.get_cohomology_maps();
    uint32_t max_dim;
    if (dimBs.size() > dimAs.size())
    {
        max_dim = (uint32_t)dimBs.size();
    }
    else
    {
        max_dim = (uint32_t)dimAs.size();
    }
    if (dimCs.size() > max_dim)
    {
        max_dim = (uint32_t)dimCs.size();
    }
    std::cout << "Long exact sequence:\n";
    std::cout << " 0 -> ";
    for (uint32_t q = 0; q < max_dim; ++q)
    {
        uint32_t dimA = 0;
        if (q < dimAs.size())
            dimA = dimAs[q];
        uint32_t dimB = 0;
        if (q < dimBs.size())
            dimB = dimBs[q];
        uint32_t dimC = 0;
        if (q < dimCs.size())
            dimC = dimCs[q];
        std::cout << dimA << " -> " << dimB << " -> " << dimC << "\n";
        std::cout << "   -> ";
    }
    std::cout << "0\n\n";
    FlatMatrix null_matrix = FlatMatrix::CreateZero(0, 0, p);
    for (uint32_t q = 0; q < max_dim; ++q)
    {
        FlatMatrix* matrix_f;
        if (q < f->size())
            matrix_f = &((*f)[q]);
        else
            matrix_f = &null_matrix;
        FlatMatrix* matrix_g;
        if (q < g->size())
            matrix_g = &((*g)[q]);
        else
            matrix_g = &null_matrix;
        FlatMatrix* matrix_h;
        if (q < h->size())
            matrix_h = &((*h)[q]);
        else
            matrix_h = &null_matrix;
        std::cout << "  Cohomology maps in degree " << q << ":\n";
        std::cout << "    f: " << matrix_f->to_string_offset(6);
        std::cout << "    g: " << matrix_g->to_string_offset(6);
        std::cout << "    h: " << matrix_h->to_string_offset(6);
    }
    std::cout << "Testing relations:\n";
    std::vector<FlatMatrix> A = cone.get_coboundary_operators();
    std::vector<FlatMatrix> B = cochain_S2.get_coboundary_operators();
    std::vector<FlatMatrix> C = cochain_S1.get_coboundary_operators();
    std::vector<FlatMatrix>* F = cone_to_domain.get_cochain_maps();
    std::vector<FlatMatrix>* G = cochain_morph.get_cochain_maps();
    std::vector<FlatMatrix>* H = codomain_to_cone.get_cochain_maps();
    for (uint32_t q = 0; q < max_dim - 1; ++q)
    {
        // we need 4 coboundary maps
        FlatMatrix* matrix_A;
        if (q < A.size())
            matrix_A = &(A[q]);
        else
            matrix_A = &null_matrix;
        FlatMatrix* matrix_B;
        if (q < B.size())
            matrix_B = &(B[q]);
        else
            matrix_B = &null_matrix;
        FlatMatrix* matrix_C;
        if (q < C.size())
            matrix_C = &(C[q]);
        else
            matrix_C = &null_matrix;
        FlatMatrix* matrix_A1;
        if (q + 1 < A.size())
            matrix_A1 = &(A[q + 1]);
        else
            matrix_A1 = &null_matrix;
        // we need 6 cochain maps
        FlatMatrix* matrix_f;
        if (q < F->size())
            matrix_f = &((*F)[q]);
        else
            matrix_f = &null_matrix;
        FlatMatrix* matrix_g;
        if (q < G->size())
            matrix_g = &((*G)[q]);
        else
            matrix_g = &null_matrix;
        FlatMatrix* matrix_h;
        if (q < H->size())
            matrix_h = &((*H)[q]);
        else
            matrix_h = &null_matrix;
        FlatMatrix* matrix_f1;
        if (q + 1 < F->size())
            matrix_f1 = &((*F)[q + 1]);
        else
            matrix_f1 = &null_matrix;
        FlatMatrix* matrix_g1;
        if (q + 1 < G->size())
            matrix_g1 = &((*G)[q + 1]);
        else
            matrix_g1 = &null_matrix;
        FlatMatrix* matrix_h1;
        if (q + 1 < H->size())
            matrix_h1 = &((*H)[q + 1]);
        else
            matrix_h1 = &null_matrix;

        std::cout << "  Relations in degree " << q << ":\n";
        std::cout << "    B(" << matrix_B->get_rows() << "x" << matrix_B->get_cols() << ")*f(" << matrix_f->get_rows() << "x" << matrix_f->get_cols() << ")\n";
        std::cout << "    f1(" << matrix_f1->get_rows() << "x" << matrix_f1->get_cols() << ")*A(" << matrix_A->get_rows() << "x" << matrix_A->get_cols() << ")\n";

        FlatMatrix Bf = (*matrix_B) * (*matrix_f);
        //std::cout << "    B*f:\n" << Bf.to_string_offset(6);
        FlatMatrix f1A = (*matrix_f1) * (*matrix_A);
        //std::cout << "    f1*A:\n" << f1A.to_string_offset(6);
        FlatMatrix Bfminusf1A = Bf - f1A;
        if (Bfminusf1A.is_zero())
            std::cout << "    B*f - f1*A = 0\n";
        else
            std::cout << "    B*f - f1*A:\n" << Bfminusf1A.to_string_offset(6);

        std::cout << "    C(" << matrix_C->get_rows() << "x" << matrix_C->get_cols() << ")*g(" << matrix_g->get_rows() << "x" << matrix_g->get_cols() << ")\n";
        std::cout << "    g1(" << matrix_g1->get_rows() << "x" << matrix_g1->get_cols() << ")*B(" << matrix_B->get_rows() << "x" << matrix_B->get_cols() << ")\n";

        FlatMatrix Cg = (*matrix_C) * (*matrix_g);
        //std::cout << "    C*g:\n" << Cg.to_string_offset(6);
        FlatMatrix g1B = (*matrix_g1) * (*matrix_B);
        //std::cout << "    g1*B:\n" << g1B.to_string_offset(6);
        FlatMatrix Cgminusg1B = Cg - g1B;
        if (Cgminusg1B.is_zero())
            std::cout << "    C*g - g1*B = 0\n";
        else
            std::cout << "    C*g - g1*B:\n" << Cgminusg1B.to_string_offset(6);

        std::cout << "    A1(" << matrix_A1->get_rows() << "x" << matrix_A1->get_cols() << ")*h(" << matrix_h->get_rows() << "x" << matrix_h->get_cols() << ")\n";
        std::cout << "    h1(" << matrix_h1->get_rows() << "x" << matrix_h1->get_cols() << ")*C(" << matrix_C->get_rows() << "x" << matrix_C->get_cols() << ")\n";

        FlatMatrix A1h = (*matrix_A1) * (*matrix_h);
        //std::cout << "    A1*h:\n" << A1h.to_string_offset(6);
        FlatMatrix h1C = (*matrix_h1) * (*matrix_C);
        //std::cout << "    h1*C:\n" << h1C.to_string_offset(6);
        FlatMatrix A1hminush1C = A1h - h1C;
        if (Cgminusg1B.is_zero())
            std::cout << "    A1*h - h1*C = 0\n";
        else
            std::cout << "    A1*h - h1*C:\n" << Cgminusg1B.to_string_offset(6);
    }
    std::cout << "\n";

    // Regression guard (P0 fix): the degree-0 restriction map
    //   g[0] = H^0(RP^2) -> H^0(dRP^2)
    // must be the 1x1 identity [1] over F_2 (both spaces connected; restriction
    // of a constant cocycle is an isomorphism). g == cochain_morph cohomology
    // maps (declared above). Fail with a nonzero exit code if this is violated.
    if (g->size() < 1 ||
        (*g)[0].get_rows() != 1 || (*g)[0].get_cols() != 1 || (*g)[0].get_entry(0, 0) != 1) {
        std::cout << "FAIL: restriction H^0(RP^2)->H^0(dRP^2) is not the 1x1 identity [1].\n";
        if (g->size() >= 1) {
            std::cout << "  got g[0] (" << (*g)[0].get_rows() << "x" << (*g)[0].get_cols() << "):\n"
                      << (*g)[0].to_string_offset(4);
        } else {
            std::cout << "  g is empty (no degree-0 cohomology map).\n";
        }
        return 1;
    }
    std::cout << "PASS: restriction H^0(RP^2)->H^0(dRP^2) = [1] (1x1 identity).\n";

    return 0;
}
