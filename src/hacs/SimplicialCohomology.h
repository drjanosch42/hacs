// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#ifndef SIMPLICIAL_COHOMOLOGY_H
#define SIMPLICIAL_COHOMOLOGY_H

#include "../ModArithmetic/ModArithmetic.h"
#include "../ModArithmetic/FlatMatrix.h"
#include "../ModArithmetic/helper.h"
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <string>
#include <set>
#include <map>
#include <cstdint>
#include <type_traits>

// Forward declarations
class CochainComplex;

#if OC_ENABLE_LEGACY
template<class Elt> class BlockFlatMatrixT;   // src/ModArithmetic/legacy/BlockFlatMatrix.h (S24 CL-L2)
#endif

class SimplicialComplex {
private:
    std::vector<std::vector<std::vector<uint32_t>>> simplices;
    size_t num_vertices;
    uint32_t dim;
    int32_t euler_characteristic;
    std::string name;

    // --- NEW MEMBERS for trivial cocycle bases ---
    // (matrix-engine Phase 3) The trivial-coefficient layer keeps its Matrix
    // SEMANTICS -- same coboundary operators, same kernel/image bases, same
    // factor_basis pivot rule -- but its STORAGE is FlatMatrixT now, so the
    // deprecated Matrix no longer reaches any pipeline consumer of this header.
    std::vector<std::vector<FlatMatrix>> trivial_cocycle_bases_Z1;
    bool trivial_cocycle_bases_computed; // Flag to check if computed

    // Stage S13: parallel uint64_t cache for the extended coefficient regime
    // (ell < 2^64). SimplicialComplex stays a combinatorial (non-templated)
    // class; only these value caches and the *_elt<Elt> value methods below are
    // element-width parameterized. The uint32_t path is unchanged.
    std::vector<std::vector<FlatMatrix64>> trivial_cocycle_bases_Z1_64;
    bool trivial_cocycle_bases_computed_64 = false;

    void validate_simplices();

    // (matrix-engine Phase 3; audit 02 Sec. 7.17) Lazily built ORDERED index over
    // `simplices[q]`, replacing the O(N) `std::find` scans at the three sites the
    // plan names (SimplicialCohomology.cpp coboundary_operator_elt,
    // LocalSystemUtils.cpp find_simplex_idx, EquivariantCochainComplexMorphismOnK.cpp
    // find_simplex_K1_idx_in_K2). It is an `std::map`, NEVER an `unordered_*`:
    // 02 Sec. 6.5(5) records that no hash container exists on any result path in
    // `Hecke/` or the morphism class by design, and R10 forbids introducing one.
    // `simplices` is set once by the constructor and never mutated afterwards
    // (there is no setter), so the cache can never go stale; it is `mutable` only
    // so the lookup can stay `const`. Single-threaded by project convention.
    mutable std::vector<std::map<std::vector<uint32_t>, uint32_t>> simplex_index_;
    mutable std::vector<char> simplex_index_built_;
    const std::map<std::vector<uint32_t>, uint32_t>& simplex_index(uint32_t q) const;

public:
    SimplicialComplex(); // Default constructor
    SimplicialComplex(const std::vector<std::vector<std::vector<uint32_t>>>& simplices, const std::string name);
    ~SimplicialComplex();

    std::string to_string() const;
    std::string to_string_offset(const uint32_t offset) const;

    // Getters
    uint32_t get_num_vertices() const { return num_vertices; }
    uint32_t get_dimension() const { return dim; }
    uint32_t get_euler_characteristic() const { return euler_characteristic; }
    std::string get_name() const { return name; }
    const std::vector<std::vector<std::vector<uint32_t>>>& get_simplices() const { return simplices; }
    uint32_t get_num_q_simplices(uint32_t q) const {
        if (q < simplices.size()) {
            return static_cast<uint32_t>(simplices[q].size());
        }
        return 0;
    }


    // (matrix-engine Phase 3) Index of `sorted_vertices` among the q-simplices,
    // or -1 when q is out of range or the simplex is absent. Value-preserving
    // replacement for `std::find(simplices[q].begin(), ..., v)`: on a duplicated
    // simplex the FIRST occurrence wins, exactly as `std::find` does, because the
    // index is built by ascending position with `emplace` (which keeps the first
    // insertion for a key). The argument must already be sorted ascending -- the
    // constructor's `validate_simplices()` enforces that for the stored simplices.
    int64_t find_simplex_index(uint32_t q, const std::vector<uint32_t>& sorted_vertices) const;

    // Coboundary operator for trivial coefficients
    FlatMatrix coboundary_operator(uint32_t q, uint32_t p) const;
    // (matrix-engine Phase 0) The non-template BlockFlatMatrix overload
    // `coboundary_operator_block(q, p, d_V)` had zero callers and was deleted;
    // use coboundary_operator_block_elt<Elt> below.

    // Stage S13: element-width-generic variants (Elt in {uint32_t, uint64_t}).
    // The uint32_t instantiation carries identical logic; the non-templated
    // method above delegates to <uint32_t> (single source of truth).
    template<class Elt>
    FlatMatrixT<Elt> coboundary_operator_elt(uint32_t q, Elt p) const;
#if OC_ENABLE_LEGACY
    // (S24 CL-L2) legacy/SimplicialCohomologyLegacy.inc -- ON builds only (the legacy cohomology arm).
    template<class Elt>
    BlockFlatMatrixT<Elt> coboundary_operator_block_elt(uint32_t q, Elt p, uint32_t d_V) const;
#endif

    // --- NEW METHODS for trivial cocycle bases ---
    // Computes and caches bases for Z_1^q = ker D_1^q
    void compute_trivial_cocycle_bases(uint32_t prime_for_coboundary);

    // Gets the pre-computed basis for Z_1^q. Ensures compute_trivial_cocycle_bases has been called.
    // Throws if not computed or q is out of bounds.
    const std::vector<FlatMatrix>& get_trivial_cocycle_basis_Z1(uint32_t q) const;

    // Gets the dimension k_q = dim Z_1^q
    uint32_t get_trivial_cocycle_basis_Z1_dim(uint32_t q) const;

    // Stage S13: element-width-generic cocycle-basis cache (per-width member).
    template<class Elt>
    void compute_trivial_cocycle_bases_elt(Elt prime_for_coboundary);
    template<class Elt>
    const std::vector<FlatMatrixT<Elt>>& get_trivial_cocycle_basis_Z1_elt(uint32_t q) const;


    // Associated cochain complex
    friend class CochainComplex; // Allow CochainComplex to access private members
    friend class SimplicialComplexMorphism; // Allow SimplicialComplexMorphism to access private members
};

class CochainComplex {
private:
    // underlying simplicial complex if defined
    const SimplicialComplex* simplicial_complex;

    std::vector<FlatMatrix> coboundary_operators;
    std::vector<std::vector<FlatMatrix>> cochain_bases;
    std::vector<std::vector<FlatMatrix>> coboundary_bases;
    std::vector<std::vector<FlatMatrix>> cohomology_bases;
    std::vector<std::vector<uint32_t>> cohomology_base_indices;
    std::vector<FlatMatrix> cohomology_projectors;
    std::vector<uint32_t> cohomology_dimensions;
    int32_t euler_characteristic;
    uint32_t p;
    std::string name;

    void compute_coboundary_operators();
    void compute_cochains();
    void compute_coboundaries();
    void compute_cohomology_bases();
    void compute_cohomology_dimensions();

public:
    CochainComplex(const SimplicialComplex* simplicial_complex, uint32_t prime);
    CochainComplex(const std::vector<FlatMatrix>* coboundary_operators, uint32_t prime, const std::string name);
    CochainComplex(std::vector<FlatMatrix>&& coboundary_operators, uint32_t prime, const std::string name);

    ~CochainComplex();

    std::string name_string() const { return name;  };
    std::string to_string() const;
    std::string to_string_offset(const uint32_t offset) const;

    // Getters for results
    uint32_t get_prime() const { return p; }
    const std::vector<FlatMatrix>& get_coboundary_operators() const { return coboundary_operators; }
    const std::vector<std::vector<FlatMatrix>>& get_cochain_bases() const { return cochain_bases; }
    const std::vector<std::vector<FlatMatrix>>& get_coboundary_bases() const { return coboundary_bases; }
    const std::vector<std::vector<FlatMatrix>>& get_cohomology_bases() const { return cohomology_bases; }
    const std::vector<std::vector<uint32_t>> get_cohomology_base_indices() const { return cohomology_base_indices; }
    const std::vector<FlatMatrix>& get_cohomology_projectors() const { return cohomology_projectors; }
    const std::vector<uint32_t>& get_cohomology_dimensions() const { return cohomology_dimensions; }
    // (matrix-engine Phase 0) Was `const std::uint32_t&`: euler_characteristic is
    // an int32_t, so the conversion created a temporary and the getter returned a
    // dangling reference to it (UB; -Wreturn-local-addr, 36 hits once -Wall/-Wextra
    // was turned on). Returns by value, in the member's own type. No caller existed.
    std::int32_t get_euler_characteristic() const { return euler_characteristic; }
    std::string get_name() const { return name; }

    // Cohomology computation
    void compute_cohomology();

    friend class CochainComplexMorphism;
};

class SimplicialComplexMorphism {
private:
    const SimplicialComplex* domain;
    const SimplicialComplex* codomain;

public:
    SimplicialComplexMorphism(const SimplicialComplex* domain, const SimplicialComplex* codomain);
    ~SimplicialComplexMorphism();

    bool is_valid() const;
    std::string to_string() const;
    std::string to_string_offset(const uint32_t offset) const;

    friend class CochainComplexMorphism;
};

class CochainComplexMorphism {
private:
    const SimplicialComplexMorphism* simplicial_morphism;
    const CochainComplexMorphism* cochain_complex_morphism;
    CochainComplex* domain_cochain_complex;
    CochainComplex* codomain_cochain_complex;
    std::vector<FlatMatrix> cochain_maps;
    std::vector<FlatMatrix> cohomology_maps;
    int32_t degree;

public:
    CochainComplexMorphism(const SimplicialComplexMorphism* simplicial_morphism,
        CochainComplex* domain_cochain_complex,
        CochainComplex* codomain_cochain_complex);
    CochainComplexMorphism(const int32_t degree, const std::vector<FlatMatrix>* operators,
        CochainComplex* domain_cochain_complex,
        CochainComplex* codomain_cochain_complex);
    CochainComplexMorphism ConeToDomainMorphism(const CochainComplex& cone, const CochainComplex& domain);
    CochainComplexMorphism CodomainToConeMorphism(const CochainComplex& codomain, const CochainComplex& cone);
    ~CochainComplexMorphism();

    void compute_cochain_maps();
    void compute_cohomology_maps();
    int32_t get_degree() const { return degree; };
    std::vector<FlatMatrix>* get_cochain_maps() { return &cochain_maps; };
    std::vector<FlatMatrix>* get_cohomology_maps() { return &cohomology_maps; };

    CochainComplex ConeComplex();

    std::string to_string() const;
    std::string to_string_offset(const uint32_t offset) const;
};

SimplicialComplex SimplicialOneSphere();
SimplicialComplex SimplicialTwoSphere();
SimplicialComplex SimplicialThreeSphere();
SimplicialComplex SimplicialCylinder();
SimplicialComplex SimplicialMoebiusBand();
SimplicialComplex SimplicialProjectivePlane();
SimplicialComplex SimplicialProjectivePlaneBoundary();

uint32_t SimplicialCohomologyValidation(void);

#endif // SIMPLICIAL_COHOMOLOGY_H
