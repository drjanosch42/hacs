// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// matrix-engine Phase 6 -- see HeckeModelDescriptor.h for the rationale
// (audit item 7.18 / plan Q8).

#include "HeckeModelDescriptor.h"

#include "../SimplicialCohomology.h"

#include <sstream>
#include <stdexcept>

namespace hecke {

std::string SimplicialModelDescriptor::shape_string() const {
    std::ostringstream oss;
    for (size_t q = 0; q < simplex_counts.size(); ++q) {
        if (q) oss << "/";
        oss << simplex_counts[q];
    }
    return oss.str();
}

bool SimplicialModelDescriptor::matches(const SimplicialComplex& K) const {
    if (K.get_dimension() != dimension) return false;
    for (uint32_t q = 0; q <= dimension; ++q) {
        if (K.get_num_q_simplices(q) != num_simplices(q)) return false;
    }
    return true;
}

const SimplicialModelDescriptor& SL2Z7VertexModel() {
    // Function-local static: no ordering dependency with any other TU's
    // statics (the descriptor is consulted from constructors).
    static const SimplicialModelDescriptor d = [] {
        SimplicialModelDescriptor m;
        m.name = "sl2z_7vertex";
        m.description = "the fixed 7-vertex triangulation of SL(2,Z)\\H "
                        "(CreateSL2ZFundamentalDomainTriangulation)";
        m.dimension = 2;
        m.simplex_counts = {7, 11, 5};
        m.relation_source = "GenerateSL2ZRelationsWithGroupData";
        return m;
    }();
    return d;
}

const SimplicialModelDescriptor* FindModelDescriptor(const std::string& name) {
    const SimplicialModelDescriptor& d = SL2Z7VertexModel();
    if (name == d.name) return &d;
    return nullptr;
}

void RequireModel(const SimplicialModelDescriptor& d, const SimplicialComplex* K,
                  const char* who) {
    const std::string prefix = std::string(who ? who : "RequireModel") + ": ";
    if (K == nullptr) {
        throw std::runtime_error(prefix + "null base complex.");
    }
    if (!d.matches(*K)) {
        std::ostringstream oss;
        oss << prefix << "base complex is not the " << d.name << " model (expected "
            << d.shape_string() << " simplices of dimension " << d.dimension << ", got ";
        for (uint32_t q = 0; q <= K->get_dimension(); ++q) {
            if (q) oss << "/";
            oss << K->get_num_q_simplices(q);
        }
        oss << " of dimension " << K->get_dimension() << ").";
        throw std::runtime_error(oss.str());
    }
}

} // namespace hecke
