// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions phase E2 cont. (slot 5, WP-1) -- the record layer's view of
// the E3 / E4b / E5 objects.  See HeckeExtComplexRecords.h for the contract
// and the [C] / [E] / [X] citations.
//
// INCLUDE DISCIPLINE (check_heckeext_includes.sh rules 1, 2, 4): its own
// header (hence HeckeExt/ and the pipeline headers it names), FpPoly for the
// characteristic polynomial, EngineCheck.h and C++ standard headers.  No
// Console/, no Cuda/, no Engine/, no BlockFlatMatrix.h, no `unordered_`.

#include "HeckeExtComplexRecords.h"

#include "../Hecke/FpPoly.h"

#include "../../ModArithmetic/EngineCheck.h"

#include <string>

namespace hecke_ext {

// ---------------------------------------------------------------------------
// The group names.
// ---------------------------------------------------------------------------
std::string BoundaryGroupName(uint32_t q) { return "H" + std::to_string(q) + "_bd"; }
std::string ConeGroupName(uint32_t q) { return "H" + std::to_string(q) + "c"; }
std::string SignedGroupName(uint32_t q, bool plus) {
    return "H" + std::to_string(q) + (plus ? "_plus" : "_minus");
}

namespace {

// [C] `sec:recordschema` item 5: the record's `ComplexRecord` carries a degree
// COUNT and derives q_max from it; ComplexDegreeFields carries q_max.  This is
// the one place the two spellings meet, and it is deliberately a total
// function of the field object -- never of a maximum over anything.
ComplexRecord FromDegreeFields(const ComplexDegreeFields& f) {
    ComplexRecord c;
    c.available = true;
    c.num_degrees = f.q_max + 1u;
    c.truncated = f.truncated;
    c.q_max_configured = (f.q_max_source == "set");
    return c;
}

} // namespace

// ---------------------------------------------------------------------------
// E5 -- the cone.
// ---------------------------------------------------------------------------
ComplexRecord BoundaryComplexRecord(const EquivariantMappingCone& cone) {
    ComplexRecord c = FromDegreeFields(cone.boundary_fields());
    // The two labels the generic cone cannot derive and was told by
    // Hecke/SL2ZBoundaryPair ("circle"; group cohomology iff ell != 2).
    c.model = cone.boundary_model();
    if (cone.boundary_is_group_cohomology_known()) {
        c.has_group_cohomology_flag = true;
        c.is_group_cohomology = cone.boundary_is_group_cohomology();
    }
    return c;
}

ComplexRecord ConeComplexRecord(const EquivariantMappingCone& cone) {
    ComplexRecord c = FromDegreeFields(cone.cone_fields());
    // [C] `sec:recordschema` item 5: a complex built from others carries
    // `q_max_source = "derived"` UNCONDITIONALLY together with the names of
    // the complexes it is derived from.  The cone's own fields already say
    // "derived"; this re-asserts it so that a record can never claim "set" on
    // a complex whose q_max is not a user's to choose.
    ENGINE_CHECK_MSG(!c.q_max_configured,
                     "HeckeExtComplexRecords: a derived complex may not carry "
                     "q_max_source = \"set\" ([C] sec:recordschema item 5)");
    c.derived_from.push_back("bulk");
    c.derived_from.push_back("boundary");

    for (const ConeLesSpot& sp : cone.les_spots()) {
        LesMapSpot out;
        out.term = sp.term;
        out.q = sp.q;
        out.in_built = sp.in_built;
        out.out_built = sp.out_built;
        out.asserted = sp.asserted;
        out.dim_middle = sp.dim_middle;
        out.rank_in = sp.rank_in;
        out.rank_out = sp.rank_out;
        out.composite_zero = sp.composite_zero;
        out.exact = sp.exact;
        out.coker_in_dim = sp.coker_in_dim;
        c.les_maps.push_back(out);
    }
    return c;
}

std::vector<GroupRecord> BoundaryGroups(const EquivariantMappingCone& cone) {
    std::vector<GroupRecord> out;
    const ConeCertificate& cert = cone.certificate();
    for (uint32_t q = 0; q < cert.dim_H_bd.size(); ++q) {
        GroupRecord g;
        g.name = BoundaryGroupName(q);
        g.available = true;
        g.dim = cert.dim_H_bd[q];
        out.push_back(g);
    }
    return out;
}

std::vector<GroupRecord> ConeGroups(const EquivariantMappingCone& cone) {
    std::vector<GroupRecord> out;
    for (uint32_t q = 0; q <= cone.q_max(); ++q) {
        GroupRecord g;
        g.name = ConeGroupName(q);
        g.available = true;
        g.dim = cone.dim_Hc(q);
        out.push_back(g);
    }
    return out;
}

LesSummary ConeLesSummary(const EquivariantMappingCone& cone, bool hecke_equivariant) {
    LesSummary s;
    s.available = true;
    s.hecke_equivariant = hecke_equivariant;
    // `exact` is a conjunction over the ASSERTED spots only: an unasserted
    // spot records a cokernel bound and claims nothing, so folding it in
    // either direction would be a claim the cone did not make.
    bool any_asserted = false;
    bool all_exact = true;
    for (const ConeLesSpot& sp : cone.les_spots()) {
        if (!sp.asserted) continue;
        any_asserted = true;
        if (!sp.exact) all_exact = false;
    }
    s.exact = any_asserted && all_exact;
    for (uint32_t q = 0; q < cone.certificate().r_flat.size(); ++q) {
        s.ranks.emplace_back("r" + std::to_string(q), cone.rank_r(q));
    }
    return s;
}

bool ConeDimH1Shriek(const EquivariantMappingCone& cone, uint32_t* out) {
    ENGINE_CHECK(out != nullptr);
    // H^1_! = ker r^1 ([C] `cl:les`(2)), so its dimension is
    // dim H^1(bulk) - rank r^1 -- and it exists exactly where r^1 was built.
    if (!cone.r_built(1)) return false;
    const ConeCertificate& cert = cone.certificate();
    if (cert.dim_H_bulk.size() <= 1) return false;
    const uint32_t dim_H1 = cert.dim_H_bulk[1];
    const uint32_t rank = cone.rank_r(1);
    ENGINE_CHECK_MSG(rank <= dim_H1,
                     "HeckeExtComplexRecords: rank r^1 exceeds dim H^1");
    *out = dim_H1 - rank;
    return true;
}

// ---------------------------------------------------------------------------
// E5-3 -- the cone action's two recorded flags.
// ---------------------------------------------------------------------------
void AttachConeActionRecord(ComplexRecord& pair, const hecke::HeckeConeAction& action) {
    ENGINE_CHECK_MSG(pair.available,
                     "HeckeExtComplexRecords: the cone flags belong inside an "
                     "AVAILABLE pair object ([C] sec:recordschema item 6(c))");
    const hecke::ConeActionCertificate& cert = action.certificate();
    pair.has_cone_flags = true;
    pair.phi_flat_identity = cert.phi_flat_identity;
    pair.residual_flat_zero = cert.residual_flat_zero;
    pair.action_record_schema = "hecke_cone_action/v1";
}

// ---------------------------------------------------------------------------
// E3 -- the involution.
// ---------------------------------------------------------------------------
IotaDegreeRecord IotaRecord(const HeckeSignSplit& split) {
    ENGINE_CHECK_MSG(split.computed(),
                     "HeckeExtComplexRecords: IotaRecord needs a computed split");
    const SignSplitCertificate& cert = split.certificate();
    IotaDegreeRecord r;
    r.degree = split.degree();
    r.status = (cert.status == SignSplitStatus::Split) ? "split" : "not_applicable";
    r.dim_H = split.dim_H();
    r.dim_plus = split.dim_plus();
    r.dim_minus = split.dim_minus();
    r.commutation_asserted = cert.commutation_asserted;
    r.operators = static_cast<uint32_t>(split.num_operators());
    r.operators_commuting = cert.num_commuting;

    // rank and char-poly of the stored iota^*, in the stored basis.
    if (cert.iota.rows > 0) {
        // MatrixT caches its echelon form lazily, so rank() is non-const; the
        // copy keeps the certificate the immutable object it is.
        ExtMatrix iota_copy = cert.iota;
        r.rank_iota = iota_copy.rank();
        r.charpoly_iota = hecke::CharPolyBerkowitz(cert.iota).coeffs;
    }

    const EisensteinLine& e = split.eisenstein_line();
    r.eisenstein_applicable = e.applicable;
    r.eisenstein_status = e.status.empty() ? std::string("not_applicable") : e.status;
    r.eisenstein_block = e.block_index;
    r.eisenstein_f = e.f;
    r.eisenstein_dim_socle = e.dim_socle;
    r.eisenstein_dim_socle_over_F = e.dim_socle_over_F;
    r.eisenstein_socle_equals_mV = e.socle_equals_mV;
    return r;
}

// [C] `obl:iota`(ii)'s rider ([OD] 2026-09-22 Q3).  A pure read of the
// certificate: `ran` is its `exact_relation_data`, and a row counts as
// COMPARED exactly where InvolutionCertificate::verify() clause 6 runs the
// Mat22 test -- the exact half ran, and the row and its matched image row
// both carry an exact group element.  Nothing here recomputes anything.
IotaExactComparisonRecord IotaExactComparison(const hecke::HeckeInvolution& involution) {
    ENGINE_CHECK_MSG(involution.computed(),
                     "HeckeExtComplexRecords: IotaExactComparison needs a computed involution");
    const hecke::InvolutionCertificate& cert = involution.certificate();
    IotaExactComparisonRecord r;
    r.present = true;
    r.ran = cert.exact_relation_data;
    r.relation_rows = static_cast<uint32_t>(cert.relations.size());
    for (size_t i = 0; i < cert.relations.size(); ++i) {
        const hecke::InvolutionCertificate::RelationRow& row = cert.relations[i];
        if (!row.has_exact) continue;
        ++r.rows_with_exact;
        if (!cert.exact_relation_data) continue;
        if (!row.matched || row.match_index >= cert.relations.size()) continue;
        if (!cert.relations[row.match_index].has_exact) continue;
        r.rows_compared.push_back(static_cast<uint32_t>(i));
    }
    return r;
}

std::vector<GroupRecord> SignGroups(const HeckeSignSplit& split) {
    std::vector<GroupRecord> out;
    if (split.certificate().status != SignSplitStatus::Split) return out;
    GroupRecord p;
    p.name = SignedGroupName(split.degree(), true);
    p.available = true;
    p.dim = split.dim_plus();
    out.push_back(p);
    GroupRecord m;
    m.name = SignedGroupName(split.degree(), false);
    m.available = true;
    m.dim = split.dim_minus();
    out.push_back(m);
    return out;
}

CanonicalPosition EisensteinCanonicalPosition(const HeckeSignSplit& split,
                                              const std::string& group) {
    CanonicalPosition cp;
    cp.group = group;
    const EisensteinLine& e = split.eisenstein_line();
    if (!e.applicable) {
        cp.eisenstein = "n/a";
    } else if (e.status == "defined") {
        // [E] D6: soc(X^+_{4,m}) = m_S X^+_{4,m}, a LINE, the bottom Loewy
        // layer -- per sign, and undefined on the unsplit H^1 where the
        // Eisenstein and congruent cuspidal reductions are not separable.
        cp.eisenstein = e.socle_equals_mV ? "socle, line, bottom Loewy layer"
                                          : "socle, line";
    } else {
        cp.eisenstein = "undefined (dim_F socle = " +
                        std::to_string(e.dim_socle_over_F) + ")";
    }
    return cp;
}

} // namespace hecke_ext
