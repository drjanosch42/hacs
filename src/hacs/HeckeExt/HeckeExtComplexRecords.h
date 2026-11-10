// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_COMPLEX_RECORDS_H
#define HECKE_EXT_COMPLEX_RECORDS_H

// ===========================================================================
// hecke-extensions phase E2 cont. (slot 5, WP-1) -- THE RECORD LAYER'S VIEW OF
// THE E3 / E4b / E5 OBJECTS.
//
// [X] = docs/implementation/hecke_extensions_implementation_plan.md v1.8
// Sec. 1.4, Sec. 6.2 E3 / E5; [E] = docs/latex/hecke_extensions.tex v1.5.2
// `sec:schema`; [C] = docs/latex/gl2_contract.tex v1.6.2 `sec:recordschema`
// items 5 and 6, cited by label.
//
// WHY THIS FILE EXISTS.  `hecke_ext/v2` is about THREE complexes -- the bulk,
// the boundary and the pair -- and about the involution's split, and every one
// of those is built by a unit that knows nothing about the record.  Somebody
// has to know both, and it is the record layer: these are pure, allocating
// translations from a COMPUTED pipeline object to the record fragment that
// describes it, and they are the only place the two vocabularies meet.
//
// THE DEGREE FIELDS ARE THE COMPLEX'S OWN.  Every `ComplexRecord` a function
// here returns takes its `num_degrees` from the object it is about and from
// nothing else -- never a maximum over the record, never a literal ([C]
// `sec:recordschema` item 5; the `HeckeReportInput::num_degrees` lesson, which
// is why that writer REFUSES a record that cannot name its complex).  The cone
// additionally carries `derived_from`, because a complex built from others
// carries `q_max_source = "derived"` unconditionally TOGETHER WITH the names
// of its inputs (item 5, "Maps, and complexes built from complexes").
//
// NOTHING HERE COMPUTES.  Every argument is an already-computed object; these
// functions read it.  In particular none of them builds a cone, a Hecke
// operator or an involution: what a caller attaches is the caller's cost.
//
// PLACEMENT.  HeckeExt/, `MatrixT<uint32_t>` where a matrix appears at all.
// It includes pipeline headers exactly as HeckeExt/HeckeBoundary.h does
// (guard rule 1 is about the reverse direction); no Console/, no Cuda/, no
// Engine/, no BlockFlatMatrix.h, no `unordered_`.
// ===========================================================================

#include "HeckeExtRecord.h"
#include "HeckeSignSplit.h"

#include "../EquivariantMappingCone.h"
#include "../Hecke/HeckeConeAction.h"
#include "../Hecke/HeckeInvolution.h"

#include <cstdint>
#include <string>
#include <vector>

namespace hecke_ext {

// ---------------------------------------------------------------------------
// The group names the record uses.  ONE definition, so that a consumer reading
// "H1c" in one record and "H1c" in another is reading the same thing, and so
// that [E] sec:schema's closed list of four INTEGER scalars stays
// distinguishable from the groups around it.
// ---------------------------------------------------------------------------
std::string BoundaryGroupName(uint32_t q);        // "H0_bd", "H1_bd"
std::string ConeGroupName(uint32_t q);            // "H0c", "H1c", "H2c"
std::string SignedGroupName(uint32_t q, bool plus);  // "H1_plus" / "H1_minus"

// ---------------------------------------------------------------------------
// E5 -- the cone.  ONE computed `EquivariantMappingCone` carries everything
// `complexes.boundary`, `complexes.pair`, the cone groups, `les`, the
// boundary groups and `dim_H1_shriek` need, so the record takes them from it
// and does not re-derive any of them from a second object.
// ---------------------------------------------------------------------------

// `complexes.boundary`: the boundary complex's OWN degree fields, plus the two
// labels the generic cone was told by Hecke/SL2ZBoundaryPair.
ComplexRecord BoundaryComplexRecord(const EquivariantMappingCone& cone);

// `complexes.pair`: the cone's own degree fields, `derived_from` = {bulk,
// boundary}, and one `les_maps` entry per spot the cone recorded.
ComplexRecord ConeComplexRecord(const EquivariantMappingCone& cone);

// The boundary's groups ("H0_bd", ...) and the cone's ("H0c", ...).
std::vector<GroupRecord> BoundaryGroups(const EquivariantMappingCone& cone);
std::vector<GroupRecord> ConeGroups(const EquivariantMappingCone& cone);

// [E] Sec. 7's `les` summary.  `exact` is the conjunction over the ASSERTED
// spots; `hecke_equivariant` is not the cone's to know (it is the action's)
// and is therefore an argument, defaulted to "not claimed".
struct LesSummary {
    bool available          = false;
    bool exact              = false;
    bool hecke_equivariant  = false;
    std::vector<std::pair<std::string, uint32_t>> ranks;   // "r0", "r1", ...
};
LesSummary ConeLesSummary(const EquivariantMappingCone& cone,
                          bool hecke_equivariant = false);

// dim H^1_! = dim ker r^1 ([C] `cl:les`(2)), which is the fourth scalar of
// [E] sec:schema's closed list.  Returns false when the cone did not build
// r^1, in which case the scalar stays `null`.
bool ConeDimH1Shriek(const EquivariantMappingCone& cone, uint32_t* out);

// ---------------------------------------------------------------------------
// E5-3 -- the cone ACTION.  It contributes exactly the two RECORDED flags of
// [C] `sec:recordschema` item 6(e) and the name of its own standalone record,
// all three inside the `pair` object.  Nothing here is branched on (rule A6).
// ---------------------------------------------------------------------------
void AttachConeActionRecord(ComplexRecord& pair, const hecke::HeckeConeAction& action);

// ---------------------------------------------------------------------------
// E3 -- the involution.  One record per degree, from the ext layer's own
// HeckeSignSplit (which is where [C] `obl:iota`(v) and [E] D6 are decided).
// `charpoly_iota` is the characteristic polynomial of the stored iota^*,
// low -> high (A-N4).
// ---------------------------------------------------------------------------
IotaDegreeRecord IotaRecord(const HeckeSignSplit& split);

// [C] `obl:iota`(ii)'s RIDER ([OD] 2026-09-22 Q3).  The rider's recording half
// -- whether the exact Mat22 comparison happened and on which relation rows --
// read off the PRODUCTION involution's own certificate.  It is a property of
// the cell (one HeckeInvolution discharges obl:iota(ii) for the whole
// complex), so it is taken from the involution and NOT from a HeckeSignSplit,
// which is per degree and carries no relation rows.
IotaExactComparisonRecord IotaExactComparison(const hecke::HeckeInvolution& involution);

// The per-sign groups of one degree, from the same object.
std::vector<GroupRecord> SignGroups(const HeckeSignSplit& split);

// [E] D6 / [E] Sec. 7's `canonical` object for one ambient group, as far as
// the involution decides it: the Eisenstein line's position.  The three other
// slots keep their defaults, which a later phase fills.
CanonicalPosition EisensteinCanonicalPosition(const HeckeSignSplit& split,
                                              const std::string& group);

} // namespace hecke_ext

#endif // HECKE_EXT_COMPLEX_RECORDS_H
