// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// RegisterHeckeExtensionVerbsDetail.h -- the PRIVATE header of the extension
// layer's console registration TUs (S24 CL-8; CODEAUDIT A10).
//
// RegisterHeckeExtensionVerbs.cpp was one 3584-line TU.  It is now split by
// area into seven TUs that share this header:
//   RegisterHeckeExtensionVerbs.cpp         the two public entry points,
//                                           RegisterC5 (every C5 verb row, in
//                                           its old order) and the C5 helpers
//                                           declared below;
//   RegisterHeckeExtensionVerbsSturm.cpp    sturm_bound, u_ell_decision,
//                                           bernoulli_mod, glover_factors,
//                                           canonical_field_polynomial, `Sturm`;
//   RegisterHeckeExtensionVerbsAlgebra.cpp  the local-algebra verbs,
//                                           check_identities, L_map,
//                                           hecke_ring_algebra;
//   RegisterHeckeExtensionVerbsRecords.cpp  ExtRecord / MaximalIdeal /
//                                           HeckeAlgebra / LoewyProfile /
//                                           ExtensionClass / Presentation;
//   RegisterHeckeExtensionVerbsValues.cpp   Atlas, GF / GFElt, Settings;
//   RegisterHeckeExtensionVerbsComplex.cpp  Involution, BoundaryCohomology,
//                                           ConeCohomology, DicksonLES, theta;
//   RegisterHeckeExtensionVerbsGallery.cpp  lfunction and the [PG]
//                                           sec:pg-missing verbs (kummer_unit,
//                                           hida_tower, level_n, nebentypus,
//                                           forms_algebra, hida_lambda).
// The registration ORDER is unchanged: every register_type / register_verb
// call still runs from the same function body in the same sequence, so the
// registry, the console transcripts and Console/serve/read_only_verbs.tsv do
// not move.  Only what crosses a TU boundary is declared here (namespace
// verbs_detail); everything else stays in each TU's anonymous namespace.
//
// THIS HEADER INCLUDES Console/ HEADERS.  It and the seven .cpp files are the
// exact-path rule-2 exceptions of check_heckeext_includes.sh (ALLOW_EXT_CONSOLE)
// and check_console_includes.sh (ALLOW_CONSOLE), reason "[OD] slot-3 close-out
// 1".  Only those seven TUs may include it; the PUBLIC header
// RegisterHeckeExtensionVerbs.h still includes no Console/ header.
// ===========================================================================
#pragma once
#ifndef HECKE_EXT_REGISTER_HECKE_EXTENSION_VERBS_DETAIL_H
#define HECKE_EXT_REGISTER_HECKE_EXTENSION_VERBS_DETAIL_H

#include "GloverFactors.h"
#include "HeckeAlgebraInvariants.h"
#include "HeckeAtlas.h"
#include "HeckeCanonicalField.h"
#include "HeckeDicksonMultiplication.h"
#include "HeckeDicksonTheta.h"
#include "HeckeDicksonThetaHecke.h"
#include "HeckeExtBindingSupport.h"
#include "HeckeExtComplexRecords.h"
#include "HeckeExtDriver.h"
#include "HeckeExtReport.h"
#include "HeckeFormsHeckeAlgebra.h"
#include "HeckeHidaLambda.h"
#include "HeckeHidaTower.h"
#include "HeckePredictions.h"
#include "HeckeRingHeckeAlgebra.h"
#include "HeckeSturmBound.h"
#include "HeckeUEllBudget.h"
#include "../NumberField/Kummer.h"
#include "LValue/LFunctionRecord.h"
#include "../Console/Bindings/Descent.h"

#include "../Console/Bindings/Bindings.h"
#include "../Console/Bindings/Payloads.h"
#include "../Console/Core/ConsoleError.h"
#include "../Console/Core/Object.h"
#include "../Console/Core/Registry.h"
#include "../Console/Core/Session.h"
#include "../Console/Core/Settings.h"
#include "../Console/Core/Value.h"

#include "../Hecke/CoefficientRing.h"
#include "../Hecke/HeckeLevelN.h"
#include "../Hecke/HeckeLevelNHeilbronn.h"
#include "../Hecke/HeckeLevelNSparse.h"
#include "../Hecke/HeckeNebentypus.h"
#include "../Hecke/HeckeOperatorFamily.h"
#include "../Hecke/HeckeOrdinaryPart.h"
#include "../Hecke/HeckeSettings.h"

#include "../Util/TextIo.h"

#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/Fnv.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <numeric>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace hecke_ext {
namespace verbs_detail {

using console::CallContext;
using console::Object;
using console::ObjectPtr;
using console::Param;
using console::Record;
using console::Registry;
using console::ResumableState;
using console::RuntimeError;
using console::Signature;
using console::SourcePos;
using console::TypeError;
using console::TypeInfo;
using console::Value;
using console::ValueKind;

constexpr const char* kTypeSturm = "Sturm";

constexpr const char* kTypeExtRecord     = "ExtRecord";
constexpr const char* kTypeMaximalIdeal  = "MaximalIdeal";
constexpr const char* kTypeHeckeAlgebra  = "HeckeAlgebra";
constexpr const char* kTypeLoewyProfile  = "LoewyProfile";
constexpr const char* kTypeExtension     = "ExtensionClass";
constexpr const char* kTypePresentation  = "Presentation";
constexpr const char* kTypeAtlas         = "Atlas";
constexpr const char* kTypeGF            = "GF";
constexpr const char* kTypeGFElt         = "GFElt";
constexpr const char* kTypeSettings      = "Settings";
constexpr const char* kTypeInvolution    = "Involution";
constexpr const char* kTypeBoundaryCoh   = "BoundaryCohomology";
constexpr const char* kTypeConeCoh       = "ConeCohomology";
constexpr const char* kTypeDicksonLES    = "DicksonLES";   // slot-12 WP-8

using Flat = FlatMatrixT<uint32_t>;

using VerbFn = Value(CallContext& ctx, const std::vector<Value>& a);

// ---------------------------------------------------------------------------
// The C5 helpers (defined in RegisterHeckeExtensionVerbs.cpp).
// ---------------------------------------------------------------------------

// The registered types this section hands out from member getters (which see
// no CallContext).  PER REGISTRY: a validation builds its own Registry (and a
// script may run one through run_validation while an outer session is alive),
// so one process-wide set of TypeInfo pointers would be overwritten by -- and
// then dangle after -- an inner registry.  Each registration therefore files
// its set under every one of ITS type pointers, and a getter finds the set
// through its owner's own type, which belongs to a live registry.
struct C5Types {
    const TypeInfo* matrix = nullptr;
    const TypeInfo* poly = nullptr;
    const TypeInfo* ideal = nullptr;
    const TypeInfo* algebra = nullptr;
    const TypeInfo* loewy = nullptr;
    const TypeInfo* extension = nullptr;
    const TypeInfo* presentation = nullptr;
    const TypeInfo* gf = nullptr;
    const TypeInfo* gfelt = nullptr;
    const TypeInfo* settings = nullptr;
};
std::mutex& TypesMutex();
std::map<const TypeInfo*, std::shared_ptr<const C5Types>>& TypesMap();
const C5Types& TypesOf(const ObjectPtr& owner);

Value U32List(const std::vector<uint32_t>& xs);
Value U64List(const std::vector<uint64_t>& xs);
Value StringList(const std::vector<std::string>& xs);

// A matrix as a console Matrix value owned by `owner` (a member getter).
Value MatrixFrom(const ObjectPtr& owner, const ExtMatrix& m);
Value FlatFrom(const ObjectPtr& owner, const Flat& m);
Value PolyFrom(const ObjectPtr& owner, const hecke::FpPoly& f);
Value MatrixListFrom(const ObjectPtr& owner, const std::vector<ExtMatrix>& ms);
// A child handle of `type` over `payload`, born computed, keeping `owner` alive.
ObjectPtr ChildOf(const TypeInfo* type, const ObjectPtr& owner, std::shared_ptr<void> payload);

uint32_t DegreeArgC5(const Value& v, const SourcePos& pos, const char* who, size_t n);

// Engine exceptions from a member getter / method become ordinary runtime
// errors naming the member, never a raw std::exception text without context.
template <class F>
Value Guard(const char* who, const SourcePos& pos, F&& f) {
    try {
        return f();
    } catch (const console::ConsoleError&) {
        throw;
    } catch (const hecke::NotImplementedInThisBuild& e) {
        throw console::NotImplementedError(std::string(who) + ": " + e.what(), pos);
    } catch (const std::exception& e) {
        throw RuntimeError(std::string(who) + ": " + e.what(), pos);
    }
}

// The pipeline verbs of this section refuse a non-default coefficient_ring at
// dispatch ([E] sec:ring (b); exit 5 through the seam).
void RequireField(const char* who);

// The Cohomology handle's pipeline objects (Console/Bindings/Payloads.h).
struct CohSource {
    const EquivariantCohomologyOnK* coh = nullptr;
    const SimplicialComplex* K = nullptr;
    const hecke::SL2ZTessellationTables* model = nullptr;
    uint32_t ell = 0, n = 0;
};
CohSource SourceOf(const ObjectPtr& H, const SourcePos& pos, const char* who);

// ---------------------------------------------------------------------------
// Per area: what the other TUs call.
// ---------------------------------------------------------------------------

// RegisterHeckeExtensionVerbsSturm.cpp
void RegisterSturmTypes(Registry& registry);
VerbFn VerbSturmBound, VerbUEllDecision, VerbBernoulliMod, VerbGloverFactors, VerbCanonicalFieldPolynomial;

// RegisterHeckeExtensionVerbsAlgebra.cpp
Value InvariantsRecord(const AlgebraInvariants& A);
VerbFn VerbNilpotentPartOne, VerbNilpotentPartList, VerbRadical, VerbTangentSpace, VerbSocle,
    VerbNilpotencyIndex, VerbIsLocal, VerbAlgebraInvariants, VerbLoewy, VerbSocleSeries, VerbCheckIdentities,
    VerbLMap, VerbHeckeRingAlgebra;

// RegisterHeckeExtensionVerbsRecords.cpp
struct HeckeAlgebraVal;
const HeckeAlgebraVal* AsHeckeAlgebra(const Value& v);
std::vector<ExtMatrix> HeckeAlgebraGenerators(const HeckeAlgebraVal& a);
void RegisterExtRecordTypes(Registry& registry);
VerbFn VerbExtRecord, VerbLocalAlgebra, VerbModuleStructure, VerbPresentation;

// RegisterHeckeExtensionVerbsValues.cpp
Value SettingsHandleFrom(const ObjectPtr& owner, const hecke::MathSettings& s);
void RegisterValueTypes(Registry& registry);
VerbFn VerbAtlas, VerbGF, VerbGFAdd, VerbGFSub, VerbGFMul, VerbGFInv, VerbGFPow, VerbGFFrobenius, VerbGFEval,
    VerbSettingsSnapshot;

// RegisterHeckeExtensionVerbsComplex.cpp
void RegisterComplexTypes(Registry& registry);
VerbFn VerbIota, VerbBoundary, VerbCone, VerbDicksonLES, VerbTheta;

// RegisterHeckeExtensionVerbsGallery.cpp
VerbFn VerbLFunction, VerbKummerUnit, VerbHidaTower, VerbLevelN, VerbNebentypus, VerbFormsAlgebra,
    VerbHidaLambda;

} // namespace verbs_detail
} // namespace hecke_ext

#endif // HECKE_EXT_REGISTER_HECKE_EXTENSION_VERBS_DETAIL_H
