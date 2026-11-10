// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Bindings.h -- console phase C1a: what every binding TU shares.
//
// Plan (docs/console/00-console-interface-plan.md v3) Sec. 9: the binding TUs
// are compiled into cohomology_app, NOT into ConsoleCore, which is what keeps
// the library STL-only (check_console_includes.sh rule 1).  They are the only
// place where a console type and a pipeline type meet.
//
// C1a binds Sec. 3.4 (Fp / Poly / Matrix), the Sec. 3.5 linear-algebra surface
// that needs no new substrate, and Sec. 3.6's Model / Complex / LocalSystem /
// Cohomology / HeckeOperator / Family.  It binds NO `decompose` and NO `report`
// verb: those objects are exactly the ones the extension stage's single golden
// re-baseline moves, so they wait for phase C1b (plan Sec. 10, "Why the split").
//
// Element width: every C1a binding is 32-bit (`uint32_t`), the width the whole
// golden grid and every pin uses.  The 64-bit regime is reachable through the
// pipeline's own templates and is a C2/C5 addition, recorded as a gap in the
// WP-7 report.
// ===========================================================================
#pragma once
#ifndef CONSOLE_BINDINGS_H
#define CONSOLE_BINDINGS_H

#include <cstdint>
#include <functional>
#include <map>
#include <set>
#include <memory>
#include <string>
#include <vector>

#include "../Core/ConsoleError.h"
#include "../Core/Object.h"
#include "../Core/Registry.h"
#include "../Core/Session.h"
#include "../Core/Value.h"

#include "../../../ModArithmetic/FlatMatrix.h"
#include "../../Hecke/FpPoly.h"

namespace hecke {
class HeckeModuleDecomposition;
}

namespace console {
namespace bind {

// --- the console type names (one spelling, used by every TU) ---------------
constexpr const char* kTypeFp        = "Fp";
constexpr const char* kTypePoly      = "Poly";
constexpr const char* kTypeMatrix    = "Matrix";
constexpr const char* kTypeModel     = "Model";
constexpr const char* kTypeComplex   = "Complex";
constexpr const char* kTypeLocalSys  = "LocalSystem";
constexpr const char* kTypeCohomology = "Cohomology";
constexpr const char* kTypeHecke     = "HeckeOperator";
constexpr const char* kTypeFamily    = "Family";
// (C1b) The decomposition surface -- the objects the extension stage's single
// golden re-baseline (E0b) moved, which is why they waited for phase C1b.
constexpr const char* kTypeDecomposition = "Decomposition";
constexpr const char* kTypeBlock         = "Block";
constexpr const char* kTypeReport        = "Report";
// (C2b, slot-7 WP-8) The harness surface's one handle type: a parsed campaign
// spec (Harness.cpp).
constexpr const char* kTypeCampaignSpec  = "CampaignSpec";
// (C5a, slot-9 WP-12) E7's two reserved types ([P] Sec. 3.6), Duality.cpp.
constexpr const char* kTypeContragredient = "ContragredientLocalSystem";
constexpr const char* kTypePairing        = "Pairing";

// --- payloads --------------------------------------------------------------
// Every one is held by shared_ptr<void> inside the handle; keep-alive edges
// (Object::keep_alive) carry the "must outlive this object" contracts the
// pipeline states in prose (plan Sec. 4.1).

struct FpVal {
    uint32_t p = 0;
    uint32_t a = 0;
};

struct PolyVal {
    hecke::FpPolyT<uint32_t> f;
};

struct MatrixVal {
    FlatMatrixT<uint32_t> m;
};

// --- helpers ---------------------------------------------------------------

// The registered TypeInfo of `name` on this call's registry.  Throws a
// CONSOLE_CHECK (a binding bug, never a user error) when it is missing.
const TypeInfo* TypeByName(CallContext& ctx, const char* name);

// A fresh, ALREADY-COMPUTED handle of `name` carrying `payload` -- the shape
// every value-like binding result (a matrix, a polynomial) takes: there is no
// lazy work to do, so the handle is born computed.
ObjectPtr MakeValueObject(CallContext& ctx, const char* type_name,
                          std::shared_ptr<void> payload);

// The same, where no CallContext is in hand -- a TypeInfo::Member getter, which
// receives only the owning handle.  The new handle inherits the owner's
// creation site and settings snapshot, which is what `?x` on a derived matrix
// should say: it was produced by the expression that produced its owner.
ObjectPtr MakeValueObjectFrom(const TypeInfo* type, const ObjectPtr& owner,
                              std::shared_ptr<void> payload);

// (console C4, slot-8 WP-10) The lazy-compute wrapper every binding hands to
// Object::create.  Plan Sec. 2 C7: `not_implemented_in_this_build` is its own
// outcome, exit 5, never exit 3.  An eager verb gets that from the registry's
// kind classifier (InstallExceptionSeam); a LAZY handle does not -- the core's
// Object::ensure_computed wraps a raw std::exception as an ordinary runtime
// error (exit 3), and the core may name no pipeline type.  So the binding,
// which may, converts a hecke::NotImplementedInThisBuild (or any exception
// whose what() starts with hecke::kNotImplementedInThisBuildTag) raised inside
// `body` into console::NotImplementedError, whose kind the handle wrapper
// preserves.  Every other exception is rethrown UNCHANGED, so no other error
// text moves.  Found by ConsoleErrorValidation row r06.
Object::ComputeFn LazyCompute(Object::ComputeFn body);

// (C5a, slot-9 WP-12; [E] v1.10.0-pre.1 sec:ring (h)) THE MEMBER GATE of a
// type whose handles may be built over Z/ell^k.  `rule(name)` classifies each
// member / method: Honour (its getter handles both rings itself), Field (the
// getter is the F_ell one and the object raises its own named
// std::logic_error on a ring handle, exit 3) or Refuse.  GateTypeOnRing wraps
// every Refuse getter so that, on a handle whose `ring_k(o) >= 2`, it raises
// hecke::NotImplementedInThisBuild naming "<Type>.<member>" and the ring (exit
// 5) BEFORE the original getter runs; on an F_ell handle the original getter
// runs unchanged.  A member registered later is therefore refused on a ring
// handle until (h)'s list is amended.
enum class RingRule { Honour, Field, Refuse };
void GateTypeOnRing(TypeInfo& t, const std::function<RingRule(const std::string&)>& rule,
                    const std::function<uint32_t(const ObjectPtr&)>& ring_k);

// (slot-11 WP-18c; [E] v1.12.0-pre.2 sec:ring (h)) THE DIGIT GATE, applied
// AFTER GateTypeOnRing.  On a handle for which `is_digit(o)` (a ring handle at
// ell^k >= 2^64, ell >= 5: the digit route Hecke/LadicCell, known at dispatch,
// forcing nothing) a member named in `pass_through` runs its (ring-gated)
// getter unchanged, a member / method named in `members` / `methods` runs the
// digit getter given there, a FIELD member / method (`field`: RingRule::Field,
// the F_ell-only accessors -- not defined over a ring at all) raises the same
// std::logic_error as on every ring handle (exit 3; WP-14 pass 2, M-1), and
// every other member / method refuses with hecke::NotImplementedInThisBuild
// naming "<Type>.<member>" and the ring (exit 5).  On every other handle
// nothing changes.
void GateTypeOnDigit(
    TypeInfo& t, const std::set<std::string>& pass_through, const std::set<std::string>& field,
    const std::map<std::string, std::function<Value(const ObjectPtr&)>>& members,
    const std::map<std::string, std::function<Value(const ObjectPtr&, const std::vector<Value>&, const SourcePos&)>>&
        methods,
    const std::function<bool(const ObjectPtr&)>& is_digit, const std::function<uint32_t(const ObjectPtr&)>& ring_k);

// (slot-12 WP-17; [P] v3.9.3 Q9 decided (a)) THE WIDE GATE, applied AFTER
// GateTypeOnDigit.  On a handle for which `is_wide(o)` (a ring handle at
// 2^32 <= ell^k < 2^64, ell >= 5: the native pipeline's uint64_t
// instantiation, the campaign job loop's own route) a member / method named in
// `members` / `methods` runs the uint64_t getter given there; EVERY other
// member / method runs its (ring-gated) getter unchanged -- so the refusal set
// (exit 5) is exactly GateTypeOnRing's, as on a uint32_t ring handle.  The
// caller lists every Honour AND every Field member (the latter calling the
// uint64_t object's own field accessor, which raises the object's named
// std::logic_error, exit 3).  On every other handle nothing changes.
void GateTypeOnWide(
    TypeInfo& t, const std::map<std::string, std::function<Value(const ObjectPtr&)>>& members,
    const std::map<std::string, std::function<Value(const ObjectPtr&, const std::vector<Value>&, const SourcePos&)>>&
        methods,
    const std::function<bool(const ObjectPtr&)>& is_wide, const std::function<uint32_t(const ObjectPtr&)>& ring_k);

// (slot-24 D6 audit m1; [C] sec:recordschema item 7 (a), [E] sec:ring (h))
// THE FIELD-VERB GATE.  Wraps every overload of each named verb so that, when
// the ring in force is not F_ell (`Z`, or `Z/ell^k` with k >= 2), it raises
// hecke::NotImplementedInThisBuild naming "console verb <name>" and the ring
// (exit 5) BEFORE the original impl runs -- at the verb's statement, nothing
// declared, nothing computed.  Under F_ell the impl runs unchanged.  It is for
// the verbs that compute over F_ell from scalar arguments and never read the
// ring themselves (a handle verb refuses inside its own impl); a name that is
// not registered is a defect (ENGINE_CHECK-style std::logic_error), so the
// list cannot go stale silently.
void GateVerbsOnField(Registry& registry, const std::vector<std::string>& names);

// (slot-12 WP-17) 2^32 <= ell^k < 2^64 with k >= 2: the wide band.
bool IsWideCell(uint64_t ell, uint32_t k);
// A residue of the wide band as a console value: an Int when it fits int64_t,
// else its decimal String (ell^k may exceed 2^63; the console has no bignum).
Value WideIntValue(uint64_t x);

// Payload access with the type checked by name, so a wrong overload is a
// console type error and never a bad cast.
const MatrixVal& AsMatrix(const Value& v, const SourcePos& pos, const char* who);
const PolyVal& AsPoly(const Value& v, const SourcePos& pos, const char* who);
const FpVal& AsFp(const Value& v, const SourcePos& pos, const char* who);
const ObjectPtr& AsObjectOf(const Value& v, const char* type_name, const SourcePos& pos,
                            const char* who);

// Value constructors.
Value MatrixValue(CallContext& ctx, FlatMatrixT<uint32_t> m);
Value PolyValue(CallContext& ctx, hecke::FpPolyT<uint32_t> f);
Value FpValue(CallContext& ctx, uint32_t p, uint32_t a);
Value IntList(const std::vector<uint32_t>& xs);

// An Int argument that must be a non-negative value below 2^32 (a degree, a
// dimension, a prime).  Throws a TypeError naming `who` otherwise.
uint32_t U32Arg(const Value& v, const SourcePos& pos, const char* who);
uint64_t U64Arg(const Value& v, const SourcePos& pos, const char* who);

// The prime a verb was handed must be the prime of every matrix it touches.
void RequireSameModulus(const FlatMatrixT<uint32_t>& a, const FlatMatrixT<uint32_t>& b,
                        const SourcePos& pos, const char* who);

// --- the per-kind golden hash (plan Sec. 2 C5) -----------------------------
// Byte for byte the definition Golden/GoldenHarness.cpp uses, over the ONE
// FNV-1a-64 of src/ModArithmetic/Fnv.h: a matrix as rows, cols, then
// every entry widened to uint64 in row-major order; a polynomial as the
// coefficient count then the coefficients LOW -> HIGH; a scalar invariant as
// that single uint64.  That is what makes `hash(T.A(1))` comparable with
// `golden("n=10 ell=691 p=2 r=1", "A(T_p)^1").hash` at all.
uint64_t HashMatrix(const FlatMatrixT<uint32_t>& m);
uint64_t HashPoly(const hecke::FpPolyT<uint32_t>& f);
uint64_t HashScalarValue(uint64_t v);
uint64_t HashMatrixList(const std::vector<FlatMatrixT<uint32_t>>& ms);

// --- the registration functions (called by RegisterPipelineVerbs) ----------
void RegisterCoreMathBindings(Registry& registry);
void RegisterModelBindings(Registry& registry);
void RegisterCohomologyBindings(Registry& registry);
void RegisterHeckeBindings(Registry& registry);
// (C1b) `decompose` / `report` / `crosschecks` + the Decomposition / Block /
// Report types, and the engine-owned MATHEMATICAL settings installer.
void RegisterDecompositionBindings(Registry& registry);
// `--serve` only: makes the file-writing method Report.write(path) throw.
void ForbidReportWrite();
// (C1b) The LocalSystemPresets local systems (S^1 / cylinder / Moebius band).
void RegisterPresetBindings(Registry& registry);
void RegisterReproducibilityBindings(Registry& registry);
void InstallExceptionSeam(Registry& registry);
// (C2a, slot-6 WP-13) The ENGINE KNOBS as `set` keys ([P] Sec. 2 C4, table
// "Engine knobs"), through the registry's settings-installer seam.  Registered
// after the mathematical settings, so `settings()` lists them after those.
void RegisterEngineSettingsBindings(Registry& registry);
// (C2b, slot-7 WP-8) The harness verbs ([P] Sec. 5, Sec. 10 C2b):
// validations / run_validation / bench / micro / campaign_spec /
// campaign_spec_text / campaign_plan / run_campaign / golden_subset / timed /
// bench_line / rss, and the CampaignSpec type.  Harness.cpp.
void RegisterHarnessBindings(Registry& registry);
// (C4, slot-8 WP-10) `help_markdown()` -- the generated reference manual,
// docs/console/console.md ([P] Sec. 4.4's `help --markdown`; Reference.cpp).
void RegisterReferenceBindings(Registry& registry);
// (C5a, slot-9 WP-12) E7: `dual(L)` / `pairing(L, D)` and the
// ContragredientLocalSystem / Pairing types (Duality.cpp).
void RegisterDualityBindings(Registry& registry);
// Their names, in registration order (for the registration arm and `help`).
const std::vector<std::string>& HarnessVerbNames();
// (C2b) Test seam for ConsoleDeterminismValidation's acceptance-(c) arm: called
// INSIDE every harness verb's lifted-settings scope, just before the work, so
// a validation can observe the engine state a run_validation actually runs
// under.  NOT a verb; empty (the default) means no probe.
void SetHarnessLiftProbe(std::function<void()> probe);

// (C2a, slot-6 WP-13) Test seams for ConsoleSettingsValidation's resume arm
// ([P] Sec. 10 C2 (h)).  NOT verbs: a validation registers its own test-only
// verb around the first.
//
//  * MakeMatrixDecomposition -- a REAL `Decomposition` handle (the shipped
//    type, its compute, its `resume`) over an explicit commuting family, so
//    the resume arm can reach a state the golden grid never produces: a block
//    whose locality a small budget cannot decide ([E] Claim 3.1's family).
//  * DecompositionOf -- the handle's HeckeModuleDecomposition, or nullptr while
//    it is still `declared`; for entrywise LocalityState comparisons.
ObjectPtr MakeMatrixDecomposition(CallContext& ctx,
                                  std::vector<std::pair<uint64_t, MatrixT<uint32_t>>> ops,
                                  uint32_t ell);
const hecke::HeckeModuleDecomposition* DecompositionOf(const ObjectPtr& o);

// Signature sugar, as CoreVerbs.cpp spells it.
Signature Sig(std::vector<Param> params, bool variadic = false);

} // namespace bind
} // namespace console

#endif // CONSOLE_BINDINGS_H
