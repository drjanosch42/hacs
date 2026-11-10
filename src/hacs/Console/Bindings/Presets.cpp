// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Presets.cpp -- console phase C1b: the LOCAL-SYSTEM presets of
// LocalSystemPresets.h:15-23 ([P] v3.1 Sec. 3.6, row LocalSystem: "presets
// s1_scalar, s1_unipotent, cylinder_*, moebius_*"; Sec. 10 Phase C1b, "a
// `presets` script over the S^1 / cylinder / Moebius local systems").
//
// C1a bound the trivial-coefficient COMPLEX presets (sphere / cylinder /
// moebius / projective_plane, Model.cpp); these are the LOCAL SYSTEMS on them,
// which is what makes the equivariant machinery exercisable away from the
// SL(2,Z) model -- a monodromy that is a scalar, and one that is unipotent and
// therefore genuinely non-semisimple.
//
// THEY CARRY NO TESSELLATION MODEL.  A preset LocalSystem's `model` handle is
// null, its `lssc` is built eagerly here, and `hecke` / `family` refuse it with
// the named error of Hecke.cpp's RequireSL2ZModel: the Hecke action is defined
// on the 7-vertex model only.  `cohomology(L)` works exactly as it does for a
// symmetric power, which is the point of the script.
//
// `LocalSystem.n` on a preset is `d_V - 1` and has no Sym^n meaning; the verb
// docs say so, and the preset script reads `.d_V`, never `.n`.
// ===========================================================================

#include "Bindings.h"
#include "Payloads.h"

#include "../../LocalSystemPresets.h"

#include <memory>
#include <string>
#include <vector>

namespace console {
namespace bind {

namespace {

// One preset verb: it builds the LocalSystemSimplicialComplex NOW (they are all
// tiny -- a circle, a cylinder or a Moebius band with d_V <= 2) and hands back
// an already-computed handle, exactly like every other value-like binding.
Value MakePreset(CallContext& ctx, const char* name, LocalSystemSimplicialComplex lssc) {
    auto p = std::make_shared<LocalSystemVal>();
    p->preset = name;
    p->ell = lssc.get_prime();
    p->n_sym = lssc.get_representation_dim() - 1;
    p->lssc.reset(new LocalSystemSimplicialComplex(std::move(lssc)));
    ObjectPtr obj = Object::create(TypeByName(ctx, kTypeLocalSys), ctx.pos, ctx.call_text,
                                   ctx.session->settings().snapshot(), p, nullptr);
    obj->ensure_computed(ctx.session->settings(), ctx.pos);   // born computed
    return Value::from_object(obj);
}

uint32_t PrimeArg(const Value& v, const SourcePos& pos, const char* who) {
    const uint32_t ell = U32Arg(v, pos, who);
    if (ell < 2) throw TypeError(std::string(who) + ": ell must be a prime >= 2", pos);
    return ell;
}

Value VerbS1Scalar(CallContext& ctx, const std::vector<Value>& a) {
    const uint32_t s = U32Arg(a[0], ctx.pos, "s1_scalar(scalar, ell)");
    const uint32_t ell = PrimeArg(a[1], ctx.pos, "s1_scalar(scalar, ell)");
    return MakePreset(ctx, "s1_scalar", LocalSystemPresets::CreateS1Scalar(s % ell, ell));
}

Value VerbS1Unipotent(CallContext& ctx, const std::vector<Value>& a) {
    const uint32_t ell = PrimeArg(a[0], ctx.pos, "s1_unipotent(ell)");
    return MakePreset(ctx, "s1_unipotent", LocalSystemPresets::CreateS1Unipotent2D(ell));
}

Value VerbCylinderScalar(CallContext& ctx, const std::vector<Value>& a) {
    const uint32_t s = U32Arg(a[0], ctx.pos, "cylinder_scalar(scalar, ell)");
    const uint32_t ell = PrimeArg(a[1], ctx.pos, "cylinder_scalar(scalar, ell)");
    return MakePreset(ctx, "cylinder_scalar",
                      LocalSystemPresets::CreateCylinderScalar(s % ell, ell));
}

Value VerbCylinderUnipotent(CallContext& ctx, const std::vector<Value>& a) {
    const uint32_t ell = PrimeArg(a[0], ctx.pos, "cylinder_unipotent(ell)");
    return MakePreset(ctx, "cylinder_unipotent",
                      LocalSystemPresets::CreateCylinderUnipotent2D(ell));
}

Value VerbMoebiusScalar(CallContext& ctx, const std::vector<Value>& a) {
    const uint32_t s = U32Arg(a[0], ctx.pos, "moebius_scalar(scalar, ell)");
    const uint32_t ell = PrimeArg(a[1], ctx.pos, "moebius_scalar(scalar, ell)");
    return MakePreset(ctx, "moebius_scalar", LocalSystemPresets::CreateMoebiusScalar(s % ell, ell));
}

Value VerbMoebiusUnipotent(CallContext& ctx, const std::vector<Value>& a) {
    const uint32_t ell = PrimeArg(a[0], ctx.pos, "moebius_unipotent(ell)");
    return MakePreset(ctx, "moebius_unipotent",
                      LocalSystemPresets::CreateMoebiusUnipotent2D(ell));
}

} // namespace

void RegisterPresetBindings(Registry& registry) {
    const char* g = "pipeline";
    auto add = [&registry, g](const char* name, Signature sig, VerbImpl impl, const char* doc) {
        registry.register_verb(name, std::move(sig), std::move(impl), doc, g);
    };

    add("s1_scalar",
        Sig({Param::of("scalar", ValueKind::Int), Param::of("ell", ValueKind::Int)}),
        &VerbS1Scalar,
        "the rank-1 local system on S^1 with scalar monodromy (a preset: no Hecke action)");
    add("s1_unipotent", Sig({Param::of("ell", ValueKind::Int)}), &VerbS1Unipotent,
        "the rank-2 unipotent local system on S^1 (a preset: no Hecke action)");
    add("cylinder_scalar",
        Sig({Param::of("scalar", ValueKind::Int), Param::of("ell", ValueKind::Int)}),
        &VerbCylinderScalar, "the scalar-monodromy local system on the cylinder");
    add("cylinder_unipotent", Sig({Param::of("ell", ValueKind::Int)}), &VerbCylinderUnipotent,
        "the rank-2 unipotent local system on the cylinder");
    add("moebius_scalar",
        Sig({Param::of("scalar", ValueKind::Int), Param::of("ell", ValueKind::Int)}),
        &VerbMoebiusScalar, "the scalar-twist local system on the Moebius band");
    add("moebius_unipotent", Sig({Param::of("ell", ValueKind::Int)}), &VerbMoebiusUnipotent,
        "the rank-2 unipotent local system on the Moebius band");
}

} // namespace bind
} // namespace console
