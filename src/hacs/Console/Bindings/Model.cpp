// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Model.cpp -- console phase C1a: `sl2z_model`, the simplicial presets and the
// `Complex` type (plan Sec. 3.6, rows Model and Complex).
//
// The Model handle OWNS its SimplicialComplex and the SL2ZTessellationTables
// built from it, in one payload with a stable address, because the tables
// borrow the complex (SL2ZTessellationTables.h:114) and HeckeTauTable borrows
// both (HeckeTauTable.h:330-331).  That is the plan's Sec. 4.1 rule -- "a
// handle keeps alive everything the C++ object may dereference, transitively"
// -- discharged by ownership rather than by an edge, which is stronger.
// ===========================================================================

#include "Bindings.h"
#include "Payloads.h"

#include "../../Hecke/HeckeModelDescriptor.h"
#include "../../Hecke/SL2ZTessellationTables.h"
#include "../../LocalSystemSL2Z.h"
#include "../../SimplicialCohomology.h"

#include <sstream>
#include <string>

namespace console {
namespace bind {

namespace {

Value VerbSl2zModel(CallContext& ctx, const std::vector<Value>& a) {
    std::string name = "sl2z_7vertex";
    if (!a.empty() && !a[0].is_nil()) {
        if (!a[0].is_string()) throw TypeError("sl2z_model(name): name must be a String", ctx.pos);
        name = a[0].string_value();
    }
    if (name != "sl2z_7vertex") {
        // The descriptor registry is the single source of truth for which
        // models exist (HeckeModelDescriptor.h:74); Gamma_0(N) / Gamma(2) is
        // inventory gap G15 and arrives with its own model.
        throw RuntimeError("sl2z_model(name): unknown model '" + name +
                               "'; this build ships only 'sl2z_7vertex' (see models())",
                           ctx.pos);
    }
    auto p = std::make_shared<ModelVal>();
    p->name = name;
    p->K = CreateSL2ZFundamentalDomainTriangulation();
    p->model.reset(new hecke::SL2ZTessellationTables(p->K));
    return Value::from_object(MakeValueObject(ctx, kTypeModel, std::move(p)));
}

Value VerbModels(CallContext& ctx, const std::vector<Value>&) {
    (void)ctx;
    std::vector<Value> out;
    out.push_back(Value::from_string("sl2z_7vertex"));
    return Value::from_list(std::move(out));
}

// --- the Complex presets (plan Sec. 3.6, SimplicialCohomology.h:233-243) ----

Value MakeComplex(CallContext& ctx, SimplicialComplex K) {
    auto p = std::make_shared<ComplexVal>();
    p->K = std::move(K);
    return Value::from_object(MakeValueObject(ctx, kTypeComplex, std::move(p)));
}

Value VerbSphere(CallContext& ctx, const std::vector<Value>& a) {
    const uint32_t n = U32Arg(a[0], ctx.pos, "sphere(n)");
    switch (n) {
        case 1: return MakeComplex(ctx, SimplicialOneSphere());
        case 2: return MakeComplex(ctx, SimplicialTwoSphere());
        case 3: return MakeComplex(ctx, SimplicialThreeSphere());
        default:
            throw RuntimeError("sphere(n): this build ships n = 1, 2, 3", ctx.pos);
    }
}

Value VerbCylinder(CallContext& ctx, const std::vector<Value>&) {
    return MakeComplex(ctx, SimplicialCylinder());
}
Value VerbMoebius(CallContext& ctx, const std::vector<Value>&) {
    return MakeComplex(ctx, SimplicialMoebiusBand());
}
Value VerbProjectivePlane(CallContext& ctx, const std::vector<Value>&) {
    return MakeComplex(ctx, SimplicialProjectivePlane());
}

Value VerbComplexFromLists(CallContext& ctx, const std::vector<Value>& a) {
    // complex([[...], ...], name) -- SimplicialCohomology.h:63.  The outer list
    // is by dimension, the middle one the simplices of that dimension, the
    // inner one the (sorted) vertex list.
    std::vector<std::vector<std::vector<uint32_t>>> data;
    for (const Value& by_dim : a[0].list_value()) {
        if (!by_dim.is_list()) {
            throw TypeError("complex(simplices, name): the outer list is by dimension", ctx.pos);
        }
        std::vector<std::vector<uint32_t>> dim;
        for (const Value& s : by_dim.list_value()) {
            if (!s.is_list()) {
                throw TypeError("complex(simplices, name): every simplex is a List of Int", ctx.pos);
            }
            std::vector<uint32_t> verts;
            for (const Value& v : s.list_value()) verts.push_back(U32Arg(v, ctx.pos, "complex"));
            dim.push_back(std::move(verts));
        }
        data.push_back(std::move(dim));
    }
    if (!a[1].is_string()) throw TypeError("complex(simplices, name): name must be a String", ctx.pos);
    return MakeComplex(ctx, SimplicialComplex(data, a[1].string_value()));
}

// --- types -----------------------------------------------------------------

void RegisterTypes(Registry& registry) {
    {
        TypeInfo t;
        t.name = kTypeModel;
        t.doc = "the SL(2,Z) orbifold model: its triangulation and the exact tessellation tables";
        t.members.push_back({"name", false, "the descriptor name", [](const ObjectPtr& o) {
                                 return Value::from_string(o->payload_as<ModelVal>()->name);
                             }});
        t.members.push_back({"dims", false, "the simplex counts by dimension",
                             [](const ObjectPtr& o) {
                                 const ModelVal* m = o->payload_as<ModelVal>();
                                 std::vector<uint32_t> counts;
                                 for (uint32_t q = 0; q <= m->K.get_dimension(); ++q)
                                     counts.push_back(m->K.get_num_q_simplices(q));
                                 return IntList(counts);
                             }});
        t.members.push_back({"vertices", false, "the vertex count", [](const ObjectPtr& o) {
                                 return Value::from_int(
                                     o->payload_as<ModelVal>()->K.get_num_vertices());
                             }});
        t.print = [](const ObjectPtr& o, std::int64_t) {
            const ModelVal* m = o->payload_as<ModelVal>();
            std::ostringstream os;
            os << "Model " << m->name << " (" << m->K.get_num_vertices() << " vertices, dim "
               << m->K.get_dimension() << ")";
            return os.str();
        };
        t.show = [](const ObjectPtr& o) { return o->payload_as<ModelVal>()->K.to_string(); };
        registry.register_type(std::move(t));
    }
    {
        TypeInfo t;
        t.name = kTypeComplex;
        t.doc = "a simplicial complex (the presets of SimplicialCohomology.h and complex(...))";
        t.members.push_back({"name", false, "the complex's name", [](const ObjectPtr& o) {
                                 return Value::from_string(o->payload_as<ComplexVal>()->K.get_name());
                             }});
        t.members.push_back({"dim", false, "the top dimension", [](const ObjectPtr& o) {
                                 return Value::from_int(o->payload_as<ComplexVal>()->K.get_dimension());
                             }});
        t.members.push_back({"euler", false, "the Euler characteristic", [](const ObjectPtr& o) {
                                 return Value::from_int(
                                     o->payload_as<ComplexVal>()->K.get_euler_characteristic());
                             }});
        t.members.push_back({"dims", false, "the simplex counts by dimension",
                             [](const ObjectPtr& o) {
                                 const SimplicialComplex& K = o->payload_as<ComplexVal>()->K;
                                 std::vector<uint32_t> counts;
                                 for (uint32_t q = 0; q <= K.get_dimension(); ++q)
                                     counts.push_back(K.get_num_q_simplices(q));
                                 return IntList(counts);
                             }});
        TypeInfo::Method simplices;
        simplices.name = "simplices";
        simplices.params = {"q"};
        simplices.required = 1;
        simplices.computed = false;
        simplices.doc = "the q-simplices, each as a List of vertex indices";
        simplices.call = [](const ObjectPtr& o, const std::vector<Value>& args,
                            const SourcePos& pos) {
            const SimplicialComplex& K = o->payload_as<ComplexVal>()->K;
            const uint32_t q = U32Arg(args[0], pos, "Complex.simplices(q)");
            if (q > K.get_dimension()) {
                throw RuntimeError("Complex.simplices(q): q is above the complex's dimension", pos);
            }
            std::vector<Value> out;
            for (const std::vector<uint32_t>& s : K.get_simplices()[q]) out.push_back(IntList(s));
            return Value::from_list(std::move(out));
        };
        t.methods.push_back(std::move(simplices));
        t.print = [](const ObjectPtr& o, std::int64_t) {
            const SimplicialComplex& K = o->payload_as<ComplexVal>()->K;
            std::ostringstream os;
            os << "Complex " << K.get_name() << " (" << K.get_num_vertices() << " vertices, dim "
               << K.get_dimension() << ", chi " << K.get_euler_characteristic() << ")";
            return os.str();
        };
        t.show = [](const ObjectPtr& o) { return o->payload_as<ComplexVal>()->K.to_string(); };
        registry.register_type(std::move(t));
    }
}

} // namespace

void RegisterModelBindings(Registry& registry) {
    RegisterTypes(registry);

    const char* g = "pipeline";
    auto add = [&registry, g](const char* name, Signature sig, VerbImpl impl, const char* doc) {
        registry.register_verb(name, std::move(sig), std::move(impl), doc, g);
    };

    add("sl2z_model", Sig({}), &VerbSl2zModel,
        "the 7-vertex SL(2,Z) orbifold model with its exact tessellation tables");
    add("sl2z_model", Sig({Param::of("name", ValueKind::String)}), &VerbSl2zModel,
        "the named model descriptor (this build ships 'sl2z_7vertex')");
    add("models", Sig({}), &VerbModels, "the model names this build can build");

    add("sphere", Sig({Param::of("n", ValueKind::Int)}), &VerbSphere,
        "the simplicial n-sphere (n = 1, 2, 3)");
    add("cylinder", Sig({}), &VerbCylinder, "the simplicial cylinder");
    add("moebius", Sig({}), &VerbMoebius, "the simplicial Moebius band");
    add("projective_plane", Sig({}), &VerbProjectivePlane, "the simplicial projective plane");
    add("complex",
        Sig({Param::of("simplices", ValueKind::List), Param::of("name", ValueKind::String)}),
        &VerbComplexFromLists, "a complex from its simplex lists, grouped by dimension");
}

} // namespace bind
} // namespace console
