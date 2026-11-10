// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Display.cpp -- see Display.h.
// ===========================================================================

#include "Display.h"

#include "../Core/Object.h"
#include "../Core/Printer.h"
#include "../Core/Registry.h"
#include "../Core/Session.h"

#include <utility>
#include <vector>

namespace console {
namespace bind {

namespace {

// One served session per process ([P] Sec. 18.2), one cell at a time: a plain
// process-wide slot suffices.
DisplaySink& Slot() {
    static DisplaySink sink;
    return sink;
}

Value VerbDisplay(CallContext& ctx, const std::vector<Value>& args) {
    // Force first, all of them, so that a failure shows nothing half-way.
    for (const Value& a : args) {
        if (!a.is_object()) continue;
        const ObjectPtr& o = a.object_value();
        if (o && !o->computed()) o->ensure_computed(ctx.session->settings(), ctx.pos);
    }
    const DisplaySink& sink = Slot();
    for (const Value& a : args) {
        if (sink) sink(a, ctx.pos);
        else ctx.session->print_line(PrintFull(a));
    }
    return Value::nil();
}

} // namespace

DisplaySinkScope::DisplaySinkScope(DisplaySink sink) : saved_(std::move(Slot())) { Slot() = std::move(sink); }

DisplaySinkScope::~DisplaySinkScope() { Slot() = std::move(saved_); }

void RegisterDisplayBindings(Registry& registry) {
    Signature s;
    s.params = {Param::any("value")};
    s.variadic = true;
    registry.register_verb("display", std::move(s), &VerbDisplay,
                           "forces each declared handle, then shows each argument: in a served notebook as a rendered "
                           "value (one value frame each, like a bare expression), elsewhere as the full canonical form "
                           "(like show)",
                           "core");
}

} // namespace bind
} // namespace console
