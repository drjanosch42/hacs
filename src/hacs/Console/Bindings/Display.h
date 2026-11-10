// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Display.h -- the verb `display(x, ...)` ([P] Sec. 3.3 / 18.4; slot-24 NB-RENDER R3).
//
// display(x, ...) forces every declared handle among its arguments (exactly as
// compute(x) does) and then shows each argument:
//   * on a terminal, in a script, in every transcript: one line per argument,
//     the full canonical form -- the text `show(x)` prints;
//   * in a served session (`--serve`): what a bare expression statement `x`
//     sends -- the canonical echo on stdout, then one `value` frame per argument,
//     so the notebook page renders it (TeX, tables) instead of plain `print` text.
// It never changes a value: the served path differs only in the frame it adds.
// The served session installs the sink below for the duration of one cell.
// ===========================================================================
#pragma once
#ifndef CONSOLE_BINDINGS_DISPLAY_H
#define CONSOLE_BINDINGS_DISPLAY_H

#include <functional>

#include "../Core/Value.h"

namespace console {

class Registry;
struct SourcePos;

namespace bind {

// The served session's frame sink: called once per argument, after forcing.
using DisplaySink = std::function<void(const Value& v, const SourcePos& pos)>;

// Installs `sink` for the lifetime of the scope (one served cell); the previous
// sink is restored on exit.  With no sink installed, display prints like show.
class DisplaySinkScope {
public:
    explicit DisplaySinkScope(DisplaySink sink);
    ~DisplaySinkScope();
    DisplaySinkScope(const DisplaySinkScope&) = delete;
    DisplaySinkScope& operator=(const DisplaySinkScope&) = delete;

private:
    DisplaySink saved_;
};

// Registers `display` (group "core", right after the core printing verbs).
void RegisterDisplayBindings(Registry& registry);

} // namespace bind
} // namespace console

#endif // CONSOLE_BINDINGS_DISPLAY_H
