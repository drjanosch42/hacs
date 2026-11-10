// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// PrintJson.h -- the value envelope of the served console, `json_version` 1
// ([P] v3.10 Sec. 18.5; hecke-extensions slot 19, NB-2).
//
//     {"t":"<TypeName>","canonical":"<PrintCanonical text>","hash":"0x..."|null,
//      "v":<per-type tree>[,"truncated":true][,"length":k][,"generic":true][,"state":"..."][,"hint":{...}]}
//
// Compiled only with OC_BUILD_NOTEBOOK=ON.  NOT a verb and not callable from
// OC; the `json(x)` verb of Sec. 4.3 is unchanged.  No TeX anywhere, no float.
//
// HOW A TREE IS BUILT.  Through the console surface only -- the registered
// members and methods of a type (TypeInfo), and the `hash` / `coeffs` verbs of
// the registry -- so this file names no pipeline type and no payload.  A member
// whose TypeInfo flag says `computed` is read only when the handle is already
// `computed`: rendering NEVER forces a `declared` handle (Sec. 4.3 / 2 C2), and
// a declared or failed handle's `hash` is null (hashing would force it).
// ===========================================================================
#pragma once
#ifndef CONSOLE_BINDINGS_PRINT_JSON_H
#define CONSOLE_BINDINGS_PRINT_JSON_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "../Core/Value.h"

namespace console {

class Registry;
class Session;

namespace serve {

constexpr int kJsonVersion = 1;

struct PrintJsonContext {
    const Registry* registry = nullptr;   // where `hash` / `coeffs` are looked up
    Session* session = nullptr;
    std::int64_t max_entries = 400;       // print_max_entries in force
};

// The envelope of `v`.
std::string PrintJson(const Value& v, const PrintJsonContext& ctx);

// The truncated form of Sec. 18.4 (the frame cap): `t`, a `canonical` cut to at
// most `canonical_cap` bytes (at a UTF-8 boundary), `hash`, `v: null`,
// `truncated: true`.
std::string PrintJsonTruncated(const Value& v, const PrintJsonContext& ctx, std::size_t canonical_cap);

// The types with a DEDICATED schema (form D of the Sec. 18.5 table), the six
// core kinds (Nil, Bool, Int, String, List, Record) included.  Every other
// registered type is rendered by the generic form (G).
const std::vector<std::string>& DedicatedSchemaTypes();
bool HasDedicatedSchema(const std::string& type_name);

// (slot-24 NB-RENDER R1) The member form D-m: the types of DedicatedSchemaTypes()
// whose `v` is {"members":{<name>:<envelope>},"calls":[[<method>,[<arg>],<envelope>],...]}.
const std::vector<std::string>& MemberFormTypes();

// (slot-24 NB-RENDER R1) The render hints ([P] Sec. 18.5 "hint"); the twin of
// Console/serve/render_hints.tsv.  `name` is a Record schema or a type name.
struct RenderHintRow {
    const char* name;
    const char* path;
    const char* kind;
    const char* attrs;   // space-separated key=value; "" = none
};
const std::vector<RenderHintRow>& RenderHintRows();

// The `hash` field alone (null or "0x...") -- exposed for the conformance arm.
std::string HashField(const Value& v, const PrintJsonContext& ctx);

} // namespace serve
} // namespace console

#endif // CONSOLE_BINDINGS_PRINT_JSON_H
