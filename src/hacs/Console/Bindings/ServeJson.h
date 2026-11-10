// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// ServeJson.h -- the minimal JSON tree, reader and writer of the served
// console ([P] v3.10 Sec. 18.11 acceptance: "the JSON writer is a small
// hand-rolled emitter in Console/Bindings/, with the escaper of Sec. 4.3's
// json, and the JSON reader for requests is a minimal hand-written parser ...
// no external library").  Hecke-extensions slot 19, NB-2.
//
// Compiled only with OC_BUILD_NOTEBOOK=ON (cmake/slot19/NB2.cmake).
//
// Numbers are kept as their SOURCE TEXT (never a double), so an integer of any
// size round-trips; `WriteJson` applies the integer rule of [P] Sec. 18.5 to
// every integer literal it re-emits (|x| <= 2^53 - 1 a number, else a decimal
// String).
// ===========================================================================
#pragma once
#ifndef CONSOLE_BINDINGS_SERVE_JSON_H
#define CONSOLE_BINDINGS_SERVE_JSON_H

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace console {
namespace serve_json {

struct Json {
    enum class Kind { Null, Bool, Number, String, Array, Object };
    Kind kind = Kind::Null;
    bool b = false;
    std::string text;                                   // Number: source text; String: the value
    std::vector<Json> items;                            // Array
    std::vector<std::pair<std::string, Json>> members;  // Object, in source order

    bool is_null() const { return kind == Kind::Null; }
    bool is_bool() const { return kind == Kind::Bool; }
    bool is_number() const { return kind == Kind::Number; }
    bool is_string() const { return kind == Kind::String; }
    bool is_array() const { return kind == Kind::Array; }
    bool is_object() const { return kind == Kind::Object; }

    // First member named `key`, or nullptr.
    const Json* get(const std::string& key) const;
    // True when the Number is an integer literal that fits int64; fills `out`.
    bool as_int64(std::int64_t& out) const;
};

// Parses exactly one JSON value (surrounding whitespace allowed).  Returns
// false and a one-line reason on any error; `max_depth` bounds nesting.
bool ParseJson(const std::string& text, Json& out, std::string& error, int max_depth = 256);

// Compact re-emission (no insignificant whitespace), members in their order;
// integer literals obey the 2^53 rule, other numbers are emitted verbatim.
std::string WriteJson(const Json& j);

// "\"...\"" with the console's JsonEscape (Console/Core/Printer.h).
std::string Quote(const std::string& s);

// The integer rule of [P] Sec. 18.5: a JSON number when |x| <= 2^53 - 1,
// else the decimal String.
std::string IntJson(std::int64_t v);
std::string UIntJson(std::uint64_t v);

} // namespace serve_json
} // namespace console

#endif // CONSOLE_BINDINGS_SERVE_JSON_H
