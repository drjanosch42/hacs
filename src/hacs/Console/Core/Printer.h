// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Printer.h -- console C0: the canonical printers.
//
// Plan Sec. 2 C5 ("Deterministic output by construction") and Sec. 4.3:
//
//   * ONE canonical printer per type, fixed for the life of a format version
//     (`#! oc-format 1` heads every .expected file, C1).
//   * Lists print `[a, b]`; Records print `{k: v, ...}`; numbers are canonical
//     residues; strings are QUOTED in canonical form.
//   * Size cap `print_max_entries` (default 400): a larger list or object
//     prints its summary form; `show(x)` prints everything.
//   * `#~`-tagged lines carry timings and are excluded from every .expected
//     comparison.
//
// Two renderings, deliberately distinguished (a C0 decision recorded in the
// WP-5 report): CANONICAL quotes a string, which is what plan Sec. 3.3's print
// form demands for a bare expression statement and inside a list; DISPLAY
// prints a top-level string bare, which is what `print("dim H^1")` must do to
// be usable.  Inside a compound value both are canonical, so a list of strings
// prints identically either way.
//
// STL-ONLY (plan Sec. 2 C8).
// ===========================================================================
#pragma once
#ifndef CONSOLE_PRINTER_H
#define CONSOLE_PRINTER_H

#include <cstdint>
#include <string>

#include "Value.h"

namespace console {

// The default of the `print_max_entries` setting (plan Sec. 4.3).
constexpr std::int64_t kDefaultPrintMaxEntries = 400;

// The prefix of every line the .expected comparison ignores (plan Sec. 2 C5).
extern const char* const kAnnotationPrefix;   // "#~ "

// Canonical form: strings quoted.  Used for a bare expression statement, for
// every element of a list or record, and by `show` (with no cap).
std::string PrintCanonical(const Value& v, std::int64_t max_entries);

// Display form: a top-level String prints bare; everything else is canonical.
std::string PrintDisplay(const Value& v, std::int64_t max_entries);

// The full form, no cap -- the verb `show(x)`.
std::string PrintFull(const Value& v);

// `?x` / `describe(x)`: type, state, settings snapshot, creation site, members
// (plan Sec. 4.4).  Multi-line, no trailing newline.
std::string Describe(const Value& v);

// JSON with a proper escaper (the plan's gap G1): Int, Bool, String, List,
// Record.  An Object with no JSON of its own renders as a string naming its
// type and state.
std::string ToJson(const Value& v);
std::string JsonEscape(const std::string& s);

} // namespace console

#endif // CONSOLE_PRINTER_H
