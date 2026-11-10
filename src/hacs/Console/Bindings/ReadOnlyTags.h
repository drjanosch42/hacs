// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// ReadOnlyTags.h -- the per-overload `read_only` tag of [P] v3.10 Sec. 18.8
// (hecke-extensions slot 19, NB-2).
//
// THE SEAM.  The tag is registry metadata, but Core's Overload gets no field
// for it ([P] Sec. 18.13: Console/Core/ changes no semantics; the one Core
// addition is the served view's Registry::transform_overloads).  The tag lives
// beside the registry, here, keyed by (verb name, Signature::to_string(name)) --
// the same text the manual's verb table and `help` print, so a key can never
// drift from what a reader sees.  Every registered overload must have exactly
// one row (ConsoleReadOnlyTagValidation (B): registered == explicitly tagged);
// an overload with no row is UNTAGGED and therefore NOT read-only ("a new verb
// is excluded until audited").  The reviewed twin of this table, with the audit
// column, is Console/serve/read_only_verbs.tsv (check (A)).
//
// Always compiled (the tag and the manual's column are registry data and are
// not guarded by OC_BUILD_NOTEBOOK, [P] Sec. 18.1 item 5); only `--serve`
// consumes it at run time.
// ===========================================================================
#pragma once
#ifndef CONSOLE_BINDINGS_READ_ONLY_TAGS_H
#define CONSOLE_BINDINGS_READ_ONLY_TAGS_H

#include <cstddef>
#include <string>
#include <vector>

namespace console {

class Registry;
struct VerbEntry;
struct Overload;

namespace serve_tags {

struct ReadOnlyTag {
    const char* verb;
    const char* signature;   // Signature::to_string(verb)
    bool read_only;
};

// Every row, in registration order.
const std::vector<ReadOnlyTag>& ReadOnlyTagTable();

enum class TagState { Untagged, ReadOnly, NotReadOnly };

TagState LookupTag(const std::string& verb, const std::string& signature);
TagState LookupTag(const VerbEntry& entry, const Overload& overload);
inline bool IsReadOnly(const VerbEntry& entry, const Overload& overload) {
    return LookupTag(entry, overload) == TagState::ReadOnly;
}

// The keys a served session may `set` ([P] Sec. 18.8, "Settable keys"):
// print_max_entries, lean, program_cache, confirm_wall_s (NBCONFIRM S4) and the eleven mathematical settings.
const std::vector<std::string>& ServeSettableKeys();
bool IsServeSettable(const std::string& key);

// Registry-wide tallies (ConsoleReadOnlyTagValidation (B), the manual's
// counts, the served `hello`).
struct TagCensus {
    std::size_t overloads = 0;          // registered overloads
    std::size_t tagged = 0;             // overloads with a row (explicitly tagged)
    std::size_t read_only = 0;          // ... whose row says true
    std::size_t read_only_verbs = 0;    // verbs with at least one read-only overload
    std::vector<std::string> untagged;  // "name | signature" of every overload with no row
    std::vector<std::string> stale;     // rows matching no registered overload
};
TagCensus CensusOf(const Registry& registry);

} // namespace serve_tags
} // namespace console

#endif // CONSOLE_BINDINGS_READ_ONLY_TAGS_H
