// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Printer.cpp -- see Printer.h.
// ===========================================================================

#include "Printer.h"

#include "Object.h"

#include <limits>

namespace console {

const char* const kAnnotationPrefix = "#~ ";

namespace {

std::string QuoteString(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;      break;
        }
    }
    out += "\"";
    return out;
}

std::string PrintValue(const Value& v, std::int64_t max_entries, bool top_level_bare_string);

std::string PrintList(const std::vector<Value>& items, std::int64_t max_entries) {
    // The cap counts ENTRIES, so a list longer than the cap prints its head and
    // says how many it withheld -- deterministic, and never unbounded output.
    const bool capped = max_entries >= 0 &&
                        static_cast<std::int64_t>(items.size()) > max_entries;
    const std::size_t shown =
        capped ? static_cast<std::size_t>(max_entries) : items.size();
    std::string out = "[";
    for (std::size_t i = 0; i < shown; ++i) {
        if (i != 0) out += ", ";
        out += PrintValue(items[i], max_entries, false);
    }
    if (capped) {
        if (shown != 0) out += ", ";
        out += "... (";
        out += std::to_string(items.size() - shown);
        out += " more of ";
        out += std::to_string(items.size());
        out += ")";
    }
    out += "]";
    return out;
}

std::string PrintRecord(const Record& r, std::int64_t max_entries) {
    std::string out = "{";
    bool first = true;
    for (const std::pair<std::string, Value>& e : r.entries()) {
        if (!first) out += ", ";
        first = false;
        out += e.first;
        out += ": ";
        out += PrintValue(e.second, max_entries, false);
    }
    out += "}";
    return out;
}

std::string PrintObject(const ObjectPtr& o, std::int64_t max_entries) {
    if (!o) return "<null object>";
    if (o->type() != nullptr && o->type()->print && o->state() != ObjectState::Failed) {
        return o->type()->print(o, max_entries);
    }
    // The default form of a handle that has no printer, or that failed.
    std::string out = o->type_name();
    out += " (";
    out += ObjectStateName(o->state());
    out += ")";
    return out;
}

std::string PrintValue(const Value& v, std::int64_t max_entries, bool top_level_bare_string) {
    switch (v.kind()) {
        case ValueKind::Nil:    return "nil";
        case ValueKind::Int:    return std::to_string(v.int_value());
        case ValueKind::Bool:   return v.bool_value() ? "true" : "false";
        case ValueKind::String: return top_level_bare_string ? v.string_value()
                                                             : QuoteString(v.string_value());
        case ValueKind::List:   return PrintList(v.list_value(), max_entries);
        case ValueKind::Record: return PrintRecord(v.record_value(), max_entries);
        case ValueKind::Object: return PrintObject(v.object_value(), max_entries);
    }
    return "nil";
}

} // namespace

std::string PrintCanonical(const Value& v, std::int64_t max_entries) {
    return PrintValue(v, max_entries, false);
}

std::string PrintDisplay(const Value& v, std::int64_t max_entries) {
    return PrintValue(v, max_entries, true);
}

std::string PrintFull(const Value& v) {
    if (v.is_object()) {
        const ObjectPtr& o = v.object_value();
        if (o && o->type() != nullptr && o->type()->show) return o->type()->show(o);
    }
    return PrintValue(v, -1, false);   // -1 = no cap
}

std::string Describe(const Value& v) {
    std::string out;
    out += "type:  " + v.type_name();
    if (!v.is_object()) {
        switch (v.kind()) {
            case ValueKind::List:
                out += "\nsize:  " + std::to_string(v.list_value().size());
                break;
            case ValueKind::Record: {
                out += "\nkeys:  ";
                const std::vector<std::string> keys = v.record_value().keys();
                for (std::size_t i = 0; i < keys.size(); ++i) {
                    if (i != 0) out += ", ";
                    out += keys[i];
                }
                break;
            }
            case ValueKind::String:
                out += "\nlength: " + std::to_string(v.string_value().size());
                break;
            default:
                break;
        }
        out += "\nvalue: " + PrintCanonical(v, kDefaultPrintMaxEntries);
        return out;
    }

    const ObjectPtr& o = v.object_value();
    if (!o) return out + "\nvalue: <null object>";

    out += "\nstate: ";
    out += ObjectStateName(o->state());
    if (o->failed()) {
        out += "\nerror: " + o->error_text();
    }
    if (o->creation_site().known() || !o->creation_expr().empty()) {
        out += "\nborn:  ";
        if (o->creation_site().known()) out += o->creation_site().to_string();
        if (!o->creation_expr().empty()) {
            if (o->creation_site().known()) out += " ";
            out += "by " + o->creation_expr();
        }
    }
    if (!o->snapshot().empty()) {
        out += "\nsettings: " + o->snapshot().to_string();
    }
    if (o->is_resumable()) {
        // R3: a budget cut-off is visible in `?x`, with the setting that cut it.
        out += "\nresumable: yes (budget setting '" + o->resumable_state().budget_setting +
               "'";
        if (!o->resumable_state().reason.empty()) {
            out += ", " + o->resumable_state().reason;
        }
        out += ")";
    }
    if (o->resumed_from()) {
        out += "\nresumed from: " + o->resumed_from()->creation_expr();
    }
    if (!o->kept_alive().empty()) {
        out += "\nkeeps alive: ";
        for (std::size_t i = 0; i < o->kept_alive().size(); ++i) {
            if (i != 0) out += ", ";
            out += o->kept_alive()[i]->type_name();
        }
    }
    if (o->type() != nullptr) {
        if (!o->type()->members.empty()) {
            out += "\nmembers:";
            for (const TypeInfo::Member& m : o->type()->members) {
                out += "\n  ." + m.name;
                out += m.computed ? "  (computed)" : "  (structural)";
                if (!m.doc.empty()) out += "  -- " + m.doc;
            }
        }
        if (!o->type()->methods.empty()) {
            out += "\nmethods:";
            for (const TypeInfo::Method& m : o->type()->methods) {
                out += "\n  ." + m.name + "(";
                for (std::size_t i = 0; i < m.params.size(); ++i) {
                    if (i != 0) out += ", ";
                    out += m.params[i];
                }
                out += ")";
                if (!m.doc.empty()) out += "  -- " + m.doc;
            }
        }
    }
    return out;
}

std::string JsonEscape(const std::string& s) {
    std::string out;
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    // \u00XX -- the control range JSON forbids raw.
                    static const char* const hex = "0123456789abcdef";
                    out += "\\u00";
                    out += hex[(c >> 4) & 0xF];
                    out += hex[c & 0xF];
                } else {
                    out += static_cast<char>(c);
                }
                break;
        }
    }
    return out;
}

std::string ToJson(const Value& v) {
    switch (v.kind()) {
        case ValueKind::Nil:    return "null";
        case ValueKind::Int:    return std::to_string(v.int_value());
        case ValueKind::Bool:   return v.bool_value() ? "true" : "false";
        case ValueKind::String: return "\"" + JsonEscape(v.string_value()) + "\"";
        case ValueKind::List: {
            std::string out = "[";
            const std::vector<Value>& items = v.list_value();
            for (std::size_t i = 0; i < items.size(); ++i) {
                if (i != 0) out += ", ";
                out += ToJson(items[i]);
            }
            out += "]";
            return out;
        }
        case ValueKind::Record: {
            std::string out = "{";
            bool first = true;
            for (const std::pair<std::string, Value>& e : v.record_value().entries()) {
                if (!first) out += ", ";
                first = false;
                out += "\"" + JsonEscape(e.first) + "\": " + ToJson(e.second);
            }
            out += "}";
            return out;
        }
        case ValueKind::Object: {
            const ObjectPtr& o = v.object_value();
            const std::string text = o ? PrintObject(o, kDefaultPrintMaxEntries)
                                       : std::string("<null object>");
            return "\"" + JsonEscape(text) + "\"";
        }
    }
    return "null";
}

} // namespace console
