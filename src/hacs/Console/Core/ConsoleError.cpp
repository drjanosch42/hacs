// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// ConsoleError.cpp -- see ConsoleError.h.
// ===========================================================================

#include "ConsoleError.h"

#include <string>

namespace console {

std::string SourcePos::to_string() const {
    if (line <= 0) return file;
    std::string out = file;
    out += ":";
    out += std::to_string(line);
    if (col > 0) {
        out += ":";
        out += std::to_string(col);
    }
    return out;
}

void ConsoleError::rebuild() {
    // "file:line:col: kind: message" + one indented line per note.  The prefix
    // is omitted when nothing about the position is known, so that an error
    // raised outside any script still reads naturally.
    formatted_.clear();
    const std::string where = pos_.to_string();
    if (!where.empty()) {
        formatted_ += where;
        formatted_ += ": ";
    }
    formatted_ += kind_;
    formatted_ += ": ";
    formatted_ += message_;
    for (const std::string& note : notes_) {
        formatted_ += "\n  ";
        formatted_ += note;
    }
}

// (C1a) The kind string of plan Sec. 2 C7's own outcome; ScriptRunner maps it
// to exit code 5.
const char* const kNotImplementedKind = "not implemented in this build";

ParseError::ParseError(std::string message, SourcePos pos, std::string source_line)
    : ConsoleError("parse error", std::move(message), std::move(pos)),
      source_line_(std::move(source_line)) {
    if (!source_line_.empty()) {
        // The offending line plus a caret under the column (plan Sec. 6,
        // "a parse error also prints the offending line with a caret").
        std::string caret;
        const int c = this->pos().col;
        for (int i = 1; i < c; ++i) {
            caret += (static_cast<size_t>(i - 1) < source_line_.size() &&
                      source_line_[static_cast<size_t>(i - 1)] == '\t')
                         ? '\t'
                         : ' ';
        }
        caret += '^';
        add_note(source_line_);
        add_note(caret);
    }
}

namespace detail {

void ConsoleCheckFailed(const char* file, int line, const char* cond, const char* msg) {
    std::string text = std::string(file) + ":" + std::to_string(line) +
                       ": CONSOLE_CHECK failed: " + cond;
    if (msg != nullptr && msg[0] != '\0') {
        text += " -- ";
        text += msg;
    }
    throw std::logic_error(text);
}

} // namespace detail

} // namespace console
