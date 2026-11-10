// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// LineInput.h -- console C0: the dumb line reader.
//
// Plan Sec. 2 C6: "dumb mode -- std::getline, prompt `oc> `, continuation
// `... ` while brackets are unbalanced or the line ends in `\`, no echo tricks;
// identical under a pipe and a terminal.  Users who want history and editing
// run `rlwrap ./cohomology_app --console`."  The hand-rolled termios editor is
// optional phase C3 and lives in Console/Platform/, which C0 does not create.
//
// STL-ONLY (plan Sec. 2 C8).
// ===========================================================================
#pragma once
#ifndef CONSOLE_LINE_INPUT_H
#define CONSOLE_LINE_INPUT_H

#include <iosfwd>
#include <string>

namespace console {

class LineInput {
public:
    LineInput(std::istream& in, std::ostream& out, bool show_prompts)
        : in_(in), out_(out), show_prompts_(show_prompts) {}

    // Reads lines until they form something that could be a complete statement
    // (NeedsContinuation() is false) or input ends.  Returns false at EOF with
    // nothing to run.  `text` never carries the prompts.
    bool read_statement(std::string& text);

    void set_prompts(std::string primary, std::string continuation) {
        primary_ = std::move(primary);
        continuation_ = std::move(continuation);
    }

private:
    std::istream& in_;
    std::ostream& out_;
    bool show_prompts_;
    std::string primary_ = "oc> ";
    std::string continuation_ = "... ";
};

} // namespace console

#endif // CONSOLE_LINE_INPUT_H
