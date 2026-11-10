// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// LineInput.cpp -- see LineInput.h.
// ===========================================================================

#include "LineInput.h"

#include "Lexer.h"

#include <istream>
#include <ostream>

namespace console {

bool LineInput::read_statement(std::string& text) {
    text.clear();
    bool first = true;
    while (true) {
        if (show_prompts_) {
            out_ << (first ? primary_ : continuation_);
            out_.flush();
        }
        std::string line;
        if (!std::getline(in_, line)) {
            if (show_prompts_) out_ << "\n";
            return !text.empty();   // trailing input without a final newline
        }
        if (!text.empty()) text += "\n";
        text += line;
        first = false;
        if (!NeedsContinuation(text)) return true;
    }
}

} // namespace console
