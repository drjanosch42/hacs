// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Parser.h -- console C0: recursive descent over the EBNF of plan Sec. 3.2
// (reproduced in Ast.h).
//
// The WHOLE input is parsed before anything runs, which is what makes a parse
// error exit code 2 "reported before anything runs" (plan Sec. 6).
//
// STL-ONLY (plan Sec. 2 C8).
// ===========================================================================
#pragma once
#ifndef CONSOLE_PARSER_H
#define CONSOLE_PARSER_H

#include <string>

#include "Ast.h"

namespace console {

// Throws ParseError (carrying the offending line and a caret) on any syntax
// error.  `file` names the input in every position.
Program Parse(const std::string& source, const std::string& file);

} // namespace console

#endif // CONSOLE_PARSER_H
