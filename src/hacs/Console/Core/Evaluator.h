// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Evaluator.h -- console C0: statements and expressions.
//
// Plan Sec. 9: "Evaluator.h/.cpp  statements, expressions, scoping (one global
// frame + for-loop shadowing), include, args".  The evaluator knows NO verb and
// NO type (Sec. 2 C3): every call goes through the Registry, every member
// access through the handle's TypeInfo.
//
// Semantics fixed by plan Sec. 3.2:
//   * `/` on Int is EXACT integer division and errors on a non-zero remainder.
//   * `^` on Int is exact and overflow-checked.
//   * every Int operation is overflow-checked; Int is int64_t.
//   * comparison of objects is `equal` as mathematical objects.
//   * `include` is resolved relative to the including file, cycle-checked, and
//     runs in the same global frame.
//
// STL-ONLY (plan Sec. 2 C8).
// ===========================================================================
#pragma once
#ifndef CONSOLE_EVALUATOR_H
#define CONSOLE_EVALUATOR_H

#include <functional>
#include <string>

#include "Ast.h"
#include "Session.h"
#include "Value.h"

namespace console {

class Evaluator {
public:
    explicit Evaluator(Session& session) : session_(session) {}

    // How `include("path.oc")` obtains a file.  The default reads the file
    // system; ScriptRunner and the validation install their own, which is what
    // lets the include machinery (relative resolution, cycles) be tested
    // without touching the disk.
    using FileLoader = std::function<bool(const std::string& path, std::string& contents)>;
    void set_file_loader(FileLoader loader) { loader_ = std::move(loader); }

    // Runs every statement in order.  Propagates ConsoleError (the caller maps
    // it to an exit code), ExitSignal and QuitSignal.
    void run(const Program& program);

    // One statement / one expression -- the REPL's granularity.
    void exec(const StmtPtr& stmt);
    Value eval(const ExprPtr& expr);

    Session& session() { return session_; }

private:
    void exec_block(const std::vector<StmtPtr>& body);
    void exec_assign(const Stmt& s);
    void exec_expr_stmt(const Stmt& s);
    void exec_for(const Stmt& s);
    void exec_if(const Stmt& s);
    void exec_set(const Stmt& s);
    void exec_include(const Stmt& s);
    void exec_help(const Stmt& s);
    void exec_expect_error(const Stmt& s);

    Value eval_call(const Expr& e);
    Value eval_member(const Expr& e);
    Value eval_method(const Expr& e);
    Value eval_index(const Expr& e);
    Value eval_binary(const Expr& e);
    Value eval_compare(const Expr& e);
    Value eval_logical(const Expr& e);
    Value eval_unary(const Expr& e);

    // Reads a member of a handle, running compute() first when the member is a
    // computed one (plan Sec. 4.2).
    Value object_member(const ObjectPtr& obj, const std::string& name, const SourcePos& pos);

    std::string help_text(const std::string& topic) const;

    Session& session_;
    FileLoader loader_;
};

// Joins `dir` and a possibly relative `path` (plan Sec. 3.2, include
// resolution).  Hand-rolled rather than <filesystem>, which is not needed for
// the POSIX-style paths the tree uses and would be the header's only heavy
// dependency.
std::string ResolvePath(const std::string& dir, const std::string& path);
// The directory part of `path` ("" when there is none).
std::string DirectoryOf(const std::string& path);

} // namespace console

#endif // CONSOLE_EVALUATOR_H
