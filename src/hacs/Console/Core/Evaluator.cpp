// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Evaluator.cpp -- see Evaluator.h.
// ===========================================================================

#include "Evaluator.h"

#include "Cancel.h"

#include "Parser.h"
#include "Printer.h"
#include "Registry.h"

#include <fstream>
#include <limits>
#include <sstream>

namespace console {

namespace {

constexpr std::int64_t kInt64Min = std::numeric_limits<std::int64_t>::min();

// Overflow-checked Int arithmetic (plan Sec. 3.1: "Type Int = int64_t").
bool AddChecked(std::int64_t a, std::int64_t b, std::int64_t& out) {
    return !__builtin_add_overflow(a, b, &out);
}
bool SubChecked(std::int64_t a, std::int64_t b, std::int64_t& out) {
    return !__builtin_sub_overflow(a, b, &out);
}
bool MulChecked(std::int64_t a, std::int64_t b, std::int64_t& out) {
    return !__builtin_mul_overflow(a, b, &out);
}

// (C1a) The ONE place a C++ exception raised by a verb or a method becomes a
// console error.  The registry's translator names the C++ class; the registry's
// KIND classifier decides which console error it becomes -- plan Sec. 2 C7's
// catch order, installed by the bindings, so that the core names no pipeline
// type.  An unrecognised kind is the ordinary runtime error (exit 3); the one
// kind C1a installs is kNotImplementedKind (exit 5).
[[noreturn]] void ThrowWrapped(const Registry& registry, const std::exception& ex,
                               std::string message, const SourcePos& pos, const ObjectPtr* obj) {
    std::string class_name;
    const bool named = registry.translate_exception(ex, class_name);
    const std::string kind = registry.exception_kind(ex);
    const std::string cause =
        std::string("caused by: ") + (named ? class_name + ": " : "") + ex.what();
    if (kind == kNotImplementedKind) {
        NotImplementedError err(std::move(message), pos);
        if (obj != nullptr && *obj) err.add_note(CreationSiteNote(*obj));
        err.add_note(cause);
        throw err;
    }
    RuntimeError err(std::move(message), pos);
    if (obj != nullptr && *obj) err.add_note(CreationSiteNote(*obj));
    err.add_note(cause);
    throw err;
}

} // namespace

std::string DirectoryOf(const std::string& path) {
    const std::size_t slash = path.find_last_of('/');
    if (slash == std::string::npos) return std::string();
    if (slash == 0) return std::string("/");
    return path.substr(0, slash);
}

std::string ResolvePath(const std::string& dir, const std::string& path) {
    if (path.empty()) return path;
    if (path[0] == '/') return path;           // absolute
    if (dir.empty()) return path;
    if (dir.back() == '/') return dir + path;
    return dir + "/" + path;
}

// --- statements ------------------------------------------------------------

void Evaluator::run(const Program& program) {
    for (const StmtPtr& s : program.statements) exec(s);
}

void Evaluator::exec(const StmtPtr& stmt) {
    CONSOLE_CHECK(stmt != nullptr);
    const Stmt& s = *stmt;
    // (NBCONFIRM S5) a cancellation point at every statement, nested ones (a `for` body) included; a no-op outside a
    // CancelScope (scripts, -c, validations).  The innermost statement names the stop.
    try {
        PollCancel("statement");
    } catch (Interrupted& x) {
        if (!x.pos.known()) x.pos = s.pos;
        throw;
    }
    switch (s.kind) {
        case StmtKind::Assign:      exec_assign(s); return;
        case StmtKind::ExprStmt:    exec_expr_stmt(s); return;
        case StmtKind::For:         exec_for(s); return;
        case StmtKind::If:          exec_if(s); return;
        case StmtKind::Set:         exec_set(s); return;
        case StmtKind::Include:     exec_include(s); return;
        case StmtKind::Help:        exec_help(s); return;
        case StmtKind::ExpectError: exec_expect_error(s); return;
        case StmtKind::Quit:        throw QuitSignal{0};
    }
}

void Evaluator::exec_block(const std::vector<StmtPtr>& body) {
    for (const StmtPtr& s : body) exec(s);
}

void Evaluator::exec_assign(const Stmt& s) {
    Value v = eval(s.value);
    if (v.is_nil()) {
        throw RuntimeError("'" + s.name + "' would be bound to nothing: " +
                               (s.value ? s.value->source_text : std::string("this expression")) +
                               " produces no value",
                           s.pos);
    }
    session_.set_var(s.name, std::move(v));
}

void Evaluator::exec_expr_stmt(const Stmt& s) {
    Value v = eval(s.value);
    if (s.suppress_print || v.is_nil()) return;
    session_.print_line(PrintCanonical(v, session_.print_max_entries()));
}

void Evaluator::exec_for(const Stmt& s) {
    Value seq = eval(s.value);
    if (!seq.is_list()) {
        throw TypeError("'for' iterates over a List, got " + seq.type_name(), s.pos);
    }
    // One global frame with for-loop shadowing (plan Sec. 9): the previous
    // binding of the loop variable is saved and restored, so a loop cannot
    // silently clobber a name.
    const Value* prev = session_.find_var(s.name);
    const bool had_prev = (prev != nullptr);
    const Value saved = had_prev ? *prev : Value();

    struct Restore {
        Session& session;
        const std::string& name;
        bool had_prev;
        const Value& saved;
        ~Restore() {
            if (had_prev) session.set_var(name, saved);
            else session.erase_var(name);
        }
    } restore{session_, s.name, had_prev, saved};

    for (const Value& item : seq.list_value()) {
        session_.set_var(s.name, item);
        exec_block(s.body);
    }
}

void Evaluator::exec_if(const Stmt& s) {
    Value cond = eval(s.value);
    if (!cond.is_bool()) {
        throw TypeError("'if' needs a Bool condition, got " + cond.type_name() +
                            " (there is no truthiness in this language)",
                        s.pos);
    }
    if (cond.bool_value()) exec_block(s.body);
    else exec_block(s.orelse);
}

void Evaluator::exec_set(const Stmt& s) {
    Value v = eval(s.value);
    // (C5a, slot-9 WP-12; decision 2 (ii) (A), a named exception to [P] C5 (b))
    // A refusal raised below the statement -- e.g. a settings installer's
    // NotImplementedError for `coefficient_ring = "Z"` -- carries no position
    // of its own; back-fill the statement's file:line:col.
    try {
        session_.settings().set(s.name, v, s.pos);
    } catch (ConsoleError& e) {
        e.set_pos_if_unknown(s.pos);
        throw;
    }
}

void Evaluator::exec_include(const Stmt& s) {
    const std::string resolved = ResolvePath(session_.current_dir(), s.text);

    // Cycle check (plan Sec. 3.2).
    for (const std::string& active : session_.include_stack()) {
        if (active == resolved) {
            std::string chain;
            for (const std::string& a : session_.include_stack()) {
                chain += a;
                chain += " -> ";
            }
            chain += resolved;
            throw RuntimeError("include cycle: " + chain, s.pos);
        }
    }

    std::string contents;
    bool ok = false;
    if (loader_) {
        ok = loader_(resolved, contents);
    } else {
        std::ifstream in(resolved.c_str());
        if (in) {
            std::ostringstream buf;
            buf << in.rdbuf();
            contents = buf.str();
            ok = true;
        }
    }
    if (!ok) {
        throw RuntimeError("include(\"" + s.text + "\"): cannot read '" + resolved + "'", s.pos);
    }

    const Program included = Parse(contents, resolved);   // ParseError propagates

    const std::string saved_dir = session_.current_dir();
    session_.include_stack().push_back(resolved);
    session_.set_current_dir(DirectoryOf(resolved));
    struct Pop {
        Session& session;
        const std::string& saved_dir;
        ~Pop() {
            session.include_stack().pop_back();
            session.set_current_dir(saved_dir);
        }
    } pop{session_, saved_dir};

    run(included);   // same global frame
}

void Evaluator::exec_help(const Stmt& s) {
    session_.print_line(help_text(s.name));
}

void Evaluator::exec_expect_error(const Stmt& s) {
    // The ONLY error-consuming construct (plan Sec. 8): there is no `try`.
    // ExitSignal / QuitSignal are not ConsoleErrors and therefore pass through.
    // (NBCONFIRM S4) inside the block a cost-gated call never asks (Session::expect_error_depth).
    struct Depth {
        Session& s;
        explicit Depth(Session& x) : s(x) { s.enter_expect_error(); }
        ~Depth() { s.leave_expect_error(); }
    };
    try {
        Depth depth(session_);
        exec_block(s.body);
    } catch (const ConsoleError& e) {
        const std::string text = e.formatted();
        if (text.find(s.text) != std::string::npos) return;
        throw AssertionFailure("expect_error: the block failed, but not with \"" + s.text +
                                   "\"; it failed with: " + text,
                               s.pos);
    }
    throw AssertionFailure("expect_error: the block did not fail (expected an error containing \"" +
                               s.text + "\")",
                           s.pos);
}

// --- expressions -----------------------------------------------------------

Value Evaluator::eval(const ExprPtr& expr) {
    CONSOLE_CHECK(expr != nullptr);
    const Expr& e = *expr;
    switch (e.kind) {
        case ExprKind::IntLit:    return Value::from_int(e.int_value);
        case ExprKind::StringLit: return Value::from_string(e.text);
        case ExprKind::BoolLit:   return Value::from_bool(e.bool_value);
        case ExprKind::Args:      return Value::from_record(session_.args());
        case ExprKind::Ident: {
            const Value* v = session_.find_var(e.text);
            if (v != nullptr) return *v;
            if (session_.registry().find_verb(e.text) != nullptr) {
                throw RuntimeError("'" + e.text + "' is a verb; call it as " + e.text + "(...)",
                                   e.pos);
            }
            throw RuntimeError("unknown name '" + e.text + "'", e.pos);
        }
        case ExprKind::ListLit: {
            std::vector<Value> items;
            items.reserve(e.items.size());
            for (const ExprPtr& item : e.items) {
                Value v = eval(item);
                if (v.is_nil()) {
                    throw RuntimeError("a list element produces no value", item->pos);
                }
                items.push_back(std::move(v));
            }
            return Value::from_list(std::move(items));
        }
        case ExprKind::Unary:      return eval_unary(e);
        case ExprKind::Binary:     return eval_binary(e);
        case ExprKind::Compare:    return eval_compare(e);
        case ExprKind::Logical:    return eval_logical(e);
        case ExprKind::Call:       return eval_call(e);
        case ExprKind::Member:     return eval_member(e);
        case ExprKind::MethodCall: return eval_method(e);
        case ExprKind::Index:      return eval_index(e);
    }
    throw RuntimeError("unsupported expression", e.pos);
}

Value Evaluator::eval_unary(const Expr& e) {
    Value v = eval(e.lhs);
    if (!v.is_int()) {
        throw TypeError("unary '-' expects Int, got " + v.type_name(), e.pos);
    }
    if (v.int_value() == kInt64Min) {
        throw RuntimeError("Int overflow in unary '-'", e.pos);
    }
    return Value::from_int(-v.int_value());
}

Value Evaluator::eval_binary(const Expr& e) {
    Value a = eval(e.lhs);
    Value b = eval(e.rhs);

    if (a.is_string() && b.is_string() && e.binary_op == BinaryOp::Add) {
        return Value::from_string(a.string_value() + b.string_value());
    }
    if (!a.is_int() || !b.is_int()) {
        throw TypeError(std::string("operator '") + BinaryOpName(e.binary_op) +
                            "' is not defined on " + a.type_name() + " and " + b.type_name(),
                        e.pos);
    }
    const std::int64_t x = a.int_value();
    const std::int64_t y = b.int_value();
    std::int64_t r = 0;
    switch (e.binary_op) {
        case BinaryOp::Add:
            if (!AddChecked(x, y, r)) throw RuntimeError("Int overflow in '+'", e.pos);
            return Value::from_int(r);
        case BinaryOp::Sub:
            if (!SubChecked(x, y, r)) throw RuntimeError("Int overflow in '-'", e.pos);
            return Value::from_int(r);
        case BinaryOp::Mul:
            if (!MulChecked(x, y, r)) throw RuntimeError("Int overflow in '*'", e.pos);
            return Value::from_int(r);
        case BinaryOp::Div:
            if (y == 0) throw RuntimeError("division by zero", e.pos);
            if (x == kInt64Min && y == -1) throw RuntimeError("Int overflow in '/'", e.pos);
            // Plan Sec. 3.2: exact integer division; a non-zero remainder is an
            // error, never a silent truncation.
            if (x % y != 0) {
                throw RuntimeError("'/' on Int is exact division and " + std::to_string(x) +
                                       " is not divisible by " + std::to_string(y),
                                   e.pos);
            }
            return Value::from_int(x / y);
        case BinaryOp::Mod:
            if (y == 0) throw RuntimeError("division by zero in '%'", e.pos);
            if (x == kInt64Min && y == -1) return Value::from_int(0);
            return Value::from_int(x % y);
        case BinaryOp::Pow: {
            if (y < 0) {
                throw RuntimeError("'^' on Int needs a non-negative exponent (got " +
                                       std::to_string(y) + ")",
                                   e.pos);
            }
            std::int64_t acc = 1;
            std::int64_t base = x;
            std::int64_t exp = y;
            while (exp > 0) {
                if ((exp & 1) != 0) {
                    if (!MulChecked(acc, base, acc)) {
                        throw RuntimeError("Int overflow in '^'", e.pos);
                    }
                }
                exp >>= 1;
                if (exp > 0) {
                    if (!MulChecked(base, base, base)) {
                        throw RuntimeError("Int overflow in '^'", e.pos);
                    }
                }
            }
            return Value::from_int(acc);
        }
    }
    throw RuntimeError("unsupported operator", e.pos);
}

Value Evaluator::eval_compare(const Expr& e) {
    Value a = eval(e.lhs);
    Value b = eval(e.rhs);
    if (e.compare_op == CompareOp::Eq) return Value::from_bool(a.equals(b));
    if (e.compare_op == CompareOp::Ne) return Value::from_bool(!a.equals(b));

    if (a.is_int() && b.is_int()) {
        const std::int64_t x = a.int_value();
        const std::int64_t y = b.int_value();
        switch (e.compare_op) {
            case CompareOp::Lt: return Value::from_bool(x < y);
            case CompareOp::Le: return Value::from_bool(x <= y);
            case CompareOp::Gt: return Value::from_bool(x > y);
            case CompareOp::Ge: return Value::from_bool(x >= y);
            default: break;
        }
    }
    if (a.is_string() && b.is_string()) {
        const std::string& x = a.string_value();
        const std::string& y = b.string_value();
        switch (e.compare_op) {
            case CompareOp::Lt: return Value::from_bool(x < y);
            case CompareOp::Le: return Value::from_bool(x <= y);
            case CompareOp::Gt: return Value::from_bool(x > y);
            case CompareOp::Ge: return Value::from_bool(x >= y);
            default: break;
        }
    }
    throw TypeError(std::string("operator '") + CompareOpName(e.compare_op) +
                        "' is not defined on " + a.type_name() + " and " + b.type_name(),
                    e.pos);
}

Value Evaluator::eval_logical(const Expr& e) {
    if (e.logical_op == LogicalOp::Not) {
        Value v = eval(e.lhs);
        if (!v.is_bool()) throw TypeError("'not' expects Bool, got " + v.type_name(), e.pos);
        return Value::from_bool(!v.bool_value());
    }
    Value a = eval(e.lhs);
    if (!a.is_bool()) {
        throw TypeError(std::string("'") + (e.logical_op == LogicalOp::And ? "and" : "or") +
                            "' expects Bool, got " + a.type_name(),
                        e.pos);
    }
    // Short-circuit.
    if (e.logical_op == LogicalOp::And && !a.bool_value()) return Value::from_bool(false);
    if (e.logical_op == LogicalOp::Or && a.bool_value()) return Value::from_bool(true);
    Value b = eval(e.rhs);
    if (!b.is_bool()) {
        throw TypeError(std::string("'") + (e.logical_op == LogicalOp::And ? "and" : "or") +
                            "' expects Bool, got " + b.type_name(),
                        e.pos);
    }
    return Value::from_bool(b.bool_value());
}

Value Evaluator::eval_call(const Expr& e) {
    const VerbEntry* entry = session_.registry().find_verb(e.text);
    if (entry == nullptr) {
        if (session_.find_var(e.text) != nullptr) {
            throw TypeError("'" + e.text + "' is a variable, not a verb", e.pos);
        }
        throw TypeError("unknown verb '" + e.text + "'", e.pos);
    }

    std::vector<Value> positional;
    std::vector<std::pair<std::string, Value>> keyword;
    for (const Argument& a : e.args) {
        Value v = eval(a.value);
        if (a.name.empty()) positional.push_back(std::move(v));
        else keyword.emplace_back(a.name, std::move(v));
    }

    std::vector<Value> bound;
    const Overload* ov = session_.registry().resolve(*entry, positional, keyword, e.pos, bound);
    CONSOLE_CHECK(ov != nullptr);

    CallContext ctx;
    ctx.session = &session_;
    ctx.registry = &session_.registry();
    ctx.pos = e.pos;
    ctx.call_text = e.source_text;

    // Plan Sec. 2 C7: every verb call runs inside ONE try.  A ConsoleError is
    // already shaped; anything else is wrapped, with its C++ class named
    // through the registry's translator (which C1 installs, so Core needs no
    // pipeline header for the HeckeOverflow / ENGINE_CHECK catch order).
    try {
        PollCancel("call");   // (NBCONFIRM S5) before every verb call
        return ov->impl(ctx, bound);
    } catch (Interrupted& x) {   // (NBCONFIRM S5) not a ConsoleError: passed on, with the innermost call's position
        if (!x.pos.known()) x.pos = e.pos;
        throw;
    } catch (const ConsoleError&) {
        throw;
    } catch (const ExitSignal&) {
        throw;
    } catch (const QuitSignal&) {
        throw;
    } catch (const std::bad_alloc&) {
        throw RuntimeError(e.text + "(...): out of memory", e.pos);
    } catch (const std::exception& ex) {
        ThrowWrapped(session_.registry(), ex, e.text + "(...) failed", e.pos, nullptr);
    }
}

Value Evaluator::object_member(const ObjectPtr& obj, const std::string& name,
                               const SourcePos& pos) {
    CONSOLE_CHECK(obj != nullptr);
    const TypeInfo* type = obj->type();
    if (type == nullptr) {
        throw TypeError("this object has no registered type", pos);
    }
    const TypeInfo::Member* m = type->find_member(name);
    if (m == nullptr) {
        std::string msg = "type " + type->name + " has no member '" + name + "'";
        if (!type->members.empty()) {
            msg += " (members: ";
            for (std::size_t i = 0; i < type->members.size(); ++i) {
                if (i != 0) msg += ", ";
                msg += type->members[i].name;
            }
            msg += ")";
        }
        throw TypeError(msg, pos);
    }
    if (m->computed) {
        // Reading a computed member is what triggers the work (plan Sec. 4.2).
        // (NBCONFIRM S5) an Interrupted leaves the handle `declared` (Object.cpp catches only ConsoleError and
        // std::exception), so the next read retries; it gains this access's position.
        try {
            obj->ensure_computed(session_.settings(), pos);
        } catch (Interrupted& x) {
            if (!x.pos.known()) x.pos = pos;
            throw;
        }
    }
    CONSOLE_CHECK_MSG(static_cast<bool>(m->get), "a registered member must have a getter");
    // (C5 leftovers, slot-10 WP-12; [S10] decision 6 (i) (A), [P] v3.8 Sec. 2 C7
    // site 1 -- one of the two sites of the one named Console/Core exception.)
    // A getter's error gets the member access's file:line:col, in the catch
    // order of a method call: a ConsoleError keeps its kind and gains the
    // position if it has none; anything else is classified by the registry
    // exactly as for a method (ThrowWrapped), with the object's creation site.
    try {
        return m->get(obj);
    } catch (ConsoleError& ce) {
        ce.set_pos_if_unknown(pos);
        throw;
    } catch (const ExitSignal&) {
        throw;
    } catch (const QuitSignal&) {
        throw;
    } catch (const std::exception& ex) {
        ThrowWrapped(session_.registry(), ex, type->name + "." + name + " failed", pos, &obj);
    }
}

Value Evaluator::eval_member(const Expr& e) {
    Value base = eval(e.lhs);
    if (base.is_record()) {
        const Value* v = base.record_value().find(e.text);
        if (v == nullptr) {
            // `args.name` that the command line did not supply is a runtime
            // error (plan Sec. 6, exit code 3).
            const bool is_args = (e.lhs->kind == ExprKind::Args);
            throw RuntimeError(is_args ? "no script parameter named '" + e.text +
                                             "' was supplied"
                                       : "record has no key '" + e.text + "'",
                               e.pos);
        }
        return *v;
    }
    if (base.is_object()) return object_member(base.object_value(), e.text, e.pos);
    throw TypeError("'." + e.text + "' is not defined on " + base.type_name(), e.pos);
}

Value Evaluator::eval_method(const Expr& e) {
    Value base = eval(e.lhs);
    if (!base.is_object()) {
        throw TypeError("'." + e.text + "(...)' is not defined on " + base.type_name(), e.pos);
    }
    const ObjectPtr& obj = base.object_value();
    const TypeInfo* type = obj ? obj->type() : nullptr;
    if (type == nullptr) throw TypeError("this object has no registered type", e.pos);

    const TypeInfo::Method* m = type->find_method(e.text);
    if (m == nullptr) {
        // A member holding a callable is not a thing in this language, so a
        // missing method is a type error naming what does exist.
        std::string msg = "type " + type->name + " has no method '" + e.text + "'";
        if (!type->methods.empty()) {
            msg += " (methods: ";
            for (std::size_t i = 0; i < type->methods.size(); ++i) {
                if (i != 0) msg += ", ";
                msg += type->methods[i].name;
            }
            msg += ")";
        }
        throw TypeError(msg, e.pos);
    }

    std::vector<Value> args;
    for (const Argument& a : e.args) {
        if (!a.name.empty()) {
            // Keyword binding for methods follows the declared parameter names.
            std::size_t idx = m->params.size();
            for (std::size_t i = 0; i < m->params.size(); ++i) {
                if (m->params[i] == a.name) { idx = i; break; }
            }
            if (idx == m->params.size()) {
                throw TypeError("method " + type->name + "." + e.text +
                                    " has no parameter named '" + a.name + "'",
                                a.pos);
            }
            if (args.size() < idx + 1) args.resize(idx + 1);
            args[idx] = eval(a.value);
            continue;
        }
        args.push_back(eval(a.value));
    }
    if (args.size() < m->required || args.size() > m->params.size()) {
        std::string sig = type->name + "." + e.text + "(";
        for (std::size_t i = 0; i < m->params.size(); ++i) {
            if (i != 0) sig += ", ";
            sig += m->params[i];
        }
        sig += ")";
        throw TypeError("wrong number of arguments for " + sig + ": got " +
                            std::to_string(args.size()),
                        e.pos);
    }

    if (m->computed) {
        try {
            obj->ensure_computed(session_.settings(), e.pos);
        } catch (Interrupted& x) {   // (NBCONFIRM S5) the handle stays `declared`
            if (!x.pos.known()) x.pos = e.pos;
            throw;
        }
    }

    CONSOLE_CHECK_MSG(static_cast<bool>(m->call), "a registered method must have an implementation");
    try {
        PollCancel("call");   // (NBCONFIRM S5) before every method call
        return m->call(obj, args, e.pos);
    } catch (Interrupted& x) {
        if (!x.pos.known()) x.pos = e.pos;
        throw;
    } catch (const ConsoleError&) {
        throw;
    } catch (const ExitSignal&) {
        throw;
    } catch (const QuitSignal&) {
        throw;
    } catch (const std::exception& ex) {
        ThrowWrapped(session_.registry(), ex, type->name + "." + e.text + "(...) failed", e.pos,
                     &obj);
    }
}

Value Evaluator::eval_index(const Expr& e) {
    Value base = eval(e.lhs);
    Value idx = eval(e.rhs);
    if (base.is_list()) {
        if (!idx.is_int()) {
            throw TypeError("a List index must be Int, got " + idx.type_name(), e.pos);
        }
        const std::vector<Value>& items = base.list_value();
        const std::int64_t i = idx.int_value();
        if (i < 0 || static_cast<std::size_t>(i) >= items.size()) {
            throw RuntimeError("index " + std::to_string(i) + " is out of range for a List of " +
                                   std::to_string(items.size()),
                               e.pos);
        }
        return items[static_cast<std::size_t>(i)];
    }
    if (base.is_record()) {
        if (!idx.is_string()) {
            throw TypeError("a Record index must be String, got " + idx.type_name(), e.pos);
        }
        const Value* v = base.record_value().find(idx.string_value());
        if (v == nullptr) {
            throw RuntimeError("record has no key '" + idx.string_value() + "'", e.pos);
        }
        return *v;
    }
    throw TypeError("indexing is not defined on " + base.type_name(), e.pos);
}

// --- help ------------------------------------------------------------------

std::string Evaluator::help_text(const std::string& topic) const {
    const Registry& reg = session_.registry();

    if (topic.empty() || topic == "?") {
        std::string out = "OC script -- console language core (C0).\n";
        out += "statements: assignment, expression, for/in, if/else, set, include, "
               "help, ?name, expect_error, quit\n";
        out += "verbs:";
        const std::vector<std::string> names = reg.verb_names();
        for (std::size_t i = 0; i < names.size(); ++i) {
            out += (i == 0 ? " " : ", ");
            out += names[i];
        }
        const std::vector<std::string> types = reg.type_names();
        if (!types.empty()) {
            out += "\ntypes:";
            for (std::size_t i = 0; i < types.size(); ++i) {
                out += (i == 0 ? " " : ", ");
                out += types[i];
            }
        }
        out += "\n'help <name>' prints one verb's overloads; '?x' describes a value.";
        return out;
    }

    // `?x` on a bound variable describes the value (plan Sec. 4.4).
    const Value* var = session_.find_var(topic);
    const VerbEntry* verb = reg.find_verb(topic);
    if (verb != nullptr) {
        std::string out = "verb " + verb->name + "  [" + verb->group + "]";
        for (const Overload& o : verb->overloads) {
            out += "\n  " + o.sig.to_string(verb->name);
            if (!o.doc.empty()) out += "\n      " + o.doc;
        }
        return out;
    }
    if (var != nullptr) return Describe(*var);

    const TypeInfo* type = reg.find_type(topic);
    if (type != nullptr) {
        std::string out = "type " + type->name;
        if (!type->doc.empty()) out += "\n  " + type->doc;
        for (const TypeInfo::Member& m : type->members) {
            out += "\n  ." + m.name + (m.computed ? "  (computed)" : "  (structural)");
            if (!m.doc.empty()) out += "  -- " + m.doc;
        }
        for (const TypeInfo::Method& m : type->methods) {
            out += "\n  ." + m.name + "(";
            for (std::size_t i = 0; i < m.params.size(); ++i) {
                if (i != 0) out += ", ";
                out += m.params[i];
            }
            out += ")";
            if (!m.doc.empty()) out += "  -- " + m.doc;
        }
        return out;
    }

    const SettingSpec* spec = session_.settings().find_spec(topic);
    if (spec != nullptr) {
        std::string out = "setting " + spec->name;
        if (!spec->doc.empty()) out += " -- " + spec->doc;
        return out;
    }

    return "no help for '" + topic + "' (it is not a verb, a type, a variable or a setting)";
}

} // namespace console
