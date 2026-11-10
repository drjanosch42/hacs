// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Value.h -- console C0: the value model.
//
// Plan Sec. 3.3 ("Core value types"): Int, Bool, String, List, Record, plus the
// Object handle of Sec. 2 C2 (defined in Object.h -- only forward-declared
// here, so a Value costs one shared_ptr and no pipeline type is ever named).
// `Nil` is not a user-visible type: it is what a statement that produces no
// value evaluates to (an assignment, `set`, `for`), and it never prints.
//
// Immutability: every compound payload is held by shared_ptr<const T>, so
// copying a Value is O(1) and no script can mutate a list it shares.  Plan
// Sec. 2 C2, "Logically immutable after construction".
//
// STL-ONLY (plan Sec. 2 C8).
// ===========================================================================
#pragma once
#ifndef CONSOLE_VALUE_H
#define CONSOLE_VALUE_H

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace console {

class Object;
class Record;

enum class ValueKind : std::uint8_t {
    Nil = 0,
    Int,
    Bool,
    String,
    List,
    Record,
    Object
};

const char* ValueKindName(ValueKind kind);

class Value {
public:
    Value() = default;   // Nil

    static Value nil() { return Value(); }
    static Value from_int(std::int64_t v);
    static Value from_bool(bool v);
    static Value from_string(std::string v);
    static Value from_list(std::vector<Value> v);
    static Value from_record(Record v);
    static Value from_record_ptr(std::shared_ptr<const Record> v);
    static Value from_object(std::shared_ptr<Object> v);

    ValueKind kind() const { return kind_; }
    bool is_nil() const { return kind_ == ValueKind::Nil; }
    bool is_int() const { return kind_ == ValueKind::Int; }
    bool is_bool() const { return kind_ == ValueKind::Bool; }
    bool is_string() const { return kind_ == ValueKind::String; }
    bool is_list() const { return kind_ == ValueKind::List; }
    bool is_record() const { return kind_ == ValueKind::Record; }
    bool is_object() const { return kind_ == ValueKind::Object; }

    // Unchecked accessors; the caller has already tested kind() (every call
    // site in the evaluator does, and reports a TypeError otherwise).
    std::int64_t int_value() const { return i_; }
    bool bool_value() const { return i_ != 0; }
    const std::string& string_value() const;
    const std::vector<Value>& list_value() const;
    const Record& record_value() const;
    const std::shared_ptr<Object>& object_value() const { return o_; }
    const std::shared_ptr<const Record>& record_ptr() const { return r_; }

    // "Int", "List", or the registered type name of an Object ("Matrix", ...).
    std::string type_name() const;

    // Deep structural equality: Int/Bool/String by value, List/Record
    // elementwise (Records also by key order), Object by handle identity
    // unless its TypeInfo registers an `equal` (C1 installs that for matrices
    // and polynomials -- plan Sec. 3.2, "Comparison on objects is `equal` as
    // mathematical objects").
    bool equals(const Value& other) const;

private:
    ValueKind kind_ = ValueKind::Nil;
    std::int64_t i_ = 0;
    std::shared_ptr<const std::string> s_;
    std::shared_ptr<const std::vector<Value>> l_;
    std::shared_ptr<const Record> r_;
    std::shared_ptr<Object> o_;
};

// An ordered, read-only key/value map.  Order is insertion order and is part of
// the canonical print form, so a Record built the same way always prints the
// same way (plan Sec. 2 C5, "Deterministic output by construction").  No hash
// container anywhere -- the same rule the engine follows on a result path.
class Record {
public:
    Record() = default;

    // Appends, or replaces in place when the key already exists (keeping its
    // original position, so replacement cannot reorder a record).
    void set(const std::string& key, Value value);
    const Value* find(const std::string& key) const;
    bool has(const std::string& key) const { return find(key) != nullptr; }

    const std::vector<std::pair<std::string, Value>>& entries() const { return entries_; }
    std::size_t size() const { return entries_.size(); }
    bool empty() const { return entries_.empty(); }

    std::vector<std::string> keys() const;

private:
    std::vector<std::pair<std::string, Value>> entries_;
};

} // namespace console

#endif // CONSOLE_VALUE_H
