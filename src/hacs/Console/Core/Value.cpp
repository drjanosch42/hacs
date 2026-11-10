// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Value.cpp -- see Value.h.
// ===========================================================================

#include "Value.h"

#include "Object.h"

namespace console {

namespace {
const std::string& EmptyString() {
    static const std::string empty;
    return empty;
}
const std::vector<Value>& EmptyList() {
    static const std::vector<Value> empty;
    return empty;
}
const Record& EmptyRecord() {
    static const Record empty;
    return empty;
}
} // namespace

const char* ValueKindName(ValueKind kind) {
    switch (kind) {
        case ValueKind::Nil:    return "Nil";
        case ValueKind::Int:    return "Int";
        case ValueKind::Bool:   return "Bool";
        case ValueKind::String: return "String";
        case ValueKind::List:   return "List";
        case ValueKind::Record: return "Record";
        case ValueKind::Object: return "Object";
    }
    return "Nil";
}

Value Value::from_int(std::int64_t v) {
    Value out;
    out.kind_ = ValueKind::Int;
    out.i_ = v;
    return out;
}

Value Value::from_bool(bool v) {
    Value out;
    out.kind_ = ValueKind::Bool;
    out.i_ = v ? 1 : 0;
    return out;
}

Value Value::from_string(std::string v) {
    Value out;
    out.kind_ = ValueKind::String;
    out.s_ = std::make_shared<const std::string>(std::move(v));
    return out;
}

Value Value::from_list(std::vector<Value> v) {
    Value out;
    out.kind_ = ValueKind::List;
    out.l_ = std::make_shared<const std::vector<Value>>(std::move(v));
    return out;
}

Value Value::from_record(Record v) {
    Value out;
    out.kind_ = ValueKind::Record;
    out.r_ = std::make_shared<const Record>(std::move(v));
    return out;
}

Value Value::from_record_ptr(std::shared_ptr<const Record> v) {
    Value out;
    out.kind_ = ValueKind::Record;
    out.r_ = std::move(v);
    if (!out.r_) out.r_ = std::make_shared<const Record>();
    return out;
}

Value Value::from_object(std::shared_ptr<Object> v) {
    Value out;
    out.kind_ = ValueKind::Object;
    out.o_ = std::move(v);
    return out;
}

const std::string& Value::string_value() const {
    return s_ ? *s_ : EmptyString();
}

const std::vector<Value>& Value::list_value() const {
    return l_ ? *l_ : EmptyList();
}

const Record& Value::record_value() const {
    return r_ ? *r_ : EmptyRecord();
}

std::string Value::type_name() const {
    if (kind_ == ValueKind::Object) {
        return o_ ? o_->type_name() : std::string("Object");
    }
    return std::string(ValueKindName(kind_));
}

bool Value::equals(const Value& other) const {
    if (kind_ != other.kind_) return false;
    switch (kind_) {
        case ValueKind::Nil:
            return true;
        case ValueKind::Int:
        case ValueKind::Bool:
            return i_ == other.i_;
        case ValueKind::String:
            return string_value() == other.string_value();
        case ValueKind::List: {
            const std::vector<Value>& a = list_value();
            const std::vector<Value>& b = other.list_value();
            if (a.size() != b.size()) return false;
            for (std::size_t i = 0; i < a.size(); ++i) {
                if (!a[i].equals(b[i])) return false;
            }
            return true;
        }
        case ValueKind::Record: {
            const Record& a = record_value();
            const Record& b = other.record_value();
            if (a.size() != b.size()) return false;
            for (std::size_t i = 0; i < a.entries().size(); ++i) {
                if (a.entries()[i].first != b.entries()[i].first) return false;
                if (!a.entries()[i].second.equals(b.entries()[i].second)) return false;
            }
            return true;
        }
        case ValueKind::Object: {
            if (o_ == other.o_) return true;
            if (!o_ || !other.o_) return false;
            // Mathematical equality when the type registers one and both sides
            // are of that type (plan Sec. 3.2); handle identity otherwise.
            if (o_->type() != nullptr && o_->type() == other.o_->type() && o_->type()->equal) {
                return o_->type()->equal(o_, other.o_);
            }
            return false;
        }
    }
    return false;
}

void Record::set(const std::string& key, Value value) {
    for (std::pair<std::string, Value>& e : entries_) {
        if (e.first == key) {
            e.second = std::move(value);
            return;
        }
    }
    entries_.emplace_back(key, std::move(value));
}

const Value* Record::find(const std::string& key) const {
    for (const std::pair<std::string, Value>& e : entries_) {
        if (e.first == key) return &e.second;
    }
    return nullptr;
}

std::vector<std::string> Record::keys() const {
    std::vector<std::string> out;
    out.reserve(entries_.size());
    for (const std::pair<std::string, Value>& e : entries_) out.push_back(e.first);
    return out;
}

} // namespace console
