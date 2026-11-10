// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Session.cpp -- see Session.h.
// ===========================================================================

#include "Session.h"

#include "Printer.h"

#include <iostream>
#include <ostream>
#include <sstream>

namespace console {

void RegisterCoreSettings(Settings& settings, bool verbose_default) {
    SettingSpec cap;
    cap.name = kSettingPrintMaxEntries;
    cap.kind = SettingKind::Int;
    cap.default_value = Value::from_int(kDefaultPrintMaxEntries);
    cap.family = SettingFamily::Engine;      // presentation only; never part of a result
    cap.doc = "entries printed before a value prints its summary form";
    cap.validate = [](const Value& v) -> std::string {
        return v.int_value() < 0 ? std::string("must be >= 0") : std::string();
    };
    settings.register_setting(std::move(cap));

    SettingSpec verbose;
    verbose.name = kSettingVerbose;
    verbose.kind = SettingKind::Bool;
    verbose.default_value = Value::from_bool(verbose_default);
    verbose.family = SettingFamily::Engine;
    verbose.doc = "let the pipeline's own stdout/stderr chatter through "
                  "(default: true in the REPL, false in scripts)";
    settings.register_setting(std::move(verbose));
}

Session::Session(Registry& registry, bool verbose_default)
    : registry_(registry), out_(&std::cout) {
    RegisterCoreSettings(settings_, verbose_default);
    // (C1b) The engine-owned settings, through the registry's installer seam
    // (Registry.h, "the THIRD seam"): C0 registers the two presentation keys,
    // the bindings register the MATHEMATICAL ones of hecke::MathSettings.  A
    // C0-only registry has no installer, so this is a no-op there.
    registry_.install_settings(settings_);
    // The baseline `reset_settings()` restores (plan Sec. 2 C4).
    settings_.set_baseline(settings_.snapshot());
}

Session::~Session() {
    // (C2a, slot-6 WP-13) The session's `set` values are applied through plain
    // setters of state that OUTLIVES the session (the engine's thread-local and
    // process-global overrides, hecke::MathSettings).  Re-applying the baseline
    // on the way out is what keeps one script's `set threads = 1` or
    // `set backend = "legacy"` from leaking into whatever runs next in the same
    // process -- the next validation of an `--all` run, say.  Installing is
    // "apply every value" (no RAII stack, A2), so this is idempotent.
    try {
        settings_.reset_to_baseline();
    } catch (...) {
        // A destructor must not throw; a setter that refuses its own baseline
        // value is a binding bug the validations catch on the way in.
    }
}

void Session::set_var(const std::string& name, Value v) {
    for (std::pair<std::string, Value>& e : vars_) {
        if (e.first == name) {
            e.second = std::move(v);
            return;
        }
    }
    vars_.emplace_back(name, std::move(v));
}

const Value* Session::find_var(const std::string& name) const {
    for (const std::pair<std::string, Value>& e : vars_) {
        if (e.first == name) return &e.second;
    }
    return nullptr;
}

bool Session::erase_var(const std::string& name) {
    for (std::size_t i = 0; i < vars_.size(); ++i) {
        if (vars_[i].first == name) {
            vars_.erase(vars_.begin() + static_cast<std::ptrdiff_t>(i));
            return true;
        }
    }
    return false;
}

std::ostream& Session::out() {
    return direct_out_ ? *direct_out_ : *out_;
}

void Session::print_line(const std::string& text) {
    out() << text << "\n";
    if (log_ != nullptr) (*log_) << text << "\n";
}

void Session::print_annotation(const std::string& text) {
    print_line(std::string(kAnnotationPrefix) + text);
}

void Session::log_input(const std::string& line) {
    if (log_ != nullptr) (*log_) << "> " << line << "\n";
}

std::int64_t Session::print_max_entries() const {
    const Value* v = settings_.get(kSettingPrintMaxEntries);
    return (v != nullptr && v->is_int()) ? v->int_value() : kDefaultPrintMaxEntries;
}

bool Session::verbose() const {
    const Value* v = settings_.get(kSettingVerbose);
    return (v != nullptr && v->is_bool()) ? v->bool_value() : false;
}

// --- ChatterCapture --------------------------------------------------------

ChatterCapture::ChatterCapture(Session& session)
    : session_(session), sink_(new std::ostringstream()) {
    // Keep writing console output to the buffer std::cout had BEFORE the swap
    // (plan Sec. 2 C5: "The session captures the original std::cout streambuf
    // at start-up and writes console output there").
    if (session_.out_ == &std::cout && !session_.direct_out_) {
        session_.direct_out_.reset(new std::ostream(std::cout.rdbuf()));
        rebound_output_ = true;
    }
    old_cout_ = std::cout.rdbuf(sink_->rdbuf());
    old_cerr_ = std::cerr.rdbuf(sink_->rdbuf());
}

ChatterCapture::~ChatterCapture() {
    if (old_cout_ != nullptr) std::cout.rdbuf(old_cout_);
    if (old_cerr_ != nullptr) std::cerr.rdbuf(old_cerr_);
    if (rebound_output_) session_.direct_out_.reset();
}

std::string ChatterCapture::text() const {
    return sink_->str();
}

} // namespace console
