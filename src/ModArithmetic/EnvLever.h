// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef ENGINE_ENV_LEVER_H
#define ENGINE_ENV_LEVER_H
// =====================================================================
// EnvLever.h -- ONE skeleton for the on/off environment levers (S24 CL-1;
// CODEAUDIT A7, C-08, F10; docs/implementation/slot24/CLEANUP_plan_1030.md)
// =====================================================================
//
// Header-only, STL-only.  About 30 levers repeat the same pieces: an atomic
// scoped override, a cached env reader with a "neither 0 nor 1" WARNING, an
// XEnabled(), a ScopedX RAII guard, a say-once EFFECTIVE line and an
// XEffectiveLine() with source=scoped|env|default.  The copies have DRIFTED;
// a Policy names each drift explicitly, so a migration keeps its lever's
// behaviour byte for byte:
//
//   Empty::Unset      ""  reads as unset   (HECKE_DESCENT_GOOD_SPARSE, ENGINE_GPU_SMITH2K)
//   Empty::Invalid    ""  reads as invalid (HECKE_DESCENT_GOOD_DENSE; HECKE_FORMS_FASTMUL
//                         and the other HeckeExt levers, which test e == nullptr only)
//   Match::Exact01    exactly "0" = off, exactly "1" = on, anything else invalid
//   Match::FirstChar  e[0] == '0' = off, e[0] == '1' = on, anything else = the default,
//                     silently (ENGINE_GPU_IMMA "off iff it starts with '0'";
//                     ENGINE_GPU_SMITH2K_VERIFY "on iff it starts with '1'")
//   invalid           always reads as the DEFAULT route ("stays on"); it prints the
//                     policy's WARNING (a printf format with ONE %s, the raw value; on
//                     stderr, at most once per lever) and reports src_invalid:
//                       "env"                         (HECKE_FORMS_FASTMUL: e != nullptr)
//                       "default"                     (HECKE_DESCENT_GOOD_SPARSE: -1)
//                       "env-invalid(default kept)"   (HECKE_DESCENT_GOOD_DENSE: -2)
//   Cache::Once       the env is read once per process (the static-lambda copies)
//   Cache::PerCall    re-read on every call (ENGINE_GPU_IMMA: validations switch it per arm)
//
// Two readings exist and both are exposed: env_enabled() is the environment's
// reading alone (what the say-once short lines print, e.g. FASTMUL's
// fast_env()), enabled() adds the Scoped override (what the route follows).
//
// A lever is a namespace-scope object with constant initialisation (constexpr
// constructor), so it is usable from any static initialiser:
//     constexpr engine::lever::Policy kFastMulPolicy{ ... };
//     engine::lever::EnvLever g_fastmul("HECKE_FORMS_FASTMUL", kFastMulPolicy);
//
// RETIRED MODE (S24 CL-L0; CLEANUP plan Sec. 2.4, 3.3; the rollback-lever retirement rule).
// A retired lever is declared as `engine::lever::RetiredLever` instead of EnvLever:
//   * OC_ENABLE_LEGACY=ON  -- RetiredLever IS EnvLever: the lever behaves exactly as before
//     its retirement (env reading, Scoped, sources, lines).
//   * OC_ENABLE_LEGACY=OFF -- the old route is compiled out, so the lever has ONE route:
//       - enabled() / env_enabled() return the policy's default (true for a rollback lever);
//       - the env reading that selected the old route (exactly "0" for a default-on lever,
//         per the policy's Match) is REFUSED at the first read:
//             REFUSED <NAME>=<value>: compiled out (OC_ENABLE_LEGACY=OFF)   (stderr, exit 1)
//         never ignored (the vacuous-arm lesson); an invalid reading keeps its WARNING;
//       - there is no Scoped (a caller that still forces the old route does not compile);
//       - source() is "retired", so the EFFECTIVE line prints `source=retired`.
//   The old-route branch at a call site sits under `#if OC_ENABLE_LEGACY` (its body in a
//   legacy/ .inc), so the OFF build carries the new route only.
// =====================================================================

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <ostream>
#include <string>

#include "LegacyConfig.h"

namespace engine {
namespace lever {

enum class Empty : unsigned char { Unset, Invalid };
enum class Match : unsigned char { Exact01, FirstChar };
enum class Cache : unsigned char { Once, PerCall };
enum class State : signed char { Unset = 0, Off = 1, On = 2, Invalid = 3 };

struct Policy {
    bool default_on = true;
    Empty empty = Empty::Unset;
    Match match = Match::Exact01;
    Cache cache = Cache::Once;
    const char* warning = nullptr;          // printf format, ONE %s; nullptr = silent
    const char* src_default = "default";    // the source text of an unset reading
    const char* src_env = "env";            // ... of a valid "0" / "1"
    const char* src_invalid = "env";        // ... of an invalid reading (see the table above)
};

// The pure parse of one raw value (nullptr = unset).  No I/O.
inline State read(const char* raw, const Policy& p) {
    if (raw == nullptr) return State::Unset;
    if (raw[0] == '\0') return p.empty == Empty::Unset ? State::Unset : State::Invalid;
    if (p.match == Match::FirstChar) {
        if (raw[0] == '0') return State::Off;
        if (raw[0] == '1') return State::On;
        return State::Invalid;
    }
    if (std::strcmp(raw, "0") == 0) return State::Off;
    if (std::strcmp(raw, "1") == 0) return State::On;
    return State::Invalid;
}

inline bool state_enabled(State s, const Policy& p) {
    return s == State::On ? true : s == State::Off ? false : p.default_on;
}

inline const char* state_source(State s, const Policy& p) {
    return s == State::Unset ? p.src_default : s == State::Invalid ? p.src_invalid : p.src_env;
}

// The WARNING text a reading prints ("" when none): the policy format with the raw value.
inline std::string warning_text(const char* raw, const Policy& p) {
    if (p.warning == nullptr || raw == nullptr || read(raw, p) != State::Invalid) return std::string();
    const int n = std::snprintf(nullptr, 0, p.warning, raw);
    if (n <= 0) return std::string();
    std::string out(static_cast<size_t>(n) + 1, '\0');
    std::snprintf(&out[0], out.size(), p.warning, raw);
    out.resize(static_cast<size_t>(n));
    return out;
}

// The reading a retired lever refuses: the valid reading that selects the non-default
// (old) route -- Off for a default-on lever, On for a default-off one.  Pure; no I/O.
inline bool retired_refuses(State s, const Policy& p) {
    return p.default_on ? s == State::Off : s == State::On;
}

class EnvLever {
public:
    constexpr EnvLever(const char* name, const Policy& policy) : name_(name), pol_(policy) {}
    EnvLever(const EnvLever&) = delete;
    EnvLever& operator=(const EnvLever&) = delete;

    const char* name() const { return name_; }
    const Policy& policy() const { return pol_; }
    const char* raw() const { return std::getenv(name_); }   // the value now
    bool is_set() const { return raw() != nullptr; }          // set, any value (the say-once condition)

    // The environment's reading (cached per the policy; warns at most once).
    State env_state() const {
        if (pol_.cache == Cache::PerCall) {
            const char* e = raw();
            const State s = read(e, pol_);
            refuse_old_env(s, e);
            if (s == State::Invalid) warn_once(e);
            return s;
        }
        std::call_once(once_, [this] {
            const char* e = raw();
            const State s = read(e, pol_);
            refuse_old_env(s, e);
            if (s == State::Invalid) warn_once(e);
            cached_.store(static_cast<int>(s), std::memory_order_relaxed);
        });
        return static_cast<State>(cached_.load(std::memory_order_relaxed));
    }
    bool env_enabled() const { return state_enabled(env_state(), pol_); }

    // The route's switch: a live Scoped override, else the environment's reading.
    bool enabled() const {
        const int o = scoped_.load(std::memory_order_relaxed);
        return o < 0 ? env_enabled() : o != 0;
    }
    const char* source() const {
        return scoped_.load(std::memory_order_relaxed) >= 0 ? "scoped" : state_source(env_state(), pol_);
    }

    // "EFFECTIVE <name> <key>=<on_text|off_text> source=<source()>" -- the common head of the
    // XEffectiveLine() copies; a caller appends its own counters.
    std::string effective_line(const char* key, const char* on_text, const char* off_text) const {
        return std::string("EFFECTIVE ") + name_ + " " + key + "=" + (enabled() ? on_text : off_text) + " source=" + source();
    }
    // "EFFECTIVE <name> <key>=<on_text|off_text> (<name>=<raw>)" -- the say-once short line; it
    // prints the ENV reading (env_enabled()), as the copies do.  Only meaningful when is_set().
    std::string short_line(const char* key, const char* on_text, const char* off_text) const {
        const char* e = raw();
        return std::string("EFFECTIVE ") + name_ + " " + key + "=" + (env_enabled() ? on_text : off_text) + " (" + name_ + "=" +
               (e ? e : "") + ")";
    }
    // Prints `line` + std::endl on `os` the first time it is called while the variable is set
    // (any value); true iff it printed.  mark_said() suppresses it (a caller that prints the full line).
    bool say_once(std::ostream& os, const std::string& line) const {
        if (!is_set() || said_.exchange(true)) return false;
        os << line << std::endl;
        return true;
    }
    void mark_said() const { said_.store(true); }

    // RAII override (nests; restores the previous override).
    class Scoped {
    public:
        Scoped(const EnvLever& l, bool on) : l_(l), saved_(l.scoped_.exchange(on ? 1 : 0)) {}
        ~Scoped() { l_.scoped_.store(saved_); }
        Scoped(const Scoped&) = delete;
        Scoped& operator=(const Scoped&) = delete;

    private:
        const EnvLever& l_;
        int saved_;
    };

protected:
    // (S25 LEVERS-25) EnvArmRetiredLever's constructor: refuse_old_env = true makes the env reading that selects the
    // old route (retired_refuses) a REFUSAL at the read; everything else (Scoped, sources, lines) is unchanged.
    constexpr EnvLever(const char* name, const Policy& policy, bool refuse_old_env)
        : name_(name), pol_(policy), refuse_old_env_(refuse_old_env) {}

private:
    void refuse_old_env(State s, const char* e) const {
        if (refuse_old_env_ && retired_refuses(s, pol_)) oc::legacy::refuse_compiled_out(std::string(name_) + "=" + e);
    }
    void warn_once(const char* e) const {
        if (pol_.warning == nullptr || warned_.exchange(true)) return;
        const std::string w = warning_text(e, pol_);
        std::fputs(w.c_str(), stderr);
    }

    const char* name_;
    Policy pol_;
    bool refuse_old_env_ = false;
    mutable std::atomic<int> scoped_{-1};
    mutable std::once_flag once_;
    mutable std::atomic<int> cached_{0};
    mutable std::atomic<bool> said_{false};
    mutable std::atomic<bool> warned_{false};
};


#if OC_ENABLE_LEGACY
// ON build: a retired lever is the live lever, unchanged.
using RetiredLever = EnvLever;
#else
// OFF build: one route; the old route's env reading is refused (see the header comment).
class RetiredLever {
public:
    constexpr RetiredLever(const char* name, const Policy& policy) : name_(name), pol_(policy) {}
    RetiredLever(const RetiredLever&) = delete;
    RetiredLever& operator=(const RetiredLever&) = delete;

    const char* name() const { return name_; }
    const Policy& policy() const { return pol_; }
    const char* raw() const { return std::getenv(name_); }
    bool is_set() const { return raw() != nullptr; }

    // The refusal text for the value now ("" when that value is admissible), without the
    // "REFUSED " prefix: "<NAME>=<value>: compiled out (OC_ENABLE_LEGACY=OFF)".  Pure query.
    std::string refusal_text() const {
        const char* e = raw();
        if (!retired_refuses(read(e, pol_), pol_)) return std::string();
        return oc::legacy::compiled_out_text(std::string(name_) + "=" + e);
    }

    // The environment's reading after the refusal check (a refusal exits 1, never returns).
    State env_state() const {
        if (pol_.cache == Cache::PerCall) return check(raw());
        std::call_once(once_, [this] { cached_.store(static_cast<int>(check(raw())), std::memory_order_relaxed); });
        return static_cast<State>(cached_.load(std::memory_order_relaxed));
    }
    bool env_enabled() const {
        (void)env_state();
        return pol_.default_on;
    }
    bool enabled() const { return env_enabled(); }
    const char* source() const {
        (void)env_state();
        return "retired";
    }

    std::string effective_line(const char* key, const char* on_text, const char* off_text) const {
        return std::string("EFFECTIVE ") + name_ + " " + key + "=" + (enabled() ? on_text : off_text) + " source=" + source();
    }
    std::string short_line(const char* key, const char* on_text, const char* off_text) const {
        const char* e = raw();
        return std::string("EFFECTIVE ") + name_ + " " + key + "=" + (env_enabled() ? on_text : off_text) + " (" + name_ + "=" +
               (e ? e : "") + ")";
    }
    bool say_once(std::ostream& os, const std::string& line) const {
        if (!is_set() || said_.exchange(true)) return false;
        os << line << std::endl;
        return true;
    }
    void mark_said() const { said_.store(true); }

private:
    State check(const char* e) const {
        const State s = read(e, pol_);
        if (retired_refuses(s, pol_)) oc::legacy::refuse_compiled_out(std::string(name_) + "=" + e);
        if (s == State::Invalid && pol_.warning != nullptr && !warned_.exchange(true)) std::fputs(warning_text(e, pol_).c_str(), stderr);
        return s;
    }

    const char* name_;
    Policy pol_;
    mutable std::once_flag once_;
    mutable std::atomic<int> cached_{0};
    mutable std::atomic<bool> said_{false};
    mutable std::atomic<bool> warned_{false};
};
#endif

// ENV-ARM RETIREMENT (S25 LEVERS-25; the rollback-lever rule's oracle / production exception).
// A lever whose OLD route must stay in every build -- the oracle of a surviving validation, or a route
// still production in some regime -- retires only its environment arm:
//   * OC_ENABLE_LEGACY=ON  -- exactly EnvLever (env reading, Scoped, sources, lines: unchanged);
//   * OC_ENABLE_LEGACY=OFF -- the env reading that selects the old route (retired_refuses) is REFUSED at
//     the read (`REFUSED <NAME>=<value>: compiled out (OC_ENABLE_LEGACY=OFF)`, stderr, exit 1); unset and
//     the other readings behave as before (an invalid reading keeps its WARNING); the Scoped override
//     stays, so a validation's comparison arm still reaches the old route; source() is unchanged
//     ("default" / "env" / "scoped"), so the EFFECTIVE lines anchored by run books do not move.
//   The read happens at the lever's first use and, in cohomology_app, once at start-up (main()).
class EnvArmRetiredLever : public EnvLever {
public:
    constexpr EnvArmRetiredLever(const char* name, const Policy& policy) : EnvLever(name, policy, OC_ENABLE_LEGACY == 0) {}
};

} // namespace lever
} // namespace engine

#endif // ENGINE_ENV_LEVER_H
