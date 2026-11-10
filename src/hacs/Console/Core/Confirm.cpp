// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Confirm.cpp -- see Confirm.h.
// ===========================================================================

#include "Confirm.h"

#include "Cancel.h"

#include "CostModel.h"
#include "Session.h"
#include "Value.h"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <istream>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace console {

const char* const kSettingConfirm = "confirm";
const char* const kSettingConfirmWallS = "confirm_wall_s";
const char* const kDeclinedKind = "not run";

void RegisterConfirmSettings(Settings& settings) {
    SettingSpec c;
    c.name = kSettingConfirm;
    c.kind = SettingKind::String;
    c.default_value = Value::from_string("ask");
    c.family = SettingFamily::Engine;   // policy only; never part of a result, a record or a hash input
    c.allowed = {"ask", "yes", "no"};
    c.doc = "a call outside a verb's silent cost envelope (NBCONFIRM): `ask` = on a terminal REPL show the estimate and ask, "
            "in a served session answer confirm_required with a per-cell token, elsewhere refuse as before (exit 3); "
            "`yes` = run it (hard memory / disk refusals still refuse); `no` = always refuse as before; not settable in a "
            "served session";
    settings.register_setting(std::move(c));

    SettingSpec w;
    w.name = kSettingConfirmWallS;
    w.kind = SettingKind::Int;
    w.default_value = Value::from_int(kDefaultConfirmWallS);
    w.family = SettingFamily::Engine;
    w.doc = "seconds: on a terminal REPL or in a served session, a call outside its silent envelope whose estimated wall time "
            "(upper edge) is above this is confirmed before it runs (NBCONFIRM; default 60; serve-settable)";
    w.validate = [](const Value& v) -> std::string {
        return v.int_value() < 0 ? std::string("must be >= 0") : std::string();
    };
    settings.register_setting(std::move(w));
}

namespace {

std::mutex& ProbeMutex() {
    static std::mutex m;
    return m;
}
HostProbe& ProbeRef() {
    static HostProbe p;
    return p;
}

std::string Lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

std::string Trim(const std::string& s) {
    std::size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

void WritePrompt(const std::string& text) {
    const HostProbe& h = GetHostProbe();
    if (h.write_prompt) h.write_prompt(text);
    else std::cerr << text << std::flush;
}

double Probe(const std::function<double()>& f, double unknown) { return f ? f() : unknown; }

std::string Fixed(double v, const char* fmt) {
    char b[64];
    std::snprintf(b, sizeof b, fmt, v);
    return b;
}

// (S4) the served console's confirmation channel and run hook
std::mutex& ChannelMutex() {
    static std::mutex m;
    return m;
}
ConfirmChannel& ChannelRef() {
    static ConfirmChannel c;
    return c;
}
ConfirmedRunHook& RunHookRef() {
    static ConfirmedRunHook h;
    return h;
}

} // namespace

void SetHostProbe(HostProbe probe) {
    std::lock_guard<std::mutex> lk(ProbeMutex());
    ProbeRef() = std::move(probe);
}

const HostProbe& GetHostProbe() { return ProbeRef(); }

ScopedHostProbe::ScopedHostProbe(HostProbe probe) : saved_(GetHostProbe()) { SetHostProbe(std::move(probe)); }
ScopedHostProbe::~ScopedHostProbe() { SetHostProbe(std::move(saved_)); }

void SetConfirmChannel(ConfirmChannel channel) {
    std::lock_guard<std::mutex> lk(ChannelMutex());
    ChannelRef() = std::move(channel);
}
bool HasConfirmChannel() {
    std::lock_guard<std::mutex> lk(ChannelMutex());
    return static_cast<bool>(ChannelRef());
}
void SetConfirmedRunHook(ConfirmedRunHook hook) {
    std::lock_guard<std::mutex> lk(ChannelMutex());
    RunHookRef() = std::move(hook);
}
void NoteConfirmedRun(const std::string& label, const cost::CostEstimate& e) {
    ConfirmedRunHook h;
    {
        std::lock_guard<std::mutex> lk(ChannelMutex());
        h = RunHookRef();
    }
    if (h) h(label, e);
}

ActiveProfile CurrentDeviceProfile() {
    ActiveProfile a;
    a.profile = cost::LoadDeviceProfile(cost::DeviceProfilePath());
    if (!a.profile.loaded) {
        a.note = "uncalibrated";
        return a;
    }
    const HostProbe& h = GetHostProbe();
    const std::string bid = h.build_id ? h.build_id() : std::string();
    std::uint64_t size = 0;
    std::int64_t mtime = 0;
    const bool have = h.binary_stat ? h.binary_stat(size, mtime) : false;
    const std::string why =
        cost::ProfileStaleReason(a.profile, cost::EmbeddedCostModels().calib_set, bid, size, mtime, have);
    if (!why.empty()) {
        a.note = "stale calibration (" + why + ")";
        return a;
    }
    a.apply = true;
    return a;
}

ConfirmMode SessionConfirmMode(const Session& session) {
    const Value* v = session.settings().get(kSettingConfirm);
    if (v == nullptr || !v->is_string()) return ConfirmMode::Ask;
    if (v->string_value() == "yes") return ConfirmMode::Yes;
    if (v->string_value() == "no") return ConfirmMode::No;
    return ConfirmMode::Ask;
}

double SessionConfirmWallS(const Session& session) {
    const Value* v = session.settings().get(kSettingConfirmWallS);
    return (v != nullptr && v->is_int()) ? static_cast<double>(v->int_value()) : static_cast<double>(kDefaultConfirmWallS);
}

unsigned SessionThreads(const Session& session) {
    const Value* v = session.settings().get("threads");
    if (v == nullptr || !v->is_int() || v->int_value() < 1) return 1;
    return static_cast<unsigned>(v->int_value());
}

std::string DescribeEstimate(const cost::CostEstimate& e) {
    std::string s = "~ " + cost::FormatSeconds(e.wall_s) + " (up to ~ " + cost::FormatSeconds(e.wall_hi) + "), ~ " +
                    cost::FormatMiB(e.rss_mib) + " (up to ~ " + cost::FormatMiB(e.rss_hi) + ") at threads = " +
                    std::to_string(e.T_eff) + " (model " + e.model + "; " + e.calibration;
    if (e.extrapolated) s += "; extrapolated x" + Fixed(e.x_factor, "%.1f");
    if (!e.low_confidence.empty()) s += "; low confidence";
    s += ")";
    return s;
}

// ---------------------------------------------------------------------------

CostGate::CostGate(const CallContext& ctx, std::string who, std::string verb)
    : ctx_(ctx), who_(std::move(who)), verb_(std::move(verb)), call_(ctx.call_text.empty() ? who_ : ctx.call_text), pos_(ctx.pos) {}

void CostGate::outside(const std::string& message) {
    if (pending_.empty()) pending_ = message;
    // A call that refuses outside the envelope anyway stops HERE, at its first crossed bound, exactly as before S3 (no later
    // check runs: e.g. level_n's index(N) for a huge N).
    if (refuses_outside()) throw TypeError(pending_, pos_);
}

void CostGate::refuse(const std::string& message) const { throw TypeError(message, pos_); }

bool CostGate::refuses_outside() const {
    if (pending_.empty()) return false;
    const ConfirmMode mode = SessionConfirmMode(*ctx_.session);
    if (mode == ConfirmMode::No) return true;
    if (mode == ConfirmMode::Yes) return false;
    // ask: only a terminal REPL (S3) or a served session with its confirmation channel (S4) may ask -- and never inside an
    // expect_error block (S4: the block expects today's refusal; asking there would make it machine-dependent)
    if (ctx_.session->expect_error_depth() > 0) return true;
    return ctx_.session->confirm_input() == nullptr && !HasConfirmChannel();
}

CostAdmission CostGate::admit(const std::vector<std::string>& cost_args, const std::vector<std::string>& notes) {
    CostAdmission a;
    if (pending_.empty()) return a;   // inside the silent envelope: unchanged, nothing estimated
    if (refuses_outside()) throw TypeError(pending_, pos_);   // today's text, today's exit code (3)
    a.outside = true;
    Session& session = *ctx_.session;
    const HostProbe& h = GetHostProbe();
    const unsigned hw = h.cpu_count ? h.cpu_count() : 0;
    const unsigned threads = SessionThreads(session);
    const ActiveProfile prof = CurrentDeviceProfile();
    const cost::CostModels& M = cost::EmbeddedCostModels();
    a.estimate = cost::Estimate(M, verb_, cost_args, threads, prof.apply ? &prof.profile : nullptr, hw);
    cost::CostEstimate& e = a.estimate;
    if (!prof.apply) e.calibration = prof.note;
    // (S3 Fix round 1, audit M1) Decide also takes the no-model case (!e.ok: refused when a memory limit is known, else strong) and
    // refuses a non-finite or non-positive estimate in every mode; nothing here overrides it.
    a.decision = cost::Decide(M, e, SessionConfirmWallS(session), Probe(h.mem_available_mib, -1.0), Probe(h.rss_cap_mib, -1.0),
                              Probe(h.c_free_gb, -1.0));
    if (!e.ok) a.decision.reasons.back() += ": " + e.error;
    const cost::CostDecision& d = a.decision;
    if (d.level == cost::CostLevel::Refuse) {
        std::string msg = who_ + ": refused before the run (" + pending_ + "): ";
        const bool cfloor = !d.reasons.empty() && d.reasons.front().rfind("C:", 0) == 0;
        const bool unusable = !e.ok || (!d.reasons.empty() && d.reasons.front().rfind("no usable estimate", 0) == 0);
        if (unusable && !cfloor) {
            msg += (d.reasons.empty() ? std::string("no cost estimate") : d.reasons.front()) +
                   "; a call outside its silent envelope runs only with a finite estimate within the memory limit (rule 19; no "
                   "override)";
        } else if (cfloor) {
            msg += "free space on C: is " + Fixed(Probe(h.c_free_gb, -1.0), "%.0f") +
                   " GB, at or below the 10 GB floor (owner rule of 2026-10-04)";
        } else {
            msg += "estimated peak RSS up to " + cost::FormatMiB(e.rss_hi) + " exceeds " + cost::FormatMiB(d.mem_limit_mib) +
                   " = min(session RSS cap, 80 % of MemAvailable) (rule 19; no override)";
            if (e.T_eff > 1) {   // fewer threads lower the peak RSS of the threaded verbs (S2 Sec. 3.3)
                const cost::CostEstimate e1 = cost::Estimate(M, verb_, cost_args, 1, prof.apply ? &prof.profile : nullptr, hw);
                if (e1.ok && e1.rss_hi < e.rss_hi * 0.95)
                    msg += "; fewer threads lower the peak RSS: estimated up to " + cost::FormatMiB(e1.rss_hi) +
                           " at threads = 1";
            }
        }
        throw RuntimeError(msg, pos_);
    }
    const ConfirmMode mode = SessionConfirmMode(session);
    if (mode == ConfirmMode::Yes) {
        session.print_annotation("confirm = yes: " + who_ + " runs outside its silent envelope (" + pending_ + "; " +
                                 cost::CostLevelName(d.level) + "): " + DescribeEstimate(e));
        NoteConfirmedRun(call_, e);
        return a;
    }
    if (d.level == cost::CostLevel::Silent) return a;   // rule 4: just outside the old bound, cheap
    if (session.confirm_input() == nullptr) {
        // (S4) a served session: the channel returns when the cell carried this call's token, and throws otherwise.
        ConfirmChannel ch;
        {
            std::lock_guard<std::mutex> lk(ChannelMutex());
            ch = ChannelRef();
        }
        if (ch) {
            ConfirmQuery q;
            q.who = who_;
            q.verb = verb_;
            q.call = call_;
            q.pending = pending_;
            q.pos = pos_;
            q.cost_args = cost_args;
            q.estimate = e;
            q.decision = d;
            q.notes = notes;
            ch(q);
            a.asked = true;
            NoteConfirmedRun(call_, e);
            return a;
        }
    }
    // ask, on a terminal REPL (refuses_outside() was false)
    std::istream* in = session.confirm_input();
    std::string text = who_ + " is outside its silent envelope (" + pending_ + ")\n  estimated " + DescribeEstimate(e) + "\n";
    if (d.level == cost::CostLevel::Strong) {
        text += "  STRONG WARNING:";
        for (std::size_t i = 0; i < d.reasons.size(); ++i) text += (i ? "; " : " ") + d.reasons[i];
        text += "\n";
    }
    for (const std::string& n : notes) text += "  note: " + n + "\n";
    if (e.calibration == "uncalibrated")
        text += "  (no device profile: `hacs --calibrate` measures this device in about half a minute)\n";
    else if (e.calibration.rfind("stale", 0) == 0)
        text += "  (the device profile is stale; `hacs --calibrate` re-measures it)\n";
    WritePrompt(text + "run it? [y/N] ");
    std::string ans;
    if (in == nullptr || !std::getline(*in, ans)) {
        WritePrompt("\n");
        throw ConfirmDeclined(pos_);
    }
    // (NBCONFIRM S5) a Ctrl-C while the question was open (the read restarts, SA_RESTART) stops the call here, before its work
    PollCancel("confirm.answer");
    ans = Lower(Trim(ans));
    if (ans != "y" && ans != "yes") throw ConfirmDeclined(pos_);
    if (d.level == cost::CostLevel::Strong) {
        WritePrompt("this is a strong warning: type `yes` to run it: ");
        if (!std::getline(*in, ans)) throw ConfirmDeclined(pos_);
        PollCancel("confirm.answer");
        if (Lower(Trim(ans)) != "yes") throw ConfirmDeclined(pos_);
    }
    a.asked = true;
    NoteConfirmedRun(call_, e);
    return a;
}

} // namespace console
