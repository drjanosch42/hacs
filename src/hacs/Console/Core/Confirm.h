// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Confirm.h -- NBCONFIRM S3 (slot 25): "ask before a long computation" instead
// of refusing it.  docs/console/NBCONFIRM_design.md Sec. 2.2-2.4 (with the S2
// corrections), owner rulings of 2026-11-06 (Q1-Q5, the S1 follow-up), [P]
// Sec. 7.4 and Sec. 8.
//
// THE RULE AT A COST-THROW SITE.  A verb validates its arguments as before,
// in the same order, through a CostGate:
//   * a mathematical / word-size / well-definedness precondition stays a
//     refusal:  gate.refuse(msg);
//   * a cost-only bound (the old "silent envelope") is recorded:  gate.outside(msg)
//     -- the FIRST such message is kept, because it is exactly the text the
//     verb threw before S3;
//   * at the decision point  gate.admit(cost_args)  returns at once when no
//     bound was crossed (inside the envelope: unchanged and silent, no estimate
//     is even computed), and otherwise applies the policy:
//
//       mode                                    confirm = ask        yes            no
//       prompt-less (script, -c, -, pipe, serve) today's TypeError    run (*)        today's TypeError
//       REPL on a terminal                      estimate + [y/N] (+)  run (*)        today's TypeError
//
//     (*) a HARD refusal (estimated peak RSS above min(session cap, 80 % of
//     MemAvailable), or C: <= 10 GB free) is a RuntimeError in every mode;
//     (+) silent when the estimate is small (rule 4), asked once (normal) or
//     twice with a typed `yes` (strong).  A declined prompt is ConfirmDeclined:
//     the REPL prints `not run`, nothing is bound.
//
//   The prompt-less refusal is independent of the estimate and of the device
//   profile (S2 audit F5), so a script's exit code is machine-independent; a
//   math refusal that follows a crossed bound in the same call reports the
//   crossed bound's text, as before S3 (the first failing check won).
//
// STL-ONLY: every device fact arrives through the HostProbe a binding installs
// (Console/Bindings/HostProbe.cpp: isatty, /proc/meminfo, statvfs of /mnt/c,
// sched_getaffinity, the ELF build id of /proc/self/exe).  Without a probe
// (a C0 session, a validation) nothing is a terminal and every device fact is
// unknown.
// ===========================================================================
#pragma once
#ifndef CONSOLE_CONFIRM_H
#define CONSOLE_CONFIRM_H

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "ConsoleError.h"
#include "CostModel.h"
#include "Registry.h"
#include "Settings.h"

namespace console {

extern const char* const kSettingConfirm;        // "confirm"
extern const char* const kSettingConfirmWallS;   // "confirm_wall_s"
constexpr std::int64_t kDefaultConfirmWallS = 60;   // owner Q1, 2026-11-06

// Registers the two keys (String ask|yes|no, default ask; Int >= 0, default 60).  Engine family: never mathematical.
void RegisterConfirmSettings(Settings& settings);

// A declined confirmation prompt (REPL only).  RunRepl prints `not run` for it; it is never seen in a prompt-less mode.
extern const char* const kDeclinedKind;   // "not run"
class ConfirmDeclined : public ConsoleError {
public:
    explicit ConfirmDeclined(SourcePos pos) : ConsoleError(kDeclinedKind, "not run", std::move(pos)) {}
};

// What the host can tell the console.  Every member may be empty (= unknown / no).
struct HostProbe {
    std::function<bool()> prompt_tty;                      // stdin AND stderr are terminals
    std::function<void(const std::string&)> write_prompt;  // write to the REAL stderr (not through a ChatterCapture)
    std::function<double()> mem_available_mib;            // /proc/meminfo MemAvailable; < 0 unknown
    std::function<double()> c_free_gb;                    // free GB on C: (/mnt/c); < 0 unknown (no C: drive)
    std::function<unsigned()> cpu_count;                  // CPUs of the affinity mask; 0 unknown
    std::function<double()> rss_cap_mib;                  // the session's --rss-cap-mib; <= 0 none (S4 wires --serve)
    std::function<std::string()> build_id;                // ELF build id of the running binary; "" none
    std::function<bool(std::uint64_t&, std::int64_t&)> binary_stat;   // size, mtime of the running binary
};
void SetHostProbe(HostProbe probe);
const HostProbe& GetHostProbe();
// For validations: installs `probe` for the scope and restores the previous one.
class ScopedHostProbe {
public:
    explicit ScopedHostProbe(HostProbe probe);
    ~ScopedHostProbe();
    ScopedHostProbe(const ScopedHostProbe&) = delete;
    ScopedHostProbe& operator=(const ScopedHostProbe&) = delete;

private:
    HostProbe saved_;
};

// The device profile the estimator uses now: loaded from DeviceProfilePath(), staleness-tested against the embedded calib set
// and the running binary.  `note` is "" (fresh), "uncalibrated" (none), or "stale calibration (<reason>)" -- a stale profile
// is not applied (factor 1).
struct ActiveProfile {
    cost::DeviceProfile profile;
    bool apply = false;
    std::string note;
};
ActiveProfile CurrentDeviceProfile();

enum class ConfirmMode { Ask, Yes, No };
ConfirmMode SessionConfirmMode(const Session& session);
double SessionConfirmWallS(const Session& session);
unsigned SessionThreads(const Session& session);   // the `threads` setting (1 when absent)

struct CostAdmission {
    bool outside = false;               // false: inside the silent envelope, nothing estimated
    cost::CostEstimate estimate;        // set when outside
    cost::CostDecision decision;
    bool asked = false;                 // a prompt was shown and answered yes
};

class CostGate {
public:
    CostGate(const CallContext& ctx, std::string who, std::string verb);

    // A cost-only bound was crossed; the first message is kept (it is today's refusal text).  When this call refuses outside
    // the envelope anyway (refuses_outside()), it throws that TypeError at once: the pre-S3 control flow, no later check runs.
    void outside(const std::string& message);
    // A precondition that stays a refusal: throws TypeError(message) at the call site.
    [[noreturn]] void refuse(const std::string& message) const;
    // Runs the argument checks `f`; a ConsoleError escaping it after a crossed bound is replaced by the crossed bound's
    // TypeError whenever this call would refuse outside the envelope anyway (byte-identical to the pre-S3 first failure).
    template <class F>
    void validate(F&& f) {
        try {
            f();
        } catch (const ConsoleError&) {
            if (!pending_.empty() && refuses_outside()) throw TypeError(pending_, pos_);
            throw;
        }
    }
    bool is_outside() const { return !pending_.empty(); }
    const std::string& pending() const { return pending_; }

    // The decision point.  `cost_args` are the estimator's arguments (CostModel.h, CostFeatures).  `notes` are shown in a
    // prompt ("the record may come back refused").  Returns when the call may run; throws otherwise (see the file comment).
    CostAdmission admit(const std::vector<std::string>& cost_args, const std::vector<std::string>& notes = {});

    // True when this call would refuse every crossed bound without estimating (prompt-less under ask, or confirm = no).
    bool refuses_outside() const;

private:
    const CallContext& ctx_;
    std::string who_, verb_, call_;
    SourcePos pos_;
    std::string pending_;
};

// ---------------------------------------------------------------------------
// (NBCONFIRM S4) The served console's confirmation channel ([P] Sec. 18.4, design Sec. 2.5).
//
// A front end without a terminal prompt may install a channel.  Under confirm = ask a call outside its silent envelope then
// gets rules 1-4 exactly as on a terminal REPL (rule 1 refuses, rule 4 runs silently), and for a normal or strong warning
// the channel is asked instead of a [y/N] prompt: it RETURNS when the call is confirmed (the served cell carried the call's
// token) and THROWS otherwise (the served console throws its own non-ConsoleError type, which unwinds to the cell as
// `done{status:"confirm_required"}`).  Only `--serve` installs one (Console/Bindings/Serve.cpp); without one every mode
// behaves as in S3.  The channel never sees a hard refusal (no token exists for one).
struct ConfirmQuery {
    std::string who;                     // "descent(N, n, primes)"
    std::string verb;                    // the estimator's verb ("descent")
    std::string call;                    // the call text ("descent(1, 62)"), or `who` when unknown
    std::string pending;                 // the crossed bound's text (today's refusal)
    SourcePos pos;                       // the call site
    std::vector<std::string> cost_args;  // the estimator's arguments
    cost::CostEstimate estimate;
    cost::CostDecision decision;
    std::vector<std::string> notes;
};
using ConfirmChannel = std::function<void(const ConfirmQuery&)>;
void SetConfirmChannel(ConfirmChannel channel);   // process-global; an empty function removes it
bool HasConfirmChannel();

// (NBCONFIRM S4) A confirmed call outside its silent envelope starts its work now: the served console arms its heartbeat
// (`progress` frames every 10 s) and tells the gateway the estimate.  CostGate::admit calls it for every confirmed or
// `confirm = yes` call; a lazy verb (descent) calls it again when its deferred work starts (its first member read).  No-op
// without a hook (every front end but --serve).
using ConfirmedRunHook = std::function<void(const std::string& label, const cost::CostEstimate& e)>;
void SetConfirmedRunHook(ConfirmedRunHook hook);
void NoteConfirmedRun(const std::string& label, const cost::CostEstimate& e);

// The text of one estimate, as the prompt and the `#~` annotation show it:
// "~ 3 min 10 s (up to ~ 9 min 2 s), ~ 2.1 GiB (up to ~ 3.2 GiB) at threads = 16 (model descent_cost/v1; uncalibrated; extrapolated x1.9)"
std::string DescribeEstimate(const cost::CostEstimate& e);

} // namespace console

#endif // CONSOLE_CONFIRM_H
