// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// CostModel.h -- NBCONFIRM S3 (slot 25): the console cost estimators of the
// capped verbs, a line-for-line port of the S2 reference evaluator
// docs/console/tools/nbconfirm_estimate.py (load_models, estimate, decide) and
// of the feature map docs/console/tools/nbconfirm_fit.py::feats.
//
// The model data is src/hacs/Console/cost/cost_models.tsv,
// embedded at CONFIGURE time (cmake/sources/Special.cmake writes it into a
// generated translation unit of ConsoleCore as one raw string literal).  It is
// the single source; nothing is read at run time.
//
// ADVISORY ONLY.  An estimate never changes a value, a record or a hash input:
// it reads arguments, the model table, the session's `threads` and (for the
// wall time only) the per-device speed profile.  The campaign driver and
// --bench never call it.  See docs/console/NBCONFIRM_design.md Sec. 2 and
// docs/console/NBCONFIRM_S2_calibration.md (Sec. 3, 4 and Fix round 1).
//
// STL-ONLY (plan Sec. 2 C8): the device facts this file needs (CPU affinity
// count, ELF build id of the running binary, MemAvailable, free space on C:)
// arrive as plain numbers from the host probe of Confirm.h, which a binding
// installs.
// ===========================================================================
#pragma once
#ifndef CONSOLE_COST_MODEL_H
#define CONSOLE_COST_MODEL_H

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace console {
namespace cost {

// One monomial: coef * prod_i x_i^{e_i}; an empty exponent list is the constant "1".
struct Monomial {
    double coef = 0.0;
    std::vector<std::pair<std::string, double>> exps;   // file order
};

// (verb, variant, regime, quantity, threads-slot): "descent|-|gen|wall|16".
using GroupKey = std::string;
GroupKey MakeKey(const std::string& verb, const std::string& var, const std::string& reg, const std::string& q,
                 const std::string& slot);

struct Safety {
    double band_floor = 1.5, x_exp = 2.0, strong_x = 2.0;
};

// The parsed model file (nbconfirm_estimate.py::load_models).
struct CostModels {
    std::map<GroupKey, std::vector<Monomial>> term;
    std::map<GroupKey, std::vector<std::pair<std::string, std::pair<double, double>>>> grid;   // feature -> (lo, hi), file order
    std::map<GroupKey, double> gmax;
    std::map<GroupKey, std::vector<std::string>> lowconf;
    std::map<GroupKey, double> resid;    // the band's log: max_log_ge1s, or max_log where that is "nan"
    std::map<std::string, Safety> safety;   // "wall" / "rss"
    std::map<std::string, std::string> klass;   // verb -> kernel class
    std::vector<std::string> fitted_on;
    // calib rows: (class, slot) -> (threads, seconds, digest, rss_mib or < 0 when the set carries none)
    struct Calib {
        unsigned threads = 1;
        double seconds = 0.0;
        std::string digest;
        double rss_mib = -1.0;
    };
    std::string calib_set;   // the set of the calib rows ("calib/v1", "calib/v2"); "" if none
    std::map<std::pair<std::string, std::string>, Calib> calib;
    // the slot names present per (verb, variant, regime, quantity), sorted as Python sorts strings
    std::map<std::string, std::vector<std::string>> slots;
};

// Parses a model file's text.  Throws std::runtime_error on a malformed row.
CostModels ParseCostModels(const std::string& text);
// The embedded cost_models.tsv text and its parse (parsed once).
const std::string& EmbeddedCostModelsText();
const CostModels& EmbeddedCostModels();

// The features of one call (nbconfirm_fit.py::feats).  `args` are the verb's
// arguments as strings, in the measurement file's spelling: integers, a prime
// list "p:q:r", the algebra name, or "-" for an absent forms_algebra list.
struct Features {
    std::string variant, regime;
    std::map<std::string, double> f;
};
bool CostFeatures(const std::string& verb, const std::vector<std::string>& args, Features& out, std::string& error);

// psi(N) = [SL2(Z) : Gamma_0(N)]  and the descent record's default primes (nbconfirm_fit.py).
std::uint64_t PsiIndex(std::uint64_t N);
std::vector<std::uint64_t> DefaultRecordPrimesOf(std::uint64_t N);

// The per-device speed profile (S2 Sec. 4.3).
struct DeviceProfile {
    bool loaded = false;
    std::string path;
    std::map<std::string, std::string> kv;
    std::map<std::pair<std::string, std::string>, double> factor;   // (class, slot) -> factor
};
// ${HACS_DEVICE_PROFILE:-${XDG_CONFIG_HOME:-$HOME/.config}/hacs/device_profile.tsv}; the literal "none" means no profile.
std::string DeviceProfilePath();
// Loads `path` (empty or "none" -> not loaded).  A missing or unreadable file -> not loaded.
DeviceProfile LoadDeviceProfile(const std::string& path);
// The staleness test (nbconfirm_calibrate.py::stale, S2 audit F4): calib set first, then the ELF build id when both sides have
// one, else size + mtime.  Returns "" when fresh, else the reason.
std::string ProfileStaleReason(const DeviceProfile& p, const std::string& calib_set, const std::string& build_id,
                               std::uint64_t binary_size, std::int64_t binary_mtime, bool have_stat);

struct CostEstimate {
    bool ok = false;           // false: no model for this (verb, variant, regime); `error` says why
    std::string error;
    std::string verb, model;   // model = "<verb>_cost/v1"; v2 for the threaded-RSS verbs (descent, hida_tower; S3 Fix round 1)
    unsigned T_eff = 1;
    double wall_s = 0.0, rss_mib = 0.0;          // central estimates (wall scaled by the device factor)
    double wall_band = 1.0, rss_band = 1.0;
    double wall_hi = 0.0, rss_hi = 0.0;          // the upper edges of the safety rule
    double x_wall = 1.0, x_rss = 1.0, x_factor = 1.0;
    bool extrapolated = false;
    std::vector<std::string> x_features, low_confidence;
    double device_factor = 1.0;
    std::string calibration = "uncalibrated";
};

// nbconfirm_estimate.py::estimate.  `profile` may be null (uncalibrated); `hw` is the CPU count of the affinity mask (0 = 1).
CostEstimate Estimate(const CostModels& M, const std::string& verb, const std::vector<std::string>& args, unsigned threads,
                      const DeviceProfile* profile, unsigned hw);

enum class CostLevel { Silent, Normal, Strong, Refuse };
const char* CostLevelName(CostLevel l);

struct CostDecision {
    CostLevel level = CostLevel::Silent;
    double kill_s = -1.0;                 // the S4 gateway wall kill; < 0 = none
    std::vector<std::string> reasons;
    double mem_limit_mib = -1.0;          // min(R_cap, 0.8 MemAvailable), < 0 when neither is known
};

// nbconfirm_estimate.py::decide for a call OUTSIDE its silent envelope (inside, nothing is estimated).  Unknown inputs are
// negative: mem_avail_mib < 0 (no /proc/meminfo), rss_cap_mib <= 0 (no session cap), c_free_gb < 0 (no C: drive).
CostDecision Decide(const CostModels& M, const CostEstimate& e, double confirm_wall_s, double mem_avail_mib,
                    double rss_cap_mib, double c_free_gb);

// Human text: "3 min 10 s", "2.1 GiB".
std::string FormatSeconds(double s);
std::string FormatMiB(double mib);

} // namespace cost
} // namespace console

#endif // CONSOLE_COST_MODEL_H
