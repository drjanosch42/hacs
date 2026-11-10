// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions phase E2 cont. (slot 5, WP-2) -- the campaign's extension
// seam.  Design contract, and why the pass runs on the main thread:
// CampaignExt.h.

#include "CampaignExt.h"
#include "ProgramCacheRelease.h"   // slot-8 WP-5: the between-cells release

#include "../HeckeExt/HeckeExtDriver.h"
#include "../Util/FieldDiet.h"
#include "../HeckeExt/HeckeExtRecord.h"
#include "../HeckeExt/HeckeExtReport.h"
#include "../HeckeExt/HeckeUEllBudget.h"
#include "../Hecke/HeckeSettings.h"
#include "../Util/Rss.h"
#include "../../ModArithmetic/Fnv.h"
#include "../../Engine/EngineRuntime.h"   // slot-8 WP-6: capture_thread_context
#include "../../Engine/ThreadPool.h"      // slot-8 WP-6: per-worker pools
#include "CampaignDriver.h"                             // slot-8 WP-6: PerJobPoolsSupported
#include "../../Cuda/CudaCtx.h"            // slot-8 WP-6b: available(), device_info()
#include "../Util/TextIo.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace campaign {

namespace {

using Clock = std::chrono::steady_clock;

double Secs(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

using oc_util::HexU64;
using oc_util::ReadFile;   // S24 CL-7 (F11): Util/TextIo.h

// -------------------------------------------------------------------------
// A depth-aware scanner over the record JSON.  It is NOT a parser: it finds a
// key at a GIVEN nesting depth and returns the extent of its value, which is
// all the resume path needs and is what keeps the record-level `"u_ell"`
// (depth 1) from being confused with the `"u_ell"` inside `"settings"`
// (depth 2).  The writer emits no escaped characters outside plain strings
// (HeckeExtReport's CheckPlain), but the scanner respects `\` anyway.
// -------------------------------------------------------------------------
class JsonScan {
public:
    explicit JsonScan(const std::string& text) : t_(text) {}

    // The half-open extent [begin, end) of the VALUE of `key` at nesting
    // depth `want_depth` inside [from, to).  Returns false when absent.
    bool find_value(const std::string& key, uint32_t want_depth, size_t from, size_t to,
                    size_t& begin, size_t& end) const {
        uint32_t depth = 0;
        for (size_t i = from; i < to; ++i) {
            const char c = t_[i];
            if (c == '"') {
                // A key is a string immediately followed by ':'.
                const size_t key_start = i + 1;
                size_t j = key_start;
                while (j < to && t_[j] != '"') { if (t_[j] == '\\') ++j; ++j; }
                if (j >= to) return false;
                size_t k = j + 1;
                while (k < to && (t_[k] == ' ' || t_[k] == '\n')) ++k;
                if (k < to && t_[k] == ':' && depth == want_depth &&
                    t_.compare(key_start, j - key_start, key) == 0) {
                    ++k;
                    while (k < to && (t_[k] == ' ' || t_[k] == '\n')) ++k;
                    begin = k;
                    end = value_end(k, to);
                    return true;
                }
                i = j;              // skip the string we just walked
                continue;
            }
            if (c == '{' || c == '[') ++depth;
            else if (c == '}' || c == ']') { if (depth == 0) return false; --depth; }
        }
        return false;
    }

    // The end of the value starting at `p`.
    size_t value_end(size_t p, size_t to) const {
        if (p >= to) return to;
        const char c = t_[p];
        if (c == '{' || c == '[') {
            uint32_t depth = 0;
            bool in_string = false;
            for (size_t i = p; i < to; ++i) {
                const char d = t_[i];
                if (in_string) {
                    if (d == '\\') { ++i; continue; }
                    if (d == '"') in_string = false;
                    continue;
                }
                if (d == '"') { in_string = true; continue; }
                if (d == '{' || d == '[') ++depth;
                else if (d == '}' || d == ']') { if (--depth == 0) return i + 1; }
            }
            return to;
        }
        if (c == '"') {
            for (size_t i = p + 1; i < to; ++i) {
                if (t_[i] == '\\') { ++i; continue; }
                if (t_[i] == '"') return i + 1;
            }
            return to;
        }
        size_t i = p;
        while (i < to && t_[i] != ',' && t_[i] != '}' && t_[i] != ']') ++i;
        return i;
    }

    std::string slice(size_t a, size_t b) const { return t_.substr(a, b - a); }

private:
    const std::string& t_;
};

bool ScanBool(const JsonScan& s, const std::string& key, uint32_t depth, size_t from, size_t to,
              bool& out) {
    size_t a = 0, b = 0;
    if (!s.find_value(key, depth, from, to, a, b)) return false;
    const std::string v = s.slice(a, b);
    out = (v == "true");
    return true;
}

bool ScanU32(const JsonScan& s, const std::string& key, uint32_t depth, size_t from, size_t to,
             uint32_t& out) {
    size_t a = 0, b = 0;
    if (!s.find_value(key, depth, from, to, a, b)) return false;
    const std::string v = s.slice(a, b);
    uint64_t acc = 0;
    bool any = false;
    for (char c : v) {
        if (c < '0' || c > '9') { if (any) break; else continue; }
        any = true;
        acc = acc * 10ull + static_cast<uint64_t>(c - '0');
        if (acc > 0xffffffffull) { acc = 0xffffffffull; break; }
    }
    if (!any) return false;
    out = static_cast<uint32_t>(acc);
    return true;
}

bool ScanString(const JsonScan& s, const std::string& key, uint32_t depth, size_t from, size_t to,
                std::string& out) {
    size_t a = 0, b = 0;
    if (!s.find_value(key, depth, from, to, a, b)) return false;
    std::string v = s.slice(a, b);
    if (v.size() >= 2 && v.front() == '"' && v.back() == '"') v = v.substr(1, v.size() - 2);
    out = v;
    return true;
}

// Every `"state_ref":{"path":..,"fnv64":..}` the record names, verified
// against the sidecar beside it ([X] R3.1 item 6).  The record is the
// campaign's persisted state and a resume reads it, so a missing or altered
// sidecar is a REFUSAL BY NAME and never a silent fresh run.
uint32_t VerifySidecars(const std::string& text, const std::string& dir,
                        const std::string& record_path, uint32_t& named) {
    named = 0;
    uint32_t verified = 0;
    size_t pos = 0;
    const std::string kNeedle = "\"state_ref\":";
    const JsonScan scan(text);
    for (;;) {
        pos = text.find(kNeedle, pos);
        if (pos == std::string::npos) break;
        const size_t obj = pos + kNeedle.size();
        const size_t obj_end = scan.value_end(obj, text.size());
        ++named;
        std::string rel, want_hex;
        if (!ScanString(scan, "path", 1, obj, obj_end, rel) ||
            !ScanString(scan, "fnv64", 1, obj, obj_end, want_hex)) {
            throw std::runtime_error("campaign: malformed state_ref in '" + record_path + "'");
        }
        std::string body;
        const bool ok = ReadFile(dir.empty() ? rel : (dir + "/" + rel), body);
        if (!ok) {
            throw std::runtime_error("campaign: the resumable state sidecar '" + rel +
                                     "' named by '" + record_path + "' is missing");
        }
        if (HexU64(engine::FnvBytes(engine::kFnvOffset64, body), false) != want_hex) {
            throw std::runtime_error("campaign: the resumable state sidecar '" + rel +
                                     "' named by '" + record_path +
                                     "' does not match its recorded fnv64");
        }
        ++verified;
        pos = obj_end;
    }
    return verified;
}

std::string JoinDistinct(const std::vector<std::string>& v) {
    std::string out;
    for (const std::string& s : v) {
        bool seen = false;
        size_t at = 0;
        while ((at = out.find(s, at)) != std::string::npos) {
            const bool left = (at == 0) || out[at - 1] == ',';
            const bool right = (at + s.size() == out.size()) || out[at + s.size()] == ',';
            if (left && right) { seen = true; break; }
            ++at;
        }
        if (seen) continue;
        if (!out.empty()) out += ",";
        out += s;
    }
    return out;
}

// -------------------------------------------------------------------------
// The eight summary.tsv values of [X] Sec. 1.5, derived from the record.
//
// [X] NAMES the columns and does not DEFINE `nonsplit_ideals` / `max_loewy`
// (slot-5 WP-2 report, finding 2).  This package defines them, in band:
//
//   max_loewy        the longest Loewy series of any module of any ideal,
//                    i.e. max over ideals and ambient groups of
//                    ModuleStructure::loewy.size();
//   nonsplit_ideals  the ideals carrying a module of Loewy length >= 2, i.e.
//                    a NON-SEMISIMPLE block -- the module is a non-split
//                    extension of its layers.  At (10, 691) the one ideal has
//                    loewy = [2, 1], so nonsplit_ideals = 1, which is the
//                    nonsplit Eisenstein/cuspidal gluing of [E] `cl:ramanujan`
//                    and the amendment A-F2 oracle.
// -------------------------------------------------------------------------
void FillSummary(const hecke_ext::HeckeExtRecord& rec, ExtCellSummary& out) {
    out.ideals = static_cast<uint32_t>(rec.ideals.size());
    std::vector<std::string> certs;
    bool any_false = false, any_heuristic = false;
    for (const hecke_ext::IdealRecord& id : rec.ideals) {
        out.max_embdim = std::max(out.max_embdim, id.embdim);
        if (!id.local) ++out.nonlocal_ideals;
        uint32_t loewy_here = 0;
        for (const hecke_ext::ModuleRecordEntry& m : id.modules)
            loewy_here = std::max(loewy_here, static_cast<uint32_t>(m.structure.loewy.size()));
        out.max_loewy = std::max(out.max_loewy, loewy_here);
        if (loewy_here >= 2) ++out.nonsplit_ideals;
        if (id.sturm.present && id.sturm.has_B_prime) {
            if (!out.has_sturm_B_prime || id.sturm.B_prime > out.sturm_B_prime) {
                out.has_sturm_B_prime = true;
                out.sturm_B_prime = id.sturm.B_prime;
            }
        }
        certs.push_back(id.certificate);
        if (id.final.value == hecke_ext::Final::False) any_false = true;
        if (id.final.value == hecke_ext::Final::Heuristic) any_heuristic = true;
    }
    out.certificate = certs.empty() ? std::string("-") : JoinDistinct(certs);
    out.final_verdict = rec.ideals.empty() ? std::string("-")
                      : (any_false ? "false" : (any_heuristic ? "heuristic" : "true"));
    out.resumable_items = rec.resumable_items();
    out.u_ell_included = rec.u_ell.present && rec.u_ell.included;
    out.u_ell_reason = rec.u_ell.present ? rec.u_ell.reason : std::string("unavailable");
    out.verify_code = rec.verify();
    out.summary_known = true;
}

} // namespace

// ---------------------------------------------------------------------------
// Public helpers
// ---------------------------------------------------------------------------

std::string ExtRecordRelPath(uint32_t n_sym, uint64_t ell, const std::string& ext_tag) {
    // [X] Sec. 1.4's default name, under an `ext/` subdirectory.  It carries
    // NO settings fingerprint: the record is the cell's persisted state, and
    // the resume path's whole job is to read the PREVIOUS budget's record and
    // continue it -- a fingerprinted name would hide it from its successor.
    //
    // (slot-6 WP-6) It DOES carry the `family` identity ([WP5-2] finding 3):
    // an `_x<16 hex>` suffix, the FNV-1a-64 of `ext_tag` (ExtSpec::
    // signature_tag()), empty (no suffix) only when `ext_tag` itself is
    // empty -- i.e. never in production, since RunExtPass returns before
    // this is called unless `spec.ext.ext` is true, and `signature_tag()` is
    // non-empty exactly then.  A resume under a different `family` therefore
    // computes a different suffix, finds no file, and re-runs the cell.
    std::string path =
        "ext/hecke_ext_ell" + std::to_string(ell) + "_n" + std::to_string(n_sym);
    if (!ext_tag.empty()) {
        const uint64_t h = engine::FnvBytes(engine::kFnvOffset64, ext_tag);
        std::ostringstream tag;
        tag << "_x" << std::hex << std::setfill('0') << std::setw(16) << h;
        path += tag.str();
    }
    path += ".json";
    return path;
}

const ExtCellSummary* ExtPassResult::find(uint32_t n_sym, uint64_t ell) const {
    ExtCellKey k;
    k.n_sym = n_sym;
    k.ell = ell;
    const auto it = cells.find(k);
    return it == cells.end() ? nullptr : &it->second;
}

PersistedUEll ReadPersistedUEll(const std::string& output_dir, uint32_t n_sym, uint64_t ell,
                                const std::string& ext_tag) {
    PersistedUEll out;
    const std::string rel = ExtRecordRelPath(n_sym, ell, ext_tag);
    const std::string path = output_dir + "/" + rel;
    std::string text;
    const bool ok = ReadFile(path, text);
    if (!ok || text.empty()) return out;
    out.file_found = true;

    // The sidecars first: a corrupt one refuses by name before any decision is
    // taken on the record's contents.
    std::filesystem::path p(path);
    out.sidecars_verified =
        VerifySidecars(text, p.parent_path().string(), rel, out.sidecars_named);

    JsonScan scan(text);
    size_t a = 0, b = 0;
    {
        // `complexes` is at depth 1, so its `pair` member is at depth 2.
        size_t ca = 0, cb = 0;
        if (scan.find_value("complexes", 1, 0, text.size(), ca, cb)) {
            size_t pa = 0, pb = 0;
            // The writer emits `"status":"computed"` on an AVAILABLE complex
            // and `"status":"unavailable"` on the item-6(c) placeholder, so
            // the placeholder -- not the key -- is what to look for.
            if (scan.find_value("pair", 1, ca, cb, pa, pb))
                out.pair_available =
                    scan.slice(pa, pb).find("\"status\":\"unavailable\"") == std::string::npos;
        }
    }
    // Depth 1 -- the CELL's object.  The `"u_ell"` of the settings block is at
    // depth 2 and is therefore invisible here.
    if (!scan.find_value("u_ell", 1, 0, text.size(), a, b)) return out;
    // `{"status":"unavailable"}` is [C] `sec:recordschema` item 6(c)'s
    // placeholder: the cell formed no verdict at all.
    if (scan.slice(a, b).find("\"status\":\"unavailable\"") != std::string::npos) return out;
    out.object_found = true;
    ScanBool(scan, "included", 1, a, b, out.included);
    ScanString(scan, "reason", 1, a, b, out.reason);
    ScanU32(scan, "budget_mb", 1, a, b, out.budget_mb);

    size_t ra = 0, rb = 0;
    if (scan.find_value("resumable", 1, a, b, ra, rb)) {
        bool complete = true;
        ScanBool(scan, "complete", 1, ra, rb, complete);
        out.resumable = !complete;
        size_t ba = 0, bb = 0;
        if (scan.find_value("budget", 1, ra, rb, ba, bb))
            ScanU32(scan, "budget_mb", 1, ba, bb, out.resumable_budget_mb);
        size_t ha = 0, hb = 0;
        if (scan.find_value("budget_history", 1, ra, rb, ha, hb)) {
            uint64_t acc = 0;
            bool any = false;
            for (size_t i = ha; i < hb; ++i) {
                const char c = text[i];
                if (c >= '0' && c <= '9') { acc = acc * 10ull + static_cast<uint64_t>(c - '0'); any = true; }
                else if (any) { out.budget_history.push_back(static_cast<uint32_t>(acc)); acc = 0; any = false; }
            }
            if (any) out.budget_history.push_back(static_cast<uint32_t>(acc));
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// The pass
//
// (slot-8 WP-6) THE PARALLEL EXT PASS.  The pass is split into two halves
// that the serial and the parallel path share verbatim:
//
//   DecideCell  -- the placeholder (ell >= 2^32) and the resume decision of
//                  [X] Sec. 1.3.1 R3.3 item 4, which reads the persisted
//                  record and may REFUSE BY NAME (throw);
//   RunCellWork -- one HeckeExtDriver, the record file, the summary.
//
// `ext_jobs == 1` (the default) interleaves them cell by cell in plan order
// on the calling thread -- the pre-WP-6 loop, down to the order its verbose
// lines print and the point at which a refusal throws.  `ext_jobs > 1`
// decides EVERY cell first (a refusal therefore throws before any engine
// runs), then runs the cells that need the engine on `ext_jobs` worker
// threads under the memory-aware LPT admission of ExtCellScheduler, and
// finally assembles the result in PLAN ORDER.  Why no value can move is in
// CampaignExt.h ("THE PARALLEL PASS").
// ---------------------------------------------------------------------------
namespace {

// One cell of the pass, carried from the decision through the run to the
// plan-order assembly.
struct CellWork {
    uint32_t n_sym = 0;
    uint64_t ell = 0;
    std::string owner_id;
    std::vector<uint64_t> family;
    ExtCellKey key;
    ExtCellSummary s;

    enum class Kind { Placeholder, Reuse, Run };
    Kind kind = Kind::Run;
    bool do_resume = false;
    uint32_t resume_from = 0;

    // What the serial loop counted, per cell, so the plan-order assembly
    // reproduces its counters exactly (a cell whose record write threw after
    // run() counts as ran AND failed, as before).
    bool counted_ran = false, counted_resumed = false, counted_failed = false;
    bool force_rerun = false;           // the engine produced a record THIS run

    std::unique_ptr<hecke_ext::HeckeExtRecord> captured;   // out_records only
    std::string log;                                      // the verbose line(s)
    ExtCellCost cost;
};

// The placeholder and the resume decision.  Throws only the refusal.
void DecideCell(const CampaignSpec& spec, const std::string& output_dir,
                const hecke::UEllSetting& u_now, CellWork& w) {
    w.key.n_sym = w.n_sym;
    w.key.ell = w.ell;
    ExtCellSummary& s = w.s;
    s.n_sym = w.n_sym;
    s.ell = w.ell;
    s.family = w.family;
    s.record_file = ExtRecordRelPath(w.n_sym, w.ell, spec.ext.signature_tag());
    s.budget_mb = u_now.effective_budget_mb();

    if (w.ell > 0xffffffffull) {
        // HeckeExtDriver is the uint32_t instantiation ([XA M3]); the cell
        // gets the item-6(c) placeholder with its reason.
        s.available = false;
        s.reason = "ell >= 2^32 (the extension layer is 32-bit)";
        w.kind = CellWork::Kind::Placeholder;
        w.counted_failed = true;
        return;
    }

    const Clock::time_point t0 = Clock::now();
    // --- the resume decision ([X] Sec. 1.3.1 R3.3 item 4) ------------------
    //
    //   persisted INCLUDED, or no `resumable`  -> reuse, a strict no-op
    //   the spec's budget <  the persisted one -> REFUSED BY NAME
    //   the spec's budget == the persisted one -> reuse
    //   the spec's budget >  the persisted one -> RESUME
    //
    // `u_ell = off` forms no prediction at all ([X] Sec. 1.5), so it is
    // never a raise and reuses.
    if (spec.resume) {
        const PersistedUEll p =
            ReadPersistedUEll(output_dir, w.n_sym, w.ell, spec.ext.signature_tag());
        // `ext_groups` is not a mathematical setting and does not
        // fingerprint, so a record built WITHOUT the cone must not be
        // reused by a run that asks for it -- that cell runs fresh.
        const bool attachments_ok = !spec.ext.with_cone() || p.pair_available;
        if (p.file_found && attachments_ok) {
            bool reuse = false;
            if (!p.object_found || p.included || !p.resumable) {
                reuse = true;                       // nothing to continue
            } else if (u_now.mode == hecke::UEllMode::Off) {
                reuse = true;                       // no prediction is formed
            } else {
                const uint32_t reached = p.budget_history.empty()
                    ? p.resumable_budget_mb : p.budget_history.back();
                const uint32_t want = u_now.effective_budget_mb();
                if (want < reached) {
                    // REFUSED BY NAME -- the driver's own message, so the
                    // campaign-level refusal and the in-process one read
                    // the same ([X] R3.1 item 2).
                    throw std::runtime_error(
                        "campaign: resume: u_ell budget below reached state (cell n=" +
                        std::to_string(w.n_sym) + " ell=" + std::to_string(w.ell) +
                        " reached " + std::to_string(reached) + " MB, the spec asks for " +
                        std::to_string(want) + " MB)");
                }
                if (want == reached) reuse = true;
                else { w.do_resume = true; w.resume_from = reached; }
            }
            if (reuse) {
                // The record stands as persisted.  `summary_known` stays
                // false: the engine did not run, nothing was re-derived,
                // and the owner job is skipped too, so its whole row is
                // dashes.
                s.available = true;
                s.reused = true;
                s.seconds = Secs(t0, Clock::now());
                w.kind = CellWork::Kind::Reuse;
                std::ostringstream log;
                log << "  [ext reuse] n=" << w.n_sym << " ell=" << w.ell
                    << " (the persisted record needs no continuation)" << "\n";
                w.log = log.str();
                return;
            }
        }
    }
    w.kind = CellWork::Kind::Run;
}

// One engine run: the driver, the record file, the summary, the verbose line.
// Never throws for an engine failure -- the cell is recorded unavailable with
// the message, exactly as the serial loop always did.
void RunCellWork(const CampaignSpec& spec, const std::string& output_dir,
                 const hecke::UEllSetting& u_now, bool capture, CellWork& w) {
    ExtCellSummary& s = w.s;
    const Clock::time_point t0 = Clock::now();
    hecke_ext::ExtDriverOptions opts;
    opts.good_primes = w.family;
    opts.with_cone = spec.ext.with_cone();
    opts.with_sign_split = spec.ext.with_sign_split();
    // (slot-24 G6) The pass reads only the records: the cone's cochain-level
    // objects need not outlive build_cone_() (memory only; HECKE_FIELD_DIET=0
    // keeps them, as before).
    opts.release_cone_cochains = oc_util::field_diet_on();
    try {
        std::unique_ptr<hecke_ext::HeckeExtDriver> d;
        if (w.do_resume) {
            // Reproduce the persisted state -- Stage 1 under the budget it
            // reached -- and then continue with slot-4 WP-2's own
            // semantics.  The continuation is a prefix cut (HeckeExtDriver.h),
            // so the result is the run the enlarged budget would have
            // produced from the start.
            hecke::UEllSetting u_old;
            u_old.mode = (w.resume_from == hecke::kDefaultUEllBudgetMb)
                             ? hecke::UEllMode::Auto : hecke::UEllMode::BudgetMb;
            u_old.budget_mb = w.resume_from;
            // (slot-8 WP-11b, [KI] Sec. 4d, owner ruling 9 (A)) `Spec`, not
            // the `ScopedUEll` default of `Console` -- the setter here is the
            // campaign spec, never an interactive console session, and this
            // is what stamps the resumed record's `settings.u_ell.source`.
            {
                const hecke::ScopedUEll guard(u_old, hecke::SettingSource::Spec);
                d.reset(new hecke_ext::HeckeExtDriver(static_cast<uint32_t>(w.n_sym),
                                                      static_cast<uint32_t>(w.ell), opts));
                d->run();
            }
            d->resume_u_ell(u_now.effective_budget_mb(), hecke::SettingSource::Spec);
            s.resumed = true;
            s.resumed_from_mb = w.resume_from;
            w.counted_resumed = true;
        } else {
            d.reset(new hecke_ext::HeckeExtDriver(static_cast<uint32_t>(w.n_sym),
                                                  static_cast<uint32_t>(w.ell), opts));
            d->run();
            w.counted_ran = true;
        }
        // The engine produced a record THIS run, so the owner's job file
        // must be rewritten even under `resume` -- otherwise a complete
        // job file from an earlier `ext = false` run (or from a lower
        // budget) would be skipped and the record would never reach its
        // `"extensions"` member.  A cell the pass REUSED sets no flag, so
        // an unchanged campaign still skips everything.
        w.force_rerun = true;
        const hecke_ext::HeckeExtRecord& rec =
            (spec.ext.sign != ExtSign::All && d->has_sign_records())
                ? d->sign_record(spec.ext.sign == ExtSign::Plus)
                : d->record();
        hecke_ext::WriteHeckeExtReport(rec, output_dir + "/" + s.record_file);
        s.available = true;
        s.ran = true;
        FillSummary(rec, s);
        // (slot-6 WP-6) capture the in-memory record for the caller, if
        // asked -- the SAME object just written, not a re-derivation.
        if (capture) w.captured.reset(new hecke_ext::HeckeExtRecord(rec));
    } catch (const std::exception& e) {
        s.available = false;
        s.reason = e.what();
        w.counted_failed = true;
    }
    s.seconds = Secs(t0, Clock::now());
    std::ostringstream log;
    log << "  [ext " << (s.available ? (s.resumed ? "resume" : "ok   ") : "FAIL ")
        << "] n=" << w.n_sym << " ell=" << w.ell << " ideals=" << s.ideals
        << " final=" << (s.final_verdict.empty() ? "-" : s.final_verdict)
        << "  " << s.seconds << " s";
    if (!s.available) log << "\n        " << s.reason;
    log << "\n";
    w.log = log.str();
}

// Folds one finished cell into the pass, in the order the caller calls it
// (always plan order).
void AssembleCell(ExtPassResult& pass, CellWork& w,
                  std::map<ExtCellKey, hecke_ext::HeckeExtRecord>* out_records) {
    if (w.counted_ran) ++pass.num_ran;
    if (w.counted_resumed) ++pass.num_resumed;
    if (w.counted_failed) ++pass.num_failed;
    if (w.kind == CellWork::Kind::Reuse) ++pass.num_reused;
    if (w.force_rerun) pass.force_rerun_job_ids.push_back(w.owner_id);
    if (out_records != nullptr && w.captured) (*out_records)[w.key] = *w.captured;
    pass.cells.emplace(w.key, w.s);
}

// The scheduling class of a worker thread is ExtWorkerClass (CampaignExt.h):
// Cpu, and (slot-8 WP-6b) Gpu, the device worker, which installs a
// thread-local device override and takes cells by its own rule in
// ExtCellScheduler::take (CampaignExt.h, "THE DEVICE WORKER").

} // namespace

// ---------------------------------------------------------------------------
// The cost model and the admission rule (public for the validation).
// ---------------------------------------------------------------------------
ExtCellCost PredictExtCellCost(uint32_t n_sym, uint64_t ell, const hecke::UEllSetting& u_ell) {
    ExtCellCost c;
    c.n_sym = n_sym;
    c.ell = ell;
    if (ell <= 0xffffffffull) {
        const hecke_ext::UEllVerdict v = hecke_ext::UEllDecision(n_sym, ell, u_ell);
        c.u_ell_admitted = v.included;
        c.u_ell_predicted_mb = v.included ? v.predicted_mb : 0.0;
    }
    c.memory_mb = static_cast<double>(kExtCellFloorMb) + c.u_ell_predicted_mb;
    return c;
}

std::vector<size_t> ExtLptOrder(const std::vector<ExtCellCost>& cells) {
    std::vector<size_t> order(cells.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    // Longest processing time first.  The U_ell term dominates every cell
    // that carries it (W3: the ten U_10007 cells were 98.8 % of the pass),
    // and its RSS model is linear in the coset count (ell + 1) -- the very
    // quantity that drives its host-serial cost -- so the predicted MB is the
    // cost key; the cells without U_ell follow, largest n first; the plan
    // index breaks every remaining tie, so the order is a pure function of
    // the plan.
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        const ExtCellCost& x = cells[a];
        const ExtCellCost& y = cells[b];
        if (x.u_ell_predicted_mb != y.u_ell_predicted_mb)
            return x.u_ell_predicted_mb > y.u_ell_predicted_mb;
        if (x.n_sym != y.n_sym) return x.n_sym > y.n_sym;
        return a < b;
    });
    return order;
}

ExtCellScheduler::ExtCellScheduler(const std::vector<ExtCellCost>& costs, double cap_mb)
    : costs_(costs), cap_mb_(cap_mb) {
    const std::vector<size_t> order = ExtLptOrder(costs_);
    pending_.assign(order.begin(), order.end());
}

void ExtCellScheduler::enable_device_worker(uint32_t cpu_slots, const std::vector<char>& gpu_listed,
                                            double device_cap_mb) {
    std::lock_guard<std::mutex> lk(mu_);
    ENGINE_CHECK_MSG(gpu_listed.empty() || gpu_listed.size() == costs_.size(),
                     "ExtCellScheduler: gpu_listed must name every cell or none");
    device_ = true;
    cpu_slots_ = cpu_slots;
    gpu_listed_ = gpu_listed;
    device_cap_mb_ = device_cap_mb;
}

bool ExtCellScheduler::admit_locked(size_t i) {
    const double m = costs_[i].memory_mb;
    if (!(running_ == 0 || running_mb_ + m <= cap_mb_)) return false;
    running_mb_ += m;
    ++running_;
    if (running_mb_ > peak_admitted_mb_) peak_admitted_mb_ = running_mb_;
    if (running_ > peak_running_) peak_running_ = running_;
    if (running_ == 1 && m > cap_mb_) ++over_cap_alone_;
    return true;
}

size_t ExtCellScheduler::take(ExtWorkerClass cls) {
    std::unique_lock<std::mutex> lk(mu_);
    if (on_gpu_.size() != costs_.size()) on_gpu_.assign(costs_.size(), 0);
    const bool listed = device_ && !gpu_listed_.empty();
    for (;;) {
        if (cls == ExtWorkerClass::Cpu) {
            // The first pending cell (in LPT order) that fits beside the
            // running ones; a cell larger than the whole cap runs when nothing
            // else does, so the pass can never stall.  (WP-6b) A cell an
            // explicit list gives the device worker is never a CPU cell.
            bool any = false;
            for (auto it = pending_.begin(); it != pending_.end(); ++it) {
                const size_t i = *it;
                if (listed && gpu_listed_[i]) continue;
                any = true;
                if (admit_locked(i)) {
                    pending_.erase(it);
                    ++cpu_running_;
                    lk.unlock();
                    cv_.notify_all();   // the device worker waits for busy CPU slots
                    return i;
                }
            }
            if (!any) return kNone;
        } else {
            ENGINE_CHECK_MSG(device_, "ExtCellScheduler::take(Gpu) without enable_device_worker()");
            if (listed) {
                // Exactly the listed cells, in LPT order among them.
                bool any = false;
                for (auto it = pending_.begin(); it != pending_.end(); ++it) {
                    const size_t i = *it;
                    if (!gpu_listed_[i]) continue;
                    any = true;
                    if (admit_locked(i)) {
                        pending_.erase(it);
                        on_gpu_[i] = 1;
                        return i;
                    }
                    break;   // LPT order among the listed cells: wait for this one
                }
                if (!any) return kNone;
            } else {
                // The cheapest pending cell: the LAST of the LPT order, which
                // puts every cell without U_ell after every cell with it
                // (ExtLptOrder) -- so the device starts from the cheapest
                // non-U_ell cells, and reaches a U_ell cell (the cheapest
                // first) only when none is left.
                if (pending_.empty()) return kNone;
                auto pick = std::prev(pending_.end());
                double other_mb = 0.0;
                for (auto it = pending_.begin(); it != pending_.end(); ++it) other_mb += costs_[*it].memory_mb;
                const size_t i = *pick;
                const double c = costs_[i].memory_mb;
                other_mb -= c;
                // Only while every CPU slot is busy; only while the modelled
                // GPU cost does not exceed the rest of the pending work per
                // CPU slot (so, by the model, the device never becomes the
                // pass's tail); only within the device-memory cap.
                const bool cpu_busy = cpu_running_ >= cpu_slots_;
                const bool worth = kExtGpuCostFactor * c <= other_mb / std::max<uint32_t>(1u, cpu_slots_);
                const bool fits_device = device_cap_mb_ <= 0.0 || c <= device_cap_mb_;
                if (!fits_device) return kNone;   // the only candidate class can never fit
                if (!worth) return kNone;         // pending work only shrinks from here
                if (cpu_busy && admit_locked(i)) {
                    pending_.erase(pick);
                    on_gpu_[i] = 1;
                    return i;
                }
            }
        }
        if (pending_.empty()) return kNone;
        cv_.wait(lk);
    }
}

void ExtCellScheduler::done(size_t i) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        running_mb_ -= costs_[i].memory_mb;
        --running_;
        if (running_ == 0) running_mb_ = 0.0;     // no floating-point residue
        if (on_gpu_.size() == costs_.size() && on_gpu_[i]) {
            // the device worker's cell: no CPU slot to free
        } else if (cpu_running_ > 0) {
            --cpu_running_;
        }
    }
    cv_.notify_all();
}

double ExtCellScheduler::peak_admitted_mb() const {
    std::lock_guard<std::mutex> lk(mu_);
    return peak_admitted_mb_;
}
uint32_t ExtCellScheduler::peak_running() const {
    std::lock_guard<std::mutex> lk(mu_);
    return peak_running_;
}
uint32_t ExtCellScheduler::over_cap_alone() const {
    std::lock_guard<std::mutex> lk(mu_);
    return over_cap_alone_;
}

ExtPassResult RunExtPass(const CampaignSpec& spec, const std::vector<Job>& jobs,
                         const std::string& output_dir, bool verbose,
                         std::map<ExtCellKey, hecke_ext::HeckeExtRecord>* out_records) {
    ExtPassResult pass;
    if (!spec.ext.ext) return pass;
    pass.enabled = true;

    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(output_dir + "/ext", ec);

    const Clock::time_point t_pass = Clock::now();

    // The cells, in the plan's EXECUTION order, with their owner and family.
    std::vector<CellWork> cells;
    for (const Job& j : jobs) {
        if (!j.ext_owner) continue;
        CellWork w;
        w.n_sym = j.n_sym;
        w.ell = j.ell;
        w.owner_id = j.id();
        w.family = j.ext_family;
        cells.push_back(std::move(w));
    }

    // The budget the spec asks for ([OD] R1: read from the settings in force,
    // which the caller has installed, never re-derived from the spec text).
    const hecke::UEllSetting u_now = hecke::math_settings().u_ell;
    const bool capture = (out_records != nullptr);

    pass.jobs = std::max<uint32_t>(1u, spec.ext_jobs);
    pass.memory_cap_mb = spec.ext_memory_mb;

    // (slot-8 WP-6b) The device worker: `on` needs a usable device (else one
    // line, and WP-6's CPU-only pass); `force` runs it regardless.  Probed
    // only under a non-default ext_gpu, so an `off` pass never creates a CUDA
    // context.
    pass.gpu_mode = spec.ext_gpu;
    bool device_worker = false;
    if (spec.ext_gpu != ExtGpuMode::Off) {
        const bool usable = engine::cuda::available();
        if (usable) {
            try {
                const engine::cuda::DeviceInfo di = engine::cuda::device_info();
                pass.gpu_device_cap_mb =
                    kExtGpuDeviceCapFraction * static_cast<double>(di.free_mem) / (1024.0 * 1024.0);
            } catch (const std::exception& e) {
                pass.gpu_fallback = std::string("device_info: ") + e.what();
            }
        } else {
            pass.gpu_fallback = engine::cuda::unavailable_reason();
            if (pass.gpu_fallback.empty()) pass.gpu_fallback = "no usable CUDA device";
        }
        if (spec.ext_gpu == ExtGpuMode::Force) {
            device_worker = true;
            if (!pass.gpu_fallback.empty())
                std::cout << "ext_gpu = force: " << pass.gpu_fallback
                          << "; the device worker runs, its GPU nodes on the CPU\n" << std::flush;
            pass.gpu_fallback.clear();
        } else if (pass.gpu_fallback.empty()) {
            device_worker = true;
        } else {
            std::cout << "ext_gpu = on: " << pass.gpu_fallback
                      << "; running the CPU-only pass\n" << std::flush;
        }
    }
    pass.gpu_worker = device_worker;

    if (pass.jobs == 1 && !device_worker) {
        // ---- the serial pass: the pre-WP-6 loop, cell by cell -------------
        for (CellWork& w : cells) {
            DecideCell(spec, output_dir, u_now, w);        // may refuse (throw)
            if (w.kind == CellWork::Kind::Run) {
                RunCellWork(spec, output_dir, u_now, capture, w);
                pass.num_cells_run += 1;
            }
            AssembleCell(pass, w, out_records);
            if (verbose && !w.log.empty()) std::cout << w.log << std::flush;
            // (slot-8 WP-5, [KI] Sec. 4c) the between-cells release: the
            // engine ran for this cell (a reused or placeholder cell did
            // not), and its driver is gone, so once the caches pass the
            // pinned threshold the programs are dropped here.  No value can
            // move -- a cleared program is recompiled on its next miss, and
            // compile() is a pure function of (program, options) -- and the
            // pass's RSS stops growing with the number of cells.
            if (w.kind == CellWork::Kind::Run) ReleaseEngineProgramCaches(kCellLoopReleaseThreshold);
            w.captured.reset();
        }
        pass.peak_rss_mb = oc_util::PeakRssMiB();
        pass.wall_s = Secs(t_pass, Clock::now());
        return pass;
    }

    // ---- the parallel pass ------------------------------------------------
    // 1. Every decision first, in plan order: a refusal throws before any
    //    engine runs.
    for (CellWork& w : cells) DecideCell(spec, output_dir, u_now, w);

    // 2. The cells that need the engine, their costs, the scheduler.
    std::vector<size_t> run_index;                   // into `cells`
    std::vector<ExtCellCost> costs;
    for (size_t i = 0; i < cells.size(); ++i) {
        if (cells[i].kind != CellWork::Kind::Run) continue;
        // A resumed cell ends under the NEW budget, so that is its cost.
        cells[i].cost = PredictExtCellCost(cells[i].n_sym, cells[i].ell, u_now);
        run_index.push_back(i);
        costs.push_back(cells[i].cost);
    }
    pass.num_cells_run = static_cast<uint32_t>(run_index.size());

    engine::tape::DeviceTally gpu_tally, cpu_tally;   // (WP-6b)
    std::vector<char> ran_on_gpu(run_index.size(), 0);
    if (!run_index.empty()) {
        ExtCellScheduler sched(costs, static_cast<double>(spec.ext_memory_mb));
        const size_t conc = std::min<size_t>(pass.jobs, run_index.size());
        if (device_worker) {
            // An explicit list marks the engine cells it names; a named cell
            // that is not one of them is ignored.
            std::vector<char> listed;
            if (!spec.ext_gpu_cells.empty()) {
                listed.assign(run_index.size(), 0);
                for (size_t k = 0; k < run_index.size(); ++k) {
                    const CellWork& w = cells[run_index[k]];
                    for (const auto& nc : spec.ext_gpu_cells)
                        if (nc.first == w.n_sym && nc.second == w.ell) listed[k] = 1;
                }
            }
            sched.enable_device_worker(static_cast<uint32_t>(conc), listed, pass.gpu_device_cap_mb);
        }
        // Captured HERE, on the launching thread: the mathematical settings
        // ([S8] Sec. 2.3 item 10 -- a worker started any other way would
        // silently run at the DEFAULTS), memory-lean, the legacy arms, and
        // every other registered thread-local consumer option.
        const std::vector<engine::tape::ThreadContextInstaller> ctx =
            engine::tape::capture_thread_context();
        const bool own_pools = (spec.threads > 0) && PerJobPoolsSupported();
        pass.per_worker_pools = own_pools;

        std::mutex err_mu;
        std::exception_ptr first_error;

        auto worker = [&](ExtWorkerClass cls) {
            std::vector<engine::tape::ThreadContextGuard> guards;
            std::unique_ptr<engine::tape::ScopedThreadDevice> device_guard;
            std::unique_ptr<engine::tape::ThreadPool> pool;
            std::unique_ptr<engine::tape::ScopedConsumerPool> pool_guard;
            try {
                guards.reserve(ctx.size());
                for (const engine::tape::ThreadContextInstaller& install : ctx)
                    guards.push_back(install());
                // (slot-8 WP-6b) After the captured context, whose device
                // override (none, in production) it replaces: the device
                // worker compiles for the GPU, and -- only when a device
                // worker exists -- every CPU worker explicitly for the CPU,
                // whatever ENGINE_DEVICE says.  `ext_gpu = off` installs
                // nothing, so its workers are WP-6's.
                if (cls == ExtWorkerClass::Gpu)
                    device_guard.reset(new engine::tape::ScopedThreadDevice(
                        engine::tape::ThreadDevice::Gpu, &gpu_tally));
                else if (device_worker)
                    device_guard.reset(new engine::tape::ScopedThreadDevice(
                        engine::tape::ThreadDevice::Cpu, &cpu_tally));
                // This worker's own engine pool (ThreadPool::parallel_for()
                // serialises concurrent callers, so sharing one pool would
                // serialise the engine work of every cell) -- after the
                // context, whose pool override is the launching thread's.
                if (own_pools) {
                    pool.reset(new engine::tape::ThreadPool(spec.threads));
                    pool_guard.reset(new engine::tape::ScopedConsumerPool(pool.get()));
                }
            } catch (...) {
                std::lock_guard<std::mutex> lk(err_mu);
                if (!first_error) first_error = std::current_exception();
                return;
            }
            for (;;) {
                const size_t k = sched.take(cls);
                if (k == ExtCellScheduler::kNone) break;
                if (cls == ExtWorkerClass::Gpu) ran_on_gpu[k] = 1;   // one writer per k
                try {
                    RunCellWork(spec, output_dir, u_now, capture, cells[run_index[k]]);
                } catch (...) {
                    // RunCellWork records engine failures itself; anything
                    // reaching here is an infrastructure error (allocation):
                    // kept, and rethrown after every worker joined.
                    std::lock_guard<std::mutex> lk(err_mu);
                    if (!first_error) first_error = std::current_exception();
                }
                sched.done(k);
                // (slot-8 WP-5) the between-cells release, from the worker
                // that finished the cell: memory-safe while other workers
                // run (they keep their shared_ptr to a replayed program and
                // recompile on the next miss), value-free, and rare at the
                // pinned threshold (ProgramCacheRelease.h).
                ReleaseEngineProgramCaches(kCellLoopReleaseThreshold);
            }
            pool_guard.reset();
            pool.reset();
            device_guard.reset();
            guards.clear();
        };

        std::vector<std::thread> threads;
        threads.reserve(conc);
        if (device_worker) threads.emplace_back(worker, ExtWorkerClass::Gpu);   // (WP-6b)
        for (size_t t = 1; t < conc; ++t) threads.emplace_back(worker, ExtWorkerClass::Cpu);
        worker(ExtWorkerClass::Cpu);                 // the calling thread joins in
        for (std::thread& t : threads) t.join();
        pass.peak_admitted_mb = sched.peak_admitted_mb();
        pass.peak_running = sched.peak_running();
        pass.over_cap_alone = sched.over_cap_alone();
        if (first_error) std::rethrow_exception(first_error);
    }

    // (WP-6b) the device worker's record, in plan order.
    if (device_worker) {
        std::vector<char> gpu_cell(cells.size(), 0);
        for (size_t k = 0; k < run_index.size(); ++k)
            if (ran_on_gpu[k]) gpu_cell[run_index[k]] = 1;
        for (size_t i = 0; i < cells.size(); ++i)
            if (gpu_cell[i]) pass.gpu_cells.push_back(cells[i].key);
        pass.gpu_programs = gpu_tally.programs.load();
        pass.gpu_program_gpu_nodes = gpu_tally.gpu_nodes.load();
        pass.cpu_programs = cpu_tally.programs.load();
        pass.cpu_program_gpu_nodes = cpu_tally.gpu_nodes.load();
    }

    // 3. Plan order: the map, the counters, the force-rerun ids, the lines.
    for (CellWork& w : cells) {
        AssembleCell(pass, w, out_records);
        if (verbose && !w.log.empty()) std::cout << w.log;
        w.captured.reset();
    }
    if (verbose && device_worker) {
        // (WP-6b) the assignment log: which cells the device worker ran.
        std::cout << "EXT_GPU_CELLS count=" << pass.gpu_cells.size();
        for (const ExtCellKey& k : pass.gpu_cells) std::cout << " " << k.n_sym << ":" << k.ell;
        std::cout << "\n";
    }
    if (verbose) std::cout << std::flush;
    pass.peak_rss_mb = oc_util::PeakRssMiB();
    pass.wall_s = Secs(t_pass, Clock::now());
    return pass;
}

std::string ExtPassLine(const ExtPassResult& pass) {
    std::ostringstream os;
    os << "EXT_PASS wall_s=" << std::fixed << std::setprecision(3) << pass.wall_s
       << " jobs=" << pass.jobs << " cells=" << pass.cells.size()
       << " engine_cells=" << pass.num_cells_run << " ran=" << pass.num_ran
       << " reused=" << pass.num_reused << " resumed=" << pass.num_resumed
       << " failed=" << pass.num_failed << std::setprecision(1)
       << " peak_rss_mb=" << pass.peak_rss_mb << " memory_cap_mb=" << pass.memory_cap_mb
       << " peak_admitted_mb=" << pass.peak_admitted_mb
       << " peak_running=" << pass.peak_running
       << " per_worker_pools=" << (pass.per_worker_pools ? 1 : 0);
    if (pass.gpu_mode != ExtGpuMode::Off) {
        // (slot-8 WP-6b) appended only under a non-default ext_gpu.
        os << " ext_gpu=" << (pass.gpu_mode == ExtGpuMode::Force ? "force" : "on")
           << " gpu_worker=" << (pass.gpu_worker ? 1 : 0)
           << " gpu_cells=" << pass.gpu_cells.size()
           << " gpu_programs=" << pass.gpu_programs
           << " gpu_nodes=" << pass.gpu_program_gpu_nodes
           << " cpu_programs=" << pass.cpu_programs
           << " cpu_gpu_nodes=" << pass.cpu_program_gpu_nodes
           << " device_cap_mb=" << pass.gpu_device_cap_mb;
        if (!pass.gpu_fallback.empty()) os << " gpu_fallback=\"" << pass.gpu_fallback << "\"";
    }
    return os.str();
}

void StreamJobExtensions(std::ostream& os, const std::string& output_dir,
                         const ExtCellSummary& cell) {
    os << "\"extensions\":";
    std::string body;
    const bool ok = cell.available && ReadFile(output_dir + "/" + cell.record_file, body);
    if (cell.available && ok && !body.empty()) {
        // The record verbatim -- one serialisation, two appearances.
        os << body;
    } else {
        // [C] `sec:recordschema` item 6(c): the placeholder, and NOTHING else
        // but the reason that made it one.
        os << "{\"status\":\"unavailable\",\"reason\":\"";
        for (char c : (cell.available ? std::string("the record file could not be read")
                                      : cell.reason)) {
            if (c == '"' || c == '\\') os << '\\' << c;
            else if (c == '\n') os << "\\n";
            else os << c;
        }
        os << "\"}";
    }
    os << ",\"extensions_file\":\"" << cell.record_file << "\",";
}

std::string ExtSummaryHeader() {
    return "\tideals\tmax_embdim\tmax_loewy\tnonsplit_ideals\tnonlocal_ideals"
           "\tsturm_B_prime\text_certificate\text_final";
}

std::string ExtSummaryRow(const ExtCellSummary* cell) {
    // A job that does not own its cell's record, a cell with no record, and a
    // cell whose record `resume` REUSED (the engine did not run, so nothing
    // was re-derived -- and that job is skipped, so every other column of its
    // row is `-` too) all print the eight dashes.
    if (cell == nullptr || !cell->summary_known)
        return "\t-\t-\t-\t-\t-\t-\t-\t-";
    std::ostringstream os;
    os << "\t" << cell->ideals << "\t" << cell->max_embdim << "\t" << cell->max_loewy
       << "\t" << cell->nonsplit_ideals << "\t" << cell->nonlocal_ideals << "\t"
       << (cell->has_sturm_B_prime ? std::to_string(cell->sturm_B_prime) : std::string("-"))
       << "\t" << (cell->certificate.empty() ? std::string("-") : cell->certificate)
       << "\t" << (cell->final_verdict.empty() ? std::string("-") : cell->final_verdict);
    return os.str();
}

std::string ExtIndexJson(const CampaignSpec& spec, const ExtPassResult& pass) {
    std::ostringstream os;
    os << "{\"ext\":true,\"family\":\"" << spec.ext.family_text << "\",\"names_ell\":"
       << (spec.ext.names_ell ? "true" : "false")
       << ",\"ext_groups\":\"" << spec.ext.groups_text << "\",\"sign\":\"" << spec.ext.sign_text
       << "\",\"with_cone\":" << (spec.ext.with_cone() ? "true" : "false")
       << ",\"with_sign_split\":" << (spec.ext.with_sign_split() ? "true" : "false")
       << ",\"cells\":" << pass.cells.size() << ",\"ran\":" << pass.num_ran
       << ",\"reused\":" << pass.num_reused << ",\"resumed\":" << pass.num_resumed
       << ",\"failed\":" << pass.num_failed << ",\"wall_s\":" << pass.wall_s
       << ",\"records\":[";
    bool first = true;
    for (const auto& kv : pass.cells) {
        if (!first) os << ",";
        first = false;
        const ExtCellSummary& c = kv.second;
        os << "{\"n_sym\":" << c.n_sym << ",\"ell\":" << c.ell
           << ",\"available\":" << (c.available ? "true" : "false")
           << ",\"file\":\"" << c.record_file << "\""
           << ",\"ran\":" << (c.ran ? "true" : "false")
           << ",\"reused\":" << (c.reused ? "true" : "false")
           << ",\"resumed\":" << (c.resumed ? "true" : "false")
           << ",\"resumed_from_mb\":" << c.resumed_from_mb
           << ",\"ideals\":" << c.ideals << ",\"final\":\"" << c.final_verdict << "\""
           << ",\"resumable_items\":" << c.resumable_items
           << ",\"verify\":" << c.verify_code << "}";
    }
    os << "]}";
    return os.str();
}

} // namespace campaign
