// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// CostModel.cpp -- see CostModel.h.  Every function names the Python function
// it ports; the order of every floating-point operation follows the Python
// text, so ConsoleCostModelValidation can hold the two to 1e-9 relative.
// ===========================================================================

#include "CostModel.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>

namespace console {
namespace cost {

// The generated translation unit (cmake/sources/Special.cmake, from cost_models.tsv at configure time).
extern const char* const kEmbeddedCostModelsTsv;

namespace {

std::vector<std::string> SplitTab(const std::string& line) {
    std::vector<std::string> f;
    std::size_t s = 0;
    for (;;) {
        const std::size_t t = line.find('\t', s);
        if (t == std::string::npos) {
            f.push_back(line.substr(s));
            break;
        }
        f.push_back(line.substr(s, t - s));
        s = t + 1;
    }
    return f;
}

std::vector<std::string> Split(const std::string& s, char c) {
    std::vector<std::string> out;
    std::size_t a = 0;
    for (;;) {
        const std::size_t b = s.find(c, a);
        if (b == std::string::npos) {
            out.push_back(s.substr(a));
            break;
        }
        out.push_back(s.substr(a, b - a));
        a = b + 1;
    }
    return out;
}

double Num(const std::string& s) {
    if (s == "nan") return std::nan("");
    char* end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    if (end == s.c_str() || *end != '\0') throw std::runtime_error("cost_models: not a number: '" + s + "'");
    return v;
}

bool Starts(const std::string& s, const char* p) { return s.compare(0, std::char_traits<char>::length(p), p) == 0; }

// q = f[1].split("_")[-1].split("/")[0]
std::string QuantityOf(const std::string& mid) {
    const std::vector<std::string> u = Split(mid, '_');
    return Split(u.back(), '/').front();
}

std::string Key4(const std::string& verb, const std::string& var, const std::string& reg, const std::string& q) {
    return verb + "|" + var + "|" + reg + "|" + q;
}

bool ParseInt(const std::string& s, long long& v) {
    if (s.empty()) return false;
    char* end = nullptr;
    v = std::strtoll(s.c_str(), &end, 10);
    return end != s.c_str() && *end == '\0';
}

bool PrimeList(const std::string& s, std::vector<std::uint64_t>& out) {
    out.clear();
    for (const std::string& t : Split(s, ':')) {
        long long v = 0;
        if (!ParseInt(t, v) || v < 0) return false;
        out.push_back(static_cast<std::uint64_t>(v));
    }
    return true;
}

double Ev(const std::vector<Monomial>& terms, const std::map<std::string, double>& f) {
    // sum(a * math.prod(f[k] ** e for k, e in m.items()) for a, m in terms)
    double s = 0.0;
    for (const Monomial& t : terms) {
        double p = 1.0;
        for (const auto& ke : t.exps) p *= std::pow(f.at(ke.first), ke.second);
        s += t.coef * p;
    }
    return s;
}

std::string Fmt(const char* fmt, double v) {
    char b[64];
    std::snprintf(b, sizeof b, fmt, v);
    return b;
}

} // namespace

GroupKey MakeKey(const std::string& verb, const std::string& var, const std::string& reg, const std::string& q,
                 const std::string& slot) {
    return Key4(verb, var, reg, q) + "|" + slot;
}

// --- load_models -----------------------------------------------------------
CostModels ParseCostModels(const std::string& text) {
    CostModels M;
    std::map<std::string, std::set<std::string>> slots;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        bool blank = true;
        for (char c : line) blank = blank && (c == ' ' || c == '\t');
        if (blank) continue;
        const std::vector<std::string> f = SplitTab(line);
        auto need = [&](std::size_t n) {
            if (f.size() < n) throw std::runtime_error("cost_models: short '" + f[0] + "' row: " + line);
        };
        const std::string q = f.size() > 1 ? QuantityOf(f[1]) : std::string();
        if (f[0] == "term") {
            need(9);
            Monomial m;
            m.coef = Num(f[7]);
            if (f[8] != "1") {
                for (const std::string& t : Split(f[8], '*')) {
                    const std::vector<std::string> ke = Split(t, '^');
                    if (ke.size() != 2) throw std::runtime_error("cost_models: bad monomial '" + f[8] + "'");
                    m.exps.emplace_back(ke[0], Num(ke[1]));
                }
            }
            M.term[MakeKey(f[2], f[3], f[4], q, f[5])].push_back(m);
            slots[Key4(f[2], f[3], f[4], q)].insert(f[5]);
        } else if (f[0] == "grid") {
            need(9);
            auto& g = M.grid[MakeKey(f[2], f[3], f[4], q, f[5])];
            bool found = false;
            for (auto& kv : g)
                if (kv.first == f[6]) {
                    kv.second = {Num(f[7]), Num(f[8])};
                    found = true;
                }
            if (!found) g.push_back({f[6], {Num(f[7]), Num(f[8])}});
        } else if (f[0] == "gmax") {
            need(7);
            M.gmax[MakeKey(f[2], f[3], f[4], q, f[5])] = Num(f[6]);
        } else if (f[0] == "lowconf") {
            need(7);
            M.lowconf[MakeKey(f[2], f[3], f[4], q, f[5])].push_back(f[6]);
        } else if (f[0] == "safety") {
            need(5);
            M.safety[f[1]] = Safety{Num(f[2]), Num(f[3]), Num(f[4])};
        } else if (f[0] == "fitted_on") {
            M.fitted_on.assign(f.begin() + 1, f.end());
        } else if (f[0] == "class") {
            need(3);
            M.klass[f[1]] = f[2];
        } else if (f[0] == "resid") {
            need(10);
            M.resid[MakeKey(f[2], f[3], f[4], q, f[5])] = f[9] != "nan" ? Num(f[9]) : Num(f[8]);
        } else if (f[0] == "calib") {
            // calib  <set>  <class>  <slot>  <threads>  <seconds>  <digest|->  [<rss_mib>]   (calib/v2 adds the RSS column)
            need(7);
            CostModels::Calib c;
            c.threads = static_cast<unsigned>(Num(f[4]));
            c.seconds = Num(f[5]);
            c.digest = f[6];
            c.rss_mib = f.size() > 7 ? Num(f[7]) : -1.0;
            M.calib[{f[2], f[3]}] = c;
            M.calib_set = f[1];
        }
        // `thread` rows document the rule; the estimator implements it (Sec. 3.3).
    }
    for (const auto& kv : slots) M.slots[kv.first] = std::vector<std::string>(kv.second.begin(), kv.second.end());
    return M;
}

const std::string& EmbeddedCostModelsText() {
    static const std::string text(kEmbeddedCostModelsTsv);
    return text;
}

const CostModels& EmbeddedCostModels() {
    static const CostModels M = ParseCostModels(EmbeddedCostModelsText());
    return M;
}

// --- nbconfirm_fit.py: psi, default_record_primes, feats -------------------
std::uint64_t PsiIndex(std::uint64_t N) {
    std::uint64_t r = N, m = N, p = 2;
    while (p * p <= m) {
        if (m % p == 0) {
            r = r / p * (p + 1);
            while (m % p == 0) m /= p;
        }
        ++p;
    }
    if (m > 1) r = r / m * (m + 1);
    return r;
}

std::vector<std::uint64_t> DefaultRecordPrimesOf(std::uint64_t N) {
    std::vector<std::uint64_t> out;
    for (std::uint64_t p : {2u, 3u, 5u, 7u, 11u, 13u}) {
        if (N % p) {
            out.push_back(p);
            if (out.size() == 3) break;
        }
    }
    return out;
}

bool CostFeatures(const std::string& v, const std::vector<std::string>& a, Features& out, std::string& error) {
    out = Features();
    auto I = [&](std::size_t i, long long& x) {
        if (i >= a.size() || !ParseInt(a[i], x)) {
            error = v + ": argument " + std::to_string(i) + " is not an integer";
            return false;
        }
        return true;
    };
    auto dflt235 = [](long long ell) {
        std::vector<std::uint64_t> P;
        for (std::uint64_t p : {2u, 3u, 5u})
            if (static_cast<long long>(p) != ell) P.push_back(p);
        return P;
    };
    std::vector<std::uint64_t> P;
    auto& f = out.f;
    if (v == "descent" || v == "hecke_structure") {
        long long N = 0, n = 0;
        if (!I(0, N) || !I(1, n)) return false;
        const bool given = a.size() > 2 && a[2] != "full" && a[2] != "anemic";
        if (given) {
            if (!PrimeList(a[2], P)) {
                error = v + ": bad prime list '" + a[2] + "'";
                return false;
            }
        } else {
            P = DefaultRecordPrimesOf(static_cast<std::uint64_t>(N));
        }
        const std::string& last = a.back();
        out.variant = (last == "full" || last == "anemic") ? last : (v == "hecke_structure" ? "full" : "-");
        std::uint64_t p2 = 0;
        for (std::uint64_t p : P) p2 += p * p;
        const double psi = static_cast<double>(PsiIndex(static_cast<std::uint64_t>(N)));
        f["N"] = static_cast<double>(N);
        f["n1"] = static_cast<double>(n + 1);
        f["psi"] = psi;
        f["D"] = static_cast<double>(PsiIndex(static_cast<std::uint64_t>(N)) * static_cast<std::uint64_t>(n + 1));
        f["p2"] = static_cast<double>(p2);
        f["pr"] = psi * std::sqrt(static_cast<double>(p2));
        out.regime = N == 1 ? "N1" : (n == 0 ? "n0" : "gen");
        return true;
    }
    if (v == "level_n" || v == "nebentypus") {
        const bool neb = v == "nebentypus";
        long long N = 0, m = 0, n = 0, ell = 0;
        if (!I(0, N)) return false;
        if (neb && !I(1, m)) return false;
        if (!I(neb ? 2 : 1, n) || !I(neb ? 3 : 2, ell)) return false;
        const std::size_t pi = neb ? 4 : 3;
        if (a.size() > pi) {
            if (!PrimeList(a[pi], P)) {
                error = v + ": bad prime list '" + a[pi] + "'";
                return false;
            }
        } else {
            P = dflt235(ell);
        }
        const std::uint64_t psi = PsiIndex(static_cast<std::uint64_t>(N));
        f["n1"] = static_cast<double>(n + 1);
        f["psi"] = static_cast<double>(psi);
        f["D"] = static_cast<double>(psi * static_cast<std::uint64_t>(n + 1));
        f["np"] = static_cast<double>(P.size());
        if (neb) {
            std::uint64_t sp = 0;
            for (std::uint64_t p : P) sp += p;
            f["sp"] = static_cast<double>(sp);
        } else {
            f["ell"] = static_cast<double>(ell);
        }
        out.variant = "-";
        out.regime = "all";
        return true;
    }
    if (v == "forms_algebra" || v == "hida_lambda") {
        const bool fa = v == "forms_algebra";
        long long k = 0, ell = 0, W = 0;
        if (fa) {
            if (!I(0, k) || !I(1, ell)) return false;
        } else {
            if (!I(0, ell) || !I(1, k) || !I(2, W)) return false;
        }
        const std::size_t pi = fa ? 2 : 3;
        if (a.size() > pi && !(fa && a[pi] == "-")) {
            if (!PrimeList(a[pi], P) || P.empty()) {
                error = v + ": bad prime list '" + a[pi] + "'";
                return false;
            }
        } else {
            P = dflt235(ell);
        }
        const double pm = static_cast<double>(*std::max_element(P.begin(), P.end()));
        f["k"] = static_cast<double>(k);
        f["pm"] = pm;
        f["ell"] = static_cast<double>(ell);
        if (!fa) f["W"] = static_cast<double>(W);
        out.variant = "-";
        out.regime = "all";
        return true;
    }
    if (v == "hida_tower") {
        long long n = 0, ell = 0, m = 0;
        if (!I(0, n) || !I(1, ell) || !I(2, m)) return false;
        f["n1"] = static_cast<double>(n + 1);
        f["m"] = static_cast<double>(m);
        f["ell"] = static_cast<double>(ell);
        out.variant = "-";
        out.regime = "all";
        return true;
    }
    error = "no cost model for the verb '" + v + "'";
    return false;
}

// --- the device profile ----------------------------------------------------
std::string DeviceProfilePath() {
    const char* p = std::getenv("HACS_DEVICE_PROFILE");
    if (p != nullptr && *p != '\0') return p;
    const char* x = std::getenv("XDG_CONFIG_HOME");
    std::string base;
    if (x != nullptr && *x != '\0') {
        base = x;
    } else {
        const char* h = std::getenv("HOME");
        base = std::string(h != nullptr ? h : "") + "/.config";
    }
    return base + "/hacs/device_profile.tsv";
}

DeviceProfile LoadDeviceProfile(const std::string& path) {
    DeviceProfile P;
    P.path = path;
    if (path.empty() || path == "none") return P;
    std::ifstream in(path.c_str());
    if (!in) return P;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        const std::vector<std::string> f = SplitTab(line);
        if (f[0] == "factor" && f.size() >= 5) {
            char* end = nullptr;
            const double v = std::strtod(f[4].c_str(), &end);
            if (end != f[4].c_str()) P.factor[{f[1], f[2]}] = v;
        } else if (f.size() == 2) {
            P.kv[f[0]] = f[1];
        }
    }
    P.loaded = true;
    return P;
}

std::string ProfileStaleReason(const DeviceProfile& p, const std::string& calib_set, const std::string& build_id,
                               std::uint64_t binary_size, std::int64_t binary_mtime, bool have_stat) {
    auto get = [&](const char* k, const char* dflt) {
        auto it = p.kv.find(k);
        return it == p.kv.end() ? std::string(dflt) : it->second;
    };
    const std::string cs = get("calib_set", "None");
    if (cs != calib_set) return "calib_set " + cs + " != " + calib_set;
    const std::string pid = get("binary_build_id", "-");
    if (!build_id.empty() && pid != "-") {
        if (build_id != pid) return "build id " + pid.substr(0, 16) + " != " + build_id.substr(0, 16);
        return std::string();
    }
    if (!have_stat || get("binary_size", "") != std::to_string(binary_size) ||
        get("binary_mtime", "") != std::to_string(binary_mtime))
        return "size/mtime differ (no build id on one side)";
    return std::string();
}

// --- estimate --------------------------------------------------------------
CostEstimate Estimate(const CostModels& M, const std::string& verb, const std::vector<std::string>& args, unsigned threads,
                      const DeviceProfile* profile, unsigned hw) {
    CostEstimate out;
    out.verb = verb;
    out.model = verb + "_cost/v1";
    Features F;
    if (!CostFeatures(verb, args, F, out.error)) return out;
    const std::string& var = F.variant;
    const std::string& reg = F.regime;
    const std::map<std::string, double>& f = F.f;
    // T = max(1, min(threads, hw or cpu_count()))
    const unsigned T = std::max(1u, std::min(threads, hw == 0 ? 1u : hw));
    out.T_eff = T;
    double xs[2] = {1.0, 1.0};
    const char* const Q[2] = {"wall", "rss"};
    for (int qi = 0; qi < 2; ++qi) {
        const std::string q = Q[qi];
        auto sit = M.slots.find(Key4(verb, var, reg, q));
        if (sit == M.slots.end()) {
            out.error = "no model for " + verb + " " + var + " " + reg + " " + q;
            return out;
        }
        const std::vector<std::string>& Ts = sit->second;
        const bool star = std::find(Ts.begin(), Ts.end(), "*") != Ts.end();
        std::vector<std::string> slots = star ? std::vector<std::string>{"*"} : std::vector<std::string>{"1", "16"};
        std::map<std::string, double> ys;
        for (const std::string& t : slots) {
            auto it = M.term.find(MakeKey(verb, var, reg, q, t));
            if (it == M.term.end()) {
                out.error = "no model for " + verb + " " + var + " " + reg + " " + q + " at threads " + t;
                return out;
            }
            ys[t] = Ev(it->second, f);
        }
        double y = 0.0;
        if (star) {
            y = ys["*"];
        } else if (q == "wall") {
            const double fs = std::min(1.0, std::max(0.0, (ys["16"] / ys["1"] - 1.0 / 16) / (15.0 / 16)));
            y = ys["1"] * (fs + (1 - fs) / T);
        } else {
            y = ys["1"] + (ys["16"] - ys["1"]) * std::log(static_cast<double>(T)) / std::log(16.0);
            // S3 Fix round 1 (S3 audit M1): beyond T = 16 the RSS is never below the T = 16 model (Python max(ys16, y))
            if (T > 16 && !(y > ys["16"])) y = ys["16"];
            out.model = verb + "_cost/v2";   // the threaded-RSS estimators carry the rule change in their id
        }
        double band = 1.0, x = 1.0;
        const std::vector<std::string> rel =
            star ? slots : (T <= 1 ? std::vector<std::string>{"1"} : T >= 16 ? std::vector<std::string>{"16"} : slots);
        for (const std::string& t : rel) {
            const GroupKey gk = MakeKey(verb, var, reg, q, t);
            auto gm = M.gmax.find(gk);
            if (gm == M.gmax.end()) {
                out.error = "no gmax row for " + gk;
                return out;
            }
            x = std::max(x, ys[t] / gm->second);   // cost space, per evaluated slot model
            auto rs = M.resid.find(gk);
            band = std::max(band, std::exp(rs == M.resid.end() ? 0.0 : rs->second));
            const std::vector<Monomial>& terms = M.term.at(gk);
            std::set<std::string> own;
            for (const Monomial& m : terms)
                for (const auto& ke : m.exps) own.insert(ke.first);
            auto gr = M.grid.find(gk);
            if (gr != M.grid.end()) {
                for (const auto& kv : gr->second) {   // upper side only
                    const std::string& k = kv.first;
                    const double hi = kv.second.second;
                    const double fk = f.at(k);
                    if (fk > hi * 1.0000001) {
                        if (std::find(out.x_features.begin(), out.x_features.end(), k) == out.x_features.end())
                            out.x_features.push_back(k);
                        if (own.count(k)) x = std::max(x, fk / hi);
                    }
                }
            }
            auto lc = M.lowconf.find(gk);
            if (lc != M.lowconf.end()) {
                for (const std::string& why : lc->second) {
                    if (Starts(why, "per_prime") && q == "wall") {
                        bool have = false;
                        double pp0 = 0.0;
                        for (const Monomial& m : terms) {
                            bool has_p2 = false;
                            for (const auto& ke : m.exps) has_p2 = has_p2 || ke.first == "p2";
                            if (!has_p2) continue;
                            double p = 1.0;
                            for (const auto& ke : m.exps) p *= std::pow(f.at(ke.first), ke.second);
                            pp0 = m.coef * p;
                            have = true;
                            break;
                        }
                        if (!have || pp0 < 0.10 * ys[t]) continue;
                    } else if (Starts(why, "per_prime")) {
                        continue;
                    }
                    const std::string tag = q + "@T=" + t + ":" + why;
                    if (std::find(out.low_confidence.begin(), out.low_confidence.end(), tag) == out.low_confidence.end())
                        out.low_confidence.push_back(tag);
                }
            }
        }
        if (qi == 0) {
            out.wall_s = y;
            out.wall_band = band;
        } else {
            out.rss_mib = y;
            out.rss_band = band;
        }
        xs[qi] = x;
    }
    out.x_wall = xs[0];
    out.x_rss = xs[1];
    out.x_factor = std::max(xs[0], xs[1]);
    out.extrapolated = out.x_factor > 1.0000001 || !out.x_features.empty();
    double fac = 1.0;
    std::string cal = "uncalibrated";
    if (profile != nullptr && profile->loaded) {
        auto kc = M.klass.find(verb);
        const std::string cls = kc == M.klass.end() ? std::string("None") : kc->second;
        auto f1i = profile->factor.find({cls, "1"});
        auto fmi = profile->factor.find({cls, "mt"});
        const double f1 = f1i == profile->factor.end() ? 0.0 : f1i->second;
        const double fm = fmi == profile->factor.end() ? 0.0 : fmi->second;
        auto ti = profile->kv.find("threads_mt");
        const long long tm = ti == profile->kv.end() ? 16 : std::atoll(ti->second.c_str());
        if (f1 != 0.0 && fm != 0.0) {   // `if f1 and fm` (a NaN is truthy, as in Python)
            if (T <= 1) {
                fac = f1;
            } else if (static_cast<long long>(T) >= tm) {
                fac = fm;
            } else {
                const double w = std::log(static_cast<double>(T)) / std::log(static_cast<double>(tm));
                fac = std::exp((1 - w) * std::log(f1) + w * std::log(fm));
            }
            auto cs = profile->kv.find("calib_set");
            cal = (cs == profile->kv.end() ? std::string("?") : cs->second) + ":" + cls;
        }
    }
    out.wall_s *= fac;
    out.device_factor = fac;
    out.calibration = cal;
    for (int qi = 0; qi < 2; ++qi) {
        auto si = M.safety.find(Q[qi]);
        const Safety s = si == M.safety.end() ? Safety{} : si->second;
        const double qv = qi == 0 ? out.wall_s : out.rss_mib;
        const double band = qi == 0 ? out.wall_band : out.rss_band;
        const double hi = qv * std::max(band, s.band_floor) * std::pow(xs[qi], s.x_exp);
        (qi == 0 ? out.wall_hi : out.rss_hi) = hi;
    }
    out.ok = true;
    return out;
}

const char* CostLevelName(CostLevel l) {
    switch (l) {
        case CostLevel::Silent: return "silent";
        case CostLevel::Normal: return "normal";
        case CostLevel::Strong: return "strong";
        case CostLevel::Refuse: return "refuse";
    }
    return "silent";
}

// --- decide ----------------------------------------------------------------
CostDecision Decide(const CostModels& M, const CostEstimate& e, double confirm_wall_s, double mem_avail_mib,
                    double rss_cap_mib, double c_free_gb) {
    CostDecision d;
    auto sw = M.safety.find("wall");
    const double strong_x = sw == M.safety.end() ? 2.0 : sw->second.strong_x;
    // lim = [v for v in (rss_cap_mib, 0.8 * mem_avail_mib if mem_avail_mib else None) if v]
    std::vector<double> lim;
    if (rss_cap_mib > 0) lim.push_back(rss_cap_mib);
    if (mem_avail_mib > 0) lim.push_back(0.8 * mem_avail_mib);
    const double lmin = lim.empty() ? -1.0 : *std::min_element(lim.begin(), lim.end());
    d.mem_limit_mib = lmin;
    if (!e.ok) {   // no model (S3 audit M1 (2)): rule 1 cannot be checked, so a known memory limit refuses
        if (!lim.empty()) {
            d.level = CostLevel::Refuse;
            d.reasons.push_back("no cost model (a memory limit is known)");
        } else if (c_free_gb >= 0 && c_free_gb <= 10) {
            d.level = CostLevel::Refuse;
            d.reasons.push_back("C: <= 10 GB free");
        } else {
            d.level = CostLevel::Strong;
            d.reasons.push_back("no cost model");
        }
        return d;
    }
    {   // S3 audit M1 (1): a non-finite or non-positive estimate is never admitted
        bool usable = true;
        for (double v : {e.wall_s, e.rss_mib, e.wall_hi, e.rss_hi}) usable = usable && std::isfinite(v) && v > 0;
        if (!usable) {
            d.level = CostLevel::Refuse;
            d.reasons.push_back("no usable estimate (wall / rss not finite and > 0)");
            return d;
        }
    }
    const bool mem_refuse = !lim.empty() && e.rss_hi > lmin;
    if (mem_refuse || (c_free_gb >= 0 && c_free_gb <= 10)) {
        d.level = CostLevel::Refuse;
        d.reasons.push_back(mem_refuse ? "rss_hi > min(R_cap, 0.8 MemAvailable)" : "C: <= 10 GB free");
        return d;
    }
    if (mem_avail_mib > 0 && e.rss_hi > 0.5 * mem_avail_mib) d.reasons.push_back("rss_hi > 0.5 MemAvailable");
    if (e.wall_hi > 3600) d.reasons.push_back("wall_hi > 1 h");
    if (e.x_factor > strong_x) d.reasons.push_back("extrapolated x > " + Fmt("%g", strong_x) + " (cost space)");
    if (!e.low_confidence.empty()) d.reasons.push_back("low-confidence model");
    if (c_free_gb >= 0 && c_free_gb <= 20) d.reasons.push_back("C: <= 20 GB free");
    if (!d.reasons.empty()) {
        d.level = CostLevel::Strong;
    } else if (e.wall_hi > confirm_wall_s || e.extrapolated) {
        d.level = CostLevel::Normal;
        d.reasons.push_back(e.wall_hi > confirm_wall_s ? "wall_hi > confirm_wall_s" : "extrapolated");
    }
    if (d.level != CostLevel::Silent) d.kill_s = std::min(6 * 3600.0, std::max(1800.0, 3 * e.wall_hi + 60));
    return d;
}

std::string FormatSeconds(double s) {
    if (!(s >= 0)) return "?";
    if (s < 1) return Fmt("%.2f s", s);
    if (s < 60) return Fmt("%.0f s", s);
    if (s < 3600) {
        const long long t = static_cast<long long>(std::llround(s));
        return std::to_string(t / 60) + " min " + std::to_string(t % 60) + " s";
    }
    if (s < 86400 * 3) {
        const long long t = static_cast<long long>(std::llround(s / 60));
        return std::to_string(t / 60) + " h " + std::to_string(t % 60) + " min";
    }
    return Fmt("%.1f days", s / 86400);
}

std::string FormatMiB(double mib) {
    if (!(mib >= 0)) return "?";
    if (mib < 1024) return Fmt("%.0f MiB", mib);
    if (mib < 1024.0 * 1024) return Fmt("%.1f GiB", mib / 1024);
    return Fmt("%.1f TiB", mib / (1024.0 * 1024));
}

} // namespace cost
} // namespace console
