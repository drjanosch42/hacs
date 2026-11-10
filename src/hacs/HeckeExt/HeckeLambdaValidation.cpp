// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HeckeLambdaValidation (standard) -- the gate of WP-HL (hecke-extensions
// slot 19): the truncated Lambda-adic ordinary Hecke algebra
// T^ord(a) (x) Lambda/(ell^K, X^m) of HeckeExt/HeckeHidaLambda.  Binding text:
// [U] docs/latex/hida_hecke_algebra.tex 0.3.0-pre.3 part 3 (sec:u-lambda,
// thm:u-interp, prop:u-lambdaprod, prop:u-eistrunc, pr:u-lambda (U11)-(U13)).
// Oracle: the R19-2 prototype (docs/implementation/slot19/R19-2_report.md,
// r192/data/forms_*.json and lambda_*.json, computed in Python with
// python-flint products, refereed by F4), never this code's own output.
//
//   (1) self-tests: the NTT product against the schoolbook product; Delta =
//       q prod (1 - q^n)^24 and E4^3 - E6^2 = 1728 Delta on 600 coefficients;
//       Berkowitz against exact determinants; the tangent-number B_k/k against
//       WP-NF's exact Bernoulli numbers (k <= 200);
//   (2) arm (Q), the q-expansion input: every ordinary Hecke matrix of
//       r192/data/forms_37.json and forms_103.json reproduced exactly (all
//       weights, e M_k and e S_k, T_p and U, mod ell^10);
//   (3) the Lambda stage: r192/data/lambda_37_32.json and lambda_103_24.json
//       reproduced exactly, field by field (the presentations F, P_q, U, G~,
//       the general route, J, the further-weight (U11) checks, alpha, rungs,
//       G~(0) mod ell^5); the gates of R19-2 Sec. 3 / [U] (U11)-(U13) verbatim,
//       KL 47910634 at (37, 32), the F4 m10 ratio F1/G = 35 mod 37 along
//       k = 32 .. 284; the hecke_lambda/v1 record lines with their digests;
//   (4) arm (T), the cohomological tower route as a cross-check (owner ruling
//       2026-10-03): HeckeHidaTower's charpolys of T_2, T_3 on H^1_ord at
//       (30, 37), (66, 37), (22, 103), m <= 2, against Eichler-Shimura from
//       the forms side, (Y - eis_p) charpoly(T_p | e S_k)^2;
//   (5) the hida_tower/v1 catalogue rows at ell <= 13 against the forms side
//       (R19-2 P7's check, in C++);
//   (6) negative controls: thm:u-interp's precision bound (K + m - 2 digits,
//       K + m - 2 weights, non-consecutive weights, forms data below K + N - 1,
//       k - 1 < W) is REFUSED; a perturbed Hecke matrix at a further weight is
//       caught by the (U11) gate and at a node by the (U13) gate.
// HECKE_LAMBDA_ALL=1 (read-only arm, docs/validation-catalogue.md env table)
// adds (691, 12) (forms_691_12.json + lambda_691_12.json) and (691, 200)
// (lambda_691_200.json; its 5 MB forms file is not tracked) and the 9
// catalogue rows at 691: ~1-3 min, < 1 GB.  Prints
// "EFFECTIVE HeckeLambdaValidation arm=<DEFAULT|ALL> cells=<k>".
// (slot-25 H1) HECKE_LAMBDA_CELLS="ell:a:K:m,..." (read-only arm, M1 of 05-hida-targets.md Sec. 2.6) REPLACES the default arms by the
// truncation of the Eisenstein component at each pair, any cuspidal rank g ([U] part 6, sec:u-eislambda): the (1+g, g) ranks, the
// monogenic route at rank 1+g, F(0) = 0, (F1) = (G~) = (Z - (alpha - c)), the general route, the structure (fibre_LxL / regular_deg_g),
// I0 = (T_p - e_p), (U1) and K9 at every weight, v(disc f_k) = 1 and the rung prediction (U23) at g = 2, and at (547, 486) the data
// (D1), (D3) of prop:u-H12.  Prints "EFFECTIVE HeckeLambdaValidation arm=CELLS cells=<k>" and "HECKE_LAMBDA_CELLS ... verdict=".
// Writes nothing, no golden object, no digest of record.
#include "HeckeLambdaValidation.h"

#include <sys/resource.h>

#include <cctype>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "../../ModArithmetic/Fnv.h"
#include "HeckeHidaEisLambda.h"
#include "HeckeHidaLambda.h"
#include "HeckeHidaTower.h"
#include "../Util/ValidationCheck.h"

namespace {

using hecke_ext::lambda::u128;
using hecke_ext::lambda::U128Text;
using hecke_ext::lambda::Vec;
using hecke_ext::lambda::Mat;
namespace L = hecke_ext::lambda;

oc_util::Checker Check("  [FAIL] ");

double Now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

long PeakRssMiB() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss / 1024;
}

// ---------------------------------------------------------------------------
// a minimal JSON reader (objects, arrays, integers kept as decimal text,
// true / false / null, strings; floats kept as text and never compared)
// ---------------------------------------------------------------------------
struct JV {
    enum Kind { Null, Bool, Num, Str, Arr, Obj } kind = Null;
    bool b = false;
    std::string s;                                       // Num (decimal text) or Str
    std::vector<JV> arr;
    std::vector<std::pair<std::string, JV>> obj;
    const JV* get(const std::string& k) const {
        for (const auto& kv : obj)
            if (kv.first == k) return &kv.second;
        return nullptr;
    }
    const JV& at(const std::string& k) const {
        const JV* v = get(k);
        if (!v) throw std::runtime_error("JSON: missing key " + k);
        return *v;
    }
};

struct JParser {
    const std::string& t;
    size_t i = 0;
    explicit JParser(const std::string& text) : t(text) {}
    void ws() {
        while (i < t.size() && (t[i] == ' ' || t[i] == '\n' || t[i] == '\r' || t[i] == '\t')) ++i;
    }
    JV parse() {
        ws();
        if (i >= t.size()) throw std::runtime_error("JSON: eof");
        JV v;
        const char c = t[i];
        if (c == '{') {
            v.kind = JV::Obj;
            ++i;
            ws();
            if (t[i] == '}') {
                ++i;
                return v;
            }
            for (;;) {
                ws();
                JV key = parse();
                ws();
                if (t[i] != ':') throw std::runtime_error("JSON: ':' expected");
                ++i;
                v.obj.emplace_back(key.s, parse());
                ws();
                if (t[i] == ',') {
                    ++i;
                    continue;
                }
                if (t[i] == '}') {
                    ++i;
                    break;
                }
                throw std::runtime_error("JSON: ',' or '}' expected");
            }
        } else if (c == '[') {
            v.kind = JV::Arr;
            ++i;
            ws();
            if (t[i] == ']') {
                ++i;
                return v;
            }
            for (;;) {
                v.arr.push_back(parse());
                ws();
                if (t[i] == ',') {
                    ++i;
                    continue;
                }
                if (t[i] == ']') {
                    ++i;
                    break;
                }
                throw std::runtime_error("JSON: ',' or ']' expected");
            }
        } else if (c == '"') {
            v.kind = JV::Str;
            ++i;
            while (t[i] != '"') {
                if (t[i] == '\\') ++i;
                v.s.push_back(t[i++]);
            }
            ++i;
        } else if (t.compare(i, 4, "true") == 0) {
            v.kind = JV::Bool;
            v.b = true;
            i += 4;
        } else if (t.compare(i, 5, "false") == 0) {
            v.kind = JV::Bool;
            i += 5;
        } else if (t.compare(i, 4, "null") == 0) {
            i += 4;
        } else {
            v.kind = JV::Num;
            while (i < t.size() && (std::isdigit(static_cast<unsigned char>(t[i])) || t[i] == '-' || t[i] == '.' ||
                                    t[i] == 'e' || t[i] == 'E' || t[i] == '+'))
                v.s.push_back(t[i++]);
        }
        return v;
    }
};

bool ReadJson(const std::string& path, JV& out) {
    std::ifstream in(path);
    if (!in) return false;
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string text = ss.str();
    JParser p(text);
    out = p.parse();
    return true;
}

JV JNum(const std::string& s) {
    JV v;
    v.kind = JV::Num;
    v.s = s;
    return v;
}
JV JNum(u128 x) { return JNum(U128Text(x)); }
JV JInt(uint64_t x) { return JNum(std::to_string(x)); }
JV JBool(bool b) {
    JV v;
    v.kind = JV::Bool;
    v.b = b;
    return v;
}
JV JStr(const std::string& s) {
    JV v;
    v.kind = JV::Str;
    v.s = s;
    return v;
}
JV JArr() {
    JV v;
    v.kind = JV::Arr;
    return v;
}
JV JObj() {
    JV v;
    v.kind = JV::Obj;
    return v;
}
JV JVec(const Vec& p) {
    JV v = JArr();
    for (u128 x : p) v.arr.push_back(JNum(x));
    return v;
}
JV JPolys(const std::vector<Vec>& ps) {
    JV v = JArr();
    for (const auto& p : ps) v.arr.push_back(JVec(p));
    return v;
}
JV JMat(const Mat& A) { return JPolys(A); }

// Compare `got` with `want`; objects by key set (order-free), arrays in order.
// Returns "" when equal, else the path of the first difference.
std::string JDiff(const JV& got, const JV& want, const std::string& path) {
    if (got.kind != want.kind) return path + " (kind)";
    switch (got.kind) {
        case JV::Null: return "";
        case JV::Bool: return got.b == want.b ? "" : path + " (" + (got.b ? "true" : "false") + " vs " + (want.b ? "true" : "false") + ")";
        case JV::Num:
        case JV::Str: return got.s == want.s ? "" : path + " (" + got.s + " vs " + want.s + ")";
        case JV::Arr:
            if (got.arr.size() != want.arr.size())
                return path + " (length " + std::to_string(got.arr.size()) + " vs " + std::to_string(want.arr.size()) + ")";
            for (size_t i = 0; i < got.arr.size(); ++i) {
                const std::string d = JDiff(got.arr[i], want.arr[i], path + "[" + std::to_string(i) + "]");
                if (!d.empty()) return d;
            }
            return "";
        case JV::Obj:
            if (got.obj.size() != want.obj.size()) {
                std::string ks;
                for (const auto& kv : want.obj)
                    if (!got.get(kv.first)) ks += " -" + kv.first;
                for (const auto& kv : got.obj)
                    if (!want.get(kv.first)) ks += " +" + kv.first;
                return path + " (keys:" + ks + ")";
            }
            for (const auto& kv : got.obj) {
                const JV* w = want.get(kv.first);
                if (!w) return path + "." + kv.first + " (unexpected key)";
                const std::string d = JDiff(kv.second, *w, path + "." + kv.first);
                if (!d.empty()) return d;
            }
            return "";
    }
    return "";
}

// ---------------------------------------------------------------------------
// our objects as the prototype's JSON
// ---------------------------------------------------------------------------
JV FormsSpaceJson(const L::FormsSpace& S) {
    JV o = JObj();
    o.obj.emplace_back("s", JInt(S.s));
    o.obj.emplace_back("N", JInt(S.N));
    JV mats = JObj();
    for (const auto& n : S.names) mats.obj.emplace_back(n, JMat(S.mats.at(n)));
    o.obj.emplace_back("mats", mats);
    o.obj.emplace_back("comm_ok", JBool(S.comm_ok));
    return o;
}

JV FormsWeightJson(const L::FormsWeight& w) {
    JV o = JObj();
    o.obj.emplace_back("ell", JInt(w.ell));
    o.obj.emplace_back("k", JInt(w.k));
    o.obj.emplace_back("W", JInt(w.W));
    o.obj.emplace_back("dim_M", JInt(w.dim_M));
    o.obj.emplace_back("dim_S", JInt(w.dim_S));
    o.obj.emplace_back("L", JInt(w.L));
    o.obj.emplace_back("M", FormsSpaceJson(w.M));
    o.obj.emplace_back("S", FormsSpaceJson(w.S));
    o.obj.emplace_back("stable_ok", JBool(w.stable_ok));
    JV e = JObj();
    for (const auto& kv : w.eis) e.obj.emplace_back(kv.first, JNum(kv.second));
    o.obj.emplace_back("eis", e);
    return o;
}

// the oracle's forms weight without its timing fields
JV StripTimings(const JV& w) {
    JV o = JObj();
    for (const auto& kv : w.obj)
        if (kv.first != "t_series_s" && kv.first != "t_total_s" && kv.first != "rss_mib") o.obj.push_back(kv);
    return o;
}

JV ComponentJson(const L::Component& C, bool eis) {
    JV o = JObj();
    o.obj.emplace_back("Y", JStr(C.Y));
    o.obj.emplace_back("F", JPolys(C.F));
    JV ops = JObj();
    for (const auto& n : C.op_names) ops.obj.emplace_back(n, JPolys(C.ops.at(n)));
    o.obj.emplace_back("ops", ops);
    if (eis) {
        o.obj.emplace_back("G", JVec(C.G));
        o.obj.emplace_back("F0_zero", JBool(C.F0_zero));
        o.obj.emplace_back("F1_eq_G", JBool(C.F1_eq_G));
        o.obj.emplace_back("G_eq_alpha", JBool(C.G_eq_alpha));
        o.obj.emplace_back("cong_log", JInt(C.cong_log));
        o.obj.emplace_back("cong_pred", JInt(C.cong_pred));
        if (C.has_general) {
            const L::GeneralRoute& g = C.general;
            JV gr = JObj();
            gr.obj.emplace_back("A_log", JInt(g.A_log));
            gr.obj.emplace_back("quot_log", JInt(g.quot_log));
            gr.obj.emplace_back("mingens", JInt(g.mingens));
            gr.obj.emplace_back("free", JBool(g.free));
            gr.obj.emplace_back("FY_zero", JBool(g.FY_zero));
            gr.obj.emplace_back("Y_generates", JBool(g.Y_generates));
            gr.obj.emplace_back("coker_log", JInt(g.coker_log));
            gr.obj.emplace_back("J", JPolys(g.J));
            gr.obj.emplace_back("J_log_index", JInt(g.J_log_index));
            gr.obj.emplace_back("J_eq_G", JBool(g.J_eq_G));
            o.obj.emplace_back("general", gr);
        }
    } else {
        JV fu = JArr();
        for (const auto& f : C.further) {
            JV t = JArr();
            t.arr.push_back(JInt(f.k));
            t.arr.push_back(JBool(f.held));
            t.arr.push_back(JInt(f.prec));
            fu.arr.push_back(t);
        }
        o.obj.emplace_back("further", fu);
    }
    return o;
}

JV ClassJson(const L::ClassResult& r) {
    JV o = JObj();
    o.obj.emplace_back("ell", JInt(r.ell));
    o.obj.emplace_back("a", JInt(r.a));
    JV ws = JArr();
    for (uint32_t k : r.weights) ws.arr.push_back(JInt(k));
    o.obj.emplace_back("weights", ws);
    o.obj.emplace_back("W", JInt(r.W));
    o.obj.emplace_back("d", JInt(r.d));
    JV rE = JArr(), rc = JArr();
    rE.arr = {JInt(r.rank_E[0]), JInt(r.rank_E[1])};
    rc.arr = {JInt(r.rank_c[0]), JInt(r.rank_c[1])};
    o.obj.emplace_back("rank_E", rE);
    o.obj.emplace_back("rank_c", rc);
    o.obj.emplace_back("Y_E", r.Y_E.empty() ? JV() : JStr(r.Y_E));
    o.obj.emplace_back("Y_c", r.Y_c.empty() ? JV() : JStr(r.Y_c));
    if (r.has_G0) o.obj.emplace_back("G0_mod_l5", JNum(r.G0_mod_l5));
    if (r.alpha_prec >= 2) o.obj.emplace_back("rung2", r.has_rung2 ? JInt(r.rung2) : JV());
    if (r.alpha_prec >= 3) o.obj.emplace_back("rung3", r.has_rung3 ? JInt(r.rung3) : JV());
    JV al = JArr();
    al.arr = {JNum(r.alpha), JInt(r.alpha_prec)};
    o.obj.emplace_back("alpha", al);
    JV trs = JArr();
    for (const auto& t : r.truncations) {
        JV x = JObj();
        x.obj.emplace_back("K", JInt(t.K));
        x.obj.emplace_back("m", JInt(t.m));
        x.obj.emplace_back("centre_k", JInt(t.centre_k));
        x.obj.emplace_back("N", JInt(t.N));
        x.obj.emplace_back("W_mono", JInt(t.W_mono));
        x.obj.emplace_back("W_gen", JInt(t.W_gen));
        if (t.E.present) x.obj.emplace_back("E", ComponentJson(t.E, true));
        if (t.c.present) x.obj.emplace_back("c", ComponentJson(t.c, false));
        trs.arr.push_back(x);
    }
    o.obj.emplace_back("truncations", trs);
    return o;
}

std::string SourcePath(const std::string& rel) {
#ifdef OC_SOURCE_DIR
    return std::string(OC_SOURCE_DIR) + "/" + rel;
#else
    return rel;
#endif
}

std::vector<uint64_t> PrimesUpTo(uint64_t n, uint64_t skip) {
    std::vector<uint64_t> out;
    for (uint64_t p = 2; p <= n; ++p) {
        bool pr = true;
        for (uint64_t d = 2; d * d <= p; ++d)
            if (p % d == 0) {
                pr = false;
                break;
            }
        if (pr && p != skip) out.push_back(p);
    }
    return out;
}

// One class of the R19-2 runs.
struct ClassSpec {
    uint64_t ell;
    uint32_t W, pmax, a, centre2;
    std::vector<uint32_t> weights;                 // the forms run's weights
    std::string forms_json, lambda_json;           // forms_json "" = not tracked
    bool has_kl;
    u128 kl;                                       // R19-4 route L, mod ell^5
    uint32_t rung2, rung3;                         // R19-2 Sec. 3 / [U] prop:u-eistrunc
    uint64_t alpha_l2;
    std::vector<std::vector<u128>> F1_m3, u_m3;    // R19-2 Sec. 3 table (m = 3, centre x_{k0})
    std::vector<uint32_t> residue_degrees;
};

std::vector<ClassSpec> Specs(bool all) {
    std::vector<ClassSpec> s;
    s.push_back({37, 10, 43, 32, 284, {32, 68, 104, 140, 176, 212, 248, 284, 320, 356, 392, 428, 464},
                 "forms_37.json", "lambda_37_32.json", true, 47910634, 284, 37580, 925,
                 {{47841, 28162, 22602}}, {{1267, 44910, 14022}}, {1}});
    s.push_back({103, 10, 53, 24, 228, {24, 126, 228, 330, 432, 534}, "forms_103.json", "lambda_103_24.json", false, 0, 228,
                 914250, 2266, {{565470, 47653, 641768}}, {{1065671, 146719, 1016652}}, {1}});
    if (all) {
        s.push_back({691, 8, 307, 12, 342252, {12, 702, 1392, 2082, 2772, 3462}, "forms_691_12.json", "lambda_691_12.json",
                     true, static_cast<u128>(118721726217077ull), 342252, 49928412, 143037,
                     {{2073, 1235997, 198376046}}, {{27386683, 274530782, 146978408}}, {}});
        s.push_back({691, 8, 307, 200, 342440, {200, 890, 1580, 2270, 2960, 3650}, "", "lambda_691_200.json", true,
                     static_cast<u128>(23579310810939ull), 342440, 159113510, 272945,
                     {{168394627, 60228727, 53263999}}, {{304423457, 326178075, 130222561}}, {1, 2, 2, 10}});
    }
    return s;
}

const char* kDataDir = "docs/implementation/slot19/r192/data/";

// Run one class end to end; returns the forms data (for the negative controls).
std::vector<L::FormsWeight> RunSpec(const ClassSpec& sp, std::vector<std::string>& records) {
    const double t0 = Now();
    const std::vector<uint64_t> primes = PrimesUpTo(sp.pmax, sp.ell);
    std::vector<L::FormsWeight> fw;
    for (uint32_t k : sp.weights) {
        fw.push_back(L::ComputeFormsWeight(sp.ell, k, sp.W, primes));
        if (sp.ell > 103)
            std::cout << "  FORMS ell=" << sp.ell << " k=" << k << " W=" << sp.W << " dim_M=" << fw.back().dim_M
                      << " s_M=" << fw.back().M.s << " s_S=" << fw.back().S.s << " L=" << fw.back().L
                      << " wall_s=" << fw.back().wall_s << " rss_mib=" << PeakRssMiB() << std::endl;
    }
    const double t_forms = Now() - t0;
    // (2) arm (Q) against the forms JSON
    if (!sp.forms_json.empty()) {
        JV want;
        const bool ok = ReadJson(SourcePath(std::string(kDataDir) + sp.forms_json), want);
        Check(ok, "oracle " + sp.forms_json + " readable");
        if (ok) {
            Check(want.at("ell").s == std::to_string(sp.ell) && want.at("W").s == std::to_string(sp.W), sp.forms_json + " ell, W");
            JV pr = JArr();
            for (uint64_t p : primes) pr.arr.push_back(JInt(p));
            const std::string dp = JDiff(pr, want.at("primes"), "primes");
            Check(dp.empty(), sp.forms_json + " primes: " + dp);
            const JV& wl = want.at("weights");
            Check(wl.arr.size() == fw.size(), sp.forms_json + " weight count");
            uint32_t same = 0;
            for (size_t j = 0; j < fw.size() && j < wl.arr.size(); ++j) {
                const std::string d = JDiff(FormsWeightJson(fw[j]), StripTimings(wl.arr[j]), "weights[" + std::to_string(j) + "]");
                Check(d.empty(), sp.forms_json + " " + d);
                if (d.empty()) ++same;
            }
            std::cout << "FORMS_ORACLE ell=" << sp.ell << " file=" << sp.forms_json << " weights=" << fw.size()
                      << " reproduced=" << same << "/" << wl.arr.size() << " (every T_p, U on e M_k, e S_k mod ell^"
                      << sp.W << ")" << std::endl;
        }
    }
    for (const auto& w : fw) Check(w.stable_ok && w.M.comm_ok && w.S.comm_ok, "forms stable / commuting at k=" + std::to_string(w.k));
    // (3) the Lambda stage
    L::ClassInput in{sp.ell, sp.W, primes, fw};
    const double t1 = Now();
    const L::ClassResult r = L::RunClass(in, sp.a, 3, {1, 2, 3}, {sp.centre2}, [](const std::string& s) {
        std::cout << "  " << s << std::endl;
    });
    const double t_lambda = Now() - t1;
    JV want;
    const bool ok = ReadJson(SourcePath(std::string(kDataDir) + sp.lambda_json), want);
    Check(ok, "oracle " + sp.lambda_json + " readable");
    if (ok) {
        const std::string d = JDiff(ClassJson(r), want, "$");
        Check(d.empty(), sp.lambda_json + " differs at " + d);
        std::cout << "LAMBDA_ORACLE ell=" << sp.ell << " a=" << sp.a << " file=" << sp.lambda_json
                  << " truncations=" << r.truncations.size() << " " << (d.empty() ? "IDENTICAL" : "DIFFERS at " + d) << std::endl;
    }
    // the R19-2 Sec. 3 table verbatim
    Check(r.lambda_one, "lambda = 1, mu = 0 (lem:u-lambdaone)");
    Check(r.has_rung2 && r.rung2 == sp.rung2, "first k with L_k >= 2 = " + std::to_string(sp.rung2));
    Check(r.has_rung3 && r.rung3 == sp.rung3, "first k with L_k >= 3 = " + std::to_string(sp.rung3) + " (PROVED, F4 M1)");
    Check(r.alpha % (static_cast<u128>(sp.ell) * sp.ell) == sp.alpha_l2, "alpha mod ell^2 = " + std::to_string(sp.alpha_l2));
    Check(r.alpha % sp.ell == 0 && r.alpha % (static_cast<u128>(sp.ell) * sp.ell) != 0, "v(alpha) = 1");
    if (sp.has_kl) Check(r.has_G0 && r.G0_mod_l5 == sp.kl, "KL: G~(0) mod ell^5 = R19-4's L_ell(1, omega^a) = " + U128Text(sp.kl));
    Check(r.residue_degrees == sp.residue_degrees, "complement residue degrees");
    uint32_t n_u11 = 0, n_u12 = 0, n_u13 = 0;
    for (const auto& t : r.truncations) {
        const auto& E = t.E;
        Check(E.present && E.has_general, "Eisenstein factor present");
        // (U13): F = Y (Y + F1), (F1) = (G~) = (Z - (alpha - c)); free of rank 2, generated by Y
        const bool u13 = E.F0_zero && E.F1_eq_G && E.G_eq_alpha && E.general.free && E.general.FY_zero && E.general.Y_generates;
        // (U12): |coker| = |Lambda/(G~)| = ell^{min(K, m L_c)}, J = (G~)
        const bool u12 = E.general.J_eq_G && E.cong_log == E.cong_pred && E.general.coker_log == E.cong_log &&
                         E.general.J_log_index == E.cong_log;
        Check(u13, "(U13) at m=" + std::to_string(t.m) + " centre " + std::to_string(t.centre_k));
        Check(u12, "(U12) at m=" + std::to_string(t.m) + " centre " + std::to_string(t.centre_k));
        n_u12 += u12;
        n_u13 += u13;
        if (t.c.present) {
            bool all = !t.c.further.empty();
            for (const auto& f : t.c.further) all = all && f.held;
            Check(all, "(U11) at every further weight, m=" + std::to_string(t.m));
            n_u11 += all;
        }
        if (t.m == 3 && t.centre_k == sp.weights[0] && E.F.size() == 3) {
            Check(E.F[1] == sp.F1_m3[0], "F1 at m = 3 (R19-2 Sec. 3 table)");
            const auto itU = E.ops.find("U");
            Check(itU != E.ops.end() && itU->second.size() == 2 && itU->second[0] == Vec{1, 0, 0} &&
                      itU->second[1] == sp.u_m3[0],
                  "U = 1 + u(Z) Y at m = 3 (R19-2 Sec. 3 table)");
        }
    }
    // F4 m10: the ratio (a_2(f_k) - eis_2)/G(x_k) = -F1(x_k)/G(x_k) = 35 mod 37 along k = 32 .. 284.
    // (F4's script f4_r192_lv.py forms F1k = a_2 - eis_2, i.e. -F1 in m10's own sign convention
    // F1 = eis_2 - a_2; so F1/G = -35 = 2 mod 37.  Both read here.)
    if (sp.ell == 37) {
        uint32_t held = 0, n = 0;
        std::string ratios;
        const L::ZMod R10(37, 10), F(37, 1);
        for (size_t j = 0; j < r.weights.size() && r.weights[j] <= 284; ++j) {
            ++n;
            const u128 mF1 = R10.neg(r.F1_at_weights[j]), G = r.G_at_weights[j];
            const uint32_t vF = R10.val(mF1, 99), vG = R10.val(G, 99);
            u128 ratio = 99;
            if (vF == vG && vG < 10) {
                const u128 f = (mF1 / R10.pw[vG]) % 37, g = (G / R10.pw[vG]) % 37;
                ratio = F.mul(f, F.inv(g));
            }
            ratios += (ratios.empty() ? "" : ",") + U128Text(ratio);
            held += ratio == 35;
        }
        Check(n == 8 && held == 8, "F4 m10: (a_2 - eis_2)(x_k)/G_32(x_k) = 35 mod 37 at the 8 weights 32..284");
        std::cout << "  F1_OVER_G ell=37 a=32 (a_2-eis_2)/G mod 37 at k=32..284: [" << ratios << "] held=" << held << "/" << n
                  << " (F4 m10; F1/G = -35 = 2)" << std::endl;
    }
    const u128* kl = sp.has_kl ? &sp.kl : nullptr;
    const auto lines = L::RecordLines(r, "Q", kl);
    for (const auto& s : lines) {
        std::cout << s << std::endl;
        records.push_back(s);
    }
    std::cout << "LAMBDA_CLASS ell=" << sp.ell << " a=" << sp.a << " truncations=" << r.truncations.size()
              << " U11=" << n_u11 << " U12=" << n_u12 << "/" << r.truncations.size() << " U13=" << n_u13 << "/"
              << r.truncations.size() << " forms_s=" << t_forms << " lambda_s=" << t_lambda << " rss_mib=" << PeakRssMiB()
              << std::endl;
    return fw;
}

template <typename F>
bool Refused(F f, std::string* msg) {
    try {
        f();
    } catch (const std::logic_error& e) {
        if (msg) *msg = e.what();
        return true;
    }
    return false;
}

Vec PolyMulMod(const Vec& a, const Vec& b, const L::ZMod& R) {
    Vec out(a.size() + b.size() - 1, 0);
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < b.size(); ++j) out[i + j] = R.add(out[i + j], R.mul(a[i], b[j]));
    return out;
}

// (Hi-ES) charpoly(T_p | H^1_ord) = (Y - eis_p) charpoly(T_p | e S_k)^2 mod ell^m
Vec EsPrediction(const L::FormsWeight& w, const std::string& op, const L::ZMod& R) {
    const Vec cs = L::Charpoly(w.S.mats.at(op), R);
    return PolyMulMod(Vec{R.neg(w.eis.at(op) % R.M), 1}, PolyMulMod(cs, cs, R), R);
}

Vec ParsePolyText(const std::string& s, const L::ZMod& R) {
    Vec out;
    std::string t = s;
    for (char& c : t)
        if (c == '[' || c == ']' || c == ',') c = ' ';
    std::istringstream is(t);
    std::string tok;
    while (is >> tok) {
        u128 x = 0;
        for (char c : tok) x = x * 10 + static_cast<u128>(c - '0');
        out.push_back(x % R.M);
    }
    return out;
}

// ---------------------------------------------------------------------------
// (slot-25 H1) the read-only arm HECKE_LAMBDA_CELLS="ell:a:K:m,..." (M1 of 05-hida-targets.md Sec. 2.6): the truncation
// T(a)_m (x) Lambda/(ell^K, X^m) of the Eisenstein component at an irregular pair of any cuspidal rank g, with [U] part 6's
// finite-weight readings ([U, lem:u-H0], [U, prop:u-H11], [U, prop:u-H12], [U, pr:u-H1]).  N = K + m - 1 consecutive weights
// a, a + (ell - 1), ... (thm:u-interp: N >= K + m - 1), forms data mod ell^{K+N-1} (the general route of prop:u-lambdaprod(3)),
// the T_p to the Sturm bound of the largest weight.  REPLACES the default arms (the self-tests run first).  At (547, 486) the
// data (D1), (D3) of prop:u-H12 are compared with the stage-05 scripts' values.
// ---------------------------------------------------------------------------
struct CellSpec {
    uint64_t ell;
    uint32_t a, K, m;
};

std::vector<CellSpec> ParseLambdaCells(const std::string& text) {
    std::vector<CellSpec> out;
    std::stringstream ss(text);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) continue;
        std::vector<uint64_t> f;
        std::stringstream ts(tok);
        std::string x;
        while (std::getline(ts, x, ':')) {
            if (x.empty() || x.find_first_not_of("0123456789") != std::string::npos)
                throw std::invalid_argument("HECKE_LAMBDA_CELLS: '" + tok + "' is not ell:a:K:m");
            f.push_back(std::stoull(x));
        }
        if (f.size() != 4 || f[0] < 5 || f[1] < 4 || f[1] % 2 || f[2] < 1 || f[3] < 1 || f[2] > 6 || f[3] > 6)
            throw std::invalid_argument("HECKE_LAMBDA_CELLS: '" + tok + "' needs ell >= 5, a >= 4 even, 1 <= K, m <= 6");
        out.push_back({f[0], static_cast<uint32_t>(f[1]), static_cast<uint32_t>(f[2]), static_cast<uint32_t>(f[3])});
    }
    return out;
}

namespace E = hecke_ext::eislambda;

std::string PolyTxt(const Vec& p) {
    std::string o = "[";
    for (size_t i = 0; i < p.size(); ++i) o += (i ? "," : "") + U128Text(p[i]);
    return o + "]";
}

void RunLambdaCell(const CellSpec& c, std::vector<std::string>& records) {
    const double t0 = Now();
    const uint64_t l = c.ell;
    const uint32_t N = c.K + c.m - 1, W = c.K + N - 1;
    std::vector<uint32_t> ks;
    for (uint32_t j = 0; j < N; ++j) ks.push_back(c.a + j * static_cast<uint32_t>(l - 1));
    const std::vector<uint64_t> primes = E::SturmPrimes(l, ks.back());
    std::cout << "LAMBDA_CELL ell=" << l << " a=" << c.a << " K=" << c.K << " m=" << c.m << " N=" << N << " W=" << W << " weights=" << ks.front()
              << ".." << ks.back() << " primes<=" << primes.back() << std::endl;
    // per weight: the forms data (independent jobs) and the block reading
    std::vector<L::FormsWeight> fw;
    std::vector<E::BlockReading> br;
    for (uint32_t k : ks) {
        fw.push_back(L::ComputeFormsWeight(l, k, W, primes));
        br.push_back(E::ReadBlockFrom(fw.back(), primes));
        const auto& b = br.back();
        std::cout << "  FORMS ell=" << l << " k=" << k << " dim_M=" << fw.back().dim_M << " s_M=" << fw.back().M.s << " s_S=" << fw.back().S.s
                  << " L=" << fw.back().L << " g=" << b.g << " v_Bk=" << b.v_Bk << " f_k=" << PolyTxt(b.f2) << " wall_s=" << fw.back().wall_s
                  << " rss_mib=" << PeakRssMiB() << std::endl;
        Check(fw.back().stable_ok && fw.back().M.comm_ok && fw.back().S.comm_ok, "forms stable / commuting at k=" + std::to_string(k));
    }
    const uint32_t g = br[0].g;
    for (const auto& b : br) Check(b.g == g, "the cuspidal rank g is the same at every weight of the class (thm:u-control)");
    Check(g >= 1, "g >= 1: the pair is irregular (cor:u-irregular)");
    const L::ZMod R(l, W);
    // (U1) on the forms side at every weight ([U, pr:u-U1]): v(f_k(e_k)) = v(det(T_2 - e_2 | block)) = v(B_k / k), and the shape
    // f_k == (x - e_k)^g mod ell; g = 2 and v(B_k/k) = 1: v(disc f_k) = 1 ([U, prop:u-H12](4))
    uint32_t u1 = 0, disc1 = 0, n_v1 = 0;
    for (const auto& b : br) {
        const bool ok = b.v_Bk >= 1 && !b.v_det.empty() && b.primes[0] == 2 && b.v_det[0] == b.v_Bk;
        u1 += ok;
        Check(ok, "(U1) forms side at k=" + std::to_string(b.k) + ": v(f_k(e_k)) = v(B_k/k) = " + std::to_string(b.v_Bk));
        bool k9 = false;
        const std::string line = E::K9Line(b, &k9);
        Check(k9, "K9's kernel checks at k=" + std::to_string(b.k) + ": " + line);
        if (b.k == ks[0]) std::cout << line << std::endl;
        if (g == 2 && b.v_Bk == 1) {
            ++n_v1;
            const Vec& f = b.f2;
            const u128 disc = R.sub(R.mul(f[1], f[1]), R.mul(4 % R.M, f[0]));
            const bool d1 = R.val(disc, W) == 1;
            disc1 += d1;
            Check(d1, "v(disc f_k) = 1 at k=" + std::to_string(b.k) + " (prop:u-H12(4): ramified DVR)");
        }
    }
    // the Lambda stage
    L::ClassInput in{l, W, primes, fw};
    const L::ClassResult r = L::RunClass(in, c.a, c.K, {c.m}, {}, [](const std::string& s) { std::cout << "  " << s << std::endl; });
    Check(r.rank_E[0] == g + 1 && r.rank_E[1] == g, "RunClass's Eisenstein factor has rank (1+g, g) = (" + std::to_string(g + 1) + "," +
                                                       std::to_string(g) + ")");
    Check(!r.Y_E.empty(), "the E-factor's monogenic route at rank 1+g: Y = T_p - eis_p generates (lem:u-H0(3))");
    Check(r.lambda_one, "lambda = 1, mu = 0 ((D3'), lem:u-lambdaone)");
    std::string structure = "open", i0 = "none_found";
    for (const auto& t : r.truncations) {
        const auto& Ec = t.E;
        Check(Ec.present && Ec.has_general, "Eisenstein factor present, general route run");
        Check(Ec.F.size() == g + 2, "F on e M_k has degree 1 + g");
        const bool u13 = Ec.F0_zero && Ec.F1_eq_G && Ec.G_eq_alpha && Ec.general.free && Ec.general.FY_zero && Ec.general.Y_generates;
        const bool u12 = Ec.general.J_eq_G && Ec.cong_log == Ec.cong_pred && Ec.general.coker_log == Ec.cong_log &&
                         Ec.general.J_log_index == Ec.cong_log;
        Check(u13, "F(0) = 0, (F1) = (G~) = (Z - (alpha - c)), A free, F(Y) = 0, Y generates, at m=" + std::to_string(t.m));
        Check(u12, "|coker| = |Lambda/(G~)| = ell^{min(K, m L_c)}, J = (G~), at m=" + std::to_string(t.m));
        // the structure ([U, prop:u-H11]; [U, prop:u-H12](1), (2), Steps 1-2 of its proof): F = Y F_S, F_S = Y^g + ... + c with c = F[1].
        // I0 = (Y) iff (c) = (G~) (here (F1) = (G~)); regular, = Z_ell[[Y]], iff moreover the Z-coefficient of c is a unit
        // (every lower coefficient lies in m_Lambda as Y is nilpotent mod m)
        if (Ec.F1_eq_G) i0 = r.Y_E.empty() ? "none_found" : "T_" + r.Y_E.substr(1);
        if (g == 1) {
            structure = "fibre_LxL";
        } else if (t.m >= 2 && Ec.F.size() == g + 2) {
            bool in_m = Ec.F[1][0] % l == 0;
            for (uint32_t i = 2; i <= g; ++i) in_m = in_m && Ec.F[i][0] % l == 0;
            const bool unit = Ec.F[1][1] % l != 0;
            structure = in_m && unit && Ec.F1_eq_G && r.lambda_one ? "regular_deg_g" : "open";
            Check(in_m, "b_i, c in m_Lambda (F == Y^{1+g} mod m_Lambda)");
        }
        // (U23) the rung prediction: f_{k*} mod ell^K from the truncation (m >= K), k* = r.rung2
        if (g >= 2 && t.m >= c.K && r.has_rung2) {
            const Vec f = E::RungPrediction(l, c.K, t.m, t.centre_k, r.rung2, Ec.F);
            const L::ZMod RK(l, c.K);
            u128 disc = 0;
            if (f.size() == 3) disc = RK.sub(RK.mul(f[1], f[1]), RK.mul(4 % RK.M, f[0]));
            const uint32_t vd = RK.val(disc, c.K);
            std::cout << "HIDA_RUNG_PREDICTION ell=" << l << " a=" << c.a << " k_star=" << r.rung2 << " f_kstar_mod_l" << c.K << "=" << PolyTxt(f)
                      << " v_disc>=" << vd << " (pr:u-H1 (U23): H1-R's anchor; the Hensel factor of chi(T_2 | S_{k*}) at 1 + 2^{k*-1} must equal it)"
                      << std::endl;
            Check(f.size() != 3 || vd >= 2, "(U23) v(disc f_{k*}) >= 2 (prop:u-H12(5): a non-maximal order)");
        }
    }
    std::cout << "HIDA_LAMBDA_STRUCTURE ell=" << l << " a=" << c.a << " g=" << g << " structure=" << structure << " I0_generator=" << i0
              << " Y=" << (r.Y_E.empty() ? "None" : r.Y_E + "-eis") << " U1_forms=" << u1 << "/" << br.size();
    if (g == 2) std::cout << " v_disc_1=" << disc1 << "/" << n_v1;
    std::cout << " rung2=" << (r.has_rung2 ? std::to_string(r.rung2) : std::string("None")) << std::endl;
    Check(g == 1 ? structure == "fibre_LxL" : structure != "open", "a decided structure (prop:u-H11 / prop:u-H12)");
    Check(i0 != "none_found", "I0 principal, generated by T_p - e_p (prop:u-H11 / prop:u-H12(2))");
    // (547, 486): the stage-05 data (D1), (D3) of prop:u-H12, reproduced in C++
    if (l == 547 && c.a == 486) {
        const L::ZMod R4(547, 4);
        Vec f4;
        for (u128 x : br[0].f2) f4.push_back(x % R4.M);
        Check(f4 == Vec{40544750686ull, 18353242915ull, 1}, "(D1) f_486 = x^2 + 18353242915 x + 40544750686 mod 547^4 (blk547.py)");
        uint32_t d1 = 0;
        for (const auto& b : br)
            if (b.k <= 2124 && b.f2.size() == 3) {
                const u128 disc = R.sub(R.mul(b.f2[1], b.f2[1]), R.mul(4 % R.M, b.f2[0]));
                u128 fe = 0;
                for (size_t j = 3; j-- > 0;) fe = R.add(R.mul(fe, b.e2), b.f2[j]);
                d1 += R.val(disc, W) == 1 && R.val(fe, W) == 1;
            }
        Check(d1 == 4, "(D1) v(disc f_k) = 1 and v(f_k(1 + 2^{k-1})) = 1 at k = 486, 1032, 1578, 2124");
        Check(br[0].v_Bk == 1 && br[1].v_Bk == 1, "(D3) v(B_k/k) = 1 at k = 486, 1032");
        Check(r.has_rung2 && r.rung2 == 55086, "(D3) the rung k* = 55086 (v(x_k - alpha) >= 2 first at 55086)");
        const auto bk = L::BernoulliOverK(547, 3, {54540u, 55086u, 55632u});
        const L::ZMod R3(547, 3);
        const uint32_t v0 = R3.val(bk.at(54540), 3), v1 = R3.val(bk.at(55086), 3), v2 = R3.val(bk.at(55632), 3);
        Check(v0 == 1 && v1 == 2 && v2 == 1, "(D3) v(B_k/k) = 2 at k* = 55086, = 1 at 55086 -/+ 546 (direct, mod 547^3)");
        std::cout << "HIDA_D1D3 ell=547 a=486 f486_mod_l4=" << PolyTxt(f4) << " D1_weights=" << d1 << "/4 v_Bk(486,1032)=(" << br[0].v_Bk << ","
                  << br[1].v_Bk << ") k_star=" << r.rung2 << " v_Bk(54540,55086,55632)=(" << v0 << "," << v1 << "," << v2 << ")" << std::endl;
    }
    for (const auto& s : L::RecordLines(r, "Q", nullptr)) {
        std::cout << s << std::endl;
        records.push_back(s);
    }
    std::cout << "LAMBDA_CELL_DONE ell=" << l << " a=" << c.a << " g=" << g << " wall_s=" << Now() - t0 << " rss_mib=" << PeakRssMiB() << std::endl;
}

}  // namespace

uint32_t HeckeLambdaValidation() {
    Check.reset();
    const double t_start = Now();
    if (const char* env_cells = std::getenv("HECKE_LAMBDA_CELLS")) {
        const std::vector<CellSpec> cells = ParseLambdaCells(env_cells);
        std::cout << "EFFECTIVE HeckeLambdaValidation arm=CELLS cells=" << cells.size() << std::endl;
        Check(L::NttSelfCheck(547, 7) && L::SeriesSelfCheck(547, 7, 600), "NTT and series self-tests at 547^7");
        std::vector<std::string> records;
        for (const auto& c : cells) {
            try {
                RunLambdaCell(c, records);
            } catch (const std::exception& e) {
                Check(false, "cell " + std::to_string(c.ell) + ":" + std::to_string(c.a) + " threw: " + e.what());
            }
        }
        uint64_t h = engine::kFnvOffset64;
        for (const auto& s : records) h = engine::FnvBytes(h, s + "\n");
        std::ostringstream hx;
        hx << std::hex << h;
        std::cout << "HECKE_LAMBDA_CELLS cells=" << cells.size() << " records=" << records.size() << " fnv=0x" << hx.str()
                  << " checks=" << Check.checks << " failing=" << Check.fails << " verdict=" << (Check.fails ? "FAIL" : "OK") << std::endl;
        std::cout << "HeckeLambdaValidation: checks=" << Check.checks << " failing=" << Check.fails << " wall_s=" << Now() - t_start
                  << " peak_rss_mib=" << PeakRssMiB() << std::endl;
        return Check.fails == 0 ? 0 : 1;
    }
    const char* env_all = std::getenv("HECKE_LAMBDA_ALL");
    const bool all = env_all && std::string(env_all) == "1";
    const std::vector<ClassSpec> specs = Specs(all);
    std::cout << "EFFECTIVE HeckeLambdaValidation arm=" << (all ? "ALL" : "DEFAULT") << " cells=" << specs.size() << std::endl;

    // ---- (1) self-tests
    Check(L::NttSelfCheck(37, 10) && L::NttSelfCheck(103, 10) && L::NttSelfCheck(691, 8), "NTT product == schoolbook");
    Check(L::SeriesSelfCheck(37, 10, 600) && L::SeriesSelfCheck(103, 10, 600) && L::SeriesSelfCheck(691, 8, 600),
          "Delta = q prod(1-q^n)^24 and E4^3 - E6^2 = 1728 Delta on 600 coefficients");
    Check(L::CharpolySelfCheck(), "Berkowitz == exact determinants (n <= 6, 7 points)");
    {
        uint32_t agree = 0, total = 0;
        for (auto lp : {std::pair<uint64_t, uint32_t>{37, 10}, {103, 9}, {691, 6}}) {
            std::vector<uint32_t> ks;
            for (uint32_t k = 2; k <= 200; k += 2)
                if (k % (lp.first - 1) != 0) ks.push_back(k);
            const auto a = L::BernoulliOverK(lp.first, lp.second, ks), b = L::BernoulliOverKExact(lp.first, lp.second, ks);
            for (uint32_t k : ks) {
                ++total;
                agree += a.at(k) == b.at(k);
            }
        }
        Check(agree == total, "tangent-number B_k/k == WP-NF exact Bernoulli (k <= 200 at 37, 103, 691)");
        std::cout << "SELFTEST bernoulli tangent_vs_exact=" << agree << "/" << total << std::endl;
    }
    std::cout << "SELFTEST ntt series charpoly done checks=" << Check.checks << " failing=" << Check.fails << std::endl;

    // ---- (2) + (3) the classes
    std::vector<std::string> records;
    std::vector<L::FormsWeight> fw37;
    for (const auto& sp : specs) {
        try {
            auto fw = RunSpec(sp, records);
            if (sp.ell == 37) fw37 = fw;
        } catch (const std::exception& e) {
            Check(false, "class (" + std::to_string(sp.ell) + ", " + std::to_string(sp.a) + ") threw: " + e.what());
        }
    }
    {
        uint64_t h = engine::kFnvOffset64;
        for (const auto& s : records) h = engine::FnvBytes(h, s + "\n");
        std::ostringstream hx;
        hx << std::hex << h;
        std::cout << "HECKE_LAMBDA_RECORDS lines=" << records.size() << " fnv=0x" << hx.str() << " (hecke_lambda/v1, not pinned)"
                  << std::endl;
        Check(records.size() == (all ? 42u : 24u), "record line count");
    }

    // ---- (6) negative controls
    {
        std::string msg;
        const std::vector<uint32_t> k5{32, 68, 104, 140, 176};
        // values of g = X (a Lambda-function): x_k mod 37^5; its truncation centred at x_32 is [x_32, 1, 0]
        Vec v5;
        const L::ZMod R5(37, 5), R3(37, 3);
        for (uint32_t k : k5) v5.push_back(R5.sub(R5.pow(38, k), 1));
        {
            Vec got;
            const bool ok = !Refused([&] { got = L::InterpolateTruncation(37, k5, v5, 5, 3, 3, 32); }, &msg);
            Check(ok && got == Vec{R3.sub(R3.pow(38, 32), 1), 1, 0}, "K + m - 1 digits accepted, g = X recovered");
        }
        Check(Refused([&] { L::InterpolateTruncation(37, k5, v5, 4, 3, 3, 32); }, &msg) &&
                  msg.find("REFUSED") != std::string::npos,
              "K + m - 2 digits REFUSED (thm:u-interp(2), sharp)");
        Check(Refused([&] { L::InterpolateTruncation(37, {32, 68, 104, 140}, Vec{1, 2, 3, 4}, 5, 3, 3, 32); }, nullptr),
              "N = K + m - 2 weights REFUSED (lem:u-wtprod)");
        Check(Refused([&] { L::InterpolateTruncation(37, {32, 68, 104, 176, 212}, v5, 5, 3, 3, 32); }, nullptr),
              "non-consecutive weights REFUSED");
        Check(Refused([&] { L::ComputeFormsWeight(37, 8, 10, {2, 3}); }, nullptr), "k - 1 < W REFUSED (U = T_ell needs it)");
        // forms data mod 37^4 cannot feed the monogenic route at m = 3 (K + m - 1 = 5), forms data mod
        // 37^5 not the general route (W' = K + N - 1 = 7)
        for (uint32_t Wf : {4u, 5u}) {
            std::vector<L::FormsWeight> fwW;
            const auto pr = PrimesUpTo(43, 37);
            for (uint32_t k : {32u, 68u, 104u, 140u, 176u}) fwW.push_back(L::ComputeFormsWeight(37, k, Wf, pr));
            L::ClassInput inW{37, Wf, pr, fwW};
            const bool ref = Refused([&] { L::RunClass(inW, 32, 3, {3}, {}, [](const std::string&) {}); }, &msg);
            const std::string want = Wf == 4 ? "monogenic route REFUSED" : "general route REFUSED";
            Check(ref && msg.find(want) != std::string::npos,
                  "forms data mod 37^" + std::to_string(Wf) + ": " + want + (Wf == 4 ? " (thm:u-interp(2))" : " (prop:u-lambdaprod(3))"));
        }
        std::cout << "NEGATIVE precision refusals done" << std::endl;
        if (!fw37.empty()) {
            const auto pr = PrimesUpTo(43, 37);
            const L::ZMod R(37, 10);
            // (a) a further weight (k = 464) perturbed: T2 + 37^2 I on e M_k (keeps every subspace and every commutation)
            {
                auto fw = fw37;
                for (size_t i = 0; i < fw.back().M.s; ++i)
                    fw.back().M.mats["T2"][i][i] = R.add(fw.back().M.mats["T2"][i][i], 37 * 37);
                L::ClassInput in{37, 10, pr, fw};
                const L::ClassResult r = L::RunClass(in, 32, 3, {1, 2, 3}, {}, [](const std::string&) {});
                uint32_t caught = 0;
                for (const auto& t : r.truncations) {
                    bool all_held = true;
                    for (const auto& f : t.c.further) all_held = all_held && f.held;
                    caught += !all_held;
                }
                Check(caught == r.truncations.size(), "perturbed further weight caught by the (U11) gate at every m");
                std::cout << "NEGATIVE perturbed_further_weight k=464 caught_by_U11=" << caught << "/" << r.truncations.size()
                          << std::endl;
            }
            // (b) a node (k = 68) perturbed: T2 + 37 I on e M_k
            {
                auto fw = fw37;
                for (size_t i = 0; i < fw[1].M.s; ++i) fw[1].M.mats["T2"][i][i] = R.add(fw[1].M.mats["T2"][i][i], 37);
                L::ClassInput in{37, 10, pr, fw};
                // caught either by thm:u-interp(1) (the interpolant of non-Lambda values is not
                // ell-integral: REFUSED) or by the (U13) gate
                uint32_t caught = 0, total = 0;
                std::string how = "U13";
                try {
                    const L::ClassResult r = L::RunClass(in, 32, 3, {1, 2, 3}, {}, [](const std::string&) {});
                    for (const auto& t : r.truncations) {
                        ++total;
                        caught += !(t.E.F0_zero && t.E.F1_eq_G && t.E.G_eq_alpha);
                    }
                } catch (const std::logic_error& e) {
                    how = std::string(e.what()).find("not ell-integral") != std::string::npos ? "integrality" : "other";
                    caught = total = 1;
                }
                Check(total > 0 && caught == total && how != "other", "perturbed node caught");
                std::cout << "NEGATIVE perturbed_node k=68 caught=" << caught << "/" << total << " by=" << how << std::endl;
            }
        }
    }

    // ---- (4) arm (T): the tower route as a cross-check at 37 and 103
    {
        struct TCell {
            uint32_t n;
            uint64_t ell;
            uint32_t m;
        };
        const std::vector<TCell> tcells{{30, 37, 2}, {66, 37, 2}, {22, 103, 2}};
        uint32_t held = 0;
        for (const auto& tc : tcells) {
            const double t0 = Now();
            const auto cell = hecke_ext::hida::ComputeTowerCell(tc.n, tc.ell, tc.m);
            for (uint32_t m = 1; m <= tc.m; ++m) {
                const L::FormsWeight w = L::ComputeFormsWeight(tc.ell, tc.n + 2, m, {2, 3});
                const L::ZMod R(tc.ell, m);
                const auto& rd = cell.module_read.at(m - 1);
                bool ok = rd.ord_rank == 2 * w.S.s + 1;
                for (size_t i = 0; i < rd.primes.size(); ++i) {
                    const std::string op = "T" + std::to_string(rd.primes[i]);
                    Vec got;
                    for (uint64_t x : rd.charpoly[i]) got.push_back(static_cast<u128>(x) % R.M);
                    ok = ok && got == EsPrediction(w, op, R);
                }
                Check(ok, "arm (T) (" + std::to_string(tc.n) + ", " + std::to_string(tc.ell) + ") m=" + std::to_string(m) +
                              ": tower charpolys == (Y - eis) charpoly(e S_k)^2");
                held += ok;
            }
            std::cout << "TOWER_CROSSCHECK n=" << tc.n << " ell=" << tc.ell << " k=" << tc.n + 2 << " m<=" << tc.m
                      << " ord_rank=" << cell.module_read.back().ord_rank << " wall_s=" << Now() - t0 << std::endl;
        }
        std::cout << "TOWER_CROSSCHECK cells=" << tcells.size() << " readings_held=" << held << std::endl;
    }

    // ---- (5) the hida_tower/v1 catalogue against the forms side
    {
        std::ifstream in(SourcePath("docs/matrix-engine/results/hida_towers/hida_towers.tsv"));
        Check(static_cast<bool>(in), "hida_towers.tsv readable");
        std::string line;
        std::vector<std::string> head;
        std::map<std::tuple<uint64_t, uint32_t, uint32_t>, L::FormsWeight> cache;
        uint32_t held = 0, failed = 0, skipped = 0, rows = 0;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::vector<std::string> f;
            {
                std::string cur;
                for (char c : line) {
                    if (c == '\t') {
                        f.push_back(cur);
                        cur.clear();
                    } else {
                        cur.push_back(c);
                    }
                }
                f.push_back(cur);
            }
            if (head.empty()) {
                head = f;
                continue;
            }
            ++rows;
            std::map<std::string, std::string> row;
            for (size_t i = 0; i < head.size() && i < f.size(); ++i) row[head[i]] = f[i];
            const uint64_t ell = std::stoull(row["ell"]);
            const uint32_t m = static_cast<uint32_t>(std::stoul(row["m"])), n = static_cast<uint32_t>(std::stoul(row["n"]));
            const uint32_t k = n + 2;
            if (ell == 2411 || n < 2 || k - 1 < m || (ell == 691 && !all)) {
                ++skipped;
                continue;
            }
            const auto key = std::make_tuple(ell, k, m);
            if (!cache.count(key)) {
                std::vector<uint64_t> pr;
                for (uint64_t p : {2ull, 3ull, 5ull, 7ull})
                    if (p != ell) pr.push_back(p);
                cache.emplace(key, L::ComputeFormsWeight(ell, k, m, pr));
            }
            const L::FormsWeight& w = cache.at(key);
            const L::ZMod R(ell, m);
            bool ok = std::stoul(row["ord_rank"]) == 2 * w.S.s + 1;
            for (const std::string op : {"T2", "T3", "T5", "T7", "U"}) {
                const std::string col = "charpoly_" + op;
                if (row[col].empty() || !w.S.mats.count(op)) continue;
                ok = ok && ParsePolyText(row[col], R) == EsPrediction(w, op, R);
            }
            if (ok) {
                ++held;
            } else {
                ++failed;
                std::cout << "  CATALOGUE_MISMATCH ell=" << ell << " n=" << n << " m=" << m << std::endl;
            }
        }
        Check(failed == 0 && held == (all ? 753u : 744u), "catalogue rows held");
        std::cout << "CATALOGUE_CHECK rows=" << rows << " held=" << held << " failed=" << failed << " skipped=" << skipped
                  << " (forms side vs hida_tower/v1, Eichler-Shimura)" << std::endl;
    }

    std::cout << "HeckeLambdaValidation: checks=" << Check.checks << " failing=" << Check.fails << " wall_s=" << Now() - t_start
              << " peak_rss_mib=" << PeakRssMiB() << std::endl;
    return Check.fails == 0 ? 0 : 1;
}
