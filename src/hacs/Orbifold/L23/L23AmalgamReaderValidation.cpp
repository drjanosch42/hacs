// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 22, BIN-3) L23AmalgamReaderValidation -- see L23AmalgamReaderValidation.h.
//
// Parts:
//   (A) rho selfcheck: the reader's structural rho_n(g) (Taylor shifts, signed reversals, diagonals, words) equals the dense
//       Sym^n matrix of [C] def:rhon for S, S^-1, T, T^-1, U, eps, diag(1, 5), the T_p coset deltas (p = 2, 3, 7) and three
//       words, at ell = 2 (K = 64, 200) and ell = 3 (K = 50, 120), n in {1 .. 12, 25}.
//   (B) the frozen tracked outputs (docs/implementation/slot22/bin3/oracle/*.out) still equal Bin3Oracle.inc.
//   (C) ladic2.py, bit for bit: every frozen (ell, n) row (default arm: a deterministic subset; BIN3_ORACLE_ALL=1: all 200)
//       is recomputed with m_target = cap = the prototype's K (240 / 160) and ladic2_line() must equal the row.  On every
//       such cell the census gates (census_l23_plan.md Sec. 2): rank_lambda1 = 2 dim M_{n+2} - 1; t1 = h0 ([C]
//       lem:bocksteinallell); ed_tell has exactly one unit; ed_eis_p exactly one ">= cap" (the Eisenstein line); ed_tp has no
//       unit; c_p = min(v_ell(p^n - 1), min ed_eis_p) ([C] lem:cprule); the Eisenstein floors 7 / 9 / 8 (ell = 2) and
//       1 / 1 / 3 (ell = 3); the T_p commute; ell = 3: iota^2 = 1, iota commutes with every T_p, dim Lambda^+ = dim S + 1,
//       dim Lambda^- = dim S, the Eisenstein ">= cap" on Lambda^+ (the [C] def:iota sign), ed^+ union ed^- = ed.
//       Audit R1: on the same cells the divisors satisfy the level-1 modular-forms sum rules (v_ell det(T_p - c | S_k), frozen in
//       Bin3SumRules.inc by gen_bin3_sumrules.py; Hodge <= Newton frozen too) -- an oracle outside the amalgam model.
//   (D) oddn_l2.py, bit for bit: odd n = 1 .. 101 (K = 4), and the closed forms: h0 = floor((floor(n/3) + 1) / 2)
//       ([C] prop:oddnltwo(2)), dim H^1(V) = h0, dim H^2(V) = (n + 1)/2 - h0, h1 = floor(n/2) + 1, exponent 2
//       (lem:centralminusone), log_2 |H^1(V/16)| = dim H^1 + dim H^2 (lem:oddnsplit), T_p = 0 on H^1(V/16) and H^1(M)
//       (prop:oddnhecke(1), prop:ltwothreescalar), T_2 = 0 on red_1 H^1(V) (prop:oddnhecke(2)), rank T_2 <= 2
//       (lem:tellrank); the CONJ prop:oddnhecke(4) (T_2 a rank-one idempotent on H^1(M)) is counted as a finding, not gated.
//       The lattice reader at odd n <= 41: rank 0, torsion = h0 copies of 1 (prop:oddnltwo(3)).
//   (E) [C] prop:ltwothreescalar: T_p - (1 + p) induces 0 on H^1(Gamma; Sym^n F_ell^2), ell in {2, 3}, every p in
//       {2, 3, 5, 7} \ {ell}, n = 1 .. 40; lem:tellrank: rank T_2 <= 2 (ell = 2), rank T_3 <= 2 (ell = 3, n even);
//       lem:oddnsplit at k = 2, 3: log_2 |H^1(V/2^k)| = (n + 1)/2 and T_p = 0 there, odd n <= 21.
//   (F) the adaptive rule: (ell, n) = (2, 64) from k_min = 16 with m_target = 64 climbs the ladder R3 -> R4 (logged) and agrees with
//       a direct run; (3, 40) from k_min = 2 with m_target = 78 climbs digits and agrees.
//   (G) negative controls: the bar formula without its last coset representative (mutate 1) is caught by the oracle
//       comparison (ell = 2, 3), by T_p = 1 + p on H^1(M) and by the oddn row; the reader without its precision guard
//       (mutate 2) fails its own checks; a tampered oracle row is caught.
// Last line: BIN3_READER ... checks= failing= verdict=OK.
#include "L23AmalgamReaderValidation.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "AmalgamReader.h"

namespace {

struct Ladic2Row { uint32_t ell, n, K, comm_v; const char* line; };
const Ladic2Row kLadic2[] = {
#define BIN3_LADIC2_ROWS
#include "Bin3Oracle.inc"
#undef BIN3_LADIC2_ROWS
};
// BIN-3 audit R1: level-1 modular-forms determinant valuations (Bin3SumRules.inc, frozen by gen_bin3_sumrules.py), independent of the amalgam
struct SumRuleRow { uint32_t ell, n; int vTl, vTl2, hodge; int vP[4], vE[4], vN[4]; };
const SumRuleRow kSumRules[] = {
#include "Bin3SumRules.inc"
};
struct OddnOracleRow { uint32_t n; uint32_t v[22]; };
const OddnOracleRow kOddn[] = {
#define BIN3_ODDN_ROWS
#include "Bin3Oracle.inc"
#undef BIN3_ODDN_ROWS
};

uint64_t g_checks = 0, g_fail = 0;
bool check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        std::cout << "  [FAIL] " << what << std::endl;
    }
    return ok;
}
std::string str(uint64_t x) { return std::to_string(x); }

std::string source_path(const std::string& rel) {
#ifdef OC_SOURCE_DIR
    return std::string(OC_SOURCE_DIR) + "/" + rel;
#else
    return rel;
#endif
}

// v_ell(p^n - 1) (exact below 39 / 63).
uint32_t v_pn_minus_1(uint32_t ell, uint32_t p, uint32_t n) {
    if (ell == 2) {
        uint64_t x = 1;
        for (uint32_t i = 0; i < n; ++i) x *= p;
        x -= 1;
        return x == 0 ? 64u : static_cast<uint32_t>(__builtin_ctzll(x));
    }
    unsigned __int128 M = 1;
    for (int i = 0; i < 39; ++i) M *= 3;
    unsigned __int128 x = 1;
    for (uint32_t i = 0; i < n; ++i) x = x * p % M;
    x = (x + M - 1) % M;
    if (x == 0) return 39;
    uint32_t v = 0;
    while (x % 3 == 0) { x /= 3; ++v; }
    return v;
}

uint32_t min_divisor(const l23::Divisors& D) {   // nu(Y): the least exponent (cap if only ">= cap")
    return D.finite.empty() ? D.cap : D.finite.front();
}
std::vector<uint32_t> merged(const l23::Divisors& a, const l23::Divisors& b) {
    std::vector<uint32_t> v = a.finite;
    v.insert(v.end(), b.finite.begin(), b.finite.end());
    std::sort(v.begin(), v.end());
    return v;
}

// ---------------------------------------------------------------------------
// (A)
// ---------------------------------------------------------------------------
void part_rho() {
    uint32_t cases = 0, bad = 0;
    for (uint32_t ell : {2u, 3u})
        for (uint32_t k : (ell == 2 ? std::vector<uint32_t>{64, 200} : std::vector<uint32_t>{50, 120}))
            for (uint32_t n : {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 25u}) {
                bad += l23::rho_selfcheck(ell, n, k);
                ++cases;
            }
    check(bad == 0, "(A) structural rho_n = dense Sym^n ([C] def:rhon): " + str(cases) + " (ell, k, n) blocks, " + str(bad) +
                        " mismatching entries");
    std::cout << "  (A) rho selfcheck: " << cases << " blocks, mismatches " << bad << std::endl;
}

// ---------------------------------------------------------------------------
// (B)
// ---------------------------------------------------------------------------
void part_sources() {
    uint32_t rows = 0, agree = 0;
    const struct { const char* f; uint32_t ell; } L[] = {{"ladic2_l2_K240.out", 2}, {"ladic2_l3_K160.out", 3}};
    for (const auto& s : L) {
        std::ifstream in(source_path(std::string("docs/implementation/slot22/bin3/oracle/") + s.f));
        if (!check(static_cast<bool>(in), std::string("(B) tracked oracle source readable: ") + s.f)) continue;
        std::string line;
        while (std::getline(in, line)) {
            if (line.rfind("ell=", 0) != 0) continue;
            ++rows;
            const size_t c = line.find(" comm_v>=");
            const std::string body = line.substr(0, c);
            for (const Ladic2Row& r : kLadic2)
                if (r.ell == s.ell && body == r.line) { ++agree; break; }
        }
    }
    std::ifstream in(source_path("docs/implementation/slot22/bin3/oracle/oddn_l2_K4.out"));
    std::string line;
    uint32_t orows = 0, oagree = 0;
    while (in && std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream is(line);
        uint32_t n;
        is >> n;
        std::vector<uint32_t> v(22);
        for (auto& x : v) is >> x;
        ++orows;
        for (const OddnOracleRow& r : kOddn)
            if (r.n == n && std::equal(v.begin(), v.end(), r.v)) { ++oagree; break; }
    }
    const uint32_t nl = sizeof(kLadic2) / sizeof(kLadic2[0]), no = sizeof(kOddn) / sizeof(kOddn[0]);
    check(rows == nl && agree == nl && orows == no && oagree == no,
          "(B) the tracked .out files equal Bin3Oracle.inc (" + str(agree) + "/" + str(nl) + " ladic2, " + str(oagree) + "/" + str(no) +
              " oddn rows)");
    std::cout << "  (B) ORACLE sources: ladic2 " << agree << "/" << nl << ", oddn " << oagree << "/" << no << std::endl;
}

// ---------------------------------------------------------------------------
// (C)
// ---------------------------------------------------------------------------
struct Tally { uint32_t cells = 0, agree = 0, gates = 0, pm_cells = 0, pm_identical_halves = 0, pm_by_mod6[6] = {0, 0, 0, 0, 0, 0}, cells_by_mod6[6] = {0, 0, 0, 0, 0, 0}; double sec = 0, max_sec = 0; uint32_t max_n = 0; };

void gate_cell(const l23::LatticeCell& c, Tally& t) {
    const std::string tag = "(C) ell=" + str(c.ell) + " n=" + str(c.n) + ": ";
    bool ok = true;
    ok &= check(c.rank_lambda == l23::expected_rank_lambda(c.n), tag + "rank_lambda1 = 2 dim M_{n+2} - 1");
    ok &= check(c.torsion.size() == c.h0, tag + "t1 = h0 (" + str(c.torsion.size()) + " vs " + str(c.h0) + ")");
    ok &= check(c.ed_tell.units() == 1, tag + "ed_tell has exactly one unit");
    ok &= check(c.commute, tag + "the T_p commute");
    const std::map<uint32_t, uint32_t> floor2 = {{3, 7}, {5, 9}, {7, 8}}, floor3 = {{2, 1}, {5, 1}, {7, 3}};
    for (const auto& kv : c.ed_eis) {
        const uint32_t p = kv.first;
        const l23::Divisors& E = kv.second;
        const l23::Divisors& Tp = c.ed_tp.at(p);
        ok &= check(E.at_least == 1, tag + "ed_eis_" + str(p) + " has exactly one >= cap (the Eisenstein line)");
        ok &= check(Tp.units() == 0, tag + "ed_tp_" + str(p) + ": min >= 1");
        const uint32_t cp = std::min(min_divisor(E), v_pn_minus_1(c.ell, p, c.n));
        ok &= check(min_divisor(Tp) == cp, tag + "c_p rule ([C] lem:cprule) at p = " + str(p) + ": min ed_tp = " + str(min_divisor(Tp)) +
                                               ", min(v(p^n - 1), min ed_eis) = " + str(cp));
        const uint32_t fl = (c.ell == 2 ? floor2 : floor3).at(p);
        ok &= check(E.finite.empty() || E.finite.front() >= fl, tag + "Eisenstein floor at p = " + str(p));
    }
    if (c.ell == 3 && c.pm_done) {
        ++t.pm_cells;
        const uint32_t S = (c.rank_lambda - 1) / 2;
        ok &= check(c.iota_ok, tag + "iota^2 = 1, iota T_p = T_p iota");
        ok &= check(c.dim_plus == S + 1 && c.dim_minus == S, tag + "dim Lambda^+ = dim S + 1, dim Lambda^- = dim S (got " +
                                                                 str(c.dim_plus) + ", " + str(c.dim_minus) + ")");
        for (const auto& kv : c.ed_eis) {
            const uint32_t p = kv.first;
            ok &= check(c.ed_eis_plus.at(p).at_least == 1 && c.ed_eis_minus.at(p).at_least == 0,
                        tag + "the Eisenstein line lies in Lambda^+ ([C] def:iota sign) at p = " + str(p));
            ok &= check(merged(c.ed_eis_plus.at(p), c.ed_eis_minus.at(p)) == kv.second.finite &&
                            merged(c.ed_tp_plus.at(p), c.ed_tp_minus.at(p)) == c.ed_tp.at(p).finite,
                        tag + "ed^+ union ed^- = ed (T_p - (1+p), T_p - (1+p^{n+1})) at p = " + str(p));
        }
        ok &= check(merged(c.ed_tell_plus, c.ed_tell_minus) == c.ed_tell.finite &&
                        c.ed_tell_plus.at_least + c.ed_tell_minus.at_least == c.ed_tell.at_least,
                    tag + "ed_tell^+ union ed_tell^- = ed_tell");
        // finding (G-conj, census_l23_plan.md / master plan Sec. 1.1): the cusp halves of ed_tell identical?
        std::vector<uint32_t> plus = c.ed_tell_plus.finite;
        auto u = std::find(plus.begin(), plus.end(), 0u);
        if (u != plus.end()) plus.erase(u);
        ++t.cells_by_mod6[c.n % 6];
        if (plus == c.ed_tell_minus.finite) { ++t.pm_identical_halves; ++t.pm_by_mod6[c.n % 6]; }
    }
    if (ok) ++t.gates;
}


// "3,3,11" / "-" / "7,9+[>=240]^1" -> (sum of the finite part, number of ">= cap" entries, finite count)
bool parse_div(const std::string& s, uint64_t& sum, uint32_t& al) {
    sum = 0;
    al = 0;
    std::string body = s;
    const size_t plus = s.find("+[>=");
    if (plus != std::string::npos) {
        body = s.substr(0, plus);
        const size_t hat = s.find("]^", plus);
        if (hat == std::string::npos) return false;
        al = static_cast<uint32_t>(std::stoul(s.substr(hat + 2)));
    }
    if (body == "-") return true;
    std::stringstream ss(body);
    std::string tok;
    while (std::getline(ss, tok, ',')) sum += std::stoull(tok);
    return true;
}

// the modular-forms sum rules (BIN3_audit.md Sec. 0) on one ladic2 line
void sumrules_cell(const Ladic2Row& r, const std::string& line) {
    const SumRuleRow* q = nullptr;
    for (const SumRuleRow& x : kSumRules) if (x.ell == r.ell && x.n == r.n) q = &x;
    const std::string tag = "(C) sumrules ell=" + str(r.ell) + " n=" + str(r.n);
    if (!check(q != nullptr && q->hodge == 1, tag + ": frozen table row present, Hodge <= Newton")) return;
    std::map<std::string, std::string> col;
    std::stringstream ss(line);
    std::string tok;
    while (ss >> tok) {
        const size_t c = tok.find(':');
        if (c != std::string::npos) col[tok.substr(0, c)] = tok.substr(c + 1);
    }
    uint64_t sum;
    uint32_t al;
    const uint32_t ps[4] = {2, 3, 5, 7};
    for (int i = 0; i < 4; ++i) {
        const uint32_t p = ps[i];
        if (p == r.ell) continue;
        check(parse_div(col["p" + str(p)], sum, al) && sum == 2u * q->vP[i] + q->vN[i] && al == 0,
              tag + ": sum ed(T_" + str(p) + " - (1+p)) = 2 v det + v(p^n - 1)");
        check(parse_div(col["Eis" + str(p)], sum, al) && sum == 2u * q->vE[i] && al == 1,
              tag + ": sum ed_eis_" + str(p) + " = 2 v det(T_p - (1 + p^(n+1))), one >= K");
    }
    check(parse_div(col["Tl"], sum, al) && sum == 2u * q->vTl && al == 0, tag + ": sum ed(T_ell) = 2 v det T_ell");
    if (q->vTl2 >= 0) {
        const bool cap = r.n + 1 >= r.K;
        check(parse_div(col["Tl2-Tl"], sum, al) && sum == 2u * q->vTl2 + (cap ? 0 : r.n + 1) && al == (cap ? 1u : 0u),
              tag + ": sum ed(T_ell^2 - T_ell) = 2 v det + (n + 1)");
    }
}

void part_ladic2(bool all, Tally& t2, Tally& t3) {
    for (const Ladic2Row& r : kLadic2) {
        // default arm: every n <= 40, every 16th n beyond (n = 56, 72, ...), and the last cell (n = 200)
        if (!all && !(r.n <= 40 || r.n % 16 == 8 || r.n == 200)) continue;
        if (!all && r.ell == 3 && r.n > 40 && r.n != 200 && r.n % 48 != 8) continue;
        l23::ReaderOptions o;
        o.ell = r.ell;
        o.m_target = r.K;
        o.cap = r.K;
        const l23::LatticeCell c = l23::read_lattice(r.n, o);
        Tally& t = r.ell == 2 ? t2 : t3;
        ++t.cells;
        t.sec += c.seconds;
        if (c.seconds > t.max_sec) { t.max_sec = c.seconds; t.max_n = r.n; }
        const std::string tag = "(C) ell=" + str(r.ell) + " n=" + str(r.n);
        if (!check(c.ok, tag + ": reader ok (" + (c.failures.empty() ? std::string("-") : c.failures.front()) + ")")) {
            for (const std::string& l : c.log) std::cout << "    " << l << std::endl;
            continue;
        }
        const std::string got = c.ladic2_line();
        if (check(got == r.line, tag + ": ladic2.py line bit for bit")) ++t.agree;
        else std::cout << "    want " << r.line << "\n    got  " << got << std::endl;
        sumrules_cell(r, got);
        gate_cell(c, t);
        if (r.n == 200 || r.n == 2) std::cout << "    " << c.log.back() << "  (" << c.seconds << " s)" << std::endl;
    }
}

// ---------------------------------------------------------------------------
// (D)
// ---------------------------------------------------------------------------
void part_oddn(uint32_t& conj_ok, uint32_t& agree) {
    for (const OddnOracleRow& r : kOddn) {
        const l23::OddnRow row = l23::oddn_row(r.n, 4);
        const std::string tag = "(D) n=" + str(r.n);
        if (check(row.v.size() == 22 && std::equal(row.v.begin(), row.v.end(), r.v), tag + ": oddn_l2.py row bit for bit")) ++agree;
        else std::cout << "    got  " << row.line() << std::endl;
        const auto& v = row.v;
        const uint32_t n = r.n, h0 = ((n / 3) + 1) / 2;
        check(v[0] == h0 && v[1] == h0, tag + ": h0 = floor((floor(n/3)+1)/2) = the Dickson count");
        check(v[2] == n / 2 + 1, tag + ": h1 = floor(n/2) + 1");
        check(v[4] == h0 && v[5] == (n + 1) / 2 - h0, tag + ": dim H^1(V) = h0, dim H^2(V) = (n+1)/2 - h0 ([C] prop:oddnltwo(3))");
        check(v[6] == 1, tag + ": 2 H^1(V/16) = 0 ([C] lem:centralminusone)");
        check(v[7] == v[4] + v[5], tag + ": log_2 |H^1(V/16)| = dim H^1 + dim H^2 ([C] lem:oddnsplit)");
        check(v[8] == 0 && v[9] == 0 && v[10] == 0, tag + ": T_3, T_5, T_7 = 0 on H^1(V/16) ([C] prop:oddnhecke(1))");
        check(v[11] == 0 && v[12] == 0 && v[13] == 0, tag + ": T_3, T_5, T_7 = 0 on H^1(M) ([C] prop:ltwothreescalar)");
        check(v[14] <= 2, tag + ": rank T_2 <= 2 on H^1(M) ([C] lem:tellrank)");
        check(v[18] == 0, tag + ": T_2 = 0 on red_1 H^1(V) ([C] prop:oddnhecke(2))");
        if (v[14] == 1 && v[17] == 1) ++conj_ok;   // CONJ prop:oddnhecke(4) (finding)
    }
    // the lattice reader at odd n: rank 0, torsion = h0 copies of 1
    uint32_t bad = 0, cells = 0;
    for (uint32_t n = 1; n <= 41; n += 2) {
        l23::ReaderOptions o;
        o.ell = 2;
        const l23::LatticeCell c = l23::read_lattice(n, o);
        const uint32_t h0 = ((n / 3) + 1) / 2;
        ++cells;
        if (!c.ok || c.rank_lambda != 0 || c.torsion != std::vector<uint32_t>(h0, 1u) || c.h0 != h0) ++bad;
    }
    check(bad == 0, "(D) lattice reader, odd n <= 41: rank 0, H^1(V) = F_2^{h0} ([C] prop:oddnltwo(3)): " + str(cells) + " cells, " +
                        str(bad) + " wrong");
}

// ---------------------------------------------------------------------------
// (E)
// ---------------------------------------------------------------------------
void part_scalar() {
    uint32_t cells = 0, bad_scalar = 0, bad_rank = 0, bad_split = 0;
    for (uint32_t ell : {2u, 3u})
        for (uint32_t n = 1; n <= 40; ++n) {
            std::vector<std::pair<uint32_t, int64_t>> ops;
            for (uint32_t p : {2u, 3u, 5u, 7u}) if (p != ell) ops.push_back({p, 1 + p});
            ops.push_back({ell, 0});
            const l23::ModCell m = l23::h1_mod(ell, n, 1, ops);
            ++cells;
            for (size_t i = 0; i + 1 < ops.size(); ++i) if (m.image_length[i] != 0) ++bad_scalar;
            if ((ell == 2 || n % 2 == 0) && m.image_length.back() > 2) ++bad_rank;
        }
    check(bad_scalar == 0, "(E) T_p - (1 + p) = 0 on H^1(Gamma; Sym^n F_ell^2), ell = 2, 3, p in {2,3,5,7} \\ {ell}, n <= 40 "
                           "([C] prop:ltwothreescalar): " + str(bad_scalar) + " non-zero");
    check(bad_rank == 0, "(E) rank T_ell <= 2 on H^1(M) ([C] lem:tellrank): " + str(bad_rank) + " violations");
    for (uint32_t n = 1; n <= 21; n += 2)
        for (uint32_t k : {2u, 3u}) {
            const l23::ModCell m = l23::h1_mod(2, n, k, {{3, 0}, {5, 0}, {7, 0}});
            if (m.length != (n + 1) / 2 || m.image_length[0] || m.image_length[1] || m.image_length[2]) ++bad_split;
        }
    check(bad_split == 0, "(E) odd n <= 21, k = 2, 3: log_2 |H^1(V/2^k)| = (n+1)/2 and T_3, T_5, T_7 = 0 ([C] lem:oddnsplit, "
                          "prop:oddnhecke(1)): " + str(bad_split) + " wrong");
    std::cout << "  (E) TP_SCALAR cells=" << cells << " nonzero=" << bad_scalar << " tell_rank_violations=" << bad_rank
              << " oddn_split_wrong=" << bad_split << std::endl;
}

// ---------------------------------------------------------------------------
// (F)
// ---------------------------------------------------------------------------
void part_adaptive() {
    {
        l23::ReaderOptions o;
        o.ell = 2;
        o.k_min = 16;
        o.m_target = 64;
        const l23::LatticeCell a = l23::read_lattice(64, o);
        for (const std::string& l : a.log) std::cout << "    " << l << std::endl;
        l23::ReaderOptions od;
        od.ell = 2;
        od.k_min = 128;
        od.cap = 40;
        const l23::LatticeCell b = l23::read_lattice(64, od);
        l23::ReaderOptions oa = o;
        oa.cap = 40;
        const l23::LatticeCell a2 = l23::read_lattice(64, oa);
        check(a.ok && a.reruns >= 1 && a.m >= 64 && a2.ok && b.ok && a2.ladic2_line() == b.ladic2_line(),
              "(F) ell = 2, n = 64: from K = 64 (R3) the cell climbs the ladder to m >= 64 (" + str(a.reruns) + " reruns, " + a.rung +
                  ") and agrees with a direct R4 run at cap 40");
    }
    {
        l23::ReaderOptions o;
        o.ell = 3;
        o.k_min = 2;
        o.m_target = 78;
        o.cap = 60;
        const l23::LatticeCell a = l23::read_lattice(40, o);
        for (const std::string& l : a.log) std::cout << "    " << l << std::endl;
        l23::ReaderOptions od = o;
        od.k_min = 156;
        const l23::LatticeCell b = l23::read_lattice(40, od);
        check(a.ok && a.reruns >= 1 && b.ok && a.ladic2_line() == b.ladic2_line(),
              "(F) ell = 3, n = 40: from K = 78 (2 digits) the cell climbs digit by digit to m >= 78 (" + str(a.reruns) +
                  " reruns) and agrees with a direct 4-digit run");
    }
}

// ---------------------------------------------------------------------------
// (G)
// ---------------------------------------------------------------------------
void part_negative() {
    uint32_t caught = 0, total = 0;
    auto row_for = [](uint32_t ell, uint32_t n) -> const Ladic2Row* {
        for (const Ladic2Row& r : kLadic2) if (r.ell == ell && r.n == n) return &r;
        return nullptr;
    };
    for (const auto& en : std::vector<std::pair<uint32_t, uint32_t>>{{2, 26}, {3, 34}}) {
        const Ladic2Row* r = row_for(en.first, en.second);
        l23::ReaderOptions o;
        o.ell = r->ell;
        o.m_target = r->K;
        o.cap = r->K;
        o.mutate = 1;
        const l23::LatticeCell c = l23::read_lattice(r->n, o);
        ++total;
        if (!c.ok || c.ladic2_line() != r->line) ++caught;
    }
    {   // mutate 1 on T_p = 1 + p over F_ell and on the oddn row
        ++total;
        const l23::ModCell m = l23::h1_mod(2, 10, 1, {{3, 4}}, 1);
        if (m.image_length[0] != 0) ++caught;
        ++total;
        const l23::ModCell m3 = l23::h1_mod(3, 10, 1, {{2, 3}}, 1);
        if (m3.image_length[0] != 0) ++caught;
        ++total;
        const l23::OddnRow row = l23::oddn_row(9, 4, 1);
        const OddnOracleRow* want = nullptr;
        for (const OddnOracleRow& q : kOddn) if (q.n == 9) want = &q;
        if (!std::equal(row.v.begin(), row.v.end(), want->v)) ++caught;
    }
    for (const auto& en : std::vector<std::pair<uint32_t, uint32_t>>{{2, 26}, {3, 34}}) {   // mutate 2: no precision guard
        l23::ReaderOptions o;
        o.ell = en.first;
        o.m_target = 0;
        o.k_cap = en.first == 2 ? 128 : 117;
        o.mutate = 2;
        const l23::LatticeCell c = l23::read_lattice(en.second, o);
        ++total;
        if (!c.ok) ++caught;
    }
    {   // a tampered oracle row
        const Ladic2Row* r = row_for(2, 10);
        l23::ReaderOptions o;
        o.ell = 2;
        o.m_target = r->K;
        o.cap = r->K;
        const l23::LatticeCell c = l23::read_lattice(10, o);
        std::string t = r->line;
        const size_t pos = t.find("Tl:3,3");
        if (pos != std::string::npos) t.replace(pos, 6, "Tl:3,4");
        ++total;
        if (c.ladic2_line() != t) ++caught;
    }
    std::cout << "  (G) NEGATIVE_CONTROL caught=" << caught << "/" << total
              << " (drop r_inf vs ladic2 at ell = 2, 3; vs T_p = 1 + p over F_2, F_3; vs oddn; no precision guard at ell = 2, 3; "
                 "tampered row)" << std::endl;
    check(caught == total, "(G) negative controls caught " + str(caught) + "/" + str(total));
}

} // namespace

uint32_t L23AmalgamReaderValidation() {
    std::cout << "=== L23AmalgamReaderValidation (BIN-3: the degree-1 amalgam reader on the Z/2^k ladder and Z/3^k vs ladic2.py / "
                 "oddn_l2.py) ===" << std::endl;
    const auto t0 = std::chrono::steady_clock::now();
    g_checks = g_fail = 0;
    const char* env = std::getenv("BIN3_ORACLE_ALL");
    const bool all = env && std::string(env) == "1";
    uint32_t armed = 0;
    for (const Ladic2Row& r : kLadic2) {
        const bool pick = all || ((r.n <= 40 || r.n % 16 == 8 || r.n == 200) &&
                                  !(r.ell == 3 && r.n > 40 && r.n != 200 && r.n % 48 != 8));
        if (pick) ++armed;
    }
    std::cout << "EFFECTIVE L23AmalgamReaderValidation arm=" << (all ? "BIN3_ORACLE_ALL" : "default") << " cells=" << armed
              << " (of " << sizeof(kLadic2) / sizeof(kLadic2[0]) << " ladic2 rows; oddn rows " << sizeof(kOddn) / sizeof(kOddn[0])
              << ")" << std::endl;
    try {
        part_rho();
        part_sources();
        Tally t2, t3;
        part_ladic2(all, t2, t3);
        std::cout << "  (C) LADIC2 ell=2 cells=" << t2.cells << " agree=" << t2.agree << " gates_ok=" << t2.gates << " sum_s=" << t2.sec
                  << " max_s=" << t2.max_sec << " (n=" << t2.max_n << ")" << std::endl;
        std::cout << "  (C) LADIC2 ell=3 cells=" << t3.cells << " agree=" << t3.agree << " gates_ok=" << t3.gates << " pm_cells="
                  << t3.pm_cells << " pm_identical_tell_halves=" << t3.pm_identical_halves << " (G-conj finding, not gated) sum_s="
                  << t3.sec << " max_s=" << t3.max_sec << " (n=" << t3.max_n << ")" << std::endl;
        std::cout << "  (C) PM_TELL_HALVES identical/cells by n mod 6: 0:" << t3.pm_by_mod6[0] << "/" << t3.cells_by_mod6[0] << " 2:"
                  << t3.pm_by_mod6[2] << "/" << t3.cells_by_mod6[2] << " 4:" << t3.pm_by_mod6[4] << "/" << t3.cells_by_mod6[4]
                  << " (the master plan's G-conj law: identical for n = 4 mod 6, different for n = 2 mod 6; finding only)" << std::endl;
        uint32_t conj_ok = 0, oagree = 0;
        part_oddn(conj_ok, oagree);
        std::cout << "  (D) ODDN cells=" << sizeof(kOddn) / sizeof(kOddn[0]) << " agree=" << oagree
                  << " conj_T2_rank_one_idempotent=" << conj_ok << " (CONJ prop:oddnhecke(4): finding, not gated)" << std::endl;
        part_scalar();
        part_adaptive();
        part_negative();
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::cout << "BIN3_READER ladic2_cells=" << (t2.agree + t3.agree) << "/" << (t2.cells + t3.cells) << " oddn_cells=" << oagree << "/"
                  << sizeof(kOddn) / sizeof(kOddn[0]) << " checks=" << g_checks << " failing=" << g_fail << " s=" << s
                  << " verdict=" << (g_fail ? "FAIL" : "OK") << std::endl;
    } catch (const std::exception& e) {
        std::cout << "  [EXCEPTION] " << e.what() << std::endl;
        return 1;
    }
    return g_fail ? 1 : 0;
}
