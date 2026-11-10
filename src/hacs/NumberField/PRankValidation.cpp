// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// NumberFieldPRankValidation (standard) -- the p-rank of Cl(K), K = Q(N^{1/p}), p = 5, by a
// Minkowski-complete relation computation (hecke-extensions slot 22, NF-PRANK;
// docs/implementation/slot22/NF-PRANK_spec.md sec. 8; predictions committed first in
// docs/implementation/slot22/NF-PRANK_predictions.md). Oracle reproduced (R-22b): the frozen Python
// prototype docs/implementation/slot22/nfprank/ (pure_quintic_rank.py, pure_quintic_hR.py ->
// hR_runs.out; chevalley.py -> chevalley_cells.tsv).
//
//   G1  field data (d_K, signature, Minkowski bound, Dedekind) at (11,5) (31,5) (41,5) (61,5) equal to
//       the prototype; Round 2 at (401,5) (25 | N - 1): index 5, d_K = 5^3 401^4, the P_1 P_2^4 shape;
//       primitives (F_q factorisation, 64-bit factoring) on fixed inputs;
//   G2  Chevalley data (t, lambda ramified, r*, exponent, non-norm characters) on all 1240 cells of
//       chevalley_cells.tsv (sha256 checked), 0 differences;
//   G3  the class-group bounds at the four prototype cells, field by field (factor base, degree >= 2
//       ideals, Z^S/Rel, rank_upper, rank_lower / genus, 5 not dividing [E:E'], R', Res, hR, hR/R' within
//       1 %, level); the descent path at (61,5) with B_0 = 3000 gives the same group; the p^2 | N - 1 path
//       at (101,5) (internal consistency only);
//   G4  negative controls: (a) an ideal removed, (b) a corrupted valuation, (c) Z[alpha] forced at
//       (401,5), (d) irregular p = 37, 59, 67, (e) the Euler product cut at X = 500.
// Read-only hooks (EFFECTIVE line): NF_PRANK_CE=CE1|CE2|CE1C|CE2C|all (the Calegari--Emerton set of
// spec sec. 8; CE1 / CE2 predict rank_upper >= 2, a rank_upper <= 1 there is a certified
// counterexample to CE Lemma 5.13 and is reported as CE_FINDING, never "fixed"), NF_PRANK_CELLS="N:p,..."
// (any cells), NF_PRANK_DENSE=<B_0> (default 20000 for the hooks). No golden object, writes nothing.
#include "PRankValidation.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "Ambiguous.h"
#include "BigInt.h"
#include "ClassGroupRel.h"
#include "PureField.h"
#include "Sha256.h"

using namespace nf;

namespace {

uint32_t g_fail = 0, g_checks = 0;

void Check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        std::cout << "  [FAIL] " << what << std::endl;
    }
}

double Now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

std::string BoundStr(const std::vector<BigInt>& b) {
    std::string s = "[";
    for (std::size_t i = 0; i < b.size(); ++i) s += (i ? ", " : "") + b[i].to_string();
    return s + "]";
}

std::string Deg2Str(const std::vector<std::pair<uint32_t, unsigned>>& d) {
    std::string s = "[";
    for (std::size_t i = 0; i < d.size(); ++i) s += (i ? ", (" : "(") + std::to_string(d[i].first) + ", " + std::to_string(d[i].second) + ")";
    return s + "]";
}

std::string Fx(double x, int prec) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(prec) << x;
    return os.str();
}

void PrintResult(const PRankResult& R) {
    std::cout << "NF_PRANK_CELL N=" << R.N << " p=" << R.p << " dK=" << R.dK.to_string() << " index=" << R.index.to_string()
              << " sig=(" << R.r1 << "," << R.r2 << ") minkowski=" << Fx(static_cast<double>(R.minkowski), 1) << " S=" << R.factor_base
              << " S0=" << R.dense_ideals << " B0=" << R.dense_bound << " sieveB=" << R.sieve_B_used << " candidates=" << R.candidates
              << " rel2=" << R.relations_2term << " rel3=" << R.relations_3term << " relq=" << R.relations_q << " relations=" << R.relations
              << " descended=" << R.descended << " (by_q=" << R.descended_by_q << ") descent_failed=" << R.descent_failed << " extra_rows=" << R.extra_rows
              << " group_bound=" << BoundStr(R.group_bound) << " rank_upper=" << R.rank_upper << " rank_lower=" << R.rank_lower
              << " genus_ok=" << R.genus_ok << " minkowski_complete=" << R.minkowski_complete << " kernel=" << R.kernel_vectors
              << " rejected_numerics=" << R.rejected_numerics << " R'=" << Fx(R.regulator_found, 6) << " index_rank=" << R.index_rank
              << " p_coprime_unit_index=" << R.p_coprime_unit_index << " X=" << R.euler_X << " Res=" << Fx(R.euler_res, 5)
              << " hR=" << Fx(R.hR_estimate, 4) << " hR/R'=" << Fx(R.h_times_index, 3) << " level=" << prank_level_name(R.level)
              << (R.refusal.empty() ? "" : " refusal=\"" + R.refusal + "\"") << (R.free_columns.empty() ? "" : " free_columns=" + R.free_columns) << " secs=" << Fx(R.secs, 2) << std::endl;
}

void Agree(uint32_t N, const std::string& field, const std::string& proto, const std::string& cpp, bool ok) {
    std::cout << "NF_PRANK_AGREE N=" << N << " p=5 field=" << field << " proto=" << proto << " cpp=" << cpp << " " << (ok ? "ok" : "DIFF") << std::endl;
    Check(ok, "(" + std::to_string(N) + ",5) " + field + ": prototype " + proto + " vs C++ " + cpp);
}

std::string SourcePath(const std::string& rel) {
#ifdef OC_SOURCE_DIR
    return std::string(OC_SOURCE_DIR) + "/" + rel;
#else
    return rel;
#endif
}

// ---------------------------------------------------------------- G1
void G1() {
    std::cout << "-- G1 field data" << std::endl;
    struct F {
        uint32_t N;
        const char* dK;
        double M;
    };
    const F cells[] = {{11, "45753125", 421.1}, {31, "2886003125", 3344.3}, {41, "8830503125", 5849.8}, {61, "43268253125", 12949.0}};
    for (const F& c : cells) {
        const PureFieldData K = pure_field(c.N, 5);
        Agree(c.N, "d_K", c.dK, K.dK.to_string(), K.dK.to_string() == c.dK);
        Agree(c.N, "signature", "(1,2)", "(" + std::to_string(K.r1) + "," + std::to_string(K.r2) + ")", K.r1 == 1 && K.r2 == 2);
        Agree(c.N, "minkowski", Fx(c.M, 1), Fx(static_cast<double>(K.minkowski), 1), std::fabs(static_cast<double>(K.minkowski) - c.M) < 0.06);
        Agree(c.N, "order_maximal", "1", std::to_string(K.dedekind_p && K.dedekind_N && K.order_maximal && K.index == BigInt(1)),
              K.dedekind_p && K.dedekind_N && K.order_maximal && K.index == BigInt(1));
        Check(K.disc_f == K.dK, "disc(x^5 - N) = d_K at (" + std::to_string(c.N) + ",5)");
    }
    // Round 2 where Dedekind passes changes nothing
    {
        const PureFieldData K = pure_field(31, 5);
        Check(round2_at(K.f, 5).index == BigInt(1) && round2_at(K.f, 31).index == BigInt(1), "Round 2 at 5 and 31 for (31,5): index 1");
    }
    // (401,5): 25 | N - 1
    {
        const PureFieldData K = pure_field(401, 5);
        const BigInt want = BigInt(125) * BigInt::pow(BigInt(401), 4);
        std::cout << "NF_PRANK_ROUND2 N=401 p=5 dedekind_5=" << K.dedekind_p << " dedekind_401=" << K.dedekind_N << " index=" << K.index.to_string()
                  << " iterations=" << K.round2_iterations << " dK=" << K.dK.to_string() << " predicted=" << want.to_string()
                  << " minkowski=" << Fx(static_cast<double>(K.minkowski), 1) << " v5(dK)=" << K.dK.valuation(5) << std::endl;
        Check(!K.dedekind_p && K.dedekind_N, "(401,5): Dedekind fails at 5 only");
        Check(K.index == BigInt(5), "(401,5): [O_K : Z[alpha]] = 5 (Round 2)");
        Check(K.dK == want, "(401,5): d_K = 5^3 * 401^4 (S8 confirmed by Round 2)");
        Check(K.order_maximal && K.minkowski > 1.1e5 && K.minkowski < 1.15e5, "(401,5): M_K ~ 1.12e5");
        const FactorBase FB = pure_factor_base(K, 10);
        std::size_t k1 = 0, k2 = 0;
        for (const PrimeIdeal& P : FB.ideals) {
            if (P.q == 5 && P.kind == 1 && P.e == 1) ++k1;
            if (P.q == 5 && P.kind == 2 && P.e == 4) ++k2;
        }
        Check(k1 == 1 && k2 == 1, "(401,5): 5 O_K = P_1 P_2^4");
        // the 5-adic root u (u^5 = 401 mod 5^Kp)
        const BigInt u5 = BigInt::pow(BigInt::from_u64(K.p_adic_root), 5) - BigInt(401);
        Check(K.p_adic_mod > 1000000 && u5.mod_u32(5) == 0 && (u5 % BigInt::from_u64(K.p_adic_mod)).is_zero(), "(401,5): u^5 = 401 mod 5^Kp");
    }
    // primitives on fixed inputs
    {
        const FqPoly f = fq_from_int({-31, 0, 0, 0, 0, 1}, 2);
        const auto facs = fq_factor_squarefree(f, 2, 5);
        Check(facs.size() == 2 && facs[0] == FqPoly({1, 1}) && facs[1] == FqPoly({1, 1, 1, 1, 1}), "x^5 - 31 = (x + 1)(x^4 + x^3 + x^2 + x + 1) mod 2");
        const auto f19 = fq_factor_squarefree(fq_from_int({-31, 0, 0, 0, 0, 1}, 19), 19, 5);
        Check(f19.size() == 3 && f19[0].size() == 2 && f19[1].size() == 3 && f19[2].size() == 3, "x^5 - 31 mod 19: one linear, two quadratic factors");
        const auto m = fq_factor_with_multiplicity(fq_from_int({-401, 0, 0, 0, 0, 1}, 5), 5);
        Check(m.size() == 1 && m[0].first == FqPoly({4, 1}) && m[0].second == 5, "x^5 - 401 = (x - 1)^5 mod 5");
        std::vector<std::pair<uint64_t, unsigned>> fu;
        Check(factor_u64(1000000007ull * 998244353ull, fu) && fu.size() == 2 && fu[0].first == 998244353ull && fu[1].first == 1000000007ull,
              "factor_u64(1000000007 * 998244353)");
        Check(is_prime_u64(18446744073709551557ull) && !is_prime_u64(3215031751ull), "Miller-Rabin on 2^64 - 59 and the strong pseudoprime 3215031751");
    }
}

// ---------------------------------------------------------------- G2
void G2() {
    std::cout << "-- G2 Chevalley data on chevalley_cells.tsv" << std::endl;
    const double t0 = Now();
    const std::string path = SourcePath("docs/implementation/slot22/nfprank/chevalley_cells.tsv");
    std::ifstream in(path, std::ios::binary);
    Check(static_cast<bool>(in), "chevalley_cells.tsv readable at " + path);
    if (!in) return;
    std::stringstream buf;
    buf << in.rdbuf();
    const std::string text = buf.str();
    Sha256 sh;
    sh.update(text.data(), text.size());
    const std::string hex = sh.hex();
    Check(hex == "3d6f6903f8ae47da017909be2ccca6a1ea961243d4471fb155fc40de71f22154", "chevalley_cells.tsv sha256 = 3d6f6903... (frozen oracle)");
    std::istringstream ls(text);
    std::string line;
    std::size_t cells = 0, diffs = 0, refused = 0, pf_bad = 0, eq_bad = 0;
    while (std::getline(ls, line)) {
        if (line.empty() || line[0] == '#' || line[0] == 'N') continue;
        std::vector<std::string> fld;
        std::stringstream fs(line);
        std::string tok;
        while (std::getline(fs, tok, '\t')) fld.push_back(tok);
        while (fld.size() < 9) fld.push_back("");
        const uint32_t N = static_cast<uint32_t>(std::stoul(fld[0])), p = static_cast<uint32_t>(std::stoul(fld[1]));
        const uint32_t t = static_cast<uint32_t>(std::stoul(fld[4])), lam = static_cast<uint32_t>(std::stoul(fld[5]));
        const uint32_t rs = static_cast<uint32_t>(std::stoul(fld[6])), ch = static_cast<uint32_t>(std::stoul(fld[7]));
        ++cells;
        const ChevalleyData C = chevalley_data(N, p);
        if (!C.refusal.empty()) {
            ++refused;
            continue;
        }
        bool ok = C.t == t && static_cast<uint32_t>(C.lambda_ramified) == lam && C.rstar == rs && C.chev_exponent == ch;
        if (p == 5) {
            std::string cs;
            for (std::size_t i = 0; i < C.nonnorm_chars.size(); ++i)
                cs += (i ? "," : "") + (C.nonnorm_chars[i] == 1 ? std::string("w") : "w^" + std::to_string(C.nonnorm_chars[i]));
            ok = ok && cs == fld[8];
        }
        if (!C.product_formula_ok) ++pf_bad;
        if (!C.equivariant_ok) ++eq_bad;
        if (!ok) {
            ++diffs;
            if (diffs <= 5) std::cout << "  G2 DIFF N=" << N << " p=" << p << std::endl;
        }
    }
    std::cout << "NF_PRANK_CHEVALLEY cells=" << cells << " differences=" << diffs << " refused=" << refused << " product_formula_bad=" << pf_bad
              << " equivariant_bad=" << eq_bad << " sha256=" << hex.substr(0, 8) << " secs=" << Fx(Now() - t0, 2) << std::endl;
    Check(cells == 1240, "G2: 1240 cells");
    Check(diffs == 0 && refused == 0, "G2: 0 differences, 0 refused");
    Check(pf_bad == 0 && eq_bad == 0, "G2: product formula and the equivariant count hold");
    // (31,5) in detail (spec sec. 7.1): rows [1,2,3,4] and [2,2,0,1]
    const ChevalleyData C = chevalley_data(31, 5);
    Check(C.rows.size() == 2 && C.rows[0] == std::vector<uint32_t>({1, 2, 3, 4}) && C.rows[1] == std::vector<uint32_t>({2, 2, 0, 1}),
          "(31,5) symbol rows [1,2,3,4], [2,2,0,1]");
}

// ---------------------------------------------------------------- G3
struct Proto {
    uint32_t N;
    std::size_t S;
    const char* deg2;
    const char* bound;
    uint32_t rank_upper;
    double Rp, Res, hR, est;
    PRankLevel level;
};

void G3() {
    std::cout << "-- G3 class-group bounds against the prototype (hR_runs.out)" << std::endl;
    const Proto cells[] = {
        {11, 88, "[(2, 4), (3, 4), (19, 2), (19, 2)]", "[5]", 1, 37.079062, 1.08206, 185.3968, 5.000, PRankLevel::CERTIFIED_EXACT},
        {31, 474, "[(2, 4), (3, 4), (7, 4), (19, 2), (19, 2), (29, 2), (29, 2)]", "[5, 5]", 2, 51.210267, 0.94081, 1280.2371, 25.000,
         PRankLevel::UPPER_CERTIFIED_LOWER_ANALYTIC},
        {41, 803, "[(2, 4), (3, 4), (7, 4), (19, 2), (19, 2), (29, 2), (29, 2), (59, 2), (59, 2)]", "[10]", 1, 248.191679, 1.04277, 2482.1050, 10.001,
         PRankLevel::CERTIFIED_EXACT},
        {61, 1606,
         "[(2, 4), (3, 4), (7, 4), (19, 2), (19, 2), (29, 2), (29, 2), (59, 2), (59, 2), (79, 2), (79, 2), (89, 2), (89, 2), (109, 2), (109, 2)]",
         "[5]", 1, 1187.888952, 1.12718, 5939.0802, 5.000, PRankLevel::CERTIFIED_EXACT},
    };
    for (const Proto& c : cells) {
        const PRankResult R = nf_prank(c.N, 5);
        PrintResult(R);
        Agree(c.N, "factor_base", std::to_string(c.S), std::to_string(R.factor_base), R.factor_base == c.S);
        Agree(c.N, "deg>=2", c.deg2, Deg2Str(R.deg2plus), Deg2Str(R.deg2plus) == c.deg2);
        Agree(c.N, "minkowski_complete", "1", std::to_string(R.minkowski_complete), R.minkowski_complete && R.descended == 0);
        Agree(c.N, "group_bound", c.bound, BoundStr(R.group_bound), BoundStr(R.group_bound) == c.bound);
        Agree(c.N, "rank_upper", std::to_string(c.rank_upper), std::to_string(R.rank_upper), R.rank_upper == c.rank_upper);
        Agree(c.N, "rank_lower", "1", std::to_string(R.rank_lower), R.rank_lower == 1);
        Agree(c.N, "genus_ok", "1", std::to_string(R.genus_ok), R.genus_ok && R.genus_bad == 0);
        Agree(c.N, "p_coprime_unit_index", "1", std::to_string(R.p_coprime_unit_index), R.p_coprime_unit_index && R.index_rank == 2);
        Agree(c.N, "R'", Fx(c.Rp, 6), Fx(R.regulator_found, 6), std::fabs(R.regulator_found - c.Rp) < 2e-5);
        Agree(c.N, "Res(X=1e7)", Fx(c.Res, 5), Fx(R.euler_res, 5), std::fabs(R.euler_res - c.Res) < 1e-5);
        Agree(c.N, "hR", Fx(c.hR, 4), Fx(R.hR_estimate, 4), std::fabs(R.hR_estimate - c.hR) < 1e-3);
        Agree(c.N, "h_times_index(1%)", Fx(c.est, 3), Fx(R.h_times_index, 3), std::fabs(R.h_times_index - c.est) < 0.01 * c.est);
        Agree(c.N, "level", prank_level_name(c.level), prank_level_name(R.level), R.level == c.level);
        Check(R.rejected_numerics == 0, "(" + std::to_string(c.N) + ",5): no unit rejected on numerics");
    }
    // the descent path: (61,5) with B_0 = 3000 gives the same group
    {
        PRankOptions o;
        o.dense_bound = 3000;
        o.analytic = false;
        const PRankResult R = nf_prank(61, 5, o);
        PrintResult(R);
        Check(R.descended == R.factor_base - R.dense_ideals && R.descent_failed == 0 && R.descended > 1000, "(61,5) B_0 = 3000: every ideal above B_0 descended");
        Check(BoundStr(R.group_bound) == "[5]" && R.rank_upper == 1 && R.genus_ok && R.level == PRankLevel::CERTIFIED_EXACT,
              "(61,5) B_0 = 3000: Z^{S_0}/R_0 = [5], CERTIFIED_EXACT (= the full Minkowski base)");
    }
    // the p^2 | N - 1 path (P_1 P_2^4) at (101,5): internal consistency only (a CE2C control, rank 1 expected, not claimed)
    {
        const PRankResult R = nf_prank(101, 5);
        PrintResult(R);
        Check(R.index == BigInt(5) && R.order_maximal && R.minkowski_complete && R.genus_ok && R.p_coprime_unit_index,
              "(101,5): Round 2, Minkowski-complete, genus ok, 5 not dividing [E:E']");
        Check(R.level != PRankLevel::REFUSED && R.rank_upper >= 1, "(101,5): not refused, rank_upper >= 1");
        // h / [E:E'] must be (numerically) an integer
        Check(std::fabs(R.h_times_index - std::round(R.h_times_index)) < 0.02 && std::round(R.h_times_index) >= 1, "(101,5): hR/R' numerically integral");
    }
}

// ---------------------------------------------------------------- G4
void G4() {
    std::cout << "-- G4 negative controls" << std::endl;
    {  // (a) an ideal of norm <= M_K removed
        PRankOptions o;
        o.analytic = false;
        o.drop_ideal = 50;
        const PRankResult R = nf_prank(31, 5, o);
        PrintResult(R);
        Check(!R.minkowski_complete && R.level == PRankLevel::REFUSED && R.refusal.find("Minkowski-complete") != std::string::npos,
              "G4 (a): an ideal removed -> minkowski_complete = false, REFUSED by name");
    }
    {  // (b) one valuation corrupted
        PRankOptions o;
        o.analytic = false;
        o.corrupt_relation = true;
        const PRankResult R = nf_prank(31, 5, o);
        PrintResult(R);
        Check(!R.genus_ok && R.genus_bad >= 1 && R.level == PRankLevel::REFUSED && R.refusal.find("genus") != std::string::npos,
              "G4 (b): corrupted valuation -> genus_ok = false, REFUSED by name");
    }
    {  // (c) Z[alpha] forced at (401,5)
        PRankOptions o;
        o.force_zalpha = true;
        const PRankResult R = nf_prank(401, 5, o);
        PrintResult(R);
        Check(!R.order_maximal && R.level == PRankLevel::REFUSED && R.refusal.find("not 5-maximal") != std::string::npos,
              "G4 (c): Z[alpha] forced at (401,5) -> order_maximal = false, REFUSED");
    }
    for (const auto& c : {std::make_pair(149u, 37u), std::make_pair(709u, 59u), std::make_pair(269u, 67u)}) {  // (d)
        const PRankResult R = nf_prank(c.first, c.second);
        PrintResult(R);
        Check(R.level == PRankLevel::REFUSED && R.refusal.find("p divides h_F") != std::string::npos,
              "G4 (d): irregular p = " + std::to_string(c.second) + " REFUSED, \"p divides h_F\"");
    }
    {  // (e) X = 500 (the truncation error is NOT monotone in X: |hR/R' - 25| = 0.08 at 10^2, 0.255 at 500, 0.112 at 10^3, 0.056 at 2*10^3;
       // 500 gives margin 0.155 over the 0.1 threshold, 10^3 only 0.012; referee R1)
        PRankOptions o;
        o.euler_X = 500;
        const PRankResult R = nf_prank(31, 5, o);
        PrintResult(R);
        Check(std::fabs(R.h_times_index - 25.0) > 0.1, "G4 (e): X = 500 moves the estimate visibly (|hR/R' - 25| > 0.1)");
    }
}

// ---------------------------------------------------------------- hooks
void Hooks() {
    uint64_t dense = 20000;
    if (const char* d = std::getenv("NF_PRANK_DENSE")) dense = std::strtoull(d, nullptr, 10);
    struct Arm {
        std::string name;
        std::vector<std::pair<uint32_t, uint32_t>> cells;
        bool predict_rank2;
    };
    std::vector<Arm> arms;
    if (const char* ce = std::getenv("NF_PRANK_CE")) {
        // comma-separated tokens: CE1, CE2, CE1C, CE2C; "all" or "1" = CE1 + CE2
        std::map<std::string, bool> want;
        std::stringstream ss(ce);
        std::string tok;
        bool bad = false;
        while (std::getline(ss, tok, ',')) {
            if (tok == "all" || tok == "1") {
                want["CE1"] = want["CE2"] = true;
            } else if (tok == "CE1" || tok == "CE2" || tok == "CE1C" || tok == "CE2C") {
                want[tok] = true;
            } else {
                bad = true;
            }
        }
        Check(!bad, "NF_PRANK_CE parses as a list of CE1, CE2, CE1C, CE2C, all");
        auto mk = [](std::initializer_list<uint32_t> Ns) {
            std::vector<std::pair<uint32_t, uint32_t>> v;
            for (uint32_t N : Ns) v.emplace_back(N, 5u);
            return v;
        };
        if (want["CE1"]) arms.push_back({"CE1", mk({31, 131, 181, 211, 281, 541, 571, 691, 761}), true});
        if (want["CE2"]) arms.push_back({"CE2", mk({401, 701, 751}), true});
        if (want["CE1C"]) arms.push_back({"CE1C", mk({11, 41, 61, 71, 191, 241, 271, 311, 331, 421, 431, 461, 491, 521, 631, 641, 661}), false});
        if (want["CE2C"]) arms.push_back({"CE2C", mk({101, 151, 251, 601}), false});
    }
    if (const char* cs = std::getenv("NF_PRANK_CELLS")) {
        Arm a{"CELLS", {}, false};
        std::stringstream ss(cs);
        std::string tok;
        bool bad = false;
        while (std::getline(ss, tok, ',')) {
            const std::size_t c = tok.find(':');
            if (c == std::string::npos) {
                bad = true;
                break;
            }
            a.cells.emplace_back(static_cast<uint32_t>(std::stoul(tok.substr(0, c))), static_cast<uint32_t>(std::stoul(tok.substr(c + 1))));
        }
        Check(!bad, "NF_PRANK_CELLS parses as N:p,...");
        if (bad) a.cells.clear();
        arms.push_back(a);
    }
    for (const Arm& a : arms) {
        std::cout << "EFFECTIVE NumberFieldPRankValidation arm=" << a.name << " cells=" << a.cells.size() << " dense_bound=" << dense << std::endl;
        for (const auto& c : a.cells) {
            PRankOptions o;
            o.dense_bound = dense;
            const PRankResult R = nf_prank(c.first, c.second, o);
            PrintResult(R);
            if (a.predict_rank2) {
                const bool certified_upper = R.minkowski_complete && R.genus_ok && R.level != PRankLevel::REFUSED;
                if (certified_upper && R.rank_upper <= 1)
                    std::cout << "CE_FINDING N=" << c.first << " p=" << c.second << " rank_upper=" << R.rank_upper
                              << " : certified counterexample to CE Lemma 5.13 (g_p >= 2) -- STOP, report (R-22a)" << std::endl;
                std::cout << "CE_VERDICT arm=" << a.name << " N=" << c.first << " p=" << c.second << " rank_upper=" << R.rank_upper
                          << " prediction=rank_upper>=2 " << (R.rank_upper >= 2 ? "HELD" : "NOT_HELD") << " level=" << prank_level_name(R.level)
                          << std::endl;
                Check(R.level != PRankLevel::REFUSED, a.name + " (" + std::to_string(c.first) + ",5): not refused");
                Check(R.rank_upper >= 2, a.name + " (" + std::to_string(c.first) + ",5): rank_upper >= 2 (CE Lemma 5.13)");
            } else {
                Check(R.level != PRankLevel::REFUSED || c.second != 5, a.name + " (" + std::to_string(c.first) + "," + std::to_string(c.second) + "): not refused");
            }
        }
    }
}

}  // namespace

uint32_t NumberFieldPRankValidation() {
    g_fail = g_checks = 0;
    const double t0 = Now();
    std::cout << "NumberFieldPRankValidation (slot 22 NF-PRANK; spec NF-PRANK_spec.md sec. 8, oracle nfprank/*.py)" << std::endl;
    try {
        G1();
        G2();
        G3();
        G4();
        Hooks();
    } catch (const std::exception& e) {
        Check(false, std::string("exception: ") + e.what());
    }
    std::cout << "NF_PRANK_SUMMARY checks=" << g_checks << " failing=" << g_fail << " secs=" << Fx(Now() - t0, 2) << std::endl;
    return g_fail;
}
