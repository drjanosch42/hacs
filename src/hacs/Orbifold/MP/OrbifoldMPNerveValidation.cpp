// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 22, MP-1b) OrbifoldMPNerveValidation -- see the header.
// Gate values: docs/implementation/slot22/MP_predictions.md P2 (MP-0, countersigned before the first run);
// primary oracle the frozen docs/implementation/slot22/mp/audit/mv_check.out (MvCheckOracle.inc).
#include "OrbifoldMPNerveValidation.h"

#include "AmalgamOracle.h"
#include "Descriptor.h"
#include "MPSet.h"
#include "Nerve.h"
#include "SparseLength.h"

#include "../../Integral/Model/IntegralGroups.h"
#include "../../Integral/Model/IsotropyModel.h"
#include "../../../ModArithmetic/DenseBuf.h"
#include "../../../ModArithmetic/DenseElim.h"
#include "../../../ModArithmetic/DenseView.h"
#include "../../../ModArithmetic/EngineCheck.h"
#include "../../../ModArithmetic/ModCtx.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

using integral_model::ModelData;
using integral_model::Simplex;
using orbifold_mp::Descriptor;
using orbifold_mp::InternedMPSet;
using orbifold_mp::LengthStats;
using orbifold_mp::SparseRows;

struct Tally {
    uint32_t checks = 0, failing = 0;
    void check(bool ok, const std::string& what) {
        ++checks;
        if (!ok) {
            ++failing;
            std::cout << "  [FAIL] " << what << "\n";
        }
    }
};

uint32_t Pow(uint32_t ell, uint32_t k) {
    uint32_t m = 1;
    for (uint32_t i = 0; i < k; ++i) m *= ell;
    return m;
}

// ---- the complex of one triangulation over one ring Z/ell^k, one n ------------------------------------
struct CellResult {
    std::vector<uint64_t> dimC;      // dim C^q, q = 0 .. q_top
    std::vector<uint64_t> len_im;    // length im d^q, q = 0 .. q_top - 1
    std::vector<int64_t> H;          // length H^q, q = 0 .. q_top - 1
    bool dd_zero = true;             // d^{q+1} d^q = 0 on the matrices
    LengthStats top;                 // stats of the last differential
};

// H^q for q = 0 .. q_top - 1 (needs d^0 .. d^{q_top - 1}).
CellResult Cell(const InternedMPSet& S, uint32_t n, uint32_t ell, uint32_t k, uint32_t q_top,
                orbifold_mp::Twist twist = orbifold_mp::Twist::Left) {
    const uint32_t m = Pow(ell, k), d = n + 1;
    CellResult R;
    for (uint32_t q = 0; q <= q_top; ++q) R.dimC.push_back(S.count_nondegenerate(q) * d);
    std::vector<SparseRows> D;
    for (uint32_t q = 0; q < q_top; ++q) {
        D.push_back(orbifold_mp::Coboundary(S, q, n, m, twist));
        const LengthStats st = orbifold_mp::RowSpanLength(D.back(), ell, k);
        R.len_im.push_back(st.length);
        if (q + 1 == q_top) R.top = st;
        if (q >= 1) R.dd_zero = R.dd_zero && orbifold_mp::ProductIsZero(D[q], D[q - 1], m);
    }
    for (uint32_t q = 0; q < q_top; ++q)
        R.H.push_back(static_cast<int64_t>(k * R.dimC[q]) - static_cast<int64_t>(R.len_im[q]) -
                      (q ? static_cast<int64_t>(R.len_im[q - 1]) : 0));
    return R;
}

// ---- the frozen file, re-read --------------------------------------------------------------------------
bool RereadMvCheck(std::vector<orbifold_mp::MvCheckRow>* rows, std::string* path) {
#ifdef OC_SOURCE_DIR
    *path = std::string(OC_SOURCE_DIR) + "/docs/implementation/slot22/mp/audit/mv_check.out";
#else
    *path = "docs/implementation/slot22/mp/audit/mv_check.out";
#endif
    std::ifstream in(*path);
    if (!in) return false;
    std::string line;
    bool in1 = false;
    while (std::getline(in, line)) {
        if (line.rfind("(1)", 0) == 0) {
            in1 = true;
            continue;
        }
        if (line.rfind("(2)", 0) == 0) break;
        if (!in1) continue;
        orbifold_mp::MvCheckRow r{};
        std::string s = line;
        for (char& c : s)
            if (c == '=' || c == '[' || c == ']' || c == ',' || c == 'l' || c == 'n') c = ' ';
        std::istringstream is(s);
        if (!(is >> r.ell >> r.n)) return false;
        for (uint32_t q = 0; q < 8; ++q)
            if (!(is >> r.dims[q])) return false;
        rows->push_back(r);
    }
    return true;
}

const orbifold_mp::MvCheckRow& OracleRow(uint32_t ell, uint32_t n) {
    for (const auto& r : orbifold_mp::MvCheckOracle())
        if (r.ell == ell && r.n == n) return r;
    ENGINE_CHECK_MSG(false, "OrbifoldMPNerveValidation: no mv_check.out row for this (ell, n)");
    return orbifold_mp::MvCheckOracle().front();
}

std::string JoinI(const std::vector<int64_t>& v) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < v.size(); ++i) os << (i ? "," : "") << v[i];
    os << "]";
    return os.str();
}

std::string JoinU(const std::vector<uint64_t>& v) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < v.size(); ++i) os << (i ? "," : "") << v[i];
    os << "]";
    return os.str();
}

integral_model::AbelianGroupType Group(uint32_t free_rank, const std::vector<std::pair<uint32_t, uint32_t>>& prime_exps) {
    integral_model::AbelianGroupType G;
    G.free_rank = free_rank;
    for (const auto& [ell, a] : prime_exps) G.add_torsion(ell, a);
    return G;
}

// ORBIFOLD_MP_CELLS (read-only): the extra nerve-H^2 values of n, from {4, 6}.  Returns false on an unknown token.
bool ParseCells(std::vector<uint32_t>* extra, std::string* raw) {
    const char* env = std::getenv("ORBIFOLD_MP_CELLS");
    if (env == nullptr || *env == '\0') return true;
    *raw = env;
    std::string s(env);
    for (char& c : s)
        if (c == ',') c = ' ';
    std::istringstream is(s);
    std::string tok;
    std::set<uint32_t> out;
    while (is >> tok) {
        if (tok == "all") {
            out.insert(4);
            out.insert(6);
        } else if (tok == "4" || tok == "h2n4") {
            out.insert(4);
        } else if (tok == "6" || tok == "h2n6") {
            out.insert(6);
        } else {
            return false;
        }
    }
    extra->assign(out.begin(), out.end());
    return true;
}

// RowSpanLength against the dense length (engine::elim::length over Z/ell^e, e >= 2; the rref rank over F_ell) on deterministic pseudo-random sparse
// matrices over Z/ell^e whose entries are biased towards non-units, so the residual path is exercised (the MP
// coboundaries themselves never leave a residual).  Returns (cases, agreeing, cases with a residual).
void SparseLengthSelfTest(uint32_t* cases, uint32_t* agree, uint32_t* with_residual) {
    uint64_t x = 0x9e3779b97f4a7c15ull;
    auto next = [&x]() {
        x = x * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<uint32_t>(x >> 33);
    };
    *cases = *agree = *with_residual = 0;
    for (uint32_t ell : {2u, 3u})
        for (uint32_t e = 1; e <= 3; ++e)
            for (uint32_t rep = 0; rep < 8; ++rep) {
                const uint32_t m = Pow(ell, e), rows = 8 + next() % 40, cols = 5 + next() % 30;
                engine::DenseBuf<uint32_t> D(rows, cols);
                SparseRows A;
                A.ncols = cols;
                for (uint32_t r = 0; r < rows; ++r) {
                    for (uint32_t c = 0; c < cols; ++c) {
                        if (next() % 5 != 0) continue;
                        uint32_t v = (next() % 3 == 0) ? next() % m : (ell * (next() % m)) % m;   // mostly non-units
                        if (v == 0) continue;
                        D.at(r, c) = v;
                        A.col.push_back(c);
                        A.val.push_back(static_cast<uint8_t>(v));
                    }
                    A.start.push_back(A.col.size());
                }
                const engine::ModCtx<uint32_t> ctx = (e == 1) ? engine::ModCtx<uint32_t>(m) : engine::ModCtx<uint32_t>::prime_power(ell, e);
                uint32_t dense = 0;
                if (e == 1) {
                    engine::DenseBuf<uint32_t> C = D;
                    dense = engine::elim::rref(ctx, engine::view(C)).rank;
                } else {
                    dense = engine::elim::length(ctx, engine::cview(D));
                }
                for (orbifold_mp::RowOrder o : {orbifold_mp::RowOrder::Natural, orbifold_mp::RowOrder::Reversed}) {
                    const LengthStats st = orbifold_mp::RowSpanLength(A, ell, e, 200000000ull, o);
                    ++*cases;
                    *agree += st.length == dense ? 1 : 0;
                    *with_residual += st.residual_rows > 0 ? 1 : 0;
                }
            }
}

}  // namespace

uint32_t OrbifoldMPNerveValidation() {
    const auto t0 = std::chrono::steady_clock::now();
    Tally T;
    std::cout << "OrbifoldMPNerveValidation: MP-1b, MP_predictions.md P2 (model=mp)\n";

    // ---- the arm -------------------------------------------------------------------------------------------
    std::vector<uint32_t> h2_n = {0, 1, 2, 3};
    std::vector<uint32_t> extra;
    std::string raw;
    if (!ParseCells(&extra, &raw)) {
        std::cout << "OrbifoldMPNerveValidation: REFUSED: ORBIFOLD_MP_CELLS=" << raw << " (accepted: 4, 6, all; comma-separated)\n";
        return 1;
    }
    if (!raw.empty()) {
        for (uint32_t n : extra) h2_n.push_back(n);
        std::cout << "EFFECTIVE OrbifoldMPNerveValidation arm=h2 cells=" << extra.size() * 3 * 2 << " n=";
        for (size_t i = 0; i < extra.size(); ++i) std::cout << (i ? "," : "") << extra[i];
        std::cout << " (ell in {2,3,5}, M7 and R_Z; default n <= 3 unchanged)\n";
    }

    // ---- the descriptors (R_Z from the inventory, as OrbifoldMPRulesValidation) -----------------------------
    const ModelData base = integral_model::SL2Z7VertexModelData();
    uint32_t rho2 = UINT32_MAX, ivert = UINT32_MAX;
    for (uint32_t r = 0; r < base.reps[0].size(); ++r) {
        if (base.order[0][r] == 6) rho2 = base.reps[0][r][0];
        if (base.order[0][r] == 4) ivert = base.reps[0][r][0];
    }
    std::vector<Simplex> arc;
    for (const Simplex& e : base.reps[1])
        if ((e[0] == rho2 && e[1] == ivert) || (e[0] == ivert && e[1] == rho2)) arc.push_back(e);
    T.check(arc.size() == 1, "R_Z: exactly one edge representative joins the order-6 and the order-4 vertex orbits");
    const ModelData dRZ = integral_model::SubdivideEdgeOrbit(base, arc.at(0));
    const Descriptor m7 = orbifold_mp::FromModelData(base, "M7");
    const Descriptor rz = orbifold_mp::FromModelData(dRZ, "R_Z");
    const std::vector<const Descriptor*> tris = {&m7, &rz};

    // ---- the interned MP sets, k <= 3 (the cochains) and k = 4 (|S_4|, d o d through degree 4) -------------
    std::vector<std::unique_ptr<InternedMPSet>> S;
    const std::vector<std::vector<uint64_t>> nd_expect = {{5, 127, 2541, 53143}, {7, 165, 2927, 56349}};
    for (size_t t = 0; t < tris.size(); ++t) {
        const Descriptor& D = *tris[t];
        const auto ts = std::chrono::steady_clock::now();
        S.push_back(std::make_unique<InternedMPSet>(D, 4));
        const InternedMPSet& I = *S.back();
        const double es = std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count();
        bool counts = true;
        for (uint32_t k = 0; k <= 3; ++k) counts = counts && I.count_nondegenerate(k) == nd_expect[t][k];
        T.check(counts, "P1.1/P1.2 |S_0..S_3|^nd of the interned set on " + D.name);
        // String for string against MP-1a's MPSet (the independent reference), k <= 3.
        const orbifold_mp::MPSet ref(D, 3);
        bool same = true;
        for (uint32_t k = 0; k <= 3 && same; ++k) {
            std::set<std::vector<int64_t>> a, b;
            for (const auto& s : ref.strings(k)) a.insert(s.encode());
            for (uint64_t i = 0; i < I.count_all(k); ++i) b.insert(I.to_string(k, i).encode());
            same = a == b && a.size() == ref.count_all(k);
        }
        T.check(same, "interned S_<=3 = MPSet's S_<=3 string for string on " + D.name);
        // Faces of S_4 close up in S_3 and satisfy d_i d_j = d_{j-1} d_i (the k = 4 level, exhaustively).
        bool ident = true;
        for (uint64_t i = 0; i < I.count_all(4) && ident; ++i)
            for (uint32_t j = 1; j <= 4; ++j)
                for (uint32_t h = 0; h < j; ++h)
                    ident = ident && I.face(3, I.face(4, i, j), h) == I.face(3, I.face(4, i, h), j - 1);
        T.check(ident, "S_4 on " + D.name + ": every face in S_3, d_i d_j = d_{j-1} d_i (exhaustive)");
        // d o d = 0 formally, every built degree (C^0 -> C^2, C^1 -> C^3, C^2 -> C^4).
        for (uint32_t q = 0; q <= 2; ++q) {
            uint64_t n_checked = 0;
            const bool ok = orbifold_mp::FormalDSquared(I, q, &n_checked);
            T.check(ok && n_checked == I.count_nondegenerate(q + 2),
                    "P2.6a d^" + std::to_string(q + 1) + " d^" + std::to_string(q) + " = 0 in Z[U] (x) Z[S_" + std::to_string(q) +
                        "^nd] on all " + std::to_string(n_checked) + " strings of S_" + std::to_string(q + 2) + "^nd, " + D.name);
        }
        std::cout << "  MP_NERVE_SET model=mp tri=" << D.name << " |U|=" << I.elements().size() << " |S_k nondeg|=[" << I.count_nondegenerate(0)
                  << ", " << I.count_nondegenerate(1) << ", " << I.count_nondegenerate(2) << ", " << I.count_nondegenerate(3) << ", "
                  << I.count_nondegenerate(4) << "] |S_k all|=[" << I.count_all(0) << ", " << I.count_all(1) << ", " << I.count_all(2) << ", "
                  << I.count_all(3) << ", " << I.count_all(4) << "] table_MB=" << std::fixed << std::setprecision(1)
                  << I.table_bytes() / 1048576.0 << " build_s=" << std::setprecision(2) << es << std::defaultfloat << "\n";
        std::cout << "  OUTPUT P5 model=mp tri=" << D.name << " |S_4^nd|=" << I.count_nondegenerate(4) << " (predicted 1.0-1.2 M)\n";
    }

    // ---- the sparse length against the dense Howell length (the residual path included) ----------------------
    {
        uint32_t cases = 0, agree = 0, resid = 0;
        SparseLengthSelfTest(&cases, &agree, &resid);
        std::cout << "  SPARSE_LENGTH self-test: cases=" << cases << " agree=" << agree << " with_residual=" << resid
                  << " (vs engine::elim::length, Z/2^e, Z/3^e, e <= 3, both row orders)\n";
        T.check(cases == 96 && agree == cases && resid > 0, "RowSpanLength = the dense Howell length on 96 pseudo-random matrices, residual path exercised");
    }

    // ---- the frozen oracle: .inc == the tracked file; AmalgamOracle == .inc ----------------------------------
    {
        std::vector<orbifold_mp::MvCheckRow> file_rows;
        std::string path;
        const bool read = RereadMvCheck(&file_rows, &path);
        bool eq = read && file_rows.size() == orbifold_mp::MvCheckOracle().size();
        for (size_t i = 0; eq && i < file_rows.size(); ++i) {
            const auto &a = file_rows[i], &b = orbifold_mp::MvCheckOracle()[i];
            eq = a.ell == b.ell && a.n == b.n && std::equal(a.dims, a.dims + 8, b.dims);
        }
        T.check(eq && file_rows.size() == 52, "MvCheckOracle.inc = section (1) of the tracked " + path + " (52 rows)");
        uint32_t agree = 0, values = 0;
        for (const auto& r : orbifold_mp::MvCheckOracle()) {
            const std::vector<uint32_t> a = orbifold_mp::AmalgamDims(r.n, r.ell, 7);
            for (uint32_t q = 0; q < 8; ++q) {
                ++values;
                agree += a[q] == r.dims[q] ? 1 : 0;
            }
        }
        std::cout << "AMALGAM_ORACLE AmalgamOracle == mv_check.inc rows=" << orbifold_mp::MvCheckOracle().size() << " values=" << values
                  << " agree=" << agree << "\n";
        T.check(values == 416 && agree == values, "AmalgamOracle == mv_check.inc on all 52 x 8 values (R5, its own check)");
    }

    // ---- P2.1 / P2.2 / P2.6c: the nerve over F_ell -------------------------------------------------------------
    uint32_t oracle_cells = 0, oracle_agree = 0, odd_acyclic = 0;
    for (size_t t = 0; t < tris.size(); ++t) {
        const InternedMPSet& I = *S[t];
        const std::string& nm = tris[t]->name;
        for (uint32_t ell : {2u, 3u, 5u, 7u}) {
            std::ostringstream line;
            line << "  NERVE model=mp tri=" << nm << " ring=F_" << ell << " (H^0,H^1) n=0..12:";
            uint32_t cell_agree = 0;
            for (uint32_t n = 0; n <= 12; ++n) {
                const bool h2 = ell != 7 && std::find(h2_n.begin(), h2_n.end(), n) != h2_n.end();
                const CellResult R = Cell(I, n, ell, 1, h2 ? 3 : 2);
                const auto& O = OracleRow(ell, n);
                T.check(R.dd_zero, "P2.6a d o d = 0 (matrices) " + nm + " F_" + std::to_string(ell) + " n=" + std::to_string(n));
                T.check(R.H[0] == O.dims[0], "P2.1 H^0 " + nm + " F_" + std::to_string(ell) + " n=" + std::to_string(n));
                T.check(R.H[1] == O.dims[1], "P2.1 H^1 " + nm + " F_" + std::to_string(ell) + " n=" + std::to_string(n));
                ++oracle_cells;
                const bool ok01 = R.H[0] == O.dims[0] && R.H[1] == O.dims[1];
                oracle_agree += ok01 ? 1 : 0;
                cell_agree += ok01 ? 1 : 0;
                line << " " << R.H[0] << "," << R.H[1];
                if (n % 2 == 1 && ell != 2) {
                    const bool nz = R.dimC[0] > 0 && R.dimC[1] > 0 && R.dimC[2] > 0;
                    const bool acyc = R.H[0] == 0 && R.H[1] == 0 && (!h2 || R.H[2] == 0);
                    T.check(nz && acyc, "P2.6c odd n=" + std::to_string(n) + ", 2 a unit (F_" + std::to_string(ell) + "), " + nm +
                                            ": the complex is non-zero and acyclic");
                    odd_acyclic += (nz && acyc) ? 1 : 0;
                }
                if (h2) {
                    const bool armed = n >= 4;
                    T.check(R.H[2] == O.dims[2], std::string(armed ? "P2.2 (armed) " : "P2.2 ") + "H^2 " + nm + " F_" + std::to_string(ell) +
                                                     " n=" + std::to_string(n));
                    ++oracle_cells;
                    oracle_agree += R.H[2] == O.dims[2] ? 1 : 0;
                    std::cout << "  NERVE_H2 model=mp tri=" << nm << " ring=F_" << ell << " n=" << n << " dimC=" << JoinU(R.dimC)
                              << " H^0..2=" << JoinI(R.H) << " oracle=[" << O.dims[0] << "," << O.dims[1] << "," << O.dims[2] << "]"
                              << (armed ? " arm=ORBIFOLD_MP_CELLS" : "") << " d2: rows=" << R.top.rows << " cols=" << R.top.cols
                              << " nnz=" << R.top.input_nnz << " pivots=" << R.top.pivots << " fill=" << R.top.pivot_nnz
                              << " max_row=" << R.top.max_pivot_row << " wall_s=" << std::fixed << std::setprecision(2) << R.top.wall_s
                              << std::defaultfloat << "\n";
                }
            }
            line << "  agree=" << cell_agree << "/13";
            std::cout << line.str() << "\n";
        }
    }
    std::cout << "ORACLE mv_check.out cells=" << oracle_cells << " agree=" << oracle_agree << "\n";
    T.check(oracle_agree == oracle_cells && oracle_cells == 104 + 24 + 6 * extra.size(),
            "ORACLE mv_check.out: every nerve cell agrees (104 H^0/H^1 + 24 H^2 default cells + the armed ones)");
    T.check(odd_acyclic == 2 * 3 * 6, "P2.6c: 36 odd-n cells (ell in {3,5,7}, n odd <= 11, both triangulations) non-zero and acyclic");

    // ---- P2.5: the integral groups on C_I (certified at 2, 3) -------------------------------------------------
    const integral_model::IsotropyModel CI;   // [C, def:isomodel] on M7, q_max = 5
    constexpr uint32_t kGoodPrime = 10007;
    std::map<uint32_t, integral_model::ModelIntegralCohomology> G;
    {
        using V = std::vector<std::pair<uint32_t, uint32_t>>;
        const std::map<uint32_t, std::vector<integral_model::AbelianGroupType>> expect = {
            {0, {Group(1, {}), Group(0, {}), Group(0, V{{2, 2}, {3, 1}}), Group(0, {})}},
            {1, {Group(0, {}), Group(0, {}), Group(0, V{{2, 1}}), Group(0, {})}},
            {2, {Group(0, {}), Group(1, V{{2, 1}}), Group(0, {}), Group(0, V{{2, 1}, {2, 1}})}},
            {4, {Group(0, {}), Group(1, V{{2, 2}, {3, 1}}), Group(0, V{{2, 2}}), Group(0, V{{2, 1}, {2, 1}, {3, 1}})}},
        };
        for (const auto& [n, E] : expect) {
            G.emplace(n, integral_model::AssembleFromModel(CI, n, {2, 3}, kGoodPrime));
            const auto& g = G.at(n);
            T.check(g.all_certified(), "P2.5 C_I certified at 2 and 3, n=" + std::to_string(n));
            std::ostringstream os;
            os << "  INTEGRAL model=C_I n=" << n << " H^0..3 =";
            for (uint32_t q = 0; q <= 3; ++q) {
                T.check(g.H[q] == E[q], "P2.5 H^" + std::to_string(q) + "(Gamma; Sym^" + std::to_string(n) + " Z) = " + E[q].invariant_factor_string() +
                                            " (got " + g.H[q].invariant_factor_string() + ")");
                os << " " << g.H[q].invariant_factor_string() << (q < 3 ? "," : "");
            }
            std::cout << os.str() << "\n";
        }
    }

    // ---- P2.3 / P2.4: lengths over Z/ell^k on the nerve, and the UCT from C_I --------------------------------
    {
        struct LRow {
            uint32_t n, ell, k, h0, h1;
        };
        const std::vector<LRow> p23 = {
            {0, 2, 1, 1, 1}, {0, 2, 2, 2, 2}, {0, 2, 3, 3, 2}, {0, 3, 1, 1, 1}, {0, 3, 2, 2, 1}, {0, 3, 3, 3, 1},
            {1, 2, 1, 0, 1}, {1, 2, 2, 0, 1}, {1, 2, 3, 0, 1}, {1, 3, 1, 0, 0}, {1, 3, 2, 0, 0}, {1, 3, 3, 0, 0},
            {2, 2, 1, 1, 2}, {2, 2, 2, 1, 3}, {2, 2, 3, 1, 4}, {2, 3, 1, 0, 1}, {2, 3, 2, 0, 2}, {2, 3, 3, 0, 3},
            {4, 2, 1, 1, 3}, {4, 2, 2, 2, 6}, {4, 2, 3, 2, 7}, {4, 3, 1, 1, 2}, {4, 3, 2, 1, 3}, {4, 3, 3, 1, 4},
        };
        uint32_t cells = 0, agree = 0, uct = 0;
        for (size_t t = 0; t < tris.size(); ++t) {
            const InternedMPSet& I = *S[t];
            const std::string& nm = tris[t]->name;
            std::ostringstream line;
            line << "  NERVE_ZLK model=mp tri=" << nm << " (n,ell,k):(H^0,H^1)";
            for (const LRow& r : p23) {
                const CellResult R = Cell(I, r.n, r.ell, r.k, 2);
                const std::vector<uint32_t> P = integral_model::PredictedLengths(G.at(r.n), r.ell, r.k);
                const std::string tag = nm + " n=" + std::to_string(r.n) + " Z/" + std::to_string(r.ell) + "^" + std::to_string(r.k);
                const bool ok = R.H[0] == r.h0 && R.H[1] == r.h1;
                T.check(ok, "P2.3 lengths H^0, H^1 " + tag + " = " + std::to_string(r.h0) + ", " + std::to_string(r.h1));
                const bool u = R.H[0] == P[0] && R.H[1] == P[1];
                T.check(u, "P2.5 nerve lengths = UCT from C_I (IntegralGroups) " + tag);
                T.check(R.dd_zero, "P2.6a d o d = 0 (matrices) " + tag);
                ++cells;
                agree += ok ? 1 : 0;
                uct += u ? 1 : 0;
                line << " (" << r.n << "," << r.ell << "," << r.k << "):(" << R.H[0] << "," << R.H[1] << ")";
            }
            // P2.4: n = 0, q = 2.
            for (uint32_t ell : {2u, 3u})
                for (uint32_t k = 1; k <= 3; ++k) {
                    const CellResult R = Cell(I, 0, ell, k, 3);
                    const int64_t expect = (ell == 2) ? std::min<int64_t>(k, 2) : 1;
                    const std::vector<uint32_t> P = integral_model::PredictedLengths(G.at(0), ell, k);
                    const std::string tag = nm + " n=0 Z/" + std::to_string(ell) + "^" + std::to_string(k);
                    T.check(R.H[2] == expect, "P2.4 length H^2 " + tag + " = " + std::to_string(expect));
                    T.check(R.H[2] == P[2], "P2.5 nerve length H^2 = UCT from C_I " + tag);
                    std::cout << "  NERVE_ZLK_H2 model=mp tri=" << nm << " n=0 ring=Z/" << ell << "^" << k << " H^0..2=" << JoinI(R.H)
                              << " UCT(C_I)=[" << P[0] << "," << P[1] << "," << P[2] << "] d2: pivots=" << R.top.pivots
                              << " residual=" << R.top.residual_rows << "x" << R.top.residual_cols << " wall_s=" << std::fixed
                              << std::setprecision(2) << R.top.wall_s << std::defaultfloat << "\n";
                }
            std::cout << line.str() << "\n";
        }
        std::cout << "  ZLK summary: cells=" << cells << " agree=" << agree << " uct_agree=" << uct << "\n";
    }

    // ---- P2.6b: the untwisted d_0 (R5 negative control) ------------------------------------------------------
    for (size_t t = 0; t < tris.size(); ++t) {
        const CellResult R = Cell(*S[t], 2, 2, 1, 2, orbifold_mp::Twist::Untwisted);
        const bool fires = !R.dd_zero || R.H[0] != 1 || R.H[1] != 2;
        std::cout << "  NC P2.6b model=mp tri=" << tris[t]->name << " untwisted d_0, F_2, n=2: d^2=0 " << (R.dd_zero ? "holds" : "BROKEN")
                  << ", H^0,H^1=" << R.H[0] << "," << R.H[1] << " (oracle 1,2) -> " << (fires ? "NC fires" : "NC SILENT") << "\n";
        T.check(fires, "P2.6b NC: the untwisted d_0 breaks d^2 = 0 or H^1 at n = 2, F_2, " + tris[t]->name);
    }

    // ---- (MP-2, F-MP1 F1) the twist rho_n(gamma_1^{-1}), an anti-homomorphism: the NC that sees the side ------------
    for (size_t t = 0; t < tris.size(); ++t) {
        const CellResult R = Cell(*S[t], 1, 2, 1, 2, orbifold_mp::Twist::Inverse);
        std::cout << "  NC F1 model=mp tri=" << tris[t]->name << " twist rho_n(gamma_1^-1), F_2, n=1: d^2=0 "
                  << (R.dd_zero ? "holds -> NC SILENT" : "BROKEN -> NC fires") << "\n";
        T.check(!R.dd_zero, "F-MP1 F1 NC: the twist rho_n(gamma_1^{-1}) breaks d o d = 0 at n = 1, F_2, " + tris[t]->name);
    }

    // ---- (MP-2, F-MP1 F2(a)) a planted wrong central stabiliser: Gamma_{rho^2} := <(ST)^2> (-I dropped) on M7 -------
    // (SC), adaptedness and every H^0 / H^1 / Z/ell^k cell are blind to it (the F-MP1 probe); the default H^2 cells see it.
    {
        Descriptor planted = m7;
        uint32_t hit = 0;
        for (auto& G : planted.orbit_stab)
            if (G.order == 6) {
                G = orbifold_mp::CyclicGroup{G.gen * G.gen, 3};
                ++hit;
            }
        T.check(hit == 1, "F-MP1 F2(a): M7 has exactly one vertex orbit of stabiliser order 6");
        const InternedMPSet P(planted, 3);
        const CellResult R = Cell(P, 0, 2, 1, 3);
        const bool fires = R.H[2] != 1;
        std::cout << "  NC F2a model=mp tri=M7 planted Gamma_rho2=<(ST)^2>: |U|=" << P.elements().size() << " |S_3^nd|="
                  << P.count_nondegenerate(3) << " F_2 n=0 H^0..2=" << JoinI(R.H) << " (oracle 1,1,1) -> "
                  << (fires ? "NC fires" : "NC SILENT") << "\n";
        T.check(fires, "F-MP1 F2(a) NC: the planted <(ST)^2> stabiliser changes H^2(F_2, n = 0) away from 1");
    }

    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "ORBIFOLD_MP OrbifoldMPNerveValidation checks=" << T.checks << " failing=" << T.failing << " wall_s=" << std::fixed
              << std::setprecision(2) << secs << std::defaultfloat << "\n";
    return T.failing == 0 ? 0 : 1;
}
