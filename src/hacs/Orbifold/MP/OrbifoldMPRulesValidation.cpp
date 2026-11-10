// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 22, MP-1a) OrbifoldMPRulesValidation -- see the header.
// Gate values: docs/implementation/slot22/MP_predictions.md P1 (committed, countersigned c529179d, before this
// validation first ran).  Oracle: the frozen outputs of mp_rules.py (mp/out_checks.txt) and of
// mp/audit/adapted_form_check.py (mp/audit/adapted_form_check.out).
#include "OrbifoldMPRulesValidation.h"

#include "Descriptor.h"
#include "MPSet.h"

#include "../../Integral/Model/IsotropyModel.h"
#include "../../../ModArithmetic/EngineCheck.h"
#include "../../../ModArithmetic/Fnv.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {

using hecke::Mat22;
using integral_model::ModelData;
using integral_model::Simplex;
using orbifold_mp::CyclicGroup;
using orbifold_mp::Descriptor;
using orbifold_mp::MPSet;
using orbifold_mp::MPString;
using orbifold_mp::Point;

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

const char* TF(bool b) { return b ? "True" : "False"; }

std::string Join(const std::vector<uint32_t>& v) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < v.size(); ++i) os << (i ? ", " : "") << v[i];
    os << "]";
    return os.str();
}

// The vertex label of the descriptor whose stabiliser order is `order` (unique; ENGINE_CHECKed).
uint32_t LabelOfOrder(const ModelData& d, uint32_t order) {
    uint32_t found = UINT32_MAX, n = 0;
    for (uint32_t r = 0; r < d.reps[0].size(); ++r)
        if (d.order[0][r] == order) {
            found = d.reps[0][r][0];
            ++n;
        }
    ENGINE_CHECK_MSG(n == 1, "OrbifoldMPRulesValidation: the stabiliser inventory has not exactly one vertex orbit of this order");
    return found;
}

std::set<std::set<uint32_t>> TopSets(const Descriptor& D) {
    std::set<std::set<uint32_t>> out;
    for (const auto& t : D.top) out.insert(std::set<uint32_t>(t.begin(), t.end()));
    return out;
}

// ---------------------------------------------------------------------------
// The scratch dump (ORBIFOLD_MP_DUMP), pattern HECKE_DUMP_TAU.
// ---------------------------------------------------------------------------
bool IsDirectory(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

std::string RealPath(const std::string& p) {
    char buf[PATH_MAX];
    if (::realpath(p.c_str(), buf) == nullptr) return std::string();
    return std::string(buf);
}

bool IsInside(const std::string& inner, const std::string& outer) {
    if (outer.empty() || inner.size() < outer.size()) return false;
    if (inner.compare(0, outer.size(), outer) != 0) return false;
    return inner.size() == outer.size() || inner[outer.size()] == '/';
}

void PutMat(std::ostream& os, const Mat22& g) { os << g.a << '\t' << g.b << '\t' << g.c << '\t' << g.d; }

struct DumpRow {
    std::string name;
    uint32_t ell;
    bool adapted, adapted_order_only, good, collar;
    orbifold_mp::QuotientCounts q;
};

// Returns 0 on success (or when unset), 1 on REFUSED / an I/O failure.
uint32_t MaybeDump(const Descriptor& rz, const std::vector<DumpRow>& rows, const MPSet& srz) {
    const char* env = std::getenv("ORBIFOLD_MP_DUMP");
    if (env == nullptr || *env == '\0') return 0;
    const std::string dir(env);
    if (!IsDirectory(dir)) {
        std::cout << "OrbifoldMPRulesValidation: REFUSED: ORBIFOLD_MP_DUMP=" << dir << " is not an existing directory\n";
        return 1;
    }
    const std::string real_dir = RealPath(dir);
#ifdef OC_SOURCE_DIR
    const std::string real_src = RealPath(OC_SOURCE_DIR);
#else
    const std::string real_src;
#endif
    if (real_dir.empty() || IsInside(real_dir, real_src)) {
        std::cout << "OrbifoldMPRulesValidation: REFUSED: ORBIFOLD_MP_DUMP=" << dir << " lies inside the source tree ("
                  << real_src << "); dump into scratch only\n";
        return 1;
    }
    std::map<std::string, std::string> files;
    {
        std::ostringstream os;
        os << "# R_Z vertex orbits: orbit\tlabel\tgen_a\tgen_b\tgen_c\tgen_d\torder\n";
        for (uint32_t o = 0; o < rz.orbit_label.size(); ++o) {
            os << o << '\t' << rz.orbit_label[o] << '\t';
            PutMat(os, rz.orbit_stab[o].gen);
            os << '\t' << rz.orbit_stab[o].order << '\n';
        }
        files["mp_rz_orbits.tsv"] = os.str();
    }
    {
        std::ostringstream os;
        os << "# R_Z points: label\torbit\tg_a\tg_b\tg_c\tg_d\tboundary   (label = g * iota(orbit))\n";
        for (const auto& [label, p] : rz.point) {
            os << label << '\t' << p.orbit << '\t';
            PutMat(os, p.g);
            os << '\t' << (rz.boundary.count(label) ? 1 : 0) << '\n';
        }
        files["mp_rz_points.tsv"] = os.str();
    }
    {
        std::ostringstream os;
        os << "# R_Z top simplices (Sigma_top, stored order): index\tv0\tv1\tv2\n";
        for (uint32_t t = 0; t < rz.top.size(); ++t) os << t << '\t' << rz.top[t][0] << '\t' << rz.top[t][1] << '\t' << rz.top[t][2] << '\n';
        files["mp_rz_top.tsv"] = os.str();
    }
    {
        std::ostringstream os;
        os << "# R_Z vertex transporters A(s0, s1) = {g : g iota(s1) meets iota(s0)}: s0\ts1\tg_a\tg_b\tg_c\tg_d\n";
        for (uint32_t a = 0; a < rz.top.size(); ++a)
            for (uint32_t b = 0; b < rz.top.size(); ++b)
                for (const Mat22& g : orbifold_mp::Arrows(rz, a, b)) {
                    os << a << '\t' << b << '\t';
                    PutMat(os, g);
                    os << '\n';
                }
        files["mp_rz_arrows.tsv"] = os.str();
    }
    {
        std::ostringstream os;
        os << "# predicates: triangulation\tell(0=full)\tadapted_subgroup\tadapted_order_only\tgood\tcollar\tV\tE\tF\tchi\n";
        for (const DumpRow& r : rows)
            os << r.name << '\t' << r.ell << '\t' << r.adapted << '\t' << r.adapted_order_only << '\t' << r.good << '\t'
               << r.collar << '\t' << r.q.V << '\t' << r.q.E << '\t' << r.q.F << '\t' << r.q.chi << '\n';
        files["mp_predicates.tsv"] = os.str();
    }
    {
        std::ostringstream os;
        os << "# R_Z MP set: k\tall\tnondegenerate\n";
        for (uint32_t k = 0; k <= 2; ++k) os << k << '\t' << srz.count_all(k) << '\t' << srz.count_nondegenerate(k) << '\n';
        files["mp_rz_sk.tsv"] = os.str();
    }
    uint64_t h = engine::kFnvOffset64;
    for (const auto& [fn, body] : files) {
        std::ofstream out(dir + "/" + fn, std::ios::binary);
        out << body;
        if (!out) {
            std::cout << "OrbifoldMPRulesValidation: REFUSED: cannot write " << dir << "/" << fn << "\n";
            return 1;
        }
        h = engine::FnvBytes(h, fn);
        h = engine::FnvBytes(h, body);
    }
    std::cout << "EFFECTIVE OrbifoldMPRulesValidation arm=dump dir=" << real_dir << " files=" << files.size() << " fnv=0x"
              << std::hex << std::setw(16) << std::setfill('0') << h << std::dec << std::setfill(' ') << "\n";
    return 0;
}

// The frozen mp_rules.py rows (mp/out_checks.txt), per triangulation (good / fails / counts do not depend on ell).
struct OracleRow {
    std::string name;
    bool adapted3, adapted2, adaptedZ, good, collar;
    std::vector<uint32_t> fails;   // mp_rules.py labels
    uint32_t V, E, F;
    int64_t chi;
};

}  // namespace

uint32_t OrbifoldMPRulesValidation() {
    const auto t0 = std::chrono::steady_clock::now();
    Tally T;
    std::cout << "OrbifoldMPRulesValidation: MP-1a, MP_predictions.md P1 (model=mp)\n";

    // ---- the descriptors -------------------------------------------------------------------------------------
    const ModelData base = integral_model::SL2Z7VertexModelData();
    const uint32_t rho2 = LabelOfOrder(base, 6), ivert = LabelOfOrder(base, 4);
    // R_Z: the edge whose ends have stabiliser orders 6 and 4, from the inventory (not by name).
    std::vector<Simplex> arc;
    for (uint32_t r = 0; r < base.reps[1].size(); ++r) {
        const Simplex& e = base.reps[1][r];
        if ((e[0] == rho2 && e[1] == ivert) || (e[0] == ivert && e[1] == rho2)) arc.push_back(e);
    }
    T.check(arc.size() == 1, "R_Z: exactly one edge representative joins the order-6 and the order-4 vertex orbits");
    const ModelData dRZ = integral_model::SubdivideEdgeOrbit(base, arc.at(0));
    // R_Z+: R_Z with every edge orbit from the horocycle to the order-6 orbit subdivided (iterated).
    const Descriptor m7 = orbifold_mp::FromModelData(base, "M7");
    const Descriptor rz = orbifold_mp::FromModelData(dRZ, "R_Z");
    std::vector<Simplex> toRho;
    for (uint32_t r = 0; r < dRZ.reps[1].size(); ++r) {
        const Simplex& e = dRZ.reps[1][r];
        const bool h0 = rz.boundary.count(e[0]) == 1, h1 = rz.boundary.count(e[1]) == 1;
        const bool o0 = rz.elliptic_order(e[0]) == 6, o1 = rz.elliptic_order(e[1]) == 6;
        if ((h0 && o1) || (h1 && o0)) toRho.push_back(e);
    }
    T.check(toRho.size() == 3, "R_Z+: three edge orbits from the horocycle to rho (v0 rho^2, v1 rho^2, v2 rho)");
    ModelData dRZp = dRZ;
    for (const Simplex& e : toRho) dRZp = integral_model::SubdivideEdgeOrbit(dRZp, e);
    // R_zeta: the three edges at zeta^2 = rho^2 subdivided (slot 17, WP-7's atZeta loop).
    ModelData dZeta = base;
    for (const Simplex& e : base.reps[1])
        if (e[0] == rho2 || e[1] == rho2) dZeta = integral_model::SubdivideEdgeOrbit(dZeta, e);
    const Descriptor rzp = orbifold_mp::FromModelData(dRZp, "R_Z+");
    const Descriptor rzeta = orbifold_mp::FromModelData(dZeta, "R_zeta");
    const Descriptor rgood = orbifold_mp::RGoodDescriptor();

    // R_Z is mp_rules.py's / mp_sl2z.py's R_Z: the midpoint m = 7 on v4 v6, m' = 8 = S m on v5 v6.
    {
        const std::set<std::set<uint32_t>> expect = {{0, 1, 4}, {1, 2, 6}, {2, 3, 5}, {1, 4, 7}, {1, 7, 6}, {2, 5, 8}, {2, 8, 6}};
        bool mprime = rz.point.count(7) && rz.point.count(8) &&
                      rz.key_of_label(8) == rz.key(Point{rz.point.at(7).orbit, Mat22::S()});
        T.check(TopSets(rz) == expect && mprime,
                "R_Z = SubdivideEdgeOrbit(M7, arc v4 v6): the seven triangles of [MP] Sec. 3.2, m' = S m");
        // R_Z+ is mp_rules.py's R_Z+ under the relabelling of its midpoints (ours 9, 10, 11, 12 = its 9, 12, 10, 11).
        const std::map<uint32_t, uint32_t> relabel = {{9, 9}, {10, 12}, {11, 10}, {12, 11}};
        std::set<std::set<uint32_t>> mapped;
        for (const auto& t : rzp.top) {
            std::set<uint32_t> s;
            for (uint32_t v : t) s.insert(relabel.count(v) ? relabel.at(v) : v);
            mapped.insert(s);
        }
        const std::set<std::set<uint32_t>> expectp = {{0, 1, 9},  {1, 10, 9}, {9, 10, 4}, {1, 2, 6},  {2, 3, 12}, {2, 11, 12}, {12, 11, 5},
                                                      {1, 10, 7}, {10, 4, 7}, {1, 7, 6},  {2, 11, 8}, {11, 5, 8},  {2, 8, 6}};
        T.check(mapped == expectp, "R_Z+ (iterated SubdivideEdgeOrbit) = mp_rules.py's R_Z+ triangle for triangle, after relabelling");
    }

    // ---- predicates: the Sec. 3.1 table --------------------------------------------------------------------
    const std::vector<OracleRow> oracle = {
        {"M7", true, true, false, false, false, {1, 2, 4, 6}, 5, 9, 5, 1},
        {"R_Z", true, true, true, false, false, {1, 2, 7}, 6, 12, 7, 1},
        {"R_Z+", true, true, true, false, false, {7}, 9, 21, 13, 1},
        {"R_good", true, true, true, true, false, {}, 13, 33, 21, 1},
    };
    const std::vector<const Descriptor*> four = {&m7, &rz, &rzp, &rgood};
    std::vector<DumpRow> dump_rows;
    uint32_t oracle_rows = 0, oracle_agree = 0;
    bool forms_agree = true, good_forms_agree = true;
    for (size_t i = 0; i < four.size(); ++i) {
        const Descriptor& D = *four[i];
        const OracleRow& O = oracle[i];
        const orbifold_mp::QuotientCounts q = orbifold_mp::Quotient(D);
        const orbifold_mp::GoodResult g = orbifold_mp::Good(D);
        const bool collar = orbifold_mp::Collar(D);
        // mp_rules.py's labels; R_Z+'s midpoints relabelled as above (only m = 7 occurs in its fails list).
        std::vector<uint32_t> fails = g.fails_at;
        if (D.name == "R_Z+")
            for (uint32_t& v : fails) v = (v == 10) ? 12u : (v == 11) ? 10u : (v == 12) ? 11u : v;
        std::sort(fails.begin(), fails.end());
        T.check(q.V == O.V && q.E == O.E && q.F == O.F && q.chi == O.chi, "P1.5 quotient (V,E,F,chi) of " + D.name);
        T.check(g.good == O.good, "P1.8 good (finite-group criterion) of " + D.name);
        T.check(collar == O.collar, "P1.9 collar of " + D.name);
        T.check(fails == O.fails, "P1.10 good_fails_at of " + D.name);
        good_forms_agree = good_forms_agree && g.good == g.good_strong && g.fails_at == g.fails_at_strong;
        for (uint32_t ell : {3u, 2u, 0u}) {
            const bool a = orbifold_mp::Adapted(D, ell, true);
            const bool ao = orbifold_mp::Adapted(D, ell, false);
            forms_agree = forms_agree && a == ao;
            const bool expect = ell == 3 ? O.adapted3 : ell == 2 ? O.adapted2 : O.adaptedZ;
            T.check(a == expect, std::string(ell == 0 ? "P1.7 adapted (full, MP's rule)" : "P1.6 adapted for T^(" + std::to_string(ell) + ")") +
                                     " of " + D.name);
            ++oracle_rows;
            if (a == expect && g.good == O.good && collar == O.collar && fails == O.fails && q.V == O.V && q.E == O.E && q.F == O.F &&
                q.chi == O.chi)
                ++oracle_agree;
            std::cout << "  MP_RULES model=mp tri=" << D.name << " " << (ell == 0 ? std::string("Z (full)") : "ell=" + std::to_string(ell))
                      << " adapted=" << TF(a) << " good=" << TF(g.good) << " collar=" << TF(collar) << " good_fails_at=" << Join(fails)
                      << " quotient(V,E,F,chi)=(" << q.V << ", " << q.E << ", " << q.F << ", " << q.chi << ")\n";
            dump_rows.push_back({D.name, ell, a, ao, g.good, collar, q});
        }
    }
    T.check(forms_agree, "subgroup-form and order-only 'adapted' agree on the four real candidates ([MP] Def. 1.3: every non-maximal stabiliser is central)");
    T.check(good_forms_agree, "the finite-group 'good' and mp_rules.py's stronger 'T(v) in one Gamma_p' agree on the four candidates (R-22b)");
    std::cout << "ORACLE mp_rules.py cells=" << oracle_rows << " agree=" << oracle_agree << "\n";
    T.check(oracle_rows == 12 && oracle_agree == 12, "ORACLE mp_rules.py: 12 rows agree");

    // ---- (SC), the invariant order and its sign bridge -------------------------------------------------------
    const std::vector<const Descriptor*> five = {&m7, &rz, &rzeta, &rzp, &rgood};
    for (const Descriptor* D : five) {
        const orbifold_mp::SCResult sc = orbifold_mp::SimplicialComplexProperty(*D);
        T.check(sc.ok && sc.pairs_checked > 0, "P1.11 (SC) on " + D->name + (sc.ok ? "" : ": " + sc.first_failure));
        T.check(orbifold_mp::InvariantOrder(*D), "P1.11 invariant vertex order (no two vertices of a simplex in one orbit) on " + D->name);
    }
    {
        const std::vector<std::pair<std::string, const ModelData*>> md = {{"M7", &base}, {"R_Z", &dRZ}, {"R_zeta", &dZeta}, {"R_Z+", &dRZp}};
        for (const auto& [nm, d] : md) {
            bool ok = false;
            const uint64_t n = orbifold_mp::OrderedChainSignBridge(*d, &ok);
            T.check(ok && n > 0, "P1.11 orbit-ordered chains = IsotropyModel's C_. (sign per face, " + std::to_string(n) + " faces) on " + nm);
        }
    }

    // ---- the MP set S_k, k <= 3 ------------------------------------------------------------------------------
    const MPSet sm7(m7, 3), srz(rz, 3), szeta(rzeta, 3);
    const std::vector<uint64_t> nd_m7 = {5, 127, 2541, 53143}, nd_rz = {7, 165, 2927, 56349};
    for (uint32_t k = 0; k <= 3; ++k) {
        T.check(sm7.count_nondegenerate(k) == nd_m7[k], "P1.1 |S_" + std::to_string(k) + "^nd| on M7 = " + std::to_string(nd_m7[k]));
        T.check(srz.count_nondegenerate(k) == nd_rz[k], "P1.2 |S_" + std::to_string(k) + "^nd| on R_Z = " + std::to_string(nd_rz[k]));
    }
    for (const auto& [nm, S] : std::vector<std::pair<std::string, const MPSet*>>{{"M7", &sm7}, {"R_Z", &srz}, {"R_zeta", &szeta}}) {
        std::cout << "  MP_SET model=mp tri=" << nm << " |S_k nondeg|=[" << S->count_nondegenerate(0) << ", " << S->count_nondegenerate(1)
                  << ", " << S->count_nondegenerate(2) << ", " << S->count_nondegenerate(3) << "] |S_k all|=[" << S->count_all(0) << ", "
                  << S->count_all(1) << ", " << S->count_all(2) << ", " << S->count_all(3) << "]\n";
        // Structure: each string once; every string re-tested by the forward intersection; every face in S_{k-1};
        // the simplicial identities d_i d_j = d_{j-1} d_i (i < j).
        bool unique = true, member = true, closed = true, identities = true;
        uint64_t ident_checked = 0;
        std::vector<std::set<std::vector<int64_t>>> enc(4);
        for (uint32_t k = 0; k <= 3; ++k)
            for (const MPString& s : S->strings(k)) {
                unique = unique && enc[k].insert(s.encode()).second;
                member = member && S->member(s);
            }
        for (uint32_t k = 1; k <= 3; ++k)
            for (const MPString& s : S->strings(k)) {
                for (uint32_t i = 0; i <= k; ++i) closed = closed && enc[k - 1].count(MPSet::face(s, i).encode()) == 1;
                if (k >= 2)
                    for (uint32_t j = 1; j <= k; ++j)
                        for (uint32_t i = 0; i < j; ++i) {
                            ++ident_checked;
                            identities = identities && MPSet::face(MPSet::face(s, j), i).encode() == MPSet::face(MPSet::face(s, i), j - 1).encode();
                        }
            }
        T.check(unique && member, "S_<=3 on " + nm + ": every string once, and in S_k by the forward-intersection re-test");
        T.check(closed && identities && ident_checked > 0,
                "P1.11 S_<=3 on " + nm + ": faces close up and d_i d_j = d_{j-1} d_i (" + std::to_string(ident_checked) + " pairs)");
    }

    // ---- the mu-form ([MP] Sec. 1.2 (ii); audit adapted_form_check) ------------------------------------------
    uint32_t af_agree = 0;
    {
        const uint64_t a1 = srz.count_all(1), a2 = srz.count_all(2), mu1 = orbifold_mp::MuFormCount(rz, 1), mu2 = orbifold_mp::MuFormCount(rz, 2);
        const uint64_t b1 = sm7.count_all(1), b2 = sm7.count_all(2), nu1 = orbifold_mp::MuFormCount(m7, 1), nu2 = orbifold_mp::MuFormCount(m7, 2);
        std::cout << "  MP_MUFORM model=mp tri=M7  |S_1|,|S_2| (choice-free, all) = " << b1 << "," << b2 << ";  MP form sum|G_v(tau)|^k = " << nu1
                  << "," << nu2 << "  -> " << ((b1 == nu1 && b2 == nu2) ? "EQUAL" : "DIFFER") << "\n";
        std::cout << "  MP_MUFORM model=mp tri=R_Z |S_1|,|S_2| (choice-free, all) = " << a1 << "," << a2 << ";  MP form sum|G_v(tau)|^k = " << mu1
                  << "," << mu2 << "  -> " << ((a1 == mu1 && a2 == mu2) ? "EQUAL" : "DIFFER") << "\n";
        const bool c1 = a1 == 172 && a2 == 3264, c2 = mu1 == 172 && mu2 == 3264, c3 = a1 == mu1 && a2 == mu2;
        const bool c4 = b1 == 132 && b2 == 2800, c5 = nu1 == 124 && nu2 == 2704, c6 = !(b1 == nu1 && b2 == nu2);
        T.check(c1, "P1.3 choice-free |S_1|, |S_2| (all) on R_Z = 172, 3264");
        T.check(c2, "P1.3 mu-form sum |Gamma_v(tau)|^k on R_Z = 172, 3264");
        T.check(c3, "P1.3 on R_Z the two counts are EQUAL");
        T.check(c4, "P1.4 choice-free |S_1|, |S_2| (all) on M7 = 132, 2800");
        T.check(c5, "P1.4 mu-form on M7 = 124, 2704");
        T.check(c6, "P1.4 on M7 the two counts DIFFER");
        af_agree = c1 + c2 + c3 + c4 + c5 + c6;
    }
    std::cout << "ORACLE adapted_form_check.py cells=6 agree=" << af_agree << "\n";

    // ---- P1.12: R_zeta, recorded (OUTPUT, no pre-registered value) --------------------------------------------
    {
        const orbifold_mp::QuotientCounts q = orbifold_mp::Quotient(rzeta);
        const orbifold_mp::GoodResult g = orbifold_mp::Good(rzeta);
        std::cout << "  OUTPUT P1.12 model=mp tri=R_zeta quotient(V,E,F,chi)=(" << q.V << ", " << q.E << ", " << q.F << ", " << q.chi
                  << ") adapted(3,2,Z)=" << TF(orbifold_mp::Adapted(rzeta, 3, true)) << "," << TF(orbifold_mp::Adapted(rzeta, 2, true)) << ","
                  << TF(orbifold_mp::Adapted(rzeta, 0, true)) << " good=" << TF(g.good) << " good_fails_at=" << Join(g.fails_at)
                  << " collar=" << TF(orbifold_mp::Collar(rzeta)) << "\n";
    }

    // ---- (MP-2, F-MP1 F4) the vertex-list complex IS the ModelData complex: orbit counts = quotient counts ------
    // (SC) is decided on the descriptor's vertex lists; this ties them to the ModelData's orbit representatives
    // (no bigon, purity: every vertex / edge orbit of K' lies in a top simplex).  (AC) is NOT decided in C++: it is
    // guarded by the cohomology gate (OrbifoldMPNerveValidation's H^2 cells, F-MP1 F2) and by Lean L0b.
    {
        const std::vector<std::pair<const ModelData*, const Descriptor*>> md = {{&base, &m7}, {&dRZ, &rz}, {&dRZp, &rzp}, {&dZeta, &rzeta}};
        for (const auto& [d, D] : md) {
            const orbifold_mp::QuotientCounts q = orbifold_mp::Quotient(*D);
            const bool eq = d->reps[0].size() == q.V && d->reps[1].size() == q.E && d->reps[2].size() == q.F;
            std::cout << "  F4 model=mp tri=" << D->name << " ModelData reps (V,E,F)=(" << d->reps[0].size() << ", " << d->reps[1].size()
                      << ", " << d->reps[2].size() << ") quotient=(" << q.V << ", " << q.E << ", " << q.F << ")\n";
            T.check(eq, "F-MP1 F4: ModelData orbit representatives = the descriptor's quotient counts (V, E, F) on " + D->name);
        }
    }

    // ---- NC1: order-only "adapted" accepts a planted non-central stabiliser the subgroup form rejects ---------
    {
        const Mat22 U = Mat22::ST(), Tm = Mat22::T();
        // Face level: Gamma_{rho^2} = <ST> (order 6) beside the non-central sub-stabiliser T <(ST)^2> T^{-1} (order 3, 3 | 6).
        const CyclicGroup g6{U, 6}, g3 = orbifold_mp::Conjugate(Tm, CyclicGroup{U * U, 3});
        const bool order_only = orbifold_mp::AdaptedFace({g6, g3}, false), subgroup = orbifold_mp::AdaptedFace({g6, g3}, true);
        T.check(order_only && !subgroup && !orbifold_mp::IsSubgroup(g3, g6),
                "P1.13 NC1 (face): order-only accepts {<ST>, T<(ST)^2>T^-1} (3 | 6), the subgroup form rejects it");
        // Descriptor level: R_Z with v0's stabiliser planted as T^2 <ST> T^-2 (order 6, not inside Gamma_{rho^2}).
        Descriptor planted = rz;
        planted.name = "R_Z(planted v0)";
        const Mat22 T2 = Tm * Tm;
        planted.orbit_stab.at(planted.point.at(0).orbit) = orbifold_mp::Conjugate(T2, CyclicGroup{U, 6});
        const bool po = orbifold_mp::Adapted(planted, 0, false), ps = orbifold_mp::Adapted(planted, 0, true);
        T.check(po && !ps, "P1.13 NC1 (descriptor): on R_Z with a planted v0 stabiliser T^2<ST>T^-2 the order-only predicate says adapted, "
                           "the subgroup form says not");
        std::cout << "  NC1 planted: order_only_face=" << TF(order_only) << " subgroup_face=" << TF(subgroup) << " order_only_descriptor=" << TF(po)
                  << " subgroup_descriptor=" << TF(ps) << "\n";
    }
    // ---- NC2: a two-vertices-in-one-orbit order is refused ---------------------------------------------------
    {
        Descriptor planted = m7;
        planted.name = "M7(planted v0 v3 v4)";
        planted.top.push_back({0, 3, 4});   // v3 = T v0: two vertices in one orbit
        bool refused = false;
        try {
            (void)orbifold_mp::OrbitOrdered(planted, planted.top.back());
        } catch (const std::logic_error& ex) {
            refused = std::string(ex.what()).find("two vertices in one orbit") != std::string::npos;
        }
        T.check(!orbifold_mp::InvariantOrder(planted) && refused,
                "P1.14 NC2: a simplex v0 v3 v4 (v3 = T v0) has no invariant order -- InvariantOrder false, OrbitOrdered refuses");
    }

    // ---- the scratch dump --------------------------------------------------------------------------------------
    const uint32_t dump_fail = MaybeDump(rz, dump_rows, srz);
    T.check(dump_fail == 0, "ORBIFOLD_MP_DUMP (scratch hook) not refused");

    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "ORBIFOLD_MP OrbifoldMPRulesValidation checks=" << T.checks << " failing=" << T.failing << " wall_s=" << std::fixed
              << std::setprecision(2) << secs << std::defaultfloat << "\n";
    return T.failing == 0 ? 0 : 1;
}
