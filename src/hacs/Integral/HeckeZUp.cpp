// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HeckeZUp -- slot-26 Z1 (UP-Z (a)): see the header.

#include "HeckeZUp.h"

#include "Descent/DescentModuleExportZ.h"
#include "IntegralCharpolyCrt.h"
#include "Descent/IsoTreeCarrierZ.h"
#include "../NumberField/Sha256.h"
#include "../../ModArithmetic/EngineCheck.h"

#include <algorithm>
#include <fstream>
#include <iomanip>

#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <sstream>
#include <vector>

namespace heckez {
namespace up {

namespace {
int32_t CertValuation(const Factored& F, uint64_t ell) {   // = HeckeZRecord.cpp CertVal
    for (const auto& pp : F.factors)
        if (pp.q == ell && pp.proof != "probable") return static_cast<int32_t>(pp.e);
    return F.complete() ? 0 : -1;
}
}  // namespace

std::vector<uint32_t> PrimesDividing(uint32_t N) {
    std::vector<uint32_t> u;
    for (uint32_t p = 2; p <= N; ++p)
        if (N % p == 0 && integral::IsPrimeU32(p)) u.push_back(p);
    return u;
}

std::string ThetaOpsText(const std::vector<uint32_t>& ops, uint32_t nU) {
    std::ostringstream o;
    for (size_t i = 0; i < ops.size(); ++i) o << (i ? "," : "") << (i + nU >= ops.size() ? "U_" : "T_") << ops[i];
    return o.str();
}

int32_t RationalThetaIndex(const RationalResult& Q, uint64_t ell) {
    for (uint32_t j : Q.discriminants_used) {
        const Factored& F = Q.thetas[j].disc;
        if (!F.complete()) continue;
        if (CertValuation(F, ell) == 0) return static_cast<int32_t>(j);
    }
    return -1;
}

FullG1Words FullG1Rule(bool tag_non_reduced, bool tag_split, const RationalResult* Q, bool route_refused, bool rss_ceiling, bool eisup_scope) {
    FullG1Words W;
    if (tag_non_reduced) {   // (d')(ii) the non-reduced branch: T (x) Q is not reduced, deg f_theta < d for every theta
        W.C1 = "refused(non_reduced)";
        W.reduced_Q = "non_reduced_certified";
        W.theta_status = "not_run";
        W.components = "not_run";
        W.C2 = "not_run";
        return W;
    }
    W.run_search = true;
    if (!Q) return W;   // before the search
    if (route_refused || rss_ceiling) {   // (R2) no Z1 route served: pending, never open(cap)
        const std::string w = rss_ceiling ? "blocked(export)" : "blocked(U_p)";
        W.C1 = W.reduced_Q = W.theta_status = W.C2 = W.components = w;   // ([C] 1.26.0-pre.4 W4) components too
        return W;
    }
    const bool cert = Q->c1_status == "certified" && Q->theta_index >= 0;
    W.C1 = cert ? "certified" : "open(cap)";
    W.reduced_Q = cert || tag_split ? "true" : "undecided";
    W.theta_status = cert ? std::string() : "open(cap)";
    W.components = cert ? (eisup_scope ? "certified" : "blocked(U_p)") : "not_run";   // (HZ-EISUP) certified in its scope
    W.C2 = cert ? std::string() : "partial(unfactored)";
    W.disc_fields = cert;
    return W;
}

FullG3Words FullG3Rule(bool tag_bound, bool all_tags_bound, bool tag_ran, bool all_tags_ran) {
    FullG3Words W;
    W.tag_status = !tag_ran ? "blocked(U_p)" : tag_bound ? "upper_bound_certified" : "not_run";
    W.ct_status = !all_tags_ran ? "blocked(U_p)" : all_tags_bound ? "upper_bound_certified" : "not_run";
    W.ct_lower = !all_tags_ran ? "blocked(U_p)" : "not_run";
    return W;
}

namespace {
std::string Sha256Of(const std::string& s) {
    nf::Sha256 h;
    h.update(s.data(), s.size());
    return h.hex();
}
std::string HexFnv(uint64_t v) {
    std::ostringstream o;
    o << "0x" << std::hex << std::setw(16) << std::setfill('0') << v;
    return o.str();
}
}  // namespace

std::vector<std::string> WriteUpMatrixFiles(uint32_t N, uint32_t n, const std::string& record_id, const std::string& record_fnv,
                                            const std::string& dir, std::vector<std::string>* refused, const UpDumpOptions& opt) {
    namespace mx = integral_descent::modexport;
    ENGINE_CHECK_MSG(N > 1, "WriteUpMatrixFiles: a full record at N > 1");
    std::vector<std::string> out;
    for (uint32_t p : PrimesDividing(N)) {
        mx::ModuleZ2Options zo;
        zo.allow_bad_primes = true;
        zo.threads = opt.threads;
        zo.step1.export_opt.allow_bad_primes = true;
        // F over Z by the free-model U_p (Z1); tau and M from the exports at every record torsion prime, 2 and 3 by the tree route (Z2);
        // where a read refuses, the file says so and carries F only
        const integral_descent::freez::IsoTreeCarrierZ car(N, n);
        const mx::FreeBlockRun run = mx::RunFreeBlock(car, {p}, zo);
        if (!run.refusal.empty()) {
            if (refused) refused->push_back("U_" + std::to_string(p) + ": the free block: " + run.refusal);
            continue;
        }
        const mx::ModuleZ Z = mx::AssembleModuleZ2(N, n, {p}, run, zo);
        const bool tm = Z.ok && Z.free_block == "exact";
        const uint32_t r = run.FB.B.cols, t = tm ? static_cast<uint32_t>(Z.d.size()) : 0;
        std::ostringstream o;
        o << "schema=hz_up_matrix/v1\nN=" << N << "\nn=" << n << "\np=" << p << "\nrecord_id=" << record_id << "\nrecord_fnv1a64=" << record_fnv
          << "\noperator=U_" << p << "^(N) = [Gamma_0(N) diag(1," << p << ") Gamma_0(N)], the adjoint of the classical U_p on p-old spaces "
          << "(prop:levelconehecke (3)(d)); F by the free-model U_p (prop:strictliftN item (9), [C] 1.26.0-pre.3)"
          << "\nbasis=export_retraction/v1 + " << run.G.basis_rule << "\nbasis_fnv=" << HexFnv(run.G.basis_fnv) << "\nr=" << r
          << "\nproof=" << (run.G.proof.empty() ? std::string("-") : run.G.proof[0]) << "\nF_fnv=" << (run.G.F_fnv.empty() ? std::string("-") : HexFnv(run.G.F_fnv[0]));
        if (tm) {
            o << "\ntau_M=served\nt=" << t << "\ntorsion_moduli=";
            for (uint32_t i = 0; i < t; ++i) o << (i ? "," : "") << Z.d[i];
            o << "\ntorsion_ells=";
            for (uint32_t i = 0; i < t; ++i) o << (i ? "," : "") << Z.tor_ell[i];
            o << "\ntau\n";
            for (uint32_t i = 0; i < t; ++i) {
                for (uint32_t j = 0; j < t; ++j) o << (j ? " " : "") << Z.tau[0][size_t(i) * t + j];
                o << "\n";
            }
            o << "M\n";
            for (const mx::LocalPartZ& P : Z.local) {
                if (P.t == 0) continue;
                ENGINE_CHECK_MSG(!P.M_global.empty() && P.M_global[0].size() == size_t(P.t) * r, "WriteUpMatrixFiles: M_global missing");
                for (uint32_t i = 0; i < P.t; ++i) {
                    for (uint32_t j = 0; j < r; ++j) o << (j ? " " : "") << P.M_global[0][size_t(i) * r + j];
                    o << "\n";
                }
            }
        } else {
            std::string w = Z.refusal;
            for (char& c : w)
                if (c == '\n') c = ' ';
            o << "\ntau_M=blocked(U_p): " << w << "\n";
        }
        o << "F\n";
        ENGINE_CHECK_MSG(!run.G.F.empty() && run.G.F[0].size() == size_t(r) * r, "WriteUpMatrixFiles: F missing");
        for (uint32_t i = 0; i < r; ++i) {
            for (uint32_t j = 0; j < r; ++j) o << (j ? " " : "") << run.G.F[0][size_t(i) * r + j].to_string();
            o << "\n";
        }
        std::string text = o.str();
        text += "sha256=" + Sha256Of(text) + "\n";
        const std::string name = record_id + ".Up" + std::to_string(p) + ".mat";
        std::ofstream f(dir + "/" + name, std::ios::binary | std::ios::trunc);
        f << text;
        ENGINE_CHECK_MSG(static_cast<bool>(f), "WriteUpMatrixFiles: cannot write the dump file");
        out.push_back(name);
    }
    std::sort(out.begin(), out.end());
    return out;
}

bool UpMatrixShaOk(const std::string& text) {
    const std::string key = "sha256=";
    if (text.size() < key.size() + 65 || text.back() != '\n') return false;
    const size_t at = text.size() - 65 - key.size();
    if (text.compare(at, key.size(), key) != 0 || (at > 0 && text[at - 1] != '\n')) return false;
    return Sha256Of(text.substr(0, at)) == text.substr(at + key.size(), 64);
}

namespace {
struct LeverRow {
    const char* name;
    const char* on;
    const char* off;
};
// the twelve descent levers of the rollback register (validation-catalogue.md, "Rollback-lever register"); policy Exact01, empty =
// unset, as each module's EnvLever (DescentTree.cpp, DescentTreePair.cpp, DescentAssembly.cpp, DescentSmith.cpp, SmithPar.cpp,
// DescentParAsm.cpp); _SMITH_PANEL is the read-only width knob (an integer in [1, 4096], else auto)
const LeverRow kLevers[] = {
    {"HECKE_DESCENT_TREE", "tree", "dense"},          {"HECKE_DESCENT_TREE_PAIR", "tree", "dense"},
    {"HECKE_DESCENT_GOOD_SPARSE", "sparse", "shipped"}, {"HECKE_DESCENT_GOOD_DENSE", "dense", "sparse"},
    {"HECKE_DESCENT_SMITH_BLOCKED", "blocked", "oracle"}, {"HECKE_DESCENT_SMITH_MT", "threaded", "serial"},
    {"HECKE_DESCENT_SMITH_F2_MT", "threaded", "serial"}, {"HECKE_DESCENT_SMITH_PANEL", nullptr, nullptr},
    {"HECKE_DESCENT_PAR_ASM", "on", "off"},            {"HECKE_DESCENT_PAR_ELL", "on", "off"},
    {"HECKE_DESCENT_EMBED_MAP", "on", "off"},          {"HECKE_DESCENT_PAR_LN", "on", "off"},
};
}  // namespace

std::string DescentEffectiveLine() {
    // each lever's own reading (route, source), then the composition the modules apply (audit S3): HECKE_DESCENT_TREE=0 makes the pair
    // dense too (DescentTreePair.cpp); HECKE_DESCENT_GOOD_SPARSE=0 restores the shipped good prime, so GOOD_DENSE does not apply.
    // GOOD_DENSE's policy reads set-but-empty as invalid (Empty::Invalid, DescentAssembly.cpp); every other lever reads it as unset.
    const size_t L = sizeof(kLevers) / sizeof(kLevers[0]);
    std::vector<std::string> route(L), source(L);
    for (size_t i = 0; i < L; ++i) {
        const LeverRow& R = kLevers[i];
        const char* e = std::getenv(R.name);
        const bool empty_invalid = std::string(R.name) == "HECKE_DESCENT_GOOD_DENSE";
        const bool set = e && (*e || empty_invalid);
        source[i] = set ? "env" : "default";
        if (!R.on) {   // the panel width
            route[i] = "auto";
            if (set) {
                char* end = nullptr;
                errno = 0;
                const long long v = std::strtoll(e, &end, 10);
                if (end != e && *end == '\0' && errno == 0 && v >= 1 && v <= 4096) route[i] = std::to_string(v);
                else source[i] = "env-invalid";
            }
        } else {
            route[i] = R.on;
            if (set) {
                const std::string v(e);
                if (v == "0") route[i] = R.off;
                else if (v != "1") source[i] = "env-invalid";
            }
        }
    }
    auto idx = [&](const char* n) {
        for (size_t i = 0; i < L; ++i)
            if (std::string(kLevers[i].name) == n) return i;
        return L;
    };
    const size_t tree = idx("HECKE_DESCENT_TREE"), pair = idx("HECKE_DESCENT_TREE_PAIR"), gs = idx("HECKE_DESCENT_GOOD_SPARSE"),
                 gd = idx("HECKE_DESCENT_GOOD_DENSE");
    if (route[tree] == "dense" && route[pair] == "tree") { route[pair] = "dense"; source[pair] += "+composed(TREE=0)"; }
    if (route[gs] == "shipped") { route[gd] = "shipped"; source[gd] += "+composed(GOOD_SPARSE=0)"; }
    std::ostringstream o;
    o << "EFFECTIVE HeckeZ descent";
    for (size_t i = 0; i < L; ++i) o << " " << kLevers[i].name << "=route=" << route[i] << ",source=" << source[i];
    return o.str();
}

void PrintDescentEffectiveOnce() {
    static std::once_flag once;
    std::call_once(once, [] { std::cout << DescentEffectiveLine() << "\n"; });
}

}  // namespace up
}  // namespace heckez
