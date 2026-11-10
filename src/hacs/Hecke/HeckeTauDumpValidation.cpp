// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 16, L16-4) HeckeTauDumpValidation -- see the header.
// Validation-only, env-interlocked table dump (owner ruling Lean Q14): a no-op
// unless HECKE_DUMP_TAU names an existing directory outside the source tree.
#include "HeckeTauDumpValidation.h"

#include "HeckeBoundaryDatum.h"
#include "HeckeCosetTables.h"
#include "HeckePairHomotopy.h"
#include "HeckeSettings.h"
#include "HeckeTauTable.h"
#include "SL2ZTessellationTables.h"
#include "ExactInt.h"
#include "../LocalSystemSL2Z.h"      // CreateSL2ZFundamentalDomainTriangulation
#include "../SimplicialCohomology.h" // SimplicialComplex
#include "../../ModArithmetic/FlatMatrix.h"
#include "../../ModArithmetic/Fnv.h"

#include <climits>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <utility>
#include <vector>

namespace {

bool IsDirectory(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

std::string RealPath(const std::string& p) {
    char buf[PATH_MAX];
    if (::realpath(p.c_str(), buf) == nullptr) return std::string();
    return std::string(buf);
}

// True iff `inner` equals `outer` or lies below it (both canonical paths).
bool IsInside(const std::string& inner, const std::string& outer) {
    if (outer.empty() || inner.size() < outer.size()) return false;
    if (inner.compare(0, outer.size(), outer) != 0) return false;
    return inner.size() == outer.size() || inner[outer.size()] == '/';
}

void PutMat(std::ostream& os, const hecke::Mat22& g) {
    os << g.a << ' ' << g.b << ' ' << g.c << ' ' << g.d;
}

// Exactly HeckePairHomotopy::tau_table_fnv64() (feed_terms included), so the
// Lean side can compare HashScalar64 of it with the golden object
// tau_table_fnv64_p2_r1.
uint64_t TauFnv(const hecke::HeckeTauTable& tau, const SimplicialComplex& K, uint64_t N, uint32_t m) {
    engine::FnvHasher h;
    h.feed(N);
    h.feed(m);
    const auto& simplices = K.get_simplices();
    for (uint32_t q = 0; q < simplices.size(); ++q) {
        for (uint32_t sigma = 0; sigma < simplices[q].size(); ++sigma) {
            for (uint32_t j = 1; j <= m; ++j) {
                h.feed(q);
                h.feed(sigma);
                h.feed(j);
                const auto& terms = tau.tau(j, q, sigma);
                h.feed(terms.size());
                for (const auto& t : terms) {
                    h.feed(static_cast<uint64_t>(static_cast<int64_t>(t.coeff)));
                    h.feed(static_cast<uint64_t>(t.gamma.a));
                    h.feed(static_cast<uint64_t>(t.gamma.b));
                    h.feed(static_cast<uint64_t>(t.gamma.c));
                    h.feed(static_cast<uint64_t>(t.gamma.d));
                    h.feed(t.sigma_idx);
                }
            }
        }
    }
    return h.value();
}

bool WriteTauFile(const std::string& path, const SimplicialComplex& K,
                  const hecke::SL2ZTessellationTables& model,
                  const hecke::HeckeCosetTables& cosets, const hecke::HeckeTauTable& tau,
                  const std::string& comment =
                      "HeckeTauDumpValidation (L16-4): p = 2, nu = 1, the tau table of the golden ext cells' cone action") {
    std::ofstream os(path);
    if (!os) return false;
    const auto& simplices = K.get_simplices();
    const uint32_t m = cosets.num_cosets();
    os << "TAUDUMP 1\n";
    os << "# " << comment << "\n";
    os << "N " << cosets.N() << "\n";
    os << "m " << m << "\n";
    os << "g "; PutMat(os, cosets.g()); os << "\n";
    for (uint32_t i = 1; i <= m; ++i) { os << "s " << i << ' '; PutMat(os, cosets.s(i)); os << "\n"; }
    for (uint32_t q = 0; q < simplices.size(); ++q)
        for (uint32_t s = 0; s < simplices[q].size(); ++s) {
            os << "simplex " << q << ' ' << s;
            for (uint32_t v : simplices[q][s]) os << ' ' << v;
            os << "\n";
        }
    for (uint32_t q = 0; q < simplices.size(); ++q)
        for (uint32_t s = 0; s < simplices[q].size(); ++s)
            os << "orbitrep " << q << ' ' << s << ' ' << model.orbit_representative(q, s) << "\n";
    for (uint32_t q = 0; q < simplices.size(); ++q)
        for (uint32_t s1 = 0; s1 < simplices[q].size(); ++s1)
            for (uint32_t s2 = 0; s2 < simplices[q].size(); ++s2) {
                if (!model.same_groupoid_orbit(q, s1, s2)) continue;
                const auto& ts = model.transporter(q, s1, s2);
                const auto& ps = model.transporter_vertex_perms(q, s1, s2);
                for (size_t k = 0; k < ts.size(); ++k) {
                    os << "tr " << q << ' ' << s1 << ' ' << s2 << ' ';
                    PutMat(os, ts[k]);
                    os << ' ' << unsigned(ps[k][0]) << ' ' << unsigned(ps[k][1]) << ' ' << unsigned(ps[k][2]) << "\n";
                }
            }
    for (uint32_t q = 0; q < simplices.size(); ++q)
        for (uint32_t s = 0; s < simplices[q].size(); ++s)
            for (uint32_t i = 1; i <= m; ++i) {
                const auto& terms = tau.tau(i, q, s);
                os << "slot " << i << ' ' << q << ' ' << s << ' ' << terms.size() << "\n";
                for (const auto& t : terms) {
                    os << "term " << t.coeff << ' ';
                    PutMat(os, t.gamma);
                    os << ' ' << t.sigma_idx << "\n";
                }
            }
    os << "regime " << tau.solve_regime() << "\n";
    os << "fnv " << TauFnv(tau, K, cosets.N(), m) << "\n";
    os << "end\n";
    return static_cast<bool>(os);
}

void PutTerms(std::ostream& os, uint32_t j, uint32_t q, uint32_t sigma, const std::vector<hecke::ChainTerm>& terms) {
    os << "slot " << j << ' ' << q << ' ' << sigma << ' ' << terms.size() << "\n";
    for (const auto& t : terms) {
        os << "term " << t.coeff << ' ';
        PutMat(os, t.gamma);
        os << ' ' << t.sigma_idx << "\n";
    }
}

void PutCosetHeader(std::ostream& os, const hecke::HeckeCosetTables& cosets, uint32_t ell, uint32_t n) {
    os << "p " << cosets.p_hecke() << "\n";
    os << "nu " << cosets.r() << "\n";
    os << "ell " << ell << "\n";
    os << "nsym " << n << "\n";
    os << "N " << cosets.N() << "\n";
    os << "m " << cosets.num_cosets() << "\n";
    os << "g "; PutMat(os, cosets.g()); os << "\n";
    for (uint32_t i = 1; i <= cosets.num_cosets(); ++i) { os << "s " << i << ' '; PutMat(os, cosets.s(i)); os << "\n"; }
}

// (slot 18, L18-3a) The tau_bd table ([C, def:taubdry], obl:taubdry): the boundary simplices per
// degree, one stored chain per slot (j, q, sigma) in table_fnv64()'s walk order, the solve regime
// and HeckeBoundaryTauTable::table_fnv64() (= HeckePairHomotopy::tau_boundary_table_fnv64(), checked
// by the caller).  The model data (simplices, orbit reps, transporters) is the tau file's.
bool WriteTaubdFile(const std::string& path, const hecke::HeckeCosetTables& cosets,
                    const hecke::HeckeBoundaryTauTable& taub, uint32_t ell, uint32_t n) {
    std::ofstream os(path);
    if (!os) return false;
    os << "TAUBDDUMP 1\n";
    os << "# HeckeTauDumpValidation (L18-3a): the certified tau_bd table of HeckeCosetTables(p, nu, ell, n)\n";
    PutCosetHeader(os, cosets, ell, n);
    for (uint32_t q = 0; q < taub.num_degrees(); ++q)
        for (uint32_t sigma : taub.boundary_simplices(q)) os << "bsimplex " << q << ' ' << sigma << "\n";
    for (uint32_t q = 0; q < taub.num_degrees(); ++q)
        for (uint32_t sigma : taub.boundary_simplices(q))
            for (uint32_t j = 1; j <= taub.num_cosets(); ++j) PutTerms(os, j, q, sigma, taub.tau(j, q, sigma));
    os << "regime " << taub.solve_regime() << "\n";
    os << "fnv " << taub.table_fnv64() << "\n";
    os << "end\n";
    return static_cast<bool>(os);
}

// (slot 18, L18-3b) The eta table ([C, def:pairhomotopy], obl:eta): records_[q][local][j-1] in
// HeckePairHomotopy::table_fnv64()'s walk order (slot j q sigma: sigma a boundary q-simplex, the
// terms on (q+1)-simplices of K), the regime, the table hash and the two base-table hashes.
bool WriteEtaFile(const std::string& path, const hecke::HeckeCosetTables& cosets,
                  const hecke::HeckePairHomotopy& eta, uint32_t ell, uint32_t n) {
    std::ofstream os(path);
    if (!os) return false;
    os << "ETADUMP 1\n";
    os << "# HeckeTauDumpValidation (L18-3b): the certified pair homotopy eta of HeckeCosetTables(p, nu, ell, n)\n";
    PutCosetHeader(os, cosets, ell, n);
    for (uint32_t q = 0; q < eta.num_degrees(); ++q)
        for (uint32_t sigma : eta.boundary_simplices(q)) os << "bsimplex " << q << ' ' << sigma << "\n";
    for (uint32_t q = 0; q < eta.num_degrees(); ++q)
        for (uint32_t sigma : eta.boundary_simplices(q))
            for (uint32_t j = 1; j <= eta.num_cosets(); ++j) PutTerms(os, j, q, sigma, eta.eta(j, q, sigma));
    os << "regime " << eta.solve_regime() << "\n";
    os << "fnv " << eta.table_fnv64() << "\n";
    os << "tau_fnv " << eta.tau_table_fnv64() << "\n";
    os << "taubd_fnv " << eta.tau_boundary_table_fnv64() << "\n";
    os << "end\n";
    return static_cast<bool>(os);
}

// (slot 18, L18-3) HECKE_DUMP_TAU_PAIR="p:nu,...": the (p, nu) whose tau / tau_bd / eta tables are
// dumped.  Strict: p prime, 2 <= p <= 97, 1 <= nu <= 2, no duplicate; the first bad entry is refused
// by its 1-based position (set-but-empty and a trailing comma included).  Read-only on values.
bool ParsePairs(const std::string& spec, std::vector<std::pair<uint32_t, uint32_t>>* out, std::string* why) {
    size_t pos = 0, idx = 0;
    while (true) {
        ++idx;
        const size_t comma = spec.find(',', pos);
        const std::string tok = spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        const size_t colon = tok.find(':');
        const std::string a = colon == std::string::npos ? tok : tok.substr(0, colon);
        const std::string b = colon == std::string::npos ? std::string() : tok.substr(colon + 1);
        const auto num = [](const std::string& t, uint32_t lo, uint32_t hi, uint32_t* v) {
            if (t.empty() || t.size() > 3 || t.find_first_not_of("0123456789") != std::string::npos) return false;
            *v = static_cast<uint32_t>(std::stoul(t));
            return *v >= lo && *v <= hi;
        };
        const auto prime = [](uint32_t x) {
            if (x < 2) return false;
            for (uint32_t d = 2; d * d <= x; ++d)
                if (x % d == 0) return false;
            return true;
        };
        uint32_t p = 0, nu = 0;
        if (colon == std::string::npos || !num(a, 2, 97, &p) || !prime(p) || !num(b, 1, 2, &nu)) {
            *why = "entry " + std::to_string(idx) + " ('" + tok + "') malformed: need p:nu, p prime <= 97, 1 <= nu <= 2";
            return false;
        }
        for (const auto& c : *out)
            if (c.first == p && c.second == nu) {
                *why = "entry " + std::to_string(idx) + " ('" + tok + "') duplicate";
                return false;
            }
        out->emplace_back(p, nu);
        if (comma == std::string::npos) return true;
        pos = comma + 1;
    }
}

void PutRho(std::ostream& os, const std::string& tag, const std::array<uint32_t, 4>& key,
            const FlatMatrix& R) {
    os << "mat " << tag << ' ' << key[0] << ' ' << key[1] << ' ' << key[2] << ' ' << key[3] << "\n";
    for (uint32_t r = 0; r < R.get_rows(); ++r) {
        os << "row";
        for (uint32_t c = 0; c < R.get_cols(); ++c) os << ' ' << R.get_entry(r, c);
        os << "\n";
    }
}

bool WriteRhoFile(const std::string& path, const SimplicialComplex& K,
                  const hecke::SL2ZTessellationTables& model, uint32_t n, uint32_t ell, uint32_t& mats_out) {
    const hecke::HeckeCosetTables cosets(2, /*nu=*/1, ell, n);
    const hecke::HeckeTauTable tau(K, model, cosets, ell, n);
    if (tau.rho_pool_is_low_rank()) return false; // never at ell >= 5, p = 2
    std::ofstream os(path);
    if (!os) return false;
    os << "RHODUMP 1\n";
    os << "# HeckeTauDumpValidation (L16-4): rho_n over Z/ell, column convention, basis u_j = x^(n-j) y^j\n";
    os << "n " << n << "\n";
    os << "ell " << ell << "\n";
    mats_out = 0;
    const auto& pool = tau.rho_pool();
    for (uint32_t slot = 0; slot < pool.size(); ++slot) {
        const std::array<uint32_t, 4> key = tau.rho_pool_key(slot);
        PutRho(os, "pool", key, pool[slot]);
        ++mats_out;
    }
    std::vector<std::pair<std::string, hecke::Mat22>> named = {
        { "T", hecke::Mat22::T() }, { "S", hecke::Mat22::S() },
        { "ST", hecke::Mat22::ST() }, { "TS", hecke::Mat22::TS() } };
    for (uint32_t i = 1; i <= cosets.num_cosets(); ++i) {
        named.emplace_back("s" + std::to_string(i), cosets.s(i));
        named.emplace_back("delta" + std::to_string(i), cosets.delta(i));
    }
    for (const auto& kv : named) {
        const std::array<uint32_t, 4> key = kv.second.reduced_mod(ell);
        const FlatMatrix A(2, 2, ell, std::vector<uint32_t>{ key[0], key[1], key[2], key[3] });
        PutRho(os, kv.first, key, ComputeSymmetricPowerMatrixFM(A, n));
        ++mats_out;
    }
    os << "end\n";
    return static_cast<bool>(os);
}

// (slot 17, L17-4 repair) HECKE_DUMP_TAU_RHO_CELLS="n:ell,..." replaces the three default rho
// cells.  Strict: decimal fields only, n <= 1000, 5 <= ell < 2^32, no duplicate; the first bad
// cell is refused by its 1-based position.  Read-only on values: it selects which cells are dumped.
bool ParseRhoCells(const std::string& spec, std::vector<std::pair<uint32_t, uint32_t>>* out, std::string* why) {
    size_t pos = 0, idx = 0;
    while (true) {
        ++idx;
        const size_t comma = spec.find(',', pos);
        const std::string tok = spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        const size_t colon = tok.find(':');
        const std::string a = colon == std::string::npos ? tok : tok.substr(0, colon);
        const std::string b = colon == std::string::npos ? std::string() : tok.substr(colon + 1);
        const auto num = [](const std::string& t, uint64_t lim, uint64_t* v) {
            if (t.empty() || t.size() > 10 || t.find_first_not_of("0123456789") != std::string::npos) return false;
            *v = std::stoull(t);
            return *v <= lim;
        };
        uint64_t n = 0, ell = 0;
        if (colon == std::string::npos || !num(a, 1000, &n) || !num(b, 0xffffffffull, &ell) || ell < 5) {
            *why = "cell " + std::to_string(idx) + " ('" + tok + "') malformed: need n:ell, n <= 1000, 5 <= ell < 2^32";
            return false;
        }
        for (const auto& c : *out)
            if (c.first == n && c.second == ell) {
                *why = "cell " + std::to_string(idx) + " ('" + tok + "') duplicate";
                return false;
            }
        out->emplace_back(static_cast<uint32_t>(n), static_cast<uint32_t>(ell));
        if (comma == std::string::npos) return true;
        pos = comma + 1;
    }
}

} // namespace

uint32_t HeckeTauDumpValidation() {
    const char* env = std::getenv("HECKE_DUMP_TAU");
    if (env == nullptr || *env == '\0') {
        std::cout << "[SKIP] HeckeTauDumpValidation: HECKE_DUMP_TAU not set (validation-only dump hook, L16-4)\n";
        return 0;
    }
    const std::string dir(env);
    if (!IsDirectory(dir)) {
        std::cout << "HeckeTauDumpValidation: REFUSED: HECKE_DUMP_TAU=" << dir << " is not an existing directory\n";
        return 1;
    }
    const std::string real_dir = RealPath(dir);
#ifdef OC_SOURCE_DIR
    const std::string real_src = RealPath(OC_SOURCE_DIR);
#else
    const std::string real_src;
#endif
    if (real_dir.empty() || IsInside(real_dir, real_src)) {
        std::cout << "HeckeTauDumpValidation: REFUSED: HECKE_DUMP_TAU=" << dir
                  << " lies inside the source tree (" << real_src << "); dump into scratch only\n";
        return 1;
    }

    const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    const hecke::SL2ZTessellationTables model(K);
    std::vector<std::pair<uint32_t, uint32_t>> cells = { { 2, 5 }, { 10, 691 }, { 16, 13 } };
    const char* cenv = std::getenv("HECKE_DUMP_TAU_RHO_CELLS");
    if (cenv != nullptr) { // set-but-empty is refused (cell 1), as HECKE_DUMP_SAT_CERT_CELLS (slot-17 WP-8)
        cells.clear();
        std::string why;
        if (!ParseRhoCells(cenv, &cells, &why)) {
            std::cout << "HeckeTauDumpValidation: REFUSED: HECKE_DUMP_TAU_RHO_CELLS: " << why << "\n";
            return 1;
        }
        std::cout << "EFFECTIVE HeckeTauDumpValidation arm=RHO_CELLS cells=" << cells.size() << "\n";
    } else {
        std::cout << "EFFECTIVE HeckeTauDumpValidation arm=DUMP cells=" << cells.size() << "\n";
    }

    std::vector<std::pair<uint32_t, uint32_t>> pairs;
    const char* penv = std::getenv("HECKE_DUMP_TAU_PAIR");
    if (penv != nullptr) { // set-but-empty is refused (entry 1)
        std::string why;
        if (!ParsePairs(penv, &pairs, &why)) {
            std::cout << "HeckeTauDumpValidation: REFUSED: HECKE_DUMP_TAU_PAIR: " << why << "\n";
            return 1;
        }
        std::cout << "EFFECTIVE HeckeTauDumpValidation arm=PAIR tables=" << pairs.size() << "\n";
    }

    uint32_t failures = 0;
    {
        // The golden cone action's table (GoldenHarness E5-3): HeckeCosetTables(2, 1, ell, n),
        // HeckeTauTable(K, model, cosets, ell, n), default policy; integral, so (n, ell) = (2, 5)
        // stands for every golden cell (one hash, 0x747f629489327720, on all of them).
        const hecke::HeckeCosetTables cosets(2, /*nu=*/1, 5u, 2u);
        const hecke::HeckeTauTable tau(K, model, cosets, 5u, 2u);
        const uint32_t cert = tau.certify();
        const std::string path = real_dir + "/tau_p2.txt";
        const bool ok = cert == 0 && WriteTauFile(path, K, model, cosets, tau);
        std::cout << "TAU_DUMP file=" << path << " certify=" << cert << " regime=" << tau.solve_regime()
                  << " fnv=" << TauFnv(tau, K, cosets.N(), cosets.num_cosets())
                  << " golden_hash=0x" << std::hex << engine::HashScalar64(TauFnv(tau, K, cosets.N(), cosets.num_cosets()))
                  << std::dec << (ok ? " OK" : " FAILED") << "\n";
        if (!ok) ++failures;
    }
    for (const auto& c : cells) {
        const std::string path = real_dir + "/rho_n" + std::to_string(c.first) + "_ell" + std::to_string(c.second) + ".txt";
        uint32_t mats = 0;
        const bool ok = WriteRhoFile(path, K, model, c.first, c.second, mats);
        std::cout << "TAU_DUMP file=" << path << " mats=" << mats << (ok ? " OK" : " FAILED") << "\n";
        if (!ok) ++failures;
    }
    for (const auto& pn : pairs) {
        // The golden recipe (GoldenHarness RunExtTauCell / RunExtCell32's E5-3 block): cosets
        // (p, nu, ell, n), tau, tau_bd, eta, u_ell off.  The tables are integral and ell- / n-
        // independent (one hash on all 31 cells), so one cell stands for all: n = 2, ell = 5
        // (ell = 13 at p = 5, where the harness skips ell = p).
        const uint32_t ell = pn.first == 5 ? 13u : 5u;
        const uint32_t n = 2;
        hecke::UEllSetting u_off;
        u_off.mode = hecke::UEllMode::Off;
        const hecke::ScopedUEll u_guard(u_off, hecke::SettingSource::Spec);
        const hecke::HeckeCosetTables cosets(pn.first, pn.second, ell, n);
        const hecke::HeckeTauTable tau(K, model, cosets, ell, n);
        const hecke::HeckeBoundaryTauTable taub(K, model, cosets);
        const hecke::HeckePairHomotopy eta(tau, taub);
        const std::string suf = "_p" + std::to_string(pn.first) + "_r" + std::to_string(pn.second);
        const uint64_t tfnv = TauFnv(tau, K, cosets.N(), cosets.num_cosets());
        const bool walks = tfnv == eta.tau_table_fnv64() && taub.table_fnv64() == eta.tau_boundary_table_fnv64();
        const std::string ptau = real_dir + "/tau" + suf + ".txt";
        const std::string pbd = real_dir + "/taubd" + suf + ".txt";
        const std::string peta = real_dir + "/eta" + suf + ".txt";
        const std::string what = "HeckeCosetTables(" + std::to_string(pn.first) + ", " + std::to_string(pn.second) +
                                 ", " + std::to_string(ell) + ", " + std::to_string(n) + ")";
        const bool ok1 = walks && tau.certify() == 0 &&
                         WriteTauFile(ptau, K, model, cosets, tau, "HeckeTauDumpValidation (L18-3): the tau table of " + what);
        const bool ok2 = walks && taub.certify() == 0 && WriteTaubdFile(pbd, cosets, taub, ell, n);
        const bool ok3 = walks && eta.certify() == 0 && WriteEtaFile(peta, cosets, eta, ell, n);
        std::cout << std::hex << "TAU_DUMP file=" << ptau << " golden_hash=0x" << engine::HashScalar64(tfnv)
                  << (ok1 ? " OK" : " FAILED") << "\n"
                  << "TAU_DUMP file=" << pbd << " golden_hash=0x" << engine::HashScalar64(taub.table_fnv64())
                  << (ok2 ? " OK" : " FAILED") << "\n"
                  << "TAU_DUMP file=" << peta << " golden_hash=0x" << engine::HashScalar64(eta.table_fnv64())
                  << " eta_zero=" << (eta.eta_zero() ? 1 : 0) << (ok3 ? " OK" : " FAILED") << std::dec << "\n";
        failures += (ok1 ? 0u : 1u) + (ok2 ? 0u : 1u) + (ok3 ? 0u : 1u);
    }
    std::cout << "HeckeTauDumpValidation: " << (1 + cells.size() + 3 * pairs.size()) << " files, " << failures
              << " failing\n";
    return failures == 0 ? 0 : 1;
}
