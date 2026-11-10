// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 17, L17-2) HeckeSatCertDumpValidation -- see the header.
// Validation-only, env-interlocked certificate dump (owner ruling Lean Q14): a no-op
// unless HECKE_DUMP_SAT_CERT names an existing directory outside the source tree.
#include "HeckeSatCertDumpValidation.h"
#include "HeckeRingHeckeAlgebra.h"
#include "../Util/CellList.h"

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>
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

struct DumpCell {
    uint32_t n = 0;
    uint64_t ell = 0;
    uint32_t kmax = 0;
};

bool IsPrime(uint64_t v) {
    if (v < 2) return false;
    for (uint64_t d = 2; d * d <= v; ++d)
        if (v % d == 0) return false;
    return true;
}

bool FitsU64(uint64_t ell, uint32_t k) {
    unsigned __int128 r = 1;
    for (uint32_t i = 0; i < k; ++i) {
        r *= ell;
        if (r >> 64) return false;
    }
    return true;
}

// "n:ell:kmax,..." -- n even in [2, 2^20], ell prime in [5, 2^31), kmax in [2, 12] with ell^kmax < 2^64.
// Returns false (and names the position) on the first malformed cell or on an empty list.  (S24 CL-6)
// Util/CellList.h's Style::SatCert grammar, byte for byte (slot-17 WP-8: a trailing ',' or ':' is
// refused by position; HeckeSmallCarriesValidation (3) pins the texts).
bool ParseCells(const std::string& s, std::vector<DumpCell>* out, std::string* why) {
    namespace cl = oc_util::cells;
    cl::Grammar g;
    g.style = cl::Style::SatCert;
    g.fields = 3;
    g.max_digits = 12;
    g.ranges = {{2, 1u << 20}, {5, (1ull << 31) - 1}, {2, 12}};
    g.accept = [](const cl::Cell& v) { return v[0] % 2 == 0 && IsPrime(v[1]) && FitsU64(v[1], static_cast<uint32_t>(v[2])); };
    g.want = "want n:ell:kmax, n even, ell >= 5 prime, 2 <= kmax <= 12";
    std::vector<cl::Cell> cells;
    if (!cl::ParseStrict(s, g, cells, *why)) return false;
    for (const cl::Cell& c : cells) out->push_back({static_cast<uint32_t>(c[0]), c[1], static_cast<uint32_t>(c[2])});
    return true;
}

std::string Join(const std::vector<uint64_t>& v) {
    std::ostringstream os;
    for (size_t i = 0; i < v.size(); ++i) os << (i ? " " : "") << v[i];
    return os.str();
}

// One block -> one file.  Returns the empty string on success, the reason otherwise.
std::string WriteBlock(const std::string& path, const hecke_ext::ringalg::LoewyCell& c, const hecke_ext::ringalg::LoewyBlock& b) {
    using namespace hecke_ext::ringalg;
    const uint32_t K = b.K_stab;
    const OpModule N = RunModule(c.n, c.ell, K, c.primes);
    const Block B = Localize(N, b.system);
    const OpModule& L = B.local;
    if (L.is_digit()) return "digit_module";
    if (L.rank() != b.free_rank) return "rank_mismatch";
    for (uint32_t e : L.exps)
        if (e != K) return "block_not_free_at_K";
    if (L.op.size() != c.primes.size()) return "op_count";
    std::ofstream os(path);
    if (!os) return "open_failed";
    os << "SATCERT 1\n";
    os << "# HeckeSatCertDumpValidation (L17-2): a torsion-free stable block e_m H^1(Gamma; Sym^n Z/ell^K) = (Z/ell^K)^f,\n";
    os << "# T_q on it, column convention (T b_j = sum_i A_ij b_i), entries mod ell^K; CertifyStable passed\n";
    os << "n " << c.n << "\n";
    os << "ell " << c.ell << "\n";
    os << "K " << K << "\n";
    os << "f " << b.free_rank << "\n";
    os << "rank_T " << b.rank_T << "\n";
    os << "D " << b.D << "\n";
    os << "primes " << Join(c.primes) << "\n";
    os << "system " << Join(b.system) << "\n";
    const AlgebraReading* T = b.last_T();
    os << "A1_hilbert " << HilbertText(b.A1.hilbert) << "\n";
    os << "AK_hilbert " << (T ? HilbertText(T->hilbert) : std::string("-")) << "\n";
    os << "verdict " << b.verdict << "\n";
    os << "stable_cert " << b.stable_cert << "\n";
    const uint32_t f = L.rank();
    for (size_t i = 0; i < L.op.size(); ++i) {
        const auto& M = L.op[i];
        if (M.n() != f) return "op_shape";
        os << "gen " << c.primes[i] << "\n";
        for (uint32_t r = 0; r < f; ++r) {
            os << "row";
            for (uint32_t s = 0; s < f; ++s) os << ' ' << M.at(r, s);
            os << "\n";
        }
    }
    os << "end\n";
    return os ? std::string() : std::string("write_failed");
}

} // namespace

// slot-17 WP-8: the cell-list grammar of HECKE_DUMP_SAT_CERT_CELLS, public so its refusals have a negative test.
bool HeckeSatCertDumpParseCells(const std::string& spec, std::string* why) {
    std::vector<DumpCell> cells;
    return ParseCells(spec, &cells, why);
}

uint32_t HeckeSatCertDumpValidation() {
    const char* env = std::getenv("HECKE_DUMP_SAT_CERT");
    if (env == nullptr || *env == '\0') {
        std::cout << "[SKIP] HeckeSatCertDumpValidation: HECKE_DUMP_SAT_CERT not set (validation-only dump hook, L17-2)\n";
        return 0;
    }
    const std::string dir(env);
    if (!IsDirectory(dir)) {
        std::cout << "HeckeSatCertDumpValidation: REFUSED: HECKE_DUMP_SAT_CERT=" << dir << " is not an existing directory\n";
        return 1;
    }
    const std::string real_dir = RealPath(dir);
#ifdef OC_SOURCE_DIR
    const std::string real_src = RealPath(OC_SOURCE_DIR);
#else
    const std::string real_src;
#endif
    if (real_dir.empty() || IsInside(real_dir, real_src)) {
        std::cout << "HeckeSatCertDumpValidation: REFUSED: HECKE_DUMP_SAT_CERT=" << dir << " lies inside the source tree (" << real_src
                  << "); dump into scratch only\n";
        return 1;
    }
    const char* cenv = std::getenv("HECKE_DUMP_SAT_CERT_CELLS");
    if (cenv && !*cenv) {   // slot-17 WP-8: set but empty is a refusal, never the silent default
        std::cout << "HeckeSatCertDumpValidation: REFUSED: HECKE_DUMP_SAT_CERT_CELLS: empty value (unset it for the default)\n";
        return 1;
    }
    const std::string spec = cenv ? std::string(cenv) : std::string("168:13:2,220:17:2,258:37:2");
    std::vector<DumpCell> cells;
    std::string why;
    if (!ParseCells(spec, &cells, &why)) {
        std::cout << "HeckeSatCertDumpValidation: REFUSED: HECKE_DUMP_SAT_CERT_CELLS: " << why << "\n";
        return 1;
    }
    std::cout << oc_util::cells::EffectiveLine("HeckeSatCertDumpValidation", "DUMP", cells.size()) << "\n";

    uint32_t files = 0, skipped = 0, failures = 0;
    for (const DumpCell& dc : cells) {
        hecke_ext::ringalg::LoewyOptions opts;
        opts.kmax = dc.kmax;
        opts.certify = true;
        hecke_ext::ringalg::LoewyCell c;
        try {
            c = hecke_ext::ringalg::ComputeRingLoewyCell(dc.n, dc.ell, opts);
        } catch (const std::exception& e) {
            std::cout << "SAT_CERT_DUMP n=" << dc.n << " ell=" << dc.ell << " FAILED cell: " << e.what() << "\n";
            ++failures;
            continue;
        }
        uint32_t k = 0, cell_files = 0;
        for (const auto& b : c.blocks) {
            ++k;
            if (b.verdict != "stable" || b.stable_cert != "certified" || !b.torsion.empty() || b.free_rank == 0) {
                ++skipped;
                continue;
            }
            const std::string path = real_dir + "/satcert_n" + std::to_string(c.n) + "_ell" + std::to_string(c.ell) + "_b" + std::to_string(k) + ".txt";
            std::string err;
            try {
                err = WriteBlock(path, c, b);
            } catch (const std::exception& e) {
                err = std::string("exception: ") + e.what();
            }
            std::vector<uint64_t> sys(b.system.begin(), b.system.end());
            std::cout << "SAT_CERT_DUMP file=" << path << " n=" << c.n << " ell=" << c.ell << " system=(" << Join(sys) << ") K=" << b.K_stab
                      << " f=" << b.free_rank << " rank_T=" << b.rank_T << " A1_hilbert=" << hecke_ext::ringalg::HilbertText(b.A1.hilbert)
                      << (err.empty() ? std::string(" OK") : " FAILED " + err) << "\n";
            if (err.empty()) {
                ++files;
                ++cell_files;
            } else {
                ++failures;
            }
        }
        if (cell_files == 0) {
            std::cout << "SAT_CERT_DUMP n=" << c.n << " ell=" << c.ell << " FAILED: no torsion-free certified stable block\n";
            ++failures;
        }
    }
    std::cout << "HeckeSatCertDumpValidation: " << files << " files, " << skipped << " blocks skipped, " << failures << " failing\n";
    return failures == 0 ? 0 : 1;
}
