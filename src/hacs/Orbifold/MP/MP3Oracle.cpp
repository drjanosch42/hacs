// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 22, MP-3) The frozen MP-3 oracle tables -- see MP3Oracle.h.
#include "MP3Oracle.h"

#include <cstdio>
#include <fstream>
#include <set>
#include <sstream>

#include "../../../ModArithmetic/EngineCheck.h"
#include "../../NumberField/Sha256.h"

namespace orbifold_mp {

#include "MP3Oracle.inc"

namespace {

std::string List(const int64_t* v, uint32_t k) {
    std::ostringstream os;
    os << "[";
    for (uint32_t i = 0; i < k; ++i) os << (i ? ", " : "") << v[i];
    os << "]";
    return os.str();
}

std::string Pad(const std::string& s, size_t w) { return s.size() >= w ? s : s + std::string(w - s.size(), ' '); }

std::string Num2(uint32_t x) {
    char b[16];
    std::snprintf(b, sizeof b, "%2u", x);
    return b;
}

}  // namespace

uint64_t MP3BdryCount(uint32_t k, bool nondegenerate) {
    ENGINE_CHECK_MSG(k <= 4, "MP3BdryCount: k <= 4");
    return kBdryCounts[k][nondegenerate ? 1 : 0];
}

const MP3BdryRow& MP3Bdry(uint32_t p, uint32_t k, uint32_t n) {
    for (const auto& r : kBdryRows)
        if (r.p == p && r.k == k && r.n == n) return r;
    ENGINE_CHECK_MSG(false, "MP3Bdry: no oracle row");
    return kBdryRows[0];
}

const MP3GroupRow& MP3Group(const std::string& g, uint32_t p, uint32_t n) {
    for (const auto& r : kGroupRows)
        if (g == r.group && r.p == p && r.n == n) return r;
    ENGINE_CHECK_MSG(false, "MP3Group: no oracle row");
    return kGroupRows[0];
}

const MP3PairRow& MP3Pair(const std::string& A, const std::string& B, uint32_t p, uint32_t n) {
    for (const auto& r : kPairRows)
        if (A == r.A && B == r.B && r.p == p && r.n == n) return r;
    ENGINE_CHECK_MSG(false, "MP3Pair: no oracle row");
    return kPairRows[0];
}

const MP3DescRow& MP3Desc(const std::string& g) {
    for (const auto& r : kDescRows)
        if (g == r.group) return r;
    ENGINE_CHECK_MSG(false, "MP3Desc: no oracle row");
    return kDescRows[0];
}

const char* MP3OracleSha256() { return kMP3OracleSha256; }

bool ReadTrackedFile(const std::string& rel_path, std::string* bytes) {
#ifdef OC_SOURCE_DIR
    const std::string path = std::string(OC_SOURCE_DIR) + "/" + rel_path;
#else
    const std::string path = rel_path;
#endif
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    std::ostringstream os;
    os << in.rdbuf();
    *bytes = os.str();
    return true;
}

std::string Sha256Hex(const std::string& bytes) {
    nf::Sha256 h;
    h.update(bytes.data(), bytes.size());
    return h.hex();
}

std::string MP3OracleFileSha256() {
    std::string bytes;
    if (!ReadTrackedFile("docs/implementation/slot22/mp/mp3/mp3_oracle.out", &bytes)) return "";
    return Sha256Hex(bytes);
}

bool MP3OracleFileAgrees(uint32_t* rows) {
    *rows = 0;
#ifdef OC_SOURCE_DIR
    const std::string path = std::string(OC_SOURCE_DIR) + "/docs/implementation/slot22/mp/mp3/mp3_oracle.out";
#else
    const std::string path = "docs/implementation/slot22/mp/mp3/mp3_oracle.out";
#endif
    std::ifstream in(path);
    if (!in) return false;
    std::set<std::string> lines;
    for (std::string ln; std::getline(in, ln);) lines.insert(ln);
    bool ok = true;
    auto want = [&](const std::string& s) {
        ++*rows;
        if (!lines.count(s)) ok = false;
    };
    {
        std::ostringstream os;
        os << "   [";
        for (uint32_t k = 0; k < 5; ++k) os << (k ? ", " : "") << "(" << kBdryCounts[k][0] << ", " << kBdryCounts[k][1] << ")";
        os << "]";
        want(os.str());
    }
    for (const auto& r : kBdryRows) {
        uint32_t m = 1;
        for (uint32_t i = 0; i < r.k; ++i) m *= r.p;
        const std::string ring = r.k == 1 ? "F_" + std::to_string(r.p) : "Z/" + std::to_string(m);
        want("  " + ring + " n=" + Num2(r.n) + " " + List(r.L, 5));
    }
    for (const auto& r : kGroupRows)
        want("  " + Pad(r.group, 8) + " index=" + Num2(r.index) + " l=" + std::to_string(r.p) + " n=" + std::to_string(r.n) + " " +
             List(r.H, 5));
    for (const auto& r : kPairRows)
        want("  " + std::string(r.B) + "<" + r.A + " [A:B]=" + std::to_string(r.index) + " l=" + std::to_string(r.p) + " n=" +
             std::to_string(r.n) + " H(A)=" + List(r.HA, 5) + " H(B)=" + List(r.HB, 5) + " rank_res=" + List(r.rank, 5) +
             " H(A,B)=" + List(r.rel, 5));
    for (const auto& r : kDescRows) {
        std::string nd = "None";
        if (r.minus_I) {
            const int64_t v[4] = {static_cast<int64_t>(r.nd[0]), static_cast<int64_t>(r.nd[1]), static_cast<int64_t>(r.nd[2]),
                                  static_cast<int64_t>(r.nd[3])};
            nd = List(v, 4);
        }
        want("  " + Pad(r.group, 8) + " index=" + Num2(r.index) + " -I_in=" + (r.minus_I ? "True" : "False") +
             " vertex_orbits=" + std::to_string(r.vertex_orbits) + " top_orbits=" + std::to_string(r.top_orbits) + " nd_S0..3=" + nd);
    }
    return ok;
}

}  // namespace orbifold_mp
