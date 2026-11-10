// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 22, MP-3b) The frozen MP-3b oracle tables -- see MP3bOracle.h.
#include "MP3bOracle.h"

#include <iterator>
#include <set>

#include "MP3Oracle.h"
#include "../../../ModArithmetic/EngineCheck.h"

namespace orbifold_mp {

#include "MP3bOracle.inc"

const MP3bPairRow& MP3bPair(const std::string& A, const std::string& B, uint32_t p, uint32_t k, uint32_t n) {
    for (const auto& r : kPairRowsB)
        if (A == r.A && B == r.B && r.p == p && r.k == k && r.n == n) return r;
    ENGINE_CHECK_MSG(false, "MP3bPair: no oracle row");
    return kPairRowsB[0];
}

bool MP3bHasDesc(const std::string& tri, const std::string& g) {
    for (const auto& r : kDescRowsB)
        if (tri == r.tri && g == r.group) return true;
    return false;
}

const MP3bDescRow& MP3bDesc(const std::string& tri, const std::string& g) {
    for (const auto& r : kDescRowsB)
        if (tri == r.tri && g == r.group) return r;
    ENGINE_CHECK_MSG(false, "MP3bDesc: no oracle row");
    return kDescRowsB[0];
}

const MP3bForcedRow& MP3bForced() { return kForcedB; }

const char* MP3bOracleSha256() { return kMP3bOracleSha256; }

bool MP3bOracleFileAgrees(uint32_t* rows, std::string* sha_file) {
    *rows = 0;
    *sha_file = "";
    std::string bytes;
    if (!ReadTrackedFile("docs/implementation/slot22/mp/mp3b/mp3b_oracle.out", &bytes)) return false;
    *sha_file = Sha256Hex(bytes);
    const std::multiset<std::string> want(std::begin(kDataLinesB), std::end(kDataLinesB));
    std::multiset<std::string> got;
    size_t pos = 0;
    while (pos < bytes.size()) {
        size_t e = bytes.find('\n', pos);
        if (e == std::string::npos) e = bytes.size();
        const std::string ln = bytes.substr(pos, e - pos);
        pos = e + 1;
        for (const std::string tag : {"  GROUP ", "  PAIR ", "  DESC ", "  FORCED_U_MINUS_I "})
            if (ln.compare(0, tag.size(), tag) == 0) {
                got.insert(ln);
                ++*rows;
                break;
            }
    }
    return got == want && *rows == want.size();
}

}  // namespace orbifold_mp
