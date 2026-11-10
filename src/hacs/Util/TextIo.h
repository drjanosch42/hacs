// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Util/TextIo.h -- the hex formatter and the whole-file reader, ONE definition
// each (S24 CL-1; CODEAUDIT A3, F11; docs/implementation/slot24/CLEANUP_plan_1030.md)
// ===========================================================================
//
// HexU64.  About 28 copies (HexU64 / Hex64 / Hex16 / Hex / ToHex) print in
// three styles, and golden / record / transcript texts depend on the style,
// so each call site keeps its own:
//   HexU64(v)                 "0x" + 16 lower-case digits, zero-padded (the most common:
//                             RegisterHeckeExtensionVerbs Hex64, DescentRecord Hex16, the
//                             Console copies, atlas_index::Hex16, ...);
//   HexU64(v, false)          16 digits, no prefix (HeckeExtReport, CampaignExt,
//                             CampaignRing Hex16, HeckeExtRamanujanValidation, ...);
//   HexU64(v, true, false)    "0x" + the minimal digits, no padding (BenchDriver Hex).
// (GoldenHarness's Hex64 appends "ull"; it is the golden table's own writer and stays.)
//
// ReadFile.  13+ copies read a whole file with ifstream(binary) << rdbuf().
// Two shapes exist and both are kept:
//   bool ReadFile(path, out)  false iff the stream does not open (Console copies);
//   std::string ReadFile(path)  "" when it does not open (Campaign copies, Slurp).
// The body is the copies' verbatim, so a directory, which opens on Linux, reads
// as "" with true -- as before.  (HeckeAtlasCli's throwing copy keeps its own
// text: `if (!ReadFile(p, s)) throw ...`.)
// Header-only, STL only.
// ===========================================================================
#pragma once
#ifndef OC_UTIL_TEXT_IO_H
#define OC_UTIL_TEXT_IO_H

#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>

namespace oc_util {

inline std::string HexU64(uint64_t v, bool prefix0x = true, bool pad16 = true) {
    static const char* const kHex = "0123456789abcdef";
    char buf[16];
    int n = 0;
    if (pad16) {
        for (int i = 0; i < 16; ++i) buf[15 - i] = kHex[(v >> (4 * i)) & 0xfu];
        n = 16;
    } else {
        char tmp[16];
        do {
            tmp[n++] = kHex[v & 0xfu];
            v >>= 4;
        } while (v != 0);
        for (int i = 0; i < n; ++i) buf[i] = tmp[n - 1 - i];
    }
    return (prefix0x ? std::string("0x") : std::string()) + std::string(buf, static_cast<size_t>(n));
}

inline bool ReadFile(const std::string& path, std::string& out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    std::ostringstream o;
    o << in.rdbuf();
    out = o.str();
    return true;
}

inline std::string ReadFile(const std::string& path) {
    std::string s;
    return ReadFile(path, s) ? s : std::string();
}

} // namespace oc_util

#endif // OC_UTIL_TEXT_IO_H
