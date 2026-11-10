// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef ORBIFOLD_MP_MP3_ORACLE_H
#define ORBIFOLD_MP_MP3_ORACLE_H

// ===========================================================================
// (hecke-extensions slot 22, MP-3) The frozen MP-3 oracle: the values of
// docs/implementation/slot22/mp/mp3/mp3_oracle.out (an independent Python
// computation, no project binary: the boundary line model, the explicit
// resolution of Gamma_d = <T> x <-I>, Mayer--Vietoris on the PERMUTATION form
// Fun(B\Gamma) (x) Sym^n of Shapiro), transcribed by mp3_oracle_to_inc.py into
// MP3Oracle.inc.  Committed with MP3_predictions.md before the first run of
// either MP-3 validation.  The validations re-read the tracked .out and check its
// sha256-free content row by row against these tables (MP3OracleFileAgrees).
// ===========================================================================

#include <cstdint>
#include <string>

namespace orbifold_mp {

struct MP3BdryRow {          // length H^q(Gamma_d; Sym^n Z/p^k), q = 0..4
    uint32_t p, k, n;
    int64_t L[5];
};
struct MP3GroupRow {         // dim H^q(B; Sym^n F_p), q = 0..4
    const char* group;
    uint32_t index, p, n;
    int64_t H[5];
};
struct MP3PairRow {          // the pair (A, B): H(A), H(B), rank res^A_B, H(A, B) (relative), q = 0..4
    const char* A;
    const char* B;
    uint32_t index, p, n;
    int64_t HA[5], HB[5], rank[5], rel[5];
};
struct MP3DescRow {          // the B-descriptor on M7
    const char* group;
    uint32_t index;
    bool minus_I;
    uint32_t vertex_orbits, top_orbits;
    uint64_t nd[4];          // |S_0..3^nd| when -I in B, else 0
};

// |S_k(dX)| (all, non-degenerate), k = 0..4.
uint64_t MP3BdryCount(uint32_t k, bool nondegenerate);
const MP3BdryRow& MP3Bdry(uint32_t p, uint32_t k, uint32_t n);                    // ENGINE_CHECK present
const MP3GroupRow& MP3Group(const std::string& g, uint32_t p, uint32_t n);
const MP3PairRow& MP3Pair(const std::string& A, const std::string& B, uint32_t p, uint32_t n);
const MP3DescRow& MP3Desc(const std::string& g);
// The tracked mp3_oracle.out re-read (path relative to the source tree baked into the binary) equals the tables:
// *rows = rows compared.  False if the file is missing or any row differs.
bool MP3OracleFileAgrees(uint32_t* rows);
// The sha256 of the transcribed mp3_oracle.out, FROZEN in MP3Oracle.inc at transcription time.
const char* MP3OracleSha256();
// (MP-3b, F-MP3 F2) The sha256 of the tracked mp3_oracle.out as read now ("" if unreadable); the validations print
// it beside the frozen value and gate their equality.
std::string MP3OracleFileSha256();
// (MP-3b) Shared helpers: read a tracked file (path relative to the source tree baked into the binary) byte for byte;
// the lower-case hex sha256 of a byte string (nf::Sha256, FIPS 180-4).
bool ReadTrackedFile(const std::string& rel_path, std::string* bytes);
std::string Sha256Hex(const std::string& bytes);

}  // namespace orbifold_mp

#endif
