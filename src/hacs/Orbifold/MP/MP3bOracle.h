// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef ORBIFOLD_MP_MP3B_ORACLE_H
#define ORBIFOLD_MP_MP3B_ORACLE_H

// ===========================================================================
// (hecke-extensions slot 22, MP-3b) The frozen MP-3b oracle: the values of
// docs/implementation/slot22/mp/mp3b/mp3b_oracle.out (an independent Python
// computation, no project binary: H^q(Gamma; Fun(B\Gamma) (x) Sym^n Z) from the
// integral cocone of the Mayer--Vietoris restriction of <S> *_{<-I>} <ST>, lengths
// over Z/p^k by the integer Smith normal form + UCT and by a local-ring
// elimination (asserted equal; = mp3_oracle's F_p MV at k = 1); the pair sequence
// from the cone of Tot(res) and from 0 -> N_A -> N_B -> Q -> 0 (asserted equal)),
// transcribed by mp3b_oracle_to_inc.py into MP3bOracle.inc.  Committed with
// MP3b_predictions.md before the first run of the MP-3b arm.
// ===========================================================================

#include <cstdint>
#include <string>

namespace orbifold_mp {

struct MP3bGroupRow {        // length H^q(B; Sym^n Z/p^k), q = 0..3
    const char* group;
    uint32_t p, k, n;
    int64_t L[4];
};
struct MP3bPairRow {         // the pair B <= A over Z/p^k: H(A), H(B), length of the image of res, H(A, B), q = 0..3
    const char* A;
    const char* B;
    uint32_t index, p, k, n;
    int64_t HA[4], HB[4], rank[4], rel[4];
};
struct MP3bDescRow {         // the B-descriptor on a triangulation ("M7" / "R_Z")
    const char* tri;
    const char* group;
    uint32_t index;
    bool minus_I;
    uint32_t vertex_orbits, top_orbits;
    uint64_t nd[4];          // |S_0..3^nd| when -I in B, else 0
};
struct MP3bForcedRow {       // the forget-coset labels with u = -I (at least)
    const char* tri;
    const char* A;
    const char* B;
    uint32_t at_least;
};

const MP3bPairRow& MP3bPair(const std::string& A, const std::string& B, uint32_t p, uint32_t k, uint32_t n);   // ENGINE_CHECK present
bool MP3bHasDesc(const std::string& tri, const std::string& g);
const MP3bDescRow& MP3bDesc(const std::string& tri, const std::string& g);                                      // ENGINE_CHECK present
const MP3bForcedRow& MP3bForced();

// The tracked mp3b_oracle.out re-read: *rows = data lines in the file; true iff the set of its data lines equals the
// transcribed one (as a multiset: no line missing, none added).  *sha_file = the sha256 of the bytes read ("" if unreadable).
bool MP3bOracleFileAgrees(uint32_t* rows, std::string* sha_file);
// The sha256 recorded at transcription (MP3bOracle.inc).
const char* MP3bOracleSha256();

}  // namespace orbifold_mp

#endif
