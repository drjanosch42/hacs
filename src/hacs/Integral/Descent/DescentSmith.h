// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_DESCENT_DESCENT_SMITH_H
#define INTEGRAL_DESCENT_DESCENT_SMITH_H

// ===========================================================================
// PAR-D2 (hecke-extensions slot 24; docs/implementation/slot24/DESCENT_parallelism.md lever (c), report PARD2_report.md):
// a blocked, threaded, SIMD replay of D2's minimal-valuation Smith form maps::SmithForm (DescentMaps.cpp), with the transforms
// kept in FACTORED form.
//
// THE CONTRACT.  The pivot sequence is EXACTLY the oracle's (full pivoting, the first unit in row-major order of the trailing
// matrix, else the first entry of minimal valuation), so sigma, the row and column permutations, the multipliers f and the
// normalised pivot rows g are the oracle's ring elements; L, L^{-1}, R, R^{-1} are functions of these (below), so every entry
// of every transform is the oracle's value in [0, ell^k).  Only the ORDER of the exact ring operations changes (deferred,
// blocked updates; Z/ell^k is a ring, so the reduced representatives agree bit for bit).
//
// THE FACTORS (positions are final; rowOrig / colOrig map a final position to the input index):
//   F[i][t] (i > t, t < s)   the multiplier of row i at pivot t (stored in A below the diagonal);
//   G[t][j] (j > t, t < s)   the normalised pivot row: (U[t][j] / ell^sigma_t) * u_t^{-1} (U = the pivot row at pivot time;
//                            stored in A above the diagonal once `right` transforms are requested);
//   F, G are unit triangular (identity outside the first s columns / rows); then
//   L^{-1} = P^T F  (LinvT row t = e_{rowOrig[t]} + sum_{i > t} F[i][t] e_{rowOrig[i]}),  L = F^{-1} P,
//   R^{-1} = G Q^T  (Rinv row t = e_{colOrig[t]} + sum_{j > t} G[t][j] e_{colOrig[j]}),  R = Q G^{-1}.
// Proved in PARD2_report.md Sec. 2 and validated entrywise against the oracle (IntegralDescentSmithBlockedValidation).
//
// THE LEVER.  HECKE_DESCENT_SMITH_BLOCKED=0 (read once; read-only, value-preserving) routes maps::SmithForm and its D2 / cone
// callers (BuildPres, InducedMap, LatticeQuotient) back to the scalar oracle code of record.  Threads (PAR-UNIFY): the process's
// one descent pool and thread budget, shared with the ring2k / ring3k Smith (engine::smithpar::default_threads(): ENGINE_THREADS,
// else the CPU affinity count, capped; src/ModArithmetic/Binary/SmithPar.h); ENGINE_THREADS=1 is serial; a step below the work
// threshold (smithpar::min_work()) stays serial.  SIMD: AVX-512 where the CPU has it, else AVX2, else scalar
// (HECKE_DESCENT_SMITH_SIMD=scalar|avx2|avx512 forces a path for measurements; never a value).
//
// OVERFLOW.  ell^k < 2^31 (ZMod's ENGINE_CHECK); at ell = 2 all arithmetic is exact mod 2^32 and masked; at odd ell every
// accumulator holds at most 3 products < (ell^k)^2 < 2^62 plus one reduced value before a reduction (< 2^64), and the SIMD
// Montgomery path reduces sums of at most 2 products < ell^k 2^32 (both certified by the bound on ell^k, ENGINE_CHECK'ed).
// ===========================================================================

#include <cstdint>
#include <string>
#include <vector>

#include "DescentMaps.h"
#include "../../../ModArithmetic/EnvLever.h"

namespace integral_descent {
namespace maps {

struct SmithEngineOptions {
    int blocked = -1;          // -1: the lever (default: blocked); 0: the oracle; 1: blocked
    uint32_t threads = 0;      // 0: the descent budget (smithpar::default_threads()); 1: serial
    int simd = -1;             // -1: auto / HECKE_DESCENT_SMITH_SIMD; 0 scalar; 1 avx2; 2 avx512 (clamped to the CPU)
    uint32_t panel = 0;        // panel width (0: default)
};

// The lever (env HECKE_DESCENT_SMITH_BLOCKED, read once; a scoped override for validations wins).
bool DescentSmithBlockedEnabled();
const engine::lever::EnvLever& DescentSmithBlockedLever();   // (S24 CL-5) the EnvLever behind it (EnvLever.h)
struct ScopedDescentSmithBlocked : engine::lever::EnvLever::Scoped {
    explicit ScopedDescentSmithBlocked(bool on) : Scoped(DescentSmithBlockedLever(), on) {}
};
// The SIMD path that runs (after clamping to the CPU): "avx512", "avx2" or "scalar".
std::string DescentSmithSimdPath(int simd = -1);
int DescentSmithClampSimd(int simd);   // the path that a request runs on this CPU (0 scalar, 1 avx2, 2 avx512)
uint32_t DescentSmithDefaultThreads();

struct SmithFactors {
    ZMod Z;
    uint32_t r = 0, c = 0;
    std::vector<uint32_t> sigma;      // == Smith::sigma
    std::vector<uint32_t> uinv;       // u_t^{-1}, t < s
    std::vector<uint32_t> rowOrig;    // final row position -> input row
    std::vector<uint32_t> colOrig;    // final column position -> input column
    Mat A;                            // F below the diagonal (columns < s); G above it (rows < s) iff g_ready
    bool g_ready = false;
    uint32_t threads = 1;
    int simd = 0;
    uint32_t rank() const { return static_cast<uint32_t>(sigma.size()); }
    uint32_t smax() const;
};

// The factorisation (consumes A).  want_g: also normalise the pivot rows into G (needed for R, R^{-1}).
SmithFactors SmithFactor(const ZMod& Z, Mat A, bool want_g, const SmithEngineOptions& opt = {});

// ---- the transforms, entrywise the oracle's ----------------------------------------------------------------------------
Mat FactorsL(const SmithFactors& S);                 // r x r
Mat FactorsLinvT(const SmithFactors& S);             // r x r
Mat FactorsRT(const SmithFactors& S);                // c x c
Mat FactorsRinv(const SmithFactors& S);              // c x c
// rows `rows` of L (|rows| x r); of L^{-1}^T (|rows| x r); of R^T (|rows| x c); of R^{-1} (|rows| x c).
Mat FactorsLRows(const SmithFactors& S, const std::vector<uint32_t>& rows);
Mat FactorsLinvTRows(const SmithFactors& S, const std::vector<uint32_t>& rows);
Mat FactorsRTRows(const SmithFactors& S, const std::vector<uint32_t>& rows);
Mat FactorsRinvRows(const SmithFactors& S, const std::vector<uint32_t>& rows);
// L * B (r x k), B reduced.
Mat FactorsApplyL(const SmithFactors& S, const Mat& B);
// The full Smith (oracle layout) from the factors.
Smith FactorsToSmith(SmithFactors&& S, bool left, bool right);

// A B mod ell^k (B reduced; A reduced, or reduced entry by entry on the fly when reduce_a; the result reduced), threaded over
// rows, SIMD (the same kernel): entrywise D2's maps::Mul(Z, A, B) (resp. maps::Mul(Z, Reduce(A, Z), B)).
Mat MulMod(const ZMod& Z, const Mat& A, const Mat& B, bool reduce_a = false);
// A1 B1 == A2 B2 entrywise mod ell^k (reduced inputs), streamed in row blocks (no product is materialised), threaded.  The shapes
// must agree (ENGINE_CHECK, as a product would); unequal outer shapes compare false.
bool ProductsEqual(const ZMod& Z, const Mat& A1, const Mat& B1, const Mat& A2, const Mat& B2);

// The oracle of record (the scalar code of D2, unchanged), and the routed entry (SmithForm calls this when the lever is on).
Smith SmithFormOracle(const ZMod& Z, Mat A, bool left, bool right);
Smith SmithFormBlocked(const ZMod& Z, Mat A, bool left, bool right, const SmithEngineOptions& opt = {});

// VALIDATION ONLY: while a capture is installed, every maps::SmithForm / SmithFactor call with r * c >= min_entries appends its
// input (and ZMod) here (the "real D2 inputs" arm).  Never installed by production code.  (PAR-ASM audit F2) The notes are
// mutex-guarded, and while either capture is installed the PAR-ASM concurrent regions run in index order on the caller
// (CaptureActive()), so the captured sequence is the serial one.  Install / remove a capture outside any parallel region.
struct SmithCapture {
    uint64_t min_entries = 0;
    uint32_t max_items = 64;
    std::vector<std::pair<ZMod, Mat>> items;
};
void SetSmithCapture(SmithCapture* cap);
void SmithCaptureNote(const ZMod& Z, const Mat& A);

// VALIDATION ONLY: the callers' inputs (BuildPres, InducedMap), for the field-by-field comparison with the lever on and off.
struct CallerCapture {
    uint64_t max_entries = uint64_t(1) << 20;   // skip a call whose largest matrix is bigger
    uint32_t max_items = 400;
    struct BP {
        ZMod Z;
        Mat P;
        uint32_t rankP;
        Mat Nx;
        uint32_t c, b;
    };
    struct IM {
        Pres X, Y;
        Mat f;
        uint32_t rankF;
    };
    std::vector<BP> bp;
    std::vector<IM> im;
};
void SetCallerCapture(CallerCapture* cap);
void CallerCaptureBuildPres(const ZMod& Z, const Mat& P, uint32_t rankP, const Mat& Nx, uint32_t c, uint32_t b);
void CallerCaptureInducedMap(const Pres& X, const Pres& Y, const Mat& f, uint32_t rankF);
// true while a SmithCapture or a CallerCapture is installed (PAR-ASM's ForEachIndex then runs serially)
bool CaptureActive();

// Statistics of the last blocked factorisation on this thread (measurements only).
struct SmithStats {
    uint64_t rows_scanned = 0, rows_skipped = 0, panels = 0, col_swaps = 0, row_updates = 0;
    double seconds = 0;
};

}  // namespace maps
}  // namespace integral_descent

#endif
