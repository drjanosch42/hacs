// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 3, WP-5 (phase E5-2) -- the equivariant mapping cone.
// See the header for the mathematics, the layout, the degree rule and the
// certificate items.

#include "EquivariantMappingCone.h"

#include "SimplicialCohomology.h"
#include "../Engine/EngineRuntime.h"
#include "../Engine/TapeElim.h"
#include "../ModArithmetic/BlockElim.h"
#include "../ModArithmetic/BlockKernels.h"
#include "../ModArithmetic/DenseElim.h"
#include "../ModArithmetic/DenseView.h"
#include "../ModArithmetic/EngineCheck.h"
#include "../ModArithmetic/Fnv.h"
#include "../ModArithmetic/Kernels.h"
#include "../ModArithmetic/ModCtx.h"
#include "Hecke/HeckeSettings.h"
#include "Util/FieldDiet.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

using engine::BlockBuilder;
using engine::BlockMat;
using engine::BlockPool;
using engine::BlockRef;
using engine::DenseBuf;
using engine::DenseView;
using engine::Layout;
using engine::ModCtx;
using engine::cview;
using engine::view;

// ===========================================================================
// (slot-24 G6) ConeOptions -- see the header.
// ===========================================================================
namespace ConeOptions {
bool unit_row_span() { return oc_util::field_diet_on(); }
void set_unit_row_span(bool on) { oc_util::set_field_diet_on(on); }
}  // namespace ConeOptions

// ===========================================================================
// ComplexDegreeFields
// ===========================================================================

ComplexDegreeFields ComplexDegreeFields::derived_from(const EquivariantCohomologyOnK& coh) {
    if (!coh.is_cohomology_computed()) throw std::runtime_error("ComplexDegreeFields: the complex is not computed");
    const size_t nd = coh.get_H_eq_dimensions().size();
    if (nd == 0) throw std::runtime_error("ComplexDegreeFields: the complex reports no degree");
    ComplexDegreeFields f;
    f.q_max = static_cast<uint32_t>(nd - 1);
    f.truncated = false;
    f.q_max_source = "derived";
    for (uint32_t q = 0; q <= f.q_max; ++q) f.degrees.push_back(q);
    return f;
}

ComplexDegreeFields ComplexDegreeFields::declared_truncation(uint32_t q_max) {
    ComplexDegreeFields f;
    f.q_max = q_max;
    f.truncated = true;
    f.q_max_source = "set";
    for (uint32_t q = 0; q <= q_max; ++q) f.degrees.push_back(q);
    return f;
}

std::string ComplexDegreeFields::to_json() const {
    std::ostringstream oss;
    oss << "{\"q_max\":" << q_max << ",\"truncated\":" << (truncated ? "true" : "false")
        << ",\"q_max_source\":\"" << q_max_source << "\",\"degrees\":[";
    for (size_t i = 0; i < degrees.size(); ++i) oss << (i ? "," : "") << degrees[i];
    oss << "]}";
    return oss.str();
}

// ===========================================================================
// Local helpers (engine kernels only; no raw mod_* call, no Matrix.h).
// ===========================================================================

namespace {

using BM = BlockMat<uint32_t>;
using Buf = DenseBuf<uint32_t>;
using CView = DenseView<const uint32_t>;

// A row-major dense copy of any view (empty shapes allowed).
Buf dense_copy(CView v) {
    Buf out(v.rows(), v.cols(), Layout::RowMajor);
    if (v.rows() > 0 && v.cols() > 0) engine::kernels::copy<uint32_t>(v, view(out));
    return out;
}

uint32_t rank_of_view(const ModCtx<uint32_t>& ctx, CView v) {
    if (v.rows() == 0 || v.cols() == 0) return 0;
    Buf tmp = dense_copy(v);
    return engine::elim::rref<uint32_t>(ctx, view(tmp)).rank;
}

bool is_zero_view(CView v) {
    for (uint32_t i = 0; i < v.rows(); ++i)
        for (uint32_t j = 0; j < v.cols(); ++j)
            if (v.at(i, j) != 0) return false;
    return true;
}

bool views_equal(CView a, CView b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) return false;
    for (uint32_t i = 0; i < a.rows(); ++i)
        for (uint32_t j = 0; j < a.cols(); ++j)
            if (a.at(i, j) != b.at(i, j)) return false;
    return true;
}

// Every column of Y lies in the column span of B (a x nb, a x ny).
//
// The predicate is the rank identity  rank [B | Y] = rank B, which over a
// field is span(Y) inside span(B).  It is read off ONE reduced row echelon
// form of the augmented matrix [B | Y]: a pivot column with index >= nb is a
// column of Y independent of B and of the Y-columns to its left, i.e. rank
// [B | Y] > rank B, and no such pivot is exactly rank [B | Y] = rank B.  That
// is the same test, on the same elimination of the same matrix, that
// engine::elim::solve_multi_rhs performs (DenseElim.cpp:622-628) before it
// back-substitutes; slot-4 WP-4 stops short of the back-substitution, which
// only builds the ny x nb coefficient matrix this caller discards (at
// n_sym = 256, q = 1 that matrix alone is ~24 MB in ~3000 allocations).
// Verdict-identical by construction: the pivot SET of a matrix over a field
// is the set of its leftmost independent columns, a property of the matrix
// and not of the elimination that finds it.
bool columns_in_span(const ModCtx<uint32_t>& ctx, CView B, CView Y) {
    if (Y.cols() == 0 || Y.rows() == 0) return true;
    if (B.rows() != Y.rows()) return false;
    if (B.cols() == 0) return is_zero_view(Y);
    const uint32_t a = B.rows(), nb = B.cols(), ny = Y.cols();
    Buf aug(a, nb + ny, Layout::RowMajor);
    engine::kernels::copy<uint32_t>(B, view(aug).sub(0, a, 0, nb));
    engine::kernels::copy<uint32_t>(Y, view(aug).sub(0, a, nb, ny));
    const engine::elim::RrefResult res = engine::elim::rref<uint32_t>(ctx, view(aug));
    for (uint32_t c : res.pivots)
        if (c >= nb) return false;
    return true;
}

// (hecke-extensions slot 5, WP-5) The SAME predicate, on the tape.
//
// `columns_in_span` above materialises the augmented matrix [B | Y] as an
// a x (nb + ny) row-major COPY of both operands and eliminates it in place.
// At n_sym = 256 that copy is the cone's binding transient inside verify():
// 62.7 MB at (q = 0, C2) and 63.4 MB at (q = 1, C3).  The tape's panel driver
// runs the SAME elimination of the SAME matrix over BORROWED pool windows of
// B and Y -- no copy of either operand -- and materialises only the RREF
// itself, rank x (nb + ny).  Stage 1 of engine::tape::factor_basis IS that
// elimination: with Rl = B and G = Y its `pivots` are exactly the columns of
// Y that are independent of B and of the Y columns to their left, so the
// predicate is `pivots.empty()`; want_projector = false skips stage 2, which
// this caller does not need.
//
// VERDICT-IDENTICAL, not approximately: the pivot SET of a matrix over a field
// is the set of its leftmost independent columns -- a property of the matrix,
// not of the elimination that finds it -- and tape::factor_basis's stage-1
// pivots are blk::factor_basis's pivots by TapeElim.h's contract (appending
// the ambient identity columns cannot change the pivot columns among the
// earlier ones), which EngineTapeEliminationValidation gates entrywise.
// CONE_SPAN_SHADOW=1 runs BOTH routes on every call and throws on the first
// disagreement -- the same shadow protocol slot-4 WP-4 used for this
// predicate's previous rewrite.
std::atomic<uint64_t> span_shadow_calls{0}, span_shadow_false{0};

// The right-hand side is presented ONE PANEL AT A TIME.  "Every column of Y
// lies in span(B)" is the conjunction of the same statement over any partition
// of Y's columns -- the predicate is a property of each column separately --
// so testing the chunks one after the other is the SAME predicate, not a
// weaker one (it is if anything the strongest reading: each column is tested
// against B alone).  The width is the elimination driver's own panel width
// (engine::blk::ElimParams::chunk, engine::tape::factor_basis's default), so
// each pass eliminates B once and then exactly one panel of Y, and the
// materialised RREF is rank x (nb + 256) instead of rank x (nb + ny).
// Measured at n_sym = 256, cone VmHWM / verify() wall:
//   ny in one pass 537.8 MB / 2.44 s;  512: 521.1 / 2.83;
//   256: 510.4 / 3.58;                 128: 505.2 / 5.05.
//
// (slot-22 PERF1, P1 of PERF_audit_large_n.md) Every pass eliminates ALL of B
// again, so a fixed 256-column panel made the test cost ceil(ny / 256)
// eliminations of B: 11 - 22 of them per test at n_sym = 700, 182 s = 42 % of
// a (700, 13) census cell.  The panel is now max(kSpanYChunk, nb): never
// narrower than B itself, so B is eliminated ceil(ny / nb) times: once for C3
// (Z_q inside Ceq_q, ny <= nb) and C5 (H_q inside Z_q), once or twice for C2
// (D Ceq_q against Ceq_{q+1}; twice at q = 1, where Ceq_2 is the narrower).
// The materialised RREF, rank x (nb + w), stays at most twice the rank x nb it
// can never be smaller than -- the memory bound is relative to B, not a fixed
// width.  The predicate, the matrix
// eliminated and the verdict are unchanged (any partition of Y's columns gives
// the same conjunction; CONE_SPAN_SHADOW still checks it against the dense
// route on every call).
constexpr uint32_t kSpanYChunk = 256;
constexpr uint64_t kSpanTapeMinArea = 1u << 20;   // elements; 4 MB at uint32_t

// (slot-24 G6) THE UNIT-ROW ROUTE of the same predicate (ConeOptions, the
// HECKE_FIELD_DIET lever).  When every column c of B has a UNIT ROW u_c
// (B[u_c, :] = e_c^T; a D5b kernel basis always has one: the identity sits at
// the free variables), B has full column rank and the coordinates of any
// y in span(B) are forced: y = B x  =>  x[c] = y[u_c].  So Y lies in span(B)
// iff Y = B X with X = Y[u, :], and on the unit rows that identity holds by
// construction: the predicate is exactly  Y[r, :] == B[r, :] . X  for every
// NON-unit row r -- one (a - nb) x nb x ny product, COMPAREd on the tape,
// instead of the RREF of the a x (nb + w) augmented matrix (1.4 GB live at
// (n, q) = (1000, 0) of the C1a cone, C2).  The same statement as
// solve_multi_rhs's unit-row shortcut (TapeElim.cpp), restricted to the rows
// where it can fail.  Verdict-identical: both decide "every column of Y is in
// span(B)" exactly; CONE_SPAN_SHADOW still checks the dense route on every
// call.  Returns 0 when some column has no unit row (the caller then runs the
// elimination route), 1 = in span, 2 = not in span.
std::atomic<uint64_t> unit_row_calls{0}, unit_row_fallbacks{0};
constexpr uint32_t kUnitRowYChunk = 2048;

int columns_in_span_unit_rows(engine::tape::Executor<uint32_t>& ex, uint32_t p, CView B, CView Y) {
    const uint32_t a = B.rows(), nb = B.cols(), ny = Y.cols();
    const auto bs = B.strides();   // element (i, j) at base[i * rs + j * cs] (logical, transposition resolved)
    const auto ys = Y.strides();
    std::vector<uint32_t> nnz(a, 0), col_of(a, 0);
    std::vector<uint32_t> val(a, 0);
    for (uint32_t c = 0; c < nb; ++c)
        for (uint32_t r = 0; r < a; ++r) {
            const uint32_t v = bs.base[static_cast<size_t>(r) * bs.rs + static_cast<size_t>(c) * bs.cs];
            if (v != 0) { ++nnz[r]; col_of[r] = c; val[r] = v; }
        }
    std::vector<uint32_t> unit_row(nb, a);
    for (uint32_t r = 0; r < a; ++r)
        if (nnz[r] == 1 && val[r] == 1u && unit_row[col_of[r]] == a) unit_row[col_of[r]] = r;
    for (uint32_t c = 0; c < nb; ++c)
        if (unit_row[c] == a) { ++unit_row_fallbacks; return 0; }
    ++unit_row_calls;
    std::vector<char> is_unit(a, 0);
    for (uint32_t c = 0; c < nb; ++c) is_unit[unit_row[c]] = 1;
    std::vector<uint32_t> rest;
    for (uint32_t r = 0; r < a; ++r) if (!is_unit[r]) rest.push_back(r);
    const uint32_t m = static_cast<uint32_t>(rest.size());
    if (m == 0) return 1;   // B is a permutation matrix: its span is everything
    Buf R(m, nb, Layout::RowMajor);   // B[rest, :]
    for (uint32_t i = 0; i < m; ++i)
        for (uint32_t c = 0; c < nb; ++c) R.at(i, c) = bs.base[static_cast<size_t>(rest[i]) * bs.rs + static_cast<size_t>(c) * bs.cs];
    const ModCtx<uint32_t> ctx(p);
    for (uint32_t t0 = 0; t0 < ny; t0 += kUnitRowYChunk) {
        const uint32_t w = std::min<uint32_t>(kUnitRowYChunk, ny - t0);
        Buf X(nb, w, Layout::RowMajor);   // X = Y[u, t0 ..]
        for (uint32_t c = 0; c < nb; ++c)
            for (uint32_t t = 0; t < w; ++t) X.at(c, t) = ys.base[static_cast<size_t>(unit_row[c]) * ys.rs + static_cast<size_t>(t0 + t) * ys.cs];
        Buf Yr(m, w, Layout::RowMajor);   // Y[rest, t0 ..]
        for (uint32_t i = 0; i < m; ++i)
            for (uint32_t t = 0; t < w; ++t) Yr.at(i, t) = ys.base[static_cast<size_t>(rest[i]) * ys.rs + static_cast<size_t>(t0 + t) * ys.cs];
        engine::tape::Recorder<uint32_t> rec(ctx);
        const engine::tape::BufId bR = rec.input(cview(R));
        const engine::tape::BufId bX = rec.input(cview(X));
        const engine::tape::BufId bY = rec.input(cview(Yr));
        const engine::tape::BufId bP = rec.temp(m, w, Layout::RowMajor);
        rec.gemm(rec.whole(bR), rec.whole(bX), rec.whole(bP), engine::kernels::GemmOp::Set);
        const uint32_t slot = rec.compare(rec.whole(bP), rec.whole(bY));
        engine::tape::Program<uint32_t> prog = rec.finish();
        engine::tape::compile(prog, engine::tape::consumer_pass_options());
        engine::tape::note_program_run(prog.stats.gpu_nodes);
        const engine::tape::RunResult rr = ex.run(static_cast<const engine::tape::Program<uint32_t>&>(prog), rec.bindings());
        ENGINE_CHECK(rr.slots[slot].written);
        if (!rr.slots[slot].equal) return 2;
    }
    return 1;
}

bool columns_in_span_tape(engine::tape::Executor<uint32_t>& ex, const BM& ambient, CView B, CView Y) {
    if (Y.cols() == 0 || Y.rows() == 0) return true;
    if (B.rows() != Y.rows()) return false;
    if (B.cols() == 0) return is_zero_view(Y);
    ENGINE_CHECK_MSG(ambient.rows() == B.rows(), "columns_in_span_tape: ambient height != operand height");
    if (ConeOptions::unit_row_span() &&
        static_cast<uint64_t>(B.rows()) * (static_cast<uint64_t>(B.cols()) + Y.cols()) >= kSpanTapeMinArea) {
        const int u = columns_in_span_unit_rows(ex, ambient.p(), B, Y);
        if (u != 0) {
            const bool in_span = (u == 1);
            if (std::getenv("CONE_SPAN_SHADOW") != nullptr) {
                const ModCtx<uint32_t> ctx(ambient.p());
                const bool ref = columns_in_span(ctx, B, Y);
                ++span_shadow_calls;
                if (!in_span) ++span_shadow_false;
                ENGINE_CHECK_MSG(ref == in_span, "CONE_SPAN_SHADOW: the unit-row span route disagrees with the dense route");
                std::cout << "  CONE_SPAN_SHADOW calls=" << span_shadow_calls.load() << " false_verdicts=" << span_shadow_false.load()
                          << " disagreements=0 (unit rows)" << std::endl;
            }
            return in_span;
        }
    }
    // Below the threshold the augmented copy is not worth avoiding and the
    // tape's fixed per-program cost (recording, borrowing, the executor) is
    // not amortised: the dense in-place route is both smaller and faster.  The
    // two routes decide the same predicate, so the threshold is a cost choice
    // and never a semantic one (CONE_SPAN_SHADOW checks both arms on every
    // call).  kSpanTapeMinArea is one 4 MB augmented matrix.
    if (static_cast<uint64_t>(B.rows()) * (static_cast<uint64_t>(B.cols()) + Y.cols()) < kSpanTapeMinArea) {
        const ModCtx<uint32_t> ctx(ambient.p());
        return columns_in_span(ctx, B, Y);
    }
    bool in_span = true;
    const uint32_t wy = std::max<uint32_t>(std::max<uint32_t>(1, kSpanYChunk), B.cols());
    for (uint32_t c0 = 0; in_span && c0 < Y.cols(); c0 += wy) {
        const uint32_t w = std::min<uint32_t>(wy, Y.cols() - c0);
        const engine::blk::FactorBasisResult<uint32_t> r =
            engine::tape::factor_basis<uint32_t>(ex, ambient.pool(), ambient.p(), ambient.row_dims(),
                                                 B, Y.sub(0, Y.rows(), c0, w),
                                                 /*chunk=*/256, /*stats=*/nullptr, /*want_projector=*/false);
        in_span = r.pivots.empty();
    }
    if (std::getenv("CONE_SPAN_SHADOW") != nullptr) {
        const ModCtx<uint32_t> ctx(ambient.p());
        const bool ref = columns_in_span(ctx, B, Y);
        ++span_shadow_calls;
        if (!in_span) ++span_shadow_false;
        ENGINE_CHECK_MSG(ref == in_span, "CONE_SPAN_SHADOW: columns_in_span_tape disagrees with the dense route");
        std::cout << "  CONE_SPAN_SHADOW calls=" << span_shadow_calls.load() << " false_verdicts=" << span_shadow_false.load()
                  << " disagreements=0" << std::endl;
    }
    return in_span;
}

// Y (A.rows() x X.cols()) = A * X for a block operator A and a dense X whose
// rows are partitioned by A's column offsets.  A with no rows or X with no
// columns gives the (empty) zero result.
Buf block_times_dense(const BM& A, CView X, Layout layout = Layout::ColMajor) {
    ENGINE_CHECK_MSG(X.rows() == A.cols(), "block_times_dense: operand height != A.cols()");
    Buf Y(A.rows(), X.cols(), layout);
    if (A.rows() > 0 && X.cols() > 0 && A.cols() > 0) engine::blk::gemm_dense<uint32_t>(A, X, view(Y));
    return Y;
}

FlatMatrix flat_of(CView v, uint32_t p) {
    const uint32_t R = v.rows(), C = v.cols();
    if (R == 0 || C == 0) return FlatMatrix::CreateZero(R, C, p);
    std::vector<uint32_t> d(static_cast<size_t>(R) * C);
    for (uint32_t i = 0; i < R; ++i)
        for (uint32_t j = 0; j < C; ++j) d[static_cast<size_t>(i) * C + j] = v.at(i, j);
    return FlatMatrix(R, C, p, d);
}

bool flat_equal(const FlatMatrix& X, const FlatMatrix& Y) {
    if (X.get_rows() != Y.get_rows() || X.get_cols() != Y.get_cols()) return false;
    for (uint32_t r = 0; r < X.get_rows(); ++r)
        for (uint32_t c = 0; c < X.get_cols(); ++c)
            if (X.get_entry(r, c) != Y.get_entry(r, c)) return false;
    return true;
}

Buf dense_of_flat(const FlatMatrix& F) {
    Buf out(F.get_rows(), F.get_cols(), Layout::RowMajor);
    for (uint32_t r = 0; r < F.get_rows(); ++r)
        for (uint32_t c = 0; c < F.get_cols(); ++c) out.at(r, c) = F.get_entry(r, c);
    return out;
}

uint32_t rank_of_flat(const ModCtx<uint32_t>& ctx, const FlatMatrix& F) {
    const Buf d = dense_of_flat(F);
    return rank_of_view(ctx, cview(d));
}

// The dense multi-vector of a BlockMat with nbc == 1 (zero-copy when contiguous).
CView mv_view(const BM& mv, Buf& scratch) { return engine::blk::multivector_view<uint32_t>(mv, scratch); }

// (hecke-extensions slot 5, WP-5) Y = A . X for a block operator A and a dense
// multi-vector X whose rows are partitioned by A's column offsets, RECORDED as
// one engine::tape gemm_dense program and served from the program cache by
// shape signature (EngineRuntime.h run_cached: record always, compile on a
// miss) -- exactly the route EquivariantCohomologyOnK's
// tape_block_times_multivector_buf takes.
//
// VALUE-PRESERVING BY CONSTRUCTION, not by inspection: compile() is idempotent
// and a pure function of (program, options), every decision it takes is
// recorded IN the program, and the kernels accumulate products of canonical
// residues exactly with one reduction per k-block (ENGINE_API.md Sec. 5), so
// k-blocking, register blocking, vectorisation, the CPU path chosen, the
// thread count and batching cannot move a bit of the result.  The recorded
// program computes the same sum, in the same fixed order, as
// engine::blk::gemm_dense.
Buf tape_block_times_dense(engine::tape::Executor<uint32_t>& ex, const char* tag, uint32_t q,
                           const BM& A, CView X, Layout layout = Layout::ColMajor) {
    using namespace engine::tape;
    ENGINE_CHECK_MSG(X.rows() == A.cols(), "tape_block_times_dense: operand height != A.cols()");
    Buf Y(A.rows(), X.cols(), layout);
    if (A.rows() > 0 && X.cols() > 0 && A.cols() > 0) {
        Recorder<uint32_t> rec(A.ctx());
        const BufId bX = rec.input(X);
        const BufId bY = rec.output(view(Y));
        rec.record_gemm_dense(A, rec.whole(bX), rec.whole(bY), engine::kernels::GemmOp::Set);
        Program<uint32_t> prog = rec.finish();
        ShapeKey key = make_key<uint32_t>(tag, A.ctx(), A.row_dim(0), q);
        key.row_dims = A.row_dims();
        key.col_dims = A.col_dims();
        key.pattern_hash = fnv_mix(fnv_mix(pattern_hash_of<uint32_t>(A), X.cols()),
                                   fnv_mix(A.cols(), static_cast<uint64_t>(layout == Layout::RowMajor)));
        run_cached<uint32_t>(ex, key, std::move(prog), rec.bindings());
    }
    return Y;
}

// ENGINE_TRACE_RSS=1: one line per step of EquivariantMappingCone::compute()
// with the wall time since the previous mark and the process's VmRSS / VmHWM
// (MB), the same shape as EquivariantCohomologyOnK's StepTrace.  Measurement
// output only (slot-5 WP-5): gated on the environment, never on a result path,
// and it reads nothing the computation writes.
struct ConeTrace {
    bool on = std::getenv("ENGINE_TRACE_RSS") != nullptr;
    std::chrono::steady_clock::time_point t = std::chrono::steady_clock::now();
    static double status_mb(const char* key) {
        FILE* f = std::fopen("/proc/self/status", "r");
        if (!f) return -1.0;
        char line[256]; double out = -1.0;
        while (std::fgets(line, sizeof line, f))
            if (std::strncmp(line, key, std::strlen(key)) == 0) { out = std::atof(line + std::strlen(key)) / 1024.0; break; }
        std::fclose(f);
        return out;
    }
    void mark(const std::string& what) {
        if (!on) return;
        const auto now = std::chrono::steady_clock::now();
        std::cout << "      [cone] " << what << ": " << std::fixed << std::setprecision(3)
                  << std::chrono::duration<double>(now - t).count() << " s, VmRSS=" << std::setprecision(1)
                  << status_mb("VmRSS:") << " MB VmHWM=" << status_mb("VmHWM:") << " MB" << std::endl;
        t = now;
    }
};

// Product of two block operators as a dense matrix, for the D^2 check.
bool block_product_is_zero(const BM& A, const BM& B) {
    if (A.rows() == 0 || B.cols() == 0 || A.cols() == 0) return true;
    ENGINE_CHECK_MSG(A.cols() == B.rows(), "block_product_is_zero: shapes disagree");
    const Buf Bd = B.to_dense(Layout::ColMajor);
    const Buf AB = block_times_dense(A, cview(Bd));
    return is_zero_view(cview(AB));
}

std::string join_u32(const std::vector<uint32_t>& v) {
    std::ostringstream oss;
    for (size_t i = 0; i < v.size(); ++i) oss << (i ? "," : "") << v[i];
    return oss.str();
}

// Golden hashing, byte-identical to Golden/GoldenHarness.cpp (rows, cols,
// entries row-major).
uint64_t hash_view(CView v) {
    uint64_t h = engine::kFnvOffset64;
    h = engine::FnvU64(h, v.rows());
    h = engine::FnvU64(h, v.cols());
    for (uint32_t i = 0; i < v.rows(); ++i)
        for (uint32_t j = 0; j < v.cols(); ++j) h = engine::FnvU64(h, static_cast<uint64_t>(v.at(i, j)));
    return h;
}
uint64_t hash_flat(const FlatMatrix& F) {
    const Buf d = dense_of_flat(F);
    return hash_view(cview(d));
}
uint64_t hash_scalar(uint64_t v) { return engine::FnvU64(engine::kFnvOffset64, v); }

// The cone's derived degree fields from its inputs' (the rule of the header note).
ComplexDegreeFields derive_cone_fields(const ComplexDegreeFields& bulk, const ComplexDegreeFields& bd,
                                       uint32_t top_K, uint32_t top_bd) {
    ComplexDegreeFields c;
    c.q_max_source = "derived";
    if (!bulk.truncated && !bd.truncated) {
        c.q_max = std::max(top_K, top_bd + 1);
        c.truncated = false;
    } else {
        c.q_max = std::min(bulk.q_max, bd.q_max + 1);
        c.truncated = true;
    }
    for (uint32_t q = 0; q <= c.q_max; ++q) c.degrees.push_back(q);
    return c;
}

} // namespace

// (slot-24 G6) Validation entry -- see the header.
namespace ConeOptions {
SpanProbe span_test_for_validation(uint32_t ell, const engine::DenseBuf<uint32_t>& B, const engine::DenseBuf<uint32_t>& Y) {
    SpanProbe out;
    const uint64_t calls0 = unit_row_calls.load(), falls0 = unit_row_fallbacks.load();
    std::vector<uint32_t> rd(B.rows == 0 ? 0u : 1u, B.rows);
    std::shared_ptr<engine::BlockPool<uint32_t>> pool = std::make_shared<engine::BlockPool<uint32_t>>();
    const BM ambient = BM::zeros(pool, rd, std::vector<uint32_t>{ B.cols }, ell);
    engine::tape::Executor<uint32_t> ex(&engine::tape::consumer_pool());
    out.verdict = columns_in_span_tape(ex, ambient, cview(B), cview(Y));
    const ModCtx<uint32_t> ctx(ell);
    out.dense = columns_in_span(ctx, cview(B), cview(Y));
    out.unit_row_route = unit_row_calls.load() > calls0;
    out.fell_back = unit_row_fallbacks.load() > falls0;
    return out;
}
UnitRowStats unit_row_stats() {
    UnitRowStats st;
    st.calls = unit_row_calls.load();
    st.fallbacks = unit_row_fallbacks.load();
    return st;
}
}  // namespace ConeOptions

// ===========================================================================
// ConeCertificate::verify -- the thirteen items, re-derived from the stored
// matrices alone.
// ===========================================================================

uint32_t ConeCertificate::verify() const {
    std::vector<std::string> f;
    return verify(f);
}

std::vector<uint32_t> ConeCertificate::failing_items() const {
    std::vector<std::string> f;
    (void)verify(f);
    std::vector<uint32_t> items;
    for (const std::string& s : f) {
        // every failure line starts with "C<item>:"
        if (s.size() > 1 && s[0] == 'C') {
            const uint32_t item = static_cast<uint32_t>(std::strtoul(s.c_str() + 1, nullptr, 10));
            if (std::find(items.begin(), items.end(), item) == items.end()) items.push_back(item);
        }
    }
    std::sort(items.begin(), items.end());
    return items;
}

uint32_t ConeCertificate::verify(std::vector<std::string>& failures) const {
    failures.clear();
    // (slot-23 PERF-P2) rank_of_view / C9's RREF / map_rank on the consumer pool (value-free;
    // a no-op when compute() already installed it on this thread).
    const engine::tape::ScopedHostElimHook host_elim_hook;
    auto fail = [&](uint32_t item, const std::string& msg) {
        failures.push_back("C" + std::to_string(item) + ": " + msg);
    };
    if (ell < 2) { fail(13, "ell < 2"); return static_cast<uint32_t>(failures.size()); }
    const ModCtx<uint32_t> ctx(ell);
    const uint32_t Qc = cone.q_max;
    const uint32_t top_K = H_bulk.empty() ? 0 : static_cast<uint32_t>(H_bulk.size() - 1);
    const uint32_t top_bd = H_bd.empty() ? 0 : static_cast<uint32_t>(H_bd.size() - 1);

    // ---- C13 first: the shapes every other item indexes into --------------
    bool shapes_ok = true;
    auto shape = [&](bool ok, const std::string& what) { if (!ok) { fail(13, what); shapes_ok = false; } };
    shape(!H_bulk.empty() && P_bulk.size() == H_bulk.size() && dim_H_bulk.size() == H_bulk.size(), "bulk bases / projectors / dims misshaped");
    shape(!H_bd.empty() && P_bd.size() == H_bd.size() && dim_H_bd.size() == H_bd.size(), "boundary bases / projectors / dims misshaped");
    shape(D.size() == Qc + 1 && Ceq.size() == Qc + 2 && Z.size() == Qc + 1 && B.size() == Qc + 1 && H.size() == Qc + 1 &&
              P.size() == Qc + 1 && B_pivots.size() == Qc + 1 && dim_Hc.size() == Qc + 1 && flat_dim.size() == Qc + 2 &&
              bulk_part_dim.size() == Qc + 2,
          "cone objects misshaped for q_max = " + std::to_string(Qc));
    shape(pi.size() == Qc + 2 && r.size() == Qc + 2 && iota.size() == Qc + 2 && pi_built.size() == Qc + 2 &&
              r_built.size() == Qc + 2 && iota_built.size() == Qc + 2,
          "map families misshaped");
    shape(r_flat.size() == static_cast<size_t>(std::min(top_K, top_bd)) + 1, "r_flat misshaped");
    if (shapes_ok) {
        const ComplexDegreeFields want = derive_cone_fields(bulk, boundary, top_K, top_bd);
        shape(want.q_max == cone.q_max && want.truncated == cone.truncated && cone.q_max_source == "derived" &&
                  want.degrees == cone.degrees,
              "the cone's degree fields are not the ones derived from its inputs'");
        shape(bulk.degrees.size() == bulk.q_max + 1 && boundary.degrees.size() == boundary.q_max + 1, "input degree lists");
        shape(bulk.truncated ? (bulk.q_max < top_K) : (bulk.q_max == top_K), "bulk q_max vs its top degree");
        shape(boundary.truncated ? (boundary.q_max < top_bd) : (boundary.q_max == top_bd), "boundary q_max vs its top degree");
        for (uint32_t q = 0; q <= Qc + 1; ++q) {
            const uint32_t want_dim = Ceq[q].rows();
            shape(flat_dim[q] == want_dim && bulk_part_dim[q] <= want_dim, "flat_dim[" + std::to_string(q) + "]");
            shape(D.size() <= q || D[q].cols() == flat_dim[q], "D[" + std::to_string(q) + "] columns");
            shape(D.size() <= q || D[q].rows() == flat_dim[q + 1], "D[" + std::to_string(q) + "] rows");
        }
        for (uint32_t q = 0; q <= Qc; ++q) {
            shape(Z[q].rows() == flat_dim[q] && B[q].rows() == flat_dim[q] && H[q].rows() == flat_dim[q], "basis heights at q = " + std::to_string(q));
            shape(H[q].cols() == dim_Hc[q] && P[q].rows() == dim_Hc[q] && P[q].cols() == flat_dim[q], "H / P shape at q = " + std::to_string(q));
        }
        for (uint32_t q = 0; q <= top_K; ++q) shape(H_bulk[q].cols() == dim_H_bulk[q] && P_bulk[q].rows() == dim_H_bulk[q], "bulk H / P at q = " + std::to_string(q));
        for (uint32_t q = 0; q <= top_bd; ++q) shape(H_bd[q].cols() == dim_H_bd[q] && P_bd[q].rows() == dim_H_bd[q], "boundary H / P at q = " + std::to_string(q));
        shape(six_invertible == (ell >= 5), "six_invertible flag");
    }
    if (!shapes_ok) return static_cast<uint32_t>(failures.size());

    // Dimension of a REPORTED term (0 above the stored range: a reported zero).
    auto dim_c = [&](int64_t q) -> uint32_t { return (q < 0 || q > static_cast<int64_t>(Qc)) ? 0u : dim_Hc[static_cast<size_t>(q)]; };
    auto dim_K = [&](int64_t q) -> uint32_t { return (q < 0 || q > static_cast<int64_t>(bulk.q_max)) ? 0u : dim_H_bulk[static_cast<size_t>(q)]; };
    auto dim_bd = [&](int64_t q) -> uint32_t { return (q < 0 || q > static_cast<int64_t>(boundary.q_max)) ? 0u : dim_H_bd[static_cast<size_t>(q)]; };

    // ---- C1: D_{q+1} D_q = 0 ------------------------------------------------
    for (uint32_t q = 0; q + 1 <= Qc; ++q)
        if (!block_product_is_zero(D[q + 1], D[q])) fail(1, "D_" + std::to_string(q + 1) + " D_" + std::to_string(q) + " != 0");

    // ---- C2..C5 per cone degree ---------------------------------------------
    std::vector<uint32_t> rank_DC(Qc + 1, 0);
    // The items below are evaluated in exactly the pre-slot-4 order, on exactly
    // the pre-slot-4 operands; slot-4 WP-4 only narrowed the SCOPES of the
    // gathered multi-vector copies (Ceq_{q+1}, Ceq_q, H_q) and of the D Ceq
    // product, so that a degree's transients are released at their last reader
    // rather than at the end of its iteration.
    ConeTrace vtr;
    // (slot-5 WP-5) The span tests of C2 / C3 / C5 run on the tape; the
    // executor is this verify() call's own, on the consumer pool.
    engine::tape::Executor<uint32_t> ex(&engine::tape::consumer_pool());
    for (uint32_t q = 0; q <= Qc; ++q) {
        Buf sZ, sB;
        const CView Zv = mv_view(Z[q], sZ);
        const CView Bv = mv_view(B[q], sB);
        const std::string at = " at q = " + std::to_string(q);
        vtr.mark("  V q=" + std::to_string(q) + " Z/B gathers");
        {
            Buf sC;
            const CView Cv = mv_view(Ceq[q], sC);
            const uint32_t k = Cv.cols();
            {
                const Buf DC = block_times_dense(D[q], Cv);
                rank_DC[q] = rank_of_view(ctx, cview(DC));
                // C2 subcomplex
                Buf sC1;
                const CView C1v = mv_view(Ceq[q + 1], sC1);
                vtr.mark("  V q=" + std::to_string(q) + " C2 operands (Cv " + std::to_string(Cv.rows()) + "x" + std::to_string(Cv.cols()) +
                         ", DC " + std::to_string(DC.rows) + "x" + std::to_string(DC.cols) + ", C1v " + std::to_string(C1v.rows()) + "x" + std::to_string(C1v.cols()) + ")");
                if (!columns_in_span_tape(ex, Ceq[q + 1], C1v, cview(DC))) fail(2, "D Ceq is not inside span Ceq_{q+1}" + at);
                vtr.mark("  V q=" + std::to_string(q) + " C2 span test");
            }   // D Ceq and the Ceq_{q+1} gather die here
            // C3 cocycles
            if (Zv.cols() != k - rank_DC[q]) fail(3, "dim Z != dim Ceq - rank(D Ceq)" + at);
            {
                const Buf DZ = block_times_dense(D[q], Zv);
                if (!is_zero_view(cview(DZ))) fail(3, "D Z != 0" + at);
            }
            if (rank_of_view(ctx, Zv) != Zv.cols()) fail(3, "Z not of full column rank" + at);
            vtr.mark("  V q=" + std::to_string(q) + " C3 up to the span test");
            if (!columns_in_span_tape(ex, Ceq[q], Cv, Zv)) fail(3, "Z not inside span Ceq" + at);
            vtr.mark("  V q=" + std::to_string(q) + " C3 span test (Cv " + std::to_string(Cv.cols()) + " | Z " + std::to_string(Zv.cols()) + ")");
        }   // the Ceq_q gather dies here: C4 and C5 do not read it
        // C4 coboundaries
        if (q == 0) {
            if (Bv.cols() != 0 || !B_pivots[0].empty()) fail(4, "B_0 must be empty");
        } else {
            Buf sCp;
            const CView Cpv = mv_view(Ceq[q - 1], sCp);
            const std::vector<uint32_t>& piv = B_pivots[q];
            if (Bv.cols() != piv.size()) fail(4, "#B columns != #pivots" + at);
            else {
                Buf sel(Cpv.rows(), static_cast<uint32_t>(piv.size()), Layout::ColMajor);
                bool piv_ok = true;
                for (size_t t = 0; t < piv.size(); ++t) {
                    if (piv[t] >= Cpv.cols()) { piv_ok = false; break; }
                    if (Cpv.rows() > 0) engine::kernels::copy<uint32_t>(Cpv.sub(0, Cpv.rows(), piv[t], 1), view(sel).sub(0, Cpv.rows(), static_cast<uint32_t>(t), 1));
                }
                if (!piv_ok) fail(4, "a B pivot is out of range" + at);
                else {
                    const Buf Dsel = block_times_dense(D[q - 1], cview(sel));
                    if (!views_equal(cview(Dsel), Bv)) fail(4, "B != D Ceq[:, pivots]" + at);
                }
                if (rank_of_view(ctx, Bv) != Bv.cols() || Bv.cols() != rank_DC[q - 1]) fail(4, "B not a basis of im D" + at);
            }
        }
        vtr.mark("  V q=" + std::to_string(q) + " C4");
        // C5 cohomology basis and projector
        {
            Buf sH;
            const CView Hv = mv_view(H[q], sH);
            const Buf DH = block_times_dense(D[q], Hv);
            if (!is_zero_view(cview(DH))) fail(5, "D H != 0" + at);
            const uint32_t dh = Hv.cols();
            const Buf PH = block_times_dense(P[q], Hv, Layout::RowMajor);
            bool ident = (PH.rows == dh && PH.cols == dh);
            for (uint32_t i = 0; ident && i < dh; ++i)
                for (uint32_t j = 0; j < dh; ++j)
                    if (PH.at(i, j) != (i == j ? 1u : 0u)) { ident = false; break; }
            if (!ident) fail(5, "P H != I" + at);
            const Buf PB = block_times_dense(P[q], Bv, Layout::RowMajor);
            if (!is_zero_view(cview(PB))) fail(5, "P B != 0" + at);
            if (!columns_in_span_tape(ex, Ceq[q], Zv, Hv)) fail(5, "H not inside span Z" + at);
            // rank [B | H] = rank B + dim H
            Buf BH(flat_dim[q], Bv.cols() + dh, Layout::ColMajor);
            if (flat_dim[q] > 0) {
                if (Bv.cols() > 0) engine::kernels::copy<uint32_t>(Bv, view(BH).sub(0, flat_dim[q], 0, Bv.cols()));
                if (dh > 0) engine::kernels::copy<uint32_t>(Hv, view(BH).sub(0, flat_dim[q], Bv.cols(), dh));
            }
            if (rank_of_view(ctx, cview(BH)) != Bv.cols() + dh) fail(5, "rank [B | H] != rank B + dim H" + at);
            if (dh != Zv.cols() - Bv.cols()) fail(5, "dim H != dim Z - rank B" + at);
        }
        vtr.mark("  V q=" + std::to_string(q) + " C5");
    }

    // ---- C6: the map families recomputed ------------------------------------
    for (uint32_t q = 0; q <= Qc + 1; ++q) {
        const std::string at = " at q = " + std::to_string(q);
        // pi: H^q_c -> H^q_eq
        if (pi_built[q]) {
            const uint32_t dc = dim_c(q), dk = dim_K(q);
            FlatMatrix want = FlatMatrix::CreateZero(dk, dc, ell);
            if (dc > 0 && dk > 0 && q <= Qc && q <= top_K) {
                Buf sH;
                const CView Hv = mv_view(H[q], sH);
                const CView bulk_part = Hv.sub(0, bulk_part_dim[q], 0, dc);
                const Buf Y = block_times_dense(P_bulk[q], bulk_part, Layout::RowMajor);
                want = flat_of(cview(Y), ell);
            }
            if (!flat_equal(want, pi[q])) fail(6, "pi recomputed != stored" + at);
        }
        // r: H^q_eq -> H^q_bd
        if (r_built[q]) {
            const uint32_t dk = dim_K(q), db = dim_bd(q);
            FlatMatrix want = FlatMatrix::CreateZero(db, dk, ell);
            if (dk > 0 && db > 0 && q <= top_K && q <= top_bd) {
                Buf sH;
                const CView HK = mv_view(H_bulk[q], sH);
                const Buf rX = block_times_dense(r_flat[q], HK);
                const Buf Y = block_times_dense(P_bd[q], cview(rX), Layout::RowMajor);
                want = flat_of(cview(Y), ell);
            }
            if (!flat_equal(want, r[q])) fail(6, "r recomputed != stored" + at);
        }
        // iota: H^q_bd -> H^{q+1}_c
        if (iota_built[q]) {
            const uint32_t db = dim_bd(q), dc1 = dim_c(static_cast<int64_t>(q) + 1);
            FlatMatrix want = FlatMatrix::CreateZero(dc1, db, ell);
            if (db > 0 && dc1 > 0 && q <= top_bd && q + 1 <= Qc) {
                Buf sH;
                const CView Hb = mv_view(H_bd[q], sH);
                const uint32_t fd = flat_dim[q + 1], off = bulk_part_dim[q + 1];
                Buf X(fd, db, Layout::ColMajor);
                if (Hb.rows() != fd - off) fail(6, "boundary basis height != boundary part" + at);
                else {
                    if (Hb.rows() > 0) engine::kernels::copy<uint32_t>(Hb, view(X).sub(off, Hb.rows(), 0, db));
                    const Buf Y = block_times_dense(P[q + 1], cview(X), Layout::RowMajor);
                    want = flat_of(cview(Y), ell);
                }
            }
            if (!flat_equal(want, iota[q])) fail(6, "iota recomputed != stored" + at);
        }
    }

    // ---- C7 / C8 / C9: the spots -------------------------------------------
    auto map_rank = [&](const FlatMatrix& M) { return rank_of_flat(ctx, M); };
    auto composite_zero = [&](const FlatMatrix& out, const FlatMatrix& in) {
        // out * in == 0
        if (out.get_cols() != in.get_rows()) return false;
        if (out.get_rows() == 0 || in.get_cols() == 0 || in.get_rows() == 0) return true;
        const Buf od = dense_of_flat(out), id = dense_of_flat(in);
        Buf prod(out.get_rows(), in.get_cols(), Layout::RowMajor);
        engine::kernels::gemm<uint32_t>(ctx, cview(od), cview(id), view(prod));
        return is_zero_view(cview(prod));
    };
    // The three maps around a spot, as (in, out) with their built flags and
    // the middle dimension; a map from / to a reported zero is a zero map.
    struct MapRef { bool built; const FlatMatrix* M; };
    auto map_pi = [&](int64_t q) -> MapRef { if (q < 0 || q > static_cast<int64_t>(Qc) + 1) return { false, nullptr }; return { pi_built[static_cast<size_t>(q)] != 0, &pi[static_cast<size_t>(q)] }; };
    auto map_r = [&](int64_t q) -> MapRef { if (q < 0 || q > static_cast<int64_t>(Qc) + 1) return { false, nullptr }; return { r_built[static_cast<size_t>(q)] != 0, &r[static_cast<size_t>(q)] }; };
    auto map_iota = [&](int64_t q) -> MapRef {
        if (q < 0) { static const FlatMatrix none; return { true, &none }; }   // from H^{-1}_bd = 0: the zero map (rank 0)
        if (q > static_cast<int64_t>(Qc) + 1) return { false, nullptr };
        return { iota_built[static_cast<size_t>(q)] != 0, &iota[static_cast<size_t>(q)] };
    };
    for (const ConeLesSpot& s : spots) {
        const int64_t q = s.q;
        MapRef in, out;
        uint32_t mid = 0;
        bool want_asserted = false;
        if (s.term == "Hc") { in = map_iota(q - 1); out = map_pi(q); mid = dim_c(q); want_asserted = boundary.reported(q - 1) && cone.reported(q) && bulk.reported(q); }
        else if (s.term == "H_bulk") { in = map_pi(q); out = map_r(q); mid = dim_K(q); want_asserted = cone.reported(q) && bulk.reported(q) && boundary.reported(q); }
        else if (s.term == "H_bd") { in = map_r(q); out = map_iota(q); mid = dim_bd(q); want_asserted = bulk.reported(q) && boundary.reported(q) && cone.reported(q + 1); }
        else { fail(13, "unknown spot term " + s.term); continue; }
        const std::string where = " at " + s.term + "^" + std::to_string(q);
        if (want_asserted != s.asserted) { fail(13, "spot status recorded != derived" + where); continue; }
        if (s.dim_middle != mid) fail(13, "spot middle dimension" + where);
        if (!s.asserted) continue;
        if (!in.built || !out.built) { fail(7, "an asserted spot has an unbuilt map" + where); continue; }
        // A zero in-map (q - 1 < 0) has no matrix: rank 0, composite trivially zero.
        const uint32_t rin = (in.M->get_rows() == 0 && in.M->get_cols() == 0) ? 0u : map_rank(*in.M);
        const uint32_t rout = (out.M->get_rows() == 0 && out.M->get_cols() == 0) ? 0u : map_rank(*out.M);
        if (in.M->get_cols() != 0 || in.M->get_rows() != 0) {
            if (!composite_zero(*out.M, *in.M)) fail(7, "composite out . in != 0" + where);
        }
        if (ring_is_field) {
            if (rin + rout != mid) fail(8, "ker != im (rank identity " + std::to_string(rin) + " + " + std::to_string(rout) + " != " + std::to_string(mid) + ")" + where);
        }
        // C9: im(pi^q) = ker r^q explicitly, at the bulk spot.
        if (s.term == "H_bulk" && ring_is_field) {
            const FlatMatrix& Pi = *in.M;   // dk x dc
            const FlatMatrix& R = *out.M;   // db x dk
            const uint32_t dk = mid;
            if (dk > 0) {
                // ker r^q as a kernel basis (columns), from an own RREF.
                Buf Rd = dense_of_flat(R);
                std::vector<uint32_t> piv;
                if (Rd.rows > 0 && Rd.cols > 0) piv = engine::elim::rref<uint32_t>(ctx, view(Rd)).pivots;
                const Buf Kr = (Rd.rows > 0 && Rd.cols > 0) ? engine::elim::kernel_basis<uint32_t>(ctx, cview(Rd), piv)
                                                            : Buf(dk, dk, Layout::RowMajor);
                Buf Kcols(dk, Kr.rows, Layout::ColMajor);
                if (Rd.rows > 0 && Rd.cols > 0) {
                    for (uint32_t t = 0; t < Kr.rows; ++t)
                        for (uint32_t i = 0; i < dk; ++i) Kcols.at(i, t) = Kr.at(t, i);
                } else {
                    for (uint32_t i = 0; i < dk; ++i) Kcols.at(i, i) = 1;
                }
                const Buf Pd = dense_of_flat(Pi);
                // im pi inside ker r: r pi = 0 (already C7) and each pi column inside span(K)
                if (!columns_in_span(ctx, cview(Kcols), cview(Pd))) fail(9, "im pi not inside ker r" + where);
                // ker r inside im pi
                if (!columns_in_span(ctx, cview(Pd), cview(Kcols))) fail(9, "ker r not inside im pi" + where);
            }
        }
    }

    // ---- C10: H^0_c = 0 -------------------------------------------------------
    if (dim_Hc[0] != 0) fail(10, "H^0_c != 0 (dim " + std::to_string(dim_Hc[0]) + ")");

    // ---- C11 / C12: H^2_eq and eq:conedims -------------------------------------
    {
        const bool H2_reported = bulk.reported(2);
        const bool H2_zero = H2_reported && dim_K(2) == 0;
        if (six_invertible && H2_reported && !H2_zero) fail(12, "H^2_eq != 0 at 6 in R^x (dim " + std::to_string(dim_K(2)) + ")");
        // eq:conedims (1) -- dim H^1_c = dim H^0(bd) - dim H^0_eq + dim ker r^1 -- is
        // UNCONDITIONAL (lem:conerelative(5); obl:coneexact(v), v1.6.0): asserted on every
        // computed cell, with no reference to dim H^2_eq. Its own guard therefore carries no
        // H2_zero conjunct.
        if (ring_is_field && cone.reported(1) && bulk.reported(0) && bulk.reported(1) &&
            boundary.reported(0) && r_built[1] && r_built[0]) {
            const uint32_t rank_r1 = map_rank(r[1]);
            const uint32_t ker_r1 = dim_K(1) - rank_r1;
            const uint32_t lhs1 = dim_c(1);
            const int64_t rhs1 = static_cast<int64_t>(dim_bd(0)) - static_cast<int64_t>(dim_K(0)) + static_cast<int64_t>(ker_r1);
            if (static_cast<int64_t>(lhs1) != rhs1) fail(11, "eq:conedims (1): dim H^1_c = " + std::to_string(lhs1) + " != " + std::to_string(rhs1));
        }
        // eq:conedims (2) -- dim H^2_c = dim H^1(bd) - rank r^1 -- needs H^2_eq = 0 in the
        // stored data (obl:coneexact(v)); where H^2_eq != 0 or is unreported it is RECORDED
        // (item 12, above) rather than asserted here.
        if (H2_zero && ring_is_field && cone.reported(2) && boundary.reported(1) && r_built[1]) {
            const uint32_t rank_r1 = map_rank(r[1]);
            const int64_t rhs2 = static_cast<int64_t>(dim_bd(1)) - static_cast<int64_t>(rank_r1);
            if (static_cast<int64_t>(dim_c(2)) != rhs2) fail(11, "eq:conedims (2): dim H^2_c = " + std::to_string(dim_c(2)) + " != " + std::to_string(rhs2));
        }
    }
    return static_cast<uint32_t>(failures.size());
}

// ===========================================================================
// EquivariantMappingCone
// ===========================================================================

EquivariantMappingCone::EquivariantMappingCone(const EquivariantCohomologyOnK* bulk, const EquivariantCohomologyOnK* boundary,
                                               const EquivariantCochainComplexMorphismOnK* restriction,
                                               EquivariantMappingConeOptions options)
    : bulk_(bulk), boundary_(boundary), res_(restriction), options_(std::move(options)) {
    // (slot-9 WP-4; E8 item 4b; [E] v1.10.0-pre.1 sec:ring (f)) Over Z/ell^k the
    // cone HONOURS the ring by the dense module route (ModuleConeT) when both
    // inputs are over the ring in force; every other combination keeps slot 8's
    // refusal (the ring in force against the inputs', by name).
    {
        const uint32_t kb = bulk ? bulk->coefficient_ring_k() : 1u, kd = boundary ? boundary->coefficient_ring_k() : 1u;
        if (kb >= 2 && kb == kd && hecke::CoefficientRingK() == kb) {
            init_composite();
            return;
        }
    }
    // (slot-8 WP-3) [E] sec:ring (b): the cone is a surface; its Z/ell^k form is item 4b's.
    hecke::RequireFieldCoefficientRing("EquivariantMappingCone",
                                       std::max(bulk ? bulk->coefficient_ring_k() : 1u,
                                                boundary ? boundary->coefficient_ring_k() : 1u));
    if (bulk_ == nullptr || boundary_ == nullptr || res_ == nullptr) throw std::invalid_argument("EquivariantMappingCone: null input");
    if (!bulk_->is_cohomology_computed() || !boundary_->is_cohomology_computed())
        throw std::runtime_error("EquivariantMappingCone: an input complex is not computed");
    if (!res_->are_maps_computed()) throw std::runtime_error("EquivariantMappingCone: the restriction morphism is not computed");
    if (res_->get_domain_lssc_cohomology() != boundary_ || res_->get_codomain_lssc_cohomology() != bulk_)
        throw std::invalid_argument("EquivariantMappingCone: the morphism's domain must be the boundary and its codomain the bulk");
    ell_ = bulk_->get_prime();
    d_V_ = bulk_->get_representation_dim();
    // (hecke-extensions slot 23, LES-2) the boundary's fibre may differ from the bulk's on the F_ell route: the level cone of
    // [C, def:levelcone] has d_A = m_A (n + 1) != d_B = m_B (n + 1).  d_bd_ == d_V_ for every boundary pair, and then every
    // statement below is the pre-LES-2 one (the Z/ell^k route keeps the equal-fibre refusal).
    d_bd_ = boundary_->get_representation_dim();
    if (boundary_->get_prime() != ell_) throw std::invalid_argument("EquivariantMappingCone: prime mismatch");
    if (d_V_ == 0 || d_bd_ == 0) throw std::invalid_argument("EquivariantMappingCone: a fibre dimension is 0 (d_V or d_V_bd)");
    if (!EquivariantCohomologyOptions::retain_intermediates())
        throw std::runtime_error("EquivariantMappingCone: needs the bulk's C_eq (retain_intermediates is off)");
    top_K_ = static_cast<uint32_t>(bulk_->get_H_eq_dimensions().size() - 1);
    top_bd_ = static_cast<uint32_t>(boundary_->get_H_eq_dimensions().size() - 1);
    // Degree fields: derived, or an E9-shaped declaration (open:ell2 rider (i)).
    auto take = [&](bool override_, const ComplexDegreeFields& given, const EquivariantCohomologyOnK& coh, uint32_t top, const char* which) {
        ComplexDegreeFields f = ComplexDegreeFields::derived_from(coh);
        if (!override_) return f;
        if (!given.truncated) {
            if (given.q_max != top) throw std::invalid_argument(std::string("EquivariantMappingCone: a set q_max on the finite ") + which +
                                                                " complex is rejected (open:ell2 rider (i))");
            return f;   // a no-op override
        }
        if (given.q_max >= top) throw std::invalid_argument(std::string("EquivariantMappingCone: a declared truncation of the ") + which +
                                                            " complex needs q_max < its top degree (degree q_max + 1 is the scaffolding)");
        ComplexDegreeFields g = ComplexDegreeFields::declared_truncation(given.q_max);
        return g;
    };
    bulk_fields_ = take(options_.override_bulk_fields, options_.bulk_fields, *bulk_, top_K_, "bulk");
    boundary_fields_ = take(options_.override_boundary_fields, options_.boundary_fields, *boundary_, top_bd_, "boundary");
    cone_fields_ = derive_cone_fields(bulk_fields_, boundary_fields_, top_K_, top_bd_);
}

EquivariantMappingCone::~EquivariantMappingCone() = default;

// (slot-9 WP-4) The composite constructor tail: the checks of the field
// constructor, restated for Z/ell^k inputs (no degree-field override: the
// finite model's fields are derived).
void EquivariantMappingCone::init_composite() {
    if (bulk_ == nullptr || boundary_ == nullptr || res_ == nullptr) throw std::invalid_argument("EquivariantMappingCone: null input");
    ring_k_ = bulk_->coefficient_ring_k();
    const engine::ModCtx<uint32_t>& ctx = bulk_->ring_context();
    if (boundary_->ring_context().p != ctx.p)
        throw std::logic_error("EquivariantMappingCone: the bulk and the boundary are over different rings");
    if (ctx.ell == 2 || ctx.ell == 3 || bulk_->is_flat_route() || boundary_->is_flat_route())
        hecke::RefuseCoefficientRing("EquivariantMappingCone", ring_k_,
                                     " at ell = " + std::to_string(ctx.ell) +
                                         " (the cone over Z/ell^k is honoured at ell >= 5 only; [E] sec:ring (f))");
    if (!bulk_->is_cohomology_computed() || !boundary_->is_cohomology_computed())
        throw std::runtime_error("EquivariantMappingCone: an input complex is not computed");
    if (!res_->are_maps_computed()) throw std::runtime_error("EquivariantMappingCone: the restriction morphism is not computed");
    if (!res_->is_composite()) throw std::logic_error("EquivariantMappingCone: the restriction morphism is over F_ell");
    if (res_->get_domain_lssc_cohomology() != boundary_ || res_->get_codomain_lssc_cohomology() != bulk_)
        throw std::invalid_argument("EquivariantMappingCone: the morphism's domain must be the boundary and its codomain the bulk");
    if (options_.override_bulk_fields || options_.override_boundary_fields)
        throw std::logic_error("EquivariantMappingCone: a degree-field override is not supported over Z/ell^k");
    ell_ = ctx.ell;
    d_V_ = bulk_->get_representation_dim();
    if (boundary_->get_representation_dim() != d_V_) throw std::invalid_argument("EquivariantMappingCone: fibre dimension mismatch");
    d_bd_ = d_V_;
    if (d_V_ == 0) throw std::invalid_argument("EquivariantMappingCone: d_V = 0");
    top_K_ = bulk_->num_degrees() - 1;
    top_bd_ = boundary_->num_degrees() - 1;
    auto derived = [](uint32_t top) {
        ComplexDegreeFields f;
        f.q_max = top;
        for (uint32_t q = 0; q <= top; ++q) f.degrees.push_back(q);
        return f;
    };
    bulk_fields_ = derived(top_K_);
    boundary_fields_ = derived(top_bd_);
    cone_fields_ = derive_cone_fields(bulk_fields_, boundary_fields_, top_K_, top_bd_);
}

void EquivariantMappingCone::require_field(const char* who) const {
    if (ring_k_ >= 2)
        throw std::logic_error(std::string("EquivariantMappingCone::") + who + ": a field (F_ell) accessor, but this cone is over " +
                               hecke::CoefficientRingText(ring_k_) + " -- use module_cone() ([E] sec:ring (f))");
}

const ModuleConeT<uint32_t>& EquivariantMappingCone::module_cone() const {
    if (ring_k_ < 2) throw std::logic_error("EquivariantMappingCone::module_cone: this cone is over F_ell");
    require_computed("module_cone");
    return *module_;
}

void EquivariantMappingCone::require_computed(const char* who) const {
    if (!computed_) throw std::runtime_error(std::string("EquivariantMappingCone::") + who + ": compute() has not run");
}

void EquivariantMappingCone::check_cone_degree(uint32_t q, uint32_t max_q, const char* who) const {
    if (q > max_q) throw std::out_of_range(std::string("EquivariantMappingCone::") + who + ": degree " + std::to_string(q) +
                                           " > " + std::to_string(max_q));
}

uint32_t EquivariantMappingCone::bulk_part_dim(uint32_t q) const {
    const SimplicialComplex* K = bulk_->get_lssc()->get_base_K_complex();
    return (q <= top_K_) ? K->get_num_q_simplices(q) * d_V_ : 0u;
}

uint32_t EquivariantMappingCone::boundary_part_dim(uint32_t q) const {
    if (q == 0) return 0;
    const SimplicialComplex* Kb = boundary_->get_lssc()->get_base_K_complex();
    return (q - 1 <= top_bd_) ? Kb->get_num_q_simplices(q - 1) * d_bd_ : 0u;
}

uint32_t EquivariantMappingCone::flat_dim(uint32_t q) const { return bulk_part_dim(q) + boundary_part_dim(q); }

std::vector<uint32_t> EquivariantMappingCone::row_dims(uint32_t q) const {
    std::vector<uint32_t> d;
    const uint32_t nb = bulk_part_dim(q) / d_V_, nd = boundary_part_dim(q) / d_bd_;
    d.assign(nb + nd, d_V_);
    for (uint32_t i = nb; i < nb + nd; ++i) d[i] = d_bd_;   // (LES-2) the boundary blocks carry d_bd (== d_V for a boundary pair)
    return d;
}

// The equivariant cone cochains as one block-diagonal multi-vector.
BM EquivariantMappingCone::assemble_Ceq(uint32_t q) const {
    const uint32_t nK = bulk_part_dim(q), nb = boundary_part_dim(q);
    const uint32_t kK = (q <= top_K_) ? bulk_->C_eq_block(q).cols() : 0u;
    const uint32_t kb = (q >= 1 && q - 1 <= top_bd_) ? boundary_->C_eq_block(q - 1).cols() : 0u;
    Buf buf(nK + nb, kK + kb, Layout::ColMajor);
    if (nK > 0 && kK > 0) {
        const Buf CK = bulk_->C_eq_block(q).to_flat(Layout::ColMajor);
        ENGINE_CHECK_MSG(CK.rows == nK, "EquivariantMappingCone: bulk C_eq height != N_q d_V");
        engine::kernels::copy<uint32_t>(cview(CK), view(buf).sub(0, nK, 0, kK));
    }
    if (nb > 0 && kb > 0) {
        const Buf Cb = boundary_->C_eq_block(q - 1).to_flat(Layout::ColMajor);
        ENGINE_CHECK_MSG(Cb.rows == nb, "EquivariantMappingCone: boundary C_eq height != N_{q-1} d_V");
        engine::kernels::copy<uint32_t>(cview(Cb), view(buf).sub(nK, nb, kK, kb));
    }
    const std::vector<uint32_t> rd = row_dims(q);
    if (rd.empty()) return BM::zeros(pool_, rd, std::vector<uint32_t>{ kK + kb }, ell_);
    return BM::from_flat(pool_, std::move(buf), rd, ell_, /*detect=*/false);
}

// The flat restriction r^q : C^q(K; V) -> C^q(K_bd; V) from the morphism's
// fibre block table (a general morphism: every block interned by content;
// the identity fibre of an inclusion is a Scalar 1 ref with no data).
BM EquivariantMappingCone::assemble_r_flat(uint32_t q) const {
    const SimplicialComplex* K = bulk_->get_lssc()->get_base_K_complex();
    const SimplicialComplex* Kb = boundary_->get_lssc()->get_base_K_complex();
    const uint32_t NK = K->get_num_q_simplices(q), Nb = Kb->get_num_q_simplices(q);
    const std::vector<uint32_t> rd(Nb, d_bd_), cd(NK, d_V_);
    BlockBuilder<uint32_t> b(pool_, rd, cd, ell_);
    const FiberBlockTable& tab = res_->get_fiber_blocks();
    if (q < tab.size()) {
        ENGINE_CHECK_MSG(tab[q].size() == Nb || tab[q].empty(), "EquivariantMappingCone: fibre table height != N_q(K_bd)");
        for (uint32_t s1 = 0; s1 < tab[q].size(); ++s1) {
            for (const PerSimplexBlock& blk : tab[q][s1]) {
                if (blk.s2_idx_in_K2 >= NK) continue;   // the morphism's own guarded-loop semantics: no such column
                ENGINE_CHECK_MSG(blk.F.get_rows() == d_bd_ && blk.F.get_cols() == d_V_, "EquivariantMappingCone: fibre block shape");
                uint32_t sv = 0;
                BlockRef<uint32_t> ref;
                if (blk.F.is_currently_scalar(sv)) ref = BlockRef<uint32_t>::scalar(sv);
                else {
                    blk.F.populate_flat_data();
                    const std::vector<uint32_t>& fd = blk.F.get_flat_data_unsafe();
                    ref = b.intern_dense(fd, d_bd_, d_V_, Layout::RowMajor, true);
                }
                b.accumulate(s1, blk.s2_idx_in_K2, ref);
            }
        }
    }
    return b.finalize();
}

// D_c^q : Cc^q -> Cc^{q+1},  (c, c_bd) |-> (-d c, d_bd c_bd + r c).
BM EquivariantMappingCone::assemble_D(uint32_t q) const {
    const SimplicialComplex* K = bulk_->get_lssc()->get_base_K_complex();
    const SimplicialComplex* Kb = boundary_->get_lssc()->get_base_K_complex();
    const std::vector<uint32_t> rd = row_dims(q + 1), cd = row_dims(q);
    BlockBuilder<uint32_t> b(pool_, rd, cd, ell_);
    const ModCtx<uint32_t>& ctx = b.ctx();
    const uint32_t NK_q = bulk_part_dim(q) / d_V_, NK_q1 = bulk_part_dim(q + 1) / d_V_;
    const uint32_t Nb_qm1 = boundary_part_dim(q) / d_bd_, Nb_q = boundary_part_dim(q + 1) / d_bd_;
    // -d_q on the bulk part.
    if (NK_q > 0 && NK_q1 > 0) {
        const FlatMatrix D1 = K->coboundary_operator_elt<uint32_t>(q, ell_);
        ENGINE_CHECK_MSG(D1.get_rows() == NK_q1 && D1.get_cols() == NK_q, "EquivariantMappingCone: bulk coboundary shape");
        for (uint32_t i = 0; i < NK_q1; ++i)
            for (uint32_t j = 0; j < NK_q; ++j) {
                const uint32_t s = D1.get_entry(i, j);
                if (s != 0) b.set_scalar(i, j, ctx.neg(s));
            }
    }
    // +d_bd,{q-1} on the boundary part.
    if (Nb_qm1 > 0 && Nb_q > 0) {
        const FlatMatrix Db = Kb->coboundary_operator_elt<uint32_t>(q - 1, ell_);
        ENGINE_CHECK_MSG(Db.get_rows() == Nb_q && Db.get_cols() == Nb_qm1, "EquivariantMappingCone: boundary coboundary shape");
        for (uint32_t i = 0; i < Nb_q; ++i)
            for (uint32_t j = 0; j < Nb_qm1; ++j) {
                const uint32_t s = Db.get_entry(i, j);
                if (s != 0) b.set_scalar(NK_q1 + i, NK_q + j, s);
            }
    }
    // +r^q from the bulk part into the boundary part.
    if (NK_q > 0 && Nb_q > 0 && q <= top_K_ && q <= top_bd_) {
        // r_flat lives in the same pool, so its refs are valid in this builder as they are.
        const BM& rq = cert_.r_flat[q];
        for (uint32_t br = 0; br < rq.nbr(); ++br)
            for (uint32_t k = rq.row_begin(br); k < rq.row_end(br); ++k)
                b.accumulate(NK_q1 + br, rq.col_at(k), rq.ref_at(k));
    }
    return b.finalize();
}

void EquivariantMappingCone::compute() {
    if (computed_) return;
    // (slot-23 PERF-P2) The host eliminations below (the DC RREF and kernel per degree, the maps'
    // ranks, the certificate's verify()) split their trailing updates over the consumer pool;
    // value-free (EngineRuntime.h, HostElimThreadDeterminismValidation).
    const engine::tape::ScopedHostElimHook host_elim_hook;
    if (ring_k_ >= 2) {   // (slot-9 WP-4) the dense module route; the tape route below is the F_ell path
        std::vector<engine::DenseBuf<uint32_t>> rf;
        for (uint32_t q = 0; q <= std::min(top_K_, top_bd_); ++q) rf.push_back(res_->pullback_matrix_module(q));
        auto m = std::make_unique<ModuleConeT<uint32_t>>(bulk_, boundary_, std::move(rf));
        m->compute();
        module_ = std::move(m);
        computed_ = true;
        return;
    }
    ConeTrace trace;
    pool_ = std::make_shared<BlockPool>();
    const uint32_t Qc = cone_fields_.q_max;
    cert_ = ConeCertificate();
    cert_.ell = ell_;
    cert_.d_V = d_V_;
    cert_.ring_is_field = true;
    cert_.six_invertible = (ell_ >= 5);
    cert_.bulk = bulk_fields_;
    cert_.boundary = boundary_fields_;
    cert_.cone = cone_fields_;

    // The inputs' stored bases and projectors (copied as block objects over
    // OUR pool, so the certificate is self-contained).
    for (uint32_t q = 0; q <= top_K_; ++q) {
        const Buf Hd = bulk_->H_eq_block(q).to_flat(Layout::ColMajor);
        const Buf Pd = bulk_->P_block(q).to_dense(Layout::RowMajor);
        const std::vector<uint32_t> rd(bulk_->get_lssc()->get_base_K_complex()->get_num_q_simplices(q), d_V_);
        cert_.H_bulk.push_back(rd.empty() ? BM::zeros(pool_, rd, std::vector<uint32_t>{ Hd.cols }, ell_)
                                          : BM::from_flat(pool_, cview(Hd), rd, ell_, Layout::ColMajor, false));
        cert_.P_bulk.push_back(BM::from_dense(pool_, cview(Pd), std::vector<uint32_t>{ Pd.rows }, rd, ell_, Layout::RowMajor, true));
        cert_.dim_H_bulk.push_back(bulk_->get_H_eq_dimensions()[q]);
    }
    for (uint32_t q = 0; q <= top_bd_; ++q) {
        const Buf Hd = boundary_->H_eq_block(q).to_flat(Layout::ColMajor);
        const Buf Pd = boundary_->P_block(q).to_dense(Layout::RowMajor);
        const std::vector<uint32_t> rd(boundary_->get_lssc()->get_base_K_complex()->get_num_q_simplices(q), d_bd_);
        cert_.H_bd.push_back(rd.empty() ? BM::zeros(pool_, rd, std::vector<uint32_t>{ Hd.cols }, ell_)
                                        : BM::from_flat(pool_, cview(Hd), rd, ell_, Layout::ColMajor, false));
        cert_.P_bd.push_back(BM::from_dense(pool_, cview(Pd), std::vector<uint32_t>{ Pd.rows }, rd, ell_, Layout::RowMajor, true));
        cert_.dim_H_bd.push_back(boundary_->get_H_eq_dimensions()[q]);
    }
    for (uint32_t q = 0; q <= std::min(top_K_, top_bd_); ++q) cert_.r_flat.push_back(assemble_r_flat(q));
    trace.mark("inputs H/P/r_flat");

    // The cone cochain groups and differentials.
    for (uint32_t q = 0; q <= Qc + 1; ++q) {
        cert_.flat_dim.push_back(flat_dim(q));
        cert_.bulk_part_dim.push_back(bulk_part_dim(q));
        cert_.Ceq.push_back(assemble_Ceq(q));
    }
    for (uint32_t q = 0; q <= Qc; ++q) cert_.D.push_back(assemble_D(q));
    trace.mark("Ceq + D");

    // Z, B, H, P per degree, ON THE TAPE (hecke-extensions slot 5, WP-5).
    //
    // The two block GEMMs (DC = D_q . C^q_eq and Z^q = C^q_eq . Kr^T) are
    // recorded as engine::tape gemm_dense programs on the consumer pool and
    // served from the program cache by shape signature; the H / P quotient
    // goes through engine::tape::factor_basis, the identity-augmentation-free
    // panel driver EquivariantCohomologyOnK already uses.  Values are
    // unchanged: the tape kernels accumulate exactly with one reduction per
    // k-block (ENGINE_API.md Sec. 5), and tape::factor_basis produces the same
    // D5b pivots, basis and projector as blk::factor_basis because appending
    // the ambient identity columns cannot change the pivot columns among the
    // earlier ones and the RREF is unique (TapeElim.h; gated entrywise by
    // EngineTapeEliminationValidation).
    //
    // What the identity augmentation cost, measured: at n_sym = 256 the
    // q = 1 call is factor_basis(B 985, Z 1028) over an ambient a = 3855, so
    // blk::factor_basis eliminates a 3855 x 5868 augmented matrix whose RREF
    // R is 3855 x 5868 = 90.4 MB, of which the identity part alone is
    // 3855^2 x 4 B = 59.4 MB.  It was the whole of the cone's high-water mark.
    const ModCtx<uint32_t> ctx(ell_);
    engine::tape::Executor<uint32_t> ex(&engine::tape::consumer_pool());
    std::vector<Buf> image_next(Qc + 2);              // B_{q+1} as a column-major buffer
    std::vector<std::vector<uint32_t>> pivots_next(Qc + 2);
    cert_.Z.assign(Qc + 1, BM());
    cert_.B.assign(Qc + 1, BM());
    cert_.H.assign(Qc + 1, BM());
    cert_.P.assign(Qc + 1, BM());
    cert_.B_pivots.assign(Qc + 1, {});
    cert_.dim_Hc.assign(Qc + 1, 0);
    for (uint32_t q = 0; q <= Qc; ++q) {
        const uint32_t k = cert_.Ceq[q].cols(), fd = cert_.flat_dim[q], fd1 = cert_.flat_dim[q + 1];
        const std::vector<uint32_t> rd = row_dims(q);
        // DC = D_q Ceq_q, its RREF, kernel and image (D5b).  The operations are
        // exactly the pre-slot-4 ones in the pre-slot-4 order; only the SCOPES
        // changed (slot-4 WP-4): the gathered C_eq multi-vector, DC, its RREF
        // copy and the kernel basis are each released at their last reader
        // instead of at the end of the iteration, so the transient peak is one
        // or two of them rather than all five at once.  No operand, no operand
        // order and no result changes.
        ConeTrace tr;
        tr.mark("  q=" + std::to_string(q) + " enter (fd=" + std::to_string(fd) + " fd1=" + std::to_string(fd1) + " k=" + std::to_string(k) + ")");
        engine::elim::RrefResult res;
        Buf Kr;
        bool kr_identity = false;   // (slot-24 G6) D_q has no rows: Kr = I
        {
            Buf R;
            {
                Buf DC;
                {
                    Buf sC;
                    const CView Cv = mv_view(cert_.Ceq[q], sC);
                    ENGINE_CHECK(Cv.cols() == k);
                    DC = tape_block_times_dense(ex, "cone.DC", q, cert_.D[q], Cv, Layout::RowMajor);
                }   // the gathered C_eq copy dies here: DC is its only reader
                tr.mark("  q=" + std::to_string(q) + " DC " + std::to_string(fd1) + "x" + std::to_string(k));
                R = dense_copy(cview(DC));
                if (R.rows > 0 && R.cols > 0) res = engine::elim::rref<uint32_t>(ctx, view(R));
                // B_{q+1} = the pivot columns of DC (the D5b image basis); taken
                // here, before the kernel basis, so that DC dies one scope earlier.
                Buf img(fd1, static_cast<uint32_t>(res.pivots.size()), Layout::ColMajor);
                if (fd1 > 0)
                    for (size_t t = 0; t < res.pivots.size(); ++t)
                        engine::kernels::copy<uint32_t>(cview(DC).sub(0, fd1, res.pivots[t], 1), view(img).sub(0, fd1, static_cast<uint32_t>(t), 1));
                image_next[q + 1] = std::move(img);
                pivots_next[q + 1] = res.pivots;
            }   // DC dies here
            // kernel: one row per kernel vector (k - rank) x k, or the identity when D has no rows
            if (R.rows > 0 && R.cols > 0) Kr = engine::elim::kernel_basis<uint32_t>(ctx, cview(R), res.pivots);
            else { kr_identity = true; Kr = Buf(k, k, Layout::RowMajor); for (uint32_t i = 0; i < k; ++i) Kr.at(i, i) = 1; }
        }   // the RREF buffer dies here
        tr.mark("  q=" + std::to_string(q) + " rref+image+kernel (rank=" + std::to_string(res.pivots.size()) + ", Kr " + std::to_string(Kr.rows) + "x" + std::to_string(Kr.cols) + ")");
        // Z = Ceq . Kr^T
        if (kr_identity && oc_util::field_diet_on() && fd > 0 && k > 0 && !rd.empty()) {
            // (slot-24 G6, Util/FieldDiet.h) D_q has no rows, so Kr is the k x k
            // identity and Z = Ceq . I^T IS Ceq: the stored basis shares Ceq's
            // pool blocks instead of a dense product copy (0.25 GB at the C1a
            // cone's q = 2, n_sym = 1000).  The same entries.  Memory only.
            Kr = Buf();
            cert_.Z[q] = cert_.Ceq[q];
        } else {
            Buf Zbuf = (fd > 0 && Kr.rows > 0 && k > 0)
                           ? tape_block_times_dense(ex, "cone.Z", q, cert_.Ceq[q], cview(Kr).transposed_view(), Layout::ColMajor)
                           : Buf(fd, Kr.rows, Layout::ColMajor);
            Kr = Buf();   // the kernel basis has no reader after Z
            cert_.Z[q] = rd.empty() ? BM::zeros(pool_, rd, std::vector<uint32_t>{ Zbuf.cols }, ell_)
                                    : BM::from_flat(pool_, std::move(Zbuf), rd, ell_, false);
        }
        tr.mark("  q=" + std::to_string(q) + " Z " + std::to_string(cert_.Z[q].rows()) + "x" + std::to_string(cert_.Z[q].cols()));
        // B_q (from the previous degree; empty at q = 0)
        if (q == 0) cert_.B[q] = BM::zeros(pool_, rd, std::vector<uint32_t>{ 0 }, ell_);
        else {
            cert_.B[q] = rd.empty() ? BM::zeros(pool_, rd, std::vector<uint32_t>{ image_next[q].cols }, ell_)
                                    : BM::from_flat(pool_, std::move(image_next[q]), rd, ell_, false);
            cert_.B_pivots[q] = pivots_next[q];
        }
        // H, P = factor_basis(B, Z)
        {
            Buf sB, sZ;
            const CView Bv = mv_view(cert_.B[q], sB);
            const CView Zv = mv_view(cert_.Z[q], sZ);
            engine::blk::FactorBasisResult<uint32_t> Hr =
                engine::tape::factor_basis<uint32_t>(ex, pool_, ell_, rd, Bv, Zv);
            const uint32_t dim_H = static_cast<uint32_t>(Hr.pivots.size());
            cert_.dim_Hc[q] = dim_H;
            cert_.H[q] = std::move(Hr.basis);
            ENGINE_CHECK(Hr.projector.rows == dim_H && Hr.projector.cols == fd);
            cert_.P[q] = BM::from_dense(pool_, std::move(Hr.projector), std::vector<uint32_t>{ dim_H }, rd, ell_, true);
        }
        tr.mark("  q=" + std::to_string(q) + " factor_basis(B " + std::to_string(cert_.B[q].cols()) + ", Z " + std::to_string(cert_.Z[q].cols()) +
                ") over a=" + std::to_string(fd) + " -> dim_H=" + std::to_string(cert_.dim_Hc[q]));
    }

    trace.mark("Z/B/H/P (all q)");
    build_maps();
    build_spots();
    trace.mark("maps + spots");

    std::vector<std::string> failures;
    const uint32_t bad = cert_.verify(failures);
    trace.mark("verify");
    if (bad != 0) {
        std::string msg = "EquivariantMappingCone::compute(): the certificate fails on the freshly built cone:";
        for (const auto& f : failures) msg += " [" + f + "]";
        throw std::runtime_error(msg);
    }
    computed_ = true;
}

// The three map families in the stored bases, wherever both end terms are
// reported (a reported zero gives a zero-dimensional matrix).
void EquivariantMappingCone::build_maps() {
    const uint32_t Qc = cone_fields_.q_max;
    cert_.pi.assign(Qc + 2, FlatMatrix());
    cert_.r.assign(Qc + 2, FlatMatrix());
    cert_.iota.assign(Qc + 2, FlatMatrix());
    cert_.pi_built.assign(Qc + 2, 0);
    cert_.r_built.assign(Qc + 2, 0);
    cert_.iota_built.assign(Qc + 2, 0);
    rank_pi_.assign(Qc + 2, 0);
    rank_r_.assign(Qc + 2, 0);
    rank_iota_.assign(Qc + 2, 0);
    const ModCtx<uint32_t> ctx(ell_);
    auto dim_c = [&](int64_t q) -> uint32_t { return (q < 0 || q > static_cast<int64_t>(Qc)) ? 0u : cert_.dim_Hc[static_cast<size_t>(q)]; };
    auto dim_K = [&](int64_t q) -> uint32_t { return (q < 0 || q > static_cast<int64_t>(bulk_fields_.q_max)) ? 0u : cert_.dim_H_bulk[static_cast<size_t>(q)]; };
    auto dim_bd = [&](int64_t q) -> uint32_t { return (q < 0 || q > static_cast<int64_t>(boundary_fields_.q_max)) ? 0u : cert_.dim_H_bd[static_cast<size_t>(q)]; };
    for (uint32_t q = 0; q <= Qc + 1; ++q) {
        // pi^q : H^q_c -> H^q_eq
        if (cone_fields_.reported(q) && bulk_fields_.reported(q)) {
            const uint32_t dc = dim_c(q), dk = dim_K(q);
            FlatMatrix M = FlatMatrix::CreateZero(dk, dc, ell_);
            if (dc > 0 && dk > 0) {
                Buf sH;
                const CView Hv = mv_view(cert_.H[q], sH);
                const Buf Y = block_times_dense(cert_.P_bulk[q], Hv.sub(0, cert_.bulk_part_dim[q], 0, dc), Layout::RowMajor);
                M = flat_of(cview(Y), ell_);
            }
            rank_pi_[q] = rank_of_flat(ctx, M);
            cert_.pi[q] = std::move(M);
            cert_.pi_built[q] = 1;
        }
        // r^q : H^q_eq -> H^q_bd
        if (bulk_fields_.reported(q) && boundary_fields_.reported(q)) {
            const uint32_t dk = dim_K(q), db = dim_bd(q);
            FlatMatrix M = FlatMatrix::CreateZero(db, dk, ell_);
            if (dk > 0 && db > 0) {
                Buf sH;
                const CView HK = mv_view(cert_.H_bulk[q], sH);
                const Buf rX = block_times_dense(cert_.r_flat[q], HK);
                const Buf Y = block_times_dense(cert_.P_bd[q], cview(rX), Layout::RowMajor);
                M = flat_of(cview(Y), ell_);
            }
            rank_r_[q] = rank_of_flat(ctx, M);
            cert_.r[q] = std::move(M);
            cert_.r_built[q] = 1;
        }
        // iota^q : H^q_bd -> H^{q+1}_c
        if (boundary_fields_.reported(q) && cone_fields_.reported(static_cast<int64_t>(q) + 1)) {
            const uint32_t db = dim_bd(q), dc1 = dim_c(static_cast<int64_t>(q) + 1);
            FlatMatrix M = FlatMatrix::CreateZero(dc1, db, ell_);
            if (db > 0 && dc1 > 0) {
                Buf sH;
                const CView Hb = mv_view(cert_.H_bd[q], sH);
                const uint32_t fd = cert_.flat_dim[q + 1], off = cert_.bulk_part_dim[q + 1];
                ENGINE_CHECK_MSG(Hb.rows() == fd - off, "EquivariantMappingCone: boundary basis height != boundary part");
                Buf X(fd, db, Layout::ColMajor);
                engine::kernels::copy<uint32_t>(Hb, view(X).sub(off, Hb.rows(), 0, db));
                const Buf Y = block_times_dense(cert_.P[q + 1], cview(X), Layout::RowMajor);
                M = flat_of(cview(Y), ell_);
            }
            rank_iota_[q] = rank_of_flat(ctx, M);
            cert_.iota[q] = std::move(M);
            cert_.iota_built[q] = 1;
        }
    }
}

void EquivariantMappingCone::build_spots() {
    const uint32_t Qc = cone_fields_.q_max;
    cert_.spots.clear();
    auto dim_c = [&](int64_t q) -> uint32_t { return (q < 0 || q > static_cast<int64_t>(Qc)) ? 0u : cert_.dim_Hc[static_cast<size_t>(q)]; };
    auto dim_K = [&](int64_t q) -> uint32_t { return (q < 0 || q > static_cast<int64_t>(bulk_fields_.q_max)) ? 0u : cert_.dim_H_bulk[static_cast<size_t>(q)]; };
    auto dim_bd = [&](int64_t q) -> uint32_t { return (q < 0 || q > static_cast<int64_t>(boundary_fields_.q_max)) ? 0u : cert_.dim_H_bd[static_cast<size_t>(q)]; };
    auto rank_pi = [&](int64_t q) -> uint32_t { return (q < 0 || q > static_cast<int64_t>(Qc) + 1) ? 0u : rank_pi_[static_cast<size_t>(q)]; };
    auto rank_r = [&](int64_t q) -> uint32_t { return (q < 0 || q > static_cast<int64_t>(Qc) + 1) ? 0u : rank_r_[static_cast<size_t>(q)]; };
    auto rank_iota = [&](int64_t q) -> uint32_t { return (q < 0 || q > static_cast<int64_t>(Qc) + 1) ? 0u : rank_iota_[static_cast<size_t>(q)]; };
    auto built_pi = [&](int64_t q) { return q >= 0 && q <= static_cast<int64_t>(Qc) + 1 && cert_.pi_built[static_cast<size_t>(q)] != 0; };
    auto built_r = [&](int64_t q) { return q >= 0 && q <= static_cast<int64_t>(Qc) + 1 && cert_.r_built[static_cast<size_t>(q)] != 0; };
    auto built_iota = [&](int64_t q) { return q < 0 || (q <= static_cast<int64_t>(Qc) + 1 && cert_.iota_built[static_cast<size_t>(q)] != 0); };
    const ModCtx<uint32_t> ctx(ell_);
    auto composite_zero = [&](const FlatMatrix& out, const FlatMatrix& in) {
        if (out.get_cols() != in.get_rows()) return false;
        if (out.get_rows() == 0 || in.get_cols() == 0 || in.get_rows() == 0) return true;
        const Buf od = dense_of_flat(out), id = dense_of_flat(in);
        Buf prod(out.get_rows(), in.get_cols(), Layout::RowMajor);
        engine::kernels::gemm<uint32_t>(ctx, cview(od), cview(id), view(prod));
        return is_zero_view(cview(prod));
    };
    // A spot is enumerated at every STORED middle term (q <= that complex's
    // q_max); a reported zero above a finite complex's top degree is a
    // vacuous spot (dimension 0) and carries no bookkeeping.
    for (uint32_t q = 0; q <= Qc + 1; ++q) {
        const int64_t Q = q;
        // Hc^q: in = iota^{q-1}, out = pi^q
        if (q <= Qc) {
            ConeLesSpot s;
            s.term = "Hc"; s.q = static_cast<int32_t>(q);
            s.in_built = built_iota(Q - 1); s.out_built = built_pi(Q);
            s.asserted = boundary_fields_.reported(Q - 1) && cone_fields_.reported(Q) && bulk_fields_.reported(Q);
            s.dim_middle = dim_c(Q); s.rank_in = rank_iota(Q - 1); s.rank_out = rank_pi(Q);
            s.composite_zero = s.in_built && s.out_built && (Q - 1 < 0 || composite_zero(cert_.pi[q], cert_.iota[q - 1]));
            s.exact = s.asserted && (s.rank_in + s.rank_out == s.dim_middle);
            s.coker_in_dim = s.dim_middle - std::min(s.dim_middle, s.rank_in);
            cert_.spots.push_back(s);
        }
        // H_bulk^q: in = pi^q, out = r^q
        if (q <= bulk_fields_.q_max) {
            ConeLesSpot s;
            s.term = "H_bulk"; s.q = static_cast<int32_t>(q);
            s.in_built = built_pi(Q); s.out_built = built_r(Q);
            s.asserted = cone_fields_.reported(Q) && bulk_fields_.reported(Q) && boundary_fields_.reported(Q);
            s.dim_middle = dim_K(Q); s.rank_in = rank_pi(Q); s.rank_out = rank_r(Q);
            s.composite_zero = s.in_built && s.out_built && composite_zero(cert_.r[q], cert_.pi[q]);
            s.exact = s.asserted && (s.rank_in + s.rank_out == s.dim_middle);
            s.coker_in_dim = s.dim_middle - std::min(s.dim_middle, s.rank_in);
            cert_.spots.push_back(s);
        }
        // H_bd^q: in = r^q, out = iota^q
        if (q <= boundary_fields_.q_max) {
            ConeLesSpot s;
            s.term = "H_bd"; s.q = static_cast<int32_t>(q);
            s.in_built = built_r(Q); s.out_built = built_iota(Q);
            s.asserted = bulk_fields_.reported(Q) && boundary_fields_.reported(Q) && cone_fields_.reported(Q + 1);
            s.dim_middle = dim_bd(Q); s.rank_in = rank_r(Q); s.rank_out = rank_iota(Q);
            s.composite_zero = s.in_built && s.out_built && composite_zero(cert_.iota[q], cert_.r[q]);
            s.exact = s.asserted && (s.rank_in + s.rank_out == s.dim_middle);
            s.coker_in_dim = s.dim_middle - std::min(s.dim_middle, s.rank_in);
            cert_.spots.push_back(s);
        }
    }
}

// --- accessors ---------------------------------------------------------------

const BM& EquivariantMappingCone::D_block(uint32_t q) const { require_field("D_block"); require_computed("D_block"); require_cochains("D_block"); check_cone_degree(q, q_max(), "D_block"); return cert_.D[q]; }
const BM& EquivariantMappingCone::Ceq_block(uint32_t q) const { require_field("Ceq_block"); require_computed("Ceq_block"); require_cochains("Ceq_block"); check_cone_degree(q, q_max() + 1, "Ceq_block"); return cert_.Ceq[q]; }
const BM& EquivariantMappingCone::Z_block(uint32_t q) const { require_field("Z_block"); require_computed("Z_block"); require_cochains("Z_block"); check_cone_degree(q, q_max(), "Z_block"); return cert_.Z[q]; }
const BM& EquivariantMappingCone::B_block(uint32_t q) const { require_field("B_block"); require_computed("B_block"); require_cochains("B_block"); check_cone_degree(q, q_max(), "B_block"); return cert_.B[q]; }
const BM& EquivariantMappingCone::H_block(uint32_t q) const { require_field("H_block"); require_computed("H_block"); require_cochains("H_block"); check_cone_degree(q, q_max(), "H_block"); return cert_.H[q]; }
const BM& EquivariantMappingCone::P_block(uint32_t q) const { require_field("P_block"); require_computed("P_block"); require_cochains("P_block"); check_cone_degree(q, q_max(), "P_block"); return cert_.P[q]; }
const BM& EquivariantMappingCone::r_flat_block(uint32_t q) const {
    require_field("r_flat_block");
    require_computed("r_flat_block");
    require_cochains("r_flat_block");
    check_cone_degree(q, std::min(top_K_, top_bd_), "r_flat_block");
    return cert_.r_flat[q];
}
uint32_t EquivariantMappingCone::dim_Hc(uint32_t q) const { require_field("dim_Hc"); require_computed("dim_Hc"); check_cone_degree(q, q_max(), "dim_Hc"); return cert_.dim_Hc[q]; }

bool EquivariantMappingCone::pi_built(uint32_t q) const { require_field("pi_built"); require_computed("pi_built"); return q < cert_.pi_built.size() && cert_.pi_built[q] != 0; }
bool EquivariantMappingCone::r_built(uint32_t q) const { require_field("r_built"); require_computed("r_built"); return q < cert_.r_built.size() && cert_.r_built[q] != 0; }
bool EquivariantMappingCone::iota_built(uint32_t q) const { require_field("iota_built"); require_computed("iota_built"); return q < cert_.iota_built.size() && cert_.iota_built[q] != 0; }
const FlatMatrix& EquivariantMappingCone::pi(uint32_t q) const {
    if (!pi_built(q)) throw std::out_of_range("EquivariantMappingCone::pi: not built in degree " + std::to_string(q));
    return cert_.pi[q];
}
const FlatMatrix& EquivariantMappingCone::r(uint32_t q) const {
    if (!r_built(q)) throw std::out_of_range("EquivariantMappingCone::r: not built in degree " + std::to_string(q));
    return cert_.r[q];
}
const FlatMatrix& EquivariantMappingCone::iota(uint32_t q) const {
    if (!iota_built(q)) throw std::out_of_range("EquivariantMappingCone::iota: not built in degree " + std::to_string(q));
    return cert_.iota[q];
}
uint32_t EquivariantMappingCone::rank_pi(uint32_t q) const { (void)pi(q); return rank_pi_[q]; }
uint32_t EquivariantMappingCone::rank_r(uint32_t q) const { (void)r(q); return rank_r_[q]; }
uint32_t EquivariantMappingCone::rank_iota(uint32_t q) const { (void)iota(q); return rank_iota_[q]; }

const ConeCertificate& EquivariantMappingCone::certificate() const { require_field("certificate"); require_computed("certificate"); require_cochains("certificate"); return cert_; }

// (slot-24 G6) See the header.  Memory only: every value a released cone still
// serves (dims, maps, ranks, spots, the degree fields) is untouched.
void EquivariantMappingCone::release_cochain_objects() {
    require_field("release_cochain_objects");
    require_computed("release_cochain_objects");
    if (cochains_released_) return;
    for (auto* v : {&cert_.D, &cert_.Ceq, &cert_.Z, &cert_.B, &cert_.H, &cert_.P, &cert_.H_bulk, &cert_.P_bulk,
                    &cert_.H_bd, &cert_.P_bd, &cert_.r_flat})
        std::vector<BM>().swap(*v);
    std::vector<std::vector<uint32_t>>().swap(cert_.B_pivots);
    pool_.reset();
    cochains_released_ = true;
}

void EquivariantMappingCone::require_cochains(const char* who) const {
    if (!cochains_released_) return;
    throw std::logic_error(std::string("EquivariantMappingCone::") + who +
                           ": the cochain-level objects were released (release_cochain_objects(), slot-24 G6); "
                           "only the dimensions, the maps, their ranks and the LES spots remain");
}

// --- golden collectors (never emitted here) -------------------------------------

std::vector<ConeGoldenObject> EquivariantMappingCone::golden_objects() const {
    require_field("golden_objects");
    require_computed("golden_objects");
    require_cochains("golden_objects");
    std::vector<ConeGoldenObject> out;
    auto scalar = [&](const std::string& name, uint64_t v) { out.push_back(ConeGoldenObject{ name, 'A', v, 0, hash_scalar(v) }); };
    auto mv = [&](const std::string& name, const BM& M) {
        const Buf d = M.to_dense(Layout::RowMajor);
        out.push_back(ConeGoldenObject{ name, 'B', M.rows(), M.cols(), hash_view(cview(d)) });
    };
    auto flat = [&](const std::string& name, const FlatMatrix& F) {
        out.push_back(ConeGoldenObject{ name, 'B', F.get_rows(), F.get_cols(), hash_flat(F) });
    };
    for (uint32_t q = 0; q <= q_max(); ++q) scalar("dim_Hc^" + std::to_string(q), cert_.dim_Hc[q]);
    for (uint32_t q = 0; q <= q_max() + 1; ++q) {
        if (cert_.pi_built[q]) scalar("rank_pi^" + std::to_string(q), rank_pi_[q]);
        if (cert_.r_built[q]) scalar("rank_r^" + std::to_string(q), rank_r_[q]);
        if (cert_.iota_built[q]) scalar("rank_iota^" + std::to_string(q), rank_iota_[q]);
    }
    for (uint32_t q = 0; q <= q_max(); ++q) {
        mv("Hc^" + std::to_string(q), cert_.H[q]);
        mv("Pc^" + std::to_string(q), cert_.P[q]);
    }
    for (uint32_t q = 0; q <= q_max() + 1; ++q) {
        if (cert_.pi_built[q]) flat("pi^" + std::to_string(q), cert_.pi[q]);
        if (cert_.r_built[q]) flat("r^" + std::to_string(q), cert_.r[q]);
        if (cert_.iota_built[q]) flat("iota^" + std::to_string(q), cert_.iota[q]);
    }
    return out;
}

uint64_t EquivariantMappingCone::golden_digest() const {
    uint64_t h = engine::kFnvOffset64;
    h = engine::FnvU64(h, ell_);
    h = engine::FnvU64(h, d_V_);
    if (d_bd_ != d_V_) h = engine::FnvU64(h, d_bd_);   // (LES-2) only for an unequal pair: every boundary-pair digest unchanged
    for (const ConeGoldenObject& g : golden_objects()) {
        h = engine::FnvBytes(h, g.object);
        h = engine::FnvU64(h, static_cast<uint64_t>(g.tier));
        h = engine::FnvU64(h, g.rows);
        h = engine::FnvU64(h, g.cols);
        h = engine::FnvU64(h, g.hash);
    }
    return h;
}

// --- text -------------------------------------------------------------------------

std::string EquivariantMappingCone::to_string_offset(uint32_t offset) const {
    const std::string ind(offset, ' ');
    std::ostringstream oss;
    oss << ind << options_.name << " (ell = " << ell_ << ", d_V = " << d_V_;
    if (d_bd_ != d_V_) oss << ", d_V_bd = " << d_bd_;   // (LES-2) only for an unequal pair: every boundary-pair text unchanged
    oss << ")";
    if (!computed_) { oss << " [not computed]\n"; return oss.str(); }
    if (ring_k_ >= 2) {   // (slot-9 WP-4) the Z/ell^k summary
        const ModuleConeT<uint32_t>& m = *module_;
        oss << " over " << hecke::CoefficientRingText(ring_k_) << " (dense module route)\n" << ind << "  lengths H^q_c:";
        for (uint32_t v : m.H_lengths()) oss << " " << v;
        oss << "\n" << ind << "  eq:coneles image lengths:";
        for (uint32_t v : m.les_im_lengths()) oss << " " << v;
        oss << "\n" << ind << "  coneexact_iv_certified = " << (m.coneexact_iv_certified() ? "true" : "false") << "\n";
        return oss.str();
    }
    oss << "\n" << ind << "  degrees: bulk q_max=" << bulk_fields_.q_max << (bulk_fields_.truncated ? " (truncated)" : "")
        << ", boundary q_max=" << boundary_fields_.q_max << (boundary_fields_.truncated ? " (truncated)" : "")
        << ", cone q_max=" << cone_fields_.q_max << (cone_fields_.truncated ? " (truncated, derived)" : " (derived)") << "\n";
    oss << ind << "  dim H^q_c: " << join_u32(cert_.dim_Hc) << "   dim H^q_eq: " << join_u32(cert_.dim_H_bulk)
        << "   dim H^q_bd: " << join_u32(cert_.dim_H_bd) << "\n";
    for (const ConeLesSpot& s : cert_.spots)
        oss << ind << "  spot " << s.term << "^" << s.q << ": dim " << s.dim_middle << ", rank in " << s.rank_in << ", rank out " << s.rank_out
            << (s.asserted ? (s.exact ? " exact" : " NOT EXACT") : " (not asserted)")
            << (s.in_built && s.out_built ? (s.composite_zero ? ", composite 0" : ", COMPOSITE != 0") : "") << "\n";
    return oss.str();
}

std::string EquivariantMappingCone::to_json() const {
    require_field("to_json");
    require_computed("to_json");
    require_cochains("to_json");
    std::ostringstream oss;
    oss << "{\"schema\":\"equivariant_mapping_cone/v1\",\"name\":\"" << options_.name << "\",\"ell\":" << ell_ << ",\"d_V\":" << d_V_;
    if (d_bd_ != d_V_) oss << ",\"d_V_bd\":" << d_bd_;   // (LES-2) only for an unequal pair
    oss << ",\"ring\":\"F_ell\",\"ring_is_field\":true,\"six_invertible\":" << (cert_.six_invertible ? "true" : "false");
    oss << ",\"bulk\":" << bulk_fields_.to_json() << ",\"boundary\":" << boundary_fields_.to_json();
    oss << ",\"cone\":{\"q_max\":" << cone_fields_.q_max << ",\"truncated\":" << (cone_fields_.truncated ? "true" : "false")
        << ",\"q_max_source\":\"derived\",\"derived_from\":[\"bulk\",\"boundary\"],\"degrees\":[" << join_u32(cone_fields_.degrees) << "]}";
    if (!options_.boundary_model.empty()) oss << ",\"boundary_model\":\"" << options_.boundary_model << "\"";
    if (options_.boundary_is_group_cohomology_known)
        oss << ",\"boundary_is_group_cohomology\":" << (options_.boundary_is_group_cohomology ? "true" : "false");
    oss << ",\"dim_Hc\":[" << join_u32(cert_.dim_Hc) << "],\"dim_H_bulk\":[" << join_u32(cert_.dim_H_bulk) << "],\"dim_H_bd\":[" << join_u32(cert_.dim_H_bd) << "]";
    oss << ",\"flat_dim\":[" << join_u32(cert_.flat_dim) << "]";
    oss << ",\"maps\":[";
    bool first = true;
    for (uint32_t q = 0; q <= q_max() + 1; ++q) {
        auto emit = [&](const char* name, const std::vector<char>& built, const std::vector<uint32_t>& ranks, const std::vector<FlatMatrix>& M) {
            if (!built[q]) return;
            oss << (first ? "" : ",") << "{\"map\":\"" << name << "\",\"q\":" << q << ",\"rows\":" << M[q].get_rows() << ",\"cols\":" << M[q].get_cols()
                << ",\"rank\":" << ranks[q] << "}";
            first = false;
        };
        emit("pi", cert_.pi_built, rank_pi_, cert_.pi);
        emit("r", cert_.r_built, rank_r_, cert_.r);
        emit("iota", cert_.iota_built, rank_iota_, cert_.iota);
    }
    oss << "],\"les\":[";
    first = true;
    for (const ConeLesSpot& s : cert_.spots) {
        oss << (first ? "" : ",") << "{\"term\":\"" << s.term << "\",\"q\":" << s.q << ",\"dim\":" << s.dim_middle << ",\"rank_in\":" << s.rank_in
            << ",\"rank_out\":" << s.rank_out << ",\"status\":\"" << (s.asserted ? "asserted_exact" : (s.in_built && s.out_built ? "complex_only" : "unreported_neighbour"))
            << "\"";
        if (s.in_built && s.out_built) oss << ",\"composite_zero\":" << (s.composite_zero ? "true" : "false");
        if (s.asserted) oss << ",\"exact\":" << (s.exact ? "true" : "false");
        else oss << ",\"coker_in_dim_recorded\":" << s.coker_in_dim;
        oss << "}";
        first = false;
    }
    oss << "]";
    const bool H2_reported = bulk_fields_.reported(2);
    const uint32_t dimH2 = (H2_reported && 2 <= bulk_fields_.q_max) ? cert_.dim_H_bulk[2] : 0u;
    oss << ",\"H2_eq\":{\"reported\":" << (H2_reported ? "true" : "false") << ",\"dim\":" << dimH2
        << ",\"zero_asserted\":" << ((cert_.six_invertible && H2_reported) ? "true" : "false") << "}";
    oss << ",\"H0_c_zero_asserted\":true,\"conedims_asserted\":" << ((H2_reported && dimH2 == 0) ? "true" : "false");
    oss << "}";
    return oss.str();
}
