// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// =====================================================================
// Kernels.h -- dense arithmetic kernels over F_p (Phase 2, plan D2/D3)
// =====================================================================
//
// Every kernel takes a ModCtx by reference and DenseViews. All arithmetic is
// exact integer arithmetic: reassociation, vectorisation and k-blocking
// cannot change a result, so the AVX-512 path and the portable path are
// bit-identical BY CONSTRUCTION and asserted so by KernelBitIdentityValidation.
//
// INVARIANT CONTRACT (input -> output):
//   gemm / gemm_nt / gemv / trmm / apply_*  : inputs Canonical, output Canonical
//   axpy / scal                             : Canonical -> Canonical
//   canonicalise (in place)                 : Lazy2p -> Canonical
//   canonicalise (Accum64 -> Elt)           : Accum64 -> Canonical
//   transpose / copy                        : any invariant, preserved
// A Lazy2p or Accum64 input to an arithmetic kernel is an ENGINE_CHECK
// failure (the k_max bound assumes operands < p).
//
// k-BLOCKING CONTRACT. gemm accumulates products of canonical residues in
// the exact accumulator (uint64_t for uint32_t elements, unsigned __int128
// for uint64_t elements) over k-blocks of at most ctx.k_max terms, reduces
// each block to a canonical residue and combines blocks with mod_add. The
// block length is ENGINE_CHECKed against k_max at every block start;
// dot_delayed() exposes a single block for callers that accumulate
// themselves and ENGINE_CHECKs k <= k_max (this is the overflow guard that
// ModContextValidation proves fires).
//
// FAST PATH. gemm<uint32_t> is an 8x16 outer-product register-blocked
// microkernel (the "k-j" order of 03 Sec. 7.2 with the register block of
// 04 App. A): A(i, k) is broadcast, so ANY A layout is free; B(k, j..j+16) is
// streamed, so B's ROWS must be contiguous (row-major B, or a transposed view
// of a column-major buffer). A B whose rows are not contiguous is packed once
// into a thread-local scratch (O(kn) exact copy) and then runs the same
// microkernel, so all four layout combinations, transposed views included,
// are correct and within one copy of the fast path. A row-major C takes the
// vectorised epilogue; a column-major C reduces cell by cell (correct, slower).
// Both microkernel bodies (AVX-512, selected at runtime; portable C++) give
// identical bits. See ENGINE_API.md Sec. 6 for measured numbers.

#include <cstddef>
#include <cstdint>

#include "CpuFeatures.h"
#include "DenseView.h"
#include "ModCtx.h"

namespace engine {
namespace kernels {

enum class GemmOp : uint8_t { Set, Add, Sub };   // C = A*B, C += A*B, C -= A*B
enum class Side   : uint8_t { Left, Right };     // C = M*B  or  C = B*M

// Runtime microkernel selection (all produce identical bits).
//   Auto     : AVX-512 when the CPU has it and the build can emit it, else AVX2
//              (slot-24 AVX2K) when the CPU has AVX2, else portable -- unless the
//              env ENGINE_KERNEL names a path (below), which Auto then follows
//   Portable : scalar / auto-vectorised C++ only
//   Avx512   : force the AVX-512 microkernel (ENGINE_CHECK if unavailable)
//   Avx2     : force the AVX2 microkernel (ENGINE_CHECK if unavailable); like
//              Portable it never takes the AVX-512 VNNI routes (the Zen 3 /
//              Coffee Lake arm)
// Env ENGINE_KERNEL (read once; read-only, cannot change a value): "portable",
// "avx2", "avx512" or "auto" (unset = auto). A set value prints one line on
// stderr, `EFFECTIVE ENGINE_KERNEL path=<p> value=<v> cpu=avx2:<0|1>,avx512:<0|1>`;
// a path the CPU lacks, or an unknown value, prints a WARNING and keeps auto.
// (S24 CL-5, C-06) KernelPath IS engine::cpu::IsaPath (CpuFeatures.h; ring2k::KernelPath and binary::RowPath are the same
// type), so the paths convert by name and by value alike.
using KernelPath = cpu::IsaPath;
void       set_kernel_path(KernelPath path);
KernelPath kernel_path();                 // the setting
bool       avx512_kernel_available();     // CPU supports it AND this build compiled it
bool       avx512_kernel_active();        // what gemm<uint32_t> will actually run
bool       avx2_kernel_available();       // (AVX2K) CPU supports AVX2 AND this build compiled it
KernelPath active_kernel_path();          // (AVX2K) the resolved path: Portable, Avx2 or Avx512
const char* kernel_path_name(KernelPath path);   // "auto" / "portable" / "avx2" / "avx512"

// (S24 CL-5, C-06) THE ONE ISA POLICY.  ENGINE_KERNEL (read once, above) governs EVERY module with ISA bodies, not only this
// file's GEMM: isa_ceiling() is its resolved value (Auto when unset, unknown or unavailable).  A module's Auto takes the best
// body it has AT OR BELOW the ceiling (isa_auto): portable = the portable bodies everywhere, avx2 = no AVX-512 body anywhere,
// avx512 / auto = the best the CPU has.  Governed: this GEMM (which follows ENGINE_KERNEL exactly, as before), ring2k,
// binary::BitMatrix / TritMatrix, the rho fill (FlatMatrix RhoFillPath), the descent Smith kernels (smithk) and ring3k's
// small-modulus rows (portable only: their bodies are target_clones).  A module's explicit setter or its own variable
// (set_kernel_path, set_row_path, ScopedRhoFillPath, HECKE_RHO_SIMD, HECKE_DESCENT_SMITH_SIMD) still wins (validation
// overrides).  Every body is bit-identical by construction, so the ceiling cannot change a value.
cpu::IsaPath isa_ceiling();
inline cpu::IsaPath isa_auto(bool has_avx2, bool has_avx512) {
    const cpu::IsaPath c = isa_ceiling();
    if (c == cpu::IsaPath::Portable) return cpu::IsaPath::Portable;
    if (has_avx512 && c != cpu::IsaPath::Avx2) return cpu::IsaPath::Avx512;
    return has_avx2 ? cpu::IsaPath::Avx2 : cpu::IsaPath::Portable;
}

// (slot-23 PERF-P5, ENGINE_API.md Sec. 6.1) the small-modulus VNNI route of
// gemm<uint32_t>: for p <= VNNI_MAX_MODULUS every canonical residue fits 7
// bits, so A and B are packed to bytes and multiplied with AVX-512 VNNI
// (vpdpbusd, u8 x s8 -> exact i32 sums over k-blocks of VNNI_KC), each block
// reduced to the canonical residue and combined with mod_add / mod_sub. The
// residue of an exact integer is unique, so the route is bit-identical to the
// u32 microkernels BY CONSTRUCTION (asserted by VnniGemmBitIdentityValidation;
// ENGINE_VNNI_SHADOW=1 recomputes every routed call on the u32 path and throws
// on the first differing cell). It cannot change a value.
//
// (slot-24 G4, ENGINE_API.md Sec. 6.2) the two-limb VNNI route for
// VNNI_MAX_MODULUS < p <= VNNI2_MAX_MODULUS = 2^14: every canonical residue
// fits 14 bits. A is split into 7-bit limbs a = a0 + 128 a1, B is paired with
// b~ = 128 b mod p, so a b == a0 b + a1 b~ (mod p); vpdpwssd (s16 x s16 pairs
// -> i32) adds a0 b + a1 b~ exactly, <= 2 * 127 * 16383 per k-term, so a
// k-block of VNNI2_KC = 512 terms stays below 2^31. Each block is reduced
// and combined exactly as on the one-limb route: the canonical residue of a
// CONGRUENT integer is the same residue, so it is bit-identical to the u32
// microkernels by construction as well.
//
//   Auto  : route when the CPU has avx512_vnni, kernel_path() != Portable,
//           p <= VNNI2_MAX_MODULUS and the shape passes the size threshold
//           (m >= 8, n >= 32, k >= 16; for p >= 2^13 also m, n >= 128)
//           (default; ENGINE_VNNI=0 or off at process start selects Off for
//           BOTH limb counts -- retired by S26 LEVERS-26: refused in an
//           OC_ENABLE_LEGACY=OFF build, read only in a legacy build; Off stays
//           reachable through set_vnni_path(Off) in every build; a set
//           ENGINE_VNNI prints one EFFECTIVE line on stderr, and a value other
//           than 0 / off / 1 also a WARNING)
//   Off   : never route (the pre-P5 u32 path; the cross_config arm)
//   Force : route every p <= VNNI2_MAX_MODULUS call whatever its shape
//           (validation use; ENGINE_CHECK if VNNI is unavailable)
enum class VnniPath : uint8_t { Auto, Off, Force };
constexpr uint32_t VNNI_MAX_MODULUS  = 128;
constexpr uint32_t VNNI_KC           = 1024;   // k-block: 1024 * 127^2 < 2^24 << 2^31
constexpr uint32_t VNNI2_MAX_MODULUS = 16384;  // 2^14: entries <= 16383 fit s16 and two 7-bit limbs
constexpr uint32_t VNNI2_KC          = 512;    // k-block: 512 * 2 * 127 * 16383 = 2130576384 < 2^31 - 1
// 0 = not routable by modulus, 1 = the one-limb (vpdpbusd) route, 2 = the two-limb (vpdpwssd) route.
constexpr uint32_t vnni_limbs_for(uint32_t p) {
    return p <= VNNI_MAX_MODULUS ? 1u : p <= VNNI2_MAX_MODULUS ? 2u : 0u;
}
void     set_vnni_path(VnniPath path);
VnniPath vnni_path();                     // the setting (ENGINE_VNNI read once at first use)
void     vnni_env_check();                // (S26 LEVERS-26) read ENGINE_VNNI now (OFF builds: `0` / `off` refused, exit 1); main() start-up
bool     vnni_kernel_available();         // CPU supports avx512_vnni + the AVX-512 set AND the build emits it
bool     vnni_route_taken(uint32_t p, uint32_t m, uint32_t n, uint32_t k);   // would gemm<uint32_t> route this call
uint64_t vnni_routed_calls();             // process-wide count of routed gemm calls, both limb counts (diagnostics)
uint64_t vnni2_routed_calls();            // of which the two-limb route (128 < p <= 2^14)
// (slot-25 PERF-VF (a)) The two-limb route's work-per-thread floor: under Auto a 128 < p <= 2^14 call routes only if
// also m*n*k >= vnni2_work_floor() * vnni_route_threads().  At 16 threads on small cells the packing and the per-thread
// scratch cost more than the route saves (slot24/CLOSE_H2H_diagnosis.md Sec. 3); a 1-thread shard keeps the route
// for its large GEMMs.  (S26 LEVERS-26) Calibrated at ext n = 160, ell = 691 under gate.lock (slot26/LEVERS26_report.md Sec. 3,
// 7 rounds, floors rotated; re-take of slot25/PERFVF_report.md Sec. 3): t16 median 2^20 0.2426 s vs 2^18 0.2544 s (-4.6 %),
// min 0.2411 vs 0.2501 (-3.6 %); t1 median 0.8531 vs 0.8632 (-1.2 %); floor 0 (every shape) 0.2741 / 0.8625, off 0.2472 / 1.1704.
// So the default is 2^20 (was 2^18 from slot-25 PERF-VF (a)); at 1 thread it drops only shapes below 128 x 128 x 64.
// Another value means changing it here and the (R) boundary block of VnniGemmBitIdentityValidation.
// A speed choice only: the routes are bit-identical.  ENGINE_VNNI2_FLOOR=<digits> overrides the floor (0 = the slot-24
// rule; anything else warns and keeps the default), read once; a set value prints one EFFECTIVE line on stderr when the
// floor is first consulted (under Auto, at the first two-limb call that passes the large-p test; PERF-VF audit N4).
// floor * threads saturates at UINT64_MAX (PERF-VF audit S1).
constexpr uint64_t VNNI2_WORK_FLOOR_DEFAULT = uint64_t(1) << 20;   // (S26 LEVERS-26) was 1 << 18
uint64_t vnni2_work_floor();              // the floor in effect (default or ENGINE_VNNI2_FLOOR)
uint32_t vnni_route_threads();            // the global pool's configured size (ENGINE_THREADS rule), resolved once
bool     vnni2_work_passes(uint32_t m, uint32_t n, uint32_t k, uint32_t threads);   // m*n*k >= floor * threads

// (slot-24 AVX2K, ENGINE_API.md Sec. 6.3) the AVX2 pair route of gemm<uint32_t>: for 2 <= p <= AVX2_PAIR_MAX_MODULUS
// every canonical residue is a non-negative i16, so A and B are packed into k-pair dwords and multiplied with
// vpmaddwd (16 exact multiply-adds per instruction) into exact i32 lane sums over k-blocks with
// (kb/2) 2 (p-1)^2 < 2^31, each block reduced to the canonical residue: bit-identical to the u32 microkernels.
//   Auto  : route when the resolved kernel path is Avx2 (CPUs without AVX-512, or ENGINE_KERNEL=avx2) and
//           m >= 4, n >= 16, k >= 16 (a speed threshold only);
//   Off   : never (the AVX2 path runs its u64 microkernel for every p); env ENGINE_AVX2_PAIR=0 selects it;
//   Force : every call with p <= AVX2_PAIR_MAX_MODULUS (validation use; ENGINE_CHECK without AVX2).
// Neither setting can change a value.
constexpr uint32_t AVX2_PAIR_MAX_MODULUS = 4096;
enum class Avx2PairPath : uint8_t { Auto, Off, Force };
void         set_avx2_pair_path(Avx2PairPath path);
Avx2PairPath avx2_pair_path();
bool         avx2_pair_route_taken(uint32_t p, uint32_t m, uint32_t n, uint32_t k);
uint64_t     avx2_pair_routed_calls();

// C op= A * B   with A: m x k, B: k x n, C: m x n (logical shapes; any layouts).
template<class Elt>
void gemm(const ModCtx<Elt>& ctx, DenseView<const Elt> A, DenseView<const Elt> B,
          DenseView<Elt> C, GemmOp op = GemmOp::Set);
template<> void gemm<uint32_t>(const ModCtx<uint32_t>&, DenseView<const uint32_t>, DenseView<const uint32_t>,
                               DenseView<uint32_t>, GemmOp);
template<> void gemm<uint64_t>(const ModCtx<uint64_t>&, DenseView<const uint64_t>, DenseView<const uint64_t>,
                               DenseView<uint64_t>, GemmOp);

// C op= A * B^T  with B given as n x k (multi-vector form: B's rows are the vectors).
template<class Elt>
void gemm_nt(const ModCtx<Elt>& ctx, DenseView<const Elt> A, DenseView<const Elt> B,
             DenseView<Elt> C, GemmOp op = GemmOp::Set);

// y op= A * x   with x a k x 1 view and y an m x 1 view (a row view is fine:
// pass x.transposed_view() when x is stored as a row).
template<class Elt>
void gemv(const ModCtx<Elt>& ctx, DenseView<const Elt> A, DenseView<const Elt> x,
          DenseView<Elt> y, GemmOp op = GemmOp::Set);

// A single delayed dot product over ONE k-block: a, b contiguous, k <= ctx.k_max
// (ENGINE_CHECK). Returns the canonical residue of sum_i a[i]*b[i].
template<class Elt>
Elt dot_delayed(const ModCtx<Elt>& ctx, const Elt* a, const Elt* b, size_t k);

// (hecke-extensions slot 11, WP-18a; ENGINE_API.md Sec. 13.12.3) the u128
// element type (composite-only, ModCtxU128.h): a PORTABLE SCALAR path, exact
// u256 accumulation k-blocked at min(k_max, 256) (the Sec. 5 rule with the
// u128 k_max), one Montgomery-128 reduce per k-block.  No AVX-512 path.
// Correct, not fast.  Every other kernel below is the generic template,
// instantiated for u128 as well.
template<> void gemm<u128>(const ModCtx<u128>&, DenseView<const u128>, DenseView<const u128>,
                           DenseView<u128>, GemmOp);
template<> void gemv<u128>(const ModCtx<u128>&, DenseView<const u128>, DenseView<const u128>,
                           DenseView<u128>, GemmOp);
template<> u128 dot_delayed<u128>(const ModCtx<u128>&, const u128*, const u128*, size_t);

// y = y +/- alpha * x  (entrywise over equal-shape views; Shoup form of alpha
// when the regime allows, Barrett / wide otherwise).
template<class Elt>
void axpy(const ModCtx<Elt>& ctx, Elt alpha, DenseView<const Elt> x, DenseView<Elt> y,
          bool subtract = false);

// x = alpha * x
template<class Elt>
void scal(const ModCtx<Elt>& ctx, Elt alpha, DenseView<Elt> x);

// C = T * B for T square with structure UpperUnitri or LowerUnitri (only the
// strict triangle of T is read; the diagonal is taken as 1). C must not alias B.
template<class Elt>
void trmm(const ModCtx<Elt>& ctx, DenseView<const Elt> T, DenseView<const Elt> B,
          DenseView<Elt> C);

// C = M*B (Side::Left) or C = B*M (Side::Right) for M with structure
// SignedAntiDiag: M is d x d, only M(i, d-1-i) in {1, p-1} is read.
template<class Elt>
void apply_signed_antidiag(const ModCtx<Elt>& ctx, DenseView<const Elt> M,
                           DenseView<const Elt> B, DenseView<Elt> C, Side side);

// C = D*B or C = B*D for D with structure Diagonal (only D(i, i) is read).
template<class Elt>
void apply_diagonal(const ModCtx<Elt>& ctx, DenseView<const Elt> D,
                    DenseView<const Elt> B, DenseView<Elt> C, Side side);

// PHYSICAL relayout: dst(i, j) = src(i, j), 16 x 16 tiles, any layouts /
// transposition flags on either side. Invariant tag copied to dst's buffer
// when dst covers its whole buffer.
template<class Elt>
void copy(DenseView<const Elt> src, DenseView<Elt> dst);

// PHYSICAL transpose: dst(i, j) = src(j, i)  (== copy(src.transposed_view(), dst)).
template<class Elt>
void transpose(DenseView<const Elt> src, DenseView<Elt> dst);

// Lazy2p -> Canonical in place. Updates the buffer tag when v covers it.
template<class Elt>
void canonicalise(const ModCtx<Elt>& ctx, DenseView<Elt> v);

// Accum64 -> Canonical: dst(i, j) = acc(i, j) mod p (u32 modulus).
void canonicalise(const ModCtx<uint32_t>& ctx, DenseView<const uint64_t> acc,
                  DenseView<uint32_t> dst);

// Row / column exchange inside one view (elimination helper).
template<class Elt>
void swap_rows(DenseView<Elt> v, uint32_t r1, uint32_t r2);

} // namespace kernels
} // namespace engine
