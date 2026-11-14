# Engine arithmetic layer — API contract (matrix-engine Phase 2)

Files: `ModCtx.h`, `DenseBuf.h/.cpp`, `DenseView.h`, `Kernels.h/.cpp`, `RefLinAlg.h/.cpp`
(all in `src/ModArithmetic/`, namespace `engine`, kernels in `engine::kernels`,
oracle in `engine::ref`). Element types: `uint32_t` (p < 2^32) and `uint64_t` (p < 2^64).
Everything is exact integer arithmetic; a single differing bit anywhere is a bug.

## 1. `ModCtx<Elt>` — construct once per modulus, pass by `const&`

| Member | Meaning |
|---|---|
| `p`, `p2`/`has_p2` | modulus; p² when it fits `Elt` |
| `barrett_m` | ⌊2^64/p⌋ (u32 only) |
| `mont_ninv`, `mont_r1`, `mont_r2`, `has_montgomery` | −p⁻¹ mod 2^64, R mod p, R² mod p (u64, odd p < 2^63) |
| `k_max` | **⌊(2^64−1)/(p−1)²⌋** (u32; 1023 at ℓ=134217757, 8 at 1518500213, 4 at 2^31−1, 1 at 2^32−5); ⌊(2^128−1)/(p−1)²⌋ saturated to 2^64−1 (u64; 64 at 2^61−1, 4 at 2^63−25). `uint64_t`, uncapped. |
| `regime` | `U32_Delayed` (k_max ≥ 8) / `U32_Shoup` (p ≤ 2^31) / `U32_Barrett64` / `U64_Montgomery` / `U64_Wide` |

Primitives (canonical in, canonical out, `== legacy mod_*_w` bit for bit):
`mul add sub neg inv pow`, `reduce(Accum)` (`Accum` = `uint64_t` / `unsigned __int128`).
u32 only: `reduce_u64(x)` — **any** x < 2^64, not only x < p²; `mod_mul_barrett`;
`shoup_available()` (p ≤ 2^31), `shoup_precompute(w)`, `shoup_mul(a, w, w')` canonical,
`shoup_mul_lazy` → [0, 2p) (`a` may be any uint32). u64 only: `mont64_to/from/mul/reduce`
(operands in Montgomery form `aR`; `mont64_reduce(T)` needs T < p·R).
Both widths (slot-11 WP-12, F-1): the **reused-operand pair** `w' = mul_prep(w)`,
`mul_prepped(x, w') == mul(x, w)` bit for bit, canonical in and out. `U64_Montgomery`:
`w' = w·R mod p` and one `mont64_reduce` per product; every other regime (all of u32,
`U64_Wide`): `w' = w` and `mul_prepped` is `mul`. `w'` is an opaque token (never stored or
hashed). `mul` itself is unchanged: a two-REDC one-shot product measured slower than the
`(u128)a·b % p` it would replace (≈ 5.3 vs 4.3 ns, WP-12 report §3).
`canonicalise(v)`: [0, 2p) → [0, p). `batch_inv(x, out, n, ZeroPolicy)` (may alias):
`Throw` (default) throws `std::runtime_error` like `mod_inv(0)`; `MapToZero` writes 0 there.
`ModCtx(p < 2)` is an `ENGINE_CHECK` failure (`std::logic_error`).

## 2. `DenseBuf<Elt>` / `DenseView<Elt>`

`DenseBuf(rows, cols, layout = RowMajor, ld = 0 /*minimal*/, invariant = Canonical, structure = Generic)`
owns zeroed storage; `DenseBuf(Elt* ext, capacity, rows, cols, layout, ld, ...)` is non-owning
(arena-ready, not zeroed). `layout ∈ {RowMajor, ColMajor}`: (i,j) at `i*ld + j` resp. `j*ld + i`;
`ld ≥ minor_dim()` (cols resp. rows). Padding is never read, written or compared.
`invariant ∈ {Canonical [0,p), Lazy2p [0,2p) (u32), Accum64 (exact u64 sums, stored in a
DenseBuf<uint64_t>)}`. `structure` is a **promise about which entries kernels may read**
(§4); `Generic` otherwise. Copying a buffer copies data and tags; a copied external buffer
becomes owned.

`DenseView<Elt>` = `{buf*, row0, rows, col0, cols, transposed}`; `DenseView<const Elt>` is the
read-only handle, `DenseView<Elt>` converts to it. `view(buf)` / `cview(buf)`. Free operations:
`sub(r0,nr,c0,nc)`, `row(i)`, `col(j)`, `rows_range`, `cols_range`, `transposed_view()`
(**logical** transpose — flag flip, the memory pattern is unchanged). `rows()/cols()` are the
logical shape after transposition; `at(i,j)` resolves layout + flag. `strides()` →
`{base, rs, cs}` with element (i,j) at `base[i*rs + j*cs]` — kernels use this, never `ld`.
`row_contiguous()` (cs == 1), `col_contiguous()` (rs == 1), `covers_whole_buffer()`,
`invariant()`, `structure()`. `same_entries(a, b)` compares logical contents across layouts.

### 2.1 `MatrixT` and the sanctioned `ExtMatrix` (slot-24 CL-M)

`MatrixT<Elt>` (`Matrix.h`, aliases `Matrix` / `Matrix64`) is the pre-engine dense matrix. It is **not** an engine type and
carries no `[[deprecated]]` attribute (CODEAUDIT C-02; the `BlockFlatMatrix.h` precedent): which translation units may
include `Matrix.h` directly is enforced by `docs/matrix-engine/tools/check_matrix_includes.sh` (`MatrixRetirementValidation`),
and `cohomology_app` builds without `-Wno-deprecated-declarations`. **HeckeExt's `ExtMatrix`** (`HeckeExt/HeckeLocalAlgebra.h`,
`using ExtMatrix = MatrixT<uint32_t>`) is the one sanctioned `MatrixT` working type (CODEAUDIT A9): the small dense `F_ell`
algebra of [X] Sec. 2.3 (`D ≤ ~200`), where the block model would buy nothing. The guard prints its spread as a `census` line.
New code uses `FlatMatrix` / `engine::blk`.

## 3. Kernels — invariant contract

| Kernel | Inputs | Output |
|---|---|---|
| `gemm(ctx, A, B, C, op)` — `C op= A·B`, `op ∈ {Set, Add, Sub}` | A, B, C **Canonical** (C also for `Set`) | Canonical |
| `gemm_nt(ctx, A, B, C, op)` — `C op= A·Bᵀ`, B given n×k | same | Canonical |
| `gemv(ctx, A, x, y, op)` — x k×1, y m×1 column views (`x.transposed_view()` for a row) | same | Canonical |
| `dot_delayed(ctx, a*, b*, k)` — one k-block, contiguous pointers | canonical, `k ≤ k_max` (**ENGINE_CHECK**) | canonical Elt |
| `axpy(ctx, α, x, y, subtract)` — `y = y ± α·x`; `scal(ctx, α, x)` | Canonical, `α < p` | Canonical |
| `trmm(ctx, T, B, C)` — `C = T·B`, T tagged `UpperUnitri`/`LowerUnitri` | Canonical, C ∌ B, T | Canonical |
| `apply_signed_antidiag(ctx, M, B, C, side)`, `apply_diagonal(ctx, D, B, C, side)` | tagged M/D, Canonical | Canonical |
| `copy(src, dst)`, `transpose(src, dst)` — **physical**, 16×16 tiles, any layouts | any | dst gets src's tag if it covers its buffer |
| `canonicalise(ctx, v)` in place | `Lazy2p` (or Canonical) | Canonical; buffer retagged only if `v` covers it |
| `canonicalise(ctx32, acc64, dst32)` | `Accum64` | Canonical |
| `swap_rows(v, r1, r2)` | any | unchanged tag |

Rules enforced by `ENGINE_CHECK` (throws `std::logic_error`, every build): shape agreement;
operands Canonical (a `Lazy2p`/`Accum64` operand to an arithmetic kernel breaks the k_max bound);
`C` must not overlap A or B **as windows** — disjoint windows of the same buffer are allowed
(Schur update `C −= L·U` in place is fine); structure tag present where required; anti-diagonal
entries ∈ {1, p−1}. `ENGINE_DEBUG_CHECK` (non-NDEBUG builds) additionally scans operands for
entries ≥ p and cross-checks the vector reduction against the scalar Barrett.

## 4. Structure tags: what is read

`UpperUnitri`/`LowerUnitri`: square; only the **strict** triangle is read, the diagonal is
taken as 1, the other triangle may hold anything (packed LU is fine). `SignedAntiDiag`: d×d,
only `M(i, d−1−i)` is read, must be 1 or p−1. `Diagonal`: only `D(i,i)`. `Generic`: everything.

## 5. k-blocking contract (the overflow bound)

Products of canonical residues are accumulated **exactly** in `Accum` over k-blocks of length
`kb ≤ min(ctx.k_max, 256)`; each block is reduced once (`ctx.reduce`) and blocks are combined
with `mod_add`/`mod_sub`. `kb ≤ k_max` is `ENGINE_CHECK`ed at every block start, so overflow is
impossible by construction; `dot_delayed` exposes one block and is the guard you call when you
accumulate yourself (never exceed `k_max` terms per accumulator). Because every intermediate is
an exact integer, **reassociation, blocking, vectorisation and the CPU path cannot change a
result**: AVX-512 and portable paths are bit-identical and both are compared to the oracle in
`KernelBitIdentityValidation`.

## 6. Layout and speed (what the retrofit / elimination engineer should pass)

`gemm<uint32_t>` is an 8×16 outer-product microkernel: `A(i,k)` is broadcast (any A layout,
no penalty), `B(k, j..j+16)` is streamed — **B's rows must be contiguous** (`B.row_contiguous()`:
row-major B, or a transposed view of a column-major buffer). Otherwise B is packed once into a
thread-local scratch (O(kn), exact copy). `C` row-major takes the vectorised epilogue; a
column-major `C` is correct but reduces cell by cell. Blocking: k in ≤256, n in 512-wide panels.
Measured (p = 134217757, single thread, default `RelWithDebInfo` build, AVX-512 path):
25 / 28 / 30 / 30 Gmodmul/s at n = 64 / 128 / 256 / 512; portable path 2.7 G (-O2) and
9.3 G (`Bench`, -O3 -march=native). The AVX-512 body is compiled with a target attribute and
picked at runtime (`__builtin_cpu_supports`), so every build gets it on this CPU;
`set_kernel_path(KernelPath::Portable|Avx2|Avx512|Auto)` forces a path (validation use; the AVX2 body: §6.3).
`gemm<uint64_t>` is a plain strided loop with `__int128` accumulation — correct, not fast.

**Index and stride domains (slot-14 WP-3, G1/G2).** `FlatMatrix` forms every `rows × cols` / offset product in `size_t` and `ENGINE_CHECK`s that the product fits `size_type` (G1); a bound tape view's row and column strides must fit the `uint32` `ld` domain, a named throw in `Engine/Recorder.h:49, 53` (G2). Below `2^32` no index changes; above it the former silent wrap is a refusal (WP-3 only adds refusals). `FlatMatrixValidation` 119 (+7), `EngineCudaMemoryValidation` 31 (arm H).

For elimination: keep panels row-major, use `dot_delayed`/`gemm` for the trailing update with
`GemmOp::Sub`, `axpy` (Shoup-form scalar, valid for p ≤ 2^31; Barrett otherwise — the kernel
chooses) for row operations, `batch_inv` for pivot normalisation, `swap_rows` for pivoting,
`trmm` with a tagged unit-triangular factor. Never read `ld` — call `strides()`.

### 6.1 The small-modulus VNNI route (slot-23 PERF-P5)

For `p ≤ VNNI_MAX_MODULUS = 128` every canonical residue is ≤ 127 and fits one byte that reads the same as `u8` and as
`s8`. `gemm<uint32_t>` then packs A into 8-row panels of k-quads (one dword = 4 bytes of a row) and B into 32-column panels
(128 bytes per k-quad), and runs an 8×32 `vpdpbusd` microkernel (16 zmm i32 accumulators). Blocking: k in
`VNNI_KC = 1024` (the bound: `1024·127² < 2^24 ≪ 2^31`, `ENGINE_CHECK`ed per block as `kb·127² < 2^31` — the byte bound the entry gate enforces whatever `p` is, so a future `VNNI_KC` change cannot pass a `p`-bound while breaking the real one), A blocks of
128 rows, B blocks of 2048 columns; packing scratch is thread-local (≤ 2 MB B + 128 KB A per thread). Each k-block's lane
sums are reduced to canonical residues (FP64 quotient, corrected; `ENGINE_DEBUG_CHECK`ed against `%` in non-`NDEBUG`
builds) and combined into C with the legacy `mod_add` / `mod_sub` — the §5 rule with a different exact accumulator. The
residue of an exact integer is unique, so the route is **bit-identical to the u32 microkernels by construction**; every
layout, transposed view and window is accepted (packing reads strides). Any modulus `≤ 128` qualifies, composite too
(`Z/ℓ^k`, `Z/2^7`).

Selection: `set_vnni_path(VnniPath::Auto | Off | Force)`. `Auto` (the default) routes when the CPU has `avx512_vnni` (+ the
§6 AVX-512 set), `kernel_path() != Portable`, and `m ≥ 8, n ≥ 32, k ≥ 16` (a speed threshold only); `Off` never routes (the
pre-P5 path); `Force` routes every `p ≤ 128` call (validation use; `ENGINE_CHECK` without VNNI). Env (read once):
`ENGINE_VNNI=0` selects `Off`; `ENGINE_VNNI_SHADOW=1` recomputes every routed call on the u32 route into a copy of C and
throws (`std::logic_error`) on the first differing cell; at exit it prints `VNNI_SHADOW calls= cells= differ=` on stderr (`differ` counts the calls that threw). Neither can change a value. A routed operand entry `≥ 128` behind
a `Canonical` tag is an `ENGINE_CHECK` refusal (the u32 route would have reduced it; the contract forbids it either way).
CPUs without VNNI (e.g. a desktop Intel Core i9-9900KF) run the §6 u32 kernel under `Auto` — the fallback arm. `vnni_routed_calls()` counts
routed calls (diagnostics). Gate: `VnniGemmBitIdentityValidation` (`standard`; each arm against `RefLinAlg`, the worst case
of all entries `p − 1` over 3 k-blocks, `n > 2048` (the second B block, also with `m > 128`), `elim::rref` / `FlatMatrix` Off vs Auto vs Force, the refusals), the digests,
`GOLDEN_STRICT=1`, and the `cross_config.sh` arm `legacy-vnni-off` (the `OC_ENABLE_LEGACY=ON` build under `ENGINE_VNNI=0`; was `RelWithDebInfo@ENGINE_VNNI=0`).
**(S26 LEVERS-26)** The env arm `ENGINE_VNNI=0` / `off` is **retired** in `OC_ENABLE_LEGACY=OFF` builds: refused (`REFUSED ENGINE_VNNI=0: compiled out (OC_ENABLE_LEGACY=OFF)`, exit 1) at start-up and at the first read; `set_vnni_path(VnniPath::Off)` and the u32 route stay in every build; ON builds read the variable as above.

### 6.2 The two-limb VNNI route (slot-24 G4)

For `VNNI_MAX_MODULUS = 128 < p ≤ VNNI2_MAX_MODULUS = 2^14` every canonical residue is ≤ 16383. `gemm<uint32_t>` splits A into
7-bit limbs `a = a0 + 128·a1` (`a0 = a & 127`, `a1 = a >> 7`, both in `[0, 127]`) and pairs each B entry with
`b~ = 128·b mod p ∈ [0, p − 1]`, so `a·b ≡ a0·b + a1·b~ (mod p)`. Packed A: 8-row panels, one dword `a0 | a1 << 16` per (row, k);
packed B: 32-column panels, one dword `b | b~ << 16` per (k, column). The 8×32 microkernel (16 zmm i32 accumulators, the §6.1
shape) runs `vpdpwssd(acc, B dwords, broadcast A dword)`, which adds `b·a0 + b~·a1` (two s16×s16 products; every word is in
`[0, 16383]`, so the s16 and u16 readings coincide) to each lane. **Bound:** per k-term a lane grows by ≤ `2·127·16383 = 4 161 282`,
so a k-block of `VNNI2_KC = 512` terms stays ≤ `2 130 576 384 < 2^31 − 1` (`static_assert`, and `ENGINE_CHECK`ed per block as
`kb·2·127·16383 < 2^31` — the entry gate refuses an entry `≥ 2^14` behind a `Canonical` tag, so the bound holds whatever `p` is);
the non-saturating `vpdpwssd` never wraps and each lane holds an exact non-negative integer congruent to the block's `Σ a·b`.
**Reduction cadence:** once per k-block (every 512 terms) by the §6.1 FP64-quotient reduction (exact here too: the quotient is
within `2^-20` of `x/p`, and `q·p ≤ x + p < 2^31`), then the legacy `mod_add` / `mod_sub` combination into C. The canonical
residue of a congruent integer is the same residue, so the route is **bit-identical to the u32 microkernels by
construction**. `b~` is computed exactly in the packer (`x = 128·b < 2^21`, a `⌊2^32/p⌋` quotient off by at most one, one
correction). Blocking: A blocks of 128 rows (256 KB packed), B blocks of 1024 columns (2 MB packed, 64 KB per 32-column panel);
the scratch is thread-local. Composite moduli in the range qualify (`3^5`, `2^8`, `5^6`, `127^2`, `2^14`).

Selection is the §6.1 machinery: `Auto` routes `p ≤ 2^14` with the same CPU / kernel-path / size conditions, and for
`p ≥ 2^13` (where the u32 kernel's epilogue switches to its FP reduction and gets ≈ 20 % faster) additionally needs
`m, n ≥ 128` (G4 audit F1; below that the two-limb route measured 0.79–0.94× on 64³ / 96³ there; a speed choice only);
**(slot-25 PERF-VF (a)) and, for every two-limb modulus, the work-per-thread floor `m·n·k ≥ vnni2_work_floor() · vnni_route_threads()`**
(`VNNI2_WORK_FLOOR_DEFAULT = 2^20` since S26 LEVERS-26, `2^18` before; `vnni_route_threads()` = the global pool's configured size by the `ENGINE_THREADS` rule of
`Engine/Threads.h`, resolved once; `vnni2_work_passes(m, n, k, threads)` is the predicate). At 16 threads on the small ext cells the
packing and the per-thread scratch cost more than the route saves (`slot24/CLOSE_H2H_diagnosis.md` §3); at 1 thread the floor drops
only shapes below 128 × 128 × 64. **Calibration (S26 LEVERS-26, `slot26/LEVERS26_report.md` §3; re-take of `slot25/PERFVF_report.md` §3):** `--bench ext 160 691 2,3,5 --repeat 3` under `gate.lock`, `PREWINDOW foreign_count=0`, five floors rotated over 7 rounds: at 16 threads the median `total_s` is 0.2741 (floor 0) / 0.2648 (2^16) / 0.2544 (2^18) / **0.2426 (2^20)** / 0.2472 (off), min 0.2501 (2^18) vs 0.2411 (2^20); at 1 thread 0.8632 (2^18) vs 0.8531 (2^20), off 1.1704. Hence the default **2^20** (2^18, the slot-25 provisional value, was 4.6 % slower at t16 on the median); `floor · threads` saturates at `UINT64_MAX`. `ENGINE_VNNI2_FLOOR=<digits>` (read once;
`0` = the slot-24 rule) overrides the floor; a set value prints `EFFECTIVE ENGINE_VNNI2_FLOOR floor=<f> value=<raw> threads=<t>` on
stderr **when the floor is first consulted** (under `Auto`, at the first two-limb call that passes the large-p test; under `ENGINE_VNNI=0`
or `Force`, or in a run without such a call, nothing prints); a value that is not a string of digits, or overflows 64 bits, keeps the
default and also prints a WARNING. The one-limb route has no floor. A speed choice only;
`Off` (`ENGINE_VNNI=0`) turns **both** limb counts off; `Force` routes every `p ≤ 2^14` call; `ENGINE_VNNI_SHADOW=1` covers both
routes. `vnni_limbs_for(p)` is `1`, `2` or `0`; `vnni2_routed_calls()` counts the two-limb calls (they are also in
`vnni_routed_calls()`). A set `ENGINE_VNNI` prints one line on stderr when it is first read (the EFFECTIVE convention):
`EFFECTIVE ENGINE_VNNI arm=<off|auto> value=<raw> vnni_available=<0|1> one_limb_max=128 two_limb_max=16384`; unset prints nothing.
The setting is resolved exactly once per process (a function-local static), also when `set_vnni_path` overrides it first; a value
other than `0`, `off`, `1` (or empty) additionally prints `WARNING engine: ENGINE_VNNI='<v>' is neither 0, off nor 1; the VNNI
route stays ON` — the semantics are unchanged (only `0` / `off` turn it off). **A refusal leaves C undefined** (G4 audit F4): the
entry gate fires per (k-block, i-block), after earlier blocks may already have been combined into C, so a caller that catches the
`std::logic_error` must not read C (the one-limb route behaves the same; the case is reachable only on contract-violating input).
Without `avx512_vnni` (e.g. an AMD Zen 3 cluster node) the u32 kernel runs, as for §6.1. Gate: `VnniGemmBitIdentityValidation`'s
(G2) arm, the two-limb (W) worst cases (all entries `p − 1` at `p = 2^14`, 16381, 131, 997 over 1–3 k-blocks), (E) at 131 / 691 /
16381, (R) / (S); the digests, `GOLDEN_STRICT=1`, the `cross_config.sh` arm `legacy-vnni-off`.

### 6.3 The AVX2 kernels (slot-24 AVX2K)

For CPUs with AVX2 but no AVX-512 (an AMD EPYC 7763 / Zen 3 node, an Intel Core i9-9900KF), which before AVX2K ran the portable body
(11.9–15× the box's wall on the C1K-b2 cells, `C1K_b2_return_report.md` §3). Two pieces, both in `Kernels.cpp`:

* **The u64 microkernel** `micro_u32_avx2<MR>` (target `avx2,fma`), every `p < 2^32`: the §6 outer product on ymm registers — a
  sub-tile of `RG ≤ 4` rows × 8 columns keeps `RG × 2` accumulators of 4 exact u64 sums (vpmuludq), an `MR × 16` tile is swept as
  `⌈MR/4⌉ × 2` sub-tiles; column tails use `vpmaskmovd` (never touches a masked lane's memory). Epilogue: `ctx.reduce_u64` +
  `combine_cell`, the portable body's own scalar formulas. Same k-blocking as §5 (`kb ≤ min(k_max, 256)`).
* **The pair route** `gemm_u32_avx2_pair`, `2 ≤ p ≤ AVX2_PAIR_MAX_MODULUS = 4096` (every C1K prime): residues are non-negative i16,
  so A is packed into 6-row panels and B into 16-column panels of k-pair dwords `x(k) | x(k+1) << 16` (an odd tail paired with 0)
  and multiplied with vpmaddwd (16 exact multiply-adds per instruction) into exact i32 lanes. k-block `kb = avx2_pair_kc(p)`: the
  largest even `kb ≤ 512` with `(kb/2)·2(p−1)² ≤ 2^31 − 1` (≥ 128 at `p = 4096`; `ENGINE_CHECK`ed per block). Each block is reduced
  to the canonical residue (FP64 quotient, one correction each way; exact for `0 ≤ x < 2^31`) and combined with mod_add / mod_sub
  in 8-lane vectors (row-major C; a strided C reduces cell by cell). An operand entry `≥ p` behind a Canonical tag is refused
  (`ENGINE_CHECK`, the lane bound needs entries `< p`); as on the VNNI routes, **a refusal leaves C undefined**.

Both are bit-identical to the portable and AVX-512 bodies by construction (exact integer sums; the canonical residue is unique).

**Selection.** `KernelPath` gains `Avx2`; `Auto` resolves to AVX-512 if the CPU has it, else AVX2 (avx2 + fma), else portable —
runtime dispatch (`__builtin_cpu_supports`, target attributes), so one x86-64 binary runs everywhere. `active_kernel_path()` is
the resolved path. A forced `Avx2` path never takes the §6.1 / §6.2 VNNI routes (it is the Zen 3 arm). The pair route
(`Avx2PairPath::Auto | Off | Force`, `set_avx2_pair_path`) is taken under `Auto` when the resolved path is `Avx2` and
`m ≥ 4, n ≥ 16, k ≥ 16` (a speed threshold only); `Force` routes every `p ≤ 4096` call (validation use). Read-only env, each
read once, each printing one `EFFECTIVE` line on stderr when set (unset prints nothing):
`ENGINE_KERNEL=portable|avx2|avx512|auto` (the path `Auto` follows; the Zen 3 proxy on an AVX-512 box is `ENGINE_KERNEL=avx2`;
an unavailable or unknown value prints a `WARNING` and stays `auto`) → `EFFECTIVE ENGINE_KERNEL path=<p> value=<v>
cpu=avx2:<0|1>,avx512:<0|1>`; `ENGINE_AVX2_PAIR=0|off` (rollback: the AVX2 path runs its u64 microkernel for every p) →
`EFFECTIVE ENGINE_AVX2_PAIR arm=<off|auto> value=<v> avx2_available=<0|1> max_modulus=4096`. Neither can change a value.
**(S24 CL-5, C-06) One ISA policy.** `KernelPath`, `ring2k::KernelPath` and `binary::RowPath` are one type,
`engine::cpu::IsaPath` (`CpuFeatures.h`, which is also the one CPU probe), and `ENGINE_KERNEL` governs every module with ISA
bodies: `kernels::isa_ceiling()` is its resolved value and a module's `Auto` takes its best body at or below it
(`kernels::isa_auto`: `portable` = portable bodies everywhere, `avx2` = no AVX-512 body anywhere, `avx512` / `auto` = the best
the CPU has). Governed: this GEMM (exactly as above), ring2k, `BitMatrix` / `TritMatrix`, the rho fill (`RhoFillPath`), the
descent Smith kernels (`smithk`) and ring3k's small-modulus rows (portable only; their bodies are `target_clones`). A module's
explicit setter or its own variable (`set_kernel_path`, `set_row_path`, `ScopedRhoFillPath`, `HECKE_RHO_SIMD`,
`HECKE_DESCENT_SMITH_SIMD`) still wins. The `EFFECTIVE ENGINE_KERNEL` line ends in
`governs=gemm,ring2k,ring3k(portable),bitmatrix,tritmatrix,rho_fill,descent_smith`.
`avx2_pair_routed_calls()` counts the routed calls. Gate: `KernelAvx2BitIdentityValidation` (`fast`; arms portable, avx2-u64,
avx2-pair, avx2-auto, avx512 against `RefLinAlg` over moduli 2 … 4294967291 incl. the pair route's edge 4093 / 4096 / 4099, every
lane remainder, k across the blocks, the worst case, layouts; (A) / (S) / (R)); `KernelBitIdentityValidation` and
`VnniGemmBitIdentityValidation` run an AVX2 arm too. Measured (box, Ryzen 9 7950X, single thread, p = 134217757, u64 microkernel):
≈ 12–15 G vs AVX-512 15–25 G and portable 1.6–2.0 G; C1K cells: `docs/implementation/slot24/AVX2K_report.md`.

### 6.4 The RNS Garner kernels (`RnsKernels.h`, `engine::rns`, slot-25 EXZ2-P2)

The multimodular lifts' entries axis: E integers held as L residue planes `r_k[e] = x_e mod q_k`, one plane per word prime.
Word-only (uint32 planes); the big integers live in the caller (`nf::BigInt` in `Integral/Descent/`), never in ModArithmetic.

* **`GarnerPlan(primes)`** — the constants of Garner's mixed radix `x = d_0 + d_1 q_0 + d_2 q_0 q_1 + …` over `q_0 … q_{L−1}`:
  `coef[k·L + i] = c_{k,i} = ∏_{j<i} q_j mod q_k` and `inv[k] = (∏_{j<k} q_j)^{−1} mod q_k`, each with its Shoup quotient
  (`ModCtx::shoup_precompute`). **Preconditions** (`ENGINE_CHECK`ed): every `q` odd, `2 < q ≤ 2^31` (the Shoup lazy product needs
  `2q ≤ 2^32`), distinct, prime (the inverse is checked). Row k depends on `q_0 … q_k` only, so a plan over a longer prime list is
  **prefix-stable**: digit planes computed under it equal those of the shorter plan (`FreeBlockCrt` extends its digits plane by plane).
* **`garner_step(plan, k, digits, res, out, E, path)`** — one digit plane:
  `d_k[e] = (r_k[e] − Σ_{i<k} d_i[e]·c_{k,i}) · inv_k mod q_k`. **Preconditions** (the caller's; `garner_digits` debug-checks the
  first): residues canonical (`r_k[e] < q_k`), digits `d_i[e] < q_i` (they may exceed q_k: every Shoup product is lazy in `[0, 2q_k)`
  for any 32-bit operand and canonicalised before the next mod-add). `out` may alias `res`. Every step returns the canonical
  residue of the same integer, so the bodies are **bit-identical by construction**.
* **`garner_digits(plan, res, E, digits, path)`** — all L planes; **`garner_negative(plan, digits, e)`** — the sign of the symmetric
  representative in `(−P/2, P/2]` (P odd): true iff the mixed-radix value exceeds `(P − 1)/2`, whose digits are `(q_k − 1)/2`;
  exactly `(P − 1)/2` is non-negative. **`garner_step_oracle`** — the scalar oracle (plain 64-bit `%`, Horner digits, no Shoup).
* **Bodies and selection.** Portable (`ModCtx::shoup_mul_lazy` per entry), AVX2 (8 × u32: vpmuludq on even / odd halves + vpmulld +
  vpminud) and AVX-512 (16 × u32, the same on zmm); the vector bodies finish a tail by the portable loop. `path` is an `IsaPath`:
  a forced path that the CPU lacks falls back (AVX-512 → AVX2 → portable); `Auto` takes `kernels::isa_auto(avx2, avx512)`, i.e. the
  best body at or below the `ENGINE_KERNEL` ceiling (§6.3). `garner_resolved_path(path)`, `garner_avx2_available()`,
  `garner_avx512_available()` report it. No env variable of its own; none can change a value.
* **Gate.** `KernelBitIdentityValidation`'s RNS battery: the three bodies against the oracle on three prime sets (incl. `2^31 − 1`
  and tiny moduli), every tail 1 … 40 and 63 / 257 / 1031 / 4099 entries, random and worst-case digits, the two-prime round trip and
  the sign rule; the multi-prime lift is exercised end to end by `HeckeZFreeBlockCrtValidation` (FNV(F) identical across bodies).

## 7. Oracle (`RefLinAlg.h`, `engine::ref`)

`gemm(p, A, B, C, op 0/1/2)`, `gemv`, `trmm(p, T, B, C, upper)`, `transpose`,
`rref(p, M) → pivot columns` (in place, D5b conventions: leftmost pivot, topmost non-zero row,
leading 1, eliminate above and below), `kernel_basis(p, A) → vector<vector<Elt>>` (one vector
per free column ascending, 1 at the free column, `−RREF[i][j_free]` at pivot i — verified
entrywise against `FlatMatrix::kernel()`). Legacy `mod_mul_w`/`mod_add_w` only; never optimise.

## 8. Elimination (`DenseElim.h`, `engine::elim`) — Phase 2 Stage B

Everything the legacy substrate derives from an RREF, on `DenseView`s, both widths.
`rref(ctx, A)` (in place; row-contiguous is the fast path, other layouts are relaid out once)
→ `RrefResult{rank, pivots}`; `rref_with_transform(ctx, A, R, U)` with **`U` = trailing `m`
columns of `RREF([A | I_m])`** (D5b); `kernel_basis(ctx, R, pivots)` → `(n−k) × n` buffer, one
**row** per free column ascending (1 at the free column, `−R[i][j_free]` at pivot `i`);
`image_basis(A, pivots)` → `k × m`, the original columns (so `α·e_j` for `α·I`, Q11);
`inverse(ctx, A, Ainv)` → `false` when singular; `solve_multi_rhs(ctx, B, Y)` (coordinates of
the columns of `Y` in the columns of `B`, throws the legacy message when a column is outside the
span); `complement_basis(ctx, Rl, G, &projector)` (greedy leftmost-independent generators modulo
the relations + the projector rows of `RREF([Rl | G | I])`); `eliminate_against(ctx, B, R,
pivots, l_cols)` (`L = B[:, pivots]`, `B −= L·R` in place, returns `L`); `sub_product(ctx, C, L,
R)` (`C −= L·R`).

**Algorithm.** Packed column-panel PLU with the legacy pivot rule (leftmost column with a
non-zero Schur-complement entry, topmost such row), bisected inside a panel down to a
left-looking Crout base with delayed accumulation; each finished left half updates its right
half — and each finished panel the trailing columns — with one unit-lower TRSM (recursive,
GEMM-split) and one `gemm(GemmOp::Sub)`. The RREF is `U1^{-1}·U`: one `batch_inv` and a
unit-upper TRSM on the `k × (n−k)` non-pivot block, which is empty at full column rank. All
accumulators hold ≤ `k_max` products.

**Why it is bit-identical to legacy.** RREF, its pivot columns (= column rank profile), the
augmented-RREF `U` and every basis above are *unique* functions of the input; a blocked scheme
only has to be correct. `DenseElimValidation` checks all of them entrywise against `ref::rref` /
`ref::kernel_basis` and against the pre-Phase-2 `FlatMatrix` / `factor_basis_block` /
`change_of_basis_fm_multi` paths (reached via `engine::elim::ScopedLegacy`), and sweeps the
blocking parameters; `DenseElimPipelineValidation` does the same on the real `A_Cq` matrices and
on a whole `compute_cohomology()` run.

**Tunables.** `Params{panel = 256, base = 32, trsm_base = 32}`; env `ENGINE_ELIM_PANEL` /
`ENGINE_ELIM_BASE` / `ENGINE_ELIM_TRSM_BASE` for tuning runs, `set_params()` for tests. Values
never affect results. Measured (default build, `p = 134217757`, `--bench micro --backend engine`):
`rref 1677×1419` full rank **0.068 s** (legacy 3.52 s), rank 258 **0.033 s** (1.45 s),
`kernel 1677×1419 r258` **0.033 s** (1.22 s), identical checksums.

**Backend switch.** `engine::backend_is_legacy()` (`ENGINE_BACKEND=legacy`, `EngineConfig.h`)
is the process-wide rollback; `engine::elim::ScopedLegacy` is a thread-local override;
`use_legacy()` is what every retrofitted `FlatMatrix` / `BlockFlatMatrix` path tests. A build
with `-DENGINE_SHADOW_LEGACY=1` runs legacy + engine on every retrofitted call and throws
`std::logic_error` on the first differing entry. Since S24 CL-L1 the shadow needs `-DOC_ENABLE_LEGACY=ON` (CMake
`FATAL_ERROR` otherwise); its comparison code is `src/ModArithmetic/legacy/ShadowLegacy.inc` (and the two
consumer `.inc` files listed in `docs/architecture-detail.md`), reached from one-line `#ifdef ENGINE_SHADOW_LEGACY` hooks.
**Since S24 CL-L2** `ScopedLegacy`, `legacy_forced()` and `use_legacy()` exist only in an `OC_ENABLE_LEGACY=ON` build
(`legacy/DenseElimLegacy.inc`; declared in `DenseElim.h` under `#if OC_ENABLE_LEGACY`), and so do the legacy routes they
select (`legacy/FlatMatrixLegacy.inc`, `legacy/BlockFlatMatrix.{h,cpp}`): in an OFF build every retrofitted call is the
engine route, `engine_path(p)` is `p >= 2`, and `ENGINE_BACKEND=legacy` is refused at start-up. `factor_basis_block`
(legacy) has one former default-route caller, `CochainComplex::compute_cohomology_bases`, which now calls
`elim::complement_basis` (the same rule; `DenseElimValidation` pins the two entrywise in ON builds).

*Direct-engine callers and the legacy arm (slot-22 PERF1).* Two paths call the engine directly and have no FlatMatrix
twin of their own: the cone's span tests (`tape::factor_basis`, twin `CONE_SPAN_SHADOW`) and the iota carry check
(iii-c) (`column_rank_profile`). The iota check honours `use_legacy()` (`ENGINE_BACKEND=legacy`, `ENGINE_SHADOW_LEGACY`)
by taking the old `FlatMatrix::get_pivot_columns()` route, so those arms still cover it (`IOTA_CARRY_ROUTE` line of
`HeckeInvolutionValidation`). (Slot-24 G6) Under `HECKE_FIELD_DIET` (default on) the cone's span tests first try the
**unit-row route** (`Y ⊂ span B` iff `Y[r,:] == B[r,:]·Y[u,:]` on the non-unit rows, one tape GEMM + COMPARE, the
statement of `solve_multi_rhs`'s unit-row shortcut restricted to the rows where it can fail) and fall back to
`tape::factor_basis` when a column of `B` has no unit row; `CONE_SPAN_SHADOW` checks either route against the dense RREF.
`tape::factor_basis` itself (memory only, no switch) releases stage 1's RREF buffer `R` as soon as its pivots and row
rank profile are read (stage 2 re-reads `[Rl | G]`), and forms stage 2's `Mact · Winv` in row slabs of 1024 rows.

### 8.1 The native `F_2` route: `engine::binary` and `ENGINE_BINARY_F2` (slot-22 BIN-1)

Files: `src/ModArithmetic/Binary/BitMatrix.{h,cpp}` (in the `ModArithmetic` library), the route in `DenseElim.{h,cpp}`. Gate:
`BinaryF2Validation` (standard). Plan: `docs/implementation/slot22/L23_master_plan.md` §2 (rung R0); report `slot22/BIN1_report.md`.

**`binary::BitMatrix`**: an `m × n` matrix over `F_2`, one bit per entry, rows of `W = ⌈n/512⌉·8` `uint64_t` words (bit `j` = bit
`j % 64` of word `j / 64`; padding bits always zero). `from_dense<Elt>(DenseView<const Elt>)` (any layout; `ENGINE_CHECK`s every entry is
0 or 1) / `to_dense<Elt>(DenseView<Elt>)`, `fnv()`, `get / set / flip / swap_rows`. Entry points: `rref(B)` (in place, D5b conventions),
`echelon(B)` (pivots only, half the work), `rank(B)`, `kernel_basis(R, pivots)` (the §8 layout: one row per free column, ascending),
`kernel(B)`, `multiply_m4rm(A, B)`; the references `rref_naive`, `multiply_naive`. Elimination is M4RI with `K = 8` (strips of up to 8
pivots, the strip's pivot rows mutually reduced, one 256-entry subset table, one lookup + one row XOR per other row); the product is M4RM
(8-row tables of `B`). Row XORs: AVX-512 / AVX2 bodies under target attributes, picked at runtime (`__builtin_cpu_supports`), and a
portable body (the only one on aarch64 / non-GCC); `set_row_path(Auto|Portable|Avx2|Avx512)` forces one (validation use). Single-threaded.

**The route.** **Opt-in, default off.** With `ENGINE_BINARY_F2=1` (exactly `1`; read once per process) or a `elim::ScopedBinaryF2(true)`
in force (process-wide; validation use), the internal RREF routine behind `rref`, `rref_with_transform`, `inverse`, `solve_multi_rhs`,
`complement_basis`, and `column_rank_profile` (echelon only; `A` left untouched) pack `A` into a `BitMatrix`, eliminate and unpack, for
a **field context with `p == 2`**, `uint32_t` or `uint64_t`. The device hook (§11.9) is asked first (at `p = 2` it can fire only under
`ENGINE_DEVICE=gpu` together with `ENGINE_GPU_ELIM`, the size floor and a working IMMA device — a necessary condition, not the
full one, BIN-1 audit F10; either route is bit-identical). `tape::rref` and the block model (`engine::blk`) are **not** routed. `elim::binary_f2_totals()` counts routed calls
(diagnostic). **It cannot change a value**: RREF, rank and pivots are unique functions of `A` (§8) and an XOR elimination is exact;
`BinaryF2Validation` proves it entrywise on 500 random + 100 real `A_Cq` matrices mod 2 on every row path, plus a whole
`compute_cohomology()` at `ell = 2` route on = off; `GOLDEN_STRICT=1 ENGINE_BINARY_F2=1 GoldenRegressionValidation` is clean.

**Measured** (7950X, one thread, census load ≈ 14, `--bench micro rref N N 0 2 --repeat 3`, min; identical checksums): `N = 1024`
0.0283 → 0.0023 s (12×), `4096` 1.371 → 0.0300 s (46×), `8192` 11.49 → 0.165 s (70×), including pack / unpack; `BitMatrix` alone at
4096: 0.022–0.028 s (target ≤ 0.06 s).

*BIN-2 audit repairs (2026-10-10).* R1 (audit F1): `BinaryF2Validation` packs with `from_dense<uint32_t>` on every row path (14 539 checks);
F2: one strided (transposed-view) round trip; F9: the redundant table `memset` is gone (`T[0]` is zero from the allocation); F10: the device
hook wording above. F5 (word-level "any candidate" test in the pivot search) is not done.

### 8.2 The `Z/2^k` ladder: `engine::ring2k` (slot-22 BIN-2)

Files: `src/ModArithmetic/Binary/Ring2k.{h,cpp}` (in the `ModArithmetic` library; joined from `cmake/slot22/BIN2.cmake`, compiled with
`-ftree-vectorize`), the reference rung `src/hacs/Binary/Ring2kReference.{h,cpp}` (`engine::ring2k::ref6`, `cohomology_app` only: it uses
`nf::BigInt`, so it lives outside `ModArithmetic/` — `check_numberfield_includes.sh` rule 2; moved there in slot 23). Gate: `Ring2kLadderValidation` (standard,
`src/hacs/Binary/Ring2kLadderValidation.{h,cpp}`). Plan: `docs/implementation/slot22/L23_master_plan.md` §2.2; report
`slot22/BIN2_report.md`. **Its first consumer is BIN-3's degree-1 amalgam reader** (`src/hacs/Orbifold/L23/AmalgamReader.{h,cpp}`, `l23::read_lattice` /
`h1_mod` / `oddn_row`, gate `L23AmalgamReaderValidation`: `smith`, `howell`, `kernel_module` and `smith_adaptive` with the rank over Q as the hint;
at ℓ = 3 the same reader runs on `engine::ladic`); no route, no setting, no golden line.
**Second consumer (slot-23 Z2K):** the (Z) phase of an ℓ = 2 orbifold census cell (`Campaign/CampaignOrbifold.cpp`,
`OrbifoldSmithTorsion` / `OrbifoldKernelModule`): the Smith readings of `δ⁰` over `Z/2^31` (R2 u32) and of `δ¹`, `δ²` over `Z/16` (R2 u16), and the
(U) kernel modules over `Z/16` (R2 u16), in place of `elim::invariant_factors` / `elim::kernel_module`. The torsion exponents are the Smith valuations
`σ ≥ 1` (`invariant_factors` `e < K` gives `K − e = σ`); the Howell form is equal entrywise. **On by default**; `L23_RING2K=0` (read once; any
other value than `0` / `1` warns and leaves it on) is the rollback, `campaign::ScopedOrbifoldRing2k` the validation override, and a verbose
orbifold campaign prints `EFFECTIVE l23_ring2k=on|off source=default|env|scoped`. ℓ = 3 is Z3K's `engine::ring3k` (§8.4). Gate `OrbifoldRing2kValidation`
(route on = off on the real coboundaries and on whole ℓ = 2 records, bodies byte-identical after masking the sum-rule oracle's wall time);
report `docs/implementation/slot23/Z2K_report.md`.

**Rungs** (`select_rung(k)`, the smallest that carries `k`; `next_rung` one up; `rungs_for(k)` all of them):

| rung | element `E` | carries `k ≤` | arithmetic |
|---|---|---|---|
| R0 | `binary::BitMatrix` (§8.1) | 1 | XOR / AND (only through the wide API) |
| R1 | `uint8_t` | 8 | native wraparound (narrow types computed in `unsigned`, never `int`) |
| R2 | `uint16_t` / `uint32_t` | 16 / 32 | native |
| R3 | `uint64_t` | 64 | native |
| R4 | `unsigned __int128` | 128 | native (three `mul`) |
| R5 | `WordN<N>`, `N ∈ {3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64}` | `64 N` (≤ 4096) | `N` little-endian words, truncated schoolbook product (`N(N+1)/2` word `mul`), carries out of the top word dropped |
| R6 | `nf::BigInt` (`ref6`) | any | **oracle only**: `floor_mod(2^k)` after every operation, extended-Euclid inverse |

**The rule: nothing ever reduces.** A rung computes in `Z/2^W` (`W` = its width) by wraparound; `Z/2^k`, `k ≤ W`, is a quotient. Only the
valuation (`ctz` of the value masked to `k` bits, capped at `k`), the zero test and the output (masked to `k` bits, canonical in `[0, 2^k)`) look
at the value mod `2^k`. Exact division by `2^v` is `>> v`; the inverse of an odd unit is Newton `x ← x(2 − ux)` from `x₀ = u` (3 bits) to the full
width, so `ux = 1` mod `2^W` and an elimination step `x − f·pivot` is exactly zero in `Z/2^W`. A multiplier read from unmasked high bits differs
by a multiple of `2^{k−v}`, which is still an elementary row operation, so no output depends on the garbage above bit `k`.

**Entry points** (templates on `Traits<E>`; the production rungs are instantiated in `Ring2k.cpp`; all `ENGINE_CHECK 1 ≤ k ≤ W`):

* `smith<Tr>(Mat<E>, k)` → `SmithResult{k, sigma, level_ranks}`: the Smith valuations `σ₁ ≤ σ₂ ≤ …` (each `< k`; the non-zero diagonal entries
  `2^σ` of the Smith form), so `rowspan(A) ≅ colspan(A) ≅ ⊕ Z/2^{k−σᵢ}`. `invariant_factors` (§13.6) is `{k − σ}` descending; `length` is `Σ (k − σ)`.
  **Algorithm (valuation levels):** at level `s`, the F₂ column rank profile of the matrix mod 2 (BIN-1's `binary::echelon`, rank `r_s`) is moved to
  the front; a blocked right-looking LU (panels of `Params::panel = 64` columns, pivot = the first **odd** entry of the column, unit-lower TRSM on
  the panel's pivot rows and one GEMM for the rows below) eliminates those `r_s` columns, never skipping one; that gives `r_s` invariants `σ = s`;
  the Schur complement on the remaining rows and columns is even, is halved exactly and is level `s + 1` over `Z/2^{k−s−1}`. Correctness: with unit
  pivots the pivot rows clear by column operations without touching the Schur block, so `Smith(A) = 0^{r_s} ∪ (1 + Smith(S/2))`. **A skipped
  column is not allowed** under the odd rule: the later pivot rows keep even, non-zero entries there that no update (panel AXPY from the pivot
  column, trailing GEMM from the panel end) ever reaches — the first BIN-2 draft did exactly that and the gate caught it (748 failures).
* `howell<Tr>(Mat<E>, k)` → `HowellForm{k, H, pivots, pivot_v}` = **the** Howell form of §13.5, equal entrywise to `elim::howell`: a blocked LU with
  **minimal-valuation** pivots (rowspan unchanged; skipped columns are 0 mod `2^k` in every later row, so nothing is lost), then the §13.5 pass
  (active list, minimal valuation / topmost, scale to `2^v`, exact elimination, closure row `2^{k−v}·P` at the bottom) on the `≤ n` echelon rows,
  then reduce-above in blocks of `Params::reduce_block = 64` pivots (each row's floor quotients against the block computed sequentially on the
  block's pivot columns, one GEMM applies them; the intra-block triangle row by row).
* `kernel_module<Tr>(Mat<E>, k)`: the Howell form of `{x : Ax = 0}`, by §13.6's rule (rows of `howell([Aᵀ | Iₙ])` leading at `≥ m`), equal entrywise
  to `elim::kernel_module`.
* References (unblocked, validation use): `smith_naive` (global minimal-valuation pivoting), `howell_naive` (§13.5 on `A` itself).
* `echelon_blocked<Tr>(A, k, PivotRule, panel, col_limit)`: the LU above (exposed for BIN-3); read its CAUTION before using `FirstOdd` — since the
  BIN-2 audit (F4) a `FirstOdd` column without an odd entry is an `ENGINE_CHECK` failure, not a silent skip.
* **The wide interface** (rung-independent): `WideMatrix{rows, cols, k, nw = ⌈k/64⌉ words per entry}`, `WideHowell`; `smith_on / howell_on /
  kernel_module_on(RungChoice, A)` (any rung with capacity `≥ k`; R0 only for `k = 1`), `smith_wide / howell_wide / kernel_module_wide` (=
  `select_rung(A.k)`), `smith_naive_on / howell_naive_on`.
* **The adaptive rung** `smith_adaptive(produce, k_min, guard, k_cap, rank_hint = −1)`: runs at the **full capacity** `K` of `select_rung(k_min)`
  (a wider word costs nothing extra in precision), and reruns one rung up (`next_rung`, `K` = its capacity, up to `k_cap`) while some `σ ≥ K − guard`
  or, with `rank_hint ≥ 0` (the rank over Q, e.g. from a large prime), while fewer than `rank_hint` valuations are `< K − guard`; one
  `RING2K_ADAPTIVE …` log line per run; `resolved = false` at `k_cap`. Plan §2.2's precision rule is the case `guard = e₁ + e₂` (BIN-3 supplies it).
  **Invisible invariants (BIN-2 audit R1).** An invariant `2^σ` with `σ ≥ K` is a zero diagonal entry over `Z/2^K` and is absent from `sigma`:
  without `rank_hint`, `resolved` means only that no *visible* `σ` lies within `guard` of `K`. `AdaptiveSmith::zeros = min(rows, cols) − |σ|`
  counts those entries, `verified` is `resolved` with a rank hint, and the log line prints `zeros=` and reads `resolved (unverified: no rank
  hint)` when `zeros > 0` without a hint. **Callers that need every invariant pass the rank over Q** (BIN-3's degree-1 reader passes the rank of
  the relation matrix computed modulo two large primes).

**Memory (BIN-2 audit R2; element width `w` bytes, `A` `m × n` of rank `r`).** `smith`, `howell` and `howell_pass` take the matrix **by value**
(pass an rvalue: `smith_on` / `howell_on` hand over the `from_wide` temporary). `smith` permutes the columns and compacts the halved Schur
complement **in place**: peak `≈ m n w` (+ an `m n`-bit `BitMatrix` per level). `howell` runs the LU in place, keeps the `r` echelon rows in the
same storage and hands it to `howell_pass`, whose closure rows (at most one per non-unit pivot, at most `n`) are allocated **on demand**: peak
`≈ (m n + c n + h n) w` with `c` closure rows and `h ≤ n` output rows. `kernel_module` forms `[Aᵀ | Iₙ]` (`n × (m + n)`) once and moves it into
`howell`. Measured (scratch probe `s22/bin3/mem/memprobe.cpp`, `u64`, a `3d × 2d` matrix of rank `≈ 2d − d/6` with `2^c`-scaled rows, `d = 1200`;
peak RSS above the input matrix `A` (`6 d² w`), in units of `d² w`): `smith` **0.0** (was 2.6), `howell` **1.9** (was 17.2), `kernel_module`
**40** (was 80; `[Aᵀ | Iₙ]` is `10 d² w`, the rest is the output and the closure rows, which this adversarial family makes numerous).

**Kernels.** The native rungs' `kernel_axpy` / `kernel_gemm_sub` have portable / AVX2 / AVX-512 (`avx512f,bw,dq,vl`) bodies under target
attributes, picked once by `__builtin_cpu_supports`; `set_kernel_path(Auto|Portable|Avx2|Avx512)` forces one (validation use; `ENGINE_CHECK`s
availability). The GEMM is a register-blocked micro-kernel (4 rows × 2 vectors of accumulators, GCC / Clang vector extensions, unsigned lanes)
inside L2 column tiles. Every body is bit-identical by construction (wraparound integer arithmetic has one answer); the gate re-runs matrices on
every path and fuzzes the kernels per path. aarch64 / non-GCC: the portable body only (checked to compile warning-free). `R4` is scalar; `R5`
has its own SIMD kernels (below, slot-23 R5V).
Single-threaded, except inside `engine::smithpar::ScopedEnable` (slot-24 PAR-2K, `Binary/SmithPar.{h,cpp}`, lever
`HECKE_DESCENT_SMITH_MT`; only the integral descent's Smith calls open it): there `echelon_blocked` splits the trailing GEMM into tiles
of C, the panel TRSM into column tiles, and runs the panel **left-looking** (the pivot search brings only the rows it reads up to date;
the other rows receive the panel's pivots in one parallel region), and `smith` splits its per-level passes (F_2 build, column
permutation, Schur halving) over row tiles. Each row receives the serial operations in the serial order, the pivots are the serial ones,
so every output is bit-identical at every thread count (`DescentSmithMtValidation`); `threads() == 1` runs the serial code verbatim.
The pool is private (ModArithmetic does not link the engine; `ENGINE_THREADS` workers, cap 256). `Params` (`set_params`) and
`counters()` are process-wide relaxed atomics; no value depends on `Params`.
(Slot-24 PAR-UNIFY.) The pool is the process's **one descent pool**, shared with PAR-D2's `maps::SmithForm`
(`smithpar::parallel_for_n`, an explicit thread count without the opt-in); the budget `smithpar::default_threads()` is a strictly parsed
`ENGINE_THREADS` (an integer in [1, 256]; garbage warns and takes the default), else the CPU affinity count (`sched_getaffinity`)
capped at 64. On the threaded route `smith` / `howell` (and ring3k's) run the LU with `smithpar::route_panel()` = 128 columns
(`HECKE_DESCENT_SMITH_PANEL=64` restores 64; the serial route keeps `Params::panel`); a value never depends on the panel.
(Slot-24 PAR-F2.) On the threaded route (`smithpar::f2_route()`: lever `HECKE_DESCENT_SMITH_F2_MT`, default on, and
`threads() > 1`) the per-level F_2 rank profile runs `binary::echelon_lazy` (BitMatrix.h): the serial M4RI strips and tables, but the
clears of up to eight strips reach each row below in one pass over row tiles, and the pivot search reads a **transposed 512-column window**
of the reduced rows (one row bitset per column, updated by every pivot) instead of scanning the rows -- the same final matrix, rank and
pivots as `binary::echelon`. Likewise `ring3k::smith_levels` / `smith_lengths` run `binary::echelon_lazy(TritMatrix&)` (a lazy blocked
Gauss), and the level passes are threaded further: the zero test over row tiles, the F_2 build by `binary::pack_low_bits` (AVX-512 /
AVX2 / portable) on the u32 / u64 rungs, the trit planes straight from A, the Schur compaction by `smithpar::compact_rows` (row tiles;
direct waves when the destinations lie below every unread source, else through a buffer of <= 32 MiB), and on the device route the
tiled L-store copy and the left-columns row permutation. `HECKE_DESCENT_SMITH_F2_MT=0` restores the PAR-2K code verbatim. Gate:
`DescentSmithMtValidation` arm (F).
**Device route (slot-24 PAR-GPU; `Binary/Ring2kDevice{.h,.cpp,LU.h}`, `Cuda/CudaRing2k.h`, `Cuda/ring2k_device.cu`, gate
`DescentSmithGpuValidation`).** `echelon_blocked` on the u32 / u64 rungs first asks an operations table (installed by cohomology_app,
`Integral/Descent/DescentSmithGpu.cpp`; ModArithmetic links no CUDA). Only under `ENGINE_DEVICE=gpu` (the thread's policy) and
`ENGINE_GPU_SMITH2K` ≠ 0, above `m·n ≥ ENGINE_GPU_SMITH2K_MIN` (2^24), with a device and its memory, the LU runs with the working matrix
resident on the device: per panel the host fetches the panel's columns (rows `r … m−1`), factors them with the lazy panel above (the host's
arithmetic, so the same pivots, swaps and L slots), and the device replays the swaps on its other columns, runs the unit-lower TRSM on the
pivot rows and the GEMM `C −= L U` (64 × 64 tiles, native wraparound `uint32_t` / `uint64_t` multiply-add, one thread per output entry, no
reduction, no atomics), updating the next panel's columns first and fetching them on a second stream (lookahead). When `m·n` does not fit,
the LU runs in phases: a window of columns at the next panel is factored on the device until its panels are done or the host store of its L
factors reaches `ENGINE_GPU_SMITH2K_HOST_MB` (4096 MiB); the columns right of the window then pass through the device in slabs that receive
the window's panels (replayed from the store), the columns left of it receive the window's swaps as one host row permutation. Each
column receives the host code's operations in the host code's order, and wraparound has one answer per entry, so the matrix, `EchelonInfo`
and every `smith` / `howell` output are bit-identical to the host route. A CudaError before the first host write falls back to the host
LU; after it, it is rethrown. Plain SIMT integer code: every architecture of `OC_CUDA_ARCHITECTURES` (75, 80, 86, 87, 89, 90, 120) runs it.

**Measured** (7950X, one thread, the box under census load ≈ 14; `n × n` uniform, Smith full rank; the engine's field RREF at `p = 134217757`
for scale). In the gate at `n = 1024`: R1 0.016 s, R2(u16) 0.011 s, R2(u32) 0.019 s, R3 0.024 s vs the field RREF 0.025 s (1.0–2.3× faster). A
scratch driver, same flags, `n = 4096`: R1 0.78 s, R2(u16) 0.69 s, R2(u32) 0.96 s, R3 1.72 s vs the field RREF 1.14 s
(`--bench micro rref 4096 4096 0 134217757`); R4 at 2048 3.5 s (R3 0.26 s). Against the engine's own `Z/2^k` route (`elim::invariant_factors` /
`elim::howell`, `n = 512`): Smith **78–97×**, Howell **12–21×** faster. R4 / R5 cost ≈ 10× the plan's `N(N+1)/2 ×` R3 model (R3 is 8-lane SIMD,
R4 / R5 scalar): R4(512) 10.5× R3, R5(N=4, 256) ≈ 109× (model 10), R5(N=16, 128) ≈ 600× (model 136; R3 at `n = 128` is too short to time well) — **plan §2.4's R5 rows are optimistic by that factor** until R5
is vectorised (SoA 32-bit limbs; BIN-3 / DEEP-0 item).

**R5 kernels (slot-23 R5V; `Ring2kR5.cpp`, `Ring2kR5Body.inc`, gate `Ring2kR5SimdValidation`).** The storage stays array-of-structures
(`Mat<WordN<N>>`); the kernels pack structure-of-arrays tiles. GEMM / AXPY (`C −= L U`): an element is cut into `NL` limbs of `B` bits
(`B = 28` for N ≤ 6, 27 for N = 8…16; `NL = ⌈64N/B⌉` = 7…38), one 64-bit lane per column (2 / 4 / 8 lanes), the truncated product's
`NL(NL+1)/2` limb products `32 × 32 → 64` (`vpmuludq`) added uncarried into register accumulators, carried every `TC` rank-one updates
(`TC NL 2^(2B) < 2^64`: 17…52), then `C − S` with a limb borrow chain and unpacked; the U tile stays ≤ 256 KiB. Row kernels (`d += s`,
`d −= s`, `d = −d`, used by the BIN-3 reader's Taylor shifts): chunks of `lcm(N, lanes)`-multiple words ≤ 64, the word sums lane-wise, the
carries of the whole chunk at once from generate / propagate bit masks cleared at each element's top word (`c = ((G ≪ 1) + P) ⊕ P`); the
portable path runs a scalar carry chain. Everything is mod `2^(64N)`, so every path is bit-identical to the scalar `Traits<WordN<N>>` body,
which stays as the oracle (`set_r5_kernel(Scalar)`). SIMD bodies for N ∈ {3, 4, 5, 6, 8, 10, 12, 16} (the reader's ladder); other N are
scalar. Measured (7950X, AVX-512, one thread): GEMM 6.3× (N = 3) … 4.1× (N = 16) the scalar body, rows 7.2× … 2.7×; DEEP (2, 500, full)
2.46×, (2, 1000, full) 2.19× end to end (`slot23/R5V_report.md`). The `≈ 10×` above is not reached: the limb split costs ≈ 2× of the 8 lanes.

**`kernel_shift_rows` (slot-23 BIN3O1; `Ring2k.cpp`, gate `L23Bin3o1Validation` (K)).** For the native rungs u8 … u64: on an `(n + 1) × w`
block (row stride `ld`), for `i = 0 … n − 1` and `j = n − 1` down to `i`, row `j −= f ·` row `j + 1` — exactly that sequence of `kernel_axpy`
calls (the BIN-3 reader's Taylor shift `p(t) → p(t − f)`), in one call. Each pass acts on every column separately, so the body walks one
vector of columns at a time and 16 passes per bottom-up walk over the rows (a register wavefront: `b[t]` = the chunk of row `j + 1` right after
pass `i₀ + t`); every pass still reads the same operands, so the wraparound values are the axpy sweep's on every path. `f = −1` / `+1` run as
plain add / subtract. Measured (7950X, AVX-512, u64): 0.024–0.034 ns per entry-pass against 0.11–0.13 for the axpy sweep (3.7–4.7×;
`n = 1000 … 2000`, `w = 167 … 333`).

### 8.3 The native `F_3` rung T0: `engine::binary::TritMatrix`, `ENGINE_TERNARY_F3` (slot-23 TER-1)

Files: `src/ModArithmetic/Binary/TritMatrix.{h,cpp}` (in the `ModArithmetic` library, joined from `cmake/slot23/TER1.cmake`), the route in `DenseElim.{h,cpp}`, the reader's `F_3` route in `src/hacs/Orbifold/L23/AmalgamReader.{h,cpp}`. Gate:
`TernaryF3Validation` (standard; `src/hacs/Binary/TernaryF3Validation.{h,cpp}`, outside `ModArithmetic/` because it uses `nf::BigInt`). Plan: `docs/implementation/slot22/L23_master_plan.md` §3 (C); `docs/implementation/slot23/00-slot23-plan.md` §2 TER-1;
predictions / report `slot23/TER1_predictions.md`, `TER1_report.md`.

**`binary::TritMatrix`**: an `m × n` matrix over `F_3` in two **one-hot** bitplanes per row (`one` = `[x = 1]`, `two` = `[x = 2]`), each plane
`W = ⌈n/512⌉·8` words (BIN-1's layout), the row stored as `[one | two]`; padding zero, never both bits set. `x + y` is 7 boolean operations per 64
entries (`t = (x₁|y₂) ⊕ (x₂|y₁); z₁ = (x₂|y₂) ⊕ t; z₂ = (x₁|y₁) ⊕ t`), `−x` swaps the planes. `from_dense<Elt>` (`ENGINE_CHECK`s every entry `< 3`) /
`to_dense<Elt>`, `from_u8` / `to_u8`, `fnv()`, `get / set / swap_rows / negate_row`. Entry points: `rref` (Gauss–Jordan one pivot at a time, every row
operation on whole words from the pivot's word on; D5b conventions), `echelon` (pivot rows not normalised), `rank`, `kernel_basis` (the §8 layout, `−R[i][j]`
at pivot `i`), `kernel`, `multiply`; references `rref_naive`, `multiply_naive`. Row bodies portable / AVX2 / AVX-512 (planes: `avx512f`; the byte kernels
`f3_add_u8 / f3_sub_u8 / f3_neg_u8`: `avx512bw`, else AVX2), chosen by BIN-1's `set_row_path` / `active_row_path` (one switch for both rungs).
Single-threaded. No M4RI table yet (a Gray-table analogue is the ≈ 2× lever the L23 plan estimated).

**The route.** **Opt-in, default off**, the twin of §8.1: `ENGINE_TERNARY_F3=1` (exactly `1`, read once) or `elim::ScopedTernaryF3(true)` routes the internal
RREF (so `rref`, `rref_with_transform`, `inverse`, `solve_multi_rhs`, `complement_basis`) and `column_rank_profile` of a **field context with `p == 3`**
(`uint32_t`, `uint64_t`) through a `TritMatrix`; the device hook and the `F_2` route are asked first. Counters `elim::ternary_f3_totals()`. **It cannot change
a value** (RREF, rank, pivots unique; the arithmetic exact); `GOLDEN_STRICT=1 ENGINE_TERNARY_F3=1 GoldenRegressionValidation` is clean.

**The reader's `F_3` route (on by default).** `l23::h1_mod(3, n, 1, ·)` runs on `Ops3F` (`uint8_t` entries): the structural `ρ_n` on the byte kernels, and
`ρ(T)^{±1}` by the characteristic-3 Taylor shift — with `M = 3^j`, `(t ± 1)^M = t^M ± 1`, so a block `q₀ + t^M q₁ + t^{2M} q₂` with shifted parts becomes
`(q₀ ± q₁ + q₂) + t^M (q₁ ∓ q₂) + t^{2M} q₂` (for `−1`: `(q₀ − q₁ + q₂) + t^M (q₁ + q₂) + t^{2M} q₂`), levels bottom up: `O(d log₃ d)` row operations instead of
`d²/2` — and its eliminations on `TritMatrix`. The `Ops3<1>` route computed in `Z/9` on `engine::ladic` (16 B per entry, scalar calls) and eliminated on the
`uint64_t` field route (a 128-bit division per product). `L23_TERNARY_F3=0` (read once; any value other than `0` / `1` warns and leaves the route on) is the rollback; `l23::ScopedTernaryF3Reader` the validation override; `l23::ternary_f3_effective_line()` (`EFFECTIVE l23_ternary_f3=on|off …`) is printed by `deep_cell` and the verbose orbifold campaign; `l23::taylor3_selfcheck` is the validation hook comparing the shift with the d²/2 one.
Measured: `(3, 1000, m2)` 71.8 s → 0.17 s, RSS 0.41 → 0.03 GB, identical values.

**`ring3k::Mod3k<N>`: deleted** (S24 CL-G3, CODEAUDIT C-04). It was an `N`-word Montgomery `Z/3^K` context (`K ≤ 40` at `N = 1`) whose one
consumer was `engine::ring3k`'s pivot-unit inverse; that inverse is now `ModCtx::inv` (§8.4). The inverse of a unit is unique, so no value moved.
`Z/3^K` beyond one word, if ever needed, is `engine::ladic` (§13.11).

### 8.4 The `Z/3^k` rung (word-sized): `engine::ring3k` (slot-23 Z3K)

Files: `src/ModArithmetic/Binary/Ring3k.{h,cpp}` (in the `ModArithmetic` library, joined from `cmake/slot23/Z3K.cmake`, compiled with `-ftree-vectorize`).
Gate: `OrbifoldMod3kValidation` (standard, `src/hacs/Campaign/OrbifoldMod3kValidation.{h,cpp}`). Predictions / report
`docs/implementation/slot23/Z3K_predictions.md`, `Z3K_report.md`.

**Domain.** `Z/3^k`, `1 ≤ k ≤ 19` (`3^19 < 2^31`), entries `uint32_t` canonical in `[0, 3^k)` (`ENGINE_CHECK`ed entrywise at entry). Entry points
`smith(A, k)` → `SmithResult{k, sigma, level_ranks}` (the Smith valuations, ascending, each `< k`; `invariant_factors` is `{k − σ}`), `howell(A, k)`
→ `HowellForm{k, H, pivots, pivot_v}`, equal entrywise to `elim::howell`; `kernel_module(A, k)`, equal entrywise to `elim::kernel_module` (§13.6's rule
via `[Aᵀ | Iₙ]`). References `smith_naive`, `howell_naive`. Every output is a unique function of its input, so no variant can change a value.

**Two variants, picked by `k`** (both reachable by name for the gate):
* **`k ≤ 5` (`3^k ≤ 243`; the census's `Z/9`): the small-modulus rows.** `uint16_t` entries; the §13.5 pass on the input rows themselves in the
  engine's own order (a row whose pivot-column entry is zero is skipped), each row update one loop modulo a compile-time `3^k` (`% M` by a constant, which GCC
  lowers to a multiply-shift; `target_clones` skylake-avx512 / haswell / default on x86_64 GCC: 32 u16 lanes on the AVX-512 clone, 16 on haswell,
  scalar on the default clone, NEON or scalar on aarch64); reduce-above row by row. `smith` by **lengths**:
  `L_j = length(rowspan(A mod 3^j))` (`L₁` = the `F₃` rank by TER-1's `TritMatrix`, `L_j` from the pass without reduce-above), and
  `L_j − L_{j−1} = #{σ < j}`.
* **`k > 5` (the census's `Z/3^19`): dense blocked.** `smith_levels`: valuation levels as `ring2k::smith` (§8.2) with `F₃` for `F₂` — the `F₃`
  column rank profile (`TritMatrix` echelon) moved to the front, a blocked unit-pivot LU (panel 64, never skipping a column), the Schur complement
  divided by 3 exactly. `howell_blocked`: a blocked minimal-valuation LU, the §13.5 pass on the echelon rows, a blocked reduce-above. Trailing
  updates are `kernels::gemm` over `ModCtx::prime_power(3, k)` (exact on a composite modulus, §13.2).
* Pivot-unit inverses: `ModCtx::chain(3, k).inv` (the extended Euclid, unique on units; `Mod3k<1>` until S24 CL-G3, §8.3). `engine::ladic` is not used: at `K ≤ 19` it is one digit, and §13.11.4 dispatches it to the
  same `elim` route.

**Why two.** The census coboundaries are sparse. On them the dense blocked LU does more work than the engine's zero-skipping scalar pass. Z3K
measured it **slower** than `elim` at `(3, 400)` (`Z/9` kernel of `δ²`: 15.0 s vs 10.6 s), and the small-modulus rows **faster** (2.9 s). For
dense input the blocked variant is the right one.

**Consumer (slot-23 Z3K):** the (Z) phase of an ℓ = 3 orbifold census cell (`Campaign/CampaignOrbifold.cpp`, `OrbifoldSmithTorsion` /
`OrbifoldKernelModule`): the Smith readings of `δ⁰` over `Z/3^19` and of `δ¹`, `δ²` over `Z/9`, and the (U) kernel modules over `Z/9`.
* **On by default.** `L23_MOD3K=0` (read once; any value other than `0` / `1` warns and leaves the route on) is the rollback, and
  `campaign::ScopedOrbifoldMod3k` is the validation override.
* A verbose orbifold campaign prints `EFFECTIVE l23_mod3k=on|off source=default|env|scoped`.
* ℓ = 2 is never routed here.

Single-threaded, except inside `engine::smithpar::ScopedEnable` (slot-24 PAR-2K, §8.2's paragraph): the `k > 5` blocked LU then splits
its trailing `kernels::gemm` into tiles of C (one call per tile; its scratch is thread-local), its panel TRSM into column tiles, runs the
left-looking panel, and `smith_levels` splits its per-level passes; bit-identical at every thread count. `Params` (panel, reduce_block) and
`counters()` are process-wide relaxed atomics.

## 9. Block model (`BlockPool.h`, `BlockMat.h`, `BlockKernels.h`, `BlockElim.h`) — Phase 3

Namespace `engine` (types), `engine::blk` (kernels + elimination). Both element widths. This is
the layer "essentially all relevant matrix/vector operations" run through (plan D3/D5); nothing
in it changes a value — every entry is a fixed function of the refs, and every basis is the D5b
one.

### 9.1 Types

| Type | Meaning |
|---|---|
| `BlockPool<Elt>` | content-addressed, immutable, refcounted store of dense blocks (`DenseBuf`), shared by every `BlockMat` built on it (`std::shared_ptr<BlockPool>`). |
| `BlockRef<Elt>` | one block: `alpha*I + beta*M^(T?)`, `M` = window `(r0, c0)` of pool buffer `pool_id`, `flags & kTransposed`. `kind ∈ {Zero, Scalar, Dense, Affine}` is canonical from `(alpha, beta)` (`normalize()`); `Scalar` needs a square block. Constructors: `zero() identity() scalar(a) dense(id, beta=1, r0, c0, transposed) affine(alpha, beta, id, ...)`. |
| `BlockMat<Elt>` | immutable block matrix: grid `row_off[nbr+1]`, `col_off[nbc+1]` (prefix sums; zero-width blocks allowed) + block-CSR pattern (`row_ptr`, `col_idx`, `refs`). |
| `BlockView<Elt>` | sub-grid / transposed window onto a `BlockMat` — metadata; `at(br, bc)` resolves through the parent, `to_block_mat()` shares the pool blocks. |
| `BlockBuilder<Elt>` | grid declared up front; `set(br, bc, ref)`, `set_identity/set_scalar/set_zero`, `set_dense(br, bc, view | row-major vector, beta=1, alpha=0, layout, detect)`, `accumulate(br, bc, ref)`, `intern_dense(...) → ref`, `compose(ref, alpha, beta)` (= `alpha*I + beta*ref`); `finalize()` moves the pattern into a `BlockMat`. **One call per block, never per cell.** |

### 9.2 Invariants (the contract)

1. **CSR sortedness.** Within a block row the entries are strictly increasing in `bc`; absent =
   structural zero; a `BlockMat` never stores a `Zero` ref. Iteration order (`row_begin/row_end`,
   `col_at(k)`, `ref_at(k)`) is a pure function of the pattern, and `finalize()` yields the same
   CSR for any insertion order. No hash container anywhere (`02 §6.5(5)`): the pool's table is a
   hand-rolled chained vector, the kernels' CSE caches are `std::map`.
2. **Pool immutability + identity.** An interned block is never modified. `intern()` returns the id
   of an existing entry with the same `(rows, cols, entries)` — layout is *not* part of the identity;
   the compare is entrywise (hash = FNV-1a over `std::hash<Elt>`-mixed entries, defect #14). Refcount:
   a fresh entry has count 0 and stays alive until adopted or `sweep()`; containers `acquire/release`;
   a count dropping to 0 frees the entry and recycles the id. `borrow(view)` wraps caller memory
   (no copy, no hash, never deduplicated; caller keeps it alive). **Entry addresses are stable**
   (`std::deque`): a view obtained from the pool survives later interns.
3. **Layout.** Each pool buffer has one physical layout. `pool.view(id, Layout)` returns the
   memoised twin in the other layout (built once, exact permutation). `BlockMat::relayout(L)` is the
   explicit relayout op; `data_view_l(ref, dr, dc, L)` reads through the twin. Preferences: GEMM `A`
   row-major / `C` row-major, multi-vectors column-major (`to_flat/from_flat` default), panels
   row-major (`BlockElim` builds them so).
4. **L1 layout (the multi-vector contract).** A `BlockMat` with `nbc == 1` over row partition
   `(d_s)` and width `k` is a basis of `k` vectors; the flat row index of component `c` of block `s`
   is `row_off[s] + c` = `s·d_V + c` for the uniform partition. `to_flat(layout=ColMajor)` /
   `from_flat(pool, view | DenseBuf&&, row_dims, p, layout, detect)` are **the** conversions (plan
   7.20): `from_flat` interns the whole `dim × k` buffer once and makes every block row a zero-copy
   window; `contiguous_view(out)` / `blk::multivector_view(mv, scratch)` hand the dense view back
   without copying when no block was dropped (`detect` turns all-zero windows into structural zeros
   and square `alpha*I` windows into `Scalar` refs). `from_dense(pool, ..., row_dims, col_dims, ...)`
   is the general re-blocker.
5. **Materialisation.** `materialise(ref, dr, dc, dst, accumulate)` writes `alpha*I + beta*M^(T?)`
   (`Set` or `+=`); `get_entry(r, c)` is `O(log)`; `to_dense(view | layout)`.
6. **Copies.** Copying a `BlockMat` is `O(nnz)` metadata + refcounts; `transposed()` flips every
   ref's flag and re-sorts the CSR by counting sort (`O(nnz)`, no data).

### 9.3 Kernels (`engine::blk`, `BlockKernels.h`)

| Kernel | Semantics |
|---|---|
| `gemm(A, B, out_layout) → BlockMat` | block product on the pattern; `(a1 I + b1 M1)(a2 I + b2 M2)` folds to `a1a2 I + a1b2 M2 + b1a2 M1 + b1b2 M1M2` — scalar×scalar touches no data; each dense `M1·M2` is computed once per distinct pair of data windows (CSE); result blocks are all-zero → structural zero, `alpha*I` → `Scalar`, else interned. |
| `gemm_dense(A, X, Y, op)` | `Y op= A·X` for a dense multi-vector `X` (`A.cols() × k`, rows partitioned by `A.col_off`); CSE on `(window, bc)`. |
| `add(A, B, subtract)`, `scale(A, alpha)` | same grid; refs sharing a window fold without data; `scale` never touches data. |
| `equal(A, B)`, `equal_dense(A, D)` | equality **as mathematical matrices** (any grids, kinds, layouts). |
| `materialise_rows/cols(A, r0, nr, dst)` | dense strips. |
| `gemm_stats()` | `dense_products`, `cse_hits`, `scalar_folds`, `axpys` (per process, `reset_gemm_stats()`). |

### 9.4 Elimination (`engine::blk`, `BlockElim.h`)

| Entry point | Result |
|---|---|
| `rref(A) → BlockRrefResult{rank, pivots, R}` | `R` = RREF without zero rows (`rank × n`, row-major); `pivots` = column rank profile ascending. |
| `rref_block(A, res)` | `R` re-blocked over `A`'s column partition (one block row; `I` / zero blocks detected). |
| `kernel_basis(A[, res], layout=ColMajor)` | `n × (n−rank)` multi-vector, row partition = `A.col_dims()`; D5b vectors, one per free column ascending. |
| `image_basis(A[, res], layout)` | `m × rank` multi-vector, row partition = `A.row_dims()`; original columns at the pivots (`alpha·e_j` for `alpha*I`). |
| `factor_basis(pool, p, row_dims, Rl, G[, params])` and `factor_basis(Rl_mv, G_mv)` | `FactorBasisResult{pivots (into G), independent_relations, basis (a × dim_H multi-vector), projector (dim_H × a row-major)}` = `factor_basis_block`'s greedy leftmost-independent generators modulo the relations and the identity-block rows `independent_relations + i` of `RREF([Rl | G | I])`. The `I` is **structural** (`Scalar` blocks over `row_dims`), `Rl`/`G` are **borrowed** and chunked into `params.chunk`-wide (default 256) column panels. |
| `solve_multi_rhs(pool, p, row_dims, B, Y[, params])` and the multi-vector overload | `ny × nb` row-major, `(t, j)` = coefficient of column `j` of `B` in column `t` of `Y`; throws `std::runtime_error("Vector is not in image of basis (right-hand side i).")` exactly as legacy. |
| `elim_stats()` | panels, panels with pivots, GEMM calls, dense RREFs, blocks gathered (of which scalar). |

**Algorithm** (`BlockElim.h` header, proof above `rref()` in `BlockElim.cpp`): left-looking over
block columns with one fully reduced RREF prefix; per panel one Schur GEMM, one `elim::rref` on the
dense panel, one GEMM for the new rows' trailing part, one reduce-above GEMM. Because the prefix is
reduced, the multipliers are the input's own entries at the pivot columns — gathered structurally
from the `BlockMat`, so structural zeros cost nothing and `I`/`alpha*I` pivot blocks are diagonals
with no data. **Bit-identity:** at the end `R == RREF(A)` (unique) with the pivot columns ascending
and one row per pivot — the legacy true-RREF row order — and every basis is the D5b function of
`(RREF, pivots, original columns)`; `BlockEliminationValidation` and `BlockModelPipelineValidation`
compare all of it entrywise against `engine::elim` and the pre-Phase-2 `BlockFlatMatrix` /
`factor_basis_block` / `change_of_basis_fm`. Panel widths, pivot-row choices and GEMM blocking never
reach the output (the `chunk` sweep in the validation proves it).

Measured on the real matrices (n_sym = 128, ℓ = 134217757, default build, single thread):
`A_C1` 1677×1419 assembly + `rref` + `kernel_basis` **11 ms / 191 allocations** (legacy
`BlockFlatMatrix` assembly + `flat_kernel`: 17 ms / 48 121 allocations); real `W¹ = C¹/Z¹`
(ambient 1419, 516 relations, 1161 generators) `factor_basis` **0.23 s / 785 allocations / 141 MB
cumulative** vs legacy `factor_basis_block` 3.1–3.3 s / 2.33 M allocations / 3.5 GB cumulative;
`H¹ = Z¹/B¹` 0.15 s; `solve_multi_rhs` with 516 RHS in `C¹` 0.08 s. Memory of `factor_basis` is
`R` = `a × (nr + ng + a)` — the transform `U` legacy also materialised; the projector rows must be
reduced against the left-null-space rows of `[Rl | G]`, which is inherent to the D5b contract.

### 9.5 Worked example — `A_Cq` and `C_eq` as a column-major block multi-vector

```cpp
using namespace engine;
auto pool = std::make_shared<BlockPool<uint32_t>>();          // one pool per cohomology run
const uint32_t d = d_V, ell = prime, neg1 = ell - 1;
std::vector<uint32_t> row_dims(live_relations.size(), d), col_dims(N_q, d);
BlockBuilder<uint32_t> b(pool, row_dims, col_dims, ell);
uint32_t ri = 0;
for (const auto* rel : live_relations) {                      // same list as compute_C_eq_bases
    // rho_n(g_r) as a row-major d x d view; interned once per DISTINCT matrix (content-addressed),
    // detect=true turns I into a Scalar ref and 0 into a structural zero
    BlockRef<uint32_t> M = b.intern_dense(rho_view(rel->M), Layout::RowMajor, true);
    if (rel->s_lhs_idx_in_Kq == rel->s_rhs_idx_in_Kq)
        b.set(ri, rel->s_lhs_idx_in_Kq, b.compose(M, 1u, neg1));          // I - M   (one ref, no data)
    else {
        b.set(ri, rel->s_lhs_idx_in_Kq, BlockRef<uint32_t>::identity());  // I
        b.set(ri, rel->s_rhs_idx_in_Kq, b.compose(M, 0u, neg1));          // -M      (shares M's data)
    }
    ++ri;
}
// A_Zq: pencils are b.set(ri, j, b.compose(M, beta_lhs, ell - beta_rhs)) -- also one ref each.
const BlockMat<uint32_t> A = b.finalize();

// C_eq = ker A_Cq as ONE column-major dim_q x dim C_eq object in the L1 layout:
const blk::BlockRrefResult<uint32_t> res = blk::rref(A);
const BlockMat<uint32_t> C_eq = blk::kernel_basis(A, res, Layout::ColMajor);   // row partition = col_dims
DenseBuf<uint32_t> scratch;
DenseView<const uint32_t> C_flat = blk::multivector_view(C_eq, scratch);       // zero-copy dim_q x k, (s*d_V + c, j)

// W^{q-1} = C/Z, H^q = Z/B, P_q:
blk::FactorBasisResult<uint32_t> W = blk::factor_basis(Z_mv, C_mv);     // multi-vector overload
//   W.basis  : ambient x dim_W multi-vector (the chosen C columns)
//   W.pivots : indices into C's columns;  W.projector : dim_W x ambient row-major
// B^q = (D1 (x) I) * W : blk::gemm(D1_block, W.basis, Layout::ColMajor) or blk::gemm_dense(D1_block, W_flat, B_flat)
// Theta_eq solve: blk::solve_multi_rhs(C_eq_mv, ThetaC_mv) -> (dim C_eq x dim C_eq) coordinates, one factorisation.
```
Ready-made assembly for the SL(2,Z) model: `BuildACqBlockMat(pool, lssc, q)` in
`src/hacs/BlockModelPipelineValidation.h`.

## 10. Engine (tape) API — Phase 4 (`src/Engine/`, namespace `engine::tape`)

Files: `Program.h` (IR), `Recorder.h` (recording), `Passes.h/.cpp` (passes 1–6, wavefronts,
arena), `ThreadPool.h/.cpp`, `Executor.h/.cpp`, `ProgramCache.h/.cpp`, `TapeElim.h/.cpp`
(elimination driver), `EngineCostModel.h` (the one table of measured constants and the
per-kernel thread counts). Static library `MatrixEngine` on top of `ModArithmetic`; it consumes
§1–§9 and modifies nothing below it. Plan D4 is binding; this section is the consumer contract.

### 10.1 The model

A heavy computation is **recorded** as a `Program<Elt>`: a straight-line list of `Node`s
`(op, in-views a/b, out-view c, scalars, k_bound, cost, device)` over program-owned buffers,
then **compiled** by passes 1–6 and **executed** by wavefronts on a persistent thread pool.
Every node's output is a mathematically unique exact object (integer arithmetic, canonical at
node boundaries, pivots a function of contents, no atomics on results), so *any* thread count,
batching, tiling or device assignment yields bit-identical buffers — asserted by
`EngineDeterminismCoreValidation` and `EngineCoreScalingValidation`, never assumed.

Op set (`Program.h` has the exact semantics):

| Op | Meaning | Thread class |
|---|---|---|
| `GEMM` | `c gop= alpha·a·b`, `gop ∈ {Set, Add, Sub}` | compute (32) |
| `BATCHED_GEMM` | pass 5's group of same-shape independent GEMMs, instances in `program.instances` | compute (32) |
| `SCHUR_UPDATE` | `c -= a·b` (the elimination's trailing update; batched separately from GEMM) | compute (32) |
| `TRSM` | `c = a⁻¹·b` with the pivot factor already inverted by the host driver; `Identity` → copy, `Scalar` → `c = alpha·b` — the "no-op for I pivots" of D4 | compute (32) |
| `AXPY` | `c = c ± alpha·a`; `a.buf == kIdentityBuf` adds `alpha·I` to the diagonal | memory (16) |
| `SCAL` | `c = alpha·c`; `alpha == 0` is the zero fill | memory (16) |
| `TRANSPOSE` | physical relayout `c(i,j) = a(i,j)` into the other layout (16×16 tiles); a *logical* transpose is `ViewRef::transposed` and free | memory (16) |
| `CANONICALIZE` | `Lazy2p → Canonical` in place | memory (16) |
| `GATHER` | `c(i,t) [+]= alpha·[row_off+i == cols[t]] + beta·a(i, cols[t])`, `cols` an index list or all columns: block → flat, `BlockRef` materialisation, column gathers of the elimination. With a **row list** (`Recorder::gather_rows` / `gather_block_row_list`, Phase 4 closure) `c(i,t)` reads `a(rows[i], cols[t])` and the identity test is `row_off + rows[i] == cols[t]`: the elimination's pivot-row gathers are one node per block; device-eligible since slot-18 WP-19a (`cuda::gather`'s row-list form, §12.8; CPU-only before) | memory (16) |
| `SCATTER` | `c = a`, any layouts: flat → block windows, plain copies | memory (16) |
| `PANEL_RREF` | in-place `elim::rref(c)`; result slot ← `{rank, pivots}` | serial |
| `PIVOT_SEARCH` | on the panel *before* its RREF with the slot's `Q`: pivot rows = row rank profile (first non-zero rows in original order) → slot; `c[0:k,0:k] ← a[prows, Q]` | serial |
| `COMPARE` | slot ← `(a == b)` as logical matrices, any layouts; per-tile flags reduced on the host | memory (16) |

Thread counts are the per-kernel table of `EngineCostModel.h` (`04 §7.5`: 16 = one per core for
memory-bound kernels, 32 for compute-bound), clamped to the pool size.

### 10.2 Buffers and views (handle semantics)

`BufId` is an index into `program.bufs`. Three kinds:

* **Input** — an external, *read-only* slot bound at run time. Shape **and layout** are fixed at
  record time (`view_shape()` derives them from the view's unit stride); a binding with a
  different shape or layout is an `ENGINE_CHECK` failure. A node that writes an Input fails at
  `finish()`. `content_key` is the CSE identity: two inputs with equal keys are promised to hold
  equal contents whenever they are bound together (`0` = unique; bits 61–63 reserved).
  `Recorder::pool_input(pool, id)` memoises one Input per distinct `(pool, id)` with an exact
  pool-class key — the pool deduplicates, so equal ids ⇔ equal contents.
* **Output** — an external slot the program writes (and may read back: accumulation into an
  output is fine). Same binding rules.
* **Temp** — program-owned, arena-backed, uninitialised at program start (write it with `Set`
  semantics or `zero()` first), lifetime `[first_level, last_level]` in wavefronts. Layout is
  chosen by pass 3 unless pinned.

`ViewRef{buf, row0, rows, col0, cols, transposed}` is a window in *buffer* coordinates plus the
free logical transpose; `sub()` takes logical coordinates; `lrows()/lcols()` are the logical
shape. `Recorder::whole(b)`, `view(b, r0, nr, c0, nc, t)`, `ref_view(pool, ref, dr, dc)` (the
logical `dr × dc` window of a `BlockRef`, transposed flag resolved).

`Bindings<Elt>{inputs, outputs, ctx?}` — the views bound to the slots, in slot order. The
Recorder fills a default `Bindings` from the views passed to `input(view)` / `output(view)`;
a replay supplies new views of identical shape/layout (and may set `ctx` to another prime of
the **same regime** — `k_bound ≤ k_max` is checked).

### 10.3 Recording

```cpp
using namespace engine::tape;
Recorder<uint32_t> rec(ctx);                          // ModCtx of the modulus
BufId X = rec.input(cview(Xbuf));                      // dim x w, column-major (L1 multi-vector)
BufId Y = rec.output(view(Ybuf));
BufId T = rec.temp(rows, cols /*, preferred layout, pin */);
rec.gemm(a, b, c, GemmOp::Add, alpha);                 // plain dense product
rec.gemm_ref(ra, Ma, rb, Mb, c, dr, dk, dc, gop);      // block product WITH KINDS: (a1 I + b1 M1)(a2 I + b2 M2)
rec.record_gemm_dense(A, rec.whole(X), rec.whole(Y));  // Y = A.X for a BlockMat A (blk::gemm_dense on tape)
rec.record_gemm_block(A, B, C);                        // C = A.B, dense output (blk::gemm on tape)
rec.record_gather_dense(A, D);                         // block -> flat
rec.gather_block_rows(pool, ref, dr, dc, lo, hi, c, accumulate, index_list);
rec.axpy / axpy_identity / scal / zero / scatter / transpose / canonicalize;
uint32_t q = rec.panel_rref(S);  uint32_t pr = rec.pivot_search(S0, q, B);  uint32_t eq = rec.compare(a, b);
Program<uint32_t> prog = rec.finish();                 // structural FNV-64 hash computed here
```

Recording does no arithmetic and is `O(#ops)`. `gemm_ref` stays **one node** until pass 1 folds
it, so a consumer emits block structure verbatim and lets the planner cash it in.

### 10.4 Passes (`compile(prog, opts)`, idempotent, a pure function of `(program, options)`)

1. **Fold** on kinds: `(a1 I + b1 M1)(a2 I + b2 M2) → a1a2·I (AXPY-I) + a1b2·M2 (AXPY) + b1a2·M1
   (AXPY) + b1b2·M1M2 (GEMM into a temp + AXPY)`. `opts.fold = false` is the validation's
   unfolded arm (materialise, multiply).
2. **CSE** on `(op, operand content keys + windows, scalars, shape)` for single-write whole-buffer
   temps — the `M·X` memo of `04 §4.3`: a product of the same pool blocks is computed once and
   AXPY'd into every consumer. Temp identities are *exact* (CSE-class indices, never hashes).
   Then **fusion**: an un-shared `T = A·B; C ±= s·T` becomes `C gop= s·A·B` with no temp.
3. **Layout** (D3 policy): votes weighted by node cost — GEMM/SCHUR/TRSM want `B` row-contiguous
   and `C` row-major, `PANEL_RREF` row-major panels; ties → the producer's recorded preference →
   `RowMajor`. Each remaining mismatch on a GEMM `B` operand gets **one** physical `TRANSPOSE`
   into a twin, memoised per `(buffer, layout)`, hoisted to the program start (Inputs) or right
   after the producer (single-write Temps); buffers modified later are left to the kernel's
   per-call pack (correct, documented in §6). A column-major multi-vector input feeding 2000
   GEMMs therefore costs one relayout, not 2000 packs.
4. **k-blocking**: `k_bound = min(k, ctx.k_max)` recorded per node; `opts.ksplit`
   (`ENGINE_KSPLIT=1`) splits `k > k_max` into sequential chunks. Off by default on the CPU because
   the dense kernels block internally at `k_max` (§5) — the pass exists for backends that do not.
5. **Batching**: within one wavefront, key `(op, gop, alpha, m, n, k, layout classes of a/b/c,
   elt width, k_bound, device)`; instances sorted by `(output buffer, row0, col0)`; a group of size
   1 stays a plain node. Members were mutually independent, so a batch keeps their level.
6. **Device** (`ENGINE_DEVICE=cpu|gpu|hybrid`, default **cpu**): `cpu` records CPU everywhere;
   `gpu` sends every device-eligible node (everything but `PANEL_RREF`, `PIVOT_SEARCH`,
   `COMPARE`) to the GPU; `hybrid` runs the static cost model of `EngineCostModel.h` per *group*
   of would-be batch members of one wavefront (the batching key contains the device, so the
   decision precedes pass 5 and follows the level analysis). Since Phase 5 Stage B the executor
   really runs GPU nodes on the CUDA layer — §12 is the contract; without a device they run on
   the CPU, and no value ever depends on it.
7. **Residency** (§12.2): whole-buffer residency simulated level by level; `H2D` / `D2H` transfer
   nodes inserted into the program, uploads hoisted past the last host write, outputs downloaded
   in an epilogue level, mixed-device writers of one buffer in one level forced to the CPU.

Then dependency analysis (RAW/WAR/WAW on overlapping windows, result-slot reads), wavefront
levels, Temp lifetimes in levels, and the **arena**: first-fit over `[first_level, last_level]`
intervals, 64-byte-aligned rows (`ld` rounded to 16 elements); unreferenced temps get no storage.

Switches (`PassOptions::from_env()`): `ENGINE_BATCH`, `ENGINE_KSPLIT`, `ENGINE_ARENA` (`0` = every
temp owns its storage), `ENGINE_CSE`, `ENGINE_FUSE`, `ENGINE_LAYOUT`, `ENGINE_DEVICE`; executor:
`ENGINE_THREADS` (default = hardware concurrency, ≤ 32), tuning knobs `ENGINE_TILE_MULT` (tiles
per thread, 4) and `ENGINE_TILE_COLS` (output column panel, 256), `ENGINE_SPIN` (spin budget of
idle workers and waiting callers before they block, 20000 `pause`s), `ENGINE_TRACE_LEVELS`;
diagnostics (Phase 4 closure, `EngineTrace.h`): `ENGINE_TRACE_NODES=1` accumulates per-op
node / task / busy totals, per-class region wall vs busy, executor and compile wall, pool region
wall / idle / mutex wait and the dense PLU's phase split (`--bench` prints them per repeat),
`ENGINE_TRACE_RSS=1` also traces the Hecke assembly phases, every `tape::rref` and every
unit-row solve. **None of them can change a value.**

### 10.5 Execution and replay

```cpp
Executor<uint32_t> ex;                    // ThreadPool::global(), or Executor(&my_pool)
RunResult r = ex.run(prog, rec.bindings(), PassOptions::from_env());   // compiles on first use
r.slots[q].rank, r.slots[q].pivots, r.slots[eq].equal;                 // result slots of this run
ex.run(cached_program, other_bindings);                                 // replay (const Program&)
```

Each wavefront expands its nodes into tasks that write **disjoint memory**: a GEMM is split over
*output tiles* (never over `k` — no reductions), a batch over its instances (tiled further when
there are fewer instances than threads), memory ops over row ranges; `PANEL_RREF` /
`PIVOT_SEARCH` are single tasks. Compute-class and memory-class tasks of a level run as two pool
regions with their own thread counts. A scaled product or a tile whose rows are not contiguous
(column-major output) is computed into a row-major thread-local scratch and combined with one
exact copy/scale/AXPY. An `Executor` owns **one arena**, grown to the largest program it has run
and reused across replays (`EngineArenaValidation` checks the base pointer is stable); use one
executor per concurrently running pipeline — they share the pool. **Regions of different callers
run concurrently** (Phase 4 closure, `ThreadPool.h`): an active region is registered, every
worker claims tasks from the oldest registered region that still has tasks and room under its
thread class, a worker finishing one region joins the next, idle workers rescan only when the
registry changes (an epoch counter), callers wake idle workers first and spin with `pause` before
blocking. Every task still runs exactly once and which worker runs it is unobservable, so the
concurrency cannot change a value. Exceptions thrown by a task are rethrown on the caller after
the region.

### 10.6 Shape signature and the program cache

```cpp
ShapeKey key; key.row_dims = A.row_dims(); key.col_dims = A.col_dims();
key.pattern_hash = pattern_hash_of(A);    // grid + CSR + kinds + coefficients + flags + pool ids in first-occurrence order
key.d_V = d; key.regime = uint8_t(ctx.regime); key.elt_width = sizeof(Elt); key.tag = "theta"; key.extra = q;
auto& cache = ProgramCache<uint32_t>::global();
if (auto cached = cache.find(key)) ex.run(*cached, bindings);          // hit: replay
else { Recorder rec(ctx); ...; Program p = rec.finish(); compile(p, opts); ex.run(p, rec.bindings()); cache.insert(key, std::move(p)); }
```

**Contract.** A cached program is valid for every binding of identical shapes *and identical
content-key pattern*, which is what the key encodes. Per the D4 caveat the consumer **must** put
the hash of the *actual* pattern it records against into `pattern_hash` (for Θ: the τ-table
pattern of the concrete `(n_sym, p, r, q, ℓ)`, computed before lookup, because the mod-`ℓ`
fallback of the triangle-fill solve can change the support); a miss falls back to recording,
never to replaying a stale program. `EngineTapeValidation` asserts replay == fresh recording on
new data and that a changed `pattern_hash` misses. The cache is a `std::map` behind a mutex
(`02 §6.5(5)`: no hash containers on result paths).

**The options fingerprint** (`pass_options_fingerprint()`, `EngineRuntime.h`; hecke-extensions
slot 9, WP-6; the ninth bit slot 10, WP-8, decision 10 (A)). `run_cached()` XORs a **9-bit**
fingerprint of the compile options into `key.extra << 55`, so programs compiled under different
options never share a slot: bits 1–64 are `fold`, `cse`, `fuse`, `layout`, `ksplit`, `batch`,
`arena`; bit 128 is set **iff `effective_policy() != Cpu`** — the policy pass 6 actually compiles
under. Before slot 9 bit 128 read the Phase 4 pair (`device_model && device_policy == GPU`), so an
options value that selects the device through `PassOptions::policy` alone
(`HeckeExtDeterminismValidation`'s `gpu` arms) was keyed as a CPU program and replayed CPU-compiled
programs (slot-8 WP-6b finding 3). Every `from_env()`, `ScopedThreadDevice` and console
`set device` mapping sets the pair and the policy together, so their fingerprints did not move
(`EngineElimPassOptionsValidation` (B): 512 / 512). **Since slot 10, bit 256 is set iff
`effective_policy() == Gpu`** specifically, so `hybrid` (bit 128 only) and `gpu` (bits 128 and 256)
key distinct cache slots — the shift moved from 56 to 55 for it (56 would drop bit 256 off the top
of a 64-bit `key.extra`; every consumer's own `extra` value is a tiny integer, so bits 0–54 stay
free for it). `EngineFingerprintDeviceBitValidation` (slot 10, WP-8) checks `cpu`, `hybrid`, `gpu`
give three distinct fingerprints and that programs compiled under each land in distinct cache
slots. Value-free (§11): a device result is the same bytes as the CPU result by construction; the
fingerprint only decides which cache slot a compiled program occupies.

### 10.7 The elimination driver (`TapeElim.h`)

```cpp
blk::BlockRrefResult<Elt> res = tape::rref(ex, A, &stats);   // == blk::rref(A) entrywise
tape::kernel_basis(ex, A) / image_basis(ex, A)               // D5b bases from that RREF
tape::factor_basis(ex, pool, p, row_dims, Rl, G, chunk)      // == blk::factor_basis
tape::solve_multi_rhs(ex, pool, p, row_dims, B, Y, chunk)    // == blk::solve_multi_rhs, same throw
```

Rank is data-dependent, so a host driver walks the block columns and records two programs per
panel: `P1` = gathers of the candidate rows and of the pivot columns → `SCHUR_UPDATE` (tiled) →
copy → `PANEL_RREF` → `PIVOT_SEARCH`; the host reads `Q`, the pivot rows and the pivot block,
classifies it (`I` ⇒ no TRSM, `αI` ⇒ scaling, general ⇒ inverse on the host — structural
shortcuts of D4); `P2` = the new rows' panel part (one row-list `GATHER` per block), gathers of
the trailing part, `SCHUR_UPDATE`, `TRSM`, and the reduce-above `SCHUR_UPDATE`. The RREF is
unique, so the result is identical to `blk::rref` (`EngineTapeEliminationValidation`, entrywise
on `BlockEliminationValidation`'s grids, both widths, 8489 checks).

`PIVOT_SEARCH` computes the row rank profile of `S[:, Q]` as the column rank profile of its
transpose from the PLU factor alone (`elim::column_rank_profile`: the pivots are fixed by
`factor()`, no back-substitution). Inside `PANEL_RREF` (`DenseElim.cpp`, Phase 4 closure) the
host-serial dense PLU splits over the pool hook whatever is a set of independent rows or columns
of exact values: the column tiles of both triangular solves (`trailing_update`'s TRSM and
`to_rref`'s `U^{-1}` solve), the row tiles of the Crout base panel's Schur step (pivot scan fused,
multiplier scaling deferred into the next column's pass), and the trailing GEMM (which reads
`L21` in place when the panel's pivot columns are contiguous). A nested hook call from inside a
hook task runs serially (`g_in_hook_task`). `Params::base` (`ENGINE_ELIM_BASE`) is the width at
which the bisection stops and the scalar Crout base takes over; below it every column step is a
vectorised rank-`k` update.

**`solve_multi_rhs` with unit rows.** When every column of `B` has a unit row (a D5b kernel basis
always has: the identity sits at the free variables) the coordinates of any `y ∈ span(B)` are
forced, `y = Bx ⇒ x[c] = y[u_c]`, so `X = Y[u, :]` is the unique solution whenever one exists and
membership is decided by the exact residual: `(BX)^T = X^T B^T` as one row-major tape GEMM
`COMPARE`d against `Y^T`, with the elimination route's throw on a miss. The same values as the
elimination of `[B | Y]` (its RREF is `[I | X]` when `B` has full column rank) for one product
instead of an `a × (nb + ny)` elimination. `certify = false` skips that product for a caller that
certifies the result itself by the same residual — `HeckeOperatorOnK`'s A-N3, recorded as
`W^T -= coords · C_eq^T` (the row-major `coords` as stored, free logical transposes elsewhere: no
physical relayout, row-contiguous output). Without unit rows the elimination route runs and
certifies by construction.

### 10.8 Worked example — Θ-style assembly and the `Θ_eq` solve

```cpp
using namespace engine; using namespace engine::tape; using kernels::GemmOp;
Executor<uint32_t> ex;
// Theta_flat (dim_q x dim_q, row-major) = sum over (sigma, i) of rho_n(s_i^{-1}) . X[sigma_k]:
// record every term as a kind-GEMM; CSE computes each distinct rho.X once, batching runs the
// distinct products of one wavefront as one BATCHED_GEMM, fusion folds un-shared ones into
// their AXPY.
Recorder<uint32_t> rec(ctx);
BufId bX = rec.input(cview(Xflat));                  // column-major multi-vector: pass 3 relayouts it ONCE
BufId bT = rec.output(view(Theta_flat));
rec.zero(rec.whole(bT));
for (const Term& t : tau_terms)                      // (sigma, i): coefficient c, rho id, source block sigma_k
    rec.gemm_ref(Elt(0), t.coeff, rec.ref_view(pool, t.rho_ref, d, d),
                 Elt(0), Elt(1), rec.view(bX, t.sigma_k * d, d, 0, d),
                 rec.view(bT, t.sigma * d, d, 0, d), d, d, d, GemmOp::Add);
Program<uint32_t> theta = rec.finish();
compile(theta, PassOptions::from_env());
ex.run(theta, rec.bindings());
ProgramCache<uint32_t>::global().insert(key_for(n_sym, p, r, q, ell, tau_pattern_hash), std::move(theta));
// Theta_eq: coordinates of the columns of Theta_flat . C_eq in the basis C_eq -- one factorisation
DenseBuf<uint32_t> coords = tape::solve_multi_rhs(ex, pool, ell, row_dims, cview(C_eq_flat), cview(ThetaC_flat));
```

Measured on the core alone (`EngineCoreScalingValidation`, default build, `p = 134217757`,
best of 3; the same program hashes identically at every thread count):

| threads | Θ-like 2000 × 129³, row-major out | Θ-like, column-major out | dense 2048³ |
|---|---|---|---|
| 1 | 0.195 s, 22.0 Gmodmul/s | 0.324 s, 13.3 | 0.301 s, 28.5 |
| 4 | 0.053 s, 81 (3.7×) | 0.086 s, 50 (3.8×) | 0.075 s, 115 (4.0×) |
| 16 | 0.018 s, 244 (**11.1×**) | 0.030 s, 145 (10.9×) | 0.026 s, 335 (**11.7×**) |
| 32 | 0.017 s, 246 (11.2×) | 0.026 s, 164 (12.4×) | 0.022 s, 390 (13.7×) |

The dense figure is 76 % of the 16-thread ceiling implied by the single-thread kernel
(16 × 27.5 = 440 G) and 65 % of doc 04's 512 G roofline; the remaining gap is L3 traffic of the
B panels and the all-core clock, not the executor. The Θ-like program's single-thread rate equals
the raw kernel loop on the same data (23 G), i.e. the tape adds no per-node cost at this block
size; its column-major-output variant pays the scratch epilogue (13 G).

### 10.9 The consumers on the tape (Phase 4 Stage B, `EngineRuntime.h`)

`src/Engine/EngineRuntime.h` is the consumer-side glue: `consumer_pool()` (the global
pool unless a validation installed a `ScopedConsumerPool`), `consumer_pass_options()`
(`from_env()` unless a `ScopedPassOptions` override is active), `run_cached(ex, key, program,
bindings)` — the **record-always / compile-on-miss** protocol (recording is `O(#ops)`; a hit
replays the cached compiled program on the fresh bindings and asserts the structural hashes agree,
a miss compiles and inserts; the key carries a pass-options fingerprint and the prime, since pass 1
folds coefficient products under the recording modulus) — and `run_concurrently(tasks)`, the
host-side fan-out (one `std::thread` per task, never a pool worker; serial when the pool has one
worker or `ENGINE_HOST_THREADS=0`; the first failing task's exception in task order is rethrown).

**Thread context.** The scoped consumer options (`EquivariantCohomologyOptions::Scoped*`,
`HeckeOperatorOptions::Scoped*`, `engine::elim::ScopedLegacy`) are `thread_local`, so pipelines
running concurrently cannot see each other's settings; `run_concurrently` snapshots the launching
thread's context through the propagators registered with `register_thread_context()` and installs
it on every task thread before the task runs (a TU owning thread-local options registers one
capture at static-init time).

**What records on the tape now:**

| consumer | programs (cache tag) | notes |
|---|---|---|
| `EquivariantCohomologyOnK::compute_cohomology_block` | `tape::kernel_basis` per `A_Cq` / `A_Zq` (per-panel programs), `cohom.Zeq` / `cohom.DC` (`gemm_dense`, replayed by shape), `tape::image_basis` for `B^q = image(D·C^{q-1})`, `tape::factor_basis` for `H^q` (projector) | step 1 (`A_C`, `C_eq`, `A_Z`, `Z_eq`) fans out over `q` (axis (c)); steps 2/3 read only step-1 objects, so the per-degree chains `B^q → H^q, P_q` fan out too. **`B^q` is the D5b image basis of `D·C^{q-1}`** — the same columns as `(D⊗I)·(C^{q-1}/Z^{q-1})` because `ker D` on `C^{q-1}_eq` is `Z^{q-1}_eq` — from a `dim_q × dim C^{q-1}` elimination instead of `dim_{q-1} × (dim Z + dim C)`, without `Z^{q-1}`. |
| `HeckeOperatorOnK::compute_block` | `hecke.theta` (per `q`: the fused assembly as kind-GEMMs — CSE computes each distinct `ω_i·ρ_k` once, batching runs a wavefront's products as one `BATCHED_GEMM` (axis (a)), block rows write disjoint outputs (axis (b)) — plus Θ5/6/7/8a/10 with `COMPARE` slots), `hecke.route2` (one per block row: route 2 as arena temps `COMPARE`d against the fused blocks over the per-degree memo `rho2`; **since S23 P4R only under the rollback `HECKE_RHO2_STREAM=0`** — by default route 2 is streamed key by key (one d × d accumulator per route-2 block; the distinct keys' `ρ_n` evaluated in batches of 64, each once, and added with delayed reduction; no per-degree memo `rho2`), on the host), `hecke.resid.chunk` (the A-N3 residual, since S23 PERF-P4a one program per 512 right-hand sides: `P = coords[t0..]·C_eqᵀ` `COMPARE`d against `W[:, t0..]ᵀ`; no `dim C_eq × dim` zero temp) (A-N3), `tape::solve_multi_rhs` for `Θ_eq` (axis (h)) | keyed by the τ-pattern hash of the concrete `(n, p, r, q, ℓ)` (D4 caveat: a changed support misses); the fused output blocks are owned by the operator and borrowed by its pool as `Theta_block(q)`. |
| `HeckeOperatorFamily::build_pipelines` / `prepare(p, r)` | — | axis (d): coset/τ construction serial in key order under one process-wide mutex (hazards 1–3, `PredicateStats` exact), `compute()` per pipeline concurrent, insertion in key order; `pipeline_cache_` a `std::map` behind `cache_mutex_`. |

**Not on the tape, and why:** τ construction (never parallelised, by plan); the `BlockBuilder`
assembly of `A_Cq` / `A_Zq` / `Θ`'s pattern (metadata); `HeckeModuleDecomposition` / `FpPoly`
(`d_q ≤ ~20`); `HeckeCrossChecks` and the legacy arms (kept verbatim as the reference).

**Host-serial nodes.** `PANEL_RREF` / `PIVOT_SEARCH` run on the calling thread outside a pool
region with `elim::ScopedParallelHook` bound to the pool, so the dense PLU's trailing updates,
triangular solves and Crout Schur steps (§10.7) use the workers; what stays serial is the
per-column pivot bookkeeping of the base panel and the host recording of the per-panel programs.
Since the Phase 4 closure the Hecke degrees fan out like the cohomology's (`run_concurrently`,
one executor per degree, the route-2 memo per degree, counters merged in `q` order; sequential in
the memory-lean mode), and `factor_basis` stage 2 runs its `Mact·Winv` product over the hook. The
remaining serial breakdown is in plan §13.7.

**Host eliminations off the tape** (hecke-extensions slot 23, PERF-P2). `engine::tape::ScopedHostElimHook`
(`EngineRuntime.h`) installs the same hook, on the calling thread, bound to `consumer_pool()` (looked
up when the hook runs, so a `ScopedConsumerPool` is honoured; the global pool object is untouched),
with the executor's compute-bound thread cap. It is placed at the entry of
`EquivariantMappingCone::compute`, `ConeCertificate::verify`, `HeckeInvolution::compute` and
`InvolutionCertificate::verify`, so their direct `engine::elim` / `FlatMatrix::rank` / `image` calls
(the cone's `DC` RREF and kernel, `rank_of_view`, ι's carry checks and `build_split` images) split
their trailing updates over the pool. It installs nothing on a one-worker pool, when the thread
already carries one, when **any** elimination hook is already in effect on the thread (precedence:
an executor's or a caller's hook is never replaced), or under `ENGINE_HOST_ELIM_HOOK=0` (rollback); a `register_thread_context`
capture gives `run_concurrently()`'s task threads of a hooked caller their own. Value-free for the
reason above (disjoint row / column tiles of exact residues; unique pivots and RREF);
`HostElimThreadDeterminismValidation` gates it on pools of 1 / 4 / 16 workers.

**`factor_basis` without the identity augmentation** (`TapeElim.h`, memory profile 05 §7 item 1):
stage 1 eliminates `[Rl | G]` alone (pivots, `independent_relations`, basis unchanged — appending
columns never changes earlier pivot columns); stage 2, only with `want_projector`, rebuilds the
projector rows of the unique RREF transform `U` of `[M | I]` from `U_top0[:, pivot_rows] =
M[pivot_rows, pivots]^{-1}` (the rows of `R` are combinations of the pivot rows alone) and
`U_bot = RREF(I[act,:] − M[act, pivots]·U_top0)`, the unique RREF basis of the left null space,
eliminated **in place**; `U_top = U_top0 − U_top0[:, pivcols(U_bot)]·U_bot`. Uniqueness of the RREF
makes it entrywise identical to `blk::factor_basis` (`EngineTapeEliminationValidation`, 8489
checks); `BlockRrefResult::pivot_rows` (both drivers) carries the row rank profile in pivot order.

**Memory-lean mode** (`EquivariantCohomologyOptions::retain_intermediates() == false`,
`HeckeOperatorOptions::theta_eq_certification_enabled() == false`; `--bench ... --lean`): step 1
and the chains run sequentially at the consumer level (pool parallelism kept), chains in
descending `q`, every object released at its last reader (`A_C`/`A_Z` and `C^{top}` after step 1,
`C^{q-1}` after `D·C^{q-1}`, `Z^q`/`B^q` after `H^q`); `C_eq_dimension(q)` survives, the dropped
accessors throw. `HeckeOperatorOnK` reads `C_eq` only for the `Θ_eq` certification and records
`"theta_eq_certified": false` in the JSON when it is off. Values do not depend on the mode.

**The CUDA backend is created lazily** (`Executor::device_usable()` or the first run of a program
with device nodes): the constructor's probe created a CUDA context on every CPU-only run
(memory profile 05 §0, 110 MB).

**The thread-local device override** (hecke-extensions slot 8, WP-6b; `EngineRuntime.h`). The
device policy is otherwise process-wide — `PassOptions::from_env()` reads `ENGINE_DEVICE`, and the
`ScopedPassOptions` override is one mutex-guarded global. `ScopedThreadDevice(d, tally)` with
`d ∈ {Cpu, Gpu}` replaces the **three device fields** (`device_policy`, `device_model`, `policy`,
exactly `from_env()`'s `cpu` / `gpu` mappings) of whatever the installing thread would otherwise
use, and nothing else; it is captured by `register_thread_context()`, so `run_concurrently()`'s
task threads inherit it, and a plain `std::thread` does not. `consumer_pass_options()` applies it
over the process-wide override or `from_env()`; the elimination driver (`TapeElim.cpp`, which never
honoured the process-wide override before slot 9) compiles under `elim_pass_options()`, which
**since hecke-extensions slot 9 (WP-6) is `consumer_pass_options()`**: the process-wide
`ScopedPassOptions` override when one is installed, `from_env()` otherwise, with the thread-local
override applied (slot-8 WP-6b finding 2: until then a `ScopedPassOptions` caller — the determinism
validations' `batch` / device arms, the console's `set batch` / `set device` — never reached the
per-panel programs `P1` / `P2` of §10.7). The per-panel programs are not cached, so no key moves,
and compile is pure, so no value can; `TapeElimStats` counts what they were compiled under
(`batch_off_programs`, `batched_programs`, `gpu_programs`, `gpu_nodes`), which is how
`EngineElimPassOptionsValidation` and the three determinism validations show the override arrives.
**With no override installed both are exactly what they were.** `hybrid` is not offered here: this
override exists for the campaign ext pass's optional device worker, a CPU/GPU choice only — since
slot-10 WP-8 `pass_options_fingerprint()` gives `hybrid` and `gpu` distinct bits (§10.6), so this is
no longer a cache-slot argument, just this override's scope; CPU- and GPU-compiled programs of
concurrent threads never share a cache slot either way. The optional `DeviceTally` (captured with it) counts every program the thread runs
through `run_cached()` or the elimination driver and the GPU-assigned nodes in it — the evidence
that a `Cpu`-overridden thread ran none. No value can move (§11: a device result is the CPU
result's bytes). The one consumer is the campaign ext pass's device worker (`ext_gpu`,
`docs/matrix-engine/campaign.md` §14); `EngineTapeConsumerValidation` §4 tests it.

## 11. CUDA backend (`src/Cuda/`, `engine::cuda`) — Phase 5

Files: `CudaCtx.h/.cpp` (runtime context, pinned pool, arena, transfers, graph capture),
`DeviceBuf.h` (device mirror + residency, `DeviceView`), `CudaKernels.h` / `cuda_kernels.cu` (the
kernels), `CudaValidation.cpp` (the gates), `CudaBench.cpp` (`--bench micro --backend gpu`, `--bench
micro cuda`). Static library `EngineCuda` (depends on `ModArithmetic` + `CUDA::cudart`; nothing
depends on it except `cohomology_app`). **Every public header is CUDA-header-free** — the tape can
include them from a plain C++ TU. Everything is exact integer arithmetic; **no floating point, no
atomics on results**; a device result is the same bytes as the CPU result by construction, and
`CudaBitIdentityValidation` asserts it by `memcmp` (10,844 checks).

### 11.1 Availability and policy

| Call | Meaning |
|---|---|
| `available()` | physical: a device exists and the runtime works (probed once, never throws). `ENGINE_CUDA_DISABLE=1` forces `false`; `ENGINE_CUDA_DEVICE=<i>` selects the device. |
| `requested_device()` / `device_enabled()` | policy: `ENGINE_DEVICE=cpu|gpu` (default **cpu**) — `device_enabled() == available() && gpu`. The tape's device assignment (pass 6) asks this; the GPU is never enabled by default (plan Phase 5). |
| `device_info()` | name, cc, SMs, memory, driver/runtime versions. |
| `CudaCtx::get()` | the process-wide context (WSL2 has no MPS: one process owns the GPU); throws `CudaError` when `!available()`. |

Every validation of the layer prints `[SKIP] ...` and returns 0 when `!available()`, so `--all` is
green on a machine without a GPU.

### 11.2 Types and the residency contract

`CudaCtx` — 3 non-blocking streams (`stream(i)`, opaque `Stream{void*}`), events (`create_event`,
`record`, `wait`, `sync`), the **pinned pool** (8 MB `cudaHostAlloc` chunks, lazily up to the
128 MB cap — `ENGINE_CUDA_PINNED_MB` / `_CHUNK_MB`; a transfer larger than the pool *recycles*
chunks through their events, so peak pinned bytes never exceed the cap: `pinned_stats()`), the
**device arena** (`dalloc`/`dfree`: exact-size free lists over 64 MB `cudaMalloc` slabs,
`arena_stats()`; since slot-14 WP-3 each slab carries its live bytes, and a slab with none is
returned to the driver by `release_cached()` / `release_device_caches()`, by a **soft trim** before
a new slab once the idle bytes grew by `ENGINE_CUDA_ARENA_SLACK_MB` (default a quarter of the
device) since the last trim — skipped while any thread holds an open capture — and by one trim +
retry on a `cudaMalloc` OOM; when a trim frees the current bump slab the bump moves to the
surviving slab with the largest free tail, which a following `dalloc` uses before cutting a new
slab (slot-14 WP-16); `release_cached()` refuses while any thread holds an open capture; no live
block is ever touched, so no value can move), the staged transfers `upload(host, dev, bytes, s)` / `download(...)` (pageable
host memory ↔ pinned chunk ↔ device; `download` returns with the bytes in place), the direct copies
`copy_d2d` / `memset_d`, and `transfer_stats()` whose `pageable_copies` **must stay 0** (it counts
the fallback taken only when no pinned chunk can be obtained at all, plus the explicit
`*_pageable` bench calls).

`DeviceBuf<Elt>` mirrors one `DenseBuf<Elt>` **with the same `rows/cols/ld/layout/invariant/
structure`** (padding included, never read or written by a kernel) and tracks residency with two
version counters:

| state | meaning | transition |
|---|---|---|
| `Host` (host_version > device_version) | host is newer | `upload(s)` copies → `Both` |
| `Device` | device is newer | `download(s)` copies → `Both` |
| `Both` | equal | `upload`/`download` are no-ops (return `false`) |
| `None` | nothing written yet | — |

The tape bumps the counters: `mark_host_written()` after a CPU node wrote the host buffer,
`mark_device_written()` after a GPU node wrote the device buffer. `upload(s, force)` /
`download(s, force)` copy the **whole footprint as bytes**. Device-only buffers
(`DeviceBuf(rows, cols, layout, ld, ...)`) have no host twin; `upload_from` / `download_to` are their
raw copies. `DeviceView<Elt>` = `{base, rs, cs, rows, cols, invariant, structure}` — the resolved
strides of `DenseView::strides()`; `sub`, `row`, `col`, `transposed_view` (flag-free: strides swap)
are the same free operations. `windows_overlap(a, b)` is **exact** for windows of one buffer, so
disjoint windows of the same buffer are allowed (`C -= L·U` in place), as on the CPU.

### 11.3 Kernels (all asynchronous on `Stream s`; `ENGINE_CHECK`s as the CPU twins)

| Kernel | Twin | Notes |
|---|---|---|
| `gemm(ctx, A, B, C, op, s, opts)` | `kernels::gemm` | shared-memory tiles `32×32` / `64×64` / `128×64` (`opts.tile` 0 = auto), all four layout combinations resolved from `A.cs == 1` / `B.cs == 1`; **split-K** `opts.splits > 1` with `opts.workspace` (`gemm_workspace_bytes(m, n, splits, elt)`, allocate before a capture), `suggest_splits(m, n, k)` is the cost-model hint |
| `gemm_nt(ctx, A, B, C, op, s)` | `kernels::gemm_nt` | `B` given `n×k` |
| `schur_update(ctx, L, U, C, s)` | `gemm(GemmOp::Sub)` | the elimination update |
| `GemmBatch<Elt>` | — | `set_shape(A, B, C)` (shape **and strides** shared by every instance), `push(A, B, C)`, `upload(s)` (instance table → device, **not** capturable), `run(ctx, op, s, tile)` (one launch per 65535 instances; capturable). Every instance writes its own `C` window; no reduction, no atomics. |
| `gemv(ctx, A, x, y, op, s)` | `kernels::gemv` | warp-per-row when `A.cs == 1`, thread-per-row otherwise; lane partials canonicalised then summed mod p (the split-K rule) |
| `axpy(ctx, α, x, y, subtract, s)`, `scal(ctx, α, x, s)` | same | u64 uses `α·R` (one REDC per element) |
| `copy(src, dst, s)`, `transpose(src, dst, s)` | same | `32×32` tiles, any strides |
| `canonicalise(ctx, v, s)`; `canonicalise(ctx32, acc64, dst32, s)` | same | `Lazy2p → Canonical`; `Accum64 → Canonical` |
| `trsm_unit(ctx, T, B, s)` | `elim`'s TRSM | `B ← T⁻¹B` in place, `T` tagged `Upper/LowerUnitri` (strict triangle read, diagonal taken as 1); 32-row diagonal solves + `gemm(Sub)` trailing updates |
| `gemm` / `gemm_nt` / `schur_update` / `GemmBatch` on the **IMMA route** (slot-22 G1) | `kernels::gemm` | `u32`, `2 ≤ p ≤ 127`, with `opts.imma_workspace` (`GemmBatch::prepare_imma`): balanced `int8` operands, `int32` tensor-core accumulation, `acc % p` epilogue; `ENGINE_GPU_IMMA=0` disables it. §11.8 |

**Arithmetic per width.** `uint32_t`: products of canonical residues accumulated exactly in
`uint64_t` (`mad.wide.u32`), one Barrett reduction (`__umul64hi` with `ctx.barrett_m`) per
k-block of `kb ≤ min(ctx.k_max, 256)` products, blocks combined with `mod_add`; when
`k_max < 16` (p > 2^30) the counter is per product (same rule, finer grain). `uint64_t`: per
product `REDC(a·b) = ab·R⁻¹` (64-bit Montgomery, `ctx.mont_ninv`), summed mod p, one final
`REDC(S·R²)`; `Regime::U64_Wide` (even p, p ≥ 2^63) uses an exact 128-by-64 shift-subtract
reduction (correct, slow, unused by the project). **The split-K rule (D4)**: each partial is
canonicalised to `[0, p)` before combination; the combine kernel sums the partials mod p in a
fixed order into `C` — an exact mod-p sum is associative, so this is the *only* place a
reduction across thread blocks exists, and it needs no atomics.

### 11.4 Graph capture

`GraphCapture g; g.begin(stream); … launches on that stream …; g.end(); g.launch(stream)` —
0.80 µs per node measured vs 6.0 µs per async stream launch and 21 µs per launch+sync (doc 04
§1.4: 0.68 / 6.11 / 24.0). Rules: no `dalloc` of a fresh block and no staged transfer inside a
capture (both are `ENGINE_CHECK` failures; `capturing_now()` tells), launches only on the captured
stream, and a replay reads whatever the operand buffers hold *at replay time* (pointers are
recorded, not values) — re-upload inputs, then replay.

### 11.5 How the tape's device assignment should call it

1. One `DeviceBuf` per tape buffer that any GPU node touches (`DeviceBuf(host_dense_buf)`); keep
   it for the life of the program (arena allocations are reused across programs).
2. Before a GPU wavefront: `buf.upload(s)` for every input in state `Host` (no-op otherwise); after
   the last CPU node that wrote a buffer, `mark_host_written()`.
3. Nodes: `GEMM → cuda::gemm / gemm_nt`, `BATCHED_GEMM → GemmBatch` (upload the table once per
   compiled program, before capture), `AXPY/SCAL → cuda::axpy/scal`, `TRANSPOSE → cuda::transpose`,
   `CANONICALIZE → cuda::canonicalise`, `SCHUR_UPDATE → cuda::schur_update`, `TRSM → cuda::trsm_unit`;
   `PANEL_RREF` / `PIVOT_SEARCH` stay on the host (`engine::elim` on a downloaded panel).
   After each node writing `buf`: `buf.mark_device_written()`.
4. Capture the wavefront once into a `GraphCapture` keyed by the program's shape signature; replay.
5. `ENGINE_VERIFY_DEVICE` (§4.3): `download(s, force)` into a scratch copy and `memcmp` against the
   CPU node's output — the contract is byte identity, never a tolerance.
6. `ENGINE_DEVICE=cpu` must remain the golden path; `device_enabled()` gates every assignment.

### 11.6 Measured (RTX 5070 Ti, cc 12.0, driver 610.53 / CUDA 13.3 UMD, nvcc 13.0, WSL2)

| | rate | roofline (04 §1.5) |
|---|---|---|
| u32 gemm 512³ / 1024³ / **2048³** / 4096³, ℓ = 134217757 | 2446 / 3816 / **4064** / 3915 G modmul/s | 4931 G → 50 / 77 / **82** / 79 % |
| u32 gemm 2048³ at 2^31−1 (k_max = 4) / 2^32−5 (k_max = 1) | 1628 / 523 G | 2460 G (eager) → 66 / 21 % |
| u64 gemm 512³ / 1024³ / 2048³ at 2^61−1 | 257 / 275 / 274 G | 629 G → 41 / 44 / 44 % |
| batched 2000 × (129×129×129): CPU 1 thread / 2000 stream launches / **one batched launch** / graph replay | 26 / 198 / **2067** / 2066 G | **79× CPU** (floor 10×) |
| pinned direct H2D / D2H; staged through the pool (4 copy threads; 8: 18.9 / 16.8); pageable direct | 55 / 55; 18.0 / 14.4; 15 / 13 GB/s | doc 04: 55.5 / 56.0; —; 24.4 / 5.1 |
| launch: graph node / async / launch+sync | 0.80 / 6.0 / 21 µs | 0.68 / 6.11 / 24.0 |

Phase 5 kernel-level floors (plan): dense u32 GEMM ≥ 40 % of the roofline at 2048 (**82 %**),
batched small-block program ≥ 10× the CPU single-thread rate (**79×**).

### 11.7 WSL2 caveats hit (kernel layer)

* `cudaHostRegister` is unsupported: only `cudaHostAlloc` memory is pinned, hence the staging
  pool; `std::vector` storage is never pinned.
* **Reading `cudaHostAlloc`ed memory from one host thread runs at 5.4 GB/s** (writing 13 GB/s),
  against 12 GB/s pageable→pageable; 4 threads reach 18 / 25 GB/s, 8 threads 24 / 28. A
  single-threaded staged D2H was therefore *slower* than a pageable `cudaMemcpy` (5.1 vs 13 GB/s);
  the pool copies through a persistent 4-thread `CopyPool` (`ENGINE_CUDA_COPY_THREADS`).
* Pageable direct copies measured 15 / 13 GB/s H2D / D2H on driver 610.53 — the D2H pathology of
  doc 04 (5.1 GB/s) is much less severe on this driver, but pinned direct (55 GB/s) still wins 4×.
* The first launch of each kernel instantiation costs milliseconds (lazy module loading):
  `--repeat 1` timings of a fresh process are not measurements — use `--repeat ≥ 3`.
* Small kernels are latency-bound: one 129³ GEMM costs ≈ 8 µs GPU-side even inside a graph
  (2000 nodes: 16.5 ms), against 2.1 ms for the same work as one batched launch — batching, not
  graphs, is the lever for the block shapes (04 §6.4); graphs remove the *host* launch cost.
* No MPS, one process per GPU; 16 GB VRAM (≈ 15 GB free) — `DeviceBuf`s are arena-cached.
  Until slot-14 WP-3 the arena never freed a slab back to the driver, so a long
  `ENGINE_DEVICE=gpu` process (a grid of cells whose buffer sizes all differ) climbed to the card's
  capacity and OOMed (slot-14 [C4] on a 40 GB A100, [C7] on a 94 GB H100, reproduced on the 16 GB
  RTX by `HeckeHidaLevelEllValidation` alone). Since WP-3: the soft trim and OOM retry above, and
  `ValidationRegistry` calls `release_device_caches()` after every validation whenever the CUDA
  context exists, under every device policy (`src/hacs/Util/DeviceRelease.h`;
  slot-14 WP-16 dropped WP-3's non-`cpu` policy gate, since the default policy's scoped-`gpu` arms
  reserve slabs too; a process that never created the context sees one atomic load and nothing
  else — no probe, no context creation).

### 11.8 The IMMA route: int8 tensor-core GEMM for `p ≤ 127` (slot-22 G1)

Files: `ImmaKernels.cuh` (the kernels; included by `cuda_kernels.cu` only), the routing in `cuda_kernels.cu`
(`gemm_impl`, `GemmBatch::run`), the public surface in `CudaKernels.h`. Origin: the P5-G prototype
(`docs/implementation/slot22/P5G_prototype_report.md` §1–§3, sources `docs/implementation/slot22/p5g/`); report
`docs/implementation/slot22/G1_report.md`. Gate: `CudaImmaBitIdentityValidation` (standard).

**Domain.** A `uint32_t` context with `2 ≤ p ≤ 127` — `imma_eligible(ctx)`. Modulus-only, like every kernel of §11.3:
fields and `Z/ell^k` (`4, 8, …, 125`) alike. `uint64_t` contexts, `p ≥ 128` and `k = 0` always take the u32 / u64 kernel.

**Arithmetic (exact, no floating point).**
* Inputs are canonical **by the `Invariant::Canonical` tag** (`check_gemm_operands`), exactly as for the u32 kernel of §11.3,
  whose Barrett path is equally unspecified for `x ≥ p`; the conversion assumes `x < p` and adds no new assumption (audit F-2 / R4).
* A canonical residue `x ∈ [0, p)` is re-encoded as the balanced residue `bal(x) = x − p` if `x > ⌊p/2⌋`, else `x`;
  `|bal(x)| ≤ hb = ⌊p/2⌋ ≤ 63`, so it is an `int8`. `bal(x) ≡ x (mod p)`.
* `mma.sync.m16n8k32.s32.s8.s8.s32` accumulates the products **exactly** in `int32`.
* **The K-chunking rule (the int32 bound).** One k-tile (64 deep) adds at most `64·hb²` in absolute value. With
  `T(p) = ⌊(2^31 − 1 − (p − 1)) / (64·hb²)⌋` (`imma_max_flush_tiles(p)`; 8454 at `p = 127`, i.e. `K ≤ 541 056`; 8729 at 125):
  a GEMM whose padded K has `≤ T(p)` tiles never reduces inside the loop; a deeper one reduces the accumulator `acc %= p`
  every `T(p)` tiles, after which `|acc| ≤ p − 1`. At every step `|acc| < 2^31`: no `int32` ever wraps.
  `GemmOpts::imma_flush_tiles` (validation) forces a smaller interval; one above `T(p)` is an `ENGINE_CHECK` failure.
* Epilogue: `v = ((acc % p) + p) % p` — the unique canonical residue of the exact dot product — then the op: `Set` `c = v`,
  `Add` / `Sub` the u32 kernel's `add32` / `sub32`. **So the bytes equal `engine::kernels::gemm`'s by construction**, as for
  every other device kernel (§11, "a device result is the same bytes as the CPU result by construction").

**Pipeline per GEMM** (`C op= A·B`, any strides, all layout / transposition combinations, `gemm_nt`, `schur_update`):
1. `imma_to_s8_kernel`: `A` (`m × k`) → `A8` (`m × Kp`, row-major, `Kp = round_up(k, 16)`, zero-padded); `Bᵀ` (`n × k`) → `B8`.
   32×32 tiles through shared memory: either source orientation is read coalesced.
2. `imma_gemm_kernel`: block tile 128×128×64, 8 warps (warp tile 64×32), 3-stage `cp.async` pipeline, `ldmatrix.x4`,
   80-byte shared-memory rows, grouped rasterisation, 61 440 B dynamic shared memory (the limit is raised once by
   `imma_supported()`). M / N tails are predicated, K tails zero-filled.
* `GemmBatch`: `prepare_imma(ctx)` (after `upload()`, before a capture) allocates one `A8 / B8` slot per instance
  (≤ 1 GiB in all, else the batch stays on the u32 kernel); `run(ctx, op, s, tile, imma)` converts and multiplies every
  instance in one launch each (`blockIdx.z` / `blockIdx.y` = instance, ≤ 65535 per launch).

**Routing** (every call; the IMMA route is taken iff **all** hold, otherwise the u32 kernel of §11.3 runs, same bytes):

| condition | where |
|---|---|
| `imma_eligible(ctx)`, `k > 0`, `m, n ≤ 2 097 120` | the call |
| `opts.imma == 1`, or `opts.imma == −1` (default) and `imma_policy()` | `GemmOpts::imma`; `0` = never |
| `opts.imma_workspace` with `≥ imma_workspace_bytes(m, n, min(k, 64))` bytes (`A8` then `B8`, each slab rounded to 256 B) | the caller |
| `imma_supported()`: a device of cc ≥ 8.0 and a kernel image with binary **and** PTX version ≥ 8.0 (probed once) | the build / device |

**K chunks.** A workspace below `imma_workspace_bytes(m, n, k)` runs K in chunks of `imma_k_chunk(m, n, k, bytes)` (the largest
multiple of 64 that fits): the first chunk applies `op`, every further one `Add` (`op` = `Set` / `Add`) or `Sub` (`op` = `Sub`) — the
exact mod-p sum of the chunks' canonical products, so the bytes are those of the one-shot product. Each chunk is its own
int32 accumulation (the K-chunking rule above applies per chunk).

Split-K (`opts.splits`) and `opts.tile` apply to the u32 kernel only. A caller that passes no workspace (the
pre-G1 call sites, `CudaBitIdentityValidation`) keeps the u32 kernel: the route is opt-in per call site. `trsm_unit` forwards
its `opts` to its inner `gemm(Sub)` updates, so a caller passing a workspace takes the route inside TRSM too (correctly); the
present direct callers pass none (audit F-8 / R4).

**The switch `ENGINE_GPU_IMMA`** (re-read on every call, like `ENGINE_CUDA_GRAPH`): unset, empty or any value not
starting with `0` = **on**; any value starting with `0` (`0`, `00`, `0x`, …; `imma_policy()` tests the first character) = off (the rollback arm). Default on, because (a) the route is bit-identical by
construction and by `CudaImmaBitIdentityValidation` / `ENGINE_VERIFY_DEVICE`, (b) it is reached only by device GEMMs, which
exist only under `ENGINE_DEVICE=gpu|hybrid` (or a direct call of this layer) — the GPU itself stays opt-in, and
**`ENGINE_DEVICE=cpu`, the golden path, never reaches it** — and (c) it needs cc ≥ 8.0 (sm_80 … sm_120; the reference RTX 5070 Ti
is sm_120), checked at run time, so an older card or an older build image silently keeps the u32 kernel. Like every
`ENGINE_*` switch it **cannot change a value**. **Retired (S24 CL-R2, slot-24 close):** the switch is the `RetiredLever`
`gpu_imma_lever()` of `RetiredDeviceLevers.h` (the same reading). With `OC_ENABLE_LEGACY=ON` it behaves as above. In the default
`OFF` build `imma_policy()` is always on, and a value starting with `0` is refused (`REFUSED ENGINE_GPU_IMMA=0: compiled out
(OC_ENABLE_LEGACY=OFF)`, exit 1; `main()` reads it at start-up). The u32 kernel stays in every build (`p ≥ 128`, `uint64_t`, `imma = 0`).

**The tape** (§12.3): at program build, when `imma_eligible(*ctx) && imma_policy() && imma_supported()`, every GPU
`GEMM` / `SCHUR_UPDATE` / `TRSM`-General node and every `BATCHED_GEMM` node (its sub-batches run one after the other,
`≤ 1 GiB`) is given the IMMA workspace of its **stream slot** (the round-robin slot of `enqueue_level`; one helper `stream_slot(t, nlev)` serves
both, and `enqueue` checks `imma_slot == slot` by `ENGINE_CHECK`, audit R1). **Invariant: one driver per backend** — a backend's
streams, programs and IMMA blocks are driven by one host thread at a time. The workspace is
**one block per stream slot and backend, capped at 128 MiB** (a node whose whole-K workspace is larger runs in K chunks
inside it; a `BATCHED_GEMM` above the cap stays on the u32 kernel): nodes of one slot run in stream order — within a run, inside a captured
graph, and across the programs of the backend, which share its streams and run one at a time (one `Executor`, one
thread) — so they share it. A program that needs more grows the block to the next power of two `≥` its need; the growth
first synchronises the streams, re-points every cached program to the new block and **drops their captured graphs**
(they recorded the old pointer; the next run re-captures), then frees the old block. So a backend holds at most three
IMMA blocks, `≤ 3 × 128 MiB` (the VRAM measured at `(13, 300 / 500 / 700)` is in `G1_report.md`). `DeviceStats::imma_nodes`, `imma_blocks`,
`imma_graph_drops`; `LaunchStats::imma_launches` counts the IMMA GEMM launches. A failed block allocation leaves the slot
on the u32 kernel. `release()` frees the blocks. The cost model of pass 6 (`hybrid`) is unchanged (it does not know the
IMMA rates: G1 open item).

**Not covered (G2–G5).** Device elimination (since slot-22 G2: §11.9, which replaces the whole tape elimination and the host
dense RREFs under `gpu` for fields `p ≤ 127`; `PANEL_RREF` stays on the host for everything else), int8 device storage of the tape's
buffers (operands are still u32 mirrors and are converted per call), `p ≥ 128` (691: the 2-limb Karatsuba route), multi-prime
streams, the `uint64_t` width.

### 11.9 The device-resident elimination: `cuda::imma_rref` for fields `p ≤ 127` (slot-22 G2)

Files: `ImmaElim.cuh` (kernels and driver; included at the end of `cuda_kernels.cu`, nowhere else), the int8 epilogues in
`ImmaKernels.cuh` (`GemmArgs::C8`), the public surface in `CudaKernels.h` (`imma_rref`, `imma_elim_eligible`, `ImmaElimStats`,
`imma_rref_device_bytes`), the routing in `src/Engine/TapeElim.{h,cpp}` and the hook in `DenseElim.{h,cpp}`. Origin: the
P5-G prototype `p5g_elim.cu` (`P5G_prototype_report.md` §4). Gate: `CudaImmaElimValidation` (standard). Report:
`docs/implementation/slot22/G2_report.md`.

**Domain.** `imma_elim_eligible(ctx)`: a `uint32_t` **field** context with `2 ≤ p ≤ 127`. **`Z/ell^k` is not covered**: its
elimination is the Howell form (§13.5, valuation pivoting), not this RREF; every entry point routed here is field-only
(§13.3) anyway. The IMMA GEMM of §11.8 stays modulus-only.

**Algorithm** (the whole elimination on the device; no host round trip but one 4-byte read of the pivot count per base):
* Storage: the matrix as balanced int8 (§11.8's `bal`), row-major, `ld = round_up(n, 32)`; uploaded in row slabs of ≤ 64 MB u32
  through the pinned pool and converted on the device.
* Right-looking blocked PLU with column-order pivoting: outer panels of 256 columns, each factored as **bases of 32 (or 16)
  columns**. A base is **one cooperative launch** (`cudaLaunchCooperativeKernel`, one block of 256 threads per SM, each row owned
  by one thread for the base): per column one grid barrier; the column's pivot key is reduced by `atomicMin` in the same pass that
  eliminates the previous column. No row moves inside a base (pivot rows are stamped); afterwards the base's pivot rows are
  swapped to the top in pivot order (the original index travels with the row). This replaces the prototype's single-block base.
* Inner / outer updates: `L11^{-1}` (one block, the inverse in shared memory), `U12ᵀ = A12ᵀ L11^{-ᵀ}` and `A22 −= L21 U12` as IMMA
  GEMMs with the **fused int8 epilogues**: `Set` (`c = bal(acc mod p)`) and `Sub` (`c = bal((c − acc) mod p)`, in place), the tile
  staged through shared memory so global traffic is 16-byte coalesced. `K ≤ 256` here, far inside the int32 bound `T(p)` of §11.8.
* To RREF (DenseElim step 2, `RREF = U1^{-1} U`): pivot rows in blocks of 256 from the bottom, `R_blk = U_blk^{-1} A_blk` (IMMA
  `Set`), rows above `−= A[above, piv_blk] R_blk` (IMMA `Sub`); the back substitution overlaps the host allocation of the sink.
* **Pivot rule**: the leftmost column with a non-zero Schur-complement entry and, in it, the active row of **smallest original
  index** — the row-rank-profile rule of `PIVOT_SEARCH` (not the swap order of DenseElim's `Plu`, which exposes no pivot rows). The chosen row set after every column `c` is then the row rank profile of
  `A[:, 0..c]`, so the returned `pivot_rows` (pivot `t` ← original row) is a unique function of `A` as well.

**Why bit-identical.** RREF, pivot columns and rank are unique functions of the input (§8); every step is exact integer arithmetic
(int8 balanced residues, int32 accumulation below `2^31`, integer `%`; no floating point), so a correct implementation reproduces
`engine::elim::rref` byte for byte, and the kernel / image / complement bases, the projector and the solves derived from it follow.
`tape::rref`'s `pivot_rows` (§10.7: per block column, the panel's rows ascending — `PIVOT_SEARCH`'s row rank profile) is rebuilt from
the device's per-pivot rows by sorting within each block column: both sides select the same set per column prefix.

**Routing** (`tape::device_elim_wanted(ctx, m, n)`, checked in this order, so the `cpu` default touches no CUDA):

| condition | |
|---|---|
| `ENGINE_GPU_ELIM` not starting with `0` | re-read per call; `0` = the rollback arm (the host / tape-panel route as before). **Retired (S24 CL-R2):** `OC_ENABLE_LEGACY=ON` only; the default `OFF` build refuses `0…` (exit 1, also at start-up in `main()`) and the condition is always true. The host route stays (it is the route under `cpu`, for `p > 127` and below the size floor) |
| `imma_elim_eligible(ctx)` | `uint32_t` field, `p ≤ 127` |
| `m · n ≥ ENGINE_GPU_ELIM_MIN` (default `1048576`) | re-read per call; a size floor, tuning only |
| `elim_pass_options().effective_policy() == Gpu` | `ENGINE_DEVICE=gpu` or the thread-local / scoped override; **never** `cpu`, never `hybrid` |
| `cuda::available() && imma_supported()` | a cc ≥ 8.0 device with an sm_80+ image |

Then (a) `tape::rref` — and with it `kernel_basis`, `image_basis`, `factor_basis` stage 1, `solve_multi_rhs`'s elimination route —
materialises `A`'s rows in slabs (`blk::materialise_rows`) and runs one `imma_rref` instead of the per-panel programs (no
elimination session, no `PANEL_RREF`; `TapeElimStats::device_elims`); (b) the `DenseElim` hook (`elim::set_device_rref_hook`,
installed by `TapeElim.cpp` at static initialisation) routes every `uint32_t` in-place RREF of `DenseElim.cpp` — `rref`,
`rref_with_transform`, `inverse`, `solve_multi_rhs`, `complement_basis` — and `column_rank_profile` (rank / pivots only, no back
substitution, `A` left untouched: its contents were unspecified anyway). That covers the cone's `DC` RREF, `rank_of_view`,
`factor_basis` stage 2's inverse and dense RREF, and the ι carry checks. A `CudaError` raised before any output is written falls
back to the host route (`DeviceElimTotals::fallbacks`); after the first output write it propagates.

**`ENGINE_VERIFY_DEVICE`.** The device elimination is not a tape node, so §12.4's re-execution does not see it; instead, in an
`ENGINE_VERIFY_DEVICE=ON` build (or with `ENGINE_GPU_ELIM_VERIFY=1`) every device elimination is re-run on the host under the
thread-local `Cpu` override and compared entrywise (rank, pivots, RREF / `R`, `pivot_rows`); a difference throws
`std::logic_error` (`DeviceElimTotals::verified`).

**Memory.** One arena block per call (`imma_rref_device_bytes(m, n)`: `m · round_up(n, 32)` bytes of matrix, three `max(m, n) × 256`
work slabs, one 64 MB staging slab), rounded up to a size class (4 per octave) so the arena's exact-size free lists reuse it across
differently shaped eliminations; freed at the end of the call. `10515 × 8668`: 160 MB. Thread-safe: each call owns a stream and its
block (concurrent calls from the cone / ι task threads run side by side).

**Counters / trace.** `cuda::imma_elim_stats()` (`calls`, `bases`, `coop_launches`, `gemms`, bytes, peak block, seconds);
`tape::device_elim_totals()` (`tape_rrefs`, `dense_rrefs`, `rank_profiles`, `fallbacks`, `verified`); `ENGINE_TRACE_IMMA_ELIM=1`
prints per call the phase times (it synchronises per phase: diagnostic only); `ENGINE_TRACE_RSS=1` marks a device `tape::rref`
line `device (imma_rref)`. Like every `ENGINE_*` switch, none of them can change a value.

**Measured** (RTX 5070 Ti, loaded box; `G2_report.md`): `10515 × 8668` random at `p = 13`: **0.29–0.31 s** including both transfers
and the host allocation of the 300 MB result (host `elim::rref`, one thread: 12.5 s); factorisation ≈ 0.12 s (271 bases), back
substitution ≈ 0.03 s.

## 12. Tape ↔ device (`src/Engine/DeviceBackend.h/.cpp`, pass 6/7 of `Passes.cpp`) — Phase 5 Stage B

`MatrixEngine` now links `EngineCuda` (`DeviceBackend.cpp` is a plain C++ TU over the
CUDA-header-free headers of §11). Every `Executor` owns one `DeviceBackend` when
`engine::cuda::available()`; otherwise it has none and GPU-assigned nodes run on the CPU
(`ExecStats::fallback_nodes` counts them). **A value never depends on the device**: every device
kernel is the exact-integer twin of its CPU kernel (§11), the k-blocking / split-K rules of §11.3
apply unchanged, and `EngineCrossDeviceValidation` asserts the hashes are identical under every
policy — with the `ENGINE_VERIFY_DEVICE` re-execution on, and off.

### 12.1 `ENGINE_DEVICE` semantics (pass 6, `pass_device`)

| policy | assignment |
|---|---|
| `cpu` (default) | every node `Device::CPU`; no transfer nodes; the golden path. |
| `gpu` | every device-eligible node `Device::GPU` (`device_eligible(op)`: all ops except `PANEL_RREF`, `PIVOT_SEARCH`, `COMPARE`, which are host-serial / host-compared — the residency pass inserts the `D2H` a `COMPARE` needs). |
| `hybrid` | the D4 inequality per **group** = the would-be batch (same level, same batching key without the device): `t_cpu = W/R_cpu + n·cpu_node + bytes_back/B_d2h` vs `t_gpu = W/R_gpu + launch + bytes_h2d/B_h2d + bytes_d2h/B_d2h`, GPU iff `t_gpu < t_cpu`; regime `ℓ ≥ 2^32` (any `uint64_t` context) always GPU (D4). Then memory-class nodes (`AXPY`, `SCAL`, `TRANSPOSE`, `SCATTER`, `CANONICALIZE`, `GATHER`) go to the GPU iff every buffer they touch is device-resident after the compute decisions. |

The `hybrid` simulation walks the groups in level order with a residency model: `resident[b]`
(the device copy is current), `dev_newer[b]` (only the device copy is current — set by a GPU
write, cleared by a CPU write). `bytes_h2d` counts the non-resident operands, `bytes_back` the
`dev_newer` ones a CPU choice would have to download, `bytes_d2h` an `Output` once. Two
refinements keep an accumulator from ping-ponging: a buffer whose **first** access is a
whole-buffer Set write by a memory node (`zero()`, a `SCATTER` copy) costs no upload (that writer
follows the compute onto the device), and a relayout twin of an Input (pass 3's `TRANSPOSE`) is
charged as its source, once. `R_cpu` is the 16-thread rate (dense `cpu_modmac_16t`, batched
`cpu_modmac_16t_small`), `R_gpu` the dense rate when the node has ≥ `gpu_dense_tiles` 64×64
output tiles, else the small-block rate, with the `gpu_kernel_floor_s` latency floor per
unbatched kernel; the u32 Shoup / Barrett-64 regimes and u64 use their measured rates. Every
constant is in `EngineCostModel.h` with the section it was measured in; `ENGINE_TRACE_PASSES=1`
prints one line per group with both estimates and the verdict.

The decision is **recorded in the node** (`Node::device`) and the transfer nodes are program
nodes, so a compiled program runs the same way under any `ENGINE_THREADS`, batching or graph
setting; `PassOptions::effective_policy()` maps the Phase 4 pair (`device_model`,
`device_policy`) to `Hybrid` for the older validations.

### 12.2 The residency pass (pass 7, `pass_residency`) and the transfer nodes

Whole-buffer residency per program buffer, simulated level by level from `Host` (Inputs /
Outputs) or `None` (Temps):

| state before the level | a GPU node touches `b` | a CPU node touches `b` |
|---|---|---|
| `Host` (host newer) | **`H2D` node** — unless the access is a whole-buffer Set write | nothing |
| `Device` (device newer) | nothing | **`D2H` node** — unless the access is a whole-buffer Set write |
| `Both` / `None` | nothing | nothing |

After the level the writers set the state (`Device` / `Host`). Two devices writing one buffer in
one level would leave neither copy whole, so such GPU writers are moved to the CPU first (fixed
point, `PassStats::forced_cpu`). `H2D` nodes are **hoisted** to the level after the buffer's last
host write (level 0 for Inputs) so the DMA overlaps the CPU levels in between
(`stats.hoisted_h2d`); `D2H` nodes stay at the level of the first CPU reader; Outputs left on the
device are downloaded in an **epilogue level** (`levels.size()` grows by one). A transfer node is
an ordinary `Node` (`op ∈ {H2D, D2H}`, `c` = the whole buffer, `device = GPU`, `cost` = bytes,
`origin = 0xFFFFFFFF`) appended to `prog.nodes`, so it is visible in `summary()`, counted by
`node_accesses` (H2D reads `c`, D2H writes it) and by the lifetime analysis.

At run time the mirrors' version pairs (§11.2) are the ground truth and the plan is asserted
against them: an `H2D` of a device-newer buffer (or a `D2H` of a host-newer one) is an
`ENGINE_CHECK` failure; a transfer whose buffer is already `Both` is a counted no-op
(`DeviceStats::skipped_transfers` — the normal case after a verify re-execution). The bytes moved
are `(major − 1)·ld + minor` elements, never the full footprint: an Input bound as a window of a
larger buffer does not own its last row's padding.

### 12.3 Execution (`DeviceBackend`, one per `Executor`)

* **Device program** = per compiled program (`Program::uid`, fresh per `compile()`): one
  `DeviceBuf` mirror per buffer any GPU or transfer node touches (allocated once, rebound to each
  run's host buffers with `rebind_host` — Inputs/Outputs start host-newer, Temps `None`), the
  `GemmBatch` tables of every `BATCHED_GEMM` (one sub-batch per stride tuple, uploaded at build),
  split-K workspaces (`cuda::suggest_splits`), scratch slabs for scaled products, device copies of
  the `GATHER` index lists, and the graphs. Kept in an LRU of `ENGINE_CUDA_PROGRAM_CACHE` (64)
  programs per executor; a different `ld` on a rebinding rebuilds it; `Executor::release_device_state()`
  frees everything (the arena's `live_bytes` return to baseline — `EngineCudaMemoryValidation`).
* **Node → kernel**: `GEMM`/`SCHUR_UPDATE` → `cuda::gemm` (`alpha ≠ 1`: product into a scratch
  slab, then `copy`+`scal` or `axpy`; since slot-22 G1 with the node's stream-slot IMMA workspace when the modulus
  qualifies, §11.8 — `DeviceStats::imma_nodes`); `BATCHED_GEMM` → `GemmBatch::run` per sub-batch;
  `TRSM` General → `gemm(B⁻¹, b)` (the host inverted the pivot), Identity → `copy`, Scalar →
  `copy`+`scal`; `AXPY` → `axpy` / `axpy_identity`; `SCAL` → `scal` / `fill(0)`;
  `TRANSPOSE`/`SCATTER` → `copy`; `CANONICALIZE` → `canonicalise`; `GATHER` → `gather` (new
  kernels `fill`, `axpy_identity`, `gather` in §11.3's file).
* **Streams and runs**: three private streams per backend (`CudaCtx::create_stream`); the GPU
  nodes of a wavefront are dealt round-robin over them between a fork event and join events.
  Consecutive wavefronts with no CPU node and no `D2H` form one **run**: enqueued in stream order
  (which carries the inter-level dependencies), captured as one graph, synchronised once
  (`DeviceStats::merged_levels`; `ENGINE_CUDA_MERGE_LEVELS=0` disables it).
* **Graph capture per run** (`GraphCapture`, §11.4): `ENGINE_CUDA_GRAPH=1` (default) launches
  directly on a program's first run and captures on its second (a one-shot program — the
  elimination driver records one per panel — never pays a capture); `2` captures on the first
  run; `0` never. A replay of a cached program is therefore one `cudaGraphLaunch` per run plus its
  transfers.
* **Order inside a level** (`Executor::execute`): transfer nodes → (verify: host sync) → GPU launch
  → CPU tasks of the level on the pool → host wait on the run's event → residency bookkeeping
  (`mark_device_written` / `mark_host_written`) → verify → `COMPARE` reductions.

### 12.4 `ENGINE_VERIFY_DEVICE`

`cmake -DENGINE_VERIFY_DEVICE=ON` (compile-time default) or `Executor::set_verify_device(true)`:
before a level's launch every buffer its GPU nodes touch is made current on the host; after the
wait the device copies of the buffers they wrote are downloaded, the same nodes are re-executed
on the CPU through the ordinary task expansion, and every written window is compared entrywise —
a difference throws `std::logic_error` naming the node, op, level, buffer and window. Verify
disables run merging (levels are compared one by one) and makes the planned `D2H` nodes no-ops;
`EngineCrossDeviceValidation` runs both verified and unverified arms for that reason.

### 12.5 Validations and measurements

* `EngineCrossDeviceValidation` (standard, ~18 s on the reference GPU): Theta-like `gemm_dense`
  (replayed), block × block, the elimination driver (`rref` / `kernel_basis` / `factor_basis` /
  `solve_multi_rhs`), gather / scatter / `COMPARE`, the op set (scaled GEMMs, TRSM kinds,
  `AXPY`-identity, `SCAL`), a scaled batched program and a `k = 4200` GEMM, over
  `p ∈ {134217757, 2^31−1, 2^32−5, 7}` (u32) and `{2^61−1, 134217757}` (u64), under 10 settings
  (`cpu`/`gpu`/`hybrid` × threads × batching × graph mode × verify): 372 checks, hashes identical,
  every GPU node of the verified arms re-executed, downloads executed on the unverified arms.
  Skips loudly without a device. The golden grid is reported **pending** until the consumers
  record on the tape.
* `EngineCudaMemoryValidation` (standard, ~1 s): a 144 MB operand through the 128 MB pinned pool
  (peak = cap, 25 chunk waits), a replayed Theta-like program, the elimination driver;
  `pageable_copies == 0`, `device_bytes() == 0` and arena `live_bytes` back to baseline after
  `release_device_state()` and after executor teardown; since slot-14 WP-3 arm (H): the release
  returns every idle slab and the pinned pool, the same kind of work without a release grows the
  arena (the counterfactual), and after a release run 2's high-water is ≤ run 1's + one slab.
* `EngineDeviceTrapValidation` (slow, ~12 s): the table below; gated on identity only.

**Cross-architecture provenance** (slot-14 WP-13, proposals P10 / P16 of
`docs/implementation/slot14/WP-D1_binding_doc_proposals.md`; evidence, not a new rule). The device route's
bit-identity has been checked on two architectures besides the reference `sm_120`:
* **A100 (`sm_80`)**, an HPC-cluster run (commit `07f413f`, GCC 12.3.0, nvcc 13.2): strict golden
  (tier A 11687 / 0, tier B 5901 / 0) under `ENGINE_DEVICE=cpu` / `gpu` / `hybrid` and under the
  `ENGINE_VERIFY_DEVICE=ON` build, the six device validations, the three determinism validations with
  C4 (a)'s 24 arms, and identical micro checksums.
* **H100 (`sm_90`)**, a second HPC-cluster run (commit `e0bcdf3`, GCC 12.3.0, nvcc 13.2): the same strict golden
  under cpu / gpu / hybrid and `ENGINE_VERIFY_DEVICE=ON` (`EngineCrossDeviceValidation` 372 / 0), all four
  pinned digests, C4 (a) 24 / 24, and `CudaBitIdentityValidation` 10844 / 0.

On both machines `--all` under a non-`cpu` policy ran out of device memory. That was the slab-release defect,
not a value defect; it is fixed by the lifetime rule of §11.2 (slot-14 WP-3).

**The unblocked-kernel trap, measured** (RTX 5070 Ti, 32-thread pool, best of 3, wall time
*including* every transfer; `ENGINE_DEVICE` as the column):

| case | cpu | gpu | hybrid | GPU nodes gpu/hybrid, MB moved gpu/hybrid |
|---|---|---|---|---|
| A. `tape::rref` 1677×1419 rank 258 (13×11, d = 129) | **33.2 ms** | 47.3 ms | 33.4 ms | 2837 / 0, 76 / 0 |
| B. `tape::rref` dense 4096² full rank (16×16, d = 256) | **705 ms** | 980 ms | 720 ms | 31702 / 15, 2471 / 252 |
| C. Θ-like 2000 × 129³, first run (direct launches) | 32.4 ms | 44.2 ms | 32.9 ms | 628 / 628, 34 / 34 |
| C. Θ-like 2000 × 129³, replay (graph) | 16.6 ms | **13.0 ms** | 16.5 ms | |
| D. dense GEMM 4096³, first run | 181 ms | 31.4 ms | 30.6 ms | 1 / 1, 201 / 201 |
| D. dense GEMM 4096³, replay | 179 ms | **30.6 ms** | 29.9 ms | |
| E1. batched d = 100 × 256, 256 in / 256 out blocks, replay (cpu = 1 thread) | 14.4 ms | **1.63 ms** | 14.5 ms | 3 / 0, 21 / 0 |
| E2. batched d = 100 × 256, 32 in / 32 out blocks, replay (cpu = 1 thread) | 9.9 ms | **0.57 ms** | 9.8 ms | 10 / 0, 2.9 / 0 |

Reading: the elimination driver records one program per panel, so nothing is amortised — the
`gpu` arm re-uploads the panel operands every time (2.5 GB for B) and is 1.4× *slower* than the
CPU; the cost model keeps A and B on the CPU (B: 15 small Schur updates go over, +2 %). The
Θ-like program pays for its 34 MB of transfers on the first run and wins on replay (the graph).
The 4096³ GEMM wins 6× on the first run already. `hybrid` is never worse than the better of the
two by more than the model's slack (C replay: 16.5 vs 13.0 — the model's `cpu_modmac_16t_small`
is the 16-thread figure, the 32-thread pool is faster).

**k-blocking evidence** (D): at `ℓ = 134217757`, `k_max = 1023`; the device kernel flushes every
`kb = min(k_max, 256) = 256` products, so a 4096-deep dot product is **16 chunks** (≥ 4). The
unblocked variant, `kernels::dot_delayed(ctx, a, b, 4096)`, trips `ModContextValidation`'s
`k ≤ k_max` `ENGINE_CHECK`; an unchecked `uint64` accumulation of 4096 products of `(p−1)²`
wraps and yields 130777117 where the exact value is 4096.

**Exit criteria (plan Phase 5), through the tape, replay, transfers included:**

| criterion | measured | verdict |
|---|---|---|
| blocked GEMM ≥ 2 TmodMAC/s at 4096³ | 2.25 T (`gpu`), 2.30 T (`hybrid`); the kernel alone 3.9 T, the 192 MB of transfers ≈ 13 ms of the 30 ms | **PASS** |
| batched d = 100 GEMM ≥ 20× CPU single-thread on a 256-instance batch | E2 (Θ shape, 32 in/out blocks): **17.2×** (0.57 ms vs 9.9 ms; 447 G vs 26 G); E1 (256 distinct in/out blocks, 21 MB moved): 8.8× | **FAIL** (transfer latency: ~30 µs per staged copy × 10 buffers + 3 graph launches ≈ 0.4 of the 0.57 ms; the kernel time is 0.13 ms = 2 T) |
| `EngineCrossDeviceValidation` exit 0 | 372 checks, 0 failed | **PASS** |
| `cohom 1000 134217757` ≤ 1.5 s under `ENGINE_DEVICE=gpu` | pending: the consumers are not on the tape | pending |

### 12.6 Re-measuring the constants

`EngineCostModel.h` is the only table. `cpu_modmac_16t` / `cpu_modmac_16t_small`:
`EngineCoreScalingValidation` (dense 2048³ and Θ-like at 16 threads). `gpu_modmac`,
`gpu_modmac_small`, the regime rates: `--bench micro cuda` (§11.6). `h2d_bytes_s` / `d2h_bytes_s`:
the staged figures of `--bench micro cuda` (not the pinned-direct 55 GB/s — tape buffers are
pageable). `gpu_launch_s`, `gpu_kernel_floor_s`: §11.6 / §11.7. `d2h_call_s` / `h2d_call_s` are
the per-call latencies of a staged copy (order of magnitude; not yet used by the model).
`ENGINE_TRACE_PASSES=1` shows every group's estimate next to `EngineDeviceTrapValidation`'s
measured wall times, which is how a re-measured table is checked.

### 12.7 What the consumers must do (hooks for the integrator)

*Status after Phase 4 Stage B (§10.9): items 1 and 4 are done for the consumers listed there;
item 2 (cross-program residency of pool inputs) is still open in general -- closed in the narrow
form of **one elimination session** by slot-18 WP-19a (§12.8); item 3 (route 2 on the GPU) is
moot — route 2 is now streamed per block row as `COMPARE` programs. Measured through the
consumers (default build, `--repeat 2`, RTX 5070 Ti): `cohom 256` cpu 0.718 s / hybrid 0.622 s /
gpu 0.732 s; `hecke 256` cpu 1.314 s / hybrid 1.153 s / gpu 1.451 s; `cohom 1000 --lean` cpu
24.6 s / 1157 MB, hybrid 25.8 s / 1451 MB, gpu **80.1 s** / 1449 MB (the CUDA context and the
pinned pool add ~290 MB). `hybrid` wins ~13 % at `n = 256` and nothing at `n = 1000`; the
one-shot per-panel elimination programs re-upload their operands (rows A/B of §12.5) and the
`gpu` arm is 3.3× slower than the CPU, so the default stays `cpu`. Phase 5's `cohom 1000 ≤ 1.5 s`
is therefore **not met** and needs item 2 (residency across programs) plus a device-side panel
factorisation.*

1. **Cache and replay programs.** Everything above amortises per *program*: the mirrors, the
   batch tables and the graph belong to `Program::uid`. A consumer that records a fresh program
   per call (as `TapeElim` does per panel) gets direct launches and a full re-upload every time —
   the `gpu` column of rows A/B. `ProgramCache` hits are what the device arm is built for.
2. **Inputs are re-uploaded on every run** — a binding may point at new data. To keep a pool
   block resident across programs the consumer would need a stable content identity across
   recorders (today `Recorder::pool_input` mints a fresh key per recorder); that cross-program
   residency cache is the one missing piece for the elimination shape and needs an owner
   decision on the identity contract before it is built.
3. **Route 2 on the GPU, the fused route on the CPU** (plan Phase 5, `02 §7.16`): run the two
   Θ programs on two executors with `PassOptions::policy` set to `Gpu` and `Cpu` respectively
   (the executors share the pool; the device one owns its streams) and `blk::equal` the results —
   no backend change needed, the option is per compile.
4. `ENGINE_DEVICE` stays **cpu by default**. On the evidence above `hybrid` can be recommended
   once the consumers replay cached programs; it must not be flipped before the consumer-level
   number (`cohom 1000`) exists.

### 12.8 The elimination session (slot-18 WP-19a, GPU G1)

The narrow form of §12.7 item 2 that [G14a] §3.1 (slot-14 WP-14a) costed: residency across the
programs of **one** `tape::rref` call (so every entry point of §10.7 that eliminates through it:
`kernel_basis`, `image_basis`, `factor_basis` stage 1, `solve_multi_rhs`'s elimination route).
No content-identity contract is needed; nothing outlives the call.

* **When.** `tape::rref` opens a session iff the elimination's pass options select a device policy
  (`effective_policy() != Cpu`) **and** a device is usable **and** `ENGINE_ELIM_SESSION` is not `0`
  (the rollback arm; **retired by S24 CL-R2**: honoured only with `OC_ENABLE_LEGACY=ON`; the default `OFF` build refuses `0…`,
  exit 1, also at start-up in `main()`. The per-program-mirror path stays: it is the path of every program outside a session). Under the default `cpu` policy the session code is never reached and no CUDA
  context is created. Nested calls on one executor open no second session.
* **What is adopted** (`DeviceBackend::session_adopt`, keyed by the host base pointer, `std::map`):
  `A`'s pool blocks (read-only: uploaded once per call, not once per panel), the growing `R`
  (*kept*: downloaded at release), and the per-panel device scratch `Lg`, `T`, `Lp`, `C`
  (*dropped*: never read by the host). An Input / Output binding whose storage is an adopted
  buffer, or a prefix window of it (same base, `ld`, layout; rows and columns ≤), is bound to the
  session's **one** mirror instead of a per-program mirror; buffer coordinates coincide, so the
  node views are unchanged. Mirrors are created lazily by the first program that binds them.
* **The run-time rules** (the version pair of §11.2 stays the ground truth): a planned `H2D` of a
  session mirror that is already device-newer is skipped (`session_skipped_h2d`: the plan of
  §12.2 assumed a fresh, host-newer binding); a `D2H` with no CPU node touching the buffer later in
  the program is deferred (`session_deferred_d2h`); a program with a CPU node touching a
  device-newer session buffer downloads it at `begin_run` (`session_syncs`); a program with no
  device work at all (`begin_run` not reached) downloads the device-newer adopted buffers it binds
  and marks the ones it writes host-newer (`session_host_run`); a binding of an adopted buffer
  with an existing mirror is mapped even when no GPU node touches it, so a CPU write marks it.
  A program bound to session mirrors is dropped from the device-program cache at `end_run` (the
  driver's programs are one-shot), and a cached program that would bind an adopted buffer through
  its own mirror is rebuilt.
* **The driver's contract** (`TapeElim.cpp`): no host code touches an adopted buffer between
  adopt and release except through tape programs. `R`'s growth copy (`GrowRows::append` beyond
  capacity) and `take()` are preceded by `session_release` (download), the new storage is
  re-adopted; the per-panel scratch is released at the end of the panel's scope (also on the
  `continue` of a pivot-free panel); an exception drops the session without downloads.
* **The row-list `GATHER` on the device.** `cuda::gather(..., rows)` reads `a(rows[i], col(t))`
  with the identity test `row_off + rows[i] == col(t)` — the executor's row-list `GATHER` is its
  CPU twin; `device_eligible_node` no longer excludes it, so `P2`'s pivot-row gathers stay on the
  device under `gpu` (and under `hybrid` when their buffers are resident).
* **Values.** Unchanged by construction: the same exact-integer kernels on the same bytes; the
  session only removes copies of bytes that are already current on the other side.
  `EngineElimSessionValidation` (standard, ≈ 9 s, [SKIP] without a device) gates it: `rref` /
  `kernel_basis` / `factor_basis` / `solve_multi_rhs` entrywise against `engine::blk` under `gpu`
  and `hybrid`, with the `ENGINE_VERIFY_DEVICE` re-execution on and off and the session on and off,
  u32 (`134217757`, `7`, `2^32−5`) and u64 (`2^61−1`); the device row-list `GATHER` against the
  CPU executor; the non-vacuity counters; fewer transfer bytes than the rollback arm;
  `pageable_copies == 0`; no device bytes left after `release_device_state()`. Strict golden
  (tier A 11827 / 0, tier B 6231 / 0) is green under `ENGINE_DEVICE=gpu` and `hybrid`.
* **Measured** (RTX 5070 Ti, loaded box, `--repeat 3` min, WP-19a): `cohom 1000 134217757 --lean`
  cpu 14.13–14.81 s, **gpu 11.90–12.85 s** (session), gpu 40.2 s with `ENGINE_ELIM_SESSION=0`,
  hybrid 14.57 s; VRAM peak 4.6 GB (session) against 15.8 GB (rollback). `EngineDeviceTrapValidation`
  row A gpu 19.3 vs cpu 26.4 ms, row B gpu 327.6 vs cpu 434.0 ms. `rref 1677×1419` (13×11, `d = 129`)
  under gpu moves 56 MB in 32 H2D copies against 161 MB in 76 without the session. The [G14a]
  stop line (`cohom 1000` gpu ≤ cpu) is **met**; the default stays `cpu` (§12.7 item 4: the
  consumer-level `hybrid` number does not yet win, and G2 — the host dense eliminations of
  `factor_basis` stage 2 on the device — is the next step).

### 12.9 The VRAM-lean device session (slot-23 PERF-P7)

Files: `Executor.{h,cpp}` (the rule), `Passes.cpp` (`PassStats::gpu_compute_nodes`), `CudaCtx.{h,cpp}` (the slack default, the
`ENGINE_TRACE_VRAM` sampler, `arena_idle_slack()`, `set_trace_tag()`). Gate: `GpuLeanSessionValidation` (standard). Report:
`docs/implementation/slot23/P7_report.md`.

* **The rule.** `Executor::run` runs a compiled program on the host path — GPU-assigned nodes on the CPU, transfer nodes
  skipped, exactly the no-device path of §12 — when (a) its GPU nodes are all memory-class (`PassStats::gpu_compute_nodes == 0`:
  no GEMM-class, `BATCHED_GEMM` or `TRSM`-General node on the GPU), (b) no elimination session is open on the executor (§12.8:
  the session's residency wins), and (c) `ENGINE_GPU_LEAN` does not start with `0` (re-read per run; default **on**; `0` is
  the rollback arm). **Since S25 LEVERS-25 the env arm is retired** (`RetiredDeviceLevers.h`, `gpu_lean_lever()`, an
  `EnvArmRetiredLever`): an `OC_ENABLE_LEGACY=OFF` build refuses `ENGINE_GPU_LEAN=0…` (exit 1, at start-up in `main()`), an ON build
  reads it as before; the rule's off state stays in every build through `engine::lever::EnvLever::Scoped(gpu_lean_lever(), false)`,
  which is how `EngineCrossDeviceValidation`'s `lean=0` arms and `GpuLeanSessionValidation` now switch it. `ExecStats::lean_host_runs` and `gpu_lean_host_runs()` (process total) count it. Such a program does no
  GEMM-class work — memory movement and `O(size)` modular updates (AXPY, SCAL, scalar / identity TRSM) — so on the device it cost one
  mirror per operand and a PCIe round trip for work linear in the bytes moved. At `(13, 700)` that was the route-2 O12(a)
  cross-check of the Hecke operator: 3325 `ρ_n` blocks, **6.2 GB of the 10 GB live peak**.
* **`ENGINE_VERIFY_DEVICE` runs (audit R3).** Under the default, lean programs never reach the device, so a verify run re-checks only
  the device's compute kernels on real workloads. A verify campaign therefore includes an **`ENGINE_GPU_LEAN=0` arm** (since S25
  LEVERS-25 on an `OC_ENABLE_LEGACY=ON` build), so that the
  memory-class kernels (AXPY, SCAL, TRANSPOSE, SCATTER, GATHER, CANONICALIZE) are re-executed and compared on the same workload as well.
  Their standing coverage is `EngineCrossDeviceValidation`'s three `lean=0` arms and `GpuLeanSessionValidation` (L).
* **Mixed state (audit NIT).** `ENGINE_GPU_LEAN` is read per run for the rule but once, at context creation, for the slack. Toggling it
  mid-process gives the new rule with the old slack, and vice versa. Neither combination can move a value; set it before the first GPU use.
* **The tally (audit R4).** `DeviceTally::gpu_programs` counts compiled placement, i.e. programs with ≥ 1 GPU-assigned node.
  `DeviceTally::lean_host_programs` counts those that the rule ran on the host, so the device runs are `gpu_programs − lean_host_programs`.
* **The slack.** The arena's soft-trim slack (§11.2) defaults to a **sixteenth** of the device (1 GB on the 16 GB card) under
  the lean default, and to a quarter, as before, under `ENGINE_GPU_LEAN=0`. It is read once at context creation;
  `ENGINE_CUDA_ARENA_SLACK_MB` still overrides it.
* **Why no value moves.** The rule chooses where a node runs, never what it computes: every device kernel is the exact twin of
  its CPU kernel (§11). The slack decides when idle slabs go back to the driver, never which bytes a live block holds. The rule
  is reached only by programs compiled under `gpu` / `hybrid`, so `cpu`, the golden path, never sees it.
* **Measured** (RTX 5070 Ti, the census C1a cell spec, 3 threads; card 16.3 GB with 2.2 GB foreign). `(13, 700)` own peak
  9.6–11.3 GB → **2.85 GB** (live 10.1 → 2.37 GB). `(7, 900)` **3.27 GB**. `(7, 1000)` **3.78 GB**. Wall is unchanged at 700.
  Records are byte-identical to `cpu`. The live peak is now Θ program 1 (`hecke.theta`) or the cone. Both scale like `n²`, so
  the card is no longer the limit for `n ≤ 1500`; host RSS is (`P7_report.md`).
* **Diagnostics** (none can change a value). `ENGINE_TRACE_VRAM=<ms>` starts a sampler that prints
  `VRAM t= live_mb= reserved_mb= dev_used_mb=` to stderr. At exit it prints `EFFECTIVE DEVICE_MEM peak_live_mb= peak_reserved_mb=
  dev_used_peak_mb= … at_peak: <tag>:<count>x<KB> …`: the live blocks at the live peak, grouped by program family. `run_cached`
  sets the family to the `ShapeKey` tag through `cuda::set_trace_tag`, and `tape::rref` sets it to `tape.rref`; other direct `ex.run`
  callers show as `-`. The sampler stops before exit teardown.

## 13. Composite modulus and non-field elimination (`Z/ell^k`) — phase E8, item 1

**Status: a contract, written before any code** (hecke-extensions plan [X] v1.10 §6.2 E8 item 1,
risk R21), reviewed in slot 7 (WP-10), and **implemented by E8 item 2** (hecke-extensions slot 8,
WP-1: `ModCtx.h`, `DenseElim.{h,cpp}`, `RefLinAlg.{h,cpp}`, the §13.3 guards, the §13.7 replay
check; gated by `HeckeReductionValidation`'s elimination arms and `ModContextValidation`'s
composite arms), **consumed by E8 item 3** (slot-8 WP-3: the pipeline over `Z/ell^k`, the §13.7
`ModCtx` overload, F4's `ρ_n` repair) and item 4a (WP-4: the reductions and the Bockstein), and
**reviewed** on Fable (slot-8 WP-12: pass 1 on §13 as amended, ACCEPT; its three pointer repairs
applied at the slot-8 close). Every name below that §1–§12 do not already define was **specified here first and
then built**; slot 8 WP-1 amended the wording (review findings m1–m10, m12, m13 of
`docs/implementation/slot7/WP-10_review.md`, the entry-point names of m11, and the deferred (C2)
design, §13.11) and **changed no rule**. The section fixes three things:

* (a) what a composite modulus `p = ell^k` may reach unchanged;
* (b) what must refuse it;
* (c) the unique outputs of the new elimination.

With these fixed, the bit-identity discipline of §5 and §8 applies to the new path, and the `F_ell`
path provably stays as it is. Line references are to the slot-8 WP-1 tree (the implementation);
the slot-7 text cited the tree at `596a47b`.

**Indexing (review m3), stated once.** The definition of §13.5 is written 1-based, as in the
literature. The algorithm (`c = 0, 1, …`, `L_0`), every entry point of §13.6 ("leading column
`≥ m`") and all the code are **0-based**.

**Scope.** `ell` prime, `k ≥ 2`, and `ell^k` representable in `Elt` (`< 2^32` for `uint32_t`,
`< 2^64` for `uint64_t`: Montgomery-64 for odd `ell^k < 2^63`, `U64_Wide` at or above `2^63` and for
`ell = 2`, by the existing `init(p)` rule, §13.2). The reduction maps `Z/ell^{k+1} → Z/ell^k` and
the `a = b = 1` Bockstein are E8 items 3–4; they consume §13. **Not in §13, and not in E8:**
* a general `Z/n` (it needs gcd pivoting, not valuation pivoting);
* the CUDA layer (§13.9);
* the `ell`-adic spectral sequence;
* general `(a, b)`;
* `ell^k` beyond a machine word: **§13.11's (C2) `ell`-adic digit induction** (`engine::ladic`,
  the slot-11 arbitrary-`k` mechanism, contract since slot-11 WP-18b) reaches it, with the native
  `u128` element type of §13.12 as its third digit width and oracle; §13.1–§13.10 below are about
  the word-sized direct route only and are not amended by §13.11.

### 13.1 The declared ring

```cpp
static ModCtx<Elt> ModCtx<Elt>::prime_power(Elt ell, uint32_t k);   // new; k >= 2
bool     is_field() const;   // new: true for ModCtx(p), false for prime_power(ell, k)
Elt      ell;                // new: the prime; == p for a field context
uint32_t k;                  // new: the exponent; == 1 for a field context
```

* **The ring is declared, never inferred.**
  * `ModCtx(p)` (§1, `ModCtx.h:151`) keeps its meaning, "`p` is prime". This is a precondition it
    does not test today and will not test. It sets `is_field() == true`, `ell == p` and `k == 1`.
  * `prime_power(ell, k)` (`ModCtx.h:158–177`) is the **only** way to obtain `is_field() == false`.
    It sets `p = ell^k`, then computes every §1 constant (`p2`, `barrett_m`, `mont_*`, `k_max`,
    `regime`) with the existing `init(p)` code, unchanged.
  * Its `ENGINE_CHECK`s: `ell ≥ 2`, `k ≥ 2`, and `ell^k` fits `Elt` (an overflow-checked product).
    The primality of `ell` is a precondition, exactly as for `ModCtx(p)`.
  * `k == 1` is rejected. The field `F_ell` therefore has exactly one representation,
    `ModCtx(ell)`, and "selected by `is_field()` alone" (§13.8) has no ambiguous case.
  * **`chain(ell, k)`** (S24 CL-1, CODEAUDIT C-07; additive) is the entry point for "Z/ell^k, k ≥ 1":
    it returns `ModCtx(ell)` at `k == 1` and `prime_power(ell, k)` for `k ≥ 2`, so the one
    representation of `F_ell` is kept. `ENGINE_CHECK`s: `k ≥ 1`, `ell ≥ 2`, and those of
    `prime_power`. `ModCtx<u128>` (§13.12) is composite-only and has no `chain`. Scalar helpers that
    need no context (`powmod`, `try_invmod`, `powfit`, …) live in the header-only, STL-only
    `ScalarMod.h` (namespace `engine::scalar`).
* **A `ModCtx(p)` built from a composite `p` is a caller error that no code can detect**, because
  the constructor does not test primality. So §13.7 requires the composite ring to travel as a
  `ModCtx`, never as a bare `Elt p`.
* Both constructors write the new members. No existing member changes value for any `p`.
  `sizeof(ModCtx)` grows — **72 → 208 bytes for `uint32_t`, 80 → 608 bytes for `uint64_t`**
  (measured, slot 8 WP-1; almost all of it the `ell^v` table) — and no value depends on it.
* **`init(p)` is public and is re-entered on a live context** (`FlatMatrix.cpp:53`, the
  thread-local `cached.init(p)`). So `init` itself resets `ell = p`, `k = 1` and `is_field()` to
  the field values, and `prime_power` sets its ring members **after** calling `init` (review m5).
  A re-initialised composite context therefore never keeps stale ring data (`ModCtx.h:182–194`).
* **`ModCtx` stays trivially copyable** (review m6): it is copied freely (`BlockMat.h:136`, `:464`,
  `Recorder.h:69`, the executor's `Bindings`). The `ell^v` table is a fixed
  `std::array<Elt, bits>` (`k ≤ bits − 1` because `ell ≥ 2`), never a `std::vector`, and
  `ModCtx.h` `static_assert`s `std::is_trivially_copyable` for both element types.

### 13.2 What carries over unchanged: the modulus-only layer

A statement about `p` alone, rather than about `p` being prime, carries over verbatim to
`p = ell^k`. Each row below is checked against the code:

| Layer | Why it is modulus-only | Code |
|---|---|---|
| `mul`, `add`, `sub`, `neg`, `reduce` (§1) | exact integer product or sum, then one reduction to `[0, p)`; the canonical residue mod `p` is unique for every `p ≥ 2` | `ModCtx.h:234–243`, `:294–297` |
| Barrett `reduce_u64` (u32) | `m = ⌊2^64/p⌋`, and `x − ⌊xm/2^64⌋·p ∈ [0, 2p)` holds for every `p ≥ 2` | `ModCtx.h:199`, `:299–309` |
| Shoup (u32, `p ≤ 2^31`) | the precomputed quotient `⌊w·2^32/p⌋`; the bound `2p ≤ 2^32` is a bound on `p` only | `ModCtx.h:318–337` |
| Montgomery-64 (u64) | needs `p` **odd** and `< 2^63`, nothing else. `ell` odd ⇒ `ell^k` odd. **`ell = 2` ⇒ `p = 2^k` is even ⇒ `Regime::U64_Wide`**: the exact `__int128 %` path, correct and slow, by the existing rule | `ModCtx.h:216–228` |
| `k_max` and the regimes (§1, §5) | `⌊(2^64−1)/(p−1)^2⌋` or its u128 analogue; the regime thresholds are on `p` | `ModCtx.h:200–205`, `:212–215` |
| GEMM, GEMM_NT, GEMV, `dot_delayed` (§3, §5) | exact accumulation of at most `min(k_max, 256)` products, then one reduction per k-block; `kb ≤ k_max` is `ENGINE_CHECK`ed for every block | `Kernels.cpp:360–363`, `:391–394`, `:470`, `:496–499` |
| the AVX-512 FP64-quotient epilogue (§6) | exact for `p ∈ [2^13, 2^32)`, which is a bound on `p`; below `2^13` the portable Barrett epilogue runs (e.g. `5^2`, `5^3`) | `Kernels.cpp:225–236`, `:295` |
| `axpy`, `scal` (Shoup or Barrett, chosen by `shoup_available()`) | as above | `Kernels.cpp:531`, `:557` |
| `trmm` with a **unit**-triangular tag, `apply_signed_antidiag` (`{1, p−1}`), `apply_diagonal`, `copy`, `transpose`, `canonicalise`, `swap_rows` | nothing is inverted anywhere; the diagonal of a unit-triangular factor is taken as 1 (§4) | `Kernels.cpp:577–585` |
| `DenseBuf`, `DenseView` (§2), structure tags (§4) | storage and index arithmetic | — |
| `engine::elim::eliminate_against`, `sub_product` (§8) | a gather and one `gemm(Sub)`; no pivot is chosen and nothing is inverted | `DenseElim.cpp:673`, `:690` |
| block kernels `blk::gemm`, `gemm_dense`, `add`, `scale`, `equal*`, materialisation (§9.3) | read `A.ctx()` and call the §3 kernels | `BlockKernels.cpp:79`, `:153`, `:198` |
| the tape's arithmetic nodes (§10.1: `GEMM`, `BATCHED_GEMM`, `SCHUR_UPDATE`, `AXPY`, `SCAL`, `TRANSPOSE`, `CANONICALIZE`, `GATHER`, `SCATTER`, `COMPARE`) and passes 1–7 | pass 1 folds coefficient products under the recording modulus; pass 4 records `k_bound = min(k, k_max)`; everything else is structure | `Executor.cpp:133`, `:188`, `:195`, `:202`, `:218–221`; pass 4: `Passes.cpp:540` |

`TRSM` (§10.1) is modulus-only **as a node**: the host driver has already inverted the pivot
factor. But its only producer is the field elimination driver (§13.3).

So the whole assembly side of the pipeline needs no kernel change over `Z/ell^k`: `A_Cq` and
`A_Zq` from given block entries, `Θ`'s kind-GEMMs, and `D·C`. The plan's "GEMM / GEMV / TRMM /
transpose and the whole tape need no change" holds **for the arithmetic**. Finding F4 (§13.10)
names one input for which it does not hold: the `ρ_n` entries.

### 13.3 What must refuse a composite context: the field-only list

An entry point is **field-only** if it inverts a pivot, reduces an exponent modulo `p − 1`, or
defines its output as an RREF. E8 item 2 adds one line at the entry of each:

```cpp
ENGINE_CHECK_MSG(ctx.is_field(), "<name>: field-only; use the Z/ell^k entry points of ENGINE_API.md §13");
```

(The code spells `§13` as `Sec. 13`, keeping the sources ASCII.) **It adds nothing else to their
bodies.** The check is O(1) and throws `std::logic_error`, so a misuse is a reported failure, not
a garbage value.

| Entry point | Why field-only | Code today |
|---|---|---|
| `ModCtx::pow(a, e)` | `mod_pow` reduces the exponent **modulo `p − 1`** (Fermat). That is wrong for a composite `p`, on units (the unit group of `Z/ell^k` has order `ell^{k−1}(ell−1)`) and on nilpotents alike. There is no negative-exponent `pow`: this, not "pow-by-negative", is the reason. **Precisely (review m4):** the reduction is a no-op for `e < p − 1`, so `pow` is correct on any modulus there and wrong only for `e ≥ p − 1`. The guard is nevertheless **wholesale** — on the entry point, for every exponent — as a policy: an exponent-conditional guard would be a different contract. | `ModCtx.h:253–256` → `ModArithmetic.cpp:85` (u32), `:275` (u64) |
| `ModCtx::batch_inv(x, out, n, zp)` | `ZeroPolicy` treats only `0` as non-invertible. A non-zero non-unit makes the running product a non-unit (the comment at `ModCtx.h:380` says this is "impossible for prime p"), and `MapToZero` cannot express that case. **Value-safe even unguarded (review m13):** on a non-unit it throws through `inv(acc)`, the Euclidean algorithm; guarding it is a policy choice, consistent with `pow`. | `ModCtx.h:367` |
| `elim::rref`, `column_rank_profile`, `rref_with_transform` | the pivot is the topmost **non-zero** entry; it is inverted (`ctx.inv`, `DenseElim.cpp:364`) and normalised by `batch_inv` (`DenseElim.cpp:466`) | `DenseElim.cpp:531`, `:543`, `:557` |
| `elim::kernel_basis(ctx, R, pivots)` | pure arithmetic, but its input is by definition an RREF (leading 1s), which a composite ring does not have | `DenseElim.cpp:575` |
| `elim::inverse`, `solve_multi_rhs`, `complement_basis` | read their result off an RREF | `DenseElim.cpp:608`, `:623`, `:646` |
| `blk::rref(A)` and `tape::rref(ex, A, …)` (and everything built on them in §9.4 and §10.7) | they reach `elim::rref` (`BlockElim.cpp:243`, `:256`; the `PANEL_RREF` / `PIVOT_SEARCH` nodes at `Executor.cpp:226`, `:232`). `tape::rref` also inverts a scalar pivot **on the host** (`TapeElim.cpp:241`), so it gets its own entry guard rather than relying on `PANEL_RREF`'s. **Both guards are tested since slot-8 WP-3**: a composite `BlockMat` exists through the §13.7 `ModCtx` overload, and `HeckeReductionPipelineValidation.cpp` arm (B) runs `blk::rref` and `tape::rref` on one and asserts the refusal | `BlockElim.cpp:165`, `TapeElim.cpp:118` |

**What is not guarded, and why:**

* **`ModCtx::inv(a)` is not field-only.** It is the extended Euclidean algorithm (`mod_inv`,
  `ModArithmetic.cpp:51–75`; `mod_inv64`, `:245–267`). That algorithm is correct on units for
  every modulus. When `gcd(a, p) > 1` it throws `std::runtime_error("Modular inverse does not
  exist.")`. This is exactly the unit-only meaning §13.4 needs, with no code change (§13.10 F1).
* **`elim::image_basis(A, pivots)`** takes no context and only gathers columns
  (`DenseElim.cpp:596`). Its pivots can only come from a guarded call.
* **Entry points that take a bare `Elt p` declare a field by their signature** and cannot see a
  composite ring. These are the oracle `engine::ref::rref` and `kernel_basis` (§7), and the §9.4 /
  §10.7 calls `blk::factor_basis`, `blk::solve_multi_rhs`, `tape::factor_basis` and
  `tape::solve_multi_rhs`. The last two build `ModCtx<Elt>(p)` at `TapeElim.cpp:426` and `:621`.

**What the unguarded field code does on a composite modulus today.** This is why the guard is a
check and not a silent dispatch. The legacy pivot rule takes the topmost non-zero entry:
* if that entry is a non-unit, `inv` throws `runtime_error`;
* if every chosen pivot happens to be a unit, the row operations are invertible and the result is
  correct. It is then equal to the Howell form (§13.5) **without its zero rows** (review m1).

So the field code never returns a wrong value on `Z/ell^k`. It either succeeds or throws the
*wrong kind* of error. The guard turns that second case into a named `logic_error` at the entry.

### 13.4 Arithmetic a composite context adds

Integers below are the canonical representatives in `[0, ell^k)`.

| Member | Definition | Domain / failure |
|---|---|---|
| `uint32_t valuation(Elt x)` | the `ell`-adic valuation: the largest `v ≤ k` with `ell^v ∣ x`; `valuation(0) = k` | every `x < p` |
| `Elt unit_part(Elt x)` | the integer `x / ell^v`, where `v = valuation(x)`. It is the unique `u ∈ [1, ell^{k−v})` prime to `ell` with `x = ell^v·u`. **`unit_part(0) = 1`** by convention (`0 = ell^k·1`) | every `x < p`; the result is a unit of `Z/ell^k` |
| `Elt inv(Elt u)` | unchanged (§13.3): the inverse on units | `valuation(u) > 0` ⇒ `std::runtime_error("Modular inverse does not exist.")`, like `mod_inv(0)` |
| `Elt divide_exact(Elt x, Elt d)` | for `d = ell^v` with `0 ≤ v ≤ valuation(x)`: the integer `x / d ∈ [0, ell^{k−v})`. It is the canonical one of the `ell^v` solutions `y` of `d·y = x`, which are unique modulo `ell^{k−v}`. (`v = k`, i.e. `d = ell_pow(k) = p`, is admissible only for `x = 0`, and gives `0`.) | `ENGINE_CHECK`: `d` is a power `ell^v` with `v ≤ k` and `v ≤ valuation(x)` |
| `Elt ell_pow(uint32_t v)` | `ell^v` as an integer, for `v ≤ k`, from a table built at construction (a fixed `std::array`, §13.1). `ell_pow(k) = p` is **not** a residue; it is returned only for bookkeeping | `ENGINE_CHECK v ≤ k` |

On a field context these members are defined, and consistent with `k = 1`: `valuation(x) = 0` for
`x ≠ 0`, `valuation(0) = 1`, `unit_part(x) = x`, `ell_pow(0) = 1` and `ell_pow(1) = p`. No
field-path code calls them. They are at `ModCtx.h:264–291`.

### 13.5 The Howell form (`engine::elim::howell`)

```cpp
template<class Elt> struct HowellResult {
    DenseBuf<Elt>          H;        // h x n, row-major: the non-zero rows of the Howell form
    std::vector<uint32_t>  pivots;   // h leading columns, strictly increasing
    std::vector<uint32_t>  pivot_v;  // h valuations: row i leads with ell^{pivot_v[i]}, 0 <= v < k
};
template<class Elt> HowellResult<Elt> howell(const ModCtx<Elt>& ctx, DenseView<const Elt> A);  // ENGINE_CHECK(!ctx.is_field())
```

Built in `DenseElim.h` / `DenseElim.cpp` (`howell` at `DenseElim.cpp:820`, the algorithm below in
`howell_rows`, `:722`). `HowellResult` also carries `rows()`, the number of pivots.

**Definition** (Howell 1986, "Spans in the module `(Z_m)^s`"; Storjohann and Mulders 1998). Let:
* `R = Z/ell^k`;
* `A` be an `m × n` matrix over `R`;
* `S = rowspan(A) ⊆ R^n`;
* for a column `j`, `S_{≥j} = {x ∈ S : x_1 = … = x_{j−1} = 0}`. Columns are 0-based in the code and
  1-based here.

`H` is **the Howell form of `S`** if and only if:

1. **Echelon:** each row of `H` is non-zero, and the leading (first non-zero) columns
   `j_1 < … < j_h` strictly increase.
2. **Pivots are canonical divisors:** row `i` leads with exactly `ell^{v_i}`, `0 ≤ v_i < k`.
3. **Reduced above:** for every pivot `(i, j_i)` and every row `i' < i`,
   `H[i'][j_i] ∈ [0, ell^{v_i})` as an integer. Entries below a pivot are `0` by condition 1.
4. **Howell property:** for every `j`, the rows with leading column `≥ j` generate `S_{≥j}`.

`h ≤ n`, and `h` may exceed `m`: the single row `(2, 1)` over `Z/4` has Howell form
`[[2, 1], [0, 2]]`. Condition 4 is what an echelon form alone lacks. `rowspan[[2, 1]]` contains
`2·(2, 1) = (0, 2)`, and without condition 4 no row with leading column 2 would generate it.

**Uniqueness (the claim WP-10 re-derives).**
1. For each `j`, the set `I_j = {x_j : x ∈ S_{≥j}}` is an ideal of the chain ring `R`. So
   `I_j = ell^{v}R` for a unique `v ∈ [0, k]`.
2. By condition 4, `j` is a pivot column if and only if `I_j ≠ 0`. Its pivot is then the generator
   `ell^{v_j}` of `I_j`; condition 2 fixes the unit.
3. The row with pivot column `j` is an `x ∈ S_{≥j}` with `x_j = ell^{v_j}` and every later pivot
   entry `x_{j'} ∈ [0, ell^{v_{j'}})`.
4. Suppose two such rows `x ≠ y`. Then `d = x − y ∈ S_{≥j+1}`. At its first non-zero column `j'`,
   `d_{j'} ∈ I_{j'}`, so `j'` is a pivot column and `ell^{v_{j'}} ∣ d_{j'}`. But
   `x_{j'}, y_{j'} ∈ [0, ell^{v_{j'}})` as integers, so `d_{j'} = 0`, a contradiction.

Hence **`H` is a function of `S` alone**: of the row space, not of `A`, its row order, or any pivot
choice.

**The algorithm and its pivot rule ("valuation pivoting").** The forward pass walks the columns
`c = 0, 1, …`, keeping an ordered list of *active* rows (initially the rows of `A`, top to bottom).
For each column:

1. Among the active rows, take the entries of column `c` of **minimal valuation** `v`. On a tie,
   take the **topmost** such row: this is the legacy rule of §7 and §8, with valuation as the first
   key. If no active entry is non-zero, `c` is not a pivot column.
2. Scale that row by `inv(unit_part(entry))`, so that it leads with `ell^v`. It leaves the active
   list and becomes the pivot row of `c`.
3. Every other active row has `valuation ≥ v` in column `c`. Subtract
   `divide_exact(entry, ell^v)` times the pivot row from it (exact).
4. **Howell closure:** if `v > 0`, append `ell^{k−v}·(pivot row)`, which is zero in column `c`, at
   the **bottom** of the active list.

Then reduce above, taking pivot columns left to right. For each pivot `(i, j_i)` and each row
`i' < i`, write `H[i'][j_i] = q·ell^{v_i} + r` with `r ∈ [0, ell^{v_i})` (integer division), and
subtract `q` times row `i`. This changes only columns `≥ j_i` of row `i'` — it sets column `j_i` to
`r` (review m2) — and later pivots reduce the columns beyond it in turn.

*Correctness.* Let `L_c` be the active list entering column `c`. The invariant is
`span(L_c) = S_{≥c}`.
* **Base:** `L_0` is the rows of `A`.
* **Step:** write the rows of `L_c` as `P` (the pivot) and `A_t = A'_t + s_t·P`, where the `A'_t`
  are the eliminated rows. Take `x = Σ b_t A_t + b·P ∈ span(L_c)` with `x_c = 0`. Then
  `(b + Σ b_t s_t)·ell^v = 0`, so `b + Σ b_t s_t = ell^{k−v}·w`, and
  `x = w·ell^{k−v}P + Σ b_t A'_t ∈ span(L_{c+1})`. Conversely `L_{c+1} ⊆ S_{≥c+1}`.
* **Consequences:** column `c`'s pivot therefore generates `I_c`. Unwinding
  `span(L_c) = span(P_c ∪ L_{c+1})` gives condition 4 for the pivot rows. Reduce-above is a
  unimodular triangular change: it preserves every `span(rows with leading column ≥ j)` and
  produces condition 3.

**At `k = 1`** every non-zero entry has valuation 0, so no closure row is ever added, every pivot is
1, and "reduced modulo 1" means 0. The Howell form is then the RREF without its zero rows. The
pivot **selection key** is the legacy one, but the **active-list discipline is not** (review m1):
the legacy PLU and `ref::rref` swap the pivot row into place (`DenseElim.cpp:363`,
`RefLinAlg.cpp:83–85`), which permutes the remaining rows, while the Howell pass removes the pivot
row and keeps the order of the rest. For rows `(0, a)`, `(0, b)`, `(1, ·)` the legacy pass takes
column 2's pivot from `(0, b)`, the Howell pass from `(0, a)`. The outputs agree by uniqueness;
the intermediate states and operation counts do not. This is the oracle cross-check of §13.9.

**No choice above is visible in the output**, by uniqueness, exactly as blocking and panel widths
are invisible for the RREF (§8, §9.4). The rule is fixed anyway, so that intermediate states,
operation counts and the oracle are reproducible. A blocked or parallel implementation must equal
`ref::howell` entrywise: no tolerance, and no "equivalent form".

**Overflow.**
* All entries stay canonical, `< ell^k`.
* `valuation`, `unit_part`, `divide_exact` and the reduce-above quotient are integer operations on
  `Elt`; products go through `ctx.mul`.
* Row updates are AXPYs, which are modulus-only (§13.2). A blocked trailing update may use
  `gemm(Sub)` under the same `k_max` bound as §5.
* **(slot-11 WP-12, F-1)** A factor applied to a whole row or column — the AXPY multiplier, the
  scaling unit's inverse, the closure factor `ell^{k−v}`, the Smith row / column multipliers of
  `invariant_factors` and `cyclic_decomposition` — goes through the §1 reused-operand pair
  (`mul_prep` once, `mul_prepped` per entry), which equals `ctx.mul` bit for bit. In the `u64`
  `U64_Montgomery` regime (odd `ell^k < 2^63`) that is one Montgomery-64 REDC per product instead
  of a `__umodti3`; REDC's precondition `T < p·R` holds because both factors are canonical
  (`T < p²`). No accumulation is delayed, so §5's `k_max` is not involved. `u32` and `U64_Wide`
  run literally `ctx.mul`. Every output is unchanged by construction; `ModCtxMontgomeryPairValidation`
  checks it entrywise (against the same context with the Montgomery flag cleared — the
  pre-package path — against `ref::howell`, and `u64` against `u32` below `2^32`, on the engine and
  on `EquivariantCohomologyOnK`'s module route). Indicative (shared box, one thread): `kernel_module`
  1.35–1.5× faster at `691^6`, `5^27`, `134217757^2` (WP-12 report §3). `RefLinAlg` is **not**
  changed: the oracle keeps its independent arithmetic.

### 13.6 Derived entry points (`engine::elim`, each with `ENGINE_CHECK(!ctx.is_field())`)

Each result is a unique function of its input, with its rule stated. None of them is the D5b basis
of §8, and none claims to reduce entrywise to an `F_ell` object (§13.10 F3). All are declared in
`DenseElim.h:174–247` and built in `DenseElim.cpp:696–1015`; each cites this section at its definition.

**The names (review m11).** `solve_multi_rhs_module` and `length` go beyond [X] E8 item 2's wording,
and are the names built. An overload with `solve_multi_rhs`'s existing signature
`(ctx, DenseView, DenseView)` would have to dispatch on `is_field()`, which §13.8 forbids; `length`
is the bookkeeping E8 item 3 requires. Two conveniences exist beside the table:
`length(ctx, const HowellResult&)` (the same sum, read off a form already in hand) and
`invariant_factors(ctx, A)` (the case `Rl = ∅`). [X] adopts the names in v1.12.

| Entry point | Result (the rule) |
|---|---|
| `HowellResult howell(ctx, A)` | as in §13.5, of `rowspan(A)` |
| `HowellResult kernel_module(ctx, A)` | the Howell form of `K = {x ∈ R^n : A x = 0}`: the **right** kernel, as for `kernel_basis` in §8. Equivalently, take the rows of `howell([Aᵀ | I_n])` whose leading column is `≥ m`, restricted to their last `n` columns. The Howell property (condition 4) says those rows generate `{(0, x) : xAᵀ = 0}`, and conditions 1–3 carry over to the restriction. |
| `HowellResult image_module(ctx, A)` | the Howell form of the **column** module `{A x} ⊆ R^m`, that is `howell(Aᵀ)`. Each row is one generator, a vector of length `m`. The rows are not a subset of the original columns: the `image_basis` rule of §8 has no analogue over `R` that is both unique and minimal. **Nor are the Howell rows a minimal generating set** (review m12): `[[2, 1], [0, 2]]` over `Z/4` has 2 rows for a cyclic module. Nothing here promises minimality. |
| `std::vector<std::vector<Elt>> solve_multi_rhs_module(ctx, B, Y)` | for each column `y_t` of `Y` (`m × ny`), the coordinates `x_t ∈ R^{nb}` with `B x_t = y_t`, **reduced modulo `kernel_module(ctx, B)`**: going left to right over the kernel's pivot columns `j`, `x_t[j] ∈ [0, ell^{v_j})`. The reduced representative of a coset `x + K` is unique (the uniqueness argument of §13.5, applied to `d ∈ K`). The layout is as in §8: `result[t][j]` is the coefficient of column `j` of `B`. If some `y_t ∉ im B`, it throws `std::runtime_error("Vector is not in image of basis (right-hand side t).")`, the §9.4 message, with `t` the **smallest** such index (review m8; for the legacy code this follows from the RREF of `[B | Y]`, `DenseElim.cpp:636–638`). With `ker B = 0`, the result is the unique solution. |
| `uint32_t length(ctx, A)` | the composition length of `rowspan(A)`: `Σ_i (k − v_i)` over its Howell pivots, so `|rowspan(A)| = ell^{length}`. This quantity replaces `dim` in E8 item 3's bookkeeping. Its rank–nullity identity is `length(kernel_module(A)) + length(image_module(A)) = n·k`. |
| `std::vector<uint32_t> invariant_factors(ctx, G, Rl)` | the exponents `e_1 ≥ e_2 ≥ … ≥ 1` with `M ≅ ⊕_i Z/ell^{e_i}`, where `M = (colspan G + colspan Rl) / colspan Rl`. Columns lie in `R^m`, following the `complement_basis` convention of §8, and `M = colspan G / colspan Rl` whenever `Rl ⊆ span G`. `invariant_factors(ctx, A)` is the case `Rl = ∅`. The result is unique by the structure theorem for finite `Z/ell^k`-modules. **Rule:** see below. |

**The rule for `invariant_factors`:**
1. Present `M = R^s / Rel`. The `s` generators are those of `N = colspan[Rl | G]`, from its Howell
   form. `Rel` is generated by the `kernel_module` of that generator matrix together with the
   coordinates (from `solve_multi_rhs_module`) of the columns of `Rl`.
2. Take the Smith form of the relation matrix over `R` by **minimal-valuation pivoting**: smallest
   valuation anywhere, then leftmost column, then topmost row. The built elimination (slot 8 WP-1)
   follows review M3's proposal: it clears the pivot's row first (column operations) and its column
   second (row operations), each by the `divide_exact` quotient times the inverse unit part of the
   pivot, and scales the pivot to `ell^σ`; the exponents do not depend on this, the future
   cyclic-summand bases (E8 item 3) will.
3. Read off the exponents:
   * each diagonal entry `ell^{σ}` with `1 ≤ σ < k` gives `e = σ`;
   * each of the `s` generators (rows of the relation matrix) whose diagonal entry is zero or
     missing gives `e = k`;
   * unit diagonal entries give nothing.
4. Check: `Σ e_i = length(N) − length(colspan Rl)` (`ENGINE_CHECK`ed on every call).

**The invariant factors are not the Howell pivots.** The pivots give only the *length*. Over `Z/4`,
`rowspan[[2, 1]] ≅ Z/4`, since it is generated by an element of order 4. Its Howell pivots
`(2, 2)` would read as `Z/2 ⊕ Z/2`. The Smith step is required (§13.10 F2).

Bases of the cyclic summands are **not** unique, and §13 promises none. E8 item 3's "a basis of
each cyclic summand" needs its own stated rule before it can be golden.

**The unit-pivot read-off and the module-stream lever (slot-23 G2-A / G2-B, `slot23/G2_plan.md` §2.1–2.2).**
`ReadoffStatus solve_unit_pivot_readoff(ctx, B, piv, Y, X, row_block = 256)` is a caller-side helper beside
`solve_multi_rhs_module`, which is untouched. When `B[piv, :] = I_nb` (the columns of `B` are Howell rows with unit
pivots: reduce-above makes every other row vanish at a unit pivot), `B` is injective and the unique solution of
`B X = Y` is `X = Y[piv, :]`; with `ker B = 0` that is `solve_multi_rhs_module`'s reduced representative entry for
entry. The helper checks `B[piv, :] = I` (O(nb²)), reads `X` off, and recomputes the residual `Y − B X` exactly over
**every** row (`kernels::gemm` in row blocks, never the whole product). `Ok`: `X` is the solution. `NotIdentity` (or a
malformed `piv`) and `Residual` (some `y_t ∉ im B`): the caller takes `solve_multi_rhs_module`, which then throws its own
§13.6 message, so even the error text is the legacy one. Callers: `EquivariantCohomologyOnKT::compute_cohomology_module()`
(`X_q`, with `ModuleDegree::piv` = the unit Howell pivots of `C^q_eq`; empty on the non-unit branch),
`EquivariantCohomologyOnKT::H_class_coordinates()` (the flat → `C^q_eq` coordinates of `Y = Θ·H`) and
`HeckeOperatorOnKT::compute_module()` (`Θ_eq`, [C, obl:welldef](iii)). In the same package `cyclic_decomposition`'s
column clear ((c2), the pivot's row) runs i-outer / j-inner over row-major storage: column `t` is not written by it and
the quotients of row `t` are taken first, so every entry receives the same single update with the same operands.
`module_stream_enabled()` (env **`HECKE_MODULE_STREAM`**, default on; exactly `0` = the legacy route, any other value
warns once on stderr and stays on; `ScopedModuleStream(bool)` overrides process-wide) selects the read-off, the `d∘d`
GEMM of the cohomology and the interchanged clear together. It cannot change a value: every route computes a unique
function of its input in exact arithmetic (`HeckeModuleStreamValidation`). `module_stream_totals()` counts the steps
taken (`solve=readoff|howell|legacy`, `dd=gemm|scalar`, `cyc=interchanged|legacy`); `module_stream_effective_text()` is the
`EFFECTIVE HECKE_MODULE_STREAM …` line; when the variable is set, the first call of `module_stream_enabled()` registers that
line to be printed once at process exit with the final counters (the EFFECTIVE convention). `ScopedReadoffCorruption` is a
validation-only negative control (one read-off entry +1 before the residual). `product_is_zero(ctx, A, B, gemm)` is the
`d∘d` predicate `A B = 0` (one GEMM, or the legacy scalar loop), factored out so that it can be tested on planted inputs.

**The product read-off and the stream counters (slot-23 G2-C / G2-D / G2-E, `slot23/G2_plan.md` §2.3–2.5, report
`slot23/G2CDE_report.md`).** `solve_unit_pivot_readoff_product(ctx, B, piv, A, G, X, row_block = 256, perturb_row = ~0u)` is
the same read-off for `Y = A G` with `Y` never materialised: `Y` is formed in row blocks, its pivot rows go straight into `X`,
only its `m − nb` non-pivot rows are kept, and the residual `Y − B X` is recomputed exactly over every row (a pivot row
against its copy, which the validation-only `ScopedReadoffCorruption` hook does not touch). The statuses and the fallback
contract are `solve_unit_pivot_readoff`'s; `perturb_row` is a validation-only hook (+1 at `Y(perturb_row, 0)`). Caller:
`HeckeOperatorOnKT::compute_module()`'s Θ_eq on the stream route (`B = G = C`, `A = Θ`), where it replaces `W = Θ·C` (a
`dim × c_q` buffer) by its non-pivot rows. The lever `HECKE_MODULE_STREAM` also selects, in the Hecke layer, the coset-outer
Θ assembly with per-coset Ω and memos, the grouped low-rank GEMM (`hecke::lowrank::sum_of_low_rank`, which now takes a
`GemmOp`; `Add` accumulates), the chunked route 2, `hecke::lowrank::FastValuation` (the ℓ-adic valuation by multiplication
with `ℓ^{-1} mod 2^W`, equal to `ModCtx::valuation` on every residue; `factor_low_rank`'s scan) and the lean retention
(`hecke::ModuleRetention::Lean`, `EquivariantCohomologyOnKT::release_module_intermediates()`). `module_stream_count(event, n)`
feeds the new `ModuleStreamTotals` fields, printed on the `EFFECTIVE` line as `omega=streamed:<h>,table:<i>
lr=grouped:<j>,scalar:<k>,early_flush:<f> route2=blocked:<l>,full:<m> w=nonpivot:<o> retention=lean:<r>` (`early_flush`: the
grouped GEMMs flushed at the stacked-width cap, a subset of `grouped`; `route2=blocked` counts σ block **rows**, 23 per operator,
not chunks).

**The lean work buffers of `cyclic_decomposition` / `invariant_factors` (slot-23 G2-G, report `slot23/G2G_report.md`).** On the stream route
(`module_stream_enabled()`) both entry points compute the same stores, Howell forms, relation matrix and operations as before, with fewer
buffers live: `N = [Rl | G]` is never formed (the store `rows_of(Nᵀ)` that `image_module(N)` builds is filled from `Rl` and `G` directly);
`Nh.H` is dropped once copied into `Γ` (`length(N)` reads `pivot_v` only); **one** Howell form of `[Γᵀ | I_s]` serves both
`kernel_module(Γ)` and the coordinates of `Rl` (each of those entry points computes exactly that form; `kernel_module` and
`solve_multi_rhs_module` now call the shared steps `kernel_from_augmented` / `solve_from_augmented`, code motion only); the kernel rows, the
coordinates and the relation matrix are dropped once read; `length(colspan Rl)` is taken right after the `s = 0` return (G2-G audit n2:
so after `N`'s elimination, as on the legacy route; before it was taken first). In `invariant_factors` `Gen` is not formed
either (`[Genᵀ | I]` is built from `Nh.H`). Exceptions and their texts are unchanged (the same checks in the same order; the early
`length(colspan Rl)` cannot throw on a valid input). `HECKE_MODULE_STREAM=0` restores the legacy buffers. Counters
`ModuleStreamTotals::cyc_lean`, `cyc_lean_legacy`, `invf_lean`, `invf_legacy` (per call), printed as `cyc_buffers=lean:<a>,legacy:<b>
invf=lean:<c>,legacy:<d>` at the end of the `EFFECTIVE HECKE_MODULE_STREAM` line. `cyclic_decomposition` reads the lever **once** per call
(the G2-B column-clear order follows the same read; audit n1). **Validation only:** `ScopedLeanBufferCorruption(true)` (G2-G audit m2)
adds 1 to the relation matrix's entry (0, 0) on the lean route of both entry points (the legacy route ignores it), so that
`HeckeModuleStreamValidation` (C) can show that a lean-only defect is caught; process-wide, not nestable across threads.

**The module route's tau table without its pool (slot-23 G2-F, report `slot23/G2F_report.md`).** On the stream route `hecke_ext::RunModuleT`
builds each operator's `hecke::HeckeTauTableT` as `TauTableContents::ChainsOnly`: the same chain records (steps 1–3 of [C, alg:tau], the same
order, so the same bytes, B_tab, B and regime) and **no** xi-term table and **no** ρ_n pool (dense or factored) — the module route reads
`tau()` and the exact coset data only and evaluates ρ_n mod ℓ^k itself. `certify()` runs every clause but (f) and (a)'s length match, which
are about the objects not held; the field-route accessors (`xi_terms`, `rho_pool*`, the pool corrupter) throw a named `std::logic_error`.
`HECKE_MODULE_STREAM=0` builds the `Full` table. Counters `ModuleStreamTotals::tau_chains`, `tau_pooled` (per table built by `RunModuleT`),
printed as `tau=chains:<a>,pooled:<b>` at the end of the `EFFECTIVE HECKE_MODULE_STREAM` line.

**The module route's term evaluation (slot-23 G2-T, report `slot23/G2T_report.md`).** A second lever, `module_eval_enabled()` (env
**`HECKE_MODULE_EVAL`**, default on; exactly `0` = the G2-F evaluation; any other value warns once and stays on; `ScopedModuleEval` overrides
it), consulted **only** by `HeckeOperatorOnKT::compute_module_stream()` (so it is inert under `HECKE_MODULE_STREAM=0`). On: every ρ_n(g) mod ℓ^k
of the stream route (Ω_i, the T_ℓ terms of route 1 and route 2, the dense terms; SymN fibre) is evaluated by `ComputeSymmetricPowerRowsRing`
(`FlatMatrix.h`) straight into a row-major buffer — the same recurrences and routes as `ComputeSymmetricPowerMatrixRing`, from one Pascal row per
operator (`RingPascalRow`), the [U_a] / [U_b] column steps 8 at a time in a skewed wavefront, the step's division by u folded into its
coefficients so that each entry is ONE exact sum of three products reduced once (u32 with p ≤ 2^31, u64 with p < 2^62; else the step as
written) — and `factor_low_rank(…, fast_scan = true)` finds its pivot by a divisibility test by ℓ^{best_v} (one multiplication per entry),
stops at the valuation floor (the last pivot's valuation, a lower bound for every entry), sweeps an all-zero R once, and updates R −= u vᵀ
by one rank-1 `kernels::gemm` (`GemmOp::Sub`). The same pivots and the same canonical residues as the per-entry scan and loop, so the same
(t, U, V); every entry of ρ_n is the residue of the same ring element. Counters `ModuleStreamTotals::eval_fast`, `eval_legacy` (operator
degrees), printed as `eval=fast:<a>,legacy:<b>` at the end of the `EFFECTIVE HECKE_MODULE_STREAM` line and, when `HECKE_MODULE_EVAL` is set, on
one `EFFECTIVE HECKE_MODULE_EVAL mode=fast|legacy source=env (HECKE_MODULE_EVAL=<v>) eval=fast:<a>,legacy:<b>` line at exit.

**The dispatched SIMD ρ_n column steps (slot-23 G2-T2, report `slot23/G2T2_report.md`).** `ComputeSymmetricPowerRowsRing`'s [U_a] / [U_b]
column steps on a u32 word with p ≤ 2^31 run on a **runtime-dispatched SIMD path** (`FlatMatrix.h`, `RhoFillPath`): AVX-512F (4 zmm = 64
wavefront lanes) if `__builtin_cpu_supports("avx512f")`, else AVX2 (4 ymm = 32 lanes), else the G2-T scalar 8-lane wavefront. The bodies are
compiled under `__attribute__((target(…)))` in `FlatMatrix.cpp`, so **no `-march` flag** is needed and an x86-64-baseline build runs them on a
CPU that has them; non-x86 builds (aarch64 / Jetson) compile the scalar path only (`-DRHO_FILL_PORTABLE_ONLY` compiles that branch on x86).
Each lane step is r_k = s′P_k + t′P_{k−1} + (−w′)r_{k−1} with three **Shoup** products by per-key constants (`ModCtx::shoup_precompute`, a lazy
residue in [0, 2p), 2p ≤ 2^32) brought to [0, p) by `min(r, r − p)`, and two canonical sums the same way: the canonical residue of the same
ring element as the scalar step, so the bytes are equal. u64 words and u32 moduli above 2^31 always take the scalar path. Lever (read-only,
value-neutral): env **`HECKE_RHO_SIMD`** (`0` / `scalar`, `avx2`, `avx512`, `1` / `auto`; an unavailable path or any other value warns once and
the automatic choice stays), `ScopedRhoFillPath` (throws on an unavailable path), counters `rho_fill_counts()` ([U] fills per path) and, when
the variable is set, one `EFFECTIVE HECKE_RHO_SIMD path=<p> source=env (HECKE_RHO_SIMD=<v>) cpu=avx2:<0|1>,avx512:<0|1>
fills=avx512:<a>,avx2:<b>,scalar:<c>` line at exit (registered at load).

**Route 2's memo across chunks (slot-23 G2-T2).** A third lever, `route2_shared_memo_enabled()` (env **`HECKE_ROUTE2_SHARED_MEMO`**, default on;
exactly `0` = G2-D's two σ-row chunks at N_q > 7, verbatim; `ScopedRoute2SharedMemo`), consulted only by `compute_module_stream()`: on, route 2
is built in ONE chunk of all N_q block rows, so its memo key(δ_iγ) → ρ (or (U, V)) spans (degree, coset) and every distinct key of a coset is
evaluated once instead of once per chunk; the route-2 buffer is then N_q d × N_q d (the size of Θ^{(q)}) instead of ⌈N_q/2⌉ d × N_q d. The
same terms summed into the same entries and compared with the same rows of Θ: no value can change. Counters `route2_memo_counts()` (degrees with
N_q > 7 per mode); `EFFECTIVE HECKE_ROUTE2_SHARED_MEMO mode=shared|chunked source=env (…) route2_degrees=shared:<a>,chunked:<b>` at exit when set.

**The parallel module route (slot-23 G2-P, report `slot23/G2P_report.md`).** A fourth lever, `module_parallel_enabled()` (env
**`HECKE_MODULE_PARALLEL`**, default on; exactly `0` = the serial route, verbatim; any other value warns once and stays on;
`ScopedModuleParallel` overrides it process-wide), effective only when `engine::tape::consumer_pool()` has more than one worker — so
`ENGINE_THREADS=1` (the census mode) **is** the serial route. Three places use the pool, each by disjoint writes in a fixed order:
(1) `compute_module_stream()` assembles Θ^{(q)} and route 2 in batches of consecutive cosets — the calling thread lists the batch's distinct
keys per coset (the serial memo scope) in the serial term order, the pool evaluates Ω_i and the keys (pure functions; each worker its own row
buffers; then the fused products Ω_i U or Ω_i ρ), and one task per σ row accumulates ITS rows in the serial coset / term order with the serial
grouping (so every GEMM / axpy of the serial route is made on the same operands; the counters are sums of per-task counts); a batch closes at
16 tasks per worker or 256 MiB of dense buffers; (2) `howell_rows` (every Howell form: the cohomology's `cyclic_decomposition`,
`kernel_module`, the coordinates) splits step 3 (eliminate the other active rows) and the reduce-above over the elimination hook, the target
rows being disjoint and the source row read-only (`hook_tiles` thresholds: ≥ 64 rows, ≥ 2^17 MACs); (3)
`solve_unit_pivot_readoff_product` runs both passes' row blocks as hook tasks (each block's destination rows fixed by `piv`; Residual iff
some block fails, the serial outcome). The hook is a `tape::ScopedHostElimHook` installed by `compute_cohomology_module()` and
`compute_module_stream()` when the lever is on (bound to `consumer_pool()`, honouring `ScopedConsumerPool`); (2) and (3) also require the
lever, so `HECKE_MODULE_PARALLEL=0` leaves them serial under any foreign hook. No value can depend on the thread count or the schedule.
Counters `module_parallel_degrees(parallel)` (operator degrees per path), `module_parallel_howell_regions()`,
`module_parallel_readoff_regions()`; `EFFECTIVE HECKE_MODULE_PARALLEL mode=on|off source=env (…) threads=<largest pool> par=parallel:<a>,serial:<b>
hook=howell:<c>,readoff:<e> elim_hook=on|off[ howell_readoff=serial]` at exit when set (registered at load, also when never consulted).
(G2-P audit F2) The lever is consulted by `compute_module_stream()`, `compute_cohomology_module()`, `howell_rows` and
`solve_unit_pivot_readoff_product` — not by the stream route alone. **`ENGINE_HOST_ELIM_HOOK=0` keeps (2) and (3) serial while (1) stays
parallel** (no elimination hook is installed); the EFFECTIVE line is then printed also without `HECKE_MODULE_PARALLEL` and reads
`hook=howell:0,readoff:0 elim_hook=off howell_readoff=serial`. (G2-P audit F3) Every task body of (1) runs under `elim::ScopedInHookTask`,
so `elim::parallel_hook()` is null in it on the calling thread as on the workers: a callee can never open a nested region from a task.
(G2-P audit F1) `elim::ScopedReadoffWorkerFault` (validation only) injects a residual (mode 1) or a throw (mode 2) into one pass-2 block of
the parallel read-off run by a pool worker other than the caller; `HeckeModuleStreamValidation` (P) (N) gates both outcomes. The extra peak
RSS at k > 1 workers is bounded by k · (32·1024·d + 12·d²) bytes + 256 MiB (u32; the k-term doubles in u64; `performance-reference.md`).

**The module-MT lever (slot-24 MT-1, report `slot24/MT1_report.md`).** A fifth lever, `module_mt_enabled()` (env **`HECKE_MODULE_MT`**,
default on; exactly `0` = the route of before MT-1, verbatim; any other value warns once and stays on; `ScopedModuleMt` overrides it
process-wide). It has two parts. **(0) At every thread count** (owner ruling 2026-10-20, census mode included): `cyclic_decomposition`'s (c2)
pivot search on the lean route (`HECKE_MODULE_STREAM` on) stops at the first unit. The serial pivot is the first entry, in column-major order
from (t, t), of minimal valuation under a strict `<`; a unit has valuation 0, the least possible, so no later entry can replace it, and the
pivot (pr, pc, pv) is the one the full scan returns. This removes the dominant serial cost of the ℓ ≈ 10³ cells (≈ 112 s of 195 s at
(600, 1009) on 16 threads, ≈ 30 % at one thread). **(a)–(e) the splits** need `module_mt_hook()` = `parallel_hook()` with both
`HECKE_MODULE_MT` and `HECKE_MODULE_PARALLEL` on, i.e. the elimination hook that `compute_cohomology_module()` / `compute_module_stream()` install
on a pool of more than one worker. (a) `cyclic_decomposition`: the columns after t in tiles (each tile its first minimum in the serial order,
exit at a unit; combined in tile order with the same strict `<`, which is the serial first minimum); the relation-matrix row operations, the U
rows and the Γ rows over disjoint row tiles (each entry receives the serial sequence of updates: the column-t quotients are taken first, column
t is not written by the row operations, Γ's column t accumulates over i in the serial order per row). (b) `gemm_rows_hook(par, ctx, A, B, C)`:
C = A B in row blocks (bit-identical by §5); used for Y = Θ H and (iii)'s fallback W = Θ C, C TE in `compute_module_stream()`; the H basis
C · dec.basis (the same scalar loop per row) and D C (X_q, `blk::gemm_dense` in column tiles) in `compute_cohomology_module()`. (c) The runtime
assertions (i) / (ii) per column on the pool; if all hold the counters advance by what the serial loops count, otherwise the serial loops run
and throw their own first message. (d) The assembly's accumulation step as one task per (σ row, σ_k block): each block receives the same
GEMM / axpy calls in the same order. (e) `solve_unit_pivot_readoff`'s residual blocks (Residual iff some block fails) and
`solve_from_augmented`'s right-hand sides (the smallest failing t's message is thrown, as serially) as hook tasks. Counters `module_mt_regions(e)`;
`EFFECTIVE HECKE_MODULE_MT mode=on|off source=env (…) regions=cyc_search:<a>,cyc_rows:<b>,gemm:<c>,check:<e>,assembly:<f> pivot_exit=<g>`
at exit when set (registered at load); `pivot_exit` counts the `cyclic_decomposition` calls on the part-(0) search (at every thread count), and
the line ends in `(HECKE_MODULE_STREAM off: inert)` or `(HECKE_MODULE_PARALLEL off: splits inert, pivot-search exit on)` where that applies.
`ScopedModuleMtForceSplit` (validation only) drops (a)'s size thresholds. The per-task buffers of the splits are negligible ((1000, 10007),
16 threads: 2.74 GiB, REF 2.70). The opt-in forms overlap `HECKE_MODULE_MT_FORMS` of the implementation round was removed at the audit
(`slot24/MT1_audit.md` F1). Gated by `HeckeModuleMtValidation`.

**The lean-peak lever (slot-24 G2-H, report `slot24/G2H_report.md`).** A sixth lever, `module_lean_peak_enabled()` (env
**`HECKE_MODULE_LEAN_PEAK`**, default on; exactly `0` = the route of before G2-H, verbatim; any other value warns once and stays on;
`ScopedModuleLeanPeak` overrides it process-wide). It acts only on the stream route (`HECKE_MODULE_STREAM` on) and moves no value: every part
only changes which buffers are live, or replaces a certificate by an exactly equivalent one. (a) `howell_rows`: when no closure row was
appended and every store row is a pivot row, the store's rows are permuted in place (cycle by cycle) into pivot order and the store becomes H
(`DenseBuf::adopt`) instead of being copied; the reduce-above then runs on the same rows in the same order. (e) The lean
`cyclic_decomposition` / `invariant_factors` take length(colspan Rl) from `howell_rows`' pivot loop alone (`pivots_only`: the pivots and
valuations are final before the reduce-above, and `length()` reads only `pivot_v`). Under the lean retention of the sat / Loewy runner
(`RunModuleT`) only: (b) `HeckeOperatorOnKT::compute_module_stream()` certifies (iii) by **`unit_pivot_residual_product(ctx, B, piv, A, G,
row_block, perturb_row)`**, which returns exactly `solve_unit_pivot_readoff_product`'s status without forming X = Y[piv, :] or Y = A G: with
B[piv, :] = I the pivot rows of the read-off's residual vanish identically and its non-pivot rows are (A[np, :] − B[np, :] A[piv, :]) G
(associativity in Z/ℓ^k), computed in tiles of ≤ 128 non-pivot rows with pivot blocks of ≤ 64 (buffers O(d) rows; flops (m − nb) l (nb + ny)
instead of m l ny + m nb ny); the same `NotIdentity` checks first, the validation hooks mapped (`ScopedReadoffCorruption` → `Residual`;
`perturb_row` at a non-pivot row moves that row of the residual, at a pivot row piv[u] moves column 0 by −B[np, u]). So Θ^{(q)}_eq, which
the lean runner never reads, is never formed; any other outcome takes the legacy fallback (W = Θ C and the Howell solve, its texts).
(c) Route 2 is assembled into Θ^{(q)} itself with every coefficient negated (`par_assemble(…, dst_s0, negate)` on the parallel route), after
Y = Θ H and (i)–(iii) have read Θ, and O12(a) is "Θ − route 2 = 0 entrywise": each entry is the same exact sum mod ℓ^k, so the predicate
equals "Θ = route 2 entrywise" on every input; no route-2 buffer (N_q d × N_q d under the shared memo). A cell failing both O12(a) and one of
(i)–(iii) would now report the latter first; every single-fault control keeps its legacy text. (d) `EquivariantCohomologyOnKT::
set_module_early_release(true)` (set by `RunModuleT` before `compute_cohomology()`): the Z^q / H^q loop first computes Zc_q = ker X_q and dK_q
for every q (ascending), then dec_q and H_q in descending q, and frees X_q / Zc_q after their last reader (X_q: ker X_q, dK_q, dec_{q+1} as Rl;
Zc_q: dec_q); the object ends released, as after `release_module_intermediates()`. Counters `lean_peak_events(e)`; `EFFECTIVE
HECKE_MODULE_LEAN_PEAK mode=on|off source=env (…) howell=adopt:<a>,copy:<b> residual=<c> route2_inplace=<e> early_release=<f> length_only=<g>`
at exit when set (registered at load; ` (HECKE_MODULE_STREAM off: inert)` appended where that applies). Tracked peak at (300, 43): 529 → 426
d² (cohomology 511 → 426, operators 529 → 410). Gated by `HeckeModuleStreamValidation` (H).

**The tile-work bound (slot-24 PAR-L1, report `slot24/PARL1_report.md`).** The hook splits of the module routes — `howell_rows` (the
elimination step 3 and the reduce-above), `gemm_rows_hook`, `unit_pivot_residual_product` and `cyclic_decomposition`'s splits (through
`cyc_tiles`) — take `mod_tiles` = `hook_tiles`' extent-based count, further capped at work / `module_tile_work()` and serial below two tiles.
The bound is env **`HECKE_MODULE_TILE_WORK`** (default 2^20 modular MACs per tile; exactly `0` = the extent-only count of G2-P / MT-1,
verbatim; a number ≥ 2 = that bound, a measurement knob; anything else warns once and keeps the default), overridden process-wide by
`ScopedModuleTileWork` (validation use; `ScopedModuleMtForceSplit` still drops (a)'s thresholds and the bound with them). Counters
`module_tile_work_capped()` / `module_tile_work_serial()`; `EFFECTIVE HECKE_MODULE_TILE_WORK mode=bounded|extent source=env (…)
min_tile_work=<w> capped=<a> serial=<b>` at exit when set (registered at load). Why: a region of ~15 tiles of ~8 000 MACs costs more in pool
wake-ups (a futex wake and a futex wait per sleeping worker, ≈ 43 futex calls per region) than its work; at (N, n) = (1, 60) the descent
record's pair modules opened ~170 000 such regions and 16 threads ran 40 % slower than 1. The tape's PLU hooks (`hook_tiles` direct) are
untouched. Tiling is value-neutral (each tile writes disjoint rows by the same operation sequence), so no value can move.

### 13.7 The block model and the tape over a composite ring

* **The ring travels in the context.**
  * `Recorder(ctx)` (`Recorder.h:69`) already stores a full `ModCtx` in the program, so the tape
    carries `is_field()` with no change.
  * A `BlockMat` built through an `Elt p` constructor stores `ModCtx<Elt>(p)` and **therefore
    declares a field for every `p`**. Building a `BlockMat` over `ell^k` through those
    constructors is the undetectable caller error of §13.1.
  * **Decided by E8 item 3 (slot-8 WP-3), recorded here since the slot-8 close** (review WP-12
    pass 1, m-1). This section offered two routes — a `ModCtx` overload on `BlockMat` and
    `BlockBuilder`, or the composite pipeline wholly on `DenseView`s — and forbade only the
    bare-`p` one. **The overload was taken**:
    * `BlockMat::zeros`, `from_flat`, `from_dense` and `init_grid` take a `const ModCtx<Elt>&`
      (`BlockMat.h:171`, `:374`, `:402`, `:552`), and so does `BlockBuilder` (`:659`);
    * the `Elt p` forms are unchanged in signature and delegate on `ModCtx<Elt>(p)`
      (`:396`, `:547`, `:653`), value-identical to what they built before;
    * `blk::gemm` and `blk::add` refuse mixed rings (`BlockKernels.cpp:83`, `:201`,
      `ENGINE_CHECK_MSG(A.ctx().is_field() == B.ctx().is_field(), …)`), and their results are
      built on `A.ctx()` (`:89`, `:204`, `:226`), not on `ModCtx(p)`.

    The `DenseView`-only route is the recorded fallback, not taken. Tested by
    `HeckeReductionPipelineValidation.cpp` arm (B) (the context survives `BlockBuilder` →
    `finalize`, `zeros`, `transposed()` and `blk::gemm`).
* **There is no block or tape Howell form in the first extent.** The `blk::` and `tape::`
  eliminations (§9.4, §10.7) stay field-only through their §13.3 guards. The composite eliminations
  are dense (`engine::elim`, §13.6), as `HeckeModuleDecomposition` is today. If a `PANEL_HOWELL`
  node is ever needed, it is a new host-serial op and is not device-eligible, like `PANEL_RREF`.
  **The cone's dense module route (slot-9 WP-4, E8 item 4b).** `Cone(r)`, the circle-model
  boundary and the route-1 cone action over `Z/ell^k` follow this rule: `ModuleConeT<Elt>`
  (`src/hacs/EquivariantMappingConeModule.cpp`), selected by `is_field()`
  alone inside `EquivariantMappingCone`, runs on the §13.6 entry points only (`howell`,
  `kernel_module`, `solve_multi_rhs_module`, `cyclic_decomposition`, `length`) with the ring
  context's element type and regime (u32 below `2^32`, the `uint64_t` instantiation with
  U64_Montgomery / U64_Wide above it — no new regime, no new bound). The cone's tape route is
  the `F_ell` path and is unchanged ([E] v1.10.0-pre.1 `sec:ring` (f)).
* **Replay and caching.**
  * A replay may substitute a binding context of the same regime (`Executor.cpp:398–407`, §10.2).
    E8 item 2 adds `b.ctx.is_field() == prog.ctx.is_field()` to that check (`Executor.cpp:404–405`):
    a field program never replays on a composite binding, nor the reverse
    (`HeckeReductionValidation`, arm (R)). The check is structural; it reads no hash or key.
  * The consumer cache key already mixes in `p` (`EngineRuntime.h:155`). A prime never equals an
    `ell^k` with `k ≥ 2`, so field and composite programs cannot collide.
  * **Any new hash or key input is mixed in only when `!is_field()`.** So every field program's
    structural hash (`Passes.cpp:194–197`), its `ShapeKey` (`ProgramCache.h:34–43`) and every
    digest stay unchanged byte for byte.
* The legacy substrate (`FlatMatrix`, `BlockFlatMatrix`, `ENGINE_BACKEND=legacy`,
  `-DENGINE_SHADOW_LEGACY`) has **no composite arm**. `ref::howell` takes over its oracle role
  (§13.9).

### 13.8 The bit-identity guarantee of the `F_ell` path

1. **Selection.** A consumer chooses the composite route by `ctx.is_field() == false` **alone**: no
   environment variable, no `ENGINE_*` switch, no other setting.
   * The context itself comes from the mathematical setting `coefficient_ring` ([X] §1.5). Its
     default, `F_ell`, creates `ModCtx(ell)`.
   * No existing function dispatches on `is_field()`. Field entry points gain only the §13.3 guard,
     which is true on every existing call; composite entry points are new names.
2. **What changes on the field path:** only the new members of `ModCtx` (written, and **never read
   on a value path** — the §13.3 guards and the §13.7 replay check read `is_field()`, review m7) and
   the §13.3 guards (never taken). No kernel, pass, pivot rule, basis rule, or hash or key input
   changes (§13.7).
3. **The evidence required** (E8 acceptance (b) and (c)), **before and after** each of E8 items 2–4:
   * `GOLDEN_STRICT=1 GoldenRegressionValidation` at the ledger's counts;
   * both determinism digests unchanged;
   * `--all` green on `RelWithDebInfo` and `Check`;
   * `cross_config.sh RelWithDebInfo Bench Check RelWithDebInfo+HECKE_PURE_INTEGER=ON
     RelWithDebInfo@ENGINE_BACKEND=legacy --validations HeckeReductionValidation,ModContextValidation`
     green. Its `gemm 512` and `rref` checksums, identical across arms, are the kernel-level
     statement.

### 13.9 CUDA (§11–§12) is not extended; the oracle and the validation

**CUDA.**
* No device kernel, no device op and no cost-model constant is added.
* The existing device kernels are modulus-only: Barrett with `ctx.barrett_m`, 64-bit Montgomery,
  `U64_Wide`, and the `kb ≤ min(k_max, 256)` rule (`cuda_kernels.cu:305–306`). Nothing in `Cuda/`
  inverts anything.
* So a composite GEMM on the device would be correct, but **no E8 validation gates it**. The E8
  validations run under the default `ENGINE_DEVICE=cpu`. The elimination stays on the host (§13.7).
* `hybrid` sends every `uint64_t` group to the GPU (`Passes.cpp:772`; the rates `:669–676`; §12.1).
  The rule is **per element type** (review m9): **any `uint64_t` context** — `ell^k < 2^32`
  included, if a consumer chooses `uint64_t` for it — takes the untested device path under
  `ENGINE_DEVICE=hybrid|gpu`, and every `ell^k ≥ 2^32` is a `uint64_t` context. That path is
  value-safe by construction, but it is not a gate.

**The oracle, `engine::ref::howell(ell, k, M)`** (in `RefLinAlg`; naive and never optimised, like
§7):
* It runs the §13.5 forward pass exactly as written (closure rows appended at the bottom), then the
  reduce-above pass, on legacy `mod_*_w` arithmetic with modulus `ell^k`.
* `k = 1` is allowed **in the oracle only**. There, the result must equal `ref::rref`'s non-zero
  rows.
* `ref::invariant_factors` uses the naive Smith form. As built, it is also **independent in its
  presentation**: its generators are the columns of `G` (not the Howell rows of `N`), its
  relations the `G`-coordinates of the kernel of `[G | Rl]`, and its pivot search is row-major (a
  different tie rule on purpose — the exponents are unique).
* The oracle takes a bare `(ell, k)` pair, so it is not an entry point of the pipeline. It returns
  `ref::RefHowell` (`rows`, `pivots`, `pivot_v`); `RefLinAlg.h` / `RefLinAlg.cpp`.

**`HeckeReductionValidation`'s elimination arms** ([X] §5; the pipeline, reduction and Bockstein
arms belong to E8 items 3–4). As built in slot 8 WP-1 (`Hecke/HeckeReductionValidation.cpp`,
`standard`, ≈ 0.5 s; seeds and counts printed), with the moduli the slot-8 plan's WP-1 adds:

* **Ring:**
  * `prime_power` at `(ell, k) ∈ {(5, 2), (5, 3), (691, 2)}` (u32) and `(134217757, 2)` (u64,
    Montgomery) — and `(2, 2)`, `(2, 3)`, `(3, 2)` (u32) and **`(2, 40)` (u64, `U64_Wide`, the
    §13.2 even-modulus route)**;
  * `mul`, `add`, `sub` and `reduce` against `ModArithmetic` on `10^6` fixed-seed samples;
  * `inv` on units, and the named error on every non-unit class (exhaustively where
    `ell^k ≤ 1000`; on `0` and 100 samples of each valuation `1 … k−1` otherwise);
  * `valuation`, `unit_part` and `divide_exact` against integer arithmetic;
  * `prime_power(ell, 1)` is rejected;
  * **every §13.3 entry point throws `logic_error` on a composite context**, and every §13.6 entry
    point throws on a field context (`blk::rref` / `tape::rref`: unreachable until a composite
    `BlockMat` exists, §13.3);
  * the §13.7 replay check, both directions, with same-ring replays exact;
  * `ModContextValidation` carries the composite arms of §1's own checks (the §1 constants equal
    `init(ell^k)`'s, `k_max` tight, `dot_delayed` exact at `k_max`) at `Z/4`, `Z/125`, `Z/691²`,
    `Z/3^20` (`U32_Barrett64`), `Z/2^31` (`U32_Shoup`), `Z/134217757²`, `Z/2^40` and `Z/3^40`
    (`U64_Wide`).
* **Howell:**
  * `elim::howell == ref::howell` entrywise, on fixed-seed and structured matrices: zero, identity,
    `ell`-multiples, rows of the closure-example type `[[2, 1]]`, rank-deficient, tall and wide;
  * **uniqueness:** the form is unchanged under row permutations, under left multiplication by
    fixed-seed unimodular matrices, and under appending rows of the span;
  * `ref::howell(ell, 1, M) == ref::rref(ell, M)` without its zero rows.
* **Derived:**
  * `kernel_module`: `A·x = 0` for every row, and `length(ker) + length(im) = n·k`;
  * `image_module`: the same span as `A`'s columns (mutual membership, checked with
    `solve_multi_rhs_module`);
  * `solve_multi_rhs_module`: `B·X = Y` exactly, `X` reduced modulo `ker B`, and the named throw for
    a right-hand side outside the image;
  * `invariant_factors`: `Σ e_i` equals the length difference; checked against
    `ref::invariant_factors`, and on the `Z/4` example, `[[2, 1]] ↦ (2)`;
  * **review M1's arm, in the column convention above.** `[[2, 1]] ↦ (2)` alone does not test F2:
    the column span of the `1 × 2` matrix `[[2, 1]]` is all of `Z/4` (it contains the unit `1`),
    so an implementation reading the exponents off Howell pivots would pass it too. The arm is
    `invariant_factors(ctx, G = [[2], [1]]) = (2)` for the `2 × 1` column `G`, with the negative
    assertion that `image_module(G)` `= howell([[2, 1]]) = [[2, 1], [0, 2]]` has pivot valuations
    `(1, 1)` — which would misread as `Z/2 ⊕ Z/2`. (`howell(G)` of the `2 × 1` matrix itself has
    **one** pivot, of valuation 0; the arm asserts that too.)

### 13.10 Findings for the owner (from writing §13 against the code)

*Status (slot 8 WP-1):* F1 and F2 are implemented as specified (`inv` unguarded and unit-only;
`invariant_factors` by the Smith step). *Status (slot-8 close, review WP-12 pass 1 m-3):* **F4 is
implemented** by slot-8 WP-3 (`ComputeSymmetricPowerMatrixRing`, `FlatMatrix.cpp:2261`; E8-P5:
division-free Pascal binomials, the unit recurrences, the `diag(ell, 1)` split), and **F3 is
discharged as stated** by [E] `sec:ring` (d): `red_H^q` is a map equality, never a basis
equality.

* **F1: `inv` is unit-only, not field-only.**
  * [X] E8 item 1 (i) lists `inv` among the field-only primitives that `ENGINE_CHECK` on a
    composite context. Point (ii) defines `inv` "on units only (`std::runtime_error` otherwise)".
  * The code is the extended Euclidean algorithm (`ModArithmetic.cpp:51–75`, `:245–267`), and it
    already has (ii)'s meaning for every modulus.
  * §13 follows (ii) and leaves `inv` unguarded. `batch_inv` and `pow` are guarded.
* **F2: the invariant factors need a Smith step.** [X] (iii) says "`invariant_factors` … from the
  Howell form of the row space". The Howell form gives the length, not the isomorphism type (the
  `Z/4` counterexample of §13.6). §13.6 specifies the Smith step.
* **F3: "equal to the `F_ell` run's objects entrywise after reduction"** ([X] §5,
  `HeckeReductionValidation`) cannot hold for bases.
  * Reduce a Howell basis over `Z/ell^2` mod `ell`. The result is not the D5b basis. In general it
    is not even a basis of the `F_ell` kernel: `ker_{Z/ell^2}(A) mod ell ⊆ ker_{F_ell}(A mod ell)`,
    and the inclusion is strict in general.
  * The engine promises unique module forms (§13.6). E8 items 3–4 must state the comparison as an
    equality of submodules, or of maps in stated bases (a change of basis), not as an equality of
    basis matrices.
* **F4: the assembly is modulus-only, but its `ρ_n` entries are not.**
  * `ComputeSymmetricPowerMatrixFM` takes its binomials from `mod_nCr_w` (`FlatMatrix.cpp:1934`,
    `:2015`, `:2030`). That is **Lucas's theorem modulo a prime** (`ModArithmetic.cpp:216–231`), with
    inverse-factorial tables. It is wrong modulo `ell^k` once `n ≥ ell`.
  * [X] E8 item 3 says "`A_Cq` / `A_Zq` assembly is modulus-only (unchanged)". That holds for the
    kernels, but `ρ_n` over `Z/ell^k` needs exact integer binomials reduced mod `ell^k`.
  * This is outside §13's scope, because it is not an engine kernel. It is recorded for item 3.

### 13.11 `ell^k` beyond a word: the (C2) `ell`-adic digit induction (`engine::ladic`) — contract

**Status: contract (slot-11 WP-18b, first commit), written before the code** (rule 21 of the slot-11
plan). The owner's clarification of 2026-09-25 — *"We had agreed ... that induction was the most
efficient and most effective way to implement arbitrary k exponents. Therefore we shall go down that
route."* — makes this route **the** arbitrary-`k` mechanism of the engine. It is generic over the
digit word `W ∈ {uint32_t, uint64_t, u128}`, with the native `u32` / `u64` / `u128` routes of
§13.1–§13.9 and §13.12 as its oracles (§13.11.7). Everything below is **composite-only, CPU, host,
single-threaded, and records no program** (§13.7's "the composite route records no program" carries
over): no tape node, no device path, no fingerprint, no golden object. The pipeline consumer is
WP-18c's (`Hecke/LadicCell`); nothing in the tree calls this route before it, so the section moves no
existing value (§13.8 is untouched: the `F_ell` and the word-sized composite routes are selected
exactly as before). The slot-8 / slot-10 design text this contract replaces is summarised in
§13.11.10 (history).

Notation: `R_K = Z/ell^K`; entries are canonical integers in `[0, ell^K)`; `v(x)` is the
`ell`-adic valuation (`v(0) = K`); "exact division" `x / ell^s` is the integer quotient of a
canonical `x` with `ell^s | x`. Matrices are `m × n`. `A ≡ B (mod ell^j)` is entrywise.

#### 13.11.1 Domain

* `ell` prime, `2 ≤ ell < 2^31` (so every base-`ell` digit is a `uint32_t`, §13.11.2), and
  `K ≥ 2` (the field has exactly one representation, `ModCtx(ell)`, §13.1; `K = 1` is refused as
  `prime_power(ell, 1)` is). There is **no cap on `K` in the engine**; the cap is the grammar's
  ([E] v1.12 `sec:ring`, WP-18c: `k ≤ 4096` and `K·⌈log2 ell⌉ ≤ 8192` bits per cell). Primality of
  `ell` is a precondition, as for `prime_power` (§13.1).
* `ell ∈ {2, 3}` is admitted **at engine level** (the oracle arms use `3^100`); the pipeline serves
  only `ell ≥ 5` by this route (WP-18c: at `ell ∈ {2, 3}` `C^q_eq` is not free and (L1)'s use on
  `d^q` does not apply — the Bredon route of slot 9 is the flat-coordinate one).

#### 13.11.2 The digit layout

* **Digit exponent.** `m = m(W, ell)` is the largest `m ≥ 1` with `ell^m < 2^{w−1}`, `w` the bit
  width of `W` (the Montgomery bound of each width; `m(32, 5) = 13`, `m(64, 5) = 27`,
  `m(32, 691) = 3`, `m(64, 691) = 6`, `m(32, 134217757) = 1`, `m(64, 134217757) = 2`). The
  base is `B = ell^m`. A **test-only** override `m' < m` is admitted (`DigitLayout::make(ell, K,
  m')`); it changes the layout, never a value.
* **Digits.** An entry `x ∈ [0, ell^K)` is `x = Σ_{i<L} x_i B^i`, `x_i ∈ [0, B)`, with
  `L = ⌈K / m⌉` digits; the top digit satisfies `x_{L−1} < ell^{K − m(L−1)}`. One word of type
  `W` per digit.
* **Planes.** A matrix (`engine::ladic::LadicMat<W>`) is `L` **digit planes**: plane `i` is the
  `m × n` row-major word matrix of the `i`-th digits. `L` is **fixed at construction** from
  `(ell, K, W)` and never grown (§13.11.8).
* **Truncation.** Reduction `R_K → R_{k'}` (`2 ≤ k' ≤ K`) is `truncate(k')`: drop the planes
  `≥ ⌈k'/m⌉` and reduce the new top digit modulo `ell^{k' − m(⌈k'/m⌉ − 1)}`. It is exact and costs
  one pass; `red(q)` and every lower-precision oracle are therefore free.
* **The width-independent serialisation.** `ell_digits(i, j)` is the `K` base-`ell` digits of the
  canonical integer, little-endian, each a `uint32_t`. Two matrices over the same `R_K` are equal
  **iff** their `ell_digits` agree entrywise, whatever `W` and `m` each uses. Every cross-width
  and cross-route comparison of §13.11.7 compares this serialisation (or the native word value,
  which has the same serialisation).
* **Arithmetic.** Every operation is on single words `W` and on the **double word** of one digit
  product (`uint64_t` for `w = 32`, `unsigned __int128` for `w = 64`, the fixed two-limb `u256` of
  §13.12.2 for `w = 128`): sums and differences digitwise with carry / borrow; products by digit
  convolution truncated at `L` digits with **lazy carry** (a column accumulator is folded by one
  division by `B` before it could exceed half the double word; every intermediate bound is stated
  at its definition); the top digit reduced modulo `ell^{K − m(L−1)}`. Valuation: the lowest
  non-zero digit `x_i` gives `v(x) = i·m + v(x_i)`. Exact division by `ell^s`, multiplication by
  `ell^s`, and `x mod ell^s` are digit shifts plus one in-digit division / multiplication by
  `ell^{s mod m}` with carry. Unit inverse: Newton, `y ← y(2 − u y)`, from the inverse mod `ell`,
  `⌈log2 K⌉` steps; a non-unit throws `std::runtime_error("Modular inverse does not exist.")`
  (§13.4's meaning).

#### 13.11.3 The precision-split lemma (L1)

**Lemma (L1).** Let `K = k1 + k2`, `k1, k2 ≥ 1`, `A ∈ M_{m×n}(R_K)`, `Ā = A mod ell^{k1}`. Suppose
`U1 ∈ GL_m(R_{k1})`, `V1 ∈ GL_n(R_{k1})` with

```
U1 Ā V1 = D1 := diag(ell^{s_1}, …, ell^{s_r}) ⊕ 0     over R_{k1},   0 ≤ s_1 ≤ … ≤ s_r < k1.
```

Let `Ũ`, `Ṽ` be **any** matrices over `R_K` reducing to `U1`, `V1` (for example their canonical
integer representatives: "digit extension by zero"). Then:

1. `Ũ ∈ GL_m(R_K)`, `Ṽ ∈ GL_n(R_K)`.
2. `R := Ũ A Ṽ = [[D + ell^{k1} E11, ell^{k1} E12], [ell^{k1} E21, ell^{k1} E22]]` with
   `D = diag(ell^{s_1}, …, ell^{s_r})` and some `E_{ab}` over `R_K`.
3. There are `X ∈ GL_m(R_K)`, `Y ∈ GL_n(R_K)` and `S' ∈ M_{(m−r)×(n−r)}(R_K)` with
   `X R Y = D ⊕ ell^{k1} S'`, where `ell^{k1} S'` depends only on `S' mod ell^{k2}`.
4. If `U2 S̄' V2 = diag(ell^{t_1}, …, ell^{t_q}) ⊕ 0` over `R_{k2}` (`S̄' = S' mod ell^{k2}`,
   `0 ≤ t_j < k2`), then the Smith form of `A` over `R_K` is
   `diag(ell^{s_1}, …, ell^{s_r}, ell^{k1+t_1}, …, ell^{k1+t_q}) ⊕ 0`, and the transforms compose:
   `(I_r ⊕ Ũ2) X Ũ · A · Ṽ Y (I_r ⊕ Ṽ2)` is that form, for any lifts `Ũ2`, `Ṽ2`.

So **the elementary divisors of `A` below `ell^{k1}` are fixed by `A mod ell^{k1}`, and the rest are
`ell^{k1}` times those of the word-sized problem `S' mod ell^{k2}`.** Applied recursively with
`k1 = m` (or less), the Smith data at precision `K` is the union of `L` word-sized Smith problems.

*Proof.*
(1) `det Ũ ≡ det U1 (mod ell^{k1})`, and `det U1` is a unit of `R_{k1}`, i.e. `≢ 0 (mod ell)`; so
`det Ũ` is a unit of `R_K`. Likewise `Ṽ`.
(2) `Ũ A Ṽ ≡ U1 Ā V1 = D1 (mod ell^{k1})`, so `R − D1 ∈ ell^{k1} M(R_K)`; `D1`'s non-zero entries
are the `ell^{s_i}` on the diagonal of the leading `r × r` block.
(3) *The pivot block is `D` times a unimodular matrix.* Since `s_i < k1`, row `i` of
`ell^{k1} E11` is divisible by `ell^{s_i}`; let `N` be the matrix with row `i` equal to that row
divided exactly by `ell^{s_i}`. Every entry of `N` is divisible by `ell^{k1 − s_i}`, hence by `ell`.
So `W := I + N ∈ GL_r(R_K)` (it is `≡ I mod ell`) and `P := D + ell^{k1} E11 = D W`.
*Clearing below and right.* The columns of `ell^{k1} E21 W^{−1}` are divisible by `ell^{k1}`, hence
by every `ell^{s_j}`; let `Z` be that matrix with column `j` divided exactly by `ell^{s_j}`, so that
`Z D = ell^{k1} E21 W^{−1}` and `Z P = ell^{k1} E21`. Symmetrically let `Q' ` be `ell^{k1} E12` with
row `i` divided exactly by `ell^{s_i}` (so `D Q' = ell^{k1} E12`, and every entry of `Q'` is divisible
by `ell^{k1 − s_i}`), and `Q := W^{−1} Q'`, so `P Q = D W W^{−1} Q' = ell^{k1} E12`. With
`X := [[I, 0], [−Z, I]]` and `Y := [[I, −Q], [0, I]] (W^{−1} ⊕ I)`, both unimodular (block
triangular with identity diagonal blocks, times `W^{−1} ⊕ I`):

```
X R [[I, −Q], [0, I]] = [[P, 0], [0, ell^{k1} E22 − Z·ell^{k1} E12]],     P W^{−1} = D.
```

Now `Z · ell^{k1} E12 = Z D Q' = ell^{k1} E21 W^{−1} Q'`, so the Schur complement is
`S = ell^{k1} S'` with `S' := E22 − E21 W^{−1} Q'` over `R_K`. As a product with the factor
`ell^{k1}`, `S mod ell^K` depends only on `S' mod ell^{K − k1} = S' mod ell^{k2}`.
(4) `Ũ2`, `Ṽ2` are unimodular by (1) applied at precision `k2`. Then
`(I ⊕ Ũ2)(D ⊕ ell^{k1} S')(I ⊕ Ṽ2) = D ⊕ ell^{k1} Ũ2 S' Ṽ2`, and `Ũ2 S' Ṽ2 ≡ diag(ell^{t_j}) ⊕ 0
(mod ell^{k2})` gives `ell^{k1} Ũ2 S' Ṽ2 ≡ ell^{k1} (diag(ell^{t_j}) ⊕ 0) (mod ell^K)`. The result
is diagonal with entries `ell^{s_i}` and `ell^{k1 + t_j}`, all `< ell^K`, the rest zero; by the
uniqueness of the Smith form over the local principal ideal ring `R_K` (the valuations of the
elementary divisors are the invariants of `coker A`), these are the elementary divisors of `A`. ∎

**Why no look-ahead is needed** (the "delicate part" of the slot-8 design, §13.11.10). The
look-ahead pivot was a property of *one-sided, digit-by-digit* Howell elimination, where a non-unit
pivot at digit `j` may become a unit only after a carry from digit `j + 1`. In (L1) the divisors
`< ell^{k1}` are determined by `A mod ell^{k1}` exactly and everything else is `ell^{k1}` times a
matrix: **no carry crosses a level boundary.** Within a level, every valuation is read off the whole
`ell`-adic number (§13.11.2), never off one digit.

#### 13.11.4 The level driver (`smith_lift`): how (L1) is realised

`smith_lift(A)` runs levels `l = 1, 2, …` with windows `[c_l, c_l + k_l)`, `c_1 = 0`,
`k_l = min(m, K − c_l)` (a test-only `level_exp` option may choose smaller `k_l`; it changes no
value), `c_{l+1} = c_l + k_l`, until `c_l ≥ K` or the active block is zero. It keeps one working
matrix at precision `K` (entry-major digits) and an **active** set of rows and columns (initially all).

* **Invariant entering level `l`:** every active entry is divisible by `ell^{c_l}` (for `l = 1`
  trivially). The active block is `ell^{c_l} S'_l` in (L1)'s notation, **held in place** — it is never
  divided out, so no precision is lost and no lift is taken.
* **(a) The word-native problem.** The window `W_l := (active block / ell^{c_l}) mod ell^{k_l}` (at
  most two digits per entry, one word after the shift) is eliminated **in word arithmetic** over
  `R_{k_l}` — `ModCtx<W>::prime_power(ell, k_l)`, or `ModCtx<W>(ell)` when `k_l = 1` — by
  valuation pivoting: repeatedly the active entry of **smallest valuation, then leftmost column,
  then topmost row** (the rule of §13.6's Smith step), row operations clearing its column, until
  the window's active block is zero. This yields the pivot sequence `(i_t, j_t)` and the valuations
  `σ_t < k_l` — the Smith data of `S'_l mod ell^{k_l}` (the implicit `V` is the completion of the
  unit-triangular pivot rows).
* **(b) The replay at precision `K`.** The same pivot sequence is replayed on the working matrix:
  for pivot `t`, `ENGINE_CHECK(v(a_{i_t j_t}) == c_l + σ_t)`; the pivot row is scaled by the inverse
  of the unit part so it leads with exactly `ell^{c_l + σ_t}`; every other active row with a
  non-zero entry `x` in column `j_t` gets `row −= (x / ell^{c_l + σ_t}) · pivot row` (the division
  exact: `ENGINE_CHECK(v(x) ≥ c_l + σ_t)`); the pivot row and column leave the active set. The
  replay's multipliers are computed at precision `K`, so the replay is a lift `Ũ` of the word
  elimination in the sense of (L1)(1): reduced modulo `ell^{c_l + k_l}` and divided by `ell^{c_l}`,
  each replay operation is the word operation (the multiplier — the quotient by the pivot's
  `ell^{σ_t}` and the inverse of its unit part alike — matters only modulo `ell^{k_l − σ_t}`, where
  the two agree; the unit-part clause added at slot-11 WP-15, the WP-14 review's m-3), so the window stays equal to the word run at every step,
  and the chosen entry has the stated valuation and is minimal in its column.
* **(c) The level's exit check** (`ENGINE_CHECK`, `O(active · L)`): every active entry is now
  divisible by `ell^{c_l + k_l}` — (L1)(3)'s `S = ell^{k1} S'`.
* **Only the row side is materialised.** In (L1)(3) the column operation `Y` changes only the
  pivot rows (`X R = [[P, ell^{k1}E12], [0, S]]` already has the Schur complement in its non-pivot
  rows). So the driver performs **row operations only**; the Smith data is read off without ever
  forming `Ṽ`. The result (`engine::ladic::Smith<W>`):
  * `s` — the exponents `s_t = c_l + σ_t` in pivot order, **non-decreasing** (within a level the
    minimal valuation cannot decrease under the row updates; across levels `σ < k_l`), all `< K`;
    `rank_K() = r = |s|` is `r_K` of the stopping rule;
  * `pivot_row`, `pivot_col` — the pivot positions;
  * `T` — the `r × n` pivot rows of `Ũ A` in pivot order: row `t` is `ell^{s_t}` times a row with a
    `1` in column `pivot_col[t]` and `0` in `pivot_col[t']` for `t' < t` (the rows of `Ũ A` not in
    `T` are zero). `T` generates `rowspan(A)`;
  * on request (`transforms`): `U ∈ GL_m(R_K)` (the replayed row operations, with the pivot rows
    moved to the top in pivot order) and `V ∈ GL_n(R_K)` (the columns `pivot_col[t]` first, completed
    by back-substitution on the unit-triangular part of `T`) with `U A V = diag(ell^{s_t}) ⊕ 0`
    **exactly** over `R_K` — the certificate the oracle arms check.
* **`L = 1` is the direct route.** When `K ≤ m` there is one level and the replay is the word run
  itself; every derived entry point below **dispatches to §13.5–§13.6 on
  `ModCtx<W>::prime_power(ell, K)`** unless the test-only `force_induction` option is set (§13.11.7's
  identity row).

#### 13.11.5 Canonicalisation: each §13.6 entry point from the Smith data

Every result of §13.5–§13.6 except the cyclic-summand basis is a **function of its input module
alone** (§13.5 "Uniqueness"; §13.6). So any generating set of the right module, passed through the
§13.5 algorithm executed in digit arithmetic at precision `K` (the **canonicalisation pass**:
minimal valuation of the *whole* number, ties to the topmost row, the closure row `ell^{K−v}·P` at
the bottom, reduce-above by integer division `x = q·ell^{v} + r`), returns **the** canonical form —
entrywise the one §13.5–§13.6 define. The generating sets are Smith-adapted:

| Entry point (`engine::ladic`) | Generating set / rule |
|---|---|
| `howell(A)` | the rows of `smith_lift(A).T` (they generate `rowspan A`) |
| `kernel_module(A)` | from `T = diag(ell^{s_t}) M'` (`M'` = `T` with row `t` divided exactly by `ell^{s_t}`, a lift; unit upper-triangular on the pivot columns in pivot order): for each non-pivot column `f`, `x_f = 1` and `x_J = −M'_J^{−1} M'_{:,f}`; for each `t` with `s_t > 0`, `x_J = M'_J^{−1}(ell^{K−s_t} e_t)`, `x_F = 0` (back-substitution, no division). These satisfy `A x = 0` and generate `ker A` (a solution `x` of `A x = 0` minus `Σ x_f g_f` has `x_F = 0` and `M' x ∈ ⊕ ell^{K − s_t} R_K`); `ENGINE_CHECK(length(ker) + length(rowspan A) == n·K)` |
| `image_module(A)` | `howell(Aᵀ)` (§13.6's definition) |
| `length(A)` | `Σ_t (K − s_t)` (no pass needed) |
| `solve_multi_rhs_module(B, Y)` | `smith_lift` of `[B | Y]` with pivots restricted to `B`'s columns (the `Y` columns ride along in the row operations and are exempt from the level checks). `y_t ∈ im B` iff its transformed entries vanish in every non-pivot row and the entry in pivot row `t'` is divisible by `ell^{s_{t'}}`; then back-substitution with exact division by `ell^{s_{t'}}` gives a solution, which is **reduced modulo `kernel_module(B)`** left to right over its pivot columns, `x[j] ∈ [0, ell^{v_j})` — the unique representative of §13.6. The smallest failing `t` throws §13.6's `std::runtime_error` text |
| `invariant_factors(A)` | `{K − s_t}`, descending (`colspan A ≅ ⊕ ell^{s_t} R_K`) |
| `invariant_factors(G, Rl)` | `M ≅ R_K^{ng} / π(ker[G | Rl])` (`π` the first `ng` coordinates; `ker` by the generators above): the `smith_lift` exponents `σ` of that relation matrix give `σ` for `1 ≤ σ`, and `K` for each of the `ng − rank` missing diagonals; descending. `ENGINE_CHECK(Σ e = length(N) − length(colspan Rl))` as §13.6 step 4 |
| `cyclic_decomposition(G, Rl)`, `cyclic_coordinates` | **(c1)–(c5) of `DenseElim.h` executed literally in digit arithmetic**: (c1)'s presentation from the canonical `image_module`, `kernel_module`, `solve_multi_rhs_module` above, then (c2)–(c5)'s pivot rule, swaps, scalings and mirrored operations unchanged. The cyclic-summand basis is a *rule*, not a function of the module; (c2)–(c5) is a deterministic function of the canonical (c1) presentation and of exact `R_K` arithmetic, so the result equals §13.6's entrywise wherever both apply |
| `stopping_rule(d0, d1, dim C^0_eq, dim C^2_eq)` | [L] §2.4 on `smith_lift`'s output: certified iff `r_K(d^0) = dim C^0_eq` and `r_K(d^1) = dim C^2_eq` |

Consequently **no second basis-rule family** is introduced (the slot-8 design's concern): every
oracle compares canonical forms, and the one rule-defined object (the cyclic basis) runs the
existing rule.

#### 13.11.6 Entry points and preconditions (`src/ModArithmetic/LadicInduction.h`, header-only)

```cpp
namespace engine::ladic {
template<class W> struct DigitLayout;          // (ell, K, m, L, B, top digit bound); make(ell, K, m_override = 0)
template<class W> class  LadicMat;             // L digit planes; from_word / to_word / truncate / ell_digits / set_ell_digits
struct Options { bool transforms = false; bool force_induction = false; uint32_t level_exp = 0; };
template<class W> struct Smith;                // s, pivot_row, pivot_col, T, level windows, [U, V]
template<class W> struct Howell;               // H (LadicMat), pivots, pivot_v   -- as §13.5's HowellResult
template<class W> struct Cyclic;               // exponents, basis, Gamma, U, position -- as §13.6's CyclicDecomposition
Smith<W>  smith_lift(const LadicMat<W>& A, const Options& = {});
Howell<W> howell(A), kernel_module(A), image_module(A);
LadicMat<W> solve_multi_rhs_module(B, Y);      // ny x nb: row t = x_t (the §13.6 result[t][j] layout)
uint32_t  length(A);  uint32_t length(const Howell<W>&);
std::vector<uint32_t> invariant_factors(A), invariant_factors(G, Rl);
Cyclic<W> cyclic_decomposition(G, Rl);  LadicMat<W> cyclic_coordinates(const Cyclic<W>&, Y);   // ny x t
StoppingRule stopping_rule(const Smith<W>& d0, const Smith<W>& d1, uint32_t dim_c0, uint32_t dim_c2);
}
```

Preconditions (`ENGINE_CHECK`, `std::logic_error`): §13.11.1's domain; operands of one call share
one layout (`ell`, `K`, `m`); shapes as §13.6. `from_word<N>(layout, view)` needs `ell^K < 2^{bits(N)}`
and canonical entries; `to_word<N>()` the same. The only width-dependent line is the word trait
`engine::ladic::Word<W>`, which **derives from WP-18a's `engine::DigitWord<W>`** (`DigitWord.h`:
`mul_wide`, `split`, `mac_split` — the digit product with carry split `a·b + c = q·B + r`) and adds
the level context of the word-native pivot search and, for the two widths with a native double word
(`uint64_t` for `u32` digits, `unsigned __int128` for `u64` digits), the lazy-carry convolution. The
`u128` width has no native double word: its digit products go through `DigitWord<u128>::mul_wide` /
`split` (schoolbook, one split per digit product; since `f90a9ee` (WP-18b) the split is a
normalised long division on 64-bit halves — Knuth's Algorithm D with two 64-bit quotient digits,
whose D3 test is exact for a two-digit divisor, so no add-back step — and WP-18a's 128-step
restoring division survives as `split_restoring`, its oracle; *corrected at slot-11 WP-15*, the
WP-14 review's m-4: this sentence still named the restoring division), its level context is `ModCtx<u128>::prime_power` (so `ell` is odd
at `W = u128`, §13.12.2; a one-`ell`-digit window runs over `Z/ell^2` and accepts only valuation-0
pivots). Also in the header: `multiply(A, B)` (a dense product over `R_K`, the certificate's and
WP-18c's assembly helper), `convert<W2>(A, m')` (re-express a matrix in another width / digit
exponent through the serialisation) and `Ring<W>` (the digit arithmetic itself).

#### 13.11.7 The oracle rule (the owner's width matrix)

Entrywise on the canonical forms — Howell `H`, `pivots`, `pivot_v`, `kernel_module`,
`image_module`, `length`, `invariant_factors`, the cyclic decomposition (`exponents`, `basis`,
`Gamma`, `U`, `position`), `cyclic_coordinates`, the solves and their refusals — compared through
`ell_digits` (§13.11.2); **no tolerance and no "equivalent form"**:

| Digits | Reach of `ell^K` | Oracle |
|---|---|---|
| `1 × 32`, `1 × 64` | `< 2^31`, `< 2^63` | identity: `L = 1` dispatches to §13.5–§13.6; the `force_induction` run equals it |
| **`2 × 32`** | `< 2^64` | **native `u64`** (§13.5–§13.6), on the `HeckeReductionValidation`-type elimination families **and** on the pipeline's own `d^q` (WP-18c's arms extend it) |
| **`4 × 32`**, **`2 × 64`** | `< 2^127` | **native `u128`** (WP-18a's §13.12 layer, merged at `2b0df4f`) — mandatory |
| `3 × 32` vs `2 × 64` vs `1 × 128` | `< 2^127` | mutual, and native `u128` |
| `w` vs `w'`, all pairs | **`≥ 2^127`** (no native oracle) | **cross-width agreement** to the same `ell^K` (`691^20`, `5^60`, `3^100`), and a naive base-`ell` oracle (`engine::ref::ladic`, `LadicRef.h`: independent single-`ell`-digit arithmetic, the §13.5 pass and a two-sided Smith read directly on `A`) on small matrices with **planted** elementary divisors — the planted divisors are the oracle |
| any | any `K` | **truncation**: `truncate(k')` of the input run at `k'` equals the native run at `k'` whenever `ell^{k'}` fits a word; the Smith exponents obey `s(A mod ell^{k'}) = {s_t : s_t < k'}`; the Howell / image forms at `K`, truncated and re-canonicalised at `k'`, equal the native forms at `k'` |
| any | any | **certificate**: with `transforms`, `U A V = diag(ell^{s_t}) ⊕ 0` over `R_K` and `U`, `V` unimodular (`det ≢ 0 mod ell`), on every arm |

Layout invariance is part of the rule: the test-only `m' < m` and `level_exp` choices must give the
identical output on every arm.

#### 13.11.8 `obl:backend`(iv): fixed-width digit planes are not arbitrary precision

Owner ruling of 2026-09-25 (slot-11 plan §10): *the fixed-width digit-plane route of the C2
induction is admitted explicitly; every operation stays in native words, so the no-bignum intent
([IB] OD-3 (c)) is kept.* The engine's reading, stated for the referee (WP-3 / WP-14):
an element of `Z/ell^K` on this route is **`L` fixed-width machine words**, `L = ⌈K/m⌉` **fixed per
call at construction** from `(ell, K, W)` and **never grown**; every primitive is a single-word
operation or one double-word digit product (`2w` bits, a fixed type) with every intermediate bounded
by a stated constant; there is no heap-growing integer type, no normalisation by magnitude, and no
arbitrary-precision library is linked. It is therefore not the arbitrary-precision arithmetic
[C] `obl:backend`(iv) excludes; `K` is a per-cell parameter, fixed before the digit run from the
certified precision (`K = k + e_max`, §13.11.11) and capped by the grammar ([E] v1.12 `sec:ring`:
`k ≤ 4096`, `K·⌈log2 ell⌉ ≤ 8192` bits). *As published* (slot-11 WP-15, [C] **v1.11.0**,
`obl:backend`(iv), "Scope of the exclusion", which carries the WP-3 referee's draft since v1.11.0-pre.2,
reworded at v1.11.0 to match this section): the digit route is **outside** (iv)'s escalation ladder
(`64`-bit → `128`-bit → fixed-width wide), which governs the exact-integer tables of [C]
`thm:backend` and is unchanged — earlier text here called it "the fixed-width wide rung" of that
ladder; the contract's reading is the binding one. `HeckeCore` and `Integral/` are untouched (their
tables stay `int64` / `__int128` under [C] `thm:backend`).

#### 13.11.9 Cost shape, and what is not built

* Per level: one word elimination of the window (`O(r_l · m · n)` word operations) and the replay
  (`O(r_l · m · n)` digit products of `O(L²)` word multiplications each); the canonicalisation pass
  is one §13.5 pass over at most `min(m, n)` Smith-adapted rows. Memory: `L` words per entry.
* **Not built in slot 11** (levers, not contract): the digit GEMMs on the AVX-512 `u32` kernels of
  §6 (the carry split is scalar), a parallel level driver, the tape, the device, `hybrid`. None can
  change a value.
* WP-18b measures one `n = 128` cell at `L ∈ {2, 4}` indicatively (no verdict timing; the lead's
  windows); a slow result reprices WP-18c's cell list, not the design. **Measured** (WP-18b,
  `LADIC_INDUCTION_PROBE_N128=1`, one thread, shared box at load ≈ 3–9, **indicative only**): the
  `d^1` of `(128, 5)` over `Z/5^26` (`X_1` 645 × 1161 in `C_eq` coordinates, `X_0` 1161 × 495):

  | route | `smith_lift(X_1)` | `kernel_module(X_1)` | `cyclic_decomposition(Zc, X_0)` |
  |---|---:|---:|---:|
  | native `u64` (§13.5–§13.6) | — | 0.042 s | 0.074 s |
  | `1 × 64` (`force_induction`) | 0.014 s | 0.11 s | 0.32 s |
  | **`2 × 32`** | 0.018 s | 0.14 s | 0.32 s |
  | `4 × 32` (`m' = 7`) | 0.029 s | 0.18 s | 0.47 s |

  The digit count costs little (`2 × 32` ≈ `1 × 64`); the ≈ 3–4× over the native route is the
  route's structure (the canonicalisation pass after the Smith lift, the kernel's generator
  back-substitution), not the digit arithmetic. `W = u128` digits run ≈ 3–4× slower than `u64`
  digits at equal `ell^K` (one native 128/64 division per digit-product split, `DigitWord<u128>::split`).

#### 13.11.10 History (the slot-8 / slot-10 design text, superseded)

Slot 8 (owner ruling [OD] S7-2L, 2026-09-23: "`mod ell^k` for arbitrarily large `k`"; design from
`docs/implementation/e8_ladic_investigation.md` §3, §4.3, §5.2) recorded (C2) as *deferred, not a
non-goal*: represent the reduced differential in `L = ⌈k/m⌉` base-`ell^m` digits, eliminate level by
level with the §13.5 pivot rule, the residual products as digit-convolution GEMMs; it named **the
delicate part** — valuation pivoting needs the valuation of the whole number, a pivot that is a
non-unit at level `j` may become a unit after the carry from level `j + 1`, so a one-sided pass needs
a look-ahead — and a second basis-rule family for lifted generating sets. Its rule — *the direct
route is (C2)'s oracle wherever both apply, on the final canonical forms* — is kept as §13.11.7's
first rows. Slot 10 (WP-9) restated the trigger as the computations T-1 / T-2 / T-3 (§13.12.7) and
specified L1-X (§13.12), recommending `u128` first "on risk rather than speed". The owner's
requirement of 2026-09-25 (later) made the induction width-generic with the native widths as oracles
(`2×32` vs `u64`, `4×32` and `2×64` vs `u128`, cross-width agreement beyond `2^127`), and the
clarification of 2026-09-25 (latest) scheduled it in slot 11 as the arbitrary-`k` mechanism (WP-18b
engine, WP-18c pipeline). The two-sided formulation (L1) dissolves the delicate part (§13.11.3) and
the canonicalisation pass removes the second rule family (§13.11.5). For `2^32 ≤ ell^k < 2^63` the
investigation's advice stands: the shipped `uint64_t` route (with slot-11 WP-12's Montgomery pair)
is the word route there; (C2) serves `ell^K ≥ 2^64` and is the oracle-checked route below it.

#### 13.11.11 The pipeline consumer (slot-11 WP-18c)

The one consumer of `engine::ladic` outside the validations is `Hecke/LadicCell.{h,cpp}`
(`hecke::ladic_cell::RunCell`), the digit route of [E] v1.12.0-pre.2 `sec:ring` at
`ell^k ≥ 2^64`, `ell ≥ 5`. It follows the recipe of §13.11.6 / WP-18b's arm (F) literally —
`kernel_module` of the relation system, `solve_multi_rhs_module` for `X_q`, `kernel_module(X_q)`,
`cyclic_decomposition(Z^q, X_{q−1})`, `invariant_factors(X_q)`, `cyclic_coordinates` for `A^{(q)}`,
`multiply` for every product — and adds only what the engine does not have: `ρ_n(g)` of an exact
integral matrix in digit arithmetic (§13.10 F4's unit recurrences and `diag(ell, 1)` split, on
`Ring<W>`), the division-free Berkowitz polynomial in digits, and the ordinary idempotent. It
instantiates no pipeline class template for a new element type; the width is a template argument
chosen at run time by `ENGINE_LADIC_WIDTH = 32 | 64 | 128` (default 64), an engine-family switch.
Its oracle matrix at the pipeline level is `HeckeLadicCellValidation`: the §13.11.7 rows `2 × 32` vs
native `u64` (the seven E8 points and the `u64` limit, every canonical object entrywise), `2^64 ≤
ell^K < 2^127` vs native `u128` (the cell's own `X_q`, `Θ^{(1)}`), cross-width `w ∈ {32, 64, 128}` on
whole cell records at `691^40` and `5^42`, and truncation. The per-cell precision bound
`K = k + e_max` (the certified word-size `K`-mode, then the stopping rule re-evaluated at `K`) is
[E]'s, not this section's.

### 13.12 L1-X: `Z/ell^k` for `2^64 ≤ ell^k < 2^127` — the `U128_Montgomery` regime (draft)

**Status (slot-11 WP-18a): step 1 of §13.12.8 BUILT — the engine layer and `DigitWord<W>`,
contract as built in §13.12.9; steps 2–4 superseded by (C2) (the role paragraph below).** The text
from here to §13.12.8 is the slot-10 design, kept; where §13.12.9 differs, §13.12.9 is the
contract. Original status line: **specified, not built** (slot-10 WP-9; slot-10 plan decision 7 (A):
design now, decide at the close whether to build). Nothing below exists in the tree. The section has the standing
of §13.11: it is a design, and it becomes a contract only when a package builds it. **It amends no
rule of §13.1–§13.11.** Every statement there about the `uint32_t` and `uint64_t` instantiations
holds unchanged. Sources: [IB] (`docs/implementation/integral_line_design_brief.md`) §4.2 and the
L1-X row of its phase table (≈ 1.0 AU, trigger-based); [L]
(`docs/implementation/e8_ladic_investigation.md`) §3 and §4.3; §13.11 above. The figures in
§13.12.5 come from a throwaway probe that was run outside the tree
(`docs/implementation/slot10/WP-9_report.md` §3).

**Role since 2026-09-25 (latest; see §13.11).** Native `u128` is built in slot 11 as an **engine
layer only** (WP-18a: §13.12.2's context, the kernels, the §13.6 entry points, the naive oracle, the
block model's composite surface — step 1 of §13.12.8) **plus the width abstraction
`engine::DigitWord<W>`**, `W ∈ {uint32_t, uint64_t, u128}`, so that (C2) is written once over the
digit word. Its job is to be (C2)'s **third digit width and its native oracle** in
`[2^64, 2^127)` (4 × 32 and 2 × 64 digits against native `u128`). **It is not the arbitrary-`k`
mechanism**: §13.12.8's step 2 (the `u128` instantiation of the pipeline classes, with the F-4 risk)
is **not built** — the pipeline reaches `ell^k ≥ 2^64` by (C2)'s digit route (WP-18c), and steps 3–4
are carried by WP-18c. §13.12.1's regime table, §13.12.6's grammar and §13.12.8's recommendation are
superseded where they conflict with this paragraph (the grammar cap becomes WP-18c's, `k ≤ 4096`
with a per-cell `8192`-bit limit, proposed for [E] v1.12).

#### 13.12.1 The four regimes of `k`, and which question needs which

| Regime | Mechanism | Status | Reach | Serves |
|---|---|---|---|---|
| (i) `ell^{k + e_max}` fits a word | the direct route, §13.1–§13.9 | **shipped** (E8) | `< 2^32` u32; `< 2^64` u64 (e.g. `5^27`, `691^6`, `134217757^2`) | every E8 object; `free_charpoly_H1` at precision `k` whenever `ell^{k + e_max} < 2^64` (at `(10, 691)`: `k ≤ 6`) |
| (ii) the module `H^q(Z/ell^k)` for **any** `k`, `ell ≥ 5` | one word-size run at `K = e_max + 1`, the certified `K`-mode ([C] `prop:certifiede`; [L] §2.4), then universal coefficients | **shipped** (E8 item 3) | every `k` | the `Z_(ell)`-type, lengths, invariant factors, Bockstein data. **No precision beyond a word is needed**: `e_max ≤ 1 + log_ell(n/(ell+1))` ([L] §2.5) |
| (iii) `T_p` on `Λ^1 ⊗ Z/ell^k`, level 1, any `k` | reduce line 2's integral output mod `ell^k`: `char_Z(T_p \| Λ^1)` (M-0, CRT residues) and, for the module, M1(b)'s integer matrices | char poly: M-0 shipped (the integer is reconstructed only under OD-3 (a)/(b), or where it fits a word); module: **needs M1(b) and OD-3** (out of slot 10, decision 2c) | every `k` | `ell`-adic eigenvalues by Hensel lifting of the reduced char poly, and the congruence module |
| (iv) families (Hida, Coleman), higher level by Shapiro | L1-H on line 1 ([IB] §4.2 (3), OD-6), at precision growing with the question | not started; a contract chapter first | needs `ell^k ≥ 2^64` eventually | ordinary projectors, `Λ`-adic interpolation |

**The owner's question, split by regime.** "Arbitrary `k`" is solved at word size for the
abstract module (ii). At level 1 it is served by reduction (iii), once M1(b) and OD-3 exist.
Regime (iv) is the one that **needs** a wider element type, and it has no consumer until L1-H
exists. The cochain-level computation at `ell^K ≥ 2^64` for a level-1 Hecke datum (trigger T-1,
§13.11) is the only use of L1-X that does not wait on another phase. It overlaps (iii), and it is
the natural **oracle** for (iii) and (iv).

#### 13.12.2 The element type and the context

* **`Elt = engine::u128`** (`unsigned __int128`, already `ModCtx.h`'s `u128` alias) becomes a
  **third element type, composite-only**. A trait `engine::is_composite_only<Elt>` is true for
  `u128` alone.
  * `ModCtx<u128>` is obtained **only** through `prime_power(ell, k)`. The field constructor
    `ModCtx<u128>(p)` and a field `init(p)` are compile-time errors (`static_assert` on the trait).
    So `is_field()` is false for **every** `u128` context. §13.8 (1)'s selection is untouched: no
    field program, field entry point or field consumer is ever instantiated for `u128`.
  * **Domain:** `ell` odd prime, `k ≥ 2`, and `ell^k < 2^127`. `ell < 2^64` follows from
    `k ≥ 2`, so `ell` is also held as a `uint64_t`. `ell = 2` is refused: `2^k` is even, so there
    is no Montgomery form. (A `U128_Pow2` regime — reduction mod `2^k` is a mask — is noted and
    **not** specified. `ell = 2` is a Bredon prime, honoured by `EquivariantCohomologyOnK` alone,
    and its certified `K` is small at every campaign `n`, [IB] §5.1.)
  * `prime_power<u128>` is **admissible below `2^64` as well**, because the oracle rule
    (§13.12.4) runs it there. The **consumer** selects `u128` only at `ell^k ≥ 2^64`
    (§13.12.3, `CoefficientRingContext`). Below that the `u32` / `u64` instantiations are
    selected exactly as today.
* **Regime `U128_Montgomery`:** `R = 2^128`, odd `p < 2^127` (so the REDC result `t < 2p` fits a
  `u128`). New constants: `mont128_ninv = −p^{−1} mod 2^128` (Newton, 6 steps from `p·p ≡ 1 mod
  8`) and `mont128_r2 = R² mod p`. `barrett_m`, `mont_*`, `has_p2`, `p2`: unused (0).
* **Representation: canonical integers in `[0, p)` everywhere.** This covers storage, views,
  `DenseBuf`, `BlockPool` content, hashing and the §13.4 members. The Montgomery form **never
  escapes a primitive**, exactly as in the `U64_Montgomery` regime of the CUDA layer. The primitives:
  * `mul(a, b) = REDC(REDC(a·b) · r2)` — two REDCs, canonical in and out, for one-shot products;
  * **the reused-operand pair** `mont128_prep(w) = w·R mod p` and `mont128_mulp(w', x) =
    REDC(w'·x) = w·x mod p` — one REDC per product. This is the row-AXPY form (scaling, the
    Howell eliminations, closure rows), the analogue of Shoup for `u32`. The probe measured it,
    and built on it the u128 Howell equals the u64 Howell entrywise at `691^6` (§13.12.5);
  * `add`, `sub`, `neg`: as for `u64` (the sum of two residues `< 2^128`, since `p < 2^127`);
  * **`inv(u)`, unit-only (§13.3's meaning):** Hensel from the `uint64_t` inverse mod `ell`,
    `x ← x(2 − u x) mod p`, `⌈log2 k⌉` steps. If `ell | u` it throws the §13.4
    `std::runtime_error("Modular inverse does not exist.")`. The reference uses the extended
    Euclidean algorithm with sign-tracked `u128` coefficients, an independent route;
  * `pow`, `batch_inv`: **absent** (field-only, §13.3). The §13.3 run-time guards become
    compile-time absence for this type;
  * `valuation`, `unit_part`, `divide_exact`, `ell_pow`: as §13.4, with `u128` integer `%` and
    `/` by `ell` (these calls are O(pivot search), not O(update)).
* **`Accum = engine::u256`**, a two-limb struct: the compiler has no 256-bit type. An exact
  product costs four 64×64→128 multiplies.
* **The §5 bound for `u128`.** A k-block of `kb` products of canonical residues is exact in
  `u256` and is reduced by `reduce(T) = REDC(REDC(T) · r2)`. REDC's precondition is `T < p·R`,
  so the block length is bounded by
  `k_max = ⌊(p·2^128 − 1) / (p − 1)²⌋` (≈ `2^128 / p`: ≈ `2^14.8` at `691^12`, `≥ 1` always,
  ≈ 2 near `2^127`). §5's rule — `kb ≤ min(k_max, 256)`, `ENGINE_CHECK`ed at every block start
  — carries over verbatim with this `k_max`. The `u32` / `u64` definitions of `k_max` are
  unchanged.
* **`ModCtx<u128>` stays trivially copyable** (§13.1, m6). The `ell^v` table is a fixed
  `std::array<u128, 127>` (≈ 2 KB, so `sizeof(ModCtx<u128>) ≈ 2.1 KB`). The context is copied
  once per `BlockMat` / relation, never per entry, so the size is acceptable. A `static_assert`
  guards trivial copyability, as for the other two types.

#### 13.12.3 What it touches, and what it does not

**Touched: instantiated for `u128`, composite routes only.** The member list is the slot-8 / slot-9
composite surface. The pipeline file list is the set of TUs that instantiate that surface for
`uint64_t` today:

| Layer | Files | What changes |
|---|---|---|
| context | `ModCtx.h` | the trait, `U128_Montgomery`, `u256`, the primitives above |
| storage | `DenseBuf.h/.cpp`, `DenseView.h` | the instantiation; nothing type-specific |
| kernels | `Kernels.h/.cpp` | `gemm` / `gemv` / `axpy` / `scal` / `transpose` / `copy` / `canonicalise` for `u128`: a portable scalar path like `gemm<uint64_t>` (§6: "correct, not fast"), with `u256` accumulation under the bound above; no AVX-512 path |
| elimination | `DenseElim.h/.cpp` | the §13.6 entry points only (`howell`, `kernel_module`, `image_module`, `solve_multi_rhs_module`, `length`, `invariant_factors`, `cyclic_decomposition`); the field entry points are not instantiated |
| oracle | `RefLinAlg.h/.cpp` | `ref::howell`, `ref::invariant_factors` on a **naive `u128` mulmod** (double-and-add) and extended Euclid, not on the Montgomery primitives: an independent arithmetic |
| block model | `BlockPool`, `BlockMat.h`, `BlockKernels.h/.cpp` | the §13.7 `ModCtx` overloads, `blk::gemm` / `gemm_dense` / `add`; `BlockPool`'s content hash over 16-byte entries |
| `ρ_n` | `FlatMatrix.cpp` (`ComputeSymmetricPowerMatrixRing`), `Hecke/CoefficientRing.{h,cpp}` (`RhoOfExactRing`, `GenerateSL2ZRelationsRing`, `GenerateSL2ZBoundaryRelationsRing`) | the division-free Pascal route and the unit recurrences (§13.10 F4) are type-generic; `Mat22::reduced_mod` gains a `u128` modulus |
| pipeline (the ring members of [E] `sec:ring` (b1)) | `LocalSystemSimplicialComplex`, `EquivariantCohomologyOnK` (module route and `K`-mode), `HeckeOperatorOnK` (module route, `free_charpoly_H1`), `EquivariantMappingConeModule` (`ModuleConeT`, `ConeModuleCertificate`), `SL2ZBoundaryPair`, `HeckeBoundaryOperator`, `HeckeConeAction` (route 1), `EquivariantCochainComplexMorphismOnK` (the constant-fibre constructor), `Hecke/HeckeReduction` | a `u128` instantiation of the module route only. **The class templates are instantiated whole today** (`template class X<uint64_t>`). So the field route of each class must be compiled out for `u128`: its bodies go under `if constexpr (!is_composite_only<Elt>)`, or become named throwing stubs. **This is the largest item and the main risk** (§13.12.8) |
| the ring's identity | `Hecke/CoefficientRing.cpp` | `CoefficientRingContext<u128>`: refuses `ell^k ≥ 2^127` and `ell = 2`, and selects `u128` only at `ell^k ≥ 2^64`. ~~**`ExactKthRoot` must not stay as it is**: it derives `ell` from the modulus and tests its primality by trial division up to `√ell` (`CoefficientRing.cpp:37-40`). That is `2^16` steps under the `2^64` scope, but up to `≈ 2^31.7` at `k = 2` under `2^127`. So the ring travels as `(ell, k)`, or the test becomes deterministic Miller–Rabin over the 64-bit base set~~ **Cleared by slot-11 WP-12 (F-2), both ways:** (1) the ring travels as `(ell, k)` — `LocalSystemSimplicialComplexT::declare_coefficient_ring(ell, k)` (checks `ell^k ==` the modulus exactly; primality is a precondition, as for `prime_power`), which `EquivariantCohomologyOnK` reads in place of `ExactKthRoot` (adopted by `SL2ZBoundaryPair`; the other producers — `Campaign/`, `Golden/`, `Console/Bindings/` — are outside WP-12's files and keep the derivation, value-identical); (2) `ExactKthRoot`'s primality test is deterministic Miller–Rabin (`hecke::IsPrimeU64`, the twelve-prime base set, `O(log ell)`), equal to the slot-8 trial division (`CoefficientRingKthRootValidation`). Step 2's remaining item is a `u128` modulus for the undeclared derivation, or declaring at every producer |
| narrowing sites | the files above | every `static_cast<uint64_t>` of a modulus, `ell_pow(·)` or `prime` on a ring path (at the WP-9 base, 56 casts in the seven ring files, by grep; a minority are ring-relevant); `FreeCharpoly::modulus` (`HeckeOperatorOnK.h:325`, `uint64_t`); the campaign's `PowU64` and `charpoly_coeffs` (`CampaignRing.cpp:115-121`, `:154`) |
| grammar | `Hecke/HeckeSettings.cpp` (`ParseCoefficientRing`, `:391-421`), `Campaign/CampaignSpec.cpp:522-540` | §13.12.6 (an [E] v1.12 item, proposed, not made here) |

**Not touched:**
* **The tape** (`Recorder`, the seven passes, `Executor`, `ProgramCache`, `EngineRuntime`). The
  composite route is dense and block-level and records no program (§13.7), so no `u128` program
  exists. The structural hash, `ShapeKey`, the fingerprint and both determinism digests cannot
  move.
* **The device** (`Cuda/`, `EngineCostModel.h`, `hybrid`). There is no `u128` device path, and
  `hybrid` routes tape groups by element type, never reaching `u128` (no tape).
* **`k_max` blocking of `u32` / `u64`** (§5) and every existing regime threshold.
* **Every field path**: `FlatMatrix`'s legacy operations, `BlockFlatMatrix`,
  `ENGINE_BACKEND=legacy`, `-DENGINE_SHADOW_LEGACY`. None has a `u128` arm (as §13.7 has no
  composite arm for them).
* **`HeckeCore` and `Integral/`.** Their tables stay `int64` / `__int128` under [C] `thm:backend`.
  The `u128` element type is fixed-width, **not arbitrary precision**, so it is outside [C]
  `obl:backend`(iv)'s ban and outside [IB] OD-3 (c), and it lives on line 1's ring route only. The
  reading "a fixed 128-bit residue type is not arbitrary precision" is the author's; the build
  package's Fable referee confirms it against [C].
* **Golden.** Nothing moves. Below `2^64` the consumer selects the `u32` / `u64` instantiation
  exactly as today, so every existing object, text, hash and digest is bit-identical by
  construction; all golden E8 points are `< 2^32`. A `u128` golden point, if one is ever emitted,
  is add-only. Its entries are serialised as two `FnvU64` words, low limb first; that rule is
  declared once in the harness and touches no existing object.

#### 13.12.4 The oracle rule

As §13.11's, with the direct route as the oracle:
* **Wherever both apply (`ell^k < 2^64`, `ell` odd), the `u128` instantiation must equal the
  `u32` / `u64` one entrywise on the canonical forms.** These are the Howell `H`, `pivots` and
  `pivot_v`, `length`, `invariant_factors`, the cyclic decomposition, the `K`-mode type, the
  reduction maps and `free_charpoly_H1`. The inputs are the same `(n, ell, k)` and the same
  settings. There is no tolerance and no "equivalent form" (§13.5).
* **Arithmetic**: the `ModCtx<u128>` primitives against the naive `u128` mulmod and extended
  Euclid of `RefLinAlg` on `10^6` fixed-seed samples, at `2^64 ± small`, at `691^12`, and at an odd
  prime power just below `2^127` (e.g. `3^80 ≈ 2^126.8`). `k_max` must be tight (a block of `k_max`
  maximal products passes REDC's precondition, `k_max + 1` does not), and `inv` must throw on
  every valuation class.
* **Beyond `2^64`, one external oracle**: M-0's integer characteristic polynomial reduced mod
  `ell^k`. At `(10, 691)` this is `(x − 2049)(x + 24)²`
  (`Integral/IntegralCharpolyCrt.cpp:80`), and `free_charpoly_H1` of `T_2` over `Z/691^12`
  (precision 12, `e_max = 0`) must equal it coefficientwise mod `691^12`.
* **Route consistency across the `2^64` boundary**: `red: Z/ell^{k+1} → Z/ell^k` from a `u128`
  object (`ell^{k+1} ≥ 2^64`) to a `u64` one (`ell^k < 2^64`) is computed in the wider type. The
  target is compared after a narrowing that is exact, because its entries are `< ell^k`.

#### 13.12.5 Cost model (indicative)

Probe: a single-threaded standalone program, g++ 11.4 `-O2` (the `RelWithDebInfo` level; `-O3
-march=native` gave the same `u128` figures), Ryzen 9 7950X, **shared box at load 11–17, no
window: indicative only**. Three repeats; the ranges are across them.

| Operation | `u32` Barrett, `691^3` | `u64` exact `%` (**the tree's `ModCtx<u64>::mul`**), `691^6` | `u64` Montgomery-64, `691^6` | **`u128` Montgomery-128**, `691^12` | `u128` Barrett-128, `691^12` |
|---|---:|---:|---:|---:|---:|
| modmul, dependent chain (ns) | 2.6 | 4.5–5.0 | 2.6–2.7 | **6.9–8.2** | 13.1–13.8 |
| row AXPY `y ← y − q·x`, reused `q` (ns/entry) | 1.1–2.3 | 2.4–3.4 | 1.2–2.7 | **7.5–10.9** | 15.7–16.2 |
| delayed dot product (ns/MAC) | 0.09–0.25 | 0.50–0.62 | — | **2.8–3.1** | — |
| scalar Howell of `[Aᵀ \| I]`, `A` 143 × 121 of rank 22 (the `(10, 691)` `A_C1` shape, `kernel_module`'s input 121 × 264; two non-unit divisors) (ms) | 1.31–1.35 | 2.07–2.69 | 1.44–2.04 | **7.6–9.7** | 11.7–11.9 |

* **Ratios** (Howell, the composite route's own cost): `u128` Montgomery-128 at `691^12` ≈
  **3.6×** the tree's `u64` route at `691^6`, ≈ **5.2×** a Montgomery-64 `u64` route, ≈ **5.8×**
  `u32` at `691^3`. The `u128` code at modulus `691^6` costs the same as at `691^12`: the price is
  the width, not the exponent. Barrett-128 is ≈ 1.5× slower than Montgomery-128, **so the regime
  is Montgomery**. For assembly (dot products): ≈ 5.7× `u64` and ≈ 12× the scalar `u32` loop. The
  `u32` AVX-512 GEMM (§6, ≈ 30 G modmul/s) is ≈ 100× faster, but only on the field route and the
  `u32` composite assembly.
* **Absolute.** At `(10, 691)` a whole `u128` Howell is < 10 ms. The scalar Howell scales as
  `rows × cols × pivots`. At `n = 128` (`A_C1` 1677 × 1419), `kernel_module`'s input is 1419 × 3096
  with ≈ 1419 pivots, about `3·10^9` row-update entries: ≈ 3 s (`u32`), ≈ 7 s (tree `u64`),
  ≈ 12–30 s (`u128`) per call — an **extrapolation**, not a measurement. Memory per dense entry:
  16 B, i.e. 2× `u64`, 4× `u32`.
* **(C2) at the same range, for comparison.** With two `u64` digits ([L] §3: ≈ 2 eliminations +
  3 digit-GEMMs per level) the arithmetic would be ≈ 2 Montgomery-64 Howells ≈ 3–4 ms at the
  probe's shape: **possibly ≈ 2× cheaper than `u128`**. It has no oracle of its own, however, needs
  a second basis-rule family and needs the look-ahead pivot (§13.11), and the Montgomery-64 Howell
  it would be built on is not in the tree either. `u128` reuses §13.5 verbatim and gets the oracle
  rule for free. **So `u128` is the first mechanism, on risk rather than speed**, and (C2) stays
  the design beyond `2^127`, with `u128` digits.
* **A finding the probe makes about today's `u64` route** (not L1-X): `ModCtx<uint64_t>::mul` is
  the exact `(u128)a·b % p` (a `__umodti3` call). The `U64_Montgomery` constants exist, but only the
  CUDA layer uses them (`cuda_kernels.cu:95`, `:790`). A `prep` / `mulp` Montgomery-64 pair in the
  composite row updates would cut the `u64` Howell by ≈ 1.45× (2.1 → 1.45 ms at the probe shape).
  It is value-preserving by construction (canonical in, canonical out) and ≈ 0.05–0.10 AU. It is
  recorded as an engine follow-up. **Done in slot-11 WP-12 (F-1)** as `ModCtx::mul_prep` /
  `mul_prepped` (§1) in the §13.5 / §13.6 row updates (§13.5 "Overflow"); indicative
  `kernel_module` ratio 0.66–0.74 (new / old) on one thread. The `u64` column of the table above
  is therefore the pre-slot-11 route; the step-1 `u128` build's `mont128_prep` / `mont128_mulp`
  pair is the same shape.

#### 13.12.6 The grammar change, proposed for [E] v1.12 (slot 11) — not made here

`coefficient_ring`'s limits are [E] `sec:ring` grammar, not engine API, so this section only
**proposes** the change:
* **(a)** the cap `2 ≤ k ≤ 63` becomes `2 ≤ k ≤ 126`. The rule behind the number is unchanged:
  63 is the largest `k` with `2^k < 2^64`, and 126 is the largest `k` with `2^k < 2^127`. Whether
  `ell^k` fits is still decided at the cell, and there odd `ell` reaches at most `k = 80` (`3^80`).
  It is enforced in `MathSettings::set()` → `ParseCoefficientRing` (`HeckeSettings.cpp:391-421`).
* **(b3)** the per-cell refusal "`ell^k ≥ 2^64`" becomes "`ell^k ≥ 2^127`, or `ell = 2` with
  `2^k ≥ 2^64`". It is enforced **at the cell**: in `CoefficientRingContext`
  (`CoefficientRing.cpp:53-58`) for every honouring object, and in the campaign's spec validation
  (`CampaignSpec.cpp:522-540`, "a grid cell with `ell^k ≥ 2^64` … refused here").
* **Golden consequence: none — the change is key-additive.** Every value it admits is refused
  today, with exit 1 at `--set` or a named refusal at the cell. So no existing run, value,
  fingerprint, campaign job id or golden line changes. The new values give new job ids only.
* *Precision on where the refusal sits:* `set()` enforces the `k` cap only. The `ell^k` threshold
  is per cell (`ell` is the cell's), in the two places named in (b3).

#### 13.12.7 The trigger

§13.11, "Trigger, restated": T-1 (level-1 cochain-level precision `≥ 2^64` where (iii) is not
available), T-2 (L1-H at precision `> 2^63`), T-3 (`≥ 2^127`, which triggers (C2)).

#### 13.12.8 The costed package (slot 11 or later), and its prerequisites

**L1-X ≈ 1.1 AU** ([IB]'s 1.0, re-derived; tier Opus, a Fable review pass). It is ordered so that
each step is gated on its own:

| Step | Content | AU | Gate |
|---|---|---:|---|
| 1. Engine increment | `ModCtx<u128>` (trait, regime, `u256`, primitives, Hensel `inv`), `DenseBuf` / kernels (scalar), the §13.6 entry points, `ref::howell` on naive arithmetic, the block model's composite surface | 0.35 | the oracle rule at the engine level (`u128` = `u64` entrywise on the `HeckeReductionValidation` elimination arms' matrices), arithmetic arms; **no consumer, no golden risk; usable as a unit** |
| 2. Pipeline instantiation | the composite-only trait through the ring members (the field route compiled out for `u128`), `CoefficientRingContext<u128>`, `(ell, k)` in place of `ExactKthRoot` (the channel exists since slot-11 WP-12; every producer must declare), the narrowing sites, `FreeCharpoly` / record widths | 0.50 | the oracle rule on the seven E8 points and the 4b cells at `k` up to the `u64` limit, both instantiations; the M-0 external oracle at `(10, 691)`, `k = 12` |
| 3. Grammar, campaign, console | [E] v1.12 (a)/(b3) as §13.12.6 (spec first, its own `-pre` step); the `set()` cap, the two cell refusals, the console's exit codes for the new values | 0.10 | refusal arms re-pinned at `2^127`; `--list`, golden, the four digests unchanged |
| 4. Validation name, review | one new `standard` validation (no existing name re-tiered), docs, the Fable pass | 0.15 | the standing gates |

**Prerequisites:** none in code. The external oracle (M-0's `char_Z` table) exists. **Owner
inputs:** (1) a named T-1 / T-2 computation with its cells; (2) [E] v1.12's grammar ruling;
(3) confirmation that a fixed-width `u128` residue is outside [C] `obl:backend`(iv) (the referee
checks it). **Risk:** medium, concentrated in step 2. Compiling the field route out of whole-class
instantiations touches the same classes as the E8 packages, and its build-time and binary-size
cost is unmeasured. Steps 1 and 3 are low risk.

**Recommendation for the slot-10 close** (the owner's decision-7 follow-up; argued in the slot-10
WP-9 report): **do not build L1-X in slot 11 unless a T-1 or T-2 computation is named.** Regimes
(i)–(ii) cover every module question at word size. Level-1 eigen-data goes by reduction (iii), and
(iv) has no consumer before L1-H. If the owner wants the capability as infrastructure regardless,
build **step 1 alone** (0.35 AU): it is self-contained and gated by the oracle rule, and it moves
nothing. The Montgomery-64 follow-up of §13.12.5 (≈ 0.05–0.10 AU) helps the shipped `u64` route
now, whichever way the ruling goes.

#### 13.12.9 As built (slot-11 WP-18a): the `u128` engine layer and `DigitWord<W>` — contract

Step 1 of §13.12.8, built to §13.12.2–§13.12.4 with the deviations listed at the end. **No
consumer, no pipeline route, no tape, no device**: every existing value, text, hash, digest and
golden line is unchanged by construction (the `u32` / `u64` code is not touched by a token, below),
and `EngineU128CompositeValidation` (`standard`) gates the layer.

**Files.**

| Layer | File | What exists for `u128` |
|---|---|---|
| context | `ModCtx.h` (the `Regime::U128_Montgomery` enumerator, appended; `regime_name`; `#include "ModCtxU128.h"` at its end) and **`ModCtxU128.h`** (new) | `engine::u256`, `detail::mul_wide128`, `ModTraits<u128>`, `is_composite_only<Elt>`, and `ModCtx<u128>` as a **full specialisation** — so `ModCtx<uint32_t>` / `ModCtx<uint64_t>` are the primary template, unchanged |
| storage | `DenseBuf.cpp` | `template class DenseBuf<unsigned __int128>` (the classes are header-only and generic) |
| kernels | `Kernels.h/.cpp` | explicit specialisations `gemm<u128>` (scalar, `u256` accumulation, k-blocked at `min(k_max, 256)`), `gemv<u128>` (= `gemm` on a column), `dot_delayed<u128>`; `gemm_nt`, `axpy`, `scal` (both on the `mul_prep` / `mul_prepped` pair), `trmm`, `apply_signed_antidiag`, `apply_diagonal`, `copy`, `transpose`, `canonicalise`, `swap_rows` instantiated from the generic templates. No AVX-512 path |
| elimination | `DenseElim.cpp` | the §13.6 entry points **only** — `howell`, `kernel_module`, `image_module`, `solve_multi_rhs_module`, `length` (both), `invariant_factors` (both), `cyclic_decomposition`, `cyclic_coordinates` — instantiated from the **unchanged** bodies. No field entry point exists for `u128`, so a field use does not link |
| oracle | `RefLinAlg.h/.cpp` | `ref::naive128::{add, sub, mul, inv}` (double-and-add mulmod; extended Euclid with sign-tracked `u128` coefficients) and `ref::gemm` / `gemv` / `trmm` / `transpose` / `howell` / `invariant_factors` instantiated on them (TU-local `ModArithmetic::mod_*_w` overloads for `unsigned __int128`); `rref` / `kernel_basis` (field) are not |
| block model | `BlockPool.h`, `BlockKernels.cpp` | `BlockPool::content_hash` mixes a 16-byte entry as its two 64-bit halves, **low limb first** (`if constexpr (sizeof(Elt) == 16)`; other widths byte-identical); `blk::gemm` / `gemm_dense` / `add` / `scale` / `equal` / `equal_dense` / `materialise_*` instantiated; `BlockMat` / `BlockBuilder`'s §13.7 `ModCtx` overloads are header-only and serve `u128` unchanged. `BlockElim` (field-only) is not instantiated |
| width abstraction | **`DigitWord.h`** (new) | `engine::DigitWord<W>`, below |

**`ModCtx<u128>` (composite-only).** Obtained only by `ModCtx<u128>::prime_power(ell, k)`:
`ell` odd, `3 ≤ ell < 2^64` (primality a precondition, as for every `prime_power`), `k ≥ 2`,
`ell^k < 2^127`, each `ENGINE_CHECK`ed (so `ell = 2`, `k = 1`, `3^81`, `691^14` are refused).
There is no `ModCtx<u128>(p)` constructor, no `init(p)`, no `pow`, no `batch_inv`: a field context or
a field-only primitive of this type **does not compile** (§13.3's run-time guards become compile-time
absence). `is_field()` is `false`. Members read by the generic composite code: `p`, `ell`, `k`,
`k_max`, `regime`, `mul`, `add`, `sub`, `neg`, `inv`, `mul_prep`, `mul_prepped`, `reduce`,
`canonicalise`, `valuation`, `unit_part`, `ell_pow`, `divide_exact` — **the same interface as
`ModCtx<uint64_t>` after WP-12's F-1**, which is why `DenseElim.cpp`'s composite rows instantiate
for `u128` with no change to a body. Montgomery-128 constants `mont128_ninv` (`−p^{−1} mod 2^128`,
Newton, 6 steps, checked `p·inv = 1`), `mont128_r1 = 2^128 mod p`, `mont128_r2 = R² mod p` (128
modular doublings of `r1`). Primitives, all canonical in and out: `mul(a, b) =
REDC(REDC(a·b)·r2)`, `mul_prep(w) = REDC(w·r2) = wR mod p`, `mul_prepped(x, w') = REDC(x·w')`,
`reduce(T) = REDC(REDC(T)·r2)` for `T < p·2^128` (a `ENGINE_DEBUG_CHECK`), `inv` by Hensel from
the `uint64_t` extended-Euclid inverse mod `ell`, `⌈log2 k⌉` steps, throwing
`std::runtime_error("Modular inverse does not exist.")` when `ell | u`. `k_max = ⌊(p·2^128 −
1)/(p − 1)²⌋` by an exact 256-bit division at construction, saturated to `2^64 − 1`
(`691^12`: 28 714; `3^80`: 2; `4294967311^2`: 18 446 743 944 860 533 412; saturated at
`4294967291^2`, `5^27`, `691^6`). `sizeof(ModCtx<u128>) = 2144`, trivially copyable
(`static_assert`).

**`engine::DigitWord<W>`, `W ∈ {uint32_t, uint64_t, u128}` — the interface WP-18b's (C2) code is
written against** (`DigitWord.h`; one static interface for the three widths):

| Member | Meaning |
|---|---|
| `Word`, `Ctx = ModCtx<W>`, `Wide` | the digit word, its context type, its exact double word (`uint64_t`, `u128`, `u256`) |
| `bits`, `digit_bits = bits − 1` | 32 / 64 / 128; every digit modulus is `< 2^{bits−1}` (the Montgomery bound of the width) |
| `digit_exponent(ell)` | `m(W, ell)`: the largest `m` with `ell^m < 2^{bits−1}`; **0 when `ell^2 ≥ 2^{bits−1}`** (the width cannot carry a composite digit: the §13.6 entry points refuse a field context). `(3: 19, 39, 80)`, `(5: 13, 27, 54)`, `(7: 11, 22, 45)`, `(691: 3, 6, 13)`, `(134217757: 0, 2, 4)`, `(4294967291: 0, 0, 3)` |
| `available(ell)` | `digit_exponent(ell) ≥ 2` |
| `fits(ell, e)`, `pow(ell, e)` | `ell^e < 2^bits`; `ell^e` as a `W` (checked) |
| `ctx(ell, e)` | `ModCtx<W>::prime_power(ell, e)`, `e ≥ 2`, `ell^e` fits (for `u128` also `ell` odd and `ell^e < 2^127`) |
| `mul_wide(a, b, hi, lo)` | the exact `2w`-bit product |
| `split(hi, lo, M, r)` | `hi·2^w + lo = q·M + r`, `0 ≤ r < M`, returns `q`; precondition `hi < M` (`ENGINE_CHECK`) |
| `mac_split(a, b, c, M, r)` | `a·b + c = q·M + r` for `a, b, c < M` (so `q < M`): the digit-convolution step with carry by `M = ell^m` |
| `widen(x)`, `narrow(x)` | `W ↔ u128` (checked): the oracle bridges between widths |
| `gemm`, `howell`, `kernel_module`, `image_module`, `solve_multi_rhs_module`, `length`, `invariant_factors`, `cyclic_decomposition`, `cyclic_coordinates` | forward to `kernels::gemm<W>` and the §13.6 entry points `elim::…<W>` verbatim — so an `L = 1` digit run **is** the direct route |

`split` / `mac_split` cost one native division for `u32` and `u64`; for `u128` WP-18a shipped a
128-step restoring division (correct, not fast), which WP-18b replaced (`f90a9ee`) by a normalised
long division on 64-bit halves (Knuth D, two quotient digits), keeping the restoring loop as
`split_restoring`, its oracle (§13.11.6, §13.11.9; *as of slot-11 WP-15*, the WP-14 review's m-4).

**The oracle rule as implemented (§13.12.4 at the engine level; `EngineU128CompositeValidation`,
26 587 checks, ≈ 1.5 s).**
* (A) **arithmetic**: `mul`, the pair, `reduce(a·b)`, `add` / `sub` / `neg` against `naive128` on
  200 000 fixed-seed samples at each of six moduli (1.2·10^6) plus boundary values (`0, 1, 2,
  p − 1, p − 2, p/2, 2^64 ± j, 2^63, 2^127 − 1` reduced) at `4294967291^2` (`2^64 −` small),
  `4294967311^2` (`2^64 +` small), `691^12`, `3^80`, and below `2^64` at `5^27`, `691^6` also
  against `ModCtx<uint64_t>` (mul, pair, inv); `inv` against extended Euclid on 20 000 units per
  modulus and throwing on `0` and on every valuation class `v = 1 … k − 1`; `k_max` tight (`k_max ·
  (p − 1)^2 < p·2^128 ≤ (k_max + 1)(p − 1)^2`, `reduce` exact on the extremal block, `dot_delayed`
  accepting `k_max` and refusing `k_max + 1`); the kernels against `ref::gemm` across k-blocks;
  the construction refusals;
* (E) **`u128 = u32 / u64` entrywise** — Howell `H` / `pivots` / `pivot_v`, `kernel_module`,
  `image_module`, `length`, `solve_multi_rhs_module` (value or refusal text), `invariant_factors`,
  `cyclic_decomposition`'s exponents / basis / `Gamma` / `U` / `position`, `cyclic_coordinates` —
  on `HeckeReductionValidation`'s elimination-arm matrices (its (H) and (M) generators replayed
  with its seeds at its odd moduli: `3^2`, `5^2`, `5^3`, `691^2` in `u32`, `134217757^2`,
  `4294967291^2` in `u64`), and on random cases at `5^27`, `691^6` (`U64_Montgomery`), `3^40`,
  `4294967291^2` (`U64_Wide`);
* (B) **beyond `2^64`**: `howell<u128> = ref::howell` (naive128), `invariant_factors =
  ref::invariant_factors`, `A·ker^T = 0`, `A·solve(A, Y) = Y` and rank–nullity by `ref::gemm`,
  and **planted** elementary divisors (`U diag(ell^{v_i}) V`, 16 per modulus) recovered by
  `cyclic_decomposition`, `invariant_factors` and `ref::invariant_factors`, at `4294967311^2`,
  `5^30`, `7^40`, `691^12`, `3^80`;
* (K) the block surface at `691^12`, `3^80` (`blk::gemm` / `gemm_dense` / `add` / `scale` against
  `ref::gemm` / `naive128`, the context carried, the pool hash distinguishing a high-limb-only
  difference and deduplicating equal content);
* (W) `DigitWord<W>`: the `m(W, ell)` table above, `mac_split` / `split` against exact `u256`
  arithmetic, `ctx(ell, m)`, and the three widths' word entry points entrywise equal at `691^3`.

M-0's external oracle (§13.12.4 third bullet) needs a pipeline route and is WP-18c's; the
route-consistency bullet (fourth) likewise.

**Deviations from §13.12.2–§13.12.3, and why.**
1. `ell` is held as a `u128` member (value `< 2^64`, checked), not a `uint64_t`: the generic
   composite bodies read `ctx.ell` with `x % ell`, `x /= ell` on `Elt`, and keeping the member's
   type `Elt` is what lets them instantiate unchanged. The Hensel start uses its `uint64_t` value.
2. `ModCtx<u128>` is a full specialisation in its own header rather than branches inside the
   primary template: the `u32` / `u64` instantiations then cannot change by construction.
3. The table is `std::array<u128, 127>` as specified; `k ≤ 80` in practice (`3^80`).
4. The pipeline rows of §13.12.3 (`ρ_n`, `CoefficientRingContext<u128>`, the pipeline classes,
   the narrowing sites, the grammar) are **not** built — superseded by (C2) (WP-18b / WP-18c).
