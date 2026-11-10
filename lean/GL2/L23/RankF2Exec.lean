/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/

/-!
# LV-0 (slot 22): `rankF2`, Gaussian elimination over `F₂` on bit-packed rows — the executable part

`L23_master_plan.md` §4.2(2): a **verified algorithm** instead of witness data. A row of an `F₂`-matrix is one `ℕ`
(bit `j` = entry `j`), a row operation is one `Nat.xor`; the kernel's GMP acceleration of `Nat.land`, `Nat.xor`,
`Nat.shiftRight`, `Nat.log2` therefore makes `decide +kernel` run native binary arithmetic.

* `reduce B r`: one pass of `r` over the basis `B`, a list of `(pivot, row)` pairs, strictly descending in the pivot
  (each pivot the top bit of its row): `r := r ^^^ b` whenever bit `p` of `r` is set.
* `insert B r`: `r` (non-zero, reduced) placed at its top bit `Nat.log2 r`.
* `rankF2 rows`: the length of the basis after all rows are processed.

This module imports nothing beyond `Init`, so a benchmark that only runs the kernel does not load Mathlib. The
soundness theorem (`rankF2 = Matrix.rank` over `ZMod 2`) is `GL2.L23.RankF2.rankF2_eq_rank` in `GL2/L23/RankF2.lean`.

Also here: the matrix generators of the LV-0 benchmarks (`lean/tools/gen_rankf2_bench.py`): `qrRows` (pseudo-random
rows) and `amalgamCols` (the amalgam-type matrix `[1 + ρ_n(S) | 1 + ρ_n(U) + ρ_n(U)²]` mod 2, by columns).
**Kernel formulation (an LV-0 finding).** Every loop is written with the recursors `List.rec` / `Nat.rec` / `Bool.rec`
and the raw primitives `Nat.land`, `Nat.shiftRight`, `Nat.xor`, `Nat.beq`, `Nat.blt`, `Nat.log2` — no equation-compiled
structural recursion (`brecOn`), no `cond` / `testBit` / `HXor` instance chains. On R(500) this is 5× faster in the
kernel than the same algorithm written with `match` and `bif` (34 s → 6.8 s). Accumulators that the next step does not
inspect are forced (`force`), else the kernel builds a chain of unevaluated `xor`s and overflows its recursion depth.
The definitions are therefore `noncomputable` (the code generator does not compile recursors); they are meant for
`decide +kernel`, and `lean/tools/gen_rankf2_bench.py` is the independent (Python) reference.
-/

noncomputable section

namespace GL2.L23.RankF2

/-- Evaluate `a` (to a literal, in the kernel) before continuing with `k a`. -/
@[inline] def force {α : Sort _} (a : Nat) (k : Nat → α) : α := @Bool.rec (fun _ => α) (k a) (k a) (Nat.beq a 0)

/-- Bit `p` of `r`, from the GMP-accelerated primitives. -/
def bit (r p : Nat) : Bool := Nat.beq (Nat.land (Nat.shiftRight r p) 1) 1

/-- One basis step: clear bit `p` of `r` with the basis row `b`. -/
def step (r : Nat) (pb : Nat × Nat) : Nat := @Bool.rec (fun _ => Nat) r (Nat.xor r pb.2) (bit r pb.1)

/-- Reduce `r` by a basis (one pass, in list order). -/
def reduce (B : List (Nat × Nat)) : Nat → Nat :=
  @List.rec (Nat × Nat) (fun _ => Nat → Nat) (fun r => r) (fun pb _ ih r => ih (step r pb)) B

/-- Insert `(p, r)` into a list strictly descending in the first component. -/
def insertP (p r : Nat) (B : List (Nat × Nat)) : List (Nat × Nat) :=
  @List.rec (Nat × Nat) (fun _ => List (Nat × Nat)) [(p, r)]
    (fun pb B ih => @Bool.rec (fun _ => List (Nat × Nat)) (pb :: ih) ((p, r) :: pb :: B) (Nat.blt pb.1 p)) B

/-- Process one row. -/
def addRow (B : List (Nat × Nat)) (r : Nat) : List (Nat × Nat) :=
  @Bool.rec (fun _ => List (Nat × Nat)) (insertP (Nat.log2 (reduce B r)) (reduce B r) B) B (Nat.beq (reduce B r) 0)

/-- The echelon basis of a list of rows, starting from the basis `B`. -/
def basis (rs : List Nat) : List (Nat × Nat) → List (Nat × Nat) :=
  @List.rec Nat (fun _ => List (Nat × Nat) → List (Nat × Nat)) (fun B => B) (fun r _ ih B => ih (addRow B r)) rs

/-- **The rank over `F₂`** of the matrix with the given bit-packed rows. -/
def rankF2 (rows : List Nat) : Nat := (basis rows []).length

/-! ### Benchmark generators -/

/-- `[f j, f (j+1), …, f (j+k-1)]`. -/
def tabulate (f : Nat → Nat) (k j : Nat) : List Nat :=
  @Nat.rec (fun _ => Nat → List Nat) (fun _ => []) (fun _ ih j => f j :: ih (j + 1)) k j

/-- `k` rows of the iteration `x ↦ (x² + 12345) mod M`, starting after `x`. -/
def qrGo (M k x : Nat) : List Nat :=
  @Nat.rec (fun _ => Nat → List Nat) (fun _ => []) (fun _ ih x => force ((x * x + 12345) % M) fun y => y :: ih y) k x

/-- The `d × d` pseudo-random benchmark matrix R(d): `M = 2^d − 1`, `x₀ = ⌊M / 3⌋`. -/
def qrRows (d : Nat) : List Nat := qrGo (2 ^ d - 1) d ((2 ^ d - 1) / 3)

/-- `Σ_{m ⊆ j} g m` (xor), by the enumeration `m ↦ (m - 1) &&& j` from `m = j` down to `0`; fuel `f ≥ j + 1`. -/
def subsetsXor (g : Nat → Nat) (j f : Nat) : Nat :=
  @Nat.rec (fun _ => Nat → Nat → Nat) (fun _ acc => acc)
    (fun _ ih m acc => force (Nat.xor acc (g m)) fun acc' =>
      @Bool.rec (fun _ => Nat) (ih (Nat.land (m - 1) j) acc') acc' (Nat.beq m 0)) f j 0

/-- Column `j` of `ρ_n(U)` mod 2, `U ≡ [[0,1],[1,1]]` acting by `x ↦ y`, `y ↦ x + y` on `e_j = x^(n-j) y^j`:
`e_j ↦ y^(n-j) (x+y)^j = Σ_m C(j, m) e_(n-m)`, and `C(j, m)` is odd iff `m ⊆ j` (Lucas). -/
def colU (n j : Nat) : Nat := subsetsXor (fun m => Nat.shiftLeft 1 (n - m)) j (j + 1)

/-- Column `j` of `ρ_n(U)² = ρ_n(U²)` mod 2, `U² ≡ [[1,1],[1,0]]`: `x ↦ x + y`, `y ↦ x`, so
`e_j ↦ (x+y)^(n-j) x^j = Σ_m C(n-j, m) e_(n-j-m)`; with `m ⊆ n-j`, `n-j-m` runs over the subsets of `n-j`. -/
def colU2 (n j : Nat) : Nat := subsetsXor (fun m => Nat.shiftLeft 1 m) (n - j) (n - j + 1)

/-- Column `j` of `1 + ρ_n(S)` mod 2, `S ≡ [[0,1],[1,0]]`: `e_j ↦ e_j + e_(n-j)`. -/
def colOnePlusS (n j : Nat) : Nat := Nat.xor (Nat.shiftLeft 1 j) (Nat.shiftLeft 1 (n - j))

/-- Column `j` of `1 + ρ_n(U) + ρ_n(U)²` mod 2. -/
def colN3 (n j : Nat) : Nat := Nat.xor (Nat.xor (Nat.shiftLeft 1 j) (colU n j)) (colU2 n j)

/-- The amalgam-type benchmark matrix A(d), `d = n + 1`: `[1 + ρ_n(S) | 1 + ρ_n(U) + ρ_n(U)²]` (`d × 2d`), given by
its `2d` columns; its rank is the rank of the list (rank of the transpose). -/
def amalgamCols (n : Nat) : List Nat := tabulate (colOnePlusS n) (n + 1) 0 ++ tabulate (colN3 n) (n + 1) 0

/-- The xor of a list (a fingerprint forcing every entry, for the generation-only benchmark). -/
def xorAll (rs : List Nat) : Nat :=
  @List.rec Nat (fun _ => Nat → Nat) (fun acc => acc) (fun r _ ih acc => force (Nat.xor acc r) ih) rs 0

end GL2.L23.RankF2

end
