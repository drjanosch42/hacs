/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.L23.RankF2Exec

/-!
# LV (slot 23): `smithPK`, elementary divisors over `ℤ/ℓ^K` on packed rows — the executable part

`L23_master_plan.md` §4.2(2) (`smith2k` / `smith3k`), `00-slot23-plan.md` (LV), `LV_plan_and_predictions.md`.
A **verified algorithm**, no witness data: the soundness theorems are `GL2.L23.SmithPK.smithPK_spec`,
`smithPK_card` and `card_of_sig` in `GL2/L23/SmithPK.lean`.

**Packing.** A row of a matrix over `ℤ/N`, `N = ℓ^K`, is one `ℕ`: entry `i` is the field `fld F x i`, bits
`[F i, F i + F)`, `F = 2H`. Fields are kept **lazily unreduced**: a row operation `r ← r + c·p` (`c < N`, the pivot row `p`
reduced, entries `< N`) is **one `Nat.mul` and one `Nat.add`** on the whole row, and an entry grows by `< N²` per pivot, so
after at most `R` pivots (`R` = number of rows) it stays `< N + R N² ≤ 2^H` — the cell hypothesis `bound`. Only a row that
becomes a pivot is reduced, field by field (`redRow`, `w` steps, once per pivot).

**Levels.** For `v = 0, …, K − 1` every remaining row is tested for a field of valuation exactly `v` (all its fields are
divisible by `ℓ^v` by then). The test of a whole row is **four GMP operations** (`tst`): with per-level constants `(c, b, T)`
and the masks `LOW = Σ (2^b − 1) 2^{F i}`, `ADD = Σ (2^H − 1 − T) 2^{F i}`, `TOP = Σ 2^H 2^{F i}`,
`tst x = (((x · c) &&& LOW) + ADD) &&& TOP`; field `i` of the result is non-zero iff `ℓ^{v+1} ∤ x_i`:
* `ℓ = 2`: `c = 1`, `b = v + 1`, `T = 0` (the low `v + 1` bits of the field);
* `ℓ` odd: `m = ℓ^{v+1}`, `c = m⁻¹ mod 2^H`, `b = H`, `T = ⌊(2^H − 1)/m⌋`: for `f < 2^H`, `m ∣ f ⟺ f c mod 2^H ≤ T`
  (the exact-division test; `f·c < 2^{2H} = 2^F` fits its field).
A non-zero result names a column `j` (its first non-zero field, `firstNZ`); the row is reduced, becomes the pivot
`(v, j, p)` (`p_j = ℓ^v u`, `ℓ ∤ u`), and column `j` is cleared from every other remaining row with `c = −(r_j / ℓ^v) · u⁻¹ mod N`
(`u⁻¹ = u^{φ(N) − 1} mod N`, `powMod`). A row that tests zero is divisible by `ℓ^{v+1}` and waits for the next level (it
stays so: its multipliers are divisible by `ℓ`). After level `K − 1` the remaining rows are `0 mod N`.

**Output.** The pivots `(v, j, p)`, latest first. Their valuations are, by the usual Smith argument, those of the
elementary divisors `≠ 0 mod ℓ^K` (an elementary divisor `0 mod ℓ^K` yields no pivot); **certified** is only
`smithPK_card`: the row span has `ℓ^{Σ (K − v)}` elements, and `smithPK_card_smul` (`SmithPKType.lean`) the type:
`#(ℓ^t · span) = ℓ^{Σ (K − (v + t))}` for every `t`, which certifies the valuations. The cell modules state `cellSig … = (Σ (K − v), counts)`.

**Kernel cost.** `R · K` tests, `≤ R` pivots, `≤ R` row operations per pivot, `w` field reads per pivot: `O(R² + R K + R w)`
GMP operations on `F w`-bit numbers — the `rankF2` budget (`LV0_report.md`), not `d³`. Measured (slot 23, single
declarations, 8 GiB guard): `d²` of `Sym^150` (`453 × 453`) in **24 s / 4.6 GiB** (`ℓ = 2`, `K = 10`) and 21 s / 4.3 GiB
(`ℓ = 3`, `K = 6`). **`Nat.log2` is not accelerated by the kernel** (it unfolds bit by bit: 0.4 s per call on an
8 500-bit number, measured); the pivot column is found by `firstNZ` instead (26 s → 2 s at `n = 50`).

**Kernel formulation** (the LV-0 finding): recursors and raw `Nat` primitives only, accumulators forced (`force`);
`noncomputable` (the code generator does not compile recursors). The independent reference is the Python Smith
elimination of `lean/tools/gen_lv_cells.py`.
-/

noncomputable section

namespace GL2.L23.SmithPK

open GL2.L23.RankF2 (force)

/-- Field `i` of width `F` of the packed row `x`: `(x >>> F i) &&& (2^F − 1)`. -/
def fld (F x i : Nat) : Nat := Nat.land (Nat.shiftRight x (Nat.mul F i)) (Nat.sub (Nat.pow 2 F) 1)

/-- `w` copies of `a`, one per field: `Σ_{i<w} a · 2^{F i}` (Horner). -/
def rep (F a w : Nat) : Nat := @Nat.rec (fun _ => Nat) 0 (fun _ acc => Nat.add a (Nat.shiftLeft acc F)) w

/-- `acc · b^e mod M` by binary exponentiation in `f` rounds (exact once `e < 2^f`). -/
def powModGo (M : Nat) (f : Nat) : Nat → Nat → Nat → Nat :=
  @Nat.rec (fun _ => Nat → Nat → Nat → Nat) (fun _ _ acc => acc)
    (fun _ ih b e acc =>
      force (Nat.mod (Nat.mul b b) M) fun b2 =>
      force (@Bool.rec (fun _ => Nat) acc (Nat.mod (Nat.mul acc b) M) (Nat.beq (Nat.mod e 2) 1)) fun acc' =>
        ih b2 (Nat.div e 2) acc') f

/-- `b^e mod M` in `f` rounds (exact for `e < 2^f`; `Nat.log2` is avoided: it is **not** accelerated by the kernel). -/
def powMod (f b e M : Nat) : Nat := powModGo M f (Nat.mod b M) e (Nat.mod 1 M)

/-- The first `i ∈ [i₀, i₀ + k)` with field `i` of `t` non-zero (`i₀ + k` if none): `≤ k` field reads. (Not `Nat.log2`:
the kernel unfolds it bit by bit, ≈ 0.4 s per call on an 8 500-bit row, measured.) -/
def firstNZ (F t : Nat) : Nat → Nat → Nat :=
  @Nat.rec (fun _ => Nat → Nat) (fun i => i)
    (fun _ ih i => @Bool.rec (fun _ => Nat) i (ih (Nat.succ i)) (Nat.beq (fld F t i) 0))

/-- The level constants `(c, b, T)` of the test "`ℓ^{v+1}` divides the field" (see the module docstring). -/
def lvC (l H v : Nat) : Nat × Nat × Nat :=
  @Bool.rec (fun _ => Nat × Nat × Nat)
    (force (Nat.pow l (Nat.succ v)) fun m =>
      (powMod H m (Nat.sub (Nat.pow 2 (Nat.sub H 1)) 1) (Nat.pow 2 H), H, Nat.div (Nat.sub (Nat.pow 2 H) 1) m))
    (1, Nat.succ v, 0)
    (Nat.beq l 2)

/-- The row test with prepared masks: field `i` of the result is non-zero iff `ℓ^{v+1} ∤ x_i`. -/
def tst (c LOW ADD TOP x : Nat) : Nat := Nat.land (Nat.add (Nat.land (Nat.mul x c) LOW) ADD) TOP

/-- Clear column `j` of `r` with the pivot row `p` (`p_j = ℓ^v u`, `lv = ℓ^v`, `ui = u⁻¹ mod N`). -/
def elim (F N lv ui j p r : Nat) : Nat :=
  force (Nat.div (Nat.mod (fld F r j) N) lv) fun b =>
  force (Nat.mod (Nat.sub N (Nat.mod (Nat.mul b ui) N)) N) fun c =>
  force (Nat.add r (Nat.mul c p)) fun r' => r'

/-- `(L.map g).reverse ++ acc`, by the recursor, forcing every entry: the spine is built **eagerly** at the first
demand (a lazy `map` would leave one unevaluated layer per pivot on every list, re-walked at every level). -/
def mapA (g : Nat → Nat) (L : List Nat) : List Nat → List Nat :=
  @List.rec Nat (fun _ => List Nat → List Nat) (fun acc => acc)
    (fun a _ ih acc => force (g a) fun b => ih (b :: acc)) L

/-- Fields `i, i+1, …, i+k−1` of `r`, each reduced mod `N`, repacked from field `0`. -/
def redGo (F N r : Nat) : Nat → Nat → Nat :=
  @Nat.rec (fun _ => Nat → Nat) (fun _ => 0)
    (fun _ ih i => force (Nat.mod (fld F r i) N) fun g => Nat.add g (Nat.shiftLeft (ih (Nat.succ i)) F))

/-- The row `r` with its `w` fields reduced mod `N` (fields `≥ w` dropped). -/
def redRow (F N w r : Nat) : Nat := redGo F N r w 0

/-- Pivots: `(v, j, p)`. -/
abbrev Piv := Nat × Nat × Nat

/-- One level `v` (`lv = ℓ^v`, `e = φ(N) − 1`, `t = tst c LOW ADD TOP`), with fuel `≥` the number of pending rows:
returns the rows that tested zero (all fields divisible by `ℓ^{v+1}`, cleared at every new pivot column) and the pivots. -/
def level (F H N w v lv e : Nat) (t : Nat → Nat) :
    Nat → List Nat → List Nat → List Piv → List Nat × List Piv :=
  @Nat.rec (fun _ => List Nat → List Nat → List Piv → List Nat × List Piv)
    (fun _ done piv => (done, piv))
    (fun _ ih pend done piv =>
      @List.rec Nat (fun _ => List Nat × List Piv) (done, piv)
        (fun r rest _ =>
          force (t r) fun tr =>
          @Bool.rec (fun _ => List Nat × List Piv)
            (force (firstNZ F tr w 0) fun j =>
             force (redRow F N w r) fun p =>
             force (powMod H (Nat.div (fld F p j) lv) e N) fun ui =>
               ih (mapA (elim F N lv ui j p) rest []) (mapA (elim F N lv ui j p) done []) ((v, j, p) :: piv))
            (ih rest (r :: done) piv)
            (Nat.beq tr 0))
        pend)

/-- One level with its constants `(c, b, T) = lvC l H v` and masks prepared once. -/
def levelAt (l H N w e R v : Nat) (rows : List Nat) (piv : List Piv) : List Nat × List Piv :=
  force (Nat.pow l v) fun lv =>
  @Prod.rec Nat (Nat × Nat) (fun _ => List Nat × List Piv) (fun c bT =>
    force c fun c =>
    force (rep (Nat.mul 2 H) (Nat.sub (Nat.pow 2 bT.1) 1) w) fun LOW =>
    force (rep (Nat.mul 2 H) (Nat.sub (Nat.sub (Nat.pow 2 H) 1) bT.2) w) fun ADD =>
    force (rep (Nat.mul 2 H) (Nat.pow 2 H) w) fun TOP =>
      level (Nat.mul 2 H) H N w v lv e (tst c LOW ADD TOP) R rows [] piv) (lvC l H v)

/-- Levels `v, v+1, …, v+k−1`. -/
def levels (l H N w e R : Nat) : Nat → Nat → List Nat → List Piv → List Piv :=
  @Nat.rec (fun _ => Nat → List Nat → List Piv → List Piv) (fun _ _ piv => piv)
    (fun _ ih v rows piv =>
      @Prod.rec (List Nat) (List Piv) (fun _ => List Piv) (fun rows' piv' => ih (Nat.succ v) rows' piv')
        (levelAt l H N w e R v rows piv))

/-- **`smithPK l K H w rows`**: the pivots `(v, j, p)` (latest first) of the elimination over `ℤ/ℓ^K` of the rows
(`w` fields of width `2H`, entries `< ℓ^K`). -/
def smithPK (l K H w : Nat) (rows : List Nat) : List Piv :=
  levels l H (Nat.pow l K) w (Nat.sub (Nat.mul (Nat.pow l (Nat.sub K 1)) (Nat.sub l 1)) 1) (List.length rows) K 0 rows []

/-- The valuations of the pivots, latest first (non-increasing). -/
def vals (P : List Piv) : List Nat := @List.rec Piv (fun _ => List Nat) [] (fun a _ ih => a.1 :: ih) P

/-- **The `ℓ`-adic valuations** of the elementary divisors `≠ 0 mod ℓ^K`, in non-increasing order. -/
def smithVals (l K H w : Nat) (rows : List Nat) : List Nat := vals (smithPK l K H w rows)

/-- `Σ (K − v)` over the pivots: `ℓ^{lenOf}` is the number of elements of the row span (`smithPK_card`). -/
def lenOf (K : Nat) (P : List Piv) : Nat :=
  @List.rec Piv (fun _ => Nat) 0 (fun a _ ih => Nat.add (Nat.sub K a.1) ih) P

/-- The number of pivots of valuation `v`. -/
def cntV (v : Nat) (P : List Piv) : Nat :=
  @List.rec Piv (fun _ => Nat) 0 (fun a _ ih => @Bool.rec (fun _ => Nat) ih (Nat.succ ih) (Nat.beq a.1 v)) P

/-- The signature of a pivot list: `(Σ (K − v), [#{v = 0}, …, #{v = K − 1}])` — the multiset of valuations in `K + 1`
small numbers (a cell statement stays short; a list of `R` valuations would be long). -/
def sig (K : Nat) (P : List Piv) : Nat × List Nat :=
  (lenOf K P, @Nat.rec (fun _ => Nat → List Nat) (fun _ => []) (fun _ ih v => cntV v P :: ih (Nat.succ v)) K 0)

/-! ### Packed columns over `ℤ/N` (the generators of the cell matrices) -/

/-- `C(m, t)` for `t = 0, …, k − 1` from `C(m, t+1) = C(m, t)(m − t)/(t + 1)`, each entry `(±C(m,t)) mod N` with the sign
`(−1)^(m+t)` when `sg = true`, packed from field `0` (width `F`). -/
def binGo (F N m : Nat) (sg : Bool) : Nat → Nat → Nat → Nat :=
  @Nat.rec (fun _ => Nat → Nat → Nat) (fun _ _ => 0)
    (fun _ ih t c =>
      force (Nat.mod c N) fun cm =>
      force (@Bool.rec (fun _ => Nat) cm (Nat.mod (Nat.sub N cm) N)
              (Bool.and sg (Nat.beq (Nat.mod (Nat.add m t) 2) 1))) fun g =>
      force (Nat.div (Nat.mul c (Nat.sub m t)) (Nat.succ t)) fun c' =>
        Nat.add g (Nat.shiftLeft (ih (Nat.succ t) c') F))

/-- `Q_m = (z − 1)^m` over `ℤ/N`: entry `t` is `C(m, t)(−1)^(m+t) mod N`, `t ≤ m`. -/
def qrowK (F N m : Nat) : Nat := binGo F N m true (Nat.succ m) 0 1

/-- The unit vector at field `k`. -/
def eK (F k : Nat) : Nat := Nat.shiftLeft 1 (Nat.mul F k)

/-- `x` moved up by `k` fields. -/
def shF (F x k : Nat) : Nat := Nat.shiftLeft x (Nat.mul F k)

/-- `−x` field-wise up to reduction: `(N − 1) x`. -/
def negK (N x : Nat) : Nat := Nat.mul (Nat.sub N 1) x

/-- `(−1)^j x` up to reduction. -/
def sgnK (N j x : Nat) : Nat := @Bool.rec (fun _ => Nat) x (negK N x) (Nat.beq (Nat.mod j 2) 1)

/-- Column `j` of `ρ_n(S)` over `ℤ/N`, `S = !![0,−1;1,0]`: `(−1)^j u_{n−j}` (unreduced). -/
def colSK (F N n j : Nat) : Nat := sgnK N j (eK F (Nat.sub n j))

/-- Column `j` of `ρ_n(U)`, `U = !![0,−1;1,1]`: `z^{n−j} Q_j`. -/
def colUK (F N n j : Nat) : Nat := shF F (qrowK F N j) (Nat.sub n j)

/-- Column `j` of `ρ_n(U²)`, `U² = !![−1,−1;1,0]`: `(−1)^j Q_{n−j}` (unreduced). -/
def colUUK (F N n j : Nat) : Nat := sgnK N j (qrowK F N (Nat.sub n j))

/-- `(1 + ρ_n(U) + ρ_n(U²)) u_j` (unreduced). -/
def colN3K (F N n j : Nat) : Nat := Nat.add (Nat.add (eK F j) (colUK F N n j)) (colUUK F N n j)

/-! The columns of the block formulas of LR-Y's tree differentials (`GL2.MP.treeD0`, `treeDS 0`, `treeDS 1`, matched by hand;
the identification is not proved, LV audit M-3) on `Sym^n (ℤ/N)²`, `n` even
(`−I` acts trivially: `N_S = 2(1 + S)`, `N_U = 2(1 + U + U²)`, `edgA 0 = 0`, `edgA 1 = 2`), coordinates
`((x, y), z) ↦` fields `[0, d)`, `[d, 2d)`, `[2d, 3d)`, `d = n + 1`; every column reduced (`redRow`, `3d` fields). -/

/-- `d⁰ = treeD0`: `(x, y) ↦ ((1 − S)x, (1 − U)y, x − y)`; columns `x = u_j` then `y = u_j`. -/
def colD0x (F N n j : Nat) : Nat :=
  redRow F N (Nat.mul 3 (Nat.succ n))
    (Nat.add (Nat.add (eK F j) (negK N (colSK F N n j))) (shF F (eK F j) (Nat.mul 2 (Nat.succ n))))

def colD0y (F N n j : Nat) : Nat :=
  redRow F N (Nat.mul 3 (Nat.succ n))
    (Nat.add (shF F (Nat.add (eK F j) (negK N (colUK F N n j))) (Nat.succ n))
      (shF F (negK N (eK F j)) (Nat.mul 2 (Nat.succ n))))

/-- `d¹ = treeDS 0`: `((x, y), z) ↦ (2(1 + S)x, 2N₃y, (1 + S)x − N₃y)`; columns `x = u_j`, `y = u_j` (the `z`-columns
vanish). -/
def colD1x (F N n j : Nat) : Nat :=
  redRow F N (Nat.mul 3 (Nat.succ n))
    (Nat.add (Nat.mul 2 (Nat.add (eK F j) (colSK F N n j)))
      (shF F (Nat.add (eK F j) (colSK F N n j)) (Nat.mul 2 (Nat.succ n))))

def colD1y (F N n j : Nat) : Nat :=
  redRow F N (Nat.mul 3 (Nat.succ n))
    (Nat.add (shF F (Nat.mul 2 (colN3K F N n j)) (Nat.succ n))
      (shF F (negK N (colN3K F N n j)) (Nat.mul 2 (Nat.succ n))))

/-- `d² = treeDS 1`: `((x, y), z) ↦ ((1 − S)x, (1 − U)y, x − y − 2z)`; columns `x`, `y` as `d⁰`, then `z = u_j`. -/
def colD2z (F N n j : Nat) : Nat :=
  redRow F N (Nat.mul 3 (Nat.succ n)) (shF F (negK N (Nat.mul 2 (eK F j))) (Nat.mul 2 (Nat.succ n)))

/-- `[f 0, …, f (k − 1)]`. -/
def tab (f : Nat → Nat) (k : Nat) : List Nat :=
  @Nat.rec (fun _ => Nat → List Nat) (fun _ => []) (fun _ ih j => force (f j) fun c => c :: ih (Nat.succ j)) k 0

/-- The columns of `d⁰`, `d¹`, `d²` (`2d`, `2d`, `3d` packed columns of `3d` fields of width `F`). -/
def colsD0 (F N n : Nat) : List Nat := tab (colD0x F N n) (Nat.succ n) ++ tab (colD0y F N n) (Nat.succ n)
def colsD1 (F N n : Nat) : List Nat := tab (colD1x F N n) (Nat.succ n) ++ tab (colD1y F N n) (Nat.succ n)
def colsD2 (F N n : Nat) : List Nat := colsD0 F N n ++ tab (colD2z F N n) (Nat.succ n)

/-- The pivots of `d^q` (`q = 0, 1, 2`) on `Sym^n (ℤ/ℓ^K)²` (`n` even), fields of width `2H`. -/
def cellPiv (l K H n q : Nat) : List Piv :=
  smithPK l K H (Nat.mul 3 (Nat.succ n))
    (@Nat.rec (fun _ => List Nat) (colsD0 (Nat.mul 2 H) (Nat.pow l K) n)
      (fun q' _ => @Nat.rec (fun _ => List Nat) (colsD1 (Nat.mul 2 H) (Nat.pow l K) n)
        (fun _ _ => colsD2 (Nat.mul 2 H) (Nat.pow l K) n) q') q)

/-- **The cell call**: the signature of the pivots of `d^q`. -/
def cellSig (l K H n q : Nat) : Nat × List Nat := sig K (cellPiv l K H n q)

/-- The valuations (list form, for small checks). -/
def cellVals (l K H n q : Nat) : List Nat := vals (cellPiv l K H n q)

end GL2.L23.SmithPK

end
