/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.L23.RankF2Exec

/-!
# LV-2 (slot 22): `rankF3`, Gaussian elimination over `F₃` on two bit-planes — the executable part

`L23_master_plan.md` §4.2(2), `LV1_audit.md` §5.2. A row of an `F₃`-matrix is a pair `(p, m)` of naturals with disjoint
bits: entry `j` is `1` if bit `j` of `p` is set, `2 = −1` if bit `j` of `m` is set, `0` otherwise. Negation swaps the planes;
addition is the bitwise formula `add3` (six GMP-accelerated `land` / `lor` / `xor` per row operation instead of one `xor`).

* `reduce3 B r`: one pass of `r` over the basis `B` (pairs `(pivot, row)`, strictly descending pivots, each row normalised
  to entry `1` at its pivot, nothing above it): `r := r − b` if `r` has `1` at the pivot, `r := r + b` if it has `2`.
* `addRow3 B r`: a non-zero remainder is normalised (negated if its top entry is `2`) and inserted at its top bit.
* `rankF3 rows`: the length of the basis after all rows are processed.

Soundness (`rankF3 = Matrix.rank` over `ZMod 3`) is `GL2.L23.RankF3.rankF3_eq_rank` in `GL2/L23/RankF3.lean`.

**The `Sym^n F₃²` generators.** Over `F₃`, `S ≡ !![0,2;1,0]`, `U ≡ !![0,2;1,1]`, `U² ≡ !![2,2;1,0]`. With
`Q_m := (z − 1)^m` (bit-packed, entry `t` = `C(m, t)(−1)^(m+t)`, `Q_{m+1} = z·Q_m − Q_m`), the columns of `ρ_n` in the basis
`u_t = x^{n−t} y^t` ([C] `eq:rhondef`) are (`GL2/L23/TreeF3Cols.lean`)

* `ρ_n(S) u_j = (−1)^j u_{n−j}`;   `ρ_n(U) u_j = z^{n−j} Q_j`;   `ρ_n(U²) u_j = (−1)^j Q_{n−j}`.

The four matrices of the `ℓ = 3`, `n` even cell theorem (`GL2.L23.TreeF3.V3_dims_even`) are given by their `n + 1` columns:
`1 + ρ_n(S)` (`colsPS`), `1 − ρ_n(U)` (`colsMU`), `1 + ρ_n(U) + ρ_n(U²)` (`colsN3`) and the stacked `[1 − ρ_n(S) ; 1 − ρ_n(U)]`
(`colsSig3`, `2(n + 1)` entries). Same kernel formulation as `RankF2Exec` (recursors, raw primitives, `force`).
-/

noncomputable section

namespace GL2.L23.RankF3

open GL2.L23.RankF2 (force bit)

/-- Force both planes of a row. -/
@[inline] def force2 {α : Sort _} (x : Nat × Nat) (k : Nat × Nat → α) : α :=
  force x.1 fun a => force x.2 fun b => k (a, b)

/-- `a &&& ~~~b`. -/
def andNot (a b : Nat) : Nat := Nat.xor a (Nat.land a b)

/-- Entrywise sum over `F₃` of two rows with disjoint planes. -/
def add3 (x y : Nat × Nat) : Nat × Nat :=
  force (Nat.lor y.1 y.2) fun u => force (Nat.lor x.1 x.2) fun v =>
    (Nat.lor (Nat.lor (andNot x.1 u) (andNot y.1 v)) (Nat.land x.2 y.2),
     Nat.lor (Nat.lor (andNot x.2 u) (andNot y.2 v)) (Nat.land x.1 y.1))

/-- Negation: swap the planes. -/
def neg3 (x : Nat × Nat) : Nat × Nat := (x.2, x.1)

/-- Multiplication by `z^k`. -/
def shl3 (x : Nat × Nat) (k : Nat) : Nat × Nat := (Nat.shiftLeft x.1 k, Nat.shiftLeft x.2 k)

/-- One basis step: clear position `q` of `r` with the (normalised) basis row `b`. -/
def step3 (r : Nat × Nat) (qb : Nat × (Nat × Nat)) : Nat × Nat :=
  @Bool.rec (fun _ => Nat × Nat)
    (@Bool.rec (fun _ => Nat × Nat) r (force2 (add3 r qb.2) fun c => c) (bit r.2 qb.1))
    (force2 (add3 r (neg3 qb.2)) fun c => c) (bit r.1 qb.1)

/-- Reduce `r` by a basis (one pass, in list order). -/
def reduce3 (B : List (Nat × (Nat × Nat))) : Nat × Nat → Nat × Nat :=
  @List.rec (Nat × (Nat × Nat)) (fun _ => Nat × Nat → Nat × Nat) (fun r => r) (fun qb _ ih r => ih (step3 r qb)) B

/-- Insert `(p, r)` into a list strictly descending in the first component. -/
def insertP3 (p : Nat) (r : Nat × Nat) (B : List (Nat × (Nat × Nat))) : List (Nat × (Nat × Nat)) :=
  @List.rec (Nat × (Nat × Nat)) (fun _ => List (Nat × (Nat × Nat))) [(p, r)]
    (fun qb B ih => @Bool.rec (fun _ => List (Nat × (Nat × Nat))) (qb :: ih) ((p, r) :: qb :: B) (Nat.blt qb.1 p)) B

/-- The support `p ||| m` of a row. -/
def supp (r : Nat × Nat) : Nat := Nat.lor r.1 r.2

/-- Normalise a non-zero reduced row at its top position `t`: negate it if the entry there is `2`. -/
def norm3 (t : Nat) (r : Nat × Nat) : Nat × Nat := @Bool.rec (fun _ => Nat × Nat) (neg3 r) r (bit r.1 t)

/-- Process one (already reduced) row. -/
def addRed3 (B : List (Nat × (Nat × Nat))) (r : Nat × Nat) : List (Nat × (Nat × Nat)) :=
  @Bool.rec (fun _ => List (Nat × (Nat × Nat))) (insertP3 (Nat.log2 (supp r)) (norm3 (Nat.log2 (supp r)) r) B) B
    (Nat.beq (supp r) 0)

/-- Process one row. -/
def addRow3 (B : List (Nat × (Nat × Nat))) (r : Nat × Nat) : List (Nat × (Nat × Nat)) :=
  force2 (reduce3 B r) (addRed3 B)

/-- The echelon basis of a list of rows, starting from the basis `B`. -/
def basis3 (rs : List (Nat × Nat)) : List (Nat × (Nat × Nat)) → List (Nat × (Nat × Nat)) :=
  @List.rec (Nat × Nat) (fun _ => List (Nat × (Nat × Nat)) → List (Nat × (Nat × Nat))) (fun B => B)
    (fun r _ ih B => ih (addRow3 B r)) rs

/-- **The rank over `F₃`** of the matrix with the given two-plane rows. -/
def rankF3 (rows : List (Nat × Nat)) : Nat := (basis3 rows []).length

/-- The row bound: disjoint planes, support below `2^w`, as a kernel-cheap `Bool`. -/
def rowsOK3 (w : Nat) (rows : List (Nat × Nat)) : Bool :=
  @List.rec (Nat × Nat) (fun _ => Bool) true
    (fun r _ ih => Bool.and (Bool.and (Nat.beq (Nat.land r.1 r.2) 0) (Nat.blt (supp r) (Nat.pow 2 w))) ih) rows

/-! ### The `Sym^n F₃²` generators -/

/-- The unit vector `u_k`. -/
def e3 (k : Nat) : Nat × Nat := (Nat.shiftLeft 1 k, 0)

/-- `(−1)^j · x`. -/
def sgn3 (j : Nat) (x : Nat × Nat) : Nat × Nat := @Bool.rec (fun _ => Nat × Nat) (neg3 x) x (Nat.beq (Nat.land j 1) 0)

/-- `Q_{m+1} = z·Q_m − Q_m`. -/
def qNext (x : Nat × Nat) : Nat × Nat := add3 (shl3 x 1) (neg3 x)

/-- `Q_m = (z − 1)^m` (entry `t` = `C(m, t)(−1)^(m+t)`). -/
def qrow (m : Nat) : Nat × Nat := @Nat.rec (fun _ => Nat × Nat) (1, 0) (fun _ x => qNext x) m

/-- `k` consecutive rows `Q_j, Q_{j+1}, …` from `x = Q_j`. -/
def qGo (k : Nat) : Nat × Nat → List (Nat × Nat) :=
  @Nat.rec (fun _ => Nat × Nat → List (Nat × Nat)) (fun _ => [])
    (fun _ ih x => x :: force2 (qNext x) ih) k

/-- `revGo` for rows. -/
def revGo3 (l : List (Nat × Nat)) : List (Nat × Nat) → List (Nat × Nat) :=
  @List.rec (Nat × Nat) (fun _ => List (Nat × Nat) → List (Nat × Nat)) (fun acc => acc) (fun x _ ih acc => ih (x :: acc)) l

/-- Column `j` of `1 + ρ_n(S)`: `u_j + (−1)^j u_{n−j}`. -/
def colPS (n j : Nat) : Nat × Nat := add3 (e3 j) (sgn3 j (e3 (n - j)))

/-- Column `j` of `1 − ρ_n(S)`. -/
def colMS (n j : Nat) : Nat × Nat := add3 (e3 j) (neg3 (sgn3 j (e3 (n - j))))

/-- Column `j` of `1 − ρ_n(U)`, from `q = Q_j`. -/
def colMU (n j : Nat) (q : Nat × Nat) : Nat × Nat := add3 (e3 j) (neg3 (shl3 q (n - j)))

/-- Column `j` of `1 + ρ_n(U) + ρ_n(U²)`, from `q = Q_j`, `q' = Q_{n−j}`. -/
def colN3 (n j : Nat) (q q' : Nat × Nat) : Nat × Nat := add3 (add3 (e3 j) (shl3 q (n - j))) (sgn3 j q')

/-- Column `j` of the stacked `[1 − ρ_n(S) ; 1 − ρ_n(U)]` (`1 − ρ_n(U)` shifted by `n + 1`), from `q = Q_j`. -/
def colSig3 (n j : Nat) (q : Nat × Nat) : Nat × Nat :=
  (Nat.lor (colMS n j).1 (Nat.shiftLeft (colMU n j q).1 (n + 1)),
   Nat.lor (colMS n j).2 (Nat.shiftLeft (colMU n j q).2 (n + 1)))

/-- `[f j, …, f (j + k − 1)]`, every element forced. -/
def tab3 (f : Nat → Nat × Nat) (k j : Nat) : List (Nat × Nat) :=
  @Nat.rec (fun _ => Nat → List (Nat × Nat)) (fun _ => []) (fun _ ih j => force2 (f j) fun c => c :: ih (j + 1)) k j

/-- Map a column builder over `Q_j, Q_{j+1}, …`. -/
def mapQ (g : Nat → Nat × Nat → Nat × Nat) (P : List (Nat × Nat)) : Nat → List (Nat × Nat) :=
  @List.rec (Nat × Nat) (fun _ => Nat → List (Nat × Nat)) (fun _ => [])
    (fun q _ ih j => force2 (g j q) fun c => c :: ih (j + 1)) P

/-- Map a two-argument column builder over `(Q_j, Q_{n−j}), …`. -/
def mapQQ (g : Nat → Nat × Nat → Nat × Nat → Nat × Nat) (P : List (Nat × Nat)) :
    List (Nat × Nat) → Nat → List (Nat × Nat) :=
  @List.rec (Nat × Nat) (fun _ => List (Nat × Nat) → Nat → List (Nat × Nat)) (fun _ _ => [])
    (fun q _ ih Q j => @List.rec (Nat × Nat) (fun _ => List (Nat × Nat)) []
      (fun q' Q' _ => force2 (g j q q') fun c => c :: ih Q' (j + 1)) Q) P

/-- **`1 + ρ_n(S)`** mod 3, by its `n + 1` columns. -/
def colsPS (n : Nat) : List (Nat × Nat) := tab3 (colPS n) (n + 1) 0

/-- **`1 − ρ_n(S)`** mod 3. -/
def colsMS (n : Nat) : List (Nat × Nat) := tab3 (colMS n) (n + 1) 0

/-- **`1 − ρ_n(U)`** mod 3. -/
def colsMU (n : Nat) : List (Nat × Nat) := mapQ (colMU n) (qGo (n + 1) (1, 0)) 0

/-- **`1 + ρ_n(U) + ρ_n(U²)`** mod 3. -/
def colsN3 (n : Nat) : List (Nat × Nat) :=
  mapQQ (colN3 n) (qGo (n + 1) (1, 0)) (revGo3 (qGo (n + 1) (1, 0)) []) 0

/-- **The stacked `[1 − ρ_n(S) ; 1 − ρ_n(U)]`** mod 3, by its `n + 1` columns of `2(n + 1)` entries. -/
def colsSig3 (n : Nat) : List (Nat × Nat) := mapQ (colSig3 n) (qGo (n + 1) (1, 0)) 0

end GL2.L23.RankF3

end
