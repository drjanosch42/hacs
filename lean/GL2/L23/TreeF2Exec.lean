/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.L23.RankF2Exec

/-!
# LV-1 (slot 22): the two `F₂`-matrices of the tree complex, bit-packed — the executable part

`L23_master_plan.md` §4.2(2), LV row; `LV0_audit.md` R1 (F1), R3 (F3), R4 (F4). Mathlib-free, so a cell module that
only runs the kernel does not load Mathlib (the LV-0 design rule).

On `V = Sym^n F₂²` (`d = n + 1`, basis `u_t = x^{n-t} y^t`, [C] `eq:rhondef`) the generators of `SL₂(ℤ)` reduce to
`S ≡ !![0,1;1,0]`, `U ≡ !![0,1;1,1]`, `U² ≡ !![1,1;1,0]` (mod 2), and their columns are rows of Pascal's triangle mod 2
(`pascal m`, bit `t` = `C(m, t) mod 2`):

* `ρ_n(S) u_j = u_{n-j}`, so column `j` of `1 + ρ_n(S)` is `2^j ^^^ 2^(n-j)` (`colOnePlusS`, LV-0);
* `ρ_n(U) u_j = Σ_r C(j, r) u_{n-j+r}`: column `j` is `pascal j <<< (n - j)` (`colUp`);
* `ρ_n(U²) u_j = Σ_t C(n-j, t) u_t`: column `j` is `pascal (n - j)` (`colU2p`).

The proofs that these are the columns of `GL2.rhoMat n` over `ZMod 2` are in `GL2/L23/TreeF2.lean` (`vec_colUp`, …).
The two matrices whose ranks give every `dim H^q(SL₂(ℤ); Sym^n F₂²)` (`GL2.L23.TreeF2.finrank_H_zero`, `finrank_H_succ`):

* `colsA n` (`2d` columns of height `d`): `[1 + ρ_n(S) | 1 + ρ_n(U) + ρ_n(U²)]`, the odd-degree tree differential;
* `colsSig n` (`d` columns of height `2d`): `[1 + ρ_n(S) ; 1 + ρ_n(U)]` (stacked; `1 + ρ_n(S)` in bits `0 … n`,
  `1 + ρ_n(U)` in bits `d … 2d − 1`), whose kernel is `H⁰`.

**Kernel formulation** (LV-0): recursors and raw `Nat` primitives; the Pascal rows are produced incrementally (one xor and
one shift per row, `pasGo`), never recomputed per column, and every list element is forced (`force`). Generation is
`O(d)` list steps against the `O(d²)` steps of the elimination.
-/

noncomputable section

namespace GL2.L23.RankF2

/-- Row `m` of Pascal's triangle mod 2, bit-packed: `pascal (m + 1) = pascal m ^^^ (pascal m <<< 1)`. -/
def pascal (m : Nat) : Nat := @Nat.rec (fun _ => Nat) 1 (fun _ p => Nat.xor p (Nat.shiftLeft p 1)) m

/-- `k` consecutive Pascal rows starting at `p` (`p = pascal j` gives `[pascal j, …, pascal (j + k − 1)]`). -/
def pasGo (k : Nat) : Nat → List Nat :=
  @Nat.rec (fun _ => Nat → List Nat) (fun _ => [])
    (fun _ ih p => p :: force (Nat.xor p (Nat.shiftLeft p 1)) ih) k

/-- `revGo l acc = l.reverse ++ acc`. -/
def revGo (l : List Nat) : List Nat → List Nat :=
  @List.rec Nat (fun _ => List Nat → List Nat) (fun acc => acc) (fun x _ ih acc => ih (x :: acc)) l

/-- Column `j` of `1 + ρ_n(U)` mod 2, from `p = pascal j`. -/
def colOnePlusUp (n j p : Nat) : Nat := Nat.xor (Nat.shiftLeft 1 j) (Nat.shiftLeft p (n - j))

/-- Column `j` of `1 + ρ_n(U) + ρ_n(U²)` mod 2, from `p = pascal j` and `q = pascal (n - j)`. -/
def colN3p (n j p q : Nat) : Nat := Nat.xor (colOnePlusUp n j p) q

/-- Column `j` of the stacked `[1 + ρ_n(S) ; 1 + ρ_n(U)]` (`2d` bits), from `p = pascal j`. -/
def colSigp (n j p : Nat) : Nat := Nat.xor (colOnePlusS n j) (Nat.shiftLeft (colOnePlusUp n j p) (n + 1))

/-- The `N₃` columns `j, j+1, …` from the lists `P` (`pascal j, …`) and `Q` (`pascal (n − j), …`). -/
def n3Go (n : Nat) (P : List Nat) : List Nat → Nat → List Nat :=
  @List.rec Nat (fun _ => List Nat → Nat → List Nat) (fun _ _ => [])
    (fun p _ ih Q j => @List.rec Nat (fun _ => List Nat) []
      (fun q Q' _ => force (colN3p n j p q) fun c => c :: ih Q' (j + 1)) Q) P

/-- The `σ` columns `j, j+1, …` from the list `P` (`pascal j, …`). -/
def sigGo (n : Nat) (P : List Nat) : Nat → List Nat :=
  @List.rec Nat (fun _ => Nat → List Nat) (fun _ => [])
    (fun p _ ih j => force (colSigp n j p) fun c => c :: ih (j + 1)) P

/-- **The odd-degree matrix** `[1 + ρ_n(S) | 1 + ρ_n(U) + ρ_n(U²)]` mod 2, by its `2(n + 1)` columns. -/
def colsA (n : Nat) : List Nat :=
  tabulate (colOnePlusS n) (n + 1) 0 ++ n3Go n (pasGo (n + 1) 1) (revGo (pasGo (n + 1) 1) []) 0

/-- **The stacked matrix** `[1 + ρ_n(S) ; 1 + ρ_n(U)]` mod 2, by its `n + 1` columns of `2(n + 1)` bits. -/
def colsSig (n : Nat) : List Nat := sigGo n (pasGo (n + 1) 1) 0

/-- The width bound `∀ r ∈ rows, r < 2^w` as a kernel-cheap `Bool` (`rowsLtK_eq`: it is `RankF2.rowsLt`). -/
def rowsLtK (w : Nat) (rows : List Nat) : Bool :=
  @List.rec Nat (fun _ => Bool) true (fun r _ ih => Bool.and (Nat.blt r (Nat.pow 2 w)) ih) rows

end GL2.L23.RankF2

end
