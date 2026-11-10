/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.L23.RankF3Exec

/-!
# LV (slot 23): `rankF2` / `rankF3` without `Nat.log2` in the kernel

**Finding (slot 23).** `Nat.log2` is *not* among the kernel's accelerated `Nat` operations: `decide +kernel` unfolds it
bit by bit (0.4 s per call on an 8 500-bit number, measured). LV-0's `rankF2` and LV-2's `rankF3` call it once per inserted
row, on rows of `d` or `2d` bits, so it dominates large cells: `rankF2 (colsSig 1000)` takes 46 s / 5.3 GiB, the same
elimination with the binary-search `lg2` below 21 s / 3.0 GiB (measured, one declaration, 8 GiB guard).

`lg2 x` is `⌊log₂ x⌋` by a binary search on `x >>> 2^k`, `k = 15, …, 0` (16 shifts and tests, GMP-accelerated), with
`Nat.log2` as the fallback for `x ≥ 2^65536` (never taken by a cell). **`lg2_eq : lg2 x = Nat.log2 x` for every `x`**,
so `rankF2F = rankF2` and `rankF3F = rankF3` (`rankF2F_eq`, `rankF3F_eq`), by the same unfolding: a cell may decide
`rankF2F rows = s` and rewrite. Mathlib-free (core lemmas only); the LV-0/1/2 modules are unchanged.
-/

noncomputable section

namespace GL2.L23.RankF2

/-- Binary search: `acc + ⌊log₂ x⌋` for `x < 2^(2^k)`. -/
def lgGo : Nat → Nat → Nat → Nat :=
  @Nat.rec (fun _ => Nat → Nat → Nat) (fun _ acc => acc)
    (fun k ih x acc => force (Nat.shiftRight x (Nat.pow 2 k)) fun y =>
      @Bool.rec (fun _ => Nat) (ih y (Nat.add acc (Nat.pow 2 k))) (ih x acc) (Nat.beq y 0))

/-- `⌊log₂ x⌋` by `lgGo 16`, `Nat.log2` beyond `2^65536`. -/
def lg2 (x : Nat) : Nat :=
  @Bool.rec (fun _ => Nat) (Nat.log2 x) (lgGo 16 x 0) (Nat.beq (Nat.shiftRight x (Nat.pow 2 16)) 0)

theorem force_eq' {α : Sort _} (a : Nat) (k : Nat → α) : force a k = k a := by
  unfold force; cases Nat.beq a 0 <;> rfl

theorem log2_add_of_div {x b : Nat} (hy : x / 2 ^ b ≠ 0) : Nat.log2 x = b + Nat.log2 (x / 2 ^ b) := by
  have hx : x ≠ 0 := fun h => hy (by rw [h, Nat.zero_div])
  rw [Nat.log2_eq_iff hx]
  have h1 := Nat.log2_self_le hy
  have h2 := @Nat.lt_log2_self (x / 2 ^ b)
  constructor
  · rw [Nat.pow_add, Nat.mul_comm]
    exact (Nat.le_div_iff_mul_le (Nat.two_pow_pos b)).mp h1
  · rw [show b + Nat.log2 (x / 2 ^ b) + 1 = (Nat.log2 (x / 2 ^ b) + 1) + b by omega, Nat.pow_add]
    exact (Nat.div_lt_iff_lt_mul (Nat.two_pow_pos b)).mp h2

theorem lgGo_eq : ∀ k x acc, x < 2 ^ (2 ^ k) → lgGo k x acc = acc + Nat.log2 x
  | 0, x, acc, hx => by
    show acc = _
    have : Nat.log2 x = 0 := by
      rcases Nat.lt_or_ge x 1 with h | h
      · rw [show x = 0 by omega]; rfl
      · exact Nat.le_zero.mp (Nat.lt_succ_iff.mp ((Nat.log2_lt (by omega)).mpr hx))
    rw [this, Nat.add_zero]
  | k + 1, x, acc, hx => by
    show force (x >>> 2 ^ k) (fun y => @Bool.rec (fun _ => Nat) (lgGo k y (acc + 2 ^ k)) (lgGo k x acc)
      (Nat.beq y 0)) = _
    rw [force_eq', Nat.shiftRight_eq_div_pow]
    have hsq : 2 ^ (2 ^ (k + 1)) = 2 ^ (2 ^ k) * 2 ^ (2 ^ k) := by rw [← Nat.pow_add, Nat.pow_succ]; congr 1; omega
    by_cases hy : x / 2 ^ (2 ^ k) = 0
    · rw [hy, show Nat.beq 0 0 = true from rfl]
      exact lgGo_eq k x acc (by rwa [Nat.div_eq_zero_iff_lt (Nat.two_pow_pos _)] at hy)
    · rw [show Nat.beq (x / 2 ^ 2 ^ k) 0 = false from Bool.eq_false_iff.mpr fun h => hy (Nat.eq_of_beq_eq_true h)]
      have hlt : x / 2 ^ 2 ^ k < 2 ^ 2 ^ k := by
        rw [Nat.div_lt_iff_lt_mul (Nat.two_pow_pos _)]; rwa [hsq] at hx
      show lgGo k (x / 2 ^ 2 ^ k) (acc + 2 ^ k) = _
      rw [lgGo_eq k _ _ hlt, log2_add_of_div hy]
      omega

/-- **`lg2` is `Nat.log2`.** -/
theorem lg2_eq (x : Nat) : lg2 x = Nat.log2 x := by
  unfold lg2
  by_cases h : x >>> 2 ^ 16 = 0
  · rw [show Nat.beq (Nat.shiftRight x (Nat.pow 2 16)) 0 = true from by
      show Nat.beq (x >>> 2 ^ 16) 0 = true; rw [h]; rfl]
    show lgGo 16 x 0 = _
    rw [lgGo_eq 16 x 0 (by rwa [Nat.shiftRight_eq_div_pow, Nat.div_eq_zero_iff_lt (Nat.two_pow_pos _)] at h),
      Nat.zero_add]
  · rw [show Nat.beq (Nat.shiftRight x (Nat.pow 2 16)) 0 = false from
      Bool.eq_false_iff.mpr fun h' => h (Nat.eq_of_beq_eq_true h')]

/-! ### `rankF2F` -/

/-- `addRow` with `lg2`. -/
def addRowF (B : List (Nat × Nat)) (r : Nat) : List (Nat × Nat) :=
  @Bool.rec (fun _ => List (Nat × Nat)) (insertP (lg2 (reduce B r)) (reduce B r) B) B (Nat.beq (reduce B r) 0)

/-- `basis` with `addRowF`. -/
def basisF (rs : List Nat) : List (Nat × Nat) → List (Nat × Nat) :=
  @List.rec Nat (fun _ => List (Nat × Nat) → List (Nat × Nat)) (fun B => B) (fun r _ ih B => ih (addRowF B r)) rs

/-- **`rankF2` with the fast `lg2`.** -/
def rankF2F (rows : List Nat) : Nat := (basisF rows []).length

theorem addRowF_eq (B : List (Nat × Nat)) (r : Nat) : addRowF B r = addRow B r := by
  unfold addRowF addRow; rw [lg2_eq]

theorem basisF_eq : ∀ (rs : List Nat) (B : List (Nat × Nat)), basisF rs B = basis rs B
  | [], _ => rfl
  | r :: rs, B => by
    show basisF rs (addRowF B r) = basis rs (addRow B r)
    rw [addRowF_eq, basisF_eq rs]

theorem rankF2F_eq (rows : List Nat) : rankF2F rows = rankF2 rows := by
  unfold rankF2F rankF2; rw [basisF_eq]

end GL2.L23.RankF2

namespace GL2.L23.RankF3

open GL2.L23.RankF2 (lg2 lg2_eq force)

/-- `addRed3` with `lg2`. -/
def addRed3F (B : List (Nat × (Nat × Nat))) (r : Nat × Nat) : List (Nat × (Nat × Nat)) :=
  @Bool.rec (fun _ => List (Nat × (Nat × Nat))) (insertP3 (lg2 (supp r)) (norm3 (lg2 (supp r)) r) B) B
    (Nat.beq (supp r) 0)

/-- `addRow3` with `addRed3F`. -/
def addRow3F (B : List (Nat × (Nat × Nat))) (r : Nat × Nat) : List (Nat × (Nat × Nat)) :=
  force2 (reduce3 B r) (addRed3F B)

/-- `basis3` with `addRow3F`. -/
def basis3F (rs : List (Nat × Nat)) : List (Nat × (Nat × Nat)) → List (Nat × (Nat × Nat)) :=
  @List.rec (Nat × Nat) (fun _ => List (Nat × (Nat × Nat)) → List (Nat × (Nat × Nat))) (fun B => B)
    (fun r _ ih B => ih (addRow3F B r)) rs

/-- **`rankF3` with the fast `lg2`.** -/
def rankF3F (rows : List (Nat × Nat)) : Nat := (basis3F rows []).length

theorem addRed3F_eq (B : List (Nat × (Nat × Nat))) (r : Nat × Nat) : addRed3F B r = addRed3 B r := by
  unfold addRed3F addRed3; rw [lg2_eq]

theorem addRow3F_eq (B : List (Nat × (Nat × Nat))) (r : Nat × Nat) : addRow3F B r = addRow3 B r := by
  unfold addRow3F addRow3
  have : addRed3F B = addRed3 B := funext (addRed3F_eq B)
  rw [this]

theorem basis3F_eq : ∀ (rs : List (Nat × Nat)) (B : List (Nat × (Nat × Nat))), basis3F rs B = basis3 rs B
  | [], _ => rfl
  | r :: rs, B => by
    show basis3F rs (addRow3F B r) = basis3 rs (addRow3 B r)
    rw [addRow3F_eq, basis3F_eq rs]

theorem rankF3F_eq (rows : List (Nat × Nat)) : rankF3F rows = rankF3 rows := by
  unfold rankF3F rankF3; rw [basis3F_eq]

end GL2.L23.RankF3

end
