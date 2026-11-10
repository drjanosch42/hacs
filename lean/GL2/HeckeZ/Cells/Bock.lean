/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import Mathlib
import GL2.HeckeZ.Cells.BockData
import GL2.HeckeZ.Check.BockK7

/-!
# HeckeZ M-HZ3: the level-one cell statements of 03 (L-HZ2)

`07-lean-plan.md` §10, the row M-HZ3: K5 on the 54 cells A3 re-ran; K6 on `(1, 8)`; K7 `zero` at `(1, 18)`, `(1, 20)` and
`bounds` at `(1, 26)`, `(1, 30)` (all at `ℓ = 5`, `𝕋' = ℤ[T_2, T_3, T_5, T_7]`); K8 (`orc:`) in `GL2.HeckeZ.Cells.BockData`.

In every cell theorem `H` stands for `H¹(SL₂(ℤ); Sym^n ℤ²)` with Hecke operators `X` (`X p = T_p`), pairwise commuting; the trusted
inputs are hypotheses of the statement:
* **TB-χ** (`hχ`): `char_Z(T_p | L)` of the integral census ed.1;
* **TB-mat** (`hmat`, `hgen`, `hM`): the slot-25 EXPORT matrices of `descent(1, n, [2,3,5,7]).module(ℓ, K)` (the data of
  `GL2.HeckeZ.Cells.BockData`, sha256s in its docstrings).
The conclusions concern `J` relative to `tors_ℓ` (`Jset … S`); by `GL2.HeckeZ.Bock.Jset_subset_of_ellPart` the `ℓ`-primary part of
the glue relative to the full torsion lies in it.
-/

namespace GL2.HeckeZ.Cells.Bock

open GL2.HeckeZ.Exec GL2.HeckeZ.Check GL2.HeckeZ.Bock GL2.HeckeZ.Cells.BockData

/-! ### K5 -/

/-- **K5 on the 54 cells** `(1, n)`, `n ∈ {8, …, 60}`, `ℓ ∈ {2, 3}`: the Lean-kernel dimensions `h^q` equal the universal-coefficient
prediction `b_q + t_q + t_{q+1}` from the descent groups (B2(1), 03 §6.1(a), A3 §4.1: 54/54). -/
theorem k5_all_spec : ∀ c ∈ k5All, K5Spec c :=
  fun c hc => k5Check_sound (List.all_eq_true.mp k5All_check c hc)

variable {H : Type*} [AddCommGroup H]

/-! ### K6 at `(1, 8)`, `ℓ = 2` -/

/-- **K6 at `(1, 8)`, `ℓ = 2`, `θ = T_2`: separation, `J_2 = 0`.** `χ_ℤ(T_2 | L) = x − 513` (TB-χ), `tors_2 = ℤ/2 ⊕ ℤ/8` with
`T_2 = diag(0, 2)` on the export's generators (TB-mat): `gcd(x + 1, x²) = 1` in `𝔽_2[x]`, so `Hom_{ℤ[T_2]}(H/tors_2, tors_2) = 0` and
the glue of `𝕋 = ℤ[X_n]` relative to `tors_2` vanishes (03 §8.2's example; the torsion point `𝔪_x` is torsion-only, `S_{10} = 0`). -/
theorem cell_k6_n8 [Module.Free ℤ (H ⧸ Submodule.torsion ℤ H)] [Module.Finite ℤ (H ⧸ Submodule.torsion ℤ H)]
    (X : ℕ → Module.End ℤ H) (hX : ∀ i j, Commute (X i) (X j)) (g : ℕ → H)
    (hχ : (torsMap (X 2)).charpoly = toPoly [-513, 1])
    (hmat : TorsModel 2 (X 2) [1, 3] [[0, 0], [0, 2]] g)
    (hgen : ∀ x (j : ℕ), ((2 : ℕ) : ℤ) ^ j • x = 0 → x ∈ torsSpan 2 g) :
    ∀ t ∈ Jset (Algebra.adjoin ℤ (Set.range X) : Set (Module.End ℤ H)) (torsSpan 2 g), t = 0 :=
  k6_J_eq_zero (c := k6_n8) k6_n8_check Nat.prime_two (X 2) g hχ hmat hgen _
    fun _ ht => commute_of_mem_adjoin hX ht 2

/-! ### K7 at `ℓ = 5` -/

/-- **K7 at `(1, 18)`: `J_5 = 0` (status `zero`).** `tors_5 = ℤ/5`, `r = 3`; with `T_5` in `𝕋'` the Hom bound is `1`. -/
theorem cell_k7_n18 (X : ℕ → Module.End ℤ H) (hX : ∀ i j, Commute (X i) (X j)) (g : H) (f : ℕ → H)
    (hM : K7Model k7_n18 X g f) :
    ∀ t ∈ Jset (Algebra.adjoin ℤ (Set.range X) : Set (Module.End ℤ H)) (Submodule.span ℤ {g}), t = 0 :=
  k7_J_zero hX (k7Check_sound k7_n18_check) hM (by decide)

/-- **K7 at `(1, 20)`: `J_5 = 0` (status `zero`).** -/
theorem cell_k7_n20 (X : ℕ → Module.End ℤ H) (hX : ∀ i j, Commute (X i) (X j)) (g : H) (f : ℕ → H)
    (hM : K7Model k7_n20 X g f) :
    ∀ t ∈ Jset (Algebra.adjoin ℤ (Set.range X) : Set (Module.End ℤ H)) (Submodule.span ℤ {g}), t = 0 :=
  k7_J_zero hX (k7Check_sound k7_n20_check) hM (by decide)

/-- **K7 at `(1, 26)` (status `bounds`)**: split for `𝕋' = ℤ[T_2, T_3, T_5, T_7]`, so `J ∩ 𝕋' = 0`; and the glue `J` of
`𝕋 = ℤ[X_n]` is finite of order `≤ 25` (`|Hom_{𝕋'}(L/5, tors_5)| = 25`, 03 §6.4). -/
theorem cell_k7_n26 (X : ℕ → Module.End ℤ H) (hX : ∀ i j, Commute (X i) (X j)) (g : H) (f : ℕ → H)
    (hM : K7Model k7_n26 X g f) :
    (∀ t ∈ Jset (Algebra.adjoin ℤ (Pset k7_n26 X) : Set (Module.End ℤ H)) (Submodule.span ℤ {g}), t = 0) ∧
    (Jset (Algebra.adjoin ℤ (Set.range X) : Set (Module.End ℤ H)) (Submodule.span ℤ {g})).Finite ∧
    Nat.card (Jset (Algebra.adjoin ℤ (Set.range X) : Set (Module.End ℤ H)) (Submodule.span ℤ {g})) ≤ 25 := by
  have hs := k7Check_sound k7_n26_check
  obtain ⟨h1, h2⟩ := k7_J_card_le hX hs hM (by decide)
  refine ⟨k7_J_split hs hM rfl, h1, h2.trans (le_of_eq (by decide))⟩

/-- **K7 at `(1, 30)` (status `bounds`)**: `tors_5 = ℤ/25`; split for `𝕋'`, so `J ∩ 𝕋' = 0`; `J` finite of order `≤ 25`
(the Smith read over `ℤ/25` with valuations `(0, 0, 0, 1, 1)`). -/
theorem cell_k7_n30 (X : ℕ → Module.End ℤ H) (hX : ∀ i j, Commute (X i) (X j)) (g : H) (f : ℕ → H)
    (hM : K7Model k7_n30 X g f) :
    (∀ t ∈ Jset (Algebra.adjoin ℤ (Pset k7_n30 X) : Set (Module.End ℤ H)) (Submodule.span ℤ {g}), t = 0) ∧
    (Jset (Algebra.adjoin ℤ (Set.range X) : Set (Module.End ℤ H)) (Submodule.span ℤ {g})).Finite ∧
    Nat.card (Jset (Algebra.adjoin ℤ (Set.range X) : Set (Module.End ℤ H)) (Submodule.span ℤ {g})) ≤ 25 := by
  have hs := k7Check_sound k7_n30_check
  obtain ⟨h1, h2⟩ := k7_J_card_le hX hs hM (by decide)
  refine ⟨k7_J_split hs hM rfl, h1, h2.trans (le_of_eq (by decide))⟩

end GL2.HeckeZ.Cells.Bock
