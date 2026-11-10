/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.HeckeZ.Cells.K9Data
import GL2.HeckeZ.Check.K9

/-!
# K9 on the 12 irregular pairs of H1: the spec and the finite-weight theorem (L-K9)

`k9Cells_spec`: every certificate of `K9Data` satisfies `K9Spec` (from `k9Cells_check`, kernel, and `k9_sound`). `k9Cells_cert`:
for every pair and **every** monic lift `F̃ ∈ ℤ_ℓ[x]` of degree `g` of `f_k mod ℓ²`, the finite-weight theorem holds at
`e_k = 1 + 2^{k−1}`: `F̃(x + e_k)` Eisenstein, `F̃` irreducible over `ℤ_ℓ` and `ℚ_ℓ`, `ℤ_ℓ[x]/(F̃)` a DVR of rank `g` with maximal
ideal `(x − e_k)` and `ℤ_ℓ[x]/(F̃, x − e_k) ≅ ℤ/ℓ`. `k9Cells_lift`: such a lift exists (the integer polynomial itself), so the
statement is not vacuous. `cell547_cert`: the (547, 486) instance spelled out ([U] `prop:u-H12` (D1) at `k = 486`).
-/

namespace GL2.HeckeZ.Cells.K9

open Polynomial GL2.HeckeZ.Exec.K9 GL2.HeckeZ.K9 GL2.HeckeZ.K10 GL2.HeckeZ.Cells.K9Data

/-- Every K9 certificate of H1 satisfies `K9Spec`. -/
theorem k9Cells_spec : ∀ c ∈ k9Cells, K9Spec c :=
  fun c hc => k9_sound c (List.all_eq_true.mp k9Cells_check c hc)

/-- **The 12 pairs, certified**: the finite-weight theorem for every monic lift of degree `g` of `f_k mod ℓ²`. -/
theorem k9Cells_cert : ∀ c ∈ k9Cells, ∀ [Fact c.ell.Prime] (F' : ℤ_[c.ell][X]), F'.Monic → F'.natDegree = c.g →
    F'.map (PadicInt.toZModPow 2) = c.Fp.map (Int.castRingHom (ZMod (c.ell ^ 2))) →
    EisFiniteWeight F' (1 + 2 ^ (c.k - 1)) :=
  fun c hc _ F' hm hd hF => k9_cert c (List.all_eq_true.mp k9Cells_check c hc) F' hm hd hF

/-- A lift exists: the integer polynomial `f_k` itself, read in `ℤ_ℓ[x]`. -/
theorem k9Cells_lift : ∀ c ∈ k9Cells, ∀ [Fact c.ell.Prime],
    (c.Fp.map (Int.castRingHom ℤ_[c.ell])).Monic ∧ (c.Fp.map (Int.castRingHom ℤ_[c.ell])).natDegree = c.g ∧
    (c.Fp.map (Int.castRingHom ℤ_[c.ell])).map (PadicInt.toZModPow 2) = c.Fp.map (Int.castRingHom (ZMod (c.ell ^ 2))) := by
  intro c _ _
  refine ⟨(GL2.HeckeZ.K9.Fp_monic c).map _, by rw [natDegree_map_eq_of_injective (RingHom.injective_int _), GL2.HeckeZ.K9.Fp_natDegree], ?_⟩
  rw [Polynomial.map_map]; congr 1; exact RingHom.ext_int _ _

/-- `547` is prime (for `ℤ_[547]`). -/
instance fact_prime_547 : Fact (Nat.Prime 547) := ⟨by norm_num⟩

/-- The pair `(547, 486)` ([U] `prop:u-H12` (D1) at `k = 486`): every monic quadratic `F̃ ∈ ℤ_547[x]` with
`F̃ ≡ x² + 62064 x + 135932 (mod 547²)` satisfies the finite-weight theorem at `e = 1 + 2^485`. -/
theorem cell547_cert (F' : ℤ_[547][X]) (hm : F'.Monic) (hd : F'.natDegree = 2)
    (hF : F'.map (PadicInt.toZModPow 2) = (toPoly [135932, 62064, 1]).map (Int.castRingHom (ZMod (547 ^ 2)))) :
    EisFiniteWeight F' (1 + 2 ^ 485) :=
  k9Cells_cert _ (List.mem_cons_self ..) F' hm hd hF

end GL2.HeckeZ.Cells.K9
