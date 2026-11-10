/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import Mathlib

/-!
# HeckeZ 03 B15(1) (LB-7): the fibre product for two ideals (L-HZ2, M-HZ3)

`docs/implementation/slot24/heckez/03-bockstein-tor.md` §4.5 (B15(1)), §8.1 (LB-7); `07-lean-plan.md` §2.3 (LB-7, shared with
LC-9); [C] `prop:heckeZglue`(5)(a). mathlib has the Chinese remainder theorem for coprime ideals only
(`Ideal.quotientInfRingEquivPiQuotient`); the general fibre product is by hand.

For ideals `I, I'` of a commutative ring `R`, `fibreHom : R/(I ∩ I') → R/I × R/I'` is injective (`fibreHom_injective`) with image
the pairs agreeing in `R/(I + I')` (`mem_range_fibreHom`):
`R/(I ∩ I') ≅ R/I ×_{R/(I+I')} R/I'`. With `I = I_fr`, `I' = I_tors` (`I ∩ I' = J`): `𝕋/J ≅ 𝕋^fr ×_{C_ft} 𝕋^tors`.
-/

namespace GL2.HeckeZ.Bock

variable {R : Type*} [CommRing R]

/-- `R/(I ∩ I') → R/I × R/I'`. -/
def fibreHom (I I' : Ideal R) : R ⧸ (I ⊓ I') →+* (R ⧸ I) × (R ⧸ I') :=
  Ideal.Quotient.lift (I ⊓ I') ((Ideal.Quotient.mk I).prod (Ideal.Quotient.mk I')) fun z hz => by
    simp [Ideal.Quotient.eq_zero_iff_mem.mpr hz.1, Ideal.Quotient.eq_zero_iff_mem.mpr hz.2]

@[simp] theorem fibreHom_mk (I I' : Ideal R) (z : R) :
    fibreHom I I' (Ideal.Quotient.mk _ z) = (Ideal.Quotient.mk I z, Ideal.Quotient.mk I' z) := rfl

/-- **Injective**: the kernel of `R → R/I × R/I'` is `I ∩ I'`. -/
theorem fibreHom_injective (I I' : Ideal R) : Function.Injective (fibreHom I I') := by
  rw [injective_iff_map_eq_zero]
  intro a ha
  obtain ⟨z, rfl⟩ := Ideal.Quotient.mk_surjective a
  simp only [fibreHom_mk, Prod.mk_eq_zero, Ideal.Quotient.eq_zero_iff_mem] at ha
  exact Ideal.Quotient.eq_zero_iff_mem.mpr ⟨ha.1, ha.2⟩

/-- **The image**: `(x, y)` comes from `R` iff `x` and `y` agree in `R/(I + I')`. -/
theorem mem_range_fibreHom (I I' : Ideal R) (p : (R ⧸ I) × (R ⧸ I')) :
    p ∈ (fibreHom I I').range ↔
      Ideal.Quotient.factor (le_sup_left : I ≤ I ⊔ I') p.1 = Ideal.Quotient.factor (le_sup_right : I' ≤ I ⊔ I') p.2 := by
  obtain ⟨⟨a, b⟩, rfl⟩ : ∃ q : R × R, (Ideal.Quotient.mk I q.1, Ideal.Quotient.mk I' q.2) = p := by
    obtain ⟨a, ha⟩ := Ideal.Quotient.mk_surjective p.1
    obtain ⟨b, hb⟩ := Ideal.Quotient.mk_surjective p.2
    exact ⟨(a, b), by rw [ha, hb]⟩
  simp only [Ideal.Quotient.factor_mk]
  constructor
  · rintro ⟨w, hw⟩
    obtain ⟨z, rfl⟩ := Ideal.Quotient.mk_surjective w
    simp only [fibreHom_mk, Prod.mk.injEq, Ideal.Quotient.eq] at hw
    rw [Ideal.Quotient.eq]
    have : a - b = (z - b) - (z - a) := by ring
    rw [this]
    exact Ideal.sub_mem _ (Ideal.mem_sup_right hw.2) (Ideal.mem_sup_left hw.1)
  · intro h
    rw [Ideal.Quotient.eq] at h
    obtain ⟨i, hi, j, hj, hij⟩ := Submodule.mem_sup.mp h
    refine ⟨Ideal.Quotient.mk _ (a - i), ?_⟩
    simp only [fibreHom_mk, Prod.mk.injEq, Ideal.Quotient.eq]
    refine ⟨by simpa using I.neg_mem hi, ?_⟩
    have : a - i - b = j := by linear_combination -hij
    rw [this]; exact hj

end GL2.HeckeZ.Bock
