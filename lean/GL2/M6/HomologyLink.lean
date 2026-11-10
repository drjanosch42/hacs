/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.Model.RmkOrbifold

/-!
# M6-A2 (slot 23): the hand-built cohomology `Z/B` against mathlib's homology, on classes

[GT] `gt_part_M6.tex`, row `lem:m6-homology-mk`. Used by `GL2.M6.ModelHecke` (A2) to compute `Θ_can` on classes.

`CochainCx.homologyEquiv` (M3-d, `lem:cochaincx-homology`) identifies the cone layer's hand-built `H^n = Z^n / B^n` with
mathlib's homology of the non-negative part. **`homologyEquiv_mk`**: the class of a cocycle `z` goes to `homologyπ` of the
cycle whose underlying cochain is `z` (an explicit cycle `c` with `iCycles c = z`), in every degree.
-/

noncomputable section

namespace GL2.Cone.CochainCx

open CategoryTheory

variable {R : Type} [CommRing R]

lemma homologyMk_inv_mk {M N P : Type} [AddCommGroup M] [AddCommGroup N] [AddCommGroup P] [Module R M] [Module R N]
    [Module R P] (f : M →ₗ[R] N) (g : N →ₗ[R] P) (w : g ∘ₗ f = 0) (z : LinearMap.ker g) :
    (homologyMk f g w).inv.hom (Submodule.Quotient.mk z) =
      (scMk f g w).homologyπ.hom ((scMk f g w).moduleCatCyclesIso.inv.hom z) :=
  (ShortComplex.moduleCatCyclesIso_inv_π_apply (scMk f g w) z).symm

lemma homologyMapIso_inv_π {S₁ S₂ : ShortComplex (ModuleCat.{0} R)} [S₁.HasHomology] [S₂.HasHomology] (e : S₁ ≅ S₂)
    (x : S₂.cycles) :
    (ShortComplex.homologyMapIso e).inv.hom (S₂.homologyπ.hom x) =
      S₁.homologyπ.hom ((ShortComplex.cyclesMap e.inv).hom x) := by
  have := congrArg (fun F => F.hom x) (ShortComplex.homologyπ_naturality e.inv)
  exact this

lemma homologyIsoSc'_inv_π (K : CochainComplex (ModuleCat.{0} R) ℕ) (i j k : ℕ) (hi : (ComplexShape.up ℕ).prev j = i)
    (hk : (ComplexShape.up ℕ).next j = k) (x : (K.sc' i j k).cycles) :
    (K.homologyIsoSc' i j k hi hk).inv.hom ((K.sc' i j k).homologyπ.hom x) =
      (K.homologyπ j).hom ((K.cyclesIsoSc' i j k hi hk).inv.hom x) := by
  have := congrArg (fun F => F.hom x) (HomologicalComplex.π_homologyIsoSc'_inv K i j k hi hk)
  exact this

lemma cyclesIsoSc'_inv_i (K : CochainComplex (ModuleCat.{0} R) ℕ) (i j k : ℕ) (hi : (ComplexShape.up ℕ).prev j = i)
    (hk : (ComplexShape.up ℕ).next j = k) (x : (K.sc' i j k).cycles) :
    (K.iCycles j).hom ((K.cyclesIsoSc' i j k hi hk).inv.hom x) = (K.sc' i j k).iCycles.hom x := by
  have := congrArg (fun F => F.hom x) (HomologicalComplex.cyclesIsoSc'_inv_iCycles K i j k hi hk)
  exact this

lemma cyclesMap_i_el {S₁ S₂ : ShortComplex (ModuleCat.{0} R)} [S₁.HasLeftHomology] [S₂.HasLeftHomology] (φ : S₁ ⟶ S₂)
    (x : S₁.cycles) : S₂.iCycles.hom ((ShortComplex.cyclesMap φ).hom x) = φ.τ₂.hom (S₁.iCycles.hom x) := by
  have := congrArg (fun F => F.hom x) (ShortComplex.cyclesMap_i φ)
  exact this

lemma moduleCatCyclesIso_inv_i (S : ShortComplex (ModuleCat.{0} R)) (x : LinearMap.ker S.g.hom) :
    S.iCycles.hom (S.moduleCatCyclesIso.inv.hom x) = (x : S.X₂) :=
  ShortComplex.moduleCatCyclesIso_inv_iCycles_apply S x

variable (C : CochainCx.{0} R)

lemma homologyEquivSucc_mk (n : ℕ) (z : C.Z ((n + 1 : ℕ) : ℤ)) :
    ∃ c : C.toNat.cycles (n + 1), (C.toNat.iCycles (n + 1)).hom c = (z : C.X ((n + 1 : ℕ) : ℤ)) ∧
      C.homologyEquivSucc n (Submodule.Quotient.mk z) = (C.toNat.homologyπ (n + 1)).hom c := by
  refine ⟨?c, ?h1, ?h2⟩
  case h2 =>
    unfold homologyEquivSucc
    erw [LinearEquiv.trans_apply, Submodule.quotEquivOfEq_mk]
    erw [Iso.toLinearEquiv_apply, Iso.symm_hom, Iso.trans_inv, Iso.trans_inv]
    erw [ConcreteCategory.comp_apply, ConcreteCategory.comp_apply]
    erw [homologyMk_inv_mk, homologyMapIso_inv_π, homologyIsoSc'_inv_π]
  case h1 =>
    erw [cyclesIsoSc'_inv_i, cyclesMap_i_el, moduleCatCyclesIso_inv_i]

lemma homologyEquivZero_mk (h : Subsingleton (C.X (-1))) (z : C.Z ((0 : ℕ) : ℤ)) :
    ∃ c : C.toNat.cycles 0, (C.toNat.iCycles 0).hom c = (z : C.X ((0 : ℕ) : ℤ)) ∧
      C.homologyEquivZero h (Submodule.Quotient.mk z) = (C.toNat.homologyπ 0).hom c := by
  refine ⟨?c, ?h1, ?h2⟩
  case h2 =>
    unfold homologyEquivZero
    erw [LinearEquiv.trans_apply, Submodule.quotEquivOfEq_mk]
    erw [Iso.toLinearEquiv_apply, Iso.symm_hom, Iso.trans_inv, Iso.trans_inv]
    erw [ConcreteCategory.comp_apply, ConcreteCategory.comp_apply]
    erw [homologyMk_inv_mk, homologyMapIso_inv_π, homologyIsoSc'_inv_π]
  case h1 =>
    erw [cyclesIsoSc'_inv_i, cyclesMap_i_el, moduleCatCyclesIso_inv_i]

/-- **`lem:m6-homology-mk`**: `homologyEquiv` sends the class of a cocycle `z` to `homologyπ` of a cycle with underlying
cochain `z`, every degree. -/
theorem homologyEquiv_mk (h : Subsingleton (C.X (-1))) (n : ℕ) (z : C.Z (n : ℤ)) :
    ∃ c : C.toNat.cycles n, (C.toNat.iCycles n).hom c = (z : C.X (n : ℤ)) ∧
      C.homologyEquiv h n (Submodule.Quotient.mk z) = (C.toNat.homologyπ n).hom c := by
  cases n with
  | zero => exact C.homologyEquivZero_mk h z
  | succ n => exact C.homologyEquivSucc_mk n z

end GL2.Cone.CochainCx

end
