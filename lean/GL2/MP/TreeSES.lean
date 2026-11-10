/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.MP.TreeCyclic

/-!
# LR-Y phase 5 (slot 22): the tree sequence `eq:treeses` as one short exact sequence in `Rep k SL₂(ℤ)`

[GT] `gt_part_MP.tex`, row `thm:mp-treeses-rep`, backing [C] `cl:amalgam` (1), `eq:treeses` (`LRY3_audit.md` MINOR-1:
until now the sequence was proved exact as the augmented chain complex of the tree, `treeses_exact`, with its terms
identified one by one, `model7TreeRep1` / `model7TreeRep0`, but not assembled as one object of `Rep k Γ`).

* `TE = Γ ⧸ ⟨−I⟩`, `TV = Γ ⧸ ⟨S⟩ ⊔ Γ ⧸ ⟨U⟩`; `sesBdB`: `γ⟨−I⟩ ↦ γ⟨S⟩ − γ⟨U⟩` on bases.
* **`treeSES k`**: the short complex `k[Γ/⟨−I⟩] → k[Γ/⟨S⟩ ⊔ Γ/⟨U⟩] → k` in `Rep k Γ` (`sesF`: the boundary, `sesG`: the
  augmentation `∑ c_x x ↦ ∑ c_x`); **`treeSES_shortExact`**: it is short exact, over every commutative ring `k`.
* The proof transports `treeses_exact` along the orbit–stabiliser bijections `model7EdgeEquiv` (`γ⟨−I⟩ ↦ γ·[ρ, i]`) and
  `model7VtxEquiv` (`γ⟨S⟩ ↦ γ·i`, `γ⟨U⟩ ↦ γ·ρ`): `sesBdB_tree` (the boundary is the tree boundary `∂[ρ, i] = i − ρ`),
  `sesAug_tree` (the augmentation is the tree augmentation); the tree has no `2`-simplices, so `∂` is injective.
-/

noncomputable section

namespace GL2.MP

open CategoryTheory Finsupp Cover
open scoped MatrixGroups

section TreeSES

/-- The edges of the tree as a `Γ`-set: `Γ ⧸ ⟨−I⟩`. -/
abbrev TE : Type := SL(2, ℤ) ⧸ gI

/-- The vertices of the tree as a `Γ`-set: `Γ ⧸ ⟨S⟩ ⊔ Γ ⧸ ⟨U⟩`. -/
abbrev TV : Type := (SL(2, ℤ) ⧸ gS) ⊕ (SL(2, ℤ) ⧸ gU)

/-- `γ⟨−I⟩ ↦ γ⟨S⟩`. -/
abbrev qS : TE → SL(2, ℤ) ⧸ gS := Subgroup.quotientMapOfLE gI_le_gS
/-- `γ⟨−I⟩ ↦ γ⟨U⟩`. -/
abbrev qU : TE → SL(2, ℤ) ⧸ gU := Subgroup.quotientMapOfLE gI_le_gU

lemma qS_mk (g : SL(2, ℤ)) : qS (QuotientGroup.mk g) = QuotientGroup.mk g := rfl
lemma qU_mk (g : SL(2, ℤ)) : qU (QuotientGroup.mk g) = QuotientGroup.mk g := rfl

lemma qS_smul (g : SL(2, ℤ)) (e : TE) : qS (g • e) = g • qS e := by
  induction e using QuotientGroup.induction_on with
  | H h => rfl

lemma qU_smul (g : SL(2, ℤ)) (e : TE) : qU (g • e) = g • qU e := by
  induction e using QuotientGroup.induction_on with
  | H h => rfl

variable (k : Type) [CommRing k]

/-- **The tree boundary on bases**: `γ⟨−I⟩ ↦ γ⟨S⟩ − γ⟨U⟩` (the edge `γ·[ρ, i]` goes to `γ·i − γ·ρ`). -/
def sesBdB (e : TE) : TV →₀ k := single (Sum.inl (qS e)) 1 - single (Sum.inr (qU e)) 1

lemma sesBdB_smul (g : SL(2, ℤ)) (e : TE) : sesBdB k (g • e) = mapDomain (g • ·) (sesBdB k e) := by
  rw [sesBdB, sesBdB, mapDomain_sub, mapDomain_single, mapDomain_single, qS_smul, qU_smul]
  rfl

/-- The augmentation functional `∑ c_x x ↦ ∑ c_x` on `MonoidAlgebra k TV`. -/
def sesAugL : (TV →₀ k) →ₗ[k] k := Finsupp.linearCombination k (fun _ => (1 : k))

lemma sesAugL_mapDomain (g : SL(2, ℤ)) (x : TV →₀ k) : sesAugL k (mapDomain (g • ·) x) = sesAugL k x := by
  induction x using Finsupp.induction_linear with
  | zero => simp
  | add x y hx hy => rw [Finsupp.mapDomain_add, map_add, hx, hy, map_add]
  | single x c => simp [sesAugL, Finsupp.mapDomain_single]

/-- `k[Γ/⟨−I⟩]`. -/
abbrev sesE : Rep k SL(2, ℤ) := Rep.ofMulAction k SL(2, ℤ) TE
/-- `k[Γ/⟨S⟩ ⊔ Γ/⟨U⟩]` (`= k[Γ/⟨S⟩] ⊕ k[Γ/⟨U⟩]` up to `Finsupp.sumFinsuppLEquivProdFinsupp`). -/
abbrev sesV : Rep k SL(2, ℤ) := Rep.ofMulAction k SL(2, ℤ) TV

/-- **The boundary** `k[Γ/⟨−I⟩] → k[Γ/⟨S⟩ ⊔ Γ/⟨U⟩]`, `γ⟨−I⟩ ↦ γ⟨S⟩ − γ⟨U⟩`. -/
def sesF : sesE k ⟶ sesV k :=
  Rep.ofHom ⟨(MonoidAlgebra.coeffLinearEquiv k).symm.toLinearMap ∘ₗ Finsupp.linearCombination k (sesBdB k) ∘ₗ
      (MonoidAlgebra.coeffLinearEquiv k).toLinearMap, fun g => by
    refine LinearMap.ext fun x => ?_
    simp only [LinearMap.coe_comp, Function.comp_apply, LinearEquiv.coe_coe, Representation.ofMulAction_def,
      LinearEquiv.apply_symm_apply, Finsupp.lmapDomain_apply]
    rw [Cover.lc_mapDomain_smul k (sesBdB k) g (sesBdB_smul k g)]⟩

lemma sesF_hom_apply (x : MonoidAlgebra k TE) :
    (sesF k).hom x = (MonoidAlgebra.coeffLinearEquiv k).symm (Finsupp.linearCombination k (sesBdB k) (MonoidAlgebra.coeffLinearEquiv k x)) := rfl

/-- **The augmentation** `k[Γ/⟨S⟩ ⊔ Γ/⟨U⟩] → k`, every basis vector `↦ 1`. -/
def sesG : sesV k ⟶ Rep.trivial k SL(2, ℤ) k :=
  Rep.ofHom ⟨sesAugL k ∘ₗ (MonoidAlgebra.coeffLinearEquiv k).toLinearMap, fun g => by
    refine LinearMap.ext fun x => ?_
    simp only [LinearMap.coe_comp, Function.comp_apply, LinearEquiv.coe_coe, Representation.ofMulAction_def,
      LinearEquiv.apply_symm_apply, Finsupp.lmapDomain_apply, Representation.trivial_apply]
    rw [sesAugL_mapDomain]⟩

lemma sesG_hom_apply (x : MonoidAlgebra k TV) : (sesG k).hom x = sesAugL k (MonoidAlgebra.coeffLinearEquiv k x) := rfl

lemma sesAugL_sesBdB (x : TE →₀ k) : sesAugL k (Finsupp.linearCombination k (sesBdB k) x) = 0 := by
  induction x using Finsupp.induction_linear with
  | zero => simp
  | add x y hx hy => rw [map_add, map_add, hx, hy, add_zero]
  | single e c => simp [sesBdB, sesAugL]

lemma sesF_sesG : sesF k ≫ sesG k = 0 := by
  apply Rep.hom_ext; apply Representation.IntertwiningMap.ext; refine LinearMap.ext fun x => ?_
  show sesAugL k (MonoidAlgebra.coeffLinearEquiv k ((MonoidAlgebra.coeffLinearEquiv k).symm
    (Finsupp.linearCombination k (sesBdB k) (MonoidAlgebra.coeffLinearEquiv k x)))) = 0
  rw [LinearEquiv.apply_symm_apply, sesAugL_sesBdB]

/-- **The tree sequence `eq:treeses` in `Rep k Γ`**: `k[Γ/⟨−I⟩] → k[Γ/⟨S⟩ ⊔ Γ/⟨U⟩] → k`. -/
def treeSES : ShortComplex (Rep k SL(2, ℤ)) := ShortComplex.mk (sesF k) (sesG k) (sesF_sesG k)

/-! #### Comparison with the augmented chain complex of the tree -/

/-- The vertices, in the order `(⟨S⟩, ⟨U⟩)`: `γ⟨S⟩ ↦ γ·i`, `γ⟨U⟩ ↦ γ·ρ`. -/
def sesVtxEquiv : TV ≃ TS model7Hyp 3 0 := (Equiv.sumComm _ _).trans model7VtxEquiv

lemma sesVtxEquiv_inl (g : SL(2, ℤ)) : sesVtxEquiv (Sum.inl (QuotientGroup.mk g)) = g • model7IV := rfl
lemma sesVtxEquiv_inr (g : SL(2, ℤ)) : sesVtxEquiv (Sum.inr (QuotientGroup.mk g)) = g • model7RhoV := rfl

lemma model7EdgeEquiv_mk (g : SL(2, ℤ)) : model7EdgeEquiv (QuotientGroup.mk g) = g • model7E := rfl

/-- The tree boundary is equivariant. -/
lemma tBdS_smul (g : SL(2, ℤ)) (x : TS model7Hyp 3 1) :
    tBdS model7Hyp 3 k (g • x) = mapDomain (g • ·) (tBdS model7Hyp 3 k x) := by
  rw [tBdS, tBdS, altSum_comp]
  rfl

/-- **The boundary is the tree boundary** `∂(γ·[ρ, i]) = γ·i − γ·ρ`, on bases. -/
lemma sesBdB_tree (e : TE) : mapDomain sesVtxEquiv (sesBdB k e) = tBdS model7Hyp 3 k (model7EdgeEquiv e) := by
  induction e using QuotientGroup.induction_on with
  | H g =>
    rw [model7EdgeEquiv_mk, tBdS_smul, model7_tBdS_E, sesBdB, mapDomain_sub, mapDomain_sub, mapDomain_single,
      mapDomain_single, mapDomain_single, mapDomain_single, qS_mk, qU_mk, sesVtxEquiv_inl, sesVtxEquiv_inr]

lemma sesBd_tree (x : TE →₀ k) :
    mapDomain sesVtxEquiv (Finsupp.linearCombination k (sesBdB k) x) =
      Finsupp.linearCombination k (tBdS model7Hyp 3 k (s := 0)) (mapDomain model7EdgeEquiv x) := by
  induction x using Finsupp.induction_linear with
  | zero => simp
  | add x y hx hy => rw [map_add, Finsupp.mapDomain_add, hx, hy, Finsupp.mapDomain_add, map_add]
  | single e c =>
    rw [linearCombination_single, Finsupp.mapDomain_smul, sesBdB_tree, Finsupp.mapDomain_single,
      linearCombination_single]

lemma treeCx_d1 (z : (treeCx model7Hyp 3 k).toCx.X 2) :
    (treeCx model7Hyp 3 k).toCx.d 1 z = Finsupp.linearCombination k (tBdS model7Hyp 3 k (s := 0)) z := rfl

/-- **The augmentation is the tree augmentation.** -/
lemma sesAug_tree (y : TV →₀ k) :
    (treeCx model7Hyp 3 k).toCx.d 0 (mapDomain sesVtxEquiv y) = single () (sesAugL k y) := by
  show Finsupp.linearCombination k (fun _ : TS model7Hyp 3 0 => single () (1 : k)) (mapDomain sesVtxEquiv y) = _
  rw [lc_const_single, sesAugL, linearCombination_mapDomain]
  rfl

/-- The tree has no `2`-simplices, so its boundary `k[𝒯_1] → k[𝒯_0]` is injective. -/
lemma tree_d1_injective (z : (treeCx model7Hyp 3 k).toCx.X 2) (hz : (treeCx model7Hyp 3 k).toCx.d 1 z = 0) :
    z = 0 := by
  obtain ⟨w, hw⟩ := (treeses_exact k).2 1 z hz
  have hw0 : w = 0 := Finsupp.ext fun b => (TS.isEmpty model7TreeRetr.top 0 b).elim
  rw [← hw, hw0, map_zero]

lemma sesBd_injective : Function.Injective (Finsupp.linearCombination k (sesBdB k)) := by
  rw [← LinearMap.ker_eq_bot, LinearMap.ker_eq_bot']
  intro x hx
  have h := sesBd_tree k x
  rw [hx, Finsupp.mapDomain_zero] at h
  have h0 := tree_d1_injective k (mapDomain model7EdgeEquiv x) h.symm
  exact Finsupp.mapDomain_injective model7EdgeEquiv.injective (h0.trans Finsupp.mapDomain_zero.symm)

lemma sesBd_exact (y : TV →₀ k) (hy : sesAugL k y = 0) :
    ∃ x, Finsupp.linearCombination k (sesBdB k) x = y := by
  have h0 : (treeCx model7Hyp 3 k).toCx.d 0 (mapDomain sesVtxEquiv y) = 0 := by
    rw [sesAug_tree, hy, single_zero]; rfl
  obtain ⟨z, hz⟩ := (treeses_exact k).2 0 _ h0
  set z' : TS model7Hyp 3 1 →₀ k := z
  have hz' : Finsupp.linearCombination k (tBdS model7Hyp 3 k (s := 0)) z' = mapDomain sesVtxEquiv y := hz
  refine ⟨mapDomain model7EdgeEquiv.symm z', Finsupp.mapDomain_injective sesVtxEquiv.injective ?_⟩
  rw [sesBd_tree, ← Finsupp.mapDomain_comp, Equiv.self_comp_symm, Finsupp.mapDomain_id]
  exact hz'

/-- **`eq:treeses` is a short exact sequence of `k[Γ]`-modules**:
`0 → k[Γ/⟨−I⟩] → k[Γ/⟨S⟩ ⊔ Γ/⟨U⟩] → k → 0`, `γ⟨−I⟩ ↦ γ⟨S⟩ − γ⟨U⟩`, `γ⟨S⟩, γ⟨U⟩ ↦ 1`, over every commutative ring `k`. -/
theorem treeSES_shortExact : (treeSES k).ShortExact where
  exact := by
    apply (forget₂ _ (ModuleCat k)).reflects_exact_of_faithful
    rw [ShortComplex.moduleCat_exact_iff]
    intro x hx
    have hx' : sesAugL k (MonoidAlgebra.coeffLinearEquiv k (x : MonoidAlgebra k TV)) = 0 := hx
    obtain ⟨y, hy⟩ := sesBd_exact k _ hx'
    refine ⟨(MonoidAlgebra.coeffLinearEquiv k).symm y, ?_⟩
    change (sesF k).hom _ = x
    show (MonoidAlgebra.coeffLinearEquiv k).symm (Finsupp.linearCombination k (sesBdB k)
      (MonoidAlgebra.coeffLinearEquiv k ((MonoidAlgebra.coeffLinearEquiv k).symm y))) = x
    rw [LinearEquiv.apply_symm_apply, hy]
    exact (MonoidAlgebra.coeffLinearEquiv k).symm_apply_apply (x : MonoidAlgebra k TV)
  mono_f := by
    rw [Rep.mono_iff_injective]
    intro x y h
    have h' : (MonoidAlgebra.coeffLinearEquiv k).symm (Finsupp.linearCombination k (sesBdB k)
        (MonoidAlgebra.coeffLinearEquiv k (x : MonoidAlgebra k TE))) =
        (MonoidAlgebra.coeffLinearEquiv k).symm (Finsupp.linearCombination k (sesBdB k)
          (MonoidAlgebra.coeffLinearEquiv k (y : MonoidAlgebra k TE))) := h
    exact (MonoidAlgebra.coeffLinearEquiv k).injective
      (sesBd_injective k ((MonoidAlgebra.coeffLinearEquiv k).symm.injective h'))
  epi_g := by
    rw [Rep.epi_iff_surjective]
    intro r
    change k at r
    refine ⟨(MonoidAlgebra.coeffLinearEquiv k).symm (single (Sum.inl (QuotientGroup.mk 1)) r), ?_⟩
    show sesAugL k (MonoidAlgebra.coeffLinearEquiv k ((MonoidAlgebra.coeffLinearEquiv k).symm
      (single (Sum.inl (QuotientGroup.mk 1) : TV) r))) = r
    rw [LinearEquiv.apply_symm_apply, sesAugL, linearCombination_single, smul_eq_mul, mul_one]

/-- The boundary on basis vectors: `γ⟨−I⟩ ↦ γ⟨S⟩ − γ⟨U⟩`. -/
lemma sesF_single (g : SL(2, ℤ)) :
    (sesF k).hom (MonoidAlgebra.single (QuotientGroup.mk g : TE) 1) =
      MonoidAlgebra.single (Sum.inl (QuotientGroup.mk g) : TV) 1 - MonoidAlgebra.single (Sum.inr (QuotientGroup.mk g)) 1 := by
  rw [sesF_hom_apply, Cover.coeff_single, linearCombination_single, one_smul, sesBdB, map_sub]
  rfl

/-- The augmentation on basis vectors: `x ↦ 1`. -/
lemma sesG_single (x : TV) : (sesG k).hom (MonoidAlgebra.single x 1) = 1 := by
  rw [sesG_hom_apply, Cover.coeff_single]
  simp [sesAugL]

end TreeSES

end GL2.MP
