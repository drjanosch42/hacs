/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.MP.TreeCyclic

/-!
# LR-Y phase 5 (slot 22): the Mayer–Vietoris sequence of the amalgam in `groupCohomology` terms

[GT] `gt_part_MP.tex`, rows `thm:mp-mv-res`, `thm:mp-mv-les`, `thm:mp-mv-natural`, backing [C] `cl:amalgam` (2),
`eq:mvamalgam` (the map `H^q(Γ; A) → H^q(⟨S⟩; A) ⊕ H^q(⟨U⟩; A)` is restriction; the connecting map
`H^q(⟨−I⟩; A) → H^{q+1}(Γ; A)`; the sequence is exact and natural in `A`).  Every commutative ring `k`, every
`A : Rep k SL₂(ℤ)`, every `q`, `q = 0` included.

Phase 3 stated the maps into and out of `H^q(Γ; A)` through the `Ext`-built `model7TreeCochCoh`; here they go through the
bar-built `model7TreeCochCohNat` (`GL2.MP.TreeCyclic`, natural in `A`), so that restriction can be computed by
`PermRes.barIso_res`.

* **The vertex inclusions.**  For `c = 0` (`⟨S⟩`, `t = S`, `N = 4`) and `c = 1` (`⟨U⟩`, `t = U`, `N = 6`), the periodic
  resolution of `⟨t_c⟩` is the `c`-th vertex summand of the tree resolution restricted to `⟨t_c⟩`: on bases
  `(0, g) ↦ (c, g)` (`incA`); `incα` is a chain map `P^{⟨t_c⟩} → P|_{⟨t_c⟩}` lifting `id_k` (`incα_eps`).
* **Restriction.**  `resΓS`, `resΓU`: mathlib's `groupCohomology.map` along `⟨S⟩ ↪ Γ`, `⟨U⟩ ↪ Γ`;
  **`resΓS_tree`**: under `model7TreeCochCohNat` and `zS.cohIso` it is the projection `((x, y), z) ↦ x` of the tree
  cochains; **`mvResN_vIso`**: the map `H^q(Γ; A) → H^q(vCoch A)` of the sequence is `(res_S, res_U)` under `vIso`.
* **The connecting map.**  `mvConn q : H^q(⟨−I⟩; A) ⟶ H^{q+1}(Γ; A)`; **`mvConn_π`**: on a cocycle `z` of the periodic
  complex of `⟨−I⟩` it is the class of the tree cochain `((0, 0), z)`.
* **The long exact sequence** `0 → H⁰(Γ) → H⁰(S) ⊕ H⁰(U) → H⁰(−I) → H¹(Γ) → ⋯`, maps `(res_S, res_U)`,
  `res_S − res_U`, `mvConn`: **`mvLES_mono_zero`**, **`mvLES_exact_vtx`**, **`mvLES_exact_edge`**, **`mvLES_exact_coh`**.
* **Naturality in `A`**: **`resΓS_natural`**, `resΓU_natural`, **`mvMid_natural`**, **`mvConn_natural`**.
-/

noncomputable section

namespace GL2.MP

open CategoryTheory Finsupp Limits
open scoped MatrixGroups

/-! ### The vertex summands of the tree resolution, restricted to the vertex stabilisers -/

section VtxInc

variable {k : Type} [CommRing k] {K : Subgroup SL(2, ℤ)} (c : Fin 2)

/-- `(0, g) ↦ (c, g)`: the bases of `k[K]` into the `c`-th vertex summand `k[Γ] e_c`. -/
def vtxB (b : FB 1 K) : FB 2 SL(2, ℤ) := ⟨c, b.g.1⟩

/-- The same, in degree `n` of the tree resolution (`FB 2 Γ` in degree `0`, `FB 2 Γ ⊕ FB 1 Γ` above). -/
def vtxP : ∀ n, FB 1 K → model7PB n
  | 0 => vtxB c
  | _ + 1 => fun b => Sum.inl (vtxB c b)

variable (k) in
/-- The vertex inclusion on bases, with coefficient `1`. -/
def incA (n : ℕ) (b : FB 1 K) : model7PB n →₀ k := single (vtxP c n b) 1

lemma lc_single_one {α β : Type} (f : α → β) (x : α →₀ k) :
    Finsupp.linearCombination k (fun a => single (f a) (1 : k)) x = mapDomain f x := by
  induction x using Finsupp.induction_linear with
  | zero => simp
  | add x y hx hy => rw [map_add, hx, hy, Finsupp.mapDomain_add]
  | single a r => rw [linearCombination_single, Finsupp.mapDomain_single, Finsupp.smul_single, smul_eq_mul, mul_one]

variable (Z : CycDat K) (ht : (Z.t : SL(2, ℤ)) = model7VT c) (hN : Z.N = model7PerV.N c)

include ht hN in
/-- **The periodic boundary of `⟨t_c⟩` is the `c`-th vertex boundary of the tree resolution**, on bases. -/
lemma mapDomain_vtxB_bdS (n : ℕ) (b : FB 1 K) :
    mapDomain (vtxB c) (Z.perD.bdS k n b) = model7PerV.bdS k n (vtxB c b) := by
  by_cases h : Even n
  · rw [Z.perD.bdS_even k h, model7PerV.bdS_even k h, PerData.difB, PerData.difB, Finsupp.mapDomain_sub,
      Finsupp.mapDomain_single, Finsupp.mapDomain_single]
    show single (vtxB c b) (1 : k) - single (⟨c, (b.g * Z.t).1⟩ : FB 2 SL(2, ℤ)) (1 : k) =
      single (vtxB c b) (1 : k) - single (⟨c, b.g.1 * model7VT c⟩ : FB 2 SL(2, ℤ)) (1 : k)
    rw [Subgroup.coe_mul, ht]
  · rw [Z.perD.bdS_odd k h, model7PerV.bdS_odd k h, PerData.nrmB, PerData.nrmB, Finsupp.mapDomain_finsetSum]
    show ∑ i ∈ Finset.range Z.N, mapDomain (vtxB c) (single (⟨b.c, b.g * Z.t ^ i⟩ : FB 1 K) (1 : k)) =
      ∑ i ∈ Finset.range (model7PerV.N c), single (⟨c, b.g.1 * model7VT c ^ i⟩ : FB 2 SL(2, ℤ)) (1 : k)
    rw [hN]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [Finsupp.mapDomain_single]
    show single (⟨c, (b.g * Z.t ^ i).1⟩ : FB 2 SL(2, ℤ)) (1 : k) = _
    rw [Subgroup.coe_mul, SubgroupClass.coe_pow, ht]

variable (hK : Function.Injective K.subtype)

include ht hN in
lemma incA_comm (n : ℕ) (b : FB 1 K) :
    Finsupp.linearCombination k (((model7PermRes k).resP K.subtype hK).bd n) (incA k c (n + 1) b) =
      Finsupp.linearCombination k (incA k c n) (Z.perD.bdS k n b) := by
  show Finsupp.linearCombination k (model7PBd k n) (single (vtxP c (n + 1) b) (1 : k)) =
    Finsupp.linearCombination k (fun b => single (vtxP c n b) (1 : k)) (Z.perD.bdS k n b)
  rw [linearCombination_single, one_smul, lc_single_one]
  rcases n with _ | n
  · exact (mapDomain_vtxB_bdS c Z ht hN 0 b).symm
  · show Finsupp.mapDomain Sum.inl (model7PerV.bdS k (n + 1) (vtxB c b)) =
      mapDomain (fun b => Sum.inl (vtxB c b)) (Z.perD.bdS k (n + 1) b)
    rw [← mapDomain_vtxB_bdS c Z ht hN (n + 1) b, ← Finsupp.mapDomain_comp]
    rfl

lemma incA_smul (n : ℕ) (g : K) (b : FB 1 K) :
    incA k c n (g • b) = mapDomain (fun y : ((model7PermRes k).resP K.subtype hK).B n => g • y) (incA k c n b) := by
  show single (vtxP c n (g • b)) (1 : k) = mapDomain (fun y : model7PB n => (g : SL(2, ℤ)) • y) (single (vtxP c n b) 1)
  rw [Finsupp.mapDomain_single]
  rcases n with _ | n <;> rfl

variable (k) in
/-- **The vertex inclusion `P^{⟨t_c⟩} → P|_{⟨t_c⟩}`**, a chain map of resolutions of `k` over `k[⟨t_c⟩]`. -/
def incα : (Z.permRes k).cx k ⟶ ((model7PermRes k).resP K.subtype hK).cx k :=
  PermRes.mapB (Z.permRes k) ((model7PermRes k).resP K.subtype hK) (incA k c)
    (fun n g b => incA_smul c hK n g b) (fun n b => incA_comm c Z ht hN hK n b)

/-- **It lifts the identity of `k`.** -/
lemma incα_eps : (incα k c Z ht hN hK).f 0 ≫ ((model7PermRes k).resP K.subtype hK).eps = (Z.permRes k).eps :=
  PermRes.mapB_eps _ _ _ _ _ fun b => by
    unfold incA; exact (linearCombination_single k _ _).trans (one_smul k 1)

/-- On `Hom` complexes, `φ ↦ φ ∘ incα` followed by evaluation at the generator is evaluation at `e_c`. -/
lemma incα_eval (A : Rep.{0} k SL(2, ℤ)) (n : ℕ) (φ : (model7PermRes k).X k n ⟶ A) :
    Z.cycEv (Rep.res K.subtype A) n ((incα k c Z ht hN hK).f n ≫ ((model7PermRes k).resPι K.subtype hK).f n ≫
      (Rep.resFunctor K.subtype).map φ) = φ.hom (MonoidAlgebra.single (vtxP c n (⟨0, 1⟩ : FB 1 K)) 1) := by
  rw [CycDat.cycEv_apply]
  show φ.hom ((MonoidAlgebra.coeffLinearEquiv k).symm (Finsupp.linearCombination k (incA k c n)
    (MonoidAlgebra.coeffLinearEquiv k (MonoidAlgebra.single (⟨0, 1⟩ : FB 1 K) (1 : k))))) = _
  rw [Cover.coeff_single, linearCombination_single, one_smul, incA]
  rfl

end VtxInc

/-! ### Restriction `H^q(Γ; A) → H^q(⟨S⟩; A) ⊕ H^q(⟨U⟩; A)` -/

section Res

variable {k : Type} [CommRing k] (A : Rep.{0} k SL(2, ℤ))

lemma gS_inj : Function.Injective gS.subtype := Subtype.val_injective
lemma gU_inj : Function.Injective gU.subtype := Subtype.val_injective

/-- `⟨S⟩` is the vertex `c = 0`. -/
abbrev incαS : (zS.permRes k).cx k ⟶ ((model7PermRes k).resP gS.subtype gS_inj).cx k :=
  incα k 0 zS rfl rfl gS_inj
/-- `⟨U⟩` is the vertex `c = 1`. -/
abbrev incαU : (zU.permRes k).cx k ⟶ ((model7PermRes k).resP gU.subtype gU_inj).cx k :=
  incα k 1 zU rfl rfl gU_inj

/-- **Restriction** `H^q(Γ; A) → H^q(⟨S⟩; A)` (mathlib's `groupCohomology.map` along `⟨S⟩ ↪ Γ`). -/
def resΓS (q : ℕ) : groupCohomology A q ⟶ groupCohomology (resS A) q :=
  groupCohomology.map gS.subtype (𝟙 (resS A)) q

/-- **Restriction** `H^q(Γ; A) → H^q(⟨U⟩; A)`. -/
def resΓU (q : ℕ) : groupCohomology A q ⟶ groupCohomology (resU A) q :=
  groupCohomology.map gU.subtype (𝟙 (resU A)) q

/-- On cochains: `φ ↦ φ ∘ incα_S`, evaluated at the generator, is the projection `((x, y), z) ↦ x`. -/
lemma incS_chain :
    resY gS.subtype _ A ≫ Cover.preY ((model7PermRes k).resPι gS.subtype gS_inj) (resS A) ≫
        Cover.preY incαS (resS A) ≫ (zS.cycCochIso (resS A)).hom =
      (treeCochIso A).hom ≫ mvπ A ≫ pS A := by
  ext n φ : 3
  refine (incα_eval 0 zS rfl rfl gS_inj A n φ).trans ?_
  rcases n with _ | n <;> rfl

/-- On cochains: the projection `((x, y), z) ↦ y` for `⟨U⟩`. -/
lemma incU_chain :
    resY gU.subtype _ A ≫ Cover.preY ((model7PermRes k).resPι gU.subtype gU_inj) (resU A) ≫
        Cover.preY incαU (resU A) ≫ (zU.cycCochIso (resU A)).hom =
      (treeCochIso A).hom ≫ mvπ A ≫ pU A := by
  ext n φ : 3
  refine (incα_eval 1 zU rfl rfl gU_inj A n φ).trans ?_
  rcases n with _ | n <;> rfl

/-- **Restriction to `⟨S⟩` on the tree cochains**: under `model7TreeCochCohNat` and `zS.cohIso`, `res_S` is induced by
`((x, y), z) ↦ x` (`x ↦ x` in degree `0`). -/
theorem resΓS_tree (q : ℕ) :
    resΓS A q ≫ (zS.cohIso (resS A) q).hom =
      (model7TreeCochCohNat A q).hom ≫ HomologicalComplex.homologyMap (mvπ A ≫ pS A) q := by
  have h := (model7PermRes k).barIso_res gS.subtype gS_inj (zS.permRes k) incαS
    (incα_eps 0 zS rfl rfl gS_inj) A q
  show resΓS A q ≫ ((zS.permRes k).barIso (resS A) q).hom ≫
      HomologicalComplex.homologyMap (zS.cycCochIso (resS A)).hom q =
    (((model7PermRes k).barIso A q).hom ≫ HomologicalComplex.homologyMap (treeCochIso A).hom q) ≫ _
  rw [← Category.assoc, resΓS, h, Category.assoc, Category.assoc, ← HomologicalComplex.homologyMap_comp,
    ← HomologicalComplex.homologyMap_comp]
  simp only [Category.assoc]
  rw [incS_chain]

/-- **Restriction to `⟨U⟩` on the tree cochains**: induced by `((x, y), z) ↦ y`. -/
theorem resΓU_tree (q : ℕ) :
    resΓU A q ≫ (zU.cohIso (resU A) q).hom =
      (model7TreeCochCohNat A q).hom ≫ HomologicalComplex.homologyMap (mvπ A ≫ pU A) q := by
  have h := (model7PermRes k).barIso_res gU.subtype gU_inj (zU.permRes k) incαU
    (incα_eps 1 zU rfl rfl gU_inj) A q
  show resΓU A q ≫ ((zU.permRes k).barIso (resU A) q).hom ≫
      HomologicalComplex.homologyMap (zU.cycCochIso (resU A)).hom q =
    (((model7PermRes k).barIso A q).hom ≫ HomologicalComplex.homologyMap (treeCochIso A).hom q) ≫ _
  rw [← Category.assoc, resΓU, h, Category.assoc, Category.assoc, ← HomologicalComplex.homologyMap_comp,
    ← HomologicalComplex.homologyMap_comp]
  simp only [Category.assoc]
  rw [incU_chain]

/-- `H^q(Γ; A) → H^q(vCoch A)`: the projection to the vertex part, through the bar-built `model7TreeCochCohNat`. -/
def mvResN (q : ℕ) : groupCohomology A q ⟶ (vCoch A).homology q :=
  (model7TreeCochCohNat A q).hom ≫ HomologicalComplex.homologyMap (mvπ A) q

/-- **`eq:mvamalgam`, the first map**: under `vIso`, `H^q(Γ; A) → H^q(vCoch A)` is `(res_S, res_U)`. -/
theorem mvResN_vIso (q : ℕ) : mvResN A q ≫ (vIso A q).hom = biprod.lift (resΓS A q) (resΓU A q) := by
  apply biprod.hom_ext
  · rw [biprod.lift_fst, (Iso.eq_comp_inv (zS.cohIso (resS A) q)).2 (resΓS_tree A q)]
    simp [mvResN, vIso, vHomIso, HomologicalComplex.homologyMap_comp]
  · rw [biprod.lift_snd, (Iso.eq_comp_inv (zU.cohIso (resU A) q)).2 (resΓU_tree A q)]
    simp [mvResN, vIso, vHomIso, HomologicalComplex.homologyMap_comp]

end Res

/-! ### The long exact sequence in `groupCohomology` terms -/

section LES

variable {k : Type} [CommRing k] (A : Rep.{0} k SL(2, ℤ))

/-- The first map `(res_S, res_U) : H^q(Γ; A) → H^q(⟨S⟩; A) ⊕ H^q(⟨U⟩; A)`. -/
def mvFirst (q : ℕ) : groupCohomology A q ⟶ groupCohomology (resS A) q ⊞ groupCohomology (resU A) q :=
  biprod.lift (resΓS A q) (resΓU A q)

/-- The middle map `res_S − res_U : H^q(⟨S⟩; A) ⊕ H^q(⟨U⟩; A) → H^q(⟨−I⟩; A)`. -/
def mvMid (q : ℕ) : groupCohomology (resS A) q ⊞ groupCohomology (resU A) q ⟶ groupCohomology (resI A) q :=
  biprod.desc (resSI A q) (-resUI A q)

/-- **The connecting map** `H^q(⟨−I⟩; A) → H^{q+1}(Γ; A)` of `eq:mvamalgam`: the inclusion of the edge part of the tree
cochains, through `eIso` and the bar-built `model7TreeCochCohNat`. -/
def mvConn (q : ℕ) : groupCohomology (resI A) q ⟶ groupCohomology A (q + 1) :=
  (eIso A q).inv ≫ HomologicalComplex.homologyMap (mvι A) (q + 1) ≫ (model7TreeCochCohNat A (q + 1)).inv

lemma mvFirst_eq (q : ℕ) : mvFirst A q = mvResN A q ≫ (vIso A q).hom := (mvResN_vIso A q).symm

lemma mvMid_eq (q : ℕ) : mvMid A q = (vIso A q).inv ≫ mvδ A q ≫ (eIso A q).hom := (mvδ_resres A q).symm

lemma vIso_mvMid (q : ℕ) : (vIso A q).hom ≫ mvMid A q = mvδ A q ≫ (eIso A q).hom := by
  rw [mvMid_eq, Iso.hom_inv_id_assoc]

lemma inv_mvFirst (q : ℕ) :
    (model7TreeCochCohNat A q).inv ≫ mvFirst A q = HomologicalComplex.homologyMap (mvπ A) q ≫ (vIso A q).hom := by
  rw [mvFirst_eq, mvResN, Category.assoc, Iso.inv_hom_id_assoc]

lemma eIso_mvConn (q : ℕ) :
    (eIso A q).hom ≫ mvConn A q =
      HomologicalComplex.homologyMap (mvι A) (q + 1) ≫ (model7TreeCochCohNat A (q + 1)).inv := by
  rw [mvConn, Iso.hom_inv_id_assoc]

lemma mvFirst_mvMid (q : ℕ) : mvFirst A q ≫ mvMid A q = 0 := by
  have h : HomologicalComplex.homologyMap (mvπ A) q ≫ mvδ A q = 0 := (mvSC_shortExact A).comp_δ q (q + 1) rfl
  rw [mvFirst_eq, mvMid_eq, mvResN]
  simp only [Category.assoc, Iso.hom_inv_id_assoc]
  rw [reassoc_of% h, zero_comp, comp_zero]

lemma mvMid_mvConn (q : ℕ) : mvMid A q ≫ mvConn A q = 0 := by
  have h : mvδ A q ≫ HomologicalComplex.homologyMap (mvι A) (q + 1) = 0 := (mvSC_shortExact A).δ_comp q (q + 1) rfl
  rw [mvMid_eq, mvConn]
  simp only [Category.assoc, Iso.hom_inv_id_assoc]
  rw [reassoc_of% h, zero_comp, comp_zero]

lemma mvConn_mvFirst (q : ℕ) : mvConn A q ≫ mvFirst A (q + 1) = 0 := by
  rw [mvConn, mvFirst_eq, mvResN]
  simp only [Category.assoc, Iso.inv_hom_id_assoc]
  rw [← HomologicalComplex.homologyMap_comp_assoc, mvι_mvπ, HomologicalComplex.homologyMap_zero, zero_comp,
    comp_zero]

/-- **Exactness at `H^q(⟨S⟩; A) ⊕ H^q(⟨U⟩; A)`**: `H^q(Γ) →(res_S, res_U) H^q(S) ⊕ H^q(U) →(res_S − res_U) H^q(−I)`. -/
theorem mvLES_exact_vtx (q : ℕ) : (ShortComplex.mk _ _ (mvFirst_mvMid A q)).Exact :=
  ShortComplex.exact_of_iso
    (ShortComplex.isoMk (model7TreeCochCohNat A q).symm (vIso A q) (eIso A q)
      (by rw [Iso.symm_hom]; exact inv_mvFirst A q) (vIso_mvMid A q))
    ((mvSC_shortExact A).homology_exact₃ q (q + 1) rfl)

/-- **Exactness at `H^q(⟨−I⟩; A)`**: `H^q(S) ⊕ H^q(U) →(res_S − res_U) H^q(−I) →(∂) H^{q+1}(Γ)`. -/
theorem mvLES_exact_edge (q : ℕ) : (ShortComplex.mk _ _ (mvMid_mvConn A q)).Exact :=
  ShortComplex.exact_of_iso
    (ShortComplex.isoMk (vIso A q) (eIso A q) (model7TreeCochCohNat A (q + 1)).symm
      (vIso_mvMid A q) (by rw [Iso.symm_hom]; exact eIso_mvConn A q))
    ((mvSC_shortExact A).homology_exact₁ q (q + 1) rfl)

/-- **Exactness at `H^{q+1}(Γ; A)`**: `H^q(−I) →(∂) H^{q+1}(Γ) →(res_S, res_U) H^{q+1}(S) ⊕ H^{q+1}(U)`. -/
theorem mvLES_exact_coh (q : ℕ) : (ShortComplex.mk _ _ (mvConn_mvFirst A q)).Exact :=
  ShortComplex.exact_of_iso
    (ShortComplex.isoMk (eIso A q) (model7TreeCochCohNat A (q + 1)).symm (vIso A (q + 1))
      (by rw [Iso.symm_hom]; exact eIso_mvConn A q) (by rw [Iso.symm_hom]; exact inv_mvFirst A (q + 1)))
    ((mvSC_shortExact A).homology_exact₂ (q + 1))

lemma eSh_homology_zero_isZero : IsZero ((eSh A).homology 0) :=
  ShortComplex.isZero_homology_of_isZero_X₂ _
    (ModuleCat.isZero_of_subsingleton (ModuleCat.of k PUnit) : IsZero (ModuleCat.of k PUnit))

/-- **The beginning: `(res_S, res_U) : H⁰(Γ; A) → H⁰(⟨S⟩; A) ⊕ H⁰(⟨U⟩; A)` is a monomorphism.** -/
theorem mvLES_mono_zero : Mono (mvFirst A 0) := by
  have hex := (mvSC_shortExact A).homology_exact₂ 0
  have hm : Mono (HomologicalComplex.homologyMap (mvπ A) 0) :=
    hex.mono_g ((eSh_homology_zero_isZero A).eq_of_src _ _)
  rw [mvFirst_eq, mvResN]
  infer_instance

/-- **The connecting map on cocycles**: for a cocycle `z` of the periodic complex of `⟨−I⟩` in degree `q`, `∂[z]` is the
class of the tree cochain `((0, 0), z)` (`(mvι A).f (q + 1)`) in degree `q + 1`. -/
theorem mvConn_π (q : ℕ) {W : ModuleCat.{0} k} (z : W ⟶ (cI A).X q) (hz : z ≫ (cI A).d q (q + 1) = 0)
    (hz' : (z ≫ (mvι A).f (q + 1)) ≫ (treeCoch A).d (q + 1) (q + 2) = 0) :
    (cI A).liftCycles z (q + 1) ((ComplexShape.up ℕ).next_eq' rfl) hz ≫ (cI A).homologyπ q ≫
        (zI.cohIso (resI A) q).inv ≫ mvConn A q =
      (treeCoch A).liftCycles (z ≫ (mvι A).f (q + 1)) (q + 2) ((ComplexShape.up ℕ).next_eq' rfl) hz' ≫
        (treeCoch A).homologyπ (q + 1) ≫ (model7TreeCochCohNat A (q + 1)).inv := by
  have hze : z ≫ (eSh A).d (q + 1) (q + 2) = 0 := by
    apply ModuleCat.hom_ext; refine LinearMap.ext fun w => ?_
    have h1 : zI.cycD (resI A) q (z.hom w) = 0 := by
      have := congrArg (fun f => f.hom w) hz
      rw [CycDat.cycCoch_d] at this
      exact this
    have h2 : edgA A q (z.hom w) = 0 := (edgA_eq A q _).trans h1
    rw [eSh_d]
    change -(edgA A q (z.hom w)) = (0 : A)
    rw [h2, neg_zero]
  have hI : (zI.cohIso (resI A) q).inv ≫ mvConn A q =
      (eHomIso A q).hom ≫ HomologicalComplex.homologyMap (mvι A) (q + 1) ≫ (model7TreeCochCohNat A (q + 1)).inv := by
    simp [mvConn, eIso]
  rw [hI, reassoc_of% (eHomIso_π A q z hz hze), HomologicalComplex.homologyπ_naturality_assoc]
  erw [HomologicalComplex.liftCycles_comp_cyclesMap_assoc]
  rfl

end LES

/-! ### Naturality in `A` -/

section Natural

/-- **Restriction is natural in the coefficients**: `res_f ∘ H^q(φ) = H^q(res φ) ∘ res_f`, for any group homomorphism `f`
(mathlib's `groupCohomology.map_comp`). -/
lemma res_natural {G G' : Type} [Group G] [Group G'] {k : Type} [CommRing k] (f : G' →* G) {A B : Rep.{0} k G}
    (φ : A ⟶ B) (q : ℕ) :
    groupCohomology.map (MonoidHom.id G) φ q ≫ groupCohomology.map f (𝟙 (Rep.res f B)) q =
      groupCohomology.map f (𝟙 (Rep.res f A)) q ≫
        groupCohomology.map (MonoidHom.id G') ((Rep.resFunctor f).map φ) q := by
  rw [← groupCohomology.map_comp, ← groupCohomology.map_comp]
  rfl

variable {k : Type} [CommRing k]

namespace CycDat

variable {C : Type} [Group C] (Z : CycDat C)

lemma cycD_nat {A B : Rep.{0} k C} (g : A ⟶ B) (n : ℕ) (x : A) : Z.cycD B n (g.hom x) = g.hom (Z.cycD A n x) := by
  unfold cycD
  split_ifs
  · simp only [LinearMap.sub_apply, LinearMap.id_apply, map_sub, Rep.hom_comm_apply]
  · simp only [LinearMap.sum_apply, map_sum, Rep.hom_comm_apply]

/-- `g : A ⟶ B` on the periodic cochain complexes (`g` in every degree). -/
def cycCochMap {A B : Rep.{0} k C} (g : A ⟶ B) : Z.cycCoch A ⟶ Z.cycCoch B :=
  CochainComplex.ofHom (X := Z.cycCoch A) (Y := Z.cycCoch B) (fun _ => ModuleCat.ofHom g.hom.toLinearMap)
    (fun i => by
      rw [cycCoch_d, cycCoch_d]
      apply ModuleCat.hom_ext; refine LinearMap.ext fun x => ?_
      exact Z.cycD_nat g i x)

lemma postY_cycCochIso {A B : Rep.{0} k C} (g : A ⟶ B) :
    Cover.postY ((Z.permRes k).cx k) g ≫ (Z.cycCochIso B).hom = (Z.cycCochIso A).hom ≫ Z.cycCochMap g := by
  ext n φ : 3
  rfl

end CycDat

/-- **The fixed comparison with the bar resolution is natural in `A`**, for any `PermRes`. -/
theorem PermRes.barIso_natural {G : Type} [Group G] (P : PermRes G k) {A B : Rep.{0} k G} (g : A ⟶ B) (q : ℕ) :
    groupCohomology.map (MonoidHom.id G) g q ≫ (P.barIso B q).hom =
      (P.barIso A q).hom ≫ HomologicalComplex.homologyMap (Cover.postY (P.cx k) g) q := by
  have key : groupCohomology.cochainsMap (MonoidHom.id _) g ≫ (groupCohomology.inhomogeneousCochainsIso B).hom ≫
      Cover.preY (P.barEquiv k).hom B =
    (groupCohomology.inhomogeneousCochainsIso A).hom ≫ Cover.preY (P.barEquiv k).hom A ≫ Cover.postY (P.cx k) g := by
    rw [← Category.assoc, Cover.cochainsMap_inhomogeneousCochainsIso, Category.assoc, Cover.postY_preY]
  have := congrArg (fun φ => HomologicalComplex.homologyMap φ q) key
  simp only [HomologicalComplex.homologyMap_comp] at this
  show HomologicalComplex.homologyMap (groupCohomology.cochainsMap (MonoidHom.id _) g) q ≫
      (HomologicalComplex.homologyMap (groupCohomology.inhomogeneousCochainsIso B).hom q ≫
        HomologicalComplex.homologyMap (Cover.preY (P.barEquiv k).hom B) q) =
    (HomologicalComplex.homologyMap (groupCohomology.inhomogeneousCochainsIso A).hom q ≫
        HomologicalComplex.homologyMap (Cover.preY (P.barEquiv k).hom A) q) ≫ _
  simp only [Category.assoc]
  exact this

/-- **`CycDat.cohIso` is natural in `A`.** -/
theorem CycDat.cohIso_natural {C : Type} [Group C] (Z : CycDat C) {A B : Rep.{0} k C} (g : A ⟶ B) (q : ℕ) :
    groupCohomology.map (MonoidHom.id C) g q ≫ (Z.cohIso B q).hom =
      (Z.cohIso A q).hom ≫ HomologicalComplex.homologyMap (Z.cycCochMap g) q := by
  show groupCohomology.map (MonoidHom.id C) g q ≫ ((Z.permRes k).barIso B q).hom ≫
      HomologicalComplex.homologyMap (Z.cycCochIso B).hom q =
    (((Z.permRes k).barIso A q).hom ≫ HomologicalComplex.homologyMap (Z.cycCochIso A).hom q) ≫ _
  rw [← Category.assoc, PermRes.barIso_natural, Category.assoc, Category.assoc,
    ← HomologicalComplex.homologyMap_comp, ← HomologicalComplex.homologyMap_comp, Z.postY_cycCochIso]

variable {A B : Rep.{0} k SL(2, ℤ)} (f : A ⟶ B)

/-- `f` on the shifted edge complexes (`0` in degree `0`, `f` above). -/
def eShMapF : ∀ n, (eSh A).X n ⟶ (eSh B).X n
  | 0 => 0
  | _ + 1 => ModuleCat.ofHom f.hom.toLinearMap

/-- `f : A ⟶ B` on the shifted edge complexes. -/
def eShMap : eSh A ⟶ eSh B :=
  CochainComplex.ofHom (X := eSh A) (Y := eSh B) (eShMapF f) (fun i => by
    rw [eSh_d, eSh_d]
    rcases i with _ | m
    · show (0 : eX A 0 ⟶ eX B 0) ≫ (0 : eX B 0 ⟶ eX B 1) =
        (0 : eX A 0 ⟶ eX A 1) ≫ ModuleCat.ofHom f.hom.toLinearMap
      exact Limits.zero_comp.trans Limits.zero_comp.symm
    · apply ModuleCat.hom_ext; refine LinearMap.ext fun (x : A) => ?_
      show (-edgA B m) (f.hom x) = f.hom ((-edgA A m) x)
      rw [LinearMap.neg_apply, LinearMap.neg_apply, map_neg, edgA_nat])

/-- **`ι` is natural**: `mvι A ≫ treeCochMap f = eShMap f ≫ mvι B`. -/
lemma mvι_natural : mvι A ≫ treeCochMap f = eShMap f ≫ mvι B := by
  ext n x : 3
  rcases n with _ | m
  · show ((treeCochMap f).f 0).hom ((mvιF A 0).hom x) = (mvιF B 0).hom ((eShMapF f 0).hom x)
    rw [show (mvιF A 0).hom x = 0 from rfl, show (mvιF B 0).hom ((eShMapF f 0).hom x) = 0 from rfl]
    exact map_zero ((treeCochMap f).f 0).hom
  · show (((f.hom 0, f.hom 0), f.hom x) : (B × B) × B) = ((0, 0), f.hom x)
    rw [map_zero]

/-- A cocycle of the periodic complex of `⟨−I⟩` is a cocycle of the shifted edge complex. -/
lemma cI_cocycle_eSh (q : ℕ) {W : ModuleCat.{0} k} (z : W ⟶ (cI A).X q) (hz : z ≫ (cI A).d q (q + 1) = 0) :
    z ≫ (eSh A).d (q + 1) (q + 2) = 0 := by
  apply ModuleCat.hom_ext; refine LinearMap.ext fun w => ?_
  have h1 : zI.cycD (resI A) q (z.hom w) = 0 := by
    have := congrArg (fun f => f.hom w) hz
    rw [CycDat.cycCoch_d] at this
    exact this
  have h2 : edgA A q (z.hom w) = 0 := (edgA_eq A q _).trans h1
  rw [eSh_d]
  change -(edgA A q (z.hom w)) = (0 : A)
  rw [h2, neg_zero]

/-- **`eHomIso` is natural.** -/
lemma eHomIso_natural (q : ℕ) :
    HomologicalComplex.homologyMap (zI.cycCochMap ((Rep.resFunctor gI.subtype).map f)) q ≫ (eHomIso B q).hom =
      (eHomIso A q).hom ≫ HomologicalComplex.homologyMap (eShMap f) (q + 1) := by
  apply homology_ext (cI A) q
  intro W x hx
  have hxB : (x ≫ (zI.cycCochMap ((Rep.resFunctor gI.subtype).map f)).f q) ≫ (cI B).d q (q + 1) = 0 := by
    rw [Category.assoc, HomologicalComplex.Hom.comm, ← Category.assoc, hx, zero_comp]
  rw [HomologicalComplex.homologyπ_naturality_assoc, HomologicalComplex.liftCycles_comp_cyclesMap_assoc,
    eHomIso_π B q _ hxB (cI_cocycle_eSh q _ hxB), reassoc_of% (eHomIso_π A q x hx (cI_cocycle_eSh q x hx)),
    HomologicalComplex.homologyπ_naturality]
  erw [HomologicalComplex.liftCycles_comp_cyclesMap_assoc]
  rfl

/-- **`eIso` is natural.** -/
lemma eIso_inv_natural (q : ℕ) :
    groupCohomology.map (MonoidHom.id gI) ((Rep.resFunctor gI.subtype).map f) q ≫ (eIso B q).inv =
      (eIso A q).inv ≫ HomologicalComplex.homologyMap (eShMap f) (q + 1) := by
  simp only [eIso, Iso.trans_inv, Iso.symm_inv]
  rw [← Category.assoc, zI.cohIso_natural, Category.assoc, eHomIso_natural, Category.assoc]

lemma model7TreeCochCohNat_inv_natural (n : ℕ) :
    (model7TreeCochCohNat A n).inv ≫ groupCohomology.map (MonoidHom.id _) f n =
      HomologicalComplex.homologyMap (treeCochMap f) n ≫ (model7TreeCochCohNat B n).inv := by
  have h' : groupCohomology.map (MonoidHom.id _) f n ≫ (model7TreeCochCohNat B n).hom =
      (model7TreeCochCohNat A n).hom ≫ HomologicalComplex.homologyMap (treeCochMap f) n :=
    model7TreeCochCohNat_natural f n
  rw [Iso.inv_comp_eq, ← reassoc_of% h', Iso.hom_inv_id, Category.comp_id]

/-- **The connecting map is natural in `A`**: `∂_B ∘ H^q(⟨−I⟩; f) = H^{q+1}(Γ; f) ∘ ∂_A`. -/
theorem mvConn_natural (q : ℕ) :
    mvConn A q ≫ groupCohomology.map (MonoidHom.id _) f (q + 1) =
      groupCohomology.map (MonoidHom.id gI) ((Rep.resFunctor gI.subtype).map f) q ≫ mvConn B q := by
  simp only [mvConn, Category.assoc]
  rw [model7TreeCochCohNat_inv_natural, ← HomologicalComplex.homologyMap_comp_assoc, mvι_natural,
    HomologicalComplex.homologyMap_comp_assoc, ← reassoc_of% (eIso_inv_natural f q)]

/-- **Restriction to `⟨S⟩` is natural in `A`.** -/
theorem resΓS_natural (q : ℕ) :
    groupCohomology.map (MonoidHom.id _) f q ≫ resΓS B q =
      resΓS A q ≫ groupCohomology.map (MonoidHom.id gS) ((Rep.resFunctor gS.subtype).map f) q :=
  res_natural gS.subtype f q

/-- **Restriction to `⟨U⟩` is natural in `A`.** -/
theorem resΓU_natural (q : ℕ) :
    groupCohomology.map (MonoidHom.id _) f q ≫ resΓU B q =
      resΓU A q ≫ groupCohomology.map (MonoidHom.id gU) ((Rep.resFunctor gU.subtype).map f) q :=
  res_natural gU.subtype f q

/-- **The first map `(res_S, res_U)` is natural in `A`.** -/
theorem mvFirst_natural (q : ℕ) :
    groupCohomology.map (MonoidHom.id _) f q ≫ mvFirst B q =
      mvFirst A q ≫ biprod.map (groupCohomology.map (MonoidHom.id gS) ((Rep.resFunctor gS.subtype).map f) q)
        (groupCohomology.map (MonoidHom.id gU) ((Rep.resFunctor gU.subtype).map f) q) := by
  apply biprod.hom_ext
  · simp only [mvFirst, Category.assoc, biprod.lift_fst, biprod.map_fst, biprod.lift_fst_assoc]
    exact resΓS_natural f q
  · simp only [mvFirst, Category.assoc, biprod.lift_snd, biprod.map_snd, biprod.lift_snd_assoc]
    exact resΓU_natural f q

/-- **The middle map `res_S − res_U` is natural in `A`.** -/
theorem mvMid_natural (q : ℕ) :
    biprod.map (groupCohomology.map (MonoidHom.id gS) ((Rep.resFunctor gS.subtype).map f) q)
        (groupCohomology.map (MonoidHom.id gU) ((Rep.resFunctor gU.subtype).map f) q) ≫ mvMid B q =
      mvMid A q ≫ groupCohomology.map (MonoidHom.id gI) ((Rep.resFunctor gI.subtype).map f) q := by
  apply biprod.hom_ext'
  · simp only [mvMid, biprod.inl_map_assoc, biprod.inl_desc, biprod.inl_desc_assoc]
    exact res_natural fS ((Rep.resFunctor gS.subtype).map f) q
  · simp only [mvMid, biprod.inr_map_assoc, biprod.inr_desc, biprod.inr_desc_assoc, Preadditive.comp_neg,
      Preadditive.neg_comp]
    exact congrArg Neg.neg (res_natural fU ((Rep.resFunctor gU.subtype).map f) q)

end Natural

end GL2.MP
