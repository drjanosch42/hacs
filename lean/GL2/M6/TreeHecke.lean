/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.M6.Classical

/-!
# M6 A-T (slot 23): Hecke operators on LR-Y's tree resolution, every degree, every ring

[GT] `gt_part_M6.tex`, rows `lem:m6-tree-tau`, `thm:m6-tree-hecke`. Brief `M6B_brief.md` §2 (A-T), §6 (the LH hand-off:
LH's degree `≥ 2` Hecke operators without cup products, master plan risk K3).

On the tree resolution `P = model7TreeRes k` (`k[Γ]² ← k[Γ]³ ← k[Γ]³ ← ⋯`, `Γ = SL₂(ℤ)`, a projective resolution over
**every** commutative ring, `−I` included):

* `tree_tau_exists`: a chain map `τ : res_{Γ''} P → res_{h ↦ g⁻¹hg} P` over `id_k` exists (`P` restricted along an
  injective homomorphism is again a free permutation resolution, LR-Y's `PermRes.resP`).
* **`tree_hecke`**: for **every** such `τ`, every `n`, every `A`, every injective `f : Γ'' → Γ` and `θ : res_f A ⟶ res A`,
  the Hecke operator `heckeY` on `Hom_Γ(P, A)` is the classical operator `cor ∘ (f, θ)_*` through LR-Y's comparison
  `PermRes.barIso` (which **is** `canIso`).
* **`tree_Tcls`**: in particular M5's `Tcls g n` on `H^n(SL₂(ℤ); V_L)`, every `n`, every ring `R` (no `6 ∈ R^×`).
-/

noncomputable section

namespace GL2.M6

open CategoryTheory groupCohomology GL2.MP groupCohomology.Cores

/-- LR-Y's `PermRes.barIso` is the canonical comparison of its resolution. -/
lemma barIso_eq_canIso {G : Type} [Group G] {k : Type} [CommRing k] (P : PermRes G k) (A : Rep.{0} k G) (n : ℕ) :
    P.barIso A n = canIso P.res A n := rfl

section Tree

open scoped MatrixGroups

variable {k : Type} [CommRing k]

/-- **`lem:m6-tree-tau`: a twisted chain datum exists on the tree resolution**, for every subgroup `K` and every
injective `f : K → SL₂(ℤ)`. -/
theorem tree_tau_exists (K : Subgroup SL(2, ℤ)) (f : K →* SL(2, ℤ)) (hf : Function.Injective f) :
    ∃ τ : resCx K.subtype (model7TreeRes k).complex ⟶ resCx f (model7TreeRes k).complex,
      ∀ x, ((model7TreeRes k).π.f 0).hom ((τ.f 0).hom x) = ((model7TreeRes k).π.f 0).hom x := by
  let P := model7PermRes k
  let L := ProjectiveResolution.lift (𝟙 _) (resPR K (model7TreeRes k)) (P.resP f hf).res
  have hL : L ≫ (P.resP f hf).res.π = (resPR K (model7TreeRes k)).π := by
    have := ProjectiveResolution.lift_commutes (𝟙 _) (resPR K (model7TreeRes k)) (P.resP f hf).res
    rwa [CategoryTheory.Functor.map_id, Category.comp_id] at this
  refine ⟨L ≫ P.resPι f hf, fun x => ?_⟩
  have h := el_of_comm L hL x
  refine Eq.trans ?_ h
  show (P.aug.f 0).hom _ = ((P.resP f hf).aug.f 0).hom _
  rw [PermRes.aug_f_zero, PermRes.aug_f_zero]
  rfl

variable (K : Subgroup SL(2, ℤ)) [K.FiniteIndex] (σ : Transversal K) (f : K →* SL(2, ℤ)) (hf : Function.Injective f)
  {A : Rep.{0} k SL(2, ℤ)} (θ : Rep.res f A ⟶ Rep.res K.subtype A)

include hf in
/-- **`thm:m6-tree-hecke` (A-T)**: on the tree resolution, for **every** `τ` over `id_k`, every `n` and every `A`,
the Hecke operator `heckeY` is the classical operator `cor ∘ (f, θ)_*`, through LR-Y's comparison `PermRes.barIso`. -/
theorem tree_hecke (τ : resCx K.subtype (model7TreeRes k).complex ⟶ resCx f (model7TreeRes k).complex)
    (hτ : ∀ x, ((model7TreeRes k).π.f 0).hom ((τ.f 0).hom x) = ((model7TreeRes k).π.f 0).hom x) (n : ℕ) :
    Tgen K σ f θ n ≫ ((model7PermRes k).barIso A n).hom =
      ((model7PermRes k).barIso A n).hom ≫
        HomologicalComplex.homologyMap (heckeY K σ f θ (model7TreeRes k) τ) n :=
  hecke_canIso K σ f hf θ (model7TreeRes k) τ hτ n

end Tree

section TreeTcls

open scoped MatrixGroups
open GL2.Model GL2.Model.XBulk GL2.Cone GL2.M4 GL2.M5

variable {R : Type} [CommRing R] {V : Type} [AddCommGroup V] [Module R V] (ρ : ΔQ →* Module.End R V)

/-- **`thm:m6-tree-hecke` for `Tcls`**: M5's `Tcls g n` on `H^n(SL₂(ℤ); V_L)`, every `n`, every ring `R` (no `6 ∈ R^×`),
is computed on the tree resolution by `heckeY` with any `τ` over `id_R`, through `PermRes.barIso`. -/
theorem tree_Tcls (g : GLQ) (hg : g ∈ ΔQ)
    (τ : resCx (Spp g).subtype (model7TreeRes R).complex ⟶ resCx (fConjS g) (model7TreeRes R).complex)
    (hτ : ∀ x, ((model7TreeRes R).π.f 0).hom ((τ.f 0).hom x) = ((model7TreeRes R).π.f 0).hom x) (n : ℕ) :
    ((model7PermRes R).barIso (VL ρ) n).hom.hom ∘ₗ Tcls ρ g hg n =
      (HomologicalComplex.homologyMap (heckeY (Spp g) (τS g) (fConjS g) (thetaS ρ g hg) (model7TreeRes R) τ) n).hom ∘ₗ
        ((model7PermRes R).barIso (VL ρ) n).hom.hom :=
  Tcls_canIso ρ g hg (model7TreeRes R) τ hτ n

end TreeTcls

end GL2.M6

end
