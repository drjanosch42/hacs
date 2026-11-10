/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.M6.ResTransfer

/-!
# M6-A1 (b) (slot 23): restriction and conjugation computed on any resolutions

[GT] `gt_part_M6.tex`, row `thm:m6-map-res`. Brief `M6B_brief.md` §2.1 (A1-b).

For an **injective** homomorphism `f : K → G`, a morphism `θ : res_f A → B` of `K`-representations, a projective resolution
`P` of `k` over `G`, a projective resolution `Q` of `k` over `K`, and **any** chain map `α : Q → res_f P` of
`K`-representations over `id_k`, mathlib's `groupCohomology.map f θ : H^n(G; A) → H^n(K; B)` is, through the canonical
comparisons `canIso`, induced by the cochain map `ψ ↦ θ ∘ res_f(ψ) ∘ α` (`map_canIso`). This covers restriction
(`f = K.subtype`, `θ = id`, `α = id`) and the conjugation `(c_g)_*` (`f = h ↦ g⁻¹ h g`, `θ = ρ(g)`, `α = τ`), at once.
It generalises LR-Y's `PermRes.barIso_res` (`θ = id`, permutation resolutions).

* `comm_of_el`: a chain map of resolutions is over a map of the resolved objects when it is so on elements in degree 0.
* `barRes f hf`: `res_f bar_G` as a projective resolution over `K` (the bar resolution is a permutation resolution with
  free action, `diagonalSuccIsoFree`), the target of the homotopy-uniqueness step.
-/

noncomputable section

namespace GL2.M6

open CategoryTheory groupCohomology GL2.MP

section Lift

variable {k : Type} [CommRing k] {K : Type} [Group K]

/-- **Over a map, on elements**: a chain map `φ : P → Q` of projective resolutions is over `f` when
`π_Q(φ_0 x) = f(π_P x)` for every `x` in degree `0`. -/
lemma comm_of_el {Y Z : Rep.{0} k K} {P : ProjectiveResolution Y} {Q : ProjectiveResolution Z} (f : Y ⟶ Z)
    (φ : P.complex ⟶ Q.complex) (h : ∀ x, (Q.π.f 0).hom ((φ.f 0).hom x) = f.hom ((P.π.f 0).hom x)) :
    φ ≫ Q.π = P.π ≫ (ChainComplex.single₀ _).map f := by
  apply HomologicalComplex.to_single_hom_ext
  rw [HomologicalComplex.comp_f, HomologicalComplex.comp_f, ChainComplex.single₀_map_f_zero]
  apply Rep.hom_ext; apply Representation.IntertwiningMap.ext; refine LinearMap.ext fun x => ?_
  exact h x

/-- The same, over the identity. -/
lemma comm_of_el' {Y : Rep.{0} k K} {P Q : ProjectiveResolution Y}
    (φ : P.complex ⟶ Q.complex) (h : ∀ x, (Q.π.f 0).hom ((φ.f 0).hom x) = (P.π.f 0).hom x) :
    φ ≫ Q.π = P.π := by
  have := comm_of_el (𝟙 Y) φ h
  rwa [CategoryTheory.Functor.map_id, Category.comp_id] at this

/-- The defining property of a chain map over the identity, on elements in degree `0`. -/
lemma el_of_comm {Y : Rep.{0} k K} {P Q : ProjectiveResolution Y} (φ : P.complex ⟶ Q.complex) (h : φ ≫ Q.π = P.π)
    (x : P.complex.X 0) : (Q.π.f 0).hom ((φ.f 0).hom x) = (P.π.f 0).hom x := by
  have := congrArg (fun F => (F.f 0).hom x) h
  simp only [HomologicalComplex.comp_f] at this
  exact this

end Lift

section BarRes

variable {G : Type} [Group G] (k : Type) [CommRing k]

/-- The diagonal action of `G` on `G^{n+1}` is free. -/
lemma diag_stab_bot (n : ℕ) (y : Fin (n + 1) → G) : MulAction.stabilizer G y = ⊥ := by
  ext g
  simp only [MulAction.mem_stabilizer_iff, Subgroup.mem_bot]
  constructor
  · intro h
    have h0 := congrFun h 0
    simp only [Pi.smul_apply, smul_eq_mul, mul_eq_right] at h0
    exact h0
  · rintro rfl; exact one_smul _ _

variable {K : Type} [Group K] (f : K →* G) (hf : Function.Injective f)

/-- **`res_f bar_G` as a projective resolution over `K`** (`f` injective). -/
def barRes : ProjectiveResolution ((Rep.resFunctor f).obj (Rep.trivial k G k)) :=
  GL2.Cone.ProjRed.resProjRes f hf (Rep.barResolution k G) (fun n => Fin (n + 1) → G)
    (fun n => (Rep.diagonalSuccIsoFree k G n).symm)
    (fun n y => by rw [diag_stab_bot n y]; infer_instance)
    (fun n y => by rw [diag_stab_bot n y, Subgroup.card_bot, Nat.cast_one]; exact isUnit_one)

lemma barRes_complex : (barRes k f hf).complex = resCx f (Rep.barComplex k G) := rfl

lemma barRes_π_zero (x : (Rep.barComplex k G).X 0) :
    ((barRes k f hf).π.f 0).hom x = ((Rep.barResolution k G).π.f 0).hom x := by
  simp [barRes, GL2.Cone.ProjRed.resProjRes]
  rfl

end BarRes

section Map

variable {G : Type} [Group G] {k : Type} [CommRing k] {K : Type} [Group K] (f : K →* G) (hf : Function.Injective f)

/-- `cochainsMap f θ = cochainsMap f id ≫ cochainsMap id θ`. -/
lemma cochainsMap_split {A : Rep.{0} k G} {B : Rep.{0} k K} (θ : Rep.res f A ⟶ B) :
    cochainsMap f θ = cochainsMap f (𝟙 (Rep.res f A)) ≫ cochainsMap (MonoidHom.id K) θ := rfl

/-- `bar_K → res_f bar_G` lies over the identity of `k`, on all of degree `0`. -/
lemma barMap_π (y : (Rep.barComplex k K).X 0) :
    ((Rep.barResolution k G).π.f 0).hom (((barMapCx (k := k) f).f 0).hom y) = ((Rep.barResolution k K).π.f 0).hom y := by
  have hu : (barMapF (k := k) f 0 ≫ (Rep.resFunctor f).map ((Rep.barResolution k G).π.f 0) :
      Rep.free k K (Fin 0 → K) ⟶ Rep.trivial k K k) =
      ((Rep.barResolution k K).π.f 0 : Rep.free k K (Fin 0 → K) ⟶ Rep.trivial k K k) := by
    apply Rep.free_ext; intro x
    exact barMapF_π (k := k) f x
  exact congrArg (fun F => F.hom y) hu

variable (P : ProjectiveResolution (Rep.trivial k G k)) (Q : ProjectiveResolution (Rep.trivial k K k))

include hf in
/-- **`thm:m6-map-res`: restriction / conjugation through any resolutions.** For an injective `f : K → G`,
`θ : res_f A ⟶ B` and any chain map `α : Q → res_f P` over `id_k`, `groupCohomology.map f θ` is induced, through the
canonical comparisons, by `ψ ↦ θ ∘ res_f(ψ) ∘ α`. -/
theorem map_canIso (α : Q.complex ⟶ resCx f P.complex)
    (hα : ∀ x, (P.π.f 0).hom ((α.f 0).hom x) = (Q.π.f 0).hom x)
    {A : Rep.{0} k G} {B : Rep.{0} k K} (θ : Rep.res f A ⟶ B) (n : ℕ) :
    groupCohomology.map f θ n ≫ (canIso Q B n).hom =
      (canIso P A n).hom ≫ HomologicalComplex.homologyMap
        (resY f P.complex A ≫ Cover.preY α (Rep.res f A) ≫ Cover.postY Q.complex θ) n := by
  set eP : P.complex ⟶ Rep.barComplex k G := (ProjectiveResolution.homotopyEquiv P (Rep.barResolution k G)).hom
  set eQ : Q.complex ⟶ Rep.barComplex k K := (ProjectiveResolution.homotopyEquiv Q (Rep.barResolution k K)).hom
  have hP : eP ≫ (Rep.barResolution k G).π = P.π := ProjectiveResolution.homotopyEquiv_hom_π _ _
  have hQ : eQ ≫ (Rep.barResolution k K).π = Q.π := ProjectiveResolution.homotopyEquiv_hom_π _ _
  -- the homotopy `α ≫ res eP ≃ eQ ≫ barMap` of maps `Q → res_f bar_G`
  let ι : Rep.trivial k K k ⟶ (Rep.resFunctor f).obj (Rep.trivial k G k) :=
    Rep.ofHom ⟨LinearMap.id, fun _ => rfl⟩
  have h1 : (α ≫ ((Rep.resFunctor f).mapHomologicalComplex _).map eP) ≫ (barRes k f hf).π =
      Q.π ≫ (ChainComplex.single₀ _).map ι := by
    refine comm_of_el ι (P := Q) (Q := barRes k f hf) _ fun x => ?_
    refine (barRes_π_zero k f hf _).trans ?_
    show ((Rep.barResolution k G).π.f 0).hom ((eP.f 0).hom ((α.f 0).hom x)) = (Q.π.f 0).hom x
    exact (el_of_comm (P := P) (Q := Rep.barResolution k G) eP hP ((α.f 0).hom x)).trans (hα x)
  have h2 : (eQ ≫ barMapCx f) ≫ (barRes k f hf).π = Q.π ≫ (ChainComplex.single₀ _).map ι := by
    refine comm_of_el ι (P := Q) (Q := barRes k f hf) _ fun x => ?_
    refine (barRes_π_zero k f hf _).trans ?_
    show ((Rep.barResolution k G).π.f 0).hom (((barMapCx f).f 0).hom ((eQ.f 0).hom x)) = (Q.π.f 0).hom x
    exact (barMap_π f _).trans (el_of_comm (P := Q) (Q := Rep.barResolution k K) eQ hQ x)
  have hH : Homotopy (α ≫ ((Rep.resFunctor f).mapHomologicalComplex _).map eP) (eQ ≫ barMapCx f) :=
    ProjectiveResolution.liftHomotopy ι _ _ h1 h2
  have hHY := (preYHomotopy hH (Rep.res f A)).homologyMap_eq n
  -- the cochain-level identities
  have key : cochainsMap f θ ≫ (inhomogeneousCochainsIso B).hom ≫ Cover.preY eQ B =
      (inhomogeneousCochainsIso A).hom ≫ resY f _ A ≫ Cover.preY (eQ ≫ barMapCx f) (Rep.res f A) ≫
        Cover.postY Q.complex θ := by
    have e1 := Cover.cochainsMap_inhomogeneousCochainsIso (G := K) (k := k) (A := Rep.res f A) θ
    have e2 := cochainsMap_barMap (k := k) f A
    have e3 := Cover.postY_preY eQ θ
    calc cochainsMap f θ ≫ (inhomogeneousCochainsIso B).hom ≫ Cover.preY eQ B
        = cochainsMap f (𝟙 (Rep.res f A)) ≫ (cochainsMap (A := Rep.res f A) (MonoidHom.id K) θ ≫
            (inhomogeneousCochainsIso B).hom) ≫ Cover.preY eQ B := by
          rw [cochainsMap_split]; simp only [Category.assoc]
      _ = cochainsMap f (𝟙 (Rep.res f A)) ≫ ((inhomogeneousCochainsIso (Rep.res f A)).hom ≫
            Cover.postY (Rep.barComplex k K) θ) ≫ Cover.preY eQ B := by rw [e1]
      _ = (cochainsMap f (𝟙 (Rep.res f A)) ≫ (inhomogeneousCochainsIso (Rep.res f A)).hom) ≫
            (Cover.postY (Rep.barComplex k K) θ ≫ Cover.preY eQ B) := by simp only [Category.assoc]
      _ = ((inhomogeneousCochainsIso A).hom ≫ resY f _ A ≫ Cover.preY (barMapCx f) (Rep.res f A)) ≫
            (Cover.preY eQ (Rep.res f A) ≫ Cover.postY Q.complex θ) := by rw [e2, e3]
      _ = _ := by rw [Cover.preY_comp]; simp only [Category.assoc]
  have key2 : (inhomogeneousCochainsIso A).hom ≫ Cover.preY eP A ≫ resY f P.complex A ≫ Cover.preY α (Rep.res f A) ≫
      Cover.postY Q.complex θ =
      (inhomogeneousCochainsIso A).hom ≫ resY f _ A ≫
        Cover.preY (α ≫ ((Rep.resFunctor f).mapHomologicalComplex _).map eP) (Rep.res f A) ≫
          Cover.postY Q.complex θ := by
    rw [← Category.assoc (Cover.preY eP A), preY_resY]
    simp only [Cover.preY_comp, Category.assoc]
  have L : groupCohomology.map f θ n ≫ (canIso Q B n).hom =
      HomologicalComplex.homologyMap (cochainsMap f θ ≫ (inhomogeneousCochainsIso B).hom ≫ Cover.preY eQ B) n := by
    rw [canIso_hom_eq Q B n eQ hQ, HomologicalComplex.homologyMap_comp, HomologicalComplex.homologyMap_comp]
    rfl
  have R : (canIso P A n).hom ≫ HomologicalComplex.homologyMap
        (resY f P.complex A ≫ Cover.preY α (Rep.res f A) ≫ Cover.postY Q.complex θ) n =
      HomologicalComplex.homologyMap ((inhomogeneousCochainsIso A).hom ≫ Cover.preY eP A ≫ resY f P.complex A ≫
        Cover.preY α (Rep.res f A) ≫ Cover.postY Q.complex θ) n := by
    rw [canIso_hom_eq P A n eP hP]
    simp only [HomologicalComplex.homologyMap_comp]
    try rfl
  rw [L, R, key, key2]
  simp only [HomologicalComplex.homologyMap_comp, hHY]
  rfl

end Map

end GL2.M6

end
