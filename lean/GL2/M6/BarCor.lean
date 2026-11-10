/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.M6.ResMap

/-!
# M6-A1 (c) (slot 23): the corestriction `corH` is the transfer on any resolution

[GT] `gt_part_M6.tex`, rows `def:m6-lam`, `lem:m6-corfin-cochain`, `thm:m6-cor-res`. Brief `M6B_brief.md` §2.1 (A1-c,
"the crux"), risk R1.

For `K ≤ G` of finite index with the vendored (left) transversal `σ` and its retraction `λ = σ.lam : G → K`
(`x ∈ λ(x) σ(x⁻¹ K)⁻¹`, so `λ(h x) = h λ(x)` for `h ∈ K`, `lam_mul_left`):

* `lamCx`: the `K`-chain map `res_K C_•(G) → C_•(K)` of **homogeneous** standard resolutions, `(y_0, …, y_n) ↦ (λ y_0,
  …, λ y_n)`; it commutes with the faces (they forget coordinates) and with the augmentation.
* `lamBar`: the same on mathlib's bar resolutions, through `isoStandardComplex`; on generators
  `[x] ↦ λ(1) [λ(x_{≤i-1})⁻¹ λ(x_{≤i})]_i` (`lamBar_single`).
* **`corCx_inhomog` (cochain level, an equality, no homotopy)**: the vendored `corFin` is the transfer `corY` on the bar
  resolution after `λ^*`: `corFin u = corY(λ^* u)` under mathlib's `inhomogeneousCochainsIso`. This is the
  identification the brief asks for in A1-c, at cochain level (F6 check 2).
* **`corH_canIso` (`thm:m6-cor-res`)**: for **every** projective resolution `P` of `k` over `G`, `corH` is induced through
  the canonical comparisons by `corY` on `Hom(P, -)`, with `res_K P` the restricted resolution (`resPR`).
-/

noncomputable section

namespace GL2.M6

open CategoryTheory groupCohomology GL2.MP groupCohomology.Cores

/-- The restriction of a projective resolution to a subgroup (mathlib's `Rep.resFunctor`, which preserves
projectives along subgroup inclusions: Shapiro's lemma). -/
abbrev resPR {G : Type} [Group G] {k : Type} [CommRing k] (K : Subgroup G) (P : ProjectiveResolution (Rep.trivial k G k)) :
    ProjectiveResolution (Rep.trivial k K k) :=
  (Rep.resFunctor K.subtype).mapProjectiveResolution P

section Lam

variable {G : Type} [Group G] (K : Subgroup G) (σ : Transversal K)

/-- **`λ(h x) = h λ(x)`** for `h ∈ K`. -/
lemma lam_mul_left (h : K) (x : G) : σ.lam ((h : G) * x) = h * σ.lam x := by
  apply Subtype.ext
  have hq : Transversal.qinv (H := K) ((h : G) * x) = Transversal.qinv (H := K) x := by
    unfold Transversal.qinv
    rw [QuotientGroup.eq]
    convert h.2 using 1
    group
  rw [Subgroup.coe_mul, Transversal.coe_lam, Transversal.coe_lam, hq, mul_assoc]

/-- `λ` on tuples. -/
def lamV (n : ℕ) (y : Fin n → G) : Fin n → K := fun i => σ.lam (y i)

lemma lamV_smul (n : ℕ) (h : K) (y : Fin n → G) : lamV K σ n ((h : G) • y) = h • lamV K σ n y := by
  funext i
  exact lam_mul_left K σ h (y i)

end Lam

section Std

variable {G : Type} [Group G] (K : Subgroup G) (σ : Transversal K) (k : Type) [CommRing k]

/-- `MonoidAlgebra.mapDomain` as a linear map. -/
def mdL {M N : Type} (f : M → N) : MonoidAlgebra k M →ₗ[k] MonoidAlgebra k N where
  toFun := MonoidAlgebra.mapDomain f
  map_add' := MonoidAlgebra.mapDomain_add f
  map_smul' c x := MonoidAlgebra.mapDomain_smul f c x

lemma mdL_single {M N : Type} (f : M → N) (y : M) (r : k) :
    mdL k f (MonoidAlgebra.single y r) = MonoidAlgebra.single (f y) r :=
  MonoidAlgebra.mapDomain_single

/-- `λ` on `k[G^{n+1}]`, as a map of `K`-representations. -/
def lamStd (n : ℕ) :
    Rep.res K.subtype (Rep.ofMulAction k G (Fin (n + 1) → G)) ⟶ Rep.ofMulAction k K (Fin (n + 1) → K) :=
  Rep.ofHom (LinearMap.intertwiningMap_of_isIntertwiningMap _ _ (mdL k (lamV K σ (n + 1))) fun h x => by
    induction x using MonoidAlgebra.induction_linear with
    | zero => simp only [map_zero]
    | add a b ha hb => simp only [map_add, ha, hb]
    | single y r =>
      show mdL k (lamV K σ (n + 1)) (Representation.ofMulAction k G (Fin (n + 1) → G) (h : G) (MonoidAlgebra.single y r)) =
        Representation.ofMulAction k K (Fin (n + 1) → K) h (mdL k (lamV K σ (n + 1)) (MonoidAlgebra.single y r))
      rw [Representation.ofMulAction_single, mdL_single, mdL_single, Representation.ofMulAction_single, lamV_smul])

lemma lamStd_single (n : ℕ) (y : Fin (n + 1) → G) (r : k) :
    (lamStd K σ k n).hom (MonoidAlgebra.single y r) = MonoidAlgebra.single (lamV K σ (n + 1) y) r :=
  mdL_single k _ y r

/-- **`λ` on the standard resolutions**: a chain map `res_K C_•(G) ⟶ C_•(K)` of `K`-representations. -/
def lamCx : resCx K.subtype (Rep.standardComplex k G) ⟶ Rep.standardComplex k K where
  f n := lamStd K σ k n
  comm' i j (hij : j + 1 = i) := by
    subst hij
    apply Rep.hom_ext; apply Representation.IntertwiningMap.ext
    refine MonoidAlgebra.lhom_ext' fun (y : Fin (j + 1 + 1) → G) => LinearMap.ext_ring ?_
    show ((Rep.standardComplex k K).d (j + 1) j).hom ((lamStd K σ k (j + 1)).hom (MonoidAlgebra.single y 1)) =
      (lamStd K σ k j).hom (((Rep.standardComplex k G).d (j + 1) j).hom (MonoidAlgebra.single y 1))
    erw [Rep.standardComplex.d_apply, Rep.standardComplex.d_apply, lamStd_single, Rep.standardComplex.d_single,
      Rep.standardComplex.d_single, map_sum]
    refine Finset.sum_congr rfl fun p _ => ?_
    erw [lamStd_single]
    rfl

/-- `λ` preserves the augmentation. -/
lemma lamStd_eps (z : Rep.ofMulAction k G (Fin 1 → G)) :
    (Rep.standardComplex.ε k K).hom ((lamStd K σ k 0).hom z) = (Rep.standardComplex.ε k G).hom z := by
  induction z using MonoidAlgebra.induction_linear with
  | zero => simp only [map_zero]
  | add a b ha hb => simp only [map_add, ha, hb]
  | single y r =>
    rw [lamStd_single]
    simp [Rep.standardComplex.ε]

end Std

section Bar

variable {G : Type} [Group G] (K : Subgroup G) (σ : Transversal K) (k : Type) [CommRing k]

/-- Composition of morphisms of representations, on elements. -/
lemma rep_comp_apply {H : Type} [Group H] {A B C : Rep.{0} k H} (f : A ⟶ B) (g : B ⟶ C) (x : A) :
    (f ≫ g).hom x = g.hom (f.hom x) := rfl

/-- `bar → standard` on generators (mathlib's `isoStandardComplex`, LR-Y's `diagonalSuccIsoFree_inv_single`). -/
lemma isoStd_hom_single (n : ℕ) (x : Fin n → G) (g : G) (r : k) :
    (((Rep.barComplex.isoStandardComplex k G).hom.f n).hom (Finsupp.single x (MonoidAlgebra.single g r))) =
      MonoidAlgebra.single (g • Fin.partialProd x) r :=
  diagonalSuccIsoFree_inv_single n x g r

set_option maxRecDepth 10000 in
/-- `standard → bar` on generators: `(y_0, …, y_n) ↦ y_0 [y_{i-1}⁻¹ y_i]_i`. -/
lemma isoStd_inv_single (n : ℕ) (y : Fin (n + 1) → G) (r : k) :
    (((Rep.barComplex.isoStandardComplex k G).inv.f n).hom (MonoidAlgebra.single y r)) =
      Finsupp.single (fun i : Fin n => (y i.castSucc)⁻¹ * y i.succ) (MonoidAlgebra.single (y 0) r) := by
  have h := isoStd_hom_single k n (fun i : Fin n => (y i.castSucc)⁻¹ * y i.succ) (y 0) r
  rw [Fin.partialProd_left_inv] at h
  have hi := congrArg (fun F => (F.f n).hom (Finsupp.single (fun i : Fin n => (y i.castSucc)⁻¹ * y i.succ)
    (MonoidAlgebra.single (y 0) r))) (Rep.barComplex.isoStandardComplex k G).hom_inv_id
  simp only [HomologicalComplex.comp_f, HomologicalComplex.id_f] at hi
  have hi' : ((Rep.barComplex.isoStandardComplex k G).inv.f n).hom (((Rep.barComplex.isoStandardComplex k G).hom.f n).hom
      (Finsupp.single (fun i : Fin n => (y i.castSucc)⁻¹ * y i.succ) (MonoidAlgebra.single (y 0) r))) =
      Finsupp.single (fun i : Fin n => (y i.castSucc)⁻¹ * y i.succ) (MonoidAlgebra.single (y 0) r) := hi
  rw [h] at hi'
  exact hi'

/-- **`λ` on the bar resolutions**: `res_K bar_G → C_•(G) → C_•(K) → bar_K`. -/
def lamBar : resCx K.subtype (Rep.barComplex k G) ⟶ Rep.barComplex k K :=
  ((Rep.resFunctor K.subtype).mapHomologicalComplex _).map (Rep.barComplex.isoStandardComplex k G).hom ≫
    lamCx K σ k ≫ (Rep.barComplex.isoStandardComplex k K).inv

set_option maxRecDepth 10000 in
lemma lamBar_apply (n : ℕ) (z : (Rep.barComplex k G).X n) :
    ((lamBar K σ k).f n).hom z = ((Rep.barComplex.isoStandardComplex k K).inv.f n).hom ((lamStd K σ k n).hom
      (((Rep.barComplex.isoStandardComplex k G).hom.f n).hom z)) := rfl

set_option maxRecDepth 10000 in
/-- `lamBar` on generators. -/
lemma lamBar_single (n : ℕ) (x : Fin n → G) (g : G) (r : k) :
    ((lamBar K σ k).f n).hom (Finsupp.single x (MonoidAlgebra.single g r)) =
      Finsupp.single (fun i : Fin n => (σ.lam (g * Fin.partialProd x i.castSucc))⁻¹ * σ.lam (g * Fin.partialProd x i.succ))
        (MonoidAlgebra.single (σ.lam (g * Fin.partialProd x 0)) r) := by
  rw [lamBar_apply, isoStd_hom_single, lamStd_single]
  exact isoStd_inv_single k n (lamV K σ (n + 1) (g • Fin.partialProd x)) r

set_option maxRecDepth 10000 in
/-- `lamBar` lies over the identity of `k`. -/
lemma lamBar_π (z : (Rep.barComplex k G).X 0) :
    ((Rep.barResolution k K).π.f 0).hom (((lamBar K σ k).f 0).hom z) = ((Rep.barResolution k G).π.f 0).hom z := by
  have hi := congrArg (fun F => (F.f 0).hom ((lamStd K σ k 0).hom (((Rep.barComplex.isoStandardComplex k G).hom.f 0).hom z)))
    (Rep.barComplex.isoStandardComplex k K).inv_hom_id
  simp only [HomologicalComplex.comp_f, HomologicalComplex.id_f] at hi
  rw [rep_comp_apply] at hi
  have e1 : ((Rep.barResolution k K).π.f 0).hom (((lamBar K σ k).f 0).hom z) =
      (Rep.standardComplex.ε k K).hom (((Rep.barComplex.isoStandardComplex k K).hom.f 0).hom
        (((Rep.barComplex.isoStandardComplex k K).inv.f 0).hom ((lamStd K σ k 0).hom
          (((Rep.barComplex.isoStandardComplex k G).hom.f 0).hom z)))) := rfl
  have e2 : ((Rep.barResolution k G).π.f 0).hom z =
      (Rep.standardComplex.ε k G).hom (((Rep.barComplex.isoStandardComplex k G).hom.f 0).hom z) := rfl
  rw [e1, e2, hi]
  exact lamStd_eps K σ k _

/-- `lamBar` as a chain map of resolutions over the identity. -/
lemma lamBar_comm : lamBar K σ k ≫ (Rep.barResolution k K).π = (resPR K (Rep.barResolution k G)).π :=
  comm_of_el' (P := resPR K (Rep.barResolution k G)) (Q := Rep.barResolution k K) (lamBar K σ k)
    fun z => lamBar_π K σ k z

end Bar

section Cor

variable {G : Type} [Group G] {k : Type} [CommRing k] (K : Subgroup G) [K.FiniteIndex] (σ : Transversal K)

attribute [local instance] groupCohomology.Cores.fintypeQuot

lemma freeLiftLEquiv_symm_single {α : Type} (A : Rep.{0} k G) (u : α → A) (x : α) (g : G) (r : k) :
    ((Rep.freeLiftLEquiv k G α A).symm u).hom (Finsupp.single x (MonoidAlgebra.single g r)) = r • A.ρ g (u x) :=
  Representation.freeLift_single_single (σ := A.ρ) x g r u

lemma barρ_single (i : ℕ) (x : Fin i → G) (g h : G) (r : k) :
    ((Rep.barComplex k G).X i).ρ g (Finsupp.single x (MonoidAlgebra.single h r)) =
      Finsupp.single x (MonoidAlgebra.single (g * h) r) :=
  Representation.free_single_single g h x r

/-- The two sides of `corCx_inhomog` on one generator. -/
lemma corFin_eq_corHom (A : Rep.{0} k G) (i : ℕ) (u : (Fin i → K) → A) (x : Fin i → G) :
    corFin A σ i u x = (corHom K σ ((lamBar K σ k).f i ≫
      (Rep.freeLiftLEquiv k K (Fin i → K) (Rep.res K.subtype A)).symm u)).hom
        (Finsupp.single x (MonoidAlgebra.single 1 1)) := by
  rw [corHom_hom_apply, corFin_apply]
  refine Finset.sum_congr rfl fun q _ => ?_
  congr 1
  erw [rep_comp_apply, barρ_single, mul_one, lamBar_single, freeLiftLEquiv_symm_single, one_smul]
  have h0 : σ.lam ((σ.σ q)⁻¹ * Fin.partialProd x 0) = 1 := by
    rw [Fin.partialProd_zero, mul_one]
    exact Subtype.ext (σ.coe_lam_σ_inv q)
  rw [h0]
  show _ = A.ρ ((1 : K) : G) _
  rw [OneMemClass.coe_one, map_one, Module.End.one_apply]

/-- **`lem:m6-corfin-cochain` (A1-c at cochain level)**: the vendored `corFin` is the transfer `corY` on the bar
resolution after `λ^*`, under mathlib's `inhomogeneousCochainsIso` -- an equality of cochain maps. -/
theorem corCx_inhomog (A : Rep.{0} k G) :
    GL2.M5.corCx A K σ ≫ (inhomogeneousCochainsIso A).hom =
      (inhomogeneousCochainsIso (Rep.res K.subtype A)).hom ≫ Cover.preY (lamBar K σ k) (Rep.res K.subtype A) ≫
        corY K σ (Rep.barComplex k G) A := by
  ext i u : 3
  apply Rep.free_ext; intro x
  refine (freeLiftLEquiv_symm_single A (corFin A σ i u) x 1 1).trans ?_
  rw [one_smul, map_one, Module.End.one_apply]
  exact corFin_eq_corHom K σ A i u x

/-- M5's corestriction `corH` as a morphism `H^n(K; res A) ⟶ H^n(G; A)`. -/
def corHm (A : Rep.{0} k G) (n : ℕ) : groupCohomology (Rep.res K.subtype A) n ⟶ groupCohomology A n :=
  HomologicalComplex.homologyMap (GL2.M5.corCx A K σ) n

lemma corHm_hom (A : Rep.{0} k G) (n : ℕ) : (corHm K σ A n).hom = GL2.M5.corH A K σ n := rfl

/-- `canIso` of the bar resolution is mathlib's inhomogeneous-cochains isomorphism. -/
lemma canIso_bar (A : Rep.{0} k G) (n : ℕ) :
    (canIso (Rep.barResolution k G) A n).hom = HomologicalComplex.homologyMap (inhomogeneousCochainsIso A).hom n := by
  rw [canIso_hom_eq (Rep.barResolution k G) A n (𝟙 _) (Category.id_comp _), Cover.preY_id,
    HomologicalComplex.homologyMap_id]
  exact Category.comp_id _

/-- **A1-c on the bar resolution**: `corH ≫ canIso(bar) = canIso(res bar) ≫ H(corY)`. -/
theorem corH_canIso_bar (A : Rep.{0} k G) (n : ℕ) :
    corHm K σ A n ≫ (canIso (Rep.barResolution k G) A n).hom =
      (canIso (resPR K (Rep.barResolution k G)) (Rep.res K.subtype A) n).hom ≫
        HomologicalComplex.homologyMap (corY K σ (Rep.barResolution k G).complex A) n := by
  rw [canIso_bar,
    canIso_hom_eq (resPR K (Rep.barResolution k G)) (Rep.res K.subtype A) n (lamBar K σ k) (lamBar_comm K σ k)]
  have key := congrArg (fun F => HomologicalComplex.homologyMap F n) (corCx_inhomog K σ A)
  simp only [HomologicalComplex.homologyMap_comp] at key
  exact key.trans (Category.assoc _ _ _).symm

/-- **`thm:m6-cor-res` (A1-c)**: for **every** projective resolution `P` of `k` over `G`, the corestriction `corH` is
induced, through the canonical comparisons, by the transfer `corY` on `Hom(P, -)`. -/
theorem corH_canIso (P : ProjectiveResolution (Rep.trivial k G k)) (A : Rep.{0} k G) (n : ℕ) :
    corHm K σ A n ≫ (canIso P A n).hom =
      (canIso (resPR K P) (Rep.res K.subtype A) n).hom ≫ HomologicalComplex.homologyMap (corY K σ P.complex A) n := by
  set eP : P.complex ⟶ (Rep.barResolution k G).complex :=
    (ProjectiveResolution.homotopyEquiv P (Rep.barResolution k G)).hom
  have hP : eP ≫ (Rep.barResolution k G).π = P.π := ProjectiveResolution.homotopyEquiv_hom_π _ _
  have hR : ((Rep.resFunctor K.subtype).mapHomologicalComplex _).map eP ≫ (resPR K (Rep.barResolution k G)).π =
      (resPR K P).π :=
    comm_of_el' (P := resPR K P) (Q := resPR K (Rep.barResolution k G)) _ fun x => el_of_comm eP hP x
  have e1 := canIso_lift P (Rep.barResolution k G) A n eP hP
  have e2 := canIso_lift (resPR K P) (resPR K (Rep.barResolution k G)) (Rep.res K.subtype A) n _ hR
  have e3 := corH_canIso_bar K σ A n
  have e4 := congrArg (fun F => HomologicalComplex.homologyMap F n) (corY_natural K σ eP A)
  simp only [HomologicalComplex.homologyMap_comp] at e4
  rw [e1, e2, ← Category.assoc, e3]
  exact (Category.assoc _ _ _).trans ((congrArg (_ ≫ ·) e4.symm).trans (Category.assoc _ _ _).symm)

end Cor

end GL2.M6

end
