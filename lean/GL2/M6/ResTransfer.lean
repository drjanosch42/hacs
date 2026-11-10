/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.MP.TreeCyclic
import GL2.M5.Transport

/-!
# M6-A1 (a), (b) (slot 23): the canonical comparison, and the transfer on any resolution

[GT] `gt_part_M6.tex`, rows `def:m6-can`, `lem:m6-can-lift`, `def:m6-corY`, `lem:m6-corY-indep`, `lem:m6-corY-res`,
`lem:m6-corY-natural`. Brief `docs/implementation/slot22/M6B_brief.md` §2.1 (A1-a, A1-b), the route of decision D3
(resolution transport, not `δ`-functors).

* `canIso P A n` (**`Θ_can`'s building block**): for a projective resolution `P` of `k` over `G`,
  `H^n(G; A) ≅ H^n(Hom_G(P, A))` through mathlib's inhomogeneous cochains `= Hom(bar, A)` and **the** comparison
  `P ≃ bar` of `ProjectiveResolution.homotopyEquiv` (as `PermRes.barIso`, `GL2.MP.TreeCyclic`; not through `Ext`).
  `canIso_hom_eq`: it is induced by **any** chain map `P → bar` over `id_k`; `canIso_lift`: for any chain map
  `φ : P → Q` over `id_k`, `canIso P = canIso Q ≫ H(φ^*)` (homotopy uniqueness).
* `corY X σ A` (**the transfer on a resolution**, Brown III.9): for `K ≤ G` of finite index with a (left) transversal
  `σ` (the vendored `Cores.Transversal`), and any chain complex `X` of `G`-representations, the cochain map
  `Hom_K(X, A) → Hom_G(X, A)`, `χ ↦ (x ↦ ∑_{q ∈ G/K} σ_q · χ(σ_q⁻¹ x))`.
  `corY_indep`: it does not depend on the transversal (cochain level); `corY_resY`: `corY ∘ res = [G : K]`
  (cochain level); `corY_natural`: it commutes with precomposition by every chain map of `G`-representations.
-/

noncomputable section

namespace GL2.M6

open CategoryTheory groupCohomology GL2.MP

section Can

variable {G : Type} [Group G] {k : Type} [CommRing k]

/-- **The canonical comparison** `H^n(G; A) ≅ H^n(Hom_G(P, A))` for a projective resolution `P` of `k`: inhomogeneous
cochains `= Hom(bar, A)` (mathlib), then the fixed comparison `P ≃ bar` (`ProjectiveResolution.homotopyEquiv`). -/
def canIso (P : ProjectiveResolution (Rep.trivial k G k)) (A : Rep.{0} k G) (n : ℕ) :
    groupCohomology A n ≅ (P.complex.linearYonedaObj k A).homology n :=
  (HomologicalComplex.homologyFunctor _ _ n).mapIso (inhomogeneousCochainsIso A) ≪≫
    (Cover.preYEquiv (ProjectiveResolution.homotopyEquiv P (Rep.barResolution k G)) A).toHomologyIso n

/-- **`canIso` is induced by any chain map `P → bar` over `id_k`.** -/
theorem canIso_hom_eq (P : ProjectiveResolution (Rep.trivial k G k)) (A : Rep.{0} k G) (n : ℕ)
    (φ : P.complex ⟶ (Rep.barResolution k G).complex) (hφ : φ ≫ (Rep.barResolution k G).π = P.π) :
    (canIso P A n).hom = HomologicalComplex.homologyMap (inhomogeneousCochainsIso A).hom n ≫
      HomologicalComplex.homologyMap (Cover.preY φ A) n := by
  have h : Homotopy (ProjectiveResolution.homotopyEquiv P (Rep.barResolution k G)).hom φ :=
    ProjectiveResolution.liftHomotopy (𝟙 _) _ _
      (by rw [ProjectiveResolution.homotopyEquiv_hom_π, CategoryTheory.Functor.map_id, Category.comp_id])
      (by rw [hφ, CategoryTheory.Functor.map_id, Category.comp_id])
  have hH := (preYHomotopy h A).homologyMap_eq n
  show HomologicalComplex.homologyMap (inhomogeneousCochainsIso A).hom n ≫
      HomologicalComplex.homologyMap (Cover.preY (ProjectiveResolution.homotopyEquiv P (Rep.barResolution k G)).hom A) n = _
  rw [hH]

/-- **Change of resolution**: for a chain map `φ : P → Q` over `id_k`, `canIso P = canIso Q ≫ H(φ^*)`. -/
theorem canIso_lift (P Q : ProjectiveResolution (Rep.trivial k G k)) (A : Rep.{0} k G) (n : ℕ)
    (φ : P.complex ⟶ Q.complex) (hφ : φ ≫ Q.π = P.π) :
    (canIso P A n).hom = (canIso Q A n).hom ≫ HomologicalComplex.homologyMap (Cover.preY φ A) n := by
  have hQ := ProjectiveResolution.homotopyEquiv_hom_π Q (Rep.barResolution k G)
  rw [canIso_hom_eq P A n (φ ≫ (ProjectiveResolution.homotopyEquiv Q (Rep.barResolution k G)).hom)
      (by rw [Category.assoc, hQ, hφ]),
    canIso_hom_eq Q A n _ hQ, Cover.preY_comp, HomologicalComplex.homologyMap_comp]
  exact (Category.assoc _ _ _).symm

end Can

/-! ### The transfer on a resolution -/

section Cor

variable {G : Type} [Group G] {k : Type} [CommRing k] (K : Subgroup G) [K.FiniteIndex]
  (σ : groupCohomology.Cores.Transversal K)

attribute [local instance] groupCohomology.Cores.fintypeQuot

/-- The transfer of one `K`-map `χ : M → A`: `x ↦ ∑_q σ_q · χ(σ_q⁻¹ x)`, as a linear map. -/
def corLin {M A : Rep.{0} k G} (χ : Rep.res K.subtype M ⟶ Rep.res K.subtype A) : M →ₗ[k] A :=
  ∑ q : G ⧸ K, A.ρ (σ.σ q) ∘ₗ χ.hom.toLinearMap ∘ₗ M.ρ (σ.σ q)⁻¹

lemma corLin_apply {M A : Rep.{0} k G} (χ : Rep.res K.subtype M ⟶ Rep.res K.subtype A) (x : M) :
    corLin K σ χ x = ∑ q : G ⧸ K, A.ρ (σ.σ q) (χ.hom (M.ρ (σ.σ q)⁻¹ x)) := by
  rw [corLin, LinearMap.sum_apply]
  rfl

omit [K.FiniteIndex] in
/-- `χ` commutes with `K`. -/
lemma resHom_comm {M A : Rep.{0} k G} (χ : Rep.res K.subtype M ⟶ Rep.res K.subtype A) (h : K) (x : M) :
    χ.hom (M.ρ (h : G) x) = A.ρ (h : G) (χ.hom x) :=
  Rep.hom_comm_apply χ h x

/-- **The transfer is `G`-equivariant.** -/
lemma corLin_comm {M A : Rep.{0} k G} (χ : Rep.res K.subtype M ⟶ Rep.res K.subtype A) (γ : G) (x : M) :
    corLin K σ χ (M.ρ γ x) = A.ρ γ (corLin K σ χ x) := by
  rw [corLin_apply, corLin_apply, map_sum]
  refine (Fintype.sum_equiv (MulAction.toPerm γ⁻¹ : Equiv.Perm (G ⧸ K)) _ _ fun q => ?_)
  -- `γ σ(γ⁻¹ q) = σ_q λ(σ_q⁻¹ γ)`
  have hm := σ.mul_σ q γ
  set h : K := σ.lam ((σ.σ q)⁻¹ * γ)
  have e1 : (σ.σ q)⁻¹ * γ = (h : G) * (σ.σ (γ⁻¹ • q))⁻¹ := by
    calc (σ.σ q)⁻¹ * γ = (σ.σ q)⁻¹ * (γ * σ.σ (γ⁻¹ • q)) * (σ.σ (γ⁻¹ • q))⁻¹ := by group
      _ = _ := by rw [hm]; group
  show A.ρ (σ.σ q) (χ.hom (M.ρ (σ.σ q)⁻¹ (M.ρ γ x))) = A.ρ γ (A.ρ (σ.σ (γ⁻¹ • q)) (χ.hom (M.ρ (σ.σ (γ⁻¹ • q))⁻¹ x)))
  rw [← Module.End.mul_apply (M.ρ (σ.σ q)⁻¹), ← map_mul, e1, map_mul, Module.End.mul_apply, resHom_comm,
    ← Module.End.mul_apply (A.ρ (σ.σ q)), ← map_mul, ← Module.End.mul_apply (A.ρ γ), ← map_mul]
  congr 2
  calc σ.σ q * (h : G) = γ * σ.σ (γ⁻¹ • q) := hm.symm

/-- The transfer as a morphism of representations. -/
def corHom {M A : Rep.{0} k G} (χ : Rep.res K.subtype M ⟶ Rep.res K.subtype A) : M ⟶ A :=
  Rep.ofHom (LinearMap.intertwiningMap_of_isIntertwiningMap M.ρ A.ρ (corLin K σ χ) fun γ x => corLin_comm K σ χ γ x)

lemma corHom_hom_apply {M A : Rep.{0} k G} (χ : Rep.res K.subtype M ⟶ Rep.res K.subtype A) (x : M) :
    (corHom K σ χ).hom x = ∑ q : G ⧸ K, A.ρ (σ.σ q) (χ.hom (M.ρ (σ.σ q)⁻¹ x)) :=
  corLin_apply K σ χ x

/-- The transfer is linear in `χ`. -/
def corHomL (M A : Rep.{0} k G) : (Rep.res K.subtype M ⟶ Rep.res K.subtype A) →ₗ[k] (M ⟶ A) where
  toFun := corHom K σ
  map_add' χ ψ := by
    apply Rep.hom_ext; apply Representation.IntertwiningMap.ext; refine LinearMap.ext fun x => ?_
    show (corHom K σ (χ + ψ)).hom x = (corHom K σ χ).hom x + (corHom K σ ψ).hom x
    simp only [corHom_hom_apply, ← Finset.sum_add_distrib, ← map_add]
    rfl
  map_smul' c χ := by
    apply Rep.hom_ext; apply Representation.IntertwiningMap.ext; refine LinearMap.ext fun x => ?_
    show (corHom K σ (c • χ)).hom x = c • (corHom K σ χ).hom x
    rw [corHom_hom_apply, corHom_hom_apply, Finset.smul_sum]
    refine Finset.sum_congr rfl fun q _ => ?_
    rw [← map_smul]
    rfl

/-- The transfer commutes with precomposition by a map of `G`-representations. -/
lemma corHom_comp {M N A : Rep.{0} k G} (F : M ⟶ N) (χ : Rep.res K.subtype N ⟶ Rep.res K.subtype A) :
    corHom K σ ((Rep.resFunctor K.subtype).map F ≫ χ) = F ≫ corHom K σ χ := by
  apply Rep.hom_ext; apply Representation.IntertwiningMap.ext; refine LinearMap.ext fun x => ?_
  show (corHom K σ ((Rep.resFunctor K.subtype).map F ≫ χ)).hom x = (corHom K σ χ).hom (F.hom x)
  simp only [corHom_hom_apply]
  refine Finset.sum_congr rfl fun q _ => ?_
  congr 1
  show χ.hom (F.hom (M.ρ (σ.σ q)⁻¹ x)) = _
  rw [Rep.hom_comm_apply]

/-- The transfer of one map does not depend on the transversal. -/
lemma corHom_indep (σ' : groupCohomology.Cores.Transversal K) {M A : Rep.{0} k G}
    (χ : Rep.res K.subtype M ⟶ Rep.res K.subtype A) : corHom K σ χ = corHom K σ' χ := by
  apply Rep.hom_ext; apply Representation.IntertwiningMap.ext; refine LinearMap.ext fun x => ?_
  show (corHom K σ χ).hom x = (corHom K σ' χ).hom x
  simp only [corHom_hom_apply]
  refine Finset.sum_congr rfl fun q _ => ?_
  -- `σ_q = σ'_q h` with `h ∈ K`
  have hmem : (σ'.σ q)⁻¹ * σ.σ q ∈ K := by
    rw [← QuotientGroup.eq, σ.mk_σ, σ'.mk_σ]
  set h : K := ⟨_, hmem⟩
  have e : σ.σ q = σ'.σ q * (h : G) := by simp [h]
  have hc : ∀ y : M, χ.hom (M.ρ (h : G)⁻¹ y) = A.ρ (h : G)⁻¹ (χ.hom y) := fun y => resHom_comm K χ h⁻¹ y
  rw [e, mul_inv_rev, map_mul M.ρ, Module.End.mul_apply, hc, map_mul A.ρ, Module.End.mul_apply,
    ← Module.End.mul_apply (A.ρ (h : G)), ← map_mul, mul_inv_cancel, map_one, Module.End.one_apply]

/-- The transfer of a restricted map is `[G : K]` times it. -/
lemma corHom_res {M A : Rep.{0} k G} (φ : M ⟶ A) :
    corHom K σ ((Rep.resFunctor K.subtype).map φ) = (K.index : k) • φ := by
  apply Rep.hom_ext; apply Representation.IntertwiningMap.ext; refine LinearMap.ext fun x => ?_
  show (corHom K σ ((Rep.resFunctor K.subtype).map φ)).hom x = (K.index : k) • φ.hom x
  rw [corHom_hom_apply]
  have : ∀ q : G ⧸ K, A.ρ (σ.σ q) (((Rep.resFunctor K.subtype).map φ).hom (M.ρ (σ.σ q)⁻¹ x)) = φ.hom x :=
    fun q => by
      show A.ρ (σ.σ q) (φ.hom (M.ρ (σ.σ q)⁻¹ x)) = φ.hom x
      rw [Rep.hom_comm_apply, ← Module.End.mul_apply, ← map_mul, mul_inv_cancel, map_one, Module.End.one_apply]
  rw [Finset.sum_congr rfl fun q _ => this q, Finset.sum_const, Finset.card_univ, ← Nat.card_eq_fintype_card,
    ← Subgroup.index, Nat.cast_smul_eq_nsmul]

/-- **The transfer on a complex**: `corY : Hom_K(res X, res A) ⟶ Hom_G(X, A)`, a cochain map. -/
def corY (X : ChainComplex (Rep.{0} k G) ℕ) (A : Rep.{0} k G) :
    (resCx K.subtype X).linearYonedaObj k (Rep.res K.subtype A) ⟶ X.linearYonedaObj k A where
  f i := ModuleCat.ofHom (corHomL K σ (X.X i) A)
  comm' i j _ := by
    ext χ : 2
    exact (corHom_comp K σ (X.d j i) χ).symm

lemma corY_f_apply (X : ChainComplex (Rep.{0} k G) ℕ) (A : Rep.{0} k G) (i : ℕ)
    (χ : Rep.res K.subtype (X.X i) ⟶ Rep.res K.subtype A) :
    ((corY K σ X A).f i).hom χ = corHom K σ χ := rfl

/-- **`lem:m6-corY-indep`: the transfer does not depend on the transversal** (cochain level). -/
theorem corY_indep (σ' : groupCohomology.Cores.Transversal K) (X : ChainComplex (Rep.{0} k G) ℕ) (A : Rep.{0} k G) :
    corY K σ X A = corY K σ' X A := by
  ext i χ : 3
  exact corHom_indep K σ σ' χ

/-- **`lem:m6-corY-res`: `cor ∘ res = [G : K]`** on `Hom_G(X, A)` (cochain level). -/
theorem corY_resY (X : ChainComplex (Rep.{0} k G) ℕ) (A : Rep.{0} k G) :
    resY K.subtype X A ≫ corY K σ X A = (K.index : k) • 𝟙 _ := by
  ext i φ : 3
  exact corHom_res K σ φ

/-- **`lem:m6-corY-natural`: the transfer commutes with every chain map of `G`-representations.** -/
theorem corY_natural {X Y : ChainComplex (Rep.{0} k G) ℕ} (F : X ⟶ Y) (A : Rep.{0} k G) :
    Cover.preY (((Rep.resFunctor K.subtype).mapHomologicalComplex _).map F) (Rep.res K.subtype A) ≫ corY K σ X A =
      corY K σ Y A ≫ Cover.preY F A := by
  ext i χ : 3
  exact corHom_comp K σ (F.f i) χ

end Cor

end GL2.M6

end
