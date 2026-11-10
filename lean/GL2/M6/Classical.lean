/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.M6.BarCor

/-!
# M6-A1, A3 (slot 23): the Hecke operator does not depend on the resolution; `corH` in every degree

[GT] `gt_part_M6.tex`, rows `def:m6-heckeY`, `thm:m6-hecke-res`, `thm:m6-corH-indep`, `thm:m6-cor-res-index`,
`thm:m6-Tcls-res`. Brief `M6B_brief.md` §2.1 (A1), §3 (A3); decision D3 (resolution transport).

**The generic theorem (A1, `hecke_canIso`).** Let `K ≤ G` be of finite index with a transversal `σ`, `f : K → G` an
injective homomorphism (for a double coset: `h ↦ g⁻¹ h g`), `θ : res_f A ⟶ res_K A` (for a double coset: `ρ(g)`), `P` **any**
projective resolution of `k` over `G` and `τ : res_K P → res_f P` **any** chain map of `K`-representations over `id_k`
(a "twisted chain datum"). The Hecke operator on the resolution,

  `heckeY : Hom_G(P, A) → Hom_G(P, A)`, `ψ ↦ (x ↦ ∑_q σ_q · θ(ψ(τ(σ_q⁻¹ x))))`

(`resY`, then `τ^*`, then `θ_*`, then the transfer `corY`), induces on cohomology, through the canonical comparison
`canIso P`, the classical operator `cor ∘ (f, θ)_* = corH ∘ groupCohomology.map f θ`. It therefore does not depend on `P` or
`τ` (or `σ`).

**A3.** `corH_indep`: the corestriction `corH` (M5, from the vendored `corFin`) does not depend on the transversal, in
**every** degree; `corH_res`: `cor ∘ res = [G : K]` in **every** degree (the vendored homotopy covers degrees 2, 3 only).

**The `SL(2, ℤ)` instance (`Tcls_canIso`).** M5's `Tcls g n = corH ∘ (c_g)_* ∘ res` is the generic operator with
`f = (h ↦ g⁻¹ h g)`, `θ = ρ(g)` (`Tcls_eq`), so for every projective resolution `P` of `R` over `SL(2, ℤ)` and every `τ`,
`Tcls g n` is computed by `heckeY` on `Hom(P, V_L)`; and `Tcls` is independent of the transversal in every degree
(`TclsT_indep`), which M5 had for `n = 0, 1` only.
-/

noncomputable section

namespace GL2.M6

open CategoryTheory groupCohomology GL2.MP groupCohomology.Cores

section Generic

variable {G : Type} [Group G] {k : Type} [CommRing k] (K : Subgroup G) [K.FiniteIndex] (σ : Transversal K)

attribute [local instance] groupCohomology.Cores.fintypeQuot

/-- `postY` of the identity. -/
lemma postY_id {H : Type} [Group H] (X : ChainComplex (Rep.{0} k H) ℕ) (B : Rep.{0} k H) : Cover.postY X (𝟙 B) = 𝟙 (X.linearYonedaObj k B) := by
  ext i ψ : 3
  exact Category.comp_id (show X.X i ⟶ B from ψ)

/-- **`thm:m6-corH-indep` (A3): `corH` does not depend on the transversal, in every degree.** -/
theorem corHm_indep (σ' : Transversal K) (A : Rep.{0} k G) (n : ℕ) : corHm K σ A n = corHm K σ' A n := by
  rw [← cancel_mono (canIso (Rep.barResolution k G) A n).hom, corH_canIso, corH_canIso, corY_indep K σ σ']

theorem corH_indep (σ' : Transversal K) (A : Rep.{0} k G) (n : ℕ) :
    GL2.M5.corH A K σ n = GL2.M5.corH A K σ' n := by
  rw [← corHm_hom, ← corHm_hom, corHm_indep K σ σ']

/-- **`thm:m6-cor-res-index` (A3): `cor ∘ res = [G : K]`, in every degree.** -/
theorem corHm_res (A : Rep.{0} k G) (n : ℕ) :
    groupCohomology.map K.subtype (𝟙 (Rep.res K.subtype A)) n ≫ corHm K σ A n = K.index • 𝟙 _ := by
  let P := Rep.barResolution k G
  have hm := map_canIso K.subtype K.subtype_injective P (resPR K P) (𝟙 _) (fun _ => rfl) (𝟙 (Rep.res K.subtype A)) n
  have hc := corH_canIso K σ P A n
  have hY : (resY K.subtype P.complex A ≫ Cover.preY (𝟙 (resPR K P).complex) (Rep.res K.subtype A) ≫
      Cover.postY (resPR K P).complex (𝟙 (Rep.res K.subtype A))) ≫ corY K σ P.complex A = K.index • 𝟙 _ := by
    ext i φ : 3
    exact (corHom_res K σ φ).trans (Nat.cast_smul_eq_nsmul k _ φ)
  have hH : HomologicalComplex.homologyMap (K.index • 𝟙 (P.complex.linearYonedaObj k A)) n = K.index • 𝟙 _ := by
    have h : (HomologicalComplex.homologyFunctor (ModuleCat k) (ComplexShape.up ℕ) n).map
        (K.index • 𝟙 (P.complex.linearYonedaObj k A)) =
        K.index • (HomologicalComplex.homologyFunctor (ModuleCat k) (ComplexShape.up ℕ) n).map
          (𝟙 (P.complex.linearYonedaObj k A)) := CategoryTheory.Functor.map_nsmul _
    rw [CategoryTheory.Functor.map_id] at h
    exact h
  rw [← cancel_mono (canIso P A n).hom]
  calc (groupCohomology.map K.subtype (𝟙 (Rep.res K.subtype A)) n ≫ corHm K σ A n) ≫ (canIso P A n).hom
      = groupCohomology.map K.subtype (𝟙 (Rep.res K.subtype A)) n ≫ (corHm K σ A n ≫ (canIso P A n).hom) :=
        Category.assoc _ _ _
    _ = groupCohomology.map K.subtype (𝟙 (Rep.res K.subtype A)) n ≫
          ((canIso (resPR K P) (Rep.res K.subtype A) n).hom ≫
            HomologicalComplex.homologyMap (corY K σ P.complex A) n) := congrArg (_ ≫ ·) hc
    _ = (groupCohomology.map K.subtype (𝟙 (Rep.res K.subtype A)) n ≫
          (canIso (resPR K P) (Rep.res K.subtype A) n).hom) ≫
            HomologicalComplex.homologyMap (corY K σ P.complex A) n := (Category.assoc _ _ _).symm
    _ = ((canIso P A n).hom ≫ HomologicalComplex.homologyMap (resY K.subtype P.complex A ≫
          Cover.preY (𝟙 (resPR K P).complex) (Rep.res K.subtype A) ≫
            Cover.postY (resPR K P).complex (𝟙 (Rep.res K.subtype A))) n) ≫
            HomologicalComplex.homologyMap (corY K σ P.complex A) n :=
        congrArg (· ≫ HomologicalComplex.homologyMap (corY K σ P.complex A) n) hm
    _ = (canIso P A n).hom ≫ HomologicalComplex.homologyMap ((resY K.subtype P.complex A ≫
          Cover.preY (𝟙 (resPR K P).complex) (Rep.res K.subtype A) ≫
            Cover.postY (resPR K P).complex (𝟙 (Rep.res K.subtype A))) ≫ corY K σ P.complex A) n :=
        (Category.assoc _ _ _).trans (congrArg (_ ≫ ·) (HomologicalComplex.homologyMap_comp _ _ _).symm)
    _ = (canIso P A n).hom ≫ (K.index • 𝟙 _) := by rw [hY, hH]
    _ = (K.index • 𝟙 _) ≫ (canIso P A n).hom := by
        rw [Preadditive.comp_nsmul, Preadditive.nsmul_comp, Category.comp_id, Category.id_comp]

theorem corH_res (A : Rep.{0} k G) (n : ℕ) :
    GL2.M5.corH A K σ n ∘ₗ (groupCohomology.map K.subtype (𝟙 (Rep.res K.subtype A)) n).hom =
      K.index • LinearMap.id := by
  have h := congrArg ModuleCat.Hom.hom (corHm_res K σ A n)
  rw [ModuleCat.hom_comp, ModuleCat.hom_nsmul, ModuleCat.hom_id] at h
  exact h

variable (f : K →* G) (hf : Function.Injective f) {A : Rep.{0} k G} (θ : Rep.res f A ⟶ Rep.res K.subtype A)
  (P : ProjectiveResolution (Rep.trivial k G k))

/-- **`def:m6-heckeY`: the Hecke operator on a resolution** `ψ ↦ ∑_q σ_q · θ(ψ(τ(σ_q⁻¹ ·)))` on `Hom_G(P, A)`. -/
def heckeY (τ : resCx K.subtype P.complex ⟶ resCx f P.complex) :
    P.complex.linearYonedaObj k A ⟶ P.complex.linearYonedaObj k A :=
  (resY f P.complex A ≫ Cover.preY τ (Rep.res f A) ≫ Cover.postY (resPR K P).complex θ) ≫
    corY K σ P.complex A

/-- The classical operator `cor ∘ (f, θ)_*` on `H^n(G; A)`. -/
def Tgen (n : ℕ) : groupCohomology A n ⟶ groupCohomology A n :=
  groupCohomology.map f θ n ≫ corHm K σ A n

include hf in
/-- **`thm:m6-hecke-res` (A1): the Hecke operator does not depend on the resolution.** For every projective resolution
`P` of `k` over `G` and every chain map `τ : res_K P → res_f P` over `id_k`, `heckeY` on `Hom_G(P, A)` induces, through
the canonical comparison, the classical operator `cor ∘ (f, θ)_*`. -/
theorem hecke_canIso (τ : resCx K.subtype P.complex ⟶ resCx f P.complex)
    (hτ : ∀ x, (P.π.f 0).hom ((τ.f 0).hom x) = (P.π.f 0).hom x) (n : ℕ) :
    Tgen K σ f θ n ≫ (canIso P A n).hom =
      (canIso P A n).hom ≫ HomologicalComplex.homologyMap (heckeY K σ f θ P τ) n := by
  have hm := map_canIso f hf P (resPR K P) τ hτ θ n
  have hc := corH_canIso K σ P A n
  unfold Tgen heckeY
  exact (Category.assoc _ _ _).trans ((congrArg (_ ≫ ·) hc).trans ((Category.assoc _ _ _).symm.trans
    ((congrArg (· ≫ HomologicalComplex.homologyMap (corY K σ P.complex A) n) hm).trans
      ((Category.assoc _ _ _).trans (congrArg (_ ≫ ·) (HomologicalComplex.homologyMap_comp _ _ _).symm)))))

end Generic

/-! ### The `SL(2, ℤ)` instance: M5's `Tcls` -/

section Model

open scoped MatrixGroups
open GL2.Model GL2.Model.XBulk GL2.Cone GL2.M4 GL2.M5

variable {R : Type} [CommRing R] {V : Type} [AddCommGroup V] [Module R V] (ρ : ΔQ →* Module.End R V)

/-- `h ↦ g⁻¹ h g`, as an injective homomorphism `Γ'' → SL(2, ℤ)`. -/
def fConjS (g : GLQ) : Spp g →* SL(2, ℤ) := (Spp g⁻¹).subtype.comp (GL2.M4.conjHom g)

lemma fConjS_injective (g : GLQ) : Function.Injective (fConjS g) := by
  intro a b h
  have h1 : ((GL2.M4.conjHom g a : SL(2, ℤ))) = (GL2.M4.conjHom g b : SL(2, ℤ)) := h
  have h2 := congrArg (fun x : SL(2, ℤ) => ((eQ.symm x : ΓQ) : GLQ)) h1
  dsimp only [GL2.M4.conjHom, MonoidHom.coe_mk, OneHom.coe_mk] at h2
  rw [MulEquiv.symm_apply_apply, MulEquiv.symm_apply_apply] at h2
  have h3 : g⁻¹ * ((eQ.symm a.1 : ΓQ) : GLQ) * g = g⁻¹ * ((eQ.symm b.1 : ΓQ) : GLQ) * g := h2
  have h4 : ((eQ.symm a.1 : ΓQ) : GLQ) = ((eQ.symm b.1 : ΓQ) : GLQ) := mul_left_cancel (mul_right_cancel h3)
  exact Subtype.ext (eQ.symm.injective (Subtype.ext h4))

/-- The coefficient map `ρ(g)` along `fConjS g`. -/
def thetaS (g : GLQ) (hg : g ∈ ΔQ) :
    Rep.res (fConjS g) (VL ρ) ⟶ Rep.res (Spp g).subtype (VL ρ) :=
  (Rep.resFunctor (GL2.M4.conjHom g)).map (𝟙 (Rep.res (Spp g⁻¹).subtype (VL ρ))) ≫ conjCoeff ρ g hg

/-- **M5's `Tcls` is the generic operator** `cor ∘ (fConjS g, ρ(g))_*` (`groupCohomology.map_comp`). -/
theorem TclsT_eq (g : GLQ) (hg : g ∈ ΔQ) (τ : Transversal (Spp g)) (n : ℕ) :
    TclsT ρ g hg τ n = (Tgen (Spp g) τ (fConjS g) (thetaS ρ g hg) n).hom := by
  unfold Tgen thetaS fConjS
  rw [groupCohomology.map_comp]
  rfl

/-- **`thm:m6-corH-indep` for `Tcls`: the classical operator does not depend on the transversal, in every degree**
(M5 had `n = 0, 1`). -/
theorem TclsT_indep (g : GLQ) (hg : g ∈ ΔQ) (τ τ' : Transversal (Spp g)) (n : ℕ) :
    TclsT ρ g hg τ n = TclsT ρ g hg τ' n := by
  rw [TclsT_eq, TclsT_eq]
  unfold Tgen
  rw [corHm_indep (Spp g) τ τ']

/-- **`thm:m6-Tcls-res` (A1 on `SL(2, ℤ)`): `Tcls g n` is computed by the Hecke operator on any resolution.** For
every projective resolution `P` of `R` over `SL(2, ℤ)` and every chain map `τ : res_{Γ''} P → res_{h ↦ g⁻¹hg} P` over
`id_R`, `Tcls g n` is, through `canIso P`, induced by `heckeY` (`ψ ↦ ∑_q σ_q ρ(g) ψ(τ(σ_q⁻¹ ·))`). -/
theorem Tcls_canIso (g : GLQ) (hg : g ∈ ΔQ) (P : ProjectiveResolution (Rep.trivial R SL(2, ℤ) R))
    (τ : resCx (Spp g).subtype P.complex ⟶ resCx (fConjS g) P.complex)
    (hτ : ∀ x, (P.π.f 0).hom ((τ.f 0).hom x) = (P.π.f 0).hom x) (n : ℕ) :
    (canIso P (VL ρ) n).hom.hom ∘ₗ Tcls ρ g hg n =
      (HomologicalComplex.homologyMap (heckeY (Spp g) (τS g) (fConjS g) (thetaS ρ g hg) P τ) n).hom ∘ₗ
        (canIso P (VL ρ) n).hom.hom := by
  have h := congrArg ModuleCat.Hom.hom
    (hecke_canIso (Spp g) (τS g) (fConjS g) (fConjS_injective g) (thetaS ρ g hg) P τ hτ n)
  rw [ModuleCat.hom_comp, ModuleCat.hom_comp] at h
  rw [Tcls, TclsT_eq]
  exact h

end Model

end GL2.M6

end
