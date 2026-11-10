import GL2.M4.ClassicalDefs
import GL2.M4.Data
import Theorems.Thm_groupCohomology_Cores_corFin_d
import Theorems.Thm_HeckeCohomology_heckeH1_eq_of_section
import Theorems.Thm_HeckeCohomology_heckeH1_delta0

/-!
# M5-T (slot 22): the transport of FLT's `heckeH1` to `End(H^n)`

[GT] `gt_part_M5.tex`, rows `def:m5-corH`, `lem:m5-corH-one`, `lem:m5-corH-zero`, `def:m5-Tcls`, `thm:m5-heckeH1`,
`thm:m5-delta0`.

The vendored `HeckeCohomology.heckeH1` (Anthropic FLT, `lean/Vendor/AnthropicFLT`, byte-identical) is the double-coset
operator `cor ∘ (c_g)_* ∘ res` **in degree 1 only**, at cocycle level. Here:

* `corCx`, `corH` (T1): the corestriction `H^n(H; res A) → H^n(G; A)` in **every** degree `n`, for `H ≤ G` of finite
  index and a transversal: the map on cohomology of the vendored cochain map `Cores.corFin` (`Cores.corFin_d`);
  `corH_H1π`: in degree 1 it is the vendored cocycle formula `Cores.cor₁` (`u ↦ (γ ↦ ∑_q σ_q · u(λ(σ_q⁻¹ γ)))`).
* `Tcls`, `TclsM` (T2): the classical operator `Tcls g n = corH ∘ (c_g)_* ∘ res` on `H^n(SL(2, ℤ); V_L)` with M4's
  `conjMap`, `resTo` ([GT] `def:m4-classical`) and **the genuine corestriction** (at a chosen transversal; independence in every degree since slot 23, `GL2.M6.TclsT_indep`) (no hypothesised `cor`), and its transport
  `TclsM g n = Θ⁻¹ ∘ Tcls g n ∘ Θ ∈ End_R(H^n)` to the model (`ΘE`: `Θ` as an isomorphism, `6 ∈ R^×`).
* `Tcls_one` (T3): in degree 1, `Tcls g 1` **is** FLT's `heckeH1 Γ' Γ'' (c_g) V_L ρ(g)` (mathlib's `H1 A` is
  `groupCohomology A 1`); the transversal step is FLT's `heckeH1_eq_of_section`, the dictionary
  `r(Γ'' x) = σ(x⁻¹ Γ'')⁻¹` between FLT's right cosets and the left-coset transversal of `corFin`.

* `corH_zero`, `TclsT_zero`, `TclsT_delta0`: in degree 0 (through `H^0 ≅` invariants) `cor a = ∑_q σ_q a`, the classical
  operator is FLT's `heckeInv`, and with FLT's `heckeH1_delta0` the classical operator commutes with `δ : H^0 → H^1` for
  short exact sequences of `Δ`-modules (the first `δ`-step of the pinning route, classical side only).

So `TclsM g 1` is FLT's `heckeH1`, transported to `End(H^1)` of the model, and `TclsM g n` is its extension to every
degree (transversal-independent in degrees 0 and 1: `TclsT_one_indep`, `TclsT_zero`). **Not claimed** (M5 cut §1, M5-C): `Θ ∘ T_{ΓgΓ} = Tcls g n ∘ Θ` ([C] `thm:classical` in [C]'s sense; M6). **Slot 23 (M6-A):** proved in `GL2.M6.ModelHecke` / `GL2.M6.ThetaExt` (`classical_Θ`,
`TclsM_eq_TdcOf`), with transversal independence in every degree (`GL2.M6.TclsT_indep`).
-/

noncomputable section

namespace GL2.M5

open CategoryTheory groupCohomology groupCohomology.Cores

/-! ### T1: the corestriction in every degree -/

section Cor

variable {k G : Type} [CommRing k] [Group G] (A : Rep.{0} k G) (H : Subgroup G) [H.FiniteIndex]
  (τ : Transversal H)

attribute [local instance] fintypeQuot

/-- The vendored `corFin` in degree `n` as a `k`-linear map. -/
def corFinL (n : ℕ) : ((Fin n → H) → A) →ₗ[k] ((Fin n → G) → A) where
  toFun := corFin A τ n
  map_add' u v := by
    funext g
    simp only [corFin_apply, Pi.add_apply, map_add, Finset.sum_add_distrib]
  map_smul' c u := by
    funext g
    simp only [corFin_apply, Pi.smul_apply, map_smul, Finset.smul_sum, RingHom.id_apply]

/-- **The corestriction cochain map** `C^•(H; res A) → C^•(G; A)` (vendored `corFin`, `corFin_d`). -/
def corCx : inhomogeneousCochains (Rep.res H.subtype A) ⟶ inhomogeneousCochains A where
  f n := ModuleCat.ofHom (corFinL A H τ n)
  comm' i j hij := by
    obtain rfl : i + 1 = j := hij
    refine ModuleCat.hom_ext (LinearMap.ext fun u => ?_)
    exact (corFin_d A H τ i u).symm

/-- **The corestriction** `cor : H^n(H; res A) → H^n(G; A)`, every `n`. -/
def corH (n : ℕ) : groupCohomology (Rep.res H.subtype A) n →ₗ[k] groupCohomology A n :=
  (HomologicalComplex.homologyMap (corCx A H τ) n).hom

/-- The vendored degree-1 transfer `cor₁` maps cocycles to cocycles (`cor₂_d₁₂`). -/
lemma cor₁_mem (u : cocycles₁ (Rep.res H.subtype A)) : cor₁ A τ u ∈ cocycles₁ A := by
  show (d₁₂ A).hom (cor₁ A τ u) = 0
  rw [← cor₂_d₁₂]
  have hu : (d₁₂ (Rep.res H.subtype A)).hom u = 0 := u.2
  rw [hu]
  have h0 := cor₂_smul A τ 0 0
  simp only [zero_smul] at h0
  exact h0

/-- `cor₁` on cocycles. -/
def cor₁Z (u : cocycles₁ (Rep.res H.subtype A)) : cocycles₁ A := ⟨cor₁ A τ u, cor₁_mem A H τ u⟩

/-- `corFin` in degree 1 is `cor₁`, read through `C^1 = Fun(G, A)`. -/
lemma corFin_one (u : H → A) (g : Fin 1 → G) :
    corFin A τ 1 (fun h => u (h 0)) g = cor₁ A τ u (g 0) := by
  rw [corFin_apply, cor₁_apply]
  refine Finset.sum_congr rfl fun q _ => ?_
  congr 2
  have h0 : Fin.partialProd g (Fin.castSucc (0 : Fin 1)) = 1 := Fin.partialProd_zero g
  have h1 : Fin.partialProd g (0 : Fin 1).succ = g 0 := by
    rw [Fin.partialProd_succ, h0, one_mul]
  show (τ.lam ((τ.σ q)⁻¹ * Fin.partialProd g (Fin.castSucc 0)))⁻¹ * τ.lam ((τ.σ q)⁻¹ * Fin.partialProd g (Fin.succ 0)) = _
  rw [h0, h1, mul_one]
  have hl : τ.lam (τ.σ q)⁻¹ = 1 := Subtype.ext (τ.coe_lam_σ_inv q)
  rw [hl, inv_one, one_mul]

/-- **T1 in degree 1**: `corH` on the class of a cocycle `u` is the class of `cor₁ u`. -/
theorem corH_H1π (u : cocycles₁ (Rep.res H.subtype A)) :
    corH A H τ 1 (H1π _ u) = H1π A (cor₁Z A H τ u) := by
  have hcyc : HomologicalComplex.cyclesMap (corCx A H τ) 1 ((isoCocycles₁ _).inv u) =
      (isoCocycles₁ A).inv (cor₁Z A H τ u) := by
    apply (ModuleCat.mono_iff_injective ((inhomogeneousCochains A).iCycles 1)).1 inferInstance
    rw [← ModuleCat.comp_apply, HomologicalComplex.cyclesMap_i, ModuleCat.comp_apply]
    have e1 := isoCocycles₁_inv_comp_iCocycles_apply (Rep.res H.subtype A) u
    have e2 := isoCocycles₁_inv_comp_iCocycles_apply A (cor₁Z A H τ u)
    simp only [iCocycles] at e1 e2
    rw [e1, e2]
    funext g
    exact corFin_one A H τ u g
  unfold corH H1π
  show (HomologicalComplex.homologyMap (corCx A H τ) 1).hom ((π _ 1).hom ((isoCocycles₁ _).inv.hom u)) =
    (π A 1).hom ((isoCocycles₁ A).inv.hom (cor₁Z A H τ u))
  rw [← hcyc]
  exact congrArg (fun F => ModuleCat.Hom.hom F ((isoCocycles₁ _).inv.hom u))
    (HomologicalComplex.homologyπ_naturality (corCx A H τ) 1)

/-- **T1 in degree 0**: on `H^0` = invariants, `cor a = ∑_q σ_q · a`. -/
theorem corH_zero (x : groupCohomology (Rep.res H.subtype A) 0) :
    (((H0Iso A).hom (corH A H τ 0 x) : A.ρ.invariants) : A) =
      ∑ q : G ⧸ H, A.ρ (τ.σ q) (((H0Iso (Rep.res H.subtype A)).hom x : (Rep.res H.subtype A).ρ.invariants) : A) := by
  obtain ⟨c, rfl⟩ : ∃ c, (π (Rep.res H.subtype A) 0).hom c = x := by
    refine ⟨(cocyclesIso₀ _).inv ((H0Iso _).hom x), ?_⟩
    rw [← Iso.hom_inv_id_apply (H0Iso _) ((π _ 0).hom _), π_comp_H0Iso_hom_apply, Iso.inv_hom_id_apply,
      Iso.hom_inv_id_apply]
  have h1 : corH A H τ 0 ((π _ 0).hom c) = (π A 0).hom ((HomologicalComplex.cyclesMap (corCx A H τ) 0).hom c) :=
    congrArg (fun F => ModuleCat.Hom.hom F c) (HomologicalComplex.homologyπ_naturality (corCx A H τ) 0)
  rw [h1, π_comp_H0Iso_hom_apply, π_comp_H0Iso_hom_apply]
  have e1 : (((cocyclesIso₀ A).hom ((HomologicalComplex.cyclesMap (corCx A H τ) 0).hom c) : A.ρ.invariants) : A) =
      (cochainsIso₀ A).hom (((inhomogeneousCochains A).iCycles 0).hom
        ((HomologicalComplex.cyclesMap (corCx A H τ) 0).hom c)) :=
    cocyclesIso₀_hom_comp_f_apply A ((HomologicalComplex.cyclesMap (corCx A H τ) 0).hom c)
  have e2 : (((cocyclesIso₀ (Rep.res H.subtype A)).hom c : (Rep.res H.subtype A).ρ.invariants) : A) =
      (cochainsIso₀ (Rep.res H.subtype A)).hom (((inhomogeneousCochains _).iCycles 0).hom c) :=
    cocyclesIso₀_hom_comp_f_apply (Rep.res H.subtype A) c
  rw [e1, e2]
  have e3 : ((inhomogeneousCochains A).iCycles 0).hom ((HomologicalComplex.cyclesMap (corCx A H τ) 0).hom c) =
      ((corCx A H τ).f 0).hom (((inhomogeneousCochains _).iCycles 0).hom c) := by
    rw [← ModuleCat.comp_apply, HomologicalComplex.cyclesMap_i, ModuleCat.comp_apply]
  rw [e3]
  show corFin A τ 0 _ _ = _
  rw [corFin_apply]
  refine Finset.sum_congr rfl fun q _ => ?_
  congr 2
  exact congrArg (((inhomogeneousCochains (Rep.res H.subtype A)).iCycles 0).hom c) (Subsingleton.elim _ _)

end Cor

/-! ### T2: the classical operator on `H^n(SL(2, ℤ); V_L)` and its transport to the model -/

section Model

open scoped MatrixGroups
open GL2.Model GL2.Model.XBulk GL2.Cone GL2.M4

variable {R : Type} [CommRing R] {V : Type} [AddCommGroup V] [Module R V] (ρ : ΔQ →* Module.End R V)

/-- `Γ'' = Spp g` has finite index in `SL(2, ℤ)` (`Γpp_finiteIndex` through `eQ`). -/
instance Spp_finiteIndex (g : GLQ) : (Spp g).FiniteIndex := by
  have := Γpp_finiteIndex g
  refine ⟨?_⟩
  rw [Spp, show (eQ.toMonoidHom : ΓQ →* SL(2, ℤ)) = (eQ : ΓQ →* SL(2, ℤ)) from rfl, Subgroup.index_map_equiv]
  exact Subgroup.FiniteIndex.index_ne_zero

/-- A (left-coset) transversal of `Γ''` in `SL(2, ℤ)`, `σ(Γ'') = 1`. -/
def τS (g : GLQ) : Transversal (Spp g) := Classical.choice Transversal.nonempty

/-- `cor ∘ (c_g)_* ∘ res` on `H^n(SL(2, ℤ); V_L)` with the corestriction of the transversal `τ` of `Γ''`. -/
def TclsT (g : GLQ) (hg : g ∈ ΔQ) (τ : Transversal (Spp g)) (n : ℕ) : Module.End R (groupCohomology (VL ρ) n) :=
  corH (VL ρ) (Spp g) τ n ∘ₗ conjMap ρ g hg n ∘ₗ resTo ρ (Spp g⁻¹) n

/-- **The classical operator** `Tcls g n = cor ∘ (c_g)_* ∘ res` on `H^n(SL(2, ℤ); V_L)`, with the genuine corestriction
`corH` (no hypothesised `cor`), every `n`, at the chosen transversal `τS g`; independent of the transversal in every degree
(slot 23, `GL2.M6.TclsT_indep`). -/
def Tcls (g : GLQ) (hg : g ∈ ΔQ) (n : ℕ) : Module.End R (groupCohomology (VL ρ) n) := TclsT ρ g hg (τS g) n

/-- `Ψ` on cohomology is an isomorphism (`psi_phi`, `phi_psi`). -/
lemma HmapOf_psi_phi (n : ℕ) :
    HmapOf (psi model7 ρ eQ (hΓΔ := ΓQ_le_ΔQ)) (psi_isChainMap model7 ρ eQ) n ∘ₗ
      HmapOf (phi model7 ρ eQ (hΓΔ := ΓQ_le_ΔQ)) (phi_isChainMap model7 ρ eQ) n = LinearMap.id := by
  apply LinearMap.ext; intro x
  induction x using Submodule.Quotient.induction_on with
  | _ z =>
    rw [LinearMap.comp_apply, HmapOf_mk, HmapOf_mk, LinearMap.id_apply]
    congr 1
    exact Subtype.ext (LinearMap.congr_fun (psi_phi model7 ρ eQ (n : ℤ)) z.1)

lemma HmapOf_phi_psi (n : ℕ) :
    HmapOf (phi model7 ρ eQ (hΓΔ := ΓQ_le_ΔQ)) (phi_isChainMap model7 ρ eQ) n ∘ₗ
      HmapOf (psi model7 ρ eQ (hΓΔ := ΓQ_le_ΔQ)) (psi_isChainMap model7 ρ eQ) n = LinearMap.id := by
  apply LinearMap.ext; intro x
  induction x using Submodule.Quotient.induction_on with
  | _ z =>
    rw [LinearMap.comp_apply, HmapOf_mk, HmapOf_mk, LinearMap.id_apply]
    congr 1
    exact Subtype.ext (LinearMap.congr_fun (phi_psi model7 ρ eQ (n : ℤ)) z.1)

/-- **`Θ` as an isomorphism** `H^n(model) ≅ H^n(SL(2, ℤ); V_L)` (`6 ∈ R^×`): `Ψ`, then `model7_rmkOrbifoldL`. -/
def ΘE (h6 : IsUnit (6 : R)) (n : ℕ) : HM ρ n ≃ₗ[R] groupCohomology (VL ρ) n :=
  (LinearEquiv.ofLinearMap (re₁₂ := RingHomInvPair.ids) (re₂₁ := RingHomInvPair.ids)
    (HmapOf (psi model7 ρ eQ (hΓΔ := ΓQ_le_ΔQ)) (psi_isChainMap model7 ρ eQ) n)
    (HmapOf (phi model7 ρ eQ (hΓΔ := ΓQ_le_ΔQ)) (phi_isChainMap model7 ρ eQ) n)
    (HmapOf_psi_phi ρ n) (HmapOf_phi_psi ρ n)).trans (re₁₂ := RingHomInvPair.ids) (re₂₁ := RingHomInvPair.ids)
    (re₂₃ := RingHomInvPair.ids) (re₃₂ := RingHomInvPair.ids)
    (model7_rmkOrbifoldL ΓQ_le_ΔQ ρ eQ h6 n)

lemma ΘE_toLinearMap (h6 : IsUnit (6 : R)) (n : ℕ) : (ΘE ρ h6 n).toLinearMap = Θ ρ h6 n :=
  LinearMap.ext fun _ => rfl

/-- **The transport to the model**: `TclsM g n = Θ⁻¹ ∘ Tcls g n ∘ Θ ∈ End_R(H^n)`. -/
def TclsM (h6 : IsUnit (6 : R)) (g : GLQ) (hg : g ∈ ΔQ) (n : ℕ) : Module.End R (HM ρ n) :=
  (ΘE ρ h6 n).symm.toLinearMap ∘ₗ Tcls ρ g hg n ∘ₗ (ΘE ρ h6 n).toLinearMap

/-- `Θ ∘ TclsM = Tcls ∘ Θ`. -/
theorem Θ_TclsM (h6 : IsUnit (6 : R)) (g : GLQ) (hg : g ∈ ΔQ) (n : ℕ) :
    Θ ρ h6 n ∘ₗ TclsM ρ h6 g hg n = Tcls ρ g hg n ∘ₗ Θ ρ h6 n := by
  rw [← ΘE_toLinearMap]
  apply LinearMap.ext; intro x
  simp [TclsM]

/-! ### T3: in degree 1, `Tcls` is FLT's `heckeH1` -/

/-- `ρ(g)` is a twist for `x ↦ g⁻¹ x g : Γ'' → Γ'` (FLT's `IsTwist`): `ρ(g) ρ(g⁻¹ s g) = ρ(s) ρ(g)`. -/
lemma isTwist (g : GLQ) (hg : g ∈ ΔQ) :
    HeckeCohomology.IsTwist (Spp g⁻¹) (Spp g) (GL2.M4.conjHom g) (VL ρ) (ρ ⟨g, hg⟩) := fun s a =>
  Rep.hom_comm_apply (conjCoeff ρ g hg) s a

attribute [local instance] fintypeQuot

/-- **T3: FLT's `heckeH1` is the degree-1 member**, for **every** transversal `τ` of `Γ''`:
`cor_τ ∘ (c_g)_* ∘ res = heckeH1 Γ' Γ'' (c_g) V_L ρ(g)` on `H^1`. -/
theorem TclsT_one (g : GLQ) (hg : g ∈ ΔQ) (τ : Transversal (Spp g)) :
    TclsT ρ g hg τ 1 =
      HeckeCohomology.heckeH1 (Spp g⁻¹) (Spp g) (GL2.M4.conjHom g) (VL ρ) (ρ ⟨g, hg⟩) (isTwist ρ g hg) := by
  apply LinearMap.ext; intro x
  induction x using H1_induction_on with
  | h f =>
    set E := QuotientGroup.quotientRightRelEquivQuotientLeftRel (Spp g) with hE
    -- the cocycle `u = (c_g)_* res f` on `Γ''`
    let u : cocycles₁ (Rep.res (Spp g).subtype (VL ρ)) :=
      mapCocycles₁ (GL2.M4.conjHom g) (conjCoeff ρ g hg) (mapCocycles₁ (Spp g⁻¹).subtype (𝟙 _) f)
    have hT : TclsT ρ g hg τ 1 (H1π _ f) = H1π _ (cor₁Z (VL ρ) (Spp g) τ u) := by
      simp only [TclsT, LinearMap.comp_apply, resTo, conjMap]
      rw [H1π_comp_map_apply, H1π_comp_map_apply]
      exact corH_H1π (VL ρ) (Spp g) τ u
    rw [hT]
    refine HeckeCohomology.heckeH1_eq_of_section (Spp g⁻¹) (Spp g) (GL2.M4.conjHom g) (VL ρ) (ρ ⟨g, hg⟩) (isTwist ρ g hg)
      (fun Q => (τ.σ (E Q))⁻¹) (fun Q γ => τ.lam ((τ.σ (E Q))⁻¹ * γ)) ?_ f _ ?_
    · intro Q γ
      rw [τ.coe_lam, inv_inv]
      congr 2
    · intro γ
      show cor₁ (VL ρ) τ u γ = _
      rw [cor₁_apply]
      refine (Fintype.sum_equiv E _ _ fun Q => ?_).symm
      rw [inv_inv]
      rfl

/-- **T3 in degree 0: FLT's `heckeInv` is the degree-0 member**: through `H^0 ≅` invariants,
`cor_τ ∘ (c_g)_* ∘ res` is `heckeInv Γ' Γ'' (c_g) V_L ρ(g)` (`a ↦ ∑ (r_Q)⁻¹ ρ(g) a` over FLT's right cosets). -/
theorem TclsT_zero (g : GLQ) (hg : g ∈ ΔQ) (τ : Transversal (Spp g)) (z : (VL ρ).ρ.invariants) :
    (H0Iso (VL ρ)).hom (TclsT ρ g hg τ 0 ((H0Iso (VL ρ)).inv z)) =
      HeckeCohomology.heckeInv (Spp g⁻¹) (Spp g) (GL2.M4.conjHom g) (VL ρ) (ρ ⟨g, hg⟩) (isTwist ρ g hg) z := by
  apply Subtype.ext
  rw [HeckeCohomology.coe_heckeInv, HeckeCohomology.norm0_apply]
  simp only [TclsT, LinearMap.comp_apply]
  rw [corH_zero]
  have hres : (((H0Iso _).hom (resTo ρ (Spp g⁻¹) 0 ((H0Iso (VL ρ)).inv z)) :
      (Rep.res (Spp g⁻¹).subtype (VL ρ)).ρ.invariants) : V) = (z : V) := by
    have := map_H0Iso_hom_f_apply (Spp g⁻¹).subtype (𝟙 (Rep.res (Spp g⁻¹).subtype (VL ρ)))
      ((H0Iso (VL ρ)).inv z)
    rw [Iso.inv_hom_id_apply] at this
    exact this
  have hcj : ∀ y, (((H0Iso _).hom (conjMap ρ g hg 0 y) : (Rep.res (Spp g).subtype (VL ρ)).ρ.invariants) : V) =
      ρ ⟨g, hg⟩ (((H0Iso _).hom y : (Rep.res (Spp g⁻¹).subtype (VL ρ)).ρ.invariants) : V) := fun y =>
    map_H0Iso_hom_f_apply (GL2.M4.conjHom g) (conjCoeff ρ g hg) y
  rw [hcj, hres]
  set E := QuotientGroup.quotientRightRelEquivQuotientLeftRel (Spp g)
  refine (Fintype.sum_equiv E (fun Q => (VL ρ).ρ (HeckeCohomology.rep (Spp g) Q)⁻¹ (ρ ⟨g, hg⟩ (z : V)))
    (fun q => (VL ρ).ρ (τ.σ q) (ρ ⟨g, hg⟩ (z : V))) fun Q => ?_).symm
  -- `(r_Q)⁻¹ = σ(E Q) h` with `h ∈ Γ''`, and `ρ(g) z` is `Γ''`-invariant
  have hmem : (τ.σ (E Q))⁻¹ * (HeckeCohomology.rep (Spp g) Q)⁻¹ ∈ Spp g := by
    have h1 : ((τ.σ (E Q) : SL(2, ℤ)) : SL(2, ℤ) ⧸ Spp g) = (((HeckeCohomology.rep (Spp g) Q)⁻¹ : SL(2, ℤ)) : SL(2, ℤ) ⧸ Spp g) := by
      rw [τ.mk_σ]
      conv_lhs => rw [show Q = HeckeCohomology.cls (Spp g) (HeckeCohomology.rep (Spp g) Q) by
        simp [HeckeCohomology.rep, HeckeCohomology.cls]]
      rfl
    exact QuotientGroup.eq.1 h1
  have hinv : ∀ s : Spp g, (VL ρ).ρ (s : SL(2, ℤ)) (ρ ⟨g, hg⟩ (z : V)) = ρ ⟨g, hg⟩ (z : V) := fun s => by
    rw [← isTwist ρ g hg s (z : V), z.2]
  have hsplit : (HeckeCohomology.rep (Spp g) Q)⁻¹ = τ.σ (E Q) * ((τ.σ (E Q))⁻¹ * (HeckeCohomology.rep (Spp g) Q)⁻¹) := by
    group
  rw [hsplit, map_mul, Module.End.mul_apply]
  congr 1
  exact hinv ⟨_, hmem⟩

/-- `Tcls g 1` is FLT's `heckeH1`. -/
theorem Tcls_one (g : GLQ) (hg : g ∈ ΔQ) :
    Tcls ρ g hg 1 = HeckeCohomology.heckeH1 (Spp g⁻¹) (Spp g) (GL2.M4.conjHom g) (VL ρ) (ρ ⟨g, hg⟩) (isTwist ρ g hg) :=
  TclsT_one ρ g hg (τS g)

/-- In degree 1 the classical operator does not depend on the transversal. -/
theorem TclsT_one_indep (g : GLQ) (hg : g ∈ ΔQ) (τ τ' : Transversal (Spp g)) :
    TclsT ρ g hg τ 1 = TclsT ρ g hg τ' 1 := by
  rw [TclsT_one, TclsT_one]

/-- **The classical operator commutes with `δ : H^0 → H^1`** (the first `δ`-step on the classical side; FLT's
`heckeH1_delta0` through `TclsT_zero`, `TclsT_one`): for a short exact sequence `V_{L,1} → V_{L,2} → V_{L,3}` of the
`SL(2, ℤ)`-modules of three `Δ`-modules whose maps commute with `ρ(g)`, `Tcls g 1 ∘ δ = δ ∘ Tcls g 0` on `H^0(V_{L,3})`. -/
theorem TclsT_delta0 {V₁ V₂ V₃ : Type} [AddCommGroup V₁] [Module R V₁] [AddCommGroup V₂] [Module R V₂]
    [AddCommGroup V₃] [Module R V₃] (ρ₁ : ΔQ →* Module.End R V₁) (ρ₂ : ΔQ →* Module.End R V₂)
    (ρ₃ : ΔQ →* Module.End R V₃) (f : VL ρ₁ ⟶ VL ρ₂) (f' : VL ρ₂ ⟶ VL ρ₃) (w : f ≫ f' = 0)
    (hX : (ShortComplex.mk f f' w).ShortExact) (g : GLQ) (hg : g ∈ ΔQ)
    (hf : ∀ a, f.hom (ρ₁ ⟨g, hg⟩ a) = ρ₂ ⟨g, hg⟩ (f.hom a))
    (hf' : ∀ b, f'.hom (ρ₂ ⟨g, hg⟩ b) = ρ₃ ⟨g, hg⟩ (f'.hom b))
    (τ : Transversal (Spp g)) (x : groupCohomology (VL ρ₃) 0) :
    TclsT ρ₁ g hg τ 1 (δ hX 0 1 rfl x) = δ hX 0 1 rfl (TclsT ρ₃ g hg τ 0 x) := by
  induction x using H0_induction_on with
  | h z =>
    rw [TclsT_one, ← Iso.hom_inv_id_apply (H0Iso (VL ρ₃)) (TclsT ρ₃ g hg τ 0 ((H0Iso (VL ρ₃)).inv z)), TclsT_zero]
    exact HeckeCohomology.heckeH1_delta0 (Spp g⁻¹) (Spp g) (GL2.M4.conjHom g) hX (ρ₁ ⟨g, hg⟩) (isTwist ρ₁ g hg)
      (ρ₂ ⟨g, hg⟩) (isTwist ρ₂ g hg) (ρ₃ ⟨g, hg⟩) (isTwist ρ₃ g hg) hf hf' z

end Model

end GL2.M5

end
