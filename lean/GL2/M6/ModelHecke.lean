/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.M6.Classical
import GL2.M6.HomologyLink

/-!
# M6-A2 (slot 23): [C] `thm:classical` on the seven-vertex model, for `Θ_can`

[GT] `gt_part_M6.tex`, rows `def:m6-thetacan`, `lem:m6-model-cochain`, `thm:m6-classical`, `thm:m6-classical-model`.
Brief `M6B_brief.md` §2.1 (A2), decision D4 (`Θ_can`).

* `ΘcanE` (**`Θ_can`**, `6 ∈ R^×`): `H^n` of the model `≅ H^n(SL₂(ℤ); V_L)`: `Ψ`, the hand-built-to-mathlib identification
  (`homologyEquiv`), `Hom_Γ(C_•(X_bulk), -) = Hom_Γ(R[Simp_•], -)` (`homEqv_iso`), then the **canonical** comparison
  `canIso` of the model resolution with the bar resolution (any lift of `id`). It differs from M4's `Θ` only in the last
  step, where M4 uses mathlib's `Ext`-built `groupCohomologyIso`; the two are equal (`GL2.M6.ThetaExt`, A-Θ).
* `Ξ`: on cochains, `Θ_can` is `φ ↦ (x ↦ φ_R(x))` (`Ξ_apply`), and **`Ξ_hecke` (cochain level)**: the model's
  `T̃_g φ = ∑_i ρ(δ_i) φ_R(τ(s_i ·))` goes to `heckeY` with `τ` (transported to `Γ'' ≤ SL₂(ℤ)`, `tauS`) and `θ = ρ(g)`,
  the coset representatives `s_i⁻¹` replacing the transversal (`corHom_reps`, `cdEquiv`).
* **`classical_can` ([C] `thm:classical` in [C]'s sense, for `Θ_can`)**: for every `g ∈ Δ`, every coset datum, every `τ`
  over `R` and every `n`, `Θ_can ∘ T_{ΓgΓ} = Tcls g n ∘ Θ_can`, with M5's `Tcls = cor ∘ (c_g)_* ∘ res` (the genuine
  corestriction). No literature hypothesis.
* **`thm_classical_can`**: [GT] `thm:classical-model`'s statement with `Θ_can` in place of `Θ`, proved outright with
  `cor = corH` (`cor ∘ res = [Γ : Γ'']` by `corH_res`): the `hcl` hypothesis is not needed for it.
-/

noncomputable section

namespace GL2.M6

open CategoryTheory groupCohomology GL2.MP groupCohomology.Cores

section Reps

variable {G : Type} [Group G] {k : Type} [CommRing k] (K : Subgroup G) [K.FiniteIndex] (σ : Transversal K)

attribute [local instance] groupCohomology.Cores.fintypeQuot

/-- **The transfer with any coset representatives** `r_q ∈ q`. -/
lemma corHom_reps {M A : Rep.{0} k G} (χ : Rep.res K.subtype M ⟶ Rep.res K.subtype A) (r : G ⧸ K → G)
    (hr : ∀ q, ((r q : G) : G ⧸ K) = q) (x : M) :
    (corHom K σ χ).hom x = ∑ q : G ⧸ K, A.ρ (r q) (χ.hom (M.ρ (r q)⁻¹ x)) := by
  rw [corHom_hom_apply]
  refine Finset.sum_congr rfl fun q _ => ?_
  have hmem : (r q)⁻¹ * σ.σ q ∈ K := by
    rw [← QuotientGroup.eq, σ.mk_σ, hr q]
  set h : K := ⟨_, hmem⟩
  have e : σ.σ q = r q * (h : G) := by simp [h]
  have hc : ∀ y : M, χ.hom (M.ρ (h : G)⁻¹ y) = A.ρ (h : G)⁻¹ (χ.hom y) := fun y => resHom_comm K χ h⁻¹ y
  rw [e, mul_inv_rev, map_mul M.ρ, Module.End.mul_apply, hc, map_mul A.ρ, Module.End.mul_apply,
    ← Module.End.mul_apply (A.ρ (h : G)), ← map_mul, mul_inv_cancel, map_one, Module.End.one_apply]

end Reps

section Model

open scoped MatrixGroups
open GL2.Model GL2.Model.XBulk GL2.Cone GL2.Cone.ProjRed GL2.Cosets GL2.M4 GL2.M5 MonoidAlgebra

variable {R : Type} [CommRing R] {V : Type} [AddCommGroup V] [Module R V] (ρ : ΔQ →* Module.End R V)

attribute [local instance] groupCohomology.Cores.fintypeQuot

/-! ### `Γ''` in `ΓQ` and in `SL₂(ℤ)` -/

/-- `Spp g → Γ''(g) ≤ ΓQ`, `x ↦ e⁻¹ x`. -/
def ιpp (g : GLQ) : Spp g →* Γpp (Γ := ΓQ) g where
  toFun x := ⟨eQ.symm x.1, mem_Spp.1 x.2⟩
  map_one' := by apply Subtype.ext; simp
  map_mul' a b := by apply Subtype.ext; simp

lemma fIncl_ιpp (g : GLQ) (x : Spp g) : fIncl g eQ (ιpp g x) = (Spp g).subtype x := by
  simp [fIncl, ιpp]

lemma fConj_ιpp (g : GLQ) (x : Spp g) : fConj g eQ (ιpp g x) = fConjS g x := rfl

/-- **The twisted chain datum on `Γ'' ≤ SL₂(ℤ)`**: a `TauR` (over `Γ''(g) ≤ ΓQ`) as a chain map
`res_{Γ''} R[Simp_•] → res_{h ↦ g⁻¹hg} R[Simp_•]` of `Spp g`-representations (the same linear maps). -/
def tauS {g : GLQ} (τ : TauR R g eQ) :
    resCx (Spp g).subtype (repCx model7 R) ⟶ resCx (fConjS g) (repCx model7 R) where
  f n := Rep.ofHom (LinearMap.intertwiningMap_of_isIntertwiningMap _ _ (τ.f n).hom.toLinearMap fun h x => by
    have := τ.twist n (ιpp g h) x
    rw [fIncl_ιpp, fConj_ιpp] at this
    exact this)
  comm' i j (hij : j + 1 = i) := by
    subst hij
    rw [Functor.mapHomologicalComplex_obj_d, Functor.mapHomologicalComplex_obj_d, repCx_d]
    apply Rep.hom_ext; apply Representation.IntertwiningMap.ext; refine LinearMap.ext fun x => ?_
    exact τ.t2 j x

lemma tauS_f_apply {g : GLQ} (τ : TauR R g eQ) (n : ℕ) (x : R[Simp model7 (n : ℤ)]) :
    ((tauS τ).f n).hom x = (τ.f n).hom x := rfl

lemma tauS_π {g : GLQ} (τ : TauR R g eQ) (h6 : IsUnit (6 : R)) (x : (repCx model7 R).X 0) :
    ((model7_projRes h6).π.f 0).hom (((tauS τ).f 0).hom x) = ((model7_projRes h6).π.f 0).hom x := by
  have e : (model7_projRes h6).π.f 0 = repEps model7 R := repAug_f_zero model7 R
  rw [e]
  exact τ.t3 x

/-! ### The coset representatives -/

lemma memΔ_of_cd {g : GLQ} (cd : CosetData ΓQ ΔQ g) : g ∈ ΔQ := by
  obtain ⟨i, -⟩ := (cd.cosets 1).exists
  have h := ΔQ.mul_mem (ΓQ_le_ΔQ (cd.s i)) (cd.δ_mem i)
  rwa [mul_inv_cancel_left] at h

/-- `i ↦ s_i⁻¹ Γ''`, a bijection `Fin m ≃ SL₂(ℤ) / Γ''`. -/
lemma cd_bij {g : GLQ} (cd : CosetData ΓQ ΔQ g) :
    Function.Bijective (fun i : Fin cd.m => (((eQ (cd.s i))⁻¹ : SL(2, ℤ)) : SL(2, ℤ) ⧸ Spp g)) := by
  constructor
  · intro i j hij
    simp only at hij
    rw [QuotientGroup.eq, inv_inv] at hij
    rw [mem_Spp] at hij
    simp only [map_mul, map_inv, MulEquiv.symm_apply_apply, Subgroup.coe_mul, Subgroup.coe_inv] at hij
    obtain ⟨i', -, hu⟩ := cd.cosets (cd.s i)
    have hi : ∃ h : ΓQ, g⁻¹ * (h : GLQ) * g ∈ ΓQ ∧ ((cd.s i : ΓQ) : GLQ) = h * cd.s i :=
      ⟨1, by simp, by simp⟩
    have hj : ∃ h : ΓQ, g⁻¹ * (h : GLQ) * g ∈ ΓQ ∧ ((cd.s i : ΓQ) : GLQ) = h * cd.s j :=
      ⟨cd.s i * (cd.s j)⁻¹, by simpa using hij, by simp⟩
    exact (hu i hi).trans (hu j hj).symm
  · intro q
    induction q using QuotientGroup.induction_on with
    | H t =>
      obtain ⟨i, ⟨h, hh, he⟩, -⟩ := cd.cosets (eQ.symm t)⁻¹
      refine ⟨i, ?_⟩
      simp only
      rw [QuotientGroup.eq, inv_inv, mem_Spp]
      have he' : ((eQ.symm t : ΓQ) : GLQ) = ((cd.s i : ΓQ) : GLQ)⁻¹ * ((h : ΓQ) : GLQ)⁻¹ := by
        have := congrArg (fun x : GLQ => x⁻¹) he
        simpa [mul_inv_rev] using this
      simp only [map_mul, MulEquiv.symm_apply_apply, Subgroup.coe_mul]
      rw [he']
      have : g⁻¹ * (((cd.s i : ΓQ) : GLQ) * (((cd.s i : ΓQ) : GLQ)⁻¹ * ((h : ΓQ) : GLQ)⁻¹)) * g =
          (g⁻¹ * (h : GLQ) * g)⁻¹ := by group
      rw [this]
      exact ΓQ.inv_mem hh

/-- The bijection of `cd_bij`. -/
def cdEquiv {g : GLQ} (cd : CosetData ΓQ ΔQ g) : Fin cd.m ≃ SL(2, ℤ) ⧸ Spp g :=
  Equiv.ofBijective _ (cd_bij cd)

/-! ### `Θ_can` on cochains -/

/-- `Θ_can` on cochains: `Ψ`, then `Hom_Γ(C_•, -) = Hom_Γ(R[Simp_•], -)`. -/
def Ξ (n : ℕ) : (EqC model7 (hΓΔ := ΓQ_le_ΔQ) ρ eQ).X n →ₗ[R] ((repCx model7 R).linearYonedaObj R (VL ρ)).X n :=
  (relHomEqv model7 (VL ρ) n).toLinearMap ∘ₗ psi model7 ρ eQ (hΓΔ := ΓQ_le_ΔQ) n

/-- **`Θ_can` on cochains is `φ ↦ φ_R`.** -/
lemma Ξ_apply (n : ℕ) (φ : (EqC model7 (hΓΔ := ΓQ_le_ΔQ) ρ eQ).X n) (x : R[Simp model7 (n : ℤ)]) :
    (Ξ ρ n φ).hom x = linR (R := R) φ.1 x := by
  have hmem : φ.1 ∈ eqHom (⊤ : Subgroup SL(2, ℤ)) ⊤ hTop (ρA (VL ρ)) (Yc model7 Subgroup.topEquiv) n :=
    (mem_eqHom_iff model7 (ρA (VL ρ)) Subgroup.topEquiv φ.1).2 (eqC_isEqv model7 ρ eQ φ)
  have hp := LinearMap.congr_fun (phi_psi model7 (ρA (VL ρ)) Subgroup.topEquiv (hΓΔ := hTop) n) ⟨φ.1, hmem⟩
  have e : Ξ ρ n φ = homEqv model7 (VL ρ) n ⟨φ.1, hmem⟩ := by
    show homEqv model7 (VL ρ) n ((phi model7 (ρA (VL ρ)) Subgroup.topEquiv (hΓΔ := hTop) n)
      (psi model7 ρ eQ (hΓΔ := ΓQ_le_ΔQ) n φ)) = _
    exact congrArg (homEqv model7 (VL ρ) n) hp
  rw [e]
  rfl

/-- `φ_R` on a single generator. -/
lemma linR_single {n : ℤ} (ψ : chains model7 n →+ V) (s : Simp model7 n) (r : R) :
    linR (R := R) ψ (MonoidAlgebra.single s r) = r • ψ (orient model7 n s) := by
  simp [linR]

lemma embR_orient (n : ℤ) (s : Simp model7 n) : embR (R := R) n (orient model7 n s) = MonoidAlgebra.single s 1 := by
  simp [embR, ofChains_orient, castF_single]

/-- **The model's Hecke operator on `R[Simp_n]`**: `φ_R ∘ T̃ = ∑_i ρ(δ_i) φ_R(τ(s_i ·))`, linearly in `x`. -/
lemma linR_heckeTildeR {g : GLQ} (cd : CosetData ΓQ ΔQ g) (τ : TauR R g eQ) (n : ℕ)
    (φ : (EqC model7 (hΓΔ := ΓQ_le_ΔQ) ρ eQ).X n) (x : R[Simp model7 (n : ℤ)]) :
    linR (R := R) (heckeTildeR ρ eQ cd τ n φ).1 x =
      ∑ i, ρ ⟨((cd.s i : ΓQ) : GLQ)⁻¹ * g, cd.δ_mem i⟩
        (linR (R := R) φ.1 ((τ.f n).hom (Representation.ofMulAction R SL(2, ℤ) (Simp model7 n) (eQ (cd.s i)) x))) := by
  induction x using MonoidAlgebra.induction_linear with
  | zero => simp
  | add a b ha hb => simp only [map_add, ha, hb, Finset.sum_add_distrib]
  | single s r =>
    erw [linR_single, heckeTildeR_apply, Finset.smul_sum]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [← map_smul]
    congr 1
    show r • linR (R := R) φ.1 ((τ.f n).hom (embR (R := R) (n : ℤ) (eQ (cd.s i) • orient model7 n s))) = _
    rw [embR_smul, embR_orient, ← map_smul, ← map_smul, ← map_smul, MonoidAlgebra.smul_single, smul_eq_mul, mul_one]

/-- **`lem:m6-model-cochain` (A2 at cochain level)**: `Θ_can ∘ T̃_g = heckeY ∘ Θ_can` on cochains, with `τ` transported
(`tauS`), `θ = ρ(g)` and any transversal. -/
theorem Ξ_hecke {g : GLQ} (cd : CosetData ΓQ ΔQ g) (τ : TauR R g eQ) (σ : Transversal (Spp g)) (h6 : IsUnit (6 : R))
    (n : ℕ) (φ : (EqC model7 (hΓΔ := ΓQ_le_ΔQ) ρ eQ).X n) :
    Ξ ρ n (heckeTildeR ρ eQ cd τ n φ) =
      ((heckeY (Spp g) σ (fConjS g) (thetaS ρ g (memΔ_of_cd cd)) (model7_projRes h6) (tauS τ)).f n) (Ξ ρ n φ) := by
  apply Rep.hom_ext; apply Representation.IntertwiningMap.ext; refine LinearMap.ext fun x => ?_
  show (Ξ ρ n (heckeTildeR ρ eQ cd τ n φ)).hom x = (corHom (Spp g) σ _).hom x
  erw [Ξ_apply, linR_heckeTildeR,
    corHom_reps (Spp g) σ _ (fun q => (eQ (cd.s ((cdEquiv cd).symm q)))⁻¹) (fun q => by
      have := (cdEquiv cd).apply_symm_apply q
      exact this)]
  refine (Fintype.sum_equiv (cdEquiv cd) _ _ fun i => ?_)
  rw [show (cdEquiv cd).symm ((cdEquiv cd) i) = i from (cdEquiv cd).symm_apply_apply i, inv_inv]
  show _ = (VL ρ).ρ (eQ (cd.s i))⁻¹ (ρ ⟨g, memΔ_of_cd cd⟩
    ((Ξ ρ n φ).hom ((τ.f n).hom (Representation.ofMulAction R SL(2, ℤ) (Simp model7 n) (eQ (cd.s i)) x))))
  rw [Ξ_apply]
  have hV : (VL ρ).ρ (eQ (cd.s i))⁻¹ = ρ ⟨((cd.s i : ΓQ) : GLQ)⁻¹, ΓQ_le_ΔQ (cd.s i)⁻¹⟩ := by
    show ρ ⟨((eQ.symm (eQ (cd.s i))⁻¹ : ΓQ) : GLQ), _⟩ = _
    congr 2
    simp
  rw [hV, ← Module.End.mul_apply, ← map_mul]
  rfl

/-! ### `Θ_can` and [C] `thm:classical` -/

/-- **`def:m6-thetacan`: `Θ_can`** as an isomorphism `H^n(model) ≅ H^n(SL₂(ℤ); V_L)` (`6 ∈ R^×`). -/
def ΘcanE (h6 : IsUnit (6 : R)) (n : ℕ) : HM ρ n ≃ₗ[R] groupCohomology (VL ρ) n :=
  (LinearEquiv.ofLinearMap (re₁₂ := RingHomInvPair.ids) (re₂₁ := RingHomInvPair.ids)
    (HmapOf (psi model7 ρ eQ (hΓΔ := ΓQ_le_ΔQ)) (psi_isChainMap model7 ρ eQ) n)
    (HmapOf (phi model7 ρ eQ (hΓΔ := ΓQ_le_ΔQ)) (phi_isChainMap model7 ρ eQ) n)
    (HmapOf_psi_phi ρ n) (HmapOf_phi_psi ρ n)).trans
    (((relCxA model7 (VL ρ)).homologyEquiv (relCxA_subsingleton model7 (VL ρ)) n).trans
      ((HomologicalComplex.homologyMapIso (homEqv_iso model7 (VL ρ)) n ≪≫
        (canIso (model7_projRes h6) (VL ρ) n).symm).toLinearEquiv))

/-- `Θ_can` as a linear map. -/
def Θcan (h6 : IsUnit (6 : R)) (n : ℕ) : HM ρ n →ₗ[R] groupCohomology (VL ρ) n := (ΘcanE ρ h6 n).toLinearMap

lemma Θcan_apply (h6 : IsUnit (6 : R)) (n : ℕ) (x : HM ρ n) :
    Θcan ρ h6 n x = (canIso (model7_projRes h6) (VL ρ) n).inv.hom
      ((HomologicalComplex.homologyMap (homEqv_iso model7 (VL ρ)).hom n).hom
        ((relCxA model7 (VL ρ)).homologyEquiv (relCxA_subsingleton model7 (VL ρ)) n
          (HmapOf (psi model7 ρ eQ (hΓΔ := ΓQ_le_ΔQ)) (psi_isChainMap model7 ρ eQ) n x))) := rfl

/-- `Θ_can` on the class of a cocycle `w`: `canIso⁻¹` of `homologyπ` of a cycle with underlying cochain `Ξ w`. -/
lemma Θcan_mk (h6 : IsUnit (6 : R)) (n : ℕ) (w : (EqC model7 (hΓΔ := ΓQ_le_ΔQ) ρ eQ).Z n) :
    ∃ c : ((repCx model7 R).linearYonedaObj R (VL ρ)).cycles n,
      (((repCx model7 R).linearYonedaObj R (VL ρ)).iCycles n).hom c = Ξ ρ n w.1 ∧
        Θcan ρ h6 n (Submodule.Quotient.mk w) =
          (canIso (model7_projRes h6) (VL ρ) n).inv.hom
            ((((repCx model7 R).linearYonedaObj R (VL ρ)).homologyπ n).hom c) := by
  obtain ⟨c₀, hc₀, e₀⟩ := (relCxA model7 (VL ρ)).homologyEquiv_mk (relCxA_subsingleton model7 (VL ρ)) n
    ⟨psi model7 ρ eQ (hΓΔ := ΓQ_le_ΔQ) n w.1, (psi_isChainMap model7 ρ eQ).mem_Z n w⟩
  refine ⟨(HomologicalComplex.cyclesMap (homEqv_iso model7 (VL ρ)).hom n).hom c₀, ?_, ?_⟩
  · have := congrArg (fun F => F.hom c₀) (HomologicalComplex.cyclesMap_i (homEqv_iso model7 (VL ρ)).hom n)
    simp only [ModuleCat.hom_comp, LinearMap.comp_apply] at this
    rw [this, hc₀]
    rfl
  · rw [Θcan_apply, HmapOf_mk]
    erw [e₀]
    congr 1
    have := congrArg (fun F => F.hom c₀)
      (HomologicalComplex.homologyπ_naturality (homEqv_iso model7 (VL ρ)).hom n)
    simp only [ModuleCat.hom_comp, LinearMap.comp_apply] at this
    exact this

/-- `Θ_can` is injective-free bookkeeping: `Tcls ∘ canIso⁻¹ = canIso⁻¹ ∘ H(heckeY)`. -/
lemma Tcls_canIso_inv (h6 : IsUnit (6 : R)) {g : GLQ} (hg : g ∈ ΔQ) (τ : TauR R g eQ) (n : ℕ)
    (y : ((repCx model7 R).linearYonedaObj R (VL ρ)).homology n) :
    Tcls ρ g hg n ((canIso (model7_projRes h6) (VL ρ) n).inv.hom y) =
      (canIso (model7_projRes h6) (VL ρ) n).inv.hom
        ((HomologicalComplex.homologyMap
          (heckeY (Spp g) (τS g) (fConjS g) (thetaS ρ g hg) (model7_projRes h6) (tauS τ)) n).hom y) := by
  have key := LinearMap.congr_fun
    (Tcls_canIso ρ g hg (model7_projRes h6) (tauS τ) (tauS_π τ h6) n) ((canIso (model7_projRes h6) (VL ρ) n).inv.hom y)
  simp only [LinearMap.comp_apply] at key
  have hi : (canIso (model7_projRes h6) (VL ρ) n).hom.hom ((canIso (model7_projRes h6) (VL ρ) n).inv.hom y) = y :=
    Iso.inv_hom_id_apply (canIso (model7_projRes h6) (VL ρ) n) y
  rw [hi] at key
  rw [← key]
  exact (Iso.hom_inv_id_apply (canIso (model7_projRes h6) (VL ρ) n) _).symm

/-- **`thm:m6-classical` (A2): [C] `thm:classical` in [C]'s sense, on the model, for `Θ_can`.** For `6 ∈ R^×`, every
`g ∈ Δ`, every coset datum, every `τ` over `R` and every `n`: `Θ_can ∘ T_{ΓgΓ} = Tcls g n ∘ Θ_can`, where
`Tcls g n = cor ∘ (c_g)_* ∘ res` with the genuine corestriction (M5). -/
theorem classical_can (h6 : IsUnit (6 : R)) {g : GLQ} (cd : CosetData ΓQ ΔQ g) (τ : TauR R g eQ) (n : ℕ) :
    Θcan ρ h6 n ∘ₗ Tdc ρ cd τ n = Tcls ρ g (memΔ_of_cd cd) n ∘ₗ Θcan ρ h6 n := by
  apply LinearMap.ext; intro x
  induction x using Submodule.Quotient.induction_on with
  | _ w =>
    rw [LinearMap.comp_apply, LinearMap.comp_apply]
    obtain ⟨c₂, hc₂, e₂⟩ := Θcan_mk ρ h6 n w
    obtain ⟨c₁, hc₁, e₁⟩ := Θcan_mk ρ h6 n
      ⟨heckeTildeR ρ eQ cd τ n w.1, (heckeTildeR_isChainMap (hΓΔ := ΓQ_le_ΔQ) ρ cd τ).mem_Z n w⟩
    have hT : Tdc ρ cd τ n (Submodule.Quotient.mk w) = Submodule.Quotient.mk
        ⟨heckeTildeR ρ eQ cd τ n w.1, (heckeTildeR_isChainMap (hΓΔ := ΓQ_le_ΔQ) ρ cd τ).mem_Z n w⟩ := rfl
    rw [hT, e₁, e₂]
    erw [Tcls_canIso_inv ρ h6 (memΔ_of_cd cd) τ n]
    congr 1
    set Y := heckeY (Spp g) (τS g) (fConjS g) (thetaS ρ g (memΔ_of_cd cd)) (model7_projRes h6) (tauS τ)
    have hπ := congrArg (fun F => F.hom c₂) (HomologicalComplex.homologyπ_naturality Y n)
    simp only [ModuleCat.hom_comp] at hπ
    have hc : c₁ = (HomologicalComplex.cyclesMap Y n).hom c₂ := by
      apply (ModuleCat.mono_iff_injective (((repCx model7 R).linearYonedaObj R (VL ρ)).iCycles n)).1 inferInstance
      have hi := congrArg (fun F => F.hom c₂) (HomologicalComplex.cyclesMap_i Y n)
      simp only [ModuleCat.hom_comp] at hi
      erw [hi, hc₁]
      exact (Ξ_hecke ρ cd τ (τS g) h6 n w.1).trans (congrArg (Y.f n).hom hc₂.symm)
    exact (congrArg (fun c => ((((repCx model7 R).linearYonedaObj R (VL ρ)).homologyπ n).hom c)) hc).trans hπ.symm

/-- **`thm:m6-classical-model`**: [GT] `thm:classical-model`'s statement with `Θ_can` for `Θ`, **proved outright** (no
`hcl`): `cor = corH` (the genuine corestriction, at any transversal) satisfies `cor ∘ res = [Γ : Γ'']` and
`Θ_can ∘ T_{ΓgΓ} = cor ∘ (c_g)_* ∘ res ∘ Θ_can`, every coset datum, every `τ`, every `n`. -/
theorem thm_classical_can (h6 : IsUnit (6 : R)) {g : GLQ} (cd : CosetData ΓQ ΔQ g) (τ : TauR R g eQ) (n : ℕ) :
    ∃ cor : groupCohomology (Rep.res (Spp g).subtype (VL ρ)) n →ₗ[R] groupCohomology (VL ρ) n,
      cor ∘ₗ resTo ρ (Spp g) n = (Spp g).index • LinearMap.id ∧
      Θcan ρ h6 n ∘ₗ Tdc ρ cd τ n = cor ∘ₗ conjMap ρ g (memΔ_of_cd cd) n ∘ₗ resTo ρ (Spp g⁻¹) n ∘ₗ Θcan ρ h6 n := by
  refine ⟨corH (VL ρ) (Spp g) (τS g) n, corH_res (Spp g) (τS g) (VL ρ) n, ?_⟩
  rw [classical_can ρ h6 cd τ n]
  rfl

/-- **`T_{ΓgΓ}` is `Θ_can⁻¹ ∘ Tcls g n ∘ Θ_can`** (M4's choice-free `TdcOf`; M5's `TclsM` with `Θ_can` for `Θ`). -/
theorem TdcOf_can (h6 : IsUnit (6 : R)) (g : GLQ) (hg : g ∈ ΔQ) (n : ℕ) :
    TdcOf ρ h6 g hg n = (ΘcanE ρ h6 n).symm.toLinearMap ∘ₗ Tcls ρ g hg n ∘ₗ (ΘcanE ρ h6 n).toLinearMap := by
  apply LinearMap.ext; intro x
  apply (ΘcanE ρ h6 n).injective
  have h := LinearMap.congr_fun (classical_can ρ h6 (Classical.choice (cosetData_nonempty g hg))
    (Classical.choice (tauR_nonempty R h6 g)) n) x
  simp only [LinearMap.comp_apply] at h
  simp only [LinearMap.comp_apply, LinearEquiv.coe_coe, LinearEquiv.apply_symm_apply]
  exact h

end Model

end GL2.M6

end
