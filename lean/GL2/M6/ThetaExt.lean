/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.M6.ModelHecke
import GL2.M4.Classical

/-!
# M6 A-Θ (slot 23): mathlib's `Ext`-built comparison is the canonical one; `hcl` is a theorem

[GT] `gt_part_M6.tex`, rows `lem:m6-isoext-lift`, `thm:m6-theta-ext`, `thm:m6-hcl`. Brief `M6B_brief.md` §2 (A-Θ), D4,
`rmk:mp-open`(3), (6).

* `isoExt_hom_lift`: mathlib's `ProjectiveResolution.isoExt` (`H^n(Hom(P, A)) ≅ Ext^n(k, A)`) is natural in the resolution:
  for any chain map `φ : P → Q` over `id_k`, `isoExt P = isoExt Q ≫ H(φ^*)` (mathlib's `isoLeftDerivedObj_inv_naturality`,
  transported through `homologyUnop` by `homologyOp_hom_naturality`).
* **`groupCohomologyIso_eq_canIso`**: mathlib's `groupCohomologyIso A n P` (through `Ext`) **is** `canIso P A n` (through the
  bar comparison), for every projective resolution `P` (`rmk:mp-open`(3) for every `P`).
* **`Θ_eq_Θcan`**: M4's `Θ` (`Ψ`, then `model7_rmkOrbifoldL`, through `Ext`) is `Θ_can`. Hence **`classical_Θ`**: [C]
  `thm:classical` in [C]'s sense for M4's `Θ` itself, every `n` (`Θ ∘ T_{ΓgΓ} = Tcls g n ∘ Θ`; equivalently M5's `TclsM` is
  `TdcOf`, `TclsM_eq_TdcOf`), and **`heckeClassical_holds`**: the literature hypothesis `lit:hecke-classical` (`hcl`,
  `HeckeClassical ρ`) is a **theorem**, so M4's `thm_classical` holds outright (`thm_classical_outright`).
-/

noncomputable section

namespace GL2.M6

open CategoryTheory Opposite GL2.MP

section Unop

variable {V : Type*} [Category V] [Abelian V] {ι : Type*} {c : ComplexShape ι}

/-- `homologyUnop` is natural (mathlib's `homologyOp_hom_naturality`, unopposed). -/
lemma homologyUnop_nat {M N : HomologicalComplex V c} (φ : M ⟶ N) (i : ι) :
    (HomologicalComplex.homologyUnop M.op i).hom ≫
        (HomologicalComplex.homologyMap ((HomologicalComplex.opFunctor V c).map φ.op) i).unop =
      HomologicalComplex.homologyMap φ i ≫ (HomologicalComplex.homologyUnop N.op i).hom := by
  have h := HomologicalComplex.homologyOp_hom_naturality φ i
  have h' := congrArg Quiver.Hom.unop h
  exact h'

lemma homologyUnop_nat_inv {M N : HomologicalComplex V c} (φ : M ⟶ N) (i : ι) :
    (HomologicalComplex.homologyMap ((HomologicalComplex.opFunctor V c).map φ.op) i).unop ≫
        (HomologicalComplex.homologyUnop N.op i).inv =
      (HomologicalComplex.homologyUnop M.op i).inv ≫ HomologicalComplex.homologyMap φ i :=
  (Iso.comp_inv_eq _).mpr (((Iso.eq_inv_comp _).mpr (homologyUnop_nat φ i)).trans (Category.assoc _ _ _).symm)

end Unop

section Ext

variable {G : Type} [Group G] {k : Type} [CommRing k]

/-- **`lem:m6-isoext-lift`: `isoExt` is natural in the resolution.** -/
lemma isoExt_hom_lift (P Q : ProjectiveResolution (Rep.trivial k G k)) (φ : P.complex ⟶ Q.complex)
    (hφ : φ ≫ Q.π = P.π) (A : Rep.{0} k G) (n : ℕ) :
    (P.isoExt n A).hom = (Q.isoExt n A).hom ≫ HomologicalComplex.homologyMap (Cover.preY φ A) n := by
  have comm : φ.f 0 ≫ Q.π.f 0 = P.π.f 0 ≫ 𝟙 _ := by
    rw [Category.comp_id, ← HomologicalComplex.comp_f, hφ]
  have hnat := ProjectiveResolution.isoLeftDerivedObj_inv_naturality (𝟙 _) P Q φ comm
    (((linearYoneda k (Rep.{0} k G)).obj A).rightOp) n
  rw [CategoryTheory.Functor.map_id, Category.comp_id] at hnat
  have hu := congrArg Quiver.Hom.unop hnat
  simp only [unop_comp] at hu
  show (P.isoLeftDerivedObj (((linearYoneda k (Rep.{0} k G)).obj A).rightOp) n).inv.unop ≫
      ((((((linearYoneda k (Rep.{0} k G)).obj A).rightOp).mapHomologicalComplex (ComplexShape.down ℕ)).obj
        P.complex).homologyUnop n).inv =
    ((Q.isoLeftDerivedObj (((linearYoneda k (Rep.{0} k G)).obj A).rightOp) n).inv.unop ≫
      ((((((linearYoneda k (Rep.{0} k G)).obj A).rightOp).mapHomologicalComplex (ComplexShape.down ℕ)).obj
        Q.complex).homologyUnop n).inv) ≫ HomologicalComplex.homologyMap (Cover.preY φ A) n
  rw [hu]
  exact (Category.assoc _ _ _).trans ((congrArg (_ ≫ ·) (homologyUnop_nat_inv (Cover.preY φ A) n)).trans
    (Category.assoc _ _ _).symm)

/-- **`thm:m6-theta-ext`: mathlib's `Ext`-built comparison is the canonical one**, for every projective resolution. -/
theorem groupCohomologyIso_eq_canIso (P : ProjectiveResolution (Rep.trivial k G k)) (A : Rep.{0} k G) (n : ℕ) :
    groupCohomologyIso A n P = canIso P A n := by
  apply Iso.ext
  set e : P.complex ⟶ (Rep.barResolution k G).complex :=
    (ProjectiveResolution.homotopyEquiv P (Rep.barResolution k G)).hom
  have hP : e ≫ (Rep.barResolution k G).π = P.π := ProjectiveResolution.homotopyEquiv_hom_π _ _
  rw [canIso_hom_eq P A n e hP]
  have hL := isoExt_hom_lift P (Rep.barResolution k G) e hP A n
  show (HomologicalComplex.homologyMap (groupCohomology.inhomogeneousCochainsIso A).hom n ≫
      ((Rep.barResolution k G).isoExt n A).inv) ≫ (P.isoExt n A).hom = _
  rw [hL]
  exact (Category.assoc _ _ _).trans (congrArg (_ ≫ ·) (Iso.inv_hom_id_assoc _ _))

end Ext

section Model

open scoped MatrixGroups
open GL2.Model GL2.Model.XBulk GL2.Cone GL2.Cone.ProjRed GL2.Cosets GL2.M4 GL2.M5

variable {R : Type} [CommRing R] {V : Type} [AddCommGroup V] [Module R V] (ρ : ΔQ →* Module.End R V)

lemma Θ_apply (h6 : IsUnit (6 : R)) (n : ℕ) (x : HM ρ n) :
    Θ ρ h6 n x = (groupCohomologyIso (VL ρ) n (model7_projRes h6)).inv.hom
      ((HomologicalComplex.homologyMap (homEqv_iso model7 (VL ρ)).hom n).hom
        ((relCxA model7 (VL ρ)).homologyEquiv (relCxA_subsingleton model7 (VL ρ)) n
          (HmapOf (psi model7 ρ eQ (hΓΔ := ΓQ_le_ΔQ)) (psi_isChainMap model7 ρ eQ) n x))) := rfl

/-- **M4's `Θ` (through `Ext`) is `Θ_can`.** -/
theorem Θ_eq_Θcan (h6 : IsUnit (6 : R)) (n : ℕ) : Θ ρ h6 n = Θcan ρ h6 n := by
  apply LinearMap.ext; intro x
  rw [Θ_apply, Θcan_apply, groupCohomologyIso_eq_canIso]

/-- **[C] `thm:classical` in [C]'s sense, for M4's `Θ`**: `Θ ∘ T_{ΓgΓ} = Tcls g n ∘ Θ`, every coset datum, `τ`, `n`. -/
theorem classical_Θ (h6 : IsUnit (6 : R)) {g : GLQ} (cd : CosetData ΓQ ΔQ g) (τ : TauR R g eQ) (n : ℕ) :
    Θ ρ h6 n ∘ₗ Tdc ρ cd τ n = Tcls ρ g (memΔ_of_cd cd) n ∘ₗ Θ ρ h6 n := by
  rw [Θ_eq_Θcan]
  exact classical_can ρ h6 cd τ n

/-- **M5's transported classical operator is the model's**: `TclsM g n = T_{ΓgΓ}` (`TdcOf`). -/
theorem TclsM_eq_TdcOf (h6 : IsUnit (6 : R)) (g : GLQ) (hg : g ∈ ΔQ) (n : ℕ) : TclsM ρ h6 g hg n = TdcOf ρ h6 g hg n := by
  apply LinearMap.ext; intro x
  apply (ΘE ρ h6 n).injective
  have h := LinearMap.congr_fun (classical_Θ ρ h6 (Classical.choice (cosetData_nonempty g hg))
    (Classical.choice (tauR_nonempty R h6 g)) n) x
  simp only [LinearMap.comp_apply] at h
  rw [← ΘE_toLinearMap] at h
  simp only [TclsM, LinearMap.comp_apply, LinearEquiv.coe_coe, LinearEquiv.apply_symm_apply]
  exact h.symm

/-- **`thm:m6-hcl`: the literature hypothesis `lit:hecke-classical` (`hcl`) is a theorem**, with `cor = corH`. -/
theorem heckeClassical_holds : GL2.Literature.HeckeClassical ρ where
  cor_formula h6 g hg n := by
    refine ⟨corH (VL ρ) (Spp g) (τS g) n, corH_res (Spp g) (τS g) (VL ρ) n, ?_⟩
    have h := classical_Θ ρ h6 (Classical.choice (cosetData_nonempty g hg)) (Classical.choice (tauR_nonempty R h6 g)) n
    exact h

/-- **M4's `thm:classical-model`, proved outright** (no `hcl`). -/
theorem thm_classical_outright : thm_classical_stmt ρ :=
  thm_classical ρ (heckeClassical_holds ρ)

end Model

end GL2.M6

end
