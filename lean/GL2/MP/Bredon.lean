import GL2.MP.RZRefine
import GL2.Model.RmkOrbifold

/-!
# MP-L3, row L10 (slot 22): the comparison `ε^MP` with the shipped Bredon complex

[GT] `gt_part_MP.tex`, rows `def:mp-lift-exact`, `def:mp-epsmp`, `thm:mp-epsmp-canonical`, `thm:mp-epsmp-quasiiso`,
backing [C] `prop:mpbredon` (1), (2).

**Lifts into exact complexes** (`ExactCx.lift'`, `ExactAug.ofQuasiIso`): mathlib's `ProjectiveResolution.lift` with the
target's projectivity removed (it is never used): a map `f : Y ⟶ Z` lifts to a chain map from a projective resolution of
`Y` to any exact augmented complex over `Z`.

**`ε^MP`** (`epsMP`, [C] `prop:mpbredon`): for an `X_bulk` descriptor `D` with H4(ii) (`XBulk.Acyclic D`, so
`C_•(X_bulk) ⊗ k → k` is exact over every ring) and an MP cover `C` with (AC) over `k`, the `Γ`-chain map
`Φ : k[𝔉'_•] → C_•(X_bulk) ⊗ k` over `id_k` (`bredonPhi`, the lift of the identity — [C]'s `𝔦 ∘ Φ^MP` up to homotopy) and

  `ε^MP : C^•_eq(K, Rel; A) ≅ Hom_Γ(C_•(X_bulk) ⊗ k, A) --Φ^*--> Hom_Γ(k[𝔉'_•], A) ≅ C^•(S; A)`,

a cochain map at **every** ring and every representation (unnormalised MP cochains).

* `bredonPhi_homotopy` ([C] (1), "its homotopy class is canonical"): any two `Γ`-chain maps `k[𝔉'_•] → C_•(X_bulk) ⊗ k`
  over the augmentations are homotopic, so `ε^MP` is canonical up to homotopy (`epsMP_homologyMap_eq`: any such `Φ'`
  gives the same map on cohomology).
* `epsMP_quasiIso` ([C] (2), the "if"): if `6 ∈ k^×` (and the stabiliser orders divide `12`, H4(iv)), `Φ` is a homotopy
  equivalence (both sides are projective resolutions), so `ε^MP` is a quasi-isomorphism — [C] `thm:specialisation`
  re-proved on the MP model.  Instances: `model7_epsMP_quasiIso`, `rz_epsMP_quasiIso` (Bredon side always `M₇`, MP side `M₇`
  resp. `R_ℤ`).

Not formalised here: [C] (2)'s "not only if" witness (`ℤ[1/2]`, `n` odd) — slot 24 (C-w) proves it in
`GL2.MP.BredonWitness` (`model7_epsMP_witness`); (3) (injectivity on `H^0`, `H^1`, via the isotropy spectral sequence) and
base change stay open.
-/

noncomputable section

namespace GL2.MP

open CategoryTheory Limits

namespace ExactCx

variable {𝒞 : Type*} [Category 𝒞] [Abelian 𝒞]

/-- **An augmentation that is a quasi-isomorphism gives an exact augmented complex.** -/
def ExactAug.ofQuasiIso {Q : ChainComplex 𝒞 ℕ} {Z : 𝒞} (π : Q ⟶ (ChainComplex.single₀ 𝒞).obj Z) [hq : QuasiIso π] :
    ExactAug Q Z where
  ε := (Q.toSingle₀Equiv Z π).1
  w := (Q.toSingle₀Equiv Z π).2
  ex0 := by
    have h0 := hq.quasiIsoAt 0
    rw [ChainComplex.quasiIsoAt₀_iff, ShortComplex.quasiIso_iff_of_zeros' _ (Q.shape 0 0 (by simp)) rfl rfl] at h0
    exact h0.1
  epi := by
    have h0 := hq.quasiIsoAt 0
    rw [ChainComplex.quasiIsoAt₀_iff, ShortComplex.quasiIso_iff_of_zeros' _ (Q.shape 0 0 (by simp)) rfl rfl] at h0
    exact h0.2
  exS n := by
    have h := hq.quasiIsoAt (n + 1)
    rw [quasiIsoAt_iff_exactAt' (hL := ChainComplex.exactAt_succ_single_obj ..),
      HomologicalComplex.exactAt_iff' _ (n + 2) (n + 1) n (by simp) (by simp)] at h
    exact h

lemma ExactAug.ofQuasiIso_ε {Q : ChainComplex 𝒞 ℕ} {Z : 𝒞} (π : Q ⟶ (ChainComplex.single₀ 𝒞).obj Z) [QuasiIso π] :
    (ExactAug.ofQuasiIso π).ε = π.f 0 := rfl

variable {Y : 𝒞} {Q : ChainComplex 𝒞 ℕ} {Z : 𝒞} (P : ProjectiveResolution Y) (E : ExactAug Q Z) (f : Y ⟶ Z)

/-- Auxiliary for `lift'`. -/
def liftZ : P.complex.X 0 ⟶ Q.X 0 :=
  haveI := E.epi
  Projective.factorThru (P.π.f 0 ≫ f) E.ε

@[reassoc (attr := simp)]
lemma liftZ_comp : liftZ P E f ≫ E.ε = P.π.f 0 ≫ f :=
  haveI := E.epi
  Projective.factorThru_comp _ _

/-- Auxiliary for `lift'`. -/
def liftO : P.complex.X 1 ⟶ Q.X 1 :=
  E.ex0.liftFromProjective (P.complex.d 1 0 ≫ liftZ P E f)
    (by
      rw [Category.assoc, liftZ_comp]
      exact (Category.assoc _ _ _).symm.trans ((congrArg (· ≫ f) P.complex_d_comp_π_f_zero).trans zero_comp))

@[reassoc (attr := simp)]
lemma liftO_comp : liftO P E f ≫ Q.d 1 0 = P.complex.d 1 0 ≫ liftZ P E f :=
  E.ex0.liftFromProjective_comp _ _

/-- **A map lifts to a chain map from a projective resolution to an exact augmented complex.** -/
def lift' : P.complex ⟶ Q :=
  ChainComplex.mkHom _ _ (liftZ P E f) (liftO P E f) (liftO_comp P E f) fun n ⟨g, g', w⟩ =>
    ⟨(E.exS n).liftFromProjective (P.complex.d (n + 2) (n + 1) ≫ g') (by simp [w]),
      (E.exS n).liftFromProjective_comp _ _⟩

lemma lift'_f_zero : (lift' P E f).f 0 = liftZ P E f := ChainComplex.mkHom_f_0 _ _ _ _ _ _

lemma lift'_aug : (lift' P E f).f 0 ≫ E.ε = P.π.f 0 ≫ f := by rw [lift'_f_zero, liftZ_comp]

lemma lift'_aug_id (E : ExactAug Q Y) : (lift' P E (𝟙 Y)).f 0 ≫ E.ε = P.π.f 0 :=
  (lift'_aug P E (𝟙 Y)).trans (Category.comp_id _)

end ExactCx

namespace Cover

open ExactCx GL2.Model GL2.Cone.ProjRed
open scoped MatrixGroups

variable {k : Type} [CommRing k] {C : Cover.{0} SL(2, ℤ)} (H : C.Hyp) (hAC : (yCx H k).toCx.Acyclic)
  (D : XBulkDatum) (hA : XBulk.Acyclic D) (v : D.V 0)

/-- `C_•(X_bulk) ⊗ k → k` as an exact augmented complex (H4(ii), every ring). -/
def repExact : ExactAug (XBulk.repCx D k) (Rep.trivial k SL(2, ℤ) k) :=
  haveI := XBulk.repAug_quasiIso k hA v
  ExactAug.ofQuasiIso (XBulk.repAug D k)

/-- `k[𝔉'_•] → k` as an exact augmented complex ((AC) over `k`). -/
def mpExact : ExactAug (mpCx C k) (Rep.trivial k SL(2, ℤ) k) :=
  haveI := mpAug_quasiIso (fCx_acyclic H k hAC)
  ExactAug.ofQuasiIso (mpAug C k)

/-- **`Φ`: the `Γ`-chain map `k[𝔉'_•] → C_•(X_bulk) ⊗ k` over `id_k`** (the lift of the identity). -/
def bredonPhi : mpCx C k ⟶ XBulk.repCx D k :=
  lift' (mpRes H k hAC) (repExact D hA v) (𝟙 _)

lemma bredonPhi_aug : (bredonPhi H hAC D hA v).f 0 ≫ (XBulk.repAug D k).f 0 = (mpAug C k).f 0 :=
  lift'_aug_id (mpRes H k hAC) (repExact D hA v)

/-- **[C] `prop:mpbredon` (1), canonical homotopy class**: any `Γ`-chain map `k[𝔉'_•] → C_•(X_bulk) ⊗ k` over the
augmentations is homotopic to `Φ`. -/
def bredonPhi_homotopy (Φ' : mpCx C k ⟶ XBulk.repCx D k) (h : Φ'.f 0 ≫ (XBulk.repAug D k).f 0 = (mpAug C k).f 0) :
    Homotopy (bredonPhi H hAC D hA v) Φ' :=
  homotopy' (P := mpRes H k hAC) (repExact D hA v) _ _ ((bredonPhi_aug H hAC D hA v).trans h.symm)

/-- **`ε^MP`** ([C] `prop:mpbredon`): `C^•_eq(K, Rel; A) → C^•(S; A)` through `Φ`, a cochain map at every ring. -/
def epsMP (A : Rep.{0} k SL(2, ℤ)) : (XBulk.relCxA D A).toNat ⟶ C.mpComplex A :=
  (XBulk.homEqv_iso D A).hom ≫ preY (bredonPhi H hAC D hA v) A ≫ (linYonIso A).hom

/-- **The homotopy class of `ε^MP` is canonical** ([C] `prop:mpbredon` (1)): any augmentation-compatible `Φ'` induces the
same map on cohomology. -/
theorem epsMP_homologyMap_eq (A : Rep.{0} k SL(2, ℤ)) (Φ' : mpCx C k ⟶ XBulk.repCx D k)
    (h : Φ'.f 0 ≫ (XBulk.repAug D k).f 0 = (mpAug C k).f 0) (n : ℕ) :
    HomologicalComplex.homologyMap (epsMP H hAC D hA v A) n =
      HomologicalComplex.homologyMap ((XBulk.homEqv_iso D A).hom ≫ preY Φ' A ≫ (linYonIso A).hom) n := by
  simp only [epsMP, HomologicalComplex.homologyMap_comp]
  rw [(preYHomotopy (bredonPhi_homotopy H hAC D hA v Φ' h) A).homologyMap_eq]

variable (h6 : IsUnit (6 : k)) (hdvd : ∀ q (σ : D.V q), (D.M q).rep σ = σ → ((D.M q).S σ σ).card ∣ 12)

/-- For `6 ∈ k^×`: `C_•(X_bulk) ⊗ k → k` is a projective resolution. -/
def repRes : ProjectiveResolution (Rep.trivial k SL(2, ℤ) k) :=
  haveI := XBulk.repAug_quasiIso k hA v
  projRes (XBulk.repCx D k) (XBulk.repAug D k) (fun m => XBulk.perm_projective D h6 hdvd m)

lemma bredonPsi_aug : (lift' (repRes D hA v h6 hdvd) (mpExact H hAC) (𝟙 _)).f 0 ≫ (mpAug C k).f 0 =
    (XBulk.repAug D k).f 0 :=
  lift'_aug_id (repRes D hA v h6 hdvd) (mpExact H hAC)

/-- **For `6 ∈ k^×`, `Φ` is a homotopy equivalence** (both sides are projective resolutions of `k`). -/
def bredonPhiEquiv : HomotopyEquiv (mpCx C k) (XBulk.repCx D k) where
  hom := bredonPhi H hAC D hA v
  inv := lift' (repRes D hA v h6 hdvd) (mpExact H hAC) (𝟙 _)
  homotopyHomInvId := homotopy' (P := mpRes H k hAC) (mpExact H hAC) _ _
    ((Category.assoc _ _ _).trans ((congrArg ((bredonPhi H hAC D hA v).f 0 ≫ ·) (bredonPsi_aug H hAC D hA v h6 hdvd)).trans
      ((bredonPhi_aug H hAC D hA v).trans (Category.id_comp _).symm)))
  homotopyInvHomId := homotopy' (P := repRes D hA v h6 hdvd) (repExact D hA v) _ _
    ((Category.assoc _ _ _).trans
      ((congrArg ((lift' (repRes D hA v h6 hdvd) (mpExact H hAC) (𝟙 _)).f 0 ≫ ·) (bredonPhi_aug H hAC D hA v)).trans
      ((bredonPsi_aug H hAC D hA v h6 hdvd).trans (Category.id_comp _).symm)))

include h6 hdvd in
/-- **[C] `prop:mpbredon` (2), the "if"**: for `6 ∈ k^×` (and stabiliser orders dividing `12`), `ε^MP` is a
quasi-isomorphism — `H^n(C^•_eq(K, Rel; A)) ≅ H^n(C^•(S; A))`, [C] `thm:specialisation` re-proved on the MP model. -/
theorem epsMP_quasiIso (A : Rep.{0} k SL(2, ℤ)) : QuasiIso (epsMP H hAC D hA v A) := by
  have : QuasiIso (preY (bredonPhi H hAC D hA v) A) :=
    (inferInstance : QuasiIso (preYEquiv (bredonPhiEquiv H hAC D hA v h6 hdvd) A).hom)
  unfold epsMP
  infer_instance

end Cover

open scoped MatrixGroups
open GL2.Model

/-- **[C] `prop:mpbredon` (2) for `M₇`**: `ε^MP : C^•_eq(K, Rel; A) → C^•(S(M₇); A)` is a quasi-isomorphism for `6 ∈ k^×`. -/
theorem model7_epsMP_quasiIso {k : Type} [CommRing k] (h6 : IsUnit (6 : k)) (A : Rep.{0} k SL(2, ℤ)) :
    QuasiIso (Cover.epsMP model7Hyp (model7_AC k) model7 model7_acyclic (M7.vx 0) A) :=
  Cover.epsMP_quasiIso model7Hyp (model7_AC k) model7 model7_acyclic (M7.vx 0) h6 (fun q σ h => model7_dvd q σ h) A

/-- **[C] `prop:mpbredon` (2) for `R_ℤ`**: `ε^MP : C^•_eq(K, Rel; A) → C^•(S(R_ℤ); A)` is a quasi-isomorphism for
`6 ∈ k^×`. -/
theorem rz_epsMP_quasiIso {k : Type} [CommRing k] (h6 : IsUnit (6 : k)) (A : Rep.{0} k SL(2, ℤ)) :
    QuasiIso (Cover.epsMP rzHyp (rz_AC k) model7 model7_acyclic (M7.vx 0) A) :=
  Cover.epsMP_quasiIso rzHyp (rz_AC k) model7 model7_acyclic (M7.vx 0) h6 (fun q σ h => model7_dvd q σ h) A

end GL2.MP
