import GL2.Model.Orbifold
import Mathlib.Algebra.Homology.ShortComplex.ModuleCat

/-!
# M3-d (L19-2): `rmk:orbifold` on the 7-vertex model

[GT] `lem:cochaincx-homology`, `lem:xbulk-homeqv`, `thm:model7-rmkorbifold`, mirroring [C] v1.18.0 `rmk:orbifold` (on the
model, `6 ∈ R^×`) and `thm:explicit_complex_equiv`(1).

* `CochainCx.toNat` / `CochainCx.homologyEquiv` -- the cone layer's hand-built `ℤ`-graded cochain complexes, read in
  non-negative degrees as mathlib `CochainComplex (ModuleCat R) ℕ`, have the same cohomology `H^n = Z^n / B^n`
  (when the degree `-1` term vanishes).
* `XBulk.homEqv` -- `Hom_{SL(2, ℤ)}(R[Simp_q], A) ≅ Hom_{SL(2, ℤ)}(C_q(X_bulk), A)` (`chainsEquivPerm`), compatible with
  the coboundaries; with `Ψ` (`psiEquiv`) this is `homEqv_iso : (repCx.linearYonedaObj R A).X q ≅ C^q_eq(K, Rel; A)`.
* `rmkOrbifold` -- for every H4-descriptor `D` with `Acyclic D` (H4(ii) in the form `eq:contractibleuse`), a vertex, and
  representative stabiliser orders dividing `12` (H4(iv)), for `6 ∈ R^×` and any representation `A` of `SL(2, ℤ)`, the
  cohomology of the relative complex `relCx` is group cohomology: `H^n(C^•_eq(K, Rel; A)) ≅ H^n(SL(2, ℤ); A)`, as
  `psiEquiv` ∘ `homEqv` ∘ `projRed` (F3-19 MINOR 3); `model7_rmkOrbifold` is its instance on the 7-vertex model.
* `rmkOrbifoldL` -- the same in [C]'s `(G, Γ, Δ, e, ρ)` form: `relCx D hΓΔ ρ e` is `relCxA D (Rep.of ρ_L)`
  definitionally (`relCx_eq_relCxA`), `ρ_L = ρ ∘ e⁻¹` (F3-19 MINOR 2).

Claimed: every `X_bulk`-descriptor satisfying H4(ii) (as `Acyclic`) and H4(iv), and `6 ∈ R^×`; the remark's `K` beyond the
descriptor framework, and [C]'s proof route (relative projectivity, Shapiro, transfer), are not.
-/

noncomputable section

namespace GL2.Cone

open CategoryTheory

variable {R : Type} [CommRing R]

namespace CochainCx

variable (C : CochainCx.{0} R)

/-- The non-negative part of a `ℤ`-graded cochain complex, as a mathlib cochain complex of `R`-modules. -/
def toNat : CochainComplex (ModuleCat R) ℕ :=
  CochainComplex.of (fun n => ModuleCat.of R (C.X n)) (fun n => ModuleCat.ofHom (C.d n)) (fun n => by
    ext x
    exact C.d_d n x)

lemma toNat_d (n : ℕ) : (C.toNat).d n (n + 1) = ModuleCat.ofHom (C.d n) := by
  unfold toNat; simp

lemma toNat_d_zero : (C.toNat).d 0 0 = 0 := C.toNat.shape 0 0 (by simp)

/-- The short complex `M --f--> N --g--> P` of modules. -/
abbrev scMk {M N P : Type} [AddCommGroup M] [AddCommGroup N] [AddCommGroup P] [Module R M] [Module R N]
    [Module R P] (f : M →ₗ[R] N) (g : N →ₗ[R] P) (w : g ∘ₗ f = 0) : ShortComplex (ModuleCat R) :=
  ShortComplex.mk (ModuleCat.ofHom f) (ModuleCat.ofHom g) (by ext x; exact LinearMap.congr_fun w x)

/-- The homology of `M --f--> N --g--> P` in `ModuleCat` is `ker g / f(M)`. -/
def homologyMk {M N P : Type} [AddCommGroup M] [AddCommGroup N] [AddCommGroup P] [Module R M] [Module R N]
    [Module R P] (f : M →ₗ[R] N) (g : N →ₗ[R] P) (w : g ∘ₗ f = 0) :
    (scMk f g w).homology ≅
      ModuleCat.of R (LinearMap.ker g ⧸ LinearMap.range (f.codRestrict (LinearMap.ker g)
        (fun x => LinearMap.congr_fun w x))) :=
  ShortComplex.moduleCatHomologyIso _

/-- The cocycles mod coboundaries in degree `p + 1`, written with the range of `d_p`. -/
lemma comap_B_succ (p : ℤ) : (C.B (p + 1)).comap (C.Z (p + 1)).subtype =
    LinearMap.range ((C.d p).codRestrict (C.Z (p + 1)) (fun x => (C.mem_Z _ _).2 (C.d_d p x))) := by
  ext z
  rw [Submodule.mem_comap, Submodule.coe_subtype, C.mem_B_succ, LinearMap.mem_range]
  constructor
  · rintro ⟨y, hy⟩; exact ⟨y, Subtype.ext hy⟩
  · rintro ⟨y, hy⟩; exact ⟨y, congrArg Subtype.val hy⟩

/-- In degree `0`, with `C^{-1} = 0`, there are no coboundaries. -/
lemma comap_B_zero (h : Subsingleton (C.X (-1))) : (C.B 0).comap (C.Z 0).subtype = ⊥ := by
  ext z
  rw [Submodule.mem_comap, Submodule.coe_subtype, Submodule.mem_bot]
  constructor
  · intro hz
    obtain ⟨y, hy⟩ := (C.mem_B_iff' (p := -1) (by norm_num) _).1 hz
    rw [Subsingleton.elim y 0, map_zero, map_zero] at hy
    exact Subtype.ext hy.symm
  · rintro rfl; exact Submodule.zero_mem _

/-- **`lem:cochaincx-homology`, degree `n + 1`.** -/
def homologyEquivSucc (n : ℕ) : C.H ((n + 1 : ℕ) : ℤ) ≃ₗ[R] (C.toNat.homology (n + 1)) :=
  (Submodule.quotEquivOfEq _ _ (C.comap_B_succ n)).trans
    ((C.toNat.homologyIsoSc' n (n + 1) (n + 2) (by simp) (by simp) ≪≫
      ShortComplex.homologyMapIso (ShortComplex.isoMk (S₁ := C.toNat.sc' n (n + 1) (n + 2))
        (S₂ := scMk (C.d n) (C.d (n + 1)) (C.d_comp_d n))
        (Iso.refl _) (Iso.refl _) (Iso.refl _)
        (by simp [toNat_d]; exact (Category.id_comp _).trans (Category.comp_id _).symm)
        (by simp [toNat_d]; exact (Category.id_comp _).trans (Category.comp_id _).symm)) ≪≫
      homologyMk (C.d n) (C.d (n + 1)) (C.d_comp_d n)).symm.toLinearEquiv)

/-- **`lem:cochaincx-homology`, degree `0`** (with `C^{-1} = 0`). -/
def homologyEquivZero (h : Subsingleton (C.X (-1))) : C.H ((0 : ℕ) : ℤ) ≃ₗ[R] (C.toNat.homology 0) :=
  (Submodule.quotEquivOfEq _ _ (by
      refine (C.comap_B_zero h).trans ?_
      symm
      rw [LinearMap.range_eq_bot]
      ext x
      rfl :
      (C.B ((0 : ℕ) : ℤ)).comap (C.Z ((0 : ℕ) : ℤ)).subtype =
        LinearMap.range ((0 : C.X 0 →ₗ[R] C.X 0).codRestrict (C.Z 0) (fun _ => Submodule.zero_mem _)))).trans
    ((C.toNat.homologyIsoSc' 0 0 1 (by simp) (by simp) ≪≫
      ShortComplex.homologyMapIso (ShortComplex.isoMk (S₁ := C.toNat.sc' 0 0 1)
        (S₂ := scMk (0 : C.X 0 →ₗ[R] C.X 0) (C.d 0) (by simp))
        (Iso.refl _) (Iso.refl _) (Iso.refl _)
        (by simp; exact Limits.comp_zero.trans Limits.zero_comp.symm)
        (by simp [toNat_d]; exact (Category.id_comp _).trans (Category.comp_id _).symm)) ≪≫
      homologyMk (0 : C.X 0 →ₗ[R] C.X 0) (C.d 0) (by simp)).symm.toLinearEquiv)

/-- **`lem:cochaincx-homology`**: for a `ℤ`-graded cochain complex with `C^{-1} = 0`, the hand-built cohomology
`H^n = Z^n / B^n` is mathlib's homology of the non-negative part. -/
def homologyEquiv (h : Subsingleton (C.X (-1))) : ∀ n : ℕ, C.H n ≃ₗ[R] (C.toNat.homology n)
  | 0 => C.homologyEquivZero h
  | n + 1 => C.homologyEquivSucc n

end CochainCx

end GL2.Cone

namespace GL2.Model

open scoped MatrixGroups
open CategoryTheory GL2.Cone GL2.Cone.ProjRed MonoidAlgebra

namespace XBulk

variable (D : XBulkDatum) {k : Type} [CommRing k] (A : Rep k SL(2, ℤ))

/-- A representation of `SL(2, ℤ)` as the cone layer's `ρ : Δ → End(V)` with `Γ = Δ = SL(2, ℤ)`. -/
def ρA : (⊤ : Submonoid SL(2, ℤ)) →* Module.End k A := A.ρ.comp (⊤ : Submonoid SL(2, ℤ)).subtype

/-- `Γ = SL(2, ℤ)` inside `G = SL(2, ℤ)`. -/
abbrev hTop : ∀ γ : (⊤ : Subgroup SL(2, ℤ)), (γ : SL(2, ℤ)) ∈ (⊤ : Submonoid SL(2, ℤ)) := fun _ => Submonoid.mem_top _

/-- The equivariant cochains `Hom_{SL(2, ℤ)}(C_q(X_bulk), A)`. -/
abbrev EqCA := EqC D (hΓΔ := hTop) (ρA A) Subgroup.topEquiv

/-- The relative complex `C^•_eq(K, Rel; A)` of `X_bulk` with coefficients in `A`. -/
abbrev relCxA := relCx D hTop (ρA A) Subgroup.topEquiv

lemma ρL_A (t : SL(2, ℤ)) : ρL hTop (ρA A) Subgroup.topEquiv t = A.ρ t := rfl

lemma isEqv_A {q : ℤ} (φ : (EqCA D A).X q) (t : SL(2, ℤ)) (y : chains D q) : φ.1 (t • y) = A.ρ t (φ.1 y) :=
  eqC_isEqv D (ρA A) Subgroup.topEquiv φ t y

/-- `φ ↦ (e_s ↦ φ(orient s))` on the free module. -/
def homFwdL {q : ℤ} (φ : chains D q →+ A) : (Simp D q →₀ k) →ₗ[k] A :=
  Finsupp.linearCombination k (fun s => φ (orient D q s))

lemma homFwdL_castF {q : ℤ} (φ : chains D q →+ A) (f : Simp D q →₀ ℤ) :
    homFwdL D A φ (castF f) = φ (toChains D q f) := by
  induction f using Finsupp.induction_linear with
  | zero => simp
  | add f g hf hg => rw [map_add, map_add, hf, hg, map_add, map_add]
  | single s n =>
    rw [castF_single]
    unfold homFwdL
    rw [Finsupp.linearCombination_single, toChains_single, map_zsmul, Int.cast_smul_eq_zsmul]

lemma homFwdL_bcF_dZ (q : ℤ) (φ : chains D q →+ A) (x : Simp D (q + 1) →₀ k) :
    homFwdL D A φ (bcF k (dZ D q) x) = homFwdL D A (φ.comp (bd D q).toAddMonoidHom) x := by
  induction x using Finsupp.induction_linear with
  | zero => simp
  | add a b ha hb => rw [map_add, map_add, map_add, ha, hb]
  | single s r =>
    rw [bcF_single, map_smul, homFwdL_castF]
    unfold homFwdL
    rw [Finsupp.linearCombination_single]
    simp only [AddMonoidHom.comp_apply, LinearMap.toAddMonoidHom_coe, dZ, LinearMap.coe_comp, Function.comp_apply,
      toChains_ofChains, toChains_single, one_smul]

/-- `Hom_{SL(2, ℤ)}(C_q(X_bulk), A) → Hom_{SL(2, ℤ)}(R[Simp_q], A)`. -/
def homFwd {q : ℤ} (φ : (EqCA D A).X q) : Rep.ofMulAction k SL(2, ℤ) (Simp D q) ⟶ A :=
  Rep.ofHom ⟨homFwdL D A φ.1 ∘ₗ (coeffLinearEquiv k).toLinearMap, fun g => by
    refine LinearMap.ext fun x => ?_
    simp only [LinearMap.coe_comp, Function.comp_apply, LinearEquiv.coe_coe, Representation.ofMulAction_def,
      LinearEquiv.apply_symm_apply, Finsupp.lmapDomain_apply]
    induction (coeffLinearEquiv k) x using Finsupp.induction_linear with
    | zero => simp
    | add f g hf hg => rw [Finsupp.mapDomain_add, map_add, map_add, hf, hg, map_add]
    | single s r =>
      rw [Finsupp.mapDomain_single]
      unfold homFwdL
      rw [Finsupp.linearCombination_single, Finsupp.linearCombination_single, orient_smul, map_smul]
      exact congrArg (r • ·) (isEqv_A D A φ g (orient D q s))⟩

lemma homFwd_apply {q : ℤ} (φ : (EqCA D A).X q) (x : k[Simp D q]) :
    (homFwd D A φ).hom x = homFwdL D A φ.1 (coeffLinearEquiv k x) := rfl

/-- `Hom_{SL(2, ℤ)}(R[Simp_q], A) → Hom_{SL(2, ℤ)}(C_q(X_bulk), A)`: `y ↦ f(e_{ofChains y})`. -/
def homBwdF {q : ℤ} (f : Rep.ofMulAction k SL(2, ℤ) (Simp D q) ⟶ A) : chains D q →+ A :=
  (f.hom.toLinearMap.toAddMonoidHom.comp ((coeffLinearEquiv k).symm.toLinearMap.toAddMonoidHom.comp
    (castF.comp (ofChains D q).toAddMonoidHom)))

lemma homBwdF_apply {q : ℤ} (f : Rep.ofMulAction k SL(2, ℤ) (Simp D q) ⟶ A) (y : chains D q) :
    homBwdF D A f y = f.hom ((coeffLinearEquiv k).symm (castF (ofChains D q y))) := rfl

lemma homBwdF_mem {q : ℤ} (f : Rep.ofMulAction k SL(2, ℤ) (Simp D q) ⟶ A) :
    homBwdF D A f ∈ eqHom (⊤ : Subgroup SL(2, ℤ)) ⊤ hTop (ρA A) (Yc D Subgroup.topEquiv) q := by
  rw [mem_eqHom_iff]
  intro t y
  rw [homBwdF_apply, homBwdF_apply, ofChains_smul, castF_mapDomain, ρL_A, ← Rep.hom_comm_apply]
  congr 1

/-- **`XBulk.homEqv`**: `Hom_{SL(2, ℤ)}(C_q(X_bulk), A) ≃ Hom_{SL(2, ℤ)}(R[Simp_q], A)` (`chainsEquivPerm`). -/
def homEqv (q : ℤ) : (EqCA D A).X q ≃ₗ[k] (Rep.ofMulAction k SL(2, ℤ) (Simp D q) ⟶ A) where
  toFun φ := homFwd D A φ
  invFun f := ⟨homBwdF D A f, homBwdF_mem D A f⟩
  map_add' φ ψ := by
    apply Rep.hom_ext; apply Representation.IntertwiningMap.ext; refine LinearMap.ext fun x => ?_
    show homFwdL D A (φ.1 + ψ.1) _ = homFwdL D A φ.1 _ + homFwdL D A ψ.1 _
    rw [← LinearMap.add_apply]
    congr 1
    apply Finsupp.lhom_ext; intro s r
    simp only [homFwdL, Finsupp.linearCombination_single, LinearMap.add_apply]
    change r • (φ.1 (orient D q s) + ψ.1 (orient D q s)) = _
    exact smul_add _ _ _
  map_smul' c φ := by
    apply Rep.hom_ext; apply Representation.IntertwiningMap.ext; refine LinearMap.ext fun x => ?_
    show homFwdL D A (c • φ.1) _ = c • homFwdL D A φ.1 _
    rw [← LinearMap.smul_apply]
    congr 1
    apply Finsupp.lhom_ext; intro s r
    simp only [homFwdL, Finsupp.linearCombination_single, LinearMap.smul_apply]
    show r • (c • φ.1 (orient D q s)) = c • r • φ.1 (orient D q s)
    exact smul_comm r c _
  left_inv φ := by
    apply Subtype.ext
    ext y
    show homFwdL D A φ.1 (coeffLinearEquiv k ((coeffLinearEquiv k).symm (castF (ofChains D q y)))) = φ.1 y
    rw [LinearEquiv.apply_symm_apply]
    have := homFwdL_castF D A φ.1 (ofChains D q y)
    exact this.trans (congrArg φ.1 (toChains_ofChains D q y))
  right_inv f := by
    apply Rep.hom_ext; apply Representation.IntertwiningMap.ext; refine LinearMap.ext fun x => ?_
    show homFwdL D A (homBwdF D A f) (coeffLinearEquiv k x) = f.hom x
    conv_rhs => rw [← (coeffLinearEquiv k).symm_apply_apply x]
    induction (coeffLinearEquiv k) x using Finsupp.induction_linear with
    | zero => simp
    | add a b ha hb => rw [map_add, ha, hb, map_add, map_add]
    | single s r =>
      unfold homFwdL
      rw [Finsupp.linearCombination_single, homBwdF_apply, ofChains_orient, castF_single, Int.cast_one]
      rw [← map_smul, ← map_smul, Finsupp.smul_single, smul_eq_mul, mul_one]

/-- `homEqv` intertwines the coboundaries: `homEqv(φ ∘ ∂) = ∂ ≫ homEqv(φ)`. -/
lemma homEqv_d (q : ℕ) (φ : (EqCA D A).X q) :
    homEqv D A ((q : ℤ) + 1) ((EqCA D A).d q φ) = repD D k q ≫ homEqv D A q φ := by
  apply Rep.hom_ext; apply Representation.IntertwiningMap.ext; refine LinearMap.ext fun x => ?_
  show homFwdL D A (φ.1.comp (bd D q).toAddMonoidHom) (coeffLinearEquiv k x) =
    homFwdL D A φ.1 (coeffLinearEquiv k ((coeffLinearEquiv k).symm (bcF k (dZ D q) (coeffLinearEquiv k x))))
  rw [LinearEquiv.apply_symm_apply]
  exact (homFwdL_bcF_dZ D A q φ.1 _).symm

/-- `Ψ⁻¹` then `homEqv`: `C^q_eq(K, Rel; A) ≃ Hom_{SL(2, ℤ)}(R[Simp_q], A)`. -/
def relHomEqv (q : ℤ) : (relCxA D A).X q ≃ₗ[k] (Rep.ofMulAction k SL(2, ℤ) (Simp D q) ⟶ A) :=
  (psiEquiv D (hΓΔ := hTop) (ρA A) Subgroup.topEquiv q).symm.trans (homEqv D A q)

lemma relCxA_subsingleton : Subsingleton ((relCxA D A).X (-1)) := by
  have := D.dim2 (-1) (Or.inl (by norm_num))
  exact ⟨fun a b => Subtype.ext (funext fun v => (IsEmpty.false v).elim)⟩

/-- **`XBulk.homEqv_iso`**: the cochain complex `Hom_{SL(2, ℤ)}(repCx, A)` is isomorphic to the relative complex
`C^•_eq(K, Rel; A)` (in non-negative degrees), by `homEqv` and `Ψ`. -/
def homEqv_iso : (relCxA D A).toNat ≅ (repCx D k).linearYonedaObj k A :=
  HomologicalComplex.Hom.isoOfComponents (fun n => (relHomEqv D A n).toModuleIso) (by
    rintro i j (rfl : i + 1 = j)
    refine ModuleCat.hom_ext (LinearMap.ext fun x => ?_)
    rw [CochainCx.toNat_d]
    simp only [ChainComplex.linearYonedaObj_d, LinearEquiv.toModuleIso_hom, repCx_d]
    change repD D k i ≫ homEqv D A i (phi D (hΓΔ := hTop) (ρA A) Subgroup.topEquiv i x) =
      homEqv D A ((i : ℤ) + 1) (phi D (hΓΔ := hTop) (ρA A) Subgroup.topEquiv ((i : ℤ) + 1) ((relCxA D A).d i x))
    have h1 := (phi_isChainMap D (hΓΔ := hTop) (ρA A) Subgroup.topEquiv).apply i (x : (relCxA D A).X i)
    exact (homEqv_d D A i _).symm.trans (congrArg (homEqv D A ((i : ℤ) + 1)) h1))

end XBulk

/-- **`rmkOrbifold`** ([C] `rmk:orbifold`, for every H4-descriptor; F3-19 MINOR 3): if `X_bulk` of the descriptor `D` is
acyclic in the form `eq:contractibleuse` (`hA`), has a vertex `v`, its representative stabiliser orders divide `12`
(H4(iv), `hdvd`), and `6 ∈ k^×` (H3), then for every representation `A` of `SL(2, ℤ)` over `k` the cohomology of the
relative complex `C^•_eq(K, Rel; A)` is the group cohomology `H^n(SL(2, ℤ); A)`, every `n ≥ 0`. -/
def rmkOrbifold (D : XBulkDatum) (hA : XBulk.Acyclic D)
    (hdvd : ∀ q (σ : D.V q), (D.M q).rep σ = σ → ((D.M q).S σ σ).card ∣ 12) (v : D.V 0)
    {k : Type} [CommRing k] (h6 : IsUnit (6 : k)) (A : Rep k SL(2, ℤ)) (n : ℕ) :
    (XBulk.relCxA D A).H n ≃ₗ[k] groupCohomology A n :=
  haveI := XBulk.repAug_quasiIso k hA v
  ((XBulk.relCxA D A).homologyEquiv (XBulk.relCxA_subsingleton D A) n).trans
    ((HomologicalComplex.homologyMapIso (XBulk.homEqv_iso D A) n ≪≫
      (projRed (XBulk.repCx D k) (XBulk.repAug D k) (fun m => XBulk.perm_projective D h6 hdvd m) A n).symm
      ).toLinearEquiv)

/-- **`model7_rmkOrbifold`** ([C] `rmk:orbifold` on the 7-vertex model): for `6 ∈ k^×` and every representation `A` of
`SL(2, ℤ)` over `k`, the cohomology of the relative complex `C^•_eq(K, Rel; A)` of the model is the group cohomology
`H^n(SL(2, ℤ); A)`.  The instance of `rmkOrbifold` at `model7_acyclic`, `model7_dvd` and the vertex `v₀`. -/
def model7_rmkOrbifold {k : Type} [CommRing k] (h6 : IsUnit (6 : k)) (A : Rep k SL(2, ℤ)) (n : ℕ) :
    (XBulk.relCxA model7 A).H n ≃ₗ[k] groupCohomology A n :=
  rmkOrbifold model7 model7_acyclic model7_dvd (M7.vx 0) h6 A n

/-! ### The `(G, Γ, Δ, e, ρ)` form (F3-19 MINOR 2)

[C]'s `L` is a `Δ`-module, `Γ ⊆ Δ ⊆ G`, `e : Γ ≅ SL(2, ℤ)`; the relative complex `relCx D hΓΔ ρ e` sees `ρ` only through
`ρ_L = ρ ∘ e⁻¹` (`ρL`).  With `ρ_L` read as a representation of `SL(2, ℤ)` (`ρLRep`), `relCx D hΓΔ ρ e` **is**
`relCxA D (Rep.of ρ_L)` (definitionally), so `rmkOrbifold` applies verbatim. -/

section General

variable {G : Type} [Group G] {Γ : Subgroup G} {Δ : Submonoid G} (hΓΔ : ∀ γ : Γ, (γ : G) ∈ Δ)
  {k : Type} [CommRing k] {V : Type} [AddCommGroup V] [Module k V] (ρ : Δ →* Module.End k V) (e : Γ ≃* SL(2, ℤ))

/-- `ρ_L(t) = ρ(e⁻¹ t)` as a representation of `SL(2, ℤ)` on `V`. -/
def ρLRep : Representation k SL(2, ℤ) V where
  toFun := XBulk.ρL hΓΔ ρ e
  map_one' := XBulk.ρL_one ρ e
  map_mul' := XBulk.ρL_mul ρ e

/-- The relative complex of `(G, Γ, Δ, e, ρ)` is the relative complex with coefficients in `Rep.of ρ_L`. -/
lemma relCx_eq_relCxA (D : XBulkDatum) : XBulk.relCx D hΓΔ ρ e = XBulk.relCxA D (Rep.of (ρLRep hΓΔ ρ e)) := rfl

/-- **`rmkOrbifoldL`** ([C] `rmk:orbifold` in the `(G, Γ, Δ, e, ρ)` form, every H4-descriptor): for `6 ∈ k^×`,
`H^n(C^•_eq(K, Rel; L)) ≅ H^n(SL(2, ℤ); V_L)` with `V_L = (V, ρ ∘ e⁻¹)`, every `n ≥ 0`. -/
def rmkOrbifoldL (D : XBulkDatum) (hA : XBulk.Acyclic D)
    (hdvd : ∀ q (σ : D.V q), (D.M q).rep σ = σ → ((D.M q).S σ σ).card ∣ 12) (v : D.V 0) (h6 : IsUnit (6 : k)) (n : ℕ) :
    (XBulk.relCx D hΓΔ ρ e).H n ≃ₗ[k] groupCohomology (Rep.of (ρLRep hΓΔ ρ e)) n :=
  rmkOrbifold D hA hdvd v h6 (Rep.of (ρLRep hΓΔ ρ e)) n

/-- `rmkOrbifoldL` on the 7-vertex model. -/
def model7_rmkOrbifoldL (h6 : IsUnit (6 : k)) (n : ℕ) :
    (XBulk.relCx model7 hΓΔ ρ e).H n ≃ₗ[k] groupCohomology (Rep.of (ρLRep hΓΔ ρ e)) n :=
  rmkOrbifoldL hΓΔ ρ e model7 model7_acyclic model7_dvd (M7.vx 0) h6 n

end General

end GL2.Model

end
