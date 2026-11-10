import GL2.Model.WellDef
import GL2.Model.RmkOrbifold

/-!
# M3-e (L19-2): `thm:welldef`(b) on the 7-vertex model -- `T_g` on cohomology does not depend on `τ`

[GT] `thm:welldef-b`, mirroring [C] v1.18.0 `thm:welldef`(b) (with `lem:tauexists`(2)), on the 7-vertex model, for
coefficients `V` over a ring `R` with `6 ∈ R^×`.

For two `g`-twisted chain data `τ, τ'` on `C_•(model7)` (over `ℤ`), the base changes `τ_R, τ'_R` are chain maps
`res_{f₁} (C_•(model7) ⊗ R) → res_{f₂} (C_•(model7) ⊗ R)` of representations of `Γ'' = Γ ∩ g Γ g⁻¹`
(`f₁` the inclusion, `f₂ = h ↦ g⁻¹ h g`), both over the augmentation (T3); since the restricted complexes are projective
resolutions (`resProjRes`, `model7_projRes`), `τ_R - τ'_R` is null-homotopic (`liftHomotopyZero`), by a homotopy `h`
that is `Γ''`-twisted (T1).  Transporting `h` through the coset sum gives, for every cocycle `z`,
`T̃_τ z - T̃_τ' z = d (h̃ z)`: the two Hecke operators agree on `H^n` (`heckeTilde_indep_tau`).

**F3-19 repairs.**
* MINOR 1 -- `τ` over `R`: [C] `def:tau` admits `τ` over `R`.  `TauR` is such a datum on the model ((T1) as morphisms of
  `Rep R Γ''`, (T2), (T3)), `heckeTildeR` its Hecke operator, `heckeTildeR_indep_tau` is (b) for it; a `τ` over `ℤ`,
  base-changed, is `TauR.ofTau` and `heckeTildeR_ofTau` identifies the operators (`heckeTilde_indep_tau'` re-derives the
  `ℤ` case).
* MINOR 4 -- `heckeRel_indep_tau` ((b) on the relative complex `H^n_eq(K, Rel; L)` itself, through `HmapOf_comp`);
  `CosetData.reindex` / `thetaG_indep_cd` / `heckeTilde_indep_cd` ((a) for any two coset data); `heckeTilde_indep_g`
  ((c) in general: any coset data and any `τ₁` for `g₁ = γ₁ g γ₂`).
-/

noncomputable section

namespace GL2.Model

open scoped MatrixGroups
open CategoryTheory GL2.Cone GL2.Cone.ProjRed MonoidAlgebra

namespace XBulk

section Tau

variable {G : Type} [Group G] {Γ : Subgroup G} (g : G)

/-- `Γ'' = Γ ∩ g Γ g⁻¹`, as a subgroup of `Γ`. -/
def Γpp : Subgroup Γ where
  carrier := {h | g⁻¹ * (h : G) * g ∈ Γ}
  one_mem' := by
    show g⁻¹ * ((1 : Γ) : G) * g ∈ Γ
    rw [Subgroup.coe_one, mul_one, inv_mul_cancel]; exact Γ.one_mem
  mul_mem' := by
    intro a b ha hb
    show g⁻¹ * ((a * b : Γ) : G) * g ∈ Γ
    have : g⁻¹ * ((a * b : Γ) : G) * g = (g⁻¹ * a * g) * (g⁻¹ * b * g) := by
      simp only [Subgroup.coe_mul]; group
    rw [this]; exact Γ.mul_mem ha hb
  inv_mem' := by
    intro a ha
    show g⁻¹ * ((a⁻¹ : Γ) : G) * g ∈ Γ
    have : g⁻¹ * ((a⁻¹ : Γ) : G) * g = (g⁻¹ * a * g)⁻¹ := by
      simp only [Subgroup.coe_inv]; group
    rw [this]; exact Γ.inv_mem ha

/-- `h ↦ g⁻¹ h g : Γ'' → Γ`. -/
def conjHom : Γpp (Γ := Γ) g →* Γ where
  toFun h := ⟨g⁻¹ * ((h : Γ) : G) * g, h.2⟩
  map_one' := by apply Subtype.ext; simp
  map_mul' a b := by apply Subtype.ext; simp only [MulMemClass.coe_mul]; group

lemma coe_conjHom (h : Γpp (Γ := Γ) g) : ((conjHom g h : Γ) : G) = g⁻¹ * ((h : Γ) : G) * g := rfl

variable (e : Γ ≃* SL(2, ℤ))

/-- `f₁ : Γ'' → SL(2, ℤ)`, the inclusion. -/
def fIncl : Γpp (Γ := Γ) g →* SL(2, ℤ) := e.toMonoidHom.comp (Γpp g).subtype

/-- `f₂ : Γ'' → SL(2, ℤ)`, `h ↦ g⁻¹ h g`. -/
def fConj : Γpp (Γ := Γ) g →* SL(2, ℤ) := e.toMonoidHom.comp (conjHom g)

lemma fIncl_injective : Function.Injective (fIncl g e) := by
  intro a b h
  exact Subtype.ext (e.injective h)

lemma fConj_injective : Function.Injective (fConj g e) := by
  intro a b h
  have h1 := congrArg Subtype.val (e.injective h)
  simp only [coe_conjHom] at h1
  apply Subtype.ext; apply Subtype.ext
  simpa using h1

end Tau

/-! ### A twisted chain datum, base-changed to the permutation modules -/

section TauR

variable (D : XBulkDatum) {G : Type} [Group G] {Γ : Subgroup G} {g : G} (e : Γ ≃* SL(2, ℤ)) (R : Type) [CommRing R]

/-- `τ` in degree `q`, as an additive map of `C_q(X_bulk)`. -/
def τc (τ : TwistedChainDatum Γ g (Yc D e)) (q : ℤ) : chains D q →+ chains D q := τ.τ q

lemma τc_twist (τ : TwistedChainDatum Γ g (Yc D e)) (q : ℤ) (h : Γpp (Γ := Γ) g) (y : chains D q) :
    τc D e τ q (fIncl g e h • y) = fConj g e h • τc D e τ q y :=
  τ.t1 q (h : Γ) (conjHom g h) (coe_conjHom g h) y

lemma τc_bd (τ : TwistedChainDatum Γ g (Yc D e)) (q : ℤ) (y : chains D (q + 1)) :
    bd D q (τc D e τ (q + 1) y) = τc D e τ q (bd D q y) := τ.t2 q y

lemma τc_aug (τ : TwistedChainDatum Γ g (Yc D e)) (y : chains D 0) : aug D (τc D e τ 0 y) = aug D y := τ.t3 y

/-- `τ` on `ℤ[Simp_q]`, through `chainsEquivPerm`. -/
def τZ (τ : TwistedChainDatum Γ g (Yc D e)) (q : ℤ) : (Simp D q →₀ ℤ) →ₗ[ℤ] (Simp D q →₀ ℤ) where
  toFun x := ofChains D q (τc D e τ q (toChains D q x))
  map_add' x y := by simp only [map_add]
  map_smul' n x := by
    simp only [map_zsmul, RingHom.id_apply]

lemma τZ_apply (τ : TwistedChainDatum Γ g (Yc D e)) (q : ℤ) (x : Simp D q →₀ ℤ) :
    τZ D e τ q x = ofChains D q (τc D e τ q (toChains D q x)) := rfl

lemma τZ_ofChains (τ : TwistedChainDatum Γ g (Yc D e)) (q : ℤ) (c : chains D q) :
    τZ D e τ q (ofChains D q c) = ofChains D q (τc D e τ q c) := by
  rw [τZ_apply, toChains_ofChains]

/-- (T1) on `ℤ[Simp_q]`. -/
lemma τZ_twist (τ : TwistedChainDatum Γ g (Yc D e)) (q : ℤ) (h : Γpp (Γ := Γ) g) (x : Simp D q →₀ ℤ) :
    τZ D e τ q (Finsupp.mapDomain (fIncl g e h • ·) x) = Finsupp.mapDomain (fConj g e h • ·) (τZ D e τ q x) := by
  rw [τZ_apply, τZ_apply, toChains_smul, τc_twist, ofChains_smul]

/-- (T2) on `ℤ[Simp_•]`. -/
lemma dZ_τZ (τ : TwistedChainDatum Γ g (Yc D e)) (q : ℤ) (x : Simp D (q + 1) →₀ ℤ) :
    dZ D q (τZ D e τ (q + 1) x) = τZ D e τ q (dZ D q x) := by
  simp only [dZ, LinearMap.coe_comp, Function.comp_apply, τZ_apply, toChains_ofChains, τc_bd]

/-- (T3) on `ℤ[Simp_0]`. -/
lemma augZ_τZ (τ : TwistedChainDatum Γ g (Yc D e)) (x : Simp D 0 →₀ ℤ) : augZ D (τZ D e τ 0 x) = augZ D x := by
  simp only [augZ, LinearMap.coe_comp, Function.comp_apply, τZ_apply, toChains_ofChains, τc_aug]

/-- Base change of a map intertwining two (possibly different) maps of the index sets. -/
lemma bcF_mapDomain' {S T : Type} {f : (S →₀ ℤ) →ₗ[ℤ] (T →₀ ℤ)} {a : S → S} {b : T → T}
    (hf : ∀ x, f (Finsupp.mapDomain a x) = Finsupp.mapDomain b (f x)) (x : S →₀ R) :
    bcF R f (Finsupp.mapDomain a x) = Finsupp.mapDomain b (bcF R f x) := by
  induction x using Finsupp.induction_linear with
  | zero => simp
  | add x y hx hy => rw [Finsupp.mapDomain_add, map_add, hx, hy, map_add, Finsupp.mapDomain_add]
  | single s r =>
    rw [Finsupp.mapDomain_single, bcF_single, bcF_single, Finsupp.mapDomain_smul]
    congr 1
    rw [← castF_mapDomain, ← hf, Finsupp.mapDomain_single]

/-- `τ_R` in degree `q`: a morphism `res_{f₁} R[Simp_q] → res_{f₂} R[Simp_q]` of representations of `Γ''`. -/
def τRep (τ : TwistedChainDatum Γ g (Yc D e)) (q : ℤ) :
    Rep.res (fIncl g e) (Rep.ofMulAction R SL(2, ℤ) (Simp D q)) ⟶
      Rep.res (fConj g e) (Rep.ofMulAction R SL(2, ℤ) (Simp D q)) :=
  Rep.ofHom ⟨(coeffLinearEquiv R).symm.toLinearMap ∘ₗ bcF R (τZ D e τ q) ∘ₗ (coeffLinearEquiv R).toLinearMap,
    fun h => by
      refine LinearMap.ext fun x => ?_
      simp only [LinearMap.coe_comp, Function.comp_apply, LinearEquiv.coe_coe, MonoidHom.coe_comp,
        Representation.ofMulAction_def, LinearEquiv.apply_symm_apply, Finsupp.lmapDomain_apply]
      rw [bcF_mapDomain' R (τZ_twist D e τ q h)]⟩

lemma τRep_hom_apply (τ : TwistedChainDatum Γ g (Yc D e)) (q : ℤ) (x : R[Simp D q]) :
    (τRep D e R τ q).hom x = (coeffLinearEquiv R).symm (bcF R (τZ D e τ q) (coeffLinearEquiv R x)) := rfl

end TauR

/-! ### On the model: the restricted resolutions, and `τ_R - τ'_R ≃ 0` -/

section Model

variable {G : Type} [Group G] {Γ : Subgroup G} (g : G) (e : Γ ≃* SL(2, ℤ)) {R : Type} [CommRing R]
  (h6 : IsUnit (6 : R))

/-- `res_{f₁} (C_•(model7) ⊗ R)`, a projective resolution of `R` over `Γ''`. -/
abbrev P₁ := resProjRes (fIncl g e) (fIncl_injective g e) (model7_projRes h6) (fun n => Simp model7 n)
  (fun _ => Iso.refl _) (fun n => finite_stab model7 n) (fun n => isUnit_card_stab model7 h6 model7_dvd n)

/-- `res_{f₂} (C_•(model7) ⊗ R)`, a projective resolution of `R` over `Γ''`. -/
abbrev P₂ := resProjRes (fConj g e) (fConj_injective g e) (model7_projRes h6) (fun n => Simp model7 n)
  (fun _ => Iso.refl _) (fun n => finite_stab model7 n) (fun n => isUnit_card_stab model7 h6 model7_dvd n)

lemma P₁_d (j : ℕ) : (P₁ g e h6).complex.d (j + 1) j = (Rep.resFunctor (fIncl g e)).map (repD model7 R j) := by
  show (Rep.resFunctor (fIncl g e)).map ((repCx model7 R).d (j + 1) j) = _
  exact congrArg _ (repCx_d model7 R j)

lemma P₂_d (j : ℕ) : (P₂ g e h6).complex.d (j + 1) j = (Rep.resFunctor (fConj g e)).map (repD model7 R j) := by
  show (Rep.resFunctor (fConj g e)).map ((repCx model7 R).d (j + 1) j) = _
  exact congrArg _ (repCx_d model7 R j)

variable {g e}

/-- **`τ_R`**: a `g`-twisted chain datum on `C_•(model7)` as a chain map `res_{f₁} → res_{f₂}` over `Γ''`. -/
def τCx (τ : TwistedChainDatum Γ g (Yc model7 e)) : (P₁ g e h6).complex ⟶ (P₂ g e h6).complex where
  f n := τRep model7 e R τ n
  comm' := by
    rintro i j (rfl : j + 1 = i)
    rw [P₁_d, P₂_d]
    apply Rep.hom_ext; apply Representation.IntertwiningMap.ext; refine LinearMap.ext fun x => ?_
    show (coeffLinearEquiv R).symm (bcF R (dZ model7 j) (coeffLinearEquiv R ((coeffLinearEquiv R).symm
      (bcF R (τZ model7 e τ (j + 1 : ℕ)) (coeffLinearEquiv R x))))) =
      (coeffLinearEquiv R).symm (bcF R (τZ model7 e τ j) (coeffLinearEquiv R ((coeffLinearEquiv R).symm
      (bcF R (dZ model7 j) (coeffLinearEquiv R x)))))
    rw [LinearEquiv.apply_symm_apply, LinearEquiv.apply_symm_apply, ← LinearMap.comp_apply, ← LinearMap.comp_apply,
      ← bcF_comp, ← bcF_comp]
    congr 3
    exact LinearMap.ext fun y => dZ_τZ model7 e τ j y

lemma eps_τRep (τ : TwistedChainDatum Γ g (Yc model7 e)) (x : R[Simp model7 0]) :
    (repEps model7 R).hom ((τRep model7 e R τ 0).hom x) = (repEps model7 R).hom x := by
  show bcA R (augZ model7) (coeffLinearEquiv R ((coeffLinearEquiv R).symm
    (bcF R (τZ model7 e τ 0) (coeffLinearEquiv R x)))) = bcA R (augZ model7) (coeffLinearEquiv R x)
  rw [LinearEquiv.apply_symm_apply, ← LinearMap.comp_apply, ← bcA_comp]
  congr 2
  exact LinearMap.ext fun y => augZ_τZ model7 e τ y

lemma eps_τRep_sub (τ τ' : TwistedChainDatum Γ g (Yc model7 e)) (x : R[Simp model7 0]) :
    (repEps model7 R).hom ((τRep model7 e R τ 0).hom x - (τRep model7 e R τ' 0).hom x) = 0 := by
  rw [map_sub, eps_τRep, eps_τRep, sub_self]

/-- `τ_R - τ'_R` lies over the zero map of `R`. -/
lemma τCx_sub_comm (τ τ' : TwistedChainDatum Γ g (Yc model7 e)) :
    (τCx h6 τ - τCx h6 τ') ≫ (P₂ g e h6).π = 0 := by
  apply HomologicalComplex.to_single_hom_ext
  rw [HomologicalComplex.comp_f, HomologicalComplex.zero_f]
  simp only [P₂, resProjRes, HomologicalComplex.comp_f, Functor.mapHomologicalComplex_map_f]
  have key : (τCx h6 τ - τCx h6 τ').f 0 ≫ Rep.resMap (fConj g e) ((model7_projRes h6).π.f 0) = 0 := by
    apply Rep.hom_ext; apply Representation.IntertwiningMap.ext; refine LinearMap.ext fun x => ?_
    change ((repAug model7 R).f 0).hom ((τRep model7 e R τ 0).hom x - (τRep model7 e R τ' 0).hom x) = 0
    rw [repAug_f_zero]
    exact eps_τRep_sub τ τ' x
  exact (Category.assoc _ _ _).symm.trans ((congrArg (· ≫ _) key).trans Limits.zero_comp)

/-- **The twisted homotopy** `τ_R - τ'_R = ∂ h + h ∂` (`lem:tauexists`(2) on the model, applied). -/
def htpy (τ τ' : TwistedChainDatum Γ g (Yc model7 e)) : Homotopy (τCx h6 τ - τCx h6 τ') 0 :=
  ProjectiveResolution.liftHomotopyZero _ (τCx_sub_comm h6 τ τ')

/-- The homotopy in degree `n`, `R[Simp_n] → R[Simp_{n+1}]`. -/
def hL (τ τ' : TwistedChainDatum Γ g (Yc model7 e)) (n : ℕ) : R[Simp model7 n] → R[Simp model7 (n + 1 : ℕ)] :=
  fun x => ((htpy h6 τ τ').hom n (n + 1)).hom x

lemma hL_add (τ τ' : TwistedChainDatum Γ g (Yc model7 e)) (n : ℕ) (x y : R[Simp model7 n]) :
    hL h6 τ τ' n (x + y) = hL h6 τ τ' n x + hL h6 τ τ' n y := map_add _ x y

lemma hL_smul (τ τ' : TwistedChainDatum Γ g (Yc model7 e)) (n : ℕ) (r : R) (x : R[Simp model7 n]) :
    hL h6 τ τ' n (r • x) = r • hL h6 τ τ' n x := map_smul _ r x

/-- (T1) for the homotopy. -/
lemma hL_twist (τ τ' : TwistedChainDatum Γ g (Yc model7 e)) (n : ℕ) (h : Γpp (Γ := Γ) g) (x : R[Simp model7 n]) :
    hL h6 τ τ' n (Representation.ofMulAction R SL(2, ℤ) (Simp model7 n) (fIncl g e h) x) =
      Representation.ofMulAction R SL(2, ℤ) (Simp model7 (n + 1 : ℕ)) (fConj g e h) (hL h6 τ τ' n x) :=
  Rep.hom_comm_apply ((htpy h6 τ τ').hom n (n + 1)) h x

lemma htpy_zero (τ τ' : TwistedChainDatum Γ g (Yc model7 e)) (x : R[Simp model7 0]) :
    (τRep model7 e R τ 0).hom x - (τRep model7 e R τ' 0).hom x = (repD model7 R 0).hom (hL h6 τ τ' 0 x) := by
  have hc := (htpy h6 τ τ').comm 0
  rw [dNext_eq_zero _ 0 (by simp), prevD_eq _ (show (ComplexShape.down ℕ).Rel 1 0 from rfl), P₂_d] at hc
  simp only [zero_add, add_zero, HomologicalComplex.zero_f] at hc
  have := congrArg (fun φ => (Rep.Hom.hom φ) x) hc
  exact this

lemma htpy_succ (τ τ' : TwistedChainDatum Γ g (Yc model7 e)) (n : ℕ) (x : R[Simp model7 (n + 1 : ℕ)]) :
    (τRep model7 e R τ (n + 1 : ℕ)).hom x - (τRep model7 e R τ' (n + 1 : ℕ)).hom x =
      hL h6 τ τ' n ((repD model7 R n).hom x) + (repD model7 R (n + 1)).hom (hL h6 τ τ' (n + 1) x) := by
  have hc := (htpy h6 τ τ').comm (n + 1)
  rw [dNext_eq _ (show (ComplexShape.down ℕ).Rel (n + 1) n from rfl),
    prevD_eq _ (show (ComplexShape.down ℕ).Rel (n + 1 + 1) (n + 1) from rfl), P₁_d, P₂_d] at hc
  simp only [add_zero, HomologicalComplex.zero_f] at hc
  have := congrArg (fun φ => (Rep.Hom.hom φ) x) hc
  exact this

end Model

/-! ### Back on the cochains: the linear extension of a cochain, and the embedding `C_q → R[Simp_q]` -/

section Cochain

variable {D : XBulkDatum} {R : Type} [CommRing R] {V : Type} [AddCommGroup V] [Module R V]

/-- `C_q(X_bulk) → R[Simp_q]`, `y ↦ chainsEquivPerm y ⊗ 1`. -/
def embR (q : ℤ) (y : chains D q) : R[Simp D q] := (coeffLinearEquiv R).symm (castF (ofChains D q y))

lemma embR_add (q : ℤ) (y y' : chains D q) : embR (R := R) q (y + y') = embR q y + embR q y' := by
  simp only [embR, map_add]

/-- The `R`-linear extension `R[Simp_q] → V` of an additive map `φ : C_q(X_bulk) → V`. -/
def linR {q : ℤ} (φ : chains D q →+ V) : R[Simp D q] →ₗ[R] V :=
  Finsupp.linearCombination R (fun s => φ (orient D q s)) ∘ₗ (coeffLinearEquiv R).toLinearMap

lemma linR_zero {q : ℤ} : linR (R := R) (0 : chains D q →+ V) = 0 := by
  ext x
  simp [linR, Finsupp.linearCombination_apply]

lemma linR_castF {q : ℤ} (φ : chains D q →+ V) (f : Simp D q →₀ ℤ) :
    linR (R := R) φ ((coeffLinearEquiv R).symm (castF f)) = φ (toChains D q f) := by
  simp only [linR, LinearMap.coe_comp, Function.comp_apply, LinearEquiv.coe_coe, LinearEquiv.apply_symm_apply]
  induction f using Finsupp.induction_linear with
  | zero => simp
  | add f g hf hg => rw [map_add, map_add, hf, hg, map_add, map_add]
  | single s n =>
    rw [castF_single, Finsupp.linearCombination_single, toChains_single, map_zsmul, Int.cast_smul_eq_zsmul]

lemma linR_embR {q : ℤ} (φ : chains D q →+ V) (y : chains D q) : linR (R := R) φ (embR q y) = φ y := by
  rw [embR, linR_castF, toChains_ofChains]

lemma embR_bd (q : ℤ) (y : chains D (q + 1)) :
    embR (R := R) q (bd D q y) = (coeffLinearEquiv R).symm (bcF R (dZ D q) (coeffLinearEquiv R (embR (q + 1) y))) := by
  rw [embR, embR, LinearEquiv.apply_symm_apply, bcF_castF]
  congr 2
  simp only [dZ, LinearMap.coe_comp, Function.comp_apply, toChains_ofChains]

lemma linR_bcF {q : ℤ} (φ : chains D q →+ V) (x : R[Simp D (q + 1)]) :
    linR (R := R) φ ((coeffLinearEquiv R).symm (bcF R (dZ D q) (coeffLinearEquiv R x))) =
      linR (φ.comp (bd D q).toAddMonoidHom) x := by
  conv_rhs => rw [← (coeffLinearEquiv R).symm_apply_apply x]
  simp only [linR, LinearMap.coe_comp, Function.comp_apply, LinearEquiv.coe_coe, LinearEquiv.apply_symm_apply]
  induction (coeffLinearEquiv R) x using Finsupp.induction_linear with
  | zero => simp
  | add a b ha hb => rw [map_add, map_add, map_add, ha, hb]
  | single s r =>
    rw [bcF_single, map_smul, Finsupp.linearCombination_single]
    have := linR_castF (R := R) φ (dZ D q (Finsupp.single s 1))
    simp only [linR, LinearMap.coe_comp, Function.comp_apply, LinearEquiv.coe_coe,
      LinearEquiv.apply_symm_apply] at this
    rw [this]
    simp only [AddMonoidHom.comp_apply, LinearMap.toAddMonoidHom_coe, dZ, LinearMap.coe_comp, Function.comp_apply,
      toChains_ofChains, toChains_single, one_smul]

lemma embR_smul {q : ℤ} (t : SL(2, ℤ)) (y : chains D q) :
    embR (R := R) q (t • y) = Representation.ofMulAction R SL(2, ℤ) (Simp D q) t (embR q y) := by
  simp only [embR, ofChains_smul, castF_mapDomain, Representation.ofMulAction_def, LinearMap.coe_comp,
    Function.comp_apply, LinearEquiv.coe_coe, LinearEquiv.apply_symm_apply, Finsupp.lmapDomain_apply]

end Cochain

/-! ### The cochain homotopy and `thm:welldef`(b) -/

section WelldefB

variable {G : Type} [Group G] {Γ : Subgroup G} {Δ : Submonoid G} {hΓΔ : ∀ γ : Γ, (γ : G) ∈ Δ}
  {R : Type} [CommRing R] {V : Type} [AddCommGroup V] [Module R V] (ρ : Δ →* Module.End R V)
  {e : Γ ≃* SL(2, ℤ)} {g : G} (h6 : IsUnit (6 : R))

variable (R) (e) in
/-- An auxiliary `Γ`-complex with terms `R[Simp_q]` (`Γ` acting through `e`); its boundary and augmentation are `0` and
are never used: it only carries the `Γ`-action for the coset transport of the homotopy. -/
def permCx (D : XBulkDatum) : GChainCx.{0} Γ where
  X q := R[Simp D q]
  act q :=
    { smul := fun γ x => Representation.ofMulAction R SL(2, ℤ) (Simp D q) (e γ) x
      one_smul := fun x => by
        show Representation.ofMulAction R SL(2, ℤ) (Simp D q) (e 1) x = x
        rw [map_one, map_one]; rfl
      mul_smul := fun a b x => by
        show Representation.ofMulAction R SL(2, ℤ) (Simp D q) (e (a * b)) x =
          Representation.ofMulAction R SL(2, ℤ) (Simp D q) (e a) (Representation.ofMulAction R SL(2, ℤ) (Simp D q) (e b) x)
        rw [map_mul, map_mul]; rfl
      smul_zero := fun a => map_zero _
      smul_add := fun a x y => map_add _ x y }
  bd _ := 0
  bd_equiv _ _ _ := by simp
  bd_bd _ _ := rfl
  aug := 0
  aug_bd _ := rfl
  aug_equiv _ _ := rfl

lemma linR_mem {D : XBulkDatum} {q : ℤ} (φ : (EqC D (hΓΔ := hΓΔ) ρ e).X q) :
    (linR (R := R) φ.1).toAddMonoidHom ∈ eqHom Γ Δ hΓΔ ρ (permCx R e D) q := by
  intro γ x
  show linR (R := R) φ.1 (Representation.ofMulAction R SL(2, ℤ) (Simp D q) (e γ) x) =
    ρΓ Γ Δ hΓΔ ρ γ (linR (R := R) φ.1 x)
  obtain ⟨f, rfl⟩ := (coeffLinearEquiv R).symm.surjective x
  simp only [linR, LinearMap.coe_comp, Function.comp_apply, LinearEquiv.coe_coe, Representation.ofMulAction_def,
    LinearEquiv.apply_symm_apply, Finsupp.lmapDomain_apply]
  induction f using Finsupp.induction_linear with
  | zero => simp
  | add a b ha hb => rw [Finsupp.mapDomain_add, map_add, map_add, ha, hb, map_add]
  | single s r =>
    rw [Finsupp.mapDomain_single, Finsupp.linearCombination_single, Finsupp.linearCombination_single, orient_smul,
      map_smul]
    exact congrArg (r • ·) (φ.2 γ (orient D q s))

lemma embR_τ (τ : TwistedChainDatum Γ g (Yc model7 e)) (q : ℤ) (c : chains model7 q) :
    embR (R := R) q (τc model7 e τ q c) = (τRep model7 e R τ q).hom (embR q c) := by
  show _ = (coeffLinearEquiv R).symm (bcF R (τZ model7 e τ q) (coeffLinearEquiv R ((coeffLinearEquiv R).symm
    (castF (ofChains model7 q c)))))
  rw [LinearEquiv.apply_symm_apply, bcF_castF, τZ_ofChains]
  rfl

variable (e) in
/-- The homotopy in degree `m`, precomposed with `C_m → R[Simp_m]`, as an additive map into `permCx`. -/
def hF (τ τ' : TwistedChainDatum Γ g (Yc model7 e)) (m : ℕ) :
    (Yc model7 e).X m →+ (permCx R e model7).X ((m : ℤ) + 1) where
  toFun y := hL h6 τ τ' m (embR m y)
  map_zero' := by
    show hL h6 τ τ' m (embR m 0) = 0
    rw [show embR (R := R) (D := model7) m 0 = 0 by simp [embR]]
    exact map_zero ((htpy h6 τ τ').hom m (m + 1)).hom
  map_add' y y' := by
    exact (congrArg (hL h6 τ τ' m) (embR_add (R := R) (D := model7) (m : ℤ) (y : chains model7 m) y')).trans
      (hL_add h6 τ τ' m _ _)

lemma hF_t1 (τ τ' : TwistedChainDatum Γ g (Yc model7 e)) (m : ℕ) : T1 Γ g (hF e h6 τ τ' m) := by
  intro h h' hh (x : chains model7 m)
  have hmem : g⁻¹ * (h : G) * g ∈ Γ := hh ▸ h'.2
  let hp : Γpp (Γ := Γ) g := ⟨h, hmem⟩
  have hc : conjHom g hp = h' := Subtype.ext hh.symm
  show hL h6 τ τ' m (embR m (e h • x)) =
    Representation.ofMulAction R SL(2, ℤ) (Simp model7 ((m : ℤ) + 1)) (e h') (hL h6 τ τ' m (embR m x))
  rw [embR_smul]
  have := hL_twist h6 τ τ' m hp (embR m x)
  rw [show fConj g e hp = e h' from congrArg e hc] at this
  exact this

variable (e) in
/-- **The cochain homotopy** `h̃ ψ = ∑_j ρ(δ_j) ψ_R(h(s_j ·))`, from `C^{m+1}_eq` to `C^m_eq`. -/
def hTilde (cd : CosetData Γ Δ g) (τ τ' : TwistedChainDatum Γ g (Yc model7 e)) (m : ℕ)
    (ψ : (EqC model7 (hΓΔ := hΓΔ) ρ e).X ((m : ℤ) + 1)) : (EqC model7 (hΓΔ := hΓΔ) ρ e).X m :=
  ⟨transport (ρ := ρ) Γ Δ cd (hF e h6 τ τ' m) (linR (R := R) ψ.1).toAddMonoidHom,
    (transport_equivariant (hΓΔ := hΓΔ) (Y := Yc model7 e) (Y' := permCx R e model7) cd (hF e h6 τ τ' m)
      (hF_t1 h6 τ τ' m) _ (linR_mem ρ ψ)).1⟩

/-- The termwise identity: for an additive `φ` with `φ ∘ ∂ = 0`, `φ(τ c) - φ(τ' c) = φ_R(h(∂ c))`. -/
lemma cocycle_diff (τ τ' : TwistedChainDatum Γ g (Yc model7 e)) (m : ℕ) (φ : chains model7 ((m : ℤ) + 1) →+ V)
    (hφ : φ.comp (bd model7 ((m : ℤ) + 1)).toAddMonoidHom = 0) (c : chains model7 ((m : ℤ) + 1)) :
    φ (τc model7 e τ _ c) - φ (τc model7 e τ' _ c) = linR (R := R) φ (hL h6 τ τ' m (embR m (bd model7 m c))) := by
  rw [← linR_embR (R := R) φ (τc model7 e τ _ c), ← linR_embR (R := R) φ (τc model7 e τ' _ c), embR_τ, embR_τ,
    ← map_sub]
  have := congrArg (linR (R := R) φ) (htpy_succ h6 τ τ' m (embR ((m + 1 : ℕ) : ℤ) c))
  refine this.trans ?_
  rw [map_add, embR_bd]
  have h2 : linR (R := R) φ ((repD model7 R (m + 1)).hom (hL h6 τ τ' (m + 1) (embR ((m + 1 : ℕ) : ℤ) c))) = 0 := by
    show linR (R := R) φ ((coeffLinearEquiv R).symm (bcF R (dZ model7 ((m : ℤ) + 1))
      (coeffLinearEquiv R (hL h6 τ τ' (m + 1) (embR ((m + 1 : ℕ) : ℤ) c))))) = 0
    rw [linR_bcF, hφ, linR_zero, LinearMap.zero_apply]
  rw [h2, add_zero]
  rfl

include h6 in
/-- The degree-`0` identity: for an additive `φ` with `φ ∘ ∂ = 0`, `φ(τ c) = φ(τ' c)`. -/
lemma cocycle_diff_zero (τ τ' : TwistedChainDatum Γ g (Yc model7 e)) (φ : chains model7 0 →+ V)
    (hφ : φ.comp (bd model7 0).toAddMonoidHom = 0) (c : chains model7 0) :
    φ (τc model7 e τ 0 c) - φ (τc model7 e τ' 0 c) = 0 := by
  rw [← linR_embR (R := R) φ (τc model7 e τ _ c), ← linR_embR (R := R) φ (τc model7 e τ' _ c), embR_τ, embR_τ,
    ← map_sub]
  have := htpy_zero h6 τ τ' (embR 0 c)
  rw [this]
  show linR (R := R) φ ((coeffLinearEquiv R).symm (bcF R (dZ model7 0)
      (coeffLinearEquiv R (hL h6 τ τ' 0 (embR 0 c))))) = 0
  rw [linR_bcF, hφ, linR_zero, LinearMap.zero_apply]

include h6 in
/-- **`heckeTilde_indep_tau`** ([C] `thm:welldef`(b) on the 7-vertex model, `6 ∈ R^×`): for two `g`-twisted chain
data `τ, τ'` on `C_•(model7)` and the same coset data, the Hecke operators `T̃_τ` and `T̃_τ'` induce the same map on
`H^n(C^•_eq)` for every `n ≥ 0`. -/
theorem heckeTilde_indep_tau (cd : CosetData Γ Δ g) (τ τ' : TwistedChainDatum Γ g (Yc model7 e)) (n : ℕ) :
    HmapOf (heckeTilde model7 ρ e (hΓΔ := hΓΔ) cd τ) (heckeTilde_isChainMap model7 ρ e cd τ) n =
      HmapOf (heckeTilde model7 ρ e (hΓΔ := hΓΔ) cd τ') (heckeTilde_isChainMap model7 ρ e cd τ') n := by
  apply LinearMap.ext; intro c
  induction c using Submodule.Quotient.induction_on with
  | _ z =>
    rw [HmapOf_mk, HmapOf_mk, CochainCx.mk_eq_mk_iff]
    have hz : (EqC model7 (hΓΔ := hΓΔ) ρ e).d n z.1 = 0 := ((EqC model7 (hΓΔ := hΓΔ) ρ e).mem_Z _ _).1 z.2
    have hφ : z.1.1.comp (bd model7 n).toAddMonoidHom = 0 := congrArg Subtype.val hz
    have hT : ∀ y : chains model7 n, (heckeTilde model7 ρ e (hΓΔ := hΓΔ) cd τ n z.1).1 y -
        (heckeTilde model7 ρ e (hΓΔ := hΓΔ) cd τ' n z.1).1 y =
        ∑ i, ρ ⟨(cd.s i : G)⁻¹ * g, cd.δ_mem i⟩ (z.1.1 (τc model7 e τ n (e (cd.s i) • y)) -
          z.1.1 (τc model7 e τ' n (e (cd.s i) • y))) := by
      intro y
      refine (congrArg₂ (· - ·) (heckeTilde_apply model7 ρ e cd τ n z.1 y)
        (heckeTilde_apply model7 ρ e cd τ' n z.1 y)).trans ?_
      rw [← Finset.sum_sub_distrib]
      exact Finset.sum_congr rfl fun i _ => (map_sub _ _ _).symm
    cases n with
    | zero =>
      have h0 : heckeTilde model7 ρ e (hΓΔ := hΓΔ) cd τ 0 z.1 - heckeTilde model7 ρ e (hΓΔ := hΓΔ) cd τ' 0 z.1 = 0 := by
        apply Subtype.ext; apply AddMonoidHom.ext; intro y
        refine (hT y).trans (Finset.sum_eq_zero fun i _ => ?_)
        exact (congrArg _ (cocycle_diff_zero h6 τ τ' z.1.1 hφ _)).trans (map_zero _)
      exact h0 ▸ Submodule.zero_mem _
    | succ m =>
      show (heckeTilde model7 ρ e (hΓΔ := hΓΔ) cd τ ((m : ℤ) + 1) z.1 -
        heckeTilde model7 ρ e (hΓΔ := hΓΔ) cd τ' ((m : ℤ) + 1) z.1) ∈ (EqC model7 (hΓΔ := hΓΔ) ρ e).B ((m : ℤ) + 1)
      rw [CochainCx.mem_B_succ]
      refine ⟨hTilde ρ e h6 cd τ τ' m z.1, ?_⟩
      apply Subtype.ext; apply AddMonoidHom.ext; intro y
      have hd : ((EqC model7 (hΓΔ := hΓΔ) ρ e).d m (hTilde ρ e h6 cd τ τ' m z.1)).1 y =
          ∑ i, ρ ⟨(cd.s i : G)⁻¹ * g, cd.δ_mem i⟩
            (linR (R := R) z.1.1 (hL h6 τ τ' m (embR m (e (cd.s i) • bd model7 m y)))) := by
        show transport (ρ := ρ) Γ Δ cd (hF e h6 τ τ' m) (linR (R := R) z.1.1).toAddMonoidHom (bd model7 m y) = _
        simp only [transport]
        exact (AddMonoidHom.finsetSum_apply _ _ _).trans rfl
      refine hd.trans (Eq.trans (Finset.sum_congr rfl fun i _ => ?_) (hT y).symm)
      exact congrArg _ ((congrArg (fun c => linR (R := R) z.1.1 (hL h6 τ τ' m (embR m c)))
        (bd_equiv model7 m (e (cd.s i)) y).symm).trans (cocycle_diff h6 τ τ' m z.1.1 hφ _).symm)

end WelldefB

end XBulk

/-! ### F3-19 MINOR 4: (a) for any two coset data, `HmapOf` of composites, and `thm:welldef`(c) in general -/

section Generic

variable {G : Type} [Group G] {Γ : Subgroup G} {Δ : Submonoid G} {hΓΔ : ∀ γ : Γ, (γ : G) ∈ Δ}
  {R : Type*} [CommRing R] {V : Type} [AddCommGroup V] [Module R V] (ρ : Δ →* Module.End R V) {g : G}

/-- **Two coset data for `Γ g Γ` differ by a re-indexing and left factors in `Γ''`**: for `cd, cd'` there are a bijection
`π` and `h_i ∈ Γ'' = Γ ∩ g Γ g⁻¹` with `s'_{π i} = h_i s_i` (the hypotheses of `thetaG_indep_reps`). -/
theorem CosetData.reindex (cd cd' : CosetData Γ Δ g) :
    ∃ (π : Fin cd.m ≃ Fin cd'.m) (h : Fin cd.m → Γ), (∀ i, g⁻¹ * (h i : G) * g ∈ Γ) ∧
      ∀ i, cd'.s (π i) = h i * cd.s i := by
  classical
  have hc : ∀ i, ∃ j, ∃ h : Γ, g⁻¹ * (h : G) * g ∈ Γ ∧ ((cd.s i : Γ) : G) = h * cd'.s j :=
    fun i => (cd'.cosets (cd.s i)).exists
  choose f hh hhmem hheq using hc
  have finj : Function.Injective f := by
    intro i i' hii
    have hmem : g⁻¹ * ((hh i * (hh i')⁻¹ : Γ) : G) * g ∈ Γ := by
      have := Γ.mul_mem (hhmem i) (Γ.inv_mem (hhmem i'))
      convert this using 1
      simp only [Subgroup.coe_mul, Subgroup.coe_inv]; group
    have hsi : ((cd.s i : Γ) : G) = ((hh i * (hh i')⁻¹ : Γ) : G) * cd.s i' := by
      have h2 := hheq i'
      rw [← hii] at h2
      rw [hheq i, h2]
      simp only [Subgroup.coe_mul, Subgroup.coe_inv]; group
    exact (cd.cosets (cd.s i)).unique ⟨1, by simp, by simp⟩ ⟨_, hmem, hsi⟩
  have fsurj : Function.Surjective f := by
    intro j
    obtain ⟨i, ⟨h, hmem, hj⟩, -⟩ := cd.cosets (cd'.s j)
    refine ⟨i, (cd'.cosets (cd'.s j)).unique ⟨h * hh i, ?_, ?_⟩ ⟨1, by simp, by simp⟩⟩
    · have := Γ.mul_mem hmem (hhmem i)
      convert this using 1
      simp only [Subgroup.coe_mul]; group
    · rw [hj, hheq i]
      simp only [Subgroup.coe_mul]; group
  refine ⟨Equiv.ofBijective f ⟨finj, fsurj⟩, fun i => (hh i)⁻¹, fun i => ?_, fun i => ?_⟩
  · have := Γ.inv_mem (hhmem i)
    convert this using 1
    simp only [Subgroup.coe_inv]; group
  · apply Subtype.ext
    show ((cd'.s (f i) : Γ) : G) = (((hh i)⁻¹ : Γ) : G) * cd.s i
    rw [hheq i]
    simp only [Subgroup.coe_inv]; group

/-- **`thm:welldef`(a)** at cochain level, `τ` fixed, for **any** two coset data of `Γ g Γ` (no hypothesis on `ℓ`). -/
theorem thetaG_indep_cd {Y : GChainCx.{0} Γ} (cd cd' : CosetData Γ Δ g) (τ : TwistedChainDatum Γ g Y) :
    thetaG (hΓΔ := hΓΔ) (ρ := ρ) cd' τ = thetaG cd τ := by
  obtain ⟨π, h, hh, hs⟩ := CosetData.reindex cd cd'
  exact thetaG_indep_reps ρ cd cd' τ π h hh hs

end Generic

section HmapOfLemmas

variable {R : Type*} [CommRing R]

lemma isChainMap_comp {A B C : CochainCx.{0} R} {f : GMap B C} {f' : GMap A B} (hf : IsChainMap f)
    (hf' : IsChainMap f') : IsChainMap (fun q => (f q).comp (f' q)) := by
  intro q
  apply LinearMap.ext; intro x
  simp only [LinearMap.comp_apply]
  rw [hf.apply, hf'.apply]

/-- `H^q` of a composite is the composite of the `H^q`. -/
lemma HmapOf_comp {A B C : CochainCx.{0} R} (f : GMap B C) (f' : GMap A B) (hf : IsChainMap f)
    (hf' : IsChainMap f') (hff' : IsChainMap (fun q => (f q).comp (f' q))) (q : ℤ) :
    HmapOf (fun q => (f q).comp (f' q)) hff' q = (HmapOf f hf q).comp (HmapOf f' hf' q) := by
  apply LinearMap.ext; intro c
  induction c using Submodule.Quotient.induction_on with
  | _ z => rfl

/-- `HmapOf` sees only the graded map. -/
lemma HmapOf_congr {A B : CochainCx.{0} R} {f f' : GMap A B} (h : f = f') (hf : IsChainMap f) (hf' : IsChainMap f')
    (q : ℤ) : HmapOf f hf q = HmapOf f' hf' q := by
  subst h; rfl

end HmapOfLemmas

namespace XBulk

/-! ### `thm:welldef`(b) for `τ` over `R` (F3-19 MINOR 1)

[C] `def:hecke` / `def:tau` admit a twisted chain datum `τ` **over `R`**: an `R`-linear chain map
`C_•(X_bulk) ⊗ R → C_•(X_bulk) ⊗ R` with (T1)-(T3).  On the model this is `TauR`: a chain map
`res_{f₁}(C_•(model7) ⊗ R) → res_{f₂}(C_•(model7) ⊗ R)` of representations of `Γ''` ((T1) is being a morphism in
`Rep R Γ''`, (T2) a chain map) over the augmentation ((T3)).  `heckeTildeR` is `T̃ : φ ↦ ∑_j ρ(δ_j) φ_R(τ_R(s_j ·))`;
`heckeTildeR_ofTau` identifies it with `heckeTilde` for a `τ` over `ℤ`, base-changed; `heckeTildeR_indep_tau` is (b). -/

section TauOverR

variable {G : Type} [Group G] {Γ : Subgroup G} {Δ : Submonoid G} {hΓΔ : ∀ γ : Γ, (γ : G) ∈ Δ}
  {R : Type} [CommRing R] {V : Type} [AddCommGroup V] [Module R V] (ρ : Δ →* Module.End R V)

variable (g : G) (e : Γ ≃* SL(2, ℤ)) (R) in
/-- **[C] `def:tau` over `R`** on the model: maps `τ_q : res_{f₁} R[Simp_q] → res_{f₂} R[Simp_q]` of representations of
`Γ''` ((T1): morphisms of `Rep R Γ''`), commuting with `∂` ((T2)) and with the augmentation ((T3)).  `R[Simp_q]` is
`C_q(model7) ⊗ R` (`chainsEquivPerm`). -/
structure TauR where
  f : ∀ n : ℕ, Rep.res (fIncl g e) (Rep.ofMulAction R SL(2, ℤ) (Simp model7 n)) ⟶
    Rep.res (fConj g e) (Rep.ofMulAction R SL(2, ℤ) (Simp model7 n))
  t2 : ∀ (n : ℕ) (x : R[Simp model7 (n + 1 : ℕ)]),
    (repD model7 R n).hom ((f (n + 1)).hom x) = (f n).hom ((repD model7 R n).hom x)
  t3 : ∀ x : R[Simp model7 0], (repEps model7 R).hom ((f 0).hom x) = (repEps model7 R).hom x

variable {g : G} {e : Γ ≃* SL(2, ℤ)}

/-- (T2) for the base change `τRep` of a `τ` over `ℤ`. -/
lemma τRep_t2 (τ : TwistedChainDatum Γ g (Yc model7 e)) (n : ℕ) (x : R[Simp model7 (n + 1 : ℕ)]) :
    (repD model7 R n).hom ((τRep model7 e R τ (n + 1 : ℕ)).hom x) = (τRep model7 e R τ n).hom ((repD model7 R n).hom x) := by
  show (coeffLinearEquiv R).symm (bcF R (dZ model7 n) (coeffLinearEquiv R ((coeffLinearEquiv R).symm
    (bcF R (τZ model7 e τ (n + 1 : ℕ)) (coeffLinearEquiv R x))))) =
    (coeffLinearEquiv R).symm (bcF R (τZ model7 e τ n) (coeffLinearEquiv R ((coeffLinearEquiv R).symm
    (bcF R (dZ model7 n) (coeffLinearEquiv R x)))))
  rw [LinearEquiv.apply_symm_apply, LinearEquiv.apply_symm_apply, ← LinearMap.comp_apply, ← LinearMap.comp_apply,
    ← bcF_comp, ← bcF_comp]
  congr 3
  exact LinearMap.ext fun y => dZ_τZ model7 e τ n y

variable (R) in
/-- A `g`-twisted chain datum over `ℤ`, base-changed to `R` (`τRep`), is a `τ` over `R`. -/
def TauR.ofTau (τ : TwistedChainDatum Γ g (Yc model7 e)) : TauR R g e :=
  ⟨fun n => τRep model7 e R τ n, τRep_t2 τ, eps_τRep τ⟩

/-- (T1) for `τ_R`, on the elements. -/
lemma TauR.twist (τ : TauR R g e) (n : ℕ) (h : Γpp (Γ := Γ) g) (x : R[Simp model7 n]) :
    (τ.f n).hom (Representation.ofMulAction R SL(2, ℤ) (Simp model7 n) (fIncl g e h) x) =
      Representation.ofMulAction R SL(2, ℤ) (Simp model7 n) (fConj g e h) ((τ.f n).hom x) :=
  Rep.hom_comm_apply (τ.f n) h x

/-- `τ_R` as a chain map `res_{f₁} (C_•(model7) ⊗ R) → res_{f₂} (C_•(model7) ⊗ R)` (`6 ∈ R^×` only to name the
resolutions `P₁`, `P₂`). -/
def TauR.σ (h6 : IsUnit (6 : R)) (τ : TauR R g e) : (P₁ g e h6).complex ⟶ (P₂ g e h6).complex where
  f n := τ.f n
  comm' := by
    rintro i j (rfl : j + 1 = i)
    rw [P₁_d, P₂_d]
    apply Rep.hom_ext; apply Representation.IntertwiningMap.ext; refine LinearMap.ext fun x => ?_
    exact τ.t2 j x

lemma TauR.eps_sub (τ τ' : TauR R g e) (x : R[Simp model7 0]) :
    (repEps model7 R).hom ((τ.f 0).hom x - (τ'.f 0).hom x) = 0 := by
  rw [map_sub, τ.t3, τ'.t3, sub_self]

lemma TauR.sub_comm (h6 : IsUnit (6 : R)) (τ τ' : TauR R g e) : (τ.σ h6 - τ'.σ h6) ≫ (P₂ g e h6).π = 0 := by
  apply HomologicalComplex.to_single_hom_ext
  rw [HomologicalComplex.comp_f, HomologicalComplex.zero_f]
  simp only [P₂, resProjRes, HomologicalComplex.comp_f, Functor.mapHomologicalComplex_map_f]
  have key : (τ.σ h6 - τ'.σ h6).f 0 ≫ Rep.resMap (fConj g e) ((model7_projRes h6).π.f 0) = 0 := by
    apply Rep.hom_ext; apply Representation.IntertwiningMap.ext; refine LinearMap.ext fun x => ?_
    change ((repAug model7 R).f 0).hom ((τ.f 0).hom x - (τ'.f 0).hom x) = 0
    rw [repAug_f_zero]
    exact τ.eps_sub τ' x
  exact (Category.assoc _ _ _).symm.trans ((congrArg (· ≫ _) key).trans Limits.zero_comp)

variable (h6 : IsUnit (6 : R))

/-- The twisted homotopy `τ_R - τ'_R = ∂ h + h ∂` (`lem:tauexists`(2) on the model). -/
def htpyR (τ τ' : TauR R g e) : Homotopy (τ.σ h6 - τ'.σ h6) 0 :=
  ProjectiveResolution.liftHomotopyZero _ (τ.sub_comm h6 τ')

/-- The homotopy in degree `n`. -/
def hLR (τ τ' : TauR R g e) (n : ℕ) : R[Simp model7 n] → R[Simp model7 (n + 1 : ℕ)] :=
  fun x => ((htpyR h6 τ τ').hom n (n + 1)).hom x

lemma hLR_twist (τ τ' : TauR R g e) (n : ℕ) (h : Γpp (Γ := Γ) g) (x : R[Simp model7 n]) :
    hLR h6 τ τ' n (Representation.ofMulAction R SL(2, ℤ) (Simp model7 n) (fIncl g e h) x) =
      Representation.ofMulAction R SL(2, ℤ) (Simp model7 (n + 1 : ℕ)) (fConj g e h) (hLR h6 τ τ' n x) :=
  Rep.hom_comm_apply ((htpyR h6 τ τ').hom n (n + 1)) h x

lemma htpyR_zero (τ τ' : TauR R g e) (x : R[Simp model7 0]) :
    (τ.f 0).hom x - (τ'.f 0).hom x = (repD model7 R 0).hom (hLR h6 τ τ' 0 x) := by
  have hc := (htpyR h6 τ τ').comm 0
  rw [dNext_eq_zero _ 0 (by simp), prevD_eq _ (show (ComplexShape.down ℕ).Rel 1 0 from rfl), P₂_d] at hc
  simp only [zero_add, add_zero, HomologicalComplex.zero_f] at hc
  have := congrArg (fun φ => (Rep.Hom.hom φ) x) hc
  exact this

lemma htpyR_succ (τ τ' : TauR R g e) (n : ℕ) (x : R[Simp model7 (n + 1 : ℕ)]) :
    (τ.f (n + 1)).hom x - (τ'.f (n + 1)).hom x =
      hLR h6 τ τ' n ((repD model7 R n).hom x) + (repD model7 R (n + 1)).hom (hLR h6 τ τ' (n + 1) x) := by
  have hc := (htpyR h6 τ τ').comm (n + 1)
  rw [dNext_eq _ (show (ComplexShape.down ℕ).Rel (n + 1) n from rfl),
    prevD_eq _ (show (ComplexShape.down ℕ).Rel (n + 1 + 1) (n + 1) from rfl), P₁_d, P₂_d] at hc
  simp only [add_zero, HomologicalComplex.zero_f] at hc
  have := congrArg (fun φ => (Rep.Hom.hom φ) x) hc
  exact this

omit h6

lemma linR_add {D : XBulkDatum} {q : ℤ} (φ ψ : chains D q →+ V) (x : R[Simp D q]) :
    linR (R := R) (φ + ψ) x = linR φ x + linR ψ x := by
  simp only [linR, LinearMap.coe_comp, Function.comp_apply, LinearEquiv.coe_coe]
  induction (coeffLinearEquiv R) x using Finsupp.induction_linear with
  | zero => simp
  | add a b ha hb => rw [map_add, map_add, map_add, ha, hb]; abel
  | single s r =>
    simp only [Finsupp.linearCombination_single, AddMonoidHom.add_apply, smul_add]

lemma linR_smul {D : XBulkDatum} {q : ℤ} (c : R) (φ : chains D q →+ V) (x : R[Simp D q]) :
    linR (R := R) (c • φ) x = c • linR φ x := by
  simp only [linR, LinearMap.coe_comp, Function.comp_apply, LinearEquiv.coe_coe]
  induction (coeffLinearEquiv R) x using Finsupp.induction_linear with
  | zero => simp
  | add a b ha hb => rw [map_add, map_add, ha, hb, smul_add]
  | single s r =>
    simp only [Finsupp.linearCombination_single, AddMonoidHom.smul_apply]
    exact smul_comm r c _

/-- `τ_R` in degree `q`, precomposed with `C_q → R[Simp_q]`, as an additive map into `permCx` (`0` in negative degrees,
where `C_q = 0`). -/
def FR (τ : TauR R g e) : ∀ q : ℤ, (Yc model7 e).X q →+ (permCx R e model7).X q
  | (n : ℕ) =>
    { toFun := fun y => (τ.f n).hom (embR (R := R) (n : ℤ) y)
      map_zero' := by
        show (τ.f n).hom (embR (R := R) (D := model7) (n : ℤ) 0) = 0
        rw [show embR (R := R) (D := model7) (n : ℤ) 0 = 0 by simp [embR]]
        exact map_zero _
      map_add' := fun y y' =>
        (congrArg (τ.f n).hom (embR_add (R := R) (D := model7) (n : ℤ) (y : chains model7 n) y')).trans
          (map_add _ _ _) }
  | Int.negSucc _ => 0

lemma FR_t1 (τ : TauR R g e) (q : ℤ) : T1 Γ g (FR τ q) := by
  rcases q with n | k
  · intro h h' hh (x : chains model7 n)
    have hmem : g⁻¹ * (h : G) * g ∈ Γ := hh ▸ h'.2
    let hp : Γpp (Γ := Γ) g := ⟨h, hmem⟩
    have hc : conjHom g hp = h' := Subtype.ext hh.symm
    show (τ.f n).hom (embR (R := R) (n : ℤ) (e h • x)) =
      Representation.ofMulAction R SL(2, ℤ) (Simp model7 (n : ℤ)) (e h') ((τ.f n).hom (embR (n : ℤ) x))
    rw [embR_smul]
    have := τ.twist n hp (embR (R := R) (n : ℤ) x)
    rw [show fConj g e hp = e h' from congrArg e hc] at this
    exact this
  · intro h h' hh x
    show (0 : (Yc model7 e).X (Int.negSucc k) →+ (permCx R e model7).X (Int.negSucc k)) (h • x) = h' • 0
    rw [AddMonoidHom.zero_apply, smul_zero]

variable (e) in
/-- **`def:hecke` for `τ` over `R`** (`eq:heckechain`): `(T̃φ)(x) = ∑_j ρ(δ_j) φ_R(τ_R(s_j x))`. -/
def heckeTildeR (cd : CosetData Γ Δ g) (τ : TauR R g e) :
    GMap (EqC model7 (hΓΔ := hΓΔ) ρ e) (EqC model7 (hΓΔ := hΓΔ) ρ e) := fun q =>
  { toFun := fun φ => ⟨transport (ρ := ρ) Γ Δ cd (FR τ q) (linR (R := R) φ.1).toAddMonoidHom,
      (transport_equivariant (hΓΔ := hΓΔ) (Y := Yc model7 e) (Y' := permCx R e model7) cd (FR τ q) (FR_t1 τ q) _
        (linR_mem ρ φ)).1⟩
    map_add' := fun φ ψ => by
      apply Subtype.ext
      exact (congrArg (transportL (ρ := ρ) cd (FR τ q))
        (AddMonoidHom.ext fun x => linR_add φ.1 ψ.1 x :
          (linR (R := R) (φ.1 + ψ.1)).toAddMonoidHom = (linR φ.1).toAddMonoidHom + (linR ψ.1).toAddMonoidHom)).trans
        (map_add _ _ _)
    map_smul' := fun c φ => by
      apply Subtype.ext
      exact (congrArg (transportL (ρ := ρ) cd (FR τ q))
        (AddMonoidHom.ext fun x => linR_smul c φ.1 x :
          (linR (R := R) (c • φ.1)).toAddMonoidHom = c • (linR φ.1).toAddMonoidHom)).trans
        (map_smul _ _ _) }

lemma heckeTildeR_apply (cd : CosetData Γ Δ g) (τ : TauR R g e) (q : ℤ) (φ : (EqC model7 (hΓΔ := hΓΔ) ρ e).X q)
    (y : (Yc model7 e).X q) :
    (heckeTildeR ρ e cd τ q φ).1 y =
      ∑ i, ρ ⟨(cd.s i : G)⁻¹ * g, cd.δ_mem i⟩ (linR (R := R) φ.1 (FR τ q (cd.s i • y))) := by
  show transport (ρ := ρ) Γ Δ cd (FR τ q) (linR (R := R) φ.1).toAddMonoidHom y = _
  simp only [transport]
  exact (AddMonoidHom.finsetSum_apply _ _ _).trans rfl

/-- The key identity behind (T2): `φ_R(τ_R(∂ y)) = (φ ∘ ∂)_R(τ_R y)`. -/
lemma linR_FR_bd (τ : TauR R g e) (q : ℤ) (φ : chains model7 q →+ V) (y : chains model7 (q + 1)) :
    linR (R := R) φ (FR τ q (bd model7 q y)) =
      linR (R := R) (φ.comp (bd model7 q).toAddMonoidHom) (FR τ (q + 1) y) := by
  rcases q with n | k
  · have h1 := congrArg (fun z => linR (R := R) φ ((τ.f n).hom z)) (embR_bd (R := R) (D := model7) (n : ℤ) y)
    have h2 := congrArg (linR (R := R) φ) (τ.t2 n (embR ((n + 1 : ℕ) : ℤ) y))
    have h3 := linR_bcF (R := R) φ ((τ.f (n + 1)).hom (embR ((n + 1 : ℕ) : ℤ) y))
    exact h1.trans (h2.symm.trans h3)
  · have hs := subsingleton_neg model7 (Int.negSucc k) (Int.negSucc_lt_zero k)
    have h0 : φ.comp (bd model7 (Int.negSucc k)).toAddMonoidHom = 0 := by
      ext z; simp only [AddMonoidHom.comp_apply, AddMonoidHom.zero_apply]
      rw [Subsingleton.elim ((bd model7 (Int.negSucc k)).toAddMonoidHom z) 0, map_zero]
    have hL : linR (R := R) φ (FR τ (Int.negSucc k) (bd model7 (Int.negSucc k) y)) = 0 := map_zero (linR (R := R) φ)
    have hR : linR (R := R) (φ.comp (bd model7 (Int.negSucc k)).toAddMonoidHom) (FR τ (Int.negSucc k + 1) y) = 0 := by
      rw [h0, linR_zero]; rfl
    exact hL.trans hR.symm

/-- **`thm:welldef`, chain map**, for `τ` over `R`. -/
theorem heckeTildeR_isChainMap (cd : CosetData Γ Δ g) (τ : TauR R g e) :
    IsChainMap (heckeTildeR (hΓΔ := hΓΔ) ρ e cd τ) := by
  intro q
  apply LinearMap.ext; intro φ
  apply Subtype.ext; apply AddMonoidHom.ext; intro y
  show (heckeTildeR ρ e cd τ q φ).1 (bd model7 q y) =
    (heckeTildeR ρ e cd τ (q + 1) ((EqC model7 (hΓΔ := hΓΔ) ρ e).d q φ)).1 y
  refine (heckeTildeR_apply ρ cd τ q φ _).trans
    ((Finset.sum_congr rfl fun i _ => congrArg _ ?_).trans (heckeTildeR_apply ρ cd τ (q + 1) _ y).symm)
  exact (congrArg (fun z => linR (R := R) φ.1 (FR τ q z)) (bd_equiv model7 q (e (cd.s i)) y).symm).trans
    (linR_FR_bd τ q φ.1 _)

/-- For a `τ` over `ℤ`, base-changed, `heckeTildeR` is `heckeTilde` (slot 18's `def:hecke`). -/
theorem heckeTildeR_ofTau (cd : CosetData Γ Δ g) (τ : TwistedChainDatum Γ g (Yc model7 e)) :
    heckeTildeR (hΓΔ := hΓΔ) ρ e cd (TauR.ofTau R τ) = heckeTilde model7 ρ e cd τ := by
  funext q
  apply LinearMap.ext; intro φ
  apply Subtype.ext; apply AddMonoidHom.ext; intro y
  refine (heckeTildeR_apply ρ cd _ q φ y).trans ((Finset.sum_congr rfl fun i _ => congrArg _ ?_).trans
    (heckeTilde_apply model7 ρ e cd τ q φ y).symm)
  generalize cd.s i • y = z
  rcases q with n | k
  · exact (congrArg (linR (R := R) φ.1) (embR_τ (R := R) τ n z).symm).trans (linR_embR φ.1 _)
  · have hs := subsingleton_neg model7 (Int.negSucc k) (Int.negSucc_lt_zero k)
    have hR : φ.1 (τ.τ (Int.negSucc k) z) = 0 := by
      have : Subsingleton ((Yc model7 e).X (Int.negSucc k)) := hs
      rw [Subsingleton.elim (τ.τ (Int.negSucc k) z) 0]
      exact map_zero φ.1
    exact (map_zero (linR (R := R) φ.1)).trans hR.symm

variable (h6 : IsUnit (6 : R))

variable (e) in
/-- The homotopy in degree `m`, precomposed with `C_m → R[Simp_m]`. -/
def hFR (τ τ' : TauR R g e) (m : ℕ) : (Yc model7 e).X m →+ (permCx R e model7).X ((m : ℤ) + 1) where
  toFun y := hLR h6 τ τ' m (embR m y)
  map_zero' := by
    show hLR h6 τ τ' m (embR m 0) = 0
    rw [show embR (R := R) (D := model7) m 0 = 0 by simp [embR]]
    exact map_zero ((htpyR h6 τ τ').hom m (m + 1)).hom
  map_add' y y' := by
    exact (congrArg (hLR h6 τ τ' m) (embR_add (R := R) (D := model7) (m : ℤ) (y : chains model7 m) y')).trans
      (map_add ((htpyR h6 τ τ').hom m (m + 1)).hom _ _)

lemma hFR_t1 (τ τ' : TauR R g e) (m : ℕ) : T1 Γ g (hFR e h6 τ τ' m) := by
  intro h h' hh (x : chains model7 m)
  have hmem : g⁻¹ * (h : G) * g ∈ Γ := hh ▸ h'.2
  let hp : Γpp (Γ := Γ) g := ⟨h, hmem⟩
  have hc : conjHom g hp = h' := Subtype.ext hh.symm
  show hLR h6 τ τ' m (embR m (e h • x)) =
    Representation.ofMulAction R SL(2, ℤ) (Simp model7 ((m : ℤ) + 1)) (e h') (hLR h6 τ τ' m (embR m x))
  rw [embR_smul]
  have := hLR_twist h6 τ τ' m hp (embR m x)
  rw [show fConj g e hp = e h' from congrArg e hc] at this
  exact this

variable (e) in
/-- The cochain homotopy `h̃ ψ = ∑_j ρ(δ_j) ψ_R(h(s_j ·))` for `τ_R, τ'_R`. -/
def hTildeR (cd : CosetData Γ Δ g) (τ τ' : TauR R g e) (m : ℕ)
    (ψ : (EqC model7 (hΓΔ := hΓΔ) ρ e).X ((m : ℤ) + 1)) : (EqC model7 (hΓΔ := hΓΔ) ρ e).X m :=
  ⟨transport (ρ := ρ) Γ Δ cd (hFR e h6 τ τ' m) (linR (R := R) ψ.1).toAddMonoidHom,
    (transport_equivariant (hΓΔ := hΓΔ) (Y := Yc model7 e) (Y' := permCx R e model7) cd (hFR e h6 τ τ' m)
      (hFR_t1 h6 τ τ' m) _ (linR_mem ρ ψ)).1⟩

/-- The termwise identity in degree `m + 1`: for `φ` with `φ ∘ ∂ = 0`, `φ_R(τ_R c) - φ_R(τ'_R c) = φ_R(h(∂ c))`. -/
lemma cocycle_diffR (τ τ' : TauR R g e) (m : ℕ) (φ : chains model7 ((m : ℤ) + 1) →+ V)
    (hφ : φ.comp (bd model7 ((m : ℤ) + 1)).toAddMonoidHom = 0) (c : chains model7 ((m : ℤ) + 1)) :
    linR (R := R) φ (FR τ ((m : ℤ) + 1) c) - linR (R := R) φ (FR τ' ((m : ℤ) + 1) c) =
      linR (R := R) φ (hLR h6 τ τ' m (embR m (bd model7 m c))) := by
  show linR (R := R) φ ((τ.f (m + 1)).hom (embR ((m + 1 : ℕ) : ℤ) c)) -
    linR (R := R) φ ((τ'.f (m + 1)).hom (embR ((m + 1 : ℕ) : ℤ) c)) = _
  rw [← map_sub, htpyR_succ h6 τ τ' m (embR ((m + 1 : ℕ) : ℤ) c), map_add, embR_bd]
  have h2 : linR (R := R) φ ((repD model7 R (m + 1)).hom (hLR h6 τ τ' (m + 1) (embR ((m + 1 : ℕ) : ℤ) c))) = 0 := by
    show linR (R := R) φ ((coeffLinearEquiv R).symm (bcF R (dZ model7 ((m : ℤ) + 1))
      (coeffLinearEquiv R (hLR h6 τ τ' (m + 1) (embR ((m + 1 : ℕ) : ℤ) c))))) = 0
    rw [linR_bcF, hφ, linR_zero, LinearMap.zero_apply]
  rw [h2, add_zero]
  rfl

include h6 in
/-- The degree-`0` identity: for `φ` with `φ ∘ ∂ = 0`, `φ_R(τ_R c) = φ_R(τ'_R c)`. -/
lemma cocycle_diffR_zero (τ τ' : TauR R g e) (φ : chains model7 0 →+ V)
    (hφ : φ.comp (bd model7 0).toAddMonoidHom = 0) (c : chains model7 0) :
    linR (R := R) φ (FR τ 0 c) - linR (R := R) φ (FR τ' 0 c) = 0 := by
  show linR (R := R) φ ((τ.f 0).hom (embR 0 c)) - linR (R := R) φ ((τ'.f 0).hom (embR 0 c)) = 0
  rw [← map_sub, htpyR_zero h6 τ τ' (embR 0 c)]
  show linR (R := R) φ ((coeffLinearEquiv R).symm (bcF R (dZ model7 0)
      (coeffLinearEquiv R (hLR h6 τ τ' 0 (embR 0 c))))) = 0
  rw [linR_bcF, hφ, linR_zero, LinearMap.zero_apply]

include h6 in
/-- **`heckeTildeR_indep_tau`** ([C] `thm:welldef`(b) on the 7-vertex model, `6 ∈ R^×`, for `τ` **over `R`**): for two
twisted chain data `τ_R, τ'_R` over `R` and the same coset data, `T̃_{τ_R}` and `T̃_{τ'_R}` induce the same map on
`H^n(C^•_eq)` for every `n ≥ 0`. -/
theorem heckeTildeR_indep_tau (cd : CosetData Γ Δ g) (τ τ' : TauR R g e) (n : ℕ) :
    HmapOf (heckeTildeR (hΓΔ := hΓΔ) ρ e cd τ) (heckeTildeR_isChainMap ρ cd τ) n =
      HmapOf (heckeTildeR (hΓΔ := hΓΔ) ρ e cd τ') (heckeTildeR_isChainMap ρ cd τ') n := by
  apply LinearMap.ext; intro c
  induction c using Submodule.Quotient.induction_on with
  | _ z =>
    rw [HmapOf_mk, HmapOf_mk, CochainCx.mk_eq_mk_iff]
    have hz : (EqC model7 (hΓΔ := hΓΔ) ρ e).d n z.1 = 0 := ((EqC model7 (hΓΔ := hΓΔ) ρ e).mem_Z _ _).1 z.2
    have hφ : z.1.1.comp (bd model7 n).toAddMonoidHom = 0 := congrArg Subtype.val hz
    have hT : ∀ y : chains model7 n, (heckeTildeR (hΓΔ := hΓΔ) ρ e cd τ n z.1).1 y -
        (heckeTildeR (hΓΔ := hΓΔ) ρ e cd τ' n z.1).1 y =
        ∑ i, ρ ⟨(cd.s i : G)⁻¹ * g, cd.δ_mem i⟩ (linR (R := R) z.1.1 (FR τ n (e (cd.s i) • y)) -
          linR (R := R) z.1.1 (FR τ' n (e (cd.s i) • y))) := by
      intro y
      refine (congrArg₂ (· - ·) (heckeTildeR_apply ρ cd τ n z.1 y) (heckeTildeR_apply ρ cd τ' n z.1 y)).trans ?_
      rw [← Finset.sum_sub_distrib]
      exact Finset.sum_congr rfl fun i _ => (map_sub _ _ _).symm
    cases n with
    | zero =>
      have h0 : heckeTildeR (hΓΔ := hΓΔ) ρ e cd τ 0 z.1 - heckeTildeR (hΓΔ := hΓΔ) ρ e cd τ' 0 z.1 = 0 := by
        apply Subtype.ext; apply AddMonoidHom.ext; intro y
        refine (hT y).trans (Finset.sum_eq_zero fun i _ => ?_)
        exact (congrArg _ (cocycle_diffR_zero h6 τ τ' z.1.1 hφ _)).trans (map_zero _)
      exact h0 ▸ Submodule.zero_mem _
    | succ m =>
      show (heckeTildeR (hΓΔ := hΓΔ) ρ e cd τ ((m : ℤ) + 1) z.1 -
        heckeTildeR (hΓΔ := hΓΔ) ρ e cd τ' ((m : ℤ) + 1) z.1) ∈ (EqC model7 (hΓΔ := hΓΔ) ρ e).B ((m : ℤ) + 1)
      rw [CochainCx.mem_B_succ]
      refine ⟨hTildeR ρ e h6 cd τ τ' m z.1, ?_⟩
      apply Subtype.ext; apply AddMonoidHom.ext; intro y
      have hd : ((EqC model7 (hΓΔ := hΓΔ) ρ e).d m (hTildeR ρ e h6 cd τ τ' m z.1)).1 y =
          ∑ i, ρ ⟨(cd.s i : G)⁻¹ * g, cd.δ_mem i⟩
            (linR (R := R) z.1.1 (hLR h6 τ τ' m (embR m (e (cd.s i) • bd model7 m y)))) := by
        show transport (ρ := ρ) Γ Δ cd (hFR e h6 τ τ' m) (linR (R := R) z.1.1).toAddMonoidHom (bd model7 m y) = _
        simp only [transport]
        exact (AddMonoidHom.finsetSum_apply _ _ _).trans rfl
      refine hd.trans (Eq.trans (Finset.sum_congr rfl fun i _ => ?_) (hT y).symm)
      exact congrArg _ ((congrArg (fun c => linR (R := R) z.1.1 (hLR h6 τ τ' m (embR m c)))
        (bd_equiv model7 m (e (cd.s i)) y).symm).trans (cocycle_diffR h6 τ τ' m z.1.1 hφ _).symm)

include h6 in
/-- The `ℤ` case recovered: `heckeTilde_indep_tau` is `heckeTildeR_indep_tau` at `TauR.ofTau`. -/
theorem heckeTilde_indep_tau' (cd : CosetData Γ Δ g) (τ τ' : TwistedChainDatum Γ g (Yc model7 e)) (n : ℕ) :
    HmapOf (heckeTilde model7 ρ e (hΓΔ := hΓΔ) cd τ) (heckeTilde_isChainMap model7 ρ e cd τ) n =
      HmapOf (heckeTilde model7 ρ e (hΓΔ := hΓΔ) cd τ') (heckeTilde_isChainMap model7 ρ e cd τ') n := by
  rw [HmapOf_congr (heckeTildeR_ofTau ρ cd τ).symm _ (heckeTildeR_isChainMap ρ cd _),
    HmapOf_congr (heckeTildeR_ofTau ρ cd τ').symm _ (heckeTildeR_isChainMap ρ cd _)]
  exact heckeTildeR_indep_tau ρ h6 cd _ _ n

end TauOverR

/-! ### F3-19 MINOR 4: (b) on the relative complex, and (c) in general -/

section WelldefC

variable {G : Type} [Group G] {Γ : Subgroup G} {Δ : Submonoid G} {hΓΔ : ∀ γ : Γ, (γ : G) ∈ Δ}
  {R : Type} [CommRing R] {V : Type} [AddCommGroup V] [Module R V] (ρ : Δ →* Module.End R V)
  {e : Γ ≃* SL(2, ℤ)} {g : G} (h6 : IsUnit (6 : R))

include h6 in
/-- **`heckeRel_indep_tau`** ([C] `thm:welldef`(b) on `H^n_eq(K, Rel; L)` literally, on the model, `6 ∈ R^×`): the
operator the implementation computes on the relative complex (`heckeRel = Ψ ∘ T̃ ∘ Φ`) does not depend on `τ` on
cohomology. -/
theorem heckeRel_indep_tau (cd : CosetData Γ Δ g) (τ τ' : TwistedChainDatum Γ g (Yc model7 e)) (n : ℕ) :
    HmapOf (heckeRel model7 ρ e (hΓΔ := hΓΔ) cd τ) (heckeRel_isChainMap model7 ρ e cd τ) n =
      HmapOf (heckeRel model7 ρ e (hΓΔ := hΓΔ) cd τ') (heckeRel_isChainMap model7 ρ e cd τ') n := by
  have hT := fun (t : TwistedChainDatum Γ g (Yc model7 e)) =>
    isChainMap_comp (heckeTilde_isChainMap model7 ρ e cd t) (phi_isChainMap model7 (hΓΔ := hΓΔ) ρ e)
  rw [HmapOf_congr (show heckeRel model7 ρ e (hΓΔ := hΓΔ) cd τ =
      fun q => (psi model7 ρ e q).comp ((heckeTilde model7 ρ e cd τ q).comp (phi model7 ρ e q)) from rfl) _
      (isChainMap_comp (psi_isChainMap model7 ρ e) (hT τ)),
    HmapOf_congr (show heckeRel model7 ρ e (hΓΔ := hΓΔ) cd τ' =
      fun q => (psi model7 ρ e q).comp ((heckeTilde model7 ρ e cd τ' q).comp (phi model7 ρ e q)) from rfl) _
      (isChainMap_comp (psi_isChainMap model7 ρ e) (hT τ'))]
  rw [HmapOf_comp _ _ (psi_isChainMap model7 ρ e) (hT τ), HmapOf_comp _ _ (psi_isChainMap model7 ρ e) (hT τ'),
    HmapOf_comp _ _ (heckeTilde_isChainMap model7 ρ e cd τ) (phi_isChainMap model7 ρ e),
    HmapOf_comp _ _ (heckeTilde_isChainMap model7 ρ e cd τ') (phi_isChainMap model7 ρ e),
    heckeTilde_indep_tau ρ h6 cd τ τ' n]

/-- **`thm:welldef`(a)** on `X_bulk` for any two coset data (cochain level, `τ` fixed, no hypothesis on `ℓ`). -/
theorem heckeTilde_indep_cd (D : XBulkDatum) (cd cd' : CosetData Γ Δ g) (τ : TwistedChainDatum Γ g (Yc D e)) :
    heckeTilde D ρ e (hΓΔ := hΓΔ) cd' τ = heckeTilde D ρ e cd τ :=
  thetaG_indep_cd ρ cd cd' τ

include h6 in
/-- **`heckeTilde_indep_g`** ([C] `thm:welldef`(c), general case, on the model, `6 ∈ R^×`): for `g₁ = γ₁ g γ₂`
(`γ₁, γ₂ ∈ Γ`), **any** coset data `cd₁` and **any** twisted chain datum `τ₁` for `g₁` give on `H^n(C^•_eq)` the same
operator as `(cd, τ)` for `g`: via (c) for the transported choices (`thetaG_transported`), (a) (`heckeTilde_indep_cd`)
and (b) (`heckeTilde_indep_tau`). -/
theorem heckeTilde_indep_g (cd : CosetData Γ Δ g) (τ : TwistedChainDatum Γ g (Yc model7 e)) (γ₁ γ₂ : Γ)
    (cd₁ : CosetData Γ Δ ((γ₁ : G) * g * γ₂)) (τ₁ : TwistedChainDatum Γ ((γ₁ : G) * g * γ₂) (Yc model7 e)) (n : ℕ) :
    HmapOf (heckeTilde model7 ρ e (hΓΔ := hΓΔ) cd₁ τ₁) (heckeTilde_isChainMap model7 ρ e cd₁ τ₁) n =
      HmapOf (heckeTilde model7 ρ e (hΓΔ := hΓΔ) cd τ) (heckeTilde_isChainMap model7 ρ e cd τ) n := by
  rw [heckeTilde_indep_tau ρ h6 cd₁ τ₁ (transportedTau τ γ₁ γ₂) n]
  apply HmapOf_congr
  rw [heckeTilde_indep_cd ρ model7 (transportedCd hΓΔ cd γ₁ γ₂) cd₁ (transportedTau τ γ₁ γ₂)]
  exact thetaG_transported ρ cd τ γ₁ γ₂

end WelldefC

end XBulk

end GL2.Model

end
