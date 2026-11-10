import GL2.MP.RZSubdiv

/-!
# MP-L2, open item 3 (slot 22): the comparison isomorphism, natural in the coefficients

[GT] `gt_part_MP.tex`, rows `def:mp-prey`, `thm:mp-comparison-natural`, backing [C] `thm:mpcomparison` (4) (naturality in
`A`, used by [C] `thm:mpnatural`).

`mpComparison` goes through `Ext` (mathlib's `groupCohomologyIso`), whose naturality in the coefficients is not in mathlib.
Here the comparison is built through a **fixed** homotopy equivalence `e : k[𝔉'_•] ≃ bar` of projective resolutions of `k`
(`ProjectiveResolution.homotopyEquiv`): `Hom(-, A)` turns it into a homotopy equivalence of cochain complexes (`preYEquiv`,
precomposition), so

  `H^n(Γ; A) ≅ H^n(Hom(bar, A)) ≅ H^n(Hom(k[𝔉'_•], A)) ≅ H^n(C^•(S; A))`

(`mpComparisonNat`: inhomogeneous cochains = `Hom(bar, A)` (mathlib's `inhomogeneousCochainsIso`), then `e`, then
`linYonIso`), and every step is induced by a chain map commuting with post-composition by `f : A ⟶ B`
(`mpComparisonNat_natural`; `mpComparisonNatIso`: the natural isomorphism of functors `Rep k Γ ⥤ ModuleCat k`).
-/

namespace GL2.MP

namespace Cover

open CategoryTheory groupCohomology

variable {G : Type} [Group G] {k : Type} [CommRing k]

/-! ### `Hom(-, A)` on chain complexes of representations -/

/-- **Precomposition** with a chain map `φ : X ⟶ Y`: `Hom(Y, A) ⟶ Hom(X, A)`. -/
noncomputable def preY {X Y : ChainComplex (Rep.{0} k G) ℕ} (φ : X ⟶ Y) (A : Rep.{0} k G) :
    Y.linearYonedaObj k A ⟶ X.linearYonedaObj k A :=
  (HomologicalComplex.unopFunctor _ _).map
    (((((linearYoneda k (Rep.{0} k G)).obj A).rightOp).mapHomologicalComplex _).map φ).op

lemma preY_f_apply {X Y : ChainComplex (Rep.{0} k G) ℕ} (φ : X ⟶ Y) (A : Rep.{0} k G) (i : ℕ) (g : Y.X i ⟶ A) :
    ((preY φ A).f i).hom g = φ.f i ≫ g := rfl

lemma preY_comp {X Y Z : ChainComplex (Rep.{0} k G) ℕ} (φ : X ⟶ Y) (ψ : Y ⟶ Z) (A : Rep.{0} k G) :
    preY (φ ≫ ψ) A = preY ψ A ≫ preY φ A := by
  unfold preY
  rw [Functor.map_comp, op_comp, Functor.map_comp]
  rfl

lemma preY_id (X : ChainComplex (Rep.{0} k G) ℕ) (A : Rep.{0} k G) : preY (𝟙 X) A = 𝟙 _ := by
  ext i g : 3
  show (𝟙 X : X ⟶ X).f i ≫ g = g
  rw [HomologicalComplex.id_f]
  exact Category.id_comp g

/-- **`Hom(-, A)` of a homotopy equivalence** is a homotopy equivalence (of the cochain complexes, reversed). -/
noncomputable def preYEquiv {X Y : ChainComplex (Rep.{0} k G) ℕ} (e : HomotopyEquiv X Y) (A : Rep.{0} k G) :
    HomotopyEquiv (Y.linearYonedaObj k A) (X.linearYonedaObj k A) where
  hom := preY e.hom A
  inv := preY e.inv A
  homotopyHomInvId :=
    (Homotopy.ofEq (preY_comp e.inv e.hom A).symm).trans
      ((Homotopy.unop ((((linearYoneda k (Rep.{0} k G)).obj A).rightOp).mapHomotopy e.homotopyInvHomId)).trans
        (Homotopy.ofEq (preY_id Y A)))
  homotopyInvHomId :=
    (Homotopy.ofEq (preY_comp e.hom e.inv A).symm).trans
      ((Homotopy.unop ((((linearYoneda k (Rep.{0} k G)).obj A).rightOp).mapHomotopy e.homotopyHomInvId)).trans
        (Homotopy.ofEq (preY_id X A)))

/-- **Post-composition** with `f : A ⟶ B`: `Hom(X, A) ⟶ Hom(X, B)`. -/
noncomputable def postY (X : ChainComplex (Rep.{0} k G) ℕ) {A B : Rep.{0} k G} (f : A ⟶ B) :
    X.linearYonedaObj k A ⟶ X.linearYonedaObj k B where
  f i := ModuleCat.ofHom (Linear.rightComp k (X.X i) f)
  comm' i j _ := by
    ext g : 2
    exact Category.assoc (X.d j i) (show X.X i ⟶ A from g) f

lemma postY_f_apply (X : ChainComplex (Rep.{0} k G) ℕ) {A B : Rep.{0} k G} (f : A ⟶ B) (i : ℕ) (g : X.X i ⟶ A) :
    ((postY X f).f i).hom g = g ≫ f := rfl

/-- Pre- and post-composition commute. -/
lemma postY_preY {X Y : ChainComplex (Rep.{0} k G) ℕ} (φ : X ⟶ Y) {A B : Rep.{0} k G} (f : A ⟶ B) :
    postY Y f ≫ preY φ B = preY φ A ≫ postY X f := by
  ext i g : 3
  exact (Category.assoc (φ.f i) (show Y.X i ⟶ A from g) f).symm

/-! ### The three naturality squares -/

/-- **Inhomogeneous cochains = `Hom(bar, A)`, naturally in `A`.** -/
lemma cochainsMap_inhomogeneousCochainsIso {A B : Rep.{0} k G} (f : A ⟶ B) :
    cochainsMap (MonoidHom.id G) f ≫ (inhomogeneousCochainsIso B).hom =
      (inhomogeneousCochainsIso A).hom ≫ postY (Rep.barComplex k G) f := by
  ext i g : 3
  show (Rep.freeLiftLEquiv k G (Fin i → G) B).symm (fun x => f.hom (g x)) =
    (Rep.freeLiftLEquiv k G (Fin i → G) A).symm g ≫ f
  apply Rep.free_ext
  intro x
  have h1 := congrFun ((Rep.freeLiftLEquiv k G (Fin i → G) B).apply_symm_apply (fun x => f.hom (g x))) x
  have h2 := congrFun ((Rep.freeLiftLEquiv k G (Fin i → G) A).apply_symm_apply g) x
  exact h1.trans (congrArg f.hom h2).symm

variable {C : Cover.{0} G} (H : C.Hyp)

/-- **`Hom(k[𝔉'_•], A) = C^•(S; A)`, naturally in `A`.** -/
lemma postY_linYonIso {A B : Rep.{0} k G} (f : A ⟶ B) :
    postY (mpCx C k) f ≫ (linYonIso B).hom = (linYonIso A).hom ≫ C.mpComplexMap f := by
  ext i φ : 3
  simp only [HomologicalComplex.comp_f, ModuleCat.hom_comp, LinearMap.coe_comp, Function.comp_apply]
  rfl

/-! ### The natural comparison isomorphism -/

variable (k) in
/-- **The fixed comparison** of the MP resolution with the bar resolution (both projective resolutions of `k`). -/
noncomputable def mpBarEquiv (hAC : (yCx H k).toCx.Acyclic) : HomotopyEquiv (mpCx C k) (Rep.barComplex k G) :=
  ProjectiveResolution.homotopyEquiv (mpRes H k hAC) (Rep.barResolution k G)

/-- **The comparison isomorphism** `H^n(Γ; A) ≅ H^n(C^•(S; A))` through the fixed comparison of resolutions:
inhomogeneous cochains `= Hom(bar, A) ≃ Hom(k[𝔉'_•], A) = C^•(S; A)`. -/
noncomputable def mpComparisonNat (hAC : (yCx H k).toCx.Acyclic) (A : Rep.{0} k G) (n : ℕ) :
    groupCohomology A n ≅ (C.mpComplex A).homology n :=
  (HomologicalComplex.homologyFunctor _ _ n).mapIso (inhomogeneousCochainsIso A) ≪≫
    (preYEquiv (mpBarEquiv k H hAC) A).toHomologyIso n ≪≫
    (HomologicalComplex.homologyFunctor _ _ n).mapIso (linYonIso A)

/-- **The comparison is natural in the coefficients** ([C] `thm:mpcomparison` (4), naturality in `A`): for every
morphism `f : A ⟶ B` of representations, `H^n(f)` on group cohomology corresponds to the map of MP complexes
induced by `f`. -/
theorem mpComparisonNat_natural (hAC : (yCx H k).toCx.Acyclic) {A B : Rep.{0} k G} (f : A ⟶ B) (n : ℕ) :
    (groupCohomology.functor k G n).map f ≫ (mpComparisonNat H hAC B n).hom =
      (mpComparisonNat H hAC A n).hom ≫ HomologicalComplex.homologyMap (C.mpComplexMap f) n := by
  have key : cochainsMap (MonoidHom.id G) f ≫ (inhomogeneousCochainsIso B).hom ≫ preY (mpBarEquiv k H hAC).hom B ≫
      (linYonIso B).hom =
    (inhomogeneousCochainsIso A).hom ≫ preY (mpBarEquiv k H hAC).hom A ≫ (linYonIso A).hom ≫ C.mpComplexMap f := by
    rw [← Category.assoc, cochainsMap_inhomogeneousCochainsIso, Category.assoc, ← Category.assoc (postY _ f),
      postY_preY, Category.assoc, postY_linYonIso]
  have := congrArg (fun φ => HomologicalComplex.homologyMap φ n) key
  simp only [HomologicalComplex.homologyMap_comp] at this
  show HomologicalComplex.homologyMap (cochainsMap (MonoidHom.id G) f) n ≫
      HomologicalComplex.homologyMap (inhomogeneousCochainsIso B).hom n ≫
      HomologicalComplex.homologyMap (preY (mpBarEquiv k H hAC).hom B) n ≫ HomologicalComplex.homologyMap (linYonIso B).hom n =
    (HomologicalComplex.homologyMap (inhomogeneousCochainsIso A).hom n ≫
      HomologicalComplex.homologyMap (preY (mpBarEquiv k H hAC).hom A) n ≫
      HomologicalComplex.homologyMap (linYonIso A).hom n) ≫ HomologicalComplex.homologyMap (C.mpComplexMap f) n
  rw [Category.assoc, Category.assoc]
  exact this

variable (k) in
/-- The MP complex as a functor of the coefficients. -/
noncomputable def mpFunctor : Rep.{0} k G ⥤ CochainComplex (ModuleCat k) ℕ where
  obj A := C.mpComplex A
  map f := C.mpComplexMap f
  map_id A := by ext n x : 3; rfl
  map_comp f g := by ext n x : 3; rfl

/-- **The natural isomorphism** `H^n(Γ; -) ≅ H^n(C^•(S; -))` of functors `Rep k Γ ⥤ ModuleCat k`. -/
noncomputable def mpComparisonNatIso (hAC : (yCx H k).toCx.Acyclic) (n : ℕ) :
    groupCohomology.functor k G n ≅ mpFunctor (C := C) k ⋙ HomologicalComplex.homologyFunctor _ _ n :=
  NatIso.ofComponents (fun A => mpComparisonNat H hAC A n) (fun f => mpComparisonNat_natural H hAC f n)

end Cover

open scoped MatrixGroups

/-- **For `SL₂(ℤ)` on `M₇`: `H^n(SL₂(ℤ); -) ≅ H^n(C^•(S(M₇); -))` as functors**, over every commutative ring, unconditional. -/
noncomputable def model7_comparisonNatIso (k : Type) [CommRing k] (n : ℕ) :=
  Cover.mpComparisonNatIso model7Hyp (model7_AC k) n

/-- **For `SL₂(ℤ)` on `R_ℤ`: `H^n(SL₂(ℤ); -) ≅ H^n(C^•(S(R_ℤ); -))` as functors**, over every commutative ring, unconditional. -/
noncomputable def rz_comparisonNatIso (k : Type) [CommRing k] (n : ℕ) :=
  Cover.mpComparisonNatIso rzHyp (rz_AC k) n

end GL2.MP
