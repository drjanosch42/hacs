import GL2.MP.Morphism

/-!
# MP-L3, rows L8 (2) and L9 (slot 22): the induced map is the restriction; refinement invariance

[GT] `gt_part_MP.tex`, rows `def:mp-exactaug`, `thm:mp-homotopy-exact`, `def:mp-barmap`, `def:mp-zu`,
`thm:mp-pull-natural`, `thm:mp-refine-invariance`, `thm:mp-label-indep`, backing [C] `thm:mpnatural` (2), (3).

**The statement** (`pull_natural`): for a morphism `M : Morph φ C C'` of MP data along `φ : Γ → Γ'` (both with (IO),
finiteness, `dim ≤ 2` and (AC) over `k`), every representation `A'` of `Γ'` and every `n`,

  `H^n(Γ'; A') --res φ--> H^n(Γ; φ^* A')`
  `     ≅ |                     | ≅`
  `H^n(C^•(S'; A')) --H^n(u^*)--> H^n(C^•(S; φ^* A'))`

commutes, the vertical isomorphisms being the comparison isomorphisms `mpComparisonNat` (natural in the coefficients),
the top arrow mathlib's `groupCohomology.map φ (𝟙 _) n` (inhomogeneous cochains `x ↦ x ∘ φ^n`), the bottom one the
cochain map with the label factor (`Morph.pullMap`).

**The proof** ([C] `thm:mpnatural`, proof of (2)): on cochains, the restriction is precomposition with the chain map
`β : bar_Γ → φ^* bar_{Γ'}`, `(g_1, …, g_n) ↦ (φ g_1, …, φ g_n)` (`barMap`, `cochainsMap_barMap`), and `u^*` is
precomposition with `k[u] : k[𝔉'_•] → φ^* k[𝔉'{}'_•]` (`zu`, `pullMap_zu`).  Both `k[u] ≫ φ^*(e')` and `e ≫ β` are
`Γ`-chain maps from the projective resolution `k[𝔉'_•]` to the **exact, not necessarily projective** complex
`φ^* bar_{Γ'}` over the same augmentation, hence homotopic (`ExactAug`, `homotopy'`: mathlib's `liftHomotopy` with the
target's projectivity removed, which it never uses).

**Consequences** ([C] `thm:mpnatural` (3)):
* `pull_homology_eq`: `H^n(u^*)` is the composite of comparison isomorphisms and the restriction — in particular it does
  **not depend on the labels** (`pull_label_indep`: two morphisms along the same `φ` induce the same map on cohomology);
* **refinement invariance** (`refine_quasiIso`): for `φ = id` (a refinement, or any morphism of MP data of the same
  group), `u^*` is a quasi-isomorphism in every degree and induces the identity of `H^•(Γ; A)` under the comparisons
  (`refine_homology_eq`).
-/

noncomputable section

namespace GL2.MP

open CategoryTheory Limits

/-! ### Chain homotopies into exact (not necessarily projective) complexes -/

namespace ExactCx

variable {𝒞 : Type*} [Category 𝒞] [Abelian 𝒞]

/-- **An exact augmented complex** `Q → Z` (a resolution of `Z`, not necessarily projective). -/
structure ExactAug (Q : ChainComplex 𝒞 ℕ) (Z : 𝒞) where
  ε : Q.X 0 ⟶ Z
  w : Q.d 1 0 ≫ ε = 0
  ex0 : (ShortComplex.mk _ _ w).Exact
  epi : Epi ε
  exS : ∀ n, (ShortComplex.mk _ _ (Q.d_comp_d (n + 2) (n + 1) n)).Exact

variable {Y : 𝒞} {Q : ChainComplex 𝒞 ℕ} {Z : 𝒞} {P : ProjectiveResolution Y} (E : ExactAug Q Z)

/-- Auxiliary for `homotopyZero'`. -/
def hzZero (f : P.complex ⟶ Q) (h : f.f 0 ≫ E.ε = 0) : P.complex.X 0 ⟶ Q.X 1 :=
  E.ex0.liftFromProjective (f.f 0) h

@[reassoc (attr := simp)]
lemma hzZero_comp (f : P.complex ⟶ Q) (h : f.f 0 ≫ E.ε = 0) : hzZero E f h ≫ Q.d 1 0 = f.f 0 :=
  E.ex0.liftFromProjective_comp _ _

/-- Auxiliary for `homotopyZero'`. -/
def hzOne (f : P.complex ⟶ Q) (h : f.f 0 ≫ E.ε = 0) : P.complex.X 1 ⟶ Q.X 2 :=
  (E.exS 0).liftFromProjective (f.f 1 - P.complex.d 1 0 ≫ hzZero E f h)
    (by rw [Preadditive.sub_comp, Category.assoc, HomologicalComplex.Hom.comm, hzZero_comp, sub_self])

@[reassoc (attr := simp)]
lemma hzOne_comp (f : P.complex ⟶ Q) (h : f.f 0 ≫ E.ε = 0) :
    hzOne E f h ≫ Q.d 2 1 = f.f 1 - P.complex.d 1 0 ≫ hzZero E f h :=
  (E.exS 0).liftFromProjective_comp _ _

/-- Auxiliary for `homotopyZero'`. -/
def hzSucc (f : P.complex ⟶ Q) (n : ℕ) (g : P.complex.X n ⟶ Q.X (n + 1)) (g' : P.complex.X (n + 1) ⟶ Q.X (n + 2))
    (w : f.f (n + 1) = P.complex.d (n + 1) n ≫ g + g' ≫ Q.d (n + 2) (n + 1)) : P.complex.X (n + 2) ⟶ Q.X (n + 3) :=
  (E.exS (n + 1)).liftFromProjective (f.f (n + 2) - P.complex.d _ _ ≫ g') (by simp [w])

@[reassoc (attr := simp)]
lemma hzSucc_comp (f : P.complex ⟶ Q) (n : ℕ) (g : P.complex.X n ⟶ Q.X (n + 1))
    (g' : P.complex.X (n + 1) ⟶ Q.X (n + 2))
    (w : f.f (n + 1) = P.complex.d (n + 1) n ≫ g + g' ≫ Q.d (n + 2) (n + 1)) :
    hzSucc E f n g g' w ≫ Q.d (n + 3) (n + 2) = f.f (n + 2) - P.complex.d _ _ ≫ g' :=
  (E.exS (n + 1)).liftFromProjective_comp _ _

/-- **A chain map from a projective resolution to an exact complex, zero on the augmentation, is null-homotopic.** -/
def homotopyZero' (f : P.complex ⟶ Q) (h : f.f 0 ≫ E.ε = 0) : Homotopy f 0 :=
  Homotopy.mkInductive _ (hzZero E f h) (by simp) (hzOne E f h) (by simp) fun n ⟨g, g', w⟩ =>
    ⟨hzSucc E f n g g' w, by simp⟩

/-- **Two chain maps from a projective resolution to an exact complex agreeing on the augmentation are homotopic.** -/
def homotopy' (f g : P.complex ⟶ Q) (h : f.f 0 ≫ E.ε = g.f 0 ≫ E.ε) : Homotopy f g :=
  Homotopy.equivSubZero.invFun (homotopyZero' E (f - g) (by simp [Preadditive.sub_comp, h]))

end ExactCx

namespace Cover

open groupCohomology Finsupp MonoidAlgebra GL2.Cone.ProjRed ExactCx

variable {G G' : Type} [Group G] [Group G'] (φ : G →* G') {k : Type} [CommRing k]

/-! ### Restriction of complexes and of `Hom` -/

/-- `φ^* X`: a chain complex of `Γ'`-representations restricted to `Γ`. -/
abbrev resC (X : ChainComplex (Rep.{0} k G') ℕ) : ChainComplex (Rep.{0} k G) ℕ :=
  ((Rep.resFunctor φ).mapHomologicalComplex _).obj X

/-- `Hom_{Γ'}(X, A) ⟶ Hom_Γ(φ^* X, φ^* A)`, `g ↦ φ^* g`. -/
def resY (X : ChainComplex (Rep.{0} k G') ℕ) (A : Rep.{0} k G') :
    X.linearYonedaObj k A ⟶ (resC φ X).linearYonedaObj k (Rep.res φ A) where
  f i := ModuleCat.ofHom
    { toFun := fun g => (Rep.resFunctor φ).map (show X.X i ⟶ A from g)
      map_add' := fun g g' => (Rep.resFunctor φ).map_add
      map_smul' := fun r g => (Rep.resFunctor φ).map_smul r (show X.X i ⟶ A from g) }
  comm' i j _ := by
    ext g : 2
    exact (Rep.resFunctor φ).map_comp (X.d j i) (show X.X i ⟶ A from g)

lemma resY_f_apply (X : ChainComplex (Rep.{0} k G') ℕ) (A : Rep.{0} k G') (i : ℕ) (g : X.X i ⟶ A) :
    ((resY φ X A).f i).hom g = (Rep.resFunctor φ).map g := rfl

lemma preY_resY {X Y : ChainComplex (Rep.{0} k G') ℕ} (ψ : X ⟶ Y) (A : Rep.{0} k G') :
    preY ψ A ≫ resY φ X A = resY φ Y A ≫ preY (((Rep.resFunctor φ).mapHomologicalComplex _).map ψ) (Rep.res φ A) := by
  ext i g : 3
  show (Rep.resFunctor φ).map (ψ.f i ≫ (show Y.X i ⟶ A from g)) =
    (Rep.resFunctor φ).map (ψ.f i) ≫ (Rep.resFunctor φ).map (show Y.X i ⟶ A from g)
  exact (Rep.resFunctor φ).map_comp (ψ.f i) (show Y.X i ⟶ A from g)

/-! ### The bar map `β : bar_Γ → φ^* bar_{Γ'}` and the restriction on inhomogeneous cochains -/

/-- `β_n`: `(g_1, …, g_n) ↦ (φ g_1, …, φ g_n)`. -/
def barMapF (n : ℕ) : Rep.free k G (Fin n → G) ⟶ Rep.res φ (Rep.free k G' (Fin n → G')) :=
  Rep.freeLift k G (Rep.res φ (Rep.free k G' (Fin n → G')))
    (fun x => (Finsupp.single (φ ∘ x) (MonoidAlgebra.single (1 : G') (1 : k)) : (Fin n → G') →₀ MonoidAlgebra k G'))

lemma barMapF_single (n : ℕ) (x : Fin n → G) (g : G) (r : k) :
    (barMapF φ n).hom (Finsupp.single x (MonoidAlgebra.single g r)) =
      Finsupp.single (φ ∘ x) (MonoidAlgebra.single (φ g) r) := by
  simp [barMapF]

/-- **The bar map** `β : bar_Γ → φ^* bar_{Γ'}`, a chain map. -/
def barMap : Rep.barComplex k G ⟶ resC φ (Rep.barComplex k G') where
  f n := barMapF φ n
  comm' i j hij := by
    obtain rfl : j + 1 = i := hij
    have e1 : (resC φ (Rep.barComplex k G')).d (j + 1) j = (Rep.resFunctor φ).map (Rep.barComplex.d k G' j) := by
      rw [Functor.mapHomologicalComplex_obj_d, Rep.barComplex.d_def]
    rw [e1, Rep.barComplex.d_def]
    apply Rep.free_ext
    intro x
    show (Rep.barComplex.d k G' j).hom ((barMapF φ (j + 1)).hom (Finsupp.single x (MonoidAlgebra.single 1 1))) =
      (barMapF φ j).hom ((Rep.barComplex.d k G j).hom (Finsupp.single x (MonoidAlgebra.single 1 1)))
    rw [barMapF_single, map_one, Rep.barComplex.d_single, Rep.barComplex.d_single, map_add, map_sum, barMapF_single]
    congr 1
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [barMapF_single, map_one, Fin.comp_contractNth (· * ·) (· * ·) (map_mul φ) i x]

/-- **The restriction on inhomogeneous cochains is precomposition with `β`.** -/
lemma cochainsMap_barMap (A : Rep.{0} k G') :
    cochainsMap φ (𝟙 (Rep.res φ A)) ≫ (inhomogeneousCochainsIso (Rep.res φ A)).hom =
      (inhomogeneousCochainsIso A).hom ≫ resY φ _ A ≫ preY (barMap φ) (Rep.res φ A) := by
  ext i x : 3
  apply Rep.free_ext
  intro y
  show ((Rep.freeLiftLEquiv k G (Fin i → G) (Rep.res φ A)).symm (fun y => x (φ ∘ y))).hom
      (Finsupp.single y (single 1 1)) =
    ((Rep.freeLiftLEquiv k G' (Fin i → G') A).symm x).hom ((barMapF φ i).hom (Finsupp.single y (single 1 1)))
  have h1 := congrFun ((Rep.freeLiftLEquiv k G (Fin i → G) (Rep.res φ A)).apply_symm_apply (fun y => x (φ ∘ y))) y
  have h2 := congrFun ((Rep.freeLiftLEquiv k G' (Fin i → G') A).apply_symm_apply x) (φ ∘ y)
  refine h1.trans ?_
  show _ = ((Rep.freeLiftLEquiv k G' (Fin i → G') A).symm x).hom ((barMapF φ i).hom (Finsupp.single y (single 1 1)))
  rw [barMapF_single, map_one]
  exact h2.symm

/-! ### The augmentation of the bar resolution -/

lemma diag_inv_single (m : ℕ) (f : Fin m → G) (g : G) (r : k) :
    (Rep.diagonalSuccIsoFree k G m).inv.hom (Finsupp.single f (MonoidAlgebra.single g r)) =
      MonoidAlgebra.single (g • Fin.partialProd f) r := by
  simp only [Rep.diagonalSuccIsoFree, Rep.diagonalSuccIsoTensorTrivial, Iso.trans_inv, Rep.hom_comp,
    Representation.IntertwiningMap.comp_apply]
  have step1 : (Rep.Hom.hom (Rep.leftRegularTensorTrivialIsoFree k G (Fin m → G)).inv)
      (Finsupp.single f (.single g r)) = .single g 1 ⊗ₜ[k] .single f r :=
    Representation.leftRegularTensorTrivialIsoFree_symm_apply_single_single f g r
  rw [step1]
  simp only [Action.tensorObj_V, Rep.tensor_V, Rep.tensor_ρ]
  have key₁ := Representation.linearizeMap_single (k := k)
    (Action.diagonalSuccIsoTensorTrivial G m).inv (g, f) ((1 : k) * r)
  have key₂ := Representation.LinearizeMonoidal.μ_apply_single_single (k := k)
    (X := Action.leftRegular G) (Y := Action.trivial G (Fin m → G)) g f 1 r
  exact ((congrArg (fun z => (Representation.linearizeMap
    (Action.diagonalSuccIsoTensorTrivial G m).inv) z) key₂).trans key₁).trans (by
    rw [one_mul]; congr 1; exact Action.diagonalSuccIsoTensorTrivial_inv_hom_apply g f)

variable (k) in
/-- The augmentation of the bar resolution. -/
abbrev barEps (G : Type) [Group G] : (Rep.barComplex k G).X 0 ⟶ Rep.trivial k G k := (Rep.barResolution k G).π.f 0

lemma barEps_d (G : Type) [Group G] : (Rep.barComplex k G).d 1 0 ≫ barEps k G = 0 :=
  (Rep.barResolution k G).complex_d_comp_π_f_zero

variable (k G) in
/-- **The augmentation of the bar resolution** sends `y · (g)` to `1`. -/
lemma barπ_single (y : Fin 0 → G) (g : G) (r : k) :
    (barEps k G).hom (Finsupp.single y (MonoidAlgebra.single g r)) = r := by
  have h1 : barEps k G = (Rep.diagonalSuccIsoFree k G 0).inv ≫ Rep.standardComplex.ε k G := by
    show ((Rep.barComplex.isoStandardComplex k G).hom ≫ Rep.standardComplex.εToSingle₀ k G).f 0 = _
    rw [HomologicalComplex.comp_f, Rep.standardComplex.εToSingle₀]
    erw [ChainComplex.toSingle₀Equiv_symm_apply_f_zero]
    rfl
  rw [h1]
  show (Rep.standardComplex.ε k G).hom ((Rep.diagonalSuccIsoFree k G 0).inv.hom _) = r
  rw [diag_inv_single]
  simp [Rep.standardComplex.ε]

/-- `φ^* bar_{Γ'}` is an exact augmented complex of `Γ`-representations. -/
def resBarAug : ExactAug (resC φ (Rep.barComplex k G')) (Rep.res φ (Rep.trivial k G' k)) where
  ε := (Rep.resFunctor φ).map (barEps k G')
  w := by
    show (Rep.resFunctor φ).map ((Rep.barComplex k G').d 1 0) ≫ (Rep.resFunctor φ).map (barEps k G') = 0
    rw [← Functor.map_comp, barEps_d, Functor.map_zero]
  ex0 := (Rep.res_map_exact φ _).2 (Rep.barResolution k G').exact₀
  epi := by
    have h : Epi (barEps k G') := (inferInstance : Epi ((Rep.barResolution k G').π.f 0))
    rw [Rep.epi_iff_surjective] at h
    exact (Rep.epi_iff_surjective _).2 h
  exS n := (Rep.res_map_exact φ _).2 ((Rep.barResolution k G').exact_succ n)

lemma barMap_aug :
    (barMap (k := k) φ).f 0 ≫ (resBarAug (k := k) φ).ε = barEps k G := by
  apply Rep.free_ext
  intro y
  have e : (barMapF φ 0).hom (Finsupp.single y (MonoidAlgebra.single 1 1)) =
      Finsupp.single (φ ∘ y) (MonoidAlgebra.single (1 : G') (1 : k)) := by rw [barMapF_single, map_one]
  show (barEps k G').hom ((barMapF φ 0).hom (Finsupp.single y (MonoidAlgebra.single 1 1))) =
    (barEps k G).hom (Finsupp.single y (MonoidAlgebra.single 1 1))
  rw [e]
  exact (barπ_single G' k (φ ∘ y) 1 1).trans (barπ_single G k y 1 1).symm

/-! ### The induced chain map `k[u] : k[𝔉'_•] → φ^* k[𝔉'{}'_•]` -/

variable {C : Cover.{0} G} {C' : Cover.{0} G'} {φ} (M : Cover.Morph φ C C')

variable (k) in
/-- `k[u]` in degree `n`. -/
def zuF (n : ℕ) : mpX C k n ⟶ Rep.res φ (mpX C' k n) :=
  Rep.ofHom ⟨(coeffLinearEquiv k).symm.toLinearMap ∘ₗ Finsupp.lmapDomain k k M.uF ∘ₗ (coeffLinearEquiv k).toLinearMap,
    fun g => by
      refine LinearMap.ext fun x => ?_
      simp only [LinearMap.coe_comp, Function.comp_apply, LinearEquiv.coe_coe, Representation.ofMulAction_def,
        LinearEquiv.apply_symm_apply, Finsupp.lmapDomain_apply, MonoidHom.coe_comp]
      rw [← Finsupp.mapDomain_comp, ← Finsupp.mapDomain_comp]
      congr 2
      funext y
      exact M.uF_smul g y⟩

lemma zuF_hom_apply (n : ℕ) (x : k[C.F' n]) :
    (zuF k M n).hom x = (coeffLinearEquiv k).symm (Finsupp.mapDomain M.uF (coeffLinearEquiv k x)) := rfl

lemma dF_mapDomain_uF (n : ℕ) (x : C.F' (n + 1) →₀ k) :
    dF k n (Finsupp.mapDomain M.uF x) = Finsupp.mapDomain M.uF (dF k n x) := by
  induction x using Finsupp.induction_linear with
  | zero => simp
  | add x y hx hy => rw [Finsupp.mapDomain_add, map_add, hx, hy, map_add, Finsupp.mapDomain_add]
  | single x c =>
    simp only [Finsupp.mapDomain_single, dF, Finsupp.linearCombination_single, Finsupp.mapDomain_smul]
    congr 1
    simp only [fBdS, altSum, Finsupp.mapDomain_finsetSum, Finsupp.mapDomain_single, Morph.uF_del]

variable (k) in
/-- **`k[u]`**, a `Γ`-chain map `k[𝔉'_•] → φ^* k[𝔉'{}'_•]`. -/
def zu : mpCx C k ⟶ resC φ (mpCx C' k) where
  f n := zuF k M n
  comm' i j hij := by
    obtain rfl : j + 1 = i := hij
    rw [mpCx_d]
    show zuF k M (j + 1) ≫ (Rep.resFunctor φ).map ((mpCx C' k).d (j + 1) j) = mpD C k j ≫ zuF k M j
    rw [mpCx_d]
    apply Rep.hom_ext; apply Representation.IntertwiningMap.ext; refine LinearMap.ext fun x => ?_
    show (coeffLinearEquiv k).symm (dF k j (coeffLinearEquiv k
        ((coeffLinearEquiv k).symm (Finsupp.mapDomain M.uF (coeffLinearEquiv k x))))) =
      (coeffLinearEquiv k).symm (Finsupp.mapDomain M.uF (coeffLinearEquiv k ((coeffLinearEquiv k).symm
      (dF k j (coeffLinearEquiv k x)))))
    rw [LinearEquiv.apply_symm_apply, LinearEquiv.apply_symm_apply, dF_mapDomain_uF]

/-- `k[u]` preserves the augmentations. -/
lemma zu_aug : (zu k M).f 0 ≫ (Rep.resFunctor φ).map (mpEps C' k) = mpEps C k := by
  apply Rep.hom_ext; apply Representation.IntertwiningMap.ext; refine LinearMap.ext fun x => ?_
  show augL k (coeffLinearEquiv k ((coeffLinearEquiv k).symm (Finsupp.mapDomain M.uF (coeffLinearEquiv k x)))) =
    augL k (coeffLinearEquiv k x)
  rw [LinearEquiv.apply_symm_apply]
  induction (coeffLinearEquiv k x) using Finsupp.induction_linear with
  | zero => simp
  | add x y hx hy => rw [Finsupp.mapDomain_add, map_add, hx, hy, map_add]
  | single x c => simp [augL, Finsupp.mapDomain_single]

/-- **`u^*` is precomposition with `k[u]`** (on `Hom_{Γ'}(k[𝔉'{}'_•], A')`, through `linYonIso`). -/
lemma pullMap_zu (A : Rep.{0} k G') :
    (linYonIso A).hom ≫ M.pullMap A = resY φ (mpCx C' k) A ≫ preY (zu k M) (Rep.res φ A) ≫ (linYonIso (Rep.res φ A)).hom := by
  ext i g : 3
  funext s
  show A.ρ (M.lab (s.1 0).2).1 (g.hom (MonoidAlgebra.single (M.uS s).toF' 1)) =
    g.hom ((zuF k M i).hom (MonoidAlgebra.single s.toF' 1))
  rw [zuF_hom_apply, coeff_single, Finsupp.mapDomain_single, Morph.uF_toF']
  have := Rep.hom_comm_apply (show mpX C' k i ⟶ A from g) (M.lab (s.1 0).2).1 (MonoidAlgebra.single (M.uS s).toF' 1)
  refine this.symm.trans ?_
  congr 1
  rw [Representation.ofMulAction_single]
  rfl

/-! ### The naturality theorem -/

variable {H : C.Hyp} {H' : C'.Hyp}

/-- **The two lifts are homotopic**: `k[u] ≫ φ^* e' ≃ e ≫ β` as `Γ`-maps `k[𝔉'_•] → φ^* bar_{Γ'}`. -/
def liftsHomotopy (hAC : (yCx H k).toCx.Acyclic) (hAC' : (yCx H' k).toCx.Acyclic) :
    Homotopy (zu k M ≫ ((Rep.resFunctor φ).mapHomologicalComplex _).map (mpBarEquiv k H' hAC').hom)
      ((mpBarEquiv k H hAC).hom ≫ barMap φ) := by
  refine homotopy' (P := mpRes H k hAC) (resBarAug φ) _ _ ?_
  have h1 : (mpBarEquiv k H hAC).hom.f 0 ≫ barEps k G = mpEps C k := by
    have := HomologicalComplex.congr_hom (ProjectiveResolution.homotopyEquiv_hom_π (mpRes H k hAC)
      (Rep.barResolution k G)) 0
    simp only [HomologicalComplex.comp_f] at this
    exact this.trans (mpAug_f_zero k)
  have h2 : (mpBarEquiv k H' hAC').hom.f 0 ≫ barEps k G' = mpEps C' k := by
    have := HomologicalComplex.congr_hom (ProjectiveResolution.homotopyEquiv_hom_π (mpRes H' k hAC')
      (Rep.barResolution k G')) 0
    simp only [HomologicalComplex.comp_f] at this
    exact this.trans (mpAug_f_zero k)
  have L : (zu k M).f 0 ≫ (Rep.resFunctor φ).map ((mpBarEquiv k H' hAC').hom.f 0) ≫
      (Rep.resFunctor φ).map (barEps k G') = mpEps C k := by
    have e : (Rep.resFunctor φ).map ((mpBarEquiv k H' hAC').hom.f 0) ≫ (Rep.resFunctor φ).map (barEps k G') =
        (Rep.resFunctor φ).map (mpEps C' k) := by rw [← Functor.map_comp]; exact congrArg _ h2
    exact (congrArg ((zu k M).f 0 ≫ ·) e).trans (zu_aug M)
  have R : (mpBarEquiv k H hAC).hom.f 0 ≫ (barMap φ).f 0 ≫ (resBarAug φ).ε = mpEps C k := by
    rw [barMap_aug]; exact h1
  exact ((Category.assoc _ _ _).trans (L.trans R.symm)).trans (Category.assoc _ _ _).symm

/-- `Hom(-, A)` sends homotopic maps to homotopic maps. -/
def preYHomotopy {X Y : ChainComplex (Rep.{0} k G) ℕ} {f g : X ⟶ Y} (h : Homotopy f g) (A : Rep.{0} k G) :
    Homotopy (preY f A) (preY g A) :=
  Homotopy.unop ((((linearYoneda k (Rep.{0} k G)).obj A).rightOp).mapHomotopy h)

/-- **[C] `thm:mpnatural` (2)**: under the comparison isomorphisms, `H^n(u^*)` is the restriction along `φ`. -/
theorem pull_natural (hAC : (yCx H k).toCx.Acyclic) (hAC' : (yCx H' k).toCx.Acyclic) (A : Rep.{0} k G') (n : ℕ) :
    groupCohomology.map φ (𝟙 (Rep.res φ A)) n ≫ (mpComparisonNat H hAC (Rep.res φ A) n).hom =
      (mpComparisonNat H' hAC' A n).hom ≫ HomologicalComplex.homologyMap (M.pullMap A) n := by
  have hh := (preYHomotopy (liftsHomotopy M hAC hAC') (Rep.res φ A)).homologyMap_eq n
  have L : cochainsMap φ (𝟙 (Rep.res φ A)) ≫ (inhomogeneousCochainsIso (Rep.res φ A)).hom ≫
      preY (mpBarEquiv k H hAC).hom (Rep.res φ A) ≫ (linYonIso (Rep.res φ A)).hom =
      (inhomogeneousCochainsIso A).hom ≫ resY φ _ A ≫ preY ((mpBarEquiv k H hAC).hom ≫ barMap φ) (Rep.res φ A) ≫
        (linYonIso (Rep.res φ A)).hom := by
    rw [← Category.assoc, cochainsMap_barMap, preY_comp]
    simp only [Category.assoc]
  have R : (inhomogeneousCochainsIso A).hom ≫ preY (mpBarEquiv k H' hAC').hom A ≫ (linYonIso A).hom ≫ M.pullMap A =
      (inhomogeneousCochainsIso A).hom ≫ resY φ _ A ≫
        preY (zu k M ≫ ((Rep.resFunctor φ).mapHomologicalComplex _).map (mpBarEquiv k H' hAC').hom) (Rep.res φ A) ≫
          (linYonIso (Rep.res φ A)).hom := by
    rw [pullMap_zu, ← Category.assoc (preY _ A), preY_resY, preY_comp]
    simp only [Category.assoc]
  have hL := congrArg (fun χ => HomologicalComplex.homologyMap χ n) L
  have hR := congrArg (fun χ => HomologicalComplex.homologyMap χ n) R
  simp only [HomologicalComplex.homologyMap_comp] at hL hR
  show HomologicalComplex.homologyMap (cochainsMap φ (𝟙 (Rep.res φ A))) n ≫
      (HomologicalComplex.homologyMap (inhomogeneousCochainsIso (Rep.res φ A)).hom n ≫
        HomologicalComplex.homologyMap (preY (mpBarEquiv k H hAC).hom (Rep.res φ A)) n ≫
          HomologicalComplex.homologyMap (linYonIso (Rep.res φ A)).hom n) =
    (HomologicalComplex.homologyMap (inhomogeneousCochainsIso A).hom n ≫
      HomologicalComplex.homologyMap (preY (mpBarEquiv k H' hAC').hom A) n ≫
        HomologicalComplex.homologyMap (linYonIso A).hom n) ≫ HomologicalComplex.homologyMap (M.pullMap A) n
  simp only [Category.assoc]
  rw [hL, hR, hh]

/-- **`H^n(u^*)` is determined by the comparisons and the restriction** ([C] `thm:mpnatural` (2), solved form). -/
theorem pull_homology_eq (hAC : (yCx H k).toCx.Acyclic) (hAC' : (yCx H' k).toCx.Acyclic) (A : Rep.{0} k G') (n : ℕ) :
    HomologicalComplex.homologyMap (M.pullMap A) n =
      (mpComparisonNat H' hAC' A n).inv ≫ groupCohomology.map φ (𝟙 (Rep.res φ A)) n ≫
        (mpComparisonNat H hAC (Rep.res φ A) n).hom := by
  rw [pull_natural M hAC hAC', Iso.inv_hom_id_assoc]

/-- **Independence of the labels on cohomology** ([C] `thm:mpnatural` (3)): two morphisms of MP data along the same `φ`
induce the same map on cohomology. -/
theorem pull_label_indep (M₂ : Cover.Morph φ C C') (hAC : (yCx H k).toCx.Acyclic) (hAC' : (yCx H' k).toCx.Acyclic)
    (A : Rep.{0} k G') (n : ℕ) :
    HomologicalComplex.homologyMap (M.pullMap A) n = HomologicalComplex.homologyMap (M₂.pullMap A) n := by
  rw [pull_homology_eq M hAC hAC', pull_homology_eq M₂ hAC hAC']

/-! ### Refinement invariance ([C] `thm:mpnatural` (3)) -/

variable {C₁ C₂ : Cover.{0} G} {H₁ : C₁.Hyp} {H₂ : C₂.Hyp} (R : Cover.Morph (MonoidHom.id G) C₁ C₂)

/-- **Refinement invariance, on cohomology**: for a morphism of MP data of one group (`φ = id`, e.g. a refinement with
its label table), `H^n(u^*)` is the composite of the comparison isomorphisms — it induces the identity of `H^n(Γ; A)`. -/
theorem refine_homology_eq (hAC₁ : (yCx H₁ k).toCx.Acyclic) (hAC₂ : (yCx H₂ k).toCx.Acyclic) (A : Rep.{0} k G)
    (n : ℕ) :
    HomologicalComplex.homologyMap (R.pullMap A) n =
      (mpComparisonNat H₂ hAC₂ A n).inv ≫ (mpComparisonNat H₁ hAC₁ A n).hom := by
  rw [pull_homology_eq R hAC₁ hAC₂]
  erw [groupCohomology.map_id, Category.id_comp]
  rfl

/-- **Refinement invariance** ([C] `thm:mpnatural` (3)): `u^*` is a quasi-isomorphism in every degree. -/
theorem refine_quasiIso (hAC₁ : (yCx H₁ k).toCx.Acyclic) (hAC₂ : (yCx H₂ k).toCx.Acyclic) (A : Rep.{0} k G) :
    QuasiIso (R.pullMap A) where
  quasiIsoAt n := by
    rw [quasiIsoAt_iff_isIso_homologyMap, refine_homology_eq R hAC₁ hAC₂]
    infer_instance

end Cover

end GL2.MP
