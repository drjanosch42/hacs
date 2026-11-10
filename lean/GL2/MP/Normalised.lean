import GL2.MP.Bredon

/-!
# MP-L3, open item (slot 22): the normalised MP complex is homotopy equivalent to the unnormalised one

[GT] `gt_part_MP.tex`, rows `def:mp-simpK`, `thm:mp-simpK-alt`, `def:mp-normcx`, `thm:mp-norm-heq`,
`thm:mp-norm-dgn`, `thm:mp-comparison-normalised`, backing [C] `def:mpset` (the normalised complex `C^•(S; M)`, functions
vanishing on degenerate strings) and [C] `thm:mpcomparison` (2)–(4) for the normalised complex.

* `simpK C k`: the labelled Čech nerve as a **simplicial object** in `Rep k Γ` (`[n] ↦ k[𝔉'_n]`, `θ ↦ (x ↦ x ∘ θ)`), and
  `simpK_altIso`: its alternating face map complex is `k[𝔉'_•]` (`mpCx`).
* For any simplicial object `X` in `Rep k Γ` and any `A`: `normCx X A ⊆ Hom_Γ(K[X], A)`, the cochains **vanishing on the
  degeneracies** (`DegeneraciesVanish`, mathlib's Dold–Kan layer), is a subcomplex, and its inclusion is a **homotopy
  equivalence** (`normHEq`): the retraction is precomposition with mathlib's `PInfty` (which kills the degeneracies), and
  `PInfty ≃ id` (`homotopyPInftyToId`) — for every ring, nothing inverted.
* `degeneraciesVanish_iff` (`thm:mp-norm-dgn`): for `X = simpK C k`, `σ_i ≫ φ = 0` for all `i` iff the MP cochain
  `homEquiv φ` vanishes on every degenerate string `s_i(s)` — so `normCx` is [C]'s normalised `C^•(S; M)` read through
  `Hom_Γ(k[𝔉'_n], A) = C^n(S; A)`.
* `nmpCx`: **[C]'s normalised MP complex in MP coordinates** (`nmpSub`: `f : S_n → A` with `f(s_i(s)) = 0`), a
  subcomplex of the unnormalised `C^•(S; A)` (`dC_mem_nmp`), isomorphic to `normCx` (`normIsoNmp`); its inclusion is a
  quasi-isomorphism (`nmpIncl_quasiIso`, a homotopy equivalence).
* `mpComparisonNorm`, `mpComparisonNmp`: `H^n(Γ; A) ≅ H^n(C^•(S; A))` for the normalised complex, under the hypotheses of
  the comparison theorem; for `SL₂(ℤ)` on `M₇` and `R_ℤ` unconditionally (`model7_comparisonNorm`, `rz_comparisonNorm`).
-/

noncomputable section

namespace GL2.MP

open CategoryTheory AlgebraicTopology AlgebraicTopology.DoldKan Finsupp MonoidAlgebra Opposite

namespace Cover

variable {G : Type} [Group G] (C : Cover.{0} G) (k : Type) [CommRing k]

/-! ### The labelled Čech nerve as a simplicial object -/

variable {C} in
/-- `x ↦ x ∘ θ`. -/
def F'map {m n : ℕ} (θ : Fin (m + 1) →o Fin (n + 1)) (x : C.F' n) : C.F' m :=
  ⟨x.1 ∘ θ, by obtain ⟨p, hp⟩ := x.2; exact ⟨p, fun i => hp _⟩⟩

variable {C} in
lemma F'map_smul {m n : ℕ} (θ : Fin (m + 1) →o Fin (n + 1)) (g : G) (x : C.F' n) :
    F'map θ (g • x) = g • F'map θ x := rfl

variable {C} in
/-- `k[x ↦ x ∘ θ]`. -/
def kmap {m n : ℕ} (θ : Fin (m + 1) →o Fin (n + 1)) : mpX C k n ⟶ mpX C k m :=
  Rep.ofHom ⟨(coeffLinearEquiv k).symm.toLinearMap ∘ₗ Finsupp.lmapDomain k k (F'map θ) ∘ₗ
      (coeffLinearEquiv k).toLinearMap, fun g => by
    refine LinearMap.ext fun x => ?_
    simp only [LinearMap.coe_comp, Function.comp_apply, LinearEquiv.coe_coe, Representation.ofMulAction_def,
      LinearEquiv.apply_symm_apply, Finsupp.lmapDomain_apply]
    rw [← Finsupp.mapDomain_comp, ← Finsupp.mapDomain_comp]
    rfl⟩

variable {C} in
lemma kmap_hom_apply {m n : ℕ} (θ : Fin (m + 1) →o Fin (n + 1)) (x : k[C.F' n]) :
    (kmap k θ).hom x = (coeffLinearEquiv k).symm (Finsupp.mapDomain (F'map θ) (coeffLinearEquiv k x)) := rfl

/-- `K[X]`, the alternating face map complex. -/
abbrev KX {k : Type} [CommRing k] (X : SimplicialObject (Rep.{0} k G)) : ChainComplex (Rep.{0} k G) ℕ :=
  AlternatingFaceMapComplex.obj X

/-- **The labelled Čech nerve as a simplicial object** `[n] ↦ k[𝔉'_n]` in `Rep k Γ`. -/
def simpK : SimplicialObject (Rep.{0} k G) where
  obj Δ := mpX C k Δ.unop.len
  map θ := kmap k θ.unop.toOrderHom
  map_id Δ := by
    apply Rep.hom_ext; apply Representation.IntertwiningMap.ext; refine LinearMap.ext fun x => ?_
    show (coeffLinearEquiv k).symm (Finsupp.mapDomain _ (coeffLinearEquiv k x)) = x
    have : (F'map (C := C) (𝟙 Δ.unop : Δ.unop ⟶ Δ.unop).toOrderHom) = id := funext fun y => rfl
    rw [show (𝟙 Δ : Δ ⟶ Δ).unop = 𝟙 Δ.unop from rfl, this, Finsupp.mapDomain_id, LinearEquiv.symm_apply_apply]
  map_comp θ θ' := by
    apply Rep.hom_ext; apply Representation.IntertwiningMap.ext; refine LinearMap.ext fun x => ?_
    show (coeffLinearEquiv k).symm (Finsupp.mapDomain _ (coeffLinearEquiv k x)) =
      (coeffLinearEquiv k).symm (Finsupp.mapDomain _ (coeffLinearEquiv k ((coeffLinearEquiv k).symm
        (Finsupp.mapDomain _ (coeffLinearEquiv k x)))))
    rw [LinearEquiv.apply_symm_apply, ← Finsupp.mapDomain_comp]
    rfl

lemma simpK_δ_hom_apply {n : ℕ} (i : Fin (n + 2)) (y : C.F' (n + 1)) (r : k) :
    ((simpK C k).δ i).hom (MonoidAlgebra.single y r) = MonoidAlgebra.single (F'.del i y) r := by
  show (coeffLinearEquiv k).symm (Finsupp.mapDomain _ (coeffLinearEquiv k (MonoidAlgebra.single y r))) = _
  rw [coeff_single, Finsupp.mapDomain_single]
  rfl

lemma simpK_σ_hom_apply {n : ℕ} (i : Fin (n + 1)) (y : C.F' n) (r : k) :
    ((simpK C k).σ i).hom (MonoidAlgebra.single y r) = MonoidAlgebra.single (F'map i.predAboveOrderHom y) r := by
  show (coeffLinearEquiv k).symm (Finsupp.mapDomain _ (coeffLinearEquiv k (MonoidAlgebra.single y r))) = _
  rw [coeff_single, Finsupp.mapDomain_single]
  rfl

lemma coeff_symm_single {X : Type} (x : X) (r : k) :
    (coeffLinearEquiv k).symm (Finsupp.single x r) = MonoidAlgebra.single x r := by
  rw [← coeff_single, LinearEquiv.symm_apply_apply]

lemma mpD_single {n : ℕ} (y : C.F' (n + 1)) (r : k) :
    (mpD C k n).hom (MonoidAlgebra.single y r) =
      ∑ i : Fin (n + 2), ((-1 : k) ^ (i : ℕ) * r) • MonoidAlgebra.single (F'.del i y) 1 := by
  rw [mpD_hom_apply, coeff_single, dF, Finsupp.linearCombination_single, fBdS, altSum, Finset.smul_sum, map_sum]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [Finsupp.smul_single, smul_eq_mul, coeff_symm_single, MonoidAlgebra.smul_single', mul_one, mul_comm]

lemma mpD_eq_sum (j : ℕ) :
    mpD C k j = ∑ i : Fin (j + 2), ((-1 : ℤ) ^ (i : ℕ)) • kmap (C := C) k (Fin.succAboveOrderEmb i).toOrderHom := by
  apply Rep.hom_ext; apply Representation.IntertwiningMap.ext; refine LinearMap.ext fun x => ?_
  induction x using MonoidAlgebra.induction_linear with
  | zero => simp
  | add x y hx hy => rw [map_add, map_add, hx, hy]
  | single y r =>
    rw [Rep.sum_hom]
    simp only [Rep.zsmul_hom, Representation.IntertwiningMap.toLinearMap_apply]
    rw [show (mpD C k j).hom (MonoidAlgebra.single y r) = _ from mpD_single C k y r]
    rw [Representation.IntertwiningMap.sum_apply]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [Representation.IntertwiningMap.coe_zsmul, Pi.smul_apply, kmap_hom_apply, coeff_single, Finsupp.mapDomain_single,
      coeff_symm_single, MonoidAlgebra.smul_single', mul_one]
    simp only [MonoidAlgebra.smul_single, zsmul_eq_mul]
    push_cast
    rfl

/-- **The alternating face map complex of `simpK` is `k[𝔉'_•]`.** -/
def simpK_altIso : KX (simpK C k) ≅ mpCx C k :=
  HomologicalComplex.Hom.isoOfComponents (fun n => Iso.refl _) (by
    rintro i j (rfl : j + 1 = i)
    rw [Iso.refl_hom, Iso.refl_hom]
    erw [Category.id_comp, Category.comp_id]
    rw [mpCx_d]
    have hd : (KX (simpK C k)).d (j + 1) j =
        ∑ i : Fin (j + 2), ((-1 : ℤ) ^ (i : ℕ)) • (simpK C k).δ i := by simp
    exact (mpD_eq_sum C k j).trans hd.symm)

/-- `Hom(-, A)` of an isomorphism of complexes. -/
def preYIso {k : Type} [CommRing k] {X Y : ChainComplex (Rep.{0} k G) ℕ} (e : X ≅ Y) (A : Rep.{0} k G) :
    Y.linearYonedaObj k A ≅ X.linearYonedaObj k A where
  hom := preY e.hom A
  inv := preY e.inv A
  hom_inv_id := by rw [← preY_comp, e.inv_hom_id, preY_id]
  inv_hom_id := by rw [← preY_comp, e.hom_inv_id, preY_id]

/-! ### The normalised cochains of a simplicial object: a homotopy equivalent subcomplex -/

section Norm

variable {k} (X : SimplicialObject (Rep.{0} k G)) (A : Rep.{0} k G)

/-- **The normalised cochains in degree `n`**: `φ : X_n → A` with `QInfty ≫ φ = 0`, i.e. vanishing on the degeneracies. -/
def normSub (n : ℕ) : Submodule k ((KX X).X n ⟶ A) :=
  LinearMap.ker (Linear.leftComp k A ((QInfty (X := X)).f n))

lemma mem_normSub_iff (n : ℕ) (φ : (KX X).X n ⟶ A) : φ ∈ normSub X A n ↔ DegeneraciesVanish (X := X) φ :=
  (degeneraciesVanish_iff_QInfty_f_comp φ).symm

lemma QInfty_comp_of_mem {n : ℕ} {φ : (KX X).X n ⟶ A} (h : φ ∈ normSub X A n) : (QInfty (X := X)).f n ≫ φ = 0 := h

lemma PInfty_comp_of_mem {n : ℕ} {φ : (KX X).X n ⟶ A} (h : φ ∈ normSub X A n) : (PInfty (X := X)).f n ≫ φ = φ := by
  have := PInfty_f_add_QInfty_f (X := X) n =≫ φ
  rw [Preadditive.add_comp, QInfty_comp_of_mem X A h, add_zero, Category.id_comp] at this
  exact this

/-- The coboundary restricted to the normalised cochains. -/
def normD (n : ℕ) : normSub X A n →ₗ[k] normSub X A (n + 1) where
  toFun φ := ⟨(KX X).d (n + 1) n ≫ φ.1, by
    show (QInfty (X := X)).f (n + 1) ≫ (KX X).d (n + 1) n ≫ φ.1 = 0
    rw [← Category.assoc, (QInfty (X := X)).comm (n + 1) n, Category.assoc, QInfty_comp_of_mem X A φ.2,
      Limits.comp_zero]⟩
  map_add' φ ψ := Subtype.ext (Preadditive.comp_add _ _ _ _ _ _)
  map_smul' r φ := Subtype.ext (Linear.comp_smul _ _ _ _ _ _)

/-- **The normalised cochain complex** `Hom_Γ(K[X], A)_{nd}`. -/
def normCx : CochainComplex (ModuleCat.{0} k) ℕ :=
  CochainComplex.of (fun n => ModuleCat.of k (normSub X A n)) (fun n => ModuleCat.ofHom (normD X A n)) fun n => by
    ext φ : 2
    apply Subtype.ext
    show (KX X).d (n + 2) (n + 1) ≫ (KX X).d (n + 1) n ≫ φ.1 = 0
    rw [← Category.assoc, HomologicalComplex.d_comp_d, Limits.zero_comp]

/-- The inclusion of the normalised cochains. -/
def normι : normCx X A ⟶ (KX X).linearYonedaObj k A where
  f n := ModuleCat.ofHom (normSub X A n).subtype
  comm' i j hij := by
    obtain rfl : i + 1 = j := hij
    simp only [normCx, CochainComplex.of_d]
    ext φ : 2
    rfl

/-- The retraction `φ ↦ PInfty ≫ φ`. -/
def normr : (KX X).linearYonedaObj k A ⟶ normCx X A where
  f n := ModuleCat.ofHom ((Linear.leftComp k A ((PInfty (X := X)).f n)).codRestrict (normSub X A n) fun φ => by
    show (QInfty (X := X)).f n ≫ (PInfty (X := X)).f n ≫ φ = 0
    rw [← Category.assoc, QInfty_f_comp_PInfty_f, Limits.zero_comp])
  comm' i j hij := by
    obtain rfl : i + 1 = j := hij
    simp only [normCx, CochainComplex.of_d]
    ext φ : 2
    apply Subtype.ext
    show (KX X).d (i + 1) i ≫ (PInfty (X := X)).f i ≫ φ = (PInfty (X := X)).f (i + 1) ≫ (KX X).d (i + 1) i ≫ φ
    exact (Category.assoc _ _ _).symm.trans ((congrArg (· ≫ (show (KX X).X i ⟶ A from φ))
      ((PInfty (X := X)).comm (i + 1) i).symm).trans (Category.assoc _ _ _))

lemma normr_normι : normι X A ≫ normr X A = 𝟙 _ := by
  ext n φ : 3
  apply Subtype.ext
  exact PInfty_comp_of_mem X A φ.2

lemma normι_normr : normr X A ≫ normι X A = preY (PInfty (X := X)) A := by
  ext n φ : 3
  rfl

/-- **The normalised cochains are a homotopy equivalent subcomplex** (every ring, nothing inverted). -/
def normHEq : HomotopyEquiv (normCx X A) ((KX X).linearYonedaObj k A) where
  hom := normι X A
  inv := normr X A
  homotopyHomInvId := Homotopy.ofEq (normr_normι X A)
  homotopyInvHomId := (Homotopy.ofEq (normι_normr X A)).trans
    ((preYHomotopy (homotopyPInftyToId X) A).trans (Homotopy.ofEq (preY_id _ A)))

end Norm

/-! ### For the labelled Čech nerve: vanishing on degeneracies = vanishing on degenerate strings -/

section MPNorm

variable {C k} (A : Rep.{0} k G)

lemma simpK_σ_single {n : ℕ} (i : Fin (n + 1)) (s : C.MPSet n) :
    ((simpK C k).σ i).hom (MonoidAlgebra.single s.toF' 1) = MonoidAlgebra.single (dgn i s).toF' 1 :=
  simpK_σ_hom_apply C k i s.toF' 1

/-- **[C] `def:mpset`'s normalisation in MP coordinates**: `σ_i ≫ φ = 0` for all `i` iff the MP cochain `homEquiv φ`
vanishes on every degenerate string `s_i(s)`. -/
theorem degeneraciesVanish_iff {n : ℕ} (φ : (KX (simpK C k)).X (n + 1) ⟶ A) :
    DegeneraciesVanish (X := simpK C k) (n := n + 1) φ ↔
      ∀ (i : Fin (n + 1)) (s : C.MPSet n), homEquiv A (n + 1) (show mpX C k (n + 1) ⟶ A from φ) (dgn i s) = 0 := by
  rw [degeneraciesVanish_succ_iff]
  constructor
  · intro h i s
    have h1 : ((simpK C k).σ i ≫ φ).hom (MonoidAlgebra.single s.toF' 1) = 0 := by rw [h i]; rfl
    show φ.hom (MonoidAlgebra.single (dgn i s).toF' 1) = 0
    rw [← simpK_σ_single]
    exact h1
  · intro h i
    have key : ∀ x : C.F' n, φ.hom (((simpK C k).σ i).hom (MonoidAlgebra.single x 1)) = 0 := fun x => by
      rw [← F'.smul_norm x, simpK_σ_hom_apply, F'map_smul, ← Representation.ofMulAction_single]
      refine (Rep.hom_comm_apply φ _ _).trans ?_
      have h0 := h i (F'.norm x)
      exact (congrArg (A.ρ _) h0).trans (map_zero _)
    have : (show mpX C k n ⟶ A from (simpK C k).σ i ≫ φ) = 0 :=
      (homE A n).injective (Subtype.ext (funext fun x => (key x).trans (by rw [map_zero]; rfl)))
    exact this

end MPNorm

/-! ### [C]'s normalised MP complex, in MP coordinates -/

section NMP

variable {C k} (A : Rep.{0} k G)

/-- **[C] `def:mpset`'s normalised cochains**: `f : S_n → A` vanishing on the degenerate strings `s_i(s)`. -/
def nmpSub : (n : ℕ) → Submodule k (C.Coch A n)
  | 0 => ⊤
  | m + 1 =>
    { carrier := {f | ∀ (i : Fin (m + 1)) (s : C.MPSet m), f (dgn i s) = 0}
      add_mem' := fun hf hg i s => by simp [hf i s, hg i s]
      zero_mem' := fun i s => rfl
      smul_mem' := fun r f hf i s => by simp [hf i s] }

lemma mem_normSub_iff_nmp (n : ℕ) (φ : (KX (simpK C k)).X n ⟶ A) :
    φ ∈ normSub (simpK C k) A n ↔ homEquiv A n (show mpX C k n ⟶ A from φ) ∈ nmpSub A n := by
  cases n with
  | zero =>
    refine iff_of_true ?_ trivial
    show (QInfty (X := simpK C k)).f 0 ≫ φ = 0
    rw [QInfty_f_0, Limits.zero_comp]
  | succ m =>
    rw [mem_normSub_iff]
    exact degeneraciesVanish_iff A φ

/-- `Hom_Γ(k[𝔉'_n], A)_{nd} ≅ C^n_{nd}(S; A)` (the restriction of `homEquiv`). -/
def nmpEquiv (n : ℕ) : normSub (simpK C k) A n ≃ₗ[k] nmpSub (C := C) A n where
  toFun φ := ⟨homEquiv A n (show mpX C k n ⟶ A from φ.1), (mem_normSub_iff_nmp A n φ.1).1 φ.2⟩
  map_add' φ ψ := Subtype.ext ((homEquiv A n).map_add _ _)
  map_smul' r φ := Subtype.ext ((homEquiv A n).map_smul _ _)
  invFun f := ⟨(homEquiv A n).symm f.1, (mem_normSub_iff_nmp A n _).2 (by
    show homEquiv A n ((homEquiv A n).symm f.1) ∈ nmpSub A n
    rw [LinearEquiv.apply_symm_apply]; exact f.2)⟩
  left_inv φ := Subtype.ext ((homEquiv A n).symm_apply_apply _)
  right_inv f := Subtype.ext ((homEquiv A n).apply_symm_apply _)

/-- The MP coboundary preserves the normalised cochains. -/
lemma dC_mem_nmp (n : ℕ) (f : C.Coch A n) (hf : f ∈ nmpSub A n) : C.dC A n f ∈ nmpSub A (n + 1) := by
  obtain ⟨φ, rfl⟩ := (homEquiv A n).surjective f
  have hφ : (show (KX (simpK C k)).X n ⟶ A from φ) ∈ normSub (simpK C k) A n := (mem_normSub_iff_nmp A n φ).2 hf
  have h2 := (normD (simpK C k) A n ⟨_, hφ⟩).2
  rw [mem_normSub_iff_nmp] at h2
  have e : homEquiv A (n + 1) (show mpX C k (n + 1) ⟶ A from (KX (simpK C k)).d (n + 1) n ≫ φ) =
      C.dC A n (homEquiv A n φ) := by
    rw [← homEquiv_d]
    congr 1
    have := (simpK_altIso C k).hom.comm (n + 1) n
    simp only [simpK_altIso] at this
    erw [Category.id_comp, Category.comp_id] at this
    rw [← this, mpCx_d]
    rfl
  rw [← e]
  exact h2

/-- **[C]'s normalised MP complex** `C^•(S; A)` (functions vanishing on degenerate strings), unnormalised coboundary
restricted. -/
def nmpCx : CochainComplex (ModuleCat.{0} k) ℕ :=
  CochainComplex.of (fun n => ModuleCat.of k (nmpSub (C := C) A n))
    (fun n => ModuleCat.ofHom ((C.dC A n).restrict fun f hf => dC_mem_nmp A n f hf)) fun n => by
      ext f : 2
      exact Subtype.ext (C.dC_dC A n f.1)

/-- **`Hom_Γ(k[𝔉'_•], A)_{nd} ≅ C^•_{nd}(S; A)`** as cochain complexes. -/
def normIsoNmp : normCx (simpK C k) A ≅ nmpCx (C := C) A :=
  HomologicalComplex.Hom.isoOfComponents (fun n => (nmpEquiv A n).toModuleIso) (by
    rintro i j (rfl : i + 1 = j)
    simp only [normCx, nmpCx, CochainComplex.of_d]
    ext φ : 2
    apply Subtype.ext
    show C.dC A i (homEquiv A i (show mpX C k i ⟶ A from φ.1)) =
      homEquiv A (i + 1) (show mpX C k (i + 1) ⟶ A from (KX (simpK C k)).d (i + 1) i ≫ φ.1)
    rw [← homEquiv_d]
    congr 1
    have := (simpK_altIso C k).hom.comm (i + 1) i
    simp only [simpK_altIso] at this
    erw [Category.id_comp, Category.comp_id] at this
    rw [← this, mpCx_d]
    rfl)

/-- The inclusion `C^•_{nd}(S; A) ⊆ C^•_{un}(S; A)`. -/
def nmpIncl : nmpCx (C := C) A ⟶ C.mpComplex A :=
  CochainComplex.ofHom (fun n => ModuleCat.ofHom (nmpSub (C := C) A n).subtype) fun n => by
    simp only [nmpCx, mpComplex, CochainComplex.of_d]
    ext f : 2
    rfl

lemma nmpIncl_eq : (normIsoNmp A).inv ≫ normι (simpK C k) A ≫ (preYIso (simpK_altIso C k) A).inv ≫ (linYonIso A).hom =
    nmpIncl A := by
  ext n f : 3
  have e : (simpK_altIso C k).inv.f n = 𝟙 _ := rfl
  show homEquiv A n ((simpK_altIso C k).inv.f n ≫ (homEquiv A n).symm f.1) = f.1
  rw [e]
  erw [Category.id_comp]
  exact (homEquiv A n).apply_symm_apply f.1

/-- **The inclusion of [C]'s normalised MP complex into the unnormalised one is a quasi-isomorphism** (indeed a homotopy
equivalence), over every ring, for every representation. -/
theorem nmpIncl_quasiIso : QuasiIso (nmpIncl (C := C) A) := by
  rw [← nmpIncl_eq]
  have : QuasiIso (normι (simpK C k) A) := (inferInstance : QuasiIso (normHEq (simpK C k) A).hom)
  infer_instance

end NMP

/-! ### The comparison theorem for the normalised complex -/

section NormComparison

variable {C k}

variable (H : C.Hyp)

/-- **[C] `thm:mpcomparison` (4) for the normalised MP complex**: `H^n(Γ; A) ≅ H^n(C^•_{nd}(S; A))`, the normalised
cochains (vanishing on the degenerate strings, `degeneraciesVanish_iff`), under (AC) over `k`. -/
def mpComparisonNorm (hAC : (yCx H k).toCx.Acyclic) (A : Rep.{0} k G) (n : ℕ) :
    groupCohomology A n ≅ (normCx (simpK C k) A).homology n :=
  mpComparisonNat H hAC A n ≪≫ HomologicalComplex.homologyMapIso (linYonIso A).symm n ≪≫
    HomologicalComplex.homologyMapIso (preYIso (simpK_altIso C k) A) n ≪≫
    ((normHEq (simpK C k) A).toHomologyIso n).symm

/-- **[C] `thm:mpcomparison` (4) for [C]'s normalised MP complex** `C^•(S; A)` (functions on strings vanishing on the
degenerate ones), under (AC) over `k`. -/
def mpComparisonNmp (hAC : (yCx H k).toCx.Acyclic) (A : Rep.{0} k G) (n : ℕ) :
    groupCohomology A n ≅ (nmpCx (C := C) A).homology n :=
  mpComparisonNorm H hAC A n ≪≫ HomologicalComplex.homologyMapIso (normIsoNmp A) n

end NormComparison

end Cover

open scoped MatrixGroups

/-- **The normalised comparison for `SL₂(ℤ)` on `M₇`**, unconditional: `H^n(SL₂(ℤ); A) ≅ H^n(C^•(S(M₇); A))`. -/
def model7_comparisonNorm {k : Type} [CommRing k] (A : Rep.{0} k SL(2, ℤ)) (n : ℕ) :
    groupCohomology A n ≅ (Cover.nmpCx (C := model7Cover) A).homology n :=
  Cover.mpComparisonNmp model7Hyp (model7_AC k) A n

/-- **The normalised comparison for `SL₂(ℤ)` on `R_ℤ`**, unconditional: `H^n(SL₂(ℤ); A) ≅ H^n(C^•(S(R_ℤ); A))`. -/
def rz_comparisonNorm {k : Type} [CommRing k] (A : Rep.{0} k SL(2, ℤ)) (n : ℕ) :
    groupCohomology A n ≅ (Cover.nmpCx (C := rzCover) A).homology n :=
  Cover.mpComparisonNmp rzHyp (rz_AC k) A n

end GL2.MP
