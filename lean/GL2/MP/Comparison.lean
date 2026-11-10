import GL2.MP.Resolution
import GL2.Cone.ProjReduction

/-!
# MP-L2, row L7 (slot 22): the comparison theorem `H^n(C^•(S; A)) ≅ H^n(Γ; A)`

[GT] `gt_part_MP.tex`, rows `def:mp-rep`, `thm:mp-projres`, `thm:mp-homequiv`, `thm:mp-comparison`, backing [C]
`thm:mpcomparison` (1)–(4).

* `mpCx C k : ChainComplex (Rep k G) ℕ`: the permutation representations `k[𝔉'_n]` with the alternating deletion
  differential, `mpAug` its augmentation to the trivial representation `k`;
* `mpAug_quasiIso`: a quasi-isomorphism, from `fCx_acyclic` (L6a) — given (AC) over `k`;
* `mpCx_projective`: every `k[𝔉'_n]` is projective, over **every** commutative ring (`Γ` acts freely on `𝔉'`, so all
  stabilisers are trivial; `GL2.Cone.ProjRed.projective_ofMulAction`);
* `mpRes`: hence a projective resolution of `k`;
* `homEquiv`, `linYonIso`: `Hom_Γ(k[𝔉'_•], A) ≅ C^•(S; A)` **on the nose** as cochain complexes (the restriction to
  normal forms of `GL2.MP.Cover.cochEquiv`; [C] `thm:mpcomparison` (3), unnormalised);
* **`mpComparison`** ([C] `thm:mpcomparison` (4), unnormalised cochains): `H^n(Γ; A) ≅ H^n(C^•(S; A))` for every `n`,
  every commutative ring `k` and every representation `A`, under (AC) over `k`, (IO), finiteness and `dim ≤ 2` — no
  invertibility, effectivity or adaptedness; the central `-I` of `SL₂(ℤ)` is allowed.
-/

noncomputable section

namespace GL2.MP

namespace Cover

open CategoryTheory Finsupp MonoidAlgebra GL2.Cone.ProjRed

variable {G : Type} [Group G] {C : Cover.{0} G} (H : C.Hyp) (k : Type) [CommRing k]

/-! ### The complex of permutation representations -/

/-- The deletion differential `k[𝔉'_{n+1}] → k[𝔉'_n]` on coefficient functions. -/
def dF (n : ℕ) : (C.F' (n + 1) →₀ k) →ₗ[k] (C.F' n →₀ k) := Finsupp.linearCombination k (fBdS (C := C) k)

lemma fBdS_smul {n : ℕ} (g : G) (x : C.F' (n + 1)) :
    fBdS k (g • x) = Finsupp.mapDomain (g • ·) (fBdS k x) := by
  simp only [fBdS, altSum, Finsupp.mapDomain_finsetSum, Finsupp.mapDomain_single, F'.del_smul]

lemma dF_mapDomain (n : ℕ) (g : G) (x : C.F' (n + 1) →₀ k) :
    dF k n (Finsupp.mapDomain (g • ·) x) = Finsupp.mapDomain (g • ·) (dF k n x) := by
  induction x using Finsupp.induction_linear with
  | zero => simp
  | add x y hx hy => rw [Finsupp.mapDomain_add, map_add, hx, hy, map_add, Finsupp.mapDomain_add]
  | single x c =>
    rw [Finsupp.mapDomain_single, dF, Finsupp.linearCombination_single, Finsupp.linearCombination_single,
      fBdS_smul, Finsupp.mapDomain_smul]

variable (C) in
/-- `k[𝔉'_n]`, the permutation representation. -/
abbrev mpX (n : ℕ) : Rep k G := Rep.ofMulAction k G (C.F' n)

variable (C) in
/-- The differential as a morphism of representations. -/
def mpD (n : ℕ) : mpX C k (n + 1) ⟶ mpX C k n :=
  Rep.ofHom ⟨(coeffLinearEquiv k).symm.toLinearMap ∘ₗ dF k n ∘ₗ (coeffLinearEquiv k).toLinearMap, fun g => by
    refine LinearMap.ext fun x => ?_
    simp only [LinearMap.coe_comp, Function.comp_apply, LinearEquiv.coe_coe, Representation.ofMulAction_def,
      LinearEquiv.apply_symm_apply, Finsupp.lmapDomain_apply]
    rw [dF_mapDomain]⟩

lemma mpD_hom_apply (n : ℕ) (x : k[C.F' (n + 1)]) :
    (mpD C k n).hom x = (coeffLinearEquiv k).symm (dF k n (coeffLinearEquiv k x)) := rfl

lemma mpD_mpD (n : ℕ) : mpD C k (n + 1) ≫ mpD C k n = 0 := by
  apply Rep.hom_ext; apply Representation.IntertwiningMap.ext; refine LinearMap.ext fun x => ?_
  show (coeffLinearEquiv k).symm (dF k n (coeffLinearEquiv k ((coeffLinearEquiv k).symm
    (dF k (n + 1) (coeffLinearEquiv k x))))) = 0
  rw [LinearEquiv.apply_symm_apply]
  have : dF k n (dF k (n + 1) (coeffLinearEquiv k x)) = 0 := (fCx C k).d_d (n + 1) _
  rw [this, map_zero]

variable (C) in
/-- **The resolution complex** `k[𝔉'_•]` in `Rep k Γ` ([C] `thm:mpcomparison` (2)). -/
def mpCx : ChainComplex (Rep k G) ℕ := ChainComplex.of (mpX C k) (mpD C k) (mpD_mpD k)

lemma mpCx_d (n : ℕ) : (mpCx C k).d (n + 1) n = mpD C k n := ChainComplex.of_d _ _ n

/-- The augmentation functional `∑ c_x x ↦ ∑ c_x`. -/
def augL : (C.F' 0 →₀ k) →ₗ[k] k := Finsupp.linearCombination k (fun _ => (1 : k))

lemma augL_mapDomain (g : G) (x : C.F' 0 →₀ k) : augL k (Finsupp.mapDomain (g • ·) x) = augL k x := by
  induction x using Finsupp.induction_linear with
  | zero => simp
  | add x y hx hy => rw [Finsupp.mapDomain_add, map_add, hx, hy, map_add]
  | single x c => simp [augL, Finsupp.mapDomain_single]

lemma fCx_d0 (x : C.F' 0 →₀ k) :
    Finsupp.linearCombination k (fBd0 (C := C) k) x = Finsupp.single () (augL k x) := by
  induction x using Finsupp.induction_linear with
  | zero => simp
  | add x y hx hy => rw [map_add, hx, hy, map_add, Finsupp.single_add]
  | single x c => simp [augL, fBd0]

variable {k} in
lemma fAcyc0 (hA : (fCx C k).toCx.Acyclic) (x : C.F' 0 →₀ k) (hx : augL k x = 0) :
    ∃ y : C.F' 1 →₀ k, dF k 0 y = x := by
  obtain ⟨y, hy⟩ := hA.2 0 x (by
    show Finsupp.linearCombination k (fBd0 (C := C) k) x = 0
    rw [fCx_d0, hx, Finsupp.single_zero])
  exact ⟨y, hy⟩

variable {k} in
lemma fAcycS (hA : (fCx C k).toCx.Acyclic) (m : ℕ) (x : C.F' (m + 1) →₀ k) (hx : dF k m x = 0) :
    ∃ y : C.F' (m + 2) →₀ k, dF k (m + 1) y = x := by
  obtain ⟨y, hy⟩ := hA.2 (m + 1) x hx
  exact ⟨y, hy⟩

variable {k} in
lemma fAcycSurj (hA : (fCx C k).toCx.Acyclic) (r : k) : ∃ y : C.F' 0 →₀ k, augL k y = r := by
  obtain ⟨y, hy⟩ := hA.1 (Finsupp.single () r)
  revert y
  intro (y : C.F' 0 →₀ k) (hy' : Finsupp.linearCombination k (fBd0 (C := C) k) y = Finsupp.single () r)
  refine ⟨y, ?_⟩
  rw [fCx_d0] at hy'
  have := congrArg (fun f : Unit →₀ k => f ()) hy'
  simpa using this

variable (C) in
/-- The augmentation `k[𝔉'_0] → k`. -/
def mpEps : mpX C k 0 ⟶ Rep.trivial k G k :=
  Rep.ofHom ⟨augL k ∘ₗ (coeffLinearEquiv k).toLinearMap, fun g => by
    refine LinearMap.ext fun x => ?_
    simp only [LinearMap.coe_comp, Function.comp_apply, LinearEquiv.coe_coe, Representation.ofMulAction_def,
      LinearEquiv.apply_symm_apply, Finsupp.lmapDomain_apply, Representation.trivial_apply]
    rw [augL_mapDomain]⟩

lemma mpEps_hom_apply (x : k[C.F' 0]) : (mpEps C k).hom x = augL k (coeffLinearEquiv k x) := rfl

lemma mpD_mpEps : (mpCx C k).d 1 0 ≫ mpEps C k = 0 := by
  rw [mpCx_d k 0]
  apply Rep.hom_ext; apply Representation.IntertwiningMap.ext; refine LinearMap.ext fun x => ?_
  show augL k (coeffLinearEquiv k ((coeffLinearEquiv k).symm (dF k 0 (coeffLinearEquiv k x)))) = 0
  rw [LinearEquiv.apply_symm_apply]
  have h : Finsupp.linearCombination k (fBd0 (C := C) k) (dF k 0 (coeffLinearEquiv k x)) = 0 :=
    (fCx C k).d_d 0 (coeffLinearEquiv k x)
  rw [fCx_d0] at h
  have := congrArg (fun f : Unit →₀ k => f ()) h
  simpa using this

variable (C) in
/-- **The augmentation** `k[𝔉'_•] → k`. -/
def mpAug : mpCx C k ⟶ (ChainComplex.single₀ (Rep k G)).obj (Rep.trivial k G k) :=
  ((mpCx C k).toSingle₀Equiv _).symm ⟨mpEps C k, mpD_mpEps k⟩

lemma mpAug_f_zero : (mpAug C k).f 0 = mpEps C k := ChainComplex.toSingle₀Equiv_symm_apply_f_zero _ _

/-! ### The quasi-isomorphism -/

variable {k}

lemma exact_zero (hA : (fCx C k).toCx.Acyclic) (w : (mpCx C k).d 1 0 ≫ (mpAug C k).f 0 = 0) :
    (ShortComplex.mk ((mpCx C k).d 1 0) ((mpAug C k).f 0) w).Exact := by
  apply (forget₂ _ (ModuleCat k)).reflects_exact_of_faithful
  rw [ShortComplex.moduleCat_exact_iff]
  intro x hx
  change ((mpAug C k).f 0).hom x = 0 at hx
  rw [mpAug_f_zero] at hx
  have hx' : augL k (coeffLinearEquiv k (x : k[C.F' 0])) = 0 := hx
  obtain ⟨y, hy⟩ := fAcyc0 hA (coeffLinearEquiv k (x : k[C.F' 0])) hx'
  refine ⟨(coeffLinearEquiv k).symm y, ?_⟩
  change ((mpCx C k).d 1 0).hom _ = x
  rw [mpCx_d k 0]
  show (coeffLinearEquiv k).symm (dF k 0 (coeffLinearEquiv k ((coeffLinearEquiv k).symm y))) = x
  rw [LinearEquiv.apply_symm_apply, hy]
  exact (coeffLinearEquiv k).symm_apply_apply (x : k[C.F' 0])

lemma exact_succ (hA : (fCx C k).toCx.Acyclic) (m : ℕ) : ((mpCx C k).sc' (m + 2) (m + 1) m).Exact := by
  apply (forget₂ _ (ModuleCat k)).reflects_exact_of_faithful
  rw [ShortComplex.moduleCat_exact_iff]
  intro x hx
  change ((mpCx C k).d (m + 1) m).hom x = 0 at hx
  rw [mpCx_d k m] at hx
  have hx' : (coeffLinearEquiv k).symm (dF k m (coeffLinearEquiv k (x : k[C.F' (m + 1)]))) = 0 := hx
  rw [LinearEquiv.map_eq_zero_iff] at hx'
  obtain ⟨y, hy⟩ := fAcycS hA m (coeffLinearEquiv k (x : k[C.F' (m + 1)])) hx'
  change ∃ y, ((mpCx C k).d (m + 1 + 1) (m + 1)).hom y = x
  rw [mpCx_d k (m + 1)]
  refine ⟨(coeffLinearEquiv k).symm y, ?_⟩
  show (coeffLinearEquiv k).symm (dF k (m + 1) (coeffLinearEquiv k ((coeffLinearEquiv k).symm y))) = x
  rw [LinearEquiv.apply_symm_apply, hy]
  exact (coeffLinearEquiv k).symm_apply_apply (x : k[C.F' (m + 1)])

lemma mpEps_surjective (hA : (fCx C k).toCx.Acyclic) : Function.Surjective (mpEps C k).hom := by
  intro r
  obtain ⟨y, hy⟩ := fAcycSurj hA r
  refine ⟨(coeffLinearEquiv k).symm y, ?_⟩
  rw [mpEps_hom_apply, LinearEquiv.apply_symm_apply, hy]

/-- **`k[𝔉'_•] → k` is a quasi-isomorphism** in `Rep k Γ` (from L6a). -/
theorem mpAug_quasiIso (hA : (fCx C k).toCx.Acyclic) : QuasiIso (mpAug C k) where
  quasiIsoAt m := by
    cases m with
    | zero =>
      rw [ChainComplex.quasiIsoAt₀_iff, ShortComplex.quasiIso_iff_of_zeros' _ rfl rfl rfl]
      constructor
      · exact exact_zero hA _
      · rw [Rep.epi_iff_surjective]
        change Function.Surjective ((mpAug C k).f 0).hom
        rw [mpAug_f_zero]
        exact mpEps_surjective hA
    | succ m =>
      rw [quasiIsoAt_iff_exactAt' (hL := ChainComplex.exactAt_succ_single_obj ..),
          HomologicalComplex.exactAt_iff' _ (m + 2) (m + 1) m (by simp) (by simp)]
      exact exact_succ hA m

variable (k) in
/-- **Every `k[𝔉'_n]` is projective, over every ring** (`Γ` acts freely on `𝔉'`). -/
theorem mpCx_projective (n : ℕ) : Projective ((mpCx C k).X n) := by
  have hstab : ∀ x : C.F' n, MulAction.stabilizer G x = ⊥ := fun x => by
    ext g; simp only [MulAction.mem_stabilizer_iff, Subgroup.mem_bot]
    exact ⟨fun h => F'.smul_eq_self h, fun h => by rw [h, one_smul]⟩
  have hfin : ∀ x : C.F' n, Finite (MulAction.stabilizer G x) := fun x => by
    rw [hstab x]; infer_instance
  have hunit : ∀ x : C.F' n, IsUnit (Nat.card (MulAction.stabilizer G x) : k) := fun x => by
    rw [hstab x, Subgroup.card_bot, Nat.cast_one]; exact isUnit_one
  exact projective_ofMulAction (k := k) hfin hunit

variable (k) in
/-- **`k[𝔉'_•] → k` is a projective resolution** ([C] `thm:mpcomparison` (2)), under (AC) over `k`. -/
def mpRes (hAC : (yCx H k).toCx.Acyclic) : ProjectiveResolution (Rep.trivial k G k) :=
  haveI := mpAug_quasiIso (fCx_acyclic H k hAC)
  projRes (mpCx C k) (mpAug C k) (mpCx_projective k)

/-! ### `Hom_Γ(k[𝔉'_•], A)` is the MP complex on the nose -/

variable (A : Rep.{0} k G)

omit H in
lemma coeff_single {X : Type} (x : X) (r : k) :
    coeffLinearEquiv k (MonoidAlgebra.single x r) = Finsupp.single x r := by simp

omit H in
lemma lcF_mapDomain {n : ℕ} (F : C.ECoch A n) (g : G) (z : C.F' n →₀ k) :
    Finsupp.linearCombination k F.1 (Finsupp.mapDomain (g • ·) z) = A.ρ g (Finsupp.linearCombination k F.1 z) := by
  induction z using Finsupp.induction_linear with
  | zero => simp
  | add x y hx hy => rw [Finsupp.mapDomain_add, map_add, hx, hy, map_add, map_add]
  | single x c =>
    rw [Finsupp.mapDomain_single, Finsupp.linearCombination_single, Finsupp.linearCombination_single, map_smul, F.2]

omit H in
/-- The lift `k[𝔉'_n] → A` of an equivariant function. -/
def liftE {n : ℕ} (F : C.ECoch A n) : mpX C k n ⟶ A :=
  Rep.ofHom ⟨Finsupp.linearCombination k F.1 ∘ₗ (coeffLinearEquiv k).toLinearMap, fun g => by
    refine LinearMap.ext fun x => ?_
    simp only [LinearMap.coe_comp, Function.comp_apply, LinearEquiv.coe_coe, Representation.ofMulAction_def,
      LinearEquiv.apply_symm_apply, Finsupp.lmapDomain_apply]
    exact lcF_mapDomain A F g _⟩

omit H in
/-- **`Hom_Γ(k[𝔉'_n], A) ≅ {F : 𝔉'_n → A equivariant}`**: `φ ↦ (x ↦ φ(x))`. -/
def homE (n : ℕ) : (mpX C k n ⟶ A) ≃ₗ[k] C.ECoch A n where
  toFun φ := ⟨fun x => φ.hom (MonoidAlgebra.single x 1), fun g x => by
    show φ.hom (MonoidAlgebra.single (g • x) 1) = A.ρ g (φ.hom (MonoidAlgebra.single x 1))
    rw [← Rep.hom_comm_apply]
    congr 1
    exact (Representation.ofMulAction_single g x (1 : k)).symm⟩
  map_add' φ ψ := Subtype.ext (funext fun x => by
    show (φ + ψ).hom (MonoidAlgebra.single x 1) = φ.hom (MonoidAlgebra.single x 1) + ψ.hom (MonoidAlgebra.single x 1)
    rw [Rep.add_hom]; rfl)
  map_smul' r φ := Subtype.ext (funext fun x => by
    show (r • φ).hom (MonoidAlgebra.single x 1) = r • φ.hom (MonoidAlgebra.single x 1)
    rw [Rep.smul_hom]; rfl)
  invFun := liftE A
  left_inv φ := by
    apply Rep.hom_ext; apply Representation.IntertwiningMap.ext; refine LinearMap.ext fun x => ?_
    show Finsupp.linearCombination k (fun y => φ.hom (MonoidAlgebra.single y 1)) (coeffLinearEquiv k x) = φ.hom x
    induction x using MonoidAlgebra.induction_linear with
    | zero => simp
    | add x y hx hy => rw [map_add, map_add, hx, hy, map_add]
    | single m r =>
      rw [coeff_single, Finsupp.linearCombination_single, ← map_smul]
      congr 1
      simp
  right_inv F := Subtype.ext (funext fun x => by
    show Finsupp.linearCombination k F.1 (coeffLinearEquiv k (MonoidAlgebra.single x 1)) = F.1 x
    simp)

omit H in
lemma homE_apply {n : ℕ} (φ : mpX C k n ⟶ A) (x : C.F' n) : (homE A n φ).1 x = φ.hom (MonoidAlgebra.single x 1) := rfl

omit H in
/-- **Precomposition with the differential is `dE`.** -/
lemma homE_d {n : ℕ} (φ : mpX C k n ⟶ A) :
    homE A (n + 1) (mpD C k n ≫ φ) = ⟨C.dE A n (homE A n φ).1, C.dE_mem A n (homE A n φ).2⟩ := by
  apply Subtype.ext; funext x
  show φ.hom ((mpD C k n).hom (MonoidAlgebra.single x 1)) = ∑ i : Fin (n + 2), ((-1 : k) ^ (i : ℕ)) •
    φ.hom (MonoidAlgebra.single (F'.del i x) 1)
  rw [mpD_hom_apply, coeff_single, dF,
    Finsupp.linearCombination_single, one_smul, fBdS, altSum, map_sum, map_sum]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [← map_smul]
  congr 1
  simp

omit H in
/-- **`Hom_Γ(k[𝔉'_n], A) = C^n(S; A)`** ([C] `thm:mpcomparison` (3)). -/
def homEquiv (n : ℕ) : (mpX C k n ⟶ A) ≃ₗ[k] C.Coch A n := (homE A n).trans (C.cochEquiv A n)

omit H in
lemma homEquiv_d {n : ℕ} (φ : mpX C k n ⟶ A) :
    homEquiv A (n + 1) (mpD C k n ≫ φ) = C.dC A n (homEquiv A n φ) := by
  rw [homEquiv, LinearEquiv.trans_apply, homE_d, cochEquiv_dE]; rfl

omit H in
/-- **The cochain complexes agree**: `Hom_Γ(k[𝔉'_•], A) ≅ C^•(S; A)` (unnormalised). -/
def linYonIso : (mpCx C k).linearYonedaObj k A ≅ C.mpComplex A :=
  HomologicalComplex.Hom.isoOfComponents (fun n => (homEquiv A n).toModuleIso) (by
    rintro i j (rfl : i + 1 = j)
    simp only [ChainComplex.linearYonedaObj_d, mpComplex, CochainComplex.of_d]
    apply ModuleCat.hom_ext; refine LinearMap.ext fun φ => ?_
    show C.dC A i (homEquiv A i φ) = homEquiv A (i + 1) ((mpCx C k).d (i + 1) i ≫ φ)
    rw [mpCx_d]
    exact (homEquiv_d A φ).symm)

/-- **[C] `thm:mpcomparison` (4), the comparison theorem** (unnormalised MP cochains): for every commutative ring `k`,
every representation `A` of `Γ` over `k` and every `n`, `H^n(Γ; A) ≅ H^n(C^•(S; A))`, given (AC) over `k`, (IO), finite
vertex sets and `dim ≤ 2`.  No order is inverted, `Γ` need not act effectively, the triangulation need not be adapted. -/
def mpComparison (hAC : (yCx H k).toCx.Acyclic) (n : ℕ) : groupCohomology A n ≅ (C.mpComplex A).homology n :=
  groupCohomologyIso A n (mpRes H k hAC) ≪≫
    (HomologicalComplex.homologyFunctor (ModuleCat k) (ComplexShape.up ℕ) n).mapIso (linYonIso A)

end Cover

end GL2.MP
