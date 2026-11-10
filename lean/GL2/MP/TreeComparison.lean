/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.MP.TreeRes

/-!
# LR-Y phase 2e (slot 22): the tree resolution as a projective resolution, and the comparison with MP

[GT] `gt_part_MP.tex`, rows `thm:mp-perm-res`, `thm:mp-tree-mp-comparison`, backing [C] `cl:amalgam`, `eq:mvamalgam`,
`thm:mpcomparison` (4)–(5) (a second resolution giving the same groups; plan of record `LV0_audit.md` §3.2, LR-2 (d)).

* `PermRes`: a complex of permutation modules on bases with free `G`-actions, equivariant boundaries and the exactness of
  `⋯ → k[B₁] → k[B₀] → k → 0`; `PermRes.res` is then a **projective resolution** of the trivial representation `k`
  (the `mpRes` route of `GL2.MP.Comparison`, for a general basis family).
* `model7PermRes k`: the cone resolution `k[Γ]² ← k[Γ]³ ← k[Γ]³ ← ⋯` of `GL2.MP.TreeRes` (`Γ = SL₂(ℤ)`) as a `PermRes`
  (`φ` and the periodic differentials are right multiplications, hence equivariant); `model7TreeRes k` the projective
  resolution.
* **`model7TreeCoh`**: `H^n(Γ; A) ≅ H^n(Hom_Γ(P_•, A))` for every `n`, every commutative ring `k` and every `A : Rep k Γ`,
  `−I` included; **`model7TreeMP`**: hence `H^n(Hom_Γ(P_•, A)) ≅ H^n(C^•(S(M₇); A))` (the MP complex).
* **`model7TreeMPEquiv`**: the lift `P_• → k[𝔉'_•]` of `id_k` to the MP resolution `mpRes` and back, a homotopy
  equivalence of complexes of `Γ`-representations (`ProjectiveResolution.homotopyEquiv`).
-/

noncomputable section

namespace GL2.MP

open CategoryTheory Finsupp MonoidAlgebra GL2.Cone.ProjRed

/-! ### Resolutions by permutation modules given on bases -/

/-- **A resolution of `k` by free permutation modules, on bases**: `k[B_n]` with free `G`-actions, equivariant boundaries,
`d ∘ d = 0`, and `⋯ → k[B₁] → k[B₀] →(Σ) k → 0` exact. -/
structure PermRes (G : Type) [Group G] (k : Type) [CommRing k] where
  B : ℕ → Type
  [act : ∀ n, MulAction G (B n)]
  bd : ∀ n, B (n + 1) → (B n →₀ k)
  bd_smul : ∀ n (g : G) (b : B (n + 1)), bd n (g • b) = mapDomain (g • ·) (bd n b)
  bdd : ∀ n (b : B (n + 2)), Finsupp.linearCombination k (bd n) (bd (n + 1) b) = 0
  free : ∀ n (g : G) (b : B n), g • b = b → g = 1
  aug_bd : ∀ b : B 1, Finsupp.linearCombination k (fun _ : B 0 => (1 : k)) (bd 0 b) = 0
  surj : ∀ r : k, ∃ x : B 0 →₀ k, Finsupp.linearCombination k (fun _ => (1 : k)) x = r
  ex0 : ∀ x : B 0 →₀ k, Finsupp.linearCombination k (fun _ => (1 : k)) x = 0 →
    ∃ y, Finsupp.linearCombination k (bd 0) y = x
  exS : ∀ n (x : B (n + 1) →₀ k), Finsupp.linearCombination k (bd n) x = 0 →
    ∃ y, Finsupp.linearCombination k (bd (n + 1)) y = x

attribute [instance] PermRes.act

namespace PermRes

variable {G : Type} [Group G] {k : Type} [CommRing k] (P : PermRes G k)

/-- The differential on coefficient functions. -/
def dP (n : ℕ) : (P.B (n + 1) →₀ k) →ₗ[k] (P.B n →₀ k) := Finsupp.linearCombination k (P.bd n)

lemma dP_mapDomain (n : ℕ) (g : G) (x : P.B (n + 1) →₀ k) :
    P.dP n (mapDomain (g • ·) x) = mapDomain (g • ·) (P.dP n x) :=
  Cover.lc_mapDomain_smul k (P.bd n) g (P.bd_smul n g) x

lemma dP_dP (n : ℕ) (x : P.B (n + 2) →₀ k) : P.dP n (P.dP (n + 1) x) = 0 := by
  induction x using Finsupp.induction_linear with
  | zero => simp
  | add x y hx hy => rw [map_add, map_add, hx, hy, add_zero]
  | single b c => rw [dP, dP, linearCombination_single, map_smul, P.bdd, smul_zero]

/-- The augmentation functional `∑ c_x x ↦ ∑ c_x`. -/
def augL : (P.B 0 →₀ k) →ₗ[k] k := Finsupp.linearCombination k (fun _ => (1 : k))

lemma augL_mapDomain (g : G) (x : P.B 0 →₀ k) : P.augL (mapDomain (g • ·) x) = P.augL x := by
  induction x using Finsupp.induction_linear with
  | zero => simp
  | add x y hx hy => rw [Finsupp.mapDomain_add, map_add, hx, hy, map_add]
  | single x c => simp [augL, Finsupp.mapDomain_single]

lemma augL_dP (x : P.B 1 →₀ k) : P.augL (P.dP 0 x) = 0 := by
  induction x using Finsupp.induction_linear with
  | zero => simp
  | add x y hx hy => rw [map_add, map_add, hx, hy, add_zero]
  | single b c => rw [dP, linearCombination_single, map_smul, augL, P.aug_bd, smul_zero]

variable (k) in
/-- `k[B_n]`, the permutation representation. -/
abbrev X (n : ℕ) : Rep k G := Rep.ofMulAction k G (P.B n)

/-- The differential as a morphism of representations. -/
def D (n : ℕ) : P.X k (n + 1) ⟶ P.X k n :=
  Rep.ofHom ⟨(coeffLinearEquiv k).symm.toLinearMap ∘ₗ P.dP n ∘ₗ (coeffLinearEquiv k).toLinearMap, fun g => by
    refine LinearMap.ext fun x => ?_
    simp only [LinearMap.coe_comp, Function.comp_apply, LinearEquiv.coe_coe, Representation.ofMulAction_def,
      LinearEquiv.apply_symm_apply, Finsupp.lmapDomain_apply]
    rw [dP_mapDomain]⟩

lemma D_hom_apply (n : ℕ) (x : k[P.B (n + 1)]) :
    (P.D n).hom x = (coeffLinearEquiv k).symm (P.dP n (coeffLinearEquiv k x)) := rfl

lemma D_D (n : ℕ) : P.D (n + 1) ≫ P.D n = 0 := by
  apply Rep.hom_ext; apply Representation.IntertwiningMap.ext; refine LinearMap.ext fun x => ?_
  show (coeffLinearEquiv k).symm (P.dP n (coeffLinearEquiv k ((coeffLinearEquiv k).symm
    (P.dP (n + 1) (coeffLinearEquiv k x))))) = 0
  rw [LinearEquiv.apply_symm_apply, P.dP_dP, map_zero]

variable (k) in
/-- **The resolving complex** `k[B_•]` in `Rep k G`. -/
def cx : ChainComplex (Rep k G) ℕ := ChainComplex.of (P.X k) P.D P.D_D

lemma cx_d (n : ℕ) : (P.cx k).d (n + 1) n = P.D n := ChainComplex.of_d _ _ n

/-- The augmentation `k[B₀] → k`. -/
def eps : P.X k 0 ⟶ Rep.trivial k G k :=
  Rep.ofHom ⟨P.augL ∘ₗ (coeffLinearEquiv k).toLinearMap, fun g => by
    refine LinearMap.ext fun x => ?_
    simp only [LinearMap.coe_comp, Function.comp_apply, LinearEquiv.coe_coe, Representation.ofMulAction_def,
      LinearEquiv.apply_symm_apply, Finsupp.lmapDomain_apply, Representation.trivial_apply]
    rw [augL_mapDomain]⟩

lemma eps_hom_apply (x : k[P.B 0]) : P.eps.hom x = P.augL (coeffLinearEquiv k x) := rfl

lemma D_eps : (P.cx k).d 1 0 ≫ P.eps = 0 := by
  rw [cx_d]
  apply Rep.hom_ext; apply Representation.IntertwiningMap.ext; refine LinearMap.ext fun x => ?_
  show P.augL (coeffLinearEquiv k ((coeffLinearEquiv k).symm (P.dP 0 (coeffLinearEquiv k x)))) = 0
  rw [LinearEquiv.apply_symm_apply, augL_dP]

/-- **The augmentation** `k[B_•] → k`. -/
def aug : P.cx k ⟶ (ChainComplex.single₀ (Rep k G)).obj (Rep.trivial k G k) :=
  ((P.cx k).toSingle₀Equiv _).symm ⟨P.eps, P.D_eps⟩

lemma aug_f_zero : P.aug.f 0 = P.eps := ChainComplex.toSingle₀Equiv_symm_apply_f_zero _ _

lemma exact_zero (w : (P.cx k).d 1 0 ≫ P.aug.f 0 = 0) :
    (ShortComplex.mk ((P.cx k).d 1 0) (P.aug.f 0) w).Exact := by
  apply (forget₂ _ (ModuleCat k)).reflects_exact_of_faithful
  rw [ShortComplex.moduleCat_exact_iff]
  intro x hx
  change (P.aug.f 0).hom x = 0 at hx
  rw [aug_f_zero] at hx
  have hx' : P.augL (coeffLinearEquiv k (x : k[P.B 0])) = 0 := hx
  obtain ⟨y, hy⟩ := P.ex0 _ hx'
  refine ⟨(coeffLinearEquiv k).symm y, ?_⟩
  change ((P.cx k).d 1 0).hom _ = x
  rw [cx_d]
  show (coeffLinearEquiv k).symm (P.dP 0 (coeffLinearEquiv k ((coeffLinearEquiv k).symm y))) = x
  rw [LinearEquiv.apply_symm_apply]
  show (coeffLinearEquiv k).symm (Finsupp.linearCombination k (P.bd 0) y) = x
  rw [hy]
  exact (coeffLinearEquiv k).symm_apply_apply (x : k[P.B 0])

lemma exact_succ (m : ℕ) : ((P.cx k).sc' (m + 2) (m + 1) m).Exact := by
  apply (forget₂ _ (ModuleCat k)).reflects_exact_of_faithful
  rw [ShortComplex.moduleCat_exact_iff]
  intro x hx
  change ((P.cx k).d (m + 1) m).hom x = 0 at hx
  rw [cx_d] at hx
  have hx' : (coeffLinearEquiv k).symm (P.dP m (coeffLinearEquiv k (x : k[P.B (m + 1)]))) = 0 := hx
  rw [LinearEquiv.map_eq_zero_iff] at hx'
  obtain ⟨y, hy⟩ := P.exS m _ hx'
  change ∃ y, ((P.cx k).d (m + 1 + 1) (m + 1)).hom y = x
  rw [cx_d]
  refine ⟨(coeffLinearEquiv k).symm y, ?_⟩
  show (coeffLinearEquiv k).symm (P.dP (m + 1) (coeffLinearEquiv k ((coeffLinearEquiv k).symm y))) = x
  rw [LinearEquiv.apply_symm_apply]
  show (coeffLinearEquiv k).symm (Finsupp.linearCombination k (P.bd (m + 1)) y) = x
  rw [hy]
  exact (coeffLinearEquiv k).symm_apply_apply (x : k[P.B (m + 1)])

lemma eps_surjective : Function.Surjective P.eps.hom := by
  intro r
  obtain ⟨y, hy⟩ := P.surj r
  refine ⟨(coeffLinearEquiv k).symm y, ?_⟩
  rw [eps_hom_apply, LinearEquiv.apply_symm_apply]
  exact hy

/-- **`k[B_•] → k` is a quasi-isomorphism.** -/
theorem aug_quasiIso : QuasiIso P.aug where
  quasiIsoAt m := by
    cases m with
    | zero =>
      rw [ChainComplex.quasiIsoAt₀_iff, ShortComplex.quasiIso_iff_of_zeros' _ rfl rfl rfl]
      constructor
      · exact P.exact_zero _
      · rw [Rep.epi_iff_surjective]
        change Function.Surjective (P.aug.f 0).hom
        rw [aug_f_zero]
        exact P.eps_surjective
    | succ m =>
      rw [quasiIsoAt_iff_exactAt' (hL := ChainComplex.exactAt_succ_single_obj ..),
          HomologicalComplex.exactAt_iff' _ (m + 2) (m + 1) m (by simp) (by simp)]
      exact P.exact_succ m

/-- **Every `k[B_n]` is projective**, over every ring (`G` acts freely). -/
theorem projective (n : ℕ) : Projective ((P.cx k).X n) := by
  have hstab : ∀ x : P.B n, MulAction.stabilizer G x = ⊥ := fun x => by
    ext g; simp only [MulAction.mem_stabilizer_iff, Subgroup.mem_bot]
    exact ⟨fun h => P.free n g x h, fun h => by rw [h, one_smul]⟩
  have hfin : ∀ x : P.B n, Finite (MulAction.stabilizer G x) := fun x => by
    rw [hstab x]; infer_instance
  have hunit : ∀ x : P.B n, IsUnit (Nat.card (MulAction.stabilizer G x) : k) := fun x => by
    rw [hstab x, Subgroup.card_bot, Nat.cast_one]; exact isUnit_one
  exact projective_ofMulAction (k := k) hfin hunit

/-- **A projective resolution of `k`.** -/
def res : ProjectiveResolution (Rep.trivial k G k) :=
  haveI := P.aug_quasiIso
  projRes (P.cx k) P.aug P.projective

/-- **`H^n(G; A) ≅ H^n(Hom_G(k[B_•], A))`.** -/
def cohIso (A : Rep.{0} k G) (n : ℕ) : groupCohomology A n ≅ ((P.cx k).linearYonedaObj k A).homology n :=
  groupCohomologyIso A n P.res

end PermRes

/-! ### The tree resolution of `M₇` -/

open Cover GL2.TauChk
open scoped MatrixGroups

/-- The bases of the tree resolution: `FB 2 Γ` (`k[Γ]²`) in degree `0`, `FB 2 Γ ⊕ FB 1 Γ` (`k[Γ]³`) above. -/
def model7PB : ℕ → Type
  | 0 => FB 2 SL(2, ℤ)
  | _ + 1 => FB 2 SL(2, ℤ) ⊕ FB 1 SL(2, ℤ)

instance model7PB.act (n : ℕ) : MulAction SL(2, ℤ) (model7PB n) :=
  match n with
  | 0 => (inferInstance : MulAction SL(2, ℤ) (FB 2 SL(2, ℤ)))
  | _ + 1 => (inferInstance : MulAction SL(2, ℤ) (FB 2 SL(2, ℤ) ⊕ FB 1 SL(2, ℤ)))

variable (k : Type) [CommRing k]

/-- The boundary of the tree resolution (the cone boundary of `model7Cone` in degrees `≥ 2`, written on `FB`):
`(b, a) ↦ (1 − t) b + φ a`, then `(b, a) ↦ (d b + φ a, −d a)`. -/
noncomputable def model7PBd : ∀ n, model7PB (n + 1) → (model7PB n →₀ k)
  | 0 => Sum.elim (model7PerV.bdS k 0) (model7PhiB k 0)
  | n + 1 => Sum.elim (fun b => Finsupp.mapDomain Sum.inl (model7PerV.bdS k (n + 1) b))
      (fun a => Finsupp.mapDomain Sum.inl (model7PhiB k (n + 1) a) - Finsupp.mapDomain Sum.inr (model7PerE.bdS k n a))

/-- **`φ` is equivariant** (right multiplication). -/
lemma model7PhiB_smul (n : ℕ) (g : SL(2, ℤ)) (b : FB 1 SL(2, ℤ)) :
    model7PhiB k n (g • b) = mapDomain (fun y : FB 2 SL(2, ℤ) => g • y) (model7PhiB k n b) := by
  unfold model7PhiB
  split_ifs
  · simp only [Finsupp.mapDomain_sub, Finsupp.mapDomain_single, FB.smul_g]; rfl
  · simp only [Finsupp.mapDomain_sub, Finsupp.mapDomain_add, Finsupp.mapDomain_single, FB.smul_g, mul_assoc]; rfl

lemma mapDomain_inl_smul {α β : Type} [MulAction SL(2, ℤ) α] [MulAction SL(2, ℤ) β] (g : SL(2, ℤ)) (x : α →₀ k) :
    mapDomain (Sum.inl : α → α ⊕ β) (mapDomain (fun y => g • y) x) =
      mapDomain (fun y : α ⊕ β => g • y) (mapDomain Sum.inl x) := by
  rw [← Finsupp.mapDomain_comp, ← Finsupp.mapDomain_comp]; rfl

lemma mapDomain_inr_smul {α β : Type} [MulAction SL(2, ℤ) α] [MulAction SL(2, ℤ) β] (g : SL(2, ℤ)) (x : β →₀ k) :
    mapDomain (Sum.inr : β → α ⊕ β) (mapDomain (fun y => g • y) x) =
      mapDomain (fun y : α ⊕ β => g • y) (mapDomain Sum.inr x) := by
  rw [← Finsupp.mapDomain_comp, ← Finsupp.mapDomain_comp]; rfl

lemma model7PBd_smul : ∀ n (g : SL(2, ℤ)) (b : model7PB (n + 1)),
    model7PBd k n (g • b) = Finsupp.mapDomain (g • ·) (model7PBd k n b)
  | 0, g, Sum.inl b => model7PerV.bdS_smul k 0 g b
  | 0, g, Sum.inr a => model7PhiB_smul k 0 g a
  | n + 1, g, Sum.inl b => by
    show Finsupp.mapDomain Sum.inl (model7PerV.bdS k (n + 1) (g • b)) =
      Finsupp.mapDomain (fun y : FB 2 SL(2, ℤ) ⊕ FB 1 SL(2, ℤ) => g • y)
        (Finsupp.mapDomain Sum.inl (model7PerV.bdS k (n + 1) b))
    rw [model7PerV.bdS_smul k (n + 1) g b, mapDomain_inl_smul]
  | n + 1, g, Sum.inr a => by
    show Finsupp.mapDomain Sum.inl (model7PhiB k (n + 1) (g • a)) - Finsupp.mapDomain Sum.inr (model7PerE.bdS k n (g • a)) =
      Finsupp.mapDomain (fun y : FB 2 SL(2, ℤ) ⊕ FB 1 SL(2, ℤ) => g • y)
        (Finsupp.mapDomain Sum.inl (model7PhiB k (n + 1) a) - Finsupp.mapDomain Sum.inr (model7PerE.bdS k n a))
    rw [model7PhiB_smul, model7PerE.bdS_smul, mapDomain_inl_smul, mapDomain_inr_smul, ← Finsupp.mapDomain_sub]

lemma model7PB_free : ∀ n (g : SL(2, ℤ)) (b : model7PB n), g • b = b → g = 1
  | 0, _, _, h => FB.smul_eq_self h
  | _ + 1, _, Sum.inl _, h => FB.smul_eq_self (Sum.inl_injective h)
  | _ + 1, _, Sum.inr _, h => FB.smul_eq_self (Sum.inr_injective h)

/-- The cone augmentation is `(c, g) ↦ 1`. -/
lemma model7_bd0 (b : FB 2 SL(2, ℤ)) : (model7Cone k).bd0 b = Finsupp.single () 1 := by
  show Finsupp.linearCombination k (tBd0 model7Hyp 3 k) (model7PerV.augB k b) = _
  rw [PerData.augB, linearCombination_single, one_smul]; rfl

lemma model7_bd0_eq : (model7Cone k).bd0 = fun _ => Finsupp.single () 1 := funext (model7_bd0 k)

lemma lc_const_single {α : Type} (x : α →₀ k) :
    Finsupp.linearCombination k (fun _ : α => Finsupp.single () (1 : k)) x =
      Finsupp.single () (Finsupp.linearCombination k (fun _ => (1 : k)) x) := by
  induction x using Finsupp.induction_linear with
  | zero => simp
  | add x y hx hy => rw [map_add, hx, hy, map_add, Finsupp.single_add]
  | single b c => simp

/-- **The tree resolution of `M₇` as a resolution by free permutation modules** (from `model7ConeRes_acyclic`). -/
def model7PermRes : PermRes SL(2, ℤ) k where
  B := model7PB
  bd := model7PBd k
  bd_smul := model7PBd_smul k
  bdd n b := by
    rcases n with _ | n
    · exact (model7Cone k).bdd 1 b
    · exact (model7Cone k).bdd (n + 2) b
  free := model7PB_free
  aug_bd b := by
    have h := (model7Cone k).bdd 0 b
    change Finsupp.linearCombination k (model7Cone k).bd0 ((model7Cone k).bd1 b) = 0 at h
    rw [model7_bd0_eq] at h
    exact Finsupp.single_eq_zero.1 ((lc_const_single (α := FB 2 SL(2, ℤ)) k _).symm.trans h)
  surj r := by
    obtain ⟨(y : FB 2 SL(2, ℤ) →₀ k), hy⟩ := (model7ConeRes_acyclic k).1 (Finsupp.single () r)
    refine ⟨y, ?_⟩
    change Finsupp.linearCombination k (model7Cone k).bd0 y = Finsupp.single () r at hy
    rw [model7_bd0_eq] at hy
    exact Finsupp.single_injective () ((lc_const_single (α := FB 2 SL(2, ℤ)) k y).symm.trans hy)
  ex0 x hx := by
    have h0 : Finsupp.linearCombination k (fun _ : model7PB 0 => Finsupp.single () (1 : k)) x = 0 := by
      rw [lc_const_single, hx, Finsupp.single_zero]
    obtain ⟨y, hy⟩ := (model7ConeRes_acyclic k).2 0 x (by
      show Finsupp.linearCombination k (model7Cone k).bd0 x = 0
      rw [model7_bd0_eq]
      exact h0)
    exact ⟨y, hy⟩
  exS n x hx := by
    rcases n with _ | n
    · obtain ⟨y, hy⟩ := (model7ConeRes_acyclic k).2 1 x hx
      exact ⟨y, hy⟩
    · obtain ⟨y, hy⟩ := (model7ConeRes_acyclic k).2 (n + 2) x hx
      exact ⟨y, hy⟩

/-- **The tree resolution** `k[Γ]² ← k[Γ]³ ← k[Γ]³ ← ⋯ → k`, a projective resolution of the trivial representation, for
every commutative ring `k`. -/
def model7TreeRes : ProjectiveResolution (Rep.trivial k SL(2, ℤ) k) := (model7PermRes k).res

variable {k}

/-- **`H^n(SL₂(ℤ); A) ≅ H^n(Hom_Γ(P_•, A))`** for the tree resolution, every `n`, every `A`, `−I` included. -/
def model7TreeCoh (A : Rep.{0} k SL(2, ℤ)) (n : ℕ) :
    groupCohomology A n ≅ (((model7PermRes k).cx k).linearYonedaObj k A).homology n :=
  (model7PermRes k).cohIso A n

/-- **The tree cohomology is the MP cohomology**: `H^n(Hom_Γ(P_•, A)) ≅ H^n(C^•(S(M₇); A))`, every `n`, `−I` included. -/
def model7TreeMP (A : Rep.{0} k SL(2, ℤ)) (n : ℕ) :
    (((model7PermRes k).cx k).linearYonedaObj k A).homology n ≅ (model7Cover.mpComplex A).homology n :=
  (model7TreeCoh A n).symm ≪≫ model7_comparison_unc A n

variable (k)

/-- **The comparison with the MP resolution**: the lift of `id_k` to `P_• → k[𝔉'_•]` (and back) is a homotopy
equivalence of complexes of representations. -/
def model7TreeMPEquiv :
    HomotopyEquiv (model7TreeRes k).complex (Cover.mpRes model7Hyp k (model7_AC k)).complex :=
  ProjectiveResolution.homotopyEquiv _ _

end GL2.MP
