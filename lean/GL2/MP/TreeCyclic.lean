/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.MP.TreeCochains
import GL2.MP.Natural

/-!
# LR-Y phase 4 (slot 22): the Mayer–Vietoris terms as cohomology of the cyclic subgroups, the middle map as `res − res`

[GT] `gt_part_MP.tex`, rows `thm:mp-cyc-coh`, `thm:mp-cyc-res`, `thm:mp-mv-terms`, `thm:mp-mv-resres`, `thm:mp-mv-low`,
`thm:mp-periodic-natural`, backing [C] `eq:mvamalgam` (the terms `H^q(⟨S⟩; A) ⊕ H^q(⟨U⟩; A)` and `H^q(⟨−I⟩; A)`, the
middle map `res − res`, the beginning `0 → H⁰(Γ; A) → A^S ⊕ A^U → A^{−I}`) and `lem:bulkperiodic` (2) (a natural
isomorphism `H^q(Γ; -) ≅ H^{q+2}(Γ; -)`, `q ≥ 2`, built through the bar comparison; not shown to coincide with phase 3's
`model7CohPeriodic`, which goes through `Ext`; still not identified with `e ∪ −`, not its compatibility with restriction).
Every commutative ring `k`, every `A : Rep k SL₂(ℤ)`.

* **Cyclic groups.**  `CycDat C`: a generator `t` of order `N`; `CycDat.permRes`: the periodic resolution
  `k[C] ←(1−t) k[C] ←(N) ⋯` as a `PermRes`; `CycDat.cycCoch A`: `A → A → ⋯` (`1 − t`, `N_t = ∑_{j<N} t^j` alternating).
* **The fixed comparison with the bar resolution.**  `PermRes.barIso`: `H^q(G; A) ≅ H^q(Hom_G(P_•, A))` through
  `ProjectiveResolution.homotopyEquiv P bar` (as in `GL2.MP.Natural`, not through `Ext`); **`CycDat.cohIso`**:
  `H^q(C; A) ≅ H^q(cycCoch A)`.
* **Restriction.**  `barMapCx`: the bar resolutions under `f : G' →* G`; `cochainsMap_barMap`: mathlib's `cochainsMap f`
  is precomposition with it; **`PermRes.barIso_res`**: for injective `f` and any chain map `α : P' → P|_{G'}` lifting
  `id_k`, `res_f = groupCohomology.map f` is induced by `φ ↦ φ ∘ α` (homotopy uniqueness, `resHomotopy`);
  **`CycDat.cohIso_res`**: between cyclic groups (`f(t') = t^d`, `N = dN'`) restriction is `x ↦ x` in even and
  `x ↦ ∑_{j<d} t^j x` in odd degrees (`cycRes_f_apply`).
* **`SL₂(ℤ)`.**  `gS, gU, gI = ⟨S⟩, ⟨U⟩, ⟨−I⟩`; `resSI`, `resUI`: mathlib's restriction to `⟨−I⟩`.
  **`vIso`**: `H^q(vCoch A) ≅ H^q(⟨S⟩; A) ⊞ H^q(⟨U⟩; A)`; **`eIso`**: `H^{q+1}(eSh A) ≅ H^q(⟨−I⟩; A)` (the identity on
  cocycles, `eHomIso_π`); **`mvδ_resres`**: `vIso⁻¹ ≫ mvδ ≫ eIso = biprod.desc res_S (−res_U)`, i.e. the middle map of
  `eq:mvamalgam` is `res_S − res_U`, sign as written.
* **The beginning.**  `invariants_eq_inf`: `A^Γ = A^S ∩ A^U`; **`mv_low_exact`**: `0 → A^Γ → A^S ⊕ A^U → A^{−I}`
  (`a ↦ (a, a)`, `(x, y) ↦ x − y`) is exact, proved directly (not as the degree-`0` case of the sequence: `mvRes 0` is
  not identified with `lowι`); `resSI_zero`, `resUI_zero`: restriction on `H⁰` is the inclusion of invariants (mathlib's
  `H0Iso`).  `invS`, `invU`, `invI`: the invariants as `Submodule k A` (LRY4_audit §5, the replay peak).
* **Stretch.**  `model7TreeCochCohNat` (`H^n(Γ; A) ≅ H^n(treeCoch A)` through the bar comparison) is natural in `A`
  (`model7TreeCochCohNat_natural`); **`model7CohPeriodicNatIso`**: a natural isomorphism `H^{m+2}(Γ; -) ≅ H^{m+4}(Γ; -)`
  of functors (the same `treePerNat`, conjugated by the bar-built comparison; not shown equal to `model7CohPeriodic`).
-/

noncomputable section

namespace GL2.MP

open CategoryTheory Finsupp

/-! ### Finite cyclic groups: the periodic resolution as a `PermRes`, and its cochain complex -/

/-- **A generator `t` of order `N`** of a group `C`. -/
structure CycDat (C : Type) [Group C] where
  t : C
  N : ℕ
  N_pos : 0 < N
  pow_N : t ^ N = 1
  pow_inj : ∀ a b : Fin N, t ^ (a : ℕ) = t ^ (b : ℕ) → a = b
  gen : ∀ g : C, ∃ j : Fin N, g = t ^ (j : ℕ)

namespace CycDat

variable {C : Type} [Group C] (Z : CycDat C)

/-- The periodic data of `C` acting on a point. -/
def perD : PerData C PUnit 1 where
  x _ := PUnit.unit
  t _ := Z.t
  N _ := Z.N
  N_pos _ := Z.N_pos
  pow_N _ := Z.pow_N
  pow_inj _ := Z.pow_inj
  fix _ := rfl
  cov _ := ⟨0, 1, rfl⟩
  sep _ _ g g' _ := ⟨Subsingleton.elim _ _, by
    obtain ⟨j, hj⟩ := Z.gen (g'⁻¹ * g)
    exact ⟨j, by rw [← hj, mul_inv_cancel_left]⟩⟩

variable (k : Type) [CommRing k]

lemma lc_augB (x : FB 1 C →₀ k) :
    Finsupp.linearCombination k (Z.perD.augB k) x =
      Finsupp.single PUnit.unit (Finsupp.linearCombination k (fun _ => (1 : k)) x) := by
  induction x using Finsupp.induction_linear with
  | zero => simp
  | add x y hx hy => rw [map_add, hx, hy, map_add, Finsupp.single_add]
  | single b c => simp [PerData.augB]

/-- **The periodic resolution of `k` over `k[C]`**, `k[C] ←(1−t) k[C] ←(N) k[C] ← ⋯`, as a `PermRes`. -/
def permRes : PermRes C k where
  B _ := FB 1 C
  bd n := Z.perD.bdS k n
  bd_smul n g b := Z.perD.bdS_smul k n g b
  bdd n b := Z.perD.bdd_succ k n b
  free _ _ _ h := FB.smul_eq_self h
  aug_bd b := by
    show Finsupp.linearCombination k (fun _ => (1 : k)) (Z.perD.bdS k 0 b) = 0
    rw [Z.perD.bdS_even k PerData.even0, PerData.difB, map_sub, linearCombination_single,
      linearCombination_single, sub_self]
  surj r := ⟨Finsupp.single ⟨0, 1⟩ r, by simp⟩
  ex0 x hx := by
    obtain ⟨y, hy⟩ := (Z.perD.perCx_acyclic k).2 0 x (by
      show Finsupp.linearCombination k (Z.perD.augB k) x = 0
      rw [lc_augB, hx, Finsupp.single_zero])
    exact ⟨y, hy⟩
  exS n x hx := (Z.perD.perCx_acyclic k).2 (n + 1) x hx

variable {k}

/-- The differential of the periodic cochain complex in degree `n`: `1 − t` for even `n`, `N_t` for odd `n`. -/
def cycD (A : Rep.{0} k C) (n : ℕ) : A →ₗ[k] A :=
  if Even n then LinearMap.id - A.ρ Z.t else ∑ j ∈ Finset.range Z.N, A.ρ (Z.t ^ j)

/-- `Hom_C(k[C], A) ≅ A`, `φ ↦ φ(e)`, in every degree. -/
def cycEv (A : Rep.{0} k C) (n : ℕ) : (((Z.permRes k).X k n) ⟶ A) ≃ₗ[k] A :=
  (homFB A 1).trans (LinearEquiv.funUnique (Fin 1) k A)

lemma cycEv_apply (A : Rep.{0} k C) (n : ℕ) (φ : ((Z.permRes k).X k n) ⟶ A) :
    Z.cycEv A n φ = (permHom A (FB 1 C) φ).1 ⟨0, 1⟩ := rfl

lemma cycEv_d (A : Rep.{0} k C) (n : ℕ) (φ : ((Z.permRes k).X k n) ⟶ A) :
    Z.cycEv A (n + 1) ((Z.permRes k).D n ≫ φ) = Z.cycD A n (Z.cycEv A n φ) := by
  rw [cycEv_apply, cycEv_apply]
  have h := permHom_D A (Z.permRes k) n φ ⟨0, 1⟩
  refine h.trans ?_
  show Finsupp.linearCombination k (permHom A (FB 1 C) φ).1 (Z.perD.bdS k n ⟨0, 1⟩) = _
  unfold cycD
  by_cases hn : Even n
  · rw [Z.perD.bdS_even k hn, lc_eqF_difB, if_pos hn]; rfl
  · rw [Z.perD.bdS_odd k hn, lc_eqF_nrmB, if_neg hn, LinearMap.sum_apply]
    rfl

lemma cycD_sq (A : Rep.{0} k C) (n : ℕ) (x : A) : Z.cycD A (n + 1) (Z.cycD A n x) = 0 := by
  obtain ⟨φ, rfl⟩ := (Z.cycEv A n).surjective x
  rw [← cycEv_d, ← cycEv_d, ← Category.assoc]
  have h : (Z.permRes k).D (n + 1) ≫ (Z.permRes k).D n = 0 := PermRes.D_D _ n
  rw [h, Limits.zero_comp, map_zero]

/-- **The periodic cochain complex** `A → A → A → ⋯` (`1 − t`, `N_t` alternating) of the cyclic group `C`. -/
def cycCoch (A : Rep.{0} k C) : CochainComplex (ModuleCat k) ℕ :=
  CochainComplex.of (fun _ => ModuleCat.of k A) (fun n => ModuleCat.ofHom (Z.cycD A n)) (fun n => by
    apply ModuleCat.hom_ext; refine LinearMap.ext fun x => ?_
    exact Z.cycD_sq A n x)

lemma cycCoch_d (A : Rep.{0} k C) (n : ℕ) : (Z.cycCoch A).d n (n + 1) = ModuleCat.ofHom (Z.cycD A n) :=
  CochainComplex.of_d (fun _ : ℕ => ModuleCat.of k A) (fun n => ModuleCat.ofHom (Z.cycD A n)) n

/-- **`Hom_C(P_•, A) ≅ cycCoch A`** (evaluation at the generator). -/
def cycCochIso (A : Rep.{0} k C) : ((Z.permRes k).cx k).linearYonedaObj k A ≅ Z.cycCoch A :=
  HomologicalComplex.Hom.isoOfComponents (fun n => (Z.cycEv A n).toModuleIso) (by
    rintro i j (rfl : i + 1 = j)
    rw [ChainComplex.linearYonedaObj_d, cycCoch_d]
    apply ModuleCat.hom_ext; refine LinearMap.ext fun φ => ?_
    show Z.cycD A i (Z.cycEv A i φ) = Z.cycEv A (i + 1) (((Z.permRes k).cx k).d (i + 1) i ≫ φ)
    rw [PermRes.cx_d]
    exact (Z.cycEv_d A i φ).symm)

end CycDat

/-! ### Cyclic subgroups `⟨t⟩` of finite order -/

/-- **The cyclic subgroup `⟨t⟩`** of an element `t` of order `N` (given as `t^N = 1` and `t^a ≠ t^b` for `a ≠ b < N`),
with the generator `t`. -/
def CycDat.zpow {G : Type} [Group G] (t : G) (N : ℕ) (hN : 0 < N) (hp : t ^ N = 1)
    (hi : ∀ a b : Fin N, t ^ (a : ℕ) = t ^ (b : ℕ) → a = b) : CycDat (Subgroup.zpowers t) where
  t := ⟨t, Subgroup.mem_zpowers t⟩
  N := N
  N_pos := hN
  pow_N := Subtype.ext (by simpa using hp)
  pow_inj a b h := hi a b (by simpa using congrArg Subtype.val h)
  gen g := by
    obtain ⟨g, hg⟩ := g
    have hfin : IsOfFinOrder t := isOfFinOrder_iff_pow_eq_one.2 ⟨N, hN, hp⟩
    obtain ⟨m, rfl⟩ := (hfin.mem_powers_iff_mem_zpowers).2 hg
    refine ⟨⟨m % N, Nat.mod_lt _ hN⟩, Subtype.ext ?_⟩
    simp only [SubgroupClass.coe_pow]
    conv_lhs => rw [← Nat.div_add_mod m N, pow_add, pow_mul, hp, one_pow, one_mul]

/-! ### The augmentation of the bar resolution on generators -/

section BarAug

open Rep MonoidalCategory in
/-- `k[Gⁿ] ≅ k[G^{n+1}]` on generators (mathlib's in-proof computation in `d_comp_diagonalSuccIsoFree_inv_eq`). -/
lemma diagonalSuccIsoFree_inv_single {G : Type} [Group G] {k : Type} [CommRing k] (m : ℕ) (f : Fin m → G) (g : G)
    (r : k) :
    (diagonalSuccIsoFree k G m).inv.hom (Finsupp.single f (MonoidAlgebra.single g r)) =
      MonoidAlgebra.single (g • Fin.partialProd f) r := by
  simp only [diagonalSuccIsoFree, diagonalSuccIsoTensorTrivial, Iso.trans_inv, Rep.hom_comp,
    Representation.IntertwiningMap.comp_apply]
  have step1 : (Hom.hom (leftRegularTensorTrivialIsoFree k G (Fin m → G)).inv)
      (Finsupp.single f (.single g r)) = .single g 1 ⊗ₜ[k] .single f r :=
    Representation.leftRegularTensorTrivialIsoFree_symm_apply_single_single f g r
  rw [step1]
  simp only [mkIso_inv, Representation.linearizeOfMulActionIso, Representation.Equiv.mk_symm,
    LinearEquiv.refl_symm, ConcreteCategory.hom_ofHom, Action.tensorObj_V, Action.trivial_V,
    Functor.mapIso_inv, tensor_V, tensor_ρ, Iso.symm_inv, Functor.Monoidal.μIso_hom, μ_hom,
    MonoidalCategory.tensorIso_inv, Representation.linearizeTrivialIso, hom_tensorHom,
    Representation.IntertwiningMap.tensor_apply, Representation.Equiv.coe_toIntertwiningMap,
    Representation.Equiv.mk_apply, LinearEquiv.refl_apply]
  have key₁ := Representation.linearizeMap_single (k := k)
    (Action.diagonalSuccIsoTensorTrivial G m).inv (g, f) ((1 : k) * r)
  have key₂ := Representation.LinearizeMonoidal.μ_apply_single_single (k := k)
    (X := Action.leftRegular G) (Y := Action.trivial G (Fin m → G)) g f 1 r
  exact ((congrArg (fun z => (Representation.linearizeMap
    (Action.diagonalSuccIsoTensorTrivial G m).inv) z) key₂).trans key₁).trans
    (by rw [one_mul]; congr 1; exact Action.diagonalSuccIsoTensorTrivial_inv_hom_apply g f)

/-- **The augmentation of the bar resolution on generators**: `(g ↦ r) ↦ r`. -/
lemma barπ_zero_single {G : Type} [Group G] {k : Type} [CommRing k] (x : Fin 0 → G) (g : G) (r : k) :
    ((Rep.barResolution k G).π.f 0).hom (Finsupp.single x (MonoidAlgebra.single g r)) = r := by
  show (Rep.standardComplex.ε k G).hom ((Rep.diagonalSuccIsoFree k G 0).inv.hom
    (Finsupp.single x (MonoidAlgebra.single g r))) = r
  rw [diagonalSuccIsoFree_inv_single]
  simp [Rep.standardComplex.ε]

end BarAug

/-! ### Restriction along a group homomorphism, on resolutions -/

section Res

variable {G G' : Type} [Group G] [Group G'] {k : Type} [CommRing k] (f : G' →* G)

/-- `res_f` on chain complexes of representations. -/
abbrev resCx (X : ChainComplex (Rep.{0} k G) ℕ) : ChainComplex (Rep.{0} k G') ℕ :=
  ((Rep.resFunctor f).mapHomologicalComplex _).obj X

/-- **Restriction on `Hom` complexes**: `Hom_G(X, A) ⟶ Hom_{G'}(res X, res A)`, `φ ↦ res φ`. -/
def resY (X : ChainComplex (Rep.{0} k G) ℕ) (A : Rep.{0} k G) :
    X.linearYonedaObj k A ⟶ (resCx f X).linearYonedaObj k (Rep.res f A) where
  f i := ModuleCat.ofHom ((Rep.resFunctor f).mapLinearMap k)
  comm' i j _ := by
    ext φ : 2
    exact (Rep.resFunctor f).map_comp (X.d j i) φ

lemma resY_f_apply (X : ChainComplex (Rep.{0} k G) ℕ) (A : Rep.{0} k G) (i : ℕ) (φ : X.X i ⟶ A) :
    ((resY f X A).f i).hom φ = (Rep.resFunctor f).map φ := rfl

/-- Restriction commutes with precomposition. -/
lemma preY_resY {X Y : ChainComplex (Rep.{0} k G) ℕ} (φ : X ⟶ Y) (A : Rep.{0} k G) :
    Cover.preY φ A ≫ resY f X A = resY f Y A ≫ Cover.preY (((Rep.resFunctor f).mapHomologicalComplex _).map φ) (Rep.res f A) := by
  ext i ψ : 3
  exact (Rep.resFunctor f).map_comp (φ.f i) ψ

/-- The bar resolutions under `f`, on generators: `(g_1, …, g_n) ↦ (f g_1, …, f g_n)`. -/
def barMapF (n : ℕ) : Rep.free k G' (Fin n → G') ⟶ Rep.res f (Rep.free k G (Fin n → G)) :=
  Rep.freeLift (k := k) (G := G') (A := Rep.res f (Rep.free k G (Fin n → G)))
    (fun x => Finsupp.single (f ∘ x) (MonoidAlgebra.single (1 : G) (1 : k)))

lemma barMapF_single (n : ℕ) (x : Fin n → G') (g : G') (r : k) :
    (barMapF f n).hom (Finsupp.single x (MonoidAlgebra.single g r)) =
      Finsupp.single (f ∘ x) (MonoidAlgebra.single (f g) r) := by
  simp [barMapF]

/-- **The bar resolutions under `f`**: a chain map `bar_{G'} ⟶ res_f bar_G`. -/
def barMapCx : Rep.barComplex k G' ⟶ resCx f (Rep.barComplex k G) where
  f n := barMapF f n
  comm' i j (h : j + 1 = i) := by
    subst h
    rw [Functor.mapHomologicalComplex_obj_d, Rep.barComplex.d_def, Rep.barComplex.d_def]
    apply Rep.free_ext; intro x
    show (Rep.barComplex.d k G j).hom ((barMapF f (j + 1)).hom (Finsupp.single x (MonoidAlgebra.single 1 1))) =
      (barMapF f j).hom ((Rep.barComplex.d k G' j).hom (Finsupp.single x (MonoidAlgebra.single 1 1)))
    rw [Rep.barComplex.d_single, map_add, map_sum, barMapF_single, map_one, Rep.barComplex.d_single]
    congr 1
    · exact (barMapF_single f j _ _ _).symm
    · refine Finset.sum_congr rfl fun i _ => ?_
      rw [barMapF_single, map_one, Fin.comp_contractNth (· * ·) (· * ·) (fun a b => map_mul f a b)]

/-- **Inhomogeneous cochains under `f` are precomposition with the bar map** (after restriction). -/
lemma cochainsMap_barMap (A : Rep.{0} k G) :
    groupCohomology.cochainsMap f (𝟙 (Rep.res f A)) ≫ (groupCohomology.inhomogeneousCochainsIso (Rep.res f A)).hom =
      (groupCohomology.inhomogeneousCochainsIso A).hom ≫ resY f _ A ≫ Cover.preY (barMapCx f) (Rep.res f A) := by
  ext i g : 3
  show (Rep.freeLiftLEquiv k G' (Fin i → G') (Rep.res f A)).symm (fun x => g (f ∘ x)) =
    barMapF f i ≫ (Rep.resFunctor f).map ((Rep.freeLiftLEquiv k G (Fin i → G) A).symm g)
  apply Rep.free_ext; intro x
  have h1 := congrFun ((Rep.freeLiftLEquiv k G' (Fin i → G') (Rep.res f A)).apply_symm_apply
    (fun x => g (f ∘ x))) x
  have h2 := congrFun ((Rep.freeLiftLEquiv k G (Fin i → G) A).apply_symm_apply g) (f ∘ x)
  refine h1.trans (h2.symm.trans ?_)
  show _ = ((Rep.freeLiftLEquiv k G (Fin i → G) A).symm g).hom ((barMapF f i).hom
    (Finsupp.single x (MonoidAlgebra.single 1 1)))
  rw [barMapF_single, map_one]
  rfl

/-- **The bar map lifts the identity of `k`**, on generators. -/
lemma barMapF_π (x : Fin 0 → G') :
    ((Rep.barResolution k G).π.f 0).hom ((barMapF f 0).hom (Finsupp.single x (MonoidAlgebra.single 1 1))) =
      ((Rep.barResolution k G').π.f 0).hom (Finsupp.single x (MonoidAlgebra.single 1 1)) := by
  rw [barMapF_single, barπ_zero_single, barπ_zero_single]

/-- `Hom` of a homotopy: precomposition with homotopic chain maps gives homotopic cochain maps. -/
def preYHomotopy {H : Type} [Group H] {X Y : ChainComplex (Rep.{0} k H) ℕ} {φ ψ : X ⟶ Y} (h : Homotopy φ ψ)
    (A : Rep.{0} k H) : Homotopy (Cover.preY φ A) (Cover.preY ψ A) :=
  Homotopy.unop ((((linearYoneda k (Rep.{0} k H)).obj A).rightOp).mapHomotopy h)

end Res

/-! ### Restricted permutation resolutions, and the comparison with the bar resolution -/

namespace PermRes

variable {G G' : Type} [Group G] [Group G'] {k : Type} [CommRing k]

/-- **The restriction of a `PermRes` along an injective homomorphism** `f : G' →* G` (same bases, `G'` acting through `f`,
still freely). -/
def resP (P : PermRes G k) (f : G' →* G) (hf : Function.Injective f) : PermRes G' k where
  B := P.B
  act n := MulAction.compHom (P.B n) f
  bd := P.bd
  bd_smul n g b := P.bd_smul n (f g) b
  bdd := P.bdd
  free n g b h := hf (by rw [P.free n (f g) b h, map_one])
  aug_bd := P.aug_bd
  surj := P.surj
  ex0 := P.ex0
  exS := P.exS

variable (k) in
/-- **The fixed comparison** of a `PermRes` with the bar resolution. -/
def barEquiv (P : PermRes G k) : HomotopyEquiv (P.cx k) (Rep.barComplex k G) :=
  ProjectiveResolution.homotopyEquiv P.res (Rep.barResolution k G)

/-- **`H^q(G; A) ≅ H^q(Hom_G(P_•, A))` through the fixed comparison with the bar resolution**:
inhomogeneous cochains `= Hom(bar, A) ≃ Hom(P_•, A)`. -/
def barIso (P : PermRes G k) (A : Rep.{0} k G) (q : ℕ) :
    groupCohomology A q ≅ ((P.cx k).linearYonedaObj k A).homology q :=
  (HomologicalComplex.homologyFunctor _ _ q).mapIso (groupCohomology.inhomogeneousCochainsIso A) ≪≫
    (Cover.preYEquiv (P.barEquiv k) A).toHomologyIso q

variable (P : PermRes G k) (f : G' →* G) (hf : Function.Injective f)

/-- The identity `k[B_n] → res_f k[B_n]`, as a map of `G'`-representations. -/
def resPToF (n : ℕ) : (P.resP f hf).X k n ⟶ Rep.res f (P.X k n) := Rep.ofHom ⟨LinearMap.id, fun _ => rfl⟩

/-- The identity `res_f k[B_n] → k[B_n]`, as a map of `G'`-representations. -/
def resPFrom (n : ℕ) : Rep.res f (P.X k n) ⟶ (P.resP f hf).X k n := Rep.ofHom ⟨LinearMap.id, fun _ => rfl⟩

/-- `ι : P|_{G'} ⟶ res_f P` (the identity in every degree). -/
def resPι : (P.resP f hf).cx k ⟶ resCx f (P.cx k) where
  f n := P.resPToF f hf n
  comm' i j (h : j + 1 = i) := by
    subst h
    rw [PermRes.cx_d, Functor.mapHomologicalComplex_obj_d, PermRes.cx_d]
    rfl

/-- `ι' : res_f P ⟶ P|_{G'}` (the identity in every degree). -/
def resPι' : resCx f (P.cx k) ⟶ (P.resP f hf).cx k where
  f n := P.resPFrom f hf n
  comm' i j (h : j + 1 = i) := by
    subst h
    rw [PermRes.cx_d, Functor.mapHomologicalComplex_obj_d, PermRes.cx_d]
    rfl

lemma resPι'_ι : P.resPι' f hf ≫ P.resPι f hf = 𝟙 _ := by
  ext n : 2
  rfl

lemma resP_eps_hom (x : (P.resP f hf).X k 0) : ((P.resP f hf).eps).hom x = P.eps.hom x := rfl

variable (P' : PermRes G' k)

/-- The composite `P' → bar_{G'} → res_f bar_G → res_f P = P|_{G'}` of the fixed comparisons and the bar map. -/
def viaBar : P'.cx k ⟶ (P.resP f hf).cx k :=
  (P'.barEquiv k).hom ≫ barMapCx f ≫ ((Rep.resFunctor f).mapHomologicalComplex _).map (P.barEquiv k).inv ≫
    P.resPι' f hf

lemma eps_barEquiv_inv (w : Rep.free k G (Fin 0 → G)) :
    P.eps.hom (((P.barEquiv k).inv.f 0).hom w) = ((Rep.barResolution k G).π.f 0).hom w := by
  have h := congrArg (fun φ => (φ.f 0).hom w)
    (ProjectiveResolution.homotopyEquiv_inv_π P.res (Rep.barResolution k G))
  simp only [HomologicalComplex.comp_f] at h
  rw [← h]
  show P.eps.hom _ = (P.aug.f 0).hom _
  rw [aug_f_zero]
  rfl

lemma barEquiv_hom_π (y : P'.X k 0) :
    ((Rep.barResolution k G').π.f 0).hom (((P'.barEquiv k).hom.f 0).hom y) = P'.eps.hom y := by
  have h := congrArg (fun φ => (φ.f 0).hom y)
    (ProjectiveResolution.homotopyEquiv_hom_π P'.res (Rep.barResolution k G'))
  simp only [HomologicalComplex.comp_f] at h
  refine h.trans ?_
  show (P'.aug.f 0).hom y = _
  rw [aug_f_zero]
  rfl

/-- **The composite lifts the identity of `k`.** -/
lemma viaBar_eps : (P.viaBar f hf P').f 0 ≫ (P.resP f hf).eps = P'.eps := by
  have hu : (barMapF f 0 ≫ (Rep.resFunctor f).map ((P.barEquiv k).inv.f 0) ≫ P.resPFrom f hf 0 ≫
      (P.resP f hf).eps : Rep.free k G' (Fin 0 → G') ⟶ Rep.trivial k G' k) =
      ((Rep.barResolution k G').π.f 0 : Rep.free k G' (Fin 0 → G') ⟶ Rep.trivial k G' k) := by
    apply Rep.free_ext; intro x
    show P.eps.hom (((P.barEquiv k).inv.f 0).hom ((barMapF f 0).hom
      (Finsupp.single x (MonoidAlgebra.single 1 1)))) = _
    rw [eps_barEquiv_inv, barMapF_π]
    rfl
  apply Rep.hom_ext; apply Representation.IntertwiningMap.ext; refine LinearMap.ext fun y => ?_
  have h := congrArg (fun φ => φ.hom (((P'.barEquiv k).hom.f 0).hom y)) hu
  exact h.trans (barEquiv_hom_π P' y)

/-- A chain map `P' → P|_{G'}` lifting the identity commutes with the augmentations. -/
lemma comm_of_eps (g : P'.cx k ⟶ (P.resP f hf).cx k) (hg : g.f 0 ≫ (P.resP f hf).eps = P'.eps) :
    g ≫ (P.resP f hf).res.π = P'.res.π ≫ (ChainComplex.single₀ (Rep k G')).map (𝟙 _) := by
  rw [CategoryTheory.Functor.map_id, Category.comp_id]
  apply HomologicalComplex.to_single_hom_ext
  show g.f 0 ≫ (P.resP f hf).aug.f 0 = P'.aug.f 0
  rw [aug_f_zero, aug_f_zero]
  exact hg

/-- **Uniqueness up to homotopy**: for a chain map `α : P' → P|_{G'}` lifting the identity, `α` followed by the comparison
of `P` with the bar resolution is homotopic to the comparison of `P'` followed by the bar map. -/
def resHomotopy (α : P'.cx k ⟶ (P.resP f hf).cx k) (hα : α.f 0 ≫ (P.resP f hf).eps = P'.eps) :
    Homotopy (α ≫ P.resPι f hf ≫ ((Rep.resFunctor f).mapHomologicalComplex _).map (P.barEquiv k).hom)
      ((P'.barEquiv k).hom ≫ barMapCx f) :=
  ((ProjectiveResolution.liftHomotopy (𝟙 _) (P := P'.res) (Q := (P.resP f hf).res) α (P.viaBar f hf P')
      (P.comm_of_eps f hf P' α hα) (P.comm_of_eps f hf P' _ (P.viaBar_eps f hf P'))).compRight _).trans
    ((Homotopy.ofEq (HomologicalComplex.hom_ext _ _ fun _ => rfl)).trans
      ((((Rep.resFunctor f).mapHomotopy (P.barEquiv k).homotopyInvHomId).compLeft
          ((P'.barEquiv k).hom ≫ barMapCx f)).trans
        (Homotopy.ofEq (HomologicalComplex.hom_ext _ _ fun _ => rfl))))

/-- **Restriction in group cohomology, through the fixed comparisons**: for a chain map `α : P' → P|_{G'}` lifting the
identity of `k`, `res_f : H^q(G; A) → H^q(G'; A)` is induced by `Hom_G(P, A) → Hom_{G'}(P', A)`, `φ ↦ φ ∘ α`. -/
theorem barIso_res (α : P'.cx k ⟶ (P.resP f hf).cx k) (hα : α.f 0 ≫ (P.resP f hf).eps = P'.eps)
    (A : Rep.{0} k G) (q : ℕ) :
    groupCohomology.map f (𝟙 (Rep.res f A)) q ≫ (P'.barIso (Rep.res f A) q).hom =
      (P.barIso A q).hom ≫ HomologicalComplex.homologyMap
        (resY f (P.cx k) A ≫ Cover.preY (P.resPι f hf) (Rep.res f A) ≫ Cover.preY α (Rep.res f A)) q := by
  have key : groupCohomology.cochainsMap f (𝟙 (Rep.res f A)) ≫
      (groupCohomology.inhomogeneousCochainsIso (Rep.res f A)).hom ≫ Cover.preY (P'.barEquiv k).hom (Rep.res f A) =
      (groupCohomology.inhomogeneousCochainsIso A).hom ≫ resY f _ A ≫
        Cover.preY ((P'.barEquiv k).hom ≫ barMapCx f) (Rep.res f A) := by
    rw [← Category.assoc, cochainsMap_barMap, Cover.preY_comp]
    simp only [Category.assoc]
  have key2 : (groupCohomology.inhomogeneousCochainsIso A).hom ≫ Cover.preY (P.barEquiv k).hom A ≫
      resY f (P.cx k) A ≫ Cover.preY (P.resPι f hf) (Rep.res f A) ≫ Cover.preY α (Rep.res f A) =
      (groupCohomology.inhomogeneousCochainsIso A).hom ≫ resY f _ A ≫
        Cover.preY (α ≫ P.resPι f hf ≫ ((Rep.resFunctor f).mapHomologicalComplex _).map (P.barEquiv k).hom)
          (Rep.res f A) := by
    rw [← Category.assoc (Cover.preY (P.barEquiv k).hom A), preY_resY]
    simp only [Cover.preY_comp, Category.assoc]
  have hH := (preYHomotopy (P.resHomotopy f hf P' α hα) (Rep.res f A)).homologyMap_eq q
  show HomologicalComplex.homologyMap (groupCohomology.cochainsMap f (𝟙 (Rep.res f A))) q ≫
      (HomologicalComplex.homologyMap (groupCohomology.inhomogeneousCochainsIso (Rep.res f A)).hom q ≫
        HomologicalComplex.homologyMap (Cover.preY (P'.barEquiv k).hom (Rep.res f A)) q) =
    (HomologicalComplex.homologyMap (groupCohomology.inhomogeneousCochainsIso A).hom q ≫
        HomologicalComplex.homologyMap (Cover.preY (P.barEquiv k).hom A) q) ≫ _
  simp only [← HomologicalComplex.homologyMap_comp, Category.assoc]
  rw [key, key2]
  simp only [HomologicalComplex.homologyMap_comp, hH]

end PermRes

/-! ### Chain maps of permutation resolutions given on bases -/

namespace PermRes

variable {G : Type} [Group G] {k : Type} [CommRing k]

/-- **A chain map `k[P_•] → k[Q_•]` given on bases** (equivariant, commuting with the boundaries). -/
def mapB (P Q : PermRes G k) (a : ∀ n, P.B n → (Q.B n →₀ k))
    (ha : ∀ n (g : G) b, a n (g • b) = mapDomain (g • ·) (a n b))
    (hc : ∀ n b, Finsupp.linearCombination k (Q.bd n) (a (n + 1) b) =
      Finsupp.linearCombination k (a n) (P.bd n b)) :
    P.cx k ⟶ Q.cx k where
  f n := Rep.ofHom ⟨(MonoidAlgebra.coeffLinearEquiv k).symm.toLinearMap ∘ₗ Finsupp.linearCombination k (a n) ∘ₗ
      (MonoidAlgebra.coeffLinearEquiv k).toLinearMap, fun g => by
    refine LinearMap.ext fun x => ?_
    simp only [LinearMap.coe_comp, Function.comp_apply, LinearEquiv.coe_coe, Representation.ofMulAction_def,
      LinearEquiv.apply_symm_apply, Finsupp.lmapDomain_apply]
    rw [Cover.lc_mapDomain_smul k (a n) g (ha n g)]⟩
  comm' i j (h : j + 1 = i) := by
    subst h
    rw [cx_d, cx_d]
    apply Rep.hom_ext; apply Representation.IntertwiningMap.ext; refine LinearMap.ext fun x => ?_
    show (MonoidAlgebra.coeffLinearEquiv k).symm (Q.dP j (MonoidAlgebra.coeffLinearEquiv k
        ((MonoidAlgebra.coeffLinearEquiv k).symm (Finsupp.linearCombination k (a (j + 1))
          (MonoidAlgebra.coeffLinearEquiv k x))))) =
      (MonoidAlgebra.coeffLinearEquiv k).symm (Finsupp.linearCombination k (a j) (MonoidAlgebra.coeffLinearEquiv k
        ((MonoidAlgebra.coeffLinearEquiv k).symm (P.dP j (MonoidAlgebra.coeffLinearEquiv k x)))))
    rw [LinearEquiv.apply_symm_apply, LinearEquiv.apply_symm_apply, dP, dP, ← lc_lc, ← lc_lc]
    have hF : (fun b => Finsupp.linearCombination k (Q.bd j) (a (j + 1) b)) =
        (fun b => Finsupp.linearCombination k (a j) (P.bd j b)) := funext fun b => hc j b
    rw [hF]

lemma mapB_f_hom (P Q : PermRes G k) (a ha hc) (n : ℕ) (x) :
    ((mapB P Q a ha hc).f n).hom x = (MonoidAlgebra.coeffLinearEquiv k).symm
      (Finsupp.linearCombination k (a n) (MonoidAlgebra.coeffLinearEquiv k x)) := rfl

/-- **It lifts the identity of `k`** when every basis vector of degree `0` goes to coefficient sum `1`. -/
lemma mapB_eps (P Q : PermRes G k) (a ha hc)
    (h0 : ∀ b, Finsupp.linearCombination k (fun _ => (1 : k)) (a 0 b) = 1) :
    (mapB P Q a ha hc).f 0 ≫ Q.eps = P.eps := by
  apply Rep.hom_ext; apply Representation.IntertwiningMap.ext; refine LinearMap.ext fun x => ?_
  show Q.augL (MonoidAlgebra.coeffLinearEquiv k ((MonoidAlgebra.coeffLinearEquiv k).symm
    (Finsupp.linearCombination k (a 0) (MonoidAlgebra.coeffLinearEquiv k x)))) = P.augL (MonoidAlgebra.coeffLinearEquiv k x)
  rw [LinearEquiv.apply_symm_apply]
  generalize MonoidAlgebra.coeffLinearEquiv k x = y
  induction y using Finsupp.induction_linear with
  | zero => simp
  | add x y hx hy => rw [map_add, map_add, hx, hy, map_add]
  | single b c => rw [linearCombination_single, map_smul, augL, h0, augL, linearCombination_single, smul_eq_mul, mul_one]

end PermRes

/-! ### Restriction between finite cyclic groups, explicitly -/

/-- `∑_{m < d N} F m = ∑_{i < N} ∑_{j < d} F (d i + j)`. -/
lemma sum_range_mul_eq {M : Type} [AddCommMonoid M] (F : ℕ → M) (d N : ℕ) :
    ∑ m ∈ Finset.range (d * N), F m = ∑ i ∈ Finset.range N, ∑ j ∈ Finset.range d, F (d * i + j) := by
  induction N with
  | zero => simp
  | succ N ih => rw [Nat.mul_succ, Finset.sum_range_add, ih, Finset.sum_range_succ]

namespace CycDat

variable {C C' : Type} [Group C] [Group C'] {k : Type} [CommRing k] (Z : CycDat C) (Z' : CycDat C') (f : C' →* C)
  (hf : Function.Injective f) (d : ℕ) (hd : f Z'.t = Z.t ^ d) (hN : Z.N = d * Z'.N)

/-- **`H^q(C; A) ≅ H^q(cycCoch A)`** for a finite cyclic group `C`, every `q`, every commutative ring `k`: inhomogeneous
cochains `= Hom(bar, A) ≃ Hom(P_•, A) = cycCoch A`, through the fixed comparison `PermRes.barEquiv`. -/
def cohIso (A : Rep.{0} k C) (q : ℕ) : groupCohomology A q ≅ (Z.cycCoch A).homology q :=
  (Z.permRes k).barIso A q ≪≫ (HomologicalComplex.homologyFunctor _ _ q).mapIso (Z.cycCochIso A)

variable (k) in
/-- **The comparison `P^{C'} → P^C|_{C'}` on bases**: `e ↦ e` in even degrees, `e ↦ ∑_{j<d} t^j e` in odd degrees
(`f(t') = t^d`). -/
def resA (n : ℕ) (b : FB 1 C') : FB 1 C →₀ k :=
  if Even n then Finsupp.single ⟨b.c, f b.g⟩ 1 else ∑ j ∈ Finset.range d, Finsupp.single ⟨b.c, f b.g * Z.t ^ j⟩ 1

lemma resA_smul (n : ℕ) (g : C') (b : FB 1 C') :
    resA k Z f d n (g • b) = mapDomain (fun y : FB 1 C => f g • y) (resA k Z f d n b) := by
  unfold resA
  split_ifs
  · rw [mapDomain_single]; simp [FB.smul_g, map_mul]; rfl
  · rw [mapDomain_finsetSum]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [mapDomain_single]; simp [FB.smul_g, map_mul, mul_assoc]; rfl

include hd hN in
lemma resA_comm (n : ℕ) (b : FB 1 C') :
    Finsupp.linearCombination k (Z.perD.bdS k n) (resA k Z f d (n + 1) b) =
      Finsupp.linearCombination k (resA k Z f d n) (Z'.perD.bdS k n b) := by
  by_cases hn : Even n
  · have hn1 : ¬ Even (n + 1) := fun h => Nat.even_add_one.1 h hn
    rw [Z.perD.bdS_even k hn, Z'.perD.bdS_even k hn]
    simp only [resA, if_neg hn1, map_sum, linearCombination_single, one_smul, PerData.difB, map_sub]
    rw [if_pos hn, if_pos hn]
    show ∑ j ∈ Finset.range d, (Finsupp.single (⟨b.c, f b.g * Z.t ^ j⟩ : FB 1 C) (1 : k) -
        Finsupp.single ⟨b.c, f b.g * Z.t ^ j * Z.t⟩ 1) =
      Finsupp.single ⟨b.c, f b.g⟩ 1 - Finsupp.single ⟨b.c, f (b.g * Z'.t)⟩ 1
    rw [map_mul, hd]
    simp only [mul_assoc, ← pow_succ]
    rw [Finset.sum_range_sub' (fun j => (Finsupp.single (⟨b.c, f b.g * Z.t ^ j⟩ : FB 1 C) (1 : k))), pow_zero,
      mul_one]
  · have hn1 : Even (n + 1) := Nat.even_add_one.2 hn
    rw [Z.perD.bdS_odd k hn, Z'.perD.bdS_odd k hn]
    simp only [resA, if_pos hn1, if_neg hn, linearCombination_single, one_smul, PerData.nrmB, map_sum]
    show ∑ m ∈ Finset.range Z.N, (Finsupp.single (⟨b.c, f b.g * Z.t ^ m⟩ : FB 1 C) (1 : k)) =
      ∑ i ∈ Finset.range Z'.N, ∑ j ∈ Finset.range d, Finsupp.single ⟨b.c, f (b.g * Z'.t ^ i) * Z.t ^ j⟩ 1
    rw [hN, sum_range_mul_eq]
    refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
    rw [map_mul, map_pow, hd, mul_assoc, ← pow_mul, ← pow_add]

variable (k) in
include hd hN in
/-- **The comparison `α : P^{C'} → P^C|_{C'}`**, a chain map of resolutions of `k` over `k[C']`. -/
def resα : (Z'.permRes k).cx k ⟶ ((Z.permRes k).resP f hf).cx k :=
  PermRes.mapB (Z'.permRes k) ((Z.permRes k).resP f hf) (resA k Z f d) (fun n g b => resA_smul Z f d n g b)
    (fun n b => resA_comm Z Z' f d hd hN n b)

include hd hN in
lemma resα_eps : (resα k Z Z' f hf d hd hN).f 0 ≫ ((Z.permRes k).resP f hf).eps = (Z'.permRes k).eps :=
  PermRes.mapB_eps _ _ _ _ _ fun b => by
    unfold resA; rw [if_pos PerData.even0]; exact (linearCombination_single k _ _).trans (one_smul k 1)

/-- **Restriction on the periodic cochains**: `x ↦ x` in even degrees, `x ↦ ∑_{j<d} t^j x` in odd degrees. -/
def cycRes (A : Rep.{0} k C) : Z.cycCoch A ⟶ Z'.cycCoch (Rep.res f A) :=
  (Z.cycCochIso A).inv ≫ resY f _ A ≫ Cover.preY ((Z.permRes k).resPι f hf) (Rep.res f A) ≫
    Cover.preY (resα k Z Z' f hf d hd hN) (Rep.res f A) ≫ (Z'.cycCochIso (Rep.res f A)).hom

include hd hN in
lemma cycRes_f_apply (A : Rep.{0} k C) (n : ℕ) (x : A) :
    ((cycRes Z Z' f hf d hd hN A).f n).hom x =
      if Even n then x else ∑ j ∈ Finset.range d, A.ρ (Z.t ^ j) x := by
  obtain ⟨φ, rfl⟩ := (Z.cycEv A n).surjective x
  have h1 : ((Z.cycCochIso A).inv.f n).hom (Z.cycEv A n φ) = φ := (Z.cycEv A n).symm_apply_apply φ
  show Z'.cycEv (Rep.res f A) n ((resα k Z Z' f hf d hd hN).f n ≫ ((Z.permRes k).resPι f hf).f n ≫
    (Rep.resFunctor f).map (((Z.cycCochIso A).inv.f n).hom (Z.cycEv A n φ))) = _
  rw [h1, cycEv_apply, cycEv_apply]
  show φ.hom ((MonoidAlgebra.coeffLinearEquiv k).symm (Finsupp.linearCombination k (resA k Z f d n)
    (MonoidAlgebra.coeffLinearEquiv k (MonoidAlgebra.single (⟨0, 1⟩ : FB 1 C') (1 : k))))) = _
  rw [Cover.coeff_single, linearCombination_single, one_smul]
  refine (permHom_lc (X := FB 1 C) A φ _).trans ?_
  unfold resA
  split_ifs
  · rw [linearCombination_single, one_smul, map_one]; rfl
  · rw [map_sum]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [linearCombination_single, one_smul, map_one, one_mul]
    exact eqF_FB A (permHom A (FB 1 C) φ) 0 (Z.t ^ j)

include hd hN in
/-- **Restriction `H^q(C; A) → H^q(C'; A)` on the periodic cochains** (through the fixed comparisons `cohIso`). -/
theorem cohIso_res (A : Rep.{0} k C) (q : ℕ) :
    groupCohomology.map f (𝟙 (Rep.res f A)) q ≫ (Z'.cohIso (Rep.res f A) q).hom =
      (Z.cohIso A q).hom ≫ HomologicalComplex.homologyMap (cycRes Z Z' f hf d hd hN A) q := by
  have h := (Z.permRes k).barIso_res f hf (Z'.permRes k) (resα k Z Z' f hf d hd hN) (resα_eps Z Z' f hf d hd hN) A q
  show groupCohomology.map f (𝟙 (Rep.res f A)) q ≫ ((Z'.permRes k).barIso (Rep.res f A) q).hom ≫
      HomologicalComplex.homologyMap (Z'.cycCochIso (Rep.res f A)).hom q =
    (((Z.permRes k).barIso A q).hom ≫ HomologicalComplex.homologyMap (Z.cycCochIso A).hom q) ≫ _
  rw [← Category.assoc, h, cycRes]
  simp only [HomologicalComplex.homologyMap_comp, Category.assoc]
  rw [← HomologicalComplex.homologyMap_comp_assoc (Z.cycCochIso A).hom (Z.cycCochIso A).inv, Iso.hom_inv_id,
    HomologicalComplex.homologyMap_id, Category.id_comp]

end CycDat

/-! ### Homology through a chosen short complex, on cocycles -/

lemma liftCycles_π_homologyIsoSc' {k : Type} [CommRing k] (K : CochainComplex (ModuleCat.{0} k) ℕ) (i j l : ℕ) (hi : (ComplexShape.up ℕ).prev j = i) (hl : (ComplexShape.up ℕ).next j = l)
    {W : ModuleCat.{0} k} (z : W ⟶ K.X j) (hz : z ≫ K.d j l = 0) :
    K.liftCycles z l hl hz ≫ K.homologyπ j ≫ (K.homologyIsoSc' i j l hi hl).hom =
      (K.sc' i j l).liftCycles z hz ≫ (K.sc' i j l).homologyπ := by
  rw [HomologicalComplex.π_homologyIsoSc'_hom, ← Category.assoc]
  congr 1
  rw [← cancel_mono (K.sc' i j l).iCycles]
  simp only [Category.assoc, HomologicalComplex.cyclesIsoSc'_hom_iCycles]
  exact (HomologicalComplex.liftCycles_i K z l hl hz).trans (ShortComplex.liftCycles_i (K.sc' i j l) z hz).symm

/-! ### `SL₂(ℤ)`: the cyclic subgroups `⟨S⟩`, `⟨U⟩`, `⟨−I⟩` -/

section Gamma

open scoped MatrixGroups
open Limits

/-- `⟨S⟩ ≤ SL₂(ℤ)` (order 4). -/
abbrev gS : Subgroup SL(2, ℤ) := Subgroup.zpowers sS
/-- `⟨U⟩ ≤ SL₂(ℤ)` (order 6). -/
abbrev gU : Subgroup SL(2, ℤ) := Subgroup.zpowers sU
/-- `⟨−I⟩ ≤ SL₂(ℤ)` (order 2). -/
abbrev gI : Subgroup SL(2, ℤ) := Subgroup.zpowers (-1)

/-- The generator `S` of `⟨S⟩`. -/
def zS : CycDat gS := CycDat.zpow sS 4 (by norm_num) sS_four sS_pow_inj
/-- The generator `U` of `⟨U⟩`. -/
def zU : CycDat gU := CycDat.zpow sU 6 (by norm_num) sU_six sU_pow_inj
/-- The generator `−I` of `⟨−I⟩`. -/
def zI : CycDat gI := CycDat.zpow (-1) 2 (by norm_num) negOne_sq (fun a b h => model7PerE.pow_inj 0 a b h)

lemma gI_le_gS : gI ≤ gS := by
  rw [Subgroup.zpowers_le, negOne_eq_sS_sq]
  exact Subgroup.pow_mem _ (Subgroup.mem_zpowers sS) 2

lemma gI_le_gU : gI ≤ gU := by
  rw [Subgroup.zpowers_le, ← sU_cube]
  exact Subgroup.pow_mem _ (Subgroup.mem_zpowers sU) 3

/-- `⟨−I⟩ ↪ ⟨S⟩`. -/
abbrev fS : gI →* gS := Subgroup.inclusion gI_le_gS
/-- `⟨−I⟩ ↪ ⟨U⟩`. -/
abbrev fU : gI →* gU := Subgroup.inclusion gI_le_gU

lemma fS_t : fS zI.t = zS.t ^ 2 := Subtype.ext sS_sq.symm
lemma fU_t : fU zI.t = zU.t ^ 3 := Subtype.ext sU_cube.symm

variable {k : Type} [CommRing k] (A : Rep.{0} k SL(2, ℤ))

/-- `A` restricted to `⟨S⟩`. -/
abbrev resS : Rep.{0} k gS := Rep.res gS.subtype A
/-- `A` restricted to `⟨U⟩`. -/
abbrev resU : Rep.{0} k gU := Rep.res gU.subtype A
/-- `A` restricted to `⟨−I⟩`. -/
abbrev resI : Rep.{0} k gI := Rep.res gI.subtype A

/-- **Restriction** `H^q(⟨S⟩; A) → H^q(⟨−I⟩; A)` (mathlib's `groupCohomology.map` along `⟨−I⟩ ↪ ⟨S⟩`). -/
def resSI (q : ℕ) : groupCohomology (resS A) q ⟶ groupCohomology (resI A) q :=
  groupCohomology.map fS (𝟙 (Rep.res fS (resS A))) q

/-- **Restriction** `H^q(⟨U⟩; A) → H^q(⟨−I⟩; A)`. -/
def resUI (q : ℕ) : groupCohomology (resU A) q ⟶ groupCohomology (resI A) q :=
  groupCohomology.map fU (𝟙 (Rep.res fU (resU A))) q

/-- The periodic cochain complexes of `⟨S⟩`, `⟨U⟩`, `⟨−I⟩`. -/
abbrev cS : CochainComplex (ModuleCat k) ℕ := zS.cycCoch (resS A)
abbrev cU : CochainComplex (ModuleCat k) ℕ := zU.cycCoch (resU A)
abbrev cI : CochainComplex (ModuleCat k) ℕ := zI.cycCoch (resI A)

lemma vtxA_eq (n : ℕ) (x y : A) : vtxA A n (x, y) = (zS.cycD (resS A) n x, zU.cycD (resU A) n y) := by
  unfold vtxA CycDat.cycD
  by_cases h : Even n
  · simp only [if_pos h]; rfl
  · simp only [if_neg h]; rfl

lemma edgA_eq (n : ℕ) (x : A) : edgA A n x = zI.cycD (resI A) n x := by
  unfold edgA CycDat.cycD
  by_cases h : Even n
  · simp only [if_pos h]; rfl
  · simp only [if_neg h]; rfl

/-! #### The vertex term: `H^q(vCoch A) ≅ H^q(⟨S⟩; A) ⊕ H^q(⟨U⟩; A)` -/

/-- `x ↦ (x, 0)`. -/
def jS : cS A ⟶ vCoch A :=
  CochainComplex.ofHom (X := cS A) (Y := vCoch A) (fun _ => ModuleCat.ofHom (LinearMap.inl k A A)) (fun i => by
    rw [CycDat.cycCoch_d, vCoch_d]
    apply ModuleCat.hom_ext; refine LinearMap.ext fun (x : A) => ?_
    show vtxA A i (x, 0) = (zS.cycD (resS A) i x, (0 : A))
    rw [vtxA_eq, map_zero])

/-- `y ↦ (0, y)`. -/
def jU : cU A ⟶ vCoch A :=
  CochainComplex.ofHom (X := cU A) (Y := vCoch A) (fun _ => ModuleCat.ofHom (LinearMap.inr k A A)) (fun i => by
    rw [CycDat.cycCoch_d, vCoch_d]
    apply ModuleCat.hom_ext; refine LinearMap.ext fun (y : A) => ?_
    show vtxA A i (0, y) = ((0 : A), zU.cycD (resU A) i y)
    rw [vtxA_eq, map_zero])

/-- `(x, y) ↦ x`. -/
def pS : vCoch A ⟶ cS A :=
  CochainComplex.ofHom (X := vCoch A) (Y := cS A) (fun _ => ModuleCat.ofHom (LinearMap.fst k A A)) (fun i => by
    rw [CycDat.cycCoch_d, vCoch_d]
    apply ModuleCat.hom_ext; refine LinearMap.ext fun (w : A × A) => ?_
    show zS.cycD (resS A) i w.1 = (vtxA A i w).1
    obtain ⟨x, y⟩ := w
    rw [vtxA_eq])

/-- `(x, y) ↦ y`. -/
def pU : vCoch A ⟶ cU A :=
  CochainComplex.ofHom (X := vCoch A) (Y := cU A) (fun _ => ModuleCat.ofHom (LinearMap.snd k A A)) (fun i => by
    rw [CycDat.cycCoch_d, vCoch_d]
    apply ModuleCat.hom_ext; refine LinearMap.ext fun (w : A × A) => ?_
    show zU.cycD (resU A) i w.2 = (vtxA A i w).2
    obtain ⟨x, y⟩ := w
    rw [vtxA_eq])

lemma jS_pS : jS A ≫ pS A = 𝟙 _ := by ext n x : 3; rfl
lemma jU_pU : jU A ≫ pU A = 𝟙 _ := by ext n x : 3; rfl
lemma jS_pU : jS A ≫ pU A = 0 := by ext n x : 3; rfl
lemma jU_pS : jU A ≫ pS A = 0 := by ext n x : 3; rfl
lemma pS_jS_add : pS A ≫ jS A + pU A ≫ jU A = 𝟙 _ := by
  ext n w : 3
  obtain ⟨x, y⟩ := w
  show ((x, (0 : A)) + ((0 : A), y) : A × A) = (x, y)
  simp

/-- **`H^q(vCoch A) ≅ H^q(cS) ⊕ H^q(cU)`** (the vertex complex is the direct sum). -/
def vHomIso (q : ℕ) : (vCoch A).homology q ≅ (cS A).homology q ⊞ (cU A).homology q where
  hom := biprod.lift (HomologicalComplex.homologyMap (pS A) q) (HomologicalComplex.homologyMap (pU A) q)
  inv := biprod.desc (HomologicalComplex.homologyMap (jS A) q) (HomologicalComplex.homologyMap (jU A) q)
  hom_inv_id := by
    rw [biprod.lift_desc, ← HomologicalComplex.homologyMap_comp, ← HomologicalComplex.homologyMap_comp,
      ← HomologicalComplex.homologyMap_add, pS_jS_add, HomologicalComplex.homologyMap_id]
  inv_hom_id := by
    apply biprod.hom_ext'
    · rw [biprod.inl_desc_assoc, Category.comp_id]
      apply biprod.hom_ext
      · rw [Category.assoc, biprod.lift_fst, ← HomologicalComplex.homologyMap_comp, jS_pS,
          HomologicalComplex.homologyMap_id, biprod.inl_fst]
      · rw [Category.assoc, biprod.lift_snd, ← HomologicalComplex.homologyMap_comp, jS_pU,
          HomologicalComplex.homologyMap_zero, biprod.inl_snd]
    · rw [biprod.inr_desc_assoc, Category.comp_id]
      apply biprod.hom_ext
      · rw [Category.assoc, biprod.lift_fst, ← HomologicalComplex.homologyMap_comp, jU_pS,
          HomologicalComplex.homologyMap_zero, biprod.inr_fst]
      · rw [Category.assoc, biprod.lift_snd, ← HomologicalComplex.homologyMap_comp, jU_pU,
          HomologicalComplex.homologyMap_id, biprod.inr_snd]

/-- **The vertex term**: `H^q(vCoch A) ≅ H^q(⟨S⟩; A) ⊕ H^q(⟨U⟩; A)`, every `q`, every commutative ring `k`. -/
def vIso (q : ℕ) : (vCoch A).homology q ≅ groupCohomology (resS A) q ⊞ groupCohomology (resU A) q :=
  vHomIso A q ≪≫ biprod.mapIso (zS.cohIso (resS A) q).symm (zU.cohIso (resU A) q).symm

lemma inl_vIso_inv (q : ℕ) :
    biprod.inl ≫ (vIso A q).inv = (zS.cohIso (resS A) q).hom ≫ HomologicalComplex.homologyMap (jS A) q := by
  simp [vIso, vHomIso]

lemma inr_vIso_inv (q : ℕ) :
    biprod.inr ≫ (vIso A q).inv = (zU.cohIso (resU A) q).hom ≫ HomologicalComplex.homologyMap (jU A) q := by
  simp [vIso, vHomIso]

/-! #### The edge term: `H^{q+1}(eSh A) ≅ H^q(⟨−I⟩; A)` -/

lemma prev_eq_pred (q : ℕ) : (ComplexShape.up ℕ).prev q = q - 1 := by
  rcases q with _ | m
  · exact CochainComplex.prev_nat_zero
  · exact CochainComplex.prev_nat_succ m

/-- The first component of the comparison of short complexes: `0` in degree `0`, `−1` above. -/
def eτ₁ : ∀ q, (cI A).X (q - 1) ⟶ (eSh A).X q
  | 0 => 0
  | _ + 1 => -𝟙 (ModuleCat.of k A)

lemma cI_d_zero : (cI A).d 0 0 = 0 := (cI A).shape _ _ (by simp)

/-- **The comparison** `(cI, q−1 → q → q+1) ⟶ (eSh, q → q+1 → q+2)`: components `(0 or −1, 1, −1)`
(identity on the middle term, hence on cocycles). -/
def eScMap (q : ℕ) : (cI A).sc' (q - 1) q (q + 1) ⟶ (eSh A).sc' q (q + 1) (q + 2) where
  τ₁ := eτ₁ A q
  τ₂ := 𝟙 (ModuleCat.of k A)
  τ₃ := -𝟙 (ModuleCat.of k A)
  comm₁₂ := by
    rcases q with _ | m
    · apply ModuleCat.hom_ext; refine LinearMap.ext fun (x : A) => ?_
      show ((eSh A).d 0 1).hom ((0 : (cI A).X 0 ⟶ (eSh A).X 0).hom x) = ((cI A).d 0 0).hom x
      rw [cI_d_zero]
      simp only [ModuleCat.hom_zero]
      exact map_zero _
    · apply ModuleCat.hom_ext; refine LinearMap.ext fun (x : A) => ?_
      show ((eSh A).d (m + 1) (m + 2)).hom (-x) = ((cI A).d m (m + 1)).hom x
      rw [eSh_d, CycDat.cycCoch_d]
      show (-edgA A m) (-x) = zI.cycD (resI A) m x
      rw [LinearMap.neg_apply, map_neg, neg_neg, edgA_eq]
  comm₂₃ := by
    apply ModuleCat.hom_ext; refine LinearMap.ext fun (x : A) => ?_
    show ((eSh A).d (q + 1) (q + 2)).hom x = -(((cI A).d q (q + 1)).hom x)
    rw [eSh_d, CycDat.cycCoch_d]
    show (-edgA A q) x = -(zI.cycD (resI A) q x)
    rw [LinearMap.neg_apply, edgA_eq]

instance (q : ℕ) : IsIso (ShortComplex.homologyMap (eScMap A q)) := by
  have hneg : IsIso (-𝟙 (ModuleCat.of k A)) := ⟨⟨-𝟙 _, by simp, by simp⟩⟩
  refine ShortComplex.isIso_homologyMap_of_epi_of_isIso_of_mono' _ ?_ ?_ ?_
  · rcases q with _ | m
    · rw [ModuleCat.epi_iff_surjective]
      intro y; exact ⟨0, Subsingleton.elim (α := PUnit) _ _⟩
    · show Epi (-𝟙 (ModuleCat.of k A))
      infer_instance
  · show IsIso (𝟙 (ModuleCat.of k A)); infer_instance
  · show Mono (-𝟙 (ModuleCat.of k A))
    exact inferInstance

/-- **`H^q(cI) ≅ H^{q+1}(eSh A)`**, the identity on cocycles (through the comparison `(0 or −1, 1, −1)`). -/
def eHomIso (q : ℕ) : (cI A).homology q ≅ (eSh A).homology (q + 1) :=
  (cI A).homologyIsoSc' (q - 1) q (q + 1) (prev_eq_pred q) ((ComplexShape.up ℕ).next_eq' rfl) ≪≫
    asIso (ShortComplex.homologyMap (eScMap A q)) ≪≫
    ((eSh A).homologyIsoSc' q (q + 1) (q + 2) (CochainComplex.prev_nat_succ q)
      ((ComplexShape.up ℕ).next_eq' rfl)).symm

/-- **The edge term**: `H^{q+1}(eSh A) ≅ H^q(⟨−I⟩; A)`, every `q`, every commutative ring `k`. -/
def eIso (q : ℕ) : (eSh A).homology (q + 1) ≅ groupCohomology (resI A) q :=
  (eHomIso A q).symm ≪≫ (zI.cohIso (resI A) q).symm

/-- `eHomIso` is the identity on cocycles. -/
lemma eHomIso_π (q : ℕ) {W : ModuleCat.{0} k} (z : W ⟶ (cI A).X q) (hz : z ≫ (cI A).d q (q + 1) = 0)
    (hz' : z ≫ (eSh A).d (q + 1) (q + 2) = 0) :
    (cI A).liftCycles z (q + 1) ((ComplexShape.up ℕ).next_eq' rfl) hz ≫ (cI A).homologyπ q ≫ (eHomIso A q).hom =
      (eSh A).liftCycles z (q + 2) ((ComplexShape.up ℕ).next_eq' rfl) hz' ≫ (eSh A).homologyπ (q + 1) := by
  have h1 := liftCycles_π_homologyIsoSc' (cI A) (q - 1) q (q + 1) (prev_eq_pred q)
    ((ComplexShape.up ℕ).next_eq' rfl) z hz
  have h3 := liftCycles_π_homologyIsoSc' (eSh A) q (q + 1) (q + 2) (CochainComplex.prev_nat_succ q)
    ((ComplexShape.up ℕ).next_eq' rfl) z hz'
  have h2 : ((cI A).sc' (q - 1) q (q + 1)).liftCycles z hz ≫ ((cI A).sc' (q - 1) q (q + 1)).homologyπ ≫
      ShortComplex.homologyMap (eScMap A q) =
      ((eSh A).sc' q (q + 1) (q + 2)).liftCycles z hz' ≫ ((eSh A).sc' q (q + 1) (q + 2)).homologyπ := by
    rw [ShortComplex.homologyπ_naturality, ← Category.assoc]
    erw [ShortComplex.liftCycles_comp_cyclesMap]
    rfl
  have h3' : ((eSh A).sc' q (q + 1) (q + 2)).liftCycles z hz' ≫ ((eSh A).sc' q (q + 1) (q + 2)).homologyπ ≫
      ((eSh A).homologyIsoSc' q (q + 1) (q + 2) (CochainComplex.prev_nat_succ q)
        ((ComplexShape.up ℕ).next_eq' rfl)).inv =
      (eSh A).liftCycles z (q + 2) ((ComplexShape.up ℕ).next_eq' rfl) hz' ≫ (eSh A).homologyπ (q + 1) := by
    rw [← reassoc_of% h3, Iso.hom_inv_id, Category.comp_id]
  simp only [eHomIso, Iso.trans_hom, Iso.symm_hom, asIso_hom]
  rw [reassoc_of% h1, reassoc_of% h2]
  exact h3'

lemma liftCycles_congr (K : CochainComplex (ModuleCat.{0} k) ℕ) {W : ModuleCat.{0} k} (i j : ℕ)
    (hj : (ComplexShape.up ℕ).next i = j) {x y : W ⟶ K.X i} (hxy : x = y) (hx : x ≫ K.d i j = 0)
    (hy : y ≫ K.d i j = 0) : K.liftCycles x j hj hx = K.liftCycles y j hj hy := by
  subst hxy; rfl

lemma liftCycles_iCycles (K : CochainComplex (ModuleCat.{0} k) ℕ) (q : ℕ) :
    K.liftCycles (K.iCycles q) (q + 1) ((ComplexShape.up ℕ).next_eq' rfl) (K.iCycles_d q (q + 1)) = 𝟙 _ := by
  rw [← cancel_mono (K.iCycles q), HomologicalComplex.liftCycles_i, Category.id_comp]

/-- Two maps out of `H^q(K)` agree when they agree on every cocycle `x : W ⟶ K^q`. -/
lemma homology_ext (K : CochainComplex (ModuleCat.{0} k) ℕ) (q : ℕ) {Y : ModuleCat.{0} k} {a b : K.homology q ⟶ Y}
    (h : ∀ (W : ModuleCat.{0} k) (x : W ⟶ K.X q) (hx : x ≫ K.d q (q + 1) = 0),
      K.liftCycles x (q + 1) ((ComplexShape.up ℕ).next_eq' rfl) hx ≫ K.homologyπ q ≫ a =
        K.liftCycles x (q + 1) ((ComplexShape.up ℕ).next_eq' rfl) hx ≫ K.homologyπ q ≫ b) : a = b := by
  have := h _ (K.iCycles q) (K.iCycles_d q (q + 1))
  rw [liftCycles_iCycles, Category.id_comp, Category.id_comp] at this
  exact (cancel_epi (K.homologyπ q)).1 this

/-- `S ↪ Γ`-restriction to `⟨−I⟩` on the periodic cochains (`x ↦ x`, `x ↦ (1 + S)x`). -/
abbrev cycResS : cS A ⟶ zI.cycCoch (Rep.res fS (resS A)) :=
  zS.cycRes zI fS (Subgroup.inclusion_injective gI_le_gS) 2 fS_t rfl (resS A)

/-- `U`-restriction to `⟨−I⟩` on the periodic cochains (`y ↦ y`, `y ↦ (1 + U + U²)y`). -/
abbrev cycResU : cU A ⟶ zI.cycCoch (Rep.res fU (resU A)) :=
  zU.cycRes zI fU (Subgroup.inclusion_injective gI_le_gU) 3 fU_t rfl (resU A)

lemma jS_phi (q : ℕ) : (jS A).f q ≫ mvφF A q = (cycResS A).f q := by
  apply ModuleCat.hom_ext; refine LinearMap.ext fun (x : A) => ?_
  show phiA A q (x, 0) = ((cycResS A).f q).hom x
  refine Eq.trans ?_ (CycDat.cycRes_f_apply zS zI fS (Subgroup.inclusion_injective gI_le_gS) 2 fS_t rfl
    (resS A) q x).symm
  unfold phiA
  by_cases h : Even q
  · simp only [if_pos h, LinearMap.sub_apply, LinearMap.fst_apply, LinearMap.snd_apply, sub_zero]
  · simp only [if_neg h, LinearMap.sub_apply, LinearMap.comp_apply, LinearMap.fst_apply, LinearMap.snd_apply,
      map_zero, sub_zero, nrmA_apply]
    rfl

lemma jU_phi (q : ℕ) : (jU A).f q ≫ mvφF A q = -(cycResU A).f q := by
  apply ModuleCat.hom_ext; refine LinearMap.ext fun (y : A) => ?_
  show phiA A q (0, y) = -(((cycResU A).f q).hom y)
  refine Eq.trans ?_ (congrArg Neg.neg (CycDat.cycRes_f_apply zU zI fU (Subgroup.inclusion_injective gI_le_gU) 3
    fU_t rfl (resU A) q y)).symm
  unfold phiA
  by_cases h : Even q
  · simp only [if_pos h, LinearMap.sub_apply, LinearMap.fst_apply, LinearMap.snd_apply, zero_sub]
    rfl
  · simp only [if_neg h, LinearMap.sub_apply, LinearMap.comp_apply, LinearMap.fst_apply, LinearMap.snd_apply,
      map_zero, zero_sub, nrmA_apply]
    rfl

/-- The connecting map on a vertex component, in cochain terms: `δ ∘ H(j) = H(r) ∘ eHomIso` for a chain map
`r : c ⟶ cI` with `j ≫ φ* = r` in every degree. -/
lemma mvδ_comp (c : CochainComplex (ModuleCat.{0} k) ℕ) (j : c ⟶ vCoch A) (r : c ⟶ cI A)
    (hr : ∀ q, j.f q ≫ mvφF A q = r.f q) (q : ℕ) :
    HomologicalComplex.homologyMap j q ≫ mvδ A q = HomologicalComplex.homologyMap r q ≫ (eHomIso A q).hom := by
  apply homology_ext c q
  intro W x hx
  have hxj : (x ≫ j.f q) ≫ (vCoch A).d q (q + 1) = 0 := by
    rw [Category.assoc, j.comm, ← Category.assoc, hx, Limits.zero_comp]
  have hxr : (x ≫ r.f q) ≫ (cI A).d q (q + 1) = 0 := by
    rw [Category.assoc, r.comm, ← Category.assoc, hx, Limits.zero_comp]
  have hxr' : (x ≫ r.f q) ≫ (eSh A).d (q + 1) (q + 2) = 0 := by
    have := mvφ_cocycle A q (x ≫ j.f q) hxj
    rw [← hr q]
    exact (congrArg (· ≫ (eSh A).d (q + 1) (q + 2)) (Category.assoc x (j.f q) (mvφF A q)).symm).trans this
  rw [HomologicalComplex.homologyπ_naturality_assoc, HomologicalComplex.homologyπ_naturality_assoc,
    HomologicalComplex.liftCycles_comp_cyclesMap_assoc, HomologicalComplex.liftCycles_comp_cyclesMap_assoc,
    mvδ_eq A q _ hxj, eHomIso_π A q _ hxr hxr']
  congr 1
  exact liftCycles_congr (eSh A) _ _ _ ((Category.assoc _ _ _).trans (congrArg (x ≫ ·) (hr q))) _ _

lemma mvδ_comp_neg (c : CochainComplex (ModuleCat.{0} k) ℕ) (j : c ⟶ vCoch A) (r : c ⟶ cI A)
    (hr : ∀ q, j.f q ≫ mvφF A q = -r.f q) (q : ℕ) :
    HomologicalComplex.homologyMap j q ≫ mvδ A q = -(HomologicalComplex.homologyMap r q ≫ (eHomIso A q).hom) := by
  rw [mvδ_comp A c j (-r) (fun q => (hr q).trans rfl) q, HomologicalComplex.homologyMap_neg, Preadditive.neg_comp]

/-- **The middle map on the `⟨S⟩`-summand is restriction**: `δ ∘ ι_S = res^{⟨S⟩}_{⟨−I⟩}` under `vIso`, `eIso`. -/
theorem mvδ_inl (q : ℕ) : biprod.inl ≫ (vIso A q).inv ≫ mvδ A q ≫ (eIso A q).hom = resSI A q := by
  have hδ := mvδ_comp A (cS A) (jS A) (cycResS A) (jS_phi A) q
  have hres := zS.cohIso_res zI fS (Subgroup.inclusion_injective gI_le_gS) 2 fS_t rfl (resS A) q
  have key := ((Iso.eq_comp_inv _).2 hres).symm
  rw [← Category.assoc, inl_vIso_inv, Category.assoc, reassoc_of% hδ]
  simp only [eIso, Iso.trans_hom, Iso.symm_hom, Iso.hom_inv_id_assoc]
  exact (Category.assoc _ _ _).symm.trans key

/-- **The middle map on the `⟨U⟩`-summand is minus restriction**: `δ ∘ ι_U = −res^{⟨U⟩}_{⟨−I⟩}`. -/
theorem mvδ_inr (q : ℕ) : biprod.inr ≫ (vIso A q).inv ≫ mvδ A q ≫ (eIso A q).hom = -resUI A q := by
  have hδ := mvδ_comp_neg A (cU A) (jU A) (cycResU A) (jU_phi A) q
  have hres := zU.cohIso_res zI fU (Subgroup.inclusion_injective gI_le_gU) 3 fU_t rfl (resU A) q
  have key := ((Iso.eq_comp_inv _).2 hres).symm
  rw [← Category.assoc, inr_vIso_inv, Category.assoc, reassoc_of% hδ]
  simp only [eIso, Iso.trans_hom, Iso.symm_hom, Preadditive.neg_comp, Preadditive.comp_neg, Category.assoc,
    Iso.hom_inv_id_assoc]
  exact congrArg Neg.neg ((Category.assoc _ _ _).symm.trans key)

/-- **`eq:mvamalgam`, the middle map**: under `vIso` (`H^q(vCoch A) ≅ H^q(⟨S⟩; A) ⊕ H^q(⟨U⟩; A)`) and `eIso`
(`H^{q+1}(eSh A) ≅ H^q(⟨−I⟩; A)`), the map `mvδ` (the middle map of `eq:mvamalgam`, phase 3's connecting map of the cone
SES) is `res_S − res_U`, every
`q`, every commutative ring `k`, every `A`. -/
theorem mvδ_resres (q : ℕ) :
    (vIso A q).inv ≫ mvδ A q ≫ (eIso A q).hom = biprod.desc (resSI A q) (-resUI A q) := by
  apply biprod.hom_ext'
  · rw [biprod.inl_desc]; exact mvδ_inl A q
  · rw [biprod.inr_desc]; exact mvδ_inr A q

/-! #### The beginning `0 → H⁰(Γ; A) → A^S ⊕ A^U → A^{−I}` -/

lemma sS_eq_S : sS = ModularGroup.S := by
  ext i j; rw [sS, GL2.TauChk.M22.toSL_coe (by decide)]; fin_cases i <;> fin_cases j <;> simp [GL2.TauChk.M22.toMat, mS, ModularGroup.coe_S]

lemma sU_eq_ST : sU = ModularGroup.S * ModularGroup.T := by
  ext i j; rw [sU, GL2.TauChk.M22.toSL_coe (by decide)]
  fin_cases i <;> fin_cases j <;>
    simp [GL2.TauChk.M22.toMat, mU, ModularGroup.coe_S, ModularGroup.coe_T, Matrix.mul_apply, Fin.sum_univ_two]

/-- `A^S`, `A^U`, `A^{−I}` as submodules of `A` (declared type `Submodule k A`, so that `⊓` and `≤` between them are
formed at `Submodule k A` and the kernel never compares `Rep.res gS.subtype A` with `Rep.res gU.subtype A` argument-wise;
LRY4_audit §5: without these, `invariants_eq_inf` alone cost ≈ 8 GB and ≈ 100 s of kernel time). -/
def invS : Submodule k A := (resS A).ρ.invariants
/-- `A^U` as a submodule of `A`. -/
def invU : Submodule k A := (resU A).ρ.invariants
/-- `A^{−I}` as a submodule of `A`. -/
def invI : Submodule k A := (resI A).ρ.invariants

lemma mem_invS (v : A) : v ∈ invS A ↔ ∀ g : gS, A.ρ g v = v := Iff.rfl
lemma mem_invU (v : A) : v ∈ invU A ↔ ∀ g : gU, A.ρ g v = v := Iff.rfl
lemma mem_invI (v : A) : v ∈ invI A ↔ ∀ g : gI, A.ρ g v = v := Iff.rfl

/-- **`A^S ∩ A^U = A^Γ`** (`S`, `U = ST` generate `SL₂(ℤ)`; mathlib's `SL2Z_generators`). -/
theorem invariants_eq_inf : A.ρ.invariants = invS A ⊓ invU A := by
  ext v
  simp only [Submodule.mem_inf, mem_invS, mem_invU, Representation.mem_invariants]
  refine ⟨fun h => ⟨fun g => h g.1, fun g => h g.1⟩, fun ⟨hS, hU⟩ g => ?_⟩
  have hS' : A.ρ sS v = v := hS ⟨sS, Subgroup.mem_zpowers sS⟩
  have hU' : A.ρ sU v = v := hU ⟨sU, Subgroup.mem_zpowers sU⟩
  have hg : g ∈ Subgroup.closure {ModularGroup.S, ModularGroup.T} := by
    rw [_root_.SpecialLinearGroup.SL2Z_generators]; trivial
  induction hg using Subgroup.closure_induction with
  | mem x hx =>
    rcases hx with rfl | rfl
    · rw [← sS_eq_S]; exact hS'
    · have hT : ModularGroup.T = sS⁻¹ * sU := by rw [sU_eq_ST, sS_eq_S, inv_mul_cancel_left]
      have hSi : A.ρ sS⁻¹ v = v := by
        conv_lhs => rw [← hS']
        rw [← Module.End.mul_apply, ← map_mul, inv_mul_cancel, map_one, Module.End.one_apply]
      rw [hT, map_mul, Module.End.mul_apply, hU', hSi]
  | one => rw [map_one, Module.End.one_apply]
  | mul x y _ _ hx hy => rw [map_mul, Module.End.mul_apply, hy, hx]
  | inv x _ hx =>
    conv_lhs => rw [← hx]
    rw [← Module.End.mul_apply, ← map_mul, inv_mul_cancel, map_one, Module.End.one_apply]

/-- `A^Γ → A^S ⊕ A^U`, `a ↦ (a, a)` (restriction in degree `0`). -/
def lowι : A.ρ.invariants →ₗ[k] invS A × invU A :=
  LinearMap.prod (Submodule.inclusion (fun _ h => ((Submodule.mem_inf.1 ((invariants_eq_inf A).le h)).1)))
    (Submodule.inclusion (fun _ h => ((Submodule.mem_inf.1 ((invariants_eq_inf A).le h)).2)))

/-- The components of `lowι` (by `rfl`; stated once, so that `mv_low_exact` does not make the kernel unfold `lowι`
inside `Prod.ext`, which cost ≈ 40 s and ≈ 3.3 GB). -/
lemma lowι_fst (a : A.ρ.invariants) : ((lowι A a).1 : A) = a := rfl
lemma lowι_snd (a : A.ρ.invariants) : ((lowι A a).2 : A) = a := rfl

lemma invS_le_invI : invS A ≤ invI A := fun _ h g => h (fS g)
lemma invU_le_invI : invU A ≤ invI A := fun _ h g => h (fU g)

/-- `A^S ⊕ A^U → A^{−I}`, `(x, y) ↦ x − y` (`res_S − res_U` in degree `0`). -/
def lowδ : invS A × invU A →ₗ[k] invI A :=
  Submodule.inclusion (invS_le_invI A) ∘ₗ LinearMap.fst _ _ _ - Submodule.inclusion (invU_le_invI A) ∘ₗ LinearMap.snd _ _ _

/-- **`eq:mvamalgam`, the beginning**: `0 → A^Γ → A^S ⊕ A^U → A^{−I}`, `a ↦ (a, a)`, `(x, y) ↦ x − y`, is exact, every
commutative ring `k`, every `A` (`H⁰` of a group is the invariants, mathlib's `groupCohomology.H0Iso`; restriction in
degree `0` is the inclusion of invariants, `resSI_zero`).  Proved directly, not as the degree-`0` case of the sequence. -/
theorem mv_low_exact : Function.Injective (lowι A) ∧ Function.Exact (lowι A) (lowδ A) := by
  refine ⟨fun a b h => Subtype.ext (congrArg (fun p => (p.1 : A)) h), fun p => ?_⟩
  obtain ⟨x, y⟩ := p
  constructor
  · intro h
    have hxy : (x : A) = y := sub_eq_zero.1 (congrArg Subtype.val h)
    have hx : (x : A) ∈ A.ρ.invariants := by
      rw [invariants_eq_inf]; exact Submodule.mem_inf.2 ⟨x.2, hxy ▸ y.2⟩
    exact ⟨⟨x, hx⟩, Prod.ext (Subtype.ext (lowι_fst A _)) (Subtype.ext ((lowι_snd A _).trans hxy))⟩
  · rintro ⟨a, ha⟩
    have h1 : (x : A) = a := (congrArg (fun p => (p.1 : A)) ha).symm
    have h2 : (y : A) = a := (congrArg (fun p => (p.2 : A)) ha).symm
    exact Subtype.ext (show (x : A) - y = 0 by rw [h1, h2, sub_self])

/-- **Restriction `H⁰(⟨S⟩; A) → H⁰(⟨−I⟩; A)` is the inclusion `A^S ⊆ A^{−I}`** (under mathlib's `H0Iso`). -/
theorem resSI_zero : resSI A 0 ≫ (groupCohomology.H0Iso (resI A)).hom ≫ (groupCohomology.shortComplexH0 (resI A)).f =
    (groupCohomology.H0Iso (resS A)).hom ≫ (groupCohomology.shortComplexH0 (resS A)).f :=
  (groupCohomology.map_H0Iso_hom_f fS (𝟙 (Rep.res fS (resS A)))).trans (by rfl)

/-- **Restriction `H⁰(⟨U⟩; A) → H⁰(⟨−I⟩; A)` is the inclusion `A^U ⊆ A^{−I}`**. -/
theorem resUI_zero : resUI A 0 ≫ (groupCohomology.H0Iso (resI A)).hom ≫ (groupCohomology.shortComplexH0 (resI A)).f =
    (groupCohomology.H0Iso (resU A)).hom ≫ (groupCohomology.shortComplexH0 (resU A)).f :=
  (groupCohomology.map_H0Iso_hom_f fU (𝟙 (Rep.res fU (resU A)))).trans (by rfl)

end Gamma

/-! ### Stretch: the periodicity `H^q(Γ; A) ≅ H^{q+2}(Γ; A)` (`q ≥ 2`), natural on `groupCohomology` -/

section PeriodicNat

open scoped MatrixGroups

variable {k : Type} [CommRing k]

/-- `Hom_Γ(P_•, -) = treeCoch -` is natural: post-composition with `f` is `treeCochMap f`. -/
lemma postY_treeCochIso {A B : Rep.{0} k SL(2, ℤ)} (f : A ⟶ B) :
    Cover.postY ((model7PermRes k).cx k) f ≫ (treeCochIso B).hom = (treeCochIso A).hom ≫ treeCochMap f := by
  ext n φ : 3
  rcases n with _ | n <;> rfl

/-- **`H^n(SL₂(ℤ); A) ≅ H^n(treeCoch A)` through the fixed comparison with the bar resolution** (natural in `A`, unlike
`model7TreeCochCoh`, which goes through `Ext`). -/
def model7TreeCochCohNat (A : Rep.{0} k SL(2, ℤ)) (n : ℕ) : groupCohomology A n ≅ (treeCoch A).homology n :=
  (model7PermRes k).barIso A n ≪≫ (HomologicalComplex.homologyFunctor _ _ n).mapIso (treeCochIso A)

/-- **Naturality in `A`** of `model7TreeCochCohNat`. -/
theorem model7TreeCochCohNat_natural {A B : Rep.{0} k SL(2, ℤ)} (f : A ⟶ B) (n : ℕ) :
    (groupCohomology.functor k SL(2, ℤ) n).map f ≫ (model7TreeCochCohNat B n).hom =
      (model7TreeCochCohNat A n).hom ≫ HomologicalComplex.homologyMap (treeCochMap f) n := by
  have key : groupCohomology.cochainsMap (MonoidHom.id _) f ≫ (groupCohomology.inhomogeneousCochainsIso B).hom ≫
      Cover.preY ((model7PermRes k).barEquiv k).hom B ≫ (treeCochIso B).hom =
    (groupCohomology.inhomogeneousCochainsIso A).hom ≫ Cover.preY ((model7PermRes k).barEquiv k).hom A ≫
      (treeCochIso A).hom ≫ treeCochMap f := by
    rw [← Category.assoc, Cover.cochainsMap_inhomogeneousCochainsIso, Category.assoc, ← Category.assoc (Cover.postY _ f),
      Cover.postY_preY, Category.assoc, postY_treeCochIso]
  have := congrArg (fun φ => HomologicalComplex.homologyMap φ n) key
  simp only [HomologicalComplex.homologyMap_comp] at this
  show HomologicalComplex.homologyMap (groupCohomology.cochainsMap (MonoidHom.id _) f) n ≫
      (HomologicalComplex.homologyMap (groupCohomology.inhomogeneousCochainsIso B).hom n ≫
        HomologicalComplex.homologyMap (Cover.preY ((model7PermRes k).barEquiv k).hom B) n) ≫
      HomologicalComplex.homologyMap (treeCochIso B).hom n =
    ((HomologicalComplex.homologyMap (groupCohomology.inhomogeneousCochainsIso A).hom n ≫
        HomologicalComplex.homologyMap (Cover.preY ((model7PermRes k).barEquiv k).hom A) n) ≫
      HomologicalComplex.homologyMap (treeCochIso A).hom n) ≫ HomologicalComplex.homologyMap (treeCochMap f) n
  simp only [Category.assoc]
  exact this

/-- **A periodicity isomorphism `H^q(SL₂(ℤ); A) ≅ H^{q+2}(SL₂(ℤ); A)` for `q ≥ 2`, natural in `A`** (`q = m + 2`), every
commutative ring `k`: the periodicity of the tree resolution, transported through the bar-built `model7TreeCochCohNat`.
Not shown to coincide with `model7CohPeriodic` (the same `treePerNat`, transported through the `Ext`-built
`model7TreeCochCoh`). -/
def model7CohPeriodicNat (A : Rep.{0} k SL(2, ℤ)) (m : ℕ) : groupCohomology A (m + 2) ≅ groupCohomology A (m + 4) :=
  model7TreeCochCohNat A (m + 2) ≪≫ (treePerNat k m).app A ≪≫ (model7TreeCochCohNat A (m + 4)).symm

/-- **Naturality of `model7CohPeriodicNat` on `groupCohomology`.** -/
theorem model7CohPeriodicNat_natural {A B : Rep.{0} k SL(2, ℤ)} (f : A ⟶ B) (m : ℕ) :
    groupCohomology.map (MonoidHom.id _) f (m + 2) ≫ (model7CohPeriodicNat B m).hom =
      (model7CohPeriodicNat A m).hom ≫ groupCohomology.map (MonoidHom.id _) f (m + 4) := by
  have h2 : groupCohomology.map (MonoidHom.id _) f (m + 2) ≫ (model7TreeCochCohNat B (m + 2)).hom =
      (model7TreeCochCohNat A (m + 2)).hom ≫ HomologicalComplex.homologyMap (treeCochMap f) (m + 2) :=
    model7TreeCochCohNat_natural f (m + 2)
  have h4 : groupCohomology.map (MonoidHom.id _) f (m + 4) ≫ (model7TreeCochCohNat B (m + 4)).hom =
      (model7TreeCochCohNat A (m + 4)).hom ≫ HomologicalComplex.homologyMap (treeCochMap f) (m + 4) :=
    model7TreeCochCohNat_natural f (m + 4)
  have hP := treePerNat_natural f m
  have h4' : (model7TreeCochCohNat A (m + 4)).inv ≫ groupCohomology.map (MonoidHom.id _) f (m + 4) =
      HomologicalComplex.homologyMap (treeCochMap f) (m + 4) ≫ (model7TreeCochCohNat B (m + 4)).inv :=
    (Iso.inv_comp_eq _).2 (((Iso.eq_comp_inv _).2 h4).trans (Category.assoc _ _ _))
  show groupCohomology.map (MonoidHom.id _) f (m + 2) ≫ ((model7TreeCochCohNat B (m + 2)).hom ≫
      ((treePerNat k m).app B).hom ≫ (model7TreeCochCohNat B (m + 4)).inv) =
    ((model7TreeCochCohNat A (m + 2)).hom ≫ ((treePerNat k m).app A).hom ≫ (model7TreeCochCohNat A (m + 4)).inv) ≫
      groupCohomology.map (MonoidHom.id _) f (m + 4)
  rw [reassoc_of% h2]
  have e1 := (Category.assoc _ _ _).symm.trans ((congrArg (· ≫ (model7TreeCochCohNat B (m + 4)).inv) hP).trans
    (Category.assoc _ _ _))
  have e2 := congrArg (((treePerNat k m).app A).hom ≫ ·) h4'.symm
  exact (congrArg ((model7TreeCochCohNat A (m + 2)).hom ≫ ·) (e1.trans e2)).trans
    ((Category.assoc _ _ _).trans (congrArg ((model7TreeCochCohNat A (m + 2)).hom ≫ ·)
      (Category.assoc _ _ _))).symm

variable (k) in
/-- **A periodicity natural isomorphism** `H^{m+2}(SL₂(ℤ); -) ≅ H^{m+4}(SL₂(ℤ); -)` of functors
`Rep k SL₂(ℤ) ⥤ ModuleCat k` (components `model7CohPeriodicNat`). -/
def model7CohPeriodicNatIso (m : ℕ) :
    groupCohomology.functor k SL(2, ℤ) (m + 2) ≅ groupCohomology.functor k SL(2, ℤ) (m + 4) :=
  NatIso.ofComponents (fun A => model7CohPeriodicNat A m) (fun f => model7CohPeriodicNat_natural f m)

end PeriodicNat

end GL2.MP
