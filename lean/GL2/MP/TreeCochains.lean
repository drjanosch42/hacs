/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.MP.TreeComparison

/-!
# LR-Y phase 3 (slot 22): the explicit cochain complex of the tree resolution, and its periodicity

[GT] `gt_part_MP.tex`, rows `thm:mp-hom-perm`, `thm:mp-tree-cochains`, `thm:mp-tree-periodic`, `thm:mp-tree-mv`, backing
[C] `cl:amalgam`, `eq:mvamalgam` (the cochain complex `Hom_Γ(P_•, A)` in coordinates; the Mayer–Vietoris sequence with
periodic middle terms) and `lem:bulkperiodic` (2) **only** as the existence of
a natural isomorphism `H^q ≅ H^{q+2}`, `q ≥ 2` (not that it is `e ∪ −`, not the `q = 1` clause; `LRY_audit.md` MINOR-4).

* `permHom`: `Hom_G(k[X], A) ≅ {F : X → A equivariant}` for any `G`-set `X` (the analogue of `homE`).
* **`homFB`**: `Hom_G(k[FB r G], A) ≅ A^r`, `φ ↦ (φ(e_c))_c` (evaluation at the generators `(c, 1)`).
* **`treeCoch A`**: the explicit cochain complex `A² → A³ → A³ → ⋯` (`Γ = SL₂(ℤ)`):
  `d⁰(x, y) = ((1 − S)x, (1 − U)y, x − y)` (`treeD0`), and `d^{m+1}((x, y), z) = (v_{m+1}(x, y), φ*_{m+1}(x, y) − e_m z)`
  (`treeDS`) with `v_j = (1 − S, 1 − U)`, `φ*_j = x − y` for even `j`, `v_j = (N_S, N_U)` (`N_t = ∑_{i<N} t^i`, `N = 4, 6`),
  `φ*_j = (1 + S)x − (1 + U + U²)y` for odd `j`, and `e_m = 1 − (−I)` (even `m`), `1 + (−I)` (odd `m`).
* **`treeCochIso A`**: `Hom_Γ(P_•, A) ≅ treeCoch A` as cochain complexes of `k`-modules; hence
  **`model7TreeCochCoh`**: `H^n(SL₂(ℤ); A) ≅ H^n(treeCoch A)`, every `n`, every commutative ring `k`, `−I` included.
* **Periodicity**: `model7PBd_periodic` (`∂_{n+3} = ∂_{n+1}` on bases, Lean's argument index: the terms are 2-periodic from degree 1 and the differentials from degree 2), `treeDS_periodic`; `treeCochFunctor`, and
  **`treePerNat`**: `H^{m+2}(treeCoch A) ≅ H^{m+4}(treeCoch A)`, a natural isomorphism of functors in `A`;
  **`model7CohPeriodic`**: `H^q(SL₂(ℤ); A) ≅ H^{q+2}(SL₂(ℤ); A)` for `q ≥ 2`.
* **Mayer–Vietoris** (`eq:mvamalgam`): `0 → eSh A → treeCoch A → vCoch A → 0` is short exact (`mvSC_shortExact`), with
  `vCoch A` the periodic cochain complex of `⟨S⟩, ⟨U⟩` (`A² → A² → ⋯`) and `eSh A` that of `⟨−I⟩` shifted by one (differential
  negated); hence the long exact sequence `⋯ → H^q(eSh) → H^q(Γ; A) → H^q(vCoch) →(δ) H^{q+1}(eSh) → H^{q+1}(Γ; A) → ⋯`
  (`mv_exact_coh`, `mv_exact_vtx`, `mv_exact_edge`), and **`mvδ_eq`**: the connecting map is `φ*` on cocycles.  The identification of the
  middle terms with `groupCohomology` of the cyclic subgroups and of `δ` with `res − res` is `GL2.MP.TreeCyclic`.
-/

noncomputable section

namespace GL2.MP

open CategoryTheory Finsupp

/-! ### `Hom_G(k[X], A)` as equivariant functions -/

section Perm

variable {G : Type} [Group G] {k : Type} [CommRing k] (A : Rep.{0} k G)

/-- **Equivariant functions** `X → A`. -/
def EqF (X : Type) [MulAction G X] : Submodule k (X → A) where
  carrier := {F | ∀ (g : G) x, F (g • x) = A.ρ g (F x)}
  add_mem' {F F'} hF hF' g x := by simp only [Pi.add_apply, hF g x, hF' g x, map_add]
  zero_mem' g x := by simp
  smul_mem' r F hF g x := by simp only [Pi.smul_apply, hF g x, map_smul]

variable {A} in
lemma lcEqF_mapDomain {X : Type} [MulAction G X] (F : EqF A X) (g : G) (z : X →₀ k) :
    Finsupp.linearCombination k F.1 (Finsupp.mapDomain (g • ·) z) = A.ρ g (Finsupp.linearCombination k F.1 z) := by
  induction z using Finsupp.induction_linear with
  | zero => simp
  | add x y hx hy => rw [Finsupp.mapDomain_add, map_add, hx, hy, map_add, map_add]
  | single x c =>
    rw [Finsupp.mapDomain_single, Finsupp.linearCombination_single, Finsupp.linearCombination_single, map_smul, F.2]

/-- The lift `k[X] → A` of an equivariant function. -/
def liftEqF {X : Type} [MulAction G X] (F : EqF A X) : Rep.ofMulAction k G X ⟶ A :=
  Rep.ofHom ⟨Finsupp.linearCombination k F.1 ∘ₗ (MonoidAlgebra.coeffLinearEquiv k).toLinearMap, fun g => by
    refine LinearMap.ext fun x => ?_
    simp only [LinearMap.coe_comp, Function.comp_apply, LinearEquiv.coe_coe, Representation.ofMulAction_def,
      LinearEquiv.apply_symm_apply, Finsupp.lmapDomain_apply]
    exact lcEqF_mapDomain F g _⟩

/-- **`Hom_G(k[X], A) ≅ {F : X → A equivariant}`**, `φ ↦ (x ↦ φ(x))`. -/
def permHom (X : Type) [MulAction G X] : (Rep.ofMulAction k G X ⟶ A) ≃ₗ[k] EqF A X where
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
  invFun := liftEqF A
  left_inv φ := by
    apply Rep.hom_ext; apply Representation.IntertwiningMap.ext; refine LinearMap.ext fun x => ?_
    show Finsupp.linearCombination k (fun y => φ.hom (MonoidAlgebra.single y 1)) (MonoidAlgebra.coeffLinearEquiv k x) =
      φ.hom x
    induction x using MonoidAlgebra.induction_linear with
    | zero => simp
    | add x y hx hy => rw [map_add, map_add, hx, hy, map_add]
    | single m r =>
      rw [Cover.coeff_single, Finsupp.linearCombination_single, ← map_smul]
      congr 1
      simp
  right_inv F := Subtype.ext (funext fun x => by
    show Finsupp.linearCombination k F.1 (MonoidAlgebra.coeffLinearEquiv k (MonoidAlgebra.single x 1)) = F.1 x
    simp)

lemma permHom_apply {X : Type} [MulAction G X] (φ : Rep.ofMulAction k G X ⟶ A) (x : X) :
    (permHom A X φ).1 x = φ.hom (MonoidAlgebra.single x 1) := rfl

/-- `φ` on a finitely supported combination of basis vectors. -/
lemma permHom_lc {X : Type} [MulAction G X] (φ : Rep.ofMulAction k G X ⟶ A) (z : X →₀ k) :
    φ.hom ((MonoidAlgebra.coeffLinearEquiv k).symm z) = Finsupp.linearCombination k (permHom A X φ).1 z := by
  conv_lhs => rw [← (permHom A X).symm_apply_apply φ]
  show Finsupp.linearCombination k (permHom A X φ).1
    (MonoidAlgebra.coeffLinearEquiv k ((MonoidAlgebra.coeffLinearEquiv k).symm z)) = _
  rw [LinearEquiv.apply_symm_apply]

/-- **An equivariant function on `FB r G` is determined by its values on the generators** `(c, 1)`. -/
lemma eqF_FB {r : ℕ} (F : EqF A (FB r G)) (c : Fin r) (g : G) : F.1 ⟨c, g⟩ = A.ρ g (F.1 ⟨c, 1⟩) := by
  rw [← F.2 g ⟨c, 1⟩]
  show F.1 ⟨c, g⟩ = F.1 ⟨c, g * 1⟩
  rw [mul_one]

/-- **Equivariant functions on `FB r G` are `A^r`.** -/
def eqFB (r : ℕ) : EqF A (FB r G) ≃ₗ[k] (Fin r → A) where
  toFun F c := F.1 ⟨c, 1⟩
  map_add' _ _ := rfl
  map_smul' _ _ := rfl
  invFun v := ⟨fun b => A.ρ b.g (v b.c), fun g b => by
    show A.ρ (g * b.g) (v b.c) = A.ρ g (A.ρ b.g (v b.c))
    rw [map_mul, Module.End.mul_apply]⟩
  left_inv F := Subtype.ext (funext fun b => (eqF_FB A F b.c b.g).symm)
  right_inv v := funext fun c => by
    show A.ρ 1 (v c) = v c
    rw [map_one, Module.End.one_apply]

/-- **`Hom_G(k[G]^r, A) ≅ A^r`** (`φ ↦ (φ(e_c))_c`), over every commutative ring `k`. -/
def homFB (r : ℕ) : (Rep.ofMulAction k G (FB r G) ⟶ A) ≃ₗ[k] (Fin r → A) := (permHom A _).trans (eqFB A r)

lemma homFB_apply {r : ℕ} (φ : Rep.ofMulAction k G (FB r G) ⟶ A) (c : Fin r) :
    homFB A r φ c = φ.hom (MonoidAlgebra.single ⟨c, 1⟩ 1) := rfl

/-- Equivariant functions on `X ⊕ Y` are pairs. -/
def eqFSum (X Y : Type) [MulAction G X] [MulAction G Y] : EqF A (X ⊕ Y) ≃ₗ[k] EqF A X × EqF A Y where
  toFun F := (⟨F.1 ∘ Sum.inl, fun g x => F.2 g (Sum.inl x)⟩, ⟨F.1 ∘ Sum.inr, fun g y => F.2 g (Sum.inr y)⟩)
  map_add' _ _ := rfl
  map_smul' _ _ := rfl
  invFun F := ⟨Sum.elim F.1.1 F.2.1, by
    rintro g (x | y)
    · exact F.1.2 g x
    · exact F.2.2 g y⟩
  left_inv F := Subtype.ext (funext fun z => by rcases z with x | y <;> rfl)
  right_inv _ := rfl

end Perm

/-! ### Equivariant functions on the bases of a periodic resolution -/

section PerEval

variable {G : Type} [Group G] {k : Type} [CommRing k] (A : Rep.{0} k G) {X : Type} [MulAction G X] {r : ℕ}

/-- `F ∘ (1 − t)` at the generator `(c, 1)`: `(1 − t_c) F(c, 1)`. -/
lemma lc_eqF_difB (D : PerData G X r) (F : EqF A (FB r G)) (c : Fin r) :
    Finsupp.linearCombination k F.1 (D.difB k ⟨c, 1⟩) = F.1 ⟨c, 1⟩ - A.ρ (D.t c) (F.1 ⟨c, 1⟩) := by
  simp only [PerData.difB, map_sub, linearCombination_single, one_smul]
  rw [eqF_FB A F c (1 * D.t c), one_mul]

/-- `F ∘ N` at the generator `(c, 1)`: `∑_{j < N_c} t_c^j F(c, 1)`. -/
lemma lc_eqF_nrmB (D : PerData G X r) (F : EqF A (FB r G)) (c : Fin r) :
    Finsupp.linearCombination k F.1 (D.nrmB k ⟨c, 1⟩) = ∑ j ∈ Finset.range (D.N c), A.ρ (D.t c ^ j) (F.1 ⟨c, 1⟩) := by
  simp only [PerData.nrmB, map_sum, linearCombination_single, one_smul]
  exact Finset.sum_congr rfl fun j _ => by rw [eqF_FB A F c (1 * D.t c ^ j), one_mul]

/-- **Precomposition with a boundary of a `PermRes`** on basis vectors. -/
lemma permHom_D (P : PermRes G k) (n : ℕ) (φ : P.X k n ⟶ A) (b : P.B (n + 1)) :
    (permHom A (P.B (n + 1)) (P.D n ≫ φ)).1 b = Finsupp.linearCombination k (permHom A (P.B n) φ).1 (P.bd n b) := by
  show φ.hom ((P.D n).hom (MonoidAlgebra.single b 1)) = _
  rw [PermRes.D_hom_apply, Cover.coeff_single, permHom_lc]
  congr 1
  show Finsupp.linearCombination k (P.bd n) (Finsupp.single b 1) = _
  rw [linearCombination_single, one_smul]

end PerEval

/-! ### The explicit cochain complex `A² → A³ → A³ → ⋯` of `SL₂(ℤ)` -/

open scoped MatrixGroups

section Tree

variable {k : Type} [CommRing k] (A : Rep.{0} k SL(2, ℤ))

/-- `N_t = ∑_{j < N} t^j` on `A`. -/
def nrmA (t : SL(2, ℤ)) (N : ℕ) : A →ₗ[k] A := ∑ j ∈ Finset.range N, A.ρ (t ^ j)

/-- `1 − t` on `A`. -/
def difA (t : SL(2, ℤ)) : A →ₗ[k] A := LinearMap.id - A.ρ t

lemma nrmA_apply (t : SL(2, ℤ)) (N : ℕ) (x : A) : nrmA A t N x = ∑ j ∈ Finset.range N, A.ρ (t ^ j) x := by
  simp [nrmA]

lemma difA_apply (t : SL(2, ℤ)) (x : A) : difA A t x = x - A.ρ t x := rfl

/-- The vertex block in degree `m`: `(1 − S, 1 − U)` for even `m`, `(N_S, N_U)` (`N = 4, 6`) for odd `m`. -/
def vtxA (m : ℕ) : A × A →ₗ[k] A × A :=
  if Even m then (difA A sS).prodMap (difA A sU) else (nrmA A sS 4).prodMap (nrmA A sU 6)

/-- The dual of the lift `φ` in degree `m`: `(x, y) ↦ x − y` for even `m`, `(1 + S)x − (1 + U + U²)y` for odd `m`. -/
def phiA (m : ℕ) : A × A →ₗ[k] A :=
  if Even m then LinearMap.fst k A A - LinearMap.snd k A A
  else nrmA A sS 2 ∘ₗ LinearMap.fst k A A - nrmA A sU 3 ∘ₗ LinearMap.snd k A A

/-- The edge block in degree `m`: `1 − (−I)` for even `m`, `N_{−I} = 1 + (−I)` for odd `m`. -/
def edgA (m : ℕ) : A →ₗ[k] A := if Even m then difA A (-1) else nrmA A (-1) 2

/-- **`d⁰ : A² → A³`**, `(x, y) ↦ ((1 − S)x, (1 − U)y, x − y)`. -/
def treeD0 : A × A →ₗ[k] (A × A) × A := (vtxA A 0).prod (phiA A 0)

/-- **`d^{m+1} : A³ → A³`**, `((x, y), z) ↦ (vtx_{m+1}(x, y), φ*_{m+1}(x, y) − d_E^m z)`. -/
def treeDS (m : ℕ) : (A × A) × A →ₗ[k] (A × A) × A :=
  (vtxA A (m + 1) ∘ₗ LinearMap.fst k (A × A) A).prod
    (phiA A (m + 1) ∘ₗ LinearMap.fst k (A × A) A - edgA A m ∘ₗ LinearMap.snd k (A × A) A)

/-- The terms: `A²` in degree `0`, `A³` above. -/
def tcX : ℕ → ModuleCat k
  | 0 => ModuleCat.of k (A × A)
  | _ + 1 => ModuleCat.of k ((A × A) × A)

/-- The differentials. -/
def tcD : ∀ n, tcX A n ⟶ tcX A (n + 1)
  | 0 => ModuleCat.ofHom (treeD0 A)
  | m + 1 => ModuleCat.ofHom (treeDS A m)

/-! #### The evaluation at the generators -/

/-- **`Hom_Γ(P_n, A) ≅ A²` or `A³`**: `φ ↦ (φ(e_i), φ(e_ρ))` in degree `0`, `((φ(e_i), φ(e_ρ)), φ(e_E))` above. -/
def treeEv : ∀ n, (((model7PermRes k).X k n) ⟶ A) ≃ₗ[k] (tcX A n)
  | 0 => (homFB A 2).trans (LinearEquiv.finTwoArrow k A)
  | _ + 1 => (permHom A _).trans ((eqFSum A _ _).trans (LinearEquiv.prodCongr
      ((eqFB A 2).trans (LinearEquiv.finTwoArrow k A)) ((eqFB A 1).trans (LinearEquiv.funUnique (Fin 1) k A))))

lemma treeEv_zero (φ : ((model7PermRes k).X k 0) ⟶ A) :
    treeEv A 0 φ = (φ.hom (MonoidAlgebra.single (⟨0, 1⟩ : FB 2 SL(2, ℤ)) 1),
      φ.hom (MonoidAlgebra.single (⟨1, 1⟩ : FB 2 SL(2, ℤ)) 1)) := rfl

lemma treeEv_succ (m : ℕ) (φ : ((model7PermRes k).X k (m + 1)) ⟶ A) :
    treeEv A (m + 1) φ =
      ((φ.hom (MonoidAlgebra.single (Sum.inl ⟨0, 1⟩ : FB 2 SL(2, ℤ) ⊕ FB 1 SL(2, ℤ)) 1),
        φ.hom (MonoidAlgebra.single (Sum.inl ⟨1, 1⟩ : FB 2 SL(2, ℤ) ⊕ FB 1 SL(2, ℤ)) 1)),
        φ.hom (MonoidAlgebra.single (Sum.inr ⟨0, 1⟩ : FB 2 SL(2, ℤ) ⊕ FB 1 SL(2, ℤ)) 1)) := rfl

/-- The vertex block on equivariant functions. -/
lemma lc_eqF_vtx (n : ℕ) (F : EqF A (FB 2 SL(2, ℤ))) :
    Finsupp.linearCombination k F.1 (model7PerV.bdS k n ⟨0, 1⟩) = (vtxA A n (F.1 ⟨0, 1⟩, F.1 ⟨1, 1⟩)).1 ∧
      Finsupp.linearCombination k F.1 (model7PerV.bdS k n ⟨1, 1⟩) = (vtxA A n (F.1 ⟨0, 1⟩, F.1 ⟨1, 1⟩)).2 := by
  unfold vtxA
  by_cases h : Even n
  · rw [model7PerV.bdS_even k h, lc_eqF_difB, lc_eqF_difB, if_pos h]; exact ⟨rfl, rfl⟩
  · rw [model7PerV.bdS_odd k h, lc_eqF_nrmB, lc_eqF_nrmB, if_neg h]
    exact ⟨(nrmA_apply A sS 4 _).symm, (nrmA_apply A sU 6 _).symm⟩

/-- The dual of `φ` on equivariant functions. -/
lemma lc_eqF_phiB (n : ℕ) (F : EqF A (FB 2 SL(2, ℤ))) :
    Finsupp.linearCombination k F.1 (model7PhiB k n ⟨0, 1⟩) = phiA A n (F.1 ⟨0, 1⟩, F.1 ⟨1, 1⟩) := by
  unfold model7PhiB phiA
  by_cases h : Even n
  · rw [if_pos h, if_pos h, map_sub, linearCombination_single, linearCombination_single, one_smul, one_smul]; rfl
  · rw [if_neg h, if_neg h]
    simp only [map_add, map_sub, linearCombination_single, one_smul, LinearMap.sub_apply, LinearMap.comp_apply,
      LinearMap.fst_apply, LinearMap.snd_apply, nrmA_apply, Finset.sum_range_succ, Finset.sum_range_zero, zero_add,
      pow_zero, pow_one, map_one, Module.End.one_apply]
    rw [eqF_FB A F 0 (1 * sS), eqF_FB A F 1 (1 * sU), eqF_FB A F 1 (1 * sU ^ 2), one_mul, one_mul, one_mul]

/-- The edge block on equivariant functions. -/
lemma lc_eqF_edg (m : ℕ) (F : EqF A (FB 1 SL(2, ℤ))) :
    Finsupp.linearCombination k F.1 (model7PerE.bdS k m ⟨0, 1⟩) = edgA A m (F.1 ⟨0, 1⟩) := by
  unfold edgA
  by_cases h : Even m
  · rw [model7PerE.bdS_even k h, lc_eqF_difB, if_pos h]; rfl
  · rw [model7PerE.bdS_odd k h, lc_eqF_nrmB, if_neg h, nrmA_apply]; rfl

/-- **The evaluation turns precomposition with `d` into the explicit differential.** -/
lemma treeEv_d (n : ℕ) (φ : ((model7PermRes k).X k n) ⟶ A) :
    treeEv A (n + 1) ((model7PermRes k).D n ≫ φ) = (tcD A n).hom (treeEv A n φ) := by
  have h1 := permHom_D A (model7PermRes k) n φ
  rcases n with _ | m
  · set F : EqF A (FB 2 SL(2, ℤ)) := permHom A (FB 2 SL(2, ℤ)) φ with hF
    have e : ∀ b, ((model7PermRes k).D 0 ≫ φ).hom (MonoidAlgebra.single b 1) =
        Finsupp.linearCombination k F.1 ((model7PermRes k).bd 0 b) := h1
    rw [treeEv_succ, treeEv_zero]
    refine Prod.ext (Prod.ext ((e (Sum.inl ⟨0, 1⟩)).trans ?_) ((e (Sum.inl ⟨1, 1⟩)).trans ?_)) ((e (Sum.inr ⟨0, 1⟩)).trans ?_)
    · show Finsupp.linearCombination k F.1 (model7PerV.bdS k 0 ⟨0, 1⟩) = F.1 ⟨0, 1⟩ - A.ρ sS (F.1 ⟨0, 1⟩)
      rw [model7PerV.bdS_even k PerData.even0, lc_eqF_difB]; rfl
    · show Finsupp.linearCombination k F.1 (model7PerV.bdS k 0 ⟨1, 1⟩) = F.1 ⟨1, 1⟩ - A.ρ sU (F.1 ⟨1, 1⟩)
      rw [model7PerV.bdS_even k PerData.even0, lc_eqF_difB]; rfl
    · show Finsupp.linearCombination k F.1 (model7PhiB k 0 ⟨0, 1⟩) = F.1 ⟨0, 1⟩ - F.1 ⟨1, 1⟩
      rw [model7PhiB, if_pos PerData.even0, map_sub, linearCombination_single, linearCombination_single, one_smul,
        one_smul]
  · set F : EqF A (FB 2 SL(2, ℤ) ⊕ FB 1 SL(2, ℤ)) := permHom A (FB 2 SL(2, ℤ) ⊕ FB 1 SL(2, ℤ)) φ with hF
    have e : ∀ b, ((model7PermRes k).D (m + 1) ≫ φ).hom (MonoidAlgebra.single b 1) =
        Finsupp.linearCombination k F.1 ((model7PermRes k).bd (m + 1) b) := h1
    set FV := (eqFSum A _ _ F).1 with hFV
    set FE := (eqFSum A _ _ F).2 with hFE
    have lV : ∀ v : FB 2 SL(2, ℤ) →₀ k, Finsupp.linearCombination k F.1 (Finsupp.mapDomain Sum.inl v) =
        Finsupp.linearCombination k FV.1 v := fun v => linearCombination_mapDomain _ _ _
    have lE : ∀ v : FB 1 SL(2, ℤ) →₀ k, Finsupp.linearCombination k F.1 (Finsupp.mapDomain Sum.inr v) =
        Finsupp.linearCombination k FE.1 v := fun v => linearCombination_mapDomain _ _ _
    rw [treeEv_succ, treeEv_succ]
    refine Prod.ext (Prod.ext ((e (Sum.inl ⟨0, 1⟩)).trans ?_) ((e (Sum.inl ⟨1, 1⟩)).trans ?_)) ((e (Sum.inr ⟨0, 1⟩)).trans ?_)
    · show Finsupp.linearCombination k F.1 (Finsupp.mapDomain Sum.inl (model7PerV.bdS k (m + 1) ⟨0, 1⟩)) =
        (vtxA A (m + 1) (FV.1 ⟨0, 1⟩, FV.1 ⟨1, 1⟩)).1
      rw [lV]; exact (lc_eqF_vtx A (m + 1) FV).1
    · show Finsupp.linearCombination k F.1 (Finsupp.mapDomain Sum.inl (model7PerV.bdS k (m + 1) ⟨1, 1⟩)) =
        (vtxA A (m + 1) (FV.1 ⟨0, 1⟩, FV.1 ⟨1, 1⟩)).2
      rw [lV]; exact (lc_eqF_vtx A (m + 1) FV).2
    · show Finsupp.linearCombination k F.1 (Finsupp.mapDomain Sum.inl (model7PhiB k (m + 1) ⟨0, 1⟩) -
          Finsupp.mapDomain Sum.inr (model7PerE.bdS k m ⟨0, 1⟩)) =
        phiA A (m + 1) (FV.1 ⟨0, 1⟩, FV.1 ⟨1, 1⟩) - edgA A m (FE.1 ⟨0, 1⟩)
      rw [map_sub, lV, lE, lc_eqF_phiB, lc_eqF_edg]

lemma tcD_sq (n : ℕ) : tcD A n ≫ tcD A (n + 1) = 0 := by
  apply ModuleCat.hom_ext; refine LinearMap.ext fun v => ?_
  obtain ⟨φ, rfl⟩ := (treeEv A n).surjective v
  show (tcD A (n + 1)).hom ((tcD A n).hom (treeEv A n φ)) = 0
  rw [← treeEv_d, ← treeEv_d, ← Category.assoc, PermRes.D_D, Limits.zero_comp, map_zero]

/-- **The explicit cochain complex** `A² → A³ → A³ → ⋯` of the tree resolution. -/
def treeCoch : CochainComplex (ModuleCat k) ℕ := CochainComplex.of (tcX A) (tcD A) (tcD_sq A)

lemma treeCoch_d (n : ℕ) : (treeCoch A).d n (n + 1) = tcD A n := CochainComplex.of_d _ _ _

/-- **`Hom_Γ(P_•, A) ≅ treeCoch A`** as cochain complexes (evaluation at the generators). -/
def treeCochIso : ((model7PermRes k).cx k).linearYonedaObj k A ≅ treeCoch A :=
  HomologicalComplex.Hom.isoOfComponents (fun n => (treeEv A n).toModuleIso) (by
    rintro i j (rfl : i + 1 = j)
    rw [ChainComplex.linearYonedaObj_d, treeCoch_d]
    apply ModuleCat.hom_ext; refine LinearMap.ext fun φ => ?_
    show (tcD A i).hom (treeEv A i φ) = treeEv A (i + 1) (((model7PermRes k).cx k).d (i + 1) i ≫ φ)
    rw [PermRes.cx_d]
    exact (treeEv_d A i φ).symm)

/-- **`H^n(SL₂(ℤ); A) ≅ H^n(A² → A³ → A³ → ⋯)`**, every `n`, every commutative ring `k`, every `A`, `−I` included. -/
def model7TreeCochCoh (n : ℕ) : groupCohomology A n ≅ (treeCoch A).homology n :=
  model7TreeCoh A n ≪≫ (HomologicalComplex.homologyFunctor (ModuleCat k) (ComplexShape.up ℕ) n).mapIso (treeCochIso A)

end Tree

/-! ### Functoriality in `A` and the periodicity `H^q ≅ H^{q+2}` (`q ≥ 2`) -/

section Periodic

variable {k : Type} [CommRing k]

lemma difA_nat {A B : Rep.{0} k SL(2, ℤ)} (f : A ⟶ B) (t : SL(2, ℤ)) (x : A) :
    f.hom (difA A t x) = difA B t (f.hom x) := by
  rw [difA_apply, difA_apply, map_sub, Rep.hom_comm_apply]

lemma nrmA_nat {A B : Rep.{0} k SL(2, ℤ)} (f : A ⟶ B) (t : SL(2, ℤ)) (N : ℕ) (x : A) :
    f.hom (nrmA A t N x) = nrmA B t N (f.hom x) := by
  rw [nrmA_apply, nrmA_apply, map_sum]
  exact Finset.sum_congr rfl fun j _ => Rep.hom_comm_apply f _ x

lemma vtxA_nat {A B : Rep.{0} k SL(2, ℤ)} (f : A ⟶ B) (n : ℕ) (x y : A) :
    vtxA B n (f.hom x, f.hom y) = (f.hom (vtxA A n (x, y)).1, f.hom (vtxA A n (x, y)).2) := by
  unfold vtxA
  split_ifs
  · simp [difA_nat]
  · simp [nrmA_nat]

lemma phiA_nat {A B : Rep.{0} k SL(2, ℤ)} (f : A ⟶ B) (n : ℕ) (x y : A) :
    phiA B n (f.hom x, f.hom y) = f.hom (phiA A n (x, y)) := by
  unfold phiA
  split_ifs
  · simp
  · simp [nrmA_nat]

lemma edgA_nat {A B : Rep.{0} k SL(2, ℤ)} (f : A ⟶ B) (n : ℕ) (x : A) : edgA B n (f.hom x) = f.hom (edgA A n x) := by
  unfold edgA
  split_ifs
  · rw [difA_nat]
  · rw [nrmA_nat]

/-- `f : A ⟶ B` on the terms: `f × f`, `(f × f) × f`. -/
def tcMap {A B : Rep.{0} k SL(2, ℤ)} (f : A ⟶ B) : ∀ n, tcX A n ⟶ tcX B n
  | 0 => ModuleCat.ofHom (f.hom.toLinearMap.prodMap f.hom.toLinearMap)
  | _ + 1 => ModuleCat.ofHom ((f.hom.toLinearMap.prodMap f.hom.toLinearMap).prodMap f.hom.toLinearMap)

lemma tcMap_comm {A B : Rep.{0} k SL(2, ℤ)} (f : A ⟶ B) (n : ℕ) :
    tcMap f n ≫ tcD B n = tcD A n ≫ tcMap f (n + 1) := by
  apply ModuleCat.hom_ext; refine LinearMap.ext fun v => ?_
  rcases n with _ | m
  · obtain ⟨x, y⟩ := v
    show treeD0 B (f.hom x, f.hom y) = ((f.hom (treeD0 A (x, y)).1.1, f.hom (treeD0 A (x, y)).1.2), f.hom (treeD0 A (x, y)).2)
    exact Prod.ext (vtxA_nat f 0 x y) (phiA_nat f 0 x y)
  · obtain ⟨⟨x, y⟩, z⟩ := v
    show treeDS B m ((f.hom x, f.hom y), f.hom z) =
      ((f.hom (treeDS A m ((x, y), z)).1.1, f.hom (treeDS A m ((x, y), z)).1.2), f.hom (treeDS A m ((x, y), z)).2)
    refine Prod.ext (vtxA_nat f (m + 1) x y) ?_
    show phiA B (m + 1) (f.hom x, f.hom y) - edgA B m (f.hom z) = f.hom (phiA A (m + 1) (x, y) - edgA A m z)
    rw [phiA_nat, edgA_nat, map_sub]

/-- `f : A ⟶ B` on the explicit cochain complexes. -/
def treeCochMap {A B : Rep.{0} k SL(2, ℤ)} (f : A ⟶ B) : treeCoch A ⟶ treeCoch B :=
  CochainComplex.ofHom (X := treeCoch A) (Y := treeCoch B) (tcMap f) (fun i => by
    rw [treeCoch_d, treeCoch_d]; exact tcMap_comm f i)

variable (k) in
/-- **`A ↦ treeCoch A`, a functor** `Rep k Γ ⥤ CochainComplex (ModuleCat k) ℕ`. -/
def treeCochFunctor : Rep.{0} k SL(2, ℤ) ⥤ CochainComplex (ModuleCat k) ℕ where
  obj := treeCoch
  map := treeCochMap
  map_id A := by
    ext n : 2
    rcases n with _ | n <;> rfl
  map_comp f g := by
    ext n : 2
    rcases n with _ | n <;> rfl

/-- **The differentials are 2-periodic from degree 1**: `d^{m+3} = d^{m+1}` (`A³ → A³`). -/
lemma treeDS_periodic (A : Rep.{0} k SL(2, ℤ)) (m : ℕ) : treeDS A (m + 2) = treeDS A m := by
  have h2 : ∀ j, Even (j + 2) ↔ Even j := fun j => by simp [Nat.even_add]
  unfold treeDS vtxA phiA edgA
  rw [show m + 2 + 1 = m + 1 + 2 by omega]
  simp only [h2]

/-- **The tree resolution is 2-periodic: its terms from degree 1, its differentials from degree 2**: `P_{n+2} = P_n` for
`n ≥ 1`, `∂_{n+2} = ∂_n` for `n ≥ 2`; in Lean's argument index `∂_{n+3} = ∂_{n+1}` on bases (`k[Γ]³ → k[Γ]³`). -/
lemma model7PBd_periodic (n : ℕ) : model7PBd k (n + 3) = model7PBd k (n + 1) := by
  have h2 : ∀ j, Even (j + 2) ↔ Even j := fun j => by simp [Nat.even_add]
  have hV : model7PerV.bdS k (n + 3) = model7PerV.bdS k (n + 1) := by
    unfold PerData.bdS; simp only [show n + 3 = n + 1 + 2 by omega, h2]
  have hE : model7PerE.bdS k (n + 2) = model7PerE.bdS k n := by unfold PerData.bdS; simp only [h2]
  have hφ : model7PhiB k (n + 3) = model7PhiB k (n + 1) := by
    unfold model7PhiB; simp only [show n + 3 = n + 1 + 2 by omega, h2]
  funext b
  rcases b with b | a
  · show Finsupp.mapDomain Sum.inl (model7PerV.bdS k (n + 3) b) = Finsupp.mapDomain Sum.inl (model7PerV.bdS k (n + 1) b)
    rw [hV]
  · show Finsupp.mapDomain Sum.inl (model7PhiB k (n + 3) a) - Finsupp.mapDomain Sum.inr (model7PerE.bdS k (n + 2) a) =
      Finsupp.mapDomain Sum.inl (model7PhiB k (n + 1) a) - Finsupp.mapDomain Sum.inr (model7PerE.bdS k n a)
    rw [hφ, hE]

variable (k) in
/-- The short complexes at `m + 2` and `m + 4` agree (identity components), naturally in `A`. -/
def treeScPer (m : ℕ) :
    treeCochFunctor k ⋙ HomologicalComplex.shortComplexFunctor' (ModuleCat k) (ComplexShape.up ℕ) (m + 1) (m + 2) (m + 3) ≅
      treeCochFunctor k ⋙
        HomologicalComplex.shortComplexFunctor' (ModuleCat k) (ComplexShape.up ℕ) (m + 3) (m + 4) (m + 5) :=
  NatIso.ofComponents
    (fun A => ShortComplex.isoMk (Iso.refl _) (Iso.refl _) (Iso.refl _)
      (by
        show 𝟙 _ ≫ (treeCoch A).d (m + 3) (m + 4) = (treeCoch A).d (m + 1) (m + 2) ≫ 𝟙 _
        rw [Category.comp_id, Category.id_comp, treeCoch_d, treeCoch_d]
        show ModuleCat.ofHom (treeDS A (m + 2)) = ModuleCat.ofHom (treeDS A m)
        rw [treeDS_periodic])
      (by
        show 𝟙 _ ≫ (treeCoch A).d (m + 4) (m + 5) = (treeCoch A).d (m + 2) (m + 3) ≫ 𝟙 _
        rw [Category.comp_id, Category.id_comp, treeCoch_d, treeCoch_d]
        show ModuleCat.ofHom (treeDS A (m + 1 + 2)) = ModuleCat.ofHom (treeDS A (m + 1))
        rw [treeDS_periodic]))
    (fun f => by ext <;> rfl)

variable (k) in
/-- **The periodicity isomorphism `H^{q}(treeCoch A) ≅ H^{q+2}(treeCoch A)` for `q ≥ 2`, natural in `A`**
(`q = m + 2`), from the literal 2-periodicity of the tree resolution. -/
def treePerNat (m : ℕ) :
    treeCochFunctor k ⋙ HomologicalComplex.homologyFunctor (ModuleCat k) (ComplexShape.up ℕ) (m + 2) ≅
      treeCochFunctor k ⋙ HomologicalComplex.homologyFunctor (ModuleCat k) (ComplexShape.up ℕ) (m + 4) :=
  Functor.isoWhiskerLeft _ (HomologicalComplex.homologyFunctorIso' (ModuleCat k) (ComplexShape.up ℕ) (m + 1) (m + 2)
      (m + 3) (CochainComplex.prev_nat_succ (m + 1)) ((ComplexShape.up ℕ).next_eq' rfl)) ≪≫
    (Functor.associator _ _ _).symm ≪≫ Functor.isoWhiskerRight (treeScPer k m) _ ≪≫ Functor.associator _ _ _ ≪≫
    Functor.isoWhiskerLeft _ (HomologicalComplex.homologyFunctorIso' (ModuleCat k) (ComplexShape.up ℕ) (m + 3) (m + 4)
      (m + 5) (CochainComplex.prev_nat_succ (m + 3)) ((ComplexShape.up ℕ).next_eq' rfl)).symm

/-- **`H^q(SL₂(ℤ); A) ≅ H^{q+2}(SL₂(ℤ); A)` for every `q ≥ 2`** (`q = m + 2`), every commutative ring `k`, every `A`:
an abstract periodicity isomorphism from the 2-periodic tree resolution (not identified with `e ∪ −`). -/
def model7CohPeriodic (A : Rep.{0} k SL(2, ℤ)) (m : ℕ) : groupCohomology A (m + 2) ≅ groupCohomology A (m + 4) :=
  model7TreeCochCoh A (m + 2) ≪≫ (treePerNat k m).app A ≪≫ (model7TreeCochCoh A (m + 4)).symm

/-- Naturality of the periodicity isomorphism on the tree cochains. -/
lemma treePerNat_natural {A B : Rep.{0} k SL(2, ℤ)} (f : A ⟶ B) (m : ℕ) :
    HomologicalComplex.homologyMap (treeCochMap f) (m + 2) ≫ ((treePerNat k m).app B).hom =
      ((treePerNat k m).app A).hom ≫ HomologicalComplex.homologyMap (treeCochMap f) (m + 4) :=
  (treePerNat k m).hom.naturality f

end Periodic

/-! ### The Mayer–Vietoris sequence of the amalgam -/

section MV

variable {k : Type} [CommRing k] (A : Rep.{0} k SL(2, ℤ))

lemma tcD_sq_apply (n : ℕ) (v : tcX A n) : (tcD A (n + 1)).hom ((tcD A n).hom v) = 0 := by
  have h := congrArg (fun f => f.hom v) (tcD_sq A n)
  simpa using h

lemma vtxA_sq (n : ℕ) (w : A × A) : vtxA A (n + 1) (vtxA A n w) = 0 := by
  rcases n with _ | m
  · exact congrArg Prod.fst (tcD_sq_apply A 0 w)
  · have h := congrArg Prod.fst (tcD_sq_apply A (m + 1) (w, 0))
    exact h

lemma edgA_sq (n : ℕ) (z : A) : edgA A (n + 1) (edgA A n z) = 0 := by
  have h := congrArg Prod.snd (tcD_sq_apply A (n + 1) (0, z))
  have h' : phiA A (n + 2) (vtxA A (n + 1) 0) - edgA A (n + 1) (phiA A (n + 1) 0 - edgA A n z) = 0 := h
  simpa using h'

/-- **The vertex complex** `A² → A² → ⋯`, `(1 − S, 1 − U)`, `(N_S, N_U)`, …: `Hom_Γ(P^{⟨S⟩} ⊕ P^{⟨U⟩}, A)` in coordinates,
the periodic cochain complex of the two vertex stabilisers. -/
def vCoch : CochainComplex (ModuleCat k) ℕ :=
  CochainComplex.of (fun _ => ModuleCat.of k (A × A)) (fun n => ModuleCat.ofHom (vtxA A n)) (fun n => by
    apply ModuleCat.hom_ext; refine LinearMap.ext fun w => ?_
    exact vtxA_sq A n w)

lemma vCoch_d (n : ℕ) : (vCoch A).d n (n + 1) = ModuleCat.ofHom (vtxA A n) :=
  CochainComplex.of_d (fun _ : ℕ => ModuleCat.of k (A × A)) (fun n => ModuleCat.ofHom (vtxA A n)) n

/-- The terms of the edge complex, shifted by one: `0` in degree `0`, `A` above. -/
def eX : ℕ → ModuleCat k
  | 0 => ModuleCat.of k PUnit
  | _ + 1 => ModuleCat.of k A

/-- Its differentials: `0`, then `−(1 − (−I))`, `−(1 + (−I))`, …. -/
def eD : ∀ n, eX A n ⟶ eX A (n + 1)
  | 0 => 0
  | n + 1 => ModuleCat.ofHom (-edgA A n)

/-- **The edge complex, shifted by one** `0 → A → A → ⋯` (`H^{q+1}` of it is `H^q` of the periodic cochain complex of
`⟨−I⟩`, the differential negated). -/
def eSh : CochainComplex (ModuleCat k) ℕ :=
  CochainComplex.of (eX A) (eD A) (fun n => by
    rcases n with _ | n
    · exact Limits.zero_comp
    · apply ModuleCat.hom_ext; refine LinearMap.ext fun z => ?_
      show (-edgA A (n + 1)) ((-edgA A n) z) = 0
      show -(edgA A (n + 1) (-(edgA A n z))) = 0
      rw [map_neg, neg_neg]
      exact edgA_sq A n z)

lemma eSh_d (n : ℕ) : (eSh A).d n (n + 1) = eD A n := CochainComplex.of_d _ _ n

/-- The inclusion `z ↦ ((0, 0), z)` of the edge part. -/
def mvιF : ∀ n, eX A n ⟶ tcX A n
  | 0 => 0
  | _ + 1 => ModuleCat.ofHom (LinearMap.inr k (A × A) A)

/-- The projection `((x, y), z) ↦ (x, y)` onto the vertex part. -/
def mvπF : ∀ n, tcX A n ⟶ ModuleCat.of k (A × A)
  | 0 => 𝟙 _
  | _ + 1 => ModuleCat.ofHom (LinearMap.fst k (A × A) A)

/-- **`ι : eSh A ⟶ treeCoch A`.** -/
def mvι : eSh A ⟶ treeCoch A :=
  CochainComplex.ofHom (X := eSh A) (Y := treeCoch A) (mvιF A) (fun i => by
    rw [eSh_d, treeCoch_d]
    rcases i with _ | m
    · exact Limits.zero_comp.trans Limits.zero_comp.symm
    · apply ModuleCat.hom_ext; refine LinearMap.ext fun z => ?_
      show treeDS A m ((0, 0), z) = ((0, 0), (-edgA A m) z)
      refine Prod.ext ?_ ?_
      · show vtxA A (m + 1) 0 = 0
        exact map_zero _
      · show phiA A (m + 1) 0 - edgA A m z = (-edgA A m) z
        rw [map_zero, zero_sub]; rfl)

/-- **`π : treeCoch A ⟶ vCoch A`.** -/
def mvπ : treeCoch A ⟶ vCoch A :=
  CochainComplex.ofHom (X := treeCoch A) (Y := vCoch A) (mvπF A) (fun i => by
    rw [treeCoch_d, vCoch_d]
    rcases i with _ | m <;> rfl)

lemma mvι_mvπ : mvι A ≫ mvπ A = 0 := by
  apply HomologicalComplex.hom_ext; intro n
  apply ModuleCat.hom_ext; refine LinearMap.ext fun z => ?_
  rcases n with _ | m
  · exact congrArg (fun f => f.hom z) (Category.comp_id (0 : eX A 0 ⟶ tcX A 0))
  · rfl

/-- The short complex `0 → eSh A → treeCoch A → vCoch A → 0`. -/
abbrev mvSC : ShortComplex (CochainComplex (ModuleCat k) ℕ) := ShortComplex.mk (mvι A) (mvπ A) (mvι_mvπ A)

/-- **It is short exact** (degreewise: `0 → A → A³ → A² → 0`, resp. `0 → 0 → A² → A² → 0`). -/
theorem mvSC_shortExact : (mvSC A).ShortExact := by
  refine HomologicalComplex.shortExact_of_degreewise_shortExact _ fun i => ?_
  rcases i with _ | m
  · refine { exact := ?_, mono_f := ?_, epi_g := ?_ }
    · rw [ShortComplex.moduleCat_exact_iff]
      intro x hx
      exact ⟨0, by
        change ((mvιF A 0).hom 0 : tcX A 0) = x
        have hx' : x = 0 := hx
        rw [hx']; rfl⟩
    · rw [ModuleCat.mono_iff_injective]
      intro a b _
      exact Subsingleton.elim (α := PUnit) a b
    · rw [ModuleCat.epi_iff_surjective]
      intro x
      exact ⟨x, rfl⟩
  · refine { exact := ?_, mono_f := ?_, epi_g := ?_ }
    · rw [ShortComplex.moduleCat_exact_iff]
      intro x hx
      obtain ⟨w, z⟩ := x
      have hw : w = 0 := hx
      exact ⟨z, by rw [hw]; rfl⟩
    · rw [ModuleCat.mono_iff_injective]
      intro a b h
      exact congrArg Prod.snd h
    · rw [ModuleCat.epi_iff_surjective]
      intro w
      exact ⟨(w, 0), rfl⟩

/-- **The connecting map** `H^q(vCoch A) ⟶ H^{q+1}(eSh A)` of the Mayer–Vietoris sequence. -/
def mvδ (q : ℕ) : (vCoch A).homology q ⟶ (eSh A).homology (q + 1) := (mvSC_shortExact A).δ q (q + 1) rfl

/-- `H^q(Γ; A) → H^q(vCoch A)`: the projection to the vertex part, through `model7TreeCochCoh`. -/
def mvRes (q : ℕ) : groupCohomology A q ⟶ (vCoch A).homology q :=
  (model7TreeCochCoh A q).hom ≫ HomologicalComplex.homologyMap (mvπ A) q

/-- `H^q(eSh A) → H^q(Γ; A)`: the inclusion of the edge part, through `model7TreeCochCoh`. -/
def mvInc (q : ℕ) : (eSh A).homology q ⟶ groupCohomology A q :=
  HomologicalComplex.homologyMap (mvι A) q ≫ (model7TreeCochCoh A q).inv

lemma mvInc_mvRes (q : ℕ) : mvInc A q ≫ mvRes A q = 0 := by
  simp only [mvInc, mvRes, Category.assoc, Iso.inv_hom_id_assoc, ← HomologicalComplex.homologyMap_comp, mvι_mvπ,
    HomologicalComplex.homologyMap_zero]

lemma mvRes_mvδ (q : ℕ) : mvRes A q ≫ mvδ A q = 0 := by
  have h : HomologicalComplex.homologyMap (mvπ A) q ≫ mvδ A q = 0 := (mvSC_shortExact A).comp_δ q (q + 1) rfl
  rw [mvRes, Category.assoc, h, Limits.comp_zero]

lemma mvδ_mvInc (q : ℕ) : mvδ A q ≫ mvInc A (q + 1) = 0 := by
  have h : mvδ A q ≫ HomologicalComplex.homologyMap (mvι A) (q + 1) = 0 := (mvSC_shortExact A).δ_comp q (q + 1) rfl
  rw [mvInc, ← Category.assoc, h, Limits.zero_comp]

/-- **Mayer–Vietoris, exactness at `H^q(Γ; A)`**: `H^q(eSh A) → H^q(Γ; A) → H^q(vCoch A)`. -/
theorem mv_exact_coh (q : ℕ) : (ShortComplex.mk _ _ (mvInc_mvRes A q)).Exact := by
  refine ShortComplex.exact_of_iso ?_ ((mvSC_shortExact A).homology_exact₂ q)
  exact ShortComplex.isoMk (Iso.refl _) (model7TreeCochCoh A q).symm (Iso.refl _)
    (by simp [mvInc]) (by simp [mvRes])

/-- **Mayer–Vietoris, exactness at `H^q(vCoch A)`**: `H^q(Γ; A) → H^q(vCoch A) → H^{q+1}(eSh A)`. -/
theorem mv_exact_vtx (q : ℕ) : (ShortComplex.mk _ _ (mvRes_mvδ A q)).Exact := by
  refine ShortComplex.exact_of_iso ?_ ((mvSC_shortExact A).homology_exact₃ q (q + 1) rfl)
  exact ShortComplex.isoMk (model7TreeCochCoh A q).symm (Iso.refl _) (Iso.refl _)
    (by simp [mvRes]) (by simp [mvδ])

/-- **Mayer–Vietoris, exactness at `H^{q+1}(eSh A)`**: `H^q(vCoch A) → H^{q+1}(eSh A) → H^{q+1}(Γ; A)`. -/
theorem mv_exact_edge (q : ℕ) : (ShortComplex.mk _ _ (mvδ_mvInc A q)).Exact := by
  refine ShortComplex.exact_of_iso ?_ ((mvSC_shortExact A).homology_exact₁ q (q + 1) rfl)
  exact ShortComplex.isoMk (Iso.refl _) (Iso.refl _) (model7TreeCochCoh A (q + 1)).symm
    (by simp [mvδ]) (by simp [mvInc])

/-- The section `(x, y) ↦ ((x, y), 0)` of the projection (the identity in degree `0`). -/
def mvσF : ∀ n, (vCoch A).X n ⟶ (treeCoch A).X n
  | 0 => 𝟙 _
  | _ + 1 => ModuleCat.ofHom (LinearMap.inl k (A × A) A)

/-- `φ*_q : A² → A`, landing in degree `q + 1` of the shifted edge complex. -/
def mvφF (q : ℕ) : (vCoch A).X q ⟶ (eSh A).X (q + 1) := ModuleCat.ofHom (phiA A q)

/-- `φ*` is a chain map on cocycles: `e_q (φ*_q v) = 0` when `v_q v = 0` (from `d ∘ d = 0` on `A² → A³ → A³`). -/
lemma edgA_phiA (q : ℕ) (v : A × A) (hv : vtxA A q v = 0) : edgA A q (phiA A q v) = 0 := by
  rcases q with _ | m
  · have h := congrArg Prod.snd (tcD_sq_apply A 0 v)
    have h' : phiA A 1 (vtxA A 0 v) - edgA A 0 (phiA A 0 v) = 0 := h
    rw [hv, map_zero, zero_sub, neg_eq_zero] at h'
    exact h'
  · have h := congrArg Prod.snd (tcD_sq_apply A (m + 1) (v, 0))
    have h' : phiA A (m + 2) (vtxA A (m + 1) v) - edgA A (m + 1) (phiA A (m + 1) v - edgA A m 0) = 0 := h
    rw [hv, map_zero, map_zero, sub_zero, zero_sub, neg_eq_zero] at h'
    exact h'

lemma vCoch_cocycle (q : ℕ) {W : ModuleCat k} (x : W ⟶ (vCoch A).X q) (hx : x ≫ (vCoch A).d q (q + 1) = 0)
    (w : W) : vtxA A q (x.hom w) = 0 := by
  have h := congrArg (fun f => f.hom w) hx
  rw [vCoch_d] at h
  exact h

lemma mvφ_lift (q : ℕ) {W : ModuleCat k} (x : W ⟶ (vCoch A).X q) (hx : x ≫ (vCoch A).d q (q + 1) = 0) :
    (x ≫ mvφF A q) ≫ (mvι A).f (q + 1) = (x ≫ mvσF A q) ≫ (treeCoch A).d q (q + 1) := by
  rw [treeCoch_d]
  apply ModuleCat.hom_ext; refine LinearMap.ext fun w => ?_
  have hw := vCoch_cocycle A q x hx w
  rcases q with _ | m
  · show ((0 : A × A), phiA A 0 (x.hom w)) = treeD0 A (x.hom w)
    exact Prod.ext hw.symm rfl
  · show ((0 : A × A), phiA A (m + 1) (x.hom w)) = treeDS A m (x.hom w, 0)
    refine Prod.ext hw.symm ?_
    show phiA A (m + 1) (x.hom w) = phiA A (m + 1) (x.hom w) - edgA A m 0
    rw [map_zero, sub_zero]

lemma mvφ_cocycle (q : ℕ) {W : ModuleCat k} (x : W ⟶ (vCoch A).X q) (hx : x ≫ (vCoch A).d q (q + 1) = 0) :
    (x ≫ mvφF A q) ≫ (eSh A).d (q + 1) (q + 2) = 0 := by
  rw [eSh_d]
  apply ModuleCat.hom_ext; refine LinearMap.ext fun w => ?_
  show (-edgA A q) (phiA A q (x.hom w)) = 0
  rw [LinearMap.neg_apply, edgA_phiA A q _ (vCoch_cocycle A q x hx w), neg_zero]

/-- **The connecting map of the Mayer–Vietoris sequence is `φ*`**: for a cocycle `x` of the vertex complex in degree
`q`, `δ[x] = [φ*_q x]` in `H^{q+1}(eSh A)` (`φ*_q(x, y) = x − y` for even `q`, `(1 + S)x − (1 + U + U²)y` for odd `q`). -/
theorem mvδ_eq (q : ℕ) {W : ModuleCat k} (x : W ⟶ (vCoch A).X q) (hx : x ≫ (vCoch A).d q (q + 1) = 0) :
    (vCoch A).liftCycles x (q + 1) ((ComplexShape.up ℕ).next_eq' rfl) hx ≫ (vCoch A).homologyπ q ≫ mvδ A q =
      (eSh A).liftCycles (x ≫ mvφF A q) (q + 2) ((ComplexShape.up ℕ).next_eq' rfl) (mvφ_cocycle A q x hx) ≫
        (eSh A).homologyπ (q + 1) :=
  (mvSC_shortExact A).δ_eq q (q + 1) rfl x hx (x ≫ mvσF A q) (by
      apply ModuleCat.hom_ext; refine LinearMap.ext fun w => ?_
      rcases q with _ | m <;> rfl)
    (x ≫ mvφF A q) (mvφ_lift A q x hx) (q + 2) ((ComplexShape.up ℕ).next_eq' rfl)

end MV

end GL2.MP
