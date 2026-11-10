/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.MP.TreeCochains
import GL2.Conventions.SymPow

/-!
# LG-1 (slot 23): Theorem O(1), `2 · H^q(SL₂(ℤ); X) = 0` when `−I` acts by `−1`, every `q`

`L23_master_plan.md` v0.4 §1 Theorem O(1) and §4.3 (row LG-1); [C] `prop:badprimes`(2), `lem:centralminusone`.
[GT] `gt_part_MP.tex`, rows `thm:lg-exp2-deg1`, `thm:lg-exp2-tree`, `thm:lg-exp2`, `thm:lg-exp2-sym`.

* **Degree 1, any group** (`cocycle_two_smul`, `H1_two_smul_eq_zero`): for a central `z` acting by `−1`, every
  1-cocycle satisfies `2 f(g) = (g − 1)(−f(z))` (put `zg = gz` into the cocycle identity), so `2 · H¹(G; A) = 0`.
* **Every degree, `SL₂(ℤ)`** (`treeTwoNull`, `treeCoch_homologyMap_two`, `two_smul_eq_zero`): on LR-Y's explicit
  complex `treeCoch A = (A² → A³ → A³ → ⋯)` (`GL2.MP.treeCoch`, `model7TreeCochCoh`), with `−I ↦ −1` one has
  `S² = U³ = −1`, `N_S = N_U = 0`, edge block `2` (even) / `0` (odd), and the explicit maps
  `h¹((x, y), z) = (N₂ x, N₃ y)`, `h^{odd ≥ 3}((x, y), z) = ((N₂ x, N₃ y), 0)`, `h^{even}((x, y), z) = ((0, 0), −z)`
  (`N₂ = 1 + S`, `N₃ = 1 + U + U²`) form a **chain homotopy from `2 · id` to `0`** — a finite certificate, the same in every
  degree (LR's complex is 2-periodic).  Hence `2 • x = 0` for every `x ∈ H^q(SL₂(ℤ); A)`, every `q`, every commutative
  ring `k` and every `A : Rep k SL₂(ℤ)` with `ρ(−I) = −1`.  No hypothesis on `2` (it need not be a unit or a non-unit).
* **`Sym^n k²`, `n` odd** (`VK`, `VK_negOneActs`, `sym_odd_two_smul_eq_zero`): `ρ_n(−I) = (−1)^n` ([C] `def:rhon`), so the
  theorem applies to `Sym^n ℤ²`, `Sym^n (ℤ/2^K)²`, `Sym^n ℤ₂²`, … (any `k`).
-/

noncomputable section

namespace GL2.L23.OddExp

open CategoryTheory GL2.MP groupCohomology
open scoped MatrixGroups

/-! ### Degree 1: the cocycle identity at a central element acting by `−1` (any group) -/

section Degree1

variable {k G : Type} [CommRing k] [Group G] {A : Rep.{0} k G}

/-- **The cocycle identity** at a central `z` with `ρ(z) = −1`: `2 f(g) = ρ(g)(−f z) − (−f z)`. -/
theorem cocycle_two_smul (f : cocycles₁ A) {z : G} (hz : ∀ g, z * g = g * z) (hzA : ∀ a : A, A.ρ z a = -a)
    (g : G) : (2 : k) • f g = A.ρ g (-f z) - (-f z) := by
  have h1 : f (z * g) = A.ρ z (f g) + f z := (mem_cocycles₁_iff (f : G → A)).1 f.2 z g
  have h2 : f (g * z) = A.ρ g (f z) + f g := (mem_cocycles₁_iff (f : G → A)).1 f.2 g z
  rw [hz g, h2, hzA] at h1
  rw [map_neg, two_smul]
  calc f g + f g = (A.ρ g (f z) + f g) + f g - A.ρ g (f z) := by abel
    _ = (-f g + f z) + f g - A.ρ g (f z) := by rw [h1]
    _ = -A.ρ g (f z) - -f z := by abel

/-- **`2 · H¹(G; A) = 0`** for a central `z` acting by `−1`. -/
theorem H1_two_smul_eq_zero {z : G} (hz : ∀ g, z * g = g * z) (hzA : ∀ a : A, A.ρ z a = -a) (x : H1 A) :
    (2 : k) • x = 0 := by
  induction x using H1_induction_on with
  | h f =>
    rw [← map_smul, H1π_eq_zero_iff]
    refine ⟨-f z, ?_⟩
    funext g
    rw [d₀₁_hom_apply]
    exact (cocycle_two_smul f hz hzA g).symm

end Degree1

/-! ### Every degree on `SL₂(ℤ)`: the chain-homotopy certificate on the tree complex -/

section Tree

variable {k : Type} [CommRing k]

/-- **`−I` acts by `−1`** on `A`. -/
def NegOneActs (A : Rep.{0} k SL(2, ℤ)) : Prop := ∀ a : A, A.ρ (-1) a = -a

variable {A : Rep.{0} k SL(2, ℤ)}

lemma rS2 (hI : NegOneActs A) (x : A) : A.ρ sS (A.ρ sS x) = -x := by
  have h : A.ρ (sS ^ 2) x = -x := by rw [sS_sq]; exact hI x
  simpa [pow_two, map_mul, Module.End.mul_apply] using h

lemma rU3 (hI : NegOneActs A) (x : A) : A.ρ sU (A.ρ sU (A.ρ sU x)) = -x := by
  have h : A.ρ (sU ^ 3) x = -x := by rw [sU_cube]; exact hI x
  simpa [pow_succ, map_mul, Module.End.mul_apply] using h

lemma nrm2S (x : A) : nrmA A sS 2 x = x + A.ρ sS x := by
  simp [nrmA_apply, Finset.sum_range_succ]

lemma nrm3U (x : A) : nrmA A sU 3 x = x + A.ρ sU x + A.ρ sU (A.ρ sU x) := by
  simp [nrmA_apply, Finset.sum_range_succ, pow_two, Module.End.mul_apply]

lemma nrm4S (hI : NegOneActs A) (x : A) : nrmA A sS 4 x = 0 := by
  have h3 : A.ρ sS (A.ρ sS (A.ρ sS x)) = -A.ρ sS x := rS2 hI _
  simp only [nrmA_apply, Finset.sum_range_succ, Finset.sum_range_zero, zero_add, pow_succ, pow_zero, one_mul,
    map_mul, Module.End.mul_apply, map_one, Module.End.one_apply]
  rw [h3, rS2 hI]; abel

lemma nrm6U (hI : NegOneActs A) (x : A) : nrmA A sU 6 x = 0 := by
  have h4 : A.ρ sU (A.ρ sU (A.ρ sU (A.ρ sU x))) = -A.ρ sU x := rU3 hI _
  have h5 : A.ρ sU (A.ρ sU (A.ρ sU (A.ρ sU (A.ρ sU x)))) = -A.ρ sU (A.ρ sU x) := rU3 hI _
  simp only [nrmA_apply, Finset.sum_range_succ, Finset.sum_range_zero, zero_add, pow_succ, pow_zero, one_mul,
    map_mul, Module.End.mul_apply, map_one, Module.End.one_apply]
  rw [h5, h4, rU3 hI]; abel

/-- `N₂ (1 − S) = 2` once `S² = −1`. -/
lemma nrm2S_dif (hI : NegOneActs A) (x : A) : nrmA A sS 2 (x - A.ρ sS x) = x + x := by
  rw [nrm2S]; simp only [map_sub, rS2 hI]; abel

/-- `(1 − S) N₂ = 2` once `S² = −1`. -/
lemma dif_nrm2S (hI : NegOneActs A) (x : A) : nrmA A sS 2 x - A.ρ sS (nrmA A sS 2 x) = x + x := by
  rw [nrm2S]; simp only [map_add, rS2 hI]; abel

/-- `N₃ (1 − U) = 2` once `U³ = −1`. -/
lemma nrm3U_dif (hI : NegOneActs A) (x : A) : nrmA A sU 3 (x - A.ρ sU x) = x + x := by
  rw [nrm3U]; simp only [map_sub, rU3 hI]; abel

/-- `(1 − U) N₃ = 2` once `U³ = −1`. -/
lemma dif_nrm3U (hI : NegOneActs A) (x : A) : nrmA A sU 3 x - A.ρ sU (nrmA A sU 3 x) = x + x := by
  rw [nrm3U]; simp only [map_add, rU3 hI]; abel

lemma edgA_even_apply (hI : NegOneActs A) {m : ℕ} (hm : Even m) (z : A) : edgA A m z = z + z := by
  rw [edgA, if_pos hm, difA_apply, hI, sub_neg_eq_add]

lemma edgA_odd_apply (hI : NegOneActs A) {m : ℕ} (hm : ¬ Even m) (z : A) : edgA A m z = 0 := by
  rw [edgA, if_neg hm]
  simp only [nrmA_apply, Finset.sum_range_succ, Finset.sum_range_zero, zero_add, pow_zero, pow_one, map_one,
    Module.End.one_apply]
  rw [hI]; abel

lemma treeD0_apply' (x y : A) : treeD0 A (x, y) = ((x - A.ρ sS x, y - A.ρ sU y), x - y) := by
  simp [treeD0, vtxA, phiA, difA]

/-- `d^{j+1}` for even `j` (an odd degree): `((x, y), z) ↦ ((0, 0), N₂ x − N₃ y − 2z)`. -/
lemma treeDS_even_apply (hI : NegOneActs A) {j : ℕ} (hj : Even j) (x y z : A) :
    treeDS A j ((x, y), z) = ((0, 0), nrmA A sS 2 x - nrmA A sU 3 y - (z + z)) := by
  have hj1 : ¬ Even (j + 1) := by rw [Nat.even_add_one]; exact not_not.mpr hj
  show (vtxA A (j + 1) (x, y), phiA A (j + 1) (x, y) - edgA A j z) = _
  rw [vtxA, phiA, if_neg hj1, if_neg hj1, LinearMap.prodMap_apply, nrm4S hI, nrm6U hI, edgA_even_apply hI hj]
  rfl

/-- `d^{j+1}` for odd `j` (an even degree): `((x, y), z) ↦ (((1 − S)x, (1 − U)y), x − y)`. -/
lemma treeDS_odd_apply (hI : NegOneActs A) {j : ℕ} (hj : ¬ Even j) (x y z : A) :
    treeDS A j ((x, y), z) = ((x - A.ρ sS x, y - A.ρ sU y), x - y) := by
  have hj1 : Even (j + 1) := by rw [Nat.even_add_one]; exact hj
  show (vtxA A (j + 1) (x, y), phiA A (j + 1) (x, y) - edgA A j z) = _
  rw [vtxA, phiA, if_pos hj1, if_pos hj1, LinearMap.prodMap_apply, edgA_odd_apply hI hj, sub_zero, difA_apply,
    difA_apply]
  rfl

variable (A)

/-- `h¹ : A³ → A²`, `((x, y), z) ↦ (N₂ x, N₃ y)`. -/
def hOne : (A × A) × A →ₗ[k] A × A :=
  (nrmA A sS 2).prodMap (nrmA A sU 3) ∘ₗ LinearMap.fst k (A × A) A

/-- `h^{odd ≥ 3} : A³ → A³`, `((x, y), z) ↦ ((N₂ x, N₃ y), 0)`. -/
def hOdd : (A × A) × A →ₗ[k] (A × A) × A := (hOne A).prod 0

/-- `h^{even ≥ 2} : A³ → A³`, `((x, y), z) ↦ ((0, 0), −z)`. -/
def hEven : (A × A) × A →ₗ[k] (A × A) × A := (0 : (A × A) × A →ₗ[k] A × A).prod (-LinearMap.snd k (A × A) A)

/-- The homotopy `h^{j+1} : X^{j+1} → X^j`. -/
def htH : ∀ j, tcX A (j + 1) ⟶ tcX A j
  | 0 => ModuleCat.ofHom (hOne A)
  | m + 1 => ModuleCat.ofHom (if Even m then hEven A else hOdd A)

/-- The homotopy in the form of `nullHomotopicMap'` (`c.Rel j i ↔ j + 1 = i`). -/
def treeH : ∀ i j, (ComplexShape.up ℕ).Rel j i → ((treeCoch A).X i ⟶ (treeCoch A).X j) :=
  fun _ j hij => eqToHom (congrArg (tcX A) hij.symm) ≫ htH A j

lemma treeH_succ (j : ℕ) (h : (ComplexShape.up ℕ).Rel j (j + 1)) : treeH A (j + 1) j h = htH A j := by
  show eqToHom _ ≫ htH A j = htH A j
  rw [eqToHom_refl, Category.id_comp]

variable {A}

lemma hOne_apply (x y z : A) : hOne A ((x, y), z) = (nrmA A sS 2 x, nrmA A sU 3 y) := rfl
lemma hOdd_apply (x y z : A) : hOdd A ((x, y), z) = ((nrmA A sS 2 x, nrmA A sU 3 y), 0) := rfl
lemma hEven_apply (x y z : A) : hEven A ((x, y), z) = ((0, 0), -z) := rfl

/-- **The certificate**: `d h + h d = 2 · id` on `treeCoch A` when `−I` acts by `−1`. -/
theorem treeTwoNull (hI : NegOneActs A) :
    Homotopy.nullHomotopicMap' (treeH A) = 𝟙 (treeCoch A) + 𝟙 (treeCoch A) := by
  ext n : 1
  rw [HomologicalComplex.add_f_apply, HomologicalComplex.id_f]
  rcases n with _ | m
  · rw [Homotopy.nullHomotopicMap'_f_of_not_rel_right (c := ComplexShape.up ℕ) (k₁ := 0) (k₀ := 1)
      (show (ComplexShape.up ℕ).Rel 0 1 from rfl)
      (fun l hl => Nat.succ_ne_zero l hl), treeH_succ, treeCoch_d]
    apply ModuleCat.hom_ext; refine LinearMap.ext fun v => ?_
    obtain ⟨x, y⟩ := v
    show hOne A (treeD0 A (x, y)) = (x, y) + (x, y)
    rw [treeD0_apply', hOne_apply, nrm2S_dif hI, nrm3U_dif hI]; rfl
  · rw [Homotopy.nullHomotopicMap'_f (c := ComplexShape.up ℕ) (k₂ := m) (k₁ := m + 1) (k₀ := m + 2)
      (show (ComplexShape.up ℕ).Rel m (m + 1) from rfl) (show (ComplexShape.up ℕ).Rel (m + 1) (m + 2) from rfl),
      treeH_succ, treeH_succ, treeCoch_d, treeCoch_d]
    apply ModuleCat.hom_ext; refine LinearMap.ext fun v => ?_
    obtain ⟨⟨x, y⟩, z⟩ := v
    rcases m with _ | m
    · show hEven A (treeDS A 0 ((x, y), z)) + treeD0 A (hOne A ((x, y), z)) = ((x, y), z) + ((x, y), z)
      rw [treeDS_even_apply hI (by decide), hEven_apply, hOne_apply, treeD0_apply', dif_nrm2S hI, dif_nrm3U hI]
      refine Prod.ext (Prod.ext ?_ ?_) ?_ <;> simp
    · by_cases hm : Even m
      · have hm1 : ¬ Even (m + 1) := by rw [Nat.even_add_one]; exact not_not.mpr hm
        show (if Even (m + 1) then hEven A else hOdd A) (treeDS A (m + 1) ((x, y), z)) +
            treeDS A m ((if Even m then hEven A else hOdd A) ((x, y), z)) = ((x, y), z) + ((x, y), z)
        rw [if_neg hm1, if_pos hm, treeDS_odd_apply hI hm1, hOdd_apply, hEven_apply, treeDS_even_apply hI hm,
          nrm2S_dif hI, nrm3U_dif hI]
        refine Prod.ext (Prod.ext ?_ ?_) ?_ <;> simp
      · have hm1 : Even (m + 1) := by rw [Nat.even_add_one]; exact hm
        show (if Even (m + 1) then hEven A else hOdd A) (treeDS A (m + 1) ((x, y), z)) +
            treeDS A m ((if Even m then hEven A else hOdd A) ((x, y), z)) = ((x, y), z) + ((x, y), z)
        rw [if_pos hm1, if_neg hm, treeDS_even_apply hI hm1, hEven_apply, hOdd_apply, treeDS_odd_apply hI hm,
          dif_nrm2S hI, dif_nrm3U hI]
        refine Prod.ext (Prod.ext ?_ ?_) ?_ <;> simp

/-- `2 · id` induces `0` on `H^n(treeCoch A)`, every `n`. -/
theorem treeCoch_homologyMap_two (hI : NegOneActs A) (n : ℕ) :
    HomologicalComplex.homologyMap (𝟙 (treeCoch A) + 𝟙 (treeCoch A)) n = 0 := by
  rw [← treeTwoNull hI, (Homotopy.nullHomotopy' (treeH A)).homologyMap_eq n, HomologicalComplex.homologyMap_zero]

/-- **Theorem O(1)**: `2 · H^n(SL₂(ℤ); A) = 0` for every `n`, every commutative ring `k`, every `A` with `−I ↦ −1`. -/
theorem two_smul_eq_zero (hI : NegOneActs A) (n : ℕ) (x : groupCohomology A n) : (2 : k) • x = 0 := by
  apply (model7TreeCochCoh A n).toLinearEquiv.injective
  rw [map_zero, two_smul, map_add]
  have h := congrArg (fun φ => φ.hom ((model7TreeCochCoh A n).toLinearEquiv x)) (treeCoch_homologyMap_two hI n)
  simpa [HomologicalComplex.homologyMap_add, HomologicalComplex.homologyMap_id] using h

/-- The same for `x + x`. -/
theorem add_self_eq_zero (hI : NegOneActs A) (n : ℕ) (x : groupCohomology A n) : x + x = 0 := by
  rw [← two_smul k x]; exact two_smul_eq_zero hI n x

/-- **`H^n(SL₂(ℤ); A) = 0`** if `−I` acts by `−1` and `2` is a unit of `k` (every `n`, any commutative ring `k`). -/
theorem eq_zero_of_two_unit (hI : NegOneActs A) (h2 : IsUnit (2 : k)) (n : ℕ) (x : groupCohomology A n) : x = 0 := by
  obtain ⟨u, hu⟩ := h2
  calc x = (↑u⁻¹ * (2 : k)) • x := by rw [← hu, Units.inv_mul, one_smul]
    _ = (↑u⁻¹ : k) • ((2 : k) • x) := mul_smul _ _ _
    _ = 0 := by rw [two_smul_eq_zero hI n x, smul_zero]

end Tree

/-! ### `Sym^n k²` for odd `n` -/

section Sym

variable (k : Type) [CommRing k]

/-- The reduction `SL₂(ℤ) → M₂(k)`. -/
def toMatK (g : SL(2, ℤ)) : GL2.Mat2 k := (g : Matrix (Fin 2) (Fin 2) ℤ).map (Int.castRingHom k)

/-- **`Sym^n k²`** as a representation of `SL₂(ℤ)`, `g ↦ ρ_n(g)` ([C] `def:rhon`, `GL2.rho`). -/
def symRepK (n : ℕ) : Representation k SL(2, ℤ) (GL2.SymN k n) where
  toFun g := GL2.rho n (toMatK k g)
  map_one' := by
    rw [show toMatK k 1 = 1 by simp [toMatK]]; exact GL2.rho_one n
  map_mul' g h := by
    rw [show toMatK k (g * h) = toMatK k g * toMatK k h by
      simp only [toMatK, Matrix.SpecialLinearGroup.coe_mul, Matrix.map_mul]]
    exact GL2.rho_mul n _ _

/-- `Sym^n k²` in `Rep k SL₂(ℤ)`. -/
abbrev VK (n : ℕ) : Rep.{0} k SL(2, ℤ) := Rep.of (symRepK k n)

variable {k}

lemma toMatK_negOne : toMatK k (-1) = (-1 : k) • (1 : GL2.Mat2 k) := by
  ext i j; fin_cases i <;> fin_cases j <;> simp [toMatK]

/-- `ρ_n(−I) = (−1)^n`. -/
lemma VK_negOne (n : ℕ) : (VK k n).ρ (-1) = (-1 : k) ^ n • (1 : Module.End k (GL2.SymN k n)) := by
  show GL2.rho n (toMatK k (-1)) = _
  rw [toMatK_negOne, GL2.rho_smul_one]

/-- For odd `n`, `−I` acts by `−1` on `Sym^n k²`. -/
theorem VK_negOneActs {n : ℕ} (hn : Odd n) : NegOneActs (VK k n) := fun a => by
  rw [VK_negOne, hn.neg_one_pow, LinearMap.smul_apply, Module.End.one_apply, neg_one_smul]

/-- **Theorem O(1) for `Sym^n k²`, `n` odd**: `2 · H^q(SL₂(ℤ); Sym^n k²) = 0`, every `q`, every commutative ring `k`. -/
theorem sym_odd_two_smul_eq_zero {n : ℕ} (hn : Odd n) (q : ℕ) (x : groupCohomology (VK k n) q) : (2 : k) • x = 0 :=
  two_smul_eq_zero (VK_negOneActs hn) q x

end Sym

end GL2.L23.OddExp
