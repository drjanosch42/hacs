/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.L23.TreeF3V3

/-!
# LG-C3 (slot 23): [R] Theorem C(1) at `ℓ = 3`, in general — `h^{q+2} = [n ≡ 0, 4 mod 6]` for every even `n`

`L23_master_plan.md` §4.3 (row "[R] Thm A, C dimension closed forms"); [R] = `census_l23_classification_review.md`
Theorem C(1); [C] `prop:ellthreeclosed`(2).  [GT] `gt_part_MP.tex`, rows `thm:lg-c3-jordan`, `thm:lg-c3-dims`.
Until now the `ℓ = 3` dimensions were per cell only (`GL2.L23.Cells.dims3_n`, even `n ≤ 64`, `chk:lv3-cells`).

* **The ranks of `D`, `D²` for a unipotent element on `Sym^n F₃²`, every `n`** (`finrank_range_Dc`, `finrank_range_Dc_sq`):
  for `C = !![1, −1; 0, 1]` (`x ↦ x`, `y ↦ y − x`) and `D = ρ_n(C) − 1`,
  `rank D = n − ⌊n/3⌋` and `rank D² = ⌊(n+1)/3⌋`; with `D³ = 0` this is the Jordan type `J₃^{⌊n/3⌋} ⊕ J_{(n mod 3)+1}`
  ([R] Theorem C(1), first sentence, for `C`, a generator of a Sylow `3`-subgroup of `GL₂(F₃)`; `u = U²` is conjugate to
  `C^{±1}`).  Only the two ranks are proved here: `D³ = 0` is not formalised (it follows from `Dop_sq_cS`, since `D² c_j`
  is `c_{j−2}` only for `j ≡ 2 mod 3` and then `D c_{j−2} = 0`), so the Jordan type is prose.  The proof is [R]'s free-basis argument made explicit: with
  `N = y³ − x²y` (invariant in characteristic `3`) and `β₀ = 1`, `β₁ = −(y + x)`, `β₂ = −y²`, the `n + 1` forms
  `c_j = x^{n−j} β_{j mod 3} N^{⌊j/3⌋}` (`cPoly`) satisfy `D c_j = c_{j−1}` for `3 ∤ j` and `D c_j = 0` for `3 ∣ j`
  (`subst_cPoly_zero`, `subst_cPoly_succ`), and they are a basis: setting `x = 1` sends `c_j` to a polynomial of degree
  exactly `j` (`dehom_cPoly_deg`, `sum_smul_eq_zero`).  In this basis `D` and `D²` are partial shifts (`Jop`), whose
  ranks are counts of residues (`finrank_range_Jop`, `cnt_ne2`, `cnt_eq0`).
* **`ρ_n(U)` for even `n`** (`finrank_range_difA_V3`, `finrank_range_nrmA_V3`): `U ≡ !![0, −1; 1, 1]` and `−U` is
  conjugate to `C` in `GL₂(F₃)` (`mQCQ`, `mQ² = 1`), and `−I` acts trivially for even `n`, so `1 − ρ_n(U)` and
  `1 + ρ_n(U) + ρ_n(U²) = (ρ_n(U) − 1)²` (characteristic `3`) have the ranks above.
* **[R] Theorem C(1), dimensions** (`V3_dims_ge2`, `V3_dims_thmC`): with LV-2's tree-complex formula
  `dim H^{q+2} + rank (1 + U + U²) + rank (1 − U) = n + 1` (`finrank_H_of_trivNegOne`),
  **`dim_{F₃} H^{q+2}(SL₂(ℤ); Sym^n F₃²) = [n ≢ 2 mod 3] = [n ≡ 0, 4 mod 6]` for every even `n` and every `q ≥ 0`**
  (odd `n`: `0` in every degree, LV-2's `V3_dims_odd`).

**Not here** (gaps of LG-C3): `h⁰` and `h¹` at `ℓ = 3` (no closed form is claimed by [R]; per cell only,
`chk:lv3-cells`); [R] Theorem C(1)'s Bredon clauses (`dim coker(ε^*|_{H¹}) = [n ≡ 0, 4 mod 6]`, `H¹(group) = H¹(Bredon)`
iff `n ≡ 2 mod 6`: they need [C] `prop:ellthreeclosed`(3), the comparison with the shipped Bredon complex, which is not
formalised); C(2) (the `Z_(3)`-lattices, Krull–Schmidt), C(3) (inflation from `SL₂(F₃)`, LIT (F1)) and C(4)–(5).
The identification of this `H^q` (mathlib's `groupCohomology`) with [C]'s `H^q(Γ; M)` is LR-Y's, as for every
`L23` theorem.
-/

noncomputable section

namespace GL2.L23.Jordan3

open MvPolynomial Module GL2.MP GL2.L23.TreeF3 GL2.L23.RankF3
open scoped MatrixGroups

/-- `F₃[x, y]`, `x = X 0`, `y = X 1`. -/
abbrev P3 := MvPolynomial (Fin 2) (ZMod 3)

lemma three_eq_zero_P : (3 : P3) = 0 := by
  calc (3 : P3) = C (3 : ZMod 3) := (map_ofNat C 3).symm
    _ = 0 := by rw [show (3 : ZMod 3) = 0 by decide, map_zero]

/-! ### The unipotent element `C` and the invariant `N = y³ − x²y` -/

/-- `C = !![1, −1; 0, 1]`: `x ↦ x`, `y ↦ y − x` (column convention, [C] `def:rhon`). -/
def mC : Mat2 (ZMod 3) := !![1, -1; 0, 1]

lemma subst_mC_X0 : subst mC (X 0 : P3) = X 0 := by
  rw [subst_X, linForm_zero]; simp [mC]

lemma subst_mC_X1 : subst mC (X 1 : P3) = X 1 - X 0 := by
  rw [subst_X, linForm_one]; simp [mC]; ring

/-- `N = y³ − x²y`, the norm of `y` over `F₃[x, N]`. -/
def Nq : P3 := X 1 ^ 3 - X 0 ^ 2 * X 1

lemma subst_Nq : subst mC Nq = Nq := by
  simp only [Nq, map_sub, map_mul, map_pow, subst_mC_X0, subst_mC_X1]
  linear_combination (X 0 ^ 2 * X 1 - X 0 * X 1 ^ 2) * three_eq_zero_P

/-- `β₀ = 1`, `β₁ = −(y + x)`, `β₂ = −y²`. -/
def beta : ℕ → P3
  | 0 => 1
  | 1 => -(X 1 + X 0)
  | _ => -(X 1 ^ 2)

lemma beta_zero : beta 0 = 1 := rfl
lemma beta_one : beta 1 = -(X 1 + X 0) := rfl
lemma beta_two : beta 2 = -(X 1 ^ 2) := rfl

lemma subst_beta0 : subst mC (beta 0) = beta 0 := by rw [beta_zero, map_one]

lemma subst_beta1 : subst mC (beta 1) = beta 1 + X 0 * beta 0 := by
  rw [beta_one, beta_zero, map_neg, map_add, subst_mC_X0, subst_mC_X1]; ring

lemma subst_beta2 : subst mC (beta 2) = beta 2 + X 0 * beta 1 := by
  rw [beta_two, beta_one, map_neg, map_pow, subst_mC_X1]
  linear_combination (X 0 * X 1) * three_eq_zero_P

/-- The Jordan basis vector `c_j = x^{n−j} β_{j mod 3} N^{⌊j/3⌋}`. -/
def cPoly (n j : ℕ) : P3 := X 0 ^ (n - j) * beta (j % 3) * Nq ^ (j / 3)

lemma subst_cPoly_zero {n j : ℕ} (hr : j % 3 = 0) : subst mC (cPoly n j) = cPoly n j := by
  unfold cPoly
  rw [hr]
  simp only [map_mul, map_pow, subst_mC_X0, subst_Nq, subst_beta0]

/-- **`D c_j = c_{j−1}`** for `3 ∤ j`. -/
lemma subst_cPoly_succ {n j : ℕ} (hj : j ≤ n) (hr : j % 3 ≠ 0) :
    subst mC (cPoly n j) = cPoly n j + cPoly n (j - 1) := by
  have e2 : (j - 1) / 3 = j / 3 := by omega
  have e3 : n - (j - 1) = n - j + 1 := by omega
  rcases (show j % 3 = 1 ∨ j % 3 = 2 by omega) with h | h
  · have e1 : (j - 1) % 3 = 0 := by omega
    unfold cPoly
    rw [h, e1, e2, e3]
    simp only [map_mul, map_pow, subst_mC_X0, subst_Nq, subst_beta1]
    ring
  · have e1 : (j - 1) % 3 = 1 := by omega
    unfold cPoly
    rw [h, e1, e2, e3]
    simp only [map_mul, map_pow, subst_mC_X0, subst_Nq, subst_beta2]
    ring

/-! ### `c_j ∈ Sym^n` -/

lemma Nq_hom : (Nq : P3).IsHomogeneous 3 :=
  (isHomogeneous_X_pow (R := ZMod 3) (1 : Fin 2) 3).sub
    ((isHomogeneous_X_pow (R := ZMod 3) (0 : Fin 2) 2).mul (isHomogeneous_X (R := ZMod 3) (σ := Fin 2) 1))

lemma beta_hom (r : ℕ) (hr : r < 3) : (beta r).IsHomogeneous r := by
  interval_cases r
  · rw [beta_zero]; exact isHomogeneous_one (σ := Fin 2) (R := ZMod 3)
  · rw [beta_one]
    exact ((isHomogeneous_X (R := ZMod 3) (σ := Fin 2) 1).add (isHomogeneous_X (R := ZMod 3) (σ := Fin 2) 0)).neg
  · rw [beta_two]; exact (isHomogeneous_X_pow (R := ZMod 3) (1 : Fin 2) 2).neg

lemma cPoly_hom {n j : ℕ} (hj : j ≤ n) : (cPoly n j).IsHomogeneous n := by
  have h := ((isHomogeneous_X_pow (R := ZMod 3) (0 : Fin 2) (n - j)).mul
    (beta_hom (j % 3) (Nat.mod_lt _ (by norm_num)))).mul (Nq_hom.pow (j / 3))
  have e : n - j + j % 3 + 3 * (j / 3) = n := by omega
  rw [e] at h
  exact h

/-- `c_j` as an element of `Sym^n F₃²`. -/
def cS (n : ℕ) (j : Fin (n + 1)) : SymN (ZMod 3) n :=
  ⟨cPoly n j, (mem_homogeneousSubmodule _ _).2 (cPoly_hom (Nat.lt_succ_iff.mp j.2))⟩

@[simp] lemma cS_coe (n : ℕ) (j : Fin (n + 1)) : ((cS n j : SymN (ZMod 3) n) : P3) = cPoly n j := rfl

/-- `D = ρ_n(C) − 1` on `Sym^n F₃²`. -/
def Dop (n : ℕ) : Module.End (ZMod 3) (SymN (ZMod 3) n) := rho n mC - 1

lemma Dop_cS (n : ℕ) (j : Fin (n + 1)) :
    Dop n (cS n j) = if (j : ℕ) % 3 = 0 then 0 else cS n ⟨(j : ℕ) - 1, Nat.lt_of_le_of_lt (Nat.sub_le _ _) j.2⟩ := by
  apply Subtype.ext
  show subst mC (cPoly n j) - cPoly n j = ((if (j : ℕ) % 3 = 0 then 0 else
    cS n ⟨(j : ℕ) - 1, Nat.lt_of_le_of_lt (Nat.sub_le _ _) j.2⟩ : SymN (ZMod 3) n) : P3)
  split_ifs with h
  · rw [subst_cPoly_zero h]; exact sub_self _
  · rw [subst_cPoly_succ (Nat.lt_succ_iff.mp j.2) h]; exact add_sub_cancel_left _ _

lemma Dop_sq_cS (n : ℕ) (j : Fin (n + 1)) :
    Dop n (Dop n (cS n j)) =
      if (j : ℕ) % 3 = 2 then cS n ⟨(j : ℕ) - 2, Nat.lt_of_le_of_lt (Nat.sub_le _ _) j.2⟩ else 0 := by
  have hj : (j : ℕ) ≤ n := Nat.lt_succ_iff.mp j.2
  apply Subtype.ext
  show subst mC (subst mC (cPoly n j) - cPoly n j) - (subst mC (cPoly n j) - cPoly n j) =
    ((if (j : ℕ) % 3 = 2 then cS n ⟨(j : ℕ) - 2, Nat.lt_of_le_of_lt (Nat.sub_le _ _) j.2⟩ else 0 :
      SymN (ZMod 3) n) : P3)
  split_ifs with h
  · rw [subst_cPoly_succ hj (by omega), add_sub_cancel_left,
      subst_cPoly_succ (show (j : ℕ) - 1 ≤ n by omega) (by omega), add_sub_cancel_left,
      show (j : ℕ) - 1 - 1 = (j : ℕ) - 2 by omega]
    rfl
  · rcases (show (j : ℕ) % 3 = 0 ∨ (j : ℕ) % 3 = 1 by omega) with h0 | h1
    · rw [subst_cPoly_zero h0, sub_self, map_zero, sub_self]
      rfl
    · rw [subst_cPoly_succ hj (by omega), add_sub_cancel_left,
        subst_cPoly_zero (show ((j : ℕ) - 1) % 3 = 0 by omega), sub_self]
      rfl

/-! ### The `c_j` are a basis: dehomogenise (`x = 1`) and count degrees -/

/-- `x ↦ 1`, `y ↦ Y`. -/
def dehom : P3 →ₐ[ZMod 3] Polynomial (ZMod 3) := aeval ![1, Polynomial.X]

@[simp] lemma dehom_X0 : dehom (X 0) = 1 := by simp [dehom]
@[simp] lemma dehom_X1 : dehom (X 1) = Polynomial.X := by simp [dehom]

lemma dehom_cPoly (n j : ℕ) :
    dehom (cPoly n j) = dehom (beta (j % 3)) * (Polynomial.X ^ 3 - Polynomial.X) ^ (j / 3) := by
  simp only [cPoly, Nq, map_mul, map_pow, map_sub, dehom_X0, dehom_X1, one_pow, one_mul]

lemma natDegree_Y3 : (Polynomial.X ^ 3 - Polynomial.X : Polynomial (ZMod 3)).natDegree = 3 := by
  rw [Polynomial.natDegree_sub_eq_left_of_natDegree_lt] <;> simp

lemma Y3_ne_zero : (Polynomial.X ^ 3 - Polynomial.X : Polynomial (ZMod 3)) ≠ 0 := by
  intro h
  have := natDegree_Y3
  rw [h, Polynomial.natDegree_zero] at this
  exact absurd this (by norm_num)

lemma dehom_beta (r : ℕ) (hr : r < 3) : (dehom (beta r)).natDegree = r ∧ dehom (beta r) ≠ 0 := by
  interval_cases r
  · rw [beta_zero, map_one]; exact ⟨Polynomial.natDegree_one, one_ne_zero⟩
  · rw [beta_one, map_neg, map_add, dehom_X0, dehom_X1]
    have h : (Polynomial.X + 1 : Polynomial (ZMod 3)).natDegree = 1 := by
      rw [Polynomial.natDegree_add_eq_left_of_natDegree_lt] <;> simp
    refine ⟨by rw [Polynomial.natDegree_neg, h], ?_⟩
    rw [neg_ne_zero]
    intro h0
    rw [h0, Polynomial.natDegree_zero] at h
    exact absurd h (by norm_num)
  · rw [beta_two, map_neg, map_pow, dehom_X1]
    refine ⟨by rw [Polynomial.natDegree_neg, Polynomial.natDegree_X_pow], ?_⟩
    rw [neg_ne_zero]
    exact pow_ne_zero _ Polynomial.X_ne_zero

/-- `c_j(1, Y)` has degree exactly `j`. -/
lemma dehom_cPoly_deg (n : ℕ) (j : Fin (n + 1)) :
    (dehom (cPoly n j)).natDegree = j ∧ dehom (cPoly n j) ≠ 0 := by
  obtain ⟨hd, hne⟩ := dehom_beta ((j : ℕ) % 3) (Nat.mod_lt _ (by norm_num))
  rw [dehom_cPoly]
  have hp : (Polynomial.X ^ 3 - Polynomial.X : Polynomial (ZMod 3)) ^ ((j : ℕ) / 3) ≠ 0 := pow_ne_zero _ Y3_ne_zero
  refine ⟨?_, mul_ne_zero hne hp⟩
  rw [Polynomial.natDegree_mul hne hp, Polynomial.natDegree_pow, natDegree_Y3, hd]
  omega

/-- Polynomials of pairwise distinct exact degrees `0, …, m − 1` are linearly independent. -/
theorem sum_smul_eq_zero {m : ℕ} (p : Fin m → Polynomial (ZMod 3)) (hd : ∀ j, (p j).natDegree = j)
    (hne : ∀ j, p j ≠ 0) (α : Fin m → ZMod 3) (h : ∑ j, α j • p j = 0) : α = 0 := by
  by_contra hα
  obtain ⟨i, hi⟩ := Function.ne_iff.mp hα
  have hS : (Finset.univ.filter fun j => α j ≠ 0).Nonempty :=
    ⟨i, Finset.mem_filter.mpr ⟨Finset.mem_univ _, hi⟩⟩
  set j0 := (Finset.univ.filter fun j => α j ≠ 0).max' hS with hj0def
  have hj0 : α j0 ≠ 0 := (Finset.mem_filter.mp (Finset.max'_mem _ hS)).2
  have hc := congrArg (fun q : Polynomial (ZMod 3) => q.coeff (j0 : ℕ)) h
  simp only [Polynomial.finsetSum_coeff, Polynomial.coeff_smul, Polynomial.coeff_zero, smul_eq_mul] at hc
  rw [Finset.sum_eq_single j0] at hc
  · have hl : (p j0).coeff (j0 : ℕ) ≠ 0 := by
      have := Polynomial.leadingCoeff_ne_zero.mpr (hne j0)
      rwa [Polynomial.leadingCoeff, hd j0] at this
    exact mul_ne_zero hj0 hl hc
  · intro b _ hb
    by_cases hαb : α b = 0
    · rw [hαb, zero_mul]
    · have hle : b ≤ j0 := Finset.le_max' _ b (Finset.mem_filter.mpr ⟨Finset.mem_univ _, hαb⟩)
      have hlt : (b : ℕ) < j0 := Fin.lt_def.mp (lt_of_le_of_ne hle hb)
      rw [Polynomial.coeff_eq_zero_of_natDegree_lt (by rw [hd b]; exact hlt), mul_zero]
  · intro hn
    exact absurd (Finset.mem_univ j0) hn

/-! ### The change of basis `Ψ : F₃^{n+1} → F₃^{n+1}`, `e_j ↦ coordinates of c_j` -/

/-- `Ψ = (coordinates in u_t = x^{n−t} y^t) ∘ (e_j ↦ c_j)`. -/
def Psi (n : ℕ) : (Fin (n + 1) → ZMod 3) →ₗ[ZMod 3] (Fin (n + 1) → ZMod 3) :=
  (uBasis n).equivFun.toLinearMap ∘ₗ Fintype.linearCombination (ZMod 3) (cS n)

lemma Psi_single (n : ℕ) (j : Fin (n + 1)) :
    Psi n (Pi.single j 1) = (uBasis n).equivFun (cS n j) := by
  simp [Psi]

lemma Psi_injective (n : ℕ) : Function.Injective (Psi n) := by
  rw [injective_iff_map_eq_zero]
  intro α hα
  have h1 : Fintype.linearCombination (ZMod 3) (cS n) α = 0 := by
    apply (uBasis n).equivFun.injective
    rw [map_zero]
    exact hα
  have h2 := congrArg (fun v : SymN (ZMod 3) n => dehom (v : P3)) h1
  simp only [Fintype.linearCombination_apply, Submodule.coe_sum, Submodule.coe_smul, map_sum, map_smul, cS_coe,
    ZeroMemClass.coe_zero, map_zero] at h2
  exact sum_smul_eq_zero (fun j => dehom (cPoly n j)) (fun j => (dehom_cPoly_deg n j).1)
    (fun j => (dehom_cPoly_deg n j).2) α h2

lemma Psi_surjective (n : ℕ) : Function.Surjective (Psi n) :=
  LinearMap.injective_iff_surjective.mp (Psi_injective n)

/-- `toLin' (ρ_n-matrix) ∘ coordinates = coordinates ∘ ρ_n`. -/
lemma toLin'_rhoMat_equivFun (n : ℕ) (A : Mat2 (ZMod 3)) (v : SymN (ZMod 3) n) :
    Matrix.toLin' (GL2.rhoMat n A) ((uBasis n).equivFun v) = (uBasis n).equivFun (rho n A v) := by
  rw [Module.Basis.equivFun_apply, Module.Basis.equivFun_apply, Matrix.toLin'_apply, GL2.rhoMat,
    LinearMap.toMatrix_mulVec_repr]

/-- `D` in coordinates: `ρ_n(C) − 1` on `F₃^{n+1}`. -/
def Dc (n : ℕ) : Module.End (ZMod 3) (Fin (n + 1) → ZMod 3) := Matrix.toLin' (GL2.rhoMat n mC) - 1

lemma Dc_equivFun (n : ℕ) (v : SymN (ZMod 3) n) :
    Dc n ((uBasis n).equivFun v) = (uBasis n).equivFun (Dop n v) := by
  simp only [Dc, Dop, LinearMap.sub_apply, Module.End.one_apply, toLin'_rhoMat_equivFun, map_sub]

/-! ### Partial shifts and their ranks -/

section Shift

variable (n s : ℕ) (Q : ℕ → Prop) [DecidablePred Q]

/-- The rows `k` with `Q k` and `k + s ≤ n`. -/
abbrev Tsub := {k : Fin (n + 1) // Q (k : ℕ) ∧ (k : ℕ) + s < n + 1}

/-- `k ↦ k + s`. -/
def shiftIdx : Tsub n s Q → Fin (n + 1) := fun k => ⟨(k.1 : ℕ) + s, k.2.2⟩

omit [DecidablePred Q] in
lemma shiftIdx_injective : Function.Injective (shiftIdx n s Q) := by
  intro a b h
  have h' : (a.1 : ℕ) + s = (b.1 : ℕ) + s := congrArg Fin.val h
  exact Subtype.ext (Fin.ext (by omega))

/-- Extension by zero from the rows `Tsub`. -/
def extZ : (Tsub n s Q → ZMod 3) →ₗ[ZMod 3] (Fin (n + 1) → ZMod 3) where
  toFun β k := if h : Q (k : ℕ) ∧ (k : ℕ) + s < n + 1 then β ⟨k, h⟩ else 0
  map_add' β γ := by
    funext k
    simp only [Pi.add_apply]
    split_ifs <;> simp
  map_smul' c β := by
    funext k
    simp only [Pi.smul_apply, smul_eq_mul, RingHom.id_apply]
    split_ifs <;> simp

lemma extZ_injective : Function.Injective (extZ n s Q) := by
  intro β γ h
  funext k
  have := congrFun h k.1
  simp only [extZ, LinearMap.coe_mk, AddHom.coe_mk, dif_pos k.2] at this
  exact this

/-- The partial shift `(J α)_k = α_{k+s}` if `Q k` and `k + s ≤ n`, else `0`. -/
def Jop : (Fin (n + 1) → ZMod 3) →ₗ[ZMod 3] (Fin (n + 1) → ZMod 3) :=
  extZ n s Q ∘ₗ LinearMap.funLeft (ZMod 3) (ZMod 3) (shiftIdx n s Q)

lemma Jop_apply (α : Fin (n + 1) → ZMod 3) (k : Fin (n + 1)) :
    Jop n s Q α k = if h : Q (k : ℕ) ∧ (k : ℕ) + s < n + 1 then α ⟨(k : ℕ) + s, h.2⟩ else 0 := rfl

lemma Jop_single (j : Fin (n + 1)) :
    Jop n s Q (Pi.single j 1) = if s ≤ (j : ℕ) ∧ Q ((j : ℕ) - s) then
      Pi.single ⟨(j : ℕ) - s, Nat.lt_of_le_of_lt (Nat.sub_le _ _) j.2⟩ 1 else 0 := by
  funext k
  rw [Jop_apply]
  by_cases hj : s ≤ (j : ℕ) ∧ Q ((j : ℕ) - s)
  · rw [if_pos hj]
    by_cases hk : Q (k : ℕ) ∧ (k : ℕ) + s < n + 1
    · rw [dif_pos hk]
      by_cases e : (k : ℕ) + s = j
      · have e1 : (⟨(k : ℕ) + s, hk.2⟩ : Fin (n + 1)) = j := Fin.ext e
        have e2 : k = ⟨(j : ℕ) - s, Nat.lt_of_le_of_lt (Nat.sub_le _ _) j.2⟩ :=
          Fin.ext (show (k : ℕ) = (j : ℕ) - s by omega)
        rw [e1, Pi.single_eq_same, e2, Pi.single_eq_same]
      · have e1 : (⟨(k : ℕ) + s, hk.2⟩ : Fin (n + 1)) ≠ j := fun h => e (congrArg Fin.val h)
        have e2 : k ≠ ⟨(j : ℕ) - s, Nat.lt_of_le_of_lt (Nat.sub_le _ _) j.2⟩ := by
          intro h
          have : (k : ℕ) = (j : ℕ) - s := congrArg Fin.val h
          omega
        rw [Pi.single_eq_of_ne e1, Pi.single_eq_of_ne e2]
    · rw [dif_neg hk]
      have e2 : k ≠ ⟨(j : ℕ) - s, Nat.lt_of_le_of_lt (Nat.sub_le _ _) j.2⟩ := by
        intro h
        have hv : (k : ℕ) = (j : ℕ) - s := congrArg Fin.val h
        apply hk
        refine ⟨by rw [hv]; exact hj.2, ?_⟩
        have := j.2
        omega
      rw [Pi.single_eq_of_ne e2]
  · rw [if_neg hj, Pi.zero_apply]
    by_cases hk : Q (k : ℕ) ∧ (k : ℕ) + s < n + 1
    · rw [dif_pos hk]
      have e1 : (⟨(k : ℕ) + s, hk.2⟩ : Fin (n + 1)) ≠ j := by
        intro h
        have hv : (k : ℕ) + s = j := congrArg Fin.val h
        apply hj
        refine ⟨by omega, ?_⟩
        rw [show (j : ℕ) - s = k by omega]
        exact hk.1
      rw [Pi.single_eq_of_ne e1]
    · rw [dif_neg hk]

lemma card_Tsub :
    Fintype.card (Tsub n s Q) = ∑ k ∈ Finset.range (n + 1), if Q k ∧ k + s < n + 1 then 1 else 0 := by
  rw [Fintype.card_subtype, Finset.card_filter]
  exact Fin.sum_univ_eq_sum_range (fun k => if Q k ∧ k + s < n + 1 then 1 else 0) (n + 1)

/-- **`rank J = #{k ≤ n : Q k, k + s ≤ n}`.** -/
theorem finrank_range_Jop :
    finrank (ZMod 3) (LinearMap.range (Jop n s Q)) =
      ∑ k ∈ Finset.range (n + 1), if Q k ∧ k + s < n + 1 then 1 else 0 := by
  rw [Jop, LinearMap.range_comp_of_range_eq_top _
      (LinearMap.range_eq_top.mpr (LinearMap.funLeft_surjective_of_injective (R := ZMod 3) (M := ZMod 3) _
        (shiftIdx_injective n s Q))),
    LinearMap.finrank_range_of_inj (extZ_injective n s Q), Module.finrank_fintype_fun_eq_card, card_Tsub]

end Shift

/-! ### Counting residues -/

lemma sum_ite_and_lt (Q : ℕ → Prop) [DecidablePred Q] (s M : ℕ) : ∀ N : ℕ,
    (∑ k ∈ Finset.range N, if Q k ∧ k + s < M then 1 else 0) =
      ∑ k ∈ Finset.range (min N (M - s)), if Q k then 1 else 0
  | 0 => by simp
  | N + 1 => by
    rw [Finset.sum_range_succ, sum_ite_and_lt Q s M N]
    by_cases h : N + s < M
    · rw [show min (N + 1) (M - s) = N + 1 by omega, show min N (M - s) = N by omega, Finset.sum_range_succ]
      by_cases hq : Q N <;> simp [hq, h]
    · rw [show min (N + 1) (M - s) = min N (M - s) by omega]
      simp [h]

lemma cnt_ne2 : ∀ N : ℕ, (∑ k ∈ Finset.range N, if k % 3 ≠ 2 then 1 else 0) = N - N / 3
  | 0 => by simp
  | N + 1 => by
    rw [Finset.sum_range_succ, cnt_ne2 N]
    split_ifs <;> omega

lemma cnt_eq0 : ∀ N : ℕ, (∑ k ∈ Finset.range N, if k % 3 = 0 then 1 else 0) = (N + 2) / 3
  | 0 => by simp
  | N + 1 => by
    rw [Finset.sum_range_succ, cnt_eq0 N]
    split_ifs <;> omega

/-! ### The ranks of `ρ_n(C) − 1` and its square over `F₃`, every `n` (the Jordan type with `D³ = 0`, prose) -/

lemma finrank_range_conj {n : ℕ} {A J : Module.End (ZMod 3) (Fin (n + 1) → ZMod 3)} (h : A ∘ₗ Psi n = Psi n ∘ₗ J) :
    finrank (ZMod 3) (LinearMap.range A) = finrank (ZMod 3) (LinearMap.range J) := by
  rw [← LinearMap.range_comp_of_range_eq_top A (LinearMap.range_eq_top.mpr (Psi_surjective n)), h,
    finrank_range_comp_inj J (Psi n) (Psi_injective n)]

lemma comm1 (n : ℕ) : Dc n ∘ₗ Psi n = Psi n ∘ₗ Jop n 1 (fun k => k % 3 ≠ 2) := by
  refine (Pi.basisFun (ZMod 3) (Fin (n + 1))).ext fun j => ?_
  rw [Pi.basisFun_apply, LinearMap.comp_apply, LinearMap.comp_apply, Psi_single, Dc_equivFun, Dop_cS, Jop_single]
  by_cases h : (j : ℕ) % 3 = 0
  · rw [if_pos h, if_neg (show ¬(1 ≤ (j : ℕ) ∧ ((j : ℕ) - 1) % 3 ≠ 2) by omega), map_zero, map_zero]
  · rw [if_neg h, if_pos (show 1 ≤ (j : ℕ) ∧ ((j : ℕ) - 1) % 3 ≠ 2 by omega), Psi_single]

lemma comm2 (n : ℕ) : (Dc n * Dc n) ∘ₗ Psi n = Psi n ∘ₗ Jop n 2 (fun k => k % 3 = 0) := by
  refine (Pi.basisFun (ZMod 3) (Fin (n + 1))).ext fun j => ?_
  rw [Pi.basisFun_apply, LinearMap.comp_apply, LinearMap.comp_apply, Psi_single, Module.End.mul_apply,
    Dc_equivFun, Dc_equivFun, Dop_sq_cS, Jop_single]
  by_cases h : (j : ℕ) % 3 = 2
  · rw [if_pos h, if_pos (show 2 ≤ (j : ℕ) ∧ ((j : ℕ) - 2) % 3 = 0 by omega), Psi_single]
  · rw [if_neg h, if_neg (show ¬(2 ≤ (j : ℕ) ∧ ((j : ℕ) - 2) % 3 = 0) by omega), map_zero, map_zero]

/-- **`rank (ρ_n(C) − 1) = n − ⌊n/3⌋`** over `F₃`, every `n` (the number of Jordan blocks is `⌊n/3⌋ + 1`). -/
theorem finrank_range_Dc (n : ℕ) : finrank (ZMod 3) (LinearMap.range (Dc n)) = n - n / 3 := by
  rw [finrank_range_conj (comm1 n), finrank_range_Jop, sum_ite_and_lt (fun k => k % 3 ≠ 2) 1 (n + 1) (n + 1),
    show min (n + 1) (n + 1 - 1) = n by omega, cnt_ne2]

/-- **`rank (ρ_n(C) − 1)² = ⌊(n+1)/3⌋`** over `F₃`, every `n` (the number of `J₃` blocks is `⌊n/3⌋`, plus one
`J₃`-like rank from the last block iff `n ≡ 2 mod 3`). -/
theorem finrank_range_Dc_sq (n : ℕ) : finrank (ZMod 3) (LinearMap.range (Dc n * Dc n)) = (n + 1) / 3 := by
  rw [finrank_range_conj (comm2 n), finrank_range_Jop, sum_ite_and_lt (fun k => k % 3 = 0) 2 (n + 1) (n + 1),
    show min (n + 1) (n + 1 - 2) = n - 1 by omega, cnt_eq0]
  omega

/-! ### `ρ_n(U)` for even `n`: conjugate to `ρ_n(C)` -/

/-- `Q = !![1, 0; 1, −1]`, an involution with `Q C Q = −U` in `GL₂(F₃)`. -/
def mQ : Mat2 (ZMod 3) := !![1, 0; 1, -1]

lemma mQ_mul_self : mQ * mQ = 1 := by
  unfold mQ; decide

lemma mQCQ : mQ * mC * mQ = (-1 : ZMod 3) • (1 : Mat2 (ZMod 3)) * mU3 := by
  unfold mQ mC mU3; decide

lemma Lq_invol (n : ℕ) (x : Fin (n + 1) → ZMod 3) :
    Matrix.toLin' (GL2.rhoMat n mQ) (Matrix.toLin' (GL2.rhoMat n mQ) x) = x := by
  simp only [Matrix.toLin'_apply]
  rw [Matrix.mulVec_mulVec, ← GL2.rhoMat_mul, mQ_mul_self, GL2.rhoMat_one, Matrix.one_mulVec]

lemma LU_eq (n : ℕ) (he : Even n) (x : Fin (n + 1) → ZMod 3) :
    Matrix.toLin' (GL2.rhoMat n mU3) x =
      Matrix.toLin' (GL2.rhoMat n mQ) (Matrix.toLin' (GL2.rhoMat n mC) (Matrix.toLin' (GL2.rhoMat n mQ) x)) := by
  have hM : GL2.rhoMat n mU3 = GL2.rhoMat n mQ * GL2.rhoMat n mC * GL2.rhoMat n mQ := by
    rw [← GL2.rhoMat_mul, ← GL2.rhoMat_mul, mQCQ, GL2.rhoMat_mul, GL2.rhoMat_smul_one, he.neg_one_pow, one_smul,
      Matrix.one_mul]
  simp only [Matrix.toLin'_apply]
  rw [hM, ← Matrix.mulVec_mulVec, ← Matrix.mulVec_mulVec]

lemma finrank_range_invol_conj {V : Type} [AddCommGroup V] [Module (ZMod 3) V] (Q A B : Module.End (ZMod 3) V)
    (hQ : ∀ x, Q (Q x) = x) (h : ∀ x, A x = Q (B (Q x))) :
    finrank (ZMod 3) (LinearMap.range A) = finrank (ZMod 3) (LinearMap.range B) := by
  have hA : A = (Q ∘ₗ B) ∘ₗ Q := LinearMap.ext h
  have hs : Function.Surjective Q := fun y => ⟨Q y, hQ y⟩
  have hi : Function.Injective Q := fun a b hab => by rw [← hQ a, hab, hQ b]
  rw [hA, LinearMap.range_comp_of_range_eq_top _ (LinearMap.range_eq_top.mpr hs), finrank_range_comp_inj B Q hi]

lemma DU_apply (n : ℕ) (he : Even n) (x : Fin (n + 1) → ZMod 3) :
    (Matrix.toLin' (GL2.rhoMat n mU3) - 1) x =
      Matrix.toLin' (GL2.rhoMat n mQ) (Dc n (Matrix.toLin' (GL2.rhoMat n mQ) x)) := by
  rw [Dc, LinearMap.sub_apply, LinearMap.sub_apply, Module.End.one_apply, Module.End.one_apply, map_sub,
    Lq_invol, LU_eq n he]

/-- **`rank (ρ_n(U) − 1) = n − ⌊n/3⌋`** on `Sym^n F₃²`, `n` even. -/
theorem finrank_range_DU (n : ℕ) (he : Even n) :
    finrank (ZMod 3) (LinearMap.range (Matrix.toLin' (GL2.rhoMat n mU3) - 1)) = n - n / 3 := by
  rw [finrank_range_invol_conj _ _ (Dc n) (Lq_invol n) (DU_apply n he), finrank_range_Dc]

/-- **`rank (ρ_n(U) − 1)² = ⌊(n+1)/3⌋`** on `Sym^n F₃²`, `n` even. -/
theorem finrank_range_DU_sq (n : ℕ) (he : Even n) :
    finrank (ZMod 3) (LinearMap.range ((Matrix.toLin' (GL2.rhoMat n mU3) - 1) *
      (Matrix.toLin' (GL2.rhoMat n mU3) - 1))) = (n + 1) / 3 := by
  have h : ∀ x, ((Matrix.toLin' (GL2.rhoMat n mU3) - 1) * (Matrix.toLin' (GL2.rhoMat n mU3) - 1)) x =
      Matrix.toLin' (GL2.rhoMat n mQ) ((Dc n * Dc n) (Matrix.toLin' (GL2.rhoMat n mQ) x)) := fun x => by
    rw [Module.End.mul_apply, Module.End.mul_apply, DU_apply n he, DU_apply n he, Lq_invol]
  rw [finrank_range_invol_conj _ _ (Dc n * Dc n) (Lq_invol n) h, finrank_range_Dc_sq]

/-! ### [R] Theorem C(1): the dimensions -/

/-- `x + f x + f² x = (f − 1)² x` in characteristic `3`. -/
lemma char3_sq {V : Type} [AddCommGroup V] [Module (ZMod 3) V] (f : Module.End (ZMod 3) V) (x : V) :
    x + f x + f (f x) = f (f x) - f x - (f x - x) := by
  have key : x + f x + f (f x) - (f (f x) - f x - (f x - x)) = (3 : ZMod 3) • f x := by module
  rw [show (3 : ZMod 3) = 0 by decide, zero_smul, sub_eq_zero] at key
  exact key

/-- `rank (1 − ρ_n(U)) = n − ⌊n/3⌋` on `V₃(n)`, `n` even. -/
theorem finrank_range_difA_V3 (n : ℕ) (he : Even n) :
    finrank (ZMod 3) (LinearMap.range (difA (V3 n) sU)) = n - n / 3 := by
  have e : difA (V3 n) sU = -(Matrix.toLin' (GL2.rhoMat n mU3) - 1) := by
    refine LinearMap.ext fun x => ?_
    rw [difA_apply, V3_rho, mod3_sU, LinearMap.neg_apply, LinearMap.sub_apply, Module.End.one_apply, neg_sub]
  rw [e, LinearMap.range_neg]
  exact finrank_range_DU n he

/-- `1 + ρ_n(U) + ρ_n(U²) = (ρ_n(U) − 1)²` (characteristic `3`), so its rank is `⌊(n+1)/3⌋`, `n` even. -/
theorem finrank_range_nrmA_V3 (n : ℕ) (he : Even n) :
    finrank (ZMod 3) (LinearMap.range (nrmA (V3 n) sU 3)) = (n + 1) / 3 := by
  have e : nrmA (V3 n) sU 3 = (Matrix.toLin' (GL2.rhoMat n mU3) - 1) * (Matrix.toLin' (GL2.rhoMat n mU3) - 1) := by
    refine LinearMap.ext fun x => ?_
    have h2 : (V3 n).ρ (sU ^ 2) x = Matrix.toLin' (GL2.rhoMat n mU3) (Matrix.toLin' (GL2.rhoMat n mU3) x) := by
      rw [pow_two, map_mul, Module.End.mul_apply, V3_rho, mod3_sU]
    rw [nrmA_three', h2, V3_rho, mod3_sU, Module.End.mul_apply, LinearMap.sub_apply, LinearMap.sub_apply,
      Module.End.one_apply, Module.End.one_apply, map_sub]
    exact char3_sq _ x
  rw [e]
  exact finrank_range_DU_sq n he

/-- **[R] Theorem C(1), dimensions, every even `n`**: `dim_{F₃} H^{q+2}(SL₂(ℤ); Sym^n F₃²) = [n ≢ 2 mod 3]`. -/
theorem V3_dims_ge2 (n : ℕ) (hn : n % 2 = 0) (q : ℕ) :
    finrank (ZMod 3) (groupCohomology (V3 n) (q + 2)) = if n % 3 = 2 then 0 else 1 := by
  have he : Even n := Nat.even_iff.mpr hn
  obtain ⟨-, -, d2⟩ := finrank_H_of_trivNegOne (V3 n) two_ne_zero_zmod3 (V3_negOne_even he)
  have h := d2 q
  rw [finrank_range_nrmA_V3 n he, finrank_range_difA_V3 n he, finrank_V3] at h
  split_ifs with h3 <;> omega

/-- **[R] Theorem C(1)**, as stated there: `h^q = [n ≡ 0, 4 mod 6]` for `q ≥ 2`, `n` even. -/
theorem V3_dims_thmC (n : ℕ) (hn : n % 2 = 0) (q : ℕ) :
    finrank (ZMod 3) (groupCohomology (V3 n) (q + 2)) = if n % 6 = 0 ∨ n % 6 = 4 then 1 else 0 := by
  rw [V3_dims_ge2 n hn q]
  split_ifs <;> omega

end GL2.L23.Jordan3
