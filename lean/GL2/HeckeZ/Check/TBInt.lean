/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import Mathlib
import GL2.HeckeZ.Exec.TBInt

/-!
# HeckeZ TB-int: soundness of the integer-matrix checks (L-HZ4, M-HZ5)

`07-lean-plan.md` §1: **TB-int** is the trust input "these integer matrices are the operators on `L` in one `ℤ`-basis"; here
it appears as `LinearMap.toMatrix b b f = lmat n n F`. The list checks of `GL2.HeckeZ.Exec.TBInt` then give identities in
`End_ℤ L`, eigenvectors, and a rank bound from a factorisation (`rank_le_of_factor`).
-/

namespace GL2.HeckeZ.Check.TBInt

open GL2.HeckeZ.Exec.TBInt

/-- The `a × b` integer matrix of a list matrix. -/
def lmat (a b : ℕ) (M : IMat) : Matrix (Fin a) (Fin b) ℤ := Matrix.of fun i j => ient M i j

lemma foldr_add_eq_sum_map (l : List ℕ) (g : ℕ → ℤ) : (l.map g).foldr (· + ·) 0 = (l.map g).sum := by
  rw [List.sum_eq_foldr]

lemma list_sum_range_map_int (g : ℕ → ℤ) (k : ℕ) :
    ((List.range k).map g).sum = ∑ i ∈ Finset.range k, g i := by
  induction k with
  | zero => simp
  | succ k ih => rw [List.range_succ, List.map_append, List.sum_append, ih, Finset.sum_range_succ]; simp

lemma ient_grid {a b : ℕ} (g : ℕ → ℕ → ℤ) {i j : ℕ} (hi : i < a) (hj : j < b) :
    ient ((List.range a).map fun i => (List.range b).map fun j => g i j) i j = g i j := by
  simp [ient, List.getD_eq_getElem?_getD, hi, hj]

lemma lmat_imul (a b m : ℕ) (A B : IMat) : lmat a b (imul a b m A B) = lmat a m A * lmat m b B := by
  ext i j
  simp only [lmat, Matrix.of_apply, imul, Matrix.mul_apply]
  rw [ient_grid (fun i j => imulEnt m A B i j) i.2 j.2, imulEnt, foldr_add_eq_sum_map, list_sum_range_map_int]
  exact (Fin.sum_univ_eq_sum_range (fun l => ient A i l * ient B l j) m).symm

lemma lmat_ishift (n : ℕ) (A : IMat) (c : ℤ) : lmat n n (ishift n A c) = lmat n n A - (c : Matrix (Fin n) (Fin n) ℤ) := by
  ext i j
  simp only [lmat, Matrix.of_apply, ishift, Matrix.sub_apply]
  rw [ient_grid (fun i j => ient A i j - (if i == j then c else 0)) i.2 j.2]
  by_cases hij : i = j
  · subst hij; simp [Matrix.intCast_apply]
  · have : (i : ℕ) ≠ j := fun h => hij (Fin.ext h)
    simp [Matrix.intCast_apply, hij, this]

lemma lmat_iadd (a b : ℕ) (A B : IMat) : lmat a b (iadd a b A B) = lmat a b A + lmat a b B := by
  ext i j
  simp only [lmat, Matrix.of_apply, iadd, Matrix.add_apply]
  rw [ient_grid (fun i j => ient A i j + ient B i j) i.2 j.2]

lemma lmat_eq_of_ieq {a b : ℕ} {A B : IMat} (h : ieq a b A B = true) : lmat a b A = lmat a b B := by
  ext i j
  simp only [ieq, List.all_eq_true, List.mem_range] at h
  simpa [lmat] using h i i.2 j j.2

variable {L : Type*} [AddCommGroup L] [Module.Free ℤ L] [Module.Finite ℤ L]

/-- The vector of a column list `v`, `Fin n → ℤ`. -/
def vecOf (n : ℕ) (v : List Int) : Fin n → ℤ := fun i => ient (icol v) i 0

/-- The element of `L` with coordinates `v` in the basis `b`. -/
noncomputable def vecL {n : ℕ} (b : Module.Basis (Fin n) ℤ L) (v : List Int) : L := b.equivFun.symm (vecOf n v)

lemma repr_vecL {n : ℕ} (b : Module.Basis (Fin n) ℤ L) (v : List Int) (i : Fin n) : b.repr (vecL b v) i = ient (icol v) i 0 := by
  have h : b.equivFun (vecL b v) = vecOf n v := b.equivFun.apply_symm_apply _
  rw [Module.Basis.equivFun_apply] at h
  exact congrFun h i

/-- A non-zero column gives a non-zero vector. -/
lemma vecL_ne_zero {n : ℕ} (b : Module.Basis (Fin n) ℤ L) {v : List Int}
    (h : ((List.range n).any fun i => ient (icol v) i 0 != 0) = true) : vecL b v ≠ 0 := by
  intro h0
  simp only [List.any_eq_true, List.mem_range, bne_iff_ne, ne_eq] at h
  obtain ⟨i, hi, hne⟩ := h
  apply hne
  have := repr_vecL b v ⟨i, hi⟩
  rw [h0, map_zero, Finsupp.zero_apply] at this
  exact this.symm

lemma ient_icol_map (c : ℤ) (v : List Int) (i : ℕ) : ient (icol (v.map (c * ·))) i 0 = c * ient (icol v) i 0 := by
  induction v generalizing i with
  | nil => simp [ient, icol]
  | cons x xs ih =>
    cases i with
    | zero => simp [ient, icol]
    | succ i => simpa [ient, icol] using ih i

/-- **Eigenvector from the check** (TB-int): `F v = c v` for the column `v` gives `f (vecL v) = c • vecL v` on `L`. -/
theorem eigen_of_check {n : ℕ} (b : Module.Basis (Fin n) ℤ L) {f : Module.End ℤ L} {F : IMat}
    (hmat : LinearMap.toMatrix b b f = lmat n n F) {v : List Int} {c : ℤ} (h : ieigen n F v c = true) :
    f (vecL b v) = c • vecL b v := by
  have hE := lmat_eq_of_ieq h
  rw [lmat_imul] at hE
  apply b.ext_elem
  intro i
  have h1 := congrFun (congrFun hE i) 0
  have h2 : (LinearMap.toMatrix b b f).mulVec (b.repr (vecL b v)) i = b.repr (f (vecL b v)) i := by
    have := congrFun (LinearMap.toMatrix_mulVec_repr b b f (vecL b v)) i
    simpa using this
  rw [← h2, hmat, LinearEquiv.map_smul, Finsupp.smul_apply, smul_eq_mul, repr_vecL]
  rw [Matrix.mulVec, dotProduct]
  rw [Matrix.mul_apply] at h1
  simp only [lmat, Matrix.of_apply] at h1 ⊢
  have e1 : ∑ j : Fin n, ient F i j * b.repr (vecL b v) j = ∑ j : Fin n, ient F i j * ient (icol v) j 0 := by
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [repr_vecL]
  rw [e1]
  have h1' : ∑ j : Fin n, ient F i j * ient (icol v) j 0 = ient (icol (v.map (c * ·))) i 0 := h1
  rw [h1', ient_icol_map]

/-- `toMatrix` of `f − c`. -/
lemma toMatrix_shift {n : ℕ} (b : Module.Basis (Fin n) ℤ L) {f : Module.End ℤ L} {F : IMat}
    (hmat : LinearMap.toMatrix b b f = lmat n n F) (c : ℤ) :
    LinearMap.toMatrix b b (f - (c : Module.End ℤ L)) = lmat n n (ishift n F c) := by
  have h1 : LinearMap.toMatrix b b ((c : Module.End ℤ L)) = (c : Matrix (Fin n) (Fin n) ℤ) := by
    have : (c : Module.End ℤ L) = (c • (1 : Module.End ℤ L)) := by rw [zsmul_eq_mul, mul_one]
    rw [this, map_zsmul, LinearMap.toMatrix_one, zsmul_eq_mul, mul_one]
  rw [map_sub, hmat, lmat_ishift, h1]

/-- `toMatrix` of a product. -/
lemma toMatrix_mul' {n : ℕ} (b : Module.Basis (Fin n) ℤ L) {f g : Module.End ℤ L} {F G : IMat}
    (hf : LinearMap.toMatrix b b f = lmat n n F) (hg : LinearMap.toMatrix b b g = lmat n n G) :
    LinearMap.toMatrix b b (f * g) = lmat n n (imul n n n F G) := by
  rw [LinearMap.toMatrix_mul, hf, hg, lmat_imul]

/-- `f = 0` from `F = 0` by check (`0` the zero list). -/
lemma eq_zero_of_check {n : ℕ} (b : Module.Basis (Fin n) ℤ L) {f : Module.End ℤ L} {F : IMat}
    (hmat : LinearMap.toMatrix b b f = lmat n n F) (h : ieq n n F [] = true) : f = 0 := by
  apply (LinearMap.toMatrix b b).injective
  rw [hmat, map_zero]
  have hE := lmat_eq_of_ieq h
  rw [hE]
  ext i j; simp [lmat, ient]

/-- **A rank bound from a factorisation**: `F · F = Q R` with `Q` `n × k`, `R` `k × n` gives
`rank (f ∘ f)(L) ≤ k` (TB-int). -/
theorem rank_le_of_factor {n k : ℕ} (b : Module.Basis (Fin n) ℤ L) {f : Module.End ℤ L} {F Q R : IMat}
    (hmat : LinearMap.toMatrix b b f = lmat n n F) (h : ieq n n (imul n n n F F) (imul n n k Q R) = true) :
    Module.finrank ℤ (LinearMap.range (f ^ 2)) ≤ k := by
  classical
  have hE := lmat_eq_of_ieq h
  rw [lmat_imul, lmat_imul, ← hmat] at hE
  let g1 : (Fin k → ℤ) →ₗ[ℤ] L := Matrix.toLin (Pi.basisFun ℤ (Fin k)) b (lmat n k Q)
  let g2 : L →ₗ[ℤ] (Fin k → ℤ) := Matrix.toLin b (Pi.basisFun ℤ (Fin k)) (lmat k n R)
  have hfac : f ^ 2 = g1 ∘ₗ g2 := by
    apply (LinearMap.toMatrix b b).injective
    rw [LinearMap.toMatrix_comp b (Pi.basisFun ℤ (Fin k)) b, LinearMap.toMatrix_toLin, LinearMap.toMatrix_toLin, pow_two,
      LinearMap.toMatrix_mul, hE]
  rw [hfac]
  have hle : LinearMap.range (g1 ∘ₗ g2) ≤ LinearMap.range g1 := LinearMap.range_comp_le_range _ _
  calc Module.finrank ℤ (LinearMap.range (g1 ∘ₗ g2)) ≤ Module.finrank ℤ (LinearMap.range g1) :=
        Submodule.finrank_mono hle
    _ ≤ Module.finrank ℤ (Fin k → ℤ) := LinearMap.finrank_range_le g1
    _ = k := by simp

end GL2.HeckeZ.Check.TBInt
