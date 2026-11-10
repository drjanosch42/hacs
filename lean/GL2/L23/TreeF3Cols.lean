/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import Mathlib
import GL2.L23.RankF3
import GL2.Checkers.RhoExec

/-!
# LV-2 (slot 22): the two-plane columns are the columns of `ρ_n` mod 3 (the F1 precondition at ℓ = 3)

The analogue of `GL2/L23/TreeF2Cols.lean` (`LV0_audit.md` R1) over `ZMod 3`, against [C] `eq:rhondef`
(`GL2.Exec.rhoEntry`, tied to `GL2.rhoMat` by `GL2.Exec.rhoMat_eq`):

* `val3_qrow`: entry `t` of `Q_m = (z − 1)^m` is `C(m, t) (−1)^(m+t)` (Pascal's rule; no Lucas theorem);
* `rhoEntry_a0c1`, `rhoEntry_c1d0`: the double sum of `eq:rhondef` for `A = !![0, b; 1, d]` and `!![a, b; 1, 0]`;
* **`vec3_colS`, `vec3_colU`, `vec3_colU2`** (F1): the two-plane column `j` of `ρ_n(S)`, `ρ_n(U)`, `ρ_n(U²)` is column `j` of
  `GL2.rhoMat n A` over `ZMod 3` for `S ≡ !![0,−1;1,0]`, `U ≡ !![0,−1;1,1]`, `U² ≡ !![−1,−1;1,0]`, every `n`, `j ≤ n`;
* `colsPS_eq`, `colsMU_eq`, `colsN3_eq`, `colsSig3_eq`: the kernel generators are `List.map … (List.range (n+1))`;
* the rows have disjoint planes below `2^(n+1)` (resp. `2^(2(n+1))`) for every `n` (`rows_colsPS`, …), so a cell needs only
  its rank decisions.
-/

namespace GL2.L23.RankF3

open GL2.Exec

/-! ### Entries of the building blocks -/

lemma val3_e3 (k t : ℕ) : val3 (e3 k) t = if t = k then 1 else 0 := by
  simp only [val3, e3]
  change (if (1 <<< k).testBit t then (1 : ZMod 3) else if (0 : ℕ).testBit t then -1 else 0) = _
  rw [Nat.one_shiftLeft, Nat.testBit_two_pow, Nat.zero_testBit]
  by_cases h : t = k
  · subst h; simp
  · simp [h, Ne.symm h]

lemma disj_e3 (k : ℕ) : Disj (e3 k) := fun j ⟨_, h⟩ => by simp [e3] at h

lemma low_e3 {k P : ℕ} (h : k < P) : Low P (e3 k) := by
  intro j hj
  refine ⟨?_, by simp [e3]⟩
  change (1 <<< k).testBit j = false
  rw [Nat.one_shiftLeft, Nat.testBit_two_pow]
  simp only [decide_eq_false_iff_not]; omega

lemma testBit_shl3 (x : ℕ × ℕ) (k t : ℕ) :
    (shl3 x k).1.testBit t = (decide (k ≤ t) && x.1.testBit (t - k)) ∧
      (shl3 x k).2.testBit t = (decide (k ≤ t) && x.2.testBit (t - k)) := by
  constructor <;> exact Nat.testBit_shiftLeft _

lemma val3_shl3 (x : ℕ × ℕ) (k t : ℕ) : val3 (shl3 x k) t = if k ≤ t then val3 x (t - k) else 0 := by
  obtain ⟨h1, h2⟩ := testBit_shl3 x k t
  simp only [val3, h1, h2]
  by_cases h : k ≤ t <;> simp [h]

lemma disj_shl3 {x : ℕ × ℕ} (hx : Disj x) (k : ℕ) : Disj (shl3 x k) := by
  intro t ⟨h1, h2⟩
  rw [(testBit_shl3 x k t).1] at h1
  rw [(testBit_shl3 x k t).2] at h2
  simp only [Bool.and_eq_true] at h1 h2
  exact hx _ ⟨h1.2, h2.2⟩

lemma low_shl3 {P : ℕ} {x : ℕ × ℕ} (hx : Low P x) (k : ℕ) : Low (P + k) (shl3 x k) := by
  intro t ht
  rw [(testBit_shl3 x k t).1, (testBit_shl3 x k t).2, (hx (t - k) (by omega)).1, (hx (t - k) (by omega)).2]
  simp

lemma sgn3_eq (j : ℕ) (x : ℕ × ℕ) : sgn3 j x = if j % 2 = 0 then x else neg3 x := by
  unfold sgn3
  rw [show Nat.land j 1 = j % 2 from Nat.and_one_is_mod j]
  by_cases h : j % 2 = 0
  · rw [if_pos h, h]; rfl
  · rw [if_neg h, show Nat.beq (j % 2) 0 = false from Bool.eq_false_iff.mpr fun h' => h (Nat.eq_of_beq_eq_true h')]

lemma neg_one_pow_of_mod (j : ℕ) : (-1 : ZMod 3) ^ j = if j % 2 = 0 then 1 else -1 := by
  rcases Nat.even_or_odd j with h | h
  · rw [h.neg_one_pow, if_pos (Nat.even_iff.mp h)]
  · rw [h.neg_one_pow, if_neg (by rw [Nat.odd_iff] at h; omega)]

lemma val3_sgn3 {x : ℕ × ℕ} (hx : Disj x) (j t : ℕ) : val3 (sgn3 j x) t = (-1) ^ j * val3 x t := by
  rw [sgn3_eq, neg_one_pow_of_mod]
  split_ifs
  · rw [one_mul]
  · rw [val3_neg3 hx, neg_one_mul]

lemma disj_sgn3 {x : ℕ × ℕ} (hx : Disj x) (j : ℕ) : Disj (sgn3 j x) := by
  rw [sgn3_eq]; split_ifs; exacts [hx, disj_neg3 hx]

lemma low_sgn3 {P : ℕ} {x : ℕ × ℕ} (hx : Low P x) (j : ℕ) : Low P (sgn3 j x) := by
  rw [sgn3_eq]; split_ifs; exacts [hx, low_neg3 hx]

/-! ### `Q_m = (z − 1)^m` -/

lemma qrow_zero : qrow 0 = (1, 0) := rfl

lemma qrow_succ (m : ℕ) : qrow (m + 1) = add3 (shl3 (qrow m) 1) (neg3 (qrow m)) := rfl

lemma disj_low_qrow (m : ℕ) : Disj (qrow m) ∧ Low (m + 1) (qrow m) := by
  induction m with
  | zero =>
    refine ⟨fun j ⟨_, h⟩ => by simp [qrow_zero] at h, fun j hj => ?_⟩
    rw [qrow_zero]
    refine ⟨?_, by simp⟩
    simp only
    rw [Nat.testBit_one_eq_true_iff_self_eq_zero.not.mpr (by omega) |> Bool.eq_false_iff.mpr]
  | succ m ih =>
    rw [qrow_succ]
    exact ⟨disj_add3 (disj_shl3 ih.1 1) (disj_neg3 ih.1),
      low_add3 (low_shl3 ih.2 1) (low_mono (by omega) (low_neg3 ih.2))⟩

/-- **Entry `t` of `Q_m` is `C(m, t)(−1)^(m+t)`.** -/
theorem val3_qrow (m t : ℕ) : val3 (qrow m) t = (m.choose t : ZMod 3) * (-1) ^ (m + t) := by
  induction m generalizing t with
  | zero =>
    rw [qrow_zero]
    cases t with
    | zero => simp [val3]
    | succ t =>
      simp only [val3, Nat.choose_zero_succ, Nat.cast_zero, zero_mul]
      rw [Nat.testBit_one_eq_true_iff_self_eq_zero.not.mpr (by omega) |> Bool.eq_false_iff.mpr]
      simp
  | succ m ih =>
    have hd := (disj_low_qrow m).1
    rw [qrow_succ, val3_add3 (disj_shl3 hd 1) (disj_neg3 hd), val3_shl3, val3_neg3 hd]
    cases t with
    | zero => simp [ih]; ring
    | succ t =>
      rw [if_pos (by omega), show t + 1 - 1 = t from rfl, ih, ih, Nat.choose_succ_succ, Nat.cast_add]
      ring

/-! ### Closed forms of `rhoEntry` -/

section Rho

variable {R : Type*} [CommRing R]

/-- `ρ_n(!![0, b; 1, d])`: `u_j ↦ Σ_r C(j, r) b^(j−r) d^r u_{n−j+r}`. -/
lemma rhoEntry_a0c1 (n j t : ℕ) (b d : R) (hj : j ≤ n) :
    rhoEntry n (0 : R) b 1 d t j =
      if n - j ≤ t then (j.choose (t - (n - j)) : R) * b ^ (j - (t - (n - j))) * d ^ (t - (n - j)) else 0 := by
  unfold rhoEntry
  rw [Finset.sum_eq_single (n - j)]
  · simp only [Nat.choose_self, Nat.sub_self, pow_zero, one_pow, mul_one, Nat.cast_one, one_mul]
    split_ifs with h
    · by_cases hr : t - (n - j) < j + 1
      · rw [Finset.sum_eq_single (t - (n - j))]
        · rw [if_pos (by omega)]
        · intro r _ hr'; rw [if_neg]; omega
        · intro h'; exact absurd (Finset.mem_range.mpr hr) h'
      · rw [Nat.choose_eq_zero_of_lt (by omega), Nat.cast_zero, zero_mul, zero_mul]
        refine Finset.sum_eq_zero fun r hr' => ?_
        rw [if_neg]; have := Finset.mem_range.mp hr'; omega
    · refine Finset.sum_eq_zero fun r _ => ?_
      rw [if_neg]; omega
  · intro s hs hsn
    have : n - j - s ≠ 0 := by have := Finset.mem_range.mp hs; omega
    refine Finset.sum_eq_zero fun r _ => ?_
    split_ifs <;> simp [zero_pow this]
  · intro h; simp at h

/-- `ρ_n(!![a, b; 1, 0])`: `u_j ↦ Σ_t C(n−j, t) a^(n−j−t) b^j u_t`. -/
lemma rhoEntry_c1d0 (n j t : ℕ) (a b : R) :
    rhoEntry n a b 1 (0 : R) t j = ((n - j).choose t : R) * a ^ (n - j - t) * b ^ j := by
  unfold rhoEntry
  have hin : ∀ s ∈ Finset.range (n - j + 1), (∑ r ∈ Finset.range (j + 1),
      if s + r = t then ((n - j).choose s : R) * a ^ (n - j - s) * 1 ^ s * (j.choose r : R) * b ^ (j - r) * (0 : R) ^ r
      else 0) = if s = t then ((n - j).choose s : R) * a ^ (n - j - s) * b ^ j else 0 := by
    intro s _
    rw [Finset.sum_eq_single 0]
    · simp
    · intro r _ hr; split_ifs <;> simp [zero_pow hr]
    · intro h; simp at h
  rw [Finset.sum_congr rfl hin, Finset.sum_ite_eq']
  split_ifs with h
  · rfl
  · rw [Nat.choose_eq_zero_of_lt (by simp at h; omega)]; simp

lemma rhoMat_apply' (n : ℕ) (A : Mat2 R) (t j : Fin (n + 1)) :
    GL2.rhoMat n A t j = rhoEntry n (A 0 0) (A 0 1) (A 1 0) (A 1 1) t j := by
  rw [← GL2.Exec.rhoMat_eq]; rfl

lemma neg_one_pow_sub_eq_add (a b : ℕ) (h : b ≤ a) : (-1 : R) ^ (a - b) = (-1) ^ (a + b) := by
  rw [show a + b = (a - b) + 2 * b by omega, pow_add, pow_mul, neg_one_sq, one_pow, mul_one]

end Rho

/-! ### F1 over `ZMod 3` -/

/-- `S ≡ !![0,−1;1,0]`, `U ≡ !![0,−1;1,1]`, `U² ≡ !![−1,−1;1,0]` over `ZMod 3`. -/
def mS3 : Mat2 (ZMod 3) := !![0, -1; 1, 0]
def mU3 : Mat2 (ZMod 3) := !![0, -1; 1, 1]
def mUU3 : Mat2 (ZMod 3) := !![-1, -1; 1, 0]

/-- **F1 for `S`**: the column `j` of `ρ_n(S)` mod 3 is `(−1)^j u_{n−j}`. -/
theorem vec3_colS (n : ℕ) (j : Fin (n + 1)) : vec3 (n + 1) (sgn3 j (e3 (n - j))) = (GL2.rhoMat n mS3).col j := by
  funext t
  have hj := Nat.lt_succ_iff.mp j.2
  simp only [vec3, Matrix.col_apply, rhoMat_apply', mS3]
  simp only [Matrix.of_apply, Matrix.cons_val', Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.empty_val',
    Matrix.cons_val_fin_one, Matrix.head_cons]
  rw [val3_sgn3 (disj_e3 _), val3_e3, rhoEntry_a0c1 n j t (-1) 0 hj]
  by_cases h : (t : ℕ) = n - j
  · rw [if_pos h, if_pos (by omega), h, Nat.sub_self]; simp
  · rw [if_neg h, mul_zero]
    split_ifs with h'
    · rw [zero_pow (by omega), mul_zero]
    · rfl

/-- **F1 for `U`**: the column `j` of `ρ_n(U)` mod 3 is `z^{n−j} Q_j`. -/
theorem vec3_colU (n : ℕ) (j : Fin (n + 1)) : vec3 (n + 1) (shl3 (qrow j) (n - j)) = (GL2.rhoMat n mU3).col j := by
  funext t
  have hj := Nat.lt_succ_iff.mp j.2
  simp only [vec3, Matrix.col_apply, rhoMat_apply', mU3]
  simp only [Matrix.of_apply, Matrix.cons_val', Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.empty_val',
    Matrix.cons_val_fin_one, Matrix.head_cons]
  rw [val3_shl3, rhoEntry_a0c1 n j t (-1) 1 hj]
  split_ifs with h
  · rw [val3_qrow, one_pow, mul_one]
    by_cases hr : t - (n - j) ≤ j
    · rw [neg_one_pow_sub_eq_add _ _ hr]
    · rw [Nat.choose_eq_zero_of_lt (by omega)]; simp
  · rfl

/-- **F1 for `U²`**: the column `j` of `ρ_n(U²)` mod 3 is `(−1)^j Q_{n−j}`. -/
theorem vec3_colU2 (n : ℕ) (j : Fin (n + 1)) :
    vec3 (n + 1) (sgn3 j (qrow (n - j))) = (GL2.rhoMat n mUU3).col j := by
  funext t
  simp only [vec3, Matrix.col_apply, rhoMat_apply', mUU3]
  simp only [Matrix.of_apply, Matrix.cons_val', Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.empty_val',
    Matrix.cons_val_fin_one, Matrix.head_cons]
  rw [val3_sgn3 (disj_low_qrow _).1, val3_qrow, rhoEntry_c1d0]
  by_cases ht : (t : ℕ) ≤ n - j
  · rw [neg_one_pow_sub_eq_add _ _ ht]; ring
  · rw [Nat.choose_eq_zero_of_lt (by omega)]; simp

/-! ### The kernel generators, read back -/

lemma tab3_eq (f : ℕ → ℕ × ℕ) (k j : ℕ) : tab3 f k j = (List.range k).map fun i => f (j + i) := by
  induction k generalizing j with
  | zero => rfl
  | succ k ih =>
    show force2 (f j) (fun c => c :: tab3 f k (j + 1)) = _
    rw [force2_eq, ih, List.range_succ_eq_map, List.map_cons, List.map_map]
    congr 1
    refine List.map_congr_left fun i _ => ?_
    simp only [Function.comp_apply]; congr 1; omega

lemma qGo_qrow (k j : ℕ) : qGo k (qrow j) = (List.range k).map fun i => qrow (j + i) := by
  induction k generalizing j with
  | zero => rfl
  | succ k ih =>
    show qrow j :: force2 (qNext (qrow j)) (qGo k) = _
    rw [force2_eq, show qNext (qrow j) = qrow (j + 1) from rfl, ih, List.range_succ_eq_map, List.map_cons, List.map_map]
    congr 1
    refine List.map_congr_left fun i _ => ?_
    simp only [Function.comp_apply]; congr 1; omega

lemma qGo_one (k : ℕ) : qGo k (1, 0) = (List.range k).map qrow := by
  rw [show ((1, 0) : ℕ × ℕ) = qrow 0 from rfl, qGo_qrow]; simp

lemma revGo3_eq (l acc : List (ℕ × ℕ)) : revGo3 l acc = l.reverse ++ acc := by
  induction l generalizing acc with
  | nil => rfl
  | cons x l ih => show revGo3 l (x :: acc) = _; rw [ih]; simp

lemma mapQ_eq (g : ℕ → ℕ × ℕ → ℕ × ℕ) (P : List (ℕ × ℕ)) (j : ℕ) :
    mapQ g P j = (List.range P.length).map fun i => g (j + i) (P.getD i (0, 0)) := by
  induction P generalizing j with
  | nil => rfl
  | cons p P ih =>
    show force2 _ (fun c => c :: mapQ g P (j + 1)) = _
    rw [force2_eq, ih (j + 1), List.length_cons, List.range_succ_eq_map, List.map_cons, List.map_map]
    congr 1
    refine List.map_congr_left fun i _ => ?_
    simp only [Function.comp_apply, List.getD_cons_succ]; congr 1; omega

lemma mapQQ_eq (g : ℕ → ℕ × ℕ → ℕ × ℕ → ℕ × ℕ) (P Q : List (ℕ × ℕ)) (j : ℕ) (h : P.length = Q.length) :
    mapQQ g P Q j = (List.range P.length).map fun i => g (j + i) (P.getD i (0, 0)) (Q.getD i (0, 0)) := by
  induction P generalizing Q j with
  | nil => rfl
  | cons p P ih =>
    cases Q with
    | nil => simp at h
    | cons q Q =>
      show force2 _ (fun c => c :: mapQQ g P Q (j + 1)) = _
      rw [force2_eq, ih Q (j + 1) (by simpa using h), List.length_cons, List.range_succ_eq_map, List.map_cons,
        List.map_map]
      congr 1
      refine List.map_congr_left fun i _ => ?_
      simp only [Function.comp_apply, List.getD_cons_succ]; congr 1; omega

lemma qList_getD (n i : ℕ) (hi : i < n + 1) : (qGo (n + 1) (1, 0)).getD i (0, 0) = qrow i := by
  rw [qGo_one, List.getD_eq_getElem _ _ (by simpa using hi)]; simp

lemma qList_rev_getD (n i : ℕ) (hi : i < n + 1) :
    (revGo3 (qGo (n + 1) (1, 0)) []).getD i (0, 0) = qrow (n - i) := by
  rw [revGo3_eq, List.append_nil, qGo_one, List.getD_eq_getElem _ _ (by simpa using hi), List.getElem_reverse]
  simp only [List.getElem_map, List.getElem_range, List.length_map, List.length_range]
  congr 1

lemma qList_length (n : ℕ) : (qGo (n + 1) (1, 0)).length = n + 1 := by rw [qGo_one]; simp

theorem colsPS_eq (n : ℕ) : colsPS n = (List.range (n + 1)).map (colPS n) := by
  unfold colsPS; rw [tab3_eq]; simp

theorem colsMU_eq (n : ℕ) : colsMU n = (List.range (n + 1)).map fun j => colMU n j (qrow j) := by
  unfold colsMU
  rw [mapQ_eq, qList_length]
  refine List.map_congr_left fun i hi => ?_
  rw [zero_add, qList_getD n i (List.mem_range.mp hi)]

theorem colsN3_eq (n : ℕ) : colsN3 n = (List.range (n + 1)).map fun j => colN3 n j (qrow j) (qrow (n - j)) := by
  unfold colsN3
  rw [mapQQ_eq _ _ _ _ (by rw [revGo3_eq, qGo_one]; simp), qList_length]
  refine List.map_congr_left fun i hi => ?_
  have hi' := List.mem_range.mp hi
  rw [zero_add, qList_getD n i hi', qList_rev_getD n i hi']

theorem colsSig3_eq (n : ℕ) : colsSig3 n = (List.range (n + 1)).map fun j => colSig3 n j (qrow j) := by
  unfold colsSig3
  rw [mapQ_eq, qList_length]
  refine List.map_congr_left fun i hi => ?_
  rw [zero_add, qList_getD n i (List.mem_range.mp hi)]

/-! ### Row bounds for every `n` -/

lemma colPS_ok (n j : ℕ) (hj : j ≤ n) : Disj (colPS n j) ∧ Low (n + 1) (colPS n j) :=
  ⟨disj_add3 (disj_e3 _) (disj_sgn3 (disj_e3 _) _),
    low_add3 (low_e3 (by omega)) (low_sgn3 (low_e3 (by omega)) _)⟩

lemma colMS_ok (n j : ℕ) (hj : j ≤ n) : Disj (colMS n j) ∧ Low (n + 1) (colMS n j) :=
  ⟨disj_add3 (disj_e3 _) (disj_neg3 (disj_sgn3 (disj_e3 _) _)),
    low_add3 (low_e3 (by omega)) (low_neg3 (low_sgn3 (low_e3 (by omega)) _))⟩

lemma shl3_qrow_ok (n j : ℕ) (hj : j ≤ n) : Disj (shl3 (qrow j) (n - j)) ∧ Low (n + 1) (shl3 (qrow j) (n - j)) :=
  ⟨disj_shl3 (disj_low_qrow j).1 _,
    low_mono (by omega) (low_shl3 (disj_low_qrow j).2 (n - j))⟩

lemma colMU_ok (n j : ℕ) (hj : j ≤ n) : Disj (colMU n j (qrow j)) ∧ Low (n + 1) (colMU n j (qrow j)) :=
  ⟨disj_add3 (disj_e3 _) (disj_neg3 (shl3_qrow_ok n j hj).1),
    low_add3 (low_e3 (by omega)) (low_neg3 (shl3_qrow_ok n j hj).2)⟩

lemma colN3_ok (n j : ℕ) (hj : j ≤ n) :
    Disj (colN3 n j (qrow j) (qrow (n - j))) ∧ Low (n + 1) (colN3 n j (qrow j) (qrow (n - j))) :=
  ⟨disj_add3 (disj_add3 (disj_e3 _) (shl3_qrow_ok n j hj).1) (disj_sgn3 (disj_low_qrow _).1 _),
    low_add3 (low_add3 (low_e3 (by omega)) (shl3_qrow_ok n j hj).2)
      (low_sgn3 (low_mono (by omega) (disj_low_qrow _).2) _)⟩

/-- The stacked column at the low positions, the high positions, and its bounds. -/
lemma colSig3_testBit (n j : ℕ) (q : ℕ × ℕ) (hlo : Low (n + 1) (colMS n j)) (t : ℕ) :
    ((colSig3 n j q).1.testBit t = if t < n + 1 then (colMS n j).1.testBit t else (colMU n j q).1.testBit (t - (n + 1))) ∧
    ((colSig3 n j q).2.testBit t = if t < n + 1 then (colMS n j).2.testBit t else (colMU n j q).2.testBit (t - (n + 1))) := by
  have e1 : (colSig3 n j q).1 = (colMS n j).1 ||| ((colMU n j q).1 <<< (n + 1)) := rfl
  have e2 : (colSig3 n j q).2 = (colMS n j).2 ||| ((colMU n j q).2 <<< (n + 1)) := rfl
  rw [e1, e2, Nat.testBit_lor, Nat.testBit_lor, Nat.testBit_shiftLeft, Nat.testBit_shiftLeft]
  by_cases ht : t < n + 1
  · simp [ht, show ¬ (n + 1 ≤ t) by omega]
  · simp [ht, show n + 1 ≤ t by omega, (hlo t (by omega)).1, (hlo t (by omega)).2]

lemma colSig3_ok (n j : ℕ) (hj : j ≤ n) :
    Disj (colSig3 n j (qrow j)) ∧ Low ((n + 1) + (n + 1)) (colSig3 n j (qrow j)) := by
  have hl := colMS_ok n j hj
  have hu := colMU_ok n j hj
  refine ⟨fun t ⟨h1, h2⟩ => ?_, fun t ht => ?_⟩
  · rw [(colSig3_testBit n j _ hl.2 t).1] at h1
    rw [(colSig3_testBit n j _ hl.2 t).2] at h2
    split_ifs at h1 h2
    · exact hl.1 t ⟨h1, h2⟩
    · exact hu.1 _ ⟨h1, h2⟩
  · rw [(colSig3_testBit n j _ hl.2 t).1, (colSig3_testBit n j _ hl.2 t).2, if_neg (by omega), if_neg (by omega)]
    exact hu.2 _ (by omega)

lemma val3_colSig3_lo (n j : ℕ) (q : ℕ × ℕ) (hlo : Low (n + 1) (colMS n j)) (t : ℕ) (ht : t < n + 1) :
    val3 (colSig3 n j q) t = val3 (colMS n j) t := by
  simp only [val3, (colSig3_testBit n j q hlo t).1, (colSig3_testBit n j q hlo t).2, if_pos ht]

lemma val3_colSig3_hi (n j : ℕ) (q : ℕ × ℕ) (hlo : Low (n + 1) (colMS n j)) (t : ℕ) :
    val3 (colSig3 n j q) (n + 1 + t) = val3 (colMU n j q) t := by
  simp only [val3, (colSig3_testBit n j q hlo _).1, (colSig3_testBit n j q hlo _).2, if_neg (show ¬ (n + 1 + t < n + 1) by omega),
    Nat.add_sub_cancel_left]

theorem rows_colsPS (n : ℕ) : ∀ r ∈ colsPS n, Disj r ∧ Low (n + 1) r := by
  rw [colsPS_eq]; intro r hr
  obtain ⟨j, hj, rfl⟩ := List.mem_map.mp hr
  exact colPS_ok n j (by have := List.mem_range.mp hj; omega)

theorem rows_colsMU (n : ℕ) : ∀ r ∈ colsMU n, Disj r ∧ Low (n + 1) r := by
  rw [colsMU_eq]; intro r hr
  obtain ⟨j, hj, rfl⟩ := List.mem_map.mp hr
  exact colMU_ok n j (by have := List.mem_range.mp hj; omega)

theorem rows_colsN3 (n : ℕ) : ∀ r ∈ colsN3 n, Disj r ∧ Low (n + 1) r := by
  rw [colsN3_eq]; intro r hr
  obtain ⟨j, hj, rfl⟩ := List.mem_map.mp hr
  exact colN3_ok n j (by have := List.mem_range.mp hj; omega)

theorem rows_colsSig3 (n : ℕ) : ∀ r ∈ colsSig3 n, Disj r ∧ Low ((n + 1) + (n + 1)) r := by
  rw [colsSig3_eq]; intro r hr
  obtain ⟨j, hj, rfl⟩ := List.mem_map.mp hr
  exact colSig3_ok n j (by have := List.mem_range.mp hj; omega)

end GL2.L23.RankF3
