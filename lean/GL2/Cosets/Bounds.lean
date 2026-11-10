import Mathlib
import GL2.Cosets.DoubleCoset

/-!
# G1 (L17-3): entry bounds ([C] `lem:bounds` (1)–(4))

`‖A‖ = max_{i,j} |A_ij|`.  (1) submultiplicativity with the factor 2 and its `k`-fold form; (2) the fixed
input data `s_k`, `s_k⁻¹`, `δ_k`, `g`; (3) the cocycle `h_k(γ)`; (4) the Euclidean reduction (P2): one
nearest-integer step `T^{-q}` and the swap `S⁻¹` never raise `‖M‖` above `max(‖M‖, 2)`, hence no
intermediate of any run of the loop exceeds `max(‖γ‖, 2)`.

Clauses (5) (the `τ`-table bound `C_K N²`, a statement about `alg:tau` on the fixed model) and (6) (the
quadratic points of the fundamental-domain reduction) concern the model and the reduction algorithm of G2,
not the coset algebra, and are not stated here.
-/

namespace GL2.Cosets

open Matrix CongruenceSubgroup
open scoped MatrixGroups

/-- `‖A‖ = max |A_ij|`. -/
def mnorm (A : Mat2 ℤ) : ℤ := max (max |A 0 0| |A 0 1|) (max |A 1 0| |A 1 1|)

lemma mnorm_nonneg (A : Mat2 ℤ) : 0 ≤ mnorm A :=
  le_trans (abs_nonneg _) (le_trans (le_max_left _ _) (le_max_left _ _))

lemma abs_le_mnorm (A : Mat2 ℤ) (i j : Fin 2) : |A i j| ≤ mnorm A := by
  unfold mnorm
  fin_cases i <;> fin_cases j <;> simp

lemma mnorm_le_iff (A : Mat2 ℤ) (B : ℤ) : mnorm A ≤ B ↔ ∀ i j, |A i j| ≤ B := by
  constructor
  · exact fun h i j => le_trans (abs_le_mnorm A i j) h
  · intro h; unfold mnorm
    exact max_le (max_le (h 0 0) (h 0 1)) (max_le (h 1 0) (h 1 1))

/-- **[C] `lem:bounds`(1):** `‖AB‖ ≤ 2 ‖A‖ ‖B‖`. -/
theorem mnorm_mul (A B : Mat2 ℤ) : mnorm (A * B) ≤ 2 * mnorm A * mnorm B := by
  rw [mnorm_le_iff]
  intro i j
  rw [Matrix.mul_apply, Fin.sum_univ_two]
  have hA := mnorm_nonneg A
  have h1 : |A i 0 * B 0 j| ≤ mnorm A * mnorm B := by
    rw [abs_mul]; exact mul_le_mul (abs_le_mnorm _ _ _) (abs_le_mnorm _ _ _) (abs_nonneg _) hA
  have h2 : |A i 1 * B 1 j| ≤ mnorm A * mnorm B := by
    rw [abs_mul]; exact mul_le_mul (abs_le_mnorm _ _ _) (abs_le_mnorm _ _ _) (abs_nonneg _) hA
  calc |A i 0 * B 0 j + A i 1 * B 1 j| ≤ |A i 0 * B 0 j| + |A i 1 * B 1 j| := abs_add_le _ _
    _ ≤ 2 * mnorm A * mnorm B := by linarith

/-- **[C] `lem:bounds`(1), the `k`-fold form:** a product of `k ≥ 1` matrices is bounded by
`2^(k-1) ∏ ‖A_t‖`. -/
theorem mnorm_prod (A : Mat2 ℤ) (l : List (Mat2 ℤ)) :
    mnorm (A :: l).prod ≤ 2 ^ l.length * ((A :: l).map mnorm).prod := by
  induction l generalizing A with
  | nil => simp
  | cons B l ih =>
    have h := ih B
    simp only [List.prod_cons, List.map_cons, List.length_cons] at h ⊢
    calc mnorm (A * (B * l.prod)) ≤ 2 * mnorm A * mnorm (B * l.prod) := mnorm_mul _ _
      _ ≤ 2 * mnorm A * (2 ^ l.length * (mnorm B * (l.map mnorm).prod)) := by
          apply mul_le_mul_of_nonneg_left h
          have := mnorm_nonneg A; positivity
      _ = 2 ^ (l.length + 1) * (mnorm A * (mnorm B * (l.map mnorm).prod)) := by ring

variable {p ν : ℕ}

/-- **[C] `lem:bounds`(2), the fixed input data:** `‖s_k‖, ‖s_k⁻¹‖, ‖δ_k‖, ‖g‖ ≤ N` for `k < m`. -/
theorem fixed_data_bound (hp : 1 ≤ p) (hν : 1 ≤ ν) {k : ℕ} (hk : k < mIdx p ν) :
    mnorm (sSL p ν k : Mat2 ℤ) ≤ (p : ℤ) ^ ν ∧ mnorm ((sSL p ν k)⁻¹ : SL(2, ℤ)) ≤ (p : ℤ) ^ ν ∧
      mnorm (deltaMat p ν k) ≤ (p : ℤ) ^ ν ∧ mnorm (gMat ((p : ℤ) ^ ν)) ≤ (p : ℤ) ^ ν := by
  have hb := fun i j => sSL_bound hν hp hk i j
  have hN1 : (1 : ℤ) ≤ (p : ℤ) ^ ν := one_le_pow₀ (by exact_mod_cast hp)
  refine ⟨(mnorm_le_iff _ _).2 (fun i j => (hb i j).1), (mnorm_le_iff _ _).2 (fun i j => (hb i j).2), ?_, ?_⟩
  · -- the second column of `s_k⁻¹` has entries in `{0, ±1}`
    rw [mnorm_le_iff]
    intro i j
    have hcol : |(((sSL p ν k)⁻¹ : SL(2, ℤ)) : Mat2 ℤ) i 1| ≤ 1 := by
      rw [coe_inv_apply, sSL_coe]; unfold sMat
      split_ifs <;> fin_cases i <;> simp
    unfold deltaMat
    have h0 := (hb i 0).2
    rw [Matrix.SpecialLinearGroup.coe_inv] at hcol h0
    have hN0 : (0 : ℤ) ≤ (p : ℤ) ^ ν := by positivity
    fin_cases j
    · simpa [Matrix.mul_apply, Fin.sum_univ_two, gMat] using h0
    · simp [Matrix.mul_apply, Fin.sum_univ_two, gMat]
      nlinarith [abs_nonneg ((sSL p ν k : Mat2 ℤ).adjugate i 1)]
  · rw [mnorm_le_iff]; intro i j
    fin_cases i <;> fin_cases j <;> simp [gMat, abs_of_pos (by positivity : (0 : ℤ) < (p : ℤ) ^ ν)] <;> linarith

/-- **[C] `lem:bounds`(3):** `‖h_k(γ)‖ ≤ 4 N² ‖γ‖`. -/
theorem hcoc_bound (hp : p.Prime) (hν : 1 ≤ ν) {k : ℕ} (hk : k < mIdx p ν) (γ : SL(2, ℤ)) :
    mnorm (hcoc p ν k γ : Mat2 ℤ) ≤ 4 * ((p : ℤ) ^ ν) ^ 2 * mnorm (γ : Mat2 ℤ) := by
  have h1 := fixed_data_bound hp.one_lt.le hν hk
  have h2 := fixed_data_bound hp.one_lt.le hν (act_lt hp hν k γ)
  have hγ := mnorm_nonneg (γ : Mat2 ℤ)
  have hs := mnorm_nonneg (sSL p ν k : Mat2 ℤ)
  have hN : (0 : ℤ) ≤ (p : ℤ) ^ ν := by positivity
  unfold hcoc
  rw [Matrix.SpecialLinearGroup.coe_mul, Matrix.SpecialLinearGroup.coe_mul]
  calc mnorm ((sSL p ν k : Mat2 ℤ) * γ * ((sSL p ν (act p ν k γ))⁻¹ : SL(2, ℤ)))
      ≤ 2 * mnorm ((sSL p ν k : Mat2 ℤ) * γ) * mnorm (((sSL p ν (act p ν k γ))⁻¹ : SL(2, ℤ)) : Mat2 ℤ) :=
        mnorm_mul _ _
    _ ≤ 2 * (2 * mnorm (sSL p ν k : Mat2 ℤ) * mnorm (γ : Mat2 ℤ)) * (p : ℤ) ^ ν := by
        apply mul_le_mul (mul_le_mul_of_nonneg_left (mnorm_mul _ _) (by norm_num)) h2.2.1 (mnorm_nonneg _)
        have := mnorm_nonneg ((sSL p ν k : Mat2 ℤ) * γ); positivity
    _ ≤ 2 * (2 * (p : ℤ) ^ ν * mnorm (γ : Mat2 ℤ)) * (p : ℤ) ^ ν := by
        apply mul_le_mul_of_nonneg_right _ hN
        apply mul_le_mul_of_nonneg_left _ (by norm_num)
        exact mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left h1.1 (by norm_num)) hγ
    _ = 4 * ((p : ℤ) ^ ν) ^ 2 * mnorm (γ : Mat2 ℤ) := by ring

/-! ## (4) The Euclidean reduction (P2) -/

/-- The integer heart of (4): after a nearest-integer step `a' = a - q c`, `|a'| ≤ |c|/2`, the new
`b' = b - q d` satisfies `|b'| ≤ max(|d|, 2)`. -/
theorem eucl_step_entry (a b c d q : ℤ) (hdet : a * d - b * c = 1) (hc : c ≠ 0)
    (hq : 2 * |a - q * c| ≤ |c|) : |b - q * d| ≤ max |d| 2 := by
  set a' := a - q * c
  set b' := b - q * d
  have hdet' : a' * d - b' * c = 1 := by simp only [a', b']; linear_combination hdet
  have hc1 : 1 ≤ |c| := Int.one_le_abs hc
  have key : |b'| * |c| ≤ |a'| * |d| + 1 := by
    rw [← abs_mul]
    have : b' * c = a' * d - 1 := by linarith
    rw [this]
    calc |a' * d - 1| ≤ |a' * d| + |(1 : ℤ)| := abs_sub _ _
      _ = |a'| * |d| + 1 := by rw [abs_mul]; simp
  have key2 : 2 * (|b'| * |c|) ≤ |c| * |d| + 2 := by
    have := mul_le_mul_of_nonneg_right hq (abs_nonneg d)
    linarith
  by_contra hcon
  push Not at hcon
  rw [max_lt_iff] at hcon
  obtain ⟨hb1, hb2⟩ := hcon
  nlinarith [abs_nonneg d, abs_nonneg b']

/-- One step of (P2) on `SL(2, ℤ)`: a nearest-integer translation `T^{-q}` or the swap `S⁻¹`. -/
inductive EuclStep : SL(2, ℤ) → SL(2, ℤ) → Prop
  | trans (M : SL(2, ℤ)) (q : ℤ) (hc : M 1 0 ≠ 0) (hq : 2 * |M 0 0 - q * M 1 0| ≤ |M 1 0|) :
      EuclStep M (ModularGroup.T ^ (-q) * M)
  | swap (M : SL(2, ℤ)) : EuclStep M (ModularGroup.S⁻¹ * M)

/-- **[C] `lem:bounds`(4), one step.** -/
theorem eucl_step_bound {M M' : SL(2, ℤ)} (h : EuclStep M M') :
    mnorm (M' : Mat2 ℤ) ≤ max (mnorm (M : Mat2 ℤ)) 2 := by
  cases h with
  | trans q hc hq =>
    have hb := eucl_step_entry (M 0 0) (M 0 1) (M 1 0) (M 1 1) q (det_SL M) hc hq
    rw [mnorm_le_iff]
    intro i j
    rw [Matrix.SpecialLinearGroup.coe_mul, ModularGroup.coe_T_zpow]
    have hle := fun i j => le_trans (abs_le_mnorm (M : Mat2 ℤ) i j) (le_max_left (mnorm (M : Mat2 ℤ)) 2)
    have h11 := abs_le_mnorm (M : Mat2 ℤ) 1 1
    fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_two]
    · -- `|a - q c| ≤ |c|/2`
      have := hle 1 0
      have h0 : |M 0 0 - q * M 1 0| ≤ |M 1 0| := by linarith [abs_nonneg (M 0 0 - q * M 1 0)]
      have e : M 0 0 + -(q * M 1 0) = M 0 0 - q * M 1 0 := by ring
      left; rw [e]; linarith [abs_le_mnorm (M : Mat2 ℤ) 1 0]
    · have e : M 0 1 + -(q * M 1 1) = M 0 1 - q * M 1 1 := by ring
      rw [e]
      exact le_max_iff.mp (le_trans hb (max_le_max h11 (le_refl _)))
    · exact le_max_iff.mp (hle 1 0)
    · exact le_max_iff.mp (hle 1 1)
  | swap =>
    rw [mnorm_le_iff]
    intro i j
    rw [Matrix.SpecialLinearGroup.coe_mul, Matrix.SpecialLinearGroup.coe_inv, ModularGroup.coe_S,
      Matrix.adjugate_fin_two]
    have hle := fun i j => le_trans (abs_le_mnorm (M : Mat2 ℤ) i j) (le_max_left (mnorm (M : Mat2 ℤ)) 2)
    fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_two]
    · exact le_max_iff.mp (hle 1 0)
    · exact le_max_iff.mp (hle 1 1)
    · exact le_max_iff.mp (hle 0 0)
    · exact le_max_iff.mp (hle 0 1)

/-- **[C] `lem:bounds`(4):** every intermediate matrix of a run of (P2) on `γ` has `‖M‖ ≤ max(‖γ‖, 2)`. -/
theorem eucl_run_bound {γ M : SL(2, ℤ)} (h : Relation.ReflTransGen EuclStep γ M) :
    mnorm (M : Mat2 ℤ) ≤ max (mnorm (γ : Mat2 ℤ)) 2 := by
  induction h with
  | refl => exact le_max_left _ _
  | tail _ hstep ih =>
    exact le_trans (eucl_step_bound hstep) (max_le ih (le_max_right _ _))

end GL2.Cosets
