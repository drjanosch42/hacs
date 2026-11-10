import Mathlib

/-!
# C9: the Bernoulli convention (`B₁ = −1/2`), `bernoulli` versus `bernoulli'`

[GT] `conv:bernoulli`, `bridge:bernoulli`.  The project's `bernoulli_mod` (`HeckeExt/HeckePredictions`) runs the
recursion `∑_{j=0}^{w} C(w+1, j) B_j = 0` (`w ≥ 1`) from `B₀ = 1`: that fixes `B₁ = −1/2` (the `B₁⁻`
convention), which is **mathlib's `bernoulli`**, not `bernoulli'` (`B₁ = +1/2`).  The two agree except at
index `1` (so at every even `k`, the only indices used downstream).
-/

namespace GL2

/-- **`conv:bernoulli`**: the project's Bernoulli numbers are mathlib's `bernoulli` (`B₁ = −1/2`). -/
abbrev projBernoulli (k : ℕ) : ℚ := bernoulli k

/-- The C++ recursion holds for `projBernoulli`: `∑_{j ≤ w} C(w+1, j) B_j = 0` for `w ≥ 1`, and `B₀ = 1`. -/
theorem projBernoulli_recursion (w : ℕ) (hw : 1 ≤ w) :
    ∑ j ∈ Finset.range (w + 1), ((w + 1).choose j : ℚ) * projBernoulli j = 0 := by
  have := sum_bernoulli (w + 1)
  rw [if_neg (by omega)] at this
  simpa [projBernoulli, mul_comm] using this

theorem projBernoulli_zero : projBernoulli 0 = 1 := bernoulli_zero

theorem projBernoulli_one : projBernoulli 1 = -1 / 2 := bernoulli_one

/-- The recursion and `B₀ = 1` determine the sequence: any `B` satisfying them is `projBernoulli`.  So the C++
`bernoulli_mod` (which runs exactly this recursion in `F_ℓ`, dividing only by `w + 1 ≤ ℓ − 1`) computes the
reduction of mathlib's `bernoulli`. -/
theorem eq_projBernoulli_of_recursion (B : ℕ → ℚ) (h0 : B 0 = 1)
    (hrec : ∀ w, 1 ≤ w → ∑ j ∈ Finset.range (w + 1), ((w + 1).choose j : ℚ) * B j = 0) :
    ∀ k, B k = projBernoulli k := by
  intro k
  induction k using Nat.strong_induction_on with
  | _ k ih =>
    rcases k with _ | k
    · rw [h0, projBernoulli_zero]
    · have h1 := hrec (k + 1) (by omega)
      have h2 := projBernoulli_recursion (k + 1) (by omega)
      rw [Finset.sum_range_succ] at h1 h2
      have hs : ∑ j ∈ Finset.range (k + 1), ((k + 1 + 1).choose j : ℚ) * B j =
          ∑ j ∈ Finset.range (k + 1), ((k + 1 + 1).choose j : ℚ) * projBernoulli j :=
        Finset.sum_congr rfl fun j hj => by rw [ih j (by have := Finset.mem_range.1 hj; omega)]
      have hc : ((k + 1 + 1).choose (k + 1) : ℚ) ≠ 0 := by
        exact_mod_cast (Nat.choose_pos (by omega)).ne'
      have : ((k + 1 + 1).choose (k + 1) : ℚ) * B (k + 1) =
          ((k + 1 + 1).choose (k + 1) : ℚ) * projBernoulli (k + 1) := by
        linear_combination h1 - h2 - hs
      exact mul_left_cancel₀ hc this

/-- **`bridge:bernoulli`**: `bernoulli' k = (−1)^k bernoulli k`; they agree for `k ≠ 1`, in particular at every
even `k`. -/
theorem bernoulli'_eq_proj (k : ℕ) : bernoulli' k = (-1) ^ k * projBernoulli k := bernoulli'_eq_bernoulli k

theorem bernoulli'_eq_proj_of_ne_one {k : ℕ} (hk : k ≠ 1) : bernoulli' k = projBernoulli k :=
  (bernoulli_eq_bernoulli'_of_ne_one hk).symm

theorem bernoulli'_eq_proj_of_even {k : ℕ} (hk : Even k) : bernoulli' k = projBernoulli k :=
  bernoulli'_eq_proj_of_ne_one (by rintro rfl; exact Nat.not_even_one hk)

/-- `B_k = 0` for odd `k ≥ 3` (the C++ `kummer_divides` degenerate branch). -/
theorem projBernoulli_odd_eq_zero {k : ℕ} (hk : Odd k) (h1 : 1 < k) : projBernoulli k = 0 := by
  have := bernoulli'_eq_zero_of_odd hk h1
  rwa [bernoulli'_eq_proj_of_ne_one (by omega)] at this

end GL2
