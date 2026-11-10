import Mathlib

/-!
# C11: `ε_K`, as one definition

[GT] `conv:epsK`, mirroring [E] v1.16.1 `sec:ring`(i) (the slot-15 WP-5 audit M2 form) and [C] v1.15.0
`prop:satdefect`.

`ε_K := K · dim_{F_ℓ}(A_K/ℓ) − length(A_K)` for a finite `ℤ/ℓ^K`-module `A_K`.  The code reads `A_K` by its cyclic
decomposition `A_K ≅ ⊕_i ℤ/ℓ^{e_i}`, `1 ≤ e_i ≤ K`, and then `dim(A_K/ℓ) = #{i}` (one generator per summand) and
`length(A_K) = ∑ e_i`.  So the definition is taken on the invariant-factor list `es = [e_i]`.

The pre-v1.15.1-pre.6 text `K · rank_{ℤ_ℓ}(B_K) − length` is a **different** quantity (rank of the block, not
`dim A_K/ℓ`); it is not defined here.
-/

namespace GL2

/-- **`conv:epsK`**: `ε_K = K · dim(A_K/ℓ) − length(A_K)`, on the exponents `es` of the cyclic decomposition. -/
def epsK (K : ℕ) (es : List ℕ) : ℤ := (K : ℤ) * es.length - (es.map fun e : ℕ => (e : ℤ)).sum

/-- `ε_K = ∑_i (K − e_i)`. -/
theorem epsK_eq_sum (K : ℕ) (es : List ℕ) :
    epsK K es = (es.map fun e : ℕ => (K : ℤ) - (e : ℤ)).sum := by
  induction es with
  | nil => simp [epsK]
  | cons e es ih =>
    simp only [epsK, List.length_cons, List.map_cons, List.sum_cons] at ih ⊢
    push_cast
    linarith

/-- `ε_K ≥ 0` when every exponent is at most `K` (a `ℤ/ℓ^K`-module). -/
theorem epsK_nonneg (K : ℕ) (es : List ℕ) (h : ∀ e ∈ es, e ≤ K) : 0 ≤ epsK K es := by
  rw [epsK_eq_sum]
  apply List.sum_nonneg
  intro x hx
  obtain ⟨e, he, rfl⟩ := List.mem_map.1 hx
  have := h e he
  omega

/-- `ε_K = 0` iff `A_K` is free over `ℤ/ℓ^K` (every exponent is `K`). -/
theorem epsK_eq_zero_iff (K : ℕ) (es : List ℕ) (h : ∀ e ∈ es, e ≤ K) :
    epsK K es = 0 ↔ ∀ e ∈ es, e = K := by
  rw [epsK_eq_sum]
  induction es with
  | nil => simp
  | cons e es ih =>
    have he := h e (by simp)
    have hrest : ∀ x ∈ es, x ≤ K := fun x hx => h x (by simp [hx])
    have hnn : 0 ≤ (es.map fun e : ℕ => (K : ℤ) - (e : ℤ)).sum := by
      apply List.sum_nonneg
      intro x hx
      obtain ⟨y, hy, rfl⟩ := List.mem_map.1 hx
      have := hrest y hy; omega
    simp only [List.map_cons, List.sum_cons, List.mem_cons, forall_eq_or_imp]
    rw [← ih hrest]
    constructor
    · intro hs; constructor <;> omega
    · rintro ⟨rfl, hs⟩; simp [hs]

/-- Sanity instance: `A_K = ℤ/ℓ^3 ⊕ ℤ/ℓ` at `K = 3` has `ε_3 = 3·2 − 4 = 2`. -/
example : epsK 3 [3, 1] = 2 := by decide

end GL2
