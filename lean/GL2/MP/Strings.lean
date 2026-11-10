import Mathlib

/-!
# MP-L1 (slot 22): strings and their alternating deletion sums

[GT] `gt_part_MP.tex`, row `lem:mp-del`. The combinatorial core shared by the MP set ([C] `def:mpset`), the labelled Čech
nerve `𝔉'` and the free bisimplicial set `𝔉` ([C] `def:mpfree`): a `k`-string is a map `Fin (k+1) → O`, its `i`-th face
deletes entry `i` (`del i f = f ∘ i.succAbove`), the faces satisfy the first simplicial identity (`del_del`, the identity
behind mathlib's `SimplexCategory.δ_comp_δ`), and hence every alternating double deletion sum vanishes (`alt_sum_sq`,
"`d ∘ d = 0`" for all of these complexes at once, whatever the coefficients).
-/

namespace GL2.MP

variable {O : Type*}

/-- The `i`-th face of a string: delete entry `i`. -/
def del {n : ℕ} (i : Fin (n + 2)) (f : Fin (n + 2) → O) : Fin (n + 1) → O := f ∘ i.succAbove

@[simp] lemma del_apply {n : ℕ} (i : Fin (n + 2)) (f : Fin (n + 2) → O) (m : Fin (n + 1)) :
    del i f m = f (i.succAbove m) := rfl

lemma del_comp {O' : Type*} {n : ℕ} (i : Fin (n + 2)) (f : Fin (n + 2) → O) (φ : O → O') :
    del i (φ ∘ f) = φ ∘ del i f := rfl

lemma succAbove_succAbove_of_le {n : ℕ} {i j : Fin (n + 2)} (H : i ≤ j) (m : Fin (n + 1)) :
    j.succ.succAbove (i.succAbove m) = i.castSucc.succAbove (j.succAbove m) := by
  rcases i with ⟨i, hi⟩
  rcases j with ⟨j, hj⟩
  rcases m with ⟨m, hm⟩
  simp only [Fin.le_def] at H
  apply Fin.ext
  simp only [Fin.succAbove, Fin.lt_def, Fin.val_castSucc, Fin.val_succ]
  split_ifs <;> simp_all <;> omega

/-- **The first simplicial identity** for deletions: `d_i d_{j+1} = d_j d_i` for `i ≤ j`. -/
theorem del_del {n : ℕ} {i j : Fin (n + 2)} (H : i ≤ j) (f : Fin (n + 3) → O) :
    del i (del j.succ f) = del j (del i.castSucc f) := by
  funext m
  simp only [del_apply, succAbove_succAbove_of_le H]

/-- The involution pairing the double deletions `(i, j)` (delete `i`, then `j`). -/
def delPair {n : ℕ} (p : Fin (n + 3) × Fin (n + 2)) : Fin (n + 3) × Fin (n + 2) :=
  if h : (p.2 : ℕ) < p.1 then (p.2.castSucc, p.1.pred (by intro e; rw [e] at h; exact Nat.not_lt_zero _ h))
  else (p.2.succ, p.1.castLT (by have := p.2.2; omega))

lemma delPair_lt {n : ℕ} (p : Fin (n + 3) × Fin (n + 2)) (h : (p.2 : ℕ) < p.1) :
    ((delPair p).1 : ℕ) = p.2 ∧ ((delPair p).2 : ℕ) = p.1 - 1 := by
  unfold delPair; rw [dif_pos h]; simp

lemma delPair_ge {n : ℕ} (p : Fin (n + 3) × Fin (n + 2)) (h : ¬ (p.2 : ℕ) < p.1) :
    ((delPair p).1 : ℕ) = p.2 + 1 ∧ ((delPair p).2 : ℕ) = p.1 := by
  unfold delPair; rw [dif_neg h]; simp

lemma delPair_involutive {n : ℕ} : Function.Involutive (delPair (n := n)) := by
  intro p
  by_cases h : (p.2 : ℕ) < p.1
  · obtain ⟨e1, e2⟩ := delPair_lt p h
    have h' : ¬ ((delPair p).2 : ℕ) < (delPair p).1 := by omega
    obtain ⟨f1, f2⟩ := delPair_ge (delPair p) h'
    exact Prod.ext (Fin.ext (by omega)) (Fin.ext (by omega))
  · obtain ⟨e1, e2⟩ := delPair_ge p h
    have h' : ((delPair p).2 : ℕ) < (delPair p).1 := by omega
    obtain ⟨f1, f2⟩ := delPair_lt (delPair p) h'
    exact Prod.ext (Fin.ext (by omega)) (Fin.ext (by omega))

lemma delPair_ne {n : ℕ} (p : Fin (n + 3) × Fin (n + 2)) : delPair p ≠ p := by
  intro e
  by_cases h : (p.2 : ℕ) < p.1
  · have := (delPair_lt p h).1; rw [e] at this; omega
  · have := (delPair_ge p h).2; rw [e] at this
    have := (delPair_ge p h).1; rw [e] at this; omega

lemma del_delPair {n : ℕ} (p : Fin (n + 3) × Fin (n + 2)) (f : Fin (n + 3) → O) :
    del (delPair p).2 (del (delPair p).1 f) = del p.2 (del p.1 f) := by
  by_cases h : (p.2 : ℕ) < p.1
  · -- `j < i`: `d_j d_i = d_{i-1} d_j` by `del_del` with `(j, i - 1)`
    have hp : p.1 ≠ 0 := by intro e; rw [e] at h; exact Nat.not_lt_zero _ h
    have H : p.2 ≤ p.1.pred hp := by rw [Fin.le_def, Fin.val_pred]; omega
    have := del_del H f
    rw [Fin.succ_pred] at this
    unfold delPair; rw [dif_pos h]; exact this.symm
  · have H : p.1.castLT (by have := p.2.2; omega) ≤ p.2 := by rw [Fin.le_def, Fin.val_castLT]; omega
    have := del_del H f
    rw [Fin.castSucc_castLT] at this
    unfold delPair; rw [dif_neg h]; exact this

lemma sign_delPair {n : ℕ} (p : Fin (n + 3) × Fin (n + 2)) :
    (-1 : ℤ) ^ ((delPair p).1 : ℕ) * (-1) ^ ((delPair p).2 : ℕ) = -((-1) ^ (p.1 : ℕ) * (-1) ^ (p.2 : ℕ)) := by
  by_cases h : (p.2 : ℕ) < p.1
  · obtain ⟨e1, e2⟩ := delPair_lt p h
    rw [e1, e2]
    obtain ⟨m, hm⟩ : ∃ m, (p.1 : ℕ) = m + 1 := ⟨p.1 - 1, by omega⟩
    rw [hm, Nat.add_sub_cancel, pow_succ]; ring
  · obtain ⟨e1, e2⟩ := delPair_ge p h
    rw [e1, e2, pow_succ]; ring

/-- **`d ∘ d = 0` for strings**: the alternating double deletion sum of any string vanishes, for any coefficients. -/
theorem alt_sum_sq {A : Type*} [AddCommGroup A] {n : ℕ} (f : Fin (n + 3) → O) (Φ : (Fin (n + 1) → O) → A) :
    ∑ i : Fin (n + 3), ∑ j : Fin (n + 2), ((-1 : ℤ) ^ (i : ℕ) * (-1) ^ (j : ℕ)) • Φ (del j (del i f)) = 0 := by
  rw [← Finset.sum_product']
  refine Finset.sum_involution (fun p _ => delPair p) (fun p _ => ?_) (fun p _ _ => delPair_ne p)
    (fun p _ => Finset.mem_product.2 ⟨Finset.mem_univ _, Finset.mem_univ _⟩) (fun p _ => delPair_involutive p)
  show ((-1 : ℤ) ^ (p.1 : ℕ) * (-1) ^ (p.2 : ℕ)) • Φ (del p.2 (del p.1 f)) +
    ((-1 : ℤ) ^ ((delPair p).1 : ℕ) * (-1) ^ ((delPair p).2 : ℕ)) • Φ (del (delPair p).2 (del (delPair p).1 f)) = 0
  rw [del_delPair, sign_delPair, neg_smul, add_neg_cancel]

end GL2.MP
