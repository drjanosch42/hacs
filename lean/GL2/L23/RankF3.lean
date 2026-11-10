/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import Mathlib
import GL2.L23.RankF2
import GL2.L23.RankF3Exec

/-!
# LV-2 (slot 22): `rankF3` is `Matrix.rank` over `ZMod 3`

The executable elimination `GL2.L23.RankF3.rankF3` (`RankF3Exec.lean`, Mathlib-free, two bit-planes per row) is proved equal
to mathlib's `Matrix.rank` of the `rows.length × w` matrix over `ZMod 3` whose entry `(i, j)` is `val3 rows[i] j` (`1` on
the first plane, `−1` on the second), for every list of rows with disjoint planes below `2^w` (`rankF3_eq_rank`), in the
style of `GL2.L23.RankF2.rankF2_eq_rank`.

**Per-bit arithmetic.** `add3` is a bitwise formula; `testBit_add3_fst` / `_snd` read it per position and `val3_add3`
checks the sixteen cases (nine with disjoint planes) of `ZMod 3` addition. **Invariant** (`Good3`): the basis is a list of
`(p, b)` with `b` disjoint, entry `1` at `p`, nothing above `p`, `p < w`, strictly descending pivots. One pass of `reduce3`
clears every pivot position (`reduce3_clear`) and changes the row by an element of the span (`reduce3_vec`); a non-zero
remainder is normalised to `1` at its top position and is new there, so the span grows by exactly that row.
-/

namespace GL2.L23.RankF3

open Submodule
open GL2.L23.RankF2 (bit bit_eq log2_facts force)

/-! ### Entries -/

/-- Entry `j` of a two-plane row: `1` (first plane), `−1` (second plane), `0`. -/
def val3 (x : ℕ × ℕ) (j : ℕ) : ZMod 3 := if x.1.testBit j then 1 else if x.2.testBit j then -1 else 0

/-- The row as a vector of length `w`. -/
def vec3 (w : ℕ) (x : ℕ × ℕ) : Fin w → ZMod 3 := fun j => val3 x j

/-- Disjoint planes. -/
def Disj (x : ℕ × ℕ) : Prop := ∀ j, ¬ (x.1.testBit j = true ∧ x.2.testBit j = true)

/-- Nothing at positions `≥ P`. -/
def Low (P : ℕ) (x : ℕ × ℕ) : Prop := ∀ j, P ≤ j → x.1.testBit j = false ∧ x.2.testBit j = false

/-! ### The kernel formulation, read back -/

lemma force_eq' {α : Sort _} (a : ℕ) (k : ℕ → α) : force a k = k a := by
  unfold force; cases Nat.beq a 0 <;> rfl

lemma force2_eq {α : Sort _} (x : ℕ × ℕ) (k : ℕ × ℕ → α) : force2 x k = k x := by
  unfold force2; rw [force_eq', force_eq']

lemma testBit_andNot (a b j : ℕ) : (andNot a b).testBit j = (a.testBit j && !b.testBit j) := by
  show (a ^^^ (a &&& b)).testBit j = _
  rw [Nat.testBit_xor, Nat.testBit_land]
  cases a.testBit j <;> cases b.testBit j <;> rfl

lemma add3_eq (x y : ℕ × ℕ) : add3 x y =
    ((andNot x.1 (y.1 ||| y.2) ||| andNot y.1 (x.1 ||| x.2)) ||| (x.2 &&& y.2),
     (andNot x.2 (y.1 ||| y.2) ||| andNot y.2 (x.1 ||| x.2)) ||| (x.1 &&& y.1)) := by
  unfold add3; rw [force_eq', force_eq']; rfl

lemma testBit_add3_fst (x y : ℕ × ℕ) (j : ℕ) : (add3 x y).1.testBit j =
    ((x.1.testBit j && !(y.1.testBit j || y.2.testBit j)) || (y.1.testBit j && !(x.1.testBit j || x.2.testBit j)) ||
      (x.2.testBit j && y.2.testBit j)) := by
  rw [add3_eq]
  simp only [Nat.testBit_lor, Nat.testBit_land, testBit_andNot]

lemma testBit_add3_snd (x y : ℕ × ℕ) (j : ℕ) : (add3 x y).2.testBit j =
    ((x.2.testBit j && !(y.1.testBit j || y.2.testBit j)) || (y.2.testBit j && !(x.1.testBit j || x.2.testBit j)) ||
      (x.1.testBit j && y.1.testBit j)) := by
  rw [add3_eq]
  simp only [Nat.testBit_lor, Nat.testBit_land, testBit_andNot]

lemma val3_add3 {x y : ℕ × ℕ} (hx : Disj x) (hy : Disj y) (j : ℕ) : val3 (add3 x y) j = val3 x j + val3 y j := by
  have h1 := hx j
  have h2 := hy j
  simp only [val3, testBit_add3_fst, testBit_add3_snd]
  cases ha : x.1.testBit j <;> cases hb : x.2.testBit j <;> cases hc : y.1.testBit j <;>
    cases hd : y.2.testBit j <;> simp_all <;> decide

lemma disj_add3 {x y : ℕ × ℕ} (hx : Disj x) (hy : Disj y) : Disj (add3 x y) := by
  intro j
  have h1 := hx j
  have h2 := hy j
  rw [testBit_add3_fst, testBit_add3_snd]
  cases ha : x.1.testBit j <;> cases hb : x.2.testBit j <;> cases hc : y.1.testBit j <;>
    cases hd : y.2.testBit j <;> simp_all

lemma val3_neg3 {x : ℕ × ℕ} (hx : Disj x) (j : ℕ) : val3 (neg3 x) j = - val3 x j := by
  have h1 := hx j
  simp only [val3, neg3]
  by_cases ha : x.1.testBit j = true <;> by_cases hb : x.2.testBit j = true <;> simp [ha, hb] at h1 ⊢

lemma disj_neg3 {x : ℕ × ℕ} (hx : Disj x) : Disj (neg3 x) := fun j ⟨h1, h2⟩ => hx j ⟨h2, h1⟩

lemma vec3_add3 {w : ℕ} {x y : ℕ × ℕ} (hx : Disj x) (hy : Disj y) : vec3 w (add3 x y) = vec3 w x + vec3 w y := by
  funext j; exact val3_add3 hx hy j

lemma vec3_neg3 {w : ℕ} {x : ℕ × ℕ} (hx : Disj x) : vec3 w (neg3 x) = - vec3 w x := by
  funext j; exact val3_neg3 hx j

/-- Positions where both rows vanish stay zero. -/
lemma add3_zero_at {x y : ℕ × ℕ} {j : ℕ} (hy1 : y.1.testBit j = false) (hy2 : y.2.testBit j = false) :
    (add3 x y).1.testBit j = x.1.testBit j ∧ (add3 x y).2.testBit j = x.2.testBit j := by
  rw [testBit_add3_fst, testBit_add3_snd, hy1, hy2]
  cases x.1.testBit j <;> cases x.2.testBit j <;> simp

lemma low_add3 {P : ℕ} {x y : ℕ × ℕ} (hx : Low P x) (hy : Low P y) : Low P (add3 x y) := by
  intro j hj
  obtain ⟨h1, h2⟩ := add3_zero_at (x := x) (hy j hj).1 (hy j hj).2
  rw [h1, h2]; exact hx j hj

lemma low_neg3 {P : ℕ} {x : ℕ × ℕ} (hx : Low P x) : Low P (neg3 x) := fun j hj => ⟨(hx j hj).2, (hx j hj).1⟩

lemma low_mono {P Q : ℕ} {x : ℕ × ℕ} (hPQ : P ≤ Q) (hx : Low P x) : Low Q x := fun j hj => hx j (le_trans hPQ hj)

lemma testBit_supp (r : ℕ × ℕ) (j : ℕ) : (supp r).testBit j = (r.1.testBit j || r.2.testBit j) := by
  show (r.1 ||| r.2).testBit j = _
  rw [Nat.testBit_lor]

lemma low_iff_supp (P : ℕ) (x : ℕ × ℕ) : Low P x ↔ supp x < 2 ^ P := by
  constructor
  · intro h
    refine Nat.lt_pow_two_of_testBit _ fun j hj => ?_
    rw [testBit_supp, (h j hj).1, (h j hj).2]; rfl
  · intro h j hj
    have := Nat.testBit_eq_false_of_lt (lt_of_lt_of_le h (Nat.pow_le_pow_right (by norm_num) hj))
    rw [testBit_supp] at this
    simpa using this

lemma supp_eq_zero {r : ℕ × ℕ} (h : supp r = 0) : ∀ j, r.1.testBit j = false ∧ r.2.testBit j = false := by
  intro j
  have := congrArg (fun x => x.testBit j) h
  simp only [testBit_supp, Nat.zero_testBit] at this
  simpa using this

lemma vec3_zero_of_supp {w : ℕ} {r : ℕ × ℕ} (h : supp r = 0) : vec3 w r = 0 := by
  funext j
  simp [vec3, val3, (supp_eq_zero h j).1, (supp_eq_zero h j).2]

/-! ### Steps -/

lemma step3_eq (r : ℕ × ℕ) (qb : ℕ × (ℕ × ℕ)) : step3 r qb =
    if r.1.testBit qb.1 then add3 r (neg3 qb.2) else if r.2.testBit qb.1 then add3 r qb.2 else r := by
  unfold step3
  rw [bit_eq, bit_eq, force2_eq, force2_eq]
  cases r.1.testBit qb.1 <;> cases r.2.testBit qb.1 <;> rfl

lemma reduce3_nil (r : ℕ × ℕ) : reduce3 [] r = r := rfl

lemma reduce3_cons (qb : ℕ × (ℕ × ℕ)) (B : List (ℕ × (ℕ × ℕ))) (r : ℕ × ℕ) :
    reduce3 (qb :: B) r = reduce3 B (step3 r qb) := rfl

lemma insertP3_nil (p : ℕ) (r : ℕ × ℕ) : insertP3 p r [] = [(p, r)] := rfl

lemma insertP3_cons (p : ℕ) (r : ℕ × ℕ) (qb : ℕ × (ℕ × ℕ)) (B : List (ℕ × (ℕ × ℕ))) :
    insertP3 p r (qb :: B) = if qb.1 < p then (p, r) :: qb :: B else qb :: insertP3 p r B := by
  show @Bool.rec (fun _ => List (ℕ × (ℕ × ℕ))) (qb :: insertP3 p r B) ((p, r) :: qb :: B) (Nat.blt qb.1 p) = _
  by_cases h : qb.1 < p
  · rw [if_pos h, show Nat.blt qb.1 p = true from Nat.blt_eq.mpr h]
  · rw [if_neg h, show Nat.blt qb.1 p = false from Bool.eq_false_iff.mpr fun h' => h (Nat.blt_eq.mp h')]

lemma norm3_eq (t : ℕ) (r : ℕ × ℕ) : norm3 t r = if r.1.testBit t then r else neg3 r := by
  unfold norm3; rw [bit_eq]; cases r.1.testBit t <;> rfl

lemma addRed3_eq (B : List (ℕ × (ℕ × ℕ))) (r : ℕ × ℕ) :
    addRed3 B r = if supp r = 0 then B else insertP3 (Nat.log2 (supp r)) (norm3 (Nat.log2 (supp r)) r) B := by
  unfold addRed3
  by_cases h : supp r = 0
  · rw [if_pos h, h]; rfl
  · rw [if_neg h, show Nat.beq (supp r) 0 = false from Bool.eq_false_iff.mpr fun h' => h (Nat.eq_of_beq_eq_true h')]

lemma addRow3_eq (B : List (ℕ × (ℕ × ℕ))) (r : ℕ × ℕ) : addRow3 B r = addRed3 B (reduce3 B r) := by
  unfold addRow3; rw [force2_eq]

lemma basis3_nil (B : List (ℕ × (ℕ × ℕ))) : basis3 [] B = B := rfl

lemma basis3_cons (r : ℕ × ℕ) (rs : List (ℕ × ℕ)) (B : List (ℕ × (ℕ × ℕ))) :
    basis3 (r :: rs) B = basis3 rs (addRow3 B r) := rfl

/-! ### The invariant -/

/-- The basis invariant. -/
def Good3 (w : ℕ) : List (ℕ × (ℕ × ℕ)) → Prop
  | [] => True
  | qb :: B => Low (qb.1 + 1) qb.2 ∧ qb.2.1.testBit qb.1 = true ∧ Disj qb.2 ∧ qb.1 < w ∧ (∀ q ∈ B, q.1 < qb.1) ∧
      Good3 w B

/-- The row vectors of a basis. -/
def rowSet3 (w : ℕ) (B : List (ℕ × (ℕ × ℕ))) : Set (Fin w → ZMod 3) := {v | ∃ qb ∈ B, vec3 w qb.2 = v}

variable {w : ℕ}

lemma rowSet3_cons (qb : ℕ × (ℕ × ℕ)) (B : List (ℕ × (ℕ × ℕ))) :
    rowSet3 w (qb :: B) = insert (vec3 w qb.2) (rowSet3 w B) := by
  ext v; simp [rowSet3, eq_comm]

lemma rowSet3_nil : rowSet3 w [] = ∅ := Set.eq_empty_of_forall_notMem (by simp [rowSet3])

lemma good3_disj (B : List (ℕ × (ℕ × ℕ))) (hB : Good3 w B) : ∀ q ∈ B, Disj q.2 := by
  induction B with
  | nil => simp
  | cons qb B ih =>
    obtain ⟨-, -, h3, -, -, hB⟩ := hB
    intro q hq; rcases List.mem_cons.mp hq with rfl | hq
    · exact h3
    · exact ih hB q hq

lemma good3_low (B : List (ℕ × (ℕ × ℕ))) (hB : Good3 w B) {P : ℕ} (hP : ∀ q ∈ B, q.1 < P) : ∀ q ∈ B, Low P q.2 := by
  induction B with
  | nil => simp
  | cons qb B ih =>
    obtain ⟨h1, -, -, -, -, hB⟩ := hB
    intro q hq; rcases List.mem_cons.mp hq with rfl | hq
    · exact low_mono (hP q (by simp)) h1
    · exact ih hB (fun q hq => hP q (by simp [hq])) q hq

lemma good3_low_w (B : List (ℕ × (ℕ × ℕ))) (hB : Good3 w B) : ∀ q ∈ B, Low w q.2 := by
  induction B with
  | nil => simp
  | cons qb B ih =>
    obtain ⟨h1, -, -, h4, -, hB⟩ := hB
    intro q hq; rcases List.mem_cons.mp hq with rfl | hq
    · exact low_mono h4 h1
    · exact ih hB q hq

/-! ### `reduce3` -/

lemma disj_step3 {r : ℕ × ℕ} (hr : Disj r) {qb : ℕ × (ℕ × ℕ)} (hb : Disj qb.2) : Disj (step3 r qb) := by
  rw [step3_eq]
  split_ifs
  · exact disj_add3 hr (disj_neg3 hb)
  · exact disj_add3 hr hb
  · exact hr

lemma disj_reduce3 (B : List (ℕ × (ℕ × ℕ))) (hB : ∀ q ∈ B, Disj q.2) {r : ℕ × ℕ} (hr : Disj r) :
    Disj (reduce3 B r) := by
  induction B generalizing r with
  | nil => exact hr
  | cons qb B ih =>
    rw [reduce3_cons]
    exact ih (fun q hq => hB q (by simp [hq])) (disj_step3 hr (hB qb (by simp)))

lemma low_step3 {P : ℕ} {r : ℕ × ℕ} (hr : Low P r) {qb : ℕ × (ℕ × ℕ)} (hb : Low P qb.2) : Low P (step3 r qb) := by
  rw [step3_eq]
  split_ifs
  · exact low_add3 hr (low_neg3 hb)
  · exact low_add3 hr hb
  · exact hr

lemma low_reduce3 (B : List (ℕ × (ℕ × ℕ))) {P : ℕ} (hB : ∀ q ∈ B, Low P q.2) {r : ℕ × ℕ} (hr : Low P r) :
    Low P (reduce3 B r) := by
  induction B generalizing r with
  | nil => exact hr
  | cons qb B ih =>
    rw [reduce3_cons]
    exact ih (fun q hq => hB q (by simp [hq])) (low_step3 hr (hB qb (by simp)))

/-- Steps with rows below `2^P` do not touch position `P`. -/
lemma reduce3_testBit_hi (B : List (ℕ × (ℕ × ℕ))) {P : ℕ} (hP : ∀ q ∈ B, Low P q.2) (r : ℕ × ℕ) :
    (reduce3 B r).1.testBit P = r.1.testBit P ∧ (reduce3 B r).2.testBit P = r.2.testBit P := by
  induction B generalizing r with
  | nil => exact ⟨rfl, rfl⟩
  | cons qb B ih =>
    rw [reduce3_cons]
    obtain ⟨i1, i2⟩ := ih (fun q hq => hP q (by simp [hq])) (step3 r qb)
    rw [i1, i2, step3_eq]
    have hb := hP qb (by simp) P le_rfl
    split_ifs
    · exact add3_zero_at (x := r) (y := neg3 qb.2) hb.2 hb.1
    · exact add3_zero_at (x := r) (y := qb.2) hb.1 hb.2
    · exact ⟨rfl, rfl⟩

lemma step3_clear {r : ℕ × ℕ} (hr : Disj r) {qb : ℕ × (ℕ × ℕ)} (hb : Disj qb.2) (h1 : qb.2.1.testBit qb.1 = true) :
    (step3 r qb).1.testBit qb.1 = false ∧ (step3 r qb).2.testBit qb.1 = false := by
  have hb2 : qb.2.2.testBit qb.1 = false := by
    have := hb qb.1; simp only [h1, true_and] at this; simpa using this
  have hr' := hr qb.1
  rw [step3_eq]
  split_ifs with ha hc
  · have hc' : r.2.testBit qb.1 = false := by simp only [ha, true_and] at hr'; simpa using hr'
    rw [testBit_add3_fst, testBit_add3_snd]
    simp [neg3, ha, hc', h1, hb2]
  · simp only [Bool.not_eq_true] at ha
    rw [testBit_add3_fst, testBit_add3_snd]
    simp [ha, hc, h1, hb2]
  · simp only [Bool.not_eq_true] at *
    exact ⟨by assumption, by assumption⟩

lemma reduce3_clear (B : List (ℕ × (ℕ × ℕ))) (hB : Good3 w B) {r : ℕ × ℕ} (hr : Disj r) :
    ∀ q ∈ B, (reduce3 B r).1.testBit q.1 = false ∧ (reduce3 B r).2.testBit q.1 = false := by
  induction B generalizing r with
  | nil => simp
  | cons qb B ih =>
    obtain ⟨-, h2, h3, -, h5, hB'⟩ := hB
    intro q hq; rcases List.mem_cons.mp hq with rfl | hq
    · rw [reduce3_cons]
      obtain ⟨i1, i2⟩ := reduce3_testBit_hi B (good3_low B hB' h5) (step3 r q)
      rw [i1, i2]
      exact step3_clear hr h3 h2
    · exact ih hB' (disj_step3 hr h3) q hq

lemma span3_insert_add {s : Set (Fin w → ZMod 3)} {x t : Fin w → ZMod 3} (ht : t ∈ span (ZMod 3) s) :
    span (ZMod 3) (insert (x + t) s) = span (ZMod 3) (insert x s) := by
  have ht' : ∀ u, t ∈ span (ZMod 3) (insert u s) := fun u => span_mono (Set.subset_insert _ _) ht
  apply le_antisymm <;> rw [span_le] <;> rintro y (rfl | hy)
  · exact add_mem (subset_span (Set.mem_insert _ _)) (ht' _)
  · exact subset_span (Set.mem_insert_of_mem _ hy)
  · have h : (y + t) - t ∈ span (ZMod 3) (insert (y + t) s) := sub_mem (subset_span (Set.mem_insert _ _)) (ht' _)
    rw [add_sub_cancel_right] at h
    exact h
  · exact subset_span (Set.mem_insert_of_mem _ hy)

lemma span3_insert_neg {s : Set (Fin w → ZMod 3)} (x : Fin w → ZMod 3) :
    span (ZMod 3) (insert (-x) s) = span (ZMod 3) (insert x s) := by
  apply le_antisymm <;> rw [span_le] <;> rintro y (rfl | hy)
  · exact neg_mem (subset_span (Set.mem_insert _ _))
  · exact subset_span (Set.mem_insert_of_mem _ hy)
  · have h : - -y ∈ span (ZMod 3) (insert (-y) s) := neg_mem (subset_span (Set.mem_insert (-y) s))
    rwa [neg_neg] at h
  · exact subset_span (Set.mem_insert_of_mem _ hy)

lemma reduce3_vec (B : List (ℕ × (ℕ × ℕ))) (hB : ∀ q ∈ B, Disj q.2) {r : ℕ × ℕ} (hr : Disj r) :
    ∃ t ∈ span (ZMod 3) (rowSet3 w B), vec3 w (reduce3 B r) = vec3 w r + t := by
  induction B generalizing r with
  | nil => exact ⟨0, zero_mem _, by simp [reduce3_nil]⟩
  | cons qb B ih =>
    have hb := hB qb (by simp)
    obtain ⟨t, ht, he⟩ := ih (fun q hq => hB q (by simp [hq])) (disj_step3 hr hb)
    have hmono : span (ZMod 3) (rowSet3 w B) ≤ span (ZMod 3) (rowSet3 w (qb :: B)) :=
      span_mono (by rw [rowSet3_cons]; exact Set.subset_insert _ _)
    have hqb : vec3 w qb.2 ∈ span (ZMod 3) (rowSet3 w (qb :: B)) :=
      subset_span (by rw [rowSet3_cons]; exact Set.mem_insert _ _)
    rw [reduce3_cons, he, step3_eq]
    split_ifs
    · refine ⟨-vec3 w qb.2 + t, add_mem (neg_mem hqb) (hmono ht), ?_⟩
      rw [vec3_add3 hr (disj_neg3 hb), vec3_neg3 hb, add_assoc]
    · refine ⟨vec3 w qb.2 + t, add_mem hqb (hmono ht), ?_⟩
      rw [vec3_add3 hr hb, add_assoc]
    · exact ⟨t, hmono ht, rfl⟩

/-! ### `insertP3` -/

lemma mem_insertP3 (p : ℕ) (r : ℕ × ℕ) (B : List (ℕ × (ℕ × ℕ))) (x : ℕ × (ℕ × ℕ)) :
    x ∈ insertP3 p r B ↔ x = (p, r) ∨ x ∈ B := by
  induction B with
  | nil => simp [insertP3_nil]
  | cons qb B ih =>
    rw [insertP3_cons]
    by_cases hlt : qb.1 < p
    · simp [hlt]
    · simp [hlt, ih]; tauto

lemma length_insertP3 (p : ℕ) (r : ℕ × ℕ) (B : List (ℕ × (ℕ × ℕ))) :
    (insertP3 p r B).length = B.length + 1 := by
  induction B with
  | nil => rfl
  | cons qb B ih =>
    rw [insertP3_cons]
    by_cases hlt : qb.1 < p <;> simp [hlt, ih]

lemma rowSet3_insertP3 (p : ℕ) (r : ℕ × ℕ) (B : List (ℕ × (ℕ × ℕ))) :
    rowSet3 w (insertP3 p r B) = insert (vec3 w r) (rowSet3 w B) := by
  ext v; simp only [rowSet3, Set.mem_ofPred_eq, mem_insertP3, Set.mem_insert_iff]
  constructor
  · rintro ⟨qb, rfl | h, rfl⟩
    · exact Or.inl rfl
    · exact Or.inr ⟨qb, h, rfl⟩
  · rintro (rfl | ⟨qb, h, rfl⟩)
    · exact ⟨(p, r), Or.inl rfl, rfl⟩
    · exact ⟨qb, Or.inr h, rfl⟩

lemma good3_insertP3 (p : ℕ) (r : ℕ × ℕ) (B : List (ℕ × (ℕ × ℕ))) (hB : Good3 w B) (h1 : Low (p + 1) r)
    (h2 : r.1.testBit p = true) (h3 : Disj r) (h4 : p < w) (hne : ∀ q ∈ B, q.1 ≠ p) : Good3 w (insertP3 p r B) := by
  induction B with
  | nil => exact ⟨h1, h2, h3, h4, by simp, trivial⟩
  | cons qb B ih =>
    obtain ⟨g1, g2, g3, g4, g5, gB⟩ := hB
    rw [insertP3_cons]
    by_cases hlt : qb.1 < p
    · rw [if_pos hlt]
      refine ⟨h1, h2, h3, h4, ?_, g1, g2, g3, g4, g5, gB⟩
      intro q hq; rcases List.mem_cons.mp hq with rfl | hq
      · exact hlt
      · exact lt_trans (g5 q hq) hlt
    · rw [if_neg hlt]
      have hgt : p < qb.1 := by
        have := hne qb (by simp); omega
      refine ⟨g1, g2, g3, g4, ?_, ih gB (fun q hq => hne q (by simp [hq]))⟩
      intro q hq
      rcases (mem_insertP3 p r B q).mp hq with rfl | hq
      · exact hgt
      · exact g5 q hq

/-! ### One row, then all rows -/

lemma addRow3_spec (B : List (ℕ × (ℕ × ℕ))) (hB : Good3 w B) {r : ℕ × ℕ} (hr : Disj r) (hrw : Low w r) :
    Good3 w (addRow3 B r) ∧
      span (ZMod 3) (rowSet3 w (addRow3 B r)) = span (ZMod 3) (insert (vec3 w r) (rowSet3 w B)) := by
  obtain ⟨t, ht, he⟩ := reduce3_vec (w := w) B (good3_disj B hB) hr
  set r' := reduce3 B r with hr'
  have hd : Disj r' := disj_reduce3 B (good3_disj B hB) hr
  have hl : Low w r' := low_reduce3 B (good3_low_w B hB) hrw
  rw [addRow3_eq, addRed3_eq, ← hr']
  by_cases h0 : supp r' = 0
  · rw [if_pos h0]
    refine ⟨hB, ?_⟩
    rw [vec3_zero_of_supp h0] at he
    have hmem : vec3 w r ∈ span (ZMod 3) (rowSet3 w B) := by
      rw [eq_neg_of_add_eq_zero_left he.symm]; exact neg_mem ht
    rw [span_insert_eq_span hmem]
  · rw [if_neg h0]
    set p := Nat.log2 (supp r')
    obtain ⟨f1, f2⟩ := log2_facts h0
    have hlow : Low (p + 1) r' := (low_iff_supp _ _).mpr f1
    have hpw : p < w := by
      have := lt_of_le_of_lt (Nat.log2_self_le h0) ((low_iff_supp _ _).mp hl)
      exact (Nat.pow_lt_pow_iff_right (by norm_num)).mp this
    rw [testBit_supp] at f2
    have hn1 : (norm3 p r').1.testBit p = true := by
      rw [norm3_eq]; split_ifs with h
      · exact h
      · have h' : r'.1.testBit p = false := by simpa using h
        rw [h', Bool.false_or] at f2; exact f2
    have hnd : Disj (norm3 p r') := by rw [norm3_eq]; split_ifs; exacts [hd, disj_neg3 hd]
    have hnl : Low (p + 1) (norm3 p r') := by rw [norm3_eq]; split_ifs; exacts [hlow, low_neg3 hlow]
    refine ⟨good3_insertP3 _ _ B hB hnl hn1 hnd hpw fun q hq he' => ?_, ?_⟩
    · have := reduce3_clear B hB hr q hq
      rw [← hr', he'] at this
      rw [this.1, this.2] at f2; exact Bool.false_ne_true f2
    · have hn : span (ZMod 3) (insert (vec3 w (norm3 p r')) (rowSet3 w B)) =
          span (ZMod 3) (insert (vec3 w r') (rowSet3 w B)) := by
        rw [norm3_eq]; split_ifs
        · rfl
        · rw [vec3_neg3 hd, span3_insert_neg]
      rw [rowSet3_insertP3, hn, he, span3_insert_add ht]

lemma basis3_spec (B : List (ℕ × (ℕ × ℕ))) (rs : List (ℕ × ℕ)) (hB : Good3 w B)
    (hrs : ∀ r ∈ rs, Disj r ∧ Low w r) :
    Good3 w (basis3 rs B) ∧
      span (ZMod 3) (rowSet3 w (basis3 rs B)) = span (ZMod 3) (rowSet3 w B ∪ vec3 w '' {r | r ∈ rs}) := by
  induction rs generalizing B with
  | nil => simp [basis3_nil, hB]
  | cons r rs ih =>
    obtain ⟨g, hs⟩ := addRow3_spec B hB (hrs r (by simp)).1 (hrs r (by simp)).2
    obtain ⟨g', hs'⟩ := ih (addRow3 B r) g (fun x hx => hrs x (by simp [hx]))
    refine ⟨g', ?_⟩
    rw [basis3_cons, hs', span_union, hs, ← span_union]
    congr 1
    ext v; simp only [Set.mem_union, Set.mem_insert_iff, Set.mem_image, Set.mem_ofPred_eq, List.mem_cons]
    constructor
    · rintro ((rfl | h) | ⟨x, hx, rfl⟩)
      · exact Or.inr ⟨r, Or.inl rfl, rfl⟩
      · exact Or.inl h
      · exact Or.inr ⟨x, Or.inr hx, rfl⟩
    · rintro (h | ⟨x, rfl | hx, rfl⟩)
      · exact Or.inl (Or.inr h)
      · exact Or.inl (Or.inl rfl)
      · exact Or.inr ⟨x, hx, rfl⟩

/-! ### Rows with distinct normalised pivots are independent -/

lemma finrank_good3 (B : List (ℕ × (ℕ × ℕ))) (hB : Good3 w B) :
    Module.finrank (ZMod 3) (span (ZMod 3) (rowSet3 w B)) = B.length := by
  induction B with
  | nil =>
    rw [rowSet3_nil]
    simp
  | cons qb B ih =>
    obtain ⟨h1, h2, h3, h4, h5, hB'⟩ := hB
    set j : Fin w := ⟨qb.1, h4⟩
    have hz : ∀ v ∈ span (ZMod 3) (rowSet3 w B), v j = 0 := by
      intro v hv
      induction hv using Submodule.span_induction with
      | mem x hx =>
        obtain ⟨q, hq, rfl⟩ := hx
        have := good3_low B hB' h5 q hq qb.1 le_rfl
        simp [vec3, val3, j, this.1, this.2]
      | zero => rfl
      | add x y _ _ hx hy => simp [hx, hy]
      | smul a x _ hx => simp [hx]
    have hv1 : vec3 w qb.2 j = 1 := by simp [vec3, val3, j, h2]
    have hnot : vec3 w qb.2 ∉ span (ZMod 3) (rowSet3 w B) := fun h => by
      have := hz _ h; rw [hv1] at this; exact one_ne_zero this
    have hne : vec3 w qb.2 ≠ 0 := fun h => by
      have := congrFun h j; rw [hv1] at this; exact one_ne_zero this
    have hdisj : Disjoint (span (ZMod 3) (rowSet3 w B)) (ZMod 3 ∙ vec3 w qb.2) :=
      (Submodule.disjoint_span_singleton' hne).mpr hnot
    have key := Submodule.finrank_sup_add_finrank_inf_eq (ZMod 3 ∙ vec3 w qb.2) (span (ZMod 3) (rowSet3 w B))
    rw [hdisj.symm.eq_bot, finrank_bot, add_zero, finrank_span_singleton hne, ih hB'] at key
    rw [rowSet3_cons, span_insert, key, List.length_cons, add_comm]

/-! ### The theorem -/

/-- The matrix over `ZMod 3` with the given two-plane rows and `w` columns. -/
def toMat3 (w : ℕ) (rows : List (ℕ × ℕ)) : Matrix (Fin rows.length) (Fin w) (ZMod 3) :=
  Matrix.of fun i j => vec3 w rows[i] j

lemma range_toMat3 (rows : List (ℕ × ℕ)) : Set.range (toMat3 w rows).row = vec3 w '' {r | r ∈ rows} := by
  ext v
  simp only [Set.mem_image, Set.mem_ofPred_eq, toMat3, Matrix.row]
  constructor
  · rintro ⟨i, rfl⟩
    exact ⟨rows[i], List.getElem_mem _, rfl⟩
  · rintro ⟨r, hr, rfl⟩
    obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem hr
    exact ⟨⟨i, hi⟩, rfl⟩

/-- **`rankF3` is the rank over `F₃`**: for rows with disjoint planes below `2^w`, `rankF3 rows` is `Matrix.rank` of the
`rows.length × w` matrix over `ZMod 3` of their entries. -/
theorem rankF3_eq_rank (w : ℕ) (rows : List (ℕ × ℕ)) (h : ∀ r ∈ rows, Disj r ∧ Low w r) :
    rankF3 rows = (toMat3 w rows).rank := by
  obtain ⟨g, hs⟩ := basis3_spec (w := w) [] rows trivial h
  rw [Matrix.rank_eq_finrank_span_row, range_toMat3, rankF3, ← finrank_good3 _ g, hs, rowSet3_nil, Set.empty_union]

/-- `rowsOK3` is the hypothesis of `rankF3_eq_rank`. -/
theorem rowsOK3_spec (w : ℕ) (rows : List (ℕ × ℕ)) (h : rowsOK3 w rows = true) : ∀ r ∈ rows, Disj r ∧ Low w r := by
  induction rows with
  | nil => simp
  | cons r rs ih =>
    have h' : ((Nat.beq (Nat.land r.1 r.2) 0 && Nat.blt (supp r) (Nat.pow 2 w)) && rowsOK3 w rs) = true := h
    simp only [Bool.and_eq_true] at h'
    obtain ⟨⟨hd, hl⟩, hrs⟩ := h'
    intro x hx; rcases List.mem_cons.mp hx with rfl | hx
    · refine ⟨fun j ⟨h1, h2⟩ => ?_, (low_iff_supp _ _).mpr (by simpa [Nat.blt_eq] using hl)⟩
      have := congrArg (fun y => y.testBit j) (Nat.eq_of_beq_eq_true hd)
      simp [h1, h2] at this
    · exact ih hrs x hx

end GL2.L23.RankF3
