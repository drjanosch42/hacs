/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import Mathlib
import GL2.L23.RankF2Exec

/-!
# LV-0 (slot 22): `rankF2` is `Matrix.rank` over `ZMod 2`

`L23_master_plan.md` §4.2(2) (verified algorithms instead of witness data). The executable elimination
`GL2.L23.RankF2.rankF2` (`RankF2Exec.lean`, Mathlib-free) is proved equal to mathlib's `Matrix.rank` of the
`rows.length × w` matrix over `ZMod 2` whose entry `(i, j)` is bit `j` of row `i`, for every list of rows `< 2^w`
(`rankF2_eq_rank`). A cell theorem is then `decide +kernel` on `rankF2 rows = k` (no literal data) followed by this
theorem (`rank_of_decide`).

**Why a verified algorithm and not a checker + witness.** A pivot-form witness for the rank of a `d × d` matrix costs
`≈ 4d²` bits of literal data, which caps it at `d ≈ 200` under the 50 KB-per-module budget of §4.1; the algorithm needs
no data, so its limit is kernel time, which LV-0 measures (`docs/implementation/slot22/LV0_report.md`).

**Invariant.** The basis is a list of `(p, b)` with `2^p ≤ b < 2^(p+1)` (so `p` is the top bit of `b`), `p < w`, and
strictly descending pivots (`Good`). One pass of `reduce` clears every pivot bit of a row (`reduce_clear`) and changes
it by an element of the span of the basis (`reduce_vec`); a non-zero remainder is new at its top bit, so `insertP`
keeps `Good` and the span grows by exactly that row. Rows with distinct top bits are independent (`finrank_good`).
-/

namespace GL2.L23.RankF2

open Submodule

/-- Bit vector of length `w`: bit `j` of `x` as an element of `ZMod 2`. -/
def vec (w x : ℕ) : Fin w → ZMod 2 := fun j => if x.testBit j then 1 else 0

/-- The matrix over `ZMod 2` with the given bit-packed rows and `w` columns. -/
def toMat (w : ℕ) (rows : List ℕ) : Matrix (Fin rows.length) (Fin w) (ZMod 2) :=
  Matrix.of fun i j => vec w rows[i] j

/-- The row vectors of a basis. -/
def rowSet (w : ℕ) (B : List (ℕ × ℕ)) : Set (Fin w → ZMod 2) := {v | ∃ pb ∈ B, vec w pb.2 = v}

/-- The basis invariant. -/
def Good (w : ℕ) : List (ℕ × ℕ) → Prop
  | [] => True
  | pb :: B => pb.2 < 2 ^ (pb.1 + 1) ∧ pb.2.testBit pb.1 = true ∧ pb.1 < w ∧ (∀ q ∈ B, q.1 < pb.1) ∧ Good w B

variable {w : ℕ}

/-! ### The kernel formulation, read back -/

lemma bit_eq (r p : ℕ) : bit r p = r.testBit p := by
  show Nat.beq ((r >>> p) &&& 1) 1 = _
  rw [Nat.testBit_eq_decide_div_mod_eq, Nat.shiftRight_eq_div_pow, Nat.and_one_is_mod]
  cases h : Nat.beq (r / 2 ^ p % 2) 1
  · exact (decide_eq_false fun h' => by simp [h'] at h).symm
  · exact (decide_eq_true (Nat.eq_of_beq_eq_true h)).symm

lemma step_eq (r : ℕ) (pb : ℕ × ℕ) : step r pb = if r.testBit pb.1 then r ^^^ pb.2 else r := by
  rw [step, bit_eq]; cases r.testBit pb.1 <;> rfl

lemma reduce_nil (r : ℕ) : reduce [] r = r := rfl

lemma reduce_cons (pb : ℕ × ℕ) (B : List (ℕ × ℕ)) (r : ℕ) : reduce (pb :: B) r = reduce B (step r pb) := rfl

lemma insertP_nil (p r : ℕ) : insertP p r [] = [(p, r)] := rfl

lemma insertP_cons (p r : ℕ) (pb : ℕ × ℕ) (B : List (ℕ × ℕ)) :
    insertP p r (pb :: B) = if pb.1 < p then (p, r) :: pb :: B else pb :: insertP p r B := by
  show @Bool.rec (fun _ => List (ℕ × ℕ)) (pb :: insertP p r B) ((p, r) :: pb :: B) (Nat.blt pb.1 p) = _
  by_cases h : pb.1 < p
  · rw [if_pos h, show Nat.blt pb.1 p = true from Nat.blt_eq.mpr h]
  · rw [if_neg h, show Nat.blt pb.1 p = false from Bool.eq_false_iff.mpr fun h' => h (Nat.blt_eq.mp h')]

lemma addRow_eq (B : List (ℕ × ℕ)) (r : ℕ) :
    addRow B r = if reduce B r = 0 then B else insertP (Nat.log2 (reduce B r)) (reduce B r) B := by
  rw [addRow]
  by_cases h : reduce B r = 0
  · rw [if_pos h, h]; rfl
  · rw [if_neg h, show Nat.beq (reduce B r) 0 = false from Bool.eq_false_iff.mpr fun h' => h (Nat.eq_of_beq_eq_true h')]

lemma basis_nil (B : List (ℕ × ℕ)) : basis [] B = B := rfl

lemma basis_cons (r : ℕ) (rs : List ℕ) (B : List (ℕ × ℕ)) : basis (r :: rs) B = basis rs (addRow B r) := rfl

lemma vec_xor (a b : ℕ) : vec w (a ^^^ b) = vec w a + vec w b := by
  funext j
  simp only [vec, Nat.testBit_xor, Pi.add_apply]
  cases a.testBit j <;> cases b.testBit j <;> decide

lemma vec_zero : vec w 0 = 0 := by funext j; simp [vec]

lemma rowSet_cons (pb : ℕ × ℕ) (B : List (ℕ × ℕ)) :
    rowSet w (pb :: B) = insert (vec w pb.2) (rowSet w B) := by
  ext v; simp [rowSet, eq_comm]

lemma rowSet_nil : rowSet w [] = ∅ := Set.eq_empty_of_forall_notMem (by simp [rowSet])

lemma span_insert_add {s : Set (Fin w → ZMod 2)} {x t : Fin w → ZMod 2} (ht : t ∈ span (ZMod 2) s) :
    span (ZMod 2) (insert (x + t) s) = span (ZMod 2) (insert x s) := by
  have ht' : ∀ u, t ∈ span (ZMod 2) (insert u s) := fun u => span_mono (Set.subset_insert _ _) ht
  apply le_antisymm <;> rw [span_le] <;> rintro y (rfl | hy)
  · exact add_mem (subset_span (Set.mem_insert _ _)) (ht' _)
  · exact subset_span (Set.mem_insert_of_mem _ hy)
  · have h : (y + t) - t ∈ span (ZMod 2) (insert (y + t) s) := sub_mem (subset_span (Set.mem_insert _ _)) (ht' _)
    rw [add_sub_cancel_right] at h
    exact h
  · exact subset_span (Set.mem_insert_of_mem _ hy)

/-! ### `reduce` -/

lemma good_lt (B : List (ℕ × ℕ)) (hB : Good w B) {P : ℕ} (hP : ∀ q ∈ B, q.1 < P) : ∀ q ∈ B, q.2 < 2 ^ P := by
  induction B with
  | nil => simp
  | cons pb B ih =>
    obtain ⟨h1, -, -, -, hB⟩ := hB
    intro q hq; rcases List.mem_cons.mp hq with rfl | hq
    · exact lt_of_lt_of_le h1 (Nat.pow_le_pow_right (by norm_num) (hP q (by simp)))
    · exact ih hB (fun q hq => hP q (by simp [hq])) q hq

lemma good_lt_w (B : List (ℕ × ℕ)) (hB : Good w B) : ∀ q ∈ B, q.2 < 2 ^ w := by
  induction B with
  | nil => simp
  | cons pb B ih =>
    obtain ⟨h1, -, h3, -, hB⟩ := hB
    intro q hq; rcases List.mem_cons.mp hq with rfl | hq
    · exact lt_of_lt_of_le h1 (Nat.pow_le_pow_right (by norm_num) h3)
    · exact ih hB q hq

lemma reduce_testBit_hi (B : List (ℕ × ℕ)) {P : ℕ} (hP : ∀ q ∈ B, q.2 < 2 ^ P) (r : ℕ) :
    (reduce B r).testBit P = r.testBit P := by
  induction B generalizing r with
  | nil => rfl
  | cons pb B ih =>
    rw [reduce_cons, ih (fun q hq => hP q (by simp [hq])), step_eq]
    cases r.testBit pb.1 <;> simp [Nat.testBit_xor, Nat.testBit_eq_false_of_lt (hP pb (by simp))]

lemma reduce_clear (B : List (ℕ × ℕ)) (hB : Good w B) (r : ℕ) : ∀ q ∈ B, (reduce B r).testBit q.1 = false := by
  induction B generalizing r with
  | nil => simp
  | cons pb B ih =>
    obtain ⟨-, h2, -, h4, hB'⟩ := hB
    intro q hq; rcases List.mem_cons.mp hq with rfl | hq
    · rw [reduce_cons, reduce_testBit_hi B (good_lt B hB' h4), step_eq]
      cases h : r.testBit q.1 <;> simp [h, Nat.testBit_xor, h2]
    · exact ih hB' _ q hq

lemma reduce_vec (B : List (ℕ × ℕ)) (r : ℕ) :
    ∃ t ∈ span (ZMod 2) (rowSet w B), vec w (reduce B r) = vec w r + t := by
  induction B generalizing r with
  | nil => exact ⟨0, zero_mem _, by simp [reduce_nil]⟩
  | cons pb B ih =>
    obtain ⟨t, ht, he⟩ := ih (step r pb)
    have hmono : span (ZMod 2) (rowSet w B) ≤ span (ZMod 2) (rowSet w (pb :: B)) :=
      span_mono (by rw [rowSet_cons]; exact Set.subset_insert _ _)
    have hpb : vec w pb.2 ∈ span (ZMod 2) (rowSet w (pb :: B)) :=
      subset_span (by rw [rowSet_cons]; exact Set.mem_insert _ _)
    rw [reduce_cons, he, step_eq]
    cases r.testBit pb.1
    · exact ⟨t, hmono ht, rfl⟩
    · refine ⟨vec w pb.2 + t, add_mem hpb (hmono ht), ?_⟩
      simp [vec_xor, add_assoc]

lemma reduce_lt (B : List (ℕ × ℕ)) (hB : ∀ q ∈ B, q.2 < 2 ^ w) {r : ℕ} (hr : r < 2 ^ w) : reduce B r < 2 ^ w := by
  induction B generalizing r with
  | nil => exact hr
  | cons pb B ih =>
    rw [reduce_cons]
    refine ih (fun q hq => hB q (by simp [hq])) ?_
    rw [step_eq]
    cases r.testBit pb.1
    · exact hr
    · exact Nat.xor_lt_two_pow hr (hB pb (by simp))

/-! ### `insertP` -/

lemma mem_insertP (p r : ℕ) (B : List (ℕ × ℕ)) (x : ℕ × ℕ) : x ∈ insertP p r B ↔ x = (p, r) ∨ x ∈ B := by
  induction B with
  | nil => simp [insertP_nil]
  | cons pb B ih =>
    rw [insertP_cons]
    by_cases hlt : pb.1 < p
    · simp [hlt]
    · simp [hlt, ih]; tauto

lemma length_insertP (p r : ℕ) (B : List (ℕ × ℕ)) : (insertP p r B).length = B.length + 1 := by
  induction B with
  | nil => rfl
  | cons pb B ih =>
    rw [insertP_cons]
    by_cases hlt : pb.1 < p <;> simp [hlt, ih]

lemma rowSet_insertP (p r : ℕ) (B : List (ℕ × ℕ)) : rowSet w (insertP p r B) = insert (vec w r) (rowSet w B) := by
  ext v; simp only [rowSet, Set.mem_ofPred_eq, mem_insertP, Set.mem_insert_iff]
  constructor
  · rintro ⟨pb, rfl | h, rfl⟩
    · exact Or.inl rfl
    · exact Or.inr ⟨pb, h, rfl⟩
  · rintro (rfl | ⟨pb, h, rfl⟩)
    · exact ⟨(p, r), Or.inl rfl, rfl⟩
    · exact ⟨pb, Or.inr h, rfl⟩

lemma good_insertP (p r : ℕ) (B : List (ℕ × ℕ)) (hB : Good w B) (h1 : r < 2 ^ (p + 1)) (h2 : r.testBit p = true)
    (h3 : p < w) (hne : ∀ q ∈ B, q.1 ≠ p) : Good w (insertP p r B) := by
  induction B with
  | nil => exact ⟨h1, h2, h3, by simp, trivial⟩
  | cons pb B ih =>
    obtain ⟨g1, g2, g3, g4, gB⟩ := hB
    rw [insertP_cons]
    by_cases hlt : pb.1 < p
    · rw [if_pos hlt]
      refine ⟨h1, h2, h3, ?_, g1, g2, g3, g4, gB⟩
      intro q hq; rcases List.mem_cons.mp hq with rfl | hq
      · exact hlt
      · exact lt_trans (g4 q hq) hlt
    · rw [if_neg hlt]
      have hgt : p < pb.1 := by
        have := hne pb (by simp); omega
      refine ⟨g1, g2, g3, ?_, ih gB (fun q hq => hne q (by simp [hq]))⟩
      intro q hq
      rcases (mem_insertP p r B q).mp hq with rfl | hq
      · exact hgt
      · exact g4 q hq

/-! ### One row, then all rows -/

lemma log2_facts {r : ℕ} (hr : r ≠ 0) : r < 2 ^ (r.log2 + 1) ∧ r.testBit r.log2 = true := by
  have hlo : 2 ^ r.log2 ≤ r := Nat.log2_self_le hr
  have hhi : r < 2 ^ (r.log2 + 1) := Nat.lt_log2_self
  refine ⟨hhi, ?_⟩
  rw [Nat.testBit_eq_decide_div_mod_eq]
  have h1 : 1 ≤ r / 2 ^ r.log2 := (Nat.le_div_iff_mul_le (by positivity)).mpr (by simpa using hlo)
  have h2 : r / 2 ^ r.log2 < 2 := (Nat.div_lt_iff_lt_mul (by positivity)).mpr (by rw [← pow_succ']; exact hhi)
  have : r / 2 ^ r.log2 = 1 := by omega
  simp [this]

lemma addRow_spec (B : List (ℕ × ℕ)) (hB : Good w B) {r : ℕ} (hr : r < 2 ^ w) :
    Good w (addRow B r) ∧ span (ZMod 2) (rowSet w (addRow B r)) = span (ZMod 2) (insert (vec w r) (rowSet w B)) := by
  obtain ⟨t, ht, he⟩ := reduce_vec (w := w) B r
  have hlt : reduce B r < 2 ^ w := reduce_lt B (good_lt_w B hB) hr
  rw [addRow_eq]
  by_cases h0 : reduce B r = 0
  · rw [if_pos h0]
    refine ⟨hB, ?_⟩
    rw [h0, vec_zero] at he
    have hmem : vec w r ∈ span (ZMod 2) (rowSet w B) := by
      rw [eq_neg_of_add_eq_zero_left he.symm]; exact neg_mem ht
    rw [span_insert_eq_span hmem]
  · rw [if_neg h0]
    obtain ⟨f1, f2⟩ := log2_facts h0
    have f3 : (reduce B r).log2 < w := by
      have := lt_of_le_of_lt (Nat.log2_self_le h0) hlt
      exact (Nat.pow_lt_pow_iff_right (by norm_num)).mp this
    refine ⟨good_insertP _ _ B hB f1 f2 f3 fun q hq he' => ?_, ?_⟩
    · have := reduce_clear B hB r q hq; rw [he'] at this; simp [f2] at this
    · rw [rowSet_insertP, he, span_insert_add ht]

lemma basis_spec (B : List (ℕ × ℕ)) (rs : List ℕ) (hB : Good w B) (hrs : ∀ r ∈ rs, r < 2 ^ w) :
    Good w (basis rs B) ∧
      span (ZMod 2) (rowSet w (basis rs B)) = span (ZMod 2) (rowSet w B ∪ vec w '' {r | r ∈ rs}) := by
  induction rs generalizing B with
  | nil => simp [basis_nil, hB]
  | cons r rs ih =>
    obtain ⟨g, hs⟩ := addRow_spec B hB (hrs r (by simp))
    obtain ⟨g', hs'⟩ := ih (addRow B r) g (fun x hx => hrs x (by simp [hx]))
    refine ⟨g', ?_⟩
    rw [basis_cons, hs', span_union, hs, ← span_union]
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

/-! ### Rows with distinct top bits are independent -/

lemma finrank_good (B : List (ℕ × ℕ)) (hB : Good w B) :
    Module.finrank (ZMod 2) (span (ZMod 2) (rowSet w B)) = B.length := by
  induction B with
  | nil =>
    rw [rowSet_nil]
    simp
  | cons pb B ih =>
    obtain ⟨h1, h2, h3, h4, hB'⟩ := hB
    set j : Fin w := ⟨pb.1, h3⟩
    -- every vector of the span of the tail vanishes at the pivot `j`
    have hz : ∀ v ∈ span (ZMod 2) (rowSet w B), v j = 0 := by
      intro v hv
      induction hv using Submodule.span_induction with
      | mem x hx =>
        obtain ⟨q, hq, rfl⟩ := hx
        simp [vec, j, Nat.testBit_eq_false_of_lt (good_lt B hB' h4 q hq)]
      | zero => rfl
      | add x y _ _ hx hy => simp [hx, hy]
      | smul a x _ hx => simp [hx]
    have hv1 : vec w pb.2 j = 1 := by simp [vec, j, h2]
    have hnot : vec w pb.2 ∉ span (ZMod 2) (rowSet w B) := fun h => by
      have := hz _ h; rw [hv1] at this; exact one_ne_zero this
    have hne : vec w pb.2 ≠ 0 := fun h => by
      have := congrFun h j; rw [hv1] at this; exact one_ne_zero this
    have hdisj : Disjoint (span (ZMod 2) (rowSet w B)) (ZMod 2 ∙ vec w pb.2) :=
      (Submodule.disjoint_span_singleton' hne).mpr hnot
    have key := Submodule.finrank_sup_add_finrank_inf_eq (ZMod 2 ∙ vec w pb.2) (span (ZMod 2) (rowSet w B))
    rw [hdisj.symm.eq_bot, finrank_bot, add_zero, finrank_span_singleton hne, ih hB'] at key
    rw [rowSet_cons, span_insert, key, List.length_cons, add_comm]

/-! ### The theorem -/

lemma range_toMat (rows : List ℕ) : Set.range (toMat w rows).row = vec w '' {r | r ∈ rows} := by
  ext v
  simp only [Set.mem_image, Set.mem_ofPred_eq, toMat, Matrix.row]
  constructor
  · rintro ⟨i, rfl⟩
    exact ⟨rows[i], List.getElem_mem _, rfl⟩
  · rintro ⟨r, hr, rfl⟩
    obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem hr
    exact ⟨⟨i, hi⟩, rfl⟩

/-- **`rankF2` is the rank over `F₂`**: for rows `< 2^w`, `rankF2 rows` is `Matrix.rank` of the `rows.length × w`
matrix over `ZMod 2` of their bits. -/
theorem rankF2_eq_rank (w : ℕ) (rows : List ℕ) (h : ∀ r ∈ rows, r < 2 ^ w) :
    rankF2 rows = (toMat w rows).rank := by
  obtain ⟨g, hs⟩ := basis_spec (w := w) [] rows trivial h
  rw [Matrix.rank_eq_finrank_span_row, range_toMat, rankF2, ← finrank_good _ g, hs, rowSet_nil, Set.empty_union]

/-- The bound `rows < 2^w`, as a `Bool` the kernel decides. -/
def rowsLt (w : ℕ) (rows : List ℕ) : Bool := rows.all fun r => r < 2 ^ w

/-- **The cell-theorem pattern**: two `decide +kernel` facts give the mathlib rank. -/
theorem rank_of_decide (w : ℕ) (rows : List ℕ) (k : ℕ) (hlt : rowsLt w rows = true) (hk : rankF2 rows = k) :
    (toMat w rows).rank = k := by
  rw [← rankF2_eq_rank w rows (by simpa [rowsLt] using hlt), hk]

end GL2.L23.RankF2
