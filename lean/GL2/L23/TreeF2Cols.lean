/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import Mathlib
import GL2.L23.RankF2
import GL2.L23.TreeF2Exec
import GL2.Checkers.RhoExec

/-!
# LV-1 (slot 22): the packed columns are the columns of `ρ_n` mod 2 (the F1 precondition)

`LV0_audit.md` R1 (F1): before any census `lean = 1`, prove that the columns the checker consumes are the mod-2 reductions of
the actual `Sym^n` matrices ([C] `eq:rhondef`, executable as `GL2.Exec.rhoEntry`, specified by `subst_u_eq_sum_rhoEntry`).

* `testBit_pascal`: bit `t` of `pascal m` is `C(m, t) mod 2` (Pascal's rule; no Lucas theorem needed).
* **`testBit_colS`, `testBit_colU`, `testBit_colU2`** (F1, bit form, every `t`): the packed column `j` of `ρ_n(S)`, `ρ_n(U)`,
  `ρ_n(U²)` mod 2 has bit `t` set iff `rhoEntry n a b c d t j` (over `ℕ`, the integer matrix entries `0, 1` of
  `S ≡ !![0,1;1,0]`, `U ≡ !![0,1;1,1]`, `U² ≡ !![1,1;1,0]`) is odd — i.e. the column **is** `packBits (rhoEntry … mod 2)`.
* **`vec_colS`, `vec_colU`, `vec_colU2`** (F1, matrix form): `vec (n+1)` of the packed column is column `j` of
  `GL2.rhoMat n A` over `ZMod 2` (the coordinate matrix of `GL2.rho n A`, `GL2.Exec.rhoMat_eq`).
* `colsA_eq`, `colsSig_eq`: the kernel generators (`TreeF2Exec`) are the column lists `List.map … (List.range (n+1))`.
-/

namespace GL2.L23.RankF2

open GL2.Exec

/-! ### The kernel generators, read back -/

lemma force_eq {α : Sort _} (a : ℕ) (k : ℕ → α) : force a k = k a := by
  unfold force; cases Nat.beq a 0 <;> rfl

lemma pascal_zero : pascal 0 = 1 := rfl

lemma pascal_succ (m : ℕ) : pascal (m + 1) = pascal m ^^^ (pascal m <<< 1) := rfl

lemma choose_mod_two_add (a b : ℕ) :
    decide ((a + b) % 2 = 1) = (decide (a % 2 = 1) ^^ decide (b % 2 = 1)) := by
  rcases Nat.mod_two_eq_zero_or_one a with ha | ha <;> rcases Nat.mod_two_eq_zero_or_one b with hb | hb <;>
    simp [Nat.add_mod, ha, hb]

/-- **Bit `t` of `pascal m` is `C(m, t) mod 2`.** -/
theorem testBit_pascal (m t : ℕ) : (pascal m).testBit t = decide (m.choose t % 2 = 1) := by
  induction m generalizing t with
  | zero =>
    rw [pascal_zero]
    cases t with
    | zero => rfl
    | succ t => simp [Nat.testBit_succ]
  | succ m ih =>
    rw [pascal_succ, Nat.testBit_xor, Nat.testBit_shiftLeft]
    cases t with
    | zero => simp [ih]
    | succ t =>
      rw [ih, Nat.choose_succ_succ, add_comm (m.choose t), choose_mod_two_add]
      simp [ih]

lemma pasGo_succ (k p : ℕ) : pasGo (k + 1) p = p :: pasGo k (p ^^^ (p <<< 1)) := by
  show p :: force _ (pasGo k) = _
  rw [force_eq]; rfl

lemma pasGo_pascal (k j : ℕ) : pasGo k (pascal j) = (List.range k).map fun i => pascal (j + i) := by
  induction k generalizing j with
  | zero => rfl
  | succ k ih =>
    rw [pasGo_succ, ← pascal_succ, ih, List.range_succ_eq_map, List.map_cons, List.map_map]
    congr 1
    refine List.map_congr_left fun i _ => ?_
    simp only [Function.comp_apply]; congr 1; omega

lemma revGo_eq (l acc : List ℕ) : revGo l acc = l.reverse ++ acc := by
  induction l generalizing acc with
  | nil => rfl
  | cons x l ih => show revGo l (x :: acc) = _; rw [ih]; simp

lemma tabulate_eq (f : ℕ → ℕ) (k j : ℕ) : tabulate f k j = (List.range k).map fun i => f (j + i) := by
  induction k generalizing j with
  | zero => rfl
  | succ k ih =>
    show f j :: tabulate f k (j + 1) = _
    rw [ih, List.range_succ_eq_map, List.map_cons, List.map_map]
    congr 1
    refine List.map_congr_left fun i _ => ?_
    simp only [Function.comp_apply]; congr 1; omega

lemma n3Go_eq (n : ℕ) (P Q : List ℕ) (j : ℕ) (h : P.length = Q.length) :
    n3Go n P Q j = (List.range P.length).map fun i => colN3p n (j + i) (P.getD i 0) (Q.getD i 0) := by
  induction P generalizing Q j with
  | nil => rfl
  | cons p P ih =>
    cases Q with
    | nil => simp at h
    | cons q Q =>
      show force _ (fun c => c :: n3Go n P Q (j + 1)) = _
      rw [force_eq, ih Q (j + 1) (by simpa using h), List.length_cons, List.range_succ_eq_map, List.map_cons,
        List.map_map]
      congr 1
      refine List.map_congr_left fun i _ => ?_
      simp only [Function.comp_apply, List.getD_cons_succ]; congr 1; omega

lemma sigGo_eq (n : ℕ) (P : List ℕ) (j : ℕ) :
    sigGo n P j = (List.range P.length).map fun i => colSigp n (j + i) (P.getD i 0) := by
  induction P generalizing j with
  | nil => rfl
  | cons p P ih =>
    show force _ (fun c => c :: sigGo n P (j + 1)) = _
    rw [force_eq, ih (j + 1), List.length_cons, List.range_succ_eq_map, List.map_cons, List.map_map]
    congr 1
    refine List.map_congr_left fun i _ => ?_
    simp only [Function.comp_apply, List.getD_cons_succ]; congr 1; omega

lemma pasGo_one (k : ℕ) : pasGo k 1 = (List.range k).map pascal := by
  have h := pasGo_pascal k 0
  rw [pascal_zero] at h
  rw [h]; simp

lemma pasList_getD (n i : ℕ) (hi : i < n + 1) : (pasGo (n + 1) 1).getD i 0 = pascal i := by
  rw [pasGo_one, List.getD_eq_getElem _ _ (by simpa using hi)]
  simp

lemma pasList_rev_getD (n i : ℕ) (hi : i < n + 1) : (revGo (pasGo (n + 1) 1) []).getD i 0 = pascal (n - i) := by
  rw [revGo_eq, List.append_nil, pasGo_one, List.getD_eq_getElem _ _ (by simpa using hi)]
  rw [List.getElem_reverse]
  simp only [List.getElem_map, List.getElem_range, List.length_map, List.length_range]
  congr 1

/-- **The odd-degree columns**, as a list over `j ≤ n`. -/
theorem colsA_eq (n : ℕ) : colsA n = (List.range (n + 1)).map (colOnePlusS n) ++
    (List.range (n + 1)).map fun j => colN3p n j (pascal j) (pascal (n - j)) := by
  unfold colsA
  rw [tabulate_eq, n3Go_eq _ _ _ _ (by simp [revGo_eq, pasGo_one])]
  simp only [zero_add]
  congr 1
  have hl : (pasGo (n + 1) 1).length = n + 1 := by rw [pasGo_one]; simp
  rw [hl]
  refine List.map_congr_left fun i hi => ?_
  have hi' := List.mem_range.mp hi
  rw [pasList_getD n i hi', pasList_rev_getD n i hi']

/-- **The stacked columns**, as a list over `j ≤ n`. -/
theorem colsSig_eq (n : ℕ) : colsSig n = (List.range (n + 1)).map fun j => colSigp n j (pascal j) := by
  unfold colsSig
  rw [sigGo_eq]
  have hl : (pasGo (n + 1) 1).length = n + 1 := by rw [pasGo_one]; simp
  rw [hl]
  refine List.map_congr_left fun i hi => ?_
  rw [zero_add, pasList_getD n i (List.mem_range.mp hi)]

lemma rowsLtK_eq (w : ℕ) (rows : List ℕ) : rowsLtK w rows = rowsLt w rows := by
  induction rows with
  | nil => rfl
  | cons r rs ih =>
    show (Nat.blt r (Nat.pow 2 w) && rowsLtK w rs) = _
    rw [ih]
    simp only [rowsLt, List.all_cons, Nat.pow_eq]
    congr 1
    cases h : r.blt (2 ^ w)
    · have : ¬ r < 2 ^ w := fun hr => by rw [← Nat.blt_eq] at hr; simp_all
      simp [this]
    · simp_all [Nat.blt_eq]

/-! ### Closed forms of `rhoEntry` for the three mod-2 shapes -/

/-- `ρ_n(!![0,1;1,0])`: `u_j ↦ u_{n-j}`. -/
lemma rhoEntry_S (n j t : ℕ) (hj : j ≤ n) : rhoEntry n (0 : ℕ) 1 1 0 t j = if t = n - j then 1 else 0 := by
  unfold rhoEntry
  rw [Finset.sum_eq_single (n - j)]
  · rw [Finset.sum_eq_single 0]
    · simp [eq_comm]
    · intro r _ hr; simp [zero_pow hr]
    · intro h; simp at h
  · intro s hs hsn
    have : n - j - s ≠ 0 := by have := Finset.mem_range.mp hs; omega
    simp [zero_pow this]
  · intro h; simp at h

/-- `ρ_n(!![0,1;1,1])`: `u_j ↦ Σ_r C(j, r) u_{n-j+r}`. -/
lemma rhoEntry_U (n j t : ℕ) (hj : j ≤ n) :
    rhoEntry n (0 : ℕ) 1 1 1 t j = if n - j ≤ t then j.choose (t - (n - j)) else 0 := by
  unfold rhoEntry
  rw [Finset.sum_eq_single (n - j)]
  · simp only [Nat.choose_self, Nat.sub_self, pow_zero, one_pow, mul_one]
    split_ifs with h
    · by_cases hr : t - (n - j) < j + 1
      · rw [Finset.sum_eq_single (t - (n - j))]
        · simp; omega
        · intro r _ hr'; rw [if_neg]; omega
        · intro h'; exact absurd (Finset.mem_range.mpr hr) h'
      · rw [Nat.choose_eq_zero_of_lt (by omega)]
        refine Finset.sum_eq_zero fun r hr' => ?_
        rw [if_neg]; have := Finset.mem_range.mp hr'; omega
    · refine Finset.sum_eq_zero fun r _ => ?_
      rw [if_neg]; omega
  · intro s hs hsn
    have : n - j - s ≠ 0 := by have := Finset.mem_range.mp hs; omega
    simp [zero_pow this]
  · intro h; simp at h

/-- `ρ_n(!![1,1;1,0])`: `u_j ↦ Σ_t C(n-j, t) u_t`. -/
lemma rhoEntry_U2 (n j t : ℕ) : rhoEntry n (1 : ℕ) 1 1 0 t j = (n - j).choose t := by
  unfold rhoEntry
  have hin : ∀ s ∈ Finset.range (n - j + 1), (∑ r ∈ Finset.range (j + 1),
      if s + r = t then (n - j).choose s * 1 ^ (n - j - s) * 1 ^ s * j.choose r * 1 ^ (j - r) * (0 : ℕ) ^ r else 0) =
      if s = t then (n - j).choose s else 0 := by
    intro s _
    rw [Finset.sum_eq_single 0]
    · simp
    · intro r _ hr; simp [zero_pow hr]
    · intro h; simp at h
  simp only [Nat.cast_id]
  rw [Finset.sum_congr rfl hin, Finset.sum_ite_eq']
  split_ifs with h
  · rfl
  · rw [Nat.choose_eq_zero_of_lt (by simp at h; omega)]

/-! ### F1, bit form: the packed columns are `packBits (rhoEntry … mod 2)` -/

/-- Column `j` of `ρ_n(S)` mod 2 (the `S` part of `colOnePlusS`). -/
theorem testBit_colS (n j t : ℕ) (hj : j ≤ n) :
    (Nat.shiftLeft 1 (n - j)).testBit t = decide (rhoEntry n (0 : ℕ) 1 1 0 t j % 2 = 1) := by
  rw [rhoEntry_S n j t hj]
  change (1 <<< (n - j)).testBit t = _
  rw [Nat.one_shiftLeft, Nat.testBit_two_pow]
  by_cases h : t = n - j <;> simp [h, eq_comm]

/-- **F1 for `U`**: `colU = packBits (rhoEntry n 0 1 1 1 · j mod 2)`, bit for bit. -/
theorem testBit_colU (n j t : ℕ) (hj : j ≤ n) :
    (Nat.shiftLeft (pascal j) (n - j)).testBit t = decide (rhoEntry n (0 : ℕ) 1 1 1 t j % 2 = 1) := by
  rw [rhoEntry_U n j t hj]
  change (pascal j <<< (n - j)).testBit t = _
  rw [Nat.testBit_shiftLeft, testBit_pascal]
  by_cases h : n - j ≤ t <;> simp [h]

/-- **F1 for `U²`**: `colU2 = packBits (rhoEntry n 1 1 1 0 · j mod 2)`, bit for bit. -/
theorem testBit_colU2 (n j t : ℕ) :
    (pascal (n - j)).testBit t = decide (rhoEntry n (1 : ℕ) 1 1 0 t j % 2 = 1) := by
  rw [rhoEntry_U2, testBit_pascal]

/-! ### F1, matrix form: `vec` of a packed column is a column of `GL2.rhoMat n` over `ZMod 2` -/

/-- `S ≡ !![0,1;1,0]`, `U ≡ !![0,1;1,1]`, `U² ≡ !![1,1;1,0]` over `ZMod 2`. -/
def mS2 : Mat2 (ZMod 2) := !![0, 1; 1, 0]
def mU2 : Mat2 (ZMod 2) := !![0, 1; 1, 1]
def mUU2 : Mat2 (ZMod 2) := !![1, 1; 1, 0]

lemma natCast_zmod2 (k : ℕ) : ((k : ℕ) : ZMod 2) = if k % 2 = 1 then 1 else 0 := by
  rw [← ZMod.natCast_mod]
  rcases Nat.mod_two_eq_zero_or_one k with h | h <;> simp [h]

lemma vec_of_testBit {w x : ℕ} (f : ℕ → ℕ) (h : ∀ t, x.testBit t = decide (f t % 2 = 1)) (t : Fin w) :
    vec w x t = ((f t : ℕ) : ZMod 2) := by
  rw [vec, h, natCast_zmod2]
  by_cases hf : f t % 2 = 1 <;> simp [hf]

lemma rhoMat_entry (n : ℕ) (A : Mat2 (ZMod 2)) (a b c d : ℕ) (hA : A = !![(a : ZMod 2), b; c, d])
    (t j : Fin (n + 1)) : GL2.rhoMat n A t j = ((rhoEntry n a b c d t j : ℕ) : ZMod 2) := by
  subst hA
  rw [← GL2.Exec.rhoMat_eq, ← rhoNat_eq, cast_rhoNat]
  simp [GL2.Exec.rhoMat]

end GL2.L23.RankF2
