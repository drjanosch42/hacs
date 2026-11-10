/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.L23.SmithPK
import GL2.L23.TreeF3Cols

/-!
# LV (slot 23): the packed columns over `ℤ/N` are the columns of `ρ_n` (the F1 gate of the Smith lane)

The `ℤ/N` analogue of LV-1's F1 (`GL2.L23.RankF2.testBit_colU`, "`colU = packBits(rhoEntry mod 2)`") and LV-2's
(`GL2.L23.RankF3.vec3_colU`), for the generators of `GL2/L23/SmithPKExec.lean`, against [C] `eq:rhondef`
(`GL2.Exec.rhoEntry`, tied to `GL2.rhoMat` by `GL2.Exec.rhoMat_eq`), for **every** `n`, `j ≤ n` and modulus `N`
(fields of width `F` with `16 N³ < 2^F`, so that no intermediate field overflows):

* `fld_qrowK`, `entN_qrowK`: field `t` of `Q_m = (z − 1)^m` is `C(m, t)(−1)^(m+t) mod N` (the kernel's binomial
  recursion `C(m, t+1) = C(m, t)(m − t)/(t + 1)` is `Nat.choose_succ_right_eq`; no Lucas theorem);
* **`vecN_colSK`, `vecN_colUK`, `vecN_colUUK`** (F1 over `ℤ/N`): column `j` of `ρ_n(S)`, `ρ_n(U)`, `ρ_n(U²)` over
  `ZMod N` for `S = !![0,−1;1,0]`, `U = !![0,−1;1,1]`, `U² = !![−1,−1;1,0]`;
* **`entN_colD0x` … `entN_colD2z`**: the packed columns of the cell matrices equal, entry by entry, explicit block
  formulas in `ρ_n` over `ZMod N`; that these are the columns of LR-Y's `treeD0`, `treeDS 0`, `treeDS 1` on
  `VN N n` is `GL2/L23/SmithPKTree.lean` (LV audit M-3) (`n` even: `N_S = 2(1 + S)`, `N_U = 2(1 + U + U²)`, `edgA 0 = 0`,
  `edgA 1 = 2`), entry by entry, in `ρ_n` over `ZMod N`.
-/

noncomputable section

namespace GL2.L23.SmithPK

open GL2.Exec
open GL2.L23.RankF2 (force)

/-! ### Fields of the building blocks -/

section Pack

variable {F : ℕ}

lemma fld_shF (F x k i : ℕ) : fld F (shF F x k) i = if k ≤ i then fld F x (i - k) else 0 := by
  have hs : shF F x k = x * 2 ^ (F * k) := Nat.shiftLeft_eq _ _
  rw [hs, fld_eq, fld_eq]
  split_ifs with h
  · obtain ⟨d, rfl⟩ : ∃ d, i = k + d := ⟨i - k, by omega⟩
    rw [show k + d - k = d by omega, mul_add, pow_add, mul_comm (2 ^ (F * k)) (2 ^ (F * d)),
      Nat.mul_div_mul_right _ _ (Nat.two_pow_pos _)]
  · obtain ⟨d, rfl⟩ : ∃ d, k = i + (d + 1) := ⟨k - i - 1, by omega⟩
    have e : x * 2 ^ (F * (i + (d + 1))) / 2 ^ (F * i) = x * 2 ^ (F * d) * 2 ^ F := by
      rw [show x * 2 ^ (F * (i + (d + 1))) = (x * 2 ^ (F * d) * 2 ^ F) * 2 ^ (F * i) by
        rw [mul_add, mul_add, mul_one, pow_add, pow_add]; ring, Nat.mul_div_cancel _ (Nat.two_pow_pos _)]
    rw [e]; exact Nat.mul_mod_left _ _

lemma fld_one (hF : 0 < F) (i : ℕ) : fld F 1 i = if i = 0 then 1 else 0 := by
  cases i with
  | zero => rw [fld_zero_idx, Nat.mod_eq_of_lt (Nat.one_lt_two_pow (by omega))]; simp
  | succ i => rw [fld_succ, Nat.div_eq_of_lt (Nat.one_lt_two_pow (by omega)), fld_zero]; simp

lemma fld_eK (hF : 0 < F) (k i : ℕ) : fld F (eK F k) i = if i = k then 1 else 0 := by
  rw [show eK F k = shF F 1 k from rfl, fld_shF]
  split_ifs with h1 h2 h2 <;> first | omega | (rw [fld_one hF]; split_ifs <;> omega)

lemma fld_add (F x y i : ℕ) (h : ∀ i' ≤ i, fld F x i' + fld F y i' < 2 ^ F) :
    fld F (x + y) i = fld F x i + fld F y i := by
  have := fld_add_mul F x y 1 i (fun i' hi' => by rw [one_mul]; exact h i' hi')
  rwa [one_mul, one_mul] at this

lemma fld_cmul (F c x i : ℕ) (h : ∀ i' ≤ i, c * fld F x i' < 2 ^ F) : fld F (c * x) i = c * fld F x i := by
  rw [mul_comm c x, fld_mul F x c i h, mul_comm]

/-- The entry `t` of a packed row, in `ZMod N`. -/
def entN (F N x t : ℕ) : ZMod N := (fld F x t : ZMod N)

/-- The signed binomial field: `C(m, t)` or, with the sign `(−1)^(m+t)` on, `−C(m, t)` reduced. -/
def bsg (N m : ℕ) (sg : Bool) (t c : ℕ) : ℕ :=
  @Bool.rec (fun _ => ℕ) (c % N) ((N - c % N) % N) (sg && Nat.beq ((m + t) % 2) 1)

lemma bsg_lt {N : ℕ} (hN : 0 < N) (m : ℕ) (sg : Bool) (t c : ℕ) : bsg N m sg t c < N := by
  unfold bsg; cases (sg && Nat.beq ((m + t) % 2) 1) <;> exact Nat.mod_lt _ hN

lemma binGo_succ (F N m : ℕ) (sg : Bool) (k t c : ℕ) :
    binGo F N m sg (k + 1) t c = bsg N m sg t c + 2 ^ F * binGo F N m sg k (t + 1) (c * (m - t) / (t + 1)) := by
  show force (c % N) (fun cm => force (@Bool.rec (fun _ => ℕ) cm ((N - cm) % N) (sg && Nat.beq ((m + t) % 2) 1))
    fun g => force (c * (m - t) / (t + 1)) fun c' => g + (binGo F N m sg k (t + 1) c' <<< F)) = _
  rw [force_eq, force_eq, force_eq, Nat.shiftLeft_eq, mul_comm (binGo _ _ _ _ _ _ _)]; rfl

lemma choose_next (m t : ℕ) : m.choose t * (m - t) / (t + 1) = m.choose (t + 1) := by
  rw [← Nat.choose_succ_right_eq, Nat.mul_div_cancel _ (Nat.succ_pos t)]

theorem fld_binGo {N : ℕ} (hN : 0 < N) (hNF : N ≤ 2 ^ F) (m : ℕ) (sg : Bool) :
    ∀ k t i, fld F (binGo F N m sg k t (m.choose t)) i =
      if i < k then bsg N m sg (t + i) (m.choose (t + i)) else 0
  | 0, t, i => by rw [show binGo F N m sg 0 t (m.choose t) = 0 from rfl, fld_zero]; simp
  | k + 1, t, i => by
    rw [binGo_succ, choose_next]
    obtain ⟨h1, h2⟩ := split_low (binGo F N m sg k (t + 1) (m.choose (t + 1))) (Nat.two_pow_pos F)
      (lt_of_lt_of_le (bsg_lt hN m sg t _) hNF)
    cases i with
    | zero => rw [fld_zero_idx, h1]; simp
    | succ i =>
      rw [fld_succ, h2, fld_binGo hN hNF m sg k (t + 1) i]
      simp only [Nat.succ_lt_succ_iff, show t + 1 + i = t + (i + 1) by ring]

/-- **Field `t` of `Q_m = (z − 1)^m`** (`qrowK`). -/
theorem fld_qrowK {N : ℕ} (hN : 0 < N) (hNF : N ≤ 2 ^ F) (m t : ℕ) :
    fld F (qrowK F N m) t = if t < m + 1 then bsg N m true t (m.choose t) else 0 := by
  have := fld_binGo hN hNF m true (m + 1) 0 t
  rwa [Nat.choose_zero_right, zero_add] at this

lemma fld_qrowK_lt {N : ℕ} (hN : 0 < N) (hNF : N ≤ 2 ^ F) (m t : ℕ) : fld F (qrowK F N m) t < N := by
  rw [fld_qrowK hN hNF]; split_ifs
  · exact bsg_lt hN _ _ _ _
  · exact hN

lemma cast_bsg_true {N : ℕ} [NeZero N] (m t c : ℕ) : (bsg N m true t c : ZMod N) = (c : ZMod N) * (-1) ^ (m + t) := by
  unfold bsg
  rcases Nat.mod_two_eq_zero_or_one (m + t) with h | h
  · rw [h, show (true && Nat.beq 0 1) = false from rfl]
    show ((c % N : ℕ) : ZMod N) = _
    rw [ZMod.natCast_mod, (Nat.even_iff.mpr h).neg_one_pow, mul_one]
  · rw [h, show (true && Nat.beq 1 1) = true from rfl]
    show (((N - c % N) % N : ℕ) : ZMod N) = _
    rw [ZMod.natCast_mod, Nat.cast_sub (Nat.mod_lt _ (NeZero.pos N)).le, ZMod.natCast_self, ZMod.natCast_mod,
      (Nat.odd_iff.mpr h).neg_one_pow]
    ring

/-- **Entry `t` of `Q_m` over `ZMod N`** is `C(m, t)(−1)^(m+t)`, every `t`. -/
theorem entN_qrowK {N : ℕ} [NeZero N] (hNF : N ≤ 2 ^ F) (m t : ℕ) :
    entN F N (qrowK F N m) t = (m.choose t : ZMod N) * (-1) ^ (m + t) := by
  rw [entN, fld_qrowK (NeZero.pos N) hNF]
  split_ifs with h
  · exact cast_bsg_true m t _
  · rw [Nat.choose_eq_zero_of_lt (by omega)]; simp

end Pack

/-! ### F1 over `ℤ/N`: the columns of `ρ_n(S)`, `ρ_n(U)`, `ρ_n(U²)` -/

section F1

variable {F N : ℕ} [NeZero N]

lemma fld_negK (hNF : N ^ 2 ≤ 2 ^ F) {x : ℕ} (hx : ∀ i, fld F x i < N) (i : ℕ) :
    fld F (negK N x) i = (N - 1) * fld F x i := by
  refine fld_cmul F _ _ _ fun i' _ => lt_of_lt_of_le ?_ hNF
  rw [sq]; exact Nat.mul_lt_mul_of_le_of_lt (Nat.sub_le _ _) (hx i') (NeZero.pos N)

lemma castN_neg (x : ℕ) : (((N - 1) * x : ℕ) : ZMod N) = -(x : ZMod N) := by
  rw [Nat.cast_mul, Nat.cast_sub (NeZero.pos N), ZMod.natCast_self, Nat.cast_one]; ring

lemma fld_sgnK (hNF : N ^ 2 ≤ 2 ^ F) {x : ℕ} (hx : ∀ i, fld F x i < N) (j i : ℕ) :
    (fld F (sgnK N j x) i : ZMod N) = (-1) ^ j * (fld F x i : ZMod N) := by
  unfold sgnK
  rcases Nat.mod_two_eq_zero_or_one j with h | h
  · rw [show Nat.beq (Nat.mod j 2) 1 = false by rw [show Nat.mod j 2 = j % 2 from rfl, h]; rfl,
      (Nat.even_iff.mpr h).neg_one_pow, one_mul]
  · rw [show Nat.beq (Nat.mod j 2) 1 = true by rw [show Nat.mod j 2 = j % 2 from rfl, h]; rfl,
      (Nat.odd_iff.mpr h).neg_one_pow]
    show (fld F (negK N x) i : ZMod N) = _
    rw [fld_negK hNF hx, castN_neg]; ring

/-- `S`, `U`, `U²` over `ZMod N`. -/
def mSN : Mat2 (ZMod N) := !![0, -1; 1, 0]
def mUN : Mat2 (ZMod N) := !![0, -1; 1, 1]
def mUUN : Mat2 (ZMod N) := !![-1, -1; 1, 0]

/-- The first `n + 1` entries as a vector. -/
def vecN (F N n x : ℕ) : Fin (n + 1) → ZMod N := fun t => entN F N x t

open GL2.L23.RankF3 (rhoEntry_a0c1 rhoEntry_c1d0 rhoMat_apply' neg_one_pow_sub_eq_add)

omit [NeZero N] in
lemma fld_eK_lt (hF : 0 < F) (hN1 : 1 < N) (k i : ℕ) : fld F (eK F k) i < N := by
  rw [fld_eK hF]; split_ifs <;> omega

/-- `ρ_n(A)` has no entry below row `n`. -/
lemma rhoEntry_eq_zero_of_lt {R : Type*} [CommRing R] (n : ℕ) (a b c d : R) {t j : ℕ} (hj : j ≤ n) (ht : n < t) :
    rhoEntry n a b c d t j = 0 := by
  unfold rhoEntry
  refine Finset.sum_eq_zero fun s hs => Finset.sum_eq_zero fun r hr => ?_
  rw [if_neg]
  have := Finset.mem_range.mp hs; have := Finset.mem_range.mp hr; omega

/-- **F1 for `S` over `ℤ/N`**, entry `t` (every `t`): column `j` of `ρ_n(S)` is `(−1)^j u_{n−j}`. -/
theorem entN_colSK (hF : 0 < F) (hN1 : 1 < N) (hNF : N ^ 2 ≤ 2 ^ F) {n j : ℕ} (hj : j ≤ n) (t : ℕ) :
    entN F N (colSK F N n j) t = rhoEntry n (0 : ZMod N) (-1) 1 0 t j := by
  rw [entN, colSK, fld_sgnK hNF (fld_eK_lt hF hN1 _), fld_eK hF, rhoEntry_a0c1 n j t (-1) 0 hj]
  simp only [Nat.sub_eq]
  by_cases h : t = n - j
  · rw [if_pos h, if_pos (by omega), h, Nat.sub_self]; simp
  · rw [if_neg h, Nat.cast_zero, mul_zero]
    split_ifs with h'
    · rw [zero_pow (by omega), mul_zero]
    · rfl

/-- **F1 for `U` over `ℤ/N`**, entry `t`: column `j` of `ρ_n(U)` is `z^{n−j} Q_j`. -/
theorem entN_colUK (hNF : N ≤ 2 ^ F) {n j : ℕ} (hj : j ≤ n) (t : ℕ) :
    entN F N (colUK F N n j) t = rhoEntry n (0 : ZMod N) (-1) 1 1 t j := by
  rw [rhoEntry_a0c1 n j t (-1) 1 hj, entN, colUK, fld_shF]
  simp only [Nat.sub_eq]
  split_ifs with h
  · rw [← entN, entN_qrowK hNF, one_pow, mul_one]
    by_cases hr : t - (n - j) ≤ j
    · rw [neg_one_pow_sub_eq_add _ _ hr]
    · rw [Nat.choose_eq_zero_of_lt (by omega)]; simp
  · simp

/-- **F1 for `U²` over `ℤ/N`**, entry `t`: column `j` of `ρ_n(U²)` is `(−1)^j Q_{n−j}`. -/
theorem entN_colUUK (hNF : N ^ 2 ≤ 2 ^ F) (n j t : ℕ) :
    entN F N (colUUK F N n j) t = rhoEntry n (-1 : ZMod N) (-1) 1 0 t j := by
  have hNF' : N ≤ 2 ^ F := le_trans (Nat.le_self_pow (by norm_num) N) hNF
  rw [entN, colUUK, fld_sgnK hNF (fld_qrowK_lt (NeZero.pos N) hNF' _), ← entN, entN_qrowK hNF', rhoEntry_c1d0]
  simp only [Nat.sub_eq]
  by_cases ht : t ≤ n - j
  · rw [neg_one_pow_sub_eq_add _ _ ht]; ring
  · rw [Nat.choose_eq_zero_of_lt (by omega)]; simp

/-- **F1 for `S` over `ℤ/N`** (matrix form). -/
theorem vecN_colSK (hF : 0 < F) (hN1 : 1 < N) (hNF : N ^ 2 ≤ 2 ^ F) (n : ℕ) (j : Fin (n + 1)) :
    vecN F N n (colSK F N n j) = (GL2.rhoMat n (mSN (N := N))).col j := by
  funext t
  rw [vecN, entN_colSK hF hN1 hNF (Nat.lt_succ_iff.mp j.2), Matrix.col_apply, rhoMat_apply']; rfl

/-- **F1 for `U` over `ℤ/N`** (matrix form). -/
theorem vecN_colUK (hNF : N ≤ 2 ^ F) (n : ℕ) (j : Fin (n + 1)) :
    vecN F N n (colUK F N n j) = (GL2.rhoMat n (mUN (N := N))).col j := by
  funext t
  rw [vecN, entN_colUK hNF (Nat.lt_succ_iff.mp j.2), Matrix.col_apply, rhoMat_apply']; rfl

/-- **F1 for `U²` over `ℤ/N`** (matrix form). -/
theorem vecN_colUUK (hNF : N ^ 2 ≤ 2 ^ F) (n : ℕ) (j : Fin (n + 1)) :
    vecN F N n (colUUK F N n j) = (GL2.rhoMat n (mUUN (N := N))).col j := by
  funext t
  rw [vecN, entN_colUUK hNF, Matrix.col_apply, rhoMat_apply']; rfl

end F1

/-! ### The cell matrices: the columns of LR-Y's tree differentials, entry by entry -/

section Blocks

variable {F N : ℕ} [NeZero N]

/-- Every field of `x` is `≤ B`. -/
def FB (F B x : ℕ) : Prop := ∀ i, fld F x i ≤ B

omit [NeZero N] in
lemma fld_add_FB {B1 B2 x y : ℕ} (hx : FB F B1 x) (hy : FB F B2 y) (h : B1 + B2 < 2 ^ F) (i : ℕ) :
    fld F (x + y) i = fld F x i + fld F y i :=
  fld_add F x y i fun i' _ => lt_of_le_of_lt (add_le_add (hx i') (hy i')) h

omit [NeZero N] in
lemma FB_add {B1 B2 x y : ℕ} (hx : FB F B1 x) (hy : FB F B2 y) (h : B1 + B2 < 2 ^ F) : FB F (B1 + B2) (x + y) :=
  fun i => by rw [fld_add_FB hx hy h]; exact add_le_add (hx i) (hy i)

omit [NeZero N] in
lemma fld_cmul_FB {B c x : ℕ} (hx : FB F B x) (h : c * B < 2 ^ F) (i : ℕ) : fld F (c * x) i = c * fld F x i :=
  fld_cmul F c x i fun i' _ => lt_of_le_of_lt (Nat.mul_le_mul_left c (hx i')) h

omit [NeZero N] in
lemma FB_cmul {B c x : ℕ} (hx : FB F B x) (h : c * B < 2 ^ F) : FB F (c * B) (c * x) :=
  fun i => by rw [fld_cmul_FB hx h]; exact Nat.mul_le_mul_left c (hx i)

omit [NeZero N] in
lemma FB_mono {B B' x : ℕ} (hx : FB F B x) (h : B ≤ B') : FB F B' x := fun i => le_trans (hx i) h

omit [NeZero N] in
lemma FB_shF {B x : ℕ} (hx : FB F B x) (k : ℕ) : FB F B (shF F x k) := fun i => by
  rw [fld_shF]; split_ifs
  · exact hx _
  · exact Nat.zero_le _

omit [NeZero N] in
lemma FB_eK (hF : 0 < F) (k : ℕ) : FB F 1 (eK F k) := fun i => by rw [fld_eK hF]; split_ifs <;> omega

lemma FB_qrowK (hNF : N ≤ 2 ^ F) (m : ℕ) : FB F N (qrowK F N m) := fun i => (fld_qrowK_lt (NeZero.pos N) hNF m i).le

omit [NeZero N] in
lemma FB_negK {B x : ℕ} (hx : FB F B x) (h : N * B < 2 ^ F) : FB F (N * B) (negK N x) := by
  have h' : (N - 1) * B < 2 ^ F := lt_of_le_of_lt (Nat.mul_le_mul_right _ (Nat.sub_le _ _)) h
  exact FB_mono (FB_cmul hx h') (Nat.mul_le_mul_right _ (Nat.sub_le _ _))

lemma FB_sgnK {B x : ℕ} (hx : FB F B x) (h : N * B < 2 ^ F) (j : ℕ) : FB F (N * B) (sgnK N j x) := by
  unfold sgnK
  cases Nat.beq (Nat.mod j 2) 1
  · exact FB_mono hx (Nat.le_mul_of_pos_left _ (NeZero.pos N))
  · exact FB_negK hx h

omit [NeZero N] in
lemma entN_add_FB {B1 B2 x y : ℕ} (hx : FB F B1 x) (hy : FB F B2 y) (h : B1 + B2 < 2 ^ F) (t : ℕ) :
    entN F N (x + y) t = entN F N x t + entN F N y t := by
  rw [entN, fld_add_FB hx hy h, Nat.cast_add]; rfl

omit [NeZero N] in
lemma entN_cmul_FB {B c x : ℕ} (hx : FB F B x) (h : c * B < 2 ^ F) (t : ℕ) :
    entN F N (c * x) t = c * entN F N x t := by
  rw [entN, fld_cmul_FB hx h, Nat.cast_mul]; rfl

lemma entN_negK {B x : ℕ} (hx : FB F B x) (h : N * B < 2 ^ F) (t : ℕ) : entN F N (negK N x) t = -entN F N x t := by
  have h' : (N - 1) * B < 2 ^ F := lt_of_le_of_lt (Nat.mul_le_mul_right _ (Nat.sub_le _ _)) h
  rw [entN, show negK N x = (N - 1) * x from rfl, fld_cmul_FB hx h', castN_neg]; rfl

omit [NeZero N] in
lemma entN_shF (x k t : ℕ) : entN F N (shF F x k) t = if k ≤ t then entN F N x (t - k) else 0 := by
  rw [entN, fld_shF]; split_ifs
  · rfl
  · simp

omit [NeZero N] in
lemma entN_eK (hF : 0 < F) (k t : ℕ) : entN F N (eK F k) t = if t = k then 1 else 0 := by
  rw [entN, fld_eK hF]; split_ifs <;> simp

lemma entN_redRow (hNF : N ≤ 2 ^ F) (w x t : ℕ) : entN F N (redRow F N w x) t = if t < w then entN F N x t else 0 := by
  rw [entN, fld_redRow (NeZero.pos N) hNF]; split_ifs
  · rw [ZMod.natCast_mod]; rfl
  · simp

lemma cube (N : ℕ) : N ^ 3 = N * N * N := by ring

/-- The size hypothesis of the block lemmas: `16 N³ < 2^F` (no intermediate field overflows). -/
structure Fits (F N : ℕ) : Prop where
  F_pos : 0 < F
  N_gt : 1 < N
  big : 16 * N ^ 3 < 2 ^ F

variable (hx : Fits F N)
include hx

omit [NeZero N] in
lemma Fits.sq_le : N ^ 2 ≤ 2 ^ F := by
  have := hx.big; have : N ^ 2 ≤ 16 * N ^ 3 := by nlinarith [hx.N_gt, cube N]
  omega

omit [NeZero N] in
lemma Fits.le : N ≤ 2 ^ F := le_trans (Nat.le_self_pow (by norm_num) N) hx.sq_le

omit [NeZero N] in
lemma Fits.lt (B : ℕ) (hB : B ≤ 16 * N ^ 3) : B < 2 ^ F := lt_of_le_of_lt hB hx.big

lemma Fits.FB_colSK (n j : ℕ) : FB F N (colSK F N n j) := by
  have := FB_sgnK (FB_eK hx.F_pos (n - j)) (hx.lt (N * 1) (by nlinarith [hx.N_gt, cube N])) j
  rwa [mul_one] at this

lemma Fits.FB_colUK (n j : ℕ) : FB F N (colUK F N n j) := FB_shF (FB_qrowK hx.le j) _

lemma Fits.FB_colUUK (n j : ℕ) : FB F (N * N) (colUUK F N n j) :=
  FB_sgnK (FB_qrowK hx.le _) (hx.lt (N * N) (by nlinarith [hx.N_gt, cube N])) j

lemma Fits.FB_colN3K (n j : ℕ) : FB F (1 + N + N * N) (colN3K F N n j) :=
  FB_add (FB_add (FB_eK hx.F_pos j) (hx.FB_colUK n j) (hx.lt (1 + N) (by nlinarith [hx.N_gt, cube N])))
    (hx.FB_colUUK n j) (hx.lt (1 + N + N * N) (by nlinarith [hx.N_gt, cube N]))

/-- Kronecker delta and the shift of a column. -/
def dl (k t : ℕ) : ZMod N := if t = k then 1 else 0
def shv (f : ℕ → ZMod N) (k t : ℕ) : ZMod N := if k ≤ t then f (t - k) else 0

omit [NeZero N] in
lemma Fits.entN_eK_dl (k t : ℕ) : entN F N (eK F k) t = dl k t := entN_eK hx.F_pos k t

/-- **The columns `x = u_j` of `d⁰`** (block formula; `treeD0` by hand): `((1 − S)u_j, 0, u_j)`. -/
theorem entN_colD0x {n j : ℕ} (hj : j ≤ n) (t : ℕ) :
    entN F N (colD0x F N n j) t = if t < 3 * (n + 1) then
      (dl j t - rhoEntry n (0 : ZMod N) (-1) 1 0 t j) + shv (dl j) (2 * (n + 1)) t else 0 := by
  have h1 := FB_negK (hx.FB_colSK n j) (hx.lt (N * N) (by nlinarith [hx.N_gt, cube N]))
  have h2 := FB_add (FB_eK hx.F_pos j) h1 (hx.lt (1 + N * N) (by nlinarith [hx.N_gt, cube N]))
  show entN F N (redRow F N (3 * (n + 1)) ((eK F j + negK N (colSK F N n j)) + shF F (eK F j) (2 * (n + 1)))) t = _
  rw [entN_redRow hx.le]
  split_ifs with ht
  · rw [entN_add_FB h2 (FB_shF (FB_eK hx.F_pos j) _) (hx.lt (1 + N * N + 1) (by nlinarith [hx.N_gt, cube N])),
      entN_add_FB (FB_eK hx.F_pos j) h1 (hx.lt (1 + N * N) (by nlinarith [hx.N_gt, cube N])),
      entN_negK (hx.FB_colSK n j) (hx.lt (N * N) (by nlinarith [hx.N_gt, cube N])),
          entN_colSK hx.F_pos hx.N_gt hx.sq_le hj,
      hx.entN_eK_dl, entN_shF]
    simp only [dl, shv, hx.entN_eK_dl]
    ring
  · rfl

/-- `(1 + U + U²) u_j` entry by entry. -/
theorem entN_colN3K {n j : ℕ} (hj : j ≤ n) (t : ℕ) :
    entN F N (colN3K F N n j) t =
      dl j t + rhoEntry n (0 : ZMod N) (-1) 1 1 t j + rhoEntry n (-1 : ZMod N) (-1) 1 0 t j := by
  show entN F N (eK F j + colUK F N n j + colUUK F N n j) t = _
  rw [entN_add_FB (FB_add (FB_eK hx.F_pos j) (hx.FB_colUK n j) (hx.lt (1 + N) (by nlinarith [hx.N_gt, cube N])))
      (hx.FB_colUUK n j) (hx.lt (1 + N + N * N) (by nlinarith [hx.N_gt, cube N])),
    entN_add_FB (FB_eK hx.F_pos j) (hx.FB_colUK n j) (hx.lt (1 + N) (by nlinarith [hx.N_gt, cube N])), hx.entN_eK_dl,
    entN_colUK hx.le hj, entN_colUUK hx.sq_le]

/-- **The columns `y = u_j` of `d⁰`** (block formula; `treeD0` by hand): `(0, (1 − U)u_j, −u_j)`. -/
theorem entN_colD0y {n j : ℕ} (hj : j ≤ n) (t : ℕ) :
    entN F N (colD0y F N n j) t = if t < 3 * (n + 1) then
      shv (fun t' => dl j t' - rhoEntry n (0 : ZMod N) (-1) 1 1 t' j) (n + 1) t -
        shv (dl j) (2 * (n + 1)) t else 0 := by
  have h1 := FB_negK (hx.FB_colUK n j) (hx.lt (N * N) (by nlinarith [hx.N_gt, cube N]))
  have h2 := FB_add (FB_eK hx.F_pos j) h1 (hx.lt (1 + N * N) (by nlinarith [hx.N_gt, cube N]))
  have h3 := FB_negK (FB_eK hx.F_pos j) (hx.lt (N * 1) (by nlinarith [hx.N_gt, cube N]))
  show entN F N (redRow F N (3 * (n + 1)) (shF F (eK F j + negK N (colUK F N n j)) (n + 1) +
    shF F (negK N (eK F j)) (2 * (n + 1)))) t = _
  rw [entN_redRow hx.le]
  split_ifs with ht
  · rw [entN_add_FB (FB_shF h2 _) (FB_shF h3 _) (hx.lt (1 + N * N + N * 1) (by nlinarith [hx.N_gt, cube N])),
      entN_shF, entN_shF]
    simp only [shv]
    split_ifs with ha hb hb
    · rw [entN_add_FB (FB_eK hx.F_pos j) h1 (hx.lt (1 + N * N) (by nlinarith [hx.N_gt, cube N])),
        entN_negK (hx.FB_colUK n j)
        (hx.lt (N * N) (by nlinarith [hx.N_gt, cube N])),
            entN_negK (FB_eK hx.F_pos j) (hx.lt (N * 1) (by nlinarith [hx.N_gt, cube N])), hx.entN_eK_dl,
        hx.entN_eK_dl, entN_colUK hx.le hj]
      simp only [dl]; ring
    · rw [entN_add_FB (FB_eK hx.F_pos j) h1 (hx.lt (1 + N * N) (by nlinarith [hx.N_gt, cube N])),
        entN_negK (hx.FB_colUK n j)
        (hx.lt (N * N) (by nlinarith [hx.N_gt, cube N])), hx.entN_eK_dl, entN_colUK hx.le hj]
      simp only [dl]; ring
    · omega
    · ring
  · rfl

/-- **The columns `x = u_j` of `d¹ = treeDS 0`** (`n` even): `(2(1 + S)u_j, 0, (1 + S)u_j)`. -/
theorem entN_colD1x {n j : ℕ} (hj : j ≤ n) (t : ℕ) :
    entN F N (colD1x F N n j) t = if t < 3 * (n + 1) then
      2 * (dl j t + rhoEntry n (0 : ZMod N) (-1) 1 0 t j) +
        shv (fun t' => dl j t' + rhoEntry n (0 : ZMod N) (-1) 1 0 t' j) (2 * (n + 1)) t else 0 := by
  have h1 := FB_add (FB_eK hx.F_pos j) (hx.FB_colSK n j) (hx.lt (1 + N) (by nlinarith [hx.N_gt, cube N]))
  have h2 := FB_cmul (c := 2) h1 (hx.lt (2 * (1 + N)) (by nlinarith [hx.N_gt, cube N]))
  show entN F N (redRow F N (3 * (n + 1)) (2 * (eK F j + colSK F N n j) +
    shF F (eK F j + colSK F N n j) (2 * (n + 1)))) t = _
  rw [entN_redRow hx.le]
  split_ifs with ht
  · rw [entN_add_FB h2 (FB_shF h1 _) (hx.lt (2 * (1 + N) + (1 + N)) (by nlinarith [hx.N_gt, cube N])), entN_shF,
      entN_cmul_FB h1 (hx.lt (2 * (1 + N)) (by nlinarith [hx.N_gt, cube N])),
          entN_add_FB (FB_eK hx.F_pos j) (hx.FB_colSK n j)
        (hx.lt (1 + N) (by nlinarith [hx.N_gt, cube N])), hx.entN_eK_dl, entN_colSK hx.F_pos hx.N_gt hx.sq_le hj]
    simp only [shv]
    split_ifs with ha
    · rw [entN_add_FB (FB_eK hx.F_pos j) (hx.FB_colSK n j) (hx.lt (1 + N) (by nlinarith [hx.N_gt, cube N])),
        hx.entN_eK_dl,
        entN_colSK hx.F_pos hx.N_gt hx.sq_le hj]
      simp only [dl]; push_cast; ring
    · simp only [dl]; push_cast; ring
  · rfl

/-- **The columns `y = u_j` of `d¹ = treeDS 0`** (`n` even): `(0, 2N₃u_j, −N₃u_j)`, `N₃ = 1 + U + U²`. -/
theorem entN_colD1y (n j t : ℕ) :
    entN F N (colD1y F N n j) t = if t < 3 * (n + 1) then
      shv (fun t' => 2 * entN F N (colN3K F N n j) t') (n + 1) t -
        shv (fun t' => entN F N (colN3K F N n j) t') (2 * (n + 1)) t else 0 := by
  have hB := hx.FB_colN3K n j
  have h1 := FB_cmul (c := 2) hB (hx.lt (2 * (1 + N + N * N)) (by nlinarith [hx.N_gt, cube N]))
  have h2 := FB_negK hB (hx.lt (N * (1 + N + N * N)) (by nlinarith [hx.N_gt, cube N]))
  show entN F N (redRow F N (3 * (n + 1)) (shF F (2 * colN3K F N n j) (n + 1) +
    shF F (negK N (colN3K F N n j)) (2 * (n + 1)))) t = _
  rw [entN_redRow hx.le]
  split_ifs with ht
  · rw [entN_add_FB (FB_shF h1 _) (FB_shF h2 _)
      (hx.lt (2 * (1 + N + N * N) + N * (1 + N + N * N)) (by nlinarith [hx.N_gt, cube N])),
      entN_shF, entN_shF]
    simp only [shv]
    split_ifs with ha hb hb
    · rw [entN_cmul_FB hB (hx.lt (2 * (1 + N + N * N)) (by nlinarith [hx.N_gt, cube N])),
        entN_negK hB (hx.lt (N * (1 + N + N * N)) (by nlinarith [hx.N_gt, cube N]))]
      push_cast; ring
    · rw [entN_cmul_FB hB (hx.lt (2 * (1 + N + N * N)) (by nlinarith [hx.N_gt, cube N]))]; push_cast; ring
    · omega
    · ring
  · rfl

/-- **The columns `z = u_j` of `d² = treeDS 1`** (`n` even): `(0, 0, −2u_j)` (the `x`, `y` columns are `d⁰`'s). -/
theorem entN_colD2z (n j t : ℕ) :
    entN F N (colD2z F N n j) t = if t < 3 * (n + 1) then -(2 * shv (dl j) (2 * (n + 1)) t) else 0 := by
  have h1 := FB_cmul (c := 2) (FB_eK hx.F_pos j) (hx.lt (2 * 1) (by nlinarith [hx.N_gt, cube N]))
  have h2 := FB_negK h1 (hx.lt (N * (2 * 1)) (by nlinarith [hx.N_gt, cube N]))
  show entN F N (redRow F N (3 * (n + 1)) (shF F (negK N (2 * eK F j)) (2 * (n + 1)))) t = _
  rw [entN_redRow hx.le]
  split_ifs with ht
  · rw [entN_shF]
    simp only [shv]
    split_ifs with ha
    · rw [entN_negK h1 (hx.lt (N * (2 * 1)) (by nlinarith [hx.N_gt, cube N])),
        entN_cmul_FB (FB_eK hx.F_pos j) (hx.lt (2 * 1) (by nlinarith [hx.N_gt, cube N])),
        hx.entN_eK_dl]
      simp only [dl]; push_cast; ring
    · ring
  · rfl

end Blocks

/-! ### The cells: reduced rows, lengths, and the cell theorem -/

section Cells

lemma tab_go (f : ℕ → ℕ) : ∀ k j, @Nat.rec (fun _ => ℕ → List ℕ) (fun _ => [])
    (fun _ ih j => force (f j) fun c => c :: ih (Nat.succ j)) k j = (List.range k).map fun i => f (j + i)
  | 0, j => rfl
  | k + 1, j => by
    show force (f j) (fun c => c :: _) = _
    rw [force_eq, tab_go f k (j + 1), List.range_succ_eq_map, List.map_cons, List.map_map]
    congr 1
    refine List.map_congr_left fun i _ => ?_
    simp only [Function.comp_apply]; congr 1; omega

lemma tab_eq (f : ℕ → ℕ) (k : ℕ) : tab f k = (List.range k).map f := by
  rw [tab, tab_go]; simp

lemma length_colsD0 (F N n : ℕ) : (colsD0 F N n).length = 2 * (n + 1) := by
  rw [colsD0, List.length_append, tab_eq, tab_eq]; simp; ring
lemma length_colsD1 (F N n : ℕ) : (colsD1 F N n).length = 2 * (n + 1) := by
  rw [colsD1, List.length_append, tab_eq, tab_eq]; simp; ring
lemma length_colsD2 (F N n : ℕ) : (colsD2 F N n).length = 3 * (n + 1) := by
  rw [colsD2, List.length_append, length_colsD0, tab_eq]; simp; ring

lemma fld_redRow_lt {F N : ℕ} (hN : 0 < N) (hNF : N ≤ 2 ^ F) (w x i : ℕ) : fld F (redRow F N w x) i < N := by
  rw [fld_redRow hN hNF]; split_ifs
  · exact Nat.mod_lt _ hN
  · exact hN

/-- Every packed column of the cell matrices has reduced fields. -/
theorem colsD_reduced {F N : ℕ} (hN : 0 < N) (hNF : N ≤ 2 ^ F) (n q : ℕ) :
    ∀ r ∈ (if q = 0 then colsD0 F N n else if q = 1 then colsD1 F N n else colsD2 F N n), ∀ i, fld F r i < N := by
  intro r hr i
  have hD0 : ∀ r ∈ colsD0 F N n, fld F r i < N := fun r hr => by
    rw [colsD0, List.mem_append, tab_eq, tab_eq] at hr
    rcases hr with h | h <;> obtain ⟨j, _, rfl⟩ := List.mem_map.mp h <;> exact fld_redRow_lt hN hNF _ _ _
  split_ifs at hr with h0 h1
  · exact hD0 r hr
  · rw [colsD1, List.mem_append, tab_eq, tab_eq] at hr
    rcases hr with h | h <;> obtain ⟨j, _, rfl⟩ := List.mem_map.mp h <;> exact fld_redRow_lt hN hNF _ _ _
  · rw [colsD2, List.mem_append, tab_eq] at hr
    rcases hr with h | h
    · exact hD0 r h
    · obtain ⟨j, _, rfl⟩ := List.mem_map.mp h; exact fld_redRow_lt hN hNF _ _ _

/-- The cell matrices of `cellPiv`. -/
lemma cellPiv_eq (l K H n q : ℕ) (hq : q ≤ 2) : cellPiv l K H n q = smithPK l K H (3 * (n + 1))
    (if q = 0 then colsD0 (2 * H) (l ^ K) n else if q = 1 then colsD1 (2 * H) (l ^ K) n
      else colsD2 (2 * H) (l ^ K) n) := by
  rcases (by omega : q = 0 ∨ q = 1 ∨ q = 2) with rfl | rfl | rfl <;> rfl

/-- **The cell theorem** (`n` even, `q ∈ {0, 1, 2}`): a kernel-decided `sig K (cellPiv l K H n q) = (s, cs)` gives
`#(span over ZMod ℓ^K of the cell columns of d^q) = ℓ^s` (the block formulas of `entN_colD0x` …; the counts `cs` are not
certified here; they are in `GL2/L23/SmithPKCellType.lean`, LV audit M-1). -/
theorem cell_card {l K H n q s : ℕ} {cs : List ℕ} (hq : q ≤ 2)
    (cx : Ctx l K H (if q = 0 then colsD0 (2 * H) (l ^ K) n else if q = 1 then colsD1 (2 * H) (l ^ K) n
      else colsD2 (2 * H) (l ^ K) n).length)
    (h : sig K (cellPiv l K H n q) = (s, cs)) :
    Nat.card (spanK l K H (3 * (n + 1)) (if q = 0 then colsD0 (2 * H) (l ^ K) n
      else if q = 1 then colsD1 (2 * H) (l ^ K) n else colsD2 (2 * H) (l ^ K) n)) = l ^ s := by
  rw [cellPiv_eq _ _ _ _ _ hq] at h
  exact card_of_sig cx (fun r hr i _ => colsD_reduced cx.N_pos (le_trans cx.N_lt.le cx.H_le_F) n q r hr i) h

/-- `cell_card` for `d⁰` (`2(n+1)` columns). -/
theorem cell_card0 {l K H n s : ℕ} {cs : List ℕ} (cx : Ctx l K H (2 * (n + 1)))
    (h : sig K (cellPiv l K H n 0) = (s, cs)) :
    Nat.card (spanK l K H (3 * (n + 1)) (colsD0 (2 * H) (l ^ K) n)) = l ^ s :=
  cell_card (q := 0) (by norm_num) (by rw [if_pos rfl, length_colsD0]; exact cx) h

/-- `cell_card` for `d¹` (`2(n+1)` non-zero columns). -/
theorem cell_card1 {l K H n s : ℕ} {cs : List ℕ} (cx : Ctx l K H (2 * (n + 1)))
    (h : sig K (cellPiv l K H n 1) = (s, cs)) :
    Nat.card (spanK l K H (3 * (n + 1)) (colsD1 (2 * H) (l ^ K) n)) = l ^ s :=
  cell_card (q := 1) (by norm_num) (by rw [if_neg (by norm_num), if_pos rfl, length_colsD1]; exact cx) h

/-- `cell_card` for `d²` (`3(n+1)` columns). -/
theorem cell_card2 {l K H n s : ℕ} {cs : List ℕ} (cx : Ctx l K H (3 * (n + 1)))
    (h : sig K (cellPiv l K H n 2) = (s, cs)) :
    Nat.card (spanK l K H (3 * (n + 1)) (colsD2 (2 * H) (l ^ K) n)) = l ^ s :=
  cell_card (q := 2) (by norm_num) (by rw [if_neg (by norm_num), if_neg (by norm_num), length_colsD2]; exact cx) h

end Cells

end GL2.L23.SmithPK

end
