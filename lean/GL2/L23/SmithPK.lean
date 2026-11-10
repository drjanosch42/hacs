/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.L23.SmithPKExec
import Mathlib.FieldTheory.Finite.Basic
import Mathlib.Data.Nat.Totient
import Mathlib.LinearAlgebra.Span.Basic
import Mathlib.LinearAlgebra.Quotient.Card
import Mathlib.SetTheory.Cardinal.Finite

/-!
# LV (slot 23): `smithPK` is sound — elementary divisors over `ℤ/ℓ^K` of packed rows

The soundness of `GL2/L23/SmithPKExec.lean` (read its docstring for the algorithm).

* **Bits** (`fld_add_mul`, `fld_land`, `fld_rep`, `fld_redRow`, `firstNZ_spec`): a row operation on packed rows is the
  row operation on the fields as long as no field overflows; masks, reduction and the column search read fields.
* **`powMod_eq`, `mul_powMod_totient`**: binary exponentiation, and `u · u^{φ(M) − 1} ≡ 1 (mod M)` (Euler).
* **`tst_spec`**: the four-operation row test (`ℓ = 2`: low bits; `ℓ` odd: the exact-division test) reads, field by field,
  "`ℓ^{v+1}` divides the entry".
* **`level_spec`, `smithPK_spec`**: the invariants of the elimination (bounds, divisibility, cleared pivot columns, the
  row span over `ZMod (ℓ^K)`), hence **`smithPK_span`**: the span of the rows equals the span of the pivot rows, and the
  pivots form an echelon family (`Ech`).
* **`card_span_ech`, `smithPK_card`**: an echelon family with valuations `v_k` spans `ℓ^{Σ (K − v_k)}` elements. Only this
  order is certified in this file; the type (`#(ℓ^t · span)` for every `t`, so the multiset of the `v_k` is that of the
  elementary divisors) is `smithPK_card_smul` in `GL2/L23/SmithPKType.lean` (LV audit M-1, follow-up R1(a)).
-/

noncomputable section

namespace GL2.L23.SmithPK

open GL2.L23.RankF2 (force)
open Nat (totient)

lemma force_eq {α : Sort _} (a : ℕ) (k : ℕ → α) : force a k = k a := by
  unfold force; cases Nat.beq a 0 <;> rfl

/-! ### Fields -/

section Bits

lemma fld_eq (F x i : ℕ) : fld F x i = x / 2 ^ (F * i) % 2 ^ F := by
  show (x >>> (F * i)) &&& (2 ^ F - 1) = _
  rw [Nat.and_two_pow_sub_one_eq_mod, Nat.shiftRight_eq_div_pow]

lemma fld_eq' (F x i : ℕ) : fld F x i = (x >>> (F * i)) % 2 ^ F := by
  show (x >>> (F * i)) &&& (2 ^ F - 1) = _
  rw [Nat.and_two_pow_sub_one_eq_mod]

lemma fld_lt (F x i : ℕ) : fld F x i < 2 ^ F := by
  rw [fld_eq]; exact Nat.mod_lt _ (by positivity)

lemma fld_zero_idx (F x : ℕ) : fld F x 0 = x % 2 ^ F := by
  rw [fld_eq]; simp

lemma fld_succ (F x i : ℕ) : fld F x (i + 1) = fld F (x / 2 ^ F) i := by
  rw [fld_eq, fld_eq, Nat.div_div_eq_div_mul, ← pow_add]
  congr 3; ring

lemma fld_zero (F i : ℕ) : fld F 0 i = 0 := by
  rw [fld_eq]; simp

lemma decomp (P x y c : ℕ) : x + c * y = (x % P + c * (y % P)) + P * (x / P + c * (y / P)) := by
  have hx := Nat.mod_add_div x P
  have hy := Nat.mod_add_div y P
  calc x + c * y = (x % P + P * (x / P)) + c * (y % P + P * (y / P)) := by rw [hx, hy]
    _ = _ := by ring

lemma split_low {P a : ℕ} (z : ℕ) (hP : 0 < P) (ha : a < P) : (a + P * z) % P = a ∧ (a + P * z) / P = z := by
  constructor
  · rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt ha]
  · rw [Nat.add_mul_div_left _ _ hP, Nat.div_eq_of_lt ha, zero_add]

/-- **A row operation on packed rows is the row operation on the fields**, as long as no field `≤ i` overflows. -/
theorem fld_add_mul (F x y c i : ℕ) (h : ∀ i' ≤ i, fld F x i' + c * fld F y i' < 2 ^ F) :
    fld F (x + c * y) i = fld F x i + c * fld F y i := by
  induction i generalizing x y with
  | zero =>
    have h0 := h 0 le_rfl
    simp only [fld_zero_idx] at h0 ⊢
    rw [decomp (2 ^ F) x y c, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt h0]
  | succ i ih =>
    have h0 := h 0 (Nat.zero_le _)
    simp only [fld_zero_idx] at h0
    have hd : (x + c * y) / 2 ^ F = x / 2 ^ F + c * (y / 2 ^ F) := by
      rw [decomp (2 ^ F) x y c, Nat.add_mul_div_left _ _ (by positivity), Nat.div_eq_of_lt h0, zero_add]
    rw [fld_succ, fld_succ, fld_succ, hd]
    exact ih _ _ fun i' hi' => by
      have := h (i' + 1) (by omega)
      rwa [fld_succ, fld_succ] at this

lemma fld_mul (F x c i : ℕ) (h : ∀ i' ≤ i, c * fld F x i' < 2 ^ F) : fld F (x * c) i = fld F x i * c := by
  have := fld_add_mul F 0 x c i (fun i' hi' => by rw [fld_zero, zero_add]; exact h i' hi')
  rw [zero_add, fld_zero, zero_add] at this
  rw [mul_comm x c, this, mul_comm]

lemma testBit_fld (F x i k : ℕ) : (fld F x i).testBit k = (decide (k < F) && x.testBit (F * i + k)) := by
  rw [fld_eq', Nat.testBit_mod_two_pow, Nat.testBit_shiftRight]

lemma testBit_eq_fld {F : ℕ} (hF : 0 < F) (x t : ℕ) : x.testBit t = (fld F x (t / F)).testBit (t % F) := by
  rw [testBit_fld, decide_eq_true (Nat.mod_lt _ hF), Bool.true_and, Nat.div_add_mod]

lemma eq_zero_of_fld {F : ℕ} (hF : 0 < F) {x : ℕ} (h : ∀ i, fld F x i = 0) : x = 0 :=
  Nat.eq_of_testBit_eq fun t => by rw [testBit_eq_fld hF, h, Nat.zero_testBit, Nat.zero_testBit]

lemma fld_land (F x y i : ℕ) : fld F (x &&& y) i = fld F x i &&& fld F y i := by
  apply Nat.eq_of_testBit_eq; intro k
  simp only [testBit_fld, Nat.testBit_and]
  cases decide (k < F) <;> simp

lemma rep_zero (F a : ℕ) : rep F a 0 = 0 := rfl

lemma rep_succ (F a w : ℕ) : rep F a (w + 1) = a + 2 ^ F * rep F a w := by
  show a + (rep F a w <<< F) = _
  rw [Nat.shiftLeft_eq, mul_comm]

theorem fld_rep (F a w i : ℕ) (ha : a < 2 ^ F) : fld F (rep F a w) i = if i < w then a else 0 := by
  induction w generalizing i with
  | zero => rw [rep_zero, fld_zero]; simp
  | succ w ih =>
    rw [rep_succ]
    obtain ⟨h1, h2⟩ := split_low (rep F a w) (Nat.two_pow_pos F) ha
    cases i with
    | zero => rw [fld_zero_idx, h1]; simp
    | succ i => rw [fld_succ, h2, ih]; simp

lemma redGo_zero (F N r i : ℕ) : redGo F N r 0 i = 0 := rfl

lemma redGo_succ (F N r k i : ℕ) :
    redGo F N r (k + 1) i = fld F r i % N + 2 ^ F * redGo F N r k (i + 1) := by
  show force (fld F r i % N) (fun g => g + (redGo F N r k (i + 1) <<< F)) = _
  rw [force_eq, Nat.shiftLeft_eq, mul_comm]

theorem fld_redGo {F N : ℕ} (hN : 0 < N) (hNF : N ≤ 2 ^ F) (r k i i' : ℕ) :
    fld F (redGo F N r k i) i' = if i' < k then fld F r (i + i') % N else 0 := by
  induction k generalizing i i' with
  | zero => rw [redGo_zero, fld_zero]; simp
  | succ k ih =>
    rw [redGo_succ]
    have hlt : fld F r i % N < 2 ^ F := lt_of_lt_of_le (Nat.mod_lt _ hN) hNF
    obtain ⟨h1, h2⟩ := split_low (redGo F N r k (i + 1)) (Nat.two_pow_pos F) hlt
    cases i' with
    | zero => rw [fld_zero_idx, h1]; simp
    | succ i' =>
      rw [fld_succ, h2, ih]
      simp only [Nat.succ_lt_succ_iff]
      rw [show i + 1 + i' = i + (i' + 1) by ring]

theorem fld_redRow {F N : ℕ} (hN : 0 < N) (hNF : N ≤ 2 ^ F) (w r i : ℕ) :
    fld F (redRow F N w r) i = if i < w then fld F r i % N else 0 := by
  rw [redRow, fld_redGo hN hNF, zero_add]

lemma firstNZ_succ (F t k i : ℕ) :
    firstNZ F t (k + 1) i = if fld F t i = 0 then firstNZ F t k (i + 1) else i := by
  show @Bool.rec (fun _ => ℕ) i (firstNZ F t k (i + 1)) (Nat.beq (fld F t i) 0) = _
  by_cases h : fld F t i = 0
  · rw [if_pos h, h]; rfl
  · rw [if_neg h, show Nat.beq (fld F t i) 0 = false from Bool.eq_false_iff.mpr fun h' => h (Nat.eq_of_beq_eq_true h')]

/-- `firstNZ` finds a non-zero field when there is one. -/
theorem firstNZ_spec (F t : ℕ) : ∀ k i, (∃ i', i ≤ i' ∧ i' < i + k ∧ fld F t i' ≠ 0) →
    firstNZ F t k i < i + k ∧ fld F t (firstNZ F t k i) ≠ 0
  | 0, i, ⟨i', h1, h2, _⟩ => absurd h2 (by omega)
  | k + 1, i, ⟨i', h1, h2, h3⟩ => by
    rw [firstNZ_succ]
    by_cases h : fld F t i = 0
    · rw [if_pos h]
      have hi' : i ≠ i' := fun e => h3 (e ▸ h)
      obtain ⟨a, b⟩ := firstNZ_spec F t k (i + 1) ⟨i', by omega, by omega, h3⟩
      exact ⟨by omega, b⟩
    · rw [if_neg h]; exact ⟨by omega, h⟩

end Bits

/-! ### Modular exponentiation and inverses -/

section PowMod

theorem powModGo_eq {M : ℕ} (hM : 0 < M) :
    ∀ f b e acc, e < 2 ^ f → acc < M → powModGo M f b e acc = acc * b ^ e % M
  | 0, b, e, acc, he, ha => by
    have : e = 0 := by simpa using he
    subst this
    show acc = _
    rw [pow_zero, mul_one, Nat.mod_eq_of_lt ha]
  | f + 1, b, e, acc, he, ha => by
    show force (b * b % M) (fun b2 => force (@Bool.rec (fun _ => ℕ) acc (acc * b % M) (Nat.beq (e % 2) 1))
      fun acc' => powModGo M f b2 (e / 2) acc') = _
    rw [force_eq, force_eq]
    have hacc : @Bool.rec (fun _ => ℕ) acc (acc * b % M) (Nat.beq (e % 2) 1) = acc * b ^ (e % 2) % M ∧
        @Bool.rec (fun _ => ℕ) acc (acc * b % M) (Nat.beq (e % 2) 1) < M := by
      rcases Nat.mod_two_eq_zero_or_one e with h | h
      · rw [h]; exact ⟨by rw [pow_zero, mul_one, Nat.mod_eq_of_lt ha]; rfl, ha⟩
      · rw [h]; exact ⟨by rw [pow_one]; rfl, Nat.mod_lt _ hM⟩
    rw [powModGo_eq hM f _ _ _ (by rw [pow_succ] at he; omega) hacc.2, hacc.1]
    have he2 : b ^ e = b ^ (e % 2) * (b * b) ^ (e / 2) := by
      rw [← sq, ← pow_mul, ← pow_add]; congr 1; omega
    rw [Nat.mod_mul_mod, ← Nat.mul_mod_mod, ← Nat.pow_mod, Nat.mul_mod_mod, he2, mul_assoc]

theorem powMod_eq {M : ℕ} (hM : 0 < M) (f b e : ℕ) (he : e < 2 ^ f) : powMod f b e M = b ^ e % M := by
  show powModGo M f (b % M) e (1 % M) = _
  rw [powModGo_eq hM f _ _ _ he (Nat.mod_lt _ hM), Nat.mod_mul_mod, one_mul, ← Nat.pow_mod]

/-- **Euler**: `u · u^{φ(M) − 1} ≡ 1 (mod M)` for `u` coprime to `M > 1`. -/
theorem mul_powMod_totient {M : ℕ} (hM : 1 < M) {u : ℕ} (hu : u.Coprime M) (f : ℕ) (he : totient M - 1 < 2 ^ f) :
    u * powMod f u (totient M - 1) M % M = 1 := by
  rw [powMod_eq (by omega) f u _ he, Nat.mul_mod_mod, ← pow_succ',
    Nat.sub_add_cancel (Nat.totient_pos.mpr (by omega))]
  have := Nat.ModEq.pow_totient hu
  rw [Nat.ModEq, Nat.one_mod_eq_one.mpr (by omega)] at this
  exact this

lemma powMod_lt {M : ℕ} (hM : 0 < M) (f b e : ℕ) (he : e < 2 ^ f) : powMod f b e M < M := by
  rw [powMod_eq hM f b e he]; exact Nat.mod_lt _ hM

end PowMod

/-! ### The row test -/

section Test

/-- The exact-division test: for `f < 2^H` and `m c ≡ 1 (mod 2^H)`, `m ∣ f ⟺ f c mod 2^H ≤ ⌊(2^H − 1)/m⌋`. -/
theorem exact_div_test {m H c f : ℕ} (hm : 0 < m) (hf : f < 2 ^ H) (hc : m * c % 2 ^ H = 1) :
    f * c % 2 ^ H ≤ (2 ^ H - 1) / m ↔ m ∣ f := by
  have hP : 0 < 2 ^ H := Nat.two_pow_pos H
  constructor
  · intro h
    have h1 : m * (f * c % 2 ^ H) < 2 ^ H :=
      calc m * (f * c % 2 ^ H) ≤ m * ((2 ^ H - 1) / m) := Nat.mul_le_mul_left m h
        _ ≤ 2 ^ H - 1 := Nat.mul_div_le _ _
        _ < 2 ^ H := Nat.sub_lt hP one_pos
    have h2 : m * (f * c % 2 ^ H) % 2 ^ H = f := by
      rw [Nat.mul_mod_mod, show m * (f * c) = f * (m * c) by ring, Nat.mul_mod, hc, mul_one, Nat.mod_mod,
        Nat.mod_eq_of_lt hf]
    rw [Nat.mod_eq_of_lt h1] at h2
    exact ⟨_, h2.symm⟩
  · rintro ⟨k, rfl⟩
    have hk : k < 2 ^ H := lt_of_le_of_lt (Nat.le_mul_of_pos_left k hm) hf
    rw [show m * k * c = k * (m * c) by ring, Nat.mul_mod, hc, mul_one, Nat.mod_mod, Nat.mod_eq_of_lt hk]
    rw [Nat.le_div_iff_mul_le hm, mul_comm]
    omega

lemma land_two_pow_eq_zero {z H : ℕ} (hz : z < 2 ^ (H + 1)) : z &&& 2 ^ H = 0 ↔ z < 2 ^ H := by
  have key : z &&& 2 ^ H = 0 ↔ z.testBit H = false := by
    constructor
    · intro h
      have := congrArg (fun x => Nat.testBit x H) h
      simpa [Nat.testBit_and, Nat.testBit_two_pow] using this
    · intro h
      apply Nat.eq_of_testBit_eq; intro k
      simp only [Nat.testBit_and, Nat.testBit_two_pow, Nat.zero_testBit]
      by_cases hk : H = k
      · subst hk; simp [h]
      · simp [hk]
  rw [key, Nat.testBit_eq_decide_div_mod_eq]
  have hd : z / 2 ^ H < 2 := by
    rw [Nat.div_lt_iff_lt_mul (Nat.two_pow_pos H)]; rw [pow_succ] at hz; linarith
  constructor
  · intro h
    have : ¬ (z / 2 ^ H % 2 = 1) := of_decide_eq_false h
    have h0 : z / 2 ^ H = 0 := by
      generalize z / 2 ^ H = y at hd this ⊢; omega
    rwa [Nat.div_eq_zero_iff_lt (Nat.two_pow_pos H)] at h0
  · intro h
    rw [Nat.div_eq_of_lt h]; simp

/-- The masks of level `(c, b, T)`, and the test of a row. -/
def tstOf (H c b T w x : ℕ) : ℕ :=
  tst c (rep (2 * H) (2 ^ b - 1) w) (rep (2 * H) (2 ^ H - 1 - T) w) (rep (2 * H) (2 ^ H) w) x

lemma tst_eq (c LOW ADD TOP x : ℕ) : tst c LOW ADD TOP x = ((x * c &&& LOW) + ADD) &&& TOP := rfl

lemma pow_lt_pow_two_mul {H : ℕ} (hH : 1 ≤ H) : 2 ^ (H + 1) ≤ 2 ^ (2 * H) :=
  Nat.pow_le_pow_right (by norm_num) (by omega)

/-- Field `i ≥ w` of the test is zero. -/
theorem tstOf_fld_ge {H c b T w x i : ℕ} (hH : 1 ≤ H) (hi : w ≤ i) : fld (2 * H) (tstOf H c b T w x) i = 0 := by
  rw [tstOf, tst_eq, fld_land, fld_rep _ _ _ _ (lt_of_lt_of_le (Nat.pow_lt_pow_right (by norm_num) (by omega))
    (pow_lt_pow_two_mul hH)), if_neg (by omega), Nat.and_zero]

/-- Field `i < w` of the test: `((x_i c mod 2^b) + 2^H − 1 − T) &&& 2^H`. -/
theorem tstOf_fld_lt {H c b T w x i : ℕ} (hH : 1 ≤ H) (hb : b ≤ H) (hc : c < 2 ^ H) (hT : T < 2 ^ H)
    (hx : ∀ i' < w, fld (2 * H) x i' < 2 ^ H) (hi : i < w) :
    fld (2 * H) (tstOf H c b T w x) i = (fld (2 * H) x i * c % 2 ^ b + (2 ^ H - 1 - T)) &&& 2 ^ H := by
  have hHF : 2 ^ H < 2 ^ (2 * H) := Nat.pow_lt_pow_right (by norm_num) (by omega)
  have hbH : 2 ^ b ≤ 2 ^ H := Nat.pow_le_pow_right (by norm_num) hb
  have hmul : ∀ i' ≤ i, fld (2 * H) (x * c) i' = fld (2 * H) x i' * c := fun i' hi' =>
    fld_mul _ _ _ _ fun i'' hi'' => by
      have h1 := hx i'' (by omega)
      calc c * fld (2 * H) x i'' < 2 ^ H * 2 ^ H := Nat.mul_lt_mul'' hc h1
        _ = 2 ^ (2 * H) := by rw [← pow_add]; ring_nf
  have hlow : ∀ i' ≤ i, fld (2 * H) (x * c &&& rep (2 * H) (2 ^ b - 1) w) i' = fld (2 * H) x i' * c % 2 ^ b :=
    fun i' hi' => by
      rw [fld_land, fld_rep _ _ _ _ (lt_of_le_of_lt (Nat.sub_le _ _) (lt_of_le_of_lt hbH hHF)), if_pos (by omega),
        Nat.and_two_pow_sub_one_eq_mod, hmul i' hi']
  rw [tstOf, tst_eq, fld_land, fld_rep _ _ _ _ hHF, if_pos hi]
  congr 1
  have := fld_add_mul (2 * H) (x * c &&& rep (2 * H) (2 ^ b - 1) w) (rep (2 * H) (2 ^ H - 1 - T) w) 1 i
    (fun i' hi' => by
      rw [hlow i' hi', fld_rep _ _ _ _ (lt_of_le_of_lt (by omega) hHF), if_pos (by omega), one_mul]
      have := Nat.mod_lt (fld (2 * H) x i' * c) (Nat.two_pow_pos b)
      have := pow_lt_pow_two_mul hH
      rw [pow_succ] at this
      omega)
  rw [one_mul] at this
  rw [this, hlow i le_rfl, fld_rep _ _ _ _ (lt_of_le_of_lt (by omega) hHF), if_pos hi, one_mul]

/-- Field `i < w` of the test is zero iff `x_i mod 2^b ≤ T` (after multiplying by `c`). -/
theorem tstOf_fld_lt_zero {H c b T w x i : ℕ} (hH : 1 ≤ H) (hb : b ≤ H) (hc : c < 2 ^ H) (hT : T < 2 ^ H)
    (hx : ∀ i' < w, fld (2 * H) x i' < 2 ^ H) (hi : i < w) :
    fld (2 * H) (tstOf H c b T w x) i = 0 ↔ fld (2 * H) x i * c % 2 ^ b ≤ T := by
  rw [tstOf_fld_lt hH hb hc hT hx hi]
  have hbH : 2 ^ b ≤ 2 ^ H := Nat.pow_le_pow_right (by norm_num) hb
  have hm := Nat.mod_lt (fld (2 * H) x i * c) (Nat.two_pow_pos b)
  rw [land_two_pow_eq_zero (by rw [pow_succ]; omega)]
  omega

/-- **The row test reads divisibility by `ℓ^{v+1}`, field by field** (`lvC`'s constants). -/
theorem tst_spec {l H v w : ℕ} (hl : l.Prime) (hH : 1 ≤ H) (hm : l ^ (v + 1) < 2 ^ H) (x : ℕ)
    (hx : ∀ i < w, fld (2 * H) x i < 2 ^ H) :
    (∀ i, w ≤ i → fld (2 * H) (tstOf H (lvC l H v).1 (lvC l H v).2.1 (lvC l H v).2.2 w x) i = 0) ∧
    ∀ i < w, (fld (2 * H) (tstOf H (lvC l H v).1 (lvC l H v).2.1 (lvC l H v).2.2 w x) i = 0 ↔
      l ^ (v + 1) ∣ fld (2 * H) x i) := by
  refine ⟨fun i hi => tstOf_fld_ge hH hi, fun i hi => ?_⟩
  by_cases h2 : l = 2
  · subst h2
    have hb : v + 1 ≤ H := (Nat.pow_lt_pow_iff_right (by norm_num)).mp hm |>.le
    show fld (2 * H) (tstOf H 1 (v + 1) 0 w x) i = 0 ↔ _
    rw [tstOf_fld_lt_zero hH hb (Nat.one_lt_two_pow (by omega)) (Nat.two_pow_pos H) hx hi, mul_one,
      Nat.le_zero, Nat.dvd_iff_mod_eq_zero]
  · have hodd : (l ^ (v + 1)).Coprime (2 ^ H) :=
      Nat.Coprime.pow _ _ ((Nat.coprime_primes hl Nat.prime_two).mpr h2)
    have hlvC : lvC l H v = (powMod H (l ^ (v + 1)) (2 ^ (H - 1) - 1) (2 ^ H), H, (2 ^ H - 1) / l ^ (v + 1)) := by
      show @Bool.rec (fun _ => ℕ × ℕ × ℕ) _ _ (Nat.beq l 2) = _
      rw [show Nat.beq l 2 = false from Bool.eq_false_iff.mpr fun h' => h2 (Nat.eq_of_beq_eq_true h')]
      exact force_eq _ _
    have hphi : totient (2 ^ H) = 2 ^ (H - 1) := by
      obtain ⟨H', rfl⟩ : ∃ H', H = H' + 1 := ⟨H - 1, by omega⟩
      rw [Nat.totient_prime_pow_succ Nat.prime_two]; simp
    have he : 2 ^ (H - 1) - 1 < 2 ^ H := lt_of_le_of_lt (Nat.sub_le _ _)
      (Nat.pow_lt_pow_right (by norm_num) (by omega))
    have hc := mul_powMod_totient (Nat.one_lt_two_pow (by omega)) hodd H (by rw [hphi]; exact he)
    rw [hphi] at hc
    have hpos : 0 < l ^ (v + 1) := pow_pos hl.pos _
    rw [hlvC]
    show fld (2 * H) (tstOf H _ H _ w x) i = 0 ↔ _
    rw [tstOf_fld_lt_zero hH le_rfl (powMod_lt (Nat.two_pow_pos H) _ _ _ he)
      (lt_of_le_of_lt (Nat.div_le_self _ _) (by omega)) hx hi]
    exact exact_div_test hpos (hx i hi) hc

end Test

/-! ### The elimination: invariants -/

section Elim

variable (l K H w : ℕ)

/-- A packed row as a vector over `ZMod (ℓ^K)` (its first `w` fields). -/
def vecK (x : ℕ) : Fin w → ZMod (l ^ K) := fun i => (fld (2 * H) x i : ZMod (l ^ K))

/-- The span over `ZMod (ℓ^K)` of the rows of a list. -/
def spanK (L : List ℕ) : Submodule (ZMod (l ^ K)) (Fin w → ZMod (l ^ K)) :=
  Submodule.span _ {y | ∃ r ∈ L, vecK l K H w r = y}

/-- The pivot rows. -/
def rowsOf (P : List Piv) : List ℕ := P.map fun q => q.2.2

/-- A remaining row at level `v` after `k` pivots `P`: lazily bounded fields, all entries divisible by `ℓ^v`, zero at
every pivot column. -/
structure RowOK (v k : ℕ) (P : List Piv) (r : ℕ) : Prop where
  bd : ∀ i < w, fld (2 * H) r i < l ^ K + k * (l ^ K) ^ 2
  dv : ∀ i < w, l ^ v ∣ fld (2 * H) r i % l ^ K
  zc : ∀ q ∈ P, q.2.1 < w → fld (2 * H) r q.2.1 % l ^ K = 0

/-- The pivots, latest first: valuation `v < K`, column `j < w`, reduced entries divisible by `ℓ^v`, the pivot entry of
valuation exactly `v`, and zero at the column of every earlier pivot. -/
def PivOK : List Piv → Prop
  | [] => True
  | q :: P => PivOK P ∧ q.1 < K ∧ q.2.1 < w ∧ (∀ i < w, fld (2 * H) q.2.2 i < l ^ K) ∧
      (∀ i < w, l ^ q.1 ∣ fld (2 * H) q.2.2 i) ∧ ¬ l ^ (q.1 + 1) ∣ fld (2 * H) q.2.2 q.2.1 ∧
      ∀ q' ∈ P, fld (2 * H) q.2.2 q'.2.1 = 0

variable {l K H w}

lemma spanK_cons (r : ℕ) (L : List ℕ) :
    spanK l K H w (r :: L) = Submodule.span _ {vecK l K H w r} ⊔ spanK l K H w L := by
  rw [spanK, spanK, ← Submodule.span_union]
  congr 1; ext y
  simp only [List.mem_cons, Set.mem_ofPred_eq, Set.mem_union, Set.mem_singleton_iff]
  constructor
  · rintro ⟨r', h | h, rfl⟩
    · exact Or.inl (h ▸ rfl)
    · exact Or.inr ⟨r', h, rfl⟩
  · rintro (rfl | ⟨r', h, rfl⟩)
    · exact ⟨r, Or.inl rfl, rfl⟩
    · exact ⟨r', Or.inr h, rfl⟩

lemma spanK_congr {L L' : List ℕ} (h : ∀ r, r ∈ L ↔ r ∈ L') : spanK l K H w L = spanK l K H w L' := by
  simp only [spanK, h]

lemma spanK_eq_bot {L : List ℕ} (h : ∀ r ∈ L, vecK l K H w r = 0) : spanK l K H w L = ⊥ := by
  rw [spanK, Submodule.span_eq_bot]
  rintro _ ⟨r, hr, rfl⟩; exact h r hr

/-- Replacing every row `r` of `L` by `g r = r + c_r z` with `z ∈ X` does not change `X ⊔ span L`. -/
theorem sup_spanK_map {L : List ℕ} {g : ℕ → ℕ} {X : Submodule (ZMod (l ^ K)) (Fin w → ZMod (l ^ K))}
    {z : Fin w → ZMod (l ^ K)} (hz : z ∈ X) (hg : ∀ r ∈ L, ∃ c : ZMod (l ^ K), vecK l K H w (g r) = vecK l K H w r + c • z) :
    X ⊔ spanK l K H w (L.map g) = X ⊔ spanK l K H w L := by
  apply le_antisymm
  · refine sup_le le_sup_left (Submodule.span_le.mpr ?_)
    rintro _ ⟨r', hr', rfl⟩
    obtain ⟨r, hr, rfl⟩ := List.mem_map.mp hr'
    obtain ⟨c, hc⟩ := hg r hr
    rw [SetLike.mem_coe, hc]
    exact add_mem (Submodule.mem_sup_right (Submodule.subset_span ⟨r, hr, rfl⟩))
      (Submodule.smul_mem _ _ (Submodule.mem_sup_left hz))
  · refine sup_le le_sup_left (Submodule.span_le.mpr ?_)
    rintro _ ⟨r, hr, rfl⟩
    obtain ⟨c, hc⟩ := hg r hr
    rw [SetLike.mem_coe, show vecK l K H w r = vecK l K H w (g r) - c • z by rw [hc]; abel]
    exact sub_mem (Submodule.mem_sup_right (Submodule.subset_span ⟨g r, List.mem_map_of_mem hr, rfl⟩))
      (Submodule.smul_mem _ _ (Submodule.mem_sup_left hz))

lemma mapA_eq (g : ℕ → ℕ) : ∀ (L acc : List ℕ), mapA g L acc = (L.map g).reverse ++ acc
  | [], acc => rfl
  | a :: L, acc => by
    show force (g a) (fun b => mapA g L (b :: acc)) = _
    rw [force_eq, mapA_eq g L]; simp

lemma elim_eq (F N lv ui j p r : ℕ) :
    elim F N lv ui j p r = r + (N - fld F r j % N / lv * ui % N) % N * p := by
  unfold elim; rw [force_eq, force_eq, force_eq]; rfl

end Elim

/-! ### One elimination step -/

section Step

/-- The cell's context: `ℓ` prime, `K ≥ 1`, `H ≥ 1`, and `ℓ^K + R ℓ^{2K} < 2^H` (`R` rows). -/
structure Ctx (l K H R : ℕ) : Prop where
  prime : l.Prime
  K_pos : 0 < K
  H_pos : 1 ≤ H
  bound : l ^ K + R * (l ^ K) ^ 2 < 2 ^ H

variable {l K H w R : ℕ}

lemma Ctx.N_pos (c : Ctx l K H R) : 0 < l ^ K := pow_pos c.prime.pos _

lemma Ctx.one_lt_N (c : Ctx l K H R) : 1 < l ^ K := Nat.one_lt_pow (by have := c.K_pos; omega) c.prime.one_lt

lemma Ctx.N_lt (c : Ctx l K H R) : l ^ K < 2 ^ H := lt_of_le_of_lt (Nat.le_add_right _ _) c.bound

lemma Ctx.H_le_F (_c : Ctx l K H R) : 2 ^ H ≤ 2 ^ (2 * H) := Nat.pow_le_pow_right (by norm_num) (by omega)

lemma Ctx.lt_H (c : Ctx l K H R) {k x : ℕ} (hk : k ≤ R) (hx : x < l ^ K + k * (l ^ K) ^ 2) : x < 2 ^ H :=
  lt_of_lt_of_le hx (le_trans (Nat.add_le_add_left (Nat.mul_le_mul_right _ hk) _) c.bound.le)

lemma Ctx.pow_dvd_N (_c : Ctx l K H R) {v : ℕ} (hv : v ≤ K) : l ^ v ∣ l ^ K := Nat.pow_dvd_pow l hv

/-- The totient exponent the kernel uses. -/
lemma Ctx.totient_eq (c : Ctx l K H R) : l ^ (K - 1) * (l - 1) - 1 = totient (l ^ K) - 1 := by
  obtain ⟨K', hK⟩ : ∃ K', K = K' + 1 := ⟨K - 1, by have := c.K_pos; omega⟩
  subst hK
  rw [Nat.totient_prime_pow_succ c.prime]; rfl

lemma Ctx.e_lt (c : Ctx l K H R) : totient (l ^ K) - 1 < 2 ^ H :=
  lt_of_le_of_lt (le_trans (Nat.sub_le _ _) (Nat.totient_le _)) c.N_lt

/-- **The pivot**: from a row with a field of valuation exactly `v`, the reduced row, its entry `ℓ^v u` and `u⁻¹`. -/
theorem pivot_facts (c : Ctx l K H R) {v k : ℕ} {P : List Piv} {r j : ℕ} (hv : v < K) (hr : RowOK l K H w v k P r)
    (hj : j < w) (hnd : ¬ l ^ (v + 1) ∣ fld (2 * H) r j) :
    let p := redRow (2 * H) (l ^ K) w r
    let u := fld (2 * H) p j / l ^ v
    (∀ i < w, fld (2 * H) p i = fld (2 * H) r i % l ^ K) ∧ (∀ i < w, fld (2 * H) p i < l ^ K) ∧
    (∀ i < w, l ^ v ∣ fld (2 * H) p i) ∧ ¬ l ^ (v + 1) ∣ fld (2 * H) p j ∧
    (∀ q ∈ P, fld (2 * H) p q.2.1 = 0) ∧ fld (2 * H) p j = l ^ v * u ∧
    u * powMod H u (totient (l ^ K) - 1) (l ^ K) % l ^ K = 1 := by
  intro p u
  have hNF : l ^ K ≤ 2 ^ (2 * H) := le_trans c.N_lt.le c.H_le_F
  have hp : ∀ i < w, fld (2 * H) p i = fld (2 * H) r i % l ^ K := fun i hi => by
    show fld (2 * H) (redRow (2 * H) (l ^ K) w r) i = _
    rw [fld_redRow c.N_pos hNF, if_pos hi]
  have hdv : ∀ i < w, l ^ v ∣ fld (2 * H) p i := fun i hi => by rw [hp i hi]; exact hr.dv i hi
  have hnd' : ¬ l ^ (v + 1) ∣ fld (2 * H) p j := by
    rw [hp j hj, Nat.dvd_mod_iff (c.pow_dvd_N (by omega))]; exact hnd
  have hu : fld (2 * H) p j = l ^ v * u := (Nat.mul_div_cancel' (hdv j hj)).symm
  refine ⟨hp, fun i hi => by rw [hp i hi]; exact Nat.mod_lt _ c.N_pos, hdv, hnd', fun q hq => ?_, hu, ?_⟩
  · by_cases hqw : q.2.1 < w
    · rw [hp _ hqw]; exact hr.zc q hq hqw
    · show fld (2 * H) (redRow (2 * H) (l ^ K) w r) q.2.1 = 0
      rw [fld_redRow c.N_pos hNF, if_neg hqw]
  · have hlu : ¬ l ∣ u := fun h => hnd' (by rw [hu, pow_succ]; exact Nat.mul_dvd_mul_left _ h)
    have hcop : u.Coprime (l ^ K) :=
      Nat.Coprime.pow_right _ ((Nat.Prime.coprime_iff_not_dvd c.prime).mpr hlu).symm
    exact mul_powMod_totient c.one_lt_N hcop H c.e_lt

/-- **One elimination**: the row `x` (level `v`, `k` pivots) cleared at column `j` by the new pivot `p`. -/
theorem elim_step (c : Ctx l K H R) {v k : ℕ} {P : List Piv} {j p u ui x : ℕ} (hk : k + 1 ≤ R) (hv : v < K)
    (hj : j < w) (hpl : ∀ i < w, fld (2 * H) p i < l ^ K) (hpd : ∀ i < w, l ^ v ∣ fld (2 * H) p i)
    (hpz : ∀ q ∈ P, fld (2 * H) p q.2.1 = 0) (ha : fld (2 * H) p j = l ^ v * u) (hui : u * ui % l ^ K = 1)
    (hx : RowOK l K H w v k P x) :
    let y := elim (2 * H) (l ^ K) (l ^ v) ui j p x
    (∃ cc : ZMod (l ^ K), vecK l K H w y = vecK l K H w x + cc • vecK l K H w p) ∧
    RowOK l K H w v (k + 1) ((v, j, p) :: P) y ∧
    (RowOK l K H w (v + 1) k P x → RowOK l K H w (v + 1) (k + 1) ((v, j, p) :: P) y) := by
  intro y
  set N := l ^ K with hN
  set b := fld (2 * H) x j % N / l ^ v with hb
  set cc := (N - b * ui % N) % N with hcc
  have hy : y = x + cc * p := elim_eq _ _ _ _ _ _ _
  have hccN : cc < N := Nat.mod_lt _ c.N_pos
  have hfld : ∀ i < w, fld (2 * H) y i = fld (2 * H) x i + cc * fld (2 * H) p i := fun i hi => by
    rw [hy]
    refine fld_add_mul _ _ _ _ _ fun i' hi' => ?_
    have h1 := hx.bd i' (by omega)
    have h2 : cc * fld (2 * H) p i' < (l ^ K) ^ 2 := by rw [sq]; exact Nat.mul_lt_mul'' hccN (hpl i' (by omega))
    have : fld (2 * H) x i' + cc * fld (2 * H) p i' < 2 ^ H := c.lt_H hk (by rw [add_mul, one_mul]; omega)
    exact lt_of_lt_of_le this c.H_le_F
  have hvN : l ^ v ∣ N := c.pow_dvd_N hv.le
  have hbd : ∀ i < w, fld (2 * H) y i < l ^ K + (k + 1) * (l ^ K) ^ 2 := fun i hi => by
    rw [hfld i hi]
    have h1 := hx.bd i hi
    have h2 : cc * fld (2 * H) p i < (l ^ K) ^ 2 := by rw [sq]; exact Nat.mul_lt_mul'' hccN (hpl i hi)
    rw [add_mul, one_mul]; omega
  have hzc : ∀ q ∈ (v, j, p) :: P, q.2.1 < w → fld (2 * H) y q.2.1 % N = 0 := fun q hq hqw => by
    rcases List.mem_cons.mp hq with rfl | hq
    · -- the new pivot column
      show (fld (2 * H) y j) % N = 0
      rw [hfld j hj, ← Nat.dvd_iff_mod_eq_zero, ← ZMod.natCast_eq_zero_iff]
      have hm : fld (2 * H) x j % N = l ^ v * b := (Nat.mul_div_cancel' (hx.dv j hj)).symm
      have e1 : (fld (2 * H) x j : ZMod N) = (l : ZMod N) ^ v * b := by
        rw [← ZMod.natCast_mod, hm]; push_cast; ring
      have e2 : (cc : ZMod N) = -((b : ZMod N) * ui) := by
        rw [hcc, ZMod.natCast_mod, Nat.cast_sub (Nat.mod_lt _ c.N_pos).le, ZMod.natCast_self, ZMod.natCast_mod]
        push_cast; ring
      have e3 : (u : ZMod N) * ui = 1 := by
        have := congrArg (fun t : ℕ => (t : ZMod N)) hui
        simpa [ZMod.natCast_mod] using this
      push_cast
      rw [e1, e2, ha]
      push_cast
      linear_combination (-(l : ZMod N) ^ v * b) * e3
    · rw [hfld _ hqw, hpz q hq, mul_zero, add_zero]; exact hx.zc q hq hqw
  refine ⟨⟨(cc : ZMod N), funext fun i => ?_⟩, ⟨hbd, fun i hi => ?_, hzc⟩, fun hx1 => ⟨hbd, fun i hi => ?_, hzc⟩⟩
  · simp only [vecK, Pi.add_apply, Pi.smul_apply, smul_eq_mul, hfld i i.2, Nat.cast_add, Nat.cast_mul]
  · rw [hfld i hi, Nat.dvd_mod_iff hvN]
    exact Nat.dvd_add ((Nat.dvd_mod_iff hvN).mp (hx.dv i hi)) (Nat.dvd_mul_left_of_dvd (hpd i hi) _)
  · have hv1N : l ^ (v + 1) ∣ N := c.pow_dvd_N hv
    rw [hfld i hi, Nat.dvd_mod_iff hv1N]
    refine Nat.dvd_add ((Nat.dvd_mod_iff hv1N).mp (hx1.dv i hi)) ?_
    -- `ℓ ∣ cc`: the multiplier of a row divisible by `ℓ^{v+1}`
    have hlb : l ∣ b := by
      obtain ⟨t, ht⟩ := hx1.dv j hj
      rw [hb, ht, pow_succ, mul_assoc, Nat.mul_div_cancel_left _ (pow_pos c.prime.pos v)]
      exact Nat.dvd_mul_right l t
    have hlN : l ∣ N := dvd_trans (dvd_pow_self l (by have := c.K_pos; omega)) dvd_rfl
    have hlc : l ∣ cc := by
      rw [hcc, Nat.dvd_mod_iff hlN]
      exact Nat.dvd_sub hlN ((Nat.dvd_mod_iff hlN).mpr (Nat.dvd_mul_right_of_dvd hlb _))
    rw [pow_succ', mul_comm l]
    exact Nat.mul_dvd_mul (hpd i hi) hlc |>.trans (by rw [mul_comm])
end Step

/-! ### One level -/

section Level

variable {l K H w R : ℕ}

lemma RowOK.mono {v k : ℕ} {P : List Piv} {r : ℕ} (h : RowOK l K H w (v + 1) k P r) : RowOK l K H w v k P r :=
  ⟨h.bd, fun i hi => dvd_trans (pow_dvd_pow l (Nat.le_succ v)) (h.dv i hi), h.zc⟩

lemma level_zero (F H N w v lv e : ℕ) (t : ℕ → ℕ) (pend done : List ℕ) (P : List Piv) :
    level F H N w v lv e t 0 pend done P = (done, P) := rfl

lemma level_nil (F H N w v lv e : ℕ) (t : ℕ → ℕ) (f : ℕ) (done : List ℕ) (P : List Piv) :
    level F H N w v lv e t (f + 1) [] done P = (done, P) := rfl

lemma level_cons_zero (F H N w v lv e : ℕ) (t : ℕ → ℕ) (f r : ℕ) (rest done : List ℕ) (P : List Piv)
    (h : t r = 0) : level F H N w v lv e t (f + 1) (r :: rest) done P = level F H N w v lv e t f rest (r :: done) P := by
  show force (t r) (fun tr => @Bool.rec (fun _ => List ℕ × List Piv) _ _ (Nat.beq tr 0)) = _
  rw [force_eq, h]; rfl

lemma level_cons_ne (F H N w v lv e : ℕ) (t : ℕ → ℕ) (f r : ℕ) (rest done : List ℕ) (P : List Piv)
    (h : t r ≠ 0) : level F H N w v lv e t (f + 1) (r :: rest) done P =
      level F H N w v lv e t f
        (mapA (elim F N lv (powMod H (fld F (redRow F N w r) (firstNZ F (t r) w 0) / lv) e N) (firstNZ F (t r) w 0)
          (redRow F N w r)) rest [])
        (mapA (elim F N lv (powMod H (fld F (redRow F N w r) (firstNZ F (t r) w 0) / lv) e N) (firstNZ F (t r) w 0)
          (redRow F N w r)) done [])
        ((v, firstNZ F (t r) w 0, redRow F N w r) :: P) := by
  show force (t r) (fun tr => @Bool.rec (fun _ => List ℕ × List Piv) _ _ (Nat.beq tr 0)) = _
  rw [force_eq, show Nat.beq (t r) 0 = false from Bool.eq_false_iff.mpr fun h' => h (Nat.eq_of_beq_eq_true h')]
  show force _ (fun j => force _ (fun p => force _ (fun ui => _))) = _
  rw [force_eq, force_eq, force_eq]; rfl

lemma mem_mapA_nil {g : ℕ → ℕ} {L : List ℕ} {y : ℕ} : y ∈ mapA g L [] ↔ ∃ x ∈ L, g x = y := by
  rw [mapA_eq, List.append_nil, List.mem_reverse, List.mem_map]

/-- **One level**: the invariants survive, and the row span is unchanged. -/
theorem level_spec (cx : Ctx l K H R) {v : ℕ} (hv : v < K) : ∀ (f : ℕ) (pend done : List ℕ) (P : List Piv),
    pend.length ≤ f → pend.length + done.length + P.length ≤ R → PivOK l K H w P →
    (∀ r ∈ pend, RowOK l K H w v P.length P r) → (∀ r ∈ done, RowOK l K H w (v + 1) P.length P r) →
    PivOK l K H w (level (2 * H) H (l ^ K) w v (l ^ v) (totient (l ^ K) - 1)
        (tstOf H (lvC l H v).1 (lvC l H v).2.1 (lvC l H v).2.2 w) f pend done P).2 ∧
    (∀ r ∈ (level (2 * H) H (l ^ K) w v (l ^ v) (totient (l ^ K) - 1)
        (tstOf H (lvC l H v).1 (lvC l H v).2.1 (lvC l H v).2.2 w) f pend done P).1,
      RowOK l K H w (v + 1) (level (2 * H) H (l ^ K) w v (l ^ v) (totient (l ^ K) - 1)
        (tstOf H (lvC l H v).1 (lvC l H v).2.1 (lvC l H v).2.2 w) f pend done P).2.length
        (level (2 * H) H (l ^ K) w v (l ^ v) (totient (l ^ K) - 1)
        (tstOf H (lvC l H v).1 (lvC l H v).2.1 (lvC l H v).2.2 w) f pend done P).2 r) ∧
    (level (2 * H) H (l ^ K) w v (l ^ v) (totient (l ^ K) - 1)
        (tstOf H (lvC l H v).1 (lvC l H v).2.1 (lvC l H v).2.2 w) f pend done P).1.length +
      (level (2 * H) H (l ^ K) w v (l ^ v) (totient (l ^ K) - 1)
        (tstOf H (lvC l H v).1 (lvC l H v).2.1 (lvC l H v).2.2 w) f pend done P).2.length ≤ R ∧
    spanK l K H w pend ⊔ spanK l K H w done ⊔ spanK l K H w (rowsOf P) =
      spanK l K H w (level (2 * H) H (l ^ K) w v (l ^ v) (totient (l ^ K) - 1)
        (tstOf H (lvC l H v).1 (lvC l H v).2.1 (lvC l H v).2.2 w) f pend done P).1 ⊔
      spanK l K H w (rowsOf (level (2 * H) H (l ^ K) w v (l ^ v) (totient (l ^ K) - 1)
        (tstOf H (lvC l H v).1 (lvC l H v).2.1 (lvC l H v).2.2 w) f pend done P).2) := by
  intro f
  induction f with
  | zero =>
    intro pend done P hf hc hP _ hd
    have : pend = [] := List.eq_nil_of_length_eq_zero (by omega)
    subst this
    rw [level_zero]
    refine ⟨hP, hd, by simpa using hc, ?_⟩
    rw [spanK_eq_bot (by simp), bot_sup_eq]
  | succ f ih =>
    intro pend done P hf hc hP hp hd
    cases pend with
    | nil =>
      rw [level_nil]
      refine ⟨hP, hd, by simpa using hc, ?_⟩
      rw [spanK_eq_bot (by simp), bot_sup_eq]
    | cons r rest =>
      set t := tstOf H (lvC l H v).1 (lvC l H v).2.1 (lvC l H v).2.2 w with ht
      have hr := hp r (List.mem_cons_self ..)
      have hPR : P.length + 1 ≤ R := by simp at hc; omega
      have hm : l ^ (v + 1) < 2 ^ H := lt_of_le_of_lt (Nat.pow_le_pow_right cx.prime.pos hv) cx.N_lt
      have hrx : ∀ i < w, fld (2 * H) r i < 2 ^ H := fun i hi => cx.lt_H (by omega) (hr.bd i hi)
      obtain ⟨hge, hlt⟩ := tst_spec (w := w) cx.prime cx.H_pos hm r hrx
      by_cases h0 : t r = 0
      · rw [level_cons_zero _ _ _ _ _ _ _ _ _ _ _ _ _ h0]
        have hr1 : RowOK l K H w (v + 1) P.length P r := ⟨hr.bd, fun i hi => by
          have := (hlt i hi).mp (by rw [← ht, h0, fld_zero])
          exact (Nat.dvd_mod_iff (cx.pow_dvd_N hv)).mpr this, hr.zc⟩
        obtain ⟨h1, h2, h3, h4⟩ := ih rest (r :: done) P (by simp at hf; omega) (by simp at hc ⊢; omega) hP
          (fun x hx => hp x (List.mem_cons_of_mem _ hx))
          (fun x hx => by
            rcases List.mem_cons.mp hx with rfl | hx
            · exact hr1
            · exact hd x hx)
        refine ⟨h1, h2, h3, ?_⟩
        rw [← h4, spanK_cons, spanK_cons]
        ac_rfl
      · rw [level_cons_ne _ _ _ _ _ _ _ _ _ _ _ _ _ h0]
        -- the pivot column
        have hex : ∃ i', 0 ≤ i' ∧ i' < 0 + w ∧ fld (2 * H) (t r) i' ≠ 0 := by
          by_contra hne
          push Not at hne
          refine h0 (eq_zero_of_fld (F := 2 * H) (by have := cx.H_pos; omega) fun i => ?_)
          by_cases hi : i < w
          · exact hne i (Nat.zero_le _) (by omega)
          · exact hge i (by omega)
        obtain ⟨hjw, hjnz⟩ := firstNZ_spec (2 * H) (t r) w 0 hex
        set j := firstNZ (2 * H) (t r) w 0 with hj
        rw [zero_add] at hjw
        have hnd : ¬ l ^ (v + 1) ∣ fld (2 * H) r j := fun h => hjnz ((hlt j hjw).mpr h)
        obtain ⟨hpf, hpl, hpd, hpnd, hpz, hpu, hui⟩ := pivot_facts cx hv hr hjw hnd
        set p := redRow (2 * H) (l ^ K) w r with hp_def
        set u := fld (2 * H) p j / l ^ v with hu_def
        set g := elim (2 * H) (l ^ K) (l ^ v) (powMod H u (totient (l ^ K) - 1) (l ^ K)) j p with hg
        have hstep : ∀ x, RowOK l K H w v P.length P x →
            (∃ cc : ZMod (l ^ K), vecK l K H w (g x) = vecK l K H w x + cc • vecK l K H w p) ∧
            RowOK l K H w v (P.length + 1) ((v, j, p) :: P) (g x) ∧
            (RowOK l K H w (v + 1) P.length P x → RowOK l K H w (v + 1) (P.length + 1) ((v, j, p) :: P) (g x)) :=
          fun x hx => elim_step cx hPR hv hjw hpl hpd hpz hpu hui hx
        have hP' : PivOK l K H w ((v, j, p) :: P) := ⟨hP, hv, hjw, hpl, hpd, hpnd, hpz⟩
        obtain ⟨h1, h2, h3, h4⟩ := ih (mapA g rest []) (mapA g done []) ((v, j, p) :: P)
          (by rw [mapA_eq]; simp at hf ⊢; omega) (by rw [mapA_eq, mapA_eq]; simp at hc ⊢; omega) hP'
          (fun y hy => by
            obtain ⟨x, hx, rfl⟩ := mem_mapA_nil.mp hy
            exact (hstep x (hp x (List.mem_cons_of_mem _ hx))).2.1)
          (fun y hy => by
            obtain ⟨x, hx, rfl⟩ := mem_mapA_nil.mp hy
            exact (hstep x (hd x hx).mono).2.2 (hd x hx))
        refine ⟨h1, h2, h3, ?_⟩
        rw [← h4]
        have hvp : vecK l K H w p = vecK l K H w r := funext fun i => by
          simp only [vecK]; rw [hpf i i.2, ZMod.natCast_mod]
        have hrest : spanK l K H w (mapA g rest []) = spanK l K H w (rest.map g) :=
          spanK_congr fun y => by rw [mapA_eq, List.append_nil, List.mem_reverse]
        have hdone : spanK l K H w (mapA g done []) = spanK l K H w (done.map g) :=
          spanK_congr fun y => by rw [mapA_eq, List.append_nil, List.mem_reverse]
        have hrows : spanK l K H w (rowsOf ((v, j, p) :: P)) =
            Submodule.span _ {vecK l K H w r} ⊔ spanK l K H w (rowsOf P) := by
          rw [show rowsOf ((v, j, p) :: P) = p :: rowsOf P from rfl, spanK_cons, hvp]
        rw [hrest, hdone, hrows, spanK_cons]
        set Z := Submodule.span (ZMod (l ^ K)) {vecK l K H w r} ⊔ spanK l K H w (rowsOf P) with hZ
        have hz : ∀ X, vecK l K H w p ∈ Z ⊔ X := fun X => by
          rw [hvp]; exact Submodule.mem_sup_left (Submodule.mem_sup_left (Submodule.mem_span_singleton_self _))
        have hgr : ∀ L : List ℕ, (∀ x ∈ L, RowOK l K H w v P.length P x) → ∀ x ∈ L,
            ∃ cc : ZMod (l ^ K), vecK l K H w (g x) = vecK l K H w x + cc • vecK l K H w p :=
          fun L hL x hx => (hstep x (hL x hx)).1
        have e1 := sup_spanK_map (hz (spanK l K H w (done.map g))) (hgr rest fun x hx => hp x (List.mem_cons_of_mem _ hx))
        have e2 := sup_spanK_map (hz (spanK l K H w rest)) (hgr done fun x hx => (hd x hx).mono)
        calc Submodule.span (ZMod (l ^ K)) {vecK l K H w r} ⊔ spanK l K H w rest ⊔ spanK l K H w done ⊔
              spanK l K H w (rowsOf P)
            = Z ⊔ spanK l K H w rest ⊔ spanK l K H w done := by rw [hZ]; ac_rfl
          _ = Z ⊔ spanK l K H w rest ⊔ spanK l K H w (done.map g) := by rw [e2]
          _ = Z ⊔ spanK l K H w (done.map g) ⊔ spanK l K H w (rest.map g) := by
              rw [sup_right_comm, e1]
          _ = _ := by rw [hZ]; ac_rfl

end Level

/-! ### All levels -/

section Levels

variable {l K H w R : ℕ}

lemma prod_rec_eq {α β γ : Type} (f : α → β → γ) (q : α × β) : @Prod.rec α β (fun _ => γ) f q = f q.1 q.2 := by
  cases q; rfl

lemma levelAt_eq (l H N w e R v : ℕ) (rows : List ℕ) (P : List Piv) :
    levelAt l H N w e R v rows P =
      level (2 * H) H N w v (l ^ v) e (tstOf H (lvC l H v).1 (lvC l H v).2.1 (lvC l H v).2.2 w) R rows [] P := by
  unfold levelAt
  rw [force_eq]
  generalize lvC l H v = q
  obtain ⟨c, b, T⟩ := q
  show force c (fun c => force _ fun LOW => force _ fun ADD => force _ fun TOP => _) = _
  rw [force_eq, force_eq, force_eq, force_eq]; rfl

lemma levels_zero (l H N w e R v : ℕ) (rows : List ℕ) (P : List Piv) : levels l H N w e R 0 v rows P = P := rfl

lemma levels_succ (l H N w e R k v : ℕ) (rows : List ℕ) (P : List Piv) :
    levels l H N w e R (k + 1) v rows P =
      levels l H N w e R k (v + 1) (levelAt l H N w e R v rows P).1 (levelAt l H N w e R v rows P).2 := by
  show @Prod.rec (List ℕ) (List Piv) (fun _ => List Piv) _ (levelAt l H N w e R v rows P) = _
  generalize levelAt l H N w e R v rows P = q
  obtain ⟨a, b⟩ := q; rfl

theorem levels_spec (cx : Ctx l K H R) : ∀ (k v : ℕ) (rows : List ℕ) (P : List Piv), v + k = K →
    (∀ r ∈ rows, RowOK l K H w v P.length P r) → rows.length + P.length ≤ R → PivOK l K H w P →
    PivOK l K H w (levels l H (l ^ K) w (totient (l ^ K) - 1) R k v rows P) ∧
    spanK l K H w rows ⊔ spanK l K H w (rowsOf P) =
      spanK l K H w (rowsOf (levels l H (l ^ K) w (totient (l ^ K) - 1) R k v rows P)) := by
  intro k
  induction k with
  | zero =>
    intro v rows P hvk hrows _ hP
    rw [levels_zero]
    refine ⟨hP, ?_⟩
    rw [spanK_eq_bot fun r hr => funext fun i => ?_, bot_sup_eq]
    have h1 := (hrows r hr).dv i i.2
    rw [add_zero] at hvk; subst hvk
    have h0 : fld (2 * H) r i % l ^ v = 0 :=
      Nat.eq_zero_of_dvd_of_lt h1 (Nat.mod_lt _ (pow_pos cx.prime.pos _))
    simp only [vecK, Pi.zero_apply]
    rw [← ZMod.natCast_mod, h0, Nat.cast_zero]
  | succ k ih =>
    intro v rows P hvk hrows hlen hP
    rw [levels_succ, levelAt_eq]
    obtain ⟨h1, h2, h3, h4⟩ := level_spec (w := w) cx (v := v) (by omega) R rows [] P (by omega) (by simpa using hlen) hP
      hrows (by simp)
    obtain ⟨g1, g2⟩ := ih (v + 1) _ _ (by omega) h2 h3 h1
    refine ⟨g1, ?_⟩
    rw [← g2, ← h4, spanK_eq_bot (L := []) (by simp), sup_bot_eq]

/-- **`smithPK` is sound**: on rows with reduced fields, the pivots satisfy `PivOK` and span the row span over
`ZMod (ℓ^K)`. -/
theorem smithPK_spec {rows : List ℕ} (cx : Ctx l K H rows.length)
    (hrows : ∀ r ∈ rows, ∀ i < w, fld (2 * H) r i < l ^ K) :
    PivOK l K H w (smithPK l K H w rows) ∧
    spanK l K H w rows = spanK l K H w (rowsOf (smithPK l K H w rows)) := by
  have e : smithPK l K H w rows =
      levels l H (l ^ K) w (totient (l ^ K) - 1) rows.length K 0 rows [] := by
    rw [← cx.totient_eq]; rfl
  rw [e]
  obtain ⟨h1, h2⟩ := levels_spec (w := w) cx K 0 rows [] (by omega)
    (fun r hr => ⟨fun i hi => by simpa using hrows r hr i hi, fun i _ => by simp, fun q hq => by simp at hq⟩)
    (by simp) trivial
  refine ⟨h1, ?_⟩
  rw [← h2, show rowsOf [] = [] from rfl, spanK_eq_bot (L := []) (by simp), sup_bot_eq]

end Levels

/-! ### The echelon family and the cardinality of the span -/

section Card

variable {l K H w : ℕ}

/-- The properties of one pivot `(v, j, p)`. -/
def PivProp (l K H w : ℕ) (q : Piv) : Prop :=
  q.1 < K ∧ q.2.1 < w ∧ (∀ i < w, fld (2 * H) q.2.2 i < l ^ K) ∧ (∀ i < w, l ^ q.1 ∣ fld (2 * H) q.2.2 i) ∧
    ¬ l ^ (q.1 + 1) ∣ fld (2 * H) q.2.2 q.2.1

/-- An echelon family in creation order: every later pivot row vanishes at the column of every earlier one. -/
def Ech (l K H w : ℕ) : List Piv → Prop
  | [] => True
  | q :: C => Ech l K H w C ∧ PivProp l K H w q ∧ ∀ q' ∈ C, fld (2 * H) q'.2.2 q.2.1 = 0

lemma ech_append_single : ∀ (C : List Piv) (q : Piv), Ech l K H w C → PivProp l K H w q →
    (∀ q' ∈ C, fld (2 * H) q.2.2 q'.2.1 = 0) → Ech l K H w (C ++ [q])
  | [], q, _, hq, _ => ⟨trivial, hq, fun _ h => by simp at h⟩
  | q0 :: C, q, ⟨hC, h0, hz⟩, hq, hqz => by
    refine ⟨ech_append_single C q hC hq fun q' hq' => hqz q' (List.mem_cons_of_mem _ hq'), h0, fun q' hq' => ?_⟩
    rcases List.mem_append.mp hq' with h | h
    · exact hz q' h
    · rw [List.mem_singleton.mp h]; exact hqz q0 (List.mem_cons_self ..)

theorem pivOK_ech : ∀ P : List Piv, PivOK l K H w P → Ech l K H w P.reverse
  | [], _ => trivial
  | q :: P, ⟨hP, h1, h2, h3, h4, h5, h6⟩ => by
    rw [List.reverse_cons]
    exact ech_append_single _ _ (pivOK_ech P hP) ⟨h1, h2, h3, h4, h5⟩ fun q' hq' => h6 q' (List.mem_reverse.mp hq')

/-- `x • p = 0 ↔ ℓ^v x = 0` for a pivot row `p` of valuation `v`. -/
theorem smul_pivot_eq_zero (hl : l.Prime) {q : Piv} (hq : PivProp l K H w q) (a : ZMod (l ^ K)) :
    a • vecK l K H w q.2.2 = 0 ↔ (l : ZMod (l ^ K)) ^ q.1 * a = 0 := by
  obtain ⟨hvK, hj, _, hdv, hnd⟩ := hq
  obtain ⟨u, hu⟩ := hdv _ hj
  have hlu : ¬ l ∣ u := fun h => hnd (by rw [hu, pow_succ]; exact Nat.mul_dvd_mul_left _ h)
  have hunit : IsUnit (u : ZMod (l ^ K)) :=
    (ZMod.isUnit_iff_coprime u (l ^ K)).mpr (Nat.Coprime.pow_right _ ((Nat.Prime.coprime_iff_not_dvd hl).mpr hlu).symm)
  constructor
  · intro h
    have := congrFun h ⟨q.2.1, hj⟩
    simp only [Pi.smul_apply, vecK, hu, smul_eq_mul, Pi.zero_apply] at this
    push_cast at this
    have h2 : (a * (l : ZMod (l ^ K)) ^ q.1) * u = 0 := by rw [← this]; ring
    rw [hunit.mul_left_eq_zero] at h2
    rw [mul_comm]; exact h2
  · intro h
    funext i
    obtain ⟨c, hc⟩ := hdv i i.2
    simp only [Pi.smul_apply, vecK, hc, smul_eq_mul, Pi.zero_apply]
    push_cast
    rw [← mul_assoc, mul_comm a, h, zero_mul]

/-- The span of one pivot row of valuation `v` has `ℓ^{K − v}` elements. -/
theorem card_span_pivot (hl : l.Prime) {q : Piv} (hq : PivProp l K H w q) :
    Nat.card (Submodule.span (ZMod (l ^ K)) {vecK l K H w q.2.2}) = l ^ (K - q.1) := by
  have hvK := hq.1
  have : NeZero (l ^ K) := ⟨(pow_pos hl.pos K).ne'⟩
  set m := l ^ (K - q.1) with hm
  have hlK : l ^ K = l ^ q.1 * m := by rw [hm, ← pow_add]; congr 1; omega
  have hann := smul_pivot_eq_zero hl hq
  have hzero : ∀ t : ℕ, ((m * t : ℕ) : ZMod (l ^ K)) • vecK l K H w q.2.2 = 0 := fun t => by
    rw [hann]
    have : ((l ^ q.1 * (m * t) : ℕ) : ZMod (l ^ K)) = 0 := by
      rw [← mul_assoc, ← hlK, Nat.cast_mul, ZMod.natCast_self, zero_mul]
    push_cast at this ⊢; exact this
  let f : Fin m → Submodule.span (ZMod (l ^ K)) {vecK l K H w q.2.2} := fun t =>
    ⟨((t : ℕ) : ZMod (l ^ K)) • vecK l K H w q.2.2, Submodule.smul_mem _ _ (Submodule.mem_span_singleton_self _)⟩
  have hf : Function.Bijective f := by
    constructor
    · intro t1 t2 h
      have h' : ((t1 : ℕ) : ZMod (l ^ K)) • vecK l K H w q.2.2 = ((t2 : ℕ) : ZMod (l ^ K)) • vecK l K H w q.2.2 :=
        congrArg Subtype.val h
      wlog hle : (t1 : ℕ) ≤ t2 generalizing t1 t2
      · exact (this h.symm h'.symm (by omega)).symm
      have hd : (((t2 : ℕ) - t1 : ℕ) : ZMod (l ^ K)) • vecK l K H w q.2.2 = 0 := by
        rw [Nat.cast_sub hle, sub_smul, h', sub_self]
      rw [hann] at hd
      have hd' : ((l ^ q.1 * ((t2 : ℕ) - t1) : ℕ) : ZMod (l ^ K)) = 0 := by push_cast; exact hd
      rw [ZMod.natCast_eq_zero_iff, hlK] at hd'
      have hdvd := Nat.dvd_of_mul_dvd_mul_left (pow_pos hl.pos _) hd'
      have : (t2 : ℕ) - t1 = 0 := Nat.eq_zero_of_dvd_of_lt hdvd (by have := t2.2; omega)
      exact Fin.ext (by omega)
    · rintro ⟨x, hx⟩
      obtain ⟨a, rfl⟩ := Submodule.mem_span_singleton.mp hx
      refine ⟨⟨a.val % m, Nat.mod_lt _ (pow_pos hl.pos _)⟩, Subtype.ext ?_⟩
      show ((a.val % m : ℕ) : ZMod (l ^ K)) • vecK l K H w q.2.2 = a • vecK l K H w q.2.2
      have ha : a = ((a.val % m : ℕ) : ZMod (l ^ K)) + ((m * (a.val / m) : ℕ) : ZMod (l ^ K)) := by
        rw [← Nat.cast_add, Nat.mod_add_div, ZMod.natCast_zmod_val]
      conv_rhs => rw [ha]
      rw [add_smul, hzero, add_zero]
  rw [← Nat.card_eq_of_bijective f hf, Nat.card_eq_fintype_card, Fintype.card_fin]

/-- Cardinality of a sum of two independent submodules. -/
theorem card_sup_of_inf_eq_bot {R M : Type} [Ring R] [AddCommGroup M] [Module R M] {A B : Submodule R M}
    (h : A ⊓ B = ⊥) : Nat.card (A ⊔ B : Submodule R M) = Nat.card A * Nat.card B := by
  rw [← Nat.card_prod]
  let f : A × B → (A ⊔ B : Submodule R M) := fun x =>
    ⟨x.1 + x.2, Submodule.add_mem _ (Submodule.mem_sup_left x.1.2) (Submodule.mem_sup_right x.2.2)⟩
  refine (Nat.card_eq_of_bijective f ⟨fun x y hxy => ?_, fun z => ?_⟩).symm
  · have e : (x.1 : M) + x.2 = y.1 + y.2 := congrArg Subtype.val hxy
    have h1 : (x.1 : M) - y.1 ∈ A ⊓ B := ⟨sub_mem x.1.2 y.1.2, by
      rw [show (x.1 : M) - y.1 = y.2 - x.2 by rw [sub_eq_sub_iff_add_eq_add, e, add_comm]]
      exact sub_mem y.2.2 x.2.2⟩
    rw [h, Submodule.mem_bot, sub_eq_zero] at h1
    have h2 : (x.2 : M) = y.2 := by rw [h1] at e; exact add_left_cancel e
    exact Prod.ext (Subtype.ext h1) (Subtype.ext h2)
  · obtain ⟨a, ha, b, hb, hab⟩ := Submodule.mem_sup.mp z.2
    exact ⟨(⟨a, ha⟩, ⟨b, hb⟩), Subtype.ext hab⟩

/-- **The span of an echelon family** with valuations `v_k` has `ℓ^{Σ (K − v_k)}` elements. -/
theorem card_spanK_ech (hl : l.Prime) : ∀ C : List Piv, Ech l K H w C →
    Nat.card (spanK l K H w (rowsOf C)) = l ^ (C.map fun q => K - q.1).sum
  | [], _ => by
    rw [show rowsOf [] = [] from rfl, spanK_eq_bot (by simp)]
    simp
  | q :: C, ⟨hC, hq, hz⟩ => by
    rw [show rowsOf (q :: C) = q.2.2 :: rowsOf C from rfl, spanK_cons, List.map_cons, List.sum_cons, pow_add,
      ← card_span_pivot hl hq, ← card_spanK_ech hl C hC]
    apply card_sup_of_inf_eq_bot
    rw [eq_bot_iff]
    rintro x ⟨hx1, hx2⟩
    obtain ⟨a, rfl⟩ := Submodule.mem_span_singleton.mp hx1
    rw [Submodule.mem_bot, smul_pivot_eq_zero hl hq]
    -- the column `j` of `q` separates
    have hvan : ∀ y ∈ spanK l K H w (rowsOf C), y ⟨q.2.1, hq.2.1⟩ = 0 := by
      intro y hy
      have : spanK l K H w (rowsOf C) ≤ LinearMap.ker (LinearMap.proj (R := ZMod (l ^ K))
          (φ := fun _ : Fin w => ZMod (l ^ K)) ⟨q.2.1, hq.2.1⟩) := by
        rw [spanK, Submodule.span_le]
        rintro _ ⟨r, hr, rfl⟩
        obtain ⟨q', hq', rfl⟩ := List.mem_map.mp hr
        simp only [SetLike.mem_coe, LinearMap.mem_ker, LinearMap.proj_apply, vecK]
        rw [hz q' hq', Nat.cast_zero]
      exact this hy
    have h0 := hvan _ hx2
    obtain ⟨_, hj, _, hdv, hnd⟩ := hq
    obtain ⟨u, hu⟩ := hdv _ hj
    have hlu : ¬ l ∣ u := fun h => hnd (by rw [hu, pow_succ]; exact Nat.mul_dvd_mul_left _ h)
    have hunit : IsUnit (u : ZMod (l ^ K)) :=
      (ZMod.isUnit_iff_coprime u (l ^ K)).mpr (Nat.Coprime.pow_right _ ((Nat.Prime.coprime_iff_not_dvd hl).mpr hlu).symm)
    simp only [Pi.smul_apply, vecK, hu, smul_eq_mul] at h0
    push_cast at h0
    have h2 : (a * (l : ZMod (l ^ K)) ^ q.1) * u = 0 := by rw [← h0]; ring
    rw [hunit.mul_left_eq_zero] at h2
    rw [mul_comm]; exact h2

/-- **The order of the row span**: the row span over `ZMod (ℓ^K)` has `ℓ^{Σ (K − v)}` elements, the sum
over the pivots `(v, j, p)` that `smithPK` returns (with the span equal to that of the pivot rows, `smithPK_spec`). -/
theorem smithPK_card {rows : List ℕ} (cx : Ctx l K H rows.length)
    (hrows : ∀ r ∈ rows, ∀ i < w, fld (2 * H) r i < l ^ K) :
    Nat.card (spanK l K H w rows) = l ^ ((smithPK l K H w rows).map fun q => K - q.1).sum := by
  obtain ⟨h1, h2⟩ := smithPK_spec (w := w) cx hrows
  rw [h2, spanK_congr (L' := rowsOf (smithPK l K H w rows).reverse) (fun r => by simp [rowsOf]),
    card_spanK_ech cx.prime _ (pivOK_ech _ h1), List.map_reverse, List.sum_reverse]

lemma lenOf_eq (K : ℕ) : ∀ P : List Piv, lenOf K P = (P.map fun q => K - q.1).sum
  | [] => rfl
  | q :: P => by
    show K - q.1 + lenOf K P = _
    rw [lenOf_eq K P, List.map_cons, List.sum_cons]

/-- **The cell form**: a kernel-decided signature `sig K (smithPK …) = (s, cs)` gives `#span = ℓ^s`. -/
theorem card_of_sig {rows : List ℕ} (cx : Ctx l K H rows.length)
    (hrows : ∀ r ∈ rows, ∀ i < w, fld (2 * H) r i < l ^ K) {s : ℕ} {cs : List ℕ}
    (h : sig K (smithPK l K H w rows) = (s, cs)) : Nat.card (spanK l K H w rows) = l ^ s := by
  rw [smithPK_card cx hrows, ← lenOf_eq, show lenOf K (smithPK l K H w rows) = s from congrArg Prod.fst h]

end Card

end GL2.L23.SmithPK

end
