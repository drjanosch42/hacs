import Mathlib.LinearAlgebra.Matrix.Adjugate
import Mathlib.Algebra.BigOperators.Finsupp.Basic
import Mathlib.Algebra.Order.Field.Rat
import Mathlib.Data.Int.GCD
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.LinearCombination
import Mathlib.Tactic.Bound

/-!
# M4-i (slot 20, L20-4): Merel's set and its path condition ([C] `lem:merelpath`)

* `merelList n` / `merelSet n`: `Mer_n = {!![a, b; c, d] : ad - bc = n, a > b ≥ 0, d > c ≥ 0}`, enumerated in the
  C++ order (`a`, `d`, `b`, `c` ascending; `hecke::level::MerelSet`), with `mem_merelSet` (the defining conditions).
* `merel_path`: for every `η` of determinant `n ≥ 1`, `∑_{ε ∈ Mer_n ∩ ηΓ} ε·D₀ = D₀` in the free abelian group on the
  cusps (`Option ℚ`, `none = ∞`), `D₀ = [∞] - [0]`, `ε·D₀ = [ε∞] - [ε0]`.

The proof is not [C]'s chain argument but its bookkeeping in one step: the successor of (ii)
(`ε ↦ ε·!![0, -1; 1, ⌈a/b⌉]`, defined when `b > 0`) is a bijection from the elements with `b > 0` onto the elements with
`c > 0`, inverse to the predecessor of (iii) (`ε ↦ ε·!![⌈d/c⌉, 1; -1, 0]`), and it carries heads to tails; the element
with `c = 0` (tail `[∞]`) and the element with `b = 0` (head `[0]`) each exist and are unique in every coset (the two
Hermite forms, by Bézout).  So the tails and the heads cancel except `[∞] - [0]`.  [C]'s bound
`|Mer_n ∩ ηΓ| ≤ x₀(ηℤ²)` is not stated here.
-/

namespace GL2.M4

open Matrix

/-- Integer `2 × 2` matrices. -/
abbrev M2Z := Matrix (Fin 2) (Fin 2) ℤ

/-- Merel's conditions ([C] `lem:merelpath`). -/
def IsMerel (n : ℤ) (ε : M2Z) : Prop :=
  ε.det = n ∧ ε 0 1 < ε 0 0 ∧ 0 ≤ ε 0 1 ∧ ε 1 0 < ε 1 1 ∧ 0 ≤ ε 1 0

/-- A matrix from its entries `(a, b, c, d)`. -/
def toM2 (e : ℤ × ℤ × ℤ × ℤ) : M2Z := !![e.1, e.2.1; e.2.2.1, e.2.2.2]

/-- **`Mer_n`, executable**, in the order `a, d, b, c` ascending (the C++ `MerelSet`'s). -/
def merelList (n : ℕ) : List (ℤ × ℤ × ℤ × ℤ) :=
  (List.range n).flatMap fun (a' : ℕ) => (List.range n).flatMap fun (d' : ℕ) =>
    (List.range (a' + 1)).flatMap fun (b : ℕ) => (List.range (d' + 1)).filterMap fun (c : ℕ) =>
      if ((a' : ℤ) + 1) * ((d' : ℤ) + 1) - (b : ℤ) * (c : ℤ) = n then
        some ((a' : ℤ) + 1, (b : ℤ), (c : ℤ), (d' : ℤ) + 1) else none

theorem mem_merelList (n : ℕ) (e : ℤ × ℤ × ℤ × ℤ) :
    e ∈ merelList n ↔ e.1 * e.2.2.2 - e.2.1 * e.2.2.1 = n ∧ e.2.1 < e.1 ∧ 0 ≤ e.2.1 ∧
      e.2.2.1 < e.2.2.2 ∧ 0 ≤ e.2.2.1 := by
  obtain ⟨a, b, c, d⟩ := e
  simp only [merelList, List.mem_flatMap, List.mem_range, List.mem_filterMap, Option.ite_none_right_eq_some,
    Option.some.injEq, Prod.mk.injEq]
  constructor
  · rintro ⟨a', ha, d', hd, b', hb, c', hc, hdet, rfl, rfl, rfl, rfl⟩
    refine ⟨hdet, by omega, by omega, by omega, by omega⟩
  · rintro ⟨hdet, hba, hb, hcd, hc⟩
    have ha1 : 1 ≤ a := by omega
    have hd1 : 1 ≤ d := by omega
    -- ad - bc ≥ ad - (a - 1)(d - 1) = a + d - 1
    have hbc : b * c ≤ (a - 1) * (d - 1) := mul_le_mul (by omega) (by omega) hc (by omega)
    have hn : a + d - 1 ≤ n := by nlinarith
    refine ⟨(a - 1).toNat, by omega, (d - 1).toNat, by omega, b.toNat, by omega, c.toNat, by omega, ?_, ?_⟩
    · rw [Int.toNat_of_nonneg (by omega), Int.toNat_of_nonneg (by omega), Int.toNat_of_nonneg hb,
        Int.toNat_of_nonneg hc]
      simpa using hdet
    · refine ⟨by omega, by omega, by omega, by omega⟩

/-- **`Mer_n`** as a finite set of integer matrices. -/
def merelSet (n : ℕ) : Finset M2Z := ((merelList n).map toM2).toFinset

theorem toM2_apply (e : ℤ × ℤ × ℤ × ℤ) :
    toM2 e 0 0 = e.1 ∧ toM2 e 0 1 = e.2.1 ∧ toM2 e 1 0 = e.2.2.1 ∧ toM2 e 1 1 = e.2.2.2 := by
  simp [toM2]

theorem det_eq (ε : M2Z) : ε.det = ε 0 0 * ε 1 1 - ε 0 1 * ε 1 0 := Matrix.det_fin_two ε

/-- **`mem_merelSet`**: `ε ∈ Mer_n` iff Merel's conditions hold. -/
theorem mem_merelSet (n : ℕ) (ε : M2Z) : ε ∈ merelSet n ↔ IsMerel n ε := by
  simp only [merelSet, List.mem_toFinset, List.mem_map]
  constructor
  · rintro ⟨e, he, rfl⟩
    rw [mem_merelList] at he
    obtain ⟨h1, h2, h3, h4⟩ := toM2_apply e
    refine ⟨?_, ?_⟩ <;> simp only [det_eq, h1, h2, h3, h4] <;> first | exact he.1 | exact he.2
  · rintro ⟨hdet, h⟩
    refine ⟨(ε 0 0, ε 0 1, ε 1 0, ε 1 1), ?_, ?_⟩
    · rw [mem_merelList]; rw [det_eq] at hdet; exact ⟨hdet, h⟩
    · ext i j; fin_cases i <;> fin_cases j <;> rfl

/-! ### Cusps, `D₀`, the coset -/

/-- The cusp of a nonzero column `(x, y)`: `none = ∞` when `y = 0`, else `x / y`. -/
def cuspOf (x y : ℤ) : Option ℚ := if y = 0 then none else some ((x : ℚ) / y)

/-- `ε·D₀ = [ε∞] - [ε0]`: tail (first column) minus head (second column). -/
noncomputable def epsD (ε : M2Z) : Option ℚ →₀ ℤ :=
  Finsupp.single (cuspOf (ε 0 0) (ε 1 0)) 1 - Finsupp.single (cuspOf (ε 0 1) (ε 1 1)) 1

/-- `D₀ = [∞] - [0]`. -/
noncomputable def D0 : Option ℚ →₀ ℤ := Finsupp.single none 1 - Finsupp.single (some 0) 1

/-- `ε ∈ ηΓ`, `Γ = SL₂(ℤ)`. -/
def InCoset (η ε : M2Z) : Prop := ∃ γ : M2Z, γ.det = 1 ∧ ε = η * γ

lemma inCoset_mul {η ε g : M2Z} (h : InCoset η ε) (hg : g.det = 1) : InCoset η (ε * g) := by
  obtain ⟨γ, hγ, rfl⟩ := h
  exact ⟨γ * g, by rw [Matrix.det_mul, hγ, hg, one_mul], by rw [Matrix.mul_assoc]⟩

/-- Two elements of one coset differ by an element of `SL₂(ℤ)` on the right. -/
lemma inCoset_diff {η ε ε' : M2Z} (h : InCoset η ε) (h' : InCoset η ε') :
    ∃ δ : M2Z, δ.det = 1 ∧ ε' = ε * δ := by
  obtain ⟨γ, hγ, rfl⟩ := h
  obtain ⟨γ', hγ', rfl⟩ := h'
  refine ⟨γ.adjugate * γ', ?_, ?_⟩
  · rw [Matrix.det_mul, Matrix.det_adjugate, hγ, hγ']; simp
  · rw [Matrix.mul_assoc, ← Matrix.mul_assoc γ, Matrix.mul_adjugate, hγ]; simp

lemma mul_entries (ε g : M2Z) :
    (ε * g) 0 0 = ε 0 0 * g 0 0 + ε 0 1 * g 1 0 ∧ (ε * g) 0 1 = ε 0 0 * g 0 1 + ε 0 1 * g 1 1 ∧
    (ε * g) 1 0 = ε 1 0 * g 0 0 + ε 1 1 * g 1 0 ∧ (ε * g) 1 1 = ε 1 0 * g 0 1 + ε 1 1 * g 1 1 := by
  simp [Matrix.mul_apply, Fin.sum_univ_two]

lemma m2_ext {A B : M2Z} (h00 : A 0 0 = B 0 0) (h01 : A 0 1 = B 0 1) (h10 : A 1 0 = B 1 0)
    (h11 : A 1 1 = B 1 1) : A = B := by
  ext i j; fin_cases i <;> fin_cases j <;> assumption

/-! ### Ceilings -/

/-- `⌈a / b⌉` for `b > 0`. -/
def cdiv (a b : ℤ) : ℤ := (a + b - 1) / b

lemma cdiv_spec {a b : ℤ} (hb : 0 < b) : a ≤ cdiv a b * b ∧ cdiv a b * b - a < b := by
  have h := Int.mul_ediv_add_emod (a + b - 1) b
  have h1 := Int.emod_nonneg (a + b - 1) hb.ne'
  have h2 := Int.emod_lt_of_pos (a + b - 1) hb
  unfold cdiv
  constructor <;> nlinarith

lemma cdiv_eq {a b k : ℤ} (hb : 0 < b) (h1 : a ≤ k * b) (h2 : k * b - a < b) : cdiv a b = k := by
  obtain ⟨g1, g2⟩ := cdiv_spec (a := a) hb
  have e1 : (cdiv a b - k) * b < b := by nlinarith
  have e2 : -b < (cdiv a b - k) * b := by nlinarith
  have f1 : cdiv a b - k < 1 := by
    by_contra hc; push Not at hc; nlinarith
  have f2 : -1 < cdiv a b - k := by
    by_contra hc; push Not at hc; nlinarith
  omega

/-! ### Successor and predecessor -/

/-- The successor of (ii): `(v₁ | v₂) ↦ (v₂ | m v₂ - v₁)`, `m = ⌈a / b⌉`. -/
def succM (ε : M2Z) : M2Z := ε * !![0, -1; 1, cdiv (ε 0 0) (ε 0 1)]

/-- The predecessor of (iii): `(v₁ | v₂) ↦ (k v₁ - v₂ | v₁)`, `k = ⌈d / c⌉`. -/
def predM (ε : M2Z) : M2Z := ε * !![cdiv (ε 1 1) (ε 1 0), 1; -1, 0]

lemma succM_entries (ε : M2Z) :
    succM ε 0 0 = ε 0 1 ∧ succM ε 0 1 = cdiv (ε 0 0) (ε 0 1) * ε 0 1 - ε 0 0 ∧
    succM ε 1 0 = ε 1 1 ∧ succM ε 1 1 = cdiv (ε 0 0) (ε 0 1) * ε 1 1 - ε 1 0 := by
  simp [succM, Matrix.mul_apply, Fin.sum_univ_two]; constructor <;> ring

lemma predM_entries (ε : M2Z) :
    predM ε 0 0 = cdiv (ε 1 1) (ε 1 0) * ε 0 0 - ε 0 1 ∧ predM ε 0 1 = ε 0 0 ∧
    predM ε 1 0 = cdiv (ε 1 1) (ε 1 0) * ε 1 0 - ε 1 1 ∧ predM ε 1 1 = ε 1 0 := by
  simp [predM, Matrix.mul_apply, Fin.sum_univ_two]; constructor <;> ring

lemma det_succ_mat (m : ℤ) : (!![0, -1; 1, m] : M2Z).det = 1 := by simp [Matrix.det_fin_two]
lemma det_pred_mat (k : ℤ) : (!![k, 1; -1, 0] : M2Z).det = 1 := by simp [Matrix.det_fin_two]

lemma succM_merel {n : ℤ} {ε : M2Z} (h : IsMerel n ε) (hb : 0 < ε 0 1) : IsMerel n (succM ε) ∧ 0 < succM ε 1 0 := by
  obtain ⟨hdet, h1, h2, h3, h4⟩ := h
  obtain ⟨s1, s2, s3, s4⟩ := succM_entries ε
  obtain ⟨c1, c2⟩ := cdiv_spec (a := ε 0 0) hb
  set m := cdiv (ε 0 0) (ε 0 1)
  have hm : 2 ≤ m := by
    by_contra hc; push Not at hc; nlinarith
  refine ⟨⟨?_, ?_, ?_, ?_, ?_⟩, ?_⟩
  · rw [succM, Matrix.det_mul, hdet, det_succ_mat, mul_one]
  · rw [s1, s2]; linarith
  · rw [s2]; linarith
  · rw [s3, s4]; nlinarith
  · rw [s3]; linarith
  · rw [s3]; linarith

lemma predM_merel {n : ℤ} {ε : M2Z} (h : IsMerel n ε) (hc : 0 < ε 1 0) : IsMerel n (predM ε) ∧ 0 < predM ε 0 1 := by
  obtain ⟨hdet, h1, h2, h3, h4⟩ := h
  obtain ⟨s1, s2, s3, s4⟩ := predM_entries ε
  obtain ⟨c1, c2⟩ := cdiv_spec (a := ε 1 1) hc
  set k := cdiv (ε 1 1) (ε 1 0)
  have hk : 2 ≤ k := by
    by_contra h'; push Not at h'; nlinarith
  refine ⟨⟨?_, ?_, ?_, ?_, ?_⟩, ?_⟩
  · rw [predM, Matrix.det_mul, hdet, det_pred_mat, mul_one]
  · rw [s1, s2]; nlinarith
  · rw [s2]; linarith
  · rw [s3, s4]; linarith
  · rw [s3]; linarith
  · rw [s2]; linarith

lemma pred_succ {n : ℤ} {ε : M2Z} (h : IsMerel n ε) (hb : 0 < ε 0 1) : predM (succM ε) = ε := by
  obtain ⟨_, h1, h2, h3, h4⟩ := h
  obtain ⟨s1, s2, s3, s4⟩ := succM_entries ε
  obtain ⟨c1, c2⟩ := cdiv_spec (a := ε 0 0) hb
  have hd : 0 < ε 1 1 := by linarith
  set m := cdiv (ε 0 0) (ε 0 1)
  have hk : cdiv (succM ε 1 1) (succM ε 1 0) = m := by
    rw [s3, s4]; exact cdiv_eq hd (by linarith) (by linarith)
  obtain ⟨p1, p2, p3, p4⟩ := predM_entries (succM ε)
  rw [hk] at p1 p3
  refine m2_ext ?_ ?_ ?_ ?_
  · rw [p1, s1, s2]; ring
  · rw [p2, s1]
  · rw [p3, s3, s4]; ring
  · rw [p4, s3]

lemma succ_pred {n : ℤ} {ε : M2Z} (h : IsMerel n ε) (hc : 0 < ε 1 0) : succM (predM ε) = ε := by
  obtain ⟨_, h1, h2, h3, h4⟩ := h
  obtain ⟨s1, s2, s3, s4⟩ := predM_entries ε
  obtain ⟨c1, c2⟩ := cdiv_spec (a := ε 1 1) hc
  have ha : 0 < ε 0 0 := by linarith
  set k := cdiv (ε 1 1) (ε 1 0)
  have hm : cdiv (predM ε 0 0) (predM ε 0 1) = k := by
    rw [s1, s2]; exact cdiv_eq ha (by linarith) (by linarith)
  obtain ⟨p1, p2, p3, p4⟩ := succM_entries (predM ε)
  rw [hm] at p2 p4
  refine m2_ext ?_ ?_ ?_ ?_
  · rw [p1, s2]
  · rw [p2, s1, s2]; ring
  · rw [p3, s4]
  · rw [p4, s3, s4]; ring

/-! ### The two Hermite forms: existence and uniqueness in a coset -/

@[simp] lemma mk00 (a b c d : ℤ) : (!![a, b; c, d] : M2Z) 0 0 = a := rfl
@[simp] lemma mk01 (a b c d : ℤ) : (!![a, b; c, d] : M2Z) 0 1 = b := rfl
@[simp] lemma mk10 (a b c d : ℤ) : (!![a, b; c, d] : M2Z) 1 0 = c := rfl
@[simp] lemma mk11 (a b c d : ℤ) : (!![a, b; c, d] : M2Z) 1 1 = d := rfl

lemma eta2 (A : M2Z) : ∃ a b c d : ℤ, A = !![a, b; c, d] := ⟨_, _, _, _, Matrix.eta_fin_two A⟩

lemma int_unit_of_mul {x y : ℤ} (h : x * y = 1) (hx : 0 < x) : x = 1 ∧ y = 1 := by
  have h1 := Int.eq_one_of_mul_eq_one_right hx.le h
  subst h1; exact ⟨rfl, by simpa using h⟩

lemma zero_of_bounded {d c t : ℤ} (hd : 0 < d) (h0 : 0 ≤ c) (h1 : c < d) (h0' : 0 ≤ c + d * t)
    (h1' : c + d * t < d) : t = 0 := by
  rcases lt_trichotomy t 0 with hl | hl | hl
  · have : d * t ≤ -d := by nlinarith
    linarith
  · exact hl
  · have : d ≤ d * t := by nlinarith
    linarith

set_option linter.unusedSimpArgs false in
/-- At most one element with `c = 0` in a coset. -/
lemma upper_unique {n : ℤ} {η ε ε' : M2Z} (h : IsMerel n ε) (h' : IsMerel n ε') (hc : ε 1 0 = 0)
    (hc' : ε' 1 0 = 0) (hη : InCoset η ε) (hη' : InCoset η ε') : ε' = ε := by
  obtain ⟨δ, hδ, rfl⟩ := inCoset_diff hη hη'
  obtain ⟨a, b, c, d, rfl⟩ := eta2 ε
  obtain ⟨x, y, z, w, rfl⟩ := eta2 δ
  obtain ⟨_, h1, h2, h3, h4⟩ := h
  obtain ⟨_, h1', h2', h3', h4'⟩ := h'
  simp only [Matrix.mul_fin_two, Matrix.det_fin_two_of, mk00, mk01, mk10, mk11, mul_zero, zero_mul, add_zero,
    zero_add, sub_zero, mul_one, one_mul] at *
  subst hc
  have hd : 0 < d := by linarith
  have hz : z = 0 := by
    have : d * z = 0 := by linarith
    rcases mul_eq_zero.mp this with h | h
    · omega
    · exact h
  subst hz
  have ha : 0 < a := by linarith
  have hx : 0 < x := by
    have : 0 < a * x := by linarith
    exact pos_of_mul_pos_right this ha.le
  obtain ⟨rfl, rfl⟩ := int_unit_of_mul (x := x) (y := w) (by linarith) hx
  have hy : y = 0 := zero_of_bounded ha h2 h1 (by linarith) (by linarith)
  subst hy
  simp

set_option linter.unusedSimpArgs false in
/-- At most one element with `b = 0` in a coset. -/
lemma lower_unique {n : ℤ} {η ε ε' : M2Z} (h : IsMerel n ε) (h' : IsMerel n ε') (hb : ε 0 1 = 0)
    (hb' : ε' 0 1 = 0) (hη : InCoset η ε) (hη' : InCoset η ε') : ε' = ε := by
  obtain ⟨δ, hδ, rfl⟩ := inCoset_diff hη hη'
  obtain ⟨a, b, c, d, rfl⟩ := eta2 ε
  obtain ⟨x, y, z, w, rfl⟩ := eta2 δ
  obtain ⟨_, h1, h2, h3, h4⟩ := h
  obtain ⟨_, h1', h2', h3', h4'⟩ := h'
  simp only [Matrix.mul_fin_two, Matrix.det_fin_two_of, mk00, mk01, mk10, mk11, mul_zero, zero_mul, add_zero,
    zero_add, sub_zero, mul_one, one_mul] at *
  subst hb
  have ha : 0 < a := by linarith
  have hy : y = 0 := by
    have : a * y = 0 := by linarith
    rcases mul_eq_zero.mp this with h | h
    · omega
    · exact h
  subst hy
  have hx : 0 < x := by
    have : 0 < a * x := by linarith
    exact pos_of_mul_pos_right this ha.le
  obtain ⟨rfl, rfl⟩ := int_unit_of_mul (x := x) (y := w) (by linarith) hx
  have hd : 0 < d := by linarith
  have hz : z = 0 := zero_of_bounded hd h4 h3 (by linarith) (by linarith)
  subst hz
  simp

lemma mul_mk (η : M2Z) (x y z w : ℤ) : η * !![x, y; z, w] =
    !![η 0 0 * x + η 0 1 * z, η 0 0 * y + η 0 1 * w; η 1 0 * x + η 1 1 * z, η 1 0 * y + η 1 1 * w] := by
  obtain ⟨a, b, c, d, rfl⟩ := eta2 η
  simp

/-- The upper Hermite form (`c = 0`) exists in every coset of determinant `n ≥ 1`. -/
lemma upper_exists {n : ℤ} (hn : 0 < n) {η : M2Z} (hη : η.det = n) :
    ∃ ε, IsMerel n ε ∧ ε 1 0 = 0 ∧ InCoset η ε := by
  obtain ⟨p, q, r, s, rfl⟩ := eta2 η
  rw [Matrix.det_fin_two_of] at hη
  have hg0 : Int.gcd r s ≠ 0 := by
    intro h0
    obtain ⟨h1, h2⟩ := Int.gcd_eq_zero_iff.mp h0
    rw [h1, h2] at hη; simp at hη; omega
  have hbez := Int.gcd_eq_gcd_ab r s
  obtain ⟨r', hr⟩ := Int.gcd_dvd_left r s
  obtain ⟨s', hs⟩ := Int.gcd_dvd_right r s
  have hgpos : (0 : ℤ) < (Int.gcd r s : ℤ) := by exact_mod_cast Nat.pos_of_ne_zero hg0
  generalize Int.gcdA r s = x at hbez
  generalize Int.gcdB r s = y at hbez
  generalize (Int.gcd r s : ℤ) = g at hbez hr hs hgpos
  have hdetγ : s' * y + r' * x = 1 := by
    have : g * (s' * y + r' * x) = g * 1 := by rw [hr, hs] at hbez; linear_combination -hbez
    exact mul_left_cancel₀ hgpos.ne' this
  set A := p * s' + q * -r'
  set B := p * x + q * y
  have hAg : A * g = n := by
    rw [← hη, hr, hs]; simp only [A]; ring
  have hA : 0 < A := pos_of_mul_pos_left (by rw [hAg]; exact hn) hgpos.le
  set t := B / A
  refine ⟨!![p, q; r, s] * !![s', x - s' * t; -r', y + r' * t], ⟨?_, ?_, ?_, ?_, ?_⟩, ?_,
    ⟨_, by rw [Matrix.det_fin_two_of]; linear_combination hdetγ, rfl⟩⟩
  · rw [Matrix.det_mul, Matrix.det_fin_two_of, Matrix.det_fin_two_of, hη]; linear_combination n * hdetγ
  all_goals simp only [mul_mk, mk00, mk01, mk10, mk11]
  · have h1 := Int.emod_lt_of_pos B hA
    rw [Int.emod_def] at h1
    simp only [B, A, t] at h1 ⊢; linarith
  · have h1 := Int.emod_nonneg B hA.ne'
    rw [Int.emod_def] at h1
    simp only [B, A, t] at h1 ⊢; linarith
  · rw [hr, hs]; nlinarith
  · rw [hr, hs]; ring_nf; exact le_refl _
  · rw [hr, hs]; ring

/-- The lower Hermite form (`b = 0`) exists in every coset of determinant `n ≥ 1`. -/
lemma lower_exists {n : ℤ} (hn : 0 < n) {η : M2Z} (hη : η.det = n) :
    ∃ ε, IsMerel n ε ∧ ε 0 1 = 0 ∧ InCoset η ε := by
  obtain ⟨p, q, r, s, rfl⟩ := eta2 η
  rw [Matrix.det_fin_two_of] at hη
  have hg0 : Int.gcd p q ≠ 0 := by
    intro h0
    obtain ⟨h1, h2⟩ := Int.gcd_eq_zero_iff.mp h0
    rw [h1, h2] at hη; simp at hη; omega
  have hbez := Int.gcd_eq_gcd_ab p q
  obtain ⟨p', hp⟩ := Int.gcd_dvd_left p q
  obtain ⟨q', hq⟩ := Int.gcd_dvd_right p q
  have hgpos : (0 : ℤ) < (Int.gcd p q : ℤ) := by exact_mod_cast Nat.pos_of_ne_zero hg0
  generalize Int.gcdA p q = x at hbez
  generalize Int.gcdB p q = y at hbez
  generalize (Int.gcd p q : ℤ) = g at hbez hp hq hgpos
  have hdetγ : x * p' + q' * y = 1 := by
    have : g * (x * p' + q' * y) = g * 1 := by rw [hp, hq] at hbez; linear_combination -hbez
    exact mul_left_cancel₀ hgpos.ne' this
  set C := r * x + s * y
  set D := -(r * q') + s * p'
  have hgD : g * D = n := by
    rw [← hη, hp, hq]; simp only [D]; ring
  have hD : 0 < D := pos_of_mul_pos_right (by rw [hgD]; exact hn) hgpos.le
  set t := C / D
  refine ⟨!![p, q; r, s] * !![x + q' * t, -q'; y - p' * t, p'], ⟨?_, ?_, ?_, ?_, ?_⟩, ?_,
    ⟨_, by rw [Matrix.det_fin_two_of]; linear_combination hdetγ, rfl⟩⟩
  · rw [Matrix.det_mul, Matrix.det_fin_two_of, Matrix.det_fin_two_of, hη]; linear_combination n * hdetγ
  all_goals simp only [mul_mk, mk00, mk01, mk10, mk11]
  · rw [hp, hq]; nlinarith
  · rw [hp, hq]; ring_nf; exact le_refl _
  · have h1 := Int.emod_lt_of_pos C hD
    rw [Int.emod_def] at h1
    simp only [C, D, t] at h1 ⊢; linarith
  · have h1 := Int.emod_nonneg C hD.ne'
    rw [Int.emod_def] at h1
    simp only [C, D, t] at h1 ⊢; linarith
  · rw [hp, hq]; ring

/-! ### `lem:merelpath` -/

/-- `Mer_n ∩ ηΓ`. -/
noncomputable def merelCoset (n : ℕ) (η : M2Z) : Finset M2Z := by
  classical exact (merelSet n).filter (InCoset η)

/-- The statement of [C] `lem:merelpath` (the path condition; the bound `≤ x₀` is not stated). -/
def merel_path_stmt : Prop :=
  ∀ (n : ℕ), 0 < n → ∀ η : M2Z, η.det = n → ∑ ε ∈ merelCoset n η, epsD ε = D0

/-- **[C] `lem:merelpath`**: `∑_{ε ∈ Mer_n ∩ ηΓ} ε·D₀ = D₀` for every `η` of determinant `n ≥ 1`. -/
theorem merel_path : merel_path_stmt := by
  classical
  intro n hn η hη
  have hn' : (0 : ℤ) < n := by exact_mod_cast hn
  set B := merelCoset n η with hB
  have memB : ∀ ε, ε ∈ B ↔ IsMerel n ε ∧ InCoset η ε := by
    intro ε; simp only [hB, merelCoset, Finset.mem_filter, mem_merelSet]
  simp only [epsD, Finset.sum_sub_distrib]
  rw [← Finset.sum_filter_add_sum_filter_not B (fun ε => ε 1 0 = 0),
    ← Finset.sum_filter_add_sum_filter_not B (fun ε => ε 0 1 = 0)]
  have hT : ∑ ε ∈ B.filter (fun ε => ε 1 0 = 0), Finsupp.single (cuspOf (ε 0 0) (ε 1 0)) (1 : ℤ) =
      Finsupp.single none 1 := by
    obtain ⟨e, he, he0, heη⟩ := upper_exists hn' hη
    have : B.filter (fun ε => ε 1 0 = 0) = {e} := by
      ext ε; simp only [Finset.mem_filter, memB, Finset.mem_singleton]
      constructor
      · rintro ⟨⟨h1, h2⟩, h3⟩; exact upper_unique he h1 he0 h3 heη h2
      · rintro rfl; exact ⟨⟨he, heη⟩, he0⟩
    rw [this, Finset.sum_singleton, he0]; simp [cuspOf]
  have hH : ∑ ε ∈ B.filter (fun ε => ε 0 1 = 0), Finsupp.single (cuspOf (ε 0 1) (ε 1 1)) (1 : ℤ) =
      Finsupp.single (some 0) 1 := by
    obtain ⟨e, he, he0, heη⟩ := lower_exists hn' hη
    have : B.filter (fun ε => ε 0 1 = 0) = {e} := by
      ext ε; simp only [Finset.mem_filter, memB, Finset.mem_singleton]
      constructor
      · rintro ⟨⟨h1, h2⟩, h3⟩; exact lower_unique he h1 he0 h3 heη h2
      · rintro rfl; exact ⟨⟨he, heη⟩, he0⟩
    have hd : e 1 1 ≠ 0 := by have := he.2.2.2; omega
    rw [this, Finset.sum_singleton, he0]; simp [cuspOf, hd]
  have hM : ∑ ε ∈ B.filter (fun ε => ¬ ε 1 0 = 0), Finsupp.single (cuspOf (ε 0 0) (ε 1 0)) (1 : ℤ) =
      ∑ ε ∈ B.filter (fun ε => ¬ ε 0 1 = 0), Finsupp.single (cuspOf (ε 0 1) (ε 1 1)) (1 : ℤ) := by
    have pos_c : ∀ ε ∈ B.filter (fun ε => ¬ ε 1 0 = 0), IsMerel n ε ∧ InCoset η ε ∧ 0 < ε 1 0 := by
      intro ε hε
      simp only [Finset.mem_filter, memB] at hε
      exact ⟨hε.1.1, hε.1.2, lt_of_le_of_ne hε.1.1.2.2.2.2 (Ne.symm hε.2)⟩
    have pos_b : ∀ ε ∈ B.filter (fun ε => ¬ ε 0 1 = 0), IsMerel n ε ∧ InCoset η ε ∧ 0 < ε 0 1 := by
      intro ε hε
      simp only [Finset.mem_filter, memB] at hε
      exact ⟨hε.1.1, hε.1.2, lt_of_le_of_ne hε.1.1.2.2.1 (Ne.symm hε.2)⟩
    refine Finset.sum_nbij' predM succM ?_ ?_ ?_ ?_ ?_
    · intro ε hε
      obtain ⟨h1, h2, h3⟩ := pos_c ε hε
      obtain ⟨m1, m2⟩ := predM_merel h1 h3
      simp only [Finset.mem_filter, memB]
      exact ⟨⟨m1, inCoset_mul h2 (det_pred_mat _)⟩, m2.ne'⟩
    · intro ε hε
      obtain ⟨h1, h2, h3⟩ := pos_b ε hε
      obtain ⟨m1, m2⟩ := succM_merel h1 h3
      simp only [Finset.mem_filter, memB]
      exact ⟨⟨m1, inCoset_mul h2 (det_succ_mat _)⟩, m2.ne'⟩
    · intro ε hε
      obtain ⟨h1, _, h3⟩ := pos_c ε hε
      exact succ_pred h1 h3
    · intro ε hε
      obtain ⟨h1, _, h3⟩ := pos_b ε hε
      exact pred_succ h1 h3
    · intro ε _
      obtain ⟨_, p2, _, p4⟩ := predM_entries ε
      rw [p2, p4]
  rw [hT, hH, hM, D0]
  abel

end GL2.M4
