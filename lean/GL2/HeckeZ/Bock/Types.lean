/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import Mathlib

/-!
# HeckeZ 03 B2 (LB-2): mod-`ℓ^K` types and integral exponents (L-HZ2, M-HZ3)

`docs/implementation/slot24/heckez/03-bockstein-tor.md` §2.2 (B2), §8.1 (LB-2); `07-lean-plan.md` §2.3 (LB-2); the note after
[C] `prop:heckeZuct`.

Write `H^q ⊗ ℤ_(ℓ) ≅ ℤ_(ℓ)^{b_q} ⊕ ⊕_i ℤ/ℓ^{a_{q,i}}`. B2(1): `H^q(Γ; V/ℓ^K)` has the invariant-factor exponents
`uctType b_q (a_q) (a_{q+1}) K = [K]^{b_q} ++ [min(a_{q,i}, K)] ++ [min(a_{q+1,i}, K)]` (the universal-coefficient sequence of B1,
split as groups, with `ℤ/ℓ^a ⊗ ℤ/ℓ^K = (ℤ/ℓ^a)[ℓ^K] = ℤ/ℓ^{min(a,K)}`). Proved here:
* `count_uctType_lt` / `count_uctType_K`: the number of summands of exponent `j < K` is `μ_q(j) + μ_{q+1}(j)`, of exponent `K` it is
  `b_q + τ_q(K) + τ_{q+1}(K)`; `length_uctType_one`: at `K = 1` the `𝔽_ℓ`-dimension is `b_q + t_q + t_{q+1}` (K5's formula);
* `recursion`: B2(2) — the type at one level `K₀` and `μ_q` determine `μ_{q+1}(j)` for `j < K₀`;
* the cyclic pieces, as groups: `card_torsionBy_zmod` — `|(ℤ/ℓ^a)[ℓ^K]| = ℓ^{min(a,K)}`; `card_quotient_eq_card_ker` — for a finite
  abelian group `G`, `|G/NG| = |G[N]|` (so `|ℤ/ℓ^a ⊗ ℤ/ℓ^K| = ℓ^{min(a,K)}` as well).
-/

namespace GL2.HeckeZ.Bock

/-! ### The type function -/

/-- B2(1): the exponents of `H^q(Γ; V/ℓ^K)` from `b_q`, the exponents `a_q` of `H^q` and `a_{q+1}` of `H^{q+1}`. -/
def uctType (b : ℕ) (aq aq1 : List ℕ) (K : ℕ) : List ℕ :=
  List.replicate b K ++ aq.map (min · K) ++ aq1.map (min · K)

theorem count_map_min_lt {l : List ℕ} {j K : ℕ} (hj : j < K) : (l.map (min · K)).count j = l.count j := by
  induction l with
  | nil => simp
  | cons a l ih =>
    simp only [List.map_cons, List.count_cons, ih]
    congr 1
    by_cases ha : a < K
    · simp [Nat.min_eq_left ha.le]
    · have : min a K = K := Nat.min_eq_right (by omega)
      simp only [this]
      have h1 : (K == j) = false := by simp; omega
      have h2 : (a == j) = false := by simp; omega
      rw [h1, h2]

/-- Summands of exponent `j < K`: `μ_q(j) + μ_{q+1}(j)`. -/
theorem count_uctType_lt (b : ℕ) (aq aq1 : List ℕ) {j K : ℕ} (hj : j < K) :
    (uctType b aq aq1 K).count j = aq.count j + aq1.count j := by
  simp only [uctType, List.count_append, count_map_min_lt hj, List.count_replicate]
  have : (K == j) = false := by simp; omega
  simp [this]

theorem count_map_min_K {l : List ℕ} {K : ℕ} : (l.map (min · K)).count K = (l.filter (K ≤ ·)).length := by
  induction l with
  | nil => simp
  | cons a l ih =>
    simp only [List.map_cons, List.count_cons, ih, List.filter_cons]
    by_cases ha : K ≤ a
    · simp [Nat.min_eq_right ha, ha]
    · have : min a K = a := Nat.min_eq_left (by omega)
      simp [this, ha]; omega

/-- Summands of exponent `K`: `b_q + τ_q(K) + τ_{q+1}(K)` (an exponent `≥ K` reads as free: the uncertified level). -/
theorem count_uctType_K (b : ℕ) (aq aq1 : List ℕ) (K : ℕ) :
    (uctType b aq aq1 K).count K = b + (aq.filter (K ≤ ·)).length + (aq1.filter (K ≤ ·)).length := by
  simp [uctType, List.count_append, count_map_min_K, List.count_replicate_self]; omega

/-- **K5's formula**: at `K = 1` (all torsion exponents `≥ 1`) the `𝔽_ℓ`-dimension of `H^q(Γ; V/ℓ)` is `b_q + t_q + t_{q+1}`. -/
theorem length_uctType (b : ℕ) (aq aq1 : List ℕ) (K : ℕ) :
    (uctType b aq aq1 K).length = b + aq.length + aq1.length := by
  simp [uctType]; omega

/-- **B2(2), the recursion**: for `j < K₀`, `μ_{q+1}(j) = #{summands of H^q(V/ℓ^{K₀}) of exponent j} − μ_q(j)`. -/
theorem recursion (b : ℕ) (aq aq1 : List ℕ) {j K₀ : ℕ} (hj : j < K₀) :
    aq1.count j = (uctType b aq aq1 K₀).count j - aq.count j := by
  rw [count_uctType_lt b aq aq1 hj]; omega

/-! ### The cyclic pieces -/

/-- **`|(ℤ/ℓ^a)[ℓ^K]| = ℓ^{min(a, K)}`.** -/
theorem card_torsionBy_zmod {ℓ : ℕ} (hℓ : 2 ≤ ℓ) (a K : ℕ) :
    Nat.card {x : ZMod (ℓ ^ a) // ((ℓ ^ K : ℕ) : ZMod (ℓ ^ a)) * x = 0} = ℓ ^ min a K := by
  have : NeZero (ℓ ^ a) := ⟨pow_ne_zero _ (by omega)⟩
  set m := min a K
  have hm : m ≤ a := min_le_left a K
  have hpos : 0 < ℓ ^ (a - m) := pow_pos (by omega) _
  have hl0 : 0 < ℓ := by omega
  -- membership: `ℓ^K x = 0` iff `ℓ^{a-m} ∣ x.val`
  have hmem : ∀ x : ZMod (ℓ ^ a), ((ℓ ^ K : ℕ) : ZMod (ℓ ^ a)) * x = 0 ↔ ℓ ^ (a - m) ∣ x.val := by
    intro x
    have hx : ((ℓ ^ K : ℕ) : ZMod (ℓ ^ a)) * x = ((ℓ ^ K * x.val : ℕ) : ZMod (ℓ ^ a)) := by
      rw [Nat.cast_mul, ZMod.natCast_zmod_val]
    rw [hx, ZMod.natCast_eq_zero_iff]
    generalize x.val = v
    by_cases hK : K ≤ a
    · have hmK : m = K := min_eq_right hK
      rw [hmK]
      have e : ℓ ^ a = ℓ ^ K * ℓ ^ (a - K) := by rw [← pow_add, Nat.add_sub_cancel' hK]
      constructor
      · intro h; rw [e] at h; exact (Nat.mul_dvd_mul_iff_left (pow_pos hl0 _)).mp h
      · intro h; rw [e]; exact (Nat.mul_dvd_mul_iff_left (pow_pos hl0 _)).mpr h
    · have hma : m = a := min_eq_left (by omega)
      rw [hma, Nat.sub_self, pow_zero]
      simp only [Nat.one_dvd, iff_true]
      exact Dvd.dvd.mul_right (pow_dvd_pow ℓ (by omega)) _
  let F : {x : ZMod (ℓ ^ a) // ((ℓ ^ K : ℕ) : ZMod (ℓ ^ a)) * x = 0} → Fin (ℓ ^ m) :=
    fun x => ⟨x.1.val / ℓ ^ (a - m), by
      rw [Nat.div_lt_iff_lt_mul hpos, ← pow_add, Nat.add_sub_cancel' hm]; exact ZMod.val_lt x.1⟩
  let G : Fin (ℓ ^ m) → {x : ZMod (ℓ ^ a) // ((ℓ ^ K : ℕ) : ZMod (ℓ ^ a)) * x = 0} :=
    fun i => ⟨((ℓ ^ (a - m) * i : ℕ) : ZMod (ℓ ^ a)), by
      rw [hmem, ZMod.val_natCast, Nat.mod_eq_of_lt]
      · exact Dvd.intro _ rfl
      · calc ℓ ^ (a - m) * (i : ℕ) < ℓ ^ (a - m) * ℓ ^ m := Nat.mul_lt_mul_of_pos_left i.2 hpos
          _ = ℓ ^ a := by rw [← pow_add, Nat.sub_add_cancel hm]⟩
  have hFG : Function.LeftInverse G F := by
    intro x
    apply Subtype.ext
    simp only [F, G]
    rw [Nat.mul_div_cancel' ((hmem x.1).mp x.2), ZMod.natCast_zmod_val]
  have hGF : Function.RightInverse G F := by
    intro i
    apply Fin.ext
    simp only [F, G]
    rw [ZMod.val_natCast, Nat.mod_eq_of_lt, Nat.mul_div_cancel_left _ hpos]
    calc ℓ ^ (a - m) * (i : ℕ) < ℓ ^ (a - m) * ℓ ^ m := Nat.mul_lt_mul_of_pos_left i.2 hpos
      _ = ℓ ^ a := by rw [← pow_add, Nat.sub_add_cancel hm]
  rw [Nat.card_congr ⟨F, G, hFG, hGF⟩, Nat.card_fin]

/-- **`|G/NG| = |G[N]|`** for a finite abelian group (the two outer terms of B1 have the same size on a torsion summand). -/
theorem card_quotient_eq_card_ker {G : Type*} [AddCommGroup G] [Finite G] (N : ℕ) :
    Nat.card (G ⧸ (nsmulAddMonoidHom (α := G) N).range) = Nat.card (nsmulAddMonoidHom (α := G) N).ker := by
  have h1 := AddSubgroup.card_eq_card_quotient_mul_card_addSubgroup (nsmulAddMonoidHom (α := G) N).range
  have h2 := AddSubgroup.card_eq_card_quotient_mul_card_addSubgroup (nsmulAddMonoidHom (α := G) N).ker
  rw [Nat.card_congr (QuotientAddGroup.quotientKerEquivRange (nsmulAddMonoidHom (α := G) N)).toEquiv] at h2
  have hpos : 0 < Nat.card G := Nat.card_pos
  have hr : 0 < Nat.card (nsmulAddMonoidHom (α := G) N).range := Nat.card_pos
  have : Nat.card (G ⧸ (nsmulAddMonoidHom (α := G) N).range) * Nat.card (nsmulAddMonoidHom (α := G) N).range =
      Nat.card (nsmulAddMonoidHom (α := G) N).ker * Nat.card (nsmulAddMonoidHom (α := G) N).range := by
    rw [← h1, h2, mul_comm]
  exact Nat.eq_of_mul_eq_mul_right hr this

end GL2.HeckeZ.Bock
