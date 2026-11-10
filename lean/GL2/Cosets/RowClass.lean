import Mathlib

/-!
# G1 (L17-3): the row-class lemma behind `def:cosetreps`

The number theory under [C] `prop:cosets`(4) and `def:cosetreps`: for a prime `p`, `ν ≥ 1`, `N = p^ν`
and a coprime pair `(c, d)` there is exactly one index `k < m = p^ν + p^(ν-1)` with `RowRel p ν k c d`,
where `RowRel` says that `(c : d) ∈ ℙ¹(ℤ/N)` is the class of the `k`-th canonical transporter:
`(k : 1)` for `k < N`, `(1 : p·(k - N))` for `N ≤ k`.  Indices are 0-based: [C]'s `j` is `k + 1`.
-/

namespace GL2.Cosets

/-- `m = p^ν + p^(ν-1)`, the number of cosets ([C] `prop:cosets`). -/
def mIdx (p ν : ℕ) : ℕ := p ^ ν + p ^ (ν - 1)

/-- The class relation: `(c : d) = (k : 1)` (`k < N`) or `(c : d) = (1 : p (k - N))` (`N ≤ k`) in
`ℙ¹(ℤ/N)`, written without division: `N ∣ c - d k`, resp. `N ∣ c · p (k - N) - d`. -/
def RowRel (p ν k : ℕ) (c d : ℤ) : Prop :=
  if k < p ^ ν then ((p : ℤ) ^ ν ∣ c - d * k) else ((p : ℤ) ^ ν ∣ c * (p * ((k : ℤ) - (p : ℤ) ^ ν)) - d)

lemma isCoprime_pow_of_not_dvd {p : ℕ} (hp : p.Prime) {x : ℤ} (h : ¬ (p : ℤ) ∣ x) (e : ℕ) :
    IsCoprime x ((p : ℤ) ^ e) := by
  have hp' : Prime (p : ℤ) := Nat.prime_iff_prime_int.mp hp
  exact ((hp'.coprime_iff_not_dvd).2 h).symm.pow_right

/-- Existence of the class. -/
theorem rowRel_exists {p ν : ℕ} (hp : p.Prime) (hν : 1 ≤ ν) {c d : ℤ} (hcd : IsCoprime c d) :
    ∃ k, k < mIdx p ν ∧ RowRel p ν k c d := by
  have hp' : Prime (p : ℤ) := Nat.prime_iff_prime_int.mp hp
  have hp0 : (0 : ℤ) < p := by exact_mod_cast hp.pos
  set N : ℤ := (p : ℤ) ^ ν with hN
  have hNpos : 0 < N := pow_pos hp0 ν
  by_cases hd : (p : ℤ) ∣ d
  · -- the class `(1 : p c₀)`
    have hc : ¬ (p : ℤ) ∣ c := by
      intro hc
      exact hp'.not_isUnit (hcd.isUnit_of_dvd' hc hd)
    obtain ⟨d', rfl⟩ := hd
    obtain ⟨u, v, huv⟩ := isCoprime_pow_of_not_dvd hp hc (ν - 1)
    set M : ℤ := (p : ℤ) ^ (ν - 1) with hM
    have hMpos : 0 < M := pow_pos hp0 _
    have hNM : N = p * M := by
      rw [hN, hM, ← pow_succ']; congr 1; omega
    set c0 : ℤ := (d' * u) % M with hc0
    have hc0nn : 0 ≤ c0 := Int.emod_nonneg _ hMpos.ne'
    have hc0lt : c0 < M := Int.emod_lt_of_pos _ hMpos
    have hc0nat : c0.toNat < p ^ (ν - 1) := by
      have : (c0.toNat : ℤ) < M := by rw [Int.toNat_of_nonneg hc0nn]; exact hc0lt
      rw [hM] at this; exact_mod_cast this
    refine ⟨(p ^ ν + c0.toNat), ?_, ?_⟩
    · unfold mIdx; omega
    · unfold RowRel
      rw [if_neg (by omega)]
      have hk : ((↑(p ^ ν + c0.toNat) : ℤ) - (p : ℤ) ^ ν) = c0 := by
        push_cast; rw [Int.toNat_of_nonneg hc0nn]; ring
      rw [hk, ← hN, hNM]
      have key : M ∣ c * c0 - d' := by
        have h1 : M ∣ c0 - d' * u := by
          exact ⟨-((d' * u) / M), by rw [hc0, Int.emod_def]; ring⟩
        have h2 : c * c0 - d' = c * (c0 - d' * u) - d' * (v * M) := by linear_combination d' * huv
        rw [h2]
        exact dvd_sub (dvd_mul_of_dvd_right h1 _) (dvd_mul_of_dvd_right (dvd_mul_left _ _) _)
      have : c * (p * c0) - p * d' = p * (c * c0 - d') := by ring
      rw [this]
      exact mul_dvd_mul_left _ key
  · -- the class `(k : 1)`
    obtain ⟨u, v, huv⟩ := isCoprime_pow_of_not_dvd hp hd ν
    set k : ℤ := (c * u) % N with hk
    have hknn : 0 ≤ k := Int.emod_nonneg _ hNpos.ne'
    have hklt : k < N := Int.emod_lt_of_pos _ hNpos
    have hlt : k.toNat < p ^ ν := by
      have : (k.toNat : ℤ) < N := by rw [Int.toNat_of_nonneg hknn]; exact hklt
      rw [hN] at this; exact_mod_cast this
    refine ⟨k.toNat, ?_, ?_⟩
    · unfold mIdx; omega
    · unfold RowRel
      rw [if_pos hlt, Int.toNat_of_nonneg hknn]
      have h1 : N ∣ k - c * u := by
        exact ⟨-((c * u) / N), by rw [hk, Int.emod_def]; ring⟩
      have h2 : c - d * k = c * (v * N) - d * (k - c * u) := by linear_combination (-c) * huv
      rw [← hN, h2]
      exact dvd_sub (dvd_mul_of_dvd_right (dvd_mul_left _ _) _) (dvd_mul_of_dvd_right h1 _)

/-- Uniqueness of the class: the canonical classes are pairwise distinct. -/
theorem rowRel_unique {p ν : ℕ} (hp : p.Prime) (hν : 1 ≤ ν) {c d : ℤ} (hcd : IsCoprime c d) {k k' : ℕ}
    (hk : k < mIdx p ν) (hk' : k' < mIdx p ν) (h : RowRel p ν k c d) (h' : RowRel p ν k' c d) : k = k' := by
  have hp' : Prime (p : ℤ) := Nat.prime_iff_prime_int.mp hp
  have hp0 : (0 : ℤ) < p := by exact_mod_cast hp.pos
  have hpN : (p : ℤ) ∣ (p : ℤ) ^ ν := dvd_pow_self _ (by omega)
  have hNpos : (0 : ℤ) < (p : ℤ) ^ ν := pow_pos hp0 ν
  have hMpos : (0 : ℤ) < (p : ℤ) ^ (ν - 1) := pow_pos hp0 _
  have hNM : (p : ℤ) ^ ν = p * (p : ℤ) ^ (ν - 1) := by rw [← pow_succ']; congr 1; omega
  have hmI : (mIdx p ν : ℤ) = (p : ℤ) ^ ν + (p : ℤ) ^ (ν - 1) := by unfold mIdx; push_cast; ring
  have hkI : (k : ℤ) < (p : ℤ) ^ ν + (p : ℤ) ^ (ν - 1) := by rw [← hmI]; exact_mod_cast hk
  have hkI' : (k' : ℤ) < (p : ℤ) ^ ν + (p : ℤ) ^ (ν - 1) := by rw [← hmI]; exact_mod_cast hk'
  -- in the second family `p ∣ d`, hence `p ∤ c`
  have notc : (p : ℤ) ∣ d → ¬ (p : ℤ) ∣ c := fun hd hc => hp'.not_isUnit (hcd.isUnit_of_dvd' hc hd)
  have fam2 : ∀ k0 : ℤ, (p : ℤ) ^ ν ∣ c * (p * k0) - d → (p : ℤ) ∣ d := by
    intro k0 hh
    have := dvd_trans hpN hh
    have h2 : (p : ℤ) ∣ c * (p * k0) := dvd_mul_of_dvd_right (dvd_mul_right _ _) _
    have := dvd_sub h2 this
    simpa using this
  have fam1 : ∀ k0 : ℤ, (p : ℤ) ^ ν ∣ c - d * k0 → ¬ (p : ℤ) ∣ d := by
    intro k0 hh hd
    have h1 := dvd_trans hpN hh
    have : (p : ℤ) ∣ c := by
      have := dvd_add h1 (dvd_mul_of_dvd_left hd k0)
      simpa using this
    exact notc hd this
  unfold RowRel at h h'
  split_ifs at h h' with h1 h2 h2
  · -- both in the first family
    have hd := fam1 _ h
    have hdiff : (p : ℤ) ^ ν ∣ d * ((k' : ℤ) - k) := by
      have := dvd_sub h h'; rw [show c - d * k - (c - d * k') = d * ((k' : ℤ) - k) by ring] at this; exact this
    have hcop : IsCoprime d ((p : ℤ) ^ ν) := isCoprime_pow_of_not_dvd hp hd ν
    have hdv := hcop.symm.dvd_of_dvd_mul_left hdiff
    have h1I : (k : ℤ) < (p : ℤ) ^ ν := by exact_mod_cast h1
    have h2I : (k' : ℤ) < (p : ℤ) ^ ν := by exact_mod_cast h2
    have := Int.eq_zero_of_abs_lt_dvd hdv (by rw [abs_lt]; constructor <;> omega)
    omega
  · exact absurd (fam2 _ h') (fam1 _ h)
  · exact absurd (fam2 _ h) (fam1 _ h')
  · have hd := fam2 _ h
    have hc := notc hd
    have hdiff : (p : ℤ) ^ ν ∣ c * (p * ((k : ℤ) - k')) := by
      have := dvd_sub h h'
      rw [show c * (p * ((k : ℤ) - (p : ℤ) ^ ν)) - d - (c * (p * ((k' : ℤ) - (p : ℤ) ^ ν)) - d)
          = c * (p * ((k : ℤ) - k')) by ring] at this
      exact this
    have hcop : IsCoprime c ((p : ℤ) ^ ν) := isCoprime_pow_of_not_dvd hp hc ν
    have hdv := hcop.symm.dvd_of_dvd_mul_left hdiff
    rw [hNM] at hdv
    have hdv' := (mul_dvd_mul_iff_left hp0.ne').1 hdv
    have h1I : (p : ℤ) ^ ν ≤ (k : ℤ) := by exact_mod_cast (Nat.le_of_not_lt h1)
    have h2I : (p : ℤ) ^ ν ≤ (k' : ℤ) := by exact_mod_cast (Nat.le_of_not_lt h2)
    have := Int.eq_zero_of_abs_lt_dvd hdv' (by rw [abs_lt]; constructor <;> omega)
    omega

/-- The class exists and is unique. -/
theorem rowRel_existsUnique {p ν : ℕ} (hp : p.Prime) (hν : 1 ≤ ν) {c d : ℤ} (hcd : IsCoprime c d) :
    ∃! k, k < mIdx p ν ∧ RowRel p ν k c d := by
  obtain ⟨k, hk, h⟩ := rowRel_exists hp hν hcd
  exact ⟨k, ⟨hk, h⟩, fun k' ⟨hk', h'⟩ => (rowRel_unique hp hν hcd hk hk' h h').symm⟩

end GL2.Cosets
