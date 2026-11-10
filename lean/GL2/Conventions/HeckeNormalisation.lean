import GL2.Conventions.SymPow

/-!
# C3: `T_{p^ν}` full versus primitive (the E1 bridge), `T(p,p) = p^n`

[GT] `conv:hecke`, `bridge:fullprimitive`, mirroring [C] v1.15.0 `def:hecke`, `prop:diamond`,
`thm:relations` (caveat), `obl:algebra`(ii),(iii).

The cosets `Γδ` of `{δ ∈ Mat₂(ℤ) : det δ = m}` are represented by the upper-triangular Hermite forms
`(m/d  b; 0  d)`, `d ∣ m`, `0 ≤ b < d` (Shimura Prop. 3.36, a literature fact; here they are an index set).
The **full** operator `T(m)` sums over all of them (`hnf m`), the **primitive** operator `T(1, m)` over those whose
entries have `gcd = 1` (`hnfPrim m`) (Shimura's notation; corrected in slot 19, earlier "the full `T(1, m)`";
the Shimura locators are from memory, the book is not in the library).  For `m = p²` the only non-primitive coset is `p I`
(`hnf_sq_partition`), so `T_{p²} = T_{prim, p²} + ρ_n(pI) = T_{prim,p²} + p^n` (`full_eq_prim_add`).
-/

namespace GL2

open Finset

/-- Hermite representatives of determinant `m`: pairs `(d, b)`, `d ∣ m`, `b < d`, standing for
`(m/d  b; 0  d)`. -/
def hnf (m : ℕ) : Finset (Σ _ : ℕ, ℕ) := (Nat.divisors m).sigma fun d => range d

/-- The matrix of a Hermite representative. -/
def hnfMat (m : ℕ) (x : Σ _ : ℕ, ℕ) : Mat2 ℤ := !![((m / x.1 : ℕ) : ℤ), (x.2 : ℤ); 0, (x.1 : ℤ)]

/-- The primitive representatives: `gcd(m/d, b, d) = 1`. -/
def hnfPrim (m : ℕ) : Finset (Σ _ : ℕ, ℕ) :=
  (hnf m).filter fun x => Nat.gcd (m / x.1) (Nat.gcd x.2 x.1) = 1

/-- `#Γ\{det = m} = σ₁(m)`. -/
theorem card_hnf (m : ℕ) : (hnf m).card = ∑ d ∈ Nat.divisors m, d := by
  simp [hnf, card_sigma]

/-- **The E1 partition at `m = p²`**: the full set is the primitive set plus the single coset `pI`. -/
theorem hnf_sq_partition (p : ℕ) (hp : p.Prime) :
    hnf (p ^ 2) = insert ⟨p, 0⟩ (hnfPrim (p ^ 2)) ∧ (⟨p, 0⟩ : Σ _ : ℕ, ℕ) ∉ hnfPrim (p ^ 2) := by
  have hp0 : 0 < p := hp.pos
  have hpp : p ^ 2 / p = p := by rw [pow_two, Nat.mul_div_cancel _ hp0]
  have hnotprim : (⟨p, 0⟩ : Σ _ : ℕ, ℕ) ∉ hnfPrim (p ^ 2) := by
    simp only [hnfPrim, mem_filter, not_and]
    intro _
    simp only [hpp, Nat.gcd_zero_left, Nat.gcd_self]
    exact hp.one_lt.ne'
  refine ⟨?_, hnotprim⟩
  ext x
  simp only [mem_insert, hnfPrim, mem_filter]
  constructor
  · intro hx
    by_cases hg : Nat.gcd (p ^ 2 / x.1) (Nat.gcd x.2 x.1) = 1
    · exact Or.inr ⟨hx, hg⟩
    · left
      obtain ⟨d, b⟩ := x
      simp only [hnf, mem_sigma, Nat.mem_divisors, mem_range] at hx
      obtain ⟨⟨hd, -⟩, hb⟩ := hx
      obtain ⟨i, hi, rfl⟩ := (Nat.dvd_prime_pow hp).1 hd
      simp only at hg hb ⊢
      interval_cases i
      · exfalso; apply hg; simp at hb; subst hb; simp
      · -- d = p: gcd(p, gcd(b, p)) ≠ 1 forces p ∣ b, so b = 0
        rw [pow_one, hpp] at hg
        rw [pow_one] at hb ⊢
        have : Nat.gcd p (Nat.gcd b p) = Nat.gcd b p := by
          rw [Nat.gcd_comm, Nat.gcd_assoc, Nat.gcd_self]
        rw [this] at hg
        have hdvd : p ∣ b := by
          rcases (Nat.coprime_or_dvd_of_prime hp b) with h | h
          · exact absurd h.symm hg
          · exact h
        have : b = 0 := Nat.eq_zero_of_dvd_of_lt hdvd hb
        subst this; rfl
      · -- d = p²: m/d = 1, gcd = 1
        exfalso; apply hg; simp [Nat.div_self (pow_pos hp0 2)]
  · rintro (h | ⟨hx, -⟩)
    · subst h
      simp [hnf, Nat.mem_divisors, hp0, hp.ne_zero, Dvd.intro_left _ (pow_two p).symm]
    · exact hx

/-- `#primitive(p²) = p² + p`. -/
theorem card_hnfPrim_sq (p : ℕ) (hp : p.Prime) : (hnfPrim (p ^ 2)).card = p ^ 2 + p := by
  obtain ⟨h1, h2⟩ := hnf_sq_partition p hp
  have hc := card_hnf (p ^ 2)
  rw [h1, card_insert_of_notMem h2, Nat.sum_divisors_prime_pow hp] at hc
  simp [Finset.sum_range_succ] at hc
  omega

/-- The non-primitive coset at `p²` is `p · I`. -/
theorem hnfMat_p (p : ℕ) (hp : 0 < p) : hnfMat (p ^ 2) ⟨p, 0⟩ = (p : ℤ) • (1 : Mat2 ℤ) := by
  have hpp : p ^ 2 / p = p := by rw [pow_two, Nat.mul_div_cancel _ hp]
  ext i j; fin_cases i <;> fin_cases j <;> simp [hnfMat, hpp]

/-- **`bridge:fullprimitive`** (E1, [C] `obl:algebra`(iii)): for any coset-additive assembly `F` (an operator
built as a sum over left cosets of a function of the representative), the full operator at `p²` is the primitive
one plus the `pI` term. -/
theorem full_eq_prim_add {M : Type*} [AddCommMonoid M] (p : ℕ) (hp : p.Prime) (F : Mat2 ℤ → M) :
    ∑ x ∈ hnf (p ^ 2), F (hnfMat (p ^ 2) x) =
      ∑ x ∈ hnfPrim (p ^ 2), F (hnfMat (p ^ 2) x) + F ((p : ℤ) • (1 : Mat2 ℤ)) := by
  obtain ⟨h1, h2⟩ := hnf_sq_partition p hp
  rw [h1, sum_insert h2, hnfMat_p p hp.pos, add_comm]

/-- **[C] `prop:diamond`**: the `pI` term acts on `Sym^n` by `p^n` (`ρ_n(p I) = p^n`). -/
theorem diamond_scalar {R : Type*} [CommRing R] (n p : ℕ) :
    rho n (((p : ℤ) • (1 : Mat2 ℤ)).map (Int.castRingHom R)) = (p : R) ^ n • (1 : Module.End R (SymN R n)) := by
  have : ((p : ℤ) • (1 : Mat2 ℤ)).map (Int.castRingHom R) = (p : R) • (1 : Mat2 R) := by
    ext i j
    simp only [Matrix.map_apply, Matrix.smul_apply, Matrix.one_apply, smul_eq_mul, Int.coe_castRingHom]
    split_ifs <;> simp
  rw [this, rho_smul_one]

/-- **[C] `obl:algebra`(ii)/(iii) as algebra**: from the Hecke-ring identity `T_p² = T_{p²} + p · T(p,p)`
(Shimura Thm. 3.24, a literature hypothesis `hrel`), `T_{p²} = T_prim + T(p,p)` (`full_eq_prim_add`) and
`T(p,p) = p^n`, the primitive operator satisfies `T_prim = T_p² − p^{n+1} − p^n`, and the full one
`T_{p²} = T_p² − p^{n+1}`: substituting one for the other is off by exactly `p^n`. -/
theorem prim_recursion {A : Type*} [Ring A] (n p : ℕ) (Tp Tfull Tprim Tpp : A)
    (hrel : Tp * Tp = Tfull + (p : A) * Tpp) (hfull : Tfull = Tprim + Tpp) (hpp : Tpp = (p : A) ^ n) :
    Tfull = Tp * Tp - (p : A) ^ (n + 1) ∧ Tprim = Tp * Tp - (p : A) ^ (n + 1) - (p : A) ^ n := by
  subst hpp
  constructor
  · rw [hrel, pow_succ']; abel
  · rw [hrel, hfull, pow_succ']; abel

end GL2
