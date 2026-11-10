import GL2.Literature.Basic

/-!
# L1: Bernoulli numbers, Kummer congruences, the Iwasawa interpolation (L16-3)

Source: Washington, *Introduction to Cyclotomic Fields*, GTM 83 -- the library copy is the **first
edition (1982)**, a scan without a text layer, read from page images. The bibliography of [C] / [E] /
[U] says "2nd ed., 1997"; the theorem numbers below are the first edition's, and the second edition
was not read (VERIFY the numbering there if that edition is the one cited).
-/

namespace GL2.Literature

/-- **Washington, Corollary 5.14 (Kummer's congruences)**, p. 61 of the first edition. `VERIFIED`.

Quotation: "Corollary 5.14 (Kummer's Congruences). Suppose m ≡ n ≢ 0 (mod p − 1) are positive even
integers. Then B_m/m ≡ B_n/n (mod p). More generally, if m and n are positive even integers with
m ≡ n (mod (p − 1)p^a) and n ≢ 0 (mod p − 1), then (1 − p^{m−1}) B_m/m ≡ (1 − p^{n−1}) B_n/n
(mod p^{a+1})." (The source's `p` is an odd prime; at `p = 2` the hypothesis `n ≢ 0 (mod 1)` is empty.)
Bernoulli numbers are Mathlib's `bernoulli` (`B₁ = -1/2`, which never occurs: `m, n` are even).

Used by: [E] `HeckePredictions` (`kummer_divides`), [U] `cor:u-irregular`, `prop:u-691` (cited with the
locator `VERIFY`, now resolved here). -/
structure Washington514 (p : ℕ) : Prop where
  kummer : p.Prime → ∀ m n : ℕ, Even m → Even n → 0 < m → 0 < n → ¬ ((p : ℤ) - 1 ∣ n) →
    (m : ℤ) ≡ n [ZMOD ((p : ℤ) - 1)] → CongModPow p 1 (bernoulli m / m) (bernoulli n / n)
  kummer_pow : p.Prime → ∀ m n : ℕ, Even m → Even n → 0 < m → 0 < n → ¬ ((p : ℤ) - 1 ∣ n) →
    ∀ a : ℕ, (m : ℤ) ≡ n [ZMOD (((p : ℤ) - 1) * (p : ℤ) ^ a)] →
      CongModPow p (a + 1) ((1 - (p : ℚ) ^ (m - 1)) * (bernoulli m / m))
        ((1 - (p : ℚ) ^ (n - 1)) * (bernoulli n / n))

/-- **Derived from Washington, Theorem 7.10** (with Theorem 7.1, Proposition 7.6 and Theorem 5.11 specialised;
renamed from `Washington710` by L16-5 minor 8, since the structure is a consequence, not the quoted theorem),
pp. 122-123 of the first
edition, specialised to `ψ = 1`, `θ = ω^{k₀}` and evaluated at `s = 1 - k`. `VERIFIED` (the statement
of 7.10; the specialisation to Bernoulli numbers is the identity
`L_p(1 - m, ω^m) = -(1 - p^{m-1})(B_m/m)` in the proof of Corollary 5.14, p. 61).

Quotation (p. 123): "Theorem 7.10. Let χ = θψ be an even Dirichlet character (θ = first kind,
ψ = second kind), and let ζ_ψ = ψ(1 + q₀)^{-1} = χ(1 + q₀)^{-1} (= a root of unity of p-power order).
Then L_p(s, χ) = f(ζ_ψ(1 + q₀)^s − 1, θ)." (p. 122: "Combining Theorem 7.1 and Proposition 7.6, we
find that there are power series f, g, h ∈ 𝒪_θ[[T]] such that ...".)

**This is not what [U] / slot-15 WP-6 quoted.** WP-6's header cites "Washington Thm 7.10: integrality of
the Mahler / Iwasawa series `f(a - 1 + aY)` for `a ≡ 1 mod ell`". Theorem 7.10 is the *interpolation
formula*, and the integrality it gives is that of `f ∈ 𝒪_θ[[T]]` (7.1 and 7.6); the substitution
`T ↦ (1 + q₀)^s - 1` with `s = 1 - k` is then the evaluation of a `ℤ_p`-power series at a point of the
maximal ideal. The structure states exactly that consequence for `p` odd (`q₀ = p`).

For `p` odd and `k₀` even, `p - 1 ∤ k₀`: there is `g ∈ ℤ_p[[T]]` with, for every `k ≥ 2` in the class,
`-(1 - p^{k-1}) B_k/k = g((1 + p)^{1-k} - 1)`, the series converging in `ℚ_p`. -/
structure WashingtonIwasawaBernoulli (p : ℕ) [Fact p.Prime] : Prop where
  interpolation : 3 ≤ p → ∀ k₀ : ℕ, Even k₀ → ¬ ((p : ℕ) - 1 ∣ k₀) →
    ∃ g : PowerSeries ℤ_[p], ∀ k : ℕ, 2 ≤ k → k ≡ k₀ [MOD (p - 1)] →
      Summable (fun n : ℕ => ((g.coeff n : ℤ_[p]) : ℚ_[p]) * (((1 + (p : ℚ_[p])) ^ (1 - (k : ℤ)) - 1) ^ n)) ∧
      (((-(1 - (p : ℚ) ^ (k - 1)) * (bernoulli k / k) : ℚ)) : ℚ_[p]) =
        ∑' n : ℕ, ((g.coeff n : ℤ_[p]) : ℚ_[p]) * (((1 + (p : ℚ_[p])) ^ (1 - (k : ℤ)) - 1) ^ n)

/-! ### Sanity instance: Kummer's congruence, decided at `p = 5`, `(m, n) = (2, 6)` -/

theorem bernoulli_six : bernoulli 6 = 1 / 42 := by decide +kernel

/-- Both sides are `p`-integral and the difference is divisible by `5`: `-1/3` and `-781/63`. -/
theorem kummer_instance_5_2_6 :
    CongModPow 5 1 ((1 - (5 : ℚ) ^ (2 - 1)) * (bernoulli 2 / (2 : ℕ)))
      ((1 - (5 : ℚ) ^ (6 - 1)) * (bernoulli 6 / (6 : ℕ))) := by
  rw [bernoulli_six, bernoulli_two]
  unfold CongModPow VGe
  norm_num

end GL2.Literature
