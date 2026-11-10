import GL2.Literature.Basic
import Mathlib.NumberTheory.ModularForms.LevelOne.Basic

/-!
# L1: Serre, *A course in arithmetic*, Ch. VII, §5 (Hecke operators on level-one forms) (L17-5)

Source: J.-P. Serre, *A Course in Arithmetic*, GTM 7, Springer (library:
`books/Zahlentheorie/Serre J.P. A course in arithmetic (Springer GTM 7, 1996).pdf`, a page-image scan with no
text layer; read as page images, 2026-10-01).  This is the English translation of the *Cours d'arithmétique*
(PUF 1970); chapter and section numbers are the same.  **Convention.**  Serre writes a modular form of weight
`2k` (`k > 0`); below the weight is `K = 2k`, so Serre's `2k - 1` is `K - 1`.  `T(n)` is Serre's normalisation
(`n^{2k-1} T(n) F(Γ(z,1))`, formula (70)), the one for which `T(n)` acts on `q`-expansions by (75)-(76); it is
the `T_n` of [C] `thm:relations` (`T(p) T(p^ν) = T(p^{ν+1}) + p^{K-1} T(p^{ν-1})`).

The Hecke operators are **interface data** here (Mathlib has no Hecke operators on `ModularForm`): the structures
take the operators as a parameter; `SerreCoursThm7` carries formula (76) of Prop. 12 as the guard that pins the
parameter to Serre's `T(n)`.
-/

open UpperHalfPlane ModularForm ModularFormClass MatrixGroups

namespace GL2.Literature

/-- Serre's formula (76): `T(n)` acts on the `q`-expansion coefficients by
`γ(m) = Σ_{a | (n, m), a ≥ 1} a^{K-1} c(mn / a²)`. -/
def HeckeCoeffFormula (K : ℤ) (T : ℕ → ModularForm 𝒮ℒ K →ₗ[ℂ] ModularForm 𝒮ℒ K) : Prop :=
  ∀ (n m : ℕ) (f : ModularForm 𝒮ℒ K), 1 ≤ n →
    (qExpansion 1 (T n f)).coeff m =
      ∑ a ∈ (Nat.gcd n m).divisors, (a : ℂ) ^ (K - 1) * (qExpansion 1 f).coeff (m * n / a ^ 2)

/-- **Serre, *Course*, VII §5.4, Theorem 7 b), from the guard alone** (L18-4; L17-5 audit MINOR-1): under formula
(76), a normalised `f` with `T n f = l • f` for the one `n` has `c(n) = l`; Prop. 12, Cor. 1 (the coefficient of `q`
in `T(n) f` is `c(n)`) is (76) at `m = 1`.  No weight hypothesis is needed. -/
theorem serre_coeff_eq_eigenvalue {K : ℤ} {T : ℕ → ModularForm 𝒮ℒ K →ₗ[ℂ] ModularForm 𝒮ℒ K}
    (hT : HeckeCoeffFormula K T) (f : ModularForm 𝒮ℒ K) (hf : (qExpansion 1 f).coeff 1 = 1)
    (n : ℕ) (l : ℂ) (hn : 1 ≤ n) (hTf : T n f = l • f) : (qExpansion 1 f).coeff n = l := by
  have h := hT n 1 f hn
  rw [hTf, FunLike.coe_smul, ModularForm.qExpansion_smul one_pos one_mem_strictPeriods_SL,
    PowerSeries.coeff_smul, hf, smul_eq_mul, mul_one] at h
  rw [h, Nat.gcd_one_right, Nat.divisors_one, Finset.sum_singleton]
  simp

/-- **Serre, *A course in arithmetic*, Ch. VII, §5.4, Theorem 7 and its Corollary 1 (printed p. 102).**
`VERIFIED` (page image).

Quotation: "Let f(z) = Σ c(n)q^n be a modular form of weight 2k, k > 0, not identically zero. We assume that f is an
*eigenfunction* of all the T(n), i.e. that there exists a complex number λ(n) such that (77) T(n)f = λ(n)f for all
n ≥ 1.  **Theorem 7.** — a) The coefficient c(1) of q in f is ≠ 0.  b) If f is normalized by the condition
c(1) = 1, then (78) c(n) = λ(n) for all n > 1. [...] **Corollary 1.** — Two modular forms of weight 2k, k > 0,
which are eigenfunctions of the T(n) with the same eigenvalues λ(n), and which are normalized, coincide."

The third field is the Corollary: normalised eigenforms are determined by their systems, so distinct normalised
eigenforms have distinct systems, the multiplicity-one ingredient of [C] `lem:heckerankone` / `prop:cohforms`(3)
(see `EichlerShimuraLevelOne`, `FormsSameSystems`).  Serre does **not** prove that `M_K` has a *basis* of
eigenforms; that is Diamond-Im §3.6, p. 55 (`DiamondImEigenspaces`, cusp forms) together with `M_k = M_k^0 ⊕ C.G_k`
(Serre §3.2, printed p. 88) and Prop. 13 (`SerreCoursProp13`, `G_k` an eigenform).

**Part b) is not a field** (L18-4, L17-5 audit MINOR-1): Serre's proof of b) uses the single `n` only ("Cor. 1 to
prop. 12 shows that the coefficient of q in T(n)f is c(n). On the other hand, by (77), it is also λ(n)c(1)"), and
**Prop. 12, Cor. 1** ("γ(1) = c(n)", printed p. 102) is formula (76) at `m = 1`, i.e. the guard `HeckeCoeffFormula`
itself.  So b) is the theorem `serre_coeff_eq_eigenvalue`, proved from the guard alone; the former field
`coeff_eq_eigenvalue` was dropped.

**Standing hypotheses of the source, as guards** (L16-5 finding 1): `0 < K`, `K` even (weight `2k`, `k > 0`),
and formula (76) for the operators.  Used by: [C] `lem:heckerankone`, `prop:cohforms`(3). -/
structure SerreCoursThm7 (K : ℤ) (T : ℕ → ModularForm 𝒮ℒ K →ₗ[ℂ] ModularForm 𝒮ℒ K) : Prop where
  c1_ne_zero : 0 < K → Even K → HeckeCoeffFormula K T → ∀ f : ModularForm 𝒮ℒ K, f ≠ 0 →
    (∀ n : ℕ, 1 ≤ n → ∃ l : ℂ, T n f = l • f) → (qExpansion 1 f).coeff 1 ≠ 0
  normalised_unique : 0 < K → Even K → HeckeCoeffFormula K T → ∀ f g : ModularForm 𝒮ℒ K,
    (qExpansion 1 f).coeff 1 = 1 → (qExpansion 1 g).coeff 1 = 1 →
    (∀ n : ℕ, 1 ≤ n → ∃ l : ℂ, T n f = l • f ∧ T n g = l • g) → f = g

/-- **Serre, *A course in arithmetic*, Ch. VII, §5.3, Proposition 11 (printed p. 101)**, formulas (72) and (73):
the Hecke relations.  `VERIFIED` (page image; repeated on p. 102 after Corollary 3).

Quotation: "**Proposition 11.** — The function T(n)f is weakly modular of weight 2k. It is holomorphic on H iff f is.
We have: (72) T(m)T(n)f = T(mn) if (m, n) = 1, (73) T(p)T(p^n)f = T(p^{n+1})f + p^{2k-1}T(p^{n-1})f, if p is prime,
n ≥ 1."  (Serre's printed (72) omits the `f`.)

Interface data: `T : ℕ → End M` on a module `M` of (weakly) modular objects of weight `K = 2k`; `M = M_K` is the
case of interest.  **Standing hypotheses as guards:** `0 < K`, `K` even.  **These guards are vacuous on the data**
(L17-5 audit MINOR-2): unlike `SerreCoursThm7`, nothing here ties the free `M`, `T` to Serre's `T(n)` (formula (76));
the structure fixes the identity's shape and locator only, and the L2 layer must instantiate it with the real
`T(n)`.  `SerreCoursThm7Cor2` is the pinned variant on `M = M_K`: the same identities for the eigenvalues `c(n)` of
a normalised eigenform, under the guard `HeckeCoeffFormula`.  The `T 1 = 1` field is Serre's
definition (formula (71) at `n = 1`).  [C] `thm:relations` is the same identity on `H^q`.  Used by: [C]
`thm:relations`, `cl:classical`. -/
structure SerreCoursProp11 (K : ℤ) (M : Type*) [AddCommGroup M] [Module ℂ M] (T : ℕ → Module.End ℂ M) : Prop where
  one : 0 < K → Even K → T 1 = 1
  mul_coprime : 0 < K → Even K → ∀ m n : ℕ, 1 ≤ m → 1 ≤ n → Nat.Coprime m n → T m * T n = T (m * n)
  prime_rec : 0 < K → Even K → ∀ (p ν : ℕ), p.Prime → 1 ≤ ν →
    T p * T (p ^ ν) = T (p ^ (ν + 1)) + ((p : ℂ) ^ (K - 1)) • T (p ^ (ν - 1))

/-- **Serre, *A course in arithmetic*, Ch. VII, §5.4, Corollary 2 to Theorem 7 (printed p. 102)**, formulas (79), (80).
`VERIFIED` (page image, L17-5 repairs, 2026-10-02).

Quotation: "**Corollary 2.** — Under the hypothesis of theorem 7, b): (79) c(m)c(n) = c(mn) if (m, n) = 1,
(80) c(p)c(p^n) = c(p^{n+1}) + p^{2k-1}c(p^{n-1}).  Indeed the eigenvalues λ(n) = c(n) satisfy the same identities
(72) and (73) as the T(n)."

The hypothesis of Theorem 7 b): `f` a modular form of weight `2k = K`, an eigenfunction of all the `T(n)` ((77)),
normalised by `c(1) = 1`.  (80) is read for `p` prime and `n ≥ 1` (the exponent `n - 1`, as in (73)); (79) for
`m, n ≥ 1` (the range of (72)).  **Standing hypotheses as guards:** `0 < K`, `K` even, and formula (76)
(`HeckeCoeffFormula`), which pins `T` to Serre's `T(n)`.  Used by: [C] `thm:relations` (forms side),
`lem:heckerankone`. -/
structure SerreCoursThm7Cor2 (K : ℤ) (T : ℕ → ModularForm 𝒮ℒ K →ₗ[ℂ] ModularForm 𝒮ℒ K) : Prop where
  mul_coprime : 0 < K → Even K → HeckeCoeffFormula K T → ∀ f : ModularForm 𝒮ℒ K,
    (qExpansion 1 f).coeff 1 = 1 → (∀ n : ℕ, 1 ≤ n → ∃ l : ℂ, T n f = l • f) →
    ∀ m n : ℕ, 1 ≤ m → 1 ≤ n → Nat.Coprime m n →
      (qExpansion 1 f).coeff m * (qExpansion 1 f).coeff n = (qExpansion 1 f).coeff (m * n)
  prime_rec : 0 < K → Even K → HeckeCoeffFormula K T → ∀ f : ModularForm 𝒮ℒ K,
    (qExpansion 1 f).coeff 1 = 1 → (∀ n : ℕ, 1 ≤ n → ∃ l : ℂ, T n f = l • f) →
    ∀ (p ν : ℕ), p.Prime → 1 ≤ ν →
      (qExpansion 1 f).coeff p * (qExpansion 1 f).coeff (p ^ ν) =
        (qExpansion 1 f).coeff (p ^ (ν + 1)) + (p : ℂ) ^ (K - 1) * (qExpansion 1 f).coeff (p ^ (ν - 1))

/-- **Serre, *A course in arithmetic*, Ch. VII, §5.5 a), Proposition 13 (printed p. 104).**
`VERIFIED` (page image, L17-5 repairs, 2026-10-02).

Quotation: "a) *Eisenstein series.* — Let k be an integer ≥ 2.  **Proposition 13.** — The Eisenstein series G_k is
an eigenfunction of T(n); the corresponding eigenvalue is σ_{2k-1}(n) and the normalized eigenfunction is
(86) (-1)^k (B_k/4k) E_k = (-1)^k (B_k/4k) + Σ_{n=1}^∞ σ_{2k-1}(n) q^n.  The corresponding Dirichlet series is
ζ(s)ζ(s-2k+1)."

Stated **weaker** than the source: the field asserts that *some* form `E` of weight `K = 2k` has the coefficients
`σ_{K-1}(n)` at every `n ≥ 1` and is an eigenform of every `T(n)` with eigenvalue `σ_{K-1}(n)`; it does not identify
`E` with `G_k`, nor record its constant term (Serre's `B_k` is the old Bernoulli numbering) or the Dirichlet series.
With `M_k = M_k^0 ⊕ C.G_k (for k ≥ 2)` (Serre §3.2, printed p. 88; verified, not a structure) this is the Eisenstein
system of [C] `lem:heckerankone` ("the Eisenstein one once").
**Standing hypotheses as guards:** `4 ≤ K` (`k ≥ 2`), `K` even, formula (76) (`HeckeCoeffFormula`).  Used by:
[C] `lem:heckerankone`, `prop:cohforms`(3). -/
structure SerreCoursProp13 (K : ℤ) (T : ℕ → ModularForm 𝒮ℒ K →ₗ[ℂ] ModularForm 𝒮ℒ K) : Prop where
  eisenstein_eigen : 4 ≤ K → Even K → HeckeCoeffFormula K T → ∃ E : ModularForm 𝒮ℒ K,
    (∀ n : ℕ, 1 ≤ n → (qExpansion 1 E).coeff n = ∑ d ∈ n.divisors, (d : ℂ) ^ (K - 1)) ∧
    ∀ n : ℕ, 1 ≤ n → T n E = (∑ d ∈ n.divisors, (d : ℂ) ^ (K - 1)) • E

/-! ### Sanity (decidable) -/

/-- The Eisenstein eigenvalues `σ_{K-1}(n)` of Serre's Prop. 13 satisfy (73) numerically at `K = 4`, `p = 2`,
`ν = 1`: `σ₃(2)² = σ₃(4) + 2³ σ₃(1)` (`81 = 73 + 8`).  A numerical check of (73)'s shape, not an instance of
the structure (that needs `σ` multiplicative and the prime-power recursion: a proof, G5). -/
theorem eisenstein_prop11_numeric :
    (∑ d ∈ Nat.divisors 2, d ^ 3) * (∑ d ∈ Nat.divisors 2, d ^ 3) =
      (∑ d ∈ Nat.divisors 4, d ^ 3) + 2 ^ 3 * (∑ d ∈ Nat.divisors 1, d ^ 3) := by
  decide

end GL2.Literature
