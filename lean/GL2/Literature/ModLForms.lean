import GL2.Literature.Basic
import Mathlib.NumberTheory.ModularForms.LevelOne.Basic

/-!
# L1: modular forms modulo `ℓ` (level one) and the Bellaïche–Khare theorem (L16-3)

Sources read from the library scans (read-only; nothing copied into the tree):

* Serre, *Congruences et formes modulaires (d'après H. P. F. Swinnerton-Dyer)*, Séminaire Bourbaki
  1971/72, exposé 416 -- the library holds an **English translation** only
  (`papers/s/Serre/Serre-Congruences_et_formes_modulaires_(dapres_HPF_Swinnerton-Dyer)_TRANSLATED.pdf`;
  the French original in the same folder is a truncated PDF: "Couldn't find trailer dictionary").
  Quotations below are from the translation, p. 4 of 19.
* Swinnerton-Dyer, *On ℓ-adic representations and congruences for coefficients of modular forms*,
  LNM 350 (1973), pp. 1-55: printed p. 22, Theorem 2.
* Bellaïche-Khare, *Level 1 Hecke algebras of modular forms modulo p*, Compositio Math. 151 (2015),
  397-415: p. 401, Theorems I-III.
-/

open UpperHalfPlane ModularForm ModularFormClass MatrixGroups

namespace GL2.Literature

/-- A level-one modular form of weight `k` with **rational, `p`-integral `q`-expansion**: the objects of
Serre's §1.2 ("modular forms of weight `k`, with rational coefficients, which are `p`-integral"). -/
structure PIntegralForm (p : ℕ) (k : ℤ) where
  f : ModularForm 𝒮ℒ k
  c : ℕ → ℚ
  coeff_eq : ∀ n, (qExpansion 1 f).coeff n = ((c n : ℚ) : ℂ)
  integral : ∀ n, VGe p 0 (c n)

/-- `f ≢ 0 (mod p)`: some coefficient is not divisible by `p`. -/
def PIntegralForm.NonzeroModP {p : ℕ} {k : ℤ} (g : PIntegralForm p k) : Prop :=
  ∃ n, ¬ VGe p 1 (g.c n)

/-- `f ≡ g (mod p^m)` coefficientwise. -/
def PIntegralForm.CongPow {p : ℕ} {k k' : ℤ} (m : ℕ) (f : PIntegralForm p k) (g : PIntegralForm p k') :
    Prop :=
  ∀ n, CongModPow p m (f.c n) (g.c n)

/-- **Serre, Bourbaki 416, §1.2, Theorem 2** (weight modulo `p - 1`); the same statement as
Swinnerton-Dyer, LNM 350, Theorem 2 (iv) (grading of `M̃` by `ℤ/(ℓ-1)`). `VERIFIED` (translation).

Quotation (translation p. 4): "Theorem 2. Suppose f and f' are p-integral modular forms of weights k
and k'. If f ≡ f' (mod p), and f ≢ 0 (mod p), then one has k ≡ k' (mod (p − 1))." The exposé's standing
hypothesis is "Suppose from now on that p ≥ 5" (p. 3), which is the field's first hypothesis here.
Swinnerton-Dyer, LNM 350, p. 22: "THEOREM 2. Suppose that ℓ > 3. Then ... (iv) M̃ is naturally isomorphic
to F_ℓ[Q̃,R̃]/(Ã-1) and has a natural grading with values in ℤ/(ℓ-1)."

Used by: [E] `cl:mkfaithful` (`m = 1` half), `cl:sigma`; [C] `rmk:thetaclassical`; [U] `cor:u-irregular`. -/
structure Bourbaki416Thm2 (p : ℕ) : Prop where
  weight_congruence : p.Prime → 5 ≤ p → ∀ {k k' : ℤ} (f : PIntegralForm p k) (g : PIntegralForm p k'),
    f.NonzeroModP → f.CongPow 1 g → k ≡ k' [ZMOD ((p : ℤ) - 1)]

/-- **Serre, Bourbaki 416, §1.2, the Remark after Theorem 2** (weight modulo `p^{m-1}(p-1)`), stated
there **without proof** ("one can show"). `VERIFIED` as a statement of the source (translation p. 4).

Quotation: "Remark. Under the hypotheses of Theorem 2, if f ≡ f' (mod p^n), then one can show that
k ≡ k' (mod p^{n-1}(p − 1))."

**Locator note (shared with WP-8).** [E] cites this fact as "Serre, LNM 350, §1, Théorème 1". The library
does **not** hold *Formes modulaires et fonctions zêta p-adiques* (LNM 350, pp. 191-268), so that locator
stays `UNVERIFIED`; this Remark is the verified statement of the `p^m` form. -/
structure Bourbaki416RemarkPow (p : ℕ) : Prop where
  weight_congruence_pow : p.Prime → 5 ≤ p → ∀ (m : ℕ), 1 ≤ m → ∀ {k k' : ℤ} (f : PIntegralForm p k)
    (g : PIntegralForm p k'), f.NonzeroModP → f.CongPow m g →
      k ≡ k' [ZMOD ((p : ℤ) ^ (m - 1) * ((p : ℤ) - 1))]

/-- **Sturm's bound, level one.** Locator `VERIFIED` (LIT-13, received 2026-10-05): Sturm, *On the
congruence of modular forms*, Number Theory (New York 1984-85), LNM 1240 (1987), pp. 275-280, Theorem 1 on p. 276:
`Γ ⊇ Γ(N)`, `k ≥ 1`, `f, g ∈ M_k(Γ, R_F)`, `ord_λ (f - g) > k [Γ(1) : Γ] / 12 → f ≡ g mod λ` (the constant term is
included; at level one the bound is `k / 12`, as below). Sturm's coefficient ring is `R_F` with a prime `λ`; the passage to
`p`-integral forms is a localisation not written in the paper. The level-one proof is by `Δ` and `j`; the general case uses
Shimura Thm 6.6 (Ch. 6, not held). No quotation is given.

If the first coefficients (`n ≤ k/12`) of a `p`-integral level-one form are divisible by `p`, all are. -/
structure SturmLevelOne (p : ℕ) : Prop where
  bound : p.Prime → ∀ {k : ℤ} (f : PIntegralForm p k),
    (∀ n : ℕ, (n : ℤ) * 12 ≤ k → VGe p 1 (f.c n)) → ∀ n, VGe p 1 (f.c n)

/-- **Bellaïche-Khare, Theorems I-III** (Compositio Math. 151 (2015), p. 401). `VERIFIED` locator; the
objects are **interface data** (see `Basic.lean`).

Quotation: "Theorem I. Assume that ρ̄ is unobstructed. Then the morphism R̃⁰_ρ̄ → A_ρ̄ is an isomorphism,
and A_ρ̄ is isomorphic to a power series ring in two variables F[[x, y]]. Theorem II. Assume that ρ̄ is
absolutely irreducible after restriction to the Galois group of ℚ(ζ_p). If ρ̄|G_{ℚ_p} is reducible,
assume in addition that ρ̄|G_{ℚ_p} is not isomorphic to χ ⊗ (1 *; 0 1) nor to χ ⊗ (1 *; 0 ω_p), where χ
is any character G_{ℚ_p} → F^* and * may be trivial or not. Then R̃⁰_ρ̄ → A_ρ̄ is an isomorphism and both
rings have dimension 2. Theorem III. In any case, R̃⁰_ρ̄ and A_ρ̄ have dimension at least 2."

* `A` is `A_ρ̄`, the local component at the modular `ρ̄ : G_{ℚ,p} → GL₂(F)` of the **shallow** (no `U_p`)
  Hecke algebra `A = lim A_k` of the `F`-algebras generated by the `T_ℓ` **and `S_ℓ`** (`ℓ ≠ p`) acting
  on `S_{≤k}(F)` (§1.2, p. 398). It is *not* the fixed-weight algebra, and *not* the `O`-algebra `T_ρ̄`.
* `unobstructed` is Definition 1 (p. 400): `dim Tan D̃⁰_ρ̄ = 2`; `hII` is Theorem II's hypothesis.
* Dimension is Krull dimension (`ringKrullDim`).
* Scope: §§2-9 assume `p > 3`; `p = 2` is Nicolas-Serre + Bellaïche, `p = 3` is the appendix (p. 401).

**Standing hypotheses of the source, as guards** (L16-5 finding 1): `F` has characteristic `p` (`CharP F p`,
which ties the parameter `p` to the data) and `ρ̄` is modular (the interface flag `modular`, like `Wiles214`'s
`ordinary`: `A_ρ̄` exists only for a modular `ρ̄`) on all three fields, and `3 < p` on Theorems II and III
("In §§ 2–9 we assume p > 3"). Theorem I carries no `p`-guard: at `p = 2, 3` it holds by the cited work
(Nicolas-Serre + Bellaïche; the appendix), as recorded above.

**What [E] `cl:bk` adds that is not in Theorems I-III:** "with `x, y` realisable as `T_q - a_q`, `T_{q'} - a_{q'}`"
is not stated there (only the `p = 3` appendix, Corollary 25, names `T_2'`, `T_7'` as generators);
"irreducible" is not a hypothesis of Theorem I. -/
structure BellaicheKhareThms (p : ℕ) (F A : Type*) [Field F] [CommRing A] [Algebra F A]
    (modular unobstructed hII : Prop) : Prop where
  thmI : CharP F p → modular → unobstructed → Nonempty (MvPowerSeries (Fin 2) F ≃ₐ[F] A)
  thmII : CharP F p → modular → 3 < p → hII → ringKrullDim A = 2
  thmIII : CharP F p → modular → 3 < p → 2 ≤ ringKrullDim A

/-! ### Sanity instances (the hypotheses are satisfiable, and the shapes are not vacuous) -/

/-- The constant form `1` of weight `0`, as a `PIntegralForm` (`c = (1, 0, 0, ...)`). -/
noncomputable def oneForm (p : ℕ) (hp : 2 ≤ p) : PIntegralForm p 0 where
  f := 1
  c := fun n => if n = 0 then 1 else 0
  coeff_eq := by
    intro n
    rw [ModularForm.qExpansion_one]
    by_cases h : n = 0 <;> simp [h, PowerSeries.coeff_one]
  integral := by
    intro n
    by_cases h : n = 0
    · subst h
      refine ⟨?_, by simp⟩
      intro hd
      have : p ∣ 1 := by simpa using hd
      exact absurd (Nat.le_of_dvd one_pos this) (by omega)
    · simpa [h] using VGe_zero p 0 hp

/-- Sanity (a): the antecedents of `Bourbaki416Thm2` / `RemarkPow` are inhabited at `p = 5`: a form that
is not `0` mod `5`, congruent to itself modulo every `5^m`. So neither structure is vacuous. -/
theorem serre_antecedents_inhabited :
    ∃ f : PIntegralForm 5 0, f.NonzeroModP ∧ ∀ m, f.CongPow m f := by
  refine ⟨oneForm 5 (by norm_num), ⟨0, ?_⟩, fun m n => ?_⟩
  · simp [oneForm, VGe]
  · simpa [CongModPow] using VGe_zero 5 m (by norm_num)

/-- Sanity (b): `BellaicheKhareThms` is consistent on the trivial data `A = F[[x, y]]` once the Krull
dimension of `F[[x,y]]` is supplied (**not** proved here: it is an interface hypothesis of the example);
Theorem I then holds by the identity isomorphism. -/
example (p : ℕ) (F : Type*) [Field F] (hdim : ringKrullDim (MvPowerSeries (Fin 2) F) = 2) :
    BellaicheKhareThms p F (MvPowerSeries (Fin 2) F) True True True :=
  ⟨fun _ _ _ => ⟨AlgEquiv.refl⟩, fun _ _ _ _ => hdim, fun _ _ _ => hdim.ge⟩

end GL2.Literature
