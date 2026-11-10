/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import Mathlib
import GL2.Literature.SturmCuspPrime
import GL2.HeckeZ.Sturm
import GL2.HeckeZ.Three
import GL2.HeckeZ.Check.Cert

/-!
# HeckeZ LC-11 at prime level (M-HZ5; slot 26): 01 Prop. 5.6 for `Γ₀(N)`, `N` prime

`01-math-core.md` Prop. 5.6 (`prop:heckeZeiscert`(1): prime level `N`, `k = 2`, one rational Eisenstein component), `07-lean-plan.md`
§10 M-HZ5. The certificate is stated at weight `2` only (`SturmCuspPrimeLevel` is false for `k > 2`); `comm_Xt`, `Tfr_Xt_eq`, `Tp_Xt`
are algebra about the shifted family and stay general in `k`. `GL2.Literature.XtPrime` replaces the Eisenstein-eigenvalue `U_N ↦ 1` by `U_N + N^{k−1} ↦ 1 + N^{k−1}` so that
`GL2.HeckeZ.Sturm.core` applies verbatim; `Tfr` of the shifted family is `Tfull` (`U_N`, `N` prime, is the only `U_p`, and `X N = 0`).
-/

namespace GL2.HeckeZ.SturmN

open GL2.Literature GL2.HeckeZ GL2.HeckeZ.Bridge GL2.HeckeZ.Cert GL2.HeckeZ.Check GL2.HeckeZ.Sturm GL2.HeckeZ.Three Polynomial

set_option linter.unusedSectionVars false

variable {L : Type*} [AddCommGroup L] [Module.Free ℤ L] [Module.Finite ℤ L]
variable {X U : ℕ → Module.End ℤ L} {N k d : ℕ}

lemma comm_Xt (hF : FullRankLevelN X U N k d) (i j : ℕ) :
    XtPrime X U N k i * XtPrime X U N k j = XtPrime X U N k j * XtPrime X U N k i := by
  have hUX : ∀ i j, U j * X i = X i * U j := fun i j => (hF.commU i j).symm
  have hscal : ∀ (c : ℤ) (a : Module.End ℤ L), (c • (1 : Module.End ℤ L)) * a = a * (c • (1 : Module.End ℤ L)) := by
    intro c a; rw [smul_mul_assoc, mul_smul_comm, one_mul, mul_one]
  simp only [XtPrime]
  split_ifs with h1 h2
  · rfl
  · rw [add_mul, mul_add, hUX j N, hscal]
  · rw [add_mul, mul_add, hF.commU i N, ← hscal]
  · exact hF.commX i j

lemma Tfr_Xt_eq (hN : N.Prime) (hF : FullRankLevelN X U N k d) : Bridge.Tfr (XtPrime X U N k) = Tfull X U := by
  have hUmem : ∀ p, U p ∈ Tfull X U := fun p => Algebra.subset_adjoin (Or.inr ⟨p, rfl⟩)
  refine le_antisymm (Algebra.adjoin_le ?_) (Algebra.adjoin_le ?_)
  · rintro _ ⟨i, rfl⟩
    simp only [XtPrime]
    split_ifs with h
    · exact Subalgebra.add_mem _ (hUmem _) (Subalgebra.zsmul_mem _ (Subalgebra.one_mem _) _)
    · exact Algebra.subset_adjoin (Or.inl ⟨i, rfl⟩)
  · rintro _ (⟨i, rfl⟩ | ⟨j, rfl⟩)
    · by_cases hi : i = N
      · rw [hi, hF.bad N (by rw [Nat.gcd_self]; exact hN.one_lt.ne')]; exact Subalgebra.zero_mem _
      · have : X i = XtPrime X U N k i := by simp [XtPrime, hi]
        rw [this]; exact Algebra.subset_adjoin ⟨i, rfl⟩
    · by_cases hj : j = N
      · subst hj
        have h1 : XtPrime X U j k j ∈ Bridge.Tfr (XtPrime X U j k) := Algebra.subset_adjoin ⟨j, rfl⟩
        have h2 : U j = XtPrime X U j k j - ((j : ℤ) ^ (k - 1)) • (1 : Module.End ℤ L) := by
          simp [XtPrime]
        rw [h2]
        exact Subalgebra.sub_mem _ h1 (Subalgebra.zsmul_mem _ (Subalgebra.one_mem _) _)
      · have hbad : ¬ (j.Prime ∧ j ∣ N) := by
          rintro ⟨hj', hd⟩
          exact hj ((Nat.prime_dvd_prime_iff_eq hj' hN).mp hd)
        rw [hF.badU j hbad]; exact Subalgebra.zero_mem _

lemma Tp_Xt {P : List ℕ} (hNP : N ∉ P) : Tp (XtPrime X U N k) P = Tp X P := by
  simp only [Tp]
  congr 1
  ext f
  constructor <;> rintro ⟨q, hq, rfl⟩ <;> refine ⟨q, hq, ?_⟩ <;>
    · have hne : q ≠ N := fun h => hNP (h ▸ hq)
      simp [XtPrime, hne]

/-- **01 Prop. 5.6 at prime level (LC-11), weight 2**: `𝕋'_(ℓ) = 𝕋^fr_(ℓ)` where `ℓ^{v+1} ∤ g`, `g ∈ (c_q(λ_q))`, given the read
at `(ℓ, v)`. `P ⊇ {primes ≤ 2(N+1)/12}`, `N ∉ P`. Weight `2` and `N` prime only: `hS` (`SturmCuspPrimeLevel`) is stated only there
(it is false for `k > 2`); `N.Prime` is `hS.prime`. -/
theorem heckeZeiscert_prime (hF : FullRankLevelN X U N 2 d) (hS : SturmCuspPrimeLevel X U N)
    {P : List ℕ} (hNP : N ∉ P) (hP : ∀ q : ℕ, q.Prime → q ≤ 2 * (N + 1) / 12 → q ∈ P)
    {ι : Type*} {Q : List ι} {pr : ι → ℕ} (D : EisData (XtPrime X U N 2) 2 Q pr) (hQP : ∀ q ∈ Q, pr q ∈ P)
    {g : ℤ} (hg : g ∈ D.gIdeal) {ℓ v : ℕ} (hℓ : ℓ.Prime) (hgv : ¬ (ℓ : ℤ) ^ (v + 1) ∣ g)
    (hR : v = 0 ∨ ∃ q₀ ∈ Q, ∃ (n : ℕ) (b : Module.Basis (Fin n) ℤ L) (i j : Fin n) (s : ℕ),
      (ℓ : ℤ) ^ s ∣ b.repr (aeval (XtPrime X U N 2 (pr q₀)) (D.c q₀) (b j)) i ∧
      ¬ (ℓ : ℤ) ^ (s + 1) ∣ b.repr (aeval (XtPrime X U N 2 (pr q₀)) (D.c q₀) (b j)) i ∧
      (ℓ : ℤ) ^ (s + v) ∣ (D.c q₀).eval (eisLam 2 (pr q₀))) :
    Loc (Tfull X U) (Tp X P) ℓ := by
  classical
  have hTeq := Tfr_Xt_eq hS.prime hF
  have hTpeq := Tp_Xt (X := X) (U := U) (N := N) (k := 2) hNP
  intro x hx
  have hxT : x ∈ Bridge.Tfr (XtPrime X U N 2) := hTeq ▸ hx
  obtain ⟨s, hs, hxs⟩ := hS.span x hx
  have hsP : s ∈ Tp X P := Algebra.adjoin_mono (by
    rintro _ ⟨q, ⟨hq, hqk⟩, rfl⟩; exact ⟨q, hP q hq hqk, rfl⟩) hs
  have hsTp : s ∈ Tp (XtPrime X U N 2) P := hTpeq ▸ hsP
  have hzT : x - s ∈ Bridge.Tfr (XtPrime X U N 2) := (Bridge.Tfr _).sub_mem hxT (Tp_le _ P hsTp)
  have hz : ∀ w ∈ cuspLat (XtPrime X U N 2) 2, (x - s) w = 0 := fun w hw => by
    rw [LinearMap.sub_apply, hxs w hw, sub_self]
  obtain ⟨y, hy, hgy⟩ := core (comm_Xt hF) D hQP hzT hz hℓ hR g hg
  -- split off the `ℓ`-part of `g` (as in `GL2.HeckeZ.Sturm.heckeZeiscert`)
  have hex : ∃ w, ¬ (ℓ : ℤ) ^ (w + 1) ∣ g := ⟨v, hgv⟩
  set w := Nat.find hex with hwdef
  have hw1 : ¬ (ℓ : ℤ) ^ (w + 1) ∣ g := Nat.find_spec hex
  have hwv : w ≤ v := Nat.find_min' hex hgv
  have hwg : (ℓ : ℤ) ^ w ∣ g := by
    rcases Nat.eq_zero_or_pos w with h0 | hpos
    · rw [h0, pow_zero]; exact one_dvd _
    · have := Nat.find_min hex (show w - 1 < w by omega)
      push Not at this
      rwa [Nat.sub_add_cancel hpos] at this
  obtain ⟨g', hg'⟩ := hwg
  have hg'ℓ : ¬ (ℓ : ℤ) ∣ g' := by
    rintro ⟨t, rfl⟩; exact hw1 ⟨t, by rw [hg', pow_succ]; ring⟩
  have hℓw : ((ℓ : ℤ) ^ w) ≠ 0 := pow_ne_zero _ (by exact_mod_cast hℓ.ne_zero)
  have hz' : g' • (x - s) = ((ℓ : ℤ) ^ (v - w)) • y := by
    have h1 : ((ℓ : ℤ) ^ w) • (g' • (x - s) - ((ℓ : ℤ) ^ (v - w)) • y) = 0 := by
      rw [smul_sub, smul_smul, smul_smul, ← hg', hgy, ← pow_add, Nat.add_sub_cancel' hwv, sub_self]
    exact sub_eq_zero.mp (GL2.HeckeZ.Cert.end_smul_eq_zero hℓw h1)
  refine ⟨g', hg'ℓ, ?_⟩
  have : g' • x = g' • s + ((ℓ : ℤ) ^ (v - w)) • y := by rw [← hz', ← smul_add]; congr 1; abel
  rw [this]
  exact (Tp X P).add_mem (Subalgebra.smul_mem _ hsP _) (Subalgebra.smul_mem _ (hTpeq ▸ hy) _)

end GL2.HeckeZ.SturmN
