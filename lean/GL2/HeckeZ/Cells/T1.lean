/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import Mathlib
import GL2.HeckeZ.Bridge
import GL2.HeckeZ.Check.K3
import GL2.HeckeZ.Cells.T1Data

/-!
# HeckeZ T1: Ramanujan's 691, the Eisenstein and `Δ` systems of `𝕋^fr` congruent mod 691, from one lattice computation (M-HZ1)

`07-lean-plan.md` §7, [GT] `chk:hz-T1`; 01 Props. 5.3 / 5.4 at `(1, 10)` (weight 12, level 1). `L` stands for
`H¹(SL₂(ℤ); Sym¹⁰ ℤ)/torsion` (rank 3) and `X n` for the Hecke operators `T_n` on it (interface data); `𝕋^fr = ℤ[X n]`.

**The index set (audit m3).** `X` is indexed by `ℕ`. The interface convention is `X 0 := 0`, `X 1 = T_1 = id`, and
`X n = T_n` for `n ≥ 2`; `X 0 = 0` and `X 1 = 1` change neither `𝕋^fr` nor the systems. The literature row
`lit:eichler-shimura-levelone` speaks of the `T_p`, `p` prime; `hES` instantiates it over all of `ℕ`. The two are
equivalent: every `T_n` is a polynomial `F_n(T_p : p ∣ n)` (with `F_0 = 0`, `F_1 = 1`), so a joint eigenspace of the
`T_p` for a system `(a_p)` is the joint eigenspace of all `X n` for the system `n ↦ F_n(a_p)`, with the same
multiplicity, and the decomposition over the primes is the decomposition over `ℕ`.

**Hypotheses, all visible in the type of `T1_691`:**
* `hES` — the literature (`lit:eichler-shimura-levelone`, `GL2.Literature.EichlerShimuraLevelOne`), at `ℓ = 691`, on
  `ℤ_691 ⊗ L` with `X n ⊗ ℤ_691`, over a field `E ⊇ ℚ_691`;
* `hχ2`, `hχ3` — **TB-χ**: the integral characteristic polynomials of `T₂`, `T₃` recorded by the descent
  (`descent(1, 10, [2,3,5,7])`, 02's `t1_descentj`);
* `hmat` — **TB-mat**: the dumped `T₂` (`HECKE_DUMP_SAT_CERT`, `(10, 691, K = 2)`) is the matrix of `T₂` on `L/691²` in
  some basis, i.e. it is conjugate mod `691²` to the matrix of `X 2` in a `ℤ`-basis of `L`.
* `hrk` — `rank L = 3`.

**Conclusion.** There are ring maps `λ, μ : 𝕋^fr → ℤ` (the Eisenstein system, `λ(T_p) = 1 + p¹¹` at `p = 2, 3`, and the
system of `Δ`, `μ(T₂) = τ(2) = −24`, `μ(T₃) = τ(3) = 252`) such that
* `λ t ≡ μ t (mod 691)` for **every** `t ∈ 𝕋^fr`: the Eisenstein and `Δ` systems of `𝕋^fr` are congruent mod 691.
  Under the classical identification of the systems, `λ(T_n) = σ₁₁(n)` (Serre GTM 7 §5.5 Prop. 13,
  `lit:serre-course-prop13`) and `μ(T_n) = τ(n)` (`Δ` a normalised eigenform), this is Ramanujan's `τ(n) ≡ σ₁₁(n)`;
  that identification is **not** part of the statement and not a consequence of `hES` (audit m4). The statement pins
  the values at `T₂` and `T₃` only;
* `t ↦ (λ t, μ t)` is injective with image exactly `{(a, c) : a ≡ c mod 691}`: `𝕋^fr ≅ ℤ ×_{ℤ/691} ℤ`;
* in particular the congruence module `C_𝕋(Eis, Δ) = 𝕋^fr/(I_Eis + I_Δ) ≅ ℤ/691`: `(a, 0)` comes from `𝕋^fr` iff `691 ∣ a`.
-/

namespace GL2.HeckeZ.T1

open GL2.Literature GL2.HeckeZ.Bridge GL2.HeckeZ.Exec GL2.HeckeZ.K3 GL2.HeckeZ.T1Data Polynomial TensorProduct

instance fact_691 : Fact (Nat.Prime 691) := ⟨by norm_num⟩

/-- The dumped `T₂` (TB-mat), read in `M₃(ℤ/691²)`. -/
def dumpT2 : Matrix (Fin 3) (Fin 3) (ZMod (691 ^ 2)) := !![477457, 0, 4860; 0, 477457, 0; 0, 0, 2049]

lemma dumpT2_eq : dumpT2 = Matrix.of fun (i j : Fin certT1.f) => (entry certT1.dump i j : ZMod (certT1.ell ^ certT1.K)) := by
  ext i j; fin_cases i <;> fin_cases j <;> rfl

/-- **The TB-χ polynomial of `T₂`, expanded** (audit m1): the coefficients `λ + 2μ = 2001`, `2λμ + μ² = −97776`,
`λμ² = 1180224` (`λ = 2049`, `μ = −24`) that `GL2.HeckeZ.T1Data.certT1_chi` compares, by `decide +kernel`, with the trace,
`c₂` and determinant of the dumped `T₂` modulo `691²`. -/
theorem chi2_expand : ((Polynomial.X + 24) ^ 2 * (Polynomial.X - 2049) : ℤ[X]) =
    Polynomial.X ^ 3 - 2001 * Polynomial.X ^ 2 - 97776 * Polynomial.X - 1180224 := by
  ring

/-- **The Eisenstein and `Δ` systems of `𝕋^fr` are congruent mod 691** (07 §7, M-HZ1; Ramanujan's congruence under the
classical identification of the systems, see the module docstring). -/
theorem T1_691 {L : Type*} [AddCommGroup L] [Module.Free ℤ L] [Module.Finite ℤ L] (X : ℕ → Module.End ℤ L)
    (hrk : Module.finrank ℤ L = 3)
    (E : Type*) [Field E] [Algebra ℤ_[691] E] [Algebra ℚ_[691] E] [IsScalarTower ℤ_[691] ℚ_[691] E]
    (hES : EichlerShimuraLevelOne (ℤ_[691] ⊗[ℤ] L) (Xp 691 X) E)
    (hχ2 : (X 2).charpoly = (Polynomial.X + 24) ^ 2 * (Polynomial.X - 2049))
    (hχ3 : (X 3).charpoly = (Polynomial.X - 252) ^ 2 * (Polynomial.X - 177148))
    (hmat : ∃ (b : Module.Basis (Fin 3) ℤ L) (U : Matrix (Fin 3) (Fin 3) (ZMod (691 ^ 2))), IsUnit U ∧
      U * (LinearMap.toMatrix b b (X 2)).map (Int.cast) = dumpT2 * U) :
    ∃ lam mu : Tfr X →+* ℤ,
      lam (gen X 2) = 2049 ∧ mu (gen X 2) = -24 ∧ lam (gen X 3) = 177148 ∧ mu (gen X 3) = 252 ∧
      (∀ t, lam t ≡ mu t [ZMOD 691]) ∧
      (∀ a c : ℤ, (∃ t, lam t = a ∧ mu t = c) ↔ a ≡ c [ZMOD 691]) ∧
      (∀ a : ℤ, (∃ t, lam t = a ∧ mu t = 0) ↔ (691 : ℤ) ∣ a) ∧
      (∀ t s, lam t = lam s → mu t = mu s → t = s) := by
  have e2 : (X 2).charpoly = (Polynomial.X - C (-24)) ^ 2 * (Polynomial.X - C 2049) := by
    rw [hχ2]; simp [sub_eq_add_neg]
  have e3 : (X 3).charpoly = (Polynomial.X - C 252) ^ 2 * (Polynomial.X - C 177148) := by
    rw [hχ3]; simp
  obtain ⟨Es, hθ, l2, m2, l3, m3⟩ :=
    eisSplit_of_ES hES hrk 2 3 (-24) 2049 252 177148 (by norm_num) (by norm_num) e2 e3
  obtain ⟨b, U, hU, hm⟩ := hmat
  have hsys : ∀ s ∈ certT1.sys, Es.lam (gen X s.1) = s.2.1 ∧ Es.mu (gen X s.1) = s.2.2 := by
    intro s hs
    simp only [certT1, List.mem_cons, List.not_mem_nil, or_false] at hs
    rcases hs with rfl | rfl
    · exact ⟨l2, m2⟩
    · exact ⟨l3, m3⟩
  obtain ⟨hlow, hup⟩ := k3_sound certT1 (k3Check_sound _ certT1_check) (by norm_num [certT1]) Es (gen X) hsys hθ
    b U hU (by rw [← dumpT2_eq]; exact hm)
  have hg : ((certT1.gP : ℕ) : ℤ) = 691 := rfl
  rw [hg] at hlow hup
  have hfib := Es.fibre_iff 691 hlow hup
  refine ⟨Es.lam, Es.mu, l2, m2, l3, m3, fun t => ?_, fun a c => ?_, fun a => ?_, fun t s h1 h2 =>
    Es.pair_injective h1 h2⟩
  · rw [Int.modEq_iff_dvd]; exact dvd_sub_comm.1 (hlow t)
  · rw [hfib, Int.modEq_iff_dvd, dvd_sub_comm]
  · rw [hfib, sub_zero]

end GL2.HeckeZ.T1
