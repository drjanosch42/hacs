/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import Mathlib
import GL2.Literature.AnemicRank
import GL2.HeckeZ.BridgeN
import GL2.HeckeZ.Check.K3N
import GL2.HeckeZ.Cells.LevelNData

/-!
# HeckeZ T3, T4: the anemic Hecke algebra at level `11` (weight `2`) and at level `2` (weight `8`) (M-HZ4b, partly; L-HZ3)

`07-lean-plan.md` §10 M-HZ4b; 01 §9 T3, T4; data `GL2.HeckeZ.Cells.LevelNData` (EXPORT, `docs/implementation/slot25/lhz3-data/`).

**Interface data** (as `GL2.Literature.AnemicRankLevelN`): `L` free of finite rank over `ℤ` stands for
`H¹(Γ₀(N); Sym^{k−2} ℤ)/tors` (rank `3` at `(11, 2)`, `4` at `(2, 8)`), `X n` for `T_n` when `(n, N) = 1` and `0` otherwise;
`𝕋 = ℤ[X n] = 𝕋^{an,fr}`, the anemic algebra on `L`.

**Hypotheses, all in the theorems' types:**
* `hR` — `hES_N_an` (`lit:hz-es-anemic`): `rank_ℤ 𝕋^{an,fr} = d^an = 2` at `(11, 2)` (one newform of level 11, `e^an_2(11) = 1`)
  and at `(2, 8)` (one newform of level 2, `e^an_8(2) = 1`);
* `hχ` — **TB-χ**: the descent record's integral characteristic polynomial of `T_2` (T3), `T_3` (T4) on `H¹/tors`, verbatim;
* `hmat` — **TB-mat**: the EXPORT's free block of `T_2` on `L/5²` (T3), of `T_3` on `L/17²` (T4) is the matrix of `X p` on
  `L/ℓ²` in some `ℤ`-basis of `L`, up to conjugation by a unit mod `ℓ²`.

**What is not claimed.** The system `λ` is the one with `λ(T_p) = 1 + p^{k−1}` at the read operator; that it is the anemic system
of the Eisenstein series (`E₂(q) − 11E₂(q^{11})`, resp. `E₈`) and `μ` that of the newform (`11a`, resp. the level-2 weight-8 newform)
is the classical reading, not part of the statements. At T4 the 2-part of the congruence module is **open**: the read at `ℓ = 2`
(`LevelNData.readT4_2`) reaches nothing, and `2⁷ ∥ λ(T₃) − μ(T₃)` is only the upper bound.
-/

namespace GL2.HeckeZ.Cells.LevelN

open GL2.Literature GL2.HeckeZ GL2.HeckeZ.Bridge GL2.HeckeZ.BridgeN GL2.HeckeZ.Exec GL2.HeckeZ.Exec.K3N GL2.HeckeZ.K3N
  GL2.HeckeZ.Cells.LevelNData Polynomial

variable {L : Type*} [AddCommGroup L] [Module.Free ℤ L] [Module.Finite ℤ L]

/-- The dumped `T₂` at `(11, 0)` on `L/5²` (TB-mat), read in `M₃(ℤ/25)`. -/
def dumpT3 : Matrix (Fin 3) (Fin 3) (ZMod (5 ^ 2)) := !![23, 0, 0; 0, 23, 24; 0, 0, 3]

/-- The dumped `T₃` at `(2, 6)` on `L/17²` (TB-mat), read in `M₄(ℤ/289)`. -/
def dumpT4 : Matrix (Fin 4) (Fin 4) (ZMod (17 ^ 2)) :=
  !![165, 0, 0, 0; 246, 12, 0, 186; 0, 0, 12, 0; 0, 0, 0, 165]

/-- **TB-χ at `(11, 0)`, factored**: the record's `x³ + x² − 8x − 12 = (x + 2)²(x − 3)`. -/
theorem chiT3_factor : (Polynomial.X ^ 3 + Polynomial.X ^ 2 - 8 * Polynomial.X - 12 : ℤ[X]) =
    (Polynomial.X - C (-2)) ^ 2 * (Polynomial.X - C 3) ^ 1 := by
  simp only [map_neg, C_ofNat, pow_one]; ring

/-- **TB-χ at `(2, 6)`, factored**: the record's
`x⁴ − 4400x³ + 4892512x² − 115526400x + 689377536 = (x − 12)²(x − 2188)²`. -/
theorem chiT4_factor : (Polynomial.X ^ 4 - 4400 * Polynomial.X ^ 3 + 4892512 * Polynomial.X ^ 2 - 115526400 * Polynomial.X +
    689377536 : ℤ[X]) = (Polynomial.X - C 12) ^ 2 * (Polynomial.X - C 2188) ^ 2 := by
  simp only [C_ofNat]; ring

/-- **T3 — the anemic Hecke algebra of `Γ₀(11)` in weight `2`**: two systems `λ`, `μ` of `𝕋^{an,fr}` with
`λ(T₂) = 3 = 1 + 2`, `μ(T₂) = −2`, congruent modulo `5` on all of `𝕋^{an,fr}`; `𝕋^{an,fr} ≅ ℤ ×_{ℤ/5} ℤ`
(`t ↦ (λ t, μ t)` injective with image `{a ≡ c mod 5}`), so the congruence module `C = 𝕋^{an,fr}/(I_λ + I_μ) ≅ ℤ/5`; and
`𝕋^{an,fr} = ℤ[T₂]`. 07 §10's acceptance "T3 `C = ℤ/5` by the at-root read".

The read (`hmat`, `certT3`) is the EXPORT dump `docs/implementation/slot25/lhz3-data/export/t3_N11_n0_ell5_K2.txt`, sha256
`cdf8a0837201abd72f6b16a4b56fafcaa0eb122ccfd8ae9303a50c3832460c1e` (LP v1.10 EXPORT route; manifest `docs/implementation/slot25/lhz3-data/manifest_lhz3.tsv`). -/
theorem T3_11 (X : ℕ → Module.End ℤ L) (hR : AnemicRankLevelN X 11 2 2)
    (hχ : (X 2).charpoly = Polynomial.X ^ 3 + Polynomial.X ^ 2 - 8 * Polynomial.X - 12)
    (hmat : ∃ (b : Module.Basis (Fin 3) ℤ L) (U : Matrix (Fin 3) (Fin 3) (ZMod (5 ^ 2))), IsUnit U ∧
      U * (LinearMap.toMatrix b b (X 2)).map (Int.cast) = dumpT3 * U) :
    ∃ lam mu : Tfr X →+* ℤ, lam (gen X 2) = 3 ∧ mu (gen X 2) = -2 ∧
      (∀ t, lam t ≡ mu t [ZMOD 5]) ∧
      (∀ a c : ℤ, (∃ t, lam t = a ∧ mu t = c) ↔ a ≡ c [ZMOD 5]) ∧
      (∀ a : ℤ, (∃ t, lam t = a ∧ mu t = 0) ↔ (5 : ℤ) ∣ a) ∧
      (∀ t s, lam t = lam s → mu t = mu s → t = s) ∧
      (∀ t, ∃ a b : ℤ, t = (a : Tfr X) + b • gen X 2) := by
  have hR2 : Module.finrank ℤ (Tfr X) ≤ 2 := hR.rank.le
  obtain ⟨Es, hθ, hl, hm⟩ := eisSplit_of_rank2 X hR2 2 (-2) 3 (by norm_num) 2 1 (by norm_num) le_rfl
    (by rw [hχ, chiT3_factor])
  have hl' : Es.lam Es.θ = certT3.lam := by rw [hθ, hl]; rfl
  have hm' : Es.mu Es.θ = certT3.mu := by rw [hθ, hm]; rfl
  have hc := k3nCheck_sound _ certT3_check
  have hmat' : ∀ r ∈ certT3.reads, TBmat r (Es.θ : Module.End ℤ L) := by
    intro r hr
    simp only [certT3, List.mem_cons, List.not_mem_nil, or_false] at hr
    subst hr
    obtain ⟨b, U, hU, h⟩ := hmat
    refine ⟨b, U, hU, ?_⟩
    rw [hθ]
    convert h using 2
    · rfl
    · ext i j; fin_cases i <;> fin_cases j <;> rfl
  obtain ⟨hlow, hfib, hcong⟩ := k3n_fibre certT3 hc certT3_complete Es hl' hm' hmat'
  have hg : certT3.lam - certT3.mu = 5 := by decide
  rw [hg] at hlow hfib hcong
  refine ⟨Es.lam, Es.mu, hl, hm, fun t => ?_, fun a c => ?_, hcong, fun t s h1 h2 => Es.pair_injective h1 h2,
    fun t => ?_⟩
  · rw [Int.modEq_iff_dvd]; exact dvd_sub_comm.1 (hlow t)
  · rw [hfib, Int.modEq_iff_dvd, dvd_sub_comm]
  · obtain ⟨a, b, h⟩ := k3n_gen certT3 hc certT3_complete Es hl' hm' hmat' t
    exact ⟨a, b, by rw [h, hθ]⟩

/-- **T4 — the anemic Hecke algebra of `Γ₀(2)` in weight `8`, its 17-part**: two systems `λ`, `μ` of `𝕋^{an,fr}` with
`λ(T₃) = 2188 = 1 + 3⁷`, `μ(T₃) = 12`, congruent modulo `17` on all of `𝕋^{an,fr}`; `t ↦ (λ t, μ t)` is injective; and the
congruence ideal `{a : (a, 0) comes from 𝕋^{an,fr}} = gℤ` has `17 ∣ g ∣ 2176 = 2⁷·17`: `(2176, 0)` comes from `𝕋^{an,fr}`,
and every `(a, 0)` that does has `17 ∣ a`. The 2-part of `g` (between `1` and `2⁷`) is not determined here.

The read (`hmat`, `certT4`) is the EXPORT dump `docs/implementation/slot25/lhz3-data/export/t4_N2_n6_ell17_K2.txt`, sha256
`ae0ba2ed825a69f820958277477e2849dafc4ca610509103467230571e0584b0`; the `ℓ = 2` aside `LevelNData.readT4_2` (used by no theorem)
is `export/t4_N2_n6_ell2_K9.txt`, sha256 `0820add8c4ff0086ec4cd6770cf43eef0cddd595e0bc3c101b95544cb6981278` (LP v1.10 EXPORT
route; manifest `docs/implementation/slot25/lhz3-data/manifest_lhz3.tsv`). -/
theorem T4_2 (X : ℕ → Module.End ℤ L) (hR : AnemicRankLevelN X 2 8 2)
    (hχ : (X 3).charpoly = Polynomial.X ^ 4 - 4400 * Polynomial.X ^ 3 + 4892512 * Polynomial.X ^ 2 -
      115526400 * Polynomial.X + 689377536)
    (hmat : ∃ (b : Module.Basis (Fin 4) ℤ L) (U : Matrix (Fin 4) (Fin 4) (ZMod (17 ^ 2))), IsUnit U ∧
      U * (LinearMap.toMatrix b b (X 3)).map (Int.cast) = dumpT4 * U) :
    ∃ lam mu : Tfr X →+* ℤ, lam (gen X 3) = 2188 ∧ mu (gen X 3) = 12 ∧
      (∀ t, lam t ≡ mu t [ZMOD 17]) ∧
      (∃ t, lam t = 2176 ∧ mu t = 0) ∧
      (∀ a : ℤ, (∃ t, lam t = a ∧ mu t = 0) → (17 : ℤ) ∣ a) ∧
      (∀ t s, lam t = lam s → mu t = mu s → t = s) := by
  have hR2 : Module.finrank ℤ (Tfr X) ≤ 2 := hR.rank.le
  obtain ⟨Es, hθ, hl, hm⟩ := eisSplit_of_rank2 X hR2 3 12 2188 (by norm_num) 2 2 (by norm_num) (by norm_num)
    (by rw [hχ, chiT4_factor])
  have hl' : Es.lam Es.θ = certT4.lam := by rw [hθ, hl]; rfl
  have hm' : Es.mu Es.θ = certT4.mu := by rw [hθ, hm]; rfl
  have hc := k3nCheck_sound _ certT4_check
  have hmat' : ∀ r ∈ certT4.reads, TBmat r (Es.θ : Module.End ℤ L) := by
    intro r hr
    simp only [certT4, List.mem_cons, List.not_mem_nil, or_false] at hr
    subst hr
    obtain ⟨b, U, hU, h⟩ := hmat
    refine ⟨b, U, hU, ?_⟩
    rw [hθ]
    convert h using 2
    · rfl
    · ext i j; fin_cases i <;> fin_cases j <;> rfl
  obtain ⟨hlow, hup⟩ := k3n_sound certT4 hc Es hl' hm' hmat'
  have hG : (certT4.G : ℤ) = 17 := by rw [certT4_bounds.1]; rfl
  have hg : certT4.lam - certT4.mu = 2176 := by decide
  rw [hG] at hlow
  rw [hg] at hup
  refine ⟨Es.lam, Es.mu, hl, hm, fun t => ?_, ?_, fun a ⟨t, ht1, ht2⟩ => ?_, fun t s h1 h2 => Es.pair_injective h1 h2⟩
  · rw [Int.modEq_iff_dvd]; exact dvd_sub_comm.1 (hlow t)
  · exact (Es.mem_congIdeal).mp hup
  · have := hlow t; rw [ht1, ht2, sub_zero] at this; exact this

end GL2.HeckeZ.Cells.LevelN
