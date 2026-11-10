/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import Mathlib
import GL2.Literature.SturmCusp
import GL2.Literature.FullRank

/-!
# `hSturm` at prime level: the Sturm bound for the cuspidal Hecke algebra over `ℤ` (LC-11 at prime level, M-HZ5; slot 26)

`07-lean-plan.md` §3 (`hSturm`), 01 Prop. 1.5(2) and §10 (M-HZ5: "LC-11 at prime level, weight 2"). A derived-consequence
structure over interface data, the prime-level companion of `GL2.Literature.SturmCuspLevelOne` (same sources: [GT]
`lit:hz-sturm-integral`: Diamond–Im Prop. 12.4.13, Stein Thm. 9.23 with `m = [SL₂(ℤ) : Γ₀(N)] = N + 1` for `N` prime, so the
Sturm bound `r = km/12 − (m − 1)/N ≤ k(N + 1)/12`; Sturm LNM 1240 Thm. 1; the level-`N` Eichler–Shimura sources).

**Scope: weight `2`, `N` prime, only.** For `k > 2` the statement is false: the prime-level Eisenstein space is two-dimensional
(`U_N`-eigenvalues `1` and `N^{k−1}`), so the lattice below is not cuspidal, and at `(N, k) = (11, 12)` the bound `k(N + 1)/12 = 12`
reaches `N` while `X N = 0`, and `U_11` is not in the anemic algebra on the old space `⟨Δ(z), Δ(11z)⟩` (Z5 audit, M1). The structure
therefore has no weight parameter and carries `N.Prime` as a field.

**Interface data.** `L` stands for `H¹(Γ₀(N); ℤ)/tors` (weight `2`), `N` prime, `X p = T_p` for `p ≠ N`, `U N = U_N`.
`XtPrime X U N 2` is the family with `X̃ N = U_N + N` (so that the cuspidal lattice of `SturmCusp.lean` reads
`Σ_{p ≠ N} (T_p − 1 − p) L + (U_N − 1) L`: `U_N − 1` kills the single Eisenstein class and only the prime-to-`N`
operators enter the Sturm bound `2(N + 1)/12 = (N + 1)/6 < N`). No statement about `U_N` beyond its Eisenstein eigenvalue `1`.
**`U_N` is the operator of the dump**, `[Γ₀(N) diag(1, N) Γ₀(N)]`, the adjoint `U_N^* = W_N U_N W_N^{-1}` of the classical `U_N`
(`prop:levelconehecke`(3)(d)); the statement transfers: `U_N^*` has eigenvalue `1` on the Eisenstein line, and on the cuspidal part
`U_N^* ∈ W_N · 𝕋^an · W_N^{-1} = 𝕋^an` (`W_N` commutes with every `T_n`, `(n, N) = 1`). -/

namespace GL2.Literature

/-- The family with `X̃ N = U_N + N^{k−1}`, `X̃ n = X n` otherwise. -/
noncomputable def XtPrime {L : Type*} [AddCommGroup L] (X U : ℕ → Module.End ℤ L) (N k : ℕ) : ℕ → Module.End ℤ L :=
  fun n => if n = N then U N + ((N : ℤ) ^ (k - 1)) • (1 : Module.End ℤ L) else X n

open Classical in
/-- **Sturm for the cuspidal Hecke algebra over `ℤ`, prime level `N`, weight `2`** (`hSturm` at level `N`; [GT]
`lit:hz-sturm-prime`, a derived consequence; **weight 2 and `N` prime only**, see the module docstring): every element of
`𝕋^fr = ℤ[X, U]` agrees on the cuspidal lattice with an element of `ℤ[X p : p prime, p ≤ 2(N + 1)/12]`. -/
structure SturmCuspPrimeLevel {L : Type*} [AddCommGroup L] (X U : ℕ → Module.End ℤ L) (N : ℕ) : Prop where
  prime : N.Prime
  span : ∀ t ∈ Tfull X U, ∃ s ∈ Algebra.adjoin ℤ (X '' {p : ℕ | p.Prime ∧ p ≤ 2 * (N + 1) / 12}),
    ∀ v ∈ cuspLat (XtPrime X U N 2) 2, t v = s v

/-- Sanity: the hypothesis is satisfiable (by the zero family on any `L`, at any prime `N`). -/
theorem sturmCuspPrimeLevel_zero {L : Type*} [AddCommGroup L] {N : ℕ} (hN : N.Prime) :
    SturmCuspPrimeLevel (L := L) (fun _ : ℕ => 0) (fun _ : ℕ => 0) N := by
  refine ⟨hN, fun t ht => ⟨t, ?_, fun _ _ => rfl⟩⟩
  have h : Tfull (L := L) (fun _ : ℕ => 0) (fun _ : ℕ => 0) = ⊥ := by
    refine le_antisymm (Algebra.adjoin_le ?_) bot_le
    rintro _ (⟨n, rfl⟩ | ⟨n, rfl⟩) <;> exact Subalgebra.zero_mem _
  rw [h] at ht
  exact (bot_le : (⊥ : Subalgebra ℤ (Module.End ℤ L)) ≤ _) ht

end GL2.Literature
