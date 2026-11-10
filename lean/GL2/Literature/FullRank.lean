/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import Mathlib

/-!
# `hES_N`: the rank of the full Hecke algebra at level `N`, with `U_p` (L-HZ4, M-HZ5; slot 26)

`docs/implementation/slot24/heckez/07-lean-plan.md` §3, row `hES_N`; [GT] `lit:hz-es-full` (`lean/blueprint/src/gt_part_HZL.tex`).
Derived, not a quotation: `lit:diamond-im-es` (Diamond–Im Prop. 12.4.10, the Hecke-equivariant Eichler–Shimura isomorphism
for `(Γ_0(N), Δ_0(N))`, `U_p`, `p ∣ N`, included: `lit:diamond-im-es` and the reading of `Δ_0(N)` in the [GT] row) gives
`H¹ ⊗ ℂ ≅ M_k(Γ_0(N)) ⊕ \bar S_k(Γ_0(N))` as modules over the full Hecke algebra `𝕋`. `𝕋` acts faithfully on `M_k` and
`(f, T) ↦ a_1(f|T)` is a perfect pairing `M_k × 𝕋_ℂ → ℂ` (`a_1(f|T_n) = a_n(f)`: left kernel: `a_n(f) = 0` for all `n ≥ 1`, so
`f` is constant, so `0`; right kernel: `t` with `a_1(tf) = 0` for all `f`: then `a_n(tf) = a_1(t T_n f) = 0` for all `n ≥ 1`).
Hence `dim_ℂ 𝕋_ℂ = dim M_k = (b₁ + e_k)/2 =: d` — with **no** reducedness hypothesis (T5 is non-reduced with `d = 13`).
`e_k = dim Eis_k(Γ_0(N))` is `lit:hz-eis-dim`. The perfect pairing on `S_k` is Diamond–Im Prop. 12.4.13 (quoted in
`lit:hz-sturm-integral`); for `M_k` it is the two-line computation above, stated here so that the [GT] row can say what is
derived. The literature rows are verified on page images; **this structure is a derived consequence, not a quotation;
accepted by the owner 2026-11-09** (the [GT] status rules; `lit:hz-es-full`).

**Interface data.** `L` stands for `H¹(Γ₀(N); Sym^{k−2} ℤ)/tors`, `X n` for `T_n` when `(n, N) = 1` (`0` otherwise, as
`AnemicRankLevelN`), `U p` for `U_p` when `p` is a prime dividing `N` (`0` otherwise). **`U p` is the operator of the dump**,
`[Γ₀(N) diag(1, p) Γ₀(N)]`, the adjoint `U_p^* = W_N U_p W_N^{-1}` of the classical `U_p` (`prop:levelconehecke`(3)(d)); the
derivation above is for the classical `U_p` (`a_1(f|U_p) = a_p(f)`) and transfers by conjugation: `W_N` commutes with every `T_n`,
`(n, N) = 1`, so `ℤ[X, U^*] = W_N ℤ[X, U] W_N^{-1}` has the same `ℤ`-rank and is commutative. The family `X` alone is the anemic family
of `GL2.Literature.AnemicRankLevelN`. The fields `commX`, `commU`, `commUU` are the commutativity of the Hecke algebra
(Diamond–Im §3, 12.4), part of the same reading.
-/

namespace GL2.Literature

/-- The full algebra `𝕋^fr = ℤ[X n, U p]`. -/
abbrev Tfull {L : Type*} [AddCommGroup L] (X U : ℕ → Module.End ℤ L) : Subalgebra ℤ (Module.End ℤ L) :=
  Algebra.adjoin ℤ (Set.range X ∪ Set.range U)

/-- **`hES_N`** (07 §3; [GT] `lit:hz-es-full`): on `L`, the full Hecke algebra `ℤ[X n, U p]` at level `N`, weight `k` has
`ℤ`-rank `d` and is commutative (`d = dim M_k(Γ_0(N)) = (b₁ + e_k)/2`; no reducedness). -/
structure FullRankLevelN {L : Type*} [AddCommGroup L] (X U : ℕ → Module.End ℤ L) (N k d : ℕ) : Prop where
  /-- the interface convention: `X n = 0` when `(n, N) ≠ 1` -/
  bad : ∀ n, Nat.gcd n N ≠ 1 → X n = 0
  /-- the interface convention: `U p = 0` unless `p` is a prime dividing `N` -/
  badU : ∀ p, ¬ (p.Prime ∧ p ∣ N) → U p = 0
  /-- the Hecke operators commute -/
  commX : ∀ i j, X i * X j = X j * X i
  /-- the Hecke operators commute -/
  commU : ∀ i j, X i * U j = U j * X i
  /-- the Hecke operators commute -/
  commUU : ∀ i j, U i * U j = U j * U i
  /-- `rank_ℤ 𝕋^fr = d = dim M_k(Γ_0(N))` -/
  rank : Module.finrank ℤ (Tfull X U) = d

/-- **Sanity instance** (the structure is satisfiable; not a statement about modular forms): on `L = ℤ` with every `X n = 0`
and `U p = 0`, `ℤ[X, U] = ℤ·1` has rank `1`, at every level `N` and weight `k`. -/
theorem fullRankLevelN_zero (N k : ℕ) :
    FullRankLevelN (L := ℤ) (fun _ => 0) (fun _ => 0) N k 1 := by
  refine ⟨fun _ _ => rfl, fun _ _ => rfl, fun _ _ => rfl, fun _ _ => rfl, fun _ _ => rfl, ?_⟩
  have h : Tfull (L := ℤ) (fun _ => 0) (fun _ => 0) = ⊥ := by
    refine le_antisymm (Algebra.adjoin_le ?_) bot_le
    rintro _ (⟨n, rfl⟩ | ⟨n, rfl⟩) <;> exact Subalgebra.zero_mem _
  rw [h, Subalgebra.finrank_bot]

end GL2.Literature
