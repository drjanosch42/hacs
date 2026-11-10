/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import Mathlib

/-!
# `hES_N_an`: the rank of the anemic Hecke algebra at level `N` (L-HZ3, M-HZ4b, partly)

`docs/implementation/slot24/heckez/07-lean-plan.md` §3, row `hES_N_an`: "`rank 𝕋^{an,fr} = d^an` of 01 Prop. 1.3(3′)"; [GT]
`lit:hz-es-anemic` (`lean/blueprint/src/gt_part_HZL.tex`, LIT, slot 24: **verified (substitute)**; the named source,
Diamond–Shurman §4.5–4.8 and §5.8, is not in the library).

**Sources (quotations in the [GT] row, read on the page images by LIT).**
* Cuspidal part: Diamond–Im, *Modular forms and modular curves*, Thm. 6.2.3 (multiplicity one for newforms), Rmk. 6.3.3
  (multiplicity one across levels: a `𝐓^{(ND)}`-eigenspace of `𝒮_k(Γ₁(N))` contains a unique newform of a unique level `M ∣ N`),
  Rmk. 6.3.4 (the old-space direct sum `⊕_{M∣N} ⊕_j ⊕_{d : dM∣N} ℂ ι_d^* g^M_j`, the `θ^M_j` distinct as eigencharacters of
  `𝐓^{(N)}`), printed pp. 62–64.
* Eisenstein part: W. Stein, *Modular forms: a computational approach*, Thms. 5.8–5.10 (the basis `E_{k,χ,ψ}(q^t)` of the
  Eisenstein subspace, eigenforms), printed p. 88.
* **Derived, not quoted** (01 Prop. 1.3(3′), STD with a proof sketch): with Eichler–Shimura at level `N` (`lit:diamond-im-es`,
  Diamond–Im Thm. 12.2.2 / Prop. 12.4.10) the anemic operators `T_n`, `(n, N) = 1`, act semisimply on `H¹ ⊗ ℂ` with one system
  per newform of each level `M ∣ N` and one per anemic Eisenstein system, so
  `rank_ℤ 𝕋^{an,fr} = d^an(N, k) = Σ_{M∣N} dim S_k^new(Γ₀(M)) + e^an_k(N)`,
  `e^an_k(N) = #{ψ primitive : cond(ψ)² ∣ N} − [k = 2, N = 1]` (the rank of a commutative semisimple algebra over a field is the
  number of its systems; strong multiplicity one separates newforms of different levels, Brauer–Nesbitt the Eisenstein systems,
  Deligne the Eisenstein from the cuspidal systems — those steps have no quotation here).

**Interface data.** `L` stands for `H¹(Γ₀(N); Sym^{k−2} ℤ)/tors`, `X n` for the Hecke operator `T_n` on it when `(n, N) = 1`;
for `(n, N) ≠ 1` the convention is `X n = 0` (field `bad`), so `ℤ[X n : n ∈ ℕ]` is the anemic algebra `𝕋^{an,fr}`. The number
`d` is `d^an(N, k)`; its value is part of the literature reading (per cell in the [GT] rows that use this structure), not
computed in Lean. Consumers: `GL2.HeckeZ.Cells.LevelN` (T3 at `(N, k) = (11, 2)`, T4 at `(2, 8)`, both `d = 2`).
-/

namespace GL2.Literature

/-- **`hES_N_an`** (07 §3; [GT] `lit:hz-es-anemic`): the anemic Hecke algebra `ℤ[X n]` on `L` (`X n = T_n` for `(n, N) = 1`,
`X n = 0` otherwise) has `ℤ`-rank `d = d^an(N, k)` (01 Prop. 1.3(3′)). -/
structure AnemicRankLevelN {L : Type*} [AddCommGroup L] (X : ℕ → Module.End ℤ L) (N k d : ℕ) : Prop where
  /-- the interface convention: `X n = 0` when `(n, N) ≠ 1` -/
  bad : ∀ n, Nat.gcd n N ≠ 1 → X n = 0
  /-- `rank_ℤ 𝕋^{an,fr} = d^an(N, k)` -/
  rank : Module.finrank ℤ (Algebra.adjoin ℤ (Set.range X)) = d

/-- **Sanity instance** (the structure is satisfiable; not a statement about modular forms): on `L = ℤ` with every `X n = 0`,
`ℤ[X] = ℤ·1` has rank `1`, at every level `N` and weight `k`. -/
theorem anemicRankLevelN_zero (N k : ℕ) :
    AnemicRankLevelN (L := ℤ) (fun _ => 0) N k 1 := by
  refine ⟨fun _ _ => rfl, ?_⟩
  have h : Algebra.adjoin ℤ (Set.range (fun _ : ℕ => (0 : Module.End ℤ ℤ))) = ⊥ := by
    refine le_antisymm (Algebra.adjoin_le ?_) bot_le
    rintro _ ⟨n, rfl⟩
    exact Subalgebra.zero_mem _
  rw [h, Subalgebra.finrank_bot]

end GL2.Literature
