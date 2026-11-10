/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.L23.TreeF2

/-!
# LV (slot 23): `dim H⁰(SL₂(ℤ); Sym^n F₂²)` from one rank

After LG-3 (`GL2.L23.Split.V2_dims_all`: `dim H^{q+1} = ⌊n/2⌋ + 1` for **every** `n` and `q`), the only per-cell
quantity left at `ℓ = 2` is `h⁰` (its closed form, the Dickson count `#{3a + 2b = n}`, needs Dickson's theorem for
`GL₂(F₂)`, which is not in Lean). It needs **one** rank, `σ = (1 − S, 1 − U)`, not LV-1's two:
`dim H⁰ + rank σ = n + 1` (`finrank_H_zero`), and `rank σ = rankF2 (colsSig n)` (`finrank_range_sigma_V2`, F1).
The cell modules `cells/L23/LV4S*.lean` (`n ∈ (700, 1000]`) decide `rankF2F (colsSig n) = s` (`= rankF2`, `GL2.L23.RankF2.rankF2F_eq`) by `decide +kernel` and
state `dims_n` with `V2_h0_of_rankS` and `Split.V2_dims_all`.
-/

namespace GL2.L23.TreeF2

open GL2.L23.RankF2 Module

/-- **`h⁰` from the rank of `σ` alone**, every `n`. -/
theorem V2_h0_of_rankS (n s h0 : ℕ) (hS : rankF2 (colsSig n) = s) (e0 : n + 1 = h0 + s) :
    finrank (ZMod 2) (groupCohomology (V2 n) 0) = h0 := by
  have h := finrank_H_zero (A := V2 n)
  rw [finrank_range_sigma_V2 n (rowsLt_colsSig n), hS, finrank_V2] at h
  omega

end GL2.L23.TreeF2
