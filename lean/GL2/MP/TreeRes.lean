/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.MP.TreeGamma
import GL2.MP.PeriodicCone

/-!
# LR-Y phase 2b–d (slot 22): the free resolution of `k` over `k[SL₂(ℤ)]` from the tree

[GT] `gt_part_MP.tex`, rows `def:mp-model7-lift`, `thm:mp-model7-cone-res`, backing [C] `cl:amalgam`, `eq:mvamalgam`
(plan of record `LV0_audit.md` §3.2, LR-2 (a)–(c)).

* `model7PerE`, `model7PerV` (**LR-2 (a)**): the periodic data of the tree edges (`Stab[ρ, i] = ⟨−I⟩`, `N = 2`) and of
  the tree vertices (`Γ_i = ⟨S⟩`, `N = 4`; `Γ_ρ = ⟨U⟩`, `N = 6`), so `perCx` gives the resolutions
  `P^{⟨−I⟩} → k[𝒯_1] = k[Γ/⟨−I⟩]` and `P^{⟨S⟩} ⊕ P^{⟨U⟩} → k[𝒯_0]`.
* `model7Phi` (**LR-2 (b)**): the lift `φ : P^{⟨−I⟩} → P^{⟨S⟩} ⊕ P^{⟨U⟩}` of the tree boundary `k[𝒯_1] → k[𝒯_0]`
  (`[ρ, i] ↦ i − ρ`), `φ = (1, −1)` in even degrees and `(1 + S, −(1 + U + U²))` in odd degrees; `φ` is a chain map by
  `S² = U³ = −I` (`model7Phi_comm`).
* `model7Cone k` / **`model7ConeRes_acyclic`** (**LR-2 (c)**): the cone complex
  `k ← k[Γ]² ← k[Γ]³ ← k[Γ]³ ← ⋯` (ranks `2, 3, 3, …`, 2-periodic from degree 2 by construction) is **acyclic**, over every
  commutative ring `k`: `ConeDat.acyclic` with `treeses_exact` (phase 1).  No degree certificate is needed: periodicity
  is built into the complex, and the argument is uniform in the degree.
-/

namespace GL2.MP

open Finsupp Cover GL2.TauChk
open scoped MatrixGroups

/-! ### LR-2 (a): the periodic data of the tree of `M₇` -/

lemma negOne_eq_sS_sq : (-1 : SL(2, ℤ)) = sS ^ 2 := sS_sq.symm

lemma negOne_sq : (-1 : SL(2, ℤ)) ^ 2 = 1 := by
  rw [negOne_eq_sS_sq, ← pow_mul]; exact sS_four

lemma negOne_ne_one : (-1 : SL(2, ℤ)) ≠ 1 := by
  intro h
  have h' := congrArg (fun g : SL(2, ℤ) => (g : Matrix (Fin 2) (Fin 2) ℤ) 0 0) h
  simp at h'

/-- **The periodic data of the tree edges**: base point `[ρ, i]`, `t = −I`, `N = 2`. -/
noncomputable def model7PerE : PerData SL(2, ℤ) (TS model7Hyp 3 1) 1 where
  x _ := model7E
  t _ := -1
  N _ := 2
  N_pos _ := by norm_num
  pow_N _ := negOne_sq
  pow_inj _ a b h := by
    fin_cases a <;> fin_cases b
    · rfl
    · exact absurd (by simpa using h.symm) negOne_ne_one
    · exact absurd (by simpa using h) negOne_ne_one
    · rfl
  fix _ := (model7_smul_E_iff _).2 ⟨1, (pow_one _).symm⟩
  cov y := by obtain ⟨g, h⟩ := model7_cov_edge y; exact ⟨0, g, h⟩
  sep c c' g g' h := by
    refine ⟨Subsingleton.elim _ _, ?_⟩
    obtain ⟨j, hj⟩ := (model7_smul_E_iff (g'⁻¹ * g)).1 (by rw [mul_smul, ← h, inv_smul_smul])
    exact ⟨j, by rw [← hj, mul_inv_cancel_left]⟩

/-- The vertex base points `i, ρ`. -/
noncomputable def model7VX : Fin 2 → TS model7Hyp 3 0 := ![model7IV, model7RhoV]

/-- The vertex stabiliser generators `S, U`. -/
noncomputable def model7VT : Fin 2 → SL(2, ℤ) := ![sS, sU]

lemma model7_sep_I (g g' : SL(2, ℤ)) (h : g' • model7IV = g • model7IV) : ∃ j : Fin 4, g = g' * sS ^ (j : ℕ) := by
  obtain ⟨j, hj⟩ := (model7_stab_I (g'⁻¹ * g)).1 ((model7_smul_IV_iff _).1 (by rw [mul_smul, ← h, inv_smul_smul]))
  exact ⟨j, by rw [← hj, mul_inv_cancel_left]⟩

lemma model7_sep_Rho (g g' : SL(2, ℤ)) (h : g' • model7RhoV = g • model7RhoV) :
    ∃ j : Fin 6, g = g' * sU ^ (j : ℕ) := by
  obtain ⟨j, hj⟩ := (model7_stab_Rho (g'⁻¹ * g)).1 ((model7_smul_RhoV_iff _).1 (by rw [mul_smul, ← h, inv_smul_smul]))
  exact ⟨j, by rw [← hj, mul_inv_cancel_left]⟩

/-- **The periodic data of the tree vertices**: base points `i` (`t = S`, `N = 4`) and `ρ` (`t = U`, `N = 6`). -/
noncomputable def model7PerV : PerData SL(2, ℤ) (TS model7Hyp 3 0) 2 where
  x := model7VX
  t := model7VT
  N := ![4, 6]
  N_pos c := by fin_cases c <;> decide
  pow_N c := by fin_cases c; exacts [sS_four, sU_six]
  pow_inj c a b h := by
    fin_cases c
    · exact sS_pow_inj a b h
    · exact sU_pow_inj a b h
  fix c := by
    fin_cases c
    · exact (model7_smul_IV_iff _).2 model7_sS_fix
    · exact (model7_smul_RhoV_iff _).2 model7_sU_fix
  cov y := by
    obtain ⟨g, h | h⟩ := model7_cov_vtx y
    · exact ⟨1, g, h⟩
    · exact ⟨0, g, h⟩
  sep c c' g g' h := by
    fin_cases c <;> fin_cases c'
    · exact ⟨rfl, model7_sep_I g g' h⟩
    · exact absurd h (model7_RhoV_ne_IV g' g)
    · exact absurd h.symm (model7_RhoV_ne_IV g g')
    · exact ⟨rfl, model7_sep_Rho g g' h⟩

/-! ### LR-2 (b): the lift `φ : P^{⟨−I⟩} → P^{⟨S⟩} ⊕ P^{⟨U⟩}` -/

variable (k : Type) [CommRing k]

/-- **The lift `φ`** on the generator `g·e` of `P^{⟨−I⟩}_n`: `(1, −1)` for even `n`, `(1 + S, −(1 + U + U²))` for odd
`n` (component `0` is the `⟨S⟩`-resolution, at `i`; component `1` the `⟨U⟩`-resolution, at `ρ`). -/
noncomputable def model7PhiB (n : ℕ) (b : FB 1 SL(2, ℤ)) : FB 2 SL(2, ℤ) →₀ k :=
  if Even n then single ⟨0, b.g⟩ 1 - single ⟨1, b.g⟩ 1
  else single ⟨0, b.g⟩ 1 + single ⟨0, b.g * sS⟩ 1 -
    (single ⟨1, b.g⟩ 1 + single ⟨1, b.g * sU⟩ 1 + single ⟨1, b.g * sU ^ 2⟩ 1)

/-- The chain map on bases: the tree boundary `k[𝒯_1] → k[𝒯_0]` in degree `0`, `φ` above. -/
noncomputable def model7F : ∀ n, PerData.perB SL(2, ℤ) (TS model7Hyp 3 1) 1 n → (PerData.perB SL(2, ℤ) (TS model7Hyp 3 0) 2 n →₀ k)
  | 0 => tBdS model7Hyp 3 k (s := 0)
  | n + 1 => model7PhiB k n

/-- `∂[ρ, i] = i − ρ`. -/
lemma model7_tBdS_E : tBdS model7Hyp 3 k model7E = single model7IV 1 - single model7RhoV 1 := by
  rw [tBdS, altSum, Fin.sum_univ_two]
  have h0 : TS.del 0 model7E = model7IV := TS0_ext (by rw [model7IV_val]; rfl)
  have h1 : TS.del 1 model7E = model7RhoV := TS0_ext (by rw [model7RhoV_val]; rfl)
  rw [h0, h1]
  simp [sub_eq_add_neg]

lemma model7_comm0 (b : FB 1 SL(2, ℤ)) :
    Finsupp.linearCombination k (model7PerV.augB k) (model7PhiB k 0 b) =
      Finsupp.linearCombination k (tBdS model7Hyp 3 k (s := 0)) (model7PerE.augB k b) := by
  rw [model7PhiB, if_pos PerData.even0, map_sub, linearCombination_single, linearCombination_single, one_smul,
    one_smul, PerData.augB, PerData.augB, PerData.augB, linearCombination_single, one_smul, tBdS_smul]
  show _ = mapDomain _ (tBdS model7Hyp 3 k model7E)
  rw [model7_tBdS_E, mapDomain_sub, mapDomain_single, mapDomain_single]
  rfl

lemma mul_sS_sS (g : SL(2, ℤ)) : g * sS * sS = g * -1 := by rw [mul_assoc, ← sq, sS_sq]
lemma mul_sU_sU (g : SL(2, ℤ)) : g * sU * sU = g * sU ^ 2 := by rw [mul_assoc, ← sq]
lemma mul_sU2_sU (g : SL(2, ℤ)) : g * sU ^ 2 * sU = g * -1 := by rw [mul_assoc, ← pow_succ, sU_cube]
lemma sS_pow3 : sS ^ 3 = -1 * sS := by rw [pow_succ, sS_sq]
lemma sU_pow4 : sU ^ 4 = -1 * sU := by rw [pow_succ, sU_cube]
lemma sU_pow5 : sU ^ 5 = -1 * sU ^ 2 := by rw [show 5 = 3 + 2 by rfl, pow_add, sU_cube]

lemma model7_commS (n : ℕ) (b : FB 1 SL(2, ℤ)) :
    Finsupp.linearCombination k (model7PerV.bdS k n) (model7PhiB k (n + 1) b) =
      Finsupp.linearCombination k (model7PhiB k n) (model7PerE.bdS k n b) := by
  by_cases hn : Even n
  · have hn' : ¬ Even (n + 1) := fun h => Nat.even_add_one.1 h hn
    rw [model7PerV.bdS_even k hn, model7PerE.bdS_even k hn, model7PhiB, if_neg hn']
    simp only [map_add, map_sub, linearCombination_single, one_smul, PerData.difB, model7PhiB, if_pos hn]
    simp only [model7PerV, model7PerE, model7VT, Matrix.cons_val_zero, Matrix.cons_val_one,
      mul_sS_sS, mul_sU_sU, mul_sU2_sU]
    abel
  · have hn' : Even (n + 1) := Nat.even_add_one.2 hn
    rw [model7PerV.bdS_odd k hn, model7PerE.bdS_odd k hn, model7PhiB, if_pos hn']
    simp only [map_sub, map_sum, linearCombination_single, one_smul, PerData.nrmB, model7PhiB, if_neg hn]
    simp only [model7PerV, model7PerE, model7VT, Matrix.cons_val_zero, Matrix.cons_val_one,
      Finset.sum_range_succ, Finset.sum_range_zero, zero_add, pow_zero, pow_one, mul_one, sS_sq, sS_pow3, sU_cube,
      sU_pow4, sU_pow5, ← mul_assoc]
    abel

/-- **`φ` is a chain map lifting the tree boundary** (on bases). -/
theorem model7Phi_comm : ∀ n (a : PerData.perB SL(2, ℤ) (TS model7Hyp 3 1) 1 (n + 1)),
    (model7PerV.perCx k).d n (model7F k (n + 1) a) =
      Finsupp.linearCombination k (model7F k n) ((model7PerE.perCx k).bd n a)
  | 0, a => model7_comm0 k a
  | n + 1, a => model7_commS k n a

/-! ### LR-2 (c): the cone resolution -/

/-- **The cone data of the tree of `M₇`**: `P^{⟨−I⟩} → k[𝒯_1]`, `P^{⟨S⟩} ⊕ P^{⟨U⟩} → k[𝒯_0]`, the lift `φ`, `ε : k[𝒯_0] → k`. -/
noncomputable def model7Cone : ConeDat k where
  RA := model7PerE.perCx k
  RB := model7PerV.perCx k
  ε := tBd0 model7Hyp 3 k
  f := model7F k
  f_comm := model7Phi_comm k
  εf a := (treeCx model7Hyp 3 k).bdd 0 a

/-- The terms: `k` in degree `0`, `k[Γ]²` in degree `1`, `k[Γ]² ⊕ k[Γ]` in every degree `≥ 2`. -/
theorem model7Cone_B (n : ℕ) :
    (model7Cone k).cx.B 0 = Unit ∧ (model7Cone k).cx.B 1 = FB 2 SL(2, ℤ) ∧
      (model7Cone k).cx.B (n + 2) = (FB 2 SL(2, ℤ) ⊕ FB 1 SL(2, ℤ)) := ⟨rfl, rfl, rfl⟩

/-- **The cone resolution is acyclic** ([C] `cl:amalgam`, `eq:mvamalgam`): the complex
`k ← k[Γ]² ← k[Γ]³ ← k[Γ]³ ← ⋯` (2-periodic from degree 2) is a resolution of `k` over `k[SL₂(ℤ)]`, for every commutative
ring `k`, `−I` included. -/
theorem model7ConeRes_acyclic : (model7Cone k).cx.toCx.Acyclic := by
  have hT := treeses_exact k
  refine (model7Cone k).acyclic (model7PerE.perCx_acyclic k) (model7PerV.perCx_acyclic k) (fun c => hT.1 c)
    (fun y hy => hT.2 0 y hy) (fun z hz => ?_)
  obtain ⟨w, hw⟩ := hT.2 1 z hz
  have hw0 : w = 0 := Finsupp.ext fun b => (TS.isEmpty model7TreeRetr.top 0 b).elim
  rw [← hw, hw0, map_zero]; rfl

end GL2.MP
