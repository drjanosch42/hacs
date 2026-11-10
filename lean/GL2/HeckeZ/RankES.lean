/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import Mathlib
import GL2.HeckeZ.Bridge
import GL2.HeckeZ.Cert

/-!
# HeckeZ HZ-0 (rank part): `rank 𝕋^fr ≤ ⌈b₁/2⌉` from Eichler–Shimura at level one (L-HZ1, M-HZ2)

`07-lean-plan.md` §2.1 (HZ-0, the bridge from the `ℤ`-lattice `L` to the `ℤ_ℓ`/`E`-objects of the literature structures),
§3 (`hES_N` at level 1 is `lit:eichler-shimura-levelone`), 01 Prop. 1.3(3) (`d = rank 𝕋^fr` from the perfect pairing /
Eichler–Shimura; at level 1, `d = dim M_k = (b₁ + 1)/2`).

**Statement** (`rankLE_of_ES`): if `ℤ_p ⊗ L` with the operators `X i ⊗ ℤ_p` satisfies `EichlerShimuraLevelOne` over a
field `E ⊇ ℚ_p` (finitely many systems, multiplicity `1` or `2`, at most one of multiplicity `1`, spanning), then any
`(b₁ + 1)/2 + 1` elements of `𝕋^fr = ℤ[X i]` satisfy a non-trivial `ℤ`-relation (`GL2.HeckeZ.Cert.RankLE`).

**Proof.**
* The joint eigenspaces `W_χ`, `χ ∈ S`, are independent (`GL2.Sat.jointEigen_indep`) and span `V = E ⊗ L`, so
  `Σ mult χ = b₁`; with `mult ∈ {1, 2}` and at most one `1`, `#S ≤ (b₁ + 1)/2`.
* Every element of `𝕋^fr` acts on each `W_χ` by a scalar (`GL2.HeckeZ.Bridge.scalar_adj`), so the `E`-span of
  `𝕋^fr ⊗ E` lies in the space `U` of endomorphisms scalar on every `W_χ`, and `U ↪ E^S` (one coordinate per `χ`): `dim U ≤ #S`.
* `ℤ`-independent elements of `End_ℤ L` stay `E`-independent in `End_E V` (their matrices in the induced basis are the
  same integer matrices; `linearIndependent_algebraMap_comp_iff`).
-/

namespace GL2.HeckeZ.RankES

open GL2.Literature GL2.Sat GL2.HeckeZ.Bridge GL2.HeckeZ.Cert TensorProduct

set_option linter.unusedSectionVars false

noncomputable section

variable {p : ℕ} [Fact p.Prime]
variable {L : Type*} [AddCommGroup L] [Module.Free ℤ L] [Module.Finite ℤ L]
variable {E : Type*} [Field E] [Algebra ℤ_[p] E] [Algebra ℚ_[p] E] [IsScalarTower ℤ_[p] ℚ_[p] E]

/-- The matrix of `bcH f` in the induced basis is the integer matrix of `f`. -/
lemma toMatrix_bcH {κ : Type*} [Fintype κ] [DecidableEq κ] (b : Module.Basis κ ℤ L) (f : Module.End ℤ L) (i j : κ) :
    LinearMap.toMatrix (bE p E b) (bE p E b) (bcH p E f) i j = ((LinearMap.toMatrix b b f i j : ℤ) : E) := by
  rw [LinearMap.toMatrix_apply, LinearMap.toMatrix_apply, bE_apply, bcH_emb, repr_emb]

/-- `ℤ`-independent endomorphisms of `L` stay `E`-independent after `⊗ ℤ_p ⊗ E`. -/
lemma linearIndependent_bcH {n : Type*} [Fintype n] {v : n → Module.End ℤ L} (hv : LinearIndependent ℤ v)
    (g : n → E) (hg : ∑ i, g i • bcH p E (v i) = 0) : g = 0 := by
  classical
  haveI := charZeroE (p := p) (E := E)
  let b := Module.Free.chooseBasis ℤ L
  let κ := Module.Free.ChooseBasisIndex ℤ L
  -- flatten matrices: an injective `ℤ`-linear map `End_ℤ L → (κ × κ → ℤ)`
  let Fz : Module.End ℤ L →ₗ[ℤ] (κ × κ → ℤ) :=
    (LinearMap.pi fun ij => (Matrix.entryLinearMap ℤ ℤ ij.1 ij.2)).comp (LinearMap.toMatrix b b).toLinearMap
  have hFz : Function.Injective Fz := by
    intro f g hfg
    apply (LinearMap.toMatrix b b).injective
    ext i j
    exact congrFun hfg (i, j)
  have h1 : LinearIndependent ℤ (fun i => Fz (v i)) := hv.map' Fz (LinearMap.ker_eq_bot.mpr hFz)
  haveI : FaithfulSMul ℤ E := (faithfulSMul_iff_algebraMap_injective ℤ E).mpr (RingHom.injective_int _)
  have h2 : LinearIndependent E (fun i => algebraMap ℤ E ∘ Fz (v i)) :=
    linearIndependent_algebraMap_comp_iff.mpr h1
  let FE : Module.End E (E ⊗[ℤ_[p]] (ℤ_[p] ⊗[ℤ] L)) →ₗ[E] (κ × κ → E) :=
    (LinearMap.pi fun ij => (Matrix.entryLinearMap E E ij.1 ij.2)).comp
      (LinearMap.toMatrix (bE p E b) (bE p E b)).toLinearMap
  have hcomp : (fun i => FE (bcH p E (v i))) = fun i => algebraMap ℤ E ∘ Fz (v i) := by
    funext i; funext ij
    simp only [FE, Fz, LinearMap.comp_apply, LinearMap.pi_apply, Matrix.entryLinearMap_apply,
      LinearEquiv.coe_coe, Function.comp_apply, eq_intCast]
    exact toMatrix_bcH b (v i) ij.1 ij.2
  have h3 := Fintype.linearIndependent_iff.mp h2 g (by
    have e := congrArg FE hg
    simp only [map_sum, map_smul, map_zero] at e
    rw [← e]
    refine Finset.sum_congr rfl fun i _ => ?_
    exact congrArg (g i • ·) (congrFun hcomp i).symm)
  funext i; exact h3 i

/-- **HZ-0, rank part**: Eichler–Shimura at level one bounds `rank 𝕋^fr` by `(b₁ + 1)/2`. -/
theorem rankLE_of_ES {ι : Type*} (X : ι → Module.End ℤ L)
    (hES : EichlerShimuraLevelOne (ℤ_[p] ⊗[ℤ] L) (Xp p X) E) :
    RankLE (Tfr X) ((Module.finrank ℤ L + 1) / 2) := by
  classical
  haveI := charZeroE (p := p) (E := E)
  obtain ⟨S, mult, hm, hm1, hdim, htop⟩ := hES.decomp
  have hind := jointEigen_indep E (Xp p X) S htop
  -- `Σ mult = b₁`
  have hsum : ∑ χ ∈ S, mult χ = Module.finrank ℤ L := by
    rw [← finrank_V p E (L := L), ← finrank_top (R := E) (M := E ⊗[ℤ_[p]] (ℤ_[p] ⊗[ℤ] L)), ← htop,
      finrank_biSup_indep E _ hind S]
    exact Finset.sum_congr rfl fun χ hχ => (hdim χ hχ).symm
  -- `#S ≤ (b₁ + 1)/2`
  have hcard : S.card ≤ (Module.finrank ℤ L + 1) / 2 := by
    have h2 : ∑ χ ∈ S, 2 ≤ ∑ χ ∈ S, (mult χ + if mult χ = 1 then 1 else 0) := by
      refine Finset.sum_le_sum fun χ hχ => ?_
      rcases hm χ hχ with h | h <;> simp [h]
    rw [Finset.sum_add_distrib, ← Finset.card_filter, hsum, Finset.sum_const, smul_eq_mul] at h2
    omega
  -- conclusion: `d + 1` independent elements would give a non-zero `g` killing every system
  intro v hv
  rw [← Fintype.not_linearIndependent_iff]
  intro hli
  have hlam : ∀ (χ : S) (i : Fin ((Module.finrank ℤ L + 1) / 2 + 1)), ∃ c : E,
      ∀ x ∈ W p E X χ, bcH p E (v i) x = c • x := fun χ i => scalar_adj χ.1 (hv i)
  choose lam hlamx using hlam
  let Ψ : (Fin ((Module.finrank ℤ L + 1) / 2 + 1) → E) →ₗ[E] (S → E) :=
    LinearMap.pi fun χ => ∑ i, (lam χ i) • LinearMap.proj i
  have hker : LinearMap.ker Ψ ≠ ⊥ := by
    apply LinearMap.ker_ne_bot_of_finrank_lt
    rw [Module.finrank_fintype_fun_eq_card, Module.finrank_fintype_fun_eq_card, Fintype.card_fin, Fintype.card_coe]
    omega
  obtain ⟨g, hgker, hg0⟩ := Submodule.exists_mem_ne_zero_of_ne_bot hker
  have hΨ : ∀ χ : S, ∑ i, lam χ i * g i = 0 := by
    intro χ
    have := congrFun (LinearMap.mem_ker.mp hgker) χ
    simpa [Ψ, mul_comm] using this
  apply hg0
  apply linearIndependent_bcH (p := p) (E := E) hli g
  -- `u = Σ g_i • bcH(v_i)` vanishes on every `W_χ`, which span `V`
  have hle : (⨆ χ ∈ S, W p E X χ) ≤ LinearMap.ker (∑ i, g i • bcH p E (v i)) := by
    refine iSup₂_le fun χ hχ x hx => LinearMap.mem_ker.mpr ?_
    simp only [LinearMap.sum_apply, LinearMap.smul_apply]
    have : ∀ i, g i • bcH p E (v i) x = (g i * lam ⟨χ, hχ⟩ i) • x := fun i => by
      rw [hlamx ⟨χ, hχ⟩ i x hx, smul_smul]
    simp_rw [this, ← Finset.sum_smul]
    rw [show ∑ i, g i * lam ⟨χ, hχ⟩ i = 0 by simpa [mul_comm] using hΨ ⟨χ, hχ⟩, zero_smul]
  rw [htop, top_le_iff, LinearMap.ker_eq_top] at hle
  exact hle
end

end GL2.HeckeZ.RankES
