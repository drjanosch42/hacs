/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import Mathlib
import GL2.HeckeZ.Bridge

/-!
# HeckeZ HZ-0 at level `N`, two systems: an Eisenstein split from the rank alone (L-HZ3, M-HZ4b, partly)

`07-lean-plan.md` §2.1 (HZ-0), §3 (`hES_N_an`; v0.3 P2a-6: "the semisimplicity sits inside the literature proof of `hES_N_an`'s
rank"), §10 (M-HZ4b). `L` is a free `ℤ`-module of finite rank, `X : ι → End_ℤ L` a family (interface data at level `N`:
`L = H¹(Γ₀(N); Sym^{k−2} ℤ)/tors`, `X n = T_n` for `(n, N) = 1`), `𝕋 = ℤ[X] = Algebra.adjoin ℤ (range X)`
(`GL2.HeckeZ.Bridge.Tfr`).

**The theorem** (`eisSplit_of_rank2`). If `rank_ℤ 𝕋 ≤ 2` and one generator `θ = X i₀` has integral characteristic polynomial
`(x − u)^a (x − w)^b` with `a, b ≥ 1` and `u ≠ w` (TB-χ), then `𝕋` has an Eisenstein split (`GL2.HeckeZ.EisSplit`) with this `θ`,
`λ(θ) = w`, `μ(θ) = u`. Nothing else is assumed: no semisimplicity (`hSemisimple_an` is not needed), no field, no `ℤ_p`.

**Proof** (07's degree count, P2a-6, made explicit):
1. `u`, `w` are roots of `χ(θ)`, hence eigenvalues of `θ` over `ℤ` (`Module.End.hasEigenvalue_iff_isRoot_charpoly`): integral
   eigenvectors `v_u`, `v_w ≠ 0`.
2. `1, θ` are `ℤ`-independent (on `v_u`, `v_w` a relation `α + βθ = 0` gives `α + βu = α + βw = 0`, so `β = α = 0`); `1, θ, t`
   are dependent for every `t ∈ 𝕋` (three vectors in rank `≤ 2`). So `c·t = α + βθ` with `c ≠ 0` (`dep`).
3. For `t = θ²`: `c θ² = α + βθ` and the two eigenvectors force `β = c(u + w)`, `α = −cuw`, so `c(θ − u)(θ − w) = 0`, and
   `(θ − u)(θ − w) = 0` since `End_ℤ L` is torsion-free: `Q ∘ P = 0`.
4. On `ker(θ − w)`, `c·t` acts by `α + βw`; on `v_w` that eigenvalue divided by `c` is an integer (`Bridge.int_of_eigen`, an
   eigenvalue of an integral endomorphism on an integral eigenvector), so every `t` acts on `ker(θ − w)` by an integer `λ(t)`,
   and `λ = evHom` is a ring map; likewise `μ` on `ker(θ − u)`.
-/

namespace GL2.HeckeZ.BridgeN

open GL2.HeckeZ GL2.HeckeZ.Bridge Polynomial

variable {L : Type*} [AddCommGroup L] [Module.Free ℤ L] [Module.Finite ℤ L] {ι : Type*}

/-- **Integrality of an eigenvalue.** If `c • f v = m • v` with `c ≠ 0`, `v ≠ 0`, then `f v = n • v` for an integer `n`
(via `Bridge.int_eigen_vec` over `ℚ_2`; any prime would do). -/
lemma int_eigen_div (f : Module.End ℤ L) {v : L} (hv : v ≠ 0) {c m : ℤ} (hc : c ≠ 0) (h : c • f v = m • v) :
    ∃ n : ℤ, f v = n • v := by
  have hc' : ((c : ℚ_[2])) ≠ 0 := by exact_mod_cast hc
  have he : emb 2 ℚ_[2] (f v) = ((m : ℚ_[2]) / c) • emb 2 ℚ_[2] v := by
    have h1 := congrArg (emb 2 ℚ_[2]) h
    rw [emb_zsmul, emb_zsmul] at h1
    rw [div_eq_inv_mul, mul_smul, ← h1, smul_smul, inv_mul_cancel₀ hc', one_smul]
  obtain ⟨n, -, hn⟩ := int_eigen_vec (p := 2) (E := ℚ_[2]) f hv he
  exact ⟨n, hn⟩

omit [Module.Finite ℤ L] in
/-- In a torsion-free group `c • x = 0`, `c ≠ 0` forces `x = 0`. -/
lemma zsmul_cancel {x : L} {c : ℤ} (hc : c ≠ 0) (h : c • x = 0) : x = 0 :=
  (smul_eq_zero.mp h).resolve_left hc

omit [Module.Finite ℤ L] in
/-- `End_ℤ L` is torsion-free. -/
lemma end_zsmul_cancel {F : Module.End ℤ L} {c : ℤ} (hc : c ≠ 0) (h : c • F = 0) : F = 0 :=
  LinearMap.ext fun v => zsmul_cancel hc (by simpa using congrArg (fun G => G v) h)

/-- An integral eigenvector for a root of the characteristic polynomial. -/
lemma exists_eigvec (f : Module.End ℤ L) (u : ℤ) (hroot : f.charpoly.IsRoot u) : ∃ v : L, v ≠ 0 ∧ f v = u • v := by
  obtain ⟨v, hv⟩ := ((Module.End.hasEigenvalue_iff_isRoot_charpoly f u).mpr hroot).exists_hasEigenvector
  exact ⟨v, hv.2, Module.End.mem_eigenspace_iff.mp hv.1⟩

variable (X : ι → Module.End ℤ L)

instance tfr_finite : Module.Finite ℤ (Tfr X) :=
  Module.IsNoetherian.finite ℤ (Tfr X).toSubmodule

/-- **The dependence** (step 2): every `t ∈ 𝕋` satisfies `c • t = α • 1 + β • θ` with `c ≠ 0`, given `rank 𝕋 ≤ 2` and two
distinct integral eigenvalues `u ≠ w` of `θ = X i₀`. -/
lemma dep (hR : Module.finrank ℤ (Tfr X) ≤ 2) (i₀ : ι) {u w : ℤ} (huw : u ≠ w) {vu vw : L} (hvu : vu ≠ 0) (hvw : vw ≠ 0)
    (eu : X i₀ vu = u • vu) (ew : X i₀ vw = w • vw) (t : Tfr X) :
    ∃ c α β : ℤ, c ≠ 0 ∧ c • (t : Module.End ℤ L) = α • (1 : Module.End ℤ L) + β • X i₀ := by
  have hni : ¬ LinearIndependent ℤ ![(1 : Tfr X), gen X i₀, t] := fun hli => by
    have := hli.fintype_card_le_finrank; simp at this; omega
  obtain ⟨g, hg, i, hi⟩ := Fintype.not_linearIndependent_iff.mp hni
  rw [Fin.sum_univ_three] at hg
  simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two, Matrix.tail_cons, Matrix.head_cons] at hg
  have hgE : g 0 • (1 : Module.End ℤ L) + g 1 • X i₀ + g 2 • (t : Module.End ℤ L) = 0 := by
    have := congrArg (fun s : Tfr X => (s : Module.End ℤ L)) hg
    simpa [gen] using this
  by_cases h2 : g 2 = 0
  · exfalso
    rw [h2, zero_smul, add_zero] at hgE
    have ev : ∀ {v : L} {a : ℤ}, v ≠ 0 → X i₀ v = a • v → g 0 + g 1 * a = 0 := by
      intro v a hv ea
      have h1 := congrArg (fun F : Module.End ℤ L => F v) hgE
      simp only [LinearMap.add_apply, LinearMap.smul_apply, Module.End.one_apply, LinearMap.zero_apply] at h1
      rw [ea] at h1
      have h2 : (g 0 + g 1 * a) • v = 0 := by rw [add_smul, mul_smul]; exact h1
      exact (smul_eq_zero.mp h2).resolve_right hv
    have h0u := ev hvu eu
    have h0w := ev hvw ew
    have h1 : g 1 = 0 := by
      have : g 1 * (u - w) = 0 := by linear_combination h0u - h0w
      exact (mul_eq_zero.mp this).resolve_right (sub_ne_zero.mpr huw)
    have h0 : g 0 = 0 := by rw [h1] at h0u; simpa using h0u
    rcases i with ⟨_ | _ | _ | n, hn⟩
    · exact hi h0
    · exact hi h1
    · exact hi h2
    · omega
  · refine ⟨g 2, -g 0, -g 1, h2, ?_⟩
    rw [eq_neg_of_add_eq_zero_right hgE]; module

/-- `θ² ∈ 𝕋`. -/
def genSq (i₀ : ι) : Tfr X := gen X i₀ * gen X i₀

/-- **`(θ − u)(θ − w) = 0`** (step 3). -/
lemma quad (hR : Module.finrank ℤ (Tfr X) ≤ 2) (i₀ : ι) {u w : ℤ} (huw : u ≠ w) {vu vw : L} (hvu : vu ≠ 0)
    (hvw : vw ≠ 0) (eu : X i₀ vu = u • vu) (ew : X i₀ vw = w • vw) (v : L) :
    X i₀ (X i₀ v - u • v) - w • (X i₀ v - u • v) = 0 := by
  obtain ⟨c, α, β, hc, h⟩ := dep X hR i₀ huw hvu hvw eu ew (genSq X i₀)
  have hsq : ((genSq X i₀ : Tfr X) : Module.End ℤ L) = X i₀ * X i₀ := rfl
  rw [hsq] at h
  have ev : ∀ {y : L} {a : ℤ}, y ≠ 0 → X i₀ y = a • y → c * a ^ 2 = α + β * a := by
    intro y a hy ea
    have h1 := congrArg (fun F : Module.End ℤ L => F y) h
    simp only [LinearMap.smul_apply, Module.End.mul_apply, LinearMap.add_apply, Module.End.one_apply] at h1
    rw [ea, map_zsmul, ea] at h1
    have h2 : (c * a ^ 2 - (α + β * a)) • y = 0 := by
      have e : (c * a ^ 2 - (α + β * a)) • y = c • a • a • y - (α • y + β • a • y) := by module
      rw [e, h1, sub_self]
    linear_combination (smul_eq_zero.mp h2).resolve_right hy
  have hu := ev hvu eu
  have hw := ev hvw ew
  have hβ : β = c * (u + w) := by
    have : (u - w) * (c * (u + w) - β) = 0 := by linear_combination hu - hw
    exact (sub_eq_zero.mp ((mul_eq_zero.mp this).resolve_left (sub_ne_zero.mpr huw))).symm
  have hα : α = -(c * u * w) := by rw [hβ] at hu; linear_combination -hu
  have hv := congrArg (fun F : Module.End ℤ L => F v) h
  simp only [LinearMap.smul_apply, Module.End.mul_apply, LinearMap.add_apply, Module.End.one_apply] at hv
  apply zsmul_cancel hc
  rw [show X i₀ (X i₀ v - u • v) - w • (X i₀ v - u • v) = X i₀ (X i₀ v) - (u + w) • X i₀ v + (u * w) • v by
    rw [map_sub, map_zsmul]; module]
  rw [smul_add, smul_sub, hv, hα, hβ]; module

/-- Every `t ∈ 𝕋` acts on `ker(θ − a)` by an integer, provided `ker(θ − a)` contains a nonzero vector (step 4). -/
lemma scalar_on (hR : Module.finrank ℤ (Tfr X) ≤ 2) (i₀ : ι) {u w : ℤ} (huw : u ≠ w) {vu vw : L} (hvu : vu ≠ 0)
    (hvw : vw ≠ 0) (eu : X i₀ vu = u • vu) (ew : X i₀ vw = w • vw) {a : ℤ} {y : L} (hy : y ≠ 0) (ey : X i₀ y = a • y)
    (t : Tfr X) : ∃ n : ℤ, (t : Module.End ℤ L) y = n • y ∧ ∀ z : L, X i₀ z = a • z → (t : Module.End ℤ L) z = n • z := by
  obtain ⟨c, α, β, hc, h⟩ := dep X hR i₀ huw hvu hvw eu ew t
  have act : ∀ z : L, X i₀ z = a • z → c • (t : Module.End ℤ L) z = (α + β * a) • z := by
    intro z ez
    have h1 := congrArg (fun F : Module.End ℤ L => F z) h
    simp only [LinearMap.smul_apply, LinearMap.add_apply, Module.End.one_apply] at h1
    rw [h1, ez, smul_smul, add_smul]
  obtain ⟨n, hn⟩ := int_eigen_div (t : Module.End ℤ L) hy hc (act y ey)
  have hcn : c * n = α + β * a := by
    have h1 := act y ey
    rw [hn, smul_smul] at h1
    have : (c * n - (α + β * a)) • y = 0 := by rw [sub_smul, h1, sub_self]
    exact sub_eq_zero.mp ((smul_eq_zero.mp this).resolve_right hy)
  refine ⟨n, hn, fun z ez => ?_⟩
  have h1 := act z ez
  rw [← hcn, ← smul_smul] at h1
  exact sub_eq_zero.mp (zsmul_cancel hc (by rw [smul_sub, h1, sub_self]))

/-- **HZ-0 at level `N`, two systems** (07 §2.1, §3 P2a-6): from `rank_ℤ 𝕋 ≤ 2` and `χ_ℤ(θ) = (x − u)^a (x − w)^b`
(`a, b ≥ 1`, `u ≠ w`; TB-χ), an Eisenstein split of `𝕋 = ℤ[X]` with `θ = X i₀`, `λ(θ) = w`, `μ(θ) = u`. -/
theorem eisSplit_of_rank2 (hR : Module.finrank ℤ (Tfr X) ≤ 2) (i₀ : ι) (u w : ℤ) (huw : u ≠ w) (a b : ℕ)
    (ha : 1 ≤ a) (hb : 1 ≤ b) (hχ : (X i₀).charpoly = (Polynomial.X - C u) ^ a * (Polynomial.X - C w) ^ b) :
    ∃ Es : EisSplit (Tfr X), Es.θ = gen X i₀ ∧ Es.lam (gen X i₀) = w ∧ Es.mu (gen X i₀) = u := by
  obtain ⟨vu, hvu, eu⟩ := exists_eigvec (X i₀) u (by
    rw [hχ, IsRoot.def, eval_mul, eval_pow, eval_sub, eval_X, eval_C, sub_self, zero_pow (by omega), zero_mul])
  obtain ⟨vw, hvw, ew⟩ := exists_eigvec (X i₀) w (by
    rw [hχ, IsRoot.def, eval_mul, eval_pow (n := b), eval_sub, eval_X, eval_C, sub_self, zero_pow (by omega), mul_zero])
  have hexW : ∀ t : Tfr X, ∃ n : ℤ, (t : Module.End ℤ L) vw = n • vw := fun t =>
    (scalar_on X hR i₀ huw hvu hvw eu ew hvw ew t).imp fun _ h => h.1
  have hexU : ∀ t : Tfr X, ∃ n : ℤ, (t : Module.End ℤ L) vu = n • vu := fun t =>
    (scalar_on X hR i₀ huw hvu hvw eu ew hvu eu t).imp fun _ h => h.1
  let lam := evHom hvw hexW
  let mu := evHom hvu hexU
  have hθ : ((gen X i₀ : Tfr X) : Module.End ℤ L) = X i₀ := rfl
  have lθ : lam (gen X i₀) = w := evHom_eq hvw hexW _ (by rw [hθ]; exact ew)
  have mθ : mu (gen X i₀) = u := evHom_eq hvu hexU _ (by rw [hθ]; exact eu)
  -- every `t` acts on `ker(θ − w)` by `λ t` and on `ker(θ − u)` by `μ t`
  have actW : ∀ (t : Tfr X) (z : L), X i₀ z = w • z → (t : Module.End ℤ L) z = lam t • z := by
    intro t z ez
    obtain ⟨n, hn, hall⟩ := scalar_on X hR i₀ huw hvu hvw eu ew hvw ew t
    rw [hall z ez, evHom_eq hvw hexW t hn]
  have actU : ∀ (t : Tfr X) (z : L), X i₀ z = u • z → (t : Module.End ℤ L) z = mu t • z := by
    intro t z ez
    obtain ⟨n, hn, hall⟩ := scalar_on X hR i₀ huw hvu hvw eu ew hvu eu t
    rw [hall z ez, evHom_eq hvu hexU t hn]
  refine ⟨{ lam := lam, mu := mu, θ := gen X i₀, hD := ?_, hQP := ?_, hA := ?_, hB := ?_ }, rfl, lθ, mθ⟩
  · rw [lθ, mθ]; exact huw.symm
  · intro v; rw [lθ, mθ, hθ]; exact quad X hR i₀ huw hvu hvw eu ew v
  · intro t z hz; rw [lθ, hθ, sub_eq_zero] at hz; exact actW t z hz
  · intro t z hz; rw [mθ, hθ, sub_eq_zero] at hz; exact actU t z hz

end GL2.HeckeZ.BridgeN
