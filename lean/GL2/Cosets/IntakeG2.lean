import GL2.Cosets.Routes
import GL2.Cosets.Boundary

/-!
# The G2 intake, continued (L18-2): `mu_term_assembled`, `gamma_cusp`

[GT] `prop:mu-assembled`, `prop:bdrycosets-cusp`, mirroring [C] v1.17.0 `prop:mu`(1) (the factors of the closed form) and
`prop:bdrycosets`(2) (the cusp characterisation `γ_j · ∞ = δ_j⁻¹ · ∞`), the carried L17-3 audit items MINOR 6 and MINOR 3.

* `mu_term_assembled` -- for every `γ ∈ Γ` the index `a = 1 · γ` of its `Γ⁰(N)`-coset is unique (`tcosets_transversal`),
  `γ t_a⁻¹ ∈ Γ⁰(N)`, and the closed-form factor `g γ t_a⁻¹` is integral of determinant `N` with
  `ρ(g γ t_a⁻¹) ρ(t_a) = ρ(g γ)` (`mu_term`): [C]'s "all factors integral of determinant `N`, no determinant-`N` matrix
  inverted", assembled.
* `gamma_cusp` -- `δ γ` lies in `B_N` (is upper triangular) iff the columns `(a, c)` of `γ` and `(d_δ, -c_δ)` of
  `adj δ` are proportional, i.e. `γ · ∞ = δ⁻¹ · ∞` in `ℙ¹(ℚ)` (stated as a ratio pair, no `OnePoint ℚ`).
-/

namespace GL2.Cosets

open Matrix CongruenceSubgroup
open scoped MatrixGroups

variable {R : Type*} [CommRing R] {p ν : ℕ}

/-- **[C] `prop:mu`(1), the factors assembled** (L17-3 MINOR 6). -/
theorem mu_term_assembled (hp : p.Prime) (hν : 1 ≤ ν) (n : ℕ) (γ : SL(2, ℤ)) :
    ∃! k, k < mIdx p ν ∧ γ * (tSL p ν k)⁻¹ ∈ Gamma0Up ((p : ℤ) ^ ν) ∧
      (gMat ((p : ℤ) ^ ν) * (γ : Mat2 ℤ) * ((tSL p ν k)⁻¹ : SL(2, ℤ))).det = (p : ℤ) ^ ν ∧
      rhoR R n (gMat ((p : ℤ) ^ ν) * (γ : Mat2 ℤ) * ((tSL p ν k)⁻¹ : SL(2, ℤ))) * rhoR R n (tSL p ν k : Mat2 ℤ) =
        rhoR R n (gMat ((p : ℤ) ^ ν) * (γ : Mat2 ℤ)) := by
  obtain ⟨k, hk, huniq⟩ := tcosets_transversal hp hν γ
  have hm := mu_term (R := R) p ν n (γ : Mat2 ℤ) (tSL p ν k)
  refine ⟨k, ⟨hk.1, hk.2, ?_, hm.2⟩, fun k' h' => huniq k' ⟨h'.1, h'.2.1⟩⟩
  rw [hm.1, Matrix.SpecialLinearGroup.det_coe, mul_one]

/-- **[C] `prop:bdrycosets`(2), the cusp characterisation** (L17-3 MINOR 3): `δ_k γ ∈ B_N` iff `γ · ∞ = δ_k⁻¹ · ∞`,
written as proportionality of `(γ₀₀, γ₁₀)` and `(δ₁₁, -δ₁₀)` (the first column of `adj δ_k`). -/
theorem gamma_cusp (p ν k : ℕ) (γ : SL(2, ℤ)) :
    deltaMat p ν k * (γ : Mat2 ℤ) ∈ BN ((p : ℤ) ^ ν) ↔
      γ 0 0 * (-(deltaMat p ν k 1 0)) - γ 1 0 * deltaMat p ν k 1 1 = 0 := by
  have hDC : deltaMat p ν k * (γ : Mat2 ℤ) ∈ DC ((p : ℤ) ^ ν) :=
    ⟨(sSL p ν k)⁻¹, γ, by rw [deltaMat]⟩
  have h10 : (deltaMat p ν k * (γ : Mat2 ℤ)) 1 0 = deltaMat p ν k 1 0 * γ 0 0 + deltaMat p ν k 1 1 * γ 1 0 := by
    rw [Matrix.mul_apply, Fin.sum_univ_two]
  constructor
  · rintro ⟨-, h⟩; rw [h10] at h; linarith
  · intro h; exact ⟨hDC, by rw [h10]; linarith⟩

end GL2.Cosets
