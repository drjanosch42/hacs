import Mathlib
import GL2.Conventions.SymPowMatrix
import GL2.Cosets.DoubleCoset

/-!
# G1 (L17-3): the two evaluation routes and the middle map ([C] `prop:tworoutes`, `prop:mu`)

Both propositions are cochain-level identities whose cochain wrapping (the `τ`-table, equivariant cochains,
`Ind`) is G2.  What carries them is matrix algebra over `ℤ` followed by one reduction, and that is what is
proved here, for an **arbitrary** stored datum: a family of terms `(n_t, γ_t, c_t)` per coset index `k`, over
any commutative ring `R`, at any `n`, with no hypothesis on `ℓ`.  So the identities hold for the `τ`-table
whatever it is, which is the sense of "term by term, as an identity of matrices".

`ρ_R(A) := ρ_n(A mod R)` is the coordinate matrix of [C] `eq:rhondef` of the reduction of an integral `A`.
-/

namespace GL2.Cosets

open Matrix CongruenceSubgroup
open scoped MatrixGroups

variable {R : Type*} [CommRing R]

/-- `ρ_n` of an integral matrix, reduced into `R`. -/
noncomputable def rhoR (R : Type*) [CommRing R] (n : ℕ) (A : Mat2 ℤ) : Matrix (Fin (n + 1)) (Fin (n + 1)) R :=
  rhoMat n (A.map (Int.castRingHom R))

theorem rhoR_mul (n : ℕ) (A B : Mat2 ℤ) : rhoR R n (A * B) = rhoR R n A * rhoR R n B := by
  rw [rhoR, Matrix.map_mul, rhoMat_mul]; rfl

/-- `ρ_R` is the reduction of `ρ_ℤ` ([C] `prop:symconv`(1), (2)): reducing once at the end is the same as
reducing the factors. -/
theorem rhoR_eq_map (n : ℕ) (A : Mat2 ℤ) : rhoR R n A = (rhoMat n A).map (Int.castRingHom R) := by
  rw [rhoR, rhoMat_map]

/-- One stored term `n_t · γ_t · c_t` of a slot. -/
structure Term (R : Type*) (n : ℕ) where
  coeff : ℤ
  gam : Mat2 ℤ
  val : Fin (n + 1) → R

variable (p ν : ℕ)

/-- The **fused** route `Σ_k ρ(s_k⁻¹) Σ_t n_t ρ(g γ_t) c_t` ([C] `def:xi`, `ω_• ∘ Ξ_g`). -/
noncomputable def fusedRoute (n : ℕ) (m : ℕ) (T : ℕ → List (Term R n)) : Fin (n + 1) → R :=
  ∑ k ∈ Finset.range m, (rhoR R n ((sSL p ν k)⁻¹ : SL(2, ℤ))).mulVec
    ((T k).map (fun t => (t.coeff : R) • (rhoR R n (gMat ((p : ℤ) ^ ν) * t.gam)).mulVec t.val)).sum

/-- **Route 2**, the direct formula `Σ_k Σ_t n_t ρ(δ_k γ_t) c_t` ([C] `eq:heckechain`), the product
`δ_k γ_t` formed exactly in `Mat₂(ℤ)` and reduced once. -/
noncomputable def directRoute (n : ℕ) (m : ℕ) (T : ℕ → List (Term R n)) : Fin (n + 1) → R :=
  ∑ k ∈ Finset.range m,
    ((T k).map (fun t => (t.coeff : R) • (rhoR R n (deltaMat p ν k * t.gam)).mulVec t.val)).sum

/-- **[C] `prop:tworoutes`, term by term:** `δ_k γ = s_k⁻¹ (g γ)` in `Mat₂(ℤ)`, and hence
`ρ(δ_k γ) = ρ(s_k⁻¹) ρ(g γ)` as matrices over every `R`. -/
theorem tworoutes_term (n k : ℕ) (γ : Mat2 ℤ) :
    deltaMat p ν k * γ = (((sSL p ν k)⁻¹ : SL(2, ℤ)) : Mat2 ℤ) * (gMat ((p : ℤ) ^ ν) * γ) ∧
    rhoR R n (deltaMat p ν k * γ) =
      rhoR R n (((sSL p ν k)⁻¹ : SL(2, ℤ)) : Mat2 ℤ) * rhoR R n (gMat ((p : ℤ) ^ ν) * γ) := by
  have h : deltaMat p ν k * γ = (((sSL p ν k)⁻¹ : SL(2, ℤ)) : Mat2 ℤ) * (gMat ((p : ℤ) ^ ν) * γ) := by
    rw [deltaMat, Matrix.mul_assoc]
  exact ⟨h, by rw [h, rhoR_mul]⟩

/-- **[C] `prop:tworoutes`:** the fused route and route 2 agree on every stored datum, over every
commutative ring, with no hypothesis on `ℓ`. -/
theorem tworoutes (n m : ℕ) (T : ℕ → List (Term R n)) : fusedRoute p ν n m T = directRoute p ν n m T := by
  unfold fusedRoute directRoute
  refine Finset.sum_congr rfl (fun k _ => ?_)
  induction T k with
  | nil => simp
  | cons t l ih =>
    simp only [List.map_cons, List.sum_cons, Matrix.mulVec_add, Matrix.mulVec_smul] at ih ⊢
    rw [ih, (tworoutes_term p ν n k t.gam).2, ← Matrix.mulVec_mulVec]

/-! ## The middle map `μ_•` ([C] `prop:mu`) -/

/-- The upper-triangular transporters `t_k` of [C] `def:cosetreps` (0-based):
`t_k = T^k` for `k < N`, `t_{N + c} = [[-p c, -1], [1, 0]]`. -/
def tMat (k : ℕ) : Mat2 ℤ :=
  if k < p ^ ν then !![1, (k : ℤ); 0, 1] else !![-((p : ℤ) * ((k : ℤ) - (p : ℤ) ^ ν)), -1; 1, 0]

def tSL (k : ℕ) : SL(2, ℤ) :=
  ⟨tMat p ν k, by unfold tMat; split_ifs <;> simp [Matrix.det_fin_two_of]⟩

lemma tSL_coe (k : ℕ) : ((tSL p ν k : SL(2, ℤ)) : Mat2 ℤ) = tMat p ν k := rfl

/-- The class of `γ` modulo `Γ⁰(N)` is the class of `t_k` iff `RowRel` holds on the top row `(b, a)`. -/
theorem mul_tInv_mem_Gamma0Up_iff (γ : SL(2, ℤ)) (k : ℕ) :
    γ * (tSL p ν k)⁻¹ ∈ Gamma0Up ((p : ℤ) ^ ν) ↔ RowRel p ν k (γ 0 1) (γ 0 0) := by
  rw [mem_Gamma0Up, Matrix.SpecialLinearGroup.coe_mul, coe_inv_apply, tSL_coe]
  unfold RowRel tMat
  split_ifs
  · simp [Matrix.mul_apply, Fin.sum_univ_two]
    rw [neg_add_eq_sub]
  · simp [Matrix.mul_apply, Fin.sum_univ_two]
    rw [← dvd_neg]; ring_nf

variable {p ν}

/-- **[C] `def:cosetreps`, the `t_k`:** they are a transversal of `Γ⁰(N)\Γ`, `m` of them. This is the
`a_t = 1 · γ_t` of [C] `prop:mu`(1). -/
theorem tcosets_transversal (hp : p.Prime) (hν : 1 ≤ ν) (γ : SL(2, ℤ)) :
    ∃! k, k < mIdx p ν ∧ γ * (tSL p ν k)⁻¹ ∈ Gamma0Up ((p : ℤ) ^ ν) := by
  simp_rw [mul_tInv_mem_Gamma0Up_iff]
  exact rowRel_existsUnique hp hν (isCoprime_top γ).symm

variable (p ν)

/-- **[C] `prop:mu`(1), the closed form, one term:** `g γ t⁻¹` is integral of determinant `N`, no
determinant-`N` matrix is inverted, and `ρ(g γ t⁻¹) ρ(t) = ρ(g γ)` ([C] `prop:mu`(3)'s computation). -/
theorem mu_term (n : ℕ) (γ : Mat2 ℤ) (t : SL(2, ℤ)) :
    (gMat ((p : ℤ) ^ ν) * γ * ((t⁻¹ : SL(2, ℤ)) : Mat2 ℤ)).det = (p : ℤ) ^ ν * γ.det ∧
    rhoR R n (gMat ((p : ℤ) ^ ν) * γ * ((t⁻¹ : SL(2, ℤ)) : Mat2 ℤ)) * rhoR R n (t : Mat2 ℤ) =
      rhoR R n (gMat ((p : ℤ) ^ ν) * γ) := by
  refine ⟨?_, ?_⟩
  · rw [Matrix.det_mul, Matrix.det_mul, Matrix.SpecialLinearGroup.det_coe, gMat_det, mul_one]
  · rw [← rhoR_mul, Matrix.mul_assoc, inv_mul_coe, Matrix.mul_one]

/-- `Ξ_g` on one slot ([C] `eq:xiexplicit`): `Σ_t n_t ρ(g γ_t) c_t`. -/
noncomputable def xiSlot (n : ℕ) (L : List (Term R n)) : Fin (n + 1) → R :=
  (L.map (fun t => (t.coeff : R) • (rhoR R n (gMat ((p : ℤ) ^ ν) * t.gam)).mulVec t.val)).sum

/-- **[C] `prop:mu`(3), the form used:** with `α_•` applied to the slot's own values,
`Σ_t n_t ρ(g γ_t t_{a_t}⁻¹) ρ(t_{a_t}) c_t = Σ_t n_t ρ(g γ_t) c_t`. -/
theorem mu_alpha (n : ℕ) (a : Mat2 ℤ → SL(2, ℤ)) (L : List (Term R n)) :
    (L.map (fun t => (t.coeff : R) •
      (rhoR R n (gMat ((p : ℤ) ^ ν) * t.gam * (((a t.gam)⁻¹ : SL(2, ℤ)) : Mat2 ℤ))).mulVec
        ((rhoR R n ((a t.gam : SL(2, ℤ)) : Mat2 ℤ)).mulVec t.val))).sum = xiSlot p ν n L := by
  unfold xiSlot
  refine congrArg List.sum (List.map_congr_left (fun t _ => ?_))
  rw [Matrix.mulVec_mulVec, (mu_term p ν n t.gam (a t.gam)).2]

/-- **[C] `prop:mu`(3), consequently:** `ω_• ∘ μ_• ∘ α_• = Θ_g` bit-identically: summing `ρ(s_k⁻¹)` times
the slot `k` of `μ_• α_•` gives the fused route, which is route 2 by `prop:tworoutes`. -/
theorem omega_mu_alpha (n m : ℕ) (a : ℕ → Mat2 ℤ → SL(2, ℤ)) (T : ℕ → List (Term R n)) :
    ∑ k ∈ Finset.range m, (rhoR R n ((sSL p ν k)⁻¹ : SL(2, ℤ))).mulVec
      ((T k).map (fun t => (t.coeff : R) •
        (rhoR R n (gMat ((p : ℤ) ^ ν) * t.gam * (((a k t.gam)⁻¹ : SL(2, ℤ)) : Mat2 ℤ))).mulVec
          ((rhoR R n ((a k t.gam : SL(2, ℤ)) : Mat2 ℤ)).mulVec t.val))).sum =
    directRoute p ν n m T := by
  rw [← tworoutes]
  unfold fusedRoute
  refine Finset.sum_congr rfl (fun k _ => ?_)
  rw [mu_alpha p ν n (a k) (T k)]
  rfl

end GL2.Cosets
