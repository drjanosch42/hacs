import GL2.Conventions.SymPow

/-!
# C2: the contragredient `ρ_n^∨(g) = ρ_n(adj g)^⊤` and the evaluation pairing

[GT] `conv:contragredient`, `prop:pairing`, mirroring [C] v1.15.0 `eq:adjrules`, `def:contragredient`,
`eq:dualaction`, `prop:pairing`(1),(3),(4),(5).

`D_n := Hom_R(Sym^n, R)` (`Module.Dual`), and `ρ_n^∨(g) φ := φ ∘ ρ_n(adj g)` (= `ρ_n(adj g)^⊤` in the dual
basis).  The pairing is evaluation `⟨v, φ⟩ = φ v`.
-/

namespace GL2

open Matrix

variable {R : Type*} [CommRing R]

/-! ### `eq:adjrules` -/

theorem adj_mul_self (g : Mat2 R) : g.adjugate * g = g.det • (1 : Mat2 R) := adjugate_mul g

theorem self_mul_adj (g : Mat2 R) : g * g.adjugate = g.det • (1 : Mat2 R) := mul_adjugate g

theorem adj_mul (g h : Mat2 R) : (g * h).adjugate = h.adjugate * g.adjugate := adjugate_mul_distrib g h

theorem adj_adj (g : Mat2 R) : g.adjugate.adjugate = g := by
  rw [adjugate_fin_two g, adjugate_fin_two_of]
  ext i j; fin_cases i <;> fin_cases j <;> simp

theorem adj_one : (1 : Mat2 R).adjugate = 1 := adjugate_one

/-! ### The contragredient -/

/-- `D_n = Hom_R(Sym^n, R)`. -/
abbrev DualN (R : Type*) [CommRing R] (n : ℕ) := Module.Dual R (SymN R n)

/-- **[C] `def:contragredient`**: `ρ_n^∨(g) := ρ_n(adj g)^⊤`, i.e. `(ρ_n^∨(g) φ)(v) = φ(ρ_n(adj g) v)`. -/
noncomputable def rhoDual (n : ℕ) (g : Mat2 R) : Module.End R (DualN R n) :=
  (rho n g.adjugate).dualMap

/-- **`eq:dualaction`** (definitional). -/
@[simp] theorem rhoDual_apply (n : ℕ) (g : Mat2 R) (φ : DualN R n) (v : SymN R n) :
    rhoDual n g φ v = φ (rho n g.adjugate v) := rfl

/-- **`prop:pairing`(1)**: `ρ_n^∨` is a monoid homomorphism on `Mat₂(R)`. -/
theorem rhoDual_mul (n : ℕ) (g h : Mat2 R) : rhoDual n (g * h) = rhoDual n g * rhoDual n h := by
  ext φ v
  simp [rhoDual_apply, adj_mul, rho_mul]

theorem rhoDual_one (n : ℕ) : rhoDual n (1 : Mat2 R) = 1 := by
  ext φ v
  simp [rho_one]

/-- **`prop:pairing`(1)**: on `SL₂`, `adj γ = γ⁻¹`, so `ρ_n^∨|_Γ` is the usual contragredient. -/
theorem adj_eq_inv_of_det_one (g : Mat2 R) (h : g.det = 1) : g.adjugate * g = 1 := by
  rw [adj_mul_self, h, one_smul]

/-- **`prop:pairing`(3), adjointness**: `⟨ρ_n(g) v, φ⟩ = ⟨v, ρ_n^∨(adj g) φ⟩`. -/
theorem pairing_adjoint (n : ℕ) (g : Mat2 R) (v : SymN R n) (φ : DualN R n) :
    φ (rho n g v) = rhoDual n g.adjugate φ v := by
  simp [adj_adj]

/-- **`prop:pairing`(4), twisted invariance**: `⟨ρ_n(g) v, ρ_n^∨(g) φ⟩ = (det g)^n ⟨v, φ⟩`. -/
theorem pairing_twisted (n : ℕ) (g : Mat2 R) (v : SymN R n) (φ : DualN R n) :
    rhoDual n g φ (rho n g v) = g.det ^ n * φ v := by
  rw [rhoDual_apply, ← Module.End.mul_apply, ← rho_mul, adj_mul_self, rho_smul_one]
  simp

/-- **`prop:pairing`(5)**: `adj(γ₁ g γ₂) = adj γ₂ · adj g · adj γ₁` (so `adj(Γ g Γ) = Γ (adj g) Γ`), and for
`g = diag(1,p)`: `adj g = diag(p,1) = S g S⁻¹` with `S = (0 −1; 1 0)`, `S⁻¹ = adj S`. -/
theorem adj_triple (a g b : Mat2 R) : (a * g * b).adjugate = b.adjugate * g.adjugate * a.adjugate := by
  rw [adj_mul, adj_mul, mul_assoc]

theorem adj_diag_eq_conj_S (p : R) :
    (!![1, 0; 0, p] : Mat2 R).adjugate = !![0, -1; 1, 0] * !![1, 0; 0, p] * (!![0, -1; 1, 0] : Mat2 R).adjugate ∧
    (!![0, -1; 1, 0] : Mat2 R) * (!![0, -1; 1, 0] : Mat2 R).adjugate = 1 := by
  constructor
  · simp [adjugate_fin_two_of]
  · simp [adjugate_fin_two_of]; ext i j; fin_cases i <;> fin_cases j <;> simp

/-- **`prop:pairing`(2), perfectness**: the evaluation pairing `Sym^n × D_n → R` is perfect over every
commutative ring: `Sym^n` is free of finite rank (`GL2.symBasis`), so `v ↦ (φ ↦ φ v)` is bijective onto
`Hom(D_n, R)` (mathlib `Module.bijective_dual_eval`). -/
theorem pairing_perfect (n : ℕ) : Function.Bijective (Module.Dual.eval R (SymN R n)) :=
  Module.bijective_dual_eval R (SymN R n)

end GL2
