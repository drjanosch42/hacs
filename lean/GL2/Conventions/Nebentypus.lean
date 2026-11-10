import GL2.Conventions.SymPow

/-!
# C8: the nebentypus convention ([C] `def:nebdata`, `lem:nebdata`, `prop:nebdecomp`(2), `prop:nebcorr`(3))

[GT] `conv:nebentypus`, mirroring [C] v1.15.0 `sec:nebentypus`.

**The convention, in one sentence.** [C]'s coefficient module `V(χ)` is `Sym^n` with `δ ∈ Δ₀(N)` acting by
`ρ_{n,χ}(δ) = χ(a_δ) · ρ_n(δ)` (the character is read off the **upper-left** entry `a`); on `Γ₀(N)` this is
`χ(d_γ)⁻¹ · ρ_n(γ)` (`rhoChi_Gamma0`), so as a `Γ₀(N)`-module [C]'s `V(χ)` is the Diamond–Shurman / Shimura / FLT
"`ψ(d_γ)`" module with `ψ = χ⁻¹` (`rhoChi_inv_Gamma0`); [C]'s diamond `⟨d⟩` acts by `χ(d)` on the character model
(`diamond_eigen`), and `T(p,p) = ρ_{n,χ}(p·I) = χ(p) p^n` (`rhoChi_smul_one`).

Everything here is elementary group theory over a commutative ring; nothing is assumed.
-/

namespace GL2.Neb

open MvPolynomial Matrix
open scoped MatrixGroups

variable {N : ℕ}

/-- `a_δ mod N` (upper-left entry). -/
def aMod (N : ℕ) (δ : Mat2 ℤ) : ZMod N := (δ 0 0 : ZMod N)

/-- `d_δ mod N` (lower-right entry). -/
def dMod (N : ℕ) (δ : Mat2 ℤ) : ZMod N := (δ 1 1 : ZMod N)

/-- `c_δ mod N` (lower-left entry). -/
def cMod (N : ℕ) (δ : Mat2 ℤ) : ZMod N := (δ 1 0 : ZMod N)

lemma aMod_mul (δ δ' : Mat2 ℤ) (h : cMod N δ' = 0) : aMod N (δ * δ') = aMod N δ * aMod N δ' := by
  simp only [aMod, cMod] at *
  simp [Matrix.mul_apply, Fin.sum_univ_two, h]

lemma cMod_mul (δ δ' : Mat2 ℤ) (h : cMod N δ = 0) (h' : cMod N δ' = 0) : cMod N (δ * δ') = 0 := by
  simp only [cMod] at *
  simp [Matrix.mul_apply, Fin.sum_univ_two, h, h']

lemma dMod_mul (δ δ' : Mat2 ℤ) (h : cMod N δ = 0) : dMod N (δ * δ') = dMod N δ * dMod N δ' := by
  simp only [dMod, cMod] at *
  simp [Matrix.mul_apply, Fin.sum_univ_two, h]

/-- [C] `def:nebdata`(2): the monoid `Δ₀(N) = {δ ∈ Mat₂(ℤ) : det δ > 0, N ∣ c_δ, gcd(a_δ, N) = 1}`. -/
def Delta0 (N : ℕ) : Submonoid (Mat2 ℤ) where
  carrier := {δ | 0 < δ.det ∧ cMod N δ = 0 ∧ IsUnit (aMod N δ)}
  mul_mem' := by
    rintro δ δ' ⟨h1, h2, h3⟩ ⟨h1', h2', h3'⟩
    refine ⟨by rw [det_mul]; positivity, cMod_mul δ δ' h2 h2', ?_⟩
    rw [aMod_mul δ δ' h2']; exact h3.mul h3'
  one_mem' := by simp [cMod, aMod]

/-- [C] `lem:nebdata`(2): `δ ↦ a_δ mod N` is a homomorphism of monoids `Δ₀(N) → ZMod N`. -/
def aHom (N : ℕ) : Delta0 N →* ZMod N where
  toFun δ := aMod N δ.1
  map_one' := by simp [aMod]
  map_mul' δ δ' := aMod_mul δ.1 δ'.1 δ'.2.2.1

/-- [C] `lem:nebdata`(1): on `Γ₀(N)`, `a_γ d_γ ≡ 1 (mod N)`. -/
theorem aMod_mul_dMod_Gamma0 (γ : SL(2, ℤ)) (hγ : γ ∈ CongruenceSubgroup.Gamma0 N) :
    aMod N γ * dMod N γ = 1 := by
  have hdet : ((γ : Mat2 ℤ).det : ZMod N) = 1 := by simp [γ.2]
  rw [det_fin_two] at hdet
  have hc : ((γ : Mat2 ℤ) 1 0 : ZMod N) = 0 := (CongruenceSubgroup.Gamma0_mem).1 hγ
  simp only [aMod, dMod]
  push_cast at hdet
  rw [hc, mul_zero, sub_zero] at hdet
  exact hdet

/-- `Γ₀(N) ⊆ Δ₀(N)`. -/
theorem Gamma0_mem_Delta0 (γ : SL(2, ℤ)) (hγ : γ ∈ CongruenceSubgroup.Gamma0 N) :
    (γ : Mat2 ℤ) ∈ Delta0 N :=
  ⟨by simp [γ.2], (CongruenceSubgroup.Gamma0_mem).1 hγ,
    IsUnit.of_mul_eq_one _ (aMod_mul_dMod_Gamma0 γ hγ)⟩

/-- On `Γ₀(N)`, `d_γ mod N` is mathlib's `CongruenceSubgroup.Gamma0Map` (= [C]'s `ς_N`). -/
theorem dMod_eq_Gamma0Map (γ : CongruenceSubgroup.Gamma0 N) :
    dMod N (γ : SL(2, ℤ)) = CongruenceSubgroup.Gamma0Map N γ := rfl

/-- The adjugate swaps the diagonal: `a_{adj δ} = d_δ` (used by `rmk:nebheilbronn` and the dictionary). -/
theorem aMod_adjugate (δ : Mat2 ℤ) : aMod N δ.adjugate = dMod N δ := by
  simp [aMod, dMod, adjugate_fin_two]

theorem dMod_adjugate (δ : Mat2 ℤ) : dMod N δ.adjugate = aMod N δ := by
  simp [aMod, dMod, adjugate_fin_two]

variable {R : Type*} [CommRing R]

/-- An integer matrix as a matrix over `R`. -/
abbrev toR (δ : Mat2 ℤ) : Mat2 R := δ.map (Int.castRingHom R)

/-- **[C] `def:nebdata`(5)**: `ρ_{n,χ}(δ) := χ(a_δ) · ρ_n(δ)` on `Sym^n(R²)`, for `δ ∈ Δ₀(N)` (in particular
`γ ∈ Γ₀(N)`).  `χ` is a Dirichlet character mod `N` with values in `R` (mathlib's `MulChar (ZMod N) R`). -/
noncomputable def rhoChi (n : ℕ) (χ : DirichletCharacter R N) (δ : Mat2 ℤ) : Module.End R (SymN R n) :=
  χ (aMod N δ) • rho n (toR δ)

/-- **[C] `lem:nebdata`(2)**: `ρ_{n,χ}` is multiplicative on `Δ₀(N)` (indeed whenever `N ∣ c_{δ'}`). -/
theorem rhoChi_mul (n : ℕ) (χ : DirichletCharacter R N) (δ δ' : Mat2 ℤ) (h' : cMod N δ' = 0) :
    rhoChi n χ (δ * δ') = rhoChi n χ δ * rhoChi n χ δ' := by
  simp only [rhoChi, aMod_mul δ δ' h', map_mul, Matrix.map_mul, rho_mul, smul_mul_smul_comm]

/-- `rhoChi` restricted to `Δ₀(N)`, as a monoid homomorphism. -/
noncomputable def rhoChiHom (n : ℕ) (χ : DirichletCharacter R N) : Delta0 N →* Module.End R (SymN R n) where
  toFun δ := rhoChi n χ δ.1
  map_one' := by
    simp only [rhoChi, OneMemClass.coe_one, Matrix.map_one _ (map_zero _) (map_one _), rho_one]
    simp [aMod]
  map_mul' δ δ' := rhoChi_mul n χ δ.1 δ'.1 δ'.2.2.1

/-- **[C] `lem:nebdata`(1),(2)**: on `Γ₀(N)`, `χ(a_γ) · χ(d_γ) = 1`. -/
theorem chi_a_mul_chi_d (χ : DirichletCharacter R N) (γ : SL(2, ℤ))
    (hγ : γ ∈ CongruenceSubgroup.Gamma0 N) : χ (aMod N γ) * χ (dMod N γ) = 1 := by
  rw [← map_mul, aMod_mul_dMod_Gamma0 γ hγ, map_one]

/-- **[C] `lem:nebdata`(2)**: on `Γ₀(N)`, `χ(a_γ) = χ⁻¹(d_γ)` (`χ⁻¹` the inverse character). -/
theorem chi_a_eq_inv_chi_d (χ : DirichletCharacter R N) (γ : SL(2, ℤ))
    (hγ : γ ∈ CongruenceSubgroup.Gamma0 N) : χ (aMod N γ) = χ⁻¹ (dMod N γ) := by
  have hm := chi_a_mul_chi_d χ γ hγ
  have hu : IsUnit (χ (dMod N γ)) := IsUnit.of_mul_eq_one_right _ hm
  rw [MulChar.inv_apply_eq_inv]
  calc χ (aMod N γ) = χ (aMod N γ) * (χ (dMod N γ) * Ring.inverse (χ (dMod N γ))) := by
        rw [Ring.mul_inverse_cancel _ hu, mul_one]
    _ = Ring.inverse (χ (dMod N γ)) := by rw [← mul_assoc, hm, one_mul]

/-- **`conv:nebentypus`, module level.** On `Γ₀(N)`: `ρ_{n,χ}(γ) = χ⁻¹(d_γ) · ρ_n(γ)`. -/
theorem rhoChi_Gamma0 (n : ℕ) (χ : DirichletCharacter R N) (γ : SL(2, ℤ))
    (hγ : γ ∈ CongruenceSubgroup.Gamma0 N) :
    rhoChi n χ γ = χ⁻¹ (dMod N γ) • rho n (toR (γ : Mat2 ℤ)) := by
  rw [rhoChi, chi_a_eq_inv_chi_d χ γ hγ]

/-- **`conv:nebentypus`, the dictionary to Diamond–Shurman / Shimura / FLT (module level).**
The DS/FLT nebentypus module "`γ ↦ ψ(d_γ) ρ_n(γ)`" (FLT `gamma0NebenRep … e` with `ψ = ω^e`) is [C]'s
`V(ψ⁻¹)`: reading an FLT/DS statement with `ψ` into [C] **inverts the character**. -/
theorem rhoChi_inv_Gamma0 (n : ℕ) (ψ : DirichletCharacter R N) (γ : SL(2, ℤ))
    (hγ : γ ∈ CongruenceSubgroup.Gamma0 N) :
    rhoChi n ψ⁻¹ γ = ψ (dMod N γ) • rho n (toR (γ : Mat2 ℤ)) := by
  rw [rhoChi_Gamma0 n ψ⁻¹ γ hγ, inv_inv]

/-- **[C] `prop:nebcorr`(3)**: `T(p,p) = [Γ₀(N) pI Γ₀(N)] = ρ_{n,χ}(p I) = χ(p) p^n` (one coset, central). -/
theorem rhoChi_smul_one (n : ℕ) (χ : DirichletCharacter R N) (p : ℤ) :
    rhoChi n χ (p • (1 : Mat2 ℤ)) = (χ (p : ZMod N) * (p : R) ^ n) • (1 : Module.End R (SymN R n)) := by
  have hmap : toR (p • (1 : Mat2 ℤ)) = (p : R) • (1 : Mat2 R) := by
    ext i j
    simp only [Matrix.map_apply, Matrix.smul_apply, Matrix.one_apply, smul_eq_mul, Int.coe_castRingHom]
    split_ifs <;> simp
  rw [rhoChi, hmap, rho_smul_one, smul_smul]
  simp [aMod]

/-- **[C] `lem:nebdata`(3)**: `-I` acts on `V(χ)` by the scalar `χ(-1)(-1)^n`. -/
theorem rhoChi_neg_one (n : ℕ) (χ : DirichletCharacter R N) :
    rhoChi n χ (-1 : Mat2 ℤ) = (χ (-1) * (-1 : R) ^ n) • (1 : Module.End R (SymN R n)) := by
  have := rhoChi_smul_one n χ (-1)
  simpa using this

/-- **[C] `prop:nebdecomp`(2) (the diamond reads `χ(d)`).** Let `F : SL₂(ℤ) → Sym^n` be an element of
`Ind_{Γ₀(N)}^{SL₂(ℤ)} V(χ)` in the function picture, `F(σ x) = ρ_{n,χ}(σ) F(x)` for `σ ∈ Γ₀(N)`.  Then [C]'s
`R_σ F (x) := ρ_n(σ) F(σ⁻¹ x)` equals `χ(d_σ) F(x)`: **`⟨d⟩ = χ(d)` on the character model.** -/
theorem diamond_eigen (n : ℕ) (χ : DirichletCharacter R N) (F : SL(2, ℤ) → SymN R n)
    (hF : ∀ σ ∈ CongruenceSubgroup.Gamma0 N, ∀ x, F (σ * x) = rhoChi n χ σ (F x))
    (σ : SL(2, ℤ)) (hσ : σ ∈ CongruenceSubgroup.Gamma0 N) (x : SL(2, ℤ)) :
    rho n (toR (σ : Mat2 ℤ)) (F (σ⁻¹ * x)) = χ (dMod N σ) • F x := by
  have h := hF σ hσ (σ⁻¹ * x)
  rw [← mul_assoc, mul_inv_cancel, one_mul, rhoChi, LinearMap.smul_apply] at h
  rw [h, smul_smul, mul_comm, chi_a_mul_chi_d χ σ hσ, one_smul]

/-- The involution `ε = diag(1,-1)` of the WP-9 audit's squared-form proof: `ε γ ε` has the same `a` (and
`d`) as `γ`, and `N ∣ c` is preserved; hence `ρ_{n,χ}(εγε) = ρ_n(ε) ρ_{n,χ}(γ) ρ_n(ε)` and `V(χ)` is a module
for `⟨Γ₀(N), ε⟩` with `ε` acting by `ρ_n(ε)`. -/
def epsMat : Mat2 ℤ := !![1, 0; 0, -1]

theorem aMod_eps_conj (γ : Mat2 ℤ) : aMod N (epsMat * γ * epsMat) = aMod N γ := by
  rw [Matrix.eta_fin_two γ]
  simp [aMod, epsMat]

theorem cMod_eps_conj (γ : Mat2 ℤ) (h : cMod N γ = 0) : cMod N (epsMat * γ * epsMat) = 0 := by
  simp only [cMod] at *
  rw [Matrix.eta_fin_two γ]
  simp [epsMat, h]

theorem rhoChi_eps_conj (n : ℕ) (χ : DirichletCharacter R N) (γ : Mat2 ℤ) :
    rhoChi n χ (epsMat * γ * epsMat) =
      rho n (toR epsMat) * rhoChi n χ γ * rho n (toR epsMat) := by
  simp only [rhoChi, aMod_eps_conj, Matrix.map_mul, rho_mul, mul_smul_comm, smul_mul_assoc]

end GL2.Neb
