import GL2.Conventions.Nebentypus

/-!
# `bridge:nebentypus-dictionary`: [C]'s `H¹(Γ₀(N); V(χ))` against `S_k(N, χ)` / `S_k(N, χ̄)`

[GT] `bridge:nebentypus-dictionary` (status `proved_mod_lit`).  Everything in this file is proved; the
literature enters only as named hypotheses in the *statements* of the final theorems.

Conventions on the classical side (Diamond–Shurman = Shimura = LMFDB = FLT):
* `f ∈ S_k(N, ψ)` means `f(γ z) = ψ(d_γ) (c z + d)^k f(z)` for `γ ∈ Γ₀(N)` (`IsNeb`);
* the slash operator `f[ε]_k (z) = (det ε)^{k-1} (c z + d)^{-k} f(ε z)` (`slash`);
* `T_p f = ∑_{j<p} f[(1 j; 0 p)]_k + ψ(p) f[(p 0; 0 1)]_k` for `p ∤ N` (DS Prop. 5.2.1 and §5.3; `classicalTp`).

What is proved here, step by step (the hand derivation of the L16-1 report §4.3, now machine-checked):

* **(D1) Holomorphic periods of `S_k(N, ψ)` are [C]-cocycles for `V(ψ⁻¹)`.** The Eichler–Shimura integrand
  `ω_f(z) = f(z) (z x + y)^n dz` of `f ∈ S_k(N, ψ)` satisfies `ω_f(γ z) d(γ z) = ρ_{n,ψ⁻¹}(γ) ω_f(z) dz`
  (`period_automorphy`).  So, as a space, the holomorphic half of [C]'s `V(χ)`-part is the periods of
  `S_k(N, χ⁻¹) = S_k(N, χ̄)`.
* **(D1') Antiholomorphic periods of `S_k(N, χ)` are also [C]-cocycles for `V(χ)`** (`antiperiod_automorphy`).
* **(D2) [C]'s Hecke operator, transported to integrands, is `χ(p)` times the classical `T_p` of
  `S_k(N, χ̄)`** (`transport_single`, `transported_eq`): [C] reads `χ(a_δ)` on `δ = adj ε`, i.e. `χ(d_ε)`.
* **(D3) Eigenvalues.** On a `T_p`-eigenform `f ∈ S_k(N, χ̄)` with eigenvalue `λ`, [C]'s `T_p` has eigenvalue
  `χ(p) λ = conj λ` (given the Petersson relation `λ = χ̄(p) conj λ`), which is the eigenvalue of
  `f^ρ ∈ S_k(N, χ)` (`eigen_hol`); on `conj g`, `g ∈ S_k(N, χ)` with eigenvalue `μ`, it is `μ` itself
  (`eigen_antihol`).  **Both halves carry the eigenvalues of `S_k(N, χ)` (LMFDB convention), each once: the
  cuspidal charpoly of `T_p` on [C]'s `χ`-part is `∏_{f ∈ S_k(N,χ)} (X − a_p(f))²`.**
* **(D4) The pin.** Independently of (D1)–(D3): an eigensystem of [C]'s operators satisfies [C]'s
  `T_{p²} = T_p² − χ(p) p^{k−1}` (`thm:nebrelations`, with `T(p,p) = χ(p)p^n` proved in
  `GL2.Neb.rhoChi_smul_one`), and a normalised eigenform of `S_k(N, ψ)` satisfies
  `a_{p²} = a_p² − ψ(p) p^{k−1}`; if they are the same system then `ψ(p) = χ(p)` (`pin_character`).
-/

namespace GL2.Neb

open MvPolynomial Matrix ComplexConjugate
open scoped MatrixGroups

variable {N : ℕ}

/-- The Möbius action of an integer matrix on `ℂ`. -/
noncomputable def mob (g : Mat2 ℤ) (z : ℂ) : ℂ := (g 0 0 * z + g 0 1) / (g 1 0 * z + g 1 1)

/-- The automorphy factor `j(g, z) = c z + d`. -/
def jf (g : Mat2 ℤ) (z : ℂ) : ℂ := g 1 0 * z + g 1 1

/-- The vector `w(z) = z x + y ∈ Sym¹`, with `g · w(z) = j(g,z) w(g z)` in the column convention. -/
noncomputable def w (z : ℂ) : MvPolynomial (Fin 2) ℂ := C z * X 0 + X 1

lemma w_isHomogeneous (z : ℂ) : (w z).IsHomogeneous 1 :=
  (isHomogeneous_C_mul_X z 0).add (isHomogeneous_X ℂ 1)

/-- `ρ₁(g) w(z) = j(g, z) w(g z)`: the column convention is the one for which `z x + y` is the period
vector (no transpose).  -/
theorem subst_w (g : Mat2 ℤ) (z : ℂ) (hz : jf g z ≠ 0) :
    subst (toR g) (w z) = jf g z • w (mob g z) := by
  have hm : jf g z * mob g z = g 0 0 * z + g 0 1 := by
    unfold mob; rw [← jf]; field_simp
  rw [smul_eq_C_mul]
  simp only [w, map_add, map_mul, algHom_C, subst_X, linForm_zero, linForm_one, Matrix.map_apply,
    Int.coe_castRingHom, algebraMap_eq]
  conv_rhs => rw [mul_add, ← mul_assoc, ← map_mul, hm]
  simp only [jf, map_add, map_mul]
  ring

/-- The Eichler–Shimura integrand `f(z) (z x + y)^n` (the coefficient of `dz`). -/
noncomputable def P (f : ℂ → ℂ) (n : ℕ) (z : ℂ) : MvPolynomial (Fin 2) ℂ := f z • w z ^ n

lemma P_isHomogeneous (f : ℂ → ℂ) (n : ℕ) (z : ℂ) : (P f n z).IsHomogeneous n := by
  have := ((w_isHomogeneous z).pow n).C_mul (f z)
  rw [one_mul] at this
  simpa [P, smul_eq_C_mul] using this

/-- `P f n z` as an element of `Sym^n(ℂ²)`. -/
noncomputable def PV (f : ℂ → ℂ) (n : ℕ) (z : ℂ) : SymN ℂ n :=
  ⟨P f n z, (mem_homogeneousSubmodule _ _).2 (P_isHomogeneous f n z)⟩

/-- `ρ_n(g)` on the integrand: `ρ_n(g) (f(z) w(z)^n) = (f(z) j(g,z)^n) w(gz)^n`. -/
theorem subst_P (g : Mat2 ℤ) (f : ℂ → ℂ) (n : ℕ) (z : ℂ) (hz : jf g z ≠ 0) :
    subst (toR g) (P f n z) = (f z * jf g z ^ n) • w (mob g z) ^ n := by
  rw [P, map_smul, map_pow, subst_w g z hz, smul_pow, smul_smul]

/-- The Diamond–Shurman nebentypus condition `f(γ z) = ψ(d_γ) (cz+d)^k f(z)` on `Γ₀(N)`
(`f ∈ M_k(Γ₀(N), ψ)` up to holomorphy and growth, which play no role here). -/
def IsNeb (N : ℕ) (k : ℕ) (ψ : DirichletCharacter ℂ N) (f : ℂ → ℂ) : Prop :=
  ∀ γ : SL(2, ℤ), γ ∈ CongruenceSubgroup.Gamma0 N → ∀ z : ℂ, jf γ z ≠ 0 →
    f (mob γ z) = ψ (dMod N γ) * jf γ z ^ k * f z

/-- **(D1)** For `f ∈ S_k(N, ψ)`, `k = n + 2`: `ω_f(γz) · (cz+d)^{-2} = ρ_{n,ψ⁻¹}(γ) ω_f(z)`, i.e. the
holomorphic period integrand is a `V(ψ⁻¹)`-valued automorphic form in [C]'s convention (`d(γz) = (cz+d)^{-2} dz`
for `det γ = 1`).  **Holomorphic periods of `S_k(N, ψ)` live in [C]'s `V(ψ⁻¹)`-part.** -/
theorem period_automorphy (n : ℕ) (ψ : DirichletCharacter ℂ N) (f : ℂ → ℂ) (hf : IsNeb N (n + 2) ψ f)
    (γ : SL(2, ℤ)) (hγ : γ ∈ CongruenceSubgroup.Gamma0 N) (z : ℂ) (hz : jf γ z ≠ 0) :
    ((jf γ z) ^ 2)⁻¹ • PV f n (mob γ z) = rhoChi n ψ⁻¹ γ (PV f n z) := by
  rw [rhoChi_inv_Gamma0 n ψ γ hγ]
  apply Subtype.ext
  simp only [PV, Submodule.coe_smul, LinearMap.smul_apply, rho_apply_coe]
  rw [subst_P (γ : Mat2 ℤ) f n z hz, P, hf γ hγ z hz, smul_smul, smul_smul]
  congr 1
  field_simp
  ring

/-- **(D1')** Antiholomorphic periods: for `g ∈ S_k(N, χ)`, the integrand `conj(g(z)) (z̄ x + y)^n d z̄`
satisfies `ω(γz) · conj(cz+d)^{-2} = ρ_{n,χ}(γ) ω(z)`: **antiholomorphic periods of `S_k(N, χ)` also live in
[C]'s `V(χ)`-part.**  (Uses `conj χ(d) = χ⁻¹(d)`, mathlib's `MulChar.star_apply'`.) -/
theorem antiperiod_automorphy [NeZero N] (n : ℕ) (χ : DirichletCharacter ℂ N) (g : ℂ → ℂ)
    (hg : IsNeb N (n + 2) χ g)
    (γ : SL(2, ℤ)) (hγ : γ ∈ CongruenceSubgroup.Gamma0 N) (z : ℂ) (hz : jf γ z ≠ 0) :
    ((conj (jf γ z)) ^ 2)⁻¹ • PV (fun t => conj (g (conj t))) n (conj (mob γ z)) =
      rhoChi n χ γ (PV (fun t => conj (g (conj t))) n (conj z)) := by
  have hreal_m : conj (mob γ z) = mob γ (conj z) := by simp [mob, map_div₀]
  have hreal_j : conj (jf γ z) = jf γ (conj z) := by simp [jf]
  have hz' : jf γ (conj z) ≠ 0 := by rw [← hreal_j]; simpa using hz
  rw [rhoChi_Gamma0 n χ γ hγ, ← MulChar.star_apply']
  apply Subtype.ext
  simp only [PV, Submodule.coe_smul, LinearMap.smul_apply, rho_apply_coe]
  rw [subst_P (γ : Mat2 ℤ) _ n (conj z) hz', P]
  simp only [Complex.conj_conj]
  rw [hreal_m, hg γ hγ z hz, smul_smul, smul_smul]
  congr 1
  simp only [map_mul, map_pow, RCLike.star_def, hreal_j]
  field_simp
  ring

/-! ### (D2) [C]'s Hecke operator on integrands -/

/-- The classical slash `f[ε]_k (z) = (det ε)^{k-1} (c z + d)^{-k} f(ε z)`. -/
noncomputable def slash (k : ℕ) (ε : Mat2 ℤ) (f : ℂ → ℂ) (z : ℂ) : ℂ :=
  ((ε.det : ℂ)) ^ (k - 1) * ((jf ε z) ^ k)⁻¹ * f (mob ε z)

/-- **(D2), one coset.** For `δ = adj ε` (`det ε = p ≠ 0`), [C]'s summand
`ρ_{n,χ}(δ) ω(δ⁻¹ z) · d(δ⁻¹z)/dz` (with `δ⁻¹ z = ε z` and `d(εz)/dz = p (cz+d)^{-2}`) applied to the integrand
of `f` is the integrand of `χ(d_ε) f[ε]_k`, `k = n + 2`. -/
theorem transport_single (n : ℕ) (χ : DirichletCharacter ℂ N) (f : ℂ → ℂ) (ε : Mat2 ℤ)
    (z : ℂ) (hz : jf ε z ≠ 0) :
    ((ε.det : ℂ) * ((jf ε z) ^ 2)⁻¹) • rhoChi n χ ε.adjugate (PV f n (mob ε z)) =
      PV (fun t => χ (dMod N ε) * slash (n + 2) ε f t) n z := by
  -- ρ₁(adj ε) w(εz) = (p / j) w(z)
  have hadj : subst (toR ε.adjugate) (w (mob ε z)) = ((ε.det : ℂ) / jf ε z) • w z := by
    have h1 := congrArg (subst (toR ε.adjugate)) (subst_w ε z hz)
    rw [← AlgHom.comp_apply, ← subst_mul, ← Matrix.map_mul, adjugate_mul, map_smul] at h1
    have hs : (ε.det • (1 : Mat2 ℤ)).map (Int.castRingHom ℂ) = ((ε.det : ℂ)) • (1 : Mat2 ℂ) := by
      ext i j
      simp only [Matrix.map_apply, Matrix.smul_apply, Matrix.one_apply, smul_eq_mul, Int.coe_castRingHom]
      split_ifs <;> simp
    rw [hs, subst_smul_one_of_isHomogeneous _ (w_isHomogeneous z), pow_one] at h1
    calc subst (toR ε.adjugate) (w (mob ε z))
        = (jf ε z)⁻¹ • (jf ε z • subst (toR ε.adjugate) (w (mob ε z))) := by
          rw [smul_smul, inv_mul_cancel₀ hz, one_smul]
      _ = ((ε.det : ℂ) / jf ε z) • w z := by
          rw [← h1, smul_smul, div_eq_mul_inv, mul_comm]
  apply Subtype.ext
  simp only [PV, rhoChi, LinearMap.smul_apply, Submodule.coe_smul, rho_apply_coe, P, map_smul,
    map_pow, hadj, smul_pow, smul_smul, aMod_adjugate, slash]
  congr 1
  have hk : n + 2 - 1 = n + 1 := rfl
  rw [hk, div_pow]
  field_simp
  ring

/-- The representatives `(1 j; 0 p)`. -/
def epsU (p j : ℤ) : Mat2 ℤ := !![1, j; 0, p]
/-- The representative `(p 0; 0 1)`. -/
def epsV (p : ℤ) : Mat2 ℤ := !![p, 0; 0, 1]

/-- **[C]'s `T_p`, transported to `q`-side functions**: `∑_ε χ(d_ε) f[ε]_k`.  Cosets, named by the cosets
themselves (F1 repair 1): `δ_t = r_t⁻¹ g` (`g = diag(1,p)`, `r_t` representing `(Γ₀(N) ∩ gΓ₀(N)g⁻¹)\Γ₀(N)`) represents
the cosets `δ Γ₀(N)` of `Γ₀(N) g Γ₀(N)` (here `diag(1,p)`, `(p −j; 0 1)`, [C] `def:cosetrepsN`(6)), and
`ε := adj δ` represents the cosets `Γ₀(N) ε` of `Γ₀(N) diag(p,1) Γ₀(N)` (here `(1 j; 0 p)`, `(p 0; 0 1)`).
**Scope (F1 repair 2):** as a formula on an arbitrary `f` the sum depends on the representatives; it is
independent of the choice of `ε` in `Γ₀(N) ε` exactly on `S_k(N, χ̄)` (`transport_rep_indep`), which is where (D1)
puts the holomorphic half. -/
noncomputable def transportedTp (k : ℕ) (χ : DirichletCharacter ℂ N) (p : ℕ) (f : ℂ → ℂ) (z : ℂ) : ℂ :=
  ∑ j ∈ Finset.range p, χ (dMod N (epsU p j)) * slash k (epsU p j) f z +
    χ (dMod N (epsV p)) * slash k (epsV p) f z

/-- **(D2), representative independence (F1 repair 2).** For `f ∈ S_k(N, χ⁻¹)` (`IsNeb N k χ⁻¹ f`) and
`γ ∈ Γ₀(N)`, the summand `χ(d_ε) f[ε]_k` does not change when `ε` is replaced by `γ ε`. -/
theorem transport_rep_indep (k : ℕ) (χ : DirichletCharacter ℂ N) (f : ℂ → ℂ) (hf : IsNeb N k χ⁻¹ f)
    (γ : SL(2, ℤ)) (hγ : γ ∈ CongruenceSubgroup.Gamma0 N) (ε : Mat2 ℤ) (z : ℂ)
    (hε : jf ε z ≠ 0) (hγz : jf γ (mob ε z) ≠ 0) :
    χ (dMod N ((γ : Mat2 ℤ) * ε)) * slash k ((γ : Mat2 ℤ) * ε) f z = χ (dMod N ε) * slash k ε f z := by
  have hc : cMod N (γ : Mat2 ℤ) = 0 := (CongruenceSubgroup.Gamma0_mem).1 hγ
  have hdet : (((γ : Mat2 ℤ) * ε).det : ℂ) = (ε.det : ℂ) := by
    rw [Matrix.det_mul, γ.2, one_mul]
  have hj : jf ((γ : Mat2 ℤ) * ε) z = jf γ (mob ε z) * jf ε z := by
    have hε' := hε
    simp only [jf] at hε'
    simp only [jf, mob, Matrix.mul_apply, Fin.sum_univ_two]
    push_cast
    field_simp
    ring
  have hm : mob ((γ : Mat2 ℤ) * ε) z = mob γ (mob ε z) := by
    have hj' : jf ((γ : Mat2 ℤ) * ε) z ≠ 0 := by rw [hj]; exact mul_ne_zero hγz hε
    simp only [jf] at hε hγz hj'
    simp only [mob, Matrix.mul_apply, Fin.sum_univ_two] at hj' ⊢
    push_cast at hj' ⊢
    field_simp
    ring
  have hu : χ (dMod N γ) * χ⁻¹ (dMod N γ) = 1 := by
    have h := aMod_mul_dMod_Gamma0 γ hγ
    have hdu : IsUnit (dMod N γ) := IsUnit.of_mul_eq_one_right _ h
    rw [MulChar.inv_apply, ← map_mul, Ring.mul_inverse_cancel _ hdu, map_one]
  rw [dMod_mul _ _ hc, map_mul, slash, slash, hdet, hj, hm, hf γ hγ (mob ε z) hγz]
  have hjk : (jf γ (mob ε z) ^ k) ≠ 0 := pow_ne_zero _ hγz
  rw [mul_pow, mul_inv]
  calc χ (dMod N ↑γ) * χ (dMod N ε) *
        (↑ε.det ^ (k - 1) * ((jf (↑γ) (mob ε z) ^ k)⁻¹ * (jf ε z ^ k)⁻¹) *
          (χ⁻¹ (dMod N ↑γ) * jf (↑γ) (mob ε z) ^ k * f (mob ε z)))
      = (χ (dMod N ↑γ) * χ⁻¹ (dMod N ↑γ)) * ((jf (↑γ) (mob ε z) ^ k)⁻¹ * jf (↑γ) (mob ε z) ^ k) *
          (χ (dMod N ε) * (↑ε.det ^ (k - 1) * (jf ε z ^ k)⁻¹ * f (mob ε z))) := by ring
    _ = _ := by rw [hu, inv_mul_cancel₀ hjk]; ring

/-- The classical `T_p` on `S_k(N, ψ)`, `p ∤ N` (DS Prop. 5.2.1, (5.3): the `∞`-representative is
`m · diag(p, 1)` with `m ∈ Γ₀(N)`, `d_m ≡ p`, contributing `ψ(p) f[diag(p,1)]_k` on `S_k(N, ψ)`). -/
noncomputable def classicalTp (k : ℕ) (ψ : DirichletCharacter ℂ N) (p : ℕ) (f : ℂ → ℂ) (z : ℂ) : ℂ :=
  ∑ j ∈ Finset.range p, slash k (epsU p j) f z + ψ (p : ZMod N) * slash k (epsV p) f z

/-- **(D2)** [C]'s `T_p` (transported) is `χ(p)` times the classical `T_p` of `S_k(N, χ⁻¹) = S_k(N, χ̄)`,
for `p ∤ N`.  (Read with (D1): [C]'s `T_p` on the holomorphic half is DS's `T_p^* = ⟨p⟩⁻¹ T_p`.) -/
theorem transported_eq (k : ℕ) (χ : DirichletCharacter ℂ N) (p : ℕ) (hp : IsUnit (p : ZMod N))
    (f : ℂ → ℂ) (z : ℂ) :
    transportedTp k χ p f z = χ (p : ZMod N) * classicalTp k χ⁻¹ p f z := by
  have hdU : ∀ j : ℕ, dMod N (epsU p j) = (p : ZMod N) := fun j => by simp [dMod, epsU]
  have hdV : dMod N (epsV p) = 1 := by simp [dMod, epsV]
  have hu : χ (p : ZMod N) * χ⁻¹ (p : ZMod N) = 1 := by
    rw [MulChar.inv_apply, ← map_mul, Ring.mul_inverse_cancel _ hp, map_one]
  simp only [transportedTp, classicalTp, hdU, hdV, map_one, one_mul, mul_add, Finset.mul_sum,
    ← mul_assoc, hu]

/-! ### (D3) Eigenvalues -/

/-- **(D3), holomorphic half.** If `f ∈ S_k(N, χ̄)` is a `T_p`-eigenform with eigenvalue `λ` and
`λ = χ̄(p) conj λ` (Petersson: `T_p^* = ⟨p⟩⁻¹ T_p`, DS Thm. 5.5.3 — a literature hypothesis), then [C]'s
transported `T_p` has eigenvalue `conj λ` on `f`, which is the eigenvalue of `f^ρ ∈ S_k(N, χ)`. -/
theorem eigen_hol [NeZero N] (k : ℕ) (χ : DirichletCharacter ℂ N) (p : ℕ) (hp : IsUnit (p : ZMod N))
    (f : ℂ → ℂ) (lam : ℂ) (heig : ∀ z, classicalTp k χ⁻¹ p f z = lam * f z)
    (hpet : lam = χ⁻¹ (p : ZMod N) * conj lam) :
    ∀ z, transportedTp k χ p f z = conj lam * f z := by
  intro z
  have hu : χ (p : ZMod N) * χ⁻¹ (p : ZMod N) = 1 := by
    rw [MulChar.inv_apply, ← map_mul, Ring.mul_inverse_cancel _ hp, map_one]
  rw [transported_eq k χ p hp, heig, ← mul_assoc]
  congr 1
  conv_lhs => rw [hpet]
  rw [← mul_assoc, hu, one_mul]

/-- **(D3), antiholomorphic half.** Conjugating (D2) for the character `χ̄`: on `conj g`, `g ∈ S_k(N, χ)` a
`T_p`-eigenform with eigenvalue `μ` and Petersson relation `μ = χ(p) conj μ`, [C]'s operator (for `χ`) has
eigenvalue `μ` — again an eigenvalue of `S_k(N, χ)`. Stated on the scalar: `conj(χ̄(p)·μ) = χ(p)·conj μ = μ`.
**The operator-level conjugation of (D2) is a hand step (not literature; F1 repair 3):** `ρ_n` is integral and
`conj ∘ χ̄ = χ`, and `dz̄` pulls back by `conj(d(εz)/dz)`, so `T_χ(conj ω_g) = conj(T_χ̄ ω_g)`; only the scalar step
is machine-checked here. -/
theorem eigen_antihol [NeZero N] (χ : DirichletCharacter ℂ N) (p : ℕ) (mu : ℂ)
    (hpet : mu = χ (p : ZMod N) * conj mu) :
    conj (χ⁻¹ (p : ZMod N) * mu) = mu := by
  rw [map_mul, ← MulChar.star_apply', RCLike.star_def, Complex.conj_conj, ← hpet]

/-- **(D4) The pin.** Let `λ` be an eigensystem of [C]'s operators on `H¹(Γ₀(N); V(χ))`, so that
`λ(p²) = λ(p)² − χ(p) p^{k−1}` ([C] `thm:nebrelations`(2) at `ν = 1`, with `T(p,p) = χ(p)p^n`), and let `a` be the
system of a normalised eigenform of `S_k(N, ψ)`, so `a(p²) = a(p)² − ψ(p) p^{k−1}` (DS Prop. 5.3.1). If
`λ = a` at `p` and `p²`, and `p ≠ 0` in the coefficient field, then `ψ(p) = χ(p)`.  [C]'s character model
therefore sees `S_k(N, χ)` — never `S_k(N, χ̄)` at a prime with `χ(p) ≠ χ̄(p)`. -/
theorem pin_character {K : Type*} [Field K] (k : ℕ) (p : K) (hp : p ≠ 0) (chip psip lp lp2 ap ap2 : K)
    (hC : lp2 = lp ^ 2 - chip * p ^ (k - 1)) (hF : ap2 = ap ^ 2 - psip * p ^ (k - 1))
    (h1 : lp = ap) (h2 : lp2 = ap2) : psip = chip := by
  have hpk : p ^ (k - 1) ≠ 0 := pow_ne_zero _ hp
  subst h1
  have : chip * p ^ (k - 1) = psip * p ^ (k - 1) := by linear_combination hC - hF - h2
  exact (mul_right_cancel₀ hpk this).symm

/-- `ε = diag(1,−1)` fixes `diag(1,p)` under conjugation (the WP-9 audit's step 1: `ι_ε` commutes with `T_p`). -/
theorem eps_conj_diag (p : ℤ) : epsMat * !![1, 0; 0, p] * epsMat = !![1, 0; 0, p] := by
  simp [epsMat]

end GL2.Neb
