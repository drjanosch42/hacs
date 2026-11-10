/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import Mathlib
import GL2.Literature.SturmCusp
import GL2.HeckeZ.RankES
import GL2.HeckeZ.Check.Cert

/-!
# HeckeZ LC-11: 01 Prop. 5.6, the single-Eisenstein-component certificate over `ℤ` (slot 25, advances M-HZ2: exit 2 met in the generation form; 17 cells partial)

`docs/implementation/slot24/heckez/01-math-core.md` Prop. 5.6 (`prop:heckeZeiscert`), Prop. 5.3 / 5.4 (the read at a root),
Prop. 1.5(2) (`hSturm`); `07-lean-plan.md` §2.2 LC-11, §3 (`hSturm`), §10 (M-HZ2: "LC-11 reproduces T1, T2").
Everything is over `ℤ`: `L` free of finite rank, `X m` the Hecke operators (interface data), `𝕋^fr = ℤ[X m]`,
`𝕋' = ℤ[X p : p ∈ P]` with `P ⊇ {p prime : p ≤ k/12}`.

**The statement** (`heckeZeiscert`). Let `Q` be a set of primes in `P`; for `q ∈ Q` let `λ_q = 1 + q^{k−1}` be a simple root
of `χ(X q)` (TB-χ: `χ = (x − λ_q) r_q`, `r_q(λ_q) ≠ 0`) and `c_q` a polynomial with `(X q − λ_q) c_q(X q) = 0` (01's
`c_{cusp,q}`: the minimal polynomial of `T_q` on the cuspidal part, from (C1), or `r_q` by Cayley–Hamilton). Put
`g ∈ (c_q(λ_q) : q ∈ Q)` (in practice `g_P = gcd_q c_q(λ_q)`). Under `hSturm` and the commutativity of the `X m`:
* **at every prime `ℓ ∤ g`**: `𝕋'_(ℓ) = 𝕋^fr_(ℓ)` — no saturation, no dump: this is how Prop. 5.6 covers `2` and `3`;
* **at a prime `ℓ` with `ℓ^{v+1} ∤ g`**, given one read at a root (01 Prop. 5.4: an entry of `c_{q₀}(X q₀)` in some `ℤ`-basis
  of valuation exactly `s`, and `ℓ^{s+v} ∣ c_{q₀}(λ_{q₀})`): again `𝕋'_(ℓ) = 𝕋^fr_(ℓ)`. This is the hypothesis
  `v_ℓ(exp η_L) ≥ v_ℓ(g_P)` of Prop. 5.6, read entrywise.

With both at every prime, `𝕋^fr = 𝕋'` (`GL2.HeckeZ.Cert.le_of_loc`); since `𝕋' ⊆ ℤ ×_{ℤ/g} 𝕋^cusp` always, this is the generation
consequence of Prop. 5.6 (`𝕋^cusp = ℤ[T_p : p ≤ k/12]|_cusp`); the fibre-product form
`𝕋^fr = ℤ ×_{ℤ/g_P} 𝕋^cusp` is not stated.

**Proof.** For `x ∈ 𝕋^fr`, `hSturm` gives `s ∈ ℤ[X p : p ≤ k/12] ⊆ 𝕋'` with `z := x − s` zero on the cuspidal lattice. So
`z (X q − λ_q) = 0`, hence `z c(X q) = c(λ_q) z` for every polynomial `c` (`kill_aeval`). `e_q := c_q(X q) ∈ 𝕋'` maps `L`
into the line `ker(X q − λ_q)`, which has rank one because `λ_q` is a simple root (`line_of_simple`, over a field
`E ⊇ ℚ_p` with mathlib's `finrank_eigenspace_le`); `z` commutes with `X q`, so it acts on that line by an integer `φ`
(`scalar_of_simple`, Bridge's `int_of_eigen`), the same for every `q` (all lines contain `z L`). Hence
`c_q(λ_q) z = φ e_q ∈ 𝕋'` for every `q ∈ Q`, so `g z = φ · (Σ u_q e_q)` and `g x ∈ 𝕋'`. At `ℓ ∣ g` the read gives
`ℓ^v ∣ φ` (the coordinate argument of `GL2.HeckeZ.EisSplit.dvd_of_read`), so `(g/ℓ^{v_ℓ g}) x ∈ 𝕋'`.

The commutativity of the `X m` comes from Eichler–Shimura (`comm_of_ES`: every `X m` is a scalar on each joint eigenspace,
and these span).
-/

namespace GL2.HeckeZ.Sturm

open GL2.Literature GL2.HeckeZ.Bridge GL2.HeckeZ.Cert GL2.HeckeZ.Check Polynomial TensorProduct

set_option linter.unusedSectionVars false

noncomputable section

variable {p : ℕ} [Fact p.Prime]
variable {L : Type*} [AddCommGroup L] [Module.Free ℤ L] [Module.Finite ℤ L]
variable {E : Type*} [Field E] [Algebra ℤ_[p] E] [Algebra ℚ_[p] E] [IsScalarTower ℤ_[p] ℚ_[p] E]

/-! ### Commutativity from Eichler–Shimura -/

/-- The `X m` commute: each is a scalar on every joint eigenspace of `hES`, and these span `E ⊗ ℤ_p ⊗ L`. -/
theorem comm_of_ES {ι : Type*} (X : ι → Module.End ℤ L)
    (hES : EichlerShimuraLevelOne (ℤ_[p] ⊗[ℤ] L) (Xp p X) E) (i j : ι) : X i * X j = X j * X i := by
  obtain ⟨S, mult, -, -, -, htop⟩ := hES.decomp
  rw [← sub_eq_zero]
  apply bcH_eq_zero p E
  have hle : (⨆ χ ∈ S, W p E X χ) ≤ LinearMap.ker (bcH p E (X i * X j - X j * X i)) := by
    refine iSup₂_le fun χ _ w hw => LinearMap.mem_ker.mpr ?_
    rw [map_sub, map_mul, map_mul, LinearMap.sub_apply, Module.End.mul_apply, Module.End.mul_apply, act_gen hw j,
      act_gen hw i, LinearMap.map_smul, LinearMap.map_smul, act_gen hw i, act_gen hw j, smul_smul, smul_smul,
      mul_comm, sub_self]
  rw [htop, top_le_iff, LinearMap.ker_eq_top] at hle
  exact hle

/-- Every element of `𝕋^fr` commutes with every `X i` when the `X` commute. -/
lemma comm_mem {ι : Type*} {X : ι → Module.End ℤ L} (hc : ∀ i j, X i * X j = X j * X i) {t : Module.End ℤ L}
    (ht : t ∈ Tfr X) (i : ι) : t * X i = X i * t := by
  induction ht using Algebra.adjoin_induction with
  | mem x hx => obtain ⟨j, rfl⟩ := hx; exact hc j i
  | algebraMap r => exact Algebra.commutes r (X i)
  | add x y _ _ hx hy => rw [add_mul, mul_add, hx, hy]
  | mul x y _ _ hx hy => rw [mul_assoc, hy, ← mul_assoc, hx, mul_assoc]

/-! ### The Eisenstein line: rank one at a simple root -/

/-- The characteristic polynomial of `f ⊗ ℤ_p ⊗ E` is that of `f`, mapped. -/
lemma charpoly_bcH (θ : Module.End ℤ L) : (bcH p E θ).charpoly = θ.charpoly.map (Int.castRingHom E) := by
  classical
  let b := Module.Free.chooseBasis ℤ L
  rw [← LinearMap.charpoly_toMatrix (bcH p E θ) (bE p E b), ← LinearMap.charpoly_toMatrix θ b]
  have : LinearMap.toMatrix (bE p E b) (bE p E b) (bcH p E θ) = (LinearMap.toMatrix b b θ).map (Int.castRingHom E) := by
    ext i j; rw [GL2.HeckeZ.RankES.toMatrix_bcH]; rfl
  rw [this, Matrix.charpoly_map]

lemma eval_map_int (r : ℤ[X]) (x : ℤ) : (r.map (Int.castRingHom E)).eval (x : E) = ((r.eval x : ℤ) : E) := by
  rw [Polynomial.eval_map, ← eq_intCast (Int.castRingHom E) x, Polynomial.eval₂_hom, eq_intCast]

/-- **The line `ker(θ − λ)` has rank one when `λ` is a simple root of `χ_θ`**: every `w` in it is an `E`-multiple of a
nonzero `a` in it (after `⊗ ℤ_p ⊗ E`). -/
lemma line_of_simple (θ : Module.End ℤ L) (lam : ℤ) (r : ℤ[X]) (hχ : θ.charpoly = (X - C lam) * r)
    (hr : r.eval lam ≠ 0) {a w : L} (ha : a ≠ 0) (hθa : θ a = lam • a) (hθw : θ w = lam • w) :
    ∃ c : E, emb p E w = c • emb p E a := by
  haveI := charZeroE (p := p) (E := E)
  set V := Module.End.eigenspace (bcH p E θ) (lam : E) with hV
  have hmem : ∀ {x : L}, θ x = lam • x → emb p E x ∈ V := fun {x} hx => by
    rw [Module.End.mem_eigenspace_iff, bcH_emb, hx, emb_zsmul]
  have hrE : ¬ (r.map (Int.castRingHom E)).IsRoot (lam : E) := by
    rw [Polynomial.IsRoot.def, eval_map_int]; exact_mod_cast hr
  have hne : (X - C (lam : E)) * r.map (Int.castRingHom E) ≠ 0 :=
    mul_ne_zero (X_sub_C_ne_zero _) (fun h0 => hrE (by rw [h0, Polynomial.IsRoot.def, eval_zero]))
  have hm : ((bcH p E θ).charpoly).rootMultiplicity (lam : E) = 1 := by
    rw [charpoly_bcH, hχ, Polynomial.map_mul, Polynomial.map_sub, Polynomial.map_X, Polynomial.map_C, eq_intCast,
      Polynomial.rootMultiplicity_mul hne, Polynomial.rootMultiplicity_X_sub_C_self,
      Polynomial.rootMultiplicity_eq_zero hrE]
  have hle : Module.finrank E V ≤ 1 := (LinearMap.finrank_eigenspace_le _ _).trans hm.le
  have hea : (⟨emb p E a, hmem hθa⟩ : V) ≠ 0 := by
    intro h; exact emb_ne_zero p E ha (congrArg Subtype.val h)
  have h1 : Module.finrank E V = 1 :=
    le_antisymm hle (Nat.succ_le_of_lt ((Module.finrank_pos_iff_exists_ne_zero (R := E) (M := V)).mpr ⟨_, hea⟩))
  obtain ⟨c, hc⟩ := (finrank_eq_one_iff_of_nonzero' (K := E) (V := V) _ hea).mp h1 ⟨emb p E w, hmem hθw⟩
  exact ⟨c, (congrArg Subtype.val hc).symm⟩

/-- **An endomorphism commuting with `θ` acts on `ker(θ − λ)` by an integer** (`λ` a simple root of `χ_θ`; the auxiliary
field of `line_of_simple` is `ℚ₂`). -/
lemma scalar_of_simple (θ : Module.End ℤ L) (lam : ℤ) (r : ℤ[X]) (hχ : θ.charpoly = (X - C lam) * r)
    (hr : r.eval lam ≠ 0) (z : Module.End ℤ L) (hz : z * θ = θ * z) :
    ∃ φ : ℤ, ∀ w, θ w = lam • w → z w = φ • w := by
  by_cases h : ∃ a, a ≠ 0 ∧ θ a = lam • a
  · obtain ⟨a, ha, hθa⟩ := h
    have hza : θ (z a) = lam • z a := by
      have := LinearMap.congr_fun hz a
      simp only [Module.End.mul_apply] at this
      rw [← this, hθa, map_zsmul]
    obtain ⟨c, hc⟩ := line_of_simple (p := 2) (E := ℚ_[2]) θ lam r hχ hr ha hθa hza
    obtain ⟨φ, -, hzaφ⟩ := int_eigen_vec (p := 2) (E := ℚ_[2]) z ha hc
    refine ⟨φ, fun w hw => ?_⟩
    obtain ⟨cw, hcw⟩ := line_of_simple (p := 2) (E := ℚ_[2]) θ lam r hχ hr ha hθa hw
    apply emb_injective 2 ℚ_[2]
    calc emb 2 ℚ_[2] (z w) = bcH 2 ℚ_[2] z (emb 2 ℚ_[2] w) := (bcH_emb 2 ℚ_[2] z w).symm
      _ = cw • emb 2 ℚ_[2] (z a) := by rw [hcw, LinearMap.map_smul, bcH_emb]
      _ = cw • ((φ : ℚ_[2]) • emb 2 ℚ_[2] a) := by rw [hzaφ, emb_zsmul]
      _ = (φ : ℚ_[2]) • emb 2 ℚ_[2] w := by rw [hcw, smul_comm]
      _ = emb 2 ℚ_[2] (φ • w) := (emb_zsmul 2 ℚ_[2] φ w).symm
  · push Not at h
    refine ⟨0, fun w hw => ?_⟩
    by_cases hw0 : w = 0
    · simp [hw0]
    · exact absurd hw (h w hw0)

/-! ### Killing the cuspidal lattice -/

lemma kill_pow {z θ : Module.End ℤ L} {μ : ℤ} (h : z * θ = μ • z) (n : ℕ) : z * θ ^ n = μ ^ n • z := by
  induction n with
  | zero => simp
  | succ n ih => rw [pow_succ, ← mul_assoc, ih, smul_mul_assoc, h, smul_smul, pow_succ]

/-- `z θ = μ z` ⇒ `z c(θ) = c(μ) z`. -/
lemma kill_aeval {z θ : Module.End ℤ L} {μ : ℤ} (h : z * θ = μ • z) (c : ℤ[X]) :
    z * aeval θ c = c.eval μ • z := by
  induction c using Polynomial.induction_on' with
  | add f g hf hg => rw [map_add, mul_add, hf, hg, eval_add, add_smul]
  | monomial n a =>
    rw [aeval_monomial, eval_monomial, Algebra.algebraMap_eq_smul_one, smul_mul_assoc, one_mul, mul_smul_comm,
      kill_pow h, smul_smul]

/-- An endomorphism vanishing on the cuspidal lattice satisfies `z X_q = λ_q z` at every prime `q`. -/
lemma kill_sub {X : ℕ → Module.End ℤ L} {k : ℕ} {z : Module.End ℤ L} (hz : ∀ v ∈ cuspLat X k, z v = 0) {q : ℕ}
    (hq : q.Prime) : z * X q = eisLam k q • z := by
  have hle : LinearMap.range (X q - eisLam k q • (LinearMap.id : Module.End ℤ L)) ≤ cuspLat X k :=
    le_iSup₂ (f := fun p (_ : p ∈ {p : ℕ | p.Prime}) =>
      LinearMap.range (X p - eisLam k p • (LinearMap.id : Module.End ℤ L))) q hq
  ext v
  have h0 := hz _ (hle (LinearMap.mem_range_self _ v))
  simp only [LinearMap.sub_apply, LinearMap.smul_apply, LinearMap.id_apply, map_sub, map_zsmul, sub_eq_zero] at h0
  simpa [Module.End.mul_apply] using h0

/-! ### The read at a root (01 Prop. 5.4, entrywise) -/

/-- **The read** (the coordinate argument of 01 Prop. 5.4; `GL2.HeckeZ.EisSplit.dvd_of_read`): if `c(λ) z = φ e`, an entry
of `e` in a basis has `ℓ`-valuation exactly `s` and `ℓ^{s+v} ∣ c(λ)`, then `ℓ^v ∣ φ`. -/
lemma dvd_of_read {n : ℕ} (b : Module.Basis (Fin n) ℤ L) {ℓ : ℕ} (hℓ : ℓ.Prime) {s v : ℕ} {cl φ : ℤ}
    {z e : Module.End ℤ L} (hze : cl • z = φ • e) (i j : Fin n) (hs : (ℓ : ℤ) ^ s ∣ b.repr (e (b j)) i)
    (hs1 : ¬ (ℓ : ℤ) ^ (s + 1) ∣ b.repr (e (b j)) i) (hcl : (ℓ : ℤ) ^ (s + v) ∣ cl) : (ℓ : ℤ) ^ v ∣ φ := by
  have hk : cl * b.repr (z (b j)) i = φ * b.repr (e (b j)) i := by
    have := congrArg (fun F : Module.End ℤ L => b.repr (F (b j)) i) hze
    simpa only [LinearMap.smul_apply, LinearEquiv.map_smul, Finsupp.smul_apply, smul_eq_mul] using this
  obtain ⟨y, hy⟩ := hs
  have hy' : ¬ (ℓ : ℤ) ∣ y := by
    rintro ⟨w, rfl⟩; exact hs1 ⟨w, by rw [hy, pow_succ]; ring⟩
  have h1 : (ℓ : ℤ) ^ v * (ℓ : ℤ) ^ s ∣ (φ * y) * (ℓ : ℤ) ^ s := by
    rw [← pow_add, add_comm]
    exact (hcl.mul_right _).trans (dvd_of_eq (by rw [hk, hy]; ring))
  have hℓ0 : (ℓ : ℤ) ^ s ≠ 0 := pow_ne_zero _ (by exact_mod_cast hℓ.ne_zero)
  have h2 := (mul_dvd_mul_iff_right hℓ0).1 h1
  have hirr : Irreducible (ℓ : ℤ) := (Nat.prime_iff_prime_int.mp hℓ).irreducible
  exact ((hirr.coprime_iff_not_dvd).2 hy').pow_left.dvd_of_dvd_mul_right h2

/-! ### 01 Prop. 5.6 -/

/-- The data of 01 Prop. 5.6 at level one, weight `k`, indexed by `Q` with primes `pr q`: TB-χ (`λ = 1 + (pr q)^{k−1}` a simple
root of `χ(X (pr q))`) and the cuspidal polynomials `c q` with `(X (pr q) − λ) c_q(X (pr q)) = 0`. -/
structure EisData (X : ℕ → Module.End ℤ L) (k : ℕ) {ι : Type*} (Q : List ι) (pr : ι → ℕ) where
  c : ι → ℤ[X]
  r : ι → ℤ[X]
  prime : ∀ q ∈ Q, (pr q).Prime
  hc : ∀ q ∈ Q, (X (pr q) - eisLam k (pr q) • (LinearMap.id : Module.End ℤ L)) * aeval (X (pr q)) (c q) = 0
  hχ : ∀ q ∈ Q, (X (pr q)).charpoly = (Polynomial.X - C (eisLam k (pr q))) * r q
  hr : ∀ q ∈ Q, (r q).eval (eisLam k (pr q)) ≠ 0

variable {ι : Type*} {Q : List ι} {pr : ι → ℕ}

/-- The ideal of `ℤ` generated by the `c_q(λ_q)`, `q ∈ Q` (01's `g_P ℤ`). -/
def EisData.gIdeal {X : ℕ → Module.End ℤ L} {k : ℕ} (D : EisData X k Q pr) : Ideal ℤ :=
  Ideal.span {x | ∃ q ∈ Q, x = (D.c q).eval (eisLam k (pr q))}

lemma aeval_mem_Tp {X : ℕ → Module.End ℤ L} {P : List ℕ} {q : ℕ} (hq : q ∈ P) (c : ℤ[X]) :
    aeval (X q) c ∈ Tp X P := by
  rw [← Polynomial.aeval_subalgebra_coe c (Tp X P) ⟨X q, mem_Tp hq⟩]; exact SetLike.coe_mem _

/-- The core: `z ∈ 𝕋^fr` zero on the cuspidal lattice and the read at `(ℓ, v)` give `g z ∈ ℓ^v 𝕋'` for every `g` in the
ideal `(c_q(λ_q))`. The read at `(ℓ, v)` (01 Prop. 5.4): trivially at `v = 0`, otherwise an entry of some `c_{q₀}(X q₀)`
of valuation exactly `s` in a `ℤ`-basis, with `ℓ^{s+v} ∣ c_{q₀}(λ_{q₀})`. -/
theorem core {X : ℕ → Module.End ℤ L} {k : ℕ} {P : List ℕ} (hcomm : ∀ i j, X i * X j = X j * X i)
    (D : EisData X k Q pr) (hQP : ∀ q ∈ Q, pr q ∈ P) {z : Module.End ℤ L} (hzT : z ∈ Tfr X)
    (hz : ∀ v ∈ cuspLat X k, z v = 0) {ℓ v : ℕ} (hℓ : ℓ.Prime) (hR : v = 0 ∨ ∃ q₀ ∈ Q, ∃ (n : ℕ) (b : Module.Basis (Fin n) ℤ L) (i j : Fin n) (s : ℕ),
      (ℓ : ℤ) ^ s ∣ b.repr (aeval (X (pr q₀)) (D.c q₀) (b j)) i ∧
      ¬ (ℓ : ℤ) ^ (s + 1) ∣ b.repr (aeval (X (pr q₀)) (D.c q₀) (b j)) i ∧
      (ℓ : ℤ) ^ (s + v) ∣ (D.c q₀).eval (eisLam k (pr q₀))) :
    ∀ g ∈ D.gIdeal, ∃ y ∈ Tp X P, g • z = ((ℓ : ℤ) ^ v) • y := by
  -- the ideal of admissible multipliers
  let I : Ideal ℤ :=
    { carrier := {a | ∃ y ∈ Tp X P, a • z = ((ℓ : ℤ) ^ v) • y}
      add_mem' := by
        rintro a a' ⟨y, hy, h⟩ ⟨y', hy', h'⟩
        exact ⟨y + y', (Tp X P).add_mem hy hy', by rw [add_smul, h, h', smul_add]⟩
      zero_mem' := ⟨0, (Tp X P).zero_mem, by simp⟩
      smul_mem' := by
        rintro c a ⟨y, hy, h⟩
        exact ⟨c • y, Subalgebra.smul_mem _ hy c, by rw [smul_eq_mul, mul_smul, h, smul_comm]⟩ }
  have hmemI : ∀ a : ℤ, a ∈ I ↔ ∃ y ∈ Tp X P, a • z = ((ℓ : ℤ) ^ v) • y := fun _ => Iff.rfl
  suffices hsub : D.gIdeal ≤ I from fun g hg => (hmemI g).mp (hsub hg)
  by_cases hz0 : z = 0
  · intro g _; exact (hmemI g).mpr ⟨0, (Tp X P).zero_mem, by simp [hz0]⟩
  -- a vector `w₀ = z v₀ ≠ 0` in every line `ker(X q − λ_q)`
  obtain ⟨v₀, hv₀⟩ : ∃ v₀, z v₀ ≠ 0 := by
    by_contra hcon; push Not at hcon; exact hz0 (LinearMap.ext hcon)
  have hw₀ : ∀ q, q.Prime → X q (z v₀) = eisLam k q • z v₀ := by
    intro q hq
    have h1 := LinearMap.congr_fun (comm_mem hcomm hzT q) v₀
    have h2 := LinearMap.congr_fun (kill_sub hz hq) v₀
    simp only [Module.End.mul_apply, LinearMap.smul_apply] at h1 h2
    rw [← h1, h2]
  -- per `q ∈ Q`: `c_q(λ_q) z = φ e_q`, with `φ` the eigenvalue of `z` on `w₀`
  have hkey : ∀ q ∈ Q, ∀ φ : ℤ, z (z v₀) = φ • z v₀ →
      (D.c q).eval (eisLam k (pr q)) • z = φ • aeval (X (pr q)) (D.c q) := by
    intro q hq φ hφ
    obtain ⟨φq, hφq⟩ := scalar_of_simple (X (pr q)) (eisLam k (pr q)) (D.r q) (D.hχ q hq) (D.hr q hq) z
      (comm_mem hcomm hzT (pr q))
    have hφe : φq = φ := by
      have := hφq _ (hw₀ (pr q) (D.prime q hq))
      rw [hφ] at this
      exact (smul_left_injective ℤ hv₀ this).symm
    rw [← kill_aeval (kill_sub hz (D.prime q hq))]
    ext x
    simp only [Module.End.mul_apply, LinearMap.smul_apply]
    rw [← hφe]
    apply hφq
    have := LinearMap.congr_fun (D.hc q hq) x
    simp only [Module.End.mul_apply, LinearMap.sub_apply, LinearMap.smul_apply, LinearMap.id_apply,
      LinearMap.zero_apply, sub_eq_zero] at this
    exact this
  -- `Q` is non-empty, else the ideal is `0`; take `φ` from one line
  intro g hg
  by_cases hQ : Q = []
  · have : D.gIdeal = ⊥ := by simp [EisData.gIdeal, hQ]
    rw [this, Ideal.mem_bot] at hg
    exact (hmemI g).mpr ⟨0, (Tp X P).zero_mem, by simp [hg]⟩
  obtain ⟨q₁, hq₁⟩ := List.exists_mem_of_ne_nil Q hQ
  obtain ⟨φ, hφ⟩ := scalar_of_simple (X (pr q₁)) (eisLam k (pr q₁)) (D.r q₁) (D.hχ q₁ hq₁) (D.hr q₁ hq₁) z
    (comm_mem hcomm hzT (pr q₁))
  have hφ₀ : z (z v₀) = φ • z v₀ := hφ _ (hw₀ (pr q₁) (D.prime q₁ hq₁))
  -- the read: `ℓ^v ∣ φ`
  obtain ⟨φ', hφ'⟩ : (ℓ : ℤ) ^ v ∣ φ := by
    rcases hR with rfl | ⟨q₀, hq₀, n, b, i, j, s, hs, hs1, hcl⟩
    · simp
    · exact dvd_of_read b hℓ (hkey q₀ hq₀ φ hφ₀) i j hs hs1 hcl
  refine (Ideal.span_le.mpr ?_) hg
  rintro _ ⟨q, hq, rfl⟩
  exact (hmemI _).mpr ⟨φ' • aeval (X (pr q)) (D.c q), Subalgebra.smul_mem _ (aeval_mem_Tp (hQP q hq) _) _,
    by rw [hkey q hq φ hφ₀, hφ', mul_smul]⟩

/-- **01 Prop. 5.6 (`prop:heckeZeiscert`, LC-11) over `ℤ`, local form.** Under `hSturm`, commuting `X m`, `P ⊇ {primes ≤ k/12}`
and the data `D` at `Q ⊆ P`: at every prime `ℓ` with `ℓ^{v+1} ∤ g` for some `g ∈ (c_q(λ_q))` and the read at `(ℓ, v)` —
in particular at every prime `ℓ ∤ g` (`v = 0`) — `𝕋'_(ℓ) = 𝕋^fr_(ℓ)`. -/
theorem heckeZeiscert {X : ℕ → Module.End ℤ L} {k : ℕ} (hcomm : ∀ i j, X i * X j = X j * X i)
    (hS : SturmCuspLevelOne X k) {P : List ℕ} (hP : ∀ q : ℕ, q.Prime → q ≤ k / 12 → q ∈ P)
    (D : EisData X k Q pr) (hQP : ∀ q ∈ Q, pr q ∈ P) {g : ℤ} (hg : g ∈ D.gIdeal) {ℓ v : ℕ} (hℓ : ℓ.Prime)
    (hgv : ¬ (ℓ : ℤ) ^ (v + 1) ∣ g) (hR : v = 0 ∨ ∃ q₀ ∈ Q, ∃ (n : ℕ) (b : Module.Basis (Fin n) ℤ L) (i j : Fin n) (s : ℕ),
      (ℓ : ℤ) ^ s ∣ b.repr (aeval (X (pr q₀)) (D.c q₀) (b j)) i ∧
      ¬ (ℓ : ℤ) ^ (s + 1) ∣ b.repr (aeval (X (pr q₀)) (D.c q₀) (b j)) i ∧
      (ℓ : ℤ) ^ (s + v) ∣ (D.c q₀).eval (eisLam k (pr q₀))) : Loc (Tfr X) (Tp X P) ℓ := by
  classical
  intro x hx
  obtain ⟨s, hs, hxs⟩ := hS.span x hx
  have hsP : s ∈ Tp X P := Algebra.adjoin_mono (by
    rintro _ ⟨q, ⟨hq, hqk⟩, rfl⟩; exact ⟨q, hP q hq hqk, rfl⟩) hs
  have hzT : x - s ∈ Tfr X := (Tfr X).sub_mem hx (Tp_le X P hsP)
  have hz : ∀ w ∈ cuspLat X k, (x - s) w = 0 := fun w hw => by
    rw [LinearMap.sub_apply, hxs w hw, sub_self]
  obtain ⟨y, hy, hgy⟩ := core hcomm D hQP hzT hz hℓ hR g hg
  -- split off the `ℓ`-part of `g`: `g = ℓ^w g'`, `ℓ ∤ g'`, `w ≤ v`
  have hex : ∃ w, ¬ (ℓ : ℤ) ^ (w + 1) ∣ g := ⟨v, hgv⟩
  set w := Nat.find hex with hwdef
  have hw1 : ¬ (ℓ : ℤ) ^ (w + 1) ∣ g := Nat.find_spec hex
  have hwv : w ≤ v := Nat.find_min' hex hgv
  have hwg : (ℓ : ℤ) ^ w ∣ g := by
    rcases Nat.eq_zero_or_pos w with h0 | hpos
    · rw [h0, pow_zero]; exact one_dvd _
    · have := Nat.find_min hex (show w - 1 < w by omega)
      push Not at this
      rwa [Nat.sub_add_cancel hpos] at this
  obtain ⟨g', hg'⟩ := hwg
  have hg'ℓ : ¬ (ℓ : ℤ) ∣ g' := by
    rintro ⟨t, rfl⟩; exact hw1 ⟨t, by rw [hg', pow_succ]; ring⟩
  have hℓw : ((ℓ : ℤ) ^ w) ≠ 0 := pow_ne_zero _ (by exact_mod_cast hℓ.ne_zero)
  have hz' : g' • (x - s) = ((ℓ : ℤ) ^ (v - w)) • y := by
    have h1 : ((ℓ : ℤ) ^ w) • (g' • (x - s) - ((ℓ : ℤ) ^ (v - w)) • y) = 0 := by
      rw [smul_sub, smul_smul, smul_smul, ← hg', hgy, ← pow_add, Nat.add_sub_cancel' hwv, sub_self]
    exact sub_eq_zero.mp (end_smul_eq_zero hℓw h1)
  refine ⟨g', hg'ℓ, ?_⟩
  have : g' • x = g' • s + ((ℓ : ℤ) ^ (v - w)) • y := by rw [← hz', ← smul_add]; congr 1; abel
  rw [this]
  exact (Tp X P).add_mem (Subalgebra.smul_mem _ hsP _) (Subalgebra.smul_mem _ hy _)

/-- **Prop. 5.6 with Eichler–Shimura**: `hES` supplies the commutativity. -/
theorem heckeZeiscert_ES {X : ℕ → Module.End ℤ L} {k : ℕ} (hES : EichlerShimuraLevelOne (ℤ_[p] ⊗[ℤ] L) (Xp p X) E)
    (hS : SturmCuspLevelOne X k) {P : List ℕ} (hP : ∀ q : ℕ, q.Prime → q ≤ k / 12 → q ∈ P)
    (D : EisData X k Q pr) (hQP : ∀ q ∈ Q, pr q ∈ P) {g : ℤ} (hg : g ∈ D.gIdeal) {ℓ v : ℕ} (hℓ : ℓ.Prime)
    (hgv : ¬ (ℓ : ℤ) ^ (v + 1) ∣ g) (hR : v = 0 ∨ ∃ q₀ ∈ Q, ∃ (n : ℕ) (b : Module.Basis (Fin n) ℤ L) (i j : Fin n) (s : ℕ),
      (ℓ : ℤ) ^ s ∣ b.repr (aeval (X (pr q₀)) (D.c q₀) (b j)) i ∧
      ¬ (ℓ : ℤ) ^ (s + 1) ∣ b.repr (aeval (X (pr q₀)) (D.c q₀) (b j)) i ∧
      (ℓ : ℤ) ^ (s + v) ∣ (D.c q₀).eval (eisLam k (pr q₀))) : Loc (Tfr X) (Tp X P) ℓ :=
  heckeZeiscert (comm_of_ES X hES) hS hP D hQP hg hℓ hgv hR

end

end GL2.HeckeZ.Sturm
