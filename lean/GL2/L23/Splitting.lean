/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.L23.OddDims

/-!
# LG-3 (slot 23): the splitting `M = M₀ ⊕ M₁` and [R] Theorem A's dimensions `h^q = ⌊n/2⌋ + 1`

`L23_master_plan.md` v0.4 §4.3 (rows "[R] Thm A, C dimension closed forms" and "layer splitting `M = M₀ ⊕ M₁`",
package LG-3); [R] = `census_l23_classification_review.md` Theorem A.
[GT] `gt_part_MP.tex`, rows `thm:lg-split-general`, `thm:lg-split-v2`, `thm:lg-f2-general`, `thm:lg-thmA-dims`;
audit `F-LG34_audit.md` (repairs R2, R5–R7 applied).

* **The splitting, general** (`isCompl_ker_range`, `ker_range_stable`): over any commutative ring `k` with `3 ∈ k^×`, an
  endomorphism `u` with `u³ = 1` splits `M = ker (u − 1) ⊕ range (u − 1)` (the idempotent `⅓(1 + u + u²)`), and any `g`
  with `g u = u g` or `g u = u² g` preserves both summands.
* **`Sym^n F₂²`** (`V2_split`, `V2_M0_stable`, `V2_M1_stable`, `V2_M0_stable_SL`, `V2_M1_stable_SL`): with `u = ρ_n(Ū²)`
  (`⟨Ū²⟩ = A₃ ⊂ S₃ = GL₂(F₂)`), `M₀ = M^{U²}` and `M₁ = (U² − 1)M` are complementary, and **stable under `ρ_n(A)` for every
  `A ∈ GL₂(F₂)`**, hence under every element of `SL₂(ℤ)` (`V2_M0_stable_SL`, `V2_M1_stable_SL`).  *In prose only (not a
  Lean statement):* an integer matrix of odd determinant (a Hecke representative at `p` odd) acts on `Sym^n F₂²` by `ρ_n`
  of its reduction, which lies in `GL₂(F₂)`, so it preserves `M₀`, `M₁` — [R] Theorem A's "stably under every
  odd-determinant integer matrix".
* **Characteristic `2`, `−I` trivial, general — no freeness** (`finrank_H_succ_general`): if `S U S = U²` on `M` (with
  `−I` trivial this says exactly that `ρ` factors through `SL₂(ℤ) → GL₂(F₂) ≅ S₃`: the theorem is about
  `F₂[GL₂(F₂)]`-modules seen as `SL₂(ℤ)`-modules, the modules of [R] Theorem A), then
  `dim H^{q+1}(SL₂(ℤ); M) + dim (1 + S)M = dim M` for every `q`.  This removes LG-2's hypothesis `M^S = (1 + S)M`: the
  key identity is `(1 + S)M ∩ M^U = (1 + S)(M^U)` (`range_inf_eq_map`), which holds without freeness.
* **[R] Theorem A, dimensions, every `n`** (`V2_finrank_rangeS_all`, `V2_dims_all`):
  **`dim_{F₂} H^{q+1}(SL₂(ℤ); Sym^n F₂²) = ⌊n/2⌋ + 1`** for every `n ≥ 0` and every `q ≥ 0` (for odd `n` this is LG-2's
  `(n + 1)/2`).  The proof does not go through the three atoms (Mayer–Vietoris on `F₂`, `F₂[Z/2]`, `St₂`): `(1 + S)M` is
  spanned by the `⌈n/2⌉` free `S`-orbit sums of monomials.

**Not here** (gaps of LG-3): the atom decomposition `M₀ ≅ 1^t ⊕ P^i`, `M₁ ≅ St₂^m` and the closed form
`c = dim M₀ = #{j ≤ n : j ≡ −n mod 3}` (an eigenvalue count over `F₄`); `h⁰ = t + i` in closed form; the Hecke
statements of [R] Theorem B (LH).  [R] Theorem C (`ℓ = 3`: `h^q = [n ≡ 0, 4 mod 6]`, `q ≥ 2`, `n` even) is proved for every
even `n` in `GL2.L23.JordanF3` (LG-C3, via the Jordan type `J₃^{⌊n/3⌋} ⊕ J_{(n mod 3)+1}`).
-/

noncomputable section

namespace GL2.L23.Split

open CategoryTheory GL2.MP GL2.L23.TreeF2 GL2.L23.OddDims GL2.L23.RankF2 Module
open scoped MatrixGroups

/-! ### The splitting, general -/

section General

variable {k M : Type} [CommRing k] [AddCommGroup M] [Module k M]

/-- **`M = ker (u − 1) ⊕ range (u − 1)`** if `u³ = 1` and `3 ∈ k^×`. -/
theorem isCompl_ker_range (u : M →ₗ[k] M) (h3 : IsUnit (3 : k)) (hu : ∀ x, u (u (u x)) = x) :
    IsCompl (LinearMap.ker (u - 1)) (LinearMap.range (u - 1)) := by
  obtain ⟨c, hc⟩ : ∃ c : k, c * 3 = 1 := ⟨_, h3.val_inv_mul⟩
  constructor
  · rw [Submodule.disjoint_def]
    intro x hx hx'
    obtain ⟨y, rfl⟩ := hx'
    rw [LinearMap.mem_ker, LinearMap.sub_apply, Module.End.one_apply, sub_eq_zero] at hx
    set z := (u - 1) y with hz
    have hz' : z = u y - y := by rw [hz, LinearMap.sub_apply, Module.End.one_apply]
    have h2 : z + u z + u (u z) = 0 := by
      rw [hz']; simp only [map_sub]; rw [hu]; abel
    rw [hx, hx] at h2
    calc z = (c * 3) • z := by rw [hc, one_smul]
      _ = c • (z + z + z) := by rw [mul_smul]; congr 1; module
      _ = 0 := by rw [h2, smul_zero]
  · rw [codisjoint_iff, eq_top_iff]
    intro x _
    have hx : x = c • (x + u x + u (u x)) + (u - 1) (-(c • ((2 : k) • x + u x))) := by
      simp only [LinearMap.sub_apply, Module.End.one_apply, map_neg, map_smul, map_add]
      conv_lhs => rw [← one_smul k x, ← hc]
      module
    rw [hx]
    refine Submodule.add_mem_sup ?_ ⟨_, rfl⟩
    rw [LinearMap.mem_ker, LinearMap.sub_apply, Module.End.one_apply, map_smul, map_add, map_add, hu, ← smul_sub,
      show u x + u (u x) + x - (x + u x + u (u x)) = 0 by abel, smul_zero]

/-- `g u = u g` or `g u = u² g` (with `u³ = 1`): `g` preserves `ker (u − 1)` and `range (u − 1)`. -/
theorem ker_range_stable (u g : M →ₗ[k] M) (hu : ∀ x, u (u (u x)) = x)
    (hg : (∀ x, g (u x) = u (g x)) ∨ (∀ x, g (u x) = u (u (g x)))) :
    (∀ x ∈ LinearMap.ker (u - 1), g x ∈ LinearMap.ker (u - 1)) ∧
      (∀ x ∈ LinearMap.range (u - 1), g x ∈ LinearMap.range (u - 1)) := by
  have hker : ∀ x, x ∈ LinearMap.ker (u - 1) ↔ u x = x := fun x => by
    rw [LinearMap.mem_ker, LinearMap.sub_apply, Module.End.one_apply, sub_eq_zero]
  rcases hg with hg | hg
  · refine ⟨fun x hx => ?_, ?_⟩
    · rw [hker] at hx ⊢; rw [← hg, hx]
    · rintro _ ⟨y, rfl⟩
      refine ⟨g y, ?_⟩
      simp only [LinearMap.sub_apply, Module.End.one_apply, map_sub, hg]
  · refine ⟨fun x hx => ?_, ?_⟩
    · rw [hker] at hx ⊢
      have h1 : u (u (g x)) = g x := by rw [← hg, hx]
      calc u (g x) = u (u (u (g x))) := by rw [h1]
        _ = g x := hu _
    · rintro _ ⟨y, rfl⟩
      refine ⟨u (g y) + g y, ?_⟩
      simp only [LinearMap.sub_apply, Module.End.one_apply, map_sub, map_add, hg]
      abel

end General

/-! ### Characteristic `2`, `−I` trivial: `dim H^{q+1} + dim (1 + S)M = dim M`, no freeness -/

section Char2

variable {A : Rep.{0} (ZMod 2) SL(2, ℤ)}

/-- **`(1 + S)M ∩ M^U = (1 + S)(M^U)`**, if `S U S = U²` on `M` (no freeness of `M` over `F₂[⟨S⟩]` needed). -/
theorem range_inf_eq_map (hI : TrivNegOne A)
    (hC : ∀ x : A, A.ρ sS (A.ρ sU (A.ρ sS x)) = A.ρ sU (A.ρ sU x)) :
    LinearMap.range (nrmA A sS 2) ⊓ LinearMap.ker (difA A sU) =
      Submodule.map (nrmA A sS 2) (LinearMap.ker (difA A sU)) := by
  apply le_antisymm
  · intro y hy
    obtain ⟨⟨v, rfl⟩, hy'⟩ := Submodule.mem_inf.mp hy
    refine ⟨nrmA A sU 3 v, ?_, ?_⟩
    · show nrmA A sU 3 v ∈ LinearMap.ker (difA A sU)
      rw [← range_piU hI]; exact ⟨v, rfl⟩
    · show nrmA A sS 2 (nrmA A sU 3 v) = nrmA A sS 2 v
      rw [← nrm_comm hI hC, piU_fix hy']
  · rintro _ ⟨w, hw, rfl⟩
    have hw' : w ∈ LinearMap.ker (difA A sU) := hw
    refine Submodule.mem_inf.mpr ⟨⟨w, rfl⟩, ?_⟩
    rw [← piU_fix hw', ← nrm_comm hI hC, ← range_piU hI]
    exact ⟨nrmA A sS 2 w, rfl⟩

variable [FiniteDimensional (ZMod 2) A]

/-- `dim (1 + S)(M^U) + dim H⁰ = dim M^U`. -/
lemma finrank_map_add_H0 :
    finrank (ZMod 2) (Submodule.map (nrmA A sS 2) (LinearMap.ker (difA A sU))) +
      finrank (ZMod 2) (groupCohomology A 0) = finrank (ZMod 2) (LinearMap.ker (difA A sU)) := by
  have hrn := LinearMap.finrank_range_add_finrank_ker (nrmA A sS 2 ∘ₗ (LinearMap.ker (difA A sU)).subtype)
  rw [LinearMap.range_comp, Submodule.range_subtype] at hrn
  have hk : finrank (ZMod 2) (LinearMap.ker (nrmA A sS 2 ∘ₗ (LinearMap.ker (difA A sU)).subtype)) =
      finrank (ZMod 2) ↥(LinearMap.ker (difA A sU) ⊓ LinearMap.ker (nrmA A sS 2)) := by
    rw [LinearMap.ker_comp, (Submodule.equivMapOfInjective _ (Submodule.injective_subtype _) _).finrank_eq,
      Submodule.map_comap_subtype]
  have hinf : LinearMap.ker (difA A sU) ⊓ LinearMap.ker (nrmA A sS 2) = LinearMap.ker (sigmaA (A := A)) := by
    rw [ker_sigmaA, nrm_eq_dif, inf_comm]
  rw [hinf] at hk
  have h0 := finrank_H_zero (A := A)
  have hσ := LinearMap.finrank_range_add_finrank_ker (sigmaA (A := A))
  omega

/-- **`dim H^{q+1}(SL₂(ℤ); M) + dim (1 + S)M = dim M`**, every `q`, for every finite-dimensional `F₂`-representation
with `−I` trivial and `S U S = U²` on `M` (no freeness hypothesis; generalises LG-2's `finrank_H_succ_of_free`).
Scope: with `−I` trivial, `S U S = U²` means `ρ` factors through `SL₂(ℤ) → GL₂(F₂) ≅ S₃`. -/
theorem finrank_H_succ_general (hI : TrivNegOne A)
    (hC : ∀ x : A, A.ρ sS (A.ρ sU (A.ρ sS x)) = A.ρ sU (A.ρ sU x)) (q : ℕ) :
    finrank (ZMod 2) (groupCohomology A (q + 1)) + finrank (ZMod 2) (LinearMap.range (nrmA A sS 2)) =
      finrank (ZMod 2) A := by
  have hH := finrank_H_succ hI q
  have h0 := finrank_H_zero (A := A)
  have hsup := Submodule.finrank_sup_add_finrank_inf_eq (LinearMap.range (nrmA A sS 2))
    (LinearMap.range (nrmA A sU 3))
  rw [← range_psiA, range_piU hI, range_inf_eq_map hI hC] at hsup
  have hm := finrank_map_add_H0 (A := A)
  omega

end Char2

/-! ### `Sym^n F₂²`: the splitting and [R] Theorem A's dimensions -/

section V2

/-- In `GL₂(F₂) ≅ S₃`, `A₃ = ⟨Ū²⟩` is normal: `A Ū² = Ū² A` or `A Ū² = Ū⁴ A`. -/
lemma gl2F2_normalises (A : Mat2 (ZMod 2)) (hA : A.det ≠ 0) :
    A * mUU2 = mUU2 * A ∨ A * mUU2 = mUU2 * mUU2 * A := by
  have key : ∀ a b c d : ZMod 2, a * d - b * c ≠ 0 →
      (!![a, b; c, d] * mUU2 = mUU2 * !![a, b; c, d] ∨
        !![a, b; c, d] * mUU2 = mUU2 * mUU2 * !![a, b; c, d]) := by
    unfold mUU2; decide
  have hA' : A = !![A 0 0, A 0 1; A 1 0, A 1 1] := Matrix.eta_fin_two A
  rw [hA'] at hA ⊢
  exact key _ _ _ _ (by rwa [Matrix.det_fin_two_of] at hA)

/-- `u = ρ_n(Ū²)` on `Sym^n F₂²`. -/
abbrev uV2 (n : ℕ) : V2 n →ₗ[ZMod 2] V2 n := (V2 n).ρ (sU ^ 2)

lemma uV2_eq (n : ℕ) : uV2 n = Matrix.toLin' (GL2.rhoMat n mUU2) := by
  rw [uV2, V2_rho, mod2_sU_sq]

lemma uV2_cube (n : ℕ) (x : V2 n) : uV2 n (uV2 n (uV2 n x)) = x := by
  have h : (V2 n).ρ (sU ^ 2 * sU ^ 2 * sU ^ 2) = 1 := by
    rw [← pow_add, ← pow_add, show 2 + 2 + 2 = 3 * 2 by rfl, pow_mul, map_pow, rho_sU_cube (V2_trivNegOne n), one_pow]
  have hx := congrArg (fun f => f x) h
  simpa [map_mul, Module.End.mul_apply] using hx

lemma three_unit : IsUnit (3 : ZMod 2) := ⟨⟨3, 3, by decide, by decide⟩, rfl⟩

/-- `M₀ = M^{U²}`, the `A₃`-invariants. -/
abbrev M0 (n : ℕ) : Submodule (ZMod 2) (V2 n) := LinearMap.ker (uV2 n - 1)

/-- `M₁ = (U² − 1)M`. -/
abbrev M1 (n : ℕ) : Submodule (ZMod 2) (V2 n) := LinearMap.range (uV2 n - 1)

/-- **`Sym^n F₂² = M₀ ⊕ M₁`** ([R] Theorem A; `|A₃| = 3` is a unit in `F₂`). -/
theorem V2_split (n : ℕ) : IsCompl (M0 n) (M1 n) :=
  isCompl_ker_range (uV2 n) three_unit (uV2_cube n)

lemma V2_matrix_normalises (n : ℕ) (A : Mat2 (ZMod 2)) (hA : A.det ≠ 0) :
    (∀ x, Matrix.toLin' (GL2.rhoMat n A) (uV2 n x) = uV2 n (Matrix.toLin' (GL2.rhoMat n A) x)) ∨
      (∀ x, Matrix.toLin' (GL2.rhoMat n A) (uV2 n x) = uV2 n (uV2 n (Matrix.toLin' (GL2.rhoMat n A) x))) := by
  rcases gl2F2_normalises A hA with h | h
  · left
    intro x
    rw [uV2_eq]
    have := congrArg (fun B => Matrix.toLin' (GL2.rhoMat n B) x) h
    simpa [GL2.rhoMat_mul, Matrix.toLin'_mul, Module.End.mul_apply] using this
  · right
    intro x
    rw [uV2_eq]
    have := congrArg (fun B => Matrix.toLin' (GL2.rhoMat n B) x) h
    simpa [GL2.rhoMat_mul, Matrix.toLin'_mul, Module.End.mul_apply] using this

/-- **`M₀` is stable under `ρ_n(A)` for every `A ∈ GL₂(F₂)`** (hence under every integer matrix of odd determinant). -/
theorem V2_M0_stable (n : ℕ) (A : Mat2 (ZMod 2)) (hA : A.det ≠ 0) (x : V2 n) (hx : x ∈ M0 n) :
    Matrix.toLin' (GL2.rhoMat n A) x ∈ M0 n :=
  (ker_range_stable (uV2 n) _ (uV2_cube n) (V2_matrix_normalises n A hA)).1 x hx

/-- **`M₁` is stable under `ρ_n(A)` for every `A ∈ GL₂(F₂)`.** -/
theorem V2_M1_stable (n : ℕ) (A : Mat2 (ZMod 2)) (hA : A.det ≠ 0) (x : V2 n) (hx : x ∈ M1 n) :
    Matrix.toLin' (GL2.rhoMat n A) x ∈ M1 n :=
  (ker_range_stable (uV2 n) _ (uV2_cube n) (V2_matrix_normalises n A hA)).2 x hx

lemma mod2_det (g : SL(2, ℤ)) : (mod2 g).det ≠ 0 := by
  have h : (mod2 g).det = Int.castRingHom (ZMod 2) (g : Matrix (Fin 2) (Fin 2) ℤ).det := by
    rw [mod2, RingHom.map_det]; rfl
  rw [h, Matrix.SpecialLinearGroup.det_coe, map_one]
  decide

/-- `M₀`, `M₁` are sub-representations of `Sym^n F₂²` under `SL₂(ℤ)`. -/
theorem V2_M0_stable_SL (n : ℕ) (g : SL(2, ℤ)) (x : V2 n) (hx : x ∈ M0 n) : (V2 n).ρ g x ∈ M0 n := by
  rw [V2_rho]; exact V2_M0_stable n _ (mod2_det g) x hx

theorem V2_M1_stable_SL (n : ℕ) (g : SL(2, ℤ)) (x : V2 n) (hx : x ∈ M1 n) : (V2 n).ρ g x ∈ M1 n := by
  rw [V2_rho]; exact V2_M1_stable n _ (mod2_det g) x hx

/-- `x ∈ ker (1 + S)` on `Sym^n F₂²` iff `x` is reversal-symmetric. -/
lemma mem_kerN_iff (n : ℕ) (x : V2 n) :
    x ∈ LinearMap.ker (nrmA (V2 n) sS 2) ↔ ∀ t, (x : Fin (n + 1) → ZMod 2) (Fin.rev t) = x t := by
  rw [LinearMap.mem_ker, nrmS_apply]
  constructor
  · intro h t
    have := congrFun (congrArg (fun y : V2 n => (y : Fin (n + 1) → ZMod 2)) h) t
    change (x : Fin (n + 1) → ZMod 2) t + ((V2 n).ρ sS x : Fin (n + 1) → ZMod 2) t = 0 at this
    rw [V2_S_apply] at this
    rw [eq_comm, ← sub_eq_zero, sub_eq_add_zmod2]
    exact this
  · intro h
    funext t
    show (x : Fin (n + 1) → ZMod 2) t + ((V2 n).ρ sS x : Fin (n + 1) → ZMod 2) t = 0
    rw [V2_S_apply, h, add_self_zero]

/-- The reversal-symmetric vectors are determined by their first `⌊n/2⌋ + 1` coordinates. -/
def kerNEquiv (n : ℕ) : LinearMap.ker (nrmA (V2 n) sS 2) ≃ₗ[ZMod 2] (Fin (n / 2 + 1) → ZMod 2) where
  toFun x s := (x.1 : Fin (n + 1) → ZMod 2) ⟨s, by have := s.2; omega⟩
  map_add' _ _ := rfl
  map_smul' _ _ := rfl
  invFun w := ⟨(fun t : Fin (n + 1) => w ⟨min t (n - t), by have := t.2; omega⟩ : Fin (n + 1) → ZMod 2), by
    rw [mem_kerN_iff]
    intro t
    show w _ = w _
    congr 1
    ext
    simp only [Fin.val_rev]
    have := t.2
    omega⟩
  left_inv x := by
    apply Subtype.ext
    funext t
    show (x.1 : Fin (n + 1) → ZMod 2) ⟨min t (n - t), _⟩ = (x.1 : Fin (n + 1) → ZMod 2) t
    by_cases ht : (t : ℕ) ≤ n - t
    · exact congrArg (x.1 : Fin (n + 1) → ZMod 2) (Fin.ext (by simp only; omega))
    · have hs := (mem_kerN_iff n x.1).mp x.2 t
      rw [← hs]
      exact congrArg (x.1 : Fin (n + 1) → ZMod 2) (Fin.ext (by simp only [Fin.val_rev]; have := t.2; omega))
  right_inv w := by
    funext s
    show w _ = w s
    exact congrArg w (Fin.ext (by simp only; have := s.2; omega))

/-- **`dim (1 + S)M = ⌈n/2⌉`** on `Sym^n F₂²`, every `n`. -/
theorem V2_finrank_rangeS_all (n : ℕ) :
    finrank (ZMod 2) (LinearMap.range (nrmA (V2 n) sS 2)) = (n + 1) / 2 := by
  have hrn := LinearMap.finrank_range_add_finrank_ker (nrmA (V2 n) sS 2)
  rw [(kerNEquiv n).finrank_eq, Module.finrank_fin_fun, finrank_V2] at hrn
  omega

/-- **[R] Theorem A, dimensions**: `dim_{F₂} H^{q+1}(SL₂(ℤ); Sym^n F₂²) = ⌊n/2⌋ + 1` for every `n` and every `q`. -/
theorem V2_dims_all (n q : ℕ) : finrank (ZMod 2) (groupCohomology (V2 n) (q + 1)) = n / 2 + 1 := by
  have h := finrank_H_succ_general (V2_trivNegOne n) (V2_conj n) q
  rw [V2_finrank_rangeS_all, finrank_V2] at h
  omega

end V2

end GL2.L23.Split
