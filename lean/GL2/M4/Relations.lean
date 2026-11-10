import GL2.Literature.HeckeModel
import GL2.Conventions.HeckeNormalisation

/-!
# M4-c, M4-d, M4-e (slot 20, L20-2, branch H): the Hecke relations on the model

[GT] `gt_part_M4.tex`, rows `thm:relations-psq-model`, `thm:relations-1-model`, `thm:relations-2-model`,
`thm:relations-comm-model`, `obl:algebra-model`, `lem:m4-hTp-hrel`; the frozen statements are the `_stmt` definitions of
`GL2.M4.Skeleton`. Each theorem is `proved_mod_lit` over one Part-L hypothesis of `GL2.Literature.HeckeModel`.

* `Tp_sq` (the `p^n I` difference) over `hTp` at `μ = 0`, by independence of `τ` (`heckeTildeR_indep_tau`).
* `relations` (the full recursion) over `hTp`: our own argument from `Tfull_add_two` (the assembly of `Tfull`,
  [C] `eq:fullassembly`, `prop:fullassembly-model`) and the two clauses of `hTp`, by induction on `ν`.
* `hTp_imp_hrel`: `hTp` gives `bridge:fullprimitive`'s `hrel` (and `hfull`) at `A = End_R(H^n)`, so the two literature
  routes agree; `hTp_prim_recursion` then feeds `GL2.prim_recursion` at `c = p^n`.
* `coprime` over `hcop`, `Tdc_commute` over `hcomm` (both stated for one datum, extended to all by `TdcOf_eq`),
  `heckeAlg_comm` from `Tdc_commute` by `Algebra.adjoin_induction`.
-/

noncomputable section

namespace GL2.M4

open GL2.Model GL2.Model.XBulk GL2.Cone GL2.Cosets GL2.Literature

variable {R : Type} [CommRing R] {V : Type} [AddCommGroup V] [Module R V] (ρ : ΔQ →* Module.End R V)

/-- A coset datum for `g` forces `g ∈ Δ` (`g = s_i · (s_i⁻¹ g)`, the coset of `1`). -/
lemma mem_ΔQ_of_cd {g : GLQ} (cd : CosetData ΓQ ΔQ g) : g ∈ ΔQ := by
  obtain ⟨i, -⟩ := (cd.cosets 1).exists
  have h := ΔQ.mul_mem (ΓQ_le_ΔQ (cd.s i)) (cd.δ_mem i)
  rwa [mul_inv_cancel_left] at h

/-- `Tdc` is `TdcOf` (`TdcOf_eq`). -/
lemma Tdc_eq_TdcOf (h6 : IsUnit (6 : R)) {g : GLQ} (cd : CosetData ΓQ ΔQ g) (τ : TauR R g eQ) (n : ℕ) :
    Tdc ρ cd τ n = TdcOf ρ h6 g (mem_ΔQ_of_cd cd) n :=
  (TdcOf_eq ρ h6 g _ cd τ n).symm

/-- `Tprim` does not depend on the family of `τ`s (`6 ∈ R^×`). -/
lemma Tprim_indep (h6 : IsUnit (6 : R)) (p : ℕ) (hp : p.Prime) (τ τ' : ∀ μ : ℕ, TauR R (gP p μ hp) eQ) (n μ : ℕ) :
    Tprim ρ p hp τ n μ = Tprim ρ p hp τ' n μ := by
  unfold Tprim
  split_ifs with h
  · exact heckeTildeR_indep_tau ρ h6 _ _ _ n
  · rfl

/-- `Tprim` at `μ ≥ 1` is `Tdc` on `cdP`. -/
lemma Tprim_pos (p : ℕ) (hp : p.Prime) (τ : ∀ μ : ℕ, TauR R (gP p μ hp) eQ) (n μ : ℕ) (h : 1 ≤ μ) :
    Tprim ρ p hp τ n μ = Tdc ρ (cdP p μ hp h) (τ μ) n := by
  unfold Tprim; rw [dif_pos h]

/-- The assembly of `Tfull` ([C] `eq:fullassembly`): `T_{p^{ν+2}} = T_{Δ^prim_{p^{ν+2}}} + c T_{p^ν}`. -/
lemma Tfull_add_two (p : ℕ) (hp : p.Prime) (τ : ∀ μ : ℕ, TauR R (gP p μ hp) eQ) (c : R) (n ν : ℕ) :
    Tfull ρ p hp τ c n (ν + 2) = Tprim ρ p hp τ n (ν + 2) + c • Tfull ρ p hp τ c n ν := by
  unfold Tfull
  have h2 : (ν + 2) / 2 + 1 = (ν / 2 + 1) + 1 := by omega
  rw [h2, Finset.sum_range_succ', Finset.smul_sum, pow_zero, one_smul, Nat.mul_zero, Nat.sub_zero, add_comm]
  congr 1
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [pow_succ', mul_smul, show ν + 2 - 2 * (j + 1) = ν - 2 * j by omega]

lemma Tfull_zero (p : ℕ) (hp : p.Prime) (τ : ∀ μ : ℕ, TauR R (gP p μ hp) eQ) (c : R) (n : ℕ) :
    Tfull ρ p hp τ c n 0 = 1 := by
  simp [Tfull, Tprim]

lemma Tfull_one (p : ℕ) (hp : p.Prime) (τ : ∀ μ : ℕ, TauR R (gP p μ hp) eQ) (c : R) (n : ℕ) :
    Tfull ρ p hp τ c n 1 = Tprim ρ p hp τ n 1 := by
  simp [Tfull]

/-- **M4-c, [GT] `thm:relations-psq-model`** (`\gtmodlit` over `hTp`): the `p^n I` difference
`T_p² = T_{Δ^prim_{p²}} + (p + 1) c`. -/
theorem Tp_sq (hTp : HeckeTpPrim ρ) : Tp_sq_stmt ρ := by
  intro p hp z hz c hs hρ h6 τ₁ τ₂ n
  let F : ∀ μ : ℕ, TauR R (gP p μ hp) eQ := fun μ => Classical.choice (tauR_nonempty R h6 (gP p μ hp))
  have e1 : Tdc ρ (cdP p 1 hp le_rfl) τ₁ n = Tprim ρ p hp F n 1 := by
    rw [Tprim_pos ρ p hp F n 1 le_rfl]; exact heckeTildeR_indep_tau ρ h6 _ _ _ n
  have e2 : Tdc ρ (cdP p 2 hp (by norm_num)) τ₂ n = Tprim ρ p hp F n 2 := by
    rw [Tprim_pos ρ p hp F n 2 (by norm_num)]; exact heckeTildeR_indep_tau ρ h6 _ _ _ n
  rw [e1, e2]
  exact hTp.sq p hp z hz c hs hρ h6 F n

/-- The recursion as ring algebra: from the assembly `F (m + 2) = P (m + 2) + c F m`, `F 0 = 1`, `F 1 = P 1`, and the two
primitive products, `P 1 F (ν + 1) = F (ν + 2) + a c F ν` (`a = p`). -/
lemma full_recursion_alg {A : Type*} [Ring A] [Algebra R A] (P F : ℕ → A) (a c : R) (hF0 : F 0 = 1) (hF1 : F 1 = P 1)
    (hF2 : ∀ m, F (m + 2) = P (m + 2) + c • F m) (hsq : P 1 * P 1 = P 2 + (a + 1) • c • (1 : A))
    (hsucc : ∀ μ, 1 ≤ μ → P 1 * P (μ + 1) = P (μ + 2) + (a * c) • P μ) :
    ∀ ν, P 1 * F (ν + 1) = F (ν + 2) + (a * c) • F ν := by
  intro ν
  induction ν using Nat.strong_induction_on with
  | _ ν ih =>
    rcases ν with _ | _ | ν
    · rw [hF1, hsq, zero_add, hF2 0, hF0, smul_smul, add_mul, one_mul, add_smul]
      abel
    · show P 1 * F (0 + 2) = F (1 + 2) + (a * c) • F 1
      rw [hF2 0, hF0, mul_add, mul_smul_comm, mul_one, hsucc 1 le_rfl, hF2 1, hF1]
      abel
    · show P 1 * F ((ν + 1) + 2) = F ((ν + 2) + 2) + (a * c) • F (ν + 2)
      rw [hF2 (ν + 1), mul_add, mul_smul_comm, show ν + 1 + 2 = (ν + 2) + 1 from rfl, hsucc (ν + 2) (by omega),
        ih ν (by omega), hF2 (ν + 2), hF2 ν]
      simp only [smul_add, smul_smul, mul_comm c (a * c)]
      abel

/-- **M4-d, [GT] `thm:relations-1-model`** (`\gtmodlit` over `hTp`): `T_{p^{ν+2}} = T_p T_{p^{ν+1}} − p c T_{p^ν}`. -/
theorem relations (hTp : HeckeTpPrim ρ) : relations_stmt ρ := by
  intro p hp z hz c hs hρ h6 τ n ν
  have key := full_recursion_alg (Tprim ρ p hp τ n) (Tfull ρ p hp τ c n) (p : R) c (Tfull_zero ρ p hp τ c n)
    (Tfull_one ρ p hp τ c n) (Tfull_add_two ρ p hp τ c n) (hTp.sq p hp z hz c hs hρ h6 τ n)
    (hTp.succ p hp z hz c hs hρ h6 τ n) ν
  rw [key]; abel

/-- **[GT] `lem:m4-hTp-hrel`** (L20-0 §3.2 check (a); `\gtmodlit` over `hTp`): at `A = End_R(H^n)`, `hTp` gives
`bridge:fullprimitive`'s hypothesis `hrel` (`T_p² = T_{p²} + p T(p,p)`) and its `hfull` (`T_{p²} = T_prim + T(p,p)`), with
`T_{p²} = Tfull 2`, `T_prim = Tprim 2`, `T(p,p) = c`. So the two literature routes agree. -/
def hTp_imp_hrel_stmt : Prop :=
  ∀ (p : ℕ) (hp : p.Prime) (z : GLQ) (hz : z ∈ ΔQ) (c : R), IsScalarQ z p → ρ ⟨z, hz⟩ = c • 1 → IsUnit (6 : R) →
    ∀ (τ : ∀ μ : ℕ, TauR R (gP p μ hp) eQ) (n : ℕ),
      Tprim ρ p hp τ n 1 * Tprim ρ p hp τ n 1 =
          Tfull ρ p hp τ c n 2 + (p : Module.End R (HM ρ n)) * (c • (1 : Module.End R (HM ρ n))) ∧
        Tfull ρ p hp τ c n 2 = Tprim ρ p hp τ n 2 + c • (1 : Module.End R (HM ρ n))

theorem hTp_imp_hrel (hTp : HeckeTpPrim ρ) : hTp_imp_hrel_stmt ρ := by
  intro p hp z hz c hs hρ h6 τ n
  have hfull : Tfull ρ p hp τ c n 2 = Tprim ρ p hp τ n 2 + c • (1 : Module.End R (HM ρ n)) := by
    rw [Tfull_add_two ρ p hp τ c n 0, Tfull_zero]
  refine ⟨?_, hfull⟩
  rw [hTp.sq p hp z hz c hs hρ h6 τ n, hfull, add_smul, one_smul, add_assoc, add_comm ((p : R) • c • _),
    Nat.cast_smul_eq_nsmul R, nsmul_eq_mul]

/-- `hTp_imp_hrel` fed to `GL2.prim_recursion` at `c = p^n` ([C] `obl:algebra`(ii), (iii) on the model). -/
def hTp_prim_recursion_stmt : Prop :=
  ∀ (n p : ℕ) (hp : p.Prime) (z : GLQ) (hz : z ∈ ΔQ), IsScalarQ z p → ρ ⟨z, hz⟩ = ((p : R) ^ n) • 1 →
    IsUnit (6 : R) → ∀ (τ : ∀ μ : ℕ, TauR R (gP p μ hp) eQ),
      Tfull ρ p hp τ ((p : R) ^ n) n 2 = Tprim ρ p hp τ n 1 * Tprim ρ p hp τ n 1 - (p : Module.End R (HM ρ n)) ^ (n + 1) ∧
        Tprim ρ p hp τ n 2 = Tprim ρ p hp τ n 1 * Tprim ρ p hp τ n 1 - (p : Module.End R (HM ρ n)) ^ (n + 1) -
          (p : Module.End R (HM ρ n)) ^ n

theorem hTp_prim_recursion (hTp : HeckeTpPrim ρ) : hTp_prim_recursion_stmt ρ := by
  intro n p hp z hz hs hρ h6 τ
  obtain ⟨hrel, hfull⟩ := hTp_imp_hrel ρ hTp p hp z hz _ hs hρ h6 τ n
  have hpp : ((p : R) ^ n) • (1 : Module.End R (HM ρ n)) = (p : Module.End R (HM ρ n)) ^ n := by
    rw [← Nat.cast_pow, Nat.cast_smul_eq_nsmul R, nsmul_eq_mul, mul_one, Nat.cast_pow]
  exact GL2.prim_recursion n p _ _ _ _ hrel hfull hpp

/-- **M4-d, [GT] `thm:relations-2-model`** (`\gtmodlit` over `hcop`): coprime multiplicativity, primitive form, for every
coset datum and `τ` (the hypothesis is for one datum each; `TdcOf_eq` gives the rest). -/
theorem coprime (hcop : HeckeCoprime ρ) : coprime_stmt ρ := by
  intro h6 M N hM hN hM0 hN0 hMN cd₁ τ₁ cd₂ τ₂ cd₃ τ₃ n
  rw [Tdc_eq_TdcOf ρ h6 cd₁, Tdc_eq_TdcOf ρ h6 cd₂, Tdc_eq_TdcOf ρ h6 cd₃]
  exact hcop.mul h6 M N hM hN hM0 hN0 hMN n

/-- **M4-e, [GT] `thm:relations-comm-model`** (`\gtmodlit` over `hcomm`): any two double-coset operators commute. -/
theorem Tdc_commute (hcomm : HeckeCommute ρ) : Tdc_commute_stmt ρ := by
  intro h6 g g' cd τ cd' τ' n
  rw [Tdc_eq_TdcOf ρ h6 cd, Tdc_eq_TdcOf ρ h6 cd']
  exact hcomm.comm h6 g _ g' _ n

/-- An `R`-subalgebra generated by pairwise commuting elements is commutative. -/
lemma commute_of_mem_adjoin {A : Type*} [Ring A] [Algebra R A] {s : Set A} (hs : ∀ x ∈ s, ∀ y ∈ s, Commute x y)
    {a b : A} (ha : a ∈ Algebra.adjoin R s) (hb : b ∈ Algebra.adjoin R s) : Commute a b := by
  have h1 : ∀ x ∈ s, ∀ b ∈ Algebra.adjoin R s, Commute x b := by
    intro x hx b hb
    induction hb using Algebra.adjoin_induction with
    | mem y hy => exact hs x hx y hy
    | algebraMap r => exact Algebra.commutes r x |>.symm
    | add y z _ _ hy hz => exact hy.add_right hz
    | mul y z _ _ hy hz => exact hy.mul_right hz
  induction ha using Algebra.adjoin_induction with
  | mem y hy => exact h1 y hy b hb
  | algebraMap r => exact Algebra.commutes r b
  | add y z _ _ hy hz => exact hy.add_left hz
  | mul y z _ _ hy hz => exact hy.mul_left hz

/-- **M4-e, [GT] `obl:algebra-model`** (`\gtmodlit` over `hcomm`): `heckeAlg ρ n` is commutative. -/
theorem heckeAlg_comm (hcomm : HeckeCommute ρ) : heckeAlg_comm_stmt ρ := by
  intro h6 n a ha b hb
  refine (commute_of_mem_adjoin ?_ ha hb).eq
  rintro _ ⟨g, cd, τ, rfl⟩ _ ⟨g', cd', τ', rfl⟩
  exact Tdc_commute ρ hcomm h6 g g' cd τ cd' τ' n

end GL2.M4

end
