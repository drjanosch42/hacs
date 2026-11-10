/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import Mathlib
import GL2.HeckeZ.Exec.Bock
import GL2.HeckeZ.Check.Cert
import GL2.HeckeZ.Bock.Split
import GL2.HeckeZ.Bock.Glue

/-!
# HeckeZ K5, K6, K7: soundness and the cell statements (L-HZ2, M-HZ3)

`07-lean-plan.md` §5.5–§5.7, §10 (M-HZ3); `03-bockstein-tor.md` §8.2. The executable checkers are Mathlib-free
(`GL2/HeckeZ/Exec/Bock.lean`); here:
* **K5** `k5Check_sound`: the emitted `h^q` equal `b_q + t_q + t_{q+1}` (B2(1) at `K = 1`; LB-2 is the statement that this
  is the dimension predicted by the groups, `GL2/HeckeZ/Bock/Types.lean`).
* **K6** `k6_sound`: with **TB-χ** (`hχ`: the recorded `χ` is the characteristic polynomial of `T_p` on `L = H/tors`) and
  **TB-mat** (`hmat`: `T_p` acts on the export's torsion generators `g_i`, of orders `ℓ^{σ_i}`, by the recorded matrix; `hgen`:
  they generate `tors_ℓ`), a passing certificate gives `Hom_{ℤ[T_p]}(H/tors_ℓ, tors_ℓ) = 0` (B7 through LB-4's
  `BezoutNil`), hence `J_ℓ = 0` for every commutative `𝕋 ∋ T_p` (`k6_J_eq_zero`).
* **K7** (cyclic `tors_ℓ = ℤ/ℓ^e`): with **TB-mat** (`K7Model`: the export's generators `g`, `f_i` and matrices on `H/ℓ^e`),
  `k7_hom_eq` (two elements of `Hom_{𝕋'}(H/tors_ℓ, tors_ℓ)` with the same Smith read are equal, the read is killed by `ℓ^{v_k}`),
  `k7_hom_zero` (status `zero`: all `v_k = 0` gives `Hom_{𝕋'} = 0`), `k7_card_le` (`|Hom_{𝕋'}| ≤ ℓ^{Σ v_k}`, the upper bound of the
  sandwich) and `k7_split` (a section gives a `𝕋'`-stable complement, the lower bound `J ∩ 𝕋' = 0`).

**Trusted inputs (Q10(a), [LP] v1.9).** Exactly the named hypotheses `hχ` (tb_chi) and `hmat` / `hgen` / `K7Model` (tb_mat, the
slot-25 EXPORT `DescentRecord.module(ell, K)`); they are stated inline as hypotheses of every theorem that uses them. Nothing the
generator computes (`∏ P_j`, `m`, Bezout pairs, sections, Smith transforms) is trusted.
-/

namespace GL2.HeckeZ.Check

open GL2.HeckeZ.Exec GL2.HeckeZ.Bock Polynomial

/-! ### K5 -/

/-- What a passing K5 certificate gives: B2(1)'s count `h^q = b_q + t_q + t_{q+1}`, `q = 0 … 3` (`t_4 := t_2`). -/
structure K5Spec (c : K5Cert) : Prop where
  len : c.b.length = 4 ∧ c.mu.length = 4 ∧ c.h.length = 4
  eq : ∀ q < 4, c.h.getD q 0 = c.b.getD q 0 + c.t q + c.t (q + 1)

/-- **Soundness of K5.** -/
theorem k5Check_sound {c : K5Cert} (h : k5Check c = true) : K5Spec c := by
  simp only [k5Check, Bool.and_eq_true, beq_iff_eq, List.all_eq_true, List.mem_range] at h
  obtain ⟨⟨⟨hb, hm⟩, hh⟩, hq⟩ := h
  exact ⟨⟨hb, hm, hh⟩, fun q hq' => by simpa [K5Cert.pred] using hq q hq'⟩

/-! ### Coordinate vectors -/

section Coord

variable {H : Type*} [AddCommGroup H]

/-- The element `Σ_{i<t} v_i g_i`. -/
def comb (t : ℕ) (g : ℕ → H) (v : List ℤ) : H := ∑ i ∈ Finset.range t, v.getD i 0 • g i

lemma getD_map_range {t i : ℕ} (hi : i < t) (f : ℕ → ℤ) : ((List.range t).map f).getD i 0 = f i := by
  rw [List.getD_eq_getElem _ _ (by simpa using hi)]; simp

variable {t : ℕ} {g : ℕ → H}

lemma comb_vadd (v w : List ℤ) : comb t g (vadd t v w) = comb t g v + comb t g w := by
  unfold comb; rw [← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun i hi => ?_
  rw [vadd, getD_map_range (Finset.mem_range.mp hi), add_smul]

lemma comb_vscale (a : ℤ) (v : List ℤ) : comb t g (vscale t a v) = a • comb t g v := by
  unfold comb; rw [Finset.smul_sum]
  refine Finset.sum_congr rfl fun i hi => ?_
  rw [vscale, getD_map_range (Finset.mem_range.mp hi), mul_smul]

lemma comb_zeros : comb t g ((List.range t).map fun _ => 0) = 0 := by
  unfold comb
  exact Finset.sum_eq_zero fun i hi => by rw [getD_map_range (Finset.mem_range.mp hi), zero_smul]

lemma smul_eq_emod_smul {M : ℤ} {x : H} (hx : M • x = 0) (a : ℤ) : a • x = (a % M) • x := by
  conv_lhs => rw [← Int.emod_add_ediv_mul a M]
  rw [add_smul, mul_smul, hx, smul_zero, add_zero]

variable {ℓ : ℕ} {sig : List ℕ}

lemma comb_vred (hord : ∀ i < t, ((ℓ ^ sig.getD i 0 : ℕ) : ℤ) • g i = 0) (v : List ℤ) :
    comb t g (vred t ℓ sig v) = comb t g v := by
  unfold comb
  refine Finset.sum_congr rfl fun i hi => ?_
  rw [vred, getD_map_range (Finset.mem_range.mp hi)]
  exact (smul_eq_emod_smul (hord i (Finset.mem_range.mp hi)) _).symm

lemma comb_eq_zero_of_vanishes (hord : ∀ i < t, ((ℓ ^ sig.getD i 0 : ℕ) : ℤ) • g i = 0) {v : List ℤ}
    (h : vanishes t ℓ sig v = true) : comb t g v = 0 := by
  unfold comb
  refine Finset.sum_eq_zero fun i hi => ?_
  have hi' := Finset.mem_range.mp hi
  have := List.all_eq_true.mp h i (List.mem_range.mpr hi')
  rw [smul_eq_emod_smul (hord i hi'), (beq_iff_eq.mp this), zero_smul]

lemma comb_unitVec {j : ℕ} (hj : j < t) : comb t g (unitVec t j) = g j := by
  unfold comb
  rw [Finset.sum_eq_single j]
  · rw [unitVec, getD_map_range hj, if_pos rfl, one_smul]
  · intro i hi hij
    rw [unitVec, getD_map_range (Finset.mem_range.mp hi), if_neg hij, zero_smul]
  · intro h; exact absurd (Finset.mem_range.mpr hj) h

variable {C : List (List ℤ)} {θ : Module.End ℤ H}

lemma comb_matVec (hact : ∀ j < t, θ (g j) = ∑ i ∈ Finset.range t, ientry C i j • g i) (v : List ℤ) :
    θ (comb t g v) = comb t g (matVec t C v) := by
  unfold comb
  rw [map_sum]
  simp_rw [map_zsmul]
  rw [Finset.sum_congr rfl fun j hj => by rw [hact j (Finset.mem_range.mp hj)]]
  simp_rw [Finset.smul_sum, smul_smul]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun i hi => ?_
  rw [matVec, getD_map_range (Finset.mem_range.mp hi), list_sum_range_map, Finset.sum_smul]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [mul_comm]

lemma comb_polyVec (hord : ∀ i < t, ((ℓ ^ sig.getD i 0 : ℕ) : ℤ) • g i = 0)
    (hact : ∀ j < t, θ (g j) = ∑ i ∈ Finset.range t, ientry C i j • g i) (P v : List ℤ) :
    aeval θ (toPoly P) (comb t g v) = comb t g (polyVec t ℓ sig C P v) := by
  induction P with
  | nil =>
    simp only [toPoly, map_zero, LinearMap.zero_apply, polyVec]
    exact comb_zeros.symm
  | cons a P ih =>
    rw [toPoly, map_add, map_mul, aeval_C, aeval_X, LinearMap.add_apply, Module.End.mul_apply, ih,
      Module.algebraMap_end_apply, polyVec, comb_vred hord, comb_vadd, comb_vscale, comb_matVec hact]

lemma comb_polyPowVec (hord : ∀ i < t, ((ℓ ^ sig.getD i 0 : ℕ) : ℤ) • g i = 0)
    (hact : ∀ j < t, θ (g j) = ∑ i ∈ Finset.range t, ientry C i j • g i) (P v : List ℤ) (m : ℕ) :
    ((aeval θ (toPoly P)) ^ m) (comb t g v) = comb t g (polyPowVec t ℓ sig C P m v) := by
  induction m with
  | zero => simp [polyPowVec]
  | succ m ih => rw [pow_succ', Module.End.mul_apply, ih, comb_polyVec hord hact, polyPowVec]

end Coord

/-! ### K6 -/

section K6

variable {H : Type*} [AddCommGroup H]

/-- **TB-mat** on the torsion generators (the export's torsion block): `ℓ^{σ_i} g_i = 0` and
`θ g_j = Σ_i C_{ij} g_i`. -/
structure TorsModel (ℓ : ℕ) (θ : Module.End ℤ H) (sig : List ℕ) (C : List (List ℤ)) (g : ℕ → H) : Prop where
  ord : ∀ i < sig.length, ((ℓ ^ sig.getD i 0 : ℕ) : ℤ) • g i = 0
  act : ∀ j < sig.length, θ (g j) = ∑ i ∈ Finset.range sig.length, ientry C i j • g i

/-- The span of the first `t` generators. -/
def torsSpan (t : ℕ) (g : ℕ → H) : Submodule ℤ H := Submodule.span ℤ (g '' {i | i < t})

/-- `θ` induces an endomorphism of `L = H/tors`. -/
theorem torsion_le_comap (θ : Module.End ℤ H) :
    Submodule.torsion ℤ H ≤ (Submodule.torsion ℤ H).comap θ := by
  intro x hx
  obtain ⟨a, ha⟩ := (Submodule.mem_torsion_iff x).mp hx
  refine (Submodule.mem_torsion_iff _).mpr ⟨a, ?_⟩
  rw [Submonoid.smul_def] at ha ⊢
  rw [← map_zsmul, ha, map_zero]

/-- The endomorphism `T̄` of `L = H/tors` induced by `T`. -/
noncomputable def torsMap (θ : Module.End ℤ H) : Module.End ℤ (H ⧸ Submodule.torsion ℤ H) :=
  (Submodule.torsion ℤ H).mapQ (Submodule.torsion ℤ H) θ (torsion_le_comap θ)

lemma torsMap_aeval (θ : Module.End ℤ H) (P : ℤ[X]) (x : H) :
    aeval (torsMap θ) P (Submodule.Quotient.mk x) = Submodule.Quotient.mk (aeval θ P x) := by
  induction P using Polynomial.induction_on generalizing x with
  | C a => simp
  | add p q hp hq => simp only [map_add, LinearMap.add_apply, hp, hq, Submodule.Quotient.mk_add]
  | monomial n a ih =>
    have e : (C a * X ^ (n + 1) : ℤ[X]) = (C a * X ^ n) * X := by ring
    rw [e, map_mul, aeval_X, Module.End.mul_apply, torsMap, Submodule.mapQ_apply, ← torsMap, ih,
      map_mul (aeval θ) (C a * X ^ n) X, aeval_X, Module.End.mul_apply]

/-- **TB-χ ⇒ `χ(T) H ⊆ tors`** (Cayley–Hamilton on `L`). -/
theorem aeval_mem_torsion [Module.Free ℤ (H ⧸ Submodule.torsion ℤ H)] [Module.Finite ℤ (H ⧸ Submodule.torsion ℤ H)]
    (θ : Module.End ℤ H) {P : ℤ[X]} (hχ : (torsMap θ).charpoly = P) (x : H) :
    aeval θ P x ∈ Submodule.torsion ℤ H := by
  have h := congrArg (· (Submodule.Quotient.mk x : H ⧸ Submodule.torsion ℤ H)) (LinearMap.aeval_self_charpoly (torsMap θ))
  simp only [hχ, torsMap_aeval, LinearMap.zero_apply] at h
  exact (Submodule.Quotient.mk_eq_zero _).mp h

/-- An endomorphism with values in `S` (killed by `ℓ^E`), killing `S ⊇ tors[ℓ^∞]`, kills all of `tors`. -/
theorem eq_zero_of_mem_torsion {ℓ : ℕ} (hℓ : ℓ.Prime) {S : Submodule ℤ H}
    (hS : ∀ x (j : ℕ), ((ℓ : ℤ) ^ j) • x = 0 → x ∈ S) {E : ℕ} (hE : ∀ x ∈ S, ((ℓ : ℤ) ^ E) • x = 0)
    (f : Module.End ℤ H) (hf1 : ∀ x, f x ∈ S) (hf2 : ∀ x ∈ S, f x = 0) {y : H}
    (hy : y ∈ Submodule.torsion ℤ H) : f y = 0 := by
  obtain ⟨⟨n, hn⟩, hny⟩ := (Submodule.mem_torsion_iff y).mp hy
  have hn0 : n ≠ 0 := nonZeroDivisors.ne_zero hn
  have hny0 : n • y = 0 := by simpa [Submonoid.smul_def] using hny
  have hny' : (n.natAbs : ℤ) • y = 0 := by
    rcases Int.natAbs_eq n with h | h
    · rw [← h]; exact hny0
    · have : (n.natAbs : ℤ) = -n := by omega
      rw [this, neg_smul, hny0, neg_zero]
  obtain ⟨a, m, hm, hnm⟩ := Nat.exists_eq_pow_mul_and_not_dvd (Int.natAbs_ne_zero.mpr hn0) ℓ hℓ.one_lt.ne'
  have hcop : ∀ k : ℕ, IsCoprime ((ℓ : ℤ) ^ k) (m : ℤ) := fun k => by
    rw [← Nat.cast_pow, Nat.isCoprime_iff_coprime]
    exact Nat.Coprime.pow_left k ((Nat.Prime.coprime_iff_not_dvd hℓ).mpr hm)
  obtain ⟨α, β, hαβ⟩ := hcop a
  have hy' : y = (α * (ℓ : ℤ) ^ a) • y + (β * m) • y := by rw [← add_smul, hαβ, one_smul]
  have hprod : (((ℓ : ℤ) ^ a) * m) • y = 0 := by
    rw [show ((ℓ : ℤ) ^ a) * m = (n.natAbs : ℤ) by rw [hnm]; push_cast; ring]; exact hny'
  have h1 : f ((β * m) • y) = 0 := by
    refine hf2 _ (hS _ a ?_)
    rw [smul_smul, show (ℓ : ℤ) ^ a * (β * m) = β * ((ℓ : ℤ) ^ a * m) by ring, mul_smul, hprod, smul_zero]
  have h2 : f ((α * (ℓ : ℤ) ^ a) • y) = 0 := by
    set z := f ((α * (ℓ : ℤ) ^ a) • y)
    have hzm : (m : ℤ) • z = 0 := by
      rw [← map_zsmul, smul_smul, show (m : ℤ) * (α * (ℓ : ℤ) ^ a) = α * ((ℓ : ℤ) ^ a * m) by ring, mul_smul, hprod,
        smul_zero, map_zero]
    have hzl : ((ℓ : ℤ) ^ E) • z = 0 := hE z (hf1 _)
    obtain ⟨α', β', h'⟩ := hcop E
    calc z = (α' * (ℓ : ℤ) ^ E + β' * m) • z := by rw [h', one_smul]
      _ = 0 := by rw [add_smul, mul_smul, mul_smul, hzl, hzm, smul_zero, smul_zero, add_zero]
  rw [hy', map_add, h1, h2, add_zero]

/-- A passing `peqMod`: `p − q` has all coefficients divisible by `m`. -/
theorem peqMod_sound {m : ℕ} {p q : List ℤ} (h : peqMod m p q = true) :
    ∃ W : ℤ[X], toPoly p = toPoly q + C (m : ℤ) * W := by
  set l := padd p (pscale (-1) q)
  have hl : ∀ a ∈ l, (m : ℤ) ∣ a := fun a ha => by
    have := List.all_eq_true.mp h a ha
    exact Int.dvd_of_emod_eq_zero (beq_iff_eq.mp this)
  have key : ∀ l : List ℤ, (∀ a ∈ l, (m : ℤ) ∣ a) → toPoly l = C (m : ℤ) * toPoly (l.map (· / (m : ℤ))) := by
    intro l
    induction l with
    | nil => simp [toPoly]
    | cons a l ih =>
      intro hd
      rw [List.map_cons, toPoly, toPoly, ih fun b hb => hd b (List.mem_cons_of_mem a hb), mul_add,
        ← C_mul, Int.mul_ediv_cancel' (hd a List.mem_cons_self)]
      ring
  refine ⟨toPoly (l.map (· / (m : ℤ))), ?_⟩
  rw [← key l hl, toPoly_padd, toPoly_pscale]
  simp

/-- `θ` preserves the span of the torsion generators. -/
theorem torsSpan_stable {ℓ : ℕ} {θ : Module.End ℤ H} {sig : List ℕ} {C : List (List ℤ)} {g : ℕ → H}
    (hmat : TorsModel ℓ θ sig C g) : ∀ x ∈ torsSpan sig.length g, θ x ∈ torsSpan sig.length g := by
  intro x hx
  induction hx using Submodule.span_induction with
  | mem y hy =>
    obtain ⟨j, hj, rfl⟩ := hy
    rw [hmat.act j hj]
    exact Submodule.sum_mem _ fun i hi =>
      Submodule.smul_mem _ _ (Submodule.subset_span ⟨i, Finset.mem_range.mp hi, rfl⟩)
  | zero => simp
  | add y z _ _ hy hz => rw [map_add]; exact Submodule.add_mem _ hy hz
  | smul a y _ hy => rw [map_smul]; exact Submodule.smul_mem _ a hy

/-- **Soundness of K6.** With TB-χ (`hχ`) and TB-mat (`hmat`, `hgen`), a passing certificate gives
`Hom_{ℤ[T_p]}(H/tors_ℓ, tors_ℓ) = 0`: every endomorphism with values in `tors_ℓ`, killing `tors_ℓ` and commuting with `T_p`
is zero (03 B7, B11(3); the separation `gcd(χ̄, ∏ P_j) = 1`). -/
theorem k6_sound {c : K6Cert} (hc : k6Check c = true) (hℓ : c.ell.Prime)
    {H : Type*} [AddCommGroup H] [Module.Free ℤ (H ⧸ Submodule.torsion ℤ H)]
    [Module.Finite ℤ (H ⧸ Submodule.torsion ℤ H)] (θ : Module.End ℤ H) (g : ℕ → H)
    (hχ : (torsMap θ).charpoly = toPoly c.chi)
    (hmat : TorsModel c.ell θ c.sig c.C g)
    (hgen : ∀ x (j : ℕ), ((c.ell : ℤ) ^ j) • x = 0 → x ∈ torsSpan c.sig.length g) :
    ∀ f ∈ HomP ({θ} : Set (Module.End ℤ H)) (torsSpan c.sig.length g), f = 0 := by
  simp only [k6Check, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true, List.mem_range] at hc
  obtain ⟨⟨_, hbez⟩, hnil⟩ := hc
  set S := torsSpan c.sig.length g
  obtain ⟨W, hW⟩ := peqMod_sound hbez
  rw [toPoly_padd, toPoly_pmul, toPoly_pmul, toPoly_single] at hW
  -- `ℓ^E` kills `S`
  have hgE : ∀ i < c.sig.length, ((c.ell : ℤ) ^ c.sig.sum) • g i = 0 := by
    intro i hi
    have hle : c.sig.getD i 0 ≤ c.sig.sum := by
      rw [List.getD_eq_getElem _ _ hi]; exact List.le_sum_of_mem (List.getElem_mem hi)
    rw [← Nat.sub_add_cancel hle, pow_add, mul_smul]
    have := hmat.ord i hi
    push_cast at this
    rw [this, smul_zero]
  have hE : ∀ x ∈ S, ((c.ell : ℤ) ^ c.sig.sum) • x = 0 := by
    intro x hx
    induction hx using Submodule.span_induction with
    | mem y hy => obtain ⟨i, hi, rfl⟩ := hy; exact hgE i hi
    | zero => simp
    | add y z _ _ hy hz => rw [smul_add, hy, hz, add_zero]
    | smul a y _ hy => rw [smul_comm, hy, smul_zero]
  -- `P(θ)^m` kills `S`
  have hG : ∀ x ∈ S, ((aeval θ (toPoly c.P)) ^ c.m) x = 0 := by
    intro x hx
    induction hx using Submodule.span_induction with
    | mem y hy =>
      obtain ⟨j, hj, rfl⟩ := hy
      rw [← comb_unitVec (t := c.sig.length) (g := g) hj, comb_polyPowVec hmat.ord hmat.act]
      exact comb_eq_zero_of_vanishes hmat.ord (hnil j hj)
    | zero => simp
    | add y z _ _ hy hz => rw [map_add, hy, hz, add_zero]
    | smul a y _ hy => rw [map_smul, hy, smul_zero]
  have hN : BezoutNil (c.ell : ℤ) θ S (toPoly c.chi) (toPoly c.P) (toPoly c.u) (toPoly c.v) W :=
    ⟨torsSpan_stable hmat, by rw [hW]; simp, ⟨c.m, hG⟩, ⟨c.sig.sum, hE⟩⟩
  rintro f ⟨hf1, hf2, hfc⟩
  ext x
  refine hN.eq_zero_of_mem (hf1 x) ?_
  have hcomm := congrArg (· x) (commute_aeval (hfc θ rfl) (toPoly c.chi)).eq
  simp only [Module.End.mul_apply] at hcomm
  rw [← hcomm]
  exact eq_zero_of_mem_torsion hℓ hgen hE f hf1 hf2 (aeval_mem_torsion θ hχ x)

/-- **K6, status `zero`**: `J_ℓ = 0` (relative to `tors_ℓ`) for every set `𝕋` of endomorphisms commuting with `T_p`. -/
theorem k6_J_eq_zero {c : K6Cert} (hc : k6Check c = true) (hℓ : c.ell.Prime)
    {H : Type*} [AddCommGroup H] [Module.Free ℤ (H ⧸ Submodule.torsion ℤ H)]
    [Module.Finite ℤ (H ⧸ Submodule.torsion ℤ H)] (θ : Module.End ℤ H) (g : ℕ → H)
    (hχ : (torsMap θ).charpoly = toPoly c.chi)
    (hmat : TorsModel c.ell θ c.sig c.C g)
    (hgen : ∀ x (j : ℕ), ((c.ell : ℤ) ^ j) • x = 0 → x ∈ torsSpan c.sig.length g)
    (T : Set (Module.End ℤ H)) (hT : ∀ t ∈ T, Commute t θ) :
    ∀ t ∈ Jset T (torsSpan c.sig.length g), t = 0 :=
  Jset_eq_zero_of_HomP (P := {θ}) (fun t ht x hx => by rw [hx]; exact hT t ht) (k6_sound hc hℓ θ g hχ hmat hgen)

end K6

end GL2.HeckeZ.Check
