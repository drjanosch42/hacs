/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import Mathlib
import GL2.HeckeZ.Exec.K3N
import GL2.HeckeZ.Check.K3
import GL2.HeckeZ.Check.K10

/-!
# HeckeZ K3N: specification, soundness, and the bridge to the congruence module (L-HZ3, M-HZ4b, partly)

`07-lean-plan.md` §5.3 (K3) at level `N`, two systems. The executable checker `GL2.HeckeZ.Exec.K3N.k3nCheck` is Mathlib-free;
here:
* `K3NSpec c` — the integer facts the checker establishes (`k3nCheck_sound`);
* `read_lower` — one read: under TB-mat for that read (`U · [θ]_b ≡ D · U mod ℓ^K`, `U` invertible), `ℓ^e ∣ λ t − μ t` for every
  `t ∈ 𝕋` (01 Prop. 5.4 at a root, read entrywise; the proof is `GL2.HeckeZ.K3.k3_sound`'s, for one read);
* `k3n_sound` — all reads: `G = ∏ ℓ^e ∣ λ t − μ t` for every `t` (the reads sit at pairwise distinct primes), and
  `λθ − μθ` lies in the congruence ideal (01 Prop. 5.3);
* `k3n_fibre` — when the certificate is complete (`G = |λθ − μθ|`): `𝕋 ≅ ℤ ×_{ℤ/g} ℤ`, `C_𝕋 ≅ ℤ/g`, and `𝕋 = ℤ[θ]` (`k3n_gen`).

**Trust boundary.** TB-mat is the hypothesis `hmat` (one basis and one unit per read); `λθ`, `μθ` enter through `hl`, `hm`, which
in the cells come from `GL2.HeckeZ.BridgeN.eisSplit_of_rank2` (TB-χ and `hES_N_an`).
-/

namespace GL2.HeckeZ.K3N

open GL2.HeckeZ GL2.HeckeZ.Exec GL2.HeckeZ.Exec.K3N GL2.HeckeZ.K3

/-- What a read establishes, as integer statements. -/
structure ReadSpec (r : K3Read) (lam mu : ℤ) : Prop where
  prime : r.ell.Prime
  hv : r.v + 1 ≤ r.K
  all : ∀ i j : Fin r.f, (r.ell : ℤ) ^ r.v ∣ (entry r.dump i j : ℤ) - if i = j then mu else 0
  wit : ∃ i j : Fin r.f, ¬ (r.ell : ℤ) ^ (r.v + 1) ∣ (entry r.dump i j : ℤ) - if i = j then mu else 0
  gap : (r.ell : ℤ) ^ (r.v + r.e) ∣ lam - mu

/-- **`K3NSpec`**: what `k3nCheck` establishes. -/
structure K3NSpec (c : K3NCert) : Prop where
  p_prime : c.p.Prime
  p_ndvd : ¬ c.N ∣ c.p
  lam_eq : c.lam = 1 + (c.p : ℤ) ^ (c.k - 1)
  hne : c.lam ≠ c.mu
  nodup : (c.reads.map K3Read.ell).Nodup
  reads : ∀ r ∈ c.reads, ReadSpec r c.lam c.mu

/-! ### Soundness of the checker -/

lemma rEntry_dvd (r : K3Read) (h2 : 2 ≤ r.ell) (mu : ℤ) (i j : ℕ) {n : ℕ} (hn : n ∣ r.m) :
    (n ∣ rEntry r mu i j) ↔ (n : ℤ) ∣ (entry r.dump i j : ℤ) - if i = j then mu else 0 := by
  have hm : (0 : ℤ) < (r.m : ℤ) := by
    have : 0 < r.m := Nat.pow_pos (by omega)
    exact_mod_cast this
  have key : (r.m : ℤ) ∣ ((entry r.dump i j : ℤ) - if i = j then mu else 0) - (rEntry r mu i j : ℤ) := by
    unfold rEntry resid
    split_ifs with hij
    · rw [Int.natCast_mod, Nat.cast_add, Int.toNat_of_nonneg (Int.emod_nonneg _ hm.ne')]
      refine (Int.ModEq.dvd ?_)
      calc ((entry r.dump i j : ℤ) + (-mu) % (r.m : ℤ)) % (r.m : ℤ)
          ≡ (entry r.dump i j : ℤ) + (-mu) % (r.m : ℤ) [ZMOD r.m] := Int.mod_modEq _ _
        _ ≡ (entry r.dump i j : ℤ) + -mu [ZMOD r.m] := Int.ModEq.add_left _ (Int.mod_modEq _ _)
        _ = (entry r.dump i j : ℤ) - mu := by ring
    · rw [Nat.add_zero, Int.natCast_mod, sub_zero]
      exact Int.ModEq.dvd (Int.mod_modEq _ _)
  have hk : (n : ℤ) ∣ ((entry r.dump i j : ℤ) - if i = j then mu else 0) - (rEntry r mu i j : ℤ) :=
    (Int.natCast_dvd_natCast.mpr hn).trans key
  rw [← Int.natCast_dvd_natCast]
  constructor
  · intro h; have := dvd_add hk h; simpa using this
  · intro h; have := dvd_sub h hk; simpa using this

lemma readOK_sound {r : K3Read} {lam mu : ℤ} (h : Exec.K3N.readOK r lam mu = true) : ReadSpec r lam mu := by
  unfold Exec.K3N.readOK shapeB allDivB witB gapB at h
  simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq, bne_iff_ne, ne_eq] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨hpr, hvK⟩, _⟩, _⟩, hw1⟩, hw2⟩, hall⟩, hwit⟩, hgap⟩ := h
  have hp := GL2.HeckeZ.K10.primeB_sound hpr
  have h2 := hp.two_le
  have hvm : r.ell ^ r.v ∣ r.m := pow_dvd_pow _ (by omega)
  have hv1m : r.ell ^ (r.v + 1) ∣ r.m := pow_dvd_pow _ hvK
  refine ⟨hp, hvK, fun i j => ?_, ⟨⟨r.wit.1, hw1⟩, ⟨r.wit.2, hw2⟩, fun hd => ?_⟩, Int.dvd_of_emod_eq_zero hgap⟩
  · have := (allFin_iff _ _).mp hall i i.2 j j.2
    simp only [beq_iff_eq] at this
    have h' := (rEntry_dvd r h2 mu i j hvm).mp (Nat.dvd_of_mod_eq_zero this)
    simpa [Fin.ext_iff] using h'
  · have := (rEntry_dvd r h2 mu r.wit.1 r.wit.2 hv1m).mpr (by push_cast at hd ⊢; simpa [Fin.ext_iff] using hd)
    exact hwit (Nat.mod_eq_zero_of_dvd this)

lemma nodupB_iff (l : List ℕ) : nodupB l = true ↔ l.Nodup := by
  induction l with
  | nil => simp [nodupB]
  | cons a l ih => simp [nodupB, ih, List.nodup_cons]

/-- **Soundness of the checker.** -/
theorem k3nCheck_sound (c : K3NCert) (h : k3nCheck c = true) : K3NSpec c := by
  unfold k3nCheck opB at h
  simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq, bne_iff_ne, ne_eq, List.all_eq_true] at h
  obtain ⟨⟨⟨⟨⟨⟨hp, hN1⟩, hpN⟩, hlam⟩, hne⟩, hnd⟩, hr⟩ := h
  refine ⟨GL2.HeckeZ.K10.primeB_sound hp, fun hd => hpN (Nat.mod_eq_zero_of_dvd hd), hlam, hne,
    (nodupB_iff _).mp hnd, fun r hmem => readOK_sound (hr r hmem)⟩

/-! ### The bridge -/

variable {L : Type*} [AddCommGroup L] {T : Subalgebra ℤ (Module.End ℤ L)}

/-- TB-mat for one read: the dumped matrix is the matrix of `θ` on `L/ℓ^K` in some `ℤ`-basis, up to a unit. -/
def TBmat (r : K3Read) (θ : Module.End ℤ L) : Prop :=
  ∃ (b : Module.Basis (Fin r.f) ℤ L) (U : Matrix (Fin r.f) (Fin r.f) (ZMod (r.ell ^ r.K))), IsUnit U ∧
    U * (LinearMap.toMatrix b b θ).map (Int.cast) = (Matrix.of fun (i j : Fin r.f) => (entry r.dump i j : ZMod (r.ell ^ r.K))) * U

/-- **One read** (01 Prop. 5.4 at a root, read entrywise): `ℓ^e ∣ λ t − μ t` for every `t ∈ 𝕋`. -/
theorem read_lower (E : EisSplit T) (r : K3Read) (lam mu : ℤ) (hr : ReadSpec r lam mu) (hl : E.lam E.θ = lam)
    (hm : E.mu E.θ = mu) (hmat : TBmat r (E.θ : Module.End ℤ L)) (t : T) :
    (r.ell : ℤ) ^ r.e ∣ E.lam t - E.mu t := by
  obtain ⟨b, U, hU, hmt⟩ := hmat
  set A : Fin r.f → Fin r.f → ℤ := fun i j => b.repr (E.P (b j)) i with hA
  set C : Fin r.f → Fin r.f → ℤ := fun i j => (entry r.dump i j : ℤ) - if i = j then mu else 0 with hC
  have hAe : ∀ i j, A i j = LinearMap.toMatrix b b (E.θ : Module.End ℤ L) i j - if i = j then mu else 0 := by
    intro i j
    simp only [hA, EisSplit.P, hm, map_sub, LinearEquiv.map_smul, Finsupp.sub_apply, Finsupp.smul_apply,
      Module.Basis.repr_self, LinearMap.toMatrix_apply, smul_eq_mul, Finsupp.single_apply]
    by_cases h : i = j
    · subst h; simp
    · simp [h, Ne.symm h]
  have hconj : U * zmat (r.ell ^ r.K) A = zmat (r.ell ^ r.K) C * U := by
    have e1 : zmat (r.ell ^ r.K) A = (LinearMap.toMatrix b b (E.θ : Module.End ℤ L)).map (Int.cast) -
        (mu : ZMod (r.ell ^ r.K)) • (1 : Matrix (Fin r.f) (Fin r.f) (ZMod (r.ell ^ r.K))) := by
      ext i j; simp only [zmat, Matrix.of_apply, hAe, Matrix.sub_apply, Matrix.map_apply, Matrix.smul_apply,
        Matrix.one_apply, smul_eq_mul]; split_ifs <;> push_cast <;> ring
    have e2 : zmat (r.ell ^ r.K) C = (Matrix.of fun (i j : Fin r.f) => (entry r.dump i j : ZMod (r.ell ^ r.K))) -
        (mu : ZMod (r.ell ^ r.K)) • (1 : Matrix (Fin r.f) (Fin r.f) (ZMod (r.ell ^ r.K))) := by
      ext i j; simp only [zmat, hC, Matrix.of_apply, Matrix.sub_apply, Matrix.smul_apply, Matrix.one_apply,
        smul_eq_mul]; split_ifs <;> push_cast <;> ring
    rw [e1, e2, mul_sub, sub_mul, hmt, Matrix.mul_smul, Matrix.smul_mul, mul_one, one_mul]
  have hvK := hr.hv
  have hdv : r.ell ^ r.v ∣ r.ell ^ r.K := pow_dvd_pow _ (by omega)
  have hdv1 : r.ell ^ (r.v + 1) ∣ r.ell ^ r.K := pow_dvd_pow _ hvK
  have hAv : ∀ i j, ((r.ell ^ r.v : ℕ) : ℤ) ∣ A i j := by
    have := (zero_iff_of_conj hdv hU hconj).mpr (by
      rw [mapMatrix_zmat, zmat_eq_zero]; intro i j; push_cast; exact hr.all i j)
    rw [mapMatrix_zmat, zmat_eq_zero] at this; exact this
  have hAv1 : ∃ i j, ¬ ((r.ell ^ (r.v + 1) : ℕ) : ℤ) ∣ A i j := by
    by_contra hcon
    push Not at hcon
    have := (zero_iff_of_conj hdv1 hU hconj).mp (by rw [mapMatrix_zmat, zmat_eq_zero]; exact hcon)
    rw [mapMatrix_zmat, zmat_eq_zero] at this
    obtain ⟨i, j, hij⟩ := hr.wit
    exact hij (by have := this i j; push_cast at this; exact this)
  obtain ⟨i, j, hij⟩ := hAv1
  have hD : (r.ell : ℤ) ^ (r.v + r.e) ∣ E.lam E.θ - E.mu E.θ := by rw [hl, hm]; exact hr.gap
  have := E.dvd_of_read b r.ell hr.prime (r.v + r.e) r.v hD (b j) i
    (by have := hAv i j; push_cast at this; exact this) (by push_cast at hij; exact hij) t
  simpa using this

/-- The product `∏ ℓ^e` over a list of reads. -/
def gProd (rs : List K3Read) : ℕ := (rs.map fun r => r.ell ^ r.e).foldr (· * ·) 1

lemma coprime_gProd {ℓ : ℕ} (hℓ : ℓ.Prime) (rs : List K3Read) (hp : ∀ r ∈ rs, r.ell.Prime)
    (hn : ℓ ∉ rs.map K3Read.ell) : Nat.Coprime ℓ (gProd rs) := by
  induction rs with
  | nil => simp [gProd]
  | cons r rs ih =>
    simp only [List.map_cons, List.mem_cons, not_or] at hn
    have h1 : Nat.Coprime ℓ (r.ell ^ r.e) :=
      Nat.Coprime.pow_right _ ((Nat.coprime_primes hℓ (hp r (by simp))).mpr hn.1)
    have h2 := ih (fun r' hr' => hp r' (List.mem_cons_of_mem _ hr')) hn.2
    simpa [gProd] using Nat.Coprime.mul_right h1 h2

/-- Lower bounds at pairwise distinct primes multiply. -/
lemma gProd_dvd (rs : List K3Read) (hp : ∀ r ∈ rs, r.ell.Prime) (hnd : (rs.map K3Read.ell).Nodup) (x : ℤ)
    (hx : ∀ r ∈ rs, (r.ell : ℤ) ^ r.e ∣ x) : (gProd rs : ℤ) ∣ x := by
  induction rs with
  | nil => simp [gProd]
  | cons r rs ih =>
    simp only [List.map_cons, List.nodup_cons] at hnd
    have hc : IsCoprime ((r.ell ^ r.e : ℕ) : ℤ) (gProd rs : ℤ) :=
      Nat.isCoprime_iff_coprime.mpr (Nat.Coprime.pow_left _
        (coprime_gProd (hp r (by simp)) rs (fun r' hr' => hp r' (List.mem_cons_of_mem _ hr')) hnd.1))
    have h1 : ((r.ell ^ r.e : ℕ) : ℤ) ∣ x := by push_cast; exact hx r (by simp)
    have h2 := ih (fun r' hr' => hp r' (List.mem_cons_of_mem _ hr')) hnd.2 (fun r' hr' => hx r' (List.mem_cons_of_mem _ hr'))
    have := hc.mul_dvd h1 h2
    simpa [gProd] using this

lemma G_eq (c : K3NCert) : c.G = gProd c.reads := rfl

/-- **The K3N bridge** (01 Props. 5.3, 5.4 at level `N`, two systems): `G = ∏ ℓ^e ∣ λ t − μ t` for every `t ∈ 𝕋`, and
`λθ − μθ` lies in the congruence ideal. -/
theorem k3n_sound (c : K3NCert) (hc : K3NSpec c) (E : EisSplit T) (hl : E.lam E.θ = c.lam) (hm : E.mu E.θ = c.mu)
    (hmat : ∀ r ∈ c.reads, TBmat r (E.θ : Module.End ℤ L)) :
    (∀ t : T, (c.G : ℤ) ∣ E.lam t - E.mu t) ∧ c.lam - c.mu ∈ E.congIdeal := by
  refine ⟨fun t => ?_, ?_⟩
  · rw [G_eq]
    exact gProd_dvd c.reads (fun r hr => (hc.reads r hr).prime) hc.nodup _
      (fun r hr => read_lower E r c.lam c.mu (hc.reads r hr) hl hm (hmat r hr) t)
  · rw [← hl, ← hm]; exact E.mem_congIdeal_sub _

/-- **The fibre product** when the certificate is complete: with `g = λθ − μθ`, a pair `(a, b)` comes from `𝕋` iff
`a ≡ b mod g`; `(a, 0)` comes from `𝕋` iff `g ∣ a` (`C_𝕋 ≅ ℤ/g`). -/
theorem k3n_fibre (c : K3NCert) (hc : K3NSpec c) (hcomp : c.complete = true) (E : EisSplit T)
    (hl : E.lam E.θ = c.lam) (hm : E.mu E.θ = c.mu) (hmat : ∀ r ∈ c.reads, TBmat r (E.θ : Module.End ℤ L)) :
    (∀ t : T, (c.lam - c.mu) ∣ E.lam t - E.mu t) ∧
    (∀ a b : ℤ, (∃ t : T, E.lam t = a ∧ E.mu t = b) ↔ (c.lam - c.mu) ∣ a - b) ∧
    (∀ a : ℤ, (∃ t : T, E.lam t = a ∧ E.mu t = 0) ↔ (c.lam - c.mu) ∣ a) := by
  obtain ⟨hlow, hup⟩ := k3n_sound c hc E hl hm hmat
  have hG : (c.G : ℤ) = ((c.lam - c.mu).natAbs : ℤ) := by
    unfold K3NCert.complete K3NCert.gUp at hcomp; simp only [beq_iff_eq] at hcomp; rw [hcomp]
  have hlow' : ∀ t : T, (c.lam - c.mu) ∣ E.lam t - E.mu t := fun t => by
    have := hlow t; rw [hG] at this; exact Int.natAbs_dvd.mp (by simpa using this)
  have hfib := E.fibre_iff (c.lam - c.mu) hlow' hup
  exact ⟨hlow', hfib, fun a => by rw [hfib, sub_zero]⟩

/-- **Generation** when the certificate is complete: `𝕋 = ℤ + ℤθ`, i.e. `𝕋 = ℤ[θ]` ((C3) at every prime in the two-system
case; `L` torsion-free). -/
theorem k3n_gen [Module.IsTorsionFree ℤ L] (c : K3NCert) (hc : K3NSpec c) (hcomp : c.complete = true) (E : EisSplit T)
    (hl : E.lam E.θ = c.lam) (hm : E.mu E.θ = c.mu) (hmat : ∀ r ∈ c.reads, TBmat r (E.θ : Module.End ℤ L)) (t : T) :
    ∃ a b : ℤ, t = (a : T) + b • E.θ := by
  obtain ⟨hlow, -, -⟩ := k3n_fibre c hc hcomp E hl hm hmat
  obtain ⟨k, hk⟩ := hlow t
  refine ⟨E.mu t - k * c.mu, k, E.pair_injective ?_ ?_⟩
  · simp only [map_add, map_intCast, map_zsmul, smul_eq_mul, hl, Int.cast_id]; linear_combination hk
  · simp only [map_add, map_intCast, map_zsmul, smul_eq_mul, hm, Int.cast_id]; ring

end GL2.HeckeZ.K3N
