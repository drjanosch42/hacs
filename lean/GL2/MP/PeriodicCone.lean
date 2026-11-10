/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.MP.TreeRetract

/-!
# LR-Y phase 2b (slot 22): periodic resolutions of cyclic stabilisers, and the cone of a short exact sequence

[GT] `gt_part_MP.tex`, rows `def:mp-fb`, `thm:mp-per-acyclic`, `thm:mp-cone-ses`, backing [C] `cl:amalgam`,
`eq:mvamalgam` (the resolution of `eq:treeses`; plan of record `LV0_audit.md` §3.2, LR-2 (a), (c)).  Data-free
homological algebra over any commutative ring `k`.

**Free modules.**  `FB r G` is the basis `{(c, g)}` of the free `k[G]`-module of rank `r` (`(c, g)` stands for `g·e_c`);
`G` acts by left multiplication, freely.  Right multiplication by an element of `k[G]` is given on this basis.

**Periodic resolutions** (`PerData`, `perCx`).  Let `G` act on `X` with base points `x_c` (`c < r`) meeting every orbit
exactly once, `Stab(x_c) = {t_c^j : j < N_c}` (`t_c^{N_c} = 1`, the powers distinct).  Then
`⋯ → k[G]^r →(N) k[G]^r →(1−t) k[G]^r →(N) ⋯ →(1−t) k[G]^r → k[X] → 0`, with `(1−t)(c, g) = (c, g) − (c, g t_c)` and
`N(c, g) = ∑_{j < N_c} (c, g t_c^j)`, is acyclic (`perCx_acyclic`): a contraction on bases (`perContr`), from a choice of
orbit representatives `g = rep(c, g)·t_c^{idx(c, g)}` (the standard contraction of `k[ℤ/N]` on each right coset).  Applied
with `X = Γ/⟨t⟩` this is the periodic resolution of `ℤ/N_c` induced to `Γ`.

**The cone** (`ConeDat`, `coneCx`).  `RA → A`, `RB → B` acyclic augmented complexes (`A`, `B` in degree `0`), `f : RA → RB` a
chain map on bases, `ε : B → k` with `ε f₀ = 0`.  The complex `k ← RB₁ ← RB₂ ⊕ RA₁ ← RB₃ ⊕ RA₂ ← ⋯`,
`D(b, a) = (d b + f a, −d a)`, is acyclic as soon as `0 → A → B → k → 0` is exact (`coneCx_acyclic`; an element-wise
chase: the cone of `f` is acyclic in degrees `≥ 2`, the tree sequence handles degrees `0, 1`).
-/

namespace GL2.MP

open Finsupp

/-! ### Free modules over `k[G]` on bases -/

/-- **The basis of the free `k[G]`-module of rank `r`**: `(c, g)` is `g·e_c`. -/
@[ext]
structure FB (r : ℕ) (G : Type) where
  c : Fin r
  g : G

namespace FB

variable {r : ℕ} {G : Type} [Group G]

/-- `G` acts on `FB r G` by left multiplication. -/
instance : MulAction G (FB r G) where
  smul h b := ⟨b.c, h * b.g⟩
  one_smul b := by show FB.mk b.c (1 * b.g) = b; rw [one_mul]
  mul_smul h h' b := by show FB.mk b.c (h * h' * b.g) = FB.mk b.c (h * (h' * b.g)); rw [mul_assoc]

@[simp] lemma smul_c (h : G) (b : FB r G) : (h • b).c = b.c := rfl
@[simp] lemma smul_g (h : G) (b : FB r G) : (h • b).g = h * b.g := rfl

/-- **The action is free.** -/
lemma smul_eq_self {h : G} {b : FB r G} (e : h • b = b) : h = 1 := by
  have : h * b.g = b.g := congrArg FB.g e
  simpa using this

end FB

/-- `lc (lc ε ∘ φ) = lc ε ∘ lc φ`. -/
lemma lc_lc {k : Type} [CommRing k] {α β γ : Type} (ε : β → (γ →₀ k)) (φ : α → (β →₀ k)) (x : α →₀ k) :
    Finsupp.linearCombination k (fun a => Finsupp.linearCombination k ε (φ a)) x =
      Finsupp.linearCombination k ε (Finsupp.linearCombination k φ x) := by
  induction x using Finsupp.induction_linear with
  | zero => simp
  | add x y hx hy => rw [map_add, map_add, hx, hy, map_add]
  | single a c => simp [map_smul]

/-- `lc (φ − ψ) = lc φ − lc ψ`. -/
lemma lc_sub_fun {k : Type} [CommRing k] {α β : Type} (φ ψ : α → (β →₀ k)) (x : α →₀ k) :
    Finsupp.linearCombination k (fun a => φ a - ψ a) x =
      Finsupp.linearCombination k φ x - Finsupp.linearCombination k ψ x := by
  induction x using Finsupp.induction_linear with
  | zero => simp
  | add x y hx hy => rw [map_add, map_add, map_add, hx, hy]; abel
  | single a c => simp [smul_sub]

/-- `lc (mapDomain g ∘ φ) = mapDomain g ∘ lc φ`. -/
lemma lc_mapDomain_comp {k : Type} [CommRing k] {α β γ : Type} (g : β → γ) (φ : α → (β →₀ k)) (x : α →₀ k) :
    Finsupp.linearCombination k (fun a => mapDomain g (φ a)) x = mapDomain g (Finsupp.linearCombination k φ x) := by
  induction x using Finsupp.induction_linear with
  | zero => simp
  | add x y hx hy => rw [map_add, map_add, hx, hy, mapDomain_add]
  | single a c => simp [mapDomain_smul]

/-! ### Periodic resolutions of cyclic stabilisers -/

/-- **Periodic data**: base points `x_c` meeting every orbit once, with cyclic stabilisers `{t_c^j : j < N_c}`. -/
structure PerData (G : Type) [Group G] (X : Type) [MulAction G X] (r : ℕ) where
  x : Fin r → X
  t : Fin r → G
  N : Fin r → ℕ
  N_pos : ∀ c, 0 < N c
  pow_N : ∀ c, t c ^ N c = 1
  pow_inj : ∀ c (a b : Fin (N c)), t c ^ (a : ℕ) = t c ^ (b : ℕ) → a = b
  fix : ∀ c, t c • x c = x c
  cov : ∀ y : X, ∃ c : Fin r, ∃ g : G, g • x c = y
  sep : ∀ (c c' : Fin r) (g g' : G), g' • x c' = g • x c → c' = c ∧ ∃ j : Fin (N c), g = g' * t c ^ (j : ℕ)

namespace PerData

variable {G : Type} [Group G] {X : Type} [MulAction G X] {r : ℕ} (D : PerData G X r)

lemma pow_fix (c : Fin r) (n : ℕ) : D.t c ^ n • D.x c = D.x c := by
  induction n with
  | zero => rw [pow_zero, one_smul]
  | succ n ih => rw [pow_succ, mul_smul, D.fix, ih]

lemma pow_mod (c : Fin r) (n : ℕ) : D.t c ^ (n % D.N c) = D.t c ^ n := by
  conv_rhs => rw [← Nat.div_add_mod n (D.N c), pow_add, pow_mul, D.pow_N, one_pow, one_mul]

variable (k : Type) [CommRing k]

/-- `(1 − t)` on bases: `(c, g) ↦ (c, g) − (c, g t_c)`. -/
noncomputable def difB (b : FB r G) : FB r G →₀ k := single b 1 - single ⟨b.c, b.g * D.t b.c⟩ 1

/-- `N` on bases: `(c, g) ↦ ∑_{j < N_c} (c, g t_c^j)`. -/
noncomputable def nrmB (b : FB r G) : FB r G →₀ k := ∑ i ∈ Finset.range (D.N b.c), single ⟨b.c, b.g * D.t b.c ^ i⟩ 1

/-- The augmentation `(c, g) ↦ g x_c`. -/
noncomputable def augB (b : FB r G) : X →₀ k := single (b.g • D.x b.c) 1

/-- The boundary in degree `n + 1` (on `FB`): `1 − t` for even `n`, `N` for odd `n`. -/
noncomputable def bdS (n : ℕ) (b : FB r G) : FB r G →₀ k := if Even n then D.difB k b else D.nrmB k b

lemma bdS_even {n : ℕ} (hn : Even n) : D.bdS k n = D.difB k := funext fun _ => if_pos hn

lemma bdS_odd {n : ℕ} (hn : ¬ Even n) : D.bdS k n = D.nrmB k := funext fun _ => if_neg hn

variable (G X r) in
/-- The basis of the augmented periodic complex: `X` in degree `0`, `FB r G` above. -/
def perB : ℕ → Type
  | 0 => X
  | _ + 1 => FB r G

/-- The boundary: `(c, g) ↦ g x_c`, then `1 − t`, `N`, `1 − t`, … -/
noncomputable def perBd : ∀ n, perB G X r (n + 1) → (perB G X r n →₀ k)
  | 0 => D.augB k
  | n + 1 => D.bdS k n

lemma single_pow (c : Fin r) (g : G) (i : ℕ) :
    (single (⟨c, g * D.t c ^ i * D.t c⟩ : FB r G) (1 : k)) = single ⟨c, g * D.t c ^ (i + 1)⟩ 1 := by
  rw [pow_succ, mul_assoc]

/-- `(1 − t) N = 0`. -/
lemma lc_difB_nrmB (b : FB r G) : Finsupp.linearCombination k (D.difB k) (D.nrmB k b) = 0 := by
  rw [nrmB, map_sum]
  simp only [linearCombination_single, one_smul, difB, single_pow]
  rw [Finset.sum_range_sub' (fun i => (single (⟨b.c, b.g * D.t b.c ^ i⟩ : FB r G) (1 : k))), pow_zero, D.pow_N,
    sub_self]

/-- `N (1 − t) = 0`. -/
lemma lc_nrmB_difB (b : FB r G) : Finsupp.linearCombination k (D.nrmB k) (D.difB k b) = 0 := by
  rw [difB, map_sub, linearCombination_single, linearCombination_single, one_smul, one_smul, nrmB, nrmB]
  simp only [mul_assoc, ← pow_succ']
  set f : ℕ → FB r G →₀ k := fun i => single ⟨b.c, b.g * D.t b.c ^ i⟩ 1 with hf
  change ∑ i ∈ Finset.range (D.N b.c), f i - ∑ i ∈ Finset.range (D.N b.c), f (i + 1) = 0
  have h := Finset.sum_range_succ' f (D.N b.c)
  rw [Finset.sum_range_succ] at h
  have hN : f (D.N b.c) = f 0 := by simp only [hf, D.pow_N, pow_zero]
  rw [hN] at h
  rw [sub_eq_zero]
  exact add_right_cancel h

lemma even0 : Even 0 := ⟨0, rfl⟩

lemma bdd_zero (b : FB r G) : Finsupp.linearCombination k (D.augB k) (D.bdS k 0 b) = 0 := by
  rw [D.bdS_even k even0, difB, map_sub, linearCombination_single, linearCombination_single, one_smul, one_smul,
    augB, augB]
  show single (b.g • D.x b.c) 1 - single ((b.g * D.t b.c) • D.x b.c) 1 = 0
  rw [mul_smul, D.fix, sub_self]

lemma bdd_succ (n : ℕ) (b : FB r G) : Finsupp.linearCombination k (D.bdS k n) (D.bdS k (n + 1) b) = 0 := by
  by_cases hn : Even n
  · rw [D.bdS_even k hn, D.bdS_odd k (fun h => Nat.even_add_one.1 h hn)]
    exact D.lc_difB_nrmB k b
  · rw [D.bdS_odd k hn, D.bdS_even k (Nat.even_add_one.2 hn)]
    exact D.lc_nrmB_difB k b

/-- **The augmented periodic complex** `⋯ → k[G]^r →(1−t) k[G]^r → k[X]`. -/
noncomputable def perCx : BCx k where
  B := perB G X r
  bd := D.perBd k
  bdd n b := by
    rcases n with _ | n
    · exact D.bdd_zero k b
    · exact D.bdd_succ k n b

/-! #### The contraction -/

/-- A chosen preimage `(c, g)` of `y` (`g x_c = y`). -/
noncomputable def sec (y : X) : FB r G := ⟨(D.cov y).choose, (D.cov y).choose_spec.choose⟩

lemma sec_spec (y : X) : (D.sec y).g • D.x (D.sec y).c = y := (D.cov y).choose_spec.choose_spec

lemma sec_c (b : FB r G) : (D.sec (b.g • D.x b.c)).c = b.c := (D.sep _ _ _ _ (D.sec_spec _)).1

/-- The representative `rep(c, g)` of the coset `g⟨t_c⟩`. -/
noncomputable def rep (b : FB r G) : G := (D.sec (b.g • D.x b.c)).g

lemma exists_idx (b : FB r G) : ∃ j : Fin (D.N b.c), b.g = D.rep b * D.t b.c ^ (j : ℕ) :=
  (D.sep _ _ _ _ (D.sec_spec _)).2

/-- The index `idx(c, g) < N_c` with `g = rep(c, g) t_c^{idx}`. -/
noncomputable def idx (b : FB r G) : Fin (D.N b.c) := (D.exists_idx b).choose

lemma idx_spec (b : FB r G) : b.g = D.rep b * D.t b.c ^ (D.idx b : ℕ) := (D.exists_idx b).choose_spec

lemma idx_eq (b : FB r G) (j : Fin (D.N b.c)) (h : b.g = D.rep b * D.t b.c ^ (j : ℕ)) : D.idx b = j :=
  D.pow_inj _ _ _ (mul_left_cancel ((D.idx_spec b).symm.trans h))

lemma sec_eq (b : FB r G) : D.sec (b.g • D.x b.c) = ⟨b.c, D.rep b⟩ := FB.ext (D.sec_c b) rfl

lemma rep_mul_pow (b : FB r G) (i : ℕ) : D.rep ⟨b.c, b.g * D.t b.c ^ i⟩ = D.rep b := by
  unfold rep
  congr 2
  show (b.g * D.t b.c ^ i) • D.x b.c = b.g • D.x b.c
  rw [mul_smul, D.pow_fix]

lemma idx_mul_pow (b : FB r G) (i : ℕ) :
    (D.idx ⟨b.c, b.g * D.t b.c ^ i⟩ : ℕ) = ((D.idx b : ℕ) + i) % D.N b.c := by
  have := D.idx_eq ⟨b.c, b.g * D.t b.c ^ i⟩ ⟨((D.idx b : ℕ) + i) % D.N b.c, Nat.mod_lt _ (D.N_pos _)⟩ (by
    show b.g * D.t b.c ^ i = D.rep ⟨b.c, b.g * D.t b.c ^ i⟩ * D.t b.c ^ (((D.idx b : ℕ) + i) % D.N b.c)
    rw [D.rep_mul_pow, D.pow_mod, pow_add, ← mul_assoc, ← D.idx_spec])
  rw [this]

lemma idx_mul (b : FB r G) : (D.idx ⟨b.c, b.g * D.t b.c⟩ : ℕ) = ((D.idx b : ℕ) + 1) % D.N b.c := by
  have := D.idx_mul_pow b 1
  rwa [pow_one] at this

lemma rep_mul (b : FB r G) : D.rep ⟨b.c, b.g * D.t b.c⟩ = D.rep b := by
  have := D.rep_mul_pow b 1
  rwa [pow_one] at this

/-- `b = (c, rep t^{idx})`. -/
lemma eq_rep_idx (b : FB r G) : b = ⟨b.c, D.rep b * D.t b.c ^ (D.idx b : ℕ)⟩ := FB.ext rfl (D.idx_spec b)

/-- The contraction in even degrees: `(c, g) ↦ −∑_{i < idx} (c, rep t^i)`. -/
noncomputable def sEv (b : FB r G) : FB r G →₀ k :=
  -∑ i ∈ Finset.range (D.idx b), single ⟨b.c, D.rep b * D.t b.c ^ i⟩ 1

/-- The contraction in odd degrees: `(c, g) ↦ (c, rep)` if `idx = N − 1`, else `0`. -/
noncomputable def sOd (b : FB r G) : FB r G →₀ k :=
  if (D.idx b : ℕ) + 1 = D.N b.c then single ⟨b.c, D.rep b⟩ 1 else 0

/-- `(1 − t) sEv = id − (c, rep)`. -/
lemma lc_difB_sEv (b : FB r G) :
    Finsupp.linearCombination k (D.difB k) (D.sEv k b) = single b 1 - single ⟨b.c, D.rep b⟩ 1 := by
  rw [sEv, map_neg, map_sum]
  simp only [linearCombination_single, one_smul, difB, single_pow]
  rw [Finset.sum_range_sub' (fun i => (single (⟨b.c, D.rep b * D.t b.c ^ i⟩ : FB r G) (1 : k))), pow_zero, mul_one,
    neg_sub, ← D.eq_rep_idx]

/-- `sEv (1 − t)`. -/
lemma lc_sEv_difB (b : FB r G) :
    Finsupp.linearCombination k (D.sEv k) (D.difB k b) =
      if (D.idx b : ℕ) + 1 = D.N b.c then -∑ i ∈ Finset.range (D.idx b), single ⟨b.c, D.rep b * D.t b.c ^ i⟩ 1
      else single b 1 := by
  rw [difB, map_sub, linearCombination_single, linearCombination_single, one_smul, one_smul]
  unfold sEv
  rw [D.rep_mul]
  have hi := D.idx_mul b
  dsimp only at hi ⊢
  split_ifs with h
  · rw [hi, h, Nat.mod_self, Finset.range_zero, Finset.sum_empty, neg_zero, sub_zero]
  · have hlt : (D.idx b : ℕ) + 1 < D.N b.c := lt_of_le_of_ne (D.idx b).2 h
    rw [hi, Nat.mod_eq_of_lt hlt, Finset.sum_range_succ, neg_add, sub_add_cancel_left, neg_neg, ← D.eq_rep_idx]

/-- `sOd N = (c, rep)`. -/
lemma lc_sOd_nrmB (b : FB r G) :
    Finsupp.linearCombination k (D.sOd k) (D.nrmB k b) = single ⟨b.c, D.rep b⟩ 1 := by
  rw [nrmB, map_sum]
  simp only [linearCombination_single, one_smul]
  unfold sOd
  simp only [D.rep_mul_pow, D.idx_mul_pow]
  have : NeZero (D.N b.c) := ⟨(D.N_pos b.c).ne'⟩
  rw [← Fin.sum_univ_eq_sum_range (fun i => if ((D.idx b : ℕ) + i) % D.N b.c + 1 = D.N b.c then
    (single (⟨b.c, D.rep b⟩ : FB r G) (1 : k)) else 0)]
  have e : ∀ i : Fin (D.N b.c), ((D.idx b : ℕ) + i) % D.N b.c = ((D.idx b + i : Fin (D.N b.c)) : ℕ) := fun i =>
    (Fin.val_add _ _).symm
  simp only [e]
  have hsum := Equiv.sum_comp (Equiv.addLeft (D.idx b)) (fun j : Fin (D.N b.c) => if (j : ℕ) + 1 = D.N b.c then
    (single (⟨b.c, D.rep b⟩ : FB r G) (1 : k)) else 0)
  simp only [Equiv.coe_addLeft] at hsum
  refine Eq.trans hsum ?_
  rw [Finset.sum_eq_single ⟨D.N b.c - 1, by have := D.N_pos b.c; omega⟩]
  · rw [if_pos (by have := D.N_pos b.c; dsimp only; omega)]
  · intro j _ hj
    rw [if_neg]
    intro h'
    exact hj (Fin.ext (by dsimp only; omega))
  · intro h; exact absurd (Finset.mem_univ _) h

/-- `N sOd`. -/
lemma lc_nrmB_sOd (b : FB r G) :
    Finsupp.linearCombination k (D.nrmB k) (D.sOd k b) =
      if (D.idx b : ℕ) + 1 = D.N b.c then ∑ i ∈ Finset.range (D.N b.c), single ⟨b.c, D.rep b * D.t b.c ^ i⟩ 1
      else 0 := by
  unfold sOd
  split_ifs
  · rw [linearCombination_single, one_smul]; rfl
  · rw [map_zero]

/-- The contraction in degree `n + 1` (on `FB`). -/
noncomputable def hS (n : ℕ) (b : FB r G) : FB r G →₀ k := if Even n then D.sEv k b else D.sOd k b

/-- The contraction in degree `0`: `y ↦ sec y`. -/
noncomputable def h0B (y : X) : FB r G →₀ k := single (D.sec y) 1

/-- The contraction on bases. -/
noncomputable def perH : ∀ n, perB G X r n → (perB G X r (n + 1) →₀ k)
  | 0 => D.h0B k
  | n + 1 => D.hS k n

lemma h0_eq (y : X) : Finsupp.linearCombination k (D.augB k) (D.h0B k y) = single y 1 := by
  rw [h0B, linearCombination_single, one_smul, augB, D.sec_spec]

lemma hs_zero (b : FB r G) :
    Finsupp.linearCombination k (D.bdS k 0) (D.hS k 0 b) + Finsupp.linearCombination k (D.h0B k) (D.augB k b) =
      single b 1 := by
  rw [D.bdS_even k even0, hS, if_pos even0, D.lc_difB_sEv, augB, linearCombination_single, one_smul, h0B, D.sec_eq,
    sub_add_cancel]

lemma hs_succ (n : ℕ) (b : FB r G) :
    Finsupp.linearCombination k (D.bdS k (n + 1)) (D.hS k (n + 1) b) +
      Finsupp.linearCombination k (D.hS k n) (D.bdS k n b) = single b 1 := by
  by_cases hn : Even n
  · have hn' : ¬ Even (n + 1) := fun h => Nat.even_add_one.1 h hn
    rw [hS, if_neg hn', bdS, if_pos hn, D.bdS_odd k hn', show D.hS k n = D.sEv k from funext fun _ => if_pos hn,
      D.lc_nrmB_sOd, D.lc_sEv_difB]
    split_ifs with h
    · have e := Finset.sum_range_succ (fun i => (single (⟨b.c, D.rep b * D.t b.c ^ i⟩ : FB r G) (1 : k))) (D.idx b)
      rw [h] at e
      rw [e, ← D.eq_rep_idx]
      abel
    · rw [zero_add]
  · have hn' : Even (n + 1) := Nat.even_add_one.2 hn
    rw [hS, if_pos hn', bdS, if_neg hn, D.bdS_even k hn', show D.hS k n = D.sOd k from funext fun _ => if_neg hn,
      D.lc_difB_sEv, D.lc_sOd_nrmB, sub_add_cancel]

/-- **The contraction of the periodic complex.** -/
noncomputable def perContr : BContr k (D.perCx k) where
  hb := D.perH k
  h0 y := D.h0_eq k y
  hs m b := by
    rcases m with _ | n
    · exact D.hs_zero k b
    · exact D.hs_succ k n b

/-- **The periodic complex is acyclic** (`→ k[X]` is a resolution). -/
theorem perCx_acyclic : (D.perCx k).toCx.Acyclic := (D.perContr k).acyclic

/-- **The periodic boundary is equivariant** (degrees `≥ 1`). -/
lemma bdS_smul (n : ℕ) (h : G) (b : FB r G) :
    D.bdS k n (h • b) = mapDomain (fun y : FB r G => h • y) (D.bdS k n b) := by
  unfold bdS
  split_ifs
  · simp only [difB, mapDomain_sub, mapDomain_single, FB.smul_c, FB.smul_g, mul_assoc]; rfl
  · simp only [nrmB, mapDomain_finsetSum, mapDomain_single, FB.smul_c, FB.smul_g, mul_assoc]; rfl

/-- The augmentation is equivariant. -/
lemma augB_smul (h : G) (b : FB r G) : D.augB k (h • b) = mapDomain (fun y : X => h • y) (D.augB k b) := by
  rw [augB, augB, mapDomain_single, FB.smul_g, FB.smul_c, mul_smul]

end PerData

/-! ### The cone of a short exact sequence of resolutions -/

/-- **Cone data**: augmented complexes `RA → A`, `RB → B` (in degree `0`), a chain map `f` on bases, `ε : B → k` with
`ε f₀ = 0`. -/
structure ConeDat (k : Type) [CommRing k] where
  RA : BCx k
  RB : BCx k
  ε : RB.B 0 → (Unit →₀ k)
  f : ∀ n, RA.B n → (RB.B n →₀ k)
  f_comm : ∀ n (a : RA.B (n + 1)), RB.d n (f (n + 1) a) = Finsupp.linearCombination k (f n) (RA.bd n a)
  εf : ∀ a, Finsupp.linearCombination k ε (f 0 a) = 0

namespace ConeDat

variable {k : Type} [CommRing k] (D : ConeDat k)

/-- The basis of the cone: `Unit`, `RB₁`, then `RB_{n+2} ⊕ RA_{n+1}`. -/
def B : ℕ → Type
  | 0 => Unit
  | 1 => D.RB.B 1
  | n + 2 => D.RB.B (n + 2) ⊕ D.RA.B (n + 1)

/-- The cone boundary in degree `1`: `ε d`. -/
noncomputable def bd0 (b : D.RB.B 1) : Unit →₀ k := Finsupp.linearCombination k D.ε (D.RB.bd 0 b)

/-- The cone boundary in degree `2`: `(b, a) ↦ d b + f a`. -/
noncomputable def bd1 : D.RB.B 2 ⊕ D.RA.B 1 → (D.RB.B 1 →₀ k) := Sum.elim (D.RB.bd 1) (D.f 1)

/-- The cone boundary in degree `n + 3`: `(b, a) ↦ (d b + f a, −d a)`. -/
noncomputable def bd2 (n : ℕ) : D.RB.B (n + 3) ⊕ D.RA.B (n + 2) → (D.RB.B (n + 2) ⊕ D.RA.B (n + 1) →₀ k) :=
  Sum.elim (fun b => mapDomain Sum.inl (D.RB.bd (n + 2) b))
    (fun a => mapDomain Sum.inl (D.f (n + 2) a) - mapDomain Sum.inr (D.RA.bd (n + 1) a))

/-- The boundary of the cone. -/
noncomputable def bd : ∀ n, D.B (n + 1) → (D.B n →₀ k)
  | 0 => D.bd0
  | 1 => D.bd1
  | n + 2 => D.bd2 n

/-- The chain map `f` on modules. -/
noncomputable def F (n : ℕ) : (D.RA.B n →₀ k) →ₗ[k] (D.RB.B n →₀ k) := Finsupp.linearCombination k (D.f n)

lemma F_comm (n : ℕ) (x : D.RA.B (n + 1) →₀ k) : D.RB.d n (D.F (n + 1) x) = D.F n (D.RA.d n x) :=
  BCx.lc_comm (A := D.RB) (B := D.RA) D.f D.f_comm n x

lemma lc_sumElim {α β γ : Type} (φ : α → (γ →₀ k)) (ψ : β → (γ →₀ k)) (x : α →₀ k) (y : β →₀ k) :
    Finsupp.linearCombination k (Sum.elim φ ψ) (mapDomain Sum.inl x + mapDomain Sum.inr y) =
      Finsupp.linearCombination k φ x + Finsupp.linearCombination k ψ y := by
  rw [map_add, linearCombination_mapDomain, linearCombination_mapDomain]; rfl

lemma bdd0 (b : D.RB.B 2 ⊕ D.RA.B 1) : Finsupp.linearCombination k D.bd0 (D.bd1 b) = 0 := by
  rcases b with b | a
  · show Finsupp.linearCombination k (fun b => Finsupp.linearCombination k D.ε (D.RB.bd 0 b)) (D.RB.bd 1 b) = 0
    rw [lc_lc]; exact (congrArg _ (D.RB.bdd 0 b)).trans (map_zero _)
  · show Finsupp.linearCombination k (fun b => Finsupp.linearCombination k D.ε (D.RB.bd 0 b)) (D.f 1 a) = 0
    rw [lc_lc]
    show Finsupp.linearCombination k D.ε (D.RB.d 0 (D.f 1 a)) = 0
    rw [D.f_comm, ← lc_lc]
    simp only [D.εf]
    rw [Finsupp.linearCombination_apply]; simp

lemma bdd1 (b : D.RB.B 3 ⊕ D.RA.B 2) : Finsupp.linearCombination k D.bd1 (D.bd2 0 b) = 0 := by
  rcases b with b | a
  · show Finsupp.linearCombination k (Sum.elim (D.RB.bd 1) (D.f 1)) (mapDomain Sum.inl (D.RB.bd 2 b)) = 0
    rw [linearCombination_mapDomain]; exact D.RB.bdd 1 b
  · show Finsupp.linearCombination k (Sum.elim (D.RB.bd 1) (D.f 1))
      (mapDomain Sum.inl (D.f 2 a) - mapDomain Sum.inr (D.RA.bd 1 a)) = 0
    rw [map_sub, linearCombination_mapDomain, linearCombination_mapDomain]
    exact sub_eq_zero.2 (D.f_comm 1 a)

lemma bdd2 (n : ℕ) (b : D.RB.B (n + 1 + 3) ⊕ D.RA.B (n + 1 + 2)) :
    Finsupp.linearCombination k (D.bd2 n) (D.bd2 (n + 1) b) = 0 := by
  rcases b with b | a
  · show Finsupp.linearCombination k (D.bd2 n) (mapDomain Sum.inl (D.RB.bd (n + 3) b)) = 0
    rw [linearCombination_mapDomain]
    show Finsupp.linearCombination k (fun b => mapDomain Sum.inl (D.RB.bd (n + 2) b)) (D.RB.bd (n + 3) b) = 0
    rw [lc_mapDomain_comp]
    exact (congrArg _ (D.RB.bdd (n + 2) b)).trans mapDomain_zero
  · show Finsupp.linearCombination k (D.bd2 n)
      (mapDomain Sum.inl (D.f (n + 3) a) - mapDomain Sum.inr (D.RA.bd (n + 2) a)) = 0
    rw [map_sub, linearCombination_mapDomain, linearCombination_mapDomain]
    show Finsupp.linearCombination k (fun b => mapDomain Sum.inl (D.RB.bd (n + 2) b)) (D.f (n + 3) a) -
      Finsupp.linearCombination k (fun a => mapDomain Sum.inl (D.f (n + 2) a) - mapDomain Sum.inr (D.RA.bd (n + 1) a))
        (D.RA.bd (n + 2) a) = 0
    rw [lc_sub_fun, lc_mapDomain_comp, lc_mapDomain_comp, lc_mapDomain_comp,
      show Finsupp.linearCombination k (D.RA.bd (n + 1)) (D.RA.bd (n + 2) a) = 0 from D.RA.bdd (n + 1) a,
      mapDomain_zero, sub_zero, sub_eq_zero]
    exact congrArg (mapDomain Sum.inl) (D.f_comm (n + 2) a)

lemma bdd : ∀ n (b : D.B (n + 2)), Finsupp.linearCombination k (D.bd n) (D.bd (n + 1) b) = 0
  | 0, b => D.bdd0 b
  | 1, b => D.bdd1 b
  | n + 2, b => D.bdd2 n b

/-- **The cone complex** `k ← RB₁ ← RB₂ ⊕ RA₁ ← RB₃ ⊕ RA₂ ← ⋯`. -/
noncomputable def cx : BCx k where
  B := D.B
  bd := D.bd
  bdd := D.bdd

/-- `x ⊕ y` in `k[α ⊕ β]`. -/
noncomputable def L {α β : Type} (x : α →₀ k) (y : β →₀ k) : α ⊕ β →₀ k := mapDomain Sum.inl x + mapDomain Sum.inr y

omit D in
lemma L_inl {α β : Type} (x : α →₀ k) (y : β →₀ k) (a : α) : L x y (Sum.inl a) = x a := by
  rw [L, Finsupp.add_apply, mapDomain_apply Sum.inl_injective, mapDomain_of_notMem_range _ _ (by simp), add_zero]

omit D in
lemma L_inr {α β : Type} (x : α →₀ k) (y : β →₀ k) (b : β) : L x y (Sum.inr b) = y b := by
  rw [L, Finsupp.add_apply, mapDomain_apply Sum.inr_injective, mapDomain_of_notMem_range _ _ (by simp), zero_add]

omit D in
/-- Every element of `k[α ⊕ β]` is `x ⊕ y`. -/
lemma eq_L {α β : Type} (q : α ⊕ β →₀ k) :
    ∃ x y, q = L x y := by
  refine ⟨(Finsupp.sumFinsuppLEquivProdFinsupp k q).1, (Finsupp.sumFinsuppLEquivProdFinsupp k q).2, ?_⟩
  ext (a | b)
  · rw [L_inl]; rfl
  · rw [L_inr]; rfl

omit D in
lemma L_eq_zero {α β : Type} {x : α →₀ k} {y : β →₀ k} (h : L x y = 0) : x = 0 ∧ y = 0 :=
  ⟨Finsupp.ext fun a => by rw [← L_inl x y, h]; rfl, Finsupp.ext fun b => by rw [← L_inr x y, h]; rfl⟩

omit D in
lemma mapDomain_neg' {α β : Type} (g : α → β) (v : α →₀ k) : mapDomain g (-v) = -mapDomain g v := by
  rw [← zero_sub, mapDomain_sub, mapDomain_zero, zero_sub]

lemma d_zero (y : D.RB.B 1 →₀ k) :
    Finsupp.linearCombination k D.bd0 y = Finsupp.linearCombination k D.ε (D.RB.d 0 y) := lc_lc D.ε (D.RB.bd 0) y

lemma d_one (x : D.RB.B 2 →₀ k) (y : D.RA.B 1 →₀ k) :
    Finsupp.linearCombination k D.bd1 (L x y) = D.RB.d 1 x + D.F 1 y := lc_sumElim _ _ x y

lemma d_succ (n : ℕ) (x : D.RB.B (n + 3) →₀ k) (y : D.RA.B (n + 2) →₀ k) :
    Finsupp.linearCombination k (D.bd2 n) (L x y) = L (D.RB.d (n + 2) x + D.F (n + 2) y) (-D.RA.d (n + 1) y) := by
  rw [bd2, L, lc_sumElim, lc_mapDomain_comp, lc_sub_fun, lc_mapDomain_comp, lc_mapDomain_comp, L, mapDomain_add,
    mapDomain_neg', sub_eq_add_neg, add_assoc]
  rfl

/-- The common step of the chase: `d x + f y = 0` and `d y = 0` make `(x, y)` a cone boundary. -/
lemma step (hA : D.RA.toCx.Acyclic) (hB : D.RB.toCx.Acyclic) (n : ℕ) (x : D.RB.B (n + 2) →₀ k)
    (y : D.RA.B (n + 1) →₀ k) (h1 : D.RB.d (n + 1) x + D.F (n + 1) y = 0) (h2 : D.RA.d n y = 0) :
    ∃ (x' : D.RB.B (n + 3) →₀ k) (y' : D.RA.B (n + 2) →₀ k),
      D.RB.d (n + 2) x' + D.F (n + 2) y' = x ∧ -D.RA.d (n + 1) y' = y := by
  obtain ⟨(y' : D.RA.B (n + 2) →₀ k), hy'⟩ := hA.2 n y h2
  have hy'' : D.RA.d (n + 1) y' = y := hy'
  have hc : D.RB.d (n + 1) (x + D.F (n + 2) y') = 0 := by
    rw [map_add, D.F_comm, hy'', h1]
  obtain ⟨(x' : D.RB.B (n + 3) →₀ k), hx'⟩ := hB.2 (n + 1) _ hc
  have hx'' : D.RB.d (n + 2) x' = x + D.F (n + 2) y' := hx'
  exact ⟨x', -y', by rw [map_neg, hx'']; abel, by rw [map_neg, neg_neg, hy'']⟩

/-- **The cone of a short exact sequence of resolutions is a resolution**: if `RA → A`, `RB → B` are acyclic and
`0 → A →(f₀) B →(ε) k → 0` is exact, the cone complex `k ← RB₁ ← RB₂ ⊕ RA₁ ← ⋯` is acyclic. -/
theorem acyclic (hA : D.RA.toCx.Acyclic) (hB : D.RB.toCx.Acyclic)
    (hε : ∀ c : Unit →₀ k, ∃ y, Finsupp.linearCombination k D.ε y = c)
    (hex : ∀ y : D.RB.B 0 →₀ k, Finsupp.linearCombination k D.ε y = 0 → ∃ z, D.F 0 z = y)
    (hinj : ∀ z : D.RA.B 0 →₀ k, D.F 0 z = 0 → z = 0) : D.cx.toCx.Acyclic := by
  refine ⟨fun c => ?_, fun n q hq => ?_⟩
  · obtain ⟨y, hy⟩ := hε c
    obtain ⟨(y' : D.RB.B 1 →₀ k), hy'⟩ := hB.1 y
    have hy'' : D.RB.d 0 y' = y := hy'
    exact ⟨y', show Finsupp.linearCombination k D.bd0 y' = c by rw [d_zero, hy'', hy]⟩
  rcases n with _ | _ | n
  · -- degree 1
    change D.RB.B 1 →₀ k at q
    have hq' : Finsupp.linearCombination k D.ε (D.RB.d 0 q) = 0 := by rw [← d_zero]; exact hq
    obtain ⟨z, hz⟩ := hex _ hq'
    obtain ⟨(z' : D.RA.B 1 →₀ k), hz'⟩ := hA.1 z
    have hz'' : D.RA.d 0 z' = z := hz'
    have hc : D.RB.d 0 (q - D.F 1 z') = 0 := by rw [map_sub, D.F_comm, hz'', hz, sub_self]
    obtain ⟨(w : D.RB.B 2 →₀ k), hw⟩ := hB.2 0 _ hc
    have hw' : D.RB.d 1 w = q - D.F 1 z' := hw
    exact ⟨L w z', show Finsupp.linearCombination k D.bd1 (L w z') = q by rw [d_one, hw', sub_add_cancel]⟩
  · -- degree 2
    change D.RB.B 2 ⊕ D.RA.B 1 →₀ k at q
    obtain ⟨x, y, rfl⟩ := eq_L q
    have hq' : D.RB.d 1 x + D.F 1 y = 0 := by rw [← d_one]; exact hq
    have h2 : D.RA.d 0 y = 0 := hinj _ (by
      rw [← D.F_comm, eq_neg_of_add_eq_zero_right hq', map_neg]
      exact neg_eq_zero.2 (D.RB.toCx.dd 0 x))
    obtain ⟨x', y', hx, hy⟩ := D.step hA hB 0 x y hq' h2
    exact ⟨L x' y', show Finsupp.linearCombination k (D.bd2 0) (L x' y') = L x y by rw [d_succ, hx, hy]⟩
  · -- degree `n + 3`
    change D.RB.B (n + 3) ⊕ D.RA.B (n + 2) →₀ k at q
    obtain ⟨x, y, rfl⟩ := eq_L q
    have hq' : L (D.RB.d (n + 2) x + D.F (n + 2) y) (-D.RA.d (n + 1) y) = 0 := by rw [← d_succ]; exact hq
    obtain ⟨h1, h2⟩ := L_eq_zero hq'
    obtain ⟨x', y', hx, hy⟩ := D.step hA hB (n + 1) x y h1 (neg_eq_zero.1 h2)
    exact ⟨L x' y', show Finsupp.linearCombination k (D.bd2 (n + 1)) (L x' y') = L x y by rw [d_succ, hx, hy]⟩

end ConeDat

end GL2.MP
