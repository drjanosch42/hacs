/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.MP.ACModels

/-!
# LR-Y phase 1 (slot 22): the Bass–Serre tree of `SL₂(ℤ)` as a retract of `Y`, and [C] `eq:treeses` (1)

[GT] `gt_part_MP.tex`, rows `lem:mp-cx-retract`, `def:mp-tree-cx`, `thm:mp-tree-acyclic`, `def:mp-datum-tree`,
`thm:mp-model7-tree`, `thm:mp-treeses-exact`, backing [C] `eq:treeses` (1)
(plan of record: `docs/implementation/slot22/LV0_audit.md` §3.2, LR-1 in the form LR-Y).

**The tree.**  For a cover with (IO) (`Cover.Hyp`) and a threshold `m`, `treeCx H m k` is the augmented ordered chain
complex of the **full subcomplex of `Y` on the vertices of order `≥ m`** (`TS H m s`: the ordered `s`-simplices of `Y` whose
first, hence every, vertex has `ord ≥ m`); it is a subcomplex of `yCx H k` (`tIncl`, injective on bases).  For `M₇` and
`m = 3` the vertices are the orbits `3 = Γ·ρ` (`Γ_ρ = ⟨U⟩`, order 6) and `4 = Γ·i` (`Γ_i = ⟨S⟩`, order 4), there are no
`2`-simplices, and every edge is a `Γ`-translate of `[ρ, i] = [pt 4, pt 6]` (`model7_tree_edges`): this is the tree
`𝒯 = Γ·[ρ, i]` of [C] `eq:treeses`.

**The retraction** (`TreeRetr`, `tRetr`).  A vertex map `f : P → P` that is the identity on the tree, lands in the tree and
carries every edge of `Y` to a vertex or an edge of the tree induces a chain map `r : k[Y_•] → k[𝒯_•]` (the induced map on
ordered chains: vertices `v ↦ f v`, edges `[a, b] ↦ ±[f a, f b]` or `0`, triangles `↦ 0`; a triangle's image spans at most
an edge because all orders lie in `{m, m + 1}`), with `r ∘ incl = id` (`tRetr_tIncl`).  For a decided descriptor the vertex
map is an **orbit map** `τ` on the vertex orbits (`MPDatum.fP`, `mk(o, g) ↦ mk(τ o, g)`, well defined when `Γ_o ≤ Γ_{τ o}`),
hence `Γ`-equivariant (`fP_smul`), and the edge condition is the finite check `MPDatum.TreeChk` on the stored triangles.
For `M₇`: `τ = (0 ↦ 3, 1 ↦ 4, 2 ↦ 4)` (`v₀ ↦ ρ`, `v₁, v₂ ↦ i`; `Γ_{v_j} = ⟨−I⟩` is central).

**The result.**  `Cx.acyclic_of_retract`: a retract of an acyclic complex is acyclic.  With `model7_AC`:
`treeses_exact k`: the augmented chain complex `0 → k[Γ·[ρ,i]] → k[Γ·ρ] ⊕ k[Γ·i] → k → 0` of the tree is acyclic over every
commutative ring `k` — [C] `eq:treeses` (1), with no appeal to Serre.  No homotopy is used, so the `N_{−I}` obstruction of
`LV0_audit.md` §3.1 does not arise.
-/

namespace GL2.MP

universe u v

namespace Cx

variable {k : Type u} [Ring k]

/-- **A retract of an acyclic complex is acyclic**: `i : A → B`, `r : B → A` chain maps with `r ∘ i = id`. -/
theorem acyclic_of_retract {A B : Cx.{u, v} k} (i : ∀ n, A.X n →ₗ[k] B.X n) (r : ∀ n, B.X n →ₗ[k] A.X n)
    (hi : ∀ n x, B.d n (i (n + 1) x) = i n (A.d n x)) (hr : ∀ n x, A.d n (r (n + 1) x) = r n (B.d n x))
    (hri : ∀ n x, r n (i n x) = x) (hB : B.Acyclic) : A.Acyclic := by
  refine ⟨fun x => ?_, fun n x hx => ?_⟩
  · obtain ⟨y, hy⟩ := hB.1 (i 0 x)
    exact ⟨r 1 y, by rw [hr, hy, hri]⟩
  · obtain ⟨y, hy⟩ := hB.2 n (i (n + 1) x) (by rw [hi, hx, map_zero])
    exact ⟨r (n + 2) y, by rw [hr, hy, hri]⟩

end Cx

open Finsupp

namespace BCx

variable {k : Type} [CommRing k]

/-- Maps given on bases commute with the differentials as soon as they do on basis elements. -/
lemma lc_comm {A B : BCx k} (φ : ∀ n, B.B n → (A.B n →₀ k))
    (h : ∀ n (b : B.B (n + 1)), A.d n (φ (n + 1) b) = Finsupp.linearCombination k (φ n) (B.bd n b)) (n : ℕ)
    (x : B.B (n + 1) →₀ k) :
    A.d n (Finsupp.linearCombination k (φ (n + 1)) x) = Finsupp.linearCombination k (φ n) (B.d n x) := by
  induction x using Finsupp.induction_linear with
  | zero => simp
  | add x y hx hy => rw [map_add, map_add, hx, hy, map_add, map_add]
  | single b c => simp only [linearCombination_single, map_smul, B.d_single, h]

end BCx

namespace Cover

variable {G : Type} [Group G] {C : Cover.{0} G} (H : C.Hyp) (m : ℕ)

/-! ### The full subcomplex on the vertices of order `≥ m` -/

/-- **The ordered `s`-simplices of the full subcomplex on `{ord ≥ m}`** (first vertex of order `≥ m`). -/
abbrev TS (s : ℕ) := {x : YS H s // m ≤ H.ord (x.1 0)}

variable {H m}

lemma TS.ord_le {s : ℕ} (x : TS H m s) (j : Fin (s + 1)) : m ≤ H.ord (x.1.1 j) :=
  x.2.trans (x.1.2.1.monotone (Fin.zero_le j))

/-- The face of a tree simplex. -/
def TS.del {s : ℕ} (j : Fin (s + 2)) (x : TS H m (s + 1)) : TS H m s := ⟨YS.del j x.1, TS.ord_le x _⟩

lemma TS.ext' {s : ℕ} {x y : TS H m s} (h : x.1.1 = y.1.1) : x = y := Subtype.ext (Subtype.ext h)

variable (H m)

/-- The basis of the augmented tree complex. -/
def tB : ℕ → Type
  | 0 => Unit
  | s + 1 => TS H m s

variable (k : Type) [CommRing k]

/-- The tree augmentation. -/
noncomputable def tBd0 (_ : TS H m 0) : Unit →₀ k := single () 1

/-- The ordered boundary of the tree. -/
noncomputable def tBdS {s : ℕ} (x : TS H m (s + 1)) : TS H m s →₀ k := altSum k (fun j => TS.del j x)

/-- The boundary of the augmented tree complex. -/
noncomputable def tBd : ∀ n, tB H m (n + 1) → (tB H m n →₀ k)
  | 0 => tBd0 H m k
  | _ + 1 => tBdS H m k

/-- The inclusion of tree simplices into the simplices of `Y`, on bases. -/
def tInclB : ∀ n, tB H m n → yB H n
  | 0 => id
  | _ + 1 => Subtype.val

lemma tInclB_injective : ∀ n, Function.Injective (tInclB H m n)
  | 0 => Function.injective_id
  | _ + 1 => Subtype.val_injective

lemma altSum_comp {B B' : Type} {s : ℕ} (g : B → B') (f : Fin (s + 2) → B) :
    mapDomain g (altSum k f) = altSum k (g ∘ f) := by
  simp [altSum, mapDomain_finsetSum, mapDomain_single]

lemma tInclB_bdS {s : ℕ} (b : TS H m (s + 1)) :
    mapDomain (Subtype.val : TS H m s → YS H s) (tBdS H m k b) = yBdS H k b.1 := by
  rw [tBdS, altSum_comp]; rfl

/-- The tree boundary is the restriction of the boundary of `Y`. -/
lemma tInclB_bd : ∀ n (b : tB H m (n + 1)),
    mapDomain (tInclB H m n) (tBd H m k n b) = yBd H k n (tInclB H m (n + 1) b)
  | 0, _ => by show mapDomain id (single () 1) = single () 1; rw [mapDomain_single]
  | _ + 1, b => tInclB_bdS H m k b

/-- `mapDomain` along an injective basis map intertwining the boundaries reflects `d ∘ d = 0`. -/
lemma tBdd (n : ℕ) (b : tB H m (n + 2)) :
    Finsupp.linearCombination k (tBd H m k n) (tBd H m k (n + 1) b) = 0 := by
  have hL : Finsupp.lmapDomain k k (tInclB H m n) ∘ₗ Finsupp.linearCombination k (tBd H m k n) =
      Finsupp.linearCombination k (yBd H k n) ∘ₗ Finsupp.lmapDomain k k (tInclB H m (n + 1)) := by
    apply Finsupp.lhom_ext; intro b c
    simp only [LinearMap.comp_apply, Finsupp.lmapDomain_apply, mapDomain_single, linearCombination_single,
      mapDomain_smul, tInclB_bd]
  apply mapDomain_injective (tInclB_injective H m n)
  have := LinearMap.congr_fun hL (tBd H m k (n + 1) b)
  simp only [LinearMap.comp_apply, Finsupp.lmapDomain_apply] at this
  rw [this, tInclB_bd, mapDomain_zero]
  exact (yCx H k).bdd n _

/-- **The augmented ordered chain complex of the full subcomplex on `{ord ≥ m}`** (the tree, for `M₇` and `m = 3`). -/
noncomputable def treeCx : BCx k where
  B := tB H m
  bd := tBd H m k
  bdd := tBdd H m k

/-- The inclusion `k[𝒯_•] → k[Y_•]`. -/
noncomputable def tIncl (n : ℕ) : (treeCx H m k).toCx.X n →ₗ[k] (yCx H k).toCx.X n :=
  Finsupp.lmapDomain k k (tInclB H m n)

lemma tIncl_d (n : ℕ) (x : (treeCx H m k).toCx.X (n + 1)) :
    (yCx H k).toCx.d n (tIncl H m k (n + 1) x) = tIncl H m k n ((treeCx H m k).toCx.d n x) := by
  have hL : (yCx H k).toCx.d n ∘ₗ tIncl H m k (n + 1) = tIncl H m k n ∘ₗ (treeCx H m k).toCx.d n := by
    apply Finsupp.lhom_ext; intro (b : tB H m (n + 1)) c
    show Finsupp.linearCombination k (yBd H k n) (mapDomain (tInclB H m (n + 1)) (single b c)) =
      mapDomain (tInclB H m n) (Finsupp.linearCombination k (tBd H m k n) (single b c))
    rw [mapDomain_single, linearCombination_single, linearCombination_single, mapDomain_smul, tInclB_bd]
  exact LinearMap.congr_fun hL x

/-! ### Retraction data: a vertex map onto the tree -/

/-- **Retraction data**: a vertex map `f` that is the identity on `{ord ≥ m}`, lands there, and carries every edge of `Y`
to a vertex or an edge; all orders are `≤ m + 1`. -/
structure TreeRetr where
  f : C.P → C.P
  top : ∀ v, H.ord v ≤ m + 1
  ord_f : ∀ v, m ≤ H.ord (f v)
  f_id : ∀ v, m ≤ H.ord v → f v = v
  cov : ∀ v, ∃ o : G × C.T, InS o (f v)
  adj : ∀ x : YS H 1, f (x.1 0) = f (x.1 1) ∨ ∃ o : G × C.T, InS o (f (x.1 0)) ∧ InS o (f (x.1 1))

variable {H m}

/-- Two adjacent vertices have different orders ((IO)). -/
lemma ord_ne_of_adj {u w : C.P} (huw : u ≠ w) (o : G × C.T) (hu : InS o u) (hw : InS o w) : H.ord u ≠ H.ord w := by
  intro e
  apply huw
  have := H.inj o.2 hu hw (by rw [H.ord_smul, H.ord_smul, e])
  simpa using this

/-- `[u, w]` is an edge of `Y`. -/
def IsE (u w : C.P) : Prop := StrictMono (H.ord ∘ ![u, w]) ∧ ∃ o : G × C.T, ∀ j, InS o (![u, w] j)

lemma isE_iff {u w : C.P} : IsE (H := H) u w ↔ H.ord u < H.ord w ∧ ∃ o : G × C.T, InS o u ∧ InS o w := by
  unfold IsE
  rw [Fin.strictMono_iff_lt_succ]
  constructor
  · rintro ⟨h1, o, ho⟩
    exact ⟨h1 0, o, ho 0, ho 1⟩
  · rintro ⟨h1, o, hu, hw⟩
    refine ⟨fun i => ?_, o, fun j => ?_⟩
    · fin_cases i; exact h1
    · fin_cases j
      · exact hu
      · exact hw

variable (H m)

open Classical in
/-- **The oriented tree edge** `[u, w]`: the basis element if `ord u < ord w`, its negative if `ord w < ord u`, else `0`. -/
noncomputable def eCh (u w : C.P) : TS H m 1 →₀ k :=
  if h : IsE (H := H) u w ∧ m ≤ H.ord u then single ⟨⟨![u, w], h.1⟩, h.2⟩ 1
  else if h' : IsE (H := H) w u ∧ m ≤ H.ord w then -single ⟨⟨![w, u], h'.1⟩, h'.2⟩ 1 else 0

variable {H m k}

lemma eCh_self (u : C.P) : eCh H m k u u = 0 := by
  have : ¬ IsE (H := H) u u := fun h => lt_irrefl _ (isE_iff.1 h).1
  simp [eCh, this]

lemma eCh_swap (u w : C.P) : eCh H m k w u = -eCh H m k u w := by
  by_cases h : IsE (H := H) u w ∧ m ≤ H.ord u
  · have hn : ¬ (IsE (H := H) w u ∧ m ≤ H.ord w) := fun h' =>
      lt_asymm (isE_iff.1 h.1).1 (isE_iff.1 h'.1).1
    simp only [eCh, dif_pos h, dif_neg hn]
  · by_cases h' : IsE (H := H) w u ∧ m ≤ H.ord w
    · simp only [eCh, dif_pos h', dif_neg h, neg_neg]
    · simp only [eCh, dif_neg h, dif_neg h', neg_zero]

variable (R : TreeRetr H m)

/-- The tree vertex `f v`. -/
def fV (v : C.P) : TS H m 0 :=
  ⟨⟨![R.f v], fun a b h => by have := a.2; have := b.2; have : a.val < b.val := h; omega, by
    obtain ⟨o, ho⟩ := R.cov v; exact ⟨o, fun j => by fin_cases j; exact ho⟩⟩, R.ord_f v⟩

variable (k)

/-- The retraction on vertices. -/
noncomputable def rB1 (x : YS H 0) : TS H m 0 →₀ k := single (fV R (x.1 0)) 1

/-- The retraction on edges. -/
noncomputable def rB2 (x : YS H 1) : TS H m 1 →₀ k := eCh H m k (R.f (x.1 0)) (R.f (x.1 1))

/-- **The retraction** `k[Y_•] → k[𝒯_•]` on bases (`0` on triangles). -/
noncomputable def rB : ∀ n, yB H n → (tB H m n →₀ k)
  | 0 => fun _ => single () 1
  | 1 => rB1 k R
  | 2 => rB2 k R
  | _ + 3 => fun _ => 0

variable {k}

/-- The boundary of an oriented edge between tree vertices that are equal or adjacent. -/
lemma tBd_eCh {u w : C.P} (hu : m ≤ H.ord u) (hw : m ≤ H.ord w)
    (hadj : u = w ∨ ∃ o : G × C.T, InS o u ∧ InS o w) (cu cw : TS H m 0) (hcu : cu.1.1 0 = u) (hcw : cw.1.1 0 = w) :
    Finsupp.linearCombination k (tBdS H m k (s := 0)) (eCh H m k u w) = single cw 1 - single cu 1 := by
  have hc : ∀ (c c' : TS H m 0), c.1.1 0 = c'.1.1 0 → c = c' := fun c c' h =>
    TS.ext' (funext fun j => by fin_cases j; exact h)
  rcases hadj with rfl | ⟨o, hou, how⟩
  · rw [eCh_self, map_zero, hc cu cw (hcu.trans hcw.symm), sub_self]
  by_cases huw : u = w
  · subst huw; rw [eCh_self, map_zero, hc cu cw (hcu.trans hcw.symm), sub_self]
  have hne := ord_ne_of_adj (H := H) huw o hou how
  have key : ∀ (a b : C.P) (h : IsE (H := H) a b ∧ m ≤ H.ord a) (ca cb : TS H m 0), ca.1.1 0 = a → cb.1.1 0 = b →
      Finsupp.linearCombination k (tBdS H m k (s := 0)) (single ⟨⟨![a, b], h.1⟩, h.2⟩ 1) =
        single cb 1 - single ca 1 := by
    intro a b h ca cb ha hb
    rw [linearCombination_single, one_smul, tBdS, altSum, Fin.sum_univ_two,
      hc (TS.del 0 ⟨⟨![a, b], h.1⟩, h.2⟩) cb (by rw [hb]; rfl), hc (TS.del 1 ⟨⟨![a, b], h.1⟩, h.2⟩) ca (by rw [ha]; rfl)]
    simp [sub_eq_add_neg]
  rcases lt_or_gt_of_ne hne with hlt | hgt
  · have h : IsE (H := H) u w ∧ m ≤ H.ord u := ⟨isE_iff.2 ⟨hlt, o, hou, how⟩, hu⟩
    rw [eCh, dif_pos h, key u w h cu cw hcu hcw]
  · have h : IsE (H := H) w u ∧ m ≤ H.ord w := ⟨isE_iff.2 ⟨hgt, o, how, hou⟩, hw⟩
    rw [← neg_neg (eCh H m k u w), ← eCh_swap, map_neg, eCh, dif_pos h, key w u h cw cu hcw hcu, neg_sub]

/-- Three vertices of orders in `{m, m+1}`, pairwise equal or adjacent, span at most an edge: the oriented boundary of
the degenerate triangle vanishes. -/
lemma eCh_tri {a b c : C.P} (ha : m ≤ H.ord a) (hb : m ≤ H.ord b) (hc : m ≤ H.ord c) (ta : H.ord a ≤ m + 1)
    (tb : H.ord b ≤ m + 1) (tc : H.ord c ≤ m + 1)
    (hab : a = b ∨ ∃ o : G × C.T, InS o a ∧ InS o b) (hac : a = c ∨ ∃ o : G × C.T, InS o a ∧ InS o c)
    (hbc : b = c ∨ ∃ o : G × C.T, InS o b ∧ InS o c) :
    eCh H m k b c - eCh H m k a c + eCh H m k a b = 0 := by
  by_cases e1 : a = b
  · subst e1; rw [eCh_self]; abel
  by_cases e2 : b = c
  · subst e2; rw [eCh_self]; abel
  by_cases e3 : a = c
  · subst e3; rw [eCh_self, eCh_swap]; abel
  exfalso
  obtain ⟨o1, h1, h1'⟩ := hab.resolve_left e1
  obtain ⟨o2, h2, h2'⟩ := hac.resolve_left e3
  obtain ⟨o3, h3, h3'⟩ := hbc.resolve_left e2
  have n1 := ord_ne_of_adj (H := H) e1 o1 h1 h1'
  have n2 := ord_ne_of_adj (H := H) e3 o2 h2 h2'
  have n3 := ord_ne_of_adj (H := H) e2 o3 h3 h3'
  omega

lemma lc_zero_fun {α M : Type} [AddCommGroup M] [Module k M] (x : α →₀ k) :
    Finsupp.linearCombination k (fun _ : α => (0 : M)) x = 0 := by
  rw [Finsupp.linearCombination_apply]; simp

lemma rB_comm0 (b : YS H 0) :
    Finsupp.linearCombination k (tBd0 H m k) (rB1 k R b) = Finsupp.linearCombination k (fun _ : Unit => single () (1 : k))
      (yBd0 H k b) := by
  simp [rB1, tBd0, yBd0]

lemma rB_comm1 (b : YS H 1) :
    Finsupp.linearCombination k (tBdS H m k (s := 0)) (rB2 k R b) = Finsupp.linearCombination k (rB1 k R) (yBdS H k b) := by
  rw [yBdS, lc_altSum]
  change _ = ∑ i : Fin 2, ((-1 : k) ^ (i : ℕ)) • rB1 k R (YS.del i b)
  rw [Fin.sum_univ_two]
  simp only [rB1, rB2, Fin.val_zero, Fin.val_one, pow_zero, pow_one, one_smul, neg_one_smul]
  rw [tBd_eCh (R.ord_f _) (R.ord_f _) (R.adj b) (fV R (b.1 0)) (fV R (b.1 1)) rfl rfl, sub_eq_add_neg]
  rfl

lemma rB_comm2 (b : YS H 2) :
    Finsupp.linearCombination k (tBdS H m k (s := 1)) 0 = Finsupp.linearCombination k (rB2 k R) (yBdS H k b) := by
  rw [map_zero, yBdS, lc_altSum]
  have hb : ∀ i j : Fin 3, i < j → R.f (b.1 i) = R.f (b.1 j) ∨ ∃ o : G × C.T, InS o (R.f (b.1 i)) ∧ InS o (R.f (b.1 j)) :=
    fun i j hij => by
      obtain ⟨o, ho⟩ := b.2.2
      exact R.adj ⟨![b.1 i, b.1 j], (isE_iff.2 ⟨b.2.1 hij, o, ho i, ho j⟩).1, o, fun l => by
        fin_cases l
        · exact ho i
        · exact ho j⟩
  have := eCh_tri (k := k) (R.ord_f (b.1 0)) (R.ord_f (b.1 1)) (R.ord_f (b.1 2)) (R.top _) (R.top _) (R.top _)
    (hb 0 1 (by decide)) (hb 0 2 (by decide)) (hb 1 2 (by decide))
  rw [← this]
  change _ = ∑ i : Fin 3, ((-1 : k) ^ (i : ℕ)) • rB2 k R (YS.del i b)
  rw [Fin.sum_univ_three]
  show _ = (-1 : k) ^ 0 • eCh H m k (R.f (b.1 1)) (R.f (b.1 2)) + (-1 : k) ^ 1 • eCh H m k (R.f (b.1 0)) (R.f (b.1 2)) +
    (-1 : k) ^ 2 • eCh H m k (R.f (b.1 0)) (R.f (b.1 1))
  simp [sub_eq_add_neg]

/-- **`r` is a chain map.** -/
lemma rB_comm : ∀ n (b : yB H (n + 1)),
    (treeCx H m k).d n (rB k R (n + 1) b) = Finsupp.linearCombination k (rB k R n) (yBd H k n b)
  | 0, b => rB_comm0 R b
  | 1, b => rB_comm1 R b
  | 2, b => rB_comm2 R b
  | n + 3, b => by
    show Finsupp.linearCombination k (tBd H m k (n + 3)) (0 : tB H m (n + 4) →₀ k) =
      Finsupp.linearCombination k (fun _ : yB H (n + 3) => (0 : tB H m (n + 3) →₀ k)) (yBd H k (n + 3) b)
    rw [map_zero, lc_zero_fun]

/-- The tree has no simplices of dimension `≥ 2` (all orders lie in `{m, m+1}`). -/
lemma TS.isEmpty (htop : ∀ v, H.ord v ≤ m + 1) (s : ℕ) (x : TS H m (s + 2)) : False := by
  have h0 := x.2
  have h2 : H.ord (x.1.1 0) < H.ord (x.1.1 1) := x.1.2.1 (by rw [Fin.lt_def]; simp)
  have h3 : H.ord (x.1.1 1) < H.ord (x.1.1 2) :=
    x.1.2.1 (by rw [Fin.lt_def]; simp [Nat.mod_eq_of_lt (show 2 < s + 2 + 1 by omega)])
  have := htop (x.1.1 2)
  omega

/-- **`r ∘ incl = id`** on bases. -/
lemma rB_tInclB : ∀ n (b : tB H m n), rB k R n (tInclB H m n b) = single b 1
  | 0, b => rfl
  | 1, b => by
    show single (fV R (b.1.1 0)) 1 = single b 1
    congr 1
    exact TS.ext' (funext fun j => by fin_cases j; exact R.f_id _ b.2)
  | 2, b => by
    show eCh H m k (R.f (b.1.1 0)) (R.f (b.1.1 1)) = single b 1
    rw [R.f_id _ b.2, R.f_id _ (TS.ord_le b 1)]
    have h : IsE (H := H) (b.1.1 0) (b.1.1 1) ∧ m ≤ H.ord (b.1.1 0) := by
      refine ⟨isE_iff.2 ⟨b.1.2.1 (by decide), ?_⟩, b.2⟩
      obtain ⟨o, ho⟩ := b.1.2.2
      exact ⟨o, ho 0, ho 1⟩
    rw [eCh, dif_pos h]
    congr 1
    exact TS.ext' (funext fun j => by fin_cases j <;> rfl)
  | s + 3, b => (TS.isEmpty R.top s b).elim

variable (k)

/-- **The retraction** `r : k[Y_•] → k[𝒯_•]`. -/
noncomputable def tRetr (n : ℕ) : (yCx H k).toCx.X n →ₗ[k] (treeCx H m k).toCx.X n :=
  Finsupp.linearCombination k (rB k R n)

/-- **`r ∘ incl = id`.** -/
theorem tRetr_tIncl (n : ℕ) (x : (treeCx H m k).toCx.X n) : tRetr k R n (tIncl H m k n x) = x := by
  have hL : tRetr k R n ∘ₗ tIncl H m k n = LinearMap.id := by
    apply Finsupp.lhom_ext; intro (b : tB H m n) c
    show Finsupp.linearCombination k (rB k R n) (mapDomain (tInclB H m n) (single b c)) = single b c
    rw [mapDomain_single, linearCombination_single, rB_tInclB, smul_single, smul_eq_mul, mul_one]
  exact LinearMap.congr_fun hL x

/-- **`r` is a chain map.** -/
theorem tRetr_d (n : ℕ) (x : (yCx H k).toCx.X (n + 1)) :
    (treeCx H m k).toCx.d n (tRetr k R (n + 1) x) = tRetr k R n ((yCx H k).toCx.d n x) :=
  BCx.lc_comm (A := treeCx H m k) (B := yCx H k) (rB k R) (rB_comm R) n x

include R in
/-- **The tree is acyclic** whenever `Y` is ((AC) and a retraction). -/
theorem treeCx_acyclic (hAC : (yCx H k).toCx.Acyclic) : (treeCx H m k).toCx.Acyclic :=
  Cx.acyclic_of_retract (tIncl H m k) (tRetr k R) (tIncl_d H m k) (tRetr_d k R) (tRetr_tIncl k R) hAC

end Cover

/-! ### Decided descriptors: the orbit map and its finite check -/

namespace MPDatum

open GL2.TauChk
open scoped MatrixGroups

/-- **The tree check** for an orbit map `τ` (on the vertex orbits) and a point map `φ` (on the lifted points), threshold
`m`: `τ` lands in `{≥ m}` and is the identity there, `Γ_o ≤ Γ_{τ o}`; `φ p` is the point `τ`-image of `p`; every image orbit
has a point in a maximal simplex; the `φ`-image of two points of a maximal simplex is one point or lies in a maximal
simplex. -/
structure TreeChk (D : MPDatum) (m : ℕ) (τ φ : ℕ → ℕ) : Prop where
  len : D.stab.length ≤ m + 2
  orb : ∀ o < D.stab.length, τ o < D.stab.length ∧ m ≤ τ o ∧ (m ≤ o → τ o = o) ∧ ∀ h ∈ D.st o, h ∈ D.st (τ o)
  pts : ∀ p < D.pts.length, φ p < D.pts.length ∧ τ (D.orb p) = D.orb (φ p) ∧ D.gp (φ p) = D.gp p
  cov : ∀ o < D.stab.length, ∃ p < D.pts.length, D.orb p = τ o ∧ ∃ σ < D.top.length, p ∈ D.tv σ
  adj : ∀ σ < D.top.length, ∀ p ∈ D.tv σ, ∀ q ∈ D.tv σ,
    φ p = φ q ∨ ∃ σ' < D.top.length, φ p ∈ D.tv σ' ∧ φ q ∈ D.tv σ'

variable {D : MPDatum} (hW : D.WF) {m : ℕ} {τ φ : ℕ → ℕ} (hT : TreeChk D m τ φ)

/-- The orbit map on `Fin`. -/
def tauF (o : Fin D.stab.length) : Fin D.stab.length := ⟨τ o, (hT.orb o o.2).1⟩

lemma sub_le_tauF (o : Fin D.stab.length) : sub hW o ≤ sub hW (tauF hT o) := by
  rintro _ ⟨h, hh, rfl⟩
  exact ⟨h, (hT.orb o o.2).2.2.2 h hh, rfl⟩

/-- **The vertex map** `mk(o, g) ↦ mk(τ o, g)` (well defined since `Γ_o ≤ Γ_{τ o}`). -/
def fP : P D hW → P D hW :=
  Quotient.map (fun x => (tauF hT x.1, x.2)) (fun x _ h => ⟨congrArg (tauF hT) h.1, sub_le_tauF hW hT x.1 h.2⟩)

lemma fP_mkP (x : Fin D.stab.length × SL(2, ℤ)) : fP hW hT (mkP hW x) = mkP hW (tauF hT x.1, x.2) := rfl

/-- **The vertex map is `Γ`-equivariant.** -/
theorem fP_smul (γ : SL(2, ℤ)) (v : P D hW) : fP hW hT (γ • v) = γ • fP hW hT v := by
  induction v using Quotient.ind with
  | _ x => rfl

lemma fP_pt (p : Fin D.pts.length) : fP hW hT (pt hW p) = pt hW ⟨φ p, (hT.pts p p.2).1⟩ := by
  unfold pt
  rw [fP_mkP]
  obtain ⟨-, h1, h2⟩ := hT.pts p p.2
  exact congrArg (mkP hW) (Prod.ext (Fin.ext h1) (by dsimp only; rw [h2]))

lemma pt_mem_vtx {σ : Fin D.top.length} {p : ℕ} (hp : p ∈ D.tv σ) :
    pt hW ⟨p, hW.top_pts σ σ.2 p hp⟩ ∈ (cover hW).vtx σ := ⟨_, hp, rfl⟩

lemma vtx_inv {o : SL(2, ℤ) × Fin D.top.length} {v : P D hW} (h : o.1⁻¹ • v ∈ (cover hW).vtx o.2) :
    ∃ p : Fin D.pts.length, (p : ℕ) ∈ D.tv o.2 ∧ v = o.1 • pt hW p := by
  obtain ⟨p, hp, e⟩ := h
  exact ⟨p, hp, by rw [← e, smul_inv_smul]⟩

lemma inS_smul (g : SL(2, ℤ)) (σ : Fin D.top.length) (v : (cover hW).P) (h : v ∈ (cover hW).vtx σ) :
    Cover.InS (C := cover hW) (g, σ) (g • v) := by
  unfold Cover.InS; dsimp only; rw [inv_smul_smul]; exact h

variable (hio : D.ioB = true) (hdim : D.dimB = true)

/-- **The retraction data of a decided descriptor** from a passed tree check. -/
def treeRetr : Cover.TreeRetr (hyp hW hio hdim) m where
  f := fP hW hT
  top v := by
    induction v using Quotient.ind with
    | _ x => show (x.1 : ℕ) ≤ m + 1; have := x.1.2; have := hT.len; omega
  ord_f v := by
    induction v using Quotient.ind with
    | _ x => exact (hT.orb x.1 x.1.2).2.1
  f_id v hv := by
    induction v using Quotient.ind with
    | _ x => exact congrArg (mkP hW) (Prod.ext (Fin.ext ((hT.orb x.1 x.1.2).2.2.1 hv)) rfl)
  cov v := by
    induction v using Quotient.ind with
    | _ x =>
      obtain ⟨p, hp, hpo, σ, hσ, hpσ⟩ := hT.cov x.1 x.1.2
      refine ⟨(x.2 * (M22.toSL (D.gp p))⁻¹, ⟨σ, hσ⟩), ⟨⟨p, hp⟩, hpσ, ?_⟩⟩
      show (x.2 * (M22.toSL (D.gp p))⁻¹)⁻¹ • mkP hW (tauF hT x.1, x.2) = mkP hW (⟨D.orb p, _⟩, M22.toSL (D.gp p))
      rw [smul_mkP]
      exact congrArg (mkP hW) (Prod.ext (Fin.ext hpo.symm) (by dsimp only; group))
  adj x := by
    obtain ⟨o, ho⟩ := x.2.2
    obtain ⟨p, hp, e0⟩ := vtx_inv hW (ho 0)
    obtain ⟨q, hq, e1⟩ := vtx_inv hW (ho 1)
    show fP hW hT (x.1 0) = fP hW hT (x.1 1) ∨ _
    rw [e0, e1, fP_smul, fP_smul, fP_pt, fP_pt]
    rcases hT.adj (o.2 : Fin D.top.length).1 (o.2 : Fin D.top.length).2 p hp q hq with h | ⟨σ', hσ', hp', hq'⟩
    · left; congr 2; exact Fin.ext h
    · right
      exact ⟨(o.1, ⟨σ', hσ'⟩), inS_smul hW o.1 ⟨σ', hσ'⟩ _ (pt_mem_vtx hW (σ := ⟨σ', hσ'⟩) hp'),
        inS_smul hW o.1 ⟨σ', hσ'⟩ _ (pt_mem_vtx hW (σ := ⟨σ', hσ'⟩) hq')⟩

end MPDatum

/-! ### `M₇`: the tree `Γ·[ρ, i]` and [C] `eq:treeses` (1) -/

open GL2.TauChk
open scoped MatrixGroups

/-- The orbit map of `M₇`: `v₀ ↦ ρ`, `v₁, v₂ ↦ i` (orbits `0, 1, 2 ↦ 3, 4, 4`). -/
def model7Tau (o : ℕ) : ℕ := [3, 4, 4, 3, 4].getD o 0

/-- The point map of `M₇` (`pt 0 ↦ pt 4 = ρ`, `pt 3 = T·pt 0 ↦ pt 5 = Tρ`, `pt 1, pt 2 ↦ pt 6 = i`). -/
def model7Phi (p : ℕ) : ℕ := [4, 6, 6, 5, 4, 5, 6].getD p 0

/-- **The tree check for `M₇`** (`decide +kernel`). -/
theorem model7_treeChk : MPDatum.TreeChk model7MP 3 model7Tau model7Phi :=
  ⟨by decide +kernel, by decide +kernel, by decide +kernel, by decide +kernel, by decide +kernel⟩

/-- **The retraction of `Y` onto the tree** for `M₇`. -/
noncomputable def model7TreeRetr : Cover.TreeRetr model7Hyp 3 :=
  MPDatum.treeRetr model7_wf model7_treeChk model7_rules.2.1 model7_dimB

/-- **[C] `eq:treeses` (1)**: the augmented chain complex `0 → k[Γ·[ρ, i]] → k[Γ·ρ] ⊕ k[Γ·i] → k → 0` of the tree of
`SL₂(ℤ)` is acyclic, over every commutative ring `k`. -/
theorem treeses_exact (k : Type) [CommRing k] : (Cover.treeCx model7Hyp 3 k).toCx.Acyclic :=
  Cover.treeCx_acyclic k model7TreeRetr (model7_AC k)

/-- `ρ = pt 4` (orbit `3`, `Γ_ρ = ⟨U⟩` of order 6). -/
noncomputable def model7Rho : model7Cover.P := MPDatum.pt model7_wf ⟨4, by decide⟩

/-- `i = pt 6` (orbit `4`, `Γ_i = ⟨S⟩` of order 4). -/
noncomputable def model7I : model7Cover.P := MPDatum.pt model7_wf ⟨6, by decide⟩

/-- `S = [[0, -1], [1, 0]]`. -/
def model7S : M22 := ⟨0, -1, 1, 0⟩

theorem model7_tree_data :
    model7MP.stab.length = 5 ∧ model7MP.top.length = 5 ∧ model7MP.orb 4 = 3 ∧ model7MP.orb 6 = 4 ∧
      model7MP.gp 4 = ⟨1, 0, 0, 1⟩ ∧ model7MP.gp 6 = ⟨1, 0, 0, 1⟩ ∧ model7S.det = 1 ∧
      model7MP.maps model7S 4 5 = true ∧ model7MP.maps model7S 6 6 = true := by
  decide +kernel

/-- The edges of `M₇` between orbits `≥ 3` inside one maximal simplex are `[pt 4, pt 6]` and `[pt 5, pt 6]`. -/
theorem model7_edge_chk : ∀ σ < 5, ∀ p ∈ model7MP.tv σ, ∀ q ∈ model7MP.tv σ, 3 ≤ model7MP.orb p →
    model7MP.orb p < model7MP.orb q → (p = 4 ∧ q = 6) ∨ (p = 5 ∧ q = 6) := by
  decide +kernel

lemma model7_ord_smul_pt (g : SL(2, ℤ)) (p : Fin model7MP.pts.length) :
    model7Hyp.ord (g • (MPDatum.pt model7_wf p : model7Cover.P)) = model7MP.orb p := rfl

/-- **The tree vertices** (orders `≥ 3`) are the `Γ`-translates of `ρ` and `i`: `Γ/⟨U⟩ ⊔ Γ/⟨S⟩`. -/
theorem model7_tree_vtx (v : model7Cover.P) : 3 ≤ model7Hyp.ord v ↔ ∃ g : SL(2, ℤ), v = g • model7Rho ∨ v = g • model7I := by
  obtain ⟨hlen, -, h4, h6, g4, g6, -⟩ := model7_tree_data
  constructor
  · induction v using Quotient.ind with
    | _ x =>
      intro hx
      change 3 ≤ (x.1 : ℕ) at hx
      have hlt : (x.1 : ℕ) < 5 := hlen ▸ x.1.2
      refine ⟨x.2, ?_⟩
      have hI : ∀ (o : ℕ) (ho : o < model7MP.stab.length) (p : Fin model7MP.pts.length), model7MP.orb p = o →
          model7MP.gp p = ⟨1, 0, 0, 1⟩ → (x.1 : ℕ) = o →
          Quotient.mk _ x = x.2 • MPDatum.pt model7_wf p := fun o ho p hpo hpg hxo => by
        show MPDatum.mkP model7_wf x = MPDatum.mkP model7_wf (_, x.2 * M22.toSL (model7MP.gp p))
        rw [hpg, M22.toSL_one, mul_one]
        exact congrArg (MPDatum.mkP model7_wf) (Prod.ext (Fin.ext (hxo.trans hpo.symm)) rfl)
      rcases (by omega : (x.1 : ℕ) = 3 ∨ (x.1 : ℕ) = 4) with h | h
      · exact Or.inl (hI 3 (by omega) _ h4 g4 h)
      · exact Or.inr (hI 4 (by omega) _ h6 g6 h)
  · rintro ⟨g, rfl | rfl⟩
    · exact ((model7_ord_smul_pt g ⟨4, by decide⟩).trans h4).ge
    · exact le_of_le_of_eq (by decide : 3 ≤ 4) ((model7_ord_smul_pt g ⟨6, by decide⟩).trans h6).symm

/-- **The tree edges are the `Γ`-translates of `[ρ, i]`**: `𝒯 = Γ·[ρ, i]` (the edge `[Tρ, i] = S·[ρ, i]` is one of them). -/
theorem model7_tree_edges (x : Cover.TS model7Hyp 3 1) :
    ∃ g : SL(2, ℤ), x.1.1 0 = g • model7Rho ∧ x.1.1 1 = g • model7I := by
  obtain ⟨-, htop, -, -, -, -, hS, hS45, hS66⟩ := model7_tree_data
  obtain ⟨o, ho⟩ := x.1.2.2
  obtain ⟨p, hp, e0⟩ := MPDatum.vtx_inv model7_wf (ho 0)
  obtain ⟨q, hq, e1⟩ := MPDatum.vtx_inv model7_wf (ho 1)
  have h3 : 3 ≤ model7MP.orb p := by have := x.2; rw [e0] at this; exact this.trans_eq (model7_ord_smul_pt _ _)
  have hlt : model7MP.orb p < model7MP.orb q := by
    have := x.1.2.1 (show (0 : Fin 2) < 1 by decide)
    simp only [Function.comp_apply] at this
    rw [e0, e1] at this
    exact (model7_ord_smul_pt _ _).symm.trans_lt (this.trans_eq (model7_ord_smul_pt _ _))
  have hσ : Fin.val (n := model7MP.top.length) o.2 < 5 := htop ▸ (Fin.isLt (n := model7MP.top.length) o.2)
  rcases model7_edge_chk (Fin.val (n := model7MP.top.length) o.2) hσ p hp q hq h3 hlt with ⟨hp4, hq6⟩ | ⟨hp5, hq6⟩
  · refine ⟨o.1, ?_, ?_⟩
    · rw [e0]; congr 2; exact Fin.ext hp4
    · rw [e1]; congr 2; exact Fin.ext hq6
  · have hρ : M22.toSL model7S • model7Rho = MPDatum.pt model7_wf ⟨5, by decide⟩ :=
      (MPDatum.smul_pt_eq_iff model7_wf model7S hS _ _).2 hS45
    have hi : M22.toSL model7S • model7I = model7I := (MPDatum.smul_pt_eq_iff model7_wf model7S hS _ _).2 hS66
    refine ⟨o.1 * M22.toSL model7S, ?_, ?_⟩
    · rw [e0, mul_smul, hρ]; congr 2; exact Fin.ext hp5
    · rw [e1, mul_smul, hi, model7I]; congr 2; exact Fin.ext hq6

/-- A point with `g_p = I` is fixed by `g` iff `g ∈ Γ_{o_p}`. -/
lemma model7_fix_iff (g : SL(2, ℤ)) (p : Fin model7MP.pts.length) (hp : model7MP.gp p = ⟨1, 0, 0, 1⟩) :
    g • MPDatum.pt model7_wf p = MPDatum.pt model7_wf p ↔
      g ∈ MPDatum.sub model7_wf ⟨model7MP.orb p, model7_wf.pts_orb p p.2⟩ := by
  unfold MPDatum.pt
  rw [hp, M22.toSL_one, MPDatum.smul_mkP, MPDatum.mkP_eq_iff]
  simp

/-- `Γ_ρ ∩ Γ_i = {±I}` on the stored stabilisers. -/
theorem model7_stab_cap : ∀ a ∈ model7MP.st (model7MP.orb 4), a ∈ model7MP.st (model7MP.orb 6) →
    a = ⟨1, 0, 0, 1⟩ ∨ a = ⟨-1, 0, 0, -1⟩ := by
  decide +kernel

/-- **The stabiliser of the edge `[ρ, i]` is `⟨−I⟩`**: an element fixing `ρ` and `i` is `±I` (and `−I` fixes both), so the
tree's edge set is `Γ/⟨−I⟩`. -/
theorem model7_tree_edge_stab (g : SL(2, ℤ)) (hρ : g • model7Rho = model7Rho) (hi : g • model7I = model7I) :
    g = M22.toSL ⟨1, 0, 0, 1⟩ ∨ g = M22.toSL ⟨-1, 0, 0, -1⟩ := by
  obtain ⟨-, -, -, -, g4, g6, -⟩ := model7_tree_data
  obtain ⟨a, ha, rfl⟩ := (model7_fix_iff g ⟨4, by decide⟩ g4).1 hρ
  obtain ⟨b, hb, e⟩ := (model7_fix_iff _ ⟨6, by decide⟩ g6).1 hi
  have hda := model7_wf.stab_det _ (model7_wf.pts_orb 4 (by decide)) a ha
  have hdb := model7_wf.stab_det _ (model7_wf.pts_orb 6 (by decide)) b hb
  have hab : b = a := M22.toSL_injOn hdb hda e
  subst hab
  rcases model7_stab_cap b ha hb with rfl | rfl
  · exact Or.inl rfl
  · exact Or.inr rfl

theorem model7_negI_fixes :
    M22.toSL ⟨-1, 0, 0, -1⟩ • model7Rho = model7Rho ∧ M22.toSL ⟨-1, 0, 0, -1⟩ • model7I = model7I :=
  ⟨(MPDatum.smul_pt_eq_iff model7_wf _ (by decide) _ _).2 (by decide +kernel),
    (MPDatum.smul_pt_eq_iff model7_wf _ (by decide) _ _).2 (by decide +kernel)⟩

end GL2.MP
