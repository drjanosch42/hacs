import Mathlib.LinearAlgebra.Prod
import Mathlib.Tactic.Abel

/-!
# MP-L1, row L6 (slot 22): the bounded assembly lemma, element-wise

[GT] `gt_part_MP.tex`, rows `def:mp-cx`, `lem:mp-cone-acyclic`, `lem:mp-cone-base`, `lem:mp-contract`,
`lem:mp-assembly-rows`, `lem:mp-assembly-cols`, `thm:mp-assembly-grid`, backing [C] `lem:mpassembly` (bounded form
`0 ≤ s ≤ 2`, implementation plan R6: element-wise in modules over a ring, no `HomologicalComplex₂`).

**Conventions.** A `Cx k` is an `ℕ`-graded chain complex of `k`-modules, `d n : X (n+1) → X n`. It is **acyclic**
(`Cx.Acyclic`) if it is exact in every degree *including degree `0`*: `d 0` is onto and every cycle of degree `n + 1` is a
boundary.  An augmented complex `⋯ → P₁ → P₀ → A → 0` is read with `A` in degree `0`; it is acyclic iff the augmentation is
a quasi-isomorphism `P_• → A`.

A double complex bounded in the `s`-direction is given **by its rows, already placed in total degree**: row `s` is a `Cx`
whose degree-`n` term is the entry `(n - s, s)` (zero for `n < s`), and the vertical maps `v : R_{s}(n+1) → R_{s-1}(n)`
lower the total degree by one.  The total complex of rows `R₀ ← R₁ ← R₂ ← R₃` is the iterated cone
`Cone(R₀ ← Cone(R₁ ← Cone(R₂ ← R₃)))` (`Cx.cone`), with `D(a, t) = (d a + v t, - D t)`.

* `cone_acyclic` (**rows**, one step): `R`, `T` acyclic ⇒ `Cone(R ← T)` acyclic; iterated, a bounded double complex with
  acyclic rows has an acyclic total complex (`tot3_acyclic_of_rows`).
* `base_acyclic`: `Cone(R ← T)` and `T` acyclic ⇒ `R` acyclic.
* `tot4_acyclic_of_cols` (**columns**): a four-row bounded double complex whose columns
  `0 → R₃(c+3) → R₂(c+2) → R₁(c+1) → R₀(c) → 0` are exact (and whose entries below the first quadrant vanish) has an
  acyclic total complex (a four-step staircase chase).
* `grid_bottom_row_acyclic` (**the form L6a uses**): if the three upper rows are acyclic and every column is exact, the
  bottom row is acyclic.  For the MP resolution: rows `s = 0, 1, 2` are the row augmentations `ℤ[𝔉_{•,s}] → ℤ[Y_s]`
  ([C] `lem:mprows`), the columns are `ℤ[𝔉_{c,•}] → ℤ[𝔉'_c]` ([C] `lem:mpcols`) and `C_•(Y) → ℤ` ((AC)), and the
  bottom row is `ℤ[𝔉'_•] → ℤ`, which is therefore a resolution ([C] `thm:mpcomparison` (2)).
* `acyclic_of_contraction`: a complex with a contracting homotopy (`d h + h d = id`, `d h = id` in degree `0`) is acyclic.

This is the acyclic-assembly form of [C] `lem:mpassembly` (whose proof reduces the quasi-isomorphism statement to it by a
mapping cone); it is all the comparison theorem uses.
-/

namespace GL2.MP

universe u v

/-- **An `ℕ`-graded chain complex of `k`-modules**, `d n : X (n+1) → X n`, `d ∘ d = 0`. -/
structure Cx (k : Type u) [Ring k] where
  X : ℕ → Type v
  [acg : ∀ n, AddCommGroup (X n)]
  [mod : ∀ n, Module k (X n)]
  d : ∀ n, X (n + 1) →ₗ[k] X n
  dd : ∀ n (x : X (n + 2)), d n (d (n + 1) x) = 0

attribute [instance] Cx.acg Cx.mod

namespace Cx

variable {k : Type u} [Ring k]

/-- **Acyclic**: exact in every degree, degree `0` included (`d 0` onto). -/
def Acyclic (C : Cx.{u, v} k) : Prop :=
  (∀ x : C.X 0, ∃ y, C.d 0 y = x) ∧ ∀ n (x : C.X (n + 1)), C.d n x = 0 → ∃ y, C.d (n + 1) y = x

/-- A contracting homotopy: `d h = id` in degree `0`, `d h + h d = id` above. -/
structure Contraction (C : Cx.{u, v} k) where
  h : ∀ n, C.X n →ₗ[k] C.X (n + 1)
  h0 : ∀ x : C.X 0, C.d 0 (h 0 x) = x
  hs : ∀ n (x : C.X (n + 1)), C.d (n + 1) (h (n + 1) x) + h n (C.d n x) = x

/-- **A contractible complex is acyclic.** -/
theorem acyclic_of_contraction (C : Cx.{u, v} k) (H : C.Contraction) : C.Acyclic := by
  refine ⟨fun x => ⟨H.h 0 x, H.h0 x⟩, fun n x hx => ⟨H.h (n + 1) x, ?_⟩⟩
  have := H.hs n x
  rwa [hx, map_zero, add_zero] at this

/-- A map lowering the degree by one, `v n : T (n+1) → R n`, commuting with the differentials. -/
structure DMap (R : Cx.{u, v} k) (T : Cx.{u, v} k) where
  v : ∀ n, T.X (n + 1) →ₗ[k] R.X n
  comm : ∀ n (t : T.X (n + 2)), R.d n (v (n + 1) t) = v n (T.d (n + 1) t)

/-- **The cone** `Cone(R ← T)`: `X n = R n × T n`, `D (a, t) = (d a + v t, - d t)`. -/
abbrev cone {R T : Cx.{u, v} k} (φ : DMap R T) : Cx.{u, v} k where
  X n := R.X n × T.X n
  d n := LinearMap.prod (R.d n ∘ₗ LinearMap.fst k _ _ + φ.v n ∘ₗ LinearMap.snd k _ _)
    (-(T.d n ∘ₗ LinearMap.snd k _ _))
  dd n x := by
    obtain ⟨a, t⟩ := x
    refine Prod.ext ?_ ?_
    · simp [R.dd, φ.comm]
    · simp [T.dd]

@[simp] lemma cone_d_fst {R T : Cx.{u, v} k} (φ : DMap R T) (n : ℕ) (x : R.X (n + 1) × T.X (n + 1)) :
    ((cone φ).d n x).1 = R.d n x.1 + φ.v n x.2 := rfl

@[simp] lemma cone_d_snd {R T : Cx.{u, v} k} (φ : DMap R T) (n : ℕ) (x : R.X (n + 1) × T.X (n + 1)) :
    ((cone φ).d n x).2 = -T.d n x.2 := rfl

/-- **Rows, one step**: `R` and `T` acyclic ⇒ `Cone(R ← T)` acyclic. -/
theorem cone_acyclic {R T : Cx.{u, v} k} (φ : DMap R T) (hR : R.Acyclic) (hT : T.Acyclic) :
    (cone φ).Acyclic := by
  refine ⟨fun x => ?_, fun n x hx => ?_⟩
  · obtain ⟨a, t⟩ := x
    obtain ⟨t', ht'⟩ := hT.1 t
    obtain ⟨a', ha'⟩ := hR.1 (a + φ.v 0 t')
    refine ⟨(a', -t'), ?_⟩
    refine Prod.ext ?_ ?_
    · show R.d 0 a' + φ.v 0 (-t') = a; rw [ha', map_neg]; abel
    · show -T.d 0 (-t') = t; rw [map_neg, neg_neg, ht']
  · obtain ⟨a, t⟩ := x
    have h1 : R.d n a + φ.v n t = 0 := congrArg Prod.fst hx
    have h2 : -T.d n t = 0 := congrArg Prod.snd hx
    obtain ⟨t', ht'⟩ := hT.2 n t (neg_eq_zero.1 h2)
    have hc : R.d n (a + φ.v (n + 1) t') = 0 := by rw [map_add, φ.comm, ht', h1]
    obtain ⟨a', ha'⟩ := hR.2 n _ hc
    refine ⟨(a', -t'), ?_⟩
    refine Prod.ext ?_ ?_
    · show R.d (n + 1) a' + φ.v (n + 1) (-t') = a; rw [ha', map_neg]; abel
    · show -T.d (n + 1) (-t') = t; rw [map_neg, neg_neg, ht']

/-- **The base row**: `Cone(R ← T)` and `T` acyclic ⇒ `R` acyclic. -/
theorem base_acyclic {R T : Cx.{u, v} k} (φ : DMap R T) (hC : (cone φ).Acyclic) (hT : T.Acyclic) :
    R.Acyclic := by
  refine ⟨fun a => ?_, fun n a ha => ?_⟩
  · obtain ⟨⟨a', t'⟩, h⟩ := hC.1 (a, 0)
    have h1 : R.d 0 a' + φ.v 0 t' = a := congrArg Prod.fst h
    have h2 : -T.d 0 t' = 0 := congrArg Prod.snd h
    obtain ⟨t'', ht''⟩ := hT.2 0 t' (neg_eq_zero.1 h2)
    refine ⟨a' + φ.v 1 t'', ?_⟩
    rw [map_add, φ.comm, ht'', h1]
  · have hcyc : (cone φ).d n (a, 0) = 0 := by
      refine Prod.ext ?_ ?_
      · show R.d n a + φ.v n 0 = 0; rw [ha, map_zero, add_zero]
      · show -T.d n 0 = 0; simp
    obtain ⟨⟨a', t'⟩, h⟩ := hC.2 n (a, 0) hcyc
    have h1 : R.d (n + 1) a' + φ.v (n + 1) t' = a := congrArg Prod.fst h
    have h2 : -T.d (n + 1) t' = 0 := congrArg Prod.snd h
    obtain ⟨t'', ht''⟩ := hT.2 (n + 1) t' (neg_eq_zero.1 h2)
    refine ⟨a' + φ.v (n + 2) t'', ?_⟩
    rw [map_add, φ.comm, ht'', h1]

/-- The degree-lowering map `Cone(R₁ ← T) → R₀` induced by `v : R₁ → R₀` (on the `R₁`-component), when `v ∘ φ = 0`. -/
def DMap.lift {R₀ R₁ T : Cx.{u, v} k} (ψ : DMap R₀ R₁) (φ : DMap R₁ T)
    (hvv : ∀ n (t : T.X (n + 2)), ψ.v n (φ.v (n + 1) t) = 0) : DMap R₀ (cone φ) where
  v n := ψ.v n ∘ₗ LinearMap.fst k _ _
  comm n x := by
    obtain ⟨a, t⟩ := x
    show R₀.d n (ψ.v (n + 1) a) = ψ.v n (R₁.d (n + 1) a + φ.v (n + 1) t)
    rw [map_add, hvv, add_zero, ψ.comm]

/-! ### Bounded double complexes with three and four rows -/

/-- **A four-row bounded double complex** `R₀ ← R₁ ← R₂ ← R₃` (rows placed in total degree), `v ∘ v = 0`. -/
structure BDC4 (k : Type u) [Ring k] where
  R₀ : Cx.{u, v} k
  R₁ : Cx.{u, v} k
  R₂ : Cx.{u, v} k
  R₃ : Cx.{u, v} k
  v₁ : DMap R₀ R₁
  v₂ : DMap R₁ R₂
  v₃ : DMap R₂ R₃
  vv₁ : ∀ n (t : R₂.X (n + 2)), v₁.v n (v₂.v (n + 1) t) = 0
  vv₂ : ∀ n (t : R₃.X (n + 2)), v₂.v n (v₃.v (n + 1) t) = 0

namespace BDC4

variable (B : BDC4.{u, v} k)

/-- The total complex of the upper three rows, `Cone(R₁ ← Cone(R₂ ← R₃))`. -/
abbrev tot3 : Cx.{u, v} k := cone (B.v₂.lift B.v₃ B.vv₂)

/-- **The total complex** `Cone(R₀ ← Cone(R₁ ← Cone(R₂ ← R₃)))`. -/
abbrev tot : Cx.{u, v} k := cone (B.v₁.lift (B.v₂.lift B.v₃ B.vv₂) (fun n t => B.vv₁ n t.1))

/-- **Rows** (the assembly lemma for acyclic rows): the upper total complex is acyclic when `R₁, R₂, R₃` are. -/
theorem tot3_acyclic_of_rows (h₁ : B.R₁.Acyclic) (h₂ : B.R₂.Acyclic) (h₃ : B.R₃.Acyclic) : B.tot3.Acyclic :=
  cone_acyclic _ h₁ (cone_acyclic _ h₂ h₃)

/-- **Rows**: all four rows acyclic ⇒ the total complex acyclic. -/
theorem tot_acyclic_of_rows (h₀ : B.R₀.Acyclic) (h₁ : B.R₁.Acyclic) (h₂ : B.R₂.Acyclic) (h₃ : B.R₃.Acyclic) :
    B.tot.Acyclic :=
  cone_acyclic _ h₀ (B.tot3_acyclic_of_rows h₁ h₂ h₃)

/-- **Exact columns** `0 → R₃(c+3) → R₂(c+2) → R₁(c+1) → R₀(c) → 0`, and the entries below the first quadrant vanish. -/
structure ColsExact : Prop where
  surj : ∀ n (a : B.R₀.X n), ∃ b, B.v₁.v n b = a
  ex₁ : ∀ n (a : B.R₁.X (n + 1)), B.v₁.v n a = 0 → ∃ b, B.v₂.v (n + 1) b = a
  ex₂ : ∀ n (a : B.R₂.X (n + 2)), B.v₂.v (n + 1) a = 0 → ∃ b, B.v₃.v (n + 2) b = a
  inj : ∀ n (a : B.R₃.X (n + 3)), B.v₃.v (n + 2) a = 0 → a = 0
  z₁₀ : ∀ a : B.R₁.X 0, a = 0
  z₂₀ : ∀ a : B.R₂.X 0, a = 0
  z₂₁ : ∀ a : B.R₂.X 1, a = 0
  z₃₀ : ∀ a : B.R₃.X 0, a = 0
  z₃₁ : ∀ a : B.R₃.X 1, a = 0
  z₃₂ : ∀ a : B.R₃.X 2, a = 0

/-- The total differential, component by component. -/
lemma tot_d (n : ℕ) (a₀ : B.R₀.X (n + 1)) (a₁ : B.R₁.X (n + 1)) (a₂ : B.R₂.X (n + 1)) (a₃ : B.R₃.X (n + 1)) :
    B.tot.d n (a₀, a₁, a₂, a₃) =
      (B.R₀.d n a₀ + B.v₁.v n a₁, -(B.R₁.d n a₁ + B.v₂.v n a₂), B.R₂.d n a₂ + B.v₃.v n a₃, -B.R₃.d n a₃) := by
  show (_, _) = _
  simp only [Prod.mk.injEq]
  refine ⟨rfl, ?_⟩
  show -(_, _) = _
  ext
  · rfl
  · show -(-(_, _)).1 = _
    simp only [Prod.fst_neg, neg_neg]; rfl
  · show -(-(_, _)).2 = _
    simp only [Prod.snd_neg, neg_neg]; rfl

/-- One staircase step at `R₁`: a cycle `(0, a₁, a₂, a₃)` of degree `n + 1` is congruent mod boundaries to `(0, 0, a₂', a₃)`. -/
private lemma step₁ (hC : B.ColsExact) (n : ℕ) (a₁ : B.R₁.X (n + 1)) (a₂ : B.R₂.X (n + 1)) (a₃ : B.R₃.X (n + 1))
    (hcyc : B.tot.d n (0, a₁, a₂, a₃) = 0) :
    ∃ b : B.tot.X (n + 2), ∃ a₂' : B.R₂.X (n + 1), (0, a₁, a₂, a₃) - B.tot.d (n + 1) b = (0, 0, a₂', a₃) := by
  rw [tot_d] at hcyc
  have h0 : B.R₀.d n 0 + B.v₁.v n a₁ = 0 := congrArg Prod.fst hcyc
  rw [map_zero, zero_add] at h0
  obtain ⟨b₂, hb₂⟩ := hC.ex₁ n a₁ h0
  refine ⟨(0, 0, -b₂, 0), a₂ + B.R₂.d (n + 1) b₂, ?_⟩
  rw [tot_d]
  simp only [map_zero, map_neg, zero_add, neg_zero, add_zero, hb₂, Prod.mk_sub_mk, sub_zero, sub_self,
    sub_neg_eq_add, neg_neg]

/-- One staircase step at `R₂`. -/
private lemma step₂ (hC : B.ColsExact) (n : ℕ) (a₂ : B.R₂.X (n + 2)) (a₃ : B.R₃.X (n + 2))
    (hcyc : B.tot.d (n + 1) (0, 0, a₂, a₃) = 0) :
    ∃ b : B.tot.X (n + 3), ∃ a₃' : B.R₃.X (n + 2), (0, 0, a₂, a₃) - B.tot.d (n + 2) b = (0, 0, 0, a₃') := by
  rw [tot_d] at hcyc
  have h1 : -(B.R₁.d (n + 1) 0 + B.v₂.v (n + 1) a₂) = 0 := congrArg (fun x => x.2.1) hcyc
  rw [map_zero, zero_add, neg_eq_zero] at h1
  obtain ⟨b₃, hb₃⟩ := hC.ex₂ n a₂ h1
  refine ⟨(0, 0, 0, b₃), a₃ + B.R₃.d (n + 2) b₃, ?_⟩
  rw [tot_d]
  simp only [map_zero, zero_add, neg_zero, add_zero, hb₃, Prod.mk_sub_mk, sub_self, sub_neg_eq_add]

/-- **Columns** (the assembly lemma for exact columns): the four-row total complex is acyclic. -/
theorem tot_acyclic_of_cols (hC : B.ColsExact) : B.tot.Acyclic := by
  -- step 0: remove the `R₀`-component by the surjectivity of `v₁`
  have step₀ : ∀ n (x : B.tot.X n), ∃ b : B.tot.X (n + 1), ∃ a₁ a₂ a₃, x - B.tot.d n b = (0, a₁, a₂, a₃) := by
    intro n x
    obtain ⟨a₀, a₁, a₂, a₃⟩ := x
    obtain ⟨b₁, hb₁⟩ := hC.surj n a₀
    refine ⟨(0, b₁, 0, 0), a₁ + B.R₁.d n b₁, a₂, a₃, ?_⟩
    rw [tot_d]
    simp only [map_zero, zero_add, add_zero, neg_zero, hb₁, Prod.mk_sub_mk, sub_self, sub_zero, sub_neg_eq_add]
  refine ⟨fun x => ?_, fun n x hx => ?_⟩
  · obtain ⟨b, a₁, a₂, a₃, h⟩ := step₀ 0 x
    refine ⟨b, ?_⟩
    rw [hC.z₁₀ a₁, hC.z₂₀ a₂, hC.z₃₀ a₃] at h
    have : x - B.tot.d 0 b = 0 := h
    exact (sub_eq_zero.1 this).symm
  · obtain ⟨b, a₁, a₂, a₃, h⟩ := step₀ (n + 1) x
    have hcyc : B.tot.d n (0, a₁, a₂, a₃) = 0 := by rw [← h, map_sub, hx, B.tot.dd, sub_zero]
    obtain ⟨b', a₂', h'⟩ := B.step₁ hC n a₁ a₂ a₃ hcyc
    have hcyc' : B.tot.d n (0, 0, a₂', a₃) = 0 := by rw [← h', map_sub, hcyc, B.tot.dd, sub_zero]
    -- the remaining element `(0, 0, a₂', a₃)` and the boundary so far
    suffices hs : ∃ c, B.tot.d (n + 1) c = ((0, 0, a₂', a₃) : B.tot.X (n + 1)) by
      obtain ⟨c, hc⟩ := hs
      refine ⟨b + b' + c, ?_⟩
      rw [map_add, map_add, hc, ← h', ← h]; abel
    rcases n with _ | n
    · exact ⟨0, by rw [map_zero, hC.z₂₁ a₂', hC.z₃₁ a₃]; rfl⟩
    · obtain ⟨b'', a₃', h''⟩ := B.step₂ hC n a₂' a₃ hcyc'
      have hcyc'' : B.tot.d (n + 1) (0, 0, 0, a₃') = 0 := by rw [← h'', map_sub, hcyc', B.tot.dd, sub_zero]
      have ha₃' : a₃' = 0 := by
        rcases n with _ | n
        · exact hC.z₃₂ a₃'
        · rw [tot_d] at hcyc''
          have h2 : B.R₂.d (n + 2) 0 + B.v₃.v (n + 2) a₃' = 0 := congrArg (fun x => x.2.2.1) hcyc''
          rw [map_zero, zero_add] at h2
          exact hC.inj n a₃' h2
      refine ⟨b'', ?_⟩
      rw [ha₃'] at h''
      exact (sub_eq_zero.1 h'').symm

/-- **The grid lemma** (the form [C] `thm:mpcomparison` (2) uses): if the upper three rows are acyclic and every
column is exact, the bottom row `R₀` is acyclic. -/
theorem grid_bottom_row_acyclic (h₁ : B.R₁.Acyclic) (h₂ : B.R₂.Acyclic) (h₃ : B.R₃.Acyclic) (hC : B.ColsExact) :
    B.R₀.Acyclic :=
  base_acyclic _ (B.tot_acyclic_of_cols hC) (B.tot3_acyclic_of_rows h₁ h₂ h₃)

end BDC4

end Cx

end GL2.MP
