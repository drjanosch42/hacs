import GL2.MP.Bisimplicial

/-!
# MP-L2, row L6a (slot 22): `k[𝔉'_•] → k` is a resolution

[GT] `gt_part_MP.tex`, rows `lem:mp-dim`, `def:mp-grid`, `thm:mp-grid-exact`, `thm:mp-resolution`, backing [C]
`thm:mpcomparison` (2).

The four-row bounded double complex (`grid`, a `Cx.BDC4`) has the rows (in total degree)

* `R₀ = k[𝔉'_•] → k` (`fCx`), the row to be shown acyclic;
* `R₁, R₂, R₃` = the augmented rows `k[𝔉_{•,s}] → k[Y_s]`, `s = 0, 1, 2` (`rowCx`), shifted by `s + 1`;

and the vertical maps the vertical faces (`vdel`, `yBdS`), the column augmentation `(x; t) ↦ t` and the augmentation of
`Y`.  Its column `0` is the augmented ordered chain complex `k[Y_•] → k` (`yCx`), its column `n + 1` is
`k[𝔉_{n,•}] → k[𝔉'_n]`.  The rows are acyclic by L4 (`rowContr`), the columns `n + 1` by L5 (`cone_identity`, the cone on
`v*`, with the dimension bound `dim_vs` at the top), the column `0` by (AC); `Cx.BDC4.grid_bottom_row_acyclic` (L6)
gives **`fCx_acyclic`: `k[𝔉'_•] → k` is acyclic** — over every commutative ring `k`, with no invertibility, effectivity or
adaptedness hypothesis.
-/

namespace GL2.MP

namespace Cover

open Finsupp

variable {G : Type} [Group G] {C : Cover.{0} G} {H : C.Hyp}

/-! ### The dimension bound -/

lemma setOf_InS (o : G × C.T) : {v : C.P | InS o v} = (o.1 • ·) '' C.vtx o.2 := by
  ext v
  simp only [Set.mem_image, InS]
  refine ⟨fun h => ⟨_, h, smul_inv_smul _ _⟩, ?_⟩
  rintro ⟨w, hw, rfl⟩
  show o.1⁻¹ • o.1 • w ∈ C.vtx o.2
  rwa [inv_smul_smul]

lemma ncard_InS_le (H : C.Hyp) (o : G × C.T) : {v : C.P | InS o v}.ncard ≤ 3 := by
  rw [setOf_InS, Set.ncard_image_of_injective _ (MulAction.injective o.1)]
  exact H.dim o.2

lemma finite_InS (H : C.Hyp) (o : G × C.T) : {v : C.P | InS o v}.Finite := by
  rw [setOf_InS]; exact (H.fin o.2).image _

/-- **At most three vertices** in a translate of a maximal simplex. -/
lemma card_le_three (H : C.Hyp) {m : ℕ} (x : Fin m → C.P) (hx : Function.Injective x) (o : G × C.T) (h : ∀ j, InS o (x j)) :
    m ≤ 3 := by
  have h1 : (Set.range x).ncard = m := by rw [Set.ncard_range_of_injective hx, Nat.card_eq_fintype_card, Fintype.card_fin]
  have h2 : Set.range x ⊆ {v : C.P | InS o v} := by rintro _ ⟨j, rfl⟩; exact h j
  have := Set.ncard_le_ncard h2 (finite_InS H o)
  have := ncard_InS_le H o
  omega

lemma injective_of_strictMono {m : ℕ} {x : Fin m → C.P} (hx : StrictMono (H.ord ∘ x)) : Function.Injective x :=
  fun a b e => hx.injective (by simp [Function.comp, e])

/-- `Y` has no `3`-simplices. -/
lemma isEmpty_YS_three : IsEmpty (YS H 3) := by
  refine ⟨fun x => ?_⟩
  obtain ⟨o, ho⟩ := x.2.2
  have := card_le_three H x.1 (injective_of_strictMono x.2.1) o ho
  omega

/-- **On a triangle, `v*` is the first vertex** (the dimension bound). -/
lemma dim_vs {n : ℕ} (p : FS H 2 n) : p.1.1 0 = p.vs := by
  by_contra h
  obtain ⟨hne, -⟩ := p.ne_vs h
  have hinj : Function.Injective (Fin.cons p.vs p.1.1 : Fin 4 → C.P) := by
    intro a b e
    have hx := injective_of_strictMono p.2.1
    cases a using Fin.cases with
    | zero =>
      cases b using Fin.cases with
      | zero => rfl
      | succ b => exact absurd e.symm (by simpa using hne b)
    | succ a =>
      cases b using Fin.cases with
      | zero => exact absurd e (by simpa using hne a)
      | succ b => simp only [Fin.cons_succ] at e; rw [hx e]
  have := card_le_three H _ hinj (p.1.2 0) (fun j => by
    cases j using Fin.cases with
    | zero => exact vstar_mem p.lab 0
    | succ j => exact p.2.2 0 j)
  omega

/-! ### The grid -/

variable (H) (k : Type) [CommRing k]

/-- `R₁ = k[𝔉_{•,0}] → k[Y_0]`, shifted once. -/
noncomputable abbrev R₁ : BCx k := (rowCx H k 0).shift
/-- `R₂`, shifted twice. -/
noncomputable abbrev R₂ : BCx k := (rowCx H k 1).shift.shift
/-- `R₃`, shifted three times. -/
noncomputable abbrev R₃ : BCx k := (rowCx H k 2).shift.shift.shift

/-- The column augmentation `(x; t) ↦ t`. -/
noncomputable def colAug {n : ℕ} (p : FS H 0 n) : C.F' n →₀ k := single p.toF' 1

/-- `v₁ : R₁ → R₀`: the augmentation of `Y` in column `0`, the column augmentation above. -/
noncomputable def v₁b : ∀ m, (R₁ H k).B (m + 1) → ((fCx C k).B m →₀ k)
  | 0 => yBd0 H k
  | _ + 1 => colAug H k

/-- `v₂ : R₂ → R₁`: the vertical faces. -/
noncomputable def v₂b : ∀ m, (R₂ H k).B (m + 1) → ((R₁ H k).B m →₀ k)
  | 0 => fun b => PEmpty.elim b
  | 1 => yBdS H k
  | _ + 2 => vdel k

/-- `v₃ : R₃ → R₂`: the vertical faces. -/
noncomputable def v₃b : ∀ m, (R₃ H k).B (m + 1) → ((R₂ H k).B m →₀ k)
  | 0 => fun b => PEmpty.elim b
  | 1 => fun b => PEmpty.elim b
  | 2 => yBdS H k
  | _ + 3 => vdel k

variable {H}

lemma comm_aug_colAug (t : FS H 0 0) :
    Finsupp.linearCombination k (fBd0 k) (colAug H k t) =
      Finsupp.linearCombination k (yBd0 H k) (rowBd0 H k t) := by
  simp [colAug, rowBd0, fBd0, yBd0]

lemma comm_dh_colAug {n : ℕ} (t : FS H 0 (n + 1)) :
    Finsupp.linearCombination k (fBdS k) (colAug H k t) =
      Finsupp.linearCombination k (colAug H k) (rowBdS H k t) := by
  rw [colAug, Finsupp.linearCombination_single, one_smul, rowBdS, lc_altSum]
  simp only [colAug, fBdS, altSum, smul_single, smul_eq_mul, mul_one, FS.toF'_dh]

lemma comm_aug_vdel {s : ℕ} (t : FS H (s + 1) 0) :
    Finsupp.linearCombination k (rowBd0 H k) (vdel k t) = Finsupp.linearCombination k (yBdS H k) (rowBd0 H k t) := by
  rw [vdel, lc_altSum, rowBd0, Finsupp.linearCombination_single, one_smul]
  simp only [rowBd0, yBdS, altSum, smul_single, smul_eq_mul, mul_one, FS.base_dv]

lemma comm_dh_vdel {s n : ℕ} (t : FS H (s + 1) (n + 1)) :
    Finsupp.linearCombination k (rowBdS H k) (vdel k t) = Finsupp.linearCombination k (vdel k) (rowBdS H k t) := by
  rw [vdel, lc_altSum, rowBdS, lc_altSum]
  simp only [rowBdS, vdel, altSum, Finset.smul_sum, smul_single, smul_eq_mul, FS.dh_dv]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
  rw [mul_comm]

variable (H)

/-- `v₁` on bases. -/
noncomputable def v₁ : BDMap k (fCx C k) (R₁ H k) where
  vb := v₁b H k
  comm m t := by
    rcases m with _ | m
    · exact comm_aug_colAug k (H := H) t
    · exact comm_dh_colAug k (H := H) (n := m) t

/-- `v₂` on bases. -/
noncomputable def v₂ : BDMap k (R₁ H k) (R₂ H k) where
  vb := v₂b H k
  comm m t := by
    rcases m with _ | _ | m
    · exact Finsupp.ext fun b => PEmpty.elim b
    · exact comm_aug_vdel k (H := H) (s := 0) t
    · exact comm_dh_vdel k (H := H) (s := 0) (n := m) t

/-- `v₃` on bases. -/
noncomputable def v₃ : BDMap k (R₂ H k) (R₃ H k) where
  vb := v₃b H k
  comm m t := by
    rcases m with _ | _ | _ | m
    · exact Finsupp.ext fun b => PEmpty.elim b
    · exact Finsupp.ext fun b => PEmpty.elim b
    · exact comm_aug_vdel k (H := H) (s := 1) t
    · exact comm_dh_vdel k (H := H) (s := 1) (n := m) t

variable {H}

lemma vv_colAug {n : ℕ} (t : FS H 1 n) : Finsupp.linearCombination k (colAug H k) (vdel k t) = 0 := by
  rw [vdel, lc_altSum, Fin.sum_univ_two]
  simp only [colAug, FS.toF'_dv, Fin.val_zero, pow_zero, one_smul, Fin.val_one, pow_one, neg_smul, one_smul,
    add_neg_cancel]

lemma vv_vdel {s n : ℕ} (t : FS H (s + 2) n) : Finsupp.linearCombination k (vdel k) (vdel k t) = 0 := by
  classical
  rw [vdel, lc_altSum]
  let Φ : (Fin (s + 1) → C.P) → (FS H s n →₀ k) := fun g =>
    if h : StrictMono (H.ord ∘ g) ∧ ∀ i j, InS (t.1.2 i) (g j) then single ⟨(g, t.1.2), h⟩ 1 else 0
  have e : ∀ (i : Fin (s + 3)), vdel k (FS.dv i t) =
      ∑ j : Fin (s + 2), ((-1 : k) ^ (j : ℕ)) • Φ (GL2.MP.del j (GL2.MP.del i t.1.1)) := fun i => by
    simp only [vdel, altSum]
    refine Finset.sum_congr rfl fun j _ => ?_
    simp only [Φ]
    split_ifs with h
    · rw [smul_single, smul_eq_mul, mul_one]; rfl
    · exact absurd (FS.dv j (FS.dv i t)).2 h
  simp_rw [e]
  exact alt_sum_sq_k t.1.1 Φ

/-- Basis-level `v ∘ v = 0` gives module-level `v ∘ v = 0`. -/
lemma vv_of_basis {R S T : BCx k} (φ : BDMap k R S) (ψ : BDMap k S T) (n : ℕ)
    (h : ∀ t : T.B (n + 2), Finsupp.linearCombination k (φ.vb n) (ψ.vb (n + 1) t) = 0) (x : T.B (n + 2) →₀ k) :
    φ.v n (ψ.v (n + 1) x) = 0 := by
  induction x using Finsupp.induction_linear with
  | zero => simp
  | add x y hx hy => rw [map_add, map_add, hx, hy, add_zero]
  | single t c => rw [BDMap.v_single, map_smul, BDMap.v, h, smul_zero]

variable (H)

/-- **The grid** ([C] `thm:mpcomparison`, proof of (2)): rows `k[𝔉'_•] → k` and `k[𝔉_{•,s}] → k[Y_s]`, `s = 0, 1, 2`. -/
noncomputable def grid : Cx.BDC4.{0, 0} k where
  R₀ := (fCx C k).toCx
  R₁ := (R₁ H k).toCx
  R₂ := (R₂ H k).toCx
  R₃ := (R₃ H k).toCx
  v₁ := (v₁ H k).toDMap
  v₂ := (v₂ H k).toDMap
  v₃ := (v₃ H k).toDMap
  vv₁ n := vv_of_basis k (v₁ H k) (v₂ H k) n fun t => by
    rcases n with _ | n
    · exact yBdd0 H k t
    · exact vv_colAug k (H := H) (n := n) t
  vv₂ n := vv_of_basis k (v₂ H k) (v₃ H k) n fun t => by
    rcases n with _ | _ | n
    · exact PEmpty.elim t
    · exact yBddS H k (s := 0) t
    · exact vv_vdel k (H := H) (s := 0) (n := n) t

/-! ### Rows and columns of the grid -/

/-- **The rows `R₁, R₂, R₃` are acyclic** (L4, shifted). -/
theorem grid_rows : (grid H k).R₁.Acyclic ∧ (grid H k).R₂.Acyclic ∧ (grid H k).R₃.Acyclic :=
  ⟨(rowContr H k 0).shift.acyclic, (rowContr H k 1).shift.shift.acyclic, (rowContr H k 2).shift.shift.shift.acyclic⟩

variable {H}

/-- `(v*; t)`: the bottom of the column contraction. -/
noncomputable def colPt0 {n : ℕ} (f : C.F' n) : FS H 0 n :=
  ⟨(fun _ => vstar (H := H) f, f.1), fun a b hab => absurd hab (by
    have ha := a.isLt; have hb := b.isLt; rw [Fin.lt_def]; omega), fun i _ => vstar_mem f i⟩

/-- `h₀ : k[𝔉'_n] → k[𝔉_{n,0}]`, `t ↦ (v*(t); t)`. -/
noncomputable def colH0 {n : ℕ} (f : C.F' n) : FS H 0 n →₀ k := single (colPt0 (H := H) f) 1

lemma colAug_colH0 {n : ℕ} (f : C.F' n) : Finsupp.linearCombination k (colAug H k) (colH0 k f) = single f 1 := by
  rw [colH0, Finsupp.linearCombination_single, one_smul, colAug]; rfl

lemma FS.eq_colPt0 {n : ℕ} (p : FS H 0 n) (h : p.1.1 0 = p.vs) : p = colPt0 p.toF' := by
  apply Subtype.ext
  refine Prod.ext ?_ rfl
  funext m
  obtain rfl : m = 0 := Fin.eq_zero m
  exact h

lemma FS.dv_one_coneUp {n : ℕ} (p : FS H 0 n) (h : p.1.1 0 ≠ p.vs) : FS.dv 1 (p.coneUp h) = colPt0 p.toF' :=
  Subtype.ext (Prod.ext (by funext m; fin_cases m; rfl) rfl)

/-- **The bottom cone identity** `v₂ h₁ + h₀ v₁ = id` on `𝔉_{n,0}`. -/
theorem cone_identity_zero {n : ℕ} (p : FS H 0 n) :
    Finsupp.linearCombination k (vdel k) (colH k p) + Finsupp.linearCombination k (colH0 k) (colAug H k p) =
      single p 1 := by
  rw [colAug, Finsupp.linearCombination_single, one_smul, colH0]
  by_cases h : p.1.1 0 = p.vs
  · rw [colH_of_eq k h, map_zero, zero_add, ← FS.eq_colPt0 p h]
  · rw [colH_of_ne k h, Finsupp.linearCombination_single, one_smul, vdel, altSum, Fin.sum_univ_two, FS.dv_zero_coneUp,
      FS.dv_one_coneUp]
    simp only [Fin.val_zero, pow_zero, Fin.val_one, pow_one]
    rw [add_assoc, ← single_add, neg_add_cancel, single_zero, add_zero]

/-- **The top identity** `h₂ v₃ = id` on `𝔉_{n,2}` (`v*` is the first vertex of a triangle). -/
theorem cone_identity_top {n : ℕ} (p : FS H 2 n) :
    Finsupp.linearCombination k (colH k) (vdel k p) = single p 1 := by
  have := cone_identity k p
  rwa [colH_of_eq k (dim_vs p), map_zero, zero_add] at this

/-- Two linear maps out of a free module agreeing on the basis agree. -/
lemma lin_ext_basis {B M : Type} [AddCommGroup M] [Module k M] {F F' : (B →₀ k) →ₗ[k] M}
    (h : ∀ b, F (single b 1) = F' (single b 1)) (x : B →₀ k) : F x = F' x := by
  have : F = F' := Finsupp.lhom_ext fun b c => by
    rw [← mul_one c, ← smul_eq_mul, ← smul_single, map_smul, map_smul, h]
  rw [this]

variable (H)

/-- **The contraction of the column `n + 1`** (L5, [C] `lem:mpcols`): `k[𝔉_{n,2}] → k[𝔉_{n,1}] → k[𝔉_{n,0}] → k[𝔉'_n]`. -/
noncomputable def colContr (n : ℕ) : (grid H k).ColContr (n + 1) where
  h₀ := Finsupp.linearCombination k (colH0 (H := H) k)
  h₁ := Finsupp.linearCombination k (colH (H := H) k (s := 0) (n := n))
  h₂ := Finsupp.linearCombination k (colH (H := H) k (s := 1) (n := n))
  e₀ a := by
    show Finsupp.linearCombination k (colAug H k) (Finsupp.linearCombination k (colH0 (H := H) k) a) = a
    exact lin_ext_basis k (F := Finsupp.linearCombination k (colAug H k) ∘ₗ
      Finsupp.linearCombination k (colH0 (H := H) k)) (F' := LinearMap.id)
      (fun (f : C.F' n) => by
        show Finsupp.linearCombination k (colAug H k) (Finsupp.linearCombination k (colH0 (H := H) k) (single f 1)) =
          single f 1
        rw [Finsupp.linearCombination_single, one_smul]
        exact colAug_colH0 k f) a
  e₁ a := by
    show Finsupp.linearCombination k (vdel k) (Finsupp.linearCombination k (colH k) a) +
      Finsupp.linearCombination k (colH0 (H := H) k) (Finsupp.linearCombination k (colAug H k) a) = a
    exact lin_ext_basis k (F := Finsupp.linearCombination k (vdel k) ∘ₗ Finsupp.linearCombination k (colH k) +
      Finsupp.linearCombination k (colH0 (H := H) k) ∘ₗ Finsupp.linearCombination k (colAug H k)) (F' := LinearMap.id)
      (fun (p : FS H 0 n) => by
        show Finsupp.linearCombination k (vdel k) (Finsupp.linearCombination k (colH k) (single p 1)) +
          Finsupp.linearCombination k (colH0 (H := H) k) (Finsupp.linearCombination k (colAug H k) (single p 1)) =
            single p 1
        rw [Finsupp.linearCombination_single, Finsupp.linearCombination_single, one_smul, one_smul]
        exact cone_identity_zero k p) a
  e₂ a := by
    show Finsupp.linearCombination k (vdel k) (Finsupp.linearCombination k (colH k) a) +
      Finsupp.linearCombination k (colH k) (Finsupp.linearCombination k (vdel k) a) = a
    exact lin_ext_basis k (F := Finsupp.linearCombination k (vdel k) ∘ₗ Finsupp.linearCombination k (colH k) +
      Finsupp.linearCombination k (colH k) ∘ₗ Finsupp.linearCombination k (vdel k)) (F' := LinearMap.id)
      (fun (p : FS H 1 n) => by
        show Finsupp.linearCombination k (vdel k) (Finsupp.linearCombination k (colH k) (single p 1)) +
          Finsupp.linearCombination k (colH k) (Finsupp.linearCombination k (vdel k) (single p 1)) = single p 1
        rw [Finsupp.linearCombination_single, Finsupp.linearCombination_single, one_smul, one_smul]
        exact cone_identity k p) a
  e₃ a := by
    show Finsupp.linearCombination k (colH k) (Finsupp.linearCombination k (vdel k) a) = a
    exact lin_ext_basis k (F := Finsupp.linearCombination k (colH k) ∘ₗ Finsupp.linearCombination k (vdel k))
      (F' := LinearMap.id)
      (fun (p : FS H 2 n) => by
        show Finsupp.linearCombination k (colH k) (Finsupp.linearCombination k (vdel k) (single p 1)) = single p 1
        rw [Finsupp.linearCombination_single, one_smul]
        exact cone_identity_top k p) a

/-- **The column `0` is `k[Y_•] → k`**: exact under (AC) (and `Y_3 = ∅`). -/
theorem colExact_zero (hAC : (yCx H k).toCx.Acyclic) : (grid H k).ColExactAt 0 where
  surj a := hAC.1 a
  ex₁ a ha := hAC.2 0 a ha
  ex₂ a ha := hAC.2 1 a ha
  inj a ha := by
    obtain ⟨y, hy⟩ := hAC.2 2 a ha
    have : y = 0 := Finsupp.ext fun b => (isEmpty_YS_three (H := H)).elim b
    rw [← hy, this, map_zero]
    rfl

/-- **The columns of the grid are exact.** -/
theorem grid_cols (hAC : (yCx H k).toCx.Acyclic) : (grid H k).ColsExact :=
  Cx.BDC4.colsExact_of _ (fun c => by
      rcases c with _ | n
      · exact colExact_zero H k hAC
      · exact Cx.BDC4.colExactAt_of_contr _ (colContr H k n))
    (fun a => Finsupp.ext fun b => PEmpty.elim b) (fun a => Finsupp.ext fun b => PEmpty.elim b)
    (fun a => Finsupp.ext fun b => PEmpty.elim b) (fun a => Finsupp.ext fun b => PEmpty.elim b)
    (fun a => Finsupp.ext fun b => PEmpty.elim b) (fun a => Finsupp.ext fun b => PEmpty.elim b)

/-- **L6a ([C] `thm:mpcomparison` (2)): `k[𝔉'_•] → k` is acyclic**, i.e. the labelled Čech nerve of the cover of `Y` by
its closed maximal simplices resolves `k` — over every commutative ring `k`, given (AC) over `k`, (IO), finiteness and
`dim ≤ 2`; no invertibility, effectivity or adaptedness. -/
theorem fCx_acyclic (hAC : (yCx H k).toCx.Acyclic) : (fCx C k).toCx.Acyclic := by
  obtain ⟨h₁, h₂, h₃⟩ := grid_rows H k
  exact (grid H k).grid_bottom_row_acyclic h₁ h₂ h₃ (grid_cols H k hAC)

end Cover

end GL2.MP
