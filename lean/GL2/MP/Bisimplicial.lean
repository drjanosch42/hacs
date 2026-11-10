import GL2.MP.BasisCx
import GL2.MP.Cochains

/-!
# MP-L2, rows L3–L5 (slot 22): the free bisimplicial set `𝔉`, its rows and columns

[GT] `gt_part_MP.tex`, rows `def:mp-hyp`, `def:mp-bisimp`, `lem:mp-rows`, `lem:mp-cols`, backing [C] `def:mpfree`,
`lem:mprows`, `lem:mpcols`.

**Hypotheses** (`Cover.Hyp`, [C] `def:mptri` (3)): an orbit-invariant vertex order `ord` injective on every maximal
simplex ((IO)), finite vertex sets, dimension `≤ 2` (at most three vertices per maximal simplex).  (PD) holds by
construction: the simplices of `Y` are the vertex strings contained in a translate of a maximal simplex.

**Objects.**  `YS s`: the ordered `s`-simplices of `Y` (strictly `ord`-increasing vertex strings in some `g ι(σ)`); `FS s n`:
the pairs `(x; (g_0,σ_0), …, (g_n,σ_n))` with `x ⊆ ⋂ g_i ι(σ_i)` ([C] `def:mpfree`'s `𝔉_{n,s}`).  Horizontal faces `dh`
delete a label, vertical faces `dv` delete a vertex of `x`; `aug : FS s 0 → YS s` and `toF' : FS 0 n → 𝔉'_n` are the
augmentations.

**Complexes** (`BCx`, on bases): `rowCx s` the augmented row `k[𝔉_{•,s}] → k[Y_s]`; `fCx` the augmented
`k[𝔉'_•] → k`; `yCx` the augmented ordered chain complex `k[Y_•] → k` of `Y`.

* `rowContr s` (**L4**, [C] `lem:mprows`): the row is contracted by prepending a fixed label `o_x ∈ O(x)` (the extra
  degeneracy of the chaotic nerve on `O(x)`), so `rowCx s` is acyclic, over any ring, non-equivariantly.
* `colContr` (**L5**, [C] `lem:mpcols`): the column `k[𝔉_{n,•}] → k[𝔉'_n]` is contracted by the cone on the
  `ord`-least common vertex `v*(t)` of the string `t` (`vstar`), non-equivariantly.
-/

namespace GL2.MP

namespace Cover

open Finsupp

variable {G : Type} [Group G] (C : Cover.{0} G)

/-- **The hypotheses (IO), finiteness, `dim ≤ 2`** of [C] `def:mptri` (3) on a cover datum. -/
structure Hyp where
  ord : C.P → ℕ
  ord_smul : ∀ (g : G) (p : C.P), ord (g • p) = ord p
  fin : ∀ σ, (C.vtx σ).Finite
  inj : ∀ σ, Set.InjOn ord (C.vtx σ)
  dim : ∀ σ, (C.vtx σ).ncard ≤ 3

variable {C}

/-- `v ∈ g ι(σ)` for the label `o = (g, σ)`. -/
def InS (o : G × C.T) (v : C.P) : Prop := o.1⁻¹ • v ∈ C.vtx o.2

variable (H : C.Hyp)

/-- **The ordered `s`-simplices of `Y`**: strictly `ord`-increasing vertex strings inside one `g ι(σ)`. -/
abbrev YS (s : ℕ) := {x : Fin (s + 1) → C.P // StrictMono (H.ord ∘ x) ∧ ∃ o : G × C.T, ∀ j, InS o (x j)}

/-- **The free bisimplicial set** `𝔉_{n,s}`: `x ∈ Y_s` and labels `(g_i, σ_i)` with `x ⊆ ⋂ g_i ι(σ_i)`. -/
abbrev FS (s n : ℕ) :=
  {p : (Fin (s + 1) → C.P) × (Fin (n + 1) → G × C.T) // StrictMono (H.ord ∘ p.1) ∧ ∀ i j, InS (p.2 i) (p.1 j)}

variable {H}

/-! ### Faces and augmentations -/

lemma strictMono_del {s : ℕ} {x : Fin (s + 2) → C.P} (hx : StrictMono (H.ord ∘ x)) (j : Fin (s + 2)) :
    StrictMono (H.ord ∘ GL2.MP.del j x) :=
  hx.comp (Fin.strictMono_succAbove j)

/-- The vertical face of an ordered simplex. -/
def YS.del {s : ℕ} (j : Fin (s + 2)) (x : YS H (s + 1)) : YS H s :=
  ⟨GL2.MP.del j x.1, strictMono_del x.2.1 j, x.2.2.elim fun o ho => ⟨o, fun m => ho _⟩⟩

/-- The horizontal face of `𝔉`: delete a label. -/
def FS.dh {s n : ℕ} (i : Fin (n + 2)) (p : FS H s (n + 1)) : FS H s n :=
  ⟨(p.1.1, GL2.MP.del i p.1.2), p.2.1, fun m j => p.2.2 _ j⟩

/-- The vertical face of `𝔉`: delete a vertex of `x`. -/
def FS.dv {s n : ℕ} (j : Fin (s + 2)) (p : FS H (s + 1) n) : FS H s n :=
  ⟨(GL2.MP.del j p.1.1, p.1.2), strictMono_del p.2.1 j, fun i m => p.2.2 i _⟩

/-- The simplex `x` of a point of `𝔉`. -/
def FS.base {s n : ℕ} (p : FS H s n) : YS H s := ⟨p.1.1, p.2.1, ⟨p.1.2 0, fun j => p.2.2 0 j⟩⟩

/-- The labels of a point of `𝔉_{n,0}`, a point of `𝔉'_n` (`x` is a common vertex). -/
def FS.toF' {n : ℕ} (p : FS H 0 n) : C.F' n := ⟨p.1.2, ⟨p.1.1 0, fun i => p.2.2 i 0⟩⟩

lemma FS.base_dh {s n : ℕ} (i : Fin (n + 2)) (p : FS H s (n + 1)) : (FS.dh i p).base = p.base := rfl

lemma FS.toF'_dh {n : ℕ} (i : Fin (n + 2)) (p : FS H 0 (n + 1)) : (FS.dh i p).toF' = F'.del i p.toF' := rfl

lemma FS.base_dv {s n : ℕ} (j : Fin (s + 2)) (p : FS H (s + 1) n) : (FS.dv j p).base = YS.del j p.base := rfl

lemma FS.dh_dv {s n : ℕ} (i : Fin (n + 2)) (j : Fin (s + 2)) (p : FS H (s + 1) (n + 1)) :
    FS.dh i (FS.dv j p) = FS.dv j (FS.dh i p) := rfl

lemma FS.toF'_dv {n : ℕ} (j : Fin 2) (p : FS H 1 n) : (FS.dv j p).toF' = (FS.dv 0 p).toF' := rfl

/-! ### The three augmented complexes -/

variable (k : Type) [CommRing k]

/-- The alternating deletion sum of a basis element. -/
noncomputable def altSum {B : Type} {m : ℕ} (f : Fin (m + 2) → B) : B →₀ k :=
  ∑ i : Fin (m + 2), single (f i) ((-1 : k) ^ (i : ℕ))

lemma lc_altSum {B B' : Type} {m : ℕ} (φ : B → (B' →₀ k)) (f : Fin (m + 2) → B) :
    Finsupp.linearCombination k φ (altSum k f) = ∑ i : Fin (m + 2), ((-1 : k) ^ (i : ℕ)) • φ (f i) := by
  simp [altSum, map_sum, Finsupp.linearCombination_single]

lemma altSum_two_const {B : Type} (b : B) : altSum k (m := 0) (fun _ => b) = 0 := by
  simp [altSum, Fin.sum_univ_two]

variable (H)

/-! The boundaries are first defined on concretely typed bases (so that rewriting works), then assembled by degree. -/

/-- The row augmentation `(x; o) ↦ x`. -/
noncomputable def rowBd0 {s : ℕ} (p : FS H s 0) : YS H s →₀ k := single p.base 1

/-- The row boundary: alternating deletion of labels. -/
noncomputable def rowBdS {s n : ℕ} (p : FS H s (n + 1)) : FS H s n →₀ k := altSum k (fun i => FS.dh i p)

theorem rowBdd0 {s : ℕ} (p : FS H s 1) :
    Finsupp.linearCombination k (rowBd0 H k) (rowBdS H k p) = 0 := by
  rw [rowBdS, lc_altSum]
  simp [rowBd0, FS.base_dh, Fin.sum_univ_two]

theorem rowBddS {s n : ℕ} (p : FS H s (n + 2)) :
    Finsupp.linearCombination k (rowBdS H k) (rowBdS H k p) = 0 := by
  classical
  rw [rowBdS, lc_altSum]
  let Φ : (Fin (n + 1) → G × C.T) → (FS H s n →₀ k) := fun g =>
    if h : StrictMono (H.ord ∘ p.1.1) ∧ ∀ i j, InS (g i) (p.1.1 j) then single ⟨(p.1.1, g), h⟩ 1 else 0
  have e : ∀ (i : Fin (n + 3)), rowBdS H k (FS.dh i p) =
      ∑ j : Fin (n + 2), ((-1 : k) ^ (j : ℕ)) • Φ (GL2.MP.del j (GL2.MP.del i p.1.2)) := fun i => by
    simp only [rowBdS, altSum]
    refine Finset.sum_congr rfl fun j _ => ?_
    simp only [Φ]
    split_ifs with h
    · rw [smul_single, smul_eq_mul, mul_one]; rfl
    · exact absurd (FS.dh j (FS.dh i p)).2 h
  simp_rw [e]
  exact alt_sum_sq_k p.1.2 Φ

/-- The basis of the augmented row `s`: `Y_s` in degree `0`, `𝔉_{n,s}` in degree `n + 1`. -/
def rowB (s : ℕ) : ℕ → Type
  | 0 => YS H s
  | n + 1 => FS H s n

/-- The boundary of the augmented row. -/
noncomputable def rowBd (s : ℕ) : ∀ m, rowB H s (m + 1) → (rowB H s m →₀ k)
  | 0 => rowBd0 H k
  | _ + 1 => rowBdS H k

/-- **The augmented row** `k[𝔉_{•,s}] → k[Y_s]` ([C] `lem:mprows`). -/
noncomputable def rowCx (s : ℕ) : BCx k where
  B := rowB H s
  bd := rowBd H k s
  bdd m p := by
    rcases m with _ | n
    · exact rowBdd0 H k (s := s) p
    · exact rowBddS H k (s := s) (n := n) p

/-- The augmentation `𝔉'_0 → pt`. -/
noncomputable def fBd0 (_ : C.F' 0) : Unit →₀ k := single () 1

/-- The boundary of `𝔉'`. -/
noncomputable def fBdS {n : ℕ} (x : C.F' (n + 1)) : C.F' n →₀ k := altSum k (fun i => F'.del i x)

theorem fBdd0 (x : C.F' 1) : Finsupp.linearCombination k (fBd0 k) (fBdS k x) = 0 := by
  rw [fBdS, lc_altSum]
  simp [fBd0, Fin.sum_univ_two]

theorem fBddS {n : ℕ} (x : C.F' (n + 2)) : Finsupp.linearCombination k (fBdS k) (fBdS k x) = 0 := by
  classical
  rw [fBdS, lc_altSum]
  let Φ : C.Lab n → (C.F' n →₀ k) := fun g => if h : C.Cov g then single ⟨g, h⟩ 1 else 0
  have e : ∀ (i : Fin (n + 3)), fBdS k (F'.del i x) =
      ∑ j : Fin (n + 2), ((-1 : k) ^ (j : ℕ)) • Φ (GL2.MP.del j (GL2.MP.del i x.1)) := fun i => by
    simp only [fBdS, altSum]
    refine Finset.sum_congr rfl fun j _ => ?_
    simp only [Φ]
    split_ifs with h
    · show _ = ((-1 : k) ^ (j : ℕ)) • single (F'.del j (F'.del i x)) 1
      rw [smul_single, smul_eq_mul, mul_one]
    · exact absurd ((x.2.del i).del j) h
  simp_rw [e]
  exact alt_sum_sq_k x.1 Φ

variable (C) in
/-- The basis of the augmented `𝔉'`-complex: a point in degree `0`, `𝔉'_n` in degree `n + 1`. -/
def fB : ℕ → Type
  | 0 => Unit
  | n + 1 => C.F' n

/-- The boundary of the augmented `𝔉'`-complex. -/
noncomputable def fBd : ∀ m, fB C (m + 1) → (fB C m →₀ k)
  | 0 => fBd0 k
  | _ + 1 => fBdS k

variable (C) in
/-- **The augmented labelled Čech complex** `k[𝔉'_•] → k`. -/
noncomputable def fCx : BCx k where
  B := fB C
  bd := fBd k
  bdd m x := by
    rcases m with _ | n
    · exact fBdd0 k x
    · exact fBddS k (n := n) x

/-- The augmentation of `Y`. -/
noncomputable def yBd0 (_ : YS H 0) : Unit →₀ k := single () 1

/-- The ordered boundary of `Y`. -/
noncomputable def yBdS {s : ℕ} (x : YS H (s + 1)) : YS H s →₀ k := altSum k (fun j => YS.del j x)

theorem yBdd0 (x : YS H 1) : Finsupp.linearCombination k (yBd0 H k) (yBdS H k x) = 0 := by
  rw [yBdS, lc_altSum]
  simp [yBd0, Fin.sum_univ_two]

theorem yBddS {s : ℕ} (x : YS H (s + 2)) : Finsupp.linearCombination k (yBdS H k) (yBdS H k x) = 0 := by
  classical
  rw [yBdS, lc_altSum]
  let Φ : (Fin (s + 1) → C.P) → (YS H s →₀ k) := fun g =>
    if h : StrictMono (H.ord ∘ g) ∧ ∃ o : G × C.T, ∀ j, InS o (g j) then single ⟨g, h⟩ 1 else 0
  have e : ∀ (i : Fin (s + 3)), yBdS H k (YS.del i x) =
      ∑ j : Fin (s + 2), ((-1 : k) ^ (j : ℕ)) • Φ (GL2.MP.del j (GL2.MP.del i x.1)) := fun i => by
    simp only [yBdS, altSum]
    refine Finset.sum_congr rfl fun j _ => ?_
    simp only [Φ]
    split_ifs with h
    · rw [smul_single, smul_eq_mul, mul_one]; rfl
    · exact absurd (YS.del j (YS.del i x)).2 h
  simp_rw [e]
  exact alt_sum_sq_k x.1 Φ

/-- The basis of the augmented ordered chain complex of `Y`. -/
def yB : ℕ → Type
  | 0 => Unit
  | s + 1 => YS H s

/-- The boundary of the augmented ordered chain complex of `Y`. -/
noncomputable def yBd : ∀ m, yB H (m + 1) → (yB H m →₀ k)
  | 0 => yBd0 H k
  | _ + 1 => yBdS H k

/-- **The augmented ordered chain complex** `k[Y_•] → k` of `Y` ((AC) is its acyclicity). -/
noncomputable def yCx : BCx k where
  B := yB H
  bd := yBd H k
  bdd m x := by
    rcases m with _ | s
    · exact yBdd0 H k x
    · exact yBddS H k (s := s) x

/-! ### L4: the rows are contractible ([C] `lem:mprows`) -/

variable {H}

lemma del_zero_cons {O : Type*} {n : ℕ} (o : O) (f : Fin (n + 1) → O) :
    GL2.MP.del 0 (Fin.cons o f : Fin (n + 2) → O) = f := by
  funext m; simp [GL2.MP.del, Fin.succAbove_zero]

lemma del_succ_cons {O : Type*} {n : ℕ} (o : O) (f : Fin (n + 2) → O) (i : Fin (n + 2)) :
    GL2.MP.del i.succ (Fin.cons o f : Fin (n + 3) → O) = Fin.cons o (GL2.MP.del i f) := by
  funext m
  cases m using Fin.cases with
  | zero => simp [GL2.MP.del]
  | succ m => simp [GL2.MP.del, Fin.succ_succAbove_succ]

lemma del_one_cons {O : Type*} (o : O) (f : Fin 1 → O) : GL2.MP.del 1 (Fin.cons o f : Fin 2 → O) = fun _ => o := by
  funext m; fin_cases m; rfl

/-- The chosen label `o_x ∈ O(x)` of an ordered simplex. -/
noncomputable def ox {s : ℕ} (x : YS H s) : G × C.T := Classical.choose x.2.2

lemma ox_spec {s : ℕ} (x : YS H s) (j : Fin (s + 1)) : InS (ox x) (x.1 j) := Classical.choose_spec x.2.2 j

/-- `(x; o_x)`. -/
noncomputable def rowPt0 {s : ℕ} (x : YS H s) : FS H s 0 := ⟨(x.1, fun _ => ox x), x.2.1, fun _ j => ox_spec x j⟩

/-- `(x; o_x, (g_0, σ_0), …)`: prepend the chosen label (the extra degeneracy). -/
noncomputable def rowPtS {s n : ℕ} (p : FS H s n) : FS H s (n + 1) :=
  ⟨(p.1.1, Fin.cons (ox p.base) p.1.2), p.2.1, fun i j => by
    cases i using Fin.cases with
    | zero => exact ox_spec p.base j
    | succ i => exact p.2.2 i j⟩

lemma dh_zero_rowPtS {s n : ℕ} (p : FS H s n) : FS.dh 0 (rowPtS p) = p :=
  Subtype.ext (Prod.ext rfl (del_zero_cons (ox p.base) p.1.2))

lemma dh_succ_rowPtS {s n : ℕ} (p : FS H s (n + 1)) (i : Fin (n + 2)) :
    FS.dh i.succ (rowPtS p) = rowPtS (FS.dh i p) :=
  Subtype.ext (Prod.ext rfl (del_succ_cons (ox p.base) p.1.2 i))

lemma dh_one_rowPtS {s : ℕ} (p : FS H s 0) : FS.dh 1 (rowPtS p) = rowPt0 p.base :=
  Subtype.ext (Prod.ext rfl (del_one_cons (ox p.base) p.1.2))

variable (H)

/-- **L4**: the contraction of the augmented row `s` by prepending `o_x` ([C] `lem:mprows`). -/
noncomputable def rowContr (s : ℕ) : BContr k (rowCx H k s) where
  hb m := match m with
    | 0 => fun x => single (rowPt0 (H := H) x) 1
    | _ + 1 => fun p => single (rowPtS (H := H) p) 1
  h0 x := by
    change YS H s at x
    show Finsupp.linearCombination k (rowBd0 H k) (single (rowPt0 x) 1) = single x 1
    rw [Finsupp.linearCombination_single, one_smul]; rfl
  hs m b := by
    rcases m with _ | n
    · change FS H s 0 at b
      show Finsupp.linearCombination k (rowBdS H k) (single (rowPtS b) 1) +
        Finsupp.linearCombination k (fun x : YS H s => single (rowPt0 (H := H) x) (1 : k)) (single b.base 1) =
          single b 1
      rw [Finsupp.linearCombination_single, one_smul, Finsupp.linearCombination_single, one_smul]
      show altSum k (fun i => FS.dh i (rowPtS b)) + _ = _
      rw [altSum, Fin.sum_univ_two, dh_zero_rowPtS, dh_one_rowPtS]
      simp only [Fin.val_zero, pow_zero, Fin.val_one, pow_one]
      rw [add_assoc, ← single_add, neg_add_cancel, single_zero, add_zero]
    · change FS H s (n + 1) at b
      show Finsupp.linearCombination k (rowBdS H k) (single (rowPtS b) 1) +
        Finsupp.linearCombination k (fun p : FS H s n => single (rowPtS (H := H) p) (1 : k))
          (altSum k (fun i => FS.dh i b)) = single b 1
      rw [Finsupp.linearCombination_single, one_smul, lc_altSum]
      show altSum k (fun i => FS.dh i (rowPtS b)) + _ = _
      rw [altSum, Fin.sum_univ_succ, dh_zero_rowPtS]
      simp only [Fin.val_zero, pow_zero, Fin.val_succ, dh_succ_rowPtS, smul_single, smul_eq_mul, mul_one]
      rw [add_assoc, ← Finset.sum_add_distrib]
      conv_rhs => rw [← add_zero (single b 1)]
      congr 1
      refine Finset.sum_eq_zero fun i _ => ?_
      rw [← single_add, pow_succ, mul_neg_one, neg_add_cancel, single_zero]

/-- **L4** ([C] `lem:mprows`): the augmented row `k[𝔉_{•,s}] → k[Y_s]` is acyclic, over any ring. -/
theorem rowCx_acyclic (s : ℕ) : (rowCx H k s).toCx.Acyclic := (rowContr H k s).acyclic

/-! ### L5: the columns are contractible ([C] `lem:mpcols`) -/

variable {H}

/-- The common vertices `⋂ g_i ι(σ_i)` of a label string. -/
def Vf {n : ℕ} (f : C.Lab n) : Set C.P := {v | ∀ i, InS (f i) v}

lemma Vf_nonempty {n : ℕ} (f : C.F' n) : (Vf f.1).Nonempty := f.2

lemma Vf_injOn {n : ℕ} (f : C.Lab n) : Set.InjOn H.ord (Vf f) := by
  intro v hv w hw e
  have h1 := H.inj (f 0).2 (hv 0) (hw 0) (by simp only [H.ord_smul, e])
  exact smul_left_cancel _ h1

/-- **The cone vertex** `v*(t)`: the `ord`-least common vertex of the string `t`. -/
noncomputable def vstar {n : ℕ} (f : C.F' n) : C.P := Function.argminOn H.ord (Vf f.1) (Vf_nonempty f)

lemma vstar_mem {n : ℕ} (f : C.F' n) : vstar (H := H) f ∈ Vf f.1 := Function.argminOn_mem _ _ _

lemma vstar_le {n : ℕ} (f : C.F' n) {v : C.P} (hv : v ∈ Vf f.1) : H.ord (vstar (H := H) f) ≤ H.ord v :=
  Function.argminOn_le H.ord (Vf f.1) hv

/-- The labels of a point of `𝔉_{n,s}`, a point of `𝔉'_n`. -/
def FS.lab {s n : ℕ} (p : FS H s n) : C.F' n := ⟨p.1.2, ⟨p.1.1 0, fun i => p.2.2 i 0⟩⟩

lemma FS.mem_Vf {s n : ℕ} (p : FS H s n) (j : Fin (s + 1)) : p.1.1 j ∈ Vf p.1.2 := fun i => p.2.2 i j

/-- `v*` of a point of `𝔉`. -/
noncomputable def FS.vs {s n : ℕ} (p : FS H s n) : C.P := vstar (H := H) p.lab

lemma FS.vs_dv {s n : ℕ} (j : Fin (s + 2)) (p : FS H (s + 1) n) : (FS.dv j p).vs = p.vs := rfl

/-- If the first vertex of `x` is not `v*`, no vertex is, and `v*` comes strictly first. -/
lemma FS.ne_vs {s n : ℕ} (p : FS H s n) (h : p.1.1 0 ≠ p.vs) :
    (∀ j, p.1.1 j ≠ p.vs) ∧ H.ord p.vs < H.ord (p.1.1 0) := by
  have hle : H.ord p.vs ≤ H.ord (p.1.1 0) := vstar_le p.lab (p.mem_Vf 0)
  have hne : H.ord p.vs ≠ H.ord (p.1.1 0) := fun e =>
    h (Vf_injOn p.1.2 (p.mem_Vf 0) (vstar_mem p.lab) e.symm)
  refine ⟨fun j hj => ?_, lt_of_le_of_ne hle hne⟩
  have h0j : H.ord (p.1.1 0) ≤ H.ord (p.1.1 j) := p.2.1.monotone (Fin.zero_le j)
  rw [hj] at h0j
  exact hne (le_antisymm hle h0j)

lemma strictMono_cons_of_lt {s : ℕ} {x : Fin (s + 1) → C.P} (hx : StrictMono (H.ord ∘ x)) {v : C.P}
    (hv : H.ord v < H.ord (x 0)) : StrictMono (H.ord ∘ (Fin.cons v x : Fin (s + 2) → C.P)) := by
  rw [Fin.strictMono_iff_lt_succ]
  intro i
  cases i using Fin.cases with
  | zero => simpa using hv
  | succ i =>
    simp only [Function.comp, ← Fin.succ_castSucc, Fin.cons_succ]
    exact hx Fin.castSucc_lt_succ

/-- `(v* x; t)`: the cone on `v*`. -/
noncomputable def FS.coneUp {s n : ℕ} (p : FS H s n) (h : p.1.1 0 ≠ p.vs) : FS H (s + 1) n :=
  ⟨(Fin.cons p.vs p.1.1, p.1.2), strictMono_cons_of_lt p.2.1 (p.ne_vs h).2, fun i j => by
    cases j using Fin.cases with
    | zero => exact vstar_mem p.lab i
    | succ j => exact p.2.2 i j⟩

lemma FS.vs_coneUp {s n : ℕ} (p : FS H s n) (h : p.1.1 0 ≠ p.vs) : (p.coneUp h).vs = p.vs := rfl

lemma FS.dv_zero_coneUp {s n : ℕ} (p : FS H s n) (h : p.1.1 0 ≠ p.vs) : FS.dv 0 (p.coneUp h) = p :=
  Subtype.ext (Prod.ext (del_zero_cons p.vs p.1.1) rfl)

lemma FS.dv_succ_coneUp {s n : ℕ} (p : FS H (s + 1) n) (h : p.1.1 0 ≠ p.vs) (j : Fin (s + 2)) :
    FS.dv j.succ (p.coneUp h) = (FS.dv j p).coneUp (fun e => (p.ne_vs h).1 _ e) :=
  Subtype.ext (Prod.ext (del_succ_cons p.vs p.1.1 j) rfl)

/-- The cone contraction on bases: `0` if `x` starts with `v*`, else `(v* x; t)`. -/
noncomputable def colH {s n : ℕ} (p : FS H s n) : FS H (s + 1) n →₀ k := by
  classical
  exact if h : p.1.1 0 = p.vs then 0 else single (p.coneUp h) 1

/-- The vertical boundary `∑_j (-1)^j (d_j x; t)`. -/
noncomputable def vdel {s n : ℕ} (p : FS H (s + 1) n) : FS H s n →₀ k := altSum k (fun j => FS.dv j p)

lemma colH_of_eq {s n : ℕ} {p : FS H s n} (h : p.1.1 0 = p.vs) : colH k p = 0 := by
  simp [colH, h]

lemma colH_of_ne {s n : ℕ} {p : FS H s n} (h : p.1.1 0 ≠ p.vs) : colH k p = single (p.coneUp h) 1 := by
  simp [colH, h]

/-- **The cone identity** `d h + h d = id` on `𝔉_{n, s+1}` ([C] `lem:mpcols`, the cone on `v*(t)`). -/
theorem cone_identity {s n : ℕ} (p : FS H (s + 1) n) :
    Finsupp.linearCombination k (vdel k) (colH k p) + Finsupp.linearCombination k (colH k) (vdel k p) =
      single p 1 := by
  by_cases h : p.1.1 0 = p.vs
  · rw [colH_of_eq k h, map_zero, zero_add, vdel, lc_altSum, Fin.sum_univ_succ]
    have hz : ∀ j : Fin (s + 1), colH k (FS.dv j.succ p) = 0 := fun j => colH_of_eq k (by
      show p.1.1 (j.succ.succAbove 0) = (FS.dv j.succ p).vs
      rw [Fin.succ_succAbove_zero, FS.vs_dv]; exact h)
    have h0 : (FS.dv 0 p).1.1 0 ≠ (FS.dv 0 p).vs := by
      show p.1.1 ((0 : Fin (s + 2)).succAbove 0) ≠ p.vs
      rw [Fin.succAbove_zero, ← h]
      intro e
      have := p.2.1 (show (0 : Fin (s + 2)) < Fin.succ 0 from Fin.succ_pos 0)
      simp only [Function.comp, e] at this
      exact lt_irrefl _ this
    rw [colH_of_ne k h0]
    simp only [hz, smul_zero, Finset.sum_const_zero, add_zero, Fin.val_zero, pow_zero, one_smul]
    congr 1
    apply Subtype.ext
    refine Prod.ext ?_ rfl
    show Fin.cons (FS.dv 0 p).vs (GL2.MP.del 0 p.1.1) = p.1.1
    rw [FS.vs_dv, ← h]
    funext m
    cases m using Fin.cases with
    | zero => rfl
    | succ m => simp [GL2.MP.del, Fin.succAbove_zero]
  · have hne := (p.ne_vs h).1
    rw [colH_of_ne k h, Finsupp.linearCombination_single, one_smul, vdel, altSum, Fin.sum_univ_succ, FS.dv_zero_coneUp,
      vdel, lc_altSum]
    have hj : ∀ j : Fin (s + 2), colH k (FS.dv j p) = single ((FS.dv j p).coneUp (fun e => hne _ e)) 1 :=
      fun j => colH_of_ne k _
    simp only [hj, FS.dv_succ_coneUp, Fin.val_zero, pow_zero, Fin.val_succ, smul_single, smul_eq_mul, mul_one]
    rw [add_assoc, ← Finset.sum_add_distrib]
    conv_rhs => rw [← add_zero (single p 1)]
    congr 1
    refine Finset.sum_eq_zero fun j _ => ?_
    rw [← single_add, pow_succ, mul_neg_one, neg_add_cancel, single_zero]

end Cover

end GL2.MP
