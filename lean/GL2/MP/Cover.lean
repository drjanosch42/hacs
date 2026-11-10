import GL2.MP.Strings

/-!
# MP-L1, rows L1 and L2 (slot 22): the choice-free MP set and its twisted cochain complex

[GT] `gt_part_MP.tex`, rows `def:mp-cover`, `def:mp-set`, `lem:mp-face-face`, `lem:mp-set-gamma`, `lem:mp-set-finite`,
`def:mp-cochains`, `thm:mp-dd`, `lem:mp-cochains-natural`, backing [C] `def:mpset` and [C] `thm:mpcomparison` (1), (3).

**The datum** (`Cover`). A group `G` acting on a type `P` (the vertices of `Y`), a type `T` of representatives of the
maximal simplices and, for each `σ : T`, the vertex set `vtx σ ⊆ P` of `ι(σ)`.  Under (SC) ([C] `def:mptri` (3)) a
finite intersection of closed simplices of `Y` is non-empty iff the simplices have a common vertex ([C] `rmk:mpwhat` (1)),
which is how non-emptiness is expressed here.  No finiteness, adaptedness or effectivity is assumed unless stated.

**Labels.** `Lab n = Fin (n+1) → G × T`; `(g_i, σ_i)` names the maximal simplex `g_i ι(σ_i)`; `Cov x` says
`⋂ g_i ι(σ_i) ≠ ∅` (a common vertex `p`, i.e. `g_i⁻¹ p ∈ ι(σ_i)` for every `i`).  The **labelled Čech nerve** is
`F' n = {x // Cov x}` ([C] `def:mpfree`), with `G` acting on every label `g_i` (`smulLab`).

**The MP set** (`MPSet`, [C] `def:mpset`) is taken in the normal form `g_0 = 1` of [C] `thm:mpcomparison` (1): the
`(σ_i; γ_i)` of [C] are `γ_i = g_{i-1}⁻¹ g_i` (`gam`), conversely `g_j = γ_1 ⋯ γ_j` (`Fin.partialProd`), and
`strEquiv` identifies `MPSet n` with [C]'s strings `{(σ; γ) : ∃ p, ∀ j, (γ_1⋯γ_j)⁻¹ p ∈ ι(σ_j)}` on the nose.  The face
`d_i` deletes the label `i` and re-normalises (`face`): `d_0` drops `(σ_0, γ_1)` (`face_zero_str`), `d_i` for `i ≥ 1` is
the plain deletion (`face_succ_val`), which in `γ`-coordinates merges `γ_i, γ_{i+1}` and drops `σ_i`; the degeneracy
repeats a label (`dgn`).  `face_face` is the simplicial identity, `face_dgn_*` the degeneracy identities.

**The twisted cochains** (`Coch`, `dC`, [C] `def:mpset`): `C^n(S; A) = (MPSet n → A)` for `A : Rep k G`,
`(df)(s) = γ_1 · f(d_0 s) + ∑_{i ≥ 1} (-1)^i f(d_i s)` (unnormalised).  `d ∘ d = 0` (`dC_dC`) is proved through the
**equivariant cochains** `ECoch n = {F : F' n → A | F(g x) = g F(x)}` with the untwisted alternating differential
(`dE`, `d ∘ d = 0` by `alt_sum_sq`) and the **on-the-nose identification** `ECoch n ≃ C^n(S; A)` (`cochEquiv`,
restriction to the normal forms; [C] `thm:mpcomparison` (3)) intertwining `dE` and `dC` (`cochEquiv_dE`): the twist `γ_1`
is the re-normalisation `d_0 x_s = γ_1 · x_{d_0 s}`.  The cochain complex is `mpComplex C A : CochainComplex (ModuleCat k) ℕ`,
natural in `A` (`dC_natural`).
-/

namespace GL2.MP

universe u

/-- **An MP cover datum**: `G` acting on the vertices `P` of `Y`, the maximal-simplex representatives `T`, their vertex
sets. -/
structure Cover (G : Type u) [Group G] where
  P : Type u
  [act : MulAction G P]
  T : Type u
  vtx : T → Set P

attribute [instance] Cover.act

namespace Cover

variable {G : Type u} [Group G] (C : Cover G)

/-! ### Labels and the labelled Čech nerve -/

/-- A label string `((g_0, σ_0), …, (g_n, σ_n))`. -/
abbrev Lab (n : ℕ) := Fin (n + 1) → G × C.T

/-- `⋂_i g_i ι(σ_i) ≠ ∅`: a common vertex. -/
def Cov {n : ℕ} (x : C.Lab n) : Prop := ∃ p : C.P, ∀ i, (x i).1⁻¹ • p ∈ C.vtx (x i).2

variable {C}

lemma Cov.del {n : ℕ} {x : C.Lab (n + 1)} (h : C.Cov x) (i : Fin (n + 2)) : C.Cov (GL2.MP.del i x) := by
  obtain ⟨p, hp⟩ := h; exact ⟨p, fun m => hp _⟩

variable (C)

/-- `G` acts on every label: `g · (g_i, σ_i) = (g g_i, σ_i)`. -/
def smulLab {n : ℕ} (g : G) (x : C.Lab n) : C.Lab n := fun i => (g * (x i).1, (x i).2)

variable {C}

@[simp] lemma smulLab_apply {n : ℕ} (g : G) (x : C.Lab n) (i : Fin (n + 1)) :
    C.smulLab g x i = (g * (x i).1, (x i).2) := rfl

lemma Cov.smul {n : ℕ} {x : C.Lab n} (h : C.Cov x) (g : G) : C.Cov (C.smulLab g x) := by
  obtain ⟨p, hp⟩ := h
  refine ⟨g • p, fun i => ?_⟩
  simp only [smulLab_apply, mul_inv_rev, mul_smul, inv_smul_smul]
  exact hp i

lemma del_smulLab {n : ℕ} (i : Fin (n + 2)) (g : G) (x : C.Lab (n + 1)) :
    GL2.MP.del i (C.smulLab g x) = C.smulLab g (GL2.MP.del i x) := rfl

lemma smulLab_smulLab {n : ℕ} (g h : G) (x : C.Lab n) : C.smulLab g (C.smulLab h x) = C.smulLab (g * h) x := by
  funext i; simp [mul_assoc]

lemma one_smulLab {n : ℕ} (x : C.Lab n) : C.smulLab 1 x = x := by funext i; simp

variable (C)

/-- The normal form `g_0 = 1`: `nf x = g_0⁻¹ · x`. -/
def nf {n : ℕ} (x : C.Lab n) : C.Lab n := C.smulLab (x 0).1⁻¹ x

variable {C}

@[simp] lemma nf_zero {n : ℕ} (x : C.Lab n) : (C.nf x 0).1 = 1 := by simp [nf]

lemma smulLab_nf {n : ℕ} (x : C.Lab n) : C.smulLab (x 0).1 (C.nf x) = x := by
  rw [nf, smulLab_smulLab, mul_inv_cancel, one_smulLab]

lemma nf_smulLab {n : ℕ} (g : G) (x : C.Lab n) : C.nf (C.smulLab g x) = C.nf x := by
  unfold nf; rw [smulLab_smulLab]; congr 1; simp [mul_inv_rev, mul_assoc]

lemma nf_of_one {n : ℕ} {x : C.Lab n} (h : (x 0).1 = 1) : C.nf x = x := by
  rw [nf, h, inv_one, one_smulLab]

lemma nf_nf {n : ℕ} (x : C.Lab n) : C.nf (C.nf x) = C.nf x := nf_of_one (nf_zero x)

lemma nf_del_nf {n : ℕ} (i : Fin (n + 2)) (x : C.Lab (n + 1)) :
    C.nf (GL2.MP.del i (C.nf x)) = C.nf (GL2.MP.del i x) := by
  show C.nf (GL2.MP.del i (C.smulLab (x 0).1⁻¹ x)) = _
  rw [del_smulLab, nf_smulLab]

variable (C)

/-- **The labelled Čech nerve** `𝔉'_n = {x : ⋂ g_i ι(σ_i) ≠ ∅}` ([C] `def:mpfree`). -/
def F' (n : ℕ) := {x : C.Lab n // C.Cov x}

instance (n : ℕ) : MulAction G (C.F' n) where
  smul g x := ⟨C.smulLab g x.1, x.2.smul g⟩
  one_smul x := Subtype.ext (one_smulLab x.1)
  mul_smul g h x := Subtype.ext (smulLab_smulLab g h x.1).symm

variable {C}

lemma F'.smul_val {n : ℕ} (g : G) (x : C.F' n) : (g • x).1 = C.smulLab g x.1 := rfl

/-- The face of `𝔉'`: delete a label. -/
def F'.del {n : ℕ} (i : Fin (n + 2)) (x : C.F' (n + 1)) : C.F' n := ⟨GL2.MP.del i x.1, x.2.del i⟩

lemma F'.del_smul {n : ℕ} (i : Fin (n + 2)) (g : G) (x : C.F' (n + 1)) : F'.del i (g • x) = g • F'.del i x := rfl

/-- **`G` acts freely on `𝔉'`** (already on the label `g_0`; [C] `thm:mpcomparison` (1)). -/
theorem F'.smul_eq_self {n : ℕ} {g : G} {x : C.F' n} (h : g • x = x) : g = 1 := by
  have h2 : g * (x.1 0).1 = (x.1 0).1 := congrArg (fun y : C.F' n => (y.1 0).1) h
  simpa using h2

variable (C)

/-! ### The MP set -/

/-- **The MP set** `S_n` ([C] `def:mpset`) in the normal form `g_0 = 1` ([C] `thm:mpcomparison` (1)). -/
def MPSet (n : ℕ) := {x : C.Lab n // C.Cov x ∧ (x 0).1 = 1}

variable {C}

/-- The normal form of a point of `𝔉'` ([C] `thm:mpcomparison` (1): `Γ \ 𝔉'_n ≅ S_n`). -/
def F'.norm {n : ℕ} (x : C.F' n) : C.MPSet n := ⟨C.nf x.1, x.2.smul _, nf_zero _⟩

/-- The inclusion of the normal forms. -/
def MPSet.toF' {n : ℕ} (s : C.MPSet n) : C.F' n := ⟨s.1, s.2.1⟩

lemma F'.norm_smul {n : ℕ} (g : G) (x : C.F' n) : F'.norm (g • x) = F'.norm x :=
  Subtype.ext (nf_smulLab g x.1)

lemma MPSet.norm_toF' {n : ℕ} (s : C.MPSet n) : F'.norm s.toF' = s := Subtype.ext (nf_of_one s.2.2)

lemma F'.smul_norm {n : ℕ} (x : C.F' n) : (x.1 0).1 • (F'.norm x).toF' = x := Subtype.ext (smulLab_nf x.1)

lemma F'.inv_smul_eq_norm {n : ℕ} (x : C.F' n) : (x.1 0).1⁻¹ • x = (F'.norm x).toF' := rfl

/-- **`Γ \ 𝔉'_n ≅ S_n`** ([C] `thm:mpcomparison` (1)): two points of `𝔉'` have the same normal form iff they are in one
orbit, and the normal forms are a transversal. -/
theorem F'.norm_eq_iff {n : ℕ} (x y : C.F' n) : F'.norm x = F'.norm y ↔ ∃ g : G, g • x = y := by
  constructor
  · intro h
    refine ⟨(y.1 0).1 * (x.1 0).1⁻¹, ?_⟩
    rw [← smul_smul, F'.inv_smul_eq_norm, h, F'.smul_norm]
  · rintro ⟨g, rfl⟩; exact (F'.norm_smul g x).symm

/-- **The face maps** `d_i : S_{n+1} → S_n`: delete the label `i` and re-normalise. -/
def face {n : ℕ} (i : Fin (n + 2)) (s : C.MPSet (n + 1)) : C.MPSet n :=
  ⟨C.nf (GL2.MP.del i s.1), (s.2.1.del i).smul _, nf_zero _⟩

lemma face_eq_norm {n : ℕ} (i : Fin (n + 2)) (s : C.MPSet (n + 1)) : face i s = F'.norm (F'.del i s.toF') := rfl

/-- `d_i` for `i ≥ 1` is the plain deletion (no re-normalisation). -/
lemma face_succ_val {n : ℕ} (i : Fin (n + 1)) (s : C.MPSet (n + 1)) : (face i.succ s).1 = GL2.MP.del i.succ s.1 := by
  apply nf_of_one
  show (s.1 (i.succ.succAbove 0)).1 = 1
  rw [Fin.succ_succAbove_zero]; exact s.2.2

/-- `d_0 x_s = γ_1 · x_{d_0 s}`: the re-normalisation of the zeroth face, `γ_1 = g_1`. -/
lemma del_zero_toF' {n : ℕ} (s : C.MPSet (n + 1)) :
    F'.del 0 s.toF' = (s.1 1).1 • (face 0 s).toF' := by
  rw [face_eq_norm]
  exact (F'.smul_norm _).symm

/-- **The simplicial identity** `d_i d_{j+1} = d_j d_i` (`i ≤ j`) on the MP set. -/
theorem face_face {n : ℕ} {i j : Fin (n + 2)} (H : i ≤ j) (s : C.MPSet (n + 2)) :
    face i (face j.succ s) = face j (face i.castSucc s) := by
  apply Subtype.ext
  show C.nf (GL2.MP.del i (C.nf (GL2.MP.del j.succ s.1))) = C.nf (GL2.MP.del j (C.nf (GL2.MP.del i.castSucc s.1)))
  rw [nf_del_nf, nf_del_nf, del_del H]

/-- **The degeneracy** `s_i : S_n → S_{n+1}`: repeat the label `i` (in `γ`-coordinates, insert `(σ_i, 1)`). -/
def dgn {n : ℕ} (i : Fin (n + 1)) (s : C.MPSet n) : C.MPSet (n + 1) :=
  ⟨s.1 ∘ i.predAbove, by obtain ⟨p, hp⟩ := s.2.1; exact ⟨p, fun m => hp _⟩, by
    show (s.1 (i.predAbove 0)).1 = 1
    rw [Fin.predAbove_right_zero]
    exact s.2.2⟩

lemma predAbove_succAbove_castSucc {n : ℕ} (i : Fin (n + 1)) (m : Fin (n + 1)) :
    i.predAbove (i.castSucc.succAbove m) = m := Fin.predAbove_succAbove i m

lemma predAbove_succAbove_succ {n : ℕ} (i : Fin (n + 1)) (m : Fin (n + 1)) :
    i.predAbove (i.succ.succAbove m) = m := by
  rcases i with ⟨i, hi⟩; rcases m with ⟨m, hm⟩
  apply Fin.ext
  dsimp [Fin.succAbove, Fin.predAbove]
  split_ifs <;> simp <;> simp at * <;> omega

/-- `d_i s_i = id`. -/
theorem face_dgn_castSucc {n : ℕ} (i : Fin (n + 1)) (s : C.MPSet n) : face i.castSucc (dgn i s) = s := by
  apply Subtype.ext
  show C.nf (s.1 ∘ i.predAbove ∘ i.castSucc.succAbove) = s.1
  have : s.1 ∘ i.predAbove ∘ i.castSucc.succAbove = s.1 := funext fun m => by
    simp only [Function.comp, predAbove_succAbove_castSucc]
  rw [this, nf_of_one s.2.2]

/-- `d_{i+1} s_i = id`. -/
theorem face_dgn_succ {n : ℕ} (i : Fin (n + 1)) (s : C.MPSet n) : face i.succ (dgn i s) = s := by
  apply Subtype.ext
  show C.nf (s.1 ∘ i.predAbove ∘ i.succ.succAbove) = s.1
  have : s.1 ∘ i.predAbove ∘ i.succ.succAbove = s.1 := funext fun m => by
    simp only [Function.comp, predAbove_succAbove_succ]
  rw [this, nf_of_one s.2.2]

/-! ### The `γ`-coordinates of [C] `def:mpset` -/

/-- `γ_{j+1} = g_j⁻¹ g_{j+1}`. -/
def gam {n : ℕ} (x : C.Lab n) : Fin n → G := fun j => (x j.castSucc).1⁻¹ * (x j.succ).1

variable (C)

/-- [C]'s strings `(σ_0, …, σ_n; γ_1, …, γ_n)` with `ι(σ_0) ∩ γ_1 ι(σ_1) ∩ … ∩ γ_1⋯γ_n ι(σ_n) ≠ ∅`. -/
def Str (n : ℕ) :=
  {s : (Fin (n + 1) → C.T) × (Fin n → G) // ∃ p : C.P, ∀ j, (Fin.partialProd s.2 j)⁻¹ • p ∈ C.vtx (s.1 j)}

variable {C}

lemma partialProd_gam {n : ℕ} {x : C.Lab n} (h : (x 0).1 = 1) (j : Fin (n + 1)) :
    Fin.partialProd (gam x) j = (x j).1 := by
  have := congrFun (Fin.partialProd_left_inv (fun i => (x i).1)) j
  simp only [h, Pi.smul_apply, smul_eq_mul, one_mul] at this
  exact this

lemma gam_partialProd {n : ℕ} (γ : Fin n → G) (σ : Fin (n + 1) → C.T) :
    gam (C := C) (fun j => (Fin.partialProd γ j, σ j)) = γ :=
  funext fun j => Fin.partialProd_right_inv γ j

/-- **[C]'s strings are the normal forms** (`strEquiv`): `(σ; γ) ↦ ((γ_1⋯γ_j, σ_j))_j`. -/
def strEquiv (n : ℕ) : C.Str n ≃ C.MPSet n where
  toFun s := ⟨fun j => (Fin.partialProd s.1.2 j, s.1.1 j), s.2, by simp⟩
  invFun x := ⟨(fun j => (x.1 j).2, gam x.1), by
    obtain ⟨p, hp⟩ := x.2.1
    exact ⟨p, fun j => by rw [partialProd_gam x.2.2]; exact hp j⟩⟩
  left_inv s := by
    apply Subtype.ext
    apply Prod.ext
    · rfl
    · exact gam_partialProd s.1.2 s.1.1
  right_inv x := by
    apply Subtype.ext
    funext j
    apply Prod.ext
    · exact partialProd_gam x.2.2 j
    · rfl

/-- **`d_0` drops `(σ_0, γ_1)`** in [C]'s coordinates. -/
theorem face_zero_str {n : ℕ} (s : C.MPSet (n + 1)) :
    (fun j => ((face 0 s).1 j).2) = Fin.tail (fun j => (s.1 j).2) ∧ gam (face 0 s).1 = Fin.tail (gam s.1) := by
  refine ⟨rfl, funext fun j => ?_⟩
  simp only [gam, face, nf, smulLab_apply, del_apply, Fin.tail, Fin.succAbove_zero]
  rw [Fin.succ_castSucc]
  group

/-- **`d_n` drops `(σ_n, γ_n)`** in [C]'s coordinates. -/
theorem face_last_str {n : ℕ} (s : C.MPSet (n + 1)) :
    (fun j => ((face (Fin.last (n + 1)) s).1 j).2) = Fin.init (fun j => (s.1 j).2) ∧
      gam (face (Fin.last (n + 1)) s).1 = Fin.init (gam s.1) := by
  have hv := face_succ_val (Fin.last n) s
  rw [Fin.succ_last] at hv
  refine ⟨funext fun j => ?_, funext fun j => ?_⟩
  · rw [hv]; simp [Fin.init]
  · rw [hv]; simp [gam, Fin.init]

/-- **Finiteness** ([C] `rmk:mpwhat` (1)): if `T` is finite and every arrow set `Ar(σ, σ') = {γ : ι(σ) ∩ γ ι(σ') ≠ ∅}` is
finite, every `S_n` is finite. -/
theorem finite_mpSet [Finite C.T] (hAr : ∀ σ σ' : C.T, Set.Finite {γ : G | ∃ q ∈ C.vtx σ, γ⁻¹ • q ∈ C.vtx σ'})
    (n : ℕ) : Finite (C.MPSet n) := by
  let Ar : C.T → C.T → Set G := fun σ σ' => {γ : G | ∃ q ∈ C.vtx σ, γ⁻¹ • q ∈ C.vtx σ'}
  have : ∀ σ σ', Finite (Ar σ σ') := fun σ σ' => (hAr σ σ').to_subtype
  let φ : C.MPSet n → Σ σ : Fin (n + 1) → C.T, ∀ j : Fin n, Ar (σ j.castSucc) (σ j.succ) := fun s =>
    ⟨fun j => (s.1 j).2, fun j => ⟨gam s.1 j, by
      obtain ⟨p, hp⟩ := s.2.1
      refine ⟨(s.1 j.castSucc).1⁻¹ • p, hp _, ?_⟩
      simp only [gam, mul_inv_rev, inv_inv, mul_smul, smul_inv_smul]
      exact hp _⟩⟩
  refine Finite.of_injective φ fun s t h => ?_
  have h1 := congrArg Sigma.fst h
  have h2 := congrArg (fun z : (Σ σ : Fin (n + 1) → C.T, ∀ j : Fin n, Ar (σ j.castSucc) (σ j.succ)) =>
    fun j => (z.2 j : G)) h
  change (fun j => (s.1 j).2) = (fun j => (t.1 j).2) at h1
  change gam s.1 = gam t.1 at h2
  apply Subtype.ext
  funext j
  refine Prod.ext ?_ (congrFun h1 j)
  rw [← partialProd_gam s.2.2, ← partialProd_gam t.2.2, h2]

end Cover

end GL2.MP
