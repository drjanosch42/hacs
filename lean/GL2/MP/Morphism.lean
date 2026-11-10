import GL2.MP.Natural

/-!
# MP-L3, row L8 (slot 22): morphisms of MP data, the label factor, the induced cochain map

[GT] `gt_part_MP.tex`, rows `def:mp-morph`, `lem:mp-morph-simplicial`, `def:mp-pull`, `thm:mp-pull-cochain`,
`lem:mp-pull-comp`, `thm:mp-untwisted`, backing [C] `def:mpmorphism` and [C] `thm:mpnatural` (1), (4).

**A morphism** (`Morph φ C C'`) of cover data along a group homomorphism `φ : Γ → Γ'` is a **label table**
`σ ↦ (h_σ, σ')` (`lab`) such that a non-empty intersection `⋂ g_i ι(σ_i)` goes to a non-empty intersection
`⋂ φ(g_i) h_{σ_i} ι'(σ'_i)` (`cov`).  This is what [C] `def:mpmorphism` (1)–(2) use of the morphism: for a `φ`-equivariant
simplicial map `f` with `f(ι(σ)) ⊆ h_σ ι'(σ')` (`Morph.ofVtx`), and for a refinement (where it is containment of closed
simplices in `|Y|`), the condition holds.  Only the labels enter the induced maps ([C] `def:mpmorphism` (3)):

* `uF : 𝔉'_n → 𝔉'{}'_n`, `((g_i, σ_i)) ↦ ((φ(g_i) h_{σ_i}, σ'_i))`, `φ`-equivariant (`uF_smul`), commuting with the
  faces (`uF_del`, on the nose) and with the augmentations;
* `uS : S_n → S'_n`, its map on normal forms (`uF_toF'`: `u(x_s) = h_{σ_0} · x_{u(s)}`, the proof of [C]
  `thm:mpnatural` (1)), a map of simplicial sets (`uS_face`), with `γ`-coordinates `h_{σ_{i-1}}⁻¹ φ(γ_i) h_{σ_i}`
  (`uS_gam`);
* **the induced cochain map with the label factor** (`pull`, [C] `def:mpmorphism` (3)):
  `(u^* f)(s) = h_{σ_0} · f(u(s))`, `C^•(S'; A') → C^•(S; φ^* A')`; it is `F ↦ F ∘ uF` on equivariant cochains
  (`cochEquiv_epull`), hence a cochain map (`pull_dC`, `pullMap`), for every ring and every representation;
  compatible with composition when labels compose by `h''_σ = ψ(h_σ) h'_{σ'}` (`Morph.comp`, `pull_comp`).

**The label factor cannot be dropped** ([C] `thm:mpnatural` (4)): the untwisted `f ↦ f ∘ u` (`untw`) differs from a
cochain map by exactly `(φ(γ_1) − h_{σ_0}⁻¹ φ(γ_1) h_{σ_1}) · f(u(d_0 s))` (`untw_defect`); it is a cochain map when all
labels act by one and the same central automorphism (`untw_dC_of_const`), and it is **not** one as soon as one string
has `φ(γ_1) v ≠ (h_{σ_0}⁻¹ φ(γ_1) h_{σ_1}) v` for some `v` (`untw_not_cochain`).  The concrete witness on
`R_ℤ → M₇` (mixed labels `±I`, `n` odd) is `GL2.MP.RZRefine`.
-/

namespace GL2.MP

namespace Cover

open CategoryTheory

variable {G G' G'' : Type} [Group G] [Group G'] [Group G'']

/-- The image `((φ(g_i) h_{σ_i}, σ'_i))_i` of a label string under a label table. -/
def mapLab (φ : G →* G') {C : Cover.{0} G} {C' : Cover.{0} G'} (lab : C.T → G' × C'.T) {n : ℕ} (x : C.Lab n) :
    C'.Lab n :=
  fun i => (φ (x i).1 * (lab (x i).2).1, (lab (x i).2).2)

/-- **A morphism of MP data** ([C] `def:mpmorphism` (1)–(2)): a label table `σ ↦ (h_σ, σ')` carrying non-empty
intersections to non-empty intersections. -/
structure Morph (φ : G →* G') (C : Cover.{0} G) (C' : Cover.{0} G') where
  lab : C.T → G' × C'.T
  cov : ∀ {n : ℕ} (x : C.Lab n), C.Cov x → C'.Cov (mapLab φ lab x)

variable {φ : G →* G'} {ψ : G' →* G''} {C : Cover.{0} G} {C' : Cover.{0} G'} {C'' : Cover.{0} G''}

/-- **A `φ`-equivariant vertex map with `f(ι(σ)) ⊆ h_σ ι'(σ')` is a morphism** ([C] `def:mpmorphism` (1): a simplicial
morphism `(φ, f)` with its label table). -/
def Morph.ofVtx (f : C.P → C'.P) (hf : ∀ (g : G) p, f (g • p) = φ g • f p) (lab : C.T → G' × C'.T)
    (hlab : ∀ σ p, p ∈ C.vtx σ → (lab σ).1⁻¹ • f p ∈ C'.vtx (lab σ).2) : Morph φ C C' where
  lab := lab
  cov x := by
    rintro ⟨p, hp⟩
    refine ⟨f p, fun i => ?_⟩
    have h := hlab _ _ (hp i)
    rw [hf, ← mul_smul, map_inv] at h
    simpa [mapLab, mul_inv_rev] using h

namespace Morph

variable (M : Morph φ C C')

/-! ### The induced map of labelled Čech nerves and of MP sets -/

/-- `u : 𝔉'_n → 𝔉'{}'_n`. -/
def uF {n : ℕ} (x : C.F' n) : C'.F' n := ⟨mapLab φ M.lab x.1, M.cov x.1 x.2⟩

lemma uF_val {n : ℕ} (x : C.F' n) (i : Fin (n + 1)) :
    (M.uF x).1 i = (φ (x.1 i).1 * (M.lab (x.1 i).2).1, (M.lab (x.1 i).2).2) := rfl

/-- **`u` is `φ`-equivariant.** -/
lemma uF_smul {n : ℕ} (g : G) (x : C.F' n) : M.uF (g • x) = φ g • M.uF x := by
  apply Subtype.ext; funext i
  show (φ (g * (x.1 i).1) * (M.lab (x.1 i).2).1, (M.lab (x.1 i).2).2) =
    (φ g * (φ (x.1 i).1 * (M.lab (x.1 i).2).1), (M.lab (x.1 i).2).2)
  rw [map_mul, mul_assoc]

/-- **`u` commutes with the faces** (on the nose). -/
lemma uF_del {n : ℕ} (i : Fin (n + 2)) (x : C.F' (n + 1)) : M.uF (F'.del i x) = F'.del i (M.uF x) := rfl

/-- `u` on the normal forms: `S_n → S'_n`. -/
def uS {n : ℕ} (s : C.MPSet n) : C'.MPSet n := F'.norm (M.uF s.toF')

/-- **`u(x_s) = h_{σ_0} · x_{u(s)}`** (the proof of [C] `thm:mpnatural` (1)). -/
lemma uF_toF' {n : ℕ} (s : C.MPSet n) : M.uF s.toF' = (M.lab (s.1 0).2).1 • (M.uS s).toF' := by
  have h : ((M.uF s.toF').1 0).1 = (M.lab (s.1 0).2).1 := by
    show φ (s.1 0).1 * _ = _
    rw [s.2.2, map_one, one_mul]; rfl
  rw [uS, ← h]
  exact (F'.smul_norm _).symm

lemma norm_uF_smul {n : ℕ} (g : G) (x : C.F' n) : F'.norm (M.uF (g • x)) = F'.norm (M.uF x) := by
  rw [uF_smul, F'.norm_smul]

lemma norm_del_norm {G₀ : Type} [Group G₀] {C₀ : Cover.{0} G₀} {n : ℕ} (i : Fin (n + 2)) (y : C₀.F' (n + 1)) :
    F'.norm (F'.del i (F'.norm y).toF') = F'.norm (F'.del i y) := by
  rw [← F'.inv_smul_eq_norm, F'.del_smul, F'.norm_smul]

/-- **`u` is a map of simplicial sets** on the MP sets. -/
lemma uS_face {n : ℕ} (i : Fin (n + 2)) (s : C.MPSet (n + 1)) : M.uS (face i s) = face i (M.uS s) := by
  rw [face_eq_norm, face_eq_norm, uS, uS, norm_del_norm, ← uF_del]
  have : (face i s).toF' = ((F'.del i s.toF').1 0).1⁻¹ • F'.del i s.toF' := by
    rw [face_eq_norm, F'.inv_smul_eq_norm]
  rw [face_eq_norm] at this
  rw [this, norm_uF_smul]

/-- **The `γ`-coordinates of `u(s)`**: `γ'_{j+1} = h_{σ_j}⁻¹ φ(γ_{j+1}) h_{σ_{j+1}}` ([C] `def:mpmorphism` (3)). -/
lemma uS_gam {n : ℕ} (s : C.MPSet n) (j : Fin n) :
    gam (M.uS s).1 j = (M.lab (s.1 j.castSucc).2).1⁻¹ * φ (gam s.1 j) * (M.lab (s.1 j.succ).2).1 := by
  show (((M.uF s.toF').1 0).1⁻¹ * ((M.uF s.toF').1 j.castSucc).1)⁻¹ * (((M.uF s.toF').1 0).1⁻¹ *
    ((M.uF s.toF').1 j.succ).1) = _
  simp only [uF_val, gam, map_mul, map_inv]
  group
  rfl

lemma uS_sigma {n : ℕ} (s : C.MPSet n) (j : Fin (n + 1)) : ((M.uS s).1 j).2 = (M.lab (s.1 j).2).2 := rfl

/-! ### The induced cochain map, with the label factor -/

variable {k : Type} [CommRing k] (A : Rep.{0} k G')

/-- `F ↦ F ∘ u` on equivariant cochains: `Hom_{Γ'}(k[𝔉'{}'_n], A') → Hom_Γ(k[𝔉'_n], φ^* A')`. -/
def epull (n : ℕ) (F : C'.ECoch A n) : C.ECoch (Rep.res φ A) n :=
  ⟨fun x => F.1 (M.uF x), fun g x => by
    show F.1 (M.uF (g • x)) = A.ρ (φ g) (F.1 (M.uF x))
    rw [uF_smul, F.2]⟩

/-- `F ↦ F ∘ u` commutes with the untwisted differentials (`u` commutes with the faces). -/
lemma epull_dE (n : ℕ) (F : C'.ECoch A n) :
    (M.epull A (n + 1) ⟨C'.dE A n F.1, C'.dE_mem A n F.2⟩).1 = C.dE (Rep.res φ A) n (M.epull A n F).1 := rfl

/-- **The induced cochain map** `(u^* f)(s) = h_{σ_0} · f(u(s))` ([C] `def:mpmorphism` (3), the label factor). -/
def pull (n : ℕ) : C'.Coch A n →ₗ[k] C.Coch (Rep.res φ A) n where
  toFun f s := A.ρ (M.lab (s.1 0).2).1 (f (M.uS s))
  map_add' f g := by funext s; simp
  map_smul' r f := by funext s; simp

lemma pull_apply (n : ℕ) (f : C'.Coch A n) (s : C.MPSet n) :
    M.pull A n f s = A.ρ (M.lab (s.1 0).2).1 (f (M.uS s)) := rfl

/-- **`u^*` is `F ↦ F ∘ u` read on the normal forms** ([C] `thm:mpnatural` (1)). -/
lemma cochEquiv_epull (n : ℕ) (F : C'.ECoch A n) :
    C.cochEquiv (Rep.res φ A) n (M.epull A n F) = M.pull A n (C'.cochEquiv A n F) := by
  funext s
  show F.1 (M.uF s.toF') = A.ρ _ (F.1 (M.uS s).toF')
  rw [uF_toF', F.2]

/-- **`u^*` is a cochain map** ([C] `thm:mpnatural` (1)), for every ring and every representation. -/
theorem pull_dC (n : ℕ) (f : C'.Coch A n) :
    M.pull A (n + 1) (C'.dC A n f) = C.dC (Rep.res φ A) n (M.pull A n f) := by
  obtain ⟨F, rfl⟩ := (C'.cochEquiv A n).surjective f
  rw [← cochEquiv_dE, ← cochEquiv_epull, ← cochEquiv_epull, ← cochEquiv_dE]
  congr 1

/-- **The induced map of MP complexes** `C^•(S'; A') ⟶ C^•(S; φ^* A')`. -/
noncomputable def pullMap : C'.mpComplex A ⟶ C.mpComplex (Rep.res φ A) :=
  CochainComplex.ofHom (fun n => ModuleCat.ofHom (M.pull A n)) fun n => by
    simp only [mpComplex, CochainComplex.of_d]
    ext f : 2
    exact (M.pull_dC A n f).symm

lemma pullMap_f_apply (n : ℕ) (f : C'.Coch A n) : ((M.pullMap A).f n).hom f = M.pull A n f := rfl

/-! ### Composition -/

/-- **Composition of morphisms**: labels compose by `h''_σ = ψ(h_σ) h'_{σ'}`. -/
def comp (M' : Morph ψ C' C'') : Morph (ψ.comp φ) C C'' where
  lab σ := (ψ (M.lab σ).1 * (M'.lab (M.lab σ).2).1, (M'.lab (M.lab σ).2).2)
  cov x hx := by
    have h := M'.cov _ (M.cov x hx)
    convert h using 1
    funext i
    simp only [mapLab, MonoidHom.comp_apply, map_mul, mul_assoc]

lemma comp_uF (M' : Morph ψ C' C'') {n : ℕ} (x : C.F' n) : (M.comp M').uF x = M'.uF (M.uF x) := by
  apply Subtype.ext; funext i
  show ((ψ.comp φ) (x.1 i).1 * (ψ (M.lab (x.1 i).2).1 * _), _) = (ψ (φ (x.1 i).1 * _) * _, _)
  rw [MonoidHom.comp_apply, map_mul, mul_assoc]
  rfl

/-- **`(u' ∘ u)^* = u^* ∘ u'^*`** ([C] `thm:mpnatural` (1), compatibility with composition). -/
theorem pull_comp (M' : Morph ψ C' C'') (A : Rep.{0} k G'') (n : ℕ) (f : C''.Coch A n) :
    (M.comp M').pull A n f = M.pull (Rep.res ψ A) n (M'.pull A n f) := by
  obtain ⟨F, rfl⟩ := (C''.cochEquiv A n).surjective f
  rw [← cochEquiv_epull, ← cochEquiv_epull, ← cochEquiv_epull]
  congr 1
  apply Subtype.ext; funext x
  show F.1 ((M.comp M').uF x) = F.1 (M'.uF (M.uF x))
  rw [comp_uF]

/-! ### The label factor cannot be dropped ([C] `thm:mpnatural` (4)) -/

/-- **The untwisted map** `f ↦ f ∘ u` (no label factor). -/
def untw (n : ℕ) : C'.Coch A n →ₗ[k] C.Coch (Rep.res φ A) n where
  toFun f s := f (M.uS s)
  map_add' _ _ := rfl
  map_smul' _ _ := rfl

/-- **The defect of the untwisted map**: `d(f ∘ u)(s) − (df)(u s) = (φ(γ_1) − h_{σ_0}⁻¹ φ(γ_1) h_{σ_1}) · f(u(d_0 s))`. -/
theorem untw_defect (n : ℕ) (f : C'.Coch A n) (s : C.MPSet (n + 1)) :
    C.dC (Rep.res φ A) n (M.untw A n f) s - M.untw A (n + 1) (C'.dC A n f) s =
      A.ρ (φ (s.1 1).1) (f (M.uS (face 0 s))) -
        A.ρ ((M.lab (s.1 0).2).1⁻¹ * φ (s.1 1).1 * (M.lab (s.1 1).2).1) (f (M.uS (face 0 s))) := by
  have hγ : ((M.uS s).1 1).1 = (M.lab (s.1 0).2).1⁻¹ * φ (s.1 1).1 * (M.lab (s.1 1).2).1 := by
    have := M.uS_gam s 0
    simp only [gam, Fin.castSucc_zero, Fin.succ_zero_eq_one, (M.uS s).2.2, inv_one, one_mul, s.2.2] at this
    exact this
  show (A.ρ (φ (s.1 1).1) (f (M.uS (face 0 s))) + ∑ i : Fin (n + 1), ((-1 : k) ^ ((i : ℕ) + 1)) •
      f (M.uS (face i.succ s))) - (A.ρ (((M.uS s).1 1).1) (f (face 0 (M.uS s))) + ∑ i : Fin (n + 1),
      ((-1 : k) ^ ((i : ℕ) + 1)) • f (face i.succ (M.uS s))) = _
  simp only [← M.uS_face, hγ]
  abel

/-- **The untwisted map is a cochain map when all labels act by one and the same central automorphism** ([C]
`thm:mpnatural` (4), the sufficient condition: all `+1`, all `-1`, or all in `ker ρ'`). -/
theorem untw_dC_of_const (hconst : ∀ σ τ : C.T, A.ρ (M.lab σ).1 = A.ρ (M.lab τ).1)
    (hcent : ∀ (σ : C.T) (g : G'), A.ρ (M.lab σ).1 * A.ρ g = A.ρ g * A.ρ (M.lab σ).1) (n : ℕ) (f : C'.Coch A n) :
    M.untw A (n + 1) (C'.dC A n f) = C.dC (Rep.res φ A) n (M.untw A n f) := by
  funext s
  have h := M.untw_defect A n f s
  have hc : A.ρ ((M.lab (s.1 0).2).1⁻¹ * φ (s.1 1).1 * (M.lab (s.1 1).2).1) = A.ρ (φ (s.1 1).1) := by
    rw [map_mul, map_mul, ← hconst (s.1 0).2 (s.1 1).2, mul_assoc, ← hcent, ← mul_assoc, ← map_mul, inv_mul_cancel,
      map_one, one_mul]
  rw [hc, sub_self, sub_eq_zero] at h
  exact h.symm

/-- **The untwisted map is not a cochain map in general** ([C] `thm:mpnatural` (4)): one string `s` with
`φ(γ_1) v ≠ (h_{σ_0}⁻¹ φ(γ_1) h_{σ_1}) v` for some `v` suffices. -/
theorem untw_not_cochain {n : ℕ} (s : C.MPSet (n + 1)) (v : A)
    (hv : A.ρ (φ (s.1 1).1) v ≠ A.ρ ((M.lab (s.1 0).2).1⁻¹ * φ (s.1 1).1 * (M.lab (s.1 1).2).1) v) :
    ∃ f : C'.Coch A n, M.untw A (n + 1) (C'.dC A n f) ≠ C.dC (Rep.res φ A) n (M.untw A n f) := by
  classical
  refine ⟨fun t => if t = M.uS (face 0 s) then v else 0, fun h => hv ?_⟩
  have hd := M.untw_defect A n (fun t => if t = M.uS (face 0 s) then v else 0) s
  rw [h, sub_self, if_pos rfl] at hd
  exact (sub_eq_zero.1 hd.symm)

end Morph

end Cover

end GL2.MP
