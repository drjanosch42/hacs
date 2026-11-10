import Mathlib

/-!
# G3, layer A: cochain complexes, the mapping cone with [C]'s sign, and its long exact sequence (L16-6)

[GT] chapter `ch:cone`, mirroring [C] v1.16.0-pre.3 `def:pair` (`eq:conediff`), `lem:conerelative`(2),(3),(5)
(`eq:coneles`, `eq:conedims`), `lem:conesign` (`eq:conesign`), `prop:coneambiguity`(1),(2), `cl:les`.

This layer is pure homological algebra: a restriction `r : C → D` of `ℤ`-graded cochain complexes of `R`-modules, its
cone `Cone(r)^q = C^{q+1} ⊕ D^q`, `D_cone(c, c_∂) = (−dc, dc_∂ + rc)`, `ι(c_∂) = (0, c_∂)`, `π(c, c_∂) = c`, and
`H^q_c := H^{q-1}(Cone(r))`.  Layer B (`GL2.Cone.Pair`) instantiates it on the equivariant cochains of the pair.

**Indexing.** `d q : X q → X (q+1)`.  A homotopy-type map of degree `−1` is indexed by its *target*,
`h q : C^{q+1} → D^q`, so that `d h + h d` is typed without casts.  Coboundaries `B q` are the image of `d (q-1)`
transported along `q - 1 + 1 = q` (`CochainCx.castX`).

Everything here is **proved**: the pure computations (`d² = 0` on the cone, `ι`, `π` cochain maps, `eq:conesign`, the
cochain-level ambiguity formula) since L16-6; the long exact sequence (elementwise and as `im = ker` of `π_*`, `r_*`, `ι_*`
on cohomology), the relative comparison, the dimension counts and the realisation over a field since L18-4.
-/

namespace GL2.Cone

universe u v

/-- A `ℤ`-graded cochain complex of `R`-modules. -/
structure CochainCx (R : Type v) [CommRing R] where
  X : ℤ → Type u
  [grp : ∀ q, AddCommGroup (X q)]
  [mod : ∀ q, Module R (X q)]
  d : ∀ q, X q →ₗ[R] X (q + 1)
  d_comp_d : ∀ q, (d (q + 1)).comp (d q) = 0

attribute [instance] CochainCx.grp CochainCx.mod

variable {R : Type v} [CommRing R]

namespace CochainCx

variable (C : CochainCx.{u} R)

/-- Transport along an equality of degrees. -/
def castX {a b : ℤ} (h : a = b) : C.X a ≃ₗ[R] C.X b := by
  subst h; exact LinearEquiv.refl R _

/-- Cocycles `Z^q = ker d^q`. -/
def Z (q : ℤ) : Submodule R (C.X q) := LinearMap.ker (C.d q)

/-- Coboundaries `B^q = d(C^{q-1})`. -/
def B (q : ℤ) : Submodule R (C.X q) := (LinearMap.range (C.d (q - 1))).map (C.castX (sub_add_cancel q 1)).toLinearMap

/-- Cohomology `H^q = Z^q / B^q`. -/
abbrev H (q : ℤ) := C.Z q ⧸ (C.B q).comap (C.Z q).subtype

lemma d_d (q : ℤ) (x : C.X q) : C.d (q + 1) (C.d q x) = 0 := by
  rw [← LinearMap.comp_apply, C.d_comp_d, LinearMap.zero_apply]

lemma mem_Z (q : ℤ) (x : C.X q) : x ∈ C.Z q ↔ C.d q x = 0 := LinearMap.mem_ker

/-- Coboundaries in degree `q`, written from any `p` with `p + 1 = q` (L18-4). -/
lemma mem_B_iff' {p q : ℤ} (h : p + 1 = q) (x : C.X q) :
    x ∈ C.B q ↔ ∃ y : C.X p, C.castX h (C.d p y) = x := by
  have key : ∀ {p p' : ℤ} (hp : p + 1 = q) (hp' : p' + 1 = q),
      (∃ y : C.X p, C.castX hp (C.d p y) = x) ↔ (∃ y : C.X p', C.castX hp' (C.d p' y) = x) := by
    intro p p' hp hp'
    obtain rfl : p = p' := by omega
    rfl
  rw [← key (sub_add_cancel q 1) h]
  simp only [B, Submodule.mem_map, LinearMap.mem_range, LinearEquiv.coe_coe]
  constructor
  · rintro ⟨z, ⟨y, rfl⟩, rfl⟩; exact ⟨y, rfl⟩
  · rintro ⟨y, rfl⟩; exact ⟨_, ⟨y, rfl⟩, rfl⟩

/-- Coboundaries in degree `p + 1` (L18-4). -/
lemma mem_B_succ {p : ℤ} (x : C.X (p + 1)) : x ∈ C.B (p + 1) ↔ ∃ y : C.X p, C.d p y = x :=
  C.mem_B_iff' rfl x

lemma B_le_Z (q : ℤ) : C.B q ≤ C.Z q := by
  intro x hx
  obtain ⟨y, rfl⟩ := (C.mem_B_iff' (sub_add_cancel q 1) x).1 hx
  rw [mem_Z]
  have key : ∀ {a b : ℤ} (h : a = b) (w : C.X a), C.d b (C.castX h w) = C.castX (congrArg (· + 1) h) (C.d a w) := by
    intro a b h w; subst h; rfl
  rw [key, C.d_d, map_zero]

/-- The class of a cocycle vanishes iff it is a coboundary (L18-4). -/
lemma mk_eq_zero_iff (q : ℤ) (z : C.Z q) :
    (Submodule.Quotient.mk z : C.H q) = 0 ↔ (z : C.X q) ∈ C.B q := by
  rw [Submodule.Quotient.mk_eq_zero, Submodule.mem_comap, Submodule.coe_subtype]

end CochainCx

open CochainCx

/-- A degree-`0` graded map `f q : C^q → D^q`. -/
abbrev GMap (C D : CochainCx.{u} R) := ∀ q, C.X q →ₗ[R] D.X q

/-- A degree-`−1` graded map, indexed by its target: `h q : C^{q+1} → D^q`. -/
abbrev GMapNeg (C D : CochainCx.{u} R) := ∀ q, C.X (q + 1) →ₗ[R] D.X q

/-- `f` is a cochain map: `d ∘ f = f ∘ d`. -/
def IsChainMap {C D : CochainCx.{u} R} (f : GMap C D) : Prop :=
  ∀ q, (D.d q).comp (f q) = (f (q + 1)).comp (C.d q)

lemma IsChainMap.apply {C D : CochainCx.{u} R} {f : GMap C D} (hf : IsChainMap f) (q : ℤ) (x : C.X q) :
    D.d q (f q x) = f (q + 1) (C.d q x) := by
  rw [← LinearMap.comp_apply, hf q, LinearMap.comp_apply]

/-- The map induced on cohomology by any cochain map (L18-4; `Hmap` is the case `f = r`). -/
def HmapOf {A B : CochainCx.{u} R} (f : GMap A B) (hf : IsChainMap f) (q : ℤ) : A.H q →ₗ[R] B.H q :=
  Submodule.mapQ _ _ ((f q).restrict fun x hx => by
      rw [CochainCx.Z, LinearMap.mem_ker] at *; rw [hf.apply, hx, map_zero])
    (by
      rintro ⟨x, hx⟩ hB
      simp only [Submodule.mem_comap, Submodule.coe_subtype, LinearMap.restrict_apply] at hB ⊢
      obtain ⟨y, ⟨z, rfl⟩, rfl⟩ := hB
      refine ⟨f (q - 1 + 1) (A.d (q - 1) z), ⟨f (q - 1) z, hf.apply _ _⟩, ?_⟩
      have key : ∀ {a b : ℤ} (h : a = b) (w : A.X a),
          B.castX h (f a w) = f b (A.castX h w) := by
        intro a b h w; subst h; rfl
      exact key _ _)

lemma IsChainMap.mem_Z {A B : CochainCx.{u} R} {f : GMap A B} (hf : IsChainMap f) (q : ℤ) (z : A.Z q) :
    f q z ∈ B.Z q := by
  rw [CochainCx.mem_Z, hf.apply, (A.mem_Z q z).1 z.2, map_zero]

lemma HmapOf_mk {A B : CochainCx.{u} R} (f : GMap A B) (hf : IsChainMap f) (q : ℤ) (z : A.Z q) :
    HmapOf f hf q (Submodule.Quotient.mk z) = Submodule.Quotient.mk ⟨f q z, hf.mem_Z q z⟩ := rfl

lemma CochainCx.mk_eq_mk_iff (C : CochainCx.{u} R) (q : ℤ) (z z' : C.Z q) :
    (Submodule.Quotient.mk z : C.H q) = Submodule.Quotient.mk z' ↔ (z : C.X q) - z' ∈ C.B q := by
  rw [Submodule.Quotient.eq, Submodule.mem_comap, Submodule.coe_subtype, Submodule.coe_sub]

section ConeSec

variable {C D : CochainCx.{u} R} (r : GMap C D) (hr : IsChainMap r)
include hr

/-- **`eq:conediff`**: the mapping cone `Cone(r)^q = C^{q+1} ⊕ D^q`, `D_cone(c, c_∂) = (−dc, dc_∂ + rc)`; `d² = 0`
(**`lem:conerelative`(2)**, proved here, in the field `d_comp_d`). -/
def cone : CochainCx.{u} R where
  X q := C.X (q + 1) × D.X q
  d q := LinearMap.prod (-(C.d (q + 1)).comp (LinearMap.fst R _ _))
    ((D.d q).comp (LinearMap.snd R _ _) + (r (q + 1)).comp (LinearMap.fst R _ _))
  d_comp_d q := by
    apply LinearMap.ext; rintro ⟨c, e⟩
    simp only [LinearMap.comp_apply, LinearMap.prod_apply, Function.prod, LinearMap.neg_apply,
      LinearMap.add_apply, LinearMap.coe_fst, LinearMap.coe_snd, LinearMap.zero_apply, map_neg, map_add]
    refine Prod.ext ?_ ?_
    · simp [C.d_d]
    · simp [D.d_d, hr.apply]

@[simp] lemma cone_d_apply (q : ℤ) (c : C.X (q + 1)) (e : D.X q) :
    (cone r hr).d q ((c, e) : C.X (q + 1) × D.X q) =
      ((-(C.d (q + 1) c), D.d q e + r (q + 1) c) : C.X (q + 1 + 1) × D.X (q + 1)) := rfl

/-- `ι : D → Cone(r)`, `c_∂ ↦ (0, c_∂)`. -/
def coneInc : GMap D (cone r hr) := fun q => LinearMap.inr R (C.X (q + 1)) (D.X q)

@[simp] lemma coneInc_apply (q : ℤ) (e : D.X q) : coneInc r hr q e = (0, e) := rfl

/-- `π : Cone(r) → C[1]`, `(c, c_∂) ↦ c` (as a map `Cone^q → C^{q+1}`). -/
def coneProj (q : ℤ) : (cone r hr).X q →ₗ[R] C.X (q + 1) := LinearMap.fst R _ _

/-- **`lem:conerelative`(2)**: `ι` is a cochain map, and `π` intertwines `D_cone` with `−d` (the differential of
`C[1]`); the sequence `0 → D^q → Cone^q → C^{q+1} → 0` is exact in every degree. -/
theorem cone_ses (q : ℤ) :
    IsChainMap (coneInc r hr) ∧
    (∀ x : (cone r hr).X q, coneProj r hr (q + 1) ((cone r hr).d q x) = -C.d (q + 1) (coneProj r hr q x)) ∧
    Function.Injective (coneInc r hr q) ∧ Function.Exact (coneInc r hr q) (coneProj r hr q) ∧
    Function.Surjective (coneProj r hr q) := by
  refine ⟨fun q => ?_, fun x => rfl, LinearMap.inr_injective, ?_, LinearMap.fst_surjective⟩
  · apply LinearMap.ext; intro e
    simp; rfl
  · intro x
    constructor
    · intro hx; exact ⟨x.2, Prod.ext hx.symm rfl⟩
    · rintro ⟨e, rfl⟩; rfl

/-- **`lem:conerelative`(2), `eq:coneles`, exactness at `H^{q+1}(C)`**: a cocycle `c ∈ Z^{q+1}(C)` has `r c ∈ B^{q+1}(D)`
iff it is `π` of a cocycle of `Cone(r)^q` modulo `B^{q+1}(C)` (the connecting map `H^{q+1}(C) → H^{q+1}(D)` is
`+r`). -/
theorem coneles_exact_C (q : ℤ) (c : C.X (q + 1)) (hc : c ∈ C.Z (q + 1)) :
    r (q + 1) c ∈ D.B (q + 1) ↔
      ∃ x ∈ (cone r hr).Z q, coneProj r hr q x - c ∈ C.B (q + 1) := by
  rw [D.mem_B_succ]
  rw [CochainCx.mem_Z] at hc
  constructor
  · rintro ⟨e, he⟩
    refine ⟨(c, -e), ?_, ?_⟩
    · show ((-(C.d (q + 1) c), D.d q (-e) + r (q + 1) c) : C.X (q + 1 + 1) × D.X (q + 1)) = 0
      refine Prod.ext ?_ ?_
      · simp [hc]
      · simp [he]
    · rw [C.mem_B_succ]; exact ⟨0, by rw [map_zero]; exact (sub_self c).symm⟩
  · rintro ⟨⟨c', e⟩, hx, hb⟩
    rw [C.mem_B_succ] at hb
    obtain ⟨b, hb⟩ := hb
    change ((-(C.d (q + 1) c'), D.d q e + r (q + 1) c') : C.X (q + 1 + 1) × D.X (q + 1)) = 0 at hx
    have h2 : D.d q e + r (q + 1) c' = 0 := congrArg Prod.snd hx
    have hcb : c = c' - C.d q b := by
      change C.d q b = c' - c at hb; rw [hb]; abel
    refine ⟨-(e + r q b), ?_⟩
    rw [hcb, map_sub, ← hr.apply, map_neg, map_add, eq_neg_of_add_eq_zero_left h2]
    abel

/-- **`eq:coneles`, exactness at `H^q(D)`**: a cocycle `c_∂ ∈ Z^q(D)` has `ι c_∂ ∈ B^q(Cone)` iff it is `r` of a
cocycle of `C^q` modulo `B^q(D)`. -/
theorem coneles_exact_D (q : ℤ) (e : D.X q) (he : e ∈ D.Z q) :
    coneInc r hr q e ∈ (cone r hr).B q ↔ ∃ c ∈ C.Z q, r q c - e ∈ D.B q := by
  obtain ⟨p, rfl⟩ : ∃ p, q = p + 1 := ⟨q - 1, by ring⟩
  rw [(cone r hr).mem_B_succ]
  constructor
  · rintro ⟨⟨c, e'⟩, h⟩
    rw [cone_d_apply, coneInc_apply] at h
    have h1 : -(C.d (p + 1) c) = 0 := congrArg Prod.fst h
    have h2 : D.d p e' + r (p + 1) c = e := congrArg Prod.snd h
    refine ⟨c, (C.mem_Z _ _).2 (neg_eq_zero.mp h1), ?_⟩
    rw [D.mem_B_succ]
    refine ⟨-e', ?_⟩
    rw [map_neg, ← h2]; abel
  · rintro ⟨c, hc, he'⟩
    rw [D.mem_B_succ] at he'
    obtain ⟨e', he'⟩ := he'
    rw [CochainCx.mem_Z] at hc
    refine ⟨(c, -e'), ?_⟩
    rw [cone_d_apply, coneInc_apply]
    refine Prod.ext ?_ ?_
    · simp [hc]
    · simp only [map_neg]; rw [he']; abel

/-- **`eq:coneles`, exactness at `H^q(Cone) = H^{q+1}_c`**: a cocycle `x` of `Cone^q` has `π x ∈ B^{q+1}(C)` iff it is
`ι` of a cocycle of `D^q` modulo `B^q(Cone)`. -/
theorem coneles_exact_cone (q : ℤ) (x : (cone r hr).X q) (hx : x ∈ (cone r hr).Z q) :
    coneProj r hr q x ∈ C.B (q + 1) ↔ ∃ e ∈ D.Z q, x - coneInc r hr q e ∈ (cone r hr).B q := by
  obtain ⟨p, rfl⟩ : ∃ p, q = p + 1 := ⟨q - 1, by ring⟩
  obtain ⟨c, e⟩ := x
  change ((-(C.d (p + 1 + 1) c), D.d (p + 1) e + r (p + 1 + 1) c) : C.X (p + 1 + 1 + 1) × D.X (p + 1 + 1)) = 0 at hx
  have hx2 : D.d (p + 1) e + r (p + 1 + 1) c = 0 := congrArg Prod.snd hx
  rw [C.mem_B_succ]
  change (∃ y, C.d (p + 1) y = c) ↔ _
  constructor
  · rintro ⟨b, rfl⟩
    refine ⟨e + r (p + 1) b, ?_, ?_⟩
    · rw [CochainCx.mem_Z, map_add, hr.apply, hx2]
    · rw [(cone r hr).mem_B_succ]
      refine ⟨(-b, 0), Prod.ext ?_ ?_⟩
      · show -(C.d (p + 1) (-b)) = C.d (p + 1) b - 0; simp
      · show D.d p 0 + r (p + 1) (-b) = e - (e + r (p + 1) b); simp
  · rintro ⟨e', -, hy⟩
    rw [(cone r hr).mem_B_succ] at hy
    obtain ⟨⟨b', e''⟩, hy⟩ := hy
    rw [cone_d_apply, coneInc_apply] at hy
    have := congrArg Prod.fst hy
    change -(C.d (p + 1) b') = c - 0 at this
    exact ⟨-b', by rw [map_neg, this, sub_zero]⟩

/-- The relative complex `ker r ⊆ C`. -/
def relCx : CochainCx.{u} R where
  X q := LinearMap.ker (r q)
  d q := (C.d q).restrict fun x hx => by
    rw [LinearMap.mem_ker] at *; rw [← hr.apply, hx, map_zero]
  d_comp_d q := by ext ⟨x, hx⟩; simp [C.d_d]

/-- **`lem:conerelative`(3)**: if `r` is onto in every degree, `c ↦ (c, 0)` is a quasi-isomorphism
`(ker r)[1] → Cone(r)`, so `H^{q+1}(ker r) ≅ H^q(Cone(r)) = H^{q+1}_c`. -/
theorem relative_iso (hsurj : ∀ q, Function.Surjective (r q)) (q : ℤ) :
    Nonempty ((relCx r hr).H (q + 1) ≃ₗ[R] (cone r hr).H q) := by
  obtain ⟨p, rfl⟩ : ∃ p, q = p + 1 := ⟨q - 1, by ring⟩
  set K := relCx r hr with hK
  set Q := cone r hr with hQ
  let f : K.Z (p + 1 + 1) →ₗ[R] Q.X (p + 1) :=
    (LinearMap.inl R (C.X (p + 1 + 1)) (D.X (p + 1))).comp
      ((LinearMap.ker (r (p + 1 + 1))).subtype.comp (K.Z (p + 1 + 1)).subtype)
  have hfZ : ∀ z, f z ∈ Q.Z (p + 1) := by
    rintro ⟨⟨c, hc⟩, hz⟩
    have hz' : C.d (p + 1 + 1) c = 0 := congrArg Subtype.val ((K.mem_Z _ _).1 hz)
    rw [LinearMap.mem_ker] at hc
    show ((-(C.d (p + 1 + 1) c), D.d (p + 1) 0 + r (p + 1 + 1) c) : C.X (p + 1 + 1 + 1) × D.X (p + 1 + 1)) = 0
    rw [hz', hc]; simp
  let Φ : K.Z (p + 1 + 1) →ₗ[R] Q.H (p + 1) := (Submodule.mkQ _).comp (f.codRestrict _ hfZ)
  have hΦ : ∀ z, Φ z = Submodule.Quotient.mk ⟨f z, hfZ z⟩ := fun z => rfl
  have hsurjΦ : Function.Surjective Φ := by
    intro w
    induction w using Submodule.Quotient.induction_on with
    | H x =>
    obtain ⟨⟨c, e⟩, hx⟩ := x
    change ((-(C.d (p + 1 + 1) c), D.d (p + 1) e + r (p + 1 + 1) c) : C.X (p + 1 + 1 + 1) × D.X (p + 1 + 1)) = 0
      at hx
    have h1 : C.d (p + 1 + 1) c = 0 := neg_eq_zero.mp (congrArg Prod.fst hx)
    have h2 : D.d (p + 1) e + r (p + 1 + 1) c = 0 := congrArg Prod.snd hx
    obtain ⟨c0, hc0⟩ := hsurj (p + 1) e
    have hk : c + C.d (p + 1) c0 ∈ LinearMap.ker (r (p + 1 + 1)) := by
      rw [LinearMap.mem_ker, map_add, ← hr.apply, hc0]; exact (add_comm _ _).trans h2
    have hkz : (⟨c + C.d (p + 1) c0, hk⟩ : K.X (p + 1 + 1)) ∈ K.Z (p + 1 + 1) := by
      show K.d (p + 1 + 1) ⟨_, hk⟩ = 0
      apply Subtype.ext
      show C.d (p + 1 + 1) (c + C.d (p + 1) c0) = 0
      rw [map_add, h1, C.d_d, add_zero]
    refine ⟨⟨_, hkz⟩, ?_⟩
    rw [hΦ, Submodule.Quotient.eq, Submodule.mem_comap, Submodule.coe_subtype]
    refine (Q.mem_B_succ _).2 ⟨(-c0, 0), Prod.ext ?_ ?_⟩
    · show -(C.d (p + 1) (-c0)) = c + C.d (p + 1) c0 - c; simp
    · show D.d p 0 + r (p + 1) (-c0) = 0 - e; simp [hc0]
  have hker : LinearMap.ker Φ = (K.B (p + 1 + 1)).comap (K.Z (p + 1 + 1)).subtype := by
    ext ⟨⟨c, hc⟩, hz⟩
    rw [LinearMap.mem_ker, hΦ, Q.mk_eq_zero_iff, Submodule.mem_comap, Submodule.coe_subtype, K.mem_B_succ,
      Q.mem_B_succ]
    change (∃ y : Q.X p, Q.d p y = ((c, 0) : Q.X (p + 1))) ↔
      ∃ y : K.X (p + 1), K.d (p + 1) y = ⟨c, hc⟩
    constructor
    · rintro ⟨⟨c'', e''⟩, hy⟩
      change ((-(C.d (p + 1) c''), D.d p e'' + r (p + 1) c'') : C.X (p + 1 + 1) × D.X (p + 1)) = (c, 0) at hy
      have hy1 : -(C.d (p + 1) c'') = c := congrArg Prod.fst hy
      have hy2 : D.d p e'' + r (p + 1) c'' = 0 := congrArg Prod.snd hy
      obtain ⟨c3, hc3⟩ := hsurj p e''
      have hm : -(c'' + C.d p c3) ∈ LinearMap.ker (r (p + 1)) := by
        rw [LinearMap.mem_ker, map_neg, map_add, ← hr.apply, hc3, neg_eq_zero]; exact (add_comm _ _).trans hy2
      refine ⟨⟨_, hm⟩, Subtype.ext ?_⟩
      show C.d (p + 1) (-(c'' + C.d p c3)) = c
      rw [map_neg, map_add, C.d_d, add_zero, hy1]
    · rintro ⟨⟨b, hb⟩, hy⟩
      have hy' : C.d (p + 1) b = c := congrArg Subtype.val hy
      rw [LinearMap.mem_ker] at hb
      refine ⟨(-b, 0), Prod.ext ?_ ?_⟩
      · show -(C.d (p + 1) (-b)) = c; rw [map_neg, neg_neg, hy']
      · show D.d p 0 + r (p + 1) (-b) = 0; rw [map_zero, map_neg, hb, neg_zero, add_zero]
  exact ⟨(Submodule.quotEquivOfEq _ _ hker.symm).trans (Φ.quotKerEquivOfSurjective hsurjΦ)⟩

/-- `H^q_c := H^{q-1}(Cone(r))`. -/
abbrev Hc (q : ℤ) := (cone r hr).H (q - 1)

/-- The map `r_* : H^q(C) → H^q(D)`. -/
def Hmap (q : ℤ) : C.H q →ₗ[R] D.H q :=
  Submodule.mapQ _ _ ((r q).restrict fun x hx => by
      rw [CochainCx.Z, LinearMap.mem_ker] at *; rw [hr.apply, hx, map_zero])
    (by
      rintro ⟨x, hx⟩ hB
      simp only [Submodule.mem_comap, Submodule.coe_subtype, LinearMap.restrict_apply] at hB ⊢
      obtain ⟨y, ⟨z, rfl⟩, rfl⟩ := hB
      refine ⟨r (q - 1 + 1) (C.d (q - 1) z), ⟨r (q - 1) z, hr.apply _ _⟩, ?_⟩
      have key : ∀ {a b : ℤ} (h : a = b) (w : C.X a),
          D.castX h (r a w) = r b (C.castX h w) := by
        intro a b h w; subst h; rfl
      exact key _ _)

lemma Hmap_mk (q : ℤ) (z : C.Z q) :
    Hmap r hr q (Submodule.Quotient.mk z) = Submodule.Quotient.mk ⟨r q z, hr.mem_Z q z⟩ := rfl

/-! ### The long exact sequence `eq:coneles` on cohomology (L18-4) -/

lemma coneInc_isChainMap : IsChainMap (coneInc r hr) := (cone_ses r hr 0).1

/-- `ι_* : H^q(D) → H^q(Cone(r))`. -/
def incH (q : ℤ) : D.H q →ₗ[R] (cone r hr).H q := HmapOf (coneInc r hr) (coneInc_isChainMap r hr) q

lemma proj_mem_Z (q : ℤ) (x : (cone r hr).Z q) : coneProj r hr q x ∈ C.Z (q + 1) := by
  have h := congrArg Prod.fst (((cone r hr).mem_Z q x).1 x.2)
  exact (C.mem_Z _ _).2 (neg_eq_zero.mp h)

/-- `π_* : H^q(Cone(r)) → H^{q+1}(C)`. -/
def projH (q : ℤ) : (cone r hr).H q →ₗ[R] C.H (q + 1) :=
  Submodule.mapQ _ _ ((coneProj r hr q).restrict fun x hx => proj_mem_Z r hr q ⟨x, hx⟩)
    (by
      rintro ⟨x, hx⟩ hB
      simp only [Submodule.mem_comap, Submodule.coe_subtype, LinearMap.restrict_apply] at hB ⊢
      obtain ⟨p, rfl⟩ : ∃ p, q = p + 1 := ⟨q - 1, by ring⟩
      obtain ⟨y, rfl⟩ := ((cone r hr).mem_B_succ x).1 hB
      exact (C.mem_B_succ _).2 ⟨-y.1, by rw [map_neg]; rfl⟩)

lemma projH_mk (q : ℤ) (z : (cone r hr).Z q) :
    projH r hr q (Submodule.Quotient.mk z) = Submodule.Quotient.mk ⟨coneProj r hr q z, proj_mem_Z r hr q z⟩ := rfl

/-- **`eq:coneles`, exactness at `H^{q+1}(C)`**, on cohomology. -/
theorem exact_C_H (q : ℤ) : LinearMap.range (projH r hr q) = LinearMap.ker (Hmap r hr (q + 1)) := by
  ext w
  induction w using Submodule.Quotient.induction_on with
  | H z =>
  rw [LinearMap.mem_ker, Hmap_mk, CochainCx.mk_eq_zero_iff, coneles_exact_C r hr q z z.2, LinearMap.mem_range]
  constructor
  · rintro ⟨v, hv⟩
    induction v using Submodule.Quotient.induction_on with
    | H x =>
    rw [projH_mk, CochainCx.mk_eq_mk_iff] at hv
    exact ⟨x, x.2, hv⟩
  · rintro ⟨x, hx, hb⟩
    exact ⟨Submodule.Quotient.mk ⟨x, hx⟩, by rw [projH_mk, CochainCx.mk_eq_mk_iff]; exact hb⟩

/-- **`eq:coneles`, exactness at `H^q(D)`**, on cohomology. -/
theorem exact_D_H (q : ℤ) : LinearMap.range (Hmap r hr q) = LinearMap.ker (incH r hr q) := by
  ext w
  induction w using Submodule.Quotient.induction_on with
  | H z =>
  rw [LinearMap.mem_ker, incH, HmapOf_mk, CochainCx.mk_eq_zero_iff]
  erw [coneles_exact_D r hr q z z.2]
  rw [LinearMap.mem_range]
  constructor
  · rintro ⟨v, hv⟩
    induction v using Submodule.Quotient.induction_on with
    | H c =>
    rw [Hmap_mk, CochainCx.mk_eq_mk_iff] at hv
    exact ⟨c, c.2, hv⟩
  · rintro ⟨c, hc, hb⟩
    exact ⟨Submodule.Quotient.mk ⟨c, hc⟩, by rw [Hmap_mk, CochainCx.mk_eq_mk_iff]; exact hb⟩

/-- **`eq:coneles`, exactness at `H^q(Cone)`**, on cohomology. -/
theorem exact_cone_H (q : ℤ) : LinearMap.range (incH r hr q) = LinearMap.ker (projH r hr q) := by
  ext w
  induction w using Submodule.Quotient.induction_on with
  | H z =>
  rw [LinearMap.mem_ker, projH_mk, CochainCx.mk_eq_zero_iff, coneles_exact_cone r hr q z z.2, LinearMap.mem_range]
  constructor
  · rintro ⟨v, hv⟩
    induction v using Submodule.Quotient.induction_on with
    | H e =>
    rw [incH, HmapOf_mk, CochainCx.mk_eq_mk_iff] at hv
    refine ⟨e, e.2, ?_⟩
    rw [← neg_sub]; exact Submodule.neg_mem _ hv
  · rintro ⟨e, he, hb⟩
    refine ⟨Submodule.Quotient.mk ⟨e, he⟩, ?_⟩
    rw [incH, HmapOf_mk, CochainCx.mk_eq_mk_iff, ← neg_sub]; exact Submodule.neg_mem _ hb

/-! ### `lem:conesign` and `prop:coneambiguity`(1) -/

/-- The cone map `Φ(c, c_∂) = (Θ c, Θ_∂ c_∂ + h c)` (`eq:conemap`'s shape, for any `Θ`, `Θ_∂`, `h`). -/
def coneMap (Θ : GMap C C) (Θb : GMap D D) (h : GMapNeg C D) : GMap (cone r hr) (cone r hr) := fun q =>
  LinearMap.prod ((Θ (q + 1)).comp (LinearMap.fst R _ _))
    ((Θb q).comp (LinearMap.snd R _ _) + (h q).comp (LinearMap.fst R _ _))

@[simp] lemma coneMap_apply (Θ : GMap C C) (Θb : GMap D D) (h : GMapNeg C D) (q : ℤ) (c : C.X (q + 1))
    (e : D.X q) :
    coneMap r hr Θ Θb h q ((c, e) : C.X (q + 1) × D.X q) =
      ((Θ (q + 1) c, Θb q e + h q c) : C.X (q + 1) × D.X q) := rfl

/-- **`lem:conesign`** (`eq:conesign`): for cochain maps `Θ`, `Θ_∂` and any `h`,
`(D_cone Φ − Φ D_cone)(c, c_∂) = (0, (d h + h d + r Θ − Θ_∂ r) c)`; so `Φ` is a cochain map iff
`d h + h d = Θ_∂ r − r Θ`. -/
theorem conesign (Θ : GMap C C) (hΘ : IsChainMap Θ) (Θb : GMap D D) (hΘb : IsChainMap Θb) (h : GMapNeg C D)
    (q : ℤ) (x : (cone r hr).X q) :
    (cone r hr).d q (coneMap r hr Θ Θb h q x) - coneMap r hr Θ Θb h (q + 1) ((cone r hr).d q x) =
      (0, D.d q (h q x.1) + h (q + 1) (C.d (q + 1) x.1) + r (q + 1) (Θ (q + 1) x.1) - Θb (q + 1) (r (q + 1) x.1)) := by
  obtain ⟨c, e⟩ := x
  refine Prod.ext ?_ ?_
  · erw [Prod.fst_sub]; simp [hΘ.apply]
  · erw [Prod.snd_sub]; simp [hΘb.apply]; abel

/-- **`lem:conesign`**, second sentence: `Φ ∘ ι = ι ∘ Θ_∂` and `π ∘ Φ = Θ ∘ π` for every `h`. -/
theorem coneMap_inc_proj (Θ : GMap C C) (Θb : GMap D D) (h : GMapNeg C D) (q : ℤ) :
    (coneMap r hr Θ Θb h q).comp (coneInc r hr q) = (coneInc r hr q).comp (Θb q) ∧
    (coneProj r hr q).comp (coneMap r hr Θ Θb h q) = (Θ (q + 1)).comp (coneProj r hr q) := by
  constructor
  · apply LinearMap.ext; intro e
    simp only [LinearMap.comp_apply, coneInc_apply, coneMap_apply, map_zero, add_zero]; rfl
  · apply LinearMap.ext; rintro ⟨c, e⟩; rfl

/-- **`prop:coneambiguity`(1)**, cochain level: for two `h`, `h'` both satisfying the cone identity, `ξ := h' − h`
satisfies `d ξ + ξ d = 0`, and `Φ' − Φ = ι ∘ ξ ∘ π` on the nose; on cohomology this is the element of
`Hom(ker r^{q+1}, coker r^q)` of [C]. -/
theorem coneambiguity (Θ : GMap C C) (Θb : GMap D D) (h h' : GMapNeg C D)
    (hh : ∀ q x, D.d q (h q x) + h (q + 1) (C.d (q + 1) x) = Θb (q + 1) (r (q + 1) x) - r (q + 1) (Θ (q + 1) x))
    (hh' : ∀ q x, D.d q (h' q x) + h' (q + 1) (C.d (q + 1) x) = Θb (q + 1) (r (q + 1) x) - r (q + 1) (Θ (q + 1) x))
    (q : ℤ) :
    (∀ x, D.d q ((h' q - h q) x) + (h' (q + 1) - h (q + 1)) (C.d (q + 1) x) = 0) ∧
    ∀ x, coneMap r hr Θ Θb h' q x - coneMap r hr Θ Θb h q x = coneInc r hr q ((h' q - h q) (coneProj r hr q x)) := by
  constructor
  · intro x
    simp only [LinearMap.sub_apply, map_sub]
    rw [show ∀ a b c d : D.X (q + 1), a - b + (c - d) = (a + c) - (b + d) from fun a b c d => by abel,
      hh q x, hh' q x, sub_self]
  · rintro ⟨c, e⟩
    refine Prod.ext ?_ ?_
    · erw [Prod.fst_sub]; simp [coneProj]
    · erw [Prod.snd_sub]; simp [coneProj]; rfl

end ConeSec

/-- **`eq:conedims`** (**`lem:conerelative`(5)**), over a field, both identities as stated in [C]: the first
unconditionally (given `H^0_c = 0`, `lem:conerelative`(4), here a hypothesis of the abstract layer), the second under
`H^2(C) = 0`. -/
theorem conedims {F : Type v} [Field F] {C D : CochainCx.{u} F} (r : GMap C D) (hr : IsChainMap r)
    [∀ q, FiniteDimensional F (C.X q)] [∀ q, FiniteDimensional F (D.X q)]
    (hc0 : Subsingleton (Hc r hr 0)) :
    Module.finrank F (Hc r hr 1) + Module.finrank F (C.H 0) =
        Module.finrank F (D.H 0) + Module.finrank F (LinearMap.ker (Hmap r hr 1)) ∧
      (Subsingleton (C.H 2) →
        Module.finrank F (Hc r hr 2) + Module.finrank F (LinearMap.range (Hmap r hr 1)) =
          Module.finrank F (D.H 1)) := by
  classical
  have : ∀ q, FiniteDimensional F ((cone r hr).X q) := fun q =>
    (inferInstance : FiniteDimensional F (C.X (q + 1) × D.X q))
  have : Subsingleton ((cone r hr).H (-1)) := hc0
  have e1 : LinearMap.range (projH r hr (-1)) = LinearMap.ker (Hmap r hr 0) := exact_C_H r hr (-1)
  have hπ : projH r hr (-1) = 0 := LinearMap.ext fun x => by rw [Subsingleton.elim x 0, map_zero]; rfl
  rw [hπ, LinearMap.range_zero] at e1
  have rn0 := LinearMap.finrank_range_add_finrank_ker (Hmap r hr 0)
  rw [← e1, finrank_bot] at rn0
  have rni0 := LinearMap.finrank_range_add_finrank_ker (incH r hr 0)
  rw [← exact_D_H r hr 0] at rni0
  have rnπ0 := LinearMap.finrank_range_add_finrank_ker (projH r hr 0)
  rw [← exact_cone_H r hr 0] at rnπ0
  have e10 : LinearMap.range (projH r hr 0) = LinearMap.ker (Hmap r hr 1) := exact_C_H r hr 0
  rw [e10] at rnπ0
  refine ⟨?_, fun h2 => ?_⟩
  · have A : Module.finrank F (LinearMap.range (incH r hr 0)) + Module.finrank F (LinearMap.range (Hmap r hr 0)) =
        Module.finrank F (D.H 0) := rni0
    have B : Module.finrank F (LinearMap.ker (Hmap r hr 1)) + Module.finrank F (LinearMap.range (incH r hr 0)) =
        Module.finrank F ((cone r hr).H 0) := rnπ0
    have Cc : Module.finrank F (LinearMap.range (Hmap r hr 0)) + 0 = Module.finrank F (C.H 0) := rn0
    change Module.finrank F ((cone r hr).H 0) + _ = _
    omega
  · have : Subsingleton (C.H (1 + 1)) := h2
    have hk : LinearMap.ker (projH r hr 1) = ⊤ :=
      LinearMap.ker_eq_top.2 (LinearMap.ext fun x => Subsingleton.elim _ _)
    have rni1 := LinearMap.finrank_range_add_finrank_ker (incH r hr 1)
    rw [exact_cone_H r hr 1, hk, finrank_top, ← exact_D_H r hr 1] at rni1
    change Module.finrank F ((cone r hr).H 1) + _ = _
    omega

/-- **`cl:les`** (the six-term sequence, abstract form): if `H^0_c = 0` and `H^2(C) = 0`, then `r_*` is injective on
`H^0`, and over a field `0 → H^0 → H^0(∂) → H^1_c → H^1 → H^1(∂) → H^2_c → 0` is exact; the Euler characteristic
identity is its alternating sum. -/
theorem sixterm {F : Type v} [Field F] {C D : CochainCx.{u} F} (r : GMap C D) (hr : IsChainMap r)
    [∀ q, FiniteDimensional F (C.X q)] [∀ q, FiniteDimensional F (D.X q)]
    (hc0 : Subsingleton (Hc r hr 0)) (h2 : Subsingleton (C.H 2)) :
    Function.Injective (Hmap r hr 0) ∧
      (Module.finrank F (C.H 0) + Module.finrank F (D.H 1) + Module.finrank F (Hc r hr 1) : ℤ) =
        Module.finrank F (D.H 0) + Module.finrank F (C.H 1) + Module.finrank F (Hc r hr 2) := by
  have hd := conedims r hr hc0
  have h2' := hd.2 h2
  have rn1 := LinearMap.finrank_range_add_finrank_ker (Hmap r hr 1)
  refine ⟨?_, by have := hd.1; omega⟩
  have : Subsingleton ((cone r hr).H (-1)) := hc0
  have e1 : LinearMap.range (projH r hr (-1)) = LinearMap.ker (Hmap r hr 0) := exact_C_H r hr (-1)
  have hπ : projH r hr (-1) = 0 := LinearMap.ext fun x => by rw [Subsingleton.elim x 0, map_zero]; rfl
  rw [hπ, LinearMap.range_zero] at e1
  exact LinearMap.ker_eq_bot.1 e1.symm

/-- **`prop:coneambiguity`(2)**, realisation over a field: every `F`-linear `λ : H^{q+1}(C) → Z^q(D)` is `ξ_*` of a
degree-`(−1)` cochain map `ξ` (`d ξ + ξ d = 0`). Not claimed over `ℤ/ℓ^e` (entry `err:e3`). -/
theorem coneambiguity_realise {F : Type v} [Field F] {C D : CochainCx.{u} F} (q : ℤ)
    (lam : C.H (q + 1) →ₗ[F] D.Z q) :
    ∃ ξ : GMapNeg C D, (∀ p x, D.d p (ξ p x) + ξ (p + 1) (C.d (p + 1) x) = 0) ∧
      ∀ z : C.Z (q + 1), (ξ q z : D.X q) = lam (Submodule.Quotient.mk z) := by
  classical
  obtain ⟨g0, hg0⟩ := LinearMap.exists_extend (lam.comp (Submodule.mkQ ((C.B (q + 1)).comap (C.Z (q + 1)).subtype)))
  let g : C.X (q + 1) →ₗ[F] D.X q := (D.Z q).subtype.comp g0
  have hg : ∀ z : C.Z (q + 1), g z = lam (Submodule.Quotient.mk z) := fun z =>
    congrArg Subtype.val (LinearMap.congr_fun hg0 z)
  have hgZ : ∀ x, D.d q (g x) = 0 := fun x => (D.mem_Z q _).1 (g0 x).2
  let ξ : GMapNeg C D := fun p => if h : p = q then h ▸ g else 0
  have hξq : ξ q = g := by simp [ξ]
  have hξne : ∀ p, p ≠ q → ξ p = 0 := fun p hp => by simp [ξ, hp]
  refine ⟨ξ, fun p x => ?_, fun z => by rw [hξq]; exact hg z⟩
  by_cases hpq : p = q
  · subst hpq
    rw [hξne (p + 1) (by omega), hξq, LinearMap.zero_apply, add_zero, hgZ]
  · rw [hξne p hpq, LinearMap.zero_apply, map_zero, zero_add]
    by_cases hpq' : p + 1 = q
    · subst hpq'
      rw [hξq]
      have hz : C.d (p + 1) x ∈ C.Z (p + 1 + 1) := (C.mem_Z _ _).2 (C.d_d _ _)
      have := hg ⟨_, hz⟩
      rw [this, (CochainCx.mk_eq_zero_iff C _ _).2 ((C.mem_B_succ _).2 ⟨x, rfl⟩), map_zero]
      rfl
    · rw [hξne (p + 1) hpq', LinearMap.zero_apply]

end GL2.Cone
