import Mathlib

/-!
# G10: saturation and the defect ([C] v1.16.0-pre.3 `sec:saturation`, `sec:satsec`) — statements (L16-6), proved (L17-1)

[GT] chapter `ch:saturation`.  Mirrors [C] `def:satdata`, `prop:saturation` (P15), `prop:satdefect` (P17).

**Setting.** `O = ℤ_[p]`; `L` a free `O`-module of finite rank; `T` a commutative `O`-algebra acting faithfully on `L`
through an injective `O`-algebra map `act : T →ₐ[O] End_O(L)` (this is [C]'s "commutative subalgebra `T ⊆ End(L)`", with
the subalgebra replaced by an abstract algebra and an injective action, so that `T` is a commutative ring in Lean).

* `tsat`: `T^sat = (T ⊗ ℚ) ∩ End(L)`, written inside `End(L)` as `{x | ∃ k, p^k x ∈ act(T)}` (every non-zero element of
  `ℤ_p` is a unit times a power of `p`);
* `kerK K`: the kernel of `T → End(L/p^K L)`, i.e. `{t | act t (L) ⊆ p^K L}`; `AK K := T ⧸ kerK K` is `𝖠_K`;
* `deltaSat`, `cSat`, `rk`: `δ^sat = length(T^sat/T)`, `c^sat` the exponent of `T^sat/T`, `r = rank T`;
* `epsSat K := K·r − length(𝖠_K)` (in `ℤ`, with `length` read through `ENat.toNat`).

Every theorem below is **proved** (L17-1, sorry-free; the statements are L16-6's, unchanged).
The algebra-level clauses of [C] (the isomorphism in (3)(b) "of `F_ℓ`-algebras", the `𝔪`-adic filtrations in (5)) are
stated at the module level, where the lengths live; the ring structure of `𝖠_K` is that of the quotient `T ⧸ kerK K`.
-/

noncomputable section

namespace GL2.Sat

open Submodule
open scoped Pointwise

variable (p : ℕ) [Fact p.Prime]

/-- The ideal `(p)` of `ℤ_p`. -/
noncomputable abbrev pI : Ideal ℤ_[p] := Ideal.span {(p : ℤ_[p])}

variable {p} {L : Type*} [AddCommGroup L] [Module ℤ_[p] L]
  {T : Type*} [CommRing T] [Algebra ℤ_[p] T] (act : T →ₐ[ℤ_[p]] Module.End ℤ_[p] L)

/-- **`def:satdata`**: `T^sat = (T ⊗ ℚ) ∩ End(L)`, as `{x ∈ End L | ∃ k, p^k x ∈ act(T)}`. -/
def tsat : Submodule ℤ_[p] (Module.End ℤ_[p] L) where
  carrier := {x | ∃ k : ℕ, ((p : ℤ_[p]) ^ k) • x ∈ LinearMap.range act.toLinearMap}
  zero_mem' := ⟨0, by simp⟩
  add_mem' := by
    rintro x y ⟨k, hk⟩ ⟨l, hl⟩
    refine ⟨k + l, ?_⟩
    rw [smul_add, pow_add, mul_smul, mul_smul, smul_comm ((p : ℤ_[p]) ^ k) ((p : ℤ_[p]) ^ l) x]
    exact add_mem (Submodule.smul_mem _ _ hk) (Submodule.smul_mem _ _ hl)
  smul_mem' := by
    rintro c x ⟨k, hk⟩
    exact ⟨k, by rw [smul_comm]; exact Submodule.smul_mem _ _ hk⟩

/-- `act(T)` inside `T^sat`, as a submodule of `T^sat`. -/
def tInSat : Submodule ℤ_[p] (tsat act) := (LinearMap.range act.toLinearMap).comap (tsat act).subtype

/-- `T^sat / T`. -/
abbrev SatQuot := tsat act ⧸ tInSat act

/-- `T` is **saturated**: `T = T^sat`. -/
def Saturated : Prop := ∀ x ∈ tsat act, x ∈ LinearMap.range act.toLinearMap

/-- `δ^sat = length(T^sat/T)`. -/
noncomputable def deltaSat : ℕ∞ := Module.length ℤ_[p] (SatQuot act)

/-- `c^sat`: the least `c` with `p^c T^sat ⊆ T` (the exponent of `T^sat/T`). -/
noncomputable def cSat : ℕ :=
  sInf {c : ℕ | ∀ x ∈ tsat act, ((p : ℤ_[p]) ^ c) • x ∈ LinearMap.range act.toLinearMap}

/-- `r = rank_{ℤ_p} T`. -/
noncomputable def rk : ℕ := Module.finrank ℤ_[p] T

/-- `ker(T → End(L/p^K L)) = {t | act t (L) ⊆ p^K L}`. -/
def kerK (K : ℕ) : Submodule ℤ_[p] T :=
  LinearMap.ker ((LinearMap.llcomp ℤ_[p] L L (L ⧸ (pI p ^ K) • (⊤ : Submodule ℤ_[p] L)) (mkQ _)) ∘ₗ
    act.toLinearMap)

/-- **`def:satdata`**: `𝖠_K = im(T → End(L/p^K L)) ≅ T ⧸ kerK K`. -/
abbrev AK (K : ℕ) := T ⧸ kerK act K

/-- `ε^sat_K := K r − length(𝖠_K)`. -/
noncomputable def epsSat (K : ℕ) : ℤ := (K * rk (p := p) (T := T) : ℤ) - (Module.length ℤ_[p] (AK act K)).toNat

/-- The canonical surjection `T/pT → 𝖠_K/p𝖠_K`. -/
noncomputable def redMap (K : ℕ) : (T ⧸ pI p • (⊤ : Submodule ℤ_[p] T)) →ₗ[ℤ_[p]]
    (AK act K ⧸ pI p • (⊤ : Submodule ℤ_[p] (AK act K))) :=
  mapQ _ _ (kerK act K).mkQ (map_le_iff_le_comap.mp (by
    rw [map_smul'']; exact smul_mono_right _ le_top))

/-! ### Auxiliary lemmas (L17-1) -/

section Aux

variable {M : Type*} [AddCommGroup M] [Module ℤ_[p] M]

theorem pI_pow_smul_top (K : ℕ) :
    (pI p ^ K) • (⊤ : Submodule ℤ_[p] M) = ((p : ℤ_[p]) ^ K) • (⊤ : Submodule ℤ_[p] M) := by
  rw [pI, Ideal.span_singleton_pow, Submodule.ideal_span_singleton_smul]

theorem mem_pI_pow_smul_top (K : ℕ) (x : M) :
    x ∈ (pI p ^ K) • (⊤ : Submodule ℤ_[p] M) ↔ ∃ y : M, x = ((p : ℤ_[p]) ^ K) • y := by
  rw [pI_pow_smul_top, Submodule.mem_smul_pointwise_iff_exists]
  simp [eq_comm]

theorem mem_pI_smul_top (x : M) :
    x ∈ pI p • (⊤ : Submodule ℤ_[p] M) ↔ ∃ y : M, x = (p : ℤ_[p]) • y := by
  simpa using mem_pI_pow_smul_top (M := M) 1 x

theorem pZp_ne_zero : (p : ℤ_[p]) ≠ 0 := by exact_mod_cast (Fact.out : p.Prime).ne_zero

theorem pZp_pow_ne_zero (k : ℕ) : ((p : ℤ_[p]) ^ k) ≠ 0 := pow_ne_zero _ pZp_ne_zero

/-- `I • ⊤` on a quotient is the image of `I • ⊤`. -/
theorem smul_top_quot (N : Submodule ℤ_[p] M) (I : Ideal ℤ_[p]) :
    I • (⊤ : Submodule ℤ_[p] (M ⧸ N)) = (I • (⊤ : Submodule ℤ_[p] M)).map N.mkQ := by
  rw [Submodule.map_smul'', Submodule.map_top, Submodule.range_mkQ]

/-- Additivity of length along `N ≤ N'`: `ℓ(M/N) = ℓ(N'/N) + ℓ(M/N')`. -/
theorem length_quot_add (N N' : Submodule ℤ_[p] M) (h : N ≤ N') :
    Module.length ℤ_[p] (M ⧸ N) =
      Module.length ℤ_[p] (N'.map N.mkQ) + Module.length ℤ_[p] (M ⧸ N') := by
  rw [Module.length_eq_add_of_exact (N'.map N.mkQ).subtype (N'.map N.mkQ).mkQ Subtype.val_injective
    (Submodule.mkQ_surjective _) (LinearMap.exact_subtype_mkQ _),
    (Submodule.quotientQuotientEquivQuotient N N' h).length_eq]

theorem map_mkQ_eq_bot_iff (N X : Submodule ℤ_[p] M) : X.map N.mkQ = ⊥ ↔ X ≤ N := by
  rw [← le_bot_iff, Submodule.map_le_iff_le_comap, Submodule.comap_bot, Submodule.ker_mkQ]

theorem enat_add_cancel {a b : ℕ∞} {c : ℕ} (h : a + c = b + c) : a = b :=
  WithTop.add_right_cancel (ENat.natCast_ne_top c) h

/-- For a module of finite length, `ℓ(M[c]) = ℓ(M/cM)`. -/
theorem length_torsionBy_eq (c : ℤ_[p]) (hM : Module.length ℤ_[p] M ≠ ⊤) :
    Module.length ℤ_[p] (Submodule.torsionBy ℤ_[p] M c) =
      Module.length ℤ_[p] (M ⧸ c • (⊤ : Submodule ℤ_[p] M)) := by
  let f := DistribSMul.toLinearMap ℤ_[p] M c
  have hr : LinearMap.range f = c • (⊤ : Submodule ℤ_[p] M) := by
    ext x
    rw [LinearMap.mem_range, Submodule.mem_smul_pointwise_iff_exists]
    exact ⟨fun ⟨y, h⟩ => ⟨y, Submodule.mem_top, h⟩, fun ⟨y, _, h⟩ => ⟨y, h⟩⟩
  have h1 := Module.length_eq_add_of_exact (LinearMap.ker f).subtype (LinearMap.ker f).mkQ
    Subtype.val_injective (Submodule.mkQ_surjective _) (LinearMap.exact_subtype_mkQ _)
  have h2 := Module.length_eq_add_of_exact (LinearMap.range f).subtype (LinearMap.range f).mkQ
    Subtype.val_injective (Submodule.mkQ_surjective _) (LinearMap.exact_subtype_mkQ _)
  rw [(f.quotKerEquivRange).length_eq] at h1
  have hfin : Module.length ℤ_[p] (LinearMap.range f) ≠ ⊤ := fun h => hM (by rw [h2, h, top_add])
  obtain ⟨n, hn⟩ := ENat.ne_top_iff_exists.mp hfin
  rw [← hr]
  change Module.length ℤ_[p] (LinearMap.ker f) = _
  rw [← hn] at h1 h2
  exact enat_add_cancel (c := n) (by rw [← h1, h2, add_comm])

/-- `ℓ(ℤ_p/p^K) = K` (`ℤ_p` is a DVR with maximal ideal `(p)`). -/
theorem length_ZpQuot (K : ℕ) : Module.length ℤ_[p] (ℤ_[p] ⧸ Ideal.span {(p : ℤ_[p]) ^ K}) = K := by
  rw [← Ideal.span_singleton_pow, ← PadicInt.maximalIdeal_eq_span_p]
  exact IsDiscreteValuationRing.length_quotient_pow_maximalIdeal _ K

/-- For `F` free of rank `r`, `ℓ(F/p^K F) = K r`. -/
theorem length_free_quot (F : Type*) [AddCommGroup F] [Module ℤ_[p] F] [Module.Free ℤ_[p] F]
    [Module.Finite ℤ_[p] F] (K : ℕ) :
    Module.length ℤ_[p] (F ⧸ (pI p ^ K) • (⊤ : Submodule ℤ_[p] F)) =
      ((K * Module.finrank ℤ_[p] F : ℕ) : ℕ∞) := by
  let b := Module.Free.chooseBasis ℤ_[p] F
  let I : Ideal ℤ_[p] := Ideal.span {(p : ℤ_[p]) ^ K}
  let φ : F →ₗ[ℤ_[p]] (Module.Free.ChooseBasisIndex ℤ_[p] F → ℤ_[p] ⧸ I) :=
    LinearMap.pi fun i => I.mkQ ∘ₗ b.coord i
  have hsurj : Function.Surjective φ := by
    intro y
    choose z hz using fun i => Submodule.mkQ_surjective I (y i)
    refine ⟨b.equivFun.symm z, funext fun i => ?_⟩
    have hb : b.repr (b.equivFun.symm z) i = z i := by
      rw [← b.equivFun_apply, LinearEquiv.apply_symm_apply]
    show I.mkQ (b.coord i (b.equivFun.symm z)) = y i
    rw [Module.Basis.coord_apply, hb, hz]
  have hker : LinearMap.ker φ = (pI p ^ K) • (⊤ : Submodule ℤ_[p] F) := by
    ext v
    rw [mem_pI_pow_smul_top, LinearMap.mem_ker]
    constructor
    · intro hv
      have hc : ∀ i, ∃ c, b.repr v i = (p : ℤ_[p]) ^ K * c := fun i => by
        have := congrFun hv i
        change I.mkQ (b.coord i v) = 0 at this
        rw [Module.Basis.coord_apply, Submodule.mkQ_apply, Submodule.Quotient.mk_eq_zero] at this
        exact Ideal.mem_span_singleton.mp this
      choose c hc using hc
      refine ⟨b.equivFun.symm c, b.repr.injective (Finsupp.ext fun i => ?_)⟩
      have hb : b.repr (b.equivFun.symm c) i = c i := by
        rw [← b.equivFun_apply, LinearEquiv.apply_symm_apply]
      rw [hc i, map_smul, Finsupp.smul_apply, hb, smul_eq_mul]
    · rintro ⟨y, rfl⟩
      funext i
      change I.mkQ (b.coord i (((p : ℤ_[p]) ^ K) • y)) = 0
      rw [map_smul, Submodule.mkQ_apply, Submodule.Quotient.mk_eq_zero, smul_eq_mul]
      exact Ideal.mem_span_singleton.mpr (dvd_mul_right _ _)
  rw [← hker, (φ.quotKerEquivOfSurjective hsurj).length_eq, Module.length_pi_of_fintype]
  simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  rw [show Module.length ℤ_[p] (ℤ_[p] ⧸ I) = K from length_ZpQuot K, Module.finrank_eq_card_chooseBasisIndex]
  push_cast; ring

/-- In a module of finite length, `N ≤ N'` with equal lengths forces `N = N'`. -/
theorem eq_of_le_of_length_eq {N N' : Submodule ℤ_[p] M} (hfin : Module.length ℤ_[p] M ≠ ⊤)
    (hle : N ≤ N') (hl : Module.length ℤ_[p] N = Module.length ℤ_[p] N') : N = N' := by
  have hF : IsFiniteLength ℤ_[p] M := Module.length_ne_top_iff.mp hfin
  obtain ⟨h1, h2⟩ := isFiniteLength_iff_isNoetherian_isArtinian.mp hF
  by_contra hne
  have := Submodule.height_strictMono (lt_of_le_of_ne hle hne)
  rw [← Module.length_submodule, ← Module.length_submodule] at this
  exact this.ne hl

end Aux

theorem mem_tsat (x : Module.End ℤ_[p] L) :
    x ∈ tsat act ↔ ∃ k : ℕ, ((p : ℤ_[p]) ^ k) • x ∈ LinearMap.range act.toLinearMap := Iff.rfl

theorem mem_kerK (K : ℕ) (t : T) :
    t ∈ kerK act K ↔ ∀ v, act t v ∈ (pI p ^ K) • (⊤ : Submodule ℤ_[p] L) := by
  simp [kerK, LinearMap.ext_iff, Submodule.Quotient.mk_eq_zero]

theorem tsat_pow_mono {x : Module.End ℤ_[p] L} {c d : ℕ} (hcd : c ≤ d)
    (h : ((p : ℤ_[p]) ^ c) • x ∈ LinearMap.range act.toLinearMap) :
    ((p : ℤ_[p]) ^ d) • x ∈ LinearMap.range act.toLinearMap := by
  obtain ⟨j, rfl⟩ := Nat.exists_eq_add_of_le hcd
  rw [pow_add, mul_comm, mul_smul]
  exact Submodule.smul_mem _ _ h

theorem mem_tsat_of_eq {t : T} {K : ℕ} {e : Module.End ℤ_[p] L} (he : act t = ((p : ℤ_[p]) ^ K) • e) :
    e ∈ tsat act := ⟨K, t, he⟩


variable [Module.Free ℤ_[p] L] [Module.Finite ℤ_[p] L]

/-- Divisibility on a basis: if `f(L) ⊆ p^K L` then `f = p^K e`. -/
theorem exists_eq_pow_smul (K : ℕ) (f : Module.End ℤ_[p] L)
    (hf : ∀ v, f v ∈ (pI p ^ K) • (⊤ : Submodule ℤ_[p] L)) :
    ∃ e : Module.End ℤ_[p] L, f = ((p : ℤ_[p]) ^ K) • e := by
  let b := Module.Free.chooseBasis ℤ_[p] L
  choose w hw using fun i => (mem_pI_pow_smul_top K (f (b i))).mp (hf (b i))
  refine ⟨b.constr ℤ_[p] w, b.ext fun i => ?_⟩
  simp [hw i]

theorem finite_T (hact : Function.Injective act) : Module.Finite ℤ_[p] T :=
  Module.Finite.of_injective act.toLinearMap hact

theorem torsionFree_T (hact : Function.Injective act) : Module.IsTorsionFree ℤ_[p] T :=
  Function.Injective.moduleIsTorsionFree act hact (fun r t => map_smul act r t)

theorem free_T (hact : Function.Injective act) : Module.Free ℤ_[p] T := by
  have := finite_T act hact
  have := torsionFree_T act hact
  exact Module.free_of_finite_type_torsion_free'

theorem cSat_set_nonempty :
    {c : ℕ | ∀ x ∈ tsat act, ((p : ℤ_[p]) ^ c) • x ∈ LinearMap.range act.toLinearMap}.Nonempty := by
  obtain ⟨S, hS⟩ := IsNoetherian.noetherian (tsat act)
  have hk : ∀ s ∈ S, ∃ k : ℕ, ((p : ℤ_[p]) ^ k) • s ∈ LinearMap.range act.toLinearMap := fun s hs =>
    (show s ∈ tsat act from hS ▸ Submodule.subset_span hs)
  choose! k hk using hk
  refine ⟨S.sup k, fun x hx => ?_⟩
  rw [← hS] at hx
  have hle : Submodule.span ℤ_[p] (S : Set (Module.End ℤ_[p] L)) ≤
      (LinearMap.range act.toLinearMap).comap (((p : ℤ_[p]) ^ (S.sup k)) • LinearMap.id) := by
    rw [Submodule.span_le]
    intro s hs
    exact tsat_pow_mono act (Finset.le_sup hs) (hk s hs)
  simpa using hle hx

theorem cSat_spec (x : Module.End ℤ_[p] L) (hx : x ∈ tsat act) :
    ((p : ℤ_[p]) ^ cSat act) • x ∈ LinearMap.range act.toLinearMap :=
  Nat.sInf_mem (cSat_set_nonempty act) x hx

theorem cSat_spec_of_le {K : ℕ} (hK : cSat act ≤ K) (x : Module.End ℤ_[p] L) (hx : x ∈ tsat act) :
    ((p : ℤ_[p]) ^ K) • x ∈ LinearMap.range act.toLinearMap :=
  tsat_pow_mono act hK (cSat_spec act x hx)


/-! ### P15: `prop:saturation` -/

/-- **`prop:saturation`(1)**: `ker(T → End(L/p^K)) = T ∩ p^K End(L) = T ∩ p^K T^sat`. -/
theorem kerK_iff (hact : Function.Injective act) (K : ℕ) (t : T) :
    (t ∈ kerK act K ↔ ∃ e : Module.End ℤ_[p] L, act t = ((p : ℤ_[p]) ^ K) • e) ∧
    (t ∈ kerK act K ↔ ∃ e ∈ tsat act, act t = ((p : ℤ_[p]) ^ K) • e) := by
  have h1 : t ∈ kerK act K ↔ ∃ e : Module.End ℤ_[p] L, act t = ((p : ℤ_[p]) ^ K) • e := by
    rw [mem_kerK]
    refine ⟨exists_eq_pow_smul K _, ?_⟩
    rintro ⟨e, he⟩ v
    rw [he, LinearMap.smul_apply, mem_pI_pow_smul_top]
    exact ⟨e v, rfl⟩
  exact ⟨h1, h1.trans ⟨fun ⟨e, he⟩ => ⟨e, mem_tsat_of_eq act he, he⟩, fun ⟨e, _, he⟩ => ⟨e, he⟩⟩⟩

theorem tsat_finite : Module.Finite ℤ_[p] (tsat act) := inferInstance

theorem tsat_free : Module.Free ℤ_[p] (tsat act) := inferInstance

/-- `T → T^sat`, `t ↦ act t`. -/
def toSat : T →ₗ[ℤ_[p]] tsat act :=
  act.toLinearMap.codRestrict (tsat act) (fun t => ⟨0, t, by simp⟩)

theorem toSat_injective (hact : Function.Injective act) : Function.Injective (toSat act) :=
  fun a b h => hact (congrArg Subtype.val h)

theorem range_toSat : LinearMap.range (toSat act) = tInSat act := by
  ext x
  simp only [LinearMap.mem_range, tInSat, Submodule.mem_comap, Submodule.subtype_apply]
  constructor
  · rintro ⟨t, rfl⟩; exact ⟨t, rfl⟩
  · rintro ⟨t, ht⟩; exact ⟨t, Subtype.ext ht⟩

theorem finrank_tsat (hact : Function.Injective act) :
    Module.finrank ℤ_[p] (tsat act) = rk (p := p) (T := T) := by
  have := finite_T act hact
  apply le_antisymm
  · let g : tsat act →ₗ[ℤ_[p]] tsat act := ((p : ℤ_[p]) ^ cSat act) • LinearMap.id
    have hg : ∀ x, g x ∈ LinearMap.range (toSat act) := fun x => by
      rw [range_toSat]; simpa [g, tInSat] using cSat_spec act x.1 x.2
    have hinj : Function.Injective (g.codRestrict _ hg) := fun a b h => by
      have h' : g a = g b := congrArg Subtype.val h
      exact smul_right_injective _ (pZp_pow_ne_zero (p := p) (cSat act)) h'
    have := LinearMap.finrank_le_finrank_of_injective hinj
    rwa [LinearMap.finrank_range_of_inj (toSat_injective act hact)] at this
  · rw [rk, ← LinearMap.finrank_range_of_inj (toSat_injective act hact)]
    exact Submodule.finrank_le _

theorem length_SatQuot_ne_top (hact : Function.Injective act) :
    Module.length ℤ_[p] (SatQuot act) ≠ ⊤ := by
  have hN : (pI p ^ cSat act) • (⊤ : Submodule ℤ_[p] (tsat act)) ≤ tInSat act := by
    intro x hx
    obtain ⟨y, rfl⟩ := (mem_pI_pow_smul_top _ x).mp hx
    simpa [tInSat] using cSat_spec act y.1 y.2
  have h := length_quot_add _ (tInSat act) hN
  rw [length_free_quot] at h
  intro htop
  rw [htop, add_top] at h
  exact ENat.natCast_ne_top _ h

theorem torsionBy_eq_top {K : ℕ} (hK : cSat act ≤ K) :
    Submodule.torsionBy ℤ_[p] (SatQuot act) ((p : ℤ_[p]) ^ K) = ⊤ := by
  rw [eq_top_iff]
  rintro x -
  induction x using Submodule.Quotient.induction_on with
  | H y =>
    rw [Submodule.mem_torsionBy_iff, ← Submodule.Quotient.mk_smul, Submodule.Quotient.mk_eq_zero]
    simpa [tInSat] using cSat_spec_of_le act hK y.1 y.2

/-- The core of `prop:satdefect`(1): `ℓ(𝖠_K) + ℓ((T^sat/T)[p^K]) = K r`. -/
theorem satdefect_core (hact : Function.Injective act) (K : ℕ) :
    Module.length ℤ_[p] (AK act K) +
      Module.length ℤ_[p] (Submodule.torsionBy ℤ_[p] (SatQuot act) ((p : ℤ_[p]) ^ K)) =
      ((K * rk (p := p) (T := T) : ℕ) : ℕ∞) := by
  have := finite_T act hact
  let ψ := ((pI p ^ K) • (⊤ : Submodule ℤ_[p] (tsat act))).mkQ ∘ₗ toSat act
  have hker : LinearMap.ker ψ = kerK act K := by
    ext t
    rw [LinearMap.mem_ker, (kerK_iff act hact K t).2]
    simp only [ψ, LinearMap.comp_apply, Submodule.mkQ_apply, Submodule.Quotient.mk_eq_zero,
      mem_pI_pow_smul_top]
    constructor
    · rintro ⟨y, hy⟩; exact ⟨y.1, y.2, congrArg Subtype.val hy⟩
    · rintro ⟨e, he, h⟩; exact ⟨⟨e, he⟩, Subtype.ext h⟩
  have hrange : LinearMap.range ψ = (tInSat act).map ((pI p ^ K) • (⊤ : Submodule ℤ_[p] (tsat act))).mkQ := by
    rw [LinearMap.range_comp, range_toSat]
  have hA : Module.length ℤ_[p] (AK act K) =
      Module.length ℤ_[p] ((tInSat act).map ((pI p ^ K) • (⊤ : Submodule ℤ_[p] (tsat act))).mkQ) := by
    rw [← hrange, ← (Submodule.quotEquivOfEq _ _ hker).length_eq]
    exact ψ.quotKerEquivRange.length_eq
  have hadd := length_quot_add ((pI p ^ K) • (⊤ : Submodule ℤ_[p] (tsat act)))
    (tInSat act ⊔ (pI p ^ K) • (⊤ : Submodule ℤ_[p] (tsat act))) le_sup_right
  rw [Submodule.map_sup, Submodule.mkQ_map_self, sup_bot_eq, ← hA, length_free_quot,
    finrank_tsat act hact] at hadd
  have hQ : Module.length ℤ_[p] (tsat act ⧸ (tInSat act ⊔ (pI p ^ K) • (⊤ : Submodule ℤ_[p] (tsat act)))) =
      Module.length ℤ_[p] (Submodule.torsionBy ℤ_[p] (SatQuot act) ((p : ℤ_[p]) ^ K)) := by
    rw [← (Submodule.quotientQuotientEquivQuotientSup (tInSat act) _).length_eq,
      ← smul_top_quot, pI_pow_smul_top, length_torsionBy_eq _ (length_SatQuot_ne_top act hact)]
  rw [← hQ, ← hadd]

/-- **`prop:saturation`(2)**: TFAE (a) `T/pT → 𝖠_1` is injective, i.e. `ker(T → 𝖠_1) ⊆ pT`; (b) `L/pL` is a faithful `T/pT`-module
(`act t (L) ⊆ pL ⇒ t ∈ pT`); (c) `T` is saturated. -/
theorem saturated_tfae (hact : Function.Injective act) :
    List.TFAE [kerK act 1 ≤ pI p • (⊤ : Submodule ℤ_[p] T),
      ∀ t : T, (∀ v : L, act t v ∈ pI p • (⊤ : Submodule ℤ_[p] L)) → t ∈ pI p • (⊤ : Submodule ℤ_[p] T),
      Saturated act] := by
  tfae_have 1 ↔ 2 := by
    simp only [SetLike.le_def, mem_kerK, pow_one]
  tfae_have 2 → 3 := by
    intro hb x hx
    obtain ⟨k, hk⟩ := hx
    induction k generalizing x with
    | zero => simpa using hk
    | succ k ih =>
      obtain ⟨t, ht⟩ := hk
      have ht' : act t = (p : ℤ_[p]) • (((p : ℤ_[p]) ^ k) • x) := by
        rw [← mul_smul, ← pow_succ']; exact ht
      have hv : ∀ v, act t v ∈ pI p • (⊤ : Submodule ℤ_[p] L) := fun v => by
        rw [mem_pI_smul_top, ht']
        exact ⟨(((p : ℤ_[p]) ^ k) • x) v, rfl⟩
      obtain ⟨s, hs⟩ := (mem_pI_smul_top t).mp (hb t hv)
      apply ih
      refine ⟨s, smul_right_injective _ (pZp_ne_zero (p := p)) ?_⟩
      change (p : ℤ_[p]) • act s = _
      rw [← map_smul, ← hs, ht']
  tfae_have 3 → 2 := by
    intro hc t ht
    obtain ⟨e, he⟩ := exists_eq_pow_smul 1 (act t) (by simpa using ht)
    obtain ⟨s, hs⟩ := hc e (mem_tsat_of_eq act he)
    rw [mem_pI_smul_top]
    refine ⟨s, hact ?_⟩
    rw [map_smul, show act s = e from hs, he, pow_one]
  tfae_finish

/-- **`prop:saturation`(2)**, the sufficient condition: if `L` contains `N = T·v ≅ T` (free of rank one) with `L/N`
torsion-free, then `T` is saturated. -/
theorem saturated_of_free_rank_one (hact : Function.Injective act) (v : L)
    (hv : Function.Injective fun t : T => act t v)
    (htf : ∀ (c : ℤ_[p]) (w : L), c ≠ 0 → c • w ∈ Set.range (fun t : T => act t v) →
      w ∈ Set.range (fun t : T => act t v)) :
    Saturated act := by
  intro x hx
  obtain ⟨k, t, ht⟩ := hx
  have ht' : act t = ((p : ℤ_[p]) ^ k) • x := ht
  obtain ⟨s, hs⟩ := htf _ (x v) (pZp_pow_ne_zero k) ⟨t, by simp [ht']⟩
  have hts : t = ((p : ℤ_[p]) ^ k) • s := hv (by simp [map_smul, hs, ht'])
  refine ⟨s, smul_right_injective _ (pZp_pow_ne_zero (p := p) k) ?_⟩
  change ((p : ℤ_[p]) ^ k) • act s = _
  rw [← map_smul, ← hts, ht']

/-- **`prop:saturation`(3), the stopping rule**: `dim 𝖠_K/p ≤ r`; TFAE (a) `dim 𝖠_K/p = r`, (b) `T/pT → 𝖠_K/p𝖠_K` is
injective (hence an isomorphism), (c) `T ∩ p^K T^sat ⊆ pT`. -/
theorem stopping_rule (hact : Function.Injective act) (K : ℕ) (hK : 1 ≤ K) :
    Module.length ℤ_[p] (AK act K ⧸ pI p • (⊤ : Submodule ℤ_[p] (AK act K))) ≤ rk (p := p) (T := T) ∧
    List.TFAE [Module.length ℤ_[p] (AK act K ⧸ pI p • (⊤ : Submodule ℤ_[p] (AK act K))) = rk (p := p) (T := T),
      Function.Injective (redMap act K),
      kerK act K ≤ pI p • (⊤ : Submodule ℤ_[p] T)] := by
  have := finite_T act hact
  have := free_T act hact
  have hAK : Module.length ℤ_[p] (AK act K ⧸ pI p • (⊤ : Submodule ℤ_[p] (AK act K))) =
      Module.length ℤ_[p] (T ⧸ (kerK act K ⊔ pI p • (⊤ : Submodule ℤ_[p] T))) := by
    rw [smul_top_quot]
    exact (Submodule.quotientQuotientEquivQuotientSup _ _).length_eq
  have hT : Module.length ℤ_[p] (T ⧸ pI p • (⊤ : Submodule ℤ_[p] T)) = rk (p := p) (T := T) := by
    have h := length_free_quot (p := p) T 1
    rw [pow_one, one_mul] at h
    exact h
  have hadd := length_quot_add (pI p • (⊤ : Submodule ℤ_[p] T))
    (kerK act K ⊔ pI p • (⊤ : Submodule ℤ_[p] T)) le_sup_right
  rw [hT, ← hAK] at hadd
  have hzero : Module.length ℤ_[p] ((kerK act K ⊔ pI p • (⊤ : Submodule ℤ_[p] T)).map
      (pI p • (⊤ : Submodule ℤ_[p] T)).mkQ) = 0 ↔ kerK act K ≤ pI p • (⊤ : Submodule ℤ_[p] T) := by
    rw [Module.length_eq_zero_iff, Submodule.subsingleton_iff_eq_bot, map_mkQ_eq_bot_iff, sup_le_iff]
    exact ⟨fun h => h.1, fun h => ⟨h, le_rfl⟩⟩
  have hinj : Function.Injective (redMap act K) ↔ kerK act K ≤ pI p • (⊤ : Submodule ℤ_[p] T) := by
    rw [← LinearMap.ker_eq_bot, redMap, Submodule.ker_mapQ, map_mkQ_eq_bot_iff, smul_top_quot,
      Submodule.comap_map_mkQ, sup_le_iff]
    exact ⟨fun h => h.1, fun h => ⟨h, le_rfl⟩⟩
  refine ⟨by rw [hadd]; exact le_add_self, ?_⟩
  tfae_have 1 ↔ 3 := by
    rw [← hzero]
    constructor
    · intro h
      rw [h] at hadd
      have h0 := enat_add_cancel (a := Module.length ℤ_[p] _) (b := 0) (by rw [zero_add]; exact hadd.symm)
      exact h0
    · intro h
      rw [hadd, h, zero_add]
  tfae_have 2 ↔ 3 := hinj
  tfae_finish

/-- **`prop:saturation`(3)**, monotonicity and the bound: if `T ∩ p^K T^sat ⊆ pT` at `K`, then at every `K' ≥ K`; and it
holds for every `K ≥ c^sat + 1`. -/
theorem stopping_rule_mono (hact : Function.Injective act) :
    (∀ K K', K ≤ K' → kerK act K ≤ pI p • (⊤ : Submodule ℤ_[p] T) →
      kerK act K' ≤ pI p • (⊤ : Submodule ℤ_[p] T)) ∧
    ∀ K, cSat act + 1 ≤ K → kerK act K ≤ pI p • (⊤ : Submodule ℤ_[p] T) := by
  refine ⟨fun K K' hKK' h => le_trans (fun t ht => ?_) h, fun K hK t ht => ?_⟩
  · rw [mem_kerK] at ht ⊢
    intro v
    exact Submodule.smul_mono_left (Ideal.pow_le_pow_right hKK') (ht v)
  · obtain ⟨e, he⟩ := (kerK_iff act hact K t).1.mp ht
    obtain ⟨s, hs⟩ := cSat_spec_of_le act (K := K - 1) (by omega) e (mem_tsat_of_eq act he)
    rw [mem_pI_smul_top]
    refine ⟨s, hact ?_⟩
    rw [map_smul, show act s = _ from hs, he, ← mul_smul, ← pow_succ', Nat.sub_add_cancel (by omega)]

/-- **`prop:saturation`(4)**: for `K ≥ c^sat`, `T ∩ p^K T^sat = p^K T^sat` and `length(𝖠_K) = K r − δ^sat`. -/
theorem length_AK_of_ge (hact : Function.Injective act) (K : ℕ) (hK : cSat act ≤ K) :
    (∀ e ∈ tsat act, ((p : ℤ_[p]) ^ K) • e ∈ LinearMap.range act.toLinearMap) ∧
    Module.length ℤ_[p] (AK act K) + deltaSat act = K * rk (p := p) (T := T) := by
  refine ⟨fun e he => cSat_spec_of_le act hK e he, ?_⟩
  have h := satdefect_core act hact K
  rw [torsionBy_eq_top act hK, Module.length_top] at h
  rw [deltaSat, h]
  push_cast
  ring

/-- **`prop:saturation`(5)**, first sentence: every maximal ideal of `T` contains `p`. (The second sentence — the
isomorphism of (3)(b) carries `𝔪`-adic filtrations to `𝔪`-adic filtrations, and everything localises over
`T = ∏_𝔪 T_𝔪` — is recorded in [GT] and not stated in Lean here.) -/
theorem maximal_ideal_contains_p (hact : Function.Injective act) (m : Ideal T) (hm : m.IsMaximal) :
    algebraMap ℤ_[p] T p ∈ m := by
  have := finite_T act hact
  have : Algebra.IsIntegral ℤ_[p] T := Algebra.IsIntegral.of_finite _ _
  have hmax := Ideal.isMaximal_comap_of_isIntegral_of_isMaximal (R := ℤ_[p]) m
  have hp : (p : ℤ_[p]) ∈ Ideal.comap (algebraMap ℤ_[p] T) m := by
    rw [IsLocalRing.eq_maximalIdeal hmax, PadicInt.maximalIdeal_eq_span_p]
    exact Ideal.subset_span rfl
  simpa using hp

/-! ### P17: `prop:satdefect` -/

/-- **`prop:satdefect`(1)**: `ε^sat_K = length((T^sat/T)[p^K])`, written without subtraction:
`length(𝖠_K) + length((T^sat/T)[p^K]) = K r`. -/
theorem satdefect_eq (hact : Function.Injective act) (K : ℕ) :
    Module.length ℤ_[p] (AK act K) +
      Module.length ℤ_[p] (Submodule.torsionBy ℤ_[p] (SatQuot act) ((p : ℤ_[p]) ^ K)) =
      K * rk (p := p) (T := T) := by
  rw [satdefect_core act hact K]
  push_cast
  ring

/-- **`prop:satdefect`(2)**: `ε_K ≤ ε_{K+1} ≤ δ^sat`; `ε_K = δ^sat` for `K ≥ c^sat`; a plateau is final. (Here
`ε_K = length((T^sat/T)[p^K])`, by (1).) -/
theorem torsionBy_mono (K : ℕ) :
    Submodule.torsionBy ℤ_[p] (SatQuot act) ((p : ℤ_[p]) ^ K) ≤
      Submodule.torsionBy ℤ_[p] (SatQuot act) ((p : ℤ_[p]) ^ (K + 1)) := fun x hx => by
  rw [Submodule.mem_torsionBy_iff] at hx ⊢
  rw [pow_succ', mul_smul, hx, smul_zero]

theorem torsionBy_stable {K : ℕ}
    (h : Submodule.torsionBy ℤ_[p] (SatQuot act) ((p : ℤ_[p]) ^ K) =
      Submodule.torsionBy ℤ_[p] (SatQuot act) ((p : ℤ_[p]) ^ (K + 1))) (j : ℕ) :
    Submodule.torsionBy ℤ_[p] (SatQuot act) ((p : ℤ_[p]) ^ (K + j)) =
      Submodule.torsionBy ℤ_[p] (SatQuot act) ((p : ℤ_[p]) ^ K) := by
  induction j with
  | zero => rfl
  | succ j ih =>
    apply le_antisymm
    · intro x hx
      have hx' : (p : ℤ_[p]) ^ (K + j) • ((p : ℤ_[p]) • x) = 0 := by
        rw [← mul_smul, ← pow_succ]
        exact (Submodule.mem_torsionBy_iff _ _).mp hx
      have h1 : (p : ℤ_[p]) • x ∈ Submodule.torsionBy ℤ_[p] (SatQuot act) ((p : ℤ_[p]) ^ (K + j)) :=
        (Submodule.mem_torsionBy_iff _ _).mpr hx'
      rw [ih, Submodule.mem_torsionBy_iff, ← mul_smul, ← pow_succ] at h1
      rw [h]
      exact (Submodule.mem_torsionBy_iff _ _).mpr h1
    · rw [← ih]
      exact torsionBy_mono act (K + j)

theorem satdefect_mono (hact : Function.Injective act) :
    let ε := fun K : ℕ => Module.length ℤ_[p] (Submodule.torsionBy ℤ_[p] (SatQuot act) ((p : ℤ_[p]) ^ K))
    (∀ K, ε K ≤ ε (K + 1) ∧ ε (K + 1) ≤ deltaSat act) ∧
    (∀ K, cSat act ≤ K → ε K = deltaSat act) ∧
    (∀ K, ε K = ε (K + 1) → ε K = deltaSat act) := by
  intro ε
  have hfin := length_SatQuot_ne_top act hact
  have hleδ : ∀ K, ε K ≤ deltaSat act := fun K =>
    Module.length_le_of_injective (Submodule.subtype _) Subtype.val_injective
  have hδ : ∀ K, cSat act ≤ K → ε K = deltaSat act := fun K hK => by
    show Module.length ℤ_[p] (Submodule.torsionBy ℤ_[p] (SatQuot act) ((p : ℤ_[p]) ^ K)) = deltaSat act
    rw [torsionBy_eq_top act hK, Module.length_top]
    rfl
  refine ⟨fun K => ⟨Module.length_le_of_injective (Submodule.inclusion (torsionBy_mono act K))
      (Submodule.inclusion_injective _), hleδ (K + 1)⟩, hδ, fun K hK => ?_⟩
  have heq := eq_of_le_of_length_eq hfin (torsionBy_mono act K) hK
  have hst := torsionBy_stable act heq (cSat act)
  show Module.length ℤ_[p] (Submodule.torsionBy ℤ_[p] (SatQuot act) ((p : ℤ_[p]) ^ K)) = deltaSat act
  rw [← hst, torsionBy_eq_top act (Nat.le_add_left _ _), Module.length_top]
  rfl

/-- **`prop:satdefect`(3)**: `δ^sat = 0` iff `T` is saturated iff `ε^sat_1 = 0`. -/
theorem satdefect_zero_iff (hact : Function.Injective act) :
    (deltaSat act = 0 ↔ Saturated act) ∧ (Saturated act ↔ epsSat act 1 = 0) := by
  have h1 : deltaSat act = 0 ↔ Saturated act := by
    rw [deltaSat, Module.length_eq_zero_iff, Submodule.Quotient.subsingleton_iff]
    constructor
    · intro h x hx
      have : (⟨x, hx⟩ : tsat act) ∈ tInSat act := h ▸ Submodule.mem_top
      exact this
    · intro h
      rw [eq_top_iff]
      rintro ⟨x, hx⟩ -
      exact h x hx
  refine ⟨h1, h1.symm.trans ?_⟩
  obtain ⟨hmono, -, hplat⟩ := satdefect_mono act hact
  have hε0 : Module.length ℤ_[p] (Submodule.torsionBy ℤ_[p] (SatQuot act) ((p : ℤ_[p]) ^ 0)) = 0 := by
    rw [Module.length_eq_zero_iff, Submodule.subsingleton_iff_eq_bot, eq_bot_iff]
    intro x hx
    rw [Submodule.mem_torsionBy_iff, pow_zero, one_smul] at hx
    simp [hx]
  have hδε : deltaSat act = 0 ↔
      Module.length ℤ_[p] (Submodule.torsionBy ℤ_[p] (SatQuot act) ((p : ℤ_[p]) ^ 1)) = 0 := by
    constructor
    · intro h
      have h' := (hmono 0).2
      rw [h] at h'
      exact le_antisymm h' zero_le
    · intro h
      have h' := hplat 0 (hε0.trans h.symm)
      rw [← h']
      exact hε0
  rw [hδε]
  have hc := satdefect_core act hact 1
  have hA : Module.length ℤ_[p] (AK act 1) ≠ ⊤ := fun h => by
    rw [h, top_add] at hc; exact ENat.natCast_ne_top _ hc.symm
  have hE : Module.length ℤ_[p] (Submodule.torsionBy ℤ_[p] (SatQuot act) ((p : ℤ_[p]) ^ 1)) ≠ ⊤ :=
    fun h => by rw [h, add_top] at hc; exact ENat.natCast_ne_top _ hc.symm
  obtain ⟨a, ha⟩ := ENat.ne_top_iff_exists.mp hA
  obtain ⟨e, he⟩ := ENat.ne_top_iff_exists.mp hE
  rw [← ha, ← he] at hc
  norm_cast at hc
  rw [← he, epsSat, ← ha, ENat.toNat_natCast, Nat.cast_eq_zero]
  push_cast
  omega

end GL2.Sat

end
