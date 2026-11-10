import GL2.Cone.Complex

/-!
# G3, layer B: the pair, coset transport, the pair homotopy `η` and the cone action (L16-6)

[GT] chapter `ch:cone`, mirroring [C] v1.16.0-pre.3 `def:pair`, `lem:conerelative`(1),(4), `def:cosettransport`,
`lem:cosettransport`, `def:pairhomotopy`, `lem:etaexists`, `lem:etaunique`, `def:coneaction`, `lem:derivedhomotopy`,
`thm:conewelldef`(a),(b),(d), `prop:coneindep`(1).

**The model is abstract** ([LP] §2.3, risk "the model is the wrong model"): `bulk` and `bdry` are `ℤ`-graded chain
complexes of abelian groups with a `Γ`-action (the simplicial chains of `X_bulk`, `X_∂`), `j` an equivariant chain map,
and [C]'s geometric hypotheses enter only through the named structure `PairHyp` (H4(ii) through `eq:contractibleuse`,
B2, and the orbit-representative freeness B1 + B3 give the (T1)-extension property).  The 7-vertex model, a trivial
centre or Bredon-only semantics are hard-wired nowhere: `Γ` is any subgroup of any group `G`, `Δ ⊇ Γ` any submonoid
acting on the coefficients `V` through `ρ`, and `−I` is not mentioned.
-/

namespace GL2.Cone

universe u v w

/-- A `ℤ`-graded chain complex of abelian groups with a `Γ`-action, `∂ q : X (q+1) → X q`, and an augmentation
`ε : X 0 → ℤ` (the simplicial chains of a `Γ`-complex). -/
structure GChainCx (Γ : Type w) [Group Γ] where
  X : ℤ → Type u
  [grp : ∀ q, AddCommGroup (X q)]
  [act : ∀ q, DistribMulAction Γ (X q)]
  bd : ∀ q, X (q + 1) →+ X q
  bd_equiv : ∀ q (γ : Γ) (x : X (q + 1)), bd q (γ • x) = γ • bd q x
  bd_bd : ∀ q (x : X (q + 1 + 1)), bd q (bd (q + 1) x) = 0
  aug : X 0 →+ ℤ
  aug_bd : ∀ x : X (0 + 1), aug (bd 0 x) = 0
  aug_equiv : ∀ (γ : Γ) (x : X 0), aug (γ • x) = aug x

attribute [instance] GChainCx.grp GChainCx.act

variable {G : Type w} [Group G] (Γ : Subgroup G) (Δ : Submonoid G) (hΓΔ : ∀ γ : Γ, (γ : G) ∈ Δ)
  {R : Type v} [CommRing R] {V : Type u} [AddCommGroup V] [Module R V] (ρ : Δ →* Module.End R V)

/-- `Γ` acting on `V` through `ρ`. -/
def ρΓ (γ : Γ) : Module.End R V := ρ ⟨γ, hΓΔ γ⟩

/-- `Hom_{ℤ[Γ]}(Y_q, V)`: the `Γ`-equivariant additive maps. -/
def eqHom (Y : GChainCx.{u} Γ) (q : ℤ) : Submodule R (Y.X q →+ V) where
  carrier := {φ | ∀ (γ : Γ) (y : Y.X q), φ (γ • y) = ρΓ Γ Δ hΓΔ ρ γ (φ y)}
  zero_mem' := by intro γ y; simp
  add_mem' := by
    intro φ ψ hφ hψ γ y
    simp [hφ γ y, hψ γ y]
  smul_mem' := by
    intro c φ hφ γ y
    simp [hφ γ y]

/-- The equivariant cochain complex `Hom_{ℤ[Γ]}(C_•(Y), V)`, `d φ = φ ∘ ∂` ([C] `def:pair`). -/
def eqCochains (Y : GChainCx.{u} Γ) : CochainCx.{u} R where
  X q := eqHom Γ Δ hΓΔ ρ Y q
  d q :=
    { toFun := fun φ => ⟨φ.1.comp (Y.bd q), fun γ y => by
        simp only [AddMonoidHom.comp_apply, Y.bd_equiv]; exact φ.2 γ _⟩
      map_add' := fun _ _ => rfl
      map_smul' := fun _ _ => rfl }
  d_comp_d q := by
    ext φ y
    simp [Y.bd_bd]

/-- **`def:pair`**: the pair `(X_bulk, X_∂)`, `j` an equivariant chain map commuting with the augmentations. -/
structure PairData (Γ : Subgroup G) where
  bulk : GChainCx.{u} Γ
  bdry : GChainCx.{u} Γ
  j : ∀ q, bdry.X q →+ bulk.X q
  j_equiv : ∀ q (γ : Γ) (x : bdry.X q), j q (γ • x) = γ • j q x
  j_bd : ∀ q (x : bdry.X (q + 1)), bulk.bd q (j (q + 1) x) = j q (bdry.bd q x)

variable {Γ Δ hΓΔ ρ}

/-- **`def:pair`**: the restriction `r = j^*`, `φ ↦ φ ∘ j`. -/
def restr (P : PairData.{u} Γ) : GMap (eqCochains Γ Δ hΓΔ ρ P.bulk) (eqCochains Γ Δ hΓΔ ρ P.bdry) := fun q =>
  { toFun := fun φ => ⟨φ.1.comp (P.j q), fun γ y => by
      simp only [AddMonoidHom.comp_apply, P.j_equiv]; exact φ.2 γ _⟩
    map_add' := fun _ _ => rfl
    map_smul' := fun _ _ => rfl }

/-- `r` is a cochain map. -/
theorem restr_isChainMap (P : PairData.{u} Γ) : IsChainMap (restr (Δ := Δ) (hΓΔ := hΓΔ) (ρ := ρ) P) := by
  intro q
  apply LinearMap.ext; intro φ; apply Subtype.ext; apply AddMonoidHom.ext; intro y
  show φ.1 (P.j q (P.bdry.bd q y)) = φ.1 (P.bulk.bd q (P.j (q + 1) y))
  rw [P.j_bd]

/-- **`lem:conerelative`(1)**: if `j` has an equivariant additive retraction `e` (extension by zero: `e` forgets the
simplices not in `X_∂`), then `r` is onto in every degree; `φ ↦ φ ∘ e` is a section, not a cochain map. -/
theorem restr_surjective (P : PairData.{u} Γ) (e : ∀ q, P.bulk.X q →+ P.bdry.X q)
    (he : ∀ q (γ : Γ) x, e q (γ • x) = γ • e q x) (hej : ∀ q x, e q (P.j q x) = x) (q : ℤ) :
    Function.Surjective (restr (Δ := Δ) (hΓΔ := hΓΔ) (ρ := ρ) P q) := by
  intro φ
  refine ⟨⟨φ.1.comp (e q), fun γ y => by simp only [AddMonoidHom.comp_apply, he]; exact φ.2 γ _⟩, ?_⟩
  apply Subtype.ext; apply AddMonoidHom.ext; intro y
  show φ.1 (e q (P.j q y)) = φ.1 y
  rw [hej]

/-! ### Twisted maps and coset transport -/

variable (Γ) in
/-- **(T1)**: `F(h · x) = (g⁻¹ h g) · F(x)` for `h ∈ Γ'' = Γ ∩ g Γ g⁻¹`. -/
def T1 (g : G) {A B : Type*} [AddCommGroup A] [AddCommGroup B] [DistribMulAction Γ A] [DistribMulAction Γ B]
    (F : A →+ B) : Prop :=
  ∀ (h h' : Γ), (h' : G) = g⁻¹ * h * g → ∀ x, F (h • x) = h' • F x

variable (Γ) in
/-- **`def:tau`**: a `g`-twisted chain datum on `Y`: (T1), (T2) `∂τ = τ∂`, (T3) `ε τ = ε`. -/
structure TwistedChainDatum (g : G) (Y : GChainCx.{u} Γ) where
  τ : ∀ q, Y.X q →+ Y.X q
  t1 : ∀ q, T1 Γ g (τ q)
  t2 : ∀ q x, Y.bd q (τ (q + 1) x) = τ q (Y.bd q x)
  t3 : ∀ x, Y.aug (τ 0 x) = Y.aug x

variable (Γ Δ) in
/-- Coset data for `Γ g Γ`: `s_1, …, s_m` represent `Γ''\Γ`, `δ_j = s_j⁻¹ g ∈ Δ` ([C] `def:hecke`). -/
structure CosetData (g : G) where
  m : ℕ
  s : Fin m → Γ
  δ_mem : ∀ i, ((s i : G)⁻¹ * g) ∈ Δ
  cosets : ∀ γ : Γ, ∃! i : Fin m, ∃ h : Γ, g⁻¹ * (h : G) * g ∈ Γ ∧ (γ : G) = h * s i

variable {g : G}

variable (Γ Δ) in
/-- **`def:cosettransport`** (`eq:cosettransport`): `(F̃ φ)(y) = ∑_j ρ(δ_j) φ(F(s_j · y))`. -/
def transport (cd : CosetData Γ Δ g) {A B : Type*} [AddCommGroup A] [AddCommGroup B] [DistribMulAction Γ A]
    (F : A →+ B) (φ : B →+ V) : A →+ V :=
  ∑ i, (ρ ⟨(cd.s i : G)⁻¹ * g, cd.δ_mem i⟩).toAddMonoidHom.comp (φ.comp (F.comp (DistribSMul.toAddMonoidHom A (cd.s i))))

/-- `φ ↦ F̃ φ` is `R`-linear. -/
def transportL (cd : CosetData Γ Δ g) {A B : Type*} [AddCommGroup A] [AddCommGroup B] [DistribMulAction Γ A]
    (F : A →+ B) : (B →+ V) →ₗ[R] (A →+ V) where
  toFun φ := transport (ρ := ρ) Γ Δ cd F φ
  map_add' φ ψ := by ext y; simp [transport, Finset.sum_add_distrib]
  map_smul' c φ := by ext y; simp [transport, Finset.smul_sum]

/-- **`lem:cosettransport`(1)**: for `F` with (T1) and `φ` equivariant, `F̃ φ` is equivariant; `F ↦ F̃` is additive. -/
theorem transport_equivariant (cd : CosetData Γ Δ g) {Y Y' : GChainCx.{u} Γ} {a b : ℤ} (F : Y.X a →+ Y'.X b)
    (hF : T1 Γ g F) (φ : Y'.X b →+ V) (hφ : φ ∈ eqHom Γ Δ hΓΔ ρ Y' b) :
    transport (ρ := ρ) Γ Δ cd F φ ∈ eqHom Γ Δ hΓΔ ρ Y a ∧
      ∀ F' : Y.X a →+ Y'.X b, transport (ρ := ρ) Γ Δ cd (F + F') φ =
        transport (ρ := ρ) Γ Δ cd F φ + transport (ρ := ρ) Γ Δ cd F' φ := by
  classical
  refine ⟨fun γ y => ?_, fun F' => by ext y; simp [transport, Finset.sum_add_distrib]⟩
  have hc : ∀ i, ∃ k, ∃ h : Γ, g⁻¹ * (h : G) * g ∈ Γ ∧ ((cd.s i * γ : Γ) : G) = h * cd.s k :=
    fun i => (cd.cosets (cd.s i * γ)).exists
  choose f hf using hc
  choose hh hhmem hheq using hf
  have finj : Function.Injective f := by
    intro i i' e
    have h1 := hheq i
    have h2 := hheq i'
    rw [← e] at h2
    simp only [Subgroup.coe_mul] at h1 h2
    have hsi : (cd.s i : G) = (hh i * (hh i')⁻¹ : Γ) * cd.s i' := by
      simp only [Subgroup.coe_mul, Subgroup.coe_inv]
      have : (cd.s i : G) = hh i * cd.s (f i) * (γ : G)⁻¹ := by rw [← h1]; group
      rw [this]
      have : (cd.s i' : G) = hh i' * cd.s (f i) * (γ : G)⁻¹ := by rw [← h2]; group
      rw [this]; group
    have hmem : g⁻¹ * ((hh i * (hh i')⁻¹ : Γ) : G) * g ∈ Γ := by
      have := Γ.mul_mem (hhmem i) (Γ.inv_mem (hhmem i'))
      convert this using 1
      simp only [Subgroup.coe_mul, Subgroup.coe_inv]; group
    exact (cd.cosets (cd.s i)).unique ⟨1, by simp, by simp⟩ ⟨_, hmem, hsi⟩
  let ef := Equiv.ofBijective f (Finite.injective_iff_bijective.mp finj)
  simp only [transport, AddMonoidHom.finsetSum_apply, AddMonoidHom.comp_apply,
    DistribSMul.toAddMonoidHom_apply, LinearMap.toAddMonoidHom_coe, map_sum]
  conv_rhs => rw [← Equiv.sum_comp ef]
  refine Finset.sum_congr rfl fun i _ => ?_
  have e1 : cd.s i • γ • y = hh i • cd.s (f i) • y := by
    rw [← mul_smul, ← mul_smul]
    congr 1
    exact Subtype.ext (hheq i)
  have e2 : (ef i : Fin cd.m) = f i := rfl
  rw [e2, e1, hF (hh i) ⟨g⁻¹ * hh i * g, hhmem i⟩ rfl, hφ]
  simp only [ρΓ]
  rw [← Module.End.mul_apply, ← map_mul, ← Module.End.mul_apply, ← map_mul]
  congr 2
  apply Subtype.ext
  simp only [Submonoid.coe_mul]
  have h1 := hheq i
  simp only [Subgroup.coe_mul] at h1
  have : (hh i : G) = cd.s i * γ * (cd.s (f i) : G)⁻¹ := by rw [h1]; group
  rw [this]; group

/-- **`lem:cosettransport`(2)**: `(∂ ∘ F)~ φ = F̃(dφ)` and `(F ∘ ∂)~ φ = d(F̃ φ)`. -/
theorem transport_bd (cd : CosetData Γ Δ g) {Y Y' : GChainCx.{u} Γ} {a b : ℤ} :
    (∀ (F : Y.X a →+ Y'.X (b + 1)) (φ : Y'.X b →+ V),
      transport (ρ := ρ) Γ Δ cd ((Y'.bd b).comp F) φ = transport (ρ := ρ) Γ Δ cd F (φ.comp (Y'.bd b))) ∧
    (∀ (F : Y.X a →+ Y'.X b) (φ : Y'.X b →+ V),
      transport (ρ := ρ) Γ Δ cd (F.comp (Y.bd a)) φ = (transport (ρ := ρ) Γ Δ cd F φ).comp (Y.bd a)) := by
  constructor
  · intro F φ; rfl
  · intro F φ
    ext y
    simp [transport, Y.bd_equiv]

/-- **`lem:cosettransport`(3),(4)**: `(j ∘ G)~ φ = G̃(r φ)` and `(G ∘ j)~ φ = r(G̃ φ)`. -/
theorem transport_j (cd : CosetData Γ Δ g) (P : PairData.{u} Γ) {a b : ℤ} :
    (∀ (Gm : P.bdry.X a →+ P.bdry.X b) (φ : P.bulk.X b →+ V),
      transport (ρ := ρ) Γ Δ cd ((P.j b).comp Gm) φ = transport (ρ := ρ) Γ Δ cd Gm (φ.comp (P.j b))) ∧
    (∀ (Gm : P.bulk.X a →+ P.bulk.X b) (φ : P.bulk.X b →+ V),
      transport (ρ := ρ) Γ Δ cd (Gm.comp (P.j a)) φ = (transport (ρ := ρ) Γ Δ cd Gm φ).comp (P.j a)) := by
  constructor
  · intro Gm φ; rfl
  · intro Gm φ
    ext y
    simp [transport, P.j_equiv]

/-! ### The pair homotopy -/

variable (Γ) in
/-- **`def:pairhomotopy`** (`eq:pairhomotopy`): `η : C_•(X_∂) → C_{•+1}(X_bulk)` with (T1) and
`∂η + η∂ = j τ_∂ − τ j`. -/
structure IsPairHomotopy (P : PairData.{u} Γ) (τ : TwistedChainDatum Γ g P.bulk) (τb : TwistedChainDatum Γ g P.bdry)
    (η : ∀ q, P.bdry.X q →+ P.bulk.X (q + 1)) : Prop where
  t1 : ∀ q, T1 Γ g (η q)
  htpy : ∀ q (y : P.bdry.X (q + 1)),
    P.bulk.bd (q + 1) (η (q + 1) y) + η q (P.bdry.bd q y) = P.j (q + 1) (τb.τ (q + 1) y) - τ.τ (q + 1) (P.j (q + 1) y)

variable (Γ) in
/-- [C]'s standing geometric hypotheses on the pair, as used by `sec:cone`: `eq:contractibleuse` (from H4(ii)),
the dimensions (H4(ii), B2), vanishing in negative degrees, and the (T1)-extension property of the boundary chains
(B1 + B3: the boundary simplices have central stabilisers acting trivially, so a (T1)-map is freely determined by its
values on orbit representatives). -/
structure PairHyp (P : PairData.{u} Γ) (g : G) : Prop where
  neg : ∀ q, q < 0 → Subsingleton (P.bulk.X q) ∧ Subsingleton (P.bdry.X q)
  contract0 : ∀ x : P.bulk.X 0, P.bulk.aug x = 0 → ∃ y, P.bulk.bd 0 y = x
  contract1 : ∀ x : P.bulk.X (0 + 1), P.bulk.bd 0 x = 0 → ∃ y, P.bulk.bd (0 + 1) y = x
  contract2 : ∀ x : P.bulk.X (0 + 1 + 1), P.bulk.bd (0 + 1) x = 0 → x = 0
  bulk_dim : ∀ q, 3 ≤ q → Subsingleton (P.bulk.X q)
  bdry_dim : ∀ q, 2 ≤ q → Subsingleton (P.bdry.X q)
  free_twisted : ∀ q, ∃ (I : Type u) (rep : I → P.bdry.X q), ∀ b (val : I → P.bulk.X b),
    ∃! F : P.bdry.X q →+ P.bulk.X b, T1 Γ g F ∧ ∀ i, F (rep i) = val i

/-! #### (T1) closure lemmas (L18-4) -/

section T1Lemmas

variable {A B C : Type*} [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] [DistribMulAction Γ A]
  [DistribMulAction Γ B] [DistribMulAction Γ C]

lemma T1.comp_left {F : A →+ B} (hF : T1 Γ g F) {Gm : B →+ C} (hG : ∀ (γ : Γ) x, Gm (γ • x) = γ • Gm x) :
    T1 Γ g (Gm.comp F) := fun h h' hh x => by
  simp only [AddMonoidHom.comp_apply, hF h h' hh x, hG]

lemma T1.comp_right {F : B →+ C} (hF : T1 Γ g F) {Gm : A →+ B} (hG : ∀ (γ : Γ) x, Gm (γ • x) = γ • Gm x) :
    T1 Γ g (F.comp Gm) := fun h h' hh x => by
  simp only [AddMonoidHom.comp_apply, hG, hF h h' hh]

lemma T1.sub {F F' : A →+ B} (hF : T1 Γ g F) (hF' : T1 Γ g F') : T1 Γ g (F - F') := fun h h' hh x => by
  simp only [AddMonoidHom.sub_apply, hF h h' hh x, hF' h h' hh x, smul_sub]

lemma T1.add {F F' : A →+ B} (hF : T1 Γ g F) (hF' : T1 Γ g F') : T1 Γ g (F + F') := fun h h' hh x => by
  simp only [AddMonoidHom.add_apply, hF h h' hh x, hF' h h' hh x, smul_add]

lemma T1.zero : T1 Γ g (0 : A →+ B) := fun h h' hh x => by simp

end T1Lemmas

/-- **`lem:etaexists`**: under [C]'s hypotheses a pair homotopy exists over `ℤ`, for every `ℓ` (no stabiliser order
inverted, no system solved over anything but `ℤ`).

**Restated (L18-4).** The hypothesis `hjaug` (`ε ∘ j = ε_∂`: `j` commutes with the augmentations, [C] `def:pair`) was
missing from the skeleton, and without it the statement is false: `Γ = 1`, `bulk = bdry = ℤ` in degree `0`, `j = id`,
`ε_∂ = 0`, `ε = id`, `τ = id`, `τ_∂ = 2` satisfy `PairHyp`, (T1)–(T3), and `∂η₀ = jτ_∂ − τj = id` has no solution since
`C₁(X_bulk) = 0`. [C] `lem:etaexists`(V) uses exactly this (`ε(jτ_∂x) − ε(τjx) = 1 − 1`); no [C] erratum. -/
theorem eta_exists (P : PairData.{u} Γ) (hP : PairHyp Γ P g) (τ : TwistedChainDatum Γ g P.bulk)
    (τb : TwistedChainDatum Γ g P.bdry) (hjaug : ∀ x, P.bulk.aug (P.j 0 x) = P.bdry.aug x) :
    ∃ η, IsPairHomotopy Γ P τ τb η := by
  classical
  have hj : ∀ q (γ : Γ) x, P.j q (γ • x) = γ • P.j q x := P.j_equiv
  have hbd : ∀ q (γ : Γ) (x : P.bulk.X (q + 1)), P.bulk.bd q (γ • x) = γ • P.bulk.bd q x := P.bulk.bd_equiv
  have hbdb : ∀ q (γ : Γ) (x : P.bdry.X (q + 1)), P.bdry.bd q (γ • x) = γ • P.bdry.bd q x := P.bdry.bd_equiv
  -- the defect `j τ_∂ − τ j` in degree `q`, a (T1) map
  let δ : ∀ q, P.bdry.X q →+ P.bulk.X q := fun q => (P.j q).comp (τb.τ q) - (τ.τ q).comp (P.j q)
  have hδ : ∀ q, T1 Γ g (δ q) := fun q =>
    T1.sub ((τb.t1 q).comp_left (hj q)) ((τ.t1 q).comp_right (hj q))
  have hδbd : ∀ q (y : P.bdry.X (q + 1)), P.bulk.bd q (δ (q + 1) y) = δ q (P.bdry.bd q y) := by
    intro q y
    simp only [δ, AddMonoidHom.sub_apply, AddMonoidHom.comp_apply, map_sub, P.j_bd, τ.t2, τb.t2]
  -- (V) vertices
  obtain ⟨I0, rep0, hfree0⟩ := hP.free_twisted 0
  have hV : ∀ i, ∃ y : P.bulk.X (0 + 1), P.bulk.bd 0 y = δ 0 (rep0 i) := by
    intro i
    refine hP.contract0 _ ?_
    simp only [δ, AddMonoidHom.sub_apply, AddMonoidHom.comp_apply, map_sub, τ.t3, hjaug, τb.t3, sub_self]
  choose y0 hy0 using hV
  obtain ⟨η0, ⟨hη0T, hη0r⟩, -⟩ := hfree0 (0 + 1) y0
  have claimV : (P.bulk.bd 0).comp η0 = δ 0 :=
    (hfree0 0 fun i => δ 0 (rep0 i)).unique ⟨hη0T.comp_left (hbd 0), fun i => by
      rw [AddMonoidHom.comp_apply, hη0r, hy0]⟩ ⟨hδ 0, fun i => rfl⟩
  -- (E) edges
  obtain ⟨I1, rep1, hfree1⟩ := hP.free_twisted (0 + 1)
  have hE : ∀ i, ∃ u : P.bulk.X (0 + 1 + 1),
      P.bulk.bd (0 + 1) u = δ (0 + 1) (rep1 i) - η0 (P.bdry.bd 0 (rep1 i)) := by
    intro i
    refine hP.contract1 _ ?_
    rw [map_sub, hδbd, ← AddMonoidHom.comp_apply (P.bulk.bd 0) η0, claimV, sub_self]
  choose u1 hu1 using hE
  obtain ⟨η1, ⟨hη1T, hη1r⟩, -⟩ := hfree1 (0 + 1 + 1) u1
  have claimE : (P.bulk.bd (0 + 1)).comp η1 + η0.comp (P.bdry.bd 0) = δ (0 + 1) :=
    (hfree1 (0 + 1) fun i => δ (0 + 1) (rep1 i)).unique
      ⟨T1.add (hη1T.comp_left (hbd (0 + 1))) (hη0T.comp_right (hbdb 0)), fun i => by
        rw [AddMonoidHom.add_apply, AddMonoidHom.comp_apply, AddMonoidHom.comp_apply, hη1r, hu1, sub_add_cancel]⟩
      ⟨hδ (0 + 1), fun i => rfl⟩
  -- assemble
  let η : ∀ q, P.bdry.X q →+ P.bulk.X (q + 1) := fun q =>
    if h : q = 0 then h ▸ η0 else if h' : q = 0 + 1 then h' ▸ η1 else 0
  have hη0 : η 0 = η0 := by simp [η]
  have hη1 : η (0 + 1) = η1 := by simp [η]
  have hηo : ∀ q, q ≠ 0 → q ≠ 0 + 1 → η q = 0 := fun q h h' => by simp only [η, dif_neg h, dif_neg h']
  refine ⟨η, fun q => ?_, fun q y => ?_⟩
  · by_cases h0 : q = 0
    · subst h0; rw [hη0]; exact hη0T
    by_cases h1 : q = 0 + 1
    · subst h1; rw [hη1]; exact hη1T
    rw [hηo q h0 h1]; exact T1.zero
  · rcases lt_trichotomy q 0 with hq | rfl | hq
    · have : Subsingleton (P.bdry.X q) := (hP.neg q hq).2
      rw [Subsingleton.elim (P.bdry.bd q y) 0, map_zero, add_zero]
      by_cases hq1 : q = -1
      · subst hq1
        change P.bulk.bd 0 (η 0 y) = δ 0 y
        rw [hη0, ← AddMonoidHom.comp_apply, claimV]
      · have : Subsingleton (P.bdry.X (q + 1)) := (hP.neg (q + 1) (by omega)).2
        rw [Subsingleton.elim y 0]; simp
    · rw [hη1, hη0]
      exact DFunLike.congr_fun claimE y
    · have : Subsingleton (P.bdry.X (q + 1)) := hP.bdry_dim (q + 1) (by omega)
      rw [Subsingleton.elim y 0]; simp

/-- `∂ : C₂(X_bulk) → C₁(X_bulk)` is injective (`contract2`). -/
lemma bd_one_injective (P : PairData.{u} Γ) (hP : PairHyp Γ P g) : Function.Injective (P.bulk.bd (0 + 1)) := by
  intro a b h
  exact sub_eq_zero.mp (hP.contract2 _ (by rw [map_sub, h, sub_self]))

/-- **`lem:etaunique`**, existence with (T1) (L18-4): two pair homotopies differ by `∂Λ − Λ∂` for a degree-`+2` map
`Λ` with (T1). Only `contract1`, `contract2` and the dimension bounds are used. -/
theorem eta_diff (P : PairData.{u} Γ) (hP : PairHyp Γ P g) (τ : TwistedChainDatum Γ g P.bulk)
    (τb : TwistedChainDatum Γ g P.bdry) (η η' : ∀ q, P.bdry.X q →+ P.bulk.X (q + 1))
    (hη : IsPairHomotopy Γ P τ τb η) (hη' : IsPairHomotopy Γ P τ τb η') :
    ∃ Λ : ∀ q, P.bdry.X q →+ P.bulk.X (q + 1 + 1), (∀ q, T1 Γ g (Λ q)) ∧
      ∀ q (y : P.bdry.X (q + 1)),
        η' (q + 1) y - η (q + 1) y = P.bulk.bd (q + 1 + 1) (Λ (q + 1) y) - Λ q (P.bdry.bd q y) := by
  classical
  let D : ∀ q, P.bdry.X q →+ P.bulk.X (q + 1) := fun q => η' q - η q
  have hD : ∀ q (y : P.bdry.X (q + 1)), P.bulk.bd (q + 1) (D (q + 1) y) + D q (P.bdry.bd q y) = 0 := by
    intro q y
    have h1 := hη.htpy q y
    have h2 := hη'.htpy q y
    simp only [D, AddMonoidHom.sub_apply, map_sub]
    rw [show ∀ a b c d : P.bulk.X (q + 1), a - b + (c - d) = (a + c) - (b + d) from fun a b c d => by abel, h1, h2,
      sub_self]
  have hDT : ∀ q, T1 Γ g (D q) := fun q => T1.sub (hη'.t1 q) (hη.t1 q)
  have hinj := bd_one_injective P hP
  -- `D₀ y` is a cycle
  have hcyc : ∀ y : P.bdry.X 0, P.bulk.bd 0 (D 0 y) = 0 := by
    intro y
    have h := hD (-1) (by exact y)
    have : Subsingleton (P.bdry.X (-1)) := (hP.neg (-1) (by norm_num)).2
    rw [Subsingleton.elim (P.bdry.bd (-1) _) 0, map_zero, add_zero] at h
    exact h
  choose L hL using fun y => hP.contract1 (D 0 y) (hcyc y)
  let Λ0 : P.bdry.X 0 →+ P.bulk.X (0 + 1 + 1) :=
    AddMonoidHom.mk' L fun a b => hinj (by rw [map_add, hL, hL, hL, map_add])
  have hΛ0 : ∀ y, P.bulk.bd (0 + 1) (Λ0 y) = D 0 y := hL
  let Λ : ∀ q, P.bdry.X q →+ P.bulk.X (q + 1 + 1) := fun q => if h : q = 0 then h ▸ Λ0 else 0
  have hΛq : Λ 0 = Λ0 := by simp [Λ]
  have hΛo : ∀ q, q ≠ 0 → Λ q = 0 := fun q h => by simp only [Λ, dif_neg h]
  refine ⟨Λ, fun q => ?_, fun q y => ?_⟩
  · by_cases h0 : q = 0
    · subst h0
      rw [hΛq]
      intro h h' hh x
      apply hinj
      rw [hΛ0, P.bulk.bd_equiv, hΛ0]
      exact hDT 0 h h' hh x
    · rw [hΛo q h0]; exact T1.zero
  · rcases lt_trichotomy q 0 with hq | rfl | hq
    · by_cases hq1 : q = -1
      · subst hq1
        rw [hΛo (-1) (by norm_num), AddMonoidHom.zero_apply, sub_zero]
        change D 0 y = P.bulk.bd (0 + 1) (Λ 0 y)
        rw [hΛq, hΛ0]
      · have : Subsingleton (P.bdry.X (q + 1)) := (hP.neg (q + 1) (by omega)).2
        rw [Subsingleton.elim y 0]; simp
    · change D (0 + 1) y = _
      rw [hΛo (0 + 1) (by norm_num), hΛq, AddMonoidHom.zero_apply, map_zero, zero_sub]
      apply hinj
      rw [map_neg, hΛ0]
      exact eq_neg_of_add_eq_zero_left (hD 0 y)
    · have : Subsingleton (P.bdry.X (q + 1)) := hP.bdry_dim (q + 1) (by omega)
      rw [Subsingleton.elim y 0]; simp

/-- **`lem:etaunique`**: two pair homotopies differ by `∂Λ − Λ∂` for a **unique** degree-`+2` map `Λ`, which then
has (T1) automatically. -/
theorem eta_unique (P : PairData.{u} Γ) (hP : PairHyp Γ P g) (τ : TwistedChainDatum Γ g P.bulk)
    (τb : TwistedChainDatum Γ g P.bdry) (η η' : ∀ q, P.bdry.X q →+ P.bulk.X (q + 1))
    (hη : IsPairHomotopy Γ P τ τb η) (hη' : IsPairHomotopy Γ P τ τb η') :
    ∃! Λ : ∀ q, P.bdry.X q →+ P.bulk.X (q + 1 + 1),
      ∀ q (y : P.bdry.X (q + 1)),
        η' (q + 1) y - η (q + 1) y = P.bulk.bd (q + 1 + 1) (Λ (q + 1) y) - Λ q (P.bdry.bd q y) := by
  classical
  obtain ⟨Λ, -, hΛ⟩ := eta_diff P hP τ τb η η' hη hη'
  refine ⟨Λ, hΛ, fun Λ' hΛ' => funext fun q => AddMonoidHom.ext fun y => ?_⟩
  have hinj := bd_one_injective P hP
  by_cases hq0 : q = 0
  · subst hq0
    have e1 := hΛ (-1) (by exact y)
    have e2 := hΛ' (-1) (by exact y)
    have : Subsingleton (P.bdry.X (-1)) := (hP.neg (-1) (by norm_num)).2
    rw [Subsingleton.elim (P.bdry.bd (-1) _) 0, map_zero, sub_zero] at e1 e2
    have e3 := e2.symm.trans e1
    exact hinj e3
  · rcases lt_or_gt_of_ne hq0 with hq | hq
    · have : Subsingleton (P.bdry.X q) := (hP.neg q hq).2
      rw [Subsingleton.elim y 0, map_zero, map_zero]
    · have : Subsingleton (P.bulk.X (q + 1 + 1)) := hP.bulk_dim _ (by omega)
      exact Subsingleton.elim _ _

/-- **`thm:conewelldef`(d)**: `η ≡ 0` is a pair homotopy iff `τ j = j τ_∂` ("`τ` is a map of pairs"). -/
theorem eta_zero_iff (P : PairData.{u} Γ) (τ : TwistedChainDatum Γ g P.bulk) (τb : TwistedChainDatum Γ g P.bdry) :
    IsPairHomotopy Γ P τ τb (fun _ => 0) ↔ ∀ q (y : P.bdry.X (q + 1)), τ.τ (q + 1) (P.j (q + 1) y) = P.j (q + 1) (τb.τ (q + 1) y) := by
  constructor
  · intro h q y
    have := h.htpy q y
    simp only [AddMonoidHom.zero_apply, map_zero, add_zero] at this
    exact (sub_eq_zero.mp this.symm).symm
  · intro h
    refine ⟨fun q h1 h2 _ x => by simp, fun q y => ?_⟩
    simp [h q y]

/-! ### The cone action -/

variable (cd : CosetData Γ Δ g) (P : PairData.{u} Γ)

/-- `Θ_g = τ̃` on the equivariant cochains (`eq:heckechain`), as a degree-`0` graded map. -/
def thetaG {Y : GChainCx.{u} Γ} (τ : TwistedChainDatum Γ g Y) : GMap (eqCochains Γ Δ hΓΔ ρ Y) (eqCochains Γ Δ hΓΔ ρ Y) :=
  fun q => (transportL cd (τ.τ q)).restrict fun φ hφ =>
    (transport_equivariant (hΓΔ := hΓΔ) cd (τ.τ q) (τ.t1 q) φ hφ).1

/-- **`def:coneaction`** (`eq:derivedh`): the derived cochain homotopy `h = η̃ : C^{q+1} → C^q_∂`. -/
def derivedH (η : ∀ q, P.bdry.X q →+ P.bulk.X (q + 1)) (hη : ∀ q, T1 Γ g (η q)) :
    GMapNeg (eqCochains Γ Δ hΓΔ ρ P.bulk) (eqCochains Γ Δ hΓΔ ρ P.bdry) := fun q =>
  (transportL cd (η q)).restrict fun φ hφ => (transport_equivariant (hΓΔ := hΓΔ) cd (η q) (hη q) φ hφ).1

/-- **`lem:derivedhomotopy`**: `d h + h d = Θ_∂ r − r Θ` on `C^{q+1}`, for every `q`, with no hypothesis on `ℓ`. -/
theorem derived_homotopy (τ : TwistedChainDatum Γ g P.bulk) (τb : TwistedChainDatum Γ g P.bdry)
    (η : ∀ q, P.bdry.X q →+ P.bulk.X (q + 1)) (hη : IsPairHomotopy Γ P τ τb η) (q : ℤ)
    (φ : (eqCochains Γ Δ hΓΔ ρ P.bulk).X (q + 1)) :
    (eqCochains Γ Δ hΓΔ ρ P.bdry).d q (derivedH cd P η hη.t1 q φ) +
        derivedH cd P η hη.t1 (q + 1) ((eqCochains Γ Δ hΓΔ ρ P.bulk).d (q + 1) φ) =
      thetaG cd τb (q + 1) (restr P (q + 1) φ) - restr P (q + 1) (thetaG cd τ (q + 1) φ) := by
  apply Subtype.ext; apply AddMonoidHom.ext; intro y
  show transport (ρ := ρ) Γ Δ cd (η q) φ.1 (P.bdry.bd q y) +
      transport (ρ := ρ) Γ Δ cd (η (q + 1)) (φ.1.comp (P.bulk.bd (q + 1))) y =
    transport (ρ := ρ) Γ Δ cd (τb.τ (q + 1)) (φ.1.comp (P.j (q + 1))) y -
      transport (ρ := ρ) Γ Δ cd (τ.τ (q + 1)) φ.1 (P.j (q + 1) y)
  simp only [transport, AddMonoidHom.finsetSum_apply, AddMonoidHom.comp_apply, DistribSMul.toAddMonoidHom_apply,
    LinearMap.toAddMonoidHom_coe, ← Finset.sum_add_distrib, ← Finset.sum_sub_distrib, ← map_add, ← map_sub]
  refine Finset.sum_congr rfl fun i _ => ?_
  congr 1
  have key := hη.htpy q (cd.s i • y)
  rw [P.bdry.bd_equiv q (cd.s i) y, P.j_equiv (q + 1) (cd.s i) y] at key
  exact congrArg _ ((add_comm _ _).trans key)

/-- The cone of the pair, `Cone(r)`. -/
abbrev pairCone : CochainCx.{u} R :=
  cone (restr (hΓΔ := hΓΔ) (ρ := ρ) P) (restr_isChainMap P)

/-- **`def:coneaction`** (`eq:conemap`): `Φ_g(c, c_∂) = (Θ_g c, Θ^∂_g c_∂ + h c)` on `Cone(r)`. -/
def PhiG (τ : TwistedChainDatum Γ g P.bulk) (τb : TwistedChainDatum Γ g P.bdry)
    (η : ∀ q, P.bdry.X q →+ P.bulk.X (q + 1)) (hη : ∀ q, T1 Γ g (η q)) :
    GMap (pairCone (hΓΔ := hΓΔ) (ρ := ρ) P) (pairCone (hΓΔ := hΓΔ) (ρ := ρ) P) :=
  coneMap _ _ (thetaG cd τ) (thetaG cd τb) (derivedH cd P η hη)

/-- **`thm:conewelldef`(a)**: `Φ_g` is a cochain map with `Φ_g ι = ι Θ^∂_g`, `π Φ_g = Θ_g π`; so it induces
`T^c_{ΓgΓ}` on `H^•_c` and every map of `eq:coneles` intertwines the three operators. (`Θ_g`, `Θ^∂_g` cochain maps is
[C] `thm:welldef` and `prop:bdrywelldef`, part of the hypothesis here; `ι`, `π` compatibility is `coneMap_inc_proj`.) -/
theorem conewelldef_a (τ : TwistedChainDatum Γ g P.bulk) (τb : TwistedChainDatum Γ g P.bdry)
    (η : ∀ q, P.bdry.X q →+ P.bulk.X (q + 1)) (hη : IsPairHomotopy Γ P τ τb η)
    (hΘ : IsChainMap (thetaG (hΓΔ := hΓΔ) (ρ := ρ) cd τ)) (hΘb : IsChainMap (thetaG (hΓΔ := hΓΔ) (ρ := ρ) cd τb)) :
    IsChainMap (PhiG (hΓΔ := hΓΔ) (ρ := ρ) cd P τ τb η hη.t1) := by
  intro q
  apply LinearMap.ext; intro x
  have hs := conesign (restr P) (restr_isChainMap P) (thetaG cd τ) hΘ (thetaG cd τb) hΘb (derivedH cd P η hη.t1) q x
  rw [derived_homotopy cd P τ τb η hη q x.1, sub_add_cancel, sub_self] at hs
  exact sub_eq_zero.mp (hs.trans rfl)

/-- **`thm:conewelldef`(b)**, **`prop:coneindep`(1)**: two pair homotopies `η, η'` give cone-homotopic `Φ_g`:
`Φ' − Φ = D Υ + Υ D` with `Υ(c, c_∂) = (0, −Λ̃ c)` (first component `0`: filtration-preserving); so `T^c_{ΓgΓ}` is
independent of `η`, over `ℤ`, for every `ℓ`. -/
theorem conewelldef_b (τ : TwistedChainDatum Γ g P.bulk) (τb : TwistedChainDatum Γ g P.bdry)
    (η η' : ∀ q, P.bdry.X q →+ P.bulk.X (q + 1)) (hη : IsPairHomotopy Γ P τ τb η)
    (hη' : IsPairHomotopy Γ P τ τb η') (hP : PairHyp Γ P g) :
    ∃ Υ : ∀ q, (pairCone (hΓΔ := hΓΔ) (ρ := ρ) P).X (q + 1) →ₗ[R] (pairCone (hΓΔ := hΓΔ) (ρ := ρ) P).X q,
      (∀ q x, (Υ q x).1 = 0) ∧
      ∀ q x, PhiG cd P τ τb η' hη'.t1 (q + 1) x - PhiG cd P τ τb η hη.t1 (q + 1) x =
        (pairCone (hΓΔ := hΓΔ) (ρ := ρ) P).d q (Υ q x) +
          Υ (q + 1) ((pairCone (hΓΔ := hΓΔ) (ρ := ρ) P).d (q + 1) x) := by
  classical
  obtain ⟨Λ, hΛT, hΛ⟩ := eta_diff P hP τ τb η η' hη hη'
  let LamT : ∀ q, (eqCochains Γ Δ hΓΔ ρ P.bulk).X (q + 1 + 1) →ₗ[R] (eqCochains Γ Δ hΓΔ ρ P.bdry).X q := fun q =>
    (transportL cd (Λ q)).restrict fun φ hφ => (transport_equivariant (hΓΔ := hΓΔ) cd (Λ q) (hΛT q) φ hφ).1
  let Υ : ∀ q, (pairCone (hΓΔ := hΓΔ) (ρ := ρ) P).X (q + 1) →ₗ[R] (pairCone (hΓΔ := hΓΔ) (ρ := ρ) P).X q :=
    fun q => LinearMap.prod 0 (-(LamT q).comp (LinearMap.fst R _ _))
  refine ⟨Υ, fun q x => rfl, fun q x => ?_⟩
  obtain ⟨c, e⟩ := x
  have key : derivedH cd P η' hη'.t1 (q + 1) c - derivedH cd P η hη.t1 (q + 1) c =
      -((eqCochains Γ Δ hΓΔ ρ P.bdry).d q (LamT q c)) +
        LamT (q + 1) ((eqCochains Γ Δ hΓΔ ρ P.bulk).d (q + 1 + 1) c) := by
    apply Subtype.ext; apply AddMonoidHom.ext; intro y
    show transport (ρ := ρ) Γ Δ cd (η' (q + 1)) c.1 y - transport (ρ := ρ) Γ Δ cd (η (q + 1)) c.1 y =
      -(transport (ρ := ρ) Γ Δ cd (Λ q) c.1 (P.bdry.bd q y)) +
        transport (ρ := ρ) Γ Δ cd (Λ (q + 1)) (c.1.comp (P.bulk.bd (q + 1 + 1))) y
    simp only [transport, AddMonoidHom.finsetSum_apply, AddMonoidHom.comp_apply, DistribSMul.toAddMonoidHom_apply,
      LinearMap.toAddMonoidHom_coe, ← Finset.sum_sub_distrib, ← Finset.sum_neg_distrib, ← Finset.sum_add_distrib,
      ← map_sub, ← map_neg, ← map_add]
    refine Finset.sum_congr rfl fun i _ => ?_
    congr 1
    have k := hΛ q (cd.s i • y)
    rw [P.bdry.bd_equiv q (cd.s i) y] at k
    exact congrArg _ (k.trans (by rw [map_neg]; abel))
  refine Prod.ext ?_ ?_
  · show thetaG cd τ (q + 1 + 1) c - thetaG cd τ (q + 1 + 1) c =
      -((eqCochains Γ Δ hΓΔ ρ P.bulk).d (q + 1) 0) + 0
    simp
  · show (thetaG cd τb (q + 1) e + derivedH cd P η' hη'.t1 (q + 1) c) -
        (thetaG cd τb (q + 1) e + derivedH cd P η hη.t1 (q + 1) c) =
      ((eqCochains Γ Δ hΓΔ ρ P.bdry).d q (-(LamT q c)) + restr P (q + 1) 0) +
        -(LamT (q + 1) (-((eqCochains Γ Δ hΓΔ ρ P.bulk).d (q + 1 + 1) c)))
    rw [map_neg, map_zero, add_zero, map_neg, neg_neg, ← key]; abel

end GL2.Cone
