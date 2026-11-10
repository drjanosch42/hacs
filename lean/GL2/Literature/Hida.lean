import GL2.Literature.Basic

/-!
# L1: Hida theory, Emerton's Eisenstein-ideal theorem, Wiles' ordinary local shape (L16-3)

Every locator below was read from the library scan (`papers/h/Hida/…`, `papers/e/Emerton/…`,
`papers/w/Wiles/…`, all read-only). In all five structures the arithmetic objects (ordinary Hecke
algebras, cohomology of `Γ₁(N p^r)`, Galois representations) are **interface data**: parameters
constrained only by the shape of the source's statement. The L2 layer supplies the real objects.
-/

namespace GL2.Literature

/-- The Iwasawa algebra `Λ = ℤ_p[[Γ]] ≅ ℤ_p[[T]]`, `γ ↦ 1 + T` (`Γ = 1 + pℤ_p`). Hida's `Λ_K` is
`𝒪_K[[T]]`; every structure below is the `K = ℚ_p` case. -/
abbrev Iw (p : ℕ) [Fact p.Prime] : Type := PowerSeries ℤ_[p]

/-- `P_k = (1 + T) - (1 + p)^k ∈ Λ`: Hida's `P_k = (1 + X) - (1 + p)^k` (ens, p. 250). -/
noncomputable def Pk (p : ℕ) [Fact p.Prime] (k : ℕ) : Iw p :=
  (1 + PowerSeries.X) - PowerSeries.C ((1 + (p : ℤ_[p])) ^ k)

/-- **Hida, *Iwasawa modules attached to congruences of cusp forms*, Ann. Sci. ÉNS (4) 19 (1986),
Theorem 3.1, p. 250** -- freeness and ranks. `VERIFIED` against the numdam scan (page image).

Quotation: "THEOREM 3.1. — Assume that p ≥ 5 and that N is divisible by p. Then the ordinary parts
ℋ₀(N; 𝒪_K) and 𝔥₀(N; 𝒪_K) of the universal Hecke algebras are free of finite rank over Λ_K. Moreover, we
have that rank_{Λ_K}(ℋ₀(N, a; 𝒪_K)) = rank_{ℤ_p}(ℳ⁰_k(Φ; ℤ_p)), rank_{Λ_K}(𝔥₀(N, a; 𝒪_K)) =
rank_{ℤ_p}(𝒮⁰_k(Φ; ℤ_p)) if k ≥ 3 and k ≡ a mod p − 1."

Interface data: `H` is the cuspidal ordinary algebra `𝔥₀(N, a; ℤ_p)` as a `Λ`-module, `a` the weight
class, `ordRank k` the `ℤ_p`-rank of the ordinary cusp forms of weight `k` for the group `Φ`.
**Standing hypotheses of the source, as guards** (L16-5 finding 1): `5 ≤ p` and `p ∣ N` ("Assume that p ≥ 5
and that N is divisible by p"); `N` is the level of the interface data `H`.
Used by: [C] `thm:hidacontrol`(3), `thm:hidaweight`(1); [U] `thm:u-control`. -/
structure HidaEns31 (p : ℕ) [Fact p.Prime] (N : ℕ) (a : ZMod (p - 1)) (H : Type*) [AddCommGroup H]
    [Module (Iw p) H] (ordRank : ℕ → ℕ) : Prop where
  free_finite : 5 ≤ p → p ∣ N → Module.Free (Iw p) H ∧ Module.Finite (Iw p) H
  rank_eq : 5 ≤ p → p ∣ N → ∀ k : ℕ, 3 ≤ k → (k : ZMod (p - 1)) = a →
    Module.finrank (Iw p) H = ordRank k

/-- **Hida (ens), Corollary 3.2, p. 250** -- control. `VERIFIED` (same page).

Quotation: "COROLLARY 3.2. — The natural morphism induces the following isomorphism:
(3.3 a) ℋ₀(N, a; 𝒪_K)/P_k ℋ₀(N, a; 𝒪_K) ≅ ℋ⁰_k(Γ₁(N₀p), a; 𝒪_K) if k ≥ j(a),
(3.3 b) 𝔥₀(N, a; 𝒪_K)/P_k 𝔥₀(N, a; 𝒪_K) ≅ 𝔥⁰_k(Γ₁(N₀p), a; 𝒪_K) if k ≥ 2."

This is the cuspidal half (3.3 b). Interface data: `h` is `𝔥₀(N, a; ℤ_p)`, `hk k` is the weight-`k`
ordinary cuspidal Hecke algebra; the isomorphism is a ring isomorphism (the source says "isomorphism"
of algebras). **Standing hypotheses of the source, as guards** (L16-5 finding 1): `5 ≤ p` and `p ∣ N`
(Thm. 3.1's assumption, standing for the corollary). **Deliberate restriction** (L16-5 minor 3): the guard
`(k : ZMod (p - 1)) = a` is **not** in (3.3 b), which holds for every `k ≥ 2` with `hk k` the `a`-part
`𝔥⁰_k(Γ₁(N₀p), a)`; it is kept because the only consumer (`Sat.hidaalgconst`) works inside the class `a`,
and it can only weaken the structure. -/
structure HidaEns32 (p : ℕ) [Fact p.Prime] (N : ℕ) (a : ZMod (p - 1)) (h : Type*) [CommRing h]
    [Algebra (Iw p) h] (hk : ℕ → Type*) [∀ k, CommRing (hk k)] : Prop where
  control : 5 ≤ p → p ∣ N → ∀ k : ℕ, 2 ≤ k → (k : ZMod (p - 1)) = a →
    Nonempty ((h ⧸ Ideal.span {algebraMap (Iw p) h (Pk p k)}) ≃+* hk k)

/-- **Hida, *Galois representations into GL₂(ℤ_p[[X]]) attached to ordinary cusp forms*, Invent. Math. 85
(1986), 545-613, Theorem 3.1, p. 562.** `VERIFIED` against the PDF (page image).

Quotation: "Theorem 3.1. Let Γ_r denote the subgroup 1 + p^r ℤ_p of Γ. Then we have (i) For each integer
r > 0, the restriction morphism of cohomology groups induces an isomorphism of the module 𝒱_r⁰ (resp.
𝒲_r⁰) onto the module (𝒱⁰)^{Γ_r} (resp. (𝒲⁰)^{Γ_r}) of all Γ_r-invariants of 𝒱⁰ (resp. 𝒲⁰). (ii) The Λ-modules
V⁰ and W⁰ are free of finite rank over Λ. (iii) rank_Λ(V⁰) = 2·rank_Λ(𝔥⁰(N; ℤ_p)) and
rank_Λ(W⁰) = rank_Λ(V⁰) + ½φ(p) Σ_{0<t|N} φ(t)φ(N/t), where φ denotes the Euler function and t runs
over all divisor of N."

Parts (ii) and (iii) only (part (i) is `HidaInvent47`'s kind of statement). `rkH = rank_Λ 𝔥⁰(N; ℤ_p)`
is interface data; `(iii)` is written with both sides doubled to avoid `ℕ`-division.
**Standing hypotheses of the source, as guards** (L16-5 finding 1): `5 ≤ p` and `N` prime to `p` (§0, p. 545:
"Assume throughout the paper that p ≥ 5", "N prime to p"). -/
structure HidaInvent31 (p : ℕ) [Fact p.Prime] (N : ℕ) (V W : Type*) [AddCommGroup V] [AddCommGroup W]
    [Module (Iw p) V] [Module (Iw p) W] (rkH : ℕ) : Prop where
  free_finite : 5 ≤ p → Nat.Coprime N p → Module.Free (Iw p) V ∧ Module.Finite (Iw p) V ∧
    Module.Free (Iw p) W ∧ Module.Finite (Iw p) W
  rank_V : 5 ≤ p → Nat.Coprime N p → Module.finrank (Iw p) V = 2 * rkH
  rank_W : 5 ≤ p → Nat.Coprime N p → 2 * Module.finrank (Iw p) W = 2 * Module.finrank (Iw p) V +
    Nat.totient p * ∑ t ∈ N.divisors, Nat.totient t * Nat.totient (N / t)

/-- **Hida (invent), Proposition 4.7, p. 572** -- cohomological control. `VERIFIED` (page image).

Quotation: "Proposition 4.7. For each integer r > 0 and for each n > 0, the restriction map combined with
e gives isomorphisms: e₀ H¹(Γ₁(N), L_n(ℤ_p)) ≃ e H¹(Φ_r, L_n(ℤ_p)), e₀ H¹_P(Γ₁(N), L_n(ℤ_p)) ≃
e H¹_P(Φ_r, L_n(ℤ_p))."

Interface data: `ordH1 n r` is the ordinary part `e H¹(Φ_r, L_n(ℤ_p))` (`r = 0`: `e₀ H¹(Γ₁(N), L_n(ℤ_p))`),
a `ℤ_p`-module. Hecke-equivariance is **not** part of this statement (the proof says the isomorphisms are
"explicitly given by e∘res and e₀∘Tr"); a bridge lemma in L2 must prove it. **Two readings, one structure**
(L16-5 minor 4): Prop. 4.7 displays two isomorphisms, for `H¹` and for the parabolic `H¹_P`; the structure is
instantiated **once per reading** (`ordH1` the ordinary `H¹`, resp. the ordinary `H¹_P`). -/
structure HidaInvent47 (p : ℕ) [Fact p.Prime] (ordH1 : ℕ → ℕ → Type*) [∀ n r, AddCommGroup (ordH1 n r)]
    [∀ n r, Module ℤ_[p] (ordH1 n r)] : Prop where
  iso : ∀ n r : ℕ, 0 < n → 0 < r → Nonempty (ordH1 n 0 ≃ₗ[ℤ_[p]] ordH1 n r)

/-- **Hida, *Galois representations into GL₂(ℤ_p[[X]])*, Invent. Math. 85 (1986), Theorem 6.6, p. 590.**
`VERIFIED` (page image, L17-5).

Quotation: "Theorem 6.6. For each m > 0, there is a canonical isomorphism: e H¹_P(Φ₁, L_n(ℤ_p)) ⊗_{ℤ_p} ℤ/p^m ℤ ≃
e H¹_P(Φ₁, L_n(ℤ/p^m ℤ)). Moreover, if n ≡ a mod p − 1, we have an isomorphism of 𝔥⁰₂(N p^∞; ℤ_p)-modules:
e H¹_P(Φ₁, L_n(ℤ_p)) ⊗_{ℤ_p} 𝕋_p ≃ e H¹_P(Φ₁, L_n(𝕋_p)) ≃ 𝒱'⁰(a)[P_n]."

Only the first sentence (the reduction mod `p^m`; it is the cohomological base-change used by [C] `sec:hidafull`
and [U]) is a field; the "Moreover" is Hecke-module-valued and is not stated.  Interface data: `ordP n` is
`e H¹_P(Φ₁, L_n(ℤ_p))`, `ordPm n m` is `e H¹_P(Φ₁, L_n(ℤ/p^m))`, both `ℤ_p`-modules (the latter killed by `p^m`);
`⊗ ℤ/p^m` is written as the quotient by `p^m`.  The isomorphism is "canonical" in the source (natural
reduction), which the structure does not record (a bridge lemma in L2).
**Standing hypotheses of the source, as guards:** `5 ≤ p` and `N` prime to `p` (§0, p. 545), `0 < m`, `0 < n`
(the Theorem's "m > 0"; `n > 0` is the standing range of §4-§6, `L_n` with `n > 0` as in Prop. 4.7). -/
structure HidaInvent66 (p : ℕ) [Fact p.Prime] (N : ℕ) (ordP : ℕ → Type*) [∀ n, AddCommGroup (ordP n)]
    [∀ n, Module ℤ_[p] (ordP n)] (ordPm : ℕ → ℕ → Type*) [∀ n m, AddCommGroup (ordPm n m)]
    [∀ n m, Module ℤ_[p] (ordPm n m)] : Prop where
  reduction : 5 ≤ p → Nat.Coprime N p → ∀ n m : ℕ, 0 < n → 0 < m →
    Nonempty ((ordP n ⧸ (Ideal.span {((p : ℤ_[p]) ^ m)} • (⊤ : Submodule ℤ_[p] (ordP n)))) ≃ₗ[ℤ_[p]] ordPm n m)

/-- **Emerton, *The Eisenstein ideal in Hida's ordinary Hecke algebra*, IMRN 1999, no. 15, 793-802** --
the Theorem of the introduction, p. 2 of the library copy (an 8-page preprint; the journal data are
therefore not checked). `VERIFIED` (text layer).

Quotation: "Theorem. Suppose that i ≢ 0 (mod p − 1). Then the pullback of I⁰_Eis via the natural map
Λ_(i) → T^ord_par,(i) is equal to the principal ideal of Λ_(i) generated by the Kubota-Leopoldt p-adic
L-function L_p(ω^i, 1 − s), and the consequent injection Λ_(i)/L_p(ω^i, 1 − s) → T^ord_par,(i)/I⁰_Eis is an
isomorphism." (Standing hypothesis: "Let us fix a prime number p ≥ 5"; tame level one.)

Interface data: `T` is `T^ord_par,(i)` as a `Λ_(i) ≅ ℤ_p[[Γ]]`-algebra, `I` the Eisenstein ideal `I⁰_Eis`,
`Lp` the element of `Λ_(i)` for `L_p(ω^i, 1 - s)`; `i` is the component index of these data.
**Standing hypotheses of the source, as guards** (L16-5 finding 1): `5 ≤ p` and the theorem's own hypothesis
`i ≢ 0 (mod p − 1)`. Used by: [U] `lem:u-emerton`. -/
structure EmertonThm (p : ℕ) [Fact p.Prime] (i : ℕ) (T : Type*) [CommRing T] [Algebra (Iw p) T]
    (I : Ideal T) (Lp : Iw p) : Prop where
  pullback : 5 ≤ p → ¬ (p - 1 ∣ i) → Ideal.comap (algebraMap (Iw p) T) I = Ideal.span {Lp}
  surjective : 5 ≤ p → ¬ (p - 1 ∣ i) →
    Function.Surjective ((Ideal.Quotient.mk I).comp (algebraMap (Iw p) T))

/-- **Wiles, *On ordinary λ-adic representations associated to modular forms*, Invent. Math. 94 (1988),
529-573, Theorem 2.1.4, p. 561.** `VERIFIED` (page image). Totally real base field `F`.

Quotation: "Theorem 2.1.4. If f is ordinary at λ and 𝔭 | Nλ then the restriction of ρ_λ to D_𝔭 has the
form ρ_λ|_{D_𝔭} ≈ (ε₁ *; 0 ε₂) where ε₂ is unramified and ε₂(Frob 𝔭) = α(𝔭, f) is a unit root of
x² − c(𝔭, f)x + ψ(𝔭)N𝔭^{k−1} = 0."

(The `*` and the unit-root property: `α` is a root of the quadratic; that it is a *unit* root needs the
valuation ring and is carried by `ordinary`.) Interface data: `D` the decomposition group at `𝔭`, `I` its
inertia, `frob` a Frobenius, `ρ` the representation, `c`, `ψNp` the quadratic's coefficients. Used by:
[U] `thm:u-nocm` case (b), cited there with the locator `VERIFY`, now resolved. -/
structure Wiles214 {D : Type*} [Group D] (I : Subgroup D) (L : Type*) [Field L] (ρ : D →* GL (Fin 2) L)
    (frob : D) (α c ψNp : L) (ordinary : Prop) : Prop where
  root : ordinary → α ^ 2 - c * α + ψNp = 0
  shape : ordinary → ∃ (P : GL (Fin 2) L) (ε₁ ε₂ : D →* Lˣ) (b : D → L),
    (∀ g, ((P⁻¹ * ρ g * P : GL (Fin 2) L) : Matrix (Fin 2) (Fin 2) L) = !![(ε₁ g : L), b g; 0, (ε₂ g : L)]) ∧
    (∀ g ∈ I, ε₂ g = 1) ∧ (ε₂ frob : L) = α

end GL2.Literature
