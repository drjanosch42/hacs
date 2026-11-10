import GL2.Literature.Basic

/-!
# L1: Diamond-Im, *Modular forms and modular curves* (L17-5 repairs)

Source: F. Diamond and J. Im, *Modular forms and modular curves*, Canadian Mathematical Society Conference
Proceedings, Volume 17 (1995), pp. 39 ff. (library: `papers/d/Diamond/Diamond_Im-Modular_forms_and_modular_curves.pdf`,
a 48-image camera scan, two printed pages per image, no text layer; read at 150 dpi, quarter-cropped, 2026-10-02;
PDF page `p` holds printed pages `2p + 36`, `2p + 37`).  The L17-5 report called it unreadable at 70 dpi; the L17-5
audit (MAJOR-1) read it at 150 dpi.  Two structures:

* `DiamondImEichlerShimura`: Theorem 12.2.2 (the Eichler-Shimura isomorphisms `β`, `β_p`) with Proposition 12.4.10
  (their compatibility with the Hecke action), the source of the Eichler-Shimura half of `EichlerShimuraLevelOne`
  and `FormsSameSystems`;
* `DiamondImEigenspaces`: §3.5-§3.6 (pp. 54-55), every eigenspace of the full Hecke algebra on forms of type
  `(k, N, ε)` is at most one-dimensional, and `S_k(N, ε)` is a direct sum of `T^{(N)}`-eigenspaces (Petersson), the
  source of the eigenbasis half on cusp forms.

At level one (`Γ = Γ_0(1) = SL_2(ℤ)`, `N = 1`, `ε = 1`) `T^{(1)}` is the algebra of all `T_n`, and `V_k = Sym^{k-2}(ℂ²)`
is [C]'s `Sym^n ⊗ ℂ` with `n = k - 2`.  Everything is **interface data**: Lean has neither group cohomology of `Γ`
nor Hecke operators on `ModularForm`; the L2 layer supplies the objects.
-/

namespace GL2.Literature

/-- The joint eigenspace of a family `X : ι → End V` of `ℂ`-linear maps for the system `χ : ι → ℂ`. -/
def eigenC {ι V : Type*} [AddCommGroup V] [Module ℂ V] (X : ι → Module.End ℂ V) (χ : ι → ℂ) :
    Submodule ℂ V :=
  ⨅ i, LinearMap.ker (X i - χ i • LinearMap.id)

/-- **Diamond-Im, *Modular forms and modular curves*, Theorem 12.2.2 (printed p. 108) and Proposition 12.4.10
(printed p. 117).** `VERIFIED` (page images, L17-5 repairs).

Quotations.  §12.1, p. 102: "Let k be an integer and Γ a congruence subgroup of SL₂(Z)."  §12.2, p. 107: "Next we
briefly explain how this generalizes to weight k ≥ 2 by replacing C with a certain (k − 1)-dimensional
representation of SL₂(Z). (See [Shi1, §8.2].) We let V_k = Symm_C^{k−2}(C²) with an action of SL₂(Z) gotten from the
standard one on C². For f in 𝓜_k(Γ) we define a class in H¹(Γ, V_k) by the cocycle (12.2.5) γ ↦ ∫_{z₀}^{γ(z₀)}
f(z)(z 1)^{k−2} dz. [...] Together with a similar construction for antiholomorphic differentials, we obtain a
C-linear map β : 𝓜_k(Γ) ⊕ 𝓢̄_k(Γ) → H¹(Γ, V_k)", p. 108: "which restricts to β_p : 𝓢_k(Γ) ⊕ 𝓢̄_k(Γ) → H¹_p(Γ, V_k).
**Theorem 12.2.2.** β and β_p are isomorphisms."  §12.4, p. 115: "Let us first consider the case of cusp forms of
weight two with respect to Γ = Γ₁(N) or Γ₀(N). Let Δ denote the corresponding semigroup Δ₁(N) or Δ₀(N)"; p. 117: "We
can similarly define an action of R(Γ, Δ) on H¹(Γ, V_k) preserving H¹_p(Γ, V_k) [...] More importantly, we have
(see [Shi1, Proposition 8.5]) **Proposition 12.4.10.** The Eichler-Shimura isomorphisms β and β_p of Theorem 12.2.2
are compatible with the action of R(Γ, Δ)."

(The "For k > 2, assume Γ = Γ₁(N) with N > 4" on p. 117 concerns only the sheaf-theoretic description (12.4.6) of
the action, not the Proposition.)

Stated **weaker** than the source: the fields assert that *some* Hecke-equivariant `ℂ`-linear isomorphisms exist
(not that the specific `β`, `β_p` of (12.2.5) are such).  Interface data, for `Γ = Γ₀(N)` or `Γ₁(N)`: `Mk` is
`𝓜_k(Γ)`, `Sk` is `𝓢_k(Γ)`, `Skb` is `𝓢̄_k(Γ)`, `H1` is `H¹(Γ, V_k)`, `H1p` is `H¹_p(Γ, V_k)`; `ι` indexes elements of
the Hecke ring `R(Γ, Δ)` (or generators of it) and `TM`, `TS`, `TSb`, `TH`, `THp` are their actions.
**Standing hypotheses of the source, as guards:** `2 ≤ k` (§12.2's weight `k ≥ 2`) and `1 ≤ N` (the level of
`Γ₀(N)`, `Γ₁(N)`).  Used by: [C] `lem:heckerankone`, `prop:cohforms`(3) (through `EichlerShimuraLevelOne`,
`FormsSameSystems`), `thm:classical`. -/
structure DiamondImEichlerShimura (k : ℤ) (N : ℕ) {ι : Type*} (Mk Sk Skb H1 H1p : Type*)
    [AddCommGroup Mk] [Module ℂ Mk] [AddCommGroup Sk] [Module ℂ Sk] [AddCommGroup Skb] [Module ℂ Skb]
    [AddCommGroup H1] [Module ℂ H1] [AddCommGroup H1p] [Module ℂ H1p]
    (TM : ι → Module.End ℂ Mk) (TS : ι → Module.End ℂ Sk) (TSb : ι → Module.End ℂ Skb)
    (TH : ι → Module.End ℂ H1) (THp : ι → Module.End ℂ H1p) : Prop where
  iso_hecke : 2 ≤ k → 1 ≤ N → ∃ β : (Mk × Skb) ≃ₗ[ℂ] H1,
    ∀ i, (β : (Mk × Skb) →ₗ[ℂ] H1) ∘ₗ LinearMap.prodMap (TM i) (TSb i) = TH i ∘ₗ (β : (Mk × Skb) →ₗ[ℂ] H1)
  iso_hecke_p : 2 ≤ k → 1 ≤ N → ∃ βp : (Sk × Skb) ≃ₗ[ℂ] H1p,
    ∀ i, (βp : (Sk × Skb) →ₗ[ℂ] H1p) ∘ₗ LinearMap.prodMap (TS i) (TSb i) = THp i ∘ₗ (βp : (Sk × Skb) →ₗ[ℂ] H1p)

/-- **Diamond-Im, §3.5 (printed p. 54) and §3.6 (printed p. 55): multiplicity one and the decomposition of cusp
forms into Hecke eigenspaces.** `VERIFIED` (page images, L17-5 repairs).

Quotations.  p. 54: "Let us next observe that every T_N-eigenspace is (at most) one-dimensional. Suppose
f ∈ 𝓜_k(N, ε) is a (non-zero) T_N-eigenform [...] Thus, if two forms of type (k, N, ε) are common eigenforms of T_n
for all n with the same system {λ_n} of eigenvalues then one is a scalar multiple of the other."  (And: a form "is a
T_N-eigenform if and only if it is a simultaneous eigenvector under all T_n.")  p. 55: "On 𝓢_k(Γ₀(N)) the Hecke
operators T_n for (n, N) = 1 are self-adjoint with respect to the Petersson product [...] Thus the operators of the
form T_n and ⟨n⟩ for n relatively prime to N form a mutually commutative set of normal operators on 𝓢_k(Γ₁(N)).
[...] Applying the spectral decomposition theorem for normal operators (e.g. [Hers, Theorem 6.10.4]), we deduce that
there is an orthogonal decomposition 𝓢_k(Γ₁(N)) = ⊕_ε 𝓢_k(N, ε) (where ε runs over all Dirichlet characters mod N
such that ε(−1) = (−1)^k), and that each 𝓢_k(N, ε) decomposes orthogonally into a direct sum of
T^{(N)}-eigenspaces [Miy2, Theorem 4.5.4]."

Stated **weaker** than the source (orthogonality is not recorded; the direct sum is read as "the finitely many
joint eigenspaces span").  Interface data: `Mk` is `𝓜_k(N, ε)` with `Tall` the operators `T_n`, all `n ≥ 1`;
`Sk` is `𝓢_k(N, ε)` with `TN` the operators of `T^{(N)}` (the `T_n` with `(n, N) = 1`).  At `N = 1` both families are
all the `T_n`, and with `M_k = M_k^0 ⊕ C.G_k` (Serre §3.2) and `SerreCoursProp13` this is the eigenbasis of `M_k`
used by [C] `lem:heckerankone`.  **Standing hypotheses of the source, as guards:** `1 ≤ N`.  Used by: [C]
`lem:heckerankone`, `prop:cohforms`(3) (through `EichlerShimuraLevelOne`, `FormsSameSystems`). -/
structure DiamondImEigenspaces (N : ℕ) {ι κ : Type*} (Mk Sk : Type*)
    [AddCommGroup Mk] [Module ℂ Mk] [AddCommGroup Sk] [Module ℂ Sk]
    (Tall : κ → Module.End ℂ Mk) (TN : ι → Module.End ℂ Sk) : Prop where
  mult_one : 1 ≤ N → ∀ χ : κ → ℂ, Module.finrank ℂ (eigenC Tall χ) ≤ 1
  decomp : 1 ≤ N → ∃ S : Finset (ι → ℂ), (⨆ χ ∈ S, eigenC TN χ) = ⊤

end GL2.Literature
