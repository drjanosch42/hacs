import GL2.MP.Assembly
import GL2.MP.Strings

/-!
# MP-L2 groundwork (slot 22): complexes of free modules given on bases

[GT] `gt_part_MP.tex`, row `def:mp-bcx` (and `lem:mp-bcx-contr`, `lem:mp-cols-of-contr`).  The complexes of the MP
resolution ([C] `def:mpfree`) are all free on explicit bases, with differentials given on basis elements.  A `BCx k` is
such a complex (`B m` the basis in degree `m`, `bd m b ∈ k[B m]` the boundary of a basis element); `toCx` is the
`Cx k` of `GL2.MP.Assembly` with `X m = B m →₀ k`.  `BContr`, `BDMap` are contractions and degree-lowering maps given on
bases; `shift` prepends an empty degree (rows of a bounded double complex placed in total degree); `ColContr` is a
contraction of one column of a four-row bounded double complex, which gives the column's exactness (`colExactAt_of_contr`).
-/

namespace GL2.MP

open Finsupp

variable {k : Type} [CommRing k]

/-- The alternating double deletion sum with `k`-signs (`alt_sum_sq`). -/
theorem alt_sum_sq_k {O M : Type*} [AddCommGroup M] [Module k M] {n : ℕ} (f : Fin (n + 3) → O)
    (Φ : (Fin (n + 1) → O) → M) :
    ∑ i : Fin (n + 3), ((-1 : k) ^ (i : ℕ)) • ∑ j : Fin (n + 2), ((-1 : k) ^ (j : ℕ)) • Φ (del j (del i f)) = 0 := by
  have h := alt_sum_sq f Φ
  simp only [Finset.smul_sum, smul_smul]
  rw [← h]
  refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
  rw [← Int.cast_smul_eq_zsmul k]
  push_cast; ring_nf

variable (k)

/-- **A complex of free modules on bases**: `B m` the basis in degree `m`, `bd m` the boundary of basis elements. -/
structure BCx where
  B : ℕ → Type
  bd : ∀ m, B (m + 1) → (B m →₀ k)
  bdd : ∀ m (b : B (m + 2)), Finsupp.linearCombination k (bd m) (bd (m + 1) b) = 0

namespace BCx

variable {k} (C : BCx k)

/-- The differential `k[B (m+1)] → k[B m]`. -/
noncomputable def d (m : ℕ) : (C.B (m + 1) →₀ k) →ₗ[k] (C.B m →₀ k) := Finsupp.linearCombination k (C.bd m)

lemma d_single (m : ℕ) (b : C.B (m + 1)) (c : k) : C.d m (single b c) = c • C.bd m b := by
  simp [d]

lemma d_d (m : ℕ) (x : C.B (m + 2) →₀ k) : C.d m (C.d (m + 1) x) = 0 := by
  induction x using Finsupp.induction_linear with
  | zero => simp
  | add x y hx hy => rw [map_add, map_add, hx, hy, add_zero]
  | single b c => rw [d_single, map_smul]; simp only [d]; rw [C.bdd, smul_zero]

/-- **The complex** `X m = k[B m]`. -/
noncomputable def toCx : Cx.{0, 0} k where
  X m := C.B m →₀ k
  d := C.d
  dd := C.d_d

end BCx

/-- **A contraction given on bases**: `d h = id` in degree `0`, `d h + h d = id` above. -/
structure BContr (C : BCx k) where
  hb : ∀ m, C.B m → (C.B (m + 1) →₀ k)
  h0 : ∀ b : C.B 0, C.d 0 (hb 0 b) = single b 1
  hs : ∀ m (b : C.B (m + 1)), C.d (m + 1) (hb (m + 1) b) + Finsupp.linearCombination k (hb m) (C.bd m b) = single b 1

namespace BContr

variable {k} {C : BCx k} (H : BContr k C)

/-- The contraction of `toCx`. -/
noncomputable def toContraction : C.toCx.Contraction where
  h m := Finsupp.linearCombination k (H.hb m)
  h0 x := by
    have hL : C.d 0 ∘ₗ Finsupp.linearCombination k (H.hb 0) = LinearMap.id := by
      apply Finsupp.lhom_ext; intro b c
      simp only [LinearMap.comp_apply, LinearMap.id_apply, Finsupp.linearCombination_single, map_smul, H.h0,
        smul_single, smul_eq_mul, mul_one]
    exact LinearMap.congr_fun hL x
  hs m x := by
    have hL : C.d (m + 1) ∘ₗ Finsupp.linearCombination k (H.hb (m + 1)) +
        Finsupp.linearCombination k (H.hb m) ∘ₗ C.d m = LinearMap.id := by
      apply Finsupp.lhom_ext; intro b c
      simp only [LinearMap.add_apply, LinearMap.comp_apply, LinearMap.id_apply, Finsupp.linearCombination_single,
        map_smul, C.d_single]
      rw [← smul_add, H.hs, smul_single, smul_eq_mul, mul_one]
    exact LinearMap.congr_fun hL x

/-- **A complex with a contraction on bases is acyclic.** -/
theorem acyclic (H : BContr k C) : C.toCx.Acyclic := Cx.acyclic_of_contraction _ H.toContraction

end BContr

/-- **A degree-lowering map given on bases**, `T (m+1) → R m`, commuting with the differentials. -/
structure BDMap (R T : BCx k) where
  vb : ∀ m, T.B (m + 1) → (R.B m →₀ k)
  comm : ∀ m (t : T.B (m + 2)), R.d m (vb (m + 1) t) = Finsupp.linearCombination k (vb m) (T.bd (m + 1) t)

namespace BDMap

variable {k} {R T : BCx k} (φ : BDMap k R T)

/-- The map `k[T (m+1)] → k[R m]`. -/
noncomputable def v (m : ℕ) : (T.B (m + 1) →₀ k) →ₗ[k] (R.B m →₀ k) := Finsupp.linearCombination k (φ.vb m)

lemma v_single (m : ℕ) (t : T.B (m + 1)) (c : k) : φ.v m (single t c) = c • φ.vb m t := by simp [v]

/-- The `Cx.DMap`. -/
noncomputable def toDMap : Cx.DMap R.toCx T.toCx where
  v := φ.v
  comm m x := by
    show R.d m (φ.v (m + 1) x) = φ.v m (T.d (m + 1) x)
    induction x using Finsupp.induction_linear with
    | zero => simp
    | add x y hx hy => rw [map_add, map_add, hx, hy, map_add, map_add]
    | single t c => rw [v_single, map_smul, T.d_single, map_smul, φ.comm]; rfl

end BDMap

/-! ### Shifting a complex by one degree -/

/-- The shifted basis: empty in degree `0`. -/
def shB (B : ℕ → Type) : ℕ → Type
  | 0 => PEmpty
  | m + 1 => B m

instance (B : ℕ → Type) : IsEmpty (shB B 0) := ⟨fun x => PEmpty.elim x⟩

/-- The shifted boundary. -/
noncomputable def shBd {B : ℕ → Type} (bd : ∀ m, B (m + 1) → (B m →₀ k)) : ∀ m, shB B (m + 1) → (shB B m →₀ k)
  | 0 => fun _ => 0
  | m + 1 => bd m

namespace BCx

variable {k} (C : BCx k)

/-- **The shift** `C[1]`: degree `m + 1` is `C`'s degree `m`, degree `0` is `0`. -/
noncomputable def shift : BCx k where
  B := shB C.B
  bd := shBd k C.bd
  bdd m b := by
    rcases m with _ | m
    · exact Subsingleton.elim (α := shB C.B 0 →₀ k) _ _
    · exact C.bdd m b

lemma shift_d_zero : C.shift.d 0 = 0 := LinearMap.ext fun _ => Subsingleton.elim (α := shB C.B 0 →₀ k) _ _

lemma shift_d_succ (m : ℕ) : C.shift.d (m + 1) = C.d m := rfl

end BCx

/-- A contraction of `C` shifts to a contraction of `C[1]`. -/
noncomputable def BContr.shift {C : BCx k} (H : BContr k C) : BContr k C.shift where
  hb m := match m with
    | 0 => fun b => PEmpty.elim b
    | m + 1 => H.hb m
  h0 b := PEmpty.elim b
  hs m b := by
    rcases m with _ | m
    · have e : C.shift.bd 0 b = 0 := rfl
      rw [e, map_zero, add_zero]
      exact H.h0 b
    · exact H.hs m b

/-! ### Exact columns from column contractions -/

namespace Cx.BDC4

variable {k} (B : Cx.BDC4.{0, 0} k)

/-- **The column `0 → R₃(c+3) → R₂(c+2) → R₁(c+1) → R₀(c) → 0` is exact.** -/
structure ColExactAt (c : ℕ) : Prop where
  surj : ∀ a : B.R₀.X c, ∃ b, B.v₁.v c b = a
  ex₁ : ∀ a : B.R₁.X (c + 1), B.v₁.v c a = 0 → ∃ b, B.v₂.v (c + 1) b = a
  ex₂ : ∀ a : B.R₂.X (c + 2), B.v₂.v (c + 1) a = 0 → ∃ b, B.v₃.v (c + 2) b = a
  inj : ∀ a : B.R₃.X (c + 3), B.v₃.v (c + 2) a = 0 → a = 0

/-- **A contraction of the column `c`**: `v h = id`, `v h + h v = id`, `h v = id`. -/
structure ColContr (c : ℕ) where
  h₀ : B.R₀.X c →ₗ[k] B.R₁.X (c + 1)
  h₁ : B.R₁.X (c + 1) →ₗ[k] B.R₂.X (c + 2)
  h₂ : B.R₂.X (c + 2) →ₗ[k] B.R₃.X (c + 3)
  e₀ : ∀ a, B.v₁.v c (h₀ a) = a
  e₁ : ∀ a, B.v₂.v (c + 1) (h₁ a) + h₀ (B.v₁.v c a) = a
  e₂ : ∀ a, B.v₃.v (c + 2) (h₂ a) + h₁ (B.v₂.v (c + 1) a) = a
  e₃ : ∀ a, h₂ (B.v₃.v (c + 2) a) = a

/-- **A contractible column is exact.** -/
theorem colExactAt_of_contr {c : ℕ} (H : B.ColContr c) : B.ColExactAt c where
  surj a := ⟨H.h₀ a, H.e₀ a⟩
  ex₁ a ha := ⟨H.h₁ a, by have := H.e₁ a; rwa [ha, map_zero, add_zero] at this⟩
  ex₂ a ha := ⟨H.h₂ a, by have := H.e₂ a; rwa [ha, map_zero, add_zero] at this⟩
  inj a ha := by have := H.e₃ a; rwa [ha, map_zero, eq_comm] at this

/-- `ColsExact` from exactness of every column and the vanishing of the entries below the first quadrant. -/
theorem colsExact_of (h : ∀ c, B.ColExactAt c) (z₁₀ : ∀ a : B.R₁.X 0, a = 0) (z₂₀ : ∀ a : B.R₂.X 0, a = 0)
    (z₂₁ : ∀ a : B.R₂.X 1, a = 0) (z₃₀ : ∀ a : B.R₃.X 0, a = 0) (z₃₁ : ∀ a : B.R₃.X 1, a = 0)
    (z₃₂ : ∀ a : B.R₃.X 2, a = 0) : B.ColsExact where
  surj n := (h n).surj
  ex₁ n := (h n).ex₁
  ex₂ n := (h n).ex₂
  inj n := (h n).inj
  z₁₀ := z₁₀
  z₂₀ := z₂₀
  z₂₁ := z₂₁
  z₃₀ := z₃₀
  z₃₁ := z₃₁
  z₃₂ := z₃₂

end Cx.BDC4

end GL2.MP
