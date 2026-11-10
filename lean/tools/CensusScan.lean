import GL2

/-!
# The per-constant scan of the Lean census (L16-5 repair of finding 2 (c), (d))

Run by `tools/lean_census.py` as `lake env lean tools/CensusScan.lean` (from `lean/`), never by
`lake build`.  It walks **every** constant whose defining module is `GL2` or `GL2.*` (namespace
independent, so a declaration outside `namespace GL2` is seen too; auxiliary constants included) and
prints, one line each, tab-separated:

* `CONST  <name>  <module>  <axioms|none>  <direct>` -- the axioms the constant depends on
  (transitively, as `#print axioms` computes them) and `direct` = the axioms its own type / value
  mention (`sorryAx` there means a `sorry` / `admit` written in *this* declaration; `Lean.ofReduceBool`
  means a `native_decide` in it);
* `AXDECL <name>  <module>` -- an `axiom` declaration in a GL2 module;
* `HYP    <decl>  <binder>  <head>  <closed>  <head_in_gl2>  <head_in_literature>` -- for every name
  listed in the environment variable `LEAN_CENSUS_HYPS` (comma separated; the `proved_mod_lit` rows),
  one line per `Prop`-valued binder of its type (the syntactic binders: `forallTelescope`, no unfolding;
  F3 repairs: the list is every proved and proved_mod_lit row).  `head` is the head constant of the binder's type
  after instantiating the earlier binders (`?fvar` if it is a local variable, `-` otherwise), `closed` is `1` iff the type mentions no
  earlier binder.
* `SCAN_END <count>`.
-/

open Lean Meta Elab Command

namespace GL2Census

/-- Module of a constant, if imported. -/
def moduleOf (env : Environment) (n : Name) : Option Name :=
  (env.getModuleIdxFor? n).map fun idx => env.header.moduleNames[idx.toNat]!

def inGL2 (env : Environment) (n : Name) : Bool :=
  match moduleOf env n with
  | some m => m.getRoot == `GL2
  | none => false

/-- Transitive axioms with a memo shared across all queries. -/
partial def axiomsOf (env : Environment) (n : Name) :
    StateM (Std.HashMap Name (Array Name)) (Array Name) := do
  if let some a := (← get).get? n then return a
  -- mark as visiting (cycles cannot occur in the kernel environment, but be safe)
  modify (·.insert n #[])
  let res ← match env.find? n with
    | some (.axiomInfo _) => pure #[n]
    | some (.quotInfo _) => pure #[]
    -- a constructor mentions its inductive, so its axioms are the inductive group's (as `#print axioms`)
    | some (.ctorInfo v) => axiomsOf env v.induct
    -- an inductive: its type and every constructor type, over the whole mutual group at once, so that no
    -- member's memo entry is taken while the group is still in progress
    | some (.inductInfo v) => do
      let group := v.all
      let mut inGroup : Array Name := #[]
      let mut exprs : Array Expr := #[]
      for I in group do
        inGroup := inGroup.push I
        if let some (.inductInfo w) := env.find? I then
          exprs := exprs.push w.type
          for c in w.ctors do
            inGroup := inGroup.push c
            if let some ci := env.find? c then exprs := exprs.push ci.type
      for m in inGroup do modify (·.insert m #[])
      let mut acc : Array Name := #[]
      for e in exprs do
        for c in e.getUsedConstants do
          unless inGroup.contains c do
            for a in (← axiomsOf env c) do
              unless acc.contains a do acc := acc.push a
      for m in inGroup do modify (·.insert m acc)
      pure acc
    | some ci => do
      let mut acc : Array Name := #[]
      let cs := ci.type.getUsedConstants ++ (match ci.value? (allowOpaque := true) with | some v => v.getUsedConstants | none => #[])
      for c in cs do
        for a in (← axiomsOf env c) do
          unless acc.contains a do acc := acc.push a
      pure acc
    | none => pure #[]
  modify (·.insert n res)
  return res

def directAxioms (env : Environment) (ci : ConstantInfo) : Array Name :=
  let cs := ci.type.getUsedConstants ++ (match ci.value? (allowOpaque := true) with | some v => v.getUsedConstants | none => #[])
  cs.foldl (fun acc c => match env.find? c with
    | some (.axiomInfo _) => if acc.contains c then acc else acc.push c
    | _ => acc) #[]

def joinNames (a : Array Name) : String :=
  if a.isEmpty then "none" else ",".intercalate ((a.map toString).qsort (· < ·)).toList

def hypLines (n : Name) : MetaM (Array String) := do
  let env ← getEnv
  let some ci := env.find? n | return #[s!"HYP\t{n}\t-\tUNRESOLVED\t-\t-\t-"]
  -- F3 repairs: `forallTelescope`, not `…Reducing`: a conclusion `a ≠ b` must not unfold to a binder `a = b`.
  forallTelescope ci.type fun xs _ => do
    let mut out := #[]
    for i in [0:xs.size] do
      let x := xs[i]!
      let ty ← instantiateMVars (← inferType x)
      unless (← isProp ty) do continue
      let decl ← x.fvarId!.getDecl
      let head := match ty.getAppFn with
        | .const c _ => c
        | _ => Name.anonymous
      let closed := !(xs.extract 0 i).any fun y => ty.containsFVar y.fvarId!
      let hs := if !head.isAnonymous then toString head else if ty.getAppFn.isFVar then "?fvar" else "-"
      let hg := if !head.isAnonymous && inGL2 env head then "1" else "0"
      let hl := if !head.isAnonymous && (`GL2.Literature).isPrefixOf head then "1" else "0"
      out := out.push s!"HYP\t{n}\t{decl.userName}\t{hs}\t{if closed then 1 else 0}\t{hg}\t{hl}"
    return out

end GL2Census

open GL2Census in
#eval show CommandElabM Unit from do
  let env ← getEnv
  let mut names : Array Name := #[]
  for (n, _) in env.constants.toList do
    if inGL2 env n then names := names.push n
  names := names.qsort (fun a b => a.toString < b.toString)
  let mut memo : Std.HashMap Name (Array Name) := {}
  for n in names do
    let some ci := env.find? n | continue
    let m := (moduleOf env n).getD Name.anonymous
    match ci with
    | .axiomInfo _ => IO.println s!"AXDECL\t{n}\t{m}"
    | _ => pure ()
    let (ax, memo') := (axiomsOf env n).run memo
    memo := memo'
    IO.println s!"CONST\t{n}\t{m}\t{joinNames ax}\t{joinNames (directAxioms env ci)}"
  let hyps := ((← IO.getEnv "LEAN_CENSUS_HYPS").getD "").splitOn ","
  for h in hyps do
    let h := h.trimAscii.toString
    if h.isEmpty then continue
    let nm := h.toName
    let ls ← liftTermElabM (hypLines nm)
    for l in ls do IO.println l
  IO.println s!"SCAN_END\t{names.size}"
