import GL2.Skeleton

/-!
Census anchor.  The per-declaration axiom report is produced by `tools/lean_census.sh`,
which generates `#print axioms` for every `\lean{}` name of
`blueprint/src/gl2_groundtruth.tex` and reads the output as text.  Kept as a module so
that `import GL2.Census` in a generated file loads the whole project root.
-/
