#!/usr/bin/env bash
# lean_census.sh -- the Lean census gate.  Builds GL2 (lake, shared deps), then lean_census.py.
#
#   lean_census.sh [--emit]     the gate on this checkout  (prints the LEAN_CENSUS line; READ THE TEXT)
#   lean_census.sh --selftest   the negative controls on a scratch copy -- each must print verdict=FAIL AND
#                               show the field it plants (CONTROL ... expect=<field>):
#                                 planted_sorry      a linked proved row closed by sorry        (unresolved, sorries)
#                                 planted_axiom      an `axiom` declaration                     (global_axioms)
#                                 stale_lean_name    a \lean{} name that does not exist         (unresolved, stale)
#                                 unlinked_admit     an UNLINKED theorem closed by `admit`      (escapes)     [L16-5]
#                                 unlinked_native    an UNLINKED theorem closed by native_decide (escapes, native) [L16-5]
#                                 vacuous_modlit     a linked `(h : False)` theorem tagged \gtmodlit, no \gtlithyp
#                                                                                               (modlit_bad)  [L16-5]
#                                 vacuous_modlit_declared   the same with \gtlithyp{h}         (modlit_bad)  [L16-5]
#                               [L17-4 repair: a native fact LINKED to a row is never kernel-checked evidence; each of these
#                                four is re-emitted (--emit, as an author would commit it, so stale=0) and must FAIL the
#                                live gate on global_axioms AND the committed gate lean_census_committed.py on nonstd_axioms]
#                                 linked_native_thm     native theorem in GL2/Algorithms, linked to a fresh thm: row
#                                 linked_native_chk     the same, added to the existing chk:stablecert \lean{}
#                                 native_on_lithyp      the same, as a literature `hypothesis` row (lit_hyp)
#                                 linked_native_outside a chk: row whose native theorem is in GL2/Skeleton.lean
#                               [F3 repairs, 2026-10-02; each re-emitted and run through the committed gate as well]
#                                 vacuous_proved     a linked `(h : False)` theorem on a PLAIN proved row (no \gtmodlit)
#                                                                         (hyp_bad; committed: hyp_bad)   [finding 3]
#                                 skip_kernel        `set_option debug.skipKernelTC true in` a false `decide +kernel`
#                                                    theorem, linked as proved   (escapes; committed: escapes) [finding 4]
#                                 lakefile_option    `debug.skipKernelTC = true` in lakefile.toml [leanOptions]
#                                                                         (escapes; committed: escapes)   [finding 4]
#                                 unlinked_native_alg an UNLINKED native_decide theorem in GL2/Algorithms/ (the former
#                                                    exemption)            (escapes; committed: escapes)   [finding 8]
#
# disk_guard.sh --min-free-gb 15 runs first when the repository's guard is found.
# Needs lean/.lake/packages -> the shared lean-deps (tools/lean_link_deps.sh).
# Selftest scratch: $LEAN_CENSUS_TMP (default ${XDG_CACHE_HOME:-$HOME/.cache}/hacs/lean_census_selftest), deleted
# afterwards.  The checkout's own lean/.lake/build is copied (cp -a) into the scratch copy when present, so a control
# rebuilds only the modules it touches.
set -uo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
repo="$(cd "$here/.." && pwd)"
guard="$repo/docs/matrix-engine/tools/disk_guard.sh"
[ -x "$guard" ] && { "$guard" --min-free-gb 15 || { echo "LEAN_CENSUS disk guard refused" >&2; exit 2; }; }

# Memory (S25 housekeeping, audit docs/hacs/LATEX_polish_s6_audit.md s5): lake 5.0 (Lean v4.33.0) has NO jobs flag
# (`lake build --help` lists none); its module jobs are tasks of the Lean runtime thread pool, each blocking on a child
# `lean` process, so the pool size bounds the concurrent children.  The census script therefore sets
# LEAN_NUM_THREADS=${LEAN_CENSUS_JOBS:-4} for the build step ONLY (default 4; an already-exported LEAN_NUM_THREADS wins).
# Manual builds stay a plain `lake build`.  Peak RSS of build and scan are printed separately (LEAN_CENSUS_PEAK).
tm=/usr/bin/time
peak() { sed -n 's/.*Maximum resident set size (kbytes): *\([0-9]*\).*/\1/p' "$1" | tail -1 | awk '{printf "%.2f", $1/1048576}'; }
jobs_n="${LEAN_NUM_THREADS:-${LEAN_CENSUS_JOBS:-4}}"
build() {
  if [ -x "$tm" ] && [ -n "${LEAN_CENSUS_TIMEFILE:-}" ]; then
    (cd "$1" && LEAN_NUM_THREADS="$jobs_n" "$tm" -v -o "$LEAN_CENSUS_TIMEFILE.build" lake build GL2 2>&1 | tail -3)
  else
    (cd "$1" && LEAN_NUM_THREADS="$jobs_n" lake build GL2 2>&1 | tail -3)
  fi
}

if [ "${1:-}" = "--selftest" ]; then
  tmp="${LEAN_CENSUS_TMP:-${XDG_CACHE_HOME:-$HOME/.cache}/hacs/lean_census_selftest}"
  fails=0
  ctls="planted_sorry planted_axiom stale_lean_name unlinked_admit unlinked_native vacuous_modlit vacuous_modlit_declared
        linked_native_thm linked_native_chk native_on_lithyp linked_native_outside
        vacuous_proved skip_kernel lakefile_option unlinked_native_alg"
  nctl=0
  committed="$repo/docs/matrix-engine/tools/lean_census_committed.py"
  for ctl in $ctls; do
    nctl=$((nctl+1)); emit=""; cexpect='nonstd_axioms=[1-9].*verdict=BAD'
    rm -rf "$tmp"; mkdir -p "$tmp"
    cp -r "$here/GL2" "$here/GL2.lean" "$here/lakefile.toml" "$here/lean-toolchain" "$here/lake-manifest.json" "$here/census.tsv" "$tmp/"
    mkdir -p "$tmp/blueprint" "$tmp/tools" "$tmp/.lake"; cp -r "$here/blueprint/src" "$tmp/blueprint/"
    cp "$here/tools/lean_census.py" "$here/tools/CensusScan.lean" "$here/tools/lean_link_deps.sh" "$tmp/tools/"
    [ -d "$here/.lake/build" ] && cp -a "$here/.lake/build" "$tmp/.lake/"
    "$tmp/tools/lean_link_deps.sh" >/dev/null
    gt="$tmp/blueprint/src/gl2_groundtruth.tex"
    case "$ctl" in
      planted_sorry)    sed -i 's/:= rfl/:= sorry/' "$tmp/GL2/Skeleton.lean"; expect='unresolved=[1-9]' ;;
      planted_axiom)    printf '\naxiom planted_ax : False\n' >> "$tmp/GL2/Skeleton.lean"; expect='global_axioms=[1-9]' ;;
      stale_lean_name)  sed -i 's/GL2.skeleton_two/GL2.skeleton_three/' "$gt"; expect='unresolved=[1-9]' ;;
      unlinked_admit)   printf '\nnamespace GL2\ntheorem selftest_admit : (1 : Nat) = 2 := by admit\nend GL2\n' >> "$tmp/GL2/Skeleton.lean"
                        expect='escapes=[1-9]' ;;
      unlinked_native)  printf '\nnamespace GL2\ntheorem selftest_native : (2 : Nat) + 2 = 4 := by native_decide\nend GL2\n' >> "$tmp/GL2/Skeleton.lean"
                        expect='native=[1-9].* escapes=[1-9]' ;;
      vacuous_modlit|vacuous_modlit_declared)
                        printf '\nnamespace GL2\ntheorem selftest_vacuous (h : False) : (1 : Nat) = 2 := h.elim\nend GL2\n' >> "$tmp/GL2/Skeleton.lean"
                        decl='A literature hypothesis that is in fact False.'
                        [ "$ctl" = vacuous_modlit_declared ] && decl='A literature hypothesis \gtlithyp{h} that is in fact False.'
                        printf '\n\\begin{theorem}[Selftest]\\label{selftest:vacuous}\n  \\lean{GL2.selftest_vacuous}\n  \\leanok\n  $1=2$ from a hypothesis.\n  \\gtmodlit{%s}\n\\end{theorem}\n\\begin{proof}\n  \\leanok\n  Trivial.\n\\end{proof}\n' "$decl" >> "$gt"
                        expect='modlit_bad=[1-9]' ;;
      linked_native_*|native_on_lithyp)
                        nat='\nnamespace GL2\ntheorem selftest_native : (2 : Nat) + 2 = 4 := by native_decide\nend GL2\n'
                        if [ "$ctl" = linked_native_outside ]; then
                          printf "$nat" >> "$tmp/GL2/Skeleton.lean"
                        else
                          printf "import GL2.Skeleton\n$nat" > "$tmp/GL2/Algorithms/SelftestNative.lean"
                          printf '\nimport GL2.Algorithms.SelftestNative\n' >> "$tmp/GL2.lean"
                        fi
                        case "$ctl" in
                          linked_native_thm)
                            printf '\n\\begin{theorem}[Selftest]\\label{thm:selftest-native}\n  \\lean{GL2.selftest_native}\n  \\leanok\n  $2+2=4$.\n\\end{theorem}\n\\begin{proof}\n  \\leanok\n  Compiled.\n\\end{proof}\n' >> "$gt" ;;
                          linked_native_chk)
                            sed -i 's/\\lean{GL2.SatChk.SatCert, /\\lean{GL2.selftest_native, GL2.SatChk.SatCert, /' "$gt"
                            grep -q 'lean{GL2.selftest_native, GL2.SatChk.SatCert' "$gt" || echo "CONTROL $ctl: plant failed" ;;
                          native_on_lithyp)
                            printf '\n\\begin{hypothesis}[Selftest]\\label{lit:selftest-native}\n  \\lean{GL2.selftest_native} \\leanok\n  $2+2=4$.\n\\end{hypothesis}\n' >> "$gt" ;;
                          linked_native_outside)
                            printf '\n\\begin{theorem}[Selftest]\\label{chk:selftest-native}\n  \\lean{GL2.selftest_native}\n  \\leanok\n  $2+2=4$.\n\\end{theorem}\n\\begin{proof}\n  \\leanok\n  Compiled.\n\\end{proof}\n' >> "$gt" ;;
                        esac
                        expect='global_axioms=[1-9]'; emit="--emit" ;;
      vacuous_proved)   printf '\nnamespace GL2\ntheorem selftest_vacuous (h : False) : (1 : Nat) = 2 := h.elim\nend GL2\n' >> "$tmp/GL2/Skeleton.lean"
                        printf '\n\\begin{theorem}[Selftest]\\label{thm:selftest-vacuous}\n  \\lean{GL2.selftest_vacuous}\n  \\leanok\n  $1=2$.\n\\end{theorem}\n\\begin{proof}\n  \\leanok\n  From False.\n\\end{proof}\n' >> "$gt"
                        expect='hyp_bad=[1-9]'; cexpect='hyp_bad=[1-9].*verdict=BAD'; emit="--emit" ;;
      skip_kernel)      printf '\nnamespace GL2\nset_option debug.skipKernelTC true in\ntheorem selftest_skip : (1 : Nat) = 2 := by decide +kernel\nend GL2\n' >> "$tmp/GL2/Skeleton.lean"
                        printf '\n\\begin{theorem}[Selftest]\\label{thm:selftest-skip}\n  \\lean{GL2.selftest_skip}\n  \\leanok\n  $1=2$.\n\\end{theorem}\n\\begin{proof}\n  \\leanok\n  Kernel skipped.\n\\end{proof}\n' >> "$gt"
                        expect='escapes=[1-9]'; cexpect='escapes=[1-9].*verdict=BAD'; emit="--emit" ;;
      lakefile_option)  sed -i 's/^autoImplicit = false$/autoImplicit = false\ndebug.skipKernelTC = true/' "$tmp/lakefile.toml"
                        grep -q '^debug.skipKernelTC = true' "$tmp/lakefile.toml" || echo "CONTROL $ctl: plant failed"
                        expect='escapes=[1-9]'; cexpect='escapes=[1-9].*verdict=BAD'; emit="--emit" ;;
      unlinked_native_alg)
                        printf 'import GL2.Skeleton\nnamespace GL2\ntheorem selftest_native_alg : (2 : Nat) + 2 = 4 := by native_decide\nend GL2\n' > "$tmp/GL2/Algorithms/SelftestNativeAlg.lean"
                        printf '\nimport GL2.Algorithms.SelftestNativeAlg\n' >> "$tmp/GL2.lean"
                        expect='native=[1-9].* escapes=[1-9]'; cexpect='escapes=[1-9].*verdict=BAD'; emit="--emit" ;;
    esac
    build "$tmp" >/dev/null
    line="$(python3 "$tmp/tools/lean_census.py" --root "$tmp" $emit | grep '^LEAN_CENSUS')"
    hit=no; echo "$line" | grep -Eq "$expect" && hit=yes
    echo "CONTROL $ctl: $line expect=$expect seen=$hit"
    case "$line" in *verdict=FAIL*) [ $hit = yes ] || fails=$((fails+1)) ;; *) fails=$((fails+1)) ;; esac
    if [ -n "$emit" ]; then  # the same planted, re-emitted census.tsv through the committed gate
      cline="$(python3 "$committed" --root "$tmp" | grep '^LEAN_CENSUS_COMMITTED')"
      chit=no; echo "$cline" | grep -Eq "$cexpect" && chit=yes
      echo "CONTROL $ctl committed: $cline expect=$cexpect seen=$chit"
      [ $chit = yes ] || fails=$((fails+1))
    fi
  done
  # S21 close: lake-free controls for the AnthropicFLT leanOptions allowance (lean_census.py VENDOR_OPTION_ALLOW):
  # the real lakefile (allowed key present) must give NO lakefile finding; a planted second key must be flagged.
  for ctl in vendor_allowed vendor_planted; do
    nctl=$((nctl+1)); rm -rf "$tmp"; mkdir -p "$tmp"; cp "$here/lakefile.toml" "$tmp/"
    [ $ctl = vendor_planted ] && sed -i 's/respectTransparency.types = false }/respectTransparency.types = false, debug.skipKernelTC = true }/' "$tmp/lakefile.toml"
    n="$(python3 -c "
import sys; sys.path.insert(0, '$here/tools'); import lean_census as c
print(sum(1 for w, _ in c.source_escapes('$tmp', []) if w == 'lakefile.toml'))")"
    want=0; [ $ctl = vendor_planted ] && want=1
    ok=no; [ "$n" = "$want" ] && ok=yes
    echo "CONTROL $ctl: lakefile_findings=$n expect=$want seen=$ok"
    [ $ok = yes ] || fails=$((fails+1))
  done
  rm -rf "$tmp"
  echo "LEAN_CENSUS_SELFTEST controls=$nctl controls_not_failing=$fails verdict=$([ $fails -eq 0 ] && echo OK || echo FAIL)"
  [ $fails -eq 0 ]; exit $?
fi

if [ -x "$tm" ]; then
  # Timing scratch: $LEAN_CENSUS_TIMEDIR (default ${XDG_CACHE_HOME:-$HOME/.cache}/hacs); never inside the repository.
  tdir="${LEAN_CENSUS_TIMEDIR:-${XDG_CACHE_HOME:-$HOME/.cache}/hacs}"; mkdir -p "$tdir"
  tf="$(mktemp "$tdir/.lean_census_time.XXXXXX")"; trap 'rm -f "$tf" "$tf.build" "$tf.scan"' EXIT
  export LEAN_CENSUS_TIMEFILE="$tf"
  build "$here" >/dev/null
  "$tm" -v -o "$tf.scan" python3 "$here/tools/lean_census.py" --root "$here" "$@"; rc=$?
  echo "LEAN_CENSUS_PEAK build_gib=$(peak "$tf.build") scan_gib=$(peak "$tf.scan") jobs=$jobs_n (budget: build <= 24, scan <= 10)"
  exit $rc
fi
build "$here" >/dev/null
python3 "$here/tools/lean_census.py" --root "$here" "$@"
