"""Pin down why Mbpp/599 scores 0/N at grade time though its canonical is trivial.

Walks the harness path step by step, in-process, on a single test, to localize the break:
    1. loader computes expected from canonical      (does fn eval run?)
    2. _to_source(expected) -> eval round-trip      (does the value survive embedding?)
    3. _compare(fresh_output, expected)             (does the grader accept a correct answer?)
    4. real sandbox run_tests on one test             (does the subprocess path agree?)
    
Run: python scripts/h0_probe_599.py 
"""

from __future__ import annotations

from codeexec.rewards.harness import _compare, _to_source, build_harness, run_tests
from codeexec.tasks.datasets import training_problems

TID = "Mbpp/599"

def main()-> None:
    p = training_problems()[TID]
    print(f"{TID}   entry={p.entry_point}   atol={p.atol!r}")
    print("canonical:")
    print(p.canonical_solution)
    
    # fresh function from the canonical
    ns: dict = {}
    exec(p.canonical_solution, ns)
    fn = ns[p.entry_point]
    
    tc = p.visible_tests[0]
    args, expected = tc.args, tc.expected
    print(f"\ntest[0] args={args!r}")
    print(f"    expected={expected!r}   type={type(expected).__name__}")
    
    #1. does fn run?
    got = fn(*args)
    print(f"    fn(*args)={got!r}   type={type(got).__name__}")
    
    #2. _to_source round-trip of expected
    src = _to_source(expected)
    print(f"    _to_source(expected)={src!r}")
    round_tripped = eval(src)
    print(f"    eval(src)={round_tripped!r} equal_to_expected={round_tripped == expected}")
    
    # 3. in-process _compare (bypasses subprocess)
    print(f"    _compare(got, expected, atol) = {_compare(got, expected, p.atol)}")
    print(f"    _compare(got, round_tripped, atol)={_compare(got, round_tripped, p.atol)}")
    
    #4. real sandbox on just this one test
    r = run_tests(p.canonical_solution, p.entry_point, [(args, expected)], p.atol)
    print(f"  sandbox run_tests single: passed={r.passed}/{r.total} "
          f"timed_out={r.timed_out} crashed={r.crashed}")
    
    #4b show the harness it generated for this one test (first/last lines)
    h = build_harness(p.canonical_solution, p.entry_point, [(args, expected)], p.atol)
    lines = h.splitlines()
    print("\n   --- generated harness (head) ---")
    for ln in lines[:12]:
        print("     ", ln)
    print(" --- (tail)  ---")
    for ln in lines[-6:]:
        print("     ", ln)

if __name__ == "__main__":
    main()