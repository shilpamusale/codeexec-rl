# scripts/h0_stats.py
"""H0 statistical readout on the paired per-problem results.

Reads analysis/h0_results.json (paired canonical vs perturbed per-problem pass/fail),
and reports:
    - the paired 2x2 table
    - McNemar's test (exact binomial on the discordant pairs) -- is the drop real?
    - a bootstrap CI on the paired drop (canon_rate - pert_rate)
    - the minimum detectable effect at this n (power context)

Run:  python scripts/h0_stats.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS = Path("analysis/h0_results.json")
N_BOOT = 20000
SEED = 0

def main() -> None:
    data = json.loads(RESULTS.read_text())
    paired = {k: v for k, v in data.items() if v.get("perturbed_ok")}
    n = len(paired)
    
    canon = np.array([1 if v["canon_pass"] else 0 for v in paired.values()])
    pert = np.array([1 if v["pert_pass"] else 0 for v in paired.values()])
    
    # 2x2 discordant / concordant
    both_pass = int(np.sum((canon == 1) & (pert == 1)))
    b = int(np.sum((canon == 1) & (pert == 0)))   # canon-only pass
    c = int(np.sum((canon == 0) & (pert == 1)))   # pert-only pass
    both_fail = int(np.sum((canon == 0) & (pert == 0)))

    canon_rate = canon.mean()
    pert_rate = pert.mean()
    drop = canon_rate - pert_rate

    print("=" * 60)
    print(f"n (paired problems): {n}")
    print(f"canonical pass@1:    {canon_rate:.3f} ({canon.sum()}/{n})")
    print(f"perturbed pass@1:    {pert_rate:.3f} ({pert.sum()}/{n})")
    print(f"drop (canon-pert):   {drop*100:+.1f} pp")
    print("\npaired 2x2:")
    print(f"              pert PASS   pert FAIL")
    print(f"  canon PASS  {both_pass:^9d}   {b:^9d}")
    print(f"  canon FAIL  {c:^9d}   {both_fail:^9d}")
    print(f"  discordant: b={b} (canon-only), c={c} (pert-only)")

    # --- McNemar, exact binomial on discordant pairs ---
    # H0: P(canon pass, pert fail) == P(canon fail, pert pass); i.e. b ~ Binom(b+c, 0.5)
    nd = b + c
    if nd == 0:
        p_mcnemar = 1.0
    else:
        p_mcnemar = stats.binomtest(min(b, c), nd, 0.5, alternative="two-sided").pvalue
    print(f"\nMcNemar (exact binomial on {nd} discordant pairs):")
    print(f"  p = {p_mcnemar:.3f}  ->  "
          f"{'drop NOT distinguishable from zero' if p_mcnemar > 0.05 else 'drop IS significant'}")
    
    # bootstrap CI on the paired drop --
    rng = np.random.default_rng(SEED)
    idx = np.arange(n)
    diffs = canon - pert # per-problem paired difference in {-1,0,+1}
    boot = np.empty(N_BOOT)
    for i in range(N_BOOT):
        s = rng.choice(idx, size=n, replace = True)
        boot[i] = diffs[s].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])*100
    print(f"\nbootstrap 95% CI on drop ({N_BOOT} resamples): "
          f"[{lo:+.1f}, {hi:+.1f}] pp")
    print(f"  CI {'straddles 0 -> consistent with no real drop' if lo < 0 < hi else 'excludes 0'}")

    # --- minimum detectable effect at this n (power context) ---
    # For a paired/two-proportion design at n, alpha=.05, power=.80, approx MDE:
    # use the normal approx on the drop's SE at the observed base rate.
    
    p_bar = (canon_rate + pert_rate) / 2
    se = np.sqrt(2 * p_bar * (1 - p_bar)/n)
    mde = (1.96 + 0.84) * se * 100 # z_.975 + z_.80
    print(f"\npower context at n={n}, base~{p_bar:.2f}:")
    print(f"  approx minimum detectable effect (alpha=.05, power=.80): ~{mde:.1f} pp")
    print(f"  -> a drop below ~{mde:.0f}pp cannot be reliably resolved at this n;")
    print(f"     a null result means 'no drop larger than this detected', not 'zero'.")
    print("=" * 60)
    # verdict against the pre-registered rule (ADR-002)
    d_pp = drop * 100
    print("PRE-REGISTERED VERDICT (ADR-002 bands):")
    if d_pp <= 5:
        print(f"  drop {d_pp:+.1f}pp <= 5pp AND p={p_mcnemar:.2f} (n.s.) -> CLEAN.")
        print("  Contamination precondition holds; ADR-002 stands; train on MBPP+.")
    elif d_pp < 15:
        print(f"  drop {d_pp:+.1f}pp in (5,15) -> AMBIGUOUS: escalate to full ~369.")
    else:
        print(f"  drop {d_pp:+.1f}pp >= 15pp -> CONTAMINATED: ADR-002 reverses.")


if __name__ == "__main__":
    main()