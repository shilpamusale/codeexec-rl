"""The experiment's task roles, over the normalized Problem loaders.
The loaders in loaders.py are named by dataset (MBPP+, HumanEval+). The experiment thinks
in roles instead, and this module names them:

- training problems: MBPP. The policy trains on these. V1 is rewarded on their visible
   tests (the canonical MBPP tests); V0 also on held-out split A. `heldout_pass` is measured
   on held-out split B, which never enters any reward (ADR-013, tasks/split.py).

 -OOD problems: HumanEval+. Separate problems, never trained on and never in any reward -
 measured only, as the out-of-distribution generalization check.

The visible/held-out partition itself already lives on each Problem, by construction in the
loaders:
    visible_tests = canonical tests (base_input) -> `visible_pass`, reward
    heldout_tests = augmented tests (plus_input), split by tasks/split.py into
                      A (V0's reward only) and B (-> `heldout_pass`, never in any reward)
Only the reward differs between arms; this measurement split is fixed here and
matches the metric definitions in THESIS.md
"""

from codeexec.tasks.loaders import load_humaneval, load_mbpp
from codeexec.tasks.problem import Problem


def training_problems() -> dict[str, Problem]:
    """The problems the policy trains on (MBPP), keyed by task_id.

    Scored on visible tests; held-out tests measure true correctness. The
    held-out tests never enter any reward computation.
    """
    return load_mbpp()


def ood_problems() -> dict[str, Problem]:
    """The out-of-distribution check problems (HumanEval+), keyed by task_id.
    Never trained on and never in any reward - measured only.
    """
    return load_humaneval()
