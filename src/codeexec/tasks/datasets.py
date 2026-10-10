"""The experiment's task roles, over the normalized Problem loaders.
The loaders in loaders.py are named by dataset (MBPP+, HumanEval+). The experiment thinks
in roles instead, and this module names them:

 - training problems:MBPP. The policy trains on these, scored on their visible tests
 (the canonical MBPP tests). Their held-out tests (MBPP+ augmented) are the measurement
 `heldout_pass` is computed on. Never enters any reward.

 -OOD problems: HumanEval+. Separate problems, never trained on and never in any reward -
 measured only, as the out-of-distribution generalization check.

The visible/held-out partition itself already lives on each Problem, by construction in the
loaders:
    visible_tests = canonical tests (base_input) -> `visible_pass`, reward
    heldout_tests = augmented tests (plus_input) -> `heldout_pass`
Only the reward differs between arms; this measurement split is fixed here and
matches the metric definitions in THESIS.md
"""

from codeexec.tasks.loaders import load_humaneval, load_mbpp
from codeexec.tasks.problem import Problem


def training_problems() -> dict[str, Problem]:
    """The problems the policy trains on (MBPP), keyed by task_id.

    Scored on visible tests; help-out tests measure true correctness. The
    held-out tests never enter any reward computation.
    """
    return load_mbpp()


def ood_problems() -> dict[str, Problem]:
    """The out-of-distribution check problems (HumanEval+), keyed by task_id.
    Never trained on and never in any reward - measured only.
    """
    return load_humaneval()
