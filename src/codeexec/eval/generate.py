"""Transformers generation wrapper for the base-model smoke test.

A thin factory that turns a loaded model + tokenizer into a GenerateFn. Greedy
(do_sample=False) matches the THESIS pass@1 definition (temperature 0). Lives in
the repo so the training phase can reuse or swap it (e.g. for vLLM); it is not
unit-tested here because it requires a loaded model (exercised in the Colab run).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def make_generate_fn(
    model: Any,
    tokenizer: Any,
    max_new_tokens: int = 256,
) -> Callable[[str], str]:
    """Return a prompt -> completion function over a loaded model + tokenizer.

    Greedy decoding (temperature 0), returning only the newly generated tokens
    (the continuation after the prompt), not the echoed prompt.
    """
    import torch  # local import: only needed where generation actually runs

    def generate(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs.input_ids.shape[1] :]
        return str(tokenizer.decode(new_tokens, skip_special_tokens=True))

    return generate
