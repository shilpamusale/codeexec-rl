"""The sandbox's base image stays pinned (ADR-015).

Bit-exact float results depend on the image's glibc, and Linux is the
platform of record, so the image must be the same on every machine and
every day. A tag alone moves when Docker Hub republishes it; a digest
cannot.
"""

from __future__ import annotations

import re
from pathlib import Path

DOCKERFILE = Path(__file__).parents[1] / "src" / "codeexec" / "sandbox" / "Dockerfile"


def test_base_image_is_pinned_by_digest() -> None:
    bases = [
        line.split()[1] for line in DOCKERFILE.read_text().splitlines() if line.startswith("FROM ")
    ]

    assert len(bases) == 1
    assert re.fullmatch(r"python:3\.11-slim@sha256:[0-9a-f]{64}", bases[0]), bases[0]
