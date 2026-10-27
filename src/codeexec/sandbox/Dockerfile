# syntax=docker/dockerfile:1

# The base image, pinned by its multi-platform index digest, so every machine
# builds the same Python and glibc: Linux is the platform of record, and
# bit-exact float results depend on the math library (ADR-015). The tag is for
# readers; Docker uses the digest. The Section A grader-parity check ran on it.
FROM python:3.11-slim@sha256:cce12316475d7d56d605f76e1d811d0dc4e7227f2d4afd29951521264f504ff8

# Don't write .pyc bytecode files (keeps the read-only filesystem happy later)
# and don't buffer stdout/stderr (so we capture output reliably).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create an unprivileged user to run untrusted code as. UID 1000 is a
# standard non-root user id. A container escape lands here — powerless.
RUN useradd --create-home --uid 1000 sandbox

# The directory untrusted code runs in, owned by the sandbox user.
WORKDIR /sandbox
RUN chown sandbox:sandbox /sandbox

# Drop from root to the unprivileged user. Everything after this runs as
# sandbox, not root — including the untrusted code.
USER sandbox
