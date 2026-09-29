# syntax=docker/dockerfile:1

FROM python:3.11-slim

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
