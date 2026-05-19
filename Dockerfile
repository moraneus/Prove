# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Optional graphviz support for rendered visualisation output.
RUN apt-get update \
 && apt-get install -y --no-install-recommends graphviz \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/prove

# Copy package metadata and source for installation.
COPY pyproject.toml README.md ./
COPY prove ./prove

RUN pip install --no-cache-dir .

# /data is the conventional mount point for user-supplied property and trace files.
WORKDIR /data

# `prove` is registered as a console script in pyproject.toml.
ENTRYPOINT ["prove"]
CMD ["--help"]
