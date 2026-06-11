# syntax=docker/dockerfile:1
FROM python:3.12-slim

# graphviz - rendered visualisation output.
# time     - GNU /usr/bin/time, used by the experiment runner scripts.
# bash, coreutils (provides /usr/bin/timeout) are already in the slim base.
RUN apt-get update \
 && apt-get install -y --no-install-recommends graphviz time bash \
 && rm -rf /var/lib/apt/lists/* \
 && ln -s /usr/bin/time    /usr/local/bin/gtime \
 && ln -s /usr/bin/timeout /usr/local/bin/gtimeout

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
