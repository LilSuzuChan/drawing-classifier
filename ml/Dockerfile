ARG PYTHON_IMAGE=python:3.12.15-slim-trixie@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.13.4@sha256:9f130421e930bfbdc563c00740466603f8723158e90297d07957c2294640262b

FROM ${UV_IMAGE} AS uv

FROM ${PYTHON_IMAGE} AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /src
COPY pyproject.toml uv.lock .python-version README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv lock --check \
    && uv sync --frozen --no-dev --group pipeline --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --group pipeline --no-editable

FROM ${PYTHON_IMAGE} AS mnist
ARG MNIST_BASE_URL=https://ossci-datasets.s3.amazonaws.com/mnist/
WORKDIR /opt/mnist/MNIST/raw
COPY docker/mnist.sha256 /tmp/mnist.sha256
RUN for file in $(awk '{print $2}' /tmp/mnist.sha256); do \
        python -c 'import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])' "${MNIST_BASE_URL}${file}" "${file}"; \
    done \
    && sha256sum --check --strict /tmp/mnist.sha256 \
    && gunzip ./*.gz

FROM ${PYTHON_IMAGE} AS runtime
RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid 1000 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin app
COPY --from=build /opt/venv /opt/venv
COPY --from=mnist /opt/mnist /opt/mnist
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    MNIST_DIR=/opt/mnist
USER 1000:1000
WORKDIR /tmp

FROM runtime AS test
USER root
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /src
COPY pyproject.toml uv.lock .python-version README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy \
    uv sync --frozen --group dev --group pipeline --no-install-project --inexact
COPY pyproject.toml train.py export.py model.py /work/ml/
COPY tests /work/ml/tests
COPY --from=web-models mnist_model.onnx /work/web/public/models/mnist_model.onnx
WORKDIR /work/ml
USER 1000:1000
ENV DRAWING_CLASSIFIER_IMAGE_TEST=1
RUN --network=none mkdir -p /tmp/pytest-report \
    && { python -m pytest -p no:cacheprovider -v -rfE --junitxml=/tmp/pytest-report/junit.xml > /tmp/pytest-report/output.txt 2>&1; echo $? > /tmp/pytest-report/exit-code; } \
    && cat /tmp/pytest-report/output.txt

FROM scratch AS test-report
COPY --from=test /tmp/pytest-report /
