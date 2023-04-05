FROM bitnami/spark:3.2.2 as spark-base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    \
    POETRY_VERSION=1.2.0 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=false \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    PYSETUP_PATH="/opt/pysetup" \
    VENV_PATH="/opt/pysetup/.venv"

# prepend poetry and venv to path
ENV PATH="$POETRY_HOME/bin:$PATH"
ENV PATH ~/.local/bin:$PATH
ENV PATH /root/.local/bin:$PATH
ENV AM_I_IN_A_DOCKER_CONTAINER Yes

# `builder-base` stage is used to build deps + create virtual environment
FROM spark-base as builder-base
WORKDIR /app
ENV HADOOP_VERSION="3.2.0"
ENV SPARK_VERSION="3.1.1"
ENV AWS_VERSION="1.11.874"
ENV SPARK_HADOOP_VERSION="3.2"
ENV AWS_REGION="eu-west-1"
# Install project dependencies
USER root
RUN mkdir -p /var/lib/apt/lists/partial
RUN apt-get -qq update \
    && apt-get -qq install -y --no-install-recommends curl \
    && curl -sSL https://install.python-poetry.org | POETRY_VERSION=$POETRY_VERSION python3 - \
    && apt-get -qq -y remove curl \
    && apt-get -qq -y autoremove \
    && rm -rf /var/lib/apt/lists/*

# copy & poetry install
COPY poetry.lock pyproject.toml ./
COPY ./shapeshifter /app/shapeshifter
RUN poetry config installer.max-workers 4
RUN poetry install --only main


FROM builder-base as final-build
ENV PYTHONPATH /app/
ENV PYTHONPATH=$SPARK_HOME/python/:$PYTHONPATH
ENV PYTHONPATH=$SPARK_HOME/python/lib/py4j-0.10.9.5-src.zip:$PYTHONPATH
COPY ./tests /app/tests
# Add Certificates and point to python module
COPY nike-root.crt /usr/local/share/ca-certificates/nike/nike-root.crt
COPY nike-tls-ca.crt /usr/local/share/ca-certificates/nike/nike-tls-ca.crt
RUN update-ca-certificates
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
