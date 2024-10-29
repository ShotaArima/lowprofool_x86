# define base image
FROM ubuntu:24.04

# define maintainer
ARG PYTHON_VERSION=3.7.2

# Install dependencies
RUN set -x \
    && apt-get update \
    && apt-get install -y \
        curl \
        git \
        build-essential \
        zlib1g-dev \
        libncurses5-dev \
        libgdbm-dev \
        libnss3-dev \
        libssl-dev \
        libreadline-dev \
        libffi-dev \
        xz-utils \
        libbz2-dev \
        liblzma-dev \
        libsqlite3-dev \
    && curl -sSL https://pyenv.run | bash > /tmp/install-pyenv.sh \
    && chmod +x /tmp/install-pyenv.sh \
    && /tmp/install-pyenv.sh

# define path
ENV PATH="/root/.pyenv/bin:/root/.pyenv/shims:${PATH}"

# Install dependencies
    # 実行を早くするオプションPYTHON_CONFIGURE_OPS="--enable-shared"
RUN PYTHON_CONFIGURE_OPS="--enable-shared" pyenv install ${PYTHON_VERSION} \
    && pyenv global ${PYTHON_VERSION} \
    && pyenv rehash

# define workdir
WORKDIR /src/

COPY src/requirements.txt /src/
RUN eval $(pyenv init --path) && pip install -r requirements.txt
RUN echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
RUN echo 'eval "$(pyenv init -)"' >> ~/.bashrc
