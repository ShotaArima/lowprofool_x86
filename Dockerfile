# define base image
FROM ubuntu:24.04

# define maintainer
ARG PYTHON_VERSION=3.7.2

# Install dependencies
RUN apt-get update && apt-get install -y wget unzip

# Install pyenv
RUN PATH = "/root/.penv/bin:$PATH"

# Install dependencies
RUN MAKEOPTS = "j$(nproc)" pyenv install ${PYTHON_VERSION}

# Set the installed Python version as the default
RUN pyenv global ${PYTHON_VERSION}

# define workdir
WORKDIR /src/

COPY src/requirements.txt /src/
RUN eval $(pyenv init --path) && pip install -r requirements.txt
RUN echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
RUN echo 'eval "$(pyenv init -)"' >> ~/.bashrc
