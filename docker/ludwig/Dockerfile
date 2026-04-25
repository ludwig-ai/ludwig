#
# Ludwig Docker image with full set of pre-requiste packages to support these capabilities
#   text features
#   image features
#   audio features
#   visualizations
#   hyperparameter optimization
#   distributed training
#   model serving
#

FROM python:3.12-slim

RUN apt-get -y update && apt-get -y install \
    git \
    libsndfile1 \
    build-essential \
    g++ \
    cmake \
    ffmpeg \
    sox \
    libsox-dev
RUN pip install -U pip

ARG LUDWIG_VERSION

WORKDIR /ludwig

COPY . .
RUN if [ -n "${LUDWIG_VERSION}" ]; then \
      pip install --no-cache-dir "ludwig[full]==${LUDWIG_VERSION}" --extra-index-url https://download.pytorch.org/whl/cpu; \
    else \
      pip install --no-cache-dir '.[full]' --extra-index-url https://download.pytorch.org/whl/cpu; \
    fi
RUN pip install --no-cache-dir --force-reinstall torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --extra-index-url https://download.pytorch.org/whl/cpu

WORKDIR /data

ENTRYPOINT ["ludwig"]
