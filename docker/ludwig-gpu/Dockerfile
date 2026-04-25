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

FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

RUN apt-get -y update && DEBIAN_FRONTEND="noninteractive" apt-get -y install \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    git \
    libsndfile1 \
    cmake \
    ffmpeg \
    sox \
    libsox-dev

RUN python3.12 -m venv /opt/ludwig-venv
ENV PATH="/opt/ludwig-venv/bin:$PATH"
RUN pip install -U pip

ARG LUDWIG_VERSION

WORKDIR /ludwig

COPY . .
RUN if [ -n "${LUDWIG_VERSION}" ]; then \
      pip install --no-cache-dir "ludwig[full]==${LUDWIG_VERSION}" --extra-index-url https://download.pytorch.org/whl/cu124; \
    else \
      pip install --no-cache-dir '.[full]' --extra-index-url https://download.pytorch.org/whl/cu124; \
    fi
RUN pip install --no-cache-dir --force-reinstall torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --extra-index-url https://download.pytorch.org/whl/cu124

WORKDIR /data

ENTRYPOINT ["ludwig"]
