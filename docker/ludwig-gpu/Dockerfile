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

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

RUN apt-get -y update && DEBIAN_FRONTEND="noninteractive" apt-get -y install \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get -y update && \
    DEBIAN_FRONTEND="noninteractive" apt-get -y install \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    git \
    libsndfile1 \
    cmake \
    ffmpeg \
    sox \
    libsox-dev

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    python3.12 -m ensurepip && \
    pip install -U pip

WORKDIR /ludwig

RUN pip install --no-cache-dir torch==2.6.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124

COPY . .
RUN pip install --no-cache-dir '.[full]'

WORKDIR /data

ENTRYPOINT ["ludwig"]
