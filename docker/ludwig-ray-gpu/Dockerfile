#
# Ludwig Docker image with Ray support and full dependencies including:
#   text features
#   image features
#   audio features
#   visualizations
#   hyperparameter optimization
#   distributed training
#   model serving
#

FROM rayproject/ray:2.9.0-py311-cu121


# Upgrade to GCC-9 from GCC-7.5
# Required for libgcc-s1 which is a dependency for libnccl2
RUN sudo apt-get update && sudo apt install -y software-properties-common && \
    sudo add-apt-repository ppa:ubuntu-toolchain-r/test && \
    sudo apt update && \
    DEBIAN_FRONTEND="noninteractive" sudo apt-get install -y \
    build-essential \
    wget \
    git \
    curl \
    libsndfile1 \
    cmake \
    tzdata \
    rsync \
    vim \
    gcc-9 \
    ffmpeg \
    sox \
    libsox-dev
RUN pip install -U pip

WORKDIR /ludwig

RUN pip install --no-cache-dir torch==2.1.0 torchtext torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

COPY . .
RUN pip install --no-cache-dir '.[full]'
