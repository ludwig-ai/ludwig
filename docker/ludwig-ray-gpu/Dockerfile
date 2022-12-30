#
# Ludwig Docker image with Ray nightly support and full dependencies including:
#   text features
#   image features
#   audio features
#   visualizations
#   hyperparameter optimization
#   distributed training
#   model serving
#

FROM rayproject/ray:2.0.0-py38-cu116

# https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212771
RUN sudo apt-key del 7fa2af80 && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb && \
    sudo dpkg -i cuda-keyring_1.0-1_all.deb && \
    sudo rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/apt.conf.d/99allow_unauth cuda-keyring_1.0-1_all.deb && \
    sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC F60F4B3D7FA2AF80

# Upgrade to GCC-9 from GCC-7.5
# Required for libgcc-s1 which is a dependency for libnccl2
RUN sudo apt-get update && sudo apt install -y software-properties-common && \
    sudo add-apt-repository ppa:ubuntu-toolchain-r/test && \
    sudo apt update && \
    sudo apt install -y gcc-9

RUN DEBIAN_FRONTEND="noninteractive" sudo apt-get install -y \
    build-essential \
    wget \
    git \
    curl \
    libsndfile1 \
    cmake \
    tzdata \
    rsync \
    vim

# Downgrade NCCL to 2.11.4 to for compatibility with CUDA 11.6
# https://github.com/horovod/horovod/issues/3625#issuecomment-1309786643
RUN sudo apt-get update && sudo apt install --allow-change-held-packages --allow-downgrades -y \
    libnccl2=2.11.4-1+cuda11.6 libnccl-dev=2.11.4-1+cuda11.6

RUN pip install -U pip

WORKDIR /ludwig

# RUN pip install torch==1.13.1 torchtext torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip install numpy --pre torch torchtext torchvision torchaudio --force-reinstall --extra-index-url https://download.pytorch.org/whl/nightly/cu116

COPY . .
RUN HOROVOD_GPU_OPERATIONS=NCCL \
    HOROVOD_WITH_PYTORCH=1 \
    HOROVOD_WITHOUT_MPI=1 \
    HOROVOD_WITHOUT_TENSORFLOW=1 \
    HOROVOD_WITHOUT_MXNET=1 \
    pip install --no-cache-dir '.[full]' --extra-index-url https://download.pytorch.org/whl/cu116 && \
    horovodrun --check-build && \
    python -c "import horovod.torch; horovod.torch.init()"
