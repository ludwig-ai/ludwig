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

FROM rayproject/ray:1.12.1-py38-cu111

# https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212771
RUN sudo apt-key del 7fa2af80 && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb && \
    sudo dpkg -i cuda-keyring_1.0-1_all.deb && \
    sudo rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/apt.conf.d/99allow_unauth cuda-keyring_1.0-1_all.deb && \
    sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC F60F4B3D7FA2AF80

RUN sudo apt-get update && DEBIAN_FRONTEND="noninteractive" sudo apt-get install -y \
    build-essential \
    wget \
    git \
    curl \
    libsndfile1 \
    cmake \
    tzdata \
    rsync \
    vim
RUN pip install -U pip

WORKDIR /ludwig

COPY . .
RUN HOROVOD_GPU_OPERATIONS=NCCL \
    HOROVOD_WITH_PYTORCH=1 \
    HOROVOD_WITHOUT_MPI=1 \
    HOROVOD_WITHOUT_TENSORFLOW=1 \
    HOROVOD_WITHOUT_MXNET=1 \
    pip install --no-cache-dir '.[full]' -f https://download.pytorch.org/whl/cu111/torch_stable.html && \
    horovodrun --check-build && \
    python -c "import horovod.torch; horovod.torch.init()"
