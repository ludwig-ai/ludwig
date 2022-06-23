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

FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

# https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212771
RUN sh -c 'echo "APT { Get { AllowUnauthenticated \"1\"; }; };" > /etc/apt/apt.conf.d/99allow_unauth' && \
    apt -o Acquire::AllowInsecureRepositories=true -o Acquire::AllowDowngradeToInsecureRepositories=true update && \
    apt-get install -y curl wget && \
    apt-key del 7fa2af80 && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/apt.conf.d/99allow_unauth cuda-keyring_1.0-1_all.deb && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC F60F4B3D7FA2AF80

RUN apt-get -y update && apt-get -y install \
    git \
    libsndfile1 \
    cmake
RUN pip install -U pip

WORKDIR /ludwig

COPY . .
RUN HOROVOD_GPU_OPERATIONS=NCCL \
    HOROVOD_WITH_PYTORCH=1 \
    HOROVOD_WITHOUT_MPI=1 \
    HOROVOD_WITHOUT_TENSORFLOW=1 \
    HOROVOD_WITHOUT_MXNET=1 \
    pip install --no-cache-dir '.[full]' && horovodrun --check-build

WORKDIR /data

ENTRYPOINT ["ludwig"]
