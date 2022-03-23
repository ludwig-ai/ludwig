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

FROM rayproject/ray:1.11.0-py37-cu111

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

# Temporary workaround until Ray 1.12
RUN pip install https://s3-us-west-2.amazonaws.com/ray-wheels/master/991a62dd4782e17a519ce29495b6e837f37f19f3/ray-2.0.0.dev0-cp37-cp37m-manylinux2014_x86_64.whl
RUN pip install google-cloud-bigquery[bqstorage,pandas]>=2.32.0
