#
# Ludwig Docker image with full set of pre-requiste packages to support development activites:
#   text features
#   image features
#   audio features
#   visualizations
#   hyperparameter optimization
#   distributed training
#   model serving
#   unit testing
#


FROM tensorflow/tensorflow:2.7.0-gpu

RUN apt-get -y update && apt-get -y install \
    git \
    libsndfile1 \
    cmake \
    libcudnn8=8.1.1.33-1+cuda11.2 \
    libnccl2=2.8.4-1+cuda11.2 \
    libnccl-dev=2.8.4-1+cuda11.2

WORKDIR /ludwig

COPY . .
RUN HOROVOD_GPU_OPERATIONS=NCCL \
    HOROVOD_WITH_TENSORFLOW=1 \
    HOROVOD_WITHOUT_MPI=1 \
    HOROVOD_WITHOUT_PYTORCH=1 \
    HOROVOD_WITHOUT_MXNET=1 \
    pip install --no-cache-dir '.[test]'

WORKDIR /data

ENTRYPOINT ["ludwig"]
