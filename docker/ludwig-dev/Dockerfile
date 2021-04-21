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


FROM tensorflow/tensorflow:2.4.0-gpu

RUN apt-get -y update && apt-get -y install \
    git \
    libsndfile1 \
    cmake \
    libcudnn7=7.6.5.32-1+cuda10.1 \
    libnccl2=2.7.8-1+cuda10.1 \
    libnccl-dev=2.7.8-1+cuda10.1

RUN git clone --depth=1 https://github.com/ludwig-ai/ludwig.git \
    && cd ludwig/ \
    && HOROVOD_GPU_OPERATIONS=NCCL \
       HOROVOD_WITH_TENSORFLOW=1 \
       HOROVOD_WITHOUT_MPI=1 \
       HOROVOD_WITHOUT_PYTORCH=1 \
       HOROVOD_WITHOUT_MXNET=1 \
       pip install --no-cache-dir '.[test]'

WORKDIR /data

ENTRYPOINT ["ludwig"]
