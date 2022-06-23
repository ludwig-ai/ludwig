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

FROM python:3.8.13-slim

RUN apt-get -y update && apt-get -y install \
    git \
    libsndfile1 \
    build-essential \
    g++ \
    cmake
RUN pip install -U pip

WORKDIR /ludwig

COPY . .
RUN HOROVOD_WITH_PYTORCH=1 \
    HOROVOD_WITHOUT_MPI=1 \
    HOROVOD_WITHOUT_TENSORFLOW=1 \
    HOROVOD_WITHOUT_MXNET=1 \
    pip install --no-cache-dir '.[full]'

WORKDIR /data

ENTRYPOINT ["ludwig"]
