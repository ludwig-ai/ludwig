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

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

RUN apt-get -y update && DEBIAN_FRONTEND="noninteractive" apt-get -y install \
    git \
    libsndfile1 \
    cmake \
    ffmpeg \
    sox \
    libsox-dev
RUN pip install -U pip

WORKDIR /ludwig

COPY . .
RUN pip install --no-cache-dir '.[full]'

WORKDIR /data

ENTRYPOINT ["ludwig"]
