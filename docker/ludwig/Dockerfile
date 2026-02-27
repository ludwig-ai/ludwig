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

FROM python:3.11-slim

RUN apt-get -y update && apt-get -y install \
    git \
    libsndfile1 \
    build-essential \
    g++ \
    cmake \
    ffmpeg \
    sox \
    libsox-dev
RUN pip install -U pip

WORKDIR /ludwig

RUN pip install --no-cache-dir torch==2.0.0 torchtext torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

COPY . .
RUN pip install --no-cache-dir '.[full]'

WORKDIR /data

ENTRYPOINT ["ludwig"]
