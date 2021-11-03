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

FROM rayproject/ray:latest

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

WORKDIR /ludwig

COPY . .
RUN HOROVOD_WITH_PYTORCH=1 \
    HOROVOD_WITHOUT_MPI=1 \
    HOROVOD_WITHOUT_TENSORFLOW=1 \
    HOROVOD_WITHOUT_MXNET=1 \
    pip install --no-cache-dir '.[full]'
