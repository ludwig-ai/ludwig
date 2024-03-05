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

FROM rayproject/ray:2.3.1-py38

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

RUN pip install --no-cache-dir torch==2.0.0 torchtext torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

COPY . .
RUN pip install --no-cache-dir '.[full]' --extra-index-url https://download.pytorch.org/whl/cpu
