#
# Ludwig Docker image with Ray support and full dependencies including:
#   text features
#   image features
#   audio features
#   visualizations
#   hyperparameter optimization
#   distributed training
#   model serving
#

FROM rayproject/ray:2.54.0-py312

RUN sudo apt-get update && DEBIAN_FRONTEND="noninteractive" sudo apt-get install -y \
	build-essential \
	wget \
	git \
	curl \
	libsndfile1 \
	cmake \
	tzdata \
	rsync \
	vim \
	ffmpeg \
	sox \
	libsox-dev
RUN pip install -U pip

ARG LUDWIG_VERSION

WORKDIR /ludwig

RUN pip install --no-cache-dir torch==2.6.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

COPY . .
RUN if [ -n "${LUDWIG_VERSION}" ]; then \
      pip install --no-cache-dir "ludwig[full]==${LUDWIG_VERSION}" --extra-index-url https://download.pytorch.org/whl/cpu; \
    else \
      pip install --no-cache-dir '.[full]' --extra-index-url https://download.pytorch.org/whl/cpu; \
    fi
