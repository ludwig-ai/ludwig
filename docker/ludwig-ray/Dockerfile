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

FROM rayproject/ray:2.0.0-py38

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

# TODO(travis): remove this once we fix CUDA 11.6 NCCL version in base image
RUN pip install torch==1.12.1 torchtext torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

COPY . .
RUN HOROVOD_WITH_PYTORCH=1 \
	HOROVOD_WITHOUT_MPI=1 \
	HOROVOD_WITHOUT_TENSORFLOW=1 \
	HOROVOD_WITHOUT_MXNET=1 \
	pip install --no-cache-dir '.[full]' --extra-index-url https://download.pytorch.org/whl/cpu && \
	horovodrun --check-build && \
	python -c "import horovod.torch; horovod.torch.init()"
