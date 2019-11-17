#!/bin/bash

set -x -e

# create data sub-directory if it does not exist
if [[ ! -d data ]]; then
  echo ">>> create data sub-directory"
  mkdir data
fi

# download raw mnist imgage files https://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz
echo ">>> Start downloading mnist images"
(cd data && \
  curl -L -O https://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz && \
  tar -xf mnist_png.tar.gz && \
  rm mnist_png.tar.gz)
echo ">>> completed download of mnist images"

echo ">>> create ludwig formatted training/test data"

python prepare_mnist_data_for_ludwig.py

echo ">>> completed data preparation"
