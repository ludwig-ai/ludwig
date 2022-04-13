#!/bin/bash

cluster_name="${1:-$CLUSTER_NAME}"
py_script=$2

head_pod=$(kubectl get pods | grep $cluster_name-head | cut -d' ' -f1)
fname=$(basename $py_script)

kubectl cp $py_script $head_pod:/home/ray/.
echo /home/ray/$fname
