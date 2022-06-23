#!/bin/bash

cluster_name="${1:-$CLUSTER_NAME}"
head_pod=$(kubectl get pods | grep $cluster_name-head | cut -d' ' -f1)
kubectl port-forward ${head_pod} 8267:8265
