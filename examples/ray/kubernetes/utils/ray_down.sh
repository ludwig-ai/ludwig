#!/bin/bash

cluster_name="${1:-$CLUSTER_NAME}"
kubectl delete -f configs/engines/user/$cluster_name.yaml
