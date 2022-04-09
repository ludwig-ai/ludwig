#!/bin/bash

cluster_name="${1:-$CLUSTER_NAME}"
kubectl delete -f clusters/$cluster_name.yaml
