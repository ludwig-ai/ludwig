#!/bin/bash

cluster_name="${1:-$CLUSTER_NAME}"
kubectl apply -f clusters/$cluster_name.yaml
