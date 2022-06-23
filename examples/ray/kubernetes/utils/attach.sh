#!/bin/bash

cluster_name="${1:-$CLUSTER_NAME}"
head_pod=$(kubectl get pods | grep $cluster_name-head | cut -d' ' -f1)
kubectl exec -it $head_pod -- /bin/bash
