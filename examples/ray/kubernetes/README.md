## Running on Kubernetes

### Connect to k8s cluster with a Ray operator

You should now be pointing to your cluster with `kubectl`. Check the nodes to make sure you're connected correctly:

```
kubectl get nodes
```

We recommend using the [Kuberay](https://github.com/ray-project/kuberay) implementation of the Ray Operator to launch Ray clusters.

### Configure the Ray cluster

First choose your preferred cluster template from `clusters`, for example:

```
export CLUSTER_NAME=ludwig-ray-cpu-cluster
```

### Start the cluster

```
./utils/ray_up.sh $CLUSTER_NAME
```

### Submit a script for execution

```
./utils/submit.sh $CLUSTER_NAME scripts/train_umber.py
```

### SSH into the head node

```
./utils/attach.sh $CLUSTER_NAME
```

### Run the Ray Dashboard

```
./utils/dashboard.sh $CLUSTER_NAME
```

Navigate to http://localhost:8267

### (For Ludwig Developers) Sync local Ludwig repo

```
./utils/rsync_up.sh $CLUSTER_NAME ~/repos/ludwig
```

### Shutdown the cluster

```
./utils/ray_down.sh $CLUSTER_NAME
```

### Connecting to remote filesystems (S3, GCS, etc.)

Build a custom Docker image deriving from `ludwig-ray` or `ludwig-ray-gpu` containing the library needed for your
data:

- `s3fs`
- `adlfs`
- `gcsfs`

Set environment variables into the cluster YAML definition with your credentials. For example, you can connect to S3 using the environment variables described in the [boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html#using-environment-variables).

You could also include the credentials directly into the Docker image if they don't need to be configured at runtime.
