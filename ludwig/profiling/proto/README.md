## Generate proto code

1. Install `protoc`

```
> sudo apt install protobuf-compiler
```

2. Install the go protocol buffers plugin.

```
> go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
> go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
```

Useful reference: https://developers.google.com/protocol-buffers/docs/gotutorial

3. Update your PATH so that the protoc compiler can find the plugins:

```
export PATH="$PATH:$(go env GOPATH)/bin"
```

3. Execute protoc.

```
> sh ludwig/profiling/proto/gen-proto.sh
```

## Sample usage

Python:

```python
from ludwig.profiling import dataset_profile_pb2

dataset_profile = dataset_profile_pb2.DatasetProfile()
dataset_profile.num_examples = 10
dataset_profile.SerializeToString()
```

Go:

Run `go get`.

```
> go get github.com/ludwig-ai/ludwig/ludwig/profiling/go
```

If there are local changes, run push the commit to a branch and specify the commit:

```
> go get github.com/ludwig-ai/ludwig/ludwig/profiling/go@commit
```

Sample go code:

```go
package main

import (
	"fmt"

	dataset_profile "github.com/ludwig-ai/ludwig/ludwig/profiling"
)

func main() {
	// A protocol buffer can be created like any struct.
	p := &dataset_profile.DatasetProfile{}
	p.NumExamples = 3

	fmt.Println(p)
}
```

## Appendix

For protoc CLI help, check `man protoc`.
