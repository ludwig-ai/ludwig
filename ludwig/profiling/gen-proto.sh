protoc --go_out=. --python_out=. \
    ludwig/profiling/whylogs_messages.proto \
    ludwig/profiling/dataset_profile.proto \
    --go_opt=paths=source_relative && {
    mv ludwig/profiling/*.pb.go ludwig/profiling/go/
} && {
    mv ludwig/profiling/*pb2.py ludwig/profiling/python
}
