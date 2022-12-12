protoc --go_out=. --python_out=. \
    ludwig/profiling/proto/whylogs_messages.proto \
    ludwig/profiling/proto/dataset_profile.proto \
    --go_opt=paths=source_relative && {
    mv ludwig/profiling/proto/*.pb.go ludwig/profiling/go/
}
