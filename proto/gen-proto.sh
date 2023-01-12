protoc --go_out=. --python_out=. \
    proto/dataset_profile.proto \
    --go_opt=paths=source_relative && {
    mv proto/*.py ludwig/profiling
} && {
    mv proto/*.go proto/go
}
