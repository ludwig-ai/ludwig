from ludwig.profiling import dataset_profile_pb2


def test_dataset_profile_works():
    dataset_profile = dataset_profile_pb2.DatasetProfile()
    dataset_profile.num_examples = 10

    from_serialized = dataset_profile_pb2.DatasetProfile()
    from_serialized.ParseFromString(dataset_profile.SerializeToString())

    assert from_serialized.num_examples == 10
