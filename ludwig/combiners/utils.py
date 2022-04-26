from ludwig.encoders.sequence_encoders import ParallelCNN, StackedCNN, StackedCNNRNN, StackedParallelCNN, StackedRNN

sequence_encoder_registry = {
    "stacked_cnn": StackedCNN,
    "parallel_cnn": ParallelCNN,
    "stacked_parallel_cnn": StackedParallelCNN,
    "rnn": StackedRNN,
    "cnnrnn": StackedCNNRNN,
    # todo: add transformer
    # 'transformer': StackedTransformer,
}
