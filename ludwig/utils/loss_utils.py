import torch


def rmspe_loss(targets: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
    """ Root mean square percentage error. """
    loss = torch.sqrt(torch.mean(
        ((targets - predictions).float() / targets) ** 2
    ))
    return loss


def mean_confidence_penalty(
        probabilities: torch.Tensor,
        num_classes: int
) -> torch.Tensor:
    max_entropy = torch.log(torch.tensor(num_classes))
    # clipping needed for avoiding log(0) = -inf
    entropy_per_class = torch.maximum(
        -probabilities * torch.log(torch.clamp(probabilities, 1e-10, 1)),
        0
    )
    entropy = torch.sum(entropy_per_class, -1)
    penalty = (max_entropy - entropy) / max_entropy
    return torch.mean(penalty)


# # used for categorical and sequence features
# def sample_values_from_classes(
#     labels,
#     sampler,
#     num_classes,
#     negative_samples,
#     unique,
#     class_counts,
#     distortion,
# ):
#     """returns sampled_values using the chosen sampler"""
#     if sampler == "fixed_unigram":
#         sampled_values = tf.random.fixed_unigram_candidate_sampler(
#             true_classes=labels,
#             num_true=1,
#             num_sampled=negative_samples,
#             unique=unique,
#             range_max=num_classes,
#             unigrams=class_counts,
#             distortion=distortion,
#         )
#     elif sampler == "uniform":
#         sampled_values = tf.random.uniform_candidate_sampler(
#             true_classes=labels,
#             num_true=1,
#             num_sampled=negative_samples,
#             unique=unique,
#             range_max=num_classes,
#         )
#     elif sampler == "log_uniform":
#         sampled_values = tf.random.log_uniform_candidate_sampler(
#             true_classes=labels,
#             num_true=1,
#             num_sampled=negative_samples,
#             unique=unique,
#             range_max=num_classes,
#         )
#     elif sampler == "learned_unigram":
#         sampled_values = tf.random.learned_unigram_candidate_sampler(
#             true_classes=labels,
#             num_true=1,
#             num_sampled=negative_samples,
#             unique=unique,
#             range_max=num_classes,
#         )
#     else:
#         raise ValueError("Unsupported sampler {}".format(sampler))
#     return sampled_values


#
# # For categorical feature
# def sampled_softmax_cross_entropy(
#     labels,
#     last_hidden,
#     num_classes=1,
#     decoder_weights=None,
#     decoder_biases=None,
#     sampler=None,
#     negative_samples=0,
#     class_counts=0,
#     distortion=1,
#     unique=False,
#     **kwargs
# ):
#     labels = tf.cast(tf.expand_dims(labels, -1), tf.int64)
#
#     sampled_values = sample_values_from_classes(
#         labels,
#         sampler,
#         num_classes,
#         negative_samples,
#         unique,
#         class_counts,
#         distortion,
#     )
#     train_loss = tf.nn.sampled_softmax_loss(
#         weights=tf.transpose(decoder_weights),
#         biases=decoder_biases,
#         labels=labels,
#         inputs=last_hidden,
#         num_sampled=negative_samples,
#         num_classes=num_classes,
#         sampled_values=sampled_values,
#     )
#
#     return train_loss
#
#
# # custom class to support Laplace smoothing of Fixed Unigram candidate sampler
# # Required because of zeros returned in the true_expected_count for
# # <PAD> and <UNK> tokens in loss['class_counts'] list
# class FixedUnigramCandidateSampler(
#     collections.namedtuple(
#         "FixedUnigramCandidateSampler",
#         (
#             "sampled_candidates",
#             "true_expected_count",
#             "sampled_expected_count",
#         ),
#     )
# ):
#     pass
#
#
# # For sequence feature
# def sequence_sampled_softmax_cross_entropy(
#     targets, train_logits, decoder_weights, decoder_biases, num_classes, **loss
# ):
#     batch_max_targets_sequence_length = tf.shape(targets)[1]
#     targets_sequence_length = sequence_length_2D(tf.cast(targets, tf.int64))
#     batch_max_train_logits_sequence_length = tf.shape(train_logits)[1]
#
#     logits_pad_len = tf.maximum(
#         0,
#         batch_max_targets_sequence_length
#         - batch_max_train_logits_sequence_length,
#     )
#     targets_pad_len = tf.maximum(
#         0,
#         batch_max_train_logits_sequence_length
#         - batch_max_targets_sequence_length,
#     )
#
#     padded_logits = tf.pad(train_logits, [[0, 0], [0, logits_pad_len], [0, 0]])
#     padded_targets = tf.pad(targets, [[0, 0], [0, targets_pad_len]])
#
#     output_exp = tf.cast(tf.reshape(padded_targets, [-1, 1]), tf.int64)
#     sampled_values = sample_values_from_classes(
#         output_exp,
#         loss["sampler"],
#         num_classes,
#         loss["negative_samples"],
#         loss["unique"],
#         loss["class_counts"],
#         loss["distortion"],
#     )
#
#     if loss["sampler"] == "fixed_unigram":
#         # regenerate sampled_values structure for specified samplers
#         # to handle any zero values in true_expected_count tensor
#         sampled_values = FixedUnigramCandidateSampler(
#             sampled_values.sampled_candidates,
#             # add smoothing constant EPSILON to handle any zero values
#             tf.add(sampled_values.true_expected_count, EPSILON),
#             sampled_values.sampled_expected_count,
#         )
#
#     def _sampled_loss(labels, logits):
#         labels = tf.cast(labels, tf.int64)
#         labels = tf.reshape(labels, [-1, 1])
#         logits = tf.cast(logits, tf.float32)
#
#         return tf.cast(
#             tf.nn.sampled_softmax_loss(
#                 weights=tf.transpose(decoder_weights),
#                 biases=decoder_biases,
#                 labels=labels,
#                 inputs=logits,
#                 num_sampled=loss["negative_samples"],
#                 num_classes=num_classes,
#                 sampled_values=sampled_values,
#             ),
#             tf.float32,
#         )
#
#     train_loss = tfa.seq2seq.sequence_loss(
#         padded_logits,
#         padded_targets,
#         tf.sequence_mask(
#             targets_sequence_length,
#             tf.shape(padded_targets)[1],
#             dtype=tf.float32,
#         ),
#         average_across_timesteps=True,
#         average_across_batch=False,
#         softmax_loss_function=_sampled_loss,
#     )
#
#     return train_loss
#
#