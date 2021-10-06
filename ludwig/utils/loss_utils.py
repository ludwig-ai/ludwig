import torch


def rmspe_loss(targets: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
    """ Root mean square percentage error. """
    loss = torch.sqrt(torch.mean(
        ((targets - predictions).float() / targets) ** 2
    ))

    return loss


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
