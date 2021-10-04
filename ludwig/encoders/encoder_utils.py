import torch


def sequence_mask(lengths: torch.Tensor, maxlen: int,
                  dtype: torch.dtype = torch.bool):
    """Returns a mask tensor representing the first N positions of each cell."""
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix
    mask.type(dtype)
    return mask
