# entmax

______________________________________________________________________

This package provides a pytorch implementation of entmax and entmax losses:
a sparse family of probability mappings and corresponding loss functions,
generalizing softmax / cross-entropy.

*Features:*

- Exact partial-sort algorithms for 1.5-entmax and 2-entmax (sparsemax).
- A bisection-based algorithm for generic alpha-entmax.
- Gradients w.r.t. alpha for adaptive, learned sparsity!

*Requirements:* python 3, pytorch >= 1.0 (and pytest for unit tests)

## Example

```python
import torch
from torch.nn.functional import softmax

from entmax import sparsemax, entmax15

x = torch.tensor([-2, 0, 0.5])

print(softmax(x, dim=0))
# tensor([0.0486, 0.3592, 0.5922])

print(sparsemax(x, dim=0))
# tensor([0.0000, 0.2500, 0.7500])

print(entmax15(x, dim=0))
# tensor([0.0000, 0.3260, 0.6740])
```

Gradients w.r.t. alpha (continued):

```python
import torch
from torch.autograd import grad

from entmax import entmax_bisect

x = torch.tensor([[-1, 0, 0.5], [1, 2, 3.5]])

alpha = torch.tensor(1.33, requires_grad=True)

p = entmax_bisect(x, alpha)

print(p)
# tensor([[0.0460, 0.3276, 0.6264],
#        [0.0026, 0.1012, 0.8963]], grad_fn=<EntmaxBisectFunctionBackward>)

print(grad(p[0, 0], alpha))
# (tensor(-0.2562),)
```

## Installation

```
pip install entmax
```

## Citations

[Sparse Sequence-to-Sequence Models](https://www.aclweb.org/anthology/P19-1146)

```
@inproceedings{entmax,
  author    = {Peters, Ben and Niculae, Vlad and Martins, Andr{\'e} FT},
  title     = {Sparse Sequence-to-Sequence Models},
  booktitle = {Proc. ACL},
  year      = {2019},
  url       = {https://www.aclweb.org/anthology/P19-1146}
}
```

[Adaptively Sparse Transformers](https://arxiv.org/pdf/1909.00015.pdf)

```
@inproceedings{correia19adaptively,
  author    = {Correia, Gon\c{c}alo M and Niculae, Vlad and Martins, Andr{\'e} FT},
  title     = {Adaptively Sparse Transformers},
  booktitle = {Proc. EMNLP-IJCNLP (to appear)},
  year      = {2019},
}
```

Further reading:

- Blondel, Martins, and Niculae, 2019. [Learning with Fenchel-Young Losses](https://arxiv.org/abs/1901.02324).
- Martins and Astudillo, 2016. [From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification](https://arxiv.org/abs/1602.02068).
- Peters and Martins, 2019 [IT-IST at the SIGMORPHON 2019 Shared Task: Sparse Two-headed Models for Inflection](https://www.aclweb.org/anthology/W19-4207).
