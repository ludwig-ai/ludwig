# Open-Set Recognition with Agnostophobia Losses

This example reproduces the key findings from:

> Dhamija, A. R., Günther, M., & Boult, T. (2018).
> **Reducing Network Agnostophobia.**
> *NeurIPS 2018.* https://arxiv.org/abs/1811.04110

Standard classifiers are trained to output high-confidence predictions for every input — even inputs
from classes never seen during training. This is called *network agnostophobia*: the network is
incapable of expressing "I don't know."

The paper proposes two loss functions that address this:

| Loss                  | Description                                                                |
| --------------------- | -------------------------------------------------------------------------- |
| **Entropic Open-Set** | CE on known samples + entropy maximisation on background samples           |
| **Objectosphere**     | CE + logit-norm push for known + entropy + norm suppression for background |

Both are available in Ludwig's category and binary output features.

## Quick start

```bash
pip install ludwig
python train_open_set.py
```

The script generates a synthetic two-class-family dataset (four known Gaussian clusters + two unknown
clusters), trains three classifiers, and prints a comparison table showing mean max probability on
unknowns — lower is better for open-set recognition.

Expected output (approximate):

```
Model                  | Max-prob (known) | Max-prob (unknown) | Norm known | Norm unknown
-----------------------|-----------------|-------------------|------------|-------------
CE Baseline            |           0.998  |              0.741 |      8.828 |        5.375
Entropic Open-Set      |           0.974  |              0.273 |      6.254 |        0.637
Objectosphere          |           0.874  |              0.363 |     13.843 |        2.361
```

## Ludwig configuration

### Entropic Open-Set Loss

```yaml
output_features:
  - name: label
    type: category
    loss:
      type: entropic_open_set
      background_class: 4   # integer index of the background/unknown class
```

### Objectosphere Loss

```yaml
output_features:
  - name: label
    type: category
    loss:
      type: objectosphere
      background_class: 4
      xi: 10.0   # minimum logit norm for known-class samples
      zeta: 0.1  # weight for unknown-class magnitude suppression
```

`background_class` is the **integer index** of the background/unknown class in Ludwig's
vocabulary for that feature. You can discover it by inspecting the saved model's
`training_set_metadata.json` file after a training run — look for the `str2idx` field of the
relevant output feature.

## Inference-time unknown detection

For **Objectosphere** models, unknown inputs can be detected using a simple threshold on the logit
L2 norm:

```python
predictions = model.predict(dataset=df)

# Retrieve raw logits via the API (requires model.collect_activations)
import torch

norms = logit_tensor.norm(dim=-1)
is_unknown = norms < threshold  # choose threshold from validation set
```

For both loss types, you can also use the **maximum softmax probability** as a simpler threshold:
samples with max-prob below some value (e.g. 0.5) are flagged as unknown.
