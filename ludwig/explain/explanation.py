from dataclasses import dataclass, field
from typing import List

import numpy as np
import numpy.typing as npt


@dataclass
class LabelExplanation:
    """Stores the feature attributions for a single label in the target feature's vocab."""

    # The attribution for each input feature.
    feature_attributions: npt.NDArray[np.float64]


@dataclass
class Explanation:
    """Stores the explanations for a single row of input data.

    Contains the feature attributions for each label in the target feature's vocab.
    """

    target: str

    # The explanations for each label in the vocab of the target feature.
    label_explanations: List[LabelExplanation] = field(default_factory=list)

    def add(self, feature_attributions: npt.NDArray[np.float64]):
        """Add the feature attributions for a single label."""
        if len(self.label_explanations) > 0:
            # Check that the feature attributions are the same shape as existing explanations.
            assert self.label_explanations[0].feature_attributions.shape == feature_attributions.shape, (
                f"Expected feature attributions of shape {self.label_explanations[0].feature_attributions.shape}, "
                f"got {feature_attributions.shape}"
            )
        self.label_explanations.append(LabelExplanation(feature_attributions))

    def to_array(self) -> npt.NDArray[np.float64]:
        """Convert the explanation to a 2D array of shape (num_labels, num_features)."""
        return np.array([le.feature_attributions for le in self.label_explanations])
