from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt

from ludwig.api_annotations import DeveloperAPI, PublicAPI


@DeveloperAPI
@dataclass
class FeatureAttribution:
    """Stores the attribution for a single input feature."""

    # The name of the input feature.
    feature_name: str

    # The scalar attribution for the input feature.
    attribution: float

    # (Optional) The attribution for each token in the input feature as an array of shape (seq_len, 2).
    token_attributions: List[Tuple[str, float]] = None


@DeveloperAPI
@dataclass
class LabelExplanation:
    """Stores the feature attributions for a single label in the target feature's vocab."""

    # The attribution for each input feature.
    feature_attributions: List[FeatureAttribution] = field(default_factory=list)

    def add(self, feature_name: str, attribution: float, token_attributions: List[Tuple[str, float]] = None):
        """Add the attribution for a single input feature."""
        self.feature_attributions.append(FeatureAttribution(feature_name, attribution, token_attributions))

    def to_array(self) -> npt.NDArray[np.float64]:
        """Convert the explanation to a 1D array of shape (num_features,)."""
        return np.array([fa.attribution for fa in self.feature_attributions])


@DeveloperAPI
@dataclass
class Explanation:
    """Stores the explanations for a single row of input data.

    Contains the feature attributions for each label in the target feature's vocab.
    """

    target: str

    # The explanations for each label in the vocab of the target feature.
    label_explanations: List[LabelExplanation] = field(default_factory=list)

    def add(
        self,
        feat_names: List[str],
        feat_attributions: npt.NDArray[np.float64],
        feat_to_token_attributions: Dict[str, List[Tuple[str, float]]] = None,
        prepend: bool = False,
    ):
        """Add the feature attributions for a single label."""
        assert len(feat_names) == len(
            feat_attributions
        ), f"Expected {len(feat_names)} feature attributions, got {len(feat_attributions)}"
        if len(self.label_explanations) > 0:
            # Check that the feature attributions are the same shape as existing explanations.
            assert self.label_explanations[0].to_array().shape == feat_attributions.shape, (
                f"Expected feature attributions of shape {self.label_explanations[0].to_array().shape}, "
                f"got {feat_attributions.shape}"
            )

        le = LabelExplanation()
        for i, feat_name in enumerate(feat_names):
            le.add(
                feat_name,
                feat_attributions[i],
                feat_to_token_attributions.get(feat_name) if feat_to_token_attributions else None,
            )
        self.label_explanations.insert(0, le) if prepend else self.label_explanations.append(le)

    def to_array(self) -> npt.NDArray[np.float64]:
        """Convert the explanation to a 2D array of shape (num_labels, num_features)."""
        return np.array([le.to_array() for le in self.label_explanations])


@PublicAPI(stability="experimental")
@dataclass
class ExplanationsResult:
    # Aggregate explanation for the entire input data.
    global_explanation: Explanation  # GlobalExplanation

    # A list of explanations, one for each row in the input data.
    # Each explanation contains the feature attributions for each label in the target feature's vocab.
    row_explanations: List[Explanation]

    # Expected value for each label in the target feature's vocab.
    expected_values: List[float]
