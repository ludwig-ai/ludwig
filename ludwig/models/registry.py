from ludwig.constants import MODEL_ECD, MODEL_GBM
from ludwig.models.ecd import ECD
from ludwig.models.gbm import GBM

model_type_registry = {
    MODEL_ECD: ECD,
    MODEL_GBM: GBM,
}
