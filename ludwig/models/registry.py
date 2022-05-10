from ludwig import constants
from ludwig.models.ecd import ECD
from ludwig.models.gbm import GBM

model_type_registry = {
    constants.ECD: ECD,
    constants.GBM: GBM,
}
