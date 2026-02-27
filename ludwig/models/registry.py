import logging

from ludwig.constants import MODEL_ECD, MODEL_LLM
from ludwig.models.ecd import ECD
from ludwig.models.llm import LLM

logger = logging.getLogger(__name__)


model_type_registry = {
    MODEL_ECD: ECD,
    MODEL_LLM: LLM,
}
