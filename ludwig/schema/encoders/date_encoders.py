from typing import ClassVar, List
from ludwig.encoders.base import Encoder
from ludwig.encoders.binary_encoders import BinaryPassthroughEncoder

from marshmallow_dataclass import dataclass
from ludwig.schema import utils as schema_utils


@dataclass
class DateEmbedEncoderConfig(schema_utils.BaseMarshmallowConfig):

    encoder_class: ClassVar[Encoder] = BinaryPassthroughEncoder

    type: str = "embed"

    embedding_size: int = 10,

    embeddings_on_cpu: bool = False,
    
    fc_layers: List[dict] = None,

    num_fc_layers: int = 0,

    output_size: int = 10,

    use_bias: bool = True,

    weights_initializer: str = "xavier_uniform",

    bias_initializer: str = "zeros",

    norm: str = None,

    norm_params: Dict = None,

    activation: str = "relu",

    dropout: float = 0,
