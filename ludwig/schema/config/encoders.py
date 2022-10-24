from dataclasses import dataclass
from typing import List


@dataclass
class InternalEncoderConfig:
    """Class for internal encoder parameters."""

    vocab: List[str] = None

    vocab_size: int = None

    should_embed: bool = True
