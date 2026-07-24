"""Context encoders for already embedded token or word sequences."""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Union, Optional

import torch
from torch import nn
from transformers import DebertaV2Config, ModernBertModel, ModernBertConfig, PretrainedConfig
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Encoder

from ..config import RNNEncoderConfig, normalize_context_encoder_config


class BaseContextEncoder(ABC, nn.Module):
    """Common interface for contextualizing dense sequence representations."""

    def __init__(self, input_size: int, encoder_size: int, output_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.encoder_size = encoder_size
        self.output_size = output_size
        self.input_projection = nn.Identity() if input_size == encoder_size else nn.Linear(input_size, encoder_size)
        self.output_projection = nn.Identity() if encoder_size == output_size else nn.Linear(encoder_size, output_size)

    @property
    @abstractmethod
    def requires_full_recompute(self) -> bool:
        """Whether appended positions can change all earlier representations."""

    @abstractmethod
    def _encode(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Run the architecture-specific encoder at ``encoder_size`` width."""

    def _parameter_dtype(self, fallback: torch.dtype) -> torch.dtype:
        parameter = next(self.parameters(), None)
        return parameter.dtype if parameter is not None and parameter.is_floating_point() else fallback

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, sequence, hidden_size]")
        if attention_mask.shape != hidden_states.shape[:2]:
            raise ValueError("attention_mask must match the first two hidden_states dimensions")

        hidden_states = hidden_states.to(dtype=self._parameter_dtype(hidden_states.dtype))
        hidden_states = self.input_projection(hidden_states)
        hidden_states = self._encode(hidden_states, attention_mask)
        hidden_states = self.output_projection(hidden_states)
        return hidden_states.masked_fill(~attention_mask.bool().unsqueeze(-1), 0)


class IdentityContextEncoder(BaseContextEncoder):
    """No-op encoder used when contextualization is disabled."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__(hidden_size, hidden_size, hidden_size)

    @property
    def requires_full_recompute(self) -> bool:
        return False

    def _encode(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return hidden_states


class DebertaV2ContextEncoder(BaseContextEncoder):
    """DeBERTa-v2 encoder stack operating directly on dense representations."""

    def __init__(self, config: DebertaV2Config, input_size: int, output_size: int) -> None:
        super().__init__(input_size, config.hidden_size, output_size)
        self.config = config
        self.encoder = DebertaV2Encoder(config)

    @property
    def requires_full_recompute(self) -> bool:
        return True

    def _encode(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return output.last_hidden_state


class ModernBertContextEncoder(BaseContextEncoder):
    """ModernBERT model using its public dense ``inputs_embeds`` path."""

    def __init__(self, config: ModernBertConfig, input_size: int, output_size: int) -> None:
        super().__init__(input_size, config.hidden_size, output_size)
        self.config = config

        # Be defensive when callers provide a pre-built config with the standard
        # vocabulary: no token IDs are consumed by this encoder.
        model_config = deepcopy(config)
        model_config.vocab_size = 1
        model_config.pad_token_id = 0
        model_config.bos_token_id = 0
        model_config.eos_token_id = 0
        model_config.cls_token_id = 0
        model_config.sep_token_id = 0
        model_config.tie_word_embeddings = False
        self.encoder = ModernBertModel(model_config)

    @property
    def requires_full_recompute(self) -> bool:
        return True

    def _encode(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return output.last_hidden_state


class RNNContextEncoder(BaseContextEncoder):
    """Packed LSTM context encoder with optional bidirectionality."""

    def __init__(self, config: RNNEncoderConfig, input_size: int, output_size: int) -> None:
        super().__init__(input_size, config.hidden_size, output_size)
        self.config = config
        num_directions = 2 if config.bidirectional else 1
        recurrent_hidden_size = config.hidden_size // num_directions
        self.encoder = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=recurrent_hidden_size,
            num_layers=config.num_hidden_layers,
            dropout=config.dropout if config.num_hidden_layers > 1 else 0.0,
            bidirectional=config.bidirectional,
            batch_first=True,
        )

    @property
    def requires_full_recompute(self) -> bool:
        return self.config.bidirectional

    def _encode(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        lengths = attention_mask.long().sum(dim=1)
        nonempty = lengths.gt(0)
        output = hidden_states.new_zeros(batch_size, sequence_length, self.encoder_size)
        if not nonempty.any():
            return output

        packed = pack_padded_sequence(
            hidden_states[nonempty],
            lengths[nonempty].detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.encoder(packed)
        encoded, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=sequence_length,
        )
        output[nonempty] = encoded
        return output


ContextEncoderConfigLike = Optional[Union[dict, PretrainedConfig]]


def build_context_encoder(
    encoder_config: ContextEncoderConfigLike,
    *,
    input_size: int,
    output_size: Optional[int] = None,
    dropout: float = 0.0,
) -> BaseContextEncoder:
    """Build a context encoder with a stable dense-input/dense-output contract."""
    if output_size is None:
        output_size = input_size
    if encoder_config is None:
        if input_size != output_size:
            raise ValueError("A disabled context encoder cannot change hidden size")
        return IdentityContextEncoder(input_size)

    config = normalize_context_encoder_config(
        encoder_config,
        hidden_size=input_size,
        dropout=dropout,
    )
    if isinstance(config, DebertaV2Config):
        return DebertaV2ContextEncoder(config, input_size, output_size)
    if isinstance(config, ModernBertConfig):
        return ModernBertContextEncoder(config, input_size, output_size)
    if isinstance(config, RNNEncoderConfig):
        return RNNContextEncoder(config, input_size, output_size)
    raise TypeError(f"Unsupported context encoder config: {type(config).__name__}")


__all__ = [
    "BaseContextEncoder",
    "DebertaV2ContextEncoder",
    "IdentityContextEncoder",
    "ModernBertContextEncoder",
    "RNNContextEncoder",
    "build_context_encoder",
]
