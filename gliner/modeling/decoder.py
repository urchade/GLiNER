"""Decoder modules for autoregressive text generation with optional constraints.

This module provides decoder architectures built on causal language models, supporting
both standard generation and prefix-constrained decoding using trie structures. It
includes custom generation implementations and numerical stability improvements.
"""

import warnings
from typing import Any, List, Union, Optional
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, LogitsProcessor, LogitsProcessorList, AutoModelForCausalLM

from ..utils import is_module_available
from ..decoding.trie import LabelsTrie

# Check for optional dependencies
IS_PEFT = is_module_available("peft")

if IS_PEFT:
    from peft import LoraConfig, get_peft_model


class NumericalStabilityProcessor(LogitsProcessor):
    """Logits processor that ensures numerical stability during generation.

    This processor handles edge cases in logit values by replacing negative infinity
    values with the minimum representable value for the dtype, clamping extreme values,
    and adding a small epsilon for stability.

    Attributes:
        epsilon: Small constant added to logits for numerical stability.
    """

    def __init__(self, epsilon: float = 1e-6) -> None:
        """Initializes the numerical stability processor.

        Args:
            epsilon: Small constant to add to logits. Defaults to 1e-6.
        """
        self.epsilon = epsilon

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Processes logits to ensure numerical stability.

        Replaces negative infinity values, clamps extreme values to prevent
        overflow/underflow, and adds epsilon for stability.

        Args:
            input_ids: Previously generated token IDs of shape (batch_size, seq_len).
            scores: Raw logit scores of shape (batch_size, vocab_size).

        Returns:
            Stabilized logit scores of shape (batch_size, vocab_size).
        """
        scores = torch.where(
            torch.isneginf(scores), torch.tensor(torch.finfo(scores.dtype).min).to(scores.device), scores
        )
        scores = torch.clamp(scores, min=-1e9, max=1e9)
        return scores + self.epsilon


class DecoderTransformer(nn.Module):
    """Wrapper for causal language model decoders with adapter support.

    This class provides a unified interface for autoregressive decoder models,
    supporting loading from pretrained weights or initialization from config.
    It also handles PEFT/LoRA adapter loading when available.

    Attributes:
        model: The underlying causal language model instance.
        config: Configuration object containing model hyperparameters.
    """

    def __init__(
        self, model_name: str, config: Any, from_pretrained: bool = False, cache_dir: Optional[Union[str, Path]] = None
    ) -> None:
        """Initializes the decoder transformer.

        Args:
            model_name: Name or path of the pretrained model to load.
            config: Configuration object containing model hyperparameters. Must have
                a `labels_decoder_config` attribute.
            from_pretrained: If True, loads pretrained weights. If False, initializes
                from config only. Defaults to False.
            cache_dir: Optional directory for caching downloaded models. Defaults to None.

        Raises:
            Warning: If adapter config is found but PEFT package is not installed.
        """
        super().__init__()
        decoder_config = config.labels_decoder_config
        if decoder_config is None:
            decoder_config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)

        kwargs = {}
        custom = False
        ModelClass = AutoModelForCausalLM

        if from_pretrained:
            self.model = ModelClass.from_pretrained(model_name, trust_remote_code=True)
        elif not custom:
            self.model = ModelClass.from_config(decoder_config, trust_remote_code=True)
        else:
            self.model = ModelClass(decoder_config, **kwargs)

        adapter_config_file = Path(model_name) / "adapter_config.json"

        if adapter_config_file.exists():
            if not IS_PEFT:
                warnings.warn(
                    "Adapter configs were detected, if you want to apply them you need to install peft package.",
                    stacklevel=2,
                )
            else:
                adapter_config = LoraConfig.from_pretrained(model_name)
                self.model = get_peft_model(self.model, adapter_config)

        self.config = config

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass through the decoder model.

        Args:
            *args: Variable positional arguments passed to the model.
            **kwargs: Variable keyword arguments passed to the model.

        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size).
        """
        output = self.model(*args, **kwargs)
        encoder_layer = output[0]
        return encoder_layer


class Decoder(nn.Module):
    """High-level decoder interface for autoregressive generation.

    This class provides a unified interface for text generation from embeddings,
    supporting both standard generation and constrained decoding using trie structures.
    It includes custom generation implementations and integrates with Hugging Face's
    generation API.

    Attributes:
        decoder_layer: The underlying DecoderTransformer instance.
        decoder_hidden_size: Hidden dimension size of the decoder model.
    """

    def __init__(
        self, config: Any, from_pretrained: bool = False, cache_dir: Optional[Union[str, Path]] = None
    ) -> None:
        """Initializes the decoder.

        Args:
            config: Configuration object containing model hyperparameters including
                `labels_decoder` (model name) and decoder-specific settings.
            from_pretrained: If True, loads pretrained weights for the decoder.
                Defaults to False.
            cache_dir: Optional directory for caching downloaded models. Defaults to None.
        """
        super().__init__()

        self.decoder_layer = DecoderTransformer(config.labels_decoder, config, from_pretrained, cache_dir=cache_dir)

        self.decoder_hidden_size = self.decoder_layer.model.config.hidden_size

    def ids_to_embeds(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        """Converts token IDs to their corresponding embeddings.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len).

        Returns:
            Token embeddings of shape (batch_size, seq_len, hidden_size).
        """
        input_ids = input_ids.to(self.decoder_layer.model.device)
        embedding_layer = self.decoder_layer.model.get_input_embeddings()
        return embedding_layer(input_ids)

    @torch.inference_mode()
    def generate_from_embeds_custom(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 32,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        temperature: float = 1.0,
        do_sample: bool = False,
        labels_trie: Optional[LabelsTrie] = None,
        **kwargs: Any,
    ) -> torch.LongTensor:
        """Custom generation implementation from embeddings with optional trie constraints.

        This method implements token-by-token generation with KV caching and support for
        trie-based constrained decoding. Unlike the standard generate method, this
        implementation provides more control over the generation process and handles
        trie constraints at each step.

        Args:
            inputs_embeds: Input embeddings of shape (batch_size, prefix_len, hidden_size)
                serving as the generation prefix.
            attention_mask: Optional attention mask of shape (batch_size, prefix_len).
                If None, assumes all prefix tokens are valid. Defaults to None.
            max_new_tokens: Maximum number of new tokens to generate. Defaults to 32.
            eos_token_id: Token ID marking end of sequence. If None, uses model's
                default. Defaults to None.
            pad_token_id: Token ID for padding. If None, uses model's default or
                eos_token_id. Defaults to None.
            temperature: Sampling temperature for controlling randomness. Values < 1
                make distribution sharper, > 1 make it more uniform. Defaults to 1.0.
            do_sample: If True, uses multinomial sampling. If False, uses greedy
                decoding (argmax). Defaults to False.
            labels_trie: Optional trie structure for constrained decoding. At each
                step, only tokens that follow valid trie paths are allowed.
                Defaults to None.
            **kwargs: Additional keyword arguments (currently unused).

        Returns:
            Generated token IDs of shape (batch_size, generated_len) where generated_len
            varies per sequence based on when EOS is reached. Sequences are padded to
            the same length with pad_token_id.
        """
        model = self.decoder_layer.model
        device, (B, L0, _) = inputs_embeds.device, inputs_embeds.shape
        cfg = model.config

        eos_token_id = eos_token_id or cfg.eos_token_id
        pad_token_id = pad_token_id or cfg.pad_token_id or eos_token_id

        # prefix mask
        if attention_mask is None:
            attention_mask = torch.ones(B, L0, dtype=torch.long, device=device)

        out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=True)
        past_key_values = out.past_key_values
        next_logits = out.logits[:, -1]  # (B, V)

        unfinished = torch.ones(B, dtype=torch.bool, device=device)
        generated = [[] for _ in range(B)]

        for _ in range(max_new_tokens):
            if labels_trie is not None:
                V = next_logits.shape[1]
                mask_tensor = torch.full((B, V), -float("inf"), device=device)
                for b in range(B):
                    if unfinished[b]:
                        current_seq = generated[b]  # Tokens generated so far

                        allowed_tokens = labels_trie.get(current_seq)

                        if len(allowed_tokens) == 0:
                            allowed_tokens = [eos_token_id]

                        mask_tensor[b, allowed_tokens] = 0
                    else:
                        mask_tensor[b, :] = 0
                next_logits = next_logits + mask_tensor

            if temperature != 1.0:
                next_logits = next_logits / temperature

            if do_sample:
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)  # (B, 1)

            for b in range(B):
                if unfinished[b]:
                    generated[b].append(next_token[b, 0].item())

            eos_hit = next_token.squeeze() == eos_token_id
            unfinished = unfinished & ~eos_hit
            if not unfinished.any():
                break

            next_token = next_token.masked_fill(~unfinished.unsqueeze(1), pad_token_id)

            attention_mask = torch.cat(
                [attention_mask, torch.ones(B, 1, dtype=torch.long, device=device)],
                dim=1,
            )

            out = model(
                input_ids=next_token, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=True
            )
            past_key_values = out.past_key_values
            next_logits = out.logits[:, -1]

        max_len = max(len(seq) for seq in generated)
        out_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long, device=device)
        for b, seq in enumerate(generated):
            if seq:
                out_ids[b, : len(seq)] = torch.tensor(seq, device=device)

        return out_ids

    @torch.inference_mode()
    def generate_from_embeds(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 32,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        temperature: float = 1.0,
        do_sample: bool = False,
        num_return_sequences: int = 1,
        labels_trie: Optional[LabelsTrie] = None,
        **kwargs: Any,
    ) -> torch.LongTensor:
        """Generation from embeddings using Hugging Face's generate API.

        This method wraps the Hugging Face generate() function to support generation
        from embeddings with optional trie-based prefix constraints. It provides a
        more feature-complete interface than generate_from_embeds_custom but may be
        less flexible for custom generation logic.

        Args:
            inputs_embeds: Input embeddings of shape (batch_size, prefix_len, hidden_size)
                serving as the generation prefix.
            attention_mask: Optional attention mask of shape (batch_size, prefix_len).
                If None, creates a mask of all ones. Defaults to None.
            max_new_tokens: Maximum number of new tokens to generate. Defaults to 32.
            eos_token_id: Token ID marking end of sequence. If None, uses model's
                default. Defaults to None.
            pad_token_id: Token ID for padding. If None, uses model's default or
                eos_token_id. Defaults to None.
            temperature: Sampling temperature for controlling randomness. Defaults to 1.0.
            do_sample: If True, uses sampling. If False, uses greedy/beam search.
                Defaults to False.
            num_return_sequences: Number of sequences to generate per input. Also
                sets num_beams when > 1. Defaults to 1.
            labels_trie: Optional trie structure for constrained decoding via
                prefix_allowed_tokens_fn. Defaults to None.
            **kwargs: Additional keyword arguments passed to model.generate().

        Returns:
            Generated token IDs of shape (batch_size * num_return_sequences, total_len)
            where total_len = prefix_len + generated_len. Includes both the input
            prefix and newly generated tokens.
        """
        model = self.decoder_layer.model
        inputs_embeds = inputs_embeds.to(dtype=model.dtype)
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=model.dtype)
        device, (B, L0, _) = inputs_embeds.device, inputs_embeds.shape
        cfg = model.config

        # Set token IDs if not provided
        eos_token_id = eos_token_id or cfg.eos_token_id
        pad_token_id = pad_token_id or cfg.pad_token_id or eos_token_id

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(B, L0, dtype=torch.long, device=device)

        # Define prefix-constrained token function if trie is provided
        if labels_trie is not None:

            def prefix_allowed_tokens(batch_idx: int, input_ids: torch.Tensor) -> List[int]:
                """Callback function for constrained decoding.

                Args:
                    batch_idx: Index of the sequence in the batch.
                    input_ids: Currently generated token IDs.

                Returns:
                    List of allowed token IDs for the next position.
                """
                current_seq = input_ids.tolist()
                allowed_tokens = labels_trie.get(current_seq)
                if not allowed_tokens:  # Empty or None
                    allowed_tokens = [eos_token_id]
                return allowed_tokens
        else:
            prefix_allowed_tokens = None

        # Generate new tokens using transformer's generate method
        generated_ids = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            temperature=temperature,
            do_sample=do_sample,
            use_cache=True,
            num_return_sequences=num_return_sequences,
            num_beams=num_return_sequences,
            prefix_allowed_tokens_fn=prefix_allowed_tokens,
            logits_processor=LogitsProcessorList(
                [
                    NumericalStabilityProcessor(),
                ]
            ),
            **kwargs,
        )

        return generated_ids

    def generate(self, *args: Any, **kwargs: Any) -> torch.LongTensor:
        """Flexible generation method supporting both embeddings and token IDs.

        This method routes to the appropriate generation function based on whether
        inputs_embeds is provided. If inputs_embeds is in kwargs, uses
        generate_from_embeds(). Otherwise, delegates to the model's native
        generate() method.

        Args:
            *args: Variable positional arguments passed to the generation method.
            **kwargs: Variable keyword arguments. If 'inputs_embeds' is present,
                routes to generate_from_embeds(), otherwise routes to model.generate().

        Returns:
            Generated token IDs. Shape depends on the specific generation method used.
        """
        if "inputs_embeds" in kwargs:
            return self.generate_from_embeds(*args, **kwargs)
        else:
            return self.decoder_layer.model.generate(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass through the decoder.

        Computes logits for the input sequence without generation.

        Args:
            *args: Variable positional arguments passed to the decoder layer.
            **kwargs: Variable keyword arguments passed to the decoder layer.

        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size).
        """
        decoded_embeddings = self.decoder_layer(*args, **kwargs)
        return decoded_embeddings
