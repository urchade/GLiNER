"""ONNX Runtime inference models for GLiNER.

This module provides ONNX Runtime implementations of various GLiNER model
architectures, including uni-encoder and bi-encoder variants for both
span-level and token-level named entity recognition, as well as relation
extraction models.
"""

import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import torch
import onnxruntime as ort

from ..modeling.outputs import GLiNERBaseOutput, GLiNERRelexOutput


class BaseORTModel(ABC):
    """Base class for ONNX Runtime inference models.

    Provides common functionality for preparing inputs, running inference,
    and managing ONNX session I/O. All concrete ORT model implementations
    should inherit from this class.

    Attributes:
        session: ONNX Runtime inference session.
        input_names: Dictionary mapping input names to their indices.
        output_names: Dictionary mapping output names to their indices.
    """

    def __init__(self, session: ort.InferenceSession):
        """Initialize the ONNX Runtime model.

        Args:
            session: ONNX Runtime inference session.
        """
        self.session = session
        self.input_names = {input_key.name: idx for idx, input_key in enumerate(self.session.get_inputs())}
        self.output_names = {output_key.name: idx for idx, output_key in enumerate(self.session.get_outputs())}

    def prepare_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        """Prepare inputs for ONNX model inference.

        Converts PyTorch tensors to numpy arrays and filters out inputs
        that are not expected by the ONNX model.

        Args:
            inputs: Dictionary of input names and PyTorch tensors.

        Returns:
            Dictionary of input names and numpy arrays ready for ONNX inference.

        Raises:
            ValueError: If inputs is not a dictionary.
        """
        if not isinstance(inputs, dict):
            raise ValueError("Inputs must be a dictionary of input names and tensors.")

        prepared_inputs = {}
        for key, tensor in inputs.items():
            if key not in self.input_names:
                warnings.warn(f"Input key '{key}' not found in ONNX model's input names. Ignored.", stacklevel=2)
                continue
            prepared_inputs[key] = tensor.cpu().detach().numpy()
        return prepared_inputs

    def run_inference(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run the ONNX model inference.

        Args:
            inputs: Prepared inputs for the model as numpy arrays.

        Returns:
            Dictionary mapping output names to their corresponding numpy arrays.
        """
        onnx_outputs = self.session.run(None, inputs)
        outputs = {name: onnx_outputs[idx] for name, idx in self.output_names.items()}
        return outputs

    @abstractmethod
    def forward(self, input_ids, attention_mask, **kwargs) -> Dict[str, Any]:
        """Perform forward pass through the model.

        Abstract method that must be implemented by subclasses to define
        model-specific forward pass logic.

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask for input tokens.
            **kwargs: Additional model-specific arguments.

        Returns:
            Dictionary containing model outputs.
        """
        pass

    def __call__(self, *args, **kwargs):
        """Make the model callable.

        Delegates to the forward method.

        Args:
            *args: Positional arguments to pass to forward.
            **kwargs: Keyword arguments to pass to forward.

        Returns:
            Output from the forward method.
        """
        return self.forward(*args, **kwargs)


class UniEncoderSpanORTModel(BaseORTModel):
    """ONNX Runtime model for uni-encoder span-level NER.

    Uses a single encoder to process both text and entity labels,
    performing span-level entity recognition.
    """

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        words_mask: torch.Tensor,
        text_lengths: torch.Tensor,
        span_idx: torch.Tensor,
        span_mask: torch.Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        """Forward pass for span model using ONNX inference.

        Args:
            input_ids: Tensor of shape (batch_size, seq_len) containing input token IDs.
            attention_mask: Tensor of shape (batch_size, seq_len) with 1s for real
                tokens and 0s for padding.
            words_mask: Tensor of shape (batch_size, seq_len) indicating word boundaries.
            text_lengths: Tensor of shape (batch_size,) containing the actual length
                of each text sequence.
            span_idx: Tensor containing indices of spans to classify.
            span_mask: Tensor indicating which spans are valid (not padding).
            **kwargs: Additional arguments (ignored).

        Returns:
            GLiNERBaseOutput containing logits for span classification.
        """
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "words_mask": words_mask,
            "text_lengths": text_lengths,
            "span_idx": span_idx,
            "span_mask": span_mask,
        }
        prepared_inputs = self.prepare_inputs(inputs)
        inference_output = self.run_inference(prepared_inputs)
        outputs = GLiNERBaseOutput(logits=inference_output["logits"])
        return outputs


class BiEncoderSpanORTModel(BaseORTModel):
    """ONNX Runtime model for bi-encoder span-level NER.

    Uses separate encoders for text and entity labels, performing
    span-level entity recognition with bi-encoder architecture.
    """

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        words_mask: torch.Tensor,
        text_lengths: torch.Tensor,
        span_idx: torch.Tensor,
        span_mask: torch.Tensor,
        labels_embeds: Optional[torch.Tensor] = None,
        labels_input_ids: Optional[torch.FloatTensor] = None,
        labels_attention_mask: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Forward pass for bi-encoder span model using ONNX inference.

        Args:
            input_ids: Tensor of shape (batch_size, seq_len) containing input token IDs.
            attention_mask: Tensor of shape (batch_size, seq_len) with 1s for real
                tokens and 0s for padding.
            words_mask: Tensor of shape (batch_size, seq_len) indicating word boundaries.
            text_lengths: Tensor of shape (batch_size,) containing the actual length
                of each text sequence.
            span_idx: Tensor containing indices of spans to classify.
            span_mask: Tensor indicating which spans are valid (not padding).
            labels_embeds: Optional pre-computed embeddings for entity labels.
                If provided, labels_input_ids and labels_attention_mask are ignored.
            labels_input_ids: Optional tensor containing token IDs for entity labels.
                Used when labels_embeds is not provided.
            labels_attention_mask: Optional attention mask for entity label tokens.
                Used when labels_embeds is not provided.
            **kwargs: Additional arguments (ignored).

        Returns:
            GLiNERBaseOutput containing logits for span classification.
        """
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "words_mask": words_mask,
            "text_lengths": text_lengths,
            "span_idx": span_idx,
            "span_mask": span_mask,
        }
        if labels_embeds is not None:
            inputs["labels_embeds"] = labels_embeds
        else:
            inputs["labels_input_ids"] = labels_input_ids
            inputs["labels_attention_mask"] = labels_attention_mask

        prepared_inputs = self.prepare_inputs(inputs)
        inference_output = self.run_inference(prepared_inputs)
        outputs = GLiNERBaseOutput(logits=inference_output["logits"])
        return outputs


class UniEncoderTokenORTModel(BaseORTModel):
    """ONNX Runtime model for uni-encoder token-level NER.

    Uses a single encoder to process both text and entity labels,
    performing token-level entity recognition.
    """

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        words_mask: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        """Forward pass for token model using ONNX inference.

        Args:
            input_ids: Tensor of shape (batch_size, seq_len) containing input token IDs.
            attention_mask: Tensor of shape (batch_size, seq_len) with 1s for real
                tokens and 0s for padding.
            words_mask: Tensor of shape (batch_size, seq_len) indicating word boundaries.
            text_lengths: Tensor of shape (batch_size,) containing the actual length
                of each text sequence.
            **kwargs: Additional arguments (ignored).

        Returns:
            GLiNERBaseOutput containing logits for token classification.
        """
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "words_mask": words_mask,
            "text_lengths": text_lengths,
        }
        prepared_inputs = self.prepare_inputs(inputs)
        inference_output = self.run_inference(prepared_inputs)
        outputs = GLiNERBaseOutput(logits=inference_output["logits"])
        return outputs


class BiEncoderTokenORTModel(BaseORTModel):
    """ONNX Runtime model for bi-encoder token-level NER.

    Uses separate encoders for text and entity labels, performing
    token-level entity recognition with bi-encoder architecture.
    """

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        words_mask: torch.Tensor,
        text_lengths: torch.Tensor,
        labels_embeds: Optional[torch.Tensor] = None,
        labels_input_ids: Optional[torch.FloatTensor] = None,
        labels_attention_mask: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Forward pass for bi-encoder token model using ONNX inference.

        Args:
            input_ids: Tensor of shape (batch_size, seq_len) containing input token IDs.
            attention_mask: Tensor of shape (batch_size, seq_len) with 1s for real
                tokens and 0s for padding.
            words_mask: Tensor of shape (batch_size, seq_len) indicating word boundaries.
            text_lengths: Tensor of shape (batch_size,) containing the actual length
                of each text sequence.
            labels_embeds: Optional pre-computed embeddings for entity labels.
                If provided, labels_input_ids and labels_attention_mask are ignored.
            labels_input_ids: Optional tensor containing token IDs for entity labels.
                Used when labels_embeds is not provided.
            labels_attention_mask: Optional attention mask for entity label tokens.
                Used when labels_embeds is not provided.
            **kwargs: Additional arguments (ignored).

        Returns:
            GLiNERBaseOutput containing logits for token classification.
        """
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "words_mask": words_mask,
            "text_lengths": text_lengths,
        }

        if labels_embeds is not None:
            inputs["labels_embeds"] = labels_embeds
        else:
            inputs["labels_input_ids"] = labels_input_ids
            inputs["labels_attention_mask"] = labels_attention_mask

        prepared_inputs = self.prepare_inputs(inputs)
        inference_output = self.run_inference(prepared_inputs)
        outputs = GLiNERBaseOutput(logits=inference_output["logits"])
        return outputs


class UniEncoderSpanRelexORTModel(BaseORTModel):
    """ONNX Runtime model for uni-encoder span-level relation extraction.

    Uses a single encoder to process text and perform both entity recognition
    and relation extraction at the span level.
    """

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        words_mask: torch.Tensor,
        text_lengths: torch.Tensor,
        span_idx: torch.Tensor,
        span_mask: torch.Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        """Forward pass for span relation extraction model using ONNX inference.

        Args:
            input_ids: Tensor of shape (batch_size, seq_len) containing input token IDs.
            attention_mask: Tensor of shape (batch_size, seq_len) with 1s for real
                tokens and 0s for padding.
            words_mask: Tensor of shape (batch_size, seq_len) indicating word boundaries.
            text_lengths: Tensor of shape (batch_size,) containing the actual length
                of each text sequence.
            span_idx: Tensor containing indices of spans to classify.
            span_mask: Tensor indicating which spans are valid (not padding).
            **kwargs: Additional arguments (ignored).

        Returns:
            GLiNERRelexOutput containing logits for span classification, relation
            indices, relation logits, and relation mask.
        """
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "words_mask": words_mask,
            "text_lengths": text_lengths,
            "span_idx": span_idx,
            "span_mask": span_mask,
        }
        prepared_inputs = self.prepare_inputs(inputs)
        inference_output = self.run_inference(prepared_inputs)
        outputs = GLiNERRelexOutput(
            logits=inference_output["logits"],
            rel_idx=inference_output["rel_idx"],
            rel_logits=inference_output["rel_logits"],
            rel_mask=inference_output["rel_mask"],
        )
        return outputs
