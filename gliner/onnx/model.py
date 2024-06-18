from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import warnings
import onnxruntime as ort
import numpy as np
import torch

from ..modeling.base import GLiNERModelOutput

class BaseORTModel(ABC):
    def __init__(self, session: ort.InferenceSession):
        self.session = session
        self.input_names = {input_key.name: idx for idx, input_key in enumerate(self.session.get_inputs())}
        self.output_names = {output_key.name: idx for idx, output_key in enumerate(self.session.get_outputs())}

    def prepare_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        """
        Prepare inputs for ONNX model inference.
        
        Args:
            inputs (Dict[str, torch.Tensor]): Dictionary of input names and tensors.
        
        Returns:
            Dict[str, np.ndarray]: Dictionary of input names and numpy arrays.
        """
        if not isinstance(inputs, dict):
            raise ValueError("Inputs must be a dictionary of input names and tensors.")
        
        prepared_inputs = {}
        for key, tensor in inputs.items():
            if key not in self.input_names:
                warnings.warn(f"Input key '{key}' not found in ONNX model's input names. Ignored.")
                continue
            prepared_inputs[key] = tensor.cpu().detach().numpy()
        return prepared_inputs

    def run_inference(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Run the ONNX model inference.
        
        Args:
            inputs (Dict[str, np.ndarray]): Prepared inputs for the model.
        
        Returns:
            Dict[str, np.ndarray]: Model's outputs as numpy arrays.
        """
        onnx_outputs = self.session.run(None, inputs)
        outputs = {name: onnx_outputs[idx] for name, idx in self.output_names.items()}
        return outputs

    @abstractmethod
    def forward(self, input_ids, attention_mask, **kwargs) -> Dict[str, Any]:
        """
        Abstract method to perform forward pass. Must be implemented by subclasses.
        """
        pass
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
class SpanORTModel(BaseORTModel):
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                words_mask: torch.Tensor, text_lengths: torch.Tensor, 
                span_idx: torch.Tensor, span_mask: torch.Tensor, **kwargs) -> Dict[str, Any]:
        """
        Forward pass for span model using ONNX inference.

        Args:
            input_ids (torch.Tensor): Input IDs tensor.
            attention_mask (torch.Tensor): Attention mask tensor.
            span_idx (torch.Tensor): Span indices tensor.
            span_mask (torch.Tensor): Span mask tensor.
            **kwargs: Additional arguments.
        
        Returns:
            Dict[str, Any]: Model outputs.
        """
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'words_mask': words_mask,
            'text_lengths': text_lengths,
            'span_idx': span_idx,
            'span_mask': span_mask
        }
        prepared_inputs = self.prepare_inputs(inputs)
        inference_output = self.run_inference(prepared_inputs)
        outputs = GLiNERModelOutput(
            logits=inference_output['logits']
        )
        return outputs

class TokenORTModel(BaseORTModel):
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                words_mask: torch.Tensor, text_lengths: torch.Tensor, 
                **kwargs) -> Dict[str, Any]:
        """
        Forward pass for token model using ONNX inference.

        Args:
            input_ids (torch.Tensor): Input IDs tensor.
            attention_mask (torch.Tensor): Attention mask tensor.
            **kwargs: Additional arguments.
        
        Returns:
            Dict[str, Any]: Model outputs.
        """
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'words_mask': words_mask,
            'text_lengths': text_lengths,
        }
        prepared_inputs = self.prepare_inputs(inputs)
        inference_output = self.run_inference(prepared_inputs)
        outputs = GLiNERModelOutput(
            logits=inference_output['logits']
        )
        return outputs