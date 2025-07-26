import os
import argparse
import numpy as np

from gliner import GLiNER

import torch
from onnxruntime.quantization import quantize_dynamic, QuantType

class GLiNERWrapper(torch.nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core.model                      

    def forward(self, input_ids, attention_mask,
                words_mask, text_lengths, span_idx = None, span_mask = None):
        return self.core(
            input_ids=input_ids,
            attention_mask=attention_mask,
            words_mask=words_mask,
            text_lengths=text_lengths,
            span_idx=span_idx, 
            span_mask = span_mask
        ).logits                              
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default= "logs/model_12000")
    parser.add_argument('--save_path', type=str, default = 'model/')
    parser.add_argument('--quantize', type=bool, default = True)
    args = parser.parse_args()
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    onnx_save_path = os.path.join(args.save_path, "model.onnx")

    print("Loading a model...")
    gliner_model = GLiNER.from_pretrained(args.model_path, load_tokenizer=True)

    text = "ONNX is an open-source format designed to enable the interoperability of AI models across various frameworks and tools."
    labels = ['format', 'model', 'tool', 'cat']

    inputs, _ = gliner_model.prepare_model_inputs([text], labels)

    if gliner_model.config.span_mode == 'token_level':
        all_inputs =  (inputs['input_ids'], inputs['attention_mask'], 
                        inputs['words_mask'], inputs['text_lengths'])
        input_names = ['input_ids', 'attention_mask', 'words_mask', 'text_lengths']
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "words_mask": {0: "batch_size", 1: "sequence_length"},
            "text_lengths": {0: "batch_size", 1: "value"},
            "logits": {0: "position", 1: "batch_size", 2: "sequence_length", 3: "num_classes"},
        }
    else:
        all_inputs =  (inputs['input_ids'], inputs['attention_mask'], 
                        inputs['words_mask'], inputs['text_lengths'],
                        inputs['span_idx'], inputs['span_mask'])
        input_names = ['input_ids', 'attention_mask', 'words_mask', 'text_lengths', 'span_idx', 'span_mask']
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "words_mask": {0: "batch_size", 1: "sequence_length"},
            "text_lengths": {0: "batch_size", 1: "value"},
            "span_idx": {0: "batch_size", 1: "num_spans", 2: "idx"},
            "span_mask": {0: "batch_size", 1: "num_spans"},
            "logits": {0: "batch_size", 1: "sequence_length", 2: "num_spans", 3: "num_classes"},
        }
    print('Converting the model...')
    model_wrapper = GLiNERWrapper(gliner_model)
    torch.onnx.export(
        model_wrapper,
        all_inputs,
        f=onnx_save_path,
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=19,
    )

    if args.quantize:
        quantized_save_path = os.path.join(args.save_path, "model_quantized.onnx")
        # Quantize the ONNX model
        print("Quantizing the model...")
        quantize_dynamic(
            onnx_save_path,  # Input model
            quantized_save_path,  # Output model
            weight_type=QuantType.QUInt8  # Quantize weights to 8-bit integers
        )
    print("Done!")