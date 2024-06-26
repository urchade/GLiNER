import os
import re
import json
import warnings
from pathlib import Path
from typing import Optional, Union, Dict, List

from transformers import AutoTokenizer, AutoConfig

import torch
from torch import nn
import numpy as np

from .modeling.base import BaseModel, SpanModel, TokenModel
from .onnx.model import BaseORTModel, SpanORTModel, TokenORTModel
from .data_processing import SpanProcessor, TokenProcessor, GLiNERDataset
from .data_processing.tokenizer import WordsSplitter
from .data_processing.collator import DataCollatorWithPadding, DataCollator
from .decoding import SpanDecoder, TokenDecoder
from .evaluation import Evaluator
from .config import GLiNERConfig

from huggingface_hub import PyTorchModelHubMixin, snapshot_download

class GLiNER(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: GLiNERConfig, 
                        model: Optional[Union[BaseModel, BaseORTModel]] = None,
                        tokenizer: Optional[Union[str, AutoTokenizer]] = None, 
                        words_splitter: Optional[Union[str, WordsSplitter]] = None, 
                        data_processor: Optional[Union[SpanProcessor, TokenProcessor]] = None, 
                        encoder_from_pretrained: bool = True):
        """
        Initialize the GLiNER model.

        Args:
            config (GLiNERConfig): Configuration object for the GLiNER model.
            model (Optional[Union[BaseModel, BaseORTModel]]): GLiNER model to use for predictions. Defaults to None.
            tokenizer (Optional[Union[str, AutoTokenizer]]): Tokenizer to use. Can be a string (path or name) or an AutoTokenizer instance. Defaults to None.
            words_splitter (Optional[Union[str, WordsSplitter]]): Words splitter to use. Can be a string or a WordsSplitter instance. Defaults to None.
            data_processor (Optional[Union[SpanProcessor, TokenProcessor]]): Data processor - object that prepare input to a model. Defaults to None.
            encoder_from_pretrained (bool): Whether to load the encoder from a pre-trained model or init from scratch. Defaults to True.
        """
        super().__init__()
        self.config = config

        if tokenizer is None and data_processor is None:
            tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        if words_splitter is None and data_processor is None:
            words_splitter = WordsSplitter(config.words_splitter_type)

        if config.span_mode == "token_level":
            if model is None:
                self.model = TokenModel(config, encoder_from_pretrained)
            else:
                self.model = model
            if data_processor is None:
                self.data_processor = TokenProcessor(config, tokenizer, words_splitter)
            else:
                self.data_processor = data_processor
            self.decoder = TokenDecoder(config)
        else:
            if model is None:
                self.model = SpanModel(config, encoder_from_pretrained)
            else:
                self.model = model
            if data_processor is None:
                self.data_processor = SpanProcessor(config, tokenizer, words_splitter)
            else:
                self.data_processor = data_processor
            self.decoder = SpanDecoder(config)

        if config.vocab_size !=-1 and config.vocab_size!=len(self.data_processor.transformer_tokenizer):
            warnings.warn(f"""Vocab size of the model ({config.vocab_size}) does't match length of tokenizer ({len(self.data_processor.transformer_tokenizer)}). 
                            You should to consider manually add new tokens to tokenizer or to load tokenizer with added tokens.""")
            
        if isinstance(self.model, BaseORTModel):
            self.onnx_model = True
        else:
            self.onnx_model = False

    def forward(self, *args, **kwargs):
        """Wrapper function for the model's forward pass."""
        output = self.model(*args, **kwargs)
        return output

    @property
    def device(self):
        device = next(self.model.parameters()).device
        return device
    
    def resize_token_embeddings(self, add_tokens, 
                                    set_class_token_index = True, 
                                    add_tokens_to_tokenizer = True, 
                                    pad_to_multiple_of=None) -> nn.Embedding:
        """
        Resize the token embeddings of the model.

        Args:
            add_tokens: The tokens to add to the embedding layer.
            set_class_token_index (bool, optional): Whether to set the class token index. Defaults to True.
            add_tokens_to_tokenizer (bool, optional): Whether to add the tokens to the tokenizer. Defaults to True.
            pad_to_multiple_of (int, optional): If set, pads the embedding size to be a multiple of this value. Defaults to None.

        Returns:
            nn.Embedding: The resized embedding layer.
        """
        if set_class_token_index:
            self.config.class_token_index = len(self.data_processor.transformer_tokenizer)+1
        if add_tokens_to_tokenizer:
            self.data_processor.transformer_tokenizer.add_tokens(add_tokens)
        new_num_tokens = len(self.data_processor.transformer_tokenizer)
        model_embeds = self.model.token_rep_layer.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.vocab_size = model_embeds.num_embeddings
        if self.config.encoder_config is not None:
            self.config.encoder_config.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def prepare_model_inputs(self, texts: str, labels: str):
        """
        Prepare inputs for the model.

        Args:
            texts (str): The input text or texts to process.
            labels (str): The corresponding labels for the input texts.
        """
        all_tokens = []
        all_start_token_idx_to_text_idx = []
        all_end_token_idx_to_text_idx = []

        for text in texts:
            tokens = []
            start_token_idx_to_text_idx = []
            end_token_idx_to_text_idx = []
            for token, start, end in self.data_processor.words_splitter(text):
                tokens.append(token)
                start_token_idx_to_text_idx.append(start)
                end_token_idx_to_text_idx.append(end)
            all_tokens.append(tokens)
            all_start_token_idx_to_text_idx.append(start_token_idx_to_text_idx)
            all_end_token_idx_to_text_idx.append(end_token_idx_to_text_idx)

        input_x = [{"tokenized_text": tk, "ner": None} for tk in all_tokens]
        raw_batch = self.data_processor.collate_raw_batch(input_x, labels)
        raw_batch["all_start_token_idx_to_text_idx"] = all_start_token_idx_to_text_idx
        raw_batch["all_end_token_idx_to_text_idx"] = all_end_token_idx_to_text_idx

        model_input = self.data_processor.collate_fn(raw_batch, prepare_labels=False)
        model_input.update({"span_idx": raw_batch['span_idx'] if 'span_idx' in raw_batch else None, 
                            "span_mask": raw_batch["span_mask"] if 'span_mask' in raw_batch else None,
                            "text_lengths": raw_batch['seq_length']})
        
        if not self.onnx_model:
            device = self.device
            for key in model_input:
                if model_input[key] is not None and isinstance(model_input[key], torch.Tensor):
                    model_input[key] = model_input[key].to(device)

        return model_input, raw_batch
    
    def predict_entities(self, text, labels, flat_ner=True, threshold=0.5, multi_label=False):
        """
        Predict entities for a single text input.

        Args:
            text: The input text to predict entities for.
            labels: The labels to predict.
            flat_ner (bool, optional): Whether to use flat NER. Defaults to True.
            threshold (float, optional): Confidence threshold for predictions. Defaults to 0.5.
            multi_label (bool, optional): Whether to allow multiple labels per entity. Defaults to False.

        Returns:
            The list of entity predictions.
        """
        return self.batch_predict_entities(
            [text], labels, flat_ner=flat_ner, threshold=threshold, multi_label=multi_label
        )[0]

    @torch.no_grad()
    def batch_predict_entities(self, texts, labels, flat_ner=True, threshold=0.5, multi_label=False):
        """
        Predict entities for a batch of texts.

        Args:
            texts (List[str]): A list of input texts to predict entities for.
            labels (List[str]): A list of labels to predict.
            flat_ner (bool, optional): Whether to use flat NER. Defaults to True.
            threshold (float, optional): Confidence threshold for predictions. Defaults to 0.5.
            multi_label (bool, optional): Whether to allow multiple labels per token. Defaults to False.

        Returns:
            The list of lists with predicted entities.
        """

        model_input, raw_batch = self.prepare_model_inputs(texts, labels)

        model_output = self.model(**model_input)[0]

        if not isinstance(model_output, torch.Tensor):
            model_output = torch.from_numpy(model_output)

        outputs = self.decoder.decode(raw_batch['tokens'], raw_batch['id_to_classes'], 
                    model_output, flat_ner=flat_ner, threshold=threshold, multi_label=multi_label)

        all_entities = []
        for i, output in enumerate(outputs):
            start_token_idx_to_text_idx = raw_batch['all_start_token_idx_to_text_idx'][i]
            end_token_idx_to_text_idx = raw_batch['all_end_token_idx_to_text_idx'][i]
            entities = []
            for start_token_idx, end_token_idx, ent_type, ent_score in output:
                start_text_idx = start_token_idx_to_text_idx[start_token_idx]
                end_text_idx = end_token_idx_to_text_idx[end_token_idx]
                entities.append({
                    "start": start_token_idx_to_text_idx[start_token_idx],
                    "end": end_token_idx_to_text_idx[end_token_idx],
                    "text": texts[i][start_text_idx:end_text_idx],
                    "label": ent_type,
                    "score": ent_score
                })
            all_entities.append(entities)

        return all_entities

    def evaluate(self, test_data, flat_ner=False, multi_label=False, threshold=0.5, batch_size=12, entity_types=None):
        """
        Evaluate the model on a given test dataset.

        Args:
            test_data (List[Dict]): The test data containing text and entity annotations.
            flat_ner (bool): Whether to use flat NER. Defaults to False.
            multi_label (bool): Whether to use multi-label classification. Defaults to False.
            threshold (float): The threshold for predictions. Defaults to 0.5.
            batch_size (int): The batch size for evaluation. Defaults to 12.
            entity_types (Optional[List[str]]): List of entity types to consider. Defaults to None.

        Returns:
            tuple: A tuple containing the evaluation output and the F1 score.
        """
        self.eval()
        # Create the dataset and data loader
        # dataset = GLiNERDataset(test_data, config = self.config, data_processor=self.data_processor,
        #                                             return_tokens = True, return_id_to_classes = True,
        #                                             prepare_labels= False, return_entities = True,
        #                                             entities=entity_types, get_negatives=False)
        # collator = DataCollatorWithPadding(self.config)
        dataset = test_data
        collator = DataCollator(self.config, data_processor=self.data_processor,
                                return_tokens=True,
                                return_entities=True,
                                return_id_to_classes=True,
                                prepare_labels=False,
                                entity_types = entity_types)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

        device = self.device
        all_preds = []
        all_trues = []

        # Iterate over data batches
        for batch in data_loader:
            # Move the batch to the appropriate device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

                # Perform predictions
            model_output = self.model(**batch)[0]

            if not isinstance(model_output, torch.Tensor):
                model_output = torch.from_numpy(model_output)

            decoded_outputs = self.decoder.decode(
                batch['tokens'], batch['id_to_classes'],
                model_output, flat_ner=flat_ner, threshold=threshold, multi_label=multi_label
            )
            all_preds.extend(decoded_outputs)
            all_trues.extend(batch["entities"])

        # Evaluate the predictions
        evaluator = Evaluator(all_trues, all_preds)
        out, f1 = evaluator.evaluate()

        return out, f1

    def predict(self, batch, flat_ner=False, threshold=0.5, multi_label=False):
        """
        Predict the entities for a given batch of data.

        Args:
            batch (Dict): A batch of data containing tokenized text and other relevant fields.
            flat_ner (bool): Whether to use flat NER. Defaults to False.
            threshold (float): The threshold for predictions. Defaults to 0.5.
            multi_label (bool): Whether to use multi-label classification. Defaults to False.

        Returns:
            List: Predicted entities for each example in the batch.
        """
        model_output = self.model(**batch)[0]

        if not isinstance(model_output, torch.Tensor):
            model_output = torch.from_numpy(model_output)

        decoded_outputs = self.decoder.decode(
            batch['tokens'], batch['id_to_classes'],
            model_output, flat_ner=flat_ner, threshold=threshold, multi_label=multi_label
        )

        return decoded_outputs

    def compile(self):
        self.model = torch.compile(self.model)

    def compile_for_training(self):
        print('Compiling transformer encoder...')
        self.model.token_rep_layer = torch.compile(self.model.token_rep_layer)
        print('Compiling RNN...')
        self.model.rnn = torch.compile(self.model.rnn)
        if hasattr(self.model, "span_rep_layer"):
            print('Compiling span representation layer...')
            self.model.span_rep_layer = torch.compile(self.model.span_rep_layer)
        if hasattr(self.model, "prompt_rep_layer"):
            print('Compiling prompt representation layer...')
            self.model.prompt_rep_layer = torch.compile(self.model.prompt_rep_layer)
        if hasattr(self.model, "scorer"):
            print('Compiling scorer...')
            self.model.scorer = torch.compile(self.model.scorer)

    def set_sampling_params(self, max_types, shuffle_types, random_drop, max_neg_type_ratio, max_len):
        """
        Sets sampling parameters on the given model.

        Parameters:
        - model: The model object to update.
        - max_types: Maximum types parameter.
        - shuffle_types: Boolean indicating whether to shuffle types.
        - random_drop: Boolean indicating whether to randomly drop elements.
        - max_neg_type_ratio: Maximum negative type ratio.
        - max_len: Maximum length parameter.
        """
        self.config.max_types = max_types
        self.config.shuffle_types = shuffle_types
        self.config.random_drop = random_drop
        self.config.max_neg_type_ratio = max_neg_type_ratio
        self.config.max_len = max_len

    def prepare_state_dict(self, state_dict):
        """
        Prepare state dict in the case of torch.compile
        """
        new_state_dict = {}
        for key, tensor in state_dict.items():
            key = re.sub("_orig_mod\.", "", key)
            new_state_dict[key] = tensor
        return new_state_dict
    
    def save_pretrained(
            self,
            save_directory: Union[str, Path],
            *,
            config: Optional[GLiNERConfig] = None,
            repo_id: Optional[str] = None,
            push_to_hub: bool = False,
            **push_to_hub_kwargs,
    ) -> Optional[str]:
        """
        Save weights in local directory.

        Args:
            save_directory (`str` or `Path`):
                Path to directory in which the model weights and configuration will be saved.
            config (`dict` or `DataclassInstance`, *optional*):
                Model configuration specified as a key/value dictionary or a dataclass instance.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Huggingface Hub after saving it.
            repo_id (`str`, *optional*):
                ID of your repository on the Hub. Used only if `push_to_hub=True`. Will default to the folder name if
                not provided.
            kwargs:
                Additional key word arguments passed along to the [`~ModelHubMixin.push_to_hub`] method.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # save model weights/files
        torch.save(self.prepare_state_dict(self.model.state_dict()), save_directory / "pytorch_model.bin")

        # save config (if provided)
        if config is None:
            config = self.config
        if config is not None:
            config.to_json_file(save_directory / "gliner_config.json")

        self.data_processor.transformer_tokenizer.save_pretrained(save_directory)
        # push to the Hub if required
        if push_to_hub:
            kwargs = push_to_hub_kwargs.copy()  # soft-copy to avoid mutating input
            if config is not None:  # kwarg for `push_to_hub`
                kwargs["config"] = config
            if repo_id is None:
                repo_id = save_directory.name  # Defaults to `save_directory` name
            return self.push_to_hub(repo_id=repo_id, **kwargs)
        return None

    @classmethod
    def _from_pretrained(
            cls,
            *,
            model_id: str,
            revision: Optional[str],
            cache_dir: Optional[Union[str, Path]],
            force_download: bool,
            proxies: Optional[Dict],
            resume_download: bool,
            local_files_only: bool,
            token: Union[str, bool, None],
            map_location: str = "cpu",
            strict: bool = False,
            load_tokenizer: Optional[bool]=False,
            resize_token_embeddings: Optional[bool]=True,
            load_onnx_model: Optional[bool]=False,
            onnx_model_file: Optional[str] = 'model.onnx',
            compile_torch_model: Optional[bool] = False,
            **model_kwargs,
    ):
        """
        Load a pretrained model from a given model ID.

        Args:
            model_id (str): Identifier of the model to load.
            revision (Optional[str]): Specific model revision to use.
            cache_dir (Optional[Union[str, Path]]): Directory to store downloaded models.
            force_download (bool): Force re-download even if the model exists.
            proxies (Optional[Dict]): Proxy configuration for downloads.
            resume_download (bool): Resume interrupted downloads.
            local_files_only (bool): Use only local files, don't download.
            token (Union[str, bool, None]): Token for API authentication.
            map_location (str): Device to map model to. Defaults to "cpu".
            strict (bool): Enforce strict state_dict loading.
            load_tokenizer (Optional[bool]): Whether to load the tokenizer. Defaults to False.
            resize_token_embeddings (Optional[bool]): Resize token embeddings. Defaults to True.
            load_onnx_model (Optional[bool]): Load ONNX version of the model. Defaults to False.
            onnx_model_file (Optional[str]): Filename for ONNX model. Defaults to 'model.onnx'.
            compile_torch_model (Optional[bool]): Compile the PyTorch model. Defaults to False.
            **model_kwargs: Additional keyword arguments for model initialization.

        Returns:
            An instance of the model loaded from the pretrained weights.
        """
        # Newer format: Use "pytorch_model.bin" and "gliner_config.json"
        model_dir = Path(model_id)# / "pytorch_model.bin"
        if not model_dir.exists():
            model_dir = snapshot_download(
                repo_id=model_id,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )
        model_file = Path(model_dir) / "pytorch_model.bin"
        config_file = Path(model_dir) / "gliner_config.json"

        if load_tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
        else:
            if os.path.exists(os.path.join(model_dir, 'tokenizer_config.json')):
                tokenizer = AutoTokenizer.from_pretrained(model_dir)
            else:
                tokenizer = None
        config_ = json.load(open(config_file))
        config = GLiNERConfig(**config_)
        
        add_tokens = ['[FLERT]', config.ent_token, config.sep_token]

        if not load_onnx_model:
            gliner = cls(config, tokenizer=tokenizer, encoder_from_pretrained=False)
            # to be able to laod GLiNER models from previous version
            if (config.class_token_index==-1 or config.vocab_size == -1) and resize_token_embeddings:
                gliner.resize_token_embeddings(add_tokens=add_tokens)
            state_dict = torch.load(model_file, map_location=torch.device(map_location))
            gliner.model.load_state_dict(state_dict, strict=strict)
            gliner.model.to(map_location)
            if compile_torch_model and 'cuda' in map_location:
                print("Compiling torch model...")
                gliner.compile()
            elif compile_torch_model:
                warnings.warn("It's not possible to compile this model putting it to CPU, you should set `map_location` to `cuda`.")
            gliner.eval()
        else:
            import onnxruntime as ort

            model_file = Path(model_dir) / onnx_model_file
            if not os.path.exists(model_file):
                raise FileNotFoundError(f"The ONNX model can't be loaded from {model_file}.")
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            ort_session = ort.InferenceSession(model_file, session_options)
            if config.span_mode=='token_level':
                model = TokenORTModel(ort_session)
            else:
                model = SpanORTModel(ort_session)

            gliner = cls(config, tokenizer=tokenizer, model=model)
            if (config.class_token_index==-1 or config.vocab_size == -1) and resize_token_embeddings:
                gliner.data_processor.transformer_tokenizer.add_tokens(add_tokens)

        return gliner