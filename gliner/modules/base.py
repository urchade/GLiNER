import argparse
import json
from abc import abstractmethod
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Union, Dict

import torch
import yaml
from gliner.modules.data import SpanData, TokenData
from gliner.modules.evaluator import Evaluator
from gliner.modules.token_splitter import WhitespaceTokenSplitter, MecabKoTokenSplitter, SpaCyTokenSplitter
from huggingface_hub import hf_hub_download
from torch import nn
from torch.utils.data import DataLoader
from huggingface_hub import PyTorchModelHubMixin


class InstructBase(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if config.span_mode == 'token_level':
            self.data_proc = TokenData(config)
        else:
            self.data_proc = SpanData(config)

        if not hasattr(config, 'token_splitter'):
            self.token_splitter = WhitespaceTokenSplitter()
        elif self.config.token_splitter == "spacy":
            lang = getattr(config, 'token_splitter_lang', None)
            self.token_splitter = SpaCyTokenSplitter(lang=lang)
        elif self.config.token_splitter == "mecab-ko":
            self.token_splitter = MecabKoTokenSplitter()

    # you have to implement forward and predict methods (not implemented here)
    @abstractmethod
    def forward(self, x):
        """Returns the loss for the given input."""
        pass

    @abstractmethod
    def predict(self, x, flat_ner=True, threshold=0.5, multi_label=False):
        """Returns the predicted entities for the given input."""
        pass

    def create_dataloader(self, data, entity_types=None, **kwargs) -> DataLoader:
        return self.data_proc.create_dataloader(data, entity_types, **kwargs)

    def predict_entities(self, text, labels, flat_ner=True, threshold=0.5, multi_label=False):
        return self.batch_predict_entities(
            [text], labels, flat_ner=flat_ner, threshold=threshold, multi_label=multi_label
        )[0]

    def batch_predict_entities(self, texts, labels, flat_ner=True, threshold=0.5, multi_label=False):
        """
        Predict entities for a batch of texts.
        texts:  List of texts | List[str]
        labels: List of labels | List[str]
        ...
        """

        all_tokens = []
        all_start_token_idx_to_text_idx = []
        all_end_token_idx_to_text_idx = []

        for text in texts:
            tokens = []
            start_token_idx_to_text_idx = []
            end_token_idx_to_text_idx = []
            for token, start, end in self.token_splitter(text):
                tokens.append(token)
                start_token_idx_to_text_idx.append(start)
                end_token_idx_to_text_idx.append(end)
            all_tokens.append(tokens)
            all_start_token_idx_to_text_idx.append(start_token_idx_to_text_idx)
            all_end_token_idx_to_text_idx.append(end_token_idx_to_text_idx)

        input_x = [{"tokenized_text": tk, "ner": None} for tk in all_tokens]
        x = self.data_proc.collate_fn(input_x, labels)
        outputs = self.predict(x, flat_ner=flat_ner, threshold=threshold, multi_label=multi_label)

        all_entities = []
        for i, output in enumerate(outputs):
            start_token_idx_to_text_idx = all_start_token_idx_to_text_idx[i]
            end_token_idx_to_text_idx = all_end_token_idx_to_text_idx[i]
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
        self.eval()
        data_loader = self.create_dataloader(test_data, batch_size=batch_size, entity_types=entity_types, shuffle=False)
        device = next(self.parameters()).device
        all_preds = []
        all_trues = []
        for x in data_loader:
            for k, v in x.items():
                if isinstance(v, torch.Tensor):
                    x[k] = v.to(device)
            batch_predictions = self.predict(x, flat_ner, threshold, multi_label)
            all_preds.extend(batch_predictions)
            all_trues.extend(x["entities"])
        evaluator = Evaluator(all_trues, all_preds)
        out, f1 = evaluator.evaluate()
        return out, f1

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

    def save_pretrained(
            self,
            save_directory: Union[str, Path],
            *,
            config: Optional[Union[dict, "DataclassInstance"]] = None,
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
        torch.save(self.state_dict(), save_directory / "pytorch_model.bin")

        # save config (if provided)
        if config is None:
            config = self.config
        if config is not None:
            if isinstance(config, argparse.Namespace) or isinstance(config, SimpleNamespace):
                config = vars(config)
            (save_directory / "gliner_config.json").write_text(json.dumps(config, indent=2))

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
            **model_kwargs,
    ):

        # Newer format: Use "pytorch_model.bin" and "gliner_config.json"
        model_file = Path(model_id) / "pytorch_model.bin"
        if not model_file.exists():
            model_file = hf_hub_download(
                repo_id=model_id,
                filename="pytorch_model.bin",
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )
        config_file = Path(model_id) / "gliner_config.json"
        if not config_file.exists():
            config_file = hf_hub_download(
                repo_id=model_id,
                filename="gliner_config.json",
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )
        config = load_config_as_namespace(config_file)
        model = cls(config)
        state_dict = torch.load(model_file, map_location=torch.device(map_location))
        model.load_state_dict(state_dict, strict=strict)
        model.to(map_location)
        return model

    def to(self, device):
        super().to(device)
        import flair
        flair.device = device
        return self


def load_config_as_namespace(config_file):
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    return argparse.Namespace(**config_dict)
