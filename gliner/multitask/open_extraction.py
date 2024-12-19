from typing import Optional, List, Union
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import torch
from datasets import load_dataset, Dataset
from gliner import GLiNER

from .base import GLiNERBasePipeline

class GLiNEROpenExtractor(GLiNERBasePipeline):
    """
    A class to use GLiNER for open information extraction inference and evaluation.

    Attributes:
        device (str): Device to run the model on, e.g., 'cuda:0' or 'cpu'.
        model (GLiNER): Loaded GLiNER model instance.
        prompt (str): Template prompt for open information extraction.

    Methods:
        process_predictions(predictions):
            Processes model predictions to extract the most likely labels.
        prepare_texts(texts, labels):
            Creates open information extraction prompts for each input text.
        __call__(texts, labels, threshold=0.5):
            Runs the model on the given texts and returns predicted labels.
        evaluate(dataset_id, labels=None, threshold=0.5, max_examples=-1):
            Evaluates the model on a dataset and computes F1 scores.
    """

    prompt = ""

    def __init__(self, model_id: str = None, model: GLiNER = None, device: str = 'cuda:0', prompt: Optional[str] = None):
        """
        Initializes the GLiNEROpenExtractor.

        Args:
            model_id (str, optional): Identifier for the model to be loaded. Defaults to None.
            model (GLiNER, optional): Preloaded GLiNER model. Defaults to None.
            device (str, optional): Device to run the model on ('cpu' or 'cuda:X'). Defaults to 'cuda:0'.
            prompt (str, optional): Template prompt for open information extraction.
        """
        # Use the provided prompt or default to the class-level prompt
        prompt = prompt if prompt is not None else self.prompt
        super().__init__(model_id=model_id, model=model, prompt=prompt, device=device)


    def process_predictions(self, predictions, **kwargs):
        """
        Processes predictions to extract the highest-scoring label(s).

        Args:
            predictions (list): List of predictions with scores.

        Returns:
            list: List of predicted labels for each input.
        """
        return predictions

    def prepare_texts(self, texts: List[str], **kwargs):
        """
        Prepares prompts for open-information extraction.

        Args:
            texts (list): List of input texts.

        Returns:
            list: List of formatted prompts.
        """
        prompts = []

        for id, text in enumerate(texts):
            prompt = f"{self.prompt} \n {text}"
            prompts.append(prompt)
        return prompts

    
    def evaluate(self, dataset_id: Optional[str] = None, dataset: Optional[Dataset] = None, 
                    labels: Optional[List[str]]=None, threshold: float =0.5, max_examples: float =-1):
        """
        Evaluates the model on a specified dataset and computes evaluation metrics.

        Args:
            dataset_id (str, optional): Identifier for the dataset to load (e.g., from Hugging Face datasets).
            dataset (Dataset, optional): A pre-loaded dataset to evaluate. If provided, `dataset_id` is ignored.
            labels (list, optional): List of target labels to consider for extraction. Defaults to None (use all).
            threshold (float): Confidence threshold for predictions. Defaults to 0.5.
            max_examples (int): Maximum number of examples to evaluate. Defaults to -1 (use all available examples).

        Returns:
            dict: A dictionary containing evaluation metrics.
        
        Raises:
            ValueError: If neither `dataset_id` nor `dataset` is provided.
        """
        raise NotImplementedError("Currently `evaluate` method is not implemented.")