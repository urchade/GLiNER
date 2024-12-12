from typing import Optional, List, Union
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import torch
from datasets import load_dataset, Dataset
from gliner import GLiNER

from .base import GLiNERBasePipeline

class GLiNERSummarizer(GLiNERBasePipeline):
    """
    A class to use GLiNER for summarization inference and evaluation.

    Attributes:
        device (str): Device to run the model on, e.g., 'cuda:0' or 'cpu'.
        model (GLiNER): Loaded GLiNER model instance.
        prompt (str): Template prompt for text summarization.

    Methods:
        process_predictions(predictions):
            Processes model predictions to extract the most likely labels.
        prepare_texts(texts, labels):
            Creates summarization prompts for each input text.
        __call__(texts, labels, threshold=0.5):
            Runs the model on the given texts and returns predicted labels.
        evaluate(dataset_id, labels=None, threshold=0.5, max_examples=-1):
            Evaluates the model on a dataset and computes F1 scores.
    """

    prompt = "Summarize the following text highlighting the most important information:"

    def __init__(self, model_id: str = None, model: GLiNER = None, device: str = 'cuda:0', prompt: Optional[str] = None):
        """
        Initializes the GLiNERSummarizer.

        Args:
            model_id (str, optional): Identifier for the model to be loaded. Defaults to None.
            model (GLiNER, optional): Preloaded GLiNER model. Defaults to None.
            device (str, optional): Device to run the model on ('cpu' or 'cuda:X'). Defaults to 'cuda:0'.
            prompt (str, optional): Template prompt for summarization.
        """
        # Use the provided prompt or default to the class-level prompt
        prompt = prompt if prompt is not None else self.prompt
        super().__init__(model_id=model_id, model=model, prompt=prompt, device=device)


    def process_predictions(self, predictions, **kwargs):
        """
        Processes predictions to extract the highest-scoring text chunk(s).

        Args:
            predictions (list): List of predictions with scores.

        Returns:
            list: List of predicted labels for each input.
        """
        batch_predicted_labels = []

        for prediction in predictions:
            # Sort predictions by score in descending order
            sorted_predictions = sorted(prediction, key=lambda entity: entity["start"], reverse=False)

            extracted_text = [pred['text'] for pred in sorted_predictions]
            batch_predicted_labels.append(' '.join(extracted_text))

        return batch_predicted_labels

    def prepare_texts(self, texts: List[str], **kwargs):
        """
        Prepares prompts for summarization by appending prompt to texts.

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

    def __call__(self, texts: Union[str, List[str]], labels: List[str] = ['summary'], 
                                    threshold: float = 0.25, batch_size: int = 8, **kwargs):
        return super().__call__(texts, labels, threshold, batch_size)
    
    def evaluate(self, dataset_id: Optional[str] = None, dataset: Optional[Dataset] = None, 
                    labels: Optional[List[str]]=None, threshold: float =0.5, max_examples: float =-1):
        """
        Evaluates the model on a specified dataset and computes evaluation metrics.

        Args:
            dataset_id (str, optional): Identifier for the dataset to load (e.g., from Hugging Face datasets).
            dataset (Dataset, optional): A pre-loaded dataset to evaluate. If provided, `dataset_id` is ignored.
            labels (list, optional): List of target labels to consider for summarization. Defaults to None (use all).
            threshold (float): Confidence threshold for predictions. Defaults to 0.5.
            max_examples (int): Maximum number of examples to evaluate. Defaults to -1 (use all available examples).

        Returns:
            dict: A dictionary containing evaluation metrics.
        
        Raises:
            ValueError: If neither `dataset_id` nor `dataset` is provided.
        """
        raise NotImplementedError("Currently `evaluate` method is not implemented.")