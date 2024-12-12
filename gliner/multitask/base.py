from abc import ABC, abstractmethod
from typing import List, Union, Optional
import torch
import warnings

from ..model import GLiNER

class GLiNERBasePipeline(ABC):
    """
    Base class for GLiNER pipelines. Provides an interface for preparing texts,
    processing predictions, and evaluating the model.

    Args:
        model_id (str): Identifier for the model to be loaded.
        prompt (str, optional): Prompt template for text preparation. Defaults to None.
        device (str, optional): Device to run the model on ('cpu' or 'cuda:X'). Defaults to 'cuda:0'.

    Attributes:
        model (GLiNER): The loaded GLiNER model.
        device (str): The device being used for computation.
        prompt (str): The prompt template for text preparation.
    """

    def __init__(self, model_id: str = None, model: GLiNER = None, prompt=None, device='cuda:0'):
        """
        Initializes the GLiNERBasePipeline.

        Args:
            model_id (str): Identifier for the model to be loaded.
            prompt (str, optional): Prompt template for text preparation. Defaults to None.
            device (str, optional): Device to run the model on ('cpu' or 'cuda:X'). Defaults to 'cuda:0'.
        """
        if 'cuda' in device and not torch.cuda.is_available():
            warnings.warn(f"{device} is not available, setting device as 'cpu'.")
            device = 'cpu'
        self.device = device

        if model is not None:
            self.model = model.to(self.device)
        elif model_id is not None:
            self.model = GLiNER.from_pretrained(model_id).to(self.device)
        else:
            raise ValueError("Either 'model_id' or 'model' must be provided to initialize the pipeline.")
        
        self.prompt = prompt

    @abstractmethod
    def prepare_texts(self, texts: List[str], *args, **kwargs):
        """
        Prepares texts for input to the model.

        Args:
            texts (List[str]): List of input texts.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: The processed texts ready for model input.
        """
        pass

    @abstractmethod
    def process_predictions(self, predictions: List[dict]):
        """
        Processes model predictions into the desired format.

        Args:
            predictions (List[dict]): Raw predictions from the model.

        Returns:
            Any: Processed predictions in the desired format.
        """
        pass

    @abstractmethod
    def evaluate(self, dataset_id: str, labels: Optional[List[str]] = None, threshold: float = 0.5):
        """
        Evaluates the model on a given dataset.

        Args:
            dataset_id (str): Identifier for the evaluation dataset.
            labels (Optional[List[str]]): List of labels to evaluate. Defaults to None.
            threshold (float): Threshold for prediction confidence. Defaults to 0.5.

        Returns:
            Any: Evaluation results.
        """
        pass

    def __call__(self, texts: Union[str, List[str]], labels: List[str] = ['match'], 
                             threshold: float = 0.5, batch_size: int = 8, **kwargs):
        """
        Runs the model on the provided texts and returns processed results.

        Args:
            texts (Union[str, List[str]]): Single or list of input texts.
            labels (Optional[List[str]]): List of class labels for text preparation. Defaults to None.
            threshold (float): Threshold for prediction confidence. Defaults to 0.5.
            batch_size (int): Batch size for processing. Defaults to 8.

        Returns:
            Any: Processed results from the model.
        """
        if isinstance(texts, str):
            texts = [texts]
        
        prompts = self.prepare_texts(texts, **kwargs)

        predictions = self.model.run(prompts, labels, threshold=threshold, batch_size=batch_size)

        results = self.process_predictions(predictions, **kwargs)

        return results