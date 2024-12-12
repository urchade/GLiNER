from typing import Optional, List
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import torch
from datasets import load_dataset, Dataset
from sklearn.metrics import f1_score
from gliner import GLiNER

from .base import GLiNERBasePipeline

class GLiNERClassifier(GLiNERBasePipeline):
    """
    A class to evaluate the GLiNER model for classification tasks using F1 scores.

    Attributes:
        device (str): Device to run the model on, e.g., 'cuda:0' or 'cpu'.
        model (GLiNER): Loaded GLiNER model instance.
        prompt (str): Template prompt for text classification.

    Methods:
        compute_f_score(predicts, true_labels):
            Computes micro, macro, and weighted F1 scores.
        prepare_dataset(dataset, classes=None, text_column='text', label_column='label', split=None, max_examples=-1):
            Prepares texts and true labels from the given dataset.
        process_predictions(predictions):
            Processes model predictions to extract the most likely labels.
        prepare_texts(texts, labels):
            Creates classification prompts for each input text.
        __call__(texts, labels, threshold=0.5):
            Runs the model on the given texts and returns predicted labels.
        evaluate(dataset_id, labels=None, threshold=0.5, max_examples=-1):
            Evaluates the model on a dataset and computes F1 scores.
    """

    prompt = "Classify text into the following classes: {}"

    def __init__(self, model_id: str = None, model: GLiNER = None, device: str = 'cuda:0', prompt: Optional[str] = None):
        """
        Initializes the GLiNERClassifier.

        Args:
            model_id (str, optional): Identifier for the model to be loaded. Defaults to None.
            model (GLiNER, optional): Preloaded GLiNER model. Defaults to None.
            device (str, optional): Device to run the model on ('cpu' or 'cuda:X'). Defaults to 'cuda:0'.
            prompt (str, optional): Template prompt for text classification. Defaults to the class-level prompt.
        """
        # Use the provided prompt or default to the class-level prompt
        prompt = prompt if prompt is not None else self.prompt
        super().__init__(model_id=model_id, model=model, prompt=prompt, device=device)


    def compute_f_score(self, predicts, true_labels):
        """
        Computes the micro, macro, and weighted F1 scores.

        Args:
            predicts (list): List of predicted labels.
            true_labels (list): List of true labels.

        Returns:
            dict: Dictionary with micro, macro, and weighted F1 scores.
        """
        micro = f1_score(true_labels, predicts, average="micro")
        macro = f1_score(true_labels, predicts, average="macro")
        weighted = f1_score(true_labels, predicts, average="weighted")
        return {"micro": micro, "macro": macro, "weighted": weighted}

    def prepare_dataset(self, dataset: Dataset, classes=None, text_column='text', label_column="label", split=None, max_examples=-1):
        """
        Prepares the dataset by extracting texts and true labels.

        Args:
            dataset (Dataset or dict): The dataset to prepare.
            classes (list, optional): List of class labels. Defaults to None.
            text_column (str): Name of the text column. Defaults to 'text'.
            label_column (str): Name of the label column. Defaults to 'label'.
            split (str, optional): Delimiter for splitting class names. Defaults to None.
            max_examples (int): Maximum number of examples to use. Defaults to -1 (use all).

        Returns:
            tuple: Texts, classes, and true labels.
        """
        if 'test' in dataset:
            test_dataset = dataset['test']
        elif isinstance(dataset, Dataset):
            test_dataset = dataset
        else:
            test_dataset = dataset['train']
        
        if classes is None:
            classes = test_dataset.features[label_column].names
            if split is not None:
                classes = [' '.join(class_.split(split)) for class_ in classes]

        texts = test_dataset[text_column]
        true_labels = test_dataset[label_column]

        if isinstance(test_dataset[label_column][0], int):
            true_labels = [classes[label] for label in true_labels]

        if max_examples > 0:
            texts = texts[:max_examples]
            true_labels = true_labels[:max_examples]

        return texts, classes, true_labels

    def process_predictions(self, predictions, multi_label=False, **kwargs):
        """
        Processes predictions to extract the highest-scoring label(s).

        Args:
            predictions (list): List of predictions with scores.
            multi_label (bool): Whether to allow multiple labels per input. Defaults to False.

        Returns:
            list: List of predicted labels for each input.
        """
        batch_predicted_labels = []

        for prediction in predictions:
            # Sort predictions by score in descending order
            sorted_predictions = sorted(prediction, key=lambda entity: entity["score"], reverse=True)

            if not sorted_predictions:
                # Default prediction if no valid predictions are found
                batch_predicted_labels.append([{'label': 'other', 'score': 1.0}])
                continue

            if not multi_label:
                # Single-label mode: select the top prediction and compute softmax score
                scores = [item['score'] for item in sorted_predictions]
                softmax_scores = torch.softmax(torch.tensor(scores), dim=0).tolist()
                top_prediction = {'label': sorted_predictions[0]['text'], 'score': softmax_scores[0]}
                batch_predicted_labels.append([top_prediction])
            else:
                # Multi-label mode: retain all predictions with original scores
                predicted_labels = [{'label': pred['text'], 'score': pred['score']} for pred in sorted_predictions]
                batch_predicted_labels.append(predicted_labels)

        return batch_predicted_labels

    def prepare_texts(self, texts, classes, **kwargs):
        """
        Prepares prompts for classification by appending labels to texts.

        Args:
            texts (list): List of input texts.
            classes (list): List of classification labels.

        Returns:
            list: List of formatted prompts.
        """
        prompts = []
        labels_ = ', '.join(classes)
        for text in texts:
            prompt = f"{self.prompt.format(labels_)} \n {text}"
            prompts.append(prompt)
        return prompts

    def evaluate(self, dataset_id: Optional[str] = None, dataset: Optional[Dataset] = None, 
                    labels: Optional[List[str]]=None, threshold: float =0.5, max_examples: float =-1):
        """
        Evaluates the model on a specified dataset and computes evaluation metrics.

        Args:
            dataset_id (str, optional): Identifier for the dataset to load (e.g., from Hugging Face datasets).
            dataset (Dataset, optional): A pre-loaded dataset to evaluate. If provided, `dataset_id` is ignored.
            labels (list, optional): List of target labels to consider for classification. Defaults to None (use all).
            threshold (float): Confidence threshold for predictions. Defaults to 0.5.
            max_examples (int): Maximum number of examples to evaluate. Defaults to -1 (use all available examples).

        Returns:
            dict: A dictionary containing evaluation metrics such as F1 scores (micro, macro, and weighted).
        
        Raises:
            ValueError: If neither `dataset_id` nor `dataset` is provided.
        """
        if dataset is None and dataset_id is not None:
            dataset = load_dataset(dataset_id)
        else:
            raise ValueError("Either 'dataset_id' or 'dataset' must be provided to start evaluation.")
        
        test_texts, classes, true_labels = self.prepare_dataset(dataset, labels, max_examples=max_examples)
        
        predictions = self.__call__(test_texts, classes=classes, threshold=threshold)
        predicted_labels = [pred[0]['label'] for pred in predictions]

        return self.compute_f_score(predicted_labels, true_labels)