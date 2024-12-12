from typing import Optional, List, Union
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import torch
from datasets import load_dataset, Dataset
from gliner import GLiNER

from .base import GLiNERBasePipeline

class GLiNERQuestionAnswerer(GLiNERBasePipeline):
    """
    A class to use GLiNER for question-answering inference and evaluation.

    Attributes:
        device (str): Device to run the model on, e.g., 'cuda:0' or 'cpu'.
        model (GLiNER): Loaded GLiNER model instance.
        prompt (str): Template prompt for text question-asnwering.

    Methods:
        process_predictions(predictions):
            Processes model predictions to extract the most likely labels.
        prepare_texts(texts, labels):
            Creates Q&A prompts for each input text.
        __call__(texts, labels, threshold=0.5):
            Runs the model on the given texts and returns predicted labels.
        evaluate(dataset_id, labels=None, threshold=0.5, max_examples=-1):
            Evaluates the model on a dataset and computes F1 scores.
    """

    prompt = "Answer the following question: {}"

    def __init__(self, model_id: str = None, model: GLiNER = None, device: str = 'cuda:0', prompt: Optional[str] = None):
        """
        Initializes the GLiNERQuestionAnswerer.

        Args:
            model_id (str, optional): Identifier for the model to be loaded. Defaults to None.
            model (GLiNER, optional): Preloaded GLiNER model. Defaults to None.
            device (str, optional): Device to run the model on ('cpu' or 'cuda:X'). Defaults to 'cuda:0'.
            prompt (str, optional): Template prompt for question-answering.
        """
        # Use the provided prompt or default to the class-level prompt
        prompt = prompt if prompt is not None else self.prompt
        super().__init__(model_id=model_id, model=model, prompt=prompt, device=device)


    def process_predictions(self, predictions, **kwargs):
        """
        Processes predictions to extract the highest-scoring answer(s).

        Args:
            predictions (list): List of predictions with scores.

        Returns:
            list: List of predicted labels for each input.
        """
        batch_predicted_labels = []

        for prediction in predictions:
            # Sort predictions by score in descending order
            sorted_predictions = sorted(prediction, key=lambda entity: entity["score"], reverse=True)

            predicted_labels = [{'answer': pred['text'], 'score': pred['score']} for pred in sorted_predictions]
            batch_predicted_labels.append(predicted_labels)

        return batch_predicted_labels

    def prepare_texts(self, texts: List[str], questions: Union[List[str], str], **kwargs):
        """
        Prepares prompts for question-answering by appending questions to texts.

        Args:
            texts (list): List of input texts.
            questions (list|str): Question or list of questions.

        Returns:
            list: List of formatted prompts.
        """
        prompts = []

        for id, text in enumerate(texts):
            if isinstance(questions, str):
                question = questions
            else:
                question = questions[0]
            prompt = f"{self.prompt.format(question)} \n {text}"
            prompts.append(prompt)
        return prompts

    def __call__(self, texts: Union[str, List[str]], questions: Union[str, List[str]], 
                                labels: List[str] = ['answer'], threshold: float = 0.5,
                                                            batch_size: int = 8, **kwargs):
        return super().__call__(texts, labels, threshold, batch_size, questions=questions)
    
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
            dict: A dictionary containing evaluation metrics such as F1 scores.
        
        Raises:
            ValueError: If neither `dataset_id` nor `dataset` is provided.
        """
        raise NotImplementedError("Currently `evaluate` method is not implemented.")

class GLiNERSquadEvaluator(GLiNERQuestionAnswerer):
    def evaluate(self, dataset_id: str = 'rajpurkar/squad_v2', dataset: Optional[Dataset] = None, 
                    labels: Optional[List[str]] = ['answer'], threshold: float = 0.5, max_examples: int = -1):
        """
        Evaluates the model on a specified dataset and computes evaluation metrics.

        Args:
            dataset_id (str, optional): Identifier for the dataset to load (e.g., from Hugging Face datasets).
            dataset (Dataset, optional): A pre-loaded dataset to evaluate. If provided, `dataset_id` is ignored.
            labels (list, optional): List of target labels to consider for classification. Defaults to ['answer'].
            threshold (float): Confidence threshold for predictions. Defaults to 0.5.
            max_examples (int): Maximum number of examples to evaluate. Defaults to -1 (use all available examples).

        Returns:
            dict: A dictionary containing evaluation metrics such as F1 Scores.

        Raises:
            ValueError: If neither `dataset_id` nor `dataset` is provided.
        """
        from evaluate import load

        # Validate input
        if not dataset and not dataset_id:
            raise ValueError("Either `dataset` or `dataset_id` must be provided.")

        # Load the dataset if not provided
        if not dataset:
            dataset = load_dataset(dataset_id, split="validation")

        if not isinstance(dataset, Dataset):
            dataset = dataset['validation']

        # Truncate dataset if max_examples is specified
        if max_examples > 0:
            dataset = dataset.shuffle().select(range(min(len(dataset), max_examples)))

        # Load evaluation metric for SQuAD
        squad_metric = load("squad_v2" if "squad_v2" in dataset_id else "squad")

        # Prepare predictions and references
        contexts = dataset['context']
        questions = dataset['question']

        raw_predictions = self(contexts, questions, labels=labels, threshold=threshold)

        predictions = []
        references = []
        for id, prediction in enumerate(raw_predictions):
            example = dataset[id]

            if len(prediction):
                predicted_answer = prediction[0]["answer"]
                no_answer_probability=0.0
            else:
                predicted_answer = ""
                no_answer_probability=1.0

            # Append to predictions and references
            predictions.append({
                "id": example["id"],
                "prediction_text": predicted_answer,
                "no_answer_probability": no_answer_probability
            })

            references.append({
                "id": example["id"],
                "answers": {"text": example["answers"]["text"], "answer_start": example["answers"]["answer_start"]}
            })

        # Compute metrics
        results = squad_metric.compute(predictions=predictions, references=references)
        return results