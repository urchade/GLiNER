from typing import Optional, List, Union
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import torch
from datasets import load_dataset, Dataset
from gliner import GLiNER

from .base import GLiNERBasePipeline

class GLiNERRelationExtractor(GLiNERBasePipeline):
    """
    A class to use GLiNER for relation extraction inference and evaluation.

    Attributes:
        device (str): Device to run the model on, e.g., 'cuda:0' or 'cpu'.
        model (GLiNER): Loaded GLiNER model instance.
        prompt (str): Template prompt for relation extraction.

    Methods:
        process_predictions(predictions):
            Processes model predictions to extract the most likely labels.
        prepare_texts(texts, labels):
            Creates relation extraction prompts for each input text.
        __call__(texts, labels, threshold=0.5):
            Runs the model on the given texts and returns predicted labels.
        evaluate(dataset_id, labels=None, threshold=0.5, max_examples=-1):
            Evaluates the model on a dataset and computes F1 scores.
    """

    prompt = "Extract relationships between entities from the text: "

    def __init__(self, model_id: str = None, model: GLiNER = None, device: str = 'cuda:0', prompt: Optional[str] = None):
        """
        Initializes the GLiNERRelationExtractor.

        Args:
            model_id (str, optional): Identifier for the model to be loaded. Defaults to None.
            model (GLiNER, optional): Preloaded GLiNER model. Defaults to None.
            device (str, optional): Device to run the model on ('cpu' or 'cuda:X'). Defaults to 'cuda:0'.
            prompt (str, optional): Template prompt for question-answering.
        """
        # Use the provided prompt or default to the class-level prompt
        prompt = prompt if prompt is not None else self.prompt
        super().__init__(model_id=model_id, model=model, prompt=prompt, device=device)

    def prepare_texts(self, texts: List[str], **kwargs):
        """
        Prepares prompts for relation extraction to texts.

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

    def prepare_source_relation(self, ner_predictions: List[dict], relations: List[str]):
        relation_labels = []
        for prediction in ner_predictions:
            curr_labels = []
            unique_entities = {ent['text'] for ent in prediction}
            for relation in relations:
                for ent in unique_entities:
                    curr_labels.append(f"{ent} <> {relation}")
            relation_labels.append(curr_labels)
        return relation_labels

    def process_predictions(self, predictions, **kwargs):
        """
        Processes predictions to extract the highest-scoring relation(s).

        Args:
            predictions (list): List of predictions with scores.

        Returns:
            list: List of predicted labels for each input.
        """
        batch_predicted_relations = []

        for prediction in predictions:
            # Sort predictions by score in descending order
            curr_relations = []

            for target in prediction:
                target_ent = target['text']
                score = target['score']
                source, relation = target['label'].split('<>')
                relation = {
                    "source": source.strip(),
                    "relation": relation.strip(),
                    "target": target_ent.strip(),
                    "score": score
                }
                curr_relations.append(relation)
            batch_predicted_relations.append(curr_relations)

        return batch_predicted_relations
    
    def __call__(self, texts: Union[str, List[str]], relations: List[str]=None, 
                                entities: List[str] = ['named entity'], 
                                relation_labels: Optional[List[List[str]]]=None, 
                                threshold: float = 0.5, batch_size: int = 8, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        
        prompts = self.prepare_texts(texts, **kwargs)

        if relation_labels is None:
            # ner
            ner_predictions = self.model.run(texts, entities, threshold=threshold, batch_size=batch_size)
            #rex
            relation_labels = self.prepare_source_relation(ner_predictions, relations)

        predictions = self.model.run(prompts, relation_labels, threshold=threshold, batch_size=batch_size)

        results = self.process_predictions(predictions, **kwargs)

        return results
    
    def evaluate(self, dataset_id: Optional[str] = None, dataset: Optional[Dataset] = None, 
                    labels: Optional[List[str]]=None, threshold: float =0.5, max_examples: float =-1):
        """
        Evaluates the model on a specified dataset and computes evaluation metrics.

        Args:
            dataset_id (str, optional): Identifier for the dataset to load (e.g., from Hugging Face datasets).
            dataset (Dataset, optional): A pre-loaded dataset to evaluate. If provided, `dataset_id` is ignored.
            labels (list, optional): List of target labels to consider for relation extraction. Defaults to None (use all).
            threshold (float): Confidence threshold for predictions. Defaults to 0.5.
            max_examples (int): Maximum number of examples to evaluate. Defaults to -1 (use all available examples).

        Returns:
            dict: A dictionary containing evaluation metrics such as F1 scores.
        
        Raises:
            ValueError: If neither `dataset_id` nor `dataset` is provided.
        """
        raise NotImplementedError("Currently `evaluate` method is not implemented.")

class GLiNERDocREDEvaluator(GLiNERRelationExtractor):
    """
    Evaluator class for document-level relation extraction tasks using the GLiNER framework.

    This class includes methods for preparing datasets, processing predictions, computing F1 scores,
    and evaluating the model's performance on document-level relation extraction tasks such as DocRED.
    """

    def prepare_dataset(self, raw_data: Dataset, text_column='sents', rel_column='labels', *args, **kwargs):
        """
        Prepares the dataset for evaluation by extracting labeled relations and corresponding text.

        Args:
            raw_data (Dataset): A list of raw dataset examples where each example contains sentences,
                            entity mentions, and relation annotations.
            text_column (str, optional): Column name in the dataset containing sentences. Defaults to 'sents'.
            rel_column (str, optional): Column name in the dataset containing relation labels. Defaults to 'labels'.

        Returns:
            tuple: A tuple containing:
                - texts_by_line (list of str): Flattened and concatenated text for each document.
                - grouped_labels (list of list of str): Grouped relation labels for each document.
                - true_labels (list of str): True relation labels in "source <> relation <> target" format.
        """
        grouped_labels = []
        true_labels = []
        texts_by_line = []

        for item in raw_data:

            vertex_set = item.get('vertexSet')
            sents = item.get(text_column, [])
            labels = item.get(rel_column, [])

            current_labels=[]

            for head_id, tail_id, relation in zip(labels['head'], labels['tail'], labels['relation_text']):
                current_index = 0
                head_data = None
                tail_data = None

                for sublist in vertex_set:
                      if current_index == head_id:
                          head_data = sublist
                      current_index += 1

                current_index = 0

                for sublist in vertex_set:
                      if current_index == tail_id:
                          tail_data = sublist
                      current_index += 1

                head_name = head_data[0]['name'] if head_data else None
                tail_name = tail_data[0]['name'] if tail_data else None

                true_labels.append(f'{head_name} <> {relation} <> {tail_name}')
                current_labels.append(f'{head_name} <> {relation}')

            grouped_labels.append(current_labels)
            result = " ".join(string for sublist in  sents for string in sublist)
            texts_by_line.append(result)

        return texts_by_line, grouped_labels, true_labels

    def process_results(self, predictions: List[dict]):
        """
        Processes model predictions into the standard "source <> relation <> target" format.

        Args:
            predictions (list of dict): List of prediction dictionaries containing 'source', 'relation', and 'target'.

        Returns:
            list of str: Processed predictions in "source <> relation <> target" format.
        """
        preds = []
        preds = []
        for predict in predictions:
            print(predict)
            for pred_ in predict:
                result = f"{pred_['source']} <> {pred_['relation']} <> {pred_['target']}"
                preds.append(result)
        return preds

    def compute_f_score(self, predicts: List[str], true_labels: List[str]):
        """
        Computes precision, recall, F1 score, and other metrics for the relation extraction task.

        Args:
            predicts (list of str): Predicted relation labels in "source <> relation <> target" format.
            true_labels (list of str): True relation labels in "source <> relation <> target" format.

        Returns:
            tuple: A tuple containing:
                - precision (float): Precision of predictions.
                - recall (float): Recall of predictions.
                - f1 (float): F1 score of predictions.
                - tp (int): Number of true positives.
                - fp (int): Number of false positives.
                - fn (int): Number of false negatives.
        """
        true_set = set(true_labels)
        pred_set = set(predicts)

        tp = len(true_set.intersection(pred_set))
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

        return {'precision': precision, 'recall': recall, 'f1': f1, 
                    'true positives': tp, 'false positives': fp,  'false negatives': fn}

    def evaluate(self, dataset_id: str = 'thunlp/docred', dataset: Optional[Dataset] = None, 
                    labels: Optional[List[str]] = None, threshold: float = 0.5, max_examples: int = -1):

        """
        Evaluates the model on a specified dataset and computes evaluation metrics.

        Args:
            dataset_id (str, optional): Identifier for the dataset to load (e.g., from Hugging Face datasets).
            dataset (Dataset, optional): A pre-loaded dataset to evaluate. If provided, `dataset_id` is ignored.
            labels (list, optional): List of target relation labels to consider. Defaults to None (use all).
            threshold (float): Confidence threshold for predictions. Defaults to 0.5.
            max_examples (int): Maximum number of examples to evaluate. Defaults to -1 (use all available examples).

        Returns:
            tuple: Evaluation metrics including precision, recall, F1 score, true positives, false positives,
                   and false negatives.

        Raises:
            ValueError: If neither `dataset_id` nor `dataset` is provided.
        """

        if not dataset and not dataset_id:
            raise ValueError("Either `dataset` or `dataset_id` must be provided.")

        # Load the dataset if not provided
        if not dataset:
            dataset = load_dataset(dataset_id, split="validation")

        if not isinstance(dataset, Dataset):
            dataset = dataset['validation']

        if max_examples > 0:
            dataset = dataset.shuffle().select(range(min(len(dataset), max_examples)))

        test_texts, labels, true_labels = self.prepare_dataset(dataset)
        predictions = self(test_texts, relation_labels=labels)
        preds = self.process_results(predictions)
        return self.compute_f_score(preds, true_labels)