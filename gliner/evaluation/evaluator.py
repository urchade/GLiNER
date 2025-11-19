from abc import ABC, abstractmethod

import numpy as np
import torch

from .utils import _prf_divide, flatten_for_eval, extract_tp_actual_correct


class BaseEvaluator(ABC):
    """Abstract base class for evaluation of NER and relation extraction tasks.

    Provides common functionality for computing precision, recall, and F1 scores
    from ground truth and predicted annotations. Subclasses must implement
    transform_data() to convert task-specific data formats.

    Attributes:
        all_true: List of ground truth annotations for all samples.
        all_outs: List of predicted annotations for all samples.
    """

    def __init__(self, all_true, all_outs):
        """Initialize the evaluator with ground truth and predictions.

        Args:
            all_true: List of ground truth annotations for all samples.
                Format depends on the specific evaluator subclass.
            all_outs: List of predicted annotations for all samples.
                Format depends on the specific evaluator subclass.
        """
        self.all_true = all_true
        self.all_outs = all_outs

    @staticmethod
    def compute_prf(y_true, y_pred, average="micro"):
        """Compute precision, recall, and F1 score.

        Calculates evaluation metrics by comparing true and predicted annotations.
        Supports both micro-averaging (aggregate all predictions) and macro-averaging
        (average per-class metrics).

        Args:
            y_true: List of ground truth annotations in flattened format.
                Each annotation is [label, span] where span is tuple of positions.
            y_pred: List of predicted annotations in flattened format.
                Each annotation is [label, span] where span is tuple of positions.
            average: Averaging strategy. Defaults to "micro".
                - "micro": Aggregate TP, FP, FN across all classes
                - Other values: Per-class metrics (requires additional logic)

        Returns:
            Dictionary containing:
            - 'precision': Precision score (float between 0 and 1)
            - 'recall': Recall score (float between 0 and 1)
            - 'f_score': F1 score (float between 0 and 1)

        Note:
            The function handles division by zero with warnings through the
            _prf_divide utility function.
        """
        y_true, y_pred = flatten_for_eval(y_true, y_pred)
        pred_sum, tp_sum, true_sum, _ = extract_tp_actual_correct(y_true, y_pred)

        if average == "micro":
            tp_sum = np.array([tp_sum.sum()])
            pred_sum = np.array([pred_sum.sum()])
            true_sum = np.array([true_sum.sum()])

        precision = _prf_divide(
            numerator=tp_sum,
            denominator=pred_sum,
            metric="precision",
            modifier="predicted",
            average=average,
            warn_for=["precision", "recall", "f-score"],
            zero_division="warn",
        )
        recall = _prf_divide(
            numerator=tp_sum,
            denominator=true_sum,
            metric="recall",
            modifier="true",
            average=average,
            warn_for=["precision", "recall", "f-score"],
            zero_division="warn",
        )

        denominator = precision + recall
        denominator[denominator == 0.0] = 1
        f_score = 2 * (precision * recall) / denominator

        return {"precision": precision[0], "recall": recall[0], "f_score": f_score[0]}

    @abstractmethod
    def transform_data(self):
        """Transform task-specific data into evaluation format.

        Abstract method that must be implemented by subclasses to convert
        their specific annotation formats into the standard format expected
        by compute_prf().

        Returns:
            Tuple of (transformed_true, transformed_pred) where each is a list
            of annotations in the format: [label, span_tuple]

        Raises:
            NotImplementedError: If called on the base class directly.
        """
        pass

    @torch.no_grad()
    def evaluate(self):
        """Evaluate predictions against ground truth.

        Transforms data using transform_data() and computes precision, recall,
        and F1 score using micro-averaging.

        Returns:
            Tuple of (output_str, f1) where:
            - output_str: Formatted string with P, R, F1 percentages
            - f1: F1 score as a float

        Note:
            This method disables gradient computation with @torch.no_grad()
            for efficiency during evaluation.
        """
        all_true_typed, all_outs_typed = self.transform_data()
        precision, recall, f1 = self.compute_prf(all_true_typed, all_outs_typed).values()
        output_str = f"P: {precision:.2%}\tR: {recall:.2%}\tF1: {f1:.2%}\n"
        return output_str, f1


class BaseNEREvaluator(BaseEvaluator):
    """Evaluator for Named Entity Recognition tasks.

    Evaluates NER predictions by comparing predicted entity spans and types
    against ground truth annotations. An entity is considered correct only
    if both the span boundaries and entity type match exactly.
    """

    def get_ground_truth(self, ents):
        """Extract ground truth entities in evaluation format.

        Args:
            ents: List of ground truth entity tuples in format (start, end, label)
                where start and end are word-level indices.

        Returns:
            List of entities in format [[label, (start, end)], ...] suitable
            for evaluation.
        """
        all_ents = []
        for s, e, lab in ents:
            all_ents.append([lab, (s, e)])
        return all_ents

    def get_predictions(self, ents):
        """Extract predicted entities in evaluation format.

        Args:
            ents: List of predicted entity tuples in format (start, end, label)
                where start and end are word-level indices.

        Returns:
            List of entities in format [[label, (start, end)], ...] suitable
            for evaluation.
        """
        all_ents = []
        for ent in ents:
            all_ents.append([ent[2], (ent[0], ent[1])])
        return all_ents

    def transform_data(self):
        """Transform NER data into evaluation format.

        Converts both ground truth and predicted entities from their original
        format into the standardized format required by compute_prf().

        Returns:
            Tuple of (all_true_ent, all_outs_ent) where:
            - all_true_ent: List of ground truth entity lists, one per sample
            - all_outs_ent: List of predicted entity lists, one per sample
            Each entity is in format [label, (start, end)]
        """
        all_true_ent = []
        all_outs_ent = []
        for i, j in zip(self.all_true, self.all_outs):
            e = self.get_ground_truth(i)
            all_true_ent.append(e)
            e = self.get_predictions(j)
            all_outs_ent.append(e)
        return all_true_ent, all_outs_ent


class BaseRelexEvaluator(BaseEvaluator):
    """Evaluator for Relation Extraction tasks.

    Evaluates relation extraction predictions by comparing predicted relations
    (head entity, tail entity, relation type) against ground truth. A relation
    is considered correct only if both entity spans and the relation type match
    exactly.

    Note:
        The input format expects entity indices rather than entity spans directly.
        Entity spans are looked up from the entity list using these indices.
    """
    def get_ground_truth(self, ents, rels):
        """Extract ground truth relations in evaluation format.

        Args:
            ents: List of entity tuples in format (start, end, label).
            rels: List of relation tuples in format (head_idx, tail_idx, rel_label)
                where head_idx and tail_idx are indices into the ents list.

        Returns:
            List of relations in format [[rel_label, (h_start, h_end, t_start, t_end)], ...]
            where h_start, h_end are head entity boundaries and t_start, t_end
            are tail entity boundaries.
        """
        all_rels = []
        for h, t, lab in rels:
            h_ent = ents[h]
            t_ent = ents[t]
            all_rels.append([lab, (h_ent[0], h_ent[1], t_ent[0], t_ent[1])])
        return all_rels

    def get_predictions(self, ents, rels):
        """Extract predicted relations in evaluation format.

        Args:
            ents: List of entity tuples in format (start, end, label).
            rels: List of predicted relation tuples in format (head_idx, rel_label, tail_idx)
                where head_idx and tail_idx are indices into the ents list.

        Returns:
            List of relations in format [[rel_label, (h_start, h_end, t_start, t_end)], ...]
            where h_start, h_end are head entity boundaries and t_start, t_end
            are tail entity boundaries.

        Note:
            The order of elements in predicted relations is (head_idx, rel_label, tail_idx),
            which differs from ground truth format (head_idx, tail_idx, rel_label).
        """
        all_rels = []
        for rel in rels:
            h = rel[0]
            lab = rel[1]
            t = rel[2]
            h_ent = ents[h]
            t_ent = ents[t]
            all_rels.append([lab, (h_ent[0], h_ent[1], t_ent[0], t_ent[1])])
        return all_rels

    def transform_data(self):
        """Transform relation extraction data into evaluation format.

        Converts both ground truth and predicted relations from their original
        format into the standardized format required by compute_prf().

        Returns:
            Tuple of (all_true_rel, all_outs_rel) where:
            - all_true_rel: List of ground truth relation lists, one per sample
            - all_outs_rel: List of predicted relation lists, one per sample
            Each relation is in format [rel_label, (h_start, h_end, t_start, t_end)]

        Note:
            The input format (self.all_true and self.all_outs) is expected to
            contain tuples of (entities, relations) for each sample.
        """
        all_true_rel = []
        all_outs_rel = []
        for true_item, pred_item in zip(self.all_true, self.all_outs):
            true_ent, true_rel = true_item
            pred_ent, pred_rel = pred_item
            e = self.get_ground_truth(true_ent, true_rel)
            all_true_rel.append(e)
            e = self.get_predictions(pred_ent, pred_rel)
            all_outs_rel.append(e)
        return all_true_rel, all_outs_rel
