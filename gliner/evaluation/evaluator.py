import warnings
from collections import defaultdict
from typing import Union, List, Literal

import numpy as np
import torch


class UndefinedMetricWarning(UserWarning):
    pass


def _prf_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
    metric: Literal["precision", "recall", "f-score"],
    modifier: str,
    average: str,
    warn_for: List[str],
    zero_division: Union[str, int] = "warn",
) -> np.ndarray:
    """Performs division and handles divide-by-zero with warnings."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.true_divide(numerator, denominator)
        result[denominator == 0] = 0.0 if zero_division in ["warn", 0] else 1.0

    if denominator == 0 and zero_division == "warn" and metric in warn_for:
        msg_start = f"{metric.title()}"
        if "f-score" in warn_for:
            msg_start += " and F-score" if metric in warn_for else "F-score"
        msg_start += " are" if "f-score" in warn_for else " is"
        _warn_prf(
            average=average,
            modifier=modifier,
            msg_start=msg_start,
            result_size=len(result),
        )

    return result


def _warn_prf(average: str, modifier: str, msg_start: str, result_size: int):
    axis0, axis1 = ("label", "sample") if average == "samples" else ("sample", "label")
    if result_size == 1:
        msg = f"{msg_start} ill-defined and being set to 0.0 due to no {modifier} {axis0}."  # noqa: E501
    else:
        msg = f"{msg_start} ill-defined and being set to 0.0 in {axis1}s with no {modifier} {axis0}s."  # noqa: E501
    msg += " Use `zero_division` parameter to control this behavior."
    warnings.warn(msg, UndefinedMetricWarning, stacklevel=3)


def extract_tp_actual_correct(y_true, y_pred):
    entities_true = defaultdict(set)
    entities_pred = defaultdict(set)

    for type_name, (start, end), idx in y_true:
        entities_true[type_name].add((start, end, idx))
    for type_name, (start, end), idx in y_pred:
        entities_pred[type_name].add((start, end, idx))

    target_names = sorted(set(entities_true.keys()) | set(entities_pred.keys()))

    tp_sum = np.array([], dtype=np.int32)
    pred_sum = np.array([], dtype=np.int32)
    true_sum = np.array([], dtype=np.int32)
    for type_name in target_names:
        entities_true_type = entities_true.get(type_name, set())
        entities_pred_type = entities_pred.get(type_name, set())
        tp_sum = np.append(tp_sum, len(entities_true_type & entities_pred_type))
        pred_sum = np.append(pred_sum, len(entities_pred_type))
        true_sum = np.append(true_sum, len(entities_true_type))

    return pred_sum, tp_sum, true_sum, target_names


def flatten_for_eval(y_true, y_pred):
    all_true = []
    all_pred = []

    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        all_true.extend([t + [i] for t in true])
        all_pred.extend([p + [i] for p in pred])

    return all_true, all_pred


def compute_prf(y_true, y_pred, average="micro"):
    y_true, y_pred = flatten_for_eval(y_true, y_pred)

    pred_sum, tp_sum, true_sum, target_names = extract_tp_actual_correct(y_true, y_pred)

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


class Evaluator:
    def __init__(self, all_true, all_outs):
        self.all_true = all_true
        self.all_outs = all_outs

    def get_entities_fr(self, ents):
        all_ents = []
        for s, e, lab in ents:
            all_ents.append([lab, (s, e)])
        return all_ents

    def get_entities_pr(self, ents):
        all_ents = []
        for s, e, lab, _ in ents:
            all_ents.append([lab, (s, e)])
        return all_ents

    def transform_data(self):
        all_true_ent = []
        all_outs_ent = []
        for i, j in zip(self.all_true, self.all_outs):
            e = self.get_entities_fr(i)
            all_true_ent.append(e)
            e = self.get_entities_pr(j)
            all_outs_ent.append(e)
        return all_true_ent, all_outs_ent

    @torch.no_grad()
    def evaluate(self):
        all_true_typed, all_outs_typed = self.transform_data()
        precision, recall, f1 = compute_prf(all_true_typed, all_outs_typed).values()
        output_str = f"P: {precision:.2%}\tR: {recall:.2%}\tF1: {f1:.2%}\n"
        return output_str, f1


def is_nested(idx1, idx2):
    # Return True if idx2 is nested inside idx1 or vice versa
    return (idx1[0] <= idx2[0] and idx1[1] >= idx2[1]) or (
        idx2[0] <= idx1[0] and idx2[1] >= idx1[1]
    )


def has_overlapping(idx1, idx2, multi_label=False):
    # Check for any overlap between two spans
    if idx1[:2] == idx2[:2]:  # Exact same boundaries can be considered as overlapping
        return not multi_label
    if idx1[0] > idx2[1] or idx2[0] > idx1[1]:
        return False
    return True


def has_overlapping_nested(idx1, idx2, multi_label=False):
    # Return True if idx1 and idx2 overlap, but neither is nested inside the other
    if idx1[:2] == idx2[:2]:  # Exact same boundaries, not considering labels here
        return not multi_label
    if (idx1[0] > idx2[1] or idx2[0] > idx1[1]) or is_nested(idx1, idx2):
        return False
    return True


from functools import partial


def greedy_search(spans, flat_ner=True, multi_label=False):  # start, end, class, score
    if flat_ner:
        has_ov = partial(has_overlapping, multi_label=multi_label)
    else:
        has_ov = partial(has_overlapping_nested, multi_label=multi_label)

    new_list = []
    span_prob = sorted(spans, key=lambda x: -x[-1])

    for i in range(len(spans)):
        b = span_prob[i]
        flag = False
        for new in new_list:
            if has_ov(b[:-1], new):
                flag = True
                break
        if not flag:
            new_list.append(b)

    new_list = sorted(new_list, key=lambda x: x[0])
    return new_list
