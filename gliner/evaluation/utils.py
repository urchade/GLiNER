import warnings
from typing import List, Union, Literal
from collections import defaultdict

import numpy as np


class UndefinedMetricWarning(UserWarning):
    pass


def extract_tp_actual_correct(y_true, y_pred):
    elements_true = defaultdict(set)
    elements_pred = defaultdict(set)

    for type_name, el, idx in y_true:
        elements_true[type_name].add((el, idx))
    for type_name, el, idx in y_pred:
        elements_pred[type_name].add((el, idx))

    target_names = sorted(set(elements_true.keys()) | set(elements_pred.keys()))

    tp_sum = np.array([], dtype=np.int32)
    pred_sum = np.array([], dtype=np.int32)
    true_sum = np.array([], dtype=np.int32)
    for type_name in target_names:
        elements_true_type = elements_true.get(type_name, set())
        elements_pred_type = elements_pred.get(type_name, set())
        tp_sum = np.append(tp_sum, len(elements_true_type & elements_pred_type))
        pred_sum = np.append(pred_sum, len(elements_pred_type))
        true_sum = np.append(true_sum, len(elements_true_type))

    return pred_sum, tp_sum, true_sum, target_names


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
        msg = f"{msg_start} ill-defined and being set to 0.0 due to no {modifier} {axis0}."
    else:
        msg = f"{msg_start} ill-defined and being set to 0.0 in {axis1}s with no {modifier} {axis0}s."
    msg += " Use `zero_division` parameter to control this behavior."
    warnings.warn(msg, UndefinedMetricWarning, stacklevel=3)


def flatten_for_eval(y_true, y_pred):
    all_true = []
    all_pred = []

    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        all_true.extend([[*t, i] for t in true])
        all_pred.extend([[*p, i] for p in pred])

    return all_true, all_pred
