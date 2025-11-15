from abc import ABC, abstractmethod
import numpy as np
import torch

from .utils import flatten_for_eval, extract_tp_actual_correct, _prf_divide

class BaseEvaluator(ABC):
    def __init__(self, all_true, all_outs):
        self.all_true = all_true
        self.all_outs = all_outs

    @staticmethod
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
    
    @abstractmethod
    def transform_data(self):
        pass

    @torch.no_grad()
    def evaluate(self):
        all_true_typed, all_outs_typed = self.transform_data()
        precision, recall, f1 = self.compute_prf(all_true_typed, all_outs_typed).values()
        output_str = f"P: {precision:.2%}\tR: {recall:.2%}\tF1: {f1:.2%}\n"
        return output_str, f1
    

class BaseNEREvaluator(BaseEvaluator):
    def get_ground_truth(self, ents):
        all_ents = []
        for s, e, lab in ents:
            all_ents.append([lab, (s, e)])
        return all_ents

    def get_predictions(self, ents):
        all_ents = []
        for ent in ents:
            all_ents.append([ent[2], (ent[0], ent[1])])
        return all_ents

    def transform_data(self):
        all_true_ent = []
        all_outs_ent = []
        for i, j in zip(self.all_true, self.all_outs):
            e = self.get_ground_truth(i)
            all_true_ent.append(e)
            e = self.get_predictions(j)
            all_outs_ent.append(e)
        return all_true_ent, all_outs_ent
    

class BaseRelexEvaluator(BaseEvaluator):
    def get_ground_truth(self, ents, rels):
        all_rels = []
        for h, t, lab in rels:
            h_ent = ents[h]
            t_ent = ents[t]
            all_rels.append([lab, (h_ent[0], h_ent[1], t_ent[0], t_ent[1])])
        return all_rels

    def get_predictions(self, ents, rels):
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
        all_true_rel = []
        all_outs_rel = []
        for i, j in zip(self.all_true, self.all_outs):
            e = self.get_ground_truth(i)
            all_true_rel.append(e)
            e = self.get_predictions(j)
            all_outs_rel.append(e)
        return all_true_rel, all_outs_rel