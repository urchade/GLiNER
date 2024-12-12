from typing import Optional
from abc import ABC, abstractmethod
from functools import partial
import torch

from .utils import has_overlapping, has_overlapping_nested


class BaseDecoder(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def decode(self, *args, **kwargs):
        pass

    def greedy_search(self, spans, flat_ner=True, multi_label=False):
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


class SpanDecoder(BaseDecoder):
    def decode(self, tokens, id_to_classes, model_output, flat_ner=False, threshold=0.5, multi_label=False):
        probs = torch.sigmoid(model_output)
        spans = []
        for i, _ in enumerate(tokens):
            probs_i = probs[i]
            
            # Support for id_to_classes being a list of dictionaries
            id_to_class_i = id_to_classes[i] if isinstance(id_to_classes, list) else id_to_classes
            
            wh_i = [i.tolist() for i in torch.where(probs_i > threshold)]
            span_i = []
            for s, k, c in zip(*wh_i):
                if s + k < len(tokens[i]):
                    span_i.append((s, s + k, id_to_class_i[c + 1], probs_i[s, k, c].item()))

            span_i = self.greedy_search(span_i, flat_ner, multi_label=multi_label)
            spans.append(span_i)
        return spans


class TokenDecoder(BaseDecoder):
    def get_indices_above_threshold(self, scores, threshold):
        scores = torch.sigmoid(scores)
        return [k.tolist() for k in torch.where(scores > threshold)]

    def calculate_span_score(self, start_idx, end_idx, scores_inside_i, start_i, end_i, id_to_classes, threshold):
        span_i = []
        for st, cls_st in zip(*start_idx):
            for ed, cls_ed in zip(*end_idx):
                if ed >= st and cls_st == cls_ed:
                    ins = scores_inside_i[st:ed + 1, cls_st]
                    if (ins < threshold).any():
                        continue
                    spn_score = ins.mean().item()
                    span_i.append((st, ed, id_to_classes[cls_st + 1], spn_score))
        return span_i

    def decode(self, tokens, id_to_classes, model_output, flat_ner=False, threshold=0.5, multi_label=False):
        scores_start, scores_end, scores_inside = model_output
        spans = []
        for i, _ in enumerate(tokens):
            id_to_class_i = id_to_classes[i] if isinstance(id_to_classes, list) else id_to_classes
            span_scores = self.calculate_span_score(
                self.get_indices_above_threshold(scores_start[i], threshold),
                self.get_indices_above_threshold(scores_end[i], threshold),
                torch.sigmoid(scores_inside[i]),
                torch.sigmoid(scores_start[i]),
                torch.sigmoid(scores_end[i]),
                id_to_class_i,
                threshold
            )
            span_i = self.greedy_search(span_scores, flat_ner, multi_label)
            spans.append(span_i)
        return spans