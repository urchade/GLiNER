from typing import Optional
from abc import ABC, abstractmethod
from functools import partial
import torch

from .utils import has_overlapping, has_overlapping_nested

class BaseDecoder(ABC):
    def __init__(self, config):
        self.config = config
    
    @abstractmethod
    def decode(self):
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

            wh_i = [i.tolist() for i in torch.where(probs_i > threshold)]
            span_i = []
            for s, k, c in zip(*wh_i):
                if s + k < len(tokens[i]):
                    span_i.append((s, s + k, id_to_classes[c + 1], probs_i[s, k, c].item()))

            span_i = self.greedy_search(span_i, flat_ner, multi_label=multi_label)
            spans.append(span_i)
        return spans
    
class TokenDecoder(BaseDecoder):
    def decode(self, tokens, id_to_classes, model_output, flat_ner=False, threshold=0.5, multi_label=False):
        scores_start, scores_end, scores_inside = model_output
        # shape: (batch_size, seq_len, num_classes)
        spans = []
        for i, _ in enumerate(tokens):
            start_i = torch.sigmoid(scores_start[i])
            end_i = torch.sigmoid(scores_end[i])
            scores_inside_i = torch.sigmoid(scores_inside[i])  # (seq_len, num_classes)

            start_idx = [k.tolist() for k in torch.where(start_i > threshold)]
            end_idx = [k.tolist() for k in torch.where(end_i > threshold)]

            span_i = []
            for st, cls_st in zip(*start_idx):
                for ed, cls_ed in zip(*end_idx):
                    if ed >= st and cls_st == cls_ed:
                        ins = scores_inside_i[st:ed + 1, cls_st]
                        # remove spans with low confidence (important for nested NER)
                        if (ins < threshold).any():
                            continue
                        span_i.append(
                            (st, ed, id_to_classes[cls_st + 1], ins.mean().item())
                        )
            span_i = self.greedy_search(span_i, flat_ner, multi_label=multi_label)
            spans.append(span_i)
        return spans