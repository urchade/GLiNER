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
    
    def update_id_to_classes(self, id_to_classes, gen_labels, batch_size):
        if self.config.labels_decoder is not None:
            if self.config.decoder_mode == 'prompt':
                new_id_to_classes = []
                cursor = 0
                for i in range(batch_size):
                    original = id_to_classes[i] if isinstance(id_to_classes, list) else id_to_classes
                    k = len(original)                    # how many labels belong to this example
                    mapping = {idx + 1: gen_labels[cursor + idx] for idx in range(k)}
                    new_id_to_classes.append(mapping)
                    cursor += k
                id_to_classes = new_id_to_classes
        return id_to_classes
    
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
    def decode(self, tokens, id_to_classes, model_output, flat_ner=False, 
                        threshold=0.5, multi_label=False, gen_labels = None):
        batch_size = len(tokens)
        id_to_classes = self.update_id_to_classes(id_to_classes, gen_labels, batch_size)
        probs = torch.sigmoid(model_output)
        spans = []
        cursor = 0
        for i, _ in enumerate(tokens):
            probs_i = probs[i]
            
            # Support for id_to_classes being a list of dictionaries
            id_to_class_i = id_to_classes[i] if isinstance(id_to_classes, list) else id_to_classes
            
            wh_i = [i.tolist() for i in torch.where(probs_i > threshold)]
            span_i = []
            for s, k, c in zip(*wh_i):
                if s + k < len(tokens[i]):
                    if self.config.decoder_mode == 'span':
                        ent_type = gen_labels[cursor]
                    else:
                        ent_type = id_to_class_i[c + 1]
                    span_i.append((s, s + k, ent_type, probs_i[s, k, c].item()))

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
                    # Get the start and end scores for this span
                    start_score = start_i[st, cls_st]
                    end_score = end_i[ed, cls_st]
                    # Concatenate the inside scores with start and end scores
                    combined = torch.cat([ins, start_score.unsqueeze(0), end_score.unsqueeze(0)])
                    # The span score is the minimum value among these scores
                    spn_score = combined.min().item()
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