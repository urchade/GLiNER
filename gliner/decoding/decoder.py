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
    def decode(
        self,
        tokens,
        id_to_classes,
        model_output,
        *,
        flat_ner: bool = False,
        threshold: float = 0.5,
        multi_label: bool = False,
        rel_id_to_classes=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        tokens : List[List[str]]
        id_to_classes : Dict[int, str] | List[Dict[int, str]]
            Maps *entity* class-ids (1-based, as 0 is “O”) to names.
            May be a single dict (shared across batch) or a list (one per sample).
        model_output : GLiNERModelOutput
        flat_ner / threshold / multi_label :  see earlier code
        rel_id_to_classes : Dict[int, str] | List[Dict[int, str]] | None
            Same idea but for *relation* class-ids (again 1-based).
            If None, relation decoding is skipped.
        """
        # ---------- entity spans ----------
        probs = torch.sigmoid(model_output.logits)          # (B, L, K, C)
        spans, ent_prob = [], []                            # entities per sample

        for i, _ in enumerate(tokens):
            probs_i = probs[i]
            id_to_class_i = id_to_classes[i] if isinstance(id_to_classes, list) else id_to_classes

            wh_i = [t.tolist() for t in torch.where(probs_i > threshold)]
            span_i = []
            for s, k, c in zip(*wh_i):
                if s + k < len(tokens[i]):
                    span_i.append((s, s + k, id_to_class_i[c + 1], probs_i[s, k, c].item()))

            span_i = self.greedy_search(span_i, flat_ner, multi_label=multi_label)
            spans.append(span_i)

        # ---------- relations ----------
        relations = [[] for _ in range(len(tokens))]        # default empty lists
        if (
            rel_id_to_classes is not None
            and model_output.rel_idx is not None
            and model_output.rel_logits is not None
        ):
            rel_idx    = model_output.rel_idx          # (B, N, 2)
            rel_logits = model_output.rel_logits       # (B, N, C_rel)
            rel_mask   = (
                model_output.rel_mask
                if model_output.rel_mask is not None
                else torch.ones(rel_idx[..., 0].shape, dtype=torch.bool, device=rel_idx.device)
            )

            rel_probs  = torch.sigmoid(rel_logits)

            for i in range(len(tokens)):
                rel_id_to_class_i = (
                    rel_id_to_classes[i] if isinstance(rel_id_to_classes, list) else rel_id_to_classes
                )

                for j in range(rel_idx.size(1)):
                    if not rel_mask[i, j]:
                        continue

                    src = rel_idx[i, j, 0].item()
                    tgt = rel_idx[i, j, 1].item()
                    if src < 0 or tgt < 0:
                        continue
                    # if either span was removed by greedy search, skip
                    if src >= len(spans[i]) or tgt >= len(spans[i]):
                        continue

                    # C_rel may include the "no-relation" slot at 0
                    for c, p in enumerate(rel_probs[i, j]):
                        prob = p.item()
                        if prob <= threshold:
                            continue
                        if (c + 1) not in rel_id_to_class_i:
                            continue
                        rel_label = rel_id_to_class_i[c + 1]
                        relations[i].append((src, rel_label, tgt, prob))

        return spans, relations


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

    def decode(self, tokens, id_to_classes, model_output, flat_ner=False, threshold=0.5, multi_label=False, **kwargs):
        model_output = model_output.permute(3, 0, 1, 2)
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
        return spans, None