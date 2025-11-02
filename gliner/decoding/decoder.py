from typing import Optional, Union, List, Dict
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
    
    def update_id_to_classes(self, id_to_classes: Union[Dict, List[Dict]], 
                            gen_labels: Optional[List[str]], 
                            batch_size: int) -> Union[Dict, List[Dict]]:
        """
        Update id_to_classes mapping with generated labels when using labels_decoder.
        
        For prompt decoder mode, maps generated labels back to class IDs.
        
        Parameters
        ----------
        id_to_classes : dict or list[dict]
            Original mapping from class IDs to class names
        gen_labels : list[str] or None
            Generated labels from the decoder
        batch_size : int
            Number of samples in the batch
            
        Returns
        -------
        Updated id_to_classes mapping
        """
        if self.config.labels_decoder is not None and gen_labels is not None:
            if self.config.decoder_mode == 'prompt':
                new_id_to_classes = []
                cursor = 0
                for i in range(batch_size):
                    original = id_to_classes[i] if isinstance(id_to_classes, list) else id_to_classes
                    k = len(original)  # Number of labels for this example
                    # Map each class ID to its generated label
                    mapping = {idx + 1: gen_labels[cursor + idx] for idx in range(k)}
                    new_id_to_classes.append(mapping)
                    cursor += k
                id_to_classes = new_id_to_classes
        return id_to_classes
    
    def greedy_search(self, spans: List[tuple], flat_ner: bool = True, 
                     multi_label: bool = False) -> List[tuple]:
        """
        Perform greedy search to remove overlapping spans.
        
        Parameters
        ----------
        spans : list[tuple]
            List of span tuples (start, end, label, gen_label, score)
        flat_ner : bool
            Whether to use flat NER (no nesting) or nested NER
        multi_label : bool
            Whether to allow multiple labels for the same span
            
        Returns
        -------
        Filtered list of non-overlapping spans
        """
        if flat_ner:
            has_ov = partial(has_overlapping, multi_label=multi_label)
        else:
            has_ov = partial(has_overlapping_nested, multi_label=multi_label)

        new_list = []
        # Sort by probability (descending)
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

        # Sort by start position
        new_list = sorted(new_list, key=lambda x: x[0])
        return new_list


class BaseSpanDecoder(BaseDecoder):
    """Base class for span-based decoders with common decoding logic."""
    
    def _get_id_to_class_for_sample(self, id_to_classes: Union[Dict[int, str], List[Dict[int, str]]], 
                                    sample_idx: int) -> Dict[int, str]:
        """
        Get id_to_classes mapping for a specific sample.
        
        Parameters
        ----------
        id_to_classes : dict or list[dict]
            Either a single mapping or per-sample mappings
        sample_idx : int
            Index of the sample in the batch
            
        Returns
        -------
        dict mapping class IDs to class names for this sample
        """
        if isinstance(id_to_classes, list):
            return id_to_classes[sample_idx]
        return id_to_classes
    
    def _find_candidate_spans(self, probs: torch.Tensor, threshold: float):
        """
        Find all span candidates above threshold.
        
        Parameters
        ----------
        probs : torch.Tensor (L, K, C)
            Probabilities for one sample
        threshold : float
            Confidence threshold
            
        Returns
        -------
        Tuple of (start_indices, width_indices, class_indices)
        """
        return torch.where(probs > threshold)
    
    def _is_valid_span(self, start: int, width: int, tokens: List[str]) -> bool:
        """
        Check if a span is valid (doesn't exceed sentence length).
        
        Parameters
        ----------
        start : int
            Start position
        width : int
            Span width (0-indexed)
        tokens : list[str]
            Tokens for this sample
            
        Returns
        -------
        bool indicating if span is valid
        """
        end = start + width + 1
        return end <= len(tokens)
    
    def _build_span_tuple(self, start: int, width: int, class_idx: int, 
                         flat_idx: int, score: float, 
                         id_to_class: Dict[int, str], 
                         span_label_map: Dict[int, List[str]]) -> tuple:
        """
        Build a span tuple. Should be overridden by subclasses.
        
        Parameters
        ----------
        start : int
            Start position
        width : int
            Width index (0-indexed)
        class_idx : int
            Class index
        flat_idx : int
            Flat span index (start * K + width)
        score : float
            Confidence score
        id_to_class : dict
            Mapping from class IDs to class names
        span_label_map : dict
            Mapping from flat indices to generated labels
            
        Returns
        -------
        tuple representing the span
        """
        raise NotImplementedError("Subclasses must implement _build_span_tuple")
    
    def _prepare_span_label_map(self, sel_idx: Optional[torch.LongTensor],
                                gen_labels: Optional[List[str]],
                                num_gen_sequences: int,
                                batch_idx: int) -> Dict[int, List[str]]:
        """
        Prepare span label mapping for a specific batch element.
        Should be overridden by subclasses that use generative labels.
        
        Parameters
        ----------
        sel_idx : torch.LongTensor (B, M) or None
            Selected span indices
        gen_labels : list[str] or None
            Generated labels
        num_gen_sequences : int
            Number of sequences per span
        batch_idx : int
            Current batch index
            
        Returns
        -------
        dict mapping flat span indices to labels (empty dict by default)
        """
        return {}
    
    def decode(
        self,
        tokens: List[List[str]],
        id_to_classes: Union[Dict[int, str], List[Dict[int, str]]],
        model_output: torch.Tensor,
        flat_ner: bool = False,
        threshold: float = 0.5,
        multi_label: bool = False,
        **kwargs
    ) -> List[List[tuple]]:
        """
        Decode model output to extract named entity spans.
        
        Parameters
        ----------
        tokens : list[list[str]]
            Tokenized input text for each sample in the batch
        id_to_classes : dict or list[dict]
            Mapping from class IDs to class names
        model_output : torch.Tensor (B, L, K, C)
            Raw logits from the model
        flat_ner : bool
            Whether to enforce non-overlapping spans
        threshold : float
            Confidence threshold for span predictions
        multi_label : bool
            Whether to allow multiple labels per span
        **kwargs : additional arguments for subclasses
            
        Returns
        -------
        list[list[tuple]]
            For each sample, list of span tuples
        """
        B, L, K, C = model_output.shape
        probs = torch.sigmoid(model_output)
        
        # Decode spans for each sample in the batch
        spans = []
        for i in range(B):
            probs_i = probs[i]
            id_to_class_i = self._get_id_to_class_for_sample(id_to_classes, i)
            
            # Prepare span label map (empty for base decoder)
            span_label_map = self._prepare_span_label_map(
                kwargs.get('sel_idx'),
                kwargs.get('gen_labels'),
                kwargs.get('num_gen_sequences', 1),
                i
            )
            
            span_i = []
            
            # Find all spans above threshold
            s_idx, k_idx, c_idx = self._find_candidate_spans(probs_i, threshold)
            
            for s, k, c in zip(s_idx.tolist(), k_idx.tolist(), c_idx.tolist()):
                # Skip if span exceeds sentence length
                if not self._is_valid_span(s, k, tokens[i]):
                    continue
                
                # Calculate flat index (matches encoder's indexing)
                flat_idx = s * K + k
                score = probs_i[s, k, c].item()
                
                # Build span tuple (implementation varies by subclass)
                span_tuple = self._build_span_tuple(
                    s, k, c, flat_idx, score, id_to_class_i, span_label_map
                )
                span_i.append(span_tuple)
            
            # Remove overlapping spans using greedy search
            span_i = self.greedy_search(span_i, flat_ner, multi_label=multi_label)
            spans.append(span_i)
        
        return spans


class SpanDecoder(BaseSpanDecoder):
    """Simple span decoder without generative labels."""
    
    def _build_span_tuple(self, start: int, width: int, class_idx: int,
                         flat_idx: int, score: float,
                         id_to_class: Dict[int, str],
                         span_label_map: Dict[int, List[str]]) -> tuple:
        """
        Build span tuple without generative labels.
        
        Returns
        -------
        tuple: (start, end, entity_type, score)
        """
        ent_type = id_to_class[class_idx + 1]  # +1 because 0 is <pad>
        return (start, start + width, ent_type, score)


class SpanGenerativeDecoder(BaseSpanDecoder):
    """Span decoder with generative label support."""
    
    def update_id_to_classes(self, id_to_classes: Union[Dict, List[Dict]], 
                            gen_labels: Optional[List[str]], 
                            batch_size: int) -> Union[Dict, List[Dict]]:
        """
        Update id_to_classes mapping with generated labels.
        
        For prompt decoder mode, maps generated labels back to class IDs.
        
        Parameters
        ----------
        id_to_classes : dict or list[dict]
            Original mapping from class IDs to class names
        gen_labels : list[str] or None
            Generated labels from the decoder
        batch_size : int
            Number of samples in the batch
            
        Returns
        -------
        Updated id_to_classes mapping
        """
        if self.config.labels_decoder is not None and gen_labels is not None:
            if self.config.decoder_mode == 'prompt':
                new_id_to_classes = []
                cursor = 0
                for i in range(batch_size):
                    original = (id_to_classes[i] if isinstance(id_to_classes, list) 
                              else id_to_classes)
                    k = len(original)  # Number of labels for this example
                    # Map each class ID to its generated label
                    mapping = {idx + 1: gen_labels[cursor + idx] for idx in range(k)}
                    new_id_to_classes.append(mapping)
                    cursor += k
                id_to_classes = new_id_to_classes
        return id_to_classes
    
    def _build_span_label_map(self, sel_idx: torch.LongTensor,
                              gen_labels: List[str],
                              num_gen_sequences: int,
                              batch_size: int) -> List[Dict[int, List[str]]]:
        """
        Build mapping from flat span indices to generated labels.
        
        For decoder_mode=='span', maps each valid span index to its generated label(s).
        
        Parameters
        ----------
        sel_idx : torch.LongTensor (B, M)
            Indices of selected spans. -1 indicates padding.
        gen_labels : list[str]
            All generated labels (flattened across batch)
        num_gen_sequences : int
            Number of label sequences generated per span
        batch_size : int
            Number of samples in the batch
            
        Returns
        -------
        list of dicts, one per batch element
            Each dict maps flat_span_idx -> list of generated labels
        """
        span_label_maps = [{} for _ in range(batch_size)]
        cursor = 0  # Tracks position in gen_labels
        
        for b in range(batch_size):
            valid_pos = (sel_idx[b] != -1)
            n = valid_pos.sum().item()  # Number of valid spans in this batch element
            
            if n > 0:
                # Get the flat indices of spans that were kept
                flat_indices = sel_idx[b, valid_pos].tolist()
                
                # Calculate the range of labels for this batch element
                start_index = cursor * num_gen_sequences
                end_index = start_index + n * num_gen_sequences
                span_labels = gen_labels[start_index:end_index]
                
                # Group labels: each span gets num_gen_sequences consecutive labels
                labels_b = [
                    span_labels[i * num_gen_sequences:(i + 1) * num_gen_sequences]
                    for i in range(n)
                ]
                
                # Create mapping from flat_index to labels
                span_label_maps[b] = dict(zip(flat_indices, labels_b))
                cursor += n
        
        return span_label_maps
    
    def _prepare_span_label_map(self, sel_idx: Optional[torch.LongTensor],
                                gen_labels: Optional[List[str]],
                                num_gen_sequences: int,
                                batch_idx: int) -> Dict[int, List[str]]:
        """
        Prepare span label mapping for this batch element.
        
        This caches the full mapping on first call and returns the appropriate
        batch element's mapping.
        """
        # Cache the full mapping if not already done
        if not hasattr(self, '_cached_span_label_maps'):
            if (self.config.decoder_mode == "span" and 
                sel_idx is not None and 
                gen_labels is not None):
                batch_size = sel_idx.shape[0]
                self._cached_span_label_maps = self._build_span_label_map(
                    sel_idx, gen_labels, num_gen_sequences, batch_size
                )
            else:
                self._cached_span_label_maps = None
        
        # Return the mapping for this batch element
        if self._cached_span_label_maps is not None:
            return self._cached_span_label_maps[batch_idx]
        return {}
    
    def _build_span_tuple(self, start: int, width: int, class_idx: int,
                         flat_idx: int, score: float,
                         id_to_class: Dict[int, str],
                         span_label_map: Dict[int, List[str]]) -> tuple:
        """
        Build span tuple with generative labels.
        
        Returns
        -------
        tuple: (start, end, entity_type, generated_entity_type, score)
        """
        ent_type = id_to_class[class_idx + 1]  # +1 because 0 is <pad>
        gen_ent_type = span_label_map.get(flat_idx, None)
        return (start, start + width, ent_type, gen_ent_type, score)
    
    def decode(self, *args, **kwargs) -> List[List[tuple]]:
        """
        Decode with generative labels support.
        
        Clears the cache before decoding to ensure fresh mappings.
        """
        # Clear cache before new decode call
        if hasattr(self, '_cached_span_label_maps'):
            delattr(self, '_cached_span_label_maps')
        
        return super().decode(*args, **kwargs)


class SpanRelexDecoder(BaseSpanDecoder):
    """Span decoder with relation extraction support."""
    
    def _build_span_tuple(self, start: int, width: int, class_idx: int,
                         flat_idx: int, score: float,
                         id_to_class: Dict[int, str],
                         span_label_map: Dict[int, List[str]]) -> tuple:
        """
        Build span tuple for relation extraction.
        
        Returns
        -------
        tuple: (start, end, entity_type, score)
        """
        ent_type = id_to_class[class_idx + 1]  # +1 because 0 is <pad>
        return (start, start + width, ent_type, score)
    
    def _decode_relations(
        self,
        model_output,
        spans: List[List[tuple]],
        rel_id_to_classes: Union[Dict[int, str], List[Dict[int, str]]],
        threshold: float,
        batch_size: int
    ) -> List[List[tuple]]:
        """
        Decode relations between entity spans.
        
        Parameters
        ----------
        model_output : GLiNERRelexOutput
            Model output containing relation predictions
        spans : list[list[tuple]]
            Decoded entity spans for each sample
        rel_id_to_classes : dict or list[dict]
            Mapping from relation class IDs to relation names
        threshold : float
            Confidence threshold for relations
        batch_size : int
            Number of samples in batch
            
        Returns
        -------
        list[list[tuple]]
            For each sample, list of (head_idx, relation_label, tail_idx, score)
        """
        relations = [[] for _ in range(batch_size)]
        
        # Check if relation outputs are available
        if not hasattr(model_output, 'rel_idx') or model_output.rel_idx is None:
            return relations
        if not hasattr(model_output, 'rel_logits') or model_output.rel_logits is None:
            return relations
        
        rel_idx = model_output.rel_idx          # (B, N, 2) - head/tail indices
        rel_logits = model_output.rel_logits    # (B, N, C_rel) - relation scores
        
        # Get or create relation mask
        if hasattr(model_output, 'rel_mask') and model_output.rel_mask is not None:
            rel_mask = model_output.rel_mask
        else:
            # Create default mask (all valid)
            rel_mask = torch.ones(
                rel_idx[..., 0].shape, 
                dtype=torch.bool, 
                device=rel_idx.device
            )
        
        rel_probs = torch.sigmoid(rel_logits)
        
        # Decode relations for each sample
        for i in range(batch_size):
            rel_id_to_class_i = (
                rel_id_to_classes[i] if isinstance(rel_id_to_classes, list)
                else rel_id_to_classes
            )
            
            # Process each potential relation
            for j in range(rel_idx.size(1)):
                # Skip if masked out
                if not rel_mask[i, j]:
                    continue
                
                head_idx = rel_idx[i, j, 0].item()
                tail_idx = rel_idx[i, j, 1].item()
                
                # Skip invalid indices
                if head_idx < 0 or tail_idx < 0:
                    continue
                
                # Skip if either span was removed by greedy search
                if head_idx >= len(spans[i]) or tail_idx >= len(spans[i]):
                    continue
                
                # Check each relation class
                for c, p in enumerate(rel_probs[i, j]):
                    prob = p.item()
                    
                    # Skip low confidence predictions
                    if prob <= threshold:
                        continue
                    
                    # Skip if class ID not in mapping
                    # (c + 1 because 0 may be "no-relation" or padding)
                    if (c + 1) not in rel_id_to_class_i:
                        continue
                    
                    rel_label = rel_id_to_class_i[c + 1]
                    
                    # Append relation: (head_idx, relation_label, tail_idx, score)
                    relations[i].append((head_idx, rel_label, tail_idx, prob))
        
        return relations
    
    def decode(
        self,
        tokens: List[List[str]],
        id_to_classes: Union[Dict[int, str], List[Dict[int, str]]],
        model_output,
        flat_ner: bool = False,
        threshold: float = 0.5,
        multi_label: bool = False,
        rel_id_to_classes: Optional[Union[Dict[int, str], List[Dict[int, str]]]] = None,
        **kwargs
    ) -> tuple:
        """
        Decode model output to extract entities and relations.
        
        Parameters
        ----------
        tokens : list[list[str]]
            Tokenized input text for each sample in the batch
        id_to_classes : dict or list[dict]
            Mapping from entity class IDs (1-based) to entity names.
            Can be a single dict (shared) or list (per-sample).
        model_output : GLiNERRelexOutput
            Model output containing both entity logits and relation predictions
        flat_ner : bool
            Whether to enforce non-overlapping entity spans
        threshold : float
            Confidence threshold for both entities and relations
        multi_label : bool
            Whether to allow multiple labels per entity span
        rel_id_to_classes : dict or list[dict] or None
            Mapping from relation class IDs (1-based) to relation names.
            If None, relation decoding is skipped.
        **kwargs : additional arguments
            
        Returns
        -------
        tuple: (spans, relations)
            spans : list[list[tuple]]
                For each sample, list of (start, end, entity_type, score)
            relations : list[list[tuple]]
                For each sample, list of (head_idx, relation_label, tail_idx, score)
        """
        # Decode entity spans using base class logic
        spans = super().decode(
            tokens=tokens,
            id_to_classes=id_to_classes,
            model_output=model_output.logits,  # Pass just the logits tensor
            flat_ner=flat_ner,
            threshold=threshold,
            multi_label=multi_label,
            **kwargs
        )
        
        # Decode relations if requested
        relations = [[] for _ in range(len(tokens))]  # Default: empty lists
        
        if rel_id_to_classes is not None:
            relations = self._decode_relations(
                model_output=model_output,
                spans=spans,
                rel_id_to_classes=rel_id_to_classes,
                threshold=threshold,
                batch_size=len(tokens)
            )
        
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
                    span_i.append((st, ed, id_to_classes[cls_st + 1], None, spn_score))
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
        return spans