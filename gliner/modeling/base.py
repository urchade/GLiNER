from pathlib import Path
from typing import Optional, Tuple, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import warnings

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from transformers.utils import ModelOutput

from .encoder import Encoder, BiEncoder
from .layers import LstmSeq2SeqEncoder, CrossFuser, create_projection_layer
from .scorers import Scorer
from .loss_functions import focal_loss_with_logits
from .span_rep import SpanRepLayer
from .relations_layers import RelationsRepLayer
from .triples_layers import TriplesScoreLayer


@dataclass
class GLiNERModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    prompts_embedding: Optional[torch.FloatTensor] = None
    prompts_embedding_mask: Optional[torch.LongTensor] = None
    words_embedding: Optional[torch.FloatTensor] = None
    mask: Optional[torch.LongTensor] = None
    rel_idx: Optional[torch.LongTensor] = None
    rel_logits: Optional[torch.FloatTensor] = None
    rel_mask: Optional[torch.FloatTensor] = None


def extract_word_embeddings(token_embeds, words_mask, attention_mask,
                            batch_size, max_text_length, embed_dim, text_lengths):
    words_embedding = torch.zeros(
        batch_size, max_text_length, embed_dim, dtype=token_embeds.dtype, device=token_embeds.device
    )

    batch_indices, word_idx = torch.where(words_mask > 0)

    target_word_idx = words_mask[batch_indices, word_idx] - 1

    words_embedding[batch_indices, target_word_idx] = token_embeds[batch_indices, word_idx]

    aranged_word_idx = torch.arange(max_text_length,
                                    dtype=attention_mask.dtype,
                                    device=token_embeds.device).expand(batch_size, -1)
    mask = aranged_word_idx < text_lengths
    return words_embedding, mask

def extract_special_embeddings(token_embeds, input_ids, attention_mask, special_token_index, embed_special=True):
    batch_size, sequence_length, embed_dim = token_embeds.shape
    special_mask = input_ids == special_token_index
    num_special = torch.sum(special_mask, dim=-1, keepdim=True)
    max_num = num_special.max()
    aranged_idx = torch.arange(max_num, dtype=attention_mask.dtype, device=token_embeds.device).expand(batch_size, -1)
    batch_indices, target_idx = torch.where(aranged_idx < num_special)
    _, special_indices = torch.where(special_mask)
    if not embed_special:
        special_indices += 1
    embeddings = torch.zeros(batch_size, max_num, embed_dim, dtype=token_embeds.dtype, device=token_embeds.device)
    mask = (aranged_idx < num_special).to(attention_mask.dtype)
    embeddings[batch_indices, target_idx] = token_embeds[batch_indices, special_indices]
    return embeddings, mask


def extract_prompt_features_and_word_embeddings(config, token_embeds, input_ids, attention_mask,
                                                text_lengths, words_mask, embed_ent_token=True, **kwargs):
    prompts_embedding, prompts_embedding_mask = extract_special_embeddings(token_embeds, input_ids, attention_mask, config.class_token_index, embed_ent_token)
    batch_size, _, embed_dim = token_embeds.shape
    max_text_length = text_lengths.max()
    words_embedding, mask = extract_word_embeddings(token_embeds, words_mask, attention_mask,
                                                    batch_size, max_text_length, embed_dim, text_lengths)
    return prompts_embedding, prompts_embedding_mask, words_embedding, mask


def extract_rel_features(config, token_embeds, input_ids, attention_mask, embed_rel_token=True, **kwargs):
    return extract_special_embeddings(token_embeds, input_ids, attention_mask, config.rel_token_index, embed_rel_token)



def build_entity_pairs(
    adj: torch.Tensor,           # (B, E, E) –  scores / mask (diag is ignored)
    span_rep: torch.Tensor,      # (B, E, D) –  entity/span embeddings
    threshold: float = 0.5,      # keep pairs with score > threshold
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract the (head, tail) indices of entity pairs for which adj > threshold.

    Returns
    -------
    pair_idx  : (B, N, 2)  int64   – padded with -1
    pair_mask : (B, N)     bool    – 1 for real pair, 0 for pad
    head_rep  : (B, N, D)  same D  – embeddings of heads
    tail_rep  : (B, N, D)          – embeddings of tails
    """
    B, E, _ = adj.shape
    device   = adj.device
    rows, cols = torch.triu_indices(E, E, offset=1, device=device)  # ignore (i,i) and duplicates

    batch_pair_lists: list[torch.Tensor] = []
    for b in range(B):
        sel = adj[b, rows, cols] > threshold          # (E*(E-1)/2,)
        pairs = torch.stack([rows[sel], cols[sel]], dim=-1)  # (M_b, 2)
        batch_pair_lists.append(pairs)

    N = max(p.shape[0] for p in batch_pair_lists) if batch_pair_lists else 0
    if N == 0:                                         # nothing selected anywhere
        pair_idx  = torch.full((B, 1, 2), -1, dtype=torch.long, device=device)
        pair_mask = torch.zeros((B, 1), dtype=torch.bool, device=device)
        D         = span_rep.shape[-1]
        head_rep  = tail_rep = torch.zeros((B, 1, D), dtype=span_rep.dtype, device=device)
        return pair_idx, pair_mask, head_rep, tail_rep

    pair_idx  = torch.full((B, N, 2), -1, dtype=torch.long,  device=device)
    pair_mask = torch.zeros((B, N),    dtype=torch.bool,     device=device)

    for b, pairs in enumerate(batch_pair_lists):
        m = pairs.shape[0]
        pair_idx[b, :m]  = pairs
        pair_mask[b, :m] = True

    batch_idx = torch.arange(B, device=device).unsqueeze(1)                    # (B,1)
    head_rep  = span_rep[batch_idx, pair_idx[..., 0].clamp_min(0)]             # (B,N,D)
    tail_rep  = span_rep[batch_idx, pair_idx[..., 1].clamp_min(0)]             # (B,N,D)

    return pair_idx, pair_mask, head_rep, tail_rep

class BaseModel(ABC, nn.Module):
    def __init__(self, config, from_pretrained = False, cache_dir: Optional[Union[str, Path]] = None):
        super(BaseModel, self).__init__()
        self.config = config

        if not config.labels_encoder:
            self.token_rep_layer = Encoder(config, from_pretrained, cache_dir = cache_dir)
        else:
            self.token_rep_layer = BiEncoder(config, from_pretrained, cache_dir=cache_dir)
        if self.config.has_rnn:
            self.rnn = LstmSeq2SeqEncoder(config)

        if config.post_fusion_schema:            
            self.cross_fuser = CrossFuser(self.config.hidden_size,
                                          self.config.hidden_size,
                                          num_heads=self.token_rep_layer.bert_layer.model.config.num_attention_heads,
                                          num_layers=self.config.num_post_fusion_layers,
                                          dropout=config.dropout,
                                          schema=config.post_fusion_schema)

    def features_enhancement(self, text_embeds, labels_embeds, text_mask=None, labels_mask=None):
        labels_embeds, text_embeds = self.cross_fuser(labels_embeds, text_embeds, labels_mask, text_mask)
        return text_embeds, labels_embeds

    def _extract_prompt_features_and_word_embeddings(self, token_embeds, input_ids, attention_mask,
                                                     text_lengths, words_mask):
        prompts_embedding, prompts_embedding_mask, words_embedding, mask = extract_prompt_features_and_word_embeddings(
            self.config,
            token_embeds,
            input_ids,
            attention_mask,
            text_lengths,
            words_mask,
            self.config.embed_ent_token)
        return prompts_embedding, prompts_embedding_mask, words_embedding, mask

    def get_uni_representations(self,
                                input_ids: Optional[torch.FloatTensor] = None,
                                attention_mask: Optional[torch.LongTensor] = None,
                                text_lengths: Optional[torch.Tensor] = None,
                                words_mask: Optional[torch.LongTensor] = None,
                                **kwargs):

        token_embeds = self.token_rep_layer(input_ids, attention_mask, **kwargs)

        prompts_embedding, prompts_embedding_mask, words_embedding, mask = self._extract_prompt_features_and_word_embeddings(
            token_embeds, input_ids, attention_mask,
            text_lengths, words_mask)

        if self.config.has_rnn:
            words_embedding = self.rnn(words_embedding, mask)

        return token_embeds, prompts_embedding, prompts_embedding_mask, words_embedding, mask

    def get_bi_representations(self,
                               input_ids: Optional[torch.FloatTensor] = None,
                               attention_mask: Optional[torch.LongTensor] = None,
                               labels_embeds: Optional[torch.FloatTensor] = None,
                               labels_input_ids: Optional[torch.FloatTensor] = None,
                               labels_attention_mask: Optional[torch.LongTensor] = None,
                               text_lengths: Optional[torch.Tensor] = None,
                               words_mask: Optional[torch.LongTensor] = None,
                               **kwargs):
        if labels_embeds is not None:
            token_embeds = self.token_rep_layer.encode_text(input_ids, attention_mask, **kwargs)
        else:
            token_embeds, labels_embeds = self.token_rep_layer(input_ids, attention_mask,
                                                               labels_input_ids, labels_attention_mask,
                                                               **kwargs)
        batch_size, sequence_length, embed_dim = token_embeds.shape
        max_text_length = text_lengths.max()
        words_embedding, mask = extract_word_embeddings(token_embeds, words_mask, attention_mask,
                                                        batch_size, max_text_length, embed_dim, text_lengths)

        labels_embeds = labels_embeds.unsqueeze(0)
        labels_embeds = labels_embeds.expand(batch_size, -1, -1)
        labels_mask = torch.ones(labels_embeds.shape[:-1], dtype=attention_mask.dtype,
                                 device=attention_mask.device)

        labels_embeds = labels_embeds.to(words_embedding.dtype)
        if hasattr(self, "cross_fuser"):
            words_embedding, labels_embeds = self.features_enhancement(words_embedding, labels_embeds, text_mask=mask,
                                                                       labels_mask=labels_mask)

        return token_embeds, labels_embeds, labels_mask, words_embedding, mask

    def get_representations(self,
                            input_ids: Optional[torch.FloatTensor] = None,
                            attention_mask: Optional[torch.LongTensor] = None,
                            labels_embeddings: Optional[torch.FloatTensor] = None,
                            labels_input_ids: Optional[torch.FloatTensor] = None,
                            labels_attention_mask: Optional[torch.LongTensor] = None,
                            text_lengths: Optional[torch.Tensor] = None,
                            words_mask: Optional[torch.LongTensor] = None,
                            **kwargs):
        if self.config.labels_encoder:
            token_embeds, prompts_embedding, prompts_embedding_mask, words_embedding, mask = self.get_bi_representations(
                input_ids, attention_mask, labels_embeddings, labels_input_ids, labels_attention_mask,
                text_lengths, words_mask, **kwargs
            )
        else:
            token_embeds, prompts_embedding, prompts_embedding_mask, words_embedding, mask = self.get_uni_representations(
                input_ids, attention_mask, text_lengths, words_mask, **kwargs
            )
        return token_embeds, prompts_embedding, prompts_embedding_mask, words_embedding, mask
    
    @staticmethod
    def _fit_length(
        embedding: torch.Tensor,    # (B, L, D)
        mask:      torch.Tensor,    # (B, L)
        target_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make `embedding` & `mask` exactly `target_len` along dim=1.

        * pad with zeros  if L < target_len  
        * truncate       if L > target_len
        """
        B, L, D = embedding.shape
        if L == target_len:
            return embedding, mask

        if L < target_len:                              # → PAD
            pad_len = target_len - L
            pad_emb = torch.zeros(B, pad_len, D,
                                  dtype=embedding.dtype,
                                  device=embedding.device)
            pad_msk = torch.zeros(B, pad_len,
                                  dtype=mask.dtype,
                                  device=mask.device)
            embedding = torch.cat([embedding, pad_emb], dim=1)
            mask      = torch.cat([mask,      pad_msk], dim=1)

        else:                                           # → TRUNCATE
            embedding = embedding[:, :target_len]
            mask      = mask[:,      :target_len]

        return embedding, mask
    
    def select_target_embedding(
        self,
        representations: torch.FloatTensor = None,  # (B, N, D)
        rep_mask: torch.LongTensor = None           # (B, N)  – 0/1 or False/True
    ):
        B, N, D = representations.shape
        lengths = rep_mask.sum(dim=-1)                  # (B,)
        max_len = lengths.max().item()

        if max_len != N:
            target_rep = representations.new_zeros(B, max_len, D)
            target_mask = rep_mask.new_zeros(B, max_len)        # same dtype/device as rep_mask

            new_col_idx = (rep_mask.cumsum(dim=1) - 1)          # (B, N)
            keep = rep_mask.bool()                     

            batch_idx, old_col_idx = torch.where(keep)          # both (*) 1-D

            new_col_idx = new_col_idx[keep]                     # (K,)  – K = total # kept tokens

            target_rep[batch_idx, new_col_idx] = representations[batch_idx, old_col_idx]
            target_mask[batch_idx, new_col_idx] = 1
        else:
            target_rep = representations
            target_mask = rep_mask

        if hasattr(self, "_enc2dec_proj"):
            target_rep = self._enc2dec_proj(target_rep) 

        return target_rep, target_mask
    
    @abstractmethod
    def forward(self, x):
        pass

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor,
              alpha: float = -1., gamma: float = 0.0, label_smoothing: float = 0.0, negatives=1., masking="label"):

        # Compute the loss per element using the focal loss function
        all_losses = focal_loss_with_logits(logits, labels,
                                            alpha=alpha,
                                            gamma=gamma,
                                            label_smoothing=label_smoothing)

        # Create a mask of the same shape as labels:
        # For elements where labels==0, sample a Bernoulli random variable that is 1 with probability `negatives`
        # For elements where labels==1, set the mask to 1 (i.e. do not change these losses)
        if masking == "global":
            mask_neg = torch.where(labels == 0,
                                   (torch.rand_like(labels) < negatives).float(),
                                   torch.ones_like(labels))

        elif masking == "label":
            neg_proposals = (labels.sum(dim=1) == 0).unsqueeze(1).expand_as(labels)

            mask_neg = torch.where(neg_proposals,
                                     (torch.rand_like(neg_proposals.float()) < negatives).float(),
                                        torch.ones_like(neg_proposals.float()))

        elif masking == "span":
            neg_proposals = (labels.sum(dim=2) == 0).unsqueeze(2).expand_as(labels)

            mask_neg = torch.where(neg_proposals,
                                   (torch.rand_like(neg_proposals.float()) < negatives).float(),
                                   torch.ones_like(neg_proposals.float()))

        else:
            mask_neg = 1.

        # Apply the mask: for positive examples, some losses will be zeroed out based on the sampling
        all_losses = all_losses * mask_neg

        return all_losses

    @abstractmethod
    def loss(self, x):
        pass


class SpanModel(BaseModel):
    def __init__(self, config, encoder_from_pretrained, cache_dir: Optional[Union[str, Path]] = None):
        super(SpanModel, self).__init__(config, encoder_from_pretrained, cache_dir = cache_dir)
        self.span_rep_layer = SpanRepLayer(span_mode = config.span_mode, 
                                           hidden_size = config.hidden_size, 
                                           max_width = config.max_width,
                                           dropout = config.dropout)

        self.prompt_rep_layer = create_projection_layer(config.hidden_size, config.dropout)
        
        if config.relations_layer is not None:
            self.relations_rep_layer = RelationsRepLayer(in_dim=config.hidden_size, relation_mode = config.relations_layer)
        
            if config.triples_layer is not None:
                self.triples_score_layer = TriplesScoreLayer(config.triples_layer)
            else:
                self.pair_rep_layer = create_projection_layer(config.hidden_size*2, config.dropout, config.hidden_size)
    
    def select_span_target_embedding(
        self,
        span_rep: torch.FloatTensor,                     # (B, L, K, D)
        span_scores: torch.FloatTensor,                  # (B, L, K, C)
        span_mask: torch.LongTensor,                     # (B, L, K)
        span_labels: Optional[torch.FloatTensor] = None, # (B, L, K, C)               
        threshold = 0.5,
        top_k = None,
    ):
        B, L, K, D = span_rep.shape

        span_rep_flat = span_rep.view(B, L * K, D)      # (B, L*K, D)
        span_mask_flat = span_mask.view(B, L * K)        # (B, L*K)

        if span_labels is not None:
            # It doesn't support multi-label scenario
            span_prob_flat = span_labels.max(dim=-1).values.view(B, L * K)  # (B, L*K)
            keep = (span_prob_flat == 1).bool() #& span_mask_flat.bool()
        else:
            span_prob_flat = torch.sigmoid(span_scores)      # (B, L, K, C)
            span_prob_flat = span_prob_flat.max(dim=-1).values.view(B, L * K)  # (B, L*K)
            keep = (span_prob_flat > threshold) & span_mask_flat.bool()
        
        if top_k is not None and top_k > 0:
            sel_scores = span_prob_flat.masked_fill(~keep, -1.0)
            top_idx = sel_scores.topk(
                k=min(top_k, sel_scores.size(1)),
                dim=1
            ).indices                                       # (B, k)
            keep.flat_data = torch.zeros_like(keep)         # reset
            keep.scatter_(1, top_idx, True)

        rep_mask = keep.long()                              # (B, L*K)

        target_rep, target_mask = self.select_target_embedding(
            representations = span_rep_flat,                # (B, L*K, D)
            rep_mask = rep_mask                      # (B, L*K)
        )

        return target_rep, target_mask
    
    def forward(self,
                input_ids: Optional[torch.FloatTensor] = None,
                attention_mask: Optional[torch.LongTensor] = None,
                labels_embeddings: Optional[torch.FloatTensor] = None,
                labels_input_ids: Optional[torch.FloatTensor] = None,
                labels_attention_mask: Optional[torch.LongTensor] = None,
                words_embedding: Optional[torch.FloatTensor] = None,
                mask: Optional[torch.LongTensor] = None,
                prompts_embedding: Optional[torch.FloatTensor] = None,
                prompts_embedding_mask: Optional[torch.LongTensor] = None,
                words_mask: Optional[torch.LongTensor] = None,
                text_lengths: Optional[torch.Tensor] = None,
                span_idx: Optional[torch.LongTensor] = None,
                span_mask: Optional[torch.LongTensor] = None,
                labels: Optional[torch.FloatTensor] = None, # B,L*K, C
                adj_matrix: Optional[torch.FloatTensor] = None, #B, E, E
                rel_matrix: Optional[torch.FloatTensor] = None, # B, E, E, C  # Adjusted to match assumption for loss calculation
                threshold: Optional[float] = 0.5,
                **kwargs
                ):

        token_embeds, prompts_embedding, prompts_embedding_mask, words_embedding, mask = self.get_representations(input_ids,
                                                                                                    attention_mask,
                                                                                                    labels_embeddings,
                                                                                                    labels_input_ids,
                                                                                                    labels_attention_mask,
                                                                                                    text_lengths,
                                                                                                    words_mask)
        target_W = span_idx.size(1) // self.config.max_width
        words_embedding, mask = self._fit_length(words_embedding, mask, target_W)         
            
        span_idx = span_idx * span_mask.unsqueeze(-1)  

        span_rep = self.span_rep_layer(words_embedding, span_idx)

        target_C = prompts_embedding.size(1)
        if labels is not None:
            target_C = max(target_C, labels.size(-1))

        prompts_embedding, prompts_embedding_mask = self._fit_length(
            prompts_embedding, prompts_embedding_mask, target_C
        )

        prompts_embedding = self.prompt_rep_layer(prompts_embedding) 

        scores = torch.einsum("BLKD,BCD->BLKC", span_rep, prompts_embedding)
        
        pair_idx, pair_mask, pair_scores = None, None, None
        rel_prompts_embedding_mask = None
        full_rel_logits = None
        if hasattr(self, "relations_rep_layer"):
            target_span_rep, target_span_mask = self.select_span_target_embedding(
                span_rep, scores, span_mask, labels, threshold
            )
            pred_adj_matrix = self.relations_rep_layer(target_span_rep, target_span_mask)

            rel_prompts_embedding, rel_prompts_embedding_mask = extract_rel_features(
                self.config, token_embeds, input_ids, attention_mask, self.config.embed_rel_token
            )

            B, E, D = target_span_rep.shape
            _, C_rel, _ = rel_prompts_embedding.shape

            head_rep = target_span_rep.unsqueeze(2).expand(B, E, E, D)
            tail_rep = target_span_rep.unsqueeze(1).expand(B, E, E, D)

            if hasattr(self, "pair_rep_layer"):
                pair_rep = torch.cat((head_rep, tail_rep), dim=-1)
                pair_rep = self.pair_rep_layer(pair_rep)
                full_rel_logits = torch.einsum("BEED,BCD->BEEC", pair_rep, rel_prompts_embedding)

            elif hasattr(self, "triples_score_layer"):
                h = head_rep.unsqueeze(3).expand(B, E, E, C_rel, D)
                t = tail_rep.unsqueeze(3).expand(B, E, E, C_rel, D)
                r = rel_prompts_embedding.unsqueeze(1).unsqueeze(1).expand(B, E, E, C_rel, D)

                h_flat = h.reshape(B * E * E * C_rel, D)
                t_flat = t.reshape(B * E * E * C_rel, D)
                r_flat = r.reshape(B * E * E * C_rel, D)

                triple_scores_flat = self.triples_score_layer(h_flat, r_flat, t_flat)
                full_rel_logits = triple_scores_flat.view(B, E, E, C_rel)

            # Build pairs for output
            pair_idx, pair_mask, head_rep, tail_rep = build_entity_pairs(
                pred_adj_matrix, target_span_rep, threshold=0.5
            )

            # Gather rel_logits for selected pairs
            batch_idx = torch.arange(B, device=pair_idx.device).unsqueeze(1).expand(B, pair_idx.size(1))
            head_idx = pair_idx[..., 0].clamp_min(0)
            tail_idx = pair_idx[..., 1].clamp_min(0)
            pair_scores = full_rel_logits[batch_idx, head_idx, tail_idx]

        loss = None
        if labels is not None:
            loss = self.loss(scores, labels, prompts_embedding_mask, span_mask, **kwargs)

            if adj_matrix is not None and rel_matrix is not None and hasattr(self, "relations_rep_layer"):
                adj_mask = target_span_mask.float().unsqueeze(1) * target_span_mask.float().unsqueeze(2)
                adj_loss = self.adj_loss(pred_adj_matrix, adj_matrix, adj_mask, **kwargs)

                rel_mask = adj_mask
                rel_loss = self.rel_loss(full_rel_logits, rel_matrix, rel_mask, rel_prompts_embedding_mask, **kwargs)

                loss = loss + adj_loss + rel_loss

        output = GLiNERModelOutput(
            logits=scores,
            loss=loss,
            prompts_embedding=prompts_embedding,
            prompts_embedding_mask=prompts_embedding_mask,
            words_embedding=words_embedding,
            mask=mask,
            rel_idx=pair_idx,
            rel_logits=pair_scores,
            rel_mask=pair_mask
        )
        return output

    def loss(self, scores, labels, prompts_embedding_mask, mask_label,
             alpha: float = -1., gamma: float = 0.0, label_smoothing: float = 0.0,
             reduction: str = 'sum', negatives=1.0, masking="label", **kwargs):

        batch_size = scores.shape[0]
        num_classes = prompts_embedding_mask.shape[-1]

        # Reshape scores and labels to match the expected shape
        BS, SL, WD, CL = scores.shape

        scores = scores.view(BS, -1, CL)
        labels = labels.view(BS, -1, CL)

        all_losses = self._loss(scores, labels, alpha, gamma, label_smoothing, negatives, masking=masking)

        masked_loss = all_losses.view(batch_size, -1, num_classes) * prompts_embedding_mask.unsqueeze(1)
        all_losses = masked_loss.view(-1, num_classes)

        mask_label = mask_label.reshape(-1, 1)

        all_losses = all_losses * mask_label.float()

        if reduction == "mean":
            loss = all_losses.mean()
        elif reduction == 'sum':
            loss = all_losses.sum()
        else:
            warnings.warn(
                f"Invalid Value for config 'loss_reduction': '{reduction} \n Supported reduction modes:"
                f" 'none', 'mean', 'sum'. It will be used 'sum' instead.")
            loss = all_losses.sum()
        return loss
    
    def adj_loss(self, logits, labels, adj_mask,
                 alpha: float = -1., gamma: float = 0.0, label_smoothing: float = 0.0,
                 reduction: str = 'sum', negatives=1.0, masking="span", **kwargs):
        B, E, E = logits.shape

        # Add singleton dimension for binary classification
        logits = logits.unsqueeze(-1)  # (B, E, E, 1)
        labels = labels.unsqueeze(-1)  # (B, E, E, 1)

        all_losses = self._loss(logits.view(B, -1, 1), labels.view(B, -1, 1),
                                alpha, gamma, label_smoothing, negatives, masking)

        masked_loss = all_losses * adj_mask.unsqueeze(-1).view(B, -1, 1)

        if reduction == "mean":
            num_valid = adj_mask.sum()
            loss = masked_loss.sum() / num_valid if num_valid > 0 else 0.0
        elif reduction == 'sum':
            loss = masked_loss.sum()
        else:
            warnings.warn(
                f"Invalid Value for config 'loss_reduction': '{reduction} \n Supported reduction modes:"
                f" 'none', 'mean', 'sum'. It will be used 'sum' instead.")
            loss = masked_loss.sum()
        return loss

    def rel_loss(self, logits, labels, rel_mask, rel_prompts_embedding_mask,
                 alpha: float = -1., gamma: float = 0.0, label_smoothing: float = 0.0,
                 reduction: str = 'sum', negatives=1.0, masking="span", **kwargs):
        B, E, E, C = logits.shape

        all_losses = self._loss(logits.view(B, -1, C), labels.view(B, -1, C),
                                alpha, gamma, label_smoothing, negatives, masking)

        pair_mask = rel_mask.view(B, -1, 1).expand(B, E*E, C)
        class_mask = rel_prompts_embedding_mask.unsqueeze(1).expand(B, E*E, C)

        masked_loss = all_losses * pair_mask * class_mask

        if reduction == "mean":
            num_valid = (pair_mask * class_mask).sum()
            loss = masked_loss.sum() / num_valid if num_valid > 0 else 0.0
        elif reduction == 'sum':
            loss = masked_loss.sum()
        else:
            warnings.warn(
                f"Invalid Value for config 'loss_reduction': '{reduction} \n Supported reduction modes:"
                f" 'none', 'mean', 'sum'. It will be used 'sum' instead.")
            loss = masked_loss.sum()
        return loss


class TokenModel(BaseModel):
    def __init__(self, config, encoder_from_pretrained, cache_dir:Optional[Union[str, Path]] = None):
        super(TokenModel, self).__init__(config, encoder_from_pretrained, cache_dir=cache_dir)
        self.scorer = Scorer(config.hidden_size, config.dropout)

    def forward(self,
                input_ids: Optional[torch.FloatTensor] = None,
                attention_mask: Optional[torch.LongTensor] = None,
                labels_embeddings: Optional[torch.FloatTensor] = None,
                labels_input_ids: Optional[torch.FloatTensor] = None,
                labels_attention_mask: Optional[torch.LongTensor] = None,
                words_embedding: Optional[torch.FloatTensor] = None,
                mask: Optional[torch.LongTensor] = None,
                prompts_embedding: Optional[torch.FloatTensor] = None,
                prompts_embedding_mask: Optional[torch.LongTensor] = None,
                words_mask: Optional[torch.LongTensor] = None,
                text_lengths: Optional[torch.Tensor] = None,
                labels: Optional[torch.FloatTensor] = None,
                **kwargs
                ):

        _, prompts_embedding, prompts_embedding_mask, words_embedding, mask = self.get_representations(input_ids,
                                                                                                    attention_mask,
                                                                                                    labels_embeddings,
                                                                                                    labels_input_ids,
                                                                                                    labels_attention_mask,
                                                                                                    text_lengths,
                                                                                                    words_mask)
        if labels is not None:
            target_W = labels.shape[1]
            words_embedding, mask = self._fit_length(words_embedding, mask, target_W)

            target_C = prompts_embedding.size(1)
            if labels is not None:
                target_C = max(target_C, labels.size(-2))

            prompts_embedding, prompts_embedding_mask = self._fit_length(
                prompts_embedding, prompts_embedding_mask, target_C
            )

        scores = self.scorer(words_embedding, prompts_embedding)

        loss = None
        if labels is not None:
            loss = self.loss(scores, labels, prompts_embedding_mask, mask, **kwargs)

        output = GLiNERModelOutput(
            logits=scores,
            loss=loss,
            prompts_embedding=prompts_embedding,
            prompts_embedding_mask=prompts_embedding_mask,
            words_embedding=words_embedding,
            mask=mask,
        )
        return output

    def loss(self, scores, labels, prompts_embedding_mask, mask,
             alpha: float = -1., gamma: float = 0.0, label_smoothing: float = 0.0,
             reduction: str = 'sum', negatives=1, **kwargs):
        all_losses = self._loss(scores, labels, alpha, gamma, label_smoothing, negatives)

        all_losses = all_losses * (mask.unsqueeze(-1) * prompts_embedding_mask.unsqueeze(1)).unsqueeze(-1)

        if reduction == "mean":
            loss = all_losses.mean()
        elif reduction == 'sum':
            loss = all_losses.sum()
        else:
            warnings.warn(
                f"Invalid Value for config 'loss_reduction': '{reduction} \n Supported reduction modes:"
                f" 'none', 'mean', 'sum'. It will be used 'sum' instead.")
            loss = all_losses.sum()
        return loss
