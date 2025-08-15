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
from .decoder import Decoder
from .layers import LstmSeq2SeqEncoder, CrossFuser, SelfAttentionBlock, create_projection_layer
from .scorers import Scorer
from .loss_functions import focal_loss_with_logits, cross_entropy_loss
from .span_rep import SpanRepLayer

@dataclass
class GLiNERModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    prompts_embedding: Optional[torch.FloatTensor] = None
    prompts_embedding_mask: Optional[torch.LongTensor] = None
    decoder_loss: Optional[torch.FloatTensor] = None
    decoder_embedding: Optional[torch.FloatTensor] = None
    decoder_embedding_mask: Optional[torch.LongTensor] = None
    decoder_span_idx: Optional[torch.LongTensor] = None
    words_embedding: Optional[torch.FloatTensor] = None
    mask: Optional[torch.LongTensor] = None

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

def extract_prompt_features_and_word_embeddings(config, token_embeds, input_ids, attention_mask,
                                                text_lengths, words_mask, embed_ent_token=True, **kwargs):
    # getting prompt embeddings
    batch_size, sequence_length, embed_dim = token_embeds.shape

    class_token_mask = input_ids == config.class_token_index
    num_class_tokens = torch.sum(class_token_mask, dim=-1, keepdim=True)

    max_embed_dim = num_class_tokens.max()
    max_text_length = text_lengths.max()
    aranged_class_idx = torch.arange(max_embed_dim,
                                     dtype=attention_mask.dtype,
                                     device=token_embeds.device).expand(batch_size, -1)

    batch_indices, target_class_idx = torch.where(aranged_class_idx < num_class_tokens)
    _, class_indices = torch.where(class_token_mask)
    if not embed_ent_token:
        class_indices += 1

    prompts_embedding = torch.zeros(
        batch_size, max_embed_dim, embed_dim, dtype=token_embeds.dtype, device=token_embeds.device
    )

    prompts_embedding_mask = (aranged_class_idx < num_class_tokens).to(attention_mask.dtype)

    prompts_embedding[batch_indices, target_class_idx] = token_embeds[batch_indices, class_indices]

    # getting words embedding
    words_embedding, mask = extract_word_embeddings(token_embeds, words_mask, attention_mask,
                                                    batch_size, max_text_length, embed_dim, text_lengths)

    return prompts_embedding, prompts_embedding_mask, words_embedding, mask


class BaseModel(ABC, nn.Module):
    data_processor = None
    def __init__(self, config, from_pretrained = False, cache_dir: Optional[Union[str, Path]] = None):
        super(BaseModel, self).__init__()
        self.config = config

        if not config.labels_encoder:
            self.token_rep_layer = Encoder(config, from_pretrained, cache_dir = cache_dir)
        else:
            self.token_rep_layer = BiEncoder(config, from_pretrained, cache_dir = cache_dir)
        
        if config.labels_decoder:
            self.decoder = Decoder(config, from_pretrained, cache_dir = cache_dir)
            if self.config.hidden_size != self.decoder.decoder_hidden_size:
                self._enc2dec_proj = create_projection_layer(
                    self.config.hidden_size,
                    self.config.dropout,
                    self.decoder.decoder_hidden_size,
                )

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

        return prompts_embedding, prompts_embedding_mask, words_embedding, mask

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

        return labels_embeds, labels_mask, words_embedding, mask

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
            prompts_embedding, prompts_embedding_mask, words_embedding, mask = self.get_bi_representations(
                input_ids, attention_mask, labels_embeddings, labels_input_ids, labels_attention_mask,
                text_lengths, words_mask, **kwargs
            )
        else:
            prompts_embedding, prompts_embedding_mask, words_embedding, mask = self.get_uni_representations(
                input_ids, attention_mask, text_lengths, words_mask, **kwargs
            )
        return prompts_embedding, prompts_embedding_mask, words_embedding, mask

    def select_decoder_embedding(
        self,
        representations: torch.FloatTensor,   # (B, N, D) – flattened span reps
        rep_mask:       torch.LongTensor,     # (B, N)    – 0/1 or False/True
    ):
        """
        Keeps only representations whose mask == 1 and (optionally) projects them
        to the decoder hidden size.

        Returns
        -------
        target_rep   : FloatTensor (B, M, D)   – kept reps, padded
        target_mask  : LongTensor  (B, M)      – 1 where real, 0 where pad
        sel_idx      : LongTensor  (B, M)      – original *flattened* span column
                                                (‑1 for pad)
        batch_idx_pad: LongTensor  (B, M)      – batch‑row index for each entry
                                                (‑1 for pad)          ← NEW
        """
        B, N, D = representations.shape
        lengths = rep_mask.sum(dim=-1)              # (B,)
        max_len = lengths.max().item()

        target_rep = representations.new_zeros(B, max_len, D)   # (B, M, D)
        target_mask = rep_mask.new_zeros(B, max_len)             # (B, M)
        sel_idx = rep_mask.new_full((B, max_len), -1)        # (B, M)

        keep = rep_mask.bool()
        if keep.any():
            new_col_idx  = (rep_mask.cumsum(dim=1) - 1)[keep]  # (K,)
            batch_idx, old_col_idx = torch.where(keep)                  # (K,)

            target_rep [batch_idx, new_col_idx] = representations[batch_idx, old_col_idx]
            target_mask[batch_idx, new_col_idx] = 1
            sel_idx[batch_idx, new_col_idx] = old_col_idx

        return target_rep, target_mask, sel_idx
    
    def get_raw_decoder_inputs(self, representations, rep_mask):
        B, S, T, D = representations.shape
        BN = B * S                                     # flattened span count
        valid_spans = rep_mask.any(-1)
        keep_mask = valid_spans.view(-1)             # (BN,)  → bool
        if not keep_mask.any():                        # corner case: nothing to keep
            empty = representations.new_empty(0, 0, D)
            return empty, representations.new_empty(0, 0, dtype=rep_mask.dtype)

        keep_idx = keep_mask.nonzero(as_tuple=False).squeeze(1)   # (M,)

        span_tokens = representations.view(BN, T, D)[keep_idx]  # (M, T, D)
        span_tokens_mask = rep_mask.view(BN, T)[keep_idx]            # (M, T)
        return span_tokens, span_tokens_mask
    
    
    def decode_labels(self, decoder_embedding: torch.FloatTensor = None, #B, N, T, D
                            decoder_embedding_mask: torch.LongTensor = None, #B, N, T
                            decoder_labels_ids: torch.FloatTensor = None, #(B*N, S)
                            decoder_labels_mask: torch.LongTensor = None,
                            decoder_labels: torch.FloatTensor = None,
                            **kwargs):
        span_tokens, span_tokens_mask = self.get_raw_decoder_inputs(decoder_embedding, decoder_embedding_mask)

        label_embeds = self.decoder.ids_to_embeds(decoder_labels_ids)        # (M, L, D)

        decoder_inputs = torch.cat([span_tokens, label_embeds[:, :-1, :]], dim=1)

        attn_inputs    = torch.cat(
            [span_tokens_mask.to(decoder_labels_mask.dtype),
            decoder_labels_mask[:, :-1]], dim=1
        )

        decoder_outputs = self.decoder(inputs_embeds=decoder_inputs,
                                    attention_mask=attn_inputs)           # (M, S+L‑1, V)

        blank_for_spans = torch.full(                       
            (decoder_labels.size(0), span_tokens.size(1)),
            -100, dtype=decoder_labels.dtype, device=decoder_labels.device
        )

        targets = torch.cat([blank_for_spans, decoder_labels], dim=1)        # (M, S+L)

        loss = cross_entropy_loss(
            decoder_outputs,                      # logits
            targets[:, 1:]                        # same length as logits
        )

        return (loss, decoder_outputs)

    def generate_labels(self, decoder_embedding: torch.FloatTensor = None, #B, N, D
                            decoder_embedding_mask: torch.LongTensor = None,
                            max_new_tokens: int = 32,
                            eos_token_id: Optional[int] = None,
                            pad_token_id: Optional[int] = None,
                            temperature: float = 1.0,
                            do_sample: bool = False,
                            num_return_sequences=1,
                            labels_trie = None,
                            **kwargs):

        span_tokens, _ = self.get_raw_decoder_inputs(decoder_embedding, decoder_embedding_mask)
        results = self.decoder.generate_from_embeds(span_tokens,
                                                    attention_mask = None,
                                                    max_new_tokens = max_new_tokens,
                                                    eos_token_id = eos_token_id,
                                                    pad_token_id = pad_token_id,
                                                    temperature = temperature,
                                                    do_sample = do_sample,
                                                    num_return_sequences=num_return_sequences,
                                                    labels_trie = labels_trie,
                                                    **kwargs)
        return results
    
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
    
    @abstractmethod
    def forward(self, x):
        pass

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor,
              alpha: float = -1., gamma: float = 0.0, prob_margin: float = 0.0, 
                    label_smoothing: float = 0.0, negatives=1., masking="label"):

        # Compute the loss per element using the focal loss function
        all_losses = focal_loss_with_logits(logits, labels,
                                            alpha=alpha,
                                            gamma=gamma,
                                            prob_margin=prob_margin,
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

        # if self.config.labels_decoder is not None:
        #     num_heads = self.decoder.decoder_hidden_size//(self.decoder.decoder_hidden_size//8)
        #     self.span_attn_layer = SelfAttentionBlock(self.decoder.decoder_hidden_size, num_heads = num_heads)

    def select_span_decoder_embedding(
        self,
        prompts_embedding, # (B, C, D)
        prompts_embedding_mask, # (B, C)
        span_rep, # (B, L, K, D)
        span_scores, # (B, L, K, C)
        span_mask, # (B, L, K)
        decoder_text_embeds=None, # (B, T, D)
        decoder_words_mask=None, # (B, T)
        span_labels = None, # (B, L, K, C)
        threshold = 0.5,
        top_k = None,
        decoder_input_ids = None, # for debugging purposes
        decoder_labels_ids = None
    ):
        if self.config.decoder_mode == "prompt":
            return self.select_decoder_embedding(
                prompts_embedding, prompts_embedding_mask
            )[:3] # ignore the extra batch‑idx from the helper
        B, L, K, D = span_rep.shape
        flat_rep = span_rep.view(B, L * K, D) # (B, N, D)
        flat_mask = span_mask.view(B, L * K) # (B, N)
        
        if span_labels is not None:
            flat_prob = span_labels.max(-1).values.view(B, L * K)
            keep = (flat_prob == 1) & flat_mask.bool()
        else:
            flat_prob = torch.sigmoid(span_scores).max(-1).values.view(B, L * K)
            keep = (flat_prob > threshold) & flat_mask.bool()
        
        if top_k:
            sel_scores = flat_prob.masked_fill(~keep, -1.0)
            top_idx = sel_scores.topk(
                k=min(top_k, sel_scores.size(1)), dim=1
            ).indices
            keep.zero_()
            keep.scatter_(1, top_idx, True)
        
        span_rep_kept, span_msk, span_sel_idx = \
            self.select_decoder_embedding(flat_rep, keep.long()) # (B, S, …)
        
        if hasattr(self, "_enc2dec_proj"):
            span_rep_kept = self._enc2dec_proj(span_rep_kept)
        span_rep_kept = span_rep_kept.unsqueeze(2)
        span_msk = span_msk.unsqueeze(-1)
       
        if decoder_text_embeds is None or decoder_words_mask is None:
            return span_rep_kept, span_msk.unsqueeze(-1), span_sel_idx
                
        if span_rep_kept.numel() == 0:
            return None, None, None
    
        decoder_text_embeds = decoder_text_embeds.to(dtype=span_rep_kept.dtype)
       
        S = span_rep_kept.shape[1]
        dec_D = span_rep_kept.shape[-1]
        span_start = span_sel_idx//self.config.max_width+1
        span_end = span_sel_idx%self.config.max_width+span_start

        token_in_span = (
            (decoder_words_mask.unsqueeze(1) >= span_start.unsqueeze(-1)) &
            (decoder_words_mask.unsqueeze(1) <= span_end.unsqueeze(-1))
        )
       
        tokens_per_span = token_in_span.sum(-1) # (B, S)
        max_tokens = int(tokens_per_span.max()) # scalar python int
        
        span_rep_new = span_rep_kept.new_zeros(B, S, max_tokens + 1, dec_D) # (B,S,T+1,D)
        span_rep_mask = torch.zeros(B, S, max_tokens + 1, dtype=torch.bool,
        device=decoder_text_embeds.device)
        
        left_offset = (max_tokens + 1 - tokens_per_span).clamp(min=0) # (B,S)
        pos_in_span = (token_in_span.cumsum(-1) - 1).masked_fill(~token_in_span, 0)
        pos_in_span = pos_in_span + left_offset.unsqueeze(-1) # shift → right
       
        b_idx, s_idx, tok_idx = torch.where(token_in_span) # (N,)
        span_rep_new[b_idx, s_idx, pos_in_span[b_idx, s_idx, tok_idx]] = \
        decoder_text_embeds[b_idx, tok_idx]
        span_rep_mask[b_idx, s_idx, pos_in_span[b_idx, s_idx, tok_idx]] = True
        kept_pos = (left_offset - 1).clamp(min=0) # (B,S)

        b_flat = torch.arange(B, device=decoder_text_embeds.device).view(-1, 1).expand(B, S).reshape(-1)
        s_flat = torch.arange(S, device=decoder_text_embeds.device).view(1, -1).expand(B, S).reshape(-1)
        t_flat = kept_pos.reshape(-1)

        span_rep_new[b_flat, s_flat, t_flat] = span_rep_kept.reshape(-1, dec_D)
        span_rep_mask[b_flat, s_flat, t_flat] = True
        span_rep_mask = span_rep_mask & span_msk.bool()
        return span_rep_new, span_rep_mask, span_sel_idx


    def forward(self,
                input_ids: Optional[torch.FloatTensor] = None,
                attention_mask: Optional[torch.LongTensor] = None,
                decoder_input_ids: Optional[torch.FloatTensor] = None,
                decoder_attention_mask: Optional[torch.LongTensor] = None,
                decoder_labels_ids: Optional[torch.FloatTensor] = None,
                decoder_labels_mask: Optional[torch.LongTensor] = None,
                decoder_words_mask: Optional[torch.LongTensor] = None,
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
                labels: Optional[torch.FloatTensor] = None,  # B,L*K, C
                decoder_labels:  Optional[torch.FloatTensor] = None,
                threshold: Optional[float] = 0.5,
                **kwargs
                ):

        prompts_embedding, prompts_embedding_mask, words_embedding, mask = self.get_representations(input_ids,
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
        
        decoder_embedding = decoder_mask = decoder_loss = decoder_span_idx = None
        if hasattr(self, "decoder"):
            if self.config.decoder_mode == 'span':
                decoder_text_embeds = self.decoder.ids_to_embeds(decoder_input_ids)
            else:
                decoder_text_embeds = None
            decoder_embedding, decoder_mask, decoder_span_idx = self.select_span_decoder_embedding(
                        prompts_embedding, prompts_embedding_mask, span_rep, scores, span_mask,
                                    decoder_text_embeds = decoder_text_embeds,
                                    decoder_words_mask = decoder_words_mask,
                                    span_labels=labels, threshold=threshold,
                                    decoder_input_ids=decoder_input_ids,
                                    decoder_labels_ids=decoder_labels_ids
            ) #(B, S, T, D)
            # decoder_embedding = self.span_attn_layer(decoder_embedding, decoder_mask)
            if decoder_labels is not None:
                decoder_loss, decoder_outputs = self.decode_labels(
                    decoder_embedding, decoder_mask, decoder_labels_ids, 
                                        decoder_labels_mask, decoder_labels
                )

        loss = None
        if labels is not None:
            loss = self.loss(scores, labels, prompts_embedding_mask, span_mask, decoder_loss=decoder_loss, **kwargs)

        output = GLiNERModelOutput(
            logits=scores,
            loss=loss,
            decoder_loss=decoder_loss,
            prompts_embedding=prompts_embedding,
            prompts_embedding_mask=prompts_embedding_mask,
            decoder_embedding=decoder_embedding,
            decoder_embedding_mask=decoder_mask,
            decoder_span_idx=decoder_span_idx,
            words_embedding=words_embedding,
            mask=mask,
        )
        return output

    def loss(self, scores, labels, prompts_embedding_mask, mask_label,
             alpha: float = -1., gamma: float = 0.0, label_smoothing: float = 0.0,
             reduction: str = 'sum', negatives=1.0, masking="label", decoder_loss = None, **kwargs):

        batch_size = scores.shape[0]
        num_classes = prompts_embedding_mask.shape[-1]

        # Reshape scores and labels to match the expected shape
        BS, SL, WD, CL = scores.shape

        scores = scores.view(BS, -1, CL)
        labels = labels.view(BS, -1, CL)

        all_losses = self._loss(scores, labels, alpha, gamma, label_smoothing, negatives, masking=masking)

        masked_loss = all_losses.view(batch_size, -1, num_classes) * prompts_embedding_mask.unsqueeze(1)
        all_losses = masked_loss.view(-1, num_classes)

        mask_label = mask_label.view(-1, 1)

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

        if decoder_loss is not None:
            loss = decoder_loss*0.75+loss*0.25
        
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

        prompts_embedding, prompts_embedding_mask, words_embedding, mask = self.get_representations(input_ids,
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
             alpha: float = -1., gamma: float = 0.0, prob_margin: float=  0.0,
             label_smoothing: float = 0.0, reduction: str = 'sum', negatives=1, **kwargs):
        all_losses = self._loss(scores, labels, alpha, gamma, prob_margin, label_smoothing, negatives)

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
