import warnings

import torch
import torch.nn.functional as F
from gliner.modules.base import InstructBase
from gliner.modules.evaluator import greedy_search
from gliner.modules.layers import LstmSeq2SeqEncoder, create_projection_layer, Scorer, TokenPromptProcessor
from gliner.modules.loss_functions import focal_loss_with_logits
from gliner.modules.span_rep import SpanRepLayer
from gliner.modules.token_rep import TokenRepLayer
from transformers.trainer import (
    get_parameter_names,
    ALL_LAYERNORM_LAYERS,
)


def get_params(opt_model, lr, weight_decay):
    decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    grouped_parameters = [
        {
            "params": [
                p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": weight_decay,
            "lr": lr,
        },
        {
            "params": [
                p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
            "lr": lr,
        }
    ]
    return grouped_parameters


class GLiNER(InstructBase):
    def __new__(cls, config):
        if config.span_mode == "token_level":
            return TokenGLiNER(config)
        else:
            return SpanGLiNER(config)


class SpanGLiNER(InstructBase):
    def __init__(self, config):
        super().__init__(config)

        # [ENT] token
        self.entity_token = "<<ENT>>"
        self.sep_token = "<<SEP>>"

        # usually a pretrained bidirectional transformer, returns first subtoken representation
        self.token_rep_layer = TokenRepLayer(model_name=config.model_name, fine_tune=config.fine_tune,
                                             subtoken_pooling=config.subtoken_pooling, hidden_size=config.hidden_size,
                                             add_tokens=[self.entity_token, self.sep_token])

        # token prompt processor
        self.token_prompt_processor = TokenPromptProcessor(self.entity_token, self.sep_token)

        # hierarchical representation of tokens (zaratiana et al, 2022)
        # https://arxiv.org/pdf/2203.14710.pdf
        self.rnn = LstmSeq2SeqEncoder(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size // 2,
            num_layers=1,
            bidirectional=True,
        )

        # span representation
        # we have a paper to study span representation for ner
        # zaratiana et al, 2022: https://aclanthology.org/2022.umios-1.1/
        self.span_rep_layer = SpanRepLayer(
            span_mode=config.span_mode,
            hidden_size=config.hidden_size,
            max_width=config.max_width,
            dropout=config.dropout,
        )

        # prompt representation (FFN)
        self.prompt_rep_layer = create_projection_layer(config.hidden_size, config.dropout)

    def get_optimizer(self, lr_encoder, lr_others, weight_decay_encoder, weight_decay_others,
                      freeze_token_rep=False, **optimizer_kwargs):
        """
        Parameters:
        - lr_encoder: Learning rate for the encoder layer.
        - lr_others: Learning rate for all other layers.
        - freeze_token_rep: whether the token representation layer should be frozen.
        """
        param_groups = []
        param_groups += get_params(self.rnn, lr_others, weight_decay_others)
        param_groups += get_params(self.span_rep_layer, lr_others, weight_decay_others)
        param_groups += get_params(self.prompt_rep_layer, lr_others, weight_decay_others)

        if not freeze_token_rep:
            # If token_rep_layer should not be frozen, add it to the optimizer with its learning rate and weight decay
            param_groups += get_params(self.token_rep_layer, lr_encoder, weight_decay_encoder)
        else:
            # If token_rep_layer should be frozen, explicitly set requires_grad to False for its parameters
            for param in self.token_rep_layer.parameters():
                param.requires_grad = False

        optimizer = torch.optim.AdamW(param_groups, **optimizer_kwargs)

        return optimizer

    def compute_score_train(self, x):

        # get device
        device = next(self.token_rep_layer.parameters()).device

        span_idx = (x["span_idx"] * x["span_mask"].unsqueeze(-1)).to(device)

        # compute token representation
        word_rep, mask, entity_type_rep, entity_type_mask = self.token_prompt_processor.process(
            x, self.token_rep_layer, "train"
        )

        # compute span representation
        word_rep = self.rnn(word_rep, mask)
        span_rep = self.span_rep_layer(word_rep, span_idx)

        # compute final entity type representation (FFN)
        entity_type_rep = self.prompt_rep_layer(entity_type_rep)  # (batch_size, len_types, hidden_size)
        num_classes = entity_type_rep.shape[1]  # number of entity types

        # similarity score
        scores = torch.einsum("BLKD,BCD->BLKC", span_rep, entity_type_rep)

        return scores, num_classes, entity_type_mask

    def forward(self, x):
        # compute span representation
        scores, num_classes, entity_type_mask = self.compute_score_train(x)
        batch_size = scores.shape[0]

        # loss for filtering classifier
        logits_label = scores.view(-1, num_classes)
        labels = x["span_label"].view(-1)  # (batch_size * num_spans)
        mask_label = labels != -1  # (batch_size * num_spans)
        labels.masked_fill_(~mask_label, 0)  # Set the labels of padding tokens to 0

        # one-hot encoding
        labels_one_hot = F.one_hot(labels, num_classes + 1).to(dtype=scores.dtype)
        labels_one_hot = labels_one_hot[:, 1:]  # Remove the first column
        # Shape of labels_one_hot: (batch_size * num_spans, num_classes)

        # compute loss (without reduction)
        all_losses = focal_loss_with_logits(logits_label, labels_one_hot,
                                            alpha=self.config.loss_alpha,
                                            gamma=self.config.loss_gamma)
        # mask loss using entity_type_mask (B, C)
        masked_loss = all_losses.view(batch_size, -1, num_classes) * entity_type_mask.unsqueeze(1)
        all_losses = masked_loss.view(-1, num_classes)
        # expand mask_label to all_losses
        mask_label = mask_label.unsqueeze(-1).expand_as(all_losses)
        # apply mask
        all_losses = all_losses * mask_label.float()
        if self.config.loss_reduction == "mean":
            loss = all_losses.mean()
        elif self.config.loss_reduction == 'sum':
            loss = all_losses.sum()
        else:
            warnings.warn(
                f"Invalid Value for config 'loss_reduction': '{self.config.loss_reduction} \n Supported reduction modes: 'none', 'mean', 'sum'. It will be used 'sum' instead.")
            loss = all_losses.sum()
        return loss

    def compute_score_eval(self, x, device):
        # check if classes_to_id is dict
        assert isinstance(x["classes_to_id"], dict), "classes_to_id must be a dict"

        span_idx = (x["span_idx"] * x["span_mask"].unsqueeze(-1)).to(device)

        word_rep, mask, entity_type_rep = self.token_prompt_processor.process(
            x, self.token_rep_layer, "eval"
        )

        entity_type_rep = self.prompt_rep_layer(entity_type_rep)  # (batch_size, len_types, hidden_size)

        word_rep = self.rnn(word_rep, mask)

        span_rep = self.span_rep_layer(word_rep, span_idx)

        local_scores = torch.einsum("BLKD,BCD->BLKC", span_rep, entity_type_rep)

        return local_scores

    @torch.no_grad()
    def predict(self, x, flat_ner=False, threshold=0.5, multi_label=False):
        self.eval()
        local_scores = self.compute_score_eval(x, device=next(self.parameters()).device)
        probs = torch.sigmoid(local_scores)

        spans = []
        for i, _ in enumerate(x["tokens"]):
            probs_i = probs[i]

            wh_i = [i.tolist() for i in torch.where(probs_i > threshold)]
            span_i = []

            for s, k, c in zip(*wh_i):
                if s + k < len(x["tokens"][i]):
                    span_i.append((s, s + k, x["id_to_classes"][c + 1], probs_i[s, k, c].item()))

            span_i = greedy_search(span_i, flat_ner, multi_label=multi_label)
            spans.append(span_i)
        return spans


class TokenGLiNER(InstructBase):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        # [ENT] token
        self.entity_token = "<<ENT>>"
        self.sep_token = "<<SEP>>"

        # usually a pretrained bidirectional transformer, returns first subtoken representation
        self.token_rep_layer = TokenRepLayer(model_name=config.model_name, fine_tune=config.fine_tune,
                                             subtoken_pooling=config.subtoken_pooling, hidden_size=config.hidden_size,
                                             add_tokens=[self.entity_token, self.sep_token])

        # hierarchical representation of tokens
        self.rnn = LstmSeq2SeqEncoder(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size // 2,
            num_layers=1,
            bidirectional=True,
        )

        # token prompt processor
        self.token_prompt_processor = TokenPromptProcessor(self.entity_token, self.sep_token)

        # span representation (FFN)
        self.scorer = Scorer(config.hidden_size, config.dropout)

    def get_optimizer(self, lr_encoder, lr_others, weight_decay_encoder, weight_decay_others,
                      freeze_token_rep=False, **optimizer_kwargs):
        """
        Parameters:
        - lr_encoder: Learning rate for the encoder layer.
        - lr_others: Learning rate for all other layers.
        - freeze_token_rep: whether the token representation layer should be frozen.
        """
        param_groups = []
        param_groups += get_params(self.rnn, lr_others, weight_decay_others)
        param_groups += get_params(self.scorer, lr_others, weight_decay_others)

        if not freeze_token_rep:
            # If token_rep_layer should not be frozen, add it to the optimizer with its learning rate and weight decay
            param_groups += get_params(self.token_rep_layer, lr_encoder, weight_decay_encoder)
        else:
            # If token_rep_layer should be frozen, explicitly set requires_grad to False for its parameters
            for param in self.token_rep_layer.parameters():
                param.requires_grad = False

        optimizer = torch.optim.AdamW(param_groups, **optimizer_kwargs)

        return optimizer

    def compute_score_train(self, x):

        device = next(self.parameters()).device

        # compute token representation
        word_rep, mask, entity_type_rep, entity_type_mask = self.token_prompt_processor.process(
            x, self.token_rep_layer, "train"
        )

        num_classes = entity_type_rep.shape[1]  # number of entity types

        # compute span representation
        word_rep = self.rnn(word_rep, mask)

        batch_size, seq_len, hidden_size = word_rep.shape

        # create a tensor with shape (batch_size, seq_len) with entries the id of the entity type of the span
        word_labels = torch.zeros(
            3, batch_size, seq_len, num_classes, dtype=torch.float
        ).to(device)

        # get batch_nums and span_pos
        for i, element in enumerate(x["entities_id"]):
            for ent in element:
                st, ed, sp_label = ent
                sp_label = sp_label - 1

                word_labels[0, i, st, sp_label] = 1  # start
                word_labels[1, i, ed, sp_label] = 1  # end
                word_labels[2, i, st:ed + 1, sp_label] = 1  # inside

        # compute scores for start, end and inside
        all_scores = self.scorer(word_rep, entity_type_rep)  # (3, batch_size, seq_len, num_classes)

        all_losses = focal_loss_with_logits(all_scores, word_labels,
                                            alpha=self.config.loss_alpha,
                                            gamma=self.config.loss_gamma)

        all_losses = all_losses * entity_type_mask.unsqueeze(1) * mask.unsqueeze(-1)

        return all_losses

    def forward(self, x):
        all_losses = self.compute_score_train(x)
        if self.config.loss_reduction == "mean":
            loss = all_losses.mean()
        elif self.config.loss_reduction == 'sum':
            loss = all_losses.sum()
        else:
            warnings.warn(
                f"Invalid Value for config 'loss_reduction': '{self.config.loss_reduction} \n Supported reduction modes:"
                f" 'none', 'mean', 'sum'. It will be used 'sum' instead.")
            loss = all_losses.sum()
        return loss

    def compute_score_eval(self, x):
        # check if classes_to_id is dict
        assert isinstance(x['classes_to_id'], dict), "classes_to_id must be a dict"

        # compute token representation
        word_rep, mask, entity_type_rep = self.token_prompt_processor.process(
            x, self.token_rep_layer, "eval"
        )

        # compute word representation using LSTM
        word_rep = self.rnn(word_rep, mask)

        # compute scores for start, end and inside
        scores_start, scores_end, scores_inside = self.scorer(word_rep, entity_type_rep)

        return scores_start, scores_end, scores_inside

    @torch.no_grad()
    def predict(self, x, flat_ner=False, threshold=0.5, multi_label=False):
        scores_start, scores_end, scores_inside = self.compute_score_eval(x)
        # shape: (batch_size, seq_len, num_classes)

        spans = []
        for i, _ in enumerate(x["tokens"]):
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
                            (st, ed, x["id_to_classes"][cls_st + 1], ins.mean().item())
                        )
            span_i = greedy_search(span_i, flat_ner, multi_label=multi_label)
            spans.append(span_i)
        return spans
